"""06 - Stamp the bootstrap manifest and stage the servable site.

Assembles everything into dist/site:

  index.html  methods.html  download.html  attribution.html
  app.js  styles.css  vendor/*
  data/manifest.json          the bootstrap manifest the client reads first
  data/tiles/*.pmtiles        hard-linked, not copied
  data/shards/** data/bench/**   written by 04
  downloads/**                written by 05

The manifest binds the source parquet sha256 and the commit the SOURCE DATA was
built at (read from the candidate run manifest and checked against the file
being frozen), not the commit this step happens to run at. Static asset URLs
carry a content hash, so nothing needs a timestamp cache-buster.

Run:  uv run python frontend/build/06_stage_site.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crschema import (  # noqa: E402
    BG_SRC,
    COMPOSITES,
    CRIME_HELP,
    CRIME_HELP_HALF_LIFE,
    CRIME_ORDER,
    DENOMINATOR_EXPOSURE,
    DENOMINATOR_EXPOSURE_COMPOSITE,
    DENOMINATOR_RESIDENT,
    DIRECT_SHARE_THRESHOLD,
    DIST,
    FIPS_TO_USPS,
    LEGEND,
    MEASURE_LABEL,
    NO_ESTIMATE_COLOR,
    NO_ESTIMATE_LABEL,
    MEASURE_ORDER,
    LEVEL_TOTAL_ESTIMATED,
    LEVEL_TOTAL_MIXED,
    LEVEL_TOTAL_REPORTED,
    MEASURE_SUBCOPY,
    MEASURES,
    OFFENSE_LABEL,
    OFFENSES,
    REPO,
    SHORT,
    SOURCE_PHRASE_DIRECT,
    SOURCE_PHRASE_MODELED,
    SOURCE_PHRASE_MOSTLY_DIRECT,
    SOURCE_PHRASE_MOSTLY_MODELED,
    SPECIAL_USE_LABEL,
    COUNT_NOUN,
    SUPPRESSED_COUNT_COPY,
    STATE_NAME,
    STATUS_BASE,
    STATUS_CARD,
    STATUS_LEGEND,
    MEASURE_STATUS,
    TRACT_SRC,
    WORK,
    YEAR,
)

PUBLIC = REPO / "frontend" / "public"
SITE = DIST / "site"
TILE_DIR = SITE / "data" / "tiles"

# Where the two heavy trees are actually served from. GitHub Pages cannot host the
# PMTiles archives or the download package, so the manifest's tile URLs and every
# download link are absolute under a versioned prefix on the object host, while the
# shards, the bench files and the app itself stay relative and ship with the Pages
# site. One knob: set CRIMERISK_ASSET_BASE_URL to stage against a different host or
# edition, or to a local origin when serving the site for verification.
EDITION = f"{YEAR}.3"
# EDITION_LABEL is the displayed label; it can move ahead of EDITION for a copy-only
# patch against the same data. Bump EDITION when the data changes.
EDITION_LABEL = f"{YEAR}.3"
DEFAULT_ASSET_BASE_URL = f"https://tiles.qqlab.io/{EDITION}"
ASSET_BASE_URL = os.environ.get("CRIMERISK_ASSET_BASE_URL", DEFAULT_ASSET_BASE_URL).rstrip("/")
DOWNLOAD_BASE_URL = f"{ASSET_BASE_URL}/downloads"

# Coverage box for search: the 48 contiguous states and DC. Alaska, Hawaii and the
# territories are excluded from the edition, so a search that lands outside this box
# is told so rather than shown "no match".
COVERAGE_BOUNDS = [-124.85, 24.35, -66.85, 49.45]

BASEMAP = {
    "style": "https://tiles.openfreemap.org/styles/positron",
    "glyphs": "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    "sprite": "https://tiles.openfreemap.org/sprites/ofm_f384/ofm",
}
# The basemap source carries its own OpenFreeMap / OpenStreetMap notice, so
# this string covers only what CrimeRisk adds.
ATTRIBUTION = (
    'CrimeRisk 2025 &middot; FBI UCR/NIBRS, U.S. Census Bureau &middot; '
    '<a href="attribution.html">sources</a>'
)

# The versioned evidence tarball (scripts/diagnostics/evidence_bundle.py) is uploaded
# beside the tiles under <edition>/evidence/. The methods and download pages link it
# when the bundle is present in dist/evidence at staging time.
EVIDENCE_DIR = DIST / "evidence"
EVIDENCE_NAME = f"crimerisk_{EDITION}_evidence.tar.gz"
EVIDENCE_URL = f"{ASSET_BASE_URL}/evidence/{EVIDENCE_NAME}"
LEDGER_NAME = f"crimerisk_{YEAR}_exception_ledger.csv"

STATIC = ["app.js", "styles.css"]
PAGES = ["index.html", "methods.html", "download.html"]
TILE_LAYERS = {
    "county": ("counties", "counties", 3, 7),
    "tract": ("tracts", "tracts", 5, 12),
    "bg": ("blockgroups", "blockgroups", 8, 12),
}


def sha256(path: Path, n: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    d = h.hexdigest()
    return d[:n] if n else d


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def source_provenance() -> dict:
    """Commit the source data was built at, proved by the candidate manifest."""
    out = {
        "basis": "candidate run manifest run.git.head_sha",
        "commit": None,
        "commit_short": None,
        "manifest_binds_source_parquet": False,
        "warnings": [],
    }
    mp = BG_SRC.parent / "manifest.json"
    if not mp.exists():
        out["warnings"].append("candidate manifest missing; source commit cannot be proved")
        return out
    man = json.loads(mp.read_text())
    git = man.get("run", {}).get("git", {})
    out["commit"] = git.get("head_sha")
    out["commit_short"] = git.get("short_sha")
    stats = man.get("output_file_stats", {})
    ok = True
    for label, path in (("block_group_ags_core", BG_SRC), ("tract_ags_core", TRACT_SRC)):
        rec = stats.get(label)
        if not rec:
            out["warnings"].append(f"manifest has no output_file_stats.{label}")
            ok = False
            continue
        if Path(rec.get("path", "")).resolve() != path.resolve():
            out["warnings"].append(f"manifest path mismatch for {label}")
            ok = False
        if int(rec.get("size_bytes", -1)) != path.stat().st_size:
            out["warnings"].append(f"manifest size mismatch for {label}")
            ok = False
    out["manifest_binds_source_parquet"] = ok
    if not out["commit"]:
        out["warnings"].append("candidate manifest carries no run.git.head_sha")
    return out


def mbtiles_bounds(name: str) -> list[float] | None:
    p = WORK / f"{name}.mbtiles"
    if not p.exists():
        return None
    con = sqlite3.connect(p)
    try:
        row = con.execute("select value from metadata where name='bounds'").fetchone()
    finally:
        con.close()
    if not row:
        return None
    return [float(x) for x in row[0].split(",")]


def tile_stats(name: str) -> dict:
    p = WORK / f"{name}.mbtiles"
    con = sqlite3.connect(p)
    try:
        rows = con.execute(
            """
            with s as (
              select zoom_level z, length(tile_data) b,
                     row_number() over (partition by zoom_level
                                        order by length(tile_data)) rn,
                     count(*) over (partition by zoom_level) n
              from tiles)
            select z, n, sum(b),
                   max(case when rn <= cast(n*0.50 as int)+1 then b end),
                   max(case when rn <= cast(n*0.95 as int)+1 then b end),
                   max(b)
            from s group by z order by z
            """
        ).fetchall()
    finally:
        con.close()
    return {
        str(r[0]): {"tiles": r[1], "total_bytes": r[2], "p50": r[3], "p95": r[4], "max": r[5]}
        for r in rows
    }


def render_attribution() -> str:
    md = (REPO / "ATTRIBUTION.md").read_text()
    body: list[str] = []
    in_list = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
            continue
        if line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list and line.startswith("  ") and line.strip():
            body[-1] = body[-1][:-5] + " " + inline(line.strip()) + "</li>"
            continue
        if in_list:
            body.append("</ul>")
            in_list = False
        if line.strip():
            body.append(f"<p>{inline(line.strip())}</p>")
    if in_list:
        body.append("</ul>")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\" />\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        "<title>Sources and attribution &mdash; CrimeRisk</title>\n"
        "<link rel=\"stylesheet\" href=\"styles.css\" />\n</head>\n<body>\n"
        "<main class=\"doc\">\n<p class=\"topbar\"><a href=\"index.html\">&larr; Map</a>"
        "<a href=\"methods.html\">Methods</a><a href=\"download.html\">Download</a></p>\n"
        + "\n".join(body)
        + "\n</main>\n</body>\n</html>\n"
    )


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"(?<![\">=])\b(https?://[^\s,)]+)", r'<a href="\1">\1</a>', s)
    return s


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return str(n)


def download_tables(edition_line: str) -> tuple[str, str, str]:
    dl = SITE / "downloads"
    rows = []
    for geo, label in (("block_group", "Block group"), ("tract", "Census tract")):
        for ext, fmt in ((".csv.gz", "CSV (gzip)"), (".parquet", "Parquet"), (".geoparquet", "GeoParquet")):
            p = dl / f"crimerisk_{YEAR}_{geo}{ext}"
            if p.exists():
                rows.append(
                    f"<tr><td>{label}</td><td>{fmt}</td>"
                    f'<td><a href="{DOWNLOAD_BASE_URL}/{p.name}">{p.name}</a></td>'
                    f"<td>{human(p.stat().st_size)}</td></tr>"
                )
    national = (
        "<table><thead><tr><th>Geography</th><th>Format</th><th>File</th><th>Size</th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    states = sorted(
        {p.name.split("_")[-1].split(".")[0] for p in (dl / "state").glob("*.csv.gz")},
        key=lambda st: STATE_NAME.get(st, st),
    )
    cells = []
    for st in states:
        links = []
        for ext, lbl in ((".csv.gz", "CSV"), (".parquet", "Parquet"), (".geoparquet", "GeoParquet")):
            for geo, g in (("block_group", "BG"), ("tract", "Tract")):
                p = dl / "state" / f"crimerisk_{YEAR}_{geo}_{st}{ext}"
                if p.exists():
                    links.append(
                        f'<a href="{DOWNLOAD_BASE_URL}/state/{p.name}">{g} {lbl}</a>'
                    )
        cells.append(
            f"<tr><td>{STATE_NAME.get(st, st)}</td><td>" + " &middot; ".join(links) + "</td></tr>"
        )
    state_tbl = (
        "<table><thead><tr><th>State</th><th>Files</th></tr></thead><tbody>"
        + "".join(cells)
        + "</tbody></table>"
    )
    return national, state_tbl, f"<p>{html.escape(edition_line)}</p>"


def status_card_copy(floors: dict, county_floor: dict | None) -> dict:
    """Card text per publication status, with the floors the source data states."""
    out = {k: v for k, v in STATUS_CARD.items() if not k.endswith("_generic")}
    try:
        out["suppressed_resident"] = STATUS_CARD["suppressed_resident"].format(**floors)
    except KeyError as missing:
        print(f"WARNING: floor {missing} not in the source parquet; resident card copy is generic")
        out["suppressed_resident"] = STATUS_CARD["suppressed_resident_generic"]
    if county_floor:
        out["suppressed_county"] = STATUS_CARD["suppressed_county"].format(**county_floor)
    else:
        out["suppressed_county"] = NO_ESTIMATE_LABEL
    return out


def status_table_html(card: dict, legend: dict) -> str:
    """Methods-page table of the unpublished states, in the card's own words."""
    rows = [
        (legend["suppressed_low_denominator"], "Grey, hatched",
         ["Per resident: " + card["suppressed_resident"],
          "Crime exposure, one offense: " + card["suppressed_exposure"],
          "Crime exposure, composite: " + card["suppressed_exposure_composite"],
          "County average: " + card["suppressed_county"]]),
        ("Tract only", "Tract shown",
         [card["tract_only"].replace("{offense}", "Murder and rape")]),
        (legend["no_data"], "Grey", [card["no_data"]]),
    ]
    body = "".join(
        f"<tr><td>{html.escape(state)}</td><td>{html.escape(look)}</td><td>"
        + "<br />".join(html.escape(r) for r in rules) + "</td></tr>"
        for state, look, rules in rows
    )
    return (
        "<table><thead><tr><th>State</th><th>Map</th><th>Card</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def evidence_html() -> str:
    """Evidence-bundle link slot; empty (and a warning) when no bundle was built."""
    tarball = EVIDENCE_DIR / EVIDENCE_NAME
    digest_file = EVIDENCE_DIR / f"{EVIDENCE_NAME}.sha256"
    if not tarball.exists() or not digest_file.exists():
        print(f"WARNING: no evidence bundle at {tarball}; the evidence link slot stays empty")
        return ""
    digest = digest_file.read_text().split()[0]
    return (
        "<h2>Evidence bundle</h2>"
        "<table><thead><tr><th>File</th><th>Size</th><th>SHA-256</th></tr></thead><tbody>"
        f'<tr><td><a href="{EVIDENCE_URL}">{EVIDENCE_NAME}</a></td>'
        f"<td>{human(tarball.stat().st_size)}</td><td><code>{digest}</code></td></tr>"
        "</tbody></table>"
    )


def ledger_html() -> str:
    """Download-page row for the boundary exception ledger (05b), if built."""
    p = SITE / "downloads" / LEDGER_NAME
    if not p.exists():
        print(f"WARNING: no exception ledger at {p}; run 05b_exception_ledger.py")
        return ""
    n = max(sum(1 for _ in p.open()) - 1, 0)
    return (
        "<h2>Boundary exception ledger</h2>"
        "<table><thead><tr><th>File</th><th>Rows</th><th>Size</th></tr></thead><tbody>"
        f'<tr><td><a href="{DOWNLOAD_BASE_URL}/{p.name}">{p.name}</a></td>'
        f"<td>{n:,}</td><td>{human(p.stat().st_size)}</td></tr></tbody></table>"
    )


def preconnect_html(origins: list[str]) -> str:
    """Early connections for the critical path: the manifest fetch starts while the
    scripts download, and the tile and basemap hosts are connected before the map
    asks for their first byte."""
    tags = ['<link rel="preload" href="data/manifest.json" as="fetch" crossorigin />']
    seen = []
    for o in origins:
        if o and o not in seen:
            seen.append(o)
            tags.append(f'<link rel="preconnect" href="{o}" crossorigin />')
    return "\n".join(tags)


def origin(url: str) -> str | None:
    m = re.match(r"^(https?://[^/]+)", url)
    return m.group(1) if m else None


def crime_help() -> dict:
    """The per-index help text, filled from the snapshot (weights and the premises fit)."""
    import pyarrow.parquet as pq

    cols = [f"expected_count_{o}" for o in OFFENSES]
    counts = {o: float(v) for o, v in zip(OFFENSES, pq.read_table(BG_SRC, columns=cols).to_pandas().sum())}
    members = {
        "ov": list(OFFENSES),
        "vi": ["murder", "rape", "robbery", "aggravated_assault"],
        "pr": ["burglary", "larceny", "motor_vehicle_theft"],
    }

    def weights(key: str) -> str:
        total = sum(counts[o] for o in members[key])
        parts = sorted(((counts[o] / total, o) for o in members[key]), reverse=True)
        return ", ".join(
            f"{OFFENSE_LABEL[o].lower()} {100 * w:.1f}%" if w < 0.01 else f"{OFFENSE_LABEL[o].lower()} {100 * w:.0f}%"
            for w, o in parts
        )

    manifest = json.loads((BG_SRC.parent / "manifest.json").read_text())
    calibration = manifest.get("summary", {}).get("burglary_commercial_calibration", {})
    gq = calibration.get("k_group_quarters")
    fields = {f"weights_{k}": weights(k) for k in members}
    fields["gq"] = f"{float(gq):.2f}" if gq is not None else "a fitted fraction of a"
    out: dict = {}
    for (key, measure), text in CRIME_HELP.items():
        label = next((OFFENSE_LABEL[o] for o in OFFENSES if SHORT[o] == key), key)
        out.setdefault(key, {})[measure] = text.format(
            label_lc=label.lower(), half_life=CRIME_HELP_HALF_LIFE.get(key, ""), **fields
        )
    return out


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE / "vendor").mkdir(parents=True, exist_ok=True)

    extract = json.loads((WORK / "extract_manifest.json").read_text())
    prov = source_provenance()

    # --- tiles: hard link into the site, no second copy on disk ------------
    tiles = {}
    tile_report = {}
    for level, (name, layer, zmin, zmax) in TILE_LAYERS.items():
        src = DIST / f"crimerisk_{name}_{YEAR}.pmtiles"
        if not src.exists():
            raise FileNotFoundError(f"{src} missing - run 03_tiles.sh")
        dst = TILE_DIR / src.name
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        tiles[level] = {
            "url": f"{ASSET_BASE_URL}/{src.name}",
            "layer": layer,
            "minzoom": zmin,
            "maxzoom": zmax,
            "bytes": src.stat().st_size,
        }
        tile_report[level] = {"archive_bytes": src.stat().st_size, "per_zoom": tile_stats(name)}

    bounds = mbtiles_bounds("blockgroups") or [-124.8, 24.4, -66.9, 49.4]

    edition = EDITION_LABEL
    # The footer and the download page carry the edition. The source commit is
    # provenance, not front-page copy, so it appears only under Technical
    # methods.
    edition_line = f"CrimeRisk edition {edition} · data year {YEAR} · 48 states and DC"
    provenance_line = (
        f"Edition {edition}, data year {YEAR}. Built from source commit "
        f"{prov.get('commit_short') or 'unknown'}; block-group parquet sha256 "
        f"{extract['source']['block_group_sha256'][:16]}…"
    )

    manifest = {
        "edition": edition,
        "data_year": YEAR,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "stamped_at_git_commit": git_head(),
        "source_data_provenance": prov,
        "source": extract["source"],
        "counts": {
            "block_groups": extract["n_block_groups"],
            "tracts": extract["n_tracts"],
            "counties": extract["n_counties"],
            "states": len(extract["states"]),
        },
        "bounds": bounds,
        "zoom": {"tract_min": 5, "bg_min": 8, "max": 13},
        "tiles": tiles,
        "basemap": BASEMAP,
        "attribution": ATTRIBUTION,
        "measures": [
            {"key": k, "label": MEASURE_LABEL[k], "subcopy": MEASURE_SUBCOPY[k]}
            for k in MEASURES
        ],
        "crimes": CRIME_ORDER,
        "crime_help": crime_help(),
        "measure_order": MEASURE_ORDER,
        "offense_index": {SHORT[o]: i for i, o in enumerate(OFFENSES)},
        "composite_members": {
            "ov": list(range(len(OFFENSES))),
            "vi": [OFFENSES.index(o) for o in ["murder", "rape", "robbery", "aggravated_assault"]],
            "pr": [OFFENSES.index(o) for o in ["burglary", "larceny", "motor_vehicle_theft"]],
        },
        # Position of each composite in the shard's direct-share array.
        "composite_index": {"ov": 0, "vi": 1, "pr": 2},
        "denominator_exposure": {SHORT[o]: DENOMINATOR_EXPOSURE[o] for o in OFFENSES},
        "coverage_bounds": COVERAGE_BOUNDS,
        "mode_direct_weight": {"0": 1.0, "1": 0.5, "2": 0.0},
        "direct_share_threshold": DIRECT_SHARE_THRESHOLD,
        "legend": LEGEND,
        "no_estimate": {"color": NO_ESTIMATE_COLOR, "label": NO_ESTIMATE_LABEL},
        # Per-measure publication status: shard records carry the codes as an array
        # (position = measure_order); tiles carry them packed base `base` in `ms`.
        "status": {
            "codes": {str(k): v for k, v in MEASURE_STATUS.items()},
            "base": STATUS_BASE,
            "tile_field": "ms",
            "legend": STATUS_LEGEND,
            "card": status_card_copy(
                extract.get("suppression_floors", {}), extract.get("county_support_floor")
            ),
        },
        "suppression_floors": extract.get("suppression_floors", {}),
        "county_support_floor": extract.get("county_support_floor"),
        "counties_below_support_floor": extract.get("n_counties_below_support_floor"),
        "special_use": {str(k): v for k, v in SPECIAL_USE_LABEL.items()},
        "count_noun": {k: list(v) for k, v in COUNT_NOUN.items()},
        "suppressed_count": {k: v for k, v in SUPPRESSED_COUNT_COPY.items() if k != "methods"},
        "state_abbr_by_fips": FIPS_TO_USPS,
        "state_name_by_fips": {f: STATE_NAME[a] for f, a in FIPS_TO_USPS.items() if a in STATE_NAME},
        "copy": {
            "tagline": "neighborhood crime compared with the U.S.",
            "edition_line": edition_line,
            "provenance_line": provenance_line,
            "source_direct": SOURCE_PHRASE_DIRECT,
            "source_modeled": SOURCE_PHRASE_MODELED,
            "source_mostly_direct": SOURCE_PHRASE_MOSTLY_DIRECT,
            "source_mostly_modeled": SOURCE_PHRASE_MOSTLY_MODELED,
            "denominator_resident": DENOMINATOR_RESIDENT,
            "denominator_exposure_composite": DENOMINATOR_EXPOSURE_COMPOSITE,
            "total_reported": LEVEL_TOTAL_REPORTED.format(year=YEAR),
            "total_estimated": LEVEL_TOTAL_ESTIMATED.format(year=YEAR),
            "total_mixed": LEVEL_TOTAL_MIXED.format(year=YEAR),
            "search_no_match": "No match found.",
            "search_unavailable": (
                "Search is unavailable right now. This is a problem with the address "
                "lookup service, not with the address."
            ),
            "search_outside_coverage": (
                "That place is outside this map. It covers the 48 contiguous states "
                "and DC; Alaska, Hawaii and the territories are not published."
            ),
            "county_chip": "County average · zoom in for neighborhoods",
            "county_note": (
                "County rate, built from the offense counts and denominators of "
                "every tract in the county."
            ),
        },
    }
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    mpath = SITE / "data" / "manifest.json"
    mpath.write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"manifest.json {mpath.stat().st_size:,} bytes")
    if mpath.stat().st_size > 50 * 1024:
        raise SystemExit("bootstrap manifest exceeds the 50 KB budget")

    # --- static assets with content hashes ---------------------------------
    hashes = {}
    for name in STATIC:
        src = PUBLIC / name
        shutil.copy2(src, SITE / name)
        hashes[name] = sha256(src, 10)
    for name in ["maplibre-gl.js", "maplibre-gl.css", "pmtiles.js"]:
        src = PUBLIC / "vendor" / name
        shutil.copy2(src, SITE / "vendor" / name)
        hashes["vendor/" + name] = sha256(src, 10)

    national, state_tbl, edition_html = download_tables(edition_line)
    evidence = evidence_html()
    ledger = ledger_html()
    early = preconnect_html(
        [origin(t["url"]) for t in tiles.values()] + [origin(BASEMAP["style"])]
    )
    for page in PAGES:
        text = (PUBLIC / page).read_text()
        text = text.replace("<!--NATIONAL-->", national)
        text = text.replace("<!--STATES-->", state_tbl)
        text = text.replace("<!--EDITION-->", edition_html)
        text = text.replace("<!--EVIDENCE-->", evidence)
        text = text.replace("<!--LEDGER-->", ledger)
        text = text.replace("<!--PRECONNECT-->", early)
        text = text.replace("<!--SUPPRESSED_COUNT-->", html.escape(SUPPRESSED_COUNT_COPY["methods"]))
        text = text.replace(
            "<!--STATUS_TABLE-->",
            status_table_html(manifest["status"]["card"], manifest["status"]["legend"]),
        )
        # The field dictionaries and the checksum file are linked by hand in
        # download.html and live in the same R2 tree as everything else.
        text = text.replace('href="downloads/', f'href="{DOWNLOAD_BASE_URL}/')
        for asset, h in hashes.items():
            text = text.replace(f'"{asset}"', f'"{asset}?v={h}"')
        (SITE / page).write_text(text)

    att = render_attribution()
    for asset, h in hashes.items():
        att = att.replace(f'"{asset}"', f'"{asset}?v={h}"')
    (SITE / "attribution.html").write_text(att)

    # Refresh checksums last, so they cover the download tree exactly as staged.
    dl = SITE / "downloads"
    if dl.exists():
        lines = []
        for f in sorted(dl.rglob("*")):
            if f.is_file() and f.name != "checksums.sha256":
                lines.append(f"{sha256(f)}  {f.relative_to(dl)}")
        (dl / "checksums.sha256").write_text("\n".join(lines) + "\n")
        print(f"checksums.sha256 covers {len(lines)} files")

    report = {
        "asset_base_url": ASSET_BASE_URL,
        "download_base_url": DOWNLOAD_BASE_URL,
        "manifest_bytes": mpath.stat().st_size,
        "tiles": tile_report,
        "asset_hashes": hashes,
        "source_data_provenance": prov,
    }
    (WORK / "site_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["archive_bytes"] for k, v in tile_report.items()}, indent=2))
    print(f"tiles and downloads served from {ASSET_BASE_URL}")
    print(f"Site staged at {SITE}")


if __name__ == "__main__":
    main()
