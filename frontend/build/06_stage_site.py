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
    CRIME_ORDER,
    DIRECT_SHARE_THRESHOLD,
    DIST,
    FIPS_TO_USPS,
    LEGEND,
    MEASURE_LABEL,
    NO_ESTIMATE_COLOR,
    NO_ESTIMATE_LABEL,
    MEASURE_ORDER,
    MEASURE_SUBCOPY,
    MEASURES,
    OFFENSES,
    REPO,
    SHORT,
    SOURCE_PHRASE_DIRECT,
    SOURCE_PHRASE_MODELED,
    SOURCE_PHRASE_MOSTLY_DIRECT,
    SOURCE_PHRASE_MOSTLY_MODELED,
    SPECIAL_USE_LABEL,
    STATE_NAME,
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
EDITION = f"{YEAR}.1"
DEFAULT_ASSET_BASE_URL = f"https://tiles.qqlab.io/{EDITION}"
ASSET_BASE_URL = os.environ.get("CRIMERISK_ASSET_BASE_URL", DEFAULT_ASSET_BASE_URL).rstrip("/")
DOWNLOAD_BASE_URL = f"{ASSET_BASE_URL}/downloads"

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
    out["dirty"] = bool(git.get("dirty"))
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

    edition = EDITION
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
        "measure_order": MEASURE_ORDER,
        "offense_index": {SHORT[o]: i for i, o in enumerate(OFFENSES)},
        "composite_members": {
            "ov": list(range(len(OFFENSES))),
            "vi": [OFFENSES.index(o) for o in ["murder", "rape", "robbery", "aggravated_assault"]],
            "pr": [OFFENSES.index(o) for o in ["burglary", "larceny", "motor_vehicle_theft"]],
        },
        # Position of each composite in the shard's direct-share array.
        "composite_index": {"ov": 0, "vi": 1, "pr": 2},
        "mode_direct_weight": {"0": 1.0, "1": 0.5, "2": 0.0},
        "direct_share_threshold": DIRECT_SHARE_THRESHOLD,
        "legend": LEGEND,
        "no_estimate": {"color": NO_ESTIMATE_COLOR, "label": NO_ESTIMATE_LABEL},
        "county_support_floor": extract.get("county_support_floor"),
        "counties_below_support_floor": extract.get("n_counties_below_support_floor"),
        "special_use": {str(k): v for k, v in SPECIAL_USE_LABEL.items()},
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
    for page in PAGES:
        text = (PUBLIC / page).read_text()
        text = text.replace("<!--NATIONAL-->", national)
        text = text.replace("<!--STATES-->", state_tbl)
        text = text.replace("<!--EDITION-->", edition_html)
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
