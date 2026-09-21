"""Frozen-contract Charlotte rape forward acquisition and exact tract scoring.

No request is made unless ``--acquire`` is supplied and all candidate, contract,
fit-cutoff, and support-mask hashes agree.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd

from crimerisk.city_feed_quarantine import flag_quarantined_coordinates
from crimerisk.exact_surface_evaluation import PREDICTION_EPSILON, score_distribution

LAYER_URL = (
    "https://gis.charlottenc.gov/arcgis/rest/services/CMPD/CMPDIncidents/MapServer/0"
)
QUERY_URL = f"{LAYER_URL}/query"
JURISDICTION_ID = "37:municipal:place:3712000"
SOURCE = "charlotte_rape_forward_2026h1"
START_LOCAL = "2026-01-01 00:00:00"
END_LOCAL_EXCLUSIVE = "2026-07-01 00:00:00"
LOCAL_TIME_ZONE = "America/New_York"
OFFENSE = "rape"
RAPE_CODE = "11A"
FIELDS = (
    "OBJECTID",
    "YEAR",
    "INCIDENT_REPORT_ID",
    "CITY",
    "CLEARANCE_STATUS",
    "HIGHEST_NIBRS_CODE",
    "HIGHEST_NIBRS_DESCRIPTION",
    "LATITUDE_PUBLIC",
    "LONGITUDE_PUBLIC",
    "DATE_REPORTED",
)
POINT_SHARE_MAX = 0.005
MIN_LOCATED = 100
MIN_POINT_COUNT = 5


def report_date_window_mask(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Interpret ArcGIS date epochs as UTC instants, then apply the local CMPD window."""
    report_date = pd.to_datetime(values, unit="ms", errors="coerce", utc=True).dt.tz_convert(
        LOCAL_TIME_ZONE
    )
    start = pd.Timestamp(START_LOCAL, tz=LOCAL_TIME_ZONE)
    end = pd.Timestamp(END_LOCAL_EXCLUSIVE, tz=LOCAL_TIME_ZONE)
    return report_date, report_date.ge(start) & report_date.lt(end)


def scoring_eligible(*, exact_point_violation: bool, retained_incidents: int) -> bool:
    """Require usable truth, without imposing an outcome-dependent sample-size cutoff."""
    return not exact_point_violation and retained_incidents > 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def where_clause() -> str:
    return (
        f"DATE_REPORTED >= timestamp '{START_LOCAL}' AND DATE_REPORTED < timestamp "
        f"'{END_LOCAL_EXCLUSIVE}' AND CITY='CHARLOTTE' AND CLEARANCE_STATUS <> "
        f"'Unfounded' AND HIGHEST_NIBRS_CODE='{RAPE_CODE}'"
    )


def query_params(*, offset: int, page_size: int) -> dict[str, str]:
    return {
        "where": where_clause(),
        "outFields": ",".join(FIELDS),
        "returnGeometry": "false",
        "orderByFields": "OBJECTID ASC",
        "resultOffset": str(offset),
        "resultRecordCount": str(page_size),
        "f": "json",
    }


def _load_contract(contract_path: Path, candidate_manifest: Path) -> dict:
    c = json.loads(contract_path.read_text())
    for key in (
        "candidate_frozen",
        "evaluation_contract_frozen",
        "selected_before_target_outcomes",
    ):
        if c.get(key) is not True:
            raise ValueError(f"Acquisition requires {key}=true")
    if (
        c.get("target_window") != "2026-01-01/2026-06-30"
        or c.get("target_date_field") != "DATE_REPORTED"
    ):
        raise ValueError(
            "Charlotte target must use the frozen Jan-Jun local report-date window"
        )
    if int(c.get("fit_effective_max_year", 9999)) > 2025:
        raise ValueError("Fit event year exceeds 2025")
    if c.get("prospective_as_of_2025_12_31") is not False:
        raise ValueError("Prospective claim forbidden")
    if sha256(candidate_manifest) != c.get("candidate_manifest_sha256"):
        raise ValueError("Candidate hash mismatch")
    for pkey, hkey in [
        ("fit_cutoff_audit_path", "fit_cutoff_audit_sha256"),
        ("support_mask_path", "support_mask_sha256"),
    ]:
        p = Path(str(c.get(pkey, "")))
        if not p.is_file() or sha256(p) != c.get(hkey):
            raise ValueError(f"{pkey} binding mismatch")
    return c


def freeze_support_mask(candidate_bg: Path, out_path: Path) -> pd.DataFrame:
    d = pd.read_parquet(
        candidate_bg, columns=["block_group_geoid", "tract_id", "eb_jurisdiction_id"]
    )
    d["block_group_geoid"] = d.block_group_geoid.astype("string").str.zfill(12)
    d["tract_id"] = d.tract_id.astype("string").str.zfill(11)
    total = d.groupby("tract_id").block_group_geoid.nunique()
    inside = d[d.eb_jurisdiction_id.astype(str).eq(JURISDICTION_ID)]
    count = inside.groupby("tract_id").block_group_geoid.nunique()
    tracts = count[count.eq(total.reindex(count.index))].index
    out = (
        inside[inside.tract_id.isin(tracts)][["block_group_geoid", "tract_id"]]
        .drop_duplicates()
        .sort_values(["tract_id", "block_group_geoid"])
    )
    if out.empty:
        raise ValueError("No fully contained Charlotte tracts")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def acquire_pages(
    out_dir: Path, *, page_size: int, max_pages: int
) -> tuple[list[dict], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(
        Request(
            LAYER_URL + "?f=json",
            headers={"User-Agent": "CrimeRisk-forward-evaluation/1"},
        ),
        timeout=60,
    ) as r:
        metadata_raw = r.read()
    metadata_path = out_dir / "layer_metadata.json"
    metadata_path.write_bytes(metadata_raw)
    pages = []
    offset = 0
    for page_index in range(max_pages):
        params = query_params(offset=offset, page_size=page_size)
        url = QUERY_URL + "?" + urlencode(params)
        with urlopen(
            Request(url, headers={"User-Agent": "CrimeRisk-forward-evaluation/1"}),
            timeout=120,
        ) as r:
            raw = r.read()
        payload = json.loads(raw)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        feats = payload.get("features", [])
        path = out_dir / f"page_{page_index:03d}_{offset:07d}.json"
        path.write_bytes(raw)
        pages.append(
            {
                "page_index": page_index,
                "offset": offset,
                "rows": len(feats),
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "query": params,
            }
        )
        if not payload.get("exceededTransferLimit", False) and len(feats) < page_size:
            break
        if not feats:
            break
        offset += len(feats)
    else:
        raise RuntimeError(f"Acquisition exceeded frozen {max_pages}-page cap")
    return pages, {
        "path": str(metadata_path.resolve()),
        "sha256": sha256(metadata_path),
    }


def prepare_truth(
    page_paths: list[Path], *, nc_bg_zip: Path, support_mask_path: Path
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for p in page_paths:
        payload = json.loads(p.read_text())
        rows.extend(f.get("attributes", {}) for f in payload.get("features", []))
    raw = pd.DataFrame(rows)
    missing = set(FIELDS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing endpoint fields {sorted(missing)}")
    raw = raw.sort_values(["OBJECTID", "INCIDENT_REPORT_ID"], kind="mergesort")
    raw["INCIDENT_REPORT_ID"] = raw.INCIDENT_REPORT_ID.astype("string").str.strip()
    blank = raw.INCIDENT_REPORT_ID.isna() | raw.INCIDENT_REPORT_ID.eq("")
    duplicates = int(raw.loc[~blank].INCIDENT_REPORT_ID.duplicated().sum())
    raw = raw.loc[~blank].drop_duplicates("INCIDENT_REPORT_ID", keep="first").copy()
    raw["report_date"], in_window = report_date_window_mask(raw.DATE_REPORTED)
    semantic = (
        raw.CITY.astype(str).eq("CHARLOTTE")
        & ~raw.CLEARANCE_STATUS.astype(str).eq("Unfounded")
        & raw.HIGHEST_NIBRS_CODE.astype(str).eq(RAPE_CODE)
    )
    mapped = raw[in_window & semantic].copy()
    mapped["offense"] = OFFENSE
    mapped["latitude"] = pd.to_numeric(mapped.LATITUDE_PUBLIC, errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped.LONGITUDE_PUBLIC, errors="coerce")
    plausible = mapped.latitude.between(34.9, 35.5) & mapped.longitude.between(
        -81.1, -80.5
    )
    located = mapped[plausible].copy()
    quarantine = flag_quarantined_coordinates(
        located, city_key="charlotte", offense_col="offense"
    )
    located = located.loc[~quarantine].copy()
    points = gpd.GeoDataFrame(
        located,
        geometry=gpd.points_from_xy(located.longitude, located.latitude),
        crs="EPSG:4326",
    )
    bg = (
        gpd.read_file(nc_bg_zip)[["GEOID", "geometry"]]
        .rename(columns={"GEOID": "block_group_geoid"})
        .to_crs("EPSG:4326")
    )
    bg["block_group_geoid"] = bg.block_group_geoid.astype(str).str.zfill(12)
    mask = pd.read_csv(support_mask_path, dtype=str)
    mask["block_group_geoid"] = mask.block_group_geoid.str.zfill(12)
    mask["tract_id"] = mask.tract_id.str.zfill(11)
    bg = bg.merge(mask, on="block_group_geoid", how="inner")
    matched = (
        gpd.sjoin(points, bg, how="inner", predicate="within").drop_duplicates(
            "INCIDENT_REPORT_ID"
        )
        if len(points)
        else points.assign(
            block_group_geoid=pd.Series(dtype=str), tract_id=pd.Series(dtype=str)
        )
    )
    counts = (
        matched.groupby("tract_id").size() if len(matched) else pd.Series(dtype=int)
    )
    truth = mask[["tract_id"]].drop_duplicates().sort_values("tract_id")
    truth["incident_count"] = truth.tract_id.map(counts).fillna(0).astype(int)
    truth.insert(0, "offense", OFFENSE)
    truth.insert(0, "year", 2026)
    truth.insert(0, "jurisdiction_id", JURISDICTION_ID)
    truth.insert(0, "source", SOURCE)
    pc = (
        located.assign(
            point_key=located.latitude.map(repr) + "|" + located.longitude.map(repr)
        )
        .groupby("point_key")
        .size()
        .sort_values(ascending=False)
    )
    top = int(pc.iloc[0]) if len(pc) else 0
    share = top / len(located) if len(located) else 0
    violation = (
        len(located) >= MIN_LOCATED
        and top >= MIN_POINT_COUNT
        and share >= POINT_SHARE_MAX
    )
    audit = {
        "raw_rows": len(rows),
        "blank_id_rows": int(blank.sum()),
        "duplicate_id_rows": duplicates,
        "unique_rows": len(raw),
        "mapped_in_window": len(mapped),
        "plausible_coordinates": int(plausible.sum()),
        "quarantined": int(quarantine.sum()),
        "whole_tract_matched": len(matched),
        "whole_tract_unmatched": len(located) - len(matched),
        "whole_tract_count": len(truth),
        "zero_incident_tract_count": int(truth.incident_count.eq(0).sum()),
        "top_point_count": top,
        "top_point_share": share,
        "exact_point_violation": violation,
        "eligible_for_scoring": scoring_eligible(
            exact_point_violation=bool(violation),
            retained_incidents=int(truth.incident_count.sum()),
        ),
    }
    return truth, audit


def score_truth(truth: Path, arms_json: Path) -> list[dict]:
    t = pd.read_parquet(truth)
    support = t.tract_id.astype(str).str.zfill(11)
    rows = []
    arms = json.loads(arms_json.read_text())
    arms = arms["arms"] if isinstance(arms, dict) else arms
    for arm in arms:
        d = pd.read_parquet(
            arm["tract_path"],
            columns=[
                "tract_id",
                "expected_count_rape",
                "primary_denominator_rape",
                "population_2025",
            ],
        )
        d.tract_id = d.tract_id.astype(str).str.zfill(11)
        d = d.set_index("tract_id").reindex(support)
        if d.isna().any().any():
            raise ValueError(f"{arm['name']} misses fixed tract support")
        for name, col in [
            (arm["name"], "expected_count_rape"),
            (f"primary_exposure__{arm['name']}", "primary_denominator_rape"),
        ]:
            rows.append(
                {
                    "surface": name,
                    **score_distribution(
                        t.incident_count,
                        d[col].reset_index(drop=True),
                        support_ids=support,
                        epsilon=PREDICTION_EPSILON,
                    ),
                }
            )
        if arm is arms[0]:
            for name, vals in [
                ("uniform", pd.Series(1.0, index=t.index)),
                ("population", d.population_2025.reset_index(drop=True)),
            ]:
                rows.append(
                    {
                        "surface": name,
                        **score_distribution(
                            t.incident_count,
                            vals,
                            support_ids=support,
                            epsilon=PREDICTION_EPSILON,
                        ),
                    }
                )
    return rows


def require_scoring_eligibility(truth_path: Path, acquisition_manifest: Path) -> None:
    """Fail closed unless acquisition QA admitted this exact truth artifact."""
    manifest = json.loads(acquisition_manifest.read_text())
    audit = manifest.get("retention_audit", {})
    if audit.get("eligible_for_scoring") is not True:
        raise ValueError("Acquisition retention audit marks truth ineligible for scoring")
    if manifest.get("truth_sha256") != sha256(truth_path):
        raise ValueError("Scoring truth hash does not match acquisition manifest")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze-support")
    f.add_argument("--candidate-bg", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("acquire")
    for n in ["contract", "candidate-manifest", "nc-bg-zip", "out-dir"]:
        a.add_argument("--" + n, type=Path, required=True)
    a.add_argument("--page-size", type=int, default=2500)
    a.add_argument("--max-pages", type=int, default=10)
    a.add_argument("--acquire", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--truth", type=Path, required=True)
    s.add_argument("--acquisition-manifest", type=Path, required=True)
    s.add_argument("--arms-json", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    x = ap.parse_args()
    if x.cmd == "freeze-support":
        freeze_support_mask(x.candidate_bg, x.out)
        return 0
    if x.cmd == "score":
        require_scoring_eligibility(x.truth, x.acquisition_manifest)
        x.out.write_text(json.dumps(score_truth(x.truth, x.arms_json), indent=2) + "\n")
        return 0
    if not x.acquire:
        raise SystemExit("Refusing network access without explicit --acquire")
    c = _load_contract(x.contract, x.candidate_manifest)
    pages, metadata = acquire_pages(
        x.out_dir / "raw_pages", page_size=x.page_size, max_pages=x.max_pages
    )
    truth, audit = prepare_truth(
        [Path(p["path"]) for p in pages],
        nc_bg_zip=x.nc_bg_zip,
        support_mask_path=Path(c["support_mask_path"]),
    )
    truth_path = x.out_dir / "charlotte_rape_2026h1_whole_tract_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    manifest = {
        "version": "charlotte_rape_forward_2026h1_v1",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date_semantics": "CMPD DATE_REPORTED, local Eastern service time, start-inclusive/end-exclusive",
        "target_event_holdout": True,
        "prospective_as_of_2025_12_31": False,
        "same_city_historical_source_reuse": True,
        "contract_path": str(x.contract.resolve()),
        "contract_sha256": sha256(x.contract),
        "candidate_manifest_sha256": sha256(x.candidate_manifest),
        "layer_metadata": metadata,
        "pages": pages,
        "truth_path": str(truth_path.resolve()),
        "truth_sha256": sha256(truth_path),
        "retention_audit": audit,
    }
    (x.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
