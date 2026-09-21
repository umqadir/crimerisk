"""Evaluation-only Chicago Jan-Jun 2026 acquisition and truth preparation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd

from crimerisk.city_feed_quarantine import flag_quarantined_coordinates
from crimerisk.city_incidents import (
    CHICAGO_BOUNDS,
    CHICAGO_JURISDICTION_ID,
    CHICAGO_RESOURCE_COLUMN_MAP,
    CHICAGO_RESOURCE_URL,
    _load_chicago_categories,
)

START = "2026-01-01T00:00:00"
END_EXCLUSIVE = "2026-07-01T00:00:00"
SOURCE = "chicago_forward_2026h1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def socrata_params(*, limit: int, offset: int) -> dict[str, str]:
    return {
        "$select": ",".join(CHICAGO_RESOURCE_COLUMN_MAP.keys()),
        "$where": f"date >= '{START}' AND date < '{END_EXCLUSIVE}'",
        "$order": "id",
        "$limit": str(limit),
        "$offset": str(offset),
    }


def validate_gate(contract_path: Path, candidate_manifest_path: Path) -> dict[str, object]:
    contract = json.loads(contract_path.read_text())
    required_true = ("candidate_frozen", "evaluation_contract_frozen")
    if any(contract.get(key) is not True for key in required_true):
        raise ValueError(f"Acquisition requires true gates: {required_true}")
    if str(contract.get("target_window")) != "2026-01-01/2026-06-30":
        raise ValueError("target_window must be exactly 2026-01-01/2026-06-30")
    if str(contract.get("evaluation_type")) != "conditional_spatial_distribution":
        raise ValueError("evaluation_type must be conditional_spatial_distribution")
    try:
        cutoff_year = int(contract.get("fit_effective_max_year"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Contract must record fit_effective_max_year") from exc
    if cutoff_year > 2025:
        raise ValueError("Candidate inputs must have an effective fit cutoff year no later than 2025")
    if contract.get("target_event_holdout") is not True:
        raise ValueError("Contract must identify this as a target-event holdout")
    if contract.get("prospective_as_of_2025_12_31") is not False:
        raise ValueError("Contract must not claim prospective as-of-2025 availability")
    if str(contract.get("claim_label")) != "retrospective out-of-period target-event spatial validation":
        raise ValueError("Contract claim label does not match the audited evidence")
    audit_path = Path(str(contract.get("fit_cutoff_audit_path", "")))
    if not audit_path.is_file() or sha256(audit_path) != str(contract.get("fit_cutoff_audit_sha256")):
        raise ValueError("Fit-cutoff audit path or hash does not match frozen contract")
    actual = sha256(candidate_manifest_path)
    if actual != str(contract.get("candidate_manifest_sha256")):
        raise ValueError("Candidate manifest hash does not match frozen contract")
    return contract


def acquire_pages(out_dir: Path, *, page_size: int) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_records: list[dict[str, object]] = []
    offset = 0
    while True:
        params = socrata_params(limit=page_size, offset=offset)
        url = f"{CHICAGO_RESOURCE_URL}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "CrimeRisk-forward-evaluation/1"})
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        page = pd.read_csv(io.BytesIO(payload), dtype=str)
        page_path = out_dir / f"page_{offset:09d}.csv"
        page_path.write_bytes(payload)
        page_records.append({
            "offset": offset, "rows": int(len(page)), "path": str(page_path.resolve()),
            "sha256": sha256(page_path), "url": url,
        })
        if len(page) < page_size:
            break
        offset += page_size
    return page_records


def prepare_truth(
    pages: list[Path], *, categories_csv: Path, il_bg_zip: Path,
    bg_crosswalk_parquet: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.concat([pd.read_csv(path, dtype=str) for path in pages], ignore_index=True)
    raw = raw.rename(columns=CHICAGO_RESOURCE_COLUMN_MAP)
    missing = set(CHICAGO_RESOURCE_COLUMN_MAP.values()) - set(raw.columns)
    if missing:
        raise ValueError(f"Chicago page schema missing {sorted(missing)}")
    raw["id"] = raw["id"].astype("string").str.strip()
    raw = raw[raw["id"].notna() & raw["id"].ne("")].drop_duplicates("id", keep="last")
    raw["incident_dt"] = pd.to_datetime(raw["date"], errors="coerce", utc=True)
    in_window = raw["incident_dt"].ge(pd.Timestamp(START, tz="UTC")) & raw["incident_dt"].lt(
        pd.Timestamp(END_EXCLUSIVE, tz="UTC")
    )
    raw = raw[in_window].copy()
    cats = _load_chicago_categories(categories_csv)
    raw["primary_type"] = raw["primary_type"].astype(str).str.strip()
    raw["description"] = raw["description"].astype(str).str.strip()
    mapped = raw.merge(cats, on=["primary_type", "description"], how="left")
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    plausible = (
        mapped["latitude"].between(CHICAGO_BOUNDS["min_latitude"], CHICAGO_BOUNDS["max_latitude"])
        & mapped["longitude"].between(CHICAGO_BOUNDS["min_longitude"], CHICAGO_BOUNDS["max_longitude"])
    )
    geocoded = mapped[plausible].copy()
    quarantine = flag_quarantined_coordinates(
        geocoded, city_key="chicago", offense_col="offense"
    )
    located = geocoded.loc[~quarantine].copy()
    points = gpd.GeoDataFrame(
        located[["id", "offense", "latitude", "longitude"]],
        geometry=gpd.points_from_xy(located["longitude"], located["latitude"]), crs="EPSG:4326",
    )
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[
        ["block_group_geoid", "jurisdiction_id"]
    ].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(CHICAGO_JURISDICTION_ID)]
    bg = gpd.read_file(il_bg_zip)[["GEOID", "geometry"]].rename(
        columns={"GEOID": "block_group_geoid"}
    )
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    if bg.crs != points.crs:
        bg = bg.to_crs(points.crs)
    matched = gpd.sjoin(points, bg, how="inner", predicate="within").drop_duplicates("id")
    truth = matched.groupby(["offense", "block_group_geoid"]).size().rename("incident_count").reset_index()
    truth.insert(0, "year", 2026)
    truth.insert(0, "jurisdiction_id", CHICAGO_JURISDICTION_ID)
    truth.insert(0, "source", SOURCE)
    audit = {
        "raw_unique_incidents": int(len(raw)), "mapped_incidents": int(len(mapped)),
        "plausibly_geocoded_incidents": int(len(geocoded)),
        "quarantined_incidents": int(quarantine.sum()), "matched_incidents": int(len(matched)),
        "offense_counts": {str(k): int(v) for k, v in matched["offense"].value_counts().items()},
    }
    return truth, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--categories-csv", type=Path, required=True)
    parser.add_argument("--il-bg-zip", type=Path, required=True)
    parser.add_argument("--bg-crosswalk", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=50000)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    if not args.acquire:
        raise SystemExit("Refusing network access without explicit --acquire")
    contract = validate_gate(args.contract, args.candidate_manifest)
    pages = acquire_pages(args.out_dir / "raw_pages", page_size=args.page_size)
    truth, audit = prepare_truth(
        [Path(str(row["path"])) for row in pages], categories_csv=args.categories_csv,
        il_bg_zip=args.il_bg_zip, bg_crosswalk_parquet=args.bg_crosswalk,
    )
    eligible_offenses = {str(value) for value in contract.get("eligible_offenses", [])}
    if not eligible_offenses:
        raise ValueError("Frozen contract must declare at least one eligible offense")
    excluded = truth[~truth["offense"].astype(str).isin(eligible_offenses)].copy()
    audit["evaluation_excluded_offense_counts"] = {
        str(k): int(v) for k, v in excluded.groupby("offense")["incident_count"].sum().items()
    }
    truth = truth[truth["offense"].astype(str).isin(eligible_offenses)].copy()
    truth_path = args.out_dir / "chicago_2026h1_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    manifest = {
        "version": "chicago_forward_2026h1_v1",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_start_inclusive": START, "target_end_exclusive": END_EXCLUSIVE,
        "evaluation_type": "conditional_spatial_distribution",
        "target_event_holdout": bool(contract.get("target_event_holdout")),
        "prospective_as_of_2025_12_31": bool(contract.get("prospective_as_of_2025_12_31")),
        "claim_label": str(contract.get("claim_label")),
        "same_city_historical_source_reuse": bool(contract.get("same_city_historical_source_reuse")),
        "same_city_historical_source_reuse_by_offense": contract.get("same_city_historical_source_reuse_by_offense", {}),
        "target_events_used_in_fit_or_selection": False,
        "contract": contract, "contract_path": str(args.contract.resolve()),
        "contract_sha256": sha256(args.contract),
        "candidate_manifest_path": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "pages": pages, "truth_path": str(truth_path.resolve()),
        "truth_sha256": sha256(truth_path), "retention_audit": audit,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
