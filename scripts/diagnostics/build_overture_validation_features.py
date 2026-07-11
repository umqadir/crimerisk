from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crosswalk_shares import normalize_block_group_allocation_shares  # noqa: E402
from crimerisk.covariates.overture_places import (  # noqa: E402
    DEFAULT_OVERTURE_RELEASE,
    OverturePlacesBuildConfig,
    aggregate_overture_places_to_block_groups,
    fetch_overture_places_for_query_groups,
    resolve_overture_release_with_metadata,
)


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _validation_jurisdictions(
    *,
    city_shares_path: Path,
    year: int,
    include_case_types: tuple[str, ...],
    exclude_case_types: tuple[str, ...],
) -> pd.DataFrame:
    city = pd.read_parquet(city_shares_path).copy()
    city = city[pd.to_numeric(city["year"], errors="coerce").eq(int(year))].copy()
    if "validation_case_type" not in city.columns:
        city["validation_case_type"] = "unknown"
    if include_case_types:
        wanted = {str(value) for value in include_case_types}
        city = city[city["validation_case_type"].astype(str).isin(wanted)].copy()
    if exclude_case_types:
        excluded = {str(value) for value in exclude_case_types}
        city = city[~city["validation_case_type"].astype(str).isin(excluded)].copy()
    meta = (
        city[["city_name", "jurisdiction_id", "state_fips", "validation_case_type"]]
        .dropna(subset=["jurisdiction_id", "state_fips"])
        .drop_duplicates("jurisdiction_id")
        .copy()
    )
    meta["state_fips"] = meta["state_fips"].astype("string").str.zfill(2)
    meta["jurisdiction_id"] = meta["jurisdiction_id"].astype(str)
    return meta.sort_values(["state_fips", "city_name"], kind="mergesort").reset_index(drop=True)


def _load_validation_block_groups(
    *,
    city_shares_path: Path,
    bg_crosswalk_path: Path,
    tiger_bg_dir: Path,
    year: int,
    include_case_types: tuple[str, ...],
    exclude_case_types: tuple[str, ...],
) -> gpd.GeoDataFrame:
    meta = _validation_jurisdictions(
        city_shares_path=city_shares_path,
        year=year,
        include_case_types=include_case_types,
        exclude_case_types=exclude_case_types,
    )
    if meta.empty:
        return gpd.GeoDataFrame(
            columns=["query_group", "city_name", "jurisdiction_id", "state_fips", "bg_id", "tract_id", "county_fips", "geometry"],
            geometry="geometry",
            crs="EPSG:4269",
        )

    crosswalk = normalize_block_group_allocation_shares(pd.read_parquet(bg_crosswalk_path).copy())
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
    crosswalk["jurisdiction_id"] = crosswalk["jurisdiction_id"].astype(str)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].isin(set(meta["jurisdiction_id"]))].copy()
    crosswalk = crosswalk.merge(meta, on=["jurisdiction_id", "state_fips"], how="inner")
    if crosswalk.empty:
        return gpd.GeoDataFrame(
            columns=["query_group", "city_name", "jurisdiction_id", "state_fips", "bg_id", "tract_id", "county_fips", "geometry"],
            geometry="geometry",
            crs="EPSG:4269",
        )

    frames: list[gpd.GeoDataFrame] = []
    for state_fips, state_crosswalk in crosswalk.groupby("state_fips", sort=True):
        bg_zip = tiger_bg_dir / f"tl_2020_{str(state_fips).zfill(2)}_bg.zip"
        if not bg_zip.exists():
            raise FileNotFoundError(f"Missing TIGER block-group zip for state {state_fips}: {bg_zip}")
        bg = gpd.read_file(bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "bg_id"})
        bg["bg_id"] = bg["bg_id"].astype("string").str.zfill(12)
        bg["tract_id"] = bg["bg_id"].str.slice(0, 11)
        bg["state_fips"] = str(state_fips).zfill(2)
        bg["county_fips"] = bg["bg_id"].str.slice(0, 5)
        merged = bg.merge(
            state_crosswalk[
                [
                    "block_group_geoid",
                    "jurisdiction_id",
                    "city_name",
                    "validation_case_type",
                ]
            ].rename(columns={"block_group_geoid": "bg_id"}),
            on="bg_id",
            how="inner",
        )
        if merged.empty:
            continue
        merged["query_group"] = merged["jurisdiction_id"]
        frames.append(
            merged[
                [
                    "query_group",
                    "city_name",
                    "jurisdiction_id",
                    "validation_case_type",
                    "state_fips",
                    "bg_id",
                    "tract_id",
                    "county_fips",
                    "geometry",
                ]
            ].copy()
        )
    if not frames:
        return gpd.GeoDataFrame(
            columns=["query_group", "city_name", "jurisdiction_id", "state_fips", "bg_id", "tract_id", "county_fips", "geometry"],
            geometry="geometry",
            crs="EPSG:4269",
        )
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs)


def _collapse_points(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return points.copy()

    def _groups(values: pd.Series) -> str:
        groups: set[str] = set()
        for value in values:
            if isinstance(value, list):
                groups.update(str(item) for item in value if str(item))
        return "|".join(sorted(groups))

    return (
        points.groupby("overture_place_id", dropna=False, sort=False)
        .agg(
            place_name=("place_name", "first"),
            primary_category=("primary_category", "first"),
            confidence=("confidence", "max"),
            country=("country", "first"),
            region=("region", "first"),
            lon=("lon", "first"),
            lat=("lat", "first"),
            matched_groups=("matched_groups", _groups),
            overture_release=("overture_release", "first"),
            query_groups=("query_group", lambda values: "|".join(sorted({str(v) for v in values if pd.notna(v)}))),
            query_group_count=("query_group", lambda values: len({str(v) for v in values if pd.notna(v)})),
        )
        .reset_index()
    )


def build_overture_validation_features(
    *,
    city_shares_path: Path,
    bg_crosswalk_path: Path,
    tiger_bg_dir: Path,
    year: int,
    include_case_types: tuple[str, ...],
    exclude_case_types: tuple[str, ...],
    release: str,
    min_confidence: float,
    bbox_buffer_deg: float,
    within_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    block_groups = _load_validation_block_groups(
        city_shares_path=city_shares_path,
        bg_crosswalk_path=bg_crosswalk_path,
        tiger_bg_dir=tiger_bg_dir,
        year=year,
        include_case_types=include_case_types,
        exclude_case_types=exclude_case_types,
    )
    cfg = OverturePlacesBuildConfig(
        release=str(release),
        min_confidence=float(min_confidence),
        bbox_buffer_deg=float(bbox_buffer_deg),
        within_km=float(within_km),
    )
    points = fetch_overture_places_for_query_groups(
        block_groups=block_groups,
        query_group_col="query_group",
        cfg=cfg,
    )
    features = aggregate_overture_places_to_block_groups(
        points=points,
        block_groups=block_groups,
        query_group_col="query_group",
        cfg=cfg,
    )
    features = features.merge(
        block_groups[["query_group", "city_name", "jurisdiction_id", "validation_case_type"]].drop_duplicates("query_group"),
        on="query_group",
        how="left",
    )
    summary = {
        "year": int(year),
        "jurisdiction_count": int(block_groups["jurisdiction_id"].nunique()) if not block_groups.empty else 0,
        "block_group_rows": int(len(block_groups)),
        "feature_rows": int(len(features)),
        "raw_query_place_rows": int(len(points)),
        "audit_place_rows": int(points["overture_place_id"].nunique()) if not points.empty else 0,
        "include_case_types": list(include_case_types),
        "exclude_case_types": list(exclude_case_types),
        "release": str(release),
        "min_confidence": float(min_confidence),
        "bbox_buffer_deg": float(bbox_buffer_deg),
        "within_km": float(within_km),
        "jurisdictions": (
            block_groups[["city_name", "jurisdiction_id", "validation_case_type"]]
            .drop_duplicates()
            .sort_values(["validation_case_type", "city_name"], kind="mergesort")
            .to_dict(orient="records")
            if not block_groups.empty
            else []
        ),
    }
    return features, _collapse_points(points), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Overture Places features for validation city BG footprints.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--city-shares-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet",
    )
    parser.add_argument(
        "--bg-crosswalk-path",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument("--tiger-bg-dir", type=Path, default=REPO_ROOT / "data" / "tiger_bg")
    parser.add_argument(
        "--include-validation-case-type",
        action="append",
        default=[],
        help="Only include this validation_case_type; may be repeated. Defaults to all non-excluded case types.",
    )
    parser.add_argument(
        "--exclude-validation-case-type",
        action="append",
        default=["suburban_county_validation_case"],
        help="Exclude this validation_case_type; may be repeated.",
    )
    parser.add_argument("--release", default=DEFAULT_OVERTURE_RELEASE)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--bbox-buffer-deg", type=float, default=0.03)
    parser.add_argument("--within-km", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "overture_validation_city_bg_features_2024.parquet",
    )
    parser.add_argument(
        "--points-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "overture_validation_city_points_2024.parquet",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "overture_validation_city_features_2024.json",
    )
    args = parser.parse_args()

    # Resolve once upfront and record how it was resolved -- previously the summary JSON recorded
    # the raw --release request (often the literal "latest", or "None" if omitted after this
    # module's default became explicit-required) instead of the actual dated release pulled.
    resolved_release, release_provenance = resolve_overture_release_with_metadata(release=args.release)
    print(f"overture_release resolved: {release_provenance}", flush=True)

    features, points, summary = build_overture_validation_features(
        city_shares_path=args.city_shares_path,
        bg_crosswalk_path=args.bg_crosswalk_path,
        tiger_bg_dir=args.tiger_bg_dir,
        year=int(args.year),
        include_case_types=tuple(str(v) for v in args.include_validation_case_type),
        exclude_case_types=tuple(str(v) for v in args.exclude_validation_case_type),
        release=resolved_release,
        min_confidence=float(args.min_confidence),
        bbox_buffer_deg=float(args.bbox_buffer_deg),
        within_km=float(args.within_km),
    )
    summary["release_provenance"] = release_provenance
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.points_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    points.to_parquet(args.points_out, index=False)
    args.summary_json_out.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
