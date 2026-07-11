#!/usr/bin/env python3
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

from crimerisk.covariates.overture_places import (  # noqa: E402
    DEFAULT_OVERTURE_RELEASE,
    OverturePlacesBuildConfig,
    aggregate_overture_places_to_block_groups,
    fetch_overture_places_for_query_groups,
    resolve_overture_release_with_metadata,
)
from crimerisk.covariates.roads import infer_state_fips_from_bg_dir, load_block_groups  # noqa: E402


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


def _load_benchmark_block_groups(
    *,
    city_shares_path: Path,
    bg_dir: Path,
    year: int,
    cities: list[str] | None,
) -> gpd.GeoDataFrame:
    city = pd.read_parquet(
        city_shares_path,
        columns=["year", "city_name", "state_fips", "block_group_geoid"],
    )
    city["year"] = pd.to_numeric(city["year"], errors="coerce")
    city["state_fips"] = city["state_fips"].astype("string").str.zfill(2)
    city["bg_id"] = city["block_group_geoid"].astype("string").str.zfill(12)
    city = city[city["year"].eq(int(year))].copy()
    if cities:
        wanted = {str(city_name).strip().lower() for city_name in cities}
        city = city[city["city_name"].astype(str).str.lower().isin(wanted)].copy()
    city = city[["city_name", "state_fips", "bg_id"]].drop_duplicates().reset_index(drop=True)
    if city.empty:
        return gpd.GeoDataFrame(
            columns=["city_name", "bg_id", "tract_id", "state_fips", "county_fips", "geometry"],
            geometry="geometry",
            crs="EPSG:4269",
        )
    collisions = city.groupby("bg_id", dropna=False)["city_name"].nunique()
    bad_ids = collisions[collisions.gt(1)].index.tolist()
    if bad_ids:
        raise ValueError(f"Benchmark city BG surface has bg_id values assigned to multiple cities: {bad_ids[:10]}")

    frames: list[gpd.GeoDataFrame] = []
    for state_fips, state_df in city.groupby("state_fips", sort=True):
        bg_zip = bg_dir / f"tl_2020_{str(state_fips).zfill(2)}_bg.zip"
        if not bg_zip.exists():
            raise FileNotFoundError(f"Missing TIGER BG zip for state {state_fips}: {bg_zip}")
        bg = load_block_groups(bg_zip)
        bg = bg[bg["bg_id"].isin(state_df["bg_id"])].copy()
        if bg.empty:
            continue
        bg = bg.merge(
            state_df[["bg_id", "city_name"]].drop_duplicates("bg_id"),
            on="bg_id",
            how="left",
        )
        frames.append(bg[["city_name", "bg_id", "tract_id", "state_fips", "county_fips", "geometry"]].copy())
    if not frames:
        return gpd.GeoDataFrame(
            columns=["city_name", "bg_id", "tract_id", "state_fips", "county_fips", "geometry"],
            geometry="geometry",
            crs="EPSG:4269",
        )
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs)


def _load_state_block_groups(
    *,
    bg_dir: Path,
    states: list[str] | None,
) -> gpd.GeoDataFrame:
    state_values = [str(state).zfill(2) for state in states] if states else infer_state_fips_from_bg_dir(bg_dir, year=2020)
    frames: list[gpd.GeoDataFrame] = []
    for state_fips in state_values:
        bg_zip = bg_dir / f"tl_2020_{str(state_fips).zfill(2)}_bg.zip"
        if not bg_zip.exists():
            raise FileNotFoundError(f"Missing TIGER BG zip for state {state_fips}: {bg_zip}")
        bg = load_block_groups(bg_zip)
        if bg.empty:
            continue
        bg["query_group"] = str(state_fips).zfill(2)
        frames.append(bg[["query_group", "bg_id", "tract_id", "state_fips", "county_fips", "geometry"]].copy())
    if not frames:
        return gpd.GeoDataFrame(
            columns=["query_group", "bg_id", "tract_id", "state_fips", "county_fips", "geometry"],
            geometry="geometry",
            crs="EPSG:4269",
        )
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs)


def _collapse_point_audit_frame(
    *,
    points: pd.DataFrame,
    query_group_col: str,
) -> pd.DataFrame:
    if points.empty:
        return points.copy()

    def _unique_sorted_text(values: pd.Series) -> str:
        cleaned = sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip()})
        return "|".join(cleaned)

    def _unique_sorted_group_values(values: pd.Series) -> list[str]:
        unique_groups: set[str] = set()
        for value in values:
            if isinstance(value, list):
                unique_groups.update(str(item).strip() for item in value if str(item).strip())
            elif pd.notna(value) and str(value).strip():
                unique_groups.add(str(value).strip())
        return sorted(unique_groups)

    agg_kwargs = dict(
        place_name=("place_name", "first"),
        primary_category=("primary_category", "first"),
        confidence=("confidence", "max"),
        country=("country", "first"),
        region=("region", "first"),
        lon=("lon", "first"),
        lat=("lat", "first"),
        overture_release=("overture_release", "first"),
        matched_groups=("matched_groups", _unique_sorted_group_values),
        query_groups=(query_group_col, _unique_sorted_text),
        query_group_count=(query_group_col, lambda values: len({str(value).strip() for value in values if pd.notna(value) and str(value).strip()})),
    )
    # Taxonomy migration (audit-only columns): carry the new-vocabulary fields through the
    # collapsed audit frame when present so the classification diff can be run directly off this
    # parquet without a second pull. Absent for releases that predate basic_category/taxonomy
    # (columns won't exist on `points` in that case).
    if "basic_category" in points.columns:
        agg_kwargs["basic_category"] = ("basic_category", "first")
    if "taxonomy_primary" in points.columns:
        agg_kwargs["taxonomy_primary"] = ("taxonomy_primary", "first")
    if "matched_groups_taxonomy" in points.columns:
        agg_kwargs["matched_groups_taxonomy"] = ("matched_groups_taxonomy", _unique_sorted_group_values)
    # Release-bump step: for releases >= 2026-06-17.0, `matched_groups` (production) is now
    # taxonomy-keyed; `matched_groups_legacy` carries the parallel legacy-classifier provenance
    # for the same (taxonomy-admitted) rows. Present as an all-null column for releases before the
    # threshold, where the active path already is the legacy path.
    if "matched_groups_legacy" in points.columns:
        agg_kwargs["matched_groups_legacy"] = ("matched_groups_legacy", _unique_sorted_group_values)

    grouped = points.groupby("overture_place_id", dropna=False, sort=False).agg(**agg_kwargs).reset_index()
    grouped["matched_group_count"] = grouped["matched_groups"].map(len).astype(int)
    if "matched_groups_taxonomy" in grouped.columns:
        grouped["matched_group_count_taxonomy"] = grouped["matched_groups_taxonomy"].map(len).astype(int)
    if "matched_groups_legacy" in grouped.columns:
        grouped["matched_group_count_legacy"] = grouped["matched_groups_legacy"].map(len).astype(int)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build bounded Overture Places BG features for benchmark-city or state-scoped block-group surfaces."
    )
    parser.add_argument("--scope", choices=["benchmark-cities", "states"], default="benchmark-cities")
    parser.add_argument(
        "--city-shares-path",
        type=Path,
        default=REPO_ROOT / "state" / "artifacts" / "v2" / "modeling" / "city_incident_share_surface.parquet",
    )
    parser.add_argument(
        "--bg-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tiger_bg",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--release", type=str, default=DEFAULT_OVERTURE_RELEASE)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--bbox-buffer-deg", type=float, default=0.03)
    parser.add_argument("--within-km", type=float, default=1.0)
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--points-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.scope == "benchmark-cities":
        block_groups = _load_benchmark_block_groups(
            city_shares_path=args.city_shares_path,
            bg_dir=args.bg_dir,
            year=int(args.year),
            cities=list(args.cities) if args.cities else None,
        )
        input_bg_row_count = int(len(block_groups))
        query_group_col = "city_name"
        default_points_out = REPO_ROOT / "data" / "Overture-Places" / "parsed" / "overture_places_benchmark_city_points_2024.parquet"
        default_out = REPO_ROOT / "data" / "Overture-Places" / "parsed" / "block_group_overture_places_benchmark_cities_2024.parquet"
    else:
        block_groups = None
        input_bg_row_count = 0
        query_group_col = "query_group"
        default_points_out = REPO_ROOT / "data" / "Overture-Places" / "parsed" / "overture_places_state_points_latest.parquet"
        default_out = REPO_ROOT / "data" / "Overture-Places" / "parsed" / "block_group_overture_places_states_latest.parquet"
    points_out = args.points_out or default_points_out
    out_path = args.out or default_out
    summary_json_out = args.summary_json_out

    # Resolve the release once, upfront, and record how it was resolved. Previously the summary
    # JSON recorded the raw --release request (often the literal "latest") instead of the actual
    # dated release pulled, so a run's provenance could not be reconstructed after the fact.
    resolved_release, release_provenance = resolve_overture_release_with_metadata(release=args.release)
    print(f"overture_release resolved: {release_provenance}", flush=True)

    cfg = OverturePlacesBuildConfig(
        release=resolved_release,
        min_confidence=float(args.min_confidence),
        bbox_buffer_deg=float(args.bbox_buffer_deg),
        within_km=float(args.within_km),
    )
    if args.scope == "benchmark-cities":
        points = fetch_overture_places_for_query_groups(
            block_groups=block_groups,
            query_group_col=query_group_col,
            cfg=cfg,
        )
        features = aggregate_overture_places_to_block_groups(
            points=points,
            block_groups=block_groups,
            query_group_col=query_group_col,
            cfg=cfg,
        )
    else:
        state_values = [str(state).zfill(2) for state in args.states] if args.states else infer_state_fips_from_bg_dir(args.bg_dir, year=2020)
        point_frames: list[pd.DataFrame] = []
        feature_frames: list[pd.DataFrame] = []
        total_bg_rows = 0
        for state_fips in state_values:
            state_bg = _load_state_block_groups(bg_dir=args.bg_dir, states=[state_fips])
            total_bg_rows += int(len(state_bg))
            state_points = fetch_overture_places_for_query_groups(
                block_groups=state_bg,
                query_group_col=query_group_col,
                cfg=cfg,
            )
            state_features = aggregate_overture_places_to_block_groups(
                points=state_points,
                block_groups=state_bg,
                query_group_col=query_group_col,
                cfg=cfg,
            )
            point_frames.append(state_points)
            feature_frames.append(state_features)
            print(
                f"state={str(state_fips).zfill(2)} bg_rows={len(state_bg):,} "
                f"query_match_rows={len(state_points):,} feature_rows={len(state_features):,}",
                flush=True,
            )
        points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
        features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
        input_bg_row_count = total_bg_rows

    audit_points = _collapse_point_audit_frame(points=points, query_group_col=query_group_col)
    if not audit_points.empty:
        audit_points = audit_points.copy()
        audit_points["matched_groups"] = audit_points["matched_groups"].map(
            lambda groups: "|".join(sorted(str(group) for group in groups)) if isinstance(groups, list) else ""
        )
    points_out.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_points.to_parquet(points_out, index=False)
    features.to_parquet(out_path, index=False)
    summary = {
        "scope": str(args.scope),
        "year": int(args.year),
        "release": resolved_release,
        "release_provenance": release_provenance,
        "min_confidence": float(args.min_confidence),
        "bbox_buffer_deg": float(args.bbox_buffer_deg),
        "within_km": float(args.within_km),
        "state_count": int(features["state_fips"].nunique()) if "state_fips" in features.columns and not features.empty else 0,
        "input_bg_rows": int(input_bg_row_count),
        "feature_rows": int(len(features)),
        "audit_place_rows": int(len(audit_points)),
        "feature_column_count": int(len([c for c in features.columns if c not in {"query_group", "bg_id", "tract_id", "state_fips", "county_fips"}])),
        "points_out": str(points_out),
        "out": str(out_path),
    }
    if summary_json_out is not None:
        summary_json_out.parent.mkdir(parents=True, exist_ok=True)
        summary_json_out.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    print(f"input_bg_rows={input_bg_row_count:,}")
    print(f"audit_place_rows={len(audit_points):,}")
    print(f"feature_rows={len(features):,}")
    print(f"points_out={points_out}")
    print(f"out={out_path}")
    if summary_json_out is not None:
        print(f"summary_json_out={summary_json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
