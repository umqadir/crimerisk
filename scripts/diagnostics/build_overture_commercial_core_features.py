#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    return value


def _nearest_haversine_km(
    *,
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> np.ndarray:
    out = np.full(len(targets), np.nan, dtype=float)
    if targets.empty or sources.empty:
        return out
    source_coords = np.deg2rad(sources[[lat_col, lon_col]].to_numpy(dtype=float))
    target_coords = np.deg2rad(targets[[lat_col, lon_col]].to_numpy(dtype=float))
    valid_sources = np.isfinite(source_coords).all(axis=1)
    valid_targets = np.isfinite(target_coords).all(axis=1)
    if not valid_sources.any() or not valid_targets.any():
        return out
    tree = BallTree(source_coords[valid_sources], metric="haversine")
    dist_rad, _ = tree.query(target_coords[valid_targets], k=1)
    out[np.flatnonzero(valid_targets)] = dist_rad[:, 0] * 6371.0088
    return out


def _nearest_by_group(
    frame: pd.DataFrame,
    *,
    group_col: str,
    core_col: str,
) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(group_col, sort=False, dropna=False):
        core = group[group[core_col].astype(bool)].copy()
        if core.empty:
            continue
        values.loc[group.index] = _nearest_haversine_km(targets=group, sources=core)
    return values


def _core_thresholds(values: pd.Series, *, quantile: float, min_score: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    by_county = numeric.groupby(level=0).transform(lambda s: max(float(s.quantile(quantile)), float(min_score)))
    return by_county


def build_commercial_core_features(
    *,
    overture_features_path: Path,
    centroids_path: Path,
    county_quantile: float,
    state_quantile: float,
    min_core_count: float,
    min_core_score: float,
    within_km: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cols = [
        "bg_id",
        "tract_id",
        "state_fips",
        "county_fips",
        "overture_consumer_destination_any_count",
        "overture_consumer_destination_any_density_sqkm",
        "overture_food_service_count",
        "overture_nightlife_alcohol_count",
        "overture_lodging_count",
        "overture_parking_count",
        "overture_atm_bank_count",
        "overture_grocery_count",
    ]
    overture = pd.read_parquet(overture_features_path, columns=cols).copy()
    centroids = pd.read_parquet(
        centroids_path,
        columns=["bg_id", "state_fips", "county_fips", "tract_id", "lon", "lat"],
    ).copy()
    for df in (overture, centroids):
        df["bg_id"] = df["bg_id"].astype("string").str.zfill(12)
        df["tract_id"] = df["tract_id"].astype("string").str.zfill(11)
        df["state_fips"] = df["state_fips"].astype("string").str.zfill(2)
        df["county_fips"] = df["county_fips"].astype("string").str.zfill(3)
    frame = overture.merge(
        centroids[["bg_id", "lon", "lat"]],
        on="bg_id",
        how="left",
        validate="one_to_one",
    )
    frame["full_county_fips"] = frame["state_fips"].astype(str).str.zfill(2) + frame["county_fips"].astype(str).str.zfill(3)
    count = pd.to_numeric(frame["overture_consumer_destination_any_count"], errors="coerce").fillna(0.0)
    density = pd.to_numeric(frame["overture_consumer_destination_any_density_sqkm"], errors="coerce").fillna(0.0)
    frame["overture_commercial_core_score"] = np.log1p(count.clip(lower=0.0)) * np.log1p(density.clip(lower=0.0))

    county_indexed = frame.set_index("full_county_fips", drop=False)
    county_threshold = _core_thresholds(
        county_indexed["overture_commercial_core_score"],
        quantile=float(county_quantile),
        min_score=float(min_core_score),
    )
    county_max_score = frame.groupby("full_county_fips", dropna=False)["overture_commercial_core_score"].transform("max")
    county_has_any_destination = count.groupby(frame["full_county_fips"]).transform("sum").gt(0.0)
    county_threshold_core = (
        (frame["overture_commercial_core_score"].to_numpy(dtype=float) >= county_threshold.to_numpy(dtype=float))
        & count.ge(float(min_core_count)).to_numpy(dtype=bool)
    )
    county_best_available_core = (
        county_has_any_destination.to_numpy(dtype=bool)
        & count.gt(0.0).to_numpy(dtype=bool)
        & np.isclose(
            frame["overture_commercial_core_score"].to_numpy(dtype=float),
            county_max_score.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
    )
    frame["overture_county_commercial_core_flag"] = (
        county_threshold_core | county_best_available_core
    ).astype(float)

    state_threshold = (
        frame.groupby("state_fips", dropna=False)["overture_commercial_core_score"]
        .transform(lambda s: max(float(pd.to_numeric(s, errors="coerce").fillna(0.0).quantile(float(state_quantile))), float(min_core_score)))
    )
    frame["overture_state_commercial_core_flag"] = (
        (frame["overture_commercial_core_score"].to_numpy(dtype=float) >= state_threshold.to_numpy(dtype=float))
        & count.ge(float(min_core_count)).to_numpy(dtype=bool)
    ).astype(float)

    frame["nearest_overture_county_commercial_core_km"] = _nearest_by_group(
        frame,
        group_col="full_county_fips",
        core_col="overture_county_commercial_core_flag",
    )
    frame["nearest_overture_state_commercial_core_km"] = _nearest_by_group(
        frame,
        group_col="state_fips",
        core_col="overture_state_commercial_core_flag",
    )
    frame["nearest_overture_commercial_core_km"] = (
        pd.to_numeric(frame["nearest_overture_county_commercial_core_km"], errors="coerce")
        .fillna(pd.to_numeric(frame["nearest_overture_state_commercial_core_km"], errors="coerce"))
    )
    frame["log_overture_nearest_commercial_core_km"] = np.log1p(
        pd.to_numeric(frame["nearest_overture_commercial_core_km"], errors="coerce").clip(lower=0.0)
    )
    frame[f"overture_commercial_core_within_{int(within_km)}km"] = (
        pd.to_numeric(frame["nearest_overture_commercial_core_km"], errors="coerce").le(float(within_km)).fillna(False).astype(float)
    )
    component_cols = [
        "overture_food_service_count",
        "overture_nightlife_alcohol_count",
        "overture_lodging_count",
        "overture_parking_count",
        "overture_atm_bank_count",
        "overture_grocery_count",
    ]
    for col in component_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["overture_commercial_core_food_nightlife_lodging_share"] = np.where(
        count.gt(0.0),
        (
            frame["overture_food_service_count"]
            + frame["overture_nightlife_alcohol_count"]
            + frame["overture_lodging_count"]
        )
        / count.replace(0.0, np.nan),
        0.0,
    )
    frame["overture_commercial_core_finance_parking_share"] = np.where(
        count.gt(0.0),
        (frame["overture_parking_count"] + frame["overture_atm_bank_count"]) / count.replace(0.0, np.nan),
        0.0,
    )
    frame["overture_commercial_core_grocery_share"] = np.where(
        count.gt(0.0),
        frame["overture_grocery_count"] / count.replace(0.0, np.nan),
        0.0,
    )
    share_cols = [
        "overture_commercial_core_food_nightlife_lodging_share",
        "overture_commercial_core_finance_parking_share",
        "overture_commercial_core_grocery_share",
    ]
    for col in share_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

    feature_cols = [
        "bg_id",
        "tract_id",
        "state_fips",
        "county_fips",
        "overture_commercial_core_score",
        "overture_county_commercial_core_flag",
        "overture_state_commercial_core_flag",
        "nearest_overture_county_commercial_core_km",
        "nearest_overture_state_commercial_core_km",
        "nearest_overture_commercial_core_km",
        "log_overture_nearest_commercial_core_km",
        f"overture_commercial_core_within_{int(within_km)}km",
        *share_cols,
    ]
    out = frame[feature_cols].copy()
    summary = {
        "rows": int(len(out)),
        "county_core_rows": int(pd.to_numeric(out["overture_county_commercial_core_flag"], errors="coerce").fillna(0).sum()),
        "state_core_rows": int(pd.to_numeric(out["overture_state_commercial_core_flag"], errors="coerce").fillna(0).sum()),
        "county_core_counties": int(frame.loc[frame["overture_county_commercial_core_flag"].eq(1.0), "full_county_fips"].nunique()),
        "counties_with_any_destination": int(frame.loc[count.gt(0.0), "full_county_fips"].nunique()),
        "state_core_states": int(frame.loc[frame["overture_state_commercial_core_flag"].eq(1.0), "state_fips"].nunique()),
        "nearest_core_non_null_rows": int(pd.to_numeric(out["nearest_overture_commercial_core_km"], errors="coerce").notna().sum()),
        "median_nearest_core_km": float(pd.to_numeric(out["nearest_overture_commercial_core_km"], errors="coerce").median()),
        "p90_nearest_core_km": float(pd.to_numeric(out["nearest_overture_commercial_core_km"], errors="coerce").quantile(0.90)),
        "within_km": float(within_km),
        "within_km_rows": int(pd.to_numeric(out[f"overture_commercial_core_within_{int(within_km)}km"], errors="coerce").fillna(0).sum()),
        "county_quantile": float(county_quantile),
        "state_quantile": float(state_quantile),
        "min_core_count": float(min_core_count),
        "min_core_score": float(min_core_score),
    }
    return out.sort_values(["state_fips", "county_fips", "tract_id", "bg_id"], kind="stable").reset_index(drop=True), summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive commercial-core proximity features from Overture Places BG features."
    )
    parser.add_argument(
        "--overture-features-path",
        type=Path,
        default=REPO_ROOT / "data" / "Overture-Places" / "parsed" / "block_group_overture_places_states_latest.parquet",
    )
    parser.add_argument(
        "--centroids-path",
        type=Path,
        default=REPO_ROOT / "data" / "tiger_bg" / "parsed" / "bg_centroids.parquet",
    )
    parser.add_argument("--county-quantile", type=float, default=0.95)
    parser.add_argument("--state-quantile", type=float, default=0.995)
    parser.add_argument("--min-core-count", type=float, default=3.0)
    parser.add_argument("--min-core-score", type=float, default=0.25)
    parser.add_argument("--within-km", type=float, default=5.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "Overture-Places" / "parsed" / "block_group_overture_commercial_core_states_latest.parquet",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "overture_commercial_core_states_latest.json",
    )
    args = parser.parse_args()

    features, summary = build_commercial_core_features(
        overture_features_path=args.overture_features_path,
        centroids_path=args.centroids_path,
        county_quantile=float(args.county_quantile),
        state_quantile=float(args.state_quantile),
        min_core_count=float(args.min_core_count),
        min_core_score=float(args.min_core_score),
        within_km=float(args.within_km),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
