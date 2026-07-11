#!/usr/bin/env python3
"""
Aggregate LandScan USA day/night rasters to release block groups and run the
offline denominator pilot requested in Q1.

This is intentionally a sidecar diagnostic: it does not change estimator inputs,
release outputs, or promotion artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from shapely.geometry import Point, box


REPO_ROOT = Path(__file__).resolve().parents[2]

LANDSCAN_DIR = REPO_ROOT / "data" / "LandScan-USA"
RASTER_DAY = LANDSCAN_DIR / "rasters" / "landscan-usa-2021-conus-day.tif"
RASTER_NIGHT = LANDSCAN_DIR / "rasters" / "landscan-usa-2021-conus-night.tif"
LANDSCAN_BG_OUT = LANDSCAN_DIR / "block_group_landscan_usa_2021.parquet"
LANDSCAN_STATE_OUT = LANDSCAN_DIR / "state_block_groups"
BG_DIR = REPO_ROOT / "data" / "tiger_bg"
RELEASE_BG = REPO_ROOT / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet"
CENSUS_POPEST = REPO_ROOT / "data" / "Census-PopEst-2020-2025" / "co-est2025-alldata.csv"
TRUTH_SURFACE = REPO_ROOT / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet"
REPORT_JSON = REPO_ROOT / "state" / "modeling" / "landscan_pilot_report.json"
REPORT_MD = REPO_ROOT / "state" / "modeling" / "landscan_pilot_report.md"
PATHOLOGICAL_CSV = REPO_ROOT / "state" / "modeling" / "landscan_pathological_cells.csv"

RATE_PER_100K = 100_000.0
PERSON_FLOOR = 50.0
OFFENSES_7 = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]
PERSON_EXPOSURE_OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "larceny",
]
STATIC_PRIMARY_OFFENSES = ["burglary", "motor_vehicle_theft"]

SPOT_BG_IDS = {
    "times_square": {
        "label": "Times Square",
        "block_group_geoid": "360610113001",
        "note": "User-specified Manhattan BG.",
    },
    "mount_rainier": {
        "label": "Mount Rainier",
        "block_group_geoid": "530530701002",
        "note": "User-specified Mount Rainier BG.",
    },
    "south_bay_boston": {
        "label": "South Bay Boston",
        "block_group_geoid": "250250921014",
        "note": "User-specified Boston South Bay BG.",
    },
}

SPOT_POINTS = {
    "lv_strip": [
        ("Las Vegas Strip - Bellagio", -115.1769, 36.1126),
        ("Las Vegas Strip - Caesars", -115.1745, 36.1162),
        ("Las Vegas Strip - Venetian", -115.1706, 36.1212),
        ("Las Vegas Strip - MGM Grand", -115.1695, 36.1024),
        ("Las Vegas Strip - Resorts World", -115.1686, 36.1336),
    ],
    "easton_columbus": [
        ("Easton Town Center, Columbus OH", -82.915, 40.050),
    ],
    "angeles_nf_foothills": [
        ("Angeles NF foothills - La Canada", -118.200, 34.214),
        ("Angeles NF - Mount Wilson", -118.0645, 34.2244),
        ("Angeles NF - San Gabriel Canyon", -117.842, 34.237),
        ("Angeles NF - Chantry Flat", -118.022, 34.195),
    ],
}


@dataclass(frozen=True)
class RasterSummary:
    sum: float
    min: float
    max: float
    valid_pixels: int
    width: int
    height: int
    crs: str
    resolution_degrees: tuple[float, float]
    bounds: tuple[float, float, float, float]
    dtype: str
    nodata: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "valid_pixels": self.valid_pixels,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "resolution_degrees": list(self.resolution_degrees),
            "bounds": list(self.bounds),
            "dtype": self.dtype,
            "nodata": self.nodata,
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _read_release_columns(extra: list[str] | None = None) -> pd.DataFrame:
    base = [
        "block_group_geoid",
        "state_fips",
        "population_2024",
        "daytime_population_jobs_proxy",
        "exposure_proxy_2024",
        "households_total",
        "commercial_premises_total",
        "non_residential_flag",
        "special_use_tract_flag",
        "expected_count_total",
        "index_total_primary_event_weighted",
    ]
    for offense in OFFENSES_7:
        base.extend(
            [
                f"expected_count_{offense}",
                f"primary_denominator_{offense}",
                f"primary_national_rate_per_100k_{offense}",
                f"index_{offense}_primary",
                f"estimate_mode_{offense}",
                f"primary_denominator_invalid_{offense}",
            ]
        )
    columns = list(dict.fromkeys([*base, *(extra or [])]))
    available = pd.read_parquet(RELEASE_BG, columns=None).columns
    columns = [col for col in columns if col in available]
    out = pd.read_parquet(RELEASE_BG, columns=columns)
    out["block_group_geoid"] = out["block_group_geoid"].astype(str).str.zfill(12)
    out["state_fips"] = out["state_fips"].astype(str).str.zfill(2)
    return out


def _state_bg_zip(state_fips: str) -> Path:
    candidates = [
        BG_DIR / f"tl_2020_{state_fips}_bg.zip",
        BG_DIR / f"tl_2023_{state_fips}_bg.zip",
        BG_DIR / f"tl_2024_{state_fips}_bg.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No TIGER block-group zip for state {state_fips}: {candidates}")


def _read_state_block_groups(state_fips: str, active_ids: set[str], target_crs: Any) -> gpd.GeoDataFrame:
    zip_path = _state_bg_zip(state_fips)
    gdf = gpd.read_file(f"zip://{zip_path}")
    geoid_col = "GEOID" if "GEOID" in gdf.columns else "GEOID20"
    gdf = gdf[[geoid_col, "geometry"]].rename(columns={geoid_col: "block_group_geoid"})
    gdf["block_group_geoid"] = gdf["block_group_geoid"].astype(str).str.zfill(12)
    gdf = gdf[gdf["block_group_geoid"].isin(active_ids)].copy()
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4269")
    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf.reset_index(drop=True)


def _window_for_bounds(bounds: tuple[float, float, float, float], src: rasterio.DatasetReader) -> Window:
    raw = from_bounds(*bounds, transform=src.transform)
    col_start = max(0, math.floor(raw.col_off))
    row_start = max(0, math.floor(raw.row_off))
    col_stop = min(src.width, math.ceil(raw.col_off + raw.width))
    row_stop = min(src.height, math.ceil(raw.row_off + raw.height))
    return Window(col_start, row_start, max(0, col_stop - col_start), max(0, row_stop - row_start))


def _iter_tiles(window: Window, tile_size: int) -> list[Window]:
    tiles: list[Window] = []
    row_end = int(window.row_off + window.height)
    col_end = int(window.col_off + window.width)
    for row in range(int(window.row_off), row_end, tile_size):
        for col in range(int(window.col_off), col_end, tile_size):
            tiles.append(
                Window(
                    col,
                    row,
                    min(tile_size, col_end - col),
                    min(tile_size, row_end - row),
                )
            )
    return tiles


def _raster_summary(path: Path) -> RasterSummary:
    with rasterio.open(path) as src:
        total = 0.0
        min_value = math.inf
        max_value = -math.inf
        valid_pixels = 0
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True)
            if arr.count() == 0:
                continue
            data = arr.compressed().astype(np.float64)
            total += float(data.sum())
            min_value = min(min_value, float(data.min()))
            max_value = max(max_value, float(data.max()))
            valid_pixels += int(data.size)
        return RasterSummary(
            sum=total,
            min=min_value,
            max=max_value,
            valid_pixels=valid_pixels,
            width=int(src.width),
            height=int(src.height),
            crs=str(src.crs),
            resolution_degrees=(float(src.res[0]), float(src.res[1])),
            bounds=(float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)),
            dtype=str(src.dtypes[0]),
            nodata=_to_float(src.nodata),
        )


def aggregate_state(
    state_fips: str,
    active_ids: set[str],
    *,
    tile_size: int,
    force: bool,
) -> dict[str, Any]:
    LANDSCAN_STATE_OUT.mkdir(parents=True, exist_ok=True)
    out_path = LANDSCAN_STATE_OUT / f"landscan_usa_2021_bg_{state_fips}.parquet"
    if out_path.exists() and not force:
        shard = pd.read_parquet(out_path)
        return {
            "state_fips": state_fips,
            "rows": int(len(shard)),
            "landscan_day_pop": float(shard["landscan_day_pop"].sum()),
            "landscan_night_pop": float(shard["landscan_night_pop"].sum()),
            "out": str(out_path.relative_to(REPO_ROOT)),
            "cached": True,
        }

    with rasterio.open(RASTER_DAY) as day_src, rasterio.open(RASTER_NIGHT) as night_src:
        if day_src.crs != night_src.crs or day_src.transform != night_src.transform or day_src.shape != night_src.shape:
            raise ValueError("Day/night LandScan rasters are not aligned.")
        gdf = _read_state_block_groups(state_fips, active_ids, day_src.crs)
        if gdf.empty:
            empty = pd.DataFrame(
                columns=["block_group_geoid", "landscan_day_pop", "landscan_night_pop"]
            )
            empty.to_parquet(out_path, index=False)
            return {
                "state_fips": state_fips,
                "rows": 0,
                "landscan_day_pop": 0.0,
                "landscan_night_pop": 0.0,
                "out": str(out_path.relative_to(REPO_ROOT)),
                "cached": False,
            }

        state_window = _window_for_bounds(tuple(gdf.total_bounds), day_src)
        day_sums = np.zeros(len(gdf), dtype=np.float64)
        night_sums = np.zeros(len(gdf), dtype=np.float64)
        geometries = gdf.geometry.to_numpy()
        spatial_index = gdf.sindex

        for tile in _iter_tiles(state_window, tile_size):
            if tile.width <= 0 or tile.height <= 0:
                continue
            tile_bounds = window_bounds(tile, day_src.transform)
            tile_geom = box(*tile_bounds)
            try:
                hits = list(spatial_index.query(tile_geom, predicate="intersects"))
            except TypeError:
                hits = list(spatial_index.query(tile_geom))
            if not hits:
                continue

            shapes = ((geometries[i], int(i) + 1) for i in hits if not geometries[i].is_empty)
            labels = features.rasterize(
                shapes,
                out_shape=(int(tile.height), int(tile.width)),
                transform=day_src.window_transform(tile),
                fill=0,
                dtype="int32",
                all_touched=False,
            )
            if int(labels.max(initial=0)) == 0:
                continue

            day_arr = day_src.read(1, window=tile, masked=False)
            night_arr = night_src.read(1, window=tile, masked=False)
            label_flat = labels.ravel()
            valid = label_flat > 0
            if day_src.nodata is not None:
                valid &= day_arr.ravel() != day_src.nodata
            if night_src.nodata is not None:
                valid &= night_arr.ravel() != night_src.nodata
            if not np.any(valid):
                continue
            label_zero = label_flat[valid] - 1
            day_sums += np.bincount(
                label_zero,
                weights=day_arr.ravel()[valid].astype(np.float64),
                minlength=len(gdf),
            )
            night_sums += np.bincount(
                label_zero,
                weights=night_arr.ravel()[valid].astype(np.float64),
                minlength=len(gdf),
            )

    shard = pd.DataFrame(
        {
            "block_group_geoid": gdf["block_group_geoid"].to_numpy(),
            "landscan_day_pop": day_sums,
            "landscan_night_pop": night_sums,
        }
    ).sort_values("block_group_geoid", kind="mergesort")
    shard.to_parquet(out_path, index=False)
    return {
        "state_fips": state_fips,
        "rows": int(len(shard)),
        "landscan_day_pop": float(shard["landscan_day_pop"].sum()),
        "landscan_night_pop": float(shard["landscan_night_pop"].sum()),
        "out": str(out_path.relative_to(REPO_ROOT)),
        "cached": False,
    }


def aggregate_landscan(*, tile_size: int, force: bool) -> dict[str, Any]:
    release = _read_release_columns(extra=[])
    active = release[["block_group_geoid", "state_fips"]].drop_duplicates()
    state_ids = {
        state_fips: set(group["block_group_geoid"].astype(str))
        for state_fips, group in active.groupby("state_fips", sort=True)
    }

    state_summaries: list[dict[str, Any]] = []
    for state_fips in sorted(state_ids):
        print(f"aggregating_state={state_fips}", flush=True)
        state_summaries.append(
            aggregate_state(state_fips, state_ids[state_fips], tile_size=tile_size, force=force)
        )

    frames = [pd.read_parquet(LANDSCAN_STATE_OUT / f"landscan_usa_2021_bg_{state}.parquet") for state in sorted(state_ids)]
    combined = pd.concat(frames, ignore_index=True)
    combined["block_group_geoid"] = combined["block_group_geoid"].astype(str).str.zfill(12)
    combined = active[["block_group_geoid"]].merge(combined, on="block_group_geoid", how="left")
    combined[["landscan_day_pop", "landscan_night_pop"]] = combined[
        ["landscan_day_pop", "landscan_night_pop"]
    ].fillna(0.0)
    combined = combined.sort_values("block_group_geoid", kind="mergesort")
    LANDSCAN_BG_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LANDSCAN_BG_OUT, index=False)
    return {
        "out": str(LANDSCAN_BG_OUT.relative_to(REPO_ROOT)),
        "rows": int(len(combined)),
        "landscan_day_pop": float(combined["landscan_day_pop"].sum()),
        "landscan_night_pop": float(combined["landscan_night_pop"].sum()),
        "state_summaries": state_summaries,
    }


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _event_weights(df: pd.DataFrame) -> dict[str, float]:
    totals = {
        offense: float(_numeric(df, f"expected_count_{offense}").clip(lower=0.0).sum())
        for offense in OFFENSES_7
    }
    total = float(sum(totals.values()))
    if total <= 0.0:
        return {offense: float("nan") for offense in OFFENSES_7}
    return {offense: totals[offense] / total for offense in OFFENSES_7}


def _candidate_publishable(df: pd.DataFrame, offense: str, denominator: pd.Series) -> pd.Series:
    if offense in PERSON_EXPOSURE_OFFENSES:
        mode = df[f"estimate_mode_{offense}"].astype("string")
        base_ok = mode.isin(["count_derived", "insufficient_exposure"])
        return base_ok & denominator.gt(0.0) & denominator.ge(PERSON_FLOOR)
    current_index = pd.to_numeric(df[f"index_{offense}_primary"], errors="coerce")
    return current_index.notna()


def _compute_candidate_indexes(df: pd.DataFrame, denominator_col: str, suffix: str) -> dict[str, Any]:
    denom_candidate = _numeric(df, denominator_col)
    weights = _event_weights(df)
    component_values: dict[str, pd.Series] = {}
    offense_summary: dict[str, Any] = {}

    for offense in OFFENSES_7:
        if offense in PERSON_EXPOSURE_OFFENSES:
            denom = denom_candidate
            publishable = _candidate_publishable(df, offense, denom)
            counts = _numeric(df, f"expected_count_{offense}").clip(lower=0.0)
            denom_sum = float(denom.loc[publishable].sum())
            count_sum = float(counts.loc[publishable].sum())
            national_rate = RATE_PER_100K * count_sum / denom_sum if denom_sum > 0.0 else float("nan")
            values = pd.Series(np.nan, index=df.index, dtype=float)
            if np.isfinite(national_rate) and national_rate > 0.0:
                values.loc[publishable] = 100.0 * (RATE_PER_100K * counts.loc[publishable] / denom.loc[publishable]) / national_rate
            component_values[offense] = values.replace([np.inf, -np.inf], np.nan)
            offense_summary[offense] = {
                "candidate_national_rate_per_100k": _to_float(national_rate),
                "published_rows": int(publishable.sum()),
                "denominator_total": _to_float(denom_sum),
                "expected_count_total": _to_float(count_sum),
            }
        else:
            values = pd.to_numeric(df[f"index_{offense}_primary"], errors="coerce")
            component_values[offense] = values
            offense_summary[offense] = {
                "candidate_national_rate_per_100k": _to_float(df[f"primary_national_rate_per_100k_{offense}"].dropna().iloc[0])
                if f"primary_national_rate_per_100k_{offense}" in df.columns
                and df[f"primary_national_rate_per_100k_{offense}"].notna().any()
                else None,
                "published_rows": int(values.notna().sum()),
                "denominator_total": _to_float(_numeric(df, f"primary_denominator_{offense}").loc[values.notna()].sum()),
                "expected_count_total": _to_float(_numeric(df, f"expected_count_{offense}").loc[values.notna()].sum()),
            }

    matrix = np.vstack([component_values[offense].to_numpy(dtype=float) for offense in OFFENSES_7])
    weight_values = np.array([float(weights[offense]) for offense in OFFENSES_7], dtype=float)
    valid_rows = np.isfinite(matrix).all(axis=0)
    valid_weights = np.isfinite(weight_values) & (weight_values > 0.0)
    composite = np.full(len(df), np.nan, dtype=float)
    weight_sum = float(weight_values[valid_weights].sum())
    if weight_sum > 0.0:
        composite[valid_rows] = np.dot(weight_values, matrix[:, valid_rows]) / weight_sum

    return {
        "suffix": suffix,
        "composite": pd.Series(composite, index=df.index, dtype=float).replace([np.inf, -np.inf], np.nan),
        "components": component_values,
        "weights": weights,
        "offense_summary": offense_summary,
    }


def _tail_stats(series: pd.Series, *, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "published_rows": 0,
            "p99": None,
            "p999": None,
            "max": None,
            "mean": None,
            "rows_at_or_above_current_p99": None,
            "rows_at_or_above_current_p999": None,
        }
    p99 = float(clean.quantile(0.99))
    p999 = float(clean.quantile(0.999))
    out = {
        "published_rows": int(clean.size),
        "p99": p99,
        "p999": p999,
        "max": float(clean.max()),
        "mean": float(clean.mean()),
    }
    if thresholds:
        for name, threshold in thresholds.items():
            out[f"rows_at_or_above_{name}"] = int(clean.ge(float(threshold)).sum())
    return out


def _census_sanity(df: pd.DataFrame, land: pd.DataFrame) -> dict[str, Any]:
    states = sorted(df["state_fips"].dropna().astype(str).str.zfill(2).unique())
    census = pd.read_csv(CENSUS_POPEST, encoding="latin1")
    census["state_fips"] = census["STATE"].astype(int).astype(str).str.zfill(2)
    state_rows = census[(census["SUMLEV"] == 40) & census["state_fips"].isin(states)]
    pop2020 = float(pd.to_numeric(state_rows["POPESTIMATE2020"], errors="coerce").sum())
    pop2024 = float(pd.to_numeric(state_rows["POPESTIMATE2024"], errors="coerce").sum())
    day_total = float(land["landscan_day_pop"].sum())
    night_total = float(land["landscan_night_pop"].sum())
    output_pop2024 = float(_numeric(df, "population_2024").sum())
    return {
        "scope": "Release block groups, CONUS plus DC; Alaska, Hawaii, and Puerto Rico excluded because the release output excludes them.",
        "release_states": states,
        "census_popestimate2020_scope_total": pop2020,
        "census_popestimate2024_scope_total": pop2024,
        "release_population_2024_sum": output_pop2024,
        "landscan_day_pop_sum": day_total,
        "landscan_night_pop_sum": night_total,
        "day_to_census2020_ratio": day_total / pop2020 if pop2020 else None,
        "night_to_census2020_ratio": night_total / pop2020 if pop2020 else None,
        "day_to_census2024_ratio": day_total / pop2024 if pop2024 else None,
        "night_to_census2024_ratio": night_total / pop2024 if pop2024 else None,
        "day_to_release_population_2024_ratio": day_total / output_pop2024 if output_pop2024 else None,
        "night_to_release_population_2024_ratio": night_total / output_pop2024 if output_pop2024 else None,
    }


def _point_to_bg(label: str, lon: float, lat: float, state_fips: str) -> dict[str, Any] | None:
    zip_path = _state_bg_zip(state_fips)
    gdf = gpd.read_file(f"zip://{zip_path}")
    geoid_col = "GEOID" if "GEOID" in gdf.columns else "GEOID20"
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4269")
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
    matches = gdf[gdf.geometry.contains(point)]
    if matches.empty:
        matches = gdf[gdf.geometry.intersects(point)]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "label": label,
        "block_group_geoid": str(row[geoid_col]).zfill(12),
        "source_lon": lon,
        "source_lat": lat,
    }


def _spot_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, spec in SPOT_BG_IDS.items():
        rows.append(
            {
                "spot_key": key,
                "spot_group": key,
                "spot_label": spec["label"],
                "block_group_geoid": spec["block_group_geoid"],
                "selection_note": spec["note"],
            }
        )
    state_for_group = {
        "lv_strip": "32",
        "easton_columbus": "39",
        "angeles_nf_foothills": "06",
    }
    seen: set[tuple[str, str]] = set()
    for group, points in SPOT_POINTS.items():
        for label, lon, lat in points:
            match = _point_to_bg(label, lon, lat, state_for_group[group])
            if not match:
                rows.append(
                    {
                        "spot_key": label,
                        "spot_group": group,
                        "spot_label": label,
                        "block_group_geoid": None,
                        "selection_note": f"No BG found for lon={lon}, lat={lat}.",
                    }
                )
                continue
            dedupe = (group, match["block_group_geoid"])
            if dedupe in seen:
                continue
            seen.add(dedupe)
            rows.append(
                {
                    "spot_key": label,
                    "spot_group": group,
                    "spot_label": label,
                    "block_group_geoid": match["block_group_geoid"],
                    "selection_note": f"Point lookup lon={lon}, lat={lat}.",
                }
            )

    spots = pd.DataFrame(rows)
    merged = spots.merge(df, on="block_group_geoid", how="left")
    return merged


def _distribution_effects(df: pd.DataFrame, current: pd.Series, amb: pd.Series, blend: pd.Series) -> dict[str, Any]:
    valid = df[["current_exposure", "D_amb", "D_blend"]].replace([np.inf, -np.inf], np.nan).dropna()
    current_clean = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    thresholds = {}
    if not current_clean.empty:
        thresholds = {
            "current_p99": float(current_clean.quantile(0.99)),
            "current_p999": float(current_clean.quantile(0.999)),
        }
    insufficient_rows = pd.Series(False, index=df.index)
    clearing: dict[str, Any] = {}
    for offense in PERSON_EXPOSURE_OFFENSES:
        mode = df[f"estimate_mode_{offense}"].astype("string")
        current_insufficient = mode.eq("insufficient_exposure")
        insufficient_rows |= current_insufficient
        clearing[offense] = {
            "currently_insufficient_exposure_rows": int(current_insufficient.sum()),
            "clear_floor_with_D_amb": int((current_insufficient & df["D_amb"].ge(PERSON_FLOOR)).sum()),
            "clear_floor_with_D_blend": int((current_insufficient & df["D_blend"].ge(PERSON_FLOOR)).sum()),
        }
    return {
        "denominator_correlation": {
            "pearson_D_amb_vs_current_exposure": _to_float(valid["D_amb"].corr(valid["current_exposure"], method="pearson")),
            "spearman_D_amb_vs_current_exposure": _to_float(valid["D_amb"].corr(valid["current_exposure"], method="spearman")),
            "pearson_D_blend_vs_current_exposure": _to_float(valid["D_blend"].corr(valid["current_exposure"], method="pearson")),
            "spearman_D_blend_vs_current_exposure": _to_float(valid["D_blend"].corr(valid["current_exposure"], method="spearman")),
        },
        "index_tail": {
            "current": _tail_stats(current, thresholds=thresholds),
            "D_amb": _tail_stats(amb, thresholds=thresholds),
            "D_blend": _tail_stats(blend, thresholds=thresholds),
            "thresholds": thresholds,
        },
        "insufficient_exposure_floor_clearing": {
            "unique_bg_currently_insufficient_any_person_offense": int(insufficient_rows.sum()),
            "unique_bg_clear_floor_with_D_amb": int((insufficient_rows & df["D_amb"].ge(PERSON_FLOOR)).sum()),
            "unique_bg_clear_floor_with_D_blend": int((insufficient_rows & df["D_blend"].ge(PERSON_FLOOR)).sum()),
            "by_offense": clearing,
        },
    }


def _commercial_share(df: pd.DataFrame) -> pd.Series:
    households = _numeric(df, "households_total")
    premises = _numeric(df, "commercial_premises_total")
    denom = households + premises
    return (premises / denom.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _truth_gradient(df: pd.DataFrame) -> dict[str, Any]:
    if not TRUTH_SURFACE.exists():
        return {"available": False, "reason": f"Missing {TRUTH_SURFACE.relative_to(REPO_ROOT)}"}
    truth = pd.read_parquet(TRUTH_SURFACE)
    truth["block_group_geoid"] = truth["block_group_geoid"].astype(str).str.zfill(12)
    truth = truth[truth["offense"].isin(PERSON_EXPOSURE_OFFENSES)].copy()
    denom_cols = [
        "block_group_geoid",
        "current_exposure",
        "D_amb",
        "D_blend",
        "households_total",
        "commercial_premises_total",
    ]
    merged = truth.merge(df[denom_cols], on="block_group_geoid", how="left")
    merged["commercial_share"] = _commercial_share(merged)
    merged["incident_count"] = pd.to_numeric(merged["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)

    denom_specs = {
        "current_exposure": "current_exposure",
        "D_amb": "D_amb",
        "D_blend": "D_blend",
    }
    rows: list[dict[str, Any]] = []
    for offense in PERSON_EXPOSURE_OFFENSES:
        base = merged[merged["offense"].eq(offense)].copy()
        base = base[base["commercial_share"].notna()]
        if len(base) < 10:
            continue
        ranks = base["commercial_share"].rank(method="first")
        try:
            base["commercial_quintile"] = pd.qcut(ranks, 5, labels=False) + 1
        except ValueError:
            continue
        for denom_name, denom_col in denom_specs.items():
            work = base.copy()
            work["denominator"] = pd.to_numeric(work[denom_col], errors="coerce")
            work = work[work["denominator"].gt(0.0)]
            q1 = work[work["commercial_quintile"].eq(1)]
            q5 = work[work["commercial_quintile"].eq(5)]
            q1_denom = float(q1["denominator"].sum())
            q5_denom = float(q5["denominator"].sum())
            q1_count = float(q1["incident_count"].sum())
            q5_count = float(q5["incident_count"].sum())
            q1_rate = RATE_PER_100K * q1_count / q1_denom if q1_denom > 0 else float("nan")
            q5_rate = RATE_PER_100K * q5_count / q5_denom if q5_denom > 0 else float("nan")
            rows.append(
                {
                    "offense": offense,
                    "denominator": denom_name,
                    "rows": int(len(work)),
                    "cities": int(work["city_name"].nunique()),
                    "q1_rows": int(len(q1)),
                    "q5_rows": int(len(q5)),
                    "q1_commercial_share_mean": _to_float(q1["commercial_share"].mean()),
                    "q5_commercial_share_mean": _to_float(q5["commercial_share"].mean()),
                    "q1_incident_count": q1_count,
                    "q5_incident_count": q5_count,
                    "q1_denominator_sum": q1_denom,
                    "q5_denominator_sum": q5_denom,
                    "q1_rate_per_100k": _to_float(q1_rate),
                    "q5_rate_per_100k": _to_float(q5_rate),
                    "q5_over_q1_gradient": _to_float(q5_rate / q1_rate if q1_rate > 0 else float("nan")),
                }
            )

    gradient = pd.DataFrame(rows)
    comparison: list[dict[str, Any]] = []
    for offense, group in gradient.groupby("offense", sort=True):
        values = group.set_index("denominator")["q5_over_q1_gradient"].to_dict()
        current = values.get("current_exposure")
        for candidate in ["D_amb", "D_blend"]:
            cand = values.get(candidate)
            comparison.append(
                {
                    "offense": offense,
                    "candidate": candidate,
                    "current_gradient": _to_float(current),
                    "candidate_gradient": _to_float(cand),
                    "closer_to_1": bool(
                        current is not None
                        and cand is not None
                        and np.isfinite(current)
                        and np.isfinite(cand)
                        and abs(math.log(max(cand, 1e-12))) < abs(math.log(max(current, 1e-12)))
                    ),
                }
            )
    comparison_df = pd.DataFrame(comparison)
    return {
        "available": True,
        "truth_surface_path": str(TRUTH_SURFACE.relative_to(REPO_ROOT)),
        "truth_rows_person_exposure": int(len(truth)),
        "covered_truth_cities_available": int(truth["city_name"].nunique()),
        "covered_truth_city_names": sorted(truth["city_name"].dropna().unique().tolist()),
        "note": "The local validation truth surface contains 22 covered cities for person-exposure offenses; no local 30-city per-BG truth surface was present.",
        "gradient_rows": gradient.to_dict(orient="records"),
        "summary": {
            "D_amb_offenses_closer_to_1": int(
                comparison_df[comparison_df["candidate"].eq("D_amb")]["closer_to_1"].sum()
            )
            if not comparison_df.empty
            else 0,
            "D_blend_offenses_closer_to_1": int(
                comparison_df[comparison_df["candidate"].eq("D_blend")]["closer_to_1"].sum()
            )
            if not comparison_df.empty
            else 0,
            "offense_comparison": comparison,
        },
    }


def _weirdness(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    residents = _numeric(work, "population_2024")
    day = _numeric(work, "landscan_day_pop")
    night = _numeric(work, "landscan_night_pop")
    work["day_to_residents"] = day / residents.replace(0.0, np.nan)
    work["night_to_residents"] = night / residents.replace(0.0, np.nan)
    work["day_minus_residents"] = day - residents
    base_cols = [
        "block_group_geoid",
        "state_fips",
        "population_2024",
        "current_exposure",
        "landscan_day_pop",
        "landscan_night_pop",
        "day_to_residents",
        "night_to_residents",
        "index_total_primary_event_weighted",
    ]
    day_lt_half = residents.ge(100.0) & day.lt(0.5 * residents)
    day_lt_quarter = residents.ge(100.0) & day.lt(0.25 * residents)
    zero_day_res = residents.gt(0.0) & day.le(0.0)
    both_zero_res = residents.gt(0.0) & day.le(0.0) & night.le(0.0)
    samples = {
        "lowest_day_to_residents_residents_ge_100": work.loc[residents.ge(100.0), base_cols]
        .sort_values(["day_to_residents", "population_2024"], ascending=[True, False], kind="mergesort")
        .head(25)
        .to_dict(orient="records"),
        "zero_day_with_largest_residents": work.loc[zero_day_res, base_cols]
        .sort_values("population_2024", ascending=False, kind="mergesort")
        .head(25)
        .to_dict(orient="records"),
        "both_day_night_zero_with_largest_residents": work.loc[both_zero_res, base_cols]
        .sort_values("population_2024", ascending=False, kind="mergesort")
        .head(25)
        .to_dict(orient="records"),
    }
    return {
        "day_pop_less_than_half_residents_residents_ge_100": int(day_lt_half.sum()),
        "day_pop_less_than_quarter_residents_residents_ge_100": int(day_lt_quarter.sum()),
        "zero_day_cells_with_residents": int(zero_day_res.sum()),
        "both_day_night_zero_cells_with_residents": int(both_zero_res.sum()),
        "samples": samples,
    }


def _round_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(6)
    return out


def evaluate_landscan() -> dict[str, Any]:
    release = _read_release_columns()
    land = pd.read_parquet(LANDSCAN_BG_OUT)
    land["block_group_geoid"] = land["block_group_geoid"].astype(str).str.zfill(12)
    df = release.merge(land, on="block_group_geoid", how="left")
    df[["landscan_day_pop", "landscan_night_pop"]] = df[["landscan_day_pop", "landscan_night_pop"]].fillna(0.0)
    df["current_exposure"] = _numeric(df, "exposure_proxy_2024")
    df["D_amb"] = np.maximum(_numeric(df, "landscan_day_pop"), _numeric(df, "landscan_night_pop"))
    df["D_blend"] = 0.5 * _numeric(df, "landscan_day_pop") + 0.5 * _numeric(df, "landscan_night_pop")

    amb = _compute_candidate_indexes(df, "D_amb", "amb")
    blend = _compute_candidate_indexes(df, "D_blend", "blend")
    df["index_total_primary_event_weighted_D_amb"] = amb["composite"]
    df["index_total_primary_event_weighted_D_blend"] = blend["composite"]
    for offense in PERSON_EXPOSURE_OFFENSES:
        df[f"index_{offense}_primary_D_amb"] = amb["components"][offense]
        df[f"index_{offense}_primary_D_blend"] = blend["components"][offense]

    spots = _spot_rows(df)
    spot_cols = [
        "spot_group",
        "spot_label",
        "block_group_geoid",
        "selection_note",
        "state_fips",
        "population_2024",
        "current_exposure",
        "landscan_day_pop",
        "landscan_night_pop",
        "D_amb",
        "D_blend",
        "expected_count_total",
        "index_total_primary_event_weighted",
        "index_total_primary_event_weighted_D_amb",
        "index_total_primary_event_weighted_D_blend",
    ]
    for offense in PERSON_EXPOSURE_OFFENSES:
        spot_cols.extend([f"expected_count_{offense}", f"index_{offense}_primary"])
        spot_cols.extend([f"index_{offense}_primary_D_amb", f"index_{offense}_primary_D_blend"])
    spot_cols = [col for col in spot_cols if col in spots.columns]
    PATHOLOGICAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    _round_for_csv(spots[spot_cols]).to_csv(PATHOLOGICAL_CSV, index=False)

    current_index = pd.to_numeric(df["index_total_primary_event_weighted"], errors="coerce")
    distribution = _distribution_effects(df, current_index, amb["composite"], blend["composite"])
    census = _census_sanity(df, land)
    truth = _truth_gradient(df)
    weirdness = _weirdness(df)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "release_output": str(RELEASE_BG.relative_to(REPO_ROOT)),
            "landscan_bg_parquet": str(LANDSCAN_BG_OUT.relative_to(REPO_ROOT)),
            "rows": int(len(df)),
            "state_count": int(df["state_fips"].nunique()),
            "person_exposure_floor": PERSON_FLOOR,
            "candidate_denominators": {
                "D_amb": "max(landscan_day_pop, landscan_night_pop)",
                "D_blend": "0.5 * landscan_day_pop + 0.5 * landscan_night_pop",
            },
            "pilot_assumption": "Counts are fixed from expected_count_*; person-exposure offense denominators are swapped for D_amb/D_blend; burglary and motor_vehicle_theft retain current primary indexes. Non-residential/special-use/vehicle-invalid suppressions are preserved; current insufficient_exposure rows can clear only when the candidate denominator is >= 50.",
        },
        "landscan_provenance": {
            "product": "LandScan USA 2021",
            "publication_date": "2022-07-09",
            "doi": "https://doi.org/10.48690/1527701",
            "portal": "https://landscan.ornl.gov/",
            "active_usa_vintage_observed_in_portal": 2021,
            "newest_global_vintage_observed_in_portal": 2024,
            "usa_2024_adjacent_note": "The portal JavaScript advertised LandScan Global 2024 and LandScan USA year options through 2021. Download API probes for USA 2024, 2023, and 2022 returned signed S3 URLs whose objects returned NoSuchKey; USA 2021 day/night assets downloaded without login.",
            "license_text_recorded": "These datasets are offered under the Creative Commons Attribution 4.0 International License. Users are free to use, copy, distribute, transmit, and adapt the data for commercial and non-commercial purposes, without restriction, as long as clear attribution of the source is provided. For more information, please refer to the CC BY 4.0 documentation.",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "resolution": "3 arc-second raster grid, approximately 90 meters at CONUS latitudes; portal copy describes LandScan resolution down to 100 meter grid cells.",
            "documented_submodels": ["Residents", "Prisoners", "Workers", "Students", "Shoppers"],
            "metadata_note": "Metadata says baseline estimates do not include transitory populations such as business travelers and tourists.",
        },
        "candidate_offense_normalizers": {
            "D_amb": amb["offense_summary"],
            "D_blend": blend["offense_summary"],
            "event_weights": amb["weights"],
        },
        "sanity": census,
        "pathological_cells": {
            "csv": str(PATHOLOGICAL_CSV.relative_to(REPO_ROOT)),
            "rows": spots[spot_cols].to_dict(orient="records"),
        },
        "distribution_effects": distribution,
        "covered_city_truth_check": truth,
        "weird_or_worse": weirdness,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")
    REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    number = _to_float(value)
    if number is None:
        return "NA"
    if abs(number) >= 1000:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "NA"
    return f"{number:,.0f}"


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(_fmt(value, 2))
            else:
                cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_markdown(report: dict[str, Any]) -> str:
    sanity = report["sanity"]
    dist = report["distribution_effects"]
    truth = report["covered_city_truth_check"]
    weird = report["weird_or_worse"]
    spots = report["pathological_cells"]["rows"]
    tail = dist["index_tail"]
    clearing = dist["insufficient_exposure_floor_clearing"]

    spot_rows = [
        {
            "label": row.get("spot_label"),
            "bg": row.get("block_group_geoid"),
            "current": _fmt_int(row.get("current_exposure")),
            "day": _fmt_int(row.get("landscan_day_pop")),
            "night": _fmt_int(row.get("landscan_night_pop")),
            "idx": _fmt(row.get("index_total_primary_event_weighted"), 1),
            "amb": _fmt(row.get("index_total_primary_event_weighted_D_amb"), 1),
            "blend": _fmt(row.get("index_total_primary_event_weighted_D_blend"), 1),
        }
        for row in spots
    ]

    gradient_rows = []
    if truth.get("available"):
        for row in truth.get("gradient_rows", []):
            gradient_rows.append(
                {
                    "offense": row["offense"],
                    "denominator": row["denominator"],
                    "cities": row["cities"],
                    "q1": _fmt(row["q1_rate_per_100k"], 1),
                    "q5": _fmt(row["q5_rate_per_100k"], 1),
                    "gradient": _fmt(row["q5_over_q1_gradient"], 2),
                }
            )

    lines = [
        "# LandScan USA Ambient Population Pilot",
        "",
        "## Provenance",
        "",
        "- Product: LandScan USA 2021 day/night CONUS rasters.",
        "- Newest USA vintage observed: 2021. The portal advertised LandScan Global 2024, but USA 2024/2023/2022 API probes returned missing S3 objects; USA 2021 downloaded without a login wall.",
        "- License text recorded: These datasets are offered under the Creative Commons Attribution 4.0 International License. Users are free to use, copy, distribute, transmit, and adapt the data for commercial and non-commercial purposes, without restriction, as long as clear attribution of the source is provided. For more information, please refer to the CC BY 4.0 documentation.",
        "- Resolution: 3 arc-second grid, approximately 90 meters at CONUS latitudes.",
        "- Documented sub-model components: Residents, Prisoners, Workers, Students, Shoppers. Metadata notes that transitory populations such as business travelers and tourists are not included.",
        "",
        "## Aggregation Sanity",
        "",
        f"- Release scope: {report['scope']['rows']:,} block groups across {report['scope']['state_count']} CONUS+DC state FIPS values.",
        f"- LandScan day sum over release BGs: {_fmt_int(sanity['landscan_day_pop_sum'])}; night sum: {_fmt_int(sanity['landscan_night_pop_sum'])}.",
        f"- Census PopEstimate 2020 scope total: {_fmt_int(sanity['census_popestimate2020_scope_total'])}; 2024 scope total: {_fmt_int(sanity['census_popestimate2024_scope_total'])}.",
        f"- Day/Census 2020 ratio: {_fmt(sanity['day_to_census2020_ratio'], 4)}; night/Census 2020 ratio: {_fmt(sanity['night_to_census2020_ratio'], 4)}.",
        f"- Day/Census 2024 ratio: {_fmt(sanity['day_to_census2024_ratio'], 4)}; night/Census 2024 ratio: {_fmt(sanity['night_to_census2024_ratio'], 4)}.",
        "",
        "## Pathological Cells",
        "",
        _markdown_table(
            spot_rows,
            [
                ("Cell", "label"),
                ("BG", "bg"),
                ("Current Exposure", "current"),
                ("LS Day", "day"),
                ("LS Night", "night"),
                ("Current Index", "idx"),
                ("D_amb Index", "amb"),
                ("D_blend Index", "blend"),
            ],
        ),
        "",
        f"Full CSV: `{report['pathological_cells']['csv']}`.",
        "",
        "## National Distribution Effects",
        "",
        f"- D_amb vs current exposure Pearson/Spearman: {_fmt(dist['denominator_correlation']['pearson_D_amb_vs_current_exposure'], 4)} / {_fmt(dist['denominator_correlation']['spearman_D_amb_vs_current_exposure'], 4)}.",
        f"- Current p99/p99.9 total index: {_fmt(tail['current']['p99'], 2)} / {_fmt(tail['current']['p999'], 2)}.",
        f"- D_amb p99/p99.9 total index: {_fmt(tail['D_amb']['p99'], 2)} / {_fmt(tail['D_amb']['p999'], 2)}.",
        f"- D_blend p99/p99.9 total index: {_fmt(tail['D_blend']['p99'], 2)} / {_fmt(tail['D_blend']['p999'], 2)}.",
        f"- Unique currently insufficient-exposure BGs clearing the floor with D_amb: {clearing['unique_bg_clear_floor_with_D_amb']:,} of {clearing['unique_bg_currently_insufficient_any_person_offense']:,}; with D_blend: {clearing['unique_bg_clear_floor_with_D_blend']:,}.",
        "",
        "## Covered-City Truth Check",
        "",
        f"- Available truth artifact: `{truth.get('truth_surface_path', 'NA')}`.",
        f"- Covered cities available locally: {truth.get('covered_truth_cities_available', 0)}. {truth.get('note', '')}",
        f"- Offenses closer to a 1.0 commercial-share gradient under D_amb: {truth.get('summary', {}).get('D_amb_offenses_closer_to_1', 0)} of {len(PERSON_EXPOSURE_OFFENSES)}; under D_blend: {truth.get('summary', {}).get('D_blend_offenses_closer_to_1', 0)} of {len(PERSON_EXPOSURE_OFFENSES)}.",
        "",
        _markdown_table(
            gradient_rows,
            [
                ("Offense", "offense"),
                ("Denom", "denominator"),
                ("Cities", "cities"),
                ("Q1 Rate", "q1"),
                ("Q5 Rate", "q5"),
                ("Q5/Q1", "gradient"),
            ],
        ),
        "",
        "## Weird Or Worse",
        "",
        f"- BGs with day population below half of residents, residents >= 100: {weird['day_pop_less_than_half_residents_residents_ge_100']:,}.",
        f"- BGs with day population below one quarter of residents, residents >= 100: {weird['day_pop_less_than_quarter_residents_residents_ge_100']:,}.",
        f"- Zero-day BGs with residents: {weird['zero_day_cells_with_residents']:,}.",
        f"- Both day and night zero with residents: {weird['both_day_night_zero_cells_with_residents']:,}.",
        "",
        "## Design Boundary",
        "",
        "This is an offline pilot only. It does not choose the production blend, offense coverage, or publication-floor rule.",
        "",
    ]
    return "\n".join(lines)


def write_readme() -> None:
    raw_downloads = LANDSCAN_DIR / "raw_downloads_2021.json"
    checksums = LANDSCAN_DIR / "raster_checksums_2021.json"
    raw = json.loads(raw_downloads.read_text()) if raw_downloads.exists() else {}
    raster_sums = json.loads(checksums.read_text()) if checksums.exists() else {}
    lines = [
        "# LandScan USA",
        "",
        "This directory stores the LandScan USA day/night sidecar inputs and outputs used for the Q1 ambient-population denominator pilot.",
        "",
        "## Provenance",
        "",
        "- Product: LandScan USA 2021.",
        "- Portal: https://landscan.ornl.gov/",
        "- DOI/citation URL recorded in metadata: https://doi.org/10.48690/1527701",
        "- Publication date in metadata: 2022-07-09.",
        "- Newest USA vintage observed in the portal/API: 2021. The portal JavaScript advertised LandScan Global 2024, but USA 2024, 2023, and 2022 download probes returned signed S3 URLs whose objects returned `NoSuchKey`.",
        "- Automated access result: no login wall for the 2021 USA day/night asset downloads via the portal download API.",
        "",
        "## License",
        "",
        "License text recorded from the portal:",
        "",
        "> These datasets are offered under the Creative Commons Attribution 4.0 International License. Users are free to use, copy, distribute, transmit, and adapt the data for commercial and non-commercial purposes, without restriction, as long as clear attribution of the source is provided. For more information, please refer to the CC BY 4.0 documentation.",
        "",
        "License URL: https://creativecommons.org/licenses/by/4.0/",
        "",
        "## Resolution And Components",
        "",
        "- Metadata resolution: 3 arc-second raster grid, approximately 90 meters at CONUS latitudes.",
        "- Portal copy describes LandScan resolution down to 100 meter grid cells.",
        "- Documented sub-model components: Residents, Prisoners, Workers, Students, and Shoppers.",
        "- Metadata note: these are baseline estimates and do not include transitory populations such as business travelers and tourists.",
        "",
        "## Files",
        "",
        "- `raw/landscan-usa-2021-day-assets.zip`: downloaded portal day asset bundle.",
        "- `raw/landscan-usa-2021-night-assets.zip`: downloaded portal night asset bundle.",
        "- `products/landscan-usa-2021-day.zip`: nested product zip extracted from the day asset bundle.",
        "- `products/landscan-usa-2021-night.zip`: nested product zip extracted from the night asset bundle.",
        "- `rasters/landscan-usa-2021-conus-day.tif`: CONUS day raster.",
        "- `rasters/landscan-usa-2021-conus-night.tif`: CONUS night raster.",
        "- `block_group_landscan_usa_2021.parquet`: release block-group aggregation with columns `block_group_geoid`, `landscan_day_pop`, `landscan_night_pop`.",
        "- `state_block_groups/`: state-level aggregation shards.",
        "- `conus_day_metadata_2021.xml` and `conus_night_metadata_2021.xml`: source metadata XMLs.",
        "- `raw_downloads_2021.json`, `zip_members_2021.json`, and `raster_checksums_2021.json`: acquisition and verification manifests.",
        "",
        "## Verification Snapshot",
        "",
    ]
    if raw:
        lines.extend(["Raw bundle checksums:", ""])
        for key, value in raw.items():
            if isinstance(value, dict):
                path = value.get("path") or value.get("out") or key
                sha = value.get("sha256")
                size = value.get("bytes") or value.get("size")
                lines.append(f"- `{path}`: {size} bytes, sha256 `{sha}`.")
        lines.append("")
    if raster_sums:
        lines.extend(["Raster checksums:", ""])
        for key, value in raster_sums.items():
            if isinstance(value, dict):
                path = value.get("path") or key
                sha = value.get("sha256")
                size = value.get("bytes") or value.get("size")
                lines.append(f"- `{path}`: {size} bytes, sha256 `{sha}`.")
        lines.append("")
    LANDSCAN_DIR.mkdir(parents=True, exist_ok=True)
    (LANDSCAN_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tile-size", type=int, default=4096)
    args = parser.parse_args()

    if not args.skip_aggregate:
        summary = aggregate_landscan(tile_size=int(args.tile_size), force=bool(args.force))
        print(json.dumps(summary, indent=2, default=_json_default), flush=True)

    if not args.skip_evaluate:
        if not LANDSCAN_BG_OUT.exists():
            raise FileNotFoundError(f"Missing {LANDSCAN_BG_OUT}; run without --skip-aggregate first.")
        day_summary = _raster_summary(RASTER_DAY)
        night_summary = _raster_summary(RASTER_NIGHT)
        report = evaluate_landscan()
        report["landscan_raster_summary"] = {
            "day": day_summary.as_dict(),
            "night": night_summary.as_dict(),
        }
        REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")
        REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
        write_readme()
        print(f"report_json={REPORT_JSON}", flush=True)
        print(f"report_md={REPORT_MD}", flush=True)
        print(f"pathological_csv={PATHOLOGICAL_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
