from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import requests


NTM_STOPS_DOWNLOAD_URL = (
    "https://ngda-transportation-geoplatform.hub.arcgis.com/api/download/v1/items/"
    "959b4adc3ff94bc6a46f2c1d515d09aa/shapefile?layers=0"
)

FIXED_GUIDEWAY_MODE_TOKENS = {
    "subway, metro",
    "tram, streetcar, light rail",
    "rail",
    "ferry",
    "cable tram",
    "aerial lift, suspended cable car",
    "funicular",
}


@dataclass(frozen=True)
class NationalTransitMapBuildConfig:
    timeout_seconds: int = 120
    projected_crs: str = "EPSG:5070"
    bbox_padding_degrees: float = 0.1


def download_ntm_stops_shapefile(
    *,
    out_zip_path: Path,
    cfg: NationalTransitMapBuildConfig = NationalTransitMapBuildConfig(),
    overwrite: bool = False,
) -> Path:
    if out_zip_path.exists() and out_zip_path.stat().st_size > 0 and not overwrite:
        return out_zip_path
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        NTM_STOPS_DOWNLOAD_URL,
        timeout=int(cfg.timeout_seconds),
        allow_redirects=True,
        headers={"User-Agent": "crimerisk-v2/1.0"},
    )
    response.raise_for_status()
    out_zip_path.write_bytes(response.content)
    if out_zip_path.stat().st_size <= 0:
        raise RuntimeError("Downloaded empty National Transit Map stops shapefile zip.")
    return out_zip_path


def _normalize_mode_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    tokens = [
        re.sub(r'\s+', " ", token).strip().strip('"').lower()
        for token in text.split(";")
        if str(token).strip()
    ]
    return [token for token in tokens if token]


def _is_fixed_guideway(mode_tokens: list[str]) -> bool:
    return any(token in FIXED_GUIDEWAY_MODE_TOKENS for token in mode_tokens)


def load_ntm_stops(*, stops_zip_path: Path) -> gpd.GeoDataFrame:
    stops = gpd.read_file(f"zip://{stops_zip_path}")[
        ["stop_id", "stop_name", "stop_type_", "download_d", "geometry"]
    ].copy()
    stops["stop_id"] = stops["stop_id"].astype("string").str.strip()
    stops["stop_name"] = stops["stop_name"].astype("string").str.strip()
    stops["download_d"] = stops["download_d"].astype("string").str.strip()
    stops["stop_mode_tokens"] = stops["stop_type_"].apply(_normalize_mode_tokens)
    stops["fixed_guideway_stop_flag"] = stops["stop_mode_tokens"].apply(_is_fixed_guideway)
    stops["transit_mode_count"] = stops["stop_mode_tokens"].apply(lambda values: len(set(values)))
    if stops.crs is None:
        stops = stops.set_crs("EPSG:4326")
    return stops[["stop_id", "stop_name", "download_d", "stop_mode_tokens", "fixed_guideway_stop_flag", "transit_mode_count", "geometry"]]


def _load_block_groups_for_state(*, bg_zip: Path) -> gpd.GeoDataFrame:
    bg = gpd.read_file(f"zip://{bg_zip}")[
        ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE", "ALAND", "geometry"]
    ].copy()
    bg["bg_id"] = bg["GEOID"].astype("string").str.zfill(12)
    bg["tract_id"] = bg["bg_id"].str.slice(0, 11)
    bg["state_fips"] = bg["STATEFP"].astype("string").str.zfill(2)
    bg["county_fips"] = bg["COUNTYFP"].astype("string").str.zfill(3)
    bg["aland20"] = pd.to_numeric(bg["ALAND"], errors="coerce").fillna(0.0)
    return bg[["bg_id", "tract_id", "state_fips", "county_fips", "aland20", "geometry"]].drop_duplicates("bg_id")


def _subset_stops_for_state(
    *,
    stops: gpd.GeoDataFrame,
    bg: gpd.GeoDataFrame,
    cfg: NationalTransitMapBuildConfig,
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bg.total_bounds
    pad = float(cfg.bbox_padding_degrees)
    subset = stops.cx[minx - pad:maxx + pad, miny - pad:maxy + pad].copy()
    return subset


def _nearest_stop_distances(
    *,
    bg: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    distance_col: str,
    projected_crs: str,
) -> pd.DataFrame:
    base = bg[["bg_id", "tract_id", "state_fips", "county_fips"]].copy()
    if stops.empty:
        base[distance_col] = np.nan
        return base
    bg_projected = bg[["bg_id", "tract_id", "state_fips", "county_fips", "geometry"]].copy().to_crs(projected_crs)
    bg_points = gpd.GeoDataFrame(
        bg_projected.drop(columns=["geometry"]).copy(),
        geometry=bg_projected.geometry.centroid,
        crs=projected_crs,
    )
    stop_points = stops[["stop_id", "geometry"]].copy().to_crs(projected_crs)
    nearest = gpd.sjoin_nearest(
        bg_points,
        stop_points,
        how="left",
        distance_col="_nearest_meters",
    )
    out = nearest[["bg_id", "tract_id", "state_fips", "county_fips", "_nearest_meters"]].copy()
    out = (
        out.groupby(["bg_id", "tract_id", "state_fips", "county_fips"], dropna=False, as_index=False)["_nearest_meters"]
        .min()
    )
    out[distance_col] = pd.to_numeric(out["_nearest_meters"], errors="coerce") / 1000.0
    return out.drop(columns=["_nearest_meters"])


def _count_modes_by_bg(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame(
            columns=["bg_id", "tract_id", "state_fips", "county_fips", "transit_mode_count_bg"]
        )
    pairs: list[tuple[str, str, str, str, str]] = []
    for row in joined.itertuples(index=False):
        mode_tokens = getattr(row, "stop_mode_tokens", []) or []
        for token in set(mode_tokens):
            pairs.append(
                (
                    str(getattr(row, "bg_id")),
                    str(getattr(row, "tract_id")),
                    str(getattr(row, "state_fips")),
                    str(getattr(row, "county_fips")),
                    str(token),
                )
            )
    if not pairs:
        return pd.DataFrame(
            columns=["bg_id", "tract_id", "state_fips", "county_fips", "transit_mode_count_bg"]
        )
    mode_frame = pd.DataFrame(
        pairs,
        columns=["bg_id", "tract_id", "state_fips", "county_fips", "mode_token"],
    ).drop_duplicates()
    counts = (
        mode_frame.groupby(["bg_id", "tract_id", "state_fips", "county_fips"], dropna=False)
        .size()
        .rename("transit_mode_count_bg")
        .reset_index()
    )
    counts["transit_mode_count_bg"] = pd.to_numeric(counts["transit_mode_count_bg"], errors="coerce").fillna(0).astype(int)
    return counts


def build_block_group_transit_stop_features(
    *,
    stops_zip_path: Path,
    bg_dir: Path,
    state_fips_values: list[str] | None = None,
    cfg: NationalTransitMapBuildConfig = NationalTransitMapBuildConfig(),
) -> pd.DataFrame:
    stops = load_ntm_stops(stops_zip_path=stops_zip_path)
    frames: list[pd.DataFrame] = []
    bg_paths = sorted(bg_dir.glob("tl_2020_*_bg.zip"))
    if state_fips_values is not None:
        allowed = {str(value).zfill(2) for value in state_fips_values}
        bg_paths = [path for path in bg_paths if path.stem.split("_")[2] in allowed]
    for bg_zip in bg_paths:
        bg = _load_block_groups_for_state(bg_zip=bg_zip)
        subset = _subset_stops_for_state(stops=stops, bg=bg, cfg=cfg)
        if subset.crs != bg.crs:
            subset = subset.to_crs(bg.crs)
        if subset.empty:
            frame = bg[["bg_id", "tract_id", "state_fips", "county_fips", "aland20"]].copy()
            frame["transit_stop_count"] = 0
            frame["fixed_guideway_stop_count"] = 0
            frame["transit_mode_count_bg"] = 0
            frame["nearest_transit_stop_km"] = np.nan
            frame["nearest_fixed_guideway_stop_km"] = np.nan
            frame["transit_stop_present"] = 0.0
            frame["fixed_guideway_stop_present"] = 0.0
            frame["transit_ntm_snapshot_date"] = pd.NA
            frames.append(frame)
            continue

        joined = gpd.sjoin(
            subset,
            bg[["bg_id", "tract_id", "state_fips", "county_fips", "aland20", "geometry"]],
            how="inner",
            predicate="intersects",
        )[
            [
                "stop_id",
                "download_d",
                "stop_mode_tokens",
                "fixed_guideway_stop_flag",
                "bg_id",
                "tract_id",
                "state_fips",
                "county_fips",
                "aland20",
            ]
        ].drop_duplicates(["stop_id", "bg_id"])
        counts = (
            joined.groupby(["bg_id", "tract_id", "state_fips", "county_fips", "aland20"], dropna=False)
            .agg(
                transit_stop_count=("stop_id", "size"),
                fixed_guideway_stop_count=("fixed_guideway_stop_flag", "sum"),
            )
            .reset_index()
        )
        counts["transit_stop_count"] = pd.to_numeric(counts["transit_stop_count"], errors="coerce").fillna(0).astype(int)
        counts["fixed_guideway_stop_count"] = pd.to_numeric(counts["fixed_guideway_stop_count"], errors="coerce").fillna(0).astype(int)
        mode_counts = _count_modes_by_bg(joined)
        snapshot_dates = (
            joined.groupby(["bg_id", "tract_id", "state_fips", "county_fips"], dropna=False)["download_d"]
            .agg(lambda values: sorted({str(v).strip() for v in values if str(v).strip()})[-1] if len(values) else pd.NA)
            .rename("transit_ntm_snapshot_date")
            .reset_index()
        )
        nearest_any = _nearest_stop_distances(
            bg=bg,
            stops=subset,
            distance_col="nearest_transit_stop_km",
            projected_crs=str(cfg.projected_crs),
        )
        nearest_fixed = _nearest_stop_distances(
            bg=bg,
            stops=subset[subset["fixed_guideway_stop_flag"]].copy(),
            distance_col="nearest_fixed_guideway_stop_km",
            projected_crs=str(cfg.projected_crs),
        )
        frame = bg[["bg_id", "tract_id", "state_fips", "county_fips", "aland20"]].copy()
        frame = frame.merge(counts, on=["bg_id", "tract_id", "state_fips", "county_fips", "aland20"], how="left")
        frame = frame.merge(mode_counts, on=["bg_id", "tract_id", "state_fips", "county_fips"], how="left")
        frame = frame.merge(snapshot_dates, on=["bg_id", "tract_id", "state_fips", "county_fips"], how="left")
        frame = frame.merge(nearest_any, on=["bg_id", "tract_id", "state_fips", "county_fips"], how="left")
        frame = frame.merge(nearest_fixed, on=["bg_id", "tract_id", "state_fips", "county_fips"], how="left")
        frame["transit_stop_count"] = pd.to_numeric(frame["transit_stop_count"], errors="coerce").fillna(0).astype(int)
        frame["fixed_guideway_stop_count"] = pd.to_numeric(frame["fixed_guideway_stop_count"], errors="coerce").fillna(0).astype(int)
        frame["transit_mode_count_bg"] = pd.to_numeric(frame["transit_mode_count_bg"], errors="coerce").fillna(0).astype(int)
        frame["transit_stop_present"] = frame["transit_stop_count"].gt(0).astype(float)
        frame["fixed_guideway_stop_present"] = frame["fixed_guideway_stop_count"].gt(0).astype(float)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "bg_id",
                "tract_id",
                "state_fips",
                "county_fips",
                "aland20",
                "transit_stop_count",
                "fixed_guideway_stop_count",
                "transit_mode_count_bg",
                "nearest_transit_stop_km",
                "nearest_fixed_guideway_stop_km",
                "transit_stop_present",
                "fixed_guideway_stop_present",
                "transit_ntm_snapshot_date",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[
        [
            "bg_id",
            "tract_id",
            "state_fips",
            "county_fips",
            "aland20",
            "transit_stop_count",
            "fixed_guideway_stop_count",
            "transit_mode_count_bg",
            "nearest_transit_stop_km",
            "nearest_fixed_guideway_stop_km",
            "transit_stop_present",
            "fixed_guideway_stop_present",
            "transit_ntm_snapshot_date",
        ]
    ].drop_duplicates(["bg_id", "tract_id", "state_fips"]).reset_index(drop=True)
