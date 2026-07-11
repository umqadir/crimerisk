from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from crimerisk.covariates.roads import RoadAggregationConfig, infer_state_fips_from_bg_dir, load_block_groups


HPMS_SERVICE_CATALOG_URL = "https://geo.dot.gov/server/rest/services/Hosted?f=pjson"
HPMS_SERVICE_RE = re.compile(r"^Hosted/HPMS_FULL_([A-Z]{2})_(\d{4})$", re.IGNORECASE)

STATE_ABBR_BY_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


@dataclass(frozen=True)
class HpmsBuildConfig:
    year: int = 2024
    timeout_seconds: int = 120
    chunk_size: int = 2000
    query_crs: int = 4326
    length_crs: str = RoadAggregationConfig().length_crs


def _service_catalog(timeout_seconds: int) -> list[dict[str, object]]:
    response = requests.get(HPMS_SERVICE_CATALOG_URL, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("services", []))


def discover_hpms_full_services(*, year: int, timeout_seconds: int = 120) -> dict[str, str]:
    out: dict[str, str] = {}
    for svc in _service_catalog(timeout_seconds):
        raw_name = str(svc.get("name", ""))
        match = HPMS_SERVICE_RE.match(raw_name)
        if not match:
            continue
        state_abbr, svc_year = match.groups()
        if int(svc_year) != int(year):
            continue
        out[state_abbr.upper()] = raw_name
    return out


def hpms_service_url(*, service_name: str) -> str:
    return f"https://geo.dot.gov/server/rest/services/{service_name}/FeatureServer/0/query"


def _query_hpms_chunk(
    *,
    service_name: str,
    offset: int,
    cfg: HpmsBuildConfig,
) -> dict[str, object]:
    response = requests.get(
        hpms_service_url(service_name=service_name),
        params={
            "where": "1=1",
            "outFields": ",".join(
                [
                    "objectid",
                    "data_year",
                    "state_id",
                    "county_id",
                    "route_id",
                    "sectionlength",
                    "f_system",
                    "facility_type",
                    "access_control",
                    "through_lanes",
                    "aadt",
                    "aadt_combination",
                    "aadt_single_unit",
                    "nhs",
                ]
            ),
            "returnGeometry": "true",
            "outSR": str(int(cfg.query_crs)),
            "orderByFields": "objectid ASC",
            "resultOffset": int(offset),
            "resultRecordCount": int(cfg.chunk_size),
            "f": "geojson",
        },
        timeout=cfg.timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def load_hpms_state_features(
    *,
    state_fips: str,
    cfg: HpmsBuildConfig = HpmsBuildConfig(),
) -> gpd.GeoDataFrame:
    state = str(state_fips).zfill(2)
    state_abbr = STATE_ABBR_BY_FIPS.get(state)
    if state_abbr is None:
        raise KeyError(f"Unsupported state_fips for HPMS: {state_fips}")
    service_name = f"Hosted/HPMS_FULL_{state_abbr}_{int(cfg.year)}"

    features: list[dict[str, object]] = []
    offset = 0
    while True:
        payload = _query_hpms_chunk(service_name=service_name, offset=offset, cfg=cfg)
        chunk = list(payload.get("features", []))
        if not chunk:
            break
        features.extend(chunk)
        exceeded = bool(payload.get("properties", {}).get("exceededTransferLimit"))
        if not exceeded or len(chunk) < int(cfg.chunk_size):
            break
        offset += len(chunk)

    if not features:
        return gpd.GeoDataFrame(
            columns=[
                "state_fips",
                "county_fips",
                "route_id",
                "data_year",
                "f_system",
                "facility_type",
                "access_control",
                "through_lanes",
                "aadt",
                "aadt_combination",
                "aadt_single_unit",
                "truck_aadt",
                "nhs",
                "sectionlength",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf = gdf.rename(columns={"state_id": "state_fips", "county_id": "county_fips"}).copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["state_fips"] = pd.to_numeric(gdf.get("state_fips"), errors="coerce").astype("Int64").astype("string").str.zfill(2)
    gdf["county_fips"] = pd.to_numeric(gdf.get("county_fips"), errors="coerce").astype("Int64").astype("string").str.zfill(3)
    gdf["route_id"] = gdf.get("route_id", pd.Series(index=gdf.index, dtype="string")).astype("string")
    for col in ["data_year", "f_system", "facility_type", "access_control", "through_lanes", "aadt", "aadt_combination", "aadt_single_unit", "nhs", "sectionlength"]:
        gdf[col] = pd.to_numeric(gdf.get(col), errors="coerce")
    gdf["truck_aadt"] = (
        gdf["aadt_combination"].fillna(0.0)
        + gdf["aadt_single_unit"].fillna(0.0)
    )
    gdf["is_nhs"] = pd.to_numeric(gdf["nhs"], errors="coerce").fillna(0.0).gt(0)
    gdf["is_arterial"] = pd.to_numeric(gdf["f_system"], errors="coerce").isin([1, 2, 3, 4])
    gdf["is_collector"] = pd.to_numeric(gdf["f_system"], errors="coerce").isin([5, 6])
    return gdf


def aggregate_hpms_to_block_groups(
    hpms: gpd.GeoDataFrame,
    block_groups: gpd.GeoDataFrame,
    *,
    cfg: HpmsBuildConfig = HpmsBuildConfig(),
) -> pd.DataFrame:
    group_cols = ["bg_id", "tract_id", "state_fips", "county_fips"]
    if block_groups.empty:
        return pd.DataFrame(columns=[*group_cols])
    base = block_groups[group_cols].drop_duplicates().copy()
    metric_cols = [
        "hpms_length_km",
        "hpms_nhs_length_km",
        "hpms_arterial_length_km",
        "hpms_collector_length_km",
        "hpms_aadt_km",
        "hpms_truck_aadt_km",
        "hpms_lane_aadt_km",
        "hpms_mean_aadt",
        "hpms_truck_share",
    ]
    if hpms.empty:
        for col in metric_cols:
            base[col] = 0.0
        return base.sort_values(group_cols, kind="stable").reset_index(drop=True)

    hpms_proj = hpms.to_crs(cfg.length_crs)
    bg_proj = block_groups.to_crs(cfg.length_crs)
    bg_geom = bg_proj[["bg_id", "geometry"]].rename(columns={"geometry": "bg_geometry"})
    joined = gpd.sjoin(
        hpms_proj[
            [
                "route_id",
                "state_fips",
                "county_fips",
                "aadt",
                "truck_aadt",
                "through_lanes",
                "is_nhs",
                "is_arterial",
                "is_collector",
                "geometry",
            ]
        ],
        bg_proj[group_cols + ["geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="hpms",
        rsuffix="bg",
    )
    if joined.empty:
        for col in metric_cols:
            base[col] = 0.0
        return base.sort_values(group_cols, kind="stable").reset_index(drop=True)

    joined = joined.rename(
        columns={
            "index_right": "bg_index",
            "index_bg": "bg_index",
            "state_fips_bg": "state_fips",
            "county_fips_bg": "county_fips",
        }
    ).merge(bg_geom, on="bg_id", how="left")
    clipped = joined.geometry.intersection(joined["bg_geometry"])
    clipped = gpd.GeoSeries(clipped, crs=cfg.length_crs)
    joined["segment_length_km"] = clipped.length / 1000.0
    joined = joined[joined["segment_length_km"] > 0].copy()
    if joined.empty:
        for col in metric_cols:
            base[col] = 0.0
        return base.sort_values(group_cols, kind="stable").reset_index(drop=True)

    joined["aadt"] = pd.to_numeric(joined["aadt"], errors="coerce").fillna(0.0)
    joined["truck_aadt"] = pd.to_numeric(joined["truck_aadt"], errors="coerce").fillna(0.0)
    joined["through_lanes"] = pd.to_numeric(joined["through_lanes"], errors="coerce").fillna(0.0)
    joined["aadt_km"] = joined["segment_length_km"] * joined["aadt"]
    joined["truck_aadt_km"] = joined["segment_length_km"] * joined["truck_aadt"]
    joined["lane_aadt_km"] = joined["segment_length_km"] * joined["through_lanes"] * joined["aadt"]

    totals = joined.groupby(group_cols, dropna=False).agg(
        hpms_length_km=("segment_length_km", "sum"),
        hpms_nhs_length_km=("segment_length_km", lambda s: float(s[joined.loc[s.index, "is_nhs"]].sum())),
        hpms_arterial_length_km=("segment_length_km", lambda s: float(s[joined.loc[s.index, "is_arterial"]].sum())),
        hpms_collector_length_km=("segment_length_km", lambda s: float(s[joined.loc[s.index, "is_collector"]].sum())),
        hpms_aadt_km=("aadt_km", "sum"),
        hpms_truck_aadt_km=("truck_aadt_km", "sum"),
        hpms_lane_aadt_km=("lane_aadt_km", "sum"),
    ).reset_index()

    totals["hpms_mean_aadt"] = np.where(
        totals["hpms_length_km"] > 0,
        totals["hpms_aadt_km"] / totals["hpms_length_km"],
        0.0,
    )
    totals["hpms_truck_share"] = np.where(
        totals["hpms_aadt_km"] > 0,
        totals["hpms_truck_aadt_km"] / totals["hpms_aadt_km"],
        0.0,
    )
    out = base.merge(totals, on=group_cols, how="left")
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out.sort_values(group_cols, kind="stable").reset_index(drop=True)


def build_state_block_group_hpms_metrics(
    *,
    state_fips: str,
    bg_zip: Path,
    cfg: HpmsBuildConfig = HpmsBuildConfig(),
) -> pd.DataFrame:
    block_groups = load_block_groups(bg_zip)
    state = str(state_fips).zfill(2)
    if not block_groups.empty:
        block_groups = block_groups[block_groups["state_fips"].eq(state)].copy()
    hpms = load_hpms_state_features(state_fips=state, cfg=cfg)
    return aggregate_hpms_to_block_groups(hpms, block_groups, cfg=cfg)


def build_hpms_frame_by_block_group(
    *,
    bg_dir: Path,
    state_fips_values: list[str] | None = None,
    cfg: HpmsBuildConfig = HpmsBuildConfig(),
) -> pd.DataFrame:
    states = state_fips_values if state_fips_values is not None else infer_state_fips_from_bg_dir(bg_dir, year=2020)
    frames: list[pd.DataFrame] = []
    for state_fips in states:
        bg_zip = bg_dir / f"tl_2020_{str(state_fips).zfill(2)}_bg.zip"
        if not bg_zip.exists():
            raise FileNotFoundError(f"Missing BG zip for state {state_fips}: {bg_zip}")
        frames.append(
            build_state_block_group_hpms_metrics(
                state_fips=str(state_fips).zfill(2),
                bg_zip=bg_zip,
                cfg=cfg,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
