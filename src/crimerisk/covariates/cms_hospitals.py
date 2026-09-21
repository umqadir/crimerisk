from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from crimerisk.covariates.roads import LENGTH_CRS, infer_state_fips_from_bg_dir, load_block_groups


CMS_HOSPITAL_DATASET_ID = "xubh-q36u"
CMS_DATASET_META_URL = f"https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/{CMS_HOSPITAL_DATASET_ID}"
CENSUS_BATCH_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"


@dataclass(frozen=True)
class CmsHospitalBuildConfig:
    timeout_seconds: int = 120
    batch_size: int = 5000
    benchmark: str = "Public_AR_Current"
    vintage: str = "Current_Current"
    point_crs: str = "EPSG:4326"


def cms_hospital_download_url(*, timeout_seconds: int = 120) -> str:
    response = requests.get(CMS_DATASET_META_URL, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    distributions = list(payload.get("distribution", []))
    for item in distributions:
        media_type = str(item.get("mediaType", "")).lower()
        download_url = str(item.get("downloadURL", "")).strip()
        if media_type == "text/csv" and download_url:
            return download_url
    raise FileNotFoundError(f"No CSV downloadURL exposed for CMS dataset {CMS_HOSPITAL_DATASET_ID}")


def download_cms_hospital_general_info(
    *,
    out_csv_path: Path,
    cfg: CmsHospitalBuildConfig = CmsHospitalBuildConfig(),
) -> Path:
    url = cms_hospital_download_url(timeout_seconds=cfg.timeout_seconds)
    response = requests.get(url, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.write_bytes(response.content)
    return out_csv_path


def load_cms_hospital_general_info(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    out = df.rename(
        columns={
            "Facility ID": "facility_id",
            "Facility Name": "facility_name",
            "Address": "address",
            "City/Town": "city",
            "State": "state_abbr",
            "ZIP Code": "zip_code",
            "County/Parish": "county_name",
            "Telephone Number": "telephone",
            "Hospital Type": "hospital_type",
            "Hospital Ownership": "hospital_ownership",
            "Emergency Services": "emergency_services",
        }
    ).copy()
    keep_cols = [
        "facility_id",
        "facility_name",
        "address",
        "city",
        "state_abbr",
        "zip_code",
        "county_name",
        "telephone",
        "hospital_type",
        "hospital_ownership",
        "emergency_services",
    ]
    out = out[keep_cols].copy()
    out["facility_id"] = out["facility_id"].astype(str).str.strip()
    out["facility_name"] = out["facility_name"].astype(str).str.strip()
    out["address"] = out["address"].astype(str).str.strip()
    out["city"] = out["city"].astype(str).str.strip()
    out["state_abbr"] = out["state_abbr"].astype(str).str.strip().str.upper()
    out["zip_code"] = out["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False).fillna("")
    out["county_name"] = out["county_name"].astype(str).str.strip()
    out["hospital_type"] = out["hospital_type"].astype(str).str.strip()
    out["hospital_ownership"] = out["hospital_ownership"].astype(str).str.strip()
    out["emergency_services"] = out["emergency_services"].astype(str).str.strip()
    out = out[out["facility_id"].ne("") & out["address"].ne("") & out["city"].ne("") & out["state_abbr"].ne("")].copy()
    return out.drop_duplicates(["facility_id"], keep="first").reset_index(drop=True)


def _geocode_batch(
    batch_df: pd.DataFrame,
    *,
    cfg: CmsHospitalBuildConfig,
) -> pd.DataFrame:
    lines = []
    for row in batch_df.itertuples(index=False):
        lines.append(",".join([str(row.facility_id), str(row.address), str(row.city), str(row.state_abbr), str(row.zip_code)]))
    payload = "\n".join(lines) + ("\n" if lines else "")
    files = {"addressFile": ("cms_hospital_batch.csv", payload.encode("utf-8"), "text/csv")}
    data = {"benchmark": cfg.benchmark, "vintage": cfg.vintage}
    response = requests.post(CENSUS_BATCH_GEOCODER_URL, data=data, files=files, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    cols = [
        "facility_id",
        "input_address",
        "match_status",
        "match_type",
        "matched_address",
        "coordinates",
        "tiger_line_id",
        "tiger_side",
    ]
    return pd.read_csv(StringIO(response.text), header=None, names=cols, dtype=str).fillna("")


def geocode_cms_hospitals(
    hospitals: pd.DataFrame,
    *,
    cfg: CmsHospitalBuildConfig = CmsHospitalBuildConfig(),
) -> pd.DataFrame:
    batches: list[pd.DataFrame] = []
    for start in range(0, len(hospitals), int(cfg.batch_size)):
        batch = hospitals.iloc[start : start + int(cfg.batch_size)][["facility_id", "address", "city", "state_abbr", "zip_code"]].copy()
        batches.append(_geocode_batch(batch, cfg=cfg))
    if not batches:
        return pd.DataFrame(
            columns=[
                "facility_id",
                "input_address",
                "match_status",
                "match_type",
                "matched_address",
                "coordinates",
                "tiger_line_id",
                "tiger_side",
                "longitude",
                "latitude",
            ]
        )
    out = pd.concat(batches, ignore_index=True)
    coords = out["coordinates"].str.split(",", n=1, expand=True)
    out["longitude"] = pd.to_numeric(coords[0], errors="coerce")
    out["latitude"] = pd.to_numeric(coords[1], errors="coerce")
    return out


def build_cms_hospital_points(
    *,
    raw_csv_path: Path,
    cfg: CmsHospitalBuildConfig = CmsHospitalBuildConfig(),
) -> gpd.GeoDataFrame:
    hospitals = load_cms_hospital_general_info(raw_csv_path)
    geocoded = geocode_cms_hospitals(hospitals, cfg=cfg)
    merged = hospitals.merge(geocoded, on="facility_id", how="left")
    merged["geocode_match"] = merged["match_status"].eq("Match")
    merged = merged[merged["geocode_match"] & merged["longitude"].notna() & merged["latitude"].notna()].copy()
    merged["is_acute_care_hospital"] = merged["hospital_type"].str.contains("Acute Care", case=False, na=False)
    merged["is_critical_access_hospital"] = merged["hospital_type"].str.contains("Critical Access", case=False, na=False)
    merged["is_psychiatric_hospital"] = merged["hospital_type"].str.contains("Psychiatric", case=False, na=False)
    merged["is_childrens_hospital"] = merged["hospital_type"].str.contains("Childrens", case=False, na=False)
    merged["is_rural_emergency_hospital"] = merged["hospital_type"].str.contains("Rural Emergency", case=False, na=False)
    merged["has_emergency_services"] = merged["emergency_services"].str.upper().eq("YES")
    geometry = gpd.points_from_xy(merged["longitude"], merged["latitude"], crs=cfg.point_crs)
    return gpd.GeoDataFrame(merged, geometry=geometry, crs=cfg.point_crs)


def aggregate_cms_hospital_points_to_block_groups(
    points: gpd.GeoDataFrame,
    block_groups: gpd.GeoDataFrame,
) -> pd.DataFrame:
    group_cols = ["bg_id", "tract_id", "state_fips", "county_fips"]
    base = block_groups[group_cols].drop_duplicates().copy()
    metric_cols = [
        "hospital_count",
        "acute_care_hospital_count",
        "critical_access_hospital_count",
        "psychiatric_hospital_count",
        "childrens_hospital_count",
        "rural_emergency_hospital_count",
        "emergency_hospital_count",
        "hospital_present",
        "acute_care_hospital_present",
        "emergency_hospital_present",
        "nearest_hospital_km",
        "nearest_acute_care_hospital_km",
        "nearest_emergency_hospital_km",
        "hospital_within_2km",
        "acute_care_hospital_within_2km",
        "emergency_hospital_within_2km",
    ]
    if points.empty or block_groups.empty:
        for col in metric_cols:
            base[col] = 0.0
        return base.sort_values(group_cols, kind="stable").reset_index(drop=True)

    if points.crs != block_groups.crs:
        points = points.to_crs(block_groups.crs)

    joined = gpd.sjoin(
        points[
            [
                "facility_id",
                "is_acute_care_hospital",
                "is_critical_access_hospital",
                "is_psychiatric_hospital",
                "is_childrens_hospital",
                "is_rural_emergency_hospital",
                "has_emergency_services",
                "geometry",
            ]
        ],
        block_groups[group_cols + ["geometry"]],
        how="inner",
        predicate="within",
        lsuffix="hospital",
        rsuffix="bg",
    )
    if joined.empty:
        for col in metric_cols:
            base[col] = 0.0
        return base.sort_values(group_cols, kind="stable").reset_index(drop=True)

    joined = joined.rename(columns={"state_fips_bg": "state_fips", "county_fips_bg": "county_fips"}).copy()
    totals = joined.groupby(group_cols, dropna=False).agg(
        hospital_count=("facility_id", "count"),
        acute_care_hospital_count=("is_acute_care_hospital", "sum"),
        critical_access_hospital_count=("is_critical_access_hospital", "sum"),
        psychiatric_hospital_count=("is_psychiatric_hospital", "sum"),
        childrens_hospital_count=("is_childrens_hospital", "sum"),
        rural_emergency_hospital_count=("is_rural_emergency_hospital", "sum"),
        emergency_hospital_count=("has_emergency_services", "sum"),
    ).reset_index()
    out = base.merge(totals, on=group_cols, how="left")
    for col in [
        "hospital_count",
        "acute_care_hospital_count",
        "critical_access_hospital_count",
        "psychiatric_hospital_count",
        "childrens_hospital_count",
        "rural_emergency_hospital_count",
        "emergency_hospital_count",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["hospital_present"] = out["hospital_count"].gt(0).astype(float)
    out["acute_care_hospital_present"] = out["acute_care_hospital_count"].gt(0).astype(float)
    out["emergency_hospital_present"] = out["emergency_hospital_count"].gt(0).astype(float)

    bg_centroids = block_groups[group_cols + ["geometry"]].drop_duplicates().to_crs(LENGTH_CRS).copy()
    bg_centroids["geometry"] = bg_centroids.geometry.centroid
    points_metric = points.to_crs(LENGTH_CRS)

    def _nearest_distance_km(points_subset: gpd.GeoDataFrame, out_col: str) -> None:
        if points_subset.empty:
            out[out_col] = pd.NA
            return
        nearest = gpd.sjoin_nearest(
            bg_centroids,
            points_subset[["geometry"]],
            how="left",
            distance_col="distance_m",
        )
        nearest = nearest[group_cols + ["distance_m"]].groupby(group_cols, dropna=False)["distance_m"].min().reset_index()
        nearest[out_col] = pd.to_numeric(nearest["distance_m"], errors="coerce") / 1000.0
        out[out_col] = out.merge(nearest[group_cols + [out_col]], on=group_cols, how="left")[out_col]

    _nearest_distance_km(points_metric, "nearest_hospital_km")
    _nearest_distance_km(points_metric[points_metric["is_acute_care_hospital"]].copy(), "nearest_acute_care_hospital_km")
    _nearest_distance_km(points_metric[points_metric["has_emergency_services"]].copy(), "nearest_emergency_hospital_km")
    out["hospital_within_2km"] = pd.to_numeric(out["nearest_hospital_km"], errors="coerce").le(2.0).fillna(False).astype(float)
    out["acute_care_hospital_within_2km"] = pd.to_numeric(out["nearest_acute_care_hospital_km"], errors="coerce").le(2.0).fillna(False).astype(float)
    out["emergency_hospital_within_2km"] = pd.to_numeric(out["nearest_emergency_hospital_km"], errors="coerce").le(2.0).fillna(False).astype(float)
    return out.sort_values(group_cols, kind="stable").reset_index(drop=True)


def build_state_block_group_cms_hospital_metrics(
    *,
    state_fips: str,
    bg_zip: Path,
    points: gpd.GeoDataFrame,
) -> pd.DataFrame:
    state = str(state_fips).zfill(2)
    block_groups = load_block_groups(bg_zip)
    if not block_groups.empty:
        block_groups = block_groups[block_groups["state_fips"].eq(state)].copy()
    return aggregate_cms_hospital_points_to_block_groups(
        points=points[points["state_fips"].eq(state)].copy(),
        block_groups=block_groups,
    )


STATE_FIPS_FROM_ABBR = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "PR": "72",
}


def build_cms_hospital_frame_by_block_group(
    *,
    bg_dir: Path,
    raw_csv_path: Path,
    state_fips_values: list[str] | None = None,
    cfg: CmsHospitalBuildConfig = CmsHospitalBuildConfig(),
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    points = build_cms_hospital_points(raw_csv_path=raw_csv_path, cfg=cfg)
    points["state_fips"] = points["state_abbr"].map(STATE_FIPS_FROM_ABBR).astype("string")
    states = state_fips_values if state_fips_values is not None else infer_state_fips_from_bg_dir(bg_dir, year=2020)
    frames: list[pd.DataFrame] = []
    for state_fips in states:
        state = str(state_fips).zfill(2)
        bg_zip = bg_dir / f"tl_2020_{state}_bg.zip"
        if not bg_zip.exists():
            continue
        frames.append(build_state_block_group_cms_hospital_metrics(state_fips=state, bg_zip=bg_zip, points=points))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return points, out
