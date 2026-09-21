from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests


NCES_EDGE_BASE_URL = "https://nces.ed.gov/programs/edge/data"


@dataclass(frozen=True)
class NcesEdgeDataset:
    key: str
    zip_filename: str
    member_xlsx: str
    source_label: str
    id_col: str
    extra_keep_cols: tuple[str, ...] = ()


NCES_EDGE_DATASETS: dict[str, NcesEdgeDataset] = {
    "public_school": NcesEdgeDataset(
        key="public_school",
        zip_filename="EDGE_GEOCODE_PUBLICSCH_2425.zip",
        member_xlsx="EDGE_GEOCODE_PUBLICSCH_2425.xlsx",
        source_label="nces_edge_public_school",
        id_col="NCESSCH",
        extra_keep_cols=("LEAID",),
    ),
    "postsecondary": NcesEdgeDataset(
        key="postsecondary",
        zip_filename="EDGE_GEOCODE_POSTSECSCH_2425.zip",
        member_xlsx="EDGE_GEOCODE_POSTSECSCH_2425.xlsx",
        source_label="nces_edge_postsecondary",
        id_col="UNITID",
    ),
}


def download_nces_edge_zip(
    *,
    dataset: NcesEdgeDataset,
    out_dir: Path,
    overwrite: bool = False,
    timeout_seconds: int = 120,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / dataset.zip_filename
    if out_path.exists() and not overwrite:
        return out_path
    response = requests.get(f"{NCES_EDGE_BASE_URL}/{dataset.zip_filename}", timeout=timeout_seconds)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


def load_nces_edge_locations(*, zip_path: Path, dataset: NcesEdgeDataset) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(dataset.member_xlsx) as f:
            frame = pd.read_excel(io.BytesIO(f.read()))

    keep_cols = [
        dataset.id_col,
        *dataset.extra_keep_cols,
        "NAME",
        "STREET",
        "CITY",
        "STATE",
        "ZIP",
        "STFIP",
        "CNTY",
        "NMCNTY",
        "LOCALE",
        "LAT",
        "LON",
        "CBSA",
        "NMCBSA",
        "CBSATYPE",
        "CSA",
        "NMCSA",
        "CD",
        "SLDL",
        "SLDU",
        "SCHOOLYEAR",
    ]
    keep_cols = [col for col in keep_cols if col in frame.columns]
    out = frame[keep_cols].copy()
    rename_map = {
        dataset.id_col: "institution_id",
        "LEAID": "district_id",
        "NAME": "institution_name",
        "STREET": "street",
        "CITY": "city",
        "STATE": "state_abbr",
        "ZIP": "zip",
        "STFIP": "state_fips",
        "CNTY": "county_geoid",
        "NMCNTY": "county_name",
        "LOCALE": "locale_code",
        "LAT": "lat",
        "LON": "lon",
        "CBSA": "cbsa",
        "NMCBSA": "cbsa_name",
        "CBSATYPE": "cbsa_type",
        "CSA": "csa",
        "NMCSA": "csa_name",
        "CD": "congressional_district",
        "SLDL": "state_legislative_district_lower",
        "SLDU": "state_legislative_district_upper",
        "SCHOOLYEAR": "school_year",
    }
    out = out.rename(columns=rename_map)
    out["source"] = dataset.source_label
    out["institution_id"] = out["institution_id"].astype("string").str.strip()
    if "district_id" in out.columns:
        out["district_id"] = out["district_id"].astype("string").str.strip()
    if "zip" in out.columns:
        out["zip"] = out["zip"].astype("string").str.strip()
    out["state_abbr"] = out["state_abbr"].astype("string").str.upper()
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["county_geoid"] = out["county_geoid"].astype("string").str.zfill(5)
    out["county_fips"] = out["county_geoid"].str[-3:]
    for col in ["lat", "lon", "locale_code", "cbsa", "csa", "congressional_district", "state_legislative_district_lower", "state_legislative_district_upper"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["bg_id"] = pd.NA
    out["tract_id"] = pd.NA
    return out


def _load_block_groups_for_state(*, bg_zip: Path) -> gpd.GeoDataFrame:
    bg = gpd.read_file(f"zip://{bg_zip}")[
        ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE", "geometry"]
    ].copy()
    bg["bg_id"] = bg["GEOID"].astype("string").str.zfill(12)
    bg["tract_id"] = bg["bg_id"].str.slice(0, 11)
    bg["state_fips"] = bg["STATEFP"].astype("string").str.zfill(2)
    bg["county_fips"] = bg["COUNTYFP"].astype("string").str.zfill(3)
    return bg[["bg_id", "tract_id", "state_fips", "county_fips", "geometry"]].drop_duplicates("bg_id")


def _assign_points_to_block_groups(
    *,
    locations: pd.DataFrame,
    bg: gpd.GeoDataFrame,
    count_col: str,
) -> pd.DataFrame:
    state = str(bg["state_fips"].iloc[0]).zfill(2)
    subset = locations[locations["state_fips"].astype("string").str.zfill(2).eq(state)].copy()
    subset = subset[subset["lat"].notna() & subset["lon"].notna()].copy()
    if subset.empty:
        return bg[["bg_id", "tract_id", "state_fips", "county_fips"]].assign(**{count_col: 0})
    points = gpd.GeoDataFrame(
        subset[["institution_id", "lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(subset["lon"], subset["lat"]),
        crs="EPSG:4326",
    ).to_crs(bg.crs)
    joined = gpd.sjoin(
        points[["institution_id", "geometry"]],
        bg[["bg_id", "tract_id", "state_fips", "county_fips", "geometry"]],
        how="left",
        predicate="within",
    )
    counts = (
        joined.groupby(["bg_id", "tract_id", "state_fips", "county_fips"], dropna=False)
        .size()
        .rename(count_col)
        .reset_index()
    )
    out = bg[["bg_id", "tract_id", "state_fips", "county_fips"]].merge(
        counts,
        on=["bg_id", "tract_id", "state_fips", "county_fips"],
        how="left",
    )
    out[count_col] = pd.to_numeric(out[count_col], errors="coerce").fillna(0).astype(int)
    return out


def _nearest_distance_to_points(
    *,
    locations: pd.DataFrame,
    bg: gpd.GeoDataFrame,
    distance_col: str,
    within_col: str,
    within_km: float,
) -> pd.DataFrame:
    state = str(bg["state_fips"].iloc[0]).zfill(2)
    subset = locations[locations["state_fips"].astype("string").str.zfill(2).eq(state)].copy()
    subset = subset[subset["lat"].notna() & subset["lon"].notna()].copy()
    base = bg[["bg_id", "tract_id", "state_fips", "county_fips"]].copy()
    if subset.empty:
        base[distance_col] = np.nan
        base[within_col] = 0.0
        return base

    bg_projected = bg[["bg_id", "tract_id", "state_fips", "county_fips", "geometry"]].copy().to_crs("EPSG:5070")
    bg_points = gpd.GeoDataFrame(
        bg_projected.drop(columns=["geometry"]).copy(),
        geometry=bg_projected.geometry.centroid,
        crs="EPSG:5070",
    )
    point_gdf = gpd.GeoDataFrame(
        subset[["institution_id"]].copy(),
        geometry=gpd.points_from_xy(subset["lon"], subset["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    nearest = gpd.sjoin_nearest(
        bg_points,
        point_gdf[["institution_id", "geometry"]],
        how="left",
        distance_col="_nearest_meters",
    )
    out = nearest[["bg_id", "tract_id", "state_fips", "county_fips", "_nearest_meters"]].copy()
    out = (
        out.groupby(["bg_id", "tract_id", "state_fips", "county_fips"], dropna=False, as_index=False)["_nearest_meters"]
        .min()
    )
    out[distance_col] = pd.to_numeric(out["_nearest_meters"], errors="coerce") / 1000.0
    out[within_col] = out[distance_col].le(float(within_km)).fillna(False).astype(float)
    return out.drop(columns=["_nearest_meters"])


def build_block_group_education_anchor_counts(
    *,
    public_school_parquet: Path,
    postsecondary_parquet: Path,
    bg_dir: Path,
) -> pd.DataFrame:
    public = pd.read_parquet(public_school_parquet)
    postsecondary = pd.read_parquet(postsecondary_parquet)
    frames: list[pd.DataFrame] = []
    for bg_zip in sorted(bg_dir.glob("tl_2020_*_bg.zip")):
        state_fips = bg_zip.stem.split("_")[2]
        bg = _load_block_groups_for_state(bg_zip=bg_zip)
        public_counts = _assign_points_to_block_groups(
            locations=public,
            bg=bg,
            count_col="public_school_count",
        )
        postsecondary_counts = _assign_points_to_block_groups(
            locations=postsecondary,
            bg=bg,
            count_col="postsecondary_count",
        )
        merged = public_counts.merge(
            postsecondary_counts,
            on=["bg_id", "tract_id", "state_fips", "county_fips"],
            how="outer",
        )
        merged["public_school_count"] = pd.to_numeric(merged["public_school_count"], errors="coerce").fillna(0).astype(int)
        merged["postsecondary_count"] = pd.to_numeric(merged["postsecondary_count"], errors="coerce").fillna(0).astype(int)
        merged["education_anchor_total"] = merged["public_school_count"] + merged["postsecondary_count"]
        merged["public_school_present"] = merged["public_school_count"].gt(0).astype(float)
        merged["postsecondary_present"] = merged["postsecondary_count"].gt(0).astype(float)
        nearest_public = _nearest_distance_to_points(
            locations=public,
            bg=bg,
            distance_col="nearest_public_school_km",
            within_col="public_school_within_1km",
            within_km=1.0,
        )
        nearest_postsecondary = _nearest_distance_to_points(
            locations=postsecondary,
            bg=bg,
            distance_col="nearest_postsecondary_km",
            within_col="postsecondary_within_2km",
            within_km=2.0,
        )
        merged = merged.merge(
            nearest_public,
            on=["bg_id", "tract_id", "state_fips", "county_fips"],
            how="left",
        )
        merged = merged.merge(
            nearest_postsecondary,
            on=["bg_id", "tract_id", "state_fips", "county_fips"],
            how="left",
        )
        frames.append(merged)
    if not frames:
        return pd.DataFrame(
            columns=[
                "bg_id",
                "tract_id",
                "state_fips",
                "county_fips",
                "public_school_count",
                "postsecondary_count",
                "education_anchor_total",
                "public_school_present",
                "postsecondary_present",
                "nearest_public_school_km",
                "public_school_within_1km",
                "nearest_postsecondary_km",
                "postsecondary_within_2km",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    out["tract_id"] = out["tract_id"].astype("string").str.zfill(11)
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["county_fips"] = out["county_fips"].astype("string").str.zfill(3)
    return out
