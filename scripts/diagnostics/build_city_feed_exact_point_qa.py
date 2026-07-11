from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DIAG_ROOT = REPO_ROOT / "scripts" / "diagnostics"
for path in (SRC_ROOT, DIAG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crimerisk.city_feed_quarantine import flag_quarantined_coordinates  # noqa: E402
from crimerisk.city_incidents import (  # noqa: E402
    BOSTON_LARCENY_CODES,
    BOSTON_MOTOR_VEHICLE_THEFT_CODES,
    CHICAGO_BOUNDS,
    SEATTLE_OFFENSE_MAP,
    SEATTLE_BOUNDS,
    _load_chicago_categories,
    _load_chicago_raw_snapshot,
    _load_nyc_categories,
    _load_nyc_raw_snapshot,
    _load_seattle_raw_snapshot,
    _map_nyc_offenses,
    _parse_nyc_datetime,
)
from build_next_phase_validation_shares import (  # noqa: E402
    CHARLOTTE_OFFENSE_MAP,
    DALLAS_OFFENSE_MAP,
    MILWAUKEE_OFFENSE_MAP,
    MONTGOMERY_COUNTY_OFFENSE_MAP,
    SAN_DIEGO_OFFENSE_MAP,
    ST_LOUIS_OFFENSE_MAP,
    TUCSON_OFFENSE_MAP,
    _cincinnati_offense_from_row,
    _dedupe_murder_events_for_share,
    _kansas_city_offense_from_row,
)


OFFENSES_7 = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]
# Tripwire thresholds (kept in sync with validate_release_outputs.py):
# flag an exact-point concentration only once the city/offense has a meaningful
# located base, so tiny-denominator artifacts do not fire.
EXACT_POINT_SHARE_MAX = 0.005
EXACT_POINT_MIN_LOCATED_COUNT = 100
# Per-point minimum deduped-incident floor: a single coordinate must carry at least
# this many deduped incidents of the offense to be flagged, alongside the >=0.5%
# share and >=100 city/offense located-count floors. Below it, a 1-4 incident point
# trivially clears 0.5% on a low-volume violent denominator (e.g. Baton Rouge murder
# 1/145, Indianapolis murder 2/166, packet-city robbery 3-5 counts) with no masking
# signature; the floor drops those artifacts without hand-listing them.
EXACT_POINT_MIN_POINT_COUNT = 5
CITY_NAME_TO_KEY = {
    "Austin": "austin",
    "Baltimore": "baltimore",
    "Boston": "boston",
    "Charlotte": "charlotte",
    "Chicago": "chicago",
    "Cincinnati": "cincinnati",
    "Dallas": "dallas",
    "Denver": "denver",
    "Durham": "durham",
    "Kansas City": "kansas_city",
    "Mesa": "mesa",
    "Milwaukee": "milwaukee",
    "Minneapolis": "minneapolis",
    "Montgomery County, MD": "montgomery_county",
    "New York": "new_york",
    "Philadelphia": "philadelphia",
    "San Diego": "san_diego",
    "San Francisco": "san_francisco",
    "Seattle": "seattle",
    "St. Louis": "st_louis",
    "Tucson": "tucson",
    "Washington": "washington_dc",
    # Gate-17 batch-A onboarded packet cities (points from packet normalized_incidents).
    "Atlanta, Georgia": "atlanta_ga",
    "Baton Rouge, Louisiana": "baton_rouge_la",
    "Cleveland, Ohio": "cleveland_oh",
    "Colorado Springs, Colorado": "colorado_springs_co",
    "Fort Worth, Texas": "fort_worth_tx",
    "Indianapolis, Indiana": "indianapolis_in",
    "Omaha, Nebraska": "omaha_ne",
    "Sacramento, California": "sacramento_ca",
}
CITY_KEY_TO_NAME = {value: key for key, value in CITY_NAME_TO_KEY.items()}
CI = REPO_ROOT / "data" / "city-incidents"
NPV = REPO_ROOT / "state" / "modeling" / "inputs" / "next_phase_validation"
GATE17_PACKETS_DIR = REPO_ROOT / "state" / "review" / "packets" / "city"
# Incident/report-id column per packet city, used for the same multi-charge event
# dedupe as the feed-city loaders.
GATE17_PACKET_INCIDENT_ID = {
    "atlanta_ga": "IncidentNumber",
    "baton_rouge_la": "incident_number",
    "cleveland_oh": "CaseNumber",
    "colorado_springs_co": "casenumber",
    "fort_worth_tx": "Case_No",
    "indianapolis_in": "CaseNum",
    "omaha_ne": "RB",
    "sacramento_ca": "Record_ID",
}


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.DataFrame(columns=columns or [])


_RECORD_COLUMNS = ["city_key", "city_name", "offense", "latitude", "longitude", "weight", "incident_id"]


def _records(
    df: pd.DataFrame,
    *,
    city_key: str,
    city_name: str,
    offense_col: str = "offense",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    weight_col: str | None = None,
    incident_id_col: str | None = None,
) -> pd.DataFrame:
    if df.empty or offense_col not in df.columns or lat_col not in df.columns or lon_col not in df.columns:
        return pd.DataFrame(columns=_RECORD_COLUMNS)
    keep_cols = [offense_col, lat_col, lon_col]
    if weight_col and weight_col in df.columns:
        keep_cols.append(weight_col)
    if incident_id_col and incident_id_col in df.columns:
        keep_cols.append(incident_id_col)
    out = df[keep_cols].copy()
    out = out.rename(columns={offense_col: "offense", lat_col: "latitude", lon_col: "longitude"})
    out["offense"] = out["offense"].astype("string")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out[out["offense"].isin(OFFENSES_7) & out["latitude"].notna() & out["longitude"].notna()].copy()
    if out.empty:
        return pd.DataFrame(columns=_RECORD_COLUMNS)
    if weight_col and weight_col in df.columns:
        out["weight"] = pd.to_numeric(out.get(weight_col), errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        out["weight"] = 1.0
    # Incident/report id, used to dedupe multi-victim/multi-charge rows before the
    # point-share computation so a single multi-charge event (e.g. St. Louis rape
    # 1 incident -> 16 rows, KC rape 2 incidents -> 8 rows) is not counted as many
    # located rows at one point. Rows with no id are treated as distinct events.
    if incident_id_col and incident_id_col in df.columns:
        out["incident_id"] = out[incident_id_col].astype("string").str.strip()
        blank = out["incident_id"].isna() | out["incident_id"].isin(["", "<NA>"])
        out.loc[blank, "incident_id"] = pd.NA
    else:
        out["incident_id"] = pd.NA
    out["city_key"] = city_key
    out["city_name"] = city_name
    return out[_RECORD_COLUMNS]


def _canonical_boston() -> pd.DataFrame:
    raw = _read_parquet(CI / "boston/parsed/boston_minimal_snapshot_2019_2024.parquet")
    if raw.empty:
        return pd.DataFrame()
    raw["year_num"] = pd.to_numeric(raw["year"], errors="coerce")
    raw = raw[raw["year_num"].between(2019, 2024)].copy()
    raw["offense_code"] = raw["offense_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    raw["offense"] = pd.NA
    raw.loc[raw["offense_code"].isin(BOSTON_LARCENY_CODES), "offense"] = "larceny"
    raw.loc[raw["offense_code"].isin(BOSTON_MOTOR_VEHICLE_THEFT_CODES), "offense"] = "motor_vehicle_theft"
    raw = raw[raw["offense"].notna()].copy()
    raw["offense_row_id"] = raw["incident_number"].astype(str).str.strip() + "::" + raw["offense"].astype(str)
    raw = raw.drop_duplicates("offense_row_id")
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw = raw.loc[~flag_quarantined_coordinates(raw, city_key="boston", offense_col="offense")].copy()
    return _records(raw, city_key="boston", city_name="Boston", incident_id_col="incident_number")


def _canonical_philadelphia() -> pd.DataFrame:
    raw = _read_parquet(CI / "philadelphia/parsed/philadelphia_minimal_snapshot_2018_2024.parquet")
    if raw.empty:
        return pd.DataFrame()
    raw["ucr_general"] = raw["ucr_general"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    raw["offense"] = pd.NA
    raw.loc[raw["ucr_general"].eq("400"), "offense"] = "aggravated_assault"
    raw.loc[raw["ucr_general"].eq("500"), "offense"] = "burglary"
    raw = raw[raw["offense"].notna()].copy()
    raw["offense_row_id"] = raw["objectid"].astype(str).str.strip() + "::" + raw["offense"].astype(str)
    raw = raw.drop_duplicates("offense_row_id")
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw = raw.loc[~flag_quarantined_coordinates(raw, city_key="philadelphia", offense_col="offense")].copy()
    return _records(raw, city_key="philadelphia", city_name="Philadelphia", incident_id_col="objectid")


def _canonical_nyc() -> pd.DataFrame:
    cats_path = REPO_ROOT / "configs" / "city_incident_categories" / "categories_new_york.csv"
    if not cats_path.exists():
        return pd.DataFrame()
    raw = _load_nyc_raw_snapshot(raw_dir=CI / "nyc", start_year=2018, end_year=2024)
    if raw.empty:
        return pd.DataFrame()
    raw["incident_dt"] = _parse_nyc_datetime(raw)
    raw["year"] = raw["incident_dt"].dt.year
    raw = raw[raw["year"].between(2018, 2024, inclusive="both")].copy()
    cats = _load_nyc_categories(cats_path)
    raw["ofns_desc"] = raw["ofns_desc"].astype(str).str.strip()
    raw["pd_desc"] = raw["pd_desc"].astype(str).str.strip()
    mapped = _map_nyc_offenses(raw, cats)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    mapped = mapped[
        mapped["latitude"].between(40.45, 40.95, inclusive="both")
        & mapped["longitude"].between(-74.30, -73.65, inclusive="both")
    ].copy()
    mapped["offense_row_id"] = mapped["cmplnt_num"].astype(str).str.strip() + "::" + mapped["offense"].astype(str)
    mapped = mapped.drop_duplicates("cmplnt_num")
    mapped = mapped.loc[
        ~flag_quarantined_coordinates(mapped, city_key="new_york", offense_col="offense")
    ].copy()
    return _records(mapped, city_key="new_york", city_name="New York", incident_id_col="cmplnt_num")


def _canonical_chicago() -> pd.DataFrame:
    cats_path = REPO_ROOT / "configs" / "city_incident_categories" / "categories_Chicago.csv"
    if not cats_path.exists():
        return pd.DataFrame()
    raw = _load_chicago_raw_snapshot(raw_dir=CI / "chicago", start_year=2018, end_year=2024)
    if raw.empty:
        return pd.DataFrame()
    raw["incident_dt"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["year"] = raw["incident_dt"].dt.year
    raw = raw[raw["year"].between(2018, 2024, inclusive="both")].copy()
    cats = _load_chicago_categories(cats_path)
    raw["primary_type"] = raw["primary_type"].astype(str).str.strip()
    raw["description"] = raw["description"].astype(str).str.strip()
    mapped = raw.merge(cats, on=["primary_type", "description"], how="inner")
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    mapped = mapped[
        mapped["latitude"].between(CHICAGO_BOUNDS["min_latitude"], CHICAGO_BOUNDS["max_latitude"], inclusive="both")
        & mapped["longitude"].between(CHICAGO_BOUNDS["min_longitude"], CHICAGO_BOUNDS["max_longitude"], inclusive="both")
    ].copy()
    mapped["offense_row_id"] = mapped["id"].astype(str).str.strip() + "::" + mapped["offense"].astype(str)
    mapped = mapped.drop_duplicates("id")
    mapped = mapped.loc[
        ~flag_quarantined_coordinates(mapped, city_key="chicago", offense_col="offense")
    ].copy()
    return _records(mapped, city_key="chicago", city_name="Chicago", incident_id_col="id")


def _canonical_seattle() -> pd.DataFrame:
    raw = _load_seattle_raw_snapshot(raw_dir=CI / "seattle", start_year=2018, end_year=2024)
    if raw.empty:
        return pd.DataFrame()
    raw["incident_dt"] = pd.to_datetime(raw["offense_date"], errors="coerce")
    report_dt = pd.to_datetime(raw["report_datetime"], errors="coerce")
    raw["incident_dt"] = raw["incident_dt"].fillna(report_dt)
    raw["year"] = raw["incident_dt"].dt.year
    raw = raw[raw["year"].between(2018, 2024, inclusive="both")].copy()
    raw["offense"] = raw["nibrs_offense_code"].astype(str).str.strip().str.upper().map(SEATTLE_OFFENSE_MAP)
    raw = raw[raw["offense"].notna()].copy()
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw = raw[
        raw["latitude"].between(SEATTLE_BOUNDS["min_latitude"], SEATTLE_BOUNDS["max_latitude"], inclusive="both")
        & raw["longitude"].between(SEATTLE_BOUNDS["min_longitude"], SEATTLE_BOUNDS["max_longitude"], inclusive="both")
    ].copy()
    raw = raw.drop_duplicates("offense_id")
    return _records(raw, city_key="seattle", city_name="Seattle", incident_id_col="offense_id")


def _next_phase_kansas_city() -> pd.DataFrame:
    raw = _read_parquet(NPV / "kansas_city_mo_kcpd_crime_2024.parquet")
    if raw.empty:
        return pd.DataFrame()
    raw["year"] = 2024
    raw["ibrs"] = raw["ibrs"].astype("string").str.strip().str.upper()
    raw["offense"] = raw.apply(_kansas_city_offense_from_row, axis=1)
    raw = raw[raw["offense"].notna()].copy()
    raw["offense_row_id"] = raw.get(":id", pd.Series(raw.index, index=raw.index)).astype("string").str.strip()
    raw = _dedupe_murder_events_for_share(raw, fallback_columns=("reported_date", "from_date", "latitude", "longitude"))
    raw = raw.drop_duplicates("offense_row_id")
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw = raw[
        raw["latitude"].between(38.75, 39.45, inclusive="both")
        & raw["longitude"].between(-95.0, -94.25, inclusive="both")
    ].copy()
    kc_incident_id = "report_no" if "report_no" in raw.columns else ":id"
    return _records(raw, city_key="kansas_city", city_name="Kansas City", incident_id_col=kc_incident_id)


def _next_phase_simple_socrata_or_csv(
    path: Path,
    *,
    city_key: str,
    city_name: str,
    code_col: str,
    lat_col: str,
    lon_col: str,
    offense_map: dict[str, str],
    id_col: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    raw = _read_parquet(path)
    if raw.empty:
        return pd.DataFrame()
    raw["offense"] = raw[code_col].astype("string").str.strip().str.upper().map(offense_map)
    raw = raw[raw["offense"].notna()].copy()
    if id_col and id_col in raw.columns:
        raw["offense_row_id"] = raw[id_col].astype("string").str.strip()
    else:
        raw["offense_row_id"] = raw.index.astype(str) + "::" + raw["offense"].astype(str)
    raw = raw.drop_duplicates("offense_row_id")
    raw["latitude"] = pd.to_numeric(raw[lat_col], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw[lon_col], errors="coerce")
    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
        raw = raw[
            raw["latitude"].between(lat_min, lat_max, inclusive="both")
            & raw["longitude"].between(lon_min, lon_max, inclusive="both")
        ].copy()
    raw = raw.loc[
        ~flag_quarantined_coordinates(raw, city_key=city_key, offense_col="offense")
    ].copy()
    return _records(
        raw,
        city_key=city_key,
        city_name=city_name,
        incident_id_col=id_col if (id_col and id_col in raw.columns) else None,
    )


def _next_phase_cincinnati() -> pd.DataFrame:
    raw = _read_parquet(NPV / "cincinnati_pdi_crime_incidents_2024.parquet")
    if raw.empty:
        return pd.DataFrame()
    raw["ucr_group"] = raw["ucr_group"].astype("string").str.strip()
    raw["offense_label"] = raw["offense"].astype("string").str.strip()
    raw["offense"] = raw.apply(_cincinnati_offense_from_row, axis=1)
    raw = raw[raw["offense"].notna()].copy()
    raw["offense_row_id"] = raw["instanceid"].astype("string").str.strip()
    raw = raw.drop_duplicates("offense_row_id")
    return _records(raw, city_key="cincinnati", city_name="Cincinnati", lat_col="latitude_x", lon_col="longitude_x", incident_id_col="instanceid")


def _next_phase_st_louis() -> pd.DataFrame:
    raw = _read_parquet(NPV / "st_louis_slmpd_nibrs_crime_files_2024.parquet")
    if raw.empty:
        return pd.DataFrame()
    # Filter to 2024-dated incidents (the SLMPD "2024" file carries 2020-2023 rows,
    # e.g. IncidentNum 23059238 dated 2023-12-23); the share surface is 2024-only so
    # the tripwire denominator must match.
    incident_year = pd.to_datetime(raw["IncidentDate"], errors="coerce", format="mixed").dt.year
    raw = raw[incident_year.eq(2024)].copy()
    if raw.empty:
        return pd.DataFrame()
    raw["offense"] = raw["NIBRS"].astype("string").str.strip().str.upper().map(ST_LOUIS_OFFENSE_MAP)
    raw = raw[raw["offense"].notna()].copy()
    raw["offense_row_id"] = (
        raw[["IncidentNum", "NIBRS", "IncidentDate", "IncidentLocation", "IntersectionOtherLoc"]]
        .fillna("")
        .astype("string")
        .agg("|".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    raw = raw.drop_duplicates("offense_row_id")
    return _records(raw, city_key="st_louis", city_name="St. Louis", lat_col="Latitude", lon_col="Longitude", incident_id_col="IncidentNum")


def _next_phase_milwaukee() -> pd.DataFrame:
    raw = _read_parquet(NPV / "milwaukee_wibr_historical_2024.parquet")
    if raw.empty or "Offense_All" not in raw.columns:
        return pd.DataFrame()
    mapped = raw.copy()
    mapped["offense_code_list"] = (
        mapped["Offense_All"].astype("string").fillna("").str.replace(",", ";", regex=False).str.split(";")
    )
    mapped = mapped.explode("offense_code_list", ignore_index=True)
    mapped["nibrs_code"] = mapped["offense_code_list"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped["nibrs_code"].map(MILWAUKEE_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    if mapped.empty:
        return pd.DataFrame()
    mapped["offense_sequence"] = mapped.groupby("_id", dropna=False).cumcount() + 1
    mapped["offense_row_id"] = (
        mapped["_id"].astype("string").str.strip()
        + "::"
        + mapped["offense_sequence"].astype("string").str.strip()
        + "::"
        + mapped["offense"].astype("string").str.strip()
    )
    mapped = mapped.drop_duplicates("offense_row_id")
    return _records(mapped, city_key="milwaukee", city_name="Milwaukee", lat_col="Address_Latitude", lon_col="Address_Longitude", incident_id_col="_id")


def _next_phase_durham() -> pd.DataFrame:
    raw = _read_parquet(NPV / "durham_dpd_incidents_ucr_nibrs_reporting.parquet")
    if raw.empty:
        return pd.DataFrame()
    raw["offense"] = raw["NIBRS Code"].astype("string").str.strip().str.upper().map(CHARLOTTE_OFFENSE_MAP) if "NIBRS Code" in raw.columns else pd.NA
    raw = raw[raw["offense"].notna()].copy()
    raw["offense_row_id"] = raw.get("Case Number", pd.Series(raw.index, index=raw.index)).astype("string").str.strip() + "::" + raw["offense"].astype(str)
    raw = raw.drop_duplicates("offense_row_id")
    durham_incident_id = "Case Number" if "Case Number" in raw.columns else None
    return _records(raw, city_key="durham", city_name="Durham", lat_col="Y", lon_col="X", incident_id_col=durham_incident_id)


def _gate17_packet_points() -> list[pd.DataFrame]:
    """Point records for gate-17 batch-A packet cities from packet normalized incidents.

    Regression-fix companion (2026-07-06): the packet cities are appended to the
    validation surface by the standard rebuild, so they must face the same
    exact-point tripwire as feed cities. Quarantine flags and the incident-id
    dedupe are applied exactly as in the feed-city loaders.
    """
    frames: list[pd.DataFrame] = []
    for city_key, id_col in sorted(GATE17_PACKET_INCIDENT_ID.items()):
        raw = _read_parquet(GATE17_PACKETS_DIR / city_key / "normalized_incidents.parquet")
        if raw.empty:
            continue
        city_name = (
            str(raw["city_name"].iloc[0])
            if "city_name" in raw.columns and len(raw)
            else CITY_KEY_TO_NAME.get(city_key, city_key)
        )
        if "offense_row_id" in raw.columns:
            raw = raw.drop_duplicates("offense_row_id")
        raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
        raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
        raw = raw.loc[
            ~flag_quarantined_coordinates(raw, city_key=city_key, offense_col="offense")
        ].copy()
        frames.append(
            _records(
                raw,
                city_key=city_key,
                city_name=city_name,
                incident_id_col=id_col if id_col in raw.columns else None,
            )
        )
    return frames


def _point_records() -> pd.DataFrame:
    frames = [
        _canonical_boston(),
        _canonical_philadelphia(),
        _canonical_nyc(),
        _canonical_chicago(),
        _canonical_seattle(),
        _next_phase_kansas_city(),
        _next_phase_simple_socrata_or_csv(
            NPV / "charlotte_cmpd_incidents_2024.parquet",
            city_key="charlotte",
            city_name="Charlotte",
            code_col="HIGHEST_NIBRS_CODE",
            lat_col="LATITUDE_PUBLIC",
            lon_col="LONGITUDE_PUBLIC",
            offense_map=CHARLOTTE_OFFENSE_MAP,
            id_col="INCIDENT_REPORT_ID",
            bounds=(34.9, 35.5, -81.1, -80.5),
        ),
        _next_phase_simple_socrata_or_csv(
            NPV / "dallas_police_incidents_2024.parquet",
            city_key="dallas",
            city_name="Dallas",
            code_col="nibrs_code",
            lat_col="latitude",
            lon_col="longitude",
            offense_map=DALLAS_OFFENSE_MAP,
            id_col=":id",
            bounds=(32.5, 33.1, -97.1, -96.4),
        ),
        _next_phase_cincinnati(),
        _next_phase_simple_socrata_or_csv(
            NPV / "san_diego_police_nibrs_offenses_2024.parquet",
            city_key="san_diego",
            city_name="San Diego",
            code_col="ibr_offense",
            lat_col="latitude",
            lon_col="longitude",
            offense_map=SAN_DIEGO_OFFENSE_MAP,
            id_col="nibrs_uniq",
            bounds=(32.5, 33.2, -117.4, -116.7),
        ),
        _next_phase_st_louis(),
        _next_phase_milwaukee(),
        _next_phase_simple_socrata_or_csv(
            NPV / "montgomery_county_md_mcpd_crime_2024.parquet",
            city_key="montgomery_county",
            city_name="Montgomery County, MD",
            code_col="nibrs_code",
            lat_col="latitude",
            lon_col="longitude",
            offense_map=MONTGOMERY_COUNTY_OFFENSE_MAP,
            id_col="incident_id",
            bounds=(38.8, 39.4, -77.6, -76.8),
        ),
        _next_phase_simple_socrata_or_csv(
            NPV / "tucson_police_incidents_2024.parquet",
            city_key="tucson",
            city_name="Tucson",
            code_col="UCRsummary",
            lat_col="geometry_y",
            lon_col="geometry_x",
            offense_map=TUCSON_OFFENSE_MAP,
        ),
        _next_phase_durham(),
        *_gate17_packet_points(),
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["city_key", "city_name", "offense", "latitude", "longitude", "weight"])
    return pd.concat(frames, ignore_index=True)


def _load_active_groups(surface_path: Path) -> pd.DataFrame:
    surface = pd.read_parquet(surface_path, columns=["city_name", "offense", "incident_count", "geocode_quality_tier"])
    surface["incident_count"] = pd.to_numeric(surface["incident_count"], errors="coerce").fillna(0.0)
    groups = (
        surface.groupby(["city_name", "offense"], dropna=False)["incident_count"]
        .sum()
        .rename("surface_located_incident_count")
        .reset_index()
    )
    groups["city_key"] = groups["city_name"].map(CITY_NAME_TO_KEY).fillna(groups["city_name"].astype(str).str.lower().str.replace(r"\W+", "_", regex=True))
    return groups[["city_key", "city_name", "offense", "surface_located_incident_count"]].sort_values(["city_name", "offense"], kind="mergesort")


def _load_exceptions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["city_key", "offense", "lat", "lon"])
    df = pd.read_csv(path, dtype=str).fillna("")
    df["city_key"] = df["city_key"].astype(str).str.strip().str.lower()
    df["offense"] = df["offense"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df[df["city_key"].ne("") & df["lat"].notna() & df["lon"].notna()].copy()


def _is_reviewed_exception(row: pd.Series, exceptions: pd.DataFrame) -> bool:
    city_ex = exceptions[exceptions["city_key"].eq(str(row["city_key"]))]
    if city_ex.empty:
        return False
    city_ex = city_ex[city_ex["offense"].isin(["any", str(row["offense"])])]
    if city_ex.empty:
        return False
    lat = float(row["max_point_lat"])
    lon = float(row["max_point_lon"])
    d_lat = np.deg2rad(city_ex["lat"].to_numpy(dtype=float) - lat)
    d_lon = np.deg2rad(city_ex["lon"].to_numpy(dtype=float) - lon)
    lat1 = np.deg2rad(lat)
    lat2 = np.deg2rad(city_ex["lat"].to_numpy(dtype=float))
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2.0) ** 2
    dist = 2.0 * 6_371_008.8 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return bool((dist <= 35.0).any())


def build_qa(*, surface_path: Path, exception_path: Path) -> pd.DataFrame:
    active = _load_active_groups(surface_path)
    points = _point_records()
    exceptions = _load_exceptions(exception_path)
    rows: list[dict[str, Any]] = []
    for group in active.itertuples(index=False):
        city_key = str(group.city_key)
        city_name = str(group.city_name)
        offense = str(group.offense)
        surface_total = float(group.surface_located_incident_count)
        subset = points[points["city_key"].eq(city_key) & points["offense"].eq(offense)].copy()
        if subset.empty:
            rows.append(
                {
                    "city_key": city_key,
                    "city_name": city_name,
                    "offense": offense,
                    "located_incident_count": surface_total,
                    "max_point_count": 0.0,
                    "max_point_share": 0.0,
                    "max_point_lat": np.nan,
                    "max_point_lon": np.nan,
                    "point_key": "",
                    "reviewed_exception": False,
                    "qa_status": "no_point_rows_or_source_block_group",
                }
            )
            continue
        subset["lat_key"] = subset["latitude"].map(lambda v: repr(float(v)))
        subset["lon_key"] = subset["longitude"].map(lambda v: repr(float(v)))
        subset["point_key"] = subset["lat_key"] + "|" + subset["lon_key"]
        # Dedupe multi-victim/multi-charge rows by incident/report id before the
        # point-share computation: one incident at one point counts once, so a single
        # multi-charge event does not overstate concentration. Rows with no id are
        # treated as distinct events (kept).
        if "incident_id" in subset.columns:
            has_id = subset["incident_id"].notna()
            deduped_id = subset.loc[has_id].drop_duplicates(subset=["incident_id", "point_key"])
            no_id = subset.loc[~has_id]
            subset = pd.concat([deduped_id, no_id], ignore_index=True)
        point_counts = (
            subset.groupby(["point_key", "lat_key", "lon_key"], dropna=False)["weight"]
            .sum()
            .rename("point_count")
            .reset_index()
            .sort_values(["point_count", "point_key"], ascending=[False, True], kind="mergesort")
        )
        point_total = float(pd.to_numeric(subset["weight"], errors="coerce").fillna(0.0).sum())
        total = surface_total if surface_total > 0.0 else point_total
        top = point_counts.iloc[0]
        row = {
            "city_key": city_key,
            "city_name": city_name,
            "offense": offense,
            "located_incident_count": total,
            "max_point_count": float(top["point_count"]),
            "max_point_share": float(top["point_count"]) / total if total > 0.0 else 0.0,
            "max_point_lat": float(top["lat_key"]),
            "max_point_lon": float(top["lon_key"]),
            "point_key": str(top["point_key"]),
            "qa_status": "computed",
        }
        row["reviewed_exception"] = _is_reviewed_exception(pd.Series(row), exceptions)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["reviewed_exception"] = out["reviewed_exception"].astype(bool)
    return out.sort_values(["city_name", "offense"], kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build city-feed exact-point concentration QA artifact.")
    parser.add_argument(
        "--surface-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet",
    )
    parser.add_argument(
        "--exceptions",
        type=Path,
        default=REPO_ROOT / "configs" / "city_feed_exact_point_exceptions.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "state" / "qa" / "city_feed_exact_point_concentration_2024.csv",
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    qa = build_qa(surface_path=args.surface_path, exception_path=args.exceptions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    qa.to_csv(args.out, index=False)
    located = pd.to_numeric(qa["located_incident_count"], errors="coerce").fillna(0.0)
    max_point_count = pd.to_numeric(qa["max_point_count"], errors="coerce").fillna(0.0)
    violations = qa[
        qa["max_point_share"].ge(EXACT_POINT_SHARE_MAX)
        & located.ge(EXACT_POINT_MIN_LOCATED_COUNT)
        & max_point_count.ge(EXACT_POINT_MIN_POINT_COUNT)
        & ~qa["reviewed_exception"]
    ].copy()
    below_floor = qa[
        qa["max_point_share"].ge(EXACT_POINT_SHARE_MAX)
        & (
            located.lt(EXACT_POINT_MIN_LOCATED_COUNT)
            | max_point_count.lt(EXACT_POINT_MIN_POINT_COUNT)
        )
        & ~qa["reviewed_exception"]
    ].copy()
    summary = {
        "output": str(args.out),
        "rows": int(len(qa)),
        "computed_rows": int(qa["qa_status"].eq("computed").sum()),
        "reviewed_exception_rows": int(qa["reviewed_exception"].sum()),
        "threshold_max_point_share": EXACT_POINT_SHARE_MAX,
        "min_located_incident_count": EXACT_POINT_MIN_LOCATED_COUNT,
        "min_point_count": EXACT_POINT_MIN_POINT_COUNT,
        "unreviewed_violation_rows": int(len(violations)),
        "below_floor_suppressed_rows": int(len(below_floor)),
        "max_point_share": float(qa["max_point_share"].max()) if not qa.empty else None,
        "unreviewed_violation_sample": violations.sort_values("max_point_share", ascending=False, kind="mergesort").head(20).to_dict(orient="records"),
    }
    text = json.dumps(_json_safe(summary), indent=2, sort_keys=True)
    print(text)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
