from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crosswalk_shares import normalize_block_group_allocation_shares  # noqa: E402
from crimerisk.city_feed_quarantine import (  # noqa: E402
    denied_texture_offenses,
    filter_denied_texture_surface,
    flag_quarantined_coordinates,
    load_texture_policy,
)


# Gate-17 batch-A onboarded packet cities are part of the standard surface rebuild.
# Regression fix 2026-07-06: these cities were previously merged by a one-off script
# (integrate_onboarded_city_shares.py) whose output any later rebuild of this surface
# silently dropped, so the residual-training set shrank from 30 to 22 cities without
# notice. Appending the share-admitted packet surfaces here — before the texture
# policy is applied — keeps them in every rebuild and subjects them to the same
# located-evidence texture policy as feed cities.
GATE17_PACKETS_DIR = REPO_ROOT / "state" / "review" / "packets" / "city"
# Committed, supervisor-adjudicated admissions SSOT (batch A verbatim + batch B with
# adjudication overrides). Replaces the gitignored orchestration-tree batch-A CSV: a
# fresh clone silently dropped every packet city from the rebuild because the reader
# returned {} when the untracked file was absent.
GATE17_ADMISSION_CSV = REPO_ROOT / "configs" / "gate17_city_offense_admissions.csv"
GATE17_PACKET_CASE_TYPE = "municipal_validation_case"
GATE17_PACKET_CASE_NOTES = "Gate-17 onboarded packet; share-admitted offenses only"
GATE17_SURFACE_COLS = [
    "city_name",
    "jurisdiction_id",
    "state_fips",
    "year",
    "offense",
    "block_group_geoid",
    "incident_count",
    "share_within_city",
    "geocode_quality_tier",
    "validation_case_type",
    "validation_source_name",
    "validation_source_url",
    "validation_case_notes",
]


def _gate17_admitted_offenses(admission_csv: Path = GATE17_ADMISSION_CSV) -> dict[str, set[str]]:
    """Map city_key -> share-admitted offenses from the committed gate-17 admissions file."""
    if not admission_csv.exists():
        raise FileNotFoundError(
            f"gate-17 admissions file missing: {admission_csv} — the packet-city surface "
            "cannot be assembled without it (a silent empty admission set drops every "
            "packet city from the rebuild)"
        )
    out: dict[str, set[str]] = {}
    with admission_csv.open() as handle:
        for row in csv.DictReader(handle):
            share_ok = str(row.get("share_admitted_flag", "")).strip().lower() in {"true", "1", "yes"}
            admitted = str(row.get("admitted_flag", "")).strip().lower() in {"true", "1", "yes"}
            if share_ok and admitted:
                out.setdefault(str(row["city_key"]).strip(), set()).add(str(row["offense"]).strip())
    return out


def _packet_share_surface_from_normalized(city_key: str) -> tuple[pd.DataFrame, int]:
    """Re-derive a packet city's BG share surface from its normalized incidents.

    Mirrors the live-feed path exactly: located incident rows are dropped where they
    fall inside a quarantined coordinate radius (``flag_quarantined_coordinates``)
    before the ``groupby(year, offense, block_group_geoid)`` aggregation that defines
    the city share surface. Returns the aggregated surface together with the number of
    located incidents the coordinate quarantine removed (0 when no quarantine row
    fires). An empty frame with a non-zero drop count means every located row was
    quarantined; an empty frame with a zero drop count means there was nothing to
    aggregate.
    """
    normalized_path = GATE17_PACKETS_DIR / city_key / "normalized_incidents.parquet"
    if not normalized_path.exists():
        return pd.DataFrame(), 0
    frame = pd.read_parquet(normalized_path)
    if frame.empty:
        return pd.DataFrame(), 0
    frame = frame.copy()
    frame["latitude"] = pd.to_numeric(frame.get("latitude"), errors="coerce")
    frame["longitude"] = pd.to_numeric(frame.get("longitude"), errors="coerce")
    frame["block_group_geoid"] = frame["block_group_geoid"].astype("string")
    frame = frame[frame["block_group_geoid"].str.fullmatch(r"\d{12}").fillna(False)].copy()
    if frame.empty:
        return pd.DataFrame(), 0
    located_before = int(len(frame))
    frame = frame.loc[
        ~flag_quarantined_coordinates(frame, city_key=city_key, offense_col="offense")
    ].copy()
    quarantined_dropped = located_before - int(len(frame))
    if frame.empty:
        return pd.DataFrame(), quarantined_dropped
    meta_cols = [
        "jurisdiction_id",
        "state_fips",
        "geocode_quality_tier",
        "city_key",
        "city_name",
        "validation_case_type",
        "validation_source_name",
        "validation_source_url",
    ]
    meta = {col: (frame[col].iloc[0] if col in frame.columns and len(frame) else "") for col in meta_cols}
    agg = (
        frame.groupby(["year", "offense", "block_group_geoid"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    city_total = (
        frame.groupby(["year", "offense"], dropna=False).size().rename("city_total").reset_index()
    )
    agg = agg.merge(city_total, on=["year", "offense"], how="left")
    agg["share_within_city"] = agg["incident_count"] / agg["city_total"]
    for col, value in meta.items():
        agg[col] = value
    return agg, quarantined_dropped


# Set by _gate17_packet_surfaces so build_validation_shares can report each integrated
# packet city's re-derived incident total and how many incidents the coordinate
# quarantine dropped from it.
GATE17_PACKET_REDERIVE_REPORT: dict[str, dict[str, object]] = {}


def _gate17_packet_surfaces(existing: pd.DataFrame) -> list[pd.DataFrame]:
    """Share-surface frames for gate-17-admitted packet cities absent from ``existing``.

    Every admitted packet city is re-derived from its ``normalized_incidents.parquet``
    through ``flag_quarantined_coordinates`` — the identical incident-level path the
    live-feed cities use — so a coordinate quarantine always takes effect and no rebuild
    can ship a stale pre-aggregated ``city_share_surface.parquet``. There is a single
    ingestion path: the pre-aggregated packet surface is never consumed. Cities already
    present in the assembled surface are skipped (idempotent); only share-admitted
    offense rows are kept, normalized to the output surface schema.
    """
    GATE17_PACKET_REDERIVE_REPORT.clear()
    admit = _gate17_admitted_offenses()
    if not admit or existing.empty:
        return []
    present_names = set(existing["city_name"].astype(str).unique())
    present_ids = set(existing["jurisdiction_id"].astype(str).unique())
    frames: list[pd.DataFrame] = []
    for city_key, offenses in sorted(admit.items()):
        normalized_path = GATE17_PACKETS_DIR / city_key / "normalized_incidents.parquet"
        if not normalized_path.exists():
            continue
        surface, quarantined_dropped = _packet_share_surface_from_normalized(city_key)
        if surface.empty:
            raise RuntimeError(
                f"gate-17 packet {city_key!r} has normalized_incidents.parquet but re-derivation "
                "produced no located rows"
            )
        city_name = str(surface["city_name"].iloc[0]) if "city_name" in surface.columns else city_key
        jurisdiction_id = (
            str(surface["jurisdiction_id"].iloc[0]) if "jurisdiction_id" in surface.columns else ""
        )
        if city_name in present_names or jurisdiction_id in present_ids:
            continue
        rederived_incident_total = float(surface["incident_count"].sum())

        surface = surface[surface["offense"].astype(str).isin(offenses)].copy()
        if surface.empty:
            continue
        surface["validation_case_type"] = GATE17_PACKET_CASE_TYPE
        surface["validation_case_notes"] = GATE17_PACKET_CASE_NOTES
        for column in GATE17_SURFACE_COLS:
            if column not in surface.columns:
                surface[column] = ""
        frames.append(surface[GATE17_SURFACE_COLS])
        GATE17_PACKET_REDERIVE_REPORT[city_key] = {
            "rederived_incident_total": rederived_incident_total,
            "quarantined_dropped_incident_total": int(quarantined_dropped),
        }
    return frames


TUCSON_JURISDICTION_ID = "04:municipal:place:0477000"
TUCSON_CITY_NAME = "Tucson"
TUCSON_SOURCE_NAME = "Tucson Police Incidents - 2024 - Open Data"
TUCSON_SOURCE_URL = "https://data-cotgis.opendata.arcgis.com/items/734392a9ac114ea89e5e3b427e55efc5"
TUCSON_QUERY_URL = (
    "https://gis.tucsonaz.gov/arcgis/rest/services/PublicMaps/OpenData_PublicSafety/MapServer/80/query"
)
TUCSON_OUT_FIELDS = (
    "OBJECTID",
    "YEAR_REPT",
    "UCRsummary",
    "UCRSummaryDesc",
    "OFFENSE",
    "STATUTDESC",
    "Crime",
    "CrimeType",
    "CENSUSBLOCK",
    "cancelled",
)
TUCSON_OFFENSE_MAP = {
    "01": "murder",
    "02": "rape",
    "03": "robbery",
    "04": "aggravated_assault",
    "05": "burglary",
    "06": "larceny",
    "07": "motor_vehicle_theft",
}


def _json_safe(value: object) -> object:
    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if value is pd.NaT:
        return None
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

CHARLOTTE_JURISDICTION_ID = "37:municipal:place:3712000"
CHARLOTTE_CITY_NAME = "Charlotte"
CHARLOTTE_SOURCE_NAME = "CMPD Incidents"
CHARLOTTE_SOURCE_URL = "https://data.charlottenc.gov/datasets/charlotte::cmpd-incidents-1/about"
CHARLOTTE_QUERY_URL = "https://gis.charlottenc.gov/arcgis/rest/services/CMPD/CMPDIncidents/MapServer/0/query"
CHARLOTTE_OUT_FIELDS = (
    "OBJECTID",
    "YEAR",
    "INCIDENT_REPORT_ID",
    "CITY",
    "CLEARANCE_STATUS",
    "HIGHEST_NIBRS_CODE",
    "HIGHEST_NIBRS_DESCRIPTION",
    "LATITUDE_PUBLIC",
    "LONGITUDE_PUBLIC",
)
CHARLOTTE_OFFENSE_MAP = {
    "09A": "murder",
    "11A": "rape",
    "120": "robbery",
    "13A": "aggravated_assault",
    "220": "burglary",
    "23A": "larceny",
    "23B": "larceny",
    "23C": "larceny",
    "23D": "larceny",
    "23E": "larceny",
    "23F": "larceny",
    "23G": "larceny",
    "23H": "larceny",
    "240": "motor_vehicle_theft",
}

DALLAS_JURISDICTION_ID = "48:municipal:place:4819000"
DALLAS_CITY_NAME = "Dallas"
DALLAS_SOURCE_NAME = "Police Incidents - Dallas OpenData"
DALLAS_SOURCE_URL = "https://www.dallasopendata.com/Public-Safety/Police-Incidents/qv6i-rri7"
DALLAS_RESOURCE_URL = "https://www.dallasopendata.com/resource/qv6i-rri7.json"
DALLAS_OFFENSE_MAP = CHARLOTTE_OFFENSE_MAP

KANSAS_CITY_JURISDICTION_ID = "29:municipal:place:2938000"
KANSAS_CITY_NAME = "Kansas City"
KANSAS_CITY_SOURCE_NAME = "KCPD Crime Data 2024"
KANSAS_CITY_SOURCE_URL = "https://data.kcmo.org/Crime/KCPD-Crime-Data-2024/isbe-v4d8"
KANSAS_CITY_RESOURCE_URL = "https://data.kcmo.org/resource/isbe-v4d8.json"
KANSAS_CITY_IBRS_CODES = tuple(sorted(CHARLOTTE_OFFENSE_MAP))
MURDER_EVENT_ID_COLUMNS = (
    "report_no",
    "incident_report_id",
    "INCIDENT_REPORT_ID",
    "case_number",
    "IncidentNum",
    "incident_id",
    "crime_id",
    "offincident",
)

CINCINNATI_JURISDICTION_ID = "39:municipal:place:3915000"
CINCINNATI_CITY_NAME = "Cincinnati"
CINCINNATI_SOURCE_NAME = "PDI Police Data Initiative Crime Incidents"
CINCINNATI_SOURCE_URL = "https://data.cincinnati-oh.gov/Safety/PDI-Police-Data-Initiative-Crime-Incidents/k59e-2pvf"
CINCINNATI_RESOURCE_URL = "https://data.cincinnati-oh.gov/resource/k59e-2pvf.json"

SAN_DIEGO_JURISDICTION_ID = "06:municipal:place:0666000"
SAN_DIEGO_CITY_NAME = "San Diego"
SAN_DIEGO_SOURCE_NAME = "Police NIBRS Crime Offenses"
SAN_DIEGO_SOURCE_URL = "https://data.sandiego.gov/datasets/police-nibrs/"
SAN_DIEGO_CSV_URL = "https://seshat.datasd.org/police_nibrs/pd_nibrs_2024_datasd.csv"
SAN_DIEGO_OFFENSE_MAP = CHARLOTTE_OFFENSE_MAP
SAN_DIEGO_CSV_COLUMNS = (
    "objectid",
    "nibrs_uniq",
    "case_number",
    "occured_on",
    "approved_on",
    "year",
    "group_type",
    "ibr_category",
    "ibr_offense",
    "ibr_offense_description",
    "pd_offense_category",
    "city",
    "state",
    "geocode_status",
    "geocode_score",
    "latitude",
    "longitude",
)

ST_LOUIS_JURISDICTION_ID = "29:municipal:place:2965000"
ST_LOUIS_CITY_NAME = "St. Louis"
ST_LOUIS_SOURCE_NAME = "SLMPD NIBRS Crime Files"
ST_LOUIS_SOURCE_URL = "https://www.slmpd.org/stats/"
ST_LOUIS_CSV_URLS = (
    "https://slmpd.org/wp-content/uploads/2024/03/January2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/03/February2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/05/Downloadable-NIBRS-Crime-File-March-2024-CSV.csv",
    "https://slmpd.org/wp-content/uploads/2024/05/April2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/06/May2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/07/June2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/08/July2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/09/August2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/10/September2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/11/October2024.csv",
    "https://slmpd.org/wp-content/uploads/2024/12/November2024.csv",
    "https://slmpd.org/wp-content/uploads/2025/01/December2024.csv",
)
ST_LOUIS_OFFENSE_MAP = {
    **CHARLOTTE_OFFENSE_MAP,
    "11B": "rape",
    "11C": "rape",
}
# Monthly SLMPD files re-emit supplemented incidents with revised VictimNum rows; including VictimNum
# creates a multi-offense overcount against the annual control, so the stable public identity excludes it.
ST_LOUIS_VICTIM_COUNTED_OFFENSES: set[str] = set()
ST_LOUIS_CSV_COLUMNS = (
    "IncidentDate",
    "OccurredFromTime",
    "IncidentNum",
    "Offense",
    "NIBRS",
    "NIBRSCategory",
    "SRS_UCR",
    "CrimeAgainst",
    "IncidentTopSRS_UCR",
    "IncidentLocation",
    "IntersectionOtherLoc",
    "District",
    "Neighborhood",
    "NbhdNum",
    "Latitude",
    "Longitude",
    "VictimNum",
    "FirearmUsed",
    "IncidentNature",
)

MILWAUKEE_JURISDICTION_ID = "55:municipal:place:5553000"
MILWAUKEE_CITY_NAME = "Milwaukee"
MILWAUKEE_SOURCE_NAME = "NIBRS Crime Data (Historical)"
MILWAUKEE_SOURCE_URL = "https://data.milwaukee.gov/dataset/wibrarchive"
MILWAUKEE_RESOURCE_ID = "395db729-a30a-4e53-ab66-faeb5e1899c8"
MILWAUKEE_OFFENSE_MAP = CHARLOTTE_OFFENSE_MAP

MONTGOMERY_COUNTY_JURISDICTION_ID = "24:county:24031"
MONTGOMERY_COUNTY_NAME = "Montgomery County, MD"
MONTGOMERY_COUNTY_SOURCE_NAME = "Montgomery County Crime"
MONTGOMERY_COUNTY_SOURCE_URL = "https://data.montgomerycountymd.gov/Public-Safety/Crime/icn6-v9z3"
MONTGOMERY_COUNTY_RESOURCE_URL = "https://data.montgomerycountymd.gov/resource/icn6-v9z3.json"
MONTGOMERY_COUNTY_OFFENSE_MAP = CHARLOTTE_OFFENSE_MAP

DURHAM_JURISDICTION_ID = "37:municipal:place:3719000"
DURHAM_CITY_NAME = "Durham"
DURHAM_SOURCE_NAME = "DPD Incidents (UCR NIBRS Reporting)"
DURHAM_SOURCE_URL = "https://live-durhamnc.opendata.arcgis.com/documents/DurhamNC::dpd-incidents-ucr-nibrs-reporting/about"
DURHAM_EXCEL_URL = "https://www.arcgis.com/sharing/rest/content/items/a12d5115aac040a99ff89de1ba26eb07/data"
DURHAM_OFFENSE_MAP = CHARLOTTE_OFFENSE_MAP
DURHAM_NONCRIME_STATUSES = {"unfounded", "closed (non-criminal)"}


def _download_arcgis_query(
    *,
    base_url: str,
    params: dict[str, str],
    out_path: Path,
    page_size: int = 2000,
    return_geometry: bool = False,
    geometry_x_col: str = "geometry_x",
    geometry_y_col: str = "geometry_y",
    force_refresh: bool = False,
) -> pd.DataFrame:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        query_params = dict(params)
        query_params["resultOffset"] = str(int(offset))
        query_params["resultRecordCount"] = str(int(page_size))
        url = f"{base_url}?{urlencode(query_params)}"
        request = Request(url, headers={"User-Agent": "crimerisk-v2/next-phase-validation"})
        with urlopen(request, timeout=600) as response:
            payload = json.load(response)
        if "error" in payload:
            raise RuntimeError(f"ArcGIS query failed: {payload['error']}")
        features = payload.get("features", []) or []
        for feature in features:
            row = dict(feature.get("attributes", {}) or {})
            if return_geometry:
                geometry = feature.get("geometry", {}) or {}
                row[geometry_x_col] = geometry.get("x")
                row[geometry_y_col] = geometry.get("y")
            rows.append(row)
        if len(features) < int(page_size):
            break
        offset += int(page_size)

    frame = pd.DataFrame(rows)
    if return_geometry:
        frame[geometry_x_col] = pd.to_numeric(frame.get(geometry_x_col), errors="coerce")
        frame[geometry_y_col] = pd.to_numeric(frame.get(geometry_y_col), errors="coerce")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return frame


def _load_city_bg_geometry(*, bg_crosswalk_path: Path, state_bg_zip: Path, jurisdiction_id: str) -> gpd.GeoDataFrame:
    crosswalk = pd.read_parquet(bg_crosswalk_path)
    crosswalk = normalize_block_group_allocation_shares(crosswalk)
    crosswalk = crosswalk[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates().copy()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(jurisdiction_id)].copy()
    if crosswalk.empty:
        return gpd.GeoDataFrame(
            columns=["block_group_geoid", "jurisdiction_id", "state_fips", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    bg = gpd.read_file(state_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype("string").str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    return bg.to_crs("EPSG:4326") if not bg.empty and bg.crs != "EPSG:4326" else bg


def _load_county_bg_geometry(*, state_bg_zip: Path, county_fips: str) -> gpd.GeoDataFrame:
    bg = gpd.read_file(state_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype("string").str.zfill(12)
    bg = bg[bg["block_group_geoid"].str.startswith(str(county_fips))].copy()
    bg["jurisdiction_id"] = f"{str(county_fips)[:2]}:county:{county_fips}"
    bg["state_fips"] = str(county_fips)[:2]
    return bg.to_crs("EPSG:4326") if not bg.empty and bg.crs != "EPSG:4326" else bg


def _download_socrata_query(
    *,
    base_url: str,
    params: dict[str, str],
    out_path: Path,
    page_size: int = 50000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        query_params = dict(params)
        query_params["$limit"] = str(int(page_size))
        query_params["$offset"] = str(int(offset))
        request = Request(
            f"{base_url}?{urlencode(query_params)}",
            headers={"User-Agent": "crimerisk-v2/next-phase-validation"},
        )
        with urlopen(request, timeout=600) as response:
            page = json.load(response)
        if not isinstance(page, list):
            raise RuntimeError(f"Socrata query failed: {page}")
        rows.extend(page)
        if len(page) < int(page_size):
            break
        offset += int(page_size)

    frame = pd.DataFrame(rows)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return frame


def _download_csv(
    *,
    url: str,
    out_path: Path,
    usecols: tuple[str, ...] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 crimerisk-v2/next-phase-validation",
            "Referer": "https://www.slmpd.org/stats/",
        },
    )
    with urlopen(request, timeout=600) as response:
        payload = response.read()
    frame = pd.read_csv(BytesIO(payload), usecols=list(usecols) if usecols else None, dtype="string")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return frame


def _dedupe_murder_events_for_share(
    mapped: pd.DataFrame,
    *,
    fallback_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if mapped.empty or "offense" not in mapped.columns or "offense_row_id" not in mapped.columns:
        return mapped
    out = mapped.copy()
    murder_mask = out["offense"].astype("string").eq("murder")
    if not bool(murder_mask.any()):
        return out

    event_id = pd.Series(pd.NA, index=out.index, dtype="string")
    for col in MURDER_EVENT_ID_COLUMNS:
        if col not in out.columns:
            continue
        values = out[col].astype("string").str.strip()
        values = values.mask(values.isna() | values.eq("") | values.eq("<NA>"))
        event_id = event_id.fillna(values)
    if fallback_columns:
        fallback_cols = [col for col in fallback_columns if col in out.columns]
        if fallback_cols:
            fallback = (
                out[fallback_cols]
                .fillna("")
                .astype("string")
                .agg("|".join, axis=1)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            fallback = fallback.mask(fallback.eq("") | fallback.eq("<NA>"))
            event_id = event_id.fillna(fallback)
    event_id = event_id.fillna(out["offense_row_id"].astype("string").str.strip())
    out.loc[murder_mask, "offense_row_id"] = "murder_event::" + event_id.loc[murder_mask].astype("string")
    return out


def _download_csvs(
    *,
    urls: tuple[str, ...],
    out_path: Path,
    usecols: tuple[str, ...] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(out_path)

    frames = []
    for url in urls:
        frames.append(
            _download_csv(
                url=url,
                out_path=out_path.parent / f"{out_path.stem}_{len(frames) + 1:02d}{out_path.suffix}",
                usecols=usecols,
                force_refresh=force_refresh,
            ).assign(source_file_url=url)
        )
    frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return frame


def _download_excel(
    *,
    url: str,
    out_path: Path,
    sheet_name: str | int = 0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 crimerisk-v2/next-phase-validation"},
    )
    with urlopen(request, timeout=600) as response:
        payload = response.read()
    frame = pd.read_excel(BytesIO(payload), sheet_name=sheet_name, dtype="string")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return frame


def _download_ckan_datastore_sql(
    *,
    resource_id: str,
    sql_template: str,
    out_path: Path,
    page_size: int = 30000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        sql = sql_template.format(resource_id=resource_id, limit=int(page_size), offset=int(offset))
        url = "https://data.milwaukee.gov/api/3/action/datastore_search_sql?" + urlencode({"sql": sql})
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 crimerisk-v2/next-phase-validation"},
        )
        with urlopen(request, timeout=600) as response:
            payload = json.load(response)
        if not payload.get("success"):
            raise RuntimeError(f"CKAN datastore SQL failed: {payload}")
        page = payload.get("result", {}).get("records", []) or []
        rows.extend(page)
        if len(page) < int(page_size):
            break
        offset += int(page_size)

    frame = pd.DataFrame(rows)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return frame


def _case_metadata_for_existing(city_shares: pd.DataFrame) -> pd.DataFrame:
    meta = city_shares[["city_name", "jurisdiction_id", "state_fips"]].drop_duplicates().copy()
    meta["validation_case_type"] = "promoted_city_control"
    meta["validation_source_name"] = "existing_city_incident_share_surface"
    meta["validation_source_url"] = ""
    meta["validation_case_notes"] = "Existing promoted city incident share surface row."
    return meta


def _build_tucson_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = raw_dir / f"tucson_police_incidents_{int(year)}.parquet"
    refresh_raw = bool(force_refresh)
    if raw_path.exists() and raw_path.stat().st_size > 0 and not refresh_raw:
        cached_cols = set(pd.read_parquet(raw_path).columns)
        if "geometry_x" not in cached_cols or "geometry_y" not in cached_cols:
            refresh_raw = True
    raw = _download_arcgis_query(
        base_url=TUCSON_QUERY_URL,
        params={
            "where": (
                f"YEAR_REPT={int(year)} AND cancelled=0 "
                "AND UCRsummary IN ('01','02','03','04','05','06','07')"
            ),
            "outFields": ",".join(TUCSON_OUT_FIELDS),
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "json",
        },
        out_path=raw_path,
        return_geometry=True,
        force_refresh=refresh_raw,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["year"] = pd.to_numeric(mapped["YEAR_REPT"], errors="coerce").astype("Int64")
    mapped = mapped[mapped["year"].eq(int(year))].copy()
    mapped["ucr_summary"] = mapped["UCRsummary"].astype("string").str.strip().str.zfill(2)
    mapped["offense"] = mapped["ucr_summary"].map(TUCSON_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()

    # Tucson code 0704 is a recovery-for-other-jurisdiction record, not an own-jurisdiction theft.
    recovery_mvt = mapped["ucr_summary"].eq("07") & (
        mapped["OFFENSE"].astype("string").str.strip().eq("0704")
        | mapped["CrimeType"].astype("string").str.strip().str.lower().eq("other")
    )
    mapped = mapped[~recovery_mvt].copy()
    mapped["offense_row_id"] = mapped["OBJECTID"].astype("string").str.strip()
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")

    crosswalk = pd.read_parquet(bg_crosswalk_path)
    crosswalk = normalize_block_group_allocation_shares(crosswalk)
    crosswalk = crosswalk[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates().copy()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(TUCSON_JURISDICTION_ID)].copy()

    mapped["geometry_x"] = pd.to_numeric(mapped.get("geometry_x"), errors="coerce")
    mapped["geometry_y"] = pd.to_numeric(mapped.get("geometry_y"), errors="coerce")
    mapped = mapped.loc[
        ~flag_quarantined_coordinates(
            mapped,
            city_key="tucson",
            lat_col="geometry_y",
            lon_col="geometry_x",
            offense_col="offense",
        )
    ].copy()

    mapped["census_block_geoid"] = (
        mapped["CENSUSBLOCK"].astype("string").str.extract(r"(\d{15})", expand=False)
    )
    mapped["block_group_geoid"] = mapped["census_block_geoid"].str.slice(0, 12)
    source_block = mapped[mapped["block_group_geoid"].astype("string").str.fullmatch(r"\d{12}").fillna(False)].copy()
    source_block_matched = source_block.merge(crosswalk, on="block_group_geoid", how="inner")
    source_block_matched["geocode_quality_tier"] = "source_census_block"

    fallback = mapped[~mapped["offense_row_id"].isin(source_block_matched["offense_row_id"].astype(str))].copy()
    fallback["geometry_x"] = pd.to_numeric(fallback.get("geometry_x"), errors="coerce")
    fallback["geometry_y"] = pd.to_numeric(fallback.get("geometry_y"), errors="coerce")
    fallback = fallback[
        fallback["geometry_y"].between(31.9, 32.6, inclusive="both")
        & fallback["geometry_x"].between(-111.4, -110.5, inclusive="both")
    ].copy()
    point_matched = pd.DataFrame(columns=source_block_matched.columns)
    if not fallback.empty:
        bg = _load_city_bg_geometry(
            bg_crosswalk_path=bg_crosswalk_path,
            state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_04_bg.zip",
            jurisdiction_id=TUCSON_JURISDICTION_ID,
        )
        if not bg.empty:
            points = gpd.GeoDataFrame(
                fallback.drop(columns=["block_group_geoid"], errors="ignore").copy(),
                geometry=gpd.points_from_xy(fallback["geometry_x"], fallback["geometry_y"]),
                crs="EPSG:4326",
            )
            point_matched = gpd.sjoin(points, bg, how="inner", predicate="within")
            point_matched = pd.DataFrame(point_matched.drop(columns=["geometry", "index_right"], errors="ignore"))
            point_matched = point_matched.drop_duplicates("offense_row_id", keep="first")
            point_matched["geocode_quality_tier"] = "reported_point_fallback"

    matched = pd.concat([source_block_matched, point_matched], ignore_index=True, sort=False)
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = TUCSON_CITY_NAME
    counts["validation_case_type"] = "municipal_validation_case"
    counts["validation_source_name"] = TUCSON_SOURCE_NAME
    counts["validation_source_url"] = TUCSON_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official Tucson public incident feed; validation-only spatial shares. "
        "Annual public-row totals are not treated as production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    source_block_counts = (
        source_block.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("source_block_rows")
        .reset_index()
    )
    point_candidate_counts = (
        fallback.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_fallback_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = (
        mapped_counts.merge(source_block_counts, on=["year", "offense"], how="left")
        .merge(point_candidate_counts, on=["year", "offense"], how="left")
        .merge(matched_counts, on=["year", "offense"], how="left")
    )
    recon["city_name"] = TUCSON_CITY_NAME
    recon["jurisdiction_id"] = TUCSON_JURISDICTION_ID
    recon["validation_case_type"] = "municipal_validation_case"
    recon["validation_source_name"] = TUCSON_SOURCE_NAME
    recon["validation_source_url"] = TUCSON_SOURCE_URL
    for col in ["source_block_rows", "point_fallback_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _build_charlotte_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = ",".join(f"'{code}'" for code in sorted(CHARLOTTE_OFFENSE_MAP))
    raw = _download_arcgis_query(
        base_url=CHARLOTTE_QUERY_URL,
        params={
            "where": (
                f"YEAR='{int(year)}' AND CITY='CHARLOTTE' "
                "AND CLEARANCE_STATUS <> 'Unfounded' "
                f"AND HIGHEST_NIBRS_CODE IN ({codes})"
            ),
            "outFields": ",".join(CHARLOTTE_OUT_FIELDS),
            "returnGeometry": "false",
            "f": "json",
        },
        out_path=raw_dir / f"charlotte_cmpd_incidents_{int(year)}.parquet",
        force_refresh=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["year"] = pd.to_numeric(mapped["YEAR"], errors="coerce").astype("Int64")
    mapped = mapped[mapped["year"].eq(int(year))].copy()
    mapped["nibrs_code"] = mapped["HIGHEST_NIBRS_CODE"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped["nibrs_code"].map(CHARLOTTE_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["offense_row_id"] = mapped["INCIDENT_REPORT_ID"].astype("string").str.strip()
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    mapped["latitude"] = pd.to_numeric(mapped["LATITUDE_PUBLIC"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["LONGITUDE_PUBLIC"], errors="coerce")
    point_candidates = mapped[
        mapped["latitude"].between(34.9, 35.5, inclusive="both")
        & mapped["longitude"].between(-81.1, -80.5, inclusive="both")
    ].copy()

    bg = _load_city_bg_geometry(
        bg_crosswalk_path=bg_crosswalk_path,
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_37_bg.zip",
        jurisdiction_id=CHARLOTTE_JURISDICTION_ID,
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["longitude"], point_candidates["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_latlon"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = CHARLOTTE_CITY_NAME
    counts["validation_case_type"] = "municipal_validation_case"
    counts["validation_source_name"] = CHARLOTTE_SOURCE_NAME
    counts["validation_source_url"] = CHARLOTTE_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official CMPD incident feed; validation-only spatial shares from highest NIBRS code. "
        "Public-row totals are not treated as production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(point_counts, on=["year", "offense"], how="left").merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = CHARLOTTE_CITY_NAME
    recon["jurisdiction_id"] = CHARLOTTE_JURISDICTION_ID
    recon["validation_case_type"] = "municipal_validation_case"
    recon["validation_source_name"] = CHARLOTTE_SOURCE_NAME
    recon["validation_source_url"] = CHARLOTTE_SOURCE_URL
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _build_montgomery_county_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del bg_crosswalk_path
    codes = ",".join(f"'{code}'" for code in sorted(MONTGOMERY_COUNTY_OFFENSE_MAP))
    raw = _download_socrata_query(
        base_url=MONTGOMERY_COUNTY_RESOURCE_URL,
        params={
            "$select": ",".join(
                [
                    "incident_id",
                    "date",
                    "nibrs_code",
                    "crimename2",
                    "agency",
                    "latitude",
                    "longitude",
                ]
            ),
            "$where": (
                f"date between '{int(year)}-01-01T00:00:00' and '{int(year)}-12-31T23:59:59' "
                "AND agency='MCPD' "
                f"AND nibrs_code in({codes})"
            ),
        },
        out_path=raw_dir / f"montgomery_county_md_mcpd_crime_{int(year)}.parquet",
        force_refresh=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["year"] = int(year)
    mapped["nibrs_code"] = mapped["nibrs_code"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped["nibrs_code"].map(MONTGOMERY_COUNTY_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["offense_row_id"] = mapped["incident_id"].astype("string").str.strip()
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    point_candidates = mapped[
        mapped["latitude"].between(38.8, 39.4, inclusive="both")
        & mapped["longitude"].between(-77.6, -76.8, inclusive="both")
    ].copy()
    bg = _load_county_bg_geometry(
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_24_bg.zip",
        county_fips="24031",
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["longitude"], point_candidates["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_latlon"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = MONTGOMERY_COUNTY_NAME
    counts["validation_case_type"] = "suburban_county_validation_case"
    counts["validation_source_name"] = MONTGOMERY_COUNTY_SOURCE_NAME
    counts["validation_source_url"] = MONTGOMERY_COUNTY_SOURCE_URL
    counts["validation_case_notes"] = (
        "Montgomery County Police Department rows from the county open crime feed; "
        "allocation-only suburban county validation case, excluded from jurisdiction-total dominance."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(point_counts, on=["year", "offense"], how="left").merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = MONTGOMERY_COUNTY_NAME
    recon["jurisdiction_id"] = MONTGOMERY_COUNTY_JURISDICTION_ID
    recon["validation_case_type"] = "suburban_county_validation_case"
    recon["validation_source_name"] = MONTGOMERY_COUNTY_SOURCE_NAME
    recon["validation_source_url"] = MONTGOMERY_COUNTY_SOURCE_URL
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _build_dallas_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = ",".join(f"'{code}'" for code in sorted(DALLAS_OFFENSE_MAP))
    raw = _download_socrata_query(
        base_url=DALLAS_RESOURCE_URL,
        params={
            "$select": ",".join(
                [
                    ":id",
                    "incidentnum",
                    "year1",
                    "date1",
                    "nibrs_code",
                    "nibrs_crime",
                    "offincident",
                    "status",
                    "ucr_disp",
                    "city",
                    "geocoded_column.latitude as latitude",
                    "geocoded_column.longitude as longitude",
                ]
            ),
            "$where": (
                f"year1={int(year)} "
                "AND city='DALLAS' "
                f"AND nibrs_code in({codes})"
            ),
        },
        out_path=raw_dir / f"dallas_police_incidents_{int(year)}.parquet",
        force_refresh=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["year"] = pd.to_numeric(mapped["year1"], errors="coerce").astype("Int64")
    mapped = mapped[mapped["year"].eq(int(year))].copy()
    mapped["nibrs_code"] = mapped["nibrs_code"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped["nibrs_code"].map(DALLAS_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["offense_row_id"] = mapped[":id"].astype("string").str.strip()
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    point_candidates = mapped[
        mapped["latitude"].between(32.5, 33.1, inclusive="both")
        & mapped["longitude"].between(-97.1, -96.4, inclusive="both")
    ].copy()

    bg = _load_city_bg_geometry(
        bg_crosswalk_path=bg_crosswalk_path,
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_48_bg.zip",
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["longitude"], point_candidates["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_latlon"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = DALLAS_CITY_NAME
    counts["validation_case_type"] = "municipal_validation_case"
    counts["validation_source_name"] = DALLAS_SOURCE_NAME
    counts["validation_source_url"] = DALLAS_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official Dallas Police incident feed; validation-only spatial shares from NIBRS offense rows. "
        "Public-row totals are not treated as production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(point_counts, on=["year", "offense"], how="left").merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = DALLAS_CITY_NAME
    recon["jurisdiction_id"] = DALLAS_JURISDICTION_ID
    recon["validation_case_type"] = "municipal_validation_case"
    recon["validation_source_name"] = DALLAS_SOURCE_NAME
    recon["validation_source_url"] = DALLAS_SOURCE_URL
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _kansas_city_offense_from_row(row: pd.Series) -> str | None:
    code = str(row.get("ibrs", "")).strip().upper()
    label = str(row.get("offense", "")).strip().upper()
    if not code or "UNFOUNDED" in label:
        return None
    if code == "09A":
        return "murder"
    if code == "11A":
        return "rape"
    if code == "120":
        return "robbery" if "ROBBERY" in label else None
    if code == "13A":
        if "ASSAULT" not in label:
            return None
        if "NON-AGGRAVATED" in label:
            return None
        if "AGGRAVATED" in label or "DEADLY" in label:
            return "aggravated_assault"
        return None
    if code == "220":
        return "burglary" if "BURGLARY" in label else None
    if code in {"23A", "23B", "23C", "23D", "23E", "23F", "23G", "23H"}:
        if any(token in label for token in ["STEALING", "THEFT", "SHOPLIFT", "PICKPOCKET", "PURSE SNATCH"]):
            if "STOLEN AUTO" in label:
                return None
            return "larceny"
        return None
    if code == "240":
        if any(token in label for token in ["RECOVERED", "ATTEMPT TO LOCATE", "PROPERTY DAMAGE"]):
            return None
        if "STOLEN AUTO" in label or "TAMPERING" in label:
            return "motor_vehicle_theft"
        return None
    return None


def _build_kansas_city_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = ",".join(f"'{code}'" for code in KANSAS_CITY_IBRS_CODES)
    params = {
        "$select": ",".join(
            [
                ":id",
                "report_no",
                "reported_date",
                "from_date",
                "offense",
                "ibrs",
                "city",
                "location.latitude as latitude",
                "location.longitude as longitude",
            ]
        ),
        "$where": (
            f"from_date between '{int(year)}-01-01T00:00:00' and '{int(year)}-12-31T23:59:59' "
            "AND city='KANSAS CITY' "
            f"AND ibrs in({codes})"
        ),
    }
    raw_path = raw_dir / f"kansas_city_mo_kcpd_crime_{int(year)}.parquet"
    raw = _download_socrata_query(
        base_url=KANSAS_CITY_RESOURCE_URL,
        params=params,
        out_path=raw_dir / f"kansas_city_mo_kcpd_crime_{int(year)}.parquet",
        force_refresh=force_refresh,
    )
    if "report_no" not in raw.columns:
        raw = _download_socrata_query(
            base_url=KANSAS_CITY_RESOURCE_URL,
            params=params,
            out_path=raw_path,
            force_refresh=True,
        )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["year"] = int(year)
    mapped["ibrs"] = mapped["ibrs"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped.apply(_kansas_city_offense_from_row, axis=1)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["offense_row_id"] = mapped[":id"].astype("string").str.strip()
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped_before_share_dedupe = mapped.copy()
    mapped = _dedupe_murder_events_for_share(
        mapped,
        fallback_columns=("reported_date", "from_date", "latitude", "longitude"),
    )
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    point_candidates = mapped[
        mapped["latitude"].between(38.75, 39.45, inclusive="both")
        & mapped["longitude"].between(-95.0, -94.25, inclusive="both")
    ].copy()

    bg = _load_city_bg_geometry(
        bg_crosswalk_path=bg_crosswalk_path,
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_29_bg.zip",
        jurisdiction_id=KANSAS_CITY_JURISDICTION_ID,
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["longitude"], point_candidates["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_latlon"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = KANSAS_CITY_NAME
    counts["validation_case_type"] = "municipal_validation_case"
    counts["validation_source_name"] = KANSAS_CITY_SOURCE_NAME
    counts["validation_source_url"] = KANSAS_CITY_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official KCPD 2024 crime feed; validation-only spatial shares from conservative IBRS/offense-label mapping. "
        "Public-row totals are not treated as production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped_before_share_dedupe.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    share_event_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("share_event_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(share_event_counts, on=["year", "offense"], how="left").merge(
        point_counts,
        on=["year", "offense"],
        how="left",
    ).merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = KANSAS_CITY_NAME
    recon["jurisdiction_id"] = KANSAS_CITY_JURISDICTION_ID
    recon["validation_case_type"] = "municipal_validation_case"
    recon["validation_source_name"] = KANSAS_CITY_SOURCE_NAME
    recon["validation_source_url"] = KANSAS_CITY_SOURCE_URL
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["share_event_rows"] = pd.to_numeric(recon["share_event_rows"], errors="coerce").fillna(0.0)
    recon["share_deduped_rows"] = (
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0) - recon["share_event_rows"]
    ).clip(lower=0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _cincinnati_offense_from_row(row: pd.Series) -> str | None:
    group = str(row.get("ucr_group", "")).strip().upper()
    offense = str(row.get("offense_label", "")).strip().upper()
    if offense == "MURDER":
        return "murder"
    if group == "RAPE" and offense == "RAPE":
        return "rape"
    if group == "ROBBERY" and "ROBBERY" in offense:
        return "robbery"
    if group == "AGGRAVATED ASSAULTS" and offense in {"AGGRAVATED ASSAULT", "FELONIOUS ASSAULT"}:
        return "aggravated_assault"
    if group == "BURGLARY/BREAKING ENTERING" and any(
        token in offense for token in ["BURGLARY", "BREAKING AND ENTERING"]
    ):
        return "burglary"
    if group == "THEFT":
        return "larceny"
    if group == "UNAUTHORIZED USE" and offense == "UNAUTHORIZED USE OF MOTOR VEHICLE":
        return "motor_vehicle_theft"
    return None


def _build_cincinnati_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _download_socrata_query(
        base_url=CINCINNATI_RESOURCE_URL,
        params={
            "$select": ",".join(
                [
                    "instanceid",
                    "date_reported",
                    "date_from",
                    "ucr",
                    "ucr_group",
                    "offense",
                    "longitude_x",
                    "latitude_x",
                    "community_council_neighborhood",
                ]
            ),
            "$where": (
                f"date_reported between '{int(year)}-01-01T00:00:00' and '{int(year)}-12-31T23:59:59' "
                "AND (ucr_group in('HOMICIDE','RAPE','ROBBERY','AGGRAVATED ASSAULTS',"
                "'BURGLARY/BREAKING ENTERING','THEFT','UNAUTHORIZED USE'))"
            ),
        },
        out_path=raw_dir / f"cincinnati_pdi_crime_incidents_{int(year)}.parquet",
        force_refresh=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["date_reported_parsed"] = pd.to_datetime(mapped["date_reported"], errors="coerce")
    mapped["year"] = mapped["date_reported_parsed"].dt.year.astype("Int64")
    mapped = mapped[mapped["year"].eq(int(year))].copy()
    mapped["ucr_group"] = mapped["ucr_group"].astype("string").str.strip()
    mapped["offense_label"] = mapped["offense"].astype("string").str.strip()
    mapped["offense"] = mapped.apply(_cincinnati_offense_from_row, axis=1)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["offense_row_id"] = mapped["instanceid"].astype("string").str.strip()
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    mapped["latitude"] = pd.to_numeric(mapped["latitude_x"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude_x"], errors="coerce")
    point_candidates = mapped[
        mapped["latitude"].between(38.9, 39.4, inclusive="both")
        & mapped["longitude"].between(-84.8, -84.2, inclusive="both")
    ].copy()

    bg = _load_city_bg_geometry(
        bg_crosswalk_path=bg_crosswalk_path,
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_39_bg.zip",
        jurisdiction_id=CINCINNATI_JURISDICTION_ID,
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["longitude"], point_candidates["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_latlon_partial_year"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = CINCINNATI_CITY_NAME
    counts["validation_case_type"] = "partial_year_municipal_validation_case"
    counts["validation_source_name"] = CINCINNATI_SOURCE_NAME
    counts["validation_source_url"] = CINCINNATI_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official Cincinnati PDI incident feed; validation-only spatial shares from available 2024 rows "
        "through the feed's current 2024 maximum date. Motor-vehicle theft uses the source's narrower "
        "UNAUTHORIZED USE OF MOTOR VEHICLE category. Public-row totals are not treated as production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(point_counts, on=["year", "offense"], how="left").merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = CINCINNATI_CITY_NAME
    recon["jurisdiction_id"] = CINCINNATI_JURISDICTION_ID
    recon["validation_case_type"] = "partial_year_municipal_validation_case"
    recon["validation_source_name"] = CINCINNATI_SOURCE_NAME
    recon["validation_source_url"] = CINCINNATI_SOURCE_URL
    recon["source_min_date_reported"] = mapped["date_reported_parsed"].min()
    recon["source_max_date_reported"] = mapped["date_reported_parsed"].max()
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _build_san_diego_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _download_csv(
        url=SAN_DIEGO_CSV_URL,
        out_path=raw_dir / f"san_diego_police_nibrs_offenses_{int(year)}.parquet",
        usecols=SAN_DIEGO_CSV_COLUMNS,
        force_refresh=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["year"] = pd.to_numeric(mapped["year"], errors="coerce").astype("Int64")
    mapped = mapped[mapped["year"].eq(int(year))].copy()
    mapped["nibrs_code"] = mapped["ibr_offense"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped["nibrs_code"].map(SAN_DIEGO_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["offense_row_id"] = mapped["nibrs_uniq"].astype("string").str.strip()
    missing_row_id = mapped["offense_row_id"].isna() | mapped["offense_row_id"].isin(["", "<NA>"])
    mapped.loc[missing_row_id, "offense_row_id"] = mapped.loc[missing_row_id, "objectid"].astype("string").str.strip()
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    mapped["occurred_on_parsed"] = pd.to_datetime(mapped["occured_on"], errors="coerce")
    point_candidates = mapped[
        mapped["latitude"].between(32.5, 33.2, inclusive="both")
        & mapped["longitude"].between(-117.4, -116.7, inclusive="both")
    ].copy()
    point_candidates = point_candidates.loc[
        ~flag_quarantined_coordinates(point_candidates, city_key="san_diego", offense_col="offense")
    ].copy()

    bg = _load_city_bg_geometry(
        bg_crosswalk_path=bg_crosswalk_path,
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_06_bg.zip",
        jurisdiction_id=SAN_DIEGO_JURISDICTION_ID,
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["longitude"], point_candidates["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_latlon"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = SAN_DIEGO_CITY_NAME
    counts["validation_case_type"] = "municipal_validation_case"
    counts["validation_source_name"] = SAN_DIEGO_SOURCE_NAME
    counts["validation_source_url"] = SAN_DIEGO_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official San Diego Police NIBRS offense CSV; validation-only spatial shares from public "
        "geocoded offense rows. Public-row totals are not treated as production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(point_counts, on=["year", "offense"], how="left").merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = SAN_DIEGO_CITY_NAME
    recon["jurisdiction_id"] = SAN_DIEGO_JURISDICTION_ID
    recon["validation_case_type"] = "municipal_validation_case"
    recon["validation_source_name"] = SAN_DIEGO_SOURCE_NAME
    recon["validation_source_url"] = SAN_DIEGO_SOURCE_URL
    recon["source_min_date_reported"] = mapped["occurred_on_parsed"].min()
    recon["source_max_date_reported"] = mapped["occurred_on_parsed"].max()
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _build_st_louis_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _download_csvs(
        urls=ST_LOUIS_CSV_URLS,
        out_path=raw_dir / f"st_louis_slmpd_nibrs_crime_files_{int(year)}.parquet",
        usecols=ST_LOUIS_CSV_COLUMNS,
        force_refresh=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["incident_date_parsed"] = pd.to_datetime(mapped["IncidentDate"], errors="coerce", format="mixed")
    mapped["year"] = mapped["incident_date_parsed"].dt.year.astype("Int64")
    mapped = mapped[mapped["year"].eq(int(year))].copy()
    mapped["nibrs_code"] = mapped["NIBRS"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped["nibrs_code"].map(ST_LOUIS_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["latitude"] = pd.to_numeric(mapped["Latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["Longitude"], errors="coerce")
    base_row_id_cols = [
        "IncidentNum",
        "NIBRS",
        "Offense",
        "IncidentLocation",
        "IntersectionOtherLoc",
        "Latitude",
        "Longitude",
    ]
    victim_row_id_cols = [*base_row_id_cols, "VictimNum"]
    base_row_id = (
        mapped[base_row_id_cols]
        .fillna("")
        .astype("string")
        .agg("|".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    victim_row_id = (
        mapped[victim_row_id_cols]
        .fillna("")
        .astype("string")
        .agg("|".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    mapped["offense_row_id"] = base_row_id
    victim_mask = mapped["offense"].isin(ST_LOUIS_VICTIM_COUNTED_OFFENSES)
    mapped.loc[victim_mask, "offense_row_id"] = victim_row_id.loc[victim_mask]
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    point_candidates = mapped[
        mapped["latitude"].between(38.45, 38.85, inclusive="both")
        & mapped["longitude"].between(-90.35, -90.10, inclusive="both")
    ].copy()

    bg = _load_city_bg_geometry(
        bg_crosswalk_path=bg_crosswalk_path,
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_29_bg.zip",
        jurisdiction_id=ST_LOUIS_JURISDICTION_ID,
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["longitude"], point_candidates["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_latlon"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = ST_LOUIS_CITY_NAME
    counts["validation_case_type"] = "municipal_validation_case"
    counts["validation_source_name"] = ST_LOUIS_SOURCE_NAME
    counts["validation_source_url"] = ST_LOUIS_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official SLMPD monthly 2024 NIBRS crime files; validation-only spatial shares from public "
        "geocoded offense rows filtered to 2024 incident dates. Public-row totals are not treated as "
        "production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(point_counts, on=["year", "offense"], how="left").merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = ST_LOUIS_CITY_NAME
    recon["jurisdiction_id"] = ST_LOUIS_JURISDICTION_ID
    recon["validation_case_type"] = "municipal_validation_case"
    recon["validation_source_name"] = ST_LOUIS_SOURCE_NAME
    recon["validation_source_url"] = ST_LOUIS_SOURCE_URL
    recon["source_min_date_reported"] = mapped["incident_date_parsed"].min()
    recon["source_max_date_reported"] = mapped["incident_date_parsed"].max()
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _build_milwaukee_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _download_ckan_datastore_sql(
        resource_id=MILWAUKEE_RESOURCE_ID,
        sql_template=(
            'SELECT "_id", "Case_Number", "Incident_Number", "Incident_Date", '
            '"Incident_Last_Edited", "Offense_Last_Edited", "Police_District", '
            '"Offense_All", "Location_All", "Address_Longitude", "Address_Latitude", "Weapon_Used_All" '
            'FROM "{resource_id}" '
            f"WHERE \"Incident_Date\" >= '{int(year)}-01-01' "
            f"AND \"Incident_Date\" < '{int(year) + 1}-01-01' "
            'ORDER BY "_id" LIMIT {limit} OFFSET {offset}'
        ),
        out_path=raw_dir / f"milwaukee_wibr_historical_{int(year)}.parquet",
        force_refresh=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["incident_date_parsed"] = pd.to_datetime(mapped["Incident_Date"], errors="coerce")
    mapped["year"] = mapped["incident_date_parsed"].dt.year.astype("Int64")
    mapped = mapped[mapped["year"].eq(int(year))].copy()
    mapped["offense_code_list"] = (
        mapped["Offense_All"]
        .astype("string")
        .fillna("")
        .str.replace(",", ";", regex=False)
        .str.split(";")
    )
    mapped = mapped.explode("offense_code_list", ignore_index=True)
    mapped["nibrs_code"] = mapped["offense_code_list"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped["nibrs_code"].map(MILWAUKEE_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["offense_sequence"] = mapped.groupby("_id", dropna=False).cumcount() + 1
    mapped["offense_row_id"] = (
        mapped["_id"].astype("string").str.strip()
        + "|"
        + mapped["nibrs_code"].astype("string").str.strip()
        + "|"
        + mapped["offense_sequence"].astype("string").str.strip()
    )
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    mapped["latitude"] = pd.to_numeric(mapped["Address_Latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["Address_Longitude"], errors="coerce")
    point_candidates = mapped[
        mapped["latitude"].between(42.85, 43.22, inclusive="both")
        & mapped["longitude"].between(-88.10, -87.80, inclusive="both")
    ].copy()

    bg = _load_city_bg_geometry(
        bg_crosswalk_path=bg_crosswalk_path,
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_55_bg.zip",
        jurisdiction_id=MILWAUKEE_JURISDICTION_ID,
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["longitude"], point_candidates["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_latlon"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = MILWAUKEE_CITY_NAME
    counts["validation_case_type"] = "municipal_validation_case"
    counts["validation_source_name"] = MILWAUKEE_SOURCE_NAME
    counts["validation_source_url"] = MILWAUKEE_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official Milwaukee historical NIBRS crime datastore; validation-only spatial shares from "
        "public geocoded rows with semicolon-delimited NIBRS offenses exploded to offense rows. "
        "Public-row totals are not treated as production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(point_counts, on=["year", "offense"], how="left").merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = MILWAUKEE_CITY_NAME
    recon["jurisdiction_id"] = MILWAUKEE_JURISDICTION_ID
    recon["validation_case_type"] = "municipal_validation_case"
    recon["validation_source_name"] = MILWAUKEE_SOURCE_NAME
    recon["validation_source_url"] = MILWAUKEE_SOURCE_URL
    recon["source_min_date_reported"] = mapped["incident_date_parsed"].min()
    recon["source_max_date_reported"] = mapped["incident_date_parsed"].max()
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def _build_durham_surface(
    *,
    year: int,
    raw_dir: Path,
    bg_crosswalk_path: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _download_excel(
        url=DURHAM_EXCEL_URL,
        sheet_name="dpd incidents (ucr nibrs)",
        out_path=raw_dir / "durham_dpd_incidents_ucr_nibrs_reporting.parquet",
        force_refresh=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    mapped = raw.copy()
    mapped["report_date"] = pd.to_datetime(mapped["Report Date"], errors="coerce")
    mapped["year"] = mapped["report_date"].dt.year.astype("Int64")
    mapped = mapped[mapped["year"].eq(int(year))].copy()
    mapped["nibrs_code"] = mapped["UCR Code"].astype("string").str.strip().str.upper()
    mapped["offense"] = mapped["nibrs_code"].map(DURHAM_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["status_normalized"] = mapped["Status"].astype("string").str.strip().str.lower()
    mapped = mapped[~mapped["status_normalized"].isin(DURHAM_NONCRIME_STATUSES)].copy()
    mapped["case_number"] = mapped["Case Number"].astype("string").str.strip()
    mapped["sequence"] = mapped["Sequence"].astype("string").str.strip()
    mapped["offense_row_id"] = (
        mapped["case_number"].fillna("")
        + "|"
        + mapped["sequence"].fillna("")
        + "|"
        + mapped["nibrs_code"].fillna("")
    )
    mapped = mapped[mapped["case_number"].ne("") & mapped["case_number"].ne("<NA>")].copy()
    mapped = mapped[mapped["offense_row_id"].ne("||")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    mapped["x"] = pd.to_numeric(mapped["X"], errors="coerce")
    mapped["y"] = pd.to_numeric(mapped["Y"], errors="coerce")
    point_candidates = mapped[
        mapped["x"].between(1_980_000, 2_090_000, inclusive="both")
        & mapped["y"].between(760_000, 870_000, inclusive="both")
    ].copy()

    bg = _load_city_bg_geometry(
        bg_crosswalk_path=bg_crosswalk_path,
        state_bg_zip=REPO_ROOT / "data" / "tiger_bg" / "tl_2020_37_bg.zip",
        jurisdiction_id=DURHAM_JURISDICTION_ID,
    )
    if point_candidates.empty or bg.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = gpd.GeoDataFrame(
        point_candidates.copy(),
        geometry=gpd.points_from_xy(point_candidates["x"], point_candidates["y"]),
        crs="EPSG:2264",
    ).to_crs("EPSG:4326")
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "reported_state_plane_xy"
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        matched.groupby(
            ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"],
            dropna=False,
        )
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = (
        counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"]
        .sum()
        .rename("city_total")
        .reset_index()
    )
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = np.where(
        pd.to_numeric(counts["city_total"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(counts["city_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    counts["city_name"] = DURHAM_CITY_NAME
    counts["validation_case_type"] = "municipal_validation_case"
    counts["validation_source_name"] = DURHAM_SOURCE_NAME
    counts["validation_source_url"] = DURHAM_SOURCE_URL
    counts["validation_case_notes"] = (
        "Official Durham Police annual UCR/NIBRS incident workbook; validation-only spatial shares "
        "from X/Y points in North Carolina State Plane feet. Unfounded and non-criminal closures "
        "are excluded; workbook totals are not treated as production count anchors."
    )
    surface = counts[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
            "validation_case_type",
            "validation_source_name",
            "validation_source_url",
            "validation_case_notes",
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)

    mapped_counts = (
        mapped.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("raw_offense_rows")
        .reset_index()
    )
    point_counts = (
        point_candidates.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("point_candidate_rows")
        .reset_index()
    )
    matched_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("matched_share_rows")
        .reset_index()
    )
    recon = mapped_counts.merge(point_counts, on=["year", "offense"], how="left").merge(
        matched_counts,
        on=["year", "offense"],
        how="left",
    )
    recon["city_name"] = DURHAM_CITY_NAME
    recon["jurisdiction_id"] = DURHAM_JURISDICTION_ID
    recon["validation_case_type"] = "municipal_validation_case"
    recon["validation_source_name"] = DURHAM_SOURCE_NAME
    recon["validation_source_url"] = DURHAM_SOURCE_URL
    recon["source_min_date_reported"] = mapped["report_date"].min()
    recon["source_max_date_reported"] = mapped["report_date"].max()
    for col in ["point_candidate_rows", "matched_share_rows"]:
        recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(
        pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0).gt(0),
        recon["matched_share_rows"] / pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return surface, recon


def build_validation_shares(
    *,
    year: int,
    city_shares_path: Path,
    bg_crosswalk_path: Path,
    raw_dir: Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(city_shares_path).copy()
    base_meta = _case_metadata_for_existing(base)
    base = base.merge(base_meta, on=["city_name", "jurisdiction_id", "state_fips"], how="left")

    validation_surfaces: list[pd.DataFrame] = []
    reconciliation_frames: list[pd.DataFrame] = []
    for builder in [
        _build_tucson_surface,
        _build_charlotte_surface,
        _build_dallas_surface,
        _build_kansas_city_surface,
        _build_cincinnati_surface,
        _build_san_diego_surface,
        _build_st_louis_surface,
        _build_milwaukee_surface,
        _build_durham_surface,
        _build_montgomery_county_surface,
    ]:
        surface, reconciliation = builder(
            year=year,
            raw_dir=raw_dir,
            bg_crosswalk_path=bg_crosswalk_path,
            force_refresh=force_refresh,
        )
        if not surface.empty:
            validation_surfaces.append(surface)
        if not reconciliation.empty:
            reconciliation_frames.append(reconciliation)

    validation_ids = {
        str(value)
        for surface in validation_surfaces
        for value in surface["jurisdiction_id"].dropna().astype(str).unique().tolist()
    }
    if validation_ids:
        base = base[~base["jurisdiction_id"].astype(str).isin(validation_ids)].copy()
    combined = pd.concat([base, *validation_surfaces], ignore_index=True, sort=False)

    # Gate-17 onboarded packet cities are part of the standard surface: append the
    # share-admitted packet surfaces (skipping any city already present) so a
    # rebuild can never silently drop them, then apply the texture policy to the
    # full 30-city assembly below.
    gate17_surfaces = _gate17_packet_surfaces(combined)
    if gate17_surfaces:
        combined = pd.concat([combined, *gate17_surfaces], ignore_index=True, sort=False)
    if GATE17_PACKET_REDERIVE_REPORT:
        quarantined = {
            k: v
            for k, v in GATE17_PACKET_REDERIVE_REPORT.items()
            if v.get("quarantined_dropped_incident_total")
        }
        print(
            "[gate17-packet-quarantine] re-derived-from-normalized packet cities: "
            + json.dumps(GATE17_PACKET_REDERIVE_REPORT, default=str)
            + " | quarantine-dropped: "
            + (json.dumps(quarantined, default=str) if quarantined else "none"),
            flush=True,
        )

    # Per-city-offense located-evidence texture policy: rape is default-deny except
    # for audit-verified-clean cities; denied pairs contribute no located share
    # evidence so the downstream posterior falls to the model prior. Applied once
    # on the assembled surface (idempotent for the already-filtered direct base).
    texture_policy = load_texture_policy()
    if not texture_policy.empty:
        combined, _texture_removed = filter_denied_texture_surface(combined, policy=texture_policy)

    combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")
    combined["block_group_geoid"] = combined["block_group_geoid"].astype("string").str.zfill(12)
    combined["state_fips"] = combined["state_fips"].astype("string").str.zfill(2)
    combined = combined.sort_values(
        ["city_name", "year", "offense", "block_group_geoid"],
        kind="mergesort",
    ).reset_index(drop=True)
    reconciliation = (
        pd.concat(reconciliation_frames, ignore_index=True, sort=False)
        if reconciliation_frames
        else pd.DataFrame()
    )
    return combined, reconciliation


def main() -> int:
    parser = argparse.ArgumentParser(description="Build expanded validation-only city incident share surface.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--city-shares-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet",
    )
    parser.add_argument(
        "--bg-crosswalk-path",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "inputs" / "next_phase_validation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet",
    )
    parser.add_argument(
        "--reconciliation-out",
        type=Path,
        default=REPO_ROOT / "materials" / "tables" / "next_phase_validation_case_reconciliation.csv",
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    combined, reconciliation = build_validation_shares(
        year=int(args.year),
        city_shares_path=args.city_shares_path,
        bg_crosswalk_path=args.bg_crosswalk_path,
        raw_dir=args.raw_dir,
        force_refresh=bool(args.force_refresh),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.reconciliation_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)
    reconciliation.to_csv(args.reconciliation_out, index=False)

    year_mask = pd.to_numeric(combined["year"], errors="coerce").eq(int(args.year))
    summary = {
        "year": int(args.year),
        "output": str(args.out),
        "reconciliation_output": str(args.reconciliation_out),
        "rows": int(len(combined)),
        "truth_case_count_for_year": int(combined.loc[year_mask, "jurisdiction_id"].nunique()),
        "case_types_for_year": combined.loc[year_mask, "validation_case_type"].value_counts(dropna=False).to_dict(),
        "validation_case_reconciliation": reconciliation.to_dict(orient="records"),
    }
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
