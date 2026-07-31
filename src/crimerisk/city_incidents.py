from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import warnings

import geopandas as gpd
import pandas as pd

from crimerisk.city_feed_quarantine import (
    flag_quarantined_coordinates,
    quarantine_counts_by_year_offense,
    record_quarantine_not_applicable,
)


class CityFeedCoverageError(RuntimeError):
    """Raised when a live city feed is asked for a year beyond its verified source coverage.

    Distinct from the generic RuntimeErrors already raised for schema drift / sanity-check
    failures only in that it always names the requested year and the verified ceiling so a
    caller (or the `build_*_incident_share_surface` warning wrapper that catches it) reports a
    loud, actionable gap instead of the pipeline silently reusing a prior year's data.
    """


NYC_JURISDICTION_ID = "36:municipal:place:3651000"
NYC_CITY_NAME = "New York"
NYC_HISTORICAL_DOWNLOAD_URL = "https://data.cityofnewyork.us/api/views/qgea-i56i/rows.csv?accessType=DOWNLOAD"
NYC_CURRENT_DOWNLOAD_URL = "https://data.cityofnewyork.us/api/views/5uac-w243/rows.csv?accessType=DOWNLOAD"
NYC_MINIMAL_COLUMNS: tuple[str, ...] = (
    "cmplnt_num",
    "cmplnt_fr_dt",
    "cmplnt_fr_tm",
    "cmplnt_to_dt",
    "cmplnt_to_tm",
    "rpt_dt",
    "ofns_desc",
    "pd_desc",
    "latitude",
    "longitude",
)
NYC_RAW_COLUMN_MAP: dict[str, str] = {
    "CMPLNT_NUM": "cmplnt_num",
    "CMPLNT_FR_DT": "cmplnt_fr_dt",
    "CMPLNT_FR_TM": "cmplnt_fr_tm",
    "CMPLNT_TO_DT": "cmplnt_to_dt",
    "CMPLNT_TO_TM": "cmplnt_to_tm",
    "RPT_DT": "rpt_dt",
    "OFNS_DESC": "ofns_desc",
    "PD_DESC": "pd_desc",
    "Latitude": "latitude",
    "Longitude": "longitude",
}

NYC_OFFENSE_MAP: dict[str, str] = {
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
NYC_MIN_INCIDENTS_IN_WINDOW = 100_000

CHICAGO_JURISDICTION_ID = "17:municipal:place:1714000"
CHICAGO_CITY_NAME = "Chicago"
CHICAGO_DOWNLOAD_URL = "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD"
CHICAGO_RESOURCE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
CHICAGO_MINIMAL_COLUMNS: tuple[str, ...] = (
    "id",
    "date",
    "primary_type",
    "description",
    "location_description",
    "latitude",
    "longitude",
)
CHICAGO_RAW_COLUMN_MAP: dict[str, str] = {
    "ID": "id",
    "Date": "date",
    "Primary Type": "primary_type",
    "Description": "description",
    "Location Description": "location_description",
    "Latitude": "latitude",
    "Longitude": "longitude",
}
CHICAGO_RESOURCE_COLUMN_MAP: dict[str, str] = {
    "id": "id",
    "date": "date",
    "primary_type": "primary_type",
    "description": "description",
    "location_description": "location_description",
    "latitude": "latitude",
    "longitude": "longitude",
}
CHICAGO_OFFENSE_MAP: dict[str, str] = {
    "09A": "murder",
    "11A": "rape",
    "11B": "rape",
    "11U": "rape",
    "120": "robbery",
    "12A": "robbery",
    "12B": "robbery",
    "12U": "robbery",
    "121": "robbery",
    "122": "robbery",
    "123": "robbery",
    "124": "robbery",
    "125": "robbery",
    "126": "robbery",
    "127": "robbery",
    "128": "robbery",
    "129": "robbery",
    "130": "robbery",
    "131": "robbery",
    "132": "robbery",
    "133": "robbery",
    "134": "robbery",
    "13A": "aggravated_assault",
    "220": "burglary",
    "22A": "burglary",
    "22B": "burglary",
    "23A": "larceny",
    "23B": "larceny",
    "23C": "larceny",
    "23D": "larceny",
    "23E": "larceny",
    "23F": "larceny",
    "23G": "larceny",
    "23H": "larceny",
    "23U": "larceny",
    "240": "motor_vehicle_theft",
}
CHICAGO_MIN_INCIDENTS_IN_WINDOW = 500_000
CHICAGO_MIN_INCIDENTS_PER_YEAR = 50_000
CHICAGO_MIN_GEOCODED_SHARE = 0.8
CHICAGO_BOUNDS = {
    "min_latitude": 41.55,
    "max_latitude": 42.10,
    "min_longitude": -87.95,
    "max_longitude": -87.45,
}

SEATTLE_JURISDICTION_ID = "53:municipal:place:5363000"
SEATTLE_CITY_NAME = "Seattle"
SEATTLE_DOWNLOAD_URL = "https://data.seattle.gov/api/views/tazs-3rd5/rows.csv?accessType=DOWNLOAD"
SEATTLE_MINIMAL_COLUMNS: tuple[str, ...] = (
    "report_number",
    "report_datetime",
    "offense_id",
    "offense_date",
    "offense_sub_category",
    "offense_category",
    "nibrs_offense_code_description",
    "nibrs_offense_code",
    "block_address",
    "beat",
    "precinct",
    "sector",
    "neighborhood",
    "reporting_area",
    "latitude",
    "longitude",
)
SEATTLE_RAW_COLUMN_MAP: dict[str, str] = {
    "Report Number": "report_number",
    "Report DateTime": "report_datetime",
    "Offense ID": "offense_id",
    "Offense Date": "offense_date",
    "Offense Sub Category": "offense_sub_category",
    "Offense Category": "offense_category",
    "NIBRS Offense Code Description": "nibrs_offense_code_description",
    "NIBRS_offense_code": "nibrs_offense_code",
    "Block Address": "block_address",
    "Beat": "beat",
    "Precinct": "precinct",
    "Sector": "sector",
    "Neighborhood": "neighborhood",
    "Reporting Area": "reporting_area",
    "Latitude": "latitude",
    "Longitude": "longitude",
}
SEATTLE_OFFENSE_MAP: dict[str, str] = {
    "09A": "murder",
    "11A": "rape",
    "120": "robbery",
    "12A": "robbery",
    "12B": "robbery",
    "13A": "aggravated_assault",
    "220": "burglary",
    "22A": "burglary",
    "22B": "burglary",
    "22U": "burglary",
    "23A": "larceny",
    "23B": "larceny",
    "23C": "larceny",
    "23D": "larceny",
    "23E": "larceny",
    "23F": "larceny",
    "23G": "larceny",
    "23H": "larceny",
    "23U": "larceny",
    "240": "motor_vehicle_theft",
}
SEATTLE_MIN_INCIDENTS_IN_WINDOW = 300_000
SEATTLE_MIN_INCIDENTS_PER_YEAR = 30_000
SEATTLE_MIN_GEOCODED_SHARE = 0.85
SEATTLE_BOUNDS = {
    "min_latitude": 47.45,
    "max_latitude": 47.78,
    "min_longitude": -122.46,
    "max_longitude": -122.20,
}
SEATTLE_BEAT_LAYER_URL = "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/Current_Beats/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson"
SEATTLE_PROJECTED_CRS = "EPSG:32610"

AUSTIN_JURISDICTION_ID = "48:municipal:place:4805000"
AUSTIN_CITY_NAME = "Austin"
AUSTIN_DOWNLOAD_URL = "https://data.austintexas.gov/api/views/fdj4-gpfu/rows.csv?accessType=DOWNLOAD"
AUSTIN_MINIMAL_COLUMNS: tuple[str, ...] = (
    "incident_report_number",
    "crime_type",
    "ucr_code",
    "occ_date_time",
    "rep_date_time",
    "location_type",
    "ucr_category",
    "category_description",
    "census_block_group",
)
AUSTIN_RAW_COLUMN_MAP: dict[str, str] = {
    "Incident Number": "incident_report_number",
    "Highest Offense Description": "crime_type",
    "Highest Offense Code": "ucr_code",
    "Occurred Date Time": "occ_date_time",
    "Report Date Time": "rep_date_time",
    "Location Type": "location_type",
    "UCR Category": "ucr_category",
    "Category Description": "category_description",
    "Census Block Group": "census_block_group",
}
AUSTIN_OFFENSE_MAP: dict[str, str] = {
    "09A": "murder",
    "11A": "rape",
    "11B": "rape",
    "11C": "rape",
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
AUSTIN_MIN_INCIDENTS_IN_WINDOW = 300_000
AUSTIN_MIN_INCIDENTS_PER_YEAR = 40_000

BOSTON_JURISDICTION_ID = "25:municipal:place:2507000"
BOSTON_CITY_NAME = "Boston"
BOSTON_DATASET_URL = "https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system"
BOSTON_PACKAGE_SHOW_URL = "https://data.boston.gov/api/3/action/package_show?id=crime-incident-reports-august-2015-to-date-source-new-system"
BOSTON_YEARLY_RESOURCE_IDS: dict[int, str] = {
    2019: "34e0ae6b-8c94-4998-ae9e-1b51551fe9ba",
    2020: "be047094-85fe-4104-a480-4fa3d03f9623",
    2021: "f4495ee9-c42c-4019-82c1-d067f07e45d2",
    2022: "313e56df-6d77-49d2-9c49-ee411f10cf58",
}
BOSTON_CURRENT_RESOURCE_ID = "b973d8cb-eeb2-4e7e-99da-c92938efc9c0"
BOSTON_MINIMAL_COLUMNS: tuple[str, ...] = (
    "incident_number",
    "offense_code",
    "offense_description",
    "occurred_on_date",
    "year",
    "latitude",
    "longitude",
)
BOSTON_RAW_COLUMN_MAP: dict[str, str] = {
    "INCIDENT_NUMBER": "incident_number",
    "OFFENSE_CODE": "offense_code",
    "OFFENSE_DESCRIPTION": "offense_description",
    "OCCURRED_ON_DATE": "occurred_on_date",
    "YEAR": "year",
    "Lat": "latitude",
    "Long": "longitude",
}
BOSTON_LARCENY_CODES = {str(code) for code in range(611, 620)}
BOSTON_MOTOR_VEHICLE_THEFT_CODES = {"706", "724", "727"}
BOSTON_LIVE_START_YEAR = 2019

BALTIMORE_JURISDICTION_ID = "24:municipal:place:2404000"
BALTIMORE_CITY_NAME = "Baltimore"
BALTIMORE_LEGACY_QUERY_URL = (
    "https://services1.arcgis.com/UWYHeuuJISiGmgXx/arcgis/rest/services/Part1_Crime_Beta/FeatureServer/0/query"
)
BALTIMORE_MINIMAL_COLUMNS: tuple[str, ...] = (
    "RowID",
    "CCNumber",
    "CrimeDateTime",
    "CrimeCode",
    "Description",
    "Latitude",
    "Longitude",
    "Total_Incidents",
)
BALTIMORE_PROMOTED_START_YEAR = 2024
BALTIMORE_PROMOTED_END_YEAR = 2024
BALTIMORE_PROMOTED_END_EXCLUSIVE = "2024-12-29 00:00:00"
BALTIMORE_MIN_INCIDENTS_IN_WINDOW = 40_000
BALTIMORE_MIN_GEOCODED_SHARE = 0.95
# Baltimore hard-cut from the legacy Part1_Crime_Beta feed to the NIBRS_GroupA_Crime_Data feed on
# 2025-01-01 (legacy frozen at BALTIMORE_PROMOTED_END_EXCLUSIVE); STATE.md 2025 refresh audit,
# 2026-07-10. Endpoint verified live against the FeatureServer metadata on 2026-07-10: layer name
# "NIBRS Group A Crime Data", coverage 2022-01-01 through present (updated weekly), fields
# RowID (OID) / CCNumber / CrimeDateTime (date) / CrimeCode / Description / Shooting /
# Latitude / Longitude / Total_Incidents, with CrimeCode carrying standard NIBRS offense codes
# (sampled 2025 rows: 13A/13B/120/220/23C/23F/23G/23H/240/290). The NIBRS lane below only serves
# years >= the cutover; 2022-2024 NIBRS-feed history is ignored in favor of the promoted legacy
# packet so the 2024 build is untouched.
BALTIMORE_NIBRS_CUTOVER_YEAR = 2025
BALTIMORE_NIBRS_MAX_VERIFIED_YEAR = 2025
BALTIMORE_NIBRS_QUERY_URL = (
    "https://services1.arcgis.com/UWYHeuuJISiGmgXx/arcgis/rest/services/NIBRS_GroupA_Crime_Data/FeatureServer/0/query"
)
BALTIMORE_NIBRS_MINIMAL_COLUMNS: tuple[str, ...] = (
    "RowID",
    "CCNumber",
    "CrimeDateTime",
    "CrimeCode",
    "Description",
    "Shooting",
    "Latitude",
    "Longitude",
    "Total_Incidents",
)
# Standard NIBRS Group A offense codes -> our 7 offenses, same mapping shape as other NIBRS-coded
# cities in this file (see e.g. ST_LOUIS_OFFENSE_MAP). Duplicated here (rather than imported from
# ST_LOUIS_OFFENSE_MAP) only because these constants are defined before it at module scope.
BALTIMORE_NIBRS_OFFENSE_MAP: dict[str, str] = {
    "09A": "murder",
    "11A": "rape",
    "11B": "rape",
    "11C": "rape",
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

PHILADELPHIA_JURISDICTION_ID = "42:municipal:cousub:4210160000"
PHILADELPHIA_CITY_NAME = "Philadelphia"
PHILADELPHIA_SQL_API_URL = "https://phl.carto.com/api/v2/sql"
PHILADELPHIA_MINIMAL_COLUMNS: tuple[str, ...] = (
    "objectid",
    "dc_key",
    "dispatch_date_time",
    "ucr_general",
    "text_general_code",
    "location_block",
    "latitude",
    "longitude",
)
PHILADELPHIA_RAW_COLUMN_MAP: dict[str, str] = {
    "objectid": "objectid",
    "dc_key": "dc_key",
    "dispatch_date_time": "dispatch_date_time",
    "ucr_general": "ucr_general",
    "text_general_code": "text_general_code",
    "location_block": "location_block",
    "latitude": "latitude",
    "longitude": "longitude",
}
PHILADELPHIA_MIN_INCIDENTS_PER_YEAR = 100_000
PHILADELPHIA_MIN_GEOCODED_SHARE = 0.90

WASHINGTON_DC_JURISDICTION_ID = "11:municipal:place:1150000"
WASHINGTON_DC_CITY_NAME = "Washington"
# DC mints a year-suffixed sibling dataset each January (crime-incidents-in-<year>); the schema has
# been stable across vintages, so the URL is safely derived from the target year.
WASHINGTON_DC_DOWNLOAD_URL_TEMPLATE = "https://opendata.dc.gov/datasets/DCGIS::crime-incidents-in-{year}.csv"
WASHINGTON_DC_MINIMAL_COLUMNS: tuple[str, ...] = (
    "objectid",
    "ccn",
    "report_dat",
    "start_date",
    "end_date",
    "offense_label",
    "method",
    "block",
    "block_group",
    "census_tract",
    "latitude",
    "longitude",
)
WASHINGTON_DC_RAW_COLUMN_MAP: dict[str, str] = {
    "OBJECTID": "objectid",
    "CCN": "ccn",
    "REPORT_DAT": "report_dat",
    "START_DATE": "start_date",
    "END_DATE": "end_date",
    "OFFENSE": "offense_label",
    "METHOD": "method",
    "BLOCK": "block",
    "BLOCK_GROUP": "block_group",
    "CENSUS_TRACT": "census_tract",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
}
WASHINGTON_DC_LIVE_START_YEAR = 2024
# Verified via the STATE.md 2025 refresh audit (2026-07-10): the crime-incidents-in-<year> sibling
# for 2025 exists. Bump only after confirming the next vintage is live at opendata.dc.gov.
WASHINGTON_DC_MAX_VERIFIED_YEAR = 2025
WASHINGTON_DC_MIN_INCIDENTS_IN_WINDOW = 25_000
WASHINGTON_DC_MIN_BLOCK_GROUP_ASSIGNMENT_SHARE = 0.995

DENVER_JURISDICTION_ID = "08:municipal:place:0820000"
DENVER_CITY_NAME = "Denver"
DENVER_QUERY_URL = "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_CRIME_OFFENSES_P/FeatureServer/324/query"
DENVER_MINIMAL_COLUMNS: tuple[str, ...] = (
    "objectid",
    "offense_category_id",
    "offense_type_id",
    "offense_code",
    "offense_code_extension",
    "victim_count",
    "reported_date",
    "incident_address",
    "latitude",
    "longitude",
    "geometry_x",
    "geometry_y",
)
DENVER_RAW_COLUMN_MAP: dict[str, str] = {
    "OBJECTID": "objectid",
    "OFFENSE_CATEGORY_ID": "offense_category_id",
    "OFFENSE_TYPE_ID": "offense_type_id",
    "OFFENSE_CODE": "offense_code",
    "OFFENSE_CODE_EXTENSION": "offense_code_extension",
    "VICTIM_COUNT": "victim_count",
    "REPORTED_DATE": "reported_date",
    "INCIDENT_ADDRESS": "incident_address",
    "GEO_LAT": "latitude",
    "GEO_LON": "longitude",
    "geometry_x": "geometry_x",
    "geometry_y": "geometry_y",
}
DENVER_LIVE_START_YEAR = 2024
# The Denver query has no year embedded in the URL (year is a REPORTED_DATE WHERE-clause bound), so
# raising this ceiling is a pure coverage-verification bump, not a source/schema change. Verified via
# the STATE.md 2025 refresh audit (2026-07-10): "all 17 live-feed cities verified alive with full
# 2025." Bump only after re-confirming the feed is alive for the new year.
DENVER_LIVE_END_YEAR = 2025
DENVER_MIN_INCIDENTS_IN_WINDOW = 40_000
DENVER_MIN_GEOCODED_SHARE = 0.999

MINNEAPOLIS_JURISDICTION_ID = "27:municipal:place:2743000"
MINNEAPOLIS_CITY_NAME = "Minneapolis"
MINNEAPOLIS_QUERY_URL = "https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/Crime_Data/FeatureServer/0/query"
MINNEAPOLIS_MINIMAL_COLUMNS: tuple[str, ...] = (
    "objectid",
    "type",
    "nibrs_code",
    "offense_category",
    "offense_label",
    "crime_count",
    "reported_date",
    "address",
    "latitude",
    "longitude",
    "geometry_x",
    "geometry_y",
)
MINNEAPOLIS_RAW_COLUMN_MAP: dict[str, str] = {
    "OBJECTID": "objectid",
    "Type": "type",
    "NIBRS_Code": "nibrs_code",
    "Offense_Category": "offense_category",
    "Offense": "offense_label",
    "Crime_Count": "crime_count",
    "Reported_Date": "reported_date",
    "Address": "address",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "geometry_x": "geometry_x",
    "geometry_y": "geometry_y",
}
MINNEAPOLIS_LIVE_START_YEAR = 2024
# No year embedded in the URL (year is a Reported_Date WHERE-clause bound); raising this ceiling is
# a pure coverage-verification bump. Verified via the STATE.md 2025 refresh audit (2026-07-10): "all
# 17 live-feed cities verified alive with full 2025." Bump only after re-confirming the feed is alive.
MINNEAPOLIS_LIVE_END_YEAR = 2025
MINNEAPOLIS_MIN_INCIDENTS_IN_WINDOW = 30_000
MINNEAPOLIS_MIN_GEOCODED_SHARE = 0.999

ST_LOUIS_JURISDICTION_ID = "29:municipal:place:2965000"
ST_LOUIS_CITY_NAME = "St. Louis"
ST_LOUIS_SOURCE_NAME = "SLMPD NIBRS Crime Files"
ST_LOUIS_SOURCE_URL = "https://www.slmpd.org/stats/"
ST_LOUIS_LIVE_START_YEAR = 2024
ST_LOUIS_MIN_INCIDENTS_IN_WINDOW = 15_000
ST_LOUIS_MIN_GEOCODED_SHARE = 0.95
# SLMPD's monthly NIBRS files are year-dated (STATE.md 2025 refresh audit, 2026-07-10 notes
# "year-dated export URLs"), but the *upload* subpath (the folder date, e.g. "2024/03") lags the
# data month irregularly -- January and February 2024 both posted in March, and December 2024
# posted the following January. That lag schedule is not something a formula can safely infer for a
# year that hasn't happened yet, so this is a per-year registry rather than a template: each year's
# monthly URLs must be confirmed against https://www.slmpd.org/stats/ and added explicitly. Only
# 2024 is populated/verified today.
ST_LOUIS_MONTHLY_FILES_BY_YEAR: dict[int, tuple[tuple[int, str, str], ...]] = {
    2024: (
        (1, "january", "https://slmpd.org/wp-content/uploads/2024/03/January2024.csv"),
        (2, "february", "https://slmpd.org/wp-content/uploads/2024/03/February2024.csv"),
        (3, "march", "https://slmpd.org/wp-content/uploads/2024/05/Downloadable-NIBRS-Crime-File-March-2024-CSV.csv"),
        (4, "april", "https://slmpd.org/wp-content/uploads/2024/05/April2024.csv"),
        (5, "may", "https://slmpd.org/wp-content/uploads/2024/06/May2024.csv"),
        (6, "june", "https://slmpd.org/wp-content/uploads/2024/07/June2024.csv"),
        (7, "july", "https://slmpd.org/wp-content/uploads/2024/08/July2024.csv"),
        (8, "august", "https://slmpd.org/wp-content/uploads/2024/09/August2024.csv"),
        (9, "september", "https://slmpd.org/wp-content/uploads/2024/10/September2024.csv"),
        (10, "october", "https://slmpd.org/wp-content/uploads/2024/11/October2024.csv"),
        (11, "november", "https://slmpd.org/wp-content/uploads/2024/12/November2024.csv"),
        (12, "december", "https://slmpd.org/wp-content/uploads/2025/01/December2024.csv"),
    ),
}
# Coverage ceiling derives from the registry so there is exactly one place to extend.
ST_LOUIS_LIVE_END_YEAR = max(ST_LOUIS_MONTHLY_FILES_BY_YEAR)
ST_LOUIS_MINIMAL_COLUMNS: tuple[str, ...] = (
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
ST_LOUIS_OFFENSE_MAP: dict[str, str] = {
    "09A": "murder",
    "11A": "rape",
    "11B": "rape",
    "11C": "rape",
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
# Monthly SLMPD files re-emit supplemented incidents with revised VictimNum rows; including VictimNum
# creates a multi-offense overcount against the annual control, so the stable public identity excludes it.
ST_LOUIS_VICTIM_COUNTED_OFFENSES: set[str] = set()

MESA_JURISDICTION_ID = "04:municipal:place:0446000"
MESA_CITY_NAME = "Mesa"
MESA_CURRENT_RESOURCE_URL = "https://data.mesaaz.gov/resource/hpbg-2wph.csv"
MESA_CURRENT_RAW_COLUMN_MAP: dict[str, str] = {
    "crime_id": "crime_id",
    "crime_type": "crime_type",
    "report_date": "report_date",
    "report_year": "report_year",
    "occurred_date": "occurred_date",
    "address": "address",
    "city": "city",
    "state": "state",
    "nibrs_code": "nibrs_code",
    "nibrs_description": "nibrs_description",
    "location": "location",
}
MESA_MINIMAL_COLUMNS: tuple[str, ...] = (
    "crime_id",
    "crime_type",
    "report_date",
    "report_year",
    "occurred_date",
    "address",
    "city",
    "state",
    "nibrs_code",
    "nibrs_description",
    "latitude",
    "longitude",
    "location",
)
MESA_LOCATION_RE = re.compile(r"POINT\s*\(([-0-9.]+)\s+([-0-9.]+)\)")
MESA_LIVE_START_YEAR = 2020

SAN_FRANCISCO_JURISDICTION_ID = "06:municipal:place:0667000"
SAN_FRANCISCO_CITY_NAME = "San Francisco"
SAN_FRANCISCO_RESOURCE_URL = "https://data.sfgov.org/resource/wg3w-h783.csv"
SAN_FRANCISCO_MINIMAL_COLUMNS: tuple[str, ...] = (
    "incident_id",
    "incident_number",
    "incident_date",
    "incident_year",
    "incident_category",
    "incident_subcategory",
    "incident_description",
    "report_type_description",
    "police_district",
    "latitude",
    "longitude",
)
SAN_FRANCISCO_RAW_COLUMN_MAP: dict[str, str] = {
    "incident_id": "incident_id",
    "incident_number": "incident_number",
    "incident_date": "incident_date",
    "incident_year": "incident_year",
    "incident_category": "incident_category",
    "incident_subcategory": "incident_subcategory",
    "incident_description": "incident_description",
    "report_type_description": "report_type_description",
    "police_district": "police_district",
    "latitude": "latitude",
    "longitude": "longitude",
}
SAN_FRANCISCO_INITIAL_TYPES = {"Initial", "Coplogic Initial", "Vehicle Initial"}

NYC_PUBLISHED_RECON = {
    2024: {
        "source_name": "NYPD year-end 2024 crime press release",
        "source_url": "https://www.nyc.gov/site/nypd/news/pr001/crime-down-across-new-york-city-2024-3-662-fewer-crimes",
        "counts": {
            "murder": 377,
            "rape": 1748,
            "robbery": 16556,
            "aggravated_assault": 29417,
            "burglary": 13029,
            "larceny": 48423,
            "motor_vehicle_theft": 14194,
        },
        "comparison_quality": {
            "murder": "exact",
            "rape": "exact",
            "robbery": "exact",
            "aggravated_assault": "exact",
            "burglary": "exact",
            "larceny": "approximate",
            "motor_vehicle_theft": "exact",
        },
        "notes": "Official NYPD year-end citywide index-crime totals. Most categories align directly to the V2 offense frame, but NYPD publishes Grand Larceny rather than the broader FBI/UCR larceny-theft concept, so the larceny comparison is only a rough sanity check.",
    }
}

CHICAGO_PUBLISHED_RECON = {
    2023: {
        "source_name": "Chicago Police Department 2023 Annual Report",
        "source_url": "https://home.chicagopolice.org/wp-content/uploads/2023-Annual-Report.pdf",
        "counts": {
            "murder": 618,
            "rape": 1941,
            "robbery": 11064,
            "aggravated_assault": 7707,
            "burglary": 7450,
            "larceny": 57226,
            "motor_vehicle_theft": 29198,
        },
        "comparison_quality": {
            "murder": "exact",
            "rape": "approximate",
            "robbery": "exact",
            "aggravated_assault": "approximate",
            "burglary": "exact",
            "larceny": "exact",
            "motor_vehicle_theft": "exact",
        },
        "notes": "CPD publishes Criminal Sexual Assault and Aggravated Assault separately from other related categories such as Aggravated Battery. The rape and aggravated_assault comparisons are therefore approximate sanity checks rather than exact one-to-one reconciliations.",
    }
}

SEATTLE_PUBLISHED_RECON = {
    2024: {
        "source_name": "WASPC Crime in Washington 2024 Seattle PD agency table",
        "source_url": "https://www.waspc.org/assets/CJIS/2024%20Crime%20in%20Washington-compressed.pdf",
        "counts": {
            "murder": 52,
            "rape": 315,
            "robbery": 1717,
            "aggravated_assault": 3952,
            "burglary": 8914,
            "larceny": 24565,
            "motor_vehicle_theft": 7452,
        },
        "comparison_quality": {
            "murder": "approximate",
            "rape": "approximate",
            "robbery": "approximate",
            "aggravated_assault": "approximate",
            "burglary": "exact",
            "larceny": "exact",
            "motor_vehicle_theft": "exact",
        },
        "notes": "WASPC publishes Seattle PD agency-level 2024 offense totals that align closely with the core V2 frame. Property offenses are close enough to treat as exact checks, while murder, rape, robbery, and aggravated assault should still be treated as approximate comparisons because the public SPD feed is offense-row based and can diverge from published person-crime counting rules.",
    }
}

AUSTIN_PUBLISHED_RECON = {
    2024: {
        "source_name": "APD Quarterly Update Data and Backup Materials",
        "source_url": "https://services.austintexas.gov/edims/document.cfm?id=448925",
        "counts": {
            "murder": 65,
            "rape": 668,
            "robbery": 844,
            "aggravated_assault": 3039,
            "burglary": 4411,
            "larceny": 21985,
            "motor_vehicle_theft": 5968,
        },
        "comparison_quality": {
            "murder": "exact",
            "rape": "approximate",
            "robbery": "exact",
            "aggravated_assault": "exact",
            "burglary": "exact",
            "larceny": "exact",
            "motor_vehicle_theft": "exact",
        },
        "notes": "Austin APD publishes Sexual Assault rather than a strict V2 rape bucket, so rape remains approximate even though the other listed offense lines are treated as direct public comparisons.",
    }
}

BOSTON_PUBLISHED_RECON = {
    2024: {
        "source_name": "Boston Police Department Crime Statistics: January 1, 2024 - December 29, 2024 vs. 2023",
        "source_url": "https://police.boston.gov/wp-content/uploads/2024/12/Pages-from-Weekly-Crime-Overview_-12-29-24.pdf",
        "counts": {
            "murder": 24,
            "rape": 163,
            "robbery": 818,
            "aggravated_assault": 2570,
            "burglary": 1092,
            "larceny": 10860,
            "motor_vehicle_theft": 1126,
        },
        "comparison_quality": {
            "murder": "blocked",
            "rape": "blocked",
            "robbery": "blocked",
            "aggravated_assault": "blocked",
            "burglary": "blocked",
            "larceny": "approximate",
            "motor_vehicle_theft": "approximate",
        },
        "notes": "Official BPD year-end Part One PDF reviewed in the Boston packet. Treat Boston as offense-selective only for larceny and motor_vehicle_theft. The public feed remains blocked for rape, robbery, aggravated_assault, and burglary, while murder stays pending explicit policy sign-off.",
    }
}

ST_LOUIS_PUBLISHED_RECON = {
    2024: {
        "source_name": "Crime in the United States 2024 annual city control",
        "source_url": "state/controls/jurisdiction_controls_2024.parquet",
        "counts": {
            "murder": 150,
            "rape": 161,
            "robbery": 692,
            "aggravated_assault": 2788,
            "burglary": 2273,
            "larceny": 9462,
            "motor_vehicle_theft": 4090,
        },
        "comparison_quality": {
            "murder": "official_control",
            "rape": "official_control",
            "robbery": "official_control",
            "aggravated_assault": "official_control",
            "burglary": "official_control",
            "larceny": "official_control",
            "motor_vehicle_theft": "official_control",
        },
        "notes": (
            "SLMPD monthly public NIBRS CSV rows are used for within-city shares. Annual totals remain "
            "the CIUS/control-lane counts unless an offense clears the strict reconciliation gate."
        ),
    }
}


def _download_file(url: str, out_path: Path, *, force_download: bool = False) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_download:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "crimerisk-v2/1.0"})
    with urlopen(request, timeout=600) as response, tmp_path.open("wb") as sink:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            sink.write(chunk)
    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty incident snapshot from {url}")
    tmp_path.replace(out_path)
    return out_path


def _download_boston_file(url: str, out_path: Path, *, force_download: bool = False) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_download:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    class _NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = build_opener(_NoRedirect)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    redirect_url = url
    try:
        with opener.open(request, timeout=600) as response:
            redirect_url = response.geturl()
    except HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise
        redirect_url = exc.headers.get("Location") or url
    redirect_url = redirect_url.replace("https://s3.amazonaws.com:443/", "https://s3.amazonaws.com/")

    final_request = Request(redirect_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(final_request, timeout=600) as response, tmp_path.open("wb") as sink:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            sink.write(chunk)
    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty Boston incident snapshot from {redirect_url}")
    tmp_path.replace(out_path)
    return out_path


def _load_boston_resource_urls() -> dict[str, str]:
    request = Request(BOSTON_PACKAGE_SHOW_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    resources = payload.get("result", {}).get("resources", [])
    resource_urls: dict[str, str] = {}
    for resource in resources:
        resource_id = str(resource.get("id", "")).strip()
        resource_url = str(resource.get("url", "")).strip()
        if resource_id and resource_url:
            resource_urls[resource_id] = resource_url
    return resource_urls


def _download_csv_query(*, base_url: str, params: dict[str, str], out_path: Path, force_download: bool = False) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_download:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    query_url = f"{base_url}?{urlencode(params)}"
    request = Request(query_url, headers={"User-Agent": "crimerisk-v2/1.0"})
    with urlopen(request, timeout=600) as response, tmp_path.open("wb") as sink:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            sink.write(chunk)
    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty incident snapshot from {query_url}")
    tmp_path.replace(out_path)
    return out_path


def _download_arcgis_query_frame(
    *,
    base_url: str,
    params: dict[str, str],
    out_path: Path,
    force_download: bool = False,
    page_size: int = 2000,
    return_geometry: bool = False,
    geometry_x_col: str = "geometry_x",
    geometry_y_col: str = "geometry_y",
    out_sr: int | None = None,
) -> pd.DataFrame:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_download:
        return pd.read_parquet(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        query_params = dict(params)
        query_params["resultOffset"] = str(int(offset))
        query_params["resultRecordCount"] = str(int(page_size))
        if return_geometry:
            query_params["returnGeometry"] = "true"
            if out_sr is not None:
                query_params["outSR"] = str(int(out_sr))
        query_url = f"{base_url}?{urlencode(query_params)}"
        request = Request(query_url, headers={"User-Agent": "crimerisk-v2/1.0"})
        with urlopen(request, timeout=600) as response:
            payload = json.load(response)
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
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return frame


def _prepared_city_cache_path(raw_dir: Path, filename: str) -> Path:
    return raw_dir / "parsed" / filename


def _load_single_field_offense_crosswalk(
    offense_crosswalk_csv: Path,
    *,
    field_aliases: dict[str, str] | None = None,
    include_counting_basis: bool = False,
) -> pd.DataFrame:
    columns = ["source_field", "source_values", "offense"]
    if include_counting_basis:
        columns.append("counting_basis")

    crosswalk = pd.read_csv(offense_crosswalk_csv, dtype=str).fillna("").copy()
    required = {"source_field", "source_value", "v2_offense", "include_in_v2"}
    if crosswalk.empty or not required.issubset(crosswalk.columns):
        return pd.DataFrame(columns=columns)

    field_aliases = {str(key).strip().lower(): str(value).strip() for key, value in (field_aliases or {}).items()}
    crosswalk["include_flag"] = crosswalk["include_in_v2"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
    crosswalk["offense"] = crosswalk["v2_offense"].astype(str).str.strip()
    crosswalk = crosswalk[crosswalk["include_flag"] & crosswalk["offense"].ne("")].copy()
    if crosswalk.empty:
        return pd.DataFrame(columns=columns)

    crosswalk["source_field"] = (
        crosswalk["source_field"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda value: field_aliases.get(value, value))
    )
    crosswalk["source_values"] = crosswalk["source_value"].astype(str).str.split("|")
    crosswalk["source_values"] = crosswalk["source_values"].apply(
        lambda values: [str(value).strip().upper() for value in values if str(value).strip()]
    )
    if include_counting_basis:
        crosswalk["counting_basis"] = crosswalk["counting_basis"].astype(str).str.strip().str.lower()
        crosswalk["counting_basis"] = crosswalk["counting_basis"].replace({"": "row_count"})
    return crosswalk[columns].copy()


def _load_reconciliation_summary_published_recon(
    reconciliation_summary_csv: Path,
    packet_offense_status_csv: Path,
    *,
    year: int = 2024,
) -> dict[int, dict[str, object]]:
    summary = pd.read_csv(reconciliation_summary_csv, dtype=str).fillna("").copy()
    statuses = pd.read_csv(packet_offense_status_csv, dtype=str).fillna("").copy()
    if summary.empty:
        return {}

    quality_map: dict[str, str] = {}
    notes_map: dict[str, str] = {}
    for row in statuses.itertuples(index=False):
        offense = str(getattr(row, "offense", "")).strip()
        if not offense:
            continue
        comparison_class = str(getattr(row, "comparison_class", "")).strip()
        ready = str(getattr(row, "production_ready", "")).strip().lower() in {"yes", "true", "1", "partial"}
        status = str(getattr(row, "city_share_integration_status", "")).strip().lower()
        quality_map[offense] = comparison_class or ("exact" if ready and status == "offense_selective" else "blocked")
        notes_map[offense] = str(getattr(row, "notes", "")).strip()

    counts: dict[str, float] = {}
    source_names: list[str] = []
    source_urls: list[str] = []
    for row in summary.itertuples(index=False):
        offense = str(getattr(row, "offense", "")).strip()
        if not offense:
            continue
        count = pd.to_numeric(getattr(row, f"official_count_{int(year)}", ""), errors="coerce")
        if pd.isna(count):
            continue
        counts[offense] = float(count)
        source_name = str(getattr(row, "official_source_name", "")).strip()
        source_url = str(getattr(row, "official_source_url", "")).strip()
        if source_name:
            source_names.append(source_name)
        if source_url:
            source_urls.append(source_url)
    if not counts:
        return {}

    notes = " | ".join([notes_map.get(offense, "") for offense in counts if notes_map.get(offense, "")])
    return {
        int(year): {
            "source_name": " ; ".join(dict.fromkeys(source_names)),
            "source_url": " ; ".join(dict.fromkeys(source_urls)),
            "counts": counts,
            "comparison_quality": {offense: quality_map.get(offense, "approximate") for offense in counts},
            "notes": notes or "Official packet reconciliation summary.",
        }
    }


def _split_delimited_codes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [code.strip() for code in str(value).split(",") if code.strip()]


def _load_nyc_snapshot_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(NYC_RAW_COLUMN_MAP),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
    )
    return df.rename(columns=NYC_RAW_COLUMN_MAP)


def _load_nyc_categories(categories_csv: Path) -> pd.DataFrame:
    cats = pd.read_csv(categories_csv, dtype=str).fillna("").copy()
    cats["ofns_desc"] = cats["ofns_desc"].astype(str).str.strip()
    cats["pd_desc"] = cats["pd_desc"].astype(str).str.strip()
    cats["NIBRS_Offense_Code"] = cats["NIBRS_Offense_Code"].astype(str).str.strip()
    cats["offense"] = cats["NIBRS_Offense_Code"].map(NYC_OFFENSE_MAP)
    cats = cats[cats["offense"].notna()].copy()
    rape_keep = cats["offense"].ne("rape") | cats["ofns_desc"].eq("RAPE") | cats["pd_desc"].str.startswith("RAPE")
    cats = cats[rape_keep].copy()
    return cats[["ofns_desc", "pd_desc", "offense"]].drop_duplicates()


def _load_chicago_snapshot_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(CHICAGO_RAW_COLUMN_MAP),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
    )
    df = df.rename(columns=CHICAGO_RAW_COLUMN_MAP)
    missing = set(CHICAGO_MINIMAL_COLUMNS) - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Chicago incident source schema changed; missing columns: {missing_cols}")
    return df


def _load_chicago_categories(categories_csv: Path) -> pd.DataFrame:
    cats = pd.read_csv(categories_csv, dtype=str).fillna("").copy()
    cats["Primary Type"] = cats["Primary Type"].astype(str).str.strip()
    cats["Description"] = cats["Description"].astype(str).str.strip()
    cats["NIBRS_Offense_Code"] = cats["NIBRS_Offense_Code"].astype(str).str.strip()
    cats["offense"] = cats["NIBRS_Offense_Code"].map(CHICAGO_OFFENSE_MAP)
    cats = cats[cats["offense"].notna()].copy()
    cats = cats.rename(columns={"Primary Type": "primary_type", "Description": "description"})
    return cats[["primary_type", "description", "offense"]].drop_duplicates()


def _load_seattle_snapshot_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(SEATTLE_RAW_COLUMN_MAP),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
    )
    df = df.rename(columns=SEATTLE_RAW_COLUMN_MAP)
    missing = set(SEATTLE_MINIMAL_COLUMNS) - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Seattle incident source schema changed; missing columns: {missing_cols}")
    return df


def _load_mesa_snapshot_csv(path: Path, *, column_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(column_map),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
    )
    df = df.rename(columns=column_map)
    for col in MESA_MINIMAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    missing = set(MESA_MINIMAL_COLUMNS) - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Mesa incident source schema changed; missing columns: {missing_cols}")
    return df.loc[:, list(MESA_MINIMAL_COLUMNS)].copy()


def _load_mesa_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    effective_start_year = max(int(start_year), MESA_LIVE_START_YEAR)
    effective_end_year = int(end_year)
    cache_path = _prepared_city_cache_path(raw_dir, f"mesa_minimal_snapshot_{effective_start_year}_{effective_end_year}.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "crime_id" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()

    if raw.empty:
        frames: list[pd.DataFrame] = []
        for year in range(effective_start_year, effective_end_year + 1):
            csv_path = _download_csv_query(
                base_url=MESA_CURRENT_RESOURCE_URL,
                params={
                    "$select": ",".join(MESA_CURRENT_RAW_COLUMN_MAP.keys()),
                    "$where": f"report_year='{year}'",
                    "$limit": "1000000",
                },
                out_path=raw_dir / f"mesa_{year}.csv",
                force_download=force_refresh,
            )
            frames.append(_load_mesa_snapshot_csv(csv_path, column_map=MESA_CURRENT_RAW_COLUMN_MAP))
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(MESA_MINIMAL_COLUMNS))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)
    raw["crime_id"] = raw["crime_id"].astype(str).str.strip()
    raw = raw[raw["crime_id"].ne("")].copy()
    raw = raw.drop_duplicates("crime_id", keep="last")
    return raw


def _load_seattle_bg_geometry(
    *,
    wa_bg_zip: Path,
    bg_crosswalk_parquet: Path,
) -> gpd.GeoDataFrame:
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(SEATTLE_JURISDICTION_ID)].copy()
    if crosswalk.empty:
        return gpd.GeoDataFrame(columns=["block_group_geoid", "jurisdiction_id", "state_fips", "geometry"], geometry="geometry", crs="EPSG:4326")

    bg = gpd.read_file(wa_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    return bg


def _load_seattle_beat_bg_weights(
    *,
    raw_dir: Path,
    wa_bg_zip: Path,
    bg_crosswalk_parquet: Path,
) -> pd.DataFrame:
    cache_path = _prepared_city_cache_path(raw_dir, "seattle_beat_bg_weights.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0:
        weights = pd.read_parquet(cache_path)
        required = {"beat", "block_group_geoid", "jurisdiction_id", "state_fips", "beat_bg_share"}
        if required.issubset(set(weights.columns)):
            return weights

    bg = _load_seattle_bg_geometry(wa_bg_zip=wa_bg_zip, bg_crosswalk_parquet=bg_crosswalk_parquet)
    if bg.empty:
        return pd.DataFrame(columns=["beat", "block_group_geoid", "jurisdiction_id", "state_fips", "beat_bg_share"])

    beats = gpd.read_file(SEATTLE_BEAT_LAYER_URL)[["beat", "geometry"]].copy()
    beats["beat"] = beats["beat"].astype(str).str.strip().str.upper()
    beats = beats[beats["beat"].ne("")].copy()
    if bg.crs != beats.crs:
        bg = bg.to_crs(beats.crs)

    bg_proj = bg.to_crs(SEATTLE_PROJECTED_CRS)
    beats_proj = beats.to_crs(SEATTLE_PROJECTED_CRS)
    overlap = gpd.overlay(
        bg_proj[["block_group_geoid", "jurisdiction_id", "state_fips", "geometry"]],
        beats_proj[["beat", "geometry"]],
        how="intersection",
    )
    if overlap.empty:
        return pd.DataFrame(columns=["beat", "block_group_geoid", "jurisdiction_id", "state_fips", "beat_bg_share"])

    overlap["intersection_area"] = overlap.geometry.area
    weights = (
        overlap.groupby(["beat", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)["intersection_area"]
        .sum()
        .rename("intersection_area")
        .reset_index()
    )
    beat_totals = weights.groupby("beat", dropna=False)["intersection_area"].sum().rename("beat_area").reset_index()
    weights = weights.merge(beat_totals, on="beat", how="left")
    weights["beat_bg_share"] = weights["intersection_area"] / weights["beat_area"]
    weights = weights[["beat", "block_group_geoid", "jurisdiction_id", "state_fips", "beat_bg_share"]].copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(cache_path, index=False)
    return weights


def _assign_seattle_block_groups(
    *,
    mapped: pd.DataFrame,
    raw_dir: Path,
    wa_bg_zip: Path,
    bg_crosswalk_parquet: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapped = mapped.copy()
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    mapped["beat"] = mapped.get("beat", pd.Series(index=mapped.index, dtype="object")).astype(str).str.strip().str.upper()

    geocoded = mapped[
        mapped["latitude"].between(SEATTLE_BOUNDS["min_latitude"], SEATTLE_BOUNDS["max_latitude"], inclusive="both")
        & mapped["longitude"].between(SEATTLE_BOUNDS["min_longitude"], SEATTLE_BOUNDS["max_longitude"], inclusive="both")
    ].copy()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()

    bg = _load_seattle_bg_geometry(wa_bg_zip=wa_bg_zip, bg_crosswalk_parquet=bg_crosswalk_parquet)
    if bg.empty:
        return pd.DataFrame(), geocoded_counts, pd.DataFrame()

    direct_counts = pd.DataFrame()
    if not geocoded.empty:
        incidents = gpd.GeoDataFrame(
            geocoded[["offense_id", "year", "offense", "latitude", "longitude"]].copy(),
            geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
            crs="EPSG:4326",
        )
        bg_direct = bg
        if bg_direct.crs != incidents.crs:
            bg_direct = bg_direct.to_crs(incidents.crs)
        matched = gpd.sjoin(incidents, bg_direct, how="inner", predicate="within")[
            ["offense_id", "year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"]
        ]
        matched = matched.drop_duplicates("offense_id").copy()
        if not matched.empty:
            direct_counts = (
                matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
                .size()
                .rename("incident_count")
                .reset_index()
            )
            direct_counts["geocode_quality_tier"] = "reported_latlon"

    unresolved = mapped.loc[~mapped["offense_id"].isin(geocoded["offense_id"])].copy()
    unresolved = unresolved[unresolved["beat"].ne("") & unresolved["beat"].ne("OOJ")].copy()
    beat_counts = pd.DataFrame()
    if not unresolved.empty:
        beat_weights = _load_seattle_beat_bg_weights(
            raw_dir=raw_dir,
            wa_bg_zip=wa_bg_zip,
            bg_crosswalk_parquet=bg_crosswalk_parquet,
        )
        unresolved_counts = (
            unresolved.groupby(["year", "offense", "beat"], dropna=False)
            .size()
            .rename("unresolved_incident_count")
            .reset_index()
        )
        beat_counts = unresolved_counts.merge(beat_weights, on="beat", how="inner")
        if not beat_counts.empty:
            beat_counts["incident_count"] = beat_counts["unresolved_incident_count"] * beat_counts["beat_bg_share"]
            beat_counts["geocode_quality_tier"] = "beat_fallback"
            beat_counts = beat_counts[
                ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "incident_count", "geocode_quality_tier"]
            ].copy()

    assigned = pd.concat([direct_counts, beat_counts], ignore_index=True)
    if not assigned.empty:
        assigned = (
            assigned.groupby(
                ["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"],
                dropna=False,
            )
            .agg(
                incident_count=("incident_count", "sum"),
                geocode_quality_tier=(
                    "geocode_quality_tier",
                    lambda values: "+".join(sorted({str(value) for value in values if str(value).strip()})),
                ),
            )
            .reset_index()
        )
    return assigned, geocoded_counts, beat_counts


def _parse_nyc_datetime(df: pd.DataFrame) -> pd.Series:
    out = df.copy()
    for tm_col in ["cmplnt_fr_tm", "cmplnt_to_tm"]:
        if tm_col in out.columns:
            out[tm_col] = out[tm_col].astype(str).replace({"24:00:00": "23:59:59", "nan": "", "(null)": ""})
    for dt_col in ["cmplnt_fr_dt", "cmplnt_to_dt", "rpt_dt"]:
        if dt_col in out.columns:
            out[dt_col] = out[dt_col].astype(str).replace({"nan": "", "(null)": ""})

    fr_date = out.get("cmplnt_fr_dt", pd.Series(index=out.index, dtype="object")).replace("", pd.NA)
    to_date = out.get("cmplnt_to_dt", pd.Series(index=out.index, dtype="object")).replace("", pd.NA)
    rpt_date = out.get("rpt_dt", pd.Series(index=out.index, dtype="object")).replace("", pd.NA)
    fr_time = out.get("cmplnt_fr_tm", pd.Series(index=out.index, dtype="object")).replace("", pd.NA)
    to_time = out.get("cmplnt_to_tm", pd.Series(index=out.index, dtype="object")).replace("", pd.NA)

    fr_date = fr_date.fillna(to_date).fillna(rpt_date)
    fr_time = fr_time.fillna(to_time).fillna("00:00:00")
    to_date = to_date.fillna(fr_date)
    to_time = to_time.fillna(fr_time)

    start_date = pd.to_datetime(fr_date, errors="coerce")
    end_date = pd.to_datetime(to_date, errors="coerce")
    rpt_date_dt = pd.to_datetime(rpt_date, errors="coerce")
    start_time = pd.to_timedelta(fr_time.astype(str), errors="coerce").fillna(pd.Timedelta(0))
    end_time = pd.to_timedelta(to_time.astype(str), errors="coerce").fillna(pd.Timedelta(0))
    start = start_date + start_time
    end = end_date + end_time
    end = end.fillna(start)
    midpoint = start + (end - start) / 2
    midpoint = midpoint.fillna(start).fillna(end).fillna(rpt_date_dt)
    return midpoint


def _map_nyc_offenses(raw: pd.DataFrame, cats: pd.DataFrame) -> pd.DataFrame:
    mapped = raw.merge(cats, on=["ofns_desc", "pd_desc"], how="left")
    unresolved = mapped["offense"].isna()
    if not bool(unresolved.any()):
        return mapped

    fallback = (
        cats.groupby("ofns_desc", dropna=False)["offense"]
        .nunique()
        .rename("offense_count")
        .reset_index()
    )
    fallback = fallback[fallback["offense_count"].eq(1)].merge(
        cats[["ofns_desc", "offense"]].drop_duplicates(),
        on="ofns_desc",
        how="left",
    )[["ofns_desc", "offense"]]
    mapped.loc[unresolved, "offense"] = (
        mapped.loc[unresolved, ["ofns_desc"]]
        .merge(fallback, on="ofns_desc", how="left")["offense"]
        .to_numpy()
    )
    return mapped


def _drop_quarantined_coordinates(
    frame: pd.DataFrame,
    *,
    city_key: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    offense_col: str | None = "offense",
) -> pd.DataFrame:
    """THE coordinate-quarantine call. Every production city builder routes through here.

    Stage 4 screen b3: `flag_quarantined_coordinates` was called from 4 of the 13 production
    builders with three different argument shapes, and nothing detected the other 9 -- a
    quarantine row keyed to them was inert rather than wrong. One helper, called once per
    builder, gives one implementation and one place where the offense scoping is decided
    (per-row when the frame carries an offense column, `any`-rows always).

    A frame with no coordinate columns DECLARES itself coordinate-free rather than silently
    skipping: Austin publishes a source block group and never a point, and that is a fact about
    the feed, not a gap in the wiring. `city_shares.build_city_incident_share_surface` asserts
    every enabled city reached one of the two outcomes.
    """
    if lat_col not in frame.columns or lon_col not in frame.columns:
        record_quarantine_not_applicable(city_key, reason="feed_carries_no_point_coordinates")
        return frame
    mask = flag_quarantined_coordinates(
        frame,
        city_key=city_key,
        lat_col=lat_col,
        lon_col=lon_col,
        offense_col=offense_col if offense_col and offense_col in frame.columns else None,
    )
    if not bool(mask.any()):
        return frame
    return frame.loc[~mask].copy()


def _empty_city_surface() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "block_group_geoid",
            "incident_count",
            "share_within_city",
            "geocode_quality_tier",
        ]
    )


def _empty_city_reconciliation() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "city_key",
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "raw_year_total",
            "mapped_offense_count",
            "geocoded_offense_count",
            "matched_offense_count",
            "quarantined_coordinate_count",
            "final_share_count",
            "published_count",
            "published_source_name",
            "published_source_url",
            "published_comparison_quality",
            "published_notes",
            "diff_final_vs_published",
            "pct_diff_final_vs_published",
        ]
    )


def _build_city_reconciliation_table(
    *,
    city_key: str,
    city_name: str,
    jurisdiction_id: str,
    state_fips: str,
    raw_year_totals: pd.Series,
    mapped_counts: pd.DataFrame,
    geocoded_counts: pd.DataFrame,
    matched_counts: pd.DataFrame,
    final_counts: pd.DataFrame,
    quarantined_counts: pd.DataFrame | None = None,
    published_recon: dict[int, dict[str, object]],
) -> pd.DataFrame:
    offense_index = pd.Index(
        [
            "murder",
            "rape",
            "robbery",
            "aggravated_assault",
            "burglary",
            "larceny",
            "motor_vehicle_theft",
        ],
        dtype="object",
    )
    if final_counts.empty:
        return _empty_city_reconciliation()

    years = sorted({int(year) for year in final_counts["year"].dropna().unique()})
    base = pd.MultiIndex.from_product([years, offense_index], names=["year", "offense"]).to_frame(index=False)

    def _prep(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["year", "offense", value_col])
        out = df[["year", "offense", value_col]].copy()
        out["year"] = out["year"].astype(int)
        out["offense"] = out["offense"].astype(str)
        return out.groupby(["year", "offense"], dropna=False)[value_col].sum().reset_index()

    raw_rows = raw_year_totals.reset_index()
    raw_rows.columns = ["year", "raw_year_total"]
    raw_rows["year"] = raw_rows["year"].astype(int)

    recon = base.merge(raw_rows, on="year", how="left")
    recon = recon.merge(_prep(mapped_counts, "mapped_offense_count"), on=["year", "offense"], how="left")
    recon = recon.merge(_prep(geocoded_counts, "geocoded_offense_count"), on=["year", "offense"], how="left")
    recon = recon.merge(_prep(matched_counts, "matched_offense_count"), on=["year", "offense"], how="left")
    if quarantined_counts is None:
        quarantined_counts = pd.DataFrame(columns=["year", "offense", "quarantined_coordinate_count"])
    recon = recon.merge(_prep(quarantined_counts, "quarantined_coordinate_count"), on=["year", "offense"], how="left")
    recon = recon.merge(_prep(final_counts, "final_share_count"), on=["year", "offense"], how="left")

    published_rows: list[dict[str, object]] = []
    for year, payload in published_recon.items():
        counts = payload.get("counts", {}) or {}
        qualities = payload.get("comparison_quality", {}) or {}
        notes = str(payload.get("notes", ""))
        for offense in offense_index:
            published_rows.append(
                {
                    "year": int(year),
                    "offense": offense,
                    "published_count": counts.get(offense),
                    "published_source_name": payload.get("source_name"),
                    "published_source_url": payload.get("source_url"),
                    "published_comparison_quality": qualities.get(offense, "approximate"),
                    "published_notes": notes,
                }
            )
    published = pd.DataFrame(published_rows)
    recon = recon.merge(published, on=["year", "offense"], how="left")

    recon["city_key"] = city_key
    recon["city_name"] = city_name
    recon["jurisdiction_id"] = jurisdiction_id
    recon["state_fips"] = state_fips

    numeric_cols = [
        "raw_year_total",
        "mapped_offense_count",
        "geocoded_offense_count",
        "matched_offense_count",
        "quarantined_coordinate_count",
        "final_share_count",
        "published_count",
    ]
    for col in numeric_cols:
        recon[col] = pd.to_numeric(recon[col], errors="coerce")

    recon["diff_final_vs_published"] = recon["final_share_count"] - recon["published_count"]
    recon["pct_diff_final_vs_published"] = recon["diff_final_vs_published"] / recon["published_count"]
    return recon[
        [
            "city_key",
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "year",
            "offense",
            "raw_year_total",
            "mapped_offense_count",
            "geocoded_offense_count",
            "matched_offense_count",
            "quarantined_coordinate_count",
            "final_share_count",
            "published_count",
            "published_source_name",
            "published_source_url",
            "published_comparison_quality",
            "published_notes",
            "diff_final_vs_published",
            "pct_diff_final_vs_published",
        ]
    ].sort_values(["year", "offense"], kind="mergesort").reset_index(drop=True)


def _load_nyc_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = _prepared_city_cache_path(raw_dir, f"nyc_minimal_snapshot_{int(start_year)}_{int(end_year)}.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "cmplnt_num" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()
    if raw.empty and "cmplnt_num" not in raw.columns:
        early_csv = _download_file(
            NYC_HISTORICAL_DOWNLOAD_URL,
            raw_dir / "nyc_historical_download.csv",
            force_download=force_refresh,
        )
        late_csv = _download_file(
            NYC_CURRENT_DOWNLOAD_URL,
            raw_dir / "nyc_current_download.csv",
            force_download=force_refresh,
        )
        frames = [_load_nyc_snapshot_csv(early_csv), _load_nyc_snapshot_csv(late_csv)]
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(NYC_MINIMAL_COLUMNS))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)
    raw["cmplnt_num"] = raw["cmplnt_num"].astype(str).str.strip()
    raw = raw[raw["cmplnt_num"].ne("")].copy()
    raw = raw.drop_duplicates("cmplnt_num", keep="last")
    return raw


def _load_chicago_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = _prepared_city_cache_path(raw_dir, f"chicago_minimal_snapshot_{int(start_year)}_{int(end_year)}.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "id" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()
    if raw.empty and "id" not in raw.columns:
        frames: list[pd.DataFrame] = []
        for year in range(int(start_year), int(end_year) + 1):
            year_csv = raw_dir / f"chicago_{year}.csv"
            params = {
                "$select": ",".join(CHICAGO_RESOURCE_COLUMN_MAP.keys()),
                "$where": f"date between '{year}-01-01T00:00:00' and '{year}-12-31T23:59:59'",
                "$limit": "5000000",
            }
            csv_path = _download_csv_query(
                base_url=CHICAGO_RESOURCE_URL,
                params=params,
                out_path=year_csv,
                force_download=force_refresh,
            )
            df_year = pd.read_csv(
                csv_path,
                dtype=str,
                usecols=lambda col: col in set(CHICAGO_RESOURCE_COLUMN_MAP),
                na_values=["", "NA", "(null)"],
                keep_default_na=True,
            )
            df_year = df_year.rename(columns=CHICAGO_RESOURCE_COLUMN_MAP)
            frames.append(df_year)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(CHICAGO_MINIMAL_COLUMNS))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)
    missing = set(CHICAGO_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Chicago incident cache is missing required columns: {missing_cols}")
    raw["id"] = raw["id"].astype(str).str.strip()
    raw = raw[raw["id"].ne("")].copy()
    raw = raw.drop_duplicates("id", keep="last")
    return raw


def _load_seattle_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = _prepared_city_cache_path(raw_dir, f"seattle_minimal_snapshot_{int(start_year)}_{int(end_year)}.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        cached_cols = set(raw.columns)
        required_cols = set(SEATTLE_MINIMAL_COLUMNS)
        if "offense_id" not in raw.columns or not required_cols.issubset(cached_cols):
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()
    if raw.empty and "offense_id" not in raw.columns:
        csv_path = _download_file(SEATTLE_DOWNLOAD_URL, raw_dir / "seattle_download.csv", force_download=force_refresh)
        raw = _load_seattle_snapshot_csv(csv_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)
    missing = set(SEATTLE_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Seattle incident cache is missing required columns: {missing_cols}")
    raw["offense_id"] = raw["offense_id"].astype(str).str.strip()
    raw = raw[raw["offense_id"].ne("")].copy()
    raw = raw.drop_duplicates("offense_id", keep="last")
    return raw


def _load_austin_snapshot_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    rename_map = {src: dst for src, dst in AUSTIN_RAW_COLUMN_MAP.items() if src in raw.columns}
    raw = raw.rename(columns=rename_map)
    missing = [col for col in AUSTIN_MINIMAL_COLUMNS if col not in raw.columns]
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Austin source is missing required columns: {missing_cols}")
    return raw.loc[:, list(AUSTIN_MINIMAL_COLUMNS)].copy()


def _load_austin_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = _prepared_city_cache_path(raw_dir, f"austin_minimal_snapshot_{int(start_year)}_{int(end_year)}.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "incident_report_number" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()
    if raw.empty and "incident_report_number" not in raw.columns:
        csv_path = _download_file(AUSTIN_DOWNLOAD_URL, raw_dir / "austin_download.csv", force_download=force_refresh)
        raw = _load_austin_snapshot_csv(csv_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)
    missing = set(AUSTIN_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Austin incident cache is missing required columns: {missing_cols}")
    raw["incident_report_number"] = raw["incident_report_number"].astype(str).str.strip()
    raw = raw[raw["incident_report_number"].ne("")].copy()
    raw = raw.drop_duplicates("incident_report_number", keep="last")
    return raw


def _load_boston_snapshot_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(BOSTON_RAW_COLUMN_MAP),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
        low_memory=False,
    )
    raw = raw.rename(columns=BOSTON_RAW_COLUMN_MAP)
    missing = set(BOSTON_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Boston source is missing required columns: {missing_cols}")
    return raw.loc[:, list(BOSTON_MINIMAL_COLUMNS)].copy()


def _load_boston_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    effective_start_year = max(int(start_year), BOSTON_LIVE_START_YEAR)
    effective_end_year = int(end_year)
    cache_path = _prepared_city_cache_path(
        raw_dir,
        f"boston_minimal_snapshot_{effective_start_year}_{effective_end_year}.parquet",
    )
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "incident_number" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()
    if raw.empty and "incident_number" not in raw.columns:
        resource_urls = _load_boston_resource_urls()
        frames: list[pd.DataFrame] = []
        for year in range(effective_start_year, min(effective_end_year, 2022) + 1):
            resource_id = BOSTON_YEARLY_RESOURCE_IDS.get(year)
            if not resource_id:
                continue
            url = resource_urls.get(resource_id)
            if not url:
                raise RuntimeError(f"Boston package metadata is missing resource URL for year {year}")
            csv_path = _download_boston_file(
                url,
                raw_dir / f"boston_{year}.csv",
                force_download=force_refresh,
            )
            frames.append(_load_boston_snapshot_csv(csv_path))
        if effective_end_year >= 2023:
            current_url = resource_urls.get(BOSTON_CURRENT_RESOURCE_ID)
            if not current_url:
                raise RuntimeError("Boston package metadata is missing the current 2023-present resource URL")
            csv_path = _download_boston_file(
                current_url,
                raw_dir / "boston_2023_present.csv",
                force_download=force_refresh,
            )
            current = _load_boston_snapshot_csv(csv_path)
            current_year = pd.to_numeric(current["year"], errors="coerce")
            current = current[current_year.between(max(effective_start_year, 2023), effective_end_year, inclusive="both")].copy()
            frames.append(current)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(BOSTON_MINIMAL_COLUMNS))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)
    missing = set(BOSTON_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Boston incident cache is missing required columns: {missing_cols}")
    raw["incident_number"] = raw["incident_number"].astype(str).str.strip()
    raw = raw[raw["incident_number"].ne("")].copy()
    raw = raw.drop_duplicates(
        subset=["incident_number", "offense_code", "offense_description", "occurred_on_date", "year", "latitude", "longitude"],
        keep="last",
    )
    return raw


def _load_san_francisco_crosswalk(crosswalk_csv: Path) -> pd.DataFrame:
    crosswalk = pd.read_csv(crosswalk_csv, dtype=str).fillna("").copy()
    crosswalk = crosswalk[crosswalk["include_in_v2"].astype(str).str.lower().eq("yes")].copy()
    crosswalk["normalized_category"] = crosswalk["normalized_category"].astype(str).str.upper().str.strip()
    crosswalk["incident_description"] = crosswalk["incident_description"].astype(str).str.strip().str.upper()
    crosswalk["v2_offense"] = crosswalk["v2_offense"].astype(str).str.strip()
    crosswalk = crosswalk[crosswalk["v2_offense"].ne("")].copy()
    return crosswalk[["normalized_category", "incident_description", "v2_offense"]].drop_duplicates()


def _normalize_san_francisco_category(series: pd.Series) -> pd.Series:
    recode = {
        "LARCENY THEFT": "LARCENY/THEFT",
        "FORGERY AND COUNTERFEITING": "FORGERY/COUNTERFEITING",
        "HUMAN TRAFFICKING, COMMERCIAL SEX ACTS": "HUMAN TRAFFICKING (A), COMMERCIAL SEX ACTS",
        "MOTOR VEHICLE THEFT?": "MOTOR VEHICLE THEFT",
        "OTHER": "OTHER OFFENSES",
        "SUSPICIOUS": "SUSPICIOUS OCC",
        "MOTOR VEHICLE THEFT": "VEHICLE THEFT",
        "WARRANT": "WARRANTS",
        "WEAPON LAWS": "WEAPONS CARRYING ETC",
        "WEAPONS OFFENCE": "WEAPONS OFFENSE",
    }
    return series.astype(str).str.upper().str.strip().replace(recode)


def _load_san_francisco_snapshot_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(SAN_FRANCISCO_RAW_COLUMN_MAP),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
    )
    raw = raw.rename(columns=SAN_FRANCISCO_RAW_COLUMN_MAP)
    missing = set(SAN_FRANCISCO_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"San Francisco source is missing required columns: {missing_cols}")
    return raw


def _load_san_francisco_raw_snapshot(
    *,
    raw_dir: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_path = _prepared_city_cache_path(raw_dir, f"san_francisco_minimal_snapshot_{int(start_year)}_{int(end_year)}.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "incident_id" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()
    if raw.empty and "incident_id" not in raw.columns:
        frames: list[pd.DataFrame] = []
        for year in range(int(start_year), int(end_year) + 1):
            csv_path = _download_csv_query(
                base_url=SAN_FRANCISCO_RESOURCE_URL,
                params={
                    "$select": ",".join(SAN_FRANCISCO_RAW_COLUMN_MAP.keys()),
                    "$where": f"incident_year='{year}'",
                    "$limit": "500000",
                },
                out_path=raw_dir / f"san_francisco_{year}.csv",
                force_download=force_refresh,
            )
            frames.append(_load_san_francisco_snapshot_csv(csv_path))
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(SAN_FRANCISCO_MINIMAL_COLUMNS))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)
    missing = set(SAN_FRANCISCO_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"San Francisco incident cache is missing required columns: {missing_cols}")
    raw["incident_number"] = raw["incident_number"].astype(str).str.strip()
    raw = raw[raw["incident_number"].ne("")].copy()
    return raw


def _prepare_san_francisco_incidents(
    *,
    raw_dir: Path,
    crosswalk_csv: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    raw = _load_san_francisco_raw_snapshot(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_refresh,
    )
    raw["incident_date"] = pd.to_datetime(raw["incident_date"], errors="coerce")
    raw["year"] = pd.to_numeric(raw["incident_year"], errors="coerce").fillna(raw["incident_date"].dt.year).astype("Int64")
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()

    crosswalk = _load_san_francisco_crosswalk(crosswalk_csv)
    raw["normalized_category"] = _normalize_san_francisco_category(raw["incident_category"])
    raw["incident_description"] = raw["incident_description"].astype(str).str.strip().str.upper()
    mapped = raw.merge(
        crosswalk,
        on=["normalized_category", "incident_description"],
        how="left",
    )
    mapped = mapped[mapped["v2_offense"].notna() & mapped["v2_offense"].astype(str).ne("")].copy()
    mapped["report_type_description"] = mapped["report_type_description"].astype(str).str.strip()
    mapped = mapped[mapped["report_type_description"].isin(SAN_FRANCISCO_INITIAL_TYPES)].copy()
    mapped = mapped.drop_duplicates(["incident_number", "v2_offense"])
    mapped = mapped[mapped["police_district"].astype(str).ne("Out of SF")].copy()
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    mapped["offense_row_id"] = (
        mapped["incident_number"].astype(str).str.strip()
        + "::"
        + mapped["v2_offense"].astype(str).str.strip()
    )
    return mapped, raw_year_totals


def _load_san_francisco_published_recon(published_reference_csv: Path, packet_status_csv: Path) -> dict[int, dict[str, object]]:
    published = pd.read_csv(published_reference_csv, dtype=str).fillna("")
    statuses = pd.read_csv(packet_status_csv, dtype=str).fillna("")
    quality_map = {}
    for row in statuses.itertuples(index=False):
        offense = str(row.offense)
        ready = str(row.production_ready).strip().lower() in {"yes", "true", "1", "partial"}
        status = str(row.city_share_integration_status).strip().lower()
        quality_map[offense] = "exact" if ready and status == "offense_selective" else "blocked"
    recon: dict[int, dict[str, object]] = {}
    for year, group in published.groupby("year", dropna=False):
        counts = {str(r.offense): float(r.published_count) for r in group.itertuples(index=False)}
        recon[int(year)] = {
            "source_name": str(group["source_name"].iloc[0]),
            "source_url": str(group["source_url"].iloc[0]),
            "counts": counts,
            "comparison_quality": {offense: quality_map.get(offense, "approximate") for offense in counts},
            "notes": str(group["source_basis"].iloc[0]),
        }
    return recon


def build_nyc_incident_share_surface(
    *,
    raw_dir: Path,
    categories_csv: Path,
    ny_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    raw = _load_nyc_raw_snapshot(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_source_refresh,
    )
    raw["incident_dt"] = _parse_nyc_datetime(raw)
    raw["year"] = raw["incident_dt"].dt.year
    in_window = raw["year"].between(int(start_year), int(end_year), inclusive="both")
    if int(in_window.sum()) < NYC_MIN_INCIDENTS_IN_WINDOW:
        warnings.warn(
            (
                "NYC incident source coverage failed sanity check; "
                f"only {int(in_window.sum())} incidents fell in {int(start_year)}-{int(end_year)}. "
                "Skipping NYC city-incident surface until the source contract is corrected."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    raw = raw[in_window].copy()

    cats = _load_nyc_categories(categories_csv)
    raw["ofns_desc"] = raw["ofns_desc"].astype(str).str.strip()
    raw["pd_desc"] = raw["pd_desc"].astype(str).str.strip()
    raw = _map_nyc_offenses(raw, cats)
    raw = raw[raw["offense"].notna()].copy()

    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw = raw[
        raw["latitude"].between(40.45, 40.95, inclusive="both")
        & raw["longitude"].between(-74.30, -73.65, inclusive="both")
    ].copy()
    if raw.empty:
        return _empty_city_surface()
    raw = _drop_quarantined_coordinates(raw, city_key="new_york")
    if raw.empty:
        return _empty_city_surface()

    incidents = gpd.GeoDataFrame(
        raw[["cmplnt_num", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(raw["longitude"], raw["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(ny_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)

    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[["cmplnt_num", "year", "offense", "block_group_geoid"]]
    matched = matched.drop_duplicates("cmplnt_num").copy()

    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = matched.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(NYC_JURISDICTION_ID)].copy()
    if matched.empty:
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = NYC_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_nyc_incident_reconciliation(
    *,
    raw_dir: Path,
    categories_csv: Path,
    ny_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    raw = _load_nyc_raw_snapshot(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_source_refresh,
    )
    raw["incident_dt"] = _parse_nyc_datetime(raw)
    raw["year"] = raw["incident_dt"].dt.year
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()

    cats = _load_nyc_categories(categories_csv)
    raw["ofns_desc"] = raw["ofns_desc"].astype(str).str.strip()
    raw["pd_desc"] = raw["pd_desc"].astype(str).str.strip()
    mapped = _map_nyc_offenses(raw, cats)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()

    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    geocoded = mapped[
        mapped["latitude"].between(40.45, 40.95, inclusive="both")
        & mapped["longitude"].between(-74.30, -73.65, inclusive="both")
    ].copy()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()

    if geocoded.empty:
        return _empty_city_reconciliation()

    quarantine_mask = flag_quarantined_coordinates(geocoded, city_key="new_york", offense_col="offense")
    quarantined_counts = quarantine_counts_by_year_offense(geocoded, quarantine_mask)
    located = geocoded.loc[~quarantine_mask].copy()
    if located.empty:
        return _empty_city_reconciliation()

    incidents = gpd.GeoDataFrame(
        located[["cmplnt_num", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(located["longitude"], located["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(ny_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[["cmplnt_num", "year", "offense", "block_group_geoid"]]
    matched = matched.drop_duplicates("cmplnt_num").copy()
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = matched.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(NYC_JURISDICTION_ID)].copy()
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()

    final_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("final_share_count")
        .reset_index()
    )
    return _build_city_reconciliation_table(
        city_key="new_york",
        city_name=NYC_CITY_NAME,
        jurisdiction_id=NYC_JURISDICTION_ID,
        state_fips="36",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        quarantined_counts=quarantined_counts,
        published_recon=NYC_PUBLISHED_RECON,
    )


def build_chicago_incident_share_surface(
    *,
    raw_dir: Path,
    categories_csv: Path,
    il_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        raw = _load_chicago_raw_snapshot(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Chicago incident source could not be loaded cleanly: {exc}. Skipping Chicago city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    raw["incident_dt"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["year"] = raw["incident_dt"].dt.year
    year_index = pd.Index(range(int(start_year), int(end_year) + 1), dtype="int64")
    year_counts = raw.groupby("year", dropna=False).size().reindex(year_index, fill_value=0)
    total_in_window = int(year_counts.sum())
    undercovered_years = [int(year) for year, count in year_counts.items() if int(count) < CHICAGO_MIN_INCIDENTS_PER_YEAR]
    if total_in_window < CHICAGO_MIN_INCIDENTS_IN_WINDOW or undercovered_years:
        warnings.warn(
            (
                "Chicago incident source coverage failed sanity check; "
                f"{int(start_year)}-{int(end_year)} total={total_in_window}, "
                f"undercovered years={undercovered_years}. "
                "Skipping Chicago city-incident surface until the source contract is corrected."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()

    cats = _load_chicago_categories(categories_csv)
    raw["primary_type"] = raw["primary_type"].astype(str).str.strip()
    raw["description"] = raw["description"].astype(str).str.strip()
    raw = raw.merge(cats, on=["primary_type", "description"], how="left")
    raw = raw[raw["offense"].notna()].copy()
    if raw.empty:
        warnings.warn(
            "Chicago offense harmonization produced no mapped incidents. Skipping Chicago city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    geocoded = raw[
        raw["latitude"].between(CHICAGO_BOUNDS["min_latitude"], CHICAGO_BOUNDS["max_latitude"], inclusive="both")
        & raw["longitude"].between(CHICAGO_BOUNDS["min_longitude"], CHICAGO_BOUNDS["max_longitude"], inclusive="both")
    ].copy()
    geocoded_share = float(len(geocoded)) / float(len(raw)) if len(raw) else 0.0
    if geocoded_share < CHICAGO_MIN_GEOCODED_SHARE:
        warnings.warn(
            (
                "Chicago coordinate coverage failed sanity check; "
                f"only {geocoded_share:.1%} of mapped incidents had plausible reported lat/lon. "
                "Skipping Chicago city-incident surface until the source contract is corrected."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    geocoded = _drop_quarantined_coordinates(geocoded, city_key="chicago")
    if geocoded.empty:
        return _empty_city_surface()

    incidents = gpd.GeoDataFrame(
        geocoded[["id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )

    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(CHICAGO_JURISDICTION_ID)].copy()
    if crosswalk.empty:
        warnings.warn(
            "Chicago jurisdiction was not present in the block-group crosswalk. Skipping Chicago city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    bg = gpd.read_file(il_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    if bg.empty:
        warnings.warn(
            "Chicago block-group geometry could not be matched to the jurisdiction crosswalk. Skipping Chicago city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)

    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["id", "year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"]
    ]
    matched = matched.drop_duplicates("id").copy()
    if matched.empty:
        warnings.warn(
            "Chicago incidents did not spatially match any Chicago block group. Skipping Chicago city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = CHICAGO_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_chicago_incident_reconciliation(
    *,
    raw_dir: Path,
    categories_csv: Path,
    il_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        raw = _load_chicago_raw_snapshot(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    raw["incident_dt"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["year"] = raw["incident_dt"].dt.year
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()

    cats = _load_chicago_categories(categories_csv)
    raw["primary_type"] = raw["primary_type"].astype(str).str.strip()
    raw["description"] = raw["description"].astype(str).str.strip()
    mapped = raw.merge(cats, on=["primary_type", "description"], how="left")
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()

    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    geocoded = mapped[
        mapped["latitude"].between(CHICAGO_BOUNDS["min_latitude"], CHICAGO_BOUNDS["max_latitude"], inclusive="both")
        & mapped["longitude"].between(CHICAGO_BOUNDS["min_longitude"], CHICAGO_BOUNDS["max_longitude"], inclusive="both")
    ].copy()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()
    if geocoded.empty:
        return _empty_city_reconciliation()

    quarantine_mask = flag_quarantined_coordinates(geocoded, city_key="chicago", offense_col="offense")
    quarantined_counts = quarantine_counts_by_year_offense(geocoded, quarantine_mask)
    located = geocoded.loc[~quarantine_mask].copy()
    if located.empty:
        return _empty_city_reconciliation()

    incidents = gpd.GeoDataFrame(
        located[["id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(located["longitude"], located["latitude"]),
        crs="EPSG:4326",
    )
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(CHICAGO_JURISDICTION_ID)].copy()
    if crosswalk.empty:
        return _empty_city_reconciliation()
    bg = gpd.read_file(il_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    if bg.empty:
        return _empty_city_reconciliation()
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["id", "year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"]
    ]
    matched = matched.drop_duplicates("id").copy()
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    final_counts = (
        matched.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("final_share_count")
        .reset_index()
    )
    return _build_city_reconciliation_table(
        city_key="chicago",
        city_name=CHICAGO_CITY_NAME,
        jurisdiction_id=CHICAGO_JURISDICTION_ID,
        state_fips="17",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        quarantined_counts=quarantined_counts,
        published_recon=CHICAGO_PUBLISHED_RECON,
    )


def build_seattle_incident_share_surface(
    *,
    raw_dir: Path,
    wa_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        raw = _load_seattle_raw_snapshot(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Seattle incident source could not be loaded cleanly: {exc}. Skipping Seattle city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    raw["incident_dt"] = pd.to_datetime(raw["offense_date"], errors="coerce")
    report_dt = pd.to_datetime(raw["report_datetime"], errors="coerce")
    raw["incident_dt"] = raw["incident_dt"].fillna(report_dt)
    raw["year"] = raw["incident_dt"].dt.year
    year_index = pd.Index(range(int(start_year), int(end_year) + 1), dtype="int64")
    year_counts = raw.groupby("year", dropna=False).size().reindex(year_index, fill_value=0)
    total_in_window = int(year_counts.sum())
    undercovered_years = [int(year) for year, count in year_counts.items() if int(count) < SEATTLE_MIN_INCIDENTS_PER_YEAR]
    if total_in_window < SEATTLE_MIN_INCIDENTS_IN_WINDOW or undercovered_years:
        warnings.warn(
            (
                "Seattle incident source coverage failed sanity check; "
                f"{int(start_year)}-{int(end_year)} total={total_in_window}, "
                f"undercovered years={undercovered_years}. "
                "Skipping Seattle city-incident surface until the source contract is corrected."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()

    raw["nibrs_offense_code"] = raw["nibrs_offense_code"].astype(str).str.strip()
    raw["offense"] = raw["nibrs_offense_code"].map(SEATTLE_OFFENSE_MAP)
    raw = raw[raw["offense"].notna()].copy()
    if raw.empty:
        warnings.warn(
            "Seattle offense harmonization produced no mapped incidents. Skipping Seattle city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    # Applied to `raw`, not to the `geocoded` subset below: `_assign_seattle_block_groups` is
    # handed `raw` and derives its own geocoded subset from it, so filtering the local subset
    # would have been inert -- the exact failure mode Stage 4 screen b3 was about. Beat-fallback
    # rows carry no coordinate and are untouched.
    raw = _drop_quarantined_coordinates(raw, city_key="seattle")
    geocoded = raw[
        raw["latitude"].between(SEATTLE_BOUNDS["min_latitude"], SEATTLE_BOUNDS["max_latitude"], inclusive="both")
        & raw["longitude"].between(SEATTLE_BOUNDS["min_longitude"], SEATTLE_BOUNDS["max_longitude"], inclusive="both")
    ].copy()
    geocoded_share = float(len(geocoded)) / float(len(raw)) if len(raw) else 0.0
    if geocoded_share < SEATTLE_MIN_GEOCODED_SHARE:
        warnings.warn(
            (
                "Seattle coordinate coverage failed sanity check; "
                f"only {geocoded_share:.1%} of mapped incidents had plausible reported lat/lon. "
                "Skipping Seattle city-incident surface until the source contract is corrected."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    assigned, _, _ = _assign_seattle_block_groups(
        mapped=raw,
        raw_dir=raw_dir,
        wa_bg_zip=wa_bg_zip,
        bg_crosswalk_parquet=bg_crosswalk_parquet,
    )
    if assigned.empty:
        warnings.warn(
            "Seattle incidents did not spatially match or recover into Seattle block groups. Skipping Seattle city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    counts = assigned.copy()
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = SEATTLE_CITY_NAME
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_seattle_incident_reconciliation(
    *,
    raw_dir: Path,
    wa_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        raw = _load_seattle_raw_snapshot(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    raw["incident_dt"] = pd.to_datetime(raw["offense_date"], errors="coerce")
    report_dt = pd.to_datetime(raw["report_datetime"], errors="coerce")
    raw["incident_dt"] = raw["incident_dt"].fillna(report_dt)
    raw["year"] = raw["incident_dt"].dt.year
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()

    raw["nibrs_offense_code"] = raw["nibrs_offense_code"].astype(str).str.strip()
    mapped = raw.copy()
    mapped["offense"] = mapped["nibrs_offense_code"].map(SEATTLE_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()

    assigned, geocoded_counts, _ = _assign_seattle_block_groups(
        mapped=mapped,
        raw_dir=raw_dir,
        wa_bg_zip=wa_bg_zip,
        bg_crosswalk_parquet=bg_crosswalk_parquet,
    )
    if assigned.empty:
        return _empty_city_reconciliation()
    matched_counts = (
        assigned.groupby(["year", "offense"], dropna=False)["incident_count"]
        .sum()
        .rename("matched_offense_count")
        .reset_index()
    )
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    return _build_city_reconciliation_table(
        city_key="seattle",
        city_name=SEATTLE_CITY_NAME,
        jurisdiction_id=SEATTLE_JURISDICTION_ID,
        state_fips="53",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=SEATTLE_PUBLISHED_RECON,
    )


def build_austin_incident_share_surface(
    *,
    raw_dir: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        raw = _load_austin_raw_snapshot(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Austin incident source could not be loaded cleanly: {exc}. Skipping Austin city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    raw["incident_dt"] = pd.to_datetime(raw["occ_date_time"], errors="coerce")
    report_dt = pd.to_datetime(raw["rep_date_time"], errors="coerce")
    raw["incident_dt"] = raw["incident_dt"].fillna(report_dt)
    raw["year"] = raw["incident_dt"].dt.year
    year_index = pd.Index(range(int(start_year), int(end_year) + 1), dtype="int64")
    year_counts = raw.groupby("year", dropna=False).size().reindex(year_index, fill_value=0)
    total_in_window = int(year_counts.sum())
    undercovered_years = [int(year) for year, count in year_counts.items() if int(count) < AUSTIN_MIN_INCIDENTS_PER_YEAR]
    if total_in_window < AUSTIN_MIN_INCIDENTS_IN_WINDOW or undercovered_years:
        warnings.warn(
            (
                "Austin incident source coverage failed sanity check; "
                f"{int(start_year)}-{int(end_year)} total={total_in_window}, "
                f"undercovered years={undercovered_years}. "
                "Skipping Austin city-incident surface until the source contract is corrected."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()

    raw["ucr_category"] = raw["ucr_category"].astype(str).str.strip().str.upper()
    raw["offense"] = raw["ucr_category"].map(AUSTIN_OFFENSE_MAP)
    raw = raw[raw["offense"].notna()].copy()
    if raw.empty:
        warnings.warn(
            "Austin offense harmonization produced no mapped incidents. Skipping Austin city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    raw["census_block_group"] = raw["census_block_group"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    # Austin publishes a source block group and never a point, so there is nothing for the
    # coordinate quarantine to act on. Declared rather than absent (Stage 4 screen b3).
    raw = _drop_quarantined_coordinates(raw, city_key="austin")
    raw["block_group_geoid"] = raw["census_block_group"].str.zfill(10)
    raw = raw[raw["block_group_geoid"].str.fullmatch(r"\d{10}")].copy()
    raw["block_group_geoid"] = "48" + raw["block_group_geoid"]
    if raw.empty:
        return _empty_city_surface()

    geocoded_counts = (
        raw.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("geocoded_offense_count")
        .reset_index()
    )

    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = raw.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(AUSTIN_JURISDICTION_ID)].copy()
    if matched.empty:
        warnings.warn(
            "Austin incidents did not match any Austin block group in the jurisdiction crosswalk. Skipping Austin city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = AUSTIN_CITY_NAME
    counts["geocode_quality_tier"] = "source_block_group"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_austin_incident_reconciliation(
    *,
    raw_dir: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        raw = _load_austin_raw_snapshot(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    raw["incident_dt"] = pd.to_datetime(raw["occ_date_time"], errors="coerce")
    report_dt = pd.to_datetime(raw["rep_date_time"], errors="coerce")
    raw["incident_dt"] = raw["incident_dt"].fillna(report_dt)
    raw["year"] = raw["incident_dt"].dt.year
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()

    mapped = raw.copy()
    mapped["ucr_category"] = mapped["ucr_category"].astype(str).str.strip().str.upper()
    mapped["offense"] = mapped["ucr_category"].map(AUSTIN_OFFENSE_MAP)
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()
    if mapped.empty:
        return _empty_city_reconciliation()

    mapped["census_block_group"] = mapped["census_block_group"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    mapped["block_group_geoid"] = mapped["census_block_group"].str.zfill(10)
    geocoded = mapped[mapped["block_group_geoid"].str.fullmatch(r"\d{10}")].copy()
    geocoded["block_group_geoid"] = "48" + geocoded["block_group_geoid"]
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()
    if geocoded.empty:
        return _empty_city_reconciliation()

    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = geocoded.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(AUSTIN_JURISDICTION_ID)].copy()
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    return _build_city_reconciliation_table(
        city_key="austin",
        city_name=AUSTIN_CITY_NAME,
        jurisdiction_id=AUSTIN_JURISDICTION_ID,
        state_fips="48",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=AUSTIN_PUBLISHED_RECON,
    )


def build_boston_incident_share_surface(
    *,
    raw_dir: Path,
    ma_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2019,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    effective_start_year = max(int(start_year), BOSTON_LIVE_START_YEAR)
    if int(end_year) < effective_start_year:
        return _empty_city_surface()
    try:
        raw = _load_boston_raw_snapshot(
            raw_dir=raw_dir,
            start_year=effective_start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Boston incident source could not be loaded cleanly: {exc}. Skipping Boston city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    raw["incident_dt"] = pd.to_datetime(raw["occurred_on_date"], errors="coerce")
    raw["year_num"] = pd.to_numeric(raw["year"], errors="coerce")
    raw["year"] = raw["year_num"].fillna(pd.to_numeric(raw["incident_dt"].dt.year, errors="coerce"))
    raw = raw[raw["year"].between(effective_start_year, int(end_year), inclusive="both")].copy()
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw = raw[raw["year"].notna()].copy()
    raw["year"] = raw["year"].astype(int)
    raw["offense_code"] = raw["offense_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    raw["offense"] = pd.NA
    raw.loc[raw["offense_code"].isin(BOSTON_LARCENY_CODES), "offense"] = "larceny"
    raw.loc[raw["offense_code"].isin(BOSTON_MOTOR_VEHICLE_THEFT_CODES), "offense"] = "motor_vehicle_theft"
    raw = raw[raw["offense"].notna()].copy()
    if raw.empty:
        return _empty_city_surface()

    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    geocoded = raw[raw["latitude"].notna() & raw["longitude"].notna()].copy()
    if geocoded.empty:
        return _empty_city_surface()

    geocoded["offense_row_id"] = (
        geocoded["incident_number"].astype(str).str.strip()
        + "::"
        + geocoded["offense"].astype(str).str.strip()
    )
    geocoded = geocoded.drop_duplicates("offense_row_id").copy()
    geocoded = _drop_quarantined_coordinates(geocoded, city_key="boston")
    if geocoded.empty:
        return _empty_city_surface()

    incidents = gpd.GeoDataFrame(
        geocoded[["offense_row_id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(ma_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "offense", "block_group_geoid"]
    ].drop_duplicates("offense_row_id")
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = matched.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(BOSTON_JURISDICTION_ID)].copy()
    if matched.empty:
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = BOSTON_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_boston_incident_reconciliation(
    *,
    raw_dir: Path,
    ma_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2019,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    effective_start_year = max(int(start_year), BOSTON_LIVE_START_YEAR)
    if int(end_year) < effective_start_year:
        return _empty_city_reconciliation()
    try:
        raw = _load_boston_raw_snapshot(
            raw_dir=raw_dir,
            start_year=effective_start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    raw["incident_dt"] = pd.to_datetime(raw["occurred_on_date"], errors="coerce")
    raw["year_num"] = pd.to_numeric(raw["year"], errors="coerce")
    raw["year"] = raw["year_num"].fillna(pd.to_numeric(raw["incident_dt"].dt.year, errors="coerce"))
    raw = raw[raw["year"].between(effective_start_year, int(end_year), inclusive="both")].copy()
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw = raw[raw["year"].notna()].copy()
    raw["year"] = raw["year"].astype(int)
    raw_year_totals = raw.groupby("year", dropna=False).size()

    raw["offense_code"] = raw["offense_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    raw["offense"] = pd.NA
    raw.loc[raw["offense_code"].isin(BOSTON_LARCENY_CODES), "offense"] = "larceny"
    raw.loc[raw["offense_code"].isin(BOSTON_MOTOR_VEHICLE_THEFT_CODES), "offense"] = "motor_vehicle_theft"
    mapped = raw[raw["offense"].notna()].copy()
    if mapped.empty:
        return _empty_city_reconciliation()
    mapped["offense_row_id"] = (
        mapped["incident_number"].astype(str).str.strip()
        + "::"
        + mapped["offense"].astype(str).str.strip()
    )
    mapped = mapped.drop_duplicates("offense_row_id").copy()
    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()

    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()
    if geocoded.empty:
        return _empty_city_reconciliation()
    quarantine_mask = flag_quarantined_coordinates(geocoded, city_key="boston")
    quarantined_counts = quarantine_counts_by_year_offense(geocoded, quarantine_mask)
    located = geocoded.loc[~quarantine_mask].copy()
    if located.empty:
        return _empty_city_reconciliation()

    incidents = gpd.GeoDataFrame(
        located[["offense_row_id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(located["longitude"], located["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(ma_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "offense", "block_group_geoid"]
    ].drop_duplicates("offense_row_id")
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = matched.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(BOSTON_JURISDICTION_ID)].copy()
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    return _build_city_reconciliation_table(
        city_key="boston",
        city_name=BOSTON_CITY_NAME,
        jurisdiction_id=BOSTON_JURISDICTION_ID,
        state_fips="25",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        quarantined_counts=quarantined_counts,
        published_recon=BOSTON_PUBLISHED_RECON,
    )


def _load_baltimore_offense_crosswalk(offense_crosswalk_csv: Path) -> pd.DataFrame:
    crosswalk = pd.read_csv(offense_crosswalk_csv, dtype=str).fillna("").copy()
    required = {
        "source_feed",
        "source_description",
        "canonical_offense",
        "include_flag",
        "selection_rule",
    }
    if crosswalk.empty or not required.issubset(crosswalk.columns):
        return pd.DataFrame(columns=["source_description", "offense", "selection_rule"])
    crosswalk = crosswalk[crosswalk["source_feed"].astype(str).str.strip().eq("legacy_srs")].copy()
    crosswalk["include_bool"] = crosswalk["include_flag"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
    crosswalk["offense"] = crosswalk["canonical_offense"].astype(str).str.strip()
    crosswalk["source_description"] = crosswalk["source_description"].astype(str).str.strip()
    crosswalk["selection_rule"] = crosswalk["selection_rule"].astype(str).str.strip()
    crosswalk = crosswalk[
        crosswalk["include_bool"]
        & crosswalk["offense"].ne("")
        & crosswalk["source_description"].ne("")
        & crosswalk["selection_rule"].ne("")
    ].copy()
    return crosswalk[["source_description", "offense", "selection_rule"]].copy()


def _load_baltimore_published_recon(
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
) -> dict[int, dict[str, object]]:
    published = pd.read_csv(published_reference_csv, dtype=str).fillna("").copy()
    statuses = pd.read_csv(packet_offense_status_csv, dtype=str).fillna("").copy()
    if published.empty:
        return {}

    published = published[published["reference_window"].astype(str).str.strip().eq("2024_ytd_through_2024_12_28")].copy()
    published = published[
        published["offense"].astype(str).str.strip().isin(
            {
                "murder",
                "rape",
                "robbery",
                "aggravated_assault",
                "burglary",
                "larceny",
                "motor_vehicle_theft",
            }
        )
    ].copy()
    if published.empty:
        return {}

    quality_map: dict[str, str] = {}
    notes_map: dict[str, str] = {}
    for row in statuses.itertuples(index=False):
        offense = str(getattr(row, "offense", "")).strip()
        if not offense:
            continue
        comparison_class = str(getattr(row, "comparison_class", "")).strip()
        ready = str(getattr(row, "production_ready", "")).strip().lower() in {"yes", "true", "1", "partial"}
        status = str(getattr(row, "city_share_integration_status", "")).strip().lower()
        quality_map[offense] = comparison_class or ("exact" if ready and status == "offense_selective" else "approximate")
        notes_map[offense] = str(getattr(row, "notes", "")).strip()

    counts: dict[str, float] = {}
    for row in published.itertuples(index=False):
        offense = str(getattr(row, "offense", "")).strip()
        count = pd.to_numeric(getattr(row, "published_count", ""), errors="coerce")
        if not offense or pd.isna(count):
            continue
        counts[offense] = float(count)
    if not counts:
        return {}

    notes = " | ".join([notes_map[offense] for offense in counts if notes_map.get(offense)])
    return {
        2024: {
            "source_name": str(published["published_source_name"].iloc[0]),
            "source_url": str(published["published_source_url"].iloc[0]),
            "counts": counts,
            "comparison_quality": {offense: quality_map.get(offense, "approximate") for offense in counts},
            "notes": notes or "Official Baltimore weekly executive-summary year-to-date table through 2024-12-28.",
        }
    }


def _load_baltimore_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    if int(end_year) > BALTIMORE_PROMOTED_END_YEAR:
        raise CityFeedCoverageError(
            f"Baltimore legacy Part1_Crime_Beta loader called for {int(end_year)}, but the legacy "
            f"feed froze at {BALTIMORE_PROMOTED_END_EXCLUSIVE}; years >= "
            f"{BALTIMORE_NIBRS_CUTOVER_YEAR} are served by _load_baltimore_nibrs_raw_snapshot."
        )
    effective_end_year = min(int(end_year), BALTIMORE_PROMOTED_END_YEAR)
    if effective_end_year < int(start_year):
        return pd.DataFrame(columns=list(BALTIMORE_MINIMAL_COLUMNS))
    start_ts = f"{int(start_year)}-01-01 00:00:00"
    if effective_end_year == BALTIMORE_PROMOTED_END_YEAR:
        end_ts = BALTIMORE_PROMOTED_END_EXCLUSIVE
    else:
        end_ts = f"{effective_end_year + 1}-01-01 00:00:00"
    cache_path = _prepared_city_cache_path(raw_dir, f"baltimore_legacy_srs_{int(start_year)}_{effective_end_year}_through_2024_12_28.parquet")
    params = {
        "where": f"CrimeDateTime >= DATE '{start_ts}' AND CrimeDateTime < DATE '{end_ts}'",
        "outFields": ",".join(BALTIMORE_MINIMAL_COLUMNS),
        "orderByFields": "RowID ASC",
        "returnGeometry": "false",
        "f": "json",
    }
    raw = _download_arcgis_query_frame(
        base_url=BALTIMORE_LEGACY_QUERY_URL,
        params=params,
        out_path=cache_path,
        force_download=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(columns=list(BALTIMORE_MINIMAL_COLUMNS))
    missing = set(BALTIMORE_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Baltimore source is missing required columns: {missing_cols}")
    raw = raw.loc[:, list(BALTIMORE_MINIMAL_COLUMNS)].copy()
    raw["RowID"] = raw["RowID"].astype(str).str.strip()
    raw = raw[raw["RowID"].ne("")].copy()
    raw = raw.drop_duplicates("RowID", keep="last")
    return raw


def _load_baltimore_nibrs_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    if int(end_year) > BALTIMORE_NIBRS_MAX_VERIFIED_YEAR:
        raise CityFeedCoverageError(
            f"Baltimore NIBRS live feed requested through {int(end_year)}, but coverage is only "
            f"verified through {BALTIMORE_NIBRS_MAX_VERIFIED_YEAR}. Confirm the NIBRS_GroupA_Crime_Data "
            "FeatureServer is alive for the requested year and bump BALTIMORE_NIBRS_MAX_VERIFIED_YEAR."
        )
    effective_start_year = max(int(start_year), BALTIMORE_NIBRS_CUTOVER_YEAR)
    effective_end_year = int(end_year)
    if effective_end_year < effective_start_year:
        return pd.DataFrame(columns=list(BALTIMORE_NIBRS_MINIMAL_COLUMNS))
    cache_path = _prepared_city_cache_path(
        raw_dir,
        f"baltimore_nibrs_group_a_{effective_start_year}_{effective_end_year}.parquet",
    )
    params = {
        "where": (
            f"CrimeDateTime >= DATE '{effective_start_year}-01-01 00:00:00' "
            f"AND CrimeDateTime < DATE '{effective_end_year + 1}-01-01 00:00:00'"
        ),
        "outFields": ",".join(BALTIMORE_NIBRS_MINIMAL_COLUMNS),
        "orderByFields": "RowID ASC",
        "returnGeometry": "false",
        "f": "json",
    }
    raw = _download_arcgis_query_frame(
        base_url=BALTIMORE_NIBRS_QUERY_URL,
        params=params,
        out_path=cache_path,
        force_download=force_refresh,
    )
    if raw.empty:
        return pd.DataFrame(columns=list(BALTIMORE_NIBRS_MINIMAL_COLUMNS))
    missing = set(BALTIMORE_NIBRS_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Baltimore NIBRS source is missing required columns: {missing_cols}")
    raw = raw.loc[:, list(BALTIMORE_NIBRS_MINIMAL_COLUMNS)].copy()
    raw["RowID"] = raw["RowID"].astype(str).str.strip()
    raw = raw[raw["RowID"].ne("")].copy()
    raw = raw.drop_duplicates("RowID", keep="last")
    return raw


def _prepare_baltimore_nibrs_incidents(
    *,
    raw_dir: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """NIBRS-lane Baltimore incidents for years >= BALTIMORE_NIBRS_CUTOVER_YEAR.

    CrimeCode carries standard NIBRS offense codes, mapped to the 7 offenses via
    BALTIMORE_NIBRS_OFFENSE_MAP like other NIBRS-coded cities; row identity is the
    feed's RowID OID per offense row (row-count basis).
    """
    effective_start_year = max(int(start_year), BALTIMORE_NIBRS_CUTOVER_YEAR)
    effective_end_year = int(end_year)
    empty = pd.DataFrame(columns=["offense_row_id", "year", "offense", "latitude", "longitude"])
    if effective_end_year < effective_start_year:
        return empty.copy(), empty.copy(), pd.Series(dtype="int64")

    raw = _load_baltimore_nibrs_raw_snapshot(
        raw_dir=raw_dir,
        start_year=effective_start_year,
        end_year=effective_end_year,
        force_refresh=force_refresh,
    )
    if raw.empty:
        raise RuntimeError("Baltimore NIBRS source returned no rows for the requested window.")

    raw["incident_dt"] = pd.to_datetime(pd.to_numeric(raw["CrimeDateTime"], errors="coerce"), unit="ms", errors="coerce")
    raw["year"] = raw["incident_dt"].dt.year
    year_index = pd.Index(range(effective_start_year, effective_end_year + 1), dtype="int64")
    year_counts = raw.groupby("year", dropna=False).size().reindex(year_index, fill_value=0)
    if int(year_counts.sum()) < BALTIMORE_MIN_INCIDENTS_IN_WINDOW:
        raise RuntimeError(
            "Baltimore NIBRS source coverage failed sanity check; "
            f"{effective_start_year}-{effective_end_year} total={int(year_counts.sum())}"
        )
    raw = raw[raw["year"].between(effective_start_year, effective_end_year, inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()

    raw["offense"] = raw["CrimeCode"].astype(str).str.strip().str.upper().map(BALTIMORE_NIBRS_OFFENSE_MAP)
    mapped = raw[raw["offense"].notna()].copy()
    if mapped.empty:
        raise RuntimeError("Baltimore NIBRS offense harmonization produced no mapped incidents.")
    mapped["offense_row_id"] = mapped["RowID"].astype(str).str.strip()
    mapped = mapped.drop_duplicates(["year", "offense", "offense_row_id"], keep="first")
    mapped = mapped[["offense_row_id", "year", "offense", "Latitude", "Longitude"]].rename(
        columns={"Latitude": "latitude", "Longitude": "longitude"}
    )
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_share = float(len(geocoded)) / float(len(mapped)) if len(mapped) else 0.0
    if geocoded_share < BALTIMORE_MIN_GEOCODED_SHARE:
        raise RuntimeError(
            "Baltimore NIBRS coordinate coverage failed sanity check; "
            f"only {geocoded_share:.1%} of mapped incidents had usable coordinates."
        )
    return mapped, geocoded, raw_year_totals


def _prepare_baltimore_incidents(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Baltimore incidents across the 2025-01-01 source cutover.

    Years <= BALTIMORE_PROMOTED_END_YEAR come from the promoted legacy Part1_Crime_Beta
    packet lane (byte-identical to the pre-parameterization behavior); years >=
    BALTIMORE_NIBRS_CUTOVER_YEAR come from the NIBRS_GroupA_Crime_Data lane. A request
    beyond the NIBRS lane's verified coverage raises CityFeedCoverageError from the
    NIBRS loader rather than silently truncating.
    """
    if int(end_year) >= BALTIMORE_NIBRS_CUTOVER_YEAR:
        nibrs_mapped, nibrs_geocoded, nibrs_totals = _prepare_baltimore_nibrs_incidents(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_refresh,
        )
        if int(start_year) >= BALTIMORE_NIBRS_CUTOVER_YEAR:
            return nibrs_mapped, nibrs_geocoded, nibrs_totals
        legacy_mapped, legacy_geocoded, legacy_totals = _prepare_baltimore_legacy_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=BALTIMORE_PROMOTED_END_YEAR,
            force_refresh=force_refresh,
        )
        mapped = pd.concat([legacy_mapped, nibrs_mapped], ignore_index=True)
        geocoded = pd.concat([legacy_geocoded, nibrs_geocoded], ignore_index=True)
        totals = pd.concat([legacy_totals, nibrs_totals]).sort_index()
        return mapped, geocoded, totals
    return _prepare_baltimore_legacy_incidents(
        raw_dir=raw_dir,
        offense_crosswalk_csv=offense_crosswalk_csv,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_refresh,
    )


def _prepare_baltimore_legacy_incidents(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    effective_start_year = max(int(start_year), BALTIMORE_PROMOTED_START_YEAR)
    effective_end_year = min(int(end_year), BALTIMORE_PROMOTED_END_YEAR)
    if effective_end_year < effective_start_year:
        empty = pd.DataFrame(columns=["offense_row_id", "year", "offense", "latitude", "longitude"])
        return empty.copy(), empty.copy(), pd.Series(dtype="int64")

    raw = _load_baltimore_raw_snapshot(
        raw_dir=raw_dir,
        start_year=effective_start_year,
        end_year=effective_end_year,
        force_refresh=force_refresh,
    )
    if raw.empty:
        raise RuntimeError("Baltimore legacy SRS source returned no rows for the promoted 2024 packet window.")

    raw["incident_dt"] = pd.to_datetime(pd.to_numeric(raw["CrimeDateTime"], errors="coerce"), unit="ms", errors="coerce")
    raw["year"] = raw["incident_dt"].dt.year
    year_index = pd.Index(range(effective_start_year, effective_end_year + 1), dtype="int64")
    year_counts = raw.groupby("year", dropna=False).size().reindex(year_index, fill_value=0)
    if int(year_counts.sum()) < BALTIMORE_MIN_INCIDENTS_IN_WINDOW:
        raise RuntimeError(
            "Baltimore legacy SRS source coverage failed sanity check; "
            f"{effective_start_year}-{effective_end_year} total={int(year_counts.sum())}"
        )
    raw = raw[raw["year"].between(effective_start_year, effective_end_year, inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()

    crosswalk = _load_baltimore_offense_crosswalk(offense_crosswalk_csv)
    if crosswalk.empty:
        raise RuntimeError("Baltimore offense crosswalk is empty or missing required packet columns.")

    mapped_frames: list[pd.DataFrame] = []
    for offense, group in crosswalk.groupby("offense", sort=True):
        descriptions = {str(v).strip() for v in group["source_description"].tolist() if str(v).strip()}
        selection_rules = {str(v).strip() for v in group["selection_rule"].tolist() if str(v).strip()}
        if not descriptions or len(selection_rules) != 1:
            raise RuntimeError(f"Baltimore offense packet rules are inconsistent for offense={offense}.")
        selection_rule = next(iter(selection_rules))
        subset = raw[raw["Description"].astype(str).str.strip().isin(descriptions)].copy()
        if subset.empty:
            continue
        subset["offense"] = offense
        if selection_rule == "row_count":
            subset["offense_row_id"] = subset["RowID"].astype(str).str.strip()
        elif selection_rule == "family_unique_ccnumber":
            subset["offense_row_id"] = subset["CCNumber"].astype(str).str.strip()
            subset = subset[subset["offense_row_id"].ne("") & subset["offense_row_id"].ne("nan")].copy()
        else:
            raise RuntimeError(f"Unsupported Baltimore packet selection rule: {selection_rule}")
        subset = subset.drop_duplicates(["year", "offense", "offense_row_id"], keep="first")
        mapped_frames.append(
            subset[["offense_row_id", "year", "offense", "Latitude", "Longitude"]].rename(
                columns={"Latitude": "latitude", "Longitude": "longitude"}
            )
        )

    if not mapped_frames:
        raise RuntimeError("Baltimore offense harmonization produced no packet-mapped incidents.")
    mapped = pd.concat(mapped_frames, ignore_index=True)
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_share = float(len(geocoded)) / float(len(mapped)) if len(mapped) else 0.0
    if geocoded_share < BALTIMORE_MIN_GEOCODED_SHARE:
        raise RuntimeError(
            "Baltimore coordinate coverage failed sanity check; "
            f"only {geocoded_share:.1%} of packet-mapped incidents had usable coordinates."
        )
    return mapped, geocoded, raw_year_totals


def build_baltimore_incident_share_surface(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
    md_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        _, geocoded, _ = _prepare_baltimore_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Baltimore incident source could not be loaded cleanly: {exc}. Skipping Baltimore city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    geocoded = _drop_quarantined_coordinates(geocoded, city_key="baltimore")
    incidents = gpd.GeoDataFrame(
        geocoded[["offense_row_id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(BALTIMORE_JURISDICTION_ID)].copy()
    if crosswalk.empty:
        warnings.warn(
            "Baltimore jurisdiction was not present in the block-group crosswalk. Skipping Baltimore city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    bg = gpd.read_file(md_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    if bg.empty:
        warnings.warn(
            "Baltimore block-group geometry could not be matched to the jurisdiction crosswalk. Skipping Baltimore city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)

    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"]
    ].drop_duplicates("offense_row_id")
    if matched.empty:
        warnings.warn(
            "Baltimore incidents did not spatially match any Baltimore block group. Skipping Baltimore city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = BALTIMORE_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_baltimore_incident_reconciliation(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
    md_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, geocoded, raw_year_totals = _prepare_baltimore_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()

    incidents = gpd.GeoDataFrame(
        geocoded[["offense_row_id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(BALTIMORE_JURISDICTION_ID)].copy()
    if crosswalk.empty:
        return _empty_city_reconciliation()

    bg = gpd.read_file(md_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    if bg.empty:
        return _empty_city_reconciliation()
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)

    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"]
    ].drop_duplicates("offense_row_id")
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    published_recon = _load_baltimore_published_recon(published_reference_csv, packet_offense_status_csv)
    return _build_city_reconciliation_table(
        city_key="baltimore",
        city_name=BALTIMORE_CITY_NAME,
        jurisdiction_id=BALTIMORE_JURISDICTION_ID,
        state_fips="24",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=published_recon,
    )


def _load_philadelphia_snapshot_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(PHILADELPHIA_RAW_COLUMN_MAP),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
        low_memory=False,
    )
    raw = raw.rename(columns=PHILADELPHIA_RAW_COLUMN_MAP)
    missing = set(PHILADELPHIA_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Philadelphia source is missing required columns: {missing_cols}")
    return raw.loc[:, list(PHILADELPHIA_MINIMAL_COLUMNS)].copy()


def _load_philadelphia_offense_crosswalk(offense_crosswalk_csv: Path) -> pd.DataFrame:
    crosswalk = pd.read_csv(offense_crosswalk_csv, dtype=str).fillna("").copy()
    required = {"source_field", "source_value", "v2_offense", "include_in_v2"}
    if crosswalk.empty or not required.issubset(crosswalk.columns):
        return pd.DataFrame(columns=["source_fields", "source_values", "offense"])

    crosswalk["include_flag"] = crosswalk["include_in_v2"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
    crosswalk["offense"] = crosswalk["v2_offense"].astype(str).str.strip()
    crosswalk = crosswalk[crosswalk["include_flag"] & crosswalk["offense"].ne("")].copy()
    if crosswalk.empty:
        return pd.DataFrame(columns=["source_fields", "source_values", "offense"])

    crosswalk["source_fields"] = crosswalk["source_field"].astype(str).str.split("|")
    crosswalk["source_values"] = crosswalk["source_value"].astype(str).str.split("|")
    crosswalk["source_fields"] = crosswalk["source_fields"].apply(lambda values: [str(v).strip() for v in values if str(v).strip()])
    crosswalk["source_values"] = crosswalk["source_values"].apply(lambda values: [str(v).strip() for v in values if str(v).strip()])
    return crosswalk[["source_fields", "source_values", "offense"]].copy()


def _load_philadelphia_published_recon(
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
) -> dict[int, dict[str, object]]:
    published = pd.read_csv(published_reference_csv, dtype=str).fillna("").copy()
    statuses = pd.read_csv(packet_offense_status_csv, dtype=str).fillna("").copy()
    if published.empty:
        return {}

    reference_to_offense = {
        "Homicide": "murder",
        "Rape": "rape",
        "Robbery/Gun": "robbery",
        "Robbery/Other": "robbery",
        "Aggravated Assault/Gun": "aggravated_assault",
        "Aggravated Assault/Other": "aggravated_assault",
        "Burglary/Residential": "burglary",
        "Burglary/Commercial": "burglary",
        "Theft from Person": "larceny",
        "Theft from Auto": "larceny",
        "Theft": "larceny",
        "Theft/Retail": "larceny",
        "Auto Theft": "motor_vehicle_theft",
    }
    quality_map: dict[str, str] = {}
    notes_map: dict[str, str] = {}
    for row in statuses.itertuples(index=False):
        offense = str(getattr(row, "offense", "")).strip()
        if not offense:
            continue
        comparison_class = str(getattr(row, "comparison_class", "")).strip()
        ready = str(getattr(row, "production_ready", "")).strip().lower() in {"yes", "true", "1", "partial"}
        status = str(getattr(row, "city_share_integration_status", "")).strip().lower()
        if comparison_class:
            quality_map[offense] = comparison_class
        else:
            quality_map[offense] = "exact" if ready and status == "offense_selective" else "blocked"
        notes_map[offense] = str(getattr(row, "notes", "")).strip()

    recon: dict[int, dict[str, object]] = {}
    for year, group in published.groupby("reference_year", dropna=False):
        counts: dict[str, float] = {}
        for row in group.itertuples(index=False):
            offense = reference_to_offense.get(str(getattr(row, "reference_label", "")).strip())
            if not offense:
                continue
            count = pd.to_numeric(getattr(row, "official_count", ""), errors="coerce")
            if pd.isna(count):
                continue
            counts[offense] = counts.get(offense, 0.0) + float(count)
        if not counts:
            continue
        notes = " | ".join([notes_map.get(offense, "") for offense in counts if notes_map.get(offense, "")])
        recon[int(year)] = {
            "source_name": str(group["source_name"].iloc[0]),
            "source_url": str(group["source_url"].iloc[0]),
            "counts": counts,
            "comparison_quality": {offense: quality_map.get(offense, "approximate") for offense in counts},
            "notes": notes or "Official Philadelphia Police citywide year-end table aggregated to V2 offense families.",
        }
    return recon


def _load_washington_dc_snapshot_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(WASHINGTON_DC_RAW_COLUMN_MAP),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
        low_memory=False,
    )
    raw = raw.rename(columns=WASHINGTON_DC_RAW_COLUMN_MAP)
    missing = set(WASHINGTON_DC_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Washington DC source is missing required columns: {missing_cols}")
    return raw.loc[:, list(WASHINGTON_DC_MINIMAL_COLUMNS)].copy()


def _load_washington_dc_offense_crosswalk(offense_crosswalk_csv: Path) -> pd.DataFrame:
    crosswalk = pd.read_csv(offense_crosswalk_csv, dtype=str).fillna("").copy()
    required = {"source_field", "source_value", "v2_offense", "include_in_v2"}
    if crosswalk.empty or not required.issubset(crosswalk.columns):
        return pd.DataFrame(columns=["source_field", "source_values", "offense"])

    field_aliases = {
        "offense": "offense_label",
        "offense_label": "offense_label",
    }
    crosswalk["include_flag"] = crosswalk["include_in_v2"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
    crosswalk["offense"] = crosswalk["v2_offense"].astype(str).str.strip()
    crosswalk = crosswalk[crosswalk["include_flag"] & crosswalk["offense"].ne("")].copy()
    if crosswalk.empty:
        return pd.DataFrame(columns=["source_field", "source_values", "offense"])

    crosswalk["source_field"] = (
        crosswalk["source_field"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda value: field_aliases.get(value, value))
    )
    crosswalk["source_values"] = crosswalk["source_value"].astype(str).str.split("|")
    crosswalk["source_values"] = crosswalk["source_values"].apply(
        lambda values: [str(v).strip().upper() for v in values if str(v).strip()]
    )
    return crosswalk[["source_field", "source_values", "offense"]].copy()


def _load_washington_dc_published_recon(
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
) -> dict[int, dict[str, object]]:
    published = pd.read_csv(published_reference_csv, dtype=str).fillna("").copy()
    statuses = pd.read_csv(packet_offense_status_csv, dtype=str).fillna("").copy()
    if published.empty:
        return {}

    reference_to_offense = {
        "Homicide": "murder",
        "Sex Abuse": "rape",
        "Assault w/ a Dangerous Weapon": "aggravated_assault",
        "Robbery": "robbery",
        "Burglary": "burglary",
        "Motor Vehicle Theft": "motor_vehicle_theft",
        "Theft from Auto": "larceny",
        "Theft (Other)": "larceny",
    }
    quality_map: dict[str, str] = {}
    notes_map: dict[str, str] = {}
    for row in statuses.itertuples(index=False):
        offense = str(getattr(row, "offense", "")).strip()
        if not offense:
            continue
        comparison_class = str(getattr(row, "comparison_class", "")).strip()
        ready = str(getattr(row, "production_ready", "")).strip().lower() in {"yes", "true", "1", "partial"}
        status = str(getattr(row, "city_share_integration_status", "")).strip().lower()
        if comparison_class:
            quality_map[offense] = comparison_class
        else:
            quality_map[offense] = "exact" if ready and status == "offense_selective" else "blocked"
        notes_map[offense] = str(getattr(row, "notes", "")).strip()

    recon: dict[int, dict[str, object]] = {}
    for year, group in published.groupby("reference_year", dropna=False):
        counts: dict[str, float] = {}
        for row in group.itertuples(index=False):
            offense = reference_to_offense.get(str(getattr(row, "reference_label", "")).strip())
            if not offense:
                continue
            count = pd.to_numeric(getattr(row, "official_count", ""), errors="coerce")
            if pd.isna(count):
                continue
            counts[offense] = counts.get(offense, 0.0) + float(count)
        if not counts:
            continue
        notes = " | ".join([notes_map.get(offense, "") for offense in counts if notes_map.get(offense, "")])
        recon[int(year)] = {
            "source_name": str(group["source_name"].iloc[0]),
            "source_url": str(group["source_url"].iloc[0]),
            "counts": counts,
            "comparison_quality": {offense: quality_map.get(offense, "approximate") for offense in counts},
            "notes": notes or "Official MPD year-end table aggregated to V2 offense families.",
        }
    return recon


def _load_washington_dc_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    if int(end_year) > WASHINGTON_DC_MAX_VERIFIED_YEAR:
        raise CityFeedCoverageError(
            f"Washington DC live feed requested through {int(end_year)}, but the "
            "crime-incidents-in-<year> sibling dataset is only verified through "
            f"{WASHINGTON_DC_MAX_VERIFIED_YEAR} (STATE.md 2025 refresh audit, 2026-07-10). Confirm "
            "the next vintage exists at opendata.dc.gov before raising WASHINGTON_DC_MAX_VERIFIED_YEAR."
        )
    effective_start_year = max(int(start_year), WASHINGTON_DC_LIVE_START_YEAR)
    target_year = min(int(end_year), WASHINGTON_DC_MAX_VERIFIED_YEAR)
    if target_year < effective_start_year:
        return pd.DataFrame(columns=list(WASHINGTON_DC_MINIMAL_COLUMNS))

    cache_path = _prepared_city_cache_path(
        raw_dir,
        f"washington_dc_minimal_snapshot_{target_year}_{target_year}.parquet",
    )
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "objectid" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()

    if raw.empty and "objectid" not in raw.columns:
        csv_path = _download_file(
            WASHINGTON_DC_DOWNLOAD_URL_TEMPLATE.format(year=target_year),
            raw_dir / f"washington_dc_{target_year}.csv",
            force_download=force_refresh,
        )
        raw = _load_washington_dc_snapshot_csv(csv_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)

    missing = set(WASHINGTON_DC_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Washington DC incident cache is missing required columns: {missing_cols}")
    raw["objectid"] = raw["objectid"].astype(str).str.strip()
    raw = raw[raw["objectid"].ne("")].copy()
    raw = raw.drop_duplicates("objectid", keep="last")
    return raw


def _assign_washington_dc_block_groups(mapped: pd.DataFrame, *, dc_bg_zip: Path) -> pd.DataFrame:
    out = mapped.copy()
    parts = out["block_group"].astype(str).str.extract(r"(\d{6})\s*(\d)", expand=True)
    source_digits = parts[0].fillna("") + parts[1].fillna("")
    source_digits = source_digits.mask(source_digits.eq(""), pd.NA)
    out["block_group_geoid"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[source_digits.notna(), "block_group_geoid"] = ("11001" + source_digits[source_digits.notna()]).astype("string")

    fallback_candidates = out[out["block_group_geoid"].isna()].copy()
    fallback_candidates["latitude"] = pd.to_numeric(fallback_candidates["latitude"], errors="coerce")
    fallback_candidates["longitude"] = pd.to_numeric(fallback_candidates["longitude"], errors="coerce")
    fallback_candidates = fallback_candidates[
        fallback_candidates["latitude"].notna() & fallback_candidates["longitude"].notna()
    ].copy()
    if fallback_candidates.empty:
        return out

    incidents = gpd.GeoDataFrame(
        fallback_candidates[["objectid"]].copy(),
        geometry=gpd.points_from_xy(fallback_candidates["longitude"], fallback_candidates["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(dc_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    joined = gpd.sjoin(incidents, bg, how="left", predicate="within")[
        ["objectid", "block_group_geoid"]
    ].drop_duplicates("objectid")
    out = out.merge(
        joined.rename(columns={"block_group_geoid": "fallback_block_group_geoid"}),
        on="objectid",
        how="left",
    )
    out["block_group_geoid"] = out["block_group_geoid"].fillna(out["fallback_block_group_geoid"])
    return out.drop(columns=["fallback_block_group_geoid"], errors="ignore")


def _prepare_washington_dc_incidents(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    dc_bg_zip: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = _load_washington_dc_raw_snapshot(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_refresh,
    )
    if raw.empty:
        raise RuntimeError("Washington DC source returned no rows for the requested window.")

    effective_start_year = max(int(start_year), WASHINGTON_DC_LIVE_START_YEAR)
    target_year = min(int(end_year), WASHINGTON_DC_MAX_VERIFIED_YEAR)
    effective_end_year = target_year
    raw["year"] = target_year
    raw = raw[raw["year"].between(effective_start_year, effective_end_year, inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()
    if int(raw_year_totals.sum()) < WASHINGTON_DC_MIN_INCIDENTS_IN_WINDOW:
        raise RuntimeError(
            "Washington DC incident source coverage failed sanity check; "
            f"{effective_start_year}-{effective_end_year} total={int(raw_year_totals.sum())}"
        )

    raw["offense_label"] = raw["offense_label"].astype(str).str.strip().str.upper()
    crosswalk = _load_washington_dc_offense_crosswalk(offense_crosswalk_csv)
    if crosswalk.empty:
        raise RuntimeError("Washington DC offense crosswalk is empty or missing required packet columns.")

    raw["offense"] = pd.NA
    for row in crosswalk.itertuples(index=False):
        source_field = str(getattr(row, "source_field", "")).strip()
        if not source_field or source_field not in raw.columns:
            continue
        values = set(getattr(row, "source_values", []) or [])
        if not values:
            continue
        mask = raw[source_field].astype(str).str.upper().isin(values)
        raw.loc[mask, "offense"] = str(getattr(row, "offense", "")).strip()

    mapped = raw[raw["offense"].notna()].copy()
    if mapped.empty:
        raise RuntimeError("Washington DC offense harmonization produced no mapped incidents.")

    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    mapped = _assign_washington_dc_block_groups(mapped, dc_bg_zip=dc_bg_zip)
    geocoded = mapped[mapped["block_group_geoid"].notna()].copy()
    geocoded_share = float(len(geocoded)) / float(len(mapped)) if len(mapped) else 0.0
    if geocoded_share < WASHINGTON_DC_MIN_BLOCK_GROUP_ASSIGNMENT_SHARE:
        raise RuntimeError(
            "Washington DC block-group assignment failed sanity check; "
            f"only {geocoded_share:.1%} of mapped incidents retained block groups."
        )
    geocoded["block_group_geoid"] = geocoded["block_group_geoid"].astype("string").str.zfill(12)
    geocoded["offense_row_id"] = geocoded["objectid"].astype(str).str.strip() + "::" + geocoded["offense"].astype(str).str.strip()
    geocoded = geocoded.drop_duplicates("offense_row_id").copy()
    return mapped, geocoded, raw_year_totals


def _load_philadelphia_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = _prepared_city_cache_path(raw_dir, f"philadelphia_minimal_snapshot_{int(start_year)}_{int(end_year)}.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "objectid" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()
    if raw.empty and "objectid" not in raw.columns:
        frames: list[pd.DataFrame] = []
        for year in range(int(start_year), int(end_year) + 1):
            query = (
                "SELECT objectid, dc_key, dispatch_date_time, ucr_general, text_general_code, "
                "location_block, ST_Y(the_geom) AS latitude, ST_X(the_geom) AS longitude "
                "FROM incidents_part1_part2 "
                f"WHERE dispatch_date_time >= '{year}-01-01' AND dispatch_date_time < '{year + 1}-01-01'"
            )
            csv_path = _download_csv_query(
                base_url=PHILADELPHIA_SQL_API_URL,
                params={
                    "filename": "incidents_part1_part2",
                    "format": "csv",
                    "q": query,
                },
                out_path=raw_dir / f"philadelphia_{year}.csv",
                force_download=force_refresh,
            )
            frames.append(_load_philadelphia_snapshot_csv(csv_path))
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(PHILADELPHIA_MINIMAL_COLUMNS))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)
    missing = set(PHILADELPHIA_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Philadelphia incident cache is missing required columns: {missing_cols}")
    raw["objectid"] = raw["objectid"].astype(str).str.strip()
    raw = raw[raw["objectid"].ne("")].copy()
    raw = raw.drop_duplicates("objectid", keep="last")
    return raw


def _prepare_philadelphia_incidents(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    raw = _load_philadelphia_raw_snapshot(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_refresh,
    )
    raw["incident_dt"] = pd.to_datetime(raw["dispatch_date_time"], errors="coerce")
    raw["year"] = raw["incident_dt"].dt.year
    year_index = pd.Index(range(int(start_year), int(end_year) + 1), dtype="int64")
    year_counts = raw.groupby("year", dropna=False).size().reindex(year_index, fill_value=0)
    undercovered_years = [int(year) for year, count in year_counts.items() if int(count) < PHILADELPHIA_MIN_INCIDENTS_PER_YEAR]
    if undercovered_years:
        raise RuntimeError(
            "Philadelphia incident source coverage failed sanity check; "
            f"{int(start_year)}-{int(end_year)} undercovered years={undercovered_years}"
        )
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()

    raw["ucr_general"] = raw["ucr_general"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    raw["text_general_code"] = raw["text_general_code"].astype(str).str.strip()
    raw["location_block"] = raw["location_block"].astype(str).str.strip()
    raw["offense"] = pd.NA

    crosswalk = _load_philadelphia_offense_crosswalk(offense_crosswalk_csv)
    for row in crosswalk.itertuples(index=False):
        fields = list(getattr(row, "source_fields", []) or [])
        values = list(getattr(row, "source_values", []) or [])
        if not fields or not values or len(fields) != len(values):
            continue
        mask = pd.Series(True, index=raw.index)
        for field, value in zip(fields, values, strict=True):
            mask = mask & raw[str(field)].astype(str).eq(str(value))
        raw.loc[mask, "offense"] = str(row.offense)

    mapped = raw[raw["offense"].notna()].copy()
    if mapped.empty:
        raise RuntimeError("Philadelphia offense harmonization produced no mapped incidents.")

    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_share = float(len(geocoded)) / float(len(mapped)) if len(mapped) else 0.0
    if geocoded_share < PHILADELPHIA_MIN_GEOCODED_SHARE:
        raise RuntimeError(
            "Philadelphia coordinate coverage failed sanity check; "
            f"only {geocoded_share:.1%} of mapped incidents retained coordinates."
        )
    geocoded["offense_row_id"] = (
        geocoded["objectid"].astype(str).str.strip()
        + "::"
        + geocoded["offense"].astype(str).str.strip()
    )
    geocoded = geocoded.drop_duplicates("offense_row_id").copy()
    return geocoded, raw_year_totals


def build_philadelphia_incident_share_surface(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
    pa_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        geocoded, _ = _prepare_philadelphia_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Philadelphia incident source could not be loaded cleanly: {exc}. Skipping Philadelphia city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    geocoded = _drop_quarantined_coordinates(geocoded, city_key="philadelphia")
    if geocoded.empty:
        return _empty_city_surface()

    incidents = gpd.GeoDataFrame(
        geocoded[["offense_row_id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(PHILADELPHIA_JURISDICTION_ID)].copy()
    if crosswalk.empty:
        warnings.warn(
            "Philadelphia jurisdiction was not present in the block-group crosswalk. Skipping Philadelphia city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    bg = gpd.read_file(pa_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    if bg.empty:
        warnings.warn(
            "Philadelphia block-group geometry could not be matched to the jurisdiction crosswalk. Skipping Philadelphia city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)

    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"]
    ].drop_duplicates("offense_row_id")
    if matched.empty:
        warnings.warn(
            "Philadelphia incidents did not spatially match any Philadelphia block group. Skipping Philadelphia city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = PHILADELPHIA_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_philadelphia_incident_reconciliation(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
    pa_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        geocoded, raw_year_totals = _prepare_philadelphia_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    mapped_counts = (
        geocoded.groupby(["year", "offense"], dropna=False)
        .size()
        .rename("mapped_offense_count")
        .reset_index()
    )
    geocoded_counts = mapped_counts.rename(columns={"mapped_offense_count": "geocoded_offense_count"})
    quarantine_mask = flag_quarantined_coordinates(geocoded, city_key="philadelphia")
    quarantined_counts = quarantine_counts_by_year_offense(geocoded, quarantine_mask)
    located = geocoded.loc[~quarantine_mask].copy()
    if located.empty:
        return _empty_city_reconciliation()

    incidents = gpd.GeoDataFrame(
        located[["offense_row_id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(located["longitude"], located["latitude"]),
        crs="EPSG:4326",
    )
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(PHILADELPHIA_JURISDICTION_ID)].copy()
    if crosswalk.empty:
        return _empty_city_reconciliation()
    bg = gpd.read_file(pa_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    if bg.empty:
        return _empty_city_reconciliation()
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"]
    ].drop_duplicates("offense_row_id")
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    published_recon = _load_philadelphia_published_recon(published_reference_csv, packet_offense_status_csv)
    return _build_city_reconciliation_table(
        city_key="philadelphia",
        city_name=PHILADELPHIA_CITY_NAME,
        jurisdiction_id=PHILADELPHIA_JURISDICTION_ID,
        state_fips="42",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        quarantined_counts=quarantined_counts,
        published_recon=published_recon,
    )


def build_washington_dc_incident_share_surface(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
    dc_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        _, geocoded, _ = _prepare_washington_dc_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            dc_bg_zip=dc_bg_zip,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Washington DC incident source could not be loaded cleanly: {exc}. Skipping Washington DC city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    geocoded = _drop_quarantined_coordinates(geocoded, city_key="washington_dc")
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = geocoded.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(WASHINGTON_DC_JURISDICTION_ID)].copy()
    if matched.empty:
        warnings.warn(
            "Washington DC incidents did not match any Washington DC block group in the jurisdiction crosswalk. Skipping Washington DC city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = WASHINGTON_DC_CITY_NAME
    counts["geocode_quality_tier"] = "source_block_group"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_washington_dc_incident_reconciliation(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
    dc_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, geocoded, raw_year_totals = _prepare_washington_dc_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            dc_bg_zip=dc_bg_zip,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()

    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = geocoded.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(WASHINGTON_DC_JURISDICTION_ID)].copy()
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    published_recon = _load_washington_dc_published_recon(published_reference_csv, packet_offense_status_csv)
    return _build_city_reconciliation_table(
        city_key="washington_dc",
        city_name=WASHINGTON_DC_CITY_NAME,
        jurisdiction_id=WASHINGTON_DC_JURISDICTION_ID,
        state_fips="11",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=published_recon,
    )


def _load_city_bg_geometry(
    *,
    bg_zip: Path,
    bg_crosswalk_parquet: Path,
    jurisdiction_id: str,
) -> gpd.GeoDataFrame:
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(jurisdiction_id)].copy()
    if crosswalk.empty:
        return gpd.GeoDataFrame(columns=["block_group_geoid", "jurisdiction_id", "state_fips", "geometry"], geometry="geometry", crs="EPSG:4326")

    bg = gpd.read_file(bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    return bg.merge(crosswalk, on="block_group_geoid", how="inner")


def _match_point_incidents_to_city_bg(
    point_rows: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    point_crs: str,
    bg: gpd.GeoDataFrame,
    keep_cols: list[str],
    unique_id_col: str,
) -> pd.DataFrame:
    if point_rows.empty or bg.empty:
        return pd.DataFrame(columns=keep_cols + ["block_group_geoid", "jurisdiction_id", "state_fips"])

    incidents = gpd.GeoDataFrame(
        point_rows[keep_cols].copy(),
        geometry=gpd.points_from_xy(point_rows[x_col], point_rows[y_col]),
        crs=point_crs,
    )
    bg_local = bg if bg.crs == incidents.crs else bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg_local, how="inner", predicate="within")[
        keep_cols + ["block_group_geoid", "jurisdiction_id", "state_fips"]
    ]
    return matched.drop_duplicates(unique_id_col)


def _load_denver_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    if int(end_year) > DENVER_LIVE_END_YEAR:
        raise CityFeedCoverageError(
            f"Denver live feed requested through {int(end_year)}, but coverage is only verified "
            f"through {DENVER_LIVE_END_YEAR}. Confirm the ODC_CRIME_OFFENSES_P feed is alive for the "
            "requested year and bump DENVER_LIVE_END_YEAR before retrying."
        )
    effective_start_year = max(int(start_year), DENVER_LIVE_START_YEAR)
    effective_end_year = min(int(end_year), DENVER_LIVE_END_YEAR)
    if effective_end_year < effective_start_year:
        return pd.DataFrame(columns=list(DENVER_MINIMAL_COLUMNS))

    cache_path = _prepared_city_cache_path(raw_dir, f"denver_minimal_snapshot_{effective_start_year}_{effective_end_year}.parquet")
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "objectid" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()

    if raw.empty and "objectid" not in raw.columns:
        frames: list[pd.DataFrame] = []
        for year in range(effective_start_year, effective_end_year + 1):
            frame = _download_arcgis_query_frame(
                base_url=DENVER_QUERY_URL,
                params={
                    "where": (
                        f"REPORTED_DATE >= DATE '{year}-01-01 00:00:00' "
                        f"AND REPORTED_DATE < DATE '{year + 1}-01-01 00:00:00'"
                    ),
                    "orderByFields": "OBJECTID ASC",
                    "outFields": ",".join(
                        [
                            "OBJECTID",
                            "OFFENSE_CATEGORY_ID",
                            "OFFENSE_TYPE_ID",
                            "OFFENSE_CODE",
                            "OFFENSE_CODE_EXTENSION",
                            "VICTIM_COUNT",
                            "REPORTED_DATE",
                            "INCIDENT_ADDRESS",
                            "GEO_LAT",
                            "GEO_LON",
                        ]
                    ),
                    "f": "json",
                },
                out_path=raw_dir / f"denver_{year}.parquet",
                force_download=force_refresh,
                return_geometry=True,
                out_sr=4326,
            )
            frame = frame.rename(columns=DENVER_RAW_COLUMN_MAP)
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(DENVER_MINIMAL_COLUMNS))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)

    missing = set(DENVER_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Denver incident cache is missing required columns: {missing_cols}")
    raw["objectid"] = raw["objectid"].astype(str).str.strip()
    raw = raw[raw["objectid"].ne("")].copy()
    raw = raw.drop_duplicates("objectid", keep="last")
    return raw.loc[:, list(DENVER_MINIMAL_COLUMNS)].copy()


def _load_denver_offense_crosswalk(offense_crosswalk_csv: Path) -> pd.DataFrame:
    return _load_single_field_offense_crosswalk(
        offense_crosswalk_csv,
        field_aliases={"offense_category_id": "offense_category_id"},
        include_counting_basis=True,
    )


def _prepare_denver_incidents(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = _load_denver_raw_snapshot(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_refresh,
    )
    raw["reported_dt"] = pd.to_datetime(pd.to_numeric(raw["reported_date"], errors="coerce"), unit="ms", errors="coerce")
    raw["year"] = raw["reported_dt"].dt.year
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()
    if int(raw_year_totals.sum()) < DENVER_MIN_INCIDENTS_IN_WINDOW:
        raise RuntimeError(
            "Denver incident source coverage failed sanity check; "
            f"{int(start_year)}-{int(end_year)} total={int(raw_year_totals.sum())}"
        )

    raw["offense_category_id"] = raw["offense_category_id"].astype(str).str.strip().str.lower()
    crosswalk = _load_denver_offense_crosswalk(offense_crosswalk_csv)
    if crosswalk.empty:
        raise RuntimeError("Denver offense crosswalk is empty or missing required packet columns.")

    raw["offense"] = pd.NA
    raw["counting_basis"] = "row_count"
    for row in crosswalk.itertuples(index=False):
        source_field = str(getattr(row, "source_field", "")).strip()
        if not source_field or source_field not in raw.columns:
            continue
        values = set(getattr(row, "source_values", []) or [])
        if not values:
            continue
        mask = raw[source_field].astype(str).str.upper().isin(values)
        raw.loc[mask, "offense"] = str(getattr(row, "offense", "")).strip()
        raw.loc[mask, "counting_basis"] = str(getattr(row, "counting_basis", "row_count")).strip().lower()

    mapped = raw[raw["offense"].notna()].copy()
    if mapped.empty:
        raise RuntimeError("Denver offense harmonization produced no mapped incidents.")

    mapped["victim_count"] = pd.to_numeric(mapped["victim_count"], errors="coerce").fillna(0.0)
    mapped["count_weight"] = 1.0
    victim_mask = mapped["counting_basis"].eq("victim_sum")
    mapped.loc[victim_mask, "count_weight"] = mapped.loc[victim_mask, "victim_count"].where(
        mapped.loc[victim_mask, "victim_count"] > 0,
        1.0,
    )

    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    mapped["geometry_y"] = pd.to_numeric(mapped["geometry_y"], errors="coerce")
    mapped["geometry_x"] = pd.to_numeric(mapped["geometry_x"], errors="coerce")
    mapped["latitude"] = mapped["latitude"].fillna(mapped["geometry_y"])
    mapped["longitude"] = mapped["longitude"].fillna(mapped["geometry_x"])
    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_share = float(len(geocoded)) / float(len(mapped)) if len(mapped) else 0.0
    if geocoded_share < DENVER_MIN_GEOCODED_SHARE:
        raise RuntimeError(
            "Denver coordinate coverage failed sanity check; "
            f"only {geocoded_share:.1%} of mapped incidents retained coordinates."
        )
    geocoded["offense_row_id"] = geocoded["objectid"].astype(str).str.strip() + "::" + geocoded["offense"].astype(str).str.strip()
    geocoded = geocoded.drop_duplicates("offense_row_id").copy()
    return mapped, geocoded, raw_year_totals


def build_denver_incident_share_surface(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    reconciliation_summary_csv: Path,
    packet_offense_status_csv: Path,
    co_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        _, geocoded, _ = _prepare_denver_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Denver incident source could not be loaded cleanly: {exc}. Skipping Denver city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    geocoded = _drop_quarantined_coordinates(geocoded, city_key="denver")
    bg = _load_city_bg_geometry(
        bg_zip=co_bg_zip,
        bg_crosswalk_parquet=bg_crosswalk_parquet,
        jurisdiction_id=DENVER_JURISDICTION_ID,
    )
    if bg.empty:
        warnings.warn(
            "Denver jurisdiction was not present in the block-group crosswalk. Skipping Denver city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    matched = _match_point_incidents_to_city_bg(
        geocoded,
        x_col="longitude",
        y_col="latitude",
        point_crs="EPSG:4326",
        bg=bg,
        keep_cols=["offense_row_id", "year", "offense", "count_weight"],
        unique_id_col="offense_row_id",
    )
    if matched.empty:
        warnings.warn(
            "Denver incidents did not spatially match any Denver block group. Skipping Denver city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)["count_weight"]
        .sum()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = DENVER_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_denver_incident_reconciliation(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    reconciliation_summary_csv: Path,
    packet_offense_status_csv: Path,
    co_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, geocoded, raw_year_totals = _prepare_denver_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    mapped_counts = mapped.groupby(["year", "offense"], dropna=False)["count_weight"].sum().rename("mapped_offense_count").reset_index()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False)["count_weight"].sum().rename("geocoded_offense_count").reset_index()
    bg = _load_city_bg_geometry(
        bg_zip=co_bg_zip,
        bg_crosswalk_parquet=bg_crosswalk_parquet,
        jurisdiction_id=DENVER_JURISDICTION_ID,
    )
    if bg.empty:
        return _empty_city_reconciliation()
    matched = _match_point_incidents_to_city_bg(
        geocoded,
        x_col="longitude",
        y_col="latitude",
        point_crs="EPSG:4326",
        bg=bg,
        keep_cols=["offense_row_id", "year", "offense", "count_weight"],
        unique_id_col="offense_row_id",
    )
    matched_counts = matched.groupby(["year", "offense"], dropna=False)["count_weight"].sum().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    published_recon = _load_reconciliation_summary_published_recon(reconciliation_summary_csv, packet_offense_status_csv)
    return _build_city_reconciliation_table(
        city_key="denver",
        city_name=DENVER_CITY_NAME,
        jurisdiction_id=DENVER_JURISDICTION_ID,
        state_fips="08",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=published_recon,
    )


def _load_minneapolis_raw_snapshot(*, raw_dir: Path, start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    if int(end_year) > MINNEAPOLIS_LIVE_END_YEAR:
        raise CityFeedCoverageError(
            f"Minneapolis live feed requested through {int(end_year)}, but coverage is only "
            f"verified through {MINNEAPOLIS_LIVE_END_YEAR}. Confirm the Crime_Data feed is alive for "
            "the requested year and bump MINNEAPOLIS_LIVE_END_YEAR before retrying."
        )
    effective_start_year = max(int(start_year), MINNEAPOLIS_LIVE_START_YEAR)
    effective_end_year = min(int(end_year), MINNEAPOLIS_LIVE_END_YEAR)
    if effective_end_year < effective_start_year:
        return pd.DataFrame(columns=list(MINNEAPOLIS_MINIMAL_COLUMNS))

    cache_path = _prepared_city_cache_path(
        raw_dir,
        f"minneapolis_minimal_snapshot_{effective_start_year}_{effective_end_year}.parquet",
    )
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "objectid" not in raw.columns:
            raw = pd.DataFrame()
    else:
        raw = pd.DataFrame()

    if raw.empty and "objectid" not in raw.columns:
        frames: list[pd.DataFrame] = []
        for year in range(effective_start_year, effective_end_year + 1):
            frame = _download_arcgis_query_frame(
                base_url=MINNEAPOLIS_QUERY_URL,
                params={
                    "where": (
                        "Type = 'Crime Offenses (NIBRS)' "
                        f"AND Reported_Date >= DATE '{year}-01-01 00:00:00' "
                        f"AND Reported_Date < DATE '{year + 1}-01-01 00:00:00'"
                    ),
                    "orderByFields": "OBJECTID ASC",
                    "outFields": ",".join(
                        [
                            "OBJECTID",
                            "Type",
                            "NIBRS_Code",
                            "Offense_Category",
                            "Offense",
                            "Crime_Count",
                            "Reported_Date",
                            "Address",
                            "Latitude",
                            "Longitude",
                        ]
                    ),
                    "f": "json",
                },
                out_path=raw_dir / f"minneapolis_{year}.parquet",
                force_download=force_refresh,
                return_geometry=True,
                out_sr=4326,
            )
            frame = frame.rename(columns=MINNEAPOLIS_RAW_COLUMN_MAP)
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(MINNEAPOLIS_MINIMAL_COLUMNS))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)

    missing = set(MINNEAPOLIS_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"Minneapolis incident cache is missing required columns: {missing_cols}")
    raw["objectid"] = raw["objectid"].astype(str).str.strip()
    raw = raw[raw["objectid"].ne("")].copy()
    raw = raw.drop_duplicates("objectid", keep="last")
    return raw.loc[:, list(MINNEAPOLIS_MINIMAL_COLUMNS)].copy()


def _load_minneapolis_offense_crosswalk(offense_crosswalk_csv: Path) -> pd.DataFrame:
    return _load_single_field_offense_crosswalk(
        offense_crosswalk_csv,
        field_aliases={"nibrs_code": "nibrs_code"},
        include_counting_basis=True,
    )


def _prepare_minneapolis_incidents(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = _load_minneapolis_raw_snapshot(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_refresh,
    )
    raw["type"] = raw["type"].astype(str).str.strip()
    raw = raw[raw["type"].eq("Crime Offenses (NIBRS)")].copy()
    raw["reported_dt"] = pd.to_datetime(pd.to_numeric(raw["reported_date"], errors="coerce"), unit="ms", errors="coerce")
    raw["year"] = raw["reported_dt"].dt.year
    raw["crime_count"] = pd.to_numeric(raw["crime_count"], errors="coerce").fillna(0.0)
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()
    if int(raw_year_totals.sum()) < MINNEAPOLIS_MIN_INCIDENTS_IN_WINDOW:
        raise RuntimeError(
            "Minneapolis incident source coverage failed sanity check; "
            f"{int(start_year)}-{int(end_year)} total={int(raw_year_totals.sum())}"
        )

    raw["nibrs_code"] = raw["nibrs_code"].astype(str).str.strip().str.upper()
    crosswalk = _load_minneapolis_offense_crosswalk(offense_crosswalk_csv)
    if crosswalk.empty:
        raise RuntimeError("Minneapolis offense crosswalk is empty or missing required packet columns.")

    raw["offense"] = pd.NA
    raw["counting_basis"] = "row_count"
    for row in crosswalk.itertuples(index=False):
        source_field = str(getattr(row, "source_field", "")).strip()
        if not source_field or source_field not in raw.columns:
            continue
        values = set(getattr(row, "source_values", []) or [])
        if not values:
            continue
        mask = raw[source_field].astype(str).str.upper().isin(values)
        raw.loc[mask, "offense"] = str(getattr(row, "offense", "")).strip()
        raw.loc[mask, "counting_basis"] = str(getattr(row, "counting_basis", "row_count")).strip().lower()

    mapped = raw[raw["offense"].notna()].copy()
    if mapped.empty:
        raise RuntimeError("Minneapolis offense harmonization produced no mapped incidents.")

    mapped["count_weight"] = 1.0
    crime_count_mask = mapped["counting_basis"].eq("crime_count")
    mapped.loc[crime_count_mask, "count_weight"] = mapped.loc[crime_count_mask, "crime_count"].where(
        mapped.loc[crime_count_mask, "crime_count"] > 0,
        1.0,
    )
    mapped["geometry_y"] = pd.to_numeric(mapped["geometry_y"], errors="coerce")
    mapped["geometry_x"] = pd.to_numeric(mapped["geometry_x"], errors="coerce")
    mapped["latitude"] = mapped["geometry_y"].fillna(pd.to_numeric(mapped["latitude"], errors="coerce"))
    mapped["longitude"] = mapped["geometry_x"].fillna(pd.to_numeric(mapped["longitude"], errors="coerce"))
    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_share = float(len(geocoded)) / float(len(mapped)) if len(mapped) else 0.0
    if geocoded_share < MINNEAPOLIS_MIN_GEOCODED_SHARE:
        raise RuntimeError(
            "Minneapolis coordinate coverage failed sanity check; "
            f"only {geocoded_share:.1%} of mapped incidents retained coordinates."
        )
    geocoded["offense_row_id"] = geocoded["objectid"].astype(str).str.strip() + "::" + geocoded["offense"].astype(str).str.strip()
    geocoded = geocoded.drop_duplicates("offense_row_id").copy()
    return mapped, geocoded, raw_year_totals


def build_minneapolis_incident_share_surface(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    reconciliation_summary_csv: Path,
    packet_offense_status_csv: Path,
    mn_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        _, geocoded, _ = _prepare_minneapolis_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Minneapolis incident source could not be loaded cleanly: {exc}. Skipping Minneapolis city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    geocoded = _drop_quarantined_coordinates(geocoded, city_key="minneapolis")
    bg = _load_city_bg_geometry(
        bg_zip=mn_bg_zip,
        bg_crosswalk_parquet=bg_crosswalk_parquet,
        jurisdiction_id=MINNEAPOLIS_JURISDICTION_ID,
    )
    if bg.empty:
        warnings.warn(
            "Minneapolis jurisdiction was not present in the block-group crosswalk. Skipping Minneapolis city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    matched = _match_point_incidents_to_city_bg(
        geocoded,
        x_col="longitude",
        y_col="latitude",
        point_crs="EPSG:4326",
        bg=bg,
        keep_cols=["offense_row_id", "year", "offense", "count_weight"],
        unique_id_col="offense_row_id",
    )
    if matched.empty:
        warnings.warn(
            "Minneapolis incidents did not spatially match any Minneapolis block group. Skipping Minneapolis city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)["count_weight"]
        .sum()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = MINNEAPOLIS_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_minneapolis_incident_reconciliation(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    reconciliation_summary_csv: Path,
    packet_offense_status_csv: Path,
    mn_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, geocoded, raw_year_totals = _prepare_minneapolis_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    mapped_counts = mapped.groupby(["year", "offense"], dropna=False)["count_weight"].sum().rename("mapped_offense_count").reset_index()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False)["count_weight"].sum().rename("geocoded_offense_count").reset_index()
    bg = _load_city_bg_geometry(
        bg_zip=mn_bg_zip,
        bg_crosswalk_parquet=bg_crosswalk_parquet,
        jurisdiction_id=MINNEAPOLIS_JURISDICTION_ID,
    )
    if bg.empty:
        return _empty_city_reconciliation()
    matched = _match_point_incidents_to_city_bg(
        geocoded,
        x_col="longitude",
        y_col="latitude",
        point_crs="EPSG:4326",
        bg=bg,
        keep_cols=["offense_row_id", "year", "offense", "count_weight"],
        unique_id_col="offense_row_id",
    )
    matched_counts = matched.groupby(["year", "offense"], dropna=False)["count_weight"].sum().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    published_recon = _load_reconciliation_summary_published_recon(reconciliation_summary_csv, packet_offense_status_csv)
    return _build_city_reconciliation_table(
        city_key="minneapolis",
        city_name=MINNEAPOLIS_CITY_NAME,
        jurisdiction_id=MINNEAPOLIS_JURISDICTION_ID,
        state_fips="27",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=published_recon,
    )


def _download_st_louis_file(url: str, out_path: Path, *, force_download: bool = False) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0 and not force_download:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 crimerisk-v2/1.0",
            "Referer": "https://www.slmpd.org/stats/",
        },
    )
    with urlopen(request, timeout=600) as response, tmp_path.open("wb") as sink:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            sink.write(chunk)
    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty St. Louis incident snapshot from {url}")
    tmp_path.replace(out_path)
    return out_path


def _load_st_louis_snapshot_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        usecols=lambda col: col in set(ST_LOUIS_MINIMAL_COLUMNS),
        na_values=["", "NA", "(null)"],
        keep_default_na=True,
    )
    missing = set(ST_LOUIS_MINIMAL_COLUMNS) - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"St. Louis incident source schema changed; missing columns: {missing_cols}")
    return df.loc[:, list(ST_LOUIS_MINIMAL_COLUMNS)].copy()


def _load_st_louis_raw_snapshot(
    *,
    raw_dir: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if int(end_year) > ST_LOUIS_LIVE_END_YEAR:
        raise CityFeedCoverageError(
            f"St. Louis live feed requested through {int(end_year)}, but monthly SLMPD file URLs "
            f"are only registered through {ST_LOUIS_LIVE_END_YEAR}. The upload subpath in each URL "
            "lags the data month irregularly and cannot be derived; confirm the new year's twelve "
            "monthly URLs at https://www.slmpd.org/stats/ and add them to "
            "ST_LOUIS_MONTHLY_FILES_BY_YEAR."
        )
    effective_start_year = max(int(start_year), ST_LOUIS_LIVE_START_YEAR)
    effective_end_year = min(int(end_year), ST_LOUIS_LIVE_END_YEAR)
    if effective_end_year < effective_start_year:
        return pd.DataFrame(columns=[*ST_LOUIS_MINIMAL_COLUMNS, "source_month", "source_file_url"])

    cache_path = _prepared_city_cache_path(
        raw_dir,
        f"st_louis_minimal_snapshot_{effective_start_year}_{effective_end_year}.parquet",
    )
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        raw = pd.read_parquet(cache_path)
    else:
        frames: list[pd.DataFrame] = []
        for year in range(effective_start_year, effective_end_year + 1):
            monthly_files = ST_LOUIS_MONTHLY_FILES_BY_YEAR.get(year)
            if monthly_files is None:
                raise CityFeedCoverageError(
                    f"St. Louis monthly SLMPD file URLs are not registered for {year}; add the "
                    "verified URLs to ST_LOUIS_MONTHLY_FILES_BY_YEAR."
                )
            for month_number, month_name, url in monthly_files:
                csv_path = _download_st_louis_file(
                    url,
                    raw_dir / f"st_louis_{year}_{month_number:02d}_{month_name}.csv",
                    force_download=force_refresh,
                )
                frame = _load_st_louis_snapshot_csv(csv_path)
                frame["source_month"] = int(month_number)
                frame["source_file_url"] = url
                frames.append(frame)
        raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(
            columns=[*ST_LOUIS_MINIMAL_COLUMNS, "source_month", "source_file_url"]
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False)

    missing = set(ST_LOUIS_MINIMAL_COLUMNS) - set(raw.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise RuntimeError(f"St. Louis incident cache is missing required columns: {missing_cols}")
    return raw.copy()


def _build_st_louis_offense_row_id(mapped: pd.DataFrame) -> pd.Series:
    base_cols = [
        "IncidentNum",
        "NIBRS",
        "Offense",
        "IncidentLocation",
        "IntersectionOtherLoc",
        "Latitude",
        "Longitude",
    ]
    victim_cols = [*base_cols, "VictimNum"]

    def _joined(cols: list[str]) -> pd.Series:
        return (
            mapped[cols]
            .fillna("")
            .astype("string")
            .agg("|".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    ids = _joined(base_cols)
    victim_ids = _joined(victim_cols)
    victim_mask = mapped["offense"].isin(ST_LOUIS_VICTIM_COUNTED_OFFENSES)
    ids.loc[victim_mask] = victim_ids.loc[victim_mask]
    return ids


def _prepare_st_louis_incidents(
    *,
    raw_dir: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    effective_start_year = max(int(start_year), ST_LOUIS_LIVE_START_YEAR)
    effective_end_year = min(int(end_year), ST_LOUIS_LIVE_END_YEAR)
    raw = _load_st_louis_raw_snapshot(
        raw_dir=raw_dir,
        start_year=effective_start_year,
        # Pass the caller's requested end year (not the clamp) so the loader's coverage
        # check raises loudly when the request exceeds the registered SLMPD URLs.
        end_year=int(end_year),
        force_refresh=force_refresh,
    )
    raw["incident_dt"] = pd.to_datetime(raw["IncidentDate"], errors="coerce", format="mixed")
    raw["year"] = raw["incident_dt"].dt.year
    raw = raw[raw["year"].between(effective_start_year, effective_end_year, inclusive="both")].copy()
    raw_year_totals = raw.groupby("year", dropna=False).size()
    if int(raw_year_totals.sum()) < ST_LOUIS_MIN_INCIDENTS_IN_WINDOW:
        raise RuntimeError(
            "St. Louis incident source coverage failed sanity check; "
            f"{effective_start_year}-{effective_end_year} total={int(raw_year_totals.sum())}"
        )

    raw["nibrs_code"] = raw["NIBRS"].astype("string").str.strip().str.upper()
    raw["offense"] = raw["nibrs_code"].map(ST_LOUIS_OFFENSE_MAP)
    mapped = raw[raw["offense"].notna()].copy()
    if mapped.empty:
        raise RuntimeError("St. Louis offense harmonization produced no mapped incidents.")

    mapped["latitude"] = pd.to_numeric(mapped["Latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["Longitude"], errors="coerce")
    mapped["offense_row_id"] = _build_st_louis_offense_row_id(mapped)
    mapped = mapped[mapped["offense_row_id"].ne("") & mapped["offense_row_id"].ne("<NA>")].copy()
    mapped = mapped.drop_duplicates("offense_row_id", keep="first")
    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_share = float(len(geocoded)) / float(len(mapped)) if len(mapped) else 0.0
    if geocoded_share < ST_LOUIS_MIN_GEOCODED_SHARE:
        raise RuntimeError(
            "St. Louis coordinate coverage failed sanity check; "
            f"only {geocoded_share:.1%} of mapped incidents retained coordinates."
        )
    return mapped, geocoded, raw_year_totals


def build_st_louis_incident_share_surface(
    *,
    raw_dir: Path,
    mo_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        _, geocoded, _ = _prepare_st_louis_incidents(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"St. Louis incident source could not be loaded cleanly: {exc}. Skipping St. Louis city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    geocoded = _drop_quarantined_coordinates(geocoded, city_key="st_louis_mo")
    bg = _load_city_bg_geometry(
        bg_zip=mo_bg_zip,
        bg_crosswalk_parquet=bg_crosswalk_parquet,
        jurisdiction_id=ST_LOUIS_JURISDICTION_ID,
    )
    if bg.empty:
        warnings.warn(
            "St. Louis jurisdiction was not present in the block-group crosswalk. Skipping St. Louis city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    matched = _match_point_incidents_to_city_bg(
        geocoded,
        x_col="longitude",
        y_col="latitude",
        point_crs="EPSG:4326",
        bg=bg,
        keep_cols=["offense_row_id", "year", "offense"],
        unique_id_col="offense_row_id",
    )
    if matched.empty:
        warnings.warn(
            "St. Louis incidents did not spatially match any St. Louis block group. Skipping St. Louis city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = ST_LOUIS_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_st_louis_incident_reconciliation(
    *,
    raw_dir: Path,
    mo_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2024,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, geocoded, raw_year_totals = _prepare_st_louis_incidents(
            raw_dir=raw_dir,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()

    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()
    bg = _load_city_bg_geometry(
        bg_zip=mo_bg_zip,
        bg_crosswalk_parquet=bg_crosswalk_parquet,
        jurisdiction_id=ST_LOUIS_JURISDICTION_ID,
    )
    if bg.empty:
        return _empty_city_reconciliation()
    matched = _match_point_incidents_to_city_bg(
        geocoded,
        x_col="longitude",
        y_col="latitude",
        point_crs="EPSG:4326",
        bg=bg,
        keep_cols=["offense_row_id", "year", "offense"],
        unique_id_col="offense_row_id",
    )
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    return _build_city_reconciliation_table(
        city_key="st_louis_mo",
        city_name=ST_LOUIS_CITY_NAME,
        jurisdiction_id=ST_LOUIS_JURISDICTION_ID,
        state_fips="29",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=ST_LOUIS_PUBLISHED_RECON,
    )


def _load_mesa_offense_crosswalk(offense_crosswalk_csv: Path) -> pd.DataFrame:
    crosswalk = pd.read_csv(offense_crosswalk_csv, dtype=str).fillna("").copy()
    required = {"source_value", "v2_offense", "counting_basis"}
    if crosswalk.empty or not required.issubset(crosswalk.columns):
        return pd.DataFrame(columns=["v2_offense", "counting_basis", "codes"])
    crosswalk["v2_offense"] = crosswalk["v2_offense"].astype(str).str.strip()
    crosswalk["counting_basis"] = crosswalk["counting_basis"].astype(str).str.strip()
    crosswalk["codes"] = crosswalk["source_value"].apply(
        lambda value: {code.strip() for code in str(value).split("|") if code.strip()}
    )
    return crosswalk[crosswalk["v2_offense"].ne("")].copy()


def _load_mesa_published_recon(published_reference_csv: Path) -> dict[int, dict[str, object]]:
    published = pd.read_csv(published_reference_csv, dtype=str).fillna("").copy()
    if published.empty or "offense" not in published.columns or "published_count" not in published.columns:
        return {}

    alias_map = {"rape_proxy_sexual_assault": "rape"}
    counts: dict[str, int] = {}
    quality: dict[str, str] = {}
    notes_by_offense: dict[str, str] = {}
    source_name = ""
    source_url = ""
    for row in published.itertuples(index=False):
        offense = alias_map.get(str(row.offense).strip(), str(row.offense).strip())
        if offense not in {
            "murder",
            "rape",
            "robbery",
            "aggravated_assault",
            "burglary",
            "larceny",
            "motor_vehicle_theft",
        }:
            continue
        count = pd.to_numeric(getattr(row, "published_count", ""), errors="coerce")
        if pd.isna(count):
            continue
        counts[offense] = int(count)
        quality[offense] = str(getattr(row, "comparison_quality", "") or "approximate").strip() or "approximate"
        notes_by_offense[offense] = str(getattr(row, "notes", "")).strip()
        if not source_name:
            source_name = str(getattr(row, "published_source_name", "")).strip()
        if not source_url:
            source_url = str(getattr(row, "published_source_url", "")).strip()

    if not counts:
        return {}

    ordered_notes = [notes_by_offense[offense] for offense in counts if notes_by_offense.get(offense)]
    return {
        2024: {
            "source_name": source_name,
            "source_url": source_url,
            "counts": counts,
            "comparison_quality": quality,
            "notes": " | ".join(ordered_notes),
        }
    }


def _prepare_mesa_incidents(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    raw = _load_mesa_raw_snapshot(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force_refresh=force_refresh,
    ).copy()
    raw["report_year_num"] = pd.to_numeric(raw["report_year"], errors="coerce")
    report_dt = pd.to_datetime(raw["report_date"], errors="coerce")
    occurred_dt = pd.to_datetime(raw["occurred_date"], errors="coerce")
    raw["year"] = (
        raw["report_year_num"]
        .fillna(pd.to_numeric(occurred_dt.dt.year, errors="coerce"))
        .fillna(pd.to_numeric(report_dt.dt.year, errors="coerce"))
    )
    raw = raw[raw["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw = raw[raw["year"].notna()].copy()
    raw["year"] = raw["year"].astype(int)
    raw_year_totals = raw.groupby("year", dropna=False).size()
    if raw.empty:
        return pd.DataFrame(), raw_year_totals

    raw["code_list"] = raw["nibrs_code"].apply(_split_delimited_codes)
    raw["first_code"] = raw["code_list"].apply(lambda codes: codes[0] if codes else None)
    coords = raw["location"].astype("string").str.extract(MESA_LOCATION_RE)
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce").fillna(
        pd.to_numeric(coords[0], errors="coerce")
    )
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce").fillna(
        pd.to_numeric(coords[1], errors="coerce")
    )

    crosswalk = _load_mesa_offense_crosswalk(offense_crosswalk_csv)
    frames: list[pd.DataFrame] = []
    for row in crosswalk.itertuples(index=False):
        if row.counting_basis == "incident_has_code":
            mask = raw["code_list"].apply(lambda codes: any(code in row.codes for code in codes))
        elif row.counting_basis == "first_listed_code_only":
            mask = raw["first_code"].isin(row.codes)
        else:
            continue
        subset = raw.loc[mask, ["crime_id", "year", "latitude", "longitude"]].copy()
        if subset.empty:
            continue
        subset["offense"] = row.v2_offense
        subset["offense_row_id"] = subset["crime_id"] + "::" + subset["offense"]
        frames.append(subset)
    if not frames:
        return pd.DataFrame(), raw_year_totals

    mapped = pd.concat(frames, ignore_index=True, sort=False)
    mapped = mapped.drop_duplicates(["offense_row_id"]).reset_index(drop=True)
    return mapped, raw_year_totals


def build_mesa_incident_share_surface(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    az_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2020,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, _ = _prepare_mesa_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"Mesa incident source could not be loaded cleanly: {exc}. Skipping Mesa city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    if mapped.empty:
        return _empty_city_surface()

    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded = _drop_quarantined_coordinates(geocoded, city_key="mesa")
    if geocoded.empty:
        return _empty_city_surface()
    incidents = gpd.GeoDataFrame(
        geocoded[["offense_row_id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(az_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "offense", "block_group_geoid"]
    ].drop_duplicates("offense_row_id")
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = matched.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(MESA_JURISDICTION_ID)].copy()
    if matched.empty:
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = MESA_CITY_NAME
    counts["geocode_quality_tier"] = "rounded_hundred_block_point"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_mesa_incident_reconciliation(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    az_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2020,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, raw_year_totals = _prepare_mesa_incidents(
            raw_dir=raw_dir,
            offense_crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()
    if mapped.empty:
        return _empty_city_reconciliation()

    mapped_counts = mapped.groupby(["year", "offense"], dropna=False).size().rename("mapped_offense_count").reset_index()
    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_counts = geocoded.groupby(["year", "offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()
    if geocoded.empty:
        return _empty_city_reconciliation()

    incidents = gpd.GeoDataFrame(
        geocoded[["offense_row_id", "year", "offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(az_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "offense", "block_group_geoid"]
    ].drop_duplicates("offense_row_id")
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = matched.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(MESA_JURISDICTION_ID)].copy()
    matched_counts = matched.groupby(["year", "offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    return _build_city_reconciliation_table(
        city_key="mesa",
        city_name=MESA_CITY_NAME,
        jurisdiction_id=MESA_JURISDICTION_ID,
        state_fips="04",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=_load_mesa_published_recon(published_reference_csv),
    )


def build_san_francisco_incident_share_surface(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path | None = None,
    packet_offense_status_csv: Path | None = None,
    ca_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, _ = _prepare_san_francisco_incidents(
            raw_dir=raw_dir,
            crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception as exc:
        warnings.warn(
            f"San Francisco incident source could not be loaded cleanly: {exc}. Skipping San Francisco city-incident surface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_city_surface()
    if mapped.empty:
        return _empty_city_surface()

    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded = _drop_quarantined_coordinates(geocoded, city_key="san_francisco", offense_col="v2_offense")
    if geocoded.empty:
        return _empty_city_surface()
    incidents = gpd.GeoDataFrame(
        geocoded[["offense_row_id", "year", "v2_offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(ca_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "v2_offense", "block_group_geoid"]
    ].drop_duplicates("offense_row_id")
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = matched.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(SAN_FRANCISCO_JURISDICTION_ID)].copy()
    if matched.empty:
        return _empty_city_surface()

    counts = (
        matched.groupby(["year", "v2_offense", "block_group_geoid", "jurisdiction_id", "state_fips"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
        .rename(columns={"v2_offense": "offense"})
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"]
    counts["city_name"] = SAN_FRANCISCO_CITY_NAME
    counts["geocode_quality_tier"] = "reported_latlon"
    return counts[
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
        ]
    ].sort_values(["year", "offense", "block_group_geoid"], kind="mergesort").reset_index(drop=True)


def build_san_francisco_incident_reconciliation(
    *,
    raw_dir: Path,
    offense_crosswalk_csv: Path,
    published_reference_csv: Path,
    packet_offense_status_csv: Path,
    ca_bg_zip: Path,
    bg_crosswalk_parquet: Path,
    start_year: int = 2018,
    end_year: int = 2024,
    force_source_refresh: bool = False,
) -> pd.DataFrame:
    try:
        mapped, raw_year_totals = _prepare_san_francisco_incidents(
            raw_dir=raw_dir,
            crosswalk_csv=offense_crosswalk_csv,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_source_refresh,
        )
    except Exception:
        return _empty_city_reconciliation()
    if mapped.empty:
        return _empty_city_reconciliation()

    mapped_counts = mapped.groupby(["year", "v2_offense"], dropna=False).size().rename("mapped_offense_count").reset_index()
    mapped_counts = mapped_counts.rename(columns={"v2_offense": "offense"})

    geocoded = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()].copy()
    geocoded_counts = geocoded.groupby(["year", "v2_offense"], dropna=False).size().rename("geocoded_offense_count").reset_index()
    geocoded_counts = geocoded_counts.rename(columns={"v2_offense": "offense"})
    if geocoded.empty:
        return _empty_city_reconciliation()

    incidents = gpd.GeoDataFrame(
        geocoded[["offense_row_id", "year", "v2_offense", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(geocoded["longitude"], geocoded["latitude"]),
        crs="EPSG:4326",
    )
    bg = gpd.read_file(ca_bg_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype(str).str.zfill(12)
    if bg.crs != incidents.crs:
        bg = bg.to_crs(incidents.crs)
    matched = gpd.sjoin(incidents, bg, how="inner", predicate="within")[
        ["offense_row_id", "year", "v2_offense", "block_group_geoid"]
    ].drop_duplicates("offense_row_id")
    crosswalk = pd.read_parquet(bg_crosswalk_parquet)[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    matched = matched.merge(crosswalk, on="block_group_geoid", how="left")
    matched = matched[matched["jurisdiction_id"].astype(str).eq(SAN_FRANCISCO_JURISDICTION_ID)].copy()
    matched_counts = matched.groupby(["year", "v2_offense"], dropna=False).size().rename("matched_offense_count").reset_index()
    matched_counts = matched_counts.rename(columns={"v2_offense": "offense"})
    final_counts = matched_counts.rename(columns={"matched_offense_count": "final_share_count"})
    published_recon = _load_san_francisco_published_recon(
        published_reference_csv=published_reference_csv,
        packet_status_csv=packet_offense_status_csv,
    )
    return _build_city_reconciliation_table(
        city_key="san_francisco",
        city_name=SAN_FRANCISCO_CITY_NAME,
        jurisdiction_id=SAN_FRANCISCO_JURISDICTION_ID,
        state_fips="06",
        raw_year_totals=raw_year_totals,
        mapped_counts=mapped_counts,
        geocoded_counts=geocoded_counts,
        matched_counts=matched_counts,
        final_counts=final_counts,
        published_recon=published_recon,
    )
