"""The external raw-data contract for the release build.

Single source of truth for which external files `build-release` consumes, where each
one lives under `data/`, and how to obtain it. Consumed by `main.py
build-input-manifest` (presence checking) and by the submission-package materials
generator (`required_inputs.csv` and the package README).

Acquisition provenance for the FBI sources is documented in docs/FBI-DATA-GUIDE.md
section 2.1.
"""

from __future__ import annotations

import json
from pathlib import Path


KAPLAN_RETURN_A_URL = "https://www.openicpsr.org/openicpsr/project/100707"
KAPLAN_NIBRS_URL = "https://www.openicpsr.org/openicpsr/project/118281"
CDE_DOWNLOADS = "https://cde.ucr.cjis.gov (Documents & Downloads page)"

_KAPLAN_NOTE = (
    "Free openICPSR account required. This build used the version distributed "
    "2025-08-21 (files dated 2025-08-15)."
)

REQUIRED_INPUTS: list[dict[str, str]] = [
    # --- FBI crime data -----------------------------------------------------------
    {
        "stage": "Reference and observations",
        "dataset": "Kaplan SRS Return A annual extract",
        "expected_path": "data/SRS-Kaplan-1960-2024/offenses_known_parquet_1960_2024_year.zip",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "1960-2024",
        "consumed_by": "agency master, observations, controls",
        "how_to_obtain": (
            f"Download 'offenses_known_parquet_1960_2024_year.zip' from Jacob Kaplan's "
            f"openICPSR project 'Offenses Known and Clearances by Arrest (Return A)': "
            f"{KAPLAN_RETURN_A_URL}. {_KAPLAN_NOTE}"
        ),
        "notes": "The canonical annual summary-crime input across the release build.",
    },
    {
        "stage": "Reporting regimes",
        "dataset": "Kaplan SRS Return A monthly extract",
        "expected_path": "data/SRS-Kaplan-1960-2024/offenses_known_parquet_1960_2024_month.zip",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "1960-2024",
        "consumed_by": "reporting regimes, observations",
        "how_to_obtain": (
            f"Download 'offenses_known_parquet_1960_2024_month.zip' from the same "
            f"openICPSR project: {KAPLAN_RETURN_A_URL}. {_KAPLAN_NOTE}"
        ),
        "notes": "Used to derive true SRS month coverage and reporting-regime labels.",
    },
    {
        "stage": "Observations",
        "dataset": "Kaplan NIBRS offense segment",
        "expected_path": "data/NIBRS-Kaplan-1991-2024/offense_segment_parquet_1991_2024.zip",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "1991-2024",
        "consumed_by": "observations (NIBRS SRS-equivalent rollup)",
        "how_to_obtain": (
            f"Download 'offense_segment_parquet_1991_2024.zip' from Jacob Kaplan's "
            f"openICPSR NIBRS project: {KAPLAN_NIBRS_URL}. {_KAPLAN_NOTE}"
        ),
        "notes": "Incident/offense flags, hotel-rule premises, and incident months.",
    },
    {
        "stage": "Observations",
        "dataset": "Kaplan NIBRS victim segment",
        "expected_path": "data/NIBRS-Kaplan-1991-2024/victim_segment_parquet_1991_2024.zip",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "1991-2024",
        "consumed_by": "observations (NIBRS SRS-equivalent rollup)",
        "how_to_obtain": (
            f"Download 'victim_segment_parquet_1991_2024.zip' from the same openICPSR "
            f"NIBRS project: {KAPLAN_NIBRS_URL}. {_KAPLAN_NOTE}"
        ),
        "notes": "Per-victim counting for murder, rape, and aggravated assault.",
    },
    {
        "stage": "Observations",
        "dataset": "Kaplan NIBRS property segment",
        "expected_path": "data/NIBRS-Kaplan-1991-2024/property_segment_parquet_1991_2024.zip",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "1991-2024",
        "consumed_by": "observations (NIBRS SRS-equivalent rollup)",
        "how_to_obtain": (
            f"Download 'property_segment_parquet_1991_2024.zip' from the same openICPSR "
            f"NIBRS project: {KAPLAN_NIBRS_URL}. {_KAPLAN_NOTE}"
        ),
        "notes": "Per-stolen-vehicle counting for motor vehicle theft.",
    },
    {
        "stage": "Reference, observations, and reporting regimes",
        "dataset": "Kaplan NIBRS batch header",
        "expected_path": "data/NIBRS-Kaplan-1991-2024/batch_header_parquet_1991_2024.zip",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "1991-2024",
        "consumed_by": "agency master, observations, reporting regimes",
        "how_to_obtain": (
            f"Download 'batch_header_parquet_1991_2024.zip' from the same openICPSR "
            f"NIBRS project: {KAPLAN_NIBRS_URL}. {_KAPLAN_NOTE}"
        ),
        "notes": "NIBRS month coverage and agency metadata.",
    },
    {
        "stage": "Observations",
        "dataset": "FBI CIUS 'Offenses Known to Law Enforcement' table bundles (2020-2024)",
        "expected_path": "data/FBI-CIUS-Annual/<year>/raw/offenses-known-to-le-<year>.zip",
        "path_kind": "per_year_zip_set",
        "required": "yes",
        "acquisition_type": "public download set",
        "vintage": "2020-2024 (one bundle per publication year)",
        "consumed_by": "observations (published CIUS local agency rows)",
        "how_to_obtain": (
            f"Download each publication year's 'Offenses Known to Law Enforcement' "
            f"table bundle from the FBI Crime Data Explorer: {CDE_DOWNLOADS}. Place each "
            f"zip under data/FBI-CIUS-Annual/<year>/raw/ with the name shown."
        ),
        "notes": "Frozen publication snapshots; the highest-priority source family.",
    },
    {
        "stage": "Observations",
        "dataset": "FBI CIUS 'Offenses Known' published tables (2018-2019)",
        "expected_path": "data/FBI-CIUS-Annual/<year>/raw/ (per-table .xls files)",
        "path_kind": "per_year_dir_set_2018_2019",
        "required": "yes",
        "acquisition_type": "public download set",
        "vintage": "2018-2019 publications",
        "consumed_by": "observations (published CIUS local agency rows)",
        "how_to_obtain": (
            "The 2018 and 2019 'Crime in the United States' publications ship per-table "
            ".xls files (Tables 8/9/11 'Offenses Known to Law Enforcement' state cuts) "
            "from the legacy CIUS pages at ucr.fbi.gov. Place the .xls files directly "
            "under data/FBI-CIUS-Annual/<year>/raw/."
        ),
        "notes": "Pre-2020 publications were per-table files, not a single zip.",
    },
    {
        "stage": "Controls and source selection",
        "dataset": "FBI NIBRS published-agency tables 2024",
        "expected_path": "data/FBI-NIBRS-Tables-2024/raw/statesAndFederal.zip",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "2024",
        "consumed_by": "controls, published-NIBRS source support",
        "how_to_obtain": (
            f"Download the 2024 NIBRS state/federal agency table bundle from the FBI "
            f"Crime Data Explorer: {CDE_DOWNLOADS}."
        ),
        "notes": "Published NIBRS agency surface used in source arbitration.",
    },
    {
        "stage": "Controls and outputs",
        "dataset": "FBI CDE estimated state totals",
        "expected_path": "data/FBI-CDE-Estimates-1979-2024/estimated_crimes_1979_2024.csv",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "1979-2024",
        "consumed_by": "state comparison surface, FBI-calibrated output",
        "how_to_obtain": (
            f"Download 'estimated_crimes_1979_2024.csv' (Summary data with estimates) "
            f"from the FBI Crime Data Explorer: {CDE_DOWNLOADS}. The pre-2025 direct S3 "
            f"link is defunct; use the CDE downloads page."
        ),
        "notes": "Benchmark layer only; never used to define local remainders.",
    },
    # --- Census population and geometry --------------------------------------------
    {
        "stage": "Controls and outputs",
        "dataset": "Census county population estimates",
        "expected_path": "data/Census-PopEst-2020-2025/co-est2025-alldata.csv",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "public download",
        "vintage": "Vintage 2025",
        "consumed_by": "controls, feature build, output denominators",
        "how_to_obtain": (
            "Download 'co-est2025-alldata.csv' from census.gov County Population Totals "
            "2020-2025: https://www2.census.gov/programs-surveys/popest/datasets/"
            "2020-2025/counties/totals/co-est2025-alldata.csv"
        ),
        "notes": "Updates 2024 populations for controls and published denominators.",
    },
    {
        "stage": "Geometry",
        "dataset": "TIGER 2020 block geometry",
        "expected_path": "data/tiger_tabblock20/tl_2020_<state_fips>_tabblock20.zip",
        "path_kind": "per_state_zip_set",
        "required": "yes",
        "acquisition_type": "public download set",
        "vintage": "TIGER/Line 2020",
        "consumed_by": "block-to-jurisdiction crosswalk",
        "how_to_obtain": (
            "Download per-state zips from "
            "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/ (one per release "
            "state, FIPS-named as shown)."
        ),
        "notes": "Release scope is the 48 contiguous states plus DC.",
    },
    {
        "stage": "Geometry",
        "dataset": "TIGER 2020 place geometry",
        "expected_path": "data/tiger_places/tl_2020_<state_fips>_place.zip",
        "path_kind": "per_state_zip_set",
        "required": "yes",
        "acquisition_type": "public download set",
        "vintage": "TIGER/Line 2020",
        "consumed_by": "jurisdiction geometry",
        "how_to_obtain": (
            "Download per-state zips from "
            "https://www2.census.gov/geo/tiger/TIGER2020/PLACE/."
        ),
        "notes": "",
    },
    {
        "stage": "Geometry",
        "dataset": "TIGER 2020 county-subdivision geometry",
        "expected_path": "data/tiger_cousub/tl_2020_<state_fips>_cousub.zip",
        "path_kind": "per_state_zip_set",
        "required": "yes",
        "acquisition_type": "public download set",
        "vintage": "TIGER/Line 2020",
        "consumed_by": "jurisdiction geometry (strong-MCD states)",
        "how_to_obtain": (
            "Download per-state zips from "
            "https://www2.census.gov/geo/tiger/TIGER2020/COUSUB/."
        ),
        "notes": "",
    },
    {
        "stage": "Geometry",
        "dataset": "TIGER 2020 block-group geometry (benchmark-city states)",
        "expected_path": "data/tiger_bg/tl_2020_<state_fips>_bg.zip",
        "path_kind": "per_state_zip_set",
        "required": "yes",
        "acquisition_type": "public download set",
        "vintage": "TIGER/Line 2020",
        "consumed_by": "city incident share surface",
        "how_to_obtain": (
            "Download per-state zips from "
            "https://www2.census.gov/geo/tiger/TIGER2020/BG/ for the benchmark-city "
            "states."
        ),
        "notes": "",
    },
    # --- Prepared covariate extracts ------------------------------------------------
    {
        "stage": "Feature build",
        "dataset": "ACS block-group covariates (prepared extract)",
        "expected_path": "data/ACS-5yr-2020-2024/parsed/acs_block_groups.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "ACS 2020-2024 5-year",
        "consumed_by": "BG prior model",
        "how_to_obtain": (
            "Prepared from ACS 2020-2024 5-year detailed tables (data.census.gov / "
            "Census API) by this project's development workspace. The parsed parquet at "
            "the exact path shown is the build contract."
        ),
        "notes": "The package does not rebuild ACS covariates from raw tables.",
    },
    {
        "stage": "Feature build",
        "dataset": "ACS tract covariates (prepared extract)",
        "expected_path": "data/ACS-5yr-2020-2024/parsed/acs_tracts_full.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "ACS 2020-2024 5-year",
        "consumed_by": "jurisdiction model features",
        "how_to_obtain": (
            "Prepared from ACS 2020-2024 5-year detailed tables, as above."
        ),
        "notes": "",
    },
    {
        "stage": "Feature build",
        "dataset": "LODES workplace-area covariates (prepared extract)",
        "expected_path": "data/LODES/parsed/lodes_wac_block_groups.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "LODES8",
        "consumed_by": "BG prior model, exposure proxy",
        "how_to_obtain": (
            "Prepared from LEHD LODES8 workplace-area-characteristic files "
            "(https://lehd.ces.census.gov/data/) aggregated to 2020 block groups."
        ),
        "notes": "",
    },
    {
        "stage": "Feature build",
        "dataset": "LandScan USA 2021 block-group daytime population (prepared extract)",
        "expected_path": "data/LandScan-USA/block_group_landscan_usa_2021.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared CC BY 4.0 public-data extract",
        "vintage": "LandScan USA 2021",
        "consumed_by": "person-exposure denominator construction",
        "how_to_obtain": (
            "Prepared from Oak Ridge National Laboratory LandScan USA 2021 day raster "
            "(https://landscan.ornl.gov), aggregated to the release 2020 block-group vocabulary. "
            "The parsed parquet at the exact path shown is the build contract."
        ),
        "notes": "LandScan USA excludes transitory populations such as tourists; retain ORNL/LandScan USA CC BY 4.0 attribution.",
    },
    {
        "stage": "Feature build",
        "dataset": "Road metrics (prepared extract)",
        "expected_path": "data/roads/parsed/block_group_road_metrics.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "TIGER/Line 2020 roads",
        "consumed_by": "BG prior model",
        "how_to_obtain": (
            "Prepared from TIGER/Line 2020 per-state roads shapefiles "
            "(https://www2.census.gov/geo/tiger/TIGER2020/ROADS/) summarized to block "
            "groups."
        ),
        "notes": "",
    },
    {
        "stage": "Feature build",
        "dataset": "HPMS metrics (prepared extract)",
        "expected_path": "data/HPMS/parsed/block_group_hpms_2024.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "2024",
        "consumed_by": "BG prior model",
        "how_to_obtain": (
            "Prepared from FHWA Highway Performance Monitoring System public shapefiles "
            "(https://www.fhwa.dot.gov/policyinformation/hpms.cfm) summarized to block "
            "groups."
        ),
        "notes": "",
    },
    {
        "stage": "Feature build",
        "dataset": "NCES EDGE education anchors (prepared extract)",
        "expected_path": "data/NCES-EDGE/parsed/block_group_education_anchors_2425.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "2024-25 school year",
        "consumed_by": "BG prior model",
        "how_to_obtain": (
            "Prepared from NCES EDGE public school location files "
            "(https://nces.ed.gov/programs/edge/) snapped to block groups."
        ),
        "notes": "",
    },
    {
        "stage": "Feature build",
        "dataset": "CMS hospital anchors (prepared extract)",
        "expected_path": "data/CMS-Hospital-General-Info/parsed/block_group_hospital_anchors.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "current CMS provider file",
        "consumed_by": "BG prior model",
        "how_to_obtain": (
            "Prepared from the CMS Hospital General Information provider file "
            "(https://data.cms.gov/provider-data/) geocoded to block groups."
        ),
        "notes": "",
    },
    {
        "stage": "Feature build",
        "dataset": "NLCD land-cover covariates (prepared extract)",
        "expected_path": "data/NLCD/parsed/block_group_nlcd_2023.parquet",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "NLCD 2023",
        "consumed_by": "BG prior model",
        "how_to_obtain": (
            "Prepared from MRLC NLCD 2023 land-cover rasters (https://www.mrlc.gov/) "
            "summarized to block groups."
        ),
        "notes": "",
    },
    # --- State publication extracts --------------------------------------------------
    {
        "stage": "State publications",
        "dataset": "Florida FDLE FIBRS offense-period extract",
        "expected_path": "data/FDLE-FIBRS-2024/parsed/fdle_fibrs_2024_offense_period_extract.csv",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "2024",
        "consumed_by": "state publication source rows",
        "how_to_obtain": (
            "Prepared from FDLE's public FIBRS/UCR reporting site "
            "(https://www.fdle.state.fl.us/CJAB/UCR). The extract CSV at the path shown "
            "is the build contract."
        ),
        "notes": "Florida agencies do not appear in the FBI national files for 2024.",
    },
    {
        "stage": "State publications",
        "dataset": "Mississippi TOPS raw report cache",
        "expected_path": "data/MS-TOPS-2024/raw/",
        "path_kind": "dir",
        "required": "no",
        "acquisition_type": "public web export cache",
        "vintage": "2024",
        "consumed_by": "state publication source rows",
        "how_to_obtain": (
            "Export 2024 offense reports from the Mississippi TOPS public portal; the "
            "build reads the cached exports under the directory shown."
        ),
        "notes": "Optional; the build proceeds without it.",
    },
    {
        "stage": "State publications",
        "dataset": "NY DCJS index crimes by county and agency extract",
        "expected_path": "data/NY-DCJS-2024/parsed/ny_dcjs_index_crimes_2021_2024.csv",
        "path_kind": "file",
        "required": "yes",
        "acquisition_type": "prepared public-data extract",
        "vintage": "2021-2024",
        "consumed_by": "state publication source rows (NY lane, 2021-2024 trend span)",
        "how_to_obtain": (
            "Run scripts/pull/pull_ny_dcjs_index_crimes.py; it pulls Socrata dataset "
            "ca8h-8gjq from data.ny.gov, caches the raw pages under data/NY-DCJS-2024/raw/, "
            "and writes this deterministic parsed extract."
        ),
        "notes": (
            "Official DCJS statewide UCR publication (agencies -> DCJS -> FBI chain); "
            "preferred per-agency for NY where a confident ORI match exists."
        ),
    },
    # --- City incident caches ---------------------------------------------------------
    {
        "stage": "City incident shares",
        "dataset": "City incident caches (12 cities)",
        "expected_path": "data/city-incidents/<city>/",
        "path_kind": "city_cache_set",
        "required": "yes",
        "acquisition_type": "public city open-data cache",
        "vintage": "pulled 2026-03/04; multi-year incident extracts",
        "consumed_by": "city incident share surface, direct city overrides",
        "how_to_obtain": (
            "Each city's cache is downloaded from that city's public open-data portal. "
            "The exact dataset endpoint, format, and field contract per city are "
            "recorded in configs/city_incident_sources.csv (shipped in this package). "
            "Cities: nyc, chicago, boston, philadelphia, baltimore, seattle, austin, "
            "mesa, san_francisco, denver, minneapolis, washington_dc."
        ),
        "notes": "Place each city's files under data/city-incidents/<city>/.",
    },
]

CITY_CACHE_DIRS = (
    "nyc",
    "chicago",
    "boston",
    "philadelphia",
    "baltimore",
    "seattle",
    "austin",
    "mesa",
    "san_francisco",
    "denver",
    "minneapolis",
    "washington_dc",
)

RELEASE_STATE_FIPS = (
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13", "16", "17", "18",
    "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
    "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45",
    "46", "47", "48", "49", "50", "51", "53", "54", "55", "56",
)

CIUS_ZIP_YEARS = tuple(range(2020, 2025))
CIUS_XLS_YEARS = (2018, 2019)


def _check_row(repo_root: Path, row: dict[str, str]) -> dict[str, object]:
    kind = row["path_kind"]
    expected = row["expected_path"]
    result: dict[str, object] = dict(row)
    if kind == "file":
        result["present"] = (repo_root / expected).exists()
    elif kind == "dir":
        path = repo_root / expected
        result["present"] = path.is_dir() and any(path.iterdir())
    elif kind == "per_state_zip_set":
        found = sum(
            1
            for fips in RELEASE_STATE_FIPS
            if (repo_root / expected.replace("<state_fips>", fips)).exists()
        )
        result["present"] = found == len(RELEASE_STATE_FIPS)
        result["detail"] = f"{found}/{len(RELEASE_STATE_FIPS)} state files present"
    elif kind == "per_year_zip_set":
        found = sum(
            1
            for year in CIUS_ZIP_YEARS
            if (repo_root / expected.replace("<year>", str(year))).exists()
        )
        result["present"] = found == len(CIUS_ZIP_YEARS)
        result["detail"] = f"{found}/{len(CIUS_ZIP_YEARS)} year bundles present"
    elif kind == "per_year_dir_set_2018_2019":
        found = sum(
            1
            for year in CIUS_XLS_YEARS
            if any((repo_root / f"data/FBI-CIUS-Annual/{year}/raw").glob("*.xls*"))
        )
        result["present"] = found == len(CIUS_XLS_YEARS)
        result["detail"] = f"{found}/{len(CIUS_XLS_YEARS)} publication years present"
    elif kind == "city_cache_set":
        found = sum(
            1
            for city in CITY_CACHE_DIRS
            if (repo_root / expected.replace("<city>", city)).is_dir()
        )
        result["present"] = found == len(CITY_CACHE_DIRS)
        result["detail"] = f"{found}/{len(CITY_CACHE_DIRS)} city caches present"
    else:
        raise ValueError(f"Unknown path_kind: {kind}")
    return result


def check_required_inputs(repo_root: Path) -> list[dict[str, object]]:
    return [_check_row(repo_root, row) for row in REQUIRED_INPUTS]


def write_required_input_manifest(*, repo_root: Path, out_path: Path) -> Path:
    rows = check_required_inputs(repo_root)
    missing_required = [
        r["expected_path"] for r in rows if r["required"] == "yes" and not r["present"]
    ]
    manifest = {
        "inputs": rows,
        "missing_required": missing_required,
        "all_required_present": not missing_required,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    return out_path
