from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crimerisk.required_inputs import REQUIRED_INPUTS


OFFENSE_ORDER = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]

OFFENSE_LABELS = {
    "murder": "Murder",
    "rape": "Rape",
    "robbery": "Robbery",
    "aggravated_assault": "Aggravated assault",
    "burglary": "Burglary",
    "larceny": "Larceny",
    "motor_vehicle_theft": "Motor vehicle theft",
    "personal": "Personal crime",
    "property": "Property crime",
    "total": "Total crime",
}

CITY_ORDER = [
    "Austin",
    "Baltimore",
    "Boston",
    "Chicago",
    "Denver",
    "Mesa",
    "Minneapolis",
    "New York",
    "Philadelphia",
    "San Francisco",
    "Seattle",
    "Washington, DC",
]

CITY_NAME_DISPLAY_MAP = {
    "Washington": "Washington, DC",
}

CITY_STATE_FIPS = {
    "Austin": "48",
    "Baltimore": "24",
    "Boston": "25",
    "Chicago": "17",
    "Denver": "08",
    "Mesa": "04",
    "Minneapolis": "27",
    "New York": "36",
    "Philadelphia": "42",
    "San Francisco": "06",
    "Seattle": "53",
    "Washington, DC": "11",
}


def _normalize_city_names(frame: pd.DataFrame, *, column: str = "city_name") -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    out = frame.copy()
    out[column] = out[column].replace(CITY_NAME_DISPLAY_MAP)
    return out

INPUT_INVENTORY_ROWS = [
    {
        "stage": "Reference",
        "dataset": "Canonical reference manifest",
        "years": "2018-2024 live build",
        "granularity": "agency and jurisdiction",
        "role": "frozen reference-layer inputs and resolved agency matching state",
        "path": "state/reference/input_manifest.json",
    },
    {
        "stage": "Observations",
        "dataset": "Jurisdiction-year observations",
        "years": "2018-2024",
        "granularity": "jurisdiction-year-offense",
        "role": "final observed offense rows after source arbitration",
        "path": "state/observations/jurisdiction_year_observations.parquet",
    },
    {
        "stage": "Controls",
        "dataset": "Promoted local publications",
        "years": "2018-2024 selected",
        "granularity": "jurisdiction-year-offense",
        "role": "explicitly promoted municipal annual publication rows",
        "path": "state/modeling/inputs/local_publication_annual.parquet",
    },
    {
        "stage": "Controls",
        "dataset": "Promoted state publications",
        "years": "2018-2024 selected",
        "granularity": "agency-year-offense",
        "role": "explicitly promoted annual state publication rows",
        "path": "state/modeling/inputs/state_publication_annual.parquet",
    },
    {
        "stage": "Validation",
        "dataset": "FBI CDE state estimates",
        "years": "1979-2024",
        "granularity": "state-year",
        "role": "state comparison and derivative calibration surface",
        "path": "state/controls/state_control_comparison.parquet",
    },
    {
        "stage": "Covariates",
        "dataset": "ACS block groups",
        "years": "2020-2024 5-year",
        "granularity": "block group",
        "role": "demographic and socioeconomic covariates",
        "path": "data/ACS-5yr-2020-2024/parsed/acs_block_groups.parquet",
    },
    {
        "stage": "Covariates",
        "dataset": "ACS tracts",
        "years": "2020-2024 5-year",
        "granularity": "tract",
        "role": "tract-level covariate support",
        "path": "data/ACS-5yr-2020-2024/parsed/acs_tracts_full.parquet",
    },
    {
        "stage": "Covariates",
        "dataset": "LODES WAC",
        "years": "2023",
        "granularity": "block group",
        "role": "employment and workplace structure covariates",
        "path": "data/LODES/parsed/lodes_wac_block_groups.parquet",
    },
    {
        "stage": "Covariates",
        "dataset": "Population estimates",
        "years": "2024",
        "granularity": "place/county/state",
        "role": "2024 population updates for controls and outputs",
        "path": "data/Census-PopEst-2020-2025/co-est2025-alldata.csv",
    },
    {
        "stage": "Covariates",
        "dataset": "Road metrics",
        "years": "2024 build",
        "granularity": "block group",
        "role": "street-network intensity covariates",
        "path": "data/roads/parsed/block_group_road_metrics.parquet",
    },
    {
        "stage": "Covariates",
        "dataset": "HPMS block-group metrics",
        "years": "2024",
        "granularity": "block group",
        "role": "highway and traffic-adjacent covariates",
        "path": "data/HPMS/parsed/block_group_hpms_2024.parquet",
    },
    {
        "stage": "Covariates",
        "dataset": "NCES EDGE education anchors",
        "years": "2024-2025",
        "granularity": "block group",
        "role": "school-anchor covariates",
        "path": "data/NCES-EDGE/parsed/block_group_education_anchors_2425.parquet",
    },
    {
        "stage": "Covariates",
        "dataset": "CMS hospital anchors",
        "years": "2024 build",
        "granularity": "block group",
        "role": "hospital-anchor covariates",
        "path": "data/CMS-Hospital-General-Info/parsed/block_group_hospital_anchors.parquet",
    },
    {
        "stage": "Covariates",
        "dataset": "NLCD land cover",
        "years": "2023",
        "granularity": "block group",
        "role": "land cover and imperviousness covariates",
        "path": "data/NLCD/parsed/block_group_nlcd_2023.parquet",
    },
    {
        "stage": "Validation",
        "dataset": "City incident share surface",
        "years": "2018-2024",
        "granularity": "block group within city",
        "role": "out-of-jurisdiction spatial validation surface",
        "path": "state/modeling/city_incident_share_surface.parquet",
    },
]

# The external raw-data contract (REQUIRED_INPUTS) lives in
# crimerisk.required_inputs and is imported above; required_inputs.csv and the
# package README acquisition directions are generated from it.

CONTIGUOUS_EXCLUDE = {"02", "15", "72"}
RELEASE_SCOPE_EXCLUDED_STATES = {"02": "Alaska", "15": "Hawaii", "72": "Puerto Rico"}
RELEASE_STATE_SCOPE_LABEL = "48 contiguous states plus District of Columbia"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prefix_relpath(rel_path: str, prefix: str) -> str:
    prefix = str(prefix).strip().strip("/")
    rel_path = str(rel_path).strip().lstrip("/")
    if not prefix:
        return rel_path
    return f"{prefix}/{rel_path}"


def _output_schema_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "column": "block_group_geoid",
            "surfaces": "block_group",
            "type": "string",
            "description": "2020 Census block-group GEOID. Join to 2020 TIGER/Line block-group geometry on GEOID.",
        },
        {
            "column": "tract_id",
            "surfaces": "block_group, tract",
            "type": "string",
            "description": "2020 Census tract GEOID. In block-group files this is the parent tract.",
        },
        {
            "column": "state_fips",
            "surfaces": "block_group, tract",
            "type": "string",
            "description": "Two-character state FIPS code. Release scope excludes 02, 15, 72, and territories other than DC.",
        },
        {
            "column": "population_2024",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Resident population. Used as the denominator for secondary resident per-100,000 rates and indexes.",
        },
        {
            "column": "eb_jurisdiction_id",
            "surfaces": "block_group, tract",
            "type": "string",
            "description": "Jurisdiction key retained for diagnostic empirical-Bayes prior metadata.",
        },
        {
            "column": "eb_jurisdiction_type",
            "surfaces": "block_group, tract",
            "type": "string",
            "description": "Jurisdiction type associated with eb_jurisdiction_id for diagnostic prior metadata.",
        },
        {
            "column": "daytime_population_jobs_proxy",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Jobs-based activity proxy derived as max(residents, residents plus workplace jobs minus resident workers).",
        },
        {
            "column": "landscan_day_pop",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "LandScan USA 2021 modeled daytime population aggregated to the geography; used only as a floor-lift for person-exposure primary denominators where larger.",
        },
        {
            "column": "exposure_proxy_2024",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Person-exposure primary denominator base: max(daytime_population_jobs_proxy, positive LandScan USA 2021 day population).",
        },
        {
            "column": "landscan_day_lifted_person_exposure",
            "surfaces": "block_group, tract",
            "type": "boolean",
            "description": "True when LandScan day population, not the jobs-based exposure proxy, lifts the person-exposure denominator.",
        },
        {
            "column": "households_total",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Occupied household component of the burglary premises denominator.",
        },
        {
            "column": "commercial_premises_total",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Legacy alias for the Overture consumer-destination POI component of the burglary denominator.",
        },
        {
            "column": "destination_poi_total",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Overture consumer-destination POI component of the burglary denominator.",
        },
        {
            "column": "lodes_retail_jobs",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "LODES CNS07 retail jobs component of the burglary denominator.",
        },
        {
            "column": "lodes_industrial_jobs",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "LODES CNS05+CNS06+CNS08 manufacturing, wholesale, and transport/warehouse jobs component of the burglary denominator.",
        },
        {
            "column": "burglary_premises_total",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Primary burglary denominator: households plus calibrated destination POI, retail-jobs, and industrial-jobs terms.",
        },
        {
            "column": "aggregate_vehicles_total",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Estimated aggregate vehicles available, used as the primary motor vehicle theft denominator.",
        },
        {
            "column": "eb_hard_min_denominator",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Legacy structural hard-minimum denominator retained for audit; current publication floors are also exposed in non_residential_household_floor and person_exposure_denominator_floor.",
        },
        {
            "column": "resident_secondary_denominator",
            "surfaces": "block_group, tract",
            "type": "float",
            "description": "Raw resident-population denominator used by all resident secondary index series.",
        },
        {
            "column": "resident_secondary_denominator_low_reliability",
            "surfaces": "block_group, tract",
            "type": "boolean",
            "description": "True when the raw resident denominator is zero; per-offense resident publication is reported by index_{offense}_resident_suppressed.",
        },
        {
            "column": "population_zero_with_positive_count",
            "surfaces": "block_group, tract",
            "type": "boolean",
            "description": "True when resident population is zero but modeled crime count is positive.",
        },
    ]
    for offense in OFFENSE_ORDER:
        label = OFFENSE_LABELS[offense]
        rows.extend(
            [
                {
                    "column": f"expected_count_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Modeled annual 2024 {label.lower()} count.",
                },
                {
                    "column": f"primary_denominator_type_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "string enum",
                    "description": f"Primary denominator family used for {label.lower()} indexes.",
                },
                {
                    "column": f"primary_denominator_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Raw primary denominator used for the published count-derived {label.lower()} primary rate and index.",
                },
                {
                    "column": f"primary_denominator_raw_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Alias of the raw primary denominator used for {label.lower()} point-rate construction.",
                },
                {
                    "column": f"primary_national_rate_per_100k_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic raw national {label.lower()} rate per 100,000 primary-denominator units, computed from the same count/denominator formula as the published rate.",
                },
                {
                    "column": f"primary_alpha_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Configured diagnostic empirical-Bayes alpha for {label.lower()}; not used to overwrite the published point.",
                },
                {
                    "column": f"primary_index_publishable_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"True when the {label.lower()} primary denominator is above the structural hard minimum and the count-derived published rate/index are emitted.",
                },
                {
                    "column": f"primary_index_suppressed_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"True when the {label.lower()} primary rate/index is suppressed because the primary denominator is at or below the structural hard minimum.",
                },
                {
                    "column": f"primary_zero_denominator_positive_count_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"True when the row has positive allocated {label.lower()} count and a structural primary denominator at or below the hard minimum.",
                },
                {
                    "column": f"raw_rate_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic direct {label.lower()} primary rate per 100,000 primary-denominator units, equal to the published rate on published rows.",
                },
                {
                    "column": f"diagnostic_eb_rate_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic empirical-Bayes {label.lower()} primary rate per 100,000 denominator units; not used in the published rate/index.",
                },
                {
                    "column": f"diagnostic_eb_national_rate_per_100k_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic empirical-Bayes national {label.lower()} rate per 100,000 denominator units.",
                },
                {
                    "column": f"diagnostic_eb_prior_rate_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic empirical-Bayes prior rate for {label.lower()} on the primary denominator scale.",
                },
                {
                    "column": f"diagnostic_eb_k_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic empirical-Bayes prior strength k for {label.lower()} on the primary denominator scale.",
                },
                {
                    "column": f"diagnostic_eb_observed_weight_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic share of the {label.lower()} empirical-Bayes estimate contributed by the row's own denominator: D / (D + k).",
                },
                {
                    "column": f"diagnostic_eb_prior_weight_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic share of the {label.lower()} empirical-Bayes estimate contributed by the nested prior: k / (D + k).",
                },
                {
                    "column": f"index_publishable_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"True when the {label.lower()} primary index is count-derived and published.",
                },
                {
                    "column": f"diagnostic_eb_low_denominator_flag_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"Diagnostic flag for {label.lower()} rows where the empirical-Bayes observed weight is below 0.50.",
                },
                {
                    "column": f"diagnostic_eb_heavy_shrinkage_flag_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"Diagnostic flag for {label.lower()} rows where the empirical-Bayes observed weight is below 0.20.",
                },
                {
                    "column": f"diagnostic_eb_extreme_shrinkage_flag_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"Diagnostic flag for {label.lower()} rows where the empirical-Bayes observed weight is below 0.05.",
                },
                {
                    "column": f"denominator_reason_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "string enum",
                    "description": f"Publication reason for the {label.lower()} primary denominator, usually publishable or zero_or_structural_denominator.",
                },
                {
                    "column": f"rate_{offense}_primary",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Published count-derived {label.lower()} primary rate per 100,000 primary-denominator units; null when suppressed.",
                },
                {
                    "column": f"index_{offense}_primary",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Published count-derived {label} primary index normalized to a national average of 100 using the same published rows; null when suppressed.",
                },
                {
                    "column": f"resident_national_rate_per_100k_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic raw national {label.lower()} resident rate per 100,000 residents.",
                },
                {
                    "column": f"diagnostic_resident_eb_rate_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic empirical-Bayes {label.lower()} resident rate per 100,000 residents; not used in the published resident rate/index.",
                },
                {
                    "column": f"resident_raw_rate_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic direct {label.lower()} secondary resident rate per 100,000 residents, equal to the published resident rate on published rows.",
                },
                {
                    "column": f"diagnostic_resident_eb_national_rate_per_100k_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic empirical-Bayes national {label.lower()} resident rate per 100,000 residents.",
                },
                {
                    "column": f"diagnostic_resident_eb_prior_rate_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic empirical-Bayes prior resident rate for {label.lower()} on the resident denominator scale.",
                },
                {
                    "column": f"diagnostic_resident_eb_k_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic empirical-Bayes prior strength k for {label.lower()} on the resident denominator scale.",
                },
                {
                    "column": f"diagnostic_resident_eb_observed_weight_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic share of the resident empirical-Bayes estimate contributed by the row's own denominator: D / (D + k).",
                },
                {
                    "column": f"diagnostic_resident_eb_prior_weight_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Diagnostic share of the resident empirical-Bayes estimate contributed by the nested prior: k / (D + k).",
                },
                {
                    "column": f"index_{offense}_resident_publishable",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"True when the {label.lower()} resident denominator is above the structural hard minimum and the count-derived resident rate/index are emitted.",
                },
                {
                    "column": f"diagnostic_resident_eb_low_denominator_flag_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"Diagnostic flag for {label.lower()} resident rows where the empirical-Bayes observed weight is below 0.50.",
                },
                {
                    "column": f"diagnostic_resident_eb_heavy_shrinkage_flag_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"Diagnostic flag for {label.lower()} resident rows where the empirical-Bayes observed weight is below 0.20.",
                },
                {
                    "column": f"diagnostic_resident_eb_extreme_shrinkage_flag_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"Diagnostic flag for {label.lower()} resident rows where the empirical-Bayes observed weight is below 0.05.",
                },
                {
                    "column": f"resident_denominator_reason_{offense}",
                    "surfaces": "block_group, tract",
                    "type": "string enum",
                    "description": f"Publication reason for the {label.lower()} resident denominator.",
                },
                {
                    "column": f"index_{offense}_resident_suppressed",
                    "surfaces": "block_group, tract",
                    "type": "boolean",
                    "description": f"True when the {label.lower()} resident secondary rate/index is suppressed because resident population is at or below the structural hard minimum.",
                },
                {
                    "column": f"rate_{offense}_resident",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Published count-derived {label.lower()} secondary resident rate per 100,000 residents; null when suppressed.",
                },
                {
                    "column": f"index_{offense}_resident",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Published count-derived {label} secondary resident index normalized to a national average of 100; null when suppressed.",
                },
            ]
        )
    for aggregate, label in [
        ("personal", "Personal crime"),
        ("property", "Property crime"),
        ("total", "Total crime"),
    ]:
        rows.extend(
            [
                {
                    "column": f"expected_count_{aggregate}",
                    "surfaces": "block_group, tract",
                    "type": "float",
                    "description": f"Modeled annual 2024 {label.lower()} count.",
                },
            ]
        )
    rows.extend(
        [
            {
                "column": "index_total_part1_resident",
                "surfaces": "block_group, tract",
                "type": "float",
                "description": "AGS-comparable total Part-I resident index, event-unweighted and count-derived from the seven expected counts.",
            },
            {
                "column": "index_personal_part1_resident",
                "surfaces": "block_group, tract",
                "type": "float",
                "description": "Personal-offense Part-I resident index using murder, rape, robbery, and aggravated assault expected counts.",
            },
            {
                "column": "index_property_part1_resident",
                "surfaces": "block_group, tract",
                "type": "float",
                "description": "Property-offense Part-I resident index using burglary, larceny, and motor vehicle theft expected counts.",
            },
            {
                "column": "index_total_primary_event_weighted",
                "surfaces": "block_group, tract",
                "type": "float",
                "description": "Seven-offense primary-index composite weighted by national expected-count offense shares.",
            },
            {
                "column": "index_total_equal_offense",
                "surfaces": "block_group, tract",
                "type": "float",
                "description": "Equal-offense mean of the seven primary offense indexes.",
            },
            {
                "column": "index_total_harm",
                "surfaces": "block_group, tract",
                "type": "float",
                "description": (
                    "Secondary count-derived total harm index (Cambridge Crime Harm Index shape): "
                    "sentencing-days severity weights applied to the seven expected counts and summed, "
                    "then normalized once by the person-exposure denominator and national harm rate -- "
                    "not an average of the seven per-offense indexes, so it is publishable wherever "
                    "person exposure is publishable."
                ),
            },
        ]
    )
    return rows


def _dtype_name(df: pd.DataFrame, column: str) -> str:
    dtype = df[column].dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    return "string"


def _default_output_column_description(column: str) -> str:
    for offense in OFFENSE_ORDER:
        label = OFFENSE_LABELS[offense].lower()
        if column == f"estimate_mode_{offense}":
            return f"Publication state for {label}, including count-derived, non-residential, special-use, insufficient-exposure, or denominator-invalid states."
        if column == f"primary_denominator_invalid_{offense}":
            return f"True when the {label} primary denominator is invalid for publication."
        if column == f"resident_denominator_invalid_{offense}":
            return f"True when the {label} resident denominator is invalid for publication."
        if column == f"direct_incident_support_flag_{offense}":
            return f"True when {label} has positive support from a promoted direct city incident surface."
        if column == f"direct_incident_support_count_{offense}":
            return f"Pooled direct city incident support count for {label} where available."
        if column.startswith(f"direct_incident_support_year") and column.endswith(offense):
            return f"Source-year support metadata for direct {label} incident evidence."
        if column == f"effective_numerator_support_{offense}":
            return f"Effective numerator support used by the {label} count-reliability layer."
        if column == f"numerator_support_source_{offense}":
            return f"Source family for the {label} effective numerator support."
        if column.startswith(f"rate_{offense}_primary_ci95_"):
            return f"Garwood/Poisson 95% interval bound for the {label} primary rate."
        if column.startswith(f"index_{offense}_primary_ci95_"):
            return f"Garwood/Poisson 95% interval bound or width metric for the {label} primary index."
        if column == f"reliability_tier_{offense}":
            return f"Count-reliability tier for the published {label} point estimate."
        if column == f"recommended_display_geography_{offense}":
            return f"Recommended display geography for {label} based on count support and interval width."
        if column == f"transient_exposure_likely_{offense}":
            return f"Flag for populated high-index {label} cells where transient/visitor exposure may still be undercounted."
        if column == f"crime_density_{offense}":
            return f"Expected {label} incidents per square mile, defined even when denominator-based rates are not published."
        if column == f"source_mode_{offense}":
            return f"Dominant {label} source mode: direct city incident, modeled transfer, or mixed."
        if column == f"source_mode_dominant_share_{offense}":
            return f"Component-count share of the dominant {label} source mode."
        if column == f"source_mode_mixed_{offense}":
            return f"True when no single {label} source mode dominates the component-count mix."
        if column.startswith("feed_") and column.endswith(offense):
            return f"Promoted city-feed quality/posterior parameter metadata for {label}; null outside direct-feed cells."
        if column == f"domain_overlap_score_{offense}":
            return f"Similarity of the {label} modeled-transfer cell to covered-city training domains, on a 0-1 scale."
        if column == f"confidence_tier_{offense}":
            return f"Combined source, denominator, support, and domain-overlap confidence tier for {label}."
        if column == f"confidence_reasons_{offense}":
            return f"Machine-readable reasons behind the {label} confidence tier."

    if column == "non_residential_household_floor":
        return "Household floor used to identify non-residential cells for publication suppression."
    if column == "person_exposure_denominator_floor":
        return "Person-exposure denominator floor used for primary and resident publication suppression."
    if column == "non_residential_flag":
        return "True when the row is non-residential under the household floor rule."
    if column == "special_use_tract_flag":
        return "True when the row belongs to a Census 98xx special-use tract."
    if column == "burglary_commercial_exposure_weight":
        return "Legacy alias for the calibrated destination-POI exposure weight used in the burglary primary denominator."
    if column == "burglary_destination_poi_exposure_weight":
        return "Calibrated destination-POI exposure weight used in the burglary primary denominator."
    if column == "burglary_retail_jobs_exposure_weight":
        return "Calibrated LODES retail-jobs exposure weight used in the burglary primary denominator."
    if column == "burglary_industrial_jobs_exposure_weight":
        return "Calibrated LODES industrial-jobs exposure weight used in the burglary primary denominator."
    if column == "landscan_day_pop":
        return "LandScan USA 2021 modeled daytime population aggregated to the geography."
    if column == "landscan_day_lifted_person_exposure":
        return "True when LandScan day population lifts the person-exposure primary denominator above the jobs-based proxy."
    if column == "land_area_sq_mi":
        return "TIGER land area in square miles, used for crime-density fields."
    if column == "crime_density_total":
        return "Total expected incidents per square mile."
    if column == "urban_stratum":
        return "Deterministic urban-form stratum used by confidence/source-mode interpretation."
    return "Output column present in the promoted release parquet; see STATE.md and OUTPUT_SCHEMA.md for field-family semantics."


def _actual_output_schema_rows(bg_output: pd.DataFrame, tract_output: pd.DataFrame) -> list[dict[str, str]]:
    manual_rows = {row["column"]: row for row in _output_schema_rows()}
    ordered_columns = list(bg_output.columns) + [col for col in tract_output.columns if col not in bg_output.columns]
    rows: list[dict[str, str]] = []
    for column in ordered_columns:
        in_bg = column in bg_output.columns
        in_tract = column in tract_output.columns
        source_df = bg_output if in_bg else tract_output
        manual = manual_rows.get(column, {})
        rows.append(
            {
                "column": column,
                "surfaces": ", ".join(
                    surface for surface, present in [("block_group", in_bg), ("tract", in_tract)] if present
                ),
                "type": _dtype_name(source_df, column),
                "description": manual.get("description") or _default_output_column_description(column),
            }
        )
    return rows


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _metric_fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def _pct_fmt(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{100.0 * value:.{digits}f}%"


def _count_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{int(round(float(value))):,}"


def _repo_relpath(path_value: object, repo_root: Path) -> str:
    if path_value is None:
        return ""
    path = Path(str(path_value))
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _resolve_repo_path(path_value: object, repo_root: Path) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = repo_root / path
    return path


def _weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if df.empty:
        return float("nan")
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    values = pd.to_numeric(df[value_col], errors="coerce")
    denom = float(weights.sum())
    if denom <= 0:
        return float(values.mean())
    return float((values * weights).sum() / denom)


def _df_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(
                lambda x: "" if pd.isna(x) else (f"{x:,.4f}" if abs(x) < 1000 else f"{x:,.0f}")
            )
    headers = [str(c) for c in display.columns]
    sep = ["---"] * len(headers)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(rows)


def _append_block(lines: list[str], text: str) -> None:
    block = textwrap.dedent(text).strip("\n")
    if not block:
        return
    lines.extend(block.splitlines())
    lines.append("")


def _save_bar(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    path: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    horizontal: bool = False,
    color: str = "#2C6E91",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = df[x_col].fillna("NA").astype(str)
    values = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    if horizontal:
        ax.barh(labels, values, color=color)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(labels, values, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_line(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_cols: list[str],
    path: Path,
    title: str,
    ylabel: str,
    labels: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = df[x_col]
    for idx, y_col in enumerate(y_cols):
        label = labels[idx] if labels is not None else y_col
        ax.plot(x, df[y_col], marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_hist(
    series: pd.Series,
    *,
    path: Path,
    title: str,
    xlabel: str,
    bins: int = 50,
    log_x: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=bins, color="#2C6E91", alpha=0.85)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    log_scale: bool = False,
    color_col: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    if color_col is None:
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=18, color="#2C6E91")
    else:
        categories = list(dict.fromkeys(df[color_col].astype(str).tolist()))
        cmap = plt.get_cmap("tab10", len(categories))
        for idx, category in enumerate(categories):
            mask = df[color_col].astype(str).eq(category)
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], alpha=0.6, s=18, color=cmap(idx), label=category)
        ax.legend(frameon=False, fontsize=8)
    max_val = float(max(df[x_col].max(), df[y_col].max()))
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#444444", linewidth=1)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_heatmap(
    pivot: pd.DataFrame,
    *,
    path: Path,
    title: str,
    cmap: str,
    cbar_label: str,
    fmt: str = ".3f",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([OFFENSE_LABELS.get(col, col) for col in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if not np.isnan(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _load_zip_geometries(zip_paths: Iterable[Path], *, geoid_col: str) -> gpd.GeoDataFrame:
    parts: list[gpd.GeoDataFrame] = []
    for zip_path in zip_paths:
        gdf = gpd.read_file(zip_path)
        if geoid_col not in gdf.columns:
            alt = "GEOID20" if "GEOID20" in gdf.columns else "GEOID"
            gdf = gdf.rename(columns={alt: geoid_col})
        parts.append(gdf[[geoid_col, "geometry"]].copy())
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs if parts else "EPSG:4269")


def _save_map(
    gdf: gpd.GeoDataFrame,
    *,
    value_col: str,
    path: Path,
    title: str,
    cmap: str = "viridis",
    q_clip: float = 0.99,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_gdf = gdf.copy()
    values = pd.to_numeric(plot_gdf[value_col], errors="coerce")
    vmax = float(values.quantile(q_clip))
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_gdf.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        linewidth=0.0,
        legend=True,
        vmin=0.0,
        vmax=vmax if vmax > 0 else None,
        missing_kwds={"color": "#f0f0f0"},
    )
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _parse_long_json_column(df: pd.DataFrame, *, json_col: str, base_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in df.to_dict(orient="records"):
        base = {col: record[col] for col in base_cols}
        parsed = json.loads(record[json_col])
        for item in parsed:
            rows.append(base | item)
    return pd.DataFrame(rows)


def _weighted_city_agg(df: pd.DataFrame, *, by: str, prefix: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in df.groupby(by, dropna=False):
        row = {by: key, "incident_total": float(group["incident_total"].sum()), "rows": int(len(group))}
        if prefix is None:
            row |= {
                "weighted_total_variation_distance_mean": _weighted_mean(group, "total_variation_distance", "incident_total"),
                "weighted_share_rmse_mean": _weighted_mean(group, "share_rmse", "incident_total"),
                "weighted_pearson_share_mean": _weighted_mean(group, "pearson_share", "incident_total"),
                "weighted_spearman_share_mean": _weighted_mean(group, "spearman_share", "incident_total"),
                "weighted_top10_capture_mean": _weighted_mean(group, "top_10pct_true_mass_in_model_top_10pct", "incident_total"),
            }
        else:
            row |= {
                f"{prefix}_weighted_total_variation_distance_mean": _weighted_mean(group, f"{prefix}_total_variation_distance", "incident_total"),
                f"{prefix}_weighted_share_rmse_mean": _weighted_mean(group, f"{prefix}_share_rmse", "incident_total"),
                f"{prefix}_weighted_pearson_share_mean": _weighted_mean(group, f"{prefix}_pearson_share", "incident_total"),
                f"{prefix}_weighted_spearman_share_mean": _weighted_mean(group, f"{prefix}_spearman_share", "incident_total"),
                f"{prefix}_weighted_top10_capture_mean": _weighted_mean(
                    group,
                    f"{prefix}_top_10pct_true_mass_in_model_top_10pct",
                    "incident_total",
                ),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_tables(repo_root: Path, *, package_repo_prefix: str = "") -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    state_dir = repo_root / "state"

    jurisdiction_master = pd.read_parquet(state_dir / "reference" / "jurisdiction_master.parquet")
    agency_master = pd.read_parquet(state_dir / "reference" / "agency_master.parquet")
    local_resolved = pd.read_parquet(state_dir / "reference" / "local_agency_resolved_full.parquet")
    nonlocal_resolved = pd.read_parquet(state_dir / "reference" / "nonlocal_agency_resolved_full.parquet")
    agency_crosswalk = pd.read_parquet(state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet")
    agency_obs = pd.read_parquet(state_dir / "observations" / "agency_year_observations.parquet")
    jurisdiction_obs = pd.read_parquet(state_dir / "observations" / "jurisdiction_year_observations.parquet")
    controls_2024 = pd.read_parquet(state_dir / "controls" / "jurisdiction_controls_2024.parquet")
    jurisdiction_year_estimates = pd.read_parquet(state_dir / "controls" / "jurisdiction_year_estimates.parquet")
    state_control = pd.read_parquet(state_dir / "controls" / "state_control_comparison.parquet")
    block_crosswalk = pd.read_parquet(state_dir / "geometry" / "block_to_jurisdiction_crosswalk.parquet")
    bg_crosswalk = pd.read_parquet(state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet")
    bg_output = pd.read_parquet(state_dir / "output" / "crimerisk_block_group_2024_ags_core.parquet")
    bg_output_fbi = pd.read_parquet(state_dir / "output" / "crimerisk_block_group_2024_fbi_calibrated.parquet")
    tract_output = pd.read_parquet(state_dir / "output" / "crimerisk_tract_2024_ags_core.parquet")
    tract_output_fbi = pd.read_parquet(state_dir / "output" / "crimerisk_tract_2024_fbi_calibrated.parquet")
    local_pub = pd.read_parquet(state_dir / "modeling" / "inputs" / "local_publication_annual.parquet")
    state_pub = pd.read_parquet(state_dir / "modeling" / "inputs" / "state_publication_annual.parquet")
    city_share_surface = _normalize_city_names(pd.read_parquet(state_dir / "modeling" / "city_incident_share_surface.parquet"))
    city_share_detail = _normalize_city_names(pd.read_parquet(state_dir / "modeling" / "city_share_benchmark_2024.parquet"))
    city_residual_detail = _normalize_city_names(pd.read_parquet(state_dir / "modeling" / "city_residual_benchmark_2024.parquet"))
    benchmark_json = json.loads((state_dir / "modeling" / "jurisdiction_model_benchmark_2024.json").read_text())
    benchmark_detail = pd.read_parquet(state_dir / "modeling" / "jurisdiction_model_benchmark_2024.parquet")
    qa_summary = json.loads((state_dir / "qa" / "build_qa_summary.json").read_text())

    tables: dict[str, pd.DataFrame] = {}

    tables["input_inventory"] = pd.DataFrame(INPUT_INVENTORY_ROWS)
    tables["input_inventory"]["path"] = tables["input_inventory"]["path"].map(
        lambda path: _prefix_relpath(str(path), package_repo_prefix)
    )
    tables["required_inputs"] = pd.DataFrame(REQUIRED_INPUTS).drop(columns=["path_kind"])
    tables["required_inputs"]["expected_path"] = tables["required_inputs"]["expected_path"].map(
        lambda path: _prefix_relpath(str(path), package_repo_prefix)
    )
    tables["output_schema"] = pd.DataFrame(_actual_output_schema_rows(bg_output, tract_output))

    tables["reference_summary"] = pd.DataFrame(
        [
            {
                "agencies": int(len(agency_master)),
                "jurisdictions": int(len(jurisdiction_master)),
                "contracted_places": int(pd.to_numeric(jurisdiction_master["is_contracted_place"], errors="coerce").fillna(False).astype(bool).sum()),
                "manual_review_flagged_jurisdictions": int(pd.to_numeric(jurisdiction_master["manual_review_flag"], errors="coerce").fillna(False).astype(bool).sum()),
                "agency_rows_with_srs": int(pd.to_numeric(agency_master["source_presence_srs"], errors="coerce").fillna(False).astype(bool).sum()),
                "agency_rows_with_nibrs": int(pd.to_numeric(agency_master["source_presence_nibrs"], errors="coerce").fillna(False).astype(bool).sum()),
                "agency_rows_with_both_srs_and_nibrs": int(
                    (
                        pd.to_numeric(agency_master["source_presence_srs"], errors="coerce").fillna(False).astype(bool)
                        & pd.to_numeric(agency_master["source_presence_nibrs"], errors="coerce").fillna(False).astype(bool)
                    ).sum()
                ),
            }
        ]
    )

    jurisdiction_type_counts = jurisdiction_master["jurisdiction_type"].value_counts().rename_axis("jurisdiction_type").reset_index(name="jurisdiction_count")
    jurisdiction_type_counts["jurisdiction_share"] = jurisdiction_type_counts["jurisdiction_count"] / int(len(jurisdiction_master))
    tables["reference_jurisdiction_type_counts"] = jurisdiction_type_counts

    source_presence_rows = []
    presence_map = {
        "srs_only": (
            pd.to_numeric(agency_master["source_presence_srs"], errors="coerce").fillna(False).astype(bool)
            & ~pd.to_numeric(agency_master["source_presence_nibrs"], errors="coerce").fillna(False).astype(bool)
        ),
        "nibrs_only": (
            ~pd.to_numeric(agency_master["source_presence_srs"], errors="coerce").fillna(False).astype(bool)
            & pd.to_numeric(agency_master["source_presence_nibrs"], errors="coerce").fillna(False).astype(bool)
        ),
        "both": (
            pd.to_numeric(agency_master["source_presence_srs"], errors="coerce").fillna(False).astype(bool)
            & pd.to_numeric(agency_master["source_presence_nibrs"], errors="coerce").fillna(False).astype(bool)
        ),
        "neither": (
            ~pd.to_numeric(agency_master["source_presence_srs"], errors="coerce").fillna(False).astype(bool)
            & ~pd.to_numeric(agency_master["source_presence_nibrs"], errors="coerce").fillna(False).astype(bool)
        ),
    }
    for label, mask in presence_map.items():
        count = int(mask.sum())
        source_presence_rows.append(
            {
                "source_presence_group": label,
                "agency_count": count,
                "agency_share": count / int(len(agency_master)),
            }
        )
    tables["reference_agency_source_presence"] = pd.DataFrame(source_presence_rows)
    tables["reference_local_resolution_geo_summary"] = (
        local_resolved.groupby("resolved_geo_type", dropna=False)
        .agg(agency_count=("ori9", "nunique"))
        .reset_index()
        .sort_values("agency_count", ascending=False, kind="mergesort")
    )
    tables["reference_local_resolution_source_summary"] = (
        local_resolved.groupby("resolution_source", dropna=False)
        .agg(agency_count=("ori9", "nunique"))
        .reset_index()
        .sort_values("agency_count", ascending=False, kind="mergesort")
    )
    tables["reference_nonlocal_agency_type_summary"] = (
        nonlocal_resolved.groupby("agency_type_norm", dropna=False)
        .agg(agency_count=("ori9", "nunique"))
        .reset_index()
        .sort_values("agency_count", ascending=False, kind="mergesort")
    )
    tables["reference_nonlocal_overlap_summary"] = (
        nonlocal_resolved.groupby("final_overlap_subtype", dropna=False)
        .agg(agency_count=("ori9", "nunique"))
        .reset_index()
        .sort_values("agency_count", ascending=False, kind="mergesort")
    )
    tables["reference_crosswalk_relationship_summary"] = (
        agency_crosswalk.groupby("relationship_type", dropna=False)
        .agg(
            row_count=("ori", "size"),
            agency_count=("ori", "nunique"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
        )
        .reset_index()
        .sort_values("row_count", ascending=False, kind="mergesort")
    )
    reference_input_rows = []
    for surface, rel_path, role in [
        ("provisional_local_agency_matches", Path("state/reference/inputs/provisional_local_agency_matches.parquet"), "promoted provisional local-resolution seed surface"),
        ("local_queue_resolved_final", Path("state/reference/inputs/local_queue_resolved_final.parquet"), "promoted analyst-reviewed local tail decisions"),
        ("nonmunicipal_special_resolved_final", Path("state/reference/inputs/nonmunicipal_special_resolved_final.parquet"), "promoted analyst-reviewed special nonlocal decisions"),
        ("nonmunicipal_auto_defaults", Path("state/reference/inputs/nonmunicipal_auto_defaults.parquet"), "promoted deterministic nonlocal defaults"),
    ]:
        frame = pd.read_parquet(repo_root / rel_path)
        reference_input_rows.append(
            {
                "surface": surface,
                "rows": int(len(frame)),
                "path": _prefix_relpath(str(rel_path), package_repo_prefix),
                "role": role,
            }
        )
    tables["reference_input_surface_summary"] = pd.DataFrame(reference_input_rows)
    manifest = json.loads((state_dir / "reference" / "input_manifest.json").read_text())
    tables["reference_manifest_summary"] = pd.DataFrame(
        [
            {
                "manifest_entry_count": int(len(manifest.get("inputs", []))),
                "required_missing_count": int(len(manifest.get("missing_required", []))),
                "contract_row_count_in_code": int(len(REQUIRED_INPUTS)),
                "all_required_present": bool(manifest.get("all_required_present", False)),
            }
        ]
    )

    tables["observation_stage_summary"] = pd.DataFrame(
        [
            {
                "agency_year_observation_rows": int(len(agency_obs)),
                "agency_count": int(agency_obs["ori9"].nunique()),
                "agency_years": int(agency_obs[["ori9", "year"]].drop_duplicates().shape[0]),
                "jurisdiction_year_observation_rows": int(len(jurisdiction_obs)),
                "jurisdiction_count": int(jurisdiction_obs["jurisdiction_id"].nunique()),
                "jurisdiction_years": int(jurisdiction_obs[["jurisdiction_id", "year"]].drop_duplicates().shape[0]),
            }
        ]
    )
    tables["agency_observation_source_by_year"] = (
        agency_obs.groupby(["year", "source"], dropna=False)
        .agg(
            row_count=("ori9", "size"),
            agency_count=("ori9", "nunique"),
            observed_count_total=("count", "sum"),
        )
        .reset_index()
        .sort_values(["year", "source"], kind="mergesort")
    )
    tables["agency_observation_source_2024"] = (
        agency_obs.loc[agency_obs["year"].eq(2024)]
        .groupby("source", dropna=False)
        .agg(
            row_count=("ori9", "size"),
            agency_count=("ori9", "nunique"),
            observed_count_total=("count", "sum"),
        )
        .reset_index()
        .sort_values("observed_count_total", ascending=False, kind="mergesort")
    )
    tables["jurisdiction_observation_source_2024"] = (
        jurisdiction_obs.loc[jurisdiction_obs["year"].eq(2024)]
        .groupby("source", dropna=False)
        .agg(
            row_count=("jurisdiction_id", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            observed_count_total=("observed_count", "sum"),
        )
        .reset_index()
        .sort_values("observed_count_total", ascending=False, kind="mergesort")
    )

    obs_source_mix = (
        jurisdiction_obs.groupby(["year", "source_origin"], dropna=False)
        .agg(
            row_count=("observed_count", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            observed_count_total=("observed_count", "sum"),
        )
        .reset_index()
        .sort_values(["year", "source_origin"], kind="mergesort")
    )
    tables["observation_source_mix_by_year"] = obs_source_mix

    obs_source_detail = (
        jurisdiction_obs.groupby(["year", "source", "raw_data_source"], dropna=False)
        .agg(
            row_count=("observed_count", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            observed_count_total=("observed_count", "sum"),
        )
        .reset_index()
        .sort_values(["year", "source", "raw_data_source"], kind="mergesort")
    )
    tables["observation_source_detail_by_year"] = obs_source_detail

    observed_share_rows = []
    for year, group in jurisdiction_year_estimates.groupby("year", dropna=False):
        usable = pd.to_numeric(group["usable_as_observed"], errors="coerce").fillna(False).astype(bool)
        est = pd.to_numeric(group["estimated_count"], errors="coerce").fillna(0.0)
        observed_share_rows.append(
            {
                "year": int(year),
                "row_count": int(len(group)),
                "observed_rows": int(usable.sum()),
                "observed_row_share": float(usable.mean()),
                "estimated_count_total": float(est.sum()),
                "observed_estimated_count_total": float(est[usable].sum()),
                "observed_estimated_count_share": float(est[usable].sum() / est.sum()) if float(est.sum()) > 0 else float("nan"),
            }
        )
    tables["observed_share_by_year"] = pd.DataFrame(observed_share_rows).sort_values("year", kind="mergesort")

    observed_share_type_rows = []
    for (year, jurisdiction_type), group in jurisdiction_year_estimates.groupby(["year", "jurisdiction_type"], dropna=False):
        usable = pd.to_numeric(group["usable_as_observed"], errors="coerce").fillna(False).astype(bool)
        est = pd.to_numeric(group["estimated_count"], errors="coerce").fillna(0.0)
        observed_share_type_rows.append(
            {
                "year": int(year),
                "jurisdiction_type": str(jurisdiction_type),
                "row_count": int(len(group)),
                "observed_rows": int(usable.sum()),
                "observed_row_share": float(usable.mean()),
                "estimated_count_total": float(est.sum()),
                "observed_estimated_count_share": float(est[usable].sum() / est.sum()) if float(est.sum()) > 0 else float("nan"),
            }
        )
    tables["observed_share_by_year_and_type"] = pd.DataFrame(observed_share_type_rows).sort_values(
        ["year", "jurisdiction_type"], kind="mergesort"
    )

    tables["promoted_local_publication_summary"] = (
        local_pub.groupby(["case_key", "jurisdiction_name", "state_abbr", "raw_data_source"], dropna=False)
        .agg(
            offense_rows=("offense", "size"),
            offense_count=("offense", "nunique"),
            count_total=("count", "sum"),
        )
        .reset_index()
        .sort_values(["state_abbr", "jurisdiction_name"], kind="mergesort")
    )

    tables["promoted_state_publication_summary"] = (
        state_pub.groupby(["state_abbr", "raw_data_source"], dropna=False)
        .agg(
            agency_count=("ori9", "nunique"),
            offense_rows=("offense", "size"),
            count_total=("count", "sum"),
        )
        .reset_index()
        .sort_values(["state_abbr", "raw_data_source"], kind="mergesort")
    )
    reporting_regimes_2024 = (
        pd.read_parquet(state_dir / "modeling" / "agency_year_reporting_regimes.parquet")
        .loc[lambda df: df["year"].eq(2024)]
        .groupby("reporting_regime", dropna=False)
        .agg(row_count=("ori9", "size"), agency_count=("ori9", "nunique"))
        .reset_index()
        .sort_values("row_count", ascending=False, kind="mergesort")
    )
    tables["agency_reporting_regime_2024"] = reporting_regimes_2024
    tables["jurisdiction_target_source_2024"] = (
        controls_2024.groupby("estimate_source", dropna=False)
        .agg(
            row_count=("jurisdiction_id", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            estimated_count_total=("adjusted_count_ags_core", "sum"),
        )
        .reset_index()
        .sort_values("estimated_count_total", ascending=False, kind="mergesort")
    )
    tables["jurisdiction_fill_leaders_2024"] = (
        controls_2024.loc[
            pd.to_numeric(controls_2024["current_year_fill_count"], errors="coerce").fillna(0.0) > 0,
            ["jurisdiction_name", "state_abbr", "offense", "adjusted_count_ags_core", "current_year_fill_count", "estimate_source"],
        ]
        .sort_values("current_year_fill_count", ascending=False, kind="mergesort")
        .head(15)
        .reset_index(drop=True)
    )

    preferred_source = (
        controls_2024.groupby("preferred_source", dropna=False)
        .agg(
            row_count=("jurisdiction_id", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            adjusted_count_total=("adjusted_count_ags_core", "sum"),
            estimated_count_total=("estimated_count_ags_core", "sum"),
        )
        .reset_index()
        .sort_values("adjusted_count_total", ascending=False, kind="mergesort")
    )
    preferred_source["row_share"] = preferred_source["row_count"] / int(len(controls_2024))
    preferred_source["adjusted_count_share"] = preferred_source["adjusted_count_total"] / float(
        pd.to_numeric(controls_2024["adjusted_count_ags_core"], errors="coerce").sum()
    )
    tables["controls_preferred_source_2024"] = preferred_source

    preferred_origin = (
        controls_2024.groupby("preferred_source_origin", dropna=False)
        .agg(
            row_count=("jurisdiction_id", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            adjusted_count_total=("adjusted_count_ags_core", "sum"),
        )
        .reset_index()
        .sort_values("adjusted_count_total", ascending=False, kind="mergesort")
    )
    preferred_origin["row_share"] = preferred_origin["row_count"] / int(len(controls_2024))
    preferred_origin["adjusted_count_share"] = preferred_origin["adjusted_count_total"] / float(
        pd.to_numeric(controls_2024["adjusted_count_ags_core"], errors="coerce").sum()
    )
    tables["controls_preferred_source_origin_2024"] = preferred_origin

    estimate_source = (
        controls_2024.groupby("estimate_source", dropna=False)
        .agg(
            row_count=("jurisdiction_id", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            adjusted_count_total=("adjusted_count_ags_core", "sum"),
        )
        .reset_index()
        .sort_values("adjusted_count_total", ascending=False, kind="mergesort")
    )
    estimate_source["row_share"] = estimate_source["row_count"] / int(len(controls_2024))
    estimate_source["adjusted_count_share"] = estimate_source["adjusted_count_total"] / float(
        pd.to_numeric(controls_2024["adjusted_count_ags_core"], errors="coerce").sum()
    )
    tables["controls_estimate_source_2024"] = estimate_source

    reporting_regime = (
        controls_2024.groupby("dominant_reporting_regime", dropna=False)
        .agg(
            row_count=("jurisdiction_id", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            adjusted_count_total=("adjusted_count_ags_core", "sum"),
        )
        .reset_index()
        .sort_values("adjusted_count_total", ascending=False, kind="mergesort")
    )
    reporting_regime["row_share"] = reporting_regime["row_count"] / int(len(controls_2024))
    reporting_regime["adjusted_count_share"] = reporting_regime["adjusted_count_total"] / float(
        pd.to_numeric(controls_2024["adjusted_count_ags_core"], errors="coerce").sum()
    )
    tables["controls_reporting_regime_2024"] = reporting_regime

    jurisdiction_type_2024 = (
        controls_2024.groupby("jurisdiction_type", dropna=False)
        .agg(
            row_count=("jurisdiction_id", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
            adjusted_count_total=("adjusted_count_ags_core", "sum"),
        )
        .reset_index()
        .sort_values("adjusted_count_total", ascending=False, kind="mergesort")
    )
    jurisdiction_type_2024["row_share"] = jurisdiction_type_2024["row_count"] / int(len(controls_2024))
    jurisdiction_type_2024["adjusted_count_share"] = jurisdiction_type_2024["adjusted_count_total"] / float(
        pd.to_numeric(controls_2024["adjusted_count_ags_core"], errors="coerce").sum()
    )
    tables["controls_jurisdiction_type_2024"] = jurisdiction_type_2024

    state_control_summary = (
        state_control.groupby("offense", dropna=False)
        .agg(
            state_count=("state_fips", "nunique"),
            population=("population", "sum"),
            ags_core_reported_total=("ags_core_reported_total", "sum"),
            ags_core_adjusted_total=("ags_core_adjusted_total", "sum"),
            fbi_cde_estimated_total=("fbi_cde_estimated_total", "sum"),
            partial_reporting_uplift_total=("partial_reporting_uplift_total", "sum"),
            current_year_fill_total=("current_year_fill_total", "sum"),
        )
        .reset_index()
        .sort_values("offense", key=lambda s: s.map({o: i for i, o in enumerate(OFFENSE_ORDER)}), kind="mergesort")
    )
    state_control_summary["reported_to_cde_ratio"] = (
        state_control_summary["ags_core_reported_total"] / state_control_summary["fbi_cde_estimated_total"]
    )
    state_control_summary["adjusted_to_cde_ratio"] = (
        state_control_summary["ags_core_adjusted_total"] / state_control_summary["fbi_cde_estimated_total"]
    )
    tables["state_control_comparison_summary"] = state_control_summary
    tables["state_control_comparison_detail"] = state_control.sort_values(["state_fips", "offense"], kind="mergesort")
    tables["controls_adjustment_summary_2024"] = pd.DataFrame(
        [
            {
                "adjustment_component": "partial_reporting_uplift",
                "row_count": int(pd.to_numeric(controls_2024["needs_partial_reporting_uplift"], errors="coerce").fillna(False).astype(bool).sum()),
                "adjusted_count_total": float(pd.to_numeric(controls_2024["partial_reporting_uplift_count"], errors="coerce").fillna(0.0).sum()),
            },
            {
                "adjustment_component": "current_year_fill",
                "row_count": int(pd.to_numeric(controls_2024["needs_current_year_fill"], errors="coerce").fillna(False).astype(bool).sum()),
                "adjusted_count_total": float(pd.to_numeric(controls_2024["current_year_fill_count"], errors="coerce").fillna(0.0).sum()),
            },
        ]
    )

    bg_share_sums = (
        bg_crosswalk.groupby("block_group_geoid", dropna=False)["allocation_share"]
        .sum()
        .reset_index(name="allocation_share_sum")
    )
    tables["geometry_block_assignment_summary"] = (
        block_crosswalk.groupby("assignment_method", dropna=False)
        .agg(
            row_count=("block_geoid", "size"),
            jurisdiction_count=("jurisdiction_id", "nunique"),
        )
        .reset_index()
        .sort_values("row_count", ascending=False, kind="mergesort")
    )
    tables["geometry_crosswalk_summary"] = pd.DataFrame(
        [
            {
                "block_crosswalk_rows": int(len(block_crosswalk)),
                "block_group_crosswalk_rows": int(len(bg_crosswalk)),
                "block_group_count": int(bg_share_sums["block_group_geoid"].nunique()),
                "allocation_sum_lt_1_count": int((bg_share_sums["allocation_share_sum"] < 1.0 - 1e-9).sum()),
                "allocation_sum_gt_1_count": int((bg_share_sums["allocation_share_sum"] > 1.0 + 1e-9).sum()),
                "allocation_sum_ne_1_count": int(((bg_share_sums["allocation_share_sum"] - 1.0).abs() > 1e-9).sum()),
                "max_allocation_share_sum": float(bg_share_sums["allocation_share_sum"].max()),
            }
        ]
    )
    bg_share_sums = bg_share_sums.merge(
        bg_output[["block_group_geoid", "expected_count_total"]],
        on="block_group_geoid",
        how="left",
    )
    tables["geometry_allocation_bucket_summary"] = pd.DataFrame(
        [
            {
                "allocation_sum_bucket": "lt_1",
                "block_group_count": int((bg_share_sums["allocation_share_sum"] < 1.0 - 1e-9).sum()),
                "output_expected_count_total": float(pd.to_numeric(bg_share_sums.loc[bg_share_sums["allocation_share_sum"] < 1.0 - 1e-9, "expected_count_total"], errors="coerce").fillna(0.0).sum()),
            },
            {
                "allocation_sum_bucket": "eq_1",
                "block_group_count": int(((bg_share_sums["allocation_share_sum"] - 1.0).abs() <= 1e-9).sum()),
                "output_expected_count_total": float(pd.to_numeric(bg_share_sums.loc[(bg_share_sums["allocation_share_sum"] - 1.0).abs() <= 1e-9, "expected_count_total"], errors="coerce").fillna(0.0).sum()),
            },
            {
                "allocation_sum_bucket": "gt_1",
                "block_group_count": int((bg_share_sums["allocation_share_sum"] > 1.0 + 1e-9).sum()),
                "output_expected_count_total": float(pd.to_numeric(bg_share_sums.loc[bg_share_sums["allocation_share_sum"] > 1.0 + 1e-9, "expected_count_total"], errors="coerce").fillna(0.0).sum()),
            },
        ]
    )
    tables["geometry_allocation_bucket_summary"]["output_expected_count_share"] = tables["geometry_allocation_bucket_summary"]["output_expected_count_total"] / float(
        pd.to_numeric(bg_output["expected_count_total"], errors="coerce").fillna(0.0).sum()
    )

    output_offense_rows = []
    total_population = float(pd.to_numeric(bg_output["population_2024"], errors="coerce").sum())
    for offense in OFFENSE_ORDER + ["personal", "property", "total"]:
        total_count = float(pd.to_numeric(bg_output[f"expected_count_{offense}"], errors="coerce").sum())
        total_count_fbi = float(pd.to_numeric(bg_output_fbi[f"expected_count_{offense}"], errors="coerce").sum())
        output_offense_rows.append(
            {
                "offense": offense,
                "ags_core_total_expected_count": total_count,
                "ags_core_rate_per_100k": (100000.0 * total_count / total_population) if total_population > 0 else float("nan"),
                "fbi_calibrated_total_expected_count": total_count_fbi,
                "fbi_calibrated_rate_per_100k": (100000.0 * total_count_fbi / total_population) if total_population > 0 else float("nan"),
                "fbi_minus_ags_expected_count": total_count_fbi - total_count,
            }
        )
    tables["output_offense_summary"] = pd.DataFrame(output_offense_rows)

    state_output_summary = (
        bg_output.groupby("state_fips", dropna=False)
        .agg(
            block_group_count=("block_group_geoid", "size"),
            population_2024=("population_2024", "sum"),
            expected_count_total=("expected_count_total", "sum"),
            expected_count_personal=("expected_count_personal", "sum"),
            expected_count_property=("expected_count_property", "sum"),
            mean_index_total_primary_event_weighted=("index_total_primary_event_weighted", "mean"),
            p95_index_total_primary_event_weighted=("index_total_primary_event_weighted", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.95))),
            mean_index_total_part1_resident=("index_total_part1_resident", "mean"),
            p95_index_total_part1_resident=("index_total_part1_resident", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.95))),
        )
        .reset_index()
    )
    state_output_summary["resident_expected_count_total_per_100k_context"] = (
        100000.0 * state_output_summary["expected_count_total"] / state_output_summary["population_2024"]
    )
    tables["output_state_summary"] = state_output_summary.sort_values("expected_count_total", ascending=False, kind="mergesort")
    released_states = sorted(bg_output["state_fips"].astype(str).str.zfill(2).unique().tolist())
    excluded_present = [state for state in sorted(RELEASE_SCOPE_EXCLUDED_STATES) if state in set(released_states)]
    tables["output_release_coverage"] = pd.DataFrame(
        [
            {
                "scope": RELEASE_STATE_SCOPE_LABEL,
                "released_state_count": int(len(released_states)),
                "excluded_state_fips": ", ".join(sorted(RELEASE_SCOPE_EXCLUDED_STATES)),
                "excluded_state_names": ", ".join(RELEASE_SCOPE_EXCLUDED_STATES[state] for state in sorted(RELEASE_SCOPE_EXCLUDED_STATES)),
                "excluded_states_present_in_outputs": ", ".join(excluded_present),
                "block_group_rows": int(len(bg_output)),
                "tract_rows": int(len(tract_output)),
                "population_2024": float(pd.to_numeric(bg_output["population_2024"], errors="coerce").fillna(0.0).sum()),
                "expected_count_total": float(pd.to_numeric(bg_output["expected_count_total"], errors="coerce").fillna(0.0).sum()),
            }
        ]
    )

    tables["top_block_groups_primary_event_weighted_index"] = (
        bg_output.sort_values("index_total_primary_event_weighted", ascending=False, kind="mergesort")[
            [
                "block_group_geoid",
                "state_fips",
                "tract_id",
                "population_2024",
                "expected_count_total",
                "index_total_primary_event_weighted",
                "index_total_part1_resident",
                "index_total_equal_offense",
            ]
        ]
        .head(50)
        .reset_index(drop=True)
    )
    tables["top_block_groups_total_equal_offense_index"] = (
        bg_output.sort_values("index_total_equal_offense", ascending=False, kind="mergesort")[
            [
                "block_group_geoid",
                "state_fips",
                "tract_id",
                "population_2024",
                "expected_count_total",
                "index_total_part1_resident",
                "index_total_equal_offense",
            ]
        ]
        .head(50)
        .reset_index(drop=True)
    )
    tables["top_tracts_primary_event_weighted_index"] = (
        tract_output.sort_values("index_total_primary_event_weighted", ascending=False, kind="mergesort")[
            ["tract_id", "state_fips", "population_2024", "expected_count_total", "index_total_primary_event_weighted", "index_total_part1_resident", "index_total_equal_offense"]
        ]
        .head(50)
        .reset_index(drop=True)
    )
    tables["top_tracts_total_equal_offense_index"] = (
        tract_output.sort_values("index_total_equal_offense", ascending=False, kind="mergesort")[
            ["tract_id", "state_fips", "population_2024", "expected_count_total", "index_total_part1_resident", "index_total_equal_offense"]
        ]
        .head(50)
        .reset_index(drop=True)
    )

    quantile_rows = []
    for surface_name, df, value_cols in [
        ("block_group", bg_output, ["index_total_primary_event_weighted", "index_total_part1_resident", "index_total_equal_offense", "expected_count_total"]),
        ("tract", tract_output, ["index_total_primary_event_weighted", "index_total_part1_resident", "index_total_equal_offense", "expected_count_total"]),
    ]:
        for value_col in value_cols:
            series = pd.to_numeric(df[value_col], errors="coerce")
            quantile_rows.append(
                {
                    "surface": surface_name,
                    "metric": value_col,
                    "q50": float(series.quantile(0.50)),
                    "q75": float(series.quantile(0.75)),
                    "q90": float(series.quantile(0.90)),
                    "q95": float(series.quantile(0.95)),
                    "q99": float(series.quantile(0.99)),
                    "max": float(series.max()),
                }
            )
    tables["output_distribution_quantiles"] = pd.DataFrame(quantile_rows)

    benchmark_overview = pd.DataFrame(
        [
            {
                "year": int(benchmark_json["year"]),
                "model_family": benchmark_json["model_family"],
                "extra_bg_feature_count": int(benchmark_json["extra_bg_feature_count"]),
                "min_training_population": int(benchmark_json["min_training_population"]),
                "overall_cv_r2_log_rate_mean": float(benchmark_json["overall_cv_r2_log_rate_mean"]),
                "overall_cv_r2_rate_mean": float(benchmark_json["overall_cv_r2_rate_mean"]),
                "overall_cv_rmse_log_rate_mean": float(benchmark_json["overall_cv_rmse_log_rate_mean"]),
                "overall_train_r2_log_rate_mean": float(benchmark_json["overall_train_r2_log_rate_mean"]),
                "feature_count": int(benchmark_detail["feature_count"].max()),
            }
        ]
    )
    tables["benchmark_overview"] = benchmark_overview

    benchmark_detail = benchmark_detail.copy()
    benchmark_detail["offense_label"] = benchmark_detail["offense"].map(OFFENSE_LABELS)
    tables["benchmark_offense_summary"] = benchmark_detail[
        [
            "offense",
            "offense_label",
            "training_rows",
            "feature_count",
            "train_r2_log_rate",
            "cv_r2_log_rate",
            "cv_r2_rate",
            "cv_rmse_log_rate",
        ]
    ].sort_values("offense", key=lambda s: s.map({o: i for i, o in enumerate(OFFENSE_ORDER)}), kind="mergesort")

    calibration_bins = _parse_long_json_column(
        benchmark_detail,
        json_col="calibration_json",
        base_cols=["offense"],
    )
    calibration_bins["offense_label"] = calibration_bins["offense"].map(OFFENSE_LABELS)
    tables["benchmark_calibration_bins"] = calibration_bins.sort_values(
        ["offense", "mean_rate_pred"], kind="mergesort"
    )

    residual_by_state = _parse_long_json_column(
        benchmark_detail,
        json_col="residual_by_state_json",
        base_cols=["offense"],
    )
    residual_by_state["offense_label"] = residual_by_state["offense"].map(OFFENSE_LABELS)
    tables["benchmark_residual_by_state"] = residual_by_state.sort_values(
        ["offense", "mean_abs_resid_log_rate"], ascending=[True, False], kind="mergesort"
    )

    residual_by_pop_band = _parse_long_json_column(
        benchmark_detail,
        json_col="residual_by_pop_band_json",
        base_cols=["offense"],
    )
    residual_by_pop_band["offense_label"] = residual_by_pop_band["offense"].map(OFFENSE_LABELS)
    tables["benchmark_residual_by_pop_band"] = residual_by_pop_band.sort_values(
        ["offense", "pop_band"], kind="mergesort"
    )

    city_truth_inventory = (
        city_share_surface.groupby(["city_name", "year", "offense"], dropna=False)
        .agg(
            block_group_count=("block_group_geoid", "nunique"),
            incident_total=("incident_count", "sum"),
            mean_share=("share_within_city", "mean"),
        )
        .reset_index()
        .sort_values(["city_name", "year", "offense"], kind="mergesort")
    )
    tables["city_truth_inventory"] = city_truth_inventory
    city_packet_status_path = state_dir / "review" / "packets" / "city" / "city_packet_status_summary.csv"
    if city_packet_status_path.exists():
        city_packet_status = pd.read_csv(city_packet_status_path)
        keep_cols = [
            "city_key",
            "production_ready",
            "city_share_integration_status",
            "offense_selective_ready_count",
            "offense_selective_ready_offenses",
            "packet_recommended_disposition",
        ]
        tables["city_packet_status_summary"] = city_packet_status[
            [col for col in keep_cols if col in city_packet_status.columns]
        ].sort_values("city_key", kind="mergesort")
    source_packet_status_path = state_dir / "review" / "packets" / "source" / "source_packet_status_summary.csv"
    if source_packet_status_path.exists():
        source_packet_status = pd.read_csv(source_packet_status_path)
        keep_cols = ["packet_type", "packet_key", "state_abbr", "packet_status", "production_input_ready", "notes"]
        tables["source_packet_status_summary"] = source_packet_status[
            [col for col in keep_cols if col in source_packet_status.columns]
        ].sort_values("state_abbr", kind="mergesort")
    source_lane_summary_path = state_dir / "review" / "packets" / "source" / "state_actionable_lanes_summary.csv"
    if source_lane_summary_path.exists():
        source_lane_summary = pd.read_csv(source_lane_summary_path)
        tables["source_state_actionable_lanes_summary"] = source_lane_summary.sort_values(
            ["state_abbr", "implementation_readiness"], kind="mergesort"
        )
    municipal_packet_status_path = state_dir / "review" / "packets" / "municipal_targets" / "packet_status_summary.csv"
    if municipal_packet_status_path.exists():
        municipal_packet_status = pd.read_csv(municipal_packet_status_path)
        keep_cols = [
            "case_key",
            "jurisdiction_name",
            "state_abbr",
            "packet_status",
            "recommended_disposition",
            "production_ready",
            "confidence",
            "has_promotable_extract",
            "extract_files",
            "non_observed_count_2024",
            "top_estimate_source_2024",
        ]
        tables["municipal_packet_status_summary"] = municipal_packet_status[
            [col for col in keep_cols if col in municipal_packet_status.columns]
        ].sort_values(["state_abbr", "jurisdiction_name"], kind="mergesort")
    tables["city_geocode_quality_2024"] = (
        city_share_surface.loc[city_share_surface["year"].eq(2024)]
        .groupby("geocode_quality_tier", dropna=False)
        .agg(
            row_count=("block_group_geoid", "size"),
            incident_total=("incident_count", "sum"),
            city_count=("city_name", "nunique"),
            offense_count=("offense", "nunique"),
        )
        .reset_index()
        .sort_values("incident_total", ascending=False, kind="mergesort")
    )

    city_share_by_city = _weighted_city_agg(city_share_detail, by="city_name").sort_values(
        "city_name", key=lambda s: s.map({c: i for i, c in enumerate(CITY_ORDER)}), kind="mergesort"
    )
    tables["city_share_benchmark_by_city"] = city_share_by_city

    city_share_by_offense = _weighted_city_agg(city_share_detail, by="offense").sort_values(
        "offense", key=lambda s: s.map({o: i for i, o in enumerate(OFFENSE_ORDER)}), kind="mergesort"
    )
    city_share_by_offense["offense_label"] = city_share_by_offense["offense"].map(OFFENSE_LABELS)
    tables["city_share_benchmark_by_offense"] = city_share_by_offense
    tables["city_share_benchmark_detail"] = city_share_detail.sort_values(["city_name", "offense"], kind="mergesort")

    city_residual_by_city_rows = []
    for city_name, group in city_residual_detail.groupby("city_name", dropna=False):
        city_residual_by_city_rows.append(
            {
                "city_name": city_name,
                "incident_total": float(group["incident_total"].sum()),
                "rows": int(len(group)),
                "baseline_weighted_total_variation_distance_mean": _weighted_mean(group, "baseline_total_variation_distance", "incident_total"),
                "residual_weighted_total_variation_distance_mean": _weighted_mean(group, "residual_total_variation_distance", "incident_total"),
                "weighted_tvd_delta": _weighted_mean(group, "tvd_delta", "incident_total"),
                "baseline_weighted_pearson_share_mean": _weighted_mean(group, "baseline_pearson_share", "incident_total"),
                "residual_weighted_pearson_share_mean": _weighted_mean(group, "residual_pearson_share", "incident_total"),
                "baseline_weighted_spearman_share_mean": _weighted_mean(group, "baseline_spearman_share", "incident_total"),
                "residual_weighted_spearman_share_mean": _weighted_mean(group, "residual_spearman_share", "incident_total"),
                "baseline_weighted_top10_capture_mean": _weighted_mean(group, "baseline_top_10pct_true_mass_in_model_top_10pct", "incident_total"),
                "residual_weighted_top10_capture_mean": _weighted_mean(group, "residual_top_10pct_true_mass_in_model_top_10pct", "incident_total"),
            }
        )
    tables["city_residual_benchmark_by_city"] = pd.DataFrame(city_residual_by_city_rows).sort_values(
        "city_name", key=lambda s: s.map({c: i for i, c in enumerate(CITY_ORDER)}), kind="mergesort"
    )

    city_residual_by_offense_rows = []
    for offense, group in city_residual_detail.groupby("offense", dropna=False):
        city_residual_by_offense_rows.append(
            {
                "offense": offense,
                "offense_label": OFFENSE_LABELS.get(offense, offense),
                "incident_total": float(group["incident_total"].sum()),
                "rows": int(len(group)),
                "baseline_weighted_total_variation_distance_mean": _weighted_mean(group, "baseline_total_variation_distance", "incident_total"),
                "residual_weighted_total_variation_distance_mean": _weighted_mean(group, "residual_total_variation_distance", "incident_total"),
                "weighted_tvd_delta": _weighted_mean(group, "tvd_delta", "incident_total"),
                "baseline_weighted_pearson_share_mean": _weighted_mean(group, "baseline_pearson_share", "incident_total"),
                "residual_weighted_pearson_share_mean": _weighted_mean(group, "residual_pearson_share", "incident_total"),
                "baseline_weighted_spearman_share_mean": _weighted_mean(group, "baseline_spearman_share", "incident_total"),
                "residual_weighted_spearman_share_mean": _weighted_mean(group, "residual_spearman_share", "incident_total"),
                "baseline_weighted_top10_capture_mean": _weighted_mean(group, "baseline_top_10pct_true_mass_in_model_top_10pct", "incident_total"),
                "residual_weighted_top10_capture_mean": _weighted_mean(group, "residual_top_10pct_true_mass_in_model_top_10pct", "incident_total"),
            }
        )
    tables["city_residual_benchmark_by_offense"] = pd.DataFrame(city_residual_by_offense_rows).sort_values(
        "offense", key=lambda s: s.map({o: i for i, o in enumerate(["burglary", "larceny", "motor_vehicle_theft", "robbery", "aggravated_assault"])}), kind="mergesort"
    )
    tables["city_residual_benchmark_detail"] = city_residual_detail.sort_values(["city_name", "offense"], kind="mergesort")

    dominant_bg = (
        bg_crosswalk.sort_values(
            ["block_group_geoid", "allocation_share", "pop_share", "blocks"],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .drop_duplicates("block_group_geoid")
        .rename(columns={"jurisdiction_id": "dominant_jurisdiction_id", "jurisdiction_type": "dominant_jurisdiction_type"})
    )
    city_jids = (
        city_share_detail[["city_name", "jurisdiction_id", "state_fips"]]
        .drop_duplicates()
        .sort_values("city_name", kind="mergesort")
    )
    bg_city = dominant_bg.merge(city_jids, left_on="dominant_jurisdiction_id", right_on="jurisdiction_id", how="inner")
    city_output_summary = (
        bg_city.merge(bg_output, left_on="block_group_geoid", right_on="block_group_geoid", how="left")
        .groupby(["city_name", "state_fips_x"], dropna=False)
        .agg(
            block_group_count=("block_group_geoid", "nunique"),
            population_2024=("population_2024", "sum"),
            expected_count_total=("expected_count_total", "sum"),
            mean_index_total_primary_event_weighted=("index_total_primary_event_weighted", "mean"),
            p95_index_total_primary_event_weighted=("index_total_primary_event_weighted", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.95))),
            mean_index_total_part1_resident=("index_total_part1_resident", "mean"),
            p95_index_total_part1_resident=("index_total_part1_resident", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.95))),
        )
        .reset_index()
        .rename(columns={"state_fips_x": "state_fips"})
    )
    city_output_summary["resident_expected_count_total_per_100k_context"] = (
        100000.0 * city_output_summary["expected_count_total"] / city_output_summary["population_2024"]
    )
    tables["city_output_surface_summary"] = city_output_summary.sort_values(
        "city_name", key=lambda s: s.map({c: i for i, c in enumerate(CITY_ORDER)}), kind="mergesort"
    )

    tables["fbi_calibrated_surface_comparison"] = pd.DataFrame(
        [
            {
                "surface": "block_group",
                "rows": int(len(bg_output)),
                "max_abs_expected_count_total_delta": float((pd.to_numeric(bg_output_fbi["expected_count_total"], errors="coerce") - pd.to_numeric(bg_output["expected_count_total"], errors="coerce")).abs().max()),
                "mean_abs_expected_count_total_delta": float((pd.to_numeric(bg_output_fbi["expected_count_total"], errors="coerce") - pd.to_numeric(bg_output["expected_count_total"], errors="coerce")).abs().mean()),
            },
            {
                "surface": "tract",
                "rows": int(len(tract_output)),
                "max_abs_expected_count_total_delta": float((pd.to_numeric(tract_output_fbi["expected_count_total"], errors="coerce") - pd.to_numeric(tract_output["expected_count_total"], errors="coerce")).abs().max()),
                "mean_abs_expected_count_total_delta": float((pd.to_numeric(tract_output_fbi["expected_count_total"], errors="coerce") - pd.to_numeric(tract_output["expected_count_total"], errors="coerce")).abs().mean()),
            },
        ]
    )

    next_phase_summary_path = state_dir / "modeling" / "next_phase_measurement_summary_2024.json"
    promoted_residual_summary_path = state_dir / "modeling" / "next_phase_city_residual_benchmark_overture_core_2024.json"
    promoted_preflight_path = state_dir / "modeling" / "promoted_next_phase_allocator_preflight_2024.json"
    output_build_manifest_path = state_dir / "output" / "crimerisk_output_build_2024.json"
    output_validation_summary_path = state_dir / "output" / "validation_summary.json"
    next_phase_summary = json.loads(next_phase_summary_path.read_text()) if next_phase_summary_path.exists() else {}
    promoted_residual_json = (
        json.loads(promoted_residual_summary_path.read_text()) if promoted_residual_summary_path.exists() else {}
    )
    promoted_preflight = json.loads(promoted_preflight_path.read_text()) if promoted_preflight_path.exists() else {}
    output_build_manifest = (
        json.loads(output_build_manifest_path.read_text()) if output_build_manifest_path.exists() else {}
    )
    output_validation_summary = (
        json.loads(output_validation_summary_path.read_text()) if output_validation_summary_path.exists() else {}
    )

    if next_phase_summary:
        tables["next_phase_truth_case_summary"] = pd.DataFrame(
            [
                {
                    "truth_case_type": case_type,
                    "case_count": int(count),
                }
                for case_type, count in sorted(next_phase_summary.get("truth_case_type_counts", {}).items())
            ]
        )
        tables["next_phase_error_budget_class_summary"] = pd.DataFrame(
            [
                {
                    "dominance_class": dominance_class,
                    "rows": int(count),
                }
                for dominance_class, count in sorted(next_phase_summary.get("error_budget_class_counts", {}).items())
            ]
        )
        tables["next_phase_decision_summary"] = pd.DataFrame(
            [
                {
                    "metric": "truth_cases",
                    "value": int(next_phase_summary.get("truth_case_count", 0)),
                    "notes": "all direct incident truth cases in the expanded diagnostic surface",
                },
                {
                    "metric": "truth_cities",
                    "value": int(next_phase_summary.get("truth_city_count", 0)),
                    "notes": "municipal truth cases; Montgomery County is kept as a county validation case",
                },
                {
                    "metric": "jurisdiction_total_truth_cases",
                    "value": int(next_phase_summary.get("jurisdiction_total_truth_case_count", 0)),
                    "notes": "cases with jurisdiction-level total truth for total-vs-allocation diagnostics",
                },
                {
                    "metric": "error_budget_rows",
                    "value": int(next_phase_summary.get("error_budget_rows", 0)),
                    "notes": "offense/case rows classified by explicit dominance rules",
                },
                {
                    "metric": "cv_prediction_rows",
                    "value": int(next_phase_summary.get("cv_prediction_rows", 0)),
                    "notes": "held-out per-jurisdiction total predictions with context columns",
                },
                {
                    "metric": "recommended_next_workstream",
                    "value": str(next_phase_summary.get("recommended_next_workstream", "")),
                    "notes": str(next_phase_summary.get("decision_rationale", "")),
                },
            ]
        )

    if promoted_residual_json:
        tables["promoted_allocator_residual_summary"] = pd.DataFrame(
            [
                {
                    "holdout_cities": int(promoted_residual_json.get("holdout_city_count", 0)),
                    "rows": int(promoted_residual_json.get("rows", 0)),
                    "incident_total": float(promoted_residual_json.get("incident_total", 0.0)),
                    "baseline_weighted_tvd": float(
                        promoted_residual_json.get("baseline_weighted_total_variation_distance_mean", float("nan"))
                    ),
                    "residual_weighted_tvd": float(
                        promoted_residual_json.get("residual_weighted_total_variation_distance_mean", float("nan"))
                    ),
                    "weighted_tvd_delta": float(promoted_residual_json.get("weighted_tvd_delta", float("nan"))),
                    "improved_rows": int(promoted_residual_json.get("improved_tvd_rows", 0)),
                    "worsened_rows": int(promoted_residual_json.get("worsened_tvd_rows", 0)),
                    "extra_bg_feature_count": int(promoted_residual_json.get("extra_bg_feature_count", 0)),
                }
            ]
        )

    if promoted_preflight:
        tables["promoted_allocator_preflight_inputs"] = pd.DataFrame(
            [
                {
                    "role": row.get("role", ""),
                    "exists": bool(row.get("exists", False)),
                    "size_bytes": int(row.get("size_bytes", 0) or 0),
                    "path": _repo_relpath(row.get("path"), repo_root),
                }
                for row in promoted_preflight.get("required_inputs", [])
            ]
        )
        tables["promoted_allocator_preflight_summary"] = pd.DataFrame(
            [
                {
                    "ready": bool(promoted_preflight.get("ready", False)),
                    "missing_required_paths": len(promoted_preflight.get("missing_required_paths", [])),
                    "default_excluded_validation_case_types": ", ".join(
                        promoted_preflight.get("default_excluded_validation_case_types", [])
                    ),
                }
            ]
        )

    if output_build_manifest:
        manifest_summary = output_build_manifest.get("summary", {})
        resolved_config = output_build_manifest.get("resolved_config", {})
        promotion = output_build_manifest.get("promotion", {})
        tables["output_build_manifest_summary"] = pd.DataFrame(
            [
                {
                    "year": int(output_build_manifest.get("year", 0)),
                    "created_at_utc": output_build_manifest.get("created_at_utc", ""),
                    "promoted_at_utc": promotion.get("promoted_at_utc", ""),
                    "candidate_run_id": promotion.get("candidate_run_id", ""),
                    "block_groups": int(manifest_summary.get("block_groups", 0)),
                    "tracts": int(manifest_summary.get("tracts", 0)),
                    "fbi_calibrated_written": bool(manifest_summary.get("fbi_calibrated_written", False)),
                    "county_anchoring_enabled": bool(manifest_summary.get("county_anchoring_enabled", False)),
                    "model_surface_prior_anchor": manifest_summary.get("model_surface_prior_anchor", ""),
                    "promoted_allocator_enabled": bool(
                        manifest_summary.get("promoted_next_phase_allocator_enabled", False)
                    ),
                    "promoted_allocator_applied": bool(
                        manifest_summary.get("promoted_next_phase_allocator_applied", False)
                    ),
                    "excluded_validation_case_types": ", ".join(
                        resolved_config.get("residual_training_exclude_validation_case_types", [])
                    ),
                    "extra_bg_feature_paths": len(resolved_config.get("residual_training_extra_bg_feature_paths", [])),
                    "candidate_manifest_sha256": promotion.get("candidate_manifest_sha256", ""),
                    "validation_summary_sha256": promotion.get("validation_summary_sha256", ""),
                }
            ]
        )
        tables["promoted_output_artifact_hashes"] = pd.DataFrame(
            [
                {
                    "artifact": row.get("name", ""),
                    "destination": row.get("destination", ""),
                    "sha256": row.get("sha256", ""),
                    "size_bytes": int(row.get("size_bytes", 0) or 0),
                }
                for row in promotion.get("copied_artifacts", [])
            ]
        )
        frontend_hash = promotion.get("frontend_snapshot_hash_check", {})
        tables["frontend_snapshot_hash_check"] = pd.DataFrame(
            [
                {
                    "snapshot_path": frontend_hash.get("snapshot_path", ""),
                    "snapshot_source_parquet": frontend_hash.get("snapshot_source_parquet", ""),
                    "snapshot_source_parquet_sha256": frontend_hash.get("snapshot_source_parquet_sha256", ""),
                    "promoted_block_group_ags_core_sha256": frontend_hash.get("promoted_block_group_ags_core_sha256", ""),
                    "matches_promoted_block_group_ags_core": bool(
                        frontend_hash.get("matches_promoted_block_group_ags_core", False)
                    ),
                }
            ]
        )
        burglary_gate = manifest_summary.get("burglary_commercial_gradient", {}).get("block_group_ags_core", {})
        tables["burglary_commercial_gate_summary"] = pd.DataFrame(
            [
                {
                    "ok": bool(burglary_gate.get("ok", False)),
                    "after_q5_q1_mean": burglary_gate.get("after_q5_q1_mean", float("nan")),
                    "after_q5_q1_mean_direct": burglary_gate.get("after_q5_q1_mean_direct", float("nan")),
                    "after_q5_q1_mean_modeled": burglary_gate.get("after_q5_q1_mean_modeled", float("nan")),
                    "regime_policy": burglary_gate.get("regime_policy", ""),
                    "rows": int(burglary_gate.get("rows", 0) or 0),
                }
            ]
        )
        burglary_cal = manifest_summary.get("burglary_commercial_calibration", {})
        tables["burglary_commercial_calibration_summary"] = pd.DataFrame(
            [
                {
                    "k_commercial": burglary_cal.get("k_commercial", float("nan")),
                    "k_destination_poi": burglary_cal.get("k_destination_poi", float("nan")),
                    "k_retail_jobs": burglary_cal.get("k_retail_jobs", float("nan")),
                    "k_industrial_jobs": burglary_cal.get("k_industrial_jobs", float("nan")),
                    "nnls_residual_norm": burglary_cal.get("nnls_residual_norm", float("nan")),
                    "single_term_nnls_residual_norm": burglary_cal.get("single_term_nnls_residual_norm", float("nan")),
                    "calibration_rows": int(burglary_cal.get("calibration_rows", 0) or 0),
                    "calibration_block_groups": int(burglary_cal.get("calibration_block_groups", 0) or 0),
                    "calibration_source": burglary_cal.get("calibration_source", ""),
                    "calibration_year_min": burglary_cal.get("calibration_year_min", ""),
                    "calibration_year_max": burglary_cal.get("calibration_year_max", ""),
                    "used_fallback": bool(burglary_cal.get("used_fallback", False)),
                }
            ]
        )

    if output_validation_summary:
        tables["output_validation_summary"] = pd.DataFrame(
            [
                {
                    "ok": bool(output_validation_summary.get("ok", False)),
                    "issue_count": len(output_validation_summary.get("issues", [])),
                    "surface_count": int(output_validation_summary.get("surface_count", 0) or 0),
                    "total_lane_ok": bool(output_validation_summary.get("total_lane_qa", {}).get("ok", False)),
                    "spatial_artifact_gates_ok": bool(
                        output_validation_summary.get("spatial_artifact_gates", {}).get("ok", False)
                    ),
                    "sparse_residual_transfer_policy_present": bool(
                        output_validation_summary.get("sparse_residual_transfer_policy", {}).get("present", False)
                    ),
                    "static_no_exposure_tempered_calls": bool(
                        output_validation_summary.get("static_no_exposure_tempered_calls", {}).get("ok", False)
                    ),
                }
            ]
        )

    dashboard_summary_path = state_dir / "modeling" / "dashboard_neighborhood_check_lookup_2024.json"
    dashboard_comparison_path = repo_root / "materials" / "tables" / "dashboard_neighborhood_coarse_comparison_lookup.csv"
    external_availability_path = state_dir / "modeling" / "external_surface_availability_2024.json"
    dashboard_summary = json.loads(dashboard_summary_path.read_text()) if dashboard_summary_path.exists() else {}
    external_availability = json.loads(external_availability_path.read_text()) if external_availability_path.exists() else {}
    if dashboard_summary:
        tables["dashboard_neighborhood_lookup_summary"] = pd.DataFrame(
            [
                {
                    "neighborhood_basis": dashboard_summary.get("neighborhood_basis", ""),
                    "neighborhood_count": int(dashboard_summary.get("neighborhood_count", 0)),
                    "tract_weight_rows": int(dashboard_summary.get("tract_weight_rows", 0)),
                    "dashboard_coarse_rows": int(dashboard_summary.get("dashboard_coarse_rows", 0)),
                    "dashboard_risk_score_rows": int(dashboard_summary.get("dashboard_risk_score_rows", 0)),
                    "risk_score_vs_expected_count_spearman": float(
                        dashboard_summary.get("dashboard_risk_score_vs_crimerisk_expected_count_total_spearman", float("nan"))
                    ),
                }
            ]
        )
    if dashboard_comparison_path.exists():
        dashboard_comparison = pd.read_csv(dashboard_comparison_path)
        keep_cols = [
            "neighborhood_id",
            "neighborhood_name",
            "expected_count_total",
            "dashboard_risk_score_area_weighted",
            "crimerisk_expected_count_total_rank_desc",
            "dashboard_risk_score_rank_desc",
            "crimerisk_expected_count_rank_minus_dashboard_rank",
        ]
        keep_cols = [col for col in keep_cols if col in dashboard_comparison.columns]
        ranked = dashboard_comparison[keep_cols].copy()
        if "crimerisk_expected_count_rank_minus_dashboard_rank" in ranked.columns:
            ranked["_abs_count_rank_delta"] = pd.to_numeric(
                ranked["crimerisk_expected_count_rank_minus_dashboard_rank"],
                errors="coerce",
            ).abs()
            tables["dashboard_neighborhood_lookup_largest_count_rank_deltas"] = (
                ranked.sort_values("_abs_count_rank_delta", ascending=False, kind="mergesort")
                .drop(columns=["_abs_count_rank_delta"])
                .head(25)
                .reset_index(drop=True)
            )
    if external_availability:
        tables["external_surface_availability_summary"] = pd.DataFrame(
            [
                {
                    "status": external_availability.get("status", ""),
                    "usable_external_surface_count": int(external_availability.get("usable_external_surface_count", 0)),
                    "candidate_rows": int(external_availability.get("candidate_rows", 0)),
                    "reference_or_methodology_count": int(
                        external_availability.get("reference_or_methodology_count", 0)
                    ),
                    "external_comparison_harness": external_availability.get("external_comparison_harness", ""),
                    "harness_scoring_target": external_availability.get("harness_scoring_target", ""),
                }
            ]
        )

    tables["geometry_assignment_method_summary"] = (
        block_crosswalk.groupby("assignment_method", dropna=False)
        .agg(
            block_rows=("block_geoid", "size"),
            block_group_count=("block_group_geoid", "nunique"),
            pop20_total=("pop20", "sum"),
            housing20_total=("housing20", "sum"),
        )
        .reset_index()
        .sort_values("block_rows", ascending=False, kind="mergesort")
    )
    bg_allocation_sums = (
        bg_crosswalk.groupby("block_group_geoid", dropna=False)["allocation_share"]
        .sum()
        .reset_index(name="allocation_share_sum")
    )
    bg_allocation_sums["allocation_status"] = "exact_one"
    bg_allocation_sums.loc[bg_allocation_sums["allocation_share_sum"].lt(0.999999), "allocation_status"] = "below_one"
    bg_allocation_sums.loc[bg_allocation_sums["allocation_share_sum"].gt(1.000001), "allocation_status"] = "above_one"
    tables["geometry_allocation_sum_summary"] = (
        bg_allocation_sums.groupby("allocation_status", dropna=False)
        .agg(
            block_group_count=("block_group_geoid", "size"),
            min_sum=("allocation_share_sum", "min"),
            max_sum=("allocation_share_sum", "max"),
        )
        .reset_index()
        .sort_values("block_group_count", ascending=False, kind="mergesort")
    )

    summary = {
        "qa_summary": qa_summary,
        "benchmark_json": benchmark_json,
        "city_share_json": json.loads((state_dir / "modeling" / "city_share_benchmark_2024.json").read_text()),
        "city_residual_json": json.loads((state_dir / "modeling" / "city_residual_benchmark_2024.json").read_text()),
        "next_phase_summary": next_phase_summary,
        "promoted_residual_json": promoted_residual_json,
        "promoted_preflight": promoted_preflight,
        "output_build_manifest": output_build_manifest,
        "output_validation_summary": output_validation_summary,
        "dashboard_lookup_summary": dashboard_summary,
        "external_surface_availability": external_availability,
        "controls_rows": int(len(controls_2024)),
        "control_unique_jurisdictions": int(controls_2024["jurisdiction_id"].nunique()),
        "output_block_groups": int(len(bg_output)),
        "output_tracts": int(len(tract_output)),
        "city_jurisdictions": city_jids,
        "dominant_bg": dominant_bg,
    }
    return tables, summary


def _make_figures_and_maps(
    *,
    repo_root: Path,
    tables: dict[str, pd.DataFrame],
    summary: dict[str, object],
    figures_dir: Path,
    maps_dir: Path,
) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []

    observed_share = tables["observed_share_by_year"]
    _save_line(
        observed_share,
        x_col="year",
        y_cols=["observed_row_share", "observed_estimated_count_share"],
        labels=["Observed row share", "Observed estimated-count share"],
        path=figures_dir / "observed_share_by_year.png",
        title="Observed coverage in the seven-year jurisdiction panel",
        ylabel="Share",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/observed_share_by_year.png",
            "category": "figure",
            "description": "Observed-share coverage by year, shown both as row share and estimated-count share.",
        }
    )

    _save_bar(
        tables["controls_preferred_source_2024"],
        x_col="preferred_source",
        y_col="adjusted_count_share",
        path=figures_dir / "controls_preferred_source_adjusted_share_2024.png",
        title="2024 adjusted crime volume by preferred source",
        xlabel="Preferred source",
        ylabel="Adjusted-count share",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/controls_preferred_source_adjusted_share_2024.png",
            "category": "figure",
            "description": "Share of 2024 adjusted crime volume carried by each preferred source.",
        }
    )

    _save_bar(
        tables["controls_estimate_source_2024"],
        x_col="estimate_source",
        y_col="adjusted_count_share",
        path=figures_dir / "controls_estimate_source_adjusted_share_2024.png",
        title="2024 adjusted crime volume by estimate source",
        xlabel="Estimate source",
        ylabel="Adjusted-count share",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/controls_estimate_source_adjusted_share_2024.png",
            "category": "figure",
            "description": "Share of 2024 adjusted crime volume that is directly observed versus panel-estimated.",
        }
    )

    _save_bar(
        tables["controls_reporting_regime_2024"],
        x_col="dominant_reporting_regime",
        y_col="adjusted_count_share",
        path=figures_dir / "controls_reporting_regime_adjusted_share_2024.png",
        title="2024 adjusted crime volume by dominant reporting regime",
        xlabel="Reporting regime",
        ylabel="Adjusted-count share",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/controls_reporting_regime_adjusted_share_2024.png",
            "category": "figure",
            "description": "Share of 2024 adjusted crime volume by dominant reporting regime.",
        }
    )

    benchmark = tables["benchmark_offense_summary"]
    _save_bar(
        benchmark.sort_values("cv_r2_log_rate", ascending=False, kind="mergesort"),
        x_col="offense_label",
        y_col="cv_r2_log_rate",
        path=figures_dir / "benchmark_cv_r2_log_rate_by_offense.png",
        title="Cross-validated log-rate fit by offense",
        xlabel="Offense",
        ylabel="CV R² (log rate)",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/benchmark_cv_r2_log_rate_by_offense.png",
            "category": "figure",
            "description": "Per-offense cross-validated log-rate fit for the canonical jurisdiction model.",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(benchmark))
    ax.bar(x - 0.18, benchmark["train_r2_log_rate"], width=0.36, label="Train R² (log rate)", color="#4C956C")
    ax.bar(x + 0.18, benchmark["cv_r2_log_rate"], width=0.36, label="CV R² (log rate)", color="#2C6E91")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark["offense_label"], rotation=30, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Train vs cross-validated fit by offense")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "benchmark_train_vs_cv_r2_log_rate.png", dpi=200)
    plt.close(fig)
    inventory.append(
        {
            "relative_path": "materials/figures/benchmark_train_vs_cv_r2_log_rate.png",
            "category": "figure",
            "description": "Train-versus-cross-validated log-rate fit by offense.",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.18, benchmark["cv_r2_log_rate"], width=0.36, label="CV R² (log rate)", color="#2C6E91")
    ax.bar(x + 0.18, benchmark["cv_r2_rate"], width=0.36, label="CV R² (rate)", color="#C75D2C")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark["offense_label"], rotation=30, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Cross-validated fit on log-rate and raw-rate targets")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "benchmark_cv_log_vs_rate_r2.png", dpi=200)
    plt.close(fig)
    inventory.append(
        {
            "relative_path": "materials/figures/benchmark_cv_log_vs_rate_r2.png",
            "category": "figure",
            "description": "Per-offense comparison of cross-validated fit on log-rate and raw-rate targets.",
        }
    )

    calibration = tables["benchmark_calibration_bins"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()
    for idx, offense in enumerate(OFFENSE_ORDER):
        ax = axes[idx]
        offense_df = calibration[calibration["offense"].eq(offense)].sort_values("mean_rate_pred", kind="mergesort")
        ax.plot(offense_df["mean_rate_pred"], offense_df["mean_rate_true"], marker="o", color="#2C6E91")
        max_val = float(max(offense_df["mean_rate_pred"].max(), offense_df["mean_rate_true"].max()))
        ax.plot([0, max_val], [0, max_val], linestyle="--", color="#444444", linewidth=1)
        ax.set_title(OFFENSE_LABELS.get(offense, offense))
        ax.set_xlabel("Mean predicted rate")
        ax.set_ylabel("Mean true rate")
        ax.grid(alpha=0.2)
    axes[-1].axis("off")
    axes[-2].axis("off")
    fig.suptitle("Calibration curves by offense", y=0.995)
    fig.tight_layout()
    fig.savefig(figures_dir / "benchmark_calibration_curves.png", dpi=200)
    plt.close(fig)
    inventory.append(
        {
            "relative_path": "materials/figures/benchmark_calibration_curves.png",
            "category": "figure",
            "description": "Per-offense calibration curves comparing mean predicted and mean true crime rates across deciles.",
        }
    )

    residual_by_pop = tables["benchmark_residual_by_pop_band"]
    pivot = residual_by_pop.pivot(index="pop_band", columns="offense", values="mean_abs_resid_log_rate")
    _save_heatmap(
        pivot,
        path=figures_dir / "benchmark_abs_residual_by_population_band.png",
        title="Mean absolute log-rate residual by population band",
        cmap="YlOrRd",
        cbar_label="Mean absolute residual",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/benchmark_abs_residual_by_population_band.png",
            "category": "figure",
            "description": "Mean absolute jurisdiction residuals by population band and offense.",
        }
    )

    city_share_detail = tables["city_share_benchmark_detail"]
    share_tvd_pivot = city_share_detail.pivot(index="city_name", columns="offense", values="total_variation_distance").reindex(CITY_ORDER)
    _save_heatmap(
        share_tvd_pivot,
        path=figures_dir / "city_share_tvd_heatmap.png",
        title="City-share benchmark: total variation distance",
        cmap="YlGnBu_r",
        cbar_label="TVD",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/city_share_tvd_heatmap.png",
            "category": "figure",
            "description": "Total variation distance between model shares and incident-based truth by city and offense.",
        }
    )

    share_pearson_pivot = city_share_detail.pivot(index="city_name", columns="offense", values="pearson_share").reindex(CITY_ORDER)
    _save_heatmap(
        share_pearson_pivot,
        path=figures_dir / "city_share_pearson_heatmap.png",
        title="City-share benchmark: Pearson correlation",
        cmap="YlGnBu",
        cbar_label="Pearson correlation",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/city_share_pearson_heatmap.png",
            "category": "figure",
            "description": "Pearson correlation between model shares and incident-based truth by city and offense.",
        }
    )

    city_residual_detail = tables["city_residual_benchmark_detail"]
    residual_tvd_pivot = city_residual_detail.pivot(index="city_name", columns="offense", values="tvd_delta").reindex(CITY_ORDER)
    _save_heatmap(
        residual_tvd_pivot,
        path=figures_dir / "city_residual_tvd_delta_heatmap.png",
        title="Residual-share refinement: TVD delta",
        cmap="RdYlGn_r",
        cbar_label="Residual - baseline TVD",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/city_residual_tvd_delta_heatmap.png",
            "category": "figure",
            "description": "Change in total variation distance after the residual-share refinement; lower is better.",
        }
    )

    residual_top10_pivot = city_residual_detail.pivot(index="city_name", columns="offense", values="top10_capture_delta").reindex(CITY_ORDER)
    _save_heatmap(
        residual_top10_pivot,
        path=figures_dir / "city_residual_top10_delta_heatmap.png",
        title="Residual-share refinement: top-10% capture delta",
        cmap="RdYlGn",
        cbar_label="Residual - baseline top-10 capture",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/city_residual_top10_delta_heatmap.png",
            "category": "figure",
            "description": "Change in top-decile hotspot capture after residual-share refinement.",
        }
    )

    _save_scatter(
        city_residual_detail,
        x_col="baseline_total_variation_distance",
        y_col="residual_total_variation_distance",
        path=figures_dir / "city_residual_before_after_tvd_scatter.png",
        title="Residual-share refinement on held-out city surfaces",
        xlabel="Baseline TVD",
        ylabel="Residual TVD",
        color_col="city_name",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/city_residual_before_after_tvd_scatter.png",
            "category": "figure",
            "description": "Before-versus-after TVD for held-out city residual benchmarking.",
        }
    )

    state_scatter = tables["state_control_comparison_detail"][["offense", "ags_core_adjusted_total", "fbi_cde_estimated_total"]].copy()
    _save_scatter(
        state_scatter,
        x_col="fbi_cde_estimated_total",
        y_col="ags_core_adjusted_total",
        path=figures_dir / "state_adjusted_vs_cde_scatter.png",
        title="Adjusted state totals versus FBI CDE totals",
        xlabel="FBI CDE estimated total",
        ylabel="AGS-core adjusted total",
        log_scale=True,
        color_col="offense",
    )
    inventory.append(
        {
            "relative_path": "materials/figures/state_adjusted_vs_cde_scatter.png",
            "category": "figure",
            "description": "State-offense adjusted totals compared with FBI CDE totals on log scales.",
        }
    )

    ratios = tables["state_control_comparison_detail"][["offense", "adjusted_to_cde_ratio"]].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_order = [o for o in OFFENSE_ORDER if o in ratios["offense"].unique()]
    data = [ratios.loc[ratios["offense"].eq(offense), "adjusted_to_cde_ratio"].dropna().to_numpy() for offense in plot_order]
    ax.boxplot(data, tick_labels=[OFFENSE_LABELS[o] for o in plot_order], showfliers=False)
    ax.axhline(1.0, linestyle="--", color="#444444", linewidth=1)
    ax.set_title("State adjusted-to-CDE ratio by offense")
    ax.set_ylabel("Adjusted / CDE ratio")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "state_adjusted_to_cde_ratio_boxplot.png", dpi=200)
    plt.close(fig)
    inventory.append(
        {
            "relative_path": "materials/figures/state_adjusted_to_cde_ratio_boxplot.png",
            "category": "figure",
            "description": "Distribution of adjusted-to-CDE state ratios by offense.",
        }
    )

    _save_hist(
        tables["top_block_groups_primary_event_weighted_index"].merge(
            pd.DataFrame({"block_group_geoid": []}),
            how="left",
        )["index_total_primary_event_weighted"] if False else pd.read_parquet(repo_root / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet")["index_total_primary_event_weighted"],
        path=figures_dir / "block_group_total_index_distribution.png",
        title="Block-group primary-event-weighted total index distribution",
        xlabel="Primary-event-weighted total index",
        bins=60,
    )
    inventory.append(
        {
            "relative_path": "materials/figures/block_group_total_index_distribution.png",
            "category": "figure",
            "description": "Distribution of total crime index values across all block groups.",
        }
    )

    _save_hist(
        pd.read_parquet(repo_root / "state" / "output" / "crimerisk_tract_2024_ags_core.parquet")["index_total_primary_event_weighted"],
        path=figures_dir / "tract_total_index_distribution.png",
        title="Tract primary-event-weighted total index distribution",
        xlabel="Primary-event-weighted total index",
        bins=60,
    )
    inventory.append(
        {
            "relative_path": "materials/figures/tract_total_index_distribution.png",
            "category": "figure",
            "description": "Distribution of total crime index values across all tracts.",
        }
    )

    fbi_compare = tables["output_offense_summary"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(fbi_compare))
    ax.bar(x - 0.18, fbi_compare["ags_core_total_expected_count"], width=0.36, label="AGS core", color="#2C6E91")
    ax.bar(x + 0.18, fbi_compare["fbi_calibrated_total_expected_count"], width=0.36, label="FBI calibrated", color="#C75D2C")
    ax.set_xticks(x)
    ax.set_xticklabels([OFFENSE_LABELS.get(o, o.title()) for o in fbi_compare["offense"]], rotation=30, ha="right")
    ax.set_ylabel("National count")
    ax.set_title("AGS-core versus FBI-calibrated national counts")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fbi_calibrated_vs_ags_core_offense_totals.png", dpi=200)
    plt.close(fig)
    inventory.append(
        {
            "relative_path": "materials/figures/fbi_calibrated_vs_ags_core_offense_totals.png",
            "category": "figure",
            "description": "National offense totals under the AGS-core and FBI-calibrated output surfaces.",
        }
    )

    tract_zips = sorted(
        p
        for p in (repo_root / "data" / "tiger_tracts").glob("tl_2020_*_tract.zip")
        if p.stem.split("_")[2] not in CONTIGUOUS_EXCLUDE
    )
    tract_geo = _load_zip_geometries(tract_zips, geoid_col="tract_id")
    tract_geo["tract_id"] = tract_geo["tract_id"].astype(str)
    tract_plot = tract_geo.merge(
        pd.read_parquet(repo_root / "state" / "output" / "crimerisk_tract_2024_ags_core.parquet"),
        on="tract_id",
        how="left",
    )
    tract_plot = tract_plot[tract_plot["state_fips"].astype(str).isin(sorted(set(tract_plot["state_fips"].astype(str)) - CONTIGUOUS_EXCLUDE))]
    _save_map(
        tract_plot,
        value_col="index_total_primary_event_weighted",
        path=maps_dir / "national_tract_index_total_conus.png",
        title="2024 primary-event-weighted total crime index by tract (contiguous U.S.)",
    )
    inventory.append(
        {
            "relative_path": "materials/maps/national_tract_index_total_conus.png",
            "category": "map",
            "description": "Contiguous-U.S. tract map of the total crime index.",
        }
    )
    city_jurisdictions = summary["city_jurisdictions"]
    dominant_bg = summary["dominant_bg"]
    bg_output = pd.read_parquet(repo_root / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet")

    city_panel_count = len(CITY_ORDER)
    city_grid_cols = 3 if city_panel_count > 4 else 2
    city_grid_rows = int(math.ceil(city_panel_count / city_grid_cols))
    city_index_grid_fig, city_index_axes = plt.subplots(
        city_grid_rows,
        city_grid_cols,
        figsize=(7 * city_grid_cols, 6 * city_grid_rows),
    )
    city_index_axes = np.atleast_1d(city_index_axes).ravel()

    for idx, city_name in enumerate(CITY_ORDER):
        row = city_jurisdictions[city_jurisdictions["city_name"].eq(city_name)].iloc[0]
        state_fips = CITY_STATE_FIPS[city_name]
        bg_zip = repo_root / "data" / "tiger_bg" / f"tl_2020_{state_fips}_bg.zip"
        city_geo = gpd.read_file(bg_zip)
        geoid_col = "GEOID" if "GEOID" in city_geo.columns else "GEOID20"
        city_geo = city_geo.rename(columns={geoid_col: "block_group_geoid"})
        city_bg_ids = dominant_bg[dominant_bg["dominant_jurisdiction_id"].eq(row["jurisdiction_id"])][["block_group_geoid"]].drop_duplicates()
        city_plot = city_geo[["block_group_geoid", "geometry"]].merge(city_bg_ids, on="block_group_geoid", how="inner").merge(
            bg_output, on="block_group_geoid", how="left"
        )

        _save_map(
            city_plot,
            value_col="index_total_primary_event_weighted",
            path=maps_dir / f"{city_name.lower().replace(' ', '_')}_bg_index_total.png",
            title=f"{city_name}: primary-event-weighted total crime index by block group",
        )
        inventory.append(
            {
                "relative_path": f"materials/maps/{city_name.lower().replace(' ', '_')}_bg_index_total.png",
                "category": "map",
                "description": f"{city_name} block-group map of the total crime index using dominant-jurisdiction block groups.",
            }
        )

        vmax = float(pd.to_numeric(city_plot["index_total_primary_event_weighted"], errors="coerce").quantile(0.99))
        city_plot.plot(
            column="index_total_primary_event_weighted",
            ax=city_index_axes[idx],
            cmap="viridis",
            linewidth=0.0,
            legend=False,
            vmin=0.0,
            vmax=vmax if vmax > 0 else None,
        )
        city_index_axes[idx].set_title(f"{city_name}: Primary total index")
        city_index_axes[idx].set_axis_off()

    for idx in range(len(CITY_ORDER), len(city_index_axes)):
        city_index_axes[idx].axis("off")

    city_index_grid_fig.tight_layout()
    city_index_grid_fig.savefig(maps_dir / "city_bg_index_total_grid.png", dpi=220)
    plt.close(city_index_grid_fig)
    inventory.append(
        {
            "relative_path": "materials/maps/city_bg_index_total_grid.png",
            "category": "map",
            "description": "Benchmark-city grid of total crime index block-group maps.",
        }
    )

    return inventory


def _write_materials_readme(*, repo_root: Path, materials_dir: Path, inventory_df: pd.DataFrame) -> Path:
    _ = repo_root
    path = materials_dir / "README.md"
    lines = [
        "# Materials",
        "",
        "This directory contains the derived tables, figures, and maps generated from the canonical",
        "2024 CrimeRisk build artifacts included in the submission package.",
        "",
        "## Layout",
        "",
        "- `tables/`: CSV tables used throughout the report and suitable for direct reuse in presentations and appendices.",
        "- `figures/`: charts and benchmark graphics.",
        "- `maps/`: national and city-level static maps.",
        "- `inventory.csv`: machine-readable inventory of the generated material files.",
        "",
        "## Inventory",
        "",
        _df_to_markdown(inventory_df[["category", "relative_path", "description"]]),
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def _write_report(
    *,
    repo_root: Path,
    report_dir: Path,
    tables: dict[str, pd.DataFrame],
    summary: dict[str, object],
    package_repo_prefix: str = "",
) -> Path:
    report_path = report_dir / "CrimeRisk_Submission_Report.md"
    qa = summary["qa_summary"]
    bench = summary["benchmark_json"]
    city_share_json = summary["city_share_json"]
    city_residual_json = summary["city_residual_json"]
    next_phase_summary = summary.get("next_phase_summary", {})
    promoted_residual_json = summary.get("promoted_residual_json", {})
    output_build_manifest = summary.get("output_build_manifest", {})
    output_manifest_summary = output_build_manifest.get("summary", {}) if isinstance(output_build_manifest, dict) else {}
    output_resolved_config = output_build_manifest.get("resolved_config", {}) if isinstance(output_build_manifest, dict) else {}
    shipped_bg_prior_path = _resolve_repo_path(
        output_manifest_summary.get("bg_prior_path")
        or output_resolved_config.get("bg_prior_path")
        or "state/modeling/bg_prior_long_2024_burglary_exposure_dedup_burglary_only.parquet",
        repo_root,
    )
    shipped_bg_prior_rel = _repo_relpath(shipped_bg_prior_path, repo_root)
    next_phase_decision_rationale = str(
        next_phase_summary.get("decision_rationale", "the allocation diagnostics dominate")
    ).strip()
    if next_phase_decision_rationale:
        next_phase_decision_rationale = (
            next_phase_decision_rationale[0].lower() + next_phase_decision_rationale[1:]
        ).rstrip(".")

    reference_summary = tables["reference_summary"].iloc[0]
    obs_summary = tables["observation_stage_summary"].iloc[0]
    preferred_source = tables["controls_preferred_source_2024"].copy()
    preferred_source["adjusted_count_share_pct"] = preferred_source["adjusted_count_share"].map(lambda x: round(100.0 * x, 2))
    preferred_origin = tables["controls_preferred_source_origin_2024"].copy()
    preferred_origin["adjusted_count_share_pct"] = preferred_origin["adjusted_count_share"].map(lambda x: round(100.0 * x, 2))
    estimate_source = tables["controls_estimate_source_2024"].copy()
    estimate_source["adjusted_count_share_pct"] = estimate_source["adjusted_count_share"].map(lambda x: round(100.0 * x, 2))
    reporting_regime = tables["controls_reporting_regime_2024"].copy()
    reporting_regime["adjusted_count_share_pct"] = reporting_regime["adjusted_count_share"].map(lambda x: round(100.0 * x, 2))
    city_by_city = tables["city_share_benchmark_by_city"].copy()
    city_resid_by_city = tables["city_residual_benchmark_by_city"].copy()
    output_summary = tables["output_offense_summary"].copy()
    city_share_by_offense = tables["city_share_benchmark_by_offense"].copy()
    city_resid_by_offense = tables["city_residual_benchmark_by_offense"].copy()
    feature_inventory = pd.read_parquet(repo_root / "state" / "modeling" / "jurisdiction_model_features_2024.parquet")
    bg_prior = pd.read_parquet(shipped_bg_prior_path)
    city_share_surface = pd.read_parquet(repo_root / "state" / "modeling" / "city_incident_share_surface.parquet", columns=["year", "city_name", "offense"])
    city_share_surface_2024 = city_share_surface.loc[city_share_surface["year"].eq(2024)]
    bg_output_live = pd.read_parquet(
        repo_root / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet",
        columns=["block_group_geoid", "population_2024", "expected_count_total"],
    )
    bg_without_prior = bg_output_live.loc[~bg_output_live["block_group_geoid"].isin(bg_prior["bg_id"].astype(str))]
    benchmark_offense = tables["benchmark_offense_summary"].copy()
    best_benchmark_row = benchmark_offense.sort_values("cv_r2_log_rate", ascending=False, kind="mergesort").iloc[0]
    worst_benchmark_row = benchmark_offense.sort_values("cv_r2_log_rate", ascending=True, kind="mergesort").iloc[0]
    best_city_row = city_by_city.sort_values("weighted_total_variation_distance_mean", ascending=True, kind="mergesort").iloc[0]
    worst_city_row = city_by_city.sort_values("weighted_total_variation_distance_mean", ascending=False, kind="mergesort").iloc[0]
    best_share_offense_row = city_share_by_offense.sort_values("weighted_total_variation_distance_mean", ascending=True, kind="mergesort").iloc[0]
    worst_share_offense_row = city_share_by_offense.sort_values("weighted_total_variation_distance_mean", ascending=False, kind="mergesort").iloc[0]
    best_resid_city_row = city_resid_by_city.sort_values("weighted_tvd_delta", ascending=True, kind="mergesort").iloc[0]
    worst_resid_offense_row = city_resid_by_offense.sort_values("weighted_tvd_delta", ascending=False, kind="mergesort").iloc[0]
    city_geocode_quality_2024 = tables["city_geocode_quality_2024"].copy()
    geometry_bucket = tables["geometry_allocation_bucket_summary"].copy()

    promoted_surfaces_table = pd.DataFrame(
        [
            {
                "surface": "local_publication_annual",
                "rows": int(pd.read_parquet(repo_root / "state" / "modeling" / "inputs" / "local_publication_annual.parquet").shape[0]),
                "coverage_units": "5 jurisdictions",
                "notes": "Atlanta, Brunswick, Hallandale Beach, New Orleans, and Quincy annual publications",
            },
            {
                "surface": "state_publication_annual",
                "rows": int(pd.read_parquet(repo_root / "state" / "modeling" / "inputs" / "state_publication_annual.parquet").shape[0]),
                "coverage_units": "286 agencies in FL and MS",
                "notes": "FDLE and Mississippi TOPS annual offense detail inputs",
            },
        ]
    )
    pipeline_stage_table = pd.DataFrame(
        [
            {
                "stage": "Reference",
                "canonical_artifact": "state/reference/jurisdiction_master.parquet",
                "scale": f"{_count_fmt(reference_summary['agencies'])} agencies; {_count_fmt(reference_summary['jurisdictions'])} jurisdictions",
                "role": "canonical agency and jurisdiction universe plus crosswalk inputs",
            },
            {
                "stage": "Observations",
                "canonical_artifact": "state/observations/jurisdiction_year_observations.parquet",
                "scale": f"{_count_fmt(obs_summary['agency_year_observation_rows'])} agency-year rows; {_count_fmt(obs_summary['jurisdiction_year_observation_rows'])} jurisdiction-year rows",
                "role": "final observed offense history before 2024 estimation and control arbitration",
            },
            {
                "stage": "Controls",
                "canonical_artifact": "state/controls/jurisdiction_controls_2024.parquet",
                "scale": f"{_count_fmt(summary['controls_rows'])} 2024 offense rows across {_count_fmt(summary['control_unique_jurisdictions'])} jurisdictions",
                "role": "released 2024 jurisdiction totals that drive spatial allocation",
            },
            {
                "stage": "Modeling",
                "canonical_artifact": shipped_bg_prior_rel,
                "scale": f"{_count_fmt(len(feature_inventory))} retained feature-inventory rows; {_count_fmt(len(bg_prior))} BG-offense prior rows",
                "role": "offense-denominator anchored BG prior used by the promoted release",
            },
            {
                "stage": "Outputs",
                "canonical_artifact": "state/output/crimerisk_block_group_2024_ags_core.parquet",
                "scale": f"{_count_fmt(summary['output_block_groups'])} block groups; {_count_fmt(summary['output_tracts'])} tracts",
                "role": "final AGS-core release surface plus diagnostic FBI-calibrated surfaces",
            },
        ]
    )
    reference_input_table = pd.DataFrame(
        [
            {
                "canonical_input": "state/reference/inputs/provisional_local_agency_matches.parquet",
                "rows": int(pd.read_parquet(repo_root / "state" / "reference" / "inputs" / "provisional_local_agency_matches.parquet").shape[0]),
                "role": "first-pass local agency candidate matches that seed the local-resolution workflow",
            },
            {
                "canonical_input": "state/reference/inputs/local_queue_resolved_final.parquet",
                "rows": int(pd.read_parquet(repo_root / "state" / "reference" / "inputs" / "local_queue_resolved_final.parquet").shape[0]),
                "role": "reviewed local tail resolutions promoted out of the local-resolution queue",
            },
            {
                "canonical_input": "state/reference/inputs/nonmunicipal_special_resolved_final.parquet",
                "rows": int(pd.read_parquet(repo_root / "state" / "reference" / "inputs" / "nonmunicipal_special_resolved_final.parquet").shape[0]),
                "role": "reviewed nonlocal special-jurisdiction and overlap classifications",
            },
            {
                "canonical_input": "state/reference/inputs/nonmunicipal_auto_defaults.parquet",
                "rows": int(pd.read_parquet(repo_root / "state" / "reference" / "inputs" / "nonmunicipal_auto_defaults.parquet").shape[0]),
                "role": "deterministic nonlocal defaults retained when manual escalation is unnecessary",
            },
        ]
    )
    review_promotion_table = pd.DataFrame(
        [
            {
                "lane": "Reference matching",
                "review_workspace": "state/review/queues/local_resolution plus state/review/runs/local_resolution",
                "promotion_or_builder": "python main.py promote-reference-inputs",
                "canonical_surface": "state/reference/inputs/*.parquet",
                "downstream_use": "build-reference-layers consumes only the promoted reference inputs",
            },
            {
                "lane": "Municipal publication extracts",
                "review_workspace": "state/review/packets/municipal_targets/<case_key>",
                "promotion_or_builder": "python main.py promote-local-publications",
                "canonical_surface": "state/modeling/inputs/local_publication/<case_key>/ and local_publication_annual.parquet",
                "downstream_use": "observation/control stages can prefer reviewed local annual rows",
            },
            {
                "lane": "City incident support",
                "review_workspace": "state/review/packets/city/<city_key>",
                "promotion_or_builder": "python main.py promote-city-incident-inputs",
                "canonical_surface": "state/modeling/inputs/city_incident/<city_key>/ and city_incident_share_surface.parquet",
                "downstream_use": "allocation uses promoted city incidents as posterior within-share evidence where gates are active",
            },
            {
                "lane": "State publication supplements",
                "review_workspace": "state/review/packets/source/states/* and source-audit scaffolding",
                "promotion_or_builder": "python main.py build-state-publication-inputs",
                "canonical_surface": "state/modeling/inputs/state_publication_annual.parquet",
                "downstream_use": "observation/control stages can prefer official state publication rows",
            },
        ]
    )
    qa_table = pd.DataFrame(
        [
            {
                "surface": "ags_core",
                "max_abs_state_offense_diff": f"{qa['outputs']['ags_core_max_abs_state_offense_diff']:.3e}",
            },
            {
                "surface": "fbi_calibrated",
                "max_abs_state_offense_diff": f"{qa['outputs']['fbi_calibrated_max_abs_state_offense_diff']:.3e}",
            },
        ]
    )
    feature_stage_table = pd.DataFrame(
        [
            {
                "metric": "Feature inventory rows",
                "value": _count_fmt(len(feature_inventory)),
                "notes": "rows in state/modeling/jurisdiction_model_features_2024.parquet",
            },
            {
                "metric": "Feature inventory columns",
                "value": _count_fmt(feature_inventory.shape[1]),
                "notes": "metadata columns in the retained feature inventory artifact",
            },
            {
                "metric": "Retained benchmark feature count",
                "value": _count_fmt(int(tables["benchmark_overview"]["feature_count"].iloc[0])),
                "notes": "count reported in the live benchmark artifact",
            },
            {
                "metric": "BG prior rows",
                "value": _count_fmt(len(bg_prior)),
                "notes": f"offense-specific BG prior rows in {shipped_bg_prior_rel}",
            },
            {
                "metric": "Training-row range by offense",
                "value": f"{_count_fmt(tables['benchmark_offense_summary']['training_rows'].min())}-{_count_fmt(tables['benchmark_offense_summary']['training_rows'].max())}",
                "notes": "jurisdiction rows per offense used in cross-validated training",
            },
        ]
    )
    share_validation_summary = pd.DataFrame(
        [
            {
                "city_count": int(city_share_json["city_count"]),
                "rows": int(city_share_json["rows"]),
                "incident_total": _count_fmt(city_share_json["incident_total"]),
                "weighted_tvd": _metric_fmt(float(city_share_json["weighted_total_variation_distance_mean"])),
                "weighted_pearson": _metric_fmt(float(city_share_json["weighted_pearson_share_mean"])),
                "weighted_spearman": _metric_fmt(float(city_share_json["weighted_spearman_share_mean"])),
                "weighted_top10_capture": _metric_fmt(float(city_share_json["weighted_top_10pct_true_mass_in_model_top_10pct_mean"])),
            }
        ]
    )
    residual_validation_summary = pd.DataFrame(
        [
            {
                "holdout_cities": int(city_residual_json["holdout_city_count"]),
                "rows": int(city_residual_json["rows"]),
                "incident_total": _count_fmt(city_residual_json["incident_total"]),
                "baseline_weighted_tvd": _metric_fmt(float(city_residual_json["baseline_weighted_total_variation_distance_mean"])),
                "residual_weighted_tvd": _metric_fmt(float(city_residual_json["residual_weighted_total_variation_distance_mean"])),
                "weighted_tvd_delta": _metric_fmt(float(city_residual_json["weighted_tvd_delta"])),
                "improved_rows": int(city_residual_json["improved_tvd_rows"]),
                "worsened_rows": int(city_residual_json["worsened_tvd_rows"]),
            }
        ]
    )
    promoted_allocator_residual_summary = tables.get("promoted_allocator_residual_summary", pd.DataFrame())
    next_phase_decision_summary = tables.get("next_phase_decision_summary", pd.DataFrame())
    next_phase_truth_case_summary = tables.get("next_phase_truth_case_summary", pd.DataFrame())
    next_phase_error_budget_class_summary = tables.get("next_phase_error_budget_class_summary", pd.DataFrame())
    promoted_allocator_preflight_summary = tables.get("promoted_allocator_preflight_summary", pd.DataFrame())
    promoted_allocator_preflight_inputs = tables.get("promoted_allocator_preflight_inputs", pd.DataFrame())
    output_build_manifest_summary = tables.get("output_build_manifest_summary", pd.DataFrame())
    output_validation_summary = tables.get("output_validation_summary", pd.DataFrame())
    promoted_output_artifact_hashes = tables.get("promoted_output_artifact_hashes", pd.DataFrame())
    frontend_snapshot_hash_check = tables.get("frontend_snapshot_hash_check", pd.DataFrame())
    burglary_commercial_gate_summary = tables.get("burglary_commercial_gate_summary", pd.DataFrame())
    burglary_commercial_calibration_summary = tables.get(
        "burglary_commercial_calibration_summary",
        pd.DataFrame(),
    )
    dashboard_neighborhood_lookup_summary = tables.get("dashboard_neighborhood_lookup_summary", pd.DataFrame())
    dashboard_neighborhood_lookup_largest_count_rank_deltas = tables.get(
        "dashboard_neighborhood_lookup_largest_count_rank_deltas",
        pd.DataFrame(),
    )
    external_surface_availability_summary = tables.get("external_surface_availability_summary", pd.DataFrame())

    key_findings = pd.DataFrame(
        [
            {
                "finding": "Jurisdiction benchmark",
                "value": f"CV R² (log rate) = {_metric_fmt(float(bench['overall_cv_r2_log_rate_mean']))}; CV R² (rate) = {_metric_fmt(float(bench['overall_cv_r2_rate_mean']))}",
            },
            {
                "finding": "Train fit",
                "value": f"Mean train R² (log rate) = {_metric_fmt(float(bench['overall_train_r2_log_rate_mean']))}",
            },
            {
                "finding": "City-share benchmark",
                "value": f"Weighted TVD = {_metric_fmt(float(city_share_json['weighted_total_variation_distance_mean']))}; weighted Pearson = {_metric_fmt(float(city_share_json['weighted_pearson_share_mean']))}",
            },
            {
                "finding": "Residual-share benchmark",
                "value": f"Weighted TVD improved by {_metric_fmt(float(city_residual_json['weighted_tvd_delta']))}; improved rows = {int(city_residual_json['improved_tvd_rows'])}",
            },
            {
                "finding": "Promoted allocator diagnostic",
                "value": (
                    f"Next-phase residual weighted TVD = "
                    f"{_metric_fmt(float(promoted_residual_json.get('residual_weighted_total_variation_distance_mean', float('nan'))))}; "
                    f"decision = {next_phase_summary.get('recommended_next_workstream', 'NA')}"
                ),
            },
            {
                "finding": "State reconciliation QA",
                "value": f"AGS core max absolute state/offense diff = {qa['outputs']['ags_core_max_abs_state_offense_diff']:.3e}; missing municipal support = {qa['geometry']['missing_municipal_support_count']}",
            },
        ]
    )

    repro_prefix = package_repo_prefix.strip().strip("/")
    repro_root_label = f"{repro_prefix}/" if repro_prefix else ""
    output_root = _prefix_relpath("state/output", repro_prefix)
    diagnostics_root = _prefix_relpath("scripts/diagnostics", repro_prefix)
    lines: list[str] = []

    _append_block(
        lines,
        f"""
        # CrimeRisk 2024 Public-Data Replication

        ## Executive Summary

        This package documents the final `2024` CrimeRisk public-data replication as a complete
        system, from raw source families through released block-group and tract outputs. The primary
        released product is the `ags_core` surface. The `fbi_calibrated` surface is included as a
        diagnostic derivative output that applies state/offense calibration ratios after the main
        allocation pipeline is complete.

        The build runs in four logical stages. First, raw FBI, Census, and locally reviewed packet
        inputs are converted into a canonical reference layer. Second, source-specific annual and
        monthly crime records are arbitrated into an observed jurisdiction panel, then extended into
        `2024` jurisdiction controls through reporting-regime classification and municipal fill logic.
        Third, those jurisdiction totals are distributed to block groups and tracts using county-anchored
        controls, jurisdiction geometry crosswalks, modeled within-jurisdiction shares, and posterior
        city-share updates where promoted incident feeds are available.
        Fourth, the resulting surfaces are validated against internal control reconciliation, a
        jurisdiction-level predictive benchmark, and benchmark-city incident-truth spatial tests.

        The final benchmark picture is mixed but clear. The jurisdiction model reaches
        `{_metric_fmt(float(bench['overall_cv_r2_log_rate_mean']))}` mean cross-validated `R²` on
        log rates and `{_metric_fmt(float(bench['overall_cv_r2_rate_mean']))}` on raw rates. The
        benchmark-city baseline spatial benchmark records weighted total variation distance
        `{_metric_fmt(float(city_share_json['weighted_total_variation_distance_mean']))}`, and the
        leave-one-city-out residual-share benchmark improves that to
        `{_metric_fmt(float(city_residual_json['residual_weighted_total_variation_distance_mean']))}`.
        QA on the released outputs shows near-exact state/offense reconciliation
        (`{qa['outputs']['ags_core_max_abs_state_offense_diff']:.3e}` maximum absolute difference in
        `ags_core`) and zero missing municipal-support rows.
        """,
    )
    lines.append(_df_to_markdown(key_findings))
    lines.append("")

    _append_block(
        lines,
        """
        ## 0. AGS Comparator And FBI Source Framing

        The current public AGS comparator is the AGS CrimeRisk Methodology 2026A. 2026A preserves the
        same seven legacy crime-category target and 2018-2024 FBI UCR source window, while adding
        NIBRS-specific wording, expanding the described modeling jurisdiction base, broadening local
        incident-data coverage, moving projection geography to Census block, and lowering the public
        jurisdiction-model fit claim from over 85% (the older 2025B language) to over 75%. This
        project targets the same SRS-compatible seven-category surface; it does not attempt a
        reproducible diff against AGS internals.

        The FBI source terms are distinct and are used precisely throughout this report. UCR is the
        umbrella FBI reporting program. SRS / Return A and NIBRS are reporting and data-collection
        systems inside UCR. CIUS / RCN are publication and table products built from UCR data. Current
        traditional CIUS-style tables use SRS data plus NIBRS data converted or summarized into
        SRS-compatible categories. This project targets SRS-compatible seven-category Part I counts,
        with explicit NIBRS-to-legacy conversion wherever raw NIBRS is used (see
        `references/FBI-DATA-GUIDE.md`). It is not a NIBRS-native counting product.
        """,
    )

    _append_block(
        lines,
        f"""
        ## 1. Released Product And Canonical Build Contract

        This report is intentionally written as a reconstruction guide to the final product surface,
        not as a chronological diary of development. A reader should be able to follow the final
        logical flow from raw inputs to outputs, understand where human review enters the system,
        and rerun the canonical build once the expected source-data files are supplied under
        `{repro_root_label}data/`.

        The guiding idea is simple: each stage in the pipeline takes a messy, partially ambiguous
        public-data problem and turns it into a narrower canonical artifact with a clearer contract.
        The report follows that same logic. For each stage it explains four things in order: what
        raw evidence came in, what code and review steps converted that evidence into a canonical
        surface, what the resulting artifact contains, and what diagnostics or caveats matter before
        handing that artifact to the next stage.

        In practice, the phrase `canonical build contract` means that the release pipeline only
        trusts a small set of named surfaces under `state/reference`, `state/observations`,
        `state/modeling`, `state/controls`, `state/geometry`, and `state/output`. Raw files,
        mutable review packets, queue outputs, and one-off exploratory work can influence those
        surfaces, but they do not become part of the live release until they are promoted or built
        into one of those canonical locations.

        Package contents:

        - `README.md`: product-facing repository overview.
        - `report/CrimeRisk_Submission_Report.md`: this report.
        - `materials/tables/`: reusable derived tables for appendices and presentation use.
        - `materials/figures/`: benchmark charts, distributions, and comparison graphics.
        - `materials/maps/`: national and city-level static maps.
        - `OUTPUT_SCHEMA.md`: package-facing data dictionary for the released Parquet outputs.
        - `{repro_root_label}`: reproducibility bundle containing code, configuration, reviewed inputs, final outputs, and the expected raw-data path contract.

        Primary released outputs:

        - `{output_root}/crimerisk_block_group_2024_ags_core.parquet`
        - `{output_root}/crimerisk_tract_2024_ags_core.parquet`
        - `{output_root}/crimerisk_block_group_2024_fbi_calibrated.parquet`
        - `{output_root}/crimerisk_tract_2024_fbi_calibrated.parquet`
        """,
    )
    lines.append(_df_to_markdown(pipeline_stage_table))
    lines.append("")
    if not output_build_manifest_summary.empty or not output_validation_summary.empty:
        _append_block(
            lines,
            """
            The promoted release evidence is carried directly in `state/output`. The manifest records
            the promoted run, copied artifact hashes, candidate validation summary hash, frontend
            snapshot hash check, county anchoring flag, model-prior anchor, and burglary commercial
            calibration. The release validator summary records `ok=true` with zero issues for the
            promoted output directory.
            """
        )
        if not output_build_manifest_summary.empty:
            lines.append(_df_to_markdown(output_build_manifest_summary))
            lines.append("")
        if not output_validation_summary.empty:
            lines.append(_df_to_markdown(output_validation_summary))
            lines.append("")
        if not promoted_output_artifact_hashes.empty:
            lines.append(_df_to_markdown(promoted_output_artifact_hashes))
            lines.append("")
        if not frontend_snapshot_hash_check.empty:
            lines.append(_df_to_markdown(frontend_snapshot_hash_check))
            lines.append("")
        if not burglary_commercial_calibration_summary.empty:
            lines.append(_df_to_markdown(burglary_commercial_calibration_summary))
            lines.append("")
        if not burglary_commercial_gate_summary.empty:
            lines.append(_df_to_markdown(burglary_commercial_gate_summary))
            lines.append("")
    _append_block(
        lines,
        f"""
        Most readers only need the report, the reusable materials, the final output tables, and the
        compact validation summaries under `validation/`. The reproducibility bundle is included so a
        technically competent reader can rerun the pipeline after supplying the expected source data.

        If they want to rebuild the released outputs, they must first supply the external files
        listed in `materials/tables/required_inputs.csv` under the exact `expected_path` values
        shown there. That table is intentionally more specific than the higher-level input inventory:
        it names the concrete crime, covariate, geometry, and city-incident files that are expected
        to live under `{repro_root_label}data/` before the release build can run. Some of those
        files are direct public downloads; some are project-prepared public-data extracts such as
        parsed covariate parquets and normalized city-incident caches.

        After those external inputs are in place, the rebuild command is:

        ```bash
        cd {repro_root_label.rstrip('/') or '.'}
        uv run python main.py build-release --emit-fbi-calibrated
        ```

        Only after `build-release` has regenerated the intermediate state should the package QA
        command be run:

        ```bash
        cd {repro_root_label.rstrip('/') or '.'}
        uv run python scripts/diagnostics/qa_build.py
        ```

        `qa_build.py` is not a validation-only command for the extracted ZIP by itself. It expects
        the rebuilt `state/reference/`, `state/observations/`, `state/controls/`, `state/geometry/`,
        and `state/modeling/` artifacts created by `build-release`, then rewrites
        `state/qa/build_qa_summary.json` and checks mechanical consistency of the rebuilt release.

        For a lightweight check of the extracted package before rebuilding, use:

        ```bash
        cd {repro_root_label.rstrip('/') or '.'}
        uv run python scripts/diagnostics/validate_release_outputs.py
        ```

        That validator reads only included release outputs and compact validation files. It checks
        output schema, geography uniqueness, release coverage, non-negative counts, aggregate count
        identities, and denominator-policy null handling.

        Three upstream points in the build still depend on reviewed promotion surfaces rather than
        blind raw ingestion: reference matching (`promote-reference-inputs`), municipal annual
        publication extracts (`promote-local-publications`), and city incident support files
        (`promote-city-incident-inputs`). Those are the places where the public data are not
        self-explanatory enough to trust a fully automated path. The rest of this report explains
        exactly what those promoted surfaces look like, how they were produced, and how the
        downstream build consumes them.
        """,
    )

    _append_block(
        lines,
        """
        ## 2. Raw Inputs And Canonical Promotion Surfaces

        The build mixes national raw source families with a small number of promoted review surfaces.
        The raw families provide the repeatable national backbone. FBI SRS annual and monthly returns
        provide the long historical summary series. Kaplan NIBRS offense, victim, property, and
        batch-header extracts provide the incident-era federal alternative when the Return A file
        lacks an agency: the build rolls them up to SRS-equivalent annual counts using the FBI's
        documented scoring rules (per-victim person crimes with the revised rape definition,
        per-vehicle motor vehicle theft, hotel-rule burglary), verified against the FBI's own
        converted rows for dual-reporting agencies. FBI CDE state
        estimates provide a state-level comparison surface rather than a direct small-area input.
        Census population updates, ACS demographic tables, employment, road, land-cover, school, and
        hospital layers provide the non-crime covariates used to build the spatial prior.

        Promoted review surfaces enter only where national public data are ambiguous or materially
        incomplete. That happens in four recurring situations: when a reporting agency cannot be
        routed cleanly into the jurisdiction system from raw identifiers alone; when a city or town
        has a usable annual publication that should outrank weaker federal annual counts; when a
        state publishes official offense detail that materially improves the live year; and when a
        city has a sufficiently reviewed incident feed that can refine within-city block-group
        shares. Those are not side channels. They are explicit reviewed inputs with named promotion
        steps and bounded downstream effects.
        """
    )
    lines.append(_df_to_markdown(tables["required_inputs"]))
    lines.append("")
    _append_block(
        lines,
        """
        The required-input table serves a different purpose from the higher-level input inventory
        that follows. The required-input table is a practical rebuild checklist: it names the exact
        external file paths that must exist before `build-release` can succeed. The broader input
        inventory describes the canonical artifacts that the release pipeline ultimately builds or
        consumes at each stage, including promoted intermediate surfaces that are regenerated inside
        the package.
        """
    )
    lines.append(_df_to_markdown(tables["input_inventory"]))
    lines.append("")
    _append_block(
        lines,
        """
        The important design choice is that downstream production stages do not read mutable packet
        trees directly. They consume canonical promoted inputs under `state/reference/inputs/` and
        `state/modeling/inputs/`, but `build-release` rematerializes those canonical surfaces from
        the retained review roots before the dependent downstream stages run. That keeps the release
        build reproducible even when the review workspace contains much more exploratory material
        than the final package should expose. It also means the report can cleanly separate two
        questions that would otherwise get mixed together: how a reviewed input was originally
        produced, and how the final build consumes it once it has been promoted.
        """
    )
    _append_block(
        lines,
        """
        That separation is also how manual research and agent-assisted review enter the system. The
        project uses `state/review/` as a staging workspace for research packets, review queues, and
        compiled batch-run outputs, with the orchestration code living under `scripts/review/`. Those
        surfaces are part of how the inputs were actually produced, but they are not the live
        production contract. A reviewed candidate only becomes part of the release build after an
        explicit promotion or builder step materializes a canonical surface under `state/reference`
        or `state/modeling/inputs`.
        """
    )
    lines.append(_df_to_markdown(review_promotion_table))
    lines.append("")
    _append_block(
        lines,
        """
        In practical terms, the review system uses three patterns. Local reference work moves from
        candidate queue tables to reviewed adjudications. Municipal-target and city lanes move
        through packet files that carry a manifest, status/recommendation file, and the specific
        extracts or crosswalks needed to justify promotion. State publication supplements sit
        between the two: source-audit packets inform the lane, but the live canonical surface is
        built by codified loaders against raw official state sources rather than by copying packet
        files directly.

        That distinction is important for anyone trying to reproduce the project from scratch. The
        review machinery tells you how the team handled ambiguity and how manually researched inputs
        were turned into stable artifacts. The canonical promoted surfaces tell you what the actual
        release build depends on. A faithful reproduction needs both pieces of information, but it
        should never confuse the staging workspace with the live input contract.

        The working repository retains both the promoted canonical inputs and the broader review
        workspace. The submission package keeps only the subset of reviewed input state that the
        canonical rebuild still consumes directly: local-resolution adjudications, municipal-target
        packets, and city packet gates under `repro/state/review/`. `build-release` recreates the
        promoted canonical inputs inside the package repro tree before downstream stages run.
        Broader source-audit packets, review support notes, and run logs remain repo-side
        provenance rather than package-facing rebuild inputs. The `state/review/...` paths are
        described here because they document how the retained rebuild inputs were established before
        the final canonical surfaces were built.

        The input manifest at `state/reference/input_manifest.json` should therefore be read as a raw
        dependency snapshot, not as the sole description of the entire system. The full operational
        contract also includes the promotion steps, covariate artifacts, and stage-level freshness
        rules that live outside that one manifest file.
        """
    )
    lines.append(_df_to_markdown(promoted_surfaces_table))
    lines.append("")

    _append_block(
        lines,
        f"""
        ## 3. Reference Layer: Agencies, Jurisdictions, And Crosswalks

        The reference layer defines the object that the rest of the project estimates: a national
        jurisdiction universe with a stable mapping from reporting agencies to municipal jurisdictions,
        statewide overlap layers, and state remainder layers. In practice this stage does five things:
        it builds the agency master from SRS and NIBRS, separates local from nonlocal agencies, resolves
        local agencies to places or county subdivisions, classifies nonlocal agencies into overlap or
        remainder buckets, and then materializes the agency-to-jurisdiction crosswalk used downstream.

        The final live scale of that layer is:
        """
    )
    lines.append(_df_to_markdown(tables["reference_summary"]))
    lines.append("")
    _append_block(
        lines,
        """
        The released jurisdiction universe is overwhelmingly municipal. There are `14,137` municipal
        jurisdictions, plus `55` state nonmunicipal remainder layers and `55` statewide overlap
        layers. Agency source-presence composition is also stable: every agency in the master has SRS,
        NIBRS, or both, with no `neither` rows in the current live artifact.
        """
    )
    lines.append(_df_to_markdown(tables["reference_jurisdiction_type_counts"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["reference_agency_source_presence"]))
    lines.append("")
    _append_block(
        lines,
        """
        The canonical reference build is driven from the promoted input surfaces under
        `state/reference/inputs/`; during a package rebuild `promote-reference-inputs`
        rematerializes those surfaces from the retained review outputs before
        `build-reference-layers` runs. Those promoted inputs are themselves the product of the
        local-resolution workflow: provisional local matches seed the queue, review batches and
        second-pass sweeps resolve hard cases, and the final reviewed outputs are promoted into the
        canonical input directory before `build-reference-layers` runs.
        """
    )
    lines.append(_df_to_markdown(reference_input_table))
    lines.append("")
    _append_block(
        lines,
        """
        This distinction matters for reproducibility. Recreating the final reference layer does not
        require rerunning the review workers because the retained queue outputs are sufficient for
        `promote-reference-inputs` to recreate the promoted surfaces. But understanding how those
        promoted surfaces were produced does require understanding the queue workflow under
        `scripts/review/local_resolution/` and the corresponding batch-run outputs under
        `state/review/runs/local_resolution/`.
        """
    )
    _append_block(
        lines,
        """
        Local agency resolution is dominated by direct place matching, with county-subdivision
        (`cousub`) fallbacks concentrated in states where township-style local government is the
        correct municipal unit. A small unresolved tail remains as null `resolved_geo_type` rows and
        is handled downstream through nonmunicipal fallback logic rather than by silently inventing
        municipalities.
        """
    )
    lines.append(_df_to_markdown(tables["reference_local_resolution_geo_summary"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["reference_local_resolution_source_summary"].head(10)))
    lines.append("")
    _append_block(
        lines,
        """
        Nonlocal agencies are then bucketed separately. In the live artifact these are mostly special
        jurisdictions, sheriffs, unknown agencies, and state law enforcement entities. Their final
        crosswalk relationships are mostly either exclusive or overlap allocations; only `54`
        crosswalk rows remain unresolved in the current canonical artifact.
        """
    )
    lines.append(_df_to_markdown(tables["reference_nonlocal_agency_type_summary"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["reference_nonlocal_overlap_summary"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["reference_crosswalk_relationship_summary"]))
    lines.append("")
    _append_block(
        lines,
        """
        The handoff into the observation layer is explicit. `build-observations` consumes the agency
        master, the jurisdiction master, and the agency-to-jurisdiction crosswalk, then attaches
        offense counts and metadata to those canonical IDs. That means any ambiguity left in the
        reference layer propagates structurally into observations; it does not disappear later.
        """
    )

    _append_block(
        lines,
        """
        ## 4. Observation Layer: Agency-Year And Jurisdiction-Year Histories

        The observation layer converts source-specific annual and monthly crime records into a single
        `2018-2024` agency-year and jurisdiction-year offense history. Source families are not simply
        stacked. Instead, the build applies a consistent arbitration contract so that direct CIUS or
        promoted-publication counts can replace weaker federal annual totals where appropriate, while
        still preserving provenance metadata about source family, source origin, and raw data source.
        """
    )
    _append_block(
        lines,
        """
        This is the point in the pipeline where heterogeneous reporting systems are forced into one
        language. Some raw inputs arrive as annual offense totals, some arrive with monthly structure
        that later helps classify reporting quality, and some exist only because a municipal or state
        publication was manually reviewed and promoted into canonical form. The observation layer does
        not try to solve every problem at once. Its job is narrower and more important: standardize the
        raw records early, attach them to canonical agencies and jurisdictions, and preserve enough
        provenance that later stages can still tell the difference between direct observation,
        publication-backed replacement rows, and weaker federal annual records.

        A reader trying to reproduce the project should think of this layer as the canonical observed
        history rather than the final answer. The final deliverable is only a `2024` surface, but the
        project cannot build that surface honestly without a historical panel. Reporting regimes,
        partial-year uplifts, trend-based fill methods, and peer estimates all depend on the quality
        and shape of the full `2018-2024` jurisdiction history assembled here.
        """
    )
    lines.append(_df_to_markdown(tables["observation_stage_summary"]))
    lines.append("")
    _agency_panel_rows = int(obs_summary["agency_year_observation_rows"])
    _agency_panel_oris = int(obs_summary["agency_count"])
    _jur_obs_rows = int(obs_summary["jurisdiction_year_observation_rows"])
    _jur_obs_count = int(obs_summary["jurisdiction_count"])
    _reference_jur_count = int(reference_summary["jurisdictions"])
    _append_block(
        lines,
        f"""
        At the agency layer, the current canonical panel contains
        `{_count_fmt(_agency_panel_rows)}` rows across `{_count_fmt(_agency_panel_oris)}`
        ORIs. At the jurisdiction layer, the weighted crosswalk rollup produces
        `{_count_fmt(_jur_obs_rows)}` rows across `{_count_fmt(_jur_obs_count)}`
        jurisdictions. The reference layer has `{_count_fmt(_reference_jur_count)}`
        jurisdictions, so `{_reference_jur_count - _jur_obs_count}` reference
        jurisdictions do not appear in the jurisdiction-observation surface at all; they only emerge
        later if the control stage has to carry them as zero-mass or estimated rows.

        The `2024` agency and jurisdiction source mix illustrates the final source ordering in live
        use:
        """
    )
    lines.append(_df_to_markdown(tables["agency_observation_source_2024"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["jurisdiction_observation_source_2024"]))
    lines.append("")
    _append_block(
        lines,
        """
        These `2024` source mixes show the architecture of the live panel clearly. CIUS and SRS
        provide most of the usable direct annual rows. The NIBRS SRS-equivalent rollup fills
        agencies the Return A file lacks. The promoted publication lanes remain
        deliberately small in row count, but they are high-leverage rows: they exist precisely where a
        reviewed local or state publication is better than the federal annual backbone for the released
        year.

        The agency panel then rolls into the jurisdiction panel through the reference crosswalk. That
        rollup is where local agency matching, overlap handling, and statewide remainder logic start to
        have quantitative consequences. If the reference layer misclassifies an agency, the observation
        layer will faithfully propagate that mistake. This is why the report treats reference building
        and observation building as one continuous contract rather than as isolated preprocessing jobs.
        """
    )
    _obs_share = tables["observed_share_by_year"].set_index("year")[
        "observed_estimated_count_share"
    ]
    _append_block(
        lines,
        f"""
        Observed coverage is strongest at the edges of the panel and weakest during the federal
        SRS-to-NIBRS transition. The key quantity is not just the share of rows marked observed, but
        the share of estimated count volume coming from rows that are usable as observed. That measure
        is `{100 * float(_obs_share.loc[2018]):.2f}%` in `2018`, falls to
        `{100 * float(_obs_share.loc[2021]):.2f}%` in `2021`, and recovers to
        `{100 * float(_obs_share.loc[2024]):.2f}%` in `2024`.
        """
    )
    lines.append("![Observed share by year](../materials/figures/observed_share_by_year.png)")
    lines.append("")
    lines.append(_df_to_markdown(tables["observed_share_by_year"]))
    lines.append("")
    _append_block(
        lines,
        """
        The implication for the rest of the build is straightforward. Jurisdictions with clean direct
        observations flow through mostly unchanged. Jurisdictions that land in the low-coverage middle
        years require stronger regime-aware estimation support in the control stage. That is why the
        project treats reporting regimes and municipal fill logic as a first-class stage rather than a
        small correction.
        """
    )
    _append_block(
        lines,
        """
        Put differently, this layer tells the rest of the project both what was observed and where the
        observed history is too weak to stand on its own. Later stages do not discover those problems
        from scratch. They inherit them from the coverage, provenance, and transition-era gaps that are
        already visible here.
        """
    )

    _append_block(
        lines,
        """
        ## 5. Reporting Regimes And Official Publication Promotion Lanes

        The control system does not treat all observed annual rows equally. It first classifies each
        agency-year into a reporting regime: `full_monthly`, `true_partial`, `annual_only_but_usable`,
        `lumpy_or_batched`, or `structurally_missing_or_unreliable`. That regime metadata determines
        whether a row can be used directly, whether a month-ratio uplift is needed, or whether the
        jurisdiction has to be estimated from history or peers.
        """
    )
    _append_block(
        lines,
        """
        This stage answers a more demanding question than "is there an annual number?" It asks what
        kind of annual number the row actually is. A twelve-month reporting record, a plausible
        nine-month partial, a bare annual-only total, and a lumpy batch-loaded return all look
        different once they are used as controls. The regime classifier separates those cases because
        the downstream control logic should not trust them equally.

        The labels are operational. `full_monthly` rows are strong direct evidence. `true_partial`
        rows remain useful, but only after an explicit coverage-based uplift. `annual_only_but_usable`
        rows are kept because their annual signal is still worth more than a generic peer estimate.
        `lumpy_or_batched` marks rows whose monthly shape looks artificial. `structurally_missing_or_unreliable`
        means the project should stop pretending the current-year direct observation is good enough and
        instead move to historical or peer-based estimation.
        """
    )
    lines.append(_df_to_markdown(tables["agency_reporting_regime_2024"]))
    lines.append("")
    _append_block(
        lines,
        """
        Two promoted official-publication lanes enter here.

        `local_publication_annual` is a narrow, human-reviewed municipal supplement. In the live build
        it contains `54` offense rows covering five jurisdictions: Atlanta, Brunswick, Hallandale
        Beach, New Orleans, and Quincy.

        `state_publication_annual` is a larger agency-level supplement. In the live build it contains
        `1,799` offense rows across `286` agencies in Florida and Mississippi and exists precisely
        because those states expose useful offense detail that is not fully captured by the federal
        annual surfaces.
        """
    )
    _append_block(
        lines,
        """
        The manual-review component matters because these publications are not generic "extra data."
        The local lane exists only where a packet assembled a credible municipal extract, offense
        mapping, and promotion recommendation. The state lane exists only where the project built and
        validated a stable loader against an official state publication source and confirmed that the
        resulting rows fit the live offense ontology. These are bounded, reviewed exceptions admitted
        into the canonical system under a stricter contract than the broad federal raw feeds.
        """
    )
    _append_block(
        lines,
        """
        The two lanes reach the live build differently.

        The local-publication lane is packet-driven. Each municipal packet under
        `state/review/packets/municipal_targets/<case_key>/` carries a `packet_manifest.json`,
        `recommendation.csv`, and, when the packet is promotable, one or more extract CSVs such as
        `published_reference_extract.csv` or `*_offense_extract.csv`. The promotion step copies only
        production-ready packet artifacts into `state/modeling/inputs/local_publication/<case_key>/`
        during the rebuild, then consolidates those canonical packet inputs into
        `local_publication_annual.parquet`. The live build reads the canonical input root and the
        consolidated parquet, not the mutable review packet tree.

        The state-publication lane is loader-driven. Research and source-audit packets exist under
        `state/review/packets/source/states/*`, but the live canonical surface is written by
        `build-state-publication-inputs`, which fetches or reads the raw official state exports and
        normalizes them directly into `state_publication_annual.parquet`. In the current release that
        loader path is active only for Florida and Mississippi.
        """
    )
    lines.append(_df_to_markdown(tables["promoted_local_publication_summary"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["promoted_state_publication_summary"]))
    lines.append("")
    _append_block(
        lines,
        """
        These promotion lanes are deliberately small. They are not a second parallel build system.
        They are bounded canonical inputs with explicit downstream roles: the state-publication surface
        is consumed during observation building, while both state and local publication surfaces can
        become preferred direct sources inside the `2024` controls layer.
        """
    )
    _append_block(
        lines,
        """
        For reproducibility, the important distinction is between reviewed staging material and live
        canonical inputs. The packets explain how a human or agent-assisted review justified promotion.
        The promoted parquets are what the build actually consumes after `build-release` regenerates
        them. Recreating the live package from raw data therefore requires understanding both
        layers, but rerunning the package rebuild does not require new manual review because the
        retained packet roots are sufficient to rematerialize those canonical local/state input
        surfaces first.
        """
    )

    _append_block(
        lines,
        f"""
        ## 6. Municipal Estimation And `2024` Controls

        The control stage is where the project stops being a historical panel and becomes a released
        `2024` surface. `build-municipal-estimates` first generates regime-aware estimates for
        municipalities that are missing, unreliable, or only partially reported in `2024`. Then
        `build-controls` arbitrates among direct CIUS counts, local publications, state publications,
        SRS, NIBRS, and the municipal-estimation surface to produce one final `2024` jurisdiction row
        per offense.

        The municipal estimation surface contains `98,616` offense rows across `14,088`
        municipalities. The dominant estimate source is still direct usable reporting, but the fill
        tail is large enough to matter because the reporting-transition years left substantial gaps.
        """
    )
    _append_block(
        lines,
        """
        This is the substantive center of the entire system. Up to this point the project has been
        constructing evidence. Here it has to make one final decision for every jurisdiction/offense
        pair in the release year: keep a direct observed total, adjust a partially observed total, or
        estimate the row because the direct evidence is missing or unreliable. The output of this stage
        is the national `2024` control table that every later spatial step must respect exactly.

        The fill ladder is intentionally conservative. Direct usable reporting comes first whenever the
        source and regime contract say it is trustworthy. True partial rows are uplifted instead of
        discarded, because a mostly observed jurisdiction-specific row is often better than any generic
        peer estimate. Historical medians, state-by-population-band peers, and log-linear trends are
        then used only when the current-year direct evidence is too weak. That ordering keeps the final
        controls anchored to observed local behavior wherever possible rather than letting estimation
        overwhelm rows that still contain real signal.
        """
    )
    lines.append(_df_to_markdown(tables["jurisdiction_target_source_2024"]))
    lines.append("")
    _src_share = {
        str(row["preferred_source"]): float(row["adjusted_count_share_pct"])
        for _, row in preferred_source.iterrows()
    }
    _append_block(
        lines,
        f"""
        The final `2024` controls table contains `{_count_fmt(summary['controls_rows'])}` offense rows
        across `{_count_fmt(summary['control_unique_jurisdictions'])}` jurisdictions. Direct CIUS
        annual publications dominate adjusted volume
        (`{_src_share.get('cius_publication_annual', 0.0):.2f}%`), followed by SRS annual returns
        (`{_src_share.get('srs_return_a_annual', 0.0):.2f}%`). The promoted state-publication lane
        contributes `{_src_share.get('state_publication_annual', 0.0):.2f}%` of adjusted volume, the
        promoted local-publication lane `{_src_share.get('local_publication_annual', 0.0):.2f}%`, and
        the NIBRS SRS-equivalent rollup `{_src_share.get('nibrs_srs_equivalent_annual', 0.0):.2f}%`.
        """
    )
    _append_block(
        lines,
        """
        The preferred-source and estimate-source views answer different but complementary questions.
        Preferred source tells us which direct source won the arbitration for the row. Estimate source
        tells us what mechanism actually produced the released adjusted count after uplifts and fallbacks
        are taken into account. Preferred source is the clean provenance answer; estimate source is the
        operational answer a replicator needs in order to understand why the control row looks the way
        it does.
        """
    )
    lines.append("![Preferred source share](../materials/figures/controls_preferred_source_adjusted_share_2024.png)")
    lines.append("")
    lines.append(_df_to_markdown(preferred_source[["preferred_source", "row_count", "jurisdiction_count", "adjusted_count_share_pct"]]))
    lines.append("")
    _append_block(
        lines,
        """
        Preferred-source origin is intentionally similar to preferred source, but the estimate-source
        view adds the downstream adjustment logic. That is where the partial-uplift and low-confidence
        current-year-fill rows become visible as separate mechanisms rather than being hidden inside a
        single adjusted total.
        """
    )
    lines.append(_df_to_markdown(preferred_origin[["preferred_source_origin", "row_count", "jurisdiction_count", "adjusted_count_share_pct"]]))
    lines.append("")
    lines.append("![Estimate source share](../materials/figures/controls_estimate_source_adjusted_share_2024.png)")
    lines.append("")
    lines.append(_df_to_markdown(estimate_source[["estimate_source", "row_count", "jurisdiction_count", "adjusted_count_share_pct"]]))
    lines.append("")
    lines.append("![Reporting regime share](../materials/figures/controls_reporting_regime_adjusted_share_2024.png)")
    lines.append("")
    lines.append(_df_to_markdown(reporting_regime[["dominant_reporting_regime", "row_count", "jurisdiction_count", "adjusted_count_share_pct"]]))
    lines.append("")
    _append_block(
        lines,
        """
        The state-control comparison is the last major diagnostic before spatial allocation. It compares
        AGS-core reported totals and AGS-core adjusted totals with the FBI CDE state estimates. The
        AGS-core output is reconciled to the internal adjusted totals. The diagnostic `fbi_calibrated`
        output later applies state/offense calibration ratios derived from this same comparison.
        """
    )
    lines.append("![Adjusted versus CDE](../materials/figures/state_adjusted_vs_cde_scatter.png)")
    lines.append("")
    lines.append(_df_to_markdown(tables["state_control_comparison_summary"]))
    lines.append("")
    _null_est_rows = int(
        pd.read_parquet(
            repo_root / "state" / "controls" / "jurisdiction_controls_2024.parquet",
            columns=["estimate_source"],
        )["estimate_source"]
        .isna()
        .sum()
    )
    _append_block(
        lines,
        f"""
        One live caveat should be stated explicitly. `{_count_fmt(_null_est_rows)}` control rows
        still have null `estimate_source` and `estimate_confidence`, but they are all zero-mass
        rows under `structurally_missing_or_unreliable`; they do not move the
        released totals. In other words, they are metadata residue rather than a hidden count leak.

        This control table is the direct handoff to the spatial model and allocation stage. Once it is
        built, every downstream quantity is about how to distribute those jurisdiction totals, not how
        to change them.
        """
    )
    _append_block(
        lines,
        """
        That handoff principle should shape how the whole package is read. After
        `jurisdiction_controls_2024.parquet` is written, the remaining modeling problem is no longer
        "what should the national crime total be?" It becomes "where inside each jurisdiction should
        that fixed total go?" The spatial model can reshape internal distribution, but it is not
        allowed to silently rewrite the jurisdiction totals established here.
        """
    )

    _append_block(
        lines,
        """
        ## 7. Covariates And The Jurisdiction Model

        The spatial model begins from a national block-group covariate frame. ACS block groups are
        aligned to `2024` population updates and merged with tract context, LODES workplace counts,
        roads, HPMS metrics, NCES school anchors, CMS hospital anchors, and NLCD land cover. The
        package retains the selected feature inventory and the offense-specific BG prior artifacts;
        the wider intermediate feature frame is generated inside the build rather than preserved as a
        canonical release artifact.
        """
    )
    _append_block(
        lines,
        """
        The model stage is easy to misunderstand if it is described only as "fit a national model."
        The model is not asked to invent national totals. It is asked to learn how crime rates vary
        across jurisdictions as a function of public covariates, then to use that learned structure to
        generate relative block-group prior mass inside each jurisdiction. That is why the benchmark is
        jurisdiction-level even though the final deliverable is block-group and tract level: the model
        is trained on jurisdiction outcomes and deployed as a within-jurisdiction spatial prior.

        The package keeps the selected feature inventory and the offense-specific prior surface rather
        than freezing every transient feature matrix produced during joins. That is a deliberate
        reproducibility choice. The stable contract is the source families, the builder logic, the
        retained inventory, and the benchmark artifacts, not a single giant intermediate table that
        would mostly duplicate information already reconstructible from the included sources.
        """
    )
    lines.append(_df_to_markdown(feature_stage_table))
    lines.append("")
    _append_block(
        lines,
        f"""
        The released jurisdiction benchmark reports `288` retained feature columns for the canonical
        build. The model family is histogram gradient boosting (`hist_gbm`) trained separately by
        offense with `min_training_population = 10,000`. The target is jurisdiction-level crime rate
        behavior, but the model is used downstream to produce relative prior mass at the block-group
        level after aggregating block-group covariates into the jurisdiction training buckets.
        """
    )
    lines.append(_df_to_markdown(tables["benchmark_overview"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["benchmark_offense_summary"]))
    lines.append("")
    lines.append("![CV log-rate fit by offense](../materials/figures/benchmark_cv_r2_log_rate_by_offense.png)")
    lines.append("")
    lines.append("![Train versus CV fit](../materials/figures/benchmark_train_vs_cv_r2_log_rate.png)")
    lines.append("")
    lines.append("![Log-rate versus raw-rate CV fit](../materials/figures/benchmark_cv_log_vs_rate_r2.png)")
    lines.append("")
    lines.append("![Calibration curves](../materials/figures/benchmark_calibration_curves.png)")
    lines.append("")
    lines.append("![Residuals by population band](../materials/figures/benchmark_abs_residual_by_population_band.png)")
    lines.append("")
    _bench_by_offense = (
        tables["benchmark_offense_summary"].set_index("offense")["cv_r2_log_rate"].astype(float)
    )
    _bench_best = str(_bench_by_offense.idxmax())
    _bench_worst = str(_bench_by_offense.idxmin())
    _append_block(
        lines,
        f"""
        The model is useful, but the report should not oversell it. The strongest offense in the
        current benchmark is {_bench_best.replace('_', ' ')}
        (`{_bench_by_offense.max():.4f}` cross-validated `R²` on log rate), while
        {_bench_worst.replace('_', ' ')} is the weakest (`{_bench_by_offense.min():.4f}`). Mean train
        fit (`{float(bench['overall_train_r2_log_rate_mean']):.4f}`) remains far above mean
        cross-validated fit (`{float(bench['overall_cv_r2_log_rate_mean']):.4f}`), which is
        exactly why the final narrative should treat the model as a transparent
        public-data approximation rather than as parity with proprietary AGS performance claims.
        """
    )
    _append_block(
        lines,
        """
        This section also explains an important release decision. Experimental feature families are
        not promoted because they are novel or because they resemble any outside commercial surface.
        They are promoted only when the held-out diagnostics justify a concrete production role. The
        current build promotes the Overture destination and commercial-core features for the residual
        allocator because the expanded city-truth benchmark shows a lower weighted TVD under the
        leave-one-jurisdiction-out residual-share diagnostic, and the output manifest records that
        promoted allocator path explicitly.
        """
    )

    _append_block(
        lines,
        f"""
        ## 8. Geometry, City Truth, And Allocation

        Once the jurisdiction totals and BG priors are available, the project needs a geometry contract
        that says which block groups belong to which jurisdictions. The live geometry build starts at
        the census-block level, assigning blocks either by direct municipal point-in-polygon match,
        by smallest-polygon overlap recovery when the point join misses, or to a state remainder when
        no municipal match exists. It then aggregates those block assignments to the block-group
        crosswalk used by the allocator.
        """
    )
    _append_block(
        lines,
        """
        This stage is the spatial equivalent of the control table. The control stage decides how much
        crime each jurisdiction has. The geometry stage decides which block groups are even eligible to
        receive that mass. If the geometry contract is wrong, the allocator can be perfectly coded and
        still produce a misleading map because it will be spreading a correct jurisdiction total across
        the wrong support.
        """
    )
    lines.append(_df_to_markdown(tables["geometry_crosswalk_summary"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["geometry_block_assignment_summary"]))
    lines.append("")
    _append_block(
        lines,
        f"""
        The live geometry surface now carries an exact BG partition. The canonical
        `block_group_to_jurisdiction_crosswalk.parquet` artifact has `239,052` block groups, and the
        current normalized build yields `0` non-unit `allocation_share` sums. That matters because the
        within-jurisdiction allocator now works from a clean partitioned support surface rather than
        relying on downstream renormalization to clean up mixed-basis crosswalk drift.

        The city incident share surface is a separate within-jurisdiction refinement layer, not a
        replacement jurisdiction control system. In the live `2024` build it contributes
        `{_count_fmt(len(city_share_surface_2024))}` block-group share rows covering
        `{_count_fmt(city_share_surface_2024[['city_name', 'offense']].drop_duplicates().shape[0])}`
        city/offense pairs across `{_count_fmt(city_share_surface_2024['city_name'].nunique())}`
        benchmark-grade cities. Share normalization within active
        city/offense pairs is exact up to floating-point noise.
        """
    )
    _append_block(
        lines,
        """
        That distinction is central. The city incident layer does not create a second jurisdiction
        control system, and it does not redefine precincts or neighborhood incident clusters as the
        national unit of truth. Instead, it supplies stronger evidence about how a jurisdiction's
        already-fixed total should be distributed internally in the small set of cities where
        benchmark-grade incident truth is available.
        """
    )
    _append_block(
        lines,
        """
        This lane is explicitly packet-gated. City packets under
        `state/review/packets/city/<city_key>/` carry `packet_status.csv`,
        `packet_offense_status.csv`, `offense_crosswalk.csv`, and, when relevant,
        `published_reference_extract.csv`. Review/scaffolding utilities under `scripts/review/city/`
        synchronize those packet artifacts with the live city-source configs, validate whether each
        city and offense is production-ready, and write the gate files that promotion can freeze.
        `promote-city-incident-inputs` then copies only the canonical gate files into
        `state/modeling/inputs/city_incident/<city_key>` during the rebuild.
        `build-city-incident-shares` reads that promoted canonical input root plus the raw city
        incident data under `data/city-incidents/`; it does not read directly from the mutable
        packet tree.
        """
    )
    lines.append(_df_to_markdown(city_by_city[["city_name", "incident_total", "rows"]]))
    lines.append("")
    _append_block(
        lines,
        """
        Final allocation then combines jurisdiction controls, the offense-specific BG prior, the
        jurisdiction crosswalk, and the promoted residual-share refinement. The residual allocator
        learns from held-out city-truth surfaces, including Overture destination and commercial-core
        block-group features, then applies that learned within-jurisdiction adjustment before
        packet-reviewed direct city incident feeds update posterior shares in active city/offense cells.
        Tract outputs are not modeled independently; they are exact rollups of the final block-group
        surface.

        The handoff into validation is therefore clean. QA checks whether released outputs reconcile to
        the control layer, the jurisdiction benchmark checks the learned jurisdiction model, and the
        city benchmarks check whether the final within-city spatial distribution is plausible against
        incident-truth surfaces.
        """
    )
    _append_block(
        lines,
        """
        In practical terms, allocation is a constrained weighting problem. The BG prior gives each
        block group and offense a baseline relative weight. The geometry crosswalk says which block
        groups can receive each jurisdiction's mass. The promoted residual layer adjusts those
        internal shares where the held-out diagnostics support doing so. The allocator then rescales
        the selected weights so that every final jurisdiction/offense total exactly matches the
        control row built earlier. City posterior updates change only the internal weight pattern for
        gated city/offense pairs; they do not change the jurisdiction totals themselves.
        """
    )

    if not output_build_manifest_summary.empty or not promoted_allocator_preflight_summary.empty:
        _append_block(
            lines,
            """
            The promoted allocator has its own production evidence rather than relying on same-city
            same-city after-posterior comparisons. The preflight verifies that the expanded validation surface and
            both promoted Overture feature artifacts are present. The output build manifest then
            records whether the promoted allocator was enabled and applied, what validation-case types
            were excluded from residual training, and which feature artifacts entered the build.
            """
        )
        if not promoted_allocator_preflight_summary.empty:
            lines.append(_df_to_markdown(promoted_allocator_preflight_summary))
            lines.append("")
        if not promoted_allocator_preflight_inputs.empty:
            lines.append(_df_to_markdown(promoted_allocator_preflight_inputs))
            lines.append("")
        if not output_build_manifest_summary.empty:
            lines.append(_df_to_markdown(output_build_manifest_summary))
            lines.append("")

    _append_block(
        lines,
        """
        ## 9. Validation: Build QA And Jurisdiction Benchmark

        The first validation layer is mechanical rather than predictive. `scripts/diagnostics/qa_build.py`
        verifies that state/offense output totals reconcile to the control tables, that tract rollups
        match block-group rollups, and that municipal jurisdictions in the control table all have
        geometry support in the spatial layer.
        """
    )
    _append_block(
        lines,
        """
        This validation layer exists to answer the most basic reproducibility question first: did the
        released outputs actually respect the build contract? A predictive benchmark is not meaningful
        if the released tables fail to reconcile to their own controls or if tract outputs do not roll
        up cleanly from block groups. Mechanical correctness therefore comes before any discussion of
        predictive skill.
        """
    )
    lines.append(_df_to_markdown(qa_table))
    lines.append("")
    _append_block(
        lines,
        f"""
        The current release passes those checks comfortably. The AGS-core surface has maximum absolute
        state/offense reconciliation error `{qa['outputs']['ags_core_max_abs_state_offense_diff']:.3e}`;
        the FBI-calibrated diagnostic derivative is similarly tight at
        `{qa['outputs']['fbi_calibrated_max_abs_state_offense_diff']:.3e}`; and the final missing
        municipal-support count is `{qa['geometry']['missing_municipal_support_count']}`.

        The second validation layer is predictive: the jurisdiction benchmark asks how well the
        retained covariates explain held-out jurisdiction crime rates. Overall values are shown below,
        followed by offense-level detail.
        """
    )
    lines.append(_df_to_markdown(tables["benchmark_overview"]))
    lines.append("")
    lines.append(_df_to_markdown(tables["benchmark_offense_summary"]))
    lines.append("")
    _append_block(
        lines,
        """
        The correct interpretation is that the public-data model is informative but materially weaker
        than AGS-style proprietary claims. The benchmark is not a failure, but it also does not justify
        language suggesting that the public-data replication explains most jurisdiction-level variance.
        """
    )
    _append_block(
        lines,
        """
        A careful reader should therefore separate two judgments. The build is mechanically coherent;
        the QA outputs establish that directly. The model is also genuinely informative; the held-out
        jurisdiction benchmark is far above zero. But the gap between train and held-out fit shows that
        public covariates alone do not yield a near-complete explanation of crime variation, and the
        thesis should say that plainly.
        """
    )

    _append_block(
        lines,
        f"""
        ## 10. Validation: City-Share And Residual-Share Benchmarks

        The benchmark-city share evaluation is the project’s main external spatial reality check. It
        compares the baseline model prior’s within-city block-group shares with incident-derived truth
        surfaces across `{_count_fmt(city_share_json['city_count'])}` benchmark cities. The canonical
        benchmark currently spans `{_count_fmt(city_share_json['incident_total'])}` incidents and
        `{_count_fmt(city_share_json['rows'])}` city/offense evaluation rows.
        """
    )
    _append_block(
        lines,
        """
        This benchmark answers a different question from the jurisdiction benchmark. The jurisdiction
        benchmark asks whether the model captures broad rate variation across jurisdictions. The city
        benchmark asks whether the final mapped surface places crime in plausible locations inside
        large, heterogeneous cities. Those are related questions, but they are not interchangeable. A
        model can look acceptable at the jurisdiction level and still distribute mass poorly within a
        city.

        Weighted total variation distance (TVD) is the most important metric here because it can be
        interpreted as the share mass that would have to move across block groups for the predicted and
        incident-truth city distributions to match. Pearson and Spearman provide level and rank
        perspectives, while top-10 capture asks whether the model concentrates enough mass in the
        highest-incident block groups. Taken together, these metrics give a much more grounded read of
        spatial quality than a single correlation would.
        """
    )
    lines.append(_df_to_markdown(share_validation_summary))
    lines.append("")
    lines.append("![City-share TVD heatmap](../materials/figures/city_share_tvd_heatmap.png)")
    lines.append("")
    lines.append("![City-share Pearson heatmap](../materials/figures/city_share_pearson_heatmap.png)")
    lines.append("")
    lines.append(_df_to_markdown(city_by_city))
    lines.append("")
    lines.append(_df_to_markdown(city_share_by_offense[["offense_label", "incident_total", "rows", "weighted_total_variation_distance_mean", "weighted_pearson_share_mean", "weighted_spearman_share_mean"]]))
    lines.append("")
    _append_block(
        lines,
        """
        These results are heterogeneous. Seattle and Austin are among the better city fits, while
        Boston is currently the weakest city-level benchmark in the live package. Offense difficulty is
        also uneven: murder and rape are the noisiest categories in the current truth set, while
        larceny and motor vehicle theft are materially more stable.

        The residual-share benchmark is a held-out validation test for the live residual-share
        refinement. It asks whether a learned residual-share model can improve within-city allocation
        on held-out cities relative to the baseline prior. That distinction matters. The benchmark is
        measured on held-out cities, while direct city incident posterior updates are applied only where a
        city/offense cell has passed packet-level review.
        """
    )
    lines.append(_df_to_markdown(residual_validation_summary))
    lines.append("")
    lines.append("![Residual TVD delta heatmap](../materials/figures/city_residual_tvd_delta_heatmap.png)")
    lines.append("")
    lines.append("![Residual top-10 capture delta](../materials/figures/city_residual_top10_delta_heatmap.png)")
    lines.append("")
    lines.append("![Residual before/after scatter](../materials/figures/city_residual_before_after_tvd_scatter.png)")
    lines.append("")
    lines.append(_df_to_markdown(city_resid_by_city))
    lines.append("")
    lines.append(_df_to_markdown(city_resid_by_offense[["offense_label", "incident_total", "rows", "baseline_weighted_total_variation_distance_mean", "residual_weighted_total_variation_distance_mean", "weighted_tvd_delta"]]))
    lines.append("")
    _append_block(
        lines,
        f"""
        In the live held-out benchmark the residual evaluation improves weighted TVD from
        `{_metric_fmt(float(city_residual_json['baseline_weighted_total_variation_distance_mean']))}` to
        `{_metric_fmt(float(city_residual_json['residual_weighted_total_variation_distance_mean']))}`,
        improves weighted Pearson from
        `{_metric_fmt(float(city_residual_json['baseline_weighted_pearson_share_mean']))}` to
        `{_metric_fmt(float(city_residual_json['residual_weighted_pearson_share_mean']))}`, and
        improves `{int(city_residual_json['improved_tvd_rows'])}` of
        `{int(city_residual_json['rows'])}` benchmark rows. That is the validation basis for keeping
        the residual-share refinement live between the generic block-group prior and direct
        city/offense posterior updates.
        """
    )
    if not promoted_allocator_residual_summary.empty or not next_phase_decision_summary.empty:
        _append_block(
            lines,
            """
            The next-phase diagnostic broadens that validation surface and separates total prediction
            error from allocation error without treating the raw block-group prediction as an additive
            decomposition. A block-group estimate is the product of a jurisdiction total and a
            within-jurisdiction spatial share. The diagnostic therefore holds one component fixed at
            truth while varying the other: `total_l1_error = abs(Y_hat - Y_ucr)`,
            `allocation_l1_error = 2 * Y_ucr * TVD`, and
            `allocation_moved_mass = Y_ucr * TVD`. Dominance classification uses the L1 allocation
            error; prose uses moved mass because it is easier to interpret as incidents that would
            need to move across block groups.
            """
        )
        if not next_phase_truth_case_summary.empty:
            lines.append(_df_to_markdown(next_phase_truth_case_summary))
            lines.append("")
        if not next_phase_error_budget_class_summary.empty:
            lines.append(_df_to_markdown(next_phase_error_budget_class_summary))
            lines.append("")
        if not promoted_allocator_residual_summary.empty:
            lines.append(_df_to_markdown(promoted_allocator_residual_summary))
            lines.append("")
        if not next_phase_decision_summary.empty:
            lines.append(_df_to_markdown(next_phase_decision_summary))
            lines.append("")
        _append_block(
            lines,
            f"""
            The current expanded diagnostic covers
            `{_count_fmt(next_phase_summary.get('truth_case_count', float('nan')))}` direct truth
            cases, including promoted large-city controls, mid-size municipal validation cases, a
            partial-year municipal case, and Montgomery County as a suburban county validation case.
            The held-out total predictions are labeled by fold type; `leave_large_city_out` is treated
            as a large-city cold-start stress test, while `leave_one_city_out` /
            `leave_one_jurisdiction_out` is the cleaner per-jurisdiction counterfactual. On the
            current evidence the decision table recommends
            `{next_phase_summary.get('recommended_next_workstream', 'NA')}` because
            {next_phase_decision_rationale}.
            """
        )
    _append_block(
        lines,
        """
        This still does not license circular evaluation. The final released outputs in benchmark
        cities also include direct city incident posterior updates where those are active, so the correct
        independent external checks remain the baseline city-share benchmark and the leave-one-city-out
        residual benchmark rather than a same-city after-posterior comparison.
        """
    )

    if not dashboard_neighborhood_lookup_summary.empty:
        _append_block(
            lines,
            """
            The consumer-facing dashboard check now uses the archived tract-to-neighborhood lookup
            rather than a small polygon smoke sample. It aggregates the released tract surface and the
            current dashboard tract table to the same neighborhood names. This is not a truth-label
            validation; it is an early integration test that asks whether the new crime layer can be
            inspected in the product's current neighborhood frame, and whether it is merely restating
            the dashboard's existing coarse risk score.
            """
        )
        lines.append(_df_to_markdown(dashboard_neighborhood_lookup_summary))
        lines.append("")
        _append_block(
            lines,
            """
            The low rank correlation is expected and useful: the current dashboard score mixes broad
            livability and hazard variables, while CrimeRisk is a specific crime layer. Large rank
            disagreements are therefore review candidates for product interpretation, not automatic
            model failures.
            """
        )
        if not dashboard_neighborhood_lookup_largest_count_rank_deltas.empty:
            lines.append(
                _df_to_markdown(
                    dashboard_neighborhood_lookup_largest_count_rank_deltas[
                        [
                            col
                            for col in [
                                "neighborhood_name",
                                "expected_count_total",
                                "dashboard_risk_score_area_weighted",
                                "crimerisk_expected_count_total_rank_desc",
                                "dashboard_risk_score_rank_desc",
                                "crimerisk_expected_count_rank_minus_dashboard_rank",
                            ]
                            if col in dashboard_neighborhood_lookup_largest_count_rank_deltas.columns
                        ]
                    ]
                )
            )
            lines.append("")

    if not external_surface_availability_summary.empty:
        _append_block(
            lines,
            """
            The external commercial-surface comparison harness is implemented, but the package does
            not include a licensed AGS, Esri Crime Indexes, or CAP CRIMECAST national output surface.
            The availability audit records the local search result and the public product/documentation
            pages checked. When an exported surface is supplied, the comparison path is
            `scripts/diagnostics/benchmark_external_surface.py`, which scores the supplied surface
            against observed incident shares rather than against our released surface.
            """
        )
        lines.append(_df_to_markdown(external_surface_availability_summary))
        lines.append("")

    _append_block(
        lines,
        """
        ## 11. Final Surface Characterization And Derivative Calibration

        The released surface is scoped to the 48 contiguous states plus DC. The table below
        summarizes release coverage and national offense totals and rates for both the primary
        AGS-core product and the diagnostic FBI-calibrated variant.
        """
    )
    lines.append(_df_to_markdown(tables["output_release_coverage"]))
    lines.append("")
    _append_block(
        lines,
        """
        The two released surfaces should be read asymmetrically. `ags_core` is the primary output
        because it stays faithful to the project's own jurisdiction-control system. `fbi_calibrated` is
        a diagnostic sensitivity surface showing what happens when those already-allocated outputs are
        scaled toward FBI CDE state/offense totals after the main build has finished. It is useful for
        interpretation, but it is not a second independently estimated model.
        """
    )
    lines.append(_df_to_markdown(output_summary))
    lines.append("")
    _append_block(
        lines,
        """
        The AGS-core surface is the main product because it stays faithful to the internally built
        jurisdiction controls. The FBI-calibrated diagnostic surface is useful as a comparison because it shows
        how sensitive the national surface is to forcing state/offense totals toward the FBI CDE
        estimates after allocation.
        """
    )
    lines.append("![AGS-core versus FBI-calibrated totals](../materials/figures/fbi_calibrated_vs_ags_core_offense_totals.png)")
    lines.append("")
    lines.append(_df_to_markdown(tables["fbi_calibrated_surface_comparison"]))
    lines.append("")
    _idx = pd.to_numeric(
        pd.read_parquet(
            repo_root / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet",
            columns=["index_total_primary_event_weighted"],
        )["index_total_primary_event_weighted"],
        errors="coerce",
    ).dropna()
    _append_block(
        lines,
        f"""
        The block-group and tract distributions are extremely right-skewed, which is expected in a
        national crime-risk surface. The median block-group primary-event-weighted total-crime index is about
        `{_idx.median():.2f}`, the `95th` percentile is about `{_idx.quantile(0.95):.2f}`, and the
        far upper tail extends above `{_count_fmt(int(_idx.max() // 1000 * 1000))}`.
        """
    )
    lines.append(_df_to_markdown(tables["output_distribution_quantiles"]))
    lines.append("")
    lines.append("![Block-group index distribution](../materials/figures/block_group_total_index_distribution.png)")
    lines.append("")
    lines.append("![Tract index distribution](../materials/figures/tract_total_index_distribution.png)")
    lines.append("")
    _append_block(
        lines,
        """
        The package also includes reusable state, city, tract, and block-group summary tables under
        `materials/tables/`. For city interpretation, the benchmark cities remain the easiest
        grounded read because they are the places where the package also contains independent
        incident-truth comparisons.
        """
    )
    _append_block(
        lines,
        """
        For presentation and writeup purposes, the tables, figures, and maps in this section should be
        read together rather than in isolation. The national maps show the broad spatial shape of the
        released surface. The summary tables quantify scale, skew, and offense composition. The
        benchmark-city summaries are the places where the surface is most directly grounded against
        independent truth. Using all three views together is the most honest way to characterize the
        final product.
        """
    )
    lines.append(_df_to_markdown(tables["city_output_surface_summary"]))
    lines.append("")
    lines.append("![National tract index map](../materials/maps/national_tract_index_total_conus.png)")
    lines.append("")
    lines.append("![Benchmark-city index grid](../materials/maps/city_bg_index_total_grid.png)")
    lines.append("")

    _append_block(
        lines,
        """
        ## 12. Limitations And Honest Interpretation

        This is a shipped public-data checkpoint, not a claim of commercial benchmark parity.
        The main release limitations are:

        - State-to-state reporting intensity and source-selection differences still affect official
          control totals. Border differences are no longer display overwrite artifacts, but they can
          still reflect real reporting or control-lane differences as well as crime risk.
        - Outside the covered-city truth set, neighborhood texture is modeled transfer raked to
          official jurisdiction totals. The total is official; the neighborhood shape is covariate-modeled.
        - Ambient/visitor denominators are flagged, not repaired. Residents plus jobs/premises/vehicles
          still miss tourists, shoppers, event crowds, airports, campuses, parks, and other transient
          populations.
        - Burglary modeled-transfer commercial concentration passed the pre-registered release gate
          but remains above the pooled covered-city truth point estimate, so it should be rechecked
          when new small-city truth arrives.
        - Kansas remains about 14% below the FBI CDE estimate and needs agency-gap investigation.
        - Florida totals are deliberately FDLE-based for this release and should not be forced to FBI
          CDE totals without a source-contract decision.
        - Murder and rape have no national within-jurisdiction covariate transfer in uncovered areas;
          they use the baseline share policy there because the city-truth evidence is too sparse.
        - The release scope is the 48 contiguous states plus DC. Alaska, Hawaii, Puerto Rico, and
          other territories are excluded rather than retained as misleading zero-count geography rows.
        """
    )

    _append_block(
        lines,
        """
        ## 13. How To Read The Results Honestly

        The strongest honest claim this package supports is not that it reproduces AGS performance. It
        supports the claim that a transparent public-data pipeline can produce a release-scope
        jurisdiction-controlled block-group and tract crime-risk surface, quantify its own failure
        modes, and expose exactly where public data and human-reviewed supplements improve or fail to
        improve the result.

        The safest thesis framing is therefore:

        1. the project successfully reconstructs the end-to-end public-data pipeline;
        2. the released outputs are numerically coherent and reproducible;
        3. the resulting model is useful but materially weaker than proprietary benchmark claims;
        4. city-truth evaluation is essential because jurisdiction fit alone does not tell the full
           story about spatial quality; and
        5. the remaining gaps are structural, not hidden implementation mistakes inside the final
           release package.
        """
    )

    _append_block(
        lines,
        f"""
        ## 14. Reproducibility And Package Use

        The final diagnostics surface remains available under `{diagnostics_root}/`. The report,
        figures, maps, and reusable CSV tables in `materials/` are already generated from the
        canonical live artifacts included in the package.

        There are three sensible ways to use the package, and they answer different replication
        questions.

        The first mode is validation-only. In that mode the reader treats the included released
        outputs as fixed and reviews the compact machine-readable summaries under `validation/`
        together with the derived tables, figures, maps, and report narrative. The command for that
        mode is `uv run python scripts/diagnostics/validate_release_outputs.py` from the package's
        `{repro_root_label}` directory.

        The second mode is canonical rebuild. In that mode the reader first supplies the expected
        source-data files under `{repro_root_label}data/`, then reruns the build sequence from
        Section 1. `build-release` promotes the retained review roots into canonical
        `state/reference/inputs/` and `state/modeling/inputs/` surfaces inside the package repro
        tree, rebuilds the intermediate state, and then confirms that the released outputs can be
        regenerated cleanly.

        The third mode is broader methodological recreation. That mode starts from the raw federal,
        Census, and covariate families, then reconstructs the promotion surfaces described earlier
        in the report: local reference resolutions, municipal publication packets, state-publication
        loaders, and city incident packet gates. The submission package does not expose the full
        repo-side research and orchestration workspace for that purpose. It includes only the
        reviewed-input state that the canonical rebuild still consumes under `state/review/`:
        local-resolution outputs, municipal-target packets, and city packet gates. It does not
        package the broader state-source research workspace, review support notes, run logs,
        benchmark machinery, or review-orchestration code that were used to decide what shipped.

        Once the expected source data have been supplied and the intermediate state has been rebuilt,
        the minimal QA rerun from the package is:

        ```bash
        cd {repro_root_label.rstrip('/') or '.'}
        uv run python scripts/diagnostics/qa_build.py
        ```

        A reader who wants to rebuild from raw/prepared data and promoted inputs can then walk the
        canonical build sequence given earlier in Section 1. The package is deliberately overinclusive on
        derived tables, figures, and maps so that the final thesis presentation can draw directly from
        the included material without re-running exploratory analysis.

        In practical terms, the report should be enough to tell a technically competent reader what has
        to be rebuilt, in what order, and why each promoted surface exists. The package is enough to
        let that reader validate the final outputs immediately and rerun the canonical build once
        source data are supplied, without rediscovering the project structure. Where the system depended on manual or agent-assisted
        review, the report names the exact queue or packet format and the downstream promoted surface
        that the review produced, so that the human part of the pipeline is documented rather than
        hidden.
        """,
    )

    report_path.write_text("\n".join(lines))
    return report_path


def generate_submission_materials(
    *,
    repo_root: Path,
    out_dir: Path,
    package_repo_prefix: str = "",
) -> dict[str, str]:
    report_dir = _ensure_dir(out_dir / "report")
    materials_dir = _ensure_dir(out_dir / "materials")
    tables_dir = _ensure_dir(materials_dir / "tables")
    figures_dir = _ensure_dir(materials_dir / "figures")
    maps_dir = _ensure_dir(materials_dir / "maps")

    tables, summary = _prepare_tables(repo_root, package_repo_prefix=package_repo_prefix)
    for name, df in tables.items():
        _write_csv(df, tables_dir / f"{name}.csv")

    inventory = _make_figures_and_maps(
        repo_root=repo_root,
        tables=tables,
        summary=summary,
        figures_dir=figures_dir,
        maps_dir=maps_dir,
    )

    table_inventory = [
        {
            "relative_path": f"materials/tables/{name}.csv",
            "category": "table",
            "description": name.replace("_", " "),
        }
        for name in sorted(tables)
    ]
    inventory_df = pd.DataFrame(table_inventory + inventory).sort_values(["category", "relative_path"], kind="mergesort")
    _write_csv(inventory_df, materials_dir / "inventory.csv")
    materials_readme_path = _write_materials_readme(repo_root=repo_root, materials_dir=materials_dir, inventory_df=inventory_df)
    report_path = _write_report(
        repo_root=repo_root,
        report_dir=report_dir,
        tables=tables,
        summary=summary,
        package_repo_prefix=package_repo_prefix,
    )

    return {
        "report_path": str(report_path),
        "materials_dir": str(materials_dir),
        "materials_readme_path": str(materials_readme_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the detailed report and derived materials for the CrimeRisk submission package."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    summary = generate_submission_materials(repo_root=args.repo_root.resolve(), out_dir=args.out_dir.resolve())
    print(summary)


if __name__ == "__main__":
    main()
