from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.allocation import HARM_WEIGHTS
from crimerisk.paths import RepoPaths
from crimerisk.published_nibrs import (
    build_published_nibrs_corroboration_mask,
    load_published_nibrs_reference_counts,
)
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    SOURCE_PRIORITY,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    build_prefer_nibrs_mask,
    initialize_preferred_source,
    source_lane_from_source,
)

STATE_OUTPUT_DIR = REPO_ROOT / "state" / "output"
PACKAGE_VALIDATION_DIR = REPO_ROOT.parent / "validation"
REPO_QA_SUMMARY = REPO_ROOT / "state" / "qa" / "build_qa_summary.json"
# Target build year under validation (--year CLI arg; default 2024). Output filenames,
# the build-manifest year assertion, target-year frame filters, the POPESTIMATE column,
# and the year-suffixed modeling/QA artifact paths below all derive from it; main()
# rebinds the paths via _apply_target_year when --year is passed.
YEAR = 2024
REPO_NEXT_PHASE_MEASUREMENT = REPO_ROOT / "state" / "modeling" / f"next_phase_measurement_summary_{YEAR}.json"
REPO_DASHBOARD_LOOKUP = REPO_ROOT / "state" / "modeling" / f"dashboard_neighborhood_check_lookup_{YEAR}.json"
REPO_EXTERNAL_AVAILABILITY = REPO_ROOT / "state" / "modeling" / f"external_surface_availability_{YEAR}.json"
REPO_BURGLARY_TAU_CALIBRATION = REPO_ROOT / "state" / "modeling" / f"burglary_tau_calibration_{YEAR}.json"
REPO_CITY_EXACT_POINT_QA = REPO_ROOT / "state" / "qa" / f"city_feed_exact_point_concentration_{YEAR}.csv"
REPO_CITY_EXACT_POINT_EXCEPTIONS = REPO_ROOT / "configs" / "city_feed_exact_point_exceptions.csv"
COUNTY_POP_2024_CSV = REPO_ROOT / "data" / "Census-PopEst-2020-2025" / "co-est2025-alldata.csv"
ACS_BG_SOURCE_PARQUET = REPO_ROOT / "data" / "ACS-5yr-2020-2024" / "parsed" / "acs_block_groups.parquet"
ACS_MISSING_BG_BACKFILL_CSV = REPO_ROOT / "configs" / "acs_missing_bg_decennial_backfill.csv"
CT_BG_2023_ZIP = REPO_ROOT / "data" / "tiger_bg" / "tl_2023_09_bg.zip"
CT_BG_2020_ZIP = REPO_ROOT / "data" / "tiger_bg" / "tl_2020_09_bg.zip"


def _apply_target_year(year: int) -> None:
    """Rebind YEAR and the year-suffixed artifact-path globals for a non-default target year."""
    global YEAR, REPO_NEXT_PHASE_MEASUREMENT, REPO_DASHBOARD_LOOKUP, REPO_EXTERNAL_AVAILABILITY
    global REPO_BURGLARY_TAU_CALIBRATION, REPO_CITY_EXACT_POINT_QA
    global EXPECTED_RESIDUAL_FEATURE_POLICY_PATH_FRAGMENT
    YEAR = int(year)
    REPO_NEXT_PHASE_MEASUREMENT = REPO_ROOT / "state" / "modeling" / f"next_phase_measurement_summary_{YEAR}.json"
    REPO_DASHBOARD_LOOKUP = REPO_ROOT / "state" / "modeling" / f"dashboard_neighborhood_check_lookup_{YEAR}.json"
    REPO_EXTERNAL_AVAILABILITY = REPO_ROOT / "state" / "modeling" / f"external_surface_availability_{YEAR}.json"
    REPO_BURGLARY_TAU_CALIBRATION = REPO_ROOT / "state" / "modeling" / f"burglary_tau_calibration_{YEAR}.json"
    REPO_CITY_EXACT_POINT_QA = REPO_ROOT / "state" / "qa" / f"city_feed_exact_point_concentration_{YEAR}.csv"
    EXPECTED_RESIDUAL_FEATURE_POLICY_PATH_FRAGMENT = f"state/modeling/feature_transfer_policy_{YEAR}.parquet"

OFFENSES_7 = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]
AGGREGATES = ["personal", "property", "total"]
PERSONAL_OFFENSES = ["murder", "rape", "robbery", "aggravated_assault"]
PROPERTY_OFFENSES = ["burglary", "larceny", "motor_vehicle_theft"]
SPARSE_BASELINE_TRANSFER_OFFENSES = {"murder", "rape"}
DENSE_FULL_RESIDUAL_TRANSFER_OFFENSES = {
    "robbery",
    "aggravated_assault",
    "larceny",
    "motor_vehicle_theft",
}
DEFAULT_EXPECTED_RESIDUAL_TRANSFER_TAU_BY_OFFENSE = {
    offense: (0.0 if offense in SPARSE_BASELINE_TRANSFER_OFFENSES else 1.0)
    for offense in OFFENSES_7
}
AGGREGATE_INDEX_FIELDS = [
    "index_total_part1_resident",
    "index_personal_part1_resident",
    "index_property_part1_resident",
    "index_total_primary_event_weighted",
    "index_total_equal_offense",
    "index_total_harm",
]
PRIMARY_DENOMINATOR_BY_OFFENSE = {
    "murder": "exposure",
    "rape": "exposure",
    "robbery": "exposure",
    "aggravated_assault": "exposure",
    "burglary": "premises",
    "larceny": "exposure",
    "motor_vehicle_theft": "vehicles",
}
# Alaska, Hawaii, and all territories (AS, GU, MP, PR, VI) are out of the CONUS+DC scope.
RELEASE_EXCLUDED_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
EXPECTED_RELEASE_STATE_COUNT = 49
EXPECTED_ROW_COUNTS = {"block_group": 238193, "tract": 83776}
EXPECTED_PROMOTED_RESIDUAL_FEATURE_PATH_FRAGMENTS = (
    "block_group_overture_places_states_latest.parquet",
    "block_group_overture_commercial_core_states_latest.parquet",
)
EXPECTED_RESIDUAL_FEATURE_POLICY_PATH_FRAGMENT = f"state/modeling/feature_transfer_policy_{YEAR}.parquet"
EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES = {"between_only", "excluded_protected"}
EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE = {
    offense: (set() if offense == "burglary" else set(EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES))
    for offense in OFFENSES_7
}
MANIFEST_RELATIVE_ROOT_MARKERS = ("state/", "data/")
# Resident indexes are population-weighted to mean 100; exposure indexes are
# exposure-weighted to mean 100. Both are exact identities on a correct build.
INDEX_MEAN_TARGET = 100.0
INDEX_MEAN_TOLERANCE = 0.1
COUNT_DERIVED_TOLERANCE = 1e-9
TRANSFER_POLICY_TOLERANCE = 1e-10
SOURCE_MIXED_SHARE_CUTOFF = 0.60
DOMAIN_SCORE_TOLERANCE = 1e-12
RATE_PER_100K = 100000.0
NON_RESIDENTIAL_HOUSEHOLD_FLOOR = 10.0
PERSON_EXPOSURE_DENOMINATOR_FLOOR = 50.0
PERSON_EXPOSURE_FLOOR_OFFENSES = frozenset(("murder", "rape", "robbery", "aggravated_assault", "larceny"))
MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR = 50.0
BURGLARY_PREMISES_DENOMINATOR_FLOOR = 10.0
SPECIAL_USE_TRACT_PREFIX = "98"
SPECIAL_USE_EXPOSURE_DENOMINATOR_FLOOR = 10.0
BURGLARY_COMMERCIAL_GRADIENT_DIRECT_MIN = 0.8
BURGLARY_COMMERCIAL_GRADIENT_DIRECT_MAX = 1.3
BURGLARY_COMMERCIAL_GRADIENT_MODELED_MIN = 0.8
# v9 multi-term burglary denominator ceiling recomputed by the pre-registered
# covered-city truth bootstrap; see state/modeling/burglary_gate_ceiling_derivation.json.
BURGLARY_COMMERCIAL_GRADIENT_MODELED_MAX = 1.45
CT_POPULATION_TOLERANCE = 1000.0
TOTAL_LANE_TOLERANCE = 1e-6
TOTAL_LANE_SAMPLE_LIMIT = 10
TOTAL_LANE_TARGET_COLUMN = "adjusted_count_ags_core"
CITY_EXACT_POINT_SHARE_MAX = 0.005
# Minimum located-incident denominator before the exact-point share rule applies.
# Below this floor a single or double incident trivially crosses 0.5% (tiny-denominator
# artifact, e.g. Cincinnati/San Diego/Charlotte murder), so the 0.5% concentration rule
# is only enforced once the city/offense has a meaningful located base.
CITY_EXACT_POINT_MIN_LOCATED_COUNT = 100
# Per-point minimum deduped-incident floor (added 2026-07-06, geocoding-integrity-v10):
# a coordinate must carry >=5 deduped incidents of the offense to be flagged, alongside
# the >=0.5% share and >=100 located-count floors. Below it, a 1-4 incident point trivially
# clears 0.5% on a low-volume violent denominator (e.g. Baton Rouge murder 1/145,
# Indianapolis murder 2/166, packet-city robbery 3-5 counts) with no masking signature.
# Kept in sync with build_city_feed_exact_point_qa.py:EXACT_POINT_MIN_POINT_COUNT.
CITY_EXACT_POINT_MIN_POINT_COUNT = 5
HIGH_POPULATION_SPOT_CHECK_N = 100
CONSOLIDATED_AGENCY_MIN_2024_COUNT = 1000.0
CONSOLIDATED_AGENCY_MIN_FBI_POPULATION = 50000.0
# This catches the 2.6x+ consolidated city-county agencies while leaving current
# high-growth population-staleness cases below the fail line unless they worsen.
CONSOLIDATED_AGENCY_POPULATION_RATIO_THRESHOLD = 1.75
CONSOLIDATED_AGENCY_FOOTPRINT_TYPE = "consolidated_agency_footprint"
COUNTY_PLAUSIBILITY_MIN_POPULATION = 100_000.0
# Peers = other >100k-population counties in the same state; a state needs at least
# this many peer counties for its median to be a meaningful reference.
COUNTY_PLAUSIBILITY_MIN_STATE_PEER_COUNTIES = 3
# Calibrated against the promoted v11 release (state/output as of 2026-07-07), not
# hand-picked to include/exclude specific counties: this is the single largest gap
# in the sorted state-peer-ratio distribution below 0.6 (gap width 0.150, between
# ratio 0.104 and 0.254 -- the next-largest gap anywhere in that range is 0.021), so
# it is the natural break the release-validation check asked for. It isolates
# Suffolk County NY (36103, ratio 0.104) as a genuine statistical outlier. It does
# NOT isolate Manatee/Escambia/Chatham (ratios 0.41/0.25/0.28) or Lee (0.57) from
# ordinary cross-county rate variance -- see docs/STATE.md for the full discussion.
COUNTY_PLAUSIBILITY_STATE_PEER_RATIO_MIN = 0.20
# No cheap prior-vintage (prior release or prior year) per-county published-count
# reference exists in this repo yet; if one is added, its floor combines with the
# peer floor below via max(), per the original check design.
COUNTY_PLAUSIBILITY_PRIOR_VINTAGE_RATIO_MIN = 0.15
SPATIAL_STATE_SHARE_ETA2_MAX = 0.25
SPATIAL_TRACT_NEIGHBOR_K = 8
SPATIAL_BG_NEIGHBOR_K = 8
SPATIAL_TRACT_NEIGHBOR_RADIUS_MILES = 20.0
SPATIAL_BG_NEIGHBOR_RADIUS_MILES = 5.0
SPATIAL_BOUNDARY_MEDIAN_RATIO_MAX = 3.0
SPATIAL_BOUNDARY_MEDIAN_LOG1P_MAX = 1.5
SPATIAL_SOURCE_SEAM_MEDIAN_RATIO_MAX = 4.0
SPATIAL_SOURCE_SEAM_MEDIAN_LOG1P_MAX = 2.5
SPATIAL_MIN_BOUNDARY_PAIR_COUNT = 500
SPATIAL_MIN_BASELINE_PAIR_COUNT = 1000
SPATIAL_TRACT_FLAT_MIN_TRACTS = 10
SPATIAL_TRACT_FLAT_MIN_EXPECTED_COUNT = 5.0
SPATIAL_TRACT_FLAT_MIN_LOG_P95_P05 = 0.05
SPATIAL_TRACT_FLAT_MAX_ABS_RANGE = 1.0
SPATIAL_HOTSPOT_TOP_N = 100
SPATIAL_DENOMINATOR_TAIL_QUANTILE = 0.005
SPATIAL_DENOMINATOR_ABSOLUTE_FLOOR = 100.0
SPATIAL_HOTSPOT_ARTIFACT_SHARE_MAX = 0.50
SPATIAL_NO_SUPPORT_TAIL_SHARE_MAX = 0.01
SPATIAL_CENTROID_MIN_MATCH_SHARE = 0.99
EARTH_RADIUS_MILES = 3958.7613
SPATIAL_PRIMARY_INDEX_FIELDS = [f"index_{offense}_primary" for offense in OFFENSES_7]
SPATIAL_RESIDENT_INDEX_FIELDS = [f"index_{offense}_resident" for offense in OFFENSES_7]
SPATIAL_STATE_SHARE_INDEX_FIELDS = [
    *SPATIAL_PRIMARY_INDEX_FIELDS,
    *SPATIAL_RESIDENT_INDEX_FIELDS,
    *AGGREGATE_INDEX_FIELDS,
]
SPATIAL_BOUNDARY_INDEX_FIELDS = [
    *SPATIAL_PRIMARY_INDEX_FIELDS,
    *AGGREGATE_INDEX_FIELDS,
]
SPATIAL_HOTSPOT_INDEX_FIELDS = [
    *SPATIAL_PRIMARY_INDEX_FIELDS,
    "index_total_part1_resident",
    "index_total_primary_event_weighted",
]
BG_CENTROIDS_PATH = REPO_ROOT / "data" / "tiger_bg" / "parsed" / "bg_centroids.parquet"

SOURCE_TO_CONTROL_COLUMNS = {
    CIUS_SOURCE: {
        "count": "reported_count_cius",
        "weight": "observation_weight_cius",
        "months": "mean_months_reported_cius",
        "relationship": "relationship_type_cius",
    },
    LOCAL_PUBLICATION_SOURCE: {
        "count": "reported_count_local_publication",
        "weight": "observation_weight_local_publication",
        "months": "mean_months_reported_local_publication",
        "relationship": "relationship_type_local_publication",
    },
    STATE_PUBLICATION_SOURCE: {
        "count": "reported_count_state_publication",
        "weight": "observation_weight_state_publication",
        "months": "mean_months_reported_state_publication",
        "relationship": "relationship_type_state_publication",
    },
    SUMMARY_SOURCE: {
        "count": "reported_count_srs",
        "weight": "observation_weight_srs",
        "months": "mean_months_reported_srs",
        "relationship": "relationship_type_srs",
    },
    NIBRS_SOURCE: {
        "count": "reported_count_nibrs",
        "weight": "observation_weight_nibrs",
        "months": "mean_months_reported_nibrs",
        "relationship": "relationship_type_nibrs",
    },
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _sample_records(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    limit: int = TOTAL_LANE_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    sample = df.head(limit).copy()
    if columns is not None:
        sample = sample[[col for col in columns if col in sample.columns]].copy()
    return json.loads(sample.to_json(orient="records"))


def _append_total_lane_issue(
    issues: list[str],
    message: str,
    rows: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> None:
    issues.append(f"{message}; sample={_sample_records(rows, columns=columns)}")


def _expected_columns(*, geography: str) -> list[str]:
    geo_cols = ["block_group_geoid", "state_fips", f"population_{YEAR}", "tract_id"]
    if geography == "tract":
        geo_cols = ["tract_id", "state_fips", f"population_{YEAR}"]
    base = [
        "daytime_population_jobs_proxy",
        "landscan_day_pop",
        "exposure_proxy_2024",
        "landscan_day_lifted_person_exposure",
        "person_exposure_before_hq_jobs_cap",
        "person_exposure_hq_jobs_cap",
        "person_exposure_hq_jobs_cap_candidate",
        "person_exposure_hq_jobs_capped",
        "households_total",
        "commercial_premises_total",
        "destination_poi_total",
        "lodes_manufacturing_jobs",
        "lodes_wholesale_jobs",
        "lodes_retail_jobs",
        "lodes_transport_warehouse_jobs",
        "lodes_industrial_jobs",
        "burglary_premises_total",
        "burglary_commercial_exposure_weight",
        "burglary_destination_poi_exposure_weight",
        "burglary_retail_jobs_exposure_weight",
        "burglary_industrial_jobs_exposure_weight",
        "aggregate_vehicles_total",
        "county_auto_commute_vehicle_share",
        "mvt_commuter_vehicle_proxy",
        "vehicle_exposure_2024",
        "land_area_sq_mi",
        "eb_hard_min_denominator",
        "non_residential_household_floor",
        "person_exposure_denominator_floor",
        "mvt_vehicle_exposure_denominator_floor",
        "non_residential_flag",
        "special_use_tract_flag",
        "resident_secondary_denominator",
        "resident_secondary_denominator_low_reliability",
        "population_zero_with_positive_count",
        "urban_stratum",
    ]
    if geography == "block_group":
        base = ["eb_jurisdiction_id", "eb_jurisdiction_type", *base]
    if geography == "tract":
        base = [
            "dominant_eb_jurisdiction_id",
            "dominant_jurisdiction_share",
            "mixed_jurisdiction_flag",
            *base,
        ]
    offense_cols: list[str] = []
    for offense in OFFENSES_7:
        offense_cols.extend(
            [
                f"primary_denominator_type_{offense}",
                f"primary_denominator_{offense}",
                f"primary_denominator_raw_{offense}",
                f"primary_national_rate_per_100k_{offense}",
                f"primary_alpha_{offense}",
                f"primary_index_publishable_{offense}",
                f"primary_index_suppressed_{offense}",
                f"primary_zero_denominator_positive_count_{offense}",
                *(
                    [f"primary_denominator_invalid_{offense}"]
                    if offense == "motor_vehicle_theft"
                    else []
                ),
                f"estimate_mode_{offense}",
                f"raw_rate_{offense}",
                f"diagnostic_eb_rate_{offense}",
                f"diagnostic_eb_national_rate_per_100k_{offense}",
                f"diagnostic_eb_prior_rate_{offense}",
                f"diagnostic_eb_k_{offense}",
                f"diagnostic_eb_observed_weight_{offense}",
                f"diagnostic_eb_prior_weight_{offense}",
                f"index_publishable_{offense}",
                f"diagnostic_eb_low_denominator_flag_{offense}",
                f"diagnostic_eb_heavy_shrinkage_flag_{offense}",
                f"diagnostic_eb_extreme_shrinkage_flag_{offense}",
                f"denominator_reason_{offense}",
                f"rate_{offense}_primary",
                f"index_{offense}_primary",
                f"direct_incident_support_flag_{offense}",
                f"direct_incident_support_count_{offense}",
                f"direct_incident_support_years_{offense}",
                f"direct_incident_support_year_min_{offense}",
                f"direct_incident_support_year_max_{offense}",
                f"effective_numerator_support_{offense}",
                f"numerator_support_source_{offense}",
                f"rate_{offense}_primary_ci95_lower",
                f"rate_{offense}_primary_ci95_upper",
                f"index_{offense}_primary_ci95_lower",
                f"index_{offense}_primary_ci95_upper",
                f"index_{offense}_primary_ci95_width",
                f"index_{offense}_primary_ci95_width_ratio",
                f"reliability_tier_{offense}",
                f"recommended_display_geography_{offense}",
                f"transient_exposure_likely_{offense}",
                f"source_mode_{offense}",
                f"source_mode_dominant_share_{offense}",
                f"source_mode_mixed_{offense}",
                f"feed_match_rate_{offense}",
                f"feed_missing_fraction_{offense}",
                f"feed_alpha_{offense}",
                f"feed_prior_fraction_{offense}",
                f"domain_overlap_score_{offense}",
                f"confidence_tier_{offense}",
                f"confidence_reasons_{offense}",
                f"resident_national_rate_per_100k_{offense}",
                f"resident_raw_rate_{offense}",
                f"diagnostic_resident_eb_rate_{offense}",
                f"diagnostic_resident_eb_national_rate_per_100k_{offense}",
                f"diagnostic_resident_eb_prior_rate_{offense}",
                f"diagnostic_resident_eb_k_{offense}",
                f"diagnostic_resident_eb_observed_weight_{offense}",
                f"diagnostic_resident_eb_prior_weight_{offense}",
                f"index_{offense}_resident_publishable",
                f"diagnostic_resident_eb_low_denominator_flag_{offense}",
                f"diagnostic_resident_eb_heavy_shrinkage_flag_{offense}",
                f"diagnostic_resident_eb_extreme_shrinkage_flag_{offense}",
                f"resident_denominator_reason_{offense}",
                *(
                    [f"resident_denominator_invalid_{offense}"]
                    if offense == "motor_vehicle_theft"
                    else []
                ),
                f"index_{offense}_resident_suppressed",
                f"rate_{offense}_resident",
                f"index_{offense}_resident",
            ]
        )
    return [
        *geo_cols,
        *base,
        *[f"expected_count_{name}" for name in OFFENSES_7],
        *offense_cols,
        *[f"expected_count_{name}" for name in AGGREGATES],
        *[f"crime_density_{offense}" for offense in OFFENSES_7],
        "crime_density_total",
        *AGGREGATE_INDEX_FIELDS,
    ]


def _max_abs(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).abs().max())


def _finite_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _count_derived_rate_index(
    *,
    counts: pd.Series,
    denominator: pd.Series,
    publishable: pd.Series,
    national_rate_per_100k: pd.Series | float | None = None,
) -> dict[str, float | pd.Series]:
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    denom = pd.to_numeric(denominator, errors="coerce").fillna(0.0).clip(lower=0.0)
    pub = pd.Series(publishable, index=count.index).fillna(False).astype(bool) & denom.gt(0.0)
    if national_rate_per_100k is None:
        denom_sum = float(denom.loc[pub].sum())
        count_sum = float(count.loc[pub].sum())
        national_rate = RATE_PER_100K * count_sum / denom_sum if denom_sum > 0 else float("nan")
    elif isinstance(national_rate_per_100k, pd.Series):
        national_values = pd.to_numeric(national_rate_per_100k, errors="coerce").dropna().unique()
        national_rate = float(national_values[0]) if len(national_values) else float("nan")
    else:
        national_rate = float(national_rate_per_100k)
    rate = pd.Series(float("nan"), index=count.index, dtype=float)
    rate.loc[pub] = RATE_PER_100K * count.loc[pub] / denom.loc[pub]
    index = pd.Series(float("nan"), index=count.index, dtype=float)
    if pd.notna(national_rate) and national_rate > 0:
        index.loc[pub] = 100.0 * rate.loc[pub] / national_rate
    return {
        "rate": rate,
        "index": index,
        "national_rate_per_100k": national_rate,
    }


def _national_expected_count_weights(df: pd.DataFrame, offenses: list[str]) -> dict[str, float]:
    totals = {
        offense: float(pd.to_numeric(df[f"expected_count_{offense}"], errors="coerce").fillna(0.0).clip(lower=0.0).sum())
        for offense in offenses
    }
    total = float(sum(totals.values()))
    if total <= 0.0:
        return {offense: float("nan") for offense in offenses}
    return {offense: totals[offense] / total for offense in offenses}


def _resident_part1_expected(
    df: pd.DataFrame,
    *,
    offenses: list[str],
) -> tuple[pd.Series, float, pd.Series]:
    denominator = pd.to_numeric(df.get("resident_secondary_denominator"), errors="coerce").fillna(0.0).clip(lower=0.0)
    counts = sum(
        pd.to_numeric(df[f"expected_count_{offense}"], errors="coerce").fillna(0.0).clip(lower=0.0)
        for offense in offenses
    )
    publishable = denominator.gt(0.0)
    for offense in offenses:
        publishable &= pd.to_numeric(df.get(f"index_{offense}_resident"), errors="coerce").notna()
    denominator_sum = float(denominator.loc[publishable].sum())
    count_sum = float(counts.loc[publishable].sum())
    national_rate = RATE_PER_100K * count_sum / denominator_sum if denominator_sum > 0.0 else float("nan")
    rate = pd.Series(float("nan"), index=df.index, dtype=float)
    rate.loc[publishable] = RATE_PER_100K * counts.loc[publishable] / denominator.loc[publishable]
    index = pd.Series(float("nan"), index=df.index, dtype=float)
    if pd.notna(national_rate) and national_rate > 0.0:
        index.loc[publishable] = 100.0 * rate.loc[publishable] / national_rate
    return index.replace([float("inf"), float("-inf")], float("nan")), national_rate, publishable


def _primary_composite_expected(
    df: pd.DataFrame,
    *,
    offenses: list[str],
    weights: dict[str, float],
) -> tuple[pd.Series, pd.Series]:
    values = [
        pd.to_numeric(df.get(f"index_{offense}_primary"), errors="coerce")
        for offense in offenses
    ]
    publishable = pd.Series(True, index=df.index)
    for value in values:
        publishable &= value.notna()
    weight_values = [float(weights[offense]) for offense in offenses]
    weight_sum = float(sum(weight for weight in weight_values if pd.notna(weight) and weight > 0.0))
    expected = pd.Series(float("nan"), index=df.index, dtype=float)
    if weight_sum > 0.0:
        weighted = sum(value * weight for value, weight in zip(values, weight_values, strict=True))
        expected.loc[publishable] = weighted.loc[publishable] / weight_sum
    return expected.replace([float("inf"), float("-inf")], float("nan")), publishable


def _harm_total_expected(df: pd.DataFrame) -> tuple[pd.Series, float, pd.Series]:
    """Count-derived index_total_harm identity: harm_count = sum(HARM_WEIGHTS[o] * expected_count_o)
    over the seven Part-I offenses, normalized once over the person-exposure denominator and its
    national rate. Publishable wherever person exposure is publishable (residential-eligible,
    non-special-use, exposure at or above the person-exposure floor) — NOT the all-or-null
    seven-index composite rule, and not nulled by any single offense's own denominator validity.
    """
    denominator = pd.to_numeric(df.get("exposure_proxy_2024"), errors="coerce").fillna(0.0).clip(lower=0.0)
    residential_eligible = pd.to_numeric(df.get("households_total"), errors="coerce").fillna(0.0).ge(
        float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
    )
    special_use = (
        df["special_use_tract_flag"].fillna(False).astype(bool)
        if "special_use_tract_flag" in df.columns
        else pd.Series(False, index=df.index)
    )
    publishable = (
        residential_eligible
        & denominator.gt(0.0)
        & ~special_use
        & denominator.ge(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
    )
    counts = sum(
        float(HARM_WEIGHTS[offense])
        * pd.to_numeric(df.get(f"expected_count_{offense}"), errors="coerce").fillna(0.0).clip(lower=0.0)
        for offense in OFFENSES_7
    )
    published = _count_derived_rate_index(counts=counts, denominator=denominator, publishable=publishable)
    return (
        pd.Series(published["index"], index=df.index, dtype=float),
        float(published["national_rate_per_100k"]),
        publishable,
    )


def _forbidden_published_columns() -> list[str]:
    old_offense_fields: list[str] = []
    for offense in OFFENSES_7:
        old_offense_fields.extend(
            [
                f"count_{offense}",
                f"rate_{offense}",
                f"index_{offense}",
                f"resident_rate_{offense}",
                f"resident_index_{offense}",
                f"resident_index_publishable_{offense}",
                f"resident_index_suppressed_{offense}",
                f"rate_ci95_lower_{offense}",
                f"rate_ci95_upper_{offense}",
                f"index_ci95_lower_{offense}",
                f"index_ci95_upper_{offense}",
                f"index_ci95_width_{offense}",
                f"index_ci95_width_ratio_{offense}",
            ]
        )
    old_aggregate_fields: list[str] = []
    for aggregate in AGGREGATES:
        old_aggregate_fields.extend(
            [
                f"count_{aggregate}",
                f"index_{aggregate}",
                f"unweighted_index_{aggregate}",
                f"index_{aggregate}_suppressed_component_count",
                f"index_{aggregate}_partial",
            ]
        )
    return old_offense_fields + old_aggregate_fields


def _max_abs_pair_delta(actual: pd.Series, expected: pd.Series) -> tuple[float, int]:
    actual_num = pd.to_numeric(actual, errors="coerce")
    expected_num = pd.to_numeric(expected, errors="coerce")
    null_mismatch = actual_num.isna() ^ expected_num.isna()
    comparable = ~(actual_num.isna() | expected_num.isna())
    if comparable.any():
        max_abs = float((actual_num.loc[comparable] - expected_num.loc[comparable]).abs().max())
    else:
        max_abs = 0.0
    return max_abs, int(null_mismatch.sum())


def _expected_density(counts: pd.Series, land_area_sq_mi: pd.Series) -> pd.Series:
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    area = pd.to_numeric(land_area_sq_mi, errors="coerce").fillna(0.0).clip(lower=0.0)
    density = np.full(len(count), np.nan, dtype=float)
    positive_area = area.gt(0.0).to_numpy(dtype=bool)
    density[positive_area] = count.to_numpy(dtype=float)[positive_area] / area.to_numpy(dtype=float)[positive_area]
    return pd.Series(density, index=count.index, dtype=float)


def _max_abs_density_delta(actual: pd.Series, expected: pd.Series) -> tuple[float, int]:
    actual_num = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    expected_num = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    actual_null = pd.isna(actual_num)
    expected_null = pd.isna(expected_num)
    null_mismatch = int(np.not_equal(actual_null, expected_null).sum())
    comparable = ~(actual_null | expected_null)
    if not bool(comparable.any()):
        return 0.0, null_mismatch
    equal_inf = np.isinf(actual_num) & np.isinf(expected_num) & (np.sign(actual_num) == np.sign(expected_num))
    finite = comparable & ~equal_inf
    if not bool(finite.any()):
        return 0.0, null_mismatch
    diff = np.abs(actual_num[finite] - expected_num[finite])
    return float(np.nanmax(diff)) if len(diff) else 0.0, null_mismatch


def _offense_from_index_field(field: str) -> str | None:
    if field.startswith("index_") and field.endswith("_primary"):
        return field.removeprefix("index_").removesuffix("_primary")
    if field.startswith("index_") and field.endswith("_resident"):
        return field.removeprefix("index_").removesuffix("_resident")
    return None


def _expected_count_column_for_index_field(field: str) -> str | None:
    offense = _offense_from_index_field(field)
    if offense in OFFENSES_7:
        return f"expected_count_{offense}"
    if "personal" in field:
        return "expected_count_personal"
    if "property" in field:
        return "expected_count_property"
    if "total" in field:
        return "expected_count_total"
    return None


def _load_bg_centroids(*, issues: list[str]) -> pd.DataFrame | None:
    if not BG_CENTROIDS_PATH.exists():
        issues.append(f"spatial_artifacts: missing BG centroid file {BG_CENTROIDS_PATH}")
        return None
    try:
        centroids = pd.read_parquet(
            BG_CENTROIDS_PATH,
            columns=["bg_id", "tract_id", "aland", "lon", "lat"],
        )
    except (KeyError, ValueError) as exc:
        issues.append(f"spatial_artifacts: BG centroid file missing required columns: {exc}")
        return None
    centroids = centroids.copy()
    centroids["block_group_geoid"] = centroids["bg_id"].astype("string").str.zfill(12)
    centroids["tract_id"] = centroids["tract_id"].astype("string").str.zfill(11)
    centroids["aland"] = pd.to_numeric(centroids["aland"], errors="coerce").fillna(0.0).clip(lower=0.0)
    centroids["lon"] = pd.to_numeric(centroids["lon"], errors="coerce")
    centroids["lat"] = pd.to_numeric(centroids["lat"], errors="coerce")
    centroids = centroids.dropna(subset=["lon", "lat"]).drop_duplicates("block_group_geoid")
    return centroids[["block_group_geoid", "tract_id", "aland", "lon", "lat"]]


def _attach_block_group_centroids(
    df: pd.DataFrame,
    *,
    issues: list[str],
    label: str,
) -> pd.DataFrame | None:
    centroids = _load_bg_centroids(issues=issues)
    if centroids is None:
        return None
    work = df.copy()
    work["block_group_geoid"] = work["block_group_geoid"].astype("string").str.zfill(12)
    merged = work.merge(
        centroids[["block_group_geoid", "lon", "lat"]],
        on="block_group_geoid",
        how="inner",
        validate="one_to_one",
    )
    match_share = float(len(merged) / len(work)) if len(work) else 0.0
    if match_share < SPATIAL_CENTROID_MIN_MATCH_SHARE:
        issues.append(
            f"spatial_artifacts.{label}: BG centroid match share {match_share:.4f} "
            f"is below {SPATIAL_CENTROID_MIN_MATCH_SHARE:.2f}"
        )
    return merged.rename(columns={"lon": "_lon", "lat": "_lat"})


def _attach_tract_centroids(
    df: pd.DataFrame,
    *,
    issues: list[str],
    label: str,
) -> pd.DataFrame | None:
    centroids = _load_bg_centroids(issues=issues)
    if centroids is None:
        return None
    work = df.copy()
    work["tract_id"] = work["tract_id"].astype("string").str.zfill(11)
    tract_centroids = centroids[centroids["tract_id"].isin(set(work["tract_id"].dropna()))].copy()
    tract_centroids["_weight"] = tract_centroids["aland"].where(tract_centroids["aland"].gt(0.0), 1.0)
    tract_centroids["_lon_weighted"] = tract_centroids["lon"] * tract_centroids["_weight"]
    tract_centroids["_lat_weighted"] = tract_centroids["lat"] * tract_centroids["_weight"]
    tract_centroids = (
        tract_centroids.groupby("tract_id", dropna=False)
        .agg(
            _weight=("_weight", "sum"),
            _lon_weighted=("_lon_weighted", "sum"),
            _lat_weighted=("_lat_weighted", "sum"),
        )
        .reset_index()
    )
    tract_centroids["_lon"] = tract_centroids["_lon_weighted"] / tract_centroids["_weight"]
    tract_centroids["_lat"] = tract_centroids["_lat_weighted"] / tract_centroids["_weight"]
    merged = work.merge(
        tract_centroids[["tract_id", "_lon", "_lat"]],
        on="tract_id",
        how="inner",
        validate="one_to_one",
    )
    match_share = float(len(merged) / len(work)) if len(work) else 0.0
    if match_share < SPATIAL_CENTROID_MIN_MATCH_SHARE:
        issues.append(
            f"spatial_artifacts.{label}: tract centroid match share {match_share:.4f} "
            f"is below {SPATIAL_CENTROID_MIN_MATCH_SHARE:.2f}"
        )
    return merged


def _nearest_neighbor_pairs(
    df: pd.DataFrame,
    *,
    k: int,
    radius_miles: float,
    issues: list[str],
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.neighbors import BallTree
    except ImportError as exc:
        issues.append(f"spatial_artifacts.{label}: scikit-learn BallTree unavailable: {exc}")
        return df.iloc[0:0].copy(), pd.DataFrame(columns=["left_pos", "right_pos", "distance_miles"])

    valid = pd.to_numeric(df["_lat"], errors="coerce").notna() & pd.to_numeric(df["_lon"], errors="coerce").notna()
    work = df.loc[valid].reset_index(drop=True).copy()
    if len(work) <= 1:
        issues.append(f"spatial_artifacts.{label}: insufficient rows with centroids for neighbor checks")
        return work, pd.DataFrame(columns=["left_pos", "right_pos", "distance_miles"])
    coords = np.radians(work[["_lat", "_lon"]].to_numpy(dtype=float))
    tree = BallTree(coords, metric="haversine")
    k_eff = min(k + 1, len(work))
    distances, indices = tree.query(coords, k=k_eff)
    left: list[int] = []
    right: list[int] = []
    miles: list[float] = []
    for i in range(len(work)):
        for pos in range(1, k_eff):
            j = int(indices[i, pos])
            if i >= j:
                continue
            distance_miles = float(distances[i, pos] * EARTH_RADIUS_MILES)
            if distance_miles <= radius_miles:
                left.append(i)
                right.append(j)
                miles.append(distance_miles)
    pairs = pd.DataFrame({"left_pos": left, "right_pos": right, "distance_miles": miles})
    return work, pairs


def _pair_diff_stats(
    values: pd.Series,
    pairs: pd.DataFrame,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    if pairs.empty:
        return {"pair_count": 0, "median_abs_log1p_diff": None, "p90_abs_log1p_diff": None, "mean_abs_log1p_diff": None}
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    left = pairs["left_pos"].to_numpy(dtype=int)
    right = pairs["right_pos"].to_numpy(dtype=int)
    comparable = (
        mask
        & np.isfinite(arr[left])
        & np.isfinite(arr[right])
        & (arr[left] >= 0.0)
        & (arr[right] >= 0.0)
    )
    if not bool(comparable.any()):
        return {"pair_count": 0, "median_abs_log1p_diff": None, "p90_abs_log1p_diff": None, "mean_abs_log1p_diff": None}
    diff = np.abs(np.log1p(arr[left[comparable]]) - np.log1p(arr[right[comparable]]))
    return {
        "pair_count": int(len(diff)),
        "median_abs_log1p_diff": float(np.median(diff)),
        "p90_abs_log1p_diff": float(np.quantile(diff, 0.90)),
        "mean_abs_log1p_diff": float(np.mean(diff)),
    }


def _sample_pair_diffs(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    mask: np.ndarray,
    *,
    field: str,
    id_col: str,
    extra_cols: list[str],
    limit: int = TOTAL_LANE_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    if pairs.empty:
        return []
    values = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=float)
    left = pairs["left_pos"].to_numpy(dtype=int)
    right = pairs["right_pos"].to_numpy(dtype=int)
    comparable = (
        mask
        & np.isfinite(values[left])
        & np.isfinite(values[right])
        & (values[left] >= 0.0)
        & (values[right] >= 0.0)
    )
    if not bool(comparable.any()):
        return []
    diff = np.abs(np.log1p(values[left[comparable]]) - np.log1p(values[right[comparable]]))
    sample_pairs = pairs.loc[comparable].copy()
    sample_pairs["_abs_log1p_diff"] = diff
    sample_pairs = sample_pairs.sort_values("_abs_log1p_diff", ascending=False, kind="mergesort").head(limit)
    records: list[dict[str, Any]] = []
    for _, row in sample_pairs.iterrows():
        left_pos = int(row["left_pos"])
        right_pos = int(row["right_pos"])
        record: dict[str, Any] = {
            "field": field,
            "left_id": df.iloc[left_pos][id_col],
            "right_id": df.iloc[right_pos][id_col],
            "left_value": float(values[left_pos]),
            "right_value": float(values[right_pos]),
            "abs_log1p_diff": float(row["_abs_log1p_diff"]),
            "distance_miles": float(row["distance_miles"]),
        }
        for col in extra_cols:
            if col in df.columns:
                record[f"left_{col}"] = df.iloc[left_pos][col]
                record[f"right_{col}"] = df.iloc[right_pos][col]
        records.append(record)
    return json.loads(pd.DataFrame(records).to_json(orient="records")) if records else []


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator <= 0.0:
        return None if numerator <= 0.0 else float("inf")
    return numerator / denominator


def _state_share_eta2(df: pd.DataFrame, *, field: str) -> tuple[float | None, int, list[dict[str, Any]]]:
    if field not in df.columns or "state_fips" not in df.columns:
        return None, 0, []
    values = pd.to_numeric(df[field], errors="coerce")
    valid = values.notna()
    if int(valid.sum()) <= 1:
        return None, int(valid.sum()), []
    working = pd.DataFrame(
        {
            "state_fips": df.loc[valid, "state_fips"].astype("string").str.zfill(2),
            "value": values.loc[valid].to_numpy(dtype=float),
        }
    )
    grand_mean = float(working["value"].mean())
    total_ss = float(((working["value"] - grand_mean) ** 2).sum())
    if total_ss <= 0.0:
        return 0.0, int(len(working)), []
    state_stats = (
        working.groupby("state_fips", dropna=False)["value"]
        .agg(row_count="size", mean="mean")
        .reset_index()
    )
    state_stats["mean_delta_abs"] = (state_stats["mean"] - grand_mean).abs()
    between_ss = float((state_stats["row_count"] * (state_stats["mean"] - grand_mean) ** 2).sum())
    sample = _sample_records(
        state_stats.sort_values("mean_delta_abs", ascending=False, kind="mergesort"),
        columns=["state_fips", "row_count", "mean", "mean_delta_abs"],
    )
    return between_ss / total_ss, int(len(working)), sample


def _check_spatial_state_share_of_variation(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    surfaces = {
        "block_group_ags_core": output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet",
        "tract_ags_core": output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet",
        "block_group_fbi_calibrated": output_dir / f"crimerisk_block_group_{YEAR}_fbi_calibrated.parquet",
        "tract_fbi_calibrated": output_dir / f"crimerisk_tract_{YEAR}_fbi_calibrated.parquet",
    }
    result: dict[str, Any] = {
        "ok": True,
        "threshold_eta2_max": float(SPATIAL_STATE_SHARE_ETA2_MAX),
        "surfaces": {},
    }
    for label, path in surfaces.items():
        if not path.exists():
            issues.append(f"spatial_artifacts.state_share_of_variation: missing surface {path}")
            result["surfaces"][label] = {"present": False, "path": str(path)}
            result["ok"] = False
            continue
        requested = ["state_fips", *SPATIAL_STATE_SHARE_INDEX_FIELDS]
        try:
            df = pd.read_parquet(path, columns=requested)
        except (KeyError, ValueError):
            df = pd.read_parquet(path)
        fields: dict[str, Any] = {}
        for field in SPATIAL_STATE_SHARE_INDEX_FIELDS:
            eta2, row_count, state_sample = _state_share_eta2(df, field=field)
            fields[field] = {"eta2": eta2, "row_count": row_count}
            if eta2 is not None and eta2 > SPATIAL_STATE_SHARE_ETA2_MAX:
                result["ok"] = False
                issues.append(
                    "spatial_artifacts.state_share_of_variation: "
                    f"{label}.{field} eta2={eta2:.6f} exceeds {SPATIAL_STATE_SHARE_ETA2_MAX:.2f}; "
                    f"sample={state_sample}"
                )
        result["surfaces"][label] = {"present": True, "path": str(path), "fields": fields}
    return result


def _check_boundary_discontinuity(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    path = output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet"
    columns = ["tract_id", "state_fips", "dominant_eb_jurisdiction_id", *SPATIAL_BOUNDARY_INDEX_FIELDS]
    if not path.exists():
        issues.append(f"spatial_artifacts.boundary_discontinuity: missing tract surface {path}")
        return {"ok": False, "present": False, "path": str(path)}
    try:
        tract = pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as exc:
        issues.append(f"spatial_artifacts.boundary_discontinuity: tract surface missing required columns: {exc}")
        return {"ok": False, "present": True, "path": str(path), "required_columns_present": False}
    tract = _attach_tract_centroids(tract, issues=issues, label="boundary_discontinuity")
    if tract is None:
        return {"ok": False, "present": True, "path": str(path), "centroids_present": False}
    tract, pairs = _nearest_neighbor_pairs(
        tract,
        k=SPATIAL_TRACT_NEIGHBOR_K,
        radius_miles=SPATIAL_TRACT_NEIGHBOR_RADIUS_MILES,
        issues=issues,
        label="boundary_discontinuity",
    )
    if pairs.empty:
        issues.append("spatial_artifacts.boundary_discontinuity: no tract neighbor pairs found")
        return {"ok": False, "present": True, "path": str(path), "neighbor_pair_count": 0}

    left = pairs["left_pos"].to_numpy(dtype=int)
    right = pairs["right_pos"].to_numpy(dtype=int)
    state = tract["state_fips"].astype("string").str.zfill(2).to_numpy()
    jurisdiction = tract["dominant_eb_jurisdiction_id"].astype("string").to_numpy()
    within_jurisdiction = jurisdiction[left] == jurisdiction[right]
    cross_jurisdiction = jurisdiction[left] != jurisdiction[right]
    cross_state = state[left] != state[right]

    fields: dict[str, Any] = {}
    ok = True
    for field in SPATIAL_BOUNDARY_INDEX_FIELDS:
        values = pd.to_numeric(tract[field], errors="coerce")
        within_stats = _pair_diff_stats(values, pairs, within_jurisdiction)
        jurisdiction_stats = _pair_diff_stats(values, pairs, cross_jurisdiction)
        state_stats = _pair_diff_stats(values, pairs, cross_state)
        jurisdiction_ratio = _ratio(
            jurisdiction_stats["median_abs_log1p_diff"],
            within_stats["median_abs_log1p_diff"],
        )
        state_ratio = _ratio(
            state_stats["median_abs_log1p_diff"],
            within_stats["median_abs_log1p_diff"],
        )
        field_summary = {
            "within_jurisdiction": within_stats,
            "cross_jurisdiction": jurisdiction_stats,
            "cross_state": state_stats,
            "cross_jurisdiction_to_within_median_ratio": jurisdiction_ratio,
            "cross_state_to_within_median_ratio": state_ratio,
        }
        fields[field] = field_summary
        for boundary_name, boundary_stats, boundary_mask, boundary_ratio in [
            ("jurisdiction", jurisdiction_stats, cross_jurisdiction, jurisdiction_ratio),
            ("state", state_stats, cross_state, state_ratio),
        ]:
            pair_count = int(boundary_stats["pair_count"] or 0)
            baseline_count = int(within_stats["pair_count"] or 0)
            median_diff = boundary_stats["median_abs_log1p_diff"]
            if (
                pair_count >= SPATIAL_MIN_BOUNDARY_PAIR_COUNT
                and baseline_count >= SPATIAL_MIN_BASELINE_PAIR_COUNT
                and boundary_ratio is not None
                and median_diff is not None
                and boundary_ratio > SPATIAL_BOUNDARY_MEDIAN_RATIO_MAX
                and float(median_diff) > SPATIAL_BOUNDARY_MEDIAN_LOG1P_MAX
            ):
                ok = False
                sample = _sample_pair_diffs(
                    tract,
                    pairs,
                    boundary_mask,
                    field=field,
                    id_col="tract_id",
                    extra_cols=["state_fips", "dominant_eb_jurisdiction_id"],
                )
                issues.append(
                    "spatial_artifacts.boundary_discontinuity: "
                    f"{field} has systematic {boundary_name}-boundary jump "
                    f"(median ratio {boundary_ratio:.3f} > {SPATIAL_BOUNDARY_MEDIAN_RATIO_MAX:.1f}, "
                    f"median abs log1p diff {float(median_diff):.3f} > "
                    f"{SPATIAL_BOUNDARY_MEDIAN_LOG1P_MAX:.1f}); sample={sample}"
                )
    return {
        "ok": ok,
        "present": True,
        "path": str(path),
        "neighbor_pair_count": int(len(pairs)),
        "neighbor_k": int(SPATIAL_TRACT_NEIGHBOR_K),
        "neighbor_radius_miles": float(SPATIAL_TRACT_NEIGHBOR_RADIUS_MILES),
        "thresholds": {
            "median_ratio_max": float(SPATIAL_BOUNDARY_MEDIAN_RATIO_MAX),
            "median_abs_log1p_max": float(SPATIAL_BOUNDARY_MEDIAN_LOG1P_MAX),
            "min_boundary_pair_count": int(SPATIAL_MIN_BOUNDARY_PAIR_COUNT),
            "min_baseline_pair_count": int(SPATIAL_MIN_BASELINE_PAIR_COUNT),
        },
        "fields": fields,
    }


def _check_source_mode_seam(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    path = output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet"
    columns = ["block_group_geoid", "state_fips", "eb_jurisdiction_id"]
    for offense in OFFENSES_7:
        columns.extend([f"index_{offense}_primary", f"source_mode_{offense}"])
    if not path.exists():
        issues.append(f"spatial_artifacts.source_mode_seam: missing BG surface {path}")
        return {"ok": False, "present": False, "path": str(path)}
    try:
        bg = pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as exc:
        issues.append(f"spatial_artifacts.source_mode_seam: BG surface missing required columns: {exc}")
        return {"ok": False, "present": True, "path": str(path), "required_columns_present": False}
    bg = _attach_block_group_centroids(bg, issues=issues, label="source_mode_seam")
    if bg is None:
        return {"ok": False, "present": True, "path": str(path), "centroids_present": False}
    bg, pairs = _nearest_neighbor_pairs(
        bg,
        k=SPATIAL_BG_NEIGHBOR_K,
        radius_miles=SPATIAL_BG_NEIGHBOR_RADIUS_MILES,
        issues=issues,
        label="source_mode_seam",
    )
    if pairs.empty:
        issues.append("spatial_artifacts.source_mode_seam: no BG neighbor pairs found")
        return {"ok": False, "present": True, "path": str(path), "neighbor_pair_count": 0}

    left = pairs["left_pos"].to_numpy(dtype=int)
    right = pairs["right_pos"].to_numpy(dtype=int)
    fields: dict[str, Any] = {}
    ok = True
    for offense in OFFENSES_7:
        field = f"index_{offense}_primary"
        source = bg[f"source_mode_{offense}"].astype("string").to_numpy()
        direct_left = source[left] == "direct_city_incident"
        direct_right = source[right] == "direct_city_incident"
        modeled_left = source[left] == "modeled_transfer"
        modeled_right = source[right] == "modeled_transfer"
        direct_modeled = (direct_left & modeled_right) | (modeled_left & direct_right)
        same_source = (direct_left & direct_right) | (modeled_left & modeled_right)
        cross_stats = _pair_diff_stats(bg[field], pairs, direct_modeled)
        same_stats = _pair_diff_stats(bg[field], pairs, same_source)
        seam_ratio = _ratio(cross_stats["median_abs_log1p_diff"], same_stats["median_abs_log1p_diff"])
        fields[offense] = {
            "field": field,
            "direct_modeled_boundary": cross_stats,
            "same_source_neighbors": same_stats,
            "direct_modeled_to_same_source_median_ratio": seam_ratio,
        }
        pair_count = int(cross_stats["pair_count"] or 0)
        baseline_count = int(same_stats["pair_count"] or 0)
        median_diff = cross_stats["median_abs_log1p_diff"]
        if (
            pair_count >= SPATIAL_MIN_BOUNDARY_PAIR_COUNT
            and baseline_count >= SPATIAL_MIN_BASELINE_PAIR_COUNT
            and seam_ratio is not None
            and median_diff is not None
            and seam_ratio > SPATIAL_SOURCE_SEAM_MEDIAN_RATIO_MAX
            and float(median_diff) > SPATIAL_SOURCE_SEAM_MEDIAN_LOG1P_MAX
        ):
            ok = False
            sample = _sample_pair_diffs(
                bg,
                pairs,
                direct_modeled,
                field=field,
                id_col="block_group_geoid",
                extra_cols=["state_fips", "eb_jurisdiction_id", f"source_mode_{offense}"],
            )
            issues.append(
                "spatial_artifacts.source_mode_seam: "
                f"{offense} direct-vs-modeled BG neighbors show a systematic seam "
                f"(median ratio {seam_ratio:.3f} > {SPATIAL_SOURCE_SEAM_MEDIAN_RATIO_MAX:.1f}, "
                f"median abs log1p diff {float(median_diff):.3f} > "
                f"{SPATIAL_SOURCE_SEAM_MEDIAN_LOG1P_MAX:.1f}); sample={sample}"
            )
    return {
        "ok": ok,
        "present": True,
        "path": str(path),
        "neighbor_pair_count": int(len(pairs)),
        "neighbor_k": int(SPATIAL_BG_NEIGHBOR_K),
        "neighbor_radius_miles": float(SPATIAL_BG_NEIGHBOR_RADIUS_MILES),
        "thresholds": {
            "median_ratio_max": float(SPATIAL_SOURCE_SEAM_MEDIAN_RATIO_MAX),
            "median_abs_log1p_max": float(SPATIAL_SOURCE_SEAM_MEDIAN_LOG1P_MAX),
            "min_boundary_pair_count": int(SPATIAL_MIN_BOUNDARY_PAIR_COUNT),
            "min_baseline_pair_count": int(SPATIAL_MIN_BASELINE_PAIR_COUNT),
        },
        "offenses": fields,
    }


def _check_tract_flatness(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    path = output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet"
    expected_cols = sorted(
        {
            col
            for field in SPATIAL_BOUNDARY_INDEX_FIELDS
            for col in [_expected_count_column_for_index_field(field)]
            if col is not None
        }
    )
    columns = ["tract_id", "state_fips", "dominant_eb_jurisdiction_id", *SPATIAL_BOUNDARY_INDEX_FIELDS, *expected_cols]
    if not path.exists():
        issues.append(f"spatial_artifacts.tract_flatness: missing tract surface {path}")
        return {"ok": False, "present": False, "path": str(path)}
    try:
        tract = pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as exc:
        issues.append(f"spatial_artifacts.tract_flatness: tract surface missing required columns: {exc}")
        return {"ok": False, "present": True, "path": str(path), "required_columns_present": False}

    fields: dict[str, Any] = {}
    ok = True
    for field in SPATIAL_BOUNDARY_INDEX_FIELDS:
        expected_col = _expected_count_column_for_index_field(field)
        working_cols = ["state_fips", "dominant_eb_jurisdiction_id", "tract_id", field]
        if expected_col is not None and expected_col in tract.columns:
            working_cols.append(expected_col)
        working = tract[working_cols].copy()
        working["_value"] = pd.to_numeric(working[field], errors="coerce")
        working = working[working["_value"].notna()].copy()
        if working.empty:
            fields[field] = {"group_count": 0, "flagged_group_count": 0, "sample": []}
            continue
        if expected_col is not None and expected_col in working.columns:
            working["_expected_count"] = pd.to_numeric(working[expected_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            working["_expected_count"] = 0.0
        grouped = working.groupby(["state_fips", "dominant_eb_jurisdiction_id"], dropna=False)
        stats = grouped["_value"].agg(
            tract_count="size",
            mean="mean",
            std="std",
            minimum="min",
            maximum="max",
            q05=lambda x: x.quantile(0.05),
            q95=lambda x: x.quantile(0.95),
        )
        stats["expected_count_sum"] = grouped["_expected_count"].sum()
        stats = stats.reset_index()
        stats["log_p95_p05_spread"] = np.log1p(stats["q95"].clip(lower=0.0)) - np.log1p(stats["q05"].clip(lower=0.0))
        stats["abs_range"] = stats["maximum"] - stats["minimum"]
        flagged = stats[
            stats["tract_count"].ge(SPATIAL_TRACT_FLAT_MIN_TRACTS)
            & stats["expected_count_sum"].ge(SPATIAL_TRACT_FLAT_MIN_EXPECTED_COUNT)
            & stats["log_p95_p05_spread"].lt(SPATIAL_TRACT_FLAT_MIN_LOG_P95_P05)
            & stats["abs_range"].lt(SPATIAL_TRACT_FLAT_MAX_ABS_RANGE)
        ].copy()
        sample = _sample_records(
            flagged.sort_values(["tract_count", "expected_count_sum"], ascending=False, kind="mergesort"),
            columns=[
                "state_fips",
                "dominant_eb_jurisdiction_id",
                "tract_count",
                "expected_count_sum",
                "mean",
                "minimum",
                "maximum",
                "log_p95_p05_spread",
                "abs_range",
            ],
        )
        fields[field] = {
            "group_count": int(len(stats)),
            "eligible_group_count": int(
                (
                    stats["tract_count"].ge(SPATIAL_TRACT_FLAT_MIN_TRACTS)
                    & stats["expected_count_sum"].ge(SPATIAL_TRACT_FLAT_MIN_EXPECTED_COUNT)
                ).sum()
            ),
            "flagged_group_count": int(len(flagged)),
            "minimum_log_p95_p05_spread": float(stats["log_p95_p05_spread"].min()) if len(stats) else None,
            "sample": sample,
        }
        if not flagged.empty:
            ok = False
            issues.append(
                "spatial_artifacts.tract_flatness: "
                f"{field} has {len(flagged)} jurisdiction(s) with suspiciously uniform tract values; "
                f"sample={sample}"
            )
    return {
        "ok": ok,
        "present": True,
        "path": str(path),
        "thresholds": {
            "min_tracts": int(SPATIAL_TRACT_FLAT_MIN_TRACTS),
            "min_expected_count": float(SPATIAL_TRACT_FLAT_MIN_EXPECTED_COUNT),
            "min_log_p95_p05_spread": float(SPATIAL_TRACT_FLAT_MIN_LOG_P95_P05),
            "max_abs_range": float(SPATIAL_TRACT_FLAT_MAX_ABS_RANGE),
        },
        "fields": fields,
    }


def _denominator_tail_mask_for_field(
    df: pd.DataFrame,
    *,
    field: str,
) -> tuple[pd.Series, dict[str, float]]:
    offense = _offense_from_index_field(field)
    if offense in OFFENSES_7 and field.endswith("_primary"):
        denom_col = f"primary_denominator_{offense}"
        denom = pd.to_numeric(df.get(denom_col), errors="coerce")
        published = pd.to_numeric(df.get(field), errors="coerce").notna() & denom.gt(0.0)
        quantile = float(denom.loc[published].quantile(SPATIAL_DENOMINATOR_TAIL_QUANTILE)) if published.any() else float("nan")
        threshold = max(quantile, SPATIAL_DENOMINATOR_ABSOLUTE_FLOOR) if pd.notna(quantile) else SPATIAL_DENOMINATOR_ABSOLUTE_FLOOR
        return denom.le(threshold).fillna(False), {denom_col: float(threshold)}
    if field == "index_total_primary_event_weighted":
        mask = pd.Series(False, index=df.index)
        thresholds: dict[str, float] = {}
        for offense_name in OFFENSES_7:
            offense_mask, offense_threshold = _denominator_tail_mask_for_field(
                df,
                field=f"index_{offense_name}_primary",
            )
            mask |= offense_mask
            thresholds.update(offense_threshold)
        return mask, thresholds
    denom = pd.to_numeric(df.get("resident_secondary_denominator"), errors="coerce")
    published = pd.to_numeric(df.get(field), errors="coerce").notna() & denom.gt(0.0)
    quantile = float(denom.loc[published].quantile(SPATIAL_DENOMINATOR_TAIL_QUANTILE)) if published.any() else float("nan")
    threshold = max(quantile, SPATIAL_DENOMINATOR_ABSOLUTE_FLOOR) if pd.notna(quantile) else SPATIAL_DENOMINATOR_ABSOLUTE_FLOOR
    return denom.le(threshold).fillna(False), {"resident_secondary_denominator": float(threshold)}


def _check_top_hotspot_audit(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    path = output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet"
    columns = [
        "tract_id",
        "state_fips",
        "dominant_eb_jurisdiction_id",
        "special_use_tract_flag",
        "resident_secondary_denominator",
        "crime_density_total",
        *SPATIAL_HOTSPOT_INDEX_FIELDS,
    ]
    for offense in OFFENSES_7:
        columns.extend(
            [
                f"primary_denominator_{offense}",
                f"estimate_mode_{offense}",
                f"confidence_tier_{offense}",
                f"crime_density_{offense}",
            ]
        )
    if not path.exists():
        issues.append(f"spatial_artifacts.top_hotspot_audit: missing tract surface {path}")
        return {"ok": False, "present": False, "path": str(path)}
    try:
        tract = pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as exc:
        issues.append(f"spatial_artifacts.top_hotspot_audit: tract surface missing required columns: {exc}")
        return {"ok": False, "present": True, "path": str(path), "required_columns_present": False}

    fields: dict[str, Any] = {}
    ok = True
    special = tract["special_use_tract_flag"].fillna(False).astype(bool)
    for field in SPATIAL_HOTSPOT_INDEX_FIELDS:
        values = pd.to_numeric(tract[field], errors="coerce")
        published = values.notna()
        top = tract.loc[published].assign(_index_value=values.loc[published]).nlargest(
            SPATIAL_HOTSPOT_TOP_N,
            "_index_value",
        )
        if top.empty:
            issues.append(f"spatial_artifacts.top_hotspot_audit: {field} has no published top-hotspot rows")
            fields[field] = {"top_n": 0, "artifact_prone_count": 0, "ok": False}
            ok = False
            continue
        denominator_tail, thresholds = _denominator_tail_mask_for_field(tract, field=field)
        offense = _offense_from_index_field(field)
        if offense in OFFENSES_7:
            suppressed_mode = tract[f"estimate_mode_{offense}"].astype("string").ne("count_derived")
            low_confidence = tract[f"confidence_tier_{offense}"].astype("string").eq("low")
            density = pd.to_numeric(tract[f"crime_density_{offense}"], errors="coerce")
            density_median = float(density.loc[published].median()) if published.any() else float("nan")
            low_density = density.lt(density_median)
            context_cols = [
                "tract_id",
                "state_fips",
                "dominant_eb_jurisdiction_id",
                field,
                f"primary_denominator_{offense}",
                f"estimate_mode_{offense}",
                f"confidence_tier_{offense}",
                f"crime_density_{offense}",
                "special_use_tract_flag",
            ]
        else:
            suppressed_mode = pd.Series(False, index=tract.index)
            component_confidence_cols = [f"confidence_tier_{name}" for name in OFFENSES_7]
            low_confidence = (
                tract[component_confidence_cols].astype("string").eq("low").any(axis=1)
                if all(col in tract.columns for col in component_confidence_cols)
                else pd.Series(False, index=tract.index)
            )
            density = pd.to_numeric(tract["crime_density_total"], errors="coerce")
            density_median = float(density.loc[published].median()) if published.any() else float("nan")
            low_density = density.lt(density_median)
            context_cols = [
                "tract_id",
                "state_fips",
                "dominant_eb_jurisdiction_id",
                field,
                "resident_secondary_denominator",
                "crime_density_total",
                "special_use_tract_flag",
            ]
        artifact_prone = special | denominator_tail | suppressed_mode
        top_artifact = artifact_prone.loc[top.index]
        top_tail = denominator_tail.loc[top.index]
        top_special = special.loc[top.index]
        top_suppressed = suppressed_mode.loc[top.index]
        top_low_confidence = low_confidence.loc[top.index]
        top_low_density = low_density.loc[top.index]
        artifact_count = int(top_artifact.sum())
        top_n = int(len(top))
        artifact_share = float(artifact_count / top_n) if top_n else 0.0
        sample = _sample_records(
            top.loc[top_artifact]
            .sort_values("_index_value", ascending=False, kind="mergesort")
            .assign(
                denominator_tail=top_tail.loc[top_artifact].to_numpy(dtype=bool),
                low_confidence=top_low_confidence.loc[top_artifact].to_numpy(dtype=bool),
                low_density=top_low_density.loc[top_artifact].to_numpy(dtype=bool),
            ),
            columns=[*context_cols, "_index_value", "denominator_tail", "low_confidence", "low_density"],
        )
        fields[field] = {
            "top_n": top_n,
            "artifact_prone_count": artifact_count,
            "genuine_count": int(top_n - artifact_count),
            "artifact_prone_share": artifact_share,
            "special_use_count": int(top_special.sum()),
            "denominator_tail_count": int(top_tail.sum()),
            "suppressed_mode_count": int(top_suppressed.sum()),
            "low_confidence_count": int(top_low_confidence.sum()),
            "low_density_count": int(top_low_density.sum()),
            "density_median": density_median,
            "denominator_tail_thresholds": thresholds,
            "sample_artifact_prone_rows": sample,
            "ok": artifact_share < SPATIAL_HOTSPOT_ARTIFACT_SHARE_MAX,
        }
        if artifact_share >= SPATIAL_HOTSPOT_ARTIFACT_SHARE_MAX:
            ok = False
            issues.append(
                "spatial_artifacts.top_hotspot_audit: "
                f"{field} top-{SPATIAL_HOTSPOT_TOP_N} artifact-prone share {artifact_share:.3f} "
                f"is not below {SPATIAL_HOTSPOT_ARTIFACT_SHARE_MAX:.2f}; sample={sample}"
            )
    return {
        "ok": ok,
        "present": True,
        "path": str(path),
        "top_n": int(SPATIAL_HOTSPOT_TOP_N),
        "thresholds": {
            "artifact_share_must_be_below": float(SPATIAL_HOTSPOT_ARTIFACT_SHARE_MAX),
            "denominator_tail_quantile": float(SPATIAL_DENOMINATOR_TAIL_QUANTILE),
            "denominator_absolute_floor": float(SPATIAL_DENOMINATOR_ABSOLUTE_FLOOR),
        },
        "fields": fields,
    }


def _check_no_support_denominator_tail_sentinels(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    surfaces = {
        "block_group_ags_core": (
            output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet",
            "block_group_geoid",
        ),
        "tract_ags_core": (
            output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet",
            "tract_id",
        ),
    }
    result: dict[str, Any] = {
        "ok": True,
        "thresholds": {
            "sentinel_share_max": float(SPATIAL_NO_SUPPORT_TAIL_SHARE_MAX),
            "denominator_tail_quantile": float(SPATIAL_DENOMINATOR_TAIL_QUANTILE),
            "denominator_absolute_floor": float(SPATIAL_DENOMINATOR_ABSOLUTE_FLOOR),
        },
        "surfaces": {},
    }
    for label, (path, id_col) in surfaces.items():
        if not path.exists():
            issues.append(f"spatial_artifacts.no_support_denominator_tail: missing surface {path}")
            result["surfaces"][label] = {"present": False, "path": str(path)}
            result["ok"] = False
            continue
        columns = [id_col, "state_fips"]
        for offense in OFFENSES_7:
            columns.extend(
                [
                    f"index_{offense}_primary",
                    f"primary_denominator_{offense}",
                    f"direct_incident_support_flag_{offense}",
                    f"effective_numerator_support_{offense}",
                    f"source_mode_{offense}",
                    f"estimate_mode_{offense}",
                    f"confidence_tier_{offense}",
                ]
            )
        try:
            surface = pd.read_parquet(path, columns=columns)
        except (KeyError, ValueError) as exc:
            issues.append(f"spatial_artifacts.no_support_denominator_tail: {label} missing required columns: {exc}")
            result["surfaces"][label] = {"present": True, "path": str(path), "required_columns_present": False}
            result["ok"] = False
            continue
        offenses: dict[str, Any] = {}
        for offense in OFFENSES_7:
            field = f"index_{offense}_primary"
            values = pd.to_numeric(surface[field], errors="coerce")
            published = values.notna()
            denominator_tail, thresholds = _denominator_tail_mask_for_field(surface, field=field)
            direct_flag = surface[f"direct_incident_support_flag_{offense}"].fillna(False).astype(bool)
            support = pd.to_numeric(surface[f"effective_numerator_support_{offense}"], errors="coerce").fillna(0.0)
            source = surface[f"source_mode_{offense}"].astype("string")
            no_direct_support = (~direct_flag) & support.le(0.0) & source.ne("direct_city_incident")
            sentinel = published & denominator_tail & no_direct_support
            published_count = int(published.sum())
            sentinel_count = int(sentinel.sum())
            sentinel_share = float(sentinel_count / published_count) if published_count else 0.0
            sample = _sample_records(
                surface.loc[sentinel]
                .assign(_index_value=values.loc[sentinel])
                .sort_values("_index_value", ascending=False, kind="mergesort"),
                columns=[
                    id_col,
                    "state_fips",
                    field,
                    f"primary_denominator_{offense}",
                    f"source_mode_{offense}",
                    f"estimate_mode_{offense}",
                    f"confidence_tier_{offense}",
                    f"effective_numerator_support_{offense}",
                    "_index_value",
                ],
            )
            offenses[offense] = {
                "published_count": published_count,
                "sentinel_count": sentinel_count,
                "sentinel_share": sentinel_share,
                "denominator_tail_thresholds": thresholds,
                "sample": sample,
                "ok": sentinel_share <= SPATIAL_NO_SUPPORT_TAIL_SHARE_MAX,
            }
            if sentinel_share > SPATIAL_NO_SUPPORT_TAIL_SHARE_MAX:
                result["ok"] = False
                issues.append(
                    "spatial_artifacts.no_support_denominator_tail: "
                    f"{label}.{offense} has no-support denominator-tail share {sentinel_share:.4f} "
                    f"> {SPATIAL_NO_SUPPORT_TAIL_SHARE_MAX:.2f}; sample={sample}"
                )
        result["surfaces"][label] = {"present": True, "path": str(path), "offenses": offenses}
    return result


def _check_spatial_artifact_gates(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    checks = {
        "state_share_of_variation": _check_spatial_state_share_of_variation(
            output_dir=output_dir,
            issues=issues,
        ),
        "boundary_discontinuity": _check_boundary_discontinuity(
            output_dir=output_dir,
            issues=issues,
        ),
        "source_mode_seam": _check_source_mode_seam(
            output_dir=output_dir,
            issues=issues,
        ),
        "tract_flatness": _check_tract_flatness(
            output_dir=output_dir,
            issues=issues,
        ),
        "top_hotspot_audit": _check_top_hotspot_audit(
            output_dir=output_dir,
            issues=issues,
        ),
        "no_support_denominator_tail": _check_no_support_denominator_tail_sentinels(
            output_dir=output_dir,
            issues=issues,
        ),
    }
    return {
        "ok": all(check.get("ok") is True for check in checks.values()),
        "checks": checks,
    }


def _check_no_exposure_tempered_calls(*, issues: list[str]) -> dict[str, Any]:
    call_sites: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            issues.append(f"could not parse production module {path}: {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name == "_apply_exposure_tempered_index":
                call_sites.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    if call_sites:
        issues.append(f"production code calls _apply_exposure_tempered_index at {call_sites}")
    return {"call_sites": call_sites, "ok": not call_sites}


def _render_subscript_column(slice_node: ast.AST) -> str | None:
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    if isinstance(slice_node, ast.JoinedStr):
        parts: list[str] = []
        for value in slice_node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("{}")
        return "".join(parts)
    return None


def _check_confidence_pure_enrichment(*, issues: list[str]) -> dict[str, Any]:
    path = REPO_ROOT / "src" / "crimerisk" / "confidence.py"
    if not path.exists():
        issues.append(f"missing confidence enrichment module: {path}")
        return {"path": str(path), "present": False}
    forbidden_prefixes = (
        "expected_count_",
        "rate_",
        "index_",
        "raw_rate_",
        "resident_raw_rate_",
        "crime_density_",
        "direct_incident_support_count_",
        "effective_numerator_support_",
    )
    forbidden_exact = set(AGGREGATE_INDEX_FIELDS)
    writes: list[str] = []
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        issues.append(f"could not parse confidence module {path}: {exc}")
        return {"path": str(path), "present": True, "parse_ok": False}

    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            column = _render_subscript_column(target.slice)
            if column is None:
                continue
            if column in forbidden_exact or any(column.startswith(prefix) for prefix in forbidden_prefixes):
                writes.append(f"{path.relative_to(REPO_ROOT)}:{target.lineno}:{column}")
    if writes:
        issues.append(f"confidence.py writes published point/count columns: {writes}")
    return {
        "path": str(path),
        "present": True,
        "parse_ok": True,
        "forbidden_writes": writes,
        "ok": not writes,
    }


def _check_surface(
    *,
    label: str,
    path: Path,
    geography: str,
    issues: list[str],
) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"{label}: missing file {path}")
        return {"label": label, "path": str(path), "present": False}

    df = pd.read_parquet(path)
    id_col = "block_group_geoid" if geography == "block_group" else "tract_id"
    expected_cols = _expected_columns(geography=geography)
    missing = [col for col in expected_cols if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_cols]
    if missing:
        issues.append(f"{label}: missing columns {missing}")
    if extra:
        issues.append(f"{label}: unexpected columns {extra}")
    forbidden_present = [col for col in _forbidden_published_columns() if col in df.columns]
    if forbidden_present:
        issues.append(f"{label}: old published schema columns still present: {forbidden_present}")

    expected_rows = EXPECTED_ROW_COUNTS.get(geography)
    if expected_rows is not None and len(df) != expected_rows:
        issues.append(f"{label}: expected {expected_rows} rows, found {len(df)}")

    if id_col in df.columns:
        ids = df[id_col].astype("string")
        if ids.isna().any():
            issues.append(f"{label}: null {id_col} values present")
        duplicate_count = int(ids.duplicated().sum())
        if duplicate_count:
            issues.append(f"{label}: duplicate {id_col} count = {duplicate_count}")
    else:
        duplicate_count = 0

    states = set(df.get("state_fips", pd.Series(dtype=str)).astype(str).str.zfill(2))
    excluded_present = sorted(states & RELEASE_EXCLUDED_STATE_FIPS)
    if excluded_present:
        issues.append(f"{label}: unsupported release states present: {excluded_present}")
    if len(states) != EXPECTED_RELEASE_STATE_COUNT:
        issues.append(f"{label}: expected {EXPECTED_RELEASE_STATE_COUNT} release states, found {len(states)}")

    nonnegative_cols = [
        col
        for col in df.columns
        if col.startswith("expected_count_")
        or col.startswith("crime_density_")
        or col.startswith("rate_")
        or (col.startswith("index_") and "publishable" not in col and "suppressed" not in col)
    ]
    for col in nonnegative_cols:
        if pd.to_numeric(df[col], errors="coerce").lt(-1e-9).any():
            issues.append(f"{label}: negative values in {col}")
    for col in [f"primary_denominator_{offense}" for offense in OFFENSES_7] + [
        "resident_secondary_denominator",
        "daytime_population_jobs_proxy",
        "landscan_day_pop",
        "exposure_proxy_2024",
        "land_area_sq_mi",
    ]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").lt(-1e-9).any():
            issues.append(f"{label}: negative values in {col}")

    if {"daytime_population_jobs_proxy", "landscan_day_pop", "exposure_proxy_2024", "landscan_day_lifted_person_exposure"}.issubset(df.columns):
        jobs_exposure = pd.to_numeric(df["daytime_population_jobs_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
        landscan_day = pd.to_numeric(df["landscan_day_pop"], errors="coerce").fillna(0.0).clip(lower=0.0)
        exposure = pd.to_numeric(df["exposure_proxy_2024"], errors="coerce").fillna(0.0).clip(lower=0.0)
        observed_lift = df["landscan_day_lifted_person_exposure"].fillna(False).astype(bool)
        if geography == "block_group":
            no_cap_exposure = np.maximum(
                jobs_exposure.to_numpy(dtype=float),
                landscan_day.where(landscan_day.gt(0.0), 0.0).to_numpy(dtype=float),
            )
            expected_exposure = no_cap_exposure.copy()
            cap_flag = (
                df["person_exposure_hq_jobs_capped"].fillna(False).astype(bool)
                if "person_exposure_hq_jobs_capped" in df.columns
                else pd.Series(False, index=df.index)
            )
            if "person_exposure_hq_jobs_cap" in df.columns:
                cap_value = pd.to_numeric(df["person_exposure_hq_jobs_cap"], errors="coerce").fillna(0.0).clip(lower=0.0)
                expected_exposure[cap_flag.to_numpy(dtype=bool)] = cap_value.loc[cap_flag].to_numpy(dtype=float)
            elif bool(cap_flag.any()):
                issues.append(f"{label}: HQ-jobs capped rows are missing person_exposure_hq_jobs_cap")
            exposure_delta = np.abs(exposure.to_numpy(dtype=float) - expected_exposure)
            if bool((exposure_delta > 1e-9).any()):
                issues.append(
                    f"{label}: exposure_proxy_2024 does not equal LandScan/jobs max after HQ-jobs cap"
                )
            if "person_exposure_before_hq_jobs_cap" in df.columns:
                before_cap = (
                    pd.to_numeric(df["person_exposure_before_hq_jobs_cap"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                    .to_numpy(dtype=float)
                )
                if bool((np.abs(before_cap - no_cap_exposure) > 1e-9).any()):
                    issues.append(f"{label}: person_exposure_before_hq_jobs_cap does not equal LandScan/jobs max")
            expected_lift = landscan_day.where(landscan_day.gt(0.0), 0.0).gt(jobs_exposure)
            if int((observed_lift != expected_lift).sum()):
                issues.append(f"{label}: landscan_day_lifted_person_exposure does not match LandScan day > jobs exposure")
        else:
            if "person_exposure_before_hq_jobs_cap" in df.columns:
                before_cap = (
                    pd.to_numeric(df["person_exposure_before_hq_jobs_cap"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                )
                cap_flag = (
                    df["person_exposure_hq_jobs_capped"].fillna(False).astype(bool)
                    if "person_exposure_hq_jobs_capped" in df.columns
                    else pd.Series(False, index=df.index)
                )
                if bool(exposure.gt(before_cap + 1e-9).any()):
                    issues.append(f"{label}: tract exposure_proxy_2024 exceeds rolled-up pre-cap exposure")
                if bool((~cap_flag & (np.abs(exposure - before_cap) > 1e-9)).any()):
                    issues.append(f"{label}: uncapped tract exposure_proxy_2024 does not equal rolled-up pre-cap exposure")
            else:
                if bool(exposure.lt(jobs_exposure - 1e-9).any()) or bool(exposure.lt(landscan_day - 1e-9).any()):
                    issues.append(
                        f"{label}: tract exposure_proxy_2024 is below a rolled-up source denominator"
                    )

    personal_diff = pd.Series(dtype=float)
    property_diff = pd.Series(dtype=float)
    total_diff = pd.Series(dtype=float)
    if all(f"expected_count_{col}" in df.columns for col in [*OFFENSES_7, *AGGREGATES]):
        personal_sum = sum(pd.to_numeric(df[f"expected_count_{name}"], errors="coerce").fillna(0.0) for name in PERSONAL_OFFENSES)
        property_sum = sum(pd.to_numeric(df[f"expected_count_{name}"], errors="coerce").fillna(0.0) for name in PROPERTY_OFFENSES)
        total_sum = personal_sum + property_sum
        personal_diff = pd.to_numeric(df["expected_count_personal"], errors="coerce").fillna(0.0) - personal_sum
        property_diff = pd.to_numeric(df["expected_count_property"], errors="coerce").fillna(0.0) - property_sum
        total_diff = pd.to_numeric(df["expected_count_total"], errors="coerce").fillna(0.0) - total_sum
        if _max_abs(personal_diff) > 1e-6:
            issues.append(f"{label}: expected_count_personal does not equal personal offense sum")
        if _max_abs(property_diff) > 1e-6:
            issues.append(f"{label}: expected_count_property does not equal property offense sum")
        if _max_abs(total_diff) > 1e-6:
            issues.append(f"{label}: expected_count_total does not equal seven-offense sum")

    density_max_abs: dict[str, float] = {}
    if "land_area_sq_mi" in df.columns:
        land_area = pd.to_numeric(df["land_area_sq_mi"], errors="coerce").fillna(0.0).clip(lower=0.0)
        for name in [*OFFENSES_7, "total"]:
            density_col = f"crime_density_{name}"
            count_col = f"expected_count_{name}"
            if density_col not in df.columns:
                issues.append(f"{label}: missing {density_col}")
                continue
            expected_density = _expected_density(df[count_col], land_area)
            density_delta, density_null_mismatch = _max_abs_density_delta(
                pd.to_numeric(df[density_col], errors="coerce"),
                expected_density,
            )
            density_max_abs[name] = density_delta
            if density_null_mismatch or density_delta > 1e-9:
                issues.append(
                    f"{label}: {density_col} is not expected_count / land_area_sq_mi "
                    f"(max abs diff {density_delta:.3e}, null mismatches {density_null_mismatch})"
                )
            density_num = pd.to_numeric(df[density_col], errors="coerce").to_numpy(dtype=float)
            non_finite = np.isinf(density_num)
            if bool(non_finite.any()):
                issues.append(
                    f"{label}: {density_col} has {int(non_finite.sum())} non-finite (inf) values "
                    "(density must be NULL where land_area is zero, never inf)"
                )
    else:
        issues.append(f"{label}: missing land_area_sq_mi")

    removed_switch_cols = [
        "denominator_policy",
        "resident_rate_publishable",
        "exposure_metric_publishable",
        "activity_to_resident_exposure_ratio",
        "resident_denominator_overrun",
    ]
    present_removed = [col for col in removed_switch_cols if col in df.columns]
    if present_removed:
        issues.append(f"{label}: removed denominator-switch columns still present: {present_removed}")

    for offense, expected_type in PRIMARY_DENOMINATOR_BY_OFFENSE.items():
        type_col = f"primary_denominator_type_{offense}"
        if type_col in df.columns:
            observed = set(df[type_col].astype("string").dropna().unique().tolist())
            if observed != {expected_type}:
                issues.append(f"{label}: {type_col} has {sorted(observed)}, expected only {expected_type!r}")

    count_derived_max_abs: dict[str, dict[str, float]] = {}
    confidence_tier_counts: dict[str, dict[str, int]] = {}
    domain_overlap_ranges: dict[str, dict[str, float | None]] = {}
    allowed_urban = {"urban_core", "urban", "suburban", "rural", "non_residential"}
    urban_stratum_counts: dict[str, int] = {}
    if "urban_stratum" in df.columns:
        urban = df["urban_stratum"].astype("string")
        observed_urban = set(urban.dropna().unique().tolist())
        unexpected_urban = sorted(observed_urban - allowed_urban)
        if unexpected_urban:
            issues.append(f"{label}: urban_stratum has unexpected values {unexpected_urban}")
        urban_stratum_counts = {str(k): int(v) for k, v in urban.value_counts(dropna=False).to_dict().items()}
    for offense in OFFENSES_7:
        for required_col in [
            f"expected_count_{offense}",
            f"rate_{offense}_primary",
            f"index_{offense}_primary",
            f"rate_{offense}_resident",
            f"index_{offense}_resident",
        ]:
            if required_col not in df.columns:
                issues.append(f"{label}: missing required expected-count companion column {required_col}")
        value = pd.to_numeric(df.get(f"index_{offense}_primary"), errors="coerce")
        rate = pd.to_numeric(df.get(f"rate_{offense}_primary"), errors="coerce")
        count = pd.to_numeric(df.get(f"expected_count_{offense}"), errors="coerce").fillna(0.0).clip(lower=0.0)
        households = pd.to_numeric(df.get("households_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
        expected_non_residential = households.lt(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
        expected_denominator_invalid = pd.Series(False, index=df.index)
        tract_ids = (
            df["tract_id"].astype("string").str.zfill(11)
            if "tract_id" in df.columns
            else df[id_col].astype("string").str.zfill(11)
        )
        expected_special_flag = tract_ids.str.slice(5, 11).str.startswith(SPECIAL_USE_TRACT_PREFIX, na=False)
        observed_special_flag = (
            df["special_use_tract_flag"].fillna(False).astype(bool)
            if "special_use_tract_flag" in df.columns
            else pd.Series(False, index=df.index)
        )
        if int((observed_special_flag != expected_special_flag).sum()):
            issues.append(f"{label}: special_use_tract_flag does not match 98xx tract-code rule")
        denominator = pd.to_numeric(df.get(f"primary_denominator_{offense}"), errors="coerce").fillna(0.0)
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES and "exposure_proxy_2024" in df.columns:
            exposure_denominator = pd.to_numeric(df["exposure_proxy_2024"], errors="coerce").fillna(0.0)
            exposure_denominator_delta = (denominator - exposure_denominator).abs().max()
            if pd.notna(exposure_denominator_delta) and float(exposure_denominator_delta) > 1e-9:
                issues.append(f"{label}: primary_denominator_{offense} does not equal exposure_proxy_2024")
        if offense == "burglary":
            k_destination = (
                pd.to_numeric(df["burglary_destination_poi_exposure_weight"], errors="coerce").fillna(0.0)
                if "burglary_destination_poi_exposure_weight" in df.columns
                else (
                    pd.to_numeric(df["burglary_commercial_exposure_weight"], errors="coerce").fillna(0.0)
                    if "burglary_commercial_exposure_weight" in df.columns
                    else pd.Series(0.0, index=df.index)
                )
            )
            k_retail = (
                pd.to_numeric(df["burglary_retail_jobs_exposure_weight"], errors="coerce").fillna(0.0)
                if "burglary_retail_jobs_exposure_weight" in df.columns
                else pd.Series(0.0, index=df.index)
            )
            k_industrial = (
                pd.to_numeric(df["burglary_industrial_jobs_exposure_weight"], errors="coerce").fillna(0.0)
                if "burglary_industrial_jobs_exposure_weight" in df.columns
                else pd.Series(0.0, index=df.index)
            )
            commercial = pd.to_numeric(df.get("commercial_premises_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
            destination = (
                pd.to_numeric(df["destination_poi_total"], errors="coerce")
                if "destination_poi_total" in df.columns
                else pd.Series(np.nan, index=df.index, dtype=float)
            ).fillna(commercial).clip(lower=0.0)
            retail_jobs = pd.to_numeric(
                df["lodes_retail_jobs"] if "lodes_retail_jobs" in df.columns else pd.Series(0.0, index=df.index),
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0)
            industrial_jobs = pd.to_numeric(
                df["lodes_industrial_jobs"]
                if "lodes_industrial_jobs" in df.columns
                else pd.Series(np.nan, index=df.index),
                errors="coerce",
            )
            if industrial_jobs.isna().any():
                manufacturing = pd.to_numeric(
                    df["lodes_manufacturing_jobs"]
                    if "lodes_manufacturing_jobs" in df.columns
                    else pd.Series(0.0, index=df.index),
                    errors="coerce",
                ).fillna(0.0).clip(lower=0.0)
                wholesale = pd.to_numeric(
                    df["lodes_wholesale_jobs"]
                    if "lodes_wholesale_jobs" in df.columns
                    else pd.Series(0.0, index=df.index),
                    errors="coerce",
                ).fillna(0.0).clip(lower=0.0)
                transport = pd.to_numeric(
                    df["lodes_transport_warehouse_jobs"]
                    if "lodes_transport_warehouse_jobs" in df.columns
                    else pd.Series(0.0, index=df.index),
                    errors="coerce",
                ).fillna(0.0).clip(lower=0.0)
                industrial_jobs = industrial_jobs.fillna(manufacturing + wholesale + transport)
            industrial_jobs = industrial_jobs.fillna(0.0).clip(lower=0.0)
            expected_burglary_denominator = (
                households
                + k_destination * destination
                + k_retail * retail_jobs
                + k_industrial * industrial_jobs
            )
            denominator_delta = (denominator - expected_burglary_denominator).abs().max()
            if pd.notna(denominator_delta) and float(denominator_delta) > 1e-9:
                issues.append(
                    f"{label}: primary_denominator_burglary does not equal households_total + "
                    "k_destination_poi * destination_poi_total + k_retail_jobs * lodes_retail_jobs + "
                    "k_industrial_jobs * lodes_industrial_jobs"
                )
            expected_special_suppressed = expected_special_flag | denominator.lt(BURGLARY_PREMISES_DENOMINATOR_FLOOR)
        elif offense == "motor_vehicle_theft":
            expected_vehicle_denominator = pd.to_numeric(
                df.get("vehicle_exposure_2024"), errors="coerce"
            ).fillna(0.0).clip(lower=0.0)
            denominator_delta = (denominator - expected_vehicle_denominator).abs().max()
            if pd.notna(denominator_delta) and float(denominator_delta) > 1e-9:
                issues.append(f"{label}: primary_denominator_motor_vehicle_theft does not equal vehicle_exposure_2024")
            expected_special_suppressed = expected_special_flag
        else:
            expected_special_suppressed = expected_special_flag
        expected_special_suppressed = expected_special_suppressed.fillna(False).astype(bool)
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES:
            expected_insufficient_exposure = denominator.lt(PERSON_EXPOSURE_DENOMINATOR_FLOOR)
        elif offense == "motor_vehicle_theft":
            expected_insufficient_exposure = denominator.lt(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR)
        else:
            expected_insufficient_exposure = pd.Series(False, index=df.index)
        expected_insufficient_exposure = expected_insufficient_exposure.fillna(False).astype(bool)
        expected_suppressed = (
            expected_non_residential
            | expected_denominator_invalid
            | expected_special_suppressed
            | expected_insufficient_exposure
        )
        publishable = (
            df[f"primary_index_publishable_{offense}"].astype(bool)
            if f"primary_index_publishable_{offense}" in df.columns
            else pd.Series(False, index=df.index)
        )
        suppressed = (
            df[f"primary_index_suppressed_{offense}"].astype(bool)
            if f"primary_index_suppressed_{offense}" in df.columns
            else pd.Series(False, index=df.index)
        )
        mode = (
            df[f"estimate_mode_{offense}"].astype("string")
            if f"estimate_mode_{offense}" in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="string")
        )
        source_col = f"source_mode_{offense}"
        dominant_share_col = f"source_mode_dominant_share_{offense}"
        mixed_col = f"source_mode_mixed_{offense}"
        domain_col = f"domain_overlap_score_{offense}"
        tier_col = f"confidence_tier_{offense}"
        reasons_col = f"confidence_reasons_{offense}"
        if source_col in df.columns:
            source = df[source_col].astype("string")
            allowed_source = {"direct_city_incident", "modeled_transfer", "mixed"}
            unexpected_source = sorted(set(source.dropna().unique().tolist()) - allowed_source)
            if unexpected_source:
                issues.append(f"{label}: {source_col} has unexpected values {unexpected_source}")
        else:
            source = pd.Series(pd.NA, index=df.index, dtype="string")
        if dominant_share_col in df.columns:
            dominant_share = pd.to_numeric(df[dominant_share_col], errors="coerce")
            if dominant_share.isna().any() or dominant_share.lt(-1e-12).any() or dominant_share.gt(1.0 + 1e-12).any():
                issues.append(f"{label}: {dominant_share_col} is not fully bounded in [0,1]")
        else:
            dominant_share = pd.Series(float("nan"), index=df.index, dtype=float)
        if mixed_col in df.columns:
            mixed = df[mixed_col].fillna(False).astype(bool)
            expected_mixed = dominant_share.lt(SOURCE_MIXED_SHARE_CUTOFF)
            expected_mixed = expected_mixed.fillna(False)
            if int((mixed != expected_mixed).sum()):
                issues.append(f"{label}: {mixed_col} does not match dominant share < {SOURCE_MIXED_SHARE_CUTOFF:g}")
            if source.loc[mixed].ne("mixed").any():
                issues.append(f"{label}: {source_col} is not 'mixed' on {mixed_col}=true rows")
        if domain_col in df.columns:
            domain = pd.to_numeric(df[domain_col], errors="coerce")
            domain_overlap_ranges[offense] = {
                "min": float(domain.min()) if domain.notna().any() else None,
                "max": float(domain.max()) if domain.notna().any() else None,
            }
            if domain.isna().any() or domain.lt(-DOMAIN_SCORE_TOLERANCE).any() or domain.gt(1.0 + DOMAIN_SCORE_TOLERANCE).any():
                issues.append(f"{label}: {domain_col} is not fully bounded in [0,1]")
            if geography == "block_group":
                direct_source = source.eq("direct_city_incident")
                direct_domain_delta = (domain.loc[direct_source] - 1.0).abs()
                if bool(direct_domain_delta.gt(1e-12).any()):
                    issues.append(f"{label}: direct-city block-group rows have {domain_col} != 1.0")
        for feed_col, lower, upper in [
            (f"feed_match_rate_{offense}", 0.0, 1.0),
            (f"feed_missing_fraction_{offense}", 0.0, 1.0),
            (f"feed_prior_fraction_{offense}", 0.0, 1.0),
            (f"feed_alpha_{offense}", 0.0, float("inf")),
        ]:
            if feed_col not in df.columns:
                continue
            feed = pd.to_numeric(df[feed_col], errors="coerce")
            present = feed.notna()
            if present.any() and (feed.loc[present].lt(lower - 1e-12).any() or feed.loc[present].gt(upper + 1e-12).any()):
                issues.append(f"{label}: {feed_col} has values outside [{lower:g}, {upper:g}]")
        if tier_col in df.columns:
            tier = df[tier_col].astype("string")
            allowed_tier = {"high", "medium", "low"}
            unexpected_tier = sorted(set(tier.dropna().unique().tolist()) - allowed_tier)
            if unexpected_tier:
                issues.append(f"{label}: {tier_col} has unexpected values {unexpected_tier}")
            confidence_tier_counts[offense] = {str(k): int(v) for k, v in tier.value_counts(dropna=False).to_dict().items()}
        if reasons_col in df.columns:
            reasons = df[reasons_col].astype("string")
            if reasons.isna().any() or reasons.str.len().fillna(0).eq(0).any():
                issues.append(f"{label}: {reasons_col} has null/empty reason strings")
        if int((suppressed != expected_suppressed).sum()):
            issues.append(
                f"{label}: primary_index_suppressed_{offense} does not match denominator eligibility"
            )
        if int((mode.eq("non_residential") != expected_non_residential).sum()):
            issues.append(f"{label}: estimate_mode_{offense}=non_residential does not match the household rule")
        expected_special_mode = expected_special_suppressed & ~expected_non_residential & ~expected_denominator_invalid
        if int((mode.eq("special_use") != expected_special_mode).sum()):
            issues.append(f"{label}: estimate_mode_{offense}=special_use does not match the per-offense special-use rule")
        expected_insufficient_mode = (
            expected_insufficient_exposure
            & ~expected_non_residential
            & ~expected_denominator_invalid
            & ~expected_special_suppressed
        )
        if int((mode.eq("insufficient_exposure") != expected_insufficient_mode).sum()):
            issues.append(
                f"{label}: estimate_mode_{offense}=insufficient_exposure does not match the "
                "offense denominator floor"
            )
        if offense not in PERSON_EXPOSURE_FLOOR_OFFENSES and offense != "motor_vehicle_theft" and mode.eq("insufficient_exposure").any():
            issues.append(f"{label}: non-person-exposure offense {offense} has insufficient_exposure estimate mode")
        if offense == "motor_vehicle_theft":
            primary_invalid = (
                df[f"primary_denominator_invalid_{offense}"].astype(bool)
                if f"primary_denominator_invalid_{offense}" in df.columns
                else pd.Series(False, index=df.index)
            )
            resident_invalid = (
                df[f"resident_denominator_invalid_{offense}"].astype(bool)
                if f"resident_denominator_invalid_{offense}" in df.columns
                else pd.Series(False, index=df.index)
            )
            if int((primary_invalid != expected_denominator_invalid).sum()):
                issues.append(
                    f"{label}: primary_denominator_invalid_{offense} should be false under the v8 "
                    "vehicle-exposure floor policy"
                )
            if int((resident_invalid != expected_denominator_invalid).sum()):
                issues.append(
                    f"{label}: resident_denominator_invalid_{offense} should be false under the v8 "
                    "vehicle-exposure floor policy"
                )
            if int((mode.eq("vehicle_denominator_invalid") != expected_denominator_invalid).sum()):
                issues.append(
                    f"{label}: estimate_mode_{offense}=vehicle_denominator_invalid should not be used "
                    "under the v8 vehicle-exposure floor policy"
                )
            denominator_reason = df[f"denominator_reason_{offense}"].astype("string")
            resident_denominator_reason = df[f"resident_denominator_reason_{offense}"].astype("string")
            if denominator_reason.loc[expected_denominator_invalid].ne("vehicle_denominator_invalid").any():
                issues.append(f"{label}: denominator_reason_{offense} does not flag invalid vehicle denominators")
            if resident_denominator_reason.loc[expected_denominator_invalid].ne("vehicle_denominator_invalid").any():
                issues.append(
                    f"{label}: resident_denominator_reason_{offense} does not flag invalid vehicle denominators"
                )
        elif mode.eq("vehicle_denominator_invalid").any():
            issues.append(f"{label}: non-MVT offense {offense} has vehicle_denominator_invalid estimate mode")
        denominator_reason = df[f"denominator_reason_{offense}"].astype("string")
        resident_denominator_reason = df[f"resident_denominator_reason_{offense}"].astype("string")
        if denominator_reason.loc[expected_special_mode].ne("special_use").any():
            issues.append(f"{label}: denominator_reason_{offense} does not flag special_use rows")
        if resident_denominator_reason.loc[expected_special_mode].ne("special_use").any():
            issues.append(f"{label}: resident_denominator_reason_{offense} does not flag special_use rows")
        if df.loc[expected_special_mode, f"expected_count_{offense}"].isna().any():
            issues.append(f"{label}: expected_count_{offense} has nulls in special_use-suppressed rows")
        if denominator_reason.loc[expected_insufficient_mode].ne("insufficient_exposure").any():
            issues.append(f"{label}: denominator_reason_{offense} does not flag insufficient_exposure rows")
        if df.loc[expected_insufficient_mode, f"expected_count_{offense}"].isna().any():
            issues.append(f"{label}: expected_count_{offense} has nulls in insufficient_exposure-suppressed rows")
        real_housing_suppressed = (
            suppressed
            & households.ge(50.0)
            & ~expected_denominator_invalid
            & ~expected_special_suppressed
            & ~expected_insufficient_exposure
        )
        if bool(real_housing_suppressed.any()):
            issues.append(
                f"{label}: primary_index_suppressed_{offense} suppresses "
                f"{int(real_housing_suppressed.sum())} rows with households_total >= 50"
            )
        if value.loc[publishable].isna().any():
            issues.append(f"{label}: index_{offense}_primary has nulls on published rows")
        if rate.loc[publishable].isna().any():
            issues.append(f"{label}: rate_{offense}_primary has nulls on published rows")
        if value.loc[~publishable].notna().any():
            issues.append(f"{label}: index_{offense}_primary is populated on non-publishable rows")
        if rate.loc[~publishable].notna().any():
            issues.append(f"{label}: rate_{offense}_primary is populated on non-publishable rows")
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES or offense == "motor_vehicle_theft":
            floor = (
                PERSON_EXPOSURE_DENOMINATOR_FLOOR
                if offense in PERSON_EXPOSURE_FLOOR_OFFENSES
                else MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR
            )
            floor_label = "exposure" if offense in PERSON_EXPOSURE_FLOOR_OFFENSES else "vehicle exposure"
            ci_cols = [
                f"rate_{offense}_primary_ci95_lower",
                f"rate_{offense}_primary_ci95_upper",
                f"index_{offense}_primary_ci95_lower",
                f"index_{offense}_primary_ci95_upper",
                f"index_{offense}_primary_ci95_width",
                f"index_{offense}_primary_ci95_width_ratio",
            ]
            low_primary = denominator.lt(floor)
            leaked_primary_cols = [
                col
                for col in [f"rate_{offense}_primary", f"index_{offense}_primary", *ci_cols]
                if col in df.columns and pd.to_numeric(df.loc[low_primary, col], errors="coerce").notna().any()
            ]
            if leaked_primary_cols:
                issues.append(
                    f"{label}: {offense} published primary rate/index fields below "
                    f"{floor:g} {floor_label}: {leaked_primary_cols}"
                )
        expected_publishable = (~expected_suppressed) & denominator.gt(0.0)
        publishable_mismatch_count = int((publishable.to_numpy(dtype=bool) != expected_publishable.to_numpy(dtype=bool)).sum())
        if publishable_mismatch_count:
            issues.append(
                f"{label}: primary_index_publishable_{offense} is not exactly eligible and denominator > 0 "
                f"for {publishable_mismatch_count} rows"
            )
        primary_expected = _count_derived_rate_index(
            counts=count,
            denominator=denominator,
            publishable=publishable,
            national_rate_per_100k=pd.to_numeric(df.get(f"primary_national_rate_per_100k_{offense}"), errors="coerce"),
        )
        primary_rate_delta, primary_rate_null_mismatch = _max_abs_pair_delta(
            rate,
            pd.Series(primary_expected["rate"], index=df.index),
        )
        primary_index_delta, primary_index_null_mismatch = _max_abs_pair_delta(
            value,
            pd.Series(primary_expected["index"], index=df.index),
        )
        raw_rate_delta, raw_rate_null_mismatch = _max_abs_pair_delta(
            pd.to_numeric(df.get(f"raw_rate_{offense}"), errors="coerce"),
            pd.Series(primary_expected["rate"], index=df.index),
        )
        count_derived_max_abs[offense] = {
            "rate": primary_rate_delta,
            "index": primary_index_delta,
            "raw_rate": raw_rate_delta,
        }
        if primary_rate_null_mismatch or primary_rate_delta > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: rate_{offense}_primary is not count-derived "
                f"(max abs diff {primary_rate_delta:.3e}, null mismatches {primary_rate_null_mismatch})"
            )
        if primary_index_null_mismatch or primary_index_delta > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: index_{offense}_primary is not count-derived "
                f"(max abs diff {primary_index_delta:.3e}, null mismatches {primary_index_null_mismatch})"
            )
        if raw_rate_null_mismatch or raw_rate_delta > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: raw_rate_{offense} is not the count/denominator formula "
                f"(max abs diff {raw_rate_delta:.3e}, null mismatches {raw_rate_null_mismatch})"
            )

        resident_value = pd.to_numeric(df.get(f"index_{offense}_resident"), errors="coerce")
        resident_rate = pd.to_numeric(df.get(f"rate_{offense}_resident"), errors="coerce")
        resident_denominator = pd.to_numeric(df.get("resident_secondary_denominator"), errors="coerce").fillna(0.0)
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES:
            expected_resident_insufficient_exposure = resident_denominator.lt(PERSON_EXPOSURE_DENOMINATOR_FLOOR)
        elif offense == "motor_vehicle_theft":
            expected_resident_insufficient_exposure = denominator.lt(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR)
        else:
            expected_resident_insufficient_exposure = pd.Series(False, index=df.index)
        expected_resident_insufficient_exposure = expected_resident_insufficient_exposure.fillna(False).astype(bool)
        expected_resident_suppressed = (
            expected_non_residential
            | expected_denominator_invalid
            | expected_special_suppressed
            | expected_resident_insufficient_exposure
        )
        expected_resident_insufficient_reason = (
            expected_resident_insufficient_exposure
            & ~expected_non_residential
            & ~expected_denominator_invalid
            & ~expected_special_suppressed
        )
        if resident_denominator_reason.loc[expected_resident_insufficient_reason].ne("insufficient_exposure").any():
            issues.append(f"{label}: resident_denominator_reason_{offense} does not flag insufficient_exposure rows")
        if df.loc[expected_resident_insufficient_reason, f"expected_count_{offense}"].isna().any():
            issues.append(f"{label}: expected_count_{offense} has nulls in resident insufficient_exposure rows")
        resident_suppressed_col = f"index_{offense}_resident_suppressed"
        resident_suppressed = (
            df[resident_suppressed_col].astype(bool)
            if resident_suppressed_col in df.columns
            else pd.Series(False, index=df.index)
        )
        if int((resident_suppressed != expected_resident_suppressed).sum()):
            issues.append(
                f"{label}: {resident_suppressed_col} does not match denominator eligibility"
            )
        resident_publishable = (~expected_resident_suppressed) & resident_denominator.gt(0.0)
        if resident_value.loc[resident_publishable].isna().any():
            issues.append(f"{label}: index_{offense}_resident has nulls on published rows")
        if resident_rate.loc[resident_publishable].isna().any():
            issues.append(f"{label}: rate_{offense}_resident has nulls on published rows")
        if resident_value.loc[~resident_publishable].notna().any():
            issues.append(f"{label}: index_{offense}_resident is populated on resident non-publishable-denominator rows")
        if resident_rate.loc[~resident_publishable].notna().any():
            issues.append(f"{label}: rate_{offense}_resident is populated on resident non-publishable-denominator rows")
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES or offense == "motor_vehicle_theft":
            floor = (
                PERSON_EXPOSURE_DENOMINATOR_FLOOR
                if offense in PERSON_EXPOSURE_FLOOR_OFFENSES
                else MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR
            )
            floor_label = "residents" if offense in PERSON_EXPOSURE_FLOOR_OFFENSES else "vehicle exposure"
            low_resident = (
                resident_denominator.lt(floor)
                if offense in PERSON_EXPOSURE_FLOOR_OFFENSES
                else denominator.lt(floor)
            )
            leaked_resident_cols = [
                col
                for col in [f"rate_{offense}_resident", f"index_{offense}_resident"]
                if col in df.columns and pd.to_numeric(df.loc[low_resident, col], errors="coerce").notna().any()
            ]
            if leaked_resident_cols:
                issues.append(
                    f"{label}: {offense} published resident rate/index fields below "
                    f"{floor:g} {floor_label}: {leaked_resident_cols}"
                )
        resident_expected = _count_derived_rate_index(
            counts=count,
            denominator=resident_denominator,
            publishable=resident_publishable,
            national_rate_per_100k=pd.to_numeric(df.get(f"resident_national_rate_per_100k_{offense}"), errors="coerce"),
        )
        resident_rate_delta, resident_rate_null_mismatch = _max_abs_pair_delta(
            resident_rate,
            pd.Series(resident_expected["rate"], index=df.index),
        )
        resident_index_delta, resident_index_null_mismatch = _max_abs_pair_delta(
            resident_value,
            pd.Series(resident_expected["index"], index=df.index),
        )
        resident_raw_delta, resident_raw_null_mismatch = _max_abs_pair_delta(
            pd.to_numeric(df.get(f"resident_raw_rate_{offense}"), errors="coerce"),
            pd.Series(resident_expected["rate"], index=df.index),
        )
        count_derived_max_abs[offense].update(
            {
                "resident_rate": resident_rate_delta,
                "resident_index": resident_index_delta,
                "resident_raw_rate": resident_raw_delta,
            }
        )
        if resident_rate_null_mismatch or resident_rate_delta > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: rate_{offense}_resident is not count-derived "
                f"(max abs diff {resident_rate_delta:.3e}, null mismatches {resident_rate_null_mismatch})"
            )
        if resident_index_null_mismatch or resident_index_delta > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: index_{offense}_resident is not count-derived "
                f"(max abs diff {resident_index_delta:.3e}, null mismatches {resident_index_null_mismatch})"
            )
        if resident_raw_null_mismatch or resident_raw_delta > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: resident_raw_rate_{offense} is not the count/resident-denominator formula "
                f"(max abs diff {resident_raw_delta:.3e}, null mismatches {resident_raw_null_mismatch})"
            )

    aggregate_max_abs: dict[str, float] = {}
    aggregate_specs = {
        "index_total_part1_resident": OFFENSES_7,
        "index_personal_part1_resident": PERSONAL_OFFENSES,
        "index_property_part1_resident": PROPERTY_OFFENSES,
    }
    for field, offenses in aggregate_specs.items():
        actual = pd.to_numeric(df.get(field), errors="coerce")
        expected, national_rate, publishable = _resident_part1_expected(df, offenses=list(offenses))
        max_abs, null_mismatch = _max_abs_pair_delta(actual, expected)
        aggregate_max_abs[field] = max_abs
        if null_mismatch or max_abs > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: {field} is not the all-or-null count-derived resident aggregate "
                f"(national_rate_per_100k {national_rate:.12g}, max abs diff {max_abs:.3e}, "
                f"null mismatches {null_mismatch})"
            )
        if int((actual.notna() != publishable).sum()):
            issues.append(f"{label}: {field} does not match component all-or-null publishability")

    primary_composite_specs = {
        "index_total_primary_event_weighted": _national_expected_count_weights(df, list(OFFENSES_7)),
        "index_total_equal_offense": {offense: 1.0 for offense in OFFENSES_7},
    }
    for field, weights in primary_composite_specs.items():
        actual = pd.to_numeric(df.get(field), errors="coerce")
        expected, publishable = _primary_composite_expected(df, offenses=list(OFFENSES_7), weights=weights)
        max_abs, null_mismatch = _max_abs_pair_delta(actual, expected)
        aggregate_max_abs[field] = max_abs
        if null_mismatch or max_abs > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: {field} is not the all-or-null primary-index composite "
                f"(max abs diff {max_abs:.3e}, null mismatches {null_mismatch})"
            )
        if int((actual.notna() != publishable).sum()):
            issues.append(f"{label}: {field} does not match seven-component all-or-null publishability")

    harm_actual = pd.to_numeric(df.get("index_total_harm"), errors="coerce")
    harm_expected, harm_national_rate, harm_publishable = _harm_total_expected(df)
    harm_max_abs, harm_null_mismatch = _max_abs_pair_delta(harm_actual, harm_expected)
    aggregate_max_abs["index_total_harm"] = harm_max_abs
    if harm_null_mismatch or harm_max_abs > COUNT_DERIVED_TOLERANCE:
        issues.append(
            f"{label}: index_total_harm is not the count-derived harm-weighted person-exposure index "
            f"(national_harm_rate_per_100k {harm_national_rate:.12g}, max abs diff {harm_max_abs:.3e}, "
            f"null mismatches {harm_null_mismatch})"
        )
    if int((harm_actual.notna() != harm_publishable).sum()):
        issues.append(f"{label}: index_total_harm nulls do not match person-exposure publishability")

    if "non_residential_flag" in df.columns:
        non_residential = df["non_residential_flag"].fillna(False).astype(bool)
        aggregate_populated_non_residential = [
            field
            for field in AGGREGATE_INDEX_FIELDS
            if field in df.columns and pd.to_numeric(df.loc[non_residential, field], errors="coerce").notna().any()
        ]
        if aggregate_populated_non_residential:
            issues.append(
                f"{label}: aggregate indexes populated in non-residential cells: "
                f"{aggregate_populated_non_residential}"
            )
    mvt_invalid_col = "primary_denominator_invalid_motor_vehicle_theft"
    if mvt_invalid_col in df.columns:
        mvt_invalid = df[mvt_invalid_col].fillna(False).astype(bool)
        must_null_on_mvt_invalid = [
            "index_total_part1_resident",
            "index_property_part1_resident",
            "index_total_primary_event_weighted",
            "index_total_equal_offense",
        ]
        populated = [
            field
            for field in must_null_on_mvt_invalid
            if field in df.columns and pd.to_numeric(df.loc[mvt_invalid, field], errors="coerce").notna().any()
        ]
        if populated:
            issues.append(f"{label}: MVT-invalid rows populate aggregate fields that require MVT: {populated}")
        _, _, personal_publishable = _resident_part1_expected(df, offenses=list(PERSONAL_OFFENSES))
        personal_actual = pd.to_numeric(df.get("index_personal_part1_resident"), errors="coerce")
        personal_allowed = mvt_invalid & personal_publishable
        if personal_actual.loc[personal_allowed].isna().any():
            issues.append(f"{label}: MVT-invalid rows suppress publishable index_personal_part1_resident")
        # index_total_harm consumes counts, not indices: MVT-invalid rows must still publish it
        # wherever person exposure is publishable.
        harm_allowed = mvt_invalid & harm_publishable
        if harm_actual.loc[harm_allowed].isna().any():
            issues.append(f"{label}: MVT-invalid rows suppress publishable index_total_harm")

    return {
        "label": label,
        "path": str(path),
        "present": True,
        "rows": int(len(df)),
        "state_count": int(len(states)),
        "expected_count_total_sum": float(pd.to_numeric(df.get("expected_count_total"), errors="coerce").fillna(0.0).sum()),
        "duplicate_id_count": duplicate_count,
        "excluded_states_present": excluded_present,
        "max_abs_count_personal_diff": _max_abs(personal_diff) if not personal_diff.empty else None,
        "max_abs_count_property_diff": _max_abs(property_diff) if not property_diff.empty else None,
        "max_abs_count_total_diff": _max_abs(total_diff) if not total_diff.empty else None,
        "density_max_abs": density_max_abs,
        "count_derived_max_abs": count_derived_max_abs,
        "aggregate_max_abs": aggregate_max_abs,
        "urban_stratum_counts": urban_stratum_counts,
        "confidence_tier_counts": confidence_tier_counts,
        "domain_overlap_ranges": domain_overlap_ranges,
    }


def _manifest_path_exists(path_value: object) -> bool:
    if path_value is None:
        return False
    try:
        path_text = str(path_value)
        direct = Path(path_text)
        if direct.exists():
            return True
        normalized = path_text.replace("\\", "/")
        for marker in MANIFEST_RELATIVE_ROOT_MARKERS:
            idx = normalized.find(marker)
            if idx < 0:
                continue
            candidate = REPO_ROOT / normalized[idx:]
            if candidate.exists():
                return True
        return False
    except OSError:
        return False


def _load_burglary_tau_calibration(*, issues: list[str]) -> dict[str, Any]:
    if not REPO_BURGLARY_TAU_CALIBRATION.exists():
        issues.append(f"missing burglary tau calibration artifact: {REPO_BURGLARY_TAU_CALIBRATION}")
        return {"present": False, "path": str(REPO_BURGLARY_TAU_CALIBRATION)}
    try:
        data = json.loads(REPO_BURGLARY_TAU_CALIBRATION.read_text())
    except Exception as exc:  # pragma: no cover - diagnostics should preserve the underlying error.
        issues.append(f"could not read burglary tau calibration artifact {REPO_BURGLARY_TAU_CALIBRATION}: {exc}")
        return {"present": True, "path": str(REPO_BURGLARY_TAU_CALIBRATION), "readable": False}

    grid = [float(value) for value in data.get("tau_grid", data.get("grid", []))]
    expected_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    if grid != expected_grid:
        issues.append(f"burglary tau calibration grid {grid} != {expected_grid}")
    production_tau = data.get("production_tau")
    if production_tau is None:
        production_tau = data.get("selected_tau_after_backstop")
    if production_tau is None:
        production_tau = data.get("one_se_tau")
    try:
        production_tau = float(production_tau)
    except (TypeError, ValueError):
        issues.append(f"burglary tau calibration missing numeric production tau: {production_tau}")
        production_tau = float("nan")
    if not np.isfinite(production_tau) or production_tau < 0.0 or production_tau > 1.0:
        issues.append(f"burglary tau calibration production tau outside [0, 1]: {production_tau}")
    return {
        "present": True,
        "path": str(REPO_BURGLARY_TAU_CALIBRATION),
        "grid": grid,
        "production_tau": production_tau,
        "one_se_tau": data.get("one_se_tau"),
        "argmin_tau": data.get("argmin_tau"),
        "backstop_applied": bool(data.get("gradient_backstop", {}).get("applied", False))
        if isinstance(data.get("gradient_backstop"), dict)
        else False,
    }


def _resolve_manifest_path(path_value: object) -> Path | None:
    if path_value is None:
        return None
    try:
        path_text = str(path_value)
        direct = Path(path_text)
        if direct.exists():
            return direct
        normalized = path_text.replace("\\", "/")
        for marker in MANIFEST_RELATIVE_ROOT_MARKERS:
            idx = normalized.find(marker)
            if idx < 0:
                continue
            candidate = REPO_ROOT / normalized[idx:]
            if candidate.exists():
                return candidate
        return direct
    except OSError:
        return None


def _check_residual_feature_policy_manifest(
    *,
    resolved: dict[str, Any],
    summary: dict[str, Any],
    input_stats: dict[str, Any],
    issues: list[str],
) -> dict[str, Any]:
    policy_path_value = resolved.get("residual_feature_policy_path")
    policy_classes = {str(value) for value in resolved.get("residual_exclude_feature_policy_classes", [])}
    expected_by_offense = EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE
    resolved_by_offense_raw = resolved.get("residual_exclude_feature_policy_classes_by_offense")
    resolved_by_offense = {
        str(offense): {
            str(value)
            for value in (
                resolved_by_offense_raw.get(offense, [])
                if isinstance(resolved_by_offense_raw, dict)
                else []
            )
        }
        for offense in OFFENSES_7
    }
    if not policy_path_value or EXPECTED_RESIDUAL_FEATURE_POLICY_PATH_FRAGMENT not in str(policy_path_value):
        issues.append(f"build manifest has unexpected residual feature-policy path: {policy_path_value}")
    if policy_classes != EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES:
        issues.append(
            "build manifest residual feature-policy excluded classes "
            f"{sorted(policy_classes)} != {sorted(EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES)}"
        )
    if resolved_by_offense != expected_by_offense:
        issues.append(
            "build manifest residual feature-policy per-offense excluded classes "
            f"{ {k: sorted(v) for k, v in resolved_by_offense.items()} } != "
            f"{ {k: sorted(v) for k, v in expected_by_offense.items()} }"
        )
    policy_path = _resolve_manifest_path(policy_path_value)
    if policy_path is None or not policy_path.exists():
        issues.append(f"build manifest residual feature-policy path does not exist: {policy_path_value}")

    input_stat = input_stats.get("residual_feature_policy")
    if not isinstance(input_stat, dict) or input_stat.get("exists") is not True:
        issues.append("build manifest input stat missing residual feature-policy parquet")

    application = summary.get("city_residual_feature_policy")
    if not isinstance(application, dict) or not application:
        issues.append("build manifest missing city_residual_feature_policy application summary")
        return {
            "present": False,
            "path": str(policy_path_value),
            "exclude_final_classes": sorted(policy_classes),
        }

    app_path = application.get("path")
    app_classes = {str(value) for value in application.get("exclude_final_classes", [])}
    app_by_offense_raw = application.get("exclude_final_classes_by_offense")
    app_by_offense = {
        str(offense): {
            str(value)
            for value in (
                app_by_offense_raw.get(offense, [])
                if isinstance(app_by_offense_raw, dict)
                else []
            )
        }
        for offense in OFFENSES_7
    }
    if not app_path or EXPECTED_RESIDUAL_FEATURE_POLICY_PATH_FRAGMENT not in str(app_path):
        issues.append(f"city_residual_feature_policy has unexpected policy path: {app_path}")
    if app_classes != EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES:
        issues.append(
            "city_residual_feature_policy excluded classes "
            f"{sorted(app_classes)} != {sorted(EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES)}"
        )
    if app_by_offense != expected_by_offense:
        issues.append(
            "city_residual_feature_policy per-offense excluded classes "
            f"{ {k: sorted(v) for k, v in app_by_offense.items()} } != "
            f"{ {k: sorted(v) for k, v in expected_by_offense.items()} }"
        )
    if not isinstance(application.get("excluded_feature_count"), int):
        issues.append("city_residual_feature_policy missing integer excluded_feature_count")

    selected_cols = [str(col) for col in application.get("selected_feature_cols", [])]
    excluded_cols = [str(col) for col in application.get("excluded_feature_cols", [])]
    candidate_cols = [str(col) for col in application.get("candidate_feature_cols", [])]
    selected_by_offense_raw = application.get("selected_feature_cols_by_offense")
    excluded_by_offense_raw = application.get("excluded_feature_cols_by_offense")
    selected_by_offense = {
        offense: [
            str(col)
            for col in (
                selected_by_offense_raw.get(offense, [])
                if isinstance(selected_by_offense_raw, dict)
                else []
            )
        ]
        for offense in OFFENSES_7
    }
    excluded_by_offense = {
        offense: [
            str(col)
            for col in (
                excluded_by_offense_raw.get(offense, [])
                if isinstance(excluded_by_offense_raw, dict)
                else []
            )
        ]
        for offense in OFFENSES_7
    }
    if not candidate_cols:
        candidate_cols = sorted(set(selected_cols) | set(excluded_cols))
    if not selected_cols:
        issues.append("city_residual_feature_policy missing selected residual feature union list")
    if not isinstance(selected_by_offense_raw, dict) or not all(selected_by_offense.values()):
        issues.append("city_residual_feature_policy missing selected residual feature lists by offense")
    if not isinstance(excluded_by_offense_raw, dict):
        issues.append("city_residual_feature_policy missing excluded residual feature lists by offense")
    if not candidate_cols:
        return {
            "present": True,
            "path": str(policy_path_value),
            "exclude_final_classes": sorted(policy_classes),
            "selected_feature_count": 0,
            "excluded_feature_count": application.get("excluded_feature_count"),
        }
    if policy_path is None or not policy_path.exists():
        return {
            "present": True,
            "path": str(policy_path_value),
            "exclude_final_classes": sorted(policy_classes),
            "selected_feature_count": len(selected_cols),
            "excluded_feature_count": application.get("excluded_feature_count"),
        }

    try:
        policy = pd.read_parquet(policy_path, columns=["feature_column", "final_class"])
    except Exception as exc:  # pragma: no cover - validation diagnostics should preserve the original error.
        issues.append(f"could not read residual feature-policy parquet {policy_path}: {exc}")
        return {
            "present": True,
            "path": str(policy_path),
            "exclude_final_classes": sorted(policy_classes),
            "selected_feature_count": len(selected_cols),
            "excluded_feature_count": application.get("excluded_feature_count"),
        }
    policy["feature_column"] = policy["feature_column"].astype("string")
    policy = policy[policy["feature_column"].notna()].copy()
    policy["feature_column"] = policy["feature_column"].astype(str)
    policy["final_class"] = policy["final_class"].astype("string").fillna("").astype(str)
    policy = policy.drop_duplicates("feature_column", keep="first").set_index("feature_column")
    unmapped_candidate = sorted(col for col in candidate_cols if col not in policy.index)
    if unmapped_candidate:
        issues.append(f"candidate residual features are missing from policy parquet: {unmapped_candidate}")
    recomputed_excluded_union: set[str] = set()
    by_offense_summary: dict[str, Any] = {}
    for offense in OFFENSES_7:
        offense_selected = selected_by_offense[offense]
        offense_excluded = excluded_by_offense[offense]
        expected_classes = expected_by_offense[offense]
        selected_unmapped = sorted(col for col in offense_selected if col not in policy.index)
        excluded_unmapped = sorted(col for col in offense_excluded if col not in policy.index)
        if selected_unmapped:
            issues.append(f"{offense} selected residual features are missing from policy parquet: {selected_unmapped}")
        if excluded_unmapped:
            issues.append(f"{offense} excluded residual features are missing from policy parquet: {excluded_unmapped}")
        recomputed_excluded = sorted(
            col
            for col in candidate_cols
            if col in policy.index and str(policy.at[col, "final_class"]) in expected_classes
        )
        recomputed_excluded_union.update(recomputed_excluded)
        if sorted(offense_excluded) != recomputed_excluded:
            issues.append(
                f"city_residual_feature_policy {offense} excluded column list does not match policy recomputation: "
                f"manifest={sorted(offense_excluded)} recomputed={recomputed_excluded}"
            )
        selected_forbidden = sorted(
            col
            for col in offense_selected
            if col in policy.index and str(policy.at[col, "final_class"]) in expected_classes
        )
        if selected_forbidden:
            issues.append(
                f"{offense} between-only/protected residual features remain selected for within-allocation: "
                f"{selected_forbidden}"
            )
        if not expected_classes and sorted(offense_selected) != sorted(candidate_cols):
            issues.append(
                f"{offense} is configured as a residual-policy carve-out but does not retain every candidate feature"
            )
        by_offense_summary[offense] = {
            "exclude_final_classes": sorted(expected_classes),
            "candidate_feature_count": int(len(candidate_cols)),
            "selected_feature_count": int(len(offense_selected)),
            "excluded_feature_count": int(len(offense_excluded)),
            "recomputed_excluded_feature_count": int(len(recomputed_excluded)),
        }
    recomputed_excluded_union_sorted = sorted(recomputed_excluded_union)
    if sorted(excluded_cols) != recomputed_excluded_union_sorted:
        issues.append(
            "city_residual_feature_policy excluded column union does not match policy recomputation: "
            f"manifest={sorted(excluded_cols)} recomputed={recomputed_excluded_union_sorted}"
        )
    if application.get("excluded_feature_count") != len(recomputed_excluded_union_sorted):
        issues.append(
            "city_residual_feature_policy excluded_feature_count "
            f"{application.get('excluded_feature_count')} != recomputed {len(recomputed_excluded_union_sorted)}"
        )
    if application.get("selected_feature_count") != len(selected_cols):
        issues.append(
            "city_residual_feature_policy selected_feature_count "
            f"{application.get('selected_feature_count')} != selected list length {len(selected_cols)}"
        )
    if application.get("candidate_feature_count") != len(candidate_cols):
        issues.append(
            "city_residual_feature_policy candidate_feature_count "
            f"{application.get('candidate_feature_count')} != selected+excluded length {len(candidate_cols)}"
        )

    return {
        "present": True,
        "path": str(policy_path),
        "exclude_final_classes": sorted(policy_classes),
        "candidate_feature_count": int(len(candidate_cols)),
        "selected_feature_count": int(len(selected_cols)),
        "excluded_feature_count": int(len(excluded_cols)),
        "recomputed_excluded_feature_count": int(len(recomputed_excluded_union_sorted)),
        "retained_proxy_review_feature_count": application.get("retained_proxy_review_feature_count"),
        "by_offense": by_offense_summary,
    }


def _check_build_manifest(
    *,
    output_dir: Path,
    issues: list[str],
    burglary_tau_calibration: dict[str, Any],
) -> dict[str, Any]:
    build_manifest_path = output_dir / "manifest.json"
    if not build_manifest_path.exists():
        build_manifest_path = output_dir / f"crimerisk_output_build_{YEAR}.json"
    manifest = _load_json(build_manifest_path)
    if manifest is None:
        issues.append(f"missing output build manifest under {output_dir}")
        return {"path": str(build_manifest_path), "present": False}

    resolved = manifest.get("resolved_config", {})
    summary = manifest.get("summary", {})
    output_stats = manifest.get("output_file_stats", {})
    input_stats = manifest.get("input_file_stats", {})

    if int(manifest.get("year") or 0) != YEAR:
        issues.append(f"build manifest year is {manifest.get('year')}, expected {YEAR}")
    burglary_gate_derivation_path = REPO_ROOT / "state" / "modeling" / "burglary_gate_ceiling_derivation.json"
    if not burglary_gate_derivation_path.exists():
        issues.append(f"missing burglary gate ceiling derivation artifact: {burglary_gate_derivation_path}")
    if summary.get("block_groups") != EXPECTED_ROW_COUNTS["block_group"]:
        issues.append(
            "build manifest block-group count "
            f"{summary.get('block_groups')} != {EXPECTED_ROW_COUNTS['block_group']}"
        )
    if summary.get("tracts") != EXPECTED_ROW_COUNTS["tract"]:
        issues.append(f"build manifest tract count {summary.get('tracts')} != {EXPECTED_ROW_COUNTS['tract']}")
    if summary.get("fbi_calibrated_written") is not True:
        issues.append("build manifest says FBI-calibrated outputs were not written")
    burglary_calibration = summary.get("burglary_commercial_calibration")
    if not isinstance(burglary_calibration, dict):
        issues.append("build manifest missing burglary_commercial_calibration")
        burglary_calibration = {}
    k_commercial = pd.to_numeric(pd.Series([burglary_calibration.get("k_commercial")]), errors="coerce").iloc[0]
    if pd.isna(k_commercial) or float(k_commercial) <= 0.0:
        issues.append(f"build manifest has invalid burglary k_commercial: {burglary_calibration.get('k_commercial')}")
    if burglary_calibration.get("denominator_form") != "households_destination_poi_retail_industrial_jobs_nnls":
        issues.append(
            "build manifest burglary calibration has unexpected denominator_form: "
            f"{burglary_calibration.get('denominator_form')}"
        )
    k_vector = burglary_calibration.get("k_vector")
    if not isinstance(k_vector, dict):
        issues.append("build manifest burglary calibration missing k_vector")
        k_vector = {}
    for key in ("k_destination_poi", "k_retail_jobs", "k_industrial_jobs"):
        value = pd.to_numeric(pd.Series([k_vector.get(key, burglary_calibration.get(key))]), errors="coerce").iloc[0]
        if pd.isna(value) or float(value) < 0.0:
            issues.append(f"build manifest burglary calibration has invalid {key}: {k_vector.get(key)}")
    if burglary_calibration.get("beats_single_term_nnls_residual") is not True:
        issues.append("build manifest burglary multi-term NNLS does not beat the single-term residual")
    if burglary_calibration.get("calibration_source") != "covered_city_direct_burglary_incident_nnls":
        issues.append(
            "build manifest burglary calibration did not use covered-city direct burglary incident NNLS: "
            f"{burglary_calibration.get('calibration_source')}"
        )
    if burglary_calibration.get("used_fallback") is True:
        issues.append("build manifest burglary calibration used fallback instead of raw direct counts")

    burglary_gradient = summary.get("burglary_commercial_gradient")
    if not isinstance(burglary_gradient, dict):
        issues.append("build manifest missing burglary_commercial_gradient")
        burglary_gradient = {}
    bg_gradient = burglary_gradient.get("block_group_ags_core")
    if not isinstance(bg_gradient, dict) or bg_gradient.get("ok") is not True:
        issues.append(f"build manifest missing valid block_group_ags_core burglary gradient: {bg_gradient}")
        bg_gradient = {}
    after_direct_gradient = pd.to_numeric(
        pd.Series([bg_gradient.get("after_q5_q1_mean_direct")]), errors="coerce"
    ).iloc[0]
    after_modeled_gradient = pd.to_numeric(
        pd.Series([bg_gradient.get("after_q5_q1_mean_modeled")]), errors="coerce"
    ).iloc[0]
    if pd.isna(after_direct_gradient):
        issues.append("block_group_ags_core burglary direct-city commercial-share gradient missing from build manifest")
    elif (
        float(after_direct_gradient) < BURGLARY_COMMERCIAL_GRADIENT_DIRECT_MIN
        or float(after_direct_gradient) > BURGLARY_COMMERCIAL_GRADIENT_DIRECT_MAX
    ):
        issues.append(
            "block_group_ags_core burglary direct-city commercial-share gradient outside approved range "
            f"[{BURGLARY_COMMERCIAL_GRADIENT_DIRECT_MIN:g}, {BURGLARY_COMMERCIAL_GRADIENT_DIRECT_MAX:g}]: "
            f"{after_direct_gradient}"
        )
    if pd.isna(after_modeled_gradient):
        issues.append("block_group_ags_core burglary modeled-transfer commercial-share gradient missing from build manifest")
    elif (
        float(after_modeled_gradient) < BURGLARY_COMMERCIAL_GRADIENT_MODELED_MIN
        or float(after_modeled_gradient) > BURGLARY_COMMERCIAL_GRADIENT_MODELED_MAX
    ):
        issues.append(
            "block_group_ags_core burglary modeled-transfer commercial-share gradient outside approved range "
            f"[{BURGLARY_COMMERCIAL_GRADIENT_MODELED_MIN:g}, {BURGLARY_COMMERCIAL_GRADIENT_MODELED_MAX:g}]: "
            f"{after_modeled_gradient}"
        )

    suppression_counts = summary.get("suppression_mode_counts")
    if not isinstance(suppression_counts, dict):
        issues.append("build manifest missing suppression_mode_counts")
        suppression_counts = {}
    aggregate_normalizers = summary.get("aggregate_index_normalizers")
    if not isinstance(aggregate_normalizers, dict):
        issues.append("build manifest missing aggregate_index_normalizers")
        aggregate_normalizers = {}
    for surface_key in ("block_group_ags_core", "tract_ags_core", "block_group_fbi_calibrated", "tract_fbi_calibrated"):
        surface_normalizers = aggregate_normalizers.get(surface_key)
        if not isinstance(surface_normalizers, dict):
            issues.append(f"build manifest missing aggregate normalizers for {surface_key}")
            continue
        resident = surface_normalizers.get("resident_part1")
        if not isinstance(resident, dict):
            issues.append(f"build manifest missing resident_part1 aggregate normalizers for {surface_key}")
        else:
            for field in (
                "index_total_part1_resident",
                "index_personal_part1_resident",
                "index_property_part1_resident",
            ):
                row = resident.get(field)
                if not isinstance(row, dict) or pd.isna(row.get("national_rate_per_100k")):
                    issues.append(f"build manifest missing national_rate_per_100k for {surface_key}.{field}")
        for key, field in (
            ("primary_event_weighted", "index_total_primary_event_weighted"),
            ("primary_equal_offense", "index_total_equal_offense"),
            ("primary_harm_weighted", "index_total_harm"),
        ):
            row = surface_normalizers.get(key)
            if not isinstance(row, dict) or row.get("field") != field or not isinstance(row.get("offense_weights"), dict):
                issues.append(f"build manifest missing reproducible weights for {surface_key}.{field}")

    if resolved.get("use_promoted_next_phase_allocator") is not True:
        issues.append("build manifest says promoted next-phase allocator was disabled")
    if resolved.get("promoted_next_phase_allocator_applied") is not True:
        issues.append("build manifest does not prove promoted next-phase allocator was applied")
    if summary.get("promoted_next_phase_allocator_applied") is not True:
        issues.append("build manifest summary does not prove promoted next-phase allocator was applied")

    residual_training_path = resolved.get("residual_training_city_shares_path")
    if not residual_training_path or f"next_phase_validation_city_incident_share_surface_{YEAR}.parquet" not in str(residual_training_path):
        issues.append(f"build manifest has unexpected residual-training surface: {residual_training_path}")
    if not _manifest_path_exists(residual_training_path):
        issues.append(f"build manifest residual-training surface does not exist: {residual_training_path}")

    excluded_case_types = set(str(value) for value in resolved.get("residual_training_exclude_validation_case_types", []))
    if "suburban_county_validation_case" not in excluded_case_types:
        issues.append("build manifest does not exclude suburban_county_validation_case from residual training")

    extra_paths = [str(path) for path in resolved.get("residual_training_extra_bg_feature_paths", [])]
    for fragment in EXPECTED_PROMOTED_RESIDUAL_FEATURE_PATH_FRAGMENTS:
        matches = [path for path in extra_paths if fragment in path]
        if not matches:
            issues.append(f"build manifest missing promoted residual feature path containing {fragment}")
        elif not any(_manifest_path_exists(path) for path in matches):
            issues.append(f"build manifest promoted residual feature path does not exist for {fragment}")

    for key in ("block_group_ags_core", "tract_ags_core", "block_group_fbi_calibrated", "tract_fbi_calibrated"):
        stat = output_stats.get(key)
        if not isinstance(stat, dict) or stat.get("exists") is not True:
            issues.append(f"build manifest output stat missing or non-present for {key}")

    residual_stat = input_stats.get("residual_training_city_shares")
    if not isinstance(residual_stat, dict) or residual_stat.get("exists") is not True:
        issues.append("build manifest input stat missing residual-training city-share surface")
    extra_stats = input_stats.get("residual_training_extra_bg_features")
    if not isinstance(extra_stats, list) or len(extra_stats) < len(EXPECTED_PROMOTED_RESIDUAL_FEATURE_PATH_FRAGMENTS):
        issues.append("build manifest input stats missing promoted residual feature files")
    elif any(not isinstance(stat, dict) or stat.get("exists") is not True for stat in extra_stats):
        issues.append("build manifest input stats include missing promoted residual feature files")

    residual_feature_policy_summary = _check_residual_feature_policy_manifest(
        resolved=resolved,
        summary=summary,
        input_stats=input_stats,
        issues=issues,
    )
    expected_tau = dict(DEFAULT_EXPECTED_RESIDUAL_TRANSFER_TAU_BY_OFFENSE)
    burglary_tau = burglary_tau_calibration.get("production_tau")
    if burglary_tau is not None and np.isfinite(float(burglary_tau)):
        expected_tau["burglary"] = float(burglary_tau)
    resolved_tau_raw = resolved.get("residual_transfer_tau_by_offense")
    summary_tau_raw = summary.get("residual_transfer_tau_by_offense")
    resolved_tau = {
        offense: float(
            resolved_tau_raw.get(offense, DEFAULT_EXPECTED_RESIDUAL_TRANSFER_TAU_BY_OFFENSE[offense])
            if isinstance(resolved_tau_raw, dict)
            else DEFAULT_EXPECTED_RESIDUAL_TRANSFER_TAU_BY_OFFENSE[offense]
        )
        for offense in OFFENSES_7
    }
    summary_tau = {
        offense: float(
            summary_tau_raw.get(offense, DEFAULT_EXPECTED_RESIDUAL_TRANSFER_TAU_BY_OFFENSE[offense])
            if isinstance(summary_tau_raw, dict)
            else DEFAULT_EXPECTED_RESIDUAL_TRANSFER_TAU_BY_OFFENSE[offense]
        )
        for offense in OFFENSES_7
    }
    for source_name, observed_tau in (("resolved_config", resolved_tau), ("summary", summary_tau)):
        for offense, expected_value in expected_tau.items():
            observed_value = observed_tau.get(offense)
            if abs(float(observed_value) - float(expected_value)) > TRANSFER_POLICY_TOLERANCE:
                issues.append(
                    f"build manifest {source_name} residual tau for {offense} "
                    f"{observed_value} != expected {expected_value}"
                )
    tau_stat = input_stats.get("burglary_tau_calibration")
    if not isinstance(tau_stat, dict) or tau_stat.get("exists") is not True:
        issues.append("build manifest input stat missing burglary tau calibration artifact")

    return {
        "path": str(build_manifest_path),
        "present": True,
        "year": manifest.get("year"),
        "created_at_utc": manifest.get("created_at_utc"),
        "promoted_next_phase_allocator_applied": bool(
            resolved.get("promoted_next_phase_allocator_applied")
        ),
        "residual_training_city_shares_path": residual_training_path,
        "residual_training_exclude_validation_case_types": sorted(excluded_case_types),
        "residual_training_extra_bg_feature_paths": extra_paths,
        "city_residual_feature_policy": residual_feature_policy_summary,
        "residual_transfer_tau_by_offense": resolved_tau,
        "expected_residual_transfer_tau_by_offense": expected_tau,
        "burglary_tau_calibration": burglary_tau_calibration,
        "aggregate_index_normalizer_surfaces": sorted(aggregate_normalizers.keys()),
        "burglary_commercial_calibration": burglary_calibration,
        "burglary_commercial_gradient_block_group_ags_core": bg_gradient,
        "suppression_mode_count_surfaces": sorted(suppression_counts.keys()),
        "output_file_stats_present": sorted(output_stats.keys()),
    }


def _load_validation_or_repo_json(*, validation_name: str, repo_path: Path) -> dict[str, Any] | None:
    return _load_json(PACKAGE_VALIDATION_DIR / validation_name) or _load_json(repo_path)


def _check_next_phase_measurement(*, issues: list[str]) -> dict[str, Any]:
    measurement = _load_validation_or_repo_json(
        validation_name=f"next_phase_measurement_summary_{YEAR}.json",
        repo_path=REPO_NEXT_PHASE_MEASUREMENT,
    )
    if measurement is None:
        issues.append("missing next-phase measurement summary")
        return {"present": False}

    truth_case_count = int(measurement.get("truth_case_count") or 0)
    truth_city_count = int(measurement.get("truth_city_count") or 0)
    error_budget_rows = int(measurement.get("error_budget_rows") or 0)
    cv_prediction_rows = int(measurement.get("cv_prediction_rows") or 0)
    recommended = str(measurement.get("recommended_next_workstream") or "")
    split_modes = set(str(value) for value in measurement.get("split_modes", []))
    required_split_modes = {"kfold", "leave_large_city_out", "leave_one_city_out"}
    class_counts = measurement.get("error_budget_class_counts", {})

    if truth_case_count < 22:
        issues.append(f"next-phase measurement has {truth_case_count} truth cases, expected at least 22")
    if truth_city_count < 21:
        issues.append(f"next-phase measurement has {truth_city_count} truth cities, expected at least 21")
    if error_budget_rows < 120:
        issues.append(f"next-phase measurement has {error_budget_rows} error-budget rows, expected at least 120")
    if cv_prediction_rows <= 0:
        issues.append("next-phase measurement has no held-out CV prediction rows")
    if recommended != "allocator_expansion_first":
        issues.append(f"next-phase measurement recommends {recommended!r}, expected allocator_expansion_first")
    if not required_split_modes.issubset(split_modes):
        issues.append(
            "next-phase measurement missing held-out split modes "
            f"{sorted(required_split_modes - split_modes)}"
        )
    if not isinstance(class_counts, dict) or "allocation_dominated" not in class_counts:
        issues.append("next-phase measurement missing allocation-dominated error-budget class count")

    return {
        "present": True,
        "truth_case_count": truth_case_count,
        "truth_city_count": truth_city_count,
        "error_budget_rows": error_budget_rows,
        "cv_prediction_rows": cv_prediction_rows,
        "recommended_next_workstream": recommended,
        "split_modes": sorted(split_modes),
        "error_budget_class_counts": class_counts,
    }


def _check_dashboard_lookup(*, issues: list[str]) -> dict[str, Any]:
    dashboard = _load_validation_or_repo_json(
        validation_name=f"dashboard_neighborhood_check_lookup_{YEAR}.json",
        repo_path=REPO_DASHBOARD_LOOKUP,
    )
    if dashboard is None:
        issues.append("missing dashboard neighborhood lookup validation summary")
        return {"present": False}

    basis = str(dashboard.get("neighborhood_basis") or "")
    neighborhood_count = int(dashboard.get("neighborhood_count") or 0)
    tract_weight_rows = int(dashboard.get("tract_weight_rows") or 0)
    dashboard_rows = int(dashboard.get("dashboard_coarse_rows") or 0)
    risk_rows = int(dashboard.get("dashboard_risk_score_rows") or 0)

    if basis != "tract_lookup":
        issues.append(f"dashboard lookup validation basis is {basis!r}, expected 'tract_lookup'")
    if neighborhood_count <= 0:
        issues.append("dashboard lookup validation has no neighborhoods")
    if tract_weight_rows <= 0:
        issues.append("dashboard lookup validation has no tract lookup rows")
    if dashboard_rows != neighborhood_count:
        issues.append(
            f"dashboard lookup coarse rows {dashboard_rows} do not match neighborhoods {neighborhood_count}"
        )
    if risk_rows <= 0:
        issues.append("dashboard lookup validation has no risk-score comparison rows")

    return {
        "present": True,
        "neighborhood_basis": basis,
        "neighborhood_count": neighborhood_count,
        "tract_weight_rows": tract_weight_rows,
        "dashboard_coarse_rows": dashboard_rows,
        "dashboard_risk_score_rows": risk_rows,
        "dashboard_risk_score_vs_crimerisk_expected_count_total_spearman": dashboard.get(
            "dashboard_risk_score_vs_crimerisk_expected_count_total_spearman"
        ),
    }


def _check_external_surface_availability(*, issues: list[str]) -> dict[str, Any]:
    availability = _load_validation_or_repo_json(
        validation_name=f"external_surface_availability_{YEAR}.json",
        repo_path=REPO_EXTERNAL_AVAILABILITY,
    )
    if availability is None:
        issues.append("missing external-surface availability summary")
        return {"present": False}

    usable_count = int(availability.get("usable_external_surface_count") or 0)
    status = str(availability.get("status") or "")
    harness = str(availability.get("external_comparison_harness") or "")
    scoring_target = str(availability.get("harness_scoring_target") or "")
    public_source_notes = availability.get("public_source_notes", [])

    if status not in {"external_surface_unavailable", "external_surface_available"}:
        issues.append(f"external-surface availability has unexpected status {status!r}")
    if "benchmark_external_surface.py" not in harness:
        issues.append("external-surface availability does not name benchmark_external_surface.py")
    if "observed incident shares" not in scoring_target:
        issues.append("external-surface availability does not document observed-incident scoring target")
    if not isinstance(public_source_notes, list) or len(public_source_notes) < 3:
        issues.append("external-surface availability missing public source notes")

    return {
        "present": True,
        "status": status,
        "usable_external_surface_count": usable_count,
        "usable_external_surface_paths": availability.get("usable_external_surface_paths", []),
        "candidate_rows": availability.get("candidate_rows"),
        "reference_or_methodology_count": availability.get("reference_or_methodology_count"),
        "external_comparison_harness": harness,
    }


def _check_connecticut_population(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    bg_path = output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet"
    if not bg_path.exists() or not COUNTY_POP_2024_CSV.exists():
        issues.append(f"missing CT population inputs: {bg_path}, {COUNTY_POP_2024_CSV}")
        return {"present": False}
    bg = pd.read_parquet(bg_path, columns=["state_fips", f"population_{YEAR}"])
    observed = float(
        pd.to_numeric(bg.loc[bg["state_fips"].astype("string").str.zfill(2).eq("09"), f"population_{YEAR}"], errors="coerce")
        .fillna(0.0)
        .sum()
    )
    try:
        pop = pd.read_csv(COUNTY_POP_2024_CSV, dtype=str)
    except UnicodeDecodeError:
        pop = pd.read_csv(COUNTY_POP_2024_CSV, dtype=str, encoding="latin-1")
    target_rows = pop[
        pop["STATE"].astype("string").str.zfill(2).eq("09")
        & pop["SUMLEV"].astype("string").eq("050")
        & pop["COUNTY"].astype("string").ne("000")
    ].copy()
    target = float(pd.to_numeric(target_rows[f"POPESTIMATE{YEAR}"], errors="coerce").fillna(0.0).sum())
    delta = observed - target
    if abs(delta) > float(CT_POPULATION_TOLERANCE):
        issues.append(
            f"Connecticut published population does not match POPEST {YEAR} county controls "
            f"within {CT_POPULATION_TOLERANCE:g}: observed={observed:.0f}, target={target:.0f}, delta={delta:.0f}"
        )
    return {
        "present": True,
        f"published_population_{YEAR}": observed,
        f"popest_county_control_{YEAR}": target,
        "delta": delta,
        "tolerance": float(CT_POPULATION_TOLERANCE),
    }


def _check_acs_bg_vocabulary_coverage(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    """Fail closed on ACS vintage vocabulary drift.

    Every published BG GEOID absent from the raw ACS BG source (an external, uncontrolled input
    whose tract/BG vocabulary can drift between vintages) must be covered by either the
    Connecticut planning-region geometry relabel or a decennial-backfill entry
    (configs/acs_missing_bg_decennial_backfill.csv). Anything else means the ACS join silently
    dropped populated land, so it is an error.
    """
    bg_path = output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet"
    if not bg_path.exists() or not ACS_BG_SOURCE_PARQUET.exists():
        issues.append(
            f"missing inputs for ACS BG vocabulary coverage check: {bg_path}, {ACS_BG_SOURCE_PARQUET}"
        )
        return {"present": False}
    published = (
        pd.read_parquet(bg_path, columns=["block_group_geoid"])["block_group_geoid"]
        .astype("string")
        .str.zfill(12)
    )
    acs_ids = set(
        pd.read_parquet(ACS_BG_SOURCE_PARQUET, columns=["bg_id"])["bg_id"].astype("string").str.zfill(12)
    )
    absent = published[~published.isin(acs_ids)]
    by_state = absent.str.slice(0, 2).value_counts().sort_index()

    ct_relabel_targets: set[str] = set()
    if (absent.str.slice(0, 2) == "09").any() and CT_BG_2023_ZIP.exists() and CT_BG_2020_ZIP.exists():
        from crimerisk.covariates.features import build_ct_bg_2023_to_2020_map

        ct_relabel_targets = set(
            build_ct_bg_2023_to_2020_map(ct_bg_2023_zip=CT_BG_2023_ZIP, ct_bg_2020_zip=CT_BG_2020_ZIP).values()
        )

    backfill_ids: set[str] = set()
    if ACS_MISSING_BG_BACKFILL_CSV.exists():
        backfill_ids = set(
            pd.read_csv(ACS_MISSING_BG_BACKFILL_CSV, dtype={"bg_id": str})["bg_id"].astype(str).str.zfill(12)
        )

    covered_relabel = absent[absent.isin(ct_relabel_targets)]
    covered_backfill = absent[~absent.isin(ct_relabel_targets) & absent.isin(backfill_ids)]
    uncovered = absent[~absent.isin(ct_relabel_targets) & ~absent.isin(backfill_ids)]
    if len(uncovered) > 0:
        uncovered_by_state = uncovered.str.slice(0, 2).value_counts().sort_index()
        issues.append(
            "ACS BG vocabulary drift: published BG GEOIDs absent from the ACS BG source and covered by "
            "neither the CT relabel nor a decennial-backfill entry "
            f"(count={len(uncovered)}, by_state={uncovered_by_state.to_dict()}, "
            f"sample={sorted(uncovered)[:10]})"
        )
    return {
        "present": True,
        "published_bg_absent_from_acs": int(len(absent)),
        "absent_by_state": {str(k): int(v) for k, v in by_state.items()},
        "covered_by_ct_relabel": int(len(covered_relabel)),
        "covered_by_decennial_backfill": int(len(covered_backfill)),
        "uncovered": int(len(uncovered)),
    }


def _check_sparse_residual_transfer_policy(
    *,
    output_dir: Path,
    issues: list[str],
    burglary_tau_calibration: dict[str, Any],
) -> dict[str, Any]:
    audit_path = output_dir / f"allocation_component_denominator_audit_{YEAR}.parquet"
    if not audit_path.exists():
        issues.append(f"missing allocation component audit for sparse residual-transfer policy: {audit_path}")
        return {"present": False, "path": str(audit_path)}

    required_cols = [
        "state_fips",
        "jurisdiction_id",
        "offense",
        "model_share",
        "city_residual_transfer_policy",
        "city_residual_transfer_tau",
        "city_residual_predicted_log_ratio",
        "city_incident_posterior_active",
        "city_posterior_model_prior_raw",
        "city_posterior_model_prior_share",
    ]
    try:
        audit = pd.read_parquet(audit_path, columns=required_cols)
    except (KeyError, ValueError) as exc:
        issues.append(f"allocation component audit missing sparse residual-transfer policy columns: {exc}")
        return {"present": True, "path": str(audit_path), "required_columns_present": False}

    audit["state_fips"] = audit["state_fips"].astype("string").str.zfill(2)
    audit["offense"] = audit["offense"].astype("string")
    policy_rows = audit[audit["city_residual_transfer_policy"].notna()].copy()
    if policy_rows.empty:
        issues.append("allocation component audit has no residual-transfer policy rows")
        return {
            "present": True,
            "path": str(audit_path),
            "required_columns_present": True,
            "policy_row_count": 0,
        }

    group_cols = ["state_fips", "jurisdiction_id", "offense"]
    policy_rows["model_share"] = pd.to_numeric(policy_rows["model_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    policy_rows["city_posterior_model_prior_raw"] = (
        pd.to_numeric(policy_rows["city_posterior_model_prior_raw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    policy_rows["city_posterior_model_prior_share"] = (
        pd.to_numeric(policy_rows["city_posterior_model_prior_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    policy_rows["city_residual_transfer_tau"] = (
        pd.to_numeric(policy_rows["city_residual_transfer_tau"], errors="coerce").fillna(np.nan)
    )
    policy_rows["city_residual_predicted_log_ratio"] = (
        pd.to_numeric(policy_rows["city_residual_predicted_log_ratio"], errors="coerce").fillna(0.0)
    )
    baseline_total = policy_rows.groupby(group_cols, dropna=False)["model_share"].transform("sum")
    policy_rows["_baseline_share"] = np.where(
        baseline_total.gt(0.0),
        policy_rows["model_share"] / baseline_total.replace(0.0, np.nan),
        0.0,
    )
    policy_rows["_baseline_total"] = baseline_total
    active = policy_rows["city_incident_posterior_active"].eq(True)
    uncovered = ~active
    expected_tau = dict(DEFAULT_EXPECTED_RESIDUAL_TRANSFER_TAU_BY_OFFENSE)
    burglary_tau = burglary_tau_calibration.get("production_tau")
    if burglary_tau is not None and np.isfinite(float(burglary_tau)):
        expected_tau["burglary"] = float(burglary_tau)
    policy_rows["_expected_tau"] = np.where(
        active,
        1.0,
        policy_rows["offense"].astype(str).map(expected_tau).fillna(1.0).astype(float),
    )
    sparse_uncovered = uncovered & policy_rows["offense"].isin(SPARSE_BASELINE_TRANSFER_OFFENSES)
    sparse_uncovered_comparable = sparse_uncovered & policy_rows["_baseline_total"].gt(0.0)
    dense_uncovered = uncovered & policy_rows["offense"].isin(DENSE_FULL_RESIDUAL_TRANSFER_OFFENSES)
    burglary_uncovered = uncovered & policy_rows["offense"].eq("burglary")

    if not bool(sparse_uncovered.any()):
        issues.append("allocation component audit has no uncovered murder/rape rows for sparse transfer validation")
    sparse_rows = policy_rows.loc[sparse_uncovered]
    sparse_compare_rows = policy_rows.loc[sparse_uncovered_comparable]
    dense_rows = policy_rows.loc[dense_uncovered]
    burglary_rows = policy_rows.loc[burglary_uncovered]

    tau_delta = (policy_rows["city_residual_transfer_tau"] - policy_rows["_expected_tau"]).abs()
    max_tau_delta = float(tau_delta.max()) if len(tau_delta) else 0.0
    if max_tau_delta > TRANSFER_POLICY_TOLERANCE:
        examples = (
            policy_rows.assign(_tau_delta=tau_delta)
            .sort_values("_tau_delta", ascending=False, kind="mergesort")
            .head(5)[
                group_cols
                + [
                    "city_incident_posterior_active",
                    "city_residual_transfer_tau",
                    "_expected_tau",
                    "_tau_delta",
                ]
            ]
            .to_dict(orient="records")
        )
        issues.append(f"residual-transfer tau audit does not match expected policy (max {max_tau_delta:.3e}): {examples}")

    sparse_policy_bad = sparse_rows["city_residual_transfer_policy"].ne("baseline_sparse_offense")
    if bool(sparse_policy_bad.any()):
        examples = (
            sparse_rows.loc[sparse_policy_bad, group_cols + ["city_residual_transfer_policy"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        issues.append(f"uncovered murder/rape rows do not use baseline_sparse_offense policy: {examples}")

    sparse_raw_delta = (
        sparse_compare_rows["city_posterior_model_prior_raw"] - sparse_compare_rows["model_share"]
    ).abs()
    sparse_share_delta = (
        sparse_compare_rows["city_posterior_model_prior_share"] - sparse_compare_rows["_baseline_share"]
    ).abs()
    max_sparse_raw_delta = float(sparse_raw_delta.max()) if len(sparse_raw_delta) else 0.0
    max_sparse_share_delta = float(sparse_share_delta.max()) if len(sparse_share_delta) else 0.0
    if max_sparse_raw_delta > TRANSFER_POLICY_TOLERANCE or max_sparse_share_delta > TRANSFER_POLICY_TOLERANCE:
        example = (
            sparse_compare_rows.assign(_share_delta=sparse_share_delta, _raw_delta=sparse_raw_delta)
            .sort_values(["_share_delta", "_raw_delta"], ascending=False, kind="mergesort")
            .head(5)[
                [
                    "state_fips",
                    "jurisdiction_id",
                    "offense",
                    "model_share",
                    "city_posterior_model_prior_raw",
                    "city_posterior_model_prior_share",
                    "_baseline_share",
                    "_raw_delta",
                    "_share_delta",
                ]
            ]
            .to_dict(orient="records")
        )
        issues.append(
            "uncovered murder/rape model-prior share carries residual lift instead of normalized baseline "
            f"(max raw delta {max_sparse_raw_delta:.3e}, max share delta {max_sparse_share_delta:.3e}; "
            f"examples {example})"
        )

    dense_policy_bad = dense_rows["city_residual_transfer_policy"].ne("full_residual_tau1")
    if bool(dense_policy_bad.any()):
        examples = (
            dense_rows.loc[dense_policy_bad, group_cols + ["city_residual_transfer_policy"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        issues.append(f"uncovered dense-five rows do not retain full_residual_tau1 policy: {examples}")

    if bool(burglary_uncovered.any()):
        expected_burglary_tau = float(expected_tau["burglary"])
        if expected_burglary_tau <= 0.0:
            burglary_policy_good = burglary_rows["city_residual_transfer_policy"].eq("calibrated_baseline_tau0")
        elif expected_burglary_tau >= 1.0:
            burglary_policy_good = burglary_rows["city_residual_transfer_policy"].eq("full_residual_tau1")
        else:
            burglary_policy_good = burglary_rows["city_residual_transfer_policy"].eq(
                f"calibrated_residual_tau{expected_burglary_tau:.2f}"
            )
        if bool((~burglary_policy_good).any()):
            examples = (
                burglary_rows.loc[~burglary_policy_good, group_cols + ["city_residual_transfer_policy"]]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            issues.append(f"uncovered burglary rows do not use the calibrated tau policy: {examples}")
    else:
        issues.append("allocation component audit has no uncovered burglary rows for calibrated tau validation")

    comparable = policy_rows[policy_rows["_baseline_total"].gt(0.0)].copy()
    expected_raw = comparable["model_share"] * np.exp(
        np.clip(
            comparable["_expected_tau"] * comparable["city_residual_predicted_log_ratio"],
            -50.0,
            50.0,
        )
    )
    expected_total = expected_raw.groupby(
        [comparable[col] for col in group_cols],
        dropna=False,
    ).transform("sum")
    expected_share = np.where(
        expected_total.gt(0.0),
        expected_raw / expected_total.replace(0.0, np.nan),
        comparable["_baseline_share"],
    )
    raw_delta = (comparable["city_posterior_model_prior_raw"] - expected_raw).abs()
    share_delta = (
        comparable["city_posterior_model_prior_share"]
        - pd.Series(expected_share, index=comparable.index, dtype=float).fillna(0.0)
    ).abs()
    max_raw_delta = float(raw_delta.max()) if len(raw_delta) else 0.0
    max_share_delta = float(share_delta.max()) if len(share_delta) else 0.0
    if max_raw_delta > TRANSFER_POLICY_TOLERANCE or max_share_delta > TRANSFER_POLICY_TOLERANCE:
        examples = (
            comparable.assign(_raw_delta=raw_delta, _share_delta=share_delta)
            .sort_values(["_share_delta", "_raw_delta"], ascending=False, kind="mergesort")
            .head(5)[
                [
                    "state_fips",
                    "jurisdiction_id",
                    "offense",
                    "model_share",
                    "city_residual_transfer_tau",
                    "city_residual_predicted_log_ratio",
                    "city_posterior_model_prior_raw",
                    "city_posterior_model_prior_share",
                    "_raw_delta",
                    "_share_delta",
                ]
            ]
            .to_dict(orient="records")
        )
        issues.append(
            "residual-transfer prior raw/share values do not match model_share * exp(tau * predicted_log_ratio) "
            f"(max raw delta {max_raw_delta:.3e}, max share delta {max_share_delta:.3e}; examples {examples})"
        )

    return {
        "present": True,
        "path": str(audit_path),
        "required_columns_present": True,
        "policy_row_count": int(len(policy_rows)),
        "sparse_uncovered_row_count": int(sparse_uncovered.sum()),
        "sparse_uncovered_comparable_row_count": int(sparse_uncovered_comparable.sum()),
        "sparse_uncovered_zero_baseline_row_count": int((sparse_uncovered & policy_rows["_baseline_total"].le(0.0)).sum()),
        "sparse_uncovered_group_count": int(sparse_rows[group_cols].drop_duplicates().shape[0]) if not sparse_rows.empty else 0,
        "dense_uncovered_row_count": int(dense_uncovered.sum()),
        "dense_uncovered_group_count": int(dense_rows[group_cols].drop_duplicates().shape[0]) if not dense_rows.empty else 0,
        "burglary_uncovered_row_count": int(burglary_uncovered.sum()),
        "burglary_uncovered_group_count": int(burglary_rows[group_cols].drop_duplicates().shape[0]) if not burglary_rows.empty else 0,
        "expected_residual_transfer_tau_by_offense": {offense: float(value) for offense, value in expected_tau.items()},
        "max_residual_transfer_tau_delta": max_tau_delta,
        "max_model_prior_raw_delta_vs_tau_policy": max_raw_delta,
        "max_model_prior_share_delta_vs_tau_policy": max_share_delta,
        "max_sparse_model_prior_raw_delta_vs_baseline": max_sparse_raw_delta,
        "max_sparse_model_prior_share_delta_vs_normalized_baseline": max_sparse_share_delta,
        "sparse_policy_bad_row_count": int(sparse_policy_bad.sum()) if len(sparse_rows) else 0,
        "dense_policy_bad_row_count": int(dense_policy_bad.sum()) if len(dense_rows) else 0,
    }


def _load_controls_for_total_lane(
    *,
    issues: list[str],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    controls_path = REPO_ROOT / "state" / "controls" / f"jurisdiction_controls_{YEAR}.parquet"
    required_cols = [
        "jurisdiction_id",
        "jurisdiction_type",
        "jurisdiction_name",
        "state_fips",
        "state_abbr",
        "geo_type",
        "geoid",
        "offense",
        "preferred_source",
        "preferred_source_lane",
        "preferred_source_family",
        "quality_tier_preferred",
        "reported_count_preferred",
        "relationship_type_preferred",
        "overlap_subtype_preferred",
        TOTAL_LANE_TARGET_COLUMN,
        "bucket_population",
        "dominant_reporting_regime",
        "dominant_preferred_source_by_regime",
        "published_nibrs_corroborated_count",
        "cius_municipal_official_count",
        *[
            value
            for source_cols in SOURCE_TO_CONTROL_COLUMNS.values()
            for value in source_cols.values()
        ],
    ]
    if not controls_path.exists():
        issues.append(f"total_lane.no_duplicate_control_totals: missing controls file {controls_path}")
        return None, {"path": str(controls_path), "present": False}
    try:
        controls = pd.read_parquet(controls_path, columns=required_cols)
    except (KeyError, ValueError) as exc:
        issues.append(f"total_lane: controls missing required columns: {exc}")
        return None, {"path": str(controls_path), "present": True, "required_columns_present": False}
    controls["state_fips"] = controls["state_fips"].astype("string").str.zfill(2)
    controls["jurisdiction_id"] = controls["jurisdiction_id"].astype("string")
    controls["jurisdiction_type"] = controls["jurisdiction_type"].astype("string")
    controls["offense"] = controls["offense"].astype("string")
    return controls, {
        "path": str(controls_path),
        "present": True,
        "required_columns_present": True,
        "rows": int(len(controls)),
        "jurisdiction_count": int(controls["jurisdiction_id"].nunique()),
    }


def _infer_crosswalk_jurisdiction_type(jurisdiction_id: pd.Series) -> pd.Series:
    text = jurisdiction_id.astype("string")
    return pd.Series(
        np.select(
            [
                text.str.endswith(":state_nonmunicipal_remainder", na=False),
                text.str.endswith(":statewide_overlap_layer", na=False),
                text.str.contains(":municipal:", na=False),
            ],
            ["state_nonmunicipal_remainder", "statewide_overlap_layer", "municipal"],
            default="other",
        ),
        index=jurisdiction_id.index,
        dtype="string",
    )


def _check_no_duplicate_control_totals(
    *,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    key_cols = ["jurisdiction_id", "offense"]
    duplicate_key_rows = controls[controls.duplicated(key_cols, keep=False)].copy()
    if not duplicate_key_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.no_duplicate_control_totals: duplicate (jurisdiction_id, offense) rows",
            duplicate_key_rows.sort_values(key_cols, kind="mergesort"),
            columns=[*key_cols, "jurisdiction_type", TOTAL_LANE_TARGET_COLUMN],
        )

    municipal_geo = controls[
        controls["jurisdiction_type"].eq("municipal")
        & controls["geo_type"].notna()
        & controls["geoid"].notna()
    ].copy()
    geo_cols = ["state_fips", "geo_type", "geoid", "offense"]
    duplicate_geo_rows = municipal_geo[municipal_geo.duplicated(geo_cols, keep=False)].copy()
    if not duplicate_geo_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.no_duplicate_control_totals: duplicate municipal geography/offense rows",
            duplicate_geo_rows.sort_values(geo_cols, kind="mergesort"),
            columns=[*geo_cols, "jurisdiction_id", TOTAL_LANE_TARGET_COLUMN],
        )

    crosswalk_path = REPO_ROOT / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    crosswalk_summary: dict[str, Any] = {"path": str(crosswalk_path), "present": crosswalk_path.exists()}
    multi_assignment_rows = pd.DataFrame()
    municipal_layer_overlap_rows = pd.DataFrame()
    if not crosswalk_path.exists():
        issues.append(f"total_lane.no_duplicate_control_totals: missing agency crosswalk {crosswalk_path}")
    else:
        crosswalk = pd.read_parquet(
            crosswalk_path,
            columns=["ori", "state_fips", "jurisdiction_id", "relationship_type", "weight"],
        )
        crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
        crosswalk["jurisdiction_id"] = crosswalk["jurisdiction_id"].astype("string")
        crosswalk["jurisdiction_type"] = _infer_crosswalk_jurisdiction_type(crosswalk["jurisdiction_id"])
        crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0)
        duplicate_crosswalk_rows = crosswalk[crosswalk.duplicated(["ori", "jurisdiction_id"], keep=False)].copy()
        if not duplicate_crosswalk_rows.empty:
            _append_total_lane_issue(
                issues,
                "total_lane.no_duplicate_control_totals: duplicate agency-to-jurisdiction crosswalk rows",
                duplicate_crosswalk_rows.sort_values(["ori", "jurisdiction_id"], kind="mergesort"),
                columns=["ori", "state_fips", "jurisdiction_id", "relationship_type", "weight"],
            )
        agency_type_sets = (
            crosswalk.groupby(["state_fips", "ori"], dropna=False)["jurisdiction_type"]
            .agg(lambda s: sorted(set(s.dropna().astype(str))))
            .reset_index()
        )
        agency_type_sets["jurisdiction_type_count"] = agency_type_sets["jurisdiction_type"].map(len)
        multi_assignment_rows = agency_type_sets[agency_type_sets["jurisdiction_type_count"].gt(1)].copy()
        municipal_layer_overlap_rows = agency_type_sets[
            agency_type_sets["jurisdiction_type"].map(
                lambda values: "municipal" in values
                and any(value in values for value in ("state_nonmunicipal_remainder", "statewide_overlap_layer"))
            )
        ].copy()
        if not municipal_layer_overlap_rows.empty:
            _append_total_lane_issue(
                issues,
                "total_lane.no_duplicate_control_totals: agency assigned to both municipal and synthetic state layer",
                municipal_layer_overlap_rows.sort_values(["state_fips", "ori"], kind="mergesort"),
                columns=["state_fips", "ori", "jurisdiction_type"],
            )
        crosswalk_summary.update(
            {
                "rows": int(len(crosswalk)),
                "duplicate_ori_jurisdiction_rows": int(len(duplicate_crosswalk_rows)),
                "multi_assignment_agency_count": int(len(multi_assignment_rows)),
                "municipal_layer_overlap_agency_count": int(len(municipal_layer_overlap_rows)),
            }
        )

    nonmunicipal_bad_relationship = controls[
        controls["jurisdiction_type"].eq("state_nonmunicipal_remainder")
        & controls["relationship_type_preferred"].notna()
        & controls["relationship_type_preferred"].astype("string").eq("overlap")
    ].copy()
    overlap_bad_relationship = controls[
        controls["jurisdiction_type"].eq("statewide_overlap_layer")
        & controls["relationship_type_preferred"].notna()
        & controls["relationship_type_preferred"].astype("string").ne("overlap")
    ].copy()
    if not nonmunicipal_bad_relationship.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.no_duplicate_control_totals: nonmunicipal remainder has overlap preferred relationship",
            nonmunicipal_bad_relationship,
            columns=["state_fips", "jurisdiction_id", "offense", "relationship_type_preferred"],
        )
    if not overlap_bad_relationship.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.no_duplicate_control_totals: overlap layer has non-overlap preferred relationship",
            overlap_bad_relationship,
            columns=["state_fips", "jurisdiction_id", "offense", "relationship_type_preferred"],
        )

    bad_count = (
        len(duplicate_key_rows)
        + len(duplicate_geo_rows)
        + int(crosswalk_summary.get("duplicate_ori_jurisdiction_rows", 0))
        + len(municipal_layer_overlap_rows)
        + len(nonmunicipal_bad_relationship)
        + len(overlap_bad_relationship)
    )
    return {
        "ok": bad_count == 0,
        "duplicate_jurisdiction_offense_rows": int(len(duplicate_key_rows)),
        "duplicate_municipal_geography_offense_rows": int(len(duplicate_geo_rows)),
        "crosswalk": crosswalk_summary,
        "nonmunicipal_bad_relationship_rows": int(len(nonmunicipal_bad_relationship)),
        "overlap_bad_relationship_rows": int(len(overlap_bad_relationship)),
        "offending_rows_sample": {
            "duplicate_jurisdiction_offense": _sample_records(
                duplicate_key_rows,
                columns=[*key_cols, "jurisdiction_type", TOTAL_LANE_TARGET_COLUMN],
            ),
            "duplicate_municipal_geography": _sample_records(
                duplicate_geo_rows,
                columns=[*geo_cols, "jurisdiction_id", TOTAL_LANE_TARGET_COLUMN],
            ),
            "municipal_layer_overlap_agencies": _sample_records(
                municipal_layer_overlap_rows,
                columns=["state_fips", "ori", "jurisdiction_type"],
            ),
        },
    }


def _globally_dead_observation_oris(agency_obs: pd.DataFrame) -> set[str]:
    if agency_obs.empty:
        return set()
    stats = agency_obs[["ori9", "count", "months_reported"]].copy()
    stats["count"] = pd.to_numeric(stats["count"], errors="coerce").fillna(0.0)
    stats["months_reported"] = pd.to_numeric(stats["months_reported"], errors="coerce").fillna(0.0)
    stats = (
        stats.groupby("ori9", dropna=False)
        .agg(total_count=("count", "sum"), max_months=("months_reported", "max"))
        .reset_index()
    )
    return set(
        stats[
            stats["ori9"].notna()
            & stats["total_count"].le(0.0)
            & stats["max_months"].le(0.0)
        ]["ori9"].astype(str)
    )


def _source_slice(
    agency_obs: pd.DataFrame,
    *,
    source: str,
    prefix: str,
) -> pd.DataFrame:
    key_cols = ["ori9", "state_fips", "state_abbr", "offense"]
    cols = [
        *key_cols,
        "count",
        "observation_weight",
        "months_reported",
        "conversion_status",
        "cius_reference_flag",
    ]
    frame = agency_obs[agency_obs["source"].eq(source)][cols].copy()
    return frame.rename(
        columns={
            "count": f"reported_count_{prefix}",
            "observation_weight": f"observation_weight_{prefix}",
            "months_reported": f"mean_months_reported_{prefix}",
            "conversion_status": f"conversion_status_{prefix}",
            "cius_reference_flag": f"cius_reference_flag_{prefix}",
        }
    )


def _build_readonly_agency_preferred_panel(*, issues: list[str]) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    obs_path = REPO_ROOT / "state" / "observations" / "agency_year_observations.parquet"
    regimes_path = REPO_ROOT / "state" / "modeling" / "agency_year_reporting_regimes.parquet"
    if not obs_path.exists() or not regimes_path.exists():
        issues.append(
            "total_lane.source_priority_honored: missing agency observations or reporting regimes "
            f"({obs_path}, {regimes_path})"
        )
        return None, {"present": False, "observations_path": str(obs_path), "regimes_path": str(regimes_path)}

    obs_cols = [
        "ori9",
        "state_fips",
        "state_abbr",
        "offense",
        "year",
        "source",
        "count",
        "observation_weight",
        "months_reported",
        "conversion_status",
        "cius_reference_flag",
    ]
    all_obs = pd.read_parquet(obs_path, columns=obs_cols)
    dead_oris = _globally_dead_observation_oris(all_obs)
    agency_obs = all_obs[
        all_obs["year"].astype(int).eq(YEAR)
        & all_obs["source"].isin(SOURCE_PRIORITY)
    ].copy()
    if dead_oris:
        agency_obs = agency_obs[~agency_obs["ori9"].astype("string").isin(sorted(dead_oris))].copy()
    agency_obs["state_fips"] = agency_obs["state_fips"].astype("string").str.zfill(2)
    agency_obs["state_abbr"] = agency_obs["state_abbr"].astype("string").str.upper()

    key_cols = ["ori9", "state_fips", "state_abbr", "offense"]
    slices = [
        _source_slice(agency_obs, source=CIUS_SOURCE, prefix="cius"),
        _source_slice(agency_obs, source=LOCAL_PUBLICATION_SOURCE, prefix="local_publication"),
        _source_slice(agency_obs, source=STATE_PUBLICATION_SOURCE, prefix="state_publication"),
        _source_slice(agency_obs, source=SUMMARY_SOURCE, prefix="srs"),
        _source_slice(agency_obs, source=NIBRS_SOURCE, prefix="nibrs"),
    ]
    if not any(not frame.empty for frame in slices):
        issues.append(f"total_lane.source_priority_honored: no {YEAR} agency observations for source-priority validation")
        return None, {
            "present": True,
            "observations_path": str(obs_path),
            "regimes_path": str(regimes_path),
            "agency_observation_rows": 0,
        }
    panel = slices[0]
    for frame in slices[1:]:
        panel = panel.merge(frame, on=key_cols, how="outer")

    regime_cols = [
        "ori9",
        "year",
        "offense",
        "reporting_regime",
        "preferred_source_by_regime",
        "srs_months_reported",
        "nibrs_months_reported",
        "srs_observation_weight",
        "nibrs_observation_weight",
        "source_override_applied",
    ]
    regimes = pd.read_parquet(regimes_path, columns=regime_cols)
    regimes = regimes[regimes["year"].astype(int).eq(YEAR)].copy()
    panel = panel.merge(
        regimes.drop(columns="year"),
        on=["ori9", "offense"],
        how="left",
    )
    panel = panel.merge(
        load_published_nibrs_reference_counts(RepoPaths.from_repo_root(REPO_ROOT), year=YEAR),
        on=["ori9", "state_abbr", "offense"],
        how="left",
    )

    has_cius = panel["reported_count_cius"].notna()
    has_local_publication = panel["reported_count_local_publication"].notna()
    has_state_publication = panel["reported_count_state_publication"].notna()
    has_srs = panel["reported_count_srs"].notna()
    has_nibrs = panel["reported_count_nibrs"].notna()
    srs_obs_weight = pd.to_numeric(panel["srs_observation_weight"], errors="coerce").fillna(
        pd.to_numeric(panel["observation_weight_srs"], errors="coerce").fillna(0.0)
    )
    nibrs_obs_weight = pd.to_numeric(panel["nibrs_observation_weight"], errors="coerce").fillna(
        pd.to_numeric(panel["observation_weight_nibrs"], errors="coerce").fillna(0.0)
    )
    srs_months = pd.to_numeric(panel["srs_months_reported"], errors="coerce").fillna(
        pd.to_numeric(panel["mean_months_reported_srs"], errors="coerce").fillna(0.0)
    )
    nibrs_months = pd.to_numeric(panel["nibrs_months_reported"], errors="coerce").fillna(
        pd.to_numeric(panel["mean_months_reported_nibrs"], errors="coerce").fillna(0.0)
    )
    published_nibrs_supports_nibrs = (
        has_nibrs
        & build_published_nibrs_corroboration_mask(
            nibrs_count=panel["reported_count_nibrs"],
            published_nibrs_count=panel.get("published_nibrs_official_count"),
            nibrs_months=nibrs_months,
            srs_count=panel["reported_count_srs"],
        )
    )
    prefer_nibrs = build_prefer_nibrs_mask(
        has_cius=has_cius,
        has_local_publication=has_local_publication,
        has_state_publication=has_state_publication,
        has_srs=has_srs,
        has_nibrs=has_nibrs,
        regime_prefers_nibrs=panel["preferred_source_by_regime"].eq(NIBRS_SOURCE),
        srs_regime_inferior=panel["reporting_regime"].isin(
            ["structurally_missing_or_unreliable", "lumpy_or_batched", "annual_only_but_usable"]
        ),
        nibrs_supports_better=nibrs_obs_weight.gt(srs_obs_weight)
        | (nibrs_obs_weight.eq(srs_obs_weight) & nibrs_months.gt(srs_months)),
        srs_count_num=pd.to_numeric(panel["reported_count_srs"], errors="coerce").fillna(0.0),
        nibrs_months=nibrs_months,
        manual_source_override=panel["source_override_applied"].astype("boolean").fillna(False).astype(bool),
        published_nibrs_supports_nibrs=published_nibrs_supports_nibrs,
    )
    panel["expected_preferred_source"] = initialize_preferred_source(
        has_cius=has_cius,
        has_local_publication=has_local_publication,
        has_state_publication=has_state_publication,
        has_srs=has_srs,
        has_nibrs=has_nibrs,
        prefer_nibrs_mask=prefer_nibrs,
    )
    panel["expected_preferred_count"] = pd.Series(np.nan, index=panel.index, dtype=float)
    for source, cols in SOURCE_TO_CONTROL_COLUMNS.items():
        prefix_count_col = cols["count"]
        mask = panel["expected_preferred_source"].eq(source)
        if prefix_count_col in panel.columns:
            panel.loc[mask, "expected_preferred_count"] = pd.to_numeric(
                panel.loc[mask, prefix_count_col], errors="coerce"
            )
    return panel, {
        "present": True,
        "observations_path": str(obs_path),
        "regimes_path": str(regimes_path),
        "agency_observation_rows": int(len(agency_obs)),
        "agency_preferred_rows": int(len(panel)),
        "dead_ori_count": int(len(dead_oris)),
        "expected_preferred_source_counts": {
            str(k): int(v) for k, v in panel["expected_preferred_source"].value_counts(dropna=False).to_dict().items()
        },
    }


def _dominant_label_by_weight(
    merged: pd.DataFrame,
    *,
    key_cols: list[str],
    label_col: str,
    weight_col: str,
    output_col: str,
) -> pd.DataFrame:
    labels = merged[label_col].astype("string")
    weights = pd.to_numeric(merged[weight_col], errors="coerce").fillna(0.0)
    mask = labels.notna() & labels.ne("")
    if not mask.any():
        return pd.DataFrame(columns=[*key_cols, output_col])
    support = (
        merged.loc[mask, key_cols]
        .assign(_label=labels.loc[mask].values, _weight=weights.loc[mask].values)
        .groupby([*key_cols, "_label"], dropna=False, as_index=False)["_weight"]
        .sum()
        .sort_values(
            [*key_cols, "_weight", "_label"],
            ascending=[True] * len(key_cols) + [False, True],
            kind="mergesort",
        )
    )
    support["_rank"] = support.groupby(key_cols, dropna=False).cumcount()
    return support[support["_rank"].eq(0)][[*key_cols, "_label"]].rename(columns={"_label": output_col})


def _aggregate_readonly_agency_preferred_to_jurisdiction(
    *,
    agency_preferred: pd.DataFrame,
    issues: list[str],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    crosswalk_path = REPO_ROOT / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    if not crosswalk_path.exists():
        issues.append(f"total_lane.source_priority_honored: missing agency crosswalk {crosswalk_path}")
        return None, {"present": False, "path": str(crosswalk_path)}
    crosswalk = pd.read_parquet(crosswalk_path, columns=["ori", "jurisdiction_id", "weight"]).rename(columns={"ori": "ori9"})
    merged = agency_preferred.merge(crosswalk, on="ori9", how="inner")
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged["expected_preferred_count"] = pd.to_numeric(
        merged["expected_preferred_count"], errors="coerce"
    ).fillna(0.0)
    merged["allocated_count"] = merged["expected_preferred_count"] * merged["weight"]
    merged["support_weight"] = merged["allocated_count"].abs().where(
        merged["allocated_count"].abs().gt(0.0),
        merged["weight"],
    )
    key_cols = ["jurisdiction_id", "offense"]
    grouped = (
        merged.groupby(key_cols, dropna=False, as_index=False)
        .agg(recomputed_reported_count=("allocated_count", "sum"))
    )
    source = _dominant_label_by_weight(
        merged,
        key_cols=key_cols,
        label_col="expected_preferred_source",
        weight_col="support_weight",
        output_col="recomputed_preferred_source",
    )
    return grouped.merge(source, on=key_cols, how="left"), {
        "present": True,
        "path": str(crosswalk_path),
        "merged_agency_rows": int(len(merged)),
        "jurisdiction_offense_rows": int(len(grouped)),
    }


def _control_prefer_nibrs_mask(controls: pd.DataFrame) -> pd.Series:
    has_cius = controls["reported_count_cius"].notna()
    has_local = controls["reported_count_local_publication"].notna()
    has_state = controls["reported_count_state_publication"].notna()
    has_srs = controls["reported_count_srs"].notna()
    has_nibrs = controls["reported_count_nibrs"].notna()
    srs_obs_weight = pd.to_numeric(controls["observation_weight_srs"], errors="coerce").fillna(0.0)
    nibrs_obs_weight = pd.to_numeric(controls["observation_weight_nibrs"], errors="coerce").fillna(0.0)
    srs_months = pd.to_numeric(controls["mean_months_reported_srs"], errors="coerce").fillna(0.0)
    nibrs_months = pd.to_numeric(controls["mean_months_reported_nibrs"], errors="coerce").fillna(0.0)
    published_nibrs_supports_nibrs = (
        has_nibrs
        & build_published_nibrs_corroboration_mask(
            nibrs_count=controls["reported_count_nibrs"],
            published_nibrs_count=controls["published_nibrs_corroborated_count"],
            srs_count=controls["reported_count_srs"],
            veto_mask=pd.to_numeric(controls["cius_municipal_official_count"], errors="coerce").notna(),
        )
    )
    return build_prefer_nibrs_mask(
        has_cius=has_cius,
        has_local_publication=has_local,
        has_state_publication=has_state,
        has_srs=has_srs,
        has_nibrs=has_nibrs,
        regime_prefers_nibrs=controls["dominant_preferred_source_by_regime"].eq(NIBRS_SOURCE),
        srs_regime_inferior=controls["dominant_reporting_regime"].isin(
            ["structurally_missing_or_unreliable", "lumpy_or_batched", "annual_only_but_usable"]
        ),
        nibrs_supports_better=nibrs_obs_weight.gt(srs_obs_weight)
        | (nibrs_obs_weight.eq(srs_obs_weight) & nibrs_months.gt(srs_months)),
        srs_count_num=pd.to_numeric(controls["reported_count_srs"], errors="coerce").fillna(0.0),
        nibrs_months=nibrs_months,
        published_nibrs_supports_nibrs=published_nibrs_supports_nibrs,
    )


def _check_source_priority_honored(
    *,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    source_rank = {source: idx for idx, source in enumerate(SOURCE_PRIORITY)}
    preferred_source = controls["preferred_source"].astype("string")
    unknown_source_rows = controls[~preferred_source.isin(SOURCE_PRIORITY)].copy()
    if not unknown_source_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.source_priority_honored: preferred_source outside declared source priority",
            unknown_source_rows,
            columns=["jurisdiction_id", "offense", "preferred_source"],
        )

    lane_expected = preferred_source.map(source_lane_from_source).astype("string")
    lane_bad = controls[controls["preferred_source_lane"].astype("string").ne(lane_expected)].copy()
    if not lane_bad.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.source_priority_honored: preferred_source_lane does not match preferred_source",
            lane_bad,
            columns=["jurisdiction_id", "offense", "preferred_source", "preferred_source_lane"],
        )

    null_preferred_rows = controls[
        controls["reported_count_preferred"].isna()
        | controls["quality_tier_preferred"].isna()
        | controls[TOTAL_LANE_TARGET_COLUMN].isna()
    ].copy()
    if not null_preferred_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.source_priority_honored: preferred source/count/quality/control target has nulls",
            null_preferred_rows,
            columns=[
                "jurisdiction_id",
                "offense",
                "preferred_source",
                "reported_count_preferred",
                "quality_tier_preferred",
                TOTAL_LANE_TARGET_COLUMN,
            ],
        )

    preferred_rank = preferred_source.map(source_rank)
    prefer_nibrs = _control_prefer_nibrs_mask(controls)
    municipal = controls["jurisdiction_type"].eq("municipal")
    precedence_bad = pd.Series(False, index=controls.index)
    for source in SOURCE_PRIORITY:
        source_cols = SOURCE_TO_CONTROL_COLUMNS[source]
        higher_source_available = (
            controls[source_cols["count"]].notna()
            & controls[source_cols["relationship"]].astype("string").eq("exclusive")
        )
        skipped_higher_source = higher_source_available & municipal & (source_rank[source] < preferred_rank)
        allowed_srs_to_nibrs_exception = (
            preferred_source.eq(NIBRS_SOURCE)
            & prefer_nibrs
            & (source == SUMMARY_SOURCE)
        )
        precedence_bad |= skipped_higher_source & ~allowed_srs_to_nibrs_exception
    municipal_precedence_bad_rows = controls[precedence_bad].copy()
    if not municipal_precedence_bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.source_priority_honored: municipal preferred_source skipped a higher-priority exclusive source",
            municipal_precedence_bad_rows,
            columns=[
                "jurisdiction_id",
                "offense",
                "preferred_source",
                "reported_count_cius",
                "reported_count_local_publication",
                "reported_count_state_publication",
                "reported_count_srs",
                "reported_count_nibrs",
            ],
        )

    agency_panel, agency_summary = _build_readonly_agency_preferred_panel(issues=issues)
    aggregate_summary: dict[str, Any] = {"present": False}
    synthetic_count_bad_rows = pd.DataFrame()
    synthetic_source_bad_rows = pd.DataFrame()
    synthetic_recomputed_rows = 0
    synthetic_missing_recomputed_rows = 0
    max_synthetic_count_delta = 0.0
    if agency_panel is not None:
        aggregate, aggregate_summary = _aggregate_readonly_agency_preferred_to_jurisdiction(
            agency_preferred=agency_panel,
            issues=issues,
        )
        if aggregate is not None:
            synthetic = controls[controls["jurisdiction_type"].isin(["state_nonmunicipal_remainder", "statewide_overlap_layer"])].copy()
            merged = synthetic.merge(aggregate, on=["jurisdiction_id", "offense"], how="left")
            has_recomputed = merged["recomputed_reported_count"].notna()
            synthetic_recomputed_rows = int(has_recomputed.sum())
            synthetic_missing_recomputed_rows = int((~has_recomputed).sum())
            if has_recomputed.any():
                count_delta = (
                    pd.to_numeric(merged["reported_count_preferred"], errors="coerce").fillna(0.0)
                    - pd.to_numeric(merged["recomputed_reported_count"], errors="coerce").fillna(0.0)
                )
                max_synthetic_count_delta = _max_abs(count_delta.loc[has_recomputed])
                synthetic_count_bad_rows = merged.loc[has_recomputed & count_delta.abs().gt(TOTAL_LANE_TOLERANCE)].copy()
                source_bad = (
                    merged.loc[has_recomputed, "preferred_source"].astype("string").fillna("")
                    .ne(merged.loc[has_recomputed, "recomputed_preferred_source"].astype("string").fillna(""))
                )
                synthetic_source_bad_rows = merged.loc[has_recomputed].loc[source_bad].copy()
            if not synthetic_count_bad_rows.empty:
                _append_total_lane_issue(
                    issues,
                    "total_lane.source_priority_honored: synthetic layer reported_count_preferred does not equal agency-level preferred-source rollup",
                    synthetic_count_bad_rows,
                    columns=[
                        "state_fips",
                        "jurisdiction_id",
                        "offense",
                        "reported_count_preferred",
                        "recomputed_reported_count",
                        "preferred_source",
                        "recomputed_preferred_source",
                    ],
                )
            if not synthetic_source_bad_rows.empty:
                _append_total_lane_issue(
                    issues,
                    "total_lane.source_priority_honored: synthetic layer preferred_source does not match agency-level dominant preferred source",
                    synthetic_source_bad_rows,
                    columns=[
                        "state_fips",
                        "jurisdiction_id",
                        "offense",
                        "preferred_source",
                        "recomputed_preferred_source",
                        "reported_count_preferred",
                    ],
                )

    estimates_path = REPO_ROOT / "state" / "controls" / "jurisdiction_year_estimates.parquet"
    estimate_alignment_summary: dict[str, Any] = {"path": str(estimates_path), "present": estimates_path.exists()}
    estimate_count_bad_rows = pd.DataFrame()
    estimate_target_bad_rows = pd.DataFrame()
    estimate_source_bad_rows = pd.DataFrame()
    if estimates_path.exists():
        estimates = pd.read_parquet(
            estimates_path,
            columns=[
                "jurisdiction_id",
                "offense",
                "year",
                "preferred_source",
                "reported_count_preferred",
                "estimated_count",
            ],
        )
        estimates = estimates[estimates["year"].astype(int).eq(YEAR)].rename(
            columns={
                "preferred_source": "estimate_preferred_source",
                "reported_count_preferred": "estimate_reported_count_preferred",
                "estimated_count": "estimate_target_count",
            }
        )
        aligned = controls.merge(
            estimates[
                [
                    "jurisdiction_id",
                    "offense",
                    "estimate_preferred_source",
                    "estimate_reported_count_preferred",
                    "estimate_target_count",
                ]
            ],
            on=["jurisdiction_id", "offense"],
            how="left",
        )
        has_estimate = aligned["estimate_reported_count_preferred"].notna()
        count_delta = (
            pd.to_numeric(aligned["reported_count_preferred"], errors="coerce").fillna(0.0)
            - pd.to_numeric(aligned["estimate_reported_count_preferred"], errors="coerce").fillna(0.0)
        )
        target_delta = (
            pd.to_numeric(aligned[TOTAL_LANE_TARGET_COLUMN], errors="coerce").fillna(0.0)
            - pd.to_numeric(aligned["estimate_target_count"], errors="coerce").fillna(0.0)
        )
        source_bad = (
            aligned["preferred_source"].astype("string").fillna("")
            .ne(aligned["estimate_preferred_source"].astype("string").fillna(""))
        )
        estimate_count_bad_rows = aligned[has_estimate & count_delta.abs().gt(TOTAL_LANE_TOLERANCE)].copy()
        estimate_target_bad_rows = aligned[has_estimate & target_delta.abs().gt(TOTAL_LANE_TOLERANCE)].copy()
        estimate_source_bad_rows = aligned[has_estimate & source_bad].copy()
        for label, rows in [
            ("reported_count_preferred", estimate_count_bad_rows),
            (TOTAL_LANE_TARGET_COLUMN, estimate_target_bad_rows),
            ("preferred_source", estimate_source_bad_rows),
        ]:
            if not rows.empty:
                _append_total_lane_issue(
                    issues,
                    f"total_lane.source_priority_honored: controls {label} drifted from target-year jurisdiction estimate panel",
                    rows,
                    columns=[
                        "jurisdiction_id",
                        "offense",
                        "preferred_source",
                        "estimate_preferred_source",
                        "reported_count_preferred",
                        "estimate_reported_count_preferred",
                        TOTAL_LANE_TARGET_COLUMN,
                        "estimate_target_count",
                    ],
                )
        estimate_alignment_summary.update(
            {
                "target_year_estimate_rows": int(len(estimates)),
                "controls_with_target_year_estimate": int(has_estimate.sum()),
                "controls_missing_target_year_estimate": int((~has_estimate).sum()),
                "reported_count_bad_rows": int(len(estimate_count_bad_rows)),
                "target_count_bad_rows": int(len(estimate_target_bad_rows)),
                "preferred_source_bad_rows": int(len(estimate_source_bad_rows)),
            }
        )
    else:
        issues.append(f"total_lane.source_priority_honored: missing jurisdiction-year estimates {estimates_path}")

    bad_count = (
        len(unknown_source_rows)
        + len(lane_bad)
        + len(null_preferred_rows)
        + len(municipal_precedence_bad_rows)
        + len(synthetic_count_bad_rows)
        + len(synthetic_source_bad_rows)
        + len(estimate_count_bad_rows)
        + len(estimate_target_bad_rows)
        + len(estimate_source_bad_rows)
    )
    return {
        "ok": bad_count == 0,
        "source_priority": list(SOURCE_PRIORITY),
        "unknown_preferred_source_rows": int(len(unknown_source_rows)),
        "lane_mismatch_rows": int(len(lane_bad)),
        "null_preferred_source_value_rows": int(len(null_preferred_rows)),
        "municipal_precedence_violation_rows": int(len(municipal_precedence_bad_rows)),
        "synthetic_rows_with_agency_recomputed_preference": synthetic_recomputed_rows,
        "synthetic_rows_missing_agency_recomputed_preference": synthetic_missing_recomputed_rows,
        "synthetic_agency_rollup_reported_count_bad_rows": int(len(synthetic_count_bad_rows)),
        "synthetic_agency_rollup_preferred_source_bad_rows": int(len(synthetic_source_bad_rows)),
        "max_synthetic_agency_rollup_reported_count_delta": max_synthetic_count_delta,
        "agency_preferred_panel": agency_summary,
        "agency_to_jurisdiction_preferred_rollup": aggregate_summary,
        "jurisdiction_year_estimate_alignment": estimate_alignment_summary,
        "offending_rows_sample": {
            "municipal_precedence": _sample_records(
                municipal_precedence_bad_rows,
                columns=[
                    "jurisdiction_id",
                    "offense",
                    "preferred_source",
                    "reported_count_cius",
                    "reported_count_local_publication",
                    "reported_count_state_publication",
                    "reported_count_srs",
                    "reported_count_nibrs",
                ],
            ),
            "synthetic_count": _sample_records(
                synthetic_count_bad_rows,
                columns=[
                    "state_fips",
                    "jurisdiction_id",
                    "offense",
                    "reported_count_preferred",
                    "recomputed_reported_count",
                ],
            ),
            "estimate_alignment": _sample_records(
                pd.concat(
                    [estimate_count_bad_rows, estimate_target_bad_rows, estimate_source_bad_rows],
                    ignore_index=True,
                ).drop_duplicates(subset=["jurisdiction_id", "offense"], keep="first"),
                columns=[
                    "jurisdiction_id",
                    "offense",
                    "preferred_source",
                    "estimate_preferred_source",
                    "reported_count_preferred",
                    "estimate_reported_count_preferred",
                    TOTAL_LANE_TARGET_COLUMN,
                    "estimate_target_count",
                ],
            ),
        },
    }


def _check_state_remainder_reconciliation(
    *,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    state_controls_path = REPO_ROOT / "state" / "controls" / "state_control_comparison.parquet"
    if not state_controls_path.exists():
        issues.append(f"total_lane.state_remainder_reconciliation: missing state controls {state_controls_path}")
        return {"ok": False, "path": str(state_controls_path), "present": False}

    state_controls = pd.read_parquet(state_controls_path)
    state_controls["state_fips"] = state_controls["state_fips"].astype("string").str.zfill(2)
    state_controls["offense"] = state_controls["offense"].astype("string")
    state_controls = state_controls[state_controls["year"].astype(int).eq(YEAR)].copy()

    part_totals = (
        controls.groupby(["state_fips", "state_abbr", "offense", "jurisdiction_type"], dropna=False)
        .agg(
            reported_total=("reported_count_preferred", "sum"),
            adjusted_total=(TOTAL_LANE_TARGET_COLUMN, "sum"),
        )
        .reset_index()
    )

    def _part(prefix: str, jurisdiction_type: str) -> pd.DataFrame:
        return part_totals[part_totals["jurisdiction_type"].eq(jurisdiction_type)][
            ["state_fips", "state_abbr", "offense", "reported_total", "adjusted_total"]
        ].rename(
            columns={
                "reported_total": f"computed_{prefix}_reported_total",
                "adjusted_total": f"computed_{prefix}_adjusted_total",
            }
        )

    computed = (
        _part("municipal", "municipal")
        .merge(_part("nonmunicipal", "state_nonmunicipal_remainder"), on=["state_fips", "state_abbr", "offense"], how="outer")
        .merge(_part("overlap", "statewide_overlap_layer"), on=["state_fips", "state_abbr", "offense"], how="outer")
    )
    for col in [
        "computed_municipal_reported_total",
        "computed_municipal_adjusted_total",
        "computed_nonmunicipal_reported_total",
        "computed_nonmunicipal_adjusted_total",
        "computed_overlap_reported_total",
        "computed_overlap_adjusted_total",
    ]:
        computed[col] = pd.to_numeric(computed[col], errors="coerce").fillna(0.0)
    computed["computed_ags_core_reported_total"] = (
        computed["computed_municipal_reported_total"]
        + computed["computed_nonmunicipal_reported_total"]
        + computed["computed_overlap_reported_total"]
    )
    computed["computed_ags_core_adjusted_total"] = (
        computed["computed_municipal_adjusted_total"]
        + computed["computed_nonmunicipal_adjusted_total"]
        + computed["computed_overlap_adjusted_total"]
    )

    compare_cols = [
        "state_fips",
        "state_abbr",
        "offense",
        "ags_core_reported_total",
        "ags_core_adjusted_total",
        "municipal_reported_total",
        "municipal_adjusted_total",
        "nonmunicipal_reported_total",
        "nonmunicipal_adjusted_total",
        "overlap_reported_total",
        "overlap_adjusted_total",
    ]
    merged = state_controls[compare_cols].merge(computed, on=["state_fips", "state_abbr", "offense"], how="outer")
    deltas: dict[str, float] = {}
    bad_masks: list[pd.Series] = []
    for stored, recomputed in [
        ("ags_core_reported_total", "computed_ags_core_reported_total"),
        ("ags_core_adjusted_total", "computed_ags_core_adjusted_total"),
        ("municipal_reported_total", "computed_municipal_reported_total"),
        ("municipal_adjusted_total", "computed_municipal_adjusted_total"),
        ("nonmunicipal_reported_total", "computed_nonmunicipal_reported_total"),
        ("nonmunicipal_adjusted_total", "computed_nonmunicipal_adjusted_total"),
        ("overlap_reported_total", "computed_overlap_reported_total"),
        ("overlap_adjusted_total", "computed_overlap_adjusted_total"),
    ]:
        delta_col = f"{stored}_delta"
        merged[delta_col] = (
            pd.to_numeric(merged[stored], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged[recomputed], errors="coerce").fillna(0.0)
        )
        deltas[delta_col] = _max_abs(merged[delta_col])
        bad_masks.append(merged[delta_col].abs().gt(TOTAL_LANE_TOLERANCE))
    expected_pairs = EXPECTED_RELEASE_STATE_COUNT * len(OFFENSES_7)
    missing_state_offense_rows = merged[
        merged["ags_core_adjusted_total"].isna()
        | merged["computed_ags_core_adjusted_total"].isna()
    ].copy()
    bad_rows = merged[pd.concat(bad_masks, axis=1).any(axis=1)].copy()
    if len(state_controls) != expected_pairs:
        issues.append(
            "total_lane.state_remainder_reconciliation: "
            f"state control rows {len(state_controls)} != expected {expected_pairs}"
        )
    if not missing_state_offense_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.state_remainder_reconciliation: missing state/offense row on one side",
            missing_state_offense_rows,
            columns=["state_fips", "state_abbr", "offense", "ags_core_adjusted_total", "computed_ags_core_adjusted_total"],
        )
    if not bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.state_remainder_reconciliation: controls do not reconcile to state-control comparison",
            bad_rows.sort_values(["state_fips", "offense"], kind="mergesort"),
            columns=[
                "state_fips",
                "state_abbr",
                "offense",
                "ags_core_adjusted_total_delta",
                "ags_core_reported_total_delta",
                "municipal_adjusted_total_delta",
                "nonmunicipal_adjusted_total_delta",
                "overlap_adjusted_total_delta",
            ],
        )
    return {
        "ok": len(state_controls) == expected_pairs and missing_state_offense_rows.empty and bad_rows.empty,
        "path": str(state_controls_path),
        "present": True,
        "state_offense_rows": int(len(state_controls)),
        "expected_state_offense_rows": int(expected_pairs),
        "missing_state_offense_rows": int(len(missing_state_offense_rows)),
        "reconciliation_bad_rows": int(len(bad_rows)),
        "max_abs_deltas": deltas,
        "offending_rows_sample": _sample_records(
            bad_rows,
            columns=[
                "state_fips",
                "state_abbr",
                "offense",
                "ags_core_adjusted_total_delta",
                "ags_core_reported_total_delta",
                "municipal_adjusted_total_delta",
                "nonmunicipal_adjusted_total_delta",
                "overlap_adjusted_total_delta",
            ],
        ),
    }


def _check_county_level_plausibility(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    """Catch a county whose PUBLISHED total is implausibly low relative to peer counties
    of comparable population in the same state -- the symptom of a jurisdiction-control
    defect where a whole gapped agency's mass never made it into a county's total (see
    the state_nonmunicipal_remainder pool fix in jurisdiction_estimator.py / trend_fills.py
    and docs/STATE.md). External-referenced (peer counties), not self-referential: a
    county never gets compared to its own prior output.
    """
    path = output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet"
    columns = ["block_group_geoid", "state_fips", f"population_{YEAR}", "expected_count_total"]
    if not path.exists():
        issues.append(f"total_lane.county_level_plausibility: missing BG surface {path}")
        return {"ok": False, "present": False, "path": str(path)}
    try:
        bg = pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as exc:
        issues.append(f"total_lane.county_level_plausibility: BG surface missing required columns: {exc}")
        return {"ok": False, "present": True, "path": str(path), "required_columns_present": False}

    bg = bg.copy()
    bg["county_geoid"] = bg["block_group_geoid"].astype("string").str.zfill(12).str.slice(0, 5)
    bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    bg[f"population_{YEAR}"] = pd.to_numeric(bg[f"population_{YEAR}"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bg["expected_count_total"] = pd.to_numeric(bg["expected_count_total"], errors="coerce").fillna(0.0).clip(lower=0.0)

    county = (
        bg.groupby(["county_geoid", "state_fips"], dropna=False)
        .agg(population=(f"population_{YEAR}", "sum"), expected_total=("expected_count_total", "sum"))
        .reset_index()
    )
    county = county[county["population"].gt(float(COUNTY_PLAUSIBILITY_MIN_POPULATION))].copy()
    county["rate_per_100k"] = RATE_PER_100K * county["expected_total"] / county["population"]

    peer = (
        county.groupby("state_fips", dropna=False)["rate_per_100k"]
        .agg(state_peer_county_count="count", state_peer_median_rate="median")
        .reset_index()
    )
    county = county.merge(peer, on="state_fips", how="left")
    has_peer_reference = county["state_peer_county_count"].ge(COUNTY_PLAUSIBILITY_MIN_STATE_PEER_COUNTIES)
    county["state_peer_ratio"] = county["rate_per_100k"] / county["state_peer_median_rate"]

    # No cheap prior-vintage (prior release/year) per-county reference exists yet; the
    # floor is currently just the state-peer floor. Written as max() over both terms so
    # a prior-vintage floor can be added later without changing this check's shape.
    county["plausibility_floor"] = (
        float(COUNTY_PLAUSIBILITY_STATE_PEER_RATIO_MIN) * county["state_peer_median_rate"]
    )
    county["below_floor"] = has_peer_reference & county["rate_per_100k"].lt(county["plausibility_floor"])

    bad_rows = county[county["below_floor"]].copy().sort_values("state_peer_ratio", kind="mergesort")
    report_columns = [
        "county_geoid",
        "state_fips",
        "population",
        "expected_total",
        "rate_per_100k",
        "state_peer_median_rate",
        "state_peer_county_count",
        "state_peer_ratio",
    ]
    if not bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.county_level_plausibility: county published total expected count per "
            f"100k is below {COUNTY_PLAUSIBILITY_STATE_PEER_RATIO_MIN:.0%} of its state's "
            f">{COUNTY_PLAUSIBILITY_MIN_POPULATION / 1000:.0f}k-population county peer median",
            bad_rows,
            columns=report_columns,
        )
    return {
        "ok": bad_rows.empty,
        "path": str(path),
        "present": True,
        "min_population": float(COUNTY_PLAUSIBILITY_MIN_POPULATION),
        "min_state_peer_counties": int(COUNTY_PLAUSIBILITY_MIN_STATE_PEER_COUNTIES),
        "state_peer_ratio_min": float(COUNTY_PLAUSIBILITY_STATE_PEER_RATIO_MIN),
        "prior_vintage_ratio_min": float(COUNTY_PLAUSIBILITY_PRIOR_VINTAGE_RATIO_MIN),
        "prior_vintage_reference_available": False,
        "counties_checked": int(len(county)),
        "counties_with_state_peer_reference": int(has_peer_reference.sum()),
        "counties_flagged": int(len(bad_rows)),
        "flagged_sample": _sample_records(bad_rows, columns=report_columns, limit=50),
    }


def _check_high_population_spot_checks(
    *,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    jurisdictions = controls[
        ["jurisdiction_id", "jurisdiction_type", "jurisdiction_name", "state_fips", "bucket_population"]
    ].drop_duplicates(subset=["jurisdiction_id"])
    top = (
        jurisdictions[jurisdictions["jurisdiction_type"].eq("municipal")]
        .assign(bucket_population_num=lambda df: pd.to_numeric(df["bucket_population"], errors="coerce").fillna(0.0))
        .sort_values(["bucket_population_num", "jurisdiction_id"], ascending=[False, True], kind="mergesort")
        .head(HIGH_POPULATION_SPOT_CHECK_N)
        .copy()
    )
    spot = top[["jurisdiction_id", "state_fips", "jurisdiction_name", "bucket_population_num"]].merge(
        controls[
            [
                "jurisdiction_id",
                "offense",
                "preferred_source",
                "reported_count_preferred",
                TOTAL_LANE_TARGET_COLUMN,
            ]
        ],
        on="jurisdiction_id",
        how="left",
    )
    expected = top[["jurisdiction_id", "state_fips", "jurisdiction_name", "bucket_population_num"]].merge(
        pd.DataFrame({"offense": OFFENSES_7}),
        how="cross",
    )
    expected = expected.merge(
        spot,
        on=["jurisdiction_id", "state_fips", "jurisdiction_name", "bucket_population_num", "offense"],
        how="left",
    )
    expected["control_target"] = pd.to_numeric(expected[TOTAL_LANE_TARGET_COLUMN], errors="coerce")
    bad_rows = expected[expected["control_target"].isna() | expected["control_target"].le(0.0)].copy()
    if not bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.high_population_spot_checks: top municipal jurisdiction missing or nonpositive modeled control",
            bad_rows.sort_values(["bucket_population_num", "jurisdiction_id", "offense"], ascending=[False, True, True], kind="mergesort"),
            columns=[
                "jurisdiction_id",
                "jurisdiction_name",
                "state_fips",
                "bucket_population_num",
                "offense",
                "preferred_source",
                "reported_count_preferred",
                TOTAL_LANE_TARGET_COLUMN,
            ],
        )
    reported_zero_rows = expected[
        pd.to_numeric(expected["reported_count_preferred"], errors="coerce").fillna(0.0).le(0.0)
    ].copy()
    return {
        "ok": bad_rows.empty,
        "top_n": int(HIGH_POPULATION_SPOT_CHECK_N),
        "checked_jurisdiction_count": int(len(top)),
        "checked_jurisdiction_offense_rows": int(len(expected)),
        "missing_or_nonpositive_control_rows": int(len(bad_rows)),
        "reported_count_preferred_zero_rows_nonblocking": int(len(reported_zero_rows)),
        "offending_rows_sample": _sample_records(
            bad_rows,
            columns=[
                "jurisdiction_id",
                "jurisdiction_name",
                "state_fips",
                "bucket_population_num",
                "offense",
                "preferred_source",
                "reported_count_preferred",
                TOTAL_LANE_TARGET_COLUMN,
            ],
        ),
        "reported_zero_rows_sample_nonblocking": _sample_records(
            reported_zero_rows,
            columns=[
                "jurisdiction_id",
                "jurisdiction_name",
                "state_fips",
                "bucket_population_num",
                "offense",
                "preferred_source",
                "reported_count_preferred",
                TOTAL_LANE_TARGET_COLUMN,
            ],
        ),
    }


def _load_reviewed_exception_oris(path: Path, *, issues: list[str]) -> tuple[set[str], dict[str, Any]]:
    if not path.exists():
        return set(), {"path": str(path), "present": False, "rows": 0}
    required = {"ori", "reason"}
    exceptions = pd.read_csv(path).copy()
    missing = required - set(exceptions.columns)
    if missing:
        issues.append(f"total_lane.consolidated_agency_population_detector: exceptions missing columns {sorted(missing)}")
        return set(), {"path": str(path), "present": True, "required_columns_present": False, "rows": int(len(exceptions))}
    exceptions["ori"] = exceptions["ori"].astype("string")
    exceptions["reason"] = exceptions["reason"].astype("string")
    missing_reason = exceptions["reason"].isna() | exceptions["reason"].str.strip().eq("")
    if bool(missing_reason.any()):
        _append_total_lane_issue(
            issues,
            "total_lane.consolidated_agency_population_detector: reviewed exception rows missing reason",
            exceptions.loc[missing_reason],
            columns=["ori", "reason"],
        )
    return set(exceptions.loc[~missing_reason, "ori"].dropna().astype(str)), {
        "path": str(path),
        "present": True,
        "required_columns_present": True,
        "rows": int(len(exceptions)),
        "missing_reason_rows": int(missing_reason.sum()),
    }


def _load_consolidated_footprints(path: Path, *, issues: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["ori", "principal_jurisdiction_id"]
    if not path.exists():
        return pd.DataFrame(columns=columns), {"path": str(path), "present": False, "rows": 0}
    footprints = pd.read_csv(path).copy()
    required = {"ori", "county_fips", "principal_jurisdiction_id", "excluded_place_geoids"}
    missing = required - set(footprints.columns)
    if missing:
        issues.append(f"total_lane.consolidated_agency_population_detector: footprints missing columns {sorted(missing)}")
        return pd.DataFrame(columns=columns), {
            "path": str(path),
            "present": True,
            "required_columns_present": False,
            "rows": int(len(footprints)),
        }
    footprints["ori"] = footprints["ori"].astype("string")
    footprints["principal_jurisdiction_id"] = footprints["principal_jurisdiction_id"].astype("string")
    return footprints[columns].copy(), {
        "path": str(path),
        "present": True,
        "required_columns_present": True,
        "rows": int(len(footprints)),
    }


def _handled_consolidated_footprint_oris(
    *,
    output_dir: Path,
    footprints: pd.DataFrame,
    issues: list[str],
) -> tuple[set[str], dict[str, Any]]:
    component_audit_path = output_dir / f"allocation_component_denominator_audit_{YEAR}.parquet"
    if footprints.empty:
        return set(), {"component_audit_path": str(component_audit_path), "configured_rows": 0}
    if not component_audit_path.exists():
        issues.append(
            "total_lane.consolidated_agency_population_detector: "
            f"missing allocation component audit {component_audit_path}"
        )
        return set(), {
            "component_audit_path": str(component_audit_path),
            "component_audit_present": False,
            "configured_rows": int(len(footprints)),
        }
    components = pd.read_parquet(
        component_audit_path,
        columns=["jurisdiction_id", "jurisdiction_type", "component_count_after"],
    )
    consolidated_jurisdictions = set(
        components.loc[
            components["jurisdiction_type"].astype("string").eq(CONSOLIDATED_AGENCY_FOOTPRINT_TYPE)
            & pd.to_numeric(components["component_count_after"], errors="coerce").fillna(0.0).ge(0.0),
            "jurisdiction_id",
        ]
        .dropna()
        .astype(str)
    )
    handled = footprints[
        footprints["principal_jurisdiction_id"].astype(str).isin(consolidated_jurisdictions)
    ].copy()
    return set(handled["ori"].dropna().astype(str)), {
        "component_audit_path": str(component_audit_path),
        "component_audit_present": True,
        "configured_rows": int(len(footprints)),
        "consolidated_component_jurisdiction_count": int(len(consolidated_jurisdictions)),
        "handled_configured_rows": int(len(handled)),
    }


def _check_consolidated_agency_population_detector(
    *,
    output_dir: Path,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    footprint_path = REPO_ROOT / "configs" / "consolidated_agency_footprints.csv"
    exception_path = REPO_ROOT / "configs" / "consolidated_agency_detector_exceptions.csv"
    footprints, footprint_summary = _load_consolidated_footprints(footprint_path, issues=issues)
    handled_oris, handled_summary = _handled_consolidated_footprint_oris(
        output_dir=output_dir,
        footprints=footprints,
        issues=issues,
    )
    exception_oris, exception_summary = _load_reviewed_exception_oris(exception_path, issues=issues)

    agency_path = REPO_ROOT / "state" / "reference" / "agency_master.parquet"
    crosswalk_path = REPO_ROOT / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    observations_path = REPO_ROOT / "state" / "observations" / "agency_year_observations.parquet"
    missing_paths = [str(path) for path in [agency_path, crosswalk_path, observations_path] if not path.exists()]
    if missing_paths:
        issues.append(
            "total_lane.consolidated_agency_population_detector: missing required input paths "
            f"{missing_paths}"
        )
        return {
            "ok": False,
            "missing_input_paths": missing_paths,
            "footprints": footprint_summary,
            "handled_footprints": handled_summary,
            "exceptions": exception_summary,
        }

    agency = pd.read_parquet(
        agency_path,
        columns=["ori9", "state_fips", "state_abbr", "agency_name_std", "agency_type_norm"],
    )
    crosswalk = pd.read_parquet(
        crosswalk_path,
        columns=["ori", "state_fips", "jurisdiction_id", "relationship_type", "resolution_source"],
    ).rename(columns={"ori": "ori9"})
    observations = pd.read_parquet(
        observations_path,
        columns=["ori9", "year", "source", "offense", "count", "population"],
    )
    observations = observations[
        observations["year"].astype("Int64").eq(YEAR)
        & observations["offense"].astype("string").isin(OFFENSES_7)
    ].copy()
    observations["count"] = pd.to_numeric(observations["count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    observations["population"] = pd.to_numeric(observations["population"], errors="coerce").fillna(0.0).clip(lower=0.0)
    fbi_population = observations.groupby("ori9", dropna=False)["population"].max().rename("fbi_population").reset_index()
    source_totals = (
        observations.groupby(["ori9", "source"], dropna=False)["count"]
        .sum()
        .rename("agency_2024_count")
        .reset_index()
        .sort_values(["ori9", "agency_2024_count", "source"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates("ori9", keep="first")
        .rename(columns={"source": "agency_2024_count_source"})
    )

    municipal_controls = (
        controls[controls["jurisdiction_type"].eq("municipal")]
        .groupby("jurisdiction_id", dropna=False)
        .agg(
            jurisdiction_name=("jurisdiction_name", "first"),
            bucket_population=("bucket_population", "max"),
            preferred_relationship_all_exclusive=(
                "relationship_type_preferred",
                lambda s: bool(s.dropna().astype(str).eq("exclusive").all()),
            ),
        )
        .reset_index()
    )
    agency["state_fips"] = agency["state_fips"].astype("string").str.zfill(2)
    crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
    rows = (
        agency[agency["agency_type_norm"].astype("string").eq("local_police")]
        .merge(fbi_population, on="ori9", how="left")
        .merge(source_totals[["ori9", "agency_2024_count", "agency_2024_count_source"]], on="ori9", how="left")
        .merge(
            crosswalk[crosswalk["relationship_type"].astype("string").eq("exclusive")],
            on=["ori9", "state_fips"],
            how="inner",
        )
        .merge(municipal_controls, on="jurisdiction_id", how="inner")
    )
    for col in ["fbi_population", "agency_2024_count", "bucket_population"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["population_bucket_ratio"] = rows["fbi_population"] / rows["bucket_population"]
    candidates = rows[
        rows["preferred_relationship_all_exclusive"].eq(True)
        & rows["agency_2024_count"].gt(float(CONSOLIDATED_AGENCY_MIN_2024_COUNT))
        & rows["fbi_population"].gt(float(CONSOLIDATED_AGENCY_MIN_FBI_POPULATION))
        & rows["bucket_population"].gt(0.0)
        & rows["population_bucket_ratio"].gt(float(CONSOLIDATED_AGENCY_POPULATION_RATIO_THRESHOLD))
    ].copy()
    candidates["handled_by_consolidated_footprint"] = candidates["ori9"].astype(str).isin(handled_oris)
    candidates["configured_consolidated_footprint"] = candidates["ori9"].astype(str).isin(
        set(footprints["ori"].dropna().astype(str))
    )
    candidates["reviewed_exception"] = candidates["ori9"].astype(str).isin(exception_oris)
    bad_rows = candidates[
        ~candidates["handled_by_consolidated_footprint"] & ~candidates["reviewed_exception"]
    ].copy()
    if not bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.consolidated_agency_population_detector: municipal exclusive local-police agency has FBI population far above assigned bucket population without consolidated footprint or reviewed exception",
            bad_rows.sort_values("population_bucket_ratio", ascending=False, kind="mergesort"),
            columns=[
                "ori9",
                "state_abbr",
                "agency_name_std",
                "jurisdiction_id",
                "jurisdiction_name",
                "agency_2024_count",
                "agency_2024_count_source",
                "fbi_population",
                "bucket_population",
                "population_bucket_ratio",
                "resolution_source",
            ],
        )
    return {
        "ok": bad_rows.empty,
        "thresholds": {
            "min_2024_count": float(CONSOLIDATED_AGENCY_MIN_2024_COUNT),
            "min_fbi_population": float(CONSOLIDATED_AGENCY_MIN_FBI_POPULATION),
            "population_bucket_ratio": float(CONSOLIDATED_AGENCY_POPULATION_RATIO_THRESHOLD),
            "rationale": "catch consolidated city-county agencies at 2.6x+ without tripping current growth-staleness cases",
        },
        "footprints": footprint_summary,
        "handled_footprints": handled_summary,
        "exceptions": exception_summary,
        "candidate_rows": int(len(candidates)),
        "unhandled_rows": int(len(bad_rows)),
        "candidate_sample": _sample_records(
            candidates.sort_values("population_bucket_ratio", ascending=False, kind="mergesort"),
            columns=[
                "ori9",
                "state_abbr",
                "agency_name_std",
                "jurisdiction_id",
                "jurisdiction_name",
                "agency_2024_count",
                "agency_2024_count_source",
                "fbi_population",
                "bucket_population",
                "population_bucket_ratio",
                "configured_consolidated_footprint",
                "handled_by_consolidated_footprint",
                "reviewed_exception",
            ],
        ),
        "offending_rows_sample": _sample_records(
            bad_rows,
            columns=[
                "ori9",
                "state_abbr",
                "agency_name_std",
                "jurisdiction_id",
                "jurisdiction_name",
                "agency_2024_count",
                "fbi_population",
                "bucket_population",
                "population_bucket_ratio",
            ],
        ),
    }


def _component_control_reconciliation(
    *,
    output_dir: Path,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    component_audit_path = output_dir / f"allocation_component_denominator_audit_{YEAR}.parquet"
    if not component_audit_path.exists():
        issues.append(
            "total_lane.published_output_total_reconciliation: "
            f"missing allocation component audit {component_audit_path}"
        )
        return {"ok": False, "path": str(component_audit_path), "present": False}
    components = pd.read_parquet(
        component_audit_path,
        columns=["state_fips", "jurisdiction_id", "jurisdiction_type", "offense", "component_count_after"],
    )
    components["state_fips"] = components["state_fips"].astype("string").str.zfill(2)
    components["jurisdiction_id"] = components["jurisdiction_id"].astype("string")
    components["offense"] = components["offense"].astype("string")
    overlap_component = components["jurisdiction_type"].astype("string").str.contains("overlap", na=False)
    components.loc[overlap_component, "jurisdiction_id"] = (
        components.loc[overlap_component, "state_fips"] + ":statewide_overlap_layer"
    )
    remainder_component = components["jurisdiction_type"].astype("string").isin(
        ["localized_remainder_county_layer", "localized_remainder_residual_layer"]
    )
    components.loc[remainder_component, "jurisdiction_id"] = (
        components.loc[remainder_component, "state_fips"] + ":state_nonmunicipal_remainder"
    )
    component_sums = (
        components.groupby(["jurisdiction_id", "offense"], dropna=False)["component_count_after"]
        .sum()
        .rename("component_total")
        .reset_index()
    )
    target = controls[
        ~controls["state_fips"].isin(RELEASE_EXCLUDED_STATE_FIPS)
    ][["jurisdiction_id", "offense", "reported_count_preferred", TOTAL_LANE_TARGET_COLUMN]].copy()
    merged = target.merge(component_sums, on=["jurisdiction_id", "offense"], how="outer")
    merged["target_delta"] = (
        pd.to_numeric(merged["component_total"], errors="coerce").fillna(0.0)
        - pd.to_numeric(merged[TOTAL_LANE_TARGET_COLUMN], errors="coerce").fillna(0.0)
    )
    merged["reported_delta_nonblocking"] = (
        pd.to_numeric(merged["component_total"], errors="coerce").fillna(0.0)
        - pd.to_numeric(merged["reported_count_preferred"], errors="coerce").fillna(0.0)
    )
    bad_rows = merged[merged["target_delta"].abs().gt(TOTAL_LANE_TOLERANCE)].copy()
    if not bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.published_output_total_reconciliation: allocation component totals do not match modeled controls",
            bad_rows.sort_values("target_delta", key=lambda s: s.abs(), ascending=False, kind="mergesort"),
            columns=[
                "jurisdiction_id",
                "offense",
                "component_total",
                TOTAL_LANE_TARGET_COLUMN,
                "target_delta",
                "reported_count_preferred",
                "reported_delta_nonblocking",
            ],
        )
    return {
        "ok": bad_rows.empty,
        "path": str(component_audit_path),
        "present": True,
        "component_rows": int(len(components)),
        "jurisdiction_offense_rows": int(len(component_sums)),
        "target_column": TOTAL_LANE_TARGET_COLUMN,
        "target_reconciliation_bad_rows": int(len(bad_rows)),
        "max_abs_target_delta": _max_abs(merged["target_delta"]),
        "reported_count_preferred_diff_rows_nonblocking": int(
            merged["reported_delta_nonblocking"].abs().gt(TOTAL_LANE_TOLERANCE).sum()
        ),
        "max_abs_reported_count_preferred_delta_nonblocking": _max_abs(merged["reported_delta_nonblocking"]),
        "offending_rows_sample": _sample_records(
            bad_rows,
            columns=[
                "jurisdiction_id",
                "offense",
                "component_total",
                TOTAL_LANE_TARGET_COLUMN,
                "target_delta",
                "reported_count_preferred",
                "reported_delta_nonblocking",
            ],
        ),
    }


def _published_surface_state_total_reconciliation(
    *,
    output_dir: Path,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    target = (
        controls.groupby(["state_fips", "offense"], dropna=False)[TOTAL_LANE_TARGET_COLUMN]
        .sum()
        .rename("state_control_total")
        .reset_index()
    )
    surface_summaries: dict[str, Any] = {}
    for geography, path in [
        ("block_group_ags_core", output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet"),
        ("tract_ags_core", output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet"),
    ]:
        if not path.exists():
            issues.append(f"total_lane.published_output_total_reconciliation: missing {geography} surface {path}")
            surface_summaries[geography] = {"ok": False, "path": str(path), "present": False}
            continue
        cols = ["state_fips", *[f"expected_count_{offense}" for offense in OFFENSES_7]]
        surface = pd.read_parquet(path, columns=cols)
        surface["state_fips"] = surface["state_fips"].astype("string").str.zfill(2)
        long = surface.melt(
            id_vars=["state_fips"],
            value_vars=[f"expected_count_{offense}" for offense in OFFENSES_7],
            var_name="expected_count_field",
            value_name="published_total",
        )
        long["offense"] = long["expected_count_field"].str.removeprefix("expected_count_")
        sums = (
            long.groupby(["state_fips", "offense"], dropna=False)["published_total"]
            .sum()
            .reset_index()
        )
        merged = target.merge(sums, on=["state_fips", "offense"], how="outer")
        merged["delta"] = (
            pd.to_numeric(merged["published_total"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["state_control_total"], errors="coerce").fillna(0.0)
        )
        bad_rows = merged[merged["delta"].abs().gt(TOTAL_LANE_TOLERANCE)].copy()
        if not bad_rows.empty:
            _append_total_lane_issue(
                issues,
                f"total_lane.published_output_total_reconciliation: {geography} state/offense sums do not match controls",
                bad_rows.sort_values("delta", key=lambda s: s.abs(), ascending=False, kind="mergesort"),
                columns=["state_fips", "offense", "published_total", "state_control_total", "delta"],
            )
        surface_summaries[geography] = {
            "ok": bad_rows.empty,
            "path": str(path),
            "present": True,
            "state_offense_rows": int(len(merged)),
            "bad_rows": int(len(bad_rows)),
            "max_abs_delta": _max_abs(merged["delta"]),
            "offending_rows_sample": _sample_records(
                bad_rows,
                columns=["state_fips", "offense", "published_total", "state_control_total", "delta"],
            ),
        }
    return {
        "ok": all(surface.get("ok") is True for surface in surface_summaries.values()),
        "surfaces": surface_summaries,
    }


def _check_published_output_total_reconciliation(
    *,
    output_dir: Path,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    component_summary = _component_control_reconciliation(
        output_dir=output_dir,
        controls=controls,
        issues=issues,
    )
    surface_summary = _published_surface_state_total_reconciliation(
        output_dir=output_dir,
        controls=controls,
        issues=issues,
    )
    return {
        "ok": component_summary.get("ok") is True and surface_summary.get("ok") is True,
        "component_to_jurisdiction_control": component_summary,
        "published_bg_tract_state_totals": surface_summary,
    }


def _check_city_feed_exact_point_tripwire(*, issues: list[str]) -> dict[str, Any]:
    path = REPO_CITY_EXACT_POINT_QA
    exception_path = REPO_CITY_EXACT_POINT_EXCEPTIONS
    share_path = REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet"
    next_phase_path = REPO_ROOT / "state" / "modeling" / f"next_phase_validation_city_incident_share_surface_{YEAR}.parquet"
    dependency_paths = [p for p in [share_path, next_phase_path, exception_path] if p.exists()]
    if not path.exists():
        issues.append(f"total_lane.city_feed_exact_point_tripwire: missing QA artifact {path}")
        return {"ok": False, "path": str(path), "present": False}

    stale_dependencies = [str(p) for p in dependency_paths if path.stat().st_mtime < p.stat().st_mtime]
    if stale_dependencies:
        issues.append(
            "total_lane.city_feed_exact_point_tripwire: QA artifact is stale relative to "
            f"{stale_dependencies}"
        )

    required = {
        "city_name",
        "offense",
        "located_incident_count",
        "max_point_count",
        "max_point_share",
        "reviewed_exception",
    }
    qa = pd.read_csv(path)
    missing = required - set(qa.columns)
    if missing:
        issues.append(f"total_lane.city_feed_exact_point_tripwire: QA artifact missing columns {sorted(missing)}")
        return {
            "ok": False,
            "path": str(path),
            "present": True,
            "required_columns_present": False,
            "rows": int(len(qa)),
            "stale_dependencies": stale_dependencies,
        }

    qa["city_name"] = qa["city_name"].astype("string")
    qa["offense"] = qa["offense"].astype("string")
    qa["max_point_share"] = pd.to_numeric(qa["max_point_share"], errors="coerce").fillna(0.0)
    qa["located_incident_count"] = pd.to_numeric(qa["located_incident_count"], errors="coerce").fillna(0.0)
    qa["max_point_count"] = pd.to_numeric(qa["max_point_count"], errors="coerce").fillna(0.0)
    qa["reviewed_exception"] = qa["reviewed_exception"].astype("string").str.lower().isin({"true", "1", "yes"})
    qa_groups = set(zip(qa["city_name"].astype(str), qa["offense"].astype(str), strict=False))
    missing_groups: list[tuple[str, str]] = []
    if next_phase_path.exists():
        surface = pd.read_parquet(next_phase_path, columns=["city_name", "offense"])
    elif share_path.exists():
        surface = pd.read_parquet(share_path, columns=["city_name", "offense"])
    else:
        surface = pd.DataFrame(columns=["city_name", "offense"])
    if not surface.empty:
        surface["city_name"] = surface["city_name"].astype("string")
        surface["offense"] = surface["offense"].astype("string")
        surface_groups = set(zip(surface["city_name"].astype(str), surface["offense"].astype(str), strict=False))
        missing_groups = sorted(surface_groups - qa_groups)
        if missing_groups:
            issues.append(
                "total_lane.city_feed_exact_point_tripwire: QA artifact missing active city/offense groups "
                f"{missing_groups[:TOTAL_LANE_SAMPLE_LIMIT]}"
            )

    bad_rows = qa[
        qa["max_point_share"].ge(float(CITY_EXACT_POINT_SHARE_MAX))
        & qa["located_incident_count"].ge(float(CITY_EXACT_POINT_MIN_LOCATED_COUNT))
        & qa["max_point_count"].ge(float(CITY_EXACT_POINT_MIN_POINT_COUNT))
        & ~qa["reviewed_exception"]
    ].copy()
    if not bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.city_feed_exact_point_tripwire: active city/offense has unreviewed exact-point concentration at or above 0.5%",
            bad_rows.sort_values("max_point_share", ascending=False, kind="mergesort"),
            columns=[
                "city_name",
                "offense",
                "located_incident_count",
                "max_point_count",
                "max_point_share",
                "max_point_lat",
                "max_point_lon",
                "point_key",
            ],
        )
    return {
        "ok": not stale_dependencies and not missing_groups and bad_rows.empty,
        "path": str(path),
        "present": True,
        "required_columns_present": True,
        "exception_path": str(exception_path),
        "rows": int(len(qa)),
        "threshold_max_point_share": float(CITY_EXACT_POINT_SHARE_MAX),
        "min_located_incident_count": float(CITY_EXACT_POINT_MIN_LOCATED_COUNT),
        "min_point_count": float(CITY_EXACT_POINT_MIN_POINT_COUNT),
        "stale_dependencies": stale_dependencies,
        "missing_active_groups": len(missing_groups),
        "unreviewed_violation_rows": int(len(bad_rows)),
        "reviewed_exception_rows": int(qa["reviewed_exception"].sum()),
        "max_point_share": _finite_or_none(qa["max_point_share"].max()) if not qa.empty else None,
        "violation_sample": _sample_records(
            bad_rows,
            columns=[
                "city_name",
                "offense",
                "located_incident_count",
                "max_point_count",
                "max_point_share",
                "max_point_lat",
                "max_point_lon",
                "point_key",
            ],
        ),
    }


def _check_total_lane_qa(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    controls, controls_summary = _load_controls_for_total_lane(issues=issues)
    if controls is None:
        return {
            "ok": False,
            "controls": controls_summary,
        }
    duplicate_summary = _check_no_duplicate_control_totals(controls=controls, issues=issues)
    source_priority_summary = _check_source_priority_honored(controls=controls, issues=issues)
    state_reconciliation_summary = _check_state_remainder_reconciliation(controls=controls, issues=issues)
    high_population_summary = _check_high_population_spot_checks(controls=controls, issues=issues)
    consolidated_agency_summary = _check_consolidated_agency_population_detector(
        output_dir=output_dir,
        controls=controls,
        issues=issues,
    )
    published_reconciliation_summary = _check_published_output_total_reconciliation(
        output_dir=output_dir,
        controls=controls,
        issues=issues,
    )
    city_exact_point_summary = _check_city_feed_exact_point_tripwire(issues=issues)
    county_plausibility_summary = _check_county_level_plausibility(output_dir=output_dir, issues=issues)
    checks = {
        "no_duplicate_control_totals": duplicate_summary,
        "source_priority_honored": source_priority_summary,
        "state_remainder_reconciliation": state_reconciliation_summary,
        "high_population_spot_checks": high_population_summary,
        "consolidated_agency_population_detector": consolidated_agency_summary,
        "published_output_total_reconciliation": published_reconciliation_summary,
        "city_feed_exact_point_tripwire": city_exact_point_summary,
        "county_level_plausibility": county_plausibility_summary,
    }
    return {
        "ok": all(check.get("ok") is True for check in checks.values()),
        "controls": controls_summary,
        "target_column": TOTAL_LANE_TARGET_COLUMN,
        "tolerance": float(TOTAL_LANE_TOLERANCE),
        "checks": checks,
    }


def build_summary(*, state_output_dir: Path = STATE_OUTPUT_DIR) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    static_overwrite_summary = _check_no_exposure_tempered_calls(issues=issues)
    confidence_pure_enrichment_summary = _check_confidence_pure_enrichment(issues=issues)
    surfaces = [
        (
            "block_group_ags_core",
            state_output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet",
            "block_group",
        ),
        (
            "tract_ags_core",
            state_output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet",
            "tract",
        ),
        (
            "block_group_fbi_calibrated",
            state_output_dir / f"crimerisk_block_group_{YEAR}_fbi_calibrated.parquet",
            "block_group",
        ),
        (
            "tract_fbi_calibrated",
            state_output_dir / f"crimerisk_tract_{YEAR}_fbi_calibrated.parquet",
            "tract",
        ),
    ]
    surface_summaries = [
        _check_surface(label=label, path=path, geography=geography, issues=issues)
        for label, path, geography in surfaces
    ]
    burglary_tau_calibration_summary = _load_burglary_tau_calibration(issues=issues)
    build_manifest_summary = _check_build_manifest(
        output_dir=state_output_dir,
        issues=issues,
        burglary_tau_calibration=burglary_tau_calibration_summary,
    )
    next_phase_measurement_summary = _check_next_phase_measurement(issues=issues)
    dashboard_lookup_summary = _check_dashboard_lookup(issues=issues)
    external_surface_availability_summary = _check_external_surface_availability(issues=issues)
    connecticut_population_summary = _check_connecticut_population(output_dir=state_output_dir, issues=issues)
    acs_bg_vocabulary_summary = _check_acs_bg_vocabulary_coverage(output_dir=state_output_dir, issues=issues)
    sparse_transfer_policy_summary = _check_sparse_residual_transfer_policy(
        output_dir=state_output_dir,
        issues=issues,
        burglary_tau_calibration=burglary_tau_calibration_summary,
    )
    total_lane_qa_summary = _check_total_lane_qa(
        output_dir=state_output_dir,
        issues=issues,
    )
    spatial_artifact_gates_summary = _check_spatial_artifact_gates(
        output_dir=state_output_dir,
        issues=issues,
    )

    present_labels = {s["label"] for s in surface_summaries if s.get("present")}
    for required_label in ("block_group_ags_core", "tract_ags_core"):
        if required_label not in present_labels:
            issues.append(f"missing required AGS-core surface: {required_label}")
    for required_label in ("block_group_fbi_calibrated", "tract_fbi_calibrated"):
        if required_label not in present_labels:
            issues.append(f"missing required FBI-calibrated surface: {required_label}")

    # Tract counts must roll up from block-group counts within tolerance, per product.
    for variant in ("ags_core", "fbi_calibrated"):
        bg_path = state_output_dir / f"crimerisk_block_group_{YEAR}_{variant}.parquet"
        tr_path = state_output_dir / f"crimerisk_tract_{YEAR}_{variant}.parquet"
        if not (bg_path.exists() and tr_path.exists()):
            continue
        count_cols = [f"expected_count_{name}" for name in [*OFFENSES_7, *AGGREGATES]]
        bg = pd.read_parquet(bg_path, columns=["block_group_geoid", *count_cols])
        tr = pd.read_parquet(tr_path, columns=["tract_id", *count_cols])
        bg = bg.assign(tract_id=bg["block_group_geoid"].astype("string").str.slice(0, 11))
        bg_rollup = bg.groupby("tract_id")[count_cols].sum()
        merged = tr.set_index(tr["tract_id"].astype("string"))[count_cols].join(
            bg_rollup, rsuffix="_bg", how="outer"
        )
        for col in count_cols:
            diff = (
                pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
                - pd.to_numeric(merged[f"{col}_bg"], errors="coerce").fillna(0.0)
            )
            if _max_abs(diff) > 1e-6:
                issues.append(
                    f"{variant}: tract {col} does not equal block-group rollup "
                    f"(max abs diff {_max_abs(diff):.3e})"
                )

    component_audit_path = state_output_dir / f"allocation_component_denominator_audit_{YEAR}.parquet"
    controls_path = REPO_ROOT / "state" / "controls" / f"jurisdiction_controls_{YEAR}.parquet"
    if component_audit_path.exists() and controls_path.exists():
        components = pd.read_parquet(
            component_audit_path,
            columns=["state_fips", "jurisdiction_id", "jurisdiction_type", "offense", "component_count_after"],
        )
        controls = pd.read_parquet(
            controls_path,
            columns=["state_fips", "jurisdiction_id", "offense", "adjusted_count_ags_core"],
        )
        components["state_fips"] = components["state_fips"].astype("string").str.zfill(2)
        controls["state_fips"] = controls["state_fips"].astype("string").str.zfill(2)
        components = components[~components["state_fips"].isin(RELEASE_EXCLUDED_STATE_FIPS)].copy()
        controls = controls[~controls["state_fips"].isin(RELEASE_EXCLUDED_STATE_FIPS)].copy()
        overlap_component = components["jurisdiction_type"].astype("string").str.contains("overlap", na=False)
        components.loc[overlap_component, "jurisdiction_id"] = (
            components.loc[overlap_component, "state_fips"] + ":statewide_overlap_layer"
        )
        remainder_component = components["jurisdiction_type"].astype("string").isin(
            ["localized_remainder_county_layer", "localized_remainder_residual_layer"]
        )
        components.loc[remainder_component, "jurisdiction_id"] = (
            components.loc[remainder_component, "state_fips"] + ":state_nonmunicipal_remainder"
        )
        component_sums = (
            components.groupby(["jurisdiction_id", "offense"], dropna=False)["component_count_after"]
            .sum()
            .rename("component_total")
            .reset_index()
        )
        target = controls[["jurisdiction_id", "offense", "adjusted_count_ags_core"]].copy()
        merged = target.merge(component_sums, on=["jurisdiction_id", "offense"], how="outer")
        diff = (
            pd.to_numeric(merged["component_total"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["adjusted_count_ags_core"], errors="coerce").fillna(0.0)
        )
        max_abs = _max_abs(diff)
        if max_abs > 1e-6:
            issues.append(
                "ags_core: allocation component expected counts do not reconcile to jurisdiction controls "
                f"(max abs diff {max_abs:.3e})"
            )
    else:
        issues.append(
            f"missing allocation component audit or controls for BG->jurisdiction reconciliation: "
            f"{component_audit_path}, {controls_path}"
        )

    qa_summary = _load_json(PACKAGE_VALIDATION_DIR / "build_qa_summary.json") or _load_json(REPO_QA_SUMMARY)
    if qa_summary is not None:
        qa_outputs = qa_summary.get("outputs", {})
        if qa_outputs.get("fbi_calibrated_present") is False:
            issues.append("validation summary says FBI-calibrated outputs are absent")
        if qa_outputs.get("fbi_calibrated_current") is False:
            issues.append("validation summary says FBI-calibrated outputs are stale")

    return {
        "ok": not issues,
        "state_output_dir": str(state_output_dir),
        "static_no_exposure_tempered_calls": static_overwrite_summary,
        "static_confidence_pure_enrichment": confidence_pure_enrichment_summary,
        "surface_count": len(surface_summaries),
        "surfaces": surface_summaries,
        "build_manifest": build_manifest_summary,
        "burglary_tau_calibration": burglary_tau_calibration_summary,
        "next_phase_measurement": next_phase_measurement_summary,
        "dashboard_lookup": dashboard_lookup_summary,
        "external_surface_availability": external_surface_availability_summary,
        "connecticut_population": connecticut_population_summary,
        "acs_bg_vocabulary_coverage": acs_bg_vocabulary_summary,
        "sparse_residual_transfer_policy": sparse_transfer_policy_summary,
        "total_lane_qa": total_lane_qa_summary,
        "spatial_artifact_gates": spatial_artifact_gates_summary,
        "validation_summary_present": qa_summary is not None,
        "issues": issues,
    }, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Target build year under validation (drives output filenames, the manifest-year assertion, and target-year frame filters).",
    )
    args = parser.parse_args()

    _apply_target_year(int(args.year))
    output_dir = args.output_dir or STATE_OUTPUT_DIR
    summary, issues = build_summary(state_output_dir=output_dir)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
