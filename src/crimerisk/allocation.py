from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

from crimerisk.benchmark_imputation import (
    county_remainder_imputed_targets,
    load_benchmark_imputation_units,
)
from crimerisk.build_freshness import artifact_is_current, write_dependency_stamp
from crimerisk.confidence import build_confidence_artifacts, enrich_confidence_surfaces
from crimerisk.crime import OFFENSES_7
from crimerisk.city_residuals import (
    CityResidualConfig,
    apply_city_residual_model,
    apply_city_residual_prediction_surface,
    attach_city_residual_features,
    city_residual_fitted_model_path,
    city_residual_prediction_manifest_path,
    city_residual_prediction_surface_path,
    fit_city_residual_model_from_truth,
    load_city_residual_fitted_model,
    load_city_residual_prediction_manifest,
    load_city_residual_prediction_surface,
)
from crimerisk.city_shares import (
    CityIncidentShareBuildConfig,
    filter_excluded_city_keys,
    write_v2_city_incident_shares,
)
from crimerisk.crosswalk_shares import (
    assert_allocation_shares_conserve,
    normalize_block_group_allocation_shares,
)
from crimerisk.denominators import (
    BURGLARY_COMMERCIAL_WEIGHT_FALLBACK,
    DENOMINATOR_SOURCE_COLUMNS,
    LANDSCAN_DAY_POP_COLUMN,
    LANDSCAN_SOURCE_YEAR,
    PERSON_EXPOSURE_DENOMINATOR_OFFENSES,
    PRIMARY_DENOMINATOR_BY_OFFENSE,
    add_offense_denominators,
)
from crimerisk.model_surface import ModelSurfaceConfig, bg_feature_dependency_paths, build_bg_feature_frame, build_model_surface
from crimerisk.mixture_allocation import (
    ENVELOPE_MODE_HARD_CLIP,
    ENVELOPE_MODE_SOFT_SHRINKAGE,
    SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT,
    SOFT_SHRINKAGE_NU,
    MixtureAllocationConfig,
    MixtureRuntime,
    apply_mixture_to_model_lane,
    assert_mixture_share_invariants,
    compress_log_excess,
    mixture_experts_path,
    mixture_shares_audit_path,
    resolve_mixture_runtime,
    soft_shrinkage_record,
    summarize_mixture_shares,
)
from crimerisk.mixture_allocation import ship_weights_path as mixture_ship_weights_path
from crimerisk.exposure_ensemble import (
    CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE,
    ENSEMBLE_OFFENSES,
    EXPOSURE_NORMALIZER_VERSION,
    NORMALIZER_SEMANTICS,
    ExposureEnsembleConfig,
    ExposureEnsembleRuntime,
    attach_exposure_normalizers,
    exposure_normalizers_path,
    normalizer_id_column,
    normalizer_id_for_offense,
    opportunity_normalizer_column,
    resolve_exposure_ensemble_runtime,
    summarize_exposure_normalizers,
)
from crimerisk.exposure_ensemble import ensemble_weights_path as exposure_ensemble_weights_path
from crimerisk.composites import (
    COMPOSITE_VERSION,
    HARM_WEIGHTED_COUNT_COLUMN,
    PERSONAL_RELATIVE_SCORE_COLUMN,
    PROPERTY_RELATIVE_SCORE_COLUMN,
    CompositeRuntime,
    aggregate_index_fields,
    apply_count_first_composites,
    composite_normalizers,
    multi_offense_score_column,
    resolve_composite_runtime,
    severity_sensitivity,
    severity_sensitivity_path,
    severity_weights_path,
    summarize_count_first_composites,
    support_for_geo_id_col,
)
from crimerisk.special_use import (
    DISPLAY_POLICY_BY_TYPE,
    EDUCATION_JOB_SHARE_MIN,
    GROUP_QUARTERS_POPULATION_SHARE_MIN,
    OPEN_NATURAL_LAND_COVER_SHARE_MIN,
    POSTSECONDARY_ANCHOR_MIN,
    SPECIAL_USE_EMPLOYMENT_FLOOR,
    SPECIAL_USE_FEATURE_COLUMNS,
    SPECIAL_USE_HOUSEHOLD_FLOOR,
    SPECIAL_USE_OUTPUT_COLUMNS,
    SPECIAL_USE_PUBLISHED_COLUMNS,
    SPECIAL_USE_TAXONOMY_VERSION,
    SPECIAL_USE_TYPES,
    UNEXPLAINED_DAYTIME_PRESENCE_RATIO_MIN,
    apply_special_use_taxonomy,
    attach_special_use_features,
    coarser_recommendation_rows,
    primary_rate_allowed as special_use_primary_rate_allowed,
    resident_rate_allowed as special_use_resident_rate_allowed,
    summarize_special_use_taxonomy,
    typed_suppressed as special_use_typed_suppressed,
)
from crimerisk.uncertainty import (
    CALIBRATION_VERSION as UNCERTAINTY_CALIBRATION_VERSION,
    DEFAULT_N_DRAWS as UNCERTAINTY_DEFAULT_N_DRAWS,
    INDEX_BREAKS as UNCERTAINTY_INDEX_BREAKS,
    UNCERTAINTY_LAYER_VERSION,
    UncertaintyEngine,
    UncertaintyRuntime,
    apply_uncertainty_layer,
    cache_frame_to_summaries,
    components_signature,
    log_dispersion,
    rare_offense_suppressed_columns as uncertainty_rare_offense_suppressed_columns,
    read_cached_cells,
    resolve_uncertainty_runtime,
    rollup_draws,
    summaries_to_cache_frame,
    summarize_offense as summarize_uncertainty_offense,
    summarize_uncertainty_layer,
    uncertainty_summary_inputs_signature,
    uncertainty_cache_path,
    write_cached_cells,
)
from crimerisk.paths import RepoPaths
from crimerisk.service_scopes import attach_service_scope_columns
from crimerisk.reference import COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES
from crimerisk.agency_identity import (
    load_agency_jurisdiction_crosswalk,
    resolve_ori_succession,
)
from crimerisk.controls import (
    ControlBuildConfig,
    _controls_dependency_paths,
    _ensure_controls_dependencies,
    controls_artifacts_are_current,
    write_v2_controls,
)
from crimerisk.stage1_adjudications import (
    build_usability_directives,
    config_dir as stage1_adjudications_config_dir,
    load_explicit_succession_rulings,
)
from crimerisk.smoothed_controls import E1_HALFLIFE_YEARS, E1_SHRINKAGE_K, kernel_weights
from crimerisk.source_provenance import (
    CIUS_ORIGIN,
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
)
from crimerisk.source_selection import build_agency_preferred_observations
from crimerisk.stage_locks import stage_write_lock
from crimerisk.stage_locks import blockers_for_stage
from crimerisk.trend_fills import (
    AGENCY_TARGET_ESTIMATE_COLUMNS,
    FILL_MAX_REFERENCE_AGE_YEARS,
    add_preferred_support_flags,
    apply_masked_gap_reclassification,
    apply_stage1_adjudicated_usability,
    build_agency_allocation_target_estimates,
    build_agency_trend_fill_panel,
    build_masked_gap_flags,
    build_reference_year_masked_gap_years,
)
from crimerisk.level_lane import apply_level_lane_admission
from crimerisk.geometry import (
    GeometryBuildConfig,
    geometry_artifacts_are_current,
    write_v2_geometry,
)


PERSONAL_OFFENSES = ["murder", "rape", "robbery", "aggravated_assault"]
PROPERTY_OFFENSES = ["burglary", "larceny", "motor_vehicle_theft"]
SPARSE_BASELINE_TRANSFER_OFFENSES = frozenset(("murder", "rape"))
# Fork 6 selected 100 for both offenses at the lower boundary of its original grid.
# The v24 re-selection completed that censored search on the current model components,
# excluding every E4 comparator cell: murder selected the interior point K=1 with a
# 12.3% Poisson-deviance gain over K=100 on 12 eligible cities. Rape had only one
# eligible non-E4 city and therefore retained 100 under the pre-registered minimum-10
# rule. This remains information-keyed, not a fixed blend: a group with I model
# pseudo-events retains I/(I+K) of its tract-level model shape. Evidence and the
# pre-result protocol live in final_phase/rare_offense_reselection/.
DEFAULT_RARE_OFFENSE_INFORMATION_CONSTANT_BY_OFFENSE = (
    ("murder", 1.0),
    ("rape", 100.0),
)
DEFAULT_RESIDUAL_TRANSFER_TAU_BY_OFFENSE = tuple(
    (
        offense,
        0.0
        if offense in SPARSE_BASELINE_TRANSFER_OFFENSES
        else 0.5
        if offense == "burglary"
        else 1.0,
    )
    for offense in OFFENSES_7
)
RELEASE_EXCLUDED_STATE_FIPS = {"02", "15", "72"}
PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES = (
    "suburban_county_validation_case",
    "partial_year_municipal_validation_case",
)
EB_HARD_MIN_DENOMINATOR = 1.0
NON_RESIDENTIAL_HOUSEHOLD_FLOOR = 10.0
# Below roughly 50 measured residents/workers, one incident implies >2,000 per 100k;
# the denominator no longer measures a stable population at risk.
PERSON_EXPOSURE_DENOMINATOR_FLOOR = 50.0
# The same floor, restated for the cell class the v2 lanes created and the legacy lane could not
# reach: a cell with essentially NO residents that nonetheless publishes an opportunity rate.
#
# On the legacy lane the household floor closed those cells outright, so the 50-unit exposure floor
# never had to carry them. The typed taxonomy opens the opportunity arm on `industrial_employment`
# and `campus_institution` by TYPE, and the exposure ensemble no longer inflates the denominator by
# taking a max, so a zero-resident industrial block group can clear the 50-unit floor on a
# denominator of 60-odd person-equivalents and paint the map from one modelled event. Measured on
# the v2 candidate: 3,825 (BG x offense) cells published an opportunity index with zero residents,
# 222 of them above the top legend break, the loudest a robbery index of 25,485 on a denominator of
# 66.5 (BG 360050157001).
#
# The rule: BELOW the resident threshold the exposure floor already uses (50 residents), publishing
# ANY primary opportunity-rate index additionally requires the offense's own opportunity normalizer
# to reach 500 units. 500 is this repo's existing `popj > 500` support convention for quoting a
# rate off a jurisdiction-population denominator, reused rather than re-invented; at the index
# levels these cells sit at it keeps the jitter of a single event inside one legend bin instead of
# spanning the whole scale. It is a SUPPORT floor, not a doubt marker: below it the cell publishes
# counts and density, which is the taxonomy's own park/prison policy, with the `insufficient_
# exposure` display semantics the other exposure floors already use and no doubt language anywhere.
#
# A cell with 50 or more residents is untouched -- the landed 50-unit floor decides it, exactly as
# before. Documented in EXPOSURE_ENSEMBLE_CONTRACT.md and SPECIAL_USE_TAXONOMY_CONTRACT.md.
ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR = 500.0
ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN = "zero_resident_opportunity_rate_floor"
PERSON_EXPOSURE_FLOOR_OFFENSES = frozenset(("murder", "rape", "robbery", "aggravated_assault", "larceny"))
MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR = 50.0
BURGLARY_PREMISES_DENOMINATOR_FLOOR = 10.0
SPECIAL_USE_TRACT_PREFIX = "98"
SPECIAL_USE_EXPOSURE_DENOMINATOR_FLOOR = 10.0
SQ_METERS_PER_SQ_MILE = 2_589_988.11
TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO = 5.0
TRANSIENT_EXPOSURE_INDEX_THRESHOLD = 1000.0
# Measured direct-feed P99.9 within-jurisdiction rate-ratio envelopes. These are
# evidence constants, not fitted knobs: model-only robbery is repaired at block-
# group support and model-only murder at its published tract support. The repair
# happens on allocation components before output rates/indexes are constructed.
MODEL_ONLY_ROBBERY_BG_RATE_RATIO_CAP = 22.0
MODEL_ONLY_MURDER_TRACT_RATE_RATIO_CAP = 57.0
MODEL_ONLY_ALLOCATION_ENVELOPE_SPECS = (
    ("robbery", "bg_id", MODEL_ONLY_ROBBERY_BG_RATE_RATIO_CAP),
    ("murder", "tract_id", MODEL_ONLY_MURDER_TRACT_RATE_RATIO_CAP),
)
# PLAN.md item 5's second target. The caps above are the "hard envelope caps applied to model-lane
# BG rates": above P99.9 the unit is pulled back to EXACTLY the cap ratio, so a unit at 40x the
# jurisdiction median and one at 400x land on the same published number. The soft lane keeps the
# same trigger, the same eligibility, the same conservation and the same redistribution, and
# changes only where the excess lands: the log-scale excess over the cap ratio is passed through
# the mixture lane's compressor (one function, one nu, both call sites) instead of being deleted.
# There is no extrapolation term here -- the feature-hull proxy is a statement about the GBM's
# training rows, and this repair runs on assembled component counts where no such distance exists.
MODEL_ONLY_ALLOCATION_ENVELOPE_MODE_COLUMN = "allocation_envelope_mode"
# Estimate modes whose per-offense point display is suppressed for CELL-level
# denominator invalidity (below-floor or ambient-blind custom-footprint exposure) while the
# expected count itself stays real and conserved. Aggregate composites take these
# components at tract support — the same compositional pattern the rare offenses
# use — because the count is defensible; only the cell's own denominator is not.
DENOMINATOR_INVALID_ESTIMATE_MODES = frozenset(
    {"insufficient_exposure", "vehicle_denominator_invalid"}
)

# --- ambient-blind custom footprints (Stage 2 fork ruling 2, wired in the Stage 4/5 batch) ---
#
# A casino, trust-parcel or campus footprint is placed CORRECTLY and its counts are real, but
# its published denominator is its resident population, and the people who generate the counts
# are visitors who appear in neither LODES nor LandScan. Stillaguamish WA carries 31 counts
# against 16 residents; Poarch Creek AL 342 against 281; Seminole FL 751 against 2,352. There
# is no per-person rate to publish there, for the same reason there is none on an airfield, so
# these cells join the existing insufficient-exposure eligibility family: the index is
# suppressed, the count and the crime density still publish, and `denominator_reason` names the
# mechanism rather than borrowing the plain floor's.
#
# Three conditions, all measured, none of them a label:
#   1. a MAJORITY of the block group's modelled mass for that offense arrives through a
#      custom-footprint overlap layer (`footprint_derived_count_share_{offense}`);
#   2. the exposure denominator carries NO ambient lift -- neither the LODES jobs proxy nor
#      LandScan daytime population raised it above resident population, so the denominator is
#      residents and nothing else;
#   3. the implied RESIDENT rate is more than 3x the national resident rate for that offense,
#      i.e. the residents alone cannot account for the counts -- measured on the LOWER bound of
#      the count's exact-Poisson interval, not on the point count.
#
# Condition 3 is measured against the national rate over the rows publishable under the
# pre-existing floors, so the rule is one pass and not circular in its own suppression.
#
# The Poisson lower bound in condition 3 is what stops the rule firing on rare-offense
# allocation noise. On the point count it fired on 66 murder and 38 rape cells whose MEDIAN
# flagged count was 0.27 and 2.30: a quarter of one modelled murder over 1,137 residents clears
# 3x the national murder rate arithmetically while saying nothing about ambient exposure, and 17
# tracts would have lost a published murder index for it. The claim the rule makes is "the
# residents cannot account for the counts", and that claim is only evidence when the counts can
# be told apart from a much smaller number. Using the interval's lower bound is the surface's own
# existing device for that (POISSON_INTERVAL_ALPHA, the same interval published on every row) and
# adds no new threshold.
FOOTPRINT_DERIVED_MASS_SHARE_FLOOR = 0.5
AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO = 3.0
INSUFFICIENT_AMBIENT_EXPOSURE_REASON = "insufficient_ambient_exposure"


def _footprint_derived_count_col(offense: str) -> str:
    return f"footprint_derived_count_{offense}"


def _footprint_derived_share_col(offense: str) -> str:
    return f"footprint_derived_count_share_{offense}"


def _footprint_ambient_exposure_missing_col(offense: str) -> str:
    return f"footprint_ambient_exposure_missing_{offense}"
# Gate-06 trial redistribution degraded held-out allocation TVD, so the release
# path keeps validity-only suppression and emits the zero-target audit for review.
APPLY_ZERO_TARGET_REDISTRIBUTION = False
# Sentencing-days severity weights in the Cambridge Crime Harm Index tradition (Sherman, Neyroud &
# Neyroud 2016, "The Cambridge Crime Harm Index"); values are round starting-point approximations,
# documented as such, not the England/Wales schedule verbatim.
HARM_WEIGHTS = {
    "murder": 5475.0,
    "rape": 1825.0,
    "robbery": 365.0,
    "aggravated_assault": 180.0,
    "burglary": 90.0,
    "larceny": 7.0,
    "motor_vehicle_theft": 30.0,
}
# The legacy composite field set. The v2 count-first lane publishes a different one; both live in
# composites.py so the two lanes cannot drift apart in two files.
AGGREGATE_INDEX_FIELDS = aggregate_index_fields(count_first=False)
# RARE_OFFENSE_TRACT_SUPPORT — the person offenses whose published per-offense index and rate
# point estimates are carried only at census tract and coarser. At block-group support a single
# year of murder/rape is Poisson noise on a model prior, so the point value is not a defensible
# quantity there (see the decision record in docs/STATE.md). Block groups keep the expected
# counts (for tract/aggregate reconciliation and reproducibility) and all reliability/diagnostic
# metadata; the block-group index and rate fields for these offenses are null by policy.
# Aggregates stay at block-group support but take their murder/rape terms at tract support: the
# harm-weighted index draws murder/rape as the tract count spread within the tract by
# person-exposure share, and the equal-offense index draws them from the parent tract's
# per-offense index (both the shrunken estimator the description and risk readings endorse).
RARE_OFFENSE_TRACT_SUPPORT = ("murder", "rape")
# alpha = EB prior strength (shrinkage toward the nested parent rate). Raised to 20 for the offenses
# whose diagnostic EB tails were historically most prior-sensitive. These values are retained only
# for diagnostic_eb_* fields; published rate/index fields are count-derived.
_EB_ALPHA_OVERRIDES = {
    "rape": 20.0,
    "aggravated_assault": 20.0,
    "burglary": 20.0,
    "motor_vehicle_theft": 20.0,
}
DEFAULT_EB_ALPHA_BY_OFFENSE = tuple(
    (offense, _EB_ALPHA_OVERRIDES.get(offense, 1.0)) for offense in OFFENSES_7
)
DEFAULT_MODEL_SURFACE_PRIOR_ANCHOR = "offense_denominator"
DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH = Path("state/modeling/feature_transfer_policy_2024.parquet")
DEFAULT_MODEL_SURFACE_EXCLUDE_FEATURE_POLICY_CLASSES = ("between_only", "excluded_protected")
DEFAULT_RESIDUAL_FEATURE_POLICY_PATH = DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH
DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES = DEFAULT_MODEL_SURFACE_EXCLUDE_FEATURE_POLICY_CLASSES
DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE = tuple(
    (offense, DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES) for offense in OFFENSES_7
)
CITY_INCIDENT_SHARE_SUM_TOLERANCE = 1e-6
CITY_INCIDENT_PARTIAL_YEAR_MIN_RATIO = 0.25
CITY_INCIDENT_PARTIAL_YEAR_MIN_COMPARISON_YEARS = 3
CITY_POSTERIOR_RECONCILIATION_TOLERANCE = 0.10
CITY_POSTERIOR_ALPHA_FLOOR = 1e-3
CITY_POSTERIOR_ALPHA_VOLUME_INCIDENTS = 25.0
CITY_POSTERIOR_ALPHA_MAX_PRIOR_FRACTION = 0.995
CITY_POSTERIOR_MATERIAL_TVD_THRESHOLD = 0.01
CITY_POSTERIOR_GROUP_COLS = ["jurisdiction_id", "state_fips", "offense"]
# Rolling-origin tract-share validation on 27 city-years / 6 cities selected the
# minimum incident-weighted TVD at a ten-year half-life and 100 prior incidents.
# The release-year jurisdiction total remains current; this history affects only
# the within-city tract distribution.  No incident is learned below tract support.
MURDER_INCIDENT_HALF_LIFE_YEARS = 10.0
MURDER_TRACT_POSTERIOR_PRIOR_INCIDENTS = 100.0
STATE_REMAINDER_SUFFIX = ":state_nonmunicipal_remainder"
STATE_REMAINDER_TYPE = "state_nonmunicipal_remainder"
STATE_OVERLAP_TYPE = "statewide_overlap_layer"
COUNTY_REMAINDER_TYPE = "localized_remainder_county_layer"
RESIDUAL_REMAINDER_TYPE = "localized_remainder_residual_layer"
COUNTY_OVERLAP_TYPE = "localized_overlap_county_layer"
# A county-anchored STATE POLICE group spreads over the non-municipal exposure of the county
# only. `county_overlap` spreads over every block group in the county by activity prior,
# incorporated ground included, which is wrong for a state police post: measured 2026-07-29,
# 25,832 of 47,131 county-anchored state-police counts (54.8%) landed on incorporated block
# groups a municipal PD owns (Stage 2 screen S2-4; KY Post 13 Hazard put ~55% of its mass on
# Hazard-city block groups KSP does not patrol).
#
# The basis is the state non-municipal remainder's block-group population share, which is
# exactly "ground no agency-bearing municipality covers" -- not "unincorporated", which would
# be wrong in New England, where towns with no police department are VSP's primary territory
# and DO appear in the remainder. Where a county has NO non-municipal exposure at all
# (Virginia independent cities, every Rhode Island county, Baltimore city) the restriction
# would strand the mass, so those fall back to the plain county spread, labelled.
COUNTY_NONMUNICIPAL_OVERLAP_KIND = "county_nonmunicipal_overlap"
COUNTY_NONMUNICIPAL_OVERLAP_TYPE = "localized_overlap_county_nonmunicipal_layer"
CONSOLIDATED_AGENCY_FOOTPRINT_TYPE = "consolidated_agency_footprint"


def _statewide_overlap_crosswalk_rows(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Return every link whose target is the state's overlap control.

    `relationship_type` describes why an agency owns the link; it is not the control
    lane.  Contract providers can therefore carry a `contract_covering_footprint`
    relationship while their non-contracted share still targets the statewide overlap
    jurisdiction.  Filtering on `relationship_type == "overlap"` silently discarded
    that mass during county localization even though Stage 1 and controls included it.
    """
    state = crosswalk["state_fips"].astype("string").str.zfill(2)
    expected_id = state + ":" + STATE_OVERLAP_TYPE
    return crosswalk[
        crosswalk["jurisdiction_id"].astype("string").eq(expected_id)
    ].copy()


def _build_overlap_evidence_spine(
    *,
    preferred: pd.DataFrame,
    agency_estimates: pd.DataFrame,
    overlap_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Attach overlap geometry to the complete admitted agency/offense universe.

    A missing target-year preferred observation is different from a reported zero. The
    allocation-estimate artifact preserves that distinction and carries admitted historical
    and repaired signals, so it defines the durable spine. Current preferred rows are unioned
    in to preserve newly reporting agencies that do not yet have an estimate row.
    """
    keys = ["ori9", "state_fips", "offense"]
    preferred = preferred.copy()
    preferred["ori9"] = preferred["ori9"].astype("string")
    preferred["state_fips"] = preferred["state_fips"].astype("string").str.zfill(2)
    preferred["offense"] = preferred["offense"].astype("string")
    if preferred.duplicated(keys).any():
        raise ValueError("Preferred overlap observations are not unique by agency/offense")

    if agency_estimates.empty:
        estimate_keys = pd.DataFrame(columns=keys)
    else:
        estimate_keys = agency_estimates[keys].copy()
        estimate_keys["ori9"] = estimate_keys["ori9"].astype("string")
        estimate_keys["state_fips"] = estimate_keys["state_fips"].astype("string").str.zfill(2)
        estimate_keys["offense"] = estimate_keys["offense"].astype("string")
        estimate_keys = estimate_keys.drop_duplicates(keys)

    universe = pd.concat([estimate_keys, preferred[keys]], ignore_index=True).drop_duplicates(keys)
    universe["agency_estimate_present"] = pd.MultiIndex.from_frame(universe[keys]).isin(
        pd.MultiIndex.from_frame(estimate_keys[keys])
    )
    universe["preferred_observation_present"] = pd.MultiIndex.from_frame(universe[keys]).isin(
        pd.MultiIndex.from_frame(preferred[keys])
    )
    universe = universe.merge(preferred, on=keys, how="left", validate="one_to_one")

    overlap_crosswalk = overlap_crosswalk.copy()
    overlap_crosswalk["ori"] = overlap_crosswalk["ori"].astype("string")
    overlap_crosswalk["state_fips"] = (
        overlap_crosswalk["state_fips"].astype("string").str.zfill(2)
    )
    return universe.merge(
        overlap_crosswalk,
        left_on=["ori9", "state_fips"],
        right_on=["ori", "state_fips"],
        how="inner",
    )


def _reviewed_succession_county_anchor_mask(
    frame: pd.DataFrame, rulings: pd.DataFrame
) -> pd.Series:
    """County placement inherited through a reviewed reporter-key migration."""
    if frame.empty or rulings.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    current = pd.MultiIndex.from_arrays(
        [
            frame["ori9"].astype("string").str.upper(),
            frame["state_abbr"].astype("string").str.upper(),
            frame["county_fips"].astype("string").str.zfill(3),
        ]
    )
    reviewed = pd.MultiIndex.from_arrays(
        [
            rulings["successor_ori"].astype("string").str.upper(),
            rulings["state"].astype("string").str.upper(),
            rulings["county_fips"].astype("string").str.zfill(3),
        ]
    )
    return pd.Series(current.isin(reviewed), index=frame.index, dtype=bool)

# --- the unlocated-mass bucket (Amendment 3 item 5) ----------------------------------------
# See analysis_scratch/final_phase/UNLOCATED_MASS_CONTRACT.md.
#
# Every overlap agency's mass has to land somewhere, and the cascade in
# `_build_overlap_group_targets` has exactly one terminal rung: spread it over every block group
# in the state by activity weight. That rung serves two populations that are not alike. A state
# highway patrol IS a statewide footprint, and spreading it is the right answer. An agency whose
# footprint the cascade could not resolve at all -- no registry ruling, no crosswalk subtype, no
# supported place, no authority-valid county -- is not statewide; it is UNKNOWN, and spreading it
# publishes a location claim about every block group in the state that no evidence supports.
#
# With `enable_unlocated_mass` on, the second population stops being placed. Its mass leaves the
# block-group surface and is published as a per-jurisdiction-offense `unlocated_count` in a
# companion table, and the conservation identity widens by exactly that much:
#
#     published block-group sum + unlocated + service exports - service imports
#         == source-state control                                (exact, per state and offense)
#
# This is a SOURCE-RESOLUTION statement, not a doubt statement. The counts are real and admitted;
# what is missing is a footprint to put them on. Nothing here says the published cells are less
# trustworthy, and no doubt language attaches to the affected block groups -- they simply stop
# receiving mass that was never theirs.
UNLOCATED_GROUP_KIND = "unlocated"
UNLOCATED_JURISDICTION_TYPE = "unlocated_mass_layer"
# The routes through the overlap cascade, declared once so `group_kind` and `resolution_route`
# cannot disagree: they are produced by ONE `np.select` over ONE condition list.
OVERLAP_ROUTE_REGISTRY_CUSTOM_FOOTPRINT = "registry_custom_footprint"
OVERLAP_ROUTE_REGISTRY_ABSORB = "registry_absorb_into_primary"
OVERLAP_ROUTE_REGISTRY_PLACE = "registry_localize_to_place"
OVERLAP_ROUTE_REGISTRY_COUNTY = "registry_localize_to_county"
OVERLAP_ROUTE_REGISTRY_STATEWIDE = "registry_keep_statewide"
OVERLAP_ROUTE_REGISTRY_HOLD = "registry_exclude_or_hold"
OVERLAP_ROUTE_STATE_POLICE_COUNTY = "state_police_county_subunit"
OVERLAP_ROUTE_COUNTY_NAME_AGREEMENT = "county_name_agreement"
OVERLAP_ROUTE_CROSSWALK_PLACE = "crosswalk_place"
OVERLAP_ROUTE_CROSSWALK_COUNTY = "crosswalk_county"
OVERLAP_ROUTE_CROSSWALK_STATEWIDE_GEOMETRY = "crosswalk_statewide_geometry"
OVERLAP_ROUTE_UNRESOLVED_NO_TARGET = "unresolved_no_local_target"
OVERLAP_ROUTE_UNSUPPORTED_COUNTY = "unresolved_unsupported_county"
OVERLAP_ROUTE_RARE_OFFENSE_DEMOTION = "rare_offense_county_evidence_demotion"
OVERLAP_ROUTE_UNATTRIBUTED_RESIDUAL = "unresolved_unattributed_state_residual"
OVERLAP_ROUTE_SERVICE_NO_ELIGIBLE_RECEIVER = "service_no_eligible_receiver"
# The two ways a custom-footprint agency's OWN admitted control mass can fail to reach a block
# group once the footprint layer is raked per ORI (see `_footprint_ledger_partition`). Both are
# resolution failures, not statewide findings, so both belong in the unlocated bucket: an ORI
# with ledger mass and nowhere to put it must be visible, never silent.
OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE = "footprint_suppressed_duplicate"
OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT = "footprint_no_placeable_support"
# The routes that are a resolution FAILURE rather than a statewide answer. Membership is the
# whole editorial content of the lane, so it is a declared constant and not a predicate scattered
# through the cascade.
#
# `registry_exclude_or_hold` is here because a registry ruling of "hold" is the reviewer saying
# the case is NOT adjudicated; routing it statewide was the cascade contradicting its own
# registry. `unresolved_unsupported_county` is here because a county target with no block-group
# prior support is a support failure, not a statewide finding.
#
# Deliberately NOT here: `registry_keep_statewide` and `crosswalk_statewide_geometry` (a reviewer
# or the crosswalk affirmatively said the footprint is the state) and
# `rare_offense_county_evidence_demotion` (a named policy rule with its own threshold -- murder
# and rape below three observed county counts do not anchor a county; that is a decision about
# evidence sufficiency, taken deliberately, and the state IS the answer it reaches).
UNLOCATED_ROUTES: frozenset[str] = frozenset(
    {
        OVERLAP_ROUTE_REGISTRY_HOLD,
        OVERLAP_ROUTE_UNRESOLVED_NO_TARGET,
        OVERLAP_ROUTE_UNSUPPORTED_COUNTY,
        OVERLAP_ROUTE_UNATTRIBUTED_RESIDUAL,
        OVERLAP_ROUTE_SERVICE_NO_ELIGIBLE_RECEIVER,
        OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE,
        OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT,
    }
)
UNLOCATED_MASS_KEY_COLUMNS: tuple[str, ...] = (
    "state_fips",
    "jurisdiction_id",
    "jurisdiction_type",
    "offense",
)
UNLOCATED_MASS_COLUMNS: tuple[str, ...] = (
    *UNLOCATED_MASS_KEY_COLUMNS,
    "unlocated_count",
    "control_count",
    "unlocated_share_of_control",
    *(f"unlocated_count_{route}" for route in sorted(UNLOCATED_ROUTES)),
)


def unlocated_mass_path(output_dir: Path, *, year: int) -> Path:
    return Path(output_dir) / f"unlocated_mass_{int(year)}.parquet"


COUNTY_ANCHOR_MIN_OBSERVED_OFFENSE_COUNT = 3.0
COUNTY_ANCHOR_MIN_EVIDENCE_OFFENSES = frozenset(("murder", "rape"))
COUNTY_ANCHOR_REPORT_ONLY_EVIDENCE_OFFENSES = frozenset(("robbery",))
RATE_PER_100K = 100000.0
POISSON_INTERVAL_ALPHA = 0.05
RELIABILITY_HIGH_SUPPORT_MIN = 25.0
RELIABILITY_MEDIUM_SUPPORT_MIN = 5.0
RELIABILITY_HIGH_MIN_SOURCE_YEARS = 3
RELIABILITY_MEDIUM_MIN_SOURCE_YEARS = 2
RELIABILITY_HIGH_INDEX_CI95_WIDTH_RATIO_MAX = 1.0
RELIABILITY_MEDIUM_INDEX_CI95_WIDTH_RATIO_MAX = 3.0


@dataclass(frozen=True)
class AllocationBuildConfig:
    year: int = 2024
    exclude_feed_city_keys: tuple[str, ...] = ()
    feed_year_end: int | None = None
    # Which already-built jurisdiction totals the spatial allocator distributes.
    # ``accounting`` is the single-year FBI-anchored edition. ``smoothed`` uses
    # the validated multi-year current-risk controls while keeping the same local
    # allocation model and publication denominators.
    control_surface: str = "accounting"
    force_controls_rebuild: bool = False
    force_reporting_regimes_rebuild: bool = False
    force_geometry_rebuild: bool = False
    force_bg_prior_rebuild: bool = False
    force_city_incident_share_rebuild: bool = False
    force_city_incident_source_refresh: bool = False
    # Where the city incident feed artifacts this build needs are written and read. None means
    # the production location under state/modeling. A build that truncates the feed year or drops
    # feed cities -- every evaluation fold does both -- is not building the published surface, so
    # it passes a run-scoped directory here and state/modeling is left exactly as it was.
    feed_inputs_dir: Path | None = None
    use_promoted_next_phase_allocator: bool = True
    residual_training_city_shares_path: Path | None = None
    residual_training_exclude_validation_case_types: tuple[str, ...] = ()
    residual_training_extra_bg_feature_paths: tuple[Path, ...] = ()
    bg_prior_path: Path | None = None
    model_surface_prior_anchor: str = DEFAULT_MODEL_SURFACE_PRIOR_ANCHOR
    model_surface_feature_policy_path: Path | None = DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH
    model_surface_exclude_feature_policy_classes: tuple[str, ...] = DEFAULT_MODEL_SURFACE_EXCLUDE_FEATURE_POLICY_CLASSES
    residual_feature_policy_path: Path | None = DEFAULT_RESIDUAL_FEATURE_POLICY_PATH
    residual_exclude_feature_policy_classes: tuple[str, ...] = DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES
    residual_exclude_feature_policy_classes_by_offense: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE
    residual_transfer_tau_by_offense: tuple[tuple[str, float], ...] = DEFAULT_RESIDUAL_TRANSFER_TAU_BY_OFFENSE
    rare_offense_information_constant_by_offense: tuple[
        tuple[str, float], ...
    ] = DEFAULT_RARE_OFFENSE_INFORMATION_CONSTANT_BY_OFFENSE
    eb_alpha_by_offense: tuple[tuple[str, float], ...] = DEFAULT_EB_ALPHA_BY_OFFENSE
    eb_hard_min_denominator: float = EB_HARD_MIN_DENOMINATOR
    burglary_commercial_weight: float | None = None
    city_posterior_reconciliation_tolerance: float = CITY_POSTERIOR_RECONCILIATION_TOLERANCE
    city_posterior_alpha_floor: float = CITY_POSTERIOR_ALPHA_FLOOR
    city_posterior_alpha_volume_incidents: float = CITY_POSTERIOR_ALPHA_VOLUME_INCIDENTS
    city_posterior_alpha_max_prior_fraction: float = CITY_POSTERIOR_ALPHA_MAX_PRIOR_FRACTION
    enable_county_anchoring: bool = True
    # v2 build chain, second core item. OFF by default: the promoted chain keeps the legacy
    # prior-only model lane until the owner promotes the mixture candidate. See
    # analysis_scratch/final_phase/MIXTURE_ALLOCATOR_CONTRACT.md.
    enable_mixture_allocation: bool = False
    mixture_experts_path: Path | None = None
    mixture_weights_path: Path | None = None
    write_mixture_share_audit: bool = True
    # v2 build chain, third core item. OFF by default: the promoted chain keeps the deployed
    # hard-max person denominator until owner promotion. See
    # analysis_scratch/final_phase/EXPOSURE_ENSEMBLE_CONTRACT.md.
    enable_exposure_ensemble: bool = False
    exposure_normalizers_path: Path | None = None
    exposure_ensemble_weights_path: Path | None = None
    exposure_residential_leg_source: str = "landscan_night"
    # v2 build chain, fourth core item. OFF by default: the promoted chain keeps the legacy
    # index-averaging composites until owner promotion. Composes with the two flags above --
    # the count-first composites consume expected counts and resident population, so they are
    # invariant to the exposure-ensemble lane entirely and move only with the model lane. See
    # analysis_scratch/final_phase/COMPOSITES_CONTRACT.md.
    enable_count_first_composites: bool = False
    severity_weights_path: Path | None = None
    # v2 build chain, sixth core item. OFF by default: the promoted chain keeps the blanket
    # 98-series suppression until owner promotion. Composes with the three flags above -- the
    # taxonomy decides WHO may publish a rate and the exposure lane decides WHAT the rate is
    # quoted against, so the two are orthogonal, and the count-first composites read the typed
    # resident gate in place of the blanket special-use term. See
    # analysis_scratch/final_phase/SPECIAL_USE_TAXONOMY_CONTRACT.md.
    enable_special_use_taxonomy: bool = False
    # v2 build chain, seventh and final core item. OFF by default: the promoted chain keeps the
    # Poisson-only reliability metadata until owner promotion. Composes with every flag above --
    # the draws are taken over whatever surface those flags produced, so the mixture lane changes
    # the shares that are perturbed, the exposure lane changes the denominator the rate quantiles
    # are quoted against, and the typed taxonomy changes which cells have a publishable index for
    # a bin probability to be about. This lane adds fields BESIDE the point payload and redefines
    # none of them. See analysis_scratch/final_phase/UNCERTAINTY_LAYER_CONTRACT.md.
    enable_uncertainty_layer: bool = False
    uncertainty_draws: int = UNCERTAINTY_DEFAULT_N_DRAWS
    uncertainty_control_dispersion_path: Path | None = None
    uncertainty_share_dispersion_path: Path | None = None
    uncertainty_calibration_path: Path | None = None
    # v2 build chain, PLAN.md item 5. OFF by default: the hard envelope caps stay until the owner
    # promotes this lane. It touches BOTH hard envelopes in the model lane -- the GBM expert's clip
    # into the learned intensity envelope (mixture_allocation.py) and the model-only within-
    # jurisdiction rate-ratio caps here -- and nothing else, so with the flag off every byte of
    # both the legacy chain and the current candidate is unchanged. The plan gates it on the
    # uncertainty layer existing, and it composes with that flag and every flag above: the shares
    # the draws perturb are the ones this lane shaped. See
    # analysis_scratch/final_phase/SOFT_SHRINKAGE_CONTRACT.md.
    enable_soft_shrinkage: bool = False
    soft_shrinkage_nu: float = SOFT_SHRINKAGE_NU
    soft_shrinkage_extrapolation_weight: float = SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT
    # The E5 silent-unit imputation package. It is a CONTROLS-lane flag (ControlBuildConfig), and
    # until now `build-outputs` had no way to pass it down, so a candidate built through
    # `build-outputs` silently got legacy controls no matter what the operator intended. Recorded
    # in the manifest for the same reason every other lane is.
    enable_imputation_v2: bool = False
    # Amendment 3 item 5. OFF by default: the promoted chain keeps forced statewide placement of
    # unresolved overlap footprints until the owner promotes this lane. It touches ONE site --
    # the terminal rung of the overlap cascade in `_build_overlap_group_targets` -- and nothing
    # else, so with the flag off every byte of both the legacy chain and the current candidate is
    # unchanged. It composes with every flag above without interacting with any of them: it
    # decides WHICH mass reaches the block-group surface at all, which is upstream of every
    # question those lanes answer (how the shares are shaped, what denominator a rate is quoted
    # against, who may publish, what the draws perturb). See
    # analysis_scratch/final_phase/UNLOCATED_MASS_CONTRACT.md.
    enable_unlocated_mass: bool = False


def promoted_residual_training_city_shares_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "modeling" / f"next_phase_validation_city_incident_share_surface_{int(year)}.parquet"


def promoted_residual_extra_bg_feature_paths(paths: RepoPaths) -> tuple[Path, ...]:
    return (
        paths.data_dir / "Overture-Places" / "parsed" / "block_group_overture_places_states_latest.parquet",
        paths.data_dir / "Overture-Places" / "parsed" / "block_group_overture_commercial_core_states_latest.parquet",
    )


def promoted_next_phase_allocator_required_paths(paths: RepoPaths, *, year: int) -> dict[str, Path]:
    extra_paths = promoted_residual_extra_bg_feature_paths(paths)
    return {
        "residual_training_city_shares": promoted_residual_training_city_shares_path(paths, year=year),
        "overture_places_bg_features": extra_paths[0],
        "overture_commercial_core_bg_features": extra_paths[1],
    }


def resolve_allocation_build_config(paths: RepoPaths, *, config: AllocationBuildConfig) -> AllocationBuildConfig:
    feed_year_end = int(config.feed_year_end) if config.feed_year_end is not None else int(config.year)
    if feed_year_end < 2018 or feed_year_end > int(config.year):
        raise ValueError(
            f"feed_year_end must be between 2018 and the build year {int(config.year)}; "
            f"got {feed_year_end}"
        )
    config = replace(
        config,
        exclude_feed_city_keys=tuple(
            sorted({str(value).strip() for value in config.exclude_feed_city_keys if str(value).strip()})
        ),
        feed_year_end=feed_year_end,
    )
    if not config.use_promoted_next_phase_allocator:
        return config

    residual_training_city_shares_path = config.residual_training_city_shares_path
    residual_training_exclude_validation_case_types = tuple(config.residual_training_exclude_validation_case_types)
    residual_training_extra_bg_feature_paths = tuple(config.residual_training_extra_bg_feature_paths)

    promoted_city_shares = promoted_residual_training_city_shares_path(paths, year=config.year)
    promoted_extra_features = promoted_residual_extra_bg_feature_paths(paths)
    missing_promoted_paths: list[Path] = []
    if residual_training_city_shares_path is None and not promoted_city_shares.exists():
        missing_promoted_paths.append(promoted_city_shares)
    if not residual_training_extra_bg_feature_paths:
        missing_promoted_paths.extend(path for path in promoted_extra_features if not path.exists())
    if missing_promoted_paths:
        missing_text = "\n".join(f"- {path}" for path in missing_promoted_paths)
        raise FileNotFoundError(
            "Promoted next-phase allocator is enabled, but required promoted artifacts are missing:\n"
            f"{missing_text}\n"
            "Build the next-phase validation and Overture feature artifacts first, or pass "
            "--no-promoted-next-phase-allocator for an explicit comparison/fallback build."
        )

    if residual_training_city_shares_path is None:
        residual_training_city_shares_path = promoted_city_shares
        if not residual_training_exclude_validation_case_types:
            residual_training_exclude_validation_case_types = PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES

    if not residual_training_extra_bg_feature_paths:
        residual_training_extra_bg_feature_paths = promoted_extra_features

    # The original promoted configuration named the 2024 policy artifact directly.
    # Treat that legacy default as a template when building another target year so a
    # routine year advance cannot silently reuse an earlier edition's feature policy.
    model_surface_feature_policy_path = config.model_surface_feature_policy_path
    if (
        int(config.year) != 2024
        and model_surface_feature_policy_path == DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH
    ):
        model_surface_feature_policy_path = Path(
            f"state/modeling/feature_transfer_policy_{int(config.year)}.parquet"
        )
    residual_feature_policy_path = config.residual_feature_policy_path
    if (
        int(config.year) != 2024
        and residual_feature_policy_path == DEFAULT_RESIDUAL_FEATURE_POLICY_PATH
    ):
        residual_feature_policy_path = Path(
            f"state/modeling/feature_transfer_policy_{int(config.year)}.parquet"
        )

    return replace(
        config,
        residual_training_city_shares_path=residual_training_city_shares_path,
        residual_training_exclude_validation_case_types=residual_training_exclude_validation_case_types,
        residual_training_extra_bg_feature_paths=residual_training_extra_bg_feature_paths,
        model_surface_feature_policy_path=model_surface_feature_policy_path,
        residual_feature_policy_path=residual_feature_policy_path,
    )


def _path_stats(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
            "size_bytes": None,
            "mtime_utc": None,
        }
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(float(stat.st_mtime), tz=timezone.utc).isoformat(),
    }


def _landscan_lift_decision_record(paths: RepoPaths, *, year: int) -> dict[str, object]:
    path = paths.state_dir / "modeling" / f"landscan_lift_allocation_decision_{int(year)}.json"
    if not path.exists():
        return {
            "decision_path": str(path),
            "decision_record_present": False,
            "allocation_branch": "publication_denominators_only_allocation_baselines_current_exposure",
        }
    try:
        record = json.loads(path.read_text())
    except Exception:
        return {
            "decision_path": str(path),
            "decision_record_present": False,
            "allocation_branch": "publication_denominators_only_allocation_baselines_current_exposure",
            "read_error": True,
        }
    return {
        "decision_path": str(path),
        "decision_record_present": True,
        "allocation_branch": record.get(
            "allocation_branch",
            "publication_denominators_only_allocation_baselines_current_exposure",
        ),
        "accepted_for_allocation": record.get("accepted_for_allocation"),
        "murder_rape_tvd_current": record.get("murder_rape_tvd_current"),
        "murder_rape_tvd_lifted": record.get("murder_rape_tvd_lifted"),
        "murder_rape_tvd_delta_lift_minus_current": record.get(
            "murder_rape_tvd_delta_lift_minus_current"
        ),
        "murder_rape_bootstrap_se_delta": record.get("murder_rape_bootstrap_se_delta"),
        "criterion": record.get("criterion"),
        "gradient_gate_status": record.get("gradient_gate_status"),
    }


def _resolve_repo_path(paths: RepoPaths, path: Path | str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    parts = resolved.parts
    if parts and parts[0] == "state":
        return paths.state_dir.joinpath(*parts[1:])
    if parts and parts[0] == "data":
        return paths.data_dir.joinpath(*parts[1:])
    if parts and parts[0] == "archive":
        return paths.archive_dir.joinpath(*parts[1:])
    return paths.repo_root / resolved


def _residual_transfer_tau_dict(values: tuple[tuple[str, float], ...] | dict[str, float]) -> dict[str, float]:
    tau_by_offense = {str(offense): float(tau) for offense, tau in DEFAULT_RESIDUAL_TRANSFER_TAU_BY_OFFENSE}
    items = values.items() if isinstance(values, dict) else values
    for offense, raw_tau in items:
        offense_key = str(offense)
        if offense_key not in OFFENSES_7:
            raise ValueError(f"unknown offense in residual transfer tau config: {offense_key!r}")
        tau = float(raw_tau)
        if not np.isfinite(tau) or tau < 0.0 or tau > 1.0:
            raise ValueError(f"residual transfer tau for {offense_key} must be finite in [0, 1], got {raw_tau!r}")
        tau_by_offense[offense_key] = tau
    return tau_by_offense


def _rare_offense_information_constant_dict(
    values: tuple[tuple[str, float], ...] | dict[str, float],
) -> dict[str, float]:
    constants = {
        str(offense): float(value)
        for offense, value in DEFAULT_RARE_OFFENSE_INFORMATION_CONSTANT_BY_OFFENSE
    }
    items = values.items() if isinstance(values, dict) else values
    for offense, raw_value in items:
        offense_key = str(offense)
        if offense_key not in SPARSE_BASELINE_TRANSFER_OFFENSES:
            raise ValueError(
                "rare-offense information constant is only defined for murder/rape, "
                f"got {offense_key!r}"
            )
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"rare-offense information constant for {offense_key} must be finite and positive, "
                f"got {raw_value!r}"
            )
        constants[offense_key] = value
    return constants


def _apply_uncovered_rare_offense_tract_shrinkage(
    merged: pd.DataFrame,
    *,
    sparse_mask: pd.Series,
    information_constants: dict[str, float],
) -> pd.DataFrame:
    """Construct the shared tract prior for murder and uncovered rape, then spread by residents.

    The model contributes only its tract-level shape. Its group-level weight is
    ``I / (I + K_offense)``, where I is the unnormalised model pseudo-count mass over
    the jurisdiction/offense support. The baseline is residential population times
    the jurisdiction crosswalk share; it therefore contains no LODES industrial-job
    or LandScan workplace lift. Inside a selected tract, the tract share is spread to
    block groups by the same residential exposure.

    Direct murder groups enter ``sparse_mask`` so their later incident update starts from the
    same tract prior as uncovered murder. Direct rape groups remain on the legacy city path.
    """
    out = merged.copy()
    audit_defaults: dict[str, object] = {
        "rare_offense_allocation_policy": pd.NA,
        "rare_offense_information_constant": np.nan,
        "rare_offense_effective_model_information": np.nan,
        "rare_offense_model_information_weight": np.nan,
        "rare_offense_residential_exposure_weight": np.nan,
        "rare_offense_tract_model_share": np.nan,
        "rare_offense_tract_exposure_share": np.nan,
        "rare_offense_within_tract_exposure_share": np.nan,
    }
    for col, value in audit_defaults.items():
        if col not in out.columns:
            out[col] = value

    sparse = pd.Series(sparse_mask, index=out.index).fillna(False).astype(bool)
    if not bool(sparse.any()):
        return out

    work = out.loc[sparse].copy()
    group_cols = list(CITY_POSTERIOR_GROUP_COLS)
    group_keys = [work[col] for col in group_cols]
    tract_keys = [*group_keys, work["tract_id"].astype("string").str.zfill(11)]

    allocation = pd.to_numeric(work["allocation_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "pop20" in work.columns:
        residential = pd.to_numeric(work["pop20"], errors="coerce").fillna(0.0).clip(lower=0.0)
        residential_weight = residential * allocation
    else:
        # Backward-compatible only for synthetic/unit-test frames. Production crosswalk
        # loading carries pop20 and therefore never uses this fallback.
        residential_weight = allocation.copy()

    group_residential_total = residential_weight.groupby(group_keys, dropna=False).transform("sum")
    residential_weight = residential_weight.where(group_residential_total.gt(0.0), allocation)
    group_residential_total = residential_weight.groupby(group_keys, dropna=False).transform("sum")
    equal_weight = pd.Series(1.0, index=work.index, dtype=float)
    residential_weight = residential_weight.where(group_residential_total.gt(0.0), equal_weight)
    group_residential_total = residential_weight.groupby(group_keys, dropna=False).transform("sum")

    model_component = pd.to_numeric(work["model_component_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    model_information = model_component.groupby(group_keys, dropna=False).transform("sum")
    model_tract_mass = model_component.groupby(tract_keys, dropna=False).transform("sum")
    residential_tract_mass = residential_weight.groupby(tract_keys, dropna=False).transform("sum")

    model_tract_share = np.divide(
        model_tract_mass.to_numpy(dtype=float),
        model_information.to_numpy(dtype=float),
        out=np.zeros(len(work), dtype=float),
        where=model_information.gt(0.0).to_numpy(dtype=bool),
    )
    exposure_tract_share = np.divide(
        residential_tract_mass.to_numpy(dtype=float),
        group_residential_total.to_numpy(dtype=float),
        out=np.zeros(len(work), dtype=float),
        where=group_residential_total.gt(0.0).to_numpy(dtype=bool),
    )
    within_tract_exposure_share = np.divide(
        residential_weight.to_numpy(dtype=float),
        residential_tract_mass.to_numpy(dtype=float),
        out=np.zeros(len(work), dtype=float),
        where=residential_tract_mass.gt(0.0).to_numpy(dtype=bool),
    )

    constants = (
        work["offense"].astype(str).map(information_constants).astype(float)
    )
    information_weight = np.divide(
        model_information.to_numpy(dtype=float),
        model_information.to_numpy(dtype=float) + constants.to_numpy(dtype=float),
        out=np.zeros(len(work), dtype=float),
        where=np.isfinite(constants.to_numpy(dtype=float)) & constants.gt(0.0).to_numpy(dtype=bool),
    )
    tract_share = (
        information_weight * model_tract_share
        + (1.0 - information_weight) * exposure_tract_share
    )
    row_share = tract_share * within_tract_exposure_share

    share_sum = pd.Series(row_share, index=work.index).groupby(group_keys, dropna=False).transform("sum")
    row_share = np.divide(
        row_share,
        share_sum.to_numpy(dtype=float),
        out=np.zeros(len(work), dtype=float),
        where=share_sum.gt(0.0).to_numpy(dtype=bool),
    )
    check = pd.Series(row_share, index=work.index).groupby(group_keys, dropna=False).sum()
    if check.empty or not np.allclose(check.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-9):
        sample = check[~np.isclose(check, 1.0, rtol=0.0, atol=1e-9)].head(10).to_dict()
        raise ValueError(
            "uncovered rare-offense tract shrinkage shares do not conserve within group: "
            f"{sample}"
        )

    out.loc[sparse, "city_posterior_model_prior_raw"] = row_share
    out.loc[sparse, "city_residual_transfer_policy"] = "tract_information_shrinkage_residential_exposure"
    out.loc[sparse, "rare_offense_allocation_policy"] = "tract_information_shrinkage_residential_exposure"
    out.loc[sparse, "rare_offense_information_constant"] = constants.to_numpy(dtype=float)
    out.loc[sparse, "rare_offense_effective_model_information"] = model_information.to_numpy(dtype=float)
    out.loc[sparse, "rare_offense_model_information_weight"] = information_weight
    out.loc[sparse, "rare_offense_residential_exposure_weight"] = residential_weight.to_numpy(dtype=float)
    out.loc[sparse, "rare_offense_tract_model_share"] = model_tract_share
    out.loc[sparse, "rare_offense_tract_exposure_share"] = exposure_tract_share
    out.loc[sparse, "rare_offense_within_tract_exposure_share"] = within_tract_exposure_share
    return out


def _model_surface_config_from_allocation(
    *,
    paths: RepoPaths,
    config: AllocationBuildConfig,
    compute_diagnostics: bool = False,
) -> ModelSurfaceConfig:
    return ModelSurfaceConfig(
        year=int(config.year),
        compute_diagnostics=bool(compute_diagnostics),
        prior_anchor=str(config.model_surface_prior_anchor),
        feature_policy_path=_resolve_repo_path(paths, config.model_surface_feature_policy_path),
        exclude_feature_policy_classes=tuple(str(value) for value in config.model_surface_exclude_feature_policy_classes),
        burglary_commercial_weight=config.burglary_commercial_weight,
    )


def _is_default_step14_arm_b_model_surface(config: ModelSurfaceConfig) -> bool:
    feature_policy = config.feature_policy_path
    policy_name = Path(feature_policy).name if feature_policy is not None else None
    return (
        str(config.prior_anchor).strip().lower() == DEFAULT_MODEL_SURFACE_PRIOR_ANCHOR
        and policy_name == f"feature_transfer_policy_{int(config.year)}.parquet"
        and {
            str(value).strip().lower()
            for value in config.exclude_feature_policy_classes
        }
        == set(DEFAULT_MODEL_SURFACE_EXCLUDE_FEATURE_POLICY_CLASSES)
    )


def _bg_prior_cache_path(
    *,
    paths: RepoPaths,
    config: AllocationBuildConfig,
    model_surface_config: ModelSurfaceConfig,
) -> Path:
    explicit_path = _resolve_repo_path(paths, config.bg_prior_path)
    if explicit_path is not None:
        return explicit_path
    if _is_default_step14_arm_b_model_surface(model_surface_config):
        return paths.state_dir / "modeling" / f"bg_prior_long_{int(config.year)}_arm_b.parquet"
    return paths.state_dir / "modeling" / f"bg_prior_long_{int(config.year)}.parquet"


def promoted_next_phase_allocator_preflight(paths: RepoPaths, *, year: int) -> dict[str, object]:
    required_paths = promoted_next_phase_allocator_required_paths(paths, year=year)
    required_inputs = [
        {
            "role": role,
            **(_path_stats(path) or {"path": str(path), "exists": False}),
        }
        for role, path in required_paths.items()
    ]
    missing = [row["path"] for row in required_inputs if not bool(row.get("exists"))]
    return {
        "year": int(year),
        "ready": not missing,
        "missing_required_paths": missing,
        "required_inputs": required_inputs,
        "default_excluded_validation_case_types": list(PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES),
    }


def _promoted_next_phase_allocator_applied(paths: RepoPaths, *, config: AllocationBuildConfig) -> bool:
    if not config.use_promoted_next_phase_allocator:
        return False
    promoted_city_shares = promoted_residual_training_city_shares_path(paths, year=config.year)
    promoted_features = set(promoted_residual_extra_bg_feature_paths(paths))
    return (
        config.residual_training_city_shares_path == promoted_city_shares
        and promoted_features.issubset(set(config.residual_training_extra_bg_feature_paths))
    )


def _allocation_build_manifest(
    *,
    paths: RepoPaths,
    config: AllocationBuildConfig,
    summary: dict[str, object],
    output_paths: dict[str, Path | None],
    run_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    model_surface_config = _model_surface_config_from_allocation(paths=paths, config=config)
    residual_transfer_tau = _residual_transfer_tau_dict(config.residual_transfer_tau_by_offense)
    rare_offense_information_constants = _rare_offense_information_constant_dict(
        config.rare_offense_information_constant_by_offense
    )
    bg_prior_path = _bg_prior_cache_path(
        paths=paths,
        config=config,
        model_surface_config=model_surface_config,
    )
    applied_exposure_summary = (
        ((summary.get("resolved_config") or {}).get("exposure_ensemble") or {})
        if isinstance(summary.get("resolved_config"), dict)
        else {}
    )
    manifest = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "year": int(config.year),
        "summary": summary,
        "resolved_config": {
            "control_surface": str(config.control_surface),
            "exclude_feed_city_keys": sorted(set(config.exclude_feed_city_keys)),
            "feed_year_end": int(
                config.feed_year_end if config.feed_year_end is not None else config.year
            ),
            "use_promoted_next_phase_allocator": bool(config.use_promoted_next_phase_allocator),
            "promoted_next_phase_allocator_applied": _promoted_next_phase_allocator_applied(paths, config=config),
            "residual_training_city_shares_path": (
                str(config.residual_training_city_shares_path)
                if config.residual_training_city_shares_path is not None
                else None
            ),
            "residual_training_exclude_validation_case_types": list(
                config.residual_training_exclude_validation_case_types
            ),
            "residual_training_extra_bg_feature_paths": [
                str(path) for path in config.residual_training_extra_bg_feature_paths
            ],
            "bg_prior_path": str(bg_prior_path),
            "model_surface_prior_anchor": str(model_surface_config.prior_anchor),
            "model_surface_feature_policy_path": (
                str(model_surface_config.feature_policy_path)
                if model_surface_config.feature_policy_path is not None
                else None
            ),
            "model_surface_exclude_feature_policy_classes": list(
                model_surface_config.exclude_feature_policy_classes
            ),
            "residual_feature_policy_path": (
                str(_resolve_repo_path(paths, config.residual_feature_policy_path))
                if config.residual_feature_policy_path is not None
                else None
            ),
            "residual_exclude_feature_policy_classes": [
                str(value) for value in config.residual_exclude_feature_policy_classes
            ],
            "residual_exclude_feature_policy_classes_by_offense": {
                str(offense): [str(value) for value in classes]
                for offense, classes in config.residual_exclude_feature_policy_classes_by_offense
            },
            "residual_transfer_tau_by_offense": {
                str(offense): float(residual_transfer_tau[offense])
                for offense in OFFENSES_7
            },
            "rare_offense_tract_information_shrinkage": {
                "offenses": sorted(SPARSE_BASELINE_TRANSFER_OFFENSES),
                "support": "tract_then_residential_exposure_spread",
                "information_weight": "effective_model_information / (effective_model_information + K_offense)",
                "residential_exposure_source": "2020 Census block-group population * jurisdiction allocation_share",
                "industrial_jobs_exposure_included": False,
                "information_constant_by_offense": rare_offense_information_constants,
                "selection": (
                    "rare_offense_information_v2: five-fold grouped Poisson deviance; "
                    "E4 cells report-only; seed 20240728"
                ),
            },
            "eb_alpha_by_offense": {str(offense): float(alpha) for offense, alpha in config.eb_alpha_by_offense},
            "eb_hard_min_denominator": float(config.eb_hard_min_denominator),
            "person_exposure_denominator_policy": {
                "offenses": sorted(PERSON_EXPOSURE_DENOMINATOR_OFFENSES),
                "publication_formula": "max(daytime_population_jobs_proxy, landscan_day_pop) where landscan_day_pop > 0; otherwise daytime_population_jobs_proxy, then apply bounded HQ-jobs cap where triggered",
                "allocation_baseline_decision": _landscan_lift_decision_record(paths, year=int(config.year)),
                "allocation_baseline_policy": (
                    "LandScan is used for publication denominators only when the decision artifact rejects "
                    "allocation use; model-surface/prior construction keeps apply_landscan_day_floor=False."
                ),
                "hq_jobs_cap": {
                    "condition": "jobs_wac >= 5000 and residents + jobs_wac > 3 * max(landscan_day_pop, residents)",
                    "cap": "3 * max(landscan_day_pop, residents)",
                    "audit_columns": [
                        "person_exposure_before_hq_jobs_cap",
                        "person_exposure_hq_jobs_cap",
                        "person_exposure_hq_jobs_cap_candidate",
                        "person_exposure_hq_jobs_capped",
                    ],
                },
                "landscan_product": "LandScan USA",
                "landscan_source_year": int(LANDSCAN_SOURCE_YEAR),
                "landscan_license": "CC BY 4.0",
                "tourist_visitor_limitation": (
                    "LandScan USA day population does not include transitory populations such as tourists; "
                    "visitor-heavy areas remain flagged rather than fully ambient-adjusted."
                ),
            },
            "motor_vehicle_theft_denominator_policy": {
                "formula": "ACS household vehicles + LODES jobs_wac * county ACS B08301 drove-alone/carpool commute share",
                "publication_floor": float(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR),
                "floor_estimate_mode": "insufficient_exposure",
                "audit_columns": [
                    "aggregate_vehicles_total",
                    "county_auto_commute_vehicle_share",
                    "mvt_commuter_vehicle_proxy",
                    "vehicle_exposure_2024",
                ],
            },
            "zero_resident_opportunity_rate_policy": {
                # The v2 lanes' own floor, self-identifying like the lane blocks below: an artifact
                # that publishes opportunity rates on cells with essentially no residents says
                # under which support rule it did so, and a quoted cell is re-derivable from
                # `population_<year>` and `primary_denominator_<offense>` alone.
                "enabled": bool(
                    config.enable_special_use_taxonomy or config.enable_exposure_ensemble
                ),
                "condition": (
                    f"population < {float(PERSON_EXPOSURE_DENOMINATOR_FLOOR):g} requires "
                    f"primary_denominator_<offense> >= "
                    f"{float(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR):g}"
                ),
                "resident_threshold": float(PERSON_EXPOSURE_DENOMINATOR_FLOOR),
                "opportunity_normalizer_floor": float(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR),
                "offenses": sorted(OFFENSES_7),
                "floor_estimate_mode": "insufficient_exposure",
                "floor_denominator_reason": "insufficient_exposure",
                "audit_columns": [ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN],
                "contracts": [
                    "analysis_scratch/final_phase/EXPOSURE_ENSEMBLE_CONTRACT.md",
                    "analysis_scratch/final_phase/SPECIAL_USE_TAXONOMY_CONTRACT.md",
                ],
            },
            "city_posterior_reconciliation_tolerance": float(config.city_posterior_reconciliation_tolerance),
            "city_posterior_alpha_floor": float(config.city_posterior_alpha_floor),
            "city_posterior_alpha_volume_incidents": float(config.city_posterior_alpha_volume_incidents),
            "city_posterior_alpha_max_prior_fraction": float(config.city_posterior_alpha_max_prior_fraction),
            "unified_murder_tract_posterior": {
                "support": "census_tract",
                "incident_half_life_years": float(MURDER_INCIDENT_HALF_LIFE_YEARS),
                "prior_incidents": float(MURDER_TRACT_POSTERIOR_PRIOR_INCIDENTS),
                "within_tract_distribution": "rare_offense_prior_only",
                "calibration_path": (
                    f"state/modeling/murder_tract_posterior_calibration_{int(config.year)}.json"
                ),
            },
            "burglary_commercial_weight_override": (
                float(config.burglary_commercial_weight)
                if config.burglary_commercial_weight is not None
                else None
            ),
            "count_reliability": {
                "interval": {
                    "method": "Garwood Poisson/gamma 95pct count interval propagated to rate and index",
                    "alpha": float(POISSON_INTERVAL_ALPHA),
                },
                "support": {
                    "direct_incident_support_count": "pooled local city incident count for the cell/offense from the live direct city surface",
                    "model_only_effective_support": 0.0,
                },
                "tiers": {
                    "high_min_effective_support": float(RELIABILITY_HIGH_SUPPORT_MIN),
                    "high_min_direct_source_years": int(RELIABILITY_HIGH_MIN_SOURCE_YEARS),
                    "high_max_index_ci95_width_ratio": float(RELIABILITY_HIGH_INDEX_CI95_WIDTH_RATIO_MAX),
                    "medium_min_effective_support": float(RELIABILITY_MEDIUM_SUPPORT_MIN),
                    "medium_min_direct_source_years": int(RELIABILITY_MEDIUM_MIN_SOURCE_YEARS),
                    "medium_max_index_ci95_width_ratio": float(RELIABILITY_MEDIUM_INDEX_CI95_WIDTH_RATIO_MAX),
                },
            },
            "mixture_allocation": {
                # Self-identifying: an artifact built on the v2 model lane says so in its own
                # manifest, so a candidate can never be mistaken for a legacy-lane build.
                "enabled": bool(config.enable_mixture_allocation),
                "experts_path": (
                    str(config.mixture_experts_path)
                    if config.mixture_experts_path is not None
                    else str(mixture_experts_path(paths, year=int(config.year)))
                ),
                "weights_path": (
                    str(config.mixture_weights_path)
                    if config.mixture_weights_path is not None
                    else str(mixture_ship_weights_path(paths))
                ),
                "contract": "analysis_scratch/final_phase/MIXTURE_ALLOCATOR_CONTRACT.md",
            },
            "exposure_ensemble": {
                # Self-identifying, same reason as the mixture block above: an artifact whose
                # rates are quoted against the v2 opportunity normalizers says so in its own
                # manifest, so it can never be mistaken for a hard-max build.
                "enabled": bool(config.enable_exposure_ensemble),
                "normalizer_version": EXPOSURE_NORMALIZER_VERSION,
                "semantics": NORMALIZER_SEMANTICS,
                "normalizers_path": (
                    str(config.exposure_normalizers_path)
                    if config.exposure_normalizers_path is not None
                    else str(exposure_normalizers_path(paths, year=int(config.year)))
                ),
                "weights_path": (
                    str(config.exposure_ensemble_weights_path)
                    if config.exposure_ensemble_weights_path is not None
                    else str(exposure_ensemble_weights_path(paths))
                ),
                "normalizer_id_by_offense": {
                    offense: normalizer_id_for_offense(
                        offense, enabled=bool(config.enable_exposure_ensemble)
                    )
                    for offense in OFFENSES_7
                },
                "contract": "analysis_scratch/final_phase/EXPOSURE_ENSEMBLE_CONTRACT.md",
                # The loaded artifact, not caller intent, is authoritative for source/version.
                **(
                    applied_exposure_summary
                    if bool(config.enable_exposure_ensemble)
                    else {}
                ),
            },
            "count_first_composites": {
                # Self-identifying, same reason as the two blocks above: an artifact whose
                # composites are count-first says so, names the severity vector it used, and
                # names the fields the lane superseded and renamed.
                "enabled": bool(config.enable_count_first_composites),
                "version": COMPOSITE_VERSION,
                "severity_weights_path": (
                    str(config.severity_weights_path)
                    if config.severity_weights_path is not None
                    else str(severity_weights_path(paths))
                ),
                "contract": "analysis_scratch/final_phase/COMPOSITES_CONTRACT.md",
            },
            "special_use_taxonomy": {
                # Self-identifying, same reason as the three blocks above: an artifact whose
                # special-use cells were typed rather than blanket-suppressed says so, names the
                # taxonomy version, and states every threshold it classified on, so a quoted cell
                # can be re-derived without the code.
                "enabled": bool(config.enable_special_use_taxonomy),
                "version": SPECIAL_USE_TAXONOMY_VERSION,
                "types": list(SPECIAL_USE_TYPES),
                "display_policy_by_type": dict(DISPLAY_POLICY_BY_TYPE),
                "thresholds": {
                    "household_floor": float(SPECIAL_USE_HOUSEHOLD_FLOOR),
                    "employment_floor": float(SPECIAL_USE_EMPLOYMENT_FLOOR),
                    "group_quarters_population_share_min": float(
                        GROUP_QUARTERS_POPULATION_SHARE_MIN
                    ),
                    "education_job_share_min": float(EDUCATION_JOB_SHARE_MIN),
                    "postsecondary_anchor_min": float(POSTSECONDARY_ANCHOR_MIN),
                    "open_natural_land_cover_share_min": float(
                        OPEN_NATURAL_LAND_COVER_SHARE_MIN
                    ),
                    "unexplained_daytime_presence_ratio_min": float(
                        UNEXPLAINED_DAYTIME_PRESENCE_RATIO_MIN
                    ),
                },
                "contract": "analysis_scratch/final_phase/SPECIAL_USE_TAXONOMY_CONTRACT.md",
            },
            "uncertainty_layer": {
                # Self-identifying, same reason as the four blocks above: an artifact carrying
                # decision probabilities says which draw design, which measured dispersions and
                # which calibration vintage produced them, and it names the break set the bin
                # probability is a statement about -- so a quoted probability is re-derivable.
                "enabled": bool(config.enable_uncertainty_layer),
                "version": UNCERTAINTY_LAYER_VERSION,
                "calibration_version": UNCERTAINTY_CALIBRATION_VERSION,
                "n_draws": int(config.uncertainty_draws),
                "index_breaks": list(UNCERTAINTY_INDEX_BREAKS),
                "contract": "analysis_scratch/final_phase/UNCERTAINTY_LAYER_CONTRACT.md",
            },
            "soft_shrinkage": {
                # Self-identifying, same reason as the five blocks above: an artifact whose model
                # lane compressed its tails instead of truncating them says so, names both hard
                # envelopes it replaced, and states the two constants, so a quoted tail cell can be
                # re-derived from the raw prediction and the bound without the code.
                **soft_shrinkage_record(
                    MixtureAllocationConfig(
                        year=int(config.year),
                        enable_soft_shrinkage=bool(config.enable_soft_shrinkage),
                        soft_shrinkage_nu=float(config.soft_shrinkage_nu),
                        soft_shrinkage_extrapolation_weight=float(
                            config.soft_shrinkage_extrapolation_weight
                        ),
                    )
                ),
                "replaces": {
                    "gbm_expert_envelope_clip": {
                        "percentiles": list(MixtureAllocationConfig().envelope_percentiles),
                        "module": "crimerisk.mixture_allocation",
                    },
                    "model_only_within_jurisdiction_rate_ratio_caps": {
                        offense: float(cap)
                        for offense, _unit, cap in MODEL_ONLY_ALLOCATION_ENVELOPE_SPECS
                    },
                },
            },
            "imputation_v2": {
                # The controls lane this build ran its dependency rebuild under. Recorded here
                # because `build-outputs` is where a candidate is assembled, and a candidate that
                # cannot say which imputation lane its controls came from is not auditable.
                "enabled": bool(config.enable_imputation_v2),
                "lane": "ControlBuildConfig.enable_imputation_v2",
                "contract": "analysis_scratch/final_phase/IMPUTATION_V2_CONTRACT.md",
            },
            "unlocated_mass": {
                # Self-identifying, same reason as the blocks above. A surface whose conservation
                # identity is `published + unlocated == control` rather than `published ==
                # control` has to say so in its own manifest, or a reader who sums the block
                # groups and finds them short of the control has no way to tell a withheld bucket
                # from lost mass.
                "enabled": bool(config.enable_unlocated_mass),
                "identity": (
                    "published_block_group_sum + unlocated + service_exports - "
                    "service_imports == source_state_control"
                    if bool(config.enable_unlocated_mass)
                    else (
                        "published_block_group_sum + service_exports - service_imports "
                        "== source_state_control"
                    )
                ),
                "companion_table": (
                    unlocated_mass_path(Path("."), year=int(config.year)).name
                    if bool(config.enable_unlocated_mass)
                    else None
                ),
                "unlocated_routes": sorted(UNLOCATED_ROUTES),
                "located_statewide_routes": [
                    OVERLAP_ROUTE_REGISTRY_STATEWIDE,
                    OVERLAP_ROUTE_CROSSWALK_STATEWIDE_GEOMETRY,
                    OVERLAP_ROUTE_RARE_OFFENSE_DEMOTION,
                ],
                "jurisdiction_type": STATE_OVERLAP_TYPE,
                "semantics": "source_resolution",
                "contract": "analysis_scratch/final_phase/UNLOCATED_MASS_CONTRACT.md",
            },
            "force_controls_rebuild": bool(config.force_controls_rebuild),
            "force_reporting_regimes_rebuild": bool(config.force_reporting_regimes_rebuild),
            "force_geometry_rebuild": bool(config.force_geometry_rebuild),
            "force_bg_prior_rebuild": bool(config.force_bg_prior_rebuild),
            "force_city_incident_share_rebuild": bool(config.force_city_incident_share_rebuild),
            "force_city_incident_source_refresh": bool(config.force_city_incident_source_refresh),
        },
        "input_file_stats": {
            "residual_training_city_shares": _path_stats(config.residual_training_city_shares_path),
            "residual_training_extra_bg_features": [
                _path_stats(path) for path in config.residual_training_extra_bg_feature_paths
            ],
            "city_residual_prediction_surface": _path_stats(
                city_residual_prediction_surface_path(paths, year=int(config.year))
            ),
            "city_residual_prediction_manifest": _path_stats(
                city_residual_prediction_manifest_path(
                    city_residual_prediction_surface_path(paths, year=int(config.year))
                )
            ),
            "city_incident_share_surface": _path_stats(
                city_incident_share_surface_path(paths, feed_inputs_dir=config.feed_inputs_dir)
            ),
            "city_incident_reconciliation": _path_stats(
                city_incident_reconciliation_path(
                    paths, year=int(config.year), feed_inputs_dir=config.feed_inputs_dir
                )
            ),
            "bg_prior_long": _path_stats(bg_prior_path),
            "model_surface_feature_policy": _path_stats(model_surface_config.feature_policy_path),
            "residual_feature_policy": _path_stats(
                _resolve_repo_path(paths, config.residual_feature_policy_path)
            ),
            "burglary_tau_calibration": _path_stats(
                paths.state_dir / "modeling" / f"burglary_tau_calibration_{int(config.year)}.json"
            ),
            "murder_tract_posterior_calibration": _path_stats(
                paths.state_dir
                / "modeling"
                / f"murder_tract_posterior_calibration_{int(config.year)}.json"
            ),
            "jurisdiction_controls": _path_stats(
                paths.state_dir / "controls" / f"jurisdiction_controls_{int(config.year)}.parquet"
            ),
            "jurisdiction_controls_smoothed": (
                _path_stats(
                    paths.state_dir
                    / "controls"
                    / f"jurisdiction_controls_smoothed_{int(config.year)}.parquet"
                )
                if str(config.control_surface) == "smoothed"
                else None
            ),
            "state_control_comparison": _path_stats(paths.state_dir / "controls" / "state_control_comparison.parquet"),
            "block_group_crosswalk": _path_stats(
                paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
            ),
            "consolidated_agency_footprints": _path_stats(
                paths.repo_root / "configs" / "consolidated_agency_footprints.csv"
            ),
            "overlap_custom_footprints": _path_stats(
                paths.repo_root / "configs" / "overlap_custom_footprints.csv"
            ),
            "service_wide_agency_scopes": _path_stats(
                paths.repo_root / "configs" / "service_wide_agency_scopes.csv"
            ),
            "primary_service_response_policies": _path_stats(
                paths.repo_root / "configs" / "primary_service_response_policies.csv"
            ),
            "service_wide_footprint_coverage": _path_stats(
                paths.repo_root / "configs" / "service_wide_footprint_coverage.csv"
            ),
            "overlap_custom_footprint_resident_coverage": _path_stats(
                paths.repo_root
                / "configs"
                / "overlap_custom_footprint_resident_coverage.csv"
            ),
            "landscan_usa_2021_block_group": _path_stats(
                paths.data_dir / "LandScan-USA" / "block_group_landscan_usa_2021.parquet"
            ),
            "landscan_lift_allocation_decision": _path_stats(
                paths.state_dir / "modeling" / f"landscan_lift_allocation_decision_{int(config.year)}.json"
            ),
            "landscan_lift_allocation_decision_summary": _path_stats(
                paths.state_dir / "modeling" / f"landscan_lift_allocation_decision_summary_{int(config.year)}.csv"
            ),
        },
        "output_file_stats": {name: _path_stats(path) for name, path in output_paths.items()},
    }
    if run_metadata is not None:
        manifest["run"] = run_metadata
    return manifest


def _load_tiger_land_area(paths: RepoPaths, *, geography: str) -> pd.DataFrame:
    if geography == "block_group":
        root = paths.data_dir / "tiger_bg"
        pattern = "tl_2020_*_bg.zip"
        geoid_col = "block_group_geoid"
        width = 12
    elif geography == "tract":
        root = paths.data_dir / "tiger_tracts"
        pattern = "tl_2020_*_tract.zip"
        geoid_col = "tract_id"
        width = 11
    else:
        raise ValueError(f"unsupported TIGER geography: {geography!r}")

    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(pattern)):
        import geopandas as gpd

        tiger = gpd.read_file(path, ignore_geometry=True, columns=["GEOID", "ALAND"])
        frames.append(
            pd.DataFrame(
                {
                    geoid_col: tiger["GEOID"].astype("string").str.zfill(width),
                    "land_area_sq_mi": (
                        pd.to_numeric(tiger["ALAND"], errors="coerce").fillna(0.0).clip(lower=0.0)
                        / float(SQ_METERS_PER_SQ_MILE)
                    ),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=[geoid_col, "land_area_sq_mi"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(geoid_col, keep="first").reset_index(drop=True)


def _attach_tiger_land_area(
    frame: pd.DataFrame,
    *,
    land_area: pd.DataFrame,
    geoid_col: str,
    fallback_aland_col: str | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if not land_area.empty:
        area = land_area.copy()
        area[geoid_col] = area[geoid_col].astype("string")
        out[geoid_col] = out[geoid_col].astype("string")
        out = out.merge(area, on=geoid_col, how="left", suffixes=("", "_tiger"))
        if "land_area_sq_mi_tiger" in out.columns:
            out["land_area_sq_mi"] = pd.to_numeric(out["land_area_sq_mi_tiger"], errors="coerce").combine_first(
                pd.to_numeric(out.get("land_area_sq_mi"), errors="coerce")
            )
            out = out.drop(columns=["land_area_sq_mi_tiger"], errors="ignore")
    if fallback_aland_col is not None and fallback_aland_col in out.columns:
        fallback = (
            pd.to_numeric(out[fallback_aland_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            / float(SQ_METERS_PER_SQ_MILE)
        )
        out["land_area_sq_mi"] = pd.to_numeric(out.get("land_area_sq_mi"), errors="coerce").combine_first(fallback)
    out["land_area_sq_mi"] = pd.to_numeric(out.get("land_area_sq_mi"), errors="coerce").fillna(0.0).clip(lower=0.0)
    return out


def _load_bg_covariates(
    paths: RepoPaths,
    *,
    year: int,
    burglary_commercial_weight: float | None = None,
    exposure_ensemble: ExposureEnsembleRuntime | None = None,
    special_use_taxonomy: bool = False,
) -> pd.DataFrame:
    bg = add_offense_denominators(
        build_bg_feature_frame(paths=paths, year=year),
        paths=paths,
        year=year,
        burglary_commercial_weight=burglary_commercial_weight,
    )
    # Derived from columns the wide covariate frame already carries, so the taxonomy adds no
    # ingestion, no download and no dependency stamp -- and a legacy build gains no column.
    if special_use_taxonomy:
        bg = attach_special_use_features(bg)
    burglary_commercial_calibration = dict(bg.attrs.get("burglary_commercial_calibration", {}))
    bg = _attach_tiger_land_area(
        bg,
        land_area=_load_tiger_land_area(paths, geography="block_group").rename(columns={"block_group_geoid": "bg_id"}),
        geoid_col="bg_id",
        fallback_aland_col="aland20",
    )
    out = bg[
        [
            "bg_id",
            "tract_id",
            "state_fips",
            "population",
            "daytime_population_jobs_proxy",
            "exposure_proxy_2024",
            LANDSCAN_DAY_POP_COLUMN,
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
            *(SPECIAL_USE_FEATURE_COLUMNS if special_use_taxonomy else ()),
        ]
    ].copy()
    # The v2 opportunity normalizers ride alongside the legacy surfaces rather than replacing
    # them: the legacy columns stay in the frame (audit, the ambient-lift rules, the resident
    # arm) and only the published per-offense denominator switches, inside `_finalize_output`.
    if exposure_ensemble is not None:
        out = attach_exposure_normalizers(out, runtime=exposure_ensemble)
    out.attrs["burglary_commercial_calibration"] = burglary_commercial_calibration
    return out

def _load_bg_crosswalk(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    bg = normalize_block_group_allocation_shares(pd.read_parquet(path))
    # Re-normalised here rather than trusted from the artifact, so assert the invariant the
    # Stage-4 recipient floor made load-bearing before any lane multiplies by it.
    assert_allocation_shares_conserve(bg)
    # `allocation_basis` / `allocation_recipient_status` are deliberately NOT carried past
    # this point. They are per-crosswalk-row facts, and the consolidated-footprint lane
    # unions rows from two jurisdictions -- carrying either column through would force a
    # dominant label onto a mixture, which is the thing Stage 3 ruled out. The screens read
    # them from the crosswalk artifact instead.
    columns = [
        "state_fips",
        "block_group_geoid",
        "jurisdiction_id",
        "jurisdiction_type",
        "allocation_share",
    ]
    # Residential population is the rare-offense shrinkage baseline. It is
    # intentionally carried from the Census crosswalk instead of reconstructed from
    # the person-exposure denominator, which includes workplace/industrial activity.
    if "pop20" in bg.columns:
        columns.append("pop20")
    return bg[columns].copy()


def _load_controls(
    paths: RepoPaths,
    *,
    year: int,
    surface: str = "accounting",
) -> pd.DataFrame:
    accounting_path = paths.state_dir / "controls" / f"jurisdiction_controls_{int(year)}.parquet"
    accounting = pd.read_parquet(accounting_path)
    if surface == "accounting":
        accounting["control_surface"] = "accounting"
        return accounting
    if surface != "smoothed":
        raise ValueError(f"unknown control surface {surface!r}; expected 'accounting' or 'smoothed'")

    smoothed_path = (
        paths.state_dir / "controls" / f"jurisdiction_controls_smoothed_{int(year)}.parquet"
    )
    smoothed = pd.read_parquet(smoothed_path)
    keys = ["jurisdiction_id", "offense"]
    required = {*keys, "accounting_count", "smoothed_count", "estimator"}
    missing = sorted(required.difference(smoothed.columns))
    if missing:
        raise ValueError(f"smoothed controls missing required columns: {missing}")
    if smoothed.duplicated(keys).any():
        raise ValueError("smoothed controls have duplicate jurisdiction/offense rows")

    risk_cols = [
        *keys,
        "accounting_count",
        "smoothed_count",
        "estimator",
        "fallback_reason",
        "temporal_kernel",
        "halflife_used",
        "shrinkage_k",
        "ewa_weight",
        "clean_year_count",
        "latest_clean_year",
    ]
    risk_cols = [col for col in risk_cols if col in smoothed.columns]
    risk = smoothed[risk_cols].rename(
        columns={
            col: f"risk_control_{col}"
            for col in risk_cols
            if col not in keys and col not in {"accounting_count", "smoothed_count"}
        }
    )
    merged = accounting.merge(risk, on=keys, how="left", validate="one_to_one")
    if merged["smoothed_count"].isna().any():
        sample = merged.loc[merged["smoothed_count"].isna(), keys].head(10).to_dict("records")
        raise ValueError(f"smoothed controls do not cover the accounting spine; sample={sample}")

    accounting_count = pd.to_numeric(merged["accounting_count"], errors="coerce")
    canonical_count = pd.to_numeric(merged["adjusted_count_ags_core"], errors="coerce")
    mismatch = (accounting_count - canonical_count).abs()
    if bool(mismatch.gt(1e-6).any()):
        raise ValueError(
            "smoothed controls were built from a different accounting edition; "
            f"max abs mismatch={float(mismatch.max()):.6g}"
        )

    merged["adjusted_count_ags_core"] = (
        pd.to_numeric(merged["smoothed_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    merged["control_surface"] = "smoothed"
    return merged


def _load_state_controls(paths: RepoPaths, *, year: int) -> pd.DataFrame:
    path = paths.state_dir / "controls" / "state_control_comparison.parquet"
    df = pd.read_parquet(path)
    return df[df["year"].astype(int) == int(year)].copy()


def _load_crosswalk(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    return pd.read_parquet(path)


def _load_agency_master(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "reference" / "agency_master.parquet"
    return pd.read_parquet(path)


def _load_jurisdiction_master(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "reference" / "jurisdiction_master.parquet"
    return pd.read_parquet(path)


def _valid_county_fips(series: pd.Series) -> pd.Series:
    county = series.astype("string").str.zfill(3)
    return county.str.fullmatch(r"\d{3}").fillna(False) & county.ne("000")


def _agency_preferred_support_flags(preferred: pd.DataFrame) -> pd.DataFrame:
    return add_preferred_support_flags(preferred)


def _supported_counties_for_jurisdiction_type(bg_crosswalk: pd.DataFrame, jurisdiction_type: str) -> set[tuple[str, str]]:
    if bg_crosswalk.empty or "block_group_geoid" not in bg_crosswalk.columns:
        return set()
    rows = bg_crosswalk[bg_crosswalk["jurisdiction_type"].astype("string").eq(jurisdiction_type)].copy()
    if rows.empty:
        return set()
    rows["state_fips"] = rows["state_fips"].astype("string").str.zfill(2)
    rows["county_geoid"] = rows["state_fips"].astype(str) + rows["block_group_geoid"].astype("string").str.zfill(12).str.slice(2, 5)
    return set(zip(rows["state_fips"].astype(str), rows["county_geoid"].astype(str), strict=True))


def _supported_counties_for_bg_prior(bg_prior: pd.DataFrame) -> set[tuple[str, str]]:
    if bg_prior.empty or "bg_id" not in bg_prior.columns:
        return set()
    rows = bg_prior[["state_fips", "bg_id"]].drop_duplicates().copy()
    rows["state_fips"] = rows["state_fips"].astype("string").str.zfill(2)
    rows["county_geoid"] = rows["state_fips"].astype(str) + rows["bg_id"].astype("string").str.zfill(12).str.slice(2, 5)
    return set(zip(rows["state_fips"].astype(str), rows["county_geoid"].astype(str), strict=True))


def _county_supported_mask(frame: pd.DataFrame, supported: set[tuple[str, str]]) -> pd.Series:
    if not supported:
        return pd.Series(False, index=frame.index)
    keys = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "state_fips": frame["state_fips"].astype("string").str.zfill(2),
                "county_geoid": frame["county_geoid"].astype("string"),
            },
            index=frame.index,
        )
    )
    supported_index = pd.MultiIndex.from_tuples(sorted(supported), names=["state_fips", "county_geoid"])
    return pd.Series(keys.isin(supported_index), index=frame.index)


_AGENCY_ESTIMATES_MEMO: dict[tuple[str, int], pd.DataFrame] = {}
_AGENCY_RISK_SIGNAL_MEMO: dict[tuple[str, int], pd.DataFrame] = {}


def _agency_estimates_cache_path(paths: RepoPaths, year: int) -> Path:
    return paths.cache_dir / "allocation" / f"agency_allocation_target_estimates_{int(year)}.parquet"


def _agency_estimates_dependency_paths(paths: RepoPaths, *, year: int) -> list[Path]:
    """Conservative superset: the controls stage consumes the same computation and
    already declares its inputs, plus the identity/adjudication surfaces and modules
    trend_fills reaches directly when it rebuilds the panel from scratch.
    """
    module_root = Path(__file__).resolve().parent
    dependencies: list[Path] = [
        *_controls_dependency_paths(paths, year=int(year)),
        *sorted(stage1_adjudications_config_dir(paths).glob("*.csv")),
        paths.data_dir
        / f"FBI-CDE-Agency-Rosters-{int(year)}"
        / "parsed"
        / f"agency_rosters_{int(year)}.parquet",
        module_root / "agency_identity.py",
        module_root / "stage1_adjudications.py",
        module_root / "level_lane.py",
        module_root / "crime" / "municipal_totals.py",
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in dependencies:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def persist_agency_allocation_target_estimates_cache(
    *,
    paths: RepoPaths,
    year: int,
    agency_estimates: pd.DataFrame,
) -> Path:
    """Persist the controls-authoritative agency estimates for allocation reuse.

    Controls and allocation consume the same canonical target frame.  The writer
    records the exact dependency set used by the existing reader and refreshes its
    process-local memo only after the parquet has been atomically installed.
    """
    if list(agency_estimates.columns) != AGENCY_TARGET_ESTIMATE_COLUMNS:
        raise ValueError(
            "agency estimate cache frame does not match the canonical column schema"
        )
    identity_columns = ["ori9", "state_fips", "offense"]
    if agency_estimates[identity_columns].isna().any(axis=None):
        raise ValueError("agency estimate cache frame contains missing identity values")
    if bool(agency_estimates.duplicated(identity_columns).any()):
        raise ValueError("agency estimate cache frame contains duplicate agency-offense rows")
    counts = pd.to_numeric(agency_estimates["estimated_count"], errors="coerce")
    if bool((~np.isfinite(counts) | counts.lt(0.0)).any()):
        raise ValueError("agency estimate cache frame contains invalid estimated counts")
    sources = agency_estimates["agency_estimate_source"].astype("string").str.strip()
    if bool((sources.isna() | sources.eq("")).any()):
        raise ValueError("agency estimate cache frame contains missing source labels")

    cache_path = _agency_estimates_cache_path(paths, int(year))
    dependency_paths = _agency_estimates_dependency_paths(paths, year=int(year))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        agency_estimates.to_parquet(temporary, index=False)
        os.replace(temporary, cache_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_dependency_stamp(cache_path, dependency_paths)
    key = (str(paths.repo_root), int(year))
    _AGENCY_ESTIMATES_MEMO[key] = agency_estimates.copy()
    return cache_path


def _build_agency_allocation_target_estimates(
    *,
    paths: RepoPaths,
    year: int,
) -> pd.DataFrame:
    """Thin wrapper: per-agency target-year estimates are computed once in
    trend_fills.py (build_agency_allocation_target_estimates) and shared by both the
    county-remainder split below and the jurisdiction-target consumption layer in
    jurisdiction_targets.py, so both consumers see identical per-agency amounts.

    The sharing is realised here rather than by threading the frame through every
    caller: the computation rebuilds the 2018..year trend-fill panel, the masked-gap
    flags and the ORI succession ledger from scratch, and callers that do not pass
    `agency_estimates=` would otherwise pay for it again in the same process. Callers
    mutate the frame in place, so every hand-out is a copy.

    One representational caveat, measured rather than assumed: the parquet round-trip
    rewrites `pd.NA` as `None` in the two object-dtype audit columns
    (`agency_estimate_review_reason`, `reference_years_excluded`), which is enough to
    make `DataFrame.equals` report False even though every cell and every `isna()` agrees.
    Both columns are provenance-only -- produced in trend_fills.py and read nowhere else --
    and every numeric, identity and index value round-trips exactly, so the cached frame
    is output-equivalent. The round-trip is idempotent, so the cache is stable from the
    first write onward.
    """
    key = (str(paths.repo_root), int(year))
    cache_path = _agency_estimates_cache_path(paths, int(year))
    dependency_paths = _agency_estimates_dependency_paths(paths, year=int(year))
    memoised = _AGENCY_ESTIMATES_MEMO.get(key)
    if memoised is not None and artifact_is_current(cache_path, dependency_paths):
        return memoised.copy()
    if artifact_is_current(cache_path, dependency_paths):
        frame = pd.read_parquet(cache_path)
    else:
        frame = build_agency_allocation_target_estimates(paths=paths, year=int(year))
        persist_agency_allocation_target_estimates_cache(
            paths=paths,
            year=int(year),
            agency_estimates=frame,
        )
    _AGENCY_ESTIMATES_MEMO[key] = frame.copy()
    return frame.copy()


def _controls_are_smoothed(controls: pd.DataFrame) -> bool:
    return bool(
        "control_surface" in controls.columns
        and controls["control_surface"].astype("string").eq("smoothed").all()
    )


def _build_agency_risk_signals(
    *,
    paths: RepoPaths,
    year: int,
    fallback_estimates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one current-risk level per reporting identity and offense.

    Surface 2 used to smooth only the parent statewide remainder/overlap control and
    then split that risk total with the target year's agency counts.  A genuine
    target-year zero therefore became a zero *risk* footprint, while another state's
    differently scaled parent produced a hard wall.  This helper carries the selected
    E1 temporal kernel down one level before the parent total is reconciled across its
    local groups.

    The input is the same admitted preferred-observation spine used by the agency fill
    lane.  Corrupt historical reference years are excluded, explicit ORI successions
    carry predecessor history into the current reporter, and a current agency estimate
    is used only when fewer than two clean years survive.  These are allocation weights;
    the independently smoothed parent control remains the exact amount published.
    """
    key = (str(paths.repo_root), int(year))
    memoised = _AGENCY_RISK_SIGNAL_MEMO.get(key)
    if memoised is not None:
        return memoised.copy()

    panel = build_agency_trend_fill_panel(
        paths=paths,
        year_start=2018,
        year_end=int(year),
    )
    panel = apply_level_lane_admission(
        panel,
        paths=paths,
        target_year=int(year),
    ).panel
    masked = build_masked_gap_flags(
        paths,
        target_year=int(year),
        agency_panel=panel,
    )
    panel = apply_masked_gap_reclassification(
        panel,
        target_year=int(year),
        masked_gap_flags=masked,
    )
    directives = build_usability_directives(paths, target_year=int(year))
    panel, _ = apply_stage1_adjudicated_usability(
        panel,
        target_year=int(year),
        directives=directives,
    )
    reference_exclusions = build_reference_year_masked_gap_years(
        paths,
        agency_panel=panel,
        candidate_years=range(2018, int(year)),
    )
    if not reference_exclusions.empty:
        excluded = pd.MultiIndex.from_frame(
            reference_exclusions[["ori9", "offense", "year"]].assign(
                ori9=lambda d: d["ori9"].astype("string").str.upper(),
                offense=lambda d: d["offense"].astype("string"),
                year=lambda d: pd.to_numeric(d["year"], errors="coerce").astype("Int64"),
            )
        )
        panel_keys = pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "ori9": panel["ori9"].astype("string").str.upper(),
                    "offense": panel["offense"].astype("string"),
                    "year": pd.to_numeric(panel["year"], errors="coerce").astype("Int64"),
                },
                index=panel.index,
            )
        )
        panel = panel.loc[~panel_keys.isin(excluded)].copy()

    succession, _ = resolve_ori_succession(
        paths=paths,
        agency_panel=panel,
        agency_jurisdiction_crosswalk=load_agency_jurisdiction_crosswalk(paths),
        target_year=int(year),
        max_reference_age_years=FILL_MAX_REFERENCE_AGE_YEARS,
    )
    successor = (
        dict(
            zip(
                succession["superseded_ori9"].astype(str),
                succession["successor_ori9"].astype(str),
                strict=True,
            )
        )
        if not succession.empty
        else {}
    )
    panel["ori9"] = (
        panel["ori9"].astype("string").str.upper().replace(successor)
    )
    clean = (
        panel["usable_as_observed"].fillna(False).astype(bool)
        & pd.to_numeric(panel["preferred_months_reported"], errors="coerce").ge(12.0)
        & pd.to_numeric(panel["preferred_observation_weight"], errors="coerce").ge(1.0)
        & pd.to_numeric(panel["preferred_count"], errors="coerce").notna()
    )
    history = panel.loc[clean, ["ori9", "state_fips", "offense", "year", "preferred_count"]].copy()
    history["state_fips"] = history["state_fips"].astype("string").str.zfill(2)
    history["offense"] = history["offense"].astype("string")
    history["year"] = pd.to_numeric(history["year"], errors="coerce").astype(int)
    history["clean_count"] = (
        pd.to_numeric(history["preferred_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    # Successor and predecessor must be one reporting identity in a shared year, not
    # two additive witnesses.  The reviewed succession contract says the streams do
    # not overlap; max is a fail-safe against a duplicate/twin year.
    history = (
        history.groupby(["ori9", "state_fips", "offense", "year"], as_index=False)["clean_count"]
        .max()
    )
    history["_halflife"] = history["offense"].map(E1_HALFLIFE_YEARS).astype(float)
    history["_weight"] = kernel_weights(
        int(year) - history["year"],
        history["_halflife"],
    )
    history["_weighted"] = history["clean_count"] * history["_weight"]
    levels = (
        history.groupby(["ori9", "state_fips", "offense"], as_index=False)
        .agg(
            _weight_sum=("_weight", "sum"),
            _weighted_sum=("_weighted", "sum"),
            clean_year_count=("year", "size"),
        )
    )
    levels["ewa_count"] = np.where(
        levels["_weight_sum"].gt(0.0),
        levels["_weighted_sum"] / levels["_weight_sum"].replace(0.0, np.nan),
        np.nan,
    )

    fallback = (
        fallback_estimates.copy()
        if fallback_estimates is not None
        else _build_agency_allocation_target_estimates(paths=paths, year=int(year))
    )
    fallback = fallback[["ori9", "state_fips", "offense", "estimated_count"]].copy()
    fallback["ori9"] = fallback["ori9"].astype("string").str.upper().replace(successor)
    fallback["state_fips"] = fallback["state_fips"].astype("string").str.zfill(2)
    fallback["offense"] = fallback["offense"].astype("string")
    fallback["estimated_count"] = (
        pd.to_numeric(fallback["estimated_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    fallback = (
        fallback.groupby(["ori9", "state_fips", "offense"], as_index=False)["estimated_count"]
        .max()
    )
    out = fallback.merge(
        levels[["ori9", "state_fips", "offense", "ewa_count", "clean_year_count"]],
        on=["ori9", "state_fips", "offense"],
        how="outer",
        validate="one_to_one",
    )
    out["clean_year_count"] = (
        pd.to_numeric(out["clean_year_count"], errors="coerce").fillna(0).astype(int)
    )
    out["risk_signal_count"] = np.where(
        out["clean_year_count"].ge(2) & pd.to_numeric(out["ewa_count"], errors="coerce").notna(),
        pd.to_numeric(out["ewa_count"], errors="coerce"),
        pd.to_numeric(out["estimated_count"], errors="coerce"),
    )
    out["risk_signal_count"] = (
        pd.to_numeric(out["risk_signal_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    out = out[["ori9", "state_fips", "offense", "risk_signal_count", "clean_year_count"]]
    _AGENCY_RISK_SIGNAL_MEMO[key] = out
    return out.copy()


def _normalize_agency_name_for_county_match(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value or "").upper()
    text = re.sub(r"\bCNTY\b", "COUNTY", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _county_name_base(value: object) -> str:
    text = _normalize_agency_name_for_county_match(value)
    suffixes = [
        "CITY AND BOROUGH",
        "COUNTY AND BOROUGH",
        "CENSUS AREA",
        "MUNICIPALITY",
        "COUNTY",
        "PARISH",
        "BOROUGH",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if text == suffix:
                return text
            if text.endswith(f" {suffix}"):
                text = text[: -len(suffix)].strip()
                changed = True
                break
    return text


def _load_county_name_lookup(paths: RepoPaths) -> dict[str, str]:
    path = paths.data_dir / "Census-PopEst-2020-2025" / "co-est2025-alldata.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["SUMLEV", "STATE", "COUNTY", "CTYNAME"], encoding="latin1")
    df = df[pd.to_numeric(df["SUMLEV"], errors="coerce").eq(50)].copy()
    df["county_geoid"] = (
        pd.to_numeric(df["STATE"], errors="coerce").astype("Int64").astype("string").str.zfill(2)
        + pd.to_numeric(df["COUNTY"], errors="coerce").astype("Int64").astype("string").str.zfill(3)
    )
    df["county_name_base"] = df["CTYNAME"].map(_county_name_base)
    return {
        str(row.county_geoid): str(row.county_name_base)
        for row in df.itertuples(index=False)
        if pd.notna(row.county_geoid) and str(row.county_name_base).strip()
    }


def _contains_county_name(agency_name_norm: pd.Series, county_name: pd.Series) -> pd.Series:
    out = pd.Series(False, index=agency_name_norm.index)
    for name in sorted(set(county_name.dropna().astype(str)), key=len, reverse=True):
        if not name:
            continue
        mask = county_name.eq(name)
        pattern = rf"(?:^| ){re.escape(name)}(?: |$)"
        out.loc[mask] = agency_name_norm.loc[mask].str.contains(pattern, regex=True, na=False)
    return out


def _state_police_county_subunit_mask(
    merged: pd.DataFrame,
    name_norm: pd.Series,
    has_county: pd.Series,
    reviewed_county_identity: pd.Series | None = None,
) -> pd.Series:
    agency_type = merged.get("agency_type_norm", pd.Series("", index=merged.index)).astype("string")
    has_state_police_token = (
        name_norm.str.contains(r"\bSTATE POLICE\b", regex=True, na=False)
        | name_norm.str.contains(r"\bSTATE PATROL\b", regex=True, na=False)
        | name_norm.str.contains(r"\bHIGHWAY PATROL\b", regex=True, na=False)
        | name_norm.str.contains(r"\bSP\b", regex=True, na=False)
        | name_norm.str.contains(r"\bHP\b", regex=True, na=False)
    )
    has_local_subunit_token = (
        name_norm.str.contains(
            r"\b(?:COUNTY|CO|PARISH|POST|TROOP|BARRACK|STATION|DISTRICT|DETACHMENT)\b",
            regex=True,
            na=False,
        )
        | name_norm.str.contains(r"\bAREA OFFICE\b", regex=True, na=False)
        | name_norm.str.contains(r"^SP [A-Z0-9]+", regex=True, na=False)
    )
    if reviewed_county_identity is not None:
        has_local_subunit_token = (
            has_local_subunit_token
            | reviewed_county_identity.reindex(merged.index).fillna(False).astype(bool)
        )
    generic_or_hq = name_norm.str.contains(
        r"\b(?:HEADQUART|HQ|STATEWIDE|ACADEMY|TRAINING|LABORATORY|FORENSIC|COMMUNICATION|ADMIN|GENERAL|CAPITOL|GAMING|SPECIAL|EXECUTIVE|INVESTIGATION|BUREAU)\b",
        regex=True,
        na=False,
    ) | name_norm.str.contains(r"\bDEPARTMENT OF PUBLIC SAFETY\b", regex=True, na=False)
    bare_state_police = name_norm.isin(["STATE POLICE", "STATE PATROL", "HIGHWAY PATROL"])
    return (
        agency_type.eq("state_law_enforcement")
        & has_county
        & has_state_police_token
        & has_local_subunit_token
        & ~generic_or_hq
        & ~bare_state_police
    )


def _smoothed_county_remainder_partition(
    *,
    remainder_controls: pd.DataFrame,
    agency_groups: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    bg_prior: pd.DataFrame | None,
) -> pd.DataFrame:
    """Reconcile a smoothed state remainder to temporally smoothed county groups.

    Every county with actual remainder geometry receives an offense-specific prior
    proportional to that offense's block-group opportunity weight.  Multi-year agency
    history moves the county away from that prior, with the same E1 ``k`` used by the
    parent Surface-2 estimator.  The resulting scores are normalized once to the parent
    control, so no state/offense mass is created or lost.
    """
    columns = [
        "state_fips",
        "offense",
        "group_kind",
        "group_id",
        "target_count",
        "reported_count",
        "observed_target_count",
        "adjustment_target_count",
        "observed_raw_count",
        "adjustment_raw_count",
        "county_anchor_evidence_count",
        "county_anchor_supported",
    ]
    parents = remainder_controls[["state_fips", "offense", "state_target"]].copy()
    parents["state_fips"] = parents["state_fips"].astype("string").str.zfill(2)
    parents["offense"] = parents["offense"].astype("string")

    exposure_share = _nonmunicipal_bg_exposure_share(bg_crosswalk)
    if bg_prior is not None and not bg_prior.empty:
        prior = bg_prior[["state_fips", "bg_id", "offense", "bg_weight"]].copy()
        prior["state_fips"] = prior["state_fips"].astype("string").str.zfill(2)
        prior["bg_id"] = prior["bg_id"].astype("string").str.zfill(12)
        prior["offense"] = prior["offense"].astype("string")
        prior = prior.merge(exposure_share, on=["state_fips", "bg_id"], how="inner")
        prior["risk_exposure"] = (
            pd.to_numeric(prior["bg_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
            * pd.to_numeric(prior["nonmunicipal_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
    else:
        source = bg_crosswalk.copy()
        key = "block_group_geoid" if "block_group_geoid" in source.columns else "bg_id"
        source = source[source["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)].copy()
        source["state_fips"] = source["state_fips"].astype("string").str.zfill(2)
        source["bg_id"] = source[key].astype("string").str.zfill(12)
        source["risk_exposure"] = pd.to_numeric(
            source.get("pop20", source["allocation_share"]), errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        prior = source[["state_fips", "bg_id", "risk_exposure"]].merge(
            parents[["state_fips", "offense"]].drop_duplicates(),
            on="state_fips",
            how="inner",
        )
    prior["group_id"] = (
        prior["state_fips"]
        + ":state_nonmunicipal_remainder:county:"
        + prior["state_fips"]
        + prior["bg_id"].str.slice(2, 5)
    )
    county = (
        prior.groupby(["state_fips", "offense", "group_id"], as_index=False)["risk_exposure"]
        .sum()
    )
    county = county[county["risk_exposure"].gt(0.0)].copy()

    signals = agency_groups.copy()
    signals["risk_signal_raw"] = (
        pd.to_numeric(signals.get("risk_signal_count"), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(signals.get("weight"), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    signals = (
        signals.groupby(["state_fips", "offense", "group_kind", "group_id"], as_index=False)[
            "risk_signal_raw"
        ].sum()
    )
    county_signal = signals[signals["group_kind"].eq("county_remainder")][
        ["state_fips", "offense", "group_id", "risk_signal_raw"]
    ]
    county = county.merge(
        county_signal,
        on=["state_fips", "offense", "group_id"],
        how="left",
        validate="one_to_one",
    ).merge(parents, on=["state_fips", "offense"], how="inner", validate="many_to_one")
    county["risk_signal_raw"] = pd.to_numeric(
        county["risk_signal_raw"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    exposure_total = county.groupby(["state_fips", "offense"])["risk_exposure"].transform("sum")
    county["prior_count"] = np.where(
        exposure_total.gt(0.0),
        county["state_target"] * county["risk_exposure"] / exposure_total.replace(0.0, np.nan),
        0.0,
    )
    signal_weight = county["risk_signal_raw"] / (
        county["risk_signal_raw"] + float(E1_SHRINKAGE_K)
    )
    county["risk_score"] = (
        signal_weight * county["risk_signal_raw"]
        + (1.0 - signal_weight) * county["prior_count"]
    )
    county["group_kind"] = "county_remainder"

    residual = signals[signals["group_kind"].eq("residual_remainder")].copy()
    residual = residual.merge(
        parents,
        on=["state_fips", "offense"],
        how="inner",
        validate="many_to_one",
    )
    residual["risk_score"] = residual["risk_signal_raw"]
    scored = pd.concat(
        [
            county[["state_fips", "offense", "group_kind", "group_id", "state_target", "risk_signal_raw", "risk_score"]],
            residual[["state_fips", "offense", "group_kind", "group_id", "state_target", "risk_signal_raw", "risk_score"]],
        ],
        ignore_index=True,
    )
    present = scored[["state_fips", "offense"]].drop_duplicates()
    missing = parents.merge(present, on=["state_fips", "offense"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")].copy()
    if not missing.empty:
        scored = pd.concat(
            [
                scored,
                missing.assign(
                    group_kind="residual_remainder",
                    group_id=(
                        missing["state_fips"] + ":state_nonmunicipal_remainder:residual"
                    ),
                    risk_signal_raw=0.0,
                    risk_score=1.0,
                )[
                    ["state_fips", "offense", "group_kind", "group_id", "state_target", "risk_signal_raw", "risk_score"]
                ],
            ],
            ignore_index=True,
        )
    score_total = scored.groupby(["state_fips", "offense"])["risk_score"].transform("sum")
    scored["target_count"] = np.where(
        score_total.gt(0.0),
        scored["state_target"] * scored["risk_score"] / score_total.replace(0.0, np.nan),
        0.0,
    )
    scored["reported_count"] = 0.0
    scored["observed_target_count"] = 0.0
    scored["adjustment_target_count"] = scored["target_count"]
    scored["observed_raw_count"] = scored["risk_signal_raw"]
    scored["adjustment_raw_count"] = 0.0
    scored["county_anchor_evidence_count"] = scored["risk_signal_raw"]
    scored["county_anchor_supported"] = scored["group_kind"].eq("county_remainder")
    return scored[columns].sort_values(
        ["state_fips", "offense", "group_kind", "group_id"], kind="mergesort"
    ).reset_index(drop=True)


def _smoothed_overlap_partition(
    *,
    overlap_controls: pd.DataFrame,
    agency_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Partition each smoothed overlap parent with multi-year agency signals."""
    signals = agency_groups.copy()
    signals["risk_signal_raw"] = (
        pd.to_numeric(signals.get("risk_signal_count"), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(signals.get("weight"), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    grouped = (
        signals.groupby(["state_fips", "offense", "group_kind", "group_id"], as_index=False)[
            "risk_signal_raw"
        ].sum()
    )
    grouped = grouped.merge(
        overlap_controls[["state_fips", "offense", "state_target"]],
        on=["state_fips", "offense"],
        how="inner",
        validate="many_to_one",
    )
    total = grouped.groupby(["state_fips", "offense"])["risk_signal_raw"].transform("sum")
    grouped = grouped.loc[total.gt(0.0)].copy()
    total = grouped.groupby(["state_fips", "offense"])["risk_signal_raw"].transform("sum")
    grouped["target_count"] = np.where(
        total.gt(0.0),
        grouped["state_target"] * grouped["risk_signal_raw"] / total.replace(0.0, np.nan),
        0.0,
    )
    return grouped[
        ["state_fips", "offense", "group_kind", "group_id", "target_count"]
    ]


FOOTPRINT_EXACT_UNLOCATED_ROUTES: frozenset[str] = frozenset(
    {OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE, OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT}
)


def _reconcile_unlocated_routes(
    out: pd.DataFrame, rescaled_keys: pd.DataFrame | None
) -> pd.DataFrame:
    """Make the route columns a true split of the bucket they describe.

    The two state rakes are linear in the raw count, so re-running them at route granularity
    used to reproduce the group target exactly. The per-ORI footprint rake that runs after them
    is NOT one of those rakes: it takes each footprint ORI's own ledger mass off the top and
    rescales every other group -- including this one -- by what is left. So the rake-derived
    route totals now describe the SHAPE of the bucket rather than its size.

    The footprint routes are exact amounts and are held fixed; the rake-derived routes are
    scaled to whatever the bucket has left. Where the rake-derived routes are empty and the
    bucket is not, the remainder is unattributed by construction and says so.
    """
    if rescaled_keys is None or rescaled_keys.empty:
        # No footprint rake touched this build, so the two state rakes still reproduce the
        # bucket exactly and the assertion downstream is the real guard it has always been.
        return out
    touched = pd.MultiIndex.from_frame(
        rescaled_keys[["state_fips", "offense"]].drop_duplicates()
    )
    rescaled = pd.Series(
        pd.MultiIndex.from_frame(out[["state_fips", "offense"]]).isin(touched),
        index=out.index,
        dtype=bool,
    )
    if not bool(rescaled.any()):
        return out
    exact_cols = [f"unlocated_count_{route}" for route in sorted(FOOTPRINT_EXACT_UNLOCATED_ROUTES)]
    rake_cols = [
        f"unlocated_count_{route}"
        for route in sorted(UNLOCATED_ROUTES - FOOTPRINT_EXACT_UNLOCATED_ROUTES)
    ]
    exact_sum = out[exact_cols].sum(axis=1)
    rake_sum = out[rake_cols].sum(axis=1)
    remaining = (out["unlocated_count"] - exact_sum).clip(lower=0.0)
    scale = np.where(
        rescaled.to_numpy() & rake_sum.gt(0.0).to_numpy(),
        remaining / rake_sum.where(rake_sum.gt(0.0), 1.0),
        1.0,
    )
    for column in rake_cols:
        out[column] = out[column].to_numpy(dtype=float) * scale
    unattributed = f"unlocated_count_{OVERLAP_ROUTE_UNATTRIBUTED_RESIDUAL}"
    out[unattributed] = np.where(
        rescaled.to_numpy() & ~rake_sum.gt(0.0).to_numpy(),
        remaining.to_numpy(dtype=float),
        out[unattributed].to_numpy(dtype=float),
    )
    return out


def _smoothed_unlocated_route_detail(
    *,
    merged: pd.DataFrame,
    grouped: pd.DataFrame,
    overlap_controls: pd.DataFrame,
    extra_routes: pd.DataFrame | None = None,
    rescaled_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Recompose the unlocated audit under the multi-year overlap partition."""
    keys = ["state_fips", "offense"]
    work = merged.copy()
    work["risk_signal_raw"] = (
        pd.to_numeric(work.get("risk_signal_count"), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(work.get("weight"), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    state_total = (
        work.groupby(keys, as_index=False)["risk_signal_raw"]
        .sum()
        .rename(columns={"risk_signal_raw": "state_signal_total"})
    )
    unlocated = work[work["group_kind"].astype("string").eq(UNLOCATED_GROUP_KIND)]
    routes = (
        unlocated.groupby([*keys, "overlap_resolution_route"], as_index=False)["risk_signal_raw"].sum()
        if not unlocated.empty
        else pd.DataFrame(columns=[*keys, "overlap_resolution_route", "risk_signal_raw"])
    )
    routes = routes.merge(state_total, on=keys, how="left").merge(
        overlap_controls[[*keys, "state_target"]], on=keys, how="left"
    )
    routes["route_target_count"] = np.where(
        pd.to_numeric(routes["state_signal_total"], errors="coerce").fillna(0.0).gt(0.0),
        pd.to_numeric(routes["state_target"], errors="coerce").fillna(0.0)
        * pd.to_numeric(routes["risk_signal_raw"], errors="coerce").fillna(0.0)
        / pd.to_numeric(routes["state_signal_total"], errors="coerce").replace(0.0, np.nan),
        0.0,
    )
    routes = routes[[*keys, "overlap_resolution_route", "route_target_count"]]
    # Same reason as the legacy detail: the per-ORI footprint rake is not one of the two state
    # rakes, so its contribution arrives as an already-final route total.
    if extra_routes is not None and not extra_routes.empty:
        routes = pd.concat(
            [routes, extra_routes[[*keys, "overlap_resolution_route", "route_target_count"]]],
            ignore_index=True,
        )
    wide = (
        routes.pivot_table(
            index=keys,
            columns="overlap_resolution_route",
            values="route_target_count",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()
        if not routes.empty
        else pd.DataFrame(columns=keys)
    )
    out = overlap_controls[[*keys, "state_target"]].rename(
        columns={"state_target": "control_count"}
    )
    out = out.merge(
        grouped[grouped["group_kind"].eq(UNLOCATED_GROUP_KIND)][[*keys, "target_count"]]
        .rename(columns={"target_count": "unlocated_count"}),
        on=keys,
        how="left",
    ).merge(wide, on=keys, how="left")
    out["unlocated_count"] = pd.to_numeric(out["unlocated_count"], errors="coerce").fillna(0.0)
    out["control_count"] = pd.to_numeric(out["control_count"], errors="coerce").fillna(0.0)
    for route in sorted(UNLOCATED_ROUTES):
        column = f"unlocated_count_{route}"
        out[column] = (
            pd.to_numeric(out[route], errors="coerce").fillna(0.0)
            if route in out.columns
            else 0.0
        )
    out = out.drop(columns=[route for route in UNLOCATED_ROUTES if route in out.columns], errors="ignore")
    out = _reconcile_unlocated_routes(out, rescaled_keys)
    route_sum = out[[f"unlocated_count_{route}" for route in sorted(UNLOCATED_ROUTES)]].sum(axis=1)
    drift = float((route_sum - out["unlocated_count"]).abs().max() or 0.0)
    if drift > 1e-6:
        raise ValueError(
            "smoothed unlocated route detail does not reconstruct the bucket; "
            f"max abs delta={drift:.3e}"
        )
    out["jurisdiction_id"] = out["state_fips"].astype("string") + ":" + STATE_OVERLAP_TYPE
    out["jurisdiction_type"] = STATE_OVERLAP_TYPE
    out["unlocated_share_of_control"] = np.where(
        out["control_count"].gt(0.0), out["unlocated_count"] / out["control_count"], 0.0
    )
    return out[list(UNLOCATED_MASS_COLUMNS)].sort_values(keys, kind="mergesort").reset_index(drop=True)


def _assert_no_negative_group_targets(group_targets: pd.DataFrame) -> None:
    """Fail closed on a negative group target.

    The trailing clip(lower=0) on target_count is floating-point hygiene, not a
    correction. A materially negative target means the groups over-drew the state
    control and the delta reconciliation parked the overdraft in the residual group, so
    clipping it away silently RE-ADDS mass the control never had -- the published
    surface then exceeds its own control. That is exactly how the v20 county-remainder
    double count (imputed mass normalized into the observed split and then added again
    as county sub-targets) reached the candidate surface, so it fails closed here
    instead of being absorbed.
    """
    if group_targets.empty:
        return
    values = pd.to_numeric(group_targets["target_count"], errors="coerce").fillna(0.0)
    negative = group_targets[values.lt(-1e-6)]
    if negative.empty:
        return
    raise ValueError(
        f"{len(negative)} county remainder group target(s) are negative, so the group "
        "split over-drew the state control: "
        + str(
            negative[["state_fips", "offense", "group_kind", "group_id", "target_count"]]
            .sort_values("target_count")
            .head(20)
            .to_dict(orient="records")
        )
    )


def _assert_imputed_county_targets_survive(
    group_targets: pd.DataFrame, imputed_county_targets: pd.DataFrame
) -> None:
    """Fail closed if benchmark-imputed county mass did not land on its own county.

    The delta reconciliation above pushes any leftover into the state residual group,
    so a mismatch between the control row and the injected county rows would silently
    smear the imputed mass back over the whole state -- reproducing the very defect
    this lane exists to fix. This asserts the county sub-target survived intact.
    """
    if imputed_county_targets is None or imputed_county_targets.empty:
        return
    expected = imputed_county_targets.groupby(
        ["state_fips", "offense", "group_id"], dropna=False, as_index=False
    )["imputed_target_count"].sum()
    expected = expected[
        pd.to_numeric(expected["imputed_target_count"], errors="coerce").fillna(0.0).gt(0.0)
    ]
    if expected.empty:
        return
    actual = group_targets[group_targets["group_kind"].eq("county_remainder")][
        ["state_fips", "offense", "group_id", "target_count"]
    ]
    merged = expected.merge(actual, on=["state_fips", "offense", "group_id"], how="left")
    merged["target_count"] = pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0)
    # A county may also carry observed mass, so the surviving target is a lower bound.
    shortfall = merged["imputed_target_count"] - merged["target_count"]
    bad = merged[shortfall.gt(1e-6)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} benchmark-imputed county sub-target(s) did not survive into the "
            "county remainder group targets: "
            + str(bad.head(20).to_dict(orient="records"))
        )


def _partition_imputed_county_targets_by_support(
    imputed_county_targets: pd.DataFrame,
    supported_counties: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate county sub-targets that still have remainder ground from displaced ones.

    Exclusive tribal footprints can remove every remainder block group in a county
    while leaving the state's remainder pool alive. A benchmark sub-target created
    before that Stage-2 displacement is then no longer county-localizable. It must not
    be dropped, and it must not be put back on the reservation. It falls back to the
    state residual remainder group, exactly like any other remainder mass without an
    authoritative supported county anchor.
    """
    if imputed_county_targets.empty:
        empty = imputed_county_targets.copy()
        return empty, empty
    work = imputed_county_targets.copy()
    state = work["state_fips"].astype("string").str.zfill(2)
    county = work["county_geoid"].astype("string").str.zfill(5)
    supported = pd.Series(
        [(str(s), str(c)) in supported_counties for s, c in zip(state, county, strict=True)],
        index=work.index,
        dtype=bool,
    )
    localized = work.loc[supported].copy()
    displaced = work.loc[~supported].copy()
    original_mass = float(
        pd.to_numeric(work["imputed_target_count"], errors="coerce").fillna(0.0).sum()
    )
    partitioned_mass = float(
        pd.to_numeric(localized["imputed_target_count"], errors="coerce").fillna(0.0).sum()
        + pd.to_numeric(displaced["imputed_target_count"], errors="coerce").fillna(0.0).sum()
    )
    if abs(original_mass - partitioned_mass) > 1e-8 or len(work) != len(localized) + len(displaced):
        raise ValueError(
            "Benchmark county sub-target support partition lost or duplicated mass: "
            f"original={original_mass:.12f}, partitioned={partitioned_mass:.12f}"
        )
    return localized, displaced


def _assert_displaced_imputed_county_targets_survive(
    group_targets: pd.DataFrame, displaced_imputed_county_targets: pd.DataFrame
) -> None:
    """Fail closed unless explicitly displaced county mass reached the state residual."""
    if displaced_imputed_county_targets.empty:
        return
    expected = (
        displaced_imputed_county_targets.groupby(
            ["state_fips", "offense"], dropna=False, as_index=False
        )["imputed_target_count"]
        .sum()
    )
    actual = (
        group_targets[group_targets["group_kind"].eq("residual_remainder")]
        .groupby(["state_fips", "offense"], dropna=False, as_index=False)[
            "adjustment_target_count"
        ]
        .sum()
    )
    merged = expected.merge(actual, on=["state_fips", "offense"], how="left")
    merged["adjustment_target_count"] = pd.to_numeric(
        merged["adjustment_target_count"], errors="coerce"
    ).fillna(0.0)
    shortfall = (
        pd.to_numeric(merged["imputed_target_count"], errors="coerce").fillna(0.0)
        - merged["adjustment_target_count"]
    )
    bad = merged[shortfall.gt(1e-6)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} fully-displaced benchmark county sub-target(s) did not survive "
            "in the state residual remainder group: "
            + str(bad.head(20).to_dict(orient="records"))
        )


def _build_county_remainder_group_targets(
    *,
    paths: RepoPaths,
    controls: pd.DataFrame,
    year: int,
    bg_crosswalk: pd.DataFrame,
    bg_prior: pd.DataFrame | None = None,
    agency_estimates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    control_cols = ["state_fips", "offense", "adjusted_count_ags_core"]
    for optional_col in (
        "reported_count_preferred",
        "adjustment_total",
        "accounting_count",
    ):
        if optional_col in controls.columns:
            control_cols.append(optional_col)
    remainder_controls = controls[controls["jurisdiction_type"].eq(STATE_REMAINDER_TYPE)][control_cols].copy()
    columns = [
        "state_fips",
        "offense",
        "group_kind",
        "group_id",
        "target_count",
        "reported_count",
        "observed_target_count",
        "adjustment_target_count",
        "observed_raw_count",
        "adjustment_raw_count",
        "county_anchor_evidence_count",
        "county_anchor_supported",
    ]
    if remainder_controls.empty:
        return pd.DataFrame(columns=columns)
    remainder_controls["state_fips"] = remainder_controls["state_fips"].astype("string").str.zfill(2)
    remainder_controls["offense"] = remainder_controls["offense"].astype("string")
    remainder_controls["state_target"] = pd.to_numeric(
        remainder_controls["adjusted_count_ags_core"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    if "accounting_count" in remainder_controls.columns:
        accounting_baseline = pd.to_numeric(
            remainder_controls["accounting_count"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        remainder_controls["state_component_scale"] = np.where(
            accounting_baseline.gt(0.0),
            remainder_controls["state_target"] / accounting_baseline,
            0.0,
        )
    else:
        remainder_controls["state_component_scale"] = 1.0
    if "reported_count_preferred" in remainder_controls.columns:
        remainder_controls["state_reported_target"] = pd.to_numeric(
            remainder_controls["reported_count_preferred"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0) * remainder_controls["state_component_scale"]
    else:
        remainder_controls["state_reported_target"] = 0.0
    remainder_controls["state_reported_target"] = np.minimum(
        remainder_controls["state_reported_target"].to_numpy(dtype=float),
        remainder_controls["state_target"].to_numpy(dtype=float),
    )

    # Class A: the state remainder control row carries the benchmark-imputed mass for
    # its silent counties (that is where the accounting identity is asserted), so the
    # pool this function splits across REPORTING agencies' counties must have it carved
    # out first. Without the carve-out the imputed mass is normalized over the observed
    # county groups AND then added again as county sub-targets below -- the double count
    # that the delta reconciliation turns into a negative residual and the trailing
    # clip(lower=0) silently restores.
    imputed_county_targets = county_remainder_imputed_targets(
        load_benchmark_imputation_units(paths, year=int(year))
    )
    if not imputed_county_targets.empty:
        imputed_county_targets = imputed_county_targets.merge(
            remainder_controls[
                ["state_fips", "offense", "state_component_scale"]
            ],
            on=["state_fips", "offense"],
            how="inner",
            validate="many_to_one",
        )
        imputed_county_targets["imputed_target_count"] = (
            pd.to_numeric(
                imputed_county_targets["imputed_target_count"], errors="coerce"
            ).fillna(0.0).clip(lower=0.0)
            * pd.to_numeric(
                imputed_county_targets["state_component_scale"], errors="coerce"
            ).fillna(0.0).clip(lower=0.0)
        )
    supported_counties = _supported_counties_for_jurisdiction_type(
        bg_crosswalk, STATE_REMAINDER_TYPE
    )
    (
        localized_imputed_county_targets,
        displaced_imputed_county_targets,
    ) = _partition_imputed_county_targets_by_support(
        imputed_county_targets,
        supported_counties,
    )
    imputed_state_totals = (
        imputed_county_targets.groupby(["state_fips", "offense"], dropna=False, as_index=False)[
            "imputed_target_count"
        ]
        .sum()
        .rename(columns={"imputed_target_count": "state_imputed_target"})
        if not imputed_county_targets.empty
        else pd.DataFrame(columns=["state_fips", "offense", "state_imputed_target"])
    )
    remainder_controls = remainder_controls.merge(
        imputed_state_totals, on=["state_fips", "offense"], how="left"
    )
    remainder_controls["state_imputed_target"] = (
        pd.to_numeric(remainder_controls["state_imputed_target"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    remainder_controls["state_imputed_target"] = np.minimum(
        remainder_controls["state_imputed_target"].to_numpy(dtype=float),
        (remainder_controls["state_target"] - remainder_controls["state_reported_target"])
        .clip(lower=0.0)
        .to_numpy(dtype=float),
    )
    remainder_controls["state_adjustment_target"] = (
        remainder_controls["state_target"]
        - remainder_controls["state_reported_target"]
        - remainder_controls["state_imputed_target"]
    ).clip(lower=0.0)

    residual_base = remainder_controls.copy()
    residual_base["group_kind"] = "residual_remainder"
    residual_base["group_id"] = (
        residual_base["state_fips"].astype("string").str.zfill(2) + ":state_nonmunicipal_remainder:residual"
    )
    residual_base["target_count"] = residual_base["state_target"]
    residual_base["reported_count"] = residual_base["state_reported_target"]
    residual_base["observed_target_count"] = residual_base["state_reported_target"]
    residual_base["adjustment_target_count"] = residual_base["state_adjustment_target"]
    residual_base["observed_raw_count"] = 0.0
    residual_base["adjustment_raw_count"] = 0.0
    residual_base["county_anchor_evidence_count"] = 0.0
    residual_base["county_anchor_supported"] = False

    crosswalk = _load_crosswalk(paths).rename(columns={"ori": "ori9"})
    crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    remainder_cw = crosswalk[crosswalk["jurisdiction_id"].astype("string").str.endswith(STATE_REMAINDER_SUFFIX, na=False)].copy()
    if remainder_cw.empty:
        return residual_base[columns].sort_values(["state_fips", "offense"], kind="mergesort").reset_index(drop=True)

    agency_master_full = _load_agency_master(paths)
    master_cols = ["ori9", "state_fips", "county_fips"]
    if "county_fips_source" in agency_master_full.columns:
        master_cols.append("county_fips_source")
    agency_master = agency_master_full[master_cols].copy()
    agency_master["state_fips"] = agency_master["state_fips"].astype("string").str.zfill(2)
    agency_master["county_fips"] = agency_master["county_fips"].astype("string").str.zfill(3)
    if "county_fips_source" not in agency_master.columns:
        agency_master["county_fips_source"] = pd.NA
    agency_master["county_fips_source"] = agency_master["county_fips_source"].astype("string")
    agency_estimates = (
        agency_estimates.copy()
        if agency_estimates is not None
        else _build_agency_allocation_target_estimates(paths=paths, year=int(year))
    )
    if agency_estimates.empty:
        return residual_base[columns].sort_values(["state_fips", "offense"], kind="mergesort").reset_index(drop=True)
    agency_estimates["state_fips"] = agency_estimates["state_fips"].astype("string").str.zfill(2)
    agency_estimates["offense"] = agency_estimates["offense"].astype("string")

    merged = agency_estimates.merge(
        remainder_cw[["ori9", "state_fips", "jurisdiction_id", "weight"]],
        on=["ori9", "state_fips"],
        how="inner",
    ).merge(agency_master, on=["ori9", "state_fips"], how="left")
    if merged.empty:
        return residual_base[columns].sort_values(["state_fips", "offense"], kind="mergesort").reset_index(drop=True)
    merged = merged.merge(
        remainder_controls[["state_fips", "offense", "state_target", "state_reported_target", "state_adjustment_target"]],
        on=["state_fips", "offense"],
        how="inner",
    )
    if merged.empty:
        return residual_base[columns].sort_values(["state_fips", "offense"], kind="mergesort").reset_index(drop=True)
    merged["reported_count_current_supported"] = pd.to_numeric(
        merged["reported_count_current_supported"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    merged["agency_adjustment_count"] = pd.to_numeric(
        merged["agency_adjustment_count"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    merged["county_fips"] = merged["county_fips"].astype("string").str.zfill(3)
    merged["county_geoid"] = merged["state_fips"].astype(str).str.zfill(2) + merged["county_fips"].astype(str).str.zfill(3)
    merged["observed_raw_count"] = merged["reported_count_current_supported"] * pd.to_numeric(
        merged["weight"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    merged["adjustment_raw_count"] = merged["agency_adjustment_count"] * pd.to_numeric(
        merged["weight"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    # County anchoring requires an authoritative placement, not merely a resolvable
    # county name -- see COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES for the evidence.
    # A name-resolved county still flows everywhere else (imputation eligibility,
    # dead/active predicates); it just cannot concentrate an agency's crime into a
    # county remainder it was never placed in.
    county_placement_is_authoritative = (
        merged["county_fips_source"]
        .astype("string")
        .isin(COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES)
        .fillna(False)
    )
    has_supported_county = (
        _valid_county_fips(merged["county_fips"])
        & county_placement_is_authoritative
        & _county_supported_mask(merged, supported_counties)
    )
    evidence = (
        merged.loc[has_supported_county]
        .groupby(["state_fips", "offense", "county_geoid"], dropna=False)["observed_raw_count"]
        .sum()
        .rename("county_anchor_evidence_count")
        .reset_index()
    )
    merged = merged.merge(evidence, on=["state_fips", "offense", "county_geoid"], how="left")
    merged["county_anchor_evidence_count"] = pd.to_numeric(
        merged["county_anchor_evidence_count"], errors="coerce"
    ).fillna(0.0)
    rare_offense_requires_evidence = merged["offense"].astype("string").isin(COUNTY_ANCHOR_MIN_EVIDENCE_OFFENSES)
    evidence_ok = ~rare_offense_requires_evidence | merged["county_anchor_evidence_count"].ge(
        float(COUNTY_ANCHOR_MIN_OBSERVED_OFFENSE_COUNT)
    )
    merged["county_anchor_supported"] = has_supported_county & evidence_ok
    merged["group_kind"] = np.where(merged["county_anchor_supported"], "county_remainder", "residual_remainder")
    merged["group_id"] = np.where(
        merged["county_anchor_supported"],
        merged["state_fips"].astype("string").str.zfill(2)
        + ":state_nonmunicipal_remainder:county:"
        + merged["county_geoid"].astype("string"),
        merged["state_fips"].astype("string").str.zfill(2) + ":state_nonmunicipal_remainder:residual",
    )
    if _controls_are_smoothed(controls):
        risk_signals = _build_agency_risk_signals(
            paths=paths,
            year=int(year),
            fallback_estimates=agency_estimates,
        )
        merged = merged.merge(
            risk_signals[["ori9", "state_fips", "offense", "risk_signal_count"]],
            on=["ori9", "state_fips", "offense"],
            how="left",
            validate="many_to_one",
        )
        merged["risk_signal_count"] = pd.to_numeric(
            merged["risk_signal_count"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)

    observed_group = (
        merged.groupby(["state_fips", "offense", "group_kind", "group_id"], dropna=False)
        .agg(
            observed_raw_count=("observed_raw_count", "sum"),
            county_anchor_evidence_count=("county_anchor_evidence_count", "max"),
            county_anchor_supported=("county_anchor_supported", "max"),
        )
        .reset_index()
    )
    observed_state = (
        observed_group.groupby(["state_fips", "offense"], dropna=False)["observed_raw_count"]
        .sum()
        .rename("observed_state_raw_total")
        .reset_index()
    )
    observed_group = observed_group.merge(
        remainder_controls[["state_fips", "offense", "state_reported_target"]],
        on=["state_fips", "offense"],
        how="inner",
    ).merge(observed_state, on=["state_fips", "offense"], how="left")
    observed_group["observed_target_count"] = np.where(
        pd.to_numeric(observed_group["observed_state_raw_total"], errors="coerce").fillna(0.0).gt(0.0),
        pd.to_numeric(observed_group["state_reported_target"], errors="coerce").fillna(0.0)
        * pd.to_numeric(observed_group["observed_raw_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(observed_group["observed_state_raw_total"], errors="coerce").fillna(1.0),
        0.0,
    )
    missing_observed = remainder_controls.merge(observed_state, on=["state_fips", "offense"], how="left")
    missing_observed = missing_observed[
        pd.to_numeric(missing_observed["observed_state_raw_total"], errors="coerce").fillna(0.0).le(0.0)
        & pd.to_numeric(missing_observed["state_reported_target"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    if not missing_observed.empty:
        missing_observed["group_kind"] = "residual_remainder"
        missing_observed["group_id"] = (
            missing_observed["state_fips"].astype("string").str.zfill(2) + ":state_nonmunicipal_remainder:residual"
        )
        missing_observed["observed_raw_count"] = 0.0
        missing_observed["observed_target_count"] = missing_observed["state_reported_target"]
        missing_observed["county_anchor_evidence_count"] = 0.0
        missing_observed["county_anchor_supported"] = False
        observed_group = pd.concat(
            [
                observed_group[
                    [
                        "state_fips",
                        "offense",
                        "group_kind",
                        "group_id",
                        "observed_raw_count",
                        "observed_target_count",
                        "county_anchor_evidence_count",
                        "county_anchor_supported",
                    ]
                ],
                missing_observed[
                    [
                        "state_fips",
                        "offense",
                        "group_kind",
                        "group_id",
                        "observed_raw_count",
                        "observed_target_count",
                        "county_anchor_evidence_count",
                        "county_anchor_supported",
                    ]
                ],
            ],
            ignore_index=True,
        )
    else:
        observed_group = observed_group[
            [
                "state_fips",
                "offense",
                "group_kind",
                "group_id",
                "observed_raw_count",
                "observed_target_count",
                "county_anchor_evidence_count",
                "county_anchor_supported",
            ]
        ].copy()

    adjustment_group = (
        merged.groupby(["state_fips", "offense", "group_kind", "group_id"], dropna=False)
        .agg(
            adjustment_raw_count=("adjustment_raw_count", "sum"),
            county_anchor_evidence_count=("county_anchor_evidence_count", "max"),
            county_anchor_supported=("county_anchor_supported", "max"),
        )
        .reset_index()
    )
    adjustment_state = (
        adjustment_group.groupby(["state_fips", "offense"], dropna=False)["adjustment_raw_count"]
        .sum()
        .rename("adjustment_state_raw_total")
        .reset_index()
    )
    adjustment_group = adjustment_group.merge(
        remainder_controls[["state_fips", "offense", "state_adjustment_target"]],
        on=["state_fips", "offense"],
        how="inner",
    ).merge(adjustment_state, on=["state_fips", "offense"], how="left")
    adjustment_group["adjustment_target_count"] = np.where(
        pd.to_numeric(adjustment_group["adjustment_state_raw_total"], errors="coerce").fillna(0.0).gt(0.0),
        pd.to_numeric(adjustment_group["state_adjustment_target"], errors="coerce").fillna(0.0)
        * pd.to_numeric(adjustment_group["adjustment_raw_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(adjustment_group["adjustment_state_raw_total"], errors="coerce").fillna(1.0),
        0.0,
    )
    missing_adjustment = remainder_controls.merge(adjustment_state, on=["state_fips", "offense"], how="left")
    missing_adjustment = missing_adjustment[
        pd.to_numeric(missing_adjustment["adjustment_state_raw_total"], errors="coerce").fillna(0.0).le(0.0)
        & pd.to_numeric(missing_adjustment["state_adjustment_target"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    if not missing_adjustment.empty:
        missing_adjustment["group_kind"] = "residual_remainder"
        missing_adjustment["group_id"] = (
            missing_adjustment["state_fips"].astype("string").str.zfill(2) + ":state_nonmunicipal_remainder:residual"
        )
        missing_adjustment["adjustment_raw_count"] = 0.0
        missing_adjustment["adjustment_target_count"] = missing_adjustment["state_adjustment_target"]
        missing_adjustment["county_anchor_evidence_count"] = 0.0
        missing_adjustment["county_anchor_supported"] = False
        adjustment_group = pd.concat(
            [
                adjustment_group[
                    [
                        "state_fips",
                        "offense",
                        "group_kind",
                        "group_id",
                        "adjustment_raw_count",
                        "adjustment_target_count",
                        "county_anchor_evidence_count",
                        "county_anchor_supported",
                    ]
                ],
                missing_adjustment[
                    [
                        "state_fips",
                        "offense",
                        "group_kind",
                        "group_id",
                        "adjustment_raw_count",
                        "adjustment_target_count",
                        "county_anchor_evidence_count",
                        "county_anchor_supported",
                    ]
                ],
            ],
            ignore_index=True,
        )
    else:
        adjustment_group = adjustment_group[
            [
                "state_fips",
                "offense",
                "group_kind",
                "group_id",
                "adjustment_raw_count",
                "adjustment_target_count",
                "county_anchor_evidence_count",
                "county_anchor_supported",
            ]
        ].copy()

    combined = observed_group.merge(
        adjustment_group,
        on=["state_fips", "offense", "group_kind", "group_id"],
        how="outer",
        suffixes=("_observed", "_adjustment"),
    )
    for col in ["observed_raw_count", "observed_target_count", "adjustment_raw_count", "adjustment_target_count"]:
        combined[col] = pd.to_numeric(combined.get(col), errors="coerce").fillna(0.0)
    combined["county_anchor_evidence_count"] = np.maximum(
        pd.to_numeric(combined.get("county_anchor_evidence_count_observed"), errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(combined.get("county_anchor_evidence_count_adjustment"), errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )
    combined["county_anchor_supported"] = (
        combined.get("county_anchor_supported_observed", pd.Series(False, index=combined.index)).fillna(False).astype(bool)
        | combined.get("county_anchor_supported_adjustment", pd.Series(False, index=combined.index)).fillna(False).astype(bool)
    )
    combined["reported_count"] = combined["observed_target_count"]
    combined["target_count"] = combined["observed_target_count"] + combined["adjustment_target_count"]

    # Class A: benchmark-constrained imputation for silent-agency county territory.
    # These counties have no reporting agency at all, so nothing above created a group
    # for them and their remainder territory would take a target of exactly zero. The
    # sub-target was sized in benchmark_imputation.py against the FBI's published state
    # estimate and is already carved out of state_adjustment_target above, so these rows
    # restore exactly the mass the observed split was denied -- the group total still
    # equals the control row by construction.
    if not imputed_county_targets.empty:
        imputed_rows = localized_imputed_county_targets.copy()
        if not displaced_imputed_county_targets.empty:
            displaced_rows = displaced_imputed_county_targets.copy()
            displaced_rows["group_id"] = (
                displaced_rows["state_fips"].astype("string").str.zfill(2)
                + ":state_nonmunicipal_remainder:residual"
            )
            imputed_rows = pd.concat([imputed_rows, displaced_rows], ignore_index=True)
        imputed_rows["state_fips"] = imputed_rows["state_fips"].astype("string").str.zfill(2)
        imputed_rows["offense"] = imputed_rows["offense"].astype("string")
        imputed_rows = imputed_rows[
            pd.to_numeric(imputed_rows["imputed_target_count"], errors="coerce").fillna(0.0).gt(0.0)
        ]
        imputed_rows = imputed_rows.merge(
            remainder_controls[["state_fips", "offense"]].drop_duplicates(),
            on=["state_fips", "offense"],
            how="inner",
        )
        if not imputed_rows.empty:
            target = pd.to_numeric(imputed_rows["imputed_target_count"], errors="coerce").fillna(0.0)
            displaced_group_ids = set(
                displaced_imputed_county_targets.assign(
                    group_id=(
                        displaced_imputed_county_targets["state_fips"]
                        .astype("string")
                        .str.zfill(2)
                        + ":state_nonmunicipal_remainder:residual"
                    )
                )["group_id"].astype("string")
            )
            is_displaced_fallback = imputed_rows["group_id"].astype("string").isin(
                displaced_group_ids
            )
            combined = pd.concat(
                [
                    combined,
                    pd.DataFrame(
                        {
                            "state_fips": imputed_rows["state_fips"].to_numpy(),
                            "offense": imputed_rows["offense"].to_numpy(),
                            "group_kind": np.where(
                                is_displaced_fallback,
                                "residual_remainder",
                                "county_remainder",
                            ),
                            "group_id": imputed_rows["group_id"].astype("string").to_numpy(),
                            "target_count": target.to_numpy(dtype=float),
                            "reported_count": 0.0,
                            "observed_target_count": 0.0,
                            "adjustment_target_count": target.to_numpy(dtype=float),
                            "observed_raw_count": 0.0,
                            "adjustment_raw_count": 0.0,
                            "county_anchor_evidence_count": 0.0,
                            "county_anchor_supported": ~is_displaced_fallback.to_numpy(),
                        }
                    ),
                ],
                ignore_index=True,
            )

    residual_rows = remainder_controls[["state_fips", "offense"]].copy()
    residual_rows["group_kind"] = "residual_remainder"
    residual_rows["group_id"] = (
        residual_rows["state_fips"].astype("string").str.zfill(2) + ":state_nonmunicipal_remainder:residual"
    )
    combined = pd.concat(
        [
            combined,
            residual_rows.assign(
                target_count=0.0,
                reported_count=0.0,
                observed_target_count=0.0,
                adjustment_target_count=0.0,
                observed_raw_count=0.0,
                adjustment_raw_count=0.0,
                county_anchor_evidence_count=0.0,
                county_anchor_supported=False,
            ),
        ],
        ignore_index=True,
    )
    out = (
        combined.groupby(["state_fips", "offense", "group_kind", "group_id"], dropna=False, as_index=False)
        .agg(
            target_count=("target_count", "sum"),
            reported_count=("reported_count", "sum"),
            observed_target_count=("observed_target_count", "sum"),
            adjustment_target_count=("adjustment_target_count", "sum"),
            observed_raw_count=("observed_raw_count", "sum"),
            adjustment_raw_count=("adjustment_raw_count", "sum"),
            county_anchor_evidence_count=("county_anchor_evidence_count", "max"),
            county_anchor_supported=("county_anchor_supported", "max"),
        )
        .merge(remainder_controls[["state_fips", "offense", "state_target"]], on=["state_fips", "offense"], how="inner")
    )
    out = out[
        out["group_kind"].eq("residual_remainder")
        | pd.to_numeric(out["target_count"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    if _controls_are_smoothed(controls):
        out = _smoothed_county_remainder_partition(
            remainder_controls=remainder_controls,
            agency_groups=merged,
            bg_crosswalk=bg_crosswalk,
            bg_prior=bg_prior,
        )
        final_sums = (
            out.groupby(["state_fips", "offense"], dropna=False)["target_count"]
            .sum()
            .reset_index(name="target_sum")
        )
        check = remainder_controls[["state_fips", "offense", "state_target"]].merge(
            final_sums, on=["state_fips", "offense"], how="left"
        )
        max_delta = float(
            (check["target_sum"].fillna(0.0) - check["state_target"]).abs().max() or 0.0
        )
        if max_delta > 1e-6:
            raise ValueError(
                "Smoothed county remainder targets do not partition controls; "
                f"max abs delta={max_delta:.3e}"
            )
        _assert_no_negative_group_targets(out)
        return out[columns]
    sums = (
        out.groupby(["state_fips", "offense"], dropna=False)["target_count"]
        .sum()
        .rename("target_sum")
        .reset_index()
    )
    deltas = remainder_controls[["state_fips", "offense", "state_target"]].merge(
        sums,
        on=["state_fips", "offense"],
        how="left",
    )
    deltas["target_sum"] = pd.to_numeric(deltas["target_sum"], errors="coerce").fillna(0.0)
    deltas["delta"] = deltas["state_target"] - deltas["target_sum"]
    if deltas["delta"].abs().gt(1e-8).any():
        for row in deltas[deltas["delta"].abs().gt(1e-8)].itertuples(index=False):
            mask = (
                out["state_fips"].eq(row.state_fips)
                & out["offense"].eq(row.offense)
                & out["group_kind"].eq("residual_remainder")
            )
            out.loc[mask, "target_count"] = pd.to_numeric(out.loc[mask, "target_count"], errors="coerce").fillna(0.0) + float(row.delta)
            out.loc[mask, "adjustment_target_count"] = (
                pd.to_numeric(out.loc[mask, "adjustment_target_count"], errors="coerce").fillna(0.0) + float(row.delta)
            )
    final_sums = out.groupby(["state_fips", "offense"], dropna=False)["target_count"].sum().reset_index(name="target_sum")
    check = remainder_controls[["state_fips", "offense", "state_target"]].merge(final_sums, on=["state_fips", "offense"], how="left")
    max_delta = float((check["target_sum"].fillna(0.0) - check["state_target"]).abs().max() or 0.0)
    if max_delta > 1e-6:
        raise ValueError(f"County remainder targets do not partition controls; max abs delta={max_delta:.3e}")
    _assert_no_negative_group_targets(out)
    _assert_imputed_county_targets_survive(out, localized_imputed_county_targets)
    _assert_displaced_imputed_county_targets_survive(
        out, displaced_imputed_county_targets
    )
    out["target_count"] = pd.to_numeric(out["target_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["reported_count"] = pd.to_numeric(out["reported_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return out[columns].sort_values(["state_fips", "offense", "group_kind", "group_id"], kind="mergesort").reset_index(drop=True)


def _combine_geocode_quality_tiers(values: pd.Series) -> str:
    tiers: set[str] = set()
    for value in values:
        for part in str(value).split("+"):
            part = part.strip()
            if part:
                tiers.add(part)
    return "+".join(sorted(tiers)) or "unknown"


def city_incident_share_surface_path(
    paths: RepoPaths, *, feed_inputs_dir: Path | None = None
) -> Path:
    """The city incident share surface this build reads and writes."""
    if feed_inputs_dir is not None:
        return Path(feed_inputs_dir) / "city_incident_share_surface.parquet"
    return paths.state_dir / "modeling" / "city_incident_share_surface.parquet"


def city_incident_reconciliation_dir(
    paths: RepoPaths, *, feed_inputs_dir: Path | None = None
) -> Path:
    if feed_inputs_dir is not None:
        return Path(feed_inputs_dir) / "city_reconciliation"
    return paths.review_analysis_dir / "city_reconciliation"


def city_incident_reconciliation_path(
    paths: RepoPaths, *, year: int, feed_inputs_dir: Path | None = None
) -> Path:
    name = f"city_incident_reconciliation_{int(year)}.parquet"
    if feed_inputs_dir is not None:
        return Path(feed_inputs_dir) / name
    return paths.state_dir / "modeling" / name


def _load_city_incident_share_surface(
    paths: RepoPaths,
    *,
    year: int,
    path: Path | None = None,
    exclude_validation_case_types: tuple[str, ...] = (),
    exclude_feed_city_keys: tuple[str, ...] = (),
) -> pd.DataFrame:
    resolved_path = path or city_incident_share_surface_path(paths)
    if not resolved_path.exists():
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
    df = pd.read_parquet(resolved_path).copy()
    df = filter_excluded_city_keys(
        df,
        exclude_city_keys=tuple(exclude_feed_city_keys),
    )
    if exclude_validation_case_types and "validation_case_type" in df.columns:
        excluded = {str(value) for value in exclude_validation_case_types}
        df = df[~df["validation_case_type"].astype(str).isin(excluded)].copy()
    if df.empty:
        return df
    df["state_fips"] = df["state_fips"].astype("string").str.zfill(2)
    df["jurisdiction_id"] = df["jurisdiction_id"].astype("string")
    df["block_group_geoid"] = df["block_group_geoid"].astype("string").str.zfill(12)
    df["offense"] = df["offense"].astype("string")
    df["city_name"] = df["city_name"].astype("string")
    df["incident_count"] = pd.to_numeric(df["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "geocode_quality_tier" not in df.columns:
        df["geocode_quality_tier"] = "unknown"
    df["geocode_quality_tier"] = df["geocode_quality_tier"].astype("string").fillna("unknown")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].notna() & df["year"].le(int(year))].copy()
    else:
        df["year"] = int(year)
    if df.empty:
        return df

    df = df.drop_duplicates().copy()

    city_year_cols = ["city_name", "jurisdiction_id", "state_fips", "year"]
    city_year_totals = (
        df.groupby(city_year_cols, dropna=False)["incident_count"]
        .sum()
        .rename("city_year_incident_total")
        .reset_index()
    )
    city_cols = ["city_name", "jurisdiction_id", "state_fips"]
    city_year_totals["city_year_median_positive_total"] = city_year_totals.groupby(
        city_cols,
        dropna=False,
    )["city_year_incident_total"].transform(lambda values: values[values.gt(0.0)].median())
    city_year_totals["city_year_positive_count"] = city_year_totals.groupby(
        city_cols,
        dropna=False,
    )["city_year_incident_total"].transform(lambda values: int(values.gt(0.0).sum()))
    enough_history = city_year_totals["city_year_positive_count"].ge(
        CITY_INCIDENT_PARTIAL_YEAR_MIN_COMPARISON_YEARS
    )
    partial_year = (
        enough_history
        & city_year_totals["city_year_median_positive_total"].gt(0.0)
        & city_year_totals["city_year_incident_total"].lt(
            city_year_totals["city_year_median_positive_total"] * CITY_INCIDENT_PARTIAL_YEAR_MIN_RATIO
        )
    )
    if bool(partial_year.any()):
        excluded_years = city_year_totals.loc[partial_year, city_year_cols + ["city_year_incident_total"]]
        warnings.warn(
            "Excluding undercovered city incident share years before pooling: "
            f"{excluded_years.to_dict(orient='records')}",
            RuntimeWarning,
            stacklevel=2,
        )
        keep_years = city_year_totals.loc[~partial_year, city_year_cols]
        df = df.merge(keep_years, on=city_year_cols, how="inner")
    if df.empty:
        return df

    detail_cols = ["city_name", "jurisdiction_id", "state_fips", "offense", "block_group_geoid"]
    year_counts = (
        df.groupby([*detail_cols, "year"], dropna=False, as_index=False)
        .agg(
            incident_count=("incident_count", "sum"),
            geocode_quality_tier=("geocode_quality_tier", _combine_geocode_quality_tiers),
        )
    )
    incident_age = (int(year) - pd.to_numeric(year_counts["year"], errors="raise")).clip(lower=0.0)
    year_counts["incident_temporal_weight"] = np.where(
        year_counts["offense"].astype(str).eq("murder"),
        np.power(0.5, incident_age / float(MURDER_INCIDENT_HALF_LIFE_YEARS)),
        1.0,
    )
    year_counts["weighted_incident_count"] = (
        pd.to_numeric(year_counts["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        * year_counts["incident_temporal_weight"]
    )
    pooled = (
        year_counts.groupby(detail_cols, dropna=False, as_index=False)
        .agg(
            incident_count=("weighted_incident_count", "sum"),
            pooled_raw_incident_count=("incident_count", "sum"),
            pooled_source_year_count=("year", "nunique"),
            pooled_source_year_min=("year", "min"),
            pooled_source_year_max=("year", "max"),
            geocode_quality_tier=("geocode_quality_tier", _combine_geocode_quality_tiers),
        )
    )
    pooled["city_total"] = pooled.groupby(
        ["city_name", "jurisdiction_id", "state_fips", "offense"],
        dropna=False,
    )["incident_count"].transform("sum")
    pooled = pooled[pd.to_numeric(pooled["city_total"], errors="coerce").fillna(0.0).gt(0.0)].copy()
    pooled["share_within_city"] = (
        pd.to_numeric(pooled["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(pooled["city_total"], errors="coerce").fillna(np.nan)
    )
    pooled["share_within_city"] = pooled["share_within_city"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pooled["year"] = int(year)
    pooled["city_share_pooling_method"] = np.where(
        pooled["offense"].astype(str).eq("murder"),
        f"incident_count_exponential_half_life_{MURDER_INCIDENT_HALF_LIFE_YEARS:g}_years",
        "incident_count_sum_all_reliable_years_unweighted",
    )

    share_sums = pooled.groupby(
        ["city_name", "jurisdiction_id", "state_fips", "offense"],
        dropna=False,
    )["share_within_city"].sum()
    bad_sums = share_sums[(share_sums - 1.0).abs().gt(CITY_INCIDENT_SHARE_SUM_TOLERANCE)]
    if not bad_sums.empty:
        raise ValueError(
            "Pooled city incident share surface has non-unit share sums: "
            f"{bad_sums.reset_index(name='share_sum').to_dict(orient='records')[:20]}"
        )
    dupes = pooled.loc[pooled.duplicated(detail_cols, keep=False), detail_cols]
    if not dupes.empty:
        raise ValueError(
            "Pooled city incident share surface contains duplicate city/offense/block-group rows: "
            f"{dupes.drop_duplicates().to_dict(orient='records')[:20]}"
        )
    return pooled.drop(columns=["city_total"]).sort_values(
        ["city_name", "jurisdiction_id", "state_fips", "offense", "block_group_geoid"],
        kind="mergesort",
    ).reset_index(drop=True)


def _empty_city_posterior_quality_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *CITY_POSTERIOR_GROUP_COLS,
            "city_posterior_quality_city_name",
            "city_posterior_quality_year",
            "city_posterior_feed_count_for_quality",
            "city_posterior_published_count_for_diagnostic",
            "city_posterior_match_rate",
            "city_posterior_mapped_count",
            "city_posterior_matched_count",
            "city_posterior_published_comparison_quality",
        ]
    )


def _load_city_posterior_quality(paths: RepoPaths, *, year: int) -> pd.DataFrame:
    path = paths.state_dir / "modeling" / f"city_incident_reconciliation_{int(year)}.parquet"
    if not path.exists():
        return _empty_city_posterior_quality_frame()

    df = pd.read_parquet(path).copy()
    required = set(CITY_POSTERIOR_GROUP_COLS + ["year", "final_share_count"])
    if df.empty or not required.issubset(df.columns):
        return _empty_city_posterior_quality_frame()

    df["jurisdiction_id"] = df["jurisdiction_id"].astype("string")
    df["state_fips"] = df["state_fips"].astype("string").str.zfill(2)
    df["offense"] = df["offense"].astype("string")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna() & df["year"].le(int(year))].copy()
    if df.empty:
        return _empty_city_posterior_quality_frame()

    for col in [
        "final_share_count",
        "published_count",
        "mapped_offense_count",
        "matched_offense_count",
        "geocoded_offense_count",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "city_name" not in df.columns:
        df["city_name"] = pd.NA
    if "published_comparison_quality" not in df.columns:
        df["published_comparison_quality"] = pd.NA

    mapped = df["mapped_offense_count"].replace(0.0, np.nan)
    geocoded = df["geocoded_offense_count"].replace(0.0, np.nan)
    match_rate = df["matched_offense_count"] / mapped
    match_rate = match_rate.fillna(df["matched_offense_count"] / geocoded)
    df["city_posterior_match_rate"] = match_rate.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.0)

    df = df.sort_values([*CITY_POSTERIOR_GROUP_COLS, "year"], kind="mergesort")
    latest = df.groupby(CITY_POSTERIOR_GROUP_COLS, dropna=False, as_index=False).tail(1).copy()
    latest = latest.rename(
        columns={
            "city_name": "city_posterior_quality_city_name",
            "year": "city_posterior_quality_year",
            "final_share_count": "city_posterior_feed_count_for_quality",
            "published_count": "city_posterior_published_count_for_diagnostic",
            "mapped_offense_count": "city_posterior_mapped_count",
            "matched_offense_count": "city_posterior_matched_count",
            "published_comparison_quality": "city_posterior_published_comparison_quality",
        }
    )
    out_cols = list(_empty_city_posterior_quality_frame().columns)
    return latest[out_cols].reset_index(drop=True)


def _finite_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _tract_incident_posterior(
    frame: pd.DataFrame,
    *,
    direct_count: pd.Series,
    prior_share: pd.Series,
    prior_incidents: float,
) -> pd.DataFrame:
    """Update a jurisdiction prior at tract support without learning a BG point pattern.

    Incident counts are first summed to tracts. Both the tract posterior and the tract's
    direct-only diagnostic share are then spread inside the tract using the prior. Murder
    therefore has one geographic estimand whether or not a city incident feed exists.
    """
    if frame.empty:
        return pd.DataFrame(
            index=frame.index,
            columns=["posterior_share", "direct_share", "alpha", "prior_fraction"],
        )
    strength = float(prior_incidents)
    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError(f"tract posterior prior incidents must be finite and positive, got {prior_incidents!r}")

    counts = pd.Series(direct_count, index=frame.index, dtype=float).fillna(0.0).clip(lower=0.0)
    prior = pd.Series(prior_share, index=frame.index, dtype=float).fillna(0.0).clip(lower=0.0)
    group_keys = [frame[col] for col in CITY_POSTERIOR_GROUP_COLS]
    tract_id = frame["tract_id"].astype("string").str.zfill(11)
    tract_keys = [*group_keys, tract_id]

    direct_total = counts.groupby(group_keys, dropna=False).transform("sum")
    tract_count = counts.groupby(tract_keys, dropna=False).transform("sum")
    tract_prior = prior.groupby(tract_keys, dropna=False).transform("sum")
    rows_in_tract = prior.groupby(tract_keys, dropna=False).transform("size").clip(lower=1.0)
    within_tract = np.where(tract_prior.gt(0.0), prior / tract_prior.replace(0.0, np.nan), 1.0 / rows_in_tract)

    tract_posterior = (tract_count + strength * tract_prior) / (direct_total + strength)
    posterior_share = tract_posterior * within_tract
    direct_tract_share = np.where(
        direct_total.gt(0.0),
        tract_count / direct_total.replace(0.0, np.nan),
        0.0,
    )
    direct_share = direct_tract_share * within_tract
    prior_fraction = strength / (direct_total + strength)

    result = pd.DataFrame(
        {
            "posterior_share": posterior_share,
            "direct_share": direct_share,
            "alpha": strength,
            "prior_fraction": prior_fraction,
        },
        index=frame.index,
    )
    sums = result["posterior_share"].groupby(group_keys, dropna=False).sum()
    if sums.empty or not np.allclose(sums.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-9):
        bad = sums[~np.isclose(sums, 1.0, rtol=0.0, atol=1e-9)].head(10).to_dict()
        raise ValueError(f"tract incident posterior shares do not conserve within group: {bad}")
    return result


def _series_distribution(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {key: None for key in ["min", "p05", "p25", "median", "p75", "p95", "max", "mean"]}
    quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": _finite_or_none(values.min()),
        "p05": _finite_or_none(quantiles.loc[0.05]),
        "p25": _finite_or_none(quantiles.loc[0.25]),
        "median": _finite_or_none(quantiles.loc[0.5]),
        "p75": _finite_or_none(quantiles.loc[0.75]),
        "p95": _finite_or_none(quantiles.loc[0.95]),
        "max": _finite_or_none(values.max()),
        "mean": _finite_or_none(values.mean()),
    }


def _city_posterior_example_row(df: pd.DataFrame, mask: pd.Series, sort_col: str) -> dict[str, object] | None:
    candidates = df.loc[mask].copy()
    if candidates.empty or sort_col not in candidates.columns:
        return None
    candidates[sort_col] = pd.to_numeric(candidates[sort_col], errors="coerce")
    candidates = candidates.sort_values(sort_col, ascending=False, kind="mergesort")
    row = candidates.iloc[0]
    keys = [
        "city_name",
        "jurisdiction_id",
        "state_fips",
        "offense",
        "feed_quality_count",
        "control_total",
        "feed_control_fraction",
        "missing_fraction",
        "match_rate",
        "volume_prior_fraction",
        "alpha",
        "posterior_prior_fraction",
        "tvd_posterior_vs_direct",
        "posterior_mass_in_zero_feed_bgs",
        "zero_feed_prior_positive_bg_count",
    ]
    out: dict[str, object] = {}
    for key in keys:
        value = row.get(key)
        if isinstance(value, str):
            out[key] = value
        elif pd.isna(value):
            out[key] = None
        else:
            parsed = _finite_or_none(value)
            out[key] = parsed if parsed is not None else str(value)
    return out


def _summarize_city_posterior_diagnostics(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "active_groups": 0,
            "material_tvd_threshold": float(CITY_POSTERIOR_MATERIAL_TVD_THRESHOLD),
            "alpha_distribution": _series_distribution(pd.Series(dtype=float)),
            "groups_materially_changed_vs_direct": 0,
        }

    tvd = pd.to_numeric(df.get("tvd_posterior_vs_direct"), errors="coerce").fillna(0.0)
    missing = pd.to_numeric(df.get("missing_fraction"), errors="coerce").fillna(0.0)
    volume_prior = pd.to_numeric(df.get("volume_prior_fraction"), errors="coerce").fillna(0.0)
    zero_feed_mass = pd.to_numeric(df.get("posterior_mass_in_zero_feed_bgs"), errors="coerce").fillna(0.0)
    material = tvd.gt(float(CITY_POSTERIOR_MATERIAL_TVD_THRESHOLD))
    under_counting = missing.gt(float(CITY_POSTERIOR_RECONCILIATION_TOLERANCE))
    sparse_or_low_volume = volume_prior.gt(0.25)
    clean_dense = (
        missing.le(float(CITY_POSTERIOR_RECONCILIATION_TOLERANCE))
        & volume_prior.le(0.05)
        & pd.to_numeric(df.get("match_rate"), errors="coerce").fillna(1.0).ge(0.95)
    )
    return {
        "active_groups": int(len(df)),
        "material_tvd_threshold": float(CITY_POSTERIOR_MATERIAL_TVD_THRESHOLD),
        "alpha_distribution": _series_distribution(df["alpha"]),
        "posterior_prior_fraction_distribution": _series_distribution(df["posterior_prior_fraction"]),
        "groups_materially_changed_vs_direct": int(material.sum()),
        "under_counting_groups": int(under_counting.sum()),
        "under_counting_groups_materially_changed": int((under_counting & material).sum()),
        "sparse_or_low_volume_groups": int(sparse_or_low_volume.sum()),
        "sparse_or_low_volume_groups_materially_changed": int((sparse_or_low_volume & material).sum()),
        "groups_with_zero_feed_prior_mass": int(zero_feed_mass.gt(0.0).sum()),
        "clean_dense_groups": int(clean_dense.sum()),
        "clean_dense_tvd_distribution": _series_distribution(tvd.loc[clean_dense]),
        "example_under_counting_group": _city_posterior_example_row(
            df,
            under_counting,
            "posterior_prior_fraction",
        ),
    }


def _build_bg_direct_incident_support(
    *,
    paths: RepoPaths,
    bg_crosswalk: pd.DataFrame,
    year: int,
    exclude_feed_city_keys: tuple[str, ...] = (),
    feed_inputs_dir: Path | None = None,
) -> pd.DataFrame:
    base_columns = ["state_fips", "block_group_geoid"]
    output_columns = list(base_columns)
    for offense in OFFENSES_7:
        output_columns.extend(
            [
                f"direct_incident_support_flag_{offense}",
                f"direct_incident_support_count_{offense}",
                f"direct_incident_support_years_{offense}",
                f"direct_incident_support_year_min_{offense}",
                f"direct_incident_support_year_max_{offense}",
                f"numerator_support_source_{offense}",
            ]
        )

    universe = bg_crosswalk[base_columns].drop_duplicates().copy()
    universe["state_fips"] = universe["state_fips"].astype("string").str.zfill(2)
    universe["block_group_geoid"] = universe["block_group_geoid"].astype("string").str.zfill(12)
    if universe.empty:
        return pd.DataFrame(columns=output_columns)

    incident = _load_city_incident_share_surface(
        paths,
        year=year,
        path=city_incident_share_surface_path(paths, feed_inputs_dir=feed_inputs_dir),
        exclude_feed_city_keys=exclude_feed_city_keys,
    )
    if incident.empty:
        out = universe.copy()
        for offense in OFFENSES_7:
            out[f"direct_incident_support_flag_{offense}"] = False
            out[f"direct_incident_support_count_{offense}"] = 0.0
            out[f"direct_incident_support_years_{offense}"] = 0.0
            out[f"direct_incident_support_year_min_{offense}"] = np.nan
            out[f"direct_incident_support_year_max_{offense}"] = np.nan
            out[f"numerator_support_source_{offense}"] = "model_only"
        return out[output_columns].copy()

    incident = incident.rename(columns={"block_group_geoid": "block_group_geoid"}).copy()
    incident["state_fips"] = incident["state_fips"].astype("string").str.zfill(2)
    incident["jurisdiction_id"] = incident["jurisdiction_id"].astype("string")
    incident["block_group_geoid"] = incident["block_group_geoid"].astype("string").str.zfill(12)
    incident["offense"] = incident["offense"].astype("string")
    incident["incident_count"] = pd.to_numeric(incident["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "pooled_source_year_count" not in incident.columns:
        incident["pooled_source_year_count"] = 1.0
    if "pooled_source_year_min" not in incident.columns:
        incident["pooled_source_year_min"] = int(year)
    if "pooled_source_year_max" not in incident.columns:
        incident["pooled_source_year_max"] = int(year)
    incident["pooled_source_year_count"] = (
        pd.to_numeric(incident["pooled_source_year_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    incident["pooled_source_year_min"] = pd.to_numeric(incident["pooled_source_year_min"], errors="coerce")
    incident["pooled_source_year_max"] = pd.to_numeric(incident["pooled_source_year_max"], errors="coerce")

    active = (
        incident.groupby(["jurisdiction_id", "state_fips", "offense"], dropna=False)
        .agg(
            direct_incident_support_years=("pooled_source_year_count", "max"),
            direct_incident_support_year_min=("pooled_source_year_min", "min"),
            direct_incident_support_year_max=("pooled_source_year_max", "max"),
        )
        .reset_index()
    )
    coverage = bg_crosswalk[
        ["state_fips", "block_group_geoid", "jurisdiction_id", "allocation_share"]
    ].copy()
    coverage["state_fips"] = coverage["state_fips"].astype("string").str.zfill(2)
    coverage["block_group_geoid"] = coverage["block_group_geoid"].astype("string").str.zfill(12)
    coverage["jurisdiction_id"] = coverage["jurisdiction_id"].astype("string")
    coverage["allocation_share"] = pd.to_numeric(
        coverage["allocation_share"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    coverage = coverage[coverage["allocation_share"].gt(0.0)].merge(
        active,
        on=["jurisdiction_id", "state_fips"],
        how="inner",
    )

    observed = (
        incident.groupby(["state_fips", "block_group_geoid", "offense"], dropna=False)
        .agg(direct_incident_support_count=("incident_count", "sum"))
        .reset_index()
    )
    support_long = (
        coverage.groupby(["state_fips", "block_group_geoid", "offense"], dropna=False)
        .agg(
            direct_incident_support_flag=("jurisdiction_id", "nunique"),
            direct_incident_support_years=("direct_incident_support_years", "max"),
            direct_incident_support_year_min=("direct_incident_support_year_min", "min"),
            direct_incident_support_year_max=("direct_incident_support_year_max", "max"),
        )
        .reset_index()
    )
    support_long["direct_incident_support_flag"] = support_long["direct_incident_support_flag"].gt(0)
    support_long = support_long.merge(
        observed,
        on=["state_fips", "block_group_geoid", "offense"],
        how="left",
    )
    support_long["direct_incident_support_count"] = pd.to_numeric(
        support_long["direct_incident_support_count"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    out = universe.copy()
    for offense in OFFENSES_7:
        one = support_long[support_long["offense"].astype("string").eq(offense)].drop(columns="offense")
        rename = {
            "direct_incident_support_flag": f"direct_incident_support_flag_{offense}",
            "direct_incident_support_count": f"direct_incident_support_count_{offense}",
            "direct_incident_support_years": f"direct_incident_support_years_{offense}",
            "direct_incident_support_year_min": f"direct_incident_support_year_min_{offense}",
            "direct_incident_support_year_max": f"direct_incident_support_year_max_{offense}",
        }
        out = out.merge(one.rename(columns=rename), on=base_columns, how="left")
        flag_col = f"direct_incident_support_flag_{offense}"
        count_col = f"direct_incident_support_count_{offense}"
        years_col = f"direct_incident_support_years_{offense}"
        out[flag_col] = out[flag_col].astype("boolean").fillna(False).astype(bool)
        out[count_col] = pd.to_numeric(out[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        out[years_col] = pd.to_numeric(out[years_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        out[f"direct_incident_support_year_min_{offense}"] = pd.to_numeric(
            out[f"direct_incident_support_year_min_{offense}"],
            errors="coerce",
        )
        out[f"direct_incident_support_year_max_{offense}"] = pd.to_numeric(
            out[f"direct_incident_support_year_max_{offense}"],
            errors="coerce",
        )
        out[f"numerator_support_source_{offense}"] = np.where(out[flag_col], "direct_city_incident", "model_only")

    return out[output_columns].copy()


OVERLAP_FOOTPRINT_OVERRIDE_COLUMNS = (
    "ori",
    "final_overlap_treatment",
    "overlap_subtype_final",
    "footprint_type",
    "target_state_fips",
    "target_county_fips",
    "target_place_fips",
    "target_jurisdiction_id",
    "displaces_county_remainder",
    "geometry_source_type",
    "geometry_source_ref",
    "confidence",
    "source_note",
    "reviewer_note",
)
# The treatments the classification cascade in `_build_overlap_group_targets` can act on.
# Previously unvalidated (Stage 2 contract SURPRISE): a typo silently fell through to the
# statewide default, which is the same failure mode as the fail-open below.
VALID_OVERLAP_TREATMENTS: frozenset[str] = frozenset(
    {
        "localize_to_custom_footprint",
        "localize_to_place",
        "localize_to_county",
        "keep_statewide_overlap",
        "absorb_into_primary_jurisdiction",
        "exclude_or_hold",
    }
)


def _load_overlap_footprint_overrides(paths: RepoPaths) -> pd.DataFrame:
    path = paths.repo_root / "configs" / "overlap_footprint_overrides.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(OVERLAP_FOOTPRINT_OVERRIDE_COLUMNS))
    overrides = pd.read_csv(path).copy()
    missing = set(OVERLAP_FOOTPRINT_OVERRIDE_COLUMNS) - set(overrides.columns)
    if missing:
        raise ValueError(f"Overlap footprint overrides missing columns: {sorted(missing)}")
    overrides["ori9"] = overrides["ori"].astype("string")
    dupes = overrides.loc[overrides.duplicated("ori9", keep=False), ["ori9"]]
    if not dupes.empty:
        raise ValueError(f"Duplicate overlap overrides by ori: {dupes.to_dict(orient='records')}")
    treatment = overrides["final_overlap_treatment"].astype("string").str.strip()
    bad_treatment = ~treatment.isin(VALID_OVERLAP_TREATMENTS)
    if bool(bad_treatment.any()):
        raise ValueError(
            "Overlap footprint overrides carry unknown final_overlap_treatment values "
            f"(valid: {sorted(VALID_OVERLAP_TREATMENTS)}): "
            f"{overrides.loc[bad_treatment, ['ori9', 'final_overlap_treatment']].to_dict(orient='records')}"
        )
    overrides["final_overlap_treatment"] = treatment
    overrides["displaces_county_remainder"] = _parse_registry_flag(
        overrides["displaces_county_remainder"]
    )
    displacing_non_custom = overrides["displaces_county_remainder"] & ~treatment.eq(
        "localize_to_custom_footprint"
    )
    if bool(displacing_non_custom.any()):
        raise ValueError(
            "displaces_county_remainder is only defined for localize_to_custom_footprint "
            "rows (an exclusive footprint has to name the block groups it takes over): "
            f"{overrides.loc[displacing_non_custom, ['ori9', 'final_overlap_treatment']].to_dict(orient='records')}"
        )
    return overrides


def _parse_registry_flag(series: pd.Series) -> pd.Series:
    """Parse a registry boolean column where blank means False."""
    text = series.astype("string").str.strip().str.lower()
    truthy = text.isin(["true", "t", "yes", "y", "1"])
    falsy = text.isin(["false", "f", "no", "n", "0"]) | text.isna() | text.eq("")
    unknown = ~(truthy | falsy)
    if bool(unknown.any()):
        raise ValueError(
            f"Unparseable registry boolean values: {sorted(set(text[unknown].dropna()))}"
        )
    return truthy.fillna(False).astype(bool)


def _assert_custom_footprint_overrides_have_rows(
    overrides: pd.DataFrame,
    custom_footprints: pd.DataFrame,
) -> None:
    """FAIL CLOSED on `localize_to_custom_footprint` with no footprint rows.

    Rung 5 of the cascade used to convert such an override into `keep_statewide_overlap`,
    silently: the registry said "put this agency on its own footprint" and the allocator
    spread it over the entire state. Measured 2026-07-29 (Stage 2 screen S2-6a): 6 ORIs /
    5,776 counts in that state -- NJ Transit Police over all of New Jersey, WMATA over all
    of DC, Port Authority NY&NJ, UC Berkeley PD and UCLA PD over all of California.

    Routing an agency statewide is a legitimate outcome; it just has to be *declared*
    (`keep_statewide_overlap` with a reviewer_note), never arrived at by omission.
    """
    if overrides.empty:
        return
    declared = overrides.loc[
        overrides["final_overlap_treatment"].astype("string").eq("localize_to_custom_footprint"),
        "ori9",
    ].astype("string")
    if declared.empty:
        return
    have_rows = (
        set(custom_footprints["ori9"].astype("string"))
        if not custom_footprints.empty
        else set()
    )
    missing = sorted(set(declared) - have_rows)
    if missing:
        raise ValueError(
            "configs/overlap_footprint_overrides.csv declares localize_to_custom_footprint "
            "for ORIs with no rows in configs/overlap_custom_footprints.csv, which would "
            "silently spread them over the whole state. Supply footprint rows or change the "
            f"treatment to keep_statewide_overlap with a reviewer_note: {missing}"
        )


OVERLAP_CUSTOM_FOOTPRINT_OUT_COLUMNS = (
    "ori9",
    "state_fips",
    "bg_id",
    "weight_share",
    "bg_population_coverage_share",
    "weight_share_basis",
    "geometry_source_type",
    "geometry_source_ref",
    "footprint_note",
    "geometry_contributor_ori",
    "service_scope_id",
    "canonical_target_ori",
    "source_state_fips",
    "allocation_scope",
    "bg_service_population_coverage_share",
    "bg_land_area_coverage_share",
    "coverage_basis",
    "bg_responsibility_population_coverage_share",
    "responsibility_fraction_basis",
)

PRIMARY_RESPONSE_POLICY_CARVEOUT_PRECEDENCE = (
    "displace_county_remainder_within_reviewed_carveout"
)
PRIMARY_SERVICE_RESPONSE_POLICY_COLUMNS = (
    "service_scope_id",
    "primary_response_policy",
    "official_source_ref",
    "evidence_artifact",
    "evidence_sha256",
)


def _load_primary_service_response_policies(paths: RepoPaths) -> pd.DataFrame:
    """Load allocation-only primary-response precedence decisions."""
    path = paths.repo_root / "configs" / "primary_service_response_policies.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(PRIMARY_SERVICE_RESPONSE_POLICY_COLUMNS))
    policies = pd.read_csv(path, dtype="string")
    missing = set(PRIMARY_SERVICE_RESPONSE_POLICY_COLUMNS) - set(policies.columns)
    if missing:
        raise ValueError(f"Primary service response policies missing columns: {sorted(missing)}")
    policies = policies[list(PRIMARY_SERVICE_RESPONSE_POLICY_COLUMNS)].copy()
    for column in PRIMARY_SERVICE_RESPONSE_POLICY_COLUMNS:
        policies[column] = policies[column].fillna("").str.strip()
    if bool(policies.duplicated("service_scope_id", keep=False).any()):
        raise ValueError("Primary service response policies duplicate a service_scope_id")
    bad = (
        policies["service_scope_id"].eq("")
        | ~policies["primary_response_policy"].eq(
            PRIMARY_RESPONSE_POLICY_CARVEOUT_PRECEDENCE
        )
        | policies["official_source_ref"].eq("")
        | policies["evidence_artifact"].eq("")
        | policies["evidence_sha256"].eq("")
    )
    if bool(bad.any()):
        raise ValueError(
            "Primary service response policy rows require a service scope, the reviewed "
            "carve-out precedence policy, and an official source"
        )
    return policies


# The declared allocation basis for each custom-footprint row. The distinction that matters is
# whether the input already contains an activity/exposure apportionment or instead supplies a
# within-BG responsibility fraction to combine with the predicted-count prior:
#
#   resident_population -- allocation uses the offense-specific predicted-count prior times
#       the ORI's within-BG resident responsibility fraction. `weight_share` remains the
#       documented covered-population fallback when the entire pool has no model signal.
#   activity_or_area -- the share is ALREADY an activity or exposure measure (station
#       boardings, annual passengers, LandScan daytime population, per-station equal share)
#       or a deliberate area apportionment onto parcels with no residents (airfields, port
#       property). Multiplying these by `bg_weight` would apply an activity term twice.
#   service_area_prior -- service-wide rows use the offense-specific predicted-count prior
#       multiplied by the service's own within-BG land-area coverage. `weight_share` remains a
#       normalized compatibility field and is not an allocation input for this basis.
CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT = "resident_population"
CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_ACTIVITY = "activity_or_area"
CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_SERVICE_AREA_PRIOR = "service_area_prior"
VALID_CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASES: frozenset[str] = frozenset(
    {
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT,
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_ACTIVITY,
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_SERVICE_AREA_PRIOR,
    }
)


def _load_overlap_custom_footprints(paths: RepoPaths) -> pd.DataFrame:
    path = paths.repo_root / "configs" / "overlap_custom_footprints.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(OVERLAP_CUSTOM_FOOTPRINT_OUT_COLUMNS))
    footprints = pd.read_csv(path).copy()
    required = {
        "ori",
        "state_fips",
        "block_group_geoid",
        "weight_share",
        # How much of the BLOCK GROUP the footprint covers, on a 2020-population basis.
        # `weight_share` is a share of the AGENCY's mass and says nothing about how much of
        # the receiving block group the footprint occupies, so it cannot drive the exclusive
        # (remainder-displacing) semantics. Nullable except on displacing footprints.
        "bg_population_coverage_share",
        # What `weight_share` measures. Declared, never inferred -- see
        # VALID_CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASES.
        "weight_share_basis",
        "geometry_source_type",
        "geometry_source_ref",
        "footprint_note",
    }
    missing = required - set(footprints.columns)
    if missing:
        raise ValueError(f"Overlap custom footprints missing columns: {sorted(missing)}")
    footprints["ori9"] = footprints["ori"].astype("string")
    footprints["state_fips"] = footprints["state_fips"].astype("string").str.zfill(2)
    footprints["bg_id"] = footprints["block_group_geoid"].astype("string").str.zfill(12)
    footprints["weight_share"] = pd.to_numeric(footprints["weight_share"], errors="coerce").fillna(0.0)
    footprints["bg_population_coverage_share"] = pd.to_numeric(
        footprints["bg_population_coverage_share"], errors="coerce"
    )
    footprints = footprints[footprints["weight_share"].gt(0)].copy()
    if footprints.empty:
        return footprints
    bad_bg = ~footprints["bg_id"].str.fullmatch(r"\d{12}").fillna(False)
    if bool(bad_bg.any()):
        raise ValueError(
            "Overlap custom footprints carry malformed block_group_geoid values: "
            f"{footprints.loc[bad_bg, ['ori9', 'block_group_geoid']].to_dict(orient='records')}"
        )
    footprints = attach_service_scope_columns(footprints, paths)
    service = footprints["allocation_scope"].astype("string").eq("service_wide")
    coverage = footprints["bg_population_coverage_share"]
    # A service footprint can cover land in a zero-resident BG. Its explicit zero says that
    # the service row remains eligible for count-prior × land-area allocation while displacing
    # none of the resident county remainder. Missing is still unknown and therefore invalid.
    missing_service_coverage = service & coverage.isna()
    if bool(missing_service_coverage.any()):
        raise ValueError(
            "Service-wide footprint rows must carry explicit bg_population_coverage_share "
            "in [0, 1]; missing on: "
            f"{footprints.loc[missing_service_coverage, ['ori9', 'state_fips', 'bg_id']].to_dict(orient='records')}"
        )
    out_of_range = coverage.notna() & (
        coverage.lt(0.0)
        | coverage.gt(1.0 + 1e-9)
        | (~service & coverage.le(0.0))
    )
    if bool(out_of_range.any()):
        raise ValueError(
            "bg_population_coverage_share must lie in [0, 1] for service-wide rows "
            "and (0, 1] for ordinary rows: "
            f"{footprints.loc[out_of_range, ['ori9', 'bg_id', 'bg_population_coverage_share']].to_dict(orient='records')}"
        )
    footprints["_normalization_pool"] = np.where(
        service,
        "service|" + footprints["service_scope_id"].astype("string"),
        "state|" + footprints["ori9"].astype("string") + "|" + footprints["state_fips"],
    )
    totals = footprints.groupby("_normalization_pool", dropna=False)["weight_share"].sum().reset_index()
    bad = totals[~np.isclose(totals["weight_share"], 1.0, atol=1e-6)]
    if not bad.empty:
        raise ValueError(
            "Overlap custom footprints must sum to 1 by ordinary ori/state or service scope; "
            f"bad rows: {bad.to_dict(orient='records')}"
        )
    dupes = footprints.loc[footprints.duplicated(["ori9", "state_fips", "bg_id"], keep=False), ["ori9", "state_fips", "bg_id"]]
    if not dupes.empty:
        raise ValueError(
            "Duplicate overlap custom footprint rows: "
            f"{dupes.to_dict(orient='records')}"
        )
    basis = footprints["weight_share_basis"].astype("string").str.strip()
    bad_basis = ~basis.isin(VALID_CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASES)
    if bool(bad_basis.any()):
        raise ValueError(
            "Overlap custom footprints carry unknown weight_share_basis values "
            f"(valid: {sorted(VALID_CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASES)}): "
            f"{footprints.loc[bad_basis, ['ori9', 'weight_share_basis']].drop_duplicates().to_dict(orient='records')}"
        )
    footprints["weight_share_basis"] = basis
    # One basis per footprint. A mixed (ori, state) would need two normalisations inside one
    # pool and there is no defensible way to add an activity-weighted share to a verbatim one.
    mixed = footprints.groupby("_normalization_pool", dropna=False)[
        "weight_share_basis"
    ].nunique().rename("bases")
    mixed = mixed[mixed.gt(1)]
    if not mixed.empty:
        raise ValueError(
            "Overlap custom footprints mix weight_share_basis values inside one allocation pool "
            f"pool: {mixed.reset_index().to_dict(orient='records')}"
        )
    footprints = footprints.drop(columns="_normalization_pool")
    return footprints[list(OVERLAP_CUSTOM_FOOTPRINT_OUT_COLUMNS)].copy()


CONCURRENT_JURISDICTION_CARVEOUT_COLUMNS = (
    "state_fips",
    "county_fips",
    "county_geoid",
    "county_name",
    "reviewer_note",
    "remainder_exposure_before_displacement",
    "remainder_exposure_after_displacement",
    "remainder_exposure_retained_share",
    "reporting_remainder_agency_mass_2024",
    "source_artifact",
)
CONCURRENT_JURISDICTION_REVIEWER_NOTE = "concurrent_jurisdiction_unresolved"


def _load_concurrent_jurisdiction_carveouts(paths: RepoPaths) -> pd.DataFrame:
    """Counties where exclusive displacement reverts to SHARED overlap treatment.

    Stage 2 fork ruling 1: exclusive displacement is the default, but in the 20 counties
    where it removes more than half of a REPORTING non-municipal remainder agency's
    block-group exposure it trades one distortion for another -- a reporting sheriff whose
    territory is zeroed out is as wrong as a double-counted reservation. Those counties keep
    shared (additive) treatment until the PL-280 concurrent-jurisdiction question is
    adjudicated per case, and every row says so in `reviewer_note`.

    Fail-closed on: missing columns, malformed county geoids, duplicate counties, and any
    `reviewer_note` other than the declared one. A carve-out that matches no displacing
    footprint also fails, in `_build_exclusive_footprint_displacement` -- a registry row
    that quietly does nothing is the same defect class as the fail-open custom footprints
    Stage 2 closed.
    """
    path = paths.repo_root / "configs" / "concurrent_jurisdiction_carveouts.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(CONCURRENT_JURISDICTION_CARVEOUT_COLUMNS))
    carveouts = pd.read_csv(path).copy()
    missing = set(CONCURRENT_JURISDICTION_CARVEOUT_COLUMNS) - set(carveouts.columns)
    if missing:
        raise ValueError(f"Concurrent-jurisdiction carve-outs missing columns: {sorted(missing)}")
    if carveouts.empty:
        return carveouts
    carveouts["state_fips"] = carveouts["state_fips"].astype("string").str.zfill(2)
    carveouts["county_fips"] = carveouts["county_fips"].astype("string").str.zfill(3)
    carveouts["county_geoid"] = carveouts["county_geoid"].astype("string").str.zfill(5)
    rebuilt = carveouts["state_fips"] + carveouts["county_fips"]
    bad_geoid = ~carveouts["county_geoid"].eq(rebuilt) | ~_valid_county_fips(carveouts["county_fips"])
    if bool(bad_geoid.any()):
        raise ValueError(
            "Concurrent-jurisdiction carve-outs carry malformed county keys: "
            f"{carveouts.loc[bad_geoid, ['state_fips', 'county_fips', 'county_geoid']].to_dict(orient='records')}"
        )
    dupes = carveouts.loc[carveouts.duplicated("county_geoid", keep=False), ["county_geoid"]]
    if not dupes.empty:
        raise ValueError(
            f"Duplicate concurrent-jurisdiction carve-out counties: {dupes.to_dict(orient='records')}"
        )
    note = carveouts["reviewer_note"].astype("string").str.strip()
    bad_note = ~note.eq(CONCURRENT_JURISDICTION_REVIEWER_NOTE)
    if bool(bad_note.any()):
        raise ValueError(
            "Concurrent-jurisdiction carve-outs may only carry reviewer_note="
            f"{CONCURRENT_JURISDICTION_REVIEWER_NOTE!r} (a carve-out is an UNRESOLVED "
            "adjudication, not a decided treatment): "
            f"{carveouts.loc[bad_note, ['county_geoid', 'reviewer_note']].to_dict(orient='records')}"
        )
    carveouts["reviewer_note"] = note
    return carveouts[list(CONCURRENT_JURISDICTION_CARVEOUT_COLUMNS)].copy()


def _build_exclusive_footprint_displacement(
    *,
    overrides: pd.DataFrame,
    custom_footprints: pd.DataFrame,
    concurrent_jurisdiction_carveouts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per block group, the share of resident exposure an exclusive footprint takes over.

    Pro's Class D correction (STATE.md, v20 program): a tribal police department is the
    PRIMARY agency on its reservation, so its footprint share must DISPLACE the county
    remainder there rather than add to it -- mutually exclusive responsibility shares, every
    agency total allocated exactly once. Additive overlap for a primary agency double-counts.

    Returns one row per (state_fips, bg_id) with `displaced_share` in (0, 1]: the fraction of
    that block group's 2020 population inside one or more displacing footprints.
    """
    empty = pd.DataFrame(columns=["state_fips", "bg_id", "displaced_share"])
    if overrides.empty or custom_footprints.empty:
        return empty
    displacing = set(
        overrides.loc[overrides["displaces_county_remainder"], "ori9"].astype("string")
    )
    if not displacing:
        return empty
    rows = custom_footprints[custom_footprints["ori9"].astype("string").isin(displacing)].copy()
    if rows.empty:
        return empty
    unset = rows["bg_population_coverage_share"].isna()
    if bool(unset.any()):
        raise ValueError(
            "Footprints declared displaces_county_remainder must carry "
            "bg_population_coverage_share on every row (it is what leaves the county "
            "remainder). Missing on: "
            f"{rows.loc[unset, ['ori9', 'state_fips', 'bg_id']].to_dict(orient='records')}"
        )
    # Concurrent-jurisdiction carve-out (Stage 2 fork ruling 1): in the declared counties the
    # footprint stays exactly where it is and keeps its own mass, but it stops DISPLACING the
    # non-municipal remainder -- shared overlap, additive, pending PL-280 adjudication.
    carveouts = (
        concurrent_jurisdiction_carveouts
        if concurrent_jurisdiction_carveouts is not None
        else pd.DataFrame(columns=["county_geoid"])
    )
    if not carveouts.empty:
        row_county = rows["state_fips"].astype("string").str.zfill(2) + rows["bg_id"].astype(
            "string"
        ).str.zfill(12).str.slice(2, 5)
        carved = set(carveouts["county_geoid"].astype("string").str.zfill(5))
        in_carveout = row_county.isin(carved)
        matched = sorted(set(row_county.loc[in_carveout]))
        unmatched = sorted(carved - set(matched))
        if unmatched:
            raise ValueError(
                "configs/concurrent_jurisdiction_carveouts.csv names counties that no "
                "displacing footprint touches, so the carve-out is inert rather than "
                "load-bearing. Remove the row or fix the county: "
                f"{unmatched}"
            )
        rows = rows.loc[~in_carveout].copy()
        if rows.empty:
            return empty

    # MAX, not sum: `bg_population_coverage_share` is defined as the share of the block group
    # covered by the UNION of all displacing footprints, so every displacing row touching a
    # block group carries the same value and set semantics are already resolved upstream (in
    # the generator, which has the block-level data). Summing would double-displace the 24
    # footprints that two ORIs share -- duplicate tribal/BIA ORI pairs and joint OTSAs.
    displaced = (
        rows.groupby(["state_fips", "bg_id"], dropna=False)["bg_population_coverage_share"]
        .max()
        .rename("displaced_share")
        .reset_index()
    )
    displaced["displaced_share"] = (
        pd.to_numeric(displaced["displaced_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )
    return displaced[displaced["displaced_share"].gt(0.0)].reset_index(drop=True)


def _build_allocation_exclusive_footprint_displacement(
    *,
    overrides: pd.DataFrame,
    custom_footprints: pd.DataFrame,
    concurrent_jurisdiction_carveouts: pd.DataFrame | None = None,
    primary_response_policies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply allocation-only primary-service precedence inside reviewed carve-outs."""
    displacement = _build_exclusive_footprint_displacement(
        overrides=overrides,
        custom_footprints=custom_footprints,
        concurrent_jurisdiction_carveouts=concurrent_jurisdiction_carveouts,
    )
    carveouts = (
        concurrent_jurisdiction_carveouts
        if concurrent_jurisdiction_carveouts is not None
        else pd.DataFrame(columns=["county_geoid"])
    )
    policies = (
        primary_response_policies
        if primary_response_policies is not None
        else pd.DataFrame(columns=list(PRIMARY_SERVICE_RESPONSE_POLICY_COLUMNS))
    )
    if custom_footprints.empty or carveouts.empty or policies.empty:
        return displacement
    declared = set(policies["service_scope_id"].astype(str))
    available = set(custom_footprints["service_scope_id"].dropna().astype(str))
    missing_scopes = sorted(declared - available)
    if missing_scopes:
        raise ValueError(
            f"Primary service response policies name unknown service scopes: {missing_scopes}"
        )
    policy_scopes = set(
        policies.loc[
            policies["primary_response_policy"].eq(
                PRIMARY_RESPONSE_POLICY_CARVEOUT_PRECEDENCE
            ),
            "service_scope_id",
        ].astype(str)
    )
    displacing = set(
        overrides.loc[overrides["displaces_county_remainder"], "ori9"].astype("string")
    )
    primary = custom_footprints[
        custom_footprints["ori9"].astype("string").isin(displacing)
        & custom_footprints["service_scope_id"].astype("string").isin(policy_scopes)
    ].copy()
    effective_scopes = set(primary["service_scope_id"].dropna().astype(str))
    inactive_scopes = sorted(policy_scopes - effective_scopes)
    if inactive_scopes:
        raise ValueError(
            "Primary service response policies must name a service whose canonical target "
            f"displaces county remainder: {inactive_scopes}"
        )
    primary["_county_geoid"] = (
        primary["state_fips"].astype("string").str.zfill(2)
        + primary["bg_id"].astype("string").str.zfill(12).str.slice(2, 5)
    )
    carved = set(carveouts["county_geoid"].astype("string").str.zfill(5))
    primary = primary[primary["_county_geoid"].isin(carved)].copy()
    if primary.empty:
        return displacement
    owner_count = primary.groupby(["state_fips", "bg_id"], dropna=False)[
        "service_scope_id"
    ].nunique()
    if bool(owner_count.gt(1).any()):
        raise ValueError(
            "Multiple primary service owners overlap inside a reviewed carve-out; "
            "an explicit union coverage fraction is required"
        )
    own = pd.to_numeric(primary["bg_service_population_coverage_share"], errors="coerce")
    invalid = own.isna() | own.lt(0.0) | own.gt(1.0 + 1e-9)
    if bool(invalid.any()):
        raise ValueError(
            "Primary service precedence requires own service population coverage in [0, 1] "
            "on every carve-out row"
        )
    primary["displaced_share"] = own
    additions = primary.loc[
        primary["displaced_share"].gt(0), ["state_fips", "bg_id", "displaced_share"]
    ]
    if additions.empty:
        return displacement
    return pd.concat([displacement, additions], ignore_index=True)


def _apply_exclusive_footprint_displacement(
    bg_crosswalk: pd.DataFrame,
    displacement: pd.DataFrame,
) -> pd.DataFrame:
    """Remove displaced exposure from the state non-municipal remainder's block-group support.

    Scales the remainder rows' `allocation_share` (which is a population share of the block
    group) by `1 - displaced_share`. Municipal rows are untouched: a reservation inside an
    incorporated place is a separate question and is not what this displacement is about.

    The remainder's TOTAL target does not change -- only the ground it spreads over -- so
    conservation is unaffected; the reservation simply stops receiving sheriff mass on top of
    the tribal agency's own.
    """
    if bg_crosswalk.empty or displacement.empty:
        return bg_crosswalk
    out = bg_crosswalk.copy()
    key = "block_group_geoid" if "block_group_geoid" in out.columns else "bg_id"
    out["_state_key"] = out["state_fips"].astype("string").str.zfill(2)
    out["_bg_key"] = out[key].astype("string").str.zfill(12)
    factor = displacement.rename(columns={"state_fips": "_state_key", "bg_id": "_bg_key"})
    out = out.merge(factor, on=["_state_key", "_bg_key"], how="left")
    remainder = out["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
    share = pd.to_numeric(out["displaced_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    keep = pd.Series(1.0, index=out.index)
    keep.loc[remainder] = 1.0 - share.loc[remainder]
    out["allocation_share"] = (
        pd.to_numeric(out["allocation_share"], errors="coerce").fillna(0.0) * keep
    )
    for column in ("pop_share", "housing_share", "block_share", "aland_share"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) * keep
    out = out.drop(columns=["_state_key", "_bg_key", "displaced_share"], errors="ignore")
    # A remainder row scaled to exactly zero owns none of the block group any more; keeping
    # it would leave a zero-weight support row that no assertion distinguishes from a real
    # one, so drop it the same way the geometry build drops unsupported jurisdictions.
    drop = out["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE) & pd.to_numeric(
        out["allocation_share"], errors="coerce"
    ).fillna(0.0).le(0.0)
    out = out.loc[~drop].reset_index(drop=True)

    def _remainder_states(frame: pd.DataFrame) -> set[str]:
        mask = frame["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
        return set(frame.loc[mask, "state_fips"].astype("string").str.zfill(2))

    lost = _remainder_states(bg_crosswalk) - _remainder_states(out)
    if lost:
        raise ValueError(
            "Exclusive footprint displacement removed every non-municipal remainder block "
            f"group in state(s) {sorted(lost)}, which would strand that state's remainder "
            "target. Check bg_population_coverage_share in configs/overlap_custom_footprints.csv."
        )
    return out


def _apply_allocation_primary_service_displacement(
    *, paths: RepoPaths, bg_crosswalk: pd.DataFrame
) -> pd.DataFrame:
    """Apply final-output displacement, including reviewed primary-service precedence."""
    displacement = _build_allocation_exclusive_footprint_displacement(
        overrides=_load_overlap_footprint_overrides(paths),
        custom_footprints=_load_overlap_custom_footprints(paths),
        concurrent_jurisdiction_carveouts=_load_concurrent_jurisdiction_carveouts(paths),
        primary_response_policies=_load_primary_service_response_policies(paths),
    )
    if displacement.empty:
        return bg_crosswalk
    return _apply_exclusive_footprint_displacement(bg_crosswalk, displacement)


def _load_consolidated_agency_footprints(paths: RepoPaths) -> pd.DataFrame:
    path = paths.repo_root / "configs" / "consolidated_agency_footprints.csv"
    columns = [
        "ori9",
        "state_fips",
        "county_fips",
        "principal_jurisdiction_id",
        "included_place_geoids",
        "excluded_place_geoids",
        "geometry_source_type",
        "geometry_source_ref",
        "reviewer_note",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    footprints = pd.read_csv(path).copy()
    required = {
        "ori",
        "state_fips",
        "county_fips",
        "principal_jurisdiction_id",
        "included_place_geoids",
        "excluded_place_geoids",
        "geometry_source_type",
        "geometry_source_ref",
        "reviewer_note",
    }
    missing = required - set(footprints.columns)
    if missing:
        raise ValueError(f"Consolidated agency footprints missing columns: {sorted(missing)}")
    footprints["ori9"] = footprints["ori"].astype("string")
    footprints["state_fips"] = footprints["state_fips"].astype("string").str.zfill(2)
    footprints["county_fips"] = footprints["county_fips"].astype("string").str.zfill(3)
    footprints["principal_jurisdiction_id"] = footprints["principal_jurisdiction_id"].astype("string")
    bad_county = ~_valid_county_fips(footprints["county_fips"])
    if bool(bad_county.any()):
        raise ValueError(
            "Consolidated agency footprints have invalid county_fips: "
            f"{footprints.loc[bad_county, ['ori9', 'county_fips']].to_dict(orient='records')}"
        )
    dupes = footprints.loc[footprints.duplicated("ori9", keep=False), ["ori9"]]
    if not dupes.empty:
        raise ValueError(f"Duplicate consolidated agency footprints by ori: {dupes.to_dict(orient='records')}")
    return footprints[columns].copy()


def _build_consolidated_agency_support(
    merged: pd.DataFrame,
    footprints: pd.DataFrame,
) -> pd.DataFrame:
    if merged.empty or footprints.empty:
        return pd.DataFrame(columns=list(merged.columns))
    frames: list[pd.DataFrame] = []
    for row in footprints.itertuples(index=False):
        state_fips = str(row.state_fips).zfill(2)
        county_geoid = f"{state_fips}{str(row.county_fips).zfill(3)}"
        principal_id = str(row.principal_jurisdiction_id)
        principal = (
            merged["state_fips"].astype("string").str.zfill(2).eq(state_fips)
            & merged["jurisdiction_id"].astype("string").eq(principal_id)
            & merged["jurisdiction_type"].astype("string").eq("municipal")
        )
        remainder = (
            merged["state_fips"].astype("string").str.zfill(2).eq(state_fips)
            & merged["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
            & (
                merged["state_fips"].astype(str).str.zfill(2)
                + merged["bg_id"].astype("string").str.zfill(12).str.slice(2, 5)
            ).eq(county_geoid)
        )
        if not bool(principal.any()):
            raise ValueError(f"Consolidated agency footprint {row.ori9} missing principal support {principal_id}")
        if not bool(remainder.any()):
            raise ValueError(f"Consolidated agency footprint {row.ori9} missing county remainder support {county_geoid}")
        support = merged.loc[principal | remainder].copy()
        support["jurisdiction_id"] = principal_id
        support["jurisdiction_type"] = CONSOLIDATED_AGENCY_FOOTPRINT_TYPE
        group_cols = ["state_fips", "bg_id", "tract_id", "offense", "jurisdiction_id", "jurisdiction_type"]
        aggregations: dict[str, tuple[str, str]] = {
            "bg_weight": ("bg_weight", "first"),
            "allocation_share": ("allocation_share", "sum"),
        }
        if "pop20" in support.columns:
            # Population is a BG property repeated on the principal/remainder rows,
            # not mass to add when the consolidated footprint unions those rows.
            aggregations["pop20"] = ("pop20", "first")
        support = (
            support.groupby(group_cols, dropna=False, as_index=False)
            .agg(**aggregations)
        )
        support["allocation_share"] = pd.to_numeric(support["allocation_share"], errors="coerce").fillna(0.0).clip(
            lower=0.0,
            upper=1.0,
        )
        frames.append(support[merged.columns].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(merged.columns))


def _build_bg_prior_long(
    paths: RepoPaths,
    *,
    config: AllocationBuildConfig,
) -> pd.DataFrame:
    model_surface_config = _model_surface_config_from_allocation(paths=paths, config=config)
    cache_path = _bg_prior_cache_path(
        paths=paths,
        config=config,
        model_surface_config=model_surface_config,
    )
    controls_path = paths.state_dir / "controls" / f"jurisdiction_controls_{int(config.year)}.parquet"
    dependency_paths: list[Path] = [
        controls_path,
        *bg_feature_dependency_paths(paths, year=int(config.year)),
        Path(__file__).resolve().parent / "denominators.py",
    ]
    if model_surface_config.feature_policy_path is not None:
        dependency_paths.append(model_surface_config.feature_policy_path)
    if cache_path.exists() and not bool(config.force_bg_prior_rebuild):
        if config.bg_prior_path is not None or _is_default_step14_arm_b_model_surface(model_surface_config):
            return pd.read_parquet(cache_path)
        cache_mtime = cache_path.stat().st_mtime
        latest_dependency_mtime = max(
            (path.stat().st_mtime for path in dependency_paths if path.exists()),
            default=None,
        )
        if latest_dependency_mtime is None or cache_mtime >= latest_dependency_mtime:
            return pd.read_parquet(cache_path)
    bg_prior, _, _ = build_model_surface(
        paths=paths,
        config=model_surface_config,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bg_prior.to_parquet(cache_path, index=False)
    return bg_prior


def _regular_residual_prior_for_burglary_only_variant(
    *,
    paths: RepoPaths,
    bg_prior: pd.DataFrame,
    year: int,
) -> pd.DataFrame | None:
    """Return arm-B for the joint residual model when only burglary prior rows changed."""
    baseline_path = paths.state_dir / "modeling" / f"bg_prior_long_{int(year)}_arm_b.parquet"
    if not baseline_path.exists():
        return None
    required = {"bg_id", "offense", "bg_weight"}
    if not required.issubset(bg_prior.columns):
        return None
    baseline = pd.read_parquet(baseline_path)
    if not required.issubset(baseline.columns) or len(baseline) != len(bg_prior):
        return None
    current = bg_prior[["bg_id", "offense", "bg_weight"]].copy()
    base = baseline[["bg_id", "offense", "bg_weight"]].copy()
    current["bg_id"] = current["bg_id"].astype("string").str.zfill(12)
    base["bg_id"] = base["bg_id"].astype("string").str.zfill(12)
    current["offense"] = current["offense"].astype(str)
    base["offense"] = base["offense"].astype(str)
    if current.duplicated(["bg_id", "offense"]).any() or base.duplicated(["bg_id", "offense"]).any():
        return None
    merged = current.merge(
        base,
        on=["bg_id", "offense"],
        how="outer",
        suffixes=("_current", "_baseline"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        return None
    delta = (
        pd.to_numeric(merged["bg_weight_current"], errors="coerce").fillna(0.0)
        - pd.to_numeric(merged["bg_weight_baseline"], errors="coerce").fillna(0.0)
    ).abs()
    burglary = merged["offense"].astype(str).eq("burglary")
    if bool(delta.loc[~burglary].gt(1e-12).any()):
        return None
    if not bool(delta.loc[burglary].gt(1e-12).any()):
        return None
    return baseline


def _apply_model_lane_shares(
    merged: pd.DataFrame, *, mixture: MixtureRuntime | None
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """The model lane's within-jurisdiction share vector, legacy or three-expert mixture.

    The legacy block below is the historical computation, unmoved: the prior's activity weight
    times the crosswalk share, normalised within `(jurisdiction_id, state_fips, offense)`.

    When a mixture runtime is supplied (v2, off by default) the share vector -- and only the
    share vector -- is replaced by the E2 three-expert mixture. `model_total` survives
    untouched because it is the model pseudo-count mass `I` behind the rare-offense tract
    shrinkage weight `I / (I + K)`, whose constants were calibrated against the legacy
    magnitude; re-expressing the component weight at the mixture's shape leaves `I` exactly
    invariant. Everything downstream -- city posterior, raking, envelopes, folding, composites --
    reads the same columns it always did.

    Contract: `analysis_scratch/final_phase/MIXTURE_ALLOCATOR_CONTRACT.md`.
    """
    merged["model_component_weight"] = merged["bg_weight"] * merged["allocation_share"]
    merged["model_total"] = merged.groupby(
        ["jurisdiction_id", "state_fips", "offense"],
        dropna=False,
    )["model_component_weight"].transform("sum")
    merged["model_share"] = np.where(
        pd.to_numeric(merged["model_total"], errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(merged["model_component_weight"], errors="coerce").fillna(0.0)
        / pd.to_numeric(merged["model_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    if mixture is None:
        return merged, None

    merged, mixture_audit = apply_mixture_to_model_lane(merged, runtime=mixture)
    assert_mixture_share_invariants(audit=mixture_audit, weights=mixture.weights)
    if mixture.audit_path is not None:
        mixture.audit_path.parent.mkdir(parents=True, exist_ok=True)
        mixture_audit.to_parquet(mixture.audit_path, index=False)
    return merged, summarize_mixture_shares(mixture_audit)


def _calibrated_residual_policy_labels(transfer_tau: pd.Series) -> pd.Series:
    """Format residual-transfer policy labels with a stable string dtype.

    An empty calibrated-tau selection otherwise retains pandas' float dtype, and
    NumPy cannot concatenate the policy prefix to that empty float array.
    """
    formatted = pd.to_numeric(transfer_tau, errors="coerce").map(
        lambda value: f"{float(value):.2f}"
    )
    return formatted.astype("string").radd("calibrated_residual_tau")


def _build_jurisdiction_component_allocations(
    *,
    paths: RepoPaths,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    controls: pd.DataFrame,
    year: int,
    residual_training_city_shares_path: Path | None = None,
    residual_training_exclude_validation_case_types: tuple[str, ...] = (),
    exclude_feed_city_keys: tuple[str, ...] = (),
    feed_year_end: int | None = None,
    feed_inputs_dir: Path | None = None,
    residual_training_extra_bg_feature_paths: tuple[Path, ...] = (),
    residual_feature_policy_path: Path | None = DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
    residual_exclude_feature_policy_classes: tuple[str, ...] = DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
    residual_exclude_feature_policy_classes_by_offense: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    residual_transfer_tau_by_offense: tuple[tuple[str, float], ...] = DEFAULT_RESIDUAL_TRANSFER_TAU_BY_OFFENSE,
    rare_offense_information_constant_by_offense: tuple[
        tuple[str, float], ...
    ] = DEFAULT_RARE_OFFENSE_INFORMATION_CONSTANT_BY_OFFENSE,
    city_posterior_reconciliation_tolerance: float = CITY_POSTERIOR_RECONCILIATION_TOLERANCE,
    city_posterior_alpha_floor: float = CITY_POSTERIOR_ALPHA_FLOOR,
    city_posterior_alpha_volume_incidents: float = CITY_POSTERIOR_ALPHA_VOLUME_INCIDENTS,
    city_posterior_alpha_max_prior_fraction: float = CITY_POSTERIOR_ALPHA_MAX_PRIOR_FRACTION,
    enable_county_anchoring: bool = True,
    agency_estimates: pd.DataFrame | None = None,
    mixture: MixtureRuntime | None = None,
) -> pd.DataFrame:
    residual_transfer_tau = _residual_transfer_tau_dict(residual_transfer_tau_by_offense)
    mixture_summary: dict[str, object] | None = None
    rare_offense_information_constants = _rare_offense_information_constant_dict(
        rare_offense_information_constant_by_offense
    )
    consolidated_footprints = _load_consolidated_agency_footprints(paths) if bool(enable_county_anchoring) else pd.DataFrame()
    # EXCLUSIVE footprints displace the county remainder before anything reads the support,
    # so the remainder targets, the county/residual split and the allocation all see the same
    # ground. See `_apply_exclusive_footprint_displacement`.
    bg_crosswalk = _apply_allocation_primary_service_displacement(
        paths=paths, bg_crosswalk=bg_crosswalk
    )
    county_remainder_targets = (
        _build_county_remainder_group_targets(
            paths=paths,
            controls=controls,
            year=year,
            bg_crosswalk=bg_crosswalk,
            bg_prior=bg_prior,
            agency_estimates=agency_estimates,
        )
        if bool(enable_county_anchoring)
        else pd.DataFrame(columns=["state_fips", "offense", "group_kind", "group_id", "target_count", "reported_count"])
    )
    merged = bg_prior.merge(
        bg_crosswalk.rename(columns={"block_group_geoid": "bg_id"}),
        on=["bg_id", "state_fips"],
        how="inner",
    )
    merged["state_fips"] = merged["state_fips"].astype("string").str.zfill(2)
    merged["bg_id"] = merged["bg_id"].astype("string").str.zfill(12)
    if bool(enable_county_anchoring) and not consolidated_footprints.empty:
        consolidated_support = _build_consolidated_agency_support(merged, consolidated_footprints)
        principal_ids = set(consolidated_footprints["principal_jurisdiction_id"].astype(str))
        principal_mask = (
            merged["jurisdiction_type"].astype("string").eq("municipal")
            & merged["jurisdiction_id"].astype("string").isin(principal_ids)
        )
        merged = pd.concat(
            [
                merged.loc[~principal_mask].copy(),
                consolidated_support,
            ],
            ignore_index=True,
        )
    if bool(enable_county_anchoring) and not county_remainder_targets.empty:
        county_targets = county_remainder_targets[county_remainder_targets["group_kind"].eq("county_remainder")].copy()
        county_targets["county_geoid"] = county_targets["group_id"].astype("string").str.extract(r"(\d{5})$", expand=False)
        county_key_index = pd.MultiIndex.from_frame(
            county_targets[["state_fips", "offense", "county_geoid"]].drop_duplicates()
        )
        state_remainder = merged["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
        original_state_remainder = merged.loc[state_remainder].copy()
        merged_county_geoid = merged["state_fips"].astype(str).str.zfill(2) + merged["bg_id"].astype(str).str.slice(2, 5)
        merged_keys = pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "state_fips": merged["state_fips"].astype("string").str.zfill(2),
                    "offense": merged["offense"].astype("string"),
                    "county_geoid": merged_county_geoid.astype("string"),
                },
                index=merged.index,
            )
        )
        county_remainder = state_remainder & pd.Series(merged_keys.isin(county_key_index), index=merged.index)
        residual_remainder = state_remainder & ~county_remainder
        merged.loc[county_remainder, "jurisdiction_id"] = (
            merged.loc[county_remainder, "state_fips"].astype(str).str.zfill(2)
            + ":state_nonmunicipal_remainder:county:"
            + merged_county_geoid.loc[county_remainder].astype(str)
        )
        merged.loc[county_remainder, "jurisdiction_type"] = COUNTY_REMAINDER_TYPE
        merged.loc[residual_remainder, "jurisdiction_id"] = (
            merged.loc[residual_remainder, "state_fips"].astype(str).str.zfill(2)
            + ":state_nonmunicipal_remainder:residual"
        )
        merged.loc[residual_remainder, "jurisdiction_type"] = RESIDUAL_REMAINDER_TYPE
        residual_targets = county_remainder_targets[
            county_remainder_targets["group_kind"].eq("residual_remainder")
            & pd.to_numeric(county_remainder_targets["target_count"], errors="coerce").fillna(0.0).gt(0.0)
        ][["state_fips", "offense", "group_id"]].drop_duplicates()
        if not residual_targets.empty:
            residual_recipient_keys = merged[
                merged["jurisdiction_type"].astype("string").eq(RESIDUAL_REMAINDER_TYPE)
            ][["state_fips", "offense", "jurisdiction_id"]].drop_duplicates()
            missing_residual_targets = residual_targets.merge(
                residual_recipient_keys.rename(columns={"jurisdiction_id": "group_id"}),
                on=["state_fips", "offense", "group_id"],
                how="left",
                indicator=True,
            )
            missing_residual_targets = missing_residual_targets[missing_residual_targets["_merge"].eq("left_only")][
                ["state_fips", "offense", "group_id"]
            ]
            if not missing_residual_targets.empty and not original_state_remainder.empty:
                fallback_residual = original_state_remainder.merge(
                    missing_residual_targets,
                    on=["state_fips", "offense"],
                    how="inner",
                )
                if not fallback_residual.empty:
                    fallback_residual["jurisdiction_id"] = fallback_residual["group_id"]
                    fallback_residual["jurisdiction_type"] = RESIDUAL_REMAINDER_TYPE
                    fallback_residual = fallback_residual.drop(columns=["group_id"], errors="ignore")
                    merged = pd.concat([merged, fallback_residual], ignore_index=True)
    merged["bg_weight"] = pd.to_numeric(merged["bg_weight"], errors="coerce").fillna(0.0)
    merged["allocation_share"] = pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0)
    merged, mixture_summary = _apply_model_lane_shares(merged, mixture=mixture)
    # Restrict the allocation frame to jurisdictions with an actual control before
    # applying the residual model.  Geometry can contain zero-control placeholder
    # groups (notably state-remainder support made entirely of Census water cells);
    # they are not allocation targets and must not expand the promoted prediction
    # surface's key space.
    if bool(enable_county_anchoring):
        municipal_target = controls[controls["jurisdiction_type"].eq("municipal")][
            ["jurisdiction_id", "offense", "adjusted_count_ags_core"]
        ].copy()
        municipal_target["target_count"] = pd.to_numeric(
            municipal_target["adjusted_count_ags_core"],
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        remainder_target = county_remainder_targets.rename(columns={"group_id": "jurisdiction_id"})[
            ["jurisdiction_id", "offense", "target_count"]
        ].copy()
        target = pd.concat(
            [
                municipal_target[["jurisdiction_id", "offense", "target_count"]],
                remainder_target,
            ],
            ignore_index=True,
        )
    else:
        target = controls[
            controls["jurisdiction_type"].isin(["municipal", STATE_REMAINDER_TYPE])
        ][["jurisdiction_id", "offense", "adjusted_count_ags_core"]].copy()
        target["target_count"] = pd.to_numeric(
            target["adjusted_count_ags_core"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        target = target.drop(columns="adjusted_count_ags_core")
    merged = merged.merge(target, on=["jurisdiction_id", "offense"], how="inner")
    effective_feed_year_end = int(feed_year_end) if feed_year_end is not None else int(year)
    incident_surface = _load_city_incident_share_surface(
        paths,
        year=effective_feed_year_end,
        path=city_incident_share_surface_path(paths, feed_inputs_dir=feed_inputs_dir),
        exclude_feed_city_keys=exclude_feed_city_keys,
    )
    residual_training_surface = (
        _load_city_incident_share_surface(
            paths,
            year=effective_feed_year_end,
            path=residual_training_city_shares_path,
            exclude_validation_case_types=residual_training_exclude_validation_case_types,
            exclude_feed_city_keys=exclude_feed_city_keys,
        )
        if residual_training_city_shares_path is not None
        else incident_surface
    )
    baseline_model_share = pd.to_numeric(merged["model_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    residual_share = baseline_model_share.copy()
    residual_predicted_log_ratio = pd.Series(0.0, index=merged.index, dtype=float)
    residual_feature_policy_application: dict[str, object] | None = None
    if not incident_surface.empty:
        incident_surface = incident_surface.rename(columns={"block_group_geoid": "bg_id"})
        incident_active = (
            incident_surface[CITY_POSTERIOR_GROUP_COLS + ["city_name"]]
            .drop_duplicates(subset=CITY_POSTERIOR_GROUP_COLS)
            .rename(columns={"city_name": "city_posterior_city_name"})
            .assign(city_incident_posterior_active=True)
        )
        merged = merged.merge(
            incident_active,
            on=CITY_POSTERIOR_GROUP_COLS,
            how="left",
        )
        merged = merged.merge(
            incident_surface[
                [
                    *CITY_POSTERIOR_GROUP_COLS,
                    "bg_id",
                    "incident_count",
                    "share_within_city",
                    "pooled_source_year_count",
                    "pooled_source_year_min",
                    "pooled_source_year_max",
                ]
            ],
            on=[*CITY_POSTERIOR_GROUP_COLS, "bg_id"],
            how="left",
        )
        regular_residual_bg_prior = _regular_residual_prior_for_burglary_only_variant(
            paths=paths,
            bg_prior=bg_prior,
            year=int(year),
        )
        promoted_prediction_path = city_residual_prediction_surface_path(paths, year=int(year))
        holdout_refit = bool(exclude_feed_city_keys) or effective_feed_year_end != int(year)
        if promoted_prediction_path.exists() and not holdout_refit:
            prediction_manifest = load_city_residual_prediction_manifest(promoted_prediction_path)
            promoted_predictions = load_city_residual_prediction_surface(promoted_prediction_path)
            reconstructed_manifest: dict[str, object] | None = None
            try:
                residual_input = apply_city_residual_prediction_surface(
                    merged,
                    predictions=promoted_predictions,
                )
            except ValueError as exc:
                if not str(exc).startswith(
                    "Promoted residual predictions do not match the current model-share input"
                ):
                    raise
                reconstructed_path = city_residual_fitted_model_path(paths, year=int(year))
                if not reconstructed_path.exists():
                    raise
                reconstructed, reconstructed_manifest = load_city_residual_fitted_model(
                    reconstructed_path
                )
                residual_input = merged.copy()
                residual_input["_row_id"] = np.arange(len(residual_input), dtype=np.int64)
                residual_input, _ = attach_city_residual_features(
                    residual_input,
                    paths=paths,
                    year=int(year),
                    feature_cols=tuple(
                        sorted(
                            set(reconstructed.feature_cols)
                            | set(reconstructed.burglary_feature_cols)
                        )
                    ),
                    extra_feature_paths=list(reconstructed.config.extra_feature_paths),
                    feature_policy_path=reconstructed.config.feature_policy_path,
                    exclude_feature_policy_classes=tuple(
                        reconstructed.config.exclude_feature_policy_classes
                    ),
                    exclude_feature_policy_classes_by_offense=(
                        reconstructed.config.exclude_feature_policy_classes_by_offense
                    ),
                )
                residual_input = apply_city_residual_model(
                    residual_input,
                    fitted=reconstructed,
                ).sort_values("_row_id", kind="mergesort")
            residual_share = pd.to_numeric(
                residual_input["residual_model_share"], errors="coerce"
            ).fillna(baseline_model_share).clip(lower=0.0)
            residual_predicted_log_ratio = pd.Series(
                pd.to_numeric(
                    residual_input["predicted_log_ratio"], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=float),
                index=merged.index,
                dtype=float,
            )
            residual_feature_policy_application = dict(
                prediction_manifest["feature_policy_application"]
            )
            if reconstructed_manifest is None:
                residual_feature_policy_application.update({
                    "application_mode": "promoted_prediction_surface",
                    "prediction_surface_path": str(promoted_prediction_path),
                    "prediction_surface_row_count": int(len(residual_input)),
                })
            else:
                residual_feature_policy_application.update({
                    "application_mode": "reconstructed_fitted_model",
                    "prediction_surface_path": str(promoted_prediction_path),
                    "fitted_model_path": str(
                        city_residual_fitted_model_path(paths, year=int(year))
                    ),
                    "fitted_model_sha256": reconstructed_manifest["artifact_sha256"],
                    "promoted_surface_reproduction": reconstructed_manifest["reproduction"],
                })
        else:
            fitted = fit_city_residual_model_from_truth(
                paths=paths,
                city_shares=residual_training_surface.rename(columns={"block_group_geoid": "bg_id"}),
                bg_prior=regular_residual_bg_prior if regular_residual_bg_prior is not None else bg_prior,
                bg_crosswalk=bg_crosswalk,
                year=int(year),
                config=CityResidualConfig(
                    extra_feature_paths=tuple(residual_training_extra_bg_feature_paths),
                    feature_policy_path=_resolve_repo_path(paths, residual_feature_policy_path),
                    exclude_feature_policy_classes=tuple(
                        str(value) for value in residual_exclude_feature_policy_classes
                    ),
                    exclude_feature_policy_classes_by_offense=tuple(
                        (str(offense), tuple(str(value) for value in classes))
                        for offense, classes in residual_exclude_feature_policy_classes_by_offense
                    ),
                ),
                burglary_bg_prior=bg_prior if regular_residual_bg_prior is not None else None,
            )
            if fitted is None:
                fitted = None
        if (not promoted_prediction_path.exists() or holdout_refit) and fitted is not None:
            residual_feature_policy_application = fitted.feature_policy_application
            residual_input = merged.copy()
            residual_input["_row_id"] = np.arange(len(residual_input), dtype=np.int64)
            residual_input, _ = attach_city_residual_features(
                residual_input,
                paths=paths,
                year=int(year),
                feature_cols=tuple(sorted(set(fitted.feature_cols) | set(fitted.burglary_feature_cols))),
                extra_feature_paths=list(fitted.config.extra_feature_paths),
                feature_policy_path=fitted.config.feature_policy_path,
                exclude_feature_policy_classes=tuple(fitted.config.exclude_feature_policy_classes),
                exclude_feature_policy_classes_by_offense=fitted.config.exclude_feature_policy_classes_by_offense,
            )
            residual_input = apply_city_residual_model(
                residual_input,
                fitted=fitted,
            ).sort_values("_row_id", kind="mergesort")
            residual_share = pd.to_numeric(
                residual_input["residual_model_share"],
                errors="coerce",
            ).fillna(baseline_model_share).clip(lower=0.0)
            residual_predicted_log_ratio = pd.Series(
                pd.to_numeric(residual_input["predicted_log_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                index=merged.index,
                dtype=float,
            )

    city_posterior_active_for_policy = (
        merged["city_incident_posterior_active"].eq(True)
        if "city_incident_posterior_active" in merged.columns
        else pd.Series(False, index=merged.index)
    )
    uncovered_transfer_tau = (
        merged["offense"]
        .astype(str)
        .map(residual_transfer_tau)
        .fillna(1.0)
        .astype(float)
    )
    applied_transfer_tau = pd.Series(
        np.where(
            city_posterior_active_for_policy.to_numpy(dtype=bool),
            1.0,
            uncovered_transfer_tau.to_numpy(dtype=float),
        ),
        index=merged.index,
        dtype=float,
    )
    merged["city_residual_transfer_tau"] = applied_transfer_tau
    merged["city_residual_predicted_log_ratio"] = residual_predicted_log_ratio

    sparse_transfer_mask = (
        merged["offense"].astype("string").isin(SPARSE_BASELINE_TRANSFER_OFFENSES)
        & ~city_posterior_active_for_policy
        & applied_transfer_tau.le(0.0)
    )
    uncovered = ~city_posterior_active_for_policy
    full_residual_mask = uncovered & applied_transfer_tau.ge(1.0)
    calibrated_residual_mask = uncovered & applied_transfer_tau.gt(0.0) & applied_transfer_tau.lt(1.0)
    baseline_tau0_mask = uncovered & applied_transfer_tau.le(0.0) & ~sparse_transfer_mask
    merged["city_residual_transfer_policy"] = "covered_city_posterior_prior_full_residual_tau1"
    merged.loc[sparse_transfer_mask, "city_residual_transfer_policy"] = "baseline_sparse_offense"
    merged.loc[baseline_tau0_mask, "city_residual_transfer_policy"] = "calibrated_baseline_tau0"
    merged.loc[full_residual_mask, "city_residual_transfer_policy"] = "full_residual_tau1"
    merged.loc[calibrated_residual_mask, "city_residual_transfer_policy"] = (
        _calibrated_residual_policy_labels(
            applied_transfer_tau.loc[calibrated_residual_mask]
        )
    )
    tempered_raw = (
        baseline_model_share.to_numpy(dtype=float)
        * np.exp(
            np.clip(
                applied_transfer_tau.to_numpy(dtype=float) * residual_predicted_log_ratio.to_numpy(dtype=float),
                -50.0,
                50.0,
            )
        )
    )
    merged["city_posterior_model_prior_raw"] = np.where(
        np.isfinite(tempered_raw),
        tempered_raw,
        pd.to_numeric(residual_share, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
    )
    # Murder uses the same tract prior in every jurisdiction. A direct city feed may update
    # this prior below, but it does not select a separate BG-level spatial estimator.
    unified_murder_prior_mask = merged["offense"].astype("string").eq("murder")
    merged = _apply_uncovered_rare_offense_tract_shrinkage(
        merged,
        sparse_mask=sparse_transfer_mask | unified_murder_prior_mask,
        information_constants=rare_offense_information_constants,
    )
    model_prior_raw = pd.to_numeric(merged["city_posterior_model_prior_raw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    model_prior_total = model_prior_raw.groupby(
        [merged[col] for col in CITY_POSTERIOR_GROUP_COLS],
        dropna=False,
    ).transform("sum")
    allocation_raw = pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    allocation_total = allocation_raw.groupby(
        [merged[col] for col in CITY_POSTERIOR_GROUP_COLS],
        dropna=False,
    ).transform("sum")
    allocation_prior = np.where(
        allocation_total.gt(0.0),
        allocation_raw / allocation_total.replace(0.0, np.nan),
        0.0,
    )
    merged["city_posterior_model_prior_share"] = np.where(
        model_prior_total.gt(0.0),
        model_prior_raw / model_prior_total.replace(0.0, np.nan),
        allocation_prior,
    )
    merged["city_posterior_model_prior_share"] = pd.to_numeric(
        merged["city_posterior_model_prior_share"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    if "city_incident_posterior_active" not in merged.columns:
        merged["city_incident_posterior_active"] = False
    if "city_posterior_city_name" not in merged.columns:
        merged["city_posterior_city_name"] = pd.NA
    for col in [
        "incident_count",
        "share_within_city",
        "pooled_source_year_count",
        "pooled_source_year_min",
        "pooled_source_year_max",
    ]:
        if col not in merged.columns:
            merged[col] = np.nan

    quality = _load_city_posterior_quality(paths, year=year)
    if not quality.empty:
        merged = merged.merge(quality, on=CITY_POSTERIOR_GROUP_COLS, how="left")
    for col in _empty_city_posterior_quality_frame().columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged["city_posterior_city_name"] = merged["city_posterior_city_name"].fillna(
        merged["city_posterior_quality_city_name"]
    )

    direct_count = pd.to_numeric(merged["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    direct_total = direct_count.groupby(
        [merged[col] for col in CITY_POSTERIOR_GROUP_COLS],
        dropna=False,
    ).transform("sum")
    direct_share = np.where(
        direct_total.gt(0.0),
        direct_count / direct_total.replace(0.0, np.nan),
        0.0,
    )
    active = merged["city_incident_posterior_active"].eq(True) & direct_total.gt(0.0)
    q = np.where(active, 1.0, 0.0)

    control_total = pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feed_quality_count = pd.to_numeric(
        merged["city_posterior_feed_count_for_quality"],
        errors="coerce",
    )
    feed_control_fraction = np.where(
        active & control_total.gt(0.0) & feed_quality_count.notna(),
        feed_quality_count.clip(lower=0.0) / control_total.replace(0.0, np.nan),
        np.nan,
    )
    feed_control_ratio = np.asarray(feed_control_fraction, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_feed_control_ratio_abs = np.abs(np.log(feed_control_ratio))
    definitional_mismatch_fraction = np.where(
        np.isfinite(log_feed_control_ratio_abs),
        1.0 - np.exp(-log_feed_control_ratio_abs),
        np.where(active & control_total.gt(0.0) & feed_quality_count.fillna(0.0).le(0.0), 1.0, 0.0),
    )
    definitional_mismatch_fraction = np.clip(definitional_mismatch_fraction, 0.0, 1.0)
    missing_fraction = np.where(
        np.isfinite(feed_control_ratio),
        np.clip(1.0 - feed_control_ratio, 0.0, 1.0),
        0.0,
    )
    quality_mismatch_fraction = np.where(
        np.isfinite(feed_control_fraction),
        definitional_mismatch_fraction,
        0.0,
    )
    match_rate = pd.to_numeric(merged["city_posterior_match_rate"], errors="coerce").fillna(1.0).clip(
        lower=0.0,
        upper=1.0,
    )
    volume_count = feed_quality_count.where(feed_quality_count.notna(), direct_total).fillna(0.0).clip(lower=0.0)
    volume_scale = max(float(city_posterior_alpha_volume_incidents), 0.0)
    volume_prior_fraction = np.where(
        active,
        volume_scale / (volume_count + volume_scale) if volume_scale > 0.0 else 0.0,
        0.0,
    )
    direct_trust_fraction = (1.0 - quality_mismatch_fraction) * match_rate.to_numpy(dtype=float) * (1.0 - volume_prior_fraction)
    target_prior_fraction = np.clip(
        1.0 - direct_trust_fraction,
        0.0,
        min(max(float(city_posterior_alpha_max_prior_fraction), 0.0), 0.999999),
    )
    direct_mass = q * direct_total.to_numpy(dtype=float)
    alpha_unfloored = np.where(
        active,
        (target_prior_fraction / np.clip(1.0 - target_prior_fraction, 1e-12, None)) * direct_mass,
        0.0,
    )
    alpha_floor = max(float(city_posterior_alpha_floor), 0.0)
    alpha = np.where(active, np.maximum(alpha_unfloored, alpha_floor), 0.0)
    posterior_prior_fraction = np.where(
        active & ((direct_mass + alpha) > 0.0),
        alpha / np.clip(direct_mass + alpha, 1e-12, None),
        0.0,
    )

    posterior_numer = q * direct_count.to_numpy(dtype=float) + alpha * merged["city_posterior_model_prior_share"].to_numpy(dtype=float)
    posterior_total = pd.Series(posterior_numer, index=merged.index).groupby(
        [merged[col] for col in CITY_POSTERIOR_GROUP_COLS],
        dropna=False,
    ).transform("sum")
    posterior_share = pd.Series(
        np.where(
            active & posterior_total.gt(0.0),
            posterior_numer / posterior_total.replace(0.0, np.nan),
            merged["city_posterior_model_prior_share"],
        ),
        index=merged.index,
        dtype=float,
    )
    direct_share = pd.Series(direct_share, index=merged.index, dtype=float)
    alpha = pd.Series(alpha, index=merged.index, dtype=float)
    posterior_prior_fraction = pd.Series(posterior_prior_fraction, index=merged.index, dtype=float)
    volume_prior_fraction = pd.Series(volume_prior_fraction, index=merged.index, dtype=float)
    target_prior_fraction = pd.Series(target_prior_fraction, index=merged.index, dtype=float)

    active_murder = active & merged["offense"].astype("string").eq("murder")
    if bool(active_murder.any()):
        murder_posterior = _tract_incident_posterior(
            merged.loc[active_murder],
            direct_count=direct_count.loc[active_murder],
            prior_share=merged.loc[active_murder, "city_posterior_model_prior_share"],
            prior_incidents=MURDER_TRACT_POSTERIOR_PRIOR_INCIDENTS,
        )
        posterior_share.loc[active_murder] = murder_posterior["posterior_share"]
        direct_share.loc[active_murder] = murder_posterior["direct_share"]
        alpha.loc[active_murder] = murder_posterior["alpha"]
        posterior_prior_fraction.loc[active_murder] = murder_posterior["prior_fraction"]
        volume_prior_fraction.loc[active_murder] = murder_posterior["prior_fraction"]
        target_prior_fraction.loc[active_murder] = murder_posterior["prior_fraction"]
        merged.loc[active_murder, "rare_offense_allocation_policy"] = (
            "unified_murder_tract_incident_posterior"
        )
        merged.loc[active_murder, "city_residual_transfer_policy"] = (
            "unified_murder_tract_incident_posterior"
        )
    merged["city_posterior_q"] = q
    merged["city_posterior_alpha"] = alpha
    merged["city_posterior_missing_fraction"] = quality_mismatch_fraction
    merged["city_posterior_one_sided_missing_fraction"] = missing_fraction
    merged["city_posterior_definitional_mismatch_fraction"] = quality_mismatch_fraction
    merged["city_posterior_log_feed_control_ratio_abs"] = np.where(
        np.isfinite(log_feed_control_ratio_abs),
        log_feed_control_ratio_abs,
        np.nan,
    )
    merged["city_posterior_match_rate_effective"] = match_rate
    merged["city_posterior_volume_prior_fraction"] = volume_prior_fraction
    merged["city_posterior_target_prior_fraction"] = target_prior_fraction
    merged["city_posterior_prior_fraction"] = posterior_prior_fraction
    merged["city_posterior_direct_share"] = direct_share
    merged["city_posterior_share"] = posterior_share.fillna(0.0).clip(lower=0.0)
    merged["within_jurisdiction_weight"] = np.where(
        active,
        merged["city_posterior_share"],
        merged["city_posterior_model_prior_share"],
    )
    merged["component_activity_weight"] = pd.to_numeric(merged["within_jurisdiction_weight"], errors="coerce").fillna(0.0)

    posterior_diag = pd.DataFrame()
    if bool(active.any()):
        diag_base = merged.loc[active].copy()
        diag_base["_abs_delta_vs_direct"] = (
            pd.to_numeric(diag_base["city_posterior_share"], errors="coerce").fillna(0.0)
            - pd.to_numeric(diag_base["city_posterior_direct_share"], errors="coerce").fillna(0.0)
        ).abs()
        diag_base["_abs_delta_vs_prior"] = (
            pd.to_numeric(diag_base["city_posterior_share"], errors="coerce").fillna(0.0)
            - pd.to_numeric(diag_base["city_posterior_model_prior_share"], errors="coerce").fillna(0.0)
        ).abs()
        diag_base["_zero_feed_prior_positive"] = direct_count.loc[diag_base.index].le(0.0) & pd.to_numeric(
            diag_base["city_posterior_model_prior_share"],
            errors="coerce",
        ).fillna(0.0).gt(0.0)
        diag_base["_posterior_zero_feed_mass"] = np.where(
            diag_base["_zero_feed_prior_positive"],
            pd.to_numeric(diag_base["city_posterior_share"], errors="coerce").fillna(0.0),
            0.0,
        )
        diag_base["_direct_nonzero"] = direct_count.loc[diag_base.index].gt(0.0)
        posterior_diag = (
            diag_base.groupby(CITY_POSTERIOR_GROUP_COLS, dropna=False)
            .agg(
                city_name=("city_posterior_city_name", "first"),
                active_bg_count=("bg_id", "nunique"),
                direct_nonzero_bg_count=("_direct_nonzero", "sum"),
                zero_feed_prior_positive_bg_count=("_zero_feed_prior_positive", "sum"),
                feed_incident_mass=("incident_count", "sum"),
                feed_quality_count=("city_posterior_feed_count_for_quality", "first"),
                control_total=("target_count", "first"),
                missing_fraction=("city_posterior_missing_fraction", "first"),
                one_sided_missing_fraction=("city_posterior_one_sided_missing_fraction", "first"),
                definitional_mismatch_fraction=("city_posterior_definitional_mismatch_fraction", "first"),
                log_feed_control_ratio_abs=("city_posterior_log_feed_control_ratio_abs", "first"),
                match_rate=("city_posterior_match_rate_effective", "first"),
                volume_prior_fraction=("city_posterior_volume_prior_fraction", "first"),
                target_prior_fraction=("city_posterior_target_prior_fraction", "first"),
                posterior_prior_fraction=("city_posterior_prior_fraction", "first"),
                q=("city_posterior_q", "first"),
                alpha=("city_posterior_alpha", "first"),
                posterior_share_sum=("city_posterior_share", "sum"),
                direct_share_sum=("city_posterior_direct_share", "sum"),
                prior_share_sum=("city_posterior_model_prior_share", "sum"),
                tvd_posterior_vs_direct=("_abs_delta_vs_direct", lambda s: 0.5 * float(s.sum())),
                tvd_posterior_vs_prior=("_abs_delta_vs_prior", lambda s: 0.5 * float(s.sum())),
                max_abs_delta_vs_direct=("_abs_delta_vs_direct", "max"),
                posterior_mass_in_zero_feed_bgs=("_posterior_zero_feed_mass", "sum"),
                quality_year=("city_posterior_quality_year", "first"),
                mapped_count=("city_posterior_mapped_count", "first"),
                matched_count=("city_posterior_matched_count", "first"),
                published_count_for_diagnostic=("city_posterior_published_count_for_diagnostic", "first"),
                published_comparison_quality=("city_posterior_published_comparison_quality", "first"),
                pooled_source_year_min=("pooled_source_year_min", "min"),
                pooled_source_year_max=("pooled_source_year_max", "max"),
            )
            .reset_index()
        )
        posterior_diag["feed_control_fraction"] = np.where(
            pd.to_numeric(posterior_diag["control_total"], errors="coerce").fillna(0.0).gt(0.0),
            pd.to_numeric(posterior_diag["feed_quality_count"], errors="coerce").fillna(np.nan)
            / pd.to_numeric(posterior_diag["control_total"], errors="coerce").replace(0.0, np.nan),
            np.nan,
        )
        posterior_diag["reconciled_within_tolerance"] = pd.to_numeric(
            posterior_diag["feed_control_fraction"],
            errors="coerce",
        ).sub(1.0).abs().le(float(city_posterior_reconciliation_tolerance))

    sums = (
        merged.groupby(["jurisdiction_id", "offense"], dropna=False)
        .agg(
            activity_total=("component_activity_weight", "sum"),
            allocation_total=("allocation_share", "sum"),
        )
        .reset_index()
    )
    merged = merged.merge(sums, on=["jurisdiction_id", "offense"], how="left")
    denom_activity = pd.to_numeric(merged["activity_total"], errors="coerce").fillna(0.0)
    denom_alloc = pd.to_numeric(merged["allocation_total"], errors="coerce").fillna(0.0)
    numer_activity = pd.to_numeric(merged["component_activity_weight"], errors="coerce").fillna(0.0)
    raw_alloc = pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0)
    merged["component_share"] = np.where(
        denom_activity > 0,
        numer_activity / denom_activity,
        np.where(denom_alloc > 0, raw_alloc / denom_alloc, 0.0),
    )
    merged["component_count"] = pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0) * merged["component_share"]
    out = merged[
        [
            "state_fips",
            "bg_id",
            "tract_id",
            "jurisdiction_id",
            "jurisdiction_type",
            "offense",
            "component_count",
            "model_share",
            "city_residual_transfer_policy",
            "city_residual_transfer_tau",
            "city_residual_predicted_log_ratio",
            "city_incident_posterior_active",
            "incident_count",
            "city_posterior_q",
            "city_posterior_alpha",
            "city_posterior_prior_fraction",
            "city_posterior_direct_share",
            "city_posterior_share",
            "city_posterior_model_prior_raw",
            "city_posterior_model_prior_share",
            "rare_offense_allocation_policy",
            "rare_offense_information_constant",
            "rare_offense_effective_model_information",
            "rare_offense_model_information_weight",
            "rare_offense_residential_exposure_weight",
            "rare_offense_tract_model_share",
            "rare_offense_tract_exposure_share",
            "rare_offense_within_tract_exposure_share",
            "component_share",
        ]
    ].copy()
    out.attrs["city_posterior_diagnostics"] = posterior_diag
    out.attrs["city_posterior_summary"] = _summarize_city_posterior_diagnostics(posterior_diag)
    out.attrs["city_residual_feature_policy"] = residual_feature_policy_application or {}
    out.attrs["mixture_allocation_summary"] = mixture_summary or {}
    return out


def _coalesce_overlap_rake_groups(
    frame: pd.DataFrame,
    *,
    raw_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Return one row per overlap rake key before the observed/adjustment join.

    A no-support lane appends its state target to the residual group.  That group can already
    exist as a zero-raw agency group.  Leaving both rows in place turns the later outer merge
    into a many-to-many join and duplicates the other lane's target.  Coalescing is part of the
    accounting identity, not a numerical cleanup.
    """
    keys = ["state_fips", "offense", "group_kind", "group_id"]
    if frame.empty:
        return frame.copy()
    return (
        frame.groupby(keys, dropna=False, as_index=False)
        .agg(
            **{
                raw_column: (raw_column, "sum"),
                target_column: (target_column, "sum"),
                "county_anchor_evidence_count": ("county_anchor_evidence_count", "max"),
            }
        )
    )


# --- the custom-footprint mass rake (per-ORI conservation) ---------------------------------
#
# The municipal lane already conserves: a place's target IS its own admitted control, and the
# block-group shares inside it sum to one.  The custom-footprint lane did not.  Its target was
# the ORI's slice of the STATE overlap pool, apportioned by the target year's raw observed
# counts (or, on the smoothed surface, by a multi-year risk signal).  Two failures follow
# mechanically and were measured on the v52 surface over the 128 exclusively-owned footprints:
#
#   * an ORI that filed nothing in the target year draws a zero slice and places nothing, even
#     though its ledger row carries real repaired or pooled mass (Standing Rock NDDI00300:
#     pooled 71.1, placed 0.0; Rosebud SD0600200: 274.3 -> 0.0; 74 of 125 place zero);
#   * an ORI that did file draws a slice of the WHOLE state pool, including every silent
#     agency's imputed mass.  Where the state overlap layer is mostly imputed (NY: 4
#     contributing agencies of 604 crosswalked, 28,510 Part-1 smoothed) the few reporters
#     absorb it all -- Akwesasne NYDI02200 filed 19 and received 3,654; Fort Belknap MTDI05400
#     admitted 144 and received 474.  The state total balanced; every cell was wrong.
#
# The fix is the municipal lane's rule, applied here: a footprint ORI's target is its OWN
# admitted mass out of `level_lane_mass_ledger`, and the rest of the state pool is what is left
# for everybody else.  On the accounting surface that is `final_control_mass` exactly.  On the
# smoothed surface the state control is a temporally smoothed version of the same accounting
# total, so the ORI's ledger SHARE is carried across and multiplied by the smoothed state
# target; the two coincide when the surface factor is 1.  Either way the state control is still
# partitioned exactly, so no national total moves.
CUSTOM_FOOTPRINT_STATUS_PLACED = "placed"
CUSTOM_FOOTPRINT_STATUS_SUPPRESSED_DUPLICATE = "suppressed_duplicate"
CUSTOM_FOOTPRINT_STATUS_NO_PLACEABLE_SUPPORT = "unlocated_no_placeable_footprint"
FOOTPRINT_MASS_CONSERVATION_COLUMNS: tuple[str, ...] = (
    "ori9",
    "state_fips",
    "offense",
    "footprint_status",
    "canonical_target_ori",
    "shared_footprint_ori_count",
    "ledger_control_mass",
    "state_ledger_mass_total",
    "ledger_share_of_state",
    "control_surface_factor",
    "surface_control_mass",
    "expected_placed_mass",
    "placed_mass",
    "placed_minus_expected",
    "relative_error",
)
# The release gate, stated per ORI over all seven offences. 5% is wide enough to absorb the one
# downstream stage that is allowed to move mass between jurisdictions -- the model-only
# allocation envelope, which clips a block group's modelled rate and hands the clipped mass to
# other components -- and narrow enough that a re-broken rake cannot hide behind it.
FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR = 0.05
# The same band per ORI and offence is a different measurement, and it is not the release gate.
# A Vermont barracks footprint carrying 0.36 expected robberies fails a 5% relative test on an
# envelope clip of a tenth of a count, which is arithmetic about rounding rather than about
# placement. Two effects push single-offence rows past a one-offence floor without any placement
# error: the model-only rate-ratio cap clips a block group's modelled rate for one offence and
# redistributes the clipped mass inside the conservation group it belongs to (for the statewide
# overlap agencies that group is the whole state, so robbery mass moves between ORIs while the
# state total is untouched), and the per-ORI totals the gate actually measures stay inside the 5%
# band throughout. VTVSP1600 robbery is the worked example: ledger 4.0, placed 5.17, +1.19
# offences on the row, 2.06% across that ORI's seven offences. The floor is therefore 2.0
# offences, and the per-offence result is advisory -- reported in the validator summary, never an
# `issues` entry -- whenever the per-ORI gate passes. The per-ORI gate above stays blocking, and
# a per-offence row that survives this floor while its ORI is also over the band still shows up
# in the blocking message, where it says which offence broke the rake.
FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR = 2.0


def footprint_mass_conservation_path(output_dir: Path, *, year: int) -> Path:
    return Path(output_dir) / f"footprint_mass_conservation_{int(year)}.parquet"


def _load_level_lane_overlap_mass(paths: RepoPaths, *, year: int) -> pd.DataFrame:
    """Each overlap agency's own admitted control mass, per state and offense.

    Only the rows whose jurisdiction link IS the statewide overlap layer are read: an agency
    that also holds a municipal link owns that mass in the municipal lane, and double-counting
    it here would let a footprint draw more than the overlap control ever held.  The ledger is
    already jurisdiction-link weighted, so summing the rows is the whole arithmetic.
    """
    path = paths.state_dir / "controls" / f"level_lane_mass_ledger_{int(year)}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is the per-ORI admitted-mass ledger the custom-footprint lane rakes to. "
            "Build the controls stage before the allocator, or the footprint layer would fall "
            "back to the state-pool split this rake exists to replace."
        )
    ledger = pd.read_parquet(path)
    jurisdiction = ledger["jurisdiction_id"].astype("string")
    expected_suffix = ":" + STATE_OVERLAP_TYPE
    overlap = ledger[
        jurisdiction.str.endswith(expected_suffix).fillna(False)
        & ledger["ownership_class"].astype("string").eq("agency_level")
    ].copy()
    if overlap.empty:
        return pd.DataFrame(columns=["ori9", "state_fips", "offense", "ledger_control_mass"])
    overlap["ori9"] = overlap["ori9"].astype("string").str.upper()
    overlap["state_fips"] = (
        overlap["jurisdiction_id"].astype("string").str.slice(0, 2).str.zfill(2)
    )
    overlap["offense"] = overlap["offense"].astype("string")
    overlap["ledger_control_mass"] = (
        pd.to_numeric(overlap["final_control_mass"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    return (
        overlap.groupby(["ori9", "state_fips", "offense"], dropna=False, as_index=False)[
            "ledger_control_mass"
        ]
        .sum()
    )


def _canonical_footprint_owners(
    custom_footprints: pd.DataFrame, ledger: pd.DataFrame
) -> pd.DataFrame:
    """Resolve footprints held by more than one ORI to a single canonical owner.

    THE RULE, and why.  Thirty-four footprints in `configs/overlap_custom_footprints.csv` carry
    byte-identical block-group sets under two to four different ORIs.  Every one of them is a
    reporter-key duplication of the same police service on the same ground -- a tribal
    department and its BIA agency for the same reservation (AZ0048900 / AZDI05200 San Carlos,
    NY0162500 / NYDI02200 Akwesasne), or two keys for one joint OTSA.  They are not two services
    with separable territory, so placing both sums two accounts of the same ground onto the same
    cells; that is exactly how Akwesasne's two block groups came to publish 3,654 Part-1 counts.
    Partially overlapping but DIFFERENT footprints (two neighbouring reservations that share a
    block group) are not touched: each keeps its own mass on its own support, and per-ORI
    conservation is unaffected.
    The canonical key is the one that actually reported: largest accepted target-year ledger
    mass, then largest ledger mass of any kind, then the lexicographically first ORI so the
    choice is stable across builds.  The others are `suppressed_duplicate` -- their admitted
    mass is NOT dropped and NOT added on top; it is published in the unlocated-mass table, which
    is the one place this project records mass that is real but has no defensible ground.
    """
    columns = ["ori9", "state_fips", "canonical_target_ori", "shared_footprint_ori_count"]
    if custom_footprints.empty:
        return pd.DataFrame(columns=columns)
    footprints = custom_footprints[["ori9", "state_fips", "bg_id"]].copy()
    footprints["ori9"] = footprints["ori9"].astype("string").str.upper()
    # The state a footprint ORI draws its control from, which is its own state for an ordinary
    # footprint and the scope's SOURCE state for a reviewed service-wide one. Keying on the
    # destination state instead would give a service agency one footprint group per state it
    # covers, and each of those would claim the same single source-state control.
    footprints["state_fips"] = (
        custom_footprints.get("source_state_fips", custom_footprints["state_fips"])
        .astype("string")
        .fillna(custom_footprints["state_fips"].astype("string"))
        .str.zfill(2)
        .to_numpy()
    )
    footprints["bg_id"] = footprints["bg_id"].astype("string").str.zfill(12)
    support = (
        footprints.drop_duplicates(["ori9", "bg_id"])
        .groupby("ori9")["bg_id"]
        .apply(lambda values: "|".join(sorted(values)))
        .rename("_support_key")
        .reset_index()
    )
    rank = (
        ledger.groupby("ori9", dropna=False)["ledger_control_mass"].sum()
        if not ledger.empty
        else pd.Series(dtype=float)
    )
    support["_ledger_mass"] = (
        support["ori9"].map(rank).astype(float).fillna(0.0) if len(rank) else 0.0
    )
    support = support.sort_values(
        ["_support_key", "_ledger_mass", "ori9"], ascending=[True, False, True], kind="mergesort"
    )
    canonical = support.groupby("_support_key")["ori9"].transform("first")
    support["canonical_target_ori"] = canonical
    support["shared_footprint_ori_count"] = support.groupby("_support_key")["ori9"].transform(
        "size"
    )
    owners = support[["ori9", "canonical_target_ori", "shared_footprint_ori_count"]]
    identity = footprints[["ori9", "state_fips"]].drop_duplicates()
    ambiguous = identity[identity.duplicated("ori9", keep=False)]
    if not ambiguous.empty:
        raise ValueError(
            "a custom-footprint ORI draws from more than one source state, so its own control "
            "cannot be identified: " + str(ambiguous.to_dict(orient="records"))
        )
    out = identity.merge(owners, on="ori9", how="left", validate="one_to_one")
    out["canonical_target_ori"] = out["canonical_target_ori"].fillna(out["ori9"])
    out["shared_footprint_ori_count"] = (
        pd.to_numeric(out["shared_footprint_ori_count"], errors="coerce").fillna(1).astype(int)
    )
    return out[columns]


def _placeable_footprint_oris(
    custom_footprints: pd.DataFrame, bg_prior: pd.DataFrame | None
) -> set[str]:
    """(ori9, state_fips) pairs whose footprint has at least one block group the prior carries.

    The allocator inner-joins the footprint rows to `bg_prior`.  An ORI whose every footprint
    block group is absent from the prior therefore contributes no components at all, and its
    target would evaporate without a trace -- which is the failure mode the unlocated bucket
    exists to make visible.  With no prior available (unit tests that exercise the target
    arithmetic alone) every footprint is treated as placeable.
    """
    if custom_footprints.empty:
        return set()
    rows = custom_footprints[["ori9", "state_fips", "bg_id", "weight_share"]].copy()
    rows["ori9"] = rows["ori9"].astype("string").str.upper()
    rows["state_fips"] = rows["state_fips"].astype("string").str.zfill(2)
    rows["bg_id"] = rows["bg_id"].astype("string").str.zfill(12)
    rows = rows[pd.to_numeric(rows["weight_share"], errors="coerce").fillna(0.0).gt(0.0)]
    if bg_prior is None or bg_prior.empty:
        return set(rows["ori9"].astype(str))
    prior_keys = set(
        zip(
            bg_prior["state_fips"].astype("string").str.zfill(2),
            bg_prior["bg_id"].astype("string").str.zfill(12),
            strict=True,
        )
    )
    supported = rows[
        [
            (state, bg) in prior_keys
            for state, bg in zip(rows["state_fips"], rows["bg_id"], strict=True)
        ]
    ]
    return set(supported["ori9"].astype(str))


def _footprint_ledger_partition(
    *,
    custom_footprints: pd.DataFrame,
    ledger: pd.DataFrame,
    overlap_controls: pd.DataFrame,
    bg_prior: pd.DataFrame | None,
    eligible_oris: set[str] | None = None,
) -> pd.DataFrame:
    """One row per (footprint ORI, state, offense): what that ORI is owed, and whether it lands.

    `ledger_share_of_state` is the ORI's share of the state overlap layer's own accounting
    total, read straight off the ledger.  `control_surface_factor` is what the published control
    surface does to that total as a whole (1.0 on the accounting surface; the temporal smoothing
    factor on the smoothed surface).  Their product times the state control is the mass the
    footprint must place, which on the accounting surface IS `final_control_mass`.
    """
    columns = list(FOOTPRINT_MASS_CONSERVATION_COLUMNS)
    if custom_footprints.empty or overlap_controls.empty:
        return pd.DataFrame(columns=columns)
    owners = _canonical_footprint_owners(custom_footprints, ledger)
    if eligible_oris is not None:
        # Only an ORI the registry actually routed to its own footprint may take mass out of
        # the state pool here. A stray coverage row is a config artifact, not a ruling.
        eligible = {str(value).upper() for value in eligible_oris}
        owners = owners[owners["ori9"].astype("string").isin(eligible)]
    if owners.empty:
        return pd.DataFrame(columns=columns)
    state_total = (
        ledger.groupby(["state_fips", "offense"], dropna=False, as_index=False)[
            "ledger_control_mass"
        ]
        .sum()
        .rename(columns={"ledger_control_mass": "state_ledger_mass_total"})
    )
    offenses = overlap_controls[["state_fips", "offense", "state_target"]].copy()
    offenses["state_fips"] = offenses["state_fips"].astype("string").str.zfill(2)
    offenses["offense"] = offenses["offense"].astype("string")
    partition = owners.merge(offenses, on="state_fips", how="inner")
    partition = partition.merge(
        ledger, on=["ori9", "state_fips", "offense"], how="left"
    ).merge(state_total, on=["state_fips", "offense"], how="left")
    partition["ledger_control_mass"] = (
        pd.to_numeric(partition["ledger_control_mass"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    partition["state_ledger_mass_total"] = (
        pd.to_numeric(partition["state_ledger_mass_total"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    total = partition["state_ledger_mass_total"]
    partition["ledger_share_of_state"] = np.where(
        total.gt(0.0), partition["ledger_control_mass"] / total.where(total.gt(0.0), 1.0), 0.0
    )
    state_target = pd.to_numeric(partition["state_target"], errors="coerce").fillna(0.0)
    partition["control_surface_factor"] = np.where(
        total.gt(0.0), state_target / total.where(total.gt(0.0), 1.0), 1.0
    )
    # What this ORI's own admitted mass is worth on the PUBLISHED control surface. On the
    # accounting surface the factor is 1 and this is `final_control_mass` exactly.
    partition["surface_control_mass"] = state_target * partition["ledger_share_of_state"]

    placeable = _placeable_footprint_oris(custom_footprints, bg_prior)
    is_canonical = partition["ori9"].astype("string").eq(
        partition["canonical_target_ori"].astype("string")
    )
    has_support = partition["ori9"].astype(str).isin(placeable)
    partition["footprint_status"] = np.where(
        ~is_canonical,
        CUSTOM_FOOTPRINT_STATUS_SUPPRESSED_DUPLICATE,
        np.where(
            has_support,
            CUSTOM_FOOTPRINT_STATUS_PLACED,
            CUSTOM_FOOTPRINT_STATUS_NO_PLACEABLE_SUPPORT,
        ),
    )
    # Only a canonical, supported footprint is expected to place anything. The other two
    # statuses owe their mass to the unlocated table, so their expected PLACED mass is zero and
    # any block-group component under them is a defect the release gate must see.
    partition["expected_placed_mass"] = np.where(
        partition["footprint_status"].eq(CUSTOM_FOOTPRINT_STATUS_PLACED),
        partition["surface_control_mass"],
        0.0,
    )
    partition["placed_mass"] = np.nan
    partition["placed_minus_expected"] = np.nan
    partition["relative_error"] = np.nan
    return partition[columns].sort_values(
        ["state_fips", "offense", "ori9"], kind="mergesort"
    ).reset_index(drop=True)


def _footprint_unlocated_routes(partition: pd.DataFrame) -> pd.DataFrame:
    """The mass the rake refuses to place, keyed by the route that refused it."""
    keys = ["state_fips", "offense", "overlap_resolution_route", "route_target_count"]
    if partition.empty:
        return pd.DataFrame(columns=keys)
    route = pd.Series(pd.NA, index=partition.index, dtype="string")
    route.loc[
        partition["footprint_status"].eq(CUSTOM_FOOTPRINT_STATUS_SUPPRESSED_DUPLICATE)
    ] = OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE
    route.loc[
        partition["footprint_status"].eq(CUSTOM_FOOTPRINT_STATUS_NO_PLACEABLE_SUPPORT)
    ] = OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT
    out = partition.assign(overlap_resolution_route=route)
    out = out[out["overlap_resolution_route"].notna()]
    if out.empty:
        return pd.DataFrame(columns=keys)
    return (
        out.groupby(
            ["state_fips", "offense", "overlap_resolution_route"], dropna=False, as_index=False
        )["surface_control_mass"]
        .sum()
        .rename(columns={"surface_control_mass": "route_target_count"})
    )


def _apply_footprint_ledger_rake(
    grouped: pd.DataFrame,
    *,
    partition: pd.DataFrame,
    overlap_controls: pd.DataFrame,
    residual_kind: str,
) -> pd.DataFrame:
    """Replace the footprint groups' pooled slice with their own ledger mass.

    Three destinations, and the state control is still partitioned exactly:
      * a canonical footprint with placeable support gets a `custom_footprint_overlap` group
        carrying exactly its own ledger mass on the published surface;
      * a suppressed duplicate, and a footprint whose support the prior does not carry, have
        their mass added to the residual group (the unlocated bucket when that lane is on);
      * every other group in the state shares what is left, in the proportions the cascade
        already gave them, so no other lane's relative answer changes.
    """
    if grouped.empty or partition.empty:
        return grouped
    # Only the five partition columns. The caller's frame also carries the state control and the
    # rake diagnostics it was built with, and those names collide with the ones this function
    # merges on -- a collision pandas resolves by SUFFIXING rather than failing.
    work = grouped[["state_fips", "offense", "group_kind", "group_id", "target_count"]].copy()
    work["state_fips"] = work["state_fips"].astype("string").str.zfill(2)
    work["offense"] = work["offense"].astype("string")
    work["group_id"] = work["group_id"].astype("string")
    work["target_count"] = pd.to_numeric(work["target_count"], errors="coerce").fillna(0.0)

    placed = partition[partition["footprint_status"].eq(CUSTOM_FOOTPRINT_STATUS_PLACED)]
    placed_rows = (
        placed.groupby(["state_fips", "offense", "ori9"], dropna=False, as_index=False)[
            "surface_control_mass"
        ]
        .sum()
        .rename(columns={"ori9": "group_id", "surface_control_mass": "target_count"})
        .assign(group_kind="custom_footprint_overlap")
    )
    unplaced = (
        partition[~partition["footprint_status"].eq(CUSTOM_FOOTPRINT_STATUS_PLACED)]
        .groupby(["state_fips", "offense"], dropna=False, as_index=False)["surface_control_mass"]
        .sum()
        .rename(columns={"surface_control_mass": "_unplaced_mass"})
    )
    footprint_total = (
        partition.groupby(["state_fips", "offense"], dropna=False, as_index=False)[
            "surface_control_mass"
        ]
        .sum()
        .rename(columns={"surface_control_mass": "_footprint_mass"})
    )

    keep = work[~work["group_kind"].eq("custom_footprint_overlap")].copy()
    # The residual lane has to exist wherever footprint mass needs holding. Under the smoothed
    # partition a state/offense keeps only the groups its own agencies produced, so a state whose
    # overlap agencies all resolved cleanly has no residual row at all -- and the suppressed
    # duplicate on one of its reservations would have nowhere to go. Seed it at zero; the
    # arithmetic below then treats it like any other lane.
    needed = unplaced[unplaced["_unplaced_mass"].gt(1e-9)][["state_fips", "offense"]]
    if not needed.empty:
        present = keep.loc[keep["group_kind"].eq(residual_kind), ["state_fips", "offense"]]
        missing = needed.merge(
            present.drop_duplicates(), on=["state_fips", "offense"], how="left", indicator=True
        )
        missing = missing[missing["_merge"].eq("left_only")].drop(columns="_merge")
        if not missing.empty:
            keep = pd.concat(
                [
                    keep,
                    missing.assign(
                        group_kind=residual_kind,
                        group_id=missing["state_fips"].astype("string"),
                        target_count=0.0,
                    ),
                ],
                ignore_index=True,
                sort=False,
            )
    controls = overlap_controls[["state_fips", "offense", "state_target"]].copy()
    controls["state_fips"] = controls["state_fips"].astype("string").str.zfill(2)
    controls["offense"] = controls["offense"].astype("string")
    keep = (
        keep.merge(controls, on=["state_fips", "offense"], how="left")
        .merge(footprint_total, on=["state_fips", "offense"], how="left")
        .merge(unplaced, on=["state_fips", "offense"], how="left")
        .reset_index(drop=True)
    )
    for column in ("state_target", "_footprint_mass", "_unplaced_mass"):
        keep[column] = pd.to_numeric(keep[column], errors="coerce").fillna(0.0)
    residual = keep["group_kind"].eq(residual_kind).to_numpy()
    other_total = (
        keep.groupby(["state_fips", "offense"], dropna=False)["target_count"]
        .sum()
        .rename("_other_total")
    )
    keep["_other_total"] = (
        other_total.reindex(pd.MultiIndex.from_frame(keep[["state_fips", "offense"]]))
        .to_numpy(dtype=float)
    )
    keep["_other_total"] = keep["_other_total"].fillna(0.0)

    # Seeding above makes this unreachable; it stays as the assertion that it did its job,
    # because unplaceable mass with nowhere to land is the silence this lane exists to prevent.
    residual_keys = set(
        map(tuple, keep.loc[residual, ["state_fips", "offense"]].to_numpy().tolist())
    )
    orphaned = keep[
        keep["_unplaced_mass"].gt(1e-9)
        & ~pd.Series(
            [tuple(row) for row in keep[["state_fips", "offense"]].to_numpy().tolist()],
            index=keep.index,
        ).isin(residual_keys)
    ]
    if not orphaned.empty:
        raise ValueError(
            "custom-footprint mass could not be placed and the state has no residual group to "
            "hold it: "
            + str(
                orphaned[["state_fips", "offense", "_unplaced_mass"]]
                .drop_duplicates()
                .head(20)
                .to_dict(orient="records")
            )
        )

    # Everything that is not a placed footprint shares what the footprints left, in the
    # proportions the cascade already gave it; the residual lane additionally receives the
    # footprint mass that has no ground to stand on.
    pool = (keep["_footprint_mass"].rsub(keep["state_target"])).clip(lower=0.0).to_numpy(dtype=float)
    denom = keep["_other_total"].to_numpy(dtype=float)
    scaled = np.where(denom > 0.0, keep["target_count"].to_numpy(dtype=float) * pool / np.where(denom > 0.0, denom, 1.0), 0.0)
    if bool((denom <= 0.0).any()):
        # No ordinary group claims anything in this state/offense: the whole remainder is the
        # residual lane's by definition.
        scaled = np.where((denom <= 0.0) & residual, pool, scaled)
    keep["target_count"] = scaled + np.where(residual, keep["_unplaced_mass"].to_numpy(dtype=float), 0.0)
    keep = keep.drop(
        columns=["state_target", "_footprint_mass", "_other_total", "_unplaced_mass"]
    )
    out = pd.concat([keep, placed_rows], ignore_index=True, sort=False)
    out["target_count"] = pd.to_numeric(out["target_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out.groupby(
        ["state_fips", "offense", "group_kind", "group_id"], dropna=False, as_index=False
    ).agg(target_count=("target_count", "sum"))

    check = controls.merge(
        out.groupby(["state_fips", "offense"], dropna=False, as_index=False)["target_count"].sum(),
        on=["state_fips", "offense"],
        how="left",
    )
    drift = float(
        (pd.to_numeric(check["target_count"], errors="coerce").fillna(0.0) - check["state_target"])
        .abs()
        .max()
        or 0.0
    )
    if drift > 1e-6:
        raise ValueError(
            f"footprint ledger rake broke the overlap partition; max abs delta={drift:.3e}"
        )
    return out


def _build_overlap_group_targets(
    *,
    paths: RepoPaths,
    controls: pd.DataFrame,
    year: int,
    enable_county_anchoring: bool = True,
    bg_prior: pd.DataFrame | None = None,
    agency_estimates: pd.DataFrame | None = None,
    preferred_observations: pd.DataFrame | None = None,
    agency_risk_signals: pd.DataFrame | None = None,
    bg_crosswalk: pd.DataFrame | None = None,
    unlocated_mass: bool = False,
) -> pd.DataFrame:
    control_cols = ["state_fips", "offense", "adjusted_count_ags_core"]
    if "reported_count_preferred" in controls.columns:
        control_cols.append("reported_count_preferred")
    overlap_controls = controls[controls["jurisdiction_type"].eq("statewide_overlap_layer")][control_cols].copy()
    if overlap_controls.empty:
        return pd.DataFrame(columns=["state_fips", "offense", "group_kind", "group_id", "target_count"])
    overlap_controls["state_fips"] = overlap_controls["state_fips"].astype("string").str.zfill(2)
    overlap_controls["offense"] = overlap_controls["offense"].astype("string")
    overlap_controls["state_target"] = pd.to_numeric(
        overlap_controls["adjusted_count_ags_core"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    if "reported_count_preferred" in overlap_controls.columns:
        overlap_controls["state_reported_target"] = pd.to_numeric(
            overlap_controls["reported_count_preferred"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
    else:
        overlap_controls["state_reported_target"] = 0.0
    overlap_controls["state_reported_target"] = np.minimum(
        overlap_controls["state_reported_target"].to_numpy(dtype=float),
        overlap_controls["state_target"].to_numpy(dtype=float),
    )
    overlap_controls["state_adjustment_target"] = (
        overlap_controls["state_target"] - overlap_controls["state_reported_target"]
    ).clip(lower=0.0)

    preferred = (
        preferred_observations.copy()
        if preferred_observations is not None
        else build_agency_preferred_observations(paths=paths, year=year)
    )[["ori9", "state_fips", "offense", "preferred_count"]].copy()
    preferred["state_fips"] = preferred["state_fips"].astype("string").str.zfill(2)
    preferred["preferred_count"] = pd.to_numeric(preferred["preferred_count"], errors="coerce").fillna(0.0)

    crosswalk = _load_crosswalk(paths)
    overlap_cw = _statewide_overlap_crosswalk_rows(crosswalk)
    if overlap_cw.empty:
        return pd.DataFrame(columns=["state_fips", "offense", "group_kind", "group_id", "target_count"])

    agency_estimates = (
        agency_estimates.copy()
        if agency_estimates is not None
        else _build_agency_allocation_target_estimates(paths=paths, year=int(year))
    )

    agency_master_full = _load_agency_master(paths)
    overlap_master_cols = [
        "ori9",
        "state_fips",
        "state_abbr",
        "county_fips",
        "place_fips",
        "agency_name_std",
        "agency_type_norm",
    ]
    if "county_fips_source" in agency_master_full.columns:
        overlap_master_cols.append("county_fips_source")
    agency_master = agency_master_full[overlap_master_cols].copy()
    agency_master["state_fips"] = agency_master["state_fips"].astype("string").str.zfill(2)
    agency_master["state_abbr"] = agency_master["state_abbr"].astype("string").str.upper()
    if "county_fips_source" not in agency_master.columns:
        agency_master["county_fips_source"] = pd.NA
    agency_master["county_fips_source"] = agency_master["county_fips_source"].astype("string")

    jm = _load_jurisdiction_master(paths)
    place_map = (
        jm[
            (jm["jurisdiction_type"].eq("municipal"))
            & (jm["geo_type"].eq("place"))
            & jm["geoid"].notna()
        ][["geoid", "jurisdiction_id"]]
        .drop_duplicates()
    )
    if bg_crosswalk is not None and not bg_crosswalk.empty:
        # A place jurisdiction with no BG-crosswalk support is not a valid
        # overlap-localization target: the place allocator inner-joins the
        # municipal BG crosswalk, so localizing to it would silently strand the
        # group's mass. This is the same normalization rule as the geometry
        # build's dead-jurisdiction exclusion (globally dead ORIs are dropped
        # from BG assignment, so those jurisdictions have no crosswalk rows):
        # treat the place as if it did not exist, so the agency falls through
        # the existing cascade to county_overlap / statewide_overlap.
        supported_municipal_ids = set(
            bg_crosswalk.loc[
                bg_crosswalk["jurisdiction_type"].astype("string").eq("municipal"),
                "jurisdiction_id",
            ].astype(str)
        )
        place_map = place_map[place_map["jurisdiction_id"].astype(str).isin(supported_municipal_ids)]
    place_to_jurisdiction = dict(zip(place_map["geoid"].astype(str), place_map["jurisdiction_id"].astype(str)))

    merged = _build_overlap_evidence_spine(
        preferred=preferred,
        agency_estimates=agency_estimates,
        overlap_crosswalk=overlap_cw,
    )
    merged = merged.merge(agency_master, on=["ori9", "state_fips"], how="left", suffixes=("", "_agency"))
    overrides = _load_overlap_footprint_overrides(paths)
    custom_footprints = _load_overlap_custom_footprints(paths)
    _assert_custom_footprint_overrides_have_rows(overrides, custom_footprints)
    if not overrides.empty:
        merged = merged.merge(
            overrides[
                [
                    "ori9",
                    "final_overlap_treatment",
                    "overlap_subtype_final",
                    "footprint_type",
                    "target_state_fips",
                    "target_county_fips",
                    "target_place_fips",
                    "target_jurisdiction_id",
                    "geometry_source_type",
                    "geometry_source_ref",
                    "confidence",
                ]
            ],
            on="ori9",
            how="left",
        )
    if not custom_footprints.empty:
        merged = merged.merge(
            custom_footprints[["ori9", "state_fips"]].drop_duplicates().assign(has_custom_footprint=True),
            on=["ori9", "state_fips"],
            how="left",
        )
    merged["geometry_hint"] = merged["geometry_hint"].astype("string")
    merged["place_geoid"] = merged["state_fips"].astype(str).str.zfill(2) + merged["place_fips"].astype("string").str.zfill(5)
    merged["county_geoid"] = merged["state_fips"].astype(str).str.zfill(2) + merged["county_fips"].astype("string").str.zfill(3)
    merged["place_jurisdiction_id"] = merged["place_geoid"].map(place_to_jurisdiction)
    merged["target_state_fips"] = merged.get("target_state_fips", pd.Series(index=merged.index)).astype("string").str.zfill(2)
    merged["target_place_fips"] = merged.get("target_place_fips", pd.Series(index=merged.index)).astype("string").str.zfill(5)
    merged["target_county_fips"] = merged.get("target_county_fips", pd.Series(index=merged.index)).astype("string").str.zfill(3)
    merged["override_place_geoid"] = merged["target_state_fips"].fillna("") + merged["target_place_fips"].fillna("")
    merged["override_county_geoid"] = merged["target_state_fips"].fillna("") + merged["target_county_fips"].fillna("")
    merged["override_place_jurisdiction_id"] = merged["override_place_geoid"].map(place_to_jurisdiction)
    hint_lower = merged["geometry_hint"].fillna("").astype(str).str.lower()
    localizable = merged["overlap_subtype"].notna() & (~hint_lower.str.contains("statewide", regex=False))
    has_place = merged["place_jurisdiction_id"].notna()
    # Same authority rule as the county remainder lane: a name-resolved county cannot
    # localize an overlap agency into a county it was never placed in. Without this the
    # canonicalization would newly county-anchor 2,149 overlap agencies carrying 9,618
    # motor-vehicle thefts -- overwhelmingly CHP area offices, whose office county is
    # not their patrol footprint (a CHP "area" spans several counties).
    succession_rulings = load_explicit_succession_rulings(paths, target_year=int(year))
    reviewed_succession_county = _reviewed_succession_county_anchor_mask(
        merged, succession_rulings
    )
    has_county = _valid_county_fips(merged["county_fips"]) & (
        merged["county_fips_source"]
        .astype("string")
        .isin(COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES)
        .fillna(False)
        | reviewed_succession_county
    )
    name_norm = merged.get("agency_name_std", pd.Series("", index=merged.index)).map(
        _normalize_agency_name_for_county_match
    )
    if bool(enable_county_anchoring):
        county_name_lookup = _load_county_name_lookup(paths)
        county_name = merged["county_geoid"].map(county_name_lookup).astype("string")
        has_county_agency_token = name_norm.str.contains(
            r"\b(?:COUNTY|PARISH|BOROUGH|CONSTABLE|PCT|PRECINCT)\b",
            regex=True,
            na=False,
        )
        county_name_agreement = (
            has_county
            & county_name.notna()
            & _contains_county_name(name_norm, county_name)
            & has_county_agency_token
        )
        state_police_county_subunit = _state_police_county_subunit_mask(
            merged,
            name_norm,
            has_county,
            reviewed_county_identity=reviewed_succession_county,
        )
    else:
        county_name_agreement = pd.Series(False, index=merged.index)
        state_police_county_subunit = pd.Series(False, index=merged.index)
    treatment = merged.get("final_overlap_treatment", pd.Series(index=merged.index, dtype="object")).astype("string")
    override_place_id = merged.get("target_jurisdiction_id", pd.Series(index=merged.index, dtype="object")).astype("string")
    override_place_id = override_place_id.where(override_place_id.notna(), merged["override_place_jurisdiction_id"])
    has_override_place = treatment.isin(["localize_to_place"]) & override_place_id.notna()
    has_override_county = treatment.eq("localize_to_county") & merged["override_county_geoid"].str.fullmatch(r"\d{5}").fillna(False)
    has_absorb_target = treatment.eq("absorb_into_primary_jurisdiction") & override_place_id.notna()
    has_custom_footprint = treatment.eq("localize_to_custom_footprint") & merged.get("has_custom_footprint", pd.Series(index=merged.index)).eq(True)
    registry_statewide = treatment.eq("keep_statewide_overlap")
    registry_hold = treatment.eq("exclude_or_hold")
    # The crosswalk's own statewide finding, separated from the cascade's fall-through. Both used
    # to arrive at the same terminal rung; only the first is a statement about the footprint.
    crosswalk_statewide_geometry = hint_lower.str.contains("statewide", regex=False)

    def _bool_mask(series: pd.Series) -> np.ndarray:
        return series.fillna(False).astype(bool).to_numpy()

    # ONE condition list, two outputs. `group_kind` is what the allocator spreads over;
    # `overlap_resolution_route` is HOW that answer was reached. Sharing the conditions is the
    # point: a route that disagreed with its kind would be a lie about provenance, and here it is
    # structurally impossible.
    cascade_conditions = [
        _bool_mask(has_custom_footprint),
        _bool_mask(has_absorb_target),
        _bool_mask(has_override_place),
        _bool_mask(has_override_county),
        _bool_mask(registry_statewide),
        _bool_mask(registry_hold),
        # State police first, so a "SP <county> COUNTY" agency (which fires BOTH masks)
        # gets the non-municipal spread rather than the whole-county one.
        _bool_mask(state_police_county_subunit),
        _bool_mask(county_name_agreement),
        _bool_mask(localizable & has_place),
        _bool_mask(localizable & has_county),
        _bool_mask(crosswalk_statewide_geometry),
    ]
    merged["group_kind"] = np.select(
        cascade_conditions,
        [
            "custom_footprint_overlap",
            "absorbed_overlap",
            "municipal_place_overlap",
            "county_overlap",
            "statewide_overlap",
            "statewide_overlap",
            COUNTY_NONMUNICIPAL_OVERLAP_KIND,
            "county_overlap",
            "municipal_place_overlap",
            "county_overlap",
            "statewide_overlap",
        ],
        default="statewide_overlap",
    )
    merged["overlap_resolution_route"] = np.select(
        cascade_conditions,
        [
            OVERLAP_ROUTE_REGISTRY_CUSTOM_FOOTPRINT,
            OVERLAP_ROUTE_REGISTRY_ABSORB,
            OVERLAP_ROUTE_REGISTRY_PLACE,
            OVERLAP_ROUTE_REGISTRY_COUNTY,
            OVERLAP_ROUTE_REGISTRY_STATEWIDE,
            OVERLAP_ROUTE_REGISTRY_HOLD,
            OVERLAP_ROUTE_STATE_POLICE_COUNTY,
            OVERLAP_ROUTE_COUNTY_NAME_AGREEMENT,
            OVERLAP_ROUTE_CROSSWALK_PLACE,
            OVERLAP_ROUTE_CROSSWALK_COUNTY,
            OVERLAP_ROUTE_CROSSWALK_STATEWIDE_GEOMETRY,
        ],
        default=OVERLAP_ROUTE_UNRESOLVED_NO_TARGET,
    )
    county_kinds = ["county_overlap", COUNTY_NONMUNICIPAL_OVERLAP_KIND]
    merged["group_id"] = np.where(
        merged["group_kind"].eq("custom_footprint_overlap"),
        merged["ori9"],
        np.where(
        merged["group_kind"].eq("absorbed_overlap"),
        override_place_id,
        np.where(
        merged["group_kind"].eq("municipal_place_overlap"),
        override_place_id.where(has_override_place, merged["place_jurisdiction_id"]),
        np.where(merged["group_kind"].isin(county_kinds), merged["county_geoid"], merged["state_fips"]),
    )))
    merged.loc[has_override_county, "group_id"] = merged.loc[has_override_county, "override_county_geoid"]

    if not agency_estimates.empty:
        agency_estimates["state_fips"] = agency_estimates["state_fips"].astype("string").str.zfill(2)
        agency_estimates["offense"] = agency_estimates["offense"].astype("string")
        merged = merged.merge(
            agency_estimates[
                [
                    "ori9",
                    "state_fips",
                    "offense",
                    "reported_count_current_supported",
                    "agency_adjustment_count",
                ]
            ],
            on=["ori9", "state_fips", "offense"],
            how="left",
        )
    else:
        merged["reported_count_current_supported"] = np.nan
        merged["agency_adjustment_count"] = np.nan
    merged["reported_count_current_supported"] = pd.to_numeric(
        merged["reported_count_current_supported"], errors="coerce"
    ).fillna(pd.to_numeric(merged["preferred_count"], errors="coerce")).fillna(0.0).clip(lower=0.0)
    merged["agency_adjustment_count"] = pd.to_numeric(
        merged["agency_adjustment_count"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    if _controls_are_smoothed(controls):
        risk_signals = (
            agency_risk_signals.copy()
            if agency_risk_signals is not None
            else _build_agency_risk_signals(
                paths=paths,
                year=int(year),
                fallback_estimates=agency_estimates,
            )
        )
        merged = merged.merge(
            risk_signals[["ori9", "state_fips", "offense", "risk_signal_count"]],
            on=["ori9", "state_fips", "offense"],
            how="left",
            validate="many_to_one",
        )
        merged["risk_signal_count"] = pd.to_numeric(
            merged["risk_signal_count"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)

    unsupported_custom = treatment.eq("localize_to_custom_footprint") & ~has_custom_footprint
    if bool(unsupported_custom.fillna(False).any()):
        raise ValueError(
            "localize_to_custom_footprint override has no footprint rows for the state its "
            "crosswalk link sits in, which would silently spread the agency statewide: "
            f"{merged.loc[unsupported_custom.fillna(False), ['ori9', 'state_fips']].drop_duplicates().to_dict(orient='records')}"
        )

    if bg_prior is not None and not bg_prior.empty:
        supported_counties = _supported_counties_for_bg_prior(bg_prior)
        county_mask = merged["group_kind"].isin(county_kinds)
        support_frame = pd.DataFrame(
            {
                "state_fips": merged["state_fips"].astype("string").str.zfill(2),
                "county_geoid": np.where(
                    county_mask,
                    merged["group_id"].astype("string"),
                    merged["county_geoid"].astype("string"),
                ),
            },
            index=merged.index,
        )
        unsupported_county = county_mask & ~_county_supported_mask(support_frame, supported_counties)
    else:
        unsupported_county = pd.Series(False, index=merged.index)
    county_evidence = (
        merged.loc[merged["group_kind"].isin(county_kinds)]
        .groupby(["state_fips", "offense", "group_id"], dropna=False)["reported_count_current_supported"]
        .sum()
        .rename("county_anchor_evidence_count")
        .reset_index()
    )
    merged = merged.merge(county_evidence, on=["state_fips", "offense", "group_id"], how="left")
    merged["county_anchor_evidence_count"] = pd.to_numeric(
        merged["county_anchor_evidence_count"], errors="coerce"
    ).fillna(0.0)
    rare_low_evidence = (
        merged["group_kind"].isin(county_kinds)
        & merged["offense"].astype("string").isin(COUNTY_ANCHOR_MIN_EVIDENCE_OFFENSES)
        & merged["county_anchor_evidence_count"].lt(float(COUNTY_ANCHOR_MIN_OBSERVED_OFFENSE_COUNT))
    )
    demote_to_statewide = unsupported_county | rare_low_evidence
    if bool(demote_to_statewide.any()):
        merged.loc[demote_to_statewide, "group_kind"] = "statewide_overlap"
        merged.loc[demote_to_statewide, "group_id"] = merged.loc[demote_to_statewide, "state_fips"]
    # Both demotions land on the same rung and are not the same finding, so they get different
    # routes. `rare_low_evidence` wins the overlap because it is the narrower, named rule.
    merged.loc[unsupported_county.fillna(False), "overlap_resolution_route"] = (
        OVERLAP_ROUTE_UNSUPPORTED_COUNTY
    )
    merged.loc[rare_low_evidence.fillna(False), "overlap_resolution_route"] = (
        OVERLAP_ROUTE_RARE_OFFENSE_DEMOTION
    )
    if bool(unlocated_mass):
        unlocated_row = merged["overlap_resolution_route"].astype("string").isin(UNLOCATED_ROUTES)
        merged.loc[unlocated_row, "group_kind"] = UNLOCATED_GROUP_KIND
        merged.loc[unlocated_row, "group_id"] = merged.loc[unlocated_row, "state_fips"]
    # Where the STATE's overlap control has no agency behind it at all -- no preferred observation
    # anywhere in the state's overlap layer, or a raked remainder no group claimed -- the cascade
    # invents a statewide row and puts the mass on it. That row has no footprint because it has no
    # agency; it is the purest case the bucket exists for.
    residual_kind = UNLOCATED_GROUP_KIND if bool(unlocated_mass) else "statewide_overlap"
    # Every gram of mass that reaches the residual rung without passing through an agency row in
    # `merged`, accumulated as it is created so the route table can account for all of it. Three
    # sources: a state whose overlap layer has no reported observation at all, the same for the
    # adjustment lane, and whatever the two rakes leave over at the end.
    residual_delta: dict[tuple[str, str], float] = {}

    def _record_residual(frame: pd.DataFrame, *, column: str) -> None:
        if not bool(unlocated_mass) or frame.empty:
            return
        for row in frame[["state_fips", "offense", column]].itertuples(index=False):
            key = (str(row.state_fips), str(row.offense))
            residual_delta[key] = residual_delta.get(key, 0.0) + float(getattr(row, column))

    observed_group = (
        merged.groupby(["state_fips", "offense", "group_kind", "group_id"], dropna=False)
        .agg(
            observed_raw_count=("reported_count_current_supported", "sum"),
            county_anchor_evidence_count=("county_anchor_evidence_count", "max"),
        )
        .reset_index()
    )
    observed_state = (
        observed_group.groupby(["state_fips", "offense"], dropna=False)["observed_raw_count"]
        .sum()
        .rename("observed_state_raw_total")
        .reset_index()
    )
    observed_group = observed_group.merge(
        overlap_controls[["state_fips", "offense", "state_reported_target"]],
        on=["state_fips", "offense"],
        how="inner",
    ).merge(observed_state, on=["state_fips", "offense"], how="left")
    observed_group["observed_target_count"] = np.where(
        pd.to_numeric(observed_group["observed_state_raw_total"], errors="coerce").fillna(0.0).gt(0.0),
        pd.to_numeric(observed_group["state_reported_target"], errors="coerce").fillna(0.0)
        * pd.to_numeric(observed_group["observed_raw_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(observed_group["observed_state_raw_total"], errors="coerce").fillna(1.0),
        0.0,
    )
    missing_observed = overlap_controls.merge(observed_state, on=["state_fips", "offense"], how="left")
    missing_observed = missing_observed[
        pd.to_numeric(missing_observed["observed_state_raw_total"], errors="coerce").fillna(0.0).le(0.0)
        & pd.to_numeric(missing_observed["state_reported_target"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    _record_residual(missing_observed, column="state_reported_target")
    if not missing_observed.empty:
        observed_group = pd.concat(
            [
                observed_group[
                    [
                        "state_fips",
                        "offense",
                        "group_kind",
                        "group_id",
                        "observed_raw_count",
                        "observed_target_count",
                        "county_anchor_evidence_count",
                    ]
                ],
                missing_observed.assign(
                    group_kind=residual_kind,
                    group_id=missing_observed["state_fips"],
                    observed_raw_count=0.0,
                    observed_target_count=missing_observed["state_reported_target"],
                    county_anchor_evidence_count=0.0,
                )[
                    [
                        "state_fips",
                        "offense",
                        "group_kind",
                        "group_id",
                        "observed_raw_count",
                        "observed_target_count",
                        "county_anchor_evidence_count",
                    ]
                ],
            ],
            ignore_index=True,
        )
    else:
        observed_group = observed_group[
            [
                "state_fips",
                "offense",
                "group_kind",
                "group_id",
                "observed_raw_count",
                "observed_target_count",
                "county_anchor_evidence_count",
            ]
        ].copy()
    observed_group = _coalesce_overlap_rake_groups(
        observed_group,
        raw_column="observed_raw_count",
        target_column="observed_target_count",
    )

    adjustment_group = (
        merged.groupby(["state_fips", "offense", "group_kind", "group_id"], dropna=False)
        .agg(
            adjustment_raw_count=("agency_adjustment_count", "sum"),
            county_anchor_evidence_count=("county_anchor_evidence_count", "max"),
        )
        .reset_index()
    )
    adjustment_state = (
        adjustment_group.groupby(["state_fips", "offense"], dropna=False)["adjustment_raw_count"]
        .sum()
        .rename("adjustment_state_raw_total")
        .reset_index()
    )
    adjustment_group = adjustment_group.merge(
        overlap_controls[["state_fips", "offense", "state_adjustment_target"]],
        on=["state_fips", "offense"],
        how="inner",
    ).merge(adjustment_state, on=["state_fips", "offense"], how="left")
    adjustment_group["adjustment_target_count"] = np.where(
        pd.to_numeric(adjustment_group["adjustment_state_raw_total"], errors="coerce").fillna(0.0).gt(0.0),
        pd.to_numeric(adjustment_group["state_adjustment_target"], errors="coerce").fillna(0.0)
        * pd.to_numeric(adjustment_group["adjustment_raw_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(adjustment_group["adjustment_state_raw_total"], errors="coerce").fillna(1.0),
        0.0,
    )
    missing_adjustment = overlap_controls.merge(adjustment_state, on=["state_fips", "offense"], how="left")
    missing_adjustment = missing_adjustment[
        pd.to_numeric(missing_adjustment["adjustment_state_raw_total"], errors="coerce").fillna(0.0).le(0.0)
        & pd.to_numeric(missing_adjustment["state_adjustment_target"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    _record_residual(missing_adjustment, column="state_adjustment_target")
    if not missing_adjustment.empty:
        adjustment_group = pd.concat(
            [
                adjustment_group[
                    [
                        "state_fips",
                        "offense",
                        "group_kind",
                        "group_id",
                        "adjustment_raw_count",
                        "adjustment_target_count",
                        "county_anchor_evidence_count",
                    ]
                ],
                missing_adjustment.assign(
                    group_kind=residual_kind,
                    group_id=missing_adjustment["state_fips"],
                    adjustment_raw_count=0.0,
                    adjustment_target_count=missing_adjustment["state_adjustment_target"],
                    county_anchor_evidence_count=0.0,
                )[
                    [
                        "state_fips",
                        "offense",
                        "group_kind",
                        "group_id",
                        "adjustment_raw_count",
                        "adjustment_target_count",
                        "county_anchor_evidence_count",
                    ]
                ],
            ],
            ignore_index=True,
        )
    else:
        adjustment_group = adjustment_group[
            [
                "state_fips",
                "offense",
                "group_kind",
                "group_id",
                "adjustment_raw_count",
                "adjustment_target_count",
                "county_anchor_evidence_count",
            ]
        ].copy()
    adjustment_group = _coalesce_overlap_rake_groups(
        adjustment_group,
        raw_column="adjustment_raw_count",
        target_column="adjustment_target_count",
    )

    grouped = observed_group.merge(
        adjustment_group,
        on=["state_fips", "offense", "group_kind", "group_id"],
        how="outer",
        suffixes=("_observed", "_adjustment"),
        validate="one_to_one",
    )
    for col in ["observed_raw_count", "observed_target_count", "adjustment_raw_count", "adjustment_target_count"]:
        grouped[col] = pd.to_numeric(grouped.get(col), errors="coerce").fillna(0.0)
    grouped["county_anchor_evidence_count"] = np.maximum(
        pd.to_numeric(grouped.get("county_anchor_evidence_count_observed"), errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(grouped.get("county_anchor_evidence_count_adjustment"), errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )
    seed_kinds = ["statewide_overlap"] + ([UNLOCATED_GROUP_KIND] if bool(unlocated_mass) else [])
    statewide_rows = pd.concat(
        [
            overlap_controls[["state_fips", "offense"]].assign(
                group_kind=kind, group_id=overlap_controls["state_fips"]
            )
            for kind in seed_kinds
        ],
        ignore_index=True,
    )
    grouped = pd.concat(
        [
            grouped,
            statewide_rows.assign(
                observed_raw_count=0.0,
                observed_target_count=0.0,
                adjustment_raw_count=0.0,
                adjustment_target_count=0.0,
                county_anchor_evidence_count=0.0,
            ),
        ],
        ignore_index=True,
    )
    grouped["target_count"] = grouped["observed_target_count"] + grouped["adjustment_target_count"]
    grouped = (
        grouped.groupby(["state_fips", "offense", "group_kind", "group_id"], dropna=False, as_index=False)
        .agg(
            target_count=("target_count", "sum"),
            reported_count=("observed_target_count", "sum"),
            observed_raw_count=("observed_raw_count", "sum"),
            adjustment_raw_count=("adjustment_raw_count", "sum"),
            county_anchor_evidence_count=("county_anchor_evidence_count", "max"),
        )
        .merge(overlap_controls[["state_fips", "offense", "state_target"]], on=["state_fips", "offense"], how="inner")
    )
    grouped = grouped[
        grouped["group_kind"].isin(seed_kinds)
        | pd.to_numeric(grouped["target_count"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    sums = grouped.groupby(["state_fips", "offense"], dropna=False)["target_count"].sum().reset_index(name="target_sum")
    deltas = overlap_controls[["state_fips", "offense", "state_target"]].merge(sums, on=["state_fips", "offense"], how="left")
    deltas["delta"] = deltas["state_target"] - pd.to_numeric(deltas["target_sum"], errors="coerce").fillna(0.0)
    if deltas["delta"].abs().gt(1e-8).any():
        _record_residual(deltas[deltas["delta"].abs().gt(1e-8)], column="delta")
        for row in deltas[deltas["delta"].abs().gt(1e-8)].itertuples(index=False):
            mask = (
                grouped["state_fips"].eq(row.state_fips)
                & grouped["offense"].eq(row.offense)
                & grouped["group_kind"].eq(residual_kind)
            )
            grouped.loc[mask, "target_count"] = (
                pd.to_numeric(grouped.loc[mask, "target_count"], errors="coerce").fillna(0.0) + float(row.delta)
            )
    final_sums = grouped.groupby(["state_fips", "offense"], dropna=False)["target_count"].sum().reset_index(name="target_sum")
    check = overlap_controls[["state_fips", "offense", "state_target"]].merge(final_sums, on=["state_fips", "offense"], how="left")
    max_delta = float((check["target_sum"].fillna(0.0) - check["state_target"]).abs().max() or 0.0)
    if max_delta > 1e-6:
        raise ValueError(f"Overlap targets do not partition controls; max abs delta={max_delta:.3e}")
    legacy_grouped = grouped.copy()
    risk_partition = pd.DataFrame()
    if _controls_are_smoothed(controls):
        risk_partition = _smoothed_overlap_partition(
            overlap_controls=overlap_controls,
            agency_groups=merged,
        )
        if not risk_partition.empty:
            active = pd.MultiIndex.from_frame(
                risk_partition[["state_fips", "offense"]].drop_duplicates()
            )
            current_keys = pd.MultiIndex.from_frame(grouped[["state_fips", "offense"]])
            grouped = pd.concat(
                [grouped.loc[~current_keys.isin(active)], risk_partition],
                ignore_index=True,
                sort=False,
            )
            risk_sums = (
                grouped.groupby(["state_fips", "offense"], dropna=False)["target_count"]
                .sum()
                .reset_index(name="target_sum")
            )
            risk_check = overlap_controls[["state_fips", "offense", "state_target"]].merge(
                risk_sums, on=["state_fips", "offense"], how="left"
            )
            risk_max_delta = float(
                (risk_check["target_sum"].fillna(0.0) - risk_check["state_target"]).abs().max()
                or 0.0
            )
            if risk_max_delta > 1e-6:
                raise ValueError(
                    "Smoothed overlap targets do not partition controls; "
                    f"max abs delta={risk_max_delta:.3e}"
                )
    grouped["target_count"] = pd.to_numeric(grouped["target_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # The custom-footprint lane is raked to each ORI's OWN admitted control here, after every
    # other partition rule has run, so the rake is the last word on footprint mass and the
    # state control is still partitioned exactly. See the block above `CUSTOM_FOOTPRINT_STATUS_*`.
    footprint_eligible = (
        set(
            overrides.loc[
                overrides["final_overlap_treatment"]
                .astype("string")
                .eq("localize_to_custom_footprint"),
                "ori9",
            ]
            .astype("string")
            .str.upper()
        )
        | set(custom_footprints.get("canonical_target_ori", pd.Series(dtype="string")).dropna().astype("string").str.upper())
        if not overrides.empty
        else None
    )
    footprint_partition = (
        _footprint_ledger_partition(
            custom_footprints=custom_footprints,
            ledger=_load_level_lane_overlap_mass(paths, year=int(year)),
            overlap_controls=overlap_controls,
            bg_prior=bg_prior,
            eligible_oris=footprint_eligible,
        )
        if not custom_footprints.empty
        else pd.DataFrame(columns=list(FOOTPRINT_MASS_CONSERVATION_COLUMNS))
    )
    footprint_routes = _footprint_unlocated_routes(footprint_partition)
    if not footprint_partition.empty:
        grouped = _apply_footprint_ledger_rake(
            grouped,
            partition=footprint_partition,
            overlap_controls=overlap_controls,
            residual_kind=residual_kind,
        )
        legacy_grouped = _apply_footprint_ledger_rake(
            legacy_grouped[["state_fips", "offense", "group_kind", "group_id", "target_count"]],
            partition=footprint_partition,
            overlap_controls=overlap_controls,
            residual_kind=residual_kind,
        )
    out = grouped[["state_fips", "offense", "group_kind", "group_id", "target_count"]].copy()
    out.attrs["footprint_mass_partition"] = footprint_partition
    if bool(unlocated_mass):
        legacy_detail = _unlocated_route_detail(
            merged=merged,
            grouped=legacy_grouped[
                ["state_fips", "offense", "group_kind", "group_id", "target_count"]
            ],
            overlap_controls=overlap_controls,
            residual_delta=residual_delta,
            extra_routes=footprint_routes,
            rescaled_keys=footprint_partition,
        )
        if _controls_are_smoothed(controls) and not risk_partition.empty:
            active = pd.MultiIndex.from_frame(
                risk_partition[["state_fips", "offense"]].drop_duplicates()
            )
            controls_keys = pd.MultiIndex.from_frame(overlap_controls[["state_fips", "offense"]])
            merged_keys = pd.MultiIndex.from_frame(merged[["state_fips", "offense"]])
            risk_detail = _smoothed_unlocated_route_detail(
                merged=merged.loc[merged_keys.isin(active)],
                grouped=out,
                overlap_controls=overlap_controls.loc[controls_keys.isin(active)],
                extra_routes=footprint_routes,
                rescaled_keys=footprint_partition,
            )
            legacy_keys = pd.MultiIndex.from_frame(legacy_detail[["state_fips", "offense"]])
            out.attrs["unlocated_mass"] = pd.concat(
                [legacy_detail.loc[~legacy_keys.isin(active)], risk_detail],
                ignore_index=True,
            ).sort_values(["state_fips", "offense"], kind="mergesort").reset_index(drop=True)
        else:
            out.attrs["unlocated_mass"] = legacy_detail
    return out


def _unlocated_route_detail(
    *,
    merged: pd.DataFrame,
    grouped: pd.DataFrame,
    overlap_controls: pd.DataFrame,
    residual_delta: dict[tuple[str, str], float],
    extra_routes: pd.DataFrame | None = None,
    rescaled_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The per-jurisdiction-offense unlocated bucket, split by the route that produced it.

    The split re-runs the SAME two state-level rakes the group targets were built with, at route
    granularity instead of group granularity. Both rakes are linear in the raw count, so summing
    the routes of a group reproduces that group's target exactly -- asserted below rather than
    assumed, because a route table that did not add up would be a decorative audit.

    The state residual (a control with no agency observation behind it anywhere, plus whatever
    rounding the two rakes left) is attributed to `unresolved_unattributed_state_residual` in
    full. It is not apportioned across routes: it is not any agency's mass.
    """
    keys = ["state_fips", "offense"]
    lane_specs = (
        ("reported_count_current_supported", "state_reported_target", "observed"),
        ("agency_adjustment_count", "state_adjustment_target", "adjustment"),
    )
    unlocated_rows = merged[merged["group_kind"].astype("string").eq(UNLOCATED_GROUP_KIND)]
    route_raw = (
        unlocated_rows.groupby([*keys, "overlap_resolution_route"], dropna=False)
        .agg(**{raw: (raw, "sum") for raw, _target, _label in lane_specs})
        .reset_index()
        if not unlocated_rows.empty
        else pd.DataFrame(columns=[*keys, "overlap_resolution_route", *(raw for raw, _t, _l in lane_specs)])
    )
    route_raw["route_target_count"] = 0.0
    for raw_col, target_col, _label in lane_specs:
        # The rake denominator is the state's raw total over EVERY group, located or not: that is
        # the denominator the group targets used, so reusing it is what makes the routes add up.
        state_raw = (
            merged.groupby(keys, dropna=False)[raw_col].sum().rename("_state_raw").reset_index()
        )
        route_raw = route_raw.merge(state_raw, on=keys, how="left").merge(
            overlap_controls[[*keys, target_col]], on=keys, how="left"
        )
        denom = pd.to_numeric(route_raw["_state_raw"], errors="coerce").fillna(0.0)
        route_raw["route_target_count"] = route_raw["route_target_count"] + np.where(
            denom.gt(0.0),
            pd.to_numeric(route_raw[target_col], errors="coerce").fillna(0.0)
            * pd.to_numeric(route_raw.get(raw_col), errors="coerce").fillna(0.0)
            / denom.replace(0.0, np.nan),
            0.0,
        )
        route_raw = route_raw.drop(columns=["_state_raw", target_col])
    residual = pd.DataFrame(
        [
            {
                "state_fips": state_fips,
                "offense": offense,
                "overlap_resolution_route": OVERLAP_ROUTE_UNATTRIBUTED_RESIDUAL,
                "route_target_count": float(delta),
            }
            for (state_fips, offense), delta in sorted(residual_delta.items())
        ],
        columns=[*keys, "overlap_resolution_route", "route_target_count"],
    )
    # Mass a NON-rake lane moved into the bucket (today: the per-ORI footprint rake) arrives as
    # a ready-made route total, because it was never apportioned by either state rake.
    extra = (
        extra_routes[[*keys, "overlap_resolution_route", "route_target_count"]].copy()
        if extra_routes is not None and not extra_routes.empty
        else pd.DataFrame(columns=[*keys, "overlap_resolution_route", "route_target_count"])
    )
    route_frames = [
        frame
        for frame in (
            route_raw[[*keys, "overlap_resolution_route", "route_target_count"]],
            residual,
            extra,
        )
        if not frame.empty
    ]
    routes = (
        pd.concat(route_frames, ignore_index=True)
        if route_frames
        else pd.DataFrame(columns=[*keys, "overlap_resolution_route", "route_target_count"])
    )
    routes["route_target_count"] = (
        pd.to_numeric(routes["route_target_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    wide = (
        routes.pivot_table(
            index=keys,
            columns="overlap_resolution_route",
            values="route_target_count",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()
        if not routes.empty
        else pd.DataFrame(columns=keys)
    )
    out = overlap_controls[[*keys, "state_target"]].rename(columns={"state_target": "control_count"}).copy()
    out = out.merge(
        grouped[grouped["group_kind"].eq(UNLOCATED_GROUP_KIND)][[*keys, "target_count"]].rename(
            columns={"target_count": "unlocated_count"}
        ),
        on=keys,
        how="left",
    ).merge(wide, on=keys, how="left")
    out["unlocated_count"] = pd.to_numeric(out["unlocated_count"], errors="coerce").fillna(0.0)
    out["control_count"] = pd.to_numeric(out["control_count"], errors="coerce").fillna(0.0)
    for route in sorted(UNLOCATED_ROUTES):
        column = f"unlocated_count_{route}"
        out[column] = (
            pd.to_numeric(out[route], errors="coerce").fillna(0.0)
            if route in out.columns
            else 0.0
        )
    out = out.drop(columns=[route for route in UNLOCATED_ROUTES if route in out.columns], errors="ignore")
    out = _reconcile_unlocated_routes(out, rescaled_keys)
    route_sum = out[[f"unlocated_count_{route}" for route in sorted(UNLOCATED_ROUTES)]].sum(axis=1)
    route_drift = route_sum - out["unlocated_count"]
    drift = float(route_drift.abs().max() or 0.0)
    if drift > 1e-6:
        recorded_residual = pd.Series(
            [
                float(residual_delta.get((str(row.state_fips), str(row.offense)), 0.0))
                for row in out[["state_fips", "offense"]].itertuples(index=False)
            ],
            index=out.index,
            dtype=float,
        )
        route_columns = [f"unlocated_count_{route}" for route in sorted(UNLOCATED_ROUTES)]
        sample_frame = (
            out.assign(
                route_sum=route_sum,
                route_drift=route_drift,
                recorded_residual=recorded_residual,
            )
            .loc[
                route_drift.abs().gt(1e-6),
                [
                    "state_fips",
                    "offense",
                    "unlocated_count",
                    "route_sum",
                    "route_drift",
                    "recorded_residual",
                    *route_columns,
                ],
            ]
            .sort_values("route_drift", key=lambda values: values.abs(), ascending=False)
            .head(10)
        )
        sample = sample_frame.to_dict(orient="records")
        bad_pairs = pd.MultiIndex.from_frame(sample_frame[["state_fips", "offense"]])
        grouped_pairs = pd.MultiIndex.from_frame(grouped[["state_fips", "offense"]])
        grouped_sample = (
            grouped.loc[
                grouped_pairs.isin(bad_pairs)
                & grouped["group_kind"].astype("string").eq(UNLOCATED_GROUP_KIND)
            ]
            .head(20)
            .to_dict(orient="records")
        )
        route_pairs = pd.MultiIndex.from_frame(routes[["state_fips", "offense"]])
        unexpected_route_sample = (
            routes.loc[
                route_pairs.isin(bad_pairs)
                & ~routes["overlap_resolution_route"].astype("string").isin(UNLOCATED_ROUTES)
            ]
            .sort_values("route_target_count", key=lambda values: values.abs(), ascending=False)
            .head(20)
            .to_dict(orient="records")
        )
        raise ValueError(
            "unlocated route detail does not reconstruct the bucket; "
            f"max abs delta={drift:.3e}; sample={sample}; "
            f"unexpected_route_sample={unexpected_route_sample}; grouped_sample={grouped_sample}"
        )
    out["jurisdiction_id"] = out["state_fips"].astype("string") + ":" + STATE_OVERLAP_TYPE
    out["jurisdiction_type"] = STATE_OVERLAP_TYPE
    out["unlocated_share_of_control"] = np.where(
        out["control_count"].gt(0.0), out["unlocated_count"] / out["control_count"], 0.0
    )
    return out[list(UNLOCATED_MASS_COLUMNS)].sort_values(
        ["state_fips", "offense"], kind="mergesort"
    ).reset_index(drop=True)


def _nonmunicipal_bg_exposure_share(bg_crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Per block group, the population share owned by the state non-municipal remainder.

    `allocation_share` in the block-group crosswalk is a 2020-population share and sums to 1
    per block group across jurisdictions, so the remainder's share is exactly the fraction of
    the block group's residents that no agency-bearing municipality covers.
    """
    columns = ["state_fips", "bg_id", "nonmunicipal_share"]
    if bg_crosswalk.empty:
        return pd.DataFrame(columns=columns)
    frame = bg_crosswalk.copy()
    key = "block_group_geoid" if "block_group_geoid" in frame.columns else "bg_id"
    remainder = frame["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
    out = (
        frame.loc[remainder]
        .assign(
            state_fips=lambda df: df["state_fips"].astype("string").str.zfill(2),
            bg_id=lambda df: df[key].astype("string").str.zfill(12),
            nonmunicipal_share=lambda df: pd.to_numeric(
                df["allocation_share"], errors="coerce"
            ).fillna(0.0),
        )
        .groupby(["state_fips", "bg_id"], dropna=False, as_index=False)["nonmunicipal_share"]
        .sum()
    )
    return out[columns]


def _custom_footprint_component_shares(merged: pd.DataFrame) -> pd.DataFrame:
    """Component share for the custom-footprint overlap lane, per `weight_share_basis`.

    Extracted so the arithmetic is testable without a targets frame. Adds
    `footprint_activity_weight`, the two pool totals, and `component_share`; the share sums to
    1 inside every allocation pool that carries any weight. Ordinary footprints retain their
    `(state, ori, offense)` pool. A reviewed service-wide footprint instead pools all destination
    states under its one source-state target.
    """
    out = merged.copy()
    weight_share = pd.to_numeric(out["weight_share"], errors="coerce").fillna(0.0)
    allocation_scope = out.get(
        "allocation_scope", pd.Series("source_state", index=out.index, dtype="string")
    ).astype("string").fillna("source_state")
    service_wide = allocation_scope.eq("service_wide")
    resident_basis = out["weight_share_basis"].astype("string").eq(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT
    )
    service_basis = out["weight_share_basis"].astype("string").eq(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_SERVICE_AREA_PRIOR
    )
    invalid_service_basis = service_wide & ~service_basis
    if bool(invalid_service_basis.any()):
        raise ValueError(
            "service-wide custom footprints require weight_share_basis=service_area_prior"
        )
    invalid_ordinary_basis = ~service_wide & service_basis
    if bool(invalid_ordinary_basis.any()):
        raise ValueError("service_area_prior is only valid for service-wide custom footprints")
    if bool(service_wide.any()) and "bg_land_area_coverage_share" not in out.columns:
        raise ValueError(
            "service-wide custom footprints require bg_land_area_coverage_share"
        )
    land_coverage = pd.to_numeric(
        out.get(
            "bg_land_area_coverage_share",
            pd.Series(np.nan, index=out.index, dtype=float),
        ),
        errors="coerce",
    )
    invalid_land_coverage = service_wide & (
        land_coverage.isna()
        | ~np.isfinite(land_coverage)
        | land_coverage.le(0.0)
        | land_coverage.gt(1.0 + 1e-9)
    )
    if bool(invalid_land_coverage.any()):
        raise ValueError(
            "service-wide bg_land_area_coverage_share must be finite and lie in (0, 1]"
        )
    ordinary_resident = resident_basis & ~service_wide
    if bool(ordinary_resident.any()) and (
        "bg_responsibility_population_coverage_share" not in out.columns
    ):
        raise ValueError(
            "ordinary resident custom footprints require "
            "bg_responsibility_population_coverage_share"
        )
    responsibility_fraction = pd.to_numeric(
        out.get(
            "bg_responsibility_population_coverage_share",
            pd.Series(np.nan, index=out.index, dtype=float),
        ),
        errors="coerce",
    )
    invalid_responsibility_fraction = ordinary_resident & (
        responsibility_fraction.isna()
        | ~np.isfinite(responsibility_fraction)
        | responsibility_fraction.le(0.0)
        | responsibility_fraction.gt(1.0 + 1e-9)
    )
    if bool(invalid_responsibility_fraction.any()):
        raise ValueError(
            "ordinary resident bg_responsibility_population_coverage_share must be "
            "finite and lie in (0, 1]"
        )
    coverage_weight = weight_share.copy()
    coverage_weight = coverage_weight.where(~ordinary_resident, responsibility_fraction)
    coverage_weight = coverage_weight.where(~service_wide, land_coverage)
    out["footprint_activity_weight"] = (
        pd.to_numeric(out["bg_weight"], errors="coerce").fillna(0.0) * coverage_weight
    ).where(resident_basis | service_basis, 0.0)
    out["bg_responsibility_population_coverage_share"] = responsibility_fraction.where(
        ordinary_resident
    )
    if "responsibility_fraction_basis" not in out.columns:
        out["responsibility_fraction_basis"] = pd.NA
    missing_responsibility_basis = ordinary_resident & (
        out["responsibility_fraction_basis"].astype("string").fillna("").str.strip().eq("")
    )
    if bool(missing_responsibility_basis.any()):
        raise ValueError(
            "ordinary resident custom footprints require responsibility_fraction_basis"
        )
    canonical = out.get(
        "canonical_target_ori", out.get("ori9", pd.Series("", index=out.index))
    ).astype("string")
    source_state = out.get("source_state_fips", out["state_fips"]).astype("string").str.zfill(2)
    out["_footprint_pool_id"] = np.where(
        service_wide,
        "service|" + source_state + "|" + canonical + "|" + out["offense"].astype("string"),
        "state|"
        + out["state_fips"].astype("string").str.zfill(2)
        + "|"
        + out["ori9"].astype("string")
        + "|"
        + out["offense"].astype("string"),
    )
    pool_cols = ["_footprint_pool_id"]
    totals = (
        out.groupby(pool_cols, dropna=False)
        .agg(
            footprint_activity_total=("footprint_activity_weight", "sum"),
            footprint_weight_total=("weight_share", "sum"),
        )
        .reset_index()
    )
    out = out.merge(totals, on=pool_cols, how="left")
    resident_basis = out["weight_share_basis"].astype("string").eq(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT
    )
    activity_total = pd.to_numeric(out["footprint_activity_total"], errors="coerce").fillna(0.0)
    invalid_service_total = service_wide & activity_total.le(0.0)
    if bool(invalid_service_total.any()):
        raise ValueError(
            "service-wide custom footprint has no positive area-covered BG prior weight"
        )
    weight_total = pd.to_numeric(out["footprint_weight_total"], errors="coerce").fillna(0.0)
    verbatim_share = np.where(
        weight_total > 0,
        pd.to_numeric(out["weight_share"], errors="coerce").fillna(0.0)
        / np.where(weight_total > 0, weight_total, 1.0),
        0.0,
    )
    # A resident-basis footprint whose whole support carries no model signal keeps its existing
    # covered-population fallback. Service-wide pools fail above rather than substituting a
    # different geographic basis.
    out["component_share"] = np.where(
        (resident_basis | service_basis).to_numpy() & (activity_total.to_numpy() > 0),
        pd.to_numeric(out["footprint_activity_weight"], errors="coerce").fillna(0.0)
        / np.where(activity_total > 0, activity_total, 1.0),
        verbatim_share,
    )
    return out.drop(columns=["_footprint_pool_id"])


SERVICE_STATE_TRANSFER_COLUMNS = (
    "year",
    "service_scope_id",
    "canonical_target_ori",
    "offense",
    "source_state_fips",
    "allocation_state_fips",
    "source_target_count",
    "allocation_count",
    "allocation_share",
    "route_reason",
)


def _build_service_state_transfer_ledger(
    components: pd.DataFrame,
    *,
    year: int,
    service_unlocated: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize final service-wide component mass by source and destination state."""
    if components.empty or "service_scope_id" not in components.columns:
        return pd.DataFrame(columns=list(SERVICE_STATE_TRANSFER_COLUMNS))
    frame = components[components["service_scope_id"].astype("string").notna()].copy()
    frame = frame[frame["service_scope_id"].astype("string").str.strip().ne("")]
    if frame.empty:
        return pd.DataFrame(columns=list(SERVICE_STATE_TRANSFER_COLUMNS))
    frame["source_state_fips"] = frame["source_state_fips"].astype("string").str.zfill(2)
    frame["allocation_state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
    frame["component_count"] = pd.to_numeric(
        frame["component_count"], errors="coerce"
    ).fillna(0.0)
    keys = [
        "service_scope_id",
        "canonical_target_ori",
        "offense",
        "source_state_fips",
    ]
    ledger = (
        frame.groupby([*keys, "allocation_state_fips"], dropna=False, as_index=False)[
            "component_count"
        ]
        .sum()
        .rename(columns={"component_count": "allocation_count"})
    )
    ledger = ledger[ledger["allocation_count"].gt(0.0)].copy()
    ledger["route_reason"] = pd.NA
    if service_unlocated is not None and not service_unlocated.empty:
        terminal = service_unlocated[
            [
                "service_scope_id",
                "canonical_target_ori",
                "offense",
                "source_state_fips",
                "unlocated_count",
                "route_reason",
            ]
        ].copy()
        terminal["allocation_state_fips"] = pd.NA
        terminal = terminal.rename(columns={"unlocated_count": "allocation_count"})
        ledger = pd.concat(
            [
                ledger,
                terminal[
                    [
                        *keys,
                        "allocation_state_fips",
                        "allocation_count",
                        "route_reason",
                    ]
                ],
            ],
            ignore_index=True,
        )
    declared = frame[keys + ["source_target_count"]].drop_duplicates()
    duplicate_declared = declared.duplicated(keys, keep=False)
    if bool(duplicate_declared.any()):
        raise ValueError(
            "service-wide components disagree on their canonical source target: "
            + str(declared.loc[duplicate_declared].to_dict(orient="records"))
        )
    totals = declared
    ledger = ledger.merge(totals, on=keys, how="left", validate="many_to_one")
    allocated = ledger.groupby(keys, dropna=False)["allocation_count"].transform("sum")
    mismatch = ~np.isclose(allocated, ledger["source_target_count"], atol=1e-6)
    if bool(mismatch.any()):
        raise ValueError(
            "service-wide allocation does not conserve its canonical source target: "
            + str(ledger.loc[mismatch].head(20).to_dict(orient="records"))
        )
    ledger["allocation_share"] = np.where(
        ledger["source_target_count"].gt(0.0),
        ledger["allocation_count"] / ledger["source_target_count"],
        0.0,
    )
    ledger.insert(0, "year", int(year))
    return ledger[list(SERVICE_STATE_TRANSFER_COLUMNS)].sort_values(
        [
            "service_scope_id",
            "canonical_target_ori",
            "offense",
            "allocation_state_fips",
            "route_reason",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _scale_components_by_source_state(
    components: pd.DataFrame,
    calibration_ratios: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Scale final component mass by its reporting source state."""
    out = components.copy()
    geographic_state = out["state_fips"].astype("string").str.zfill(2)
    if "source_state_fips" in out.columns:
        source_state = out["source_state_fips"].astype("string").str.strip()
        source_state = source_state.where(
            source_state.notna() & source_state.ne(""), geographic_state
        )
    else:
        source_state = geographic_state
    out["source_state_fips"] = source_state.astype("string").str.zfill(2)
    ratio = np.array(
        [
            float(calibration_ratios.get((str(state), str(offense)), 1.0))
            for state, offense in zip(out["source_state_fips"], out["offense"], strict=True)
        ],
        dtype=float,
    )
    if not bool(np.isfinite(ratio).all()) or bool((ratio < 0.0).any()):
        raise ValueError("source-state calibration ratios must be finite and non-negative")
    out["component_count"] = (
        pd.to_numeric(out["component_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        * ratio
    )
    return out


def _build_footprint_mass_conservation(
    components: pd.DataFrame, partition: pd.DataFrame | None
) -> pd.DataFrame:
    """Per-ORI proof that the footprint layer placed exactly the mass it was raked to.

    Built from the SAME final component table the published counts are summed from, so it is a
    measurement of the surface and not a restatement of the intent. A suppressed duplicate and a
    footprint with no placeable support must place nothing; everything else must place its own
    ledger mass on the published control surface.
    """
    columns = list(FOOTPRINT_MASS_CONSERVATION_COLUMNS)
    if partition is None or partition.empty:
        return pd.DataFrame(columns=columns)
    out = partition.copy()
    footprint = components[
        components["jurisdiction_type"].astype("string").eq("custom_footprint_overlap_layer")
    ]
    if footprint.empty:
        placed = pd.DataFrame(columns=["ori9", "offense", "placed_mass"])
    else:
        placed = (
            footprint.assign(
                ori9=footprint["jurisdiction_id"].astype("string").str.upper(),
                offense=footprint["offense"].astype("string"),
            )
            .groupby(["ori9", "offense"], dropna=False, as_index=False)["component_count"]
            .sum()
            .rename(columns={"component_count": "placed_mass"})
        )
    out = out.drop(columns=["placed_mass", "placed_minus_expected", "relative_error"]).merge(
        placed, on=["ori9", "offense"], how="left"
    )
    out["placed_mass"] = pd.to_numeric(out["placed_mass"], errors="coerce").fillna(0.0)
    out["placed_minus_expected"] = out["placed_mass"] - out["expected_placed_mass"]
    expected = pd.to_numeric(out["expected_placed_mass"], errors="coerce").fillna(0.0)
    out["relative_error"] = np.where(
        expected.gt(0.0),
        out["placed_minus_expected"].abs() / expected.where(expected.gt(0.0), 1.0),
        np.where(out["placed_mass"].abs().gt(1e-9), np.inf, 0.0),
    )
    # A footprint the rake refused to place must have placed nothing at all; expected is zero
    # there, so any placed mass is an infinite relative error and blocks on its own.
    return out[columns].sort_values(
        ["state_fips", "offense", "ori9"], kind="mergesort"
    ).reset_index(drop=True)


def _footprint_conservation_by_ori(conservation: pd.DataFrame) -> pd.DataFrame:
    """The conservation table at the granularity the release gate is stated in: one ORI.

    An ORI's control is one control across the seven offences; a tenth of a count moved between
    two of them by the envelope is not a placement failure, and summing first is what makes the
    5% band mean what it says.
    """
    columns = ["ori9", "state_fips", "expected_placed_mass", "placed_mass", "relative_error"]
    if conservation.empty:
        return pd.DataFrame(columns=columns)
    out = (
        conservation.groupby(["ori9", "state_fips"], dropna=False, as_index=False)
        .agg(
            expected_placed_mass=("expected_placed_mass", "sum"),
            placed_mass=("placed_mass", "sum"),
        )
    )
    expected = pd.to_numeric(out["expected_placed_mass"], errors="coerce").fillna(0.0)
    placed = pd.to_numeric(out["placed_mass"], errors="coerce").fillna(0.0)
    out["relative_error"] = np.where(
        expected.gt(0.0),
        (placed - expected).abs() / expected.where(expected.gt(0.0), 1.0),
        np.where(placed.abs().gt(1e-9), np.inf, 0.0),
    )
    return out[columns]


def _component_count_tables(
    components: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return expected and custom-footprint count tables from final components."""
    keys = ["state_fips", "bg_id", "tract_id"]
    counts = (
        components.groupby([*keys, "offense"], dropna=False)["component_count"]
        .sum()
        .reset_index()
        .pivot_table(
            index=keys,
            columns="offense",
            values="component_count",
            fill_value=0.0,
        )
        .reset_index()
        .rename(columns={offense: _expected_count_col(offense) for offense in OFFENSES_7})
    )
    footprint = components[
        components["jurisdiction_type"].astype("string").eq(
            "custom_footprint_overlap_layer"
        )
    ]
    footprint_counts = (
        footprint.groupby([*keys, "offense"], dropna=False)["component_count"]
        .sum()
        .reset_index()
        .pivot_table(
            index=keys,
            columns="offense",
            values="component_count",
            fill_value=0.0,
        )
        .reset_index()
        if not footprint.empty
        else pd.DataFrame(columns=keys)
    )
    footprint_counts = footprint_counts.rename(
        columns={offense: _footprint_derived_count_col(offense) for offense in OFFENSES_7}
    )
    for offense in OFFENSES_7:
        column = _footprint_derived_count_col(offense)
        if column not in footprint_counts.columns:
            footprint_counts[column] = 0.0
    return counts, footprint_counts


def _build_overlap_allocations(
    *,
    paths: RepoPaths,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    controls: pd.DataFrame,
    year: int,
    enable_county_anchoring: bool = True,
    agency_estimates: pd.DataFrame | None = None,
    unlocated_mass: bool = False,
) -> pd.DataFrame:
    targets = _build_overlap_group_targets(
        paths=paths,
        controls=controls,
        year=year,
        enable_county_anchoring=bool(enable_county_anchoring),
        bg_prior=bg_prior,
        agency_estimates=agency_estimates,
        bg_crosswalk=bg_crosswalk,
        unlocated_mass=bool(unlocated_mass),
    )
    unlocated = targets.attrs.get("unlocated_mass")
    footprint_partition = targets.attrs.get("footprint_mass_partition")
    empty_components = pd.DataFrame(
        columns=[
            "state_fips",
            "source_state_fips",
            "bg_id",
            "tract_id",
            "jurisdiction_id",
            "jurisdiction_type",
            "offense",
            "component_count",
            "service_scope_id",
            "canonical_target_ori",
            "source_target_count",
        ]
    )
    if unlocated is not None:
        empty_components.attrs["unlocated_mass"] = unlocated
    if footprint_partition is not None:
        empty_components.attrs["footprint_mass_partition"] = footprint_partition
    if targets.empty:
        return empty_components

    out_frames: list[pd.DataFrame] = []
    exclusive_displacement = _build_allocation_exclusive_footprint_displacement(
        overrides=_load_overlap_footprint_overrides(paths),
        custom_footprints=_load_overlap_custom_footprints(paths),
        concurrent_jurisdiction_carveouts=_load_concurrent_jurisdiction_carveouts(paths),
        primary_response_policies=_load_primary_service_response_policies(paths),
    )

    def _exclude_exclusive_primary_ground(
        frame: pd.DataFrame, *, weight_col: str
    ) -> pd.DataFrame:
        """Remove tribal-primary exposure from county-policing overlap pools."""
        if frame.empty or exclusive_displacement.empty:
            return frame
        out = frame.copy()
        out["_state_key"] = out["state_fips"].astype("string").str.zfill(2)
        out["_bg_key"] = out["bg_id"].astype("string").str.zfill(12)
        displacement = exclusive_displacement.rename(
            columns={"state_fips": "_state_key", "bg_id": "_bg_key"}
        )
        out = out.merge(displacement, on=["_state_key", "_bg_key"], how="left")
        keep = 1.0 - pd.to_numeric(
            out["displaced_share"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        out[weight_col] = (
            pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0) * keep
        )
        return out.drop(
            columns=["_state_key", "_bg_key", "displaced_share"], errors="ignore"
        )

    place_targets = targets[targets["group_kind"].eq("municipal_place_overlap")].copy()
    if not place_targets.empty:
        place_crosswalk = bg_crosswalk[bg_crosswalk["jurisdiction_type"].eq("municipal")].rename(columns={"block_group_geoid": "bg_id"}).copy()
        merged = bg_prior.merge(place_crosswalk, on=["bg_id", "state_fips"], how="inner")
        merged = merged.merge(
            place_targets.rename(columns={"group_id": "jurisdiction_id"})[["state_fips", "jurisdiction_id", "offense", "target_count"]],
            on=["state_fips", "jurisdiction_id", "offense"],
            how="inner",
        )
        merged["component_activity_weight"] = (
            pd.to_numeric(merged["bg_weight"], errors="coerce").fillna(0.0)
            * pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0)
        )
        sums = (
            merged.groupby(["state_fips", "jurisdiction_id", "offense"], dropna=False)
            .agg(activity_total=("component_activity_weight", "sum"), allocation_total=("allocation_share", "sum"))
            .reset_index()
        )
        merged = merged.merge(sums, on=["state_fips", "jurisdiction_id", "offense"], how="left")
        denom_activity = pd.to_numeric(merged["activity_total"], errors="coerce").fillna(0.0)
        denom_alloc = pd.to_numeric(merged["allocation_total"], errors="coerce").fillna(0.0)
        numer_activity = pd.to_numeric(merged["component_activity_weight"], errors="coerce").fillna(0.0)
        raw_alloc = pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0)
        merged["component_share"] = np.where(
            denom_activity > 0,
            numer_activity / denom_activity,
            np.where(denom_alloc > 0, raw_alloc / denom_alloc, 0.0),
        )
        merged["component_count"] = pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0) * merged["component_share"]
        merged["jurisdiction_type"] = "localized_overlap_place_layer"
        out_frames.append(merged[["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"]].copy())

    county_targets = targets[targets["group_kind"].eq("county_overlap")].copy()
    if not county_targets.empty:
        county_bg = bg_prior[["state_fips", "bg_id", "tract_id", "offense", "bg_weight"]].copy()
        county_bg = _exclude_exclusive_primary_ground(county_bg, weight_col="bg_weight")
        county_bg["group_id"] = county_bg["state_fips"].astype(str).str.zfill(2) + county_bg["tract_id"].astype(str).str.slice(2, 5)
        merged = county_bg.merge(county_targets[["state_fips", "group_id", "offense", "target_count"]], on=["state_fips", "group_id", "offense"], how="inner")
        totals = (
            merged.groupby(["state_fips", "group_id", "offense"], dropna=False)["bg_weight"]
            .sum()
            .rename("group_bg_weight_total")
            .reset_index()
        )
        merged = merged.merge(totals, on=["state_fips", "group_id", "offense"], how="left")
        denom = pd.to_numeric(merged["group_bg_weight_total"], errors="coerce").fillna(0.0)
        merged["component_share"] = np.where(
            denom > 0,
            pd.to_numeric(merged["bg_weight"], errors="coerce").fillna(0.0) / denom,
            0.0,
        )
        merged["component_count"] = pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0) * merged["component_share"]
        merged["jurisdiction_id"] = merged["group_id"]
        merged["jurisdiction_type"] = COUNTY_OVERLAP_TYPE
        out_frames.append(merged[["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"]].copy())

    county_nonmunicipal_targets = targets[
        targets["group_kind"].eq(COUNTY_NONMUNICIPAL_OVERLAP_KIND)
    ].copy()
    if not county_nonmunicipal_targets.empty:
        county_bg = bg_prior[["state_fips", "bg_id", "tract_id", "offense", "bg_weight"]].copy()
        county_bg["group_id"] = county_bg["state_fips"].astype(str).str.zfill(2) + county_bg["tract_id"].astype(str).str.slice(2, 5)
        exposure = _nonmunicipal_bg_exposure_share(bg_crosswalk)
        # Normalize the join keys on BOTH sides. A silent key-dtype mismatch here would leave
        # every non-municipal share at 0, the fallback below would fire everywhere, and the fix
        # would become an invisible no-op -- so the join is also asserted to have matched.
        county_bg["_state_key"] = county_bg["state_fips"].astype("string").str.zfill(2)
        county_bg["_bg_key"] = county_bg["bg_id"].astype("string").str.zfill(12)
        exposure = exposure.rename(columns={"state_fips": "_state_key", "bg_id": "_bg_key"})
        county_bg = county_bg.merge(exposure, on=["_state_key", "_bg_key"], how="left")
        county_bg["nonmunicipal_share"] = (
            pd.to_numeric(county_bg["nonmunicipal_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
        if not exposure.empty and not county_bg["nonmunicipal_share"].gt(0.0).any():
            raise ValueError(
                "state-police non-municipal county spread matched no block groups against the "
                "block-group crosswalk's non-municipal remainder support; the join keys are wrong"
            )
        county_bg = county_bg.drop(columns=["_state_key", "_bg_key"], errors="ignore")
        merged = county_bg.merge(
            county_nonmunicipal_targets[["state_fips", "group_id", "offense", "target_count"]],
            on=["state_fips", "group_id", "offense"],
            how="inner",
        )
        merged["nonmunicipal_weight"] = (
            pd.to_numeric(merged["bg_weight"], errors="coerce").fillna(0.0)
            * merged["nonmunicipal_share"]
        )
        merged = _exclude_exclusive_primary_ground(
            merged, weight_col="nonmunicipal_weight"
        )
        totals = (
            merged.groupby(["state_fips", "group_id", "offense"], dropna=False)
            .agg(
                nonmunicipal_total=("nonmunicipal_weight", "sum"),
                activity_total=("bg_weight", "sum"),
            )
            .reset_index()
        )
        merged = merged.merge(totals, on=["state_fips", "group_id", "offense"], how="left")
        nonmunicipal_total = pd.to_numeric(merged["nonmunicipal_total"], errors="coerce").fillna(0.0)
        activity_total = pd.to_numeric(merged["activity_total"], errors="coerce").fillna(0.0)
        # Fail-SAFE, not fail-open: a county with no non-municipal exposure at all (VA
        # independent cities, RI, Baltimore city) keeps the plain whole-county spread rather
        # than stranding the state police's mass on an empty support set.
        merged["component_share"] = np.where(
            nonmunicipal_total > 0,
            pd.to_numeric(merged["nonmunicipal_weight"], errors="coerce").fillna(0.0)
            / nonmunicipal_total.where(nonmunicipal_total > 0, 1.0),
            np.where(
                activity_total > 0,
                pd.to_numeric(merged["bg_weight"], errors="coerce").fillna(0.0)
                / activity_total.where(activity_total > 0, 1.0),
                0.0,
            ),
        )
        merged["component_count"] = pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0) * merged["component_share"]
        merged["jurisdiction_id"] = merged["group_id"]
        merged["jurisdiction_type"] = COUNTY_NONMUNICIPAL_OVERLAP_TYPE
        out_frames.append(merged[["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"]].copy())

    absorbed_targets = targets[targets["group_kind"].eq("absorbed_overlap")].copy()
    if not absorbed_targets.empty:
        place_crosswalk = bg_crosswalk[bg_crosswalk["jurisdiction_type"].eq("municipal")].rename(columns={"block_group_geoid": "bg_id"}).copy()
        merged = bg_prior.merge(place_crosswalk, on=["bg_id", "state_fips"], how="inner")
        merged = merged.merge(
            absorbed_targets.rename(columns={"group_id": "jurisdiction_id"})[["state_fips", "jurisdiction_id", "offense", "target_count"]],
            on=["state_fips", "jurisdiction_id", "offense"],
            how="inner",
        )
        merged["component_activity_weight"] = (
            pd.to_numeric(merged["bg_weight"], errors="coerce").fillna(0.0)
            * pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0)
        )
        sums = (
            merged.groupby(["state_fips", "jurisdiction_id", "offense"], dropna=False)
            .agg(activity_total=("component_activity_weight", "sum"), allocation_total=("allocation_share", "sum"))
            .reset_index()
        )
        merged = merged.merge(sums, on=["state_fips", "jurisdiction_id", "offense"], how="left")
        denom_activity = pd.to_numeric(merged["activity_total"], errors="coerce").fillna(0.0)
        denom_alloc = pd.to_numeric(merged["allocation_total"], errors="coerce").fillna(0.0)
        numer_activity = pd.to_numeric(merged["component_activity_weight"], errors="coerce").fillna(0.0)
        raw_alloc = pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0)
        merged["component_share"] = np.where(
            denom_activity > 0,
            numer_activity / denom_activity,
            np.where(denom_alloc > 0, raw_alloc / denom_alloc, 0.0),
        )
        merged["component_count"] = pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0) * merged["component_share"]
        merged["jurisdiction_type"] = "absorbed_overlap_layer"
        out_frames.append(merged[["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"]].copy())

    custom_targets = targets[targets["group_kind"].eq("custom_footprint_overlap")].copy()
    if not custom_targets.empty:
        custom_footprints = _load_overlap_custom_footprints(paths)
        if not custom_footprints.empty:
            # Resident footprints combine the predicted-count prior with the ORI's within-BG
            # responsibility fraction. Service-wide footprints use their own documented land
            # coverage. A share that already measures activity (boardings, throughput, LandScan
            # daytime) or deliberately apportions area onto unpopulated parcels remains verbatim.
            # One declared basis per allocation pool is enforced in the loader.
            footprint_scope = custom_footprints.get(
                "allocation_scope",
                pd.Series("source_state", index=custom_footprints.index, dtype="string"),
            ).astype("string").fillna("source_state")
            ordinary_footprints = custom_footprints[~footprint_scope.eq("service_wide")].copy()
            service_footprints = custom_footprints[footprint_scope.eq("service_wide")].copy()
            custom_frames: list[pd.DataFrame] = []

            if not ordinary_footprints.empty:
                ordinary = (
                    bg_prior[["state_fips", "bg_id", "tract_id", "offense", "bg_weight"]]
                    .merge(ordinary_footprints, on=["state_fips", "bg_id"], how="inner")
                    .merge(
                        custom_targets.rename(columns={"group_id": "ori9"})[
                            ["state_fips", "ori9", "offense", "target_count"]
                        ],
                        on=["state_fips", "ori9", "offense"],
                        how="inner",
                    )
                )
                ordinary["source_state_fips"] = ordinary["state_fips"]
                ordinary["canonical_target_ori"] = ordinary["ori9"]
                ordinary["service_scope_id"] = pd.NA
                ordinary["allocation_scope"] = "source_state"
                custom_frames.append(ordinary)

            if not service_footprints.empty:
                service_footprints["canonical_target_ori"] = service_footprints[
                    "canonical_target_ori"
                ].astype("string")
                service_target_ids = set(service_footprints["canonical_target_ori"].dropna())
                service_targets = custom_targets[
                    custom_targets["group_id"].astype("string").isin(service_target_ids)
                ].copy()
                duplicate_targets = service_targets.duplicated(
                    ["group_id", "offense"], keep=False
                )
                if bool(duplicate_targets.any()):
                    raise ValueError(
                        "service-wide custom footprint has more than one positive canonical "
                        "source-state target: "
                        + str(
                            service_targets.loc[
                                duplicate_targets,
                                ["state_fips", "group_id", "offense", "target_count"],
                            ].to_dict(orient="records")
                        )
                    )
                service_targets = service_targets.rename(
                    columns={
                        "state_fips": "_target_source_state_fips",
                        "group_id": "canonical_target_ori",
                    }
                )
                service = (
                    bg_prior[["state_fips", "bg_id", "tract_id", "offense", "bg_weight"]]
                    .merge(service_footprints, on=["state_fips", "bg_id"], how="inner")
                    .merge(
                        service_targets[
                            [
                                "_target_source_state_fips",
                                "canonical_target_ori",
                                "offense",
                                "target_count",
                            ]
                        ],
                        on=["canonical_target_ori", "offense"],
                        how="inner",
                        validate="many_to_one",
                    )
                )
                declared_source = service["source_state_fips"].astype("string").str.zfill(2)
                target_source = service["_target_source_state_fips"].astype("string").str.zfill(2)
                bad_source = declared_source.ne(target_source)
                if bool(bad_source.any()):
                    raise ValueError(
                        "service-wide footprint source state disagrees with canonical target: "
                        + str(
                            service.loc[
                                bad_source,
                                [
                                    "service_scope_id",
                                    "canonical_target_ori",
                                    "source_state_fips",
                                    "_target_source_state_fips",
                                ],
                            ]
                            .drop_duplicates()
                            .to_dict(orient="records")
                        )
                    )
                service["source_state_fips"] = target_source
                service = service.drop(columns="_target_source_state_fips")
                service["ori9"] = service["canonical_target_ori"]
                custom_frames.append(service)

            if custom_frames:
                merged = _custom_footprint_component_shares(
                    pd.concat(custom_frames, ignore_index=True)
                )
                merged["component_count"] = (
                    pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0)
                    * merged["component_share"]
                )
                merged["source_target_count"] = pd.to_numeric(
                    merged["target_count"], errors="coerce"
                ).fillna(0.0)
                merged["jurisdiction_id"] = merged["canonical_target_ori"]
                merged["jurisdiction_type"] = "custom_footprint_overlap_layer"
                out_frames.append(
                    merged[
                        [
                            "state_fips",
                            "source_state_fips",
                            "bg_id",
                            "tract_id",
                            "jurisdiction_id",
                            "jurisdiction_type",
                            "offense",
                            "component_count",
                            "service_scope_id",
                            "canonical_target_ori",
                            "source_target_count",
                            "bg_responsibility_population_coverage_share",
                            "responsibility_fraction_basis",
                        ]
                    ].copy()
                )

    statewide_targets = targets[targets["group_kind"].eq("statewide_overlap")].copy()
    if not statewide_targets.empty:
        bg_weights = bg_prior[["state_fips", "bg_id", "tract_id", "offense", "bg_weight"]].copy()
        totals = bg_weights.groupby(["state_fips", "offense"], dropna=False)["bg_weight"].sum().rename("state_bg_weight_total").reset_index()
        bg_weights = bg_weights.merge(totals, on=["state_fips", "offense"], how="left")
        bg_weights["state_bg_share"] = np.where(
            pd.to_numeric(bg_weights["state_bg_weight_total"], errors="coerce").fillna(0.0) > 0,
            pd.to_numeric(bg_weights["bg_weight"], errors="coerce").fillna(0.0) / pd.to_numeric(bg_weights["state_bg_weight_total"], errors="coerce").fillna(1.0),
            0.0,
        )
        merged = bg_weights.merge(statewide_targets[["state_fips", "group_id", "offense", "target_count"]], on=["state_fips", "offense"], how="inner")
        merged["component_count"] = pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0) * pd.to_numeric(merged["state_bg_share"], errors="coerce").fillna(0.0)
        merged["jurisdiction_id"] = merged["group_id"]
        merged["jurisdiction_type"] = "statewide_overlap_layer"
        out_frames.append(merged[["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"]].copy())

    # `UNLOCATED_GROUP_KIND` deliberately has NO branch here. That absence is the lane: a group
    # the cascade could not resolve produces no block-group components at all, so its mass never
    # reaches the surface, and the companion table below is where it is published instead. The
    # group is still in `targets`, so it still partitions the control -- the mass is accounted
    # for, just not placed.
    if not out_frames:
        return empty_components
    components = pd.concat(out_frames, ignore_index=True)
    if "source_state_fips" not in components.columns:
        components["source_state_fips"] = components["state_fips"]
    else:
        components["source_state_fips"] = components["source_state_fips"].fillna(
            components["state_fips"]
        )
    components["source_state_fips"] = (
        components["source_state_fips"].astype("string").str.zfill(2)
    )
    if unlocated is not None:
        components.attrs["unlocated_mass"] = unlocated
    if footprint_partition is not None:
        components.attrs["footprint_mass_partition"] = footprint_partition
    return components


def _raw_denominator(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


def _eb_alpha_dict(config: AllocationBuildConfig) -> dict[str, float]:
    out = {offense: 1.0 for offense in OFFENSES_7}
    for offense, alpha in config.eb_alpha_by_offense:
        if str(offense) in out:
            out[str(offense)] = float(alpha)
    return out


def _count_derived_rate_index(
    *,
    counts: pd.Series,
    denominator: pd.Series,
    publishable: pd.Series,
    normalization_publishable: pd.Series | None = None,
) -> dict[str, float | pd.Series]:
    denom = _raw_denominator(denominator)
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    pub = pd.Series(publishable, index=count.index).fillna(False).astype(bool) & denom.gt(0.0)
    norm_pub = (
        pd.Series(normalization_publishable, index=count.index).fillna(False).astype(bool) & denom.gt(0.0)
        if normalization_publishable is not None
        else pub
    )
    denom_sum = float(denom.loc[norm_pub].sum())
    count_sum = float(count.loc[norm_pub].sum())
    national_rate = RATE_PER_100K * count_sum / denom_sum if denom_sum > 0 else float("nan")

    rate = pd.Series(np.nan, index=count.index, dtype=float)
    rate.loc[pub] = RATE_PER_100K * count.loc[pub] / denom.loc[pub]
    index = pd.Series(np.nan, index=count.index, dtype=float)
    if np.isfinite(national_rate) and national_rate > 0:
        index.loc[pub] = 100.0 * rate.loc[pub] / national_rate
    return {
        "rate": rate.replace([np.inf, -np.inf], np.nan),
        "index": index.replace([np.inf, -np.inf], np.nan),
        "national_rate_per_100k": national_rate,
    }


def _crime_density(counts: pd.Series, land_area_sq_mi: pd.Series) -> pd.Series:
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    area = pd.to_numeric(land_area_sq_mi, errors="coerce").fillna(0.0).clip(lower=0.0)
    # Density is undefined where there is no land area (water-only cells): publish NULL,
    # never inf. A finite count over zero land cannot be a per-area rate.
    density = np.full(len(count), np.nan, dtype=float)
    positive_area = area.gt(0.0).to_numpy(dtype=bool)
    density[positive_area] = count.to_numpy(dtype=float)[positive_area] / area.to_numpy(dtype=float)[positive_area]
    return pd.Series(density, index=count.index, dtype=float)


def _poisson_count_interval(
    counts: pd.Series,
    *,
    alpha: float = POISSON_INTERVAL_ALPHA,
) -> tuple[pd.Series, pd.Series]:
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    count_values = count.to_numpy(dtype=float)
    lower = np.zeros(len(count_values), dtype=float)
    positive = count_values > 0.0
    lower[positive] = 0.5 * chi2.ppf(float(alpha) / 2.0, 2.0 * count_values[positive])
    upper = 0.5 * chi2.ppf(1.0 - float(alpha) / 2.0, 2.0 * (count_values + 1.0))
    return (
        pd.Series(lower, index=count.index, dtype=float).replace([np.inf, -np.inf], np.nan),
        pd.Series(upper, index=count.index, dtype=float).replace([np.inf, -np.inf], np.nan),
    )


def _rate_index_interval(
    *,
    counts: pd.Series,
    denominator: pd.Series,
    publishable: pd.Series,
    national_rate_per_100k: float,
) -> dict[str, pd.Series]:
    denom = _raw_denominator(denominator)
    pub = pd.Series(publishable, index=denom.index).fillna(False).astype(bool) & denom.gt(0.0)
    count_lower, count_upper = _poisson_count_interval(counts)
    rate_lower = pd.Series(np.nan, index=denom.index, dtype=float)
    rate_upper = pd.Series(np.nan, index=denom.index, dtype=float)
    rate_lower.loc[pub] = RATE_PER_100K * count_lower.loc[pub] / denom.loc[pub]
    rate_upper.loc[pub] = RATE_PER_100K * count_upper.loc[pub] / denom.loc[pub]
    index_lower = pd.Series(np.nan, index=denom.index, dtype=float)
    index_upper = pd.Series(np.nan, index=denom.index, dtype=float)
    national_rate = float(national_rate_per_100k)
    if np.isfinite(national_rate) and national_rate > 0.0:
        index_lower.loc[pub] = 100.0 * rate_lower.loc[pub] / national_rate
        index_upper.loc[pub] = 100.0 * rate_upper.loc[pub] / national_rate
    return {
        "rate_lower": rate_lower.replace([np.inf, -np.inf], np.nan),
        "rate_upper": rate_upper.replace([np.inf, -np.inf], np.nan),
        "index_lower": index_lower.replace([np.inf, -np.inf], np.nan),
        "index_upper": index_upper.replace([np.inf, -np.inf], np.nan),
    }


def _support_series(
    out: pd.DataFrame,
    column: str,
    *,
    default: float = 0.0,
) -> pd.Series:
    if column not in out.columns:
        return pd.Series(default, index=out.index, dtype=float)
    return pd.to_numeric(out[column], errors="coerce").fillna(default).clip(lower=0.0)


def _support_flag_series(out: pd.DataFrame, column: str) -> pd.Series:
    if column not in out.columns:
        return pd.Series(False, index=out.index)
    return pd.Series(out[column], index=out.index).fillna(False).astype(bool)


def _reliability_tier(
    *,
    publishable: pd.Series,
    effective_support: pd.Series,
    direct_support_years: pd.Series,
    index_width_ratio: pd.Series,
) -> pd.Series:
    pub = pd.Series(publishable, index=effective_support.index).fillna(False).astype(bool)
    support = pd.to_numeric(effective_support, errors="coerce").fillna(0.0)
    years = pd.to_numeric(direct_support_years, errors="coerce").fillna(0.0)
    ratio = pd.to_numeric(index_width_ratio, errors="coerce")
    tier = pd.Series("low", index=effective_support.index, dtype="string")
    high = (
        pub
        & support.ge(float(RELIABILITY_HIGH_SUPPORT_MIN))
        & years.ge(float(RELIABILITY_HIGH_MIN_SOURCE_YEARS))
        & ratio.le(float(RELIABILITY_HIGH_INDEX_CI95_WIDTH_RATIO_MAX))
    )
    medium = (
        pub
        & ~high
        & support.ge(float(RELIABILITY_MEDIUM_SUPPORT_MIN))
        & years.ge(float(RELIABILITY_MEDIUM_MIN_SOURCE_YEARS))
        & ratio.le(float(RELIABILITY_MEDIUM_INDEX_CI95_WIDTH_RATIO_MAX))
    )
    tier.loc[medium] = "medium"
    tier.loc[high] = "high"
    return tier


def _recommended_display_geography(
    *,
    tier: pd.Series,
    publishable: pd.Series,
    geo_id_col: str,
) -> pd.Series:
    pub = pd.Series(publishable, index=tier.index).fillna(False).astype(bool)
    current_geo = "block_group" if geo_id_col == "block_group_geoid" else "tract"
    low_geo = "tract_or_larger" if geo_id_col == "block_group_geoid" else "jurisdiction_or_county"
    recommended = pd.Series(low_geo, index=tier.index, dtype="string")
    recommended.loc[pd.Series(tier, index=tier.index).astype("string").isin(["high", "medium"])] = current_geo
    recommended.loc[~pub] = "not_published"
    return recommended


def _estimate_mode(
    *,
    non_residential: pd.Series,
    publishable: pd.Series,
    denominator: pd.Series,
    denominator_invalid: pd.Series | None = None,
    denominator_invalid_mode: str = "denominator_invalid",
) -> pd.Series:
    mode = pd.Series("count_derived", index=publishable.index, dtype="string")
    mode.loc[~publishable] = "zero_primary_denominator"
    mode.loc[pd.to_numeric(denominator, errors="coerce").fillna(0.0).gt(0.0) & ~publishable] = (
        "below_legacy_min_denominator"
    )
    if denominator_invalid is not None:
        invalid = pd.Series(denominator_invalid, index=publishable.index).fillna(False).astype(bool)
        mode.loc[invalid] = str(denominator_invalid_mode)
    mode.loc[non_residential] = "non_residential"
    return mode


def _rates_match_count_formula(raw_rate: pd.Series, count_rate: pd.Series) -> bool:
    left = pd.to_numeric(raw_rate, errors="coerce").to_numpy(dtype=np.float64)
    right = pd.to_numeric(count_rate, errors="coerce").to_numpy(dtype=np.float64)
    return bool(np.array_equal(left, right, equal_nan=True))


def _warn_if_raw_rate_mismatch(
    *,
    label: str,
    raw_rate: pd.Series,
    count_rate: pd.Series,
) -> None:
    if _rates_match_count_formula(raw_rate, count_rate):
        return
    raw = pd.to_numeric(raw_rate, errors="coerce")
    direct = pd.to_numeric(count_rate, errors="coerce")
    diff = (raw - direct).replace([np.inf, -np.inf], np.nan).abs()
    warnings.warn(
        f"{label}: diagnostic EB raw_rate differs from count/denominator formula; "
        f"published rate/index use the direct count-derived formula. max_abs_diff={float(diff.max()):.12g}",
        RuntimeWarning,
        stacklevel=2,
    )


def _empirical_bayes_index(
    out: pd.DataFrame,
    *,
    offense: str,
    counts: pd.Series,
    denominator: pd.Series,
    geo_id_col: str,
    alpha: float,
    hard_min: float,
    jurisdiction_col: str | None = None,
) -> dict[str, object]:
    denom = _raw_denominator(denominator)
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    publishable = denom.gt(float(hard_min))
    valid_denom_sum = float(denom.loc[publishable].sum())
    valid_count_sum = float(count.loc[publishable].sum())
    national_rate = valid_count_sum / valid_denom_sum if valid_denom_sum > 0 else 0.0
    k = float(alpha) / national_rate if national_rate > 0 else float("inf")

    raw_rate = pd.Series(np.nan, index=out.index, dtype=float)
    raw_rate.loc[publishable] = RATE_PER_100K * count.loc[publishable] / denom.loc[publishable]
    diagnostic_eb_prior_rate = pd.Series(np.nan, index=out.index, dtype=float)
    diagnostic_eb_rate = pd.Series(np.nan, index=out.index, dtype=float)
    diagnostic_eb_index = pd.Series(np.nan, index=out.index, dtype=float)
    diagnostic_eb_observed_weight = pd.Series(np.nan, index=out.index, dtype=float)
    diagnostic_eb_prior_weight = pd.Series(np.nan, index=out.index, dtype=float)

    state = out["state_fips"].astype("string").str.zfill(2)
    jurisdiction: pd.Series | None = None
    if jurisdiction_col is not None:
        if jurisdiction_col not in out.columns:
            raise KeyError(f"missing required EB jurisdiction column {jurisdiction_col!r}")
        jurisdiction = out[jurisdiction_col].astype("string")
        missing_jurisdiction = jurisdiction.isna() | jurisdiction.str.strip().eq("")
        if bool(missing_jurisdiction.any()):
            sample_geoids = (
                out.loc[missing_jurisdiction, geo_id_col].astype("string").head(10).fillna("<NA>").tolist()
                if geo_id_col in out.columns
                else []
            )
            raise ValueError(
                f"{jurisdiction_col} is required for EB diagnostics; "
                f"found {int(missing_jurisdiction.sum())} missing row(s). sample_{geo_id_col}={sample_geoids}"
            )

    if not np.isfinite(k) or k <= 0 or not publishable.any():
        denominator_reason = pd.Series("zero_or_structural_denominator", index=out.index, dtype="string")
        denominator_reason.loc[publishable] = "invalid_national_rate"
        return {
            "denominator_raw": denom,
            "raw_rate": raw_rate,
            "diagnostic_eb_rate": diagnostic_eb_rate,
            "diagnostic_eb_index": diagnostic_eb_index,
            "national_rate_per_100k": national_rate * RATE_PER_100K,
            "diagnostic_eb_national_rate_per_100k": np.nan,
            "diagnostic_eb_prior_rate": diagnostic_eb_prior_rate,
            "diagnostic_eb_k": k,
            "diagnostic_eb_observed_weight": diagnostic_eb_observed_weight,
            "diagnostic_eb_prior_weight": diagnostic_eb_prior_weight,
            "index_publishable": publishable & False,
            "diagnostic_eb_low_denominator_flag": publishable & False,
            "diagnostic_eb_heavy_shrinkage_flag": publishable & False,
            "diagnostic_eb_extreme_shrinkage_flag": publishable & False,
            "denominator_reason": denominator_reason,
        }

    support_count = count.where(publishable, 0.0)
    support_denom = denom.where(publishable, 0.0)
    eb_frame = pd.DataFrame(
        {
            "_state": state,
            "_tract": out["tract_id"].astype("string") if "tract_id" in out.columns else out[geo_id_col].astype("string"),
            "_count": support_count,
            "_denom": support_denom,
        },
        index=out.index,
    )
    if jurisdiction is not None:
        eb_frame["_jurisdiction"] = jurisdiction
    else:
        eb_frame["_jurisdiction"] = state

    state_totals = eb_frame.groupby("_state", dropna=False)[["_count", "_denom"]].sum().reset_index()
    state_rates = eb_frame[["_state"]].merge(state_totals, on="_state", how="left")
    state_count = pd.Series(pd.to_numeric(state_rates["_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    state_denom = pd.Series(pd.to_numeric(state_rates["_denom"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    state_count_excl = (state_count - support_count).clip(lower=0.0)
    state_denom_excl = (state_denom - support_denom).clip(lower=0.0)
    state_rate_raw = pd.Series(
        np.where(
            state_denom_excl.gt(float(hard_min)),
            state_count_excl / state_denom_excl.replace(0.0, np.nan),
            national_rate,
        ),
        index=out.index,
        dtype=float,
    ).replace([np.inf, -np.inf], np.nan).fillna(national_rate)

    jur_totals = eb_frame.groupby(["_state", "_jurisdiction"], dropna=False)[["_count", "_denom"]].sum().reset_index()
    jur_rates = eb_frame[["_state", "_jurisdiction"]].merge(
        jur_totals,
        on=["_state", "_jurisdiction"],
        how="left",
    )
    jur_count = pd.Series(pd.to_numeric(jur_rates["_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    jur_denom = pd.Series(pd.to_numeric(jur_rates["_denom"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    jur_count_excl = (jur_count - support_count).clip(lower=0.0)
    jur_denom_excl = (jur_denom - support_denom).clip(lower=0.0)
    r_jur = pd.Series(((jur_count_excl + k * state_rate_raw) / (jur_denom_excl + k)).to_numpy(dtype=float), index=out.index)

    tract_totals = eb_frame.groupby(["_state", "_tract"], dropna=False)[["_count", "_denom"]].sum().reset_index()
    tract_rates = eb_frame[["_state", "_tract"]].merge(tract_totals, on=["_state", "_tract"], how="left")
    tract_count = pd.Series(pd.to_numeric(tract_rates["_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    tract_denom = pd.Series(pd.to_numeric(tract_rates["_denom"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    tract_count_excl = (tract_count - support_count).clip(lower=0.0)
    tract_denom_excl = (tract_denom - support_denom).clip(lower=0.0)
    tract_prior_raw = (tract_count_excl + k * r_jur) / (tract_denom_excl + k)
    use_tract = tract_denom_excl.gt(float(hard_min)) & np.isfinite(tract_prior_raw)
    prior_rate_raw = pd.Series(np.where(use_tract, tract_prior_raw, r_jur), index=out.index, dtype=float)

    diagnostic_eb_raw_rate = (count + k * prior_rate_raw) / (denom + k)
    diagnostic_eb_raw_rate = diagnostic_eb_raw_rate.where(publishable, np.nan)
    diagnostic_eb_national_rate = float((denom.loc[publishable] * diagnostic_eb_raw_rate.loc[publishable]).sum() / denom.loc[publishable].sum())
    diagnostic_eb_index.loc[publishable] = 100.0 * diagnostic_eb_raw_rate.loc[publishable] / diagnostic_eb_national_rate
    diagnostic_eb_rate.loc[publishable] = RATE_PER_100K * diagnostic_eb_raw_rate.loc[publishable]
    diagnostic_eb_prior_rate.loc[publishable] = RATE_PER_100K * prior_rate_raw.loc[publishable]
    diagnostic_eb_observed_weight.loc[publishable] = denom.loc[publishable] / (denom.loc[publishable] + k)
    diagnostic_eb_prior_weight.loc[publishable] = k / (denom.loc[publishable] + k)
    diagnostic_eb_low_denominator = publishable & diagnostic_eb_observed_weight.lt(0.5)
    diagnostic_eb_heavy_shrinkage = publishable & diagnostic_eb_observed_weight.lt(0.20)
    diagnostic_eb_extreme_shrinkage = publishable & diagnostic_eb_observed_weight.lt(0.05)
    denominator_reason = pd.Series("zero_or_structural_denominator", index=out.index, dtype="string")
    denominator_reason.loc[publishable] = "publishable"

    return {
        "denominator_raw": denom,
        "raw_rate": raw_rate.replace([np.inf, -np.inf], np.nan),
        "diagnostic_eb_rate": diagnostic_eb_rate.replace([np.inf, -np.inf], np.nan),
        "diagnostic_eb_index": diagnostic_eb_index.replace([np.inf, -np.inf], np.nan),
        "national_rate_per_100k": national_rate * RATE_PER_100K,
        "diagnostic_eb_national_rate_per_100k": diagnostic_eb_national_rate * RATE_PER_100K,
        "diagnostic_eb_prior_rate": diagnostic_eb_prior_rate.replace([np.inf, -np.inf], np.nan),
        "diagnostic_eb_k": k,
        "diagnostic_eb_observed_weight": diagnostic_eb_observed_weight,
        "diagnostic_eb_prior_weight": diagnostic_eb_prior_weight,
        "index_publishable": publishable,
        "diagnostic_eb_low_denominator_flag": diagnostic_eb_low_denominator,
        "diagnostic_eb_heavy_shrinkage_flag": diagnostic_eb_heavy_shrinkage,
        "diagnostic_eb_extreme_shrinkage_flag": diagnostic_eb_extreme_shrinkage,
        "denominator_reason": denominator_reason,
    }


def _expected_count_col(name: str) -> str:
    return f"expected_count_{name}"


def _source_count_col(out: pd.DataFrame, offense: str) -> str:
    expected_col = _expected_count_col(offense)
    legacy_col = f"count_{offense}"
    if expected_col in out.columns:
        return expected_col
    return legacy_col


def _national_expected_count_weights(out: pd.DataFrame, offenses: list[str]) -> dict[str, float]:
    totals = {
        offense: float(pd.to_numeric(out[_expected_count_col(offense)], errors="coerce").fillna(0.0).clip(lower=0.0).sum())
        for offense in offenses
    }
    total = float(sum(totals.values()))
    if total <= 0.0 or not np.isfinite(total):
        return {offense: float("nan") for offense in offenses}
    return {offense: totals[offense] / total for offense in offenses}


def _full_component_index_composite(
    out: pd.DataFrame,
    offenses: list[str],
    *,
    index_suffix: str,
    weights: dict[str, float],
    index_overrides: dict[str, pd.Series] | None = None,
) -> pd.Series:
    # index_overrides lets a rare offense contribute its tract-support index in place of the
    # (unpublished) block-group one, so the equal-offense composite stays defined at block group.
    overrides = index_overrides or {}
    values = np.vstack(
        [
            pd.to_numeric(
                overrides[offense] if offense in overrides else out[f"index_{offense}_{index_suffix}"],
                errors="coerce",
            ).to_numpy(dtype=float)
            for offense in offenses
        ]
    )
    weight_values = np.array([float(weights[offense]) for offense in offenses], dtype=float)
    valid_rows = np.isfinite(values).all(axis=0)
    valid_weights = np.isfinite(weight_values) & (weight_values > 0.0)
    weight_sum = float(weight_values[valid_weights].sum())
    composite = np.full(values.shape[1], np.nan, dtype=float)
    if weight_sum > 0.0:
        composite[valid_rows] = np.dot(weight_values, values[:, valid_rows]) / weight_sum
    return pd.Series(composite, index=out.index, dtype=float).replace([np.inf, -np.inf], np.nan)


def _resident_part1_index(
    out: pd.DataFrame,
    offenses: list[str],
    *,
    ignored_component_publishability: frozenset[str] = frozenset(),
) -> tuple[pd.Series, float]:
    denom = _raw_denominator(out["resident_secondary_denominator"])
    counts = sum(
        pd.to_numeric(out[_expected_count_col(offense)], errors="coerce").fillna(0.0).clip(lower=0.0)
        for offense in offenses
    )
    publishable = denom.gt(0.0)
    for offense in offenses:
        if offense in ignored_component_publishability:
            continue
        publishable_col = f"index_{offense}_resident_publishable"
        if publishable_col in out.columns:
            component_publishable = out[publishable_col].fillna(False).astype(bool)
        else:
            component_publishable = pd.to_numeric(
                out[f"index_{offense}_resident"], errors="coerce"
            ).notna()
        # `estimate_mode_*` is the PRIMARY (opportunity-denominator) arm's display code. It used
        # to be OR-ed in here, which let a component suppressed on the EXPOSURE family un-veto an
        # aggregate on the RESIDENT family -- two different denominators deciding one cell. Each
        # composite is now gated by its own denominator family only, so the resident aggregates
        # read `index_{offense}_resident_publishable` and nothing else.
        publishable &= component_publishable
    denom_sum = float(denom.loc[publishable].sum())
    count_sum = float(counts.loc[publishable].sum())
    national_rate = RATE_PER_100K * count_sum / denom_sum if denom_sum > 0.0 else float("nan")
    rate = pd.Series(np.nan, index=out.index, dtype=float)
    rate.loc[publishable] = RATE_PER_100K * counts.loc[publishable] / denom.loc[publishable]
    index = pd.Series(np.nan, index=out.index, dtype=float)
    if np.isfinite(national_rate) and national_rate > 0.0:
        index.loc[publishable] = 100.0 * rate.loc[publishable] / national_rate
    return index.replace([np.inf, -np.inf], np.nan), national_rate


def _harm_weighted_total_index(
    out: pd.DataFrame,
    offenses: list[str],
    *,
    count_overrides: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """Count-derived total harm index (Crime Harm Index shape): harm_count is the sentencing-days
    weighted SUM of expected counts across the seven primary offenses (not an average of already-
    normalized per-offense indices), normalized by the same person-exposure denominator and
    publication floor/eligibility rule used by the person-offense primary indices (murder, rape,
    robbery, aggravated assault, larceny). One normalization at the end, mirroring how the other
    count-derived indices in this function are computed.

    count_overrides lets a rare offense enter the harm sum at its tract-support count (spread
    within the tract by person exposure) instead of its noisy block-group count, so the aggregate
    is more robust than its parts while the sentencing-day weights stay untouched.
    """
    overrides = count_overrides or {}
    denom = _raw_denominator(out["exposure_proxy_2024"])
    insufficient_exposure = denom.lt(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
    residential_eligible = pd.to_numeric(out["households_total"], errors="coerce").fillna(0.0).ge(
        float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
    )
    special_use_tract = pd.Series(out["special_use_tract_flag"], index=out.index).fillna(False).astype(bool)
    publishable = residential_eligible & denom.gt(0.0) & ~special_use_tract & ~insufficient_exposure
    counts = sum(
        float(HARM_WEIGHTS[offense])
        * pd.to_numeric(
            overrides[offense] if offense in overrides else out[_expected_count_col(offense)],
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        for offense in offenses
    )
    published = _count_derived_rate_index(counts=counts, denominator=denom, publishable=publishable)
    return pd.Series(published["index"], index=out.index, dtype=float)


def _aggregate_index_normalizers(
    surface: pd.DataFrame,
    *,
    composites: CompositeRuntime | None = None,
    support: str | None = None,
) -> dict[str, object]:
    if composites is not None:
        # Count-first lane: the reproducible record is the count total, the common-denominator
        # total and the reference rate per field -- plus the severity vector for the harm field.
        # The two surviving index averages keep their offense-weight record under their new names.
        event_weights = _national_expected_count_weights(surface, list(OFFENSES_7))
        personal_weights = _national_expected_count_weights(surface, list(PERSONAL_OFFENSES))
        property_weights = _national_expected_count_weights(surface, list(PROPERTY_OFFENSES))
        return {
            "count_first_burden": composite_normalizers(
                surface, runtime=composites, support=str(support or "tract")
            ),
            "multi_offense_relative_score_event_weighted": {
                "field": "multi_offense_relative_score_event_weighted",
                "offense_weights": event_weights,
                "weight_source": "national expected_count offense shares in this surface",
                "unit": "dimensionless relative score over mixed-denominator per-offense indexes",
            },
            "multi_offense_relative_score_equal_offense": {
                "field": "multi_offense_relative_score_equal_offense",
                "offense_weights": {offense: 1.0 / float(len(OFFENSES_7)) for offense in OFFENSES_7},
                "weight_source": "equal weight per Part-I offense",
                "unit": "dimensionless relative score over mixed-denominator per-offense indexes",
            },
            PERSONAL_RELATIVE_SCORE_COLUMN: {
                "field": PERSONAL_RELATIVE_SCORE_COLUMN,
                "offense_weights": personal_weights,
                "weight_source": "national expected_count offense shares within personal offenses in this surface",
                "unit": "dimensionless relative score over mixed-denominator per-offense indexes",
            },
            PROPERTY_RELATIVE_SCORE_COLUMN: {
                "field": PROPERTY_RELATIVE_SCORE_COLUMN,
                "offense_weights": property_weights,
                "weight_source": "national expected_count offense shares within property offenses in this surface",
                "unit": "dimensionless relative score over mixed-denominator per-offense indexes",
            },
        }
    resident_specs = {
        "index_total_part1_resident": OFFENSES_7,
        "index_personal_part1_resident": PERSONAL_OFFENSES,
        "index_property_part1_resident": PROPERTY_OFFENSES,
    }
    resident: dict[str, dict[str, float | int | list[str]]] = {}
    denom = _raw_denominator(surface["resident_secondary_denominator"])
    for field, offenses in resident_specs.items():
        counts = sum(
            pd.to_numeric(surface[_expected_count_col(offense)], errors="coerce").fillna(0.0).clip(lower=0.0)
            for offense in offenses
        )
        publishable = pd.to_numeric(surface[field], errors="coerce").notna() & denom.gt(0.0)
        denominator_total = float(denom.loc[publishable].sum())
        expected_count_total = float(counts.loc[publishable].sum())
        resident[field] = {
            "offenses": list(offenses),
            "published_rows": int(publishable.sum()),
            "expected_count_total": expected_count_total,
            "resident_denominator_total": denominator_total,
            "national_rate_per_100k": (
                RATE_PER_100K * expected_count_total / denominator_total
                if denominator_total > 0.0
                else float("nan")
            ),
        }

    event_weights = _national_expected_count_weights(surface, list(OFFENSES_7))
    return {
        "resident_part1": resident,
        "primary_event_weighted": {
            "field": "index_total_primary_event_weighted",
            "offense_weights": event_weights,
            "weight_source": "national expected_count offense shares in this surface",
        },
        "primary_equal_offense": {
            "field": "index_total_equal_offense",
            "offense_weights": {offense: 1.0 / float(len(OFFENSES_7)) for offense in OFFENSES_7},
            "weight_source": "equal weight per Part-I offense",
        },
        "primary_harm_weighted": {
            "field": "index_total_harm",
            "offense_weights": dict(HARM_WEIGHTS),
            "weight_source": (
                "Sentencing-days severity weights in the Cambridge Crime Harm Index tradition "
                "(Sherman, Neyroud & Neyroud 2016, 'The Cambridge Crime Harm Index'); values are "
                "round starting-point approximations, documented as such, not the England/Wales "
                "schedule verbatim. Source: src/crimerisk/allocation.py:HARM_WEIGHTS."
            ),
            "definition": (
                "Count-derived total harm (Crime Harm Index shape), not an average of the seven "
                "already-normalized per-offense indices: harm_count = sum(weight_o * expected_count_o) "
                "over the seven primary offenses; index_total_harm = 100 * (harm_count / person_exposure) "
                "/ national_harm_rate, one normalization at the end. Uses the same person-exposure "
                "denominator (exposure_proxy_2024) and publication floor/eligibility as the "
                "murder/rape/robbery/aggravated-assault/larceny primary indices, so it is publishable "
                "wherever person exposure is publishable -- it does not require all seven per-offense "
                "indices to be finite."
            ),
        },
    }


def _rare_offense_published_point_fields(offense: str) -> tuple[str, ...]:
    """The per-offense index and rate point fields (and their confidence intervals) that a
    consumer would read as this area's own value for the offense. These are nulled at block-group
    support for the rare person offenses; every other per-offense field (expected count,
    diagnostics, reliability, denominator, support) is retained."""
    return (
        f"raw_rate_{offense}",
        f"rate_{offense}_primary",
        f"rate_{offense}_primary_ci95_lower",
        f"rate_{offense}_primary_ci95_upper",
        f"index_{offense}_primary",
        f"index_{offense}_primary_ci95_lower",
        f"index_{offense}_primary_ci95_upper",
        f"index_{offense}_primary_ci95_width",
        f"index_{offense}_primary_ci95_width_ratio",
        f"resident_raw_rate_{offense}",
        f"rate_{offense}_resident",
        f"index_{offense}_resident",
        # v2 uncertainty lane: the rate and index QUANTILES are per-offense point payload in
        # exactly this sense -- a reader would take `index_murder_primary_p90` as this block
        # group's own murder value -- so they follow the point they describe. Named
        # unconditionally; the caller nulls only the columns actually present, so a legacy build
        # is untouched. The COUNT quantiles are deliberately absent from this list: the count is
        # conserved and already published at block group, and its interval says nothing the point
        # does not already say at that support.
        *uncertainty_rare_offense_suppressed_columns(offense),
    )


def _within_tract_person_exposure_share(bg: pd.DataFrame, group: pd.Series) -> pd.Series:
    """Each block group's share of its parent tract's person exposure, with resident-population
    then equal-weight fallbacks so the shares always sum to 1 within a tract. That guarantee is
    what makes redistributing a tract count to its block groups conserve the tract total exactly.
    `group` is the (normalized) parent-tract id per block group."""
    candidates = [
        pd.to_numeric(bg["exposure_proxy_2024"], errors="coerce").fillna(0.0).clip(lower=0.0),
        pd.to_numeric(bg["resident_secondary_denominator"], errors="coerce").fillna(0.0).clip(lower=0.0),
        pd.Series(1.0, index=bg.index, dtype=float),
    ]
    weight = candidates[0].copy()
    for fallback in candidates[1:]:
        group_sum = weight.groupby(group).transform("sum")
        weight = weight.where(group_sum.gt(0.0), fallback)
    group_sum = weight.groupby(group).transform("sum")
    share = np.divide(
        weight.to_numpy(dtype=float),
        group_sum.to_numpy(dtype=float),
        out=np.zeros(len(bg), dtype=float),
        where=group_sum.gt(0.0).to_numpy(dtype=bool),
    )
    return pd.Series(share, index=bg.index, dtype=float)


def apply_rare_offense_tract_support(
    bg: pd.DataFrame,
    tract: pd.DataFrame,
    *,
    tract_geo_id_col: str = "tract_id",
    count_first_composites: bool = False,
) -> pd.DataFrame:
    """RARE_OFFENSE_TRACT_SUPPORT publication policy (docs/STATE.md decision record).

    Applied to a finalized block-group surface at publication time. Murder and rape are
    Poisson-noise-dominated at block-group support, so their published per-offense index and rate
    point estimates are carried only at census tract and coarser. This step:

      1. Re-expresses the two rare-offense-sensitive aggregate indices at block-group support,
         with the murder/rape terms taken at their honest (tract) support:
           - index_total_harm: murder/rape enter as the tract count spread within the tract by
             person-exposure share (conserving every tract total exactly); the five volume
             offenses enter at full block-group resolution; the CHI sentencing-day weights are
             untouched. The aggregate is more robust than its parts, and stays recomputable from
             published fields (five block-group counts + tract murder/rape counts + exposure
             shares + fixed weights).
           - index_total_equal_offense: the murder/rape terms use the parent tract's per-offense
             index; the five volume offenses use the block-group index.
      2. Nulls the block-group murder/rape per-offense index and rate point fields (and their
         confidence intervals). Expected counts and all reliability/diagnostic metadata remain,
         so the counts still reconcile block group -> tract -> national.

    The tract surface is returned untouched: it carries murder/rape at its native support.

    On the count-first composite lane (`count_first_composites=True`) step 1 has nothing to do for
    the burdens: they are functions of expected counts and resident population, both of which this
    policy leaves alone, and the harm burden is already null at block-group support by its own
    tract-support rule. The two index-average relative scores still take their rare terms at tract
    support, under their renamed columns. That the burdens come through this step bit-identical is
    asserted in the tests -- it is the count-first construction's whole point: no composite needs
    an override when it never consumed a per-offense index.
    """
    out = bg.copy()
    group = out[tract_geo_id_col].astype("string").str.zfill(11)
    share = _within_tract_person_exposure_share(out, group)

    redistributed_counts: dict[str, pd.Series] = {}
    for offense in () if count_first_composites else RARE_OFFENSE_TRACT_SUPPORT:
        expected = pd.to_numeric(out[_expected_count_col(offense)], errors="coerce").fillna(0.0).clip(lower=0.0)
        tract_total = expected.groupby(group).transform("sum")
        redistributed = tract_total * share
        conservation = (
            pd.DataFrame({"tract": group, "expected": expected, "redistributed": redistributed})
            .groupby("tract")[["expected", "redistributed"]]
            .sum()
        )
        max_abs_dev = (
            float((conservation["redistributed"] - conservation["expected"]).abs().max())
            if not conservation.empty
            else 0.0
        )
        if not np.isfinite(max_abs_dev) or max_abs_dev > 1e-6:
            raise ValueError(
                f"rare-offense tract redistribution not conserved for {offense}: "
                f"max within-tract absolute deviation {max_abs_dev}"
            )
        redistributed_counts[offense] = redistributed

    if not count_first_composites:
        out["index_total_harm"] = _harm_weighted_total_index(
            out,
            list(OFFENSES_7),
            count_overrides=redistributed_counts,
        )

    tract_index_cols = [f"index_{offense}_primary" for offense in RARE_OFFENSE_TRACT_SUPPORT]
    tract_index = tract[[tract_geo_id_col, *tract_index_cols]].copy()
    tract_index[tract_geo_id_col] = tract_index[tract_geo_id_col].astype("string").str.zfill(11)
    joined = (
        pd.DataFrame({tract_geo_id_col: group.to_numpy()})
        .merge(tract_index, on=tract_geo_id_col, how="left")
    )
    index_overrides = {
        offense: pd.Series(joined[f"index_{offense}_primary"].to_numpy(dtype=float), index=out.index)
        for offense in RARE_OFFENSE_TRACT_SUPPORT
    }
    # Two rules meet here, and only one of them is an override.
    #
    # Murder and rape take their term from the parent TRACT because the block-group point is
    # withheld by a SUPPORT policy -- one year of murder at block group is Poisson noise, not a
    # suppression. The tract index is taken verbatim: where the tract itself publishes no index,
    # the component is genuinely suppressed and the composite goes null with it.
    #
    # The five volume offenses get NO override at all. A cell whose burglary, larceny, robbery,
    # assault or MVT point is suppressed for denominator invalidity has no publishable value for
    # that offense, and substituting its parent tract's value published a composite asserting a
    # component the cell itself refuses to show (Poarch Creek: index_larceny_primary NaN beside
    # multi_offense_relative_score_event_weighted 1,878.8). Suppression propagates instead.
    out[
        multi_offense_score_column("index_total_equal_offense", count_first=count_first_composites)
    ] = _full_component_index_composite(
        out,
        list(OFFENSES_7),
        index_suffix="primary",
        weights={offense: 1.0 for offense in OFFENSES_7},
        index_overrides=index_overrides,
    )
    # The national-count-weighted composite averages per-offense primary indices too, so its
    # murder/rape terms likewise come from tract support. Their weight here is only the offenses'
    # tiny national count share, so this barely moves the surface -- it keeps the aggregate
    # recomputable from published fields and drains the last of the block-group murder noise out
    # of the default total layer.
    out[
        multi_offense_score_column(
            "index_total_primary_event_weighted", count_first=count_first_composites
        )
    ] = _full_component_index_composite(
        out,
        list(OFFENSES_7),
        index_suffix="primary",
        weights=_national_expected_count_weights(out, list(OFFENSES_7)),
        index_overrides=index_overrides,
    )
    if count_first_composites:
        out[PERSONAL_RELATIVE_SCORE_COLUMN] = _full_component_index_composite(
            out,
            list(PERSONAL_OFFENSES),
            index_suffix="primary",
            weights=_national_expected_count_weights(out, list(PERSONAL_OFFENSES)),
            index_overrides={
                offense: index_overrides[offense]
                for offense in PERSONAL_OFFENSES
                if offense in index_overrides
            },
        )
        out[PROPERTY_RELATIVE_SCORE_COLUMN] = _full_component_index_composite(
            out,
            list(PROPERTY_OFFENSES),
            index_suffix="primary",
            weights=_national_expected_count_weights(out, list(PROPERTY_OFFENSES)),
            index_overrides={
                offense: index_overrides[offense]
                for offense in PROPERTY_OFFENSES
                if offense in index_overrides
            },
        )

    for offense in RARE_OFFENSE_TRACT_SUPPORT:
        for field in _rare_offense_published_point_fields(offense):
            if field in out.columns:
                out[field] = np.nan
        # The transient-exposure guard reads `index_{offense}_resident`, which step 2 has just
        # nulled at block-group support: the flag is evaluated before this policy runs, so 88
        # of its 162 firings (murder 86, rape 2) sat on an index no consumer can ever see.
        # Cleared here rather than left as a flag whose own trigger field is null on the row.
        flag_col = f"transient_exposure_likely_{offense}"
        if flag_col in out.columns:
            out[flag_col] = False
        # Publication flags describe the point payload, not the pre-policy
        # estimator. Once the block-group point is null, every corresponding
        # publishable flag must say false and every suppression flag true.
        for publishable_col in (
            f"index_publishable_{offense}",
            f"primary_index_publishable_{offense}",
            f"index_{offense}_resident_publishable",
        ):
            if publishable_col in out.columns:
                out[publishable_col] = False
        for suppressed_col in (
            f"primary_index_suppressed_{offense}",
            f"index_{offense}_resident_suppressed",
        ):
            if suppressed_col in out.columns:
                out[suppressed_col] = True
        recommendation_col = f"recommended_display_geography_{offense}"
        if recommendation_col in out.columns:
            parent_available = pd.to_numeric(index_overrides[offense], errors="coerce").notna()
            out[recommendation_col] = pd.Series(
                np.where(parent_available, "tract", "not_published"),
                index=out.index,
                dtype="string",
            )

    # The resident volume aggregates retain murder/rape at their count share even though the
    # block-group point payload and its publishability flags are absent by policy. Recompute them
    # after that policy is final so both their row mask and national normalizer are derived from
    # the published component contract. In particular, a rare-offense-only suppression must not
    # leave a stale null (or stale national normalizer) on an otherwise publishable aggregate.
    #
    # Nothing to recompute on the count-first lane: its burdens never read a publishability flag,
    # so a rare-offense point suppression cannot leave a stale null or a stale normalizer on them.
    resident_specs = (
        {}
        if count_first_composites
        else {
            "index_total_part1_resident": list(OFFENSES_7),
            "index_personal_part1_resident": list(PERSONAL_OFFENSES),
            "index_property_part1_resident": list(PROPERTY_OFFENSES),
        }
    )
    for field, offenses in resident_specs.items():
        ignored = frozenset(offenses).intersection(RARE_OFFENSE_TRACT_SUPPORT)
        out[field] = _resident_part1_index(
            out,
            offenses,
            ignored_component_publishability=ignored,
        )[0]
    return out


def _attach_primary_denominator_for_audit(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["primary_denominator_type"] = out["offense"].map(PRIMARY_DENOMINATOR_BY_OFFENSE).astype("string")
    out["primary_denominator_raw"] = 0.0
    for denominator_type, source_col in DENOMINATOR_SOURCE_COLUMNS.items():
        if denominator_type == "resident" or source_col not in out.columns:
            continue
        mask = out["primary_denominator_type"].eq(denominator_type)
        out.loc[mask, "primary_denominator_raw"] = pd.to_numeric(
            out.loc[mask, source_col],
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
    return out


def _components_with_primary_denominator(all_components: pd.DataFrame, bg_covariates: pd.DataFrame) -> pd.DataFrame:
    components = all_components.copy()
    components["bg_id"] = components["bg_id"].astype("string").str.zfill(12)
    components["state_fips"] = components["state_fips"].astype("string").str.zfill(2)
    components["component_count"] = pd.to_numeric(components["component_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    cov = bg_covariates.copy()
    cov["bg_id"] = cov["bg_id"].astype("string").str.zfill(12)
    exposure_cov = (
        pd.to_numeric(cov["exposure_proxy_2024"], errors="coerce")
        if "exposure_proxy_2024" in cov.columns
        else pd.Series(np.nan, index=cov.index, dtype=float)
    )
    jobs_cov = (
        pd.to_numeric(cov["daytime_population_jobs_proxy"], errors="coerce")
        if "daytime_population_jobs_proxy" in cov.columns
        else pd.Series(0.0, index=cov.index, dtype=float)
    )
    cov["exposure_proxy_2024"] = exposure_cov.combine_first(jobs_cov).fillna(0.0).clip(lower=0.0)
    if LANDSCAN_DAY_POP_COLUMN not in cov.columns:
        cov[LANDSCAN_DAY_POP_COLUMN] = 0.0
    cov[LANDSCAN_DAY_POP_COLUMN] = pd.to_numeric(
        cov[LANDSCAN_DAY_POP_COLUMN],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    if "landscan_day_lifted_person_exposure" not in cov.columns:
        cov["landscan_day_lifted_person_exposure"] = False
    cov["landscan_day_lifted_person_exposure"] = (
        cov["landscan_day_lifted_person_exposure"].fillna(False).astype(bool)
    )
    if "land_area_sq_mi" not in cov.columns:
        cov["land_area_sq_mi"] = 0.0
    cov["land_area_sq_mi"] = pd.to_numeric(cov["land_area_sq_mi"], errors="coerce").fillna(0.0).clip(lower=0.0)
    components = components.merge(
        cov[
            [
                "bg_id",
                "population",
                "daytime_population_jobs_proxy",
                "exposure_proxy_2024",
                LANDSCAN_DAY_POP_COLUMN,
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
            ]
        ],
        on="bg_id",
        how="left",
    )
    return _attach_primary_denominator_for_audit(components)


def _softened_retained_rate_ratio(
    rate_ratio: pd.Series, *, cap_ratio: float, nu: float
) -> pd.Series:
    """The rate ratio a unit KEEPS under the soft envelope.

    At or below the cap the unit is untouched, exactly as under the hard cap. Above it the excess
    is compressed rather than deleted, so 40x and 400x remain distinguishable after the repair and
    the map keeps an ordering it earned. Strictly increasing and strictly below the identity above
    the cap, so the repair still always removes mass and never adds any.
    """
    values = pd.to_numeric(rate_ratio, errors="coerce")
    above = values.gt(float(cap_ratio)) & np.isfinite(values)
    if not bool(above.any()):
        return values
    excess = np.log(values[above].to_numpy(dtype=float) / float(cap_ratio))
    retained = float(cap_ratio) * np.exp(compress_log_excess(excess, nu=nu))
    out = values.copy()
    out.loc[above] = retained
    return out


def _apply_model_only_allocation_envelopes(
    components: pd.DataFrame,
    *,
    soft_shrinkage: bool = False,
    soft_shrinkage_nu: float = SOFT_SHRINKAGE_NU,
) -> pd.DataFrame:
    """Clip fabricated model-only allocation tails and conserve every source total.

    The published tail diagnostic assigns each unit to the jurisdiction contributing
    the most all-offense mass, then compares the offense rate with that jurisdiction's
    median. We reproduce that definition here on the not-yet-published component
    counts. For units above the measured direct-feed envelope, each contributing
    model-only component gives up its proportional share of the excess. That mass is
    redistributed within the same source jurisdiction/offense, proportional to its
    untouched model-only components. Direct-posterior components never donate or
    receive mass.

    A source jurisdiction with no other model-only recipient cannot move mass without
    violating jurisdiction conservation. Its proposed removal is cancelled and
    recorded as unredistributable; the measured P99.9 gate, rather than an impossible
    cross-jurisdiction transfer, owns that residue.
    """
    if components.empty:
        return components.copy()

    out = components.copy()
    out["component_count"] = pd.to_numeric(
        out["component_count"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    out["allocation_envelope_component_count_before"] = out["component_count"]
    out["allocation_envelope_component_share_before"] = pd.to_numeric(
        out.get("component_share"), errors="coerce"
    )
    out["allocation_envelope_policy"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["allocation_envelope_cap_ratio"] = np.nan
    out["allocation_envelope_clipped_source"] = False
    out["allocation_envelope_redistribution_recipient"] = False
    out["allocation_envelope_unredistributable_source"] = False
    if soft_shrinkage:
        out[MODEL_ONLY_ALLOCATION_ENVELOPE_MODE_COLUMN] = ENVELOPE_MODE_SOFT_SHRINKAGE
        out["allocation_envelope_rate_ratio"] = np.nan
        out["allocation_envelope_retained_rate_ratio"] = np.nan

    # Component fallback rows can retain a municipal-looking `jurisdiction_id`
    # while their `jurisdiction_type` routes them to the statewide overlap control.
    # Conserve the canonical published control identity, not that raw fallback
    # label, or redistribution can leak a few thousandths between a city and the
    # overlap layer even while the raw-label assertion passes.
    conservation_col = "_allocation_envelope_control_jurisdiction_id"
    out[conservation_col] = out["jurisdiction_id"].astype("string")
    service_scope = out.get(
        "service_scope_id", pd.Series(pd.NA, index=out.index, dtype="string")
    ).astype("string")
    service_wide = service_scope.notna() & service_scope.str.strip().ne("")
    if "source_state_fips" not in out.columns:
        out["source_state_fips"] = out["state_fips"]
    source_state = (
        out["source_state_fips"]
        .fillna(out["state_fips"])
        .astype("string")
        .str.zfill(2)
    )
    conservation_state_col = "_allocation_envelope_conservation_state_fips"
    out[conservation_state_col] = out["state_fips"].astype("string").str.zfill(2)
    out.loc[service_wide, conservation_state_col] = source_state.loc[service_wide]
    out.loc[service_wide, conservation_col] = (
        "service:" + service_scope.loc[service_wide] + ":" + source_state.loc[service_wide]
    )
    overlap = out["jurisdiction_type"].astype("string").str.contains(
        "overlap", na=False
    ) & ~service_wide
    out.loc[overlap, conservation_col] = (
        out.loc[overlap, "state_fips"].astype("string") + ":statewide_overlap_layer"
    )
    remainder = out["jurisdiction_type"].astype("string").isin(
        ["localized_remainder_county_layer", "localized_remainder_residual_layer"]
    )
    out.loc[remainder, conservation_col] = (
        out.loc[remainder, "state_fips"].astype("string")
        + ":state_nonmunicipal_remainder"
    )
    group_cols = [conservation_state_col, conservation_col, "offense"]
    totals_before = out.groupby(group_cols, dropna=False)["component_count"].sum()

    # The tail's jurisdiction identity is mass-dominant across all seven offenses,
    # matching the snapshot census/provenance derivation that measured the constants.
    for offense, unit_col, cap_ratio in MODEL_ONLY_ALLOCATION_ENVELOPE_SPECS:
        all_unit_mass = (
            out.groupby([unit_col, "jurisdiction_id"], dropna=False)["component_count"]
            .sum()
            .reset_index(name="_all_offense_mass")
        )
        dominant = all_unit_mass.loc[
            all_unit_mass.groupby(unit_col, dropna=False)["_all_offense_mass"].idxmax(),
            [unit_col, "jurisdiction_id"],
        ].rename(columns={"jurisdiction_id": "_dominant_jurisdiction_id"})

        offense_rows = out["offense"].astype("string").eq(offense)
        if not bool(offense_rows.any()):
            continue
        active = out.get(
            "city_incident_posterior_active",
            pd.Series(False, index=out.index),
        ).eq(True).fillna(False)

        # Covariates are duplicated across component rows. Collapse to BG first so a
        # tract denominator is the sum of its unique BG denominators, not a sum over
        # overlapping component identities.
        bg_cov = (
            out.loc[offense_rows, ["bg_id", "tract_id", "primary_denominator_raw", "households_total"]]
            .drop_duplicates("bg_id")
            .copy()
        )
        bg_cov["_denominator"] = pd.to_numeric(
            bg_cov["primary_denominator_raw"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        bg_cov["_households"] = pd.to_numeric(
            bg_cov["households_total"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        if unit_col == "bg_id":
            unit_cov = bg_cov[["bg_id", "_denominator", "_households"]].copy()
        else:
            unit_cov = (
                bg_cov.groupby("tract_id", dropna=False)
                .agg(_denominator=("_denominator", "sum"), _households=("_households", "sum"))
                .reset_index()
            )
        unit_cov["_special_use"] = unit_cov[unit_col].astype("string").str.slice(2, 4).eq(
            SPECIAL_USE_TRACT_PREFIX
        )

        unit = (
            out.loc[offense_rows]
            .groupby(unit_col, dropna=False)
            .agg(
                _offense_count=("component_count", "sum"),
                _direct_active=(
                    "city_incident_posterior_active",
                    lambda values: values.eq(True).fillna(False).any(),
                ),
            )
            .reset_index()
        )
        unit = unit.merge(dominant, on=unit_col, how="left", validate="one_to_one")
        unit = unit.merge(unit_cov, on=unit_col, how="left", validate="one_to_one")
        unit["_eligible"] = (
            unit["_households"].ge(float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR))
            & unit["_denominator"].ge(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
            & ~unit["_special_use"].fillna(False).astype(bool)
        )
        unit["_rate"] = np.where(
            unit["_denominator"].gt(0.0),
            unit["_offense_count"] / unit["_denominator"],
            np.nan,
        )
        eligible_rate = unit["_rate"].where(unit["_eligible"])
        unit["_jurisdiction_median_rate"] = eligible_rate.groupby(
            unit["_dominant_jurisdiction_id"], dropna=False
        ).transform("median")
        unit["_rate_ratio"] = unit["_rate"] / unit["_jurisdiction_median_rate"].replace(0.0, np.nan)
        unit["_retained_ratio"] = (
            _softened_retained_rate_ratio(
                unit["_rate_ratio"], cap_ratio=float(cap_ratio), nu=float(soft_shrinkage_nu)
            )
            if soft_shrinkage
            else float(cap_ratio)
        )
        unit["_cap_count"] = (
            unit["_retained_ratio"]
            * unit["_jurisdiction_median_rate"]
            * unit["_denominator"]
        )
        unit["_remove_needed"] = (
            unit["_offense_count"] - unit["_cap_count"]
        ).clip(lower=0.0)
        touched = (
            unit["_eligible"]
            & ~unit["_direct_active"].fillna(False).astype(bool)
            & unit["_rate_ratio"].gt(float(cap_ratio))
            & unit["_remove_needed"].gt(0.0)
        )
        if not bool(touched.any()):
            continue

        touched_units = unit.loc[
            touched, [unit_col, "_offense_count", "_remove_needed", "_rate_ratio", "_retained_ratio"]
        ].copy()
        work = out.loc[offense_rows].copy()
        work["_row_index"] = work.index
        work = work.merge(touched_units, on=unit_col, how="left", validate="many_to_one")
        work["_remove_needed"] = pd.to_numeric(
            work["_remove_needed"], errors="coerce"
        ).fillna(0.0)
        work["_offense_count"] = pd.to_numeric(
            work["_offense_count"], errors="coerce"
        ).fillna(0.0)
        work["_proposed_remove"] = np.where(
            work["_offense_count"].gt(0.0) & ~active.loc[work["_row_index"]].to_numpy(dtype=bool),
            work["_remove_needed"]
            * work["component_count"]
            / work["_offense_count"].replace(0.0, np.nan),
            0.0,
        )
        work["_proposed_remove"] = pd.to_numeric(
            work["_proposed_remove"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)

        unit_direct = unit.set_index(unit_col)["_direct_active"]
        touched_ids = set(touched_units[unit_col].astype(str))
        work["_recipient"] = (
            ~work[unit_col].astype(str).isin(touched_ids)
            & ~work[unit_col].map(unit_direct).fillna(False).astype(bool)
            & ~active.loc[work["_row_index"]].to_numpy(dtype=bool)
            & work["component_count"].gt(0.0)
        )
        recipient_weight = work["component_count"].where(work["_recipient"], 0.0)
        recipient_total = recipient_weight.groupby(
            work[conservation_col], dropna=False
        ).transform("sum")
        can_redistribute = work[conservation_col].map(
            recipient_total.groupby(work[conservation_col], dropna=False).max().gt(0.0)
        ).fillna(False).astype(bool)
        work["_remove"] = work["_proposed_remove"].where(can_redistribute, 0.0)
        removed_total = work["_remove"].groupby(
            work[conservation_col], dropna=False
        ).transform("sum")
        work["_add"] = np.where(
            work["_recipient"] & recipient_total.gt(0.0),
            removed_total * work["component_count"] / recipient_total.replace(0.0, np.nan),
            0.0,
        )
        work["_add"] = pd.to_numeric(work["_add"], errors="coerce").fillna(0.0)
        updated = (work["component_count"] - work["_remove"] + work["_add"]).clip(lower=0.0)
        idx = work["_row_index"].to_numpy()
        out.loc[idx, "component_count"] = updated.to_numpy(dtype=float)
        policy = f"model_only_{offense}_{'bg' if unit_col == 'bg_id' else 'tract'}_rate_ratio_cap"
        changed_source = work["_remove"].gt(1e-15).to_numpy()
        changed_recipient = work["_add"].gt(1e-15).to_numpy()
        stranded = (
            work["_proposed_remove"].gt(1e-15) & ~can_redistribute
        ).to_numpy()
        affected = changed_source | changed_recipient | stranded
        out.loc[idx[affected], "allocation_envelope_policy"] = policy
        out.loc[idx[affected], "allocation_envelope_cap_ratio"] = float(cap_ratio)
        out.loc[idx[changed_source], "allocation_envelope_clipped_source"] = True
        out.loc[idx[changed_recipient], "allocation_envelope_redistribution_recipient"] = True
        out.loc[idx[stranded], "allocation_envelope_unredistributable_source"] = True
        if soft_shrinkage:
            # The compressor's own inputs and output, per affected source row. Published so the
            # release validator -- and any reader -- can re-derive the retained ratio from the
            # artifact and the declared nu alone, instead of taking the producer's word for it.
            source_rows = idx[changed_source | stranded]
            out.loc[source_rows, "allocation_envelope_rate_ratio"] = pd.to_numeric(
                work.loc[changed_source | stranded, "_rate_ratio"], errors="coerce"
            ).to_numpy(dtype=float)
            out.loc[source_rows, "allocation_envelope_retained_rate_ratio"] = pd.to_numeric(
                work.loc[changed_source | stranded, "_retained_ratio"], errors="coerce"
            ).to_numpy(dtype=float)

    totals_after = out.groupby(group_cols, dropna=False)["component_count"].sum()
    delta = totals_after.sub(totals_before, fill_value=0.0)
    if not np.allclose(delta.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            "model-only allocation envelope repair failed jurisdiction conservation: "
            + str(delta.loc[delta.abs().gt(1e-9)].head(20).to_dict())
        )
    group_total = out.groupby(group_cols, dropna=False)["component_count"].transform("sum")
    out["component_share"] = np.where(
        group_total.gt(0.0), out["component_count"] / group_total, 0.0
    )
    out["allocation_envelope_component_count_after"] = out["component_count"]
    out["allocation_envelope_component_delta"] = (
        out["allocation_envelope_component_count_after"]
        - out["allocation_envelope_component_count_before"]
    )
    out["allocation_envelope_component_share_after"] = out["component_share"]
    return out.drop(columns=[conservation_col, conservation_state_col])


def _dominant_bg_jurisdiction(bg_crosswalk: pd.DataFrame) -> pd.DataFrame:
    crosswalk = bg_crosswalk[
        ["block_group_geoid", "jurisdiction_id", "jurisdiction_type", "allocation_share"]
    ].copy()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    crosswalk["allocation_share"] = pd.to_numeric(crosswalk["allocation_share"], errors="coerce").fillna(0.0)
    dominant = (
        crosswalk.sort_values(["block_group_geoid", "allocation_share"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("block_group_geoid", keep="first")
        .rename(
            columns={
                "jurisdiction_id": "eb_jurisdiction_id",
                "jurisdiction_type": "eb_jurisdiction_type",
            }
        )
    )
    return dominant[["block_group_geoid", "eb_jurisdiction_id", "eb_jurisdiction_type"]].reset_index(drop=True)


def _dominant_tract_jurisdiction(
    bg_crosswalk: pd.DataFrame, bg_surface: pd.DataFrame, *, population_col: str
) -> pd.DataFrame:
    crosswalk = bg_crosswalk[["state_fips", "block_group_geoid", "jurisdiction_id", "allocation_share"]].copy()
    crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    crosswalk["jurisdiction_id"] = crosswalk["jurisdiction_id"].astype("string")
    crosswalk["allocation_share"] = pd.to_numeric(crosswalk["allocation_share"], errors="coerce").fillna(0.0).clip(lower=0.0)

    bg = bg_surface[["block_group_geoid", "state_fips", population_col]].copy()
    bg["block_group_geoid"] = bg["block_group_geoid"].astype("string").str.zfill(12)
    bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    bg[population_col] = pd.to_numeric(bg[population_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    crosswalk = crosswalk.merge(bg, on=["block_group_geoid", "state_fips"], how="inner")
    crosswalk["tract_id"] = crosswalk["block_group_geoid"].str.slice(0, 11)
    crosswalk["resident_weighted_mass"] = crosswalk["allocation_share"] * crosswalk[population_col]
    resident_total = crosswalk.groupby(["state_fips", "tract_id"], dropna=False)["resident_weighted_mass"].transform("sum")
    crosswalk["dominance_mass"] = np.where(
        resident_total.gt(0.0),
        crosswalk["resident_weighted_mass"],
        crosswalk["allocation_share"],
    )
    grouped = (
        crosswalk.groupby(["state_fips", "tract_id", "jurisdiction_id"], dropna=False)["dominance_mass"]
        .sum()
        .reset_index()
    )
    total = grouped.groupby(["state_fips", "tract_id"], dropna=False)["dominance_mass"].transform("sum")
    grouped["dominant_jurisdiction_share"] = np.where(
        total.gt(0.0),
        grouped["dominance_mass"] / total,
        np.nan,
    )
    dominant = (
        grouped.sort_values(
            ["state_fips", "tract_id", "dominance_mass", "jurisdiction_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["state_fips", "tract_id"], keep="first")
        .rename(columns={"jurisdiction_id": "dominant_eb_jurisdiction_id"})
    )
    missing_share = dominant["dominant_jurisdiction_share"].isna()
    if bool(missing_share.any()):
        sample = dominant.loc[missing_share, "tract_id"].astype("string").head(10).tolist()
        raise ValueError(f"unable to assign dominant tract EB jurisdiction for {int(missing_share.sum())} tract(s): {sample}")
    dominant["mixed_jurisdiction_flag"] = dominant["dominant_jurisdiction_share"].lt(0.999999)
    return dominant[
        ["state_fips", "tract_id", "dominant_eb_jurisdiction_id", "dominant_jurisdiction_share", "mixed_jurisdiction_flag"]
    ].reset_index(drop=True)


def _rollup_tracts_from_bg(
    bg_surface: pd.DataFrame, tract_jurisdiction: pd.DataFrame, *, population_col: str
) -> pd.DataFrame:
    rollup_cols = (
        [_expected_count_col(offense) for offense in OFFENSES_7]
        # Footprint-derived MASS, not its share: a share cannot be summed, and rolling the mass
        # up and re-deriving the share inside `_finalize_output` gives the tract surface the same
        # compositional quantity the block-group surface has, with no second code path.
        + [_footprint_derived_count_col(offense) for offense in OFFENSES_7]
        + [
            population_col,
            "daytime_population_jobs_proxy",
            "exposure_proxy_2024",
            LANDSCAN_DAY_POP_COLUMN,
            "person_exposure_before_hq_jobs_cap",
            "person_exposure_hq_jobs_cap",
            "households_total",
            "commercial_premises_total",
            "destination_poi_total",
            "lodes_manufacturing_jobs",
            "lodes_wholesale_jobs",
            "lodes_retail_jobs",
            "lodes_transport_warehouse_jobs",
            "lodes_industrial_jobs",
            "burglary_premises_total",
            "aggregate_vehicles_total",
            "mvt_commuter_vehicle_proxy",
            "vehicle_exposure_2024",
            "land_area_sq_mi",
        ]
        # Opportunity normalizers are extensive person-equivalent mass, so a tract's normalizer
        # is the sum of its block groups' -- the same fold every other denominator gets. Present
        # only when the v2 lane built them.
        + [
            opportunity_normalizer_column(offense)
            for offense in ENSEMBLE_OFFENSES
            if opportunity_normalizer_column(offense) in bg_surface.columns
        ]
        # Every special-use classification input is an extensive count -- of people, jobs,
        # institutions or 30 m land-cover pixels -- so a tract's inputs are the sum of its block
        # groups' and the tract classifies itself through the same cascade. That is why the
        # taxonomy has no tract-specific code path and no majority-vote-over-children rule.
        + [
            column
            for column in SPECIAL_USE_FEATURE_COLUMNS
            if column in bg_surface.columns
        ]
    )
    support_count_cols = [
        f"direct_incident_support_count_{offense}"
        for offense in OFFENSES_7
        if f"direct_incident_support_count_{offense}" in bg_surface.columns
    ]
    tract = (
        bg_surface.groupby(["state_fips", "tract_id"], dropna=False)[list(rollup_cols) + support_count_cols]
        .sum()
        .reset_index()
    )
    if "burglary_commercial_exposure_weight" in bg_surface.columns:
        weight = (
            bg_surface.groupby(["state_fips", "tract_id"], dropna=False)["burglary_commercial_exposure_weight"]
            .max()
            .rename("burglary_commercial_exposure_weight")
            .reset_index()
        )
        tract = tract.merge(weight, on=["state_fips", "tract_id"], how="left")
    for weight_col in [
        "burglary_destination_poi_exposure_weight",
        "burglary_retail_jobs_exposure_weight",
        "burglary_industrial_jobs_exposure_weight",
    ]:
        if weight_col in bg_surface.columns:
            weight = (
                bg_surface.groupby(["state_fips", "tract_id"], dropna=False)[weight_col]
                .max()
                .rename(weight_col)
                .reset_index()
            )
            tract = tract.merge(weight, on=["state_fips", "tract_id"], how="left")
    if "landscan_day_lifted_person_exposure" in bg_surface.columns:
        lift = (
            bg_surface.groupby(["state_fips", "tract_id"], dropna=False)["landscan_day_lifted_person_exposure"]
            .max()
            .rename("landscan_day_lifted_person_exposure")
            .reset_index()
        )
        tract = tract.merge(lift, on=["state_fips", "tract_id"], how="left")
    for flag_col in ["person_exposure_hq_jobs_cap_candidate", "person_exposure_hq_jobs_capped"]:
        if flag_col in bg_surface.columns:
            flag = (
                bg_surface.groupby(["state_fips", "tract_id"], dropna=False)[flag_col]
                .max()
                .rename(flag_col)
                .reset_index()
            )
            tract = tract.merge(flag, on=["state_fips", "tract_id"], how="left")
    if "county_auto_commute_vehicle_share" in bg_surface.columns:
        veh = pd.to_numeric(bg_surface.get("vehicle_exposure_2024"), errors="coerce").fillna(0.0).clip(lower=0.0)
        weighted_share = (
            pd.DataFrame(
                {
                    "state_fips": bg_surface["state_fips"].astype("string").str.zfill(2),
                    "tract_id": bg_surface["tract_id"].astype("string").str.zfill(11),
                    "_share_weighted": pd.to_numeric(
                        bg_surface["county_auto_commute_vehicle_share"], errors="coerce"
                    ).fillna(0.0)
                    * veh,
                    "_vehicle_exposure": veh,
                }
            )
            .groupby(["state_fips", "tract_id"], dropna=False)[["_share_weighted", "_vehicle_exposure"]]
            .sum()
            .reset_index()
        )
        weighted_share["county_auto_commute_vehicle_share"] = np.where(
            weighted_share["_vehicle_exposure"].gt(0.0),
            weighted_share["_share_weighted"] / weighted_share["_vehicle_exposure"],
            np.nan,
        )
        tract = tract.merge(
            weighted_share[["state_fips", "tract_id", "county_auto_commute_vehicle_share"]],
            on=["state_fips", "tract_id"],
            how="left",
        )
    for offense in OFFENSES_7:
        flag_col = f"direct_incident_support_flag_{offense}"
        years_col = f"direct_incident_support_years_{offense}"
        year_min_col = f"direct_incident_support_year_min_{offense}"
        year_max_col = f"direct_incident_support_year_max_{offense}"
        source_col = f"numerator_support_source_{offense}"
        group = bg_surface.groupby(["state_fips", "tract_id"], dropna=False)
        if flag_col in bg_surface.columns:
            tract_flag = group[flag_col].max().rename(flag_col).reset_index()
            tract = tract.merge(tract_flag, on=["state_fips", "tract_id"], how="left")
        if years_col in bg_surface.columns:
            tract_years = group[years_col].max().rename(years_col).reset_index()
            tract = tract.merge(tract_years, on=["state_fips", "tract_id"], how="left")
        if year_min_col in bg_surface.columns:
            tract_year_min = group[year_min_col].min().rename(year_min_col).reset_index()
            tract = tract.merge(tract_year_min, on=["state_fips", "tract_id"], how="left")
        if year_max_col in bg_surface.columns:
            tract_year_max = group[year_max_col].max().rename(year_max_col).reset_index()
            tract = tract.merge(tract_year_max, on=["state_fips", "tract_id"], how="left")
        if flag_col in tract.columns:
            tract[source_col] = np.where(tract[flag_col].fillna(False).astype(bool), "direct_city_incident", "model_only")
    tract["state_fips"] = tract["state_fips"].astype("string").str.zfill(2)
    tract["tract_id"] = tract["tract_id"].astype("string").str.zfill(11)
    out = tract.merge(tract_jurisdiction, on=["state_fips", "tract_id"], how="left")
    missing = out["dominant_eb_jurisdiction_id"].isna() | out["dominant_eb_jurisdiction_id"].astype("string").str.strip().eq("")
    if bool(missing.any()):
        sample = out.loc[missing, "tract_id"].astype("string").head(10).tolist()
        raise ValueError(f"tract rollup missing dominant EB jurisdiction for {int(missing.sum())} tract(s): {sample}")
    return out


def _redistribute_zero_target_components(
    all_components: pd.DataFrame,
    bg_covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "state_fips",
        "source_state_fips",
        "bg_id",
        "tract_id",
        "jurisdiction_id",
        "jurisdiction_type",
        "offense",
        "service_scope_id",
        "canonical_target_ori",
        "source_target_count",
        "component_count_before",
        "component_count_after",
        "redistributed_delta",
        "model_share",
        "city_residual_transfer_policy",
        "city_residual_transfer_tau",
        "city_residual_predicted_log_ratio",
        "city_incident_posterior_active",
        "incident_count",
        "city_posterior_q",
        "city_posterior_alpha",
        "city_posterior_prior_fraction",
        "city_posterior_direct_share",
        "city_posterior_share",
        "city_posterior_model_prior_raw",
        "city_posterior_model_prior_share",
        "rare_offense_allocation_policy",
        "rare_offense_information_constant",
        "rare_offense_effective_model_information",
        "rare_offense_model_information_weight",
        "rare_offense_residential_exposure_weight",
        "rare_offense_tract_model_share",
        "rare_offense_tract_exposure_share",
        "rare_offense_within_tract_exposure_share",
        "component_share",
        "bg_responsibility_population_coverage_share",
        "responsibility_fraction_basis",
        # Present only on a soft-envelope build, so a hard-cap audit is byte-identical to what it
        # was before the lane existed and a soft one says so on every affected row.
        MODEL_ONLY_ALLOCATION_ENVELOPE_MODE_COLUMN,
        "allocation_envelope_rate_ratio",
        "allocation_envelope_retained_rate_ratio",
        "allocation_envelope_policy",
        "allocation_envelope_cap_ratio",
        "allocation_envelope_clipped_source",
        "allocation_envelope_redistribution_recipient",
        "allocation_envelope_unredistributable_source",
        "allocation_envelope_component_count_before",
        "allocation_envelope_component_count_after",
        "allocation_envelope_component_delta",
        "allocation_envelope_component_share_before",
        "allocation_envelope_component_share_after",
        "primary_denominator_type",
        "primary_denominator_raw",
        "land_area_sq_mi",
        "population",
        "daytime_population_jobs_proxy",
        "exposure_proxy_2024",
        LANDSCAN_DAY_POP_COLUMN,
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
        "group_zero_source_mass",
        "group_recipient_count",
        "group_recipient_delta_mass",
        "group_total_before",
        "group_total_after",
        "group_total_delta",
        "audit_reason",
        "redistribution_status",
    ]
    if all_components.empty:
        return all_components.copy(), pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    components = (
        all_components.copy()
        if "primary_denominator_raw" in all_components.columns
        else _components_with_primary_denominator(all_components, bg_covariates)
    )
    # A water-centroid fallback can carry the wrong jurisdiction label (Milwaukee's
    # lake cell arrived as the county remainder while the land recipients are the
    # city). Route water at the county/offense level so the bad fallback label does
    # not strand it; this is the narrowest geography that always contains the true
    # land destination. The broad zero-denominator switch remains off this release.
    components["_routing_county"] = components["bg_id"].astype("string").str.slice(0, 5)
    # Keep service-wide imports in their source lineage while evacuating invalid water cells.
    # Ordinary components all carry a blank service key and therefore retain the legacy
    # state/county/offense pool exactly.
    if "source_state_fips" not in components.columns:
        components["source_state_fips"] = components["state_fips"]
    components["source_state_fips"] = components["source_state_fips"].fillna(
        components["state_fips"]
    ).astype("string").str.zfill(2)
    for column in ("service_scope_id", "canonical_target_ori"):
        if column not in components.columns:
            components[column] = pd.NA
    service_scope = components["service_scope_id"].astype("string")
    service_wide = service_scope.notna() & service_scope.str.strip().ne("")
    local_group_cols = [
        "state_fips",
        "_routing_county",
        "offense",
        "source_state_fips",
        "service_scope_id",
        "canonical_target_ori",
    ]
    before = pd.to_numeric(components["component_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    denominator = pd.to_numeric(components["primary_denominator_raw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    land = pd.to_numeric(components["land_area_sq_mi"], errors="coerce").fillna(0.0).clip(lower=0.0)
    water_source = before.gt(0.0) & land.le(0.0)
    zero_denominator_source = before.gt(0.0) & denominator.le(0.0)
    # The broad zero-denominator redistribution remains behind its measured TVD
    # guardrail. Water-only Census cells are not a modeling choice: they are an
    # invalid geocode/fallback destination and are always evacuated to land in the
    # same jurisdiction/offense pool.
    zero_source = water_source | (zero_denominator_source & bool(APPLY_ZERO_TARGET_REDISTRIBUTION))
    eligible = denominator.gt(0.0) & land.gt(0.0)

    components["_zero_source_mass"] = np.where(zero_source, before, 0.0)
    components["_eligible_count_weight"] = np.where(eligible, before, 0.0)
    components["_eligible_denominator_weight"] = np.where(eligible, denominator, 0.0)
    local_zero_mass = components.groupby(local_group_cols, dropna=False)[
        "_zero_source_mass"
    ].transform("sum")
    eligible_count_weight_sum = components.groupby(local_group_cols, dropna=False)[
        "_eligible_count_weight"
    ].transform("sum")
    eligible_denominator_weight_sum = components.groupby(local_group_cols, dropna=False)[
        "_eligible_denominator_weight"
    ].transform("sum")
    use_count_weight = eligible_count_weight_sum.gt(0.0)
    use_denominator_weight = ~use_count_weight & eligible_denominator_weight_sum.gt(0.0)
    recipient_weight = np.where(
        eligible & use_count_weight,
        components["_eligible_count_weight"],
        np.where(eligible & use_denominator_weight, components["_eligible_denominator_weight"], 0.0),
    )
    recipient_weight_total = np.where(use_count_weight, eligible_count_weight_sum, eligible_denominator_weight_sum)
    local_can_redistribute = local_zero_mass.gt(0.0) & pd.Series(
        recipient_weight_total, index=components.index
    ).gt(0.0)
    local_delta = np.where(
        eligible & local_can_redistribute,
        local_zero_mass * recipient_weight / recipient_weight_total,
        0.0,
    )

    # A service footprint's county split is modeled rather than an observed incident location.
    # If a county has no publishable receiving cell, back off to the same documented service
    # footprint. Rows outside that service (including an independently served enclave) carry a
    # different/null scope key and are never candidates.
    service_group_cols = [
        "offense",
        "source_state_fips",
        "service_scope_id",
        "canonical_target_ori",
    ]
    fallback_source = zero_source & service_wide & ~local_can_redistribute
    components["_service_fallback_source_mass"] = np.where(fallback_source, before, 0.0)
    service_fallback_mass = components.groupby(service_group_cols, dropna=False)[
        "_service_fallback_source_mass"
    ].transform("sum")
    service_count_weight = np.where(eligible & service_wide, before, 0.0)
    service_denominator_weight = np.where(eligible & service_wide, denominator, 0.0)
    components["_service_count_weight"] = service_count_weight
    components["_service_denominator_weight"] = service_denominator_weight
    service_count_total = components.groupby(service_group_cols, dropna=False)[
        "_service_count_weight"
    ].transform("sum")
    service_denominator_total = components.groupby(service_group_cols, dropna=False)[
        "_service_denominator_weight"
    ].transform("sum")
    use_service_count = service_count_total.gt(0.0)
    service_recipient_weight = np.where(
        eligible & service_wide & use_service_count,
        service_count_weight,
        np.where(eligible & service_wide, service_denominator_weight, 0.0),
    )
    service_recipient_total = np.where(
        use_service_count, service_count_total, service_denominator_total
    )
    service_can_redistribute = service_fallback_mass.gt(0.0) & pd.Series(
        service_recipient_total, index=components.index
    ).gt(0.0)
    service_delta = np.where(
        eligible & service_wide & service_can_redistribute,
        service_fallback_mass * service_recipient_weight / service_recipient_total,
        0.0,
    )
    redistributed_delta = local_delta + service_delta
    source_can_redistribute = local_can_redistribute | (
        fallback_source & service_can_redistribute
    )
    terminal_service_source = fallback_source & ~service_can_redistribute
    stranded_water = water_source & ~source_can_redistribute & ~terminal_service_source
    if bool(stranded_water.any()):
        sample = components.loc[
            stranded_water, ["state_fips", "bg_id", "jurisdiction_id", "offense", "component_count"]
        ].head(20)
        raise ValueError(
            f"{int(stranded_water.sum())} positive water-only allocation(s) have no land recipient: "
            + str(sample.to_dict(orient="records"))
        )
    after = before + redistributed_delta
    after = pd.Series(after, index=components.index, dtype=float)
    after.loc[zero_source & source_can_redistribute] = 0.0
    after.loc[terminal_service_source] = 0.0
    components["component_count_before"] = before
    components["component_count_after"] = after
    components["redistributed_delta"] = after - before
    components["component_count"] = after
    components["_recipient_delta_mass"] = np.where(components["redistributed_delta"].gt(0.0), components["redistributed_delta"], 0.0)
    components["_recipient_count"] = np.where(components["redistributed_delta"].gt(1e-12), 1, 0)
    components["_routing_pool_id"] = np.where(
        service_wide,
        "service|"
        + components["source_state_fips"].astype("string")
        + "|"
        + components["service_scope_id"].astype("string")
        + "|"
        + components["canonical_target_ori"].astype("string")
        + "|"
        + components["offense"].astype("string"),
        "county|"
        + components["state_fips"].astype("string")
        + "|"
        + components["_routing_county"].astype("string")
        + "|"
        + components["offense"].astype("string"),
    )
    audit_group_cols = ["_routing_pool_id"]
    components["group_zero_source_mass"] = components.groupby(
        audit_group_cols, dropna=False
    )["_zero_source_mass"].transform("sum")
    components["group_recipient_count"] = components.groupby(audit_group_cols, dropna=False)[
        "_recipient_count"
    ].transform("sum")
    components["group_recipient_delta_mass"] = components.groupby(
        audit_group_cols, dropna=False
    )["_recipient_delta_mass"].transform("sum")
    components["group_total_before"] = components.groupby(audit_group_cols, dropna=False)[
        "component_count_before"
    ].transform("sum")
    components["group_total_after"] = components.groupby(audit_group_cols, dropna=False)[
        "component_count_after"
    ].transform("sum")
    components["group_total_delta"] = components["group_total_after"] - components["group_total_before"]
    components["audit_reason"] = np.select(
        [water_source, zero_source, components["redistributed_delta"].gt(0.0)],
        [
            "positive_allocation_water_only_cell",
            "positive_allocation_zero_offense_relevant_denominator",
            "zero_target_redistribution_recipient",
        ],
        default="unchanged",
    )
    components["redistribution_status"] = np.select(
        [
            water_source & source_can_redistribute,
            zero_source & source_can_redistribute,
            zero_denominator_source & ~water_source & (not bool(APPLY_ZERO_TARGET_REDISTRIBUTION)),
            zero_source & ~source_can_redistribute,
            components["redistributed_delta"].gt(0.0),
        ],
        [
            "source_redistributed_water_only",
            "source_redistributed",
            "source_not_redistributed_tvd_guardrail",
            "source_unlocated_no_eligible_service_target",
            "recipient",
        ],
        default="unchanged",
    )
    component_audit = components[[col for col in columns if col in components.columns]].copy()
    audit = components[water_source | zero_denominator_source].copy()
    component_output_columns = [
        "state_fips",
        "source_state_fips",
        "bg_id",
        "tract_id",
        "jurisdiction_id",
        "jurisdiction_type",
        "offense",
        "component_count",
        "service_scope_id",
        "canonical_target_ori",
        "source_target_count",
    ]
    out_components = components[
        [column for column in component_output_columns if column in components.columns]
    ].copy()
    service_unlocated = (
        components.loc[terminal_service_source]
        .groupby(
            [
                "source_state_fips",
                "service_scope_id",
                "canonical_target_ori",
                "offense",
                "source_target_count",
            ],
            dropna=False,
            as_index=False,
        )["component_count_before"]
        .sum()
        .rename(columns={"component_count_before": "unlocated_count"})
    )
    if not service_unlocated.empty:
        service_unlocated["route_reason"] = OVERLAP_ROUTE_SERVICE_NO_ELIGIBLE_RECEIVER
    out_components.attrs["service_unlocated_mass"] = service_unlocated
    audit = audit[[col for col in columns if col in audit.columns]].sort_values(
        ["offense", "state_fips", "jurisdiction_id", "bg_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    component_audit = component_audit.sort_values(
        ["offense", "state_fips", "jurisdiction_id", "bg_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out_components, audit, component_audit


def _build_publishability_audit(surface: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offense in OFFENSES_7:
        index_col = f"index_{offense}_primary"
        raw_rate_col = f"raw_rate_{offense}"
        suppressed_col = f"primary_index_suppressed_{offense}"
        low_denominator_col = f"diagnostic_eb_low_denominator_flag_{offense}"
        heavy_col = f"diagnostic_eb_heavy_shrinkage_flag_{offense}"
        extreme_col = f"diagnostic_eb_extreme_shrinkage_flag_{offense}"
        zero_col = f"primary_zero_denominator_positive_count_{offense}"
        denominator_col = f"primary_denominator_{offense}"
        count_col = _expected_count_col(offense)
        mode_col = f"estimate_mode_{offense}"
        suppressed = surface[suppressed_col].astype(bool) if suppressed_col in surface.columns else pd.Series(False, index=surface.index)
        non_residential_mode = (
            surface[mode_col].astype("string").eq("non_residential")
            if mode_col in surface.columns
            else suppressed
        )
        special_use_mode = (
            surface[mode_col].astype("string").eq("special_use")
            if mode_col in surface.columns
            else pd.Series(False, index=surface.index)
        )
        low_denominator = (
            surface[low_denominator_col].astype(bool)
            if low_denominator_col in surface.columns
            else pd.Series(False, index=surface.index)
        )
        heavy = surface[heavy_col].astype(bool) if heavy_col in surface.columns else pd.Series(False, index=surface.index)
        extreme = surface[extreme_col].astype(bool) if extreme_col in surface.columns else pd.Series(False, index=surface.index)
        zero_positive = surface[zero_col].astype(bool) if zero_col in surface.columns else pd.Series(False, index=surface.index)
        denominator = pd.to_numeric(surface.get(denominator_col), errors="coerce").fillna(0.0)
        counts = pd.to_numeric(surface.get(count_col), errors="coerce").fillna(0.0)
        households = pd.to_numeric(surface.get("households_total"), errors="coerce").fillna(0.0)
        indexes = pd.to_numeric(surface.get(index_col), errors="coerce")
        raw_rate = pd.to_numeric(surface.get(raw_rate_col), errors="coerce")
        hard_min = float(surface["eb_hard_min_denominator"].dropna().iloc[0]) if "eb_hard_min_denominator" in surface.columns else float(EB_HARD_MIN_DENOMINATOR)
        rows.append(
            {
                "offense": offense,
                "primary_denominator_type": PRIMARY_DENOMINATOR_BY_OFFENSE[offense],
                "hard_min_denominator": hard_min,
                "diagnostic_eb_k": float(pd.to_numeric(surface.get(f"diagnostic_eb_k_{offense}"), errors="coerce").dropna().iloc[0]),
                "suppressed_bg_count": int(suppressed.sum()),
                "non_residential_mode_bg_count": int(non_residential_mode.sum()),
                "special_use_mode_bg_count": int(special_use_mode.sum()),
                "suppressed_households_min": float(households[suppressed].min()) if bool(suppressed.any()) else float("nan"),
                "suppressed_households_p50": float(households[suppressed].quantile(0.50)) if bool(suppressed.any()) else float("nan"),
                "suppressed_households_p95": float(households[suppressed].quantile(0.95)) if bool(suppressed.any()) else float("nan"),
                "suppressed_households_max": float(households[suppressed].max()) if bool(suppressed.any()) else float("nan"),
                "suppressed_households_ge_50_count": int((suppressed & households.ge(50.0)).sum()),
                "low_denominator_bg_count": int(low_denominator.sum()),
                "heavy_shrinkage_bg_count": int(heavy.sum()),
                "extreme_shrinkage_bg_count": int(extreme.sum()),
                "published_bg_count": int(indexes.notna().sum()),
                "zero_denominator_positive_count_bg_count": int(zero_positive.sum()),
                "zero_denominator_positive_count_mass": float(counts[zero_positive].sum()),
                "suppressed_count_mass": float(counts[suppressed].sum()),
                "hard_min_denominator_row_count": int(denominator.le(hard_min).sum()),
                "max_raw_rate_per_100k": float(raw_rate.max()) if raw_rate.notna().any() else float("nan"),
                "max_published_primary_index": float(indexes.max()) if indexes.notna().any() else float("nan"),
                "primary_index_gt_5000_count": int(indexes.gt(5000.0).sum()),
                "primary_index_gt_10000_count": int(indexes.gt(10000.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _burglary_commercial_gradient_diagnostics(surface: pd.DataFrame) -> dict[str, object]:
    required = {
        "households_total",
        "commercial_premises_total",
        _expected_count_col("burglary"),
        "index_burglary_primary",
        "primary_index_publishable_burglary",
    }
    if not required.issubset(surface.columns):
        return {"ok": False, "reason": "missing_required_columns"}

    households = pd.to_numeric(surface["households_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
    commercial = pd.to_numeric(surface["commercial_premises_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
    count = pd.to_numeric(surface[_expected_count_col("burglary")], errors="coerce").fillna(0.0).clip(lower=0.0)
    after_index = pd.to_numeric(surface["index_burglary_primary"], errors="coerce")
    publishable = surface["primary_index_publishable_burglary"].fillna(False).astype(bool)
    unweighted_denominator = households + commercial
    eligible = publishable & unweighted_denominator.gt(0.0)
    if int(eligible.sum()) < 5:
        return {"ok": False, "reason": "too_few_publishable_rows", "rows": int(eligible.sum())}

    denominator_sum = float(unweighted_denominator.loc[eligible].sum())
    count_sum = float(count.loc[eligible].sum())
    national_rate = RATE_PER_100K * count_sum / denominator_sum if denominator_sum > 0.0 else float("nan")
    before_index = pd.Series(np.nan, index=surface.index, dtype=float)
    if np.isfinite(national_rate) and national_rate > 0.0:
        before_rate = RATE_PER_100K * count.loc[eligible] / unweighted_denominator.loc[eligible]
        before_index.loc[eligible] = 100.0 * before_rate / national_rate

    commercial_share = commercial / unweighted_denominator.replace(0.0, np.nan)
    quintile = pd.Series(pd.NA, index=surface.index, dtype="Int64")
    quintile.loc[eligible] = pd.qcut(
        commercial_share.loc[eligible].rank(method="first"),
        5,
        labels=False,
    ).astype("Int64")
    rows: list[dict[str, object]] = []
    for q in range(5):
        mask = eligible & quintile.eq(q)
        rows.append(
            {
                "quintile": int(q + 1),
                "rows": int(mask.sum()),
                "commercial_share_mean": float(commercial_share.loc[mask].mean()) if bool(mask.any()) else float("nan"),
                "before_index_mean": float(before_index.loc[mask].mean()) if bool(mask.any()) else float("nan"),
                "after_index_mean": float(after_index.loc[mask].mean()) if bool(mask.any()) else float("nan"),
                "before_index_median": float(before_index.loc[mask].median()) if bool(mask.any()) else float("nan"),
                "after_index_median": float(after_index.loc[mask].median()) if bool(mask.any()) else float("nan"),
            }
        )

    regime_quintiles: dict[str, list[dict[str, object]]] = {}
    regime_ratios: dict[str, float] = {}
    if "source_mode_burglary" in surface.columns:
        source_mode = surface["source_mode_burglary"].astype("string")
        # The gate has two regimes: direct-city rows stay direct; mixed/other rows
        # are evaluated with modeled transfer because their count mass is not
        # direct-dominant enough to qualify for the direct-city band.
        regime_masks = {
            "direct_city_incident": source_mode.eq("direct_city_incident"),
            "modeled_transfer": source_mode.ne("direct_city_incident"),
        }
        for regime, regime_mask in regime_masks.items():
            regime_rows: list[dict[str, object]] = []
            for q in range(5):
                mask = eligible & regime_mask & quintile.eq(q)
                regime_rows.append(
                    {
                        "quintile": int(q + 1),
                        "rows": int(mask.sum()),
                        "commercial_share_mean": (
                            float(commercial_share.loc[mask].mean()) if bool(mask.any()) else float("nan")
                        ),
                        "after_index_mean": float(after_index.loc[mask].mean()) if bool(mask.any()) else float("nan"),
                        "after_index_median": (
                            float(after_index.loc[mask].median()) if bool(mask.any()) else float("nan")
                        ),
                    }
                )
            q1 = regime_rows[0]["after_index_mean"]
            q5 = regime_rows[-1]["after_index_mean"]
            regime_ratios[regime] = (
                float(q5) / float(q1)
                if np.isfinite(float(q5)) and np.isfinite(float(q1)) and float(q1) != 0.0
                else float("nan")
            )
            regime_quintiles[regime] = regime_rows

    before_q1 = rows[0]["before_index_mean"]
    before_q5 = rows[-1]["before_index_mean"]
    after_q1 = rows[0]["after_index_mean"]
    after_q5 = rows[-1]["after_index_mean"]
    out = {
        "ok": True,
        "rows": int(eligible.sum()),
        "quintile_basis": "commercial_premises_total / (households_total + commercial_premises_total) over published burglary rows",
        "before_denominator": "households_total + commercial_premises_total",
        "after_denominator": "primary_denominator_burglary",
        "before_q5_q1_mean": (
            float(before_q5) / float(before_q1)
            if np.isfinite(float(before_q5)) and np.isfinite(float(before_q1)) and float(before_q1) != 0.0
            else float("nan")
        ),
        "after_q5_q1_mean": (
            float(after_q5) / float(after_q1)
            if np.isfinite(float(after_q5)) and np.isfinite(float(after_q1)) and float(after_q1) != 0.0
            else float("nan")
        ),
        "quintiles": rows,
    }
    if regime_ratios:
        out.update(
            {
                "after_q5_q1_mean_direct": regime_ratios.get("direct_city_incident", float("nan")),
                "after_q5_q1_mean_modeled": regime_ratios.get("modeled_transfer", float("nan")),
                "regime_quintiles": regime_quintiles,
                "regime_policy": "direct_city_incident rows are direct; mixed and other rows are evaluated as modeled_transfer",
            }
        )
    return out


def _suppression_mode_summary(surface: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for offense in OFFENSES_7:
        mode_col = f"estimate_mode_{offense}"
        suppressed_col = f"primary_index_suppressed_{offense}"
        if mode_col not in surface.columns:
            continue
        mode = surface[mode_col].astype("string")
        suppressed = (
            surface[suppressed_col].fillna(False).astype(bool)
            if suppressed_col in surface.columns
            else mode.ne("count_derived")
        )
        out[offense] = {
            "suppressed_cells": int(suppressed.sum()),
            "non_residential": int(mode.eq("non_residential").sum()),
            "insufficient_exposure": int(mode.eq("insufficient_exposure").sum()),
            "special_use": int(mode.eq("special_use").sum()),
            "vehicle_denominator_invalid": int(mode.eq("vehicle_denominator_invalid").sum()),
            "zero_primary_denominator": int(mode.eq("zero_primary_denominator").sum()),
            "count_derived": int(mode.eq("count_derived").sum()),
        }
    return out


def _offense_primary_normalizer(
    out: pd.DataFrame,
    *,
    offense: str,
    enable_exposure_ensemble: bool,
) -> tuple[pd.Series, str | None]:
    """The offense's published normalizer: legacy denominator, or the v2 opportunity ensemble.

    Legacy (the default) is the pre-change expression, unmoved: the offense's denominator-family
    column, coerced, null-filled and floored at zero. It returns no normalizer id, so a
    legacy-lane artifact carries exactly the columns it carried before this lane existed.

    With the ensemble on, the five person-exposure offenses read the named normalizer built by
    `exposure_ensemble.py` (`person_ens_v1`, and `larceny_opp_v1` for larceny). Burglary keeps its
    premises denominator and motor vehicle theft its vehicle denominator -- E3's verdicts, not an
    omission. Coverage fails closed: a missing normalizer column with the flag on would quietly
    fall back to a zero denominator and suppress every rate in the country.
    """
    denominator_type = PRIMARY_DENOMINATOR_BY_OFFENSE[offense]
    legacy = pd.to_numeric(
        out[DENOMINATOR_SOURCE_COLUMNS[denominator_type]],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    if not enable_exposure_ensemble:
        return legacy, None
    if offense not in ENSEMBLE_OFFENSES:
        return legacy, normalizer_id_for_offense(offense)
    column = opportunity_normalizer_column(offense)
    if column not in out.columns:
        raise ValueError(
            f"{column} is absent but the exposure ensemble is enabled; build it with "
            "`build-exposure-normalizers` before running the v2 output lane."
        )
    normalizer = pd.to_numeric(out[column], errors="coerce")
    if bool(normalizer.isna().any()):
        raise ValueError(
            f"{column} carries {int(normalizer.isna().sum())} null rows; the v2 exposure lane "
            "fails closed rather than publishing a zero denominator."
        )
    return normalizer.clip(lower=0.0), normalizer_id_for_offense(offense)


def _finalize_output(
    frame: pd.DataFrame,
    *,
    geo_id_col: str,
    population_col: str,
    config: AllocationBuildConfig,
    jurisdiction_col: str = "eb_jurisdiction_id",
    composites: CompositeRuntime | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    count_first_composites = bool(getattr(config, "enable_count_first_composites", False))
    if count_first_composites and composites is None:
        raise ValueError(
            "enable_count_first_composites is set but no severity-vector runtime was resolved; "
            "the composite lane fails closed rather than falling back to an unversioned vector."
        )
    eb_alpha = _eb_alpha_dict(config)
    eb_hard_min = float(config.eb_hard_min_denominator)
    pop = pd.to_numeric(out[population_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    daytime_proxy = (
        pd.to_numeric(out["daytime_population_jobs_proxy"], errors="coerce")
        if "daytime_population_jobs_proxy" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    daytime_proxy = daytime_proxy.fillna(pop).clip(lower=0.0)
    landscan_day = (
        pd.to_numeric(out[LANDSCAN_DAY_POP_COLUMN], errors="coerce")
        if LANDSCAN_DAY_POP_COLUMN in out.columns
        else pd.Series(0.0, index=out.index, dtype=float)
    ).fillna(0.0).clip(lower=0.0)
    landscan_positive = landscan_day.where(landscan_day.gt(0.0), 0.0)
    fallback_exposure_proxy = pd.Series(
        np.maximum(daytime_proxy.to_numpy(dtype=float), landscan_positive.to_numpy(dtype=float)),
        index=out.index,
        dtype=float,
    )
    exposure_proxy = (
        pd.to_numeric(out["exposure_proxy_2024"], errors="coerce")
        if "exposure_proxy_2024" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    exposure_proxy = (
        exposure_proxy.where(exposure_proxy.notna(), fallback_exposure_proxy)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    hq_capped_input = (
        out["person_exposure_hq_jobs_capped"].fillna(False).astype(bool)
        if "person_exposure_hq_jobs_capped" in out.columns
        else pd.Series(False, index=out.index)
    )
    exposure_no_shrink = pd.Series(
        np.maximum(exposure_proxy.to_numpy(dtype=float), daytime_proxy.to_numpy(dtype=float)),
        index=out.index,
        dtype=float,
    )
    exposure_proxy = exposure_proxy.where(hq_capped_input, exposure_no_shrink).clip(lower=0.0)
    landscan_lifted = (
        out["landscan_day_lifted_person_exposure"].fillna(False).astype(bool)
        if "landscan_day_lifted_person_exposure" in out.columns
        else landscan_positive.gt(daytime_proxy)
    )
    households = pd.to_numeric(out.get("households_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
    commercial_premises = pd.to_numeric(out.get("commercial_premises_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
    destination_poi = (
        pd.to_numeric(out["destination_poi_total"], errors="coerce")
        if "destination_poi_total" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    destination_poi = destination_poi.fillna(commercial_premises).clip(lower=0.0)
    lodes_manufacturing_jobs = pd.to_numeric(
        out["lodes_manufacturing_jobs"] if "lodes_manufacturing_jobs" in out.columns else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    lodes_wholesale_jobs = pd.to_numeric(
        out["lodes_wholesale_jobs"] if "lodes_wholesale_jobs" in out.columns else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    lodes_retail_jobs = pd.to_numeric(
        out["lodes_retail_jobs"] if "lodes_retail_jobs" in out.columns else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    lodes_transport_warehouse_jobs = pd.to_numeric(
        out["lodes_transport_warehouse_jobs"]
        if "lodes_transport_warehouse_jobs" in out.columns
        else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    lodes_industrial_jobs = pd.to_numeric(
        out["lodes_industrial_jobs"] if "lodes_industrial_jobs" in out.columns else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    lodes_industrial_jobs = lodes_industrial_jobs.fillna(
        lodes_manufacturing_jobs + lodes_wholesale_jobs + lodes_transport_warehouse_jobs
    ).clip(lower=0.0)
    destination_weight = (
        pd.to_numeric(out["burglary_destination_poi_exposure_weight"], errors="coerce")
        if "burglary_destination_poi_exposure_weight" in out.columns
        else (
            pd.to_numeric(out["burglary_commercial_exposure_weight"], errors="coerce")
            if "burglary_commercial_exposure_weight" in out.columns
            else pd.Series(np.nan, index=out.index, dtype=float)
        )
    )
    retail_weight = (
        pd.to_numeric(out["burglary_retail_jobs_exposure_weight"], errors="coerce")
        if "burglary_retail_jobs_exposure_weight" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    industrial_weight = (
        pd.to_numeric(out["burglary_industrial_jobs_exposure_weight"], errors="coerce")
        if "burglary_industrial_jobs_exposure_weight" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    non_null_destination_weights = destination_weight.dropna()
    calibrated_destination_weight_default = (
        float(non_null_destination_weights.iloc[0])
        if not non_null_destination_weights.empty
        else (
            float(config.burglary_commercial_weight)
            if config.burglary_commercial_weight is not None
            else float(BURGLARY_COMMERCIAL_WEIGHT_FALLBACK)
        )
    )
    destination_weight = destination_weight.fillna(calibrated_destination_weight_default).clip(lower=0.0)
    retail_weight = retail_weight.fillna(0.0).clip(lower=0.0)
    industrial_weight = industrial_weight.fillna(0.0).clip(lower=0.0)
    burglary_formula = (
        households
        + destination_weight * destination_poi
        + retail_weight * lodes_retail_jobs
        + industrial_weight * lodes_industrial_jobs
    )
    burglary_premises = pd.to_numeric(out.get("burglary_premises_total"), errors="coerce").fillna(
        burglary_formula
    ).clip(lower=0.0)
    vehicles = pd.to_numeric(out.get("aggregate_vehicles_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
    mvt_commuter_vehicle_proxy = (
        pd.to_numeric(
            out["mvt_commuter_vehicle_proxy"]
            if "mvt_commuter_vehicle_proxy" in out.columns
            else pd.Series(0.0, index=out.index, dtype=float),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )
    vehicle_exposure = (
        pd.to_numeric(
            out["vehicle_exposure_2024"]
            if "vehicle_exposure_2024" in out.columns
            else pd.Series(np.nan, index=out.index, dtype=float),
            errors="coerce",
        )
        .fillna(vehicles + mvt_commuter_vehicle_proxy)
        .clip(lower=0.0)
    )
    land_area_sq_mi = (
        pd.to_numeric(out["land_area_sq_mi"], errors="coerce")
        if "land_area_sq_mi" in out.columns
        else pd.Series(0.0, index=out.index, dtype=float)
    ).fillna(0.0).clip(lower=0.0)
    tract_id = (
        out["tract_id"].astype("string").str.zfill(11)
        if "tract_id" in out.columns
        else out[geo_id_col].astype("string").str.zfill(11)
    )
    special_use_tract = tract_id.str.slice(5, 11).str.startswith(str(SPECIAL_USE_TRACT_PREFIX), na=False)
    out[population_col] = pop
    out["daytime_population_jobs_proxy"] = daytime_proxy
    out[LANDSCAN_DAY_POP_COLUMN] = landscan_day
    out["exposure_proxy_2024"] = exposure_proxy
    out["landscan_day_lifted_person_exposure"] = landscan_lifted
    for col, default in [
        ("person_exposure_before_hq_jobs_cap", exposure_proxy),
        ("person_exposure_hq_jobs_cap", pd.Series(np.nan, index=out.index, dtype=float)),
    ]:
        existing = (
            pd.to_numeric(out[col], errors="coerce")
            if col in out.columns
            else pd.Series(np.nan, index=out.index, dtype=float)
        )
        out[col] = existing.fillna(default)
    for col in ["person_exposure_hq_jobs_cap_candidate", "person_exposure_hq_jobs_capped"]:
        out[col] = out.get(col, pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["households_total"] = households
    out["commercial_premises_total"] = commercial_premises
    out["destination_poi_total"] = destination_poi
    out["lodes_manufacturing_jobs"] = lodes_manufacturing_jobs
    out["lodes_wholesale_jobs"] = lodes_wholesale_jobs
    out["lodes_retail_jobs"] = lodes_retail_jobs
    out["lodes_transport_warehouse_jobs"] = lodes_transport_warehouse_jobs
    out["lodes_industrial_jobs"] = lodes_industrial_jobs
    out["burglary_premises_total"] = burglary_premises
    out["burglary_commercial_exposure_weight"] = destination_weight
    out["burglary_destination_poi_exposure_weight"] = destination_weight
    out["burglary_retail_jobs_exposure_weight"] = retail_weight
    out["burglary_industrial_jobs_exposure_weight"] = industrial_weight
    out["aggregate_vehicles_total"] = vehicles
    out["mvt_commuter_vehicle_proxy"] = mvt_commuter_vehicle_proxy
    out["vehicle_exposure_2024"] = vehicle_exposure
    out["county_auto_commute_vehicle_share"] = (
        pd.to_numeric(
            out["county_auto_commute_vehicle_share"]
            if "county_auto_commute_vehicle_share" in out.columns
            else pd.Series(0.0, index=out.index, dtype=float),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    out["land_area_sq_mi"] = land_area_sq_mi
    out["eb_hard_min_denominator"] = eb_hard_min
    out["resident_secondary_denominator"] = pop
    out["resident_secondary_denominator_low_reliability"] = pop.le(0.0)
    residential_eligible = households.ge(float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR))
    non_residential = ~residential_eligible
    mvt_vehicle_denominator_invalid = pd.Series(False, index=out.index)
    out["non_residential_household_floor"] = float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
    out["person_exposure_denominator_floor"] = float(PERSON_EXPOSURE_DENOMINATOR_FLOOR)
    out["mvt_vehicle_exposure_denominator_floor"] = float(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR)
    out["non_residential_flag"] = non_residential
    out["special_use_tract_flag"] = special_use_tract

    # --- v2 typed special-use taxonomy -------------------------------------------------------
    # The 98-series code stays on the frame exactly as before: it is the Census fact, and this
    # lane's whole point is that the fact is a warning flag rather than a measurement. What
    # changes is who reads it. With the taxonomy on, the two publication gates below come from
    # the cell's TYPE; with it off they are the legacy `residential_eligible` mask, unchanged,
    # which is what keeps a legacy build byte-identical.
    special_use_taxonomy = bool(getattr(config, "enable_special_use_taxonomy", False))
    exposure_ensemble = bool(getattr(config, "enable_exposure_ensemble", False))
    if special_use_taxonomy:
        out = apply_special_use_taxonomy(
            out, population_col=population_col, special_use_tract_flag=special_use_tract
        )
    primary_arm_allowed = (
        special_use_primary_rate_allowed(out, residential_eligible=residential_eligible)
        if special_use_taxonomy
        else residential_eligible
    )
    resident_arm_allowed = (
        special_use_resident_rate_allowed(out, residential_eligible=residential_eligible)
        if special_use_taxonomy
        else residential_eligible
    )
    primary_typed_suppressed = special_use_typed_suppressed(out, allowed=primary_arm_allowed)
    resident_typed_suppressed = special_use_typed_suppressed(out, allowed=resident_arm_allowed)

    # What the per-offense denominators ARE, said on the frame rather than only in the docs: an
    # opportunity-normalized intensity is not person-time risk (REVIEW_SOL_NEUTRAL.md sec.3).
    # Absent entirely on a legacy build, which still divides by the hard max.
    if exposure_ensemble:
        out["opportunity_normalizer_semantics"] = NORMALIZER_SEMANTICS

    # --- the near-zero-resident opportunity floor (see the constant's block) ------------------
    # The class this closes exists only where a v2 lane opened it: the typed taxonomy opens the
    # opportunity arm by TYPE regardless of households, and the ensemble denominator no longer
    # inflates itself by taking a max. Either flag on its own is enough to reach the class, so the
    # gate is the OR of the two; with BOTH off the legacy build is byte-identical, which is what
    # keeps this expression out of the legacy path entirely.
    zero_resident_opportunity_lane = special_use_taxonomy or exposure_ensemble
    near_zero_resident = pop.lt(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
    if zero_resident_opportunity_lane:
        out[ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN] = float(
            ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR
        )

    total_counts = []
    personal_counts = np.zeros(len(out), dtype=float)
    property_counts = np.zeros(len(out), dtype=float)
    for offense in OFFENSES_7:
        count_col = _source_count_col(out, offense)
        expected_count_col = _expected_count_col(offense)
        count = pd.to_numeric(out.get(count_col), errors="coerce").fillna(0.0).clip(lower=0.0)
        out[expected_count_col] = count
        total_counts.append(count.to_numpy(dtype=float))
        if offense in PERSONAL_OFFENSES:
            personal_counts += count.to_numpy(dtype=float)
        else:
            property_counts += count.to_numpy(dtype=float)

        denominator_type = PRIMARY_DENOMINATOR_BY_OFFENSE[offense]
        raw_denominator, normalizer_id = _offense_primary_normalizer(
            out,
            offense=offense,
            enable_exposure_ensemble=exposure_ensemble,
        )
        out[f"primary_denominator_type_{offense}"] = denominator_type
        out[f"primary_denominator_{offense}"] = raw_denominator
        if normalizer_id is not None:
            out[normalizer_id_column(offense)] = normalizer_id
        primary_denominator_invalid = (
            mvt_vehicle_denominator_invalid
            if offense == "motor_vehicle_theft"
            else pd.Series(False, index=out.index)
        )
        burglary_premises_floor = (
            raw_denominator.lt(float(BURGLARY_PREMISES_DENOMINATOR_FLOOR))
            if offense == "burglary"
            else pd.Series(False, index=out.index)
        )
        burglary_premises_floor = (
            pd.Series(burglary_premises_floor, index=out.index).fillna(False).astype(bool)
        )
        if special_use_taxonomy:
            # Nothing is suppressed for carrying a 98-series code any more; the typed gates above
            # own that decision. The burglary premises floor is all that survives here, and it is
            # not a special-use fact at all -- it is an exposure floor, so it moves to the
            # insufficient-exposure term below and reports itself as one.
            special_use_suppressed = pd.Series(False, index=out.index)
        else:
            special_use_suppressed = (
                pd.Series(special_use_tract | burglary_premises_floor, index=out.index)
                .fillna(False)
                .astype(bool)
            )
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES:
            primary_insufficient_exposure = raw_denominator.lt(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
        elif offense == "motor_vehicle_theft":
            primary_insufficient_exposure = raw_denominator.lt(float(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR))
        else:
            primary_insufficient_exposure = pd.Series(False, index=out.index)
        primary_insufficient_exposure = (
            pd.Series(primary_insufficient_exposure, index=out.index).fillna(False).astype(bool)
        )
        if special_use_taxonomy:
            primary_insufficient_exposure = primary_insufficient_exposure | burglary_premises_floor
        if zero_resident_opportunity_lane:
            # An opportunity rate on a cell where essentially nobody lives needs real opportunity
            # mass behind it. Folded into the insufficient-exposure term rather than given a term
            # of its own, because it IS that term's mechanism at a support level the landed floor
            # was never asked to cover: same display code, same reason string, same treatment in
            # the aggregate composites (the count is real and enters at tract support), and no new
            # vocabulary for a viewer to learn. `ANY` primary opportunity-rate index means all
            # seven offenses, each against its own normalizer's units.
            primary_insufficient_exposure = primary_insufficient_exposure | (
                near_zero_resident
                & raw_denominator.lt(float(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR))
            )

        # The resident arm's own floor, hoisted above the primary block: the ambient-blind
        # footprint rule below is defined against the RESIDENT rate, so its baseline
        # publishable mask has to exist before either arm is finalised.
        resident_denominator = out["resident_secondary_denominator"]
        # The resident arm has one denominator and one support rule for every
        # offense: resident population. Premises and vehicle floors belong only
        # to their opportunity-normalized primary arms. Coupling them here made
        # a populated block group lose its burglary-per-resident value merely
        # because a premises feature was sparse, and did the same to MVT when its
        # vehicle proxy was invalid.
        resident_insufficient_exposure = resident_denominator.lt(
            float(PERSON_EXPOSURE_DENOMINATOR_FLOOR)
        )
        resident_insufficient_exposure = (
            pd.Series(resident_insufficient_exposure, index=out.index).fillna(False).astype(bool)
        )

        # --- ambient-blind custom footprint (see the constants block) ---------------------
        footprint_count = pd.to_numeric(
            out[_footprint_derived_count_col(offense)]
            if _footprint_derived_count_col(offense) in out.columns
            else pd.Series(0.0, index=out.index, dtype=float),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        footprint_share = pd.Series(
            np.where(count.to_numpy() > 0.0, footprint_count.to_numpy() / np.where(count > 0.0, count, 1.0), 0.0),
            index=out.index,
            dtype=float,
        ).clip(0.0, 1.0)
        # "No ambient lift": the published exposure denominator is resident population and
        # nothing else -- neither the LODES jobs proxy nor LandScan raised it.
        ambient_exposure_lift = exposure_proxy.gt(pop)
        baseline_resident_publishable = (
            resident_arm_allowed
            & resident_denominator.gt(0.0)
            & ~special_use_suppressed
            & ~resident_insufficient_exposure
        )
        baseline_resident = _count_derived_rate_index(
            counts=count,
            denominator=resident_denominator,
            publishable=baseline_resident_publishable,
        )
        baseline_national_resident_rate = float(baseline_resident["national_rate_per_100k"])
        # The rate the ratio is measured on uses the count's exact-Poisson LOWER bound, so a
        # fractional modelled count cannot make the claim on its own (see the constants block).
        count_lower, _count_upper = _poisson_count_interval(count)
        conservative_resident_rate = pd.Series(np.nan, index=out.index, dtype=float)
        rate_measurable = baseline_resident_publishable & resident_denominator.gt(0.0)
        conservative_resident_rate.loc[rate_measurable] = (
            RATE_PER_100K
            * count_lower.loc[rate_measurable]
            / resident_denominator.loc[rate_measurable]
        )
        resident_rate_ratio = pd.Series(np.nan, index=out.index, dtype=float)
        if np.isfinite(baseline_national_resident_rate) and baseline_national_resident_rate > 0:
            resident_rate_ratio = conservative_resident_rate / baseline_national_resident_rate
        # This diagnostic can invalidate only a person-exposure denominator. A
        # premises-normalized burglary rate and a vehicle-normalized MVT rate do
        # not depend on ambient person exposure, and a resident rate deliberately
        # answers the resident-denominator question even when daytime presence is
        # much larger. Applying this flag to all arms created valid-but-grey cells.
        footprint_ambient_exposure_missing = (
            footprint_share.gt(float(FOOTPRINT_DERIVED_MASS_SHARE_FLOOR))
            & ~ambient_exposure_lift
            & resident_rate_ratio.ge(float(AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO))
        ).fillna(False)
        if offense not in PERSON_EXPOSURE_FLOOR_OFFENSES:
            footprint_ambient_exposure_missing = pd.Series(False, index=out.index)
        if (
            bool(config.enable_exposure_ensemble)
            and config.exposure_residential_leg_source
            == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
            and offense in ENSEMBLE_OFFENSES
        ):
            # This gate diagnoses missing ambient population support. In the explicit Census
            # residential arm the selected residential component is resident population by
            # contract; raw LandScan non-coverage must remain diagnostic evidence, not invalidate
            # an otherwise valid selected denominator.
            footprint_ambient_exposure_missing = pd.Series(False, index=out.index)
        out[_footprint_derived_share_col(offense)] = footprint_share
        out[_footprint_ambient_exposure_missing_col(offense)] = footprint_ambient_exposure_missing

        primary_eb = _empirical_bayes_index(
            out,
            offense=offense,
            counts=count,
            denominator=raw_denominator,
            geo_id_col=geo_id_col,
            alpha=float(eb_alpha[offense]),
            hard_min=eb_hard_min,
            jurisdiction_col=jurisdiction_col,
        )
        primary_publishable = (
            primary_arm_allowed
            & raw_denominator.gt(0.0)
            & ~primary_denominator_invalid
            & ~special_use_suppressed
            & ~primary_insufficient_exposure
            & ~footprint_ambient_exposure_missing
        )
        primary_published = _count_derived_rate_index(
            counts=count,
            denominator=raw_denominator,
            publishable=primary_publishable,
        )
        _warn_if_raw_rate_mismatch(
            label=f"{geo_id_col}:{offense}:primary",
            raw_rate=pd.Series(primary_eb["raw_rate"], index=out.index),
            count_rate=pd.Series(primary_published["rate"], index=out.index),
        )
        primary_denominator_reason = pd.Series(primary_eb["denominator_reason"], index=out.index, dtype="string")
        primary_denominator_reason.loc[primary_denominator_invalid] = "vehicle_denominator_invalid"
        special_reason_mask = special_use_suppressed & ~non_residential & ~primary_denominator_invalid
        primary_denominator_reason.loc[special_reason_mask] = "special_use"
        # "The arm was open and a floor closed it" is the condition these two masks are trying to
        # express; off the typed lane the household rule IS the arm, so the two spellings agree.
        # On it they must not: a campus that publishes by type but sits below the exposure floor
        # reports the floor, not the household count it was never gated on.
        primary_arm_open = primary_arm_allowed if special_use_taxonomy else ~non_residential
        primary_insufficient_reason_mask = (
            primary_insufficient_exposure
            & primary_arm_open
            & ~primary_denominator_invalid
            & ~special_use_suppressed
        )
        primary_denominator_reason.loc[primary_insufficient_reason_mask] = "insufficient_exposure"
        # A cell that is BOTH below the plain exposure floor and ambient-blind reports the
        # floor: the floor is the harder structural fact and does not need the footprint.
        footprint_reason_mask = (
            footprint_ambient_exposure_missing
            & primary_arm_open
            & ~primary_denominator_invalid
            & ~special_use_suppressed
            & ~primary_insufficient_exposure
        )
        primary_denominator_reason.loc[footprint_reason_mask] = INSUFFICIENT_AMBIENT_EXPOSURE_REASON
        # `denominator_reason` had no `non_residential` value at all (Stage 5 F7): 2,306 BG rows
        # per offense carried "publishable" next to a null index, because the households floor
        # nulls the index without touching the reason. Assigned LAST so the precedence matches
        # `_estimate_mode`, where `non_residential` also wins.
        #
        # Two changes once the taxonomy is on, and only then. The household floor may no longer
        # overwrite the reason on a cell that PUBLISHES -- a campus or an employment district
        # publishes an exposure rate with fewer than ten households, and "non_residential" beside
        # a live rate would be a contradiction. And a cell the TYPE closed reports `special_use`
        # last of all, because the type is the more fundamental fact than any floor it also trips:
        # `special_use_type` beside it names which type.
        if not special_use_taxonomy:
            primary_denominator_reason.loc[non_residential] = "non_residential"
        if special_use_taxonomy:
            primary_denominator_reason.loc[primary_typed_suppressed] = "special_use"
        out[f"primary_denominator_raw_{offense}"] = primary_eb["denominator_raw"]
        out[f"primary_national_rate_per_100k_{offense}"] = primary_published["national_rate_per_100k"]
        out[f"primary_alpha_{offense}"] = float(eb_alpha[offense])
        out[f"raw_rate_{offense}"] = primary_published["rate"]
        out[f"diagnostic_eb_rate_{offense}"] = primary_eb["diagnostic_eb_rate"]
        out[f"diagnostic_eb_national_rate_per_100k_{offense}"] = primary_eb["diagnostic_eb_national_rate_per_100k"]
        out[f"diagnostic_eb_prior_rate_{offense}"] = primary_eb["diagnostic_eb_prior_rate"]
        out[f"diagnostic_eb_k_{offense}"] = primary_eb["diagnostic_eb_k"]
        out[f"diagnostic_eb_observed_weight_{offense}"] = primary_eb["diagnostic_eb_observed_weight"]
        out[f"diagnostic_eb_prior_weight_{offense}"] = primary_eb["diagnostic_eb_prior_weight"]
        out[f"index_publishable_{offense}"] = primary_publishable
        out[f"diagnostic_eb_low_denominator_flag_{offense}"] = primary_eb["diagnostic_eb_low_denominator_flag"]
        out[f"diagnostic_eb_heavy_shrinkage_flag_{offense}"] = primary_eb["diagnostic_eb_heavy_shrinkage_flag"]
        out[f"diagnostic_eb_extreme_shrinkage_flag_{offense}"] = primary_eb["diagnostic_eb_extreme_shrinkage_flag"]
        out[f"denominator_reason_{offense}"] = primary_denominator_reason
        out[f"primary_index_publishable_{offense}"] = primary_publishable
        out[f"primary_index_suppressed_{offense}"] = (
            (~primary_arm_allowed if special_use_taxonomy else non_residential)
            | primary_denominator_invalid
            | special_use_suppressed
            | primary_insufficient_exposure
            | footprint_ambient_exposure_missing
        )
        out[f"primary_zero_denominator_positive_count_{offense}"] = raw_denominator.le(0.0) & count.gt(0.0)
        if offense == "motor_vehicle_theft":
            out[f"primary_denominator_invalid_{offense}"] = primary_denominator_invalid
        estimate_mode = _estimate_mode(
            non_residential=(
                pd.Series(False, index=out.index) if special_use_taxonomy else non_residential
            ),
            publishable=primary_publishable,
            denominator=raw_denominator,
            denominator_invalid=primary_denominator_invalid,
            denominator_invalid_mode="vehicle_denominator_invalid",
        )
        estimate_mode.loc[special_reason_mask] = "special_use"
        estimate_mode.loc[primary_insufficient_reason_mask] = "insufficient_exposure"
        # The ambient-blind footprint class shares the DISPLAY vocabulary of the plain
        # exposure floor -- "too little exposure for a per-person rate" is exactly true of a
        # casino parcel measured by its residents -- so no new `estimate_mode` value is minted
        # and the viewer's five codes still cover the surface. `denominator_reason` above is
        # where the mechanism is named. See docs/PIPELINE.md on the two vocabularies.
        estimate_mode.loc[footprint_reason_mask] = "insufficient_exposure"
        # No new display code is minted for the typed lane either: a type that closes the rate
        # reports `special_use`, which is the viewer's existing special-use hatch, and the type
        # itself rides in `special_use_type` beside it. Assigned last, same precedence as the
        # reason above.
        if special_use_taxonomy:
            estimate_mode.loc[primary_typed_suppressed] = "special_use"
        out[f"estimate_mode_{offense}"] = estimate_mode
        out[f"rate_{offense}_primary"] = primary_published["rate"]
        out[f"index_{offense}_primary"] = primary_published["index"]

        direct_flag_col = f"direct_incident_support_flag_{offense}"
        direct_count_col = f"direct_incident_support_count_{offense}"
        direct_years_col = f"direct_incident_support_years_{offense}"
        direct_year_min_col = f"direct_incident_support_year_min_{offense}"
        direct_year_max_col = f"direct_incident_support_year_max_{offense}"
        support_source_col = f"numerator_support_source_{offense}"
        direct_flag = _support_flag_series(out, direct_flag_col)
        direct_count = _support_series(out, direct_count_col)
        direct_years = _support_series(out, direct_years_col)
        direct_year_min = (
            pd.to_numeric(out[direct_year_min_col], errors="coerce")
            if direct_year_min_col in out.columns
            else pd.Series(np.nan, index=out.index, dtype=float)
        )
        direct_year_max = (
            pd.to_numeric(out[direct_year_max_col], errors="coerce")
            if direct_year_max_col in out.columns
            else pd.Series(np.nan, index=out.index, dtype=float)
        )
        effective_support = direct_count.where(direct_flag, 0.0)
        interval = _rate_index_interval(
            counts=count,
            denominator=raw_denominator,
            publishable=primary_publishable,
            national_rate_per_100k=float(primary_published["national_rate_per_100k"]),
        )
        index_width = (interval["index_upper"] - interval["index_lower"]).replace([np.inf, -np.inf], np.nan)
        point_index = pd.to_numeric(out[f"index_{offense}_primary"], errors="coerce")
        index_width_ratio = pd.Series(np.nan, index=out.index, dtype=float)
        positive_point = point_index.gt(0.0) & point_index.notna()
        index_width_ratio.loc[positive_point] = index_width.loc[positive_point] / point_index.loc[positive_point]
        reliability_tier = _reliability_tier(
            publishable=primary_publishable,
            effective_support=effective_support,
            direct_support_years=direct_years,
            index_width_ratio=index_width_ratio,
        )
        out[direct_flag_col] = direct_flag
        out[direct_count_col] = direct_count
        out[direct_years_col] = direct_years
        out[direct_year_min_col] = direct_year_min
        out[direct_year_max_col] = direct_year_max
        out[f"effective_numerator_support_{offense}"] = effective_support
        out[support_source_col] = np.where(direct_flag, "direct_city_incident", "model_only")
        out[f"rate_{offense}_primary_ci95_lower"] = interval["rate_lower"]
        out[f"rate_{offense}_primary_ci95_upper"] = interval["rate_upper"]
        out[f"index_{offense}_primary_ci95_lower"] = interval["index_lower"]
        out[f"index_{offense}_primary_ci95_upper"] = interval["index_upper"]
        out[f"index_{offense}_primary_ci95_width"] = index_width
        out[f"index_{offense}_primary_ci95_width_ratio"] = index_width_ratio.replace([np.inf, -np.inf], np.nan)
        out[f"reliability_tier_{offense}"] = reliability_tier
        recommended_geography = _recommended_display_geography(
            tier=reliability_tier,
            publishable=primary_publishable,
            geo_id_col=geo_id_col,
        )
        # Murder and rape have fixed tract publication support. Reliability
        # describes uncertainty in that estimate; it does not move the displayed
        # value to an unrelated county or jurisdiction surface.
        if offense in RARE_OFFENSE_TRACT_SUPPORT and geo_id_col != "block_group_geoid":
            recommended_geography.loc[primary_publishable] = "tract"
        if special_use_taxonomy:
            # The consult's "count/density and coarser recommendation". A typed cell's COUNT is
            # real and conserved, so the rate it cannot carry exists one level up; "not_published"
            # would tell a reader there is nothing to see, which is false.
            recommended_geography.loc[
                coarser_recommendation_rows(out, publishable=primary_publishable)
            ] = ("tract_or_larger" if geo_id_col == "block_group_geoid" else "jurisdiction_or_county")
        out[f"recommended_display_geography_{offense}"] = recommended_geography

        resident_eb = _empirical_bayes_index(
            out,
            offense=offense,
            counts=count,
            denominator=out["resident_secondary_denominator"],
            geo_id_col=geo_id_col,
            alpha=float(eb_alpha[offense]),
            hard_min=eb_hard_min,
            jurisdiction_col=jurisdiction_col,
        )
        resident_denominator_invalid = pd.Series(False, index=out.index)
        # `resident_denominator` and `resident_insufficient_exposure` are computed above the
        # primary block, because the ambient-blind footprint rule is defined on the resident
        # rate and needs them first.
        resident_publishable = baseline_resident_publishable & ~resident_denominator_invalid
        resident_published = _count_derived_rate_index(
            counts=count,
            denominator=resident_denominator,
            publishable=resident_publishable,
        )
        _warn_if_raw_rate_mismatch(
            label=f"{geo_id_col}:{offense}:resident",
            raw_rate=pd.Series(resident_eb["raw_rate"], index=out.index),
            count_rate=pd.Series(resident_published["rate"], index=out.index),
        )
        resident_denominator_reason = pd.Series(resident_eb["denominator_reason"], index=out.index, dtype="string")
        resident_denominator_reason.loc[resident_denominator_invalid] = "vehicle_denominator_invalid"
        resident_denominator_reason.loc[special_reason_mask] = "special_use"
        resident_insufficient_reason_mask = (
            resident_insufficient_exposure
            & (resident_arm_allowed if special_use_taxonomy else ~non_residential)
            & ~resident_denominator_invalid
            & ~special_use_suppressed
        )
        resident_denominator_reason.loc[resident_insufficient_reason_mask] = "insufficient_exposure"
        # Same Stage 5 F7 hole on the resident arm (771 BG rows), and the same two typed-lane
        # changes as the primary arm above.
        if not special_use_taxonomy:
            resident_denominator_reason.loc[non_residential] = "non_residential"
        if special_use_taxonomy:
            resident_denominator_reason.loc[resident_typed_suppressed] = "special_use"
        out[f"resident_raw_rate_{offense}"] = resident_published["rate"]
        out[f"diagnostic_resident_eb_rate_{offense}"] = resident_eb["diagnostic_eb_rate"]
        out[f"diagnostic_resident_eb_national_rate_per_100k_{offense}"] = resident_eb["diagnostic_eb_national_rate_per_100k"]
        out[f"diagnostic_resident_eb_prior_rate_{offense}"] = resident_eb["diagnostic_eb_prior_rate"]
        out[f"diagnostic_resident_eb_k_{offense}"] = resident_eb["diagnostic_eb_k"]
        out[f"diagnostic_resident_eb_observed_weight_{offense}"] = resident_eb["diagnostic_eb_observed_weight"]
        out[f"diagnostic_resident_eb_prior_weight_{offense}"] = resident_eb["diagnostic_eb_prior_weight"]
        out[f"index_{offense}_resident_publishable"] = resident_publishable
        out[f"diagnostic_resident_eb_low_denominator_flag_{offense}"] = resident_eb["diagnostic_eb_low_denominator_flag"]
        out[f"diagnostic_resident_eb_heavy_shrinkage_flag_{offense}"] = resident_eb["diagnostic_eb_heavy_shrinkage_flag"]
        out[f"diagnostic_resident_eb_extreme_shrinkage_flag_{offense}"] = resident_eb["diagnostic_eb_extreme_shrinkage_flag"]
        out[f"resident_denominator_reason_{offense}"] = resident_denominator_reason
        out[f"resident_national_rate_per_100k_{offense}"] = resident_published["national_rate_per_100k"]
        if offense == "motor_vehicle_theft":
            out[f"resident_denominator_invalid_{offense}"] = resident_denominator_invalid
        out[f"index_{offense}_resident_suppressed"] = (
            (~resident_arm_allowed if special_use_taxonomy else non_residential)
            | resident_denominator_invalid
            | special_use_suppressed
            | resident_insufficient_exposure
        )
        out[f"rate_{offense}_resident"] = resident_published["rate"]
        out[f"index_{offense}_resident"] = resident_published["index"]

        transient_ratio_values = np.full(len(out), np.nan, dtype=float)
        np.divide(
            exposure_proxy.to_numpy(dtype=float),
            pop.replace(0.0, np.nan).to_numpy(dtype=float),
            out=transient_ratio_values,
            where=pop.gt(0.0).to_numpy(dtype=bool),
        )
        transient_ratio = pd.Series(transient_ratio_values, index=out.index, dtype=float)
        # One column, not seven: the ratio is exposure over residents and does not vary by
        # offense. Only the index condition below does.
        out["transient_exposure_daytime_to_resident_ratio"] = transient_ratio
        # The predicate is measured on the published points and remains advisory. It does not
        # change publication mode, denominator reason, or any point value.
        #
        # 1. The `households_total >= 10` term was dead. Every one of the 290 candidate cells it
        #    excluded is non-residential, and a non-residential cell publishes no index at all,
        #    so the index condition was already False there: measured, 0 cells were blocked by
        #    the household floor alone. It is gone rather than left as a term that reads
        #    load-bearing and is not.
        #
        # 2. The threshold was applied to the wrong estimator. The condition
        #    `exposure / population >= 5` says the RESIDENT denominator understates the
        #    population at risk -- but murder, rape, robbery, aggravated assault and larceny
        #    already publish against person EXPOSURE, so the primary index has absorbed the
        #    transience the ratio is complaining about. The estimator the ratio indicts is
        #    `index_{offense}_resident`, and that is what the flag now reads. Measured on the
        #    promoted surface this moves the reachable population from 82 block groups to
        #    ~2,087 -- the guard now fires on its own definition instead of on a surface that
        #    has already been corrected.
        out[f"transient_exposure_likely_{offense}"] = (
            pop.gt(0.0)
            & transient_ratio.ge(float(TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO))
            & pd.to_numeric(out[f"index_{offense}_resident"], errors="coerce").ge(
                float(TRANSIENT_EXPOSURE_INDEX_THRESHOLD)
            )
        ) | (
            pop.gt(0.0)
            & pd.to_numeric(out["commercial_premises_total"], errors="coerce").fillna(0.0).ge(5.0)
            & transient_ratio.le(1.1)
            & pd.to_numeric(out[f"index_{offense}_primary"], errors="coerce").ge(800.0)
        )


    total_counts_arr = np.sum(np.vstack(total_counts), axis=0)
    out[_expected_count_col("personal")] = personal_counts
    out[_expected_count_col("property")] = property_counts
    out[_expected_count_col("total")] = total_counts_arr
    out["population_zero_with_positive_count"] = pop.le(0) & pd.Series(total_counts_arr, index=out.index).gt(0)
    for offense in OFFENSES_7:
        out[f"crime_density_{offense}"] = _crime_density(out[_expected_count_col(offense)], land_area_sq_mi)
    out["crime_density_total"] = _crime_density(pd.Series(total_counts_arr, index=out.index), land_area_sq_mi)

    # The legacy resident aggregates and the exposure-normalised harm index are SUPERSEDED on the
    # count-first lane, not duplicated: their replacements divide the same counts by the same
    # resident denominator under a publication rule that no per-offense denominator can veto, and
    # the harm index moves to tract support. Publishing both would leave a block-group harm number
    # on the frame, which is exactly what the tract-support rule forbids.
    if not count_first_composites:
        out["index_total_part1_resident"] = _resident_part1_index(out, list(OFFENSES_7))[0]
        out["index_personal_part1_resident"] = _resident_part1_index(out, list(PERSONAL_OFFENSES))[0]
        out["index_property_part1_resident"] = _resident_part1_index(out, list(PROPERTY_OFFENSES))[0]
        out["index_total_harm"] = _harm_weighted_total_index(out, list(OFFENSES_7))

    # `transient_exposure_likely_*` is diagnostic only. A high resident burden in a
    # daytime destination is often the literal answer to the selected resident
    # measure, while the primary arm already uses its offense-specific opportunity
    # denominator. Do not turn either valid rate into missing data merely because it
    # is large; an implausible value must remain visible and be repaired upstream.

    # The index-averaging composites are computed from the final published points.
    # Terms
    # whose cell display is suppressed for denominator invalidity enter at their
    # own tract support (identity grouping at tract). The block-group frame's
    # values here are provisional: apply_rare_offense_tract_support recomputes
    # them downstream with the tract-parquet rare-offense overrides layered in.
    composite_overrides: dict[str, pd.Series] = {}
    # Identical arithmetic on both lanes; only the NAME changes. An average of indexes whose
    # denominators differ is a dimensionless relative score, not a total rate, and the count-first
    # lane says so on the column (REVIEW_SOL_NEUTRAL.md sec.4).
    out[
        multi_offense_score_column(
            "index_total_primary_event_weighted", count_first=count_first_composites
        )
    ] = _full_component_index_composite(
        out,
        list(OFFENSES_7),
        index_suffix="primary",
        weights=_national_expected_count_weights(out, list(OFFENSES_7)),
        index_overrides=composite_overrides,
    )
    out[
        multi_offense_score_column("index_total_equal_offense", count_first=count_first_composites)
    ] = _full_component_index_composite(
        out,
        list(OFFENSES_7),
        index_suffix="primary",
        weights={offense: 1.0 for offense in OFFENSES_7},
        index_overrides=composite_overrides,
    )
    if count_first_composites:
        out[PERSONAL_RELATIVE_SCORE_COLUMN] = _full_component_index_composite(
            out,
            list(PERSONAL_OFFENSES),
            index_suffix="primary",
            weights=_national_expected_count_weights(out, list(PERSONAL_OFFENSES)),
            index_overrides={
                offense: composite_overrides[offense]
                for offense in PERSONAL_OFFENSES
                if offense in composite_overrides
            },
        )
        out[PROPERTY_RELATIVE_SCORE_COLUMN] = _full_component_index_composite(
            out,
            list(PROPERTY_OFFENSES),
            index_suffix="primary",
            weights=_national_expected_count_weights(out, list(PROPERTY_OFFENSES)),
            index_overrides={
                offense: composite_overrides[offense]
                for offense in PROPERTY_OFFENSES
                if offense in composite_overrides
            },
        )
    # Count-first composites last, and from published fields only: they read expected counts and
    # the resident denominator. Advisory diagnostics cannot move them.
    if count_first_composites:
        assert composites is not None  # narrowed by the guard at the top of the function
        out = apply_count_first_composites(
            out, runtime=composites, support=support_for_geo_id_col(geo_id_col)
        )

    ordered_cols = [geo_id_col, "state_fips", population_col]
    if "tract_id" in out.columns and geo_id_col != "tract_id":
        ordered_cols.append("tract_id")
    if "eb_jurisdiction_id" in out.columns:
        ordered_cols += ["eb_jurisdiction_id", "eb_jurisdiction_type"]
    if "dominant_eb_jurisdiction_id" in out.columns:
        ordered_cols += [
            "dominant_eb_jurisdiction_id",
            "dominant_jurisdiction_share",
            "mixed_jurisdiction_flag",
        ]
    ordered_cols += [
        "daytime_population_jobs_proxy",
        LANDSCAN_DAY_POP_COLUMN,
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
        # The near-zero-resident opportunity floor, beside the three floors above and gated on the
        # two lanes that created the class it closes. Absent on a legacy build.
        ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN,
        # v2 exposure-ensemble lane only: what the published per-offense denominators ARE. Filtered
        # out below on a legacy build, where the person denominator is still the hard max.
        "opportunity_normalizer_semantics",
        "non_residential_flag",
        "special_use_tract_flag",
        # v2 typed special-use lane only: the seven extensive classification inputs and the seven
        # typed outputs. Published together on purpose -- the type recomputes from the inputs, so
        # a reviewer (and the release validator) can re-derive every gate from the artifact alone.
        # Gated on the FLAG rather than on presence, so a stray classification column riding in on
        # an input frame cannot leak into a legacy artifact.
        *(SPECIAL_USE_PUBLISHED_COLUMNS if special_use_taxonomy else ()),
        "resident_secondary_denominator",
        "resident_secondary_denominator_low_reliability",
        "population_zero_with_positive_count",
        "transient_exposure_daytime_to_resident_ratio",
    ]
    ordered_cols += [_expected_count_col(offense) for offense in OFFENSES_7]
    for offense in OFFENSES_7:
        ordered_cols += [
            f"primary_denominator_type_{offense}",
            f"primary_denominator_{offense}",
            # v2 exposure-ensemble lane only; absent (and filtered out below) on a legacy build.
            normalizer_id_column(offense),
            opportunity_normalizer_column(offense),
            f"primary_denominator_raw_{offense}",
            f"primary_national_rate_per_100k_{offense}",
            f"primary_alpha_{offense}",
            f"primary_index_publishable_{offense}",
            f"primary_index_suppressed_{offense}",
            f"primary_zero_denominator_positive_count_{offense}",
        ]
        if offense == "motor_vehicle_theft":
            ordered_cols.append(f"primary_denominator_invalid_{offense}")
        ordered_cols += [
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
            _footprint_derived_count_col(offense),
            _footprint_derived_share_col(offense),
            _footprint_ambient_exposure_missing_col(offense),
            f"transient_exposure_likely_{offense}",
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
        ]
        if offense == "motor_vehicle_theft":
            ordered_cols.append(f"resident_denominator_invalid_{offense}")
        ordered_cols += [
            f"index_{offense}_resident_suppressed",
            f"rate_{offense}_resident",
            f"index_{offense}_resident",
        ]
    ordered_cols += [
        _expected_count_col("personal"),
        _expected_count_col("property"),
        _expected_count_col("total"),
        *[f"crime_density_{offense}" for offense in OFFENSES_7],
        "crime_density_total",
        # The harm index's numerator, published beside the counts so the index recomputes from
        # published fields at every support. Count-first lane only; filtered out below otherwise.
        HARM_WEIGHTED_COUNT_COLUMN,
        *aggregate_index_fields(count_first=count_first_composites),
    ]
    ordered_cols = [col for col in ordered_cols if col in out.columns]
    return out[ordered_cols].sort_values([geo_id_col], kind="mergesort").reset_index(drop=True)


def _controls_imputation_lane_differs(
    *, paths: RepoPaths, config: AllocationBuildConfig
) -> bool:
    """Do the controls on disk carry a different imputation lane than this build asked for?

    The lane is configuration, not an input file, so the dependency stamp cannot see it: without
    this check a `--enable-imputation-v2` build would reuse legacy controls, pass every freshness
    test, and record a lane in its manifest that its numbers do not actually carry. The controls
    build writes its own lane record, so this reads that rather than inferring anything.
    """
    record_path = paths.state_dir / "controls" / f"benchmark_imputation_{int(config.year)}.json"
    if not record_path.exists():
        return False  # nothing to reuse; the freshness check below owns this case
    try:
        record = json.loads(record_path.read_text()).get("imputation_v2") or {}
    except (OSError, json.JSONDecodeError):
        return True
    on_disk = all(
        bool((record.get(lane) or {}).get("enabled"))
        for lane in ("size_aware_municipal_rates", "cell_exposure_floor", "empirical_bounds")
    )
    return on_disk != bool(config.enable_imputation_v2)


def _ensure_output_dependencies(
    *,
    paths: RepoPaths,
    config: AllocationBuildConfig,
) -> None:
    controls_path = paths.state_dir / "controls" / f"jurisdiction_controls_{int(config.year)}.parquet"
    state_controls_path = paths.state_dir / "controls" / "state_control_comparison.parquet"
    jurisdiction_year_estimates_path = (
        paths.state_dir / "controls" / "jurisdiction_year_estimates.parquet"
    )
    control_build_config = ControlBuildConfig(
        year=int(config.year),
        force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
        enable_imputation_v2=bool(config.enable_imputation_v2),
    )
    controls_need_rebuild = (
        config.force_controls_rebuild
        or config.force_reporting_regimes_rebuild
        or _controls_imputation_lane_differs(paths=paths, config=config)
        or not controls_artifacts_are_current(
            paths,
            year=int(config.year),
            state_out_path=state_controls_path,
            jurisdiction_out_path=controls_path,
            jurisdiction_year_estimates_out_path=jurisdiction_year_estimates_path,
        )
    )

    # Refresh the observation/reference/reporting inputs before geometry. Controls
    # used to do this only after taking the controls lock, which meant an annual
    # observation refresh could replace reference-layer inputs after geometry had
    # already consumed them. It also forced a nested reporting-regime build while
    # controls was active. The preflight establishes the actual dependency order:
    # observations/reporting -> geometry -> controls.
    if controls_need_rebuild:
        _ensure_controls_dependencies(
            paths=paths,
            config=control_build_config,
            observation_ignore_blockers=("outputs",),
        )

    # Jurisdiction controls are built from the jurisdiction-ownership skeleton, and that
    # skeleton is derived from the block-group crosswalk.  Rebuilding controls before a
    # requested geometry refresh can therefore strand newly resolved jurisdictions: their
    # agency fills exist, but the stale ownership skeleton has no row on which to consume
    # them.  Geometry is the upstream dependency and must be current first.
    block_crosswalk_path = paths.state_dir / "geometry" / "block_to_jurisdiction_crosswalk.parquet"
    block_group_crosswalk_path = (
        paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    )
    if (
        config.force_geometry_rebuild
        or not geometry_artifacts_are_current(
            paths,
            block_out_path=block_crosswalk_path,
            block_group_out_path=block_group_crosswalk_path,
        )
    ):
        write_v2_geometry(
            paths=paths,
            block_out_path=block_crosswalk_path,
            block_group_out_path=block_group_crosswalk_path,
            config=GeometryBuildConfig(),
            force_rebuild=True,
            blocked_by=blockers_for_stage("geometry", ignore=("outputs",)),
        )

    if (
        controls_need_rebuild
        or not controls_artifacts_are_current(
            paths,
            year=int(config.year),
            state_out_path=state_controls_path,
            jurisdiction_out_path=controls_path,
            jurisdiction_year_estimates_out_path=jurisdiction_year_estimates_path,
        )
    ):
        write_v2_controls(
            paths=paths,
            state_out_path=state_controls_path,
            jurisdiction_out_path=controls_path,
            jurisdiction_year_estimates_out_path=jurisdiction_year_estimates_path,
            config=control_build_config,
            blocked_by=blockers_for_stage("controls", ignore=("outputs",)),
            observation_ignore_blockers=("outputs",),
        )

    if str(config.control_surface) == "smoothed":
        # The risk controls depend on the accounting controls and jurisdiction-year panel.
        # A controls rebuild above therefore invalidates them even when the file exists.
        from crimerisk.smoothed_controls import (
            SmoothedControlConfig,
            smoothed_controls_artifact_is_current,
            write_v2_smoothed_controls,
        )

        if not smoothed_controls_artifact_is_current(
            paths, year=int(config.year)
        ):
            write_v2_smoothed_controls(
                paths=paths,
                config=SmoothedControlConfig(year=int(config.year)),
            )

    feed_year_end = int(config.feed_year_end if config.feed_year_end is not None else config.year)
    write_v2_city_incident_shares(
        paths=paths,
        out_path=city_incident_share_surface_path(paths, feed_inputs_dir=config.feed_inputs_dir),
        reconciliation_dir=city_incident_reconciliation_dir(
            paths, feed_inputs_dir=config.feed_inputs_dir
        ),
        combined_reconciliation_path=city_incident_reconciliation_path(
            paths, year=feed_year_end, feed_inputs_dir=config.feed_inputs_dir
        ),
        config=CityIncidentShareBuildConfig(
            year_start=2018,
            year_end=feed_year_end,
            exclude_city_keys=tuple(config.exclude_feed_city_keys),
            force_rebuild=bool(config.force_city_incident_share_rebuild),
            force_source_refresh=bool(config.force_city_incident_source_refresh),
        ),
        blocked_by=blockers_for_stage("city_incident_shares", ignore=("outputs",)),
    )


def _uncertainty_dependency_paths(runtime: UncertaintyRuntime) -> list[Path]:
    """What the cached draws were built from: the measured dispersion tables and the engine.

    The components, finalized surfaces, direct-posterior audit, and calibration ratios are in
    memory rather than on disk, so their identities ride in the cache FILENAME as content digests
    instead. Both mechanisms together mean a cache hit is only possible when the draws and every
    published summary derived from them would be identical anyway.

    What this cache actually stores is SUMMARIES, not raw draws, and a summary is a function of the
    surface's publication mask as well as of the draws: `summarize_uncertainty_offense` reads
    `primary_index_publishable_<offense>`, the published denominator and the national rate off the
    finalised frame and nulls every decision field outside the publishable set. So the modules that
    DECIDE publication are dependencies of this cache in exactly the sense `uncertainty.py` is.
    Stamped after a measured miss: the near-zero-resident opportunity floor withdrew 367 robbery
    cells and the cache returned probabilities for all 367 of them, which the release validator
    caught as decision fields orphaned from a null index. A rebuild must not be able to move the
    publication mask and keep the old decision layer.
    """
    module_root = Path(__file__).resolve().parent
    return [
        *[Path(path) for path in runtime.config_paths],
        module_root / "uncertainty.py",
        # The publication mask's producers: the finalise path, the typed gate it reads, and the
        # normalizer the floors are measured against.
        module_root / "allocation.py",
        module_root / "special_use.py",
        module_root / "exposure_ensemble.py",
    ]


def _build_uncertainty_surfaces(
    *,
    paths: RepoPaths,
    config: AllocationBuildConfig,
    runtime: UncertaintyRuntime,
    components: pd.DataFrame,
    controls: pd.DataFrame,
    component_audit: pd.DataFrame,
    block_group_ags_core: pd.DataFrame,
    tract_ags_core: pd.DataFrame,
    block_group_fbi_calibrated: pd.DataFrame,
    tract_fbi_calibrated: pd.DataFrame,
    state_calibration_ratios: dict[tuple[str, str], float],
    population_col: str,
) -> dict[str, pd.DataFrame]:
    """One set of draws, four surfaces, cached on the content of what produced it."""
    surfaces: dict[str, pd.DataFrame] = {
        "block_group_ags_core": block_group_ags_core,
        "tract_ags_core": tract_ags_core,
        "block_group_fbi_calibrated": block_group_fbi_calibrated,
        "tract_fbi_calibrated": tract_fbi_calibrated,
    }
    geo_col = {
        "block_group_ags_core": "block_group_geoid",
        "tract_ags_core": "tract_id",
        "block_group_fbi_calibrated": "block_group_geoid",
        "tract_fbi_calibrated": "tract_id",
    }
    summary_inputs_signature = uncertainty_summary_inputs_signature(
        surfaces=surfaces,
        geo_columns=geo_col,
        component_audit=component_audit,
        state_calibration_ratios=state_calibration_ratios,
    )
    signature = hashlib.blake2b(
        (
            components_signature(components, controls)
            + summary_inputs_signature
            + runtime.signature()
        ).encode(),
        digest_size=12,
    ).hexdigest()
    cache_path = uncertainty_cache_path(paths, year=int(config.year), signature=signature)
    dependency_paths = _uncertainty_dependency_paths(runtime)
    cached = read_cached_cells(cache_path, dependency_paths=dependency_paths)

    if cached is not None:
        summaries_by_surface = {
            label: cache_frame_to_summaries(
                cached, surface=label, geo_ids=frame[geo_col[label]]
            )
            for label, frame in surfaces.items()
        }
    else:
        engine = UncertaintyEngine(
            components=components,
            controls=controls,
            component_audit=component_audit,
            block_group_ids=block_group_ags_core["block_group_geoid"],
            runtime=runtime,
        )
        # Every block group's parent tract, resolved once against the tract surface's own row
        # order, so the rollup lands on the tract the surface publishes rather than on a
        # reconstruction of it.
        tract_index = pd.Series(
            np.arange(len(tract_ags_core), dtype=np.int64),
            index=tract_ags_core["tract_id"].astype("string").str.zfill(11),
        )
        bg_tract_row = (
            block_group_ags_core["block_group_geoid"].astype("string").str.slice(0, 11).map(tract_index)
        )
        if bg_tract_row.isna().any():
            missing = int(bg_tract_row.isna().sum())
            raise ValueError(
                f"uncertainty layer: {missing} block group(s) have no parent row on the tract "
                "surface; the rollup would silently drop their mass."
            )
        bg_tract_row = bg_tract_row.to_numpy(dtype=np.int64)

        summaries_by_surface = {label: {} for label in surfaces}
        for offense in OFFENSES_7:
            draws = engine.offense_draws(offense)
            calibrated_draws = engine.offense_draws(
                offense,
                source_state_multipliers={
                    state: ratio
                    for (state, ratio_offense), ratio in state_calibration_ratios.items()
                    if ratio_offense == offense
                },
            )

            def _by_geography(offense_draws):
                return {
                    "block_group": (
                        offense_draws.counts,
                        offense_draws.control_only,
                        offense_draws.share_only,
                        offense_draws.point_counts,
                    ),
                    "tract": (
                        rollup_draws(
                            offense_draws.counts,
                            group_row=bg_tract_row,
                            n_groups=len(tract_ags_core),
                        ),
                        rollup_draws(
                            offense_draws.control_only,
                            group_row=bg_tract_row,
                            n_groups=len(tract_ags_core),
                        ),
                        rollup_draws(
                            offense_draws.share_only,
                            group_row=bg_tract_row,
                            n_groups=len(tract_ags_core),
                        ),
                        np.bincount(
                            bg_tract_row,
                            weights=offense_draws.point_counts,
                            minlength=len(tract_ags_core),
                        ),
                    ),
                }

            per_surface = {
                "ags_core": _by_geography(draws),
                "fbi_calibrated": _by_geography(calibrated_draws),
            }
            for label, frame in surfaces.items():
                geography = "tract" if geo_col[label] == "tract_id" else "block_group"
                calibration = (
                    "fbi_calibrated" if label.endswith("fbi_calibrated") else "ags_core"
                )
                counts, control_only, share_only, point = per_surface[calibration][geography]
                summaries_by_surface[label][offense] = summarize_uncertainty_offense(
                    frame,
                    offense=offense,
                    count_draws=counts,
                    control_log_sd=log_dispersion(control_only, point),
                    share_log_sd=log_dispersion(share_only, point),
                    runtime=runtime,
                )
        write_cached_cells(
            cache_path,
            summaries_to_cache_frame(
                {
                    label: (surfaces[label][geo_col[label]], summaries_by_surface[label])
                    for label in surfaces
                }
            ),
            dependency_paths=dependency_paths,
        )

    return {
        label: apply_uncertainty_layer(
            frame, summaries=summaries_by_surface[label], runtime=runtime
        )
        for label, frame in surfaces.items()
    }


def build_v2_outputs(
    *,
    paths: RepoPaths,
    config: AllocationBuildConfig = AllocationBuildConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config = resolve_allocation_build_config(paths, config=config)
    controls = _load_controls(
        paths,
        year=config.year,
        surface=str(config.control_surface),
    )
    state_controls = _load_state_controls(paths, year=config.year)
    bg_prior = _build_bg_prior_long(
        paths,
        config=config,
    )
    bg_crosswalk = _load_bg_crosswalk(paths)
    bg_crosswalk = bg_crosswalk[
        ~bg_crosswalk["state_fips"].astype(str).str.zfill(2).isin(RELEASE_EXCLUDED_STATE_FIPS)
    ].copy()
    agency_allocation_estimates = (
        _build_agency_allocation_target_estimates(paths=paths, year=int(config.year))
        if bool(config.enable_county_anchoring)
        else pd.DataFrame()
    )
    mixture_runtime = (
        resolve_mixture_runtime(
            paths=paths,
            config=MixtureAllocationConfig(
                year=int(config.year),
                enable_soft_shrinkage=bool(config.enable_soft_shrinkage),
                soft_shrinkage_nu=float(config.soft_shrinkage_nu),
                soft_shrinkage_extrapolation_weight=float(config.soft_shrinkage_extrapolation_weight),
                # The exposure expert is normalised over the same opportunity surface the
                # denominator lane quotes rates against, so one path override governs both.
                exposure_normalizers_path=config.exposure_normalizers_path,
            ),
            experts_path=config.mixture_experts_path,
            weights_path=config.mixture_weights_path,
            audit_path=(
                mixture_shares_audit_path(paths, year=int(config.year))
                if bool(config.write_mixture_share_audit)
                else None
            ),
        )
        if bool(config.enable_mixture_allocation)
        else None
    )
    exposure_ensemble_runtime = (
        resolve_exposure_ensemble_runtime(
            paths=paths,
            config=ExposureEnsembleConfig(year=int(config.year)),
            normalizers_path=config.exposure_normalizers_path,
            weights_path=config.exposure_ensemble_weights_path,
        )
        if bool(config.enable_exposure_ensemble)
        else None
    )
    if exposure_ensemble_runtime is not None:
        config = replace(
            config,
            exposure_residential_leg_source=exposure_ensemble_runtime.residential_leg_source,
        )
    composite_runtime = (
        resolve_composite_runtime(
            paths=paths,
            year=int(config.year),
            weights_path=config.severity_weights_path,
        )
        if bool(config.enable_count_first_composites)
        else None
    )

    jurisdiction_components = _build_jurisdiction_component_allocations(
        paths=paths,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        controls=controls,
        year=config.year,
        residual_training_city_shares_path=config.residual_training_city_shares_path,
        residual_training_exclude_validation_case_types=config.residual_training_exclude_validation_case_types,
        exclude_feed_city_keys=tuple(config.exclude_feed_city_keys),
        feed_year_end=config.feed_year_end,
        feed_inputs_dir=config.feed_inputs_dir,
        residual_training_extra_bg_feature_paths=tuple(config.residual_training_extra_bg_feature_paths),
        residual_feature_policy_path=config.residual_feature_policy_path,
        residual_exclude_feature_policy_classes=tuple(config.residual_exclude_feature_policy_classes),
        residual_exclude_feature_policy_classes_by_offense=tuple(
            config.residual_exclude_feature_policy_classes_by_offense
        ),
        residual_transfer_tau_by_offense=tuple(config.residual_transfer_tau_by_offense),
        rare_offense_information_constant_by_offense=tuple(
            config.rare_offense_information_constant_by_offense
        ),
        city_posterior_reconciliation_tolerance=float(config.city_posterior_reconciliation_tolerance),
        city_posterior_alpha_floor=float(config.city_posterior_alpha_floor),
        city_posterior_alpha_volume_incidents=float(config.city_posterior_alpha_volume_incidents),
        city_posterior_alpha_max_prior_fraction=float(config.city_posterior_alpha_max_prior_fraction),
        enable_county_anchoring=bool(config.enable_county_anchoring),
        agency_estimates=agency_allocation_estimates,
        mixture=mixture_runtime,
    )
    mixture_allocation_summary = jurisdiction_components.attrs.get("mixture_allocation_summary", {})
    city_posterior_diagnostics = jurisdiction_components.attrs.get("city_posterior_diagnostics", pd.DataFrame())
    city_posterior_summary = jurisdiction_components.attrs.get(
        "city_posterior_summary",
        _summarize_city_posterior_diagnostics(pd.DataFrame()),
    )
    city_residual_feature_policy = jurisdiction_components.attrs.get("city_residual_feature_policy", {})
    overlap_components = _build_overlap_allocations(
        paths=paths,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        controls=controls,
        year=config.year,
        enable_county_anchoring=bool(config.enable_county_anchoring),
        agency_estimates=agency_allocation_estimates,
        unlocated_mass=bool(config.enable_unlocated_mass),
    )
    unlocated_mass = overlap_components.attrs.get("unlocated_mass")
    footprint_mass_partition = overlap_components.attrs.get("footprint_mass_partition")
    all_components = pd.concat([jurisdiction_components, overlap_components], ignore_index=True)
    if "source_state_fips" not in all_components.columns:
        all_components["source_state_fips"] = all_components["state_fips"]
    else:
        all_components["source_state_fips"] = all_components["source_state_fips"].fillna(
            all_components["state_fips"]
        )
    all_components["source_state_fips"] = (
        all_components["source_state_fips"].astype("string").str.zfill(2)
    )
    bg_cov_raw = _load_bg_covariates(
        paths,
        year=config.year,
        burglary_commercial_weight=config.burglary_commercial_weight,
        exposure_ensemble=exposure_ensemble_runtime,
        special_use_taxonomy=bool(config.enable_special_use_taxonomy),
    )
    burglary_commercial_calibration = dict(bg_cov_raw.attrs.get("burglary_commercial_calibration", {}))
    exposure_ensemble_summary = (
        summarize_exposure_normalizers(bg_cov_raw, runtime=exposure_ensemble_runtime)
        if exposure_ensemble_runtime is not None
        else {"enabled": False}
    )
    all_components = _apply_model_only_allocation_envelopes(
        _components_with_primary_denominator(all_components, bg_cov_raw),
        soft_shrinkage=bool(config.enable_soft_shrinkage),
        soft_shrinkage_nu=float(config.soft_shrinkage_nu),
    )
    all_components, zero_target_audit, component_audit = _redistribute_zero_target_components(
        all_components,
        bg_cov_raw,
    )
    service_unlocated = all_components.attrs.get("service_unlocated_mass")
    service_state_transfer = _build_service_state_transfer_ledger(
        all_components,
        year=int(config.year),
        service_unlocated=(
            service_unlocated if isinstance(service_unlocated, pd.DataFrame) else None
        ),
    )
    if isinstance(service_unlocated, pd.DataFrame) and not service_unlocated.empty:
        service_companion = service_unlocated.copy()
        service_companion["state_fips"] = service_companion["source_state_fips"]
        service_companion["jurisdiction_id"] = service_companion["canonical_target_ori"]
        service_companion["jurisdiction_type"] = UNLOCATED_JURISDICTION_TYPE
        service_companion["control_count"] = service_companion["source_target_count"]
        service_companion["unlocated_share_of_control"] = np.where(
            service_companion["control_count"].gt(0.0),
            service_companion["unlocated_count"] / service_companion["control_count"],
            0.0,
        )
        for route in sorted(UNLOCATED_ROUTES):
            service_companion[f"unlocated_count_{route}"] = np.where(
                service_companion["route_reason"].eq(route),
                service_companion["unlocated_count"],
                0.0,
            )
        if unlocated_mass is None:
            unlocated_mass = service_companion
        else:
            existing_unlocated = unlocated_mass.copy()
            for column in service_companion.columns:
                if column not in existing_unlocated.columns:
                    existing_unlocated[column] = pd.NA
            for column in existing_unlocated.columns:
                if column not in service_companion.columns:
                    service_companion[column] = pd.NA
            unlocated_mass = pd.concat(
                [existing_unlocated, service_companion[existing_unlocated.columns]],
                ignore_index=True,
            )

    # Carry custom-footprint provenance as mass so the tract rollup derives its share at the
    # support where it is published.
    footprint_mass_conservation = _build_footprint_mass_conservation(
        all_components, footprint_mass_partition
    )
    bg_counts, footprint_counts = _component_count_tables(all_components)
    bg_counts = bg_counts.merge(footprint_counts, on=["state_fips", "bg_id", "tract_id"], how="left")
    for offense in OFFENSES_7:
        col = _footprint_derived_count_col(offense)
        bg_counts[col] = pd.to_numeric(bg_counts[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    population_col = f"population_{int(config.year)}"
    bg_cov = bg_cov_raw.rename(
        columns={"bg_id": "block_group_geoid", "population": population_col}
    )
    bg_universe = bg_crosswalk[["state_fips", "block_group_geoid"]].drop_duplicates().copy()
    bg_universe["state_fips"] = bg_universe["state_fips"].astype("string").str.zfill(2)
    bg_universe["block_group_geoid"] = bg_universe["block_group_geoid"].astype("string").str.zfill(12)
    bg_universe["tract_id"] = bg_universe["block_group_geoid"].str.slice(0, 11)
    bg_dominant_jurisdiction = _dominant_bg_jurisdiction(bg_crosswalk)
    bg_direct_support = _build_bg_direct_incident_support(
        paths=paths,
        bg_crosswalk=bg_crosswalk,
        year=int(config.feed_year_end if config.feed_year_end is not None else config.year),
        exclude_feed_city_keys=tuple(config.exclude_feed_city_keys),
        feed_inputs_dir=config.feed_inputs_dir,
    )
    bg_out = bg_universe.merge(
        bg_counts,
        left_on=["block_group_geoid", "tract_id", "state_fips"],
        right_on=["bg_id", "tract_id", "state_fips"],
        how="left",
    ).merge(
        bg_cov,
        left_on=["block_group_geoid", "tract_id", "state_fips"],
        right_on=["block_group_geoid", "tract_id", "state_fips"],
        how="left",
    ).merge(
        bg_dominant_jurisdiction,
        on="block_group_geoid",
        how="left",
    ).merge(
        bg_direct_support,
        on=["state_fips", "block_group_geoid"],
        how="left",
    )
    for offense in OFFENSES_7:
        col = _expected_count_col(offense)
        bg_out[col] = pd.to_numeric(bg_out.get(col), errors="coerce").fillna(0.0)
        footprint_col = _footprint_derived_count_col(offense)
        bg_out[footprint_col] = pd.to_numeric(bg_out.get(footprint_col), errors="coerce").fillna(0.0)
    bg_out[population_col] = pd.to_numeric(bg_out.get(population_col), errors="coerce").fillna(0.0)
    bg_out = bg_out.drop(columns=["bg_id"], errors="ignore")
    bg_out = _finalize_output(
        bg_out,
        geo_id_col="block_group_geoid",
        population_col=population_col,
        config=config,
        composites=composite_runtime,
    )

    tract_jurisdiction = _dominant_tract_jurisdiction(bg_crosswalk, bg_out, population_col=population_col)
    tract_counts = _rollup_tracts_from_bg(bg_out, tract_jurisdiction, population_col=population_col)
    tract_counts = _attach_tiger_land_area(
        tract_counts,
        land_area=_load_tiger_land_area(paths, geography="tract"),
        geoid_col="tract_id",
    )
    tract_out = _finalize_output(
        tract_counts,
        geo_id_col="tract_id",
        population_col=population_col,
        config=config,
        jurisdiction_col="dominant_eb_jurisdiction_id",
        composites=composite_runtime,
    )

    ratios = state_controls[["state_fips", "offense", "ags_core_adjusted_total", "fbi_cde_estimated_total"]].copy()
    ratios["calibration_ratio"] = np.where(
        pd.to_numeric(ratios["ags_core_adjusted_total"], errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(ratios["fbi_cde_estimated_total"], errors="coerce").fillna(0.0)
        / pd.to_numeric(ratios["ags_core_adjusted_total"], errors="coerce").fillna(1.0),
        1.0,
    )
    ratio_map = {
        (str(row.state_fips).zfill(2), str(row.offense)): float(row.calibration_ratio)
        for row in ratios.itertuples(index=False)
        if pd.notna(row.calibration_ratio) and np.isfinite(row.calibration_ratio)
    }

    calibrated_components = _scale_components_by_source_state(all_components, ratio_map)
    calibrated_counts, calibrated_footprint_counts = _component_count_tables(
        calibrated_components
    )
    calibrated_counts = calibrated_counts.merge(
        calibrated_footprint_counts,
        on=["state_fips", "bg_id", "tract_id"],
        how="left",
    )
    bg_cal = (
        bg_out.drop(
            columns=[
                *[_expected_count_col(offense) for offense in OFFENSES_7],
                *[_footprint_derived_count_col(offense) for offense in OFFENSES_7],
            ],
            errors="ignore",
        )
        .merge(
            calibrated_counts,
            left_on=["state_fips", "block_group_geoid", "tract_id"],
            right_on=["state_fips", "bg_id", "tract_id"],
            how="left",
            validate="one_to_one",
        )
        .drop(columns=["bg_id"], errors="ignore")
    )
    for offense in OFFENSES_7:
        for column in (_expected_count_col(offense), _footprint_derived_count_col(offense)):
            bg_cal[column] = pd.to_numeric(bg_cal.get(column), errors="coerce").fillna(0.0)

    bg_cal = _finalize_output(
        bg_cal,
        geo_id_col="block_group_geoid",
        population_col=population_col,
        config=config,
        composites=composite_runtime,
    )
    tract_cal_counts = _rollup_tracts_from_bg(bg_cal, tract_jurisdiction, population_col=population_col)
    tract_cal_counts = _attach_tiger_land_area(
        tract_cal_counts,
        land_area=_load_tiger_land_area(paths, geography="tract"),
        geoid_col="tract_id",
    )
    tract_cal = _finalize_output(
        tract_cal_counts,
        geo_id_col="tract_id",
        population_col=population_col,
        config=config,
        jurisdiction_col="dominant_eb_jurisdiction_id",
        composites=composite_runtime,
    )

    confidence_artifacts = build_confidence_artifacts(
        paths=paths,
        year=int(config.year),
        block_group_surface=bg_out,
        component_audit=component_audit,
        city_posterior_diagnostics=city_posterior_diagnostics,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        residual_training_city_shares_path=config.residual_training_city_shares_path,
        residual_training_exclude_validation_case_types=tuple(config.residual_training_exclude_validation_case_types),
        exclude_feed_city_keys=tuple(config.exclude_feed_city_keys),
        feed_year_end=config.feed_year_end,
        residual_training_extra_bg_feature_paths=tuple(config.residual_training_extra_bg_feature_paths),
        residual_feature_policy_path=_resolve_repo_path(paths, config.residual_feature_policy_path),
        residual_exclude_feature_policy_classes=tuple(config.residual_exclude_feature_policy_classes),
        residual_exclude_feature_policy_classes_by_offense=tuple(
            config.residual_exclude_feature_policy_classes_by_offense
        ),
    )
    bg_out, tract_out = enrich_confidence_surfaces(
        block_group_surface=bg_out,
        tract_surface=tract_out,
        artifacts=confidence_artifacts,
        population_col=population_col,
        year=int(config.year),
    )
    bg_cal, tract_cal = enrich_confidence_surfaces(
        block_group_surface=bg_cal,
        tract_surface=tract_cal,
        artifacts=confidence_artifacts,
        population_col=population_col,
        year=int(config.year),
    )

    # --- v2 uncertainty layer ---------------------------------------------------------------
    # Applied LAST, after every point field and every confidence field exists, and applied to all
    # four surfaces from ONE set of draws. Two consequences are deliberate: the tier can read the
    # benchmark share, the unresolved-level flag and the domain score that confidence.py publishes
    # rather than recomputing its own versions of them; and the four surfaces can never disagree
    # about the same cell, because the tract surface is the block-group draws summed within tract
    # and the calibrated twin is the same draws under the same per-state scalar.
    uncertainty_runtime: UncertaintyRuntime | None = None
    if bool(getattr(config, "enable_uncertainty_layer", False)):
        uncertainty_runtime = resolve_uncertainty_runtime(
            paths,
            n_draws=int(config.uncertainty_draws),
            control_dispersion_file=config.uncertainty_control_dispersion_path,
            share_dispersion_file=config.uncertainty_share_dispersion_path,
            calibration_file=config.uncertainty_calibration_path,
        )
        surfaces = _build_uncertainty_surfaces(
            paths=paths,
            config=config,
            runtime=uncertainty_runtime,
            components=all_components,
            controls=controls,
            component_audit=component_audit,
            block_group_ags_core=bg_out,
            tract_ags_core=tract_out,
            block_group_fbi_calibrated=bg_cal,
            tract_fbi_calibrated=tract_cal,
            state_calibration_ratios=ratio_map,
            population_col=population_col,
        )
        bg_out = surfaces["block_group_ags_core"]
        tract_out = surfaces["tract_ags_core"]
        bg_cal = surfaces["block_group_fbi_calibrated"]
        tract_cal = surfaces["tract_fbi_calibrated"]

    diagnostics: dict[str, object] = {
        "burglary_commercial_calibration": burglary_commercial_calibration,
        "burglary_commercial_gradient": {
            "block_group_ags_core": _burglary_commercial_gradient_diagnostics(bg_out),
            "tract_ags_core": _burglary_commercial_gradient_diagnostics(tract_out),
            "block_group_fbi_calibrated": _burglary_commercial_gradient_diagnostics(bg_cal),
            "tract_fbi_calibrated": _burglary_commercial_gradient_diagnostics(tract_cal),
        },
        "suppression_mode_counts": {
            "block_group_ags_core": _suppression_mode_summary(bg_out),
            "tract_ags_core": _suppression_mode_summary(tract_out),
            "block_group_fbi_calibrated": _suppression_mode_summary(bg_cal),
            "tract_fbi_calibrated": _suppression_mode_summary(tract_cal),
        },
        "city_posterior_diagnostics": city_posterior_diagnostics,
        "city_posterior_summary": city_posterior_summary,
        "city_residual_feature_policy": city_residual_feature_policy,
        "mixture_allocation_summary": mixture_allocation_summary,
        "exposure_ensemble_summary": exposure_ensemble_summary,
        "special_use_taxonomy_summary": {
            "block_group_ags_core": summarize_special_use_taxonomy(bg_out),
            "tract_ags_core": summarize_special_use_taxonomy(tract_out),
        },
        "uncertainty_layer_summary": summarize_uncertainty_layer(
            runtime=uncertainty_runtime,
            surfaces=(
                {"block_group_ags_core": bg_out, "tract_ags_core": tract_out}
                if uncertainty_runtime is not None
                else None
            ),
        ),
        "count_first_composites_summary": (
            summarize_count_first_composites(runtime=composite_runtime)
            if composite_runtime is not None
            else {"enabled": False}
        ),
        # Severity sensitivity is measured at the support the harm composite publishes at.
        # Measuring it at block group would price the stability of a surface this lane refuses
        # to publish.
        "count_first_composite_sensitivity": (
            pd.concat(
                [
                    severity_sensitivity(
                        surface,
                        weights=composite_runtime.weights,
                        label=label,
                        support="tract",
                    )
                    for label, surface in (("tract_ags_core", tract_out), ("tract_fbi_calibrated", tract_cal))
                ],
                ignore_index=True,
            )
            if composite_runtime is not None
            else pd.DataFrame()
        ),
        "unlocated_mass": unlocated_mass,
        "service_state_transfer": service_state_transfer,
        "footprint_mass_conservation": footprint_mass_conservation,
    }
    if unlocated_mass is not None or not service_state_transfer.empty:
        # The identity the lane exists to make true, asserted on the surface that was actually
        # built rather than on the targets that fed it. Cross-state service allocations reconcile
        # geographic publication back to the one source-state control through the bound ledger.
        diagnostics["unlocated_mass_conservation"] = _assert_unlocated_mass_conservation(
            surface=bg_out,
            controls=controls,
            unlocated=(
                unlocated_mass
                if isinstance(unlocated_mass, pd.DataFrame)
                else pd.DataFrame(columns=["state_fips", "offense", "unlocated_count"])
            ),
            service_state_transfer=service_state_transfer,
            year=int(config.year),
        )
    return bg_out, tract_out, bg_cal, tract_cal, zero_target_audit, component_audit, diagnostics


def _assert_unlocated_mass_conservation(
    *,
    surface: pd.DataFrame,
    controls: pd.DataFrame,
    unlocated: pd.DataFrame,
    service_state_transfer: pd.DataFrame | None = None,
    year: int,
) -> dict[str, object]:
    """Reconstruct source-state controls from geographic publication and explicit transfers.

    With the lane OFF this is not called. With it on, the widened identity is asserted at build
    time and mirrored from the published artifacts by the release validator.
    """
    keys = ["state_fips", "offense"]
    control_total = (
        controls.assign(state_fips=controls["state_fips"].astype("string").str.zfill(2))
        .groupby(keys, dropna=False)["adjusted_count_ags_core"]
        .sum()
        .rename("control_count")
        .reset_index()
    )
    published = (
        surface.assign(state_fips=surface["state_fips"].astype("string").str.zfill(2))
        .melt(
            id_vars=["state_fips"],
            value_vars=[_expected_count_col(offense) for offense in OFFENSES_7],
            var_name="_field",
            value_name="published_count",
        )
        .assign(offense=lambda frame: frame["_field"].str.removeprefix("expected_count_"))
        .groupby(keys, dropna=False)["published_count"]
        .sum()
        .reset_index()
    )
    withheld = (
        unlocated.groupby(keys, dropna=False)["unlocated_count"].sum().reset_index()
    )
    merged = (
        control_total.merge(published, on=keys, how="outer")
        .merge(withheld, on=keys, how="outer")
        .fillna({"control_count": 0.0, "published_count": 0.0, "unlocated_count": 0.0})
    )
    merged["service_transfer_adjustment"] = 0.0
    if service_state_transfer is not None and not service_state_transfer.empty:
        located = service_state_transfer[
            service_state_transfer["allocation_state_fips"].notna()
        ].copy()
        located["source_state_fips"] = located["source_state_fips"].astype("string").str.zfill(2)
        located["allocation_state_fips"] = (
            located["allocation_state_fips"].astype("string").str.zfill(2)
        )
        cross_state = located[
            located["source_state_fips"].ne(located["allocation_state_fips"])
        ]
        exports = (
            cross_state.groupby(["source_state_fips", "offense"], as_index=False)[
                "allocation_count"
            ]
            .sum()
            .rename(
                columns={
                    "source_state_fips": "state_fips",
                    "allocation_count": "service_exports",
                }
            )
        )
        imports = (
            cross_state.groupby(["allocation_state_fips", "offense"], as_index=False)[
                "allocation_count"
            ]
            .sum()
            .rename(
                columns={
                    "allocation_state_fips": "state_fips",
                    "allocation_count": "service_imports",
                }
            )
        )
        merged = merged.merge(exports, on=keys, how="outer").merge(
            imports, on=keys, how="outer"
        )
        for column in ("control_count", "published_count", "unlocated_count"):
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
        merged["service_transfer_adjustment"] = (
            pd.to_numeric(merged["service_exports"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["service_imports"], errors="coerce").fillna(0.0)
        )
    merged["delta"] = (
        merged["published_count"]
        + merged["unlocated_count"]
        + merged["service_transfer_adjustment"]
        - merged["control_count"]
    )
    max_delta = float(merged["delta"].abs().max() or 0.0)
    if max_delta > 1e-6:
        worst = merged.reindex(merged["delta"].abs().sort_values(ascending=False).index).head(5)
        raise ValueError(
            "published block groups plus unlocated mass do not reconstruct the controls; "
            f"max abs delta={max_delta:.3e}: {worst.to_dict(orient='records')}"
        )
    return {
        "year": int(year),
        "identity": (
            "published_block_group_sum + unlocated + service_exports - service_imports "
            "== source_state_control"
        ),
        "state_offense_rows": int(len(merged)),
        "max_abs_delta": max_delta,
        "unlocated_total": float(merged["unlocated_count"].sum()),
        "control_total": float(merged["control_count"].sum()),
    }


def write_v2_outputs(
    *,
    paths: RepoPaths,
    block_group_ags_core_out: Path,
    tract_ags_core_out: Path,
    block_group_fbi_out: Path | None = None,
    tract_fbi_out: Path | None = None,
    build_manifest_out: Path | None = None,
    run_metadata: dict[str, object] | None = None,
    config: AllocationBuildConfig = AllocationBuildConfig(),
) -> dict[str, object]:
    resolved_config = resolve_allocation_build_config(paths, config=config)
    residual_transfer_tau = _residual_transfer_tau_dict(resolved_config.residual_transfer_tau_by_offense)
    rare_offense_information_constants = _rare_offense_information_constant_dict(
        resolved_config.rare_offense_information_constant_by_offense
    )
    with stage_write_lock(paths=paths, stage="outputs"):
        _ensure_output_dependencies(paths=paths, config=resolved_config)
        bg_out, tract_out, bg_cal, tract_cal, zero_target_audit, component_audit, diagnostics = build_v2_outputs(
            paths=paths,
            config=resolved_config,
        )
        # Publication-support policy: carry murder/rape per-offense indices only at tract and
        # coarser, with the rare-offense-sensitive aggregates re-expressed compositionally. Applied
        # here at publication time so every internal diagnostic and confidence surface upstream is
        # computed from the full block-group signal (see RARE_OFFENSE_TRACT_SUPPORT).
        count_first_composites = bool(resolved_config.enable_count_first_composites)
        bg_out = apply_rare_offense_tract_support(
            bg_out, tract_out, count_first_composites=count_first_composites
        )
        bg_cal = apply_rare_offense_tract_support(
            bg_cal, tract_cal, count_first_composites=count_first_composites
        )
        out_paths = [block_group_ags_core_out, tract_ags_core_out]
        if block_group_fbi_out is not None and tract_fbi_out is not None:
            out_paths.extend([block_group_fbi_out, tract_fbi_out])
        for out_path in out_paths:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        bg_out.to_parquet(block_group_ags_core_out, index=False)
        tract_out.to_parquet(tract_ags_core_out, index=False)
        summary = {
            "block_groups": int(len(bg_out)),
            "tracts": int(len(tract_out)),
            "fbi_calibrated_written": False,
            "county_anchoring_enabled": bool(resolved_config.enable_county_anchoring),
            "promoted_next_phase_allocator_enabled": bool(resolved_config.use_promoted_next_phase_allocator),
            "promoted_next_phase_allocator_applied": _promoted_next_phase_allocator_applied(
                paths,
                config=resolved_config,
            ),
            "residual_training_city_shares_path": (
                str(resolved_config.residual_training_city_shares_path)
                if resolved_config.residual_training_city_shares_path is not None
                else None
            ),
            "residual_training_exclude_validation_case_types": list(
                resolved_config.residual_training_exclude_validation_case_types
            ),
            "residual_training_extra_bg_feature_paths": [
                str(path) for path in resolved_config.residual_training_extra_bg_feature_paths
            ],
            "bg_prior_path": str(
                _bg_prior_cache_path(
                    paths=paths,
                    config=resolved_config,
                    model_surface_config=_model_surface_config_from_allocation(paths=paths, config=resolved_config),
                )
            ),
            "model_surface_prior_anchor": str(resolved_config.model_surface_prior_anchor),
            "model_surface_feature_policy_path": (
                str(_resolve_repo_path(paths, resolved_config.model_surface_feature_policy_path))
                if resolved_config.model_surface_feature_policy_path is not None
                else None
            ),
            "model_surface_exclude_feature_policy_classes": list(
                resolved_config.model_surface_exclude_feature_policy_classes
            ),
            "residual_feature_policy_path": (
                str(_resolve_repo_path(paths, resolved_config.residual_feature_policy_path))
                if resolved_config.residual_feature_policy_path is not None
                else None
            ),
            "residual_exclude_feature_policy_classes": [
                str(value) for value in resolved_config.residual_exclude_feature_policy_classes
            ],
            "residual_exclude_feature_policy_classes_by_offense": {
                str(offense): [str(value) for value in classes]
                for offense, classes in resolved_config.residual_exclude_feature_policy_classes_by_offense
            },
            "residual_transfer_tau_by_offense": {
                str(offense): float(residual_transfer_tau[offense])
                for offense in OFFENSES_7
            },
            "rare_offense_tract_information_shrinkage": {
                "offenses": sorted(SPARSE_BASELINE_TRANSFER_OFFENSES),
                "support": "tract_then_residential_exposure_spread",
                "information_weight": "effective_model_information / (effective_model_information + K_offense)",
                "residential_exposure_source": "2020 Census block-group population * jurisdiction allocation_share",
                "industrial_jobs_exposure_included": False,
                "information_constant_by_offense": rare_offense_information_constants,
                "selection": (
                    "rare_offense_information_v2: five-fold grouped Poisson deviance; "
                    "E4 cells report-only; seed 20240728"
                ),
            },
            "unified_murder_tract_posterior": {
                "support": "census_tract",
                "incident_half_life_years": float(MURDER_INCIDENT_HALF_LIFE_YEARS),
                "prior_incidents": float(MURDER_TRACT_POSTERIOR_PRIOR_INCIDENTS),
                "within_tract_distribution": "rare_offense_prior_only",
                "calibration_path": (
                    f"state/modeling/murder_tract_posterior_calibration_{int(resolved_config.year)}.json"
                ),
            },
            "model_only_allocation_envelopes": {
                "robbery": {
                    "support": "block_group",
                    "within_jurisdiction_rate_ratio_cap": float(MODEL_ONLY_ROBBERY_BG_RATE_RATIO_CAP),
                },
                "murder": {
                    "support": "tract",
                    "within_jurisdiction_rate_ratio_cap": float(MODEL_ONLY_MURDER_TRACT_RATE_RATIO_CAP),
                },
                "redistribution": "within source jurisdiction/offense proportional to untouched model-only component mass",
                "direct_city_components_eligible": False,
            },
            "burglary_commercial_calibration": diagnostics.get("burglary_commercial_calibration", {}),
            "person_exposure_denominator_policy": {
                "offenses": sorted(PERSON_EXPOSURE_DENOMINATOR_OFFENSES),
                "publication_formula": "max(daytime_population_jobs_proxy, landscan_day_pop) where landscan_day_pop > 0; otherwise daytime_population_jobs_proxy, then apply bounded HQ-jobs cap where triggered",
                "allocation_baseline_decision": _landscan_lift_decision_record(paths, year=int(resolved_config.year)),
                "allocation_baseline_policy": (
                    "LandScan is used for publication denominators only when the decision artifact rejects "
                    "allocation use; model-surface/prior construction keeps apply_landscan_day_floor=False."
                ),
                "hq_jobs_cap": {
                    "condition": "jobs_wac >= 5000 and residents + jobs_wac > 3 * max(landscan_day_pop, residents)",
                    "cap": "3 * max(landscan_day_pop, residents)",
                    "audit_columns": [
                        "person_exposure_before_hq_jobs_cap",
                        "person_exposure_hq_jobs_cap",
                        "person_exposure_hq_jobs_cap_candidate",
                        "person_exposure_hq_jobs_capped",
                    ],
                },
                "landscan_source": "LandScan USA 2021 (ORNL), CC BY 4.0",
                "tourist_visitor_limitation": (
                    "No public dataset provides tourist/visitor ambient population; visitor-heavy areas may still "
                    "overstate per-person risk and are flagged."
                ),
            },
            "motor_vehicle_theft_denominator_policy": {
                "formula": "ACS household vehicles + LODES jobs_wac * county ACS B08301 drove-alone/carpool commute share",
                "publication_floor": float(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR),
                "floor_estimate_mode": "insufficient_exposure",
                "audit_columns": [
                    "aggregate_vehicles_total",
                    "county_auto_commute_vehicle_share",
                    "mvt_commuter_vehicle_proxy",
                    "vehicle_exposure_2024",
                ],
            },
            "burglary_commercial_gradient": diagnostics.get("burglary_commercial_gradient", {}),
            "suppression_mode_counts": diagnostics.get("suppression_mode_counts", {}),
            "city_posterior_share": diagnostics.get("city_posterior_summary", {}),
            "city_residual_feature_policy": diagnostics.get("city_residual_feature_policy", {}),
            "mixture_allocation": diagnostics.get("mixture_allocation_summary", {}),
            "exposure_ensemble": diagnostics.get("exposure_ensemble_summary", {}),
            "count_first_composites": diagnostics.get("count_first_composites_summary", {}),
            "special_use_taxonomy": diagnostics.get("special_use_taxonomy_summary", {}),
        }
        if block_group_fbi_out is not None and tract_fbi_out is not None:
            bg_cal.to_parquet(block_group_fbi_out, index=False)
            tract_cal.to_parquet(tract_fbi_out, index=False)
            summary["fbi_calibrated_written"] = True
        composite_runtime = (
            resolve_composite_runtime(
                paths=paths,
                year=int(resolved_config.year),
                weights_path=resolved_config.severity_weights_path,
            )
            if count_first_composites
            else None
        )
        summary["aggregate_index_normalizers"] = {
            label: _aggregate_index_normalizers(
                surface, composites=composite_runtime, support=support
            )
            for label, surface, support in (
                ("block_group_ags_core", bg_out, "block_group"),
                ("tract_ags_core", tract_out, "tract"),
                ("block_group_fbi_calibrated", bg_cal, "block_group"),
                ("tract_fbi_calibrated", tract_cal, "tract"),
            )
        }
        manifest_path = build_manifest_out or (
            block_group_ags_core_out.parent / f"crimerisk_output_build_{int(resolved_config.year)}.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        zero_target_audit_path = manifest_path.parent / f"zero_target_denominator_audit_{int(resolved_config.year)}.parquet"
        component_audit_path = manifest_path.parent / f"allocation_component_denominator_audit_{int(resolved_config.year)}.parquet"
        service_state_transfer_path = (
            manifest_path.parent
            / f"allocation_service_state_transfer_{int(resolved_config.year)}.parquet"
        )
        publishability_audit_path = manifest_path.parent / f"denominator_publishability_audit_{int(resolved_config.year)}.csv"
        city_posterior_diagnostics_path = manifest_path.parent / f"city_posterior_diagnostics_{int(resolved_config.year)}.parquet"
        zero_target_audit.to_parquet(zero_target_audit_path, index=False)
        component_audit.to_parquet(component_audit_path, index=False)
        service_state_transfer = diagnostics.get("service_state_transfer")
        if not isinstance(service_state_transfer, pd.DataFrame):
            service_state_transfer = pd.DataFrame(
                columns=list(SERVICE_STATE_TRANSFER_COLUMNS)
            )
        if not service_state_transfer.empty and not service_state_transfer["year"].eq(
            int(resolved_config.year)
        ).all():
            raise ValueError("service-state transfer rows do not match the allocation build year")
        service_state_transfer.to_parquet(service_state_transfer_path, index=False)
        _build_publishability_audit(bg_out).to_csv(publishability_audit_path, index=False)
        city_posterior_diagnostics = diagnostics.get("city_posterior_diagnostics")
        if isinstance(city_posterior_diagnostics, pd.DataFrame):
            city_posterior_diagnostics.to_parquet(city_posterior_diagnostics_path, index=False)
            summary["city_posterior_diagnostics_rows"] = int(len(city_posterior_diagnostics))
            summary["city_posterior_diagnostics_path"] = str(city_posterior_diagnostics_path)
        composite_sensitivity = diagnostics.get("count_first_composite_sensitivity")
        if isinstance(composite_sensitivity, pd.DataFrame) and not composite_sensitivity.empty:
            # The published record of "how much of the harm ranking is the severity choice",
            # written beside the manifest so it ships with the edition it describes.
            sensitivity_path = severity_sensitivity_path(
                manifest_path.parent, year=int(resolved_config.year)
            )
            composite_sensitivity.to_csv(sensitivity_path, index=False)
            summary["composite_severity_sensitivity_path"] = str(sensitivity_path)
            summary["composite_severity_sensitivity"] = [
                {key: (None if pd.isna(value) else value) for key, value in row.items()}
                for row in composite_sensitivity.to_dict(orient="records")
            ]
        unlocated_mass = diagnostics.get("unlocated_mass")
        unlocated_mass_out = None
        if isinstance(unlocated_mass, pd.DataFrame):
            # The companion table. It ships beside the surface it completes, because a published
            # surface whose conservation identity refers to a file that did not travel with it is
            # not a conserved surface.
            unlocated_mass_out = unlocated_mass_path(manifest_path.parent, year=int(resolved_config.year))
            unlocated_mass.to_parquet(unlocated_mass_out, index=False)
            summary["unlocated_mass_path"] = str(unlocated_mass_out)
            summary["unlocated_mass_rows"] = int(len(unlocated_mass))
            summary["unlocated_mass"] = {
                "enabled": True,
                "total": float(pd.to_numeric(unlocated_mass["unlocated_count"], errors="coerce").fillna(0.0).sum()),
                "by_offense": {
                    str(offense): float(value)
                    for offense, value in unlocated_mass.groupby("offense")["unlocated_count"].sum().items()
                },
                "by_route": {
                    f"unlocated_count_{route}": float(
                        pd.to_numeric(unlocated_mass[f"unlocated_count_{route}"], errors="coerce").fillna(0.0).sum()
                    )
                    for route in sorted(UNLOCATED_ROUTES)
                },
                "conservation": diagnostics.get("unlocated_mass_conservation", {}),
            }
        footprint_conservation = diagnostics.get("footprint_mass_conservation")
        if isinstance(footprint_conservation, pd.DataFrame):
            # The per-ORI conservation proof travels with the surface for the same reason the
            # unlocated table does: a conservation claim whose evidence stayed behind is not a
            # claim anyone can check.
            footprint_conservation_out = footprint_mass_conservation_path(
                manifest_path.parent, year=int(resolved_config.year)
            )
            footprint_conservation.to_parquet(footprint_conservation_out, index=False)
            relative_error = pd.to_numeric(
                footprint_conservation.get("relative_error"), errors="coerce"
            )
            by_ori = _footprint_conservation_by_ori(footprint_conservation)
            summary["footprint_mass_conservation_path"] = str(footprint_conservation_out)
            summary["footprint_mass_conservation_rows"] = int(len(footprint_conservation))
            summary["footprint_mass_conservation"] = {
                "max_relative_error_threshold": float(
                    FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR
                ),
                "min_absolute_error_threshold": float(
                    FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR
                ),
                "oris_over_threshold": int(
                    pd.to_numeric(by_ori["relative_error"], errors="coerce")
                    .gt(FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR)
                    .sum()
                ),
                "max_ori_relative_error": (
                    float(
                        pd.to_numeric(by_ori["relative_error"], errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .max()
                    )
                    if len(by_ori)
                    else 0.0
                ),
                "ori_offense_rows_over_threshold": int(
                    relative_error.gt(FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR).sum()
                ),
                "max_relative_error": (
                    float(relative_error.replace([np.inf, -np.inf], np.nan).max())
                    if len(relative_error)
                    else 0.0
                ),
                "status_counts": {
                    str(status): int(count)
                    for status, count in footprint_conservation["footprint_status"]
                    .value_counts()
                    .items()
                },
                "placed_mass_total": float(
                    pd.to_numeric(
                        footprint_conservation["placed_mass"], errors="coerce"
                    ).fillna(0.0).sum()
                ),
                "ledger_control_mass_total": float(
                    pd.to_numeric(
                        footprint_conservation["ledger_control_mass"], errors="coerce"
                    ).fillna(0.0).sum()
                ),
            }
        summary["zero_target_denominator_audit_rows"] = int(len(zero_target_audit))
        summary["zero_target_denominator_audit_path"] = str(zero_target_audit_path)
        summary["allocation_component_denominator_audit_rows"] = int(len(component_audit))
        summary["allocation_component_denominator_audit_path"] = str(component_audit_path)
        summary["service_state_transfer_path"] = str(service_state_transfer_path)
        summary["service_state_transfer_rows"] = int(len(service_state_transfer))
        service_coverage_path = (
            paths.repo_root / "configs" / "service_wide_footprint_coverage.csv"
        )
        summary["service_wide_custom_allocation"] = {
            "enabled": bool(len(service_state_transfer)),
            "contract_version": "service_state_transfer_v1",
            "scope_config_path": "configs/service_wide_agency_scopes.csv",
            "allocation_weight_basis": "count_prior_x_within_bg_land_fraction",
            "footprint_coverage_path": "configs/service_wide_footprint_coverage.csv",
            "footprint_coverage_sha256": (
                hashlib.sha256(service_coverage_path.read_bytes()).hexdigest()
                if service_coverage_path.exists()
                else None
            ),
            "scope_ids": sorted(
                set(service_state_transfer["service_scope_id"].dropna().astype(str))
            ),
        }
        primary_policy_path = (
            paths.repo_root / "configs" / "primary_service_response_policies.csv"
        )
        primary_policies = _load_primary_service_response_policies(paths)
        summary["service_wide_custom_allocation"].update(
            primary_response_policy_path="configs/primary_service_response_policies.csv",
            primary_response_policy_sha256=(
                hashlib.sha256(primary_policy_path.read_bytes()).hexdigest()
                if primary_policy_path.exists()
                else None
            ),
            primary_response_scope_ids=sorted(
                set(primary_policies["service_scope_id"].astype(str))
            ),
        )
        resident_coverage_path = (
            paths.repo_root
            / "configs"
            / "overlap_custom_footprint_resident_coverage.csv"
        )
        summary["ordinary_resident_custom_allocation"] = {
            "allocation_weight_basis": (
                "count_prior_x_within_bg_responsibility_population_fraction"
            ),
            "footprint_coverage_path": (
                "configs/overlap_custom_footprint_resident_coverage.csv"
            ),
            "footprint_coverage_sha256": (
                hashlib.sha256(resident_coverage_path.read_bytes()).hexdigest()
                if resident_coverage_path.exists()
                else None
            ),
            "zero_model_signal_fallback": "declared_covered_population_weight_share",
        }
        summary["denominator_publishability_audit_path"] = str(publishability_audit_path)
        output_paths = {
            "block_group_ags_core": block_group_ags_core_out,
            "tract_ags_core": tract_ags_core_out,
            "block_group_fbi_calibrated": block_group_fbi_out,
            "tract_fbi_calibrated": tract_fbi_out,
            "zero_target_denominator_audit": zero_target_audit_path,
            "allocation_component_denominator_audit": component_audit_path,
            "allocation_service_state_transfer": service_state_transfer_path,
            "denominator_publishability_audit": publishability_audit_path,
            "city_posterior_diagnostics": (
                city_posterior_diagnostics_path
                if isinstance(city_posterior_diagnostics, pd.DataFrame)
                else None
            ),
            "composite_severity_sensitivity": (
                severity_sensitivity_path(manifest_path.parent, year=int(resolved_config.year))
                if summary.get("composite_severity_sensitivity_path")
                else None
            ),
            "unlocated_mass": unlocated_mass_out,
            "footprint_mass_conservation": (
                footprint_mass_conservation_path(
                    manifest_path.parent, year=int(resolved_config.year)
                )
                if summary.get("footprint_mass_conservation_path")
                else None
            ),
        }
        manifest = _allocation_build_manifest(
            paths=paths,
            config=resolved_config,
            summary=summary,
            output_paths=output_paths,
            run_metadata=run_metadata,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        summary["build_manifest_path"] = str(manifest_path)
        return summary
