from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.allocation import (
    CUSTOM_FOOTPRINT_STATUS_PLACED,
    FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR,
    FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR,
    HARM_WEIGHTS,
    POISSON_INTERVAL_ALPHA,
    _footprint_conservation_by_ori,
    footprint_mass_conservation_path,
)
from crimerisk.composites import (
    COMMON_DENOMINATOR_COLUMN,
    COUNT_FIRST_AGGREGATE_INDEX_FIELDS,
    EVENT_BURDEN_COLUMN,
    HARM_BURDEN_COLUMN,
    HARM_WEIGHTED_COUNT_COLUMN,
    LEGACY_AGGREGATE_INDEX_FIELDS,
    PERSONAL_BURDEN_COLUMN,
    PERSONAL_OFFENSES,
    PERSONAL_RELATIVE_SCORE_COLUMN,
    PROPERTY_BURDEN_COLUMN,
    PROPERTY_OFFENSES,
    PROPERTY_RELATIVE_SCORE_COLUMN,
    RENAMED_COMPOSITE_FIELDS,
    gated_component_offenses,
    load_severity_weights,
    primary_vector_id,
    severity_vector,
    severity_weights_path,
)
from crimerisk.build_freshness import (
    content_digest,
    dependency_digests,
    dependency_stamp_path,
    digest_multiset,
)
from crimerisk.controls import assert_level_lane_mass_ledger
from crimerisk.editions import GEOMETRY_COLUMN as EDITION_GEOMETRY_COLUMN
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
    source_lane_from_source,
)
from crimerisk.source_selection import (
    LANE_STANDING_ORDER,
    PREFERRED_LANE_COLUMNS,
    _agency_year_lane_signals,
    _rank_agency_year_lanes,
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
REPO_MURDER_TRACT_POSTERIOR_CALIBRATION = (
    REPO_ROOT / "state" / "modeling" / f"murder_tract_posterior_calibration_{YEAR}.json"
)
REPO_CITY_EXACT_POINT_QA = REPO_ROOT / "state" / "qa" / f"city_feed_exact_point_concentration_{YEAR}.csv"
REPO_CITY_EXACT_POINT_EXCEPTIONS = REPO_ROOT / "configs" / "city_feed_exact_point_exceptions.csv"
COUNTY_POP_2024_CSV = REPO_ROOT / "data" / "Census-PopEst-2020-2025" / "co-est2025-alldata.csv"
ACS_BG_SOURCE_PARQUET = REPO_ROOT / "data" / "ACS-5yr-2020-2024" / "parsed" / "acs_block_groups.parquet"
ACS_MISSING_BG_BACKFILL_CSV = REPO_ROOT / "configs" / "acs_missing_bg_decennial_backfill.csv"
CT_BG_2023_ZIP = REPO_ROOT / "data" / "tiger_bg" / "tl_2023_09_bg.zip"
CT_BG_2020_ZIP = REPO_ROOT / "data" / "tiger_bg" / "tl_2020_09_bg.zip"

# The evaluation this release is bound to. A release passes on its own held-out
# numbers against the published baselines, not on which workstream the measurement
# harness happens to recommend next: a recommendation is a research conclusion, and
# gating on one only asserts that the research has not changed its mind.
RELEASE_GOLD_RUN_ID = "gold_v53r2"
RELEASE_GOLD_DIR = REPO_ROOT / "state" / "eval" / RELEASE_GOLD_RUN_ID
RELEASE_GOLD_BUILD_YEAR = 2025
# The candidate the public surface is built from, as the frontend's own snapshot
# configuration names it. The release evaluation must be the evaluation of that
# candidate, so the two names are checked against each other rather than asserted.
RELEASE_CANDIDATE = "v53-2025"
RELEASE_SNAPSHOT_CONFIG = REPO_ROOT / "frontend" / "build" / "snapshot_config.env"
# Rape has one eligible spatial fold city, so its single-city comparison is reported
# and not gated. Every other offense must beat the population null on held-out TVD.
RELEASE_GOLD_UNGATED_OFFENSES = ("rape",)
RELEASE_GOLD_BASELINE_ARM = "population"
# Whether a spatial TVD loss to the baseline arm blocks promotion. 2025.1.1 republishes
# the 2025.1 surface unchanged and that surface loses on motor vehicle theft, so in this
# release the comparison is a reported result (`summary["status"] = "reported"`, the
# losses listed under `summary["failures"]`), not an issue. It becomes blocking with the
# next edition's model change. The presence and candidate-binding requirements block
# either way.
RELEASE_GOLD_TVD_BLOCKING = False


def _published_surface_path(output_dir: Path, *, geography: str, variant: str) -> Path:
    filename = f"crimerisk_{geography}_{YEAR}_{variant}.parquet"
    return output_dir / "diagnostics" / filename if variant == "cde_exact_sensitivity" else output_dir / filename


def _apply_target_year(year: int) -> None:
    """Rebind YEAR and the year-suffixed artifact-path globals for a non-default target year."""
    global YEAR, REPO_NEXT_PHASE_MEASUREMENT, REPO_DASHBOARD_LOOKUP, REPO_EXTERNAL_AVAILABILITY
    global REPO_BURGLARY_TAU_CALIBRATION, REPO_MURDER_TRACT_POSTERIOR_CALIBRATION
    global REPO_CITY_EXACT_POINT_QA
    global EXPECTED_RESIDUAL_FEATURE_POLICY_PATH_FRAGMENT
    YEAR = int(year)
    REPO_NEXT_PHASE_MEASUREMENT = REPO_ROOT / "state" / "modeling" / f"next_phase_measurement_summary_{YEAR}.json"
    REPO_DASHBOARD_LOOKUP = REPO_ROOT / "state" / "modeling" / f"dashboard_neighborhood_check_lookup_{YEAR}.json"
    REPO_EXTERNAL_AVAILABILITY = REPO_ROOT / "state" / "modeling" / f"external_surface_availability_{YEAR}.json"
    REPO_BURGLARY_TAU_CALIBRATION = REPO_ROOT / "state" / "modeling" / f"burglary_tau_calibration_{YEAR}.json"
    REPO_MURDER_TRACT_POSTERIOR_CALIBRATION = (
        REPO_ROOT / "state" / "modeling" / f"murder_tract_posterior_calibration_{YEAR}.json"
    )
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
AGGREGATE_INDEX_FIELDS = list(LEGACY_AGGREGATE_INDEX_FIELDS)
# v2 count-first composite lane (docs/archive/2026-09/analysis_scratch/final_phase/COMPOSITES_CONTRACT.md). A surface
# carries exactly one of the two field sets; the mirror below asserts that rather than assuming it,
# because a half-migrated frame would publish two composites of the same thing under two rules.
COUNT_FIRST_INDEX_FIELDS = list(COUNT_FIRST_AGGREGATE_INDEX_FIELDS)
ALL_COMPOSITE_INDEX_FIELDS = [*AGGREGATE_INDEX_FIELDS, *COUNT_FIRST_INDEX_FIELDS]
# Index field -> the published count it normalises, for the spatial diagnostics. The substring
# rules below cover the legacy names; the count-first names are explicit.
COMPOSITE_INDEX_COUNT_COLUMNS = {
    EVENT_BURDEN_COLUMN: "expected_count_total",
    PERSONAL_BURDEN_COLUMN: "expected_count_personal",
    PROPERTY_BURDEN_COLUMN: "expected_count_property",
    HARM_BURDEN_COLUMN: HARM_WEIGHTED_COUNT_COLUMN,
    "multi_offense_relative_score_event_weighted": "expected_count_total",
    "multi_offense_relative_score_equal_offense": "expected_count_total",
    PERSONAL_RELATIVE_SCORE_COLUMN: "expected_count_personal",
    PROPERTY_RELATIVE_SCORE_COLUMN: "expected_count_property",
}


def _count_first_composite_lane(df: pd.DataFrame) -> bool:
    """Which composite lane a surface was built on, read off the surface itself."""
    return EVENT_BURDEN_COLUMN in df.columns


def _special_use_taxonomy_lane(df: pd.DataFrame) -> bool:
    """Whether the surface was built with the typed special-use taxonomy, read off the surface."""
    return SPECIAL_USE_TYPE_COLUMN in df.columns


# v2 uncertainty layer. Every constant below is a TRANSCRIPTION, not an import: the mirror's job
# is to disagree with the producer when the producer is wrong, which it cannot do if it shares the
# producer's numbers. `analysis_scratch/final_phase/UNCERTAINTY_LAYER_CONTRACT.md` is the source.
UNCERTAINTY_INDEX_BREAKS = [
    3.125,
    6.25,
    12.5,
    25.0,
    50.0,
    75.0,
    100.0,
    133.0,
    200.0,
    400.0,
    800.0,
    1600.0,
    3200.0,
    6400.0,
    12800.0,
]
UNCERTAINTY_INDEX_BIN_LABELS = [
    "<3.125",
    "3.125-6.25",
    "6.25-12.5",
    "12.5-25",
    "25-50",
    "50-75",
    "75-100",
    "100-133",
    "133-200",
    "200-400",
    "400-800",
    "800-1600",
    "1600-3200",
    "3200-6400",
    "6400-12800",
    ">=12800",
]
UNCERTAINTY_TIER_HIGH_MIN = 0.80
UNCERTAINTY_TIER_MEDIUM_MIN = 0.60
UNCERTAINTY_TIER_BENCHMARK_SHARE_MAX = 0.50
UNCERTAINTY_DOMAIN_LOW_CUTOFF = 1.0 / 3.0
UNCERTAINTY_QUANTILE_ORDER = ["p10", "p25", "p50", "p75", "p90"]


def _uncertainty_layer_lane(df: pd.DataFrame) -> bool:
    """Whether the surface carries the uncertainty layer, read off the surface itself."""
    return "uncertainty_layer_version" in df.columns


def _uncertainty_expected_bin(index_point: pd.Series) -> pd.Series:
    series = pd.to_numeric(index_point, errors="coerce")
    values = series.to_numpy(dtype=float)
    positions = np.searchsorted(np.asarray(UNCERTAINTY_INDEX_BREAKS, dtype=float), values, side="right")
    finite = np.isfinite(values)
    labels = np.full(len(values), None, dtype=object)
    labels[finite] = [UNCERTAINTY_INDEX_BIN_LABELS[int(position)] for position in positions[finite]]
    return pd.Series(labels, index=series.index, dtype="string")


def _uncertainty_expected_tier(df: pd.DataFrame, offense: str) -> pd.Series:
    def numeric(column: str, default: float) -> pd.Series:
        if column not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[column], errors="coerce").fillna(default)

    def boolean(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        return df[column].astype("boolean").fillna(False).astype(bool)

    probability = pd.to_numeric(df.get(f"prob_displayed_index_bin_{offense}"), errors="coerce")
    published = boolean(f"primary_index_publishable_{offense}")
    point = pd.to_numeric(df.get(f"index_{offense}_primary"), errors="coerce")
    withheld = ~published | point.isna() | probability.isna()
    forced_low = (
        numeric(f"benchmark_imputed_share_{offense}", 0.0).gt(UNCERTAINTY_TIER_BENCHMARK_SHARE_MAX)
        | boolean(f"footprint_ambient_exposure_missing_{offense}")
        | boolean(f"unresolved_level_flag_{offense}")
        | numeric(f"domain_overlap_score_{offense}", 1.0).lt(UNCERTAINTY_DOMAIN_LOW_CUTOFF)
    )
    tier = pd.Series("low", index=df.index, dtype="string")
    tier.loc[~withheld & ~forced_low & probability.ge(UNCERTAINTY_TIER_MEDIUM_MIN)] = "medium"
    tier.loc[~withheld & ~forced_low & probability.ge(UNCERTAINTY_TIER_HIGH_MIN)] = "high"
    tier.loc[withheld] = "withheld"
    return tier


def _uncertainty_layer_issues(
    df: pd.DataFrame, *, label: str, contract: "ReleaseContract | None" = None
) -> list[str]:
    issues: list[str] = []
    rare_nulled = "block_group" in label
    if contract is not None:
        # The manifest's own account of the layer, checked against this file's transcription
        # rather than trusted in place of it -- a quoted quantile is only re-derivable if the
        # break set and the vintage are the ones the mirror scored.
        if contract.uncertainty_version != UNCERTAINTY_LAYER_VERSION:
            issues.append(
                f"{label}: manifest uncertainty version is {contract.uncertainty_version!r}, "
                f"expected {UNCERTAINTY_LAYER_VERSION!r}"
            )
        if list(contract.uncertainty_index_breaks) != list(UNCERTAINTY_INDEX_BREAKS):
            issues.append(
                f"{label}: manifest uncertainty index_breaks are "
                f"{list(contract.uncertainty_index_breaks)}, expected {UNCERTAINTY_INDEX_BREAKS}"
            )
    version = df.get("uncertainty_layer_version")
    if version is None:
        issues.append(f"{label}: uncertainty_layer_version is missing from an uncertainty-layer surface")
    else:
        observed = sorted(set(version.astype("string").dropna().unique().tolist()))
        if observed != [UNCERTAINTY_LAYER_VERSION] or int(version.isna().sum()):
            issues.append(
                f"{label}: uncertainty_layer_version is {observed}, expected "
                f"[{UNCERTAINTY_LAYER_VERSION!r}] on every row"
            )
    for offense in OFFENSES_7:
        support_column = f"uncertainty_support_class_{offense}"
        if support_column not in df.columns:
            issues.append(f"{label}: {support_column} is missing from an uncertainty-layer surface")
        else:
            support = df[support_column].astype("string")
            unexpected = sorted(set(support.dropna().unique().tolist()) - UNCERTAINTY_SUPPORT_CLASSES)
            if unexpected:
                issues.append(f"{label}: {support_column} has unexpected values {unexpected}")
            if int(support.isna().sum()):
                issues.append(f"{label}: {support_column} is null on {int(support.isna().sum())} rows")
        for prefix in UNCERTAINTY_DECOMPOSITION_COLUMN_PREFIXES:
            column = f"{prefix}{offense}"
            if column not in df.columns:
                issues.append(f"{label}: {column} is missing from an uncertainty-layer surface")
                continue
            # The decomposition is published so a reader can see which component carries the
            # width. A negative or non-finite log sd is not a component, it is a bug. Nulls are
            # not: a cell with no allocated mass for the offense has no control or share
            # dispersion, and the contract's rule is null rather than a manufactured zero.
            values = pd.to_numeric(df[column], errors="coerce")
            finite = values.to_numpy(dtype=float)
            if bool(values.lt(0.0).any()) or bool(np.isinf(finite).any()):
                issues.append(f"{label}: {column} carries negative or non-finite log standard deviations")
        # The denominator term is the measured disagreement of two public person proxies, defined
        # on every cell. The two offenses with a single denominator construction publish a zero
        # rather than a null, so the component is visibly absent instead of silently assumed.
        denominator_column = f"uncertainty_denominator_log_sd_{offense}"
        if denominator_column in df.columns:
            denominator_sd = pd.to_numeric(df[denominator_column], errors="coerce")
            if int(denominator_sd.isna().sum()):
                issues.append(
                    f"{label}: {denominator_column} is null on {int(denominator_sd.isna().sum())} "
                    "rows; the denominator component is defined on every cell"
                )
            if offense in ("burglary", "motor_vehicle_theft") and bool(denominator_sd.gt(0.0).any()):
                issues.append(
                    f"{label}: {denominator_column} is positive on "
                    f"{int(denominator_sd.gt(0.0).sum())} rows; an offense with one denominator "
                    "construction publishes a zero denominator component"
                )
        tier_column = f"decision_reliability_tier_{offense}"
        if tier_column in df.columns:
            tiers = df[tier_column].astype("string")
            unexpected_tiers = sorted(set(tiers.dropna().unique().tolist()) - UNCERTAINTY_TIER_VOCABULARY)
            if unexpected_tiers:
                issues.append(f"{label}: {tier_column} has unexpected values {unexpected_tiers}")
        # The count quantiles are NOT withdrawn for the rare offenses at block group: the count is
        # conserved and already published there. Only the rate/index layer follows the point.
        if rare_nulled and offense in SPARSE_BASELINE_TRANSFER_OFFENSES:
            empty_counts = [
                f"expected_count_{offense}_{quantile}"
                for quantile in UNCERTAINTY_QUANTILE_SUFFIXES["expected_count"]
                if f"expected_count_{offense}_{quantile}" in df.columns
                and not pd.to_numeric(df[f"expected_count_{offense}_{quantile}"], errors="coerce").notna().any()
            ]
            if empty_counts:
                issues.append(
                    f"{label}: rare-offense block-group count quantiles are entirely null for "
                    f"{offense}: {sorted(empty_counts)}; the count is conserved and published here"
                )
            populated_labels = [
                column
                for column in (f"displayed_index_bin_{offense}", tier_column)
                if column in df.columns and df[column].astype("string").notna().any()
            ]
            if populated_labels:
                issues.append(
                    f"{label}: rare-offense block-group support leaves uncertainty label fields "
                    f"populated for {offense}: {sorted(populated_labels)}"
                )
    for offense in OFFENSES_7:
        bin_column = f"displayed_index_bin_{offense}"
        tier_column = f"decision_reliability_tier_{offense}"
        probability_column = f"prob_displayed_index_bin_{offense}"
        above_column = f"prob_index_{offense}_above_100"
        for column in (bin_column, tier_column, probability_column, above_column):
            if column not in df.columns:
                issues.append(f"{label}: {column} is missing from an uncertainty-layer surface")
        if tier_column not in df.columns:
            continue

        published = (
            df[f"primary_index_publishable_{offense}"].fillna(False).astype(bool)
            if f"primary_index_publishable_{offense}" in df.columns
            else pd.Series(False, index=df.index)
        )
        # Murder and rape carry no per-offense point at block group, so their rate/index quantiles,
        # bin and tier are null there by the same publication policy as the point itself.
        if rare_nulled and offense in SPARSE_BASELINE_TRANSFER_OFFENSES:
            populated = [
                column
                for column in (
                    *[f"index_{offense}_primary_{q}" for q in UNCERTAINTY_QUANTILE_ORDER],
                    probability_column,
                    above_column,
                )
                if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()
            ]
            if populated:
                issues.append(
                    f"{label}: rare-offense block-group support leaves uncertainty point fields "
                    f"populated for {offense}: {sorted(populated)}"
                )
            continue

        for column in (probability_column, above_column):
            values = pd.to_numeric(df[column], errors="coerce")
            outside = int(((values < 0.0) | (values > 1.0)).sum())
            if outside:
                issues.append(f"{label}: {column} is outside [0, 1] on {outside} rows")
            orphaned = int((values.notna() & ~published).sum())
            if orphaned:
                issues.append(
                    f"{label}: {column} is populated on {orphaned} rows that publish no index"
                )

        expected_bin = _uncertainty_expected_bin(df.get(f"index_{offense}_primary")).where(published)
        published_bin = df[bin_column].astype("string")
        mismatch = int((published_bin.fillna("") != expected_bin.fillna("")).sum())
        if mismatch:
            issues.append(
                f"{label}: {bin_column} does not match the published break set on {mismatch} rows"
            )

        expected_tier = _uncertainty_expected_tier(df, offense)
        tier_mismatch = int((df[tier_column].astype("string").fillna("") != expected_tier.fillna("")).sum())
        if tier_mismatch:
            issues.append(
                f"{label}: {tier_column} does not match the decision-stability rule on "
                f"{tier_mismatch} rows"
            )

        for prefix in (f"expected_count_{offense}", f"rate_{offense}_primary", f"index_{offense}_primary"):
            present = [
                f"{prefix}_{quantile}"
                for quantile in UNCERTAINTY_QUANTILE_ORDER
                if f"{prefix}_{quantile}" in df.columns
            ]
            for lower, upper in zip(present, present[1:]):
                low = pd.to_numeric(df[lower], errors="coerce")
                high = pd.to_numeric(df[upper], errors="coerce")
                inverted = int((low > high + 1e-9).sum())
                if inverted:
                    issues.append(
                        f"{label}: {lower} exceeds {upper} on {inverted} rows; a quantile ladder "
                        "must be monotone"
                    )
    return issues


# --- the release contract ---------------------------------------------------------------------
# Which assertion set a surface is held to is a property of the BUILD, not of this file. Every v2
# lane records itself in `resolved_config` of the build's own manifest, so the gate reads the
# contract the artifact was produced under and holds it to that contract's identities. A manifest
# with no v2 block -- every promoted build to date -- yields the legacy contract, and the legacy
# path is bit-identical to the assertions this file made before the lanes existed.
#
# The lanes are never INFERRED from the columns. The surface's own column evidence is cross-checked
# against the manifest instead, so a half-migrated frame -- a lane's columns without its manifest
# block, or a manifest block whose columns never landed -- fails the release rather than quietly
# selecting the assertion set that happens to pass.
V2_LANE_KEYS: tuple[str, ...] = (
    "mixture_allocation",
    "exposure_ensemble",
    "count_first_composites",
    "special_use_taxonomy",
    "uncertainty_layer",
    "soft_shrinkage",
    "unlocated_mass",
)

# Amendment 3 item 5, mirrored. Transcribed from
# `docs/archive/2026-09/analysis_scratch/final_phase/UNLOCATED_MASS_CONTRACT.md` rather than imported, for the same
# reason every other constant in this block is transcribed.
UNLOCATED_MASS_IDENTITY = (
    "published_block_group_sum + unlocated + service_exports - "
    "service_imports == source_state_control"
)
SERVICE_TRANSFER_MASS_IDENTITY = (
    "published_block_group_sum + service_exports - service_imports "
    "== source_state_control"
)
LEGACY_MASS_IDENTITY = "published_block_group_sum == control"
UNLOCATED_MASS_JURISDICTION_TYPE = "statewide_overlap_layer"
UNLOCATED_MASS_ROUTES: tuple[str, ...] = (
    "footprint_no_placeable_support",
    "footprint_suppressed_duplicate",
    "registry_exclude_or_hold",
    "service_no_eligible_receiver",
    "unresolved_no_local_target",
    "unresolved_unattributed_state_residual",
    "unresolved_unsupported_county",
)
UNLOCATED_MASS_COLUMNS: tuple[str, ...] = (
    "state_fips",
    "jurisdiction_id",
    "jurisdiction_type",
    "offense",
    "unlocated_count",
    "control_count",
    "unlocated_share_of_control",
    *(f"unlocated_count_{route}" for route in UNLOCATED_MASS_ROUTES),
)


# Amendment 3 item 6, mirrored. Transcribed from
# `docs/archive/2026-09/analysis_scratch/final_phase/COVERAGE_SCENARIOS_CONTRACT.md`, not imported from the producer,
# for the same reason every other constant in this block is transcribed.
COVERAGE_SCENARIOS_DIRNAME = "scenarios"
COVERAGE_SCENARIO_ALL_KEY = "ALL"
COVERAGE_SCENARIO_COLUMNS: tuple[str, ...] = (
    "state_fips",
    "jurisdiction_id",
    "offense",
    "count_published",
    "count_observed",
    "count_central",
    "count_upper",
    "benchmark_imputed_count",
    "benchmark_imputed_municipal",
    "benchmark_imputed_county",
    "benchmark_imputed_central",
    "benchmark_imputed_upper",
)
COVERAGE_SCENARIO_STATE_DELTA_COLUMNS: tuple[str, ...] = (
    "state_fips",
    "offense",
    "count_published",
    "count_observed",
    "count_central",
    "count_upper",
    "benchmark_imputed_count",
    "coverage_supplied_count",
    "scenario_dependent_count",
    "imputed_share_of_published",
    "coverage_supplied_share_of_published",
    "scenario_dependent_share_of_published",
    "central_minus_published",
    "central_minus_published_share",
)


def _unlocated_mass_path(output_dir: Path) -> Path:
    return output_dir / f"unlocated_mass_{YEAR}.parquet"


def _unlocated_mass_table(output_dir: Path) -> pd.DataFrame | None:
    """The companion table, or None when it is absent. Read once, used by two checks."""
    path = _unlocated_mass_path(output_dir)
    if not path.exists():
        return None
    return pd.read_parquet(path)

# PLAN.md item 5, mirrored. Transcribed from
# `docs/archive/2026-09/analysis_scratch/final_phase/SOFT_SHRINKAGE_CONTRACT.md`, not imported, for the same reason
# every other constant in this block is transcribed: a mirror that shares the producer's numbers
# cannot disagree with the producer.
SOFT_SHRINKAGE_ENVELOPE_MODE = "soft_shrinkage"
SOFT_SHRINKAGE_NU = 2.0
SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT = 0.75
SOFT_SHRINKAGE_MODE_COLUMN = "allocation_envelope_mode"
SOFT_SHRINKAGE_RATIO_COLUMN = "allocation_envelope_rate_ratio"
SOFT_SHRINKAGE_RETAINED_COLUMN = "allocation_envelope_retained_rate_ratio"
SOFT_SHRINKAGE_AUDIT_COLUMNS: tuple[str, ...] = (
    SOFT_SHRINKAGE_MODE_COLUMN,
    SOFT_SHRINKAGE_RATIO_COLUMN,
    SOFT_SHRINKAGE_RETAINED_COLUMN,
)
# The two hard envelopes the lane replaces, and the caps the model-only one used.
MODEL_ONLY_RATE_RATIO_CAPS = {"robbery": 22.0, "murder": 57.0}

# v2 exposure-denominator ensembles, mirrored. Every constant below is a TRANSCRIPTION of
# `analysis_scratch/final_phase/EXPOSURE_ENSEMBLE_CONTRACT.md`, not an import: the mirror's job is
# to disagree with the producer when the producer is wrong, and it cannot do that if it shares the
# producer's numbers. The manifest's own copies are checked AGAINST these, not trusted in place of
# them.
EXPOSURE_NORMALIZER_VERSION = "exposure_ensemble_v1"
CENSUS_RESIDENTIAL_NORMALIZER_VERSION = "exposure_ensemble_v2_census_residential"
LANDSCAN_NIGHT_RESIDENTIAL_SOURCE = "landscan_night"
CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE = "census_release_population"
EXPOSURE_NORMALIZER_SEMANTICS = "opportunity_normalized_intensity_not_person_time_risk"
EXPOSURE_NORMALIZER_ID_BY_OFFENSE = {
    "murder": "person_ens_v1",
    "rape": "person_ens_v1",
    "robbery": "person_ens_v1",
    "aggravated_assault": "person_ens_v1",
    "larceny": "larceny_opp_v1",
    # E3's verdicts: the deployed premises and vehicle denominators win, so this lane names them
    # and leaves them alone. A build that moved either one would fail the identity below.
    "burglary": "premises_nnls_v1",
    "motor_vehicle_theft": "vehicle_exposure_v1",
}
# The five offenses whose published denominator the lane actually replaces.
EXPOSURE_ENSEMBLE_OFFENSES: tuple[str, ...] = (
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "larceny",
)
EXPOSURE_NORMALIZER_COLUMN_PREFIX = "opportunity_normalizer_"
EXPOSURE_NORMALIZER_SEMANTICS_COLUMN = "opportunity_normalizer_semantics"
# `person_ens_v1`'s three legs in E3's own order, and the larceny opportunity hybrid's parts.
EXPOSURE_PERSON_LEG_COLUMNS = {
    "landscan_night": "landscan_night_leg",
    "landscan_day": "landscan_day_leg",
    "daytime_jobs": "daytime_jobs_leg",
}
EXPOSURE_PERSON_ENSEMBLE_WEIGHTS: dict[str, dict[str, float]] = {
    "murder": {"landscan_night": 0.9, "landscan_day": 0.0, "daytime_jobs": 0.1},
    "rape": {"landscan_night": 0.7, "landscan_day": 0.1, "daytime_jobs": 0.2},
    "robbery": {"landscan_night": 0.7, "landscan_day": 0.3, "daytime_jobs": 0.0},
    "aggravated_assault": {"landscan_night": 0.7, "landscan_day": 0.1, "daytime_jobs": 0.2},
    "larceny": {"landscan_night": 0.4, "landscan_day": 0.3, "daytime_jobs": 0.3},
}
EXPOSURE_LARCENY_HYBRID_WEIGHTS: dict[str, float] = {
    "person_ensemble": 0.5,
    "destination_poi": 0.3,
    "retail_jobs": 0.2,
    # Tested and rejected by E3; carried at zero so the rejection is on the record and a build
    # that quietly gave the vehicle leg weight would fail here.
    "vehicles": 0.0,
}
EXPOSURE_LARCENY_HYBRID_SOURCE_COLUMNS = {
    "destination_poi": "destination_poi_total",
    "retail_jobs": "lodes_retail_jobs",
    "vehicles": "vehicle_exposure_2024",
}
EXPOSURE_WEIGHTS_CSV = "configs/exposure_ensemble_weights_v1.csv"
EXPOSURE_SIMPLEX_TOLERANCE = 1e-12
# The reference-total invariant: each normalizer totals the published universe's resident
# population. A single global scalar cannot move a share, a rank or an index, so this is a level
# assertion and it is exact to floating point.
EXPOSURE_REFERENCE_TOTAL_RELATIVE_TOLERANCE = 1e-9
# Recomposition from the artifact's own legs at the frozen weights. Larceny's hybrid mixes three
# rescaled parts, so its worst cell lands ~5e-16 relative away from the producer's summation order.
EXPOSURE_RECOMPOSITION_RELATIVE_TOLERANCE = 1e-12

UNCERTAINTY_LAYER_VERSION = "uncertainty_layer_v1"
UNCERTAINTY_SUPPORT_CLASSES = frozenset({"direct", "model", "benchmark"})
UNCERTAINTY_TIER_VOCABULARY = frozenset({"high", "medium", "low", "withheld"})
UNCERTAINTY_QUANTILE_SUFFIXES = {
    "expected_count": ("p10", "p50", "p90"),
    "rate_primary": ("p10", "p50", "p90"),
    # The p25/p75 pair is the 50% interval the calibration protocol scores.
    "index_primary": ("p10", "p25", "p50", "p75", "p90"),
}
UNCERTAINTY_DECOMPOSITION_COLUMN_PREFIXES = (
    "uncertainty_control_log_sd_",
    "uncertainty_share_log_sd_",
    "uncertainty_denominator_log_sd_",
)

SPECIAL_USE_TAXONOMY_VERSION = "special_use_taxonomy_v2"
SERVICE_STATE_TRANSFER_COLUMNS: tuple[str, ...] = (
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
SERVICE_STATE_TRANSFER_KEY: tuple[str, ...] = (
    "year",
    "service_scope_id",
    "canonical_target_ori",
    "offense",
    "source_state_fips",
    "allocation_state_fips",
)
COUNT_FIRST_COMPOSITES_VERSION = "count_first_composites_v1"


def _service_state_transfer_path(output_dir: Path) -> Path:
    return output_dir / f"allocation_service_state_transfer_{YEAR}.parquet"

def _check_service_state_transfer_table(
    *,
    ledger: pd.DataFrame,
    component_audit: pd.DataFrame,
    unlocated_mass: pd.DataFrame | None = None,
    issues: list[str],
    label: str = "release_contract.service_state_transfer",
) -> dict[str, Any]:
    """Validate service-wide source-to-geographic-state mass without trusting the producer.

    The ledger is a source-control statement while ``state_fips`` on component rows is geographic.
    A valid table has one row per source/destination allocation, reconstructs every source target,
    and independently matches the final service components placed in each destination state.
    """
    summary: dict[str, Any] = {"rows": int(len(ledger)), "ok": True}
    missing = sorted(set(SERVICE_STATE_TRANSFER_COLUMNS) - set(ledger.columns))
    if missing:
        issues.append(f"{label}: ledger is missing columns {missing}")
        summary.update(ok=False, missing_columns=missing)
        return summary

    work = ledger[list(SERVICE_STATE_TRANSFER_COLUMNS)].copy()
    required_text = [
        "service_scope_id",
        "canonical_target_ori",
        "offense",
        "source_state_fips",
    ]
    blank = pd.Series(False, index=work.index)
    for column in required_text:
        blank |= work[column].isna() | work[column].astype("string").str.strip().eq(
            ""
        ).fillna(True)
    if bool(blank.any()):
        issues.append(
            f"{label}: ledger has {int(blank.sum())} rows with missing identity fields"
        )

    year = pd.to_numeric(work["year"], errors="coerce")
    bad_year = year.isna() | year.ne(int(YEAR))
    if bool(bad_year.any()):
        issues.append(
            f"{label}: ledger has {int(bad_year.sum())} rows outside build year {YEAR}"
        )
    work["source_state_fips"] = work["source_state_fips"].astype("string").str.zfill(2)
    allocation_state = work["allocation_state_fips"].astype("string").str.strip()
    terminal = allocation_state.isna() | allocation_state.eq("")
    work["allocation_state_fips"] = allocation_state.where(~terminal).str.zfill(2)
    route_reason = work["route_reason"].astype("string").fillna("").str.strip()
    invalid_terminal_route = terminal & route_reason.ne("service_no_eligible_receiver")
    invalid_located_route = ~terminal & route_reason.ne("")
    if bool(invalid_terminal_route.any() | invalid_located_route.any()):
        issues.append(
            f"{label}: route_reason must be blank for located rows and "
            "service_no_eligible_receiver for terminal unlocated rows"
        )

    duplicate = work.duplicated(subset=list(SERVICE_STATE_TRANSFER_KEY), keep=False)
    if bool(duplicate.any()):
        issues.append(
            f"{label}: ledger has {int(duplicate.sum())} duplicate allocation-key rows"
        )

    numeric_columns = ["source_target_count", "allocation_count", "allocation_share"]
    numeric = {
        column: pd.to_numeric(work[column], errors="coerce")
        for column in numeric_columns
    }
    invalid_numeric = pd.Series(False, index=work.index)
    for values in numeric.values():
        invalid_numeric |= values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
    if bool(invalid_numeric.any()):
        issues.append(
            f"{label}: ledger has {int(invalid_numeric.sum())} rows with non-finite counts or shares"
        )
    negative = (
        numeric["source_target_count"].lt(-TOTAL_LANE_TOLERANCE)
        | numeric["allocation_count"].lt(-TOTAL_LANE_TOLERANCE)
        | numeric["allocation_share"].lt(-TOTAL_LANE_TOLERANCE)
    )
    if bool(negative.any()):
        issues.append(
            f"{label}: ledger has {int(negative.sum())} rows with negative counts or shares"
        )

    source_key = [
        "year",
        "service_scope_id",
        "canonical_target_ori",
        "offense",
        "source_state_fips",
    ]
    target_cardinality = work.groupby(source_key, dropna=False)[
        "source_target_count"
    ].nunique(dropna=False)
    inconsistent_target = target_cardinality.ne(1)
    if bool(inconsistent_target.any()):
        issues.append(
            f"{label}: {int(inconsistent_target.sum())} source targets disagree across destination rows"
        )
    source = (
        work.assign(
            source_target_count=numeric["source_target_count"],
            allocation_count=numeric["allocation_count"],
            allocation_share=numeric["allocation_share"],
        )
        .groupby(source_key, dropna=False)
        .agg(
            source_target_count=("source_target_count", "first"),
            allocation_count=("allocation_count", "sum"),
            allocation_share=("allocation_share", "sum"),
        )
        .reset_index()
    )
    source["count_delta"] = source["allocation_count"] - source["source_target_count"]
    expected_share = (
        source["source_target_count"].gt(TOTAL_LANE_TOLERANCE).astype(float)
    )
    source["share_delta"] = source["allocation_share"] - expected_share
    count_bad = source["count_delta"].abs().gt(TOTAL_LANE_TOLERANCE)
    share_bad = source["share_delta"].abs().gt(TOTAL_LANE_TOLERANCE)
    if bool(count_bad.any()):
        issues.append(
            f"{label}: {int(count_bad.sum())} source targets are not reconstructed by allocations "
            f"(max abs delta {_max_abs(source['count_delta']):.6g})"
        )
    if bool(share_bad.any()):
        issues.append(
            f"{label}: {int(share_bad.sum())} source targets have allocation shares that do not sum "
            f"to one (or zero for a zero target)"
        )

    audit_required = {
        "state_fips",
        "offense",
        "service_scope_id",
        "canonical_target_ori",
        "source_state_fips",
        "source_target_count",
        "component_count_after",
        "jurisdiction_type",
    }
    audit_missing = sorted(audit_required - set(component_audit.columns))
    component_bad = pd.DataFrame()
    if audit_missing:
        issues.append(f"{label}: component audit is missing columns {audit_missing}")
    else:
        components = component_audit.copy()
        service = components[components["service_scope_id"].notna()].copy()
        service = service[
            service["service_scope_id"].astype("string").str.strip().ne("")
        ]
        wrong_type = service["jurisdiction_type"].astype("string").ne(
            "custom_footprint_overlap_layer"
        )
        if bool(wrong_type.any()):
            issues.append(
                f"{label}: {int(wrong_type.sum())} service rows have the wrong jurisdiction type"
            )
        service["source_state_fips"] = (
            service["source_state_fips"].astype("string").str.zfill(2)
        )
        service["allocation_state_fips"] = (
            service["state_fips"].astype("string").str.zfill(2)
        )
        service["component_count_after"] = pd.to_numeric(
            service["component_count_after"], errors="coerce"
        )
        service["source_target_count"] = pd.to_numeric(
            service["source_target_count"], errors="coerce"
        )
        component_invalid = service["component_count_after"].isna() | ~np.isfinite(
            service["component_count_after"].to_numpy(dtype=float)
        )
        if bool(component_invalid.any()):
            issues.append(
                f"{label}: component audit has {int(component_invalid.sum())} non-finite service counts"
            )
        component_negative = service["component_count_after"].lt(
            -TOTAL_LANE_TOLERANCE
        )
        if bool(component_negative.any()):
            issues.append(
                f"{label}: component audit has {int(component_negative.sum())} negative service counts"
            )
        audit_source = (
            service.groupby(source_key[1:], dropna=False)["source_target_count"]
            .agg(["first", "nunique"])
            .reset_index()
            .rename(columns={"first": "audit_source_target_count"})
        )
        audit_source_bad = audit_source["nunique"].ne(1)
        if bool(audit_source_bad.any()):
            issues.append(
                f"{label}: {int(audit_source_bad.sum())} component-audit source targets disagree "
                "across geographic rows"
            )
        located_source = work.loc[~terminal, source_key].drop_duplicates()
        audit_expected = source.merge(located_source, on=source_key, how="inner")
        source_match = audit_expected.merge(
            audit_source, on=source_key[1:], how="outer"
        )
        source_match["delta"] = pd.to_numeric(
            source_match["source_target_count"], errors="coerce"
        ).fillna(0.0) - pd.to_numeric(
            source_match["audit_source_target_count"], errors="coerce"
        ).fillna(0.0)
        audit_target_bad = source_match["delta"].abs().gt(TOTAL_LANE_TOLERANCE)
        if bool(audit_target_bad.any()):
            issues.append(
                f"{label}: {int(audit_target_bad.sum())} ledger source targets do not match the "
                f"component audit (max abs delta {_max_abs(source_match['delta']):.6g})"
            )
        component_key = [
            "service_scope_id",
            "canonical_target_ori",
            "offense",
            "source_state_fips",
            "allocation_state_fips",
        ]
        component_sums = (
            service.groupby(component_key, dropna=False)["component_count_after"]
            .sum()
            .reset_index()
        )
        ledger_sums = (
            work.loc[~terminal].groupby(component_key, dropna=False)["allocation_count"]
            .sum()
            .reset_index()
        )
        component_match = ledger_sums.merge(
            component_sums,
            on=component_key,
            how="outer",
            validate="one_to_one",
        )
        component_match["delta"] = pd.to_numeric(
            component_match["component_count_after"], errors="coerce"
        ).fillna(0.0) - pd.to_numeric(
            component_match["allocation_count"], errors="coerce"
        ).fillna(0.0)
        component_bad = component_match[
            component_match["delta"].abs().gt(TOTAL_LANE_TOLERANCE)
        ]
        if not component_bad.empty:
            issues.append(
                f"{label}: {len(component_bad)} destination allocations do not match final service "
                f"components (max abs delta {_max_abs(component_match['delta']):.6g})"
            )

    terminal_bad = pd.DataFrame()
    if bool(terminal.any()):
        unlocated_required = {
            "source_state_fips",
            "service_scope_id",
            "canonical_target_ori",
            "offense",
            "unlocated_count",
            "route_reason",
        }
        unlocated_missing = sorted(
            unlocated_required - set(unlocated_mass.columns if unlocated_mass is not None else [])
        )
        if unlocated_missing:
            issues.append(
                f"{label}: terminal allocations require unlocated columns {unlocated_missing}"
            )
        else:
            terminal_key = source_key[1:] + ["route_reason"]
            terminal_sums = (
                work.loc[terminal]
                .groupby(terminal_key, dropna=False)["allocation_count"]
                .sum()
                .reset_index()
            )
            unlocated = unlocated_mass.copy()
            unlocated["source_state_fips"] = (
                unlocated["source_state_fips"].astype("string").str.zfill(2)
            )
            unlocated_sums = (
                unlocated.loc[
                    unlocated["route_reason"].astype("string").eq(
                        "service_no_eligible_receiver"
                    )
                ]
                .groupby(terminal_key, dropna=False)["unlocated_count"]
                .sum()
                .reset_index()
            )
            terminal_match = terminal_sums.merge(
                unlocated_sums, on=terminal_key, how="outer", validate="one_to_one"
            )
            terminal_match["delta"] = pd.to_numeric(
                terminal_match["allocation_count"], errors="coerce"
            ).fillna(0.0) - pd.to_numeric(
                terminal_match["unlocated_count"], errors="coerce"
            ).fillna(0.0)
            terminal_bad = terminal_match[
                terminal_match["delta"].abs().gt(TOTAL_LANE_TOLERANCE)
            ]
            if not terminal_bad.empty:
                issues.append(
                    f"{label}: {len(terminal_bad)} terminal allocations do not match explicit "
                    "unlocated mass"
                )

    summary.update(
        ok=not any(issue.startswith(label) for issue in issues),
        source_target_rows=int(len(source)),
        cross_state_rows=int(
            (~terminal & work["source_state_fips"].ne(work["allocation_state_fips"])).sum()
        ),
        terminal_unlocated_rows=int(terminal.sum()),
        terminal_unlocated_bad_rows=int(len(terminal_bad)),
        national_source_target_total=float(source["source_target_count"].sum()),
        national_allocation_total=float(numeric["allocation_count"].sum()),
        max_abs_source_count_delta=_max_abs(source["count_delta"]),
        max_abs_source_share_delta=_max_abs(source["share_delta"]),
        component_destination_bad_rows=int(len(component_bad)),
    )
    return summary

def _check_service_state_transfer(
    *, contract: ReleaseContract, state_output_dir: Path, issues: list[str]
) -> dict[str, Any]:
    """Require the transfer ledger exactly when the build declares service-wide allocation."""
    label = "release_contract.service_state_transfer"
    path = _service_state_transfer_path(state_output_dir)
    declared = contract.service_wide_custom_allocation_enabled
    summary: dict[str, Any] = {
        "declared": declared,
        "contract_version": contract.service_wide_custom_allocation_version,
        "scope_ids": list(contract.service_wide_custom_scope_ids),
        "companion_table_path": str(path),
        "companion_table_present": path.exists(),
    }
    if not declared:
        if path.exists():
            issues.append(
                f"{label}: ledger is present but the build does not declare service-wide allocation"
            )
        return summary
    if not path.exists():
        issues.append(f"{label}: service-wide allocation is declared but ledger {path} is absent")
        return summary
    if contract.service_state_transfer_companion_table != path.name:
        issues.append(
            f"{label}: manifest names {contract.service_state_transfer_companion_table!r}, "
            f"expected {path.name!r}"
        )
    if not contract.service_wide_custom_allocation_version:
        issues.append(f"{label}: manifest omits the service-wide allocation contract version")
    if not contract.service_wide_custom_scope_ids:
        issues.append(f"{label}: manifest declares no service scope ids")
    audit_path = state_output_dir / f"allocation_component_denominator_audit_{YEAR}.parquet"
    if not audit_path.exists():
        issues.append(f"{label}: component audit {audit_path} is absent")
        return summary
    ledger = pd.read_parquet(path)
    audit_columns = [
        "state_fips",
        "offense",
        "service_scope_id",
        "canonical_target_ori",
        "source_state_fips",
        "source_target_count",
        "component_count_after",
        "jurisdiction_type",
    ]
    try:
        component_audit = pd.read_parquet(audit_path, columns=audit_columns)
    except (KeyError, ValueError) as exc:
        issues.append(f"{label}: component audit cannot supply service columns: {exc}")
        return summary
    observed_scopes = sorted(
        ledger["service_scope_id"].dropna().astype(str).unique().tolist()
        if "service_scope_id" in ledger.columns
        else []
    )
    if observed_scopes != sorted(contract.service_wide_custom_scope_ids):
        issues.append(
            f"{label}: ledger scope ids {observed_scopes} disagree with manifest "
            f"{sorted(contract.service_wide_custom_scope_ids)}"
        )
    summary.update(
        _check_service_state_transfer_table(
            ledger=ledger,
            component_audit=component_audit,
            unlocated_mass=_unlocated_mass_table(state_output_dir),
            issues=issues,
            label=label,
        )
    )
    return summary


def _check_custom_footprint_fraction_contract(
    *, contract: ReleaseContract, state_output_dir: Path, issues: list[str]
) -> dict[str, Any]:
    """Validate the declared count-prior responsibility fractions and their bound inputs."""
    label = "release_contract.custom_footprint_fraction_basis"
    manifest = _load_json(contract.manifest_path) if contract.manifest_path else None
    if not isinstance(manifest, dict):
        return {"present": False}
    summary_block = manifest.get("summary") or {}
    input_stats = manifest.get("input_file_stats") or {}
    declarations = {
        "service_wide_custom_allocation": (
            "count_prior_x_within_bg_land_fraction",
            "service_wide_footprint_coverage",
            {
                "service_scope_id", "state_fips", "block_group_geoid",
                "bg_service_population_coverage_share", "bg_land_area_coverage_share",
                "coverage_basis",
            },
        ),
        "ordinary_resident_custom_allocation": (
            "count_prior_x_within_bg_responsibility_population_fraction",
            "overlap_custom_footprint_resident_coverage",
            {
                "ori", "state_fips", "block_group_geoid",
                "bg_responsibility_population_coverage_share",
                "responsibility_fraction_basis",
            },
        ),
    }
    result: dict[str, Any] = {"present": True, "declarations": {}}
    for block_name, (expected_basis, input_key, required_columns) in declarations.items():
        block = summary_block.get(block_name)
        if not isinstance(block, dict):
            if contract.service_wide_custom_allocation_enabled:
                issues.append(f"{label}: manifest is missing {block_name}")
            continue
        if block.get("allocation_weight_basis") != expected_basis:
            issues.append(
                f"{label}: {block_name} allocation_weight_basis is "
                f"{block.get('allocation_weight_basis')!r}, expected {expected_basis!r}"
            )
        declared_path = block.get("footprint_coverage_path")
        declared_sha = block.get("footprint_coverage_sha256")
        stat = input_stats.get(input_key) if isinstance(input_stats, dict) else None
        input_path = Path(str(stat.get("path"))) if isinstance(stat, dict) and stat.get("path") else None
        if not declared_path or not declared_sha or input_path is None:
            issues.append(
                f"{label}: {block_name} must bind coverage path, SHA256, and input_file_stats"
            )
            continue
        if not input_path.exists() or not str(input_path).endswith(str(declared_path)):
            issues.append(
                f"{label}: {block_name} declared coverage path does not match its manifest input"
            )
            continue
        actual_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if actual_sha != str(declared_sha):
            issues.append(f"{label}: {block_name} coverage SHA256 does not match {input_path}")
        table = pd.read_csv(input_path, dtype="string")
        missing = sorted(required_columns - set(table.columns))
        if missing:
            issues.append(f"{label}: {block_name} coverage is missing columns {missing}")
            continue
        table["state_fips"] = table["state_fips"].str.strip().str.zfill(2)
        table["block_group_geoid"] = (
            table["block_group_geoid"].str.strip().str.zfill(12)
        )
        if block_name == "service_wide_custom_allocation":
            fraction_specs = (
                ("bg_service_population_coverage_share", True),
                ("bg_land_area_coverage_share", False),
            )
            key = ["service_scope_id", "state_fips", "block_group_geoid"]
            basis_column = "coverage_basis"
        else:
            fraction_specs = (("bg_responsibility_population_coverage_share", False),)
            key = ["ori", "state_fips", "block_group_geoid"]
            basis_column = "responsibility_fraction_basis"
        if bool(table.duplicated(key, keep=False).any()):
            issues.append(f"{label}: {block_name} coverage has duplicate identity rows")
        if bool(table[basis_column].isna().any() | table[basis_column].str.strip().eq("").any()):
            issues.append(f"{label}: {block_name} coverage has blank fraction basis")
        for fraction_column, allow_zero in fraction_specs:
            fraction = pd.to_numeric(table[fraction_column], errors="coerce")
            lower_bad = fraction.lt(0.0) if allow_zero else fraction.le(0.0)
            invalid = fraction.isna() | ~np.isfinite(fraction) | lower_bad | fraction.gt(1.0 + 1e-9)
            if bool(invalid.any()):
                interval = "[0, 1]" if allow_zero else "(0, 1]"
                issues.append(
                    f"{label}: {block_name} {fraction_column} must be finite in {interval}"
                )
        if block_name == "service_wide_custom_allocation":
            footprint_stat = input_stats.get("overlap_custom_footprints")
            scope_stat = input_stats.get("service_wide_agency_scopes")
            policy_stat = input_stats.get("primary_service_response_policies")
            footprint_path = (
                Path(str(footprint_stat.get("path")))
                if isinstance(footprint_stat, dict) and footprint_stat.get("path")
                else None
            )
            scope_path = (
                Path(str(scope_stat.get("path")))
                if isinstance(scope_stat, dict) and scope_stat.get("path")
                else None
            )
            if (
                footprint_path is None or not footprint_path.exists()
                or scope_path is None or not scope_path.exists()
            ):
                issues.append(f"{label}: service coverage has no bound scope/base footprint input")
            else:
                footprint = pd.read_csv(
                    footprint_path,
                    dtype="string",
                    usecols=[
                        "ori", "state_fips", "block_group_geoid", "weight_share_basis",
                        "bg_population_coverage_share",
                    ],
                )
                footprint = footprint[
                    footprint["weight_share_basis"].eq("service_area_prior")
                ].copy()
                scopes = pd.read_csv(
                    scope_path,
                    dtype="string",
                    usecols=["service_scope_id", "canonical_target_ori"],
                ).drop_duplicates()
                covered = table.merge(
                    scopes, on="service_scope_id", how="left", validate="many_to_one"
                ).rename(columns={"canonical_target_ori": "ori"})
                footprint["state_fips"] = footprint["state_fips"].str.zfill(2)
                footprint["block_group_geoid"] = footprint[
                    "block_group_geoid"
                ].str.zfill(12)
                service_key = ["ori", "state_fips", "block_group_geoid"]
                joined = footprint.merge(
                    covered, on=service_key, how="outer", indicator=True
                )
                if not joined["_merge"].eq("both").all():
                    issues.append(
                        f"{label}: service coverage keys do not equal the service-area "
                        "footprint keys"
                    )
                own = pd.to_numeric(
                    joined["bg_service_population_coverage_share"], errors="coerce"
                )
                union = pd.to_numeric(joined["bg_population_coverage_share"], errors="coerce")
                invalid_union = (
                    union.isna()
                    | ~np.isfinite(union)
                    | union.lt(0.0)
                    | union.gt(1.0 + 1e-9)
                )
                if bool(invalid_union.any()):
                    issues.append(
                        f"{label}: service union coverage must be explicit and finite in "
                        f"[0, 1] on {int(invalid_union.sum())} rows"
                    )
                outside_union = (
                    own.notna() & ~invalid_union & own.gt(union + 1e-9)
                )
                if bool(outside_union.any()):
                    issues.append(
                        f"{label}: service own-population fraction exceeds union geometry "
                        f"on {int(outside_union.sum())} rows"
                    )
            policy_path = (
                Path(str(policy_stat.get("path")))
                if isinstance(policy_stat, dict) and policy_stat.get("path")
                else None
            )
            declared_policy_path = block.get("primary_response_policy_path")
            declared_policy_sha = block.get("primary_response_policy_sha256")
            declared_policy_scopes = sorted(block.get("primary_response_scope_ids") or [])
            if policy_path is None or not policy_path.exists():
                issues.append(f"{label}: service allocation has no bound primary-response policy")
            elif (
                not declared_policy_path
                or not str(policy_path).endswith(str(declared_policy_path))
                or hashlib.sha256(policy_path.read_bytes()).hexdigest()
                != str(declared_policy_sha)
            ):
                issues.append(
                    f"{label}: primary-response policy path/SHA does not match its manifest input"
                )
            else:
                policy = pd.read_csv(policy_path, dtype="string")
                required_policy = {
                    "service_scope_id", "primary_response_policy", "official_source_ref",
                    "evidence_artifact", "evidence_sha256",
                }
                missing_policy = sorted(required_policy - set(policy.columns))
                if missing_policy:
                    issues.append(
                        f"{label}: primary-response policy is missing columns {missing_policy}"
                    )
                else:
                    observed_policy_scopes = sorted(
                        policy["service_scope_id"].dropna().astype(str).tolist()
                    )
                    blank_evidence = policy[
                        ["official_source_ref", "evidence_artifact", "evidence_sha256"]
                    ].isna().any(axis=1) | policy[
                        ["official_source_ref", "evidence_artifact", "evidence_sha256"]
                    ].fillna("").apply(lambda column: column.str.strip().eq("")).any(axis=1)
                    if observed_policy_scopes != declared_policy_scopes:
                        issues.append(
                            f"{label}: primary-response policy scopes disagree with manifest"
                        )
                    if bool(blank_evidence.any()):
                        issues.append(
                            f"{label}: primary-response policy has blank source evidence"
                        )
        if block_name == "ordinary_resident_custom_allocation":
            footprint_stat = input_stats.get("overlap_custom_footprints")
            footprint_path = (
                Path(str(footprint_stat.get("path")))
                if isinstance(footprint_stat, dict) and footprint_stat.get("path")
                else None
            )
            if footprint_path is None or not footprint_path.exists():
                issues.append(f"{label}: ordinary coverage has no bound base footprint input")
            else:
                footprint = pd.read_csv(
                    footprint_path,
                    dtype="string",
                    usecols=[
                        "ori", "state_fips", "block_group_geoid", "weight_share_basis",
                        "bg_population_coverage_share",
                    ],
                )
                footprint = footprint[
                    footprint["weight_share_basis"].eq("resident_population")
                ].copy()
                footprint["state_fips"] = footprint["state_fips"].str.strip().str.zfill(2)
                footprint["block_group_geoid"] = (
                    footprint["block_group_geoid"].str.strip().str.zfill(12)
                )
                joined = footprint.merge(table, on=key, how="outer", indicator=True)
                if not joined["_merge"].eq("both").all():
                    issues.append(
                        f"{label}: ordinary responsibility coverage keys do not equal the "
                        "resident-population footprint keys"
                    )
                own = pd.to_numeric(
                    joined["bg_responsibility_population_coverage_share"], errors="coerce"
                )
                union = pd.to_numeric(joined["bg_population_coverage_share"], errors="coerce")
                outside_union = own.notna() & union.notna() & own.gt(union + 1e-9)
                if bool(outside_union.any()):
                    issues.append(
                        f"{label}: ordinary own-responsibility fraction exceeds union geometry "
                        f"on {int(outside_union.sum())} rows"
                    )
        result["declarations"][block_name] = {
            "rows": int(len(table)), "path": str(input_path), "sha256": actual_sha,
        }

    audit_path = state_output_dir / f"allocation_component_denominator_audit_{YEAR}.parquet"
    if contract.service_wide_custom_allocation_enabled and audit_path.exists():
        audit = pd.read_parquet(
            audit_path,
            columns=[
                "bg_responsibility_population_coverage_share",
                "responsibility_fraction_basis",
            ],
        )
        responsibility = pd.to_numeric(
            audit["bg_responsibility_population_coverage_share"], errors="coerce"
        )
        active = responsibility.notna()
        invalid = active & (
            ~np.isfinite(responsibility) | responsibility.le(0.0) | responsibility.gt(1.0 + 1e-9)
        )
        blank_basis = active & audit["responsibility_fraction_basis"].astype(
            "string"
        ).fillna("").str.strip().eq("")
        if bool(invalid.any()):
            issues.append(f"{label}: component audit has invalid responsibility fractions")
        if bool(blank_basis.any()):
            issues.append(f"{label}: component audit has responsibility fractions without a basis")
        result["audit_responsibility_rows"] = int(active.sum())
    result["ok"] = not any(issue.startswith(label) for issue in issues)
    return result


class ReleaseContract:
    """The v2 lanes a build declares, plus the named constants each declared lane publishes under.

    `lanes` is the authority for which assertion set runs. The remaining attributes are the
    manifest's own account of itself; they are asserted against this file's transcriptions rather
    than used in place of them.

    Deliberately a plain class rather than a dataclass: this module is loaded by file path in the
    lane test suites, and `@dataclass` under `from __future__ import annotations` needs the module
    registered in `sys.modules` to resolve its string annotations.
    """

    __slots__ = (
        "lanes",
        "manifest_path",
        "manifest_present",
        "exposure_normalizer_ids",
        "exposure_normalizer_version",
        "exposure_semantics",
        "exposure_normalizers_path",
        "exposure_residential_leg_source",
        "exposure_weights_path",
        "uncertainty_index_breaks",
        "uncertainty_version",
        "special_use_version",
        "special_use_thresholds",
        "count_first_version",
        "zero_resident_opportunity_policy",
        "soft_shrinkage_nu",
        "soft_shrinkage_extrapolation_weight",
        "soft_shrinkage_envelope_mode",
        "imputation_v2_enabled",
        "unlocated_mass_identity",
        "unlocated_mass_companion_table",
        "unlocated_mass_routes",
        "unlocated_mass_semantics",
        "service_wide_custom_allocation_enabled",
        "service_wide_custom_allocation_version",
        "service_wide_custom_scope_ids",
        "service_state_transfer_companion_table",
    )

    def __init__(
        self,
        *,
        lanes=frozenset(),
        manifest_path=None,
        manifest_present=False,
        exposure_normalizer_ids=None,
        exposure_normalizer_version=None,
        exposure_semantics=None,
        exposure_normalizers_path=None,
        exposure_residential_leg_source=None,
        exposure_weights_path=None,
        uncertainty_index_breaks=(),
        uncertainty_version=None,
        special_use_version=None,
        special_use_thresholds=None,
        count_first_version=None,
        zero_resident_opportunity_policy=None,
        soft_shrinkage_nu=None,
        soft_shrinkage_extrapolation_weight=None,
        soft_shrinkage_envelope_mode=None,
        imputation_v2_enabled=None,
        unlocated_mass_identity=None,
        unlocated_mass_companion_table=None,
        unlocated_mass_routes=(),
        unlocated_mass_semantics=None,
        service_wide_custom_allocation_enabled=False,
        service_wide_custom_allocation_version=None,
        service_wide_custom_scope_ids=(),
        service_state_transfer_companion_table=None,
    ) -> None:
        self.lanes = frozenset(lanes)
        self.manifest_path = manifest_path
        self.manifest_present = bool(manifest_present)
        self.exposure_normalizer_ids = dict(exposure_normalizer_ids or {})
        self.exposure_normalizer_version = exposure_normalizer_version
        self.exposure_semantics = exposure_semantics
        self.exposure_normalizers_path = exposure_normalizers_path
        self.exposure_residential_leg_source = exposure_residential_leg_source
        self.exposure_weights_path = exposure_weights_path
        self.uncertainty_index_breaks = tuple(uncertainty_index_breaks)
        self.uncertainty_version = uncertainty_version
        self.special_use_version = special_use_version
        self.special_use_thresholds = dict(special_use_thresholds or {})
        self.count_first_version = count_first_version
        self.zero_resident_opportunity_policy = dict(
            zero_resident_opportunity_policy or {}
        )
        self.soft_shrinkage_nu = soft_shrinkage_nu
        self.soft_shrinkage_extrapolation_weight = soft_shrinkage_extrapolation_weight
        self.soft_shrinkage_envelope_mode = soft_shrinkage_envelope_mode
        self.imputation_v2_enabled = imputation_v2_enabled
        self.unlocated_mass_identity = unlocated_mass_identity
        self.unlocated_mass_companion_table = unlocated_mass_companion_table
        self.unlocated_mass_routes = tuple(unlocated_mass_routes or ())
        self.unlocated_mass_semantics = unlocated_mass_semantics
        self.service_wide_custom_allocation_enabled = bool(
            service_wide_custom_allocation_enabled
        )
        self.service_wide_custom_allocation_version = service_wide_custom_allocation_version
        self.service_wide_custom_scope_ids = tuple(service_wide_custom_scope_ids or ())
        self.service_state_transfer_companion_table = service_state_transfer_companion_table

    def __repr__(self) -> str:
        return f"ReleaseContract(lanes={sorted(self.lanes)!r})"

    def enabled(self, lane: str) -> bool:
        return lane in self.lanes

    @property
    def mixture_allocation(self) -> bool:
        return self.enabled("mixture_allocation")

    @property
    def exposure_ensemble(self) -> bool:
        return self.enabled("exposure_ensemble")

    @property
    def count_first_composites(self) -> bool:
        return self.enabled("count_first_composites")

    @property
    def special_use_taxonomy(self) -> bool:
        return self.enabled("special_use_taxonomy")

    @property
    def uncertainty_layer(self) -> bool:
        return self.enabled("uncertainty_layer")

    @property
    def soft_shrinkage(self) -> bool:
        return self.enabled("soft_shrinkage")

    @property
    def unlocated_mass(self) -> bool:
        return self.enabled("unlocated_mass")

    @property
    def zero_resident_opportunity_floor(self) -> bool:
        """Whether the near-zero-resident opportunity floor was in force on this build.

        Not a lane of its own: it is the publication rule the two lanes that CREATED the
        zero-resident opportunity-rate class have to carry, so either one turns it on. Read off the
        declared lanes rather than off the policy block, so a build that dropped the block from its
        manifest fails the assertion below instead of silently exempting itself from the rule.
        """
        return self.special_use_taxonomy or self.exposure_ensemble


LEGACY_RELEASE_CONTRACT = ReleaseContract()


def _build_manifest_path(output_dir: Path) -> Path:
    """The build manifest, under either of the two names a release tree uses."""
    path = output_dir / "manifest.json"
    if not path.exists():
        path = output_dir / f"crimerisk_output_build_{YEAR}.json"
    return path


def _load_release_contract(output_dir: Path) -> ReleaseContract:
    path = _build_manifest_path(output_dir)
    manifest = _load_json(path)
    if manifest is None:
        # No manifest is its own issue, raised by `_check_build_manifest`. The contract falls back
        # to legacy so the surfaces are still held to the strictest set this file knows.
        return ReleaseContract(manifest_path=path, manifest_present=False)
    resolved = manifest.get("resolved_config") or {}
    lanes = frozenset(
        lane
        for lane in V2_LANE_KEYS
        if isinstance(resolved.get(lane), dict) and bool(resolved[lane].get("enabled"))
    )
    exposure = resolved.get("exposure_ensemble") or {}
    uncertainty = resolved.get("uncertainty_layer") or {}
    taxonomy = resolved.get("special_use_taxonomy") or {}
    composites = resolved.get("count_first_composites") or {}
    shrinkage = resolved.get("soft_shrinkage") or {}
    imputation = resolved.get("imputation_v2") or {}
    unlocated = resolved.get("unlocated_mass") or {}
    summary = manifest.get("summary") or {}
    service_wide = summary.get("service_wide_custom_allocation") or {}
    normalizer_ids = exposure.get("normalizer_id_by_offense") or {}
    breaks = uncertainty.get("index_breaks") or []
    thresholds = taxonomy.get("thresholds") or {}
    return ReleaseContract(
        lanes=lanes,
        manifest_path=path,
        manifest_present=True,
        exposure_normalizer_ids={
            str(key): str(value) for key, value in normalizer_ids.items()
        },
        exposure_normalizer_version=exposure.get("normalizer_version"),
        exposure_semantics=exposure.get("semantics"),
        exposure_normalizers_path=exposure.get("normalizers_path"),
        exposure_residential_leg_source=(exposure.get("residential_leg") or {}).get(
            "source"
        ),
        exposure_weights_path=exposure.get("weights_path"),
        uncertainty_index_breaks=tuple(float(value) for value in breaks),
        uncertainty_version=uncertainty.get("version"),
        special_use_version=taxonomy.get("version"),
        special_use_thresholds={
            str(key): float(value)
            for key, value in thresholds.items()
            if value is not None
        },
        count_first_version=composites.get("version"),
        zero_resident_opportunity_policy=(
            resolved.get("zero_resident_opportunity_rate_policy") or {}
        ),
        soft_shrinkage_nu=shrinkage.get("nu"),
        soft_shrinkage_extrapolation_weight=shrinkage.get("extrapolation_weight"),
        soft_shrinkage_envelope_mode=shrinkage.get("envelope_mode"),
        imputation_v2_enabled=imputation.get("enabled"),
        unlocated_mass_identity=unlocated.get("identity"),
        unlocated_mass_companion_table=unlocated.get("companion_table"),
        unlocated_mass_routes=tuple(
            str(route) for route in (unlocated.get("unlocated_routes") or ())
        ),
        unlocated_mass_semantics=unlocated.get("semantics"),
        service_wide_custom_allocation_enabled=service_wide.get("enabled", False),
        service_wide_custom_allocation_version=service_wide.get("contract_version"),
        service_wide_custom_scope_ids=tuple(
            str(scope_id) for scope_id in (service_wide.get("scope_ids") or ())
        ),
        service_state_transfer_companion_table=(
            Path(str(summary["service_state_transfer_path"])).name
            if summary.get("service_state_transfer_path")
            else None
        ),
    )


def _exposure_ensemble_lane(df: pd.DataFrame) -> bool:
    """Whether the surface carries the v2 exposure ensemble, read off the surface itself."""
    return EXPOSURE_NORMALIZER_SEMANTICS_COLUMN in df.columns


def _contract_lane_evidence(df: pd.DataFrame) -> dict[str, bool]:
    """What the surface's own columns say about which lanes produced it."""
    return {
        # The mixture allocator moves counts and mints no column, so the surface cannot testify
        # about it; its manifest block and the model-total preservation check carry that lane.
        "exposure_ensemble": _exposure_ensemble_lane(df),
        "count_first_composites": _count_first_composite_lane(df),
        "special_use_taxonomy": _special_use_taxonomy_lane(df),
        "uncertainty_layer": _uncertainty_layer_lane(df),
    }


def _exposure_normalizer_column(offense: str) -> str:
    return f"{EXPOSURE_NORMALIZER_COLUMN_PREFIX}{offense}"


def _exposure_normalizer_id_column(offense: str) -> str:
    return f"primary_denominator_normalizer_id_{offense}"


def _exposure_ensemble_columns() -> list[str]:
    """The columns the exposure lane adds, and only when it is on."""
    columns = [EXPOSURE_NORMALIZER_SEMANTICS_COLUMN]
    columns.extend(_exposure_normalizer_id_column(offense) for offense in OFFENSES_7)
    columns.extend(_exposure_normalizer_column(offense) for offense in EXPOSURE_ENSEMBLE_OFFENSES)
    return columns


def _uncertainty_layer_columns() -> list[str]:
    """The columns the uncertainty layer adds, and only when it is on."""
    columns = ["uncertainty_layer_version"]
    for offense in OFFENSES_7:
        columns.extend(
            f"expected_count_{offense}_{quantile}"
            for quantile in UNCERTAINTY_QUANTILE_SUFFIXES["expected_count"]
        )
        columns.extend(
            f"rate_{offense}_primary_{quantile}"
            for quantile in UNCERTAINTY_QUANTILE_SUFFIXES["rate_primary"]
        )
        columns.extend(
            f"index_{offense}_primary_{quantile}"
            for quantile in UNCERTAINTY_QUANTILE_SUFFIXES["index_primary"]
        )
        columns.extend(
            [
                f"prob_index_{offense}_above_100",
                f"displayed_index_bin_{offense}",
                f"prob_displayed_index_bin_{offense}",
                f"decision_reliability_tier_{offense}",
                f"uncertainty_support_class_{offense}",
                *[f"{prefix}{offense}" for prefix in UNCERTAINTY_DECOMPOSITION_COLUMN_PREFIXES],
            ]
        )
    return columns


def _load_exposure_weight_table(*, issues: list[str], label: str) -> pd.DataFrame | None:
    """The frozen weight table, read with this file's own reader and checked against E3's vectors.

    The contract's rule is "frozen, read in, never re-derived": a new vector is a new versioned
    file. So the assertion is that the shipped table still IS the transcription below, on the
    simplex, pointed at the surface columns this lane reads.
    """
    path = REPO_ROOT / EXPOSURE_WEIGHTS_CSV
    if not path.exists():
        issues.append(f"{label}: exposure ensemble is enabled but {EXPOSURE_WEIGHTS_CSV} is absent")
        return None
    table = pd.read_csv(path, dtype={"normalizer_id": "string", "offense": "string", "leg": "string"})
    required = {"normalizer_id", "offense", "leg", "surface_column", "weight"}
    missing = sorted(required - set(table.columns))
    if missing:
        issues.append(f"{label}: {EXPOSURE_WEIGHTS_CSV} is missing columns {missing}")
        return None
    if bool(table.duplicated(subset=["normalizer_id", "offense", "leg"]).any()):
        issues.append(f"{label}: {EXPOSURE_WEIGHTS_CSV} carries duplicate (normalizer_id, offense, leg) rows")
        return None

    person = table[table["normalizer_id"].eq("person_ens_v1")]
    for offense, expected in EXPOSURE_PERSON_ENSEMBLE_WEIGHTS.items():
        rows = person[person["offense"].eq(offense)]
        shipped = {
            str(row.leg): float(row.weight)
            for row in rows.itertuples()
        }
        if shipped != expected:
            issues.append(
                f"{label}: person_ens_v1 weights for {offense} are {shipped}, not E3's {expected}"
            )
        total = float(sum(shipped.values())) if shipped else float("nan")
        if not (pd.notna(total) and abs(total - 1.0) <= EXPOSURE_SIMPLEX_TOLERANCE):
            issues.append(f"{label}: person_ens_v1 weights for {offense} do not sum to 1 (sum {total})")
        for row in rows.itertuples():
            expected_column = EXPOSURE_PERSON_LEG_COLUMNS.get(str(row.leg))
            if expected_column is None:
                issues.append(f"{label}: person_ens_v1 carries an unknown leg {row.leg!r}")
            if float(row.weight) < 0.0:
                issues.append(f"{label}: person_ens_v1 leg {row.leg!r} for {offense} carries a negative weight")

    hybrid = table[table["normalizer_id"].eq("larceny_opp_v1")]
    shipped_hybrid = {str(row.leg): float(row.weight) for row in hybrid.itertuples()}
    if shipped_hybrid != EXPOSURE_LARCENY_HYBRID_WEIGHTS:
        issues.append(
            f"{label}: larceny_opp_v1 parts are {shipped_hybrid}, not E3's "
            f"{EXPOSURE_LARCENY_HYBRID_WEIGHTS}"
        )
    hybrid_total = float(sum(shipped_hybrid.values())) if shipped_hybrid else float("nan")
    if not (pd.notna(hybrid_total) and abs(hybrid_total - 1.0) <= EXPOSURE_SIMPLEX_TOLERANCE):
        issues.append(f"{label}: larceny_opp_v1 parts do not sum to 1 (sum {hybrid_total})")
    return table


def _expected_exposure_normalizers(
    artifact: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Recompose every named normalizer from the artifact's own legs at the frozen weights.

    This is the contract's `Composition` section evaluated independently:

        person_o       = sum_leg w(o, leg) * leg
        hybrid_larceny = 0.5 * person_larceny
                       + 0.3 * destination_poi * (sum person_larceny / sum destination_poi)
                       + 0.2 * retail_jobs     * (sum person_larceny / sum retail_jobs)
        normalizer_o   = s_o * unscaled_o,  s_o = sum population / sum unscaled_o

    Every part is rescaled to the person ensemble's reference total BEFORE mixing, because a POI
    count is not a person; the single national scalar afterwards fixes the level the rate is
    quoted in and cannot move a share, a rank or an index.
    """
    legs = {
        leg: pd.to_numeric(artifact[column], errors="coerce").fillna(0.0)
        for leg, column in EXPOSURE_PERSON_LEG_COLUMNS.items()
        if column in artifact.columns
    }
    if "residential_leg" in artifact.columns:
        legs["landscan_night"] = pd.to_numeric(
            artifact["residential_leg"], errors="coerce"
        ).fillna(0.0)
    population_total = float(
        pd.to_numeric(artifact.get("population"), errors="coerce").fillna(0.0).sum()
    )
    expected: dict[str, pd.Series] = {}
    for offense in EXPOSURE_ENSEMBLE_OFFENSES:
        weights = EXPOSURE_PERSON_ENSEMBLE_WEIGHTS[offense]
        person = pd.Series(0.0, index=artifact.index, dtype=float)
        for leg, weight in weights.items():
            if leg not in legs:
                return {}
            person = person + float(weight) * legs[leg]
        if offense == "larceny":
            unscaled = float(EXPOSURE_LARCENY_HYBRID_WEIGHTS["person_ensemble"]) * person
            person_total = float(person.sum())
            for part, weight in EXPOSURE_LARCENY_HYBRID_WEIGHTS.items():
                if part == "person_ensemble" or float(weight) == 0.0:
                    continue
                column = EXPOSURE_LARCENY_HYBRID_SOURCE_COLUMNS[part]
                if column not in artifact.columns:
                    return {}
                values = pd.to_numeric(artifact[column], errors="coerce").fillna(0.0)
                part_total = float(values.sum())
                part_scale = person_total / part_total if part_total > 0.0 else 0.0
                unscaled = unscaled + float(weight) * values * part_scale
        else:
            unscaled = person
        unscaled_total = float(unscaled.sum())
        scale = population_total / unscaled_total if unscaled_total > 0.0 else float("nan")
        expected[offense] = unscaled * scale
    return expected


def _check_exposure_normalizer_recomposition(
    *, contract: ReleaseContract, issues: list[str]
) -> dict[str, Any]:
    """Rebuild every published normalizer from the build's own leg artifact and the frozen weights.

    The published surface carries the normalizer but not the LandScan night leg, so the identity
    `normalizer = s * sum_leg w * leg` is only recomputable against the artifact the manifest
    names. Coverage fails closed: a manifest that quotes a normalizer path with no file behind it
    is an artifact whose denominators cannot be reproduced.
    """
    label = "exposure_ensemble"
    summary: dict[str, Any] = {"enabled": True, "recomposed": False}
    _load_exposure_weight_table(issues=issues, label=label)

    expected_version = (
        CENSUS_RESIDENTIAL_NORMALIZER_VERSION
        if contract.exposure_residential_leg_source
        == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
        else EXPOSURE_NORMALIZER_VERSION
    )
    if contract.exposure_normalizer_version != expected_version:
        issues.append(
            f"{label}: manifest normalizer_version is {contract.exposure_normalizer_version!r}, "
            f"expected {expected_version!r}"
        )
    if contract.exposure_semantics != EXPOSURE_NORMALIZER_SEMANTICS:
        issues.append(
            f"{label}: manifest semantics is {contract.exposure_semantics!r}, expected "
            f"{EXPOSURE_NORMALIZER_SEMANTICS!r}"
        )
    manifest_ids = dict(contract.exposure_normalizer_ids)
    if manifest_ids != EXPOSURE_NORMALIZER_ID_BY_OFFENSE:
        issues.append(
            f"{label}: manifest normalizer_id_by_offense is {manifest_ids}, expected "
            f"{EXPOSURE_NORMALIZER_ID_BY_OFFENSE}"
        )

    normalizers_path = _resolve_manifest_path(contract.exposure_normalizers_path)
    if normalizers_path is None or not normalizers_path.exists():
        issues.append(
            f"{label}: the manifest names normalizer artifact "
            f"{contract.exposure_normalizers_path!r}, which is absent; the published denominators "
            "cannot be recomposed"
        )
        summary["normalizers_path"] = str(contract.exposure_normalizers_path)
        return summary
    summary["normalizers_path"] = str(normalizers_path)

    artifact = pd.read_parquet(normalizers_path)
    if contract.exposure_residential_leg_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE:
        required_residential = {
            "raw_landscan_night_pop",
            "residential_leg",
            "residential_leg_source",
            "population",
            "landscan_night_leg",
        }
        missing_residential = sorted(required_residential - set(artifact.columns))
        if missing_residential:
            issues.append(
                f"{label}: Census residential artifact is missing {missing_residential}"
            )
        else:
            sources = set(artifact["residential_leg_source"].astype(str))
            if sources != {CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE}:
                issues.append(
                    f"{label}: residential source column carries {sorted(sources)}"
                )
            population = pd.to_numeric(artifact["population"], errors="coerce")
            residential = pd.to_numeric(artifact["residential_leg"], errors="coerce")
            if not np.allclose(population, residential, rtol=0.0, atol=1e-9):
                issues.append(
                    f"{label}: Census residential leg does not equal release population nationwide"
                )
    elif contract.exposure_residential_leg_source not in (
        None,
        LANDSCAN_NIGHT_RESIDENTIAL_SOURCE,
    ):
        issues.append(
            f"{label}: manifest residential source is "
            f"{contract.exposure_residential_leg_source!r}"
        )
    expected = _expected_exposure_normalizers(artifact)
    if not expected:
        issues.append(f"{label}: the normalizer artifact does not carry the contract's leg columns")
        return summary
    summary["recomposed"] = True
    deltas: dict[str, float] = {}
    for offense, values in expected.items():
        column = _exposure_normalizer_column(offense)
        if column not in artifact.columns:
            issues.append(f"{label}: the normalizer artifact is missing {column}")
            continue
        published = pd.to_numeric(artifact[column], errors="coerce")
        if bool(published.isna().any()):
            issues.append(f"{label}: {column} carries nulls in the normalizer artifact")
        denominator = published.abs().where(published.abs().gt(0.0))
        relative = ((published - values).abs() / denominator).max()
        relative = float(relative) if pd.notna(relative) else 0.0
        deltas[offense] = relative
        if relative > EXPOSURE_RECOMPOSITION_RELATIVE_TOLERANCE:
            issues.append(
                f"{label}: {column} does not recompose from its published legs at the frozen "
                f"weights (max relative diff {relative:.3e})"
            )
        # Positivity where people are: a normalizer that reports no exposure for an inhabited
        # cell suppresses that cell's rate on a data artifact.
        population = pd.to_numeric(artifact.get("population"), errors="coerce").fillna(0.0)
        zero_with_population = int((published.fillna(0.0).le(0.0) & population.gt(0.0)).sum())
        if zero_with_population:
            issues.append(
                f"{label}: {column} is zero on {zero_with_population} block groups that carry "
                "resident population"
            )
    summary["max_relative_recomposition_delta"] = deltas
    return summary


def _exposure_ensemble_surface_issues(
    df: pd.DataFrame,
    *,
    label: str,
    geography: str,
    contract: ReleaseContract,
) -> list[str]:
    """The lane's identities on a published surface.

    `primary_denominator_<offense>` continues to carry the value the rate was actually divided by,
    so `rate = 1e5 * count / primary_denominator` holds on both lanes and is asserted elsewhere.
    What is asserted here is WHICH surface that value came from.
    """
    issues: list[str] = []
    semantics = df.get(EXPOSURE_NORMALIZER_SEMANTICS_COLUMN)
    if semantics is None:
        issues.append(f"{label}: {EXPOSURE_NORMALIZER_SEMANTICS_COLUMN} is missing from an exposure-ensemble surface")
    else:
        values = sorted(set(semantics.astype("string").dropna().unique().tolist()))
        if values != [EXPOSURE_NORMALIZER_SEMANTICS] or int(semantics.isna().sum()):
            issues.append(
                f"{label}: {EXPOSURE_NORMALIZER_SEMANTICS_COLUMN} is {values}, expected "
                f"[{EXPOSURE_NORMALIZER_SEMANTICS!r}] on every row"
            )

    population = pd.to_numeric(df.get(f"population_{YEAR}"), errors="coerce").fillna(0.0)
    population_total = float(population.sum())
    for offense in OFFENSES_7:
        id_column = _exposure_normalizer_id_column(offense)
        expected_id = EXPOSURE_NORMALIZER_ID_BY_OFFENSE[offense]
        if id_column not in df.columns:
            issues.append(f"{label}: {id_column} is missing from an exposure-ensemble surface")
        else:
            ids = df[id_column].astype("string")
            observed = sorted(set(ids.dropna().unique().tolist()))
            if observed != [expected_id] or int(ids.isna().sum()):
                issues.append(
                    f"{label}: {id_column} is {observed}, expected [{expected_id!r}] on every row"
                )
        family = df.get(f"primary_denominator_type_{offense}")
        if family is not None:
            expected_family = PRIMARY_DENOMINATOR_BY_OFFENSE[offense]
            observed_family = sorted(set(family.astype("string").dropna().unique().tolist()))
            if observed_family != [expected_family]:
                issues.append(
                    f"{label}: primary_denominator_type_{offense} is {observed_family}; the lane "
                    f"keeps the legacy family vocabulary [{expected_family!r}]"
                )

    denominators = {
        offense: pd.to_numeric(df.get(f"primary_denominator_{offense}"), errors="coerce")
        for offense in EXPOSURE_ENSEMBLE_OFFENSES
    }
    exposure_proxy = pd.to_numeric(df.get("exposure_proxy_2024"), errors="coerce")
    for offense in EXPOSURE_ENSEMBLE_OFFENSES:
        column = _exposure_normalizer_column(offense)
        if column not in df.columns:
            issues.append(
                f"{label}: {column} is missing; the exposure ensemble is enabled but the named "
                "normalizer surface was not published"
            )
            continue
        normalizer = pd.to_numeric(df[column], errors="coerce")
        if bool(normalizer.isna().any()):
            issues.append(f"{label}: {column} carries {int(normalizer.isna().sum())} null rows")
        if bool(normalizer.lt(-1e-9).any()):
            issues.append(f"{label}: negative values in {column}")
        denominator = denominators[offense]
        delta = (denominator - normalizer).abs().max()
        if pd.isna(delta) or float(delta) > 1e-9:
            issues.append(
                f"{label}: primary_denominator_{offense} does not equal {column}; the published "
                f"rate must be quoted against the named normalizer {EXPOSURE_NORMALIZER_ID_BY_OFFENSE[offense]!r}"
            )
        # No kink: the legacy denominator is a hard MAX of two competing surfaces, whose derivative
        # with respect to the losing leg jumps 0 -> 1 exactly where the two are closest. The v2
        # normalizer is a fixed convex mix, so the hard-max identity must NOT still hold -- if it
        # does, the lane is declared and did nothing.
        if exposure_proxy is not None:
            kink_delta = (denominator - exposure_proxy).abs().max()
            if pd.notna(kink_delta) and float(kink_delta) <= 1e-9:
                issues.append(
                    f"{label}: primary_denominator_{offense} still satisfies the legacy hard-max "
                    "identity exposure_proxy_2024; the exposure ensemble is enabled but the "
                    "denominator was not replaced"
                )
        # Reference-total policy: each normalizer totals the published universe's resident
        # population. Both geographies cover the same universe, so both must.
        normalizer_total = float(normalizer.fillna(0.0).sum())
        if population_total > 0.0:
            relative = abs(normalizer_total - population_total) / population_total
            if relative > EXPOSURE_REFERENCE_TOTAL_RELATIVE_TOLERANCE:
                issues.append(
                    f"{label}: {column} totals {normalizer_total:.6f} against a reference "
                    f"population of {population_total:.6f} (relative {relative:.3e}); every named "
                    "normalizer is rescaled to the reference universe's resident population"
                )
        zero_with_population = int((normalizer.fillna(0.0).le(0.0) & population.gt(0.0)).sum())
        if zero_with_population:
            issues.append(
                f"{label}: {column} is zero on {zero_with_population} {geography} rows that carry "
                "resident population"
            )
    return issues


def _present_fields(fields: list[str], available: set[str]) -> list[str]:
    return [field for field in fields if field in available]


def _surface_column_names(path: Path) -> set[str]:
    """Column names without materialising the frame -- the spatial checks request columns by name
    and one lane's composites are absent on the other's surface."""
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names)
# The rare person offenses whose published per-offense index and rate point fields are carried
# only at census tract and coarser (docs/archive/2026-09/STATE.md, "rare-offense publication support"). At block
# group they are null by policy; the aggregates that consume a per-offense index or a rare-offense
# count take the rare terms at tract support so they stay published and recomputable at block group.
RARE_OFFENSE_TRACT_SUPPORT = ("murder", "rape")
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
    offense: set(EXPECTED_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES) for offense in OFFENSES_7
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
# The v2 lanes' near-zero-resident opportunity floor, transcribed rather than imported like every
# other rule in this file. Below the exposure floor's own resident threshold (50 residents) a
# primary opportunity-rate index additionally needs 500 units of the offense's own normalizer
# behind it; below that the cell publishes counts and density under the existing
# `insufficient_exposure` display semantics. Contracts:
# analysis_scratch/final_phase/EXPOSURE_ENSEMBLE_CONTRACT.md and SPECIAL_USE_TAXONOMY_CONTRACT.md.
ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR = 500.0
ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN = "zero_resident_opportunity_rate_floor"
PERSON_EXPOSURE_FLOOR_OFFENSES = frozenset(("murder", "rape", "robbery", "aggravated_assault", "larceny"))
MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR = 50.0
BURGLARY_PREMISES_DENOMINATOR_FLOOR = 10.0
# Producer mirror (declared locally like every other recomputed rule): modes whose
# per-offense display is suppressed for cell-level denominator invalidity while the
# count stays real. Composites take these terms at tract support.
DENOMINATOR_INVALID_ESTIMATE_MODES = frozenset(
    {"insufficient_exposure", "vehicle_denominator_invalid"}
)
SPECIAL_USE_TRACT_PREFIX = "98"
SPECIAL_USE_EXPOSURE_DENOMINATOR_FLOOR = 10.0
# Ambient-blind custom footprints (allocation.py). Restated here, like every other floor in this
# file, so the gate recomputes the rule from published fields rather than trusting the module that
# wrote them.
FOOTPRINT_DERIVED_MASS_SHARE_FLOOR = 0.5
AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO = 3.0
INSUFFICIENT_AMBIENT_EXPOSURE_REASON = "insufficient_ambient_exposure"
# Advisory transient-exposure diagnostic. These constants are transcribed independently from the
# producer so the release gate can verify both the flag and that it has no publication side effect.
TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO = 5.0
TRANSIENT_EXPOSURE_RESIDENT_INDEX_THRESHOLD = 1000.0
TRANSIENT_COMMERCIAL_PREMISES_FLOOR = 5.0
TRANSIENT_COMMERCIAL_RATIO_CEILING = 1.1
TRANSIENT_COMMERCIAL_PRIMARY_INDEX_THRESHOLD = 800.0

# --- v2 typed special-use taxonomy, mirrored ------------------------------------------------
# Declared locally like every other recomputed rule in this file. The mirror re-derives the type
# and both publication gates from the seven EXTENSIVE classification inputs the surface publishes,
# so a drift in either direction fails the release rather than the validator agreeing with the
# producer's own booleans. Contract:
# analysis_scratch/final_phase/SPECIAL_USE_TAXONOMY_CONTRACT.md
SPECIAL_USE_TYPE_COLUMN = "special_use_type"
SPECIAL_USE_ORDINARY = "ordinary"
SPECIAL_USE_CAMPUS = "campus_institution"
SPECIAL_USE_PRISON = "institutional_facility"
SPECIAL_USE_GROUP_QUARTERS_OTHER = "group_quarters_other"
SPECIAL_USE_PARK = "park_open_space"
SPECIAL_USE_TRANSIENT = "transient_destination"
SPECIAL_USE_INDUSTRIAL = "industrial_employment"
SPECIAL_USE_UNKNOWN = "unknown_special_use"
SPECIAL_USE_PRIMARY_RATE_TYPES = frozenset(
    {
        SPECIAL_USE_ORDINARY,
        SPECIAL_USE_CAMPUS,
        SPECIAL_USE_PRISON,
        SPECIAL_USE_GROUP_QUARTERS_OTHER,
        SPECIAL_USE_PARK,
        SPECIAL_USE_TRANSIENT,
        SPECIAL_USE_INDUSTRIAL,
        SPECIAL_USE_UNKNOWN,
    }
)
SPECIAL_USE_RESIDENT_RATE_TYPES = SPECIAL_USE_PRIMARY_RATE_TYPES
SPECIAL_USE_GROUP_QUARTERS_SHARE_MIN = 0.50
SPECIAL_USE_EDUCATION_JOB_SHARE_MIN = 0.50
SPECIAL_USE_POSTSECONDARY_ANCHOR_MIN = 1.0
SPECIAL_USE_OPEN_NATURAL_SHARE_MIN = 0.75
SPECIAL_USE_UNEXPLAINED_DAYTIME_RATIO_MIN = 2.5
SPECIAL_USE_FEATURE_COLUMNS = (
    "special_use_acs_population",
    "special_use_household_population",
    "special_use_jobs_total",
    "special_use_jobs_education",
    "special_use_postsecondary_count",
    "special_use_open_natural_pixels",
    "special_use_classified_pixels",
    "special_use_gq_total_2020",
    "special_use_gq_institutional_2020",
    "special_use_gq_correctional_2020",
    "special_use_gq_juvenile_2020",
    "special_use_gq_nursing_2020",
    "special_use_gq_college_2020",
    "special_use_gq_military_2020",
    "special_use_gq_other_institutional_2020",
    "special_use_gq_other_noninstitutional_2020",
)


def _special_use_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    positive = denominator.gt(0.0)
    return (numerator / denominator.where(positive)).replace(
        [float("inf"), float("-inf")], float("nan")
    ).clip(lower=0.0, upper=1.0)


def _expected_special_use_taxonomy(df: pd.DataFrame, *, population_col: str) -> pd.DataFrame:
    """Re-derive type, primary gate and resident gate from published inputs alone."""
    numeric = lambda column: pd.to_numeric(df.get(column), errors="coerce").fillna(0.0).clip(lower=0.0)
    population = numeric(population_col)
    households = numeric("households_total")
    acs_population = numeric("special_use_acs_population")
    household_population = numeric("special_use_household_population")
    jobs = numeric("special_use_jobs_total")
    education_jobs = numeric("special_use_jobs_education")
    postsecondary = numeric("special_use_postsecondary_count")
    open_natural = numeric("special_use_open_natural_pixels")
    classified = numeric("special_use_classified_pixels")
    daytime = numeric("landscan_day_pop")
    gq_total_2020 = numeric("special_use_gq_total_2020")
    gq_institutional_2020 = numeric("special_use_gq_institutional_2020")
    gq_college_2020 = numeric("special_use_gq_college_2020")
    tract_ids = (
        df["tract_id"].astype("string").str.zfill(11)
        if "tract_id" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    flag = tract_ids.str.slice(5, 11).str.startswith(SPECIAL_USE_TRACT_PREFIX, na=False)

    group_quarters = _special_use_share(
        (acs_population - household_population).clip(lower=0.0), acs_population
    )
    education = _special_use_share(education_jobs, jobs)
    open_share = _special_use_share(open_natural, classified)
    explained = population + jobs
    daytime_ratio = (daytime / explained.where(explained.gt(0.0))).replace(
        [float("inf"), float("-inf")], float("nan")
    )

    group_quarters_dominant = group_quarters.fillna(0.0).ge(
        SPECIAL_USE_GROUP_QUARTERS_SHARE_MIN
    )
    institutional = group_quarters_dominant & _special_use_share(
        gq_institutional_2020, gq_total_2020
    ).fillna(0.0).ge(SPECIAL_USE_GROUP_QUARTERS_SHARE_MIN)
    college_group_quarters = group_quarters_dominant & _special_use_share(
        gq_college_2020, gq_total_2020
    ).fillna(0.0).ge(SPECIAL_USE_GROUP_QUARTERS_SHARE_MIN)
    other_group_quarters = group_quarters_dominant & ~institutional & ~college_group_quarters
    workforce = jobs.ge(PERSON_EXPOSURE_DENOMINATOR_FLOOR)
    candidate = flag | population.le(0.0) | households.lt(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
    released = (
        candidate
        & households.ge(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
        & population.gt(0.0)
        & ~group_quarters_dominant
    )
    unresolved = candidate & ~released

    expected_type = pd.Series(SPECIAL_USE_ORDINARY, index=df.index, dtype=object)
    assigned = pd.Series(False, index=df.index)
    for name, rule in (
        (
            SPECIAL_USE_CAMPUS,
            college_group_quarters
            | (
                postsecondary.ge(SPECIAL_USE_POSTSECONDARY_ANCHOR_MIN)
                & (
                    group_quarters_dominant
                    | education.fillna(0.0).ge(SPECIAL_USE_EDUCATION_JOB_SHARE_MIN)
                )
            ),
        ),
        (SPECIAL_USE_PRISON, institutional),
        (SPECIAL_USE_GROUP_QUARTERS_OTHER, other_group_quarters),
        (SPECIAL_USE_PARK, open_share.fillna(0.0).ge(SPECIAL_USE_OPEN_NATURAL_SHARE_MIN)),
        (
            SPECIAL_USE_TRANSIENT,
            workforce & daytime_ratio.fillna(0.0).ge(SPECIAL_USE_UNEXPLAINED_DAYTIME_RATIO_MIN),
        ),
        (SPECIAL_USE_INDUSTRIAL, workforce),
    ):
        hit = unresolved & ~assigned & rule.fillna(False)
        expected_type.loc[hit] = name
        assigned = assigned | hit
    expected_type.loc[unresolved & ~assigned] = SPECIAL_USE_UNKNOWN

    primary_allowed = expected_type.isin(sorted(SPECIAL_USE_PRIMARY_RATE_TYPES))
    resident_allowed = expected_type.isin(sorted(SPECIAL_USE_RESIDENT_RATE_TYPES))
    return pd.DataFrame(
        {
            "special_use_type": expected_type.astype("string"),
            "special_use_candidate_flag": candidate,
            "special_use_primary_rate_allowed": primary_allowed.astype(bool),
            "special_use_resident_rate_allowed": resident_allowed.astype(bool),
        },
        index=df.index,
    )


def _poisson_count_interval(counts: pd.Series, *, alpha: float = POISSON_INTERVAL_ALPHA):
    """Exact Poisson (chi-square) interval on a count, restated here like the floors above so
    the gate recomputes the ambient-blind rule rather than importing the code that wrote it."""
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    values = count.to_numpy(dtype=float)
    lower = np.zeros(len(values), dtype=float)
    positive = values > 0.0
    lower[positive] = 0.5 * chi2.ppf(float(alpha) / 2.0, 2.0 * values[positive])
    upper = 0.5 * chi2.ppf(1.0 - float(alpha) / 2.0, 2.0 * (values + 1.0))
    return (
        pd.Series(lower, index=count.index, dtype=float),
        pd.Series(upper, index=count.index, dtype=float),
    )
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
# Class A (v20): jurisdiction-target mass supplied by benchmark-constrained imputation
# for territory no agency reported. It is part of the control target but has no
# counterpart in the agency/jurisdiction estimate panel by construction.
BENCHMARK_IMPUTED_COUNT_COLUMN = "benchmark_imputed_count"
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

# --- Stage 2 footprint plausibility (productionized 2026-07-30) ------------------------
# The composite screen the Stage 2 first-read audit designed and measured, promoted to a
# fail-closed release invariant. It asks one question of every exclusive municipal crosswalk
# link: is the piece of ground this agency's mass landed on plausible for the population the
# agency itself says it serves?
#
# Two directions, because the audit proved one screen only sees one of them:
#   CONCENTRATION -- implied rate > 3x national AND the footprint population disagrees with
#     the agency's own service population. Measured over all 26,767 links: 34 hits / 85,705
#     counts, of which 2 are the known-good consolidated footprints (LVMPD, LMPD), 18 tribal
#     and 14 genuine municipal misresolutions (Germantown WI town-for-village, York County
#     Regional on a 530-person borough, River Falls resolved into a different county).
#   DILUTION -- footprint population >= 2x the service population. Invisible to the rate
#     test (it LOWERS the implied rate): 9 municipal hits, 0 of them caught by the composite,
#     plus the 25 tribal reverse cases (Seminole Tribal -> Hollywood city, 751 counts into a
#     153k city). This is the conservation error a rate screen cannot see.
#
# The 3x threshold is the audit's: the bare rate>3x condition alone flags 195 links dominated
# by cities that are genuinely that dangerous with a CONSISTENT service population (Memphis
# 4.2x, Oakland 4.2x, St Louis 3.0x), so the service-population condition is what makes the
# screen a review queue rather than noise.
#
# A hit clears by being fixed or by carrying a reviewed registry row -- an ORI named in
# configs/local_resolution_overrides.csv or configs/consolidated_agency_footprints.csv. The
# registry note is the audit trail; there is no separate exemption list to drift.
STAGE2_RATE_RATIO_THRESHOLD = 3.0
# Footprint/service-population ratio band outside which the two disagree. Symmetric, because
# the same mismatch signal drives both directions.
STAGE2_SERVICE_POP_LOW_RATIO = 0.5
STAGE2_SERVICE_POP_HIGH_RATIO = 2.0

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
# ordinary cross-county rate variance -- see docs/archive/2026-09/STATE.md for the full discussion.
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
# The union over both composite lanes; each read site keeps only the fields the surface carries.
SPATIAL_STATE_SHARE_INDEX_FIELDS = [
    *SPATIAL_PRIMARY_INDEX_FIELDS,
    *SPATIAL_RESIDENT_INDEX_FIELDS,
    *ALL_COMPOSITE_INDEX_FIELDS,
]
SPATIAL_BOUNDARY_INDEX_FIELDS = [
    *SPATIAL_PRIMARY_INDEX_FIELDS,
    *ALL_COMPOSITE_INDEX_FIELDS,
]
SPATIAL_HOTSPOT_INDEX_FIELDS = [
    *SPATIAL_PRIMARY_INDEX_FIELDS,
    "index_total_part1_resident",
    "index_total_primary_event_weighted",
    EVENT_BURDEN_COLUMN,
    "multi_offense_relative_score_event_weighted",
]
BG_CENTROIDS_PATH = REPO_ROOT / "data" / "tiger_bg" / "parsed" / "bg_centroids.parquet"

# The control panel carries one per-lane column: that lane's own crosswalk-weighted
# rollup of what the agencies reported. The per-lane weight/months/relationship columns
# it used to carry existed only to feed the jurisdiction-level source preference that
# Stage 3's consumption restructure deleted.
SOURCE_TO_CONTROL_COLUMNS = {
    CIUS_SOURCE: {"count": "reported_count_cius"},
    LOCAL_PUBLICATION_SOURCE: {"count": "reported_count_local_publication"},
    STATE_PUBLICATION_SOURCE: {"count": "reported_count_state_publication"},
    SUMMARY_SOURCE: {"count": "reported_count_srs"},
    NIBRS_SOURCE: {"count": "reported_count_nibrs"},
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _allocation_control_targets(
    *, output_dir: Path, issues: list[str]
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load the control surface declared by this candidate's manifest.

    Surface 1 candidates reconcile to the one-year accounting target. Surface 2
    candidates reconcile to the smoothed risk target. The accounting spine and
    keys remain common to both; only the numeric target changes.
    """
    manifest_path = _build_manifest_path(output_dir)
    manifest = _load_json(manifest_path) or {}
    surface = str(manifest.get("resolved_config", {}).get("control_surface") or "accounting")
    keys = ["state_fips", "jurisdiction_id", "offense"]
    if surface == "accounting":
        path = REPO_ROOT / "state" / "controls" / f"jurisdiction_controls_{YEAR}.parquet"
        target_column = "adjusted_count_ags_core"
    elif surface == "smoothed":
        path = (
            REPO_ROOT
            / "state"
            / "controls"
            / f"jurisdiction_controls_smoothed_{YEAR}.parquet"
        )
        target_column = "smoothed_count"
    else:
        issues.append(
            f"manifest declares unknown control_surface {surface!r}; expected accounting or smoothed"
        )
        return None, {
            "manifest_path": str(manifest_path),
            "control_surface": surface,
            "present": False,
        }
    if not path.exists():
        issues.append(f"missing {surface} allocation controls: {path}")
        return None, {
            "manifest_path": str(manifest_path),
            "control_surface": surface,
            "path": str(path),
            "present": False,
        }
    controls = pd.read_parquet(path, columns=[*keys, target_column]).rename(
        columns={target_column: "control_target"}
    )
    controls["state_fips"] = controls["state_fips"].astype("string").str.zfill(2)
    if controls.duplicated(["jurisdiction_id", "offense"]).any():
        issues.append(f"{surface} allocation controls contain duplicate jurisdiction/offense rows")
    target = pd.to_numeric(controls["control_target"], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        issues.append(f"{surface} allocation controls contain a nonfinite target")
    if target.lt(0.0).any():
        issues.append(f"{surface} allocation controls contain a negative target")
    return controls, {
        "manifest_path": str(manifest_path),
        "control_surface": surface,
        "path": str(path),
        "target_column": target_column,
        "present": True,
        "rows": int(len(controls)),
    }


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


def _expected_columns(
    *,
    geography: str,
    count_first_composites: bool = False,
    special_use_taxonomy: bool = False,
    exposure_ensemble: bool = False,
    uncertainty_layer: bool = False,
) -> list[str]:
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
        # The near-zero-resident opportunity floor rides beside the three floors above, and only on
        # the lanes that created the class it closes. Off both lanes the column is absent, which is
        # part of what makes a legacy build byte-identical.
        *(
            (ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN,)
            if (special_use_taxonomy or exposure_ensemble)
            else ()
        ),
        "non_residential_flag",
        "special_use_tract_flag",
        *(
            (
                *SPECIAL_USE_FEATURE_COLUMNS,
                "special_use_type",
                "special_use_type_evidence",
                "special_use_candidate_flag",
                "special_use_display_policy",
                "special_use_primary_rate_allowed",
                "special_use_resident_rate_allowed",
                "special_use_transient_confidence_flag",
            )
            if special_use_taxonomy
            else ()
        ),
        "resident_secondary_denominator",
        "resident_secondary_denominator_low_reliability",
        "population_zero_with_positive_count",
        "transient_exposure_daytime_to_resident_ratio",
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
                f"spatial_share_reliability_tier_{offense}",
                f"level_reliability_tier_{offense}",
                f"level_provenance_text_{offense}",
                f"level_admission_status_{offense}",
                f"level_admission_reason_{offense}",
                f"level_repair_mode_{offense}",
                f"level_repair_share_{offense}",
                f"external_check_status_{offense}",
                f"benchmark_conflict_kind_{offense}",
                f"benchmark_weight_{offense}",
                f"unresolved_level_flag_{offense}",
                f"recommended_display_geography_{offense}",
                # Ambient-blind custom footprints: the compositional mass, its share, and the
                # rule's own flag. Listed so the loader carries them -- the Stage 2 batch found
                # this validator's entire total lane silently dead because three columns it
                # needed had been dropped from the surface and the loader failed quietly.
                f"footprint_derived_count_{offense}",
                f"footprint_derived_count_share_{offense}",
                f"footprint_ambient_exposure_missing_{offense}",
                f"transient_exposure_likely_{offense}",
                f"source_mode_{offense}",
                f"source_mode_dominant_share_{offense}",
                f"source_mode_mixed_{offense}",
                f"feed_match_rate_{offense}",
                f"feed_missing_fraction_{offense}",
                f"feed_alpha_{offense}",
                f"feed_prior_fraction_{offense}",
                # Class A (v20): share of the cell's mass whose jurisdiction target came
                # from benchmark-constrained imputation, so imputed territory reads as
                # modeled in the published confidence metadata.
                f"benchmark_imputed_share_{offense}",
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
        *([HARM_WEIGHTED_COUNT_COLUMN] if count_first_composites else []),
        *(COUNT_FIRST_INDEX_FIELDS if count_first_composites else AGGREGATE_INDEX_FIELDS),
        *(_exposure_ensemble_columns() if exposure_ensemble else []),
        *(_uncertainty_layer_columns() if uncertainty_layer else []),
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


def _rare_offense_point_fields(offense: str) -> tuple[str, ...]:
    """The per-offense index and rate point fields (and their confidence intervals) that are null
    at block-group support for the rare person offenses (mirror of allocation's field list)."""
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
    )


def _within_tract_person_exposure_share(df: pd.DataFrame, tract_ids: pd.Series) -> pd.Series:
    """Independently reproduce allocation's within-tract person-exposure share: each block group's
    share of its parent tract's person exposure, with resident-population then equal-weight
    fallbacks so the shares always sum to 1 within a tract (exact count conservation)."""
    candidates = [
        pd.to_numeric(df.get("exposure_proxy_2024"), errors="coerce").fillna(0.0).clip(lower=0.0),
        pd.to_numeric(df.get("resident_secondary_denominator"), errors="coerce").fillna(0.0).clip(lower=0.0),
        pd.Series(1.0, index=df.index, dtype=float),
    ]
    weight = candidates[0].copy()
    for fallback in candidates[1:]:
        group_sum = weight.groupby(tract_ids).transform("sum")
        weight = weight.where(group_sum.gt(0.0), fallback)
    group_sum = weight.groupby(tract_ids).transform("sum")
    share = pd.Series(0.0, index=df.index, dtype=float)
    mask = group_sum.gt(0.0)
    share.loc[mask] = weight.loc[mask] / group_sum.loc[mask]
    return share


def _redistributed_rare_offense_counts(
    df: pd.DataFrame, tract_ids: pd.Series, offenses: tuple[str, ...] = RARE_OFFENSE_TRACT_SUPPORT
) -> dict[str, pd.Series]:
    """The tract count of each rare offense spread within the tract by person-exposure share — the
    rare-offense input the block-group harm index consumes (recomputable from published fields)."""
    share = _within_tract_person_exposure_share(df, tract_ids)
    redistributed: dict[str, pd.Series] = {}
    for offense in offenses:
        expected = pd.to_numeric(df.get(f"expected_count_{offense}"), errors="coerce").fillna(0.0).clip(lower=0.0)
        tract_total = expected.groupby(tract_ids).transform("sum")
        redistributed[offense] = tract_total * share
    return redistributed


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
        # Gate on the published per-offense resident publishability FLAG, not the resident index
        # value: the value is null at block group for the rare offenses (tract-support policy), but
        # the aggregate is a count-derived resident volume total that still publishes wherever each
        # component offense is denominator-eligible (murder/rape enter only at their count share).
        offense_publishable = (
            df.get(f"index_{offense}_resident_publishable", pd.Series(False, index=df.index))
            .fillna(False)
            .astype(bool)
        )
        # Murder and rape have no block-group point payload.  Their point flags
        # therefore match that null payload (false), while the resident aggregate
        # continues to consume their counts at the common person-exposure support.
        # The remaining personal offense supplies that shared eligibility gate.
        rare_bg_payload = (
            offense in RARE_OFFENSE_TRACT_SUPPORT
            and f"index_{offense}_resident" in df.columns
            and pd.to_numeric(df[f"index_{offense}_resident"], errors="coerce").isna().all()
        )
        # A component suppressed only for cell-level denominator invalidity
        # (insufficient_exposure / vehicle_denominator_invalid) keeps a real
        # count and must not veto the aggregate — mirror of the producer rule.
        mode_exempt = (
            df.get(f"estimate_mode_{offense}", pd.Series(pd.NA, index=df.index))
            .astype("string")
            .isin(DENOMINATOR_INVALID_ESTIMATE_MODES)
            .fillna(False)
        )
        if not rare_bg_payload:
            publishable &= offense_publishable | mode_exempt
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
    index_overrides: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, pd.Series]:
    # index_overrides carries the parent-tract per-offense index for the rare offenses at block
    # group (their block-group index is null by policy); every other offense uses its own surface
    # index. Empty at tract, where the rare offenses carry their native index.
    overrides = index_overrides or {}
    values = [
        (
            pd.to_numeric(overrides[offense], errors="coerce")
            if offense in overrides
            else pd.to_numeric(df.get(f"index_{offense}_primary"), errors="coerce")
        )
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


def _harm_total_expected(
    df: pd.DataFrame,
    *,
    count_overrides: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, float, pd.Series]:
    """Count-derived index_total_harm identity: harm_count = sum(HARM_WEIGHTS[o] * expected_count_o)
    over the seven Part-I offenses, normalized once over the person-exposure denominator and its
    national rate. Publishable wherever person exposure is publishable (residential-eligible,
    non-special-use, exposure at or above the person-exposure floor) — NOT the all-or-null
    seven-index composite rule, and not nulled by any single offense's own denominator validity.

    count_overrides carries the tract-support rare-offense counts (tract count spread within the
    tract by person-exposure share) that the block-group harm index consumes for murder/rape;
    every other offense enters at its own expected count. Empty at tract (native counts).
    """
    overrides = count_overrides or {}
    denominator = pd.to_numeric(df.get("exposure_proxy_2024"), errors="coerce").fillna(0.0).clip(lower=0.0)
    if _special_use_taxonomy_lane(df):
        residential_eligible = _expected_special_use_taxonomy(
            df, population_col=f"population_{YEAR}"
        )["special_use_primary_rate_allowed"]
        special_use = pd.Series(False, index=df.index)
    else:
        residential_eligible = pd.to_numeric(
            df.get("households_total"), errors="coerce"
        ).fillna(0.0).ge(float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR))
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
        * pd.to_numeric(
            overrides[offense] if offense in overrides else df.get(f"expected_count_{offense}"),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        for offense in OFFENSES_7
    )
    published = _count_derived_rate_index(counts=counts, denominator=denominator, publishable=publishable)
    return (
        pd.Series(published["index"], index=df.index, dtype=float),
        float(published["national_rate_per_100k"]),
        publishable,
    )


def _severity_weights_table() -> pd.DataFrame:
    return load_severity_weights(severity_weights_path(RepoPaths.from_repo_root(REPO_ROOT)))


def _count_first_publishable(
    df: pd.DataFrame, offenses: tuple[str, ...] | list[str] | None = None
) -> pd.Series:
    """Mirror of the count-first publication rule: where the COMMON denominator is usable.

    Independent of the producer's copy of the floors on purpose -- these are this file's own
    constants, so a drift in either direction fails the release rather than agreeing with itself.
    No per-offense OPPORTUNITY denominator term appears here; that absence IS the rule.

    `offenses` adds the one component term that lives on this composite's own denominator family:
    a burden may not assert a component whose RESIDENT rate the surface suppresses. Passing none
    gives the plain denominator rule.

    On a typed special-use surface the residential term is the taxonomy's RESIDENT gate, itself
    re-derived here from the surface's extensive inputs. On an ordinary cell that gate reduces to
    `households >= 10`, so the two spellings agree wherever the taxonomy was not asked.
    """
    denominator = pd.to_numeric(df.get(COMMON_DENOMINATOR_COLUMN), errors="coerce").fillna(0.0).clip(lower=0.0)
    components = pd.Series(True, index=df.index)
    for offense in offenses or ():
        column = f"index_{offense}_resident_publishable"
        if column in df.columns:
            components &= df[column].fillna(False).astype(bool)
    if _special_use_taxonomy_lane(df):
        residential = _expected_special_use_taxonomy(df, population_col=f"population_{YEAR}")[
            "special_use_resident_rate_allowed"
        ]
        return (
            residential
            & denominator.gt(0.0)
            & denominator.ge(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
            & components
        )
    households = pd.to_numeric(df.get("households_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
    special_use = (
        df["special_use_tract_flag"].fillna(False).astype(bool)
        if "special_use_tract_flag" in df.columns
        else pd.Series(False, index=df.index)
    )
    return (
        households.ge(float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR))
        & denominator.gt(0.0)
        & denominator.ge(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
        & ~special_use
        & components
    )


def _weighted_count_expected(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)
    for offense, weight in weights.items():
        if float(weight) == 0.0:
            continue
        total = total + float(weight) * pd.to_numeric(
            df.get(f"expected_count_{offense}"), errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
    return total


def _count_first_expected(
    df: pd.DataFrame,
    *,
    weights: dict[str, float],
    publishable: pd.Series,
) -> tuple[pd.Series, float]:
    """100 * (weighted count / common denominator) / reference rate, over the publishable rows."""
    denominator = pd.to_numeric(df.get(COMMON_DENOMINATOR_COLUMN), errors="coerce").fillna(0.0).clip(lower=0.0)
    counts = _weighted_count_expected(df, weights)
    pub = publishable & denominator.gt(0.0)
    denominator_sum = float(denominator.loc[pub].sum())
    count_sum = float(counts.loc[pub].sum())
    reference_rate = RATE_PER_100K * count_sum / denominator_sum if denominator_sum > 0.0 else float("nan")
    index = pd.Series(float("nan"), index=df.index, dtype=float)
    if pd.notna(reference_rate) and reference_rate > 0.0:
        index.loc[pub] = (
            100.0 * (RATE_PER_100K * counts.loc[pub] / denominator.loc[pub]) / reference_rate
        )
    return index.replace([float("inf"), float("-inf")], float("nan")), reference_rate


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
    if field in COMPOSITE_INDEX_COUNT_COLUMNS:
        return COMPOSITE_INDEX_COUNT_COLUMNS[field]
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
        "block_group_cde_exact_sensitivity": _published_surface_path(output_dir, geography="block_group", variant="cde_exact_sensitivity"),
        "tract_cde_exact_sensitivity": _published_surface_path(output_dir, geography="tract", variant="cde_exact_sensitivity"),
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
        surface_fields = _present_fields(
            SPATIAL_STATE_SHARE_INDEX_FIELDS, _surface_column_names(path)
        )
        requested = ["state_fips", *surface_fields]
        try:
            df = pd.read_parquet(path, columns=requested)
        except (KeyError, ValueError):
            df = pd.read_parquet(path)
        fields: dict[str, Any] = {}
        for field in surface_fields:
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
    if not path.exists():
        issues.append(f"spatial_artifacts.boundary_discontinuity: missing tract surface {path}")
        return {"ok": False, "present": False, "path": str(path)}
    boundary_fields = _present_fields(SPATIAL_BOUNDARY_INDEX_FIELDS, _surface_column_names(path))
    columns = ["tract_id", "state_fips", "dominant_eb_jurisdiction_id", *boundary_fields]
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
    for field in boundary_fields:
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
    if not path.exists():
        issues.append(f"spatial_artifacts.tract_flatness: missing tract surface {path}")
        return {"ok": False, "present": False, "path": str(path)}
    available = _surface_column_names(path)
    boundary_fields = _present_fields(SPATIAL_BOUNDARY_INDEX_FIELDS, available)
    expected_cols = sorted(
        {
            col
            for field in boundary_fields
            for col in [_expected_count_column_for_index_field(field)]
            if col is not None and col in available
        }
    )
    columns = ["tract_id", "state_fips", "dominant_eb_jurisdiction_id", *boundary_fields, *expected_cols]
    try:
        tract = pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as exc:
        issues.append(f"spatial_artifacts.tract_flatness: tract surface missing required columns: {exc}")
        return {"ok": False, "present": True, "path": str(path), "required_columns_present": False}

    fields: dict[str, Any] = {}
    ok = True
    for field in boundary_fields:
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
    hotspot_fields = (
        _present_fields(SPATIAL_HOTSPOT_INDEX_FIELDS, _surface_column_names(path))
        if path.exists()
        else list(SPATIAL_HOTSPOT_INDEX_FIELDS)
    )
    columns = [
        "tract_id",
        "state_fips",
        "dominant_eb_jurisdiction_id",
        "special_use_tract_flag",
        "resident_secondary_denominator",
        "crime_density_total",
        *hotspot_fields,
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
    for field in hotspot_fields:
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
    rare_support_tract_path: Path | None = None,
    contract: ReleaseContract = LEGACY_RELEASE_CONTRACT,
) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"{label}: missing file {path}")
        return {"label": label, "path": str(path), "present": False}

    df = pd.read_parquet(path)
    id_col = "block_group_geoid" if geography == "block_group" else "tract_id"
    # The build's manifest decides which assertion set runs; the surface's own columns are held
    # against it rather than consulted in its place.
    evidence = _contract_lane_evidence(df)
    for lane, observed in evidence.items():
        declared = contract.enabled(lane)
        if observed != declared:
            issues.append(
                f"{label}: the build manifest "
                f"{'enables' if declared else 'does not enable'} {lane}, but the surface's columns "
                f"say {'it is on' if observed else 'it is off'}"
            )
    count_first = contract.count_first_composites
    taxonomy_lane = contract.special_use_taxonomy
    exposure_lane = contract.exposure_ensemble
    uncertainty_lane = contract.uncertainty_layer
    expected_cols = _expected_columns(
        geography=geography,
        count_first_composites=count_first,
        special_use_taxonomy=taxonomy_lane,
        exposure_ensemble=exposure_lane,
        uncertainty_layer=uncertainty_lane,
    )
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

    # Rare-offense tract-support policy (docs/archive/2026-09/STATE.md). At block group the murder/rape per-offense
    # index and rate point fields are null by policy; the aggregates that consume a per-offense
    # index or a rare-offense count take those terms at tract support. Load the parent-tract
    # rare-offense primary index and build the within-tract redistributed rare counts so the
    # composites and the harm index recompute exactly from published fields.
    surface_tract_ids = (
        df["tract_id"].astype("string").str.zfill(11)
        if "tract_id" in df.columns
        else df[id_col].astype("string").str.zfill(11)
    )
    rare_nulled_offenses = set(RARE_OFFENSE_TRACT_SUPPORT) if geography == "block_group" else set()
    rare_primary_index_overrides: dict[str, pd.Series] = {}
    rare_harm_count_overrides: dict[str, pd.Series] = {}
    if rare_nulled_offenses:
        rare_harm_count_overrides = _redistributed_rare_offense_counts(df, surface_tract_ids)
        if rare_support_tract_path is not None and rare_support_tract_path.exists():
            tract_cols = ["tract_id", *[f"index_{o}_primary" for o in RARE_OFFENSE_TRACT_SUPPORT]]
            tract_rare = pd.read_parquet(rare_support_tract_path, columns=tract_cols)
            tract_rare["tract_id"] = tract_rare["tract_id"].astype("string").str.zfill(11)
            for offense in RARE_OFFENSE_TRACT_SUPPORT:
                tract_map = dict(zip(tract_rare["tract_id"], tract_rare[f"index_{offense}_primary"], strict=False))
                # Taken verbatim. A parent tract that publishes no index has a SUPPRESSED
                # component, and the composite goes null with it rather than being handed a
                # substitute value the surface itself refuses to show.
                override = surface_tract_ids.map(tract_map).astype("float64")
                # Producer mirror: a parent tract whose denominator is invalid publishes a
                # null index; its tract count / tract raw denominator remains the composite
                # term. Recomputed from published fields.
                rare_primary_index_overrides[offense] = override
        else:
            issues.append(
                f"{label}: rare-offense tract-support index unavailable for aggregate recompute "
                f"(tract surface {rare_support_tract_path})"
            )
    # No volume-offense override. A cell whose own point is suppressed for denominator
    # invalidity has no publishable value for that offense, and the relative scores go null with
    # it -- the same all-or-null rule the producer applies (see allocation.py, the comment above
    # the index-average composites).

    # v2 typed special-use taxonomy, mirrored once for all seven offenses. The type and both
    # publication gates are re-derived from the surface's own extensive inputs; the published
    # columns are then checked against them, so neither side can drift silently.
    expected_taxonomy = (
        _expected_special_use_taxonomy(df, population_col=f"population_{YEAR}")
        if taxonomy_lane
        else None
    )
    if expected_taxonomy is not None:
        for column in (
            "special_use_type",
            "special_use_primary_rate_allowed",
            "special_use_resident_rate_allowed",
            "special_use_candidate_flag",
        ):
            if column not in df.columns:
                issues.append(f"{label}: {column} is missing from a typed special-use surface")
                continue
            if column == "special_use_type":
                mismatch = int(
                    (df[column].astype("string") != expected_taxonomy[column]).sum()
                )
            else:
                mismatch = int(
                    (
                        df[column].fillna(False).astype(bool).to_numpy(dtype=bool)
                        != expected_taxonomy[column].to_numpy(dtype=bool)
                    ).sum()
                )
            if mismatch:
                issues.append(
                    f"{label}: {column} does not match the special-use taxonomy rule for "
                    f"{mismatch} rows"
                )
        # The compatibility flags describe whether the taxonomy itself closes an arm.  They are
        # intentionally true for every type.  Population and exposure sufficiency are validated
        # below through the offense-specific denominator gates and the actual published values.

    # The near-zero-resident opportunity floor. Not a lane of its own: it is the rule the two lanes
    # that created the zero-resident opportunity-rate class have to carry, so it is in force when
    # either is declared. The resident side of the predicate is recomputed once for all seven
    # offenses; the normalizer side varies by offense and lives in the loop below.
    zero_resident_floor_lane = contract.zero_resident_opportunity_floor
    zero_resident_population = pd.to_numeric(
        df.get(f"population_{YEAR}"), errors="coerce"
    ).fillna(0.0).clip(lower=0.0).lt(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
    published_zero_resident_floor = pd.to_numeric(
        df.get(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN), errors="coerce"
    )
    if zero_resident_floor_lane:
        if ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN not in df.columns:
            issues.append(
                f"{label}: {ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN} is missing from a surface "
                "built on a lane that publishes opportunity rates on zero-resident cells"
            )
        elif not bool(
            published_zero_resident_floor.eq(float(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR)).all()
        ):
            issues.append(
                f"{label}: {ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN} does not equal "
                f"{float(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR):g} on every row"
            )
    elif ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN in df.columns:
        issues.append(
            f"{label}: {ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN} is present on a legacy-lane "
            "surface, which must be column-for-column identical to the pre-v2 frame"
        )

    # v2 uncertainty layer, mirrored the same way: the bin label, the tier and every ordering
    # relation are re-derived from the surface's own published fields with this file's own
    # transcription of the break set and the tier rule, so a producer bug cannot be validated by
    # the producer's own constants.
    if uncertainty_lane:
        issues.extend(_uncertainty_layer_issues(df, label=label, contract=contract))

    # v2 exposure-denominator ensembles. The per-offense identities live in the offense loop
    # below; the lane-level naming, reference-total and positivity invariants are here.
    if exposure_lane:
        issues.extend(
            _exposure_ensemble_surface_issues(
                df, label=label, geography=geography, contract=contract
            )
        )

    for offense in OFFENSES_7:
        rare_nulled = offense in rare_nulled_offenses
        count_derived_max_abs[offense] = {}
        for required_col in [
            f"expected_count_{offense}",
            f"rate_{offense}_primary",
            f"index_{offense}_primary",
            f"rate_{offense}_resident",
            f"index_{offense}_resident",
        ]:
            if required_col not in df.columns:
                issues.append(f"{label}: missing required expected-count companion column {required_col}")
        if rare_nulled:
            # Positive assertion of the support rule: every published per-offense index/rate point
            # field (and its confidence interval) for this rare offense is null at block group.
            populated = [
                field
                for field in _rare_offense_point_fields(offense)
                if field in df.columns and pd.to_numeric(df[field], errors="coerce").notna().any()
            ]
            if populated:
                issues.append(
                    f"{label}: {offense} per-offense index/rate fields must be null at block-group "
                    f"support but are populated: {populated}"
                )
        value = pd.to_numeric(df.get(f"index_{offense}_primary"), errors="coerce")
        rate = pd.to_numeric(df.get(f"rate_{offense}_primary"), errors="coerce")
        count = pd.to_numeric(df.get(f"expected_count_{offense}"), errors="coerce").fillna(0.0).clip(lower=0.0)
        households = pd.to_numeric(df.get("households_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
        expected_non_residential = households.lt(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
        # Which gate decides the arm: the legacy household rule, or the cell's type. Off the typed
        # lane the two are identical by construction, which is what makes this one expression.
        expected_primary_allowed = (
            expected_taxonomy["special_use_primary_rate_allowed"]
            if expected_taxonomy is not None
            else ~expected_non_residential
        )
        expected_resident_allowed = (
            expected_taxonomy["special_use_resident_rate_allowed"]
            if expected_taxonomy is not None
            else ~expected_non_residential
        )
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
        # Which surface the offense's rate was divided by. Off the exposure lane it is the legacy
        # hard max; on it, the named per-offense normalizer, asserted in
        # `_exposure_ensemble_surface_issues` above so the identity is stated once per offense.
        if (
            offense in PERSON_EXPOSURE_FLOOR_OFFENSES
            and not exposure_lane
            and "exposure_proxy_2024" in df.columns
        ):
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
            # Typed lane: the premises floor is an EXPOSURE floor, not a special-use fact, so it
            # moves to the insufficient-exposure term below and reports itself as one.
            expected_special_suppressed = (
                pd.Series(False, index=df.index)
                if expected_taxonomy is not None
                else expected_special_flag | denominator.lt(BURGLARY_PREMISES_DENOMINATOR_FLOOR)
            )
            expected_burglary_premises_floor = denominator.lt(BURGLARY_PREMISES_DENOMINATOR_FLOOR)
        elif offense == "motor_vehicle_theft":
            expected_vehicle_denominator = pd.to_numeric(
                df.get("vehicle_exposure_2024"), errors="coerce"
            ).fillna(0.0).clip(lower=0.0)
            denominator_delta = (denominator - expected_vehicle_denominator).abs().max()
            if pd.notna(denominator_delta) and float(denominator_delta) > 1e-9:
                issues.append(f"{label}: primary_denominator_motor_vehicle_theft does not equal vehicle_exposure_2024")
            expected_special_suppressed = (
                pd.Series(False, index=df.index)
                if expected_taxonomy is not None
                else expected_special_flag
            )
            expected_burglary_premises_floor = pd.Series(False, index=df.index)
        else:
            expected_special_suppressed = (
                pd.Series(False, index=df.index)
                if expected_taxonomy is not None
                else expected_special_flag
            )
            expected_burglary_premises_floor = pd.Series(False, index=df.index)
        expected_special_suppressed = expected_special_suppressed.fillna(False).astype(bool)
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES:
            expected_insufficient_exposure = denominator.lt(PERSON_EXPOSURE_DENOMINATOR_FLOOR)
        elif offense == "motor_vehicle_theft":
            expected_insufficient_exposure = denominator.lt(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR)
        else:
            expected_insufficient_exposure = pd.Series(False, index=df.index)
        expected_insufficient_exposure = expected_insufficient_exposure.fillna(False).astype(bool)
        if expected_taxonomy is not None:
            expected_insufficient_exposure = (
                expected_insufficient_exposure
                | expected_burglary_premises_floor.fillna(False).astype(bool)
            )
        if zero_resident_floor_lane:
            # The near-zero-resident opportunity floor, recomputed from the surface's own published
            # resident population and the offense's own published normalizer. Folded into the
            # insufficient-exposure term exactly where the producer folds it, so every downstream
            # expectation in this loop -- publishability, the suppressed flag, the reason and the
            # display mode -- inherits it without a second spelling of the rule.
            expected_insufficient_exposure = expected_insufficient_exposure | (
                zero_resident_population
                & denominator.lt(float(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR))
            )
        # --- ambient-blind custom footprint, recomputed from published fields ---------------
        # allocation.py: a MAJORITY of the offense's mass through a custom-footprint overlap
        # layer, NO ambient lift on the exposure denominator, and an implied resident rate above
        # 3x the national resident rate over the rows publishable under the pre-existing floors.
        footprint_count = pd.to_numeric(
            df.get(f"footprint_derived_count_{offense}"), errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        expected_footprint_share = pd.Series(
            np.where(count.to_numpy() > 0.0, footprint_count.to_numpy() / np.where(count > 0.0, count, 1.0), 0.0),
            index=df.index,
            dtype=float,
        ).clip(0.0, 1.0)
        published_footprint_share = pd.to_numeric(
            df.get(f"footprint_derived_count_share_{offense}"), errors="coerce"
        )
        if f"footprint_derived_count_share_{offense}" in df.columns:
            share_delta = (published_footprint_share - expected_footprint_share).abs().max()
            if pd.notna(share_delta) and float(share_delta) > 1e-9:
                issues.append(
                    f"{label}: footprint_derived_count_share_{offense} does not equal "
                    f"footprint_derived_count_{offense} / expected_count_{offense}"
                )
        resident_denominator_for_footprint = pd.to_numeric(
            df.get("resident_secondary_denominator"), errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        exposure_for_footprint = pd.to_numeric(df.get("exposure_proxy_2024"), errors="coerce").fillna(0.0)
        # `resident_secondary_denominator` IS the resident population on both surfaces, and it is
        # present on every row, so it stands in for the year-suffixed population column here.
        population_for_footprint = resident_denominator_for_footprint
        transient_col = f"transient_exposure_likely_{offense}"
        observed_transient_flag = (
            df[transient_col].fillna(False).astype(bool)
            if transient_col in df.columns
            else pd.Series(False, index=df.index, dtype=bool)
        )
        published_transient_ratio = pd.to_numeric(
            df.get("transient_exposure_daytime_to_resident_ratio"), errors="coerce"
        )
        expected_transient_ratio = exposure_for_footprint / population_for_footprint.where(
            population_for_footprint.gt(0.0)
        )
        ratio_delta, ratio_null_mismatch = _max_abs_pair_delta(
            published_transient_ratio,
            expected_transient_ratio,
        )
        if ratio_null_mismatch or ratio_delta > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: transient_exposure_daytime_to_resident_ratio is not "
                f"exposure_proxy_2024 / resident_secondary_denominator "
                f"(max abs diff {ratio_delta:.3e}, null mismatches {ratio_null_mismatch})"
            )
        expected_transient_flag = (
            population_for_footprint.gt(0.0)
            & expected_transient_ratio.ge(TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO)
            & pd.to_numeric(df.get(f"index_{offense}_resident"), errors="coerce").ge(
                TRANSIENT_EXPOSURE_RESIDENT_INDEX_THRESHOLD
            )
        ) | (
            population_for_footprint.gt(0.0)
            & pd.to_numeric(df.get("commercial_premises_total"), errors="coerce")
            .fillna(0.0)
            .ge(TRANSIENT_COMMERCIAL_PREMISES_FLOOR)
            & expected_transient_ratio.le(TRANSIENT_COMMERCIAL_RATIO_CEILING)
            & value.ge(TRANSIENT_COMMERCIAL_PRIMARY_INDEX_THRESHOLD)
        )
        if rare_nulled:
            expected_transient_flag = pd.Series(False, index=df.index, dtype=bool)
        transient_flag_mismatch = int(
            (
                observed_transient_flag.to_numpy(dtype=bool)
                != expected_transient_flag.to_numpy(dtype=bool)
            ).sum()
        )
        if transient_flag_mismatch:
            issues.append(
                f"{label}: {transient_col} does not match the advisory transient-exposure "
                f"diagnostic for {transient_flag_mismatch} rows"
            )
        resident_floor_hit = resident_denominator_for_footprint.lt(
            PERSON_EXPOSURE_DENOMINATOR_FLOOR
        )
        expected_resident_floor_hit = resident_floor_hit.fillna(False).astype(bool)
        baseline_resident_publishable = (
            expected_resident_allowed
            & resident_denominator_for_footprint.gt(0.0)
            & ~expected_special_suppressed
            & ~expected_resident_floor_hit
        )
        baseline_resident = _count_derived_rate_index(
            counts=count,
            denominator=resident_denominator_for_footprint,
            publishable=baseline_resident_publishable,
        )
        baseline_national = float(baseline_resident["national_rate_per_100k"])
        # Mirrors allocation.py: the rate is measured on the count's exact-Poisson LOWER bound,
        # so a fractional modelled count cannot carry the claim on its own.
        count_lower, _count_upper = _poisson_count_interval(count)
        measurable = baseline_resident_publishable & resident_denominator_for_footprint.gt(0.0)
        conservative_rate = pd.Series(float("nan"), index=df.index, dtype=float)
        conservative_rate.loc[measurable] = (
            RATE_PER_100K * count_lower.loc[measurable] / resident_denominator_for_footprint.loc[measurable]
        )
        baseline_ratio = pd.Series(float("nan"), index=df.index, dtype=float)
        if np.isfinite(baseline_national) and baseline_national > 0:
            baseline_ratio = conservative_rate / baseline_national
        expected_footprint_ambient_missing = (
            expected_footprint_share.gt(FOOTPRINT_DERIVED_MASS_SHARE_FLOOR)
            & ~exposure_for_footprint.gt(population_for_footprint)
            & baseline_ratio.ge(AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO)
        ).fillna(False)
        if offense not in PERSON_EXPOSURE_FLOOR_OFFENSES:
            expected_footprint_ambient_missing = pd.Series(False, index=df.index)
        flag_col = f"footprint_ambient_exposure_missing_{offense}"
        if flag_col in df.columns:
            published_flag = df[flag_col].fillna(False).astype(bool)
            flag_mismatch = int(
                (published_flag.to_numpy(dtype=bool) != expected_footprint_ambient_missing.to_numpy(dtype=bool)).sum()
            )
            if flag_mismatch:
                issues.append(
                    f"{label}: {flag_col} does not match the ambient-blind footprint rule for "
                    f"{flag_mismatch} rows"
                )
        else:
            issues.append(f"{label}: {flag_col} is missing from the surface")

        # Transient exposure is an advisory diagnostic. It must not change point values,
        # publication flags, modes, or denominator reasons.
        expected_primary_publishable = (
            expected_primary_allowed
            & denominator.gt(0.0)
            & ~expected_denominator_invalid
            & ~expected_special_suppressed
            & ~expected_insufficient_exposure
            & ~expected_footprint_ambient_missing
        )
        # `baseline_resident_publishable` above is the producer's own baseline (arm open, resident
        # denominator positive, no special-use suppression, above the resident floor); the resident
        # arm's publishability is that, less the two rules the footprint block needed it for.
        expected_resident_publishable = (
            baseline_resident_publishable & ~expected_denominator_invalid
        )
        expected_suppressed = (
            (~expected_primary_allowed)
            | expected_denominator_invalid
            | expected_special_suppressed
            | expected_insufficient_exposure
            | expected_footprint_ambient_missing
        )
        if rare_nulled:
            # Block-group murder/rape point fields are intentionally absent.  The
            # publishability flags describe the payload, not latent denominator
            # eligibility; tract-support fields carry the publishable point value.
            expected_suppressed = pd.Series(True, index=df.index, dtype=bool)
        publishable = (
            df[f"primary_index_publishable_{offense}"].astype(bool)
            if f"primary_index_publishable_{offense}" in df.columns
            else pd.Series(False, index=df.index)
        )
        transient_eligible_but_hidden = expected_transient_flag & expected_primary_publishable & ~publishable
        if bool(transient_eligible_but_hidden.any()):
            issues.append(
                f"{label}: advisory {transient_col} hides the primary rate on "
                f"{int(transient_eligible_but_hidden.sum())} otherwise eligible rows"
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
        expected_arm_open = (
            expected_primary_allowed if expected_taxonomy is not None else ~expected_non_residential
        )
        # Typed lane: a cell the TYPE closed reports `special_use` last of all, and the household
        # rule may only claim a cell that does not publish.
        expected_typed_suppressed = (
            expected_taxonomy["special_use_type"].ne(SPECIAL_USE_ORDINARY) & ~expected_primary_allowed
            if expected_taxonomy is not None
            else pd.Series(False, index=df.index)
        )
        expected_footprint_mode = (
            expected_footprint_ambient_missing
            & expected_arm_open
            & ~expected_denominator_invalid
            & ~expected_special_suppressed
            & ~expected_insufficient_exposure
            & ~expected_typed_suppressed
        )
        expected_insufficient_mode = (
            expected_insufficient_exposure
            & expected_arm_open
            & ~expected_denominator_invalid
            & ~expected_special_suppressed
            & ~expected_typed_suppressed
        )
        # The household flag is descriptive on typed surfaces.  It remains a publication rule
        # only for legacy surfaces that predate the denominator-only contract.
        expected_non_residential_claim = (
            pd.Series(False, index=df.index, dtype=bool)
            if expected_taxonomy is not None
            else expected_non_residential
        )
        expected_non_residential_mode = (
            pd.Series(False, index=df.index, dtype=bool)
            if expected_taxonomy is not None
            else expected_non_residential
        )
        if int((mode.eq("non_residential") != expected_non_residential_mode).sum()):
            issues.append(f"{label}: estimate_mode_{offense}=non_residential does not match the household rule")
        expected_special_mode = (
            expected_typed_suppressed
            if expected_taxonomy is not None
            else expected_special_suppressed & ~expected_non_residential & ~expected_denominator_invalid
        )
        if int((mode.eq("special_use") != expected_special_mode).sum()):
            issues.append(f"{label}: estimate_mode_{offense}=special_use does not match the per-offense special-use rule")
        # `insufficient_exposure` is the DISPLAY code for two mechanisms -- the plain denominator
        # floor and the ambient-blind custom footprint. `denominator_reason` separates them; the
        # viewer's five codes cover the surface without a new one (see docs/PIPELINE.md).
        expected_insufficient_display_mode = expected_insufficient_mode | expected_footprint_mode
        if int((mode.eq("insufficient_exposure") != expected_insufficient_display_mode).sum()):
            issues.append(
                f"{label}: estimate_mode_{offense}=insufficient_exposure does not match the "
                "offense denominator floor or the ambient-blind footprint rule"
            )
        if (
            offense not in PERSON_EXPOSURE_FLOOR_OFFENSES
            and offense != "motor_vehicle_theft"
            and bool(
                (
                    mode.eq("insufficient_exposure")
                    & ~expected_footprint_mode
                    # Typed lane only: burglary's 10-premises floor is an EXPOSURE floor, so it
                    # moves out of the special-use term and reports itself as one. Off the lane
                    # this term is an all-false series and the mask is unchanged.
                    & ~expected_insufficient_mode
                ).any()
            )
        ):
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
        if (
            expected_taxonomy is None
            and resident_denominator_reason.loc[expected_special_mode].ne("special_use").any()
        ):
            issues.append(f"{label}: resident_denominator_reason_{offense} does not flag special_use rows")
        if df.loc[expected_special_mode, f"expected_count_{offense}"].isna().any():
            issues.append(f"{label}: expected_count_{offense} has nulls in special_use-suppressed rows")
        # On typed surfaces neither the household flag nor the taxonomy can replace the actual
        # denominator reason.  Legacy surfaces retain their original precedence.
        expected_insufficient_reason = (
            expected_insufficient_mode
            & ~expected_non_residential_claim
            & ~expected_typed_suppressed
            if expected_taxonomy is not None
            else expected_insufficient_mode
        )
        if denominator_reason.loc[expected_insufficient_reason].ne("insufficient_exposure").any():
            issues.append(f"{label}: denominator_reason_{offense} does not flag insufficient_exposure rows")
        if df.loc[expected_insufficient_mode, f"expected_count_{offense}"].isna().any():
            issues.append(f"{label}: expected_count_{offense} has nulls in insufficient_exposure-suppressed rows")
        expected_resident_typed_suppressed = (
            expected_taxonomy["special_use_type"].ne(SPECIAL_USE_ORDINARY)
            & ~expected_resident_allowed
            if expected_taxonomy is not None
            else pd.Series(False, index=df.index)
        )
        expected_non_residential_reason = (
            expected_non_residential_claim & ~expected_typed_suppressed
            if expected_taxonomy is not None
            else expected_non_residential_mode
        )
        if denominator_reason.loc[expected_non_residential_reason].ne("non_residential").any():
            issues.append(f"{label}: denominator_reason_{offense} does not flag non_residential rows")
        expected_resident_non_residential_claim = (
            pd.Series(False, index=df.index, dtype=bool)
            if expected_taxonomy is not None
            else expected_non_residential
        )
        expected_resident_non_residential_mode = (
            expected_resident_non_residential_claim & ~expected_resident_typed_suppressed
            if expected_taxonomy is not None
            else expected_non_residential
        )
        if resident_denominator_reason.loc[expected_resident_non_residential_mode].ne("non_residential").any():
            issues.append(f"{label}: resident_denominator_reason_{offense} does not flag non_residential rows")
        if expected_taxonomy is not None:
            if resident_denominator_reason.loc[expected_resident_typed_suppressed].ne("special_use").any():
                issues.append(
                    f"{label}: resident_denominator_reason_{offense} does not flag typed special-use rows"
                )
        if denominator_reason.loc[expected_footprint_mode].ne(INSUFFICIENT_AMBIENT_EXPOSURE_REASON).any():
            issues.append(
                f"{label}: denominator_reason_{offense} does not flag ambient-blind footprint rows "
                f"as {INSUFFICIENT_AMBIENT_EXPOSURE_REASON}"
            )
        if df.loc[expected_footprint_mode, f"expected_count_{offense}"].isna().any():
            issues.append(
                f"{label}: expected_count_{offense} has nulls in ambient-blind-footprint-suppressed rows"
            )
        real_housing_suppressed = (
            suppressed
            & households.ge(50.0)
            & ~expected_denominator_invalid
            & ~expected_special_suppressed
            & ~expected_insufficient_exposure
            # An ambient-blind footprint cell DOES carry households (a casino parcel with a few
            # hundred residents is the whole point of the class), so it is a declared exception
            # to "no populated cell is suppressed" rather than a violation of it.
            & ~expected_footprint_ambient_missing
            # So is a typed special-use cell: a prison tract carries hundreds of "households" in
            # the ACS sense and is still not a neighbourhood with a residents-at-risk rate.
            & ~expected_typed_suppressed
        )
        if not rare_nulled and bool(real_housing_suppressed.any()):
            issues.append(
                f"{label}: primary_index_suppressed_{offense} suppresses "
                f"{int(real_housing_suppressed.sum())} rows with households_total >= 50"
            )
        if not rare_nulled and value.loc[publishable].isna().any():
            issues.append(f"{label}: index_{offense}_primary has nulls on published rows")
        if not rare_nulled and rate.loc[publishable].isna().any():
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
        # The count-derived rate/index identity is only asserted where the point value is
        # published: the rare offenses carry null block-group point fields by policy (their
        # count-derived value lives at tract support), so skip the identity for them here — the
        # positive null assertion above and the tract-surface identity together cover them.
        if not rare_nulled:
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
            count_derived_max_abs[offense].update(
                {
                    "rate": primary_rate_delta,
                    "index": primary_index_delta,
                    "raw_rate": raw_rate_delta,
                }
            )
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
        expected_resident_insufficient_exposure = resident_denominator.lt(
            PERSON_EXPOSURE_DENOMINATOR_FLOOR
        )
        expected_resident_insufficient_exposure = expected_resident_insufficient_exposure.fillna(False).astype(bool)
        # Which gate opens the resident arm: the legacy household rule, or the cell's TYPE. Off the
        # typed lane the two are identical by construction.
        expected_resident_arm_closed = (
            ~expected_resident_allowed if expected_taxonomy is not None else expected_non_residential
        )
        expected_resident_suppressed = (
            expected_resident_arm_closed
            | expected_denominator_invalid
            | expected_special_suppressed
            | expected_resident_insufficient_exposure
        )
        if rare_nulled:
            expected_resident_suppressed = pd.Series(True, index=df.index, dtype=bool)
        expected_resident_insufficient_reason = (
            expected_resident_insufficient_exposure
            & ~expected_resident_arm_closed
            & ~expected_denominator_invalid
            & ~expected_special_suppressed
        )
        if expected_taxonomy is not None:
            # The producer's order on this vocabulary: the exposure floor, then the ambient-blind
            # footprint reason, then the household floor, then the type.
            expected_resident_insufficient_reason = (
                expected_resident_insufficient_exposure
                & ~expected_resident_arm_closed
                & ~expected_denominator_invalid
                & ~expected_special_suppressed
                & ~expected_footprint_mode
                & ~expected_resident_non_residential_claim
                & ~expected_resident_typed_suppressed
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
        resident_publishable_col = f"index_{offense}_resident_publishable"
        observed_resident_publishable = (
            df[resident_publishable_col].fillna(False).astype(bool)
            if resident_publishable_col in df.columns
            else pd.Series(False, index=df.index, dtype=bool)
        )
        resident_publishable_mismatch = int(
            (
                observed_resident_publishable.to_numpy(dtype=bool)
                != resident_publishable.to_numpy(dtype=bool)
            ).sum()
        )
        if resident_publishable_mismatch:
            issues.append(
                f"{label}: {resident_publishable_col} is not exactly eligible and resident "
                f"denominator > 0 for {resident_publishable_mismatch} rows"
            )
        transient_resident_eligible_but_hidden = (
            expected_transient_flag
            & expected_resident_publishable
            & ~observed_resident_publishable
        )
        if bool(transient_resident_eligible_but_hidden.any()):
            issues.append(
                f"{label}: advisory {transient_col} hides the resident rate on "
                f"{int(transient_resident_eligible_but_hidden.sum())} otherwise eligible rows"
            )
        if not rare_nulled and resident_value.loc[resident_publishable].isna().any():
            issues.append(f"{label}: index_{offense}_resident has nulls on published rows")
        if not rare_nulled and resident_rate.loc[resident_publishable].isna().any():
            issues.append(f"{label}: rate_{offense}_resident has nulls on published rows")
        if resident_value.loc[~resident_publishable].notna().any():
            issues.append(f"{label}: index_{offense}_resident is populated on resident non-publishable-denominator rows")
        if resident_rate.loc[~resident_publishable].notna().any():
            issues.append(f"{label}: rate_{offense}_resident is populated on resident non-publishable-denominator rows")
        floor = PERSON_EXPOSURE_DENOMINATOR_FLOOR
        low_resident = resident_denominator.lt(floor)
        leaked_resident_cols = [
            col
            for col in [f"rate_{offense}_resident", f"index_{offense}_resident"]
            if col in df.columns and pd.to_numeric(df.loc[low_resident, col], errors="coerce").notna().any()
        ]
        if leaked_resident_cols:
            issues.append(
                f"{label}: {offense} published resident rate/index fields below "
                f"{floor:g} residents: {leaked_resident_cols}"
            )
        if not rare_nulled:
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
    # Which arm each composite hangs off, so the "no composite in a cell whose arm is shut" gate
    # asks the right question on the typed lane. Empty off it, where every arm is the household
    # floor and `_surface_result` falls back to `non_residential_flag` unchanged.
    composite_arm_closed: dict[str, pd.Series] = {}
    if expected_taxonomy is not None:
        resident_arm_closed = ~expected_taxonomy["special_use_resident_rate_allowed"]
        primary_arm_closed = ~expected_taxonomy["special_use_primary_rate_allowed"]
        for name in (
            EVENT_BURDEN_COLUMN,
            PERSONAL_BURDEN_COLUMN,
            PROPERTY_BURDEN_COLUMN,
            HARM_BURDEN_COLUMN,
            "index_total_part1_resident",
            "index_personal_part1_resident",
            "index_property_part1_resident",
        ):
            composite_arm_closed[name] = resident_arm_closed
        for name in (
            "multi_offense_relative_score_event_weighted",
            "multi_offense_relative_score_equal_offense",
            "multi_offense_relative_score_personal_event_weighted",
            "multi_offense_relative_score_property_event_weighted",
            "index_total_primary_event_weighted",
            "index_total_equal_offense",
            # The legacy harm index normalises on person exposure, so its arm is the primary one.
            "index_total_harm",
        ):
            composite_arm_closed[name] = primary_arm_closed
    if count_first:
        # --- v2 count-first composite lane -------------------------------------------------
        # Every field here recomputes from published counts plus the resident denominator, and
        # from nothing else. The mirror deliberately reads no per-offense denominator, no
        # estimate mode and no PRIMARY publishability flag: if any of those were load-bearing
        # the composite would not be count-first. The resident-arm component gate it does read
        # is on the composite's own denominator.
        composite_support = "block_group" if geography == "block_group" else "tract"
        burden_publishable = _count_first_publishable(
            df, gated_component_offenses(tuple(OFFENSES_7), composite_support)
        )
        severity = severity_vector(_severity_weights_table(), primary_vector_id(_severity_weights_table()))
        burden_specs = {
            EVENT_BURDEN_COLUMN: OFFENSES_7,
            PERSONAL_BURDEN_COLUMN: PERSONAL_OFFENSES,
            PROPERTY_BURDEN_COLUMN: PROPERTY_OFFENSES,
        }
        for field, offenses in burden_specs.items():
            actual = pd.to_numeric(df.get(field), errors="coerce")
            field_publishable = _count_first_publishable(
                df, gated_component_offenses(tuple(offenses), composite_support)
            )
            expected, reference_rate = _count_first_expected(
                df, weights={offense: 1.0 for offense in offenses}, publishable=field_publishable
            )
            max_abs, null_mismatch = _max_abs_pair_delta(actual, expected)
            aggregate_max_abs[field] = max_abs
            if null_mismatch or max_abs > COUNT_DERIVED_TOLERANCE:
                issues.append(
                    f"{label}: {field} is not the count-first burden over the common denominator "
                    f"(reference_rate_per_100k {reference_rate:.12g}, max abs diff {max_abs:.3e}, "
                    f"null mismatches {null_mismatch})"
                )
            if int((actual.notna() != field_publishable).sum()):
                issues.append(
                    f"{label}: {field} nulls do not match the common-denominator publication rule"
                )

        harm_count_actual = pd.to_numeric(df.get(HARM_WEIGHTED_COUNT_COLUMN), errors="coerce")
        harm_count_expected = _weighted_count_expected(df, severity)
        harm_count_max_abs, harm_count_null_mismatch = _max_abs_pair_delta(
            harm_count_actual, harm_count_expected
        )
        aggregate_max_abs[HARM_WEIGHTED_COUNT_COLUMN] = harm_count_max_abs
        if harm_count_null_mismatch or harm_count_max_abs > COUNT_DERIVED_TOLERANCE:
            issues.append(
                f"{label}: {HARM_WEIGHTED_COUNT_COLUMN} is not the severity-weighted sum of the "
                f"seven published counts (max abs diff {harm_count_max_abs:.3e}, null mismatches "
                f"{harm_count_null_mismatch})"
            )

        harm_actual = pd.to_numeric(df.get(HARM_BURDEN_COLUMN), errors="coerce")
        if geography == "block_group":
            # The tract-support rule, enforced rather than documented: roughly half the harm mass
            # sits on murder and rape, whose block-group points this surface does not publish.
            populated = int(harm_actual.notna().sum())
            aggregate_max_abs[HARM_BURDEN_COLUMN] = 0.0
            if populated:
                issues.append(
                    f"{label}: {HARM_BURDEN_COLUMN} is published on {populated} block groups; the "
                    "harm composite carries tract support and coarser only"
                )
        else:
            harm_expected, harm_reference_rate = _count_first_expected(
                df, weights=severity, publishable=burden_publishable
            )
            harm_max_abs, harm_null_mismatch = _max_abs_pair_delta(harm_actual, harm_expected)
            aggregate_max_abs[HARM_BURDEN_COLUMN] = harm_max_abs
            if harm_null_mismatch or harm_max_abs > COUNT_DERIVED_TOLERANCE:
                issues.append(
                    f"{label}: {HARM_BURDEN_COLUMN} is not the count-first harm burden over the "
                    f"common denominator (reference_rate_per_100k {harm_reference_rate:.12g}, "
                    f"max abs diff {harm_max_abs:.3e}, null mismatches {harm_null_mismatch})"
                )
            if int((harm_actual.notna() != burden_publishable).sum()):
                issues.append(
                    f"{label}: {HARM_BURDEN_COLUMN} nulls do not match the common-denominator "
                    "publication rule"
                )
        harm_publishable = burden_publishable if geography != "block_group" else pd.Series(
            False, index=df.index
        )

        # The two surviving index averages are unchanged arithmetic under honest names.
        renamed_composite_specs = {
            "multi_offense_relative_score_event_weighted": (
                list(OFFENSES_7),
                _national_expected_count_weights(df, list(OFFENSES_7)),
            ),
            "multi_offense_relative_score_equal_offense": (
                list(OFFENSES_7),
                {offense: 1.0 for offense in OFFENSES_7},
            ),
            PERSONAL_RELATIVE_SCORE_COLUMN: (
                list(PERSONAL_OFFENSES),
                _national_expected_count_weights(df, list(PERSONAL_OFFENSES)),
            ),
            PROPERTY_RELATIVE_SCORE_COLUMN: (
                list(PROPERTY_OFFENSES),
                _national_expected_count_weights(df, list(PROPERTY_OFFENSES)),
            ),
        }
        relative_publishable: dict[str, pd.Series] = {}
        for field, (offenses, weights) in renamed_composite_specs.items():
            actual = pd.to_numeric(df.get(field), errors="coerce")
            expected, publishable = _primary_composite_expected(
                df,
                offenses=offenses,
                weights=weights,
                index_overrides=rare_primary_index_overrides,
            )
            relative_publishable[field] = publishable
            max_abs, null_mismatch = _max_abs_pair_delta(actual, expected)
            aggregate_max_abs[field] = max_abs
            if null_mismatch or max_abs > COUNT_DERIVED_TOLERANCE:
                issues.append(
                    f"{label}: {field} is not the all-or-null primary-index relative score "
                    f"(max abs diff {max_abs:.3e}, null mismatches {null_mismatch})"
                )
            if int((actual.notna() != publishable).sum()):
                issues.append(
                    f"{label}: {field} does not match its component all-or-null publishability"
                )
        legacy_present = [
            field for field in AGGREGATE_INDEX_FIELDS if field in df.columns
        ]
        if legacy_present:
            issues.append(
                f"{label}: count-first composite surface still carries superseded composite "
                f"fields {legacy_present}"
            )
        return _surface_result(
            label=label,
            path=path,
            df=df,
            states=states,
            duplicate_count=duplicate_count,
            personal_diff=personal_diff,
            property_diff=property_diff,
            total_diff=total_diff,
            density_max_abs=density_max_abs,
            count_derived_max_abs=count_derived_max_abs,
            aggregate_max_abs=aggregate_max_abs,
            urban_stratum_counts=urban_stratum_counts,
            confidence_tier_counts=confidence_tier_counts,
            domain_overlap_ranges=domain_overlap_ranges,
            issues=issues,
            composite_fields=COUNT_FIRST_INDEX_FIELDS,
            mvt_must_null_fields=[
                "multi_offense_relative_score_event_weighted",
                "multi_offense_relative_score_equal_offense",
                PROPERTY_RELATIVE_SCORE_COLUMN,
            ],
            mvt_must_publish=[
                (EVENT_BURDEN_COLUMN, burden_publishable),
                (PROPERTY_BURDEN_COLUMN, burden_publishable),
                (HARM_BURDEN_COLUMN, harm_publishable),
                (PERSONAL_RELATIVE_SCORE_COLUMN, relative_publishable[PERSONAL_RELATIVE_SCORE_COLUMN]),
            ],
            composite_arm_closed=composite_arm_closed,
        )

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
        # At block group the murder/rape components enter at tract support (their block-group index
        # is null); at tract the composite uses native per-offense indices.
        expected, publishable = _primary_composite_expected(
            df,
            offenses=list(OFFENSES_7),
            weights=weights,
            index_overrides=rare_primary_index_overrides,
        )
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
    # At block group the harm index takes murder/rape as the within-tract redistributed tract count
    # (five volume offenses at full block-group resolution); at tract it uses native counts.
    harm_expected, harm_national_rate, harm_publishable = _harm_total_expected(
        df, count_overrides=rare_harm_count_overrides
    )
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

    _, _, legacy_personal_publishable = _resident_part1_expected(df, offenses=list(PERSONAL_OFFENSES))
    return _surface_result(
        label=label,
        path=path,
        df=df,
        states=states,
        duplicate_count=duplicate_count,
        personal_diff=personal_diff,
        property_diff=property_diff,
        total_diff=total_diff,
        density_max_abs=density_max_abs,
        count_derived_max_abs=count_derived_max_abs,
        aggregate_max_abs=aggregate_max_abs,
        urban_stratum_counts=urban_stratum_counts,
        confidence_tier_counts=confidence_tier_counts,
        domain_overlap_ranges=domain_overlap_ranges,
        issues=issues,
        composite_fields=AGGREGATE_INDEX_FIELDS,
        mvt_must_null_fields=[
            "index_total_part1_resident",
            "index_property_part1_resident",
            "index_total_primary_event_weighted",
            "index_total_equal_offense",
        ],
        mvt_must_publish=[
            ("index_personal_part1_resident", legacy_personal_publishable),
            # index_total_harm consumes counts, not indices: MVT-invalid rows must still publish
            # it wherever person exposure is publishable.
            ("index_total_harm", harm_publishable),
        ],
        composite_arm_closed=composite_arm_closed,
    )


def _surface_result(
    *,
    label: str,
    path: Path,
    df: pd.DataFrame,
    states: set[str],
    duplicate_count: int,
    personal_diff: pd.Series,
    property_diff: pd.Series,
    total_diff: pd.Series,
    density_max_abs: dict[str, float],
    count_derived_max_abs: dict[str, dict[str, float]],
    aggregate_max_abs: dict[str, float],
    urban_stratum_counts: dict[str, int],
    confidence_tier_counts: dict[str, dict[str, int]],
    domain_overlap_ranges: dict[str, dict[str, float | None]],
    issues: list[str],
    composite_fields: list[str],
    mvt_must_null_fields: list[str],
    mvt_must_publish: list[tuple[str, pd.Series]],
    composite_arm_closed: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    """The composite-lane-independent tail of `_check_surface`.

    `mvt_must_publish` is the load-bearing asymmetry: a field that consumes COUNTS must survive a
    cell whose vehicle denominator is invalid, and a field that averages per-offense INDEXES must
    not. Both lanes assert it; only the field names differ.

    `composite_arm_closed` names, per composite field, the mask under which that composite's ARM is
    shut. Off the typed special-use lane every arm is shut by the same household floor, which is
    what `non_residential_flag` records; on it a campus publishes both arms with fewer than ten
    households and an employment district publishes only the exposure one, so the blanket household
    test would read a correct surface as a defect.
    """
    excluded_present = sorted(states & RELEASE_EXCLUDED_STATE_FIPS)
    if "non_residential_flag" in df.columns:
        non_residential = df["non_residential_flag"].fillna(False).astype(bool)
        arm_closed = composite_arm_closed or {}
        aggregate_populated_non_residential = [
            name
            for name in composite_fields
            if name in df.columns
            and pd.to_numeric(
                df.loc[arm_closed.get(name, non_residential), name], errors="coerce"
            ).notna().any()
        ]
        if aggregate_populated_non_residential:
            issues.append(
                f"{label}: aggregate indexes populated in non-residential cells: "
                f"{aggregate_populated_non_residential}"
            )
    mvt_invalid_col = "primary_denominator_invalid_motor_vehicle_theft"
    if mvt_invalid_col in df.columns:
        mvt_invalid = df[mvt_invalid_col].fillna(False).astype(bool)
        populated = [
            field
            for field in mvt_must_null_fields
            if field in df.columns and pd.to_numeric(df.loc[mvt_invalid, field], errors="coerce").notna().any()
        ]
        if populated:
            issues.append(f"{label}: MVT-invalid rows populate aggregate fields that require MVT: {populated}")
        for field, publishable in mvt_must_publish:
            if field not in df.columns:
                continue
            actual = pd.to_numeric(df[field], errors="coerce")
            allowed = mvt_invalid & publishable
            if actual.loc[allowed].isna().any():
                issues.append(f"{label}: MVT-invalid rows suppress publishable {field}")

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


def _load_murder_tract_posterior_calibration(
    *, issues: list[str], holdout_run: bool = False
) -> dict[str, Any]:
    path = REPO_MURDER_TRACT_POSTERIOR_CALIBRATION
    if not path.exists():
        issues.append(f"missing murder tract-posterior calibration artifact: {path}")
        return {"present": False, "path": str(path)}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - preserve the underlying diagnostic.
        issues.append(f"could not read murder tract-posterior calibration artifact {path}: {exc}")
        return {"present": True, "path": str(path), "readable": False}

    if int(data.get("year") or 0) != YEAR:
        issues.append(f"murder tract-posterior calibration year {data.get('year')} != {YEAR}")
    if data.get("offense") != "murder" or data.get("support") != "census_tract":
        issues.append(
            "murder tract-posterior calibration has unexpected offense/support: "
            f"{data.get('offense')}/{data.get('support')}"
        )
    if data.get("selection_metric") != "rolling_origin_incident_weighted_tract_tvd":
        issues.append(
            "murder tract-posterior calibration has unexpected selection metric: "
            f"{data.get('selection_metric')}"
        )
    if data.get("history_rule") != "strictly_earlier_city_years_only":
        issues.append(
            "murder tract-posterior calibration does not use strictly earlier years"
        )
    if data.get("candidate_prior_refit_per_fold") is not False:
        issues.append("murder tract-posterior calibration does not disclose its frozen-prior limitation")

    selected = data.get("selected") if isinstance(data.get("selected"), dict) else {}
    try:
        half_life = float(selected.get("half_life_years"))
        prior_incidents = float(selected.get("prior_incidents"))
        selected_tvd = float(selected.get("incident_weighted_tvd"))
    except (TypeError, ValueError):
        half_life = prior_incidents = selected_tvd = float("nan")
    if not np.isfinite(half_life) or half_life <= 0.0:
        issues.append(f"murder tract-posterior calibration has invalid half-life: {half_life}")
    if not np.isfinite(prior_incidents) or prior_incidents <= 0.0:
        issues.append(
            f"murder tract-posterior calibration has invalid prior incidents: {prior_incidents}"
        )
    if not np.isfinite(selected_tvd) or selected_tvd < 0.0:
        issues.append(f"murder tract-posterior calibration has invalid selected TVD: {selected_tvd}")
    if not holdout_run and (
        int(data.get("fold_count") or 0) < 20 or int(data.get("city_count") or 0) < 5
    ):
        issues.append(
            "murder tract-posterior calibration lacks the required rolling support: "
            f"folds={data.get('fold_count')}, cities={data.get('city_count')}"
        )

    candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
    candidate_rows = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        try:
            candidate_rows.append(
                (
                    float(row["incident_weighted_tvd"]),
                    float(row["incident_weighted_cross_entropy"]),
                    -float(row["prior_incidents"]),
                    float(row["half_life_years"]),
                    float(row["half_life_years"]),
                    float(row["prior_incidents"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not candidate_rows:
        issues.append("murder tract-posterior calibration has no readable candidate table")
    else:
        best = min(candidate_rows)
        if abs(float(best[4]) - half_life) > 1e-12 or abs(float(best[5]) - prior_incidents) > 1e-12:
            issues.append(
                "murder tract-posterior calibration selected parameters do not match the declared rule: "
                f"selected={half_life}/{prior_incidents}, recomputed={best[4]}/{best[5]}"
            )
    return {
        "present": True,
        "path": str(path),
        "half_life_years": half_life,
        "prior_incidents": prior_incidents,
        "incident_weighted_tvd": selected_tvd,
        "fold_count": int(data.get("fold_count") or 0),
        "city_count": int(data.get("city_count") or 0),
        "holdout_floor_acknowledged": bool(holdout_run),
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
    murder_tract_posterior_calibration: dict[str, Any],
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

    expected_murder_half_life = murder_tract_posterior_calibration.get("half_life_years")
    expected_murder_prior = murder_tract_posterior_calibration.get("prior_incidents")
    for source_name, source in (("resolved_config", resolved), ("summary", summary)):
        policy = source.get("unified_murder_tract_posterior")
        if not isinstance(policy, dict):
            issues.append(f"build manifest {source_name} missing unified_murder_tract_posterior")
            continue
        if policy.get("support") != "census_tract":
            issues.append(
                f"build manifest {source_name} murder posterior support is {policy.get('support')!r}"
            )
        if policy.get("within_tract_distribution") != "rare_offense_prior_only":
            issues.append(
                f"build manifest {source_name} murder posterior has unexpected within-tract rule: "
                f"{policy.get('within_tract_distribution')}"
            )
        calibration_path = policy.get("calibration_path")
        if not calibration_path or not _manifest_path_exists(calibration_path):
            issues.append(
                f"build manifest {source_name} murder posterior calibration path is missing: "
                f"{calibration_path}"
            )
        try:
            observed_half_life = float(policy.get("incident_half_life_years"))
            observed_prior = float(policy.get("prior_incidents"))
        except (TypeError, ValueError):
            observed_half_life = observed_prior = float("nan")
        if expected_murder_half_life is None or not np.isfinite(float(expected_murder_half_life)):
            issues.append("murder tract-posterior calibration did not supply a finite half-life")
        elif not np.isfinite(observed_half_life) or abs(observed_half_life - float(expected_murder_half_life)) > 1e-12:
            issues.append(
                f"build manifest {source_name} murder half-life {observed_half_life} "
                f"!= calibrated {expected_murder_half_life}"
            )
        if expected_murder_prior is None or not np.isfinite(float(expected_murder_prior)):
            issues.append("murder tract-posterior calibration did not supply finite prior incidents")
        elif not np.isfinite(observed_prior) or abs(observed_prior - float(expected_murder_prior)) > 1e-12:
            issues.append(
                f"build manifest {source_name} murder prior incidents {observed_prior} "
                f"!= calibrated {expected_murder_prior}"
            )

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
    if summary.get("cde_exact_sensitivity_written") is not True:
        issues.append("build manifest says the CDE-exact diagnostic sensitivity outputs were not written")
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
    composite_lane = resolved.get("count_first_composites")
    count_first_manifest = bool(
        isinstance(composite_lane, dict) and composite_lane.get("enabled") is True
    )
    for surface_key in ("block_group_ags_core", "tract_ags_core", "block_group_cde_exact_sensitivity", "tract_cde_exact_sensitivity"):
        surface_normalizers = aggregate_normalizers.get(surface_key)
        if not isinstance(surface_normalizers, dict):
            issues.append(f"build manifest missing aggregate normalizers for {surface_key}")
            continue
        if count_first_manifest:
            # Count-first lane: the reproducible record is the count total, the common-denominator
            # total and the reference rate per field, plus the severity vector for the harm field.
            burden = surface_normalizers.get("count_first_burden")
            if not isinstance(burden, dict) or not isinstance(burden.get("fields"), dict):
                issues.append(f"build manifest missing count_first_burden normalizers for {surface_key}")
            else:
                for field in (
                    EVENT_BURDEN_COLUMN,
                    PERSONAL_BURDEN_COLUMN,
                    PROPERTY_BURDEN_COLUMN,
                    HARM_BURDEN_COLUMN,
                ):
                    row = burden["fields"].get(field)
                    if not isinstance(row, dict) or "reference_rate_per_100k" not in row:
                        issues.append(
                            f"build manifest missing reference_rate_per_100k for {surface_key}.{field}"
                        )
                harm_row = burden["fields"].get(HARM_BURDEN_COLUMN)
                if isinstance(harm_row, dict) and not isinstance(harm_row.get("severity_weights"), dict):
                    issues.append(f"build manifest missing severity weights for {surface_key}.{HARM_BURDEN_COLUMN}")
            for key in (
                "multi_offense_relative_score_event_weighted",
                "multi_offense_relative_score_equal_offense",
            ):
                row = surface_normalizers.get(key)
                if not isinstance(row, dict) or row.get("field") != key or not isinstance(row.get("offense_weights"), dict):
                    issues.append(f"build manifest missing reproducible weights for {surface_key}.{key}")
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

    for key in ("block_group_ags_core", "tract_ags_core", "block_group_cde_exact_sensitivity", "tract_cde_exact_sensitivity"):
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
    murder_calibration_stat = input_stats.get("murder_tract_posterior_calibration")
    if not isinstance(murder_calibration_stat, dict) or murder_calibration_stat.get("exists") is not True:
        issues.append("build manifest input stat missing murder tract-posterior calibration artifact")

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
        "murder_tract_posterior_calibration": murder_tract_posterior_calibration,
        "aggregate_index_normalizer_surfaces": sorted(aggregate_normalizers.keys()),
        "burglary_commercial_calibration": burglary_calibration,
        "burglary_commercial_gradient_block_group_ags_core": bg_gradient,
        "suppression_mode_count_surfaces": sorted(suppression_counts.keys()),
        "output_file_stats_present": sorted(output_stats.keys()),
    }


def _load_validation_or_repo_json(*, validation_name: str, repo_path: Path) -> dict[str, Any] | None:
    return _load_json(PACKAGE_VALIDATION_DIR / validation_name) or _load_json(repo_path)


def _next_phase_truth_inventory() -> dict[str, Any]:
    """Recount the target-year truth rows independently of the measurement summary.

    The expanded validation surface is the preferred input when present.  A calendar-specific
    absolute row floor is not stable: cities and offenses enter only when target-year incident
    truth is available.  The release gate instead requires the measurement artifact to exhaust
    the target year's actual source inventory, while retaining the twelve promoted controls as a
    fixed minimum backbone.
    """
    candidates = [
        REPO_ROOT / "state" / "modeling" / f"next_phase_validation_city_incident_share_surface_{YEAR}.parquet",
        REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return {"present": False, "candidate_paths": [str(candidate) for candidate in candidates]}

    frame = pd.read_parquet(path)
    required = {"year", "jurisdiction_id", "offense"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"present": True, "path": str(path), "missing_columns": missing}

    frame = frame[pd.to_numeric(frame["year"], errors="coerce").eq(int(YEAR))].copy()
    frame = frame.dropna(subset=["jurisdiction_id", "offense"])
    cases = frame.drop_duplicates("jurisdiction_id")
    case_type = (
        cases["validation_case_type"].astype("string").fillna("")
        if "validation_case_type" in cases.columns
        else pd.Series("", index=cases.index, dtype="string")
    )
    cv_cases = cases[~case_type.eq("suburban_county_validation_case")]

    error_source = frame
    if "validation_case_type" in error_source.columns:
        error_source = error_source[
            ~error_source["validation_case_type"]
            .astype("string")
            .fillna("")
            .eq("suburban_county_validation_case")
        ]
    pair_columns = [
        column
        for column in ["city_name", "jurisdiction_id", "state_fips", "offense"]
        if column in error_source.columns
    ]
    error_budget_rows = int(error_source[pair_columns].drop_duplicates().shape[0])
    return {
        "present": True,
        "path": str(path),
        "missing_columns": [],
        "truth_case_count": int(cases["jurisdiction_id"].nunique()),
        "truth_city_count": int(cv_cases["jurisdiction_id"].nunique()),
        "error_budget_rows": error_budget_rows,
        "promoted_city_control_count": int(case_type.eq("promoted_city_control").sum()),
    }


def _check_next_phase_measurement(
    *, issues: list[str], holdout_run: bool = False
) -> dict[str, Any]:
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
    inventory = _next_phase_truth_inventory()

    if not inventory.get("present"):
        issues.append("missing next-phase target-year truth inventory")
    elif inventory.get("missing_columns"):
        issues.append(
            "next-phase target-year truth inventory missing columns "
            f"{inventory['missing_columns']}"
        )
    else:
        expected = {
            "truth_case_count": int(inventory["truth_case_count"]),
            "truth_city_count": int(inventory["truth_city_count"]),
            "error_budget_rows": int(inventory["error_budget_rows"]),
        }
        observed = {
            "truth_case_count": truth_case_count,
            "truth_city_count": truth_city_count,
            "error_budget_rows": error_budget_rows,
        }
        for field, expected_value in expected.items():
            if observed[field] != expected_value:
                issues.append(
                    f"next-phase measurement {field} {observed[field]} != "
                    f"target-year truth inventory {expected_value}"
                )
        if (
            not holdout_run
            and int(inventory.get("promoted_city_control_count") or 0) < 12
        ):
            issues.append(
                "next-phase target-year truth inventory has fewer than 12 promoted city controls"
            )
    if cv_prediction_rows <= 0:
        issues.append("next-phase measurement has no held-out CV prediction rows")
    # The recommended next workstream is reported, never gated: see
    # `_check_release_evaluation` for the requirement that replaced it.
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
        "target_year_truth_inventory": inventory,
        "holdout_floor_acknowledged": bool(holdout_run),
    }


def _release_candidate_from_snapshot_config() -> str | None:
    """The candidate directory the published snapshot is built from, or None."""
    if not RELEASE_SNAPSHOT_CONFIG.exists():
        return None
    for line in RELEASE_SNAPSHOT_CONFIG.read_text().splitlines():
        line = line.strip()
        if not line.startswith("CRIMERISK_SNAPSHOT_SRC="):
            continue
        value = line.partition("=")[2].strip().strip('"').strip("'")
        parts = Path(value).parts
        if "candidates" in parts:
            return parts[parts.index("candidates") + 1]
    return None


def _check_release_evaluation(*, issues: list[str]) -> dict[str, Any]:
    """Bind promotion to the release's own held-out evaluation.

    Three requirements, all of them about this release rather than about a research
    preference: the gold results table for the release run is present; it is the
    evaluation of the candidate the public surface is actually built from; and on the
    spatial folds the `ours` arm beats the population null on TVD for every gated
    offense. The first two always block. The third blocks only when
    `RELEASE_GOLD_TVD_BLOCKING` is set; otherwise every loss is listed under
    `failures` and `status` says "reported", so the summary states the result without
    promoting it to an issue.
    """
    results_path = RELEASE_GOLD_DIR / "results.csv"
    manifest_path = RELEASE_GOLD_DIR / "run_manifest.json"
    summary: dict[str, Any] = {
        "run_id": RELEASE_GOLD_RUN_ID,
        "results_path": str(results_path),
        "present": results_path.exists(),
        "candidate_expected": RELEASE_CANDIDATE,
        "status": "blocking" if RELEASE_GOLD_TVD_BLOCKING else "reported",
    }
    if not results_path.exists():
        issues.append(
            f"missing release evaluation results for {RELEASE_GOLD_RUN_ID} at {results_path}"
        )
        return summary

    published_candidate = _release_candidate_from_snapshot_config()
    summary["candidate_published"] = published_candidate
    if published_candidate is None:
        issues.append(
            "release evaluation cannot be bound to a candidate: "
            f"{RELEASE_SNAPSHOT_CONFIG} names no candidate snapshot source"
        )
    elif published_candidate != RELEASE_CANDIDATE:
        issues.append(
            f"release evaluation is bound to candidate {RELEASE_CANDIDATE!r} but the "
            f"published snapshot is built from {published_candidate!r}"
        )
    candidate_dir = REPO_ROOT / "state" / "candidates" / RELEASE_CANDIDATE
    summary["candidate_dir_present"] = candidate_dir.is_dir()
    if not candidate_dir.is_dir():
        issues.append(f"release evaluation candidate directory is missing: {candidate_dir}")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        summary["manifest_run_id"] = manifest.get("run_id")
        summary["manifest_year"] = (manifest.get("config") or {}).get("year")
        if manifest.get("run_id") != RELEASE_GOLD_RUN_ID:
            issues.append(
                f"release evaluation manifest names run {manifest.get('run_id')!r}, "
                f"expected {RELEASE_GOLD_RUN_ID!r}"
            )
        if int((manifest.get("config") or {}).get("year") or 0) != RELEASE_GOLD_BUILD_YEAR:
            issues.append(
                "release evaluation manifest build year "
                f"{(manifest.get('config') or {}).get('year')!r}, expected {RELEASE_GOLD_BUILD_YEAR}"
            )
    else:
        issues.append(f"missing release evaluation run manifest at {manifest_path}")

    results = pd.read_csv(results_path)
    run_ids = sorted(set(results["run_id"].astype(str))) if "run_id" in results.columns else []
    summary["results_run_ids"] = run_ids
    if run_ids != [RELEASE_GOLD_RUN_ID]:
        issues.append(
            f"release evaluation results name run ids {run_ids}, expected [{RELEASE_GOLD_RUN_ID!r}]"
        )

    spatial = results[
        results["fold_type"].astype(str).eq("spatial")
        & results["metric"].astype(str).eq("tvd")
    ]
    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for offense in OFFENSES_7:
        rows = spatial[spatial["offense"].astype(str).eq(offense)]
        ours = rows[rows["arm"].astype(str).eq("ours")]["estimate"]
        null = rows[rows["arm"].astype(str).eq(RELEASE_GOLD_BASELINE_ARM)]["estimate"]
        gated = offense not in RELEASE_GOLD_UNGATED_OFFENSES
        if ours.empty or null.empty:
            comparisons.append({"offense": offense, "gated": gated, "present": False})
            if gated:
                issues.append(
                    f"release evaluation has no spatial TVD comparison for {offense}"
                )
            continue
        ours_tvd = float(ours.iloc[0])
        null_tvd = float(null.iloc[0])
        comparisons.append(
            {
                "offense": offense,
                "gated": gated,
                "present": True,
                "ours_tvd": ours_tvd,
                f"{RELEASE_GOLD_BASELINE_ARM}_tvd": null_tvd,
                "beats_baseline": bool(ours_tvd < null_tvd),
            }
        )
        if gated and not ours_tvd < null_tvd:
            failures.append(
                f"release evaluation: spatial {offense} TVD {ours_tvd:.4f} does not beat "
                f"the {RELEASE_GOLD_BASELINE_ARM} arm's {null_tvd:.4f}"
            )
    summary["spatial_tvd_vs_baseline"] = comparisons
    summary["failures"] = failures
    summary["passed"] = not failures
    if RELEASE_GOLD_TVD_BLOCKING:
        issues.extend(failures)
    return summary


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
        "rare_offense_allocation_policy",
        "rare_offense_information_constant",
        "rare_offense_effective_model_information",
        "rare_offense_model_information_weight",
        "rare_offense_tract_model_share",
        "rare_offense_tract_exposure_share",
        "rare_offense_within_tract_exposure_share",
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
    active_murder = active & policy_rows["offense"].eq("murder")
    rare_prior_rows = sparse_uncovered | active_murder
    sparse_uncovered_comparable = rare_prior_rows & policy_rows["_baseline_total"].gt(0.0)
    dense_uncovered = uncovered & policy_rows["offense"].isin(DENSE_FULL_RESIDUAL_TRANSFER_OFFENSES)
    burglary_uncovered = uncovered & policy_rows["offense"].eq("burglary")

    if not bool(sparse_uncovered.any()):
        issues.append("allocation component audit has no uncovered murder/rape rows for sparse transfer validation")
    sparse_rows = policy_rows.loc[rare_prior_rows]
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

    sparse_policy_name = "tract_information_shrinkage_residential_exposure"
    expected_sparse_policy = pd.Series(sparse_policy_name, index=sparse_rows.index, dtype="string")
    expected_sparse_policy.loc[active_murder.loc[sparse_rows.index]] = (
        "unified_murder_tract_incident_posterior"
    )
    sparse_policy_bad = (
        sparse_rows["city_residual_transfer_policy"].astype("string").ne(expected_sparse_policy)
        | sparse_rows["rare_offense_allocation_policy"].astype("string").ne(expected_sparse_policy)
    )
    if bool(sparse_policy_bad.any()):
        examples = (
            sparse_rows.loc[sparse_policy_bad, group_cols + ["city_residual_transfer_policy"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        issues.append(
            "rare-offense prior rows do not use the declared tract-support policy: "
            f"{examples}"
        )

    sparse_information = pd.to_numeric(
        sparse_compare_rows["rare_offense_effective_model_information"], errors="coerce"
    )
    sparse_constant = pd.to_numeric(
        sparse_compare_rows["rare_offense_information_constant"], errors="coerce"
    )
    sparse_weight = pd.to_numeric(
        sparse_compare_rows["rare_offense_model_information_weight"], errors="coerce"
    )
    expected_sparse_weight = sparse_information / (sparse_information + sparse_constant)
    sparse_weight_delta = (sparse_weight - expected_sparse_weight).abs()
    sparse_expected_raw = (
        sparse_weight
        * pd.to_numeric(sparse_compare_rows["rare_offense_tract_model_share"], errors="coerce")
        + (1.0 - sparse_weight)
        * pd.to_numeric(sparse_compare_rows["rare_offense_tract_exposure_share"], errors="coerce")
    ) * pd.to_numeric(
        sparse_compare_rows["rare_offense_within_tract_exposure_share"], errors="coerce"
    )
    sparse_expected_total = sparse_expected_raw.groupby(
        [sparse_compare_rows[col] for col in group_cols], dropna=False
    ).transform("sum")
    sparse_expected_share = np.where(
        sparse_expected_total.gt(0.0),
        sparse_expected_raw / sparse_expected_total.replace(0.0, np.nan),
        0.0,
    )
    sparse_expected_normalized = pd.Series(
        sparse_expected_share, index=sparse_compare_rows.index, dtype=float
    )
    sparse_raw_delta = (
        sparse_compare_rows["city_posterior_model_prior_raw"] - sparse_expected_normalized
    ).abs()
    sparse_share_delta = (
        sparse_compare_rows["city_posterior_model_prior_share"]
        - sparse_expected_normalized
    ).abs()
    max_sparse_raw_delta = float(sparse_raw_delta.max()) if len(sparse_raw_delta) else 0.0
    max_sparse_share_delta = float(sparse_share_delta.max()) if len(sparse_share_delta) else 0.0
    max_sparse_weight_delta = float(sparse_weight_delta.max()) if len(sparse_weight_delta) else 0.0
    if (
        max_sparse_raw_delta > TRANSFER_POLICY_TOLERANCE
        or max_sparse_share_delta > TRANSFER_POLICY_TOLERANCE
        or max_sparse_weight_delta > TRANSFER_POLICY_TOLERANCE
    ):
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
                    "_raw_delta",
                    "_share_delta",
                ]
            ]
            .to_dict(orient="records")
        )
        issues.append(
            "uncovered murder/rape tract-information shrinkage does not reproduce its audit fields "
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

    # The generic tau formula applies to dense offenses and covered-city priors.
    # Fork 6 replaces uncovered murder/rape tau=0 rows with the audited tract-level
    # information-shrinkage formula checked above.
    comparable = policy_rows[
        policy_rows["_baseline_total"].gt(0.0) & ~rare_prior_rows
    ].copy()
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
        "max_sparse_information_weight_delta": max_sparse_weight_delta,
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
        # Null preferred_source is the honest value for a skeleton row with no
        # target-year agency contribution. This column is load-bearing for that
        # distinction: omitting it made _check_source_priority_honored fall back to
        # "every row has an agency" and falsely rejected all 28 no-evidence rows.
        "contributing_agency_count",
        "dominant_reporting_regime",
        # `dominant_preferred_source_by_regime`, `published_nibrs_corroborated_count` and
        # `cius_municipal_official_count` used to be required here. The Stage 3
        # consumption-plus-skeleton restructure dropped all three from the control schema and
        # this list was not updated, so the whole total lane failed to load with
        # "controls missing required columns" -- i.e. every total-lane check silently stopped
        # running. None of the three is read anywhere in this file. Removed rather than
        # re-added: a required column that nothing consumes is a tripwire without a purpose.
        # (Found 2026-07-30 while landing the Stage 2 fix batch; it belongs to the Stage 3/4
        # batch's ledger, not to Stage 2.)
        # Class A benchmark-constrained imputation: the control target of a silent
        # jurisdiction is the panel estimate plus this, because the silent agency has no
        # panel row at all by design (v19 drops silent agencies rather than fabricating
        # an anchor). Carried here so the panel-alignment check can reconcile it
        # explicitly rather than exempting the rows.
        BENCHMARK_IMPUTED_COUNT_COLUMN,
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
        # Zero-weight covered-ORI rows preserve source identity but do not assign
        # numerator mass.  Covering agencies are intentionally multi-footprint:
        # a sheriff/state-remainder service area may include covered municipal
        # footprints.  Neither case is the duplicate-total failure this gate was
        # created to detect.
        positive_crosswalk = crosswalk[crosswalk["weight"].gt(0.0)].copy()
        agency_type_sets = (
            positive_crosswalk.groupby(["state_fips", "ori"], dropna=False)
            .agg(
                jurisdiction_type=(
                    "jurisdiction_type", lambda s: sorted(set(s.dropna().astype(str)))
                ),
                relationship_types=(
                    "relationship_type", lambda s: sorted(set(s.dropna().astype(str)))
                ),
            )
            .reset_index()
        )
        agency_type_sets["jurisdiction_type_count"] = agency_type_sets["jurisdiction_type"].map(len)
        multi_assignment_rows = agency_type_sets[agency_type_sets["jurisdiction_type_count"].gt(1)].copy()
        municipal_layer_overlap_rows = agency_type_sets[
            agency_type_sets.apply(
                lambda row: (
                    "municipal" in row["jurisdiction_type"]
                    and any(
                        value in row["jurisdiction_type"]
                        for value in ("state_nonmunicipal_remainder", "statewide_overlap_layer")
                    )
                    and not all(
                        value.startswith("contract_") for value in row["relationship_types"]
                    )
                ),
                axis=1,
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

    # Lane choice is a property of the agency-year (see source_selection's docstring),
    # so the expectation is rebuilt the same way: rank the present lanes once per
    # agency and read every offense from the winner, falling to the next ranked lane
    # only where the winner does not publish that offense at all.
    nibrs_months = pd.to_numeric(panel["nibrs_months_reported"], errors="coerce").fillna(
        pd.to_numeric(panel["mean_months_reported_nibrs"], errors="coerce").fillna(0.0)
    )
    published_nibrs_supports_nibrs = (
        panel["reported_count_nibrs"].notna()
        & build_published_nibrs_corroboration_mask(
            nibrs_count=panel["reported_count_nibrs"],
            published_nibrs_count=panel.get("published_nibrs_official_count"),
            nibrs_months=nibrs_months,
            srs_count=panel["reported_count_srs"],
        )
    )
    manual_source_override = (
        panel["source_override_applied"].astype("boolean").fillna(False).astype(bool)
    )
    override_lane = (
        panel.loc[manual_source_override, ["ori9", "preferred_source_by_regime"]]
        .dropna()
        .drop_duplicates(subset=["ori9"], keep="first")
        .set_index("ori9")["preferred_source_by_regime"]
        .astype("string")
    )
    ranked = _rank_agency_year_lanes(
        _agency_year_lane_signals(panel, corroborated=published_nibrs_supports_nibrs),
        override_lane=override_lane,
    )
    expected_source = pd.Series(pd.NA, index=panel.index, dtype="string")
    best_rank = pd.Series(np.inf, index=panel.index, dtype=float)
    for source in LANE_STANDING_ORDER:
        count_col, _ = PREFERRED_LANE_COLUMNS[source]
        rank = pd.to_numeric(
            panel["ori9"].map(
                ranked[ranked["source"].eq(source)].set_index("ori9")["lane_rank"]
            ),
            errors="coerce",
        )
        candidate = panel[count_col].notna() & rank.notna() & rank.lt(best_rank)
        expected_source = expected_source.where(~candidate, source)
        best_rank = best_rank.where(~candidate, rank)
    panel["expected_preferred_source"] = expected_source
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


def _check_source_priority_honored(
    *,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    preferred_source = controls["preferred_source"].astype("string")
    # A control row whose jurisdiction had no contributing agency in the target year has
    # no reported lane to name, and says so with a null rather than a fabricated label.
    # Its whole target is benchmark-imputed or zero, and `estimate_source` carries that.
    has_contributing_agency = (
        pd.to_numeric(controls.get("contributing_agency_count"), errors="coerce").fillna(1).gt(0)
        if "contributing_agency_count" in controls.columns
        else pd.Series(True, index=controls.index)
    )
    unknown_source_rows = controls[
        ~preferred_source.isin(SOURCE_PRIORITY) & has_contributing_agency
    ].copy()
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

    # Level-lane v2 deliberately changes the relationship between raw agency-lane
    # rollups and published controls: rejected fragments are audit-only,
    # review-held rows remain unchanged, and reason-coded repairs own separate mass.
    # Once the canonical ledger exists, it supersedes the legacy raw-rollup equality
    # check and is the independent composition check for this boundary.
    ledger_path = REPO_ROOT / "state" / "controls" / f"level_lane_mass_ledger_{YEAR}.parquet"
    if ledger_path.exists():
        ledger = pd.read_parquet(ledger_path)
        assert_level_lane_mass_ledger(ledger=ledger, controls=controls)
        final_mass = pd.to_numeric(ledger["final_control_mass"], errors="coerce").fillna(0.0)
        unresolved_mass = pd.to_numeric(ledger["unresolved_mass"], errors="coerce").fillna(0.0)
        published_unresolved = final_mass.gt(1e-9) & (
            ledger["level1_admission_status"].astype("string").eq("unresolved_review")
            | ledger["level_repair_mode"].astype("string").eq("unchanged_review_hold")
            | unresolved_mass.gt(1e-9)
        )
        if published_unresolved.any():
            issues.append(
                "level_lane.zero_published_unresolved_mass: "
                f"{int(published_unresolved.sum())} positive-mass ledger rows retain "
                f"{float(final_mass.loc[published_unresolved].sum()):.6f} unresolved mass"
            )
        agency_panel = None
        agency_summary = {
            "present": True,
            "validation_basis": "level_lane_mass_ledger",
            "path": str(ledger_path),
            "rows": int(len(ledger)),
            "legacy_raw_rollup_equality_superseded": True,
            "published_unresolved_rows": int(published_unresolved.sum()),
            "published_unresolved_mass": float(final_mass.loc[published_unresolved].sum()),
        }
    else:
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
            # Every lane, not only the two synthetic ones. Since Stage 3 consumes the
            # agency estimates instead of re-deriving a jurisdiction-level preference,
            # `reported_count_preferred` IS the crosswalk-weighted agency rollup on the
            # municipal lane too, and this recompute is the independent check on it.
            merged = controls.merge(aggregate, on=["jurisdiction_id", "offense"], how="left")
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
                    "total_lane.source_priority_honored: control reported_count_preferred does not equal the agency-level preferred-source rollup",
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
                    "total_lane.source_priority_honored: control preferred_source does not match the agency-level dominant preferred source",
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
        # A benchmark-imputed jurisdiction is silent, so v19's zero-fabrication rule
        # leaves it with no estimate row (or a zero one) and its whole control target is
        # the imputed sub-target. Reconcile that explicitly -- panel estimate plus
        # benchmark_imputed_count -- rather than exempting the rows, so a drift in the
        # non-imputed part of an imputed jurisdiction's target still fails.
        benchmark_imputed = pd.to_numeric(
            aligned.get(
                BENCHMARK_IMPUTED_COUNT_COLUMN,
                pd.Series(0.0, index=aligned.index, dtype=float),
            ),
            errors="coerce",
        ).fillna(0.0)
        target_delta = (
            pd.to_numeric(aligned[TOTAL_LANE_TARGET_COLUMN], errors="coerce").fillna(0.0)
            - pd.to_numeric(aligned["estimate_target_count"], errors="coerce").fillna(0.0)
            - benchmark_imputed
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
        "control_rows_with_agency_recomputed_preference": synthetic_recomputed_rows,
        "control_rows_missing_agency_recomputed_preference": synthetic_missing_recomputed_rows,
        "agency_rollup_reported_count_bad_rows": int(len(synthetic_count_bad_rows)),
        "agency_rollup_preferred_source_bad_rows": int(len(synthetic_source_bad_rows)),
        "max_agency_rollup_reported_count_delta": max_synthetic_count_delta,
        "agency_preferred_panel": agency_summary,
        "agency_to_jurisdiction_preferred_rollup": aggregate_summary,
        "jurisdiction_year_estimate_alignment": estimate_alignment_summary,
        "offending_rows_sample": {
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
    and docs/archive/2026-09/STATE.md). External-referenced (peer counties), not self-referential: a
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
    # A published zero is a valid point estimate, including for murder in a large
    # city.  This gate is about missing or invalid controls, not forcing every
    # city/offense cell to be positive.
    bad_rows = expected[expected["control_target"].isna() | expected["control_target"].lt(0.0)].copy()
    if not bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.high_population_spot_checks: top municipal jurisdiction missing or negative modeled control",
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
        "missing_or_negative_control_rows": int(len(bad_rows)),
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


def _stage2_reviewed_footprint_oris(*, issues: list[str]) -> tuple[set[str], dict[str, Any]]:
    """ORIs whose footprint carries a reviewed registry decision.

    `local_resolution_overrides.csv` is the ORI-keyed local/nonlocal resolution registry (a row
    there means a reviewer decided where this agency's ground is), and
    `consolidated_agency_footprints.csv` is the city-county consolidation registry the plain
    screen cannot know about -- it will always flag LVMPD and LMPD as concentrated because
    their FBI service population is county-wide while their principal jurisdiction is the city.
    `overlap_footprint_overrides.csv` is included for completeness: an agency with a reviewed
    overlap footprint is not being placed by an automatic municipal match at all.
    """
    summary: dict[str, Any] = {}
    oris: set[str] = set()
    for name, column in (
        ("local_resolution_overrides.csv", "ori"),
        ("consolidated_agency_footprints.csv", "ori"),
        ("overlap_footprint_overrides.csv", "ori"),
    ):
        path = REPO_ROOT / "configs" / name
        if not path.exists():
            summary[name] = {"present": False, "rows": 0}
            continue
        try:
            frame = pd.read_csv(path, dtype=str)
        except Exception as exc:  # pragma: no cover - config read failure is an issue, not a crash
            issues.append(f"total_lane.stage2_footprint_plausibility: cannot read configs/{name}: {exc}")
            summary[name] = {"present": True, "rows": 0, "error": str(exc)}
            continue
        values = {
            str(value).strip().upper()
            for value in frame.get(column, pd.Series(dtype=str)).dropna()
        }
        values.discard("")
        oris |= values
        summary[name] = {"present": True, "rows": int(len(frame)), "oris": int(len(values))}
    return oris, summary


def _check_stage2_footprint_plausibility(
    *,
    controls: pd.DataFrame,
    issues: list[str],
) -> dict[str, Any]:
    """Fail-closed Stage 2 invariant: agency mass must land on a plausible footprint.

    See the STAGE2_* constants for the design, the measured hit sets and the thresholds.
    """
    reviewed_oris, registry_summary = _stage2_reviewed_footprint_oris(issues=issues)
    crosswalk_path = REPO_ROOT / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    agency_path = REPO_ROOT / "state" / "reference" / "agency_master.parquet"
    observations_path = REPO_ROOT / "state" / "observations" / "agency_year_observations.parquet"
    bg_crosswalk_path = (
        REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    )
    missing = [
        str(path)
        for path in (crosswalk_path, agency_path, observations_path, bg_crosswalk_path)
        if not path.exists()
    ]
    if missing:
        issues.append(
            f"total_lane.stage2_footprint_plausibility: missing required input paths {missing}"
        )
        return {"ok": False, "missing_input_paths": missing, "registries": registry_summary}

    crosswalk = pd.read_parquet(
        crosswalk_path,
        columns=["ori", "state_fips", "jurisdiction_id", "relationship_type", "resolution_source"],
    ).rename(columns={"ori": "ori9"})
    crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
    links = crosswalk[
        crosswalk["relationship_type"].astype("string").eq("exclusive")
        & crosswalk["jurisdiction_id"].astype("string").str.contains(":municipal:", na=False)
        & ~crosswalk["state_fips"].isin(RELEASE_EXCLUDED_STATE_FIPS)
    ].copy()

    agency = pd.read_parquet(
        agency_path, columns=["ori9", "state_abbr", "agency_name_std", "agency_type_norm"]
    )
    links = links.merge(agency, on="ori9", how="left")

    observations_all = pd.read_parquet(
        observations_path, columns=["ori9", "year", "offense", "count", "population"]
    )
    observations_all["population"] = pd.to_numeric(
        observations_all["population"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    # Filled current-year controls often have no target-year agency population.
    # Compare the Census footprint to the latest positive FBI service population
    # available on or before the edition year, rather than treating the missing
    # current-year field as evidence of a bad crosswalk.
    service_pop = (
        observations_all[
            observations_all["year"].astype("Int64").le(YEAR)
            & observations_all["population"].gt(0.0)
        ]
        .groupby(["ori9", "year"], dropna=False)["population"]
        .max()
        .reset_index()
        .sort_values(["ori9", "year"], kind="mergesort")
        .groupby("ori9", dropna=False)
        .tail(1)
        .rename(columns={"population": "service_pop", "year": "service_pop_year"})
    )
    observations = observations_all[
        observations_all["year"].astype("Int64").le(YEAR)
        & observations_all["offense"].astype("string").isin(OFFENSES_7)
    ].copy()
    observations["count"] = pd.to_numeric(observations["count"], errors="coerce").fillna(0.0)
    # Attribute a shared jurisdiction's modeled mass only to links with actual
    # reporting support.  Current-roster aliases and predecessor ORIs often have
    # no target-year observation while the canonical link supplies the historical
    # series used by the control.  A target-year-only split assigned half the mass
    # to those dormant aliases and manufactured footprint failures.  Collapse
    # duplicate source rows within offense/year, then use all observations through
    # the edition year solely as a support weight for this diagnostic.
    reported = (
        observations.groupby(["ori9", "year", "offense"], dropna=False)["count"]
        .max()
        .groupby("ori9")
        .sum()
        .rename("agency_reported_count")
        .reset_index()
    )
    links = links.merge(service_pop, on="ori9", how="left").merge(reported, on="ori9", how="left")

    # Footprint population and the mass actually targeted at that footprint, per jurisdiction.
    municipal_controls = controls[controls["jurisdiction_type"].eq("municipal")].copy()
    municipal_controls["adjusted_count_ags_core"] = pd.to_numeric(
        municipal_controls["adjusted_count_ags_core"], errors="coerce"
    ).fillna(0.0)
    jurisdiction = (
        municipal_controls.groupby("jurisdiction_id", dropna=False)
        .agg(
            jurisdiction_name=("jurisdiction_name", "first"),
            footprint_pop=("bucket_population", "max"),
            jurisdiction_count=("adjusted_count_ags_core", "sum"),
        )
        .reset_index()
    )
    links = links.merge(jurisdiction, on="jurisdiction_id", how="left")

    bg_crosswalk = pd.read_parquet(
        bg_crosswalk_path, columns=["state_fips", "jurisdiction_type", "pop20"]
    )
    in_scope_pop = float(
        pd.to_numeric(
            bg_crosswalk.loc[
                ~bg_crosswalk["state_fips"].astype("string").str.zfill(2).isin(RELEASE_EXCLUDED_STATE_FIPS),
                "pop20",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )
    in_scope_counts = float(
        pd.to_numeric(
            controls.loc[
                ~controls["state_fips"].astype("string").str.zfill(2).isin(RELEASE_EXCLUDED_STATE_FIPS),
                "adjusted_count_ags_core",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )
    national_rate = (in_scope_counts / in_scope_pop * 100_000.0) if in_scope_pop > 0 else None
    if not national_rate or national_rate <= 0:
        issues.append(
            "total_lane.stage2_footprint_plausibility: cannot compute a national reference rate"
        )
        return {"ok": False, "registries": registry_summary}

    # Attribute the jurisdiction's control mass across its links so a zero-mass link (e.g. the
    # dormant half of a duplicate-ORI twin pair) is not screened as if it carried the whole
    # jurisdiction. Where every link on a jurisdiction reports zero -- a fill-only or
    # imputation-only jurisdiction, which is exactly the tribal fill class the screen must see
    # -- the mass is split evenly rather than dropped.
    links["agency_reported_count"] = pd.to_numeric(
        links["agency_reported_count"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    reported_total = links.groupby("jurisdiction_id", dropna=False)["agency_reported_count"].transform("sum")
    link_count = links.groupby("jurisdiction_id", dropna=False)["ori9"].transform("size")
    links["link_mass_share"] = np.where(
        reported_total > 0,
        links["agency_reported_count"] / reported_total.where(reported_total > 0, 1.0),
        1.0 / link_count.where(link_count > 0, 1.0),
    )
    links["footprint_pop"] = pd.to_numeric(links["footprint_pop"], errors="coerce")
    links["service_pop"] = pd.to_numeric(links["service_pop"], errors="coerce")
    links["jurisdiction_count"] = pd.to_numeric(links["jurisdiction_count"], errors="coerce").fillna(0.0)
    links["link_mass"] = links["jurisdiction_count"] * links["link_mass_share"]
    links["implied_rate"] = np.where(
        links["footprint_pop"].fillna(0.0) > 0,
        links["jurisdiction_count"] / links["footprint_pop"].where(links["footprint_pop"] > 0, 1.0) * 100_000.0,
        np.nan,
    )
    links["rate_ratio"] = links["implied_rate"] / float(national_rate)
    links["footprint_over_service_pop"] = np.where(
        links["service_pop"].fillna(0.0) > 0,
        links["footprint_pop"] / links["service_pop"].where(links["service_pop"] > 0, 1.0),
        np.nan,
    )
    has_service_pop = links["service_pop"].fillna(0.0).gt(0.0)
    service_pop_bad = (~has_service_pop) | links["footprint_over_service_pop"].lt(
        float(STAGE2_SERVICE_POP_LOW_RATIO)
    ) | links["footprint_over_service_pop"].gt(float(STAGE2_SERVICE_POP_HIGH_RATIO))
    links["reviewed_registry_row"] = links["ori9"].astype("string").str.upper().isin(reviewed_oris)

    concentration = links[
        links["link_mass"].gt(0.0)
        & links["rate_ratio"].gt(float(STAGE2_RATE_RATIO_THRESHOLD))
        & service_pop_bad
    ].copy()
    dilution = links[
        links["link_mass"].gt(0.0)
        & has_service_pop
        & links["footprint_over_service_pop"].ge(float(STAGE2_SERVICE_POP_HIGH_RATIO))
    ].copy()

    report_columns = [
        "ori9",
        "state_abbr",
        "agency_name_std",
        "agency_type_norm",
        "jurisdiction_id",
        "jurisdiction_name",
        "resolution_source",
        "footprint_pop",
        "service_pop",
        "footprint_over_service_pop",
        "jurisdiction_count",
        "agency_reported_count",
        "link_mass",
        "implied_rate",
        "rate_ratio",
    ]
    unreviewed: dict[str, pd.DataFrame] = {}
    for label, frame, sort_col, ascending in (
        ("concentration", concentration, "rate_ratio", False),
        ("dilution", dilution, "footprint_over_service_pop", False),
    ):
        bad = frame[~frame["reviewed_registry_row"]].sort_values(
            sort_col, ascending=ascending, kind="mergesort"
        )
        unreviewed[label] = bad
        if not bad.empty:
            _append_total_lane_issue(
                issues,
                "total_lane.stage2_footprint_plausibility: "
                f"{label} hit on an exclusive municipal crosswalk link with no reviewed registry row "
                "(configs/local_resolution_overrides.csv or configs/consolidated_agency_footprints.csv)",
                bad,
                columns=report_columns,
            )
    return {
        "ok": all(frame.empty for frame in unreviewed.values()),
        "registries": registry_summary,
        "thresholds": {
            "rate_ratio": float(STAGE2_RATE_RATIO_THRESHOLD),
            "service_pop_low_ratio": float(STAGE2_SERVICE_POP_LOW_RATIO),
            "service_pop_high_ratio": float(STAGE2_SERVICE_POP_HIGH_RATIO),
            "national_reference_rate_per_100k": float(national_rate),
            "in_scope_counts": float(in_scope_counts),
            "in_scope_population": float(in_scope_pop),
        },
        "links_screened": int(len(links)),
        "concentration_hits": int(len(concentration)),
        "concentration_hits_unreviewed": int(len(unreviewed["concentration"])),
        "dilution_hits": int(len(dilution)),
        "dilution_hits_unreviewed": int(len(unreviewed["dilution"])),
        "concentration_sample": _sample_records(
            concentration.sort_values("rate_ratio", ascending=False, kind="mergesort"),
            columns=[*report_columns, "reviewed_registry_row"],
        ),
        "dilution_sample": _sample_records(
            dilution.sort_values("footprint_over_service_pop", ascending=False, kind="mergesort"),
            columns=[*report_columns, "reviewed_registry_row"],
        ),
    }


def _component_control_reconciliation(
    *,
    output_dir: Path,
    controls: pd.DataFrame,
    issues: list[str],
    contract: ReleaseContract = LEGACY_RELEASE_CONTRACT,
) -> dict[str, Any]:
    component_audit_path = (
        output_dir / f"allocation_component_denominator_audit_{YEAR}.parquet"
    )
    if not component_audit_path.exists():
        issues.append(
            "total_lane.published_output_total_reconciliation: "
            f"missing allocation component audit {component_audit_path}"
        )
        return {"ok": False, "path": str(component_audit_path), "present": False}
    component_columns = [
        "state_fips",
        "jurisdiction_id",
        "jurisdiction_type",
        "offense",
        "component_count_before",
        "component_count_after",
    ]
    if contract.service_wide_custom_allocation_enabled:
        component_columns.extend(["service_scope_id", "source_state_fips"])
    components = pd.read_parquet(component_audit_path, columns=component_columns)
    components["state_fips"] = components["state_fips"].astype("string").str.zfill(2)
    components["jurisdiction_id"] = components["jurisdiction_id"].astype("string")
    components["offense"] = components["offense"].astype("string")
    components = components[
        ~components["state_fips"].isin(RELEASE_EXCLUDED_STATE_FIPS)
    ].copy()
    unlocated = _unlocated_mass_table(output_dir)

    # ``component_count_before`` is the control-attributed mass. The allocator then evacuates
    # water-only Census cells to land recipients in the same county/offense pool. That spatial
    # repair is intentionally allowed to cross the synthetic municipal/remainder/overlap labels,
    # so post-repair placement mass cannot be compared to jurisdiction controls. Assert the two
    # contracts separately: pre-repair attribution equals every control, while before/after mass
    # is conserved exactly within each state/offense and the published surfaces test the final
    # spatial placement below.
    spatial_conservation = (
        components.groupby(["state_fips", "offense"], dropna=False)[
            ["component_count_before", "component_count_after"]
        ]
        .sum()
        .reset_index()
    )
    spatial_conservation["delta"] = pd.to_numeric(
        spatial_conservation["component_count_after"], errors="coerce"
    ).fillna(0.0) - pd.to_numeric(
        spatial_conservation["component_count_before"], errors="coerce"
    ).fillna(0.0)
    if contract.service_wide_custom_allocation_enabled and unlocated is not None:
        service_terminal = unlocated[
            unlocated.get("route_reason", pd.Series("", index=unlocated.index))
            .astype("string")
            .eq("service_no_eligible_receiver")
        ].copy()
        if not service_terminal.empty:
            if "source_state_fips" not in service_terminal.columns:
                issues.append(
                    "total_lane.published_output_total_reconciliation: service terminal "
                    "unlocated rows are missing source_state_fips"
                )
                service_terminal = service_terminal.iloc[0:0].copy()
            elif service_terminal["source_state_fips"].astype("string").fillna("").str.strip().eq("").any():
                issues.append(
                    "total_lane.published_output_total_reconciliation: service terminal "
                    "unlocated rows have blank source_state_fips"
                )
                service_terminal = service_terminal.iloc[0:0].copy()
        if not service_terminal.empty:
            terminal_state = (
                service_terminal.assign(
                    state_fips=service_terminal["source_state_fips"]
                    .astype("string")
                    .str.zfill(2)
                )
                .groupby(["state_fips", "offense"], dropna=False)["unlocated_count"]
                .sum()
                .rename("terminal_unlocated_count")
                .reset_index()
            )
            spatial_conservation = spatial_conservation.merge(
                terminal_state, on=["state_fips", "offense"], how="outer"
            )
            spatial_conservation["delta"] = pd.to_numeric(
                spatial_conservation["delta"], errors="coerce"
            ).fillna(0.0) + pd.to_numeric(
                spatial_conservation["terminal_unlocated_count"], errors="coerce"
            ).fillna(0.0)
    spatial_bad = spatial_conservation[
        spatial_conservation["delta"].abs().gt(TOTAL_LANE_TOLERANCE)
    ].copy()
    if not spatial_bad.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.published_output_total_reconciliation: post-repair component placement does not conserve state/offense mass",
            spatial_bad.sort_values(
                "delta", key=lambda s: s.abs(), ascending=False, kind="mergesort"
            ),
            columns=[
                "state_fips",
                "offense",
                "component_count_before",
                "component_count_after",
                "delta",
            ],
        )

    overlap_component = (
        components["jurisdiction_type"]
        .astype("string")
        .str.contains("overlap", na=False)
    )
    overlap_control_state = components["state_fips"].copy()
    if contract.service_wide_custom_allocation_enabled:
        service_component = components["service_scope_id"].notna() & components[
            "service_scope_id"
        ].astype("string").str.strip().ne("")
        overlap_control_state = overlap_control_state.where(
            ~service_component,
            components["source_state_fips"].astype("string").str.zfill(2),
        )
    components.loc[overlap_component, "jurisdiction_id"] = (
        overlap_control_state.loc[overlap_component] + ":statewide_overlap_layer"
    )
    remainder_component = (
        components["jurisdiction_type"]
        .astype("string")
        .isin(
            ["localized_remainder_county_layer", "localized_remainder_residual_layer"]
        )
    )
    components.loc[remainder_component, "jurisdiction_id"] = (
        components.loc[remainder_component, "state_fips"]
        + ":state_nonmunicipal_remainder"
    )
    component_sums = (
        components.groupby(["jurisdiction_id", "offense"], dropna=False)[
            "component_count_before"
        ]
        .sum()
        .rename("component_total")
        .reset_index()
    )
    # Unlocated-mass lane: the conservation identity becomes
    # component_total + unlocated == control (UNLOCATED_MASS_IDENTITY), so the
    # withheld mass is credited back before the delta. Absent table = lane off =
    # legacy identity unchanged.
    if unlocated is not None:
        credit_rows = unlocated.copy()
        if contract.service_wide_custom_allocation_enabled:
            terminal = credit_rows.get(
                "route_reason", pd.Series("", index=credit_rows.index)
            ).astype("string").eq("service_no_eligible_receiver")
            if bool(terminal.any()):
                if "source_state_fips" not in credit_rows.columns:
                    issues.append(
                        "total_lane.published_output_total_reconciliation: service terminal "
                        "unlocated rows are missing source_state_fips"
                    )
                else:
                    source_state = credit_rows.loc[terminal, "source_state_fips"].astype(
                        "string"
                    ).fillna("").str.strip()
                    if bool(source_state.eq("").any()):
                        issues.append(
                            "total_lane.published_output_total_reconciliation: service terminal "
                            "unlocated rows have blank source_state_fips"
                        )
                    else:
                        credit_rows.loc[terminal, "jurisdiction_id"] = (
                            source_state.str.zfill(2) + ":statewide_overlap_layer"
                        )
        credit = (
            credit_rows.groupby(["jurisdiction_id", "offense"], dropna=False)[
                "unlocated_count"
            ]
            .sum()
            .rename("unlocated_credit")
            .reset_index()
        )
        component_sums = component_sums.merge(
            credit, on=["jurisdiction_id", "offense"], how="left"
        )
        component_sums["component_total"] = component_sums[
            "component_total"
        ] + pd.to_numeric(
            component_sums.pop("unlocated_credit"), errors="coerce"
        ).fillna(0.0)
    spatial_sums = (
        components.groupby(["jurisdiction_id", "offense"], dropna=False)[
            "component_count_after"
        ]
        .sum()
        .rename("post_repair_spatial_total")
        .reset_index()
    )
    control_target_column = (
        "control_target"
        if "control_target" in controls.columns
        else TOTAL_LANE_TARGET_COLUMN
    )
    target = controls[~controls["state_fips"].isin(RELEASE_EXCLUDED_STATE_FIPS)][
        ["jurisdiction_id", "offense", control_target_column]
    ].copy()
    target = target.rename(columns={control_target_column: "control_target"})
    merged = target.merge(
        component_sums, on=["jurisdiction_id", "offense"], how="outer"
    ).merge(spatial_sums, on=["jurisdiction_id", "offense"], how="outer")
    merged["target_delta"] = pd.to_numeric(
        merged["component_total"], errors="coerce"
    ).fillna(0.0) - pd.to_numeric(merged["control_target"], errors="coerce").fillna(0.0)
    merged["post_repair_spatial_delta_nonblocking"] = pd.to_numeric(
        merged["post_repair_spatial_total"], errors="coerce"
    ).fillna(0.0) - pd.to_numeric(merged["control_target"], errors="coerce").fillna(0.0)
    bad_rows = merged[merged["target_delta"].abs().gt(TOTAL_LANE_TOLERANCE)].copy()
    if not bad_rows.empty:
        _append_total_lane_issue(
            issues,
            "total_lane.published_output_total_reconciliation: allocation component totals do not match modeled controls",
            bad_rows.sort_values(
                "target_delta", key=lambda s: s.abs(), ascending=False, kind="mergesort"
            ),
            columns=[
                "jurisdiction_id",
                "offense",
                "component_total",
                "control_target",
                "target_delta",
            ],
        )
    return {
        "ok": bad_rows.empty and spatial_bad.empty,
        "path": str(component_audit_path),
        "present": True,
        "component_rows": int(len(components)),
        "jurisdiction_offense_rows": int(len(component_sums)),
        "jurisdiction_control_component_column": "component_count_before",
        "post_repair_spatial_component_column": "component_count_after",
        "target_column": "control_target",
        "target_reconciliation_bad_rows": int(len(bad_rows)),
        "max_abs_target_delta": _max_abs(merged["target_delta"]),
        "post_repair_spatial_jurisdiction_diff_rows_nonblocking": int(
            merged["post_repair_spatial_delta_nonblocking"]
            .abs()
            .gt(TOTAL_LANE_TOLERANCE)
            .sum()
        ),
        "max_abs_post_repair_spatial_jurisdiction_delta_nonblocking": _max_abs(
            merged["post_repair_spatial_delta_nonblocking"]
        ),
        "post_repair_state_offense_conservation_bad_rows": int(len(spatial_bad)),
        "max_abs_post_repair_state_offense_conservation_delta": _max_abs(
            spatial_conservation["delta"]
        ),
        "offending_rows_sample": _sample_records(
            bad_rows,
            columns=[
                "jurisdiction_id",
                "offense",
                "component_total",
                "control_target",
                "target_delta",
            ],
        ),
    }


def _published_surface_state_total_reconciliation(
    *,
    output_dir: Path,
    controls: pd.DataFrame,
    issues: list[str],
    contract: ReleaseContract = LEGACY_RELEASE_CONTRACT,
) -> dict[str, Any]:
    control_target_column = (
        "control_target"
        if "control_target" in controls.columns
        else TOTAL_LANE_TARGET_COLUMN
    )
    target = (
        controls.groupby(["state_fips", "offense"], dropna=False)[control_target_column]
        .sum()
        .rename("state_control_total")
        .reset_index()
    )
    transfer_net = pd.DataFrame(columns=["state_fips", "offense", "transfer_net_incoming"])
    if contract.service_wide_custom_allocation_enabled:
        transfer_path = _service_state_transfer_path(output_dir)
        if transfer_path.exists():
            transfer = pd.read_parquet(
                transfer_path,
                columns=[
                    "service_scope_id",
                    "canonical_target_ori",
                    "offense",
                    "source_state_fips",
                    "allocation_state_fips",
                    "source_target_count",
                    "allocation_count",
                    "route_reason",
                ],
            )
            transfer["source_state_fips"] = (
                transfer["source_state_fips"].astype("string").str.zfill(2)
            )
            transfer["allocation_state_fips"] = (
                transfer["allocation_state_fips"].astype("string").str.zfill(2)
            )
            incoming = (
                transfer.groupby(["allocation_state_fips", "offense"], dropna=False)[
                    "allocation_count"
                ]
                .sum()
                .rename("incoming")
                .reset_index()
                .rename(columns={"allocation_state_fips": "state_fips"})
            )
            located_transfer = transfer[
                transfer["allocation_state_fips"].notna()
                & transfer["allocation_state_fips"].astype("string").str.strip().ne("")
            ]
            outgoing = (
                located_transfer.groupby(
                    ["source_state_fips", "offense"], dropna=False
                )["allocation_count"]
                .sum()
                .rename("outgoing")
                .reset_index()
                .rename(columns={"source_state_fips": "state_fips"})
            )
            transfer_net = incoming.merge(
                outgoing, on=["state_fips", "offense"], how="outer"
            ).fillna({"incoming": 0.0, "outgoing": 0.0})
            transfer_net["transfer_net_incoming"] = (
                pd.to_numeric(transfer_net["incoming"], errors="coerce").fillna(0.0)
                - pd.to_numeric(transfer_net["outgoing"], errors="coerce").fillna(0.0)
            )
            transfer_net = transfer_net[["state_fips", "offense", "transfer_net_incoming"]]
    # Select the exact identity declared by the producer. The withheld side comes from the
    # companion table, while service transfers reconcile geographic placement back to the source
    # state control.
    identity = (
        SERVICE_TRANSFER_MASS_IDENTITY
        if contract.service_wide_custom_allocation_enabled
        else LEGACY_MASS_IDENTITY
    )
    withheld = pd.DataFrame(columns=["state_fips", "offense", "unlocated_count"])
    if contract.unlocated_mass:
        table = _unlocated_mass_table(output_dir)
        if table is None:
            # Absence is already an issue in `_check_unlocated_mass`; do not double-report it,
            # but do NOT silently fall back to the narrow identity either -- the surface is short
            # of its controls by construction and would fail below, which is the correct outcome.
            table = withheld
        identity = UNLOCATED_MASS_IDENTITY
        withheld = (
            table.assign(state_fips=table["state_fips"].astype("string").str.zfill(2))
            .groupby(["state_fips", "offense"], dropna=False)["unlocated_count"]
            .sum()
            .reset_index()
            if not table.empty
            else withheld
        )
    surface_summaries: dict[str, Any] = {}
    for geography, path in [
        (
            "block_group_ags_core",
            output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet",
        ),
        ("tract_ags_core", output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet"),
    ]:
        if not path.exists():
            issues.append(
                f"total_lane.published_output_total_reconciliation: missing {geography} surface {path}"
            )
            surface_summaries[geography] = {
                "ok": False,
                "path": str(path),
                "present": False,
            }
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
        long["offense"] = long["expected_count_field"].str.removeprefix(
            "expected_count_"
        )
        sums = (
            long.groupby(["state_fips", "offense"], dropna=False)["published_total"]
            .sum()
            .reset_index()
        )
        merged = (
            target.merge(sums, on=["state_fips", "offense"], how="outer")
            .merge(withheld, on=["state_fips", "offense"], how="left")
            .merge(transfer_net, on=["state_fips", "offense"], how="left")
        )
        merged["unlocated_count"] = pd.to_numeric(
            merged.get("unlocated_count"), errors="coerce"
        ).fillna(0.0)
        merged["transfer_net_incoming"] = pd.to_numeric(
            merged.get("transfer_net_incoming"), errors="coerce"
        ).fillna(0.0)
        merged["delta"] = (
            pd.to_numeric(merged["published_total"], errors="coerce").fillna(0.0)
            + merged["unlocated_count"]
            - merged["transfer_net_incoming"]
            - pd.to_numeric(merged["state_control_total"], errors="coerce").fillna(0.0)
        )
        report_columns = [
            "state_fips",
            "offense",
            "published_total",
            "unlocated_count",
            "transfer_net_incoming",
            "state_control_total",
            "delta",
        ]
        bad_rows = merged[merged["delta"].abs().gt(TOTAL_LANE_TOLERANCE)].copy()
        if not bad_rows.empty:
            _append_total_lane_issue(
                issues,
                f"total_lane.published_output_total_reconciliation: {geography} state/offense sums do not satisfy {identity}",
                bad_rows.sort_values(
                    "delta", key=lambda s: s.abs(), ascending=False, kind="mergesort"
                ),
                columns=report_columns,
            )
        surface_summaries[geography] = {
            "ok": bad_rows.empty,
            "path": str(path),
            "present": True,
            "identity": identity,
            "state_offense_rows": int(len(merged)),
            "bad_rows": int(len(bad_rows)),
            "max_abs_delta": _max_abs(merged["delta"]),
            "unlocated_total": float(merged["unlocated_count"].sum()),
            "offending_rows_sample": _sample_records(bad_rows, columns=report_columns),
        }
    return {
        "ok": all(surface.get("ok") is True for surface in surface_summaries.values()),
        "identity": identity,
        "surfaces": surface_summaries,
    }


def _check_published_output_total_reconciliation(
    *,
    output_dir: Path,
    controls: pd.DataFrame,
    issues: list[str],
    contract: ReleaseContract = LEGACY_RELEASE_CONTRACT,
) -> dict[str, Any]:
    component_summary = _component_control_reconciliation(
        output_dir=output_dir,
        controls=controls,
        issues=issues,
        contract=contract,
    )
    surface_summary = _published_surface_state_total_reconciliation(
        output_dir=output_dir,
        controls=controls,
        issues=issues,
        contract=contract,
    )
    return {
        "ok": component_summary.get("ok") is True and surface_summary.get("ok") is True,
        "component_to_jurisdiction_control": component_summary,
        "published_bg_tract_state_totals": surface_summary,
    }


def _stale_qa_dependencies(path: Path, dependency_paths: list[Path]) -> list[str]:
    """Content-addressed staleness, falling back to the mtime rule when unstamped.

    The QA artifact is byte-identical across rebuilds whenever its inputs are unchanged,
    and rebuilding it costs ~99 s, so the decision to recompute is keyed to the input
    bytes rather than to a promotion that only touched mtimes.
    """
    stamp_path = dependency_stamp_path(path)
    recorded: dict[str, Any] | None = None
    if stamp_path.exists():
        try:
            payload = json.loads(stamp_path.read_text())
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("dependencies"), dict):
            if payload.get("artifact") == content_digest(path):
                recorded = payload["dependencies"]
    if recorded is None:
        return [str(p) for p in dependency_paths if path.stat().st_mtime < p.stat().st_mtime]
    current = dependency_digests(dependency_paths)
    # Compared by content, not by where the content lived when the stamp was written: the same
    # tree checked out at a second path is not a changed input. See crimerisk.build_freshness.
    if digest_multiset(recorded) == digest_multiset(current):
        return []
    recorded_digests = set(recorded.values())
    current_digests = set(current.values())
    return sorted(
        {key for key, digest in current.items() if digest not in recorded_digests}
        | {key for key, digest in recorded.items() if digest not in current_digests}
    )


def _check_city_feed_exact_point_tripwire(*, issues: list[str]) -> dict[str, Any]:
    path = REPO_CITY_EXACT_POINT_QA
    exception_path = REPO_CITY_EXACT_POINT_EXCEPTIONS
    share_path = REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet"
    next_phase_path = REPO_ROOT / "state" / "modeling" / f"next_phase_validation_city_incident_share_surface_{YEAR}.parquet"
    dependency_paths = [p for p in [share_path, next_phase_path, exception_path] if p.exists()]
    if not path.exists():
        issues.append(f"total_lane.city_feed_exact_point_tripwire: missing QA artifact {path}")
        return {"ok": False, "path": str(path), "present": False}

    stale_dependencies = _stale_qa_dependencies(path, dependency_paths)
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


def _check_total_lane_qa(
    *,
    output_dir: Path,
    issues: list[str],
    contract: ReleaseContract = LEGACY_RELEASE_CONTRACT,
) -> dict[str, Any]:
    controls, controls_summary = _load_controls_for_total_lane(issues=issues)
    if controls is None:
        return {
            "ok": False,
            "controls": controls_summary,
        }
    allocation_controls, allocation_control_summary = _allocation_control_targets(
        output_dir=output_dir,
        issues=issues,
    )
    if allocation_controls is None:
        return {
            "ok": False,
            "controls": controls_summary,
            "allocation_controls": allocation_control_summary,
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
        controls=allocation_controls,
        issues=issues,
        contract=contract,
    )
    city_exact_point_summary = _check_city_feed_exact_point_tripwire(issues=issues)
    county_plausibility_summary = _check_county_level_plausibility(output_dir=output_dir, issues=issues)
    stage2_footprint_summary = _check_stage2_footprint_plausibility(controls=controls, issues=issues)
    checks = {
        "no_duplicate_control_totals": duplicate_summary,
        "source_priority_honored": source_priority_summary,
        "state_remainder_reconciliation": state_reconciliation_summary,
        "high_population_spot_checks": high_population_summary,
        "consolidated_agency_population_detector": consolidated_agency_summary,
        "published_output_total_reconciliation": published_reconciliation_summary,
        "city_feed_exact_point_tripwire": city_exact_point_summary,
        "county_level_plausibility": county_plausibility_summary,
        "stage2_footprint_plausibility": stage2_footprint_summary,
    }
    return {
        "ok": all(check.get("ok") is True for check in checks.values()),
        "controls": controls_summary,
        "allocation_controls": allocation_control_summary,
        "target_column": allocation_control_summary.get("target_column"),
        "tolerance": float(TOTAL_LANE_TOLERANCE),
        "checks": checks,
    }


def _check_rare_offense_tract_support(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    """Assert the rare-offense publication-support rule as a single cross-surface statement: the
    block-group murder/rape per-offense index/rate points are null while their expected counts are
    retained, and the tract surface carries the murder/rape indices at a broad, published scale
    (docs/archive/2026-09/STATE.md decision record). Per-surface count-derived checks live in _check_surface; this
    is the explicit policy gate."""
    summary: dict[str, Any] = {"variants": {}}
    ok = True
    for variant in ("ags_core", "cde_exact_sensitivity"):
        bg_path = _published_surface_path(output_dir, geography="block_group", variant=variant)
        tr_path = _published_surface_path(output_dir, geography="tract", variant=variant)
        if not (bg_path.exists() and tr_path.exists()):
            issues.append(f"rare_offense_tract_support: missing surface(s) for {variant}")
            ok = False
            continue
        variant_summary: dict[str, Any] = {}
        point_cols: list[str] = []
        count_cols: list[str] = []
        for offense in RARE_OFFENSE_TRACT_SUPPORT:
            point_cols.extend(_rare_offense_point_fields(offense))
            count_cols.append(f"expected_count_{offense}")
        bg = pd.read_parquet(bg_path, columns=["block_group_geoid", *point_cols, *count_cols])
        tr = pd.read_parquet(tr_path, columns=["tract_id", *[f"index_{o}_primary" for o in RARE_OFFENSE_TRACT_SUPPORT]])
        for offense in RARE_OFFENSE_TRACT_SUPPORT:
            populated_points = [
                field
                for field in _rare_offense_point_fields(offense)
                if pd.to_numeric(bg[field], errors="coerce").notna().any()
            ]
            if populated_points:
                issues.append(
                    f"rare_offense_tract_support ({variant}): block-group {offense} index/rate points "
                    f"must be null but are populated: {populated_points}"
                )
                ok = False
            bg_count = pd.to_numeric(bg[f"expected_count_{offense}"], errors="coerce")
            if bg_count.isna().any():
                issues.append(f"rare_offense_tract_support ({variant}): block-group expected_count_{offense} has nulls")
                ok = False
            tract_index_populated = int(pd.to_numeric(tr[f"index_{offense}_primary"], errors="coerce").notna().sum())
            if tract_index_populated < 1000:
                issues.append(
                    f"rare_offense_tract_support ({variant}): tract index_{offense}_primary is populated on only "
                    f"{tract_index_populated} rows — the rare offense is not carried at tract support"
                )
                ok = False
            variant_summary[offense] = {
                "block_group_expected_count_sum": float(bg_count.fillna(0.0).sum()),
                "tract_index_populated_rows": tract_index_populated,
            }
        summary["variants"][variant] = variant_summary
    summary["ok"] = ok
    return summary


def _check_soft_shrinkage(
    *, contract: ReleaseContract, state_output_dir: Path, issues: list[str]
) -> dict[str, Any]:
    """PLAN item 5's mirror: the compressor is RE-DERIVED from what the artifact published.

    The lane's whole claim is an identity between two numbers the component audit publishes on every
    affected row -- the unit's measured within-jurisdiction rate ratio and the ratio it was allowed
    to keep. This recomputes the second from the first using this file's own transcription of the
    compressor and of `nu`, so a build that declared the lane while shipping hard-capped numbers
    (or shipped soft numbers under a different `nu` than it declared) fails the release.

    Both directions are checked. A manifest that does NOT declare the lane must not carry the
    lane's audit columns, which is the half-migration guard every other lane here has.
    """
    label = "release_contract.soft_shrinkage"
    audit_path = state_output_dir / f"allocation_component_denominator_audit_{YEAR}.parquet"
    summary: dict[str, Any] = {
        "declared": contract.soft_shrinkage,
        "audit_path": str(audit_path),
        "audit_present": audit_path.exists(),
    }
    if not audit_path.exists():
        if contract.soft_shrinkage:
            issues.append(f"{label}: declared but the component audit {audit_path} is absent")
        return summary

    available = _surface_column_names(audit_path)
    present = [column for column in SOFT_SHRINKAGE_AUDIT_COLUMNS if column in available]
    summary["audit_columns_present"] = present
    if not contract.soft_shrinkage:
        if present:
            issues.append(
                f"{label}: the manifest does not declare the lane but the component audit carries "
                f"its columns {present} -- a half-migrated artifact, not a legacy one"
            )
        return summary

    if sorted(present) != sorted(SOFT_SHRINKAGE_AUDIT_COLUMNS):
        issues.append(
            f"{label}: declared but the component audit is missing "
            f"{sorted(set(SOFT_SHRINKAGE_AUDIT_COLUMNS) - set(present))}"
        )
        return summary

    if contract.soft_shrinkage_envelope_mode != SOFT_SHRINKAGE_ENVELOPE_MODE:
        issues.append(
            f"{label}: manifest envelope_mode is {contract.soft_shrinkage_envelope_mode!r}, "
            f"expected {SOFT_SHRINKAGE_ENVELOPE_MODE!r}"
        )
    for name, declared, expected in (
        ("nu", contract.soft_shrinkage_nu, SOFT_SHRINKAGE_NU),
        (
            "extrapolation_weight",
            contract.soft_shrinkage_extrapolation_weight,
            SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT,
        ),
    ):
        if declared is None or not np.isclose(float(declared), expected, rtol=0.0, atol=1e-12):
            issues.append(
                f"{label}: manifest {name} is {declared!r}, expected the ratified {expected!r}"
            )

    audit = pd.read_parquet(
        audit_path,
        columns=[
            "offense",
            "allocation_envelope_policy",
            "allocation_envelope_cap_ratio",
            *SOFT_SHRINKAGE_AUDIT_COLUMNS,
        ],
    )
    modes = set(audit[SOFT_SHRINKAGE_MODE_COLUMN].dropna().astype(str))
    if modes - {SOFT_SHRINKAGE_ENVELOPE_MODE}:
        issues.append(f"{label}: component audit carries envelope modes {sorted(modes)}")

    ratio = pd.to_numeric(audit[SOFT_SHRINKAGE_RATIO_COLUMN], errors="coerce")
    retained = pd.to_numeric(audit[SOFT_SHRINKAGE_RETAINED_COLUMN], errors="coerce")
    cap = pd.to_numeric(audit["allocation_envelope_cap_ratio"], errors="coerce")
    scored = ratio.notna() & retained.notna() & cap.notna()
    summary["source_rows_scored"] = int(scored.sum())
    if not bool(scored.any()):
        # No unit anywhere exceeded a cap. Legal, and worth recording rather than passing silently.
        summary["max_abs_retained_error"] = 0.0
        return summary

    nu = float(SOFT_SHRINKAGE_NU)
    r = ratio[scored].to_numpy(dtype=float)
    c = cap[scored].to_numpy(dtype=float)
    expected_retained = np.where(r > c, c * np.exp(nu * np.log1p(np.log(r / c) / nu)), r)
    error = np.abs(retained[scored].to_numpy(dtype=float) - expected_retained)
    relative = error / np.maximum(np.abs(expected_retained), 1e-12)
    summary["max_abs_retained_error"] = float(error.max())
    summary["max_rel_retained_error"] = float(relative.max())
    if float(relative.max()) > 1e-9:
        issues.append(
            f"{label}: the retained rate ratio the audit publishes does not recompute from the "
            f"declared compressor (max relative error {relative.max():.3g})"
        )
    # The lane keeps ordering and always removes mass: strictly above the cap, strictly below the
    # measured ratio, wherever the unit exceeded its cap at all.
    over = r > c
    if bool(over.any()):
        kept = retained[scored].to_numpy(dtype=float)[over]
        if np.any(kept <= c[over]) or np.any(kept >= r[over]):
            issues.append(
                f"{label}: a retained ratio left the open interval (cap, measured ratio) -- the "
                f"compressor must always shrink and never truncate"
            )
    caps_by_offense = (
        audit.loc[scored, ["offense", "allocation_envelope_cap_ratio"]]
        .assign(offense=lambda frame: frame["offense"].astype(str))
        .groupby("offense")["allocation_envelope_cap_ratio"]
        .agg(lambda values: sorted(set(values.round(9))))
        .to_dict()
    )
    summary["cap_ratio_by_offense"] = {key: list(value) for key, value in caps_by_offense.items()}
    for offense, values in caps_by_offense.items():
        expected_cap = MODEL_ONLY_RATE_RATIO_CAPS.get(offense)
        if expected_cap is None or list(values) != [expected_cap]:
            issues.append(
                f"{label}: offense {offense!r} was repaired at cap ratios {values}, expected "
                f"{[expected_cap] if expected_cap else 'no soft-envelope repair'}"
            )
    return summary


def _check_unlocated_mass(
    *, contract: ReleaseContract, state_output_dir: Path, issues: list[str]
) -> dict[str, Any]:
    """Amendment 3 item 5's mirror: the companion table is checked as a TABLE, not as a claim.

    The conservation identity itself is asserted where every other conservation identity is
    asserted -- in `_published_surface_state_total_reconciliation`, which widens by exactly this
    table when the lane is declared. What is checked here is that the table is a well-formed
    statement in its own right: the declared shape, non-negative counts that never exceed the
    control they were withheld from, and a route decomposition that adds back up to the bucket.

    Both directions, like every other lane: a manifest that does not declare the lane must not
    ship the companion table, and one that does must not ship the legacy identity string.
    """
    label = "release_contract.unlocated_mass"
    path = _unlocated_mass_path(state_output_dir)
    summary: dict[str, Any] = {
        "declared": contract.unlocated_mass,
        "companion_table_path": str(path),
        "companion_table_present": path.exists(),
        "declared_identity": contract.unlocated_mass_identity,
    }
    if not contract.unlocated_mass:
        if path.exists():
            issues.append(
                f"{label}: the manifest does not declare the lane but the companion table {path} "
                f"is present -- a half-migrated artifact, not a legacy one"
            )
        expected_identity = (
            SERVICE_TRANSFER_MASS_IDENTITY
            if contract.service_wide_custom_allocation_enabled
            else LEGACY_MASS_IDENTITY
        )
        if contract.manifest_present and contract.unlocated_mass_identity not in (
            None,
            expected_identity,
        ):
            issues.append(
                f"{label}: the lane is off but the manifest declares the identity "
                f"{contract.unlocated_mass_identity!r}, expected {expected_identity!r}"
            )
        return summary

    if not path.exists():
        issues.append(f"{label}: declared but the companion table {path} is absent")
        return summary
    if contract.unlocated_mass_identity != UNLOCATED_MASS_IDENTITY:
        issues.append(
            f"{label}: manifest identity is {contract.unlocated_mass_identity!r}, expected "
            f"{UNLOCATED_MASS_IDENTITY!r}"
        )
    if contract.unlocated_mass_companion_table != path.name:
        issues.append(
            f"{label}: manifest names companion table {contract.unlocated_mass_companion_table!r}, "
            f"but the edition carries {path.name!r}"
        )
    if sorted(contract.unlocated_mass_routes) != sorted(UNLOCATED_MASS_ROUTES):
        issues.append(
            f"{label}: manifest declares unlocated routes {sorted(contract.unlocated_mass_routes)}, "
            f"expected {sorted(UNLOCATED_MASS_ROUTES)}"
        )
    # The bucket is a source-resolution statement. A build that relabelled it as a confidence
    # statement would be publishing doubt language under a lane that promises not to.
    if contract.unlocated_mass_semantics != "source_resolution":
        issues.append(
            f"{label}: manifest semantics is {contract.unlocated_mass_semantics!r}, expected "
            f"'source_resolution'"
        )

    table = _unlocated_mass_table(state_output_dir)
    assert table is not None
    missing = [column for column in UNLOCATED_MASS_COLUMNS if column not in table.columns]
    if missing:
        issues.append(f"{label}: companion table is missing columns {missing}")
        return summary
    summary["rows"] = int(len(table))

    unlocated = pd.to_numeric(table["unlocated_count"], errors="coerce").fillna(0.0)
    control = pd.to_numeric(table["control_count"], errors="coerce").fillna(0.0)
    summary["unlocated_total"] = float(unlocated.sum())
    summary["control_total"] = float(control.sum())
    summary["unlocated_share_of_control"] = (
        float(unlocated.sum() / control.sum()) if float(control.sum()) > 0 else 0.0
    )
    summary["unlocated_by_offense"] = {
        str(offense): float(value)
        for offense, value in table.groupby("offense")["unlocated_count"].sum().items()
    }
    summary["unlocated_by_route"] = {
        route: float(pd.to_numeric(table[f"unlocated_count_{route}"], errors="coerce").fillna(0.0).sum())
        for route in UNLOCATED_MASS_ROUTES
    }

    if bool((unlocated < -TOTAL_LANE_TOLERANCE).any()):
        issues.append(f"{label}: companion table carries negative unlocated counts")
    over = unlocated - control
    if bool((over > TOTAL_LANE_TOLERANCE).any()):
        issues.append(
            f"{label}: unlocated mass exceeds the control it was withheld from on "
            f"{int((over > TOTAL_LANE_TOLERANCE).sum())} rows (max excess {float(over.max()):.6g})"
        )
    types = set(table["jurisdiction_type"].dropna().astype(str))
    if types - {UNLOCATED_MASS_JURISDICTION_TYPE}:
        issues.append(f"{label}: companion table carries jurisdiction types {sorted(types)}")
    expected_ids = table["state_fips"].astype(str) + ":" + UNLOCATED_MASS_JURISDICTION_TYPE
    if not table["jurisdiction_id"].astype(str).eq(expected_ids).all():
        issues.append(f"{label}: companion table jurisdiction_id does not key to its state")
    route_sum = sum(
        pd.to_numeric(table[f"unlocated_count_{route}"], errors="coerce").fillna(0.0)
        for route in UNLOCATED_MASS_ROUTES
    )
    route_drift = float((route_sum - unlocated).abs().max() or 0.0)
    summary["max_abs_route_decomposition_delta"] = route_drift
    if route_drift > TOTAL_LANE_TOLERANCE:
        issues.append(
            f"{label}: the route decomposition does not reconstruct the bucket "
            f"(max abs delta {route_drift:.3g})"
        )
    share = pd.to_numeric(table["unlocated_share_of_control"], errors="coerce").fillna(0.0)
    expected_share = np.where(control.gt(0.0), unlocated / control.replace(0.0, np.nan), 0.0)
    share_drift = float(np.abs(share.to_numpy(dtype=float) - np.nan_to_num(expected_share)).max() or 0.0)
    summary["max_abs_share_delta"] = share_drift
    if share_drift > 1e-9:
        issues.append(
            f"{label}: the published share is not the ratio of the two counts beside it "
            f"(max abs delta {share_drift:.3g})"
        )
    return summary


def _check_release_contract(
    *, contract: ReleaseContract, state_output_dir: Path, issues: list[str]
) -> dict[str, Any]:
    """The lane-level assertions that are about the BUILD rather than about one surface."""
    summary: dict[str, Any] = {
        "manifest_path": str(contract.manifest_path) if contract.manifest_path else None,
        "manifest_present": contract.manifest_present,
        "lanes": sorted(contract.lanes),
    }
    summary["soft_shrinkage"] = _check_soft_shrinkage(
        contract=contract, state_output_dir=state_output_dir, issues=issues
    )
    summary["unlocated_mass"] = _check_unlocated_mass(
        contract=contract, state_output_dir=state_output_dir, issues=issues
    )
    summary["service_state_transfer"] = _check_service_state_transfer(
        contract=contract, state_output_dir=state_output_dir, issues=issues
    )
    summary["custom_footprint_fraction_basis"] = _check_custom_footprint_fraction_contract(
        contract=contract, state_output_dir=state_output_dir, issues=issues
    )
    if contract.count_first_composites and contract.count_first_version != COUNT_FIRST_COMPOSITES_VERSION:
        issues.append(
            f"release_contract: manifest count_first_composites version is "
            f"{contract.count_first_version!r}, expected {COUNT_FIRST_COMPOSITES_VERSION!r}"
        )
    if contract.special_use_taxonomy:
        if contract.special_use_version != SPECIAL_USE_TAXONOMY_VERSION:
            issues.append(
                f"release_contract: manifest special_use_taxonomy version is "
                f"{contract.special_use_version!r}, expected {SPECIAL_USE_TAXONOMY_VERSION!r}"
            )
        # The thresholds a quoted cell is re-derivable from, checked against this file's own
        # transcription of the taxonomy contract rather than read out of the producer.
        expected_thresholds = {
            "household_floor": NON_RESIDENTIAL_HOUSEHOLD_FLOOR,
            "employment_floor": PERSON_EXPOSURE_DENOMINATOR_FLOOR,
            "group_quarters_population_share_min": SPECIAL_USE_GROUP_QUARTERS_SHARE_MIN,
            "education_job_share_min": SPECIAL_USE_EDUCATION_JOB_SHARE_MIN,
            "postsecondary_anchor_min": SPECIAL_USE_POSTSECONDARY_ANCHOR_MIN,
            "open_natural_land_cover_share_min": SPECIAL_USE_OPEN_NATURAL_SHARE_MIN,
            "unexplained_daytime_presence_ratio_min": SPECIAL_USE_UNEXPLAINED_DAYTIME_RATIO_MIN,
        }
        drift = {
            name: (contract.special_use_thresholds.get(name), value)
            for name, value in expected_thresholds.items()
            if contract.special_use_thresholds.get(name) != value
        }
        if drift:
            issues.append(
                f"release_contract: special-use thresholds in the manifest disagree with the "
                f"taxonomy contract (manifest, expected): {drift}"
            )
    if contract.zero_resident_opportunity_floor:
        # The floor is a property of the BUILD, so the build has to declare it. A v2 build whose
        # manifest is silent about the rule fails here rather than being quietly exempted from it.
        policy = contract.zero_resident_opportunity_policy
        if not policy:
            issues.append(
                "release_contract: the manifest declares a lane that publishes opportunity rates "
                "on near-zero-resident cells but carries no zero_resident_opportunity_rate_policy "
                "block"
            )
        else:
            expected_policy = {
                "enabled": True,
                "resident_threshold": PERSON_EXPOSURE_DENOMINATOR_FLOOR,
                "opportunity_normalizer_floor": ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR,
                "floor_estimate_mode": "insufficient_exposure",
            }
            policy_drift = {
                name: (policy.get(name), value)
                for name, value in expected_policy.items()
                if policy.get(name) != value
            }
            if policy_drift:
                issues.append(
                    "release_contract: zero_resident_opportunity_rate_policy in the manifest "
                    f"disagrees with the contract (manifest, expected): {policy_drift}"
                )
            if sorted(policy.get("offenses") or []) != sorted(OFFENSES_7):
                issues.append(
                    "release_contract: zero_resident_opportunity_rate_policy must apply to all "
                    f"seven offenses, found {policy.get('offenses')!r}"
                )
        summary["zero_resident_opportunity_rate_policy"] = dict(policy)
    if contract.exposure_ensemble:
        summary["exposure_ensemble"] = _check_exposure_normalizer_recomposition(
            contract=contract, issues=issues
        )
        # The normalizers are extensive counts of exposure, so the tract surface must carry the
        # sum of its block groups' -- the same roll-up identity the counts are held to.
        bg_path = _published_surface_path(state_output_dir, geography="block_group", variant="ags_core")
        tract_path = _published_surface_path(state_output_dir, geography="tract", variant="ags_core")
        if bg_path.exists() and tract_path.exists():
            normalizer_columns = [
                _exposure_normalizer_column(offense) for offense in EXPOSURE_ENSEMBLE_OFFENSES
            ]
            bg = pd.read_parquet(bg_path, columns=["block_group_geoid", *normalizer_columns])
            tract = pd.read_parquet(tract_path, columns=["tract_id", *normalizer_columns])
            bg = bg.assign(tract_id=bg["block_group_geoid"].astype("string").str.slice(0, 11))
            rollup = bg.groupby("tract_id")[normalizer_columns].sum()
            merged = tract.set_index(tract["tract_id"].astype("string"))[normalizer_columns].join(
                rollup, rsuffix="_bg", how="outer"
            )
            rollup_deltas: dict[str, float] = {}
            for column in normalizer_columns:
                delta = _max_abs(
                    pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
                    - pd.to_numeric(merged[f"{column}_bg"], errors="coerce").fillna(0.0)
                )
                rollup_deltas[column] = delta
                if delta > 1e-6:
                    issues.append(
                        f"release_contract: tract {column} does not equal the block-group rollup "
                        f"(max abs diff {delta:.3e})"
                    )
            summary["exposure_ensemble"]["tract_rollup_max_abs"] = rollup_deltas
    if contract.mixture_allocation:
        manifest = _load_json(contract.manifest_path) if contract.manifest_path else None
        block = ((manifest or {}).get("resolved_config") or {}).get("mixture_allocation") or {}
        for key in ("experts_path", "weights_path", "contract"):
            if not block.get(key):
                issues.append(
                    f"release_contract: mixture_allocation is enabled but the manifest block has "
                    f"no {key}; the allocation that moved the counts is not self-identifying"
                )
        summary["mixture_allocation"] = {
            key: block.get(key) for key in ("contract", "experts_path", "weights_path")
        }
    return summary



# --- the edition rollup gate ---------------------------------------------------------------
#
# Same contract-aware shape as the surface gate above: the assertion set is READ OFF THE
# EDITION'S OWN MANIFEST. `edition.json` names the rollup geographies the edition claims to
# publish and `rollups/rollup_summary.json` names them again with their measured conservation;
# this gate holds the edition to the geographies IT declares, and treats a disagreement between
# the two declarations (or between either and the files on disk) as a release failure rather than
# quietly validating whichever set happens to pass.
#
# Three assertions, all real mirrors recomputed from the published tables:
#
#   1. CONSERVATION IS RE-DERIVED, not read. Rolled counts are summed off the rollup parquet and
#      national counts off the edition's own block-group surface; the residual is the NAMED
#      `outside_universe_total` the summary publishes. `rolled + named residual == national`,
#      offense by offense, plus population.
#   2. RATES ARE RECOMPUTED FROM COUNTS on a sample of rows at every support --
#      `100000 * count / denominator`, then `100 * rate / the national rate published beside it`,
#      in both the primary and resident lanes -- and a row whose publication flag is false must
#      carry a null rate and a null index. This is the count-derived index policy (docs/archive/2026-09/STATE.md)
#      asserted at the support the value is quoted at.
#   3. THE FIELD DICTIONARY IS CURRENT vs the surfaces. The generator already fails closed on a
#      column it cannot describe; what that cannot catch is a dictionary generated against an
#      older schema and shipped beside a newer table. Every published column must appear in the
#      edition's own FIELD_DICTIONARY.md, no documented column may be absent from the table, and
#      the doc's declared column counts must match the schemas.

# Rollup sums and the national reference are the same float64 additions in a different order, so
# the identity holds to summation epsilon. Floor of 1e-9 absolute, widened to 4 ULP of the
# national total where that is larger: one ULP at the larceny total (5.4e6) is 9.3e-10 and the
# measured worst case on 2024A-annual is exactly one ULP, so a flat 1e-9 would be a knife-edge
# gate on the largest offenses while remaining the binding constraint on the smallest (one ULP at
# the murder total of 1.7e4 is 1.8e-12).
ROLLUP_CONSERVATION_ABS_TOLERANCE = 1e-9
ROLLUP_CONSERVATION_ULP_MULTIPLE = 4.0
# Rows spot-checked per support. Evenly spaced over the sorted table rather than the head, so the
# sample spans the geography instead of one state's worth of it, and deterministic so a failing
# release names the same rows on every run.
ROLLUP_RATE_SAMPLE_ROWS = 60
# A recomputation of the producer's own arithmetic in the same float64: equality to a few ULP.
ROLLUP_RATE_REL_TOLERANCE = 1e-12
ROLLUP_SUPPORT_COLUMN = "rollup_support"
DICTIONARY_SECTION_PATTERN = re.compile(r"^## (?P<label>.+?) \((?P<geography>[a-z_]+)\)\s*$")
DICTIONARY_COLUMN_PATTERN = re.compile(r"^\| `(?P<column>[A-Za-z0-9_]+)` \|")
DICTIONARY_SURFACE_ROW_PATTERN = re.compile(
    r"^\| (?P<label>[^|]+?) \| (?P<geography>[a-z_]+) \| (?P<columns>[\d,]+) \| "
    r"(?P<rows>[\d,]*) \| `(?P<file>[^`]+)` \|\s*$"
)


def _conservation_tolerance(reference: float) -> float:
    return max(
        ROLLUP_CONSERVATION_ABS_TOLERANCE,
        ROLLUP_CONSERVATION_ULP_MULTIPLE * float(np.spacing(abs(float(reference)) or 1.0)),
    )


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(str(path)).schema_arrow.names)


def _column_sums(path: Path, columns: list[str]) -> dict[str, float]:
    present = [column for column in columns if column in set(_parquet_columns(path))]
    frame = pd.read_parquet(path, columns=present)
    return {
        column: float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())
        for column in present
    }


def _edition_geography_id_column(geography: str) -> str:
    return {
        "county": "county_geoid",
        "cbsa": "cbsa_code",
        "zcta": "zcta5",
        "state": "state_fips",
        "block_group": "block_group_geoid",
        "tract": "tract_id",
    }[geography]


def _edition_published_tables(
    *, edition_dir: Path, edition: dict[str, Any], rollup_summary: dict[str, Any]
) -> dict[str, Path]:
    """Every table the edition publishes, geography -> path, resolved edition-relative.

    The summary records absolute build-time paths; an edition directory must be movable, so the
    basename is resolved inside this edition rather than followed.
    """
    tables: dict[str, Path] = {}
    for geography, entry in (rollup_summary.get("geographies") or {}).items():
        tables[str(geography)] = edition_dir / "rollups" / Path(str(entry.get("path"))).name
    for record in edition.get("geoparquet") or []:
        geography = str(record.get("geography"))
        if geography:
            tables[geography] = edition_dir / str(record.get("path"))
    if "block_group" not in tables or "tract" not in tables:
        for geography, record in (edition.get("source_surfaces") or {}).items():
            if geography not in tables and record.get("path"):
                tables[str(geography)] = Path(str(record["path"]))
    return tables


def _check_edition_rollup_conservation(
    *,
    geography: str,
    rollup_path: Path,
    entry: dict[str, Any],
    national: dict[str, float],
    population_col: str,
    issues: list[str],
) -> dict[str, Any]:
    """`rolled + named residual == national`, re-derived from the published tables."""
    report = entry.get("conservation") or {}
    count_columns = [f"expected_count_{offense}" for offense in OFFENSES_7]
    rolled = _column_sums(rollup_path, [*count_columns, population_col])
    result: dict[str, Any] = {
        "support": geography,
        "rollup_path": str(rollup_path),
        "units": int(entry.get("rows") or 0),
        "covers_universe": bool(report.get("covers_universe")),
        "block_groups_outside_universe": report.get("block_groups_outside_universe"),
        "offenses": {},
    }
    worst = 0.0
    for offense in OFFENSES_7:
        column = f"expected_count_{offense}"
        offense_report = (report.get("offenses") or {}).get(offense) or {}
        if "outside_universe_total" not in offense_report:
            issues.append(
                f"edition_rollups[{geography}]: the summary names no outside_universe residual for "
                f"{offense}; uncovered mass must be a named residual, never a dropped tail"
            )
            continue
        residual = float(offense_report["outside_universe_total"])
        reference = float(national.get(column, 0.0))
        actual = float(rolled.get(column, 0.0))
        difference = actual + residual - reference
        tolerance = _conservation_tolerance(reference)
        worst = max(worst, abs(difference))
        if abs(difference) > tolerance:
            issues.append(
                f"edition_rollups[{geography}]: {column} does not conserve -- rolled "
                f"{actual!r} + named residual {residual!r} - national {reference!r} = "
                f"{difference:.3e} (tolerance {tolerance:.3e})"
            )
        # The summary's own claimed rollup total must be the total actually written.
        claimed = offense_report.get("rollup_total")
        if claimed is not None and abs(float(claimed) - actual) > _conservation_tolerance(actual):
            issues.append(
                f"edition_rollups[{geography}]: rollup_summary claims a {column} total of "
                f"{float(claimed)!r} but the published table sums to {actual!r}"
            )
        if bool(report.get("covers_universe")) and residual != 0.0:
            issues.append(
                f"edition_rollups[{geography}]: geography covers the universe but names a "
                f"{column} residual of {residual!r}"
            )
        result["offenses"][offense] = {
            "rolled": actual,
            "named_residual": residual,
            "national": reference,
            "difference": difference,
            "tolerance": tolerance,
        }
    population_report = report.get("population") or {}
    if "outside_universe_total" in population_report:
        residual = float(population_report["outside_universe_total"])
        reference = float(national.get(population_col, 0.0))
        actual = float(rolled.get(population_col, 0.0))
        difference = actual + residual - reference
        tolerance = _conservation_tolerance(reference)
        worst = max(worst, abs(difference))
        if abs(difference) > tolerance:
            issues.append(
                f"edition_rollups[{geography}]: {population_col} does not conserve -- rolled "
                f"{actual!r} + named residual {residual!r} - national {reference!r} = "
                f"{difference:.3e} (tolerance {tolerance:.3e})"
            )
        result["population"] = {
            "rolled": actual,
            "named_residual": residual,
            "national": reference,
            "difference": difference,
        }
    else:
        issues.append(
            f"edition_rollups[{geography}]: the summary names no outside_universe population residual"
        )
    # A geography that does not cover the country must SAY so and account for the block groups it
    # leaves out; a silent zero there is the dropped tail this gate exists to catch.
    outside_block_groups = report.get("block_groups_outside_universe")
    if not bool(report.get("covers_universe")):
        if not outside_block_groups:
            issues.append(
                f"edition_rollups[{geography}]: geography does not cover the universe but reports "
                "no block groups outside it"
            )
    elif outside_block_groups:
        issues.append(
            f"edition_rollups[{geography}]: geography covers the universe but reports "
            f"{outside_block_groups} block groups outside it"
        )
    result["max_abs_conservation_difference"] = worst
    return result


def _rollup_sample_index(row_count: int, *, limit: int = ROLLUP_RATE_SAMPLE_ROWS) -> list[int]:
    if row_count <= 0:
        return []
    if row_count <= limit:
        return list(range(row_count))
    return sorted({int(value) for value in np.linspace(0, row_count - 1, limit).round()})


def _check_edition_rollup_rates(
    *, geography: str, rollup_path: Path, issues: list[str]
) -> dict[str, Any]:
    """Recompute rate and index from the counts and denominators published beside them."""
    columns = set(_parquet_columns(rollup_path))
    id_column = _edition_geography_id_column(geography)
    wanted = [id_column, "land_area_sq_mi", COMMON_DENOMINATOR_COLUMN]
    for offense in OFFENSES_7:
        wanted += [
            f"expected_count_{offense}",
            f"primary_denominator_{offense}",
            f"primary_national_rate_per_100k_{offense}",
            f"primary_index_publishable_{offense}",
            f"rate_{offense}_primary",
            f"index_{offense}_primary",
            f"resident_national_rate_per_100k_{offense}",
            f"index_{offense}_resident_publishable",
            f"rate_{offense}_resident",
            f"index_{offense}_resident",
            f"crime_density_{offense}",
        ]
    wanted += [f"expected_count_{name}" for name in AGGREGATES]
    missing = [column for column in wanted if column not in columns]
    if missing:
        issues.append(
            f"edition_rollups[{geography}]: published rollup is missing "
            f"{len(missing)} count-derivation columns: {missing[:8]}"
        )
    frame = pd.read_parquet(rollup_path, columns=[c for c in wanted if c in columns])
    sample = frame.iloc[_rollup_sample_index(len(frame))].reset_index(drop=True)
    checked = 0

    def _mismatch(actual: pd.Series, expected: pd.Series, mask: pd.Series) -> pd.Series:
        scale = expected.abs().where(expected.abs() > 0.0, 1.0)
        return mask & (actual - expected).abs().gt(ROLLUP_RATE_REL_TOLERANCE * scale)

    def _report(label: str, bad: pd.Series, actual: pd.Series, expected: pd.Series) -> None:
        if not bool(bad.any()):
            return
        rows = sample.loc[bad, [id_column]].copy()
        rows["published"] = actual[bad]
        rows["recomputed_from_counts"] = expected[bad]
        issues.append(
            f"edition_rollups[{geography}]: {label} is not recomputable from the counts and "
            f"denominators published beside it ({int(bad.sum())} of {len(sample)} sampled rows); "
            f"sample={_sample_records(rows, limit=5)}"
        )

    for offense in OFFENSES_7:
        count = pd.to_numeric(sample[f"expected_count_{offense}"], errors="coerce")
        for lane, denominator_column, national_column in (
            ("primary", f"primary_denominator_{offense}", f"primary_national_rate_per_100k_{offense}"),
            ("resident", COMMON_DENOMINATOR_COLUMN, f"resident_national_rate_per_100k_{offense}"),
        ):
            flag_column = (
                f"primary_index_publishable_{offense}"
                if lane == "primary"
                else f"index_{offense}_resident_publishable"
            )
            rate_column = f"rate_{offense}_{lane}"
            index_column = f"index_{offense}_{lane}"
            if not {flag_column, rate_column, index_column, denominator_column}.issubset(columns):
                continue
            publishable = sample[flag_column].fillna(False).astype(bool)
            denominator = pd.to_numeric(sample[denominator_column], errors="coerce")
            rate = pd.to_numeric(sample[rate_column], errors="coerce")
            index = pd.to_numeric(sample[index_column], errors="coerce")
            national_rate = pd.to_numeric(sample[national_column], errors="coerce")
            expected_rate = 100000.0 * count / denominator
            _report(f"{rate_column}", _mismatch(rate, expected_rate, publishable), rate, expected_rate)
            expected_index = 100.0 * rate / national_rate
            _report(
                f"{index_column}", _mismatch(index, expected_index, publishable), index, expected_index
            )
            withheld = ~publishable & (rate.notna() | index.notna())
            if bool(withheld.any()):
                issues.append(
                    f"edition_rollups[{geography}]: {int(withheld.sum())} sampled rows carry a "
                    f"{rate_column}/{index_column} value with {flag_column} false; a withheld "
                    "value must be null, not a number"
                )
            checked += int(publishable.sum())
        density_column = f"crime_density_{offense}"
        if density_column in columns and "land_area_sq_mi" in columns:
            land = pd.to_numeric(sample["land_area_sq_mi"], errors="coerce")
            density = pd.to_numeric(sample[density_column], errors="coerce")
            expected_density = count / land.where(land.gt(0.0))
            _report(
                density_column,
                _mismatch(density, expected_density, expected_density.notna()),
                density,
                expected_density,
            )

    for label, members in (
        ("personal", PERSONAL_OFFENSES),
        ("property", PROPERTY_OFFENSES),
        ("total", OFFENSES_7),
    ):
        column = f"expected_count_{label}"
        if column not in columns:
            continue
        actual = pd.to_numeric(sample[column], errors="coerce")
        expected = sum(
            pd.to_numeric(sample[f"expected_count_{offense}"], errors="coerce") for offense in members
        )
        _report(column, _mismatch(actual, expected, actual.notna()), actual, expected)

    return {
        "support": geography,
        "rows": int(len(frame)),
        "sampled_rows": int(len(sample)),
        "publishable_lane_values_checked": checked,
        "relative_tolerance": ROLLUP_RATE_REL_TOLERANCE,
    }


def _documented_dictionary(text: str) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Columns documented per geography, and the column count the doc's own surface table claims."""
    documented: dict[str, set[str]] = {}
    declared: dict[str, int] = {}
    current: str | None = None
    for line in text.splitlines():
        surface_row = DICTIONARY_SURFACE_ROW_PATTERN.match(line)
        if surface_row is not None:
            declared[surface_row.group("geography")] = int(
                surface_row.group("columns").replace(",", "")
            )
            continue
        section = DICTIONARY_SECTION_PATTERN.match(line)
        if section is not None:
            current = section.group("geography")
            documented.setdefault(current, set())
            continue
        if line.startswith("## "):
            current = None
            continue
        if current is not None:
            column = DICTIONARY_COLUMN_PATTERN.match(line)
            if column is not None:
                documented[current].add(column.group("column"))
    return documented, declared


def _check_edition_field_dictionary(
    *, edition_dir: Path, tables: dict[str, Path], issues: list[str]
) -> dict[str, Any]:
    """The dictionary shipped in the edition must describe exactly what the edition publishes."""
    dictionary_path = edition_dir / "FIELD_DICTIONARY.md"
    if not dictionary_path.exists():
        issues.append(f"edition_rollups: the edition ships no field dictionary at {dictionary_path}")
        return {"present": False, "path": str(dictionary_path)}
    documented, declared = _documented_dictionary(dictionary_path.read_text())
    summary: dict[str, Any] = {
        "present": True,
        "path": str(dictionary_path),
        "documented_geographies": sorted(documented),
        "surfaces": {},
    }
    for geography in sorted(tables):
        path = tables[geography]
        if not path.exists():
            continue
        # Geometry is added by the packaging step, not published by the surface the dictionary was
        # generated from, so it is not a documented field and its absence is not staleness.
        published = {
            column for column in _parquet_columns(path) if column != EDITION_GEOMETRY_COLUMN
        }
        described = documented.get(geography)
        if described is None:
            issues.append(
                f"edition_rollups[{geography}]: the edition publishes {path.name} but the field "
                "dictionary has no section for that geography"
            )
            continue
        undocumented = sorted(published - described)
        stale = sorted(described - published)
        if undocumented:
            issues.append(
                f"edition_rollups[{geography}]: {len(undocumented)} published columns are absent "
                f"from the field dictionary: {undocumented[:8]}"
            )
        if stale:
            issues.append(
                f"edition_rollups[{geography}]: the field dictionary documents "
                f"{len(stale)} columns the published table does not carry: {stale[:8]}"
            )
        claimed = declared.get(geography)
        if claimed is not None and claimed != len(described):
            issues.append(
                f"edition_rollups[{geography}]: the dictionary's surface table claims {claimed} "
                f"columns but its own section describes {len(described)}"
            )
        summary["surfaces"][geography] = {
            "published_columns": len(published),
            "documented_columns": len(described),
            "declared_columns": claimed,
            "undocumented": undocumented[:8],
            "stale": stale[:8],
        }
    return summary


def _check_edition_coverage_scenarios(*, edition_dir: Path, issues: list[str]) -> dict[str, Any]:
    """Amendment 3 item 6's gate: the scenario ladder, its identity, and its per-state deltas.

    Four things are asserted, all recomputed from the tables rather than read off their summary:

    1. `observed <= central <= upper`, EXACTLY -- every row, not on aggregate. A ladder that only
       holds after summing is not a ladder; it is two errors cancelling.
    2. The central scenario's NON-IMPUTED mass equals the published surface's non-imputed mass.
       This is what makes "central is the published surface, centred" true: the correction moves
       the imputed component and nothing else.
    3. The per-state delta table reconstructs from the jurisdiction table.
    4. Every published share is the ratio of the two counts published beside it.

    The gate is skipped for an edition that declares no scenarios, and fails for one that declares
    them without shipping the tables -- the same both-directions rule the lane checks use.
    """
    label = "edition_coverage_scenarios"
    summary: dict[str, Any] = {"edition_dir": str(edition_dir)}
    edition = _load_json(edition_dir / "edition.json") or {}
    declared = edition.get("coverage_scenarios")
    scenario_dir = edition_dir / COVERAGE_SCENARIOS_DIRNAME
    scenario_path = scenario_dir / f"coverage_scenarios_{YEAR}.parquet"
    delta_path = scenario_dir / f"coverage_scenario_state_deltas_{YEAR}.parquet"
    summary["declared"] = declared is not None
    summary["scenarios_present"] = scenario_path.exists()
    summary["state_deltas_present"] = delta_path.exists()
    if declared is None and not scenario_path.exists():
        summary["checked"] = False
        return summary
    if declared is None:
        issues.append(
            f"{label}: the edition ships {scenario_path.name} but edition.json declares no "
            f"coverage_scenarios block -- a half-migrated edition, not a legacy one"
        )
        return summary
    for what, path in (("scenario table", scenario_path), ("state delta table", delta_path)):
        if not path.exists():
            issues.append(f"{label}: edition.json declares coverage scenarios but the {what} {path} is absent")
            return summary
    summary["checked"] = True
    if declared.get("block_group_scenarios_published") is not False:
        issues.append(
            f"{label}: edition.json must record block_group_scenarios_published=false -- the "
            f"bounds were measured at jurisdiction support and this lane does not push them down"
        )
    if declared.get("support") != "jurisdiction":
        issues.append(f"{label}: declared support is {declared.get('support')!r}, expected 'jurisdiction'")

    scenarios = pd.read_parquet(scenario_path)
    missing = [column for column in COVERAGE_SCENARIO_COLUMNS if column not in scenarios.columns]
    if missing:
        issues.append(f"{label}: the scenario table is missing columns {missing}")
        return summary
    summary["jurisdiction_offense_rows"] = int(len(scenarios))

    def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    observed = _numeric(scenarios, "count_observed")
    central = _numeric(scenarios, "count_central")
    upper = _numeric(scenarios, "count_upper")
    published = _numeric(scenarios, "count_published")
    imputed = _numeric(scenarios, "benchmark_imputed_count")
    imputed_central = _numeric(scenarios, "benchmark_imputed_central")

    low_violations = int((central - observed < -TOTAL_LANE_TOLERANCE).sum())
    high_violations = int((upper - central < -TOTAL_LANE_TOLERANCE).sum())
    summary["ordering_violations"] = {
        "observed_gt_central": low_violations,
        "central_gt_upper": high_violations,
    }
    if low_violations:
        issues.append(f"{label}: {low_violations} rows have central < observed")
    if high_violations:
        issues.append(f"{label}: {high_violations} rows have upper < central")

    non_imputed_drift = float(((central - imputed_central) - (published - imputed)).abs().max() or 0.0)
    summary["max_abs_non_imputed_identity_delta"] = non_imputed_drift
    if non_imputed_drift > TOTAL_LANE_TOLERANCE:
        issues.append(
            f"{label}: the central scenario's non-imputed mass is not the published surface's "
            f"(max abs delta {non_imputed_drift:.3g}) -- the correction moved mass it does not own"
        )
    lane_drift = float(
        (
            _numeric(scenarios, "benchmark_imputed_municipal")
            + _numeric(scenarios, "benchmark_imputed_county")
            - imputed
        )
        .abs()
        .max()
        or 0.0
    )
    summary["max_abs_lane_split_delta"] = lane_drift
    if lane_drift > TOTAL_LANE_TOLERANCE:
        issues.append(
            f"{label}: the municipal/county lane split does not reconstruct the imputed mass "
            f"(max abs delta {lane_drift:.3g})"
        )
    summary["national"] = {
        "observed": float(observed.sum()),
        "central": float(central.sum()),
        "upper": float(upper.sum()),
        "published": float(published.sum()),
        "benchmark_imputed": float(imputed.sum()),
    }

    deltas = pd.read_parquet(delta_path)
    missing_delta = [c for c in COVERAGE_SCENARIO_STATE_DELTA_COLUMNS if c not in deltas.columns]
    if missing_delta:
        issues.append(f"{label}: the state delta table is missing columns {missing_delta}")
        return summary
    summary["state_offense_rows"] = int(len(deltas))
    value_columns = ["count_published", "count_observed", "count_central", "count_upper", "benchmark_imputed_count"]
    recomputed = (
        scenarios.groupby(["state_fips", "offense"], dropna=False)[value_columns].sum().reset_index()
    )
    per_offense = deltas[~deltas["offense"].astype(str).eq(COVERAGE_SCENARIO_ALL_KEY)]
    merged = recomputed.merge(
        per_offense, on=["state_fips", "offense"], how="outer", suffixes=("_recomputed", "")
    )
    delta_drift = 0.0
    for column in value_columns:
        drift = (
            pd.to_numeric(merged[f"{column}_recomputed"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
        ).abs()
        delta_drift = max(delta_drift, float(drift.max() or 0.0))
    summary["max_abs_state_delta_reconstruction"] = delta_drift
    if delta_drift > TOTAL_LANE_TOLERANCE:
        issues.append(
            f"{label}: the per-state delta table does not reconstruct from the jurisdiction table "
            f"(max abs delta {delta_drift:.3g})"
        )
    # The `ALL` rows are the state's own offense rows summed, which is what makes a share of "the
    # state's total" a real share and not a differently-scoped number.
    all_rows = deltas[deltas["offense"].astype(str).eq(COVERAGE_SCENARIO_ALL_KEY)]
    summed = per_offense.groupby("state_fips", dropna=False)[value_columns].sum().reset_index()
    all_merged = summed.merge(all_rows, on="state_fips", how="outer", suffixes=("_recomputed", ""))
    all_drift = 0.0
    for column in value_columns:
        drift = (
            pd.to_numeric(all_merged[f"{column}_recomputed"], errors="coerce").fillna(0.0)
            - pd.to_numeric(all_merged[column], errors="coerce").fillna(0.0)
        ).abs()
        all_drift = max(all_drift, float(drift.max() or 0.0))
    summary["max_abs_all_offense_row_delta"] = all_drift
    if all_drift > TOTAL_LANE_TOLERANCE:
        issues.append(
            f"{label}: the ALL-offense rows are not the state's offense rows summed "
            f"(max abs delta {all_drift:.3g})"
        )

    share_drift = 0.0
    denominator = pd.to_numeric(deltas["count_published"], errors="coerce").fillna(0.0)
    for share_column, numerator_expr in (
        ("imputed_share_of_published", pd.to_numeric(deltas["benchmark_imputed_count"], errors="coerce").fillna(0.0)),
        ("coverage_supplied_share_of_published", pd.to_numeric(deltas["coverage_supplied_count"], errors="coerce").fillna(0.0)),
        ("scenario_dependent_share_of_published", pd.to_numeric(deltas["scenario_dependent_count"], errors="coerce").fillna(0.0)),
        ("central_minus_published_share", pd.to_numeric(deltas["central_minus_published"], errors="coerce").fillna(0.0)),
    ):
        expected = np.where(denominator.ne(0.0), numerator_expr / denominator.replace(0.0, np.nan), 0.0)
        published_share = pd.to_numeric(deltas[share_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        drift = float(np.abs(published_share - np.nan_to_num(expected)).max() or 0.0)
        share_drift = max(share_drift, drift)
        if drift > 1e-9:
            issues.append(
                f"{label}: {share_column} is not the ratio of the counts published beside it "
                f"(max abs delta {drift:.3g})"
            )
    summary["max_abs_share_delta"] = share_drift
    return summary


def _check_edition_rollups(*, edition_dir: Path, issues: list[str]) -> dict[str, Any]:
    """The rollup section of the release gate, asserted against the edition's own contract."""
    summary: dict[str, Any] = {"edition_dir": str(edition_dir)}
    edition = _load_json(edition_dir / "edition.json")
    rollup_summary = _load_json(edition_dir / "rollups" / "rollup_summary.json")
    if edition is None:
        issues.append(f"edition_rollups: no edition.json under {edition_dir}")
        return {**summary, "present": False}
    if rollup_summary is None:
        issues.append(f"edition_rollups: no rollups/rollup_summary.json under {edition_dir}")
        return {**summary, "present": False}
    summary["present"] = True
    summary["edition_id"] = edition.get("edition_id")
    summary["rollup_version"] = rollup_summary.get("version")
    summary["coverage_universe"] = (rollup_summary.get("coverage_universe") or {}).get("description")
    population_col = f"population_{int(rollup_summary.get('year'))}"

    # The assertion set is the edition's own: both declarations must name the same geographies.
    declared_in_edition = sorted((edition.get("rollups") or {}).keys())
    declared_in_summary = sorted((rollup_summary.get("geographies") or {}).keys())
    summary["geographies"] = declared_in_summary
    if declared_in_edition != declared_in_summary:
        issues.append(
            f"edition_rollups: edition.json declares rollups {declared_in_edition} but "
            f"rollup_summary.json declares {declared_in_summary}; a release cannot be validated "
            "against two different contracts"
        )
    if not declared_in_summary:
        issues.append("edition_rollups: the edition declares no rollup geography")
        return summary

    tables = _edition_published_tables(
        edition_dir=edition_dir, edition=edition, rollup_summary=rollup_summary
    )
    block_group_path = tables.get("block_group")
    national: dict[str, float] = {}
    if block_group_path is None or not Path(block_group_path).exists():
        issues.append(
            "edition_rollups: the edition names no readable block-group surface, so conservation "
            "cannot be re-derived (only re-read)"
        )
    else:
        national = _column_sums(
            Path(block_group_path),
            [*[f"expected_count_{offense}" for offense in OFFENSES_7], population_col],
        )
        summary["national_reference_surface"] = str(block_group_path)

    conservation: dict[str, Any] = {}
    rates: dict[str, Any] = {}
    for geography in declared_in_summary:
        entry = (rollup_summary.get("geographies") or {})[geography]
        rollup_path = tables.get(geography)
        if rollup_path is None or not Path(rollup_path).exists():
            issues.append(
                f"edition_rollups[{geography}]: the edition declares this rollup but "
                f"{rollup_path} is not present"
            )
            continue
        rollup_path = Path(rollup_path)
        rows = int(entry.get("rows") or 0)
        actual_rows = int(pd.read_parquet(rollup_path, columns=[ROLLUP_SUPPORT_COLUMN]).shape[0])
        if rows != actual_rows:
            issues.append(
                f"edition_rollups[{geography}]: the summary claims {rows} rows but the published "
                f"table has {actual_rows}"
            )
        if national:
            conservation[geography] = _check_edition_rollup_conservation(
                geography=geography,
                rollup_path=rollup_path,
                entry=entry,
                national=national,
                population_col=population_col,
                issues=issues,
            )
        rates[geography] = _check_edition_rollup_rates(
            geography=geography, rollup_path=rollup_path, issues=issues
        )
    summary["conservation"] = conservation
    summary["rates_from_counts"] = rates
    summary["field_dictionary"] = _check_edition_field_dictionary(
        edition_dir=edition_dir, tables=tables, issues=issues
    )
    return summary


def _check_footprint_mass_conservation(*, output_dir: Path, issues: list[str]) -> dict[str, Any]:
    """Per-ORI conservation on the custom-footprint overlap layer.

    The municipal lane publishes exactly the control it was given; so must this one. Every
    footprint ORI's placed mass is compared against its own admitted `final_control_mass`
    carried onto the published control surface, and anything more than 5% off blocks the
    release. The two non-placing statuses are held to the complementary rule: a suppressed
    duplicate and a footprint with no placeable support must place NOTHING, because their mass
    is published in the unlocated table instead.
    """
    path = footprint_mass_conservation_path(output_dir, year=YEAR)
    summary: dict[str, Any] = {"path": str(path), "present": path.exists()}
    if not path.exists():
        issues.append(f"footprint_mass_conservation: missing per-ORI conservation table {path}")
        summary["ok"] = False
        return summary
    frame = pd.read_parquet(path)
    summary["rows"] = int(len(frame))
    summary["ori_count"] = int(frame["ori9"].nunique())
    summary["max_relative_error_threshold"] = float(
        FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR
    )
    summary["status_counts"] = {
        str(status): int(count) for status, count in frame["footprint_status"].value_counts().items()
    }
    relative_error = pd.to_numeric(frame["relative_error"], errors="coerce")
    placed = pd.to_numeric(frame["placed_mass"], errors="coerce").fillna(0.0)
    expected = pd.to_numeric(frame["expected_placed_mass"], errors="coerce").fillna(0.0)
    summary["placed_mass_total"] = float(placed.sum())
    summary["expected_placed_mass_total"] = float(expected.sum())
    summary["ledger_control_mass_total"] = float(
        pd.to_numeric(frame["ledger_control_mass"], errors="coerce").fillna(0.0).sum()
    )
    finite_error = relative_error.replace([np.inf, -np.inf], np.nan)
    summary["max_relative_error"] = (
        float(finite_error.max()) if bool(finite_error.notna().any()) else 0.0
    )

    # The gate, as stated: per ORI, over all seven offences.
    by_ori = _footprint_conservation_by_ori(frame)
    ori_error = pd.to_numeric(by_ori["relative_error"], errors="coerce")
    over_ori = by_ori[ori_error.gt(float(FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR))]
    summary["oris_over_threshold"] = int(len(over_ori))
    summary["max_ori_relative_error"] = (
        float(ori_error.replace([np.inf, -np.inf], np.nan).max()) if len(by_ori) else 0.0
    )
    if len(over_ori):
        issues.append(
            f"footprint_mass_conservation: {len(over_ori)} footprint ORI(s) place mass more than "
            f"{FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR:.0%} away from their own ledger "
            "control; sample="
            + str(
                over_ori.sort_values("relative_error", ascending=False)
                .head(10)
                .to_dict(orient="records")
            )
        )
    # The same band per offence, floored at two whole offences. This is a tripwire, not the gate:
    # the model-only rate-ratio cap clips one offence's modelled rate in a block group and hands
    # the clipped mass to the rest of its conservation group, which for a statewide-overlap agency
    # is the whole state, so a single offence row can move by more than a count while the ORI the
    # gate measures stays well inside the band. It is therefore reported whenever the per-ORI gate
    # passes and only blocks alongside a per-ORI failure, where it names the offence that broke
    # the rake. See FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR in crimerisk.allocation.
    absolute_error = (
        pd.to_numeric(frame["placed_mass"], errors="coerce").fillna(0.0)
        - pd.to_numeric(frame["expected_placed_mass"], errors="coerce").fillna(0.0)
    ).abs()
    over = frame[
        relative_error.gt(float(FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR))
        & absolute_error.gt(float(FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR))
    ]
    over_sample = (
        over.sort_values("relative_error", ascending=False)
        .head(10)[
            [
                "ori9",
                "state_fips",
                "offense",
                "footprint_status",
                "ledger_control_mass",
                "expected_placed_mass",
                "placed_mass",
                "relative_error",
            ]
        ]
        .to_dict(orient="records")
    )
    summary["rows_over_threshold"] = int(len(over))
    summary["min_absolute_error_threshold"] = float(FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR)
    # Advisory exactly when the blocking per-ORI gate passed.
    per_offense_blocking = bool(len(over_ori))
    summary["per_offense_check_blocking"] = per_offense_blocking
    summary["rows_over_threshold_advisory"] = 0 if per_offense_blocking else int(len(over))
    if len(over):
        summary["rows_over_threshold_sample"] = over_sample
    if len(over) and per_offense_blocking:
        issues.append(
            f"footprint_mass_conservation: {len(over)} ORI/offense row(s) place mass more than "
            f"{FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR:.0%} and more than "
            f"{FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR:g} offence away from their own "
            "ledger control; sample=" + str(over_sample)
        )
    non_placing = frame[~frame["footprint_status"].eq(CUSTOM_FOOTPRINT_STATUS_PLACED)]
    leaked = non_placing[
        pd.to_numeric(non_placing["placed_mass"], errors="coerce").fillna(0.0).abs().gt(1e-6)
    ]
    summary["non_placing_rows_with_mass"] = int(len(leaked))
    if len(leaked):
        issues.append(
            f"footprint_mass_conservation: {len(leaked)} suppressed or unplaceable footprint "
            "row(s) still placed mass on block groups; sample="
            + str(
                leaked.head(10)[["ori9", "state_fips", "offense", "footprint_status", "placed_mass"]]
                .to_dict(orient="records")
            )
        )
    # The per-offence tripwire is advisory on its own; only the per-ORI gate and the
    # non-placing-status rule can fail the release here.
    summary["ok"] = not (len(over_ori) or len(leaked))
    return summary


def _composite_component_specs() -> tuple[
    dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]
]:
    """Which offenses each composite asserts, split by the denominator family it is built on."""
    exposure = {
        RENAMED_COMPOSITE_FIELDS["index_total_primary_event_weighted"]: tuple(OFFENSES_7),
        RENAMED_COMPOSITE_FIELDS["index_total_equal_offense"]: tuple(OFFENSES_7),
        PERSONAL_RELATIVE_SCORE_COLUMN: tuple(PERSONAL_OFFENSES),
        PROPERTY_RELATIVE_SCORE_COLUMN: tuple(PROPERTY_OFFENSES),
    }
    resident = {
        EVENT_BURDEN_COLUMN: tuple(OFFENSES_7),
        PERSONAL_BURDEN_COLUMN: tuple(PERSONAL_OFFENSES),
        PROPERTY_BURDEN_COLUMN: tuple(PROPERTY_OFFENSES),
        HARM_BURDEN_COLUMN: tuple(OFFENSES_7),
    }
    return exposure, resident


def _check_composite_suppression_propagation(
    *, output_dir: Path, issues: list[str]
) -> dict[str, Any]:
    """No composite may publish while one of its own components is suppressed.

    The family is the whole point. An index-average over per-offense PRIMARY indexes asserts the
    opportunity-denominator points, so it is gated by `primary_index_suppressed_*`; a resident
    burden asserts the resident-denominator points, so it is gated by
    `index_*_resident_suppressed`. Murder and rape are read at the parent tract at block-group
    support, which is where those composites take their term from.
    """
    exposure_specs, resident_specs = _composite_component_specs()
    summary: dict[str, Any] = {"variants": {}}
    ok = True
    for variant in ("ags_core", "cde_exact_sensitivity"):
        bg_path = _published_surface_path(output_dir, geography="block_group", variant=variant)
        tr_path = _published_surface_path(output_dir, geography="tract", variant=variant)
        if not (bg_path.exists() and tr_path.exists()):
            continue
        composite_columns = sorted({*exposure_specs, *resident_specs})
        flag_columns = [f"primary_index_suppressed_{o}" for o in OFFENSES_7] + [
            f"index_{o}_resident_suppressed" for o in OFFENSES_7
        ]
        bg_available = _surface_column_names(bg_path)
        tr_available = _surface_column_names(tr_path)
        bg = pd.read_parquet(
            bg_path,
            columns=[
                c
                for c in ["block_group_geoid", *composite_columns, *flag_columns]
                if c in bg_available
            ],
        )
        tr = pd.read_parquet(
            tr_path,
            columns=[c for c in ["tract_id", *flag_columns] if c in tr_available],
        )
        tract_key = bg["block_group_geoid"].astype("string").str.slice(0, 11)
        tract_indexed = tr.set_index(tr["tract_id"].astype("string"))
        variant_summary: dict[str, Any] = {}

        def suppressed(offenses: tuple[str, ...], *, family: str) -> pd.Series:
            mask = pd.Series(False, index=bg.index)
            for offense in offenses:
                column = (
                    f"primary_index_suppressed_{offense}"
                    if family == "primary"
                    else f"index_{offense}_resident_suppressed"
                )
                if offense in RARE_OFFENSE_TRACT_SUPPORT:
                    if column not in tract_indexed.columns:
                        continue
                    parent = tract_key.map(tract_indexed[column])
                    mask = mask | parent.fillna(False).astype(bool)
                elif column in bg.columns:
                    mask = mask | bg[column].fillna(False).astype(bool)
            return mask

        for family, specs in (("primary", exposure_specs), ("resident", resident_specs)):
            for column, offenses in specs.items():
                if column not in bg.columns:
                    continue
                published = pd.to_numeric(bg[column], errors="coerce").notna()
                leak = int((published & suppressed(offenses, family=family)).sum())
                variant_summary[column] = {
                    "published_rows": int(published.sum()),
                    "suppressed_component_rows": leak,
                }
                if leak:
                    ok = False
                    issues.append(
                        f"composite_suppression_propagation[{variant}]: {column} publishes on "
                        f"{leak} block group(s) whose own {family}-family component is suppressed"
                    )
        summary["variants"][variant] = variant_summary
    summary["ok"] = ok
    return summary


def build_summary(
    *,
    state_output_dir: Path = STATE_OUTPUT_DIR,
    edition_dir: Path | None = None,
    holdout_run: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    # Read the contract first: every surface assertion below is selected by the lanes the build
    # declares, and a build that declares none is held to the legacy contract exactly as before.
    contract = _load_release_contract(state_output_dir)
    static_overwrite_summary = _check_no_exposure_tempered_calls(issues=issues)
    confidence_pure_enrichment_summary = _check_confidence_pure_enrichment(issues=issues)
    tract_ags_core_path = state_output_dir / f"crimerisk_tract_{YEAR}_ags_core.parquet"
    tract_cde_exact_sensitivity_path = _published_surface_path(
        state_output_dir, geography="tract", variant="cde_exact_sensitivity"
    )
    # Each block-group surface names the tract surface that carries its rare-offense (murder/rape)
    # index at tract support, used to recompute the block-group aggregates that consume it.
    surfaces = [
        (
            "block_group_ags_core",
            state_output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet",
            "block_group",
            tract_ags_core_path,
        ),
        (
            "tract_ags_core",
            tract_ags_core_path,
            "tract",
            None,
        ),
        (
            "block_group_cde_exact_sensitivity",
            _published_surface_path(state_output_dir, geography="block_group", variant="cde_exact_sensitivity"),
            "block_group",
            tract_cde_exact_sensitivity_path,
        ),
        (
            "tract_cde_exact_sensitivity",
            tract_cde_exact_sensitivity_path,
            "tract",
            None,
        ),
    ]
    surface_summaries = [
        _check_surface(
            label=label,
            path=path,
            geography=geography,
            issues=issues,
            rare_support_tract_path=rare_support_tract_path,
            contract=contract,
        )
        for label, path, geography, rare_support_tract_path in surfaces
    ]
    release_contract_summary = _check_release_contract(
        contract=contract, state_output_dir=state_output_dir, issues=issues
    )
    burglary_tau_calibration_summary = _load_burglary_tau_calibration(issues=issues)
    murder_tract_posterior_calibration_summary = _load_murder_tract_posterior_calibration(
        issues=issues,
        holdout_run=holdout_run,
    )
    build_manifest_summary = _check_build_manifest(
        output_dir=state_output_dir,
        issues=issues,
        burglary_tau_calibration=burglary_tau_calibration_summary,
        murder_tract_posterior_calibration=murder_tract_posterior_calibration_summary,
    )
    next_phase_measurement_summary = _check_next_phase_measurement(
        issues=issues,
        holdout_run=holdout_run,
    )
    release_evaluation_summary = _check_release_evaluation(issues=issues)
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
        contract=contract,
    )
    spatial_artifact_gates_summary = _check_spatial_artifact_gates(
        output_dir=state_output_dir,
        issues=issues,
    )
    rare_offense_tract_support_summary = _check_rare_offense_tract_support(
        output_dir=state_output_dir,
        issues=issues,
    )
    footprint_mass_conservation_summary = _check_footprint_mass_conservation(
        output_dir=state_output_dir,
        issues=issues,
    )
    composite_suppression_summary = _check_composite_suppression_propagation(
        output_dir=state_output_dir,
        issues=issues,
    )
    # Emitted only when an edition is under validation, so a summary for a bare surface directory
    # is the document it has always been, byte for byte.
    edition_rollup_summary = (
        _check_edition_rollups(edition_dir=Path(edition_dir), issues=issues)
        if edition_dir is not None
        else None
    )
    edition_coverage_scenario_summary = (
        _check_edition_coverage_scenarios(edition_dir=Path(edition_dir), issues=issues)
        if edition_dir is not None
        else None
    )

    present_labels = {s["label"] for s in surface_summaries if s.get("present")}
    for required_label in ("block_group_ags_core", "tract_ags_core"):
        if required_label not in present_labels:
            issues.append(f"missing required AGS-core surface: {required_label}")
    for required_label in ("block_group_cde_exact_sensitivity", "tract_cde_exact_sensitivity"):
        if required_label not in present_labels:
            issues.append(f"missing required CDE-exact diagnostic sensitivity surface: {required_label}")

    # Tract counts must roll up from block-group counts within tolerance, per product.
    for variant in ("ags_core", "cde_exact_sensitivity"):
        bg_path = _published_surface_path(state_output_dir, geography="block_group", variant=variant)
        tr_path = _published_surface_path(state_output_dir, geography="tract", variant=variant)
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
    allocation_controls, allocation_control_summary = _allocation_control_targets(
        output_dir=state_output_dir,
        issues=issues,
    )
    if component_audit_path.exists() and allocation_controls is not None:
        component_columns = [
            "state_fips", "jurisdiction_id", "jurisdiction_type", "offense",
            "component_count_before",
        ]
        if contract.service_wide_custom_allocation_enabled:
            component_columns.extend(["service_scope_id", "source_state_fips"])
        components = pd.read_parquet(
            component_audit_path,
            columns=component_columns,
        )
        controls = allocation_controls.copy()
        components["state_fips"] = components["state_fips"].astype("string").str.zfill(2)
        controls["state_fips"] = controls["state_fips"].astype("string").str.zfill(2)
        components = components[~components["state_fips"].isin(RELEASE_EXCLUDED_STATE_FIPS)].copy()
        controls = controls[~controls["state_fips"].isin(RELEASE_EXCLUDED_STATE_FIPS)].copy()
        overlap_component = components["jurisdiction_type"].astype("string").str.contains("overlap", na=False)
        overlap_control_state = components["state_fips"].copy()
        if contract.service_wide_custom_allocation_enabled:
            service_component = components["service_scope_id"].notna() & components[
                "service_scope_id"
            ].astype("string").str.strip().ne("")
            overlap_control_state = overlap_control_state.where(
                ~service_component,
                components["source_state_fips"].astype("string").str.zfill(2),
            )
        components.loc[overlap_component, "jurisdiction_id"] = (
            overlap_control_state.loc[overlap_component] + ":statewide_overlap_layer"
        )
        remainder_component = components["jurisdiction_type"].astype("string").isin(
            ["localized_remainder_county_layer", "localized_remainder_residual_layer"]
        )
        components.loc[remainder_component, "jurisdiction_id"] = (
            components.loc[remainder_component, "state_fips"] + ":state_nonmunicipal_remainder"
        )
        component_sums = (
            components.groupby(["jurisdiction_id", "offense"], dropna=False)["component_count_before"]
            .sum()
            .rename("component_total")
            .reset_index()
        )
        # Unlocated-mass lane credit (same identity shift as the total-lane
        # check: component_total + unlocated == control; absent table = lane
        # off = legacy identity unchanged).
        unlocated = _unlocated_mass_table(state_output_dir)
        if unlocated is not None:
            credit_rows = unlocated.copy()
            if contract.service_wide_custom_allocation_enabled:
                terminal = credit_rows.get(
                    "route_reason", pd.Series("", index=credit_rows.index)
                ).astype("string").eq("service_no_eligible_receiver")
                if bool(terminal.any()):
                    if "source_state_fips" not in credit_rows.columns:
                        issues.append(
                            "source_state_control_reconciliation: service terminal "
                            "unlocated rows are missing source_state_fips"
                        )
                    else:
                        source_state = credit_rows.loc[
                            terminal, "source_state_fips"
                        ].astype("string").fillna("").str.strip()
                        if bool(source_state.eq("").any()):
                            issues.append(
                                "source_state_control_reconciliation: service terminal "
                                "unlocated rows have blank source_state_fips"
                            )
                        else:
                            credit_rows.loc[terminal, "jurisdiction_id"] = (
                                source_state.str.zfill(2) + ":statewide_overlap_layer"
                            )
            credit = (
                credit_rows.groupby(["jurisdiction_id", "offense"], dropna=False)["unlocated_count"]
                .sum()
                .rename("unlocated_credit")
                .reset_index()
            )
            component_sums = component_sums.merge(
                credit, on=["jurisdiction_id", "offense"], how="left"
            )
            component_sums["component_total"] = component_sums[
                "component_total"
            ] + pd.to_numeric(
                component_sums.pop("unlocated_credit"), errors="coerce"
            ).fillna(0.0)
        target = controls[["jurisdiction_id", "offense", "control_target"]].copy()
        merged = target.merge(component_sums, on=["jurisdiction_id", "offense"], how="outer")
        diff = (
            pd.to_numeric(merged["component_total"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["control_target"], errors="coerce").fillna(0.0)
        )
        max_abs = _max_abs(diff)
        if max_abs > 1e-6:
            issues.append(
                "ags_core: allocation component expected counts do not reconcile to declared controls "
                f"(max abs diff {max_abs:.3e})"
            )
    else:
        issues.append(
            f"missing allocation component audit or controls for BG->jurisdiction reconciliation: "
            f"{component_audit_path}, {allocation_control_summary.get('path')}"
        )

    qa_summary = _load_json(PACKAGE_VALIDATION_DIR / "build_qa_summary.json") or _load_json(REPO_QA_SUMMARY)
    # Repository/package QA summaries describe the promoted state/output tree.
    # Candidate validation is bound to the candidate manifest and surfaces above;
    # inheriting promoted-tree freshness flags would make a valid diagnostic split
    # fail solely because an older release did not contain the renamed artifacts.
    validating_promoted_tree = state_output_dir.resolve() == STATE_OUTPUT_DIR.resolve()
    if qa_summary is not None and validating_promoted_tree:
        qa_outputs = qa_summary.get("outputs", {})
        if qa_outputs.get("cde_exact_sensitivity_present") is False:
            issues.append("validation summary says CDE-exact diagnostic sensitivity outputs are absent")
        if qa_outputs.get("cde_exact_sensitivity_current") is False:
            issues.append("validation summary says CDE-exact diagnostic sensitivity outputs are stale")

    return {
        "ok": not issues,
        "state_output_dir": str(state_output_dir),
        "static_no_exposure_tempered_calls": static_overwrite_summary,
        "static_confidence_pure_enrichment": confidence_pure_enrichment_summary,
        "surface_count": len(surface_summaries),
        "surfaces": surface_summaries,
        # Reported only when the build declares a v2 lane, so a legacy release's summary is the
        # document it has always been, byte for byte.
        **({"release_contract": release_contract_summary} if contract.lanes else {}),
        "build_manifest": build_manifest_summary,
        "burglary_tau_calibration": burglary_tau_calibration_summary,
        "murder_tract_posterior_calibration": murder_tract_posterior_calibration_summary,
        "next_phase_measurement": next_phase_measurement_summary,
        "release_evaluation": release_evaluation_summary,
        "dashboard_lookup": dashboard_lookup_summary,
        "external_surface_availability": external_surface_availability_summary,
        "connecticut_population": connecticut_population_summary,
        "acs_bg_vocabulary_coverage": acs_bg_vocabulary_summary,
        "sparse_residual_transfer_policy": sparse_transfer_policy_summary,
        "total_lane_qa": total_lane_qa_summary,
        "allocation_control_reconciliation": allocation_control_summary,
        "spatial_artifact_gates": spatial_artifact_gates_summary,
        "rare_offense_tract_support": rare_offense_tract_support_summary,
        "footprint_mass_conservation": footprint_mass_conservation_summary,
        "composite_suppression_propagation": composite_suppression_summary,
        **({"edition_rollups": edition_rollup_summary} if edition_rollup_summary is not None else {}),
        **(
            {"edition_coverage_scenarios": edition_coverage_scenario_summary}
            if edition_coverage_scenario_summary is not None
            else {}
        ),
        "validation_summary_present": qa_summary is not None,
        "issues": issues,
    }, issues



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument(
        "--holdout-run",
        action="store_true",
        help=(
            "Acknowledge that candidate evaluation intentionally has fewer city controls; "
            "release-run support floors remain enforced without this flag."
        ),
    )
    parser.add_argument(
        "--edition-dir",
        type=Path,
        default=None,
        help=(
            "Packaged edition directory (state/editions/<edition-id>). When passed, the gate adds "
            "the rollup section: conservation re-derived from the published tables, rates "
            "recomputed from counts at every support, and the field dictionary checked current "
            "against the schemas."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Target build year under validation (drives output filenames, the manifest-year assertion, and target-year frame filters).",
    )
    args = parser.parse_args()

    _apply_target_year(int(args.year))
    output_dir = args.output_dir or STATE_OUTPUT_DIR
    summary, issues = build_summary(
        state_output_dir=output_dir,
        edition_dir=args.edition_dir,
        holdout_run=bool(args.holdout_run),
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
