from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

from crimerisk.benchmark_imputation import (
    county_remainder_imputed_targets,
    load_benchmark_imputation_units,
)
from crimerisk.confidence import build_confidence_artifacts, enrich_confidence_surfaces
from crimerisk.crime import OFFENSES_7
from crimerisk.city_residuals import CityResidualConfig, apply_city_residual_model, attach_city_residual_features, fit_city_residual_model_from_truth
from crimerisk.city_shares import CityIncidentShareBuildConfig, write_v2_city_incident_shares
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
from crimerisk.paths import RepoPaths
from crimerisk.reference import COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES
from crimerisk.controls import ControlBuildConfig, controls_artifacts_are_current, write_v2_controls
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
    add_preferred_support_flags,
    build_agency_allocation_target_estimates,
)
from crimerisk.geometry import (
    GeometryBuildConfig,
    geometry_artifacts_are_current,
    write_v2_geometry,
)


PERSONAL_OFFENSES = ["murder", "rape", "robbery", "aggravated_assault"]
PROPERTY_OFFENSES = ["burglary", "larceny", "motor_vehicle_theft"]
SPARSE_BASELINE_TRANSFER_OFFENSES = frozenset(("murder", "rape"))
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
PERSON_EXPOSURE_FLOOR_OFFENSES = frozenset(("murder", "rape", "robbery", "aggravated_assault", "larceny"))
MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR = 50.0
BURGLARY_PREMISES_DENOMINATOR_FLOOR = 10.0
SPECIAL_USE_TRACT_PREFIX = "98"
SPECIAL_USE_EXPOSURE_DENOMINATOR_FLOOR = 10.0
SQ_METERS_PER_SQ_MILE = 2_589_988.11
TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO = 5.0
TRANSIENT_EXPOSURE_INDEX_THRESHOLD = 1000.0

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
AGGREGATE_INDEX_FIELDS = (
    "index_total_part1_resident",
    "index_personal_part1_resident",
    "index_property_part1_resident",
    "index_total_primary_event_weighted",
    "index_total_equal_offense",
    "index_total_harm",
)
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
    force_controls_rebuild: bool = False
    force_reporting_regimes_rebuild: bool = False
    force_geometry_rebuild: bool = False
    force_bg_prior_rebuild: bool = False
    force_city_incident_share_rebuild: bool = False
    force_city_incident_source_refresh: bool = False
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
    eb_alpha_by_offense: tuple[tuple[str, float], ...] = DEFAULT_EB_ALPHA_BY_OFFENSE
    eb_hard_min_denominator: float = EB_HARD_MIN_DENOMINATOR
    burglary_commercial_weight: float | None = None
    city_posterior_reconciliation_tolerance: float = CITY_POSTERIOR_RECONCILIATION_TOLERANCE
    city_posterior_alpha_floor: float = CITY_POSTERIOR_ALPHA_FLOOR
    city_posterior_alpha_volume_incidents: float = CITY_POSTERIOR_ALPHA_VOLUME_INCIDENTS
    city_posterior_alpha_max_prior_fraction: float = CITY_POSTERIOR_ALPHA_MAX_PRIOR_FRACTION
    enable_county_anchoring: bool = True


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

    return replace(
        config,
        residual_training_city_shares_path=residual_training_city_shares_path,
        residual_training_exclude_validation_case_types=residual_training_exclude_validation_case_types,
        residual_training_extra_bg_feature_paths=residual_training_extra_bg_feature_paths,
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
    return resolved if resolved.is_absolute() else paths.repo_root / resolved


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
        and policy_name == DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH.name
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
    bg_prior_path = _bg_prior_cache_path(
        paths=paths,
        config=config,
        model_surface_config=model_surface_config,
    )
    manifest = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "year": int(config.year),
        "summary": summary,
        "resolved_config": {
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
            "city_posterior_reconciliation_tolerance": float(config.city_posterior_reconciliation_tolerance),
            "city_posterior_alpha_floor": float(config.city_posterior_alpha_floor),
            "city_posterior_alpha_volume_incidents": float(config.city_posterior_alpha_volume_incidents),
            "city_posterior_alpha_max_prior_fraction": float(config.city_posterior_alpha_max_prior_fraction),
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
            "city_incident_share_surface": _path_stats(
                paths.state_dir / "modeling" / "city_incident_share_surface.parquet"
            ),
            "city_incident_reconciliation": _path_stats(
                paths.state_dir / "modeling" / f"city_incident_reconciliation_{int(config.year)}.parquet"
            ),
            "bg_prior_long": _path_stats(bg_prior_path),
            "model_surface_feature_policy": _path_stats(model_surface_config.feature_policy_path),
            "residual_feature_policy": _path_stats(
                _resolve_repo_path(paths, config.residual_feature_policy_path)
            ),
            "burglary_tau_calibration": _path_stats(
                paths.state_dir / "modeling" / f"burglary_tau_calibration_{int(config.year)}.json"
            ),
            "jurisdiction_controls": _path_stats(
                paths.state_dir / "controls" / f"jurisdiction_controls_{int(config.year)}.parquet"
            ),
            "state_control_comparison": _path_stats(paths.state_dir / "controls" / "state_control_comparison.parquet"),
            "block_group_crosswalk": _path_stats(
                paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
            ),
            "consolidated_agency_footprints": _path_stats(
                paths.repo_root / "configs" / "consolidated_agency_footprints.csv"
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
) -> pd.DataFrame:
    bg = add_offense_denominators(
        build_bg_feature_frame(paths=paths, year=year),
        paths=paths,
        year=year,
        burglary_commercial_weight=burglary_commercial_weight,
    )
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
        ]
    ].copy()
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
    return bg[
        ["state_fips", "block_group_geoid", "jurisdiction_id", "jurisdiction_type", "allocation_share"]
    ].copy()


def _load_controls(paths: RepoPaths, *, year: int) -> pd.DataFrame:
    path = paths.state_dir / "controls" / f"jurisdiction_controls_{int(year)}.parquet"
    return pd.read_parquet(path)


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


def _build_agency_allocation_target_estimates(
    *,
    paths: RepoPaths,
    year: int,
) -> pd.DataFrame:
    """Thin wrapper: per-agency target-year estimates are computed once in
    trend_fills.py (build_agency_allocation_target_estimates) and shared by both the
    county-remainder split below and the jurisdiction-target consumption layer in
    jurisdiction_targets.py, so both consumers see identical per-agency amounts.
    """
    return build_agency_allocation_target_estimates(paths=paths, year=int(year))


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


def _state_police_county_subunit_mask(merged: pd.DataFrame, name_norm: pd.Series, has_county: pd.Series) -> pd.Series:
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
    generic_or_hq = name_norm.str.contains(
        r"\b(?:HEADQUART|HQ|ACADEMY|TRAINING|LABORATORY|FORENSIC|COMMUNICATION|ADMIN|GENERAL|CAPITOL|GAMING|SPECIAL|EXECUTIVE|INVESTIGATION|BUREAU)\b",
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
    agency_estimates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    control_cols = ["state_fips", "offense", "adjusted_count_ags_core"]
    for optional_col in ("reported_count_preferred", "adjustment_total"):
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
    if "reported_count_preferred" in remainder_controls.columns:
        remainder_controls["state_reported_target"] = pd.to_numeric(
            remainder_controls["reported_count_preferred"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
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


def _load_city_incident_share_surface(
    paths: RepoPaths,
    *,
    year: int,
    path: Path | None = None,
    exclude_validation_case_types: tuple[str, ...] = (),
) -> pd.DataFrame:
    resolved_path = path or paths.state_dir / "modeling" / "city_incident_share_surface.parquet"
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
    pooled = (
        year_counts.groupby(detail_cols, dropna=False, as_index=False)
        .agg(
            incident_count=("incident_count", "sum"),
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
    pooled["city_share_pooling_method"] = "incident_count_sum_all_reliable_years_unweighted"

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

    incident = _load_city_incident_share_surface(paths, year=year)
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
)

# What `weight_share` MEASURES, declared per footprint row rather than inferred from the
# geometry-source string. Stage 2 fork ruling 3: the custom-footprint lane allocated by
# `weight_share` alone, with no activity or exposure term, while every county lane multiplies
# its share by `bg_weight`. The distinction that matters is what the share already contains:
#
#   resident_population -- the share IS a 2020 resident-population share of the footprint
#       (tribal AIANNH block-assignment footprints, state-police post footprints). Spreading
#       an agency total over it by population alone is the size-blind form the county lanes
#       do not use, so these get the same `bg_weight x share` activity basis.
#   activity_or_area -- the share is ALREADY an activity or exposure measure (station
#       boardings, annual passengers, LandScan daytime population, per-station equal share)
#       or a deliberate area apportionment onto parcels with no residents (airfields, port
#       property). Multiplying these by `bg_weight` would apply an activity term twice.
CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT = "resident_population"
CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_ACTIVITY = "activity_or_area"
VALID_CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASES: frozenset[str] = frozenset(
    {CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_ACTIVITY}
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
    out_of_range = footprints["bg_population_coverage_share"].notna() & (
        footprints["bg_population_coverage_share"].le(0.0)
        | footprints["bg_population_coverage_share"].gt(1.0 + 1e-9)
    )
    if bool(out_of_range.any()):
        raise ValueError(
            "bg_population_coverage_share must lie in (0, 1]: "
            f"{footprints.loc[out_of_range, ['ori9', 'bg_id', 'bg_population_coverage_share']].to_dict(orient='records')}"
        )
    totals = footprints.groupby(["ori9", "state_fips"], dropna=False)["weight_share"].sum().reset_index()
    bad = totals[~np.isclose(totals["weight_share"], 1.0, atol=1e-6)]
    if not bad.empty:
        raise ValueError(
            "Overlap custom footprints must sum to 1 by ori/state; bad rows: "
            f"{bad[['ori9', 'state_fips', 'weight_share']].to_dict(orient='records')}"
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
    mixed = (
        footprints.groupby(["ori9", "state_fips"], dropna=False)["weight_share_basis"].nunique().rename("bases")
    )
    mixed = mixed[mixed.gt(1)]
    if not mixed.empty:
        raise ValueError(
            "Overlap custom footprints mix weight_share_basis values inside one (ori, state) "
            f"pool: {mixed.reset_index().to_dict(orient='records')}"
        )
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


def _load_consolidated_agency_footprints(paths: RepoPaths) -> pd.DataFrame:
    path = paths.repo_root / "configs" / "consolidated_agency_footprints.csv"
    columns = [
        "ori9",
        "state_fips",
        "county_fips",
        "principal_jurisdiction_id",
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
        support = (
            support.groupby(group_cols, dropna=False, as_index=False)
            .agg(
                bg_weight=("bg_weight", "first"),
                allocation_share=("allocation_share", "sum"),
            )
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


def _build_jurisdiction_component_allocations(
    *,
    paths: RepoPaths,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    controls: pd.DataFrame,
    year: int,
    residual_training_city_shares_path: Path | None = None,
    residual_training_exclude_validation_case_types: tuple[str, ...] = (),
    residual_training_extra_bg_feature_paths: tuple[Path, ...] = (),
    residual_feature_policy_path: Path | None = DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
    residual_exclude_feature_policy_classes: tuple[str, ...] = DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
    residual_exclude_feature_policy_classes_by_offense: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    residual_transfer_tau_by_offense: tuple[tuple[str, float], ...] = DEFAULT_RESIDUAL_TRANSFER_TAU_BY_OFFENSE,
    city_posterior_reconciliation_tolerance: float = CITY_POSTERIOR_RECONCILIATION_TOLERANCE,
    city_posterior_alpha_floor: float = CITY_POSTERIOR_ALPHA_FLOOR,
    city_posterior_alpha_volume_incidents: float = CITY_POSTERIOR_ALPHA_VOLUME_INCIDENTS,
    city_posterior_alpha_max_prior_fraction: float = CITY_POSTERIOR_ALPHA_MAX_PRIOR_FRACTION,
    enable_county_anchoring: bool = True,
    agency_estimates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    residual_transfer_tau = _residual_transfer_tau_dict(residual_transfer_tau_by_offense)
    consolidated_footprints = _load_consolidated_agency_footprints(paths) if bool(enable_county_anchoring) else pd.DataFrame()
    # EXCLUSIVE footprints displace the county remainder before anything reads the support,
    # so the remainder targets, the county/residual split and the allocation all see the same
    # ground. See `_apply_exclusive_footprint_displacement`.
    exclusive_displacement = _build_exclusive_footprint_displacement(
        overrides=_load_overlap_footprint_overrides(paths),
        custom_footprints=_load_overlap_custom_footprints(paths),
        concurrent_jurisdiction_carveouts=_load_concurrent_jurisdiction_carveouts(paths),
    )
    if not exclusive_displacement.empty:
        bg_crosswalk = _apply_exclusive_footprint_displacement(bg_crosswalk, exclusive_displacement)
    county_remainder_targets = (
        _build_county_remainder_group_targets(
            paths=paths,
            controls=controls,
            year=year,
            bg_crosswalk=bg_crosswalk,
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
    incident_surface = _load_city_incident_share_surface(
        paths,
        year=year,
    )
    residual_training_surface = (
        _load_city_incident_share_surface(
            paths,
            year=year,
            path=residual_training_city_shares_path,
            exclude_validation_case_types=residual_training_exclude_validation_case_types,
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
        if fitted is not None:
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
        "calibrated_residual_tau"
        + applied_transfer_tau.loc[calibrated_residual_mask].map(lambda value: f"{float(value):.2f}")
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
        target["target_count"] = pd.to_numeric(target["adjusted_count_ags_core"], errors="coerce").fillna(0.0).clip(lower=0.0)
        target = target.drop(columns="adjusted_count_ags_core")
    merged = merged.merge(target, on=["jurisdiction_id", "offense"], how="inner")

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
    posterior_share = np.where(
        active & posterior_total.gt(0.0),
        posterior_numer / posterior_total.replace(0.0, np.nan),
        merged["city_posterior_model_prior_share"],
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
    merged["city_posterior_share"] = pd.Series(posterior_share, index=merged.index, dtype=float).fillna(0.0).clip(lower=0.0)
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
            "component_share",
        ]
    ].copy()
    out.attrs["city_posterior_diagnostics"] = posterior_diag
    out.attrs["city_posterior_summary"] = _summarize_city_posterior_diagnostics(posterior_diag)
    out.attrs["city_residual_feature_policy"] = residual_feature_policy_application or {}
    return out


def _build_overlap_group_targets(
    *,
    paths: RepoPaths,
    controls: pd.DataFrame,
    year: int,
    enable_county_anchoring: bool = True,
    bg_prior: pd.DataFrame | None = None,
    agency_estimates: pd.DataFrame | None = None,
    bg_crosswalk: pd.DataFrame | None = None,
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

    preferred = build_agency_preferred_observations(
        paths=paths,
        year=year,
    )[["ori9", "state_fips", "offense", "preferred_count"]].copy()
    preferred["state_fips"] = preferred["state_fips"].astype("string").str.zfill(2)
    preferred["preferred_count"] = pd.to_numeric(preferred["preferred_count"], errors="coerce").fillna(0.0)

    crosswalk = _load_crosswalk(paths)
    overlap_cw = crosswalk[crosswalk["relationship_type"].eq("overlap")].copy()
    if overlap_cw.empty:
        return pd.DataFrame(columns=["state_fips", "offense", "group_kind", "group_id", "target_count"])

    agency_master_full = _load_agency_master(paths)
    overlap_master_cols = [
        "ori9",
        "state_fips",
        "county_fips",
        "place_fips",
        "agency_name_std",
        "agency_type_norm",
    ]
    if "county_fips_source" in agency_master_full.columns:
        overlap_master_cols.append("county_fips_source")
    agency_master = agency_master_full[overlap_master_cols].copy()
    agency_master["state_fips"] = agency_master["state_fips"].astype("string").str.zfill(2)
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

    merged = preferred.merge(overlap_cw, left_on=["ori9", "state_fips"], right_on=["ori", "state_fips"], how="inner")
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
    has_county = _valid_county_fips(merged["county_fips"]) & merged[
        "county_fips_source"
    ].astype("string").isin(COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES).fillna(False)
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
        state_police_county_subunit = _state_police_county_subunit_mask(merged, name_norm, has_county)
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
    force_statewide = treatment.isin(["keep_statewide_overlap", "exclude_or_hold"])

    def _bool_mask(series: pd.Series) -> np.ndarray:
        return series.fillna(False).astype(bool).to_numpy()

    merged["group_kind"] = np.select(
        [
            _bool_mask(has_custom_footprint),
            _bool_mask(has_absorb_target),
            _bool_mask(has_override_place),
            _bool_mask(has_override_county),
            _bool_mask(force_statewide),
            # State police first, so a "SP <county> COUNTY" agency (which fires BOTH masks)
            # gets the non-municipal spread rather than the whole-county one.
            _bool_mask(state_police_county_subunit),
            _bool_mask(county_name_agreement),
            _bool_mask(localizable & has_place),
            _bool_mask(localizable & has_county),
        ],
        [
            "custom_footprint_overlap",
            "absorbed_overlap",
            "municipal_place_overlap",
            "county_overlap",
            "statewide_overlap",
            COUNTY_NONMUNICIPAL_OVERLAP_KIND,
            "county_overlap",
            "municipal_place_overlap",
            "county_overlap",
        ],
        default="statewide_overlap",
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

    agency_estimates = (
        agency_estimates.copy()
        if agency_estimates is not None
        else _build_agency_allocation_target_estimates(paths=paths, year=int(year))
    )
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
                    group_kind="statewide_overlap",
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
                    group_kind="statewide_overlap",
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

    grouped = observed_group.merge(
        adjustment_group,
        on=["state_fips", "offense", "group_kind", "group_id"],
        how="outer",
        suffixes=("_observed", "_adjustment"),
    )
    for col in ["observed_raw_count", "observed_target_count", "adjustment_raw_count", "adjustment_target_count"]:
        grouped[col] = pd.to_numeric(grouped.get(col), errors="coerce").fillna(0.0)
    grouped["county_anchor_evidence_count"] = np.maximum(
        pd.to_numeric(grouped.get("county_anchor_evidence_count_observed"), errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(grouped.get("county_anchor_evidence_count_adjustment"), errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )
    statewide_rows = overlap_controls[["state_fips", "offense"]].copy()
    statewide_rows["group_kind"] = "statewide_overlap"
    statewide_rows["group_id"] = statewide_rows["state_fips"]
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
        grouped["group_kind"].eq("statewide_overlap")
        | pd.to_numeric(grouped["target_count"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    sums = grouped.groupby(["state_fips", "offense"], dropna=False)["target_count"].sum().reset_index(name="target_sum")
    deltas = overlap_controls[["state_fips", "offense", "state_target"]].merge(sums, on=["state_fips", "offense"], how="left")
    deltas["delta"] = deltas["state_target"] - pd.to_numeric(deltas["target_sum"], errors="coerce").fillna(0.0)
    if deltas["delta"].abs().gt(1e-8).any():
        for row in deltas[deltas["delta"].abs().gt(1e-8)].itertuples(index=False):
            mask = (
                grouped["state_fips"].eq(row.state_fips)
                & grouped["offense"].eq(row.offense)
                & grouped["group_kind"].eq("statewide_overlap")
            )
            grouped.loc[mask, "target_count"] = (
                pd.to_numeric(grouped.loc[mask, "target_count"], errors="coerce").fillna(0.0) + float(row.delta)
            )
    final_sums = grouped.groupby(["state_fips", "offense"], dropna=False)["target_count"].sum().reset_index(name="target_sum")
    check = overlap_controls[["state_fips", "offense", "state_target"]].merge(final_sums, on=["state_fips", "offense"], how="left")
    max_delta = float((check["target_sum"].fillna(0.0) - check["state_target"]).abs().max() or 0.0)
    if max_delta > 1e-6:
        raise ValueError(f"Overlap targets do not partition controls; max abs delta={max_delta:.3e}")
    grouped["target_count"] = pd.to_numeric(grouped["target_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return grouped[["state_fips", "offense", "group_kind", "group_id", "target_count"]]


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
    1 inside every (state, ori, offense) pool that carries any weight.
    """
    out = merged.copy()
    weight_share = pd.to_numeric(out["weight_share"], errors="coerce").fillna(0.0)
    resident_basis = out["weight_share_basis"].astype("string").eq(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT
    )
    out["footprint_activity_weight"] = (
        pd.to_numeric(out["bg_weight"], errors="coerce").fillna(0.0) * weight_share
    ).where(resident_basis, 0.0)
    pool_cols = ["state_fips", "ori9", "offense"]
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
    weight_total = pd.to_numeric(out["footprint_weight_total"], errors="coerce").fillna(0.0)
    verbatim_share = np.where(
        weight_total > 0,
        pd.to_numeric(out["weight_share"], errors="coerce").fillna(0.0)
        / np.where(weight_total > 0, weight_total, 1.0),
        0.0,
    )
    # Fail-SAFE, not fail-open: a resident-basis footprint whose whole support carries zero
    # activity weight (an offense whose denominator is absent everywhere inside the footprint)
    # keeps the declared population apportionment rather than stranding the agency's mass on an
    # empty support set.
    out["component_share"] = np.where(
        resident_basis.to_numpy() & (activity_total.to_numpy() > 0),
        pd.to_numeric(out["footprint_activity_weight"], errors="coerce").fillna(0.0)
        / np.where(activity_total > 0, activity_total, 1.0),
        verbatim_share,
    )
    return out


def _build_overlap_allocations(
    *,
    paths: RepoPaths,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    controls: pd.DataFrame,
    year: int,
    enable_county_anchoring: bool = True,
    agency_estimates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    targets = _build_overlap_group_targets(
        paths=paths,
        controls=controls,
        year=year,
        enable_county_anchoring=bool(enable_county_anchoring),
        bg_prior=bg_prior,
        agency_estimates=agency_estimates,
        bg_crosswalk=bg_crosswalk,
    )
    if targets.empty:
        return pd.DataFrame(columns=["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"])

    out_frames: list[pd.DataFrame] = []

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
            # Stage 2 fork ruling 3. A footprint whose `weight_share` is a resident-population
            # share gets the same activity basis every county lane uses (`bg_weight x share`,
            # normalised inside the pool); a footprint whose share already measures activity
            # (boardings, throughput, LandScan daytime) or is a deliberate area apportionment
            # onto unpopulated parcels is used verbatim, because multiplying it by `bg_weight`
            # would apply an activity term twice. `weight_share_basis` is declared per row and one
            # basis per (ori, state) is enforced in the loader.
            merged = (
                bg_prior[["state_fips", "bg_id", "tract_id", "offense", "bg_weight"]]
                .merge(custom_footprints, on=["state_fips", "bg_id"], how="inner")
                .merge(custom_targets.rename(columns={"group_id": "ori9"})[["state_fips", "ori9", "offense", "target_count"]], on=["state_fips", "ori9", "offense"], how="inner")
            )
            merged = _custom_footprint_component_shares(merged)
            merged["component_count"] = (
                pd.to_numeric(merged["target_count"], errors="coerce").fillna(0.0)
                * merged["component_share"]
            )
            merged["jurisdiction_id"] = merged["ori9"]
            merged["jurisdiction_type"] = "custom_footprint_overlap_layer"
            out_frames.append(merged[["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"]].copy())

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

    if not out_frames:
        return pd.DataFrame(columns=["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"])
    return pd.concat(out_frames, ignore_index=True)


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


def _resident_part1_index(out: pd.DataFrame, offenses: list[str]) -> tuple[pd.Series, float]:
    denom = _raw_denominator(out["resident_secondary_denominator"])
    counts = sum(
        pd.to_numeric(out[_expected_count_col(offense)], errors="coerce").fillna(0.0).clip(lower=0.0)
        for offense in offenses
    )
    component_indexes = [
        pd.to_numeric(out[f"index_{offense}_resident"], errors="coerce")
        for offense in offenses
    ]
    publishable = denom.gt(0.0)
    for component_index in component_indexes:
        publishable &= component_index.notna()
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


def _aggregate_index_normalizers(surface: pd.DataFrame) -> dict[str, object]:
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
    """
    out = bg.copy()
    group = out[tract_geo_id_col].astype("string").str.zfill(11)
    share = _within_tract_person_exposure_share(out, group)

    redistributed_counts: dict[str, pd.Series] = {}
    for offense in RARE_OFFENSE_TRACT_SUPPORT:
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
    out["index_total_equal_offense"] = _full_component_index_composite(
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
    out["index_total_primary_event_weighted"] = _full_component_index_composite(
        out,
        list(OFFENSES_7),
        index_suffix="primary",
        weights=_national_expected_count_weights(out, list(OFFENSES_7)),
        index_overrides=index_overrides,
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
            ]
        ],
        on="bg_id",
        how="left",
    )
    return _attach_primary_denominator_for_audit(components)


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
        "bg_id",
        "tract_id",
        "jurisdiction_id",
        "jurisdiction_type",
        "offense",
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
        "component_share",
        "primary_denominator_type",
        "primary_denominator_raw",
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

    components = _components_with_primary_denominator(all_components, bg_covariates)
    group_cols = ["state_fips", "jurisdiction_id", "jurisdiction_type", "offense"]
    before = pd.to_numeric(components["component_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    denominator = pd.to_numeric(components["primary_denominator_raw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    zero_source = before.gt(0.0) & denominator.le(0.0)
    eligible = denominator.gt(0.0)

    components["_zero_source_mass"] = np.where(zero_source, before, 0.0)
    components["_eligible_count_weight"] = np.where(eligible, before, 0.0)
    components["_eligible_denominator_weight"] = np.where(eligible, denominator, 0.0)
    zero_mass = components.groupby(group_cols, dropna=False)["_zero_source_mass"].transform("sum")
    eligible_count_weight_sum = components.groupby(group_cols, dropna=False)["_eligible_count_weight"].transform("sum")
    eligible_denominator_weight_sum = components.groupby(group_cols, dropna=False)["_eligible_denominator_weight"].transform("sum")
    use_count_weight = eligible_count_weight_sum.gt(0.0)
    use_denominator_weight = ~use_count_weight & eligible_denominator_weight_sum.gt(0.0)
    recipient_weight = np.where(
        eligible & use_count_weight,
        components["_eligible_count_weight"],
        np.where(eligible & use_denominator_weight, components["_eligible_denominator_weight"], 0.0),
    )
    recipient_weight_total = np.where(use_count_weight, eligible_count_weight_sum, eligible_denominator_weight_sum)
    group_can_redistribute = zero_mass.gt(0.0) & pd.Series(recipient_weight_total, index=components.index).gt(0.0)
    redistributed_delta = np.where(
        eligible & group_can_redistribute,
        zero_mass * recipient_weight / recipient_weight_total,
        0.0,
    )
    if not APPLY_ZERO_TARGET_REDISTRIBUTION:
        redistributed_delta = np.zeros(len(components), dtype=float)
        group_can_redistribute = pd.Series(False, index=components.index)
    after = before + redistributed_delta
    after = pd.Series(after, index=components.index, dtype=float)
    after.loc[zero_source & group_can_redistribute] = 0.0
    components["component_count_before"] = before
    components["component_count_after"] = after
    components["redistributed_delta"] = after - before
    components["component_count"] = after
    components["_recipient_delta_mass"] = np.where(components["redistributed_delta"].gt(0.0), components["redistributed_delta"], 0.0)
    components["_recipient_count"] = np.where(components["redistributed_delta"].gt(1e-12), 1, 0)
    components["group_zero_source_mass"] = zero_mass
    components["group_recipient_count"] = components.groupby(group_cols, dropna=False)["_recipient_count"].transform("sum")
    components["group_recipient_delta_mass"] = components.groupby(group_cols, dropna=False)["_recipient_delta_mass"].transform("sum")
    components["group_total_before"] = components.groupby(group_cols, dropna=False)["component_count_before"].transform("sum")
    components["group_total_after"] = components.groupby(group_cols, dropna=False)["component_count_after"].transform("sum")
    components["group_total_delta"] = components["group_total_after"] - components["group_total_before"]
    components["audit_reason"] = np.where(
        zero_source,
        "positive_allocation_zero_offense_relevant_denominator",
        "zero_target_redistribution_recipient",
    )
    components["redistribution_status"] = np.select(
        [
            zero_source & bool(APPLY_ZERO_TARGET_REDISTRIBUTION) & group_can_redistribute,
            zero_source & (not bool(APPLY_ZERO_TARGET_REDISTRIBUTION)),
            zero_source & ~group_can_redistribute,
            components["redistributed_delta"].gt(0.0),
        ],
        [
            "source_redistributed",
            "source_not_redistributed_tvd_guardrail",
            "source_not_redistributed_no_eligible_target",
            "recipient",
        ],
        default="unchanged",
    )
    component_audit = components[[col for col in columns if col in components.columns]].copy()
    audit = components[zero_source].copy()
    out_components = components[
        ["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"]
    ].copy()
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


def _finalize_output(
    frame: pd.DataFrame,
    *,
    geo_id_col: str,
    population_col: str,
    config: AllocationBuildConfig,
    jurisdiction_col: str = "eb_jurisdiction_id",
) -> pd.DataFrame:
    out = frame.copy()
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
    exposure_proxy = exposure_proxy.combine_first(fallback_exposure_proxy).fillna(0.0).clip(lower=0.0)
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
        raw_denominator = pd.to_numeric(
            out[DENOMINATOR_SOURCE_COLUMNS[denominator_type]],
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        out[f"primary_denominator_type_{offense}"] = denominator_type
        out[f"primary_denominator_{offense}"] = raw_denominator
        primary_denominator_invalid = (
            mvt_vehicle_denominator_invalid
            if offense == "motor_vehicle_theft"
            else pd.Series(False, index=out.index)
        )
        if offense == "burglary":
            special_use_suppressed = special_use_tract | raw_denominator.lt(float(BURGLARY_PREMISES_DENOMINATOR_FLOOR))
        else:
            special_use_suppressed = special_use_tract
        special_use_suppressed = pd.Series(special_use_suppressed, index=out.index).fillna(False).astype(bool)
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES:
            primary_insufficient_exposure = raw_denominator.lt(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
        elif offense == "motor_vehicle_theft":
            primary_insufficient_exposure = raw_denominator.lt(float(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR))
        else:
            primary_insufficient_exposure = pd.Series(False, index=out.index)
        primary_insufficient_exposure = (
            pd.Series(primary_insufficient_exposure, index=out.index).fillna(False).astype(bool)
        )

        # The resident arm's own floor, hoisted above the primary block: the ambient-blind
        # footprint rule below is defined against the RESIDENT rate, so its baseline
        # publishable mask has to exist before either arm is finalised.
        resident_denominator = out["resident_secondary_denominator"]
        if offense in PERSON_EXPOSURE_FLOOR_OFFENSES:
            resident_insufficient_exposure = resident_denominator.lt(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
        elif offense == "motor_vehicle_theft":
            resident_insufficient_exposure = raw_denominator.lt(float(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR))
        else:
            resident_insufficient_exposure = pd.Series(False, index=out.index)
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
            residential_eligible
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
        footprint_ambient_exposure_missing = (
            footprint_share.gt(float(FOOTPRINT_DERIVED_MASS_SHARE_FLOOR))
            & ~ambient_exposure_lift
            & resident_rate_ratio.ge(float(AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO))
        ).fillna(False)
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
            residential_eligible
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
        primary_insufficient_reason_mask = (
            primary_insufficient_exposure
            & ~non_residential
            & ~primary_denominator_invalid
            & ~special_use_suppressed
        )
        primary_denominator_reason.loc[primary_insufficient_reason_mask] = "insufficient_exposure"
        # A cell that is BOTH below the plain exposure floor and ambient-blind reports the
        # floor: the floor is the harder structural fact and does not need the footprint.
        footprint_reason_mask = (
            footprint_ambient_exposure_missing
            & ~non_residential
            & ~primary_denominator_invalid
            & ~special_use_suppressed
            & ~primary_insufficient_exposure
        )
        primary_denominator_reason.loc[footprint_reason_mask] = INSUFFICIENT_AMBIENT_EXPOSURE_REASON
        # `denominator_reason` had no `non_residential` value at all (Stage 5 F7): 2,306 BG rows
        # per offense carried "publishable" next to a null index, because the households floor
        # nulls the index without touching the reason. Assigned LAST so the precedence matches
        # `_estimate_mode`, where `non_residential` also wins.
        primary_denominator_reason.loc[non_residential] = "non_residential"
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
            non_residential
            | primary_denominator_invalid
            | special_use_suppressed
            | primary_insufficient_exposure
            | footprint_ambient_exposure_missing
        )
        out[f"primary_zero_denominator_positive_count_{offense}"] = raw_denominator.le(0.0) & count.gt(0.0)
        if offense == "motor_vehicle_theft":
            out[f"primary_denominator_invalid_{offense}"] = primary_denominator_invalid
        estimate_mode = _estimate_mode(
            non_residential=non_residential,
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
        out[f"recommended_display_geography_{offense}"] = _recommended_display_geography(
            tier=reliability_tier,
            publishable=primary_publishable,
            geo_id_col=geo_id_col,
        )

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
        resident_denominator_invalid = primary_denominator_invalid
        # `resident_denominator` and `resident_insufficient_exposure` are computed above the
        # primary block, because the ambient-blind footprint rule is defined on the resident
        # rate and needs them first.
        resident_publishable = (
            baseline_resident_publishable
            & ~resident_denominator_invalid
            & ~footprint_ambient_exposure_missing
        )
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
            & ~non_residential
            & ~resident_denominator_invalid
            & ~special_use_suppressed
        )
        resident_denominator_reason.loc[resident_insufficient_reason_mask] = "insufficient_exposure"
        resident_denominator_reason.loc[footprint_reason_mask] = INSUFFICIENT_AMBIENT_EXPOSURE_REASON
        # Same Stage 5 F7 hole on the resident arm (771 BG rows).
        resident_denominator_reason.loc[non_residential] = "non_residential"
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
            non_residential
            | resident_denominator_invalid
            | special_use_suppressed
            | resident_insufficient_exposure
            | footprint_ambient_exposure_missing
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
        # Two Stage-5 defects fixed here; the flag stays ADVISORY and gates nothing (Stage 4/5
        # batch: report the firing population, do not couple it to suppression this release).
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
        )


    total_counts_arr = np.sum(np.vstack(total_counts), axis=0)
    out[_expected_count_col("personal")] = personal_counts
    out[_expected_count_col("property")] = property_counts
    out[_expected_count_col("total")] = total_counts_arr
    out["population_zero_with_positive_count"] = pop.le(0) & pd.Series(total_counts_arr, index=out.index).gt(0)
    for offense in OFFENSES_7:
        out[f"crime_density_{offense}"] = _crime_density(out[_expected_count_col(offense)], land_area_sq_mi)
    out["crime_density_total"] = _crime_density(pd.Series(total_counts_arr, index=out.index), land_area_sq_mi)

    out["index_total_part1_resident"] = _resident_part1_index(out, list(OFFENSES_7))[0]
    out["index_personal_part1_resident"] = _resident_part1_index(out, list(PERSONAL_OFFENSES))[0]
    out["index_property_part1_resident"] = _resident_part1_index(out, list(PROPERTY_OFFENSES))[0]
    out["index_total_primary_event_weighted"] = _full_component_index_composite(
        out,
        list(OFFENSES_7),
        index_suffix="primary",
        weights=_national_expected_count_weights(out, list(OFFENSES_7)),
    )
    out["index_total_equal_offense"] = _full_component_index_composite(
        out,
        list(OFFENSES_7),
        index_suffix="primary",
        weights={offense: 1.0 for offense in OFFENSES_7},
    )
    out["index_total_harm"] = _harm_weighted_total_index(out, list(OFFENSES_7))

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
        "non_residential_flag",
        "special_use_tract_flag",
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
        *AGGREGATE_INDEX_FIELDS,
    ]
    ordered_cols = [col for col in ordered_cols if col in out.columns]
    return out[ordered_cols].sort_values([geo_id_col], kind="mergesort").reset_index(drop=True)


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
    if (
        config.force_controls_rebuild
        or config.force_reporting_regimes_rebuild
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
            config=ControlBuildConfig(
                year=int(config.year),
                force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
            ),
            blocked_by=blockers_for_stage("controls", ignore=("outputs",)),
            observation_ignore_blockers=("outputs",),
        )

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

    write_v2_city_incident_shares(
        paths=paths,
        out_path=paths.state_dir / "modeling" / "city_incident_share_surface.parquet",
        config=CityIncidentShareBuildConfig(
            year_start=2018,
            year_end=int(config.year),
            force_rebuild=bool(config.force_city_incident_share_rebuild),
            force_source_refresh=bool(config.force_city_incident_source_refresh),
        ),
        blocked_by=blockers_for_stage("city_incident_shares", ignore=("outputs",)),
    )


def build_v2_outputs(
    *,
    paths: RepoPaths,
    config: AllocationBuildConfig = AllocationBuildConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config = resolve_allocation_build_config(paths, config=config)
    controls = _load_controls(paths, year=config.year)
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

    jurisdiction_components = _build_jurisdiction_component_allocations(
        paths=paths,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        controls=controls,
        year=config.year,
        residual_training_city_shares_path=config.residual_training_city_shares_path,
        residual_training_exclude_validation_case_types=config.residual_training_exclude_validation_case_types,
        residual_training_extra_bg_feature_paths=tuple(config.residual_training_extra_bg_feature_paths),
        residual_feature_policy_path=config.residual_feature_policy_path,
        residual_exclude_feature_policy_classes=tuple(config.residual_exclude_feature_policy_classes),
        residual_exclude_feature_policy_classes_by_offense=tuple(
            config.residual_exclude_feature_policy_classes_by_offense
        ),
        residual_transfer_tau_by_offense=tuple(config.residual_transfer_tau_by_offense),
        city_posterior_reconciliation_tolerance=float(config.city_posterior_reconciliation_tolerance),
        city_posterior_alpha_floor=float(config.city_posterior_alpha_floor),
        city_posterior_alpha_volume_incidents=float(config.city_posterior_alpha_volume_incidents),
        city_posterior_alpha_max_prior_fraction=float(config.city_posterior_alpha_max_prior_fraction),
        enable_county_anchoring=bool(config.enable_county_anchoring),
        agency_estimates=agency_allocation_estimates,
    )
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
    )
    all_components = pd.concat([jurisdiction_components, overlap_components], ignore_index=True)
    bg_cov_raw = _load_bg_covariates(
        paths,
        year=config.year,
        burglary_commercial_weight=config.burglary_commercial_weight,
    )
    burglary_commercial_calibration = dict(bg_cov_raw.attrs.get("burglary_commercial_calibration", {}))
    all_components, zero_target_audit, component_audit = _redistribute_zero_target_components(
        all_components,
        bg_cov_raw,
    )

    bg_counts = (
        all_components.groupby(["state_fips", "bg_id", "tract_id", "offense"], dropna=False)["component_count"]
        .sum()
        .reset_index()
        .pivot_table(index=["state_fips", "bg_id", "tract_id"], columns="offense", values="component_count", fill_value=0.0)
        .reset_index()
    )
    bg_counts = bg_counts.rename(columns={offense: _expected_count_col(offense) for offense in OFFENSES_7})

    # Compositional input to the ambient-blind footprint eligibility rule: per block group and
    # offense, how much of the modelled mass arrived through a custom-footprint overlap layer.
    # Carried as MASS so the tract rollup can sum it; the share is derived where it is used.
    footprint_components = all_components[
        all_components["jurisdiction_type"].astype("string").eq("custom_footprint_overlap_layer")
    ]
    footprint_counts = (
        footprint_components.groupby(["state_fips", "bg_id", "tract_id", "offense"], dropna=False)[
            "component_count"
        ]
        .sum()
        .reset_index()
        .pivot_table(
            index=["state_fips", "bg_id", "tract_id"],
            columns="offense",
            values="component_count",
            fill_value=0.0,
        )
        .reset_index()
        if not footprint_components.empty
        else pd.DataFrame(columns=["state_fips", "bg_id", "tract_id"])
    )
    footprint_counts = footprint_counts.rename(
        columns={offense: _footprint_derived_count_col(offense) for offense in OFFENSES_7}
    )
    for offense in OFFENSES_7:
        col = _footprint_derived_count_col(offense)
        if col not in footprint_counts.columns:
            footprint_counts[col] = 0.0
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
        year=int(config.year),
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

    bg_cal = bg_out.copy()
    for offense in OFFENSES_7:
        ratio = bg_cal["state_fips"].map(lambda st: ratio_map.get((str(st).zfill(2), offense), 1.0))
        expected_col = _expected_count_col(offense)
        bg_cal[expected_col] = pd.to_numeric(bg_cal[expected_col], errors="coerce").fillna(0.0) * ratio
        # Scaled by the same per-state ratio as the count it is a part of, so the derived
        # footprint share is identical on both twins rather than off by the calibration factor.
        footprint_col = _footprint_derived_count_col(offense)
        bg_cal[footprint_col] = pd.to_numeric(bg_cal[footprint_col], errors="coerce").fillna(0.0) * ratio

    bg_cal = _finalize_output(
        bg_cal,
        geo_id_col="block_group_geoid",
        population_col=population_col,
        config=config,
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
    )
    bg_cal, tract_cal = enrich_confidence_surfaces(
        block_group_surface=bg_cal,
        tract_surface=tract_cal,
        artifacts=confidence_artifacts,
        population_col=population_col,
    )
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
    }
    return bg_out, tract_out, bg_cal, tract_cal, zero_target_audit, component_audit, diagnostics


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
        bg_out = apply_rare_offense_tract_support(bg_out, tract_out)
        bg_cal = apply_rare_offense_tract_support(bg_cal, tract_cal)
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
        }
        if block_group_fbi_out is not None and tract_fbi_out is not None:
            bg_cal.to_parquet(block_group_fbi_out, index=False)
            tract_cal.to_parquet(tract_fbi_out, index=False)
            summary["fbi_calibrated_written"] = True
        summary["aggregate_index_normalizers"] = {
            "block_group_ags_core": _aggregate_index_normalizers(bg_out),
            "tract_ags_core": _aggregate_index_normalizers(tract_out),
            "block_group_fbi_calibrated": _aggregate_index_normalizers(bg_cal),
            "tract_fbi_calibrated": _aggregate_index_normalizers(tract_cal),
        }
        manifest_path = build_manifest_out or (
            block_group_ags_core_out.parent / f"crimerisk_output_build_{int(resolved_config.year)}.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        zero_target_audit_path = manifest_path.parent / f"zero_target_denominator_audit_{int(resolved_config.year)}.parquet"
        component_audit_path = manifest_path.parent / f"allocation_component_denominator_audit_{int(resolved_config.year)}.parquet"
        publishability_audit_path = manifest_path.parent / f"denominator_publishability_audit_{int(resolved_config.year)}.csv"
        city_posterior_diagnostics_path = manifest_path.parent / f"city_posterior_diagnostics_{int(resolved_config.year)}.parquet"
        zero_target_audit.to_parquet(zero_target_audit_path, index=False)
        component_audit.to_parquet(component_audit_path, index=False)
        _build_publishability_audit(bg_out).to_csv(publishability_audit_path, index=False)
        city_posterior_diagnostics = diagnostics.get("city_posterior_diagnostics")
        if isinstance(city_posterior_diagnostics, pd.DataFrame):
            city_posterior_diagnostics.to_parquet(city_posterior_diagnostics_path, index=False)
            summary["city_posterior_diagnostics_rows"] = int(len(city_posterior_diagnostics))
            summary["city_posterior_diagnostics_path"] = str(city_posterior_diagnostics_path)
        summary["zero_target_denominator_audit_rows"] = int(len(zero_target_audit))
        summary["zero_target_denominator_audit_path"] = str(zero_target_audit_path)
        summary["allocation_component_denominator_audit_rows"] = int(len(component_audit))
        summary["allocation_component_denominator_audit_path"] = str(component_audit_path)
        summary["denominator_publishability_audit_path"] = str(publishability_audit_path)
        output_paths = {
            "block_group_ags_core": block_group_ags_core_out,
            "tract_ags_core": tract_ags_core_out,
            "block_group_fbi_calibrated": block_group_fbi_out,
            "tract_fbi_calibrated": tract_fbi_out,
            "zero_target_denominator_audit": zero_target_audit_path,
            "allocation_component_denominator_audit": component_audit_path,
            "denominator_publishability_audit": publishability_audit_path,
            "city_posterior_diagnostics": (
                city_posterior_diagnostics_path
                if isinstance(city_posterior_diagnostics, pd.DataFrame)
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
