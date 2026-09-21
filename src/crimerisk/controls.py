"""Stage 3 — the jurisdiction control panel.

Assembled, not estimated. `jurisdiction_targets` builds the ownership skeleton and sums
the Stage-1 agency estimates onto it; this module turns that into the published control
row (target, uplift/fill split, provenance labels), runs `benchmark_imputation` on the
PRE-imputation controls so silent territory is sized against the FBI benchmark before any
mass is added, and reconciles the result to the CDE state series.

The build order is load-bearing and is the reason the Jackson MS shape cannot recur:
the skeleton exists before any mass is aggregated, benchmark eligibility is decided from
the agency ledger against that skeleton, and imputed mass is added last.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.benchmark_imputation import (
    BenchmarkImputation,
    BenchmarkImputationConfig,
    apply_benchmark_imputation_to_controls,
    build_benchmark_imputation,
    write_benchmark_imputation_artifacts,
    COUNTY_UNIT_KIND,
    MUNICIPAL_UNIT_KIND,
    STATE_REMAINDER_SUFFIX,
)
from crimerisk.build_freshness import artifact_is_current, write_dependency_stamp
from crimerisk.jurisdiction_targets import (
    IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN,
    JurisdictionTargetConfig,
    LANE_REPORTED_COLUMNS,
    LANE_TARGET_COMPONENT_COLUMNS,
    _assert_row_identity,
    build_agency_target_panel_slice,
    build_jurisdiction_ownership,
    build_jurisdiction_year_estimates,
    build_ownership_exclusions,
    load_crosswalk,
    write_jurisdiction_ownership_exclusions,
)
from crimerisk.level_lane import (
    AdmissionArtifacts,
    admission_dependency_paths,
    apply_level_lane_admission,
)
from crimerisk.ct_town_lane import load_ct_town_coverage_registry, residualize_ctcsp_overlap
from crimerisk.federal_ori_scope import exclude_national_hq_rows, load_federal_ori_scope_registry
from crimerisk.observations import (
    ObservationBuildConfig,
    get_v2_observation_paths,
    observations_artifacts_are_current,
    write_v2_observations,
)
from crimerisk.paths import RepoPaths
from crimerisk.service_scopes import service_wide_scope_dependency_paths
from crimerisk.reporting_regimes import (
    ReportingRegimeBuildConfig,
    get_v2_reporting_regimes_path,
    reporting_regime_dependency_paths,
    reporting_regimes_artifact_is_current,
    write_v2_reporting_regimes,
)
from crimerisk.stage_locks import blockers_for_stage, stage_write_lock
from crimerisk.trend_fills import (
    FILL_MAX_REFERENCE_AGE_YEARS,
    apply_stage1_adjudicated_usability,
    apply_masked_gap_reclassification,
    build_agency_allocation_target_estimates,
    build_agency_trend_fill_panel,
    build_masked_gap_flags,
    build_usability_directives,
    resolve_ori_succession,
    load_agency_jurisdiction_crosswalk,
)
from crimerisk.scope import PRODUCTION_SCOPE_EXCLUDE
from crimerisk.stage1_adjudications import succession_dependency_paths


CDE_OFFENSE_MAP: dict[str, str] = {
    "murder": "homicide",
    "rape": "rape_revised",
    "robbery": "robbery",
    "aggravated_assault": "aggravated_assault",
    "burglary": "burglary",
    "larceny": "larceny",
    "motor_vehicle_theft": "motor_vehicle_theft",
}


CONTROL_KEY_COLUMNS = [
    "jurisdiction_id",
    "jurisdiction_type",
    "jurisdiction_name",
    "state_fips",
    "state_abbr",
    "geo_type",
    "geoid",
    "offense",
]


CONTROL_COLUMNS = [
    *CONTROL_KEY_COLUMNS,
    "year",
    # ownership / exposure
    "bucket_population",
    "pop_band",
    "owns_block_group_geometry",
    "ownership_basis",
    "crosswalk_agency_count",
    "contributing_agency_count",
    "estimating_agency_count",
    # the reported side, and every candidate lane's own rollup
    "reported_count_preferred",
    *LANE_REPORTED_COLUMNS.values(),
    "observation_weight_preferred",
    "mean_months_reported_preferred",
    "quality_tier_preferred",
    # descriptive provenance -- never a selection input
    "preferred_source",
    "preferred_source_family",
    "preferred_source_origin",
    "preferred_raw_data_source",
    "preferred_source_lane",
    "preferred_reporting_mode",
    "preferred_conversion_status",
    "preferred_state_exception_flag",
    "preferred_cius_reference_flag",
    "dominant_reporting_regime",
    "relationship_type_preferred",
    "overlap_subtype_preferred",
    # component provenance: which lane and which estimate class carried the mass
    *LANE_TARGET_COMPONENT_COLUMNS,
    "observed_component_count",
    "partial_component_count",
    "fill_component_count",
    # the target and its decomposition
    "adjusted_count_ags_core",
    "estimated_count_ags_core",
    "adjustment_total",
    IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN,
    "needs_partial_reporting_uplift",
    "partial_reporting_uplift_count",
    "needs_zero_month_fill",
    "current_year_fill_count",
    "needs_current_year_fill",
    "current_year_fill_refusal_reason",
    "estimate_source",
    "estimate_confidence",
    "estimated_from_panel",
    # Level-lane disclosure is independent of the downstream spatial-share lane.
    "level_admission_status",
    "level_semantic_status",
    "level_admission_reason",
    "level_repair_mode",
    "level_repair_share",
    "level_external_check_status",
    "level_unresolved_review",
    "level_benchmark_weight",
    "external_check_status",
    "benchmark_conflict_kind",
    "benchmark_weight",
    "unresolved_level_flag",
]


@dataclass(frozen=True)
class ControlBuildConfig:
    year: int = 2024
    exclude_scope_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE))
    force_reporting_regimes_rebuild: bool = False
    # Opt-in to the E5 imputation fixes as a package: size-aware municipal rates, the
    # cell-exposure floor and the attached empirical bounds. Default OFF -- the promoted
    # chain keeps the measured-as-shipped rule until owner promotion.
    # Contract: analysis_scratch/final_phase/IMPUTATION_V2_CONTRACT.md.
    enable_imputation_v2: bool = False


def _benchmark_imputation_config(config: ControlBuildConfig) -> BenchmarkImputationConfig:
    return BenchmarkImputationConfig(
        year=int(config.year),
        enable_size_aware_municipal_rates=bool(config.enable_imputation_v2),
        enable_cell_exposure_floor=bool(config.enable_imputation_v2),
        attach_empirical_bounds=bool(config.enable_imputation_v2),
    )


def _jurisdiction_target_config(config: ControlBuildConfig) -> JurisdictionTargetConfig:
    return JurisdictionTargetConfig(
        year_start=2018,
        target_year=int(config.year),
        exclude_scope_state_abbrs=config.exclude_scope_state_abbrs,
        force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
    )


def _cde_estimates_path(paths: RepoPaths, *, year: int) -> Path:
    """Raw FBI CDE estimated-crimes bundle for a target year.

    The CDE publishes one cumulative series file per data year
    (estimated_crimes_1979_<year>.csv, in data/FBI-CDE-Estimates-1979-<year>/),
    released with "Reported Crimes in the Nation, <year>" (~Aug of the following
    year). Only bundles that have been downloaded into data/ are usable.
    """
    return (
        paths.data_dir
        / f"FBI-CDE-Estimates-1979-{int(year)}"
        / f"estimated_crimes_1979_{int(year)}.csv"
    )


def _load_cde_estimates(
    paths: RepoPaths, *, year: int, exclude_state_abbrs: tuple[str, ...]
) -> pd.DataFrame:
    path = _cde_estimates_path(paths, year=year)
    if not path.exists():
        raise FileNotFoundError(
            f"FBI CDE estimates bundle for {int(year)} not present at {path}. "
            f"The FBI publishes estimated_crimes_1979_{int(year)}.csv with "
            f"'Reported Crimes in the Nation, {int(year)}' (~Aug {int(year) + 1} by precedent); "
            f"download it into data/FBI-CDE-Estimates-1979-{int(year)}/ before running a "
            f"{int(year)} controls build."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["year"].astype(int) == int(year)].copy()
    df["state_abbr"] = df["state_abbr"].astype("string").str.strip().str.upper()
    df = df[df["state_abbr"].notna()].copy()
    df = df[~df["state_abbr"].isin({"<NA>", "NAN", *exclude_state_abbrs})].copy()
    for col in ["population", *CDE_OFFENSE_MAP.values()]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
            )
    return df


def _controls_dependency_paths(paths: RepoPaths, *, year: int) -> list[Path]:
    reporting_config = ReportingRegimeBuildConfig(year_start=2018, year_end=int(year))
    dependencies = [
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        paths.state_dir / "reference" / "agency_master.parquet",
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        paths.repo_root / "configs" / "federal_ori_scope.csv",
        paths.repo_root / "configs" / "overlap_custom_footprints.csv",
        *service_wide_scope_dependency_paths(paths),
        get_v2_reporting_regimes_path(paths),
        _cde_estimates_path(paths, year=year),
        paths.data_dir
        / f"FBI-NIBRS-Tables-{int(year)}"
        / "parsed"
        / f"nibrs_offense_type_by_agency_{int(year)}.parquet",
        Path(__file__),
        paths.repo_root / "src" / "crimerisk" / "jurisdiction_targets.py",
        paths.repo_root / "src" / "crimerisk" / "source_selection.py",
        paths.repo_root / "src" / "crimerisk" / "source_provenance.py",
        paths.repo_root / "src" / "crimerisk" / "trend_fills.py",
        paths.repo_root / "src" / "crimerisk" / "benchmark_imputation.py",
        *succession_dependency_paths(paths),
        *admission_dependency_paths(paths),
        paths.repo_root
        / "analysis_scratch"
        / f"agency_target_estimates_{int(year)}.parquet",
        paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        *reporting_regime_dependency_paths(paths, config=reporting_config),
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in dependencies:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def controls_artifacts_are_current(
    paths: RepoPaths,
    *,
    year: int,
    state_out_path: Path,
    jurisdiction_out_path: Path,
    jurisdiction_year_estimates_out_path: Path | None = None,
) -> bool:
    dependencies = _controls_dependency_paths(paths, year=year)
    outputs = [state_out_path, jurisdiction_out_path]
    if jurisdiction_year_estimates_out_path is not None:
        outputs.append(jurisdiction_year_estimates_out_path)
    return all(artifact_is_current(path, dependencies) for path in outputs)


def _ensure_controls_dependencies(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> None:
    agency_out_path, jurisdiction_out_path = get_v2_observation_paths(paths)
    observation_config = ObservationBuildConfig(year_start=2018, year_end=int(config.year))
    if not observations_artifacts_are_current(
        paths,
        config=observation_config,
        agency_out_path=agency_out_path,
        jurisdiction_out_path=jurisdiction_out_path,
    ):
        write_v2_observations(
            paths=paths,
            agency_out_path=agency_out_path,
            jurisdiction_out_path=jurisdiction_out_path,
            config=observation_config,
            blocked_by=blockers_for_stage(
                "observations", ignore=("controls", *observation_ignore_blockers)
            ),
            reference_ignore_blockers=("controls", *observation_ignore_blockers),
        )

    reporting_path = get_v2_reporting_regimes_path(paths)
    reporting_config = ReportingRegimeBuildConfig(
        year_start=2018, year_end=int(config.year)
    )
    if (
        config.force_reporting_regimes_rebuild
        or not reporting_regimes_artifact_is_current(
            paths,
            config=reporting_config,
            out_path=reporting_path,
        )
    ):
        write_v2_reporting_regimes(
            paths=paths,
            out_path=reporting_path,
            config=reporting_config,
            blocked_by=blockers_for_stage("reporting_regimes", ignore=("controls",)),
            observation_ignore_blockers=("controls", *observation_ignore_blockers),
        )


@dataclass(frozen=True)
class Stage1Consumption:
    """Everything Stage 3 reads out of Stage 1, built once per controls build.

    One object because the four pieces have to agree: the estimates are computed from
    the panel, the succession ledger decides which ORIs the estimates dropped, and
    `benchmark_imputation` has to see the same ledger or a superseded ORI is excluded
    from the fill lane and readmitted to the benchmark lane.
    """

    agency_panel: pd.DataFrame
    agency_preferred_target_year: pd.DataFrame
    agency_estimates: pd.DataFrame
    succession_ledger: pd.DataFrame
    admission_disposition: pd.DataFrame = field(default_factory=pd.DataFrame)
    admission_review_queue: pd.DataFrame = field(default_factory=pd.DataFrame)
    # What the Stage-1 adjudication registries did on this build, for the controls summary.
    stage1_adjudication_counts: dict = field(default_factory=dict)


def _consume_precomputed_masked_gap_fills(
    *,
    paths: RepoPaths,
    year: int,
    agency_estimates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover masked-gap fills already computed by the accepted estimator run.

    The v21 controls refactor recomputed estimates after the masked-gap treatment
    artifact was produced.  A later agency-wide adjudication/recency pass then removed
    some offense rows (Savannah exposed it), so controls never saw values that had
    already passed the masked-gap detector and fill ladder.  Only rows carrying the
    artifact's explicit ``masked_gap_reclassified`` provenance are eligible here.
    """
    artifact = paths.repo_root / "analysis_scratch" / f"agency_target_estimates_{int(year)}.parquet"
    audit_columns = [
        "ori9",
        "offense",
        "computed_fill_count",
        "prewire_fill_count",
        "consumed_fill_count",
        "consumption_status",
    ]
    if not artifact.exists():
        return agency_estimates, pd.DataFrame(columns=audit_columns)
    computed = pd.read_parquet(artifact)
    required = {
        "ori9",
        "offense",
        "estimated_count",
        "agency_adjustment_count",
        "agency_estimate_source",
        "masked_gap_reclassified",
    }
    missing = sorted(required - set(computed.columns))
    if missing:
        raise ValueError(f"computed masked-gap fill artifact is missing columns {missing}: {artifact}")
    eligible = computed[
        computed["masked_gap_reclassified"].fillna(False).astype(bool)
        & pd.to_numeric(computed["agency_adjustment_count"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    if eligible.empty:
        return agency_estimates, pd.DataFrame(columns=audit_columns)
    eligible["ori9"] = eligible["ori9"].astype("string")
    eligible["offense"] = eligible["offense"].astype("string")
    eligible = eligible.drop_duplicates(["ori9", "offense"], keep="last")

    current = agency_estimates.copy()
    current["ori9"] = current["ori9"].astype("string")
    current["offense"] = current["offense"].astype("string")
    key = ["ori9", "offense"]
    current_adjustment = current.set_index(key)["agency_adjustment_count"].pipe(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    eligible_index = eligible.set_index(key)
    eligible_adjustment = pd.to_numeric(
        eligible_index["agency_adjustment_count"], errors="coerce"
    ).fillna(0.0)
    prewire = current_adjustment.reindex(eligible_adjustment.index).fillna(0.0)
    if {
        "level1_admission_status", "repair_mass", "invalid_fragment_audit_mass"
    }.issubset(current.columns):
        # The scratch artifact predates reason-coded mass semantics. It remains an audit
        # comparator, but may not overwrite a v2 repair (that would reintroduce its
        # max-current ratchet and double-count rejected fragments).
        audit = eligible_adjustment.rename("computed_fill_count").reset_index()
        audit["prewire_fill_count"] = prewire.to_numpy()
        audit["consumed_fill_count"] = prewire.to_numpy()
        audit["consumption_status"] = np.where(
            prewire.ge(eligible_adjustment - 1e-6),
            "v2_recomputed",
            "explicitly_refused_legacy_artifact",
        )
        audit_path = paths.state_dir / "qa" / "stage1_screen" / "i_computed_fill_consumption.csv"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)
        return current.sort_values(["state_fips", "ori9", "offense"], kind="mergesort").reset_index(drop=True), audit
    recover = eligible_adjustment.gt(prewire + 1e-6)
    recovered_keys = set(eligible_adjustment.index[recover].tolist())
    if recovered_keys:
        keep_current = ~pd.MultiIndex.from_frame(current[key]).isin(recovered_keys)
        replacement = eligible_index.loc[recover].reset_index().reindex(columns=current.columns)
        current = pd.concat([current.loc[keep_current], replacement], ignore_index=True)

    consumed = (
        current.set_index(key)["agency_adjustment_count"]
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .reindex(eligible_adjustment.index)
        .fillna(0.0)
    )
    audit = eligible_adjustment.rename("computed_fill_count").reset_index()
    audit["prewire_fill_count"] = prewire.to_numpy()
    audit["consumed_fill_count"] = consumed.to_numpy()
    audit["consumption_status"] = np.where(
        audit["consumed_fill_count"].ge(audit["computed_fill_count"] - 1e-6),
        np.where(
            audit["prewire_fill_count"].ge(audit["computed_fill_count"] - 1e-6),
            "already_consumed",
            "recovered_by_wiring",
        ),
        "unconsumed",
    )
    bad = audit[audit["consumption_status"].eq("unconsumed")]
    if not bad.empty:
        raise ValueError(
            "computed masked-gap fills remain unconsumed after wiring: "
            + str(bad.head(20).to_dict(orient="records"))
        )
    audit_path = paths.state_dir / "qa" / "stage1_screen" / "i_computed_fill_consumption.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    return current.sort_values(["state_fips", "ori9", "offense"], kind="mergesort").reset_index(drop=True), audit


def build_stage1_consumption(
    *, paths: RepoPaths, config: ControlBuildConfig
) -> Stage1Consumption:
    year = int(config.year)
    agency_panel = build_agency_trend_fill_panel(
        paths=paths,
        year_start=2018,
        year_end=year,
        force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
        exclude_state_abbrs=tuple(config.exclude_scope_state_abbrs),
    )
    admission: AdmissionArtifacts = apply_level_lane_admission(
        agency_panel, paths=paths, target_year=year
    )
    agency_panel = admission.panel
    masked_gap_flags = build_masked_gap_flags(
        paths, target_year=year, agency_panel=agency_panel
    )
    agency_panel = apply_masked_gap_reclassification(
        agency_panel, target_year=year, masked_gap_flags=masked_gap_flags
    )
    directives = build_usability_directives(paths, target_year=year)
    agency_panel, legacy_admission_counts = apply_stage1_adjudicated_usability(
        agency_panel,
        target_year=year,
        directives=directives,
    )
    # The succession rule plus its adjudicated residue, resolved once so the estimates,
    # the controls and `benchmark_imputation` all read the same ledger -- the adjudicated
    # rows have to reach every consumer for the same reason the rule's rows do.
    succession_ledger, succession_summary = resolve_ori_succession(
        paths=paths,
        agency_panel=agency_panel,
        agency_jurisdiction_crosswalk=load_agency_jurisdiction_crosswalk(paths),
        target_year=year,
        max_reference_age_years=FILL_MAX_REFERENCE_AGE_YEARS,
    )
    agency_estimates = build_agency_allocation_target_estimates(
        paths=paths,
        year=year,
        agency_panel=agency_panel,
        masked_gap_flags=masked_gap_flags,
        succession_ledger=succession_ledger,
    )
    agency_estimates, fill_consumption_audit = _consume_precomputed_masked_gap_fills(
        paths=paths,
        year=year,
        agency_estimates=agency_estimates,
    )
    consumption = Stage1Consumption(
        agency_panel=agency_panel,
        agency_preferred_target_year=build_agency_target_panel_slice(
            agency_panel=agency_panel, target_year=year
        ),
        agency_estimates=agency_estimates,
        succession_ledger=succession_ledger,
        admission_disposition=admission.disposition,
        admission_review_queue=admission.review_queue,
    )
    consumption.stage1_adjudication_counts.update(succession_summary)
    consumption.stage1_adjudication_counts.update(legacy_admission_counts)
    consumption.stage1_adjudication_counts.update(
        {
            "computed_masked_gap_fills": int(len(fill_consumption_audit)),
            "computed_masked_gap_fills_recovered_by_wiring": int(
                fill_consumption_audit["consumption_status"].eq("recovered_by_wiring").sum()
            ),
        }
    )
    consumption.stage1_adjudication_counts.update(
        agency_estimates.attrs.get("stage1_adjudications", {})
    )
    admission_path = paths.state_dir / "controls" / f"level_lane_admission_{year}.parquet"
    review_path = paths.state_dir / "controls" / f"level_lane_review_queue_{year}.csv"
    admission_path.parent.mkdir(parents=True, exist_ok=True)
    admission.disposition.to_parquet(admission_path, index=False)
    admission.review_queue.to_csv(review_path, index=False)
    return consumption


def build_jurisdiction_controls(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig = ControlBuildConfig(),
    jurisdiction_year_estimates: pd.DataFrame | None = None,
    stage1: Stage1Consumption | None = None,
) -> pd.DataFrame:
    """The pre-imputation control panel: the target-year slice of the consumed panel.

    The control IS the jurisdiction-year estimate for the target year. They were two
    constructions overlaid on each other before this rewrite -- a per-offense source
    preference over the observation rollup, with the canonical panel merged on top
    wherever it happened to have a row -- and the overlay is what let the metadata of one
    lane be published against the count of another.
    """
    if jurisdiction_year_estimates is None:
        if stage1 is None:
            stage1 = build_stage1_consumption(paths=paths, config=config)
        jurisdiction_year_estimates = build_jurisdiction_year_estimates(
            paths=paths,
            config=_jurisdiction_target_config(config),
            agency_panel=stage1.agency_panel,
            agency_estimates=stage1.agency_estimates,
            succession_ledger=stage1.succession_ledger,
        )
    out = jurisdiction_year_estimates[
        jurisdiction_year_estimates["year"].astype(int).eq(int(config.year))
    ].copy()

    estimated = pd.to_numeric(out["estimated_count"], errors="coerce").fillna(0.0)
    reported = pd.to_numeric(out["reported_count_preferred"], errors="coerce").fillna(0.0)
    uplift = pd.to_numeric(
        out["partial_reporting_uplift_count"], errors="coerce"
    ).fillna(0.0)

    out["adjusted_count_ags_core"] = estimated
    out["estimated_count_ags_core"] = estimated
    # Signed by definition: identity resolution can remove duplicate reported mass,
    # while uplift, fill, and benchmark imputation add mass.
    out["adjustment_total"] = estimated - reported
    # Keyed on the uplift the agencies actually carried, not on a jurisdiction-level
    # months column. The old flag read `mean_months_reported_preferred`, an unweighted
    # mean over contributing agencies, so it over-fired on complete cities that share a
    # jurisdiction with a silent agency and under-fired on the remainder pools, where
    # 3,231 agencies share 47 rows and 75,865 counts of genuine partial-year uplift were
    # published as `current_year_fill` instead.
    out["needs_partial_reporting_uplift"] = uplift.gt(1e-12)
    out["needs_zero_month_fill"] = reported.le(0.0) & estimated.gt(0.0)
    out["needs_current_year_fill"] = out["estimate_confidence"].astype("string").eq("low")
    if "current_year_fill_refusal_reason" not in out.columns:
        out["current_year_fill_refusal_reason"] = pd.Series(
            pd.NA, index=out.index, dtype="string"
        )
    out["year"] = int(config.year)
    return (
        out.reindex(columns=CONTROL_COLUMNS)
        .sort_values(
            ["state_fips", "jurisdiction_type", "jurisdiction_id", "offense"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_controls_benchmark_imputation(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig,
    controls: pd.DataFrame,
    stage1: Stage1Consumption | None = None,
    imputation_config: BenchmarkImputationConfig | None = None,
) -> BenchmarkImputation:
    """Class A: size and place the mass that silent agencies never contributed.

    Runs on the PRE-imputation controls, because the accounting identity's locked side
    is exactly those observed targets; the result is then added back on top of them.
    """
    if stage1 is None:
        stage1 = build_stage1_consumption(paths=paths, config=config)
    cde = _load_cde_estimates(
        paths, year=config.year, exclude_state_abbrs=config.exclude_scope_state_abbrs
    )
    return build_benchmark_imputation(
        paths=paths,
        controls=controls,
        cde_estimates=cde,
        agency_preferred=stage1.agency_preferred_target_year,
        agency_estimates=stage1.agency_estimates,
        succession_ledger=stage1.succession_ledger,
        config=imputation_config or _benchmark_imputation_config(config),
    )


def build_state_control_comparison(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig = ControlBuildConfig(),
    controls: pd.DataFrame | None = None,
    benchmark_imputation: BenchmarkImputation | None = None,
) -> pd.DataFrame:
    if controls is None:
        controls = build_jurisdiction_controls(paths=paths, config=config)
    cde = _load_cde_estimates(
        paths, year=config.year, exclude_state_abbrs=config.exclude_scope_state_abbrs
    )

    state_parts = (
        controls.groupby(
            ["state_fips", "state_abbr", "jurisdiction_type", "offense"], dropna=False
        )
        .agg(
            preferred_total=("reported_count_preferred", "sum"),
            adjusted_total=("adjusted_count_ags_core", "sum"),
            srs_total=("reported_count_srs", "sum"),
            nibrs_total=("reported_count_nibrs", "sum"),
            partial_reporting_uplift_total=("partial_reporting_uplift_count", "sum"),
            current_year_fill_total=("current_year_fill_count", "sum"),
            identity_resolution_adjustment_total=(
                IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN,
                "sum",
            ),
            uplift_candidate_total=(
                "reported_count_preferred",
                lambda s: float(
                    s[controls.loc[s.index, "needs_partial_reporting_uplift"]].sum()
                ),
            ),
            zero_month_pool_total=(
                "reported_count_preferred",
                lambda s: float(
                    s[controls.loc[s.index, "needs_zero_month_fill"]].sum()
                ),
            ),
        )
        .reset_index()
    )

    totals = (
        state_parts.groupby(["state_fips", "state_abbr", "offense"], dropna=False)
        .agg(
            ags_core_reported_total=("preferred_total", "sum"),
            ags_core_adjusted_total=("adjusted_total", "sum"),
            internal_srs_total=("srs_total", "sum"),
            internal_nibrs_total=("nibrs_total", "sum"),
            partial_reporting_pool=("uplift_candidate_total", "sum"),
            partial_reporting_uplift_total=("partial_reporting_uplift_total", "sum"),
            current_year_fill_total=("current_year_fill_total", "sum"),
            identity_resolution_adjustment_total=(
                "identity_resolution_adjustment_total",
                "sum",
            ),
            zero_month_pool_total=("zero_month_pool_total", "sum"),
        )
        .reset_index()
    )

    def part_frame(kind: str, prefix: str) -> pd.DataFrame:
        return state_parts[state_parts["jurisdiction_type"] == kind][
            [
                "state_fips",
                "state_abbr",
                "offense",
                "preferred_total",
                "adjusted_total",
                "srs_total",
                "nibrs_total",
                "partial_reporting_uplift_total",
                "current_year_fill_total",
                "identity_resolution_adjustment_total",
            ]
        ].rename(
            columns={
                "preferred_total": f"{prefix}_reported_total",
                "adjusted_total": f"{prefix}_adjusted_total",
                "srs_total": f"{prefix}_srs_total",
                "nibrs_total": f"{prefix}_nibrs_total",
                "partial_reporting_uplift_total": f"{prefix}_partial_reporting_uplift_total",
                "current_year_fill_total": f"{prefix}_current_year_fill_total",
                "identity_resolution_adjustment_total": (
                    f"{prefix}_identity_resolution_adjustment_total"
                ),
            }
        )

    municipal = part_frame("municipal", "municipal")
    nonmunicipal = part_frame("state_nonmunicipal_remainder", "nonmunicipal")
    overlap = part_frame("statewide_overlap_layer", "overlap")

    out = totals.merge(
        municipal, on=["state_fips", "state_abbr", "offense"], how="left"
    )
    out = out.merge(
        nonmunicipal, on=["state_fips", "state_abbr", "offense"], how="left"
    )
    out = out.merge(overlap, on=["state_fips", "state_abbr", "offense"], how="left")

    cde_long = cde.melt(
        id_vars=["state_abbr", "state_name", "population"],
        value_vars=list(CDE_OFFENSE_MAP.values()),
        var_name="cde_offense",
        value_name="fbi_cde_estimated_total",
    )
    reverse_map = {v: k for k, v in CDE_OFFENSE_MAP.items()}
    cde_long["offense"] = cde_long["cde_offense"].map(reverse_map)

    state_fips_map = (
        controls[["state_abbr", "state_fips"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["state_abbr", "state_fips"])
        .drop_duplicates(subset=["state_abbr"], keep="first")
    )
    cde_long = cde_long.merge(state_fips_map, on="state_abbr", how="left")

    out = out.merge(
        cde_long[
            [
                "state_fips",
                "state_abbr",
                "state_name",
                "population",
                "offense",
                "fbi_cde_estimated_total",
            ]
        ],
        on=["state_fips", "state_abbr", "offense"],
        how="left",
    )

    out["cde_gap_to_reported"] = (
        out["fbi_cde_estimated_total"] - out["ags_core_reported_total"]
    )
    out["cde_gap_to_adjusted"] = (
        out["fbi_cde_estimated_total"] - out["ags_core_adjusted_total"]
    )
    out["cde_gap_positive_only"] = out["cde_gap_to_reported"].clip(lower=0.0)
    out["cde_gap_positive_only_adjusted"] = out["cde_gap_to_adjusted"].clip(lower=0.0)
    out["reported_to_cde_ratio"] = (
        out["ags_core_reported_total"] / out["fbi_cde_estimated_total"]
    )
    out["adjusted_to_cde_ratio"] = (
        out["ags_core_adjusted_total"] / out["fbi_cde_estimated_total"]
    )
    out["partial_pool_share_of_reported"] = (
        out["partial_reporting_pool"] / out["ags_core_reported_total"]
    )
    out["gap_to_partial_pool_ratio"] = (
        out["cde_gap_positive_only"] / out["partial_reporting_pool"]
    )
    if benchmark_imputation is not None and not benchmark_imputation.state_identity.empty:
        identity = benchmark_imputation.state_identity[
            [
                "state_fips",
                "offense",
                "benchmark_residual",
                "modeled_pool",
                "modeled_pool_variance",
                "benchmark_relative_uncertainty",
                "benchmark_standard_error",
                "benchmark_variance",
                "benchmark_weight",
                "imputed_total",
                "posterior_total",
                "standardized_disagreement",
                "unused_benchmark_headroom",
                "unfilled_modeled_pool",
                "silent_unit_count",
                "silent_unit_population",
                "conflict_kind",
            ]
        ].rename(
            columns={
                "benchmark_residual": "benchmark_residual_pre_imputation",
                "modeled_pool": "benchmark_modeled_pool",
                "modeled_pool_variance": "benchmark_modeled_pool_variance",
                "benchmark_relative_uncertainty": "benchmark_relative_uncertainty",
                "benchmark_standard_error": "benchmark_standard_error",
                "benchmark_variance": "benchmark_variance",
                "benchmark_weight": "benchmark_weight",
                "imputed_total": "benchmark_imputed_total",
                "posterior_total": "benchmark_posterior_total",
                "standardized_disagreement": "benchmark_standardized_disagreement",
                "unused_benchmark_headroom": "benchmark_unused_headroom",
                "unfilled_modeled_pool": "benchmark_unfilled_modeled_pool",
                "silent_unit_count": "benchmark_silent_unit_count",
                "silent_unit_population": "benchmark_silent_unit_population",
                "conflict_kind": "benchmark_conflict_kind",
            }
        )
        identity["state_fips"] = identity["state_fips"].astype("string").str.zfill(2)
        out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
        out = out.merge(identity, on=["state_fips", "offense"], how="left")
    else:
        for column, default in (
            ("benchmark_residual_pre_imputation", 0.0),
            ("benchmark_modeled_pool", 0.0),
            ("benchmark_modeled_pool_variance", 0.0),
            ("benchmark_relative_uncertainty", np.nan),
            ("benchmark_standard_error", np.nan),
            ("benchmark_variance", np.nan),
            ("benchmark_weight", 0.0),
            ("benchmark_imputed_total", 0.0),
            ("benchmark_posterior_total", 0.0),
            ("benchmark_standardized_disagreement", np.nan),
            ("benchmark_unused_headroom", 0.0),
            ("benchmark_unfilled_modeled_pool", 0.0),
            ("benchmark_silent_unit_count", 0),
            ("benchmark_silent_unit_population", 0.0),
        ):
            out[column] = default
        out["benchmark_conflict_kind"] = "not_evaluated"
    out["year"] = int(config.year)
    out["cde_exact_sensitivity_factor"] = np.divide(
        pd.to_numeric(out["fbi_cde_estimated_total"], errors="coerce").fillna(0.0),
        pd.to_numeric(out["ags_core_adjusted_total"], errors="coerce").replace(0.0, np.nan),
    )

    return (
        out[
            [
                "year",
                "state_fips",
                "state_abbr",
                "state_name",
                "population",
                "offense",
                "ags_core_reported_total",
                "ags_core_adjusted_total",
                "internal_srs_total",
                "internal_nibrs_total",
                "municipal_reported_total",
                "municipal_adjusted_total",
                "nonmunicipal_reported_total",
                "nonmunicipal_adjusted_total",
                "overlap_reported_total",
                "overlap_adjusted_total",
                "municipal_srs_total",
                "nonmunicipal_srs_total",
                "overlap_srs_total",
                "municipal_nibrs_total",
                "nonmunicipal_nibrs_total",
                "overlap_nibrs_total",
                "partial_reporting_pool",
                "partial_reporting_uplift_total",
                "current_year_fill_total",
                "identity_resolution_adjustment_total",
                "zero_month_pool_total",
                "municipal_partial_reporting_uplift_total",
                "municipal_current_year_fill_total",
                "nonmunicipal_partial_reporting_uplift_total",
                "nonmunicipal_current_year_fill_total",
                "overlap_partial_reporting_uplift_total",
                "overlap_current_year_fill_total",
                "municipal_identity_resolution_adjustment_total",
                "nonmunicipal_identity_resolution_adjustment_total",
                "overlap_identity_resolution_adjustment_total",
                "partial_pool_share_of_reported",
                "fbi_cde_estimated_total",
                "cde_gap_to_reported",
                "cde_gap_to_adjusted",
                "cde_gap_positive_only",
                "cde_gap_positive_only_adjusted",
                "reported_to_cde_ratio",
                "adjusted_to_cde_ratio",
                "gap_to_partial_pool_ratio",
                "benchmark_residual_pre_imputation",
                "benchmark_modeled_pool",
                "benchmark_modeled_pool_variance",
                "benchmark_relative_uncertainty",
                "benchmark_standard_error",
                "benchmark_variance",
                "benchmark_weight",
                "benchmark_imputed_total",
                "benchmark_posterior_total",
                "benchmark_standardized_disagreement",
                "benchmark_unused_headroom",
                "benchmark_unfilled_modeled_pool",
                "benchmark_silent_unit_count",
                "benchmark_silent_unit_population",
                "benchmark_conflict_kind",
                "cde_exact_sensitivity_factor",
            ]
        ]
        .sort_values(["state_fips", "offense"], kind="mergesort")
        .reset_index(drop=True)
    )


LEVEL_LANE_MASS_COLUMNS = [
    "ori9",
    "year",
    "offense",
    "jurisdiction_id",
    "jurisdiction_link_weight",
    "ownership_class",
    "level1_admission_status",
    "level2_semantic_status",
    "level_admission_reason",
    "level_repair_mode",
    "external_check_status",
    "admitted_input_mass",
    "accepted_observed_mass",
    "repair_mass",
    "benchmark_mass",
    "unresolved_mass",
    "invalid_fragment_audit_mass",
    "final_control_mass",
    "consumption_status",
]


def build_level_lane_mass_ledger(
    *,
    paths: RepoPaths,
    year: int,
    stage1: Stage1Consumption,
    benchmark_imputation: BenchmarkImputation,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical ownership ledger for every accepted, repaired, pooled, or held count.

    Agency mass is keyed by the real ORI and its jurisdiction crosswalk link. Pooled
    benchmark territory has no defensible agency allocation, so its ORI key is an
    explicit ``POOL::<unit_id>`` sentinel rather than an invented agency split.
    """
    _panel, estimates, _ct_audit = residualize_ctcsp_overlap(
        agency_panel=stage1.agency_panel,
        agency_estimates=stage1.agency_estimates,
        registry=load_ct_town_coverage_registry(paths, require_exists=False),
        target_year=int(year),
    )
    estimates = exclude_national_hq_rows(
        estimates, registry=load_federal_ori_scope_registry(paths)
    ).copy()
    crosswalk = load_crosswalk(paths)[["ori9", "jurisdiction_id", "weight"]].copy()
    for frame in (estimates, crosswalk):
        frame["ori9"] = frame["ori9"].astype("string").str.upper()
    estimates = estimates.merge(
        crosswalk, on="ori9", how="left", validate="many_to_many", indicator=True
    )
    positive_unmapped = estimates[
        estimates["_merge"].eq("left_only")
        & pd.to_numeric(estimates["estimated_count"], errors="coerce").fillna(0.0).gt(1e-9)
    ]
    if not positive_unmapped.empty:
        raise ValueError(
            "positive level-lane estimates have no jurisdiction link: "
            + str(positive_unmapped[["ori9", "offense", "estimated_count"]].head(20).to_dict(orient="records"))
        )
    estimates = estimates[estimates["_merge"].eq("both")].drop(columns="_merge")
    weight = pd.to_numeric(estimates["weight"], errors="coerce").fillna(0.0)
    agency = pd.DataFrame(
        {
            "ori9": estimates["ori9"],
            "year": int(year),
            "offense": estimates["offense"].astype("string"),
            "jurisdiction_id": estimates["jurisdiction_id"].astype("string"),
            "jurisdiction_link_weight": weight,
            "ownership_class": "agency_level",
            "level1_admission_status": estimates["level1_admission_status"].astype("string"),
            "level2_semantic_status": estimates["level2_semantic_status"].astype("string"),
            "level_admission_reason": estimates["level_admission_reason"].astype("string"),
            "level_repair_mode": estimates["level_repair_mode"].astype("string"),
            "external_check_status": estimates["external_check_status"].astype("string"),
            "admitted_input_mass": pd.to_numeric(estimates["reported_count_current"], errors="coerce").fillna(0.0) * weight,
            "accepted_observed_mass": pd.to_numeric(estimates["accepted_observed_mass"], errors="coerce").fillna(0.0) * weight,
            "repair_mass": pd.to_numeric(estimates["repair_mass"], errors="coerce").fillna(0.0) * weight,
            "benchmark_mass": 0.0,
            "unresolved_mass": pd.to_numeric(estimates["unresolved_mass"], errors="coerce").fillna(0.0) * weight,
            "invalid_fragment_audit_mass": pd.to_numeric(estimates["invalid_fragment_audit_mass"], errors="coerce").fillna(0.0) * weight,
        }
    )
    agency["final_control_mass"] = (
        agency["accepted_observed_mass"]
        + agency["repair_mass"]
        + agency["unresolved_mass"]
    )
    agency["consumption_status"] = "consumed"

    unit_rows: list[dict[str, object]] = []
    for row in benchmark_imputation.units.itertuples(index=False):
        jurisdiction_id = (
            str(row.unit_id)
            if str(row.unit_kind) == MUNICIPAL_UNIT_KIND
            else f"{str(row.state_fips).zfill(2)}{STATE_REMAINDER_SUFFIX}"
        )
        mass = float(max(0.0, pd.to_numeric(row.imputed_count, errors="coerce")))
        unit_rows.append(
            {
                "ori9": f"POOL::{row.unit_id}",
                "year": int(year),
                "offense": str(row.offense),
                "jurisdiction_id": jurisdiction_id,
                "jurisdiction_link_weight": 1.0,
                "ownership_class": "pooled_silent_unit",
                "level1_admission_status": "coverage_defective",
                "level2_semantic_status": "semantically_unusable",
                "level_admission_reason": "pooled_silent_territory",
                "level_repair_mode": "soft_benchmark_reconciliation",
                "external_check_status": "cde_soft_benchmark",
                "admitted_input_mass": 0.0,
                "accepted_observed_mass": 0.0,
                "repair_mass": 0.0,
                "benchmark_mass": mass,
                "unresolved_mass": 0.0,
                "invalid_fragment_audit_mass": 0.0,
                "final_control_mass": mass,
                "consumption_status": "consumed",
            }
        )
    pooled = pd.DataFrame(unit_rows, columns=LEVEL_LANE_MASS_COLUMNS)
    ledger = pd.concat([agency.reindex(columns=LEVEL_LANE_MASS_COLUMNS), pooled], ignore_index=True)
    assert_level_lane_mass_ledger(ledger=ledger, controls=controls)
    return ledger.sort_values(["jurisdiction_id", "offense", "ownership_class", "ori9"], kind="mergesort").reset_index(drop=True)


def assert_level_lane_mass_ledger(
    *, ledger: pd.DataFrame, controls: pd.DataFrame, tolerance: float = 1e-6
) -> None:
    components = ["accepted_observed_mass", "repair_mass", "benchmark_mass", "unresolved_mass"]
    numeric = ledger[
        ["admitted_input_mass", *components, "invalid_fragment_audit_mass", "final_control_mass"]
    ].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or (numeric < -tolerance).any().any():
        raise ValueError("level-lane mass ledger contains nonfinite or negative mass")
    residual = numeric["final_control_mass"] - numeric[components].sum(axis=1)
    if residual.abs().gt(tolerance).any():
        raise ValueError("level-lane mass ledger row components do not sum to final mass")
    invalid = ledger["level1_admission_status"].isin({"coverage_defective", "source_identity_failure"}) | ledger[
        "level2_semantic_status"
    ].isin({"channel_omitted", "wrong_category_mapping", "documented_swap", "semantically_unusable"})
    wrongly_accepted = invalid & numeric["accepted_observed_mass"].gt(tolerance)
    # Pooled sentinel rows have no rejected fragment and are outside the agency test.
    wrongly_accepted &= ledger["ownership_class"].eq("agency_level")
    if wrongly_accepted.any():
        raise ValueError("invalid agency fragments entered accepted observed mass")
    structural = ledger["level1_admission_status"].eq("corroborated_structural_zero")
    if numeric.loc[structural, "final_control_mass"].abs().gt(tolerance).any():
        raise ValueError("corroborated structural zero was not locked to zero")
    if not ledger["consumption_status"].isin({"consumed", "explicitly_refused"}).all():
        raise ValueError("level-lane ledger contains an unconsumed repair")

    # One-path ownership shapes. Held agency rows equal the admitted input after
    # jurisdiction-link weighting. Benchmark units are separate sentinel rows;
    # they cannot inherit or hide an agency disposition.
    agency = ledger["ownership_class"].eq("agency_level")
    held = agency & ledger["level_repair_mode"].eq("unchanged_review_hold")
    held_bad = held & (
        (numeric["final_control_mass"] - numeric["admitted_input_mass"]).abs().gt(tolerance)
        | (numeric["unresolved_mass"] - numeric["admitted_input_mass"]).abs().gt(tolerance)
        | numeric[["accepted_observed_mass", "repair_mass", "benchmark_mass"]]
        .abs()
        .gt(tolerance)
        .any(axis=1)
    )
    if held_bad.any():
        raise ValueError(
            "unchanged review hold changed admitted input or shares ownership "
            "with a repair/benchmark component"
        )
    pooled = ledger["ownership_class"].eq("pooled_silent_unit")
    pooled_bad = pooled & (
        ~ledger["level_repair_mode"].eq("soft_benchmark_reconciliation")
        | (numeric["final_control_mass"] - numeric["benchmark_mass"]).abs().gt(tolerance)
        | numeric[
            ["admitted_input_mass", "accepted_observed_mass", "repair_mass", "unresolved_mass"]
        ]
        .abs()
        .gt(tolerance)
        .any(axis=1)
    )
    if pooled_bad.any():
        raise ValueError("pooled silent unit has more than the benchmark sizing owner")

    actual = (
        ledger.groupby(["jurisdiction_id", "offense"], dropna=False, as_index=False)["final_control_mass"]
        .sum()
    )
    expected = controls[["jurisdiction_id", "offense", "adjusted_count_ags_core"]].copy()
    check = expected.merge(actual, on=["jurisdiction_id", "offense"], how="outer")
    delta = pd.to_numeric(check["adjusted_count_ags_core"], errors="coerce").fillna(0.0) - pd.to_numeric(
        check["final_control_mass"], errors="coerce"
    ).fillna(0.0)
    bad = check[delta.abs().gt(tolerance)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} jurisdiction-offense controls disagree with the level-lane mass ledger: "
            + str(bad.head(20).to_dict(orient="records"))
        )


def attach_level_lane_disclosures(
    *, controls: pd.DataFrame, ledger: pd.DataFrame, benchmark_imputation: BenchmarkImputation
) -> pd.DataFrame:
    out = controls.drop(
        columns=[
            "level_admission_status", "level_semantic_status", "level_admission_reason",
            "level_repair_mode", "level_repair_share", "level_external_check_status",
            "level_unresolved_review", "level_benchmark_weight", "external_check_status",
            "benchmark_conflict_kind", "benchmark_weight", "unresolved_level_flag",
        ],
        errors="ignore",
    ).copy()
    work = ledger.copy()
    work["_sizing_mass"] = (
        pd.to_numeric(work["repair_mass"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["benchmark_mass"], errors="coerce").fillna(0.0)
    )
    work["_has_sizing"] = work["_sizing_mass"].gt(1e-12)
    work["_dominance_mass"] = pd.to_numeric(work["final_control_mass"], errors="coerce").fillna(0.0)
    work = work.sort_values(
        [
            "jurisdiction_id",
            "offense",
            "_has_sizing",
            "_sizing_mass",
            "_dominance_mass",
            "ownership_class",
            "ori9",
        ],
        ascending=[True, True, False, False, False, True, True],
        kind="mergesort",
    )
    dominant = work.drop_duplicates(["jurisdiction_id", "offense"], keep="first")[
        [
            "jurisdiction_id", "offense", "level1_admission_status",
            "level2_semantic_status", "level_admission_reason", "level_repair_mode",
            "external_check_status",
        ]
    ].rename(
        columns={
            "level1_admission_status": "level_admission_status",
            "level2_semantic_status": "level_semantic_status",
            "external_check_status": "level_external_check_status",
        }
    )
    sums = work.groupby(["jurisdiction_id", "offense"], dropna=False, as_index=False).agg(
        _level_total=("final_control_mass", "sum"),
        _level_repair=("repair_mass", "sum"),
        _level_benchmark=("benchmark_mass", "sum"),
        _level_unresolved=("unresolved_mass", "sum"),
        _level_admitted_input=("admitted_input_mass", "sum"),
    )
    sums["level_repair_share"] = np.divide(
        sums["_level_repair"] + sums["_level_benchmark"],
        sums["_level_total"].replace(0.0, np.nan),
    ).fillna(0.0)
    sums["level_unresolved_review"] = sums["_level_unresolved"].gt(1e-9)
    sums["_level_composite_sizing"] = sums["_level_repair"].gt(1e-9) & sums[
        "_level_benchmark"
    ].gt(1e-9)
    disclosure = dominant.merge(
        sums[
            [
                "jurisdiction_id",
                "offense",
                "level_repair_share",
                "level_unresolved_review",
                "_level_composite_sizing",
            ]
        ],
        on=["jurisdiction_id", "offense"], how="left",
    )
    if benchmark_imputation.state_identity.empty:
        disclosure["level_benchmark_weight"] = 0.0
        disclosure["benchmark_conflict_kind"] = "not_evaluated"
    else:
        state_weight = benchmark_imputation.state_identity[
            ["state_fips", "offense", "benchmark_weight", "conflict_kind"]
        ].rename(
            columns={
                "benchmark_weight": "level_benchmark_weight",
                "conflict_kind": "benchmark_conflict_kind",
            }
        )
        state_for_jurisdiction = out[["jurisdiction_id", "state_fips"]].drop_duplicates("jurisdiction_id")
        disclosure = disclosure.merge(state_for_jurisdiction, on="jurisdiction_id", how="left")
        disclosure = disclosure.merge(state_weight, on=["state_fips", "offense"], how="left").drop(columns="state_fips")
    out = out.merge(disclosure, on=["jurisdiction_id", "offense"], how="left")
    out.loc[
        out["_level_composite_sizing"].astype("boolean").fillna(False).astype(bool),
        "level_repair_mode",
    ] = "reason_coded_level_repair_plus_soft_benchmark"
    out = out.drop(columns="_level_composite_sizing")
    out["level_benchmark_weight"] = pd.to_numeric(out["level_benchmark_weight"], errors="coerce").fillna(0.0)
    out["level_unresolved_review"] = (
        out["level_unresolved_review"].astype("boolean").fillna(False).astype(bool)
    )
    out["level_repair_share"] = pd.to_numeric(out["level_repair_share"], errors="coerce").fillna(0.0)
    # Zero-mass skeleton rows have no ledger owner. Keep their disclosure
    # explicit: no sizing path ran, rather than leaking null provenance.
    no_owner = out["level_repair_mode"].isna()
    out.loc[no_owner, "level_admission_status"] = "coverage_defective"
    out.loc[no_owner, "level_semantic_status"] = "semantically_unusable"
    out.loc[no_owner, "level_admission_reason"] = "no_admitted_agency_evidence"
    out.loc[no_owner, "level_repair_mode"] = "none"
    out.loc[no_owner, "level_external_check_status"] = "unavailable"
    out["external_check_status"] = out["level_external_check_status"].astype("string")
    out["unresolved_level_flag"] = out["level_unresolved_review"].astype(bool)
    out["benchmark_weight"] = out["level_benchmark_weight"]
    out["benchmark_conflict_kind"] = out["benchmark_conflict_kind"].astype("string").fillna("not_evaluated")
    held_controls = out["level_repair_mode"].astype("string").eq("unchanged_review_hold")
    if held_controls.any():
        held_check = out.loc[
            held_controls, ["jurisdiction_id", "offense", "adjusted_count_ags_core"]
        ].merge(
            sums[
                [
                    "jurisdiction_id",
                    "offense",
                    "_level_admitted_input",
                    "_level_repair",
                    "_level_benchmark",
                ]
            ],
            on=["jurisdiction_id", "offense"],
            how="left",
            validate="one_to_one",
        )
        held_bad = (
            (
                pd.to_numeric(held_check["adjusted_count_ags_core"], errors="coerce")
                - pd.to_numeric(held_check["_level_admitted_input"], errors="coerce")
            ).abs().gt(1e-6)
            | pd.to_numeric(held_check["_level_repair"], errors="coerce").abs().gt(1e-6)
            | pd.to_numeric(held_check["_level_benchmark"], errors="coerce").abs().gt(1e-6)
        )
        if held_bad.any():
            raise ValueError(
                "control disclosed as unchanged_review_hold differs from its admitted "
                "post-link input before benchmark reconciliation: "
                + str(held_check.loc[held_bad].head(20).to_dict(orient="records"))
            )
    return out


def build_controls_bundle(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig = ControlBuildConfig(),
    imputation_config: BenchmarkImputationConfig | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    BenchmarkImputation,
    pd.DataFrame,
]:
    """The whole stage in one order, so nothing can be assembled out of sequence.

    skeleton -> aggregation -> pre-imputation controls -> benchmark eligibility and
    sizing -> imputation landed -> state reconciliation.
    """
    target_config = _jurisdiction_target_config(config)
    stage1 = build_stage1_consumption(paths=paths, config=config)
    ownership = build_jurisdiction_ownership(paths=paths, config=target_config)
    exclusions = build_ownership_exclusions(
        ownership=ownership,
        agency_estimates=stage1.agency_estimates,
        crosswalk=load_crosswalk(paths),
    )
    jurisdiction_year_estimates = build_jurisdiction_year_estimates(
        paths=paths,
        config=target_config,
        agency_panel=stage1.agency_panel,
        agency_estimates=stage1.agency_estimates,
        ownership=ownership,
        succession_ledger=stage1.succession_ledger,
    )
    controls = build_jurisdiction_controls(
        paths=paths,
        config=config,
        jurisdiction_year_estimates=jurisdiction_year_estimates,
    )
    benchmark_imputation = build_controls_benchmark_imputation(
        paths=paths,
        config=config,
        controls=controls,
        stage1=stage1,
        imputation_config=imputation_config,
    )
    controls = apply_benchmark_imputation_to_controls(
        controls, units=benchmark_imputation.units
    )
    _assert_row_identity(controls)
    mass_ledger = build_level_lane_mass_ledger(
        paths=paths,
        year=int(config.year),
        stage1=stage1,
        benchmark_imputation=benchmark_imputation,
        controls=controls,
    )
    controls = attach_level_lane_disclosures(
        controls=controls,
        ledger=mass_ledger,
        benchmark_imputation=benchmark_imputation,
    )
    benchmark_imputation = replace(benchmark_imputation, mass_ledger=mass_ledger)
    state_controls = build_state_control_comparison(
        paths=paths,
        config=config,
        controls=controls,
        benchmark_imputation=benchmark_imputation,
    )
    return (
        controls,
        state_controls,
        jurisdiction_year_estimates,
        exclusions,
        benchmark_imputation,
        stage1.agency_estimates,
    )


def write_v2_controls(
    *,
    paths: RepoPaths,
    state_out_path: Path,
    jurisdiction_out_path: Path,
    jurisdiction_year_estimates_out_path: Path | None = None,
    config: ControlBuildConfig = ControlBuildConfig(),
    blocked_by: tuple[str, ...] | None = None,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> tuple[Path, Path, Path | None]:
    with stage_write_lock(paths=paths, stage="controls", blocked_by=blocked_by):
        _ensure_controls_dependencies(
            paths=paths,
            config=config,
            observation_ignore_blockers=observation_ignore_blockers,
        )
        (
            jurisdiction_controls,
            state_controls,
            jurisdiction_year_estimates,
            exclusions,
            benchmark_imputation,
            agency_estimates,
        ) = build_controls_bundle(paths=paths, config=config)
        write_benchmark_imputation_artifacts(
            benchmark_imputation, paths=paths, year=int(config.year)
        )
        mass_ledger_path = (
            paths.state_dir / "controls" / f"level_lane_mass_ledger_{int(config.year)}.parquet"
        )
        benchmark_imputation.mass_ledger.to_parquet(mass_ledger_path, index=False)
        write_jurisdiction_ownership_exclusions(
            exclusions, paths=paths, year=int(config.year)
        )

        jurisdiction_out_path.parent.mkdir(parents=True, exist_ok=True)
        state_out_path.parent.mkdir(parents=True, exist_ok=True)
        if jurisdiction_year_estimates_out_path is not None:
            jurisdiction_year_estimates_out_path.parent.mkdir(
                parents=True, exist_ok=True
            )
        jurisdiction_controls.to_parquet(jurisdiction_out_path, index=False)
        state_controls.to_parquet(state_out_path, index=False)
        if jurisdiction_year_estimates_out_path is not None:
            jurisdiction_year_estimates.to_parquet(
                jurisdiction_year_estimates_out_path, index=False
            )
        dependencies = _controls_dependency_paths(paths, year=int(config.year))
        write_dependency_stamp(jurisdiction_out_path, dependencies)
        write_dependency_stamp(state_out_path, dependencies)
        if jurisdiction_year_estimates_out_path is not None:
            write_dependency_stamp(jurisdiction_year_estimates_out_path, dependencies)
        # Allocation must consume the exact post-adjudication, post-masked-gap table
        # used by these controls. Publish it only after every control artifact and
        # dependency stamp is durable, so the downstream output stage can reuse one
        # authoritative agency target table without reconstructing it.
        from crimerisk.allocation import persist_agency_allocation_target_estimates_cache

        persist_agency_allocation_target_estimates_cache(
            paths=paths,
            year=int(config.year),
            agency_estimates=agency_estimates,
        )
        return (
            state_out_path,
            jurisdiction_out_path,
            jurisdiction_year_estimates_out_path,
        )
