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

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from crimerisk.benchmark_imputation import (
    BenchmarkImputation,
    BenchmarkImputationConfig,
    apply_benchmark_imputation_to_controls,
    build_benchmark_imputation,
    write_benchmark_imputation_artifacts,
)
from crimerisk.build_freshness import artifact_is_current
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
from crimerisk.observations import (
    ObservationBuildConfig,
    get_v2_observation_paths,
    observations_artifacts_are_current,
    write_v2_observations,
)
from crimerisk.paths import RepoPaths
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
    build_agency_allocation_target_estimates,
    build_agency_trend_fill_panel,
    resolve_ori_succession,
    load_agency_jurisdiction_crosswalk,
)
from crimerisk.scope import PRODUCTION_SCOPE_EXCLUDE


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
    "estimate_source",
    "estimate_confidence",
    "estimated_from_panel",
]


@dataclass(frozen=True)
class ControlBuildConfig:
    year: int = 2024
    exclude_scope_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE))
    force_reporting_regimes_rebuild: bool = False


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
    # What the Stage-1 adjudication registries did on this build, for the controls summary.
    stage1_adjudication_counts: dict = field(default_factory=dict)


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
        succession_ledger=succession_ledger,
    )
    consumption = Stage1Consumption(
        agency_panel=agency_panel,
        agency_preferred_target_year=build_agency_target_panel_slice(
            agency_panel=agency_panel, target_year=year
        ),
        agency_estimates=agency_estimates,
        succession_ledger=succession_ledger,
    )
    consumption.stage1_adjudication_counts.update(succession_summary)
    consumption.stage1_adjudication_counts.update(
        agency_estimates.attrs.get("stage1_adjudications", {})
    )
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
        config=imputation_config or BenchmarkImputationConfig(year=int(config.year)),
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
                "imputed_total",
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
                "imputed_total": "benchmark_imputed_total",
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
            ("benchmark_imputed_total", 0.0),
            ("benchmark_unused_headroom", 0.0),
            ("benchmark_unfilled_modeled_pool", 0.0),
            ("benchmark_silent_unit_count", 0),
            ("benchmark_silent_unit_population", 0.0),
        ):
            out[column] = default
        out["benchmark_conflict_kind"] = "not_evaluated"
    out["year"] = int(config.year)

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
                "benchmark_imputed_total",
                "benchmark_unused_headroom",
                "benchmark_unfilled_modeled_pool",
                "benchmark_silent_unit_count",
                "benchmark_silent_unit_population",
                "benchmark_conflict_kind",
            ]
        ]
        .sort_values(["state_fips", "offense"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_controls_bundle(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig = ControlBuildConfig(),
    imputation_config: BenchmarkImputationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, BenchmarkImputation]:
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
        ) = build_controls_bundle(paths=paths, config=config)
        write_benchmark_imputation_artifacts(
            benchmark_imputation, paths=paths, year=int(config.year)
        )
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
        return (
            state_out_path,
            jurisdiction_out_path,
            jurisdiction_year_estimates_out_path,
        )
