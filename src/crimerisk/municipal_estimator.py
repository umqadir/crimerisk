from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.build_freshness import artifact_is_current
from crimerisk.crime import OFFENSES_7
from crimerisk.crime.municipal_totals import _assign_pop_band, _project_count_log_linear, MunicipalTotalsConfig
from crimerisk.observations import (
    ObservationBuildConfig,
    get_v2_observation_paths,
    observations_artifacts_are_current,
    write_v2_observations,
)
from crimerisk.paths import RepoPaths
from crimerisk.stage_locks import blockers_for_stage, stage_write_lock
from crimerisk.panel_guardrails import suppress_extreme_summary_spikes
from crimerisk.reporting_regimes import (
    ReportingRegimeBuildConfig,
    get_v2_reporting_regimes_path,
    reporting_regime_dependency_paths,
    reporting_regimes_artifact_is_current,
    write_v2_reporting_regimes,
)
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    assign_preferred_value,
    build_prefer_nibrs_mask,
    initialize_preferred_source,
)
from crimerisk.trend_fills import (
    MASKED_GAP_LADDER_KINDS,
    TrendFillAudit,
    build_agency_trend_fill_panel,
    build_masked_gap_flags,
    build_reference_year_masked_gap_years,
    build_trend_fill_lookup,
    scaled_history_median,
)


PRODUCTION_SCOPE_EXCLUDE = {"AK", "AS", "CZ", "GU", "HI", "MP", "PR", "VI"}


@dataclass(frozen=True)
class MunicipalEstimatorConfig:
    year_start: int = 2018
    year_end: int = 2024
    target_year: int = 2024
    min_usable_years_for_trend: int = 3
    max_year_gap_for_trend: int = 1
    state_trend_clip_low: float = 0.75
    state_trend_clip_high: float = 1.25
    exclude_scope_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE))
    pop_bands: tuple[tuple[float, float, str], ...] = MunicipalTotalsConfig().pop_bands
    force_reporting_regimes_rebuild: bool = False


def _load_jurisdiction_master(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "reference" / "jurisdiction_master.parquet"
    return pd.read_parquet(path)


def _load_jurisdiction_year_observations(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "observations" / "jurisdiction_year_observations.parquet"
    return pd.read_parquet(path)


def _load_crosswalk(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    crosswalk = pd.read_parquet(path).rename(columns={"ori": "ori9"})
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0)
    return crosswalk


def _load_reporting_regimes(
    paths: RepoPaths,
    *,
    config: MunicipalEstimatorConfig,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> pd.DataFrame:
    path = get_v2_reporting_regimes_path(paths)
    reporting_config = ReportingRegimeBuildConfig(year_start=config.year_start, year_end=config.year_end)
    if (
        not config.force_reporting_regimes_rebuild
        and reporting_regimes_artifact_is_current(paths, config=reporting_config, out_path=path)
    ):
        regimes = pd.read_parquet(path)
    else:
        write_v2_reporting_regimes(
            paths=paths,
            out_path=path,
            config=reporting_config,
            observation_ignore_blockers=observation_ignore_blockers,
        )
        regimes = pd.read_parquet(path)
    regimes = regimes[(regimes["year"].astype(int) >= config.year_start) & (regimes["year"].astype(int) <= config.year_end)].copy()
    return regimes


def get_v2_municipal_estimates_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "modeling" / f"municipal_estimates_{int(year)}.parquet"


def _observation_build_config(config: MunicipalEstimatorConfig) -> ObservationBuildConfig:
    return ObservationBuildConfig(
        year_start=int(config.year_start),
        year_end=int(config.year_end),
    )


def _ensure_municipal_estimate_dependencies(
    *,
    paths: RepoPaths,
    config: MunicipalEstimatorConfig,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> None:
    agency_out_path, jurisdiction_out_path = get_v2_observation_paths(paths)
    observation_config = _observation_build_config(config)
    if observations_artifacts_are_current(
        paths,
        config=observation_config,
        agency_out_path=agency_out_path,
        jurisdiction_out_path=jurisdiction_out_path,
    ):
        return
    write_v2_observations(
        paths=paths,
        agency_out_path=agency_out_path,
        jurisdiction_out_path=jurisdiction_out_path,
        config=observation_config,
        blocked_by=blockers_for_stage(
            "observations",
            ignore=("municipal_estimates", *observation_ignore_blockers),
        ),
        reference_ignore_blockers=("municipal_estimates", *observation_ignore_blockers),
    )
    reporting_path = get_v2_reporting_regimes_path(paths)
    reporting_config = ReportingRegimeBuildConfig(year_start=config.year_start, year_end=config.year_end)
    if not reporting_regimes_artifact_is_current(paths, config=reporting_config, out_path=reporting_path):
        write_v2_reporting_regimes(
            paths=paths,
            out_path=reporting_path,
            config=reporting_config,
            blocked_by=blockers_for_stage("reporting_regimes", ignore=("municipal_estimates",)),
            observation_ignore_blockers=("municipal_estimates", *observation_ignore_blockers),
        )


def municipal_estimate_dependency_paths(
    paths: RepoPaths,
    *,
    config: MunicipalEstimatorConfig,
) -> list[Path]:
    reporting_config = ReportingRegimeBuildConfig(
        year_start=int(config.year_start),
        year_end=int(config.year_end),
    )
    return [
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        paths.state_dir / "observations" / "jurisdiction_year_observations.parquet",
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        get_v2_reporting_regimes_path(paths),
        Path(__file__),
        paths.repo_root / "src" / "crimerisk" / "panel_guardrails.py",
        paths.repo_root / "src" / "crimerisk" / "source_provenance.py",
        paths.repo_root / "src" / "crimerisk" / "trend_fills.py",
        *reporting_regime_dependency_paths(paths, config=reporting_config),
    ]


def municipal_estimates_artifact_is_current(
    paths: RepoPaths,
    *,
    config: MunicipalEstimatorConfig,
    out_path: Path | None = None,
) -> bool:
    artifact_path = out_path or get_v2_municipal_estimates_path(paths, year=config.target_year)
    return artifact_is_current(
        artifact_path,
        municipal_estimate_dependency_paths(paths, config=config),
    )


def _summarize_jurisdiction_reporting_regimes(
    *,
    reporting_regimes: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    merged = reporting_regimes.merge(crosswalk[["ori9", "jurisdiction_id", "weight"]], on="ori9", how="inner")
    merged["allocated_support"] = (
        pd.to_numeric(merged["support_weight"], errors="coerce").fillna(1.0)
        * pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    )

    support_by_regime = (
        merged.groupby(["jurisdiction_id", "year", "offense", "reporting_regime"], dropna=False)["allocated_support"]
        .sum()
        .reset_index()
    )
    dominant_regime = (
        support_by_regime.sort_values(
            ["jurisdiction_id", "year", "offense", "allocated_support"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=["jurisdiction_id", "year", "offense"], keep="first")
        .rename(columns={"reporting_regime": "dominant_reporting_regime"})[
            ["jurisdiction_id", "year", "offense", "dominant_reporting_regime"]
        ]
    )

    support_by_pref = (
        merged.groupby(
            ["jurisdiction_id", "year", "offense", "preferred_source_by_regime"], dropna=False
        )["allocated_support"]
        .sum()
        .reset_index()
    )
    dominant_pref = (
        support_by_pref.sort_values(
            ["jurisdiction_id", "year", "offense", "allocated_support"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=["jurisdiction_id", "year", "offense"], keep="first")
        .rename(columns={"preferred_source_by_regime": "dominant_preferred_source_by_regime"})[
            ["jurisdiction_id", "year", "offense", "dominant_preferred_source_by_regime"]
        ]
    )

    return dominant_regime.merge(dominant_pref, on=["jurisdiction_id", "year", "offense"], how="left")


def _build_regime_aware_panel(
    *,
    paths: RepoPaths,
    config: MunicipalEstimatorConfig,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> pd.DataFrame:
    obs = _load_jurisdiction_year_observations(paths)
    obs["state_abbr"] = obs["state_abbr"].astype("string").str.upper()
    obs = obs[
        obs["jurisdiction_type"].eq("municipal")
        & obs["year"].between(config.year_start, config.year_end)
        & ~obs["state_abbr"].isin(set(config.exclude_scope_state_abbrs))
    ].copy()

    base_cols = [
        "jurisdiction_id",
        "jurisdiction_type",
        "jurisdiction_name",
        "state_fips",
        "state_abbr",
        "geo_type",
        "geoid",
        "year",
        "offense",
    ]
    local_publication = (
        obs[obs["source"].eq(LOCAL_PUBLICATION_SOURCE)][base_cols + ["observed_count", "mean_months_reported"]]
        .rename(
            columns={
                "observed_count": "reported_count_local_publication",
                "mean_months_reported": "mean_months_reported_local_publication",
            }
        )
    )
    cius = (
        obs[obs["source"].eq(CIUS_SOURCE)][base_cols + ["observed_count", "mean_months_reported"]]
        .rename(columns={"observed_count": "reported_count_cius", "mean_months_reported": "mean_months_reported_cius"})
    )
    state_pub = (
        obs[obs["source"].eq(STATE_PUBLICATION_SOURCE)][base_cols + ["observed_count", "mean_months_reported"]]
        .rename(
            columns={
                "observed_count": "reported_count_state_publication",
                "mean_months_reported": "mean_months_reported_state_publication",
            }
        )
    )
    srs = (
        obs[obs["source"].eq(SUMMARY_SOURCE)][base_cols + ["observed_count", "mean_months_reported"]]
        .rename(columns={"observed_count": "reported_count_srs", "mean_months_reported": "mean_months_reported_srs"})
    )
    nibrs = (
        obs[obs["source"].eq(NIBRS_SOURCE)][base_cols + ["observed_count", "mean_months_reported"]]
        .rename(columns={"observed_count": "reported_count_nibrs", "mean_months_reported": "mean_months_reported_nibrs"})
    )
    panel = (
        cius.merge(local_publication, on=base_cols, how="outer")
        .merge(state_pub, on=base_cols, how="outer")
        .merge(srs, on=base_cols, how="outer")
        .merge(nibrs, on=base_cols, how="outer")
    )

    crosswalk = _load_crosswalk(paths)
    regimes = _load_reporting_regimes(
        paths,
        config=config,
        observation_ignore_blockers=("municipal_estimates", *observation_ignore_blockers),
    )
    regime_summary = _summarize_jurisdiction_reporting_regimes(reporting_regimes=regimes, crosswalk=crosswalk)
    panel = panel.merge(regime_summary, on=["jurisdiction_id", "year", "offense"], how="left")

    has_cius = panel["reported_count_cius"].notna()
    has_local_publication = panel["reported_count_local_publication"].notna()
    has_state_publication = panel["reported_count_state_publication"].notna()
    has_srs = panel["reported_count_srs"].notna()
    has_nibrs = panel["reported_count_nibrs"].notna()
    regime_prefers_nibrs = panel["dominant_preferred_source_by_regime"].eq(NIBRS_SOURCE)
    srs_regime_inferior = panel["dominant_reporting_regime"].isin(
        ["structurally_missing_or_unreliable", "lumpy_or_batched", "annual_only_but_usable"]
    )
    srs_months = pd.to_numeric(panel["mean_months_reported_srs"], errors="coerce").fillna(0.0)
    nibrs_months = pd.to_numeric(panel["mean_months_reported_nibrs"], errors="coerce").fillna(0.0)
    srs_count_num = pd.to_numeric(panel["reported_count_srs"], errors="coerce").fillna(0.0)
    nibrs_supports_better = nibrs_months.gt(srs_months)
    prefer_nibrs = build_prefer_nibrs_mask(
        has_cius=has_cius,
        has_local_publication=has_local_publication,
        has_state_publication=has_state_publication,
        has_srs=has_srs,
        has_nibrs=has_nibrs,
        regime_prefers_nibrs=regime_prefers_nibrs,
        srs_regime_inferior=srs_regime_inferior,
        nibrs_supports_better=nibrs_supports_better,
        srs_count_num=srs_count_num,
        nibrs_months=nibrs_months,
    )
    panel["preferred_source"] = initialize_preferred_source(
        has_cius=has_cius,
        has_local_publication=has_local_publication,
        has_state_publication=has_state_publication,
        has_srs=has_srs,
        has_nibrs=has_nibrs,
        prefer_nibrs_mask=prefer_nibrs,
    )
    panel = assign_preferred_value(
        panel,
        output_col="reported_count_preferred",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "reported_count_cius",
            LOCAL_PUBLICATION_SOURCE: "reported_count_local_publication",
            STATE_PUBLICATION_SOURCE: "reported_count_state_publication",
            SUMMARY_SOURCE: "reported_count_srs",
            NIBRS_SOURCE: "reported_count_nibrs",
        },
    )
    panel = assign_preferred_value(
        panel,
        output_col="mean_months_reported_preferred",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "mean_months_reported_cius",
            LOCAL_PUBLICATION_SOURCE: "mean_months_reported_local_publication",
            STATE_PUBLICATION_SOURCE: "mean_months_reported_state_publication",
            SUMMARY_SOURCE: "mean_months_reported_srs",
            NIBRS_SOURCE: "mean_months_reported_nibrs",
        },
    )
    regime = panel["dominant_reporting_regime"].astype("string")
    preferred_source = panel["preferred_source"].astype("string")
    counts_present = panel["reported_count_preferred"].notna()
    panel["usable_as_observed"] = counts_present & (
        preferred_source.eq(CIUS_SOURCE)
        | (
            preferred_source.eq(LOCAL_PUBLICATION_SOURCE)
            & regime.eq("annual_only_but_usable")
        )
        | (
            preferred_source.eq(STATE_PUBLICATION_SOURCE)
            & regime.eq("annual_only_but_usable")
        )
        | (
            preferred_source.eq(SUMMARY_SOURCE)
            & regime.isin(["full_monthly", "lumpy_or_batched", "annual_only_but_usable"])
        )
        | (
            preferred_source.eq(NIBRS_SOURCE)
            & ~regime.eq("structurally_missing_or_unreliable")
        )
    )
    panel["reported_count_preferred"] = pd.to_numeric(panel["reported_count_preferred"], errors="coerce")
    return panel


def _load_population_from_geometry(paths: RepoPaths) -> pd.DataFrame:
    geometry_dir = paths.state_dir / "geometry"
    block_path = geometry_dir / "block_to_jurisdiction_crosswalk.parquet"
    if block_path.exists():
        blocks = pd.read_parquet(block_path, columns=["jurisdiction_id", "jurisdiction_type", "pop20"])
    else:
        state_dir = geometry_dir / "blocks_by_state"
        state_files = sorted(state_dir.glob("*.parquet"))
        if not state_files:
            raise FileNotFoundError(block_path)
        frames = [
            pd.read_parquet(state_file, columns=["jurisdiction_id", "jurisdiction_type", "pop20"])
            for state_file in state_files
        ]
        blocks = pd.concat(frames, ignore_index=True)
    blocks = blocks[blocks["jurisdiction_type"].eq("municipal")].copy()
    pop = (
        blocks.groupby("jurisdiction_id", dropna=False)["pop20"]
        .sum()
        .reset_index()
        .rename(columns={"pop20": "bucket_population"})
    )
    pop["bucket_population"] = pd.to_numeric(pop["bucket_population"], errors="coerce").fillna(0.0)
    return pop


def _build_municipal_population(
    *,
    paths: RepoPaths,
    config: MunicipalEstimatorConfig,
) -> pd.DataFrame:
    juris = _load_jurisdiction_master(paths)
    juris["state_abbr"] = juris["state_abbr"].astype("string").str.upper()
    muni = juris[
        juris["jurisdiction_type"].eq("municipal")
        & ~juris["state_abbr"].isin(set(config.exclude_scope_state_abbrs))
    ][["jurisdiction_id", "state_fips", "state_abbr", "jurisdiction_name", "geo_type", "geoid"]].copy()
    pop = _load_population_from_geometry(paths)
    muni = muni.merge(pop, on="jurisdiction_id", how="left")
    muni["bucket_population"] = pd.to_numeric(muni["bucket_population"], errors="coerce").fillna(0.0)
    muni["pop_band"] = muni["bucket_population"].map(lambda v: _assign_pop_band(v, config.pop_bands))
    muni["population_basis"] = "tabblock20_pop20"
    return muni


def _build_state_trend_map(panel: pd.DataFrame, *, config: MunicipalEstimatorConfig) -> dict[tuple[str, str], float]:
    paired = panel[
        panel["usable_as_observed"] & panel["year"].isin([config.target_year - 1, config.target_year])
    ][["jurisdiction_id", "state_fips", "offense", "year", "reported_count_preferred"]].copy()
    if paired.empty:
        return {}

    pivot = (
        paired.pivot_table(
            index=["jurisdiction_id", "state_fips", "offense"],
            columns="year",
            values="reported_count_preferred",
            aggfunc="sum",
        )
        .reset_index()
    )
    prev_col = config.target_year - 1
    curr_col = config.target_year
    if prev_col not in pivot.columns or curr_col not in pivot.columns:
        return {}
    pivot = pivot.dropna(subset=[prev_col, curr_col]).copy()
    if pivot.empty:
        return {}

    state_trend = (
        pivot.groupby(["state_fips", "offense"], dropna=False)
        .agg(total_prev=(prev_col, "sum"), total_curr=(curr_col, "sum"))
        .reset_index()
    )
    state_trend["trend_ratio"] = (
        state_trend["total_curr"] / state_trend["total_prev"]
    ).clip(lower=config.state_trend_clip_low, upper=config.state_trend_clip_high)
    return {
        (str(row.state_fips).zfill(2), str(row.offense)): float(row.trend_ratio)
        for row in state_trend.itertuples(index=False)
        if pd.notna(row.trend_ratio) and np.isfinite(row.trend_ratio)
    }


def _map_masked_gaps_to_jurisdictions(
    *,
    paths: RepoPaths,
    masked_gap_flags: pd.DataFrame,
) -> dict[tuple[str, str], tuple[str, float]]:
    """Map agency-level masked-gap flags (trend_fills.build_masked_gap_flags) to the
    jurisdictions they dominate through the crosswalk (dominant links only, weight >
    0.5). Returns {(jurisdiction_id, offense): (kind, effective_months)}; when multiple
    flagged agencies map to one jurisdiction-offense, ladder-class kinds
    (collapsed_count / partial_months_escalated) win over partial_months (stronger
    signal) and partial effective months take the minimum.
    """
    if masked_gap_flags is None or masked_gap_flags.empty:
        return {}
    crosswalk = _load_crosswalk(paths)
    links = crosswalk[
        pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0).gt(0.5)
    ][["ori9", "jurisdiction_id"]].copy()
    merged = masked_gap_flags.merge(links, on="ori9", how="inner")
    out: dict[tuple[str, str], tuple[str, float]] = {}
    for row in merged.itertuples(index=False):
        key = (str(row.jurisdiction_id), str(row.offense))
        kind = str(row.masked_gap_kind)
        months_value = pd.to_numeric(row.masked_gap_effective_months, errors="coerce")
        months = float(months_value) if pd.notna(months_value) else float("nan")
        prev = out.get(key)
        if prev is None:
            out[key] = (kind, months)
            continue
        prev_kind, prev_months = prev
        if prev_kind in MASKED_GAP_LADDER_KINDS or kind in MASKED_GAP_LADDER_KINDS:
            out[key] = (
                prev_kind if prev_kind in MASKED_GAP_LADDER_KINDS else kind,
                float("nan"),
            )
        else:
            candidates = [m for m in (prev_months, months) if np.isfinite(m)]
            out[key] = ("partial_months", min(candidates) if candidates else float("nan"))
    return out


def _map_reference_year_exclusions_to_jurisdictions(
    *,
    paths: RepoPaths,
    reference_year_exclusions: pd.DataFrame,
) -> dict[tuple[str, str], set[int]]:
    """Map agency-level reference-year exclusions (trend_fills.
    build_reference_year_masked_gap_years) to the jurisdictions they dominate through
    the crosswalk (dominant links only, weight > 0.5) -- the same link rule
    _map_masked_gaps_to_jurisdictions uses for the target-year masked-gap flags. Returns
    {(jurisdiction_id, offense): {excluded years}}: if a dominant agency's own history at
    year Y is itself a masked gap, the jurisdiction-level aggregate for year Y is not a
    trustworthy fill reference either.
    """
    if reference_year_exclusions is None or reference_year_exclusions.empty:
        return {}
    crosswalk = _load_crosswalk(paths)
    links = crosswalk[
        pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0).gt(0.5)
    ][["ori9", "jurisdiction_id"]].copy()
    merged = reference_year_exclusions.merge(links, on="ori9", how="inner")
    out: dict[tuple[str, str], set[int]] = {}
    for row in merged.itertuples(index=False):
        key = (str(row.jurisdiction_id), str(row.offense))
        year_value = pd.to_numeric(row.year, errors="coerce")
        if pd.isna(year_value):
            continue
        out.setdefault(key, set()).add(int(year_value))
    return out


def build_municipal_estimates(
    *,
    paths: RepoPaths,
    config: MunicipalEstimatorConfig = MunicipalEstimatorConfig(),
    observation_ignore_blockers: tuple[str, ...] = (),
) -> pd.DataFrame:
    panel = _build_regime_aware_panel(
        paths=paths,
        config=config,
        observation_ignore_blockers=observation_ignore_blockers,
    )
    muni = _build_municipal_population(paths=paths, config=config)
    panel = panel.merge(
        muni[["jurisdiction_id", "bucket_population"]],
        on="jurisdiction_id",
        how="left",
    )
    panel["bucket_population"] = pd.to_numeric(panel["bucket_population"], errors="coerce").fillna(0.0)
    panel = suppress_extreme_summary_spikes(panel)
    agency_trend_panel = build_agency_trend_fill_panel(
        paths=paths,
        year_start=int(config.year_start),
        year_end=int(config.year_end),
        force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
        exclude_state_abbrs=tuple(config.exclude_scope_state_abbrs),
    )
    trend_fill_lookup = build_trend_fill_lookup(
        agency_trend_panel,
        entity_col="ori9",
        count_col="preferred_count",
        target_year=int(config.target_year),
    )
    jurisdiction_masked_gaps = _map_masked_gaps_to_jurisdictions(
        paths=paths,
        masked_gap_flags=build_masked_gap_flags(
            paths,
            target_year=int(config.target_year),
            agency_panel=agency_trend_panel,
        ),
    )
    reference_year_candidate_years = sorted(
        int(y)
        for y in agency_trend_panel["year"].dropna().unique().tolist()
        if int(y) < int(config.target_year)
    ) if not agency_trend_panel.empty else []
    jurisdiction_reference_year_exclusions = _map_reference_year_exclusions_to_jurisdictions(
        paths=paths,
        reference_year_exclusions=build_reference_year_masked_gap_years(
            paths,
            agency_panel=agency_trend_panel,
            candidate_years=reference_year_candidate_years,
        ),
    )

    target_rows = panel[panel["year"].eq(config.target_year)][
        [
            "jurisdiction_id",
            "offense",
            "preferred_source",
            "dominant_reporting_regime",
            "reported_count_preferred",
            "mean_months_reported_preferred",
            "usable_as_observed",
        ]
    ].copy()
    meta = (
        muni.assign(_key=1)
        .merge(pd.DataFrame({"offense": list(OFFENSES_7), "_key": 1}), on="_key", how="inner")
        .drop(columns="_key")
    )
    out = meta.merge(target_rows, on=["jurisdiction_id", "offense"], how="left")

    series = panel[
        [
            "jurisdiction_id",
            "state_fips",
            "offense",
            "year",
            "preferred_source",
            "dominant_reporting_regime",
            "reported_count_preferred",
            "mean_months_reported_preferred",
            "usable_as_observed",
        ]
    ].copy()
    series["reported_count_preferred"] = pd.to_numeric(series["reported_count_preferred"], errors="coerce")
    series["mean_months_reported_preferred"] = pd.to_numeric(series["mean_months_reported_preferred"], errors="coerce")

    est_map: dict[tuple[str, str], float] = {}
    src_map: dict[tuple[str, str], str] = {}
    usable_map: dict[tuple[str, str], int] = {}
    last_year_map: dict[tuple[str, str], int | float] = {}
    trend_ratio_map: dict[tuple[str, str], float] = {}
    trend_ratio_source_map: dict[tuple[str, str], str] = {}
    trend_panel_agency_count_map: dict[tuple[str, str], int | float] = {}
    trend_panel_mass_share_base_map: dict[tuple[str, str], float] = {}
    trend_panel_mass_share_target_map: dict[tuple[str, str], float] = {}
    masked_gap_map: dict[tuple[str, str], bool] = {}
    excluded_ref_years_map: dict[tuple[str, str], list[int]] = {}

    def record_trend_audit(key: tuple[str, str], audit: TrendFillAudit) -> None:
        trend_ratio_map[key] = float(audit.ratio)
        trend_ratio_source_map[key] = str(audit.source)
        trend_panel_agency_count_map[key] = (
            int(audit.panel_agency_count)
            if audit.panel_agency_count is not None
            else np.nan
        )
        trend_panel_mass_share_base_map[key] = (
            float(audit.panel_mass_share_base)
            if audit.panel_mass_share_base is not None
            else np.nan
        )
        trend_panel_mass_share_target_map[key] = (
            float(audit.panel_mass_share_target)
            if audit.panel_mass_share_target is not None
            else np.nan
        )

    def record_no_trend_audit(key: tuple[str, str], source: str) -> None:
        record_trend_audit(
            key,
            TrendFillAudit(
                ratio=1.0,
                source=source,
                panel_agency_count=None,
                panel_mass_share_base=None,
                panel_mass_share_target=None,
            ),
        )

    for (jurisdiction_id, offense), grp in series.groupby(["jurisdiction_id", "offense"], sort=False):
        key = (str(jurisdiction_id), str(offense))
        grp = grp.sort_values("year")
        usable = grp[grp["usable_as_observed"].eq(True) & grp["reported_count_preferred"].notna()].copy()
        excluded_years_for_key = jurisdiction_reference_year_exclusions.get(key, set())
        usable_hist_pre_exclusion = usable[usable["year"] < config.target_year].copy()
        usable_hist = usable_hist_pre_exclusion
        if excluded_years_for_key:
            usable_hist = usable_hist[~usable_hist["year"].astype(int).isin(excluded_years_for_key)].copy()
        usable_years = int(len(usable_hist))
        usable_map[key] = usable_years
        last_year = int(usable_hist["year"].max()) if not usable_hist.empty else np.nan
        last_year_map[key] = last_year
        excluded_ref_years_map[key] = sorted(
            int(y)
            for y in usable_hist_pre_exclusion["year"].astype(int).unique().tolist()
            if int(y) in excluded_years_for_key
        )

        target = grp[grp["year"].eq(config.target_year)]
        masked_gap = jurisdiction_masked_gaps.get(key) if not target.empty else None
        if masked_gap is not None:
            # Masked-gap reclassification (trend_fills.build_masked_gap_flags): the
            # target-year row claims a clean complete year but is materially incomplete,
            # so it must NOT be trusted as observed. partial_months reuses the existing
            # true-partial-ratio semantics with effective months from the NIBRS rollup;
            # collapsed_count falls through to the regular gapped-year ladder below.
            masked_gap_map[key] = True
            masked_kind, masked_months = masked_gap
            target_count = float(pd.to_numeric(target["reported_count_preferred"].iloc[0], errors="coerce") or 0.0)
            if (
                masked_kind == "partial_months"
                and np.isfinite(masked_months)
                and 0 < masked_months < 12
                and target_count > 0
            ):
                est_map[key] = float(max(target_count, target_count * (12.0 / masked_months)))
                src_map[key] = "true_partial_month_ratio"
                record_no_trend_audit(key, "not_applicable_masked_gap_true_partial")
                continue
        elif not target.empty and bool(target["usable_as_observed"].iloc[0]):
            est_map[key] = float(pd.to_numeric(target["reported_count_preferred"].iloc[0], errors="coerce") or 0.0)
            src_map[key] = "reported_regime_usable"
            record_no_trend_audit(key, "not_applicable_observed")
            continue

        if not target.empty and masked_gap is None:
            target_source = str(target["preferred_source"].iloc[0])
            target_regime = str(target["dominant_reporting_regime"].iloc[0])
            target_count = float(pd.to_numeric(target["reported_count_preferred"].iloc[0], errors="coerce") or 0.0)
            target_months = float(pd.to_numeric(target["mean_months_reported_preferred"].iloc[0], errors="coerce") or 0.0)
            if (
                target_source in {LOCAL_PUBLICATION_SOURCE, SUMMARY_SOURCE, STATE_PUBLICATION_SOURCE}
                and target_regime == "true_partial"
                and 0 < target_months < 12
                and target_count > 0
            ):
                est_map[key] = float(max(target_count, target_count * (12.0 / target_months)))
                src_map[key] = "true_partial_month_ratio"
                record_no_trend_audit(key, "not_applicable_true_partial")
                continue

        state_fips = str(grp["state_fips"].dropna().astype(str).str.zfill(2).iloc[0]) if grp["state_fips"].notna().any() else None

        if usable_years >= config.min_usable_years_for_trend and pd.notna(last_year) and (config.target_year - int(last_year)) <= config.max_year_gap_for_trend:
            est = _project_count_log_linear(
                years=usable_hist["year"].to_numpy(dtype=float),
                counts=usable_hist["reported_count_preferred"].to_numpy(dtype=float),
                target_year=config.target_year,
            )
            if np.isfinite(est):
                est_map[key] = float(max(0.0, est))
                src_map[key] = "trend_log_linear"
                record_no_trend_audit(key, "not_applicable_trend_log_linear")
                continue

        if usable_years >= 1 and pd.notna(last_year):
            latest = usable_hist.sort_values("year").iloc[-1]
            latest_count = float(pd.to_numeric(latest["reported_count_preferred"], errors="coerce") or 0.0)
            year_gap = config.target_year - int(latest["year"])
            trend_audit = trend_fill_lookup.resolve(
                state_fips=state_fips,
                offense=offense,
                base_year=last_year,
            )
            if year_gap == 1:
                est_map[key] = float(max(0.0, latest_count * float(trend_audit.ratio)))
                src_map[key] = "carryforward_state_trend"
                record_trend_audit(key, trend_audit)
                continue

            scaled_median, median_audit = scaled_history_median(
                usable_hist,
                state_fips=state_fips,
                offense=offense,
                trend_fill_lookup=trend_fill_lookup,
            )
            if scaled_median is not None and np.isfinite(float(scaled_median)):
                if year_gap <= config.max_year_gap_for_trend:
                    est_map[key] = float(max(0.0, float(scaled_median)))
                    src_map[key] = "hist_median_state_trend"
                    record_trend_audit(key, median_audit)
                else:
                    est_map[key] = float(max(0.0, float(scaled_median)))
                    src_map[key] = "hist_median"
                    record_trend_audit(key, median_audit)
                continue

        est_map[key] = float("nan")
        src_map[key] = "missing"
        record_no_trend_audit(key, "not_applicable_no_history")

    out["usable_history_years"] = [usable_map.get((j, o), 0) for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)]
    out["last_usable_year"] = [last_year_map.get((j, o), np.nan) for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)]
    out["estimated_count"] = [est_map.get((j, o), float("nan")) for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)]
    out["estimate_source"] = [src_map.get((j, o), "missing") for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)]
    out["fill_trend_ratio_applied"] = [
        trend_ratio_map.get((j, o), np.nan)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["fill_trend_ratio_source"] = [
        trend_ratio_source_map.get((j, o), pd.NA)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["fill_trend_panel_agency_count"] = [
        trend_panel_agency_count_map.get((j, o), np.nan)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["fill_trend_panel_mass_share_base"] = [
        trend_panel_mass_share_base_map.get((j, o), np.nan)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["fill_trend_panel_mass_share_target"] = [
        trend_panel_mass_share_target_map.get((j, o), np.nan)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["masked_gap_reclassified"] = [
        bool(masked_gap_map.get((j, o), False))
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["reference_years_excluded"] = [
        ",".join(str(y) for y in excluded_ref_years_map.get((j, o), [])) or pd.NA
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]

    is_missing = ~np.isfinite(pd.to_numeric(out["estimated_count"], errors="coerce"))
    peers = out.loc[
        ~is_missing & (out["bucket_population"] > 0),
        ["state_fips", "offense", "pop_band", "bucket_population", "estimated_count"],
    ].copy()
    if not peers.empty:
        peer_group = (
            peers.groupby(["state_fips", "offense", "pop_band"], dropna=False)
            .agg(sum_count=("estimated_count", "sum"), sum_pop=("bucket_population", "sum"))
            .reset_index()
        )
        peer_group["peer_rate"] = np.where(
            peer_group["sum_pop"] > 0,
            1e5 * peer_group["sum_count"] / peer_group["sum_pop"],
            np.nan,
        )
        peer_rate_map = {
            (str(row.state_fips).zfill(2), str(row.offense), str(row.pop_band)): float(row.peer_rate)
            for row in peer_group.itertuples(index=False)
            if pd.notna(row.peer_rate) and np.isfinite(row.peer_rate)
        }

        state_group = (
            peers.groupby(["state_fips", "offense"], dropna=False)
            .agg(
            sum_count=("estimated_count", "sum"),
            sum_pop=("bucket_population", "sum"),
        )
        )
        state_rate_map = {
            (str(state_fips).zfill(2), str(offense)): float(1e5 * row.sum_count / row.sum_pop)
            for (state_fips, offense), row in state_group.iterrows()
            if float(row.sum_pop) > 0
        }
        state_count_median_map = {
            (str(state_fips).zfill(2), str(offense)): float(pd.to_numeric(grp["estimated_count"], errors="coerce").median())
            for (state_fips, offense), grp in peers.groupby(["state_fips", "offense"], dropna=False)
        }
        national_group = peers.groupby("offense", dropna=False).agg(
            sum_count=("estimated_count", "sum"),
            sum_pop=("bucket_population", "sum"),
        )
        national_rate_map = {
            str(offense): float(1e5 * row.sum_count / row.sum_pop)
            for offense, row in national_group.iterrows()
            if float(row.sum_pop) > 0
        }
        national_count_median_map = {
            str(offense): float(pd.to_numeric(grp["estimated_count"], errors="coerce").median())
            for offense, grp in peers.groupby("offense", dropna=False)
        }
    else:
        peer_rate_map = {}
        state_rate_map = {}
        state_count_median_map = {}
        national_rate_map = {}
        national_count_median_map = {}

    fill_counts: list[float] = []
    fill_sources: list[str] = []
    for row in out.loc[is_missing, ["state_fips", "offense", "pop_band", "bucket_population"]].itertuples(index=False):
        state_fips = str(row.state_fips).zfill(2)
        offense = str(row.offense)
        pop_val = float(row.bucket_population) if pd.notna(row.bucket_population) else 0.0
        if pop_val > 0:
            rate = peer_rate_map.get((state_fips, offense, str(row.pop_band)))
            source = "peer_state_pop_band"
            if rate is None or not np.isfinite(rate):
                rate = state_rate_map.get((state_fips, offense))
                source = "peer_state_overall"
            if rate is None or not np.isfinite(rate):
                rate = national_rate_map.get(offense)
                source = "peer_national_overall"
            if rate is not None and np.isfinite(rate):
                fill_counts.append(float(max(0.0, rate) / 1e5 * pop_val))
                fill_sources.append(source)
                continue

        median_count = state_count_median_map.get((state_fips, offense))
        source = "peer_state_count_median"
        if median_count is None or not np.isfinite(median_count):
            median_count = national_count_median_map.get(offense)
            source = "peer_national_count_median"
        if median_count is None or not np.isfinite(median_count):
            fill_counts.append(0.0)
            fill_sources.append("peer_fallback_zero")
        else:
            fill_counts.append(float(max(0.0, median_count)))
            fill_sources.append(source)

    out.loc[is_missing, "estimated_count"] = fill_counts
    out.loc[is_missing, "estimate_source"] = fill_sources
    out.loc[is_missing, "fill_trend_ratio_applied"] = 1.0
    out.loc[is_missing, "fill_trend_ratio_source"] = "not_applicable_peer_fill"
    out.loc[is_missing, "fill_trend_panel_agency_count"] = np.nan
    out.loc[is_missing, "fill_trend_panel_mass_share_base"] = np.nan
    out.loc[is_missing, "fill_trend_panel_mass_share_target"] = np.nan

    out["estimated_count"] = pd.to_numeric(out["estimated_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["used_for_fill"] = ~out["estimate_source"].eq("reported_regime_usable")
    out["target_year"] = int(config.target_year)

    cols = [
        "jurisdiction_id",
        "state_fips",
        "state_abbr",
        "jurisdiction_name",
        "geo_type",
        "geoid",
        "offense",
        "bucket_population",
        "population_basis",
        "pop_band",
        "preferred_source",
        "dominant_reporting_regime",
        "reported_count_preferred",
        "usable_history_years",
        "last_usable_year",
        "estimated_count",
        "estimate_source",
        "fill_trend_ratio_applied",
        "fill_trend_ratio_source",
        "fill_trend_panel_agency_count",
        "fill_trend_panel_mass_share_base",
        "fill_trend_panel_mass_share_target",
        "masked_gap_reclassified",
        "reference_years_excluded",
        "used_for_fill",
        "target_year",
    ]
    return out[cols].sort_values(["state_fips", "jurisdiction_id", "offense"], kind="mergesort").reset_index(drop=True)


def write_v2_municipal_estimates(
    *,
    paths: RepoPaths,
    out_path: Path,
    config: MunicipalEstimatorConfig = MunicipalEstimatorConfig(),
    blocked_by: tuple[str, ...] | None = None,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> dict[str, int]:
    with stage_write_lock(paths=paths, stage="municipal_estimates", blocked_by=blocked_by):
        _ensure_municipal_estimate_dependencies(
            paths=paths,
            config=config,
            observation_ignore_blockers=observation_ignore_blockers,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        estimates = build_municipal_estimates(
            paths=paths,
            config=config,
            observation_ignore_blockers=observation_ignore_blockers,
        )
        estimates.to_parquet(out_path, index=False)
        return {
            "rows": int(len(estimates)),
            "jurisdictions": int(estimates["jurisdiction_id"].nunique()),
        }
