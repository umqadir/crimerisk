from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.crime.municipal_totals import MunicipalTotalsConfig, _assign_pop_band, _project_count_log_linear
from crimerisk.municipal_estimator import (
    MunicipalEstimatorConfig,
    get_v2_municipal_estimates_path,
    municipal_estimates_artifact_is_current,
    write_v2_municipal_estimates,
)
from crimerisk.paths import RepoPaths
from crimerisk.panel_guardrails import suppress_extreme_summary_spikes
from crimerisk.source_provenance import (
    CIUS_ORIGIN,
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    default_conversion_status_from_source,
    raw_data_source_from_parts,
    source_family_from_source,
    source_origin_from_parts,
)
from crimerisk.source_selection import build_agency_preferred_observations
from crimerisk.trend_fills import (
    TrendFillAudit,
    build_agency_target_estimates_from_panel,
    build_agency_trend_fill_lookup,
    build_agency_trend_fill_panel,
    build_masked_gap_flags,
    build_reference_year_masked_gap_years,
    build_trend_fill_lookup,
    load_agency_type_lookup,
    scaled_history_median,
)


PRODUCTION_SCOPE_EXCLUDE = {"AK", "AS", "CZ", "GU", "HI", "MP", "PR", "VI"}
STATE_REMAINDER_SUFFIX = ":state_nonmunicipal_remainder"
STATE_REMAINDER_TYPE = "state_nonmunicipal_remainder"


@dataclass(frozen=True)
class JurisdictionEstimatorConfig:
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
    force_municipal_estimates_rebuild: bool = False


def _load_crosswalk(paths: RepoPaths) -> pd.DataFrame:
    crosswalk = pd.read_parquet(paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet")
    crosswalk = crosswalk.rename(columns={"ori": "ori9"})
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0)
    return crosswalk


def _load_jurisdiction_master(paths: RepoPaths) -> pd.DataFrame:
    return pd.read_parquet(paths.state_dir / "reference" / "jurisdiction_master.parquet")


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = values.notna() & w.gt(0)
    if not mask.any():
        mask = values.notna()
        if not mask.any():
            return float("nan")
        return float(values.loc[mask].mean())
    return float(np.average(values.loc[mask], weights=w.loc[mask]))


def _dominant_label_frame(
    *,
    merged: pd.DataFrame,
    key_cols: list[str],
    label_col: str,
    output_col: str,
) -> pd.DataFrame:
    labels = merged[label_col].astype("string")
    weights = pd.to_numeric(merged["support_weight"], errors="coerce").fillna(0.0)
    mask = labels.notna() & labels.ne("")
    if not mask.any():
        return pd.DataFrame(columns=[*key_cols, output_col])
    support = (
        merged.loc[mask, key_cols]
        .assign(_label=labels.loc[mask].values, _weight=weights.loc[mask].values)
        .groupby([*key_cols, "_label"], dropna=False, as_index=False)["_weight"]
        .sum()
        .sort_values([*key_cols, "_weight", "_label"], ascending=[True] * len(key_cols) + [False, True], kind="mergesort")
    )
    support["_rank"] = support.groupby(key_cols, dropna=False).cumcount()
    top = support[support["_rank"].eq(0)][[*key_cols, "_label"]].rename(columns={"_label": output_col})
    return top.reset_index(drop=True)


def build_jurisdiction_year_preferred_panel(
    *,
    paths: RepoPaths,
    config: JurisdictionEstimatorConfig = JurisdictionEstimatorConfig(),
) -> pd.DataFrame:
    crosswalk = _load_crosswalk(paths)
    frames: list[pd.DataFrame] = []
    for year in range(int(config.year_start), int(config.year_end) + 1):
        preferred = build_agency_preferred_observations(
            paths=paths,
            year=year,
            force_reporting_regimes_rebuild=config.force_reporting_regimes_rebuild,
        )
        if preferred.empty:
            continue
        preferred["year"] = int(year)
        frames.append(preferred)
    if not frames:
        return pd.DataFrame()

    agency_panel = pd.concat(frames, ignore_index=True)
    agency_panel["state_abbr"] = agency_panel["state_abbr"].astype("string").str.upper()
    agency_panel = agency_panel[~agency_panel["state_abbr"].isin(set(config.exclude_scope_state_abbrs))].copy()

    merged = agency_panel.merge(
        crosswalk[["ori9", "jurisdiction_id", "weight"]],
        on="ori9",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged["preferred_count"] = pd.to_numeric(merged["preferred_count"], errors="coerce").fillna(0.0)
    merged["preferred_observation_weight"] = pd.to_numeric(
        merged["preferred_observation_weight"], errors="coerce"
    ).fillna(0.0)
    merged["preferred_months_reported"] = pd.to_numeric(
        merged["preferred_months_reported"], errors="coerce"
    )
    merged["allocated_count"] = merged["preferred_count"] * merged["weight"]
    merged["support_weight"] = merged["allocated_count"].abs()
    merged["support_weight"] = merged["support_weight"].where(merged["support_weight"].gt(0), merged["weight"])

    key_cols = ["jurisdiction_id", "year", "offense"]
    grouped = (
        merged.groupby(key_cols, dropna=False, as_index=False)
        .agg(
            reported_count_preferred=("allocated_count", "sum"),
            contributing_agencies=("ori9", "nunique"),
            total_support_weight=("support_weight", "sum"),
            true_partial_support=(
                "support_weight",
                lambda s: float(s[merged.loc[s.index, "reporting_regime"].eq("true_partial")].sum()),
            ),
        )
    )

    merged["obs_weight_num"] = (
        pd.to_numeric(merged["preferred_observation_weight"], errors="coerce").fillna(0.0)
        * pd.to_numeric(merged["support_weight"], errors="coerce").fillna(0.0)
    )
    merged["obs_weight_den"] = pd.to_numeric(merged["support_weight"], errors="coerce").fillna(0.0)
    merged["month_weight_num"] = (
        pd.to_numeric(merged["preferred_months_reported"], errors="coerce").fillna(0.0)
        * pd.to_numeric(merged["support_weight"], errors="coerce").fillna(0.0)
    )
    merged["month_weight_den"] = pd.to_numeric(merged["support_weight"], errors="coerce").where(
        pd.to_numeric(merged["preferred_months_reported"], errors="coerce").notna(),
        0.0,
    )
    weighted_meta = (
        merged.groupby(key_cols, dropna=False, as_index=False)
        .agg(
            obs_weight_num=("obs_weight_num", "sum"),
            obs_weight_den=("obs_weight_den", "sum"),
            month_weight_num=("month_weight_num", "sum"),
            month_weight_den=("month_weight_den", "sum"),
        )
    )
    weighted_meta["observation_weight_preferred"] = weighted_meta["obs_weight_num"] / weighted_meta["obs_weight_den"].where(
        weighted_meta["obs_weight_den"].gt(0),
        other=np.nan,
    )
    weighted_meta["mean_months_reported_preferred"] = weighted_meta["month_weight_num"] / weighted_meta["month_weight_den"].where(
        weighted_meta["month_weight_den"].gt(0),
        other=np.nan,
    )
    weighted_meta = weighted_meta[
        [*key_cols, "observation_weight_preferred", "mean_months_reported_preferred"]
    ].copy()

    source_meta = _dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="preferred_source",
        output_col="preferred_source",
    )
    family_meta = _dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="preferred_source_family",
        output_col="preferred_source_family",
    )
    origin_meta = _dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="preferred_source_origin",
        output_col="preferred_source_origin",
    )
    raw_source_meta = _dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="preferred_raw_data_source",
        output_col="preferred_raw_data_source",
    )
    conversion_meta = _dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="preferred_conversion_status",
        output_col="preferred_conversion_status",
    )
    regime_meta = _dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="reporting_regime",
        output_col="dominant_reporting_regime",
    )

    out = grouped.merge(weighted_meta, on=key_cols, how="left")
    out = out.merge(source_meta, on=key_cols, how="left")
    out = out.merge(family_meta, on=key_cols, how="left")
    out = out.merge(origin_meta, on=key_cols, how="left")
    out = out.merge(raw_source_meta, on=key_cols, how="left")
    out = out.merge(conversion_meta, on=key_cols, how="left")
    out = out.merge(regime_meta, on=key_cols, how="left")

    jurisdiction_master = _load_jurisdiction_master(paths)[
        [
            "jurisdiction_id",
            "jurisdiction_type",
            "jurisdiction_name",
            "state_fips",
            "state_abbr",
            "geo_type",
            "geoid",
        ]
    ].drop_duplicates()
    out = out.merge(jurisdiction_master, on="jurisdiction_id", how="left")
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["state_abbr"] = out["state_abbr"].astype("string").str.upper()
    return out.sort_values(["state_fips", "jurisdiction_type", "jurisdiction_id", "year", "offense"], kind="mergesort").reset_index(drop=True)


def _load_jurisdiction_population(
    *,
    paths: RepoPaths,
    config: JurisdictionEstimatorConfig,
) -> pd.DataFrame:
    block_path = paths.state_dir / "geometry" / "block_to_jurisdiction_crosswalk.parquet"
    blocks = pd.read_parquet(
        block_path,
        columns=["state_fips", "jurisdiction_id", "jurisdiction_type", "pop20"],
    )
    blocks["state_fips"] = blocks["state_fips"].astype("string").str.zfill(2)
    pop = (
        blocks.groupby(["state_fips", "jurisdiction_id", "jurisdiction_type"], dropna=False)["pop20"]
        .sum()
        .reset_index()
        .rename(columns={"pop20": "bucket_population"})
    )
    pop["bucket_population"] = pd.to_numeric(pop["bucket_population"], errors="coerce").fillna(0.0)
    pop["pop_band"] = pop["bucket_population"].map(lambda value: _assign_pop_band(value, config.pop_bands))
    return pop


def _mark_usable_observations(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    counts = pd.to_numeric(out["reported_count_preferred"], errors="coerce")
    months = pd.to_numeric(out["mean_months_reported_preferred"], errors="coerce")
    regime = out["dominant_reporting_regime"].astype("string")
    source = out["preferred_source"].astype("string")
    origin = out["preferred_source_origin"].astype("string")

    summary_usable = (
        source.isin([CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE])
        & counts.notna()
        & (
            origin.eq(CIUS_ORIGIN)
            | (
                source.eq(LOCAL_PUBLICATION_SOURCE)
                & regime.eq("annual_only_but_usable")
            )
            | (
                source.eq(STATE_PUBLICATION_SOURCE)
                & regime.eq("annual_only_but_usable")
            )
            | regime.isin(["full_monthly", "lumpy_or_batched", "annual_only_but_usable"])
        )
    )
    nibrs_usable = (
        source.eq(NIBRS_SOURCE)
        & counts.notna()
        & ~regime.eq("structurally_missing_or_unreliable")
    )
    out["usable_as_observed"] = summary_usable | nibrs_usable
    out["current_row_is_true_partial"] = (
        source.isin([LOCAL_PUBLICATION_SOURCE, SUMMARY_SOURCE, STATE_PUBLICATION_SOURCE])
        & regime.eq("true_partial")
        & months.between(1, 11, inclusive="both")
        & counts.fillna(0.0).gt(0)
    )
    return out


def _load_target_year_municipal_estimates(
    *,
    paths: RepoPaths,
    config: JurisdictionEstimatorConfig,
) -> pd.DataFrame:
    year = int(config.target_year)
    path = get_v2_municipal_estimates_path(paths, year=year)
    municipal_config = MunicipalEstimatorConfig(
        year_start=2018,
        year_end=year,
        target_year=year,
        force_reporting_regimes_rebuild=config.force_reporting_regimes_rebuild,
    )
    if (
        not config.force_municipal_estimates_rebuild
        and municipal_estimates_artifact_is_current(paths, config=municipal_config, out_path=path)
    ):
        return pd.read_parquet(path)
    write_v2_municipal_estimates(paths=paths, out_path=path, config=municipal_config)
    return pd.read_parquet(path)


def _normalize_estimate_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["estimate_confidence"] = np.select(
        [
            out["estimate_source"].astype("string").str.startswith("observed_"),
            out["estimate_source"].isin(
                ["true_partial_month_ratio", "trend_log_linear", "carryforward_state_trend", "hist_median_state_trend", "hist_median"]
            ),
        ],
        [
            "high",
            "medium",
        ],
        default="low",
    )
    out["estimated_from_panel"] = ~out["estimate_source"].astype("string").str.startswith("observed_")
    return out


def _observed_label_from_preferred_source(source: object) -> str:
    source_text = str(source)
    if source_text == CIUS_SOURCE:
        return "observed_cius_local"
    if source_text == LOCAL_PUBLICATION_SOURCE:
        return "observed_local_publication"
    if source_text == STATE_PUBLICATION_SOURCE:
        return "observed_state_publication"
    if source_text == SUMMARY_SOURCE:
        return "observed_srs"
    if source_text == NIBRS_SOURCE:
        return "observed_nibrs_rollup"
    return "observed_other"


def _apply_target_year_municipal_overrides(
    *,
    estimates: pd.DataFrame,
    municipal_estimates: pd.DataFrame,
    config: JurisdictionEstimatorConfig,
) -> pd.DataFrame:
    if estimates.empty or municipal_estimates.empty:
        return estimates
    target_year = int(config.target_year)
    municipal_estimates = municipal_estimates.copy()
    for col, default in {
        "fill_trend_ratio_applied": np.nan,
        "fill_trend_ratio_source": pd.NA,
        "fill_trend_panel_agency_count": np.nan,
        "fill_trend_panel_mass_share_base": np.nan,
        "fill_trend_panel_mass_share_target": np.nan,
        "masked_gap_reclassified": False,
    }.items():
        if col not in municipal_estimates.columns:
            municipal_estimates[col] = default
    muni = municipal_estimates.rename(
        columns={
            "preferred_source": "municipal_preferred_source",
            "dominant_reporting_regime": "municipal_reporting_regime",
            "reported_count_preferred": "municipal_reported_count_preferred",
            "estimated_count": "municipal_estimated_count",
            "estimate_source": "municipal_estimate_source",
            "used_for_fill": "municipal_used_for_fill",
            "fill_trend_ratio_applied": "municipal_fill_trend_ratio_applied",
            "fill_trend_ratio_source": "municipal_fill_trend_ratio_source",
            "fill_trend_panel_agency_count": "municipal_fill_trend_panel_agency_count",
            "fill_trend_panel_mass_share_base": "municipal_fill_trend_panel_mass_share_base",
            "fill_trend_panel_mass_share_target": "municipal_fill_trend_panel_mass_share_target",
            "masked_gap_reclassified": "municipal_masked_gap_reclassified",
        }
    )[
        [
            "jurisdiction_id",
            "offense",
            "municipal_preferred_source",
            "municipal_reporting_regime",
            "municipal_reported_count_preferred",
            "municipal_estimated_count",
            "municipal_estimate_source",
            "municipal_used_for_fill",
            "municipal_fill_trend_ratio_applied",
            "municipal_fill_trend_ratio_source",
            "municipal_fill_trend_panel_agency_count",
            "municipal_fill_trend_panel_mass_share_base",
            "municipal_fill_trend_panel_mass_share_target",
            "municipal_masked_gap_reclassified",
        ]
    ].copy()
    out = estimates.merge(muni, on=["jurisdiction_id", "offense"], how="left")
    use_muni = (
        out["year"].eq(target_year)
        & out["jurisdiction_type"].astype("string").eq("municipal")
        & out["municipal_estimated_count"].notna()
    )
    if not use_muni.any():
        return out.drop(
            columns=[
                "municipal_preferred_source",
                "municipal_reporting_regime",
                "municipal_reported_count_preferred",
                "municipal_estimated_count",
                "municipal_estimate_source",
                "municipal_used_for_fill",
                "municipal_masked_gap_reclassified",
            ],
            errors="ignore",
        )

    out.loc[use_muni, "preferred_source"] = out.loc[use_muni, "municipal_preferred_source"]
    out.loc[use_muni, "preferred_source_family"] = out.loc[use_muni, "preferred_source"].map(source_family_from_source)
    out.loc[use_muni, "preferred_source_origin"] = [
        source_origin_from_parts(source=source) for source in out.loc[use_muni, "preferred_source"]
    ]
    out.loc[use_muni, "preferred_raw_data_source"] = [
        raw_data_source_from_parts(source=source) for source in out.loc[use_muni, "preferred_source"]
    ]
    out.loc[use_muni, "preferred_conversion_status"] = [
        default_conversion_status_from_source(source) for source in out.loc[use_muni, "preferred_source"]
    ]
    out.loc[use_muni, "dominant_reporting_regime"] = out.loc[use_muni, "municipal_reporting_regime"]
    out.loc[use_muni, "reported_count_preferred"] = pd.to_numeric(
        out.loc[use_muni, "municipal_reported_count_preferred"], errors="coerce"
    )
    out.loc[use_muni, "estimated_count"] = pd.to_numeric(
        out.loc[use_muni, "municipal_estimated_count"], errors="coerce"
    )

    exact_annual_preferred = (
        out["preferred_source"].astype("string").eq(CIUS_SOURCE)
        | (
            out["preferred_source"].astype("string").eq(LOCAL_PUBLICATION_SOURCE)
            & out["dominant_reporting_regime"].astype("string").eq("annual_only_but_usable")
        )
    )
    out.loc[use_muni & exact_annual_preferred, "observation_weight_preferred"] = 1.0
    out.loc[use_muni & exact_annual_preferred, "mean_months_reported_preferred"] = 12.0

    municipal_used_for_fill = pd.Series(out["municipal_used_for_fill"], index=out.index, dtype="boolean").fillna(False).astype(bool)
    observed_mask = use_muni & ~municipal_used_for_fill
    out.loc[observed_mask, "estimate_source"] = out.loc[observed_mask, "preferred_source"].map(_observed_label_from_preferred_source)
    out.loc[use_muni & ~observed_mask, "estimate_source"] = out.loc[use_muni & ~observed_mask, "municipal_estimate_source"]
    for col in [
        "fill_trend_ratio_applied",
        "fill_trend_ratio_source",
        "fill_trend_panel_agency_count",
        "fill_trend_panel_mass_share_base",
        "fill_trend_panel_mass_share_target",
    ]:
        muni_col = f"municipal_{col}"
        if muni_col in out.columns:
            out.loc[use_muni, col] = out.loc[use_muni, muni_col]
    if "masked_gap_reclassified" not in out.columns:
        out["masked_gap_reclassified"] = False
    out.loc[use_muni, "masked_gap_reclassified"] = (
        pd.Series(out["municipal_masked_gap_reclassified"], index=out.index, dtype="boolean")
        .fillna(False)
        .astype(bool)
        .loc[use_muni]
    )
    out["masked_gap_reclassified"] = out["masked_gap_reclassified"].fillna(False).astype(bool)

    out = _mark_usable_observations(out)
    out["current_row_is_true_partial"] = (
        out["preferred_source"].astype("string").isin([LOCAL_PUBLICATION_SOURCE, SUMMARY_SOURCE, STATE_PUBLICATION_SOURCE])
        & out["dominant_reporting_regime"].astype("string").eq("true_partial")
        & pd.to_numeric(out["mean_months_reported_preferred"], errors="coerce").between(1, 11, inclusive="both")
        & pd.to_numeric(out["reported_count_preferred"], errors="coerce").fillna(0.0).gt(0)
    )
    out = _normalize_estimate_confidence(out)
    return out.drop(
        columns=[
            "municipal_preferred_source",
            "municipal_reporting_regime",
            "municipal_reported_count_preferred",
            "municipal_estimated_count",
            "municipal_estimate_source",
            "municipal_used_for_fill",
            "municipal_fill_trend_ratio_applied",
            "municipal_fill_trend_ratio_source",
            "municipal_fill_trend_panel_agency_count",
            "municipal_fill_trend_panel_mass_share_base",
            "municipal_fill_trend_panel_mass_share_target",
            "municipal_masked_gap_reclassified",
        ],
        errors="ignore",
    )


def _apply_target_year_remainder_pool_overrides(
    *,
    estimates: pd.DataFrame,
    paths: RepoPaths,
    config: JurisdictionEstimatorConfig,
    agency_panel: pd.DataFrame,
    trend_fill_lookup: object,
) -> pd.DataFrame:
    """Fix the state_nonmunicipal_remainder pool's target-year estimate so it reflects
    the whole pool, not just the agencies that happened to report this year.

    build_jurisdiction_year_preferred_panel aggregates reported_count_preferred for a
    remainder jurisdiction from CONTRIBUTING (reporting) agencies only -- an agency
    mapped into the pool that has no target-year row simply doesn't appear in the sum,
    so a fully-missing agency (e.g. a county PD with a total current-year reporting gap)
    is invisible to the jurisdiction-level trend/regime logic in
    _build_jurisdiction_estimates_for_target_year. That logic sees only the reporters'
    aggregate, judges it "usable_as_observed", and returns estimated_count == reported
    (zero adjustment budget), even when a major contributing agency is entirely absent.

    The per-agency estimator (trend_fills.build_agency_target_estimates_from_panel) does
    see every agency, including non-reporters, and is already the source of truth
    allocation._build_county_remainder_group_targets uses to split the remainder pool's
    adjustment budget back out across counties via agency_adjustment_count * crosswalk
    weight. This override reuses that SAME per-agency estimate (no second fill
    computation) to correct the pool's target-year estimated_count to
    reported_count_preferred (unchanged reporters-only baseline) plus the summed
    per-agency fill mass for pool agencies with a target-year gap, so the adjustment
    budget downstream is no longer forced to zero.
    """
    if estimates.empty:
        return estimates
    target_year = int(config.target_year)
    use_remainder = estimates["year"].eq(target_year) & estimates["jurisdiction_type"].astype("string").eq(
        STATE_REMAINDER_TYPE
    )
    if not use_remainder.any():
        return estimates

    reference_years = sorted(
        int(y)
        for y in agency_panel["year"].dropna().unique().tolist()
        if int(y) < target_year
    )
    agency_estimates = build_agency_target_estimates_from_panel(
        agency_panel,
        target_year=target_year,
        trend_fill_lookup=trend_fill_lookup,
        agency_type_lookup=load_agency_type_lookup(paths),
        masked_gap_flags=build_masked_gap_flags(
            paths,
            target_year=target_year,
            agency_panel=agency_panel,
        ),
        reference_year_exclusions=build_reference_year_masked_gap_years(
            paths,
            agency_panel=agency_panel,
            candidate_years=reference_years,
        ),
    )
    if agency_estimates.empty:
        return estimates
    agency_estimates = agency_estimates.copy()
    agency_estimates["state_fips"] = agency_estimates["state_fips"].astype("string").str.zfill(2)
    agency_estimates["offense"] = agency_estimates["offense"].astype("string")

    crosswalk = _load_crosswalk(paths)
    crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
    remainder_cw = crosswalk[
        crosswalk["jurisdiction_id"].astype("string").str.endswith(STATE_REMAINDER_SUFFIX, na=False)
    ][["ori9", "state_fips", "jurisdiction_id", "weight"]].copy()
    if remainder_cw.empty:
        return estimates

    rollup = agency_estimates.merge(remainder_cw, on=["ori9", "state_fips"], how="inner")
    if rollup.empty:
        return estimates
    rollup["weight"] = pd.to_numeric(rollup["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    rollup["_fill_raw"] = pd.to_numeric(rollup["agency_adjustment_count"], errors="coerce").fillna(0.0).clip(
        lower=0.0
    ) * rollup["weight"]

    pool_fill = (
        rollup.groupby(["jurisdiction_id", "offense"], dropna=False)["_fill_raw"]
        .sum()
        .rename("remainder_pool_fill_count")
        .reset_index()
    )
    rollup["_masked_fill"] = (
        rollup["masked_gap_reclassified"].fillna(False).astype(bool) & rollup["_fill_raw"].gt(0.0)
        if "masked_gap_reclassified" in rollup.columns
        else False
    )
    pool_masked = (
        rollup.groupby(["jurisdiction_id", "offense"], dropna=False)["_masked_fill"]
        .any()
        .rename("remainder_pool_masked_gap")
        .reset_index()
    )

    contributing = rollup[rollup["_fill_raw"].gt(0.0)].copy()
    if contributing.empty:
        dominant_label = pd.DataFrame(columns=["jurisdiction_id", "offense", "remainder_pool_fill_method"])
    else:
        label_mass = (
            contributing.groupby(["jurisdiction_id", "offense", "agency_estimate_source"], dropna=False)["_fill_raw"]
            .sum()
            .reset_index()
        )
        dominant_label = (
            label_mass.sort_values(
                ["jurisdiction_id", "offense", "_fill_raw", "agency_estimate_source"],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            .drop_duplicates(subset=["jurisdiction_id", "offense"], keep="first")
            .rename(columns={"agency_estimate_source": "remainder_pool_fill_method"})[
                ["jurisdiction_id", "offense", "remainder_pool_fill_method"]
            ]
        )

    out = estimates.merge(pool_fill, on=["jurisdiction_id", "offense"], how="left")
    out = out.merge(dominant_label, on=["jurisdiction_id", "offense"], how="left")
    out = out.merge(pool_masked, on=["jurisdiction_id", "offense"], how="left")
    out["remainder_pool_fill_count"] = pd.to_numeric(out["remainder_pool_fill_count"], errors="coerce").fillna(0.0)
    if "masked_gap_reclassified" not in out.columns:
        out["masked_gap_reclassified"] = False

    apply_mask = use_remainder & out["remainder_pool_fill_count"].gt(1e-9)
    if apply_mask.any():
        reported = pd.to_numeric(out.loc[apply_mask, "reported_count_preferred"], errors="coerce").fillna(0.0)
        out.loc[apply_mask, "estimated_count"] = reported + out.loc[apply_mask, "remainder_pool_fill_count"]
        out.loc[apply_mask, "estimate_source"] = out.loc[apply_mask, "remainder_pool_fill_method"].fillna(
            "remainder_pool_agency_rollup"
        )
        out.loc[apply_mask, "fill_trend_ratio_source"] = out.loc[apply_mask, "remainder_pool_fill_method"].fillna(
            "remainder_pool_agency_rollup"
        )
        out.loc[apply_mask, "masked_gap_reclassified"] = (
            pd.Series(out["masked_gap_reclassified"], index=out.index, dtype="boolean").fillna(False).astype(bool)
            | pd.Series(out["remainder_pool_masked_gap"], index=out.index, dtype="boolean").fillna(False).astype(bool)
        ).loc[apply_mask]
        out = _normalize_estimate_confidence(out)
    out["masked_gap_reclassified"] = out["masked_gap_reclassified"].fillna(False).astype(bool)
    return out.drop(
        columns=["remainder_pool_fill_count", "remainder_pool_fill_method", "remainder_pool_masked_gap"],
        errors="ignore",
    )


def _build_state_trend_maps(
    *,
    panel: pd.DataFrame,
    config: JurisdictionEstimatorConfig,
) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str], float]]:
    paired = panel[
        panel["usable_as_observed"]
        & panel["year"].isin([config.target_year - 1, config.target_year])
    ][["state_fips", "jurisdiction_type", "offense", "year", "reported_count_preferred"]].copy()
    if paired.empty:
        return {}, {}

    pivot = (
        paired.pivot_table(
            index=["state_fips", "jurisdiction_type", "offense"],
            columns="year",
            values="reported_count_preferred",
            aggfunc="sum",
        )
        .reset_index()
    )
    prev_col = config.target_year - 1
    curr_col = config.target_year
    if prev_col not in pivot.columns or curr_col not in pivot.columns:
        return {}, {}
    pivot = pivot.dropna(subset=[prev_col, curr_col]).copy()
    if pivot.empty:
        return {}, {}

    pivot["trend_ratio"] = (
        pd.to_numeric(pivot[curr_col], errors="coerce")
        / pd.to_numeric(pivot[prev_col], errors="coerce").replace(0.0, np.nan)
    ).clip(lower=config.state_trend_clip_low, upper=config.state_trend_clip_high)
    state_type = {
        (str(row.state_fips).zfill(2), str(row.jurisdiction_type), str(row.offense)): float(row.trend_ratio)
        for row in pivot.itertuples(index=False)
        if pd.notna(row.trend_ratio) and np.isfinite(row.trend_ratio)
    }

    overall = (
        paired.groupby(["state_fips", "offense", "year"], dropna=False)["reported_count_preferred"]
        .sum()
        .reset_index()
        .pivot_table(index=["state_fips", "offense"], columns="year", values="reported_count_preferred", aggfunc="sum")
        .reset_index()
    )
    if prev_col in overall.columns and curr_col in overall.columns:
        overall["trend_ratio"] = (
            pd.to_numeric(overall[curr_col], errors="coerce")
            / pd.to_numeric(overall[prev_col], errors="coerce").replace(0.0, np.nan)
        ).clip(lower=config.state_trend_clip_low, upper=config.state_trend_clip_high)
        state_overall = {
            (str(row.state_fips).zfill(2), str(row.offense)): float(row.trend_ratio)
            for row in overall.itertuples(index=False)
            if pd.notna(row.trend_ratio) and np.isfinite(row.trend_ratio)
        }
    else:
        state_overall = {}
    return state_type, state_overall


def _build_jurisdiction_estimates_for_target_year(
    *,
    paths: RepoPaths,
    panel: pd.DataFrame,
    population: pd.DataFrame,
    config: JurisdictionEstimatorConfig = JurisdictionEstimatorConfig(),
    target_year: int,
    trend_fill_lookup: object | None = None,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    if trend_fill_lookup is None:
        trend_fill_lookup = build_agency_trend_fill_lookup(
            paths=paths,
            year_start=int(config.year_start),
            year_end=int(config.year_end),
            target_year=int(target_year),
            force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
            exclude_state_abbrs=tuple(config.exclude_scope_state_abbrs),
        )

    target_rows = panel[panel["year"].eq(int(target_year))].copy()
    meta = (
        target_rows[
            [
                "jurisdiction_id",
                "jurisdiction_type",
                "jurisdiction_name",
                "state_fips",
                "state_abbr",
                "geo_type",
                "geoid",
                "offense",
                "bucket_population",
                "pop_band",
                "preferred_source",
                "preferred_source_family",
                "preferred_source_origin",
                "preferred_raw_data_source",
                "preferred_conversion_status",
                "dominant_reporting_regime",
                "reported_count_preferred",
                "observation_weight_preferred",
                "mean_months_reported_preferred",
                "usable_as_observed",
                "current_row_is_true_partial",
            ]
        ].copy()
    )
    meta["year"] = int(target_year)

    if meta.empty:
        return pd.DataFrame()

    series = panel[
        [
            "jurisdiction_id",
            "jurisdiction_type",
            "state_fips",
            "offense",
            "year",
            "reported_count_preferred",
            "usable_as_observed",
            "preferred_source_origin",
            "current_row_is_true_partial",
            "mean_months_reported_preferred",
        ]
    ].copy()
    series["reported_count_preferred"] = pd.to_numeric(series["reported_count_preferred"], errors="coerce")

    est_map: dict[tuple[str, str], float] = {}
    src_map: dict[tuple[str, str], str] = {}
    history_map: dict[tuple[str, str], int] = {}
    last_year_map: dict[tuple[str, str], float] = {}
    trend_ratio_map: dict[tuple[str, str], float] = {}
    trend_ratio_source_map: dict[tuple[str, str], str] = {}
    trend_panel_agency_count_map: dict[tuple[str, str], int | float] = {}
    trend_panel_mass_share_base_map: dict[tuple[str, str], float] = {}
    trend_panel_mass_share_target_map: dict[tuple[str, str], float] = {}

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
        grp = grp.sort_values("year").copy()
        current = grp[grp["year"].eq(int(target_year))]
        current_count = (
            float(pd.to_numeric(current["reported_count_preferred"].iloc[0], errors="coerce"))
            if not current.empty and pd.notna(current["reported_count_preferred"].iloc[0])
            else 0.0
        )
        usable_hist = grp[(grp["usable_as_observed"].eq(True)) & grp["year"].lt(int(target_year))].copy()
        history_map[key] = int(len(usable_hist))
        last_year_map[key] = float(usable_hist["year"].max()) if not usable_hist.empty else np.nan

        if not current.empty and bool(current["usable_as_observed"].iloc[0]):
            origin = str(current["preferred_source_origin"].iloc[0])
            est_map[key] = float(max(0.0, current_count))
            src_map[key] = f"observed_{origin}"
            record_no_trend_audit(key, "not_applicable_observed")
            continue

        if not current.empty and bool(current["current_row_is_true_partial"].iloc[0]):
            months = float(pd.to_numeric(current["mean_months_reported_preferred"].iloc[0], errors="coerce") or 0.0)
            if 0 < months < 12 and current_count > 0:
                est_map[key] = float(max(current_count, current_count * (12.0 / months)))
                src_map[key] = "true_partial_month_ratio"
                record_no_trend_audit(key, "not_applicable_true_partial")
                continue

        state_fips = str(grp["state_fips"].dropna().astype(str).str.zfill(2).iloc[0]) if grp["state_fips"].notna().any() else None

        if len(usable_hist) >= config.min_usable_years_for_trend:
            last_year = int(usable_hist["year"].max())
            if (int(target_year) - last_year) <= config.max_year_gap_for_trend:
                est = _project_count_log_linear(
                    years=usable_hist["year"].to_numpy(dtype=float),
                    counts=usable_hist["reported_count_preferred"].to_numpy(dtype=float),
                    target_year=int(target_year),
                )
                if np.isfinite(est):
                    est_map[key] = float(max(current_count, max(0.0, est)))
                    src_map[key] = "trend_log_linear"
                    record_no_trend_audit(key, "not_applicable_trend_log_linear")
                    continue

        if len(usable_hist) >= 1:
            latest = usable_hist.sort_values("year").iloc[-1]
            latest_count = float(pd.to_numeric(latest["reported_count_preferred"], errors="coerce") or 0.0)
            year_gap = int(target_year) - int(latest["year"])
            trend_audit = trend_fill_lookup.resolve(
                state_fips=state_fips,
                offense=offense,
                base_year=latest["year"],
            )
            if year_gap == 1:
                est_map[key] = float(max(current_count, max(0.0, latest_count * float(trend_audit.ratio))))
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
                    est_map[key] = float(max(current_count, max(0.0, float(scaled_median))))
                    src_map[key] = "hist_median_state_trend"
                    record_trend_audit(key, median_audit)
                else:
                    est_map[key] = float(max(current_count, max(0.0, float(scaled_median))))
                    src_map[key] = "hist_median"
                    record_trend_audit(key, median_audit)
                continue

        est_map[key] = float("nan")
        src_map[key] = "missing"
        record_no_trend_audit(key, "not_applicable_no_history")

    out = meta.copy()
    out["usable_history_years"] = [history_map.get((str(j), str(o)), 0) for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)]
    out["last_usable_year"] = [last_year_map.get((str(j), str(o)), np.nan) for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)]
    out["estimated_count"] = [est_map.get((str(j), str(o)), float("nan")) for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)]
    out["estimate_source"] = [src_map.get((str(j), str(o)), "missing") for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)]
    out["fill_trend_ratio_applied"] = [
        trend_ratio_map.get((str(j), str(o)), np.nan)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["fill_trend_ratio_source"] = [
        trend_ratio_source_map.get((str(j), str(o)), pd.NA)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["fill_trend_panel_agency_count"] = [
        trend_panel_agency_count_map.get((str(j), str(o)), np.nan)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["fill_trend_panel_mass_share_base"] = [
        trend_panel_mass_share_base_map.get((str(j), str(o)), np.nan)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]
    out["fill_trend_panel_mass_share_target"] = [
        trend_panel_mass_share_target_map.get((str(j), str(o)), np.nan)
        for j, o in zip(out["jurisdiction_id"], out["offense"], strict=True)
    ]

    is_missing = ~np.isfinite(pd.to_numeric(out["estimated_count"], errors="coerce"))
    peers = out.loc[
        ~is_missing & out["bucket_population"].gt(0),
        ["state_fips", "jurisdiction_type", "offense", "pop_band", "bucket_population", "estimated_count"],
    ].copy()

    if not peers.empty:
        peer_group = (
            peers.groupby(["state_fips", "jurisdiction_type", "offense", "pop_band"], dropna=False)
            .agg(sum_count=("estimated_count", "sum"), sum_pop=("bucket_population", "sum"))
            .reset_index()
        )
        peer_group["peer_rate"] = np.where(
            peer_group["sum_pop"] > 0,
            1e5 * peer_group["sum_count"] / peer_group["sum_pop"],
            np.nan,
        )
        peer_rate_map = {
            (str(row.state_fips).zfill(2), str(row.jurisdiction_type), str(row.offense), str(row.pop_band)): float(row.peer_rate)
            for row in peer_group.itertuples(index=False)
            if pd.notna(row.peer_rate) and np.isfinite(row.peer_rate)
        }
        state_type_group = (
            peers.groupby(["state_fips", "jurisdiction_type", "offense"], dropna=False)
            .agg(sum_count=("estimated_count", "sum"), sum_pop=("bucket_population", "sum"))
            .reset_index()
        )
        state_type_rate_map = {
            (str(row.state_fips).zfill(2), str(row.jurisdiction_type), str(row.offense)): float(1e5 * row.sum_count / row.sum_pop)
            for row in state_type_group.itertuples(index=False)
            if float(row.sum_pop) > 0
        }
        national_type_group = (
            peers.groupby(["jurisdiction_type", "offense"], dropna=False)
            .agg(sum_count=("estimated_count", "sum"), sum_pop=("bucket_population", "sum"))
            .reset_index()
        )
        national_type_rate_map = {
            (str(row.jurisdiction_type), str(row.offense)): float(1e5 * row.sum_count / row.sum_pop)
            for row in national_type_group.itertuples(index=False)
            if float(row.sum_pop) > 0
        }
    else:
        peer_rate_map = {}
        state_type_rate_map = {}
        national_type_rate_map = {}

    fill_counts: list[float] = []
    fill_sources: list[str] = []
    for row in out.loc[is_missing, ["state_fips", "jurisdiction_type", "offense", "pop_band", "bucket_population", "reported_count_preferred"]].itertuples(index=False):
        state_fips = str(row.state_fips).zfill(2)
        jurisdiction_type = str(row.jurisdiction_type)
        offense = str(row.offense)
        reported = float(pd.to_numeric(row.reported_count_preferred, errors="coerce") or 0.0)
        pop_val = float(row.bucket_population) if pd.notna(row.bucket_population) else 0.0
        if pop_val > 0:
            rate = peer_rate_map.get((state_fips, jurisdiction_type, offense, str(row.pop_band)))
            source = "peer_state_type_pop_band"
            if rate is None or not np.isfinite(rate):
                rate = state_type_rate_map.get((state_fips, jurisdiction_type, offense))
                source = "peer_state_type_overall"
            if rate is None or not np.isfinite(rate):
                rate = national_type_rate_map.get((jurisdiction_type, offense))
                source = "peer_national_type_overall"
            if rate is not None and np.isfinite(rate):
                fill_counts.append(float(max(reported, max(0.0, rate) / 1e5 * pop_val)))
                fill_sources.append(source)
                continue
        fill_counts.append(float(max(0.0, reported)))
        fill_sources.append("reported_floor_only")

    out.loc[is_missing, "estimated_count"] = fill_counts
    out.loc[is_missing, "estimate_source"] = fill_sources
    out.loc[is_missing, "fill_trend_ratio_applied"] = 1.0
    out.loc[is_missing, "fill_trend_ratio_source"] = "not_applicable_peer_fill"
    out.loc[is_missing, "fill_trend_panel_agency_count"] = np.nan
    out.loc[is_missing, "fill_trend_panel_mass_share_base"] = np.nan
    out.loc[is_missing, "fill_trend_panel_mass_share_target"] = np.nan
    out["estimated_count"] = pd.to_numeric(out["estimated_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["estimated_count"] = np.maximum(
        out["estimated_count"].to_numpy(dtype=float),
        pd.to_numeric(out["reported_count_preferred"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )
    out = _normalize_estimate_confidence(out)
    return out.sort_values(["state_fips", "jurisdiction_type", "jurisdiction_id", "year", "offense"], kind="mergesort").reset_index(drop=True)


def build_jurisdiction_year_estimates(
    *,
    paths: RepoPaths,
    config: JurisdictionEstimatorConfig = JurisdictionEstimatorConfig(),
) -> pd.DataFrame:
    panel = build_jurisdiction_year_preferred_panel(paths=paths, config=config)
    if panel.empty:
        return pd.DataFrame()
    panel = _mark_usable_observations(panel)
    population = _load_jurisdiction_population(paths=paths, config=config)
    panel = panel.merge(population, on=["state_fips", "jurisdiction_id", "jurisdiction_type"], how="left")
    panel["bucket_population"] = pd.to_numeric(panel["bucket_population"], errors="coerce").fillna(0.0)
    panel["pop_band"] = panel["pop_band"].fillna(
        panel["bucket_population"].map(lambda value: _assign_pop_band(value, config.pop_bands))
    )
    panel = suppress_extreme_summary_spikes(panel)

    frames: list[pd.DataFrame] = []
    agency_trend_panel = build_agency_trend_fill_panel(
        paths=paths,
        year_start=int(config.year_start),
        year_end=int(config.year_end),
        force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
        exclude_state_abbrs=tuple(config.exclude_scope_state_abbrs),
    )
    trend_fill_lookups = {
        int(target_year): build_trend_fill_lookup(
            agency_trend_panel,
            entity_col="ori9",
            count_col="preferred_count",
            target_year=int(target_year),
        )
        for target_year in range(int(config.year_start), int(config.year_end) + 1)
    }
    for target_year in range(int(config.year_start), int(config.year_end) + 1):
        year_frame = _build_jurisdiction_estimates_for_target_year(
            paths=paths,
            panel=panel,
            population=population,
            config=config,
            target_year=target_year,
            trend_fill_lookup=trend_fill_lookups[int(target_year)],
        )
        if not year_frame.empty:
            frames.append(year_frame)
    if not frames:
        return pd.DataFrame()
    estimates = pd.concat(frames, ignore_index=True)
    municipal_estimates = _load_target_year_municipal_estimates(paths=paths, config=config)
    estimates = _apply_target_year_municipal_overrides(
        estimates=estimates,
        municipal_estimates=municipal_estimates,
        config=config,
    )
    estimates = _apply_target_year_remainder_pool_overrides(
        estimates=estimates,
        paths=paths,
        config=config,
        agency_panel=agency_trend_panel,
        trend_fill_lookup=trend_fill_lookups[int(config.target_year)],
    )
    return estimates.sort_values(
        ["state_fips", "jurisdiction_type", "jurisdiction_id", "year", "offense"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_jurisdiction_estimates_2024(
    *,
    paths: RepoPaths,
    config: JurisdictionEstimatorConfig = JurisdictionEstimatorConfig(),
) -> pd.DataFrame:
    panel = build_jurisdiction_year_estimates(paths=paths, config=config)
    if panel.empty:
        return pd.DataFrame()
    out = panel[panel["year"].eq(int(config.target_year))].copy()
    rename_map = {
            "preferred_source": f"preferred_source_{int(config.target_year)}",
            "preferred_source_family": f"preferred_source_family_{int(config.target_year)}",
            "preferred_source_origin": f"preferred_source_origin_{int(config.target_year)}",
            "preferred_raw_data_source": f"preferred_raw_data_source_{int(config.target_year)}",
            "preferred_conversion_status": f"preferred_conversion_status_{int(config.target_year)}",
            "dominant_reporting_regime": f"dominant_reporting_regime_{int(config.target_year)}",
        "reported_count_preferred": f"reported_count_preferred_{int(config.target_year)}",
        "observation_weight_preferred": f"observation_weight_preferred_{int(config.target_year)}",
        "mean_months_reported_preferred": f"mean_months_reported_preferred_{int(config.target_year)}",
        "usable_as_observed": f"target_is_observed_{int(config.target_year)}",
        "estimated_count": f"estimated_count_{int(config.target_year)}",
    }
    out = out.rename(columns=rename_map)
    return out


def write_jurisdiction_year_estimates(
    *,
    paths: RepoPaths,
    out_path: Path,
    config: JurisdictionEstimatorConfig = JurisdictionEstimatorConfig(),
) -> Path:
    estimates = build_jurisdiction_year_estimates(paths=paths, config=config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_parquet(out_path, index=False)
    return out_path
