from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from crimerisk.build_freshness import artifact_is_current
from crimerisk.crime.srs import SRS_OFFENSE_COLUMN_MAP, ensure_srs_month_year_parquet
from crimerisk.observations import (
    ObservationBuildConfig,
    get_v2_observation_paths,
    observation_dependency_paths,
    observations_artifacts_are_current,
    write_v2_observations,
)
from crimerisk.paths import RepoPaths
from crimerisk.stage1_adjudications import (
    assert_registry_bit,
    build_zero_missing_directives,
    registry_dependency_paths,
)
from crimerisk.stage_locks import blockers_for_stage, stage_write_lock
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
)


# The regime reason a Stage-1 adjudicated zero-vs-missing verdict writes. Named rather than reusing
# "manual_override" so a consumer can tell a reviewed 2024 case from a hand-curated config row, and
# so `trend_fills._assert_no_fill_where_the_chosen_lane_reported` can recognise it.
STAGE1_ADJUDICATED_MISSING_REASON = "stage1_adjudicated_zero_reads_missing"

MONTH_ORDER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass(frozen=True)
class ReportingRegimeBuildConfig:
    year_start: int = 2018
    year_end: int = 2024
    override_path: Path | None = None
    source_override_path: Path | None = None


def _resolve_reporting_regime_config(
    *,
    paths: RepoPaths,
    config: ReportingRegimeBuildConfig,
) -> ReportingRegimeBuildConfig:
    override_path = config.override_path
    if override_path is None:
        candidate = paths.repo_root / "configs" / "reporting_regime_overrides.csv"
        override_path = candidate if candidate.exists() else None

    source_override_path = config.source_override_path
    if source_override_path is None:
        candidate = paths.repo_root / "configs" / "source_preference_overrides.csv"
        source_override_path = candidate if candidate.exists() else None

    return ReportingRegimeBuildConfig(
        year_start=int(config.year_start),
        year_end=int(config.year_end),
        override_path=override_path,
        source_override_path=source_override_path,
    )


def get_v2_reporting_regimes_path(paths: RepoPaths) -> Path:
    return paths.state_dir / "modeling" / "agency_year_reporting_regimes.parquet"


def _observation_build_config(config: ReportingRegimeBuildConfig) -> ObservationBuildConfig:
    return ObservationBuildConfig(
        year_start=int(config.year_start),
        year_end=int(config.year_end),
    )


def _ensure_reporting_regime_dependencies(
    paths: RepoPaths,
    *,
    config: ReportingRegimeBuildConfig,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> None:
    agency_path, jurisdiction_path = get_v2_observation_paths(paths)
    observation_config = _observation_build_config(config)
    if observations_artifacts_are_current(
        paths,
        config=observation_config,
        agency_out_path=agency_path,
        jurisdiction_out_path=jurisdiction_path,
    ):
        return
    write_v2_observations(
        paths=paths,
        agency_out_path=agency_path,
        jurisdiction_out_path=jurisdiction_path,
        config=observation_config,
        blocked_by=blockers_for_stage(
            "observations",
            ignore=("reporting_regimes", *observation_ignore_blockers),
        ),
        reference_ignore_blockers=("reporting_regimes", *observation_ignore_blockers),
    )


def reporting_regime_dependency_paths(
    paths: RepoPaths,
    *,
    config: ReportingRegimeBuildConfig = ReportingRegimeBuildConfig(),
) -> list[Path]:
    resolved = _resolve_reporting_regime_config(paths=paths, config=config)
    agency_observation_path, _ = get_v2_observation_paths(paths)
    deps: list[Path] = [
        agency_observation_path,
        paths.data_dir / "SRS-Kaplan-1960-2024" / "offenses_known_parquet_1960_2024_month.zip",
        Path(__file__),
        *observation_dependency_paths(paths, config=_observation_build_config(resolved)),
    ]
    if resolved.override_path is not None:
        deps.append(resolved.override_path)
    if resolved.source_override_path is not None:
        deps.append(resolved.source_override_path)
    deps.extend(registry_dependency_paths(paths))
    return deps


def reporting_regimes_artifact_is_current(
    paths: RepoPaths,
    *,
    config: ReportingRegimeBuildConfig = ReportingRegimeBuildConfig(),
    out_path: Path | None = None,
) -> bool:
    resolved = _resolve_reporting_regime_config(paths=paths, config=config)
    if not observations_artifacts_are_current(paths, config=_observation_build_config(resolved)):
        return False
    artifact_path = out_path or get_v2_reporting_regimes_path(paths)
    return artifact_is_current(
        artifact_path,
        reporting_regime_dependency_paths(paths, config=resolved),
    )


def _load_agency_year_observations(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    return pd.read_parquet(path)


def _load_srs_month_mask_detail(
    *,
    paths: RepoPaths,
    config: ReportingRegimeBuildConfig,
) -> pd.DataFrame:
    srs_month_zip = paths.data_dir / "SRS-Kaplan-1960-2024" / "offenses_known_parquet_1960_2024_month.zip"
    frames: list[pd.DataFrame] = []
    for year in range(config.year_start, config.year_end + 1):
        month_parquet_path = ensure_srs_month_year_parquet(
            zip_path=srs_month_zip,
            year=year,
            cache_dir=paths.cache_dir,
        ).parquet_path
        offense_select = " union all ".join(
            [
                f"""
                SELECT
                  COALESCE(NULLIF(ori9, ''), ori || '00') AS ori9,
                  CAST(year AS INTEGER) AS year,
                  '{offense}' AS offense,
                  LOWER(month) AS month_name,
                  CAST(COALESCE(month_missing, 0) AS INTEGER) AS month_missing,
                  COALESCE(CAST({source_col} AS DOUBLE), 0.0) AS count
                FROM read_parquet('{month_parquet_path.as_posix()}')
                WHERE CAST(year AS INTEGER) = {year}
                  AND COALESCE(NULLIF(ori9, ''), ori || '00') IS NOT NULL
                """.strip()
                for offense, source_col in SRS_OFFENSE_COLUMN_MAP.items()
            ]
        )
        monthly = duckdb.sql(offense_select).df()
        frames.append(monthly)

    monthly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if monthly.empty:
        return pd.DataFrame(
            columns=[
                "ori9",
                "year",
                "offense",
                "srs_non_missing_months",
                "srs_nonzero_reported_months",
                "srs_reported_month_set",
            ]
        )

    monthly["month_missing"] = pd.to_numeric(monthly["month_missing"], errors="coerce").fillna(0).astype(int)
    monthly["count"] = pd.to_numeric(monthly["count"], errors="coerce").fillna(0.0)
    monthly["month_num"] = monthly["month_name"].map(MONTH_ORDER)

    detail = (
        monthly.groupby(["ori9", "year", "offense"], dropna=False)
        .agg(
            srs_non_missing_months=("month_missing", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) == 0).sum())),
            srs_nonzero_reported_months=(
                "count",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum()),
            ),
            srs_reported_month_set=(
                "month_num",
                lambda s: ",".join(
                    str(int(x))
                    for x in sorted(
                        monthly.loc[s.index]
                        .loc[monthly.loc[s.index, "month_missing"].eq(0), "month_num"]
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                ),
            ),
        )
        .reset_index()
    )
    return detail


def _load_reporting_regime_overrides(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "ori",
                "year",
                "offense",
                "final_reporting_regime",
                "evidence_type",
                "source_note",
                "confidence",
                "reviewer_note",
            ]
        )
    if path.suffix.lower() == ".parquet":
        overrides = pd.read_parquet(path).copy()
    else:
        overrides = pd.read_csv(path).copy()
    required = {
        "ori",
        "year",
        "offense",
        "final_reporting_regime",
        "evidence_type",
        "source_note",
        "confidence",
        "reviewer_note",
    }
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(f"Reporting regime overrides missing columns: {sorted(missing)}")
    overrides["ori9"] = overrides["ori"].astype("string")
    overrides["year"] = pd.to_numeric(overrides["year"], errors="coerce").astype("Int64")
    overrides["offense"] = overrides["offense"].astype("string")
    overrides["final_reporting_regime"] = overrides["final_reporting_regime"].astype("string")
    valid_regimes = {
        "full_monthly",
        "true_partial",
        "lumpy_or_batched",
        "annual_only_but_usable",
        "structurally_missing_or_unreliable",
    }
    invalid = overrides.loc[~overrides["final_reporting_regime"].isin(valid_regimes), ["ori9", "year", "offense", "final_reporting_regime"]]
    if not invalid.empty:
        raise ValueError(f"Invalid reporting regime overrides: {invalid.to_dict(orient='records')}")
    dupes = overrides.loc[overrides.duplicated(["ori9", "year", "offense"], keep=False), ["ori9", "year", "offense"]]
    if not dupes.empty:
        raise ValueError(f"Duplicate reporting regime overrides: {dupes.to_dict(orient='records')}")
    return overrides[
        [
            "ori9",
            "year",
            "offense",
            "final_reporting_regime",
            "evidence_type",
            "source_note",
            "confidence",
            "reviewer_note",
        ]
    ].copy()


def _load_source_preference_overrides(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "ori9",
                "year",
                "offense",
                "preferred_source",
                "evidence_type",
                "source_note",
                "confidence",
                "reviewer_note",
            ]
        )
    if path.suffix.lower() == ".parquet":
        overrides = pd.read_parquet(path).copy()
    else:
        overrides = pd.read_csv(path).copy()
    required = {
        "ori",
        "year",
        "offense",
        "preferred_source",
        "evidence_type",
        "source_note",
        "confidence",
        "reviewer_note",
    }
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(f"Source preference overrides missing columns: {sorted(missing)}")
    overrides["ori9"] = overrides["ori"].astype("string")
    overrides["year"] = pd.to_numeric(overrides["year"], errors="coerce").astype("Int64")
    overrides["offense"] = overrides["offense"].astype("string")
    overrides["preferred_source"] = overrides["preferred_source"].astype("string")
    valid_sources = {
        CIUS_SOURCE,
        LOCAL_PUBLICATION_SOURCE,
        STATE_PUBLICATION_SOURCE,
        SUMMARY_SOURCE,
        NIBRS_SOURCE,
    }
    invalid = overrides.loc[
        ~overrides["preferred_source"].isin(valid_sources),
        ["ori9", "year", "offense", "preferred_source"],
    ]
    if not invalid.empty:
        raise ValueError(f"Invalid source preference overrides: {invalid.to_dict(orient='records')}")
    dupes = overrides.loc[
        overrides.duplicated(["ori9", "year", "offense"], keep=False),
        ["ori9", "year", "offense"],
    ]
    if not dupes.empty:
        raise ValueError(f"Duplicate source preference overrides: {dupes.to_dict(orient='records')}")
    return overrides[
        [
            "ori9",
            "year",
            "offense",
            "preferred_source",
            "evidence_type",
            "source_note",
            "confidence",
            "reviewer_note",
        ]
    ].copy()


def build_agency_year_reporting_regimes(
    *,
    paths: RepoPaths,
    config: ReportingRegimeBuildConfig = ReportingRegimeBuildConfig(),
    agency_year_observations: pd.DataFrame | None = None,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> pd.DataFrame:
    config = _resolve_reporting_regime_config(paths=paths, config=config)
    if agency_year_observations is None:
        _ensure_reporting_regime_dependencies(
            paths=paths,
            config=config,
            observation_ignore_blockers=observation_ignore_blockers,
        )
    obs = agency_year_observations.copy() if agency_year_observations is not None else _load_agency_year_observations(paths)
    obs = obs[(obs["year"].astype(int) >= config.year_start) & (obs["year"].astype(int) <= config.year_end)].copy()

    cius = (
        obs[obs["source"].eq(CIUS_SOURCE)]
        .rename(
            columns={
                "count": "cius_count",
                "conversion_status": "cius_conversion_status",
                "months_reported": "cius_months_reported",
                "quality_tier": "cius_quality_tier",
                "observation_weight": "cius_observation_weight",
            }
        )[
            [
                "ori9",
                "year",
                "offense",
                "state_fips",
                "state_abbr",
                "agency_name_std",
                "agency_type_norm",
                "cius_count",
                "cius_conversion_status",
                "cius_months_reported",
                "cius_quality_tier",
                "cius_observation_weight",
            ]
        ]
    )
    local_pub = (
        obs[obs["source"].eq(LOCAL_PUBLICATION_SOURCE)]
        .rename(
            columns={
                "count": "local_publication_count",
                "conversion_status": "local_publication_conversion_status",
                "months_reported": "local_publication_months_reported",
                "quality_tier": "local_publication_quality_tier",
                "observation_weight": "local_publication_observation_weight",
            }
        )[
            [
                "ori9",
                "year",
                "offense",
                "state_fips",
                "state_abbr",
                "agency_name_std",
                "agency_type_norm",
                "local_publication_count",
                "local_publication_conversion_status",
                "local_publication_months_reported",
                "local_publication_quality_tier",
                "local_publication_observation_weight",
            ]
        ]
    )
    state_pub = (
        obs[obs["source"].eq(STATE_PUBLICATION_SOURCE)]
        .rename(
            columns={
                "count": "state_publication_count",
                "conversion_status": "state_publication_conversion_status",
                "months_reported": "state_publication_months_reported",
                "quality_tier": "state_publication_quality_tier",
                "observation_weight": "state_publication_observation_weight",
            }
        )[
            [
                "ori9",
                "year",
                "offense",
                "state_fips",
                "state_abbr",
                "agency_name_std",
                "agency_type_norm",
                "state_publication_count",
                "state_publication_conversion_status",
                "state_publication_months_reported",
                "state_publication_quality_tier",
                "state_publication_observation_weight",
            ]
        ]
    )
    srs = (
        obs[obs["source"].eq(SUMMARY_SOURCE)]
        .rename(
            columns={
                "count": "srs_count",
                "conversion_status": "srs_conversion_status",
                "months_reported": "srs_months_reported",
                "months_missing": "srs_months_missing",
                "monthly_lumpiness_flag": "srs_monthly_lumpiness_flag",
                "quality_tier": "srs_quality_tier",
                "observation_weight": "srs_observation_weight",
                "reported_months_original": "srs_reported_months_original",
                "annual_batch_detected": "srs_annual_batch_detected",
                "annual_batch_detector_reason": "srs_annual_batch_detector_reason",
                "annual_batch_panel_median_full_year_total": "srs_annual_batch_panel_median_full_year_total",
                "annual_batch_panel_max_full_year_total": "srs_annual_batch_panel_max_full_year_total",
                "annual_batch_absolute_total_flag": "srs_annual_batch_absolute_total_flag",
                "annual_batch_panel_median_flag": "srs_annual_batch_panel_median_flag",
                "nonzero_months": "srs_nonzero_months_any_part1",
                "max_month_share": "srs_max_month_share",
                "annual_month_diff_ratio": "srs_annual_month_diff_ratio",
                "annual_part1_total": "srs_annual_part1_total",
            }
        )[
            [
                "ori9",
                "year",
                "offense",
                "state_fips",
                "state_abbr",
                "agency_name_std",
                "agency_type_norm",
                "srs_count",
                "srs_conversion_status",
                "srs_months_reported",
                "srs_months_missing",
                "srs_monthly_lumpiness_flag",
                "srs_quality_tier",
                "srs_observation_weight",
                "srs_reported_months_original",
                "srs_annual_batch_detected",
                "srs_annual_batch_detector_reason",
                "srs_annual_batch_panel_median_full_year_total",
                "srs_annual_batch_panel_max_full_year_total",
                "srs_annual_batch_absolute_total_flag",
                "srs_annual_batch_panel_median_flag",
                "srs_nonzero_months_any_part1",
                "srs_max_month_share",
                "srs_annual_month_diff_ratio",
                "srs_annual_part1_total",
            ]
        ]
    )
    nibrs = (
        obs[obs["source"].eq(NIBRS_SOURCE)]
        .rename(
            columns={
                "count": "nibrs_count",
                "conversion_status": "nibrs_conversion_status",
                "months_reported": "nibrs_months_reported",
                "quality_tier": "nibrs_quality_tier",
                "observation_weight": "nibrs_observation_weight",
                "reported_months_original": "nibrs_reported_months_original",
                "annual_batch_detected": "nibrs_annual_batch_detected",
                "annual_batch_detector_reason": "nibrs_annual_batch_detector_reason",
                "annual_batch_panel_median_full_year_total": "nibrs_annual_batch_panel_median_full_year_total",
                "annual_batch_panel_max_full_year_total": "nibrs_annual_batch_panel_max_full_year_total",
                "annual_batch_absolute_total_flag": "nibrs_annual_batch_absolute_total_flag",
                "annual_batch_panel_median_flag": "nibrs_annual_batch_panel_median_flag",
            }
        )[
            [
                "ori9",
                "year",
                "offense",
                "state_fips",
                "state_abbr",
                "agency_name_std",
                "agency_type_norm",
                "nibrs_count",
                "nibrs_conversion_status",
                "nibrs_months_reported",
                "nibrs_quality_tier",
                "nibrs_observation_weight",
                "nibrs_reported_months_original",
                "nibrs_annual_batch_detected",
                "nibrs_annual_batch_detector_reason",
                "nibrs_annual_batch_panel_median_full_year_total",
                "nibrs_annual_batch_panel_max_full_year_total",
                "nibrs_annual_batch_absolute_total_flag",
                "nibrs_annual_batch_panel_median_flag",
            ]
        ]
    )

    merge_keys = ["ori9", "year", "offense", "state_fips", "state_abbr", "agency_name_std", "agency_type_norm"]
    base = (
        cius.merge(local_pub, on=merge_keys, how="outer")
        .merge(state_pub, on=merge_keys, how="outer")
        .merge(srs, on=merge_keys, how="outer")
        .merge(nibrs, on=merge_keys, how="outer")
    )
    month_detail = _load_srs_month_mask_detail(paths=paths, config=config)
    base = base.merge(month_detail, on=["ori9", "year", "offense"], how="left")

    base["state_fips"] = base["state_fips"].astype("string").str.zfill(2)
    base["state_abbr"] = base["state_abbr"].astype("string").str.upper()
    base["agency_name_std"] = base["agency_name_std"].astype("string")
    base["agency_type_norm"] = base["agency_type_norm"].astype("string")

    srs_count = pd.to_numeric(base["srs_count"], errors="coerce").fillna(0.0)
    srs_months = pd.to_numeric(base["srs_months_reported"], errors="coerce").fillna(0.0)
    nibrs_count = pd.to_numeric(base["nibrs_count"], errors="coerce").fillna(0.0)
    nibrs_months = pd.to_numeric(base["nibrs_months_reported"], errors="coerce").fillna(0.0)
    srs_has = base["srs_count"].notna()
    nibrs_has = base["nibrs_count"].notna()
    match_flag = (
        base["srs_non_missing_months"].isna()
        | (
            pd.to_numeric(base["srs_months_reported"], errors="coerce")
            == pd.to_numeric(base["srs_non_missing_months"], errors="coerce")
        )
    )
    base["srs_reported_months_match_flag"] = match_flag
    lumpy_flag = base["srs_monthly_lumpiness_flag"].eq(True)
    cius_has = base["cius_count"].notna()
    local_publication_has = base["local_publication_count"].notna()
    state_publication_has = base["state_publication_count"].notna()
    srs_batched_annual = (
        base["srs_conversion_status"].astype("string").eq("batched_annual")
        | base["srs_annual_batch_detected"].fillna(False).astype(bool)
    )

    base["reporting_regime"] = "structurally_missing_or_unreliable"
    base["regime_reason"] = "no_usable_srs_or_nibrs_support"
    base["preferred_source_by_regime"] = pd.NA
    base["regime_confidence"] = "medium"

    base.loc[cius_has, "reporting_regime"] = "annual_only_but_usable"
    base.loc[cius_has, "regime_reason"] = "cius_complete_reporter_reference"
    base.loc[cius_has, "preferred_source_by_regime"] = CIUS_SOURCE
    base.loc[cius_has, "regime_confidence"] = "high"

    local_publication_complete_mask = (
        local_publication_has
        & ~cius_has
        & pd.to_numeric(base["local_publication_months_reported"], errors="coerce").fillna(12).ge(12)
    )
    base.loc[local_publication_complete_mask, "reporting_regime"] = "annual_only_but_usable"
    base.loc[local_publication_complete_mask, "regime_reason"] = "local_publication_complete_reporter_reference"
    base.loc[local_publication_complete_mask, "preferred_source_by_regime"] = LOCAL_PUBLICATION_SOURCE
    base.loc[local_publication_complete_mask, "regime_confidence"] = "high"

    local_publication_partial_mask = (
        local_publication_has
        & ~cius_has
        & pd.to_numeric(base["local_publication_months_reported"], errors="coerce").between(1, 11, inclusive="both")
    )
    base.loc[local_publication_partial_mask, "reporting_regime"] = "true_partial"
    base.loc[local_publication_partial_mask, "regime_reason"] = "local_publication_partial_reference"
    base.loc[local_publication_partial_mask, "preferred_source_by_regime"] = LOCAL_PUBLICATION_SOURCE
    base.loc[local_publication_partial_mask, "regime_confidence"] = "high"

    state_publication_complete_mask = (
        state_publication_has
        & ~cius_has
        & ~local_publication_has
        & pd.to_numeric(base["state_publication_months_reported"], errors="coerce").fillna(12).ge(12)
    )
    base.loc[state_publication_complete_mask, "reporting_regime"] = "annual_only_but_usable"
    base.loc[state_publication_complete_mask, "regime_reason"] = "state_publication_complete_reporter_reference"
    base.loc[state_publication_complete_mask, "preferred_source_by_regime"] = STATE_PUBLICATION_SOURCE
    base.loc[state_publication_complete_mask, "regime_confidence"] = "high"

    state_publication_partial_mask = (
        state_publication_has
        & ~cius_has
        & ~local_publication_has
        & pd.to_numeric(base["state_publication_months_reported"], errors="coerce").between(1, 11, inclusive="both")
    )
    base.loc[state_publication_partial_mask, "reporting_regime"] = "true_partial"
    base.loc[state_publication_partial_mask, "regime_reason"] = "state_publication_partial_quarter_ytd"
    base.loc[state_publication_partial_mask, "preferred_source_by_regime"] = STATE_PUBLICATION_SOURCE
    base.loc[state_publication_partial_mask, "regime_confidence"] = "high"

    srs_annual_batch_mask = srs_has & srs_batched_annual
    srs_annual_batch_mask = srs_annual_batch_mask & ~cius_has & ~local_publication_has & ~state_publication_has
    base.loc[srs_annual_batch_mask, "reporting_regime"] = "annual_only_but_usable"
    base.loc[srs_annual_batch_mask, "regime_reason"] = "srs_annual_batch_detector"
    base.loc[srs_annual_batch_mask, "preferred_source_by_regime"] = SUMMARY_SOURCE
    base.loc[srs_annual_batch_mask, "regime_confidence"] = "high"

    full_mask = srs_has & srs_months.ge(12) & ~lumpy_flag & (match_flag | base["srs_non_missing_months"].isna())
    full_mask = full_mask & ~srs_batched_annual & ~cius_has & ~local_publication_has & ~state_publication_has
    base.loc[full_mask, "reporting_regime"] = "full_monthly"
    base.loc[full_mask, "regime_reason"] = "srs_12_month_nonlumpy"
    base.loc[full_mask, "preferred_source_by_regime"] = SUMMARY_SOURCE
    base.loc[full_mask, "regime_confidence"] = "high"

    true_partial_mask = (
        srs_has
        & srs_months.between(1, 11, inclusive="both")
        & ~lumpy_flag
        & match_flag
    )
    true_partial_mask = true_partial_mask & ~cius_has & ~local_publication_has & ~state_publication_has
    base.loc[true_partial_mask, "reporting_regime"] = "true_partial"
    base.loc[true_partial_mask, "regime_reason"] = "srs_partial_month_set_matches_metadata"
    base.loc[true_partial_mask, "preferred_source_by_regime"] = SUMMARY_SOURCE
    base.loc[true_partial_mask, "regime_confidence"] = "high"

    lumpy_mask = (
        srs_has
        & (
            lumpy_flag
            | (
                srs_months.between(1, 11, inclusive="both")
                & base["srs_non_missing_months"].notna()
                & ~match_flag
            )
        )
    )
    lumpy_mask = lumpy_mask & ~cius_has & ~local_publication_has & ~state_publication_has
    base.loc[lumpy_mask, "reporting_regime"] = "lumpy_or_batched"
    base.loc[lumpy_mask, "regime_reason"] = np.where(
        lumpy_flag.loc[lumpy_mask],
        "srs_lumpiness_signal",
        "srs_month_mask_mismatch",
    )
    base.loc[lumpy_mask, "preferred_source_by_regime"] = SUMMARY_SOURCE
    base.loc[lumpy_mask, "regime_confidence"] = "high"

    unassigned_mask = base["reporting_regime"].eq("structurally_missing_or_unreliable")
    annual_usable_srs_mask = (
        unassigned_mask
        & srs_has
        & srs_count.gt(0)
    )
    base.loc[annual_usable_srs_mask, "reporting_regime"] = "annual_only_but_usable"
    base.loc[annual_usable_srs_mask, "regime_reason"] = "srs_annual_observation_without_reliable_monthly_support"
    base.loc[annual_usable_srs_mask, "preferred_source_by_regime"] = SUMMARY_SOURCE
    base.loc[annual_usable_srs_mask, "regime_confidence"] = "medium"

    unassigned_mask = base["reporting_regime"].eq("structurally_missing_or_unreliable")
    # Months, not count: a NIBRS agency-year that reported is a report for every Part I
    # offense it covers, and an offense with no incidents in it is a zero, not an absence.
    # Gating this rung on a positive count sent exactly those true zeros to the fill
    # ladder -- the same zero-vs-missing misread the panel now resolves at emission.
    # The rung splits on coverage, because a partial year is a partial year whichever
    # lane reports it: the batch header's months are the agency-year's coverage, so
    # 1-11 of them is the same `true_partial` the SRS lane has always carried and
    # licenses the same x12/months annualisation. Reading them as complete years at
    # their partial value understated 8,006 rows across 1,410 agencies in 2024.
    usable_nibrs_mask = unassigned_mask & nibrs_has & nibrs_months.gt(0)
    nibrs_partial_mask = usable_nibrs_mask & nibrs_months.between(1, 11, inclusive="both")
    annual_usable_nibrs_mask = usable_nibrs_mask & ~nibrs_partial_mask
    base.loc[annual_usable_nibrs_mask, "reporting_regime"] = "annual_only_but_usable"
    base.loc[annual_usable_nibrs_mask, "regime_reason"] = "nibrs_annual_support_when_srs_not_usable"
    base.loc[annual_usable_nibrs_mask, "preferred_source_by_regime"] = NIBRS_SOURCE
    base.loc[annual_usable_nibrs_mask, "regime_confidence"] = "medium"
    base.loc[nibrs_partial_mask, "reporting_regime"] = "true_partial"
    base.loc[nibrs_partial_mask, "regime_reason"] = "nibrs_batch_header_partial_year"
    base.loc[nibrs_partial_mask, "preferred_source_by_regime"] = NIBRS_SOURCE
    base.loc[nibrs_partial_mask, "regime_confidence"] = "medium"

    structural_missing_mask = (
        base["reporting_regime"].eq("structurally_missing_or_unreliable")
        & (
            (~cius_has | pd.to_numeric(base["cius_count"], errors="coerce").fillna(0.0).eq(0))
            & (~local_publication_has | pd.to_numeric(base["local_publication_count"], errors="coerce").fillna(0.0).eq(0))
            & (~srs_has | srs_count.eq(0))
            & (~nibrs_has | nibrs_count.eq(0))
        )
    )
    base.loc[structural_missing_mask, "reporting_regime"] = "structurally_missing_or_unreliable"
    base.loc[structural_missing_mask, "regime_reason"] = "no_positive_srs_or_nibrs_observation"
    base.loc[structural_missing_mask, "preferred_source_by_regime"] = pd.NA
    base.loc[structural_missing_mask, "regime_confidence"] = "high"

    base["override_applied"] = False
    base["override_evidence_type"] = pd.NA
    base["override_source_note"] = pd.NA
    base["override_reviewer_note"] = pd.NA
    base["source_override_applied"] = False
    base["source_override_evidence_type"] = pd.NA
    base["source_override_source_note"] = pd.NA
    base["source_override_reviewer_note"] = pd.NA

    overrides = _load_reporting_regime_overrides(config.override_path)
    if not overrides.empty:
        base = base.merge(overrides, on=["ori9", "year", "offense"], how="left")
        has_override = base["final_reporting_regime"].notna()
        base.loc[has_override, "reporting_regime"] = base.loc[has_override, "final_reporting_regime"]
        base.loc[has_override, "regime_reason"] = "manual_override"
        base.loc[has_override, "override_applied"] = True
        base.loc[has_override, "override_evidence_type"] = base.loc[has_override, "evidence_type"]
        base.loc[has_override, "override_source_note"] = base.loc[has_override, "source_note"]
        base.loc[has_override, "override_reviewer_note"] = base.loc[has_override, "reviewer_note"]
        override_conf = pd.to_numeric(base["confidence"], errors="coerce")
        base.loc[has_override, "regime_confidence"] = np.where(
            override_conf.loc[has_override].ge(0.85),
            "high",
            "medium",
        )
        override_pref = np.where(
            base.loc[has_override, "reporting_regime"].eq("structurally_missing_or_unreliable"),
            pd.NA,
            np.where(
                base.loc[has_override, "cius_count"].notna(),
                CIUS_SOURCE,
                np.where(
                    base.loc[has_override, "local_publication_count"].notna(),
                    LOCAL_PUBLICATION_SOURCE,
                    np.where(
                        base.loc[has_override, "state_publication_count"].notna(),
                        STATE_PUBLICATION_SOURCE,
                        np.where(
                            srs_count.loc[has_override].gt(0),
                            SUMMARY_SOURCE,
                            np.where(nibrs_count.loc[has_override].gt(0), NIBRS_SOURCE, pd.NA),
                        ),
                    ),
                ),
            ),
        )
        base.loc[has_override, "preferred_source_by_regime"] = override_pref
        base = base.drop(columns=["final_reporting_regime", "evidence_type", "source_note", "confidence", "reviewer_note"])

    source_overrides = _load_source_preference_overrides(config.source_override_path)
    if not source_overrides.empty:
        base = base.merge(
            source_overrides.rename(
                columns={
                    "preferred_source": "override_preferred_source",
                    "evidence_type": "source_override_evidence_type_in",
                    "source_note": "source_override_source_note_in",
                    "confidence": "source_override_confidence_in",
                    "reviewer_note": "source_override_reviewer_note_in",
                }
            ),
            on=["ori9", "year", "offense"],
            how="left",
        )
        has_source_override = base["override_preferred_source"].notna()
        base.loc[has_source_override, "preferred_source_by_regime"] = base.loc[
            has_source_override, "override_preferred_source"
        ].astype("string")
        base.loc[has_source_override, "source_override_applied"] = True
        base.loc[has_source_override, "source_override_evidence_type"] = base.loc[
            has_source_override, "source_override_evidence_type_in"
        ]
        base.loc[has_source_override, "source_override_source_note"] = base.loc[
            has_source_override, "source_override_source_note_in"
        ]
        base.loc[has_source_override, "source_override_reviewer_note"] = base.loc[
            has_source_override, "source_override_reviewer_note_in"
        ]
        base = base.drop(
            columns=[
                "override_preferred_source",
                "source_override_evidence_type_in",
                "source_override_source_note_in",
                "source_override_confidence_in",
                "source_override_reviewer_note_in",
            ]
        )

    cius_preferred_mask = base["preferred_source_by_regime"].eq(CIUS_SOURCE)
    cius_preferred_count_present = base.loc[cius_preferred_mask, "cius_count"].notna()
    base.loc[cius_preferred_mask, "reporting_regime"] = "annual_only_but_usable"
    base.loc[cius_preferred_mask, "regime_reason"] = "cius_complete_reporter_reference"
    base.loc[cius_preferred_mask, "regime_confidence"] = "high"
    base.loc[cius_preferred_count_present.index[~cius_preferred_count_present], "reporting_regime"] = (
        "structurally_missing_or_unreliable"
    )
    base.loc[cius_preferred_count_present.index[~cius_preferred_count_present], "regime_reason"] = (
        "no_usable_cius_support"
    )

    local_publication_preferred_mask = base["preferred_source_by_regime"].eq(LOCAL_PUBLICATION_SOURCE)
    local_publication_preferred_count_present = base.loc[local_publication_preferred_mask, "local_publication_count"].notna()
    local_publication_preferred_months = pd.to_numeric(
        base.loc[local_publication_preferred_mask, "local_publication_months_reported"],
        errors="coerce",
    ).fillna(12)
    local_publication_preferred_complete = local_publication_preferred_count_present & local_publication_preferred_months.ge(12)
    local_publication_preferred_partial = local_publication_preferred_count_present & local_publication_preferred_months.between(1, 11, inclusive="both")
    base.loc[local_publication_preferred_mask, "reporting_regime"] = "structurally_missing_or_unreliable"
    base.loc[local_publication_preferred_mask, "regime_reason"] = "no_usable_local_publication_support"
    base.loc[local_publication_preferred_mask, "regime_confidence"] = "high"
    base.loc[
        local_publication_preferred_complete.index[local_publication_preferred_complete],
        "reporting_regime",
    ] = "annual_only_but_usable"
    base.loc[
        local_publication_preferred_complete.index[local_publication_preferred_complete],
        "regime_reason",
    ] = "local_publication_complete_reporter_reference"
    base.loc[
        local_publication_preferred_partial.index[local_publication_preferred_partial],
        "reporting_regime",
    ] = "true_partial"
    base.loc[
        local_publication_preferred_partial.index[local_publication_preferred_partial],
        "regime_reason",
    ] = "local_publication_partial_reference"
    base.loc[local_publication_preferred_mask, "regime_confidence"] = "high"

    state_publication_preferred_mask = base["preferred_source_by_regime"].eq(STATE_PUBLICATION_SOURCE)
    state_publication_preferred_count_present = base.loc[state_publication_preferred_mask, "state_publication_count"].notna()
    state_publication_preferred_months = pd.to_numeric(
        base.loc[state_publication_preferred_mask, "state_publication_months_reported"],
        errors="coerce",
    ).fillna(12)
    state_publication_preferred_complete = state_publication_preferred_count_present & state_publication_preferred_months.ge(12)
    state_publication_preferred_partial = state_publication_preferred_count_present & state_publication_preferred_months.between(1, 11, inclusive="both")
    base.loc[state_publication_preferred_mask, "reporting_regime"] = "structurally_missing_or_unreliable"
    base.loc[state_publication_preferred_mask, "regime_reason"] = "no_usable_state_publication_support"
    base.loc[state_publication_preferred_mask, "regime_confidence"] = "high"
    base.loc[state_publication_preferred_complete.index[state_publication_preferred_complete], "reporting_regime"] = "annual_only_but_usable"
    base.loc[state_publication_preferred_complete.index[state_publication_preferred_complete], "regime_reason"] = "state_publication_complete_reporter_reference"
    base.loc[state_publication_preferred_partial.index[state_publication_preferred_partial], "reporting_regime"] = "true_partial"
    base.loc[state_publication_preferred_partial.index[state_publication_preferred_partial], "regime_reason"] = "state_publication_partial_quarter_ytd"
    base.loc[state_publication_preferred_mask, "regime_confidence"] = "high"

    # ---- Stage-1 adjudicated zero-vs-missing, applied last -------------------------------------
    # The rules above resolve zero-vs-missing at emission (a submitted NIBRS agency-year's empty
    # offense is a zero; an all-zero state publication behind zero SRS months is a non-report).
    # What no rule can settle is a full-year-backed published zero: the record says the agency
    # reported twelve months and counted nothing, and only the agency's own history, capacity and
    # population say whether that is a quiet year or an empty feed. Those cases were adjudicated one
    # by one, and a `misread_missing` verdict lands HERE -- the same statement the rule's
    # `structural_missing_mask` makes, about the cases the rule could not reach.
    #
    # It is applied after every rung, including the publication-lane blocks, because a reviewed
    # verdict is not one more rung competing with the others. `preferred_source_by_regime` is
    # cleared for the same reason the manual-override block clears it: a row still naming a lane
    # would be re-read as that lane's usable annual figure.
    adjudicated_zero_counts: dict[str, int] = {}
    zero_directives = build_zero_missing_directives(paths, target_year=int(config.year_end))
    unusable = zero_directives[zero_directives["directive"].eq("reads_missing")]
    if not unusable.empty:
        key = pd.MultiIndex.from_arrays(
            [base["ori9"].astype("string").str.upper(), pd.to_numeric(base["year"], errors="coerce")]
        )
        directive_index = pd.MultiIndex.from_arrays(
            [unusable["ori9"].astype("string").str.upper(), unusable["year"].astype(int)]
        )
        adjudicated = pd.Series(key.isin(directive_index), index=base.index)
        case_by_key = (
            unusable.assign(_k=list(zip(unusable["ori9"].str.upper(), unusable["year"].astype(int))))
            .set_index("_k")[["case_id", "verdict", "confidence"]]
        )
        base.loc[adjudicated, "reporting_regime"] = "structurally_missing_or_unreliable"
        base.loc[adjudicated, "regime_reason"] = STAGE1_ADJUDICATED_MISSING_REASON
        base.loc[adjudicated, "preferred_source_by_regime"] = pd.NA
        base.loc[adjudicated, "regime_confidence"] = "high"
        base.loc[adjudicated, "override_applied"] = True
        base.loc[adjudicated, "override_evidence_type"] = "stage1_adhoc_case_review"
        keys = pd.Index(list(zip(base.loc[adjudicated, "ori9"].astype("string").str.upper(),
                                 pd.to_numeric(base.loc[adjudicated, "year"], errors="coerce"))))
        base.loc[adjudicated, "override_source_note"] = (
            case_by_key["case_id"].reindex(keys).to_numpy()
        )
        base.loc[adjudicated, "override_reviewer_note"] = (
            case_by_key["verdict"].reindex(keys).to_numpy()
        )
        adjudicated_zero_counts = {
            "adjudicated_zero_rulings": int(len(zero_directives)),
            "adjudicated_zero_reads_missing_rulings": int(len(unusable)),
            "adjudicated_zero_rows_marked_missing": int(adjudicated.sum()),
            "adjudicated_zero_agency_years_marked_missing": int(
                base.loc[adjudicated, ["ori9", "year"]].drop_duplicates().shape[0]
            ),
            "adjudicated_zero_oris_not_in_panel": int(
                len(unusable)
                - base.loc[adjudicated, ["ori9", "year"]].drop_duplicates().shape[0]
            ),
        }
        assert_registry_bit(
            adjudicated_zero_counts,
            what="reporting regimes: Stage-1 adjudicated zero-vs-missing",
            expected_key="adjudicated_zero_rows_marked_missing",
        )
        print(
            "build_v2_reporting_regimes: Stage-1 adjudicated zeros -> missing on "
            f"{adjudicated_zero_counts['adjudicated_zero_agency_years_marked_missing']} agency-years "
            f"({adjudicated_zero_counts['adjudicated_zero_rows_marked_missing']} rows)",
            flush=True,
        )

    base["supports_partial_annualization"] = base["reporting_regime"].eq("true_partial")
    base["supports_as_reported_observation"] = base["reporting_regime"].isin(
        ["full_monthly", "lumpy_or_batched", "annual_only_but_usable"]
    )
    base["is_structurally_missing_or_unreliable"] = base["reporting_regime"].eq("structurally_missing_or_unreliable")
    base["support_weight"] = np.maximum(srs_count, nibrs_count)
    base["support_weight"] = np.maximum(base["support_weight"], pd.to_numeric(base["cius_count"], errors="coerce").fillna(0.0))
    base["support_weight"] = np.maximum(
        base["support_weight"], pd.to_numeric(base["local_publication_count"], errors="coerce").fillna(0.0)
    )
    base["support_weight"] = np.maximum(
        base["support_weight"], pd.to_numeric(base["state_publication_count"], errors="coerce").fillna(0.0)
    )
    base["support_weight"] = np.where(base["support_weight"] > 0, base["support_weight"], 1.0)

    cols = [
        "ori9",
        "year",
        "offense",
        "state_fips",
        "state_abbr",
        "agency_name_std",
        "agency_type_norm",
        "cius_count",
        "cius_conversion_status",
        "cius_months_reported",
        "cius_quality_tier",
        "cius_observation_weight",
        "local_publication_count",
        "local_publication_conversion_status",
        "local_publication_months_reported",
        "local_publication_quality_tier",
        "local_publication_observation_weight",
        "state_publication_count",
        "state_publication_conversion_status",
        "state_publication_months_reported",
        "state_publication_quality_tier",
        "state_publication_observation_weight",
        "srs_count",
        "srs_conversion_status",
        "srs_months_reported",
        "srs_months_missing",
        "srs_monthly_lumpiness_flag",
        "srs_quality_tier",
        "srs_observation_weight",
        "srs_reported_months_original",
        "srs_annual_batch_detected",
        "srs_annual_batch_detector_reason",
        "srs_annual_batch_panel_median_full_year_total",
        "srs_annual_batch_panel_max_full_year_total",
        "srs_annual_batch_absolute_total_flag",
        "srs_annual_batch_panel_median_flag",
        "srs_nonzero_months_any_part1",
        "srs_max_month_share",
        "srs_annual_month_diff_ratio",
        "srs_annual_part1_total",
        "srs_non_missing_months",
        "srs_nonzero_reported_months",
        "srs_reported_month_set",
        "srs_reported_months_match_flag",
        "nibrs_count",
        "nibrs_conversion_status",
        "nibrs_months_reported",
        "nibrs_quality_tier",
        "nibrs_observation_weight",
        "nibrs_reported_months_original",
        "nibrs_annual_batch_detected",
        "nibrs_annual_batch_detector_reason",
        "nibrs_annual_batch_panel_median_full_year_total",
        "nibrs_annual_batch_panel_max_full_year_total",
        "nibrs_annual_batch_absolute_total_flag",
        "nibrs_annual_batch_panel_median_flag",
        "reporting_regime",
        "regime_reason",
        "regime_confidence",
        "preferred_source_by_regime",
        "override_applied",
        "override_evidence_type",
        "override_source_note",
        "override_reviewer_note",
        "source_override_applied",
        "source_override_evidence_type",
        "source_override_source_note",
        "source_override_reviewer_note",
        "supports_partial_annualization",
        "supports_as_reported_observation",
        "is_structurally_missing_or_unreliable",
        "support_weight",
    ]
    out = base[cols].sort_values(["ori9", "year", "offense"], kind="mergesort").reset_index(drop=True)
    out.attrs["stage1_adjudications"] = adjudicated_zero_counts
    return out


def write_v2_reporting_regimes(
    *,
    paths: RepoPaths,
    out_path: Path,
    config: ReportingRegimeBuildConfig = ReportingRegimeBuildConfig(),
    blocked_by: tuple[str, ...] | None = None,
    observation_ignore_blockers: tuple[str, ...] = (),
) -> dict[str, int]:
    with stage_write_lock(paths=paths, stage="reporting_regimes", blocked_by=blocked_by):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        regimes = build_agency_year_reporting_regimes(
            paths=paths,
            config=config,
            observation_ignore_blockers=observation_ignore_blockers,
        )
        regimes.to_parquet(out_path, index=False)
        return {
            "rows": int(len(regimes)),
            "agency_years": int(regimes[["ori9", "year"]].drop_duplicates().shape[0]),
            "oris": int(regimes["ori9"].nunique()),
            **regimes.attrs.get("stage1_adjudications", {}),
        }
