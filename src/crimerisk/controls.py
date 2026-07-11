from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.build_freshness import artifact_is_current
from crimerisk.jurisdiction_estimator import (
    JurisdictionEstimatorConfig,
    build_jurisdiction_year_estimates,
)
from crimerisk.municipal_estimator import (
    MunicipalEstimatorConfig,
    get_v2_municipal_estimates_path,
    municipal_estimate_dependency_paths,
    municipal_estimates_artifact_is_current,
    write_v2_municipal_estimates,
)
from crimerisk.paths import RepoPaths
from crimerisk.published_nibrs import (
    build_published_nibrs_corroboration_mask,
)
from crimerisk.reporting_regimes import (
    ReportingRegimeBuildConfig,
    get_v2_reporting_regimes_path,
    reporting_regime_dependency_paths,
    reporting_regimes_artifact_is_current,
    write_v2_reporting_regimes,
)
from crimerisk.stage_locks import blockers_for_stage, stage_write_lock
from crimerisk.source_provenance import (
    CIUS_ORIGIN,
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    assign_preferred_value,
    build_prefer_nibrs_mask,
    default_conversion_status_from_source,
    initialize_preferred_source,
    raw_data_source_from_parts,
    reporting_mode_from_source,
    source_family_from_source,
    source_lane_from_source,
    source_origin_from_parts,
)
from crimerisk.source_selection import (
    build_agency_preferred_observations as build_agency_preferred_rows,
)


CDE_OFFENSE_MAP: dict[str, str] = {
    "murder": "homicide",
    "rape": "rape_revised",
    "robbery": "robbery",
    "aggravated_assault": "aggravated_assault",
    "burglary": "burglary",
    "larceny": "larceny",
    "motor_vehicle_theft": "motor_vehicle_theft",
}

PRODUCTION_SCOPE_EXCLUDE = {"AK", "AS", "CZ", "GU", "HI", "MP", "PR", "VI"}


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


@dataclass(frozen=True)
class SourceDiagnosticSpec:
    source: str
    rename_map: dict[str, str]


SOURCE_DIAGNOSTIC_SPECS = (
    SourceDiagnosticSpec(
        source=CIUS_SOURCE,
        rename_map={
            "observed_count": "reported_count_cius",
            "observation_weight": "observation_weight_cius",
            "conversion_status": "conversion_status_cius",
            "quality_tier": "quality_tier_cius",
            "mean_months_reported": "mean_months_reported_cius",
            "max_months_reported": "max_months_reported_cius",
            "any_monthly_lumpiness_flag": "any_monthly_lumpiness_flag_cius",
            "any_manual_review_flag": "any_manual_review_flag_cius",
            "any_crosswalk_review_flag": "any_crosswalk_review_flag_cius",
            "contributing_agencies": "contributing_agencies_cius",
            "relationship_type": "relationship_type_cius",
            "overlap_subtype": "overlap_subtype_cius",
        },
    ),
    SourceDiagnosticSpec(
        source=LOCAL_PUBLICATION_SOURCE,
        rename_map={
            "observed_count": "reported_count_local_publication",
            "observation_weight": "observation_weight_local_publication",
            "conversion_status": "conversion_status_local_publication",
            "quality_tier": "quality_tier_local_publication",
            "mean_months_reported": "mean_months_reported_local_publication",
            "max_months_reported": "max_months_reported_local_publication",
            "any_monthly_lumpiness_flag": "any_monthly_lumpiness_flag_local_publication",
            "any_manual_review_flag": "any_manual_review_flag_local_publication",
            "any_crosswalk_review_flag": "any_crosswalk_review_flag_local_publication",
            "contributing_agencies": "contributing_agencies_local_publication",
            "relationship_type": "relationship_type_local_publication",
            "overlap_subtype": "overlap_subtype_local_publication",
        },
    ),
    SourceDiagnosticSpec(
        source=STATE_PUBLICATION_SOURCE,
        rename_map={
            "observed_count": "reported_count_state_publication",
            "observation_weight": "observation_weight_state_publication",
            "conversion_status": "conversion_status_state_publication",
            "quality_tier": "quality_tier_state_publication",
            "mean_months_reported": "mean_months_reported_state_publication",
            "max_months_reported": "max_months_reported_state_publication",
            "any_monthly_lumpiness_flag": "any_monthly_lumpiness_flag_state_publication",
            "any_manual_review_flag": "any_manual_review_flag_state_publication",
            "any_crosswalk_review_flag": "any_crosswalk_review_flag_state_publication",
            "contributing_agencies": "contributing_agencies_state_publication",
            "relationship_type": "relationship_type_state_publication",
            "overlap_subtype": "overlap_subtype_state_publication",
        },
    ),
    SourceDiagnosticSpec(
        source=SUMMARY_SOURCE,
        rename_map={
            "observed_count": "reported_count_srs",
            "observation_weight": "observation_weight_srs",
            "conversion_status": "conversion_status_srs",
            "quality_tier": "quality_tier_srs",
            "mean_months_reported": "mean_months_reported_srs",
            "max_months_reported": "max_months_reported_srs",
            "any_monthly_lumpiness_flag": "any_monthly_lumpiness_flag_srs",
            "any_manual_review_flag": "any_manual_review_flag_srs",
            "any_crosswalk_review_flag": "any_crosswalk_review_flag_srs",
            "contributing_agencies": "contributing_agencies_srs",
            "relationship_type": "relationship_type_srs",
            "overlap_subtype": "overlap_subtype_srs",
        },
    ),
    SourceDiagnosticSpec(
        source=NIBRS_SOURCE,
        rename_map={
            "observed_count": "reported_count_nibrs",
            "observation_weight": "observation_weight_nibrs",
            "quality_tier": "quality_tier_nibrs",
            "mean_months_reported": "mean_months_reported_nibrs",
            "max_months_reported": "max_months_reported_nibrs",
            "any_monthly_lumpiness_flag": "any_monthly_lumpiness_flag_nibrs",
            "any_manual_review_flag": "any_manual_review_flag_nibrs",
            "any_crosswalk_review_flag": "any_crosswalk_review_flag_nibrs",
            "contributing_agencies": "contributing_agencies_nibrs",
            "relationship_type": "relationship_type_nibrs",
            "overlap_subtype": "overlap_subtype_nibrs",
        },
    ),
)


CONTROL_SOURCE_VALUE_COLUMNS = {
    "reported_count_preferred": {
        CIUS_SOURCE: "reported_count_cius",
        LOCAL_PUBLICATION_SOURCE: "reported_count_local_publication",
        STATE_PUBLICATION_SOURCE: "reported_count_state_publication",
        SUMMARY_SOURCE: "reported_count_srs",
        NIBRS_SOURCE: "reported_count_nibrs",
    },
    "observation_weight_preferred": {
        CIUS_SOURCE: "observation_weight_cius",
        LOCAL_PUBLICATION_SOURCE: "observation_weight_local_publication",
        STATE_PUBLICATION_SOURCE: "observation_weight_state_publication",
        SUMMARY_SOURCE: "observation_weight_srs",
        NIBRS_SOURCE: "observation_weight_nibrs",
    },
    "quality_tier_preferred": {
        CIUS_SOURCE: "quality_tier_cius",
        LOCAL_PUBLICATION_SOURCE: "quality_tier_local_publication",
        STATE_PUBLICATION_SOURCE: "quality_tier_state_publication",
        SUMMARY_SOURCE: "quality_tier_srs",
        NIBRS_SOURCE: "quality_tier_nibrs",
    },
    "mean_months_reported_preferred": {
        CIUS_SOURCE: "mean_months_reported_cius",
        LOCAL_PUBLICATION_SOURCE: "mean_months_reported_local_publication",
        STATE_PUBLICATION_SOURCE: "mean_months_reported_state_publication",
        SUMMARY_SOURCE: "mean_months_reported_srs",
        NIBRS_SOURCE: "mean_months_reported_nibrs",
    },
    "relationship_type_preferred": {
        CIUS_SOURCE: "relationship_type_cius",
        LOCAL_PUBLICATION_SOURCE: "relationship_type_local_publication",
        STATE_PUBLICATION_SOURCE: "relationship_type_state_publication",
        SUMMARY_SOURCE: "relationship_type_srs",
        NIBRS_SOURCE: "relationship_type_nibrs",
    },
    "overlap_subtype_preferred": {
        CIUS_SOURCE: "overlap_subtype_cius",
        LOCAL_PUBLICATION_SOURCE: "overlap_subtype_local_publication",
        STATE_PUBLICATION_SOURCE: "overlap_subtype_state_publication",
        SUMMARY_SOURCE: "overlap_subtype_srs",
        NIBRS_SOURCE: "overlap_subtype_nibrs",
    },
}


@dataclass(frozen=True)
class ControlBuildConfig:
    year: int = 2024
    exclude_scope_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE))
    force_reporting_regimes_rebuild: bool = False
    force_municipal_estimates_rebuild: bool = False


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


def _load_jurisdiction_year_observations(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "observations" / "jurisdiction_year_observations.parquet"
    return pd.read_parquet(path)


def _load_agency_year_observations(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    return pd.read_parquet(path)


def _load_crosswalk(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    crosswalk = pd.read_parquet(path)
    crosswalk = crosswalk.rename(columns={"ori": "ori9"})
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(
        0.0
    )
    return crosswalk


def _load_reporting_regimes(
    paths: RepoPaths,
    *,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    path = get_v2_reporting_regimes_path(paths)
    config = ReportingRegimeBuildConfig()
    if not force_rebuild and reporting_regimes_artifact_is_current(
        paths, config=config, out_path=path
    ):
        return pd.read_parquet(path)
    write_v2_reporting_regimes(paths=paths, out_path=path, config=config)
    return pd.read_parquet(path)


def _controls_dependency_paths(paths: RepoPaths, *, year: int) -> list[Path]:
    reporting_config = ReportingRegimeBuildConfig(year_start=2018, year_end=int(year))
    municipal_config = MunicipalEstimatorConfig(
        year_start=2018,
        year_end=int(year),
        target_year=int(year),
    )
    dependencies = [
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        paths.state_dir / "observations" / "jurisdiction_year_observations.parquet",
        paths.state_dir / "reference" / "agency_master.parquet",
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        get_v2_reporting_regimes_path(paths),
        get_v2_municipal_estimates_path(paths, year=year),
        _cde_estimates_path(paths, year=year),
        paths.data_dir
        / f"FBI-NIBRS-Tables-{int(year)}"
        / "parsed"
        / f"nibrs_offense_type_by_agency_{int(year)}.parquet",
        Path(__file__),
        paths.repo_root / "src" / "crimerisk" / "jurisdiction_estimator.py",
        paths.repo_root / "src" / "crimerisk" / "source_selection.py",
        paths.repo_root / "src" / "crimerisk" / "source_provenance.py",
        paths.repo_root / "src" / "crimerisk" / "trend_fills.py",
        *reporting_regime_dependency_paths(paths, config=reporting_config),
        *municipal_estimate_dependency_paths(paths, config=municipal_config),
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

    municipal_path = get_v2_municipal_estimates_path(paths, year=int(config.year))
    municipal_config = MunicipalEstimatorConfig(
        year_start=2018,
        year_end=int(config.year),
        target_year=int(config.year),
        force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
    )
    if (
        config.force_municipal_estimates_rebuild
        or not municipal_estimates_artifact_is_current(
            paths,
            config=municipal_config,
            out_path=municipal_path,
        )
    ):
        write_v2_municipal_estimates(
            paths=paths,
            out_path=municipal_path,
            config=municipal_config,
            blocked_by=blockers_for_stage("municipal_estimates", ignore=("controls",)),
            observation_ignore_blockers=("controls", *observation_ignore_blockers),
        )


def _load_cius_municipal_reference_counts(
    paths: RepoPaths,
    *,
    year: int,
) -> pd.DataFrame:
    columns = ["jurisdiction_id", "offense", "cius_municipal_official_count"]
    obs = _load_jurisdiction_year_observations(paths)
    obs = obs[
        obs["year"].astype(int).eq(int(year))
        & obs["jurisdiction_type"].eq("municipal")
        & obs["source"].eq(CIUS_SOURCE)
    ].copy()
    if obs.empty:
        return pd.DataFrame(columns=columns)
    out = obs[["jurisdiction_id", "offense", "observed_count"]].rename(
        columns={"observed_count": "cius_municipal_official_count"}
    )
    return out.drop_duplicates(
        subset=["jurisdiction_id", "offense"], keep="first"
    ).reset_index(drop=True)


def _summarize_jurisdiction_reporting_regimes(
    *,
    reporting_regimes: pd.DataFrame,
    crosswalk: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    regimes = reporting_regimes[
        reporting_regimes["year"].astype(int) == int(year)
    ].copy()
    if regimes.empty:
        return pd.DataFrame(
            columns=[
                "jurisdiction_id",
                "offense",
                "dominant_reporting_regime",
                "dominant_preferred_source_by_regime",
                "true_partial_support",
                "lumpy_or_batched_support",
                "annual_only_but_usable_support",
                "structurally_missing_or_unreliable_support",
            ]
        )

    regimes["support_weight"] = pd.to_numeric(
        regimes["support_weight"], errors="coerce"
    ).fillna(1.0)
    merged = regimes.merge(
        crosswalk[["ori9", "jurisdiction_id", "weight"]], on="ori9", how="inner"
    )
    merged["allocated_support"] = merged["support_weight"] * pd.to_numeric(
        merged["weight"], errors="coerce"
    ).fillna(1.0)

    support_by_regime = (
        merged.groupby(
            ["jurisdiction_id", "offense", "reporting_regime"], dropna=False
        )["allocated_support"]
        .sum()
        .reset_index()
    )
    dominant_regime = (
        support_by_regime.sort_values(
            ["jurisdiction_id", "offense", "allocated_support"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=["jurisdiction_id", "offense"], keep="first")
        .rename(columns={"reporting_regime": "dominant_reporting_regime"})[
            ["jurisdiction_id", "offense", "dominant_reporting_regime"]
        ]
    )

    support_by_pref = (
        merged.groupby(
            ["jurisdiction_id", "offense", "preferred_source_by_regime"], dropna=False
        )["allocated_support"]
        .sum()
        .reset_index()
    )
    dominant_pref = (
        support_by_pref.sort_values(
            ["jurisdiction_id", "offense", "allocated_support"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=["jurisdiction_id", "offense"], keep="first")
        .rename(
            columns={
                "preferred_source_by_regime": "dominant_preferred_source_by_regime"
            }
        )[["jurisdiction_id", "offense", "dominant_preferred_source_by_regime"]]
    )

    pivot = support_by_regime.pivot_table(
        index=["jurisdiction_id", "offense"],
        columns="reporting_regime",
        values="allocated_support",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for col in [
        "true_partial",
        "lumpy_or_batched",
        "annual_only_but_usable",
        "structurally_missing_or_unreliable",
        "full_monthly",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot.rename(
        columns={
            "true_partial": "true_partial_support",
            "lumpy_or_batched": "lumpy_or_batched_support",
            "annual_only_but_usable": "annual_only_but_usable_support",
            "structurally_missing_or_unreliable": "structurally_missing_or_unreliable_support",
            "full_monthly": "full_monthly_support",
        }
    )
    return dominant_regime.merge(
        dominant_pref, on=["jurisdiction_id", "offense"], how="left"
    ).merge(
        pivot,
        on=["jurisdiction_id", "offense"],
        how="left",
    )


def _summarize_jurisdiction_published_nibrs_support(
    *,
    agency_pref: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    panel = agency_pref
    if panel.empty or "published_nibrs_official_count" not in panel.columns:
        return pd.DataFrame(
            columns=[
                "jurisdiction_id",
                "offense",
                "published_nibrs_corroborated_count",
            ]
        )

    nibrs_months = pd.to_numeric(
        panel["nibrs_months_reported"], errors="coerce"
    ).fillna(pd.to_numeric(panel["mean_months_reported_nibrs"], errors="coerce"))

    corroborated = panel[
        build_published_nibrs_corroboration_mask(
            nibrs_count=panel["reported_count_nibrs"],
            published_nibrs_count=panel["published_nibrs_official_count"],
            nibrs_months=nibrs_months,
            srs_count=panel["reported_count_srs"],
        )
    ][["ori9", "offense", "published_nibrs_official_count"]].copy()
    if corroborated.empty:
        return pd.DataFrame(
            columns=[
                "jurisdiction_id",
                "offense",
                "published_nibrs_corroborated_count",
            ]
        )

    corroborated = corroborated.merge(
        crosswalk[["ori9", "jurisdiction_id", "weight"]],
        on="ori9",
        how="inner",
    )
    corroborated["published_nibrs_corroborated_count"] = pd.to_numeric(
        corroborated["published_nibrs_official_count"], errors="coerce"
    ).fillna(0.0) * pd.to_numeric(corroborated["weight"], errors="coerce").fillna(0.0)
    return (
        corroborated.groupby(["jurisdiction_id", "offense"], dropna=False)[
            "published_nibrs_corroborated_count"
        ]
        .sum()
        .reset_index()
    )


def _normalize_jurisdiction_estimate_slice(
    *,
    estimates: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    if estimates.empty:
        return estimates.copy()

    out = estimates.copy()
    if "year" in out.columns:
        out = out[out["year"].astype(int).eq(int(year))].copy()
        out = out.drop(columns=["year"], errors="ignore")

    year_suffix_map = {
        f"preferred_source_{int(year)}": "preferred_source",
        f"preferred_source_family_{int(year)}": "preferred_source_family",
        f"preferred_source_origin_{int(year)}": "preferred_source_origin",
        f"preferred_raw_data_source_{int(year)}": "preferred_raw_data_source",
        f"dominant_reporting_regime_{int(year)}": "dominant_reporting_regime",
        f"reported_count_preferred_{int(year)}": "reported_count_preferred",
        f"observation_weight_preferred_{int(year)}": "observation_weight_preferred",
        f"mean_months_reported_preferred_{int(year)}": "mean_months_reported_preferred",
        f"target_is_observed_{int(year)}": "usable_as_observed",
        f"estimated_count_{int(year)}": "estimated_count",
    }
    applicable = {
        old: new for old, new in year_suffix_map.items() if old in out.columns
    }
    if applicable:
        out = out.rename(columns=applicable)
    return out


def _source_diagnostic_frame(
    obs: pd.DataFrame, spec: SourceDiagnosticSpec
) -> pd.DataFrame:
    source_cols = list(spec.rename_map.values())
    return (
        obs[obs["source"].eq(spec.source)]
        .rename(columns=spec.rename_map)[CONTROL_KEY_COLUMNS + source_cols]
        .copy()
    )


def _build_jurisdiction_source_diagnostics(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig,
) -> pd.DataFrame:
    obs_all = _load_jurisdiction_year_observations(paths)
    obs_all["state_abbr"] = obs_all["state_abbr"].astype("string").str.upper()
    obs_all = obs_all[
        ~obs_all["state_abbr"].isin(set(config.exclude_scope_state_abbrs))
    ].copy()
    obs = obs_all[obs_all["year"].astype(int).eq(int(config.year))].copy()

    frames = [_source_diagnostic_frame(obs, spec) for spec in SOURCE_DIAGNOSTIC_SPECS]
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=CONTROL_KEY_COLUMNS, how="outer")
    return out


def _attach_jurisdiction_source_context(
    *,
    diagnostics: pd.DataFrame,
    paths: RepoPaths,
    config: ControlBuildConfig,
    agency_pref: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    reporting_regimes = _load_reporting_regimes(
        paths,
        force_rebuild=config.force_reporting_regimes_rebuild,
    )
    jurisdiction_regimes = _summarize_jurisdiction_reporting_regimes(
        reporting_regimes=reporting_regimes,
        crosswalk=crosswalk,
        year=config.year,
    )
    out = diagnostics.merge(
        jurisdiction_regimes, on=["jurisdiction_id", "offense"], how="left"
    )
    out = out.merge(
        _summarize_jurisdiction_published_nibrs_support(
            agency_pref=agency_pref,
            crosswalk=crosswalk,
        ),
        on=["jurisdiction_id", "offense"],
        how="left",
    )
    out = out.merge(
        _load_cius_municipal_reference_counts(paths, year=config.year),
        on=["jurisdiction_id", "offense"],
        how="left",
    )
    return out


def _build_jurisdiction_observation_fallback_controls(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    out = diagnostics.copy()
    has_cius = out["reported_count_cius"].notna()
    has_local_publication = out["reported_count_local_publication"].notna()
    has_state_publication = out["reported_count_state_publication"].notna()
    has_srs = out["reported_count_srs"].notna()
    has_nibrs = out["reported_count_nibrs"].notna()

    srs_obs_weight = pd.to_numeric(
        out["observation_weight_srs"], errors="coerce"
    ).fillna(0.0)
    nibrs_obs_weight = pd.to_numeric(
        out["observation_weight_nibrs"], errors="coerce"
    ).fillna(0.0)
    srs_months = pd.to_numeric(out["mean_months_reported_srs"], errors="coerce").fillna(
        0.0
    )
    nibrs_months = pd.to_numeric(
        out["mean_months_reported_nibrs"], errors="coerce"
    ).fillna(0.0)
    srs_count_num = pd.to_numeric(out["reported_count_srs"], errors="coerce").fillna(
        0.0
    )
    regime_prefers_nibrs = out["dominant_preferred_source_by_regime"].eq(NIBRS_SOURCE)
    srs_regime_inferior = out["dominant_reporting_regime"].isin(
        [
            "structurally_missing_or_unreliable",
            "lumpy_or_batched",
            "annual_only_but_usable",
        ]
    )
    nibrs_supports_better = nibrs_obs_weight.gt(srs_obs_weight) | (
        nibrs_obs_weight.eq(srs_obs_weight) & nibrs_months.gt(srs_months)
    )
    published_nibrs_supports_nibrs = (
        has_nibrs
        & build_published_nibrs_corroboration_mask(
            nibrs_count=out["reported_count_nibrs"],
            published_nibrs_count=out["published_nibrs_corroborated_count"],
            srs_count=out["reported_count_srs"],
            veto_mask=pd.to_numeric(
                out["cius_municipal_official_count"], errors="coerce"
            ).notna(),
        )
    )

    out["preferred_source"] = initialize_preferred_source(
        has_cius=has_cius,
        has_local_publication=has_local_publication,
        has_state_publication=has_state_publication,
        has_srs=has_srs,
        has_nibrs=has_nibrs,
        prefer_nibrs_mask=build_prefer_nibrs_mask(
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
            published_nibrs_supports_nibrs=published_nibrs_supports_nibrs,
        ),
    )
    for output_col, source_to_input_col in CONTROL_SOURCE_VALUE_COLUMNS.items():
        out = assign_preferred_value(
            out,
            output_col=output_col,
            preferred_source_col="preferred_source",
            source_to_input_col=source_to_input_col,
        )
    out["preferred_source_lane"] = (
        out["preferred_source"].map(source_lane_from_source).astype("string")
    )
    out["preferred_reporting_mode"] = (
        out["preferred_source"].map(reporting_mode_from_source).astype("string")
    )
    out["conversion_status_nibrs"] = default_conversion_status_from_source(NIBRS_SOURCE)
    out = assign_preferred_value(
        out,
        output_col="preferred_conversion_status",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "conversion_status_cius",
            LOCAL_PUBLICATION_SOURCE: "conversion_status_local_publication",
            STATE_PUBLICATION_SOURCE: "conversion_status_state_publication",
            SUMMARY_SOURCE: "conversion_status_srs",
            NIBRS_SOURCE: "conversion_status_nibrs",
        },
    )
    out = out.drop(columns=["conversion_status_nibrs"])
    out["preferred_state_exception_flag"] = out["preferred_source"].eq(
        STATE_PUBLICATION_SOURCE
    )
    out["preferred_cius_reference_flag"] = out["preferred_source"].eq(CIUS_SOURCE)
    return out


def _load_canonical_jurisdiction_panel(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig,
    jurisdiction_year_estimates: pd.DataFrame | None,
) -> pd.DataFrame:
    if jurisdiction_year_estimates is None:
        estimates = build_jurisdiction_year_estimates(
            paths=paths,
            config=JurisdictionEstimatorConfig(
                year_start=2018,
                year_end=int(config.year),
                target_year=int(config.year),
                force_reporting_regimes_rebuild=config.force_reporting_regimes_rebuild,
                force_municipal_estimates_rebuild=config.force_municipal_estimates_rebuild,
            ),
        )
    else:
        estimates = jurisdiction_year_estimates.copy()
    estimates = _normalize_jurisdiction_estimate_slice(
        estimates=estimates, year=int(config.year)
    )
    return estimates.rename(
        columns={
            "preferred_source": "panel_preferred_source",
            "preferred_source_family": "panel_preferred_source_family",
            "preferred_source_origin": "panel_preferred_source_origin",
            "preferred_raw_data_source": "panel_preferred_raw_data_source",
            "preferred_conversion_status": "panel_preferred_conversion_status",
            "dominant_reporting_regime": "panel_dominant_reporting_regime",
            "reported_count_preferred": "panel_reported_count_preferred",
            "observation_weight_preferred": "panel_observation_weight_preferred",
            "mean_months_reported_preferred": "panel_mean_months_reported_preferred",
            "usable_as_observed": "panel_target_is_observed",
            "estimated_count": "panel_estimated_count",
        }
    )


def _overlay_canonical_panel_controls(
    *,
    fallback_controls: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    out = fallback_controls.merge(
        panel,
        on=CONTROL_KEY_COLUMNS,
        how="left",
    )

    panel_count = pd.to_numeric(out["panel_reported_count_preferred"], errors="coerce")
    fallback_count = pd.to_numeric(out["reported_count_preferred"], errors="coerce")
    out["reported_count_preferred"] = panel_count.where(
        panel_count.notna(), fallback_count
    )

    panel_weight = pd.to_numeric(
        out["panel_observation_weight_preferred"], errors="coerce"
    )
    fallback_weight = pd.to_numeric(
        out["observation_weight_preferred"], errors="coerce"
    )
    out["observation_weight_preferred"] = panel_weight.where(
        panel_weight.notna(), fallback_weight
    )

    panel_months = pd.to_numeric(
        out["panel_mean_months_reported_preferred"], errors="coerce"
    )
    fallback_months = pd.to_numeric(
        out["mean_months_reported_preferred"], errors="coerce"
    )
    out["mean_months_reported_preferred"] = panel_months.where(
        panel_months.notna(), fallback_months
    )

    out["preferred_source"] = out["panel_preferred_source"].fillna(
        out["preferred_source"]
    )
    out["preferred_source_family"] = out["panel_preferred_source_family"].fillna(
        out["preferred_source"].map(source_family_from_source)
    )
    out["preferred_source_origin"] = out["panel_preferred_source_origin"].fillna(
        out["preferred_source"]
        .map(lambda source: source_origin_from_parts(source=source))
        .astype("string")
    )
    out["preferred_raw_data_source"] = out["panel_preferred_raw_data_source"].fillna(
        out.apply(
            lambda row: raw_data_source_from_parts(source=row["preferred_source"]),
            axis=1,
        )
    )
    out["preferred_source_lane"] = (
        out["preferred_source"].map(source_lane_from_source).astype("string")
    )
    out["preferred_reporting_mode"] = (
        out["preferred_source"].map(reporting_mode_from_source).astype("string")
    )
    out["preferred_conversion_status"] = out[
        "panel_preferred_conversion_status"
    ].fillna(
        out["preferred_source"]
        .map(default_conversion_status_from_source)
        .astype("string")
    )
    out["preferred_state_exception_flag"] = out["preferred_source"].eq(
        STATE_PUBLICATION_SOURCE
    )
    out["preferred_cius_reference_flag"] = out["preferred_source_origin"].eq(
        CIUS_ORIGIN
    )
    return out


def _finalize_jurisdiction_controls_from_panel(
    out: pd.DataFrame, *, config: ControlBuildConfig
) -> pd.DataFrame:
    reported_num = pd.to_numeric(
        out["reported_count_preferred"], errors="coerce"
    ).fillna(0.0)
    estimated_num = pd.to_numeric(out["panel_estimated_count"], errors="coerce").fillna(
        reported_num
    )
    out["adjusted_count_ags_core"] = estimated_num
    out["estimated_count_ags_core"] = estimated_num
    out["estimated_from_panel"] = (
        out["estimated_from_panel"].astype("boolean").fillna(False).astype(bool)
    )

    dominant_regime = (
        out["panel_dominant_reporting_regime"]
        .fillna(out["dominant_reporting_regime"])
        .astype("string")
        .fillna("")
    )
    preferred_source = out["preferred_source"].astype("string").fillna("")
    out["needs_partial_reporting_uplift"] = (
        (
            preferred_source.isin(
                [LOCAL_PUBLICATION_SOURCE, SUMMARY_SOURCE, STATE_PUBLICATION_SOURCE]
            )
            & dominant_regime.eq("true_partial")
            & pd.to_numeric(out["mean_months_reported_preferred"], errors="coerce")
            .fillna(12)
            .between(1, 11, inclusive="both")
            & reported_num.gt(0)
        )
        .fillna(False)
        .astype(bool)
    )
    out["partial_reporting_uplift_count"] = np.where(
        out["needs_partial_reporting_uplift"],
        (estimated_num - reported_num).clip(lower=0.0),
        0.0,
    )
    out["needs_zero_month_fill"] = reported_num.le(0) & estimated_num.gt(0)
    out["current_year_fill_count"] = (
        estimated_num
        - reported_num
        - pd.to_numeric(out["partial_reporting_uplift_count"], errors="coerce").fillna(
            0.0
        )
    ).clip(lower=0.0)
    out["adjustment_total"] = (estimated_num - reported_num).clip(lower=0.0)
    out["needs_current_year_fill"] = (
        out["estimate_confidence"].astype("string").eq("low")
    )
    out["dominant_reporting_regime"] = out["panel_dominant_reporting_regime"].fillna(
        out["dominant_reporting_regime"]
    )
    out["year"] = int(config.year)
    return out.drop(
        columns=[
            "panel_preferred_source",
            "panel_preferred_source_family",
            "panel_preferred_source_origin",
            "panel_preferred_raw_data_source",
            "panel_dominant_reporting_regime",
            "panel_reported_count_preferred",
            "panel_observation_weight_preferred",
            "panel_mean_months_reported_preferred",
            "panel_target_is_observed",
            "panel_estimated_count",
        ],
        errors="ignore",
    )


def build_jurisdiction_controls(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig = ControlBuildConfig(),
    jurisdiction_year_estimates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    diagnostics = _build_jurisdiction_source_diagnostics(paths=paths, config=config)
    crosswalk = _load_crosswalk(paths)
    agency_pref = build_agency_preferred_rows(
        paths=paths,
        year=config.year,
        force_reporting_regimes_rebuild=config.force_reporting_regimes_rebuild,
    )
    diagnostics = _attach_jurisdiction_source_context(
        diagnostics=diagnostics,
        paths=paths,
        config=config,
        agency_pref=agency_pref,
        crosswalk=crosswalk,
    )
    fallback_controls = _build_jurisdiction_observation_fallback_controls(diagnostics)
    panel = _load_canonical_jurisdiction_panel(
        paths=paths,
        config=config,
        jurisdiction_year_estimates=jurisdiction_year_estimates,
    )
    out = _overlay_canonical_panel_controls(
        fallback_controls=fallback_controls,
        panel=panel,
    )
    out = _finalize_jurisdiction_controls_from_panel(out, config=config)

    return out.sort_values(
        ["state_fips", "jurisdiction_type", "jurisdiction_id", "offense"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_state_control_comparison(
    *,
    paths: RepoPaths,
    config: ControlBuildConfig = ControlBuildConfig(),
    controls: pd.DataFrame | None = None,
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
            ]
        ].rename(
            columns={
                "preferred_total": f"{prefix}_reported_total",
                "adjusted_total": f"{prefix}_adjusted_total",
                "srs_total": f"{prefix}_srs_total",
                "nibrs_total": f"{prefix}_nibrs_total",
                "partial_reporting_uplift_total": f"{prefix}_partial_reporting_uplift_total",
                "current_year_fill_total": f"{prefix}_current_year_fill_total",
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
                "zero_month_pool_total",
                "municipal_partial_reporting_uplift_total",
                "municipal_current_year_fill_total",
                "nonmunicipal_partial_reporting_uplift_total",
                "nonmunicipal_current_year_fill_total",
                "overlap_partial_reporting_uplift_total",
                "overlap_current_year_fill_total",
                "partial_pool_share_of_reported",
                "fbi_cde_estimated_total",
                "cde_gap_to_reported",
                "cde_gap_to_adjusted",
                "cde_gap_positive_only",
                "cde_gap_positive_only_adjusted",
                "reported_to_cde_ratio",
                "adjusted_to_cde_ratio",
                "gap_to_partial_pool_ratio",
            ]
        ]
        .sort_values(["state_fips", "offense"], kind="mergesort")
        .reset_index(drop=True)
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
        jurisdiction_year_estimates = build_jurisdiction_year_estimates(
            paths=paths,
            config=JurisdictionEstimatorConfig(
                year_start=2018,
                year_end=int(config.year),
                target_year=int(config.year),
                force_reporting_regimes_rebuild=config.force_reporting_regimes_rebuild,
                force_municipal_estimates_rebuild=config.force_municipal_estimates_rebuild,
            ),
        )
        jurisdiction_controls = build_jurisdiction_controls(
            paths=paths,
            config=config,
            jurisdiction_year_estimates=jurisdiction_year_estimates,
        )
        state_controls = build_state_control_comparison(
            paths=paths, config=config, controls=jurisdiction_controls
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
