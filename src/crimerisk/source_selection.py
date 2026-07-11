from __future__ import annotations

import pandas as pd

from crimerisk.paths import RepoPaths
from crimerisk.published_nibrs import (
    build_published_nibrs_corroboration_mask,
    load_published_nibrs_reference_counts,
)
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    CIUS_ORIGIN,
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
from crimerisk.reporting_regimes import (
    ReportingRegimeBuildConfig,
    get_v2_reporting_regimes_path,
    reporting_regimes_artifact_is_current,
    write_v2_reporting_regimes,
)


def _normalize_optional_panel_schema(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    numeric_cols = [
        "reported_count_local_publication",
        "observation_weight_local_publication",
        "mean_months_reported_local_publication",
        "reported_count_state_publication",
        "observation_weight_state_publication",
        "mean_months_reported_state_publication",
        "published_nibrs_official_count",
    ]
    string_cols = [
        "conversion_status_local_publication",
        "conversion_status_state_publication",
        "published_nibrs_agency_type",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in string_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def _load_agency_year_observations(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    return pd.read_parquet(path)


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


def _globally_dead_observation_oris(agency_obs: pd.DataFrame) -> set[str]:
    if agency_obs.empty:
        return set()
    stats = agency_obs[["ori9", "count", "months_reported"]].copy()
    stats["count"] = pd.to_numeric(stats["count"], errors="coerce").fillna(0.0)
    stats["months_reported"] = pd.to_numeric(
        stats["months_reported"], errors="coerce"
    ).fillna(0.0)
    stats = (
        stats.groupby("ori9", dropna=False)
        .agg(total_count=("count", "sum"), max_months=("months_reported", "max"))
        .reset_index()
    )
    dead = stats[
        stats["ori9"].notna()
        & stats["total_count"].le(0.0)
        & stats["max_months"].le(0.0)
    ]["ori9"].astype(str)
    return set(dead.tolist())


def build_agency_preferred_observations(
    *,
    paths: RepoPaths,
    year: int,
    force_reporting_regimes_rebuild: bool = False,
) -> pd.DataFrame:
    all_agency_obs = _load_agency_year_observations(paths)
    dead_oris = _globally_dead_observation_oris(all_agency_obs)
    agency_obs = all_agency_obs[
        (all_agency_obs["year"].astype(int) == int(year))
        & all_agency_obs["source"].isin(
            [
                CIUS_SOURCE,
                LOCAL_PUBLICATION_SOURCE,
                STATE_PUBLICATION_SOURCE,
                SUMMARY_SOURCE,
                NIBRS_SOURCE,
            ]
        )
    ].copy()
    if dead_oris:
        agency_obs = agency_obs[
            ~agency_obs["ori9"].astype("string").isin(sorted(dead_oris))
        ].copy()
    if agency_obs.empty:
        return pd.DataFrame(
            columns=[
                "ori9",
                "state_fips",
                "state_abbr",
                "offense",
                "preferred_source",
                "preferred_source_family",
                "preferred_source_origin",
                "preferred_raw_data_source",
                "preferred_count",
                "preferred_observation_weight",
                "preferred_months_reported",
                "preferred_source_lane",
                "preferred_reporting_mode",
                "preferred_conversion_status",
                "preferred_state_exception_flag",
                "preferred_cius_reference_flag",
                "reporting_regime",
                "preferred_source_by_regime",
                "reported_count_local_publication",
                "reported_count_srs",
                "reported_count_nibrs",
                "mean_months_reported_local_publication",
                "mean_months_reported_srs",
                "mean_months_reported_nibrs",
            ]
        )

    agency_obs["state_fips"] = agency_obs["state_fips"].astype("string").str.zfill(2)
    agency_obs["state_abbr"] = agency_obs["state_abbr"].astype("string").str.upper()
    key_cols = ["ori9", "state_fips", "state_abbr", "offense"]

    srs = agency_obs[agency_obs["source"].eq(SUMMARY_SOURCE)].rename(
        columns={
            "count": "reported_count_srs",
            "observation_weight": "observation_weight_srs",
            "months_reported": "mean_months_reported_srs",
            "conversion_status": "conversion_status_srs",
            "cius_reference_flag": "cius_reference_flag_summary",
        }
    )[
        key_cols
        + [
            "reported_count_srs",
            "observation_weight_srs",
            "mean_months_reported_srs",
            "conversion_status_srs",
            "cius_reference_flag_summary",
        ]
    ]
    cius = agency_obs[agency_obs["source"].eq(CIUS_SOURCE)].rename(
        columns={
            "count": "reported_count_cius",
            "observation_weight": "observation_weight_cius",
            "months_reported": "mean_months_reported_cius",
            "conversion_status": "conversion_status_cius",
            "cius_reference_flag": "cius_reference_flag_cius",
        }
    )[
        key_cols
        + [
            "reported_count_cius",
            "observation_weight_cius",
            "mean_months_reported_cius",
            "conversion_status_cius",
            "cius_reference_flag_cius",
        ]
    ]
    local_pub = agency_obs[agency_obs["source"].eq(LOCAL_PUBLICATION_SOURCE)].rename(
        columns={
            "count": "reported_count_local_publication",
            "observation_weight": "observation_weight_local_publication",
            "months_reported": "mean_months_reported_local_publication",
            "conversion_status": "conversion_status_local_publication",
        }
    )[
        key_cols
        + [
            "reported_count_local_publication",
            "observation_weight_local_publication",
            "mean_months_reported_local_publication",
            "conversion_status_local_publication",
        ]
    ]
    state_pub = agency_obs[agency_obs["source"].eq(STATE_PUBLICATION_SOURCE)].rename(
        columns={
            "count": "reported_count_state_publication",
            "observation_weight": "observation_weight_state_publication",
            "months_reported": "mean_months_reported_state_publication",
            "conversion_status": "conversion_status_state_publication",
        }
    )[
        key_cols
        + [
            "reported_count_state_publication",
            "observation_weight_state_publication",
            "mean_months_reported_state_publication",
            "conversion_status_state_publication",
        ]
    ]
    nibrs = agency_obs[agency_obs["source"].eq(NIBRS_SOURCE)].rename(
        columns={
            "count": "reported_count_nibrs",
            "observation_weight": "observation_weight_nibrs",
            "months_reported": "mean_months_reported_nibrs",
        }
    )[
        key_cols
        + [
            "reported_count_nibrs",
            "observation_weight_nibrs",
            "mean_months_reported_nibrs",
        ]
    ]
    panel = (
        cius.merge(local_pub, on=key_cols, how="outer")
        .merge(state_pub, on=key_cols, how="outer")
        .merge(srs, on=key_cols, how="outer")
        .merge(nibrs, on=key_cols, how="outer")
    )

    reporting_regimes = _load_reporting_regimes(
        paths,
        force_rebuild=force_reporting_regimes_rebuild,
    )
    regimes = reporting_regimes[reporting_regimes["year"].astype(int).eq(int(year))][
        [
            "ori9",
            "offense",
            "reporting_regime",
            "preferred_source_by_regime",
            "srs_months_reported",
            "nibrs_months_reported",
            "srs_observation_weight",
            "nibrs_observation_weight",
            "source_override_applied",
        ]
    ].copy()
    panel = panel.merge(regimes, on=["ori9", "offense"], how="left")
    panel = panel.merge(
        load_published_nibrs_reference_counts(paths, year=year),
        on=["ori9", "state_abbr", "offense"],
        how="left",
    )
    panel = _normalize_optional_panel_schema(panel)

    has_cius = panel["reported_count_cius"].notna()
    has_local_publication = panel["reported_count_local_publication"].notna()
    has_state_publication = panel["reported_count_state_publication"].notna()
    has_srs = panel["reported_count_srs"].notna()
    has_nibrs = panel["reported_count_nibrs"].notna()
    srs_obs_weight = pd.to_numeric(
        panel["srs_observation_weight"], errors="coerce"
    ).fillna(
        pd.to_numeric(panel["observation_weight_srs"], errors="coerce").fillna(0.0)
    )
    nibrs_obs_weight = pd.to_numeric(
        panel["nibrs_observation_weight"], errors="coerce"
    ).fillna(
        pd.to_numeric(panel["observation_weight_nibrs"], errors="coerce").fillna(0.0)
    )
    srs_months = pd.to_numeric(panel["srs_months_reported"], errors="coerce").fillna(
        pd.to_numeric(panel["mean_months_reported_srs"], errors="coerce").fillna(0.0)
    )
    nibrs_months = pd.to_numeric(
        panel["nibrs_months_reported"], errors="coerce"
    ).fillna(
        pd.to_numeric(panel["mean_months_reported_nibrs"], errors="coerce").fillna(0.0)
    )
    srs_count_num = pd.to_numeric(panel["reported_count_srs"], errors="coerce").fillna(
        0.0
    )
    regime_prefers_nibrs = panel["preferred_source_by_regime"].eq(NIBRS_SOURCE)
    srs_regime_inferior = panel["reporting_regime"].isin(
        [
            "structurally_missing_or_unreliable",
            "lumpy_or_batched",
            "annual_only_but_usable",
        ]
    )
    nibrs_supports_better = nibrs_obs_weight.gt(srs_obs_weight) | (
        nibrs_obs_weight.eq(srs_obs_weight) & nibrs_months.gt(srs_months)
    )
    manual_source_override = (
        panel["source_override_applied"].astype("boolean").fillna(False).astype(bool)
    )
    published_nibrs_supports_nibrs = (
        has_nibrs
        & build_published_nibrs_corroboration_mask(
            nibrs_count=panel["reported_count_nibrs"],
            published_nibrs_count=panel["published_nibrs_official_count"],
            nibrs_months=nibrs_months,
            srs_count=panel["reported_count_srs"],
        )
    )
    prefer_nibrs_mask = build_prefer_nibrs_mask(
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
        manual_source_override=manual_source_override,
        published_nibrs_supports_nibrs=published_nibrs_supports_nibrs,
    )
    panel["preferred_source"] = initialize_preferred_source(
        has_cius=has_cius,
        has_local_publication=has_local_publication,
        has_state_publication=has_state_publication,
        has_srs=has_srs,
        has_nibrs=has_nibrs,
        prefer_nibrs_mask=prefer_nibrs_mask,
    )
    panel = assign_preferred_value(
        panel,
        output_col="preferred_count",
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
        output_col="preferred_observation_weight",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "observation_weight_cius",
            LOCAL_PUBLICATION_SOURCE: "observation_weight_local_publication",
            STATE_PUBLICATION_SOURCE: "observation_weight_state_publication",
            SUMMARY_SOURCE: "observation_weight_srs",
            NIBRS_SOURCE: "observation_weight_nibrs",
        },
    )
    panel = assign_preferred_value(
        panel,
        output_col="preferred_months_reported",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "mean_months_reported_cius",
            LOCAL_PUBLICATION_SOURCE: "mean_months_reported_local_publication",
            STATE_PUBLICATION_SOURCE: "mean_months_reported_state_publication",
            SUMMARY_SOURCE: "mean_months_reported_srs",
            NIBRS_SOURCE: "mean_months_reported_nibrs",
        },
    )
    panel["preferred_source_lane"] = (
        panel["preferred_source"].map(source_lane_from_source).astype("string")
    )
    panel["preferred_reporting_mode"] = (
        panel["preferred_source"].map(reporting_mode_from_source).astype("string")
    )
    panel["preferred_conversion_status"] = (
        panel["preferred_source"]
        .map(default_conversion_status_from_source)
        .astype("string")
    )
    panel["conversion_status_nibrs"] = panel["preferred_conversion_status"]
    panel = assign_preferred_value(
        panel,
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
    panel = panel.drop(columns=["conversion_status_nibrs"])
    panel["preferred_source_family"] = (
        panel["preferred_source"].map(source_family_from_source).astype("string")
    )
    panel["preferred_source_origin"] = [
        source_origin_from_parts(source=source) for source in panel["preferred_source"]
    ]
    panel["preferred_raw_data_source"] = [
        raw_data_source_from_parts(source=source)
        for source in panel["preferred_source"]
    ]
    panel["preferred_state_exception_flag"] = panel["preferred_source"].eq(
        STATE_PUBLICATION_SOURCE
    )
    panel["preferred_cius_reference_flag"] = panel["preferred_source_origin"].eq(
        CIUS_ORIGIN
    )
    return panel.sort_values(
        ["state_fips", "ori9", "offense"], kind="mergesort"
    ).reset_index(drop=True)
