from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.benchmark_imputation import load_benchmark_imputation_units
from crimerisk.city_residuals import prepare_city_residual_frame
from crimerisk.crime import OFFENSES_7
from crimerisk.model_surface import build_bg_feature_frame, merge_extra_bg_features
from crimerisk.paths import RepoPaths


DIRECT_SOURCE_MODE = "direct_city_incident"
MODELED_SOURCE_MODE = "modeled_transfer"
MIXED_SOURCE_MODE = "mixed"
SOURCE_MIXED_SHARE_CUTOFF = 0.60
DOMAIN_LOW_CUTOFF = 1.0 / 3.0
DOMAIN_IN_DOMAIN_CUTOFF = 2.0 / 3.0
DOMAIN_FALLBACK_SCORE = 0.50
# Territory whose jurisdiction target came from benchmark-constrained imputation
# (Class A: no agency reported it at all) reads as modelled, not measured. Above this
# share of a cell's mass the confidence tier is forced down and the reason is named.
BENCHMARK_IMPUTED_SHARE_LOW_CUTOFF = 0.50


@dataclass(frozen=True)
class ConfidenceArtifacts:
    block_group_metrics: pd.DataFrame
    tract_component_metrics: pd.DataFrame


def build_confidence_artifacts(
    *,
    paths: RepoPaths,
    year: int,
    block_group_surface: pd.DataFrame,
    component_audit: pd.DataFrame,
    city_posterior_diagnostics: pd.DataFrame,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    residual_training_city_shares_path: Path | None,
    residual_training_exclude_validation_case_types: tuple[str, ...],
    residual_training_extra_bg_feature_paths: tuple[Path, ...],
    residual_feature_policy_path: Path | None,
    residual_exclude_feature_policy_classes: tuple[str, ...],
    residual_exclude_feature_policy_classes_by_offense: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> ConfidenceArtifacts:
    """Build pure metadata lookups used by the release confidence layer.

    Domain overlap uses the same residual feature columns selected by
    city_residuals.py. For uncovered/modeled cells, the score is an exponential
    transform of robust standardized RMS distance to the covered-city training
    distribution for the same offense; covered city-posterior components score 1.
    """

    component = _normalize_component_audit(component_audit)
    bg_base = _block_group_base(block_group_surface, population_col=f"population_{int(year)}")
    bg_feature_frame = _load_residual_feature_frame(
        paths=paths,
        year=int(year),
        extra_feature_paths=list(residual_training_extra_bg_feature_paths),
    )
    urban = _urban_stratum_lookup(bg_base=bg_base, bg_feature_frame=bg_feature_frame)

    source_bg_long = _component_source_long(component, geo_col="bg_id")
    source_bg = _source_long_to_wide(
        source_bg_long,
        geo_col="bg_id",
        out_geo_col="block_group_geoid",
    )
    feed_bg = _feed_quality_wide(
        component,
        city_posterior_diagnostics,
        geo_col="bg_id",
        out_geo_col="block_group_geoid",
    )
    imputation_units = load_benchmark_imputation_units(paths, year=int(year))
    imputed_bg = _benchmark_imputed_share_wide(
        component,
        imputation_units,
        geo_col="bg_id",
        out_geo_col="block_group_geoid",
    )
    model_share_bg = _component_model_share_long(component, geo_col="bg_id")
    modeled_domain = _modeled_domain_scores(
        paths=paths,
        year=int(year),
        bg_base=bg_base,
        bg_feature_frame=bg_feature_frame,
        model_share_long=model_share_bg,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        residual_training_city_shares_path=residual_training_city_shares_path,
        residual_training_exclude_validation_case_types=residual_training_exclude_validation_case_types,
        residual_training_extra_bg_feature_paths=residual_training_extra_bg_feature_paths,
        residual_feature_policy_path=residual_feature_policy_path,
        residual_exclude_feature_policy_classes=residual_exclude_feature_policy_classes,
        residual_exclude_feature_policy_classes_by_offense=residual_exclude_feature_policy_classes_by_offense,
    )

    bg_metrics = bg_base.merge(urban, on=["state_fips", "block_group_geoid"], how="left")
    bg_metrics = bg_metrics.merge(source_bg, on=["state_fips", "block_group_geoid"], how="left")
    bg_metrics = bg_metrics.merge(feed_bg, on=["state_fips", "block_group_geoid"], how="left")
    if not imputed_bg.empty:
        bg_metrics = bg_metrics.merge(imputed_bg, on=["state_fips", "block_group_geoid"], how="left")
    bg_metrics = bg_metrics.merge(modeled_domain, on=["state_fips", "block_group_geoid"], how="left")
    for offense in OFFENSES_7:
        _fill_source_defaults(bg_metrics, offense=offense)
        modeled_col = f"_modeled_domain_overlap_score_{offense}"
        direct_share_col = f"_source_direct_share_{offense}"
        modeled_score = pd.to_numeric(bg_metrics.get(modeled_col), errors="coerce").fillna(DOMAIN_FALLBACK_SCORE)
        direct_share = pd.to_numeric(bg_metrics.get(direct_share_col), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        score = direct_share + (1.0 - direct_share) * modeled_score
        direct_mode = bg_metrics[f"source_mode_{offense}"].astype("string").eq(DIRECT_SOURCE_MODE)
        score.loc[direct_mode] = 1.0
        bg_metrics[f"domain_overlap_score_{offense}"] = score.clip(0.0, 1.0)
    bg_metrics = bg_metrics[_published_metric_columns(bg_metrics, geo_col="block_group_geoid")].copy()

    source_tract_long = _component_source_long(component, geo_col="tract_id")
    source_tract = _source_long_to_wide(
        source_tract_long,
        geo_col="tract_id",
        out_geo_col="tract_id",
    )
    feed_tract = _feed_quality_wide(
        component,
        city_posterior_diagnostics,
        geo_col="tract_id",
        out_geo_col="tract_id",
    )
    imputed_tract = _benchmark_imputed_share_wide(
        component,
        imputation_units,
        geo_col="tract_id",
        out_geo_col="tract_id",
    )
    tract_component_metrics = source_tract.merge(feed_tract, on=["state_fips", "tract_id"], how="outer")
    if not imputed_tract.empty:
        tract_component_metrics = tract_component_metrics.merge(
            imputed_tract, on=["state_fips", "tract_id"], how="left"
        )
    for offense in OFFENSES_7:
        _fill_source_defaults(tract_component_metrics, offense=offense)
    tract_component_metrics = tract_component_metrics[
        _published_metric_columns(tract_component_metrics, geo_col="tract_id", include_urban=False, include_domain=False)
    ].copy()

    return ConfidenceArtifacts(
        block_group_metrics=bg_metrics,
        tract_component_metrics=tract_component_metrics,
    )


def enrich_confidence_surfaces(
    *,
    block_group_surface: pd.DataFrame,
    tract_surface: pd.DataFrame,
    artifacts: ConfidenceArtifacts,
    population_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bg = block_group_surface.copy()
    bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    bg["block_group_geoid"] = bg["block_group_geoid"].astype("string").str.zfill(12)
    bg = bg.merge(
        artifacts.block_group_metrics,
        on=["state_fips", "block_group_geoid"],
        how="left",
    )
    bg = _drop_internal_confidence_columns(_append_confidence_tiers(bg))

    tract = tract_surface.copy()
    tract["state_fips"] = tract["state_fips"].astype("string").str.zfill(2)
    tract["tract_id"] = tract["tract_id"].astype("string").str.zfill(11)
    tract = tract.merge(
        artifacts.tract_component_metrics,
        on=["state_fips", "tract_id"],
        how="left",
    )
    tract = tract.merge(
        _rollup_domain_to_tract(bg),
        on=["state_fips", "tract_id"],
        how="left",
    )
    tract = tract.merge(
        _rollup_urban_stratum_to_tract(bg, population_col=population_col),
        on=["state_fips", "tract_id"],
        how="left",
    )
    for offense in OFFENSES_7:
        _fill_source_defaults(tract, offense=offense)
        tract[f"domain_overlap_score_{offense}"] = (
            pd.to_numeric(tract.get(f"domain_overlap_score_{offense}"), errors="coerce")
            .fillna(DOMAIN_FALLBACK_SCORE)
            .clip(0.0, 1.0)
        )
    tract = _drop_internal_confidence_columns(_append_confidence_tiers(tract))
    return bg, tract


def _block_group_base(surface: pd.DataFrame, *, population_col: str) -> pd.DataFrame:
    cols = [
        "state_fips",
        "block_group_geoid",
        "tract_id",
        population_col,
        "daytime_population_jobs_proxy",
        "exposure_proxy_2024",
        "land_area_sq_mi",
        "non_residential_flag",
    ]
    out = surface[[col for col in cols if col in surface.columns]].copy()
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["block_group_geoid"] = out["block_group_geoid"].astype("string").str.zfill(12)
    if "tract_id" not in out.columns:
        out["tract_id"] = out["block_group_geoid"].str.slice(0, 11)
    out["tract_id"] = out["tract_id"].astype("string").str.zfill(11)
    return out.drop_duplicates(["state_fips", "block_group_geoid"]).reset_index(drop=True)


def _numeric_series(df: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _normalize_component_audit(component_audit: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "state_fips",
        "bg_id",
        "tract_id",
        "jurisdiction_id",
        "offense",
        "component_count_after",
        "component_count",
        "component_share",
        "model_share",
        "city_incident_posterior_active",
        "city_posterior_model_prior_share",
    ]
    out = component_audit[[col for col in cols if col in component_audit.columns]].copy()
    for col in ("state_fips", "bg_id", "tract_id", "jurisdiction_id", "offense"):
        if col not in out.columns:
            out[col] = pd.NA
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    out["tract_id"] = out["tract_id"].astype("string").str.zfill(11)
    out["jurisdiction_id"] = out["jurisdiction_id"].astype("string")
    out["offense"] = out["offense"].astype("string")
    if "component_count_after" not in out.columns:
        out["component_count_after"] = out.get("component_count", 0.0)
    out["component_count_after"] = pd.to_numeric(out["component_count_after"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["component_share"] = _numeric_series(out, "component_share").fillna(0.0).clip(lower=0.0)
    out["model_share"] = _numeric_series(out, "model_share").fillna(0.0).clip(lower=0.0)
    out["city_posterior_model_prior_share"] = (
        _numeric_series(out, "city_posterior_model_prior_share").fillna(out["model_share"]).clip(lower=0.0)
    )
    if "city_incident_posterior_active" not in out.columns:
        out["city_incident_posterior_active"] = False
    out["city_incident_posterior_active"] = (
        out["city_incident_posterior_active"].astype("boolean").fillna(False).astype(bool)
    )
    return out


def _component_weights(component: pd.DataFrame, *, geo_col: str) -> pd.Series:
    group_cols = ["state_fips", geo_col, "offense"]
    count = pd.to_numeric(component["component_count_after"], errors="coerce").fillna(0.0).clip(lower=0.0)
    share = pd.to_numeric(component.get("component_share"), errors="coerce").fillna(0.0).clip(lower=0.0)
    count_total = count.groupby([component[col] for col in group_cols], dropna=False).transform("sum")
    share_total = share.groupby([component[col] for col in group_cols], dropna=False).transform("sum")
    return pd.Series(
        np.where(count_total.gt(0.0), count, np.where(share_total.gt(0.0), share, 0.0)),
        index=component.index,
        dtype=float,
    )


def _component_source_long(component: pd.DataFrame, *, geo_col: str) -> pd.DataFrame:
    if component.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                geo_col,
                "offense",
                "_source_direct_share",
                "_source_modeled_share",
                "source_mode",
                "source_mode_dominant_share",
                "source_mode_mixed",
            ]
        )
    work = component.copy()
    work["_weight"] = _component_weights(work, geo_col=geo_col)
    work["_source_bucket"] = np.where(work["city_incident_posterior_active"], DIRECT_SOURCE_MODE, MODELED_SOURCE_MODE)
    grouped = (
        work.groupby(["state_fips", geo_col, "offense", "_source_bucket"], dropna=False)["_weight"]
        .sum()
        .unstack("_source_bucket", fill_value=0.0)
        .reset_index()
    )
    if DIRECT_SOURCE_MODE not in grouped.columns:
        grouped[DIRECT_SOURCE_MODE] = 0.0
    if MODELED_SOURCE_MODE not in grouped.columns:
        grouped[MODELED_SOURCE_MODE] = 0.0
    direct = pd.to_numeric(grouped[DIRECT_SOURCE_MODE], errors="coerce").fillna(0.0).clip(lower=0.0)
    modeled = pd.to_numeric(grouped[MODELED_SOURCE_MODE], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = direct + modeled
    direct_share = pd.Series(np.where(total.gt(0.0), direct / total.replace(0.0, np.nan), 0.0), index=grouped.index).fillna(0.0)
    modeled_share = pd.Series(np.where(total.gt(0.0), modeled / total.replace(0.0, np.nan), 1.0), index=grouped.index).fillna(1.0)
    dominant = np.maximum(direct_share, modeled_share)
    mixed = total.gt(0.0) & pd.Series(dominant, index=grouped.index).lt(SOURCE_MIXED_SHARE_CUTOFF)
    source_mode = np.where(
        mixed,
        MIXED_SOURCE_MODE,
        np.where(direct_share.ge(modeled_share), DIRECT_SOURCE_MODE, MODELED_SOURCE_MODE),
    )
    return pd.DataFrame(
        {
            "state_fips": grouped["state_fips"].astype("string").str.zfill(2),
            geo_col: grouped[geo_col].astype("string").str.zfill(12 if geo_col == "bg_id" else 11),
            "offense": grouped["offense"].astype("string"),
            "_source_direct_share": direct_share.clip(0.0, 1.0),
            "_source_modeled_share": modeled_share.clip(0.0, 1.0),
            "source_mode": pd.Series(source_mode, index=grouped.index, dtype="string"),
            "source_mode_dominant_share": pd.Series(dominant, index=grouped.index, dtype=float).clip(0.0, 1.0),
            "source_mode_mixed": mixed.astype(bool),
        }
    )


def _benchmark_imputed_share_wide(
    component: pd.DataFrame,
    imputation_units: pd.DataFrame,
    *,
    geo_col: str,
    out_geo_col: str,
) -> pd.DataFrame:
    """Share of each cell's mass whose jurisdiction target is benchmark-imputed.

    An imputation unit has zero locked observed mass by construction, so the whole of
    its target is imputed and the per-jurisdiction fraction is exactly one; the cell
    share is then the component-weighted fraction of imputed jurisdictions.
    """
    out_columns = ["state_fips", out_geo_col] + [
        f"benchmark_imputed_share_{offense}" for offense in OFFENSES_7
    ]
    if component.empty or imputation_units is None or imputation_units.empty:
        return pd.DataFrame(columns=out_columns)
    units = imputation_units[
        pd.to_numeric(imputation_units["imputed_count"], errors="coerce").fillna(0.0).gt(0.0)
    ][["unit_id", "offense"]].drop_duplicates()
    if units.empty:
        return pd.DataFrame(columns=out_columns)
    units = units.rename(columns={"unit_id": "jurisdiction_id"})
    units["jurisdiction_id"] = units["jurisdiction_id"].astype("string")
    units["offense"] = units["offense"].astype("string")
    units["_imputed"] = 1.0

    work = component.copy()
    work["_weight"] = _component_weights(work, geo_col=geo_col)
    work = work.merge(units, on=["jurisdiction_id", "offense"], how="left")
    work["_imputed"] = pd.to_numeric(work["_imputed"], errors="coerce").fillna(0.0)
    work["_imputed_weight"] = work["_weight"] * work["_imputed"]
    grouped = (
        work.groupby(["state_fips", geo_col, "offense"], dropna=False)
        .agg(_weight=("_weight", "sum"), _imputed_weight=("_imputed_weight", "sum"))
        .reset_index()
    )
    grouped["share"] = np.where(
        pd.to_numeric(grouped["_weight"], errors="coerce").fillna(0.0).gt(0.0),
        grouped["_imputed_weight"] / grouped["_weight"].replace(0.0, np.nan),
        0.0,
    )
    grouped["share"] = pd.to_numeric(grouped["share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    base = grouped[["state_fips", geo_col]].drop_duplicates().rename(columns={geo_col: out_geo_col})
    for offense in OFFENSES_7:
        one = grouped[grouped["offense"].eq(offense)][["state_fips", geo_col, "share"]].rename(
            columns={geo_col: out_geo_col, "share": f"benchmark_imputed_share_{offense}"}
        )
        base = base.merge(one, on=["state_fips", out_geo_col], how="left")
    base["state_fips"] = base["state_fips"].astype("string").str.zfill(2)
    base[out_geo_col] = base[out_geo_col].astype("string").str.zfill(12 if out_geo_col == "block_group_geoid" else 11)
    return base


def _source_long_to_wide(source_long: pd.DataFrame, *, geo_col: str, out_geo_col: str) -> pd.DataFrame:
    if source_long.empty:
        return pd.DataFrame(columns=["state_fips", out_geo_col])
    base = source_long[["state_fips", geo_col]].drop_duplicates().rename(columns={geo_col: out_geo_col})
    for offense in OFFENSES_7:
        one = source_long[source_long["offense"].eq(offense)].copy()
        one = one.drop(columns=["offense"]).rename(columns={geo_col: out_geo_col})
        rename = {
            "source_mode": f"source_mode_{offense}",
            "source_mode_dominant_share": f"source_mode_dominant_share_{offense}",
            "source_mode_mixed": f"source_mode_mixed_{offense}",
            "_source_direct_share": f"_source_direct_share_{offense}",
            "_source_modeled_share": f"_source_modeled_share_{offense}",
        }
        base = base.merge(one.rename(columns=rename), on=["state_fips", out_geo_col], how="left")
    return base


def _feed_quality_wide(
    component: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    geo_col: str,
    out_geo_col: str,
) -> pd.DataFrame:
    required = {"jurisdiction_id", "state_fips", "offense", "match_rate", "missing_fraction", "alpha", "posterior_prior_fraction"}
    if component.empty or diagnostics.empty or not required.issubset(diagnostics.columns):
        return pd.DataFrame(columns=["state_fips", out_geo_col])
    diag = diagnostics[list(required)].copy()
    diag["state_fips"] = diag["state_fips"].astype("string").str.zfill(2)
    diag["jurisdiction_id"] = diag["jurisdiction_id"].astype("string")
    diag["offense"] = diag["offense"].astype("string")
    rename = {
        "match_rate": "feed_match_rate",
        "missing_fraction": "feed_missing_fraction",
        "alpha": "feed_alpha",
        "posterior_prior_fraction": "feed_prior_fraction",
    }
    diag = diag.rename(columns=rename)
    work = component[component["city_incident_posterior_active"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["state_fips", out_geo_col])
    work["_weight"] = _component_weights(work, geo_col=geo_col)
    work = work.merge(diag, on=["jurisdiction_id", "state_fips", "offense"], how="left")
    metrics = ("feed_match_rate", "feed_missing_fraction", "feed_alpha", "feed_prior_fraction")
    for metric in metrics:
        value = pd.to_numeric(work[metric], errors="coerce")
        weight = pd.to_numeric(work["_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        valid = value.notna() & weight.gt(0.0)
        work[f"_{metric}_weighted"] = np.where(valid, value * weight, 0.0)
        work[f"_{metric}_weight"] = np.where(valid, weight, 0.0)
    agg_spec: dict[str, tuple[str, str]] = {}
    for metric in metrics:
        agg_spec[f"{metric}_weighted"] = (f"_{metric}_weighted", "sum")
        agg_spec[f"{metric}_weight"] = (f"_{metric}_weight", "sum")
        agg_spec[f"{metric}_first"] = (metric, "first")
    long = (
        work.groupby(["state_fips", geo_col, "offense"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )
    if long.empty:
        return pd.DataFrame(columns=["state_fips", out_geo_col])
    for metric in metrics:
        long[metric] = np.where(
            pd.to_numeric(long[f"{metric}_weight"], errors="coerce").fillna(0.0).gt(0.0),
            long[f"{metric}_weighted"] / long[f"{metric}_weight"].replace(0.0, np.nan),
            long[f"{metric}_first"],
        )
    long["state_fips"] = long["state_fips"].astype("string").str.zfill(2)
    long[out_geo_col] = long[geo_col].astype("string").str.zfill(12 if out_geo_col == "block_group_geoid" else 11)
    long["offense"] = long["offense"].astype("string")
    base = long[["state_fips", out_geo_col]].drop_duplicates()
    for offense in OFFENSES_7:
        one = long[long["offense"].eq(offense)].drop(columns=["offense"])
        rename_cols = {
            "feed_match_rate": f"feed_match_rate_{offense}",
            "feed_missing_fraction": f"feed_missing_fraction_{offense}",
            "feed_alpha": f"feed_alpha_{offense}",
            "feed_prior_fraction": f"feed_prior_fraction_{offense}",
        }
        keep = ["state_fips", out_geo_col, *metrics]
        base = base.merge(one[keep].rename(columns=rename_cols), on=["state_fips", out_geo_col], how="left")
    return base


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    value = pd.to_numeric(values, errors="coerce")
    weight = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = value.notna() & weight.gt(0.0)
    if bool(mask.any()):
        return float(np.average(value.loc[mask], weights=weight.loc[mask]))
    non_null = value.dropna()
    return float(non_null.iloc[0]) if not non_null.empty else float("nan")


def _component_model_share_long(component: pd.DataFrame, *, geo_col: str) -> pd.DataFrame:
    if component.empty:
        return pd.DataFrame(columns=["state_fips", geo_col, "offense", "model_share_for_domain"])
    work = component.copy()
    work["_weight"] = _component_weights(work, geo_col=geo_col)
    work["_model_share_for_domain"] = (
        _numeric_series(work, "city_posterior_model_prior_share")
        .fillna(_numeric_series(work, "model_share"))
        .fillna(0.0)
        .clip(lower=0.0)
    )
    work["_weighted_model_share"] = work["_model_share_for_domain"] * work["_weight"]
    grouped = (
        work.groupby(["state_fips", geo_col, "offense"], dropna=False)
        .agg(
            weight_sum=("_weight", "sum"),
            weighted_model_share=("_weighted_model_share", "sum"),
            mean_model_share=("_model_share_for_domain", "mean"),
        )
        .reset_index()
    )
    grouped["model_share_for_domain"] = np.where(
        pd.to_numeric(grouped["weight_sum"], errors="coerce").fillna(0.0).gt(0.0),
        grouped["weighted_model_share"] / grouped["weight_sum"].replace(0.0, np.nan),
        grouped["mean_model_share"],
    )
    grouped["state_fips"] = grouped["state_fips"].astype("string").str.zfill(2)
    grouped[geo_col] = grouped[geo_col].astype("string").str.zfill(12 if geo_col == "bg_id" else 11)
    grouped["offense"] = grouped["offense"].astype("string")
    return grouped[["state_fips", geo_col, "offense", "model_share_for_domain"]]


def _load_residual_feature_frame(
    *,
    paths: RepoPaths,
    year: int,
    extra_feature_paths: list[Path],
) -> pd.DataFrame:
    bg = build_bg_feature_frame(paths=paths, year=int(year))
    bg = merge_extra_bg_features(bg, extra_feature_paths=extra_feature_paths)
    bg["bg_id"] = bg["bg_id"].astype("string").str.zfill(12)
    bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    if "tract_id" in bg.columns:
        bg["tract_id"] = bg["tract_id"].astype("string").str.zfill(11)
    return bg


def _load_residual_training_city_shares(
    *,
    path: Path | None,
    exclude_validation_case_types: tuple[str, ...],
) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_parquet(path).copy()
    if exclude_validation_case_types and "validation_case_type" in df.columns:
        excluded = {str(value) for value in exclude_validation_case_types}
        df = df[~df["validation_case_type"].astype(str).isin(excluded)].copy()
    return df


def _modeled_domain_scores(
    *,
    paths: RepoPaths,
    year: int,
    bg_base: pd.DataFrame,
    bg_feature_frame: pd.DataFrame,
    model_share_long: pd.DataFrame,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    residual_training_city_shares_path: Path | None,
    residual_training_exclude_validation_case_types: tuple[str, ...],
    residual_training_extra_bg_feature_paths: tuple[Path, ...],
    residual_feature_policy_path: Path | None,
    residual_exclude_feature_policy_classes: tuple[str, ...],
    residual_exclude_feature_policy_classes_by_offense: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> pd.DataFrame:
    training_shares = _load_residual_training_city_shares(
        path=residual_training_city_shares_path,
        exclude_validation_case_types=residual_training_exclude_validation_case_types,
    )
    if training_shares.empty:
        out = bg_base[["state_fips", "block_group_geoid"]].copy()
        for offense in OFFENSES_7:
            out[f"_modeled_domain_overlap_score_{offense}"] = DOMAIN_FALLBACK_SCORE
        return out

    training, feature_cols = prepare_city_residual_frame(
        paths=paths,
        city_shares=training_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=int(year),
        extra_feature_paths=list(residual_training_extra_bg_feature_paths),
        feature_policy_path=residual_feature_policy_path,
        exclude_feature_policy_classes=tuple(residual_exclude_feature_policy_classes),
        exclude_feature_policy_classes_by_offense=residual_exclude_feature_policy_classes_by_offense,
    )
    policy_application = training.attrs.get("city_residual_feature_policy_application", {})
    selected_by_offense = (
        policy_application.get("selected_feature_cols_by_offense", {})
        if isinstance(policy_application, dict)
        else {}
    )
    feature_cols = [col for col in feature_cols if col in bg_feature_frame.columns and col in training.columns]
    out = bg_base[["state_fips", "block_group_geoid"]].copy()
    bg_features = bg_base[["state_fips", "block_group_geoid"]].rename(columns={"block_group_geoid": "bg_id"}).merge(
        bg_feature_frame[["state_fips", "bg_id", *feature_cols]].drop_duplicates(["state_fips", "bg_id"]),
        on=["state_fips", "bg_id"],
        how="left",
    )
    bg_features = bg_features.sort_values(["state_fips", "bg_id"], kind="mergesort").reset_index(drop=True)
    out = out.sort_values(["state_fips", "block_group_geoid"], kind="mergesort").reset_index(drop=True)
    model_share_lookup = model_share_long.copy()
    if not model_share_lookup.empty:
        model_share_lookup["state_fips"] = model_share_lookup["state_fips"].astype("string").str.zfill(2)
        model_share_lookup["bg_id"] = model_share_lookup["bg_id"].astype("string").str.zfill(12)
        model_share_lookup["offense"] = model_share_lookup["offense"].astype("string")

    for offense in OFFENSES_7:
        train_o = training[training["offense"].astype("string").eq(offense)].copy()
        if train_o.empty:
            out[f"_modeled_domain_overlap_score_{offense}"] = DOMAIN_FALLBACK_SCORE
            continue
        offense_feature_cols = selected_by_offense.get(str(offense), feature_cols)
        offense_feature_cols = [
            str(col)
            for col in offense_feature_cols
            if str(col) in feature_cols and str(col) in train_o.columns and str(col) in bg_features.columns
        ]
        offense_feature_values = (
            bg_features[offense_feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
            if offense_feature_cols
            else np.empty((len(bg_features), 0), dtype=np.float64)
        )
        offense_model_share = model_share_lookup[model_share_lookup["offense"].eq(offense)][
            ["state_fips", "bg_id", "model_share_for_domain"]
        ]
        target_model = bg_features[["state_fips", "bg_id"]].merge(
            offense_model_share,
            on=["state_fips", "bg_id"],
            how="left",
        )["model_share_for_domain"]
        out[f"_modeled_domain_overlap_score_{offense}"] = _domain_score_for_offense(
            train_o=train_o,
            feature_cols=offense_feature_cols,
            target_feature_values=offense_feature_values,
            target_model_share=pd.to_numeric(target_model, errors="coerce").fillna(0.0).clip(lower=0.0),
        )
    return out


def _domain_score_for_offense(
    *,
    train_o: pd.DataFrame,
    feature_cols: list[str],
    target_feature_values: np.ndarray,
    target_model_share: pd.Series,
) -> np.ndarray:
    train_features = (
        train_o[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        if feature_cols
        else np.empty((len(train_o), 0), dtype=np.float64)
    )
    train_model = pd.to_numeric(train_o["model_share"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    target_model = target_model_share.to_numpy(dtype=np.float64)
    train_log = np.log(np.clip(train_model, 1e-12, None))
    target_log = np.log(np.clip(target_model, 1e-12, None))

    train_matrix = np.column_stack([train_features, train_model, train_log])
    target_matrix = np.column_stack([target_feature_values, target_model, target_log])
    if train_matrix.shape[1] == 0:
        return np.full(len(target_model_share), DOMAIN_FALLBACK_SCORE, dtype=float)

    mean = np.nanmean(train_matrix, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(train_matrix, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)

    train_filled = np.where(np.isfinite(train_matrix), train_matrix, mean)
    train_z = (train_filled - mean) / std
    train_distance = np.sqrt(np.mean(np.square(train_z), axis=1))
    scale = float(np.nanquantile(train_distance, 0.90)) if len(train_distance) else 1.0
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0

    target_filled = np.where(np.isfinite(target_matrix), target_matrix, mean)
    target_z = (target_filled - mean) / std
    target_distance = np.sqrt(np.mean(np.square(target_z), axis=1))
    score = np.exp(-0.5 * np.square(target_distance / scale))
    return np.clip(np.where(np.isfinite(score), score, DOMAIN_FALLBACK_SCORE), 0.0, 1.0)


def _urban_stratum_lookup(*, bg_base: pd.DataFrame, bg_feature_frame: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "state_fips",
        "bg_id",
        "nlcd_impervious_mean",
        "nlcd_developed_low_share",
        "nlcd_developed_high_share",
        "nlcd_nonroad_urban_descriptor_share",
    ]
    existing = [col for col in feature_cols if col in bg_feature_frame.columns]
    features = bg_feature_frame[existing].drop_duplicates(["state_fips", "bg_id"]).copy()
    out = bg_base.merge(
        features.rename(columns={"bg_id": "block_group_geoid"}),
        on=["state_fips", "block_group_geoid"],
        how="left",
    )
    land_area = _numeric_series(out, "land_area_sq_mi").fillna(0.0).clip(lower=0.0)
    activity = (
        _numeric_series(out, "exposure_proxy_2024", default=np.nan).combine_first(
            _numeric_series(out, "daytime_population_jobs_proxy")
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )
    density_sq_mi = pd.Series(np.where(land_area.gt(0.0), activity / land_area.replace(0.0, np.nan), 0.0), index=out.index).fillna(0.0)
    impervious = _numeric_series(out, "nlcd_impervious_mean").fillna(0.0).clip(lower=0.0, upper=100.0)
    developed_low = _numeric_series(out, "nlcd_developed_low_share").fillna(0.0).clip(0.0, 1.0)
    developed_high = _numeric_series(out, "nlcd_developed_high_share").fillna(0.0).clip(0.0, 1.0)
    developed = (developed_low + developed_high).clip(0.0, 1.0)
    urban_descriptor = _numeric_series(out, "nlcd_nonroad_urban_descriptor_share").fillna(0.0).clip(0.0, 1.0)
    non_residential = out.get("non_residential_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool)

    urban_core = (
        density_sq_mi.ge(10_000.0)
        | (density_sq_mi.ge(5_000.0) & impervious.ge(50.0) & developed_high.ge(0.20))
        | (urban_descriptor.ge(0.50) & impervious.ge(50.0))
    )
    urban = density_sq_mi.ge(2_500.0) | (density_sq_mi.ge(1_000.0) & impervious.ge(30.0) & developed.ge(0.50))
    suburban = density_sq_mi.ge(250.0) | impervious.ge(10.0) | developed.ge(0.20)
    stratum = np.select(
        [non_residential, urban_core, urban, suburban],
        ["non_residential", "urban_core", "urban", "suburban"],
        default="rural",
    )
    return out[["state_fips", "block_group_geoid"]].assign(urban_stratum=pd.Series(stratum, index=out.index, dtype="string"))


def _fill_source_defaults(df: pd.DataFrame, *, offense: str) -> None:
    mode_col = f"source_mode_{offense}"
    share_col = f"source_mode_dominant_share_{offense}"
    mixed_col = f"source_mode_mixed_{offense}"
    direct_share_col = f"_source_direct_share_{offense}"
    modeled_share_col = f"_source_modeled_share_{offense}"
    if mode_col not in df.columns:
        df[mode_col] = MODELED_SOURCE_MODE
    df[mode_col] = df[mode_col].astype("string").fillna(MODELED_SOURCE_MODE)
    if share_col not in df.columns:
        df[share_col] = 1.0
    df[share_col] = pd.to_numeric(df[share_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    if mixed_col not in df.columns:
        df[mixed_col] = False
    df[mixed_col] = df[mixed_col].astype("boolean").fillna(False).astype(bool)
    if direct_share_col not in df.columns:
        df[direct_share_col] = np.where(df[mode_col].eq(DIRECT_SOURCE_MODE), 1.0, 0.0)
    df[direct_share_col] = pd.to_numeric(df[direct_share_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if modeled_share_col not in df.columns:
        df[modeled_share_col] = 1.0 - df[direct_share_col]
    df[modeled_share_col] = pd.to_numeric(df[modeled_share_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    imputed_col = f"benchmark_imputed_share_{offense}"
    if imputed_col not in df.columns:
        df[imputed_col] = 0.0
    df[imputed_col] = pd.to_numeric(df[imputed_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _drop_internal_confidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    internal_cols = [
        col
        for col in df.columns
        if col.startswith("_source_direct_share_")
        or col.startswith("_source_modeled_share_")
        or col.startswith("_modeled_domain_overlap_score_")
    ]
    if not internal_cols:
        return df
    return df.drop(columns=internal_cols)


def _published_metric_columns(
    df: pd.DataFrame,
    *,
    geo_col: str,
    include_urban: bool = True,
    include_domain: bool = True,
) -> list[str]:
    cols = ["state_fips", geo_col]
    if include_urban and "urban_stratum" in df.columns:
        cols.append("urban_stratum")
    for offense in OFFENSES_7:
        for col in (
            f"source_mode_{offense}",
            f"source_mode_dominant_share_{offense}",
            f"source_mode_mixed_{offense}",
            f"feed_match_rate_{offense}",
            f"feed_missing_fraction_{offense}",
            f"feed_alpha_{offense}",
            f"feed_prior_fraction_{offense}",
            f"benchmark_imputed_share_{offense}",
            f"domain_overlap_score_{offense}",
        ):
            if col in df.columns and (include_domain or not col.startswith("domain_overlap_score_")):
                cols.append(col)
    return cols


def _rollup_domain_to_tract(bg: pd.DataFrame) -> pd.DataFrame:
    keys = ["state_fips", "tract_id"]
    base = bg[keys].drop_duplicates().copy()
    for offense in OFFENSES_7:
        score_col = f"domain_overlap_score_{offense}"
        count_col = f"expected_count_{offense}"
        work = bg[keys + [score_col, count_col]].copy()
        score = pd.to_numeric(work[score_col], errors="coerce").fillna(DOMAIN_FALLBACK_SCORE).clip(0.0, 1.0)
        count = pd.to_numeric(work[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        work["_weighted_score"] = score * count
        work["_count"] = count
        grouped = work.groupby(keys, dropna=False).agg(
            weighted_score=("_weighted_score", "sum"),
            count=("_count", "sum"),
            mean_score=(score_col, lambda s: float(pd.to_numeric(s, errors="coerce").fillna(DOMAIN_FALLBACK_SCORE).mean())),
        ).reset_index()
        grouped[f"domain_overlap_score_{offense}"] = np.where(
            pd.to_numeric(grouped["count"], errors="coerce").fillna(0.0).gt(0.0),
            grouped["weighted_score"] / grouped["count"].replace(0.0, np.nan),
            grouped["mean_score"],
        )
        base = base.merge(grouped[keys + [f"domain_overlap_score_{offense}"]], on=keys, how="left")
    return base


def _rollup_urban_stratum_to_tract(bg: pd.DataFrame, *, population_col: str) -> pd.DataFrame:
    work = bg[
        [
            col
            for col in [
                "state_fips",
                "tract_id",
                "urban_stratum",
                "exposure_proxy_2024",
                "daytime_population_jobs_proxy",
                population_col,
            ]
            if col in bg.columns
        ]
    ].copy()
    exposure = (
        pd.to_numeric(work["exposure_proxy_2024"], errors="coerce")
        if "exposure_proxy_2024" in work.columns
        else pd.Series(np.nan, index=work.index, dtype=float)
    )
    jobs_exposure = (
        pd.to_numeric(work["daytime_population_jobs_proxy"], errors="coerce")
        if "daytime_population_jobs_proxy" in work.columns
        else pd.Series(0.0, index=work.index, dtype=float)
    )
    exposure = exposure.combine_first(jobs_exposure).fillna(0.0).clip(lower=0.0)
    population = pd.to_numeric(work[population_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work["_weight"] = np.where(exposure.gt(0.0), exposure, np.where(population.gt(0.0), population, 1.0))
    grouped = (
        work.groupby(["state_fips", "tract_id", "urban_stratum"], dropna=False)["_weight"]
        .sum()
        .reset_index()
        .sort_values(["state_fips", "tract_id", "_weight", "urban_stratum"], ascending=[True, True, False, True], kind="mergesort")
    )
    return grouped.drop_duplicates(["state_fips", "tract_id"], keep="first")[
        ["state_fips", "tract_id", "urban_stratum"]
    ].reset_index(drop=True)


def _append_confidence_tiers(surface: pd.DataFrame) -> pd.DataFrame:
    out = surface.copy()
    for offense in OFFENSES_7:
        mode = out[f"estimate_mode_{offense}"].astype("string") if f"estimate_mode_{offense}" in out.columns else pd.Series("unknown", index=out.index, dtype="string")
        reliability = out[f"reliability_tier_{offense}"].astype("string") if f"reliability_tier_{offense}" in out.columns else pd.Series("low", index=out.index, dtype="string")
        source = out[f"source_mode_{offense}"].astype("string") if f"source_mode_{offense}" in out.columns else pd.Series(MODELED_SOURCE_MODE, index=out.index, dtype="string")
        domain = _numeric_series(out, f"domain_overlap_score_{offense}", default=DOMAIN_FALLBACK_SCORE).fillna(DOMAIN_FALLBACK_SCORE).clip(0.0, 1.0)
        imputed_share = (
            _numeric_series(out, f"benchmark_imputed_share_{offense}", default=0.0)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        valid_denominator = mode.eq("count_derived")
        modeled_or_mixed = source.isin([MODELED_SOURCE_MODE, MIXED_SOURCE_MODE])
        benchmark_imputed = imputed_share.ge(BENCHMARK_IMPUTED_SHARE_LOW_CUTOFF)
        low = (
            (~valid_denominator)
            | reliability.eq("low")
            | (modeled_or_mixed & domain.lt(DOMAIN_LOW_CUTOFF))
            | benchmark_imputed
        )
        high = (
            ~low
            & valid_denominator
            & (
                source.eq(DIRECT_SOURCE_MODE)
                | (reliability.eq("high") & domain.ge(DOMAIN_IN_DOMAIN_CUTOFF))
            )
        )
        tier = pd.Series("medium", index=out.index, dtype="string")
        tier.loc[low] = "low"
        tier.loc[high] = "high"
        out[f"confidence_tier_{offense}"] = tier
        transient = (
            out[f"transient_exposure_likely_{offense}"].fillna(False).astype(bool)
            if f"transient_exposure_likely_{offense}" in out.columns
            else pd.Series(False, index=out.index)
        )
        out[f"confidence_reasons_{offense}"] = [
            _confidence_reason(
                mode_value=mode_value,
                reliability_value=reliability_value,
                source_value=source_value,
                domain_value=domain_value,
                transient_value=transient_value,
                benchmark_imputed_value=benchmark_imputed_value,
            )
            for mode_value, reliability_value, source_value, domain_value, transient_value, benchmark_imputed_value in zip(
                mode.astype(object),
                reliability.astype(object),
                source.astype(object),
                domain.to_numpy(dtype=float),
                transient.to_numpy(dtype=bool),
                benchmark_imputed.to_numpy(dtype=bool),
                strict=True,
            )
        ]
    return out


def _confidence_reason(
    *,
    mode_value: object,
    reliability_value: object,
    source_value: object,
    domain_value: float,
    transient_value: bool,
    benchmark_imputed_value: bool = False,
) -> str:
    reasons: list[str] = []
    mode = str(mode_value)
    reliability = str(reliability_value)
    source = str(source_value)
    if mode != "count_derived":
        reasons.append(f"denominator_suppressed:{mode}")
    if reliability == "low":
        reasons.append("count_reliability_low")
    elif reliability == "high":
        reasons.append("count_reliability_high")
    else:
        reasons.append("count_reliability_medium")
    if source == DIRECT_SOURCE_MODE:
        reasons.append("direct_city_incident_source")
    elif source == MIXED_SOURCE_MODE:
        reasons.append("mixed_source_mode")
    else:
        reasons.append("modeled_transfer_source")
    if float(domain_value) < DOMAIN_LOW_CUTOFF:
        reasons.append("domain_overlap_low")
    elif float(domain_value) >= DOMAIN_IN_DOMAIN_CUTOFF:
        reasons.append("domain_overlap_in_domain")
    else:
        reasons.append("domain_overlap_moderate")
    if transient_value:
        reasons.append("transient_exposure_likely")
    if bool(benchmark_imputed_value):
        reasons.append("benchmarked_nonreporter_imputation")
    return "|".join(reasons)
