"""Build the step-13 feature transfer and proxy policy artifact.

This diagnostic is analysis-only. It does not change the production estimator.
It reuses the existing city-blocked residual-validation split and the
jurisdiction model training frame, fits each held-out fold once, then measures
feature-group permutation degradation on the held-out rows.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for _thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env_var] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for _path in (SRC_ROOT, SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import nested_city_cv_harness as nested_cv
from crimerisk.allocation import promoted_residual_extra_bg_feature_paths
from crimerisk.city_residuals import (
    CityResidualConfig,
    apply_city_residual_model,
    fit_city_residual_model,
)
from crimerisk.covariates.feature_filters import is_protected_column
from crimerisk.crime import OFFENSES_7
from crimerisk.model_surface import (
    ModelSurfaceConfig,
    _build_model,
    _build_offense_training_state,
    _build_sparse_offense_pooling_target,
    _prepare_model_surface_context,
)
from crimerisk.paths import RepoPaths


EPS = 1e-12
DEFAULT_TVD_FLOOR = 0.0025
DEFAULT_DEVIANCE_FLOOR = 0.01
DEFAULT_PILOT_GROUPS = (
    "income_wealth",
    "roads_transport",
    "state_fixed_effect",
)
PROXY_GROUPS = {
    "digital_access_proxy",
    "education_attainment",
    "employment_labor",
    "housing_tenure_overcrowding",
    "income_wealth",
    "lodes_education",
    "lodes_earnings",
    "poverty_snap",
    "rent_burden",
}
DIRECT_EXCLUDED_GROUPS = {
    "direct_protected_composition",
    "familial_status_composition",
    "national_origin_proxy",
    "sex_composition",
}
LOCAL_MECHANISM_GROUPS = {
    "activity_exposure_commercial",
    "commute_transport_exposure",
    "institutional_anchors",
    "land_cover",
    "lodes_age",
    "lodes_industry_activity",
    "population_density",
    "residential_mobility_domestic",
    "roads_transport",
    "vehicle_availability",
}


@dataclass(frozen=True)
class Timing:
    step: str
    elapsed_sec: float
    detail: str


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is not None and pd.isna(value):
        return None
    return value


def _seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _normalize_share(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(np.nansum(arr))
    if np.isfinite(total) and total > 0:
        return arr / total
    if len(arr) == 0:
        return arr
    return np.full(len(arr), 1.0 / float(len(arr)))


def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(y_true, dtype=float), 0.0, None)
    mu = np.clip(np.asarray(y_pred, dtype=float), EPS, None)
    out = np.empty_like(mu, dtype=float)
    positive = y > 0
    out[positive] = 2.0 * (y[positive] * np.log(y[positive] / mu[positive]) - (y[positive] - mu[positive]))
    out[~positive] = 2.0 * mu[~positive]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _policy_group_for_feature(column: str) -> str:
    col = str(column).strip().lower()
    if col.startswith("state_"):
        return "state_fixed_effect"
    if is_protected_column(col):
        return "direct_protected_composition"
    if col == "lodes_cs01_share" or col.startswith("male_") or col in {"median_age_male", "median_age_female"}:
        return "sex_composition"
    if (
        col.startswith("hh_family_")
        or col.startswith("hh_under18")
        or "single_parent" in col
        or "children" in col
    ):
        return "familial_status_composition"
    if col == "tract_mobility_abroad_share" or "foreign_born" in col:
        return "national_origin_proxy"
    if col in {
        "median_household_income_2024",
        "per_capita_income_2024",
        "median_home_value_2024",
        "median_rent_2024",
    } or col.startswith("hh_income_"):
        return "income_wealth"
    if col in {"tract_poverty_rate", "snap_received_share"}:
        return "poverty_snap"
    if col.startswith("rent_"):
        return "rent_burden"
    if (
        col.startswith("edu_")
        or col == "lodes_cd01_share"
        or col == "lodes_cd02_share"
        or col == "lodes_cd03_share"
        or col == "lodes_cd04_share"
    ):
        return "education_attainment" if col.startswith("edu_") else "lodes_education"
    if col in {"unemployment_rate", "unemployed_per_capita", "employed_per_capita", "labor_force_per_capita"}:
        return "employment_labor"
    if col.startswith("lodes_ce"):
        return "lodes_earnings"
    if col.startswith("lodes_ca"):
        return "lodes_age"
    if col.startswith("lodes_cns"):
        return "lodes_industry_activity"
    if (
        col.startswith("tenure_")
        or col.startswith("vacancy_")
        or col in {
            "housing_occupied_share",
            "overcrowded_occupied_share",
            "owner_overcrowded_share",
            "renter_overcrowded_share",
            "owner_units_per_capita",
            "renter_units_per_capita",
            "vacant_units_per_capita",
        }
        or col.startswith("structure_")
    ):
        return "housing_tenure_overcrowding"
    if "internet" in col or "broadband" in col or "computer" in col or "cellular" in col or "dialup" in col or "satellite" in col:
        return "digital_access_proxy"
    if (
        col.startswith("hpms_")
        or col.startswith("transit_")
        or col.startswith("fixed_guideway_")
        or col.startswith("nearest_transit_")
        or col.startswith("nearest_fixed_guideway_")
        or "road" in col
        or col.endswith("_length_km")
        or col in {"limited_access_road_share"}
    ):
        return "roads_transport"
    if col.startswith("nlcd_") or col in {"land_area_sqkm", "log_aland20"}:
        return "land_cover"
    if col in {
        "education_anchor_density_sqkm",
        "public_school_density_sqkm",
        "postsecondary_density_sqkm",
        "public_school_present",
        "postsecondary_present",
        "nearest_acute_care_hospital_km",
        "nearest_emergency_hospital_km",
    }:
        return "institutional_anchors"
    if (
        col.startswith("overture_")
        or col.startswith("log_overture_")
        or col.startswith("nearest_overture_")
        or col.startswith("jobs_")
        or col.startswith("log_jobs_")
        or col.startswith("daytime_population_jobs_proxy")
        or col.startswith("log_daytime_population_jobs_proxy")
        or col.startswith("workplace_")
        or col.startswith("resident_jobs")
    ):
        return "activity_exposure_commercial"
    if col.startswith("commute_") or col.startswith("resident_same_bg"):
        return "commute_transport_exposure"
    if col.startswith("tract_mobility_"):
        return "residential_mobility_domestic"
    if (
        "population_density" in col
        or col in {
            "pop_2024",
            "pop_factor_2024",
            "log_total_population",
            "log_housing_density_sqkm",
            "housing_density_sqkm",
            "occupied_units_per_capita",
        }
    ):
        return "population_density"
    if "vehicle" in col or col.startswith("avg_vehicles"):
        return "vehicle_availability"
    if col.startswith("hh_") or col.startswith("avg_household_size"):
        return "household_composition_other"
    return "other_covariate"


def _policy_flags_for_group(group: str) -> dict[str, bool]:
    group = str(group)
    return {
        "direct_protected_flag": group == "direct_protected_composition",
        "hard_excluded_protected_flag": group in DIRECT_EXCLUDED_GROUPS,
        "proxy_review_flag": group in PROXY_GROUPS,
        "local_mechanism_flag": group in LOCAL_MECHANISM_GROUPS,
    }


def _permute_feature_group(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    seed: int,
    bg_key_col: str | None = "bg_id",
) -> pd.DataFrame:
    out = frame.copy()
    cols = [col for col in columns if col in out.columns]
    if not cols or len(out) <= 1:
        return out
    rng = np.random.default_rng(int(seed))
    if bg_key_col is None or bg_key_col not in out.columns:
        perm = rng.permutation(len(out))
        out.loc[:, cols] = out.iloc[perm][cols].to_numpy()
        for col in cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    unique = out[[bg_key_col, *cols]].drop_duplicates(bg_key_col, keep="first").reset_index(drop=True)
    if len(unique) <= 1:
        return out
    perm = rng.permutation(len(unique))
    shuffled = unique[[bg_key_col]].copy()
    shuffled.loc[:, cols] = unique.iloc[perm][cols].to_numpy()
    out = out.drop(columns=cols).merge(shuffled, on=bg_key_col, how="left", sort=False)
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[frame.columns.tolist()]


def _weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float | None:
    v = pd.to_numeric(values, errors="coerce")
    mask = v.notna() & np.isfinite(v)
    if not bool(mask.any()):
        return None
    if weights is None:
        return float(v.loc[mask].mean())
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    w = w.loc[mask]
    if float(w.sum()) <= 0:
        return float(v.loc[mask].mean())
    return float(np.average(v.loc[mask].to_numpy(dtype=float), weights=w.to_numpy(dtype=float)))


def _bootstrap_mean_ci(
    rows: pd.DataFrame,
    *,
    value_col: str,
    weight_col: str | None,
    unit_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    if rows.empty or value_col not in rows.columns:
        return None, None, None
    work = rows[[unit_col, value_col] + ([weight_col] if weight_col is not None else [])].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if weight_col is not None:
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
    work = work[work[value_col].notna() & np.isfinite(work[value_col])]
    if work.empty:
        return None, None, None
    if weight_col is None:
        unit = work.groupby(unit_col, dropna=False, sort=True).agg(value=(value_col, "mean")).reset_index()
        point = float(unit["value"].mean())
    else:
        work["_weighted_value"] = work[value_col] * work[weight_col]
        unit = (
            work.groupby(unit_col, dropna=False, sort=True)
            .agg(numerator=("_weighted_value", "sum"), denominator=(weight_col, "sum"))
            .reset_index()
        )
        denom = float(unit["denominator"].sum())
        point = float(unit["numerator"].sum() / denom) if denom > 0 else None
    if point is None:
        return None, None, None
    if int(n_boot) <= 1 or len(unit) <= 1:
        return point, None, None
    rng = np.random.default_rng(int(seed))
    stats: list[float] = []
    n = int(len(unit))
    if weight_col is None:
        values = unit["value"].to_numpy(dtype=float)
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            stat = float(np.nanmean(values[idx]))
            if np.isfinite(stat):
                stats.append(stat)
    else:
        numerators = unit["numerator"].to_numpy(dtype=float)
        denominators = unit["denominator"].to_numpy(dtype=float)
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            denom = float(denominators[idx].sum())
            if denom > 0:
                stats.append(float(numerators[idx].sum() / denom))
    if not stats:
        return point, None, None
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _summarize_axis(
    rows: pd.DataFrame,
    *,
    value_col: str,
    weight_col: str | None,
    n_boot: int,
    seed: int,
    floor: float,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    output: list[dict[str, object]] = []
    group_cols = ["feature_group", "offense"]
    for keys, group in rows.groupby(group_cols, dropna=False, sort=True):
        feature_group, offense = (str(keys[0]), str(keys[1]))
        point, low, high = _bootstrap_mean_ci(
            group,
            value_col=value_col,
            weight_col=weight_col,
            unit_col="unit_id",
            n_boot=n_boot,
            seed=seed + _seed(feature_group, offense, value_col) % 100000,
        )
        output.append(
            {
                "feature_group": feature_group,
                "offense": offense,
                "delta": point,
                "ci95_low": low,
                "ci95_high": high,
                "transfers": bool(low is not None and low > 0.0 and point is not None and point >= float(floor)),
                "unit_count": int(group["unit_id"].nunique()),
                "row_count": int(len(group)),
            }
        )

    for feature_group, group in rows.groupby("feature_group", dropna=False, sort=True):
        point, low, high = _bootstrap_mean_ci(
            group,
            value_col=value_col,
            weight_col=weight_col,
            unit_col="unit_id",
            n_boot=n_boot,
            seed=seed + _seed(feature_group, "overall", value_col) % 100000,
        )
        output.append(
            {
                "feature_group": str(feature_group),
                "offense": "overall",
                "delta": point,
                "ci95_low": low,
                "ci95_high": high,
                "transfers": bool(low is not None and low > 0.0 and point is not None and point >= float(floor)),
                "unit_count": int(group["unit_id"].nunique()),
                "row_count": int(len(group)),
            }
        )
    return pd.DataFrame(output)


def _score_within_group(group: pd.DataFrame, pred_col: str) -> dict[str, object]:
    truth = _normalize_share(group["true_share"])
    pred = _normalize_share(group[pred_col])
    return {
        "tvd": float(0.5 * np.abs(pred - truth).sum()),
        "incident_total": float(pd.to_numeric(group["incident_count"], errors="coerce").fillna(0.0).sum()),
    }


def _within_permutation_rows(
    *,
    paths: RepoPaths,
    year: int,
    feature_groups: dict[str, list[str]],
    config: CityResidualConfig,
    role_inventory_path: Path,
    city_shares_path: Path,
    bg_prior_path: Path,
    bg_crosswalk_path: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object], list[Timing]]:
    timings: list[Timing] = []
    started = time.perf_counter()
    extra_feature_paths = list(promoted_residual_extra_bg_feature_paths(paths))
    base_frame, feature_cols, _role_meta, _roles = nested_cv._build_base_frame(
        paths=paths,
        year=year,
        role_inventory_path=role_inventory_path,
        city_shares_path=city_shares_path,
        bg_prior_path=bg_prior_path,
        bg_crosswalk_path=bg_crosswalk_path,
        extra_feature_paths=extra_feature_paths,
    )
    if base_frame.empty:
        raise RuntimeError("No evaluable city/offense rows available for within-axis permutation audit.")
    timings.append(Timing("within_load_base_frame", time.perf_counter() - started, f"rows={len(base_frame)} features={len(feature_cols)}"))

    rows: list[dict[str, object]] = []
    trainable_city_keys = sorted(
        base_frame.loc[
            base_frame["inventory_role"].isin(nested_cv.TRAINABLE_ROLES),
            "city_key",
        ].dropna().astype(str).unique().tolist()
    )
    trainable_set = set(trainable_city_keys)
    outer_city_keys = sorted(base_frame["city_key"].dropna().astype(str).unique().tolist())
    fold_started = time.perf_counter()
    for fold_idx, outer_city_key in enumerate(outer_city_keys, start=1):
        train = base_frame[
            base_frame["inventory_role"].isin(nested_cv.TRAINABLE_ROLES)
            & (
                True
                if outer_city_key not in trainable_set
                else ~base_frame["city_key"].astype(str).eq(str(outer_city_key))
            )
        ].copy()
        test = base_frame[base_frame["city_key"].astype(str).eq(str(outer_city_key))].copy()
        if test.empty:
            continue
        selected_features = nested_cv._select_fold_feature_cols(train, tuple(feature_cols))
        with threadpool_limits(limits=1):
            fitted = fit_city_residual_model(
                train,
                feature_cols=selected_features,
                config=config,
            )
            baseline = apply_city_residual_model(test, fitted=fitted)
        if "residual_model_share" not in baseline.columns:
            baseline["residual_model_share"] = baseline["model_share"]
        baseline_scores: dict[tuple[str, str], dict[str, object]] = {}
        for (city_key, offense), grp in baseline.groupby(["city_key", "offense"], dropna=False, sort=True):
            baseline_scores[(str(city_key), str(offense))] = _score_within_group(grp, "residual_model_share")

        selected_set = set(selected_features)
        for feature_group, columns in feature_groups.items():
            group_cols = [col for col in columns if col in selected_set and col in test.columns]
            if group_cols:
                permuted = _permute_feature_group(
                    test,
                    columns=group_cols,
                    seed=seed + _seed("within", outer_city_key, feature_group) % 100000,
                    bg_key_col="bg_id",
                )
                with threadpool_limits(limits=1):
                    scored = apply_city_residual_model(permuted, fitted=fitted)
                if "residual_model_share" not in scored.columns:
                    scored["residual_model_share"] = scored["model_share"]
            else:
                scored = baseline
            for (city_key, offense), grp in scored.groupby(["city_key", "offense"], dropna=False, sort=True):
                key = (str(city_key), str(offense))
                base = baseline_scores.get(key)
                if base is None:
                    continue
                perm = _score_within_group(grp, "residual_model_share")
                rows.append(
                    {
                        "feature_group": str(feature_group),
                        "unit_id": f"{city_key}|{offense}",
                        "city_key": str(city_key),
                        "offense": str(offense),
                        "delta_tvd": float(perm["tvd"]) - float(base["tvd"]),
                        "baseline_tvd": float(base["tvd"]),
                        "permuted_tvd": float(perm["tvd"]),
                        "incident_total": float(base["incident_total"]),
                        "permuted_column_count": int(len(group_cols)),
                        "heldout_row_count": int(len(grp)),
                    }
                )
        print(
            f"[feature-policy] within fold {fold_idx}/{len(outer_city_keys)} {outer_city_key} complete",
            flush=True,
        )
    timings.append(Timing("within_fit_and_permute", time.perf_counter() - fold_started, f"outer_folds={len(outer_city_keys)} groups={len(feature_groups)}"))
    meta = {
        "base_frame_rows": int(len(base_frame)),
        "base_feature_count": int(len(feature_cols)),
        "outer_fold_count": int(len(outer_city_keys)),
        "trainable_city_count": int(len(trainable_city_keys)),
        "extra_bg_feature_paths": [str(path) for path in extra_feature_paths],
    }
    return pd.DataFrame(rows), meta, timings


def _between_permutation_rows(
    *,
    paths: RepoPaths,
    year: int,
    feature_groups: dict[str, list[str]],
    config: ModelSurfaceConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object], list[Timing]]:
    timings: list[Timing] = []
    started = time.perf_counter()
    controls, _bg_crosswalk, _bg, training, feature_cols, train_state_cols, base_x, _bg_x, extra_feature_cols = _prepare_model_surface_context(
        paths=paths,
        config=config,
        extra_bg_feature_paths=None,
    )
    timings.append(
        Timing(
            "between_load_training_frame",
            time.perf_counter() - started,
            f"jurisdictions={len(training)} feature_count={len(feature_cols) + len(train_state_cols)}",
        )
    )
    rows: list[dict[str, object]] = []
    fit_started = time.perf_counter()
    for offense in OFFENSES_7:
        offense_state = _build_offense_training_state(
            training=training,
            controls=controls,
            base_x=base_x,
            config=config,
            offense=offense,
            feature_cols=feature_cols,
            train_state_cols=train_state_cols,
        )
        frame = offense_state["frame"]
        counts = offense_state["counts"]
        pop = offense_state["pop"]
        observed_only_mask = offense_state["observed_only_mask"]
        train_mask_np = np.asarray(offense_state["train_mask_np"], dtype=bool)
        x_train = offense_state["x_train"].copy()
        model_feature_cols = list(offense_state["model_feature_cols"])
        fit_rate, _pooling_meta = _build_sparse_offense_pooling_target(
            frame=frame,
            counts=counts,
            pop=pop,
            offense=offense,
            config=config,
            observed_only_mask=observed_only_mask,
        )
        y_fit = np.log1p(fit_rate)
        y_train = np.asarray(y_fit[train_mask_np], dtype=float)
        actual_counts = pd.to_numeric(counts.loc[offense_state["train_mask"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        train_pop = pd.to_numeric(pop.loc[offense_state["train_mask"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        train_frame = frame.loc[offense_state["train_mask"]].reset_index(drop=True)
        kfold = KFold(
            n_splits=int(offense_state["cv_fold_count"]),
            shuffle=True,
            random_state=0,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(x_train), start=1):
            model = _build_model(config, feature_names=model_feature_cols)
            x_fold_train = x_train.iloc[train_idx].replace([np.inf, -np.inf], np.nan)
            x_fold_test = x_train.iloc[test_idx].replace([np.inf, -np.inf], np.nan)
            with threadpool_limits(limits=1):
                model.fit(x_fold_train, y_train[train_idx])
                baseline_log_rate = np.asarray(model.predict(x_fold_test), dtype=float)
            baseline_rate = np.clip(np.expm1(np.clip(baseline_log_rate, 0.0, 12.0)), 0.0, None)
            baseline_count = baseline_rate / 1e5 * train_pop[test_idx]
            baseline_dev = _poisson_deviance(actual_counts[test_idx], baseline_count)
            test_meta = train_frame.iloc[test_idx].reset_index(drop=True)
            for feature_group, columns in feature_groups.items():
                group_cols = [col for col in columns if col in x_fold_test.columns]
                if group_cols:
                    x_perm = _permute_feature_group(
                        x_fold_test,
                        columns=group_cols,
                        seed=seed + _seed("between", offense, fold_idx, feature_group) % 100000,
                        bg_key_col=None,
                    )
                    with threadpool_limits(limits=1):
                        perm_log_rate = np.asarray(model.predict(x_perm), dtype=float)
                    perm_rate = np.clip(np.expm1(np.clip(perm_log_rate, 0.0, 12.0)), 0.0, None)
                    perm_count = perm_rate / 1e5 * train_pop[test_idx]
                    perm_dev = _poisson_deviance(actual_counts[test_idx], perm_count)
                else:
                    perm_dev = baseline_dev
                    perm_count = baseline_count
                delta = perm_dev - baseline_dev
                for i, row in enumerate(test_meta.itertuples(index=False)):
                    rows.append(
                        {
                            "feature_group": str(feature_group),
                            "unit_id": f"{getattr(row, 'jurisdiction_id')}|{offense}",
                            "jurisdiction_id": str(getattr(row, "jurisdiction_id")),
                            "state_fips": str(getattr(row, "state_fips")).zfill(2),
                            "offense": str(offense),
                            "fold_id": f"kfold_{fold_idx}",
                            "delta_deviance": float(delta[i]),
                            "baseline_deviance": float(baseline_dev[i]),
                            "permuted_deviance": float(perm_dev[i]),
                            "actual_count": float(actual_counts[test_idx][i]),
                            "baseline_predicted_count": float(baseline_count[i]),
                            "permuted_predicted_count": float(perm_count[i]),
                            "permuted_column_count": int(len(group_cols)),
                        }
                    )
        print(f"[feature-policy] between offense {offense} complete", flush=True)
    timings.append(Timing("between_fit_and_permute", time.perf_counter() - fit_started, f"offenses={len(OFFENSES_7)} groups={len(feature_groups)}"))
    meta = {
        "training_rows": int(len(training)),
        "base_feature_count": int(len(feature_cols)),
        "state_fixed_effect_count": int(len(train_state_cols)),
        "extra_feature_count": int(len(extra_feature_cols)),
    }
    return pd.DataFrame(rows), meta, timings


def _build_feature_inventory(
    *,
    groups_to_score: set[str] | None,
    within_feature_cols: list[str],
    between_feature_cols: list[str],
    feature_meta_path: Path,
) -> pd.DataFrame:
    meta_frames: list[pd.DataFrame] = []
    if feature_meta_path.exists():
        meta = pd.read_parquet(feature_meta_path).copy()
        meta["feature_column"] = meta["feature_column"].astype(str)
        meta_frames.append(meta[["feature_column", "feature_source"]].drop_duplicates())
    within = pd.DataFrame(
        {
            "feature_column": [str(col) for col in within_feature_cols],
            "within_axis_feature": True,
        }
    )
    between = pd.DataFrame(
        {
            "feature_column": [str(col) for col in between_feature_cols],
            "between_axis_feature": True,
        }
    )
    features = pd.concat([within[["feature_column"]], between[["feature_column"]]], ignore_index=True).drop_duplicates()
    for frame in meta_frames:
        features = features.merge(frame, on="feature_column", how="left")
    features = features.merge(within, on="feature_column", how="left").merge(between, on="feature_column", how="left")
    features["within_axis_feature"] = features["within_axis_feature"].map(lambda value: bool(value) if pd.notna(value) else False)
    features["between_axis_feature"] = features["between_axis_feature"].map(lambda value: bool(value) if pd.notna(value) else False)
    features["feature_group"] = features["feature_column"].map(_policy_group_for_feature)
    fallback_source = pd.Series(
        np.where(
            features["feature_column"].str.startswith("state_"),
            "state_fixed_effect",
            np.where(features["between_axis_feature"], "base_covariate", "residual_allocator_only"),
        ),
        index=features.index,
    )
    features["feature_source"] = features["feature_source"].fillna(fallback_source)
    if groups_to_score is not None:
        features = features[features["feature_group"].isin(groups_to_score)].copy()
    flag_frame = pd.DataFrame([_policy_flags_for_group(group) for group in features["feature_group"]])
    features = pd.concat([features.reset_index(drop=True), flag_frame.reset_index(drop=True)], axis=1)
    return features.sort_values(["feature_group", "feature_column"], kind="mergesort").reset_index(drop=True)


def _wide_axis_metrics(
    summary: pd.DataFrame,
    *,
    prefix: str,
) -> dict[str, dict[str, object]]:
    by_group: dict[str, dict[str, object]] = {}
    if summary.empty:
        return by_group
    for row in summary.itertuples(index=False):
        group = str(getattr(row, "feature_group"))
        offense = str(getattr(row, "offense"))
        dest = by_group.setdefault(group, {})
        suffix = offense
        dest[f"{prefix}_{suffix}"] = getattr(row, "delta")
        dest[f"{prefix}_ci95_low_{suffix}"] = getattr(row, "ci95_low")
        dest[f"{prefix}_ci95_high_{suffix}"] = getattr(row, "ci95_high")
        dest[f"{prefix}_transfers_{suffix}"] = bool(getattr(row, "transfers"))
        dest[f"{prefix}_unit_count_{suffix}"] = int(getattr(row, "unit_count"))
    return by_group


def _classify_group(
    *,
    feature_group: str,
    within_metrics: dict[str, object],
    between_metrics: dict[str, object],
) -> tuple[str, str]:
    flags = _policy_flags_for_group(feature_group)
    if flags["hard_excluded_protected_flag"]:
        return (
            "excluded_protected",
            "Hard exclusion: FHA-protected-class or protected-class-adjacent demographic composition; predictive power is not considered.",
        )
    if flags["proxy_review_flag"]:
        return (
            "proxy_review",
            "Kept for now but flagged as redlining-adjacent SES proxy; later estimator steps must treat it as review-gated.",
        )
    within_transfer = bool(within_metrics.get("delta_tvd_transfers_overall")) or any(
        bool(value)
        for key, value in within_metrics.items()
        if key.startswith("delta_tvd_transfers_") and key != "delta_tvd_transfers_overall"
    )
    between_transfer = bool(between_metrics.get("delta_deviance_transfers_overall")) or any(
        bool(value)
        for key, value in between_metrics.items()
        if key.startswith("delta_deviance_transfers_") and key != "delta_deviance_transfers_overall"
    )
    if between_transfer and within_transfer:
        return (
            "between_and_within",
            "Permutation degradation is stable on both held-out jurisdiction totals and held-out within-city shares; evidence supports local-mechanism transfer.",
        )
    if within_transfer:
        return (
            "within_only",
            "Permutation degradation is stable for held-out within-city allocation but not for jurisdiction-level totals.",
        )
    if between_transfer:
        return (
            "between_only",
            "Permutation degradation is stable for jurisdiction-level totals but not for within-city allocation; this is ecological/redlining-prone for within-area allocation.",
        )
    return (
        "unstable_drop",
        "Neither held-out axis has a stable permutation degradation above the practical floor.",
    )


def _assemble_artifact(
    *,
    feature_inventory: pd.DataFrame,
    within_summary: pd.DataFrame,
    between_summary: pd.DataFrame,
    tvd_floor: float,
    deviance_floor: float,
    bootstrap_iterations: int,
    run_mode: str,
) -> pd.DataFrame:
    within_wide = _wide_axis_metrics(within_summary, prefix="delta_tvd")
    between_wide = _wide_axis_metrics(between_summary, prefix="delta_deviance")
    rows: list[dict[str, object]] = []
    group_counts = feature_inventory.groupby("feature_group", dropna=False)["feature_column"].transform("count")
    inventory = feature_inventory.assign(feature_count_in_group=group_counts)
    for feature in inventory.itertuples(index=False):
        group = str(feature.feature_group)
        wm = within_wide.get(group, {})
        bm = between_wide.get(group, {})
        final_class, rationale = _classify_group(feature_group=group, within_metrics=wm, between_metrics=bm)
        row: dict[str, object] = {
            "feature_column": str(feature.feature_column),
            "feature_group": group,
            "feature_source": str(feature.feature_source),
            "source": str(feature.feature_source),
            "within_axis_feature": bool(feature.within_axis_feature),
            "between_axis_feature": bool(feature.between_axis_feature),
            "feature_count_in_group": int(feature.feature_count_in_group),
            "direct_protected_flag": bool(feature.direct_protected_flag),
            "hard_excluded_protected_flag": bool(feature.hard_excluded_protected_flag),
            "proxy_review_flag": bool(feature.proxy_review_flag),
            "local_mechanism_flag": bool(feature.local_mechanism_flag),
            "final_class": final_class,
            "rationale": rationale,
            "run_mode": str(run_mode),
            "methodology": "fold_safe_group_permutation_importance",
            "within_metric": "held_out_city_offense_delta_tvd_incident_weighted",
            "between_metric": "held_out_jurisdiction_offense_delta_poisson_count_deviance_mean",
            "tvd_practical_floor": float(tvd_floor),
            "deviance_practical_floor": float(deviance_floor),
            "bootstrap_iterations": int(bootstrap_iterations),
            "created_at_utc": _now_iso(),
        }
        row.update(wm)
        row.update(bm)
        for offense in ("overall", *OFFENSES_7):
            row.setdefault(f"delta_tvd_{offense}", None)
            row.setdefault(f"delta_tvd_ci95_low_{offense}", None)
            row.setdefault(f"delta_tvd_ci95_high_{offense}", None)
            row.setdefault(f"delta_tvd_transfers_{offense}", False)
            row.setdefault(f"delta_deviance_{offense}", None)
            row.setdefault(f"delta_deviance_ci95_low_{offense}", None)
            row.setdefault(f"delta_deviance_ci95_high_{offense}", None)
            row.setdefault(f"delta_deviance_transfers_{offense}", False)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["feature_group", "feature_column"], kind="mergesort").reset_index(drop=True)


def _write_summary(
    *,
    summary_path: Path,
    artifact: pd.DataFrame,
    within_meta: dict[str, object],
    between_meta: dict[str, object],
    timings: list[Timing],
    args: argparse.Namespace,
) -> None:
    group_summary = (
        artifact.drop_duplicates("feature_group")
        .sort_values("feature_group", kind="mergesort")
        [
            [
                "feature_group",
                "final_class",
                "feature_count_in_group",
                "delta_tvd_overall",
                "delta_tvd_ci95_low_overall",
                "delta_tvd_ci95_high_overall",
                "delta_tvd_transfers_overall",
                "delta_deviance_overall",
                "delta_deviance_ci95_low_overall",
                "delta_deviance_ci95_high_overall",
                "delta_deviance_transfers_overall",
                "rationale",
            ]
        ]
        .to_dict(orient="records")
    )
    payload = {
        "created_at_utc": _now_iso(),
        "run_mode": str(args.mode),
        "artifact_path": str(args.artifact_out),
        "methodology": {
            "fit_rule": "fit each held-out fold once per axis; permute feature-group columns only in held-out rows",
            "within_axis": "city-blocked residual allocator folds from nested_city_cv_harness; metric is incident-weighted held-out TVD increase",
            "between_axis": "jurisdiction model 5-fold CV; metric is held-out Poisson count deviance increase",
            "bootstrap_unit": "city/offense for within, jurisdiction/offense for between",
            "classification_floor": {
                "delta_tvd": float(args.tvd_floor),
                "delta_deviance": float(args.deviance_floor),
            },
        },
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "seed": int(args.seed),
        "groups_scored": sorted(artifact["feature_group"].dropna().astype(str).unique().tolist()),
        "within_meta": _clean_json(within_meta),
        "between_meta": _clean_json(between_meta),
        "timings": [
            {
                "step": timing.step,
                "elapsed_sec": float(timing.elapsed_sec),
                "detail": timing.detail,
            }
            for timing in timings
        ],
        "group_summary": _clean_json(group_summary),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_clean_json(payload), indent=2, sort_keys=True, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--groups", type=str, default=None, help="Comma-separated policy groups to score.")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20240623)
    parser.add_argument("--tvd-floor", type=float, default=DEFAULT_TVD_FLOOR)
    parser.add_argument("--deviance-floor", type=float, default=DEFAULT_DEVIANCE_FLOOR)
    parser.add_argument("--hist-learning-rate", type=float, default=0.03)
    parser.add_argument("--hist-max-depth", type=int, default=5)
    parser.add_argument("--hist-max-iter", type=int, default=500)
    parser.add_argument("--hist-min-samples-leaf", type=int, default=20)
    parser.add_argument("--hist-l2-regularization", type=float, default=1.0)
    parser.add_argument(
        "--role-inventory-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "city_role_inventory_2024.parquet",
    )
    parser.add_argument(
        "--city-shares-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet",
    )
    parser.add_argument(
        "--bg-prior-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "bg_prior_long_2024.parquet",
    )
    parser.add_argument(
        "--bg-crosswalk-path",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument(
        "--feature-meta-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "jurisdiction_model_features_2024.parquet",
    )
    parser.add_argument(
        "--artifact-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "feature_transfer_policy_2024.parquet",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "feature_transfer_policy_2024.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    residual_config = CityResidualConfig(
        hist_learning_rate=float(args.hist_learning_rate),
        hist_max_depth=int(args.hist_max_depth),
        hist_max_iter=int(args.hist_max_iter),
        hist_min_samples_leaf=int(args.hist_min_samples_leaf),
        hist_l2_regularization=float(args.hist_l2_regularization),
        extra_feature_paths=tuple(promoted_residual_extra_bg_feature_paths(paths)),
    )
    between_config = ModelSurfaceConfig(
        year=int(args.year),
        hist_learning_rate=float(args.hist_learning_rate),
        hist_max_depth=int(args.hist_max_depth),
        hist_max_iter=int(args.hist_max_iter),
        hist_min_samples_leaf=int(args.hist_min_samples_leaf),
        hist_l2_regularization=float(args.hist_l2_regularization),
        compute_diagnostics=False,
    )

    print("[feature-policy] loading feature columns for inventory", flush=True)
    base_frame, residual_feature_cols, _role_meta, _roles = nested_cv._build_base_frame(
        paths=paths,
        year=int(args.year),
        role_inventory_path=args.role_inventory_path,
        city_shares_path=args.city_shares_path,
        bg_prior_path=args.bg_prior_path,
        bg_crosswalk_path=args.bg_crosswalk_path,
        extra_feature_paths=list(promoted_residual_extra_bg_feature_paths(paths)),
        feature_policy_path=None,
        exclude_feature_policy_classes=(),
    )
    if base_frame.empty:
        raise SystemExit("No evaluable city/offense rows available.")
    _controls, _bg_crosswalk, _bg, _training, jurisdiction_feature_cols, train_state_cols, _base_x, _bg_x, _extra = _prepare_model_surface_context(
        paths=paths,
        config=between_config,
        extra_bg_feature_paths=None,
    )
    all_between_cols = [*jurisdiction_feature_cols, *train_state_cols]

    requested_groups: set[str] | None = None
    if args.groups:
        requested_groups = {part.strip() for part in str(args.groups).split(",") if part.strip()}
    elif args.mode == "pilot":
        requested_groups = set(DEFAULT_PILOT_GROUPS)

    feature_inventory = _build_feature_inventory(
        groups_to_score=requested_groups,
        within_feature_cols=list(residual_feature_cols),
        between_feature_cols=all_between_cols,
        feature_meta_path=args.feature_meta_path,
    )
    if feature_inventory.empty:
        raise SystemExit("No features selected for the requested group scope.")
    feature_groups = {
        group: sorted(group_df["feature_column"].astype(str).tolist())
        for group, group_df in feature_inventory.groupby("feature_group", dropna=False, sort=True)
    }
    print(
        f"[feature-policy] scoring {len(feature_groups)} policy groups across {len(feature_inventory)} feature rows",
        flush=True,
    )

    within_rows, within_meta, within_timings = _within_permutation_rows(
        paths=paths,
        year=int(args.year),
        feature_groups=feature_groups,
        config=residual_config,
        role_inventory_path=args.role_inventory_path,
        city_shares_path=args.city_shares_path,
        bg_prior_path=args.bg_prior_path,
        bg_crosswalk_path=args.bg_crosswalk_path,
        seed=int(args.seed),
    )
    within_summary = _summarize_axis(
        within_rows,
        value_col="delta_tvd",
        weight_col="incident_total",
        n_boot=int(args.bootstrap_iterations),
        seed=int(args.seed) + 101,
        floor=float(args.tvd_floor),
    )

    between_rows, between_meta, between_timings = _between_permutation_rows(
        paths=paths,
        year=int(args.year),
        feature_groups=feature_groups,
        config=between_config,
        seed=int(args.seed),
    )
    between_summary = _summarize_axis(
        between_rows,
        value_col="delta_deviance",
        weight_col=None,
        n_boot=int(args.bootstrap_iterations),
        seed=int(args.seed) + 202,
        floor=float(args.deviance_floor),
    )

    artifact = _assemble_artifact(
        feature_inventory=feature_inventory,
        within_summary=within_summary,
        between_summary=between_summary,
        tvd_floor=float(args.tvd_floor),
        deviance_floor=float(args.deviance_floor),
        bootstrap_iterations=int(args.bootstrap_iterations),
        run_mode=str(args.mode),
    )
    args.artifact_out.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_parquet(args.artifact_out, index=False)
    timings = [
        Timing("total", time.perf_counter() - started, f"mode={args.mode} groups={len(feature_groups)}"),
        *within_timings,
        *between_timings,
    ]
    _write_summary(
        summary_path=args.summary_json_out,
        artifact=artifact,
        within_meta=within_meta,
        between_meta=between_meta,
        timings=timings,
        args=args,
    )
    print(f"[feature-policy] wrote artifact: {args.artifact_out}", flush=True)
    print(f"[feature-policy] wrote summary: {args.summary_json_out}", flush=True)
    print(f"[feature-policy] elapsed_sec={time.perf_counter() - started:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
