"""Leakage-free nested city-blocked validation for 2024 city share transfer.

This is validation infrastructure only. It reads the frozen upstream BG prior,
city role inventory, city truth surfaces, and published/count-reliability outputs;
it does not modify production allocation or estimator code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any
import warnings

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
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.allocation import (
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    promoted_residual_extra_bg_feature_paths,
)
from crimerisk.city_residuals import (
    CityResidualConfig,
    apply_city_residual_model,
    fit_city_residual_model,
    prepare_city_residual_frame,
)
from crimerisk.city_share_benchmark import safe_corr
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths


EVALUABLE_ROLES = ("direct_posterior_live", "residual_training_only", "validation_holdout_only")
TRAINABLE_ROLES = ("direct_posterior_live", "residual_training_only")
SPARSE_OFFENSES = frozenset(("murder", "rape"))
EPS = 1e-12
RELIABILITY_STEMS = (
    "direct_incident_support_flag",
    "direct_incident_support_count",
    "direct_incident_support_years",
    "direct_incident_support_year_min",
    "direct_incident_support_year_max",
    "effective_numerator_support",
    "numerator_support_source",
    "rate_ci95_lower",
    "rate_ci95_upper",
    "index_ci95_lower",
    "index_ci95_upper",
    "index_ci95_width",
    "index_ci95_width_ratio",
    "reliability_tier",
    "recommended_display_geography",
)
PRIMARY_CI_STEM_TO_TEMPLATE = {
    "rate_ci95_lower": "rate_{offense}_primary_ci95_lower",
    "rate_ci95_upper": "rate_{offense}_primary_ci95_upper",
    "index_ci95_lower": "index_{offense}_primary_ci95_lower",
    "index_ci95_upper": "index_{offense}_primary_ci95_upper",
    "index_ci95_width": "index_{offense}_primary_ci95_width",
    "index_ci95_width_ratio": "index_{offense}_primary_ci95_width_ratio",
}

_BASE_FRAME: pd.DataFrame | None = None
_BASE_FEATURE_COLS: tuple[str, ...] = ()
_FIT_CONFIG: CityResidualConfig | None = None


def _reliability_col(stem: str, offense: str) -> str:
    template = PRIMARY_CI_STEM_TO_TEMPLATE.get(stem)
    if template is not None:
        return template.format(offense=offense)
    return f"{stem}_{offense}"


@dataclass(frozen=True)
class PredictionJob:
    excluded_train_city_keys: tuple[str, ...]
    predict_city_keys: tuple[str, ...]


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
    if pd.isna(value) if value is not None else False:
        return None
    return value


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _seed_offset(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") % 100000


def _join_key(parts: tuple[str, ...]) -> str:
    return "|".join(sorted(str(part) for part in parts if str(part)))


def _normalize_share(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(np.nansum(arr))
    if np.isfinite(total) and total > 0:
        return arr / total
    if len(arr) == 0:
        return arr
    return np.full(len(arr), 1.0 / float(len(arr)))


def _tempered_share(model_share: pd.Series | np.ndarray, predicted_log_ratio: pd.Series | np.ndarray, tau: float) -> np.ndarray:
    base = np.clip(np.asarray(model_share, dtype=float), 0.0, None)
    eta = np.asarray(predicted_log_ratio, dtype=float)
    weights = base * np.exp(float(tau) * np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0))
    return _normalize_share(weights)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = v.notna() & np.isfinite(v) & w.gt(0.0)
    if not bool(mask.any()):
        return None
    return float(np.average(v.loc[mask].to_numpy(dtype=float), weights=w.loc[mask].to_numpy(dtype=float)))


def _equal_city_mean(df: pd.DataFrame, value_col: str) -> float | None:
    if df.empty or value_col not in df.columns:
        return None
    city_values = (
        df.groupby("outer_city_key", dropna=False)[value_col]
        .mean(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if city_values.empty:
        return None
    return float(city_values.mean())


def _select_fold_feature_cols(frame: pd.DataFrame, feature_cols: tuple[str, ...]) -> list[str]:
    selected: list[str] = []
    n = max(int(len(frame)), 1)
    for col in feature_cols:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if float(series.notna().mean()) < 0.01:
            continue
        if int(series.nunique(dropna=True)) < 2:
            continue
        if float(series.fillna(0.0).ne(0.0).sum()) / float(n) < 0.001:
            continue
        selected.append(col)
    return selected


def _fit_predict_job(job: PredictionJob) -> pd.DataFrame:
    if _BASE_FRAME is None or _FIT_CONFIG is None:
        raise RuntimeError("nested CV worker globals were not initialized")

    excluded = set(job.excluded_train_city_keys)
    predict = set(job.predict_city_keys)
    frame = _BASE_FRAME
    train = frame[
        frame["inventory_role"].isin(TRAINABLE_ROLES)
        & ~frame["city_key"].astype(str).isin(excluded)
    ].copy()
    test = frame[frame["city_key"].astype(str).isin(predict)].copy()
    fold_id = _join_key(tuple(sorted(excluded)))
    if test.empty:
        return pd.DataFrame()

    feature_cols = _select_fold_feature_cols(train, _BASE_FEATURE_COLS)
    with threadpool_limits(limits=1):
        fitted = fit_city_residual_model(
            train,
            feature_cols=feature_cols,
            config=_FIT_CONFIG,
        )
        test = apply_city_residual_model(test, fitted=fitted)
    if "predicted_log_ratio" not in test.columns:
        test["predicted_log_ratio"] = 0.0
    test["excluded_train_city_keys"] = fold_id
    test["excluded_train_city_count"] = int(len(excluded))
    test["predicted_city_key"] = test["city_key"].astype(str)
    test["fit_training_city_count"] = int(train["city_key"].astype(str).nunique())
    test["fit_training_row_count"] = int(len(train))
    test["fit_feature_column_count"] = int(len(feature_cols))
    test["fit_training_incident_total"] = float(
        train.groupby(["city_key", "offense"], dropna=False)["incident_count"]
        .sum()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum()
    )
    keep = [
        "excluded_train_city_keys",
        "excluded_train_city_count",
        "predicted_city_key",
        "city_key",
        "city_name",
        "jurisdiction_id",
        "state_fips",
        "offense",
        "inventory_role",
        "bg_id",
        "incident_count",
        "incident_total",
        "true_share",
        "model_share",
        "predicted_log_ratio",
        "residual_model_share",
        "primary_geocode_quality_tier",
        "fallback_incident_share",
        "offense_volume_band",
        "fit_training_city_count",
        "fit_training_row_count",
        "fit_feature_column_count",
        "fit_training_incident_total",
    ]
    return test[[col for col in keep if col in test.columns]].copy()


def _run_prediction_jobs(
    *,
    jobs: list[PredictionJob],
    max_workers: int,
) -> pd.DataFrame:
    if max_workers <= 1:
        frames: list[pd.DataFrame] = []
        for idx, job in enumerate(jobs, start=1):
            print(
                f"[nested-cv] fit {idx}/{len(jobs)} excluded={_join_key(job.excluded_train_city_keys) or '<none>'}",
                flush=True,
            )
            frames.append(_fit_predict_job(job))
        return pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()

    def collect_from_executor(executor: ProcessPoolExecutor | ThreadPoolExecutor, *, mode: str) -> pd.DataFrame:
        frames = []
        future_to_job = {executor.submit(_fit_predict_job, job): job for job in jobs}
        done = 0
        for future in as_completed(future_to_job):
            done += 1
            job = future_to_job[future]
            frames.append(future.result())
            print(
                f"[nested-cv] fit {done}/{len(jobs)} complete ({mode}) "
                f"excluded={_join_key(job.excluded_train_city_keys) or '<none>'}",
                flush=True,
            )
        return pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    try:
        with ProcessPoolExecutor(max_workers=int(max_workers), mp_context=ctx) as executor:
            return collect_from_executor(executor, mode="process")
    except PermissionError as exc:
        print(
            f"[nested-cv] process pool unavailable ({exc}); falling back to threads",
            flush=True,
        )
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        return collect_from_executor(executor, mode="thread")


def _build_prediction_jobs(trainable_city_keys: list[str], validation_city_keys: list[str]) -> list[PredictionJob]:
    jobs: list[PredictionJob] = []
    if validation_city_keys:
        jobs.append(PredictionJob(excluded_train_city_keys=(), predict_city_keys=tuple(validation_city_keys)))
    for city_key in trainable_city_keys:
        jobs.append(PredictionJob(excluded_train_city_keys=(city_key,), predict_city_keys=(city_key,)))
    for left, right in itertools.combinations(trainable_city_keys, 2):
        jobs.append(PredictionJob(excluded_train_city_keys=(left, right), predict_city_keys=(left, right)))
    return jobs


def _per_city_offense_tvd_grid(pred: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (city_key, offense), group in pred.groupby(["predicted_city_key", "offense"], dropna=False, sort=True):
        truth = _normalize_share(group["true_share"])
        model = pd.to_numeric(group["model_share"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        eta = pd.to_numeric(group["predicted_log_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        incident_total = float(pd.to_numeric(group["incident_count"], errors="coerce").fillna(0.0).sum())
        for tau in taus:
            pred_share = _tempered_share(model, eta, float(tau))
            rows.append(
                {
                    "city_key": str(city_key),
                    "offense": str(offense),
                    "tau": float(tau),
                    "tvd": float(0.5 * np.abs(pred_share - truth).sum()),
                    "incident_total": incident_total,
                }
            )
    return pd.DataFrame(rows)


def _select_tau(
    inner_pred: pd.DataFrame,
    *,
    offense: str,
    taus: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, object] | None:
    grid = _per_city_offense_tvd_grid(inner_pred[inner_pred["offense"].astype(str).eq(str(offense))], taus)
    if grid.empty:
        return None

    pivot = grid.pivot_table(index="city_key", columns="tau", values="tvd", aggfunc="mean")
    if pivot.empty:
        return None
    tau_cols = np.array([float(col) for col in pivot.columns])
    mat = pivot.to_numpy(dtype=float)
    w = (
        grid.groupby("city_key", dropna=False)["incident_total"]
        .first()
        .reindex(pivot.index)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    def _wmean(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        denom = float(weights.sum())
        if denom > 0:
            return (rows * weights[:, None]).sum(axis=0) / denom
        return rows.mean(axis=0)

    curve = _wmean(mat, w)
    curve_equal = mat.mean(axis=0)
    selectable = tau_cols <= 1.0 + 1e-9
    tau01 = tau_cols[selectable]
    curve01 = curve[selectable]
    argmin = int(np.argmin(curve01))
    tau_argmin = float(tau01[argmin])
    tvd_min = float(curve01[argmin])

    rng = np.random.default_rng(int(seed))
    n_cities = int(len(pivot.index))
    boot_minima: list[float] = []
    if n_cities > 1 and n_boot > 1:
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n_cities, size=n_cities)
            boot_minima.append(float(np.nanmin(_wmean(mat[idx], w[idx])[selectable])))
    se = float(np.std(boot_minima, ddof=1)) if len(boot_minima) > 1 else 0.0
    within = np.where(curve01 <= tvd_min + se)[0]
    selected_idx = int(within.min()) if within.size else argmin
    tau_one_se = float(tau01[selected_idx])

    tvd_at_0 = float(curve[np.isclose(tau_cols, 0.0)][0]) if np.isclose(tau_cols, 0.0).any() else None
    tvd_at_1 = float(curve[np.isclose(tau_cols, 1.0)][0]) if np.isclose(tau_cols, 1.0).any() else None
    tvd_at_selected = float(curve01[selected_idx])
    return {
        "offense": str(offense),
        "inner_city_count": n_cities,
        "inner_incident_total": float(grid[["city_key", "incident_total"]].drop_duplicates()["incident_total"].sum()),
        "tau_argmin": tau_argmin,
        "tau_one_se": tau_one_se,
        "tvd_at_tau_argmin": tvd_min,
        "tvd_at_tau_one_se": tvd_at_selected,
        "tvd_at_0_exposure": tvd_at_0,
        "tvd_at_1_residual": tvd_at_1,
        "gain_vs_exposure": (float(tvd_at_0) - tvd_at_selected) if tvd_at_0 is not None else None,
        "gain_vs_residual": (float(tvd_at_1) - tvd_at_selected) if tvd_at_1 is not None else None,
        "bootstrap_se": se,
        "tvd_curve_incident_weighted": {f"{t:.4f}": float(v) for t, v in zip(tau_cols, curve, strict=False)},
        "tvd_curve_equal_city": {f"{t:.4f}": float(v) for t, v in zip(tau_cols, curve_equal, strict=False)},
    }


def _calibration_curve(group: pd.DataFrame, predicted_share: np.ndarray) -> tuple[list[dict[str, object]], float | None]:
    if group.empty:
        return [], None
    truth = _normalize_share(group["true_share"])
    pred = _normalize_share(predicted_share)
    order = np.argsort(-pred)
    chunks = np.array_split(order, min(10, len(order)))
    curve: list[dict[str, object]] = []
    errors: list[float] = []
    for idx, chunk in enumerate(chunks, start=1):
        obs = float(truth[chunk].sum())
        exp = float(pred[chunk].sum())
        errors.append(abs(obs - exp))
        curve.append(
            {
                "decile_rank_pred_desc": int(idx),
                "n_bg": int(len(chunk)),
                "observed_mass": obs,
                "predicted_mass": exp,
                "observed_predicted_mass_ratio": (obs / exp) if exp > 0 else None,
            }
        )
    ice = float(np.mean(errors)) if errors else None
    return curve, ice


def _score_variant(
    group: pd.DataFrame,
    *,
    predicted_share: np.ndarray,
    baseline_top_decile_capture: float,
) -> dict[str, object]:
    truth = _normalize_share(group["true_share"])
    pred = _normalize_share(predicted_share)
    diff = pred - truth
    n_top = max(1, int(np.ceil(len(group) * 0.10)))
    top_idx = np.argsort(-pred)[:n_top]
    top_capture = float(truth[top_idx].sum())
    curve, ice = _calibration_curve(group, pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spearman = safe_corr(pd.Series(truth), pd.Series(pred), method="spearman")
    return {
        "bg_count": int(len(group)),
        "truth_nonzero_bg_count": int(np.count_nonzero(truth > 0)),
        "predicted_nonzero_bg_count": int(np.count_nonzero(pred > 0)),
        "tvd": float(0.5 * np.abs(diff).sum()),
        "top_decile_capture": top_capture,
        "top_decile_capture_delta_vs_exposure": float(top_capture - baseline_top_decile_capture),
        "spearman_share": spearman,
        "calibration_observed_predicted_mass_ratio": float(truth.sum() / pred.sum()) if pred.sum() > 0 else None,
        "integrated_calibration_error": ice,
        "calibration_decile_curve_json": json.dumps(_clean_json(curve), sort_keys=True),
    }


def _tier_mode(values: pd.Series) -> str | None:
    cleaned = [str(v) for v in values.dropna().tolist() if str(v) and str(v).lower() != "nan"]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _city_offense_reliability(
    group: pd.DataFrame,
    *,
    reliability_bg: pd.DataFrame | None,
    offense: str,
) -> dict[str, object]:
    out = {
        "direct_incident_support_flag_share": None,
        "direct_incident_support_count_sum": None,
        "direct_incident_support_years_max": None,
        "effective_numerator_support_sum": None,
        "numerator_support_source_mode": None,
        "rate_ci95_lower_truth_weighted_mean": None,
        "rate_ci95_upper_truth_weighted_mean": None,
        "index_ci95_lower_truth_weighted_mean": None,
        "index_ci95_upper_truth_weighted_mean": None,
        "index_ci95_width_ratio_truth_weighted_mean": None,
        "reliability_tier_mode": None,
        "recommended_display_geography_mode": None,
    }
    if reliability_bg is None or reliability_bg.empty:
        return out
    rel = group[["bg_id", "true_share"]].merge(reliability_bg, on="bg_id", how="left")
    weights = pd.to_numeric(rel["true_share"], errors="coerce").fillna(0.0)

    def col(stem: str) -> str:
        return _reliability_col(stem, offense)

    if col("direct_incident_support_flag") in rel.columns:
        out["direct_incident_support_flag_share"] = float(rel[col("direct_incident_support_flag")].fillna(False).astype(bool).mean())
    if col("direct_incident_support_count") in rel.columns:
        out["direct_incident_support_count_sum"] = float(pd.to_numeric(rel[col("direct_incident_support_count")], errors="coerce").fillna(0.0).sum())
    if col("direct_incident_support_years") in rel.columns:
        out["direct_incident_support_years_max"] = float(pd.to_numeric(rel[col("direct_incident_support_years")], errors="coerce").fillna(0.0).max())
    if col("effective_numerator_support") in rel.columns:
        out["effective_numerator_support_sum"] = float(pd.to_numeric(rel[col("effective_numerator_support")], errors="coerce").fillna(0.0).sum())
    if col("numerator_support_source") in rel.columns:
        out["numerator_support_source_mode"] = _tier_mode(rel[col("numerator_support_source")])
    for stem in (
        "rate_ci95_lower",
        "rate_ci95_upper",
        "index_ci95_lower",
        "index_ci95_upper",
        "index_ci95_width_ratio",
    ):
        if col(stem) in rel.columns:
            out[f"{stem}_truth_weighted_mean"] = _weighted_mean(rel[col(stem)], weights)
    if col("reliability_tier") in rel.columns:
        out["reliability_tier_mode"] = _tier_mode(rel[col("reliability_tier")])
    if col("recommended_display_geography") in rel.columns:
        out["recommended_display_geography_mode"] = _tier_mode(rel[col("recommended_display_geography")])
    return out


def _transfer_policy_for_offense(selection: dict[str, object] | None, offense: str) -> tuple[str, float]:
    if offense in SPARSE_OFFENSES:
        return "sparse_reliability_deferred_baseline", 0.0
    if selection is None:
        return "insufficient_inner_support_exposure_baseline", 0.0
    tau = float(selection.get("tau_one_se") or 0.0)
    gain = selection.get("gain_vs_exposure")
    if tau > 0.0 and gain is not None and float(gain) > 0.0:
        return "inner_cv_tempered_residual", tau
    return "inner_cv_exposure_baseline", 0.0


def _score_outer_fold(
    *,
    outer_city_key: str,
    outer_pred: pd.DataFrame,
    inner_pred: pd.DataFrame,
    taus: np.ndarray,
    tau_bootstrap_iterations: int,
    seed: int,
    reliability_bg: pd.DataFrame | None,
    training_incident_total_by_excluded_key: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if outer_pred.empty:
        return [], []

    tau_by_offense = {
        offense: _select_tau(
            inner_pred,
            offense=offense,
            taus=taus,
            n_boot=int(tau_bootstrap_iterations),
            seed=int(seed) + i * 1009,
        )
        for i, offense in enumerate(OFFENSES_7)
    }
    tau_rows = [
        {"outer_city_key": str(outer_city_key), **selection}
        for selection in tau_by_offense.values()
        if selection is not None
    ]
    rows: list[dict[str, object]] = []
    for (_, offense), group in outer_pred.groupby(["city_key", "offense"], dropna=False, sort=True):
        group = group.copy()
        offense = str(offense)
        city_key = str(group["city_key"].iloc[0])
        selection = tau_by_offense.get(offense)
        selected_policy, selected_tau = _transfer_policy_for_offense(selection, offense)
        diagnostic_tau = float(selection.get("tau_one_se") or 0.0) if selection is not None else 0.0
        baseline_share = _normalize_share(group["model_share"])
        baseline_top = _score_variant(
            group,
            predicted_share=baseline_share,
            baseline_top_decile_capture=0.0,
        )["top_decile_capture"]
        variant_specs = [
            ("exposure_baseline", "frozen_upstream_bg_prior_baseline", 0.0, baseline_share),
            (
                "outer_residual_tau1",
                "outer_excluded_full_residual_diagnostic",
                1.0,
                _normalize_share(group["residual_model_share"]),
            ),
            (
                "nested_inner_tau",
                "inner_cv_tau_diagnostic",
                diagnostic_tau,
                _tempered_share(group["model_share"], group["predicted_log_ratio"], diagnostic_tau),
            ),
            (
                "nested_selected_policy",
                selected_policy,
                selected_tau,
                _tempered_share(group["model_share"], group["predicted_log_ratio"], selected_tau),
            ),
        ]
        reliability = _city_offense_reliability(group, reliability_bg=reliability_bg, offense=offense)
        for variant, policy, tau_applied, pred_share in variant_specs:
            metrics = _score_variant(group, predicted_share=pred_share, baseline_top_decile_capture=float(baseline_top))
            row = {
                "outer_city_key": city_key,
                "outer_city_name": str(group["city_name"].iloc[0]),
                "jurisdiction_id": str(group["jurisdiction_id"].iloc[0]),
                "state_fips": str(group["state_fips"].iloc[0]).zfill(2),
                "offense": offense,
                "inventory_role": str(group["inventory_role"].iloc[0]),
                "sparsity_class": "sparse" if offense in SPARSE_OFFENSES else "dense",
                "model_variant": variant,
                "transfer_policy": policy,
                "tau_applied": float(tau_applied),
                "inner_tau_one_se": selection.get("tau_one_se") if selection is not None else None,
                "inner_tau_argmin": selection.get("tau_argmin") if selection is not None else None,
                "inner_tvd_at_0_exposure": selection.get("tvd_at_0_exposure") if selection is not None else None,
                "inner_tvd_at_1_residual": selection.get("tvd_at_1_residual") if selection is not None else None,
                "inner_tvd_at_tau_one_se": selection.get("tvd_at_tau_one_se") if selection is not None else None,
                "inner_gain_vs_exposure": selection.get("gain_vs_exposure") if selection is not None else None,
                "inner_city_count": selection.get("inner_city_count") if selection is not None else 0,
                "incident_total": float(pd.to_numeric(group["incident_count"], errors="coerce").fillna(0.0).sum()),
                "primary_geocode_quality_tier": str(group["primary_geocode_quality_tier"].iloc[0]),
                "fallback_incident_share": float(pd.to_numeric(group["fallback_incident_share"], errors="coerce").fillna(0.0).iloc[0]),
                "offense_volume_band": str(group["offense_volume_band"].iloc[0]),
                "fit_excluded_train_city_keys": str(group["excluded_train_city_keys"].iloc[0]),
                "fit_training_city_count": int(group["fit_training_city_count"].iloc[0]),
                "fit_training_row_count": int(group["fit_training_row_count"].iloc[0]),
                "fit_feature_column_count": int(group["fit_feature_column_count"].iloc[0]),
                "fit_training_incident_total": float(
                    training_incident_total_by_excluded_key.get(
                        str(group["excluded_train_city_keys"].iloc[0]),
                        float(group["fit_training_incident_total"].iloc[0]),
                    )
                ),
                **metrics,
                **reliability,
            }
            rows.append(row)
    return rows, tau_rows


def _bootstrap_ci(
    df: pd.DataFrame,
    *,
    value_col: str,
    mode: str,
    n_boot: int,
    seed: int,
) -> dict[str, float | None]:
    if df.empty or value_col not in df.columns:
        return {"low": None, "high": None}
    work = df[["outer_city_key", value_col, "incident_total"]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work["incident_total"] = pd.to_numeric(work["incident_total"], errors="coerce").fillna(0.0)
    work = work[work[value_col].notna() & np.isfinite(work[value_col])]
    if work.empty:
        return {"low": None, "high": None}
    if mode == "incident_weighted":
        work["_weighted_value"] = work[value_col] * work["incident_total"]
        city = (
            work.groupby("outer_city_key", dropna=False, sort=True)
            .agg(numerator=("_weighted_value", "sum"), denominator=("incident_total", "sum"))
            .reset_index()
        )
    else:
        city = (
            work.groupby("outer_city_key", dropna=False, sort=True)
            .agg(city_value=(value_col, "mean"))
            .reset_index()
        )
    n_cities = int(len(city))
    if n_cities <= 1 or n_boot <= 1:
        return {"low": None, "high": None}
    rng = np.random.default_rng(int(seed))
    stats: list[float] = []
    sample_size = n_cities
    if mode == "incident_weighted":
        numerators = city["numerator"].to_numpy(dtype=float)
        denominators = city["denominator"].to_numpy(dtype=float)
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n_cities, size=sample_size)
            denom = float(denominators[idx].sum())
            if denom > 0:
                stats.append(float(numerators[idx].sum() / denom))
    else:
        city_values = city["city_value"].to_numpy(dtype=float)
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n_cities, size=sample_size)
            stat = float(np.nanmean(city_values[idx]))
            if np.isfinite(stat):
                stats.append(stat)
    if not stats:
        return {"low": None, "high": None}
    return {
        "low": float(np.percentile(stats, 2.5)),
        "high": float(np.percentile(stats, 97.5)),
    }


def _summarize_group(
    df: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    metrics = {
        "rows": int(len(df)),
        "city_count": int(df["outer_city_key"].nunique()) if not df.empty else 0,
        "offense_count": int(df["offense"].nunique()) if not df.empty else 0,
        "incident_total": float(pd.to_numeric(df.get("incident_total"), errors="coerce").fillna(0.0).sum()) if not df.empty else 0.0,
    }
    for col in (
        "tvd",
        "top_decile_capture",
        "top_decile_capture_delta_vs_exposure",
        "spearman_share",
        "integrated_calibration_error",
    ):
        metrics[f"{col}_incident_weighted_mean"] = _weighted_mean(df[col], df["incident_total"]) if col in df.columns else None
        metrics[f"{col}_equal_city_mean"] = _equal_city_mean(df, col)
    for col in ("tvd", "top_decile_capture_delta_vs_exposure", "integrated_calibration_error"):
        metrics[f"{col}_incident_weighted_city_bootstrap_ci95"] = _bootstrap_ci(
            df,
            value_col=col,
            mode="incident_weighted",
            n_boot=n_boot,
            seed=seed + _seed_offset("iw", col),
        )
        metrics[f"{col}_equal_city_bootstrap_ci95"] = _bootstrap_ci(
            df,
            value_col=col,
            mode="equal_city",
            n_boot=n_boot,
            seed=seed + _seed_offset("eq", col),
        )
    return metrics


def _aggregate_rows(
    diagnostics: pd.DataFrame,
    *,
    group_cols: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    if diagnostics.empty:
        return []
    rows: list[dict[str, object]] = []
    for keys, group in diagnostics.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: str(value) for col, value in zip(group_cols, keys, strict=False)}
        row.update(_summarize_group(group, n_boot=n_boot, seed=seed + len(rows) * 997))
        rows.append(row)
    return rows


def _resolve_reliability_output_path(repo_root: Path, requested: Path | None) -> tuple[Path | None, list[str], dict[str, list[str]]]:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            repo_root / "state" / "output" / "crimerisk_block_group_2024_fbi_calibrated.parquet",
            repo_root / "state" / "candidates" / "count_reliability_layer" / "crimerisk_block_group_2024_fbi_calibrated.parquet",
            repo_root / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet",
            repo_root / "state" / "candidates" / "count_reliability_layer" / "crimerisk_block_group_2024_ags_core.parquet",
        ]
    )
    seen: set[Path] = set()
    for path in candidates:
        path = Path(path)
        if path in seen or not path.exists():
            continue
        seen.add(path)
        schema_cols = pq.ParquetFile(path).schema.names
        reliability_cols = [
            c
            for c in schema_cols
            if any(c == _reliability_col(stem, offense) for stem in RELIABILITY_STEMS for offense in OFFENSES_7)
        ]
        if reliability_cols:
            by_offense = {
                offense: [
                    c
                    for c in reliability_cols
                    if any(c == _reliability_col(stem, offense) for stem in RELIABILITY_STEMS)
                ]
                for offense in OFFENSES_7
            }
            return path, reliability_cols, by_offense
    return None, [], {offense: [] for offense in OFFENSES_7}


def _load_reliability_bg(path: Path | None, reliability_cols: list[str]) -> pd.DataFrame | None:
    if path is None or not reliability_cols:
        return None
    schema_cols = pq.ParquetFile(path).schema.names
    id_col = "block_group_geoid" if "block_group_geoid" in schema_cols else "bg_id"
    rel = pd.read_parquet(path, columns=[id_col, *reliability_cols]).copy()
    rel = rel.rename(columns={id_col: "bg_id"})
    rel["bg_id"] = rel["bg_id"].astype("string").str.zfill(12).astype(str)
    return rel.drop_duplicates("bg_id")


def _build_base_frame(
    *,
    paths: RepoPaths,
    year: int,
    role_inventory_path: Path,
    city_shares_path: Path,
    bg_prior_path: Path,
    bg_crosswalk_path: Path,
    extra_feature_paths: list[Path],
    feature_policy_path: Path | None,
    exclude_feature_policy_classes: tuple[str, ...],
    exclude_feature_policy_classes_by_offense: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    offenses: set[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    roles = pd.read_parquet(role_inventory_path).copy()
    roles["city_key"] = roles["city_key"].astype(str)
    roles["jurisdiction_id"] = roles["jurisdiction_id"].astype(str)
    roles["state_fips"] = roles["state_fips"].astype("string").str.zfill(2).astype(str)
    roles["offense"] = roles["offense"].astype(str)
    roles = roles[roles["role"].isin(EVALUABLE_ROLES)].copy()
    if offenses is not None:
        roles = roles[roles["offense"].isin(offenses)].copy()
    role_meta = roles[
        [
            "city_key",
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "offense",
            "role",
            "residual_validation_case_type",
        ]
    ].drop_duplicates(["jurisdiction_id", "offense"]).rename(columns={"role": "inventory_role"})

    city_shares = pd.read_parquet(city_shares_path).copy()
    city_shares["jurisdiction_id"] = city_shares["jurisdiction_id"].astype(str)
    city_shares["offense"] = city_shares["offense"].astype(str)
    if offenses is not None:
        city_shares = city_shares[city_shares["offense"].isin(offenses)].copy()
    city_shares = city_shares.merge(
        role_meta[["city_key", "jurisdiction_id", "offense", "inventory_role"]],
        on=["jurisdiction_id", "offense"],
        how="inner",
    )
    bg_prior = pd.read_parquet(bg_prior_path)
    bg_crosswalk = pd.read_parquet(bg_crosswalk_path)
    frame, feature_cols = prepare_city_residual_frame(
        paths=paths,
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=int(year),
        extra_feature_paths=extra_feature_paths,
        feature_policy_path=feature_policy_path,
        exclude_feature_policy_classes=exclude_feature_policy_classes,
        exclude_feature_policy_classes_by_offense=exclude_feature_policy_classes_by_offense,
    )
    if frame.empty:
        return frame, tuple(feature_cols), role_meta, roles

    frame = frame.merge(
        role_meta[
            [
                "city_key",
                "jurisdiction_id",
                "offense",
                "inventory_role",
                "residual_validation_case_type",
            ]
        ],
        on=["jurisdiction_id", "offense"],
        how="left",
    )
    frame["city_key"] = frame["city_key"].astype(str)
    frame["inventory_role"] = frame["inventory_role"].astype(str)
    frame["bg_id"] = frame["bg_id"].astype("string").str.zfill(12).astype(str)
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2).astype(str)
    for col in ("incident_count", "incident_total", "true_share", "model_share"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["incident_total"] = frame.groupby(["city_key", "offense"], dropna=False)["incident_count"].transform("sum")
    return frame, tuple(feature_cols), role_meta, roles


def _collect_outer_predictions(
    *,
    prediction_cache: pd.DataFrame,
    outer_city_key: str,
    trainable_city_keys: set[str],
) -> pd.DataFrame:
    excluded_key = outer_city_key if outer_city_key in trainable_city_keys else ""
    pred = prediction_cache[
        prediction_cache["excluded_train_city_keys"].astype(str).eq(excluded_key)
        & prediction_cache["predicted_city_key"].astype(str).eq(str(outer_city_key))
    ].copy()
    return pred


def _collect_inner_predictions(
    *,
    prediction_cache: pd.DataFrame,
    outer_city_key: str,
    trainable_city_keys: list[str],
) -> pd.DataFrame:
    rows = []
    outer_trainable = outer_city_key in set(trainable_city_keys)
    for inner_city_key in trainable_city_keys:
        if inner_city_key == outer_city_key:
            continue
        excluded = _join_key(tuple(sorted((outer_city_key, inner_city_key)))) if outer_trainable else inner_city_key
        one = prediction_cache[
            prediction_cache["excluded_train_city_keys"].astype(str).eq(excluded)
            & prediction_cache["predicted_city_key"].astype(str).eq(str(inner_city_key))
        ]
        if not one.empty:
            rows.append(one)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--hist-learning-rate", type=float, default=0.03)
    parser.add_argument("--hist-max-depth", type=int, default=5)
    parser.add_argument("--hist-max-iter", type=int, default=500)
    parser.add_argument("--hist-min-samples-leaf", type=int, default=20)
    parser.add_argument("--hist-l2-regularization", type=float, default=1.0)
    parser.add_argument("--tau-max", type=float, default=1.5)
    parser.add_argument("--tau-steps", type=int, default=61)
    parser.add_argument("--tau-bootstrap-iterations", type=int, default=200)
    parser.add_argument("--summary-bootstrap-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20240622)
    parser.add_argument("--reuse-prediction-cache", action="store_true")
    parser.add_argument(
        "--offense",
        action="append",
        choices=OFFENSES_7,
        default=[],
        help="Restrict the harness to one offense. May be repeated.",
    )
    parser.add_argument(
        "--feature-policy-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "feature_transfer_policy_2024.parquet",
        help="Feature policy used for residual feature exclusions and burglary exposure-duplicate detection.",
    )
    parser.add_argument(
        "--no-feature-policy",
        action="store_true",
        help="Disable all feature-policy-backed residual exclusions and burglary exposure-duplicate exclusions.",
    )
    parser.add_argument(
        "--residual-exclude-feature-policy-class",
        action="append",
        default=list(DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES),
        help="Feature-transfer final_class to exclude from residual allocator features. May be repeated.",
    )
    parser.add_argument(
        "--no-residual-feature-policy-exclusions",
        action="store_true",
        help=(
            "Keep the feature-policy file available for existing burglary exposure-duplicate handling, "
            "but do not exclude residual features by final_class."
        ),
    )
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=REPO_ROOT / "state" / "cache" / "nested_city_cv_prediction_cache_2024.parquet",
    )
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
    parser.add_argument("--count-reliability-output-path", type=Path, default=None)
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "nested_city_cv_2024.parquet",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "nested_city_cv_2024.json",
    )
    args = parser.parse_args()

    year = int(args.year)
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    extra_feature_paths = list(promoted_residual_extra_bg_feature_paths(paths))
    offense_filter = {str(value) for value in args.offense} if args.offense else None
    feature_policy_path = None if bool(args.no_feature_policy) else args.feature_policy_path
    residual_exclude_classes = (
        ()
        if bool(args.no_feature_policy) or bool(args.no_residual_feature_policy_exclusions)
        else tuple(str(value) for value in args.residual_exclude_feature_policy_class)
    )
    base_frame, feature_cols, role_meta, evaluable_inventory = _build_base_frame(
        paths=paths,
        year=year,
        role_inventory_path=args.role_inventory_path,
        city_shares_path=args.city_shares_path,
        bg_prior_path=args.bg_prior_path,
        bg_crosswalk_path=args.bg_crosswalk_path,
        extra_feature_paths=extra_feature_paths,
        feature_policy_path=feature_policy_path,
        exclude_feature_policy_classes=residual_exclude_classes,
        exclude_feature_policy_classes_by_offense=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
        offenses=offense_filter,
    )
    if base_frame.empty:
        raise SystemExit("No evaluable city/offense rows available for nested city CV.")

    trainable_city_keys = sorted(
        base_frame.loc[base_frame["inventory_role"].isin(TRAINABLE_ROLES), "city_key"].dropna().astype(str).unique().tolist()
    )
    outer_city_keys = sorted(base_frame["city_key"].dropna().astype(str).unique().tolist())
    validation_city_keys = sorted(set(outer_city_keys) - set(trainable_city_keys))
    prediction_jobs = _build_prediction_jobs(trainable_city_keys, validation_city_keys)
    print(
        f"[nested-cv] {len(outer_city_keys)} outer cities; {len(trainable_city_keys)} trainable cities; "
        f"{len(validation_city_keys)} test-only cities; {len(prediction_jobs)} residual fit jobs",
        flush=True,
    )

    global _BASE_FRAME, _BASE_FEATURE_COLS, _FIT_CONFIG
    _BASE_FRAME = base_frame
    _BASE_FEATURE_COLS = tuple(feature_cols)
    _FIT_CONFIG = CityResidualConfig(
        hist_learning_rate=float(args.hist_learning_rate),
        hist_max_depth=int(args.hist_max_depth),
        hist_max_iter=int(args.hist_max_iter),
        hist_min_samples_leaf=int(args.hist_min_samples_leaf),
        hist_l2_regularization=float(args.hist_l2_regularization),
        extra_feature_paths=tuple(extra_feature_paths),
        feature_policy_path=feature_policy_path,
        exclude_feature_policy_classes=residual_exclude_classes,
        exclude_feature_policy_classes_by_offense=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    )

    if args.reuse_prediction_cache and args.prediction_cache.exists():
        print(f"[nested-cv] loading prediction cache: {args.prediction_cache}", flush=True)
        prediction_cache = pd.read_parquet(args.prediction_cache)
    else:
        prediction_cache = _run_prediction_jobs(jobs=prediction_jobs, max_workers=int(args.jobs))
        args.prediction_cache.parent.mkdir(parents=True, exist_ok=True)
        prediction_cache.to_parquet(args.prediction_cache, index=False)
        print(f"[nested-cv] wrote prediction cache: {args.prediction_cache}", flush=True)

    reliability_path, reliability_cols, reliability_cols_by_offense = _resolve_reliability_output_path(
        REPO_ROOT,
        args.count_reliability_output_path,
    )
    reliability_bg = _load_reliability_bg(reliability_path, reliability_cols)
    if reliability_path is not None:
        print(f"[nested-cv] count-reliability fields: {reliability_path}", flush=True)
    else:
        print("[nested-cv] count-reliability fields not found; sparse rows will carry null reliability summaries", flush=True)

    trainable_base = base_frame[base_frame["inventory_role"].isin(TRAINABLE_ROLES)].copy()
    training_incident_total_by_excluded_key: dict[str, float] = {}
    for excluded_key in sorted(prediction_cache["excluded_train_city_keys"].astype(str).fillna("").unique().tolist()):
        excluded = {part for part in str(excluded_key).split("|") if part}
        train = trainable_base[~trainable_base["city_key"].astype(str).isin(excluded)]
        total = (
            train.groupby(["city_key", "offense"], dropna=False)["incident_count"]
            .sum()
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum()
        )
        training_incident_total_by_excluded_key[str(excluded_key)] = float(total)

    taus = np.round(np.linspace(0.0, float(args.tau_max), int(args.tau_steps)), 4)
    diagnostic_rows: list[dict[str, object]] = []
    tau_rows: list[dict[str, object]] = []
    trainable_set = set(trainable_city_keys)
    for idx, outer_city_key in enumerate(outer_city_keys, start=1):
        outer_pred = _collect_outer_predictions(
            prediction_cache=prediction_cache,
            outer_city_key=outer_city_key,
            trainable_city_keys=trainable_set,
        )
        inner_pred = _collect_inner_predictions(
            prediction_cache=prediction_cache,
            outer_city_key=outer_city_key,
            trainable_city_keys=trainable_city_keys,
        )
        rows, selections = _score_outer_fold(
            outer_city_key=outer_city_key,
            outer_pred=outer_pred,
            inner_pred=inner_pred,
            taus=taus,
            tau_bootstrap_iterations=int(args.tau_bootstrap_iterations),
            seed=int(args.seed) + idx * 7919,
            reliability_bg=reliability_bg,
            training_incident_total_by_excluded_key=training_incident_total_by_excluded_key,
        )
        diagnostic_rows.extend(rows)
        tau_rows.extend(selections)
        print(
            f"[nested-cv] scored outer {idx}/{len(outer_city_keys)} {outer_city_key}: "
            f"{len(rows)} variant rows",
            flush=True,
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    if diagnostics.empty:
        raise SystemExit("Nested city CV produced no diagnostics rows.")

    expected_pairs = evaluable_inventory[["city_key", "offense", "role"]].drop_duplicates().copy()
    observed_pairs = diagnostics[["outer_city_key", "offense"]].drop_duplicates().rename(
        columns={"outer_city_key": "city_key"}
    )
    missing_pairs = expected_pairs.merge(observed_pairs, on=["city_key", "offense"], how="left", indicator=True)
    missing_pairs = missing_pairs[missing_pairs["_merge"].eq("left_only")].drop(columns="_merge")

    diagnostics = diagnostics.sort_values(
        ["outer_city_key", "offense", "model_variant"],
        kind="mergesort",
    ).reset_index(drop=True)
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(args.diagnostics_out, index=False)

    tau_selection = pd.DataFrame(tau_rows)
    selected_policy = diagnostics[diagnostics["model_variant"].eq("nested_selected_policy")].copy()
    aggregate = {
        "overall_by_variant": _aggregate_rows(
            diagnostics,
            group_cols=["model_variant"],
            n_boot=int(args.summary_bootstrap_iterations),
            seed=int(args.seed) + 11,
        ),
        "by_offense": _aggregate_rows(
            diagnostics,
            group_cols=["model_variant", "offense"],
            n_boot=int(args.summary_bootstrap_iterations),
            seed=int(args.seed) + 101,
        ),
        "by_inventory_role": _aggregate_rows(
            diagnostics,
            group_cols=["model_variant", "inventory_role"],
            n_boot=int(args.summary_bootstrap_iterations),
            seed=int(args.seed) + 211,
        ),
        "by_role_offense": _aggregate_rows(
            diagnostics,
            group_cols=["model_variant", "inventory_role", "offense"],
            n_boot=int(args.summary_bootstrap_iterations),
            seed=int(args.seed) + 307,
        ),
        "by_city": _aggregate_rows(
            diagnostics,
            group_cols=["model_variant", "outer_city_key"],
            n_boot=int(args.summary_bootstrap_iterations),
            seed=int(args.seed) + 401,
        ),
        "by_sparsity_class": _aggregate_rows(
            diagnostics,
            group_cols=["model_variant", "sparsity_class"],
            n_boot=int(args.summary_bootstrap_iterations),
            seed=int(args.seed) + 503,
        ),
        "by_offense_volume_band": _aggregate_rows(
            diagnostics,
            group_cols=["model_variant", "offense_volume_band"],
            n_boot=int(args.summary_bootstrap_iterations),
            seed=int(args.seed) + 601,
        ),
        "sparse_by_reliability_tier": _aggregate_rows(
            diagnostics[diagnostics["sparsity_class"].eq("sparse")],
            group_cols=["model_variant", "reliability_tier_mode"],
            n_boot=int(args.summary_bootstrap_iterations),
            seed=int(args.seed) + 701,
        ),
    }

    summary = {
        "year": year,
        "created_at_utc": _now_iso(),
        "harness": "nested_city_blocked_cv",
        "diagnostics_path": str(args.diagnostics_out),
        "summary_json_path": str(args.summary_json_out),
        "prediction_cache_path": str(args.prediction_cache),
        "split_design": {
            "outer_fold_unit": "city_key",
            "outer_fold_count": int(len(outer_city_keys)),
            "evaluable_roles": list(EVALUABLE_ROLES),
            "trainable_roles": list(TRAINABLE_ROLES),
            "test_only_roles": ["validation_holdout_only"],
            "outer_city_keys": outer_city_keys,
            "trainable_city_keys": trainable_city_keys,
            "validation_holdout_only_city_keys": validation_city_keys,
        },
        "per_stage_fold_exclusion": {
            "residual_allocator_training_excludes_outer_city": True,
            "fold_feature_selection_excludes_outer_city": True,
            "tau_and_offense_transfer_policy_use_inner_city_folds_excluding_outer_city": True,
            "posterior_or_direct_city_inputs_mask_outer_city_from_prediction": True,
            "validation_holdout_only_never_trained_on": True,
            "onboarded_not_integrated_and_rejected_excluded": True,
        },
        "upstream_baseline_scope": {
            "bg_prior_long_2024": str(args.bg_prior_path),
            "official_jurisdiction_totals": "frozen upstream controls; not re-fit or outer-city excluded in this step",
            "stricter_end_to_end_total_refit": "deferred enhancement",
            "covered_city_same_city_incident_time_split": "deferred enhancement",
        },
        "residual_fit_config": {
            "hist_learning_rate": float(args.hist_learning_rate),
            "hist_max_depth": int(args.hist_max_depth),
            "hist_max_iter": int(args.hist_max_iter),
            "hist_min_samples_leaf": int(args.hist_min_samples_leaf),
            "hist_l2_regularization": float(args.hist_l2_regularization),
            "extra_bg_feature_paths": [str(path) for path in extra_feature_paths],
            "offense_filter": sorted(offense_filter) if offense_filter is not None else None,
            "feature_policy_path": str(feature_policy_path) if feature_policy_path is not None else None,
            "feature_policy_enabled": feature_policy_path is not None,
            "exclude_feature_policy_classes": sorted(residual_exclude_classes),
            "exclude_feature_policy_classes_by_offense": {
                str(offense): [str(value) for value in classes]
                for offense, classes in DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE
            },
            "base_candidate_feature_count": int(len(feature_cols)),
            "feature_policy_application": _clean_json(
                base_frame.attrs.get("city_residual_feature_policy_application", {})
            ),
            "residual_fit_job_count": int(len(prediction_jobs)),
        },
        "tau_selection": {
            "grid": [float(t) for t in taus],
            "rule": "inner_city_blocked_one_SE_conservative_incident_weighted_tau_in_0_1",
            "tau_bootstrap_iterations": int(args.tau_bootstrap_iterations),
            "sparse_offense_policy": "murder and rape are reported with reliability/support bands; nested_selected_policy keeps baseline instead of applying aggressive in-fold movement",
            "outer_offense_rows": _clean_json(tau_selection.to_dict(orient="records")) if not tau_selection.empty else [],
        },
        "count_reliability": {
            "source_path": str(reliability_path) if reliability_path is not None else None,
            "columns_detected": reliability_cols,
            "columns_by_offense": reliability_cols_by_offense,
        },
        "row_counts": {
            "diagnostic_rows": int(len(diagnostics)),
            "city_offense_cases": int(diagnostics[["outer_city_key", "offense"]].drop_duplicates().shape[0]),
            "model_variants": sorted(diagnostics["model_variant"].unique().tolist()),
            "missing_evaluable_city_offense_cases": int(len(missing_pairs)),
        },
        "missing_evaluable_city_offense_cases": _clean_json(missing_pairs.to_dict(orient="records")),
        "selected_policy_counts": (
            selected_policy.groupby(["sparsity_class", "transfer_policy"], dropna=False)
            .size()
            .reset_index(name="rows")
            .to_dict(orient="records")
        ),
        "aggregate": _clean_json(aggregate),
        "supersedes_as_heldout_evidence": [
            "state/modeling/city_share_benchmark_2024.*",
            "state/modeling/city_residual_benchmark_2024.*",
            "state/modeling/next_phase_city_residual_benchmark_2024.*",
            "state/modeling/share_temper_tau_calibration_2024.json",
        ],
    }
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.write_text(json.dumps(_clean_json(summary), indent=2, sort_keys=True, allow_nan=False))

    print(f"[nested-cv] wrote diagnostics: {args.diagnostics_out}", flush=True)
    print(f"[nested-cv] wrote summary: {args.summary_json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
