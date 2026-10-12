from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut


@dataclass(frozen=True)
class CrossValidatedDiagnostics:
    cv_r2_log_rate: float
    cv_rmse_log_rate: float
    cv_r2_rate: float
    calibration: list[dict[str, float | int]]
    residual_by_state: list[dict[str, float | int]]
    residual_by_pop_band: list[dict[str, float | int]]


@dataclass(frozen=True)
class FeatureSanityDiagnostics:
    training_rows: int
    feature_count: int
    min_non_null_share: float
    p10_non_null_share: float
    p50_non_null_share: float
    p90_non_null_share: float
    exact_or_near_complement_pairs: list[dict[str, float | str]]
    strongest_positive_pairs: list[dict[str, float | str]]
    strongest_negative_pairs: list[dict[str, float | str]]
    high_abs_correlation_pairs: list[dict[str, float | str]]
    effective_rank: int
    condition_number: float | None
    feature_group_counts: list[dict[str, float | int | str]]
    low_coverage_features: list[dict[str, float | str]]


def _pop_band(pop: float) -> str:
    if not np.isfinite(pop) or pop < 0:
        return "unknown"
    if pop < 5_000:
        return "<5k"
    if pop < 10_000:
        return "5k-10k"
    if pop < 25_000:
        return "10k-25k"
    if pop < 50_000:
        return "25k-50k"
    if pop < 100_000:
        return "50k-100k"
    if pop < 250_000:
        return "100k-250k"
    if pop < 500_000:
        return "250k-500k"
    if pop < 1_000_000:
        return "500k-1m"
    return "1m+"


def cross_validated_diagnostics(
    *,
    x: pd.DataFrame,
    y_log_rate: np.ndarray,
    y_eval_log_rate: np.ndarray | None = None,
    state_fips: np.ndarray,
    bucket_population: np.ndarray,
    model_factory: Callable[[], object],
    group_labels: np.ndarray | None = None,
    eligible_holdout_groups: np.ndarray | None = None,
    split_mode: str = "kfold",
    n_splits: int = 5,
    random_state: int = 0,
) -> CrossValidatedDiagnostics:
    y = np.asarray(y_log_rate, dtype=float)
    y_eval = np.asarray(y_eval_log_rate, dtype=float) if y_eval_log_rate is not None else np.asarray(y_log_rate, dtype=float)
    states = np.asarray(state_fips, dtype=str)
    pops = np.asarray(bucket_population, dtype=float)
    if len(x) != len(y) or len(y) != len(y_eval) or len(y) != len(states) or len(y) != len(pops):
        raise ValueError("x, y_log_rate, y_eval_log_rate, state_fips, bucket_population must have the same length")

    preds = np.full(len(y), np.nan, dtype=float)
    if split_mode == "kfold":
        splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
        split_iter = splitter.split(x)
    elif split_mode == "leave_group_out":
        if group_labels is None:
            raise ValueError("group_labels are required for leave_group_out diagnostics")
        groups = np.asarray(group_labels)
        if len(groups) != len(y):
            raise ValueError("group_labels must have the same length as y_log_rate")
        valid_groups = pd.Series(groups).fillna("unknown").astype(str).to_numpy()
        if len(set(valid_groups.tolist())) < 2:
            empty = CrossValidatedDiagnostics(
                cv_r2_log_rate=float("nan"),
                cv_rmse_log_rate=float("nan"),
                cv_r2_rate=float("nan"),
                calibration=[],
                residual_by_state=[],
                residual_by_pop_band=[],
            )
            return empty
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(x, y, groups=valid_groups)
    elif split_mode == "leave_selected_groups_out":
        if group_labels is None:
            raise ValueError("group_labels are required for leave_selected_groups_out diagnostics")
        groups = np.asarray(group_labels)
        if len(groups) != len(y):
            raise ValueError("group_labels must have the same length as y_log_rate")
        valid_groups = pd.Series(groups).fillna("unknown").astype(str).to_numpy()
        if eligible_holdout_groups is None:
            selected_groups = sorted(set(valid_groups.tolist()))
        else:
            eligible = pd.Series(np.asarray(eligible_holdout_groups)).dropna().astype(str)
            eligible_set = set(eligible.tolist())
            selected_groups = [group for group in pd.unique(valid_groups).tolist() if str(group) in eligible_set]
        split_iter = []
        for group in selected_groups:
            train_idx = np.flatnonzero(valid_groups != group)
            test_idx = np.flatnonzero(valid_groups == group)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            split_iter.append((train_idx, test_idx))
        if len(split_iter) == 0:
            empty = CrossValidatedDiagnostics(
                cv_r2_log_rate=float("nan"),
                cv_rmse_log_rate=float("nan"),
                cv_r2_rate=float("nan"),
                calibration=[],
                residual_by_state=[],
                residual_by_pop_band=[],
            )
            return empty
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    for train_idx, test_idx in split_iter:
        model = model_factory()
        x_train = x.iloc[train_idx].replace([np.inf, -np.inf], np.nan)
        x_test = x.iloc[test_idx].replace([np.inf, -np.inf], np.nan)
        model.fit(x_train, y[train_idx])
        preds[test_idx] = np.asarray(model.predict(x_test), dtype=float)

    mask = np.isfinite(y_eval) & np.isfinite(preds)
    if int(mask.sum()) < 5:
        empty = CrossValidatedDiagnostics(
            cv_r2_log_rate=float("nan"),
            cv_rmse_log_rate=float("nan"),
            cv_r2_rate=float("nan"),
            calibration=[],
            residual_by_state=[],
            residual_by_pop_band=[],
        )
        return empty

    resid = y_eval[mask] - preds[mask]
    mse = float(np.mean(resid**2))
    rmse = float(np.sqrt(mse))
    cv_r2 = float(r2_score(y_eval[mask], preds[mask]))

    rate_true = np.expm1(y_eval[mask])
    rate_pred = np.expm1(preds[mask])
    cv_r2_rate = float(r2_score(rate_true, rate_pred))

    # Calibration: bin by predicted rate quantiles.
    calib_rows: list[dict[str, float | int]] = []
    try:
        bins = pd.qcut(rate_pred, q=10, duplicates="drop")
        df = pd.DataFrame({"bin": bins.astype(str), "rate_true": rate_true, "rate_pred": rate_pred})
        for bin_name, grp in df.groupby("bin", sort=False):
            calib_rows.append(
                {
                    "bin": str(bin_name),
                    "n": int(len(grp)),
                    "mean_rate_pred": float(np.mean(grp["rate_pred"])),
                    "mean_rate_true": float(np.mean(grp["rate_true"])),
                }
            )
    except Exception:
        calib_rows = []

    # Residual patterns by state and by population band.
    states_m = states[mask]
    pops_m = pops[mask]

    by_state_rows: list[dict[str, float | int]] = []
    for st in sorted(set(states_m.tolist())):
        sel = states_m == st
        if int(sel.sum()) == 0:
            continue
        r = resid[sel]
        by_state_rows.append(
            {
                "state_fips": str(st),
                "n": int(sel.sum()),
                "mean_resid_log_rate": float(np.mean(r)),
                "mean_abs_resid_log_rate": float(np.mean(np.abs(r))),
            }
        )

    bands = np.array([_pop_band(p) for p in pops_m], dtype=str)
    by_band_rows: list[dict[str, float | int]] = []
    for band in ["<5k", "5k-10k", "10k-25k", "25k-50k", "50k-100k", "100k-250k", "250k-500k", "500k-1m", "1m+", "unknown"]:
        sel = bands == band
        if int(sel.sum()) == 0:
            continue
        r = resid[sel]
        by_band_rows.append(
            {
                "pop_band": str(band),
                "n": int(sel.sum()),
                "mean_resid_log_rate": float(np.mean(r)),
                "mean_abs_resid_log_rate": float(np.mean(np.abs(r))),
            }
        )

    return CrossValidatedDiagnostics(
        cv_r2_log_rate=cv_r2,
        cv_rmse_log_rate=rmse,
        cv_r2_rate=cv_r2_rate,
        calibration=calib_rows,
        residual_by_state=by_state_rows,
        residual_by_pop_band=by_band_rows,
    )


def feature_sanity_diagnostics(
    *,
    x: pd.DataFrame,
    feature_group_resolver: Callable[[str], str] | None = None,
    low_coverage_threshold: float = 0.95,
    high_abs_correlation_threshold: float = 0.95,
    top_k_pairs: int = 15,
) -> FeatureSanityDiagnostics:
    numeric = x.apply(pd.to_numeric, errors="coerce")
    if numeric.empty:
        return FeatureSanityDiagnostics(
            training_rows=0,
            feature_count=0,
            min_non_null_share=float("nan"),
            p10_non_null_share=float("nan"),
            p50_non_null_share=float("nan"),
            p90_non_null_share=float("nan"),
            exact_or_near_complement_pairs=[],
            strongest_positive_pairs=[],
            strongest_negative_pairs=[],
            high_abs_correlation_pairs=[],
            effective_rank=0,
            condition_number=None,
            feature_group_counts=[],
            low_coverage_features=[],
        )

    non_null_share = numeric.notna().mean(axis=0).astype(float)
    coverage_values = non_null_share.to_numpy(dtype=float)

    corr = numeric.corr(numeric_only=True)
    corr_vals = corr.to_numpy(dtype=float)
    cols = corr.columns.tolist()
    pair_rows: list[dict[str, float | str]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr_vals[i, j]
            if not np.isfinite(value):
                continue
            pair_rows.append(
                {
                    "feature_a": str(cols[i]),
                    "feature_b": str(cols[j]),
                    "correlation": float(value),
                    "abs_correlation": float(abs(value)),
                }
            )
    pair_rows_sorted = sorted(pair_rows, key=lambda row: float(row["abs_correlation"]), reverse=True)

    complement_pairs = [
        row for row in pair_rows_sorted if float(row["correlation"]) <= -0.98
    ][: int(top_k_pairs)]
    strongest_positive = [
        row for row in sorted(pair_rows, key=lambda row: float(row["correlation"]), reverse=True)
        if float(row["correlation"]) > 0
    ][: int(top_k_pairs)]
    strongest_negative = [
        row for row in sorted(pair_rows, key=lambda row: float(row["correlation"]))
        if float(row["correlation"]) < 0
    ][: int(top_k_pairs)]
    high_abs_pairs = [
        row for row in pair_rows_sorted if float(row["abs_correlation"]) >= float(high_abs_correlation_threshold)
    ][: int(top_k_pairs)]

    imputed = numeric.fillna(numeric.median(axis=0))
    centered = imputed - imputed.mean(axis=0)
    scale = centered.std(axis=0, ddof=0).replace(0, 1.0)
    standardized = centered.divide(scale, axis=1).to_numpy(dtype=float)
    singular_values = np.linalg.svd(standardized, compute_uv=False, full_matrices=False)
    finite_sv = singular_values[np.isfinite(singular_values)]
    positive_sv = finite_sv[finite_sv > 1e-12]
    if len(positive_sv) == 0:
        effective_rank = 0
        condition_number = None
    else:
        effective_rank = int(len(positive_sv))
        condition_number = float(positive_sv.max() / positive_sv.min())

    if feature_group_resolver is None:
        feature_group_counts: list[dict[str, float | int | str]] = []
    else:
        groups = pd.Series([str(feature_group_resolver(col)) for col in cols], name="feature_group")
        feature_group_counts = (
            groups.value_counts(dropna=False)
            .rename_axis("feature_group")
            .reset_index(name="feature_count")
            .sort_values(["feature_count", "feature_group"], ascending=[False, True], kind="mergesort")
            .to_dict(orient="records")
        )

    low_coverage_features = (
        non_null_share[non_null_share < float(low_coverage_threshold)]
        .sort_values(kind="mergesort")
        .reset_index()
        .rename(columns={"index": "feature", 0: "non_null_share"})
        .to_dict(orient="records")
    )
    for row in low_coverage_features:
        row["feature"] = str(row["feature"])
        row["non_null_share"] = float(row["non_null_share"])

    return FeatureSanityDiagnostics(
        training_rows=int(len(numeric)),
        feature_count=int(numeric.shape[1]),
        min_non_null_share=float(np.nanmin(coverage_values)) if len(coverage_values) else float("nan"),
        p10_non_null_share=float(np.nanquantile(coverage_values, 0.10)) if len(coverage_values) else float("nan"),
        p50_non_null_share=float(np.nanquantile(coverage_values, 0.50)) if len(coverage_values) else float("nan"),
        p90_non_null_share=float(np.nanquantile(coverage_values, 0.90)) if len(coverage_values) else float("nan"),
        exact_or_near_complement_pairs=complement_pairs,
        strongest_positive_pairs=strongest_positive,
        strongest_negative_pairs=strongest_negative,
        high_abs_correlation_pairs=high_abs_pairs,
        effective_rank=effective_rank,
        condition_number=condition_number,
        feature_group_counts=feature_group_counts,
        low_coverage_features=low_coverage_features,
    )
