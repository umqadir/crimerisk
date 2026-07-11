"""Build the O6 burglary uncovered-transfer tau calibration artifact.

The script consumes a prediction cache produced by nested_city_cv_harness.py.
It selects the cold-start outer prediction rows for burglary and applies the
one-SE rule on incident-weighted city-blocked TVD over the fixed grid
{0, 0.25, 0.5, 0.75, 1.0}. The gradient backstop is filled after the candidate
build, because selection itself must remain blind to the plausibility gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crime import OFFENSES_7


TAU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
TRAINABLE_ROLES = ("direct_posterior_live", "residual_training_only")
EVALUABLE_ROLES = (*TRAINABLE_ROLES, "validation_holdout_only")


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


def _join_key(parts: tuple[str, ...]) -> str:
    return "|".join(sorted(str(part) for part in parts if str(part)))


def _normalize(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(np.nansum(arr))
    if total > 0.0 and np.isfinite(total):
        return arr / total
    if len(arr) == 0:
        return arr
    return np.full(len(arr), 1.0 / float(len(arr)))


def _tempered_share(model_share: pd.Series, predicted_log_ratio: pd.Series, tau: float) -> np.ndarray:
    weights = np.clip(np.asarray(model_share, dtype=float), 0.0, None) * np.exp(
        np.clip(float(tau) * np.nan_to_num(np.asarray(predicted_log_ratio, dtype=float)), -50.0, 50.0)
    )
    return _normalize(weights)


def _outer_predictions(*, prediction_cache: pd.DataFrame, role_inventory: pd.DataFrame) -> pd.DataFrame:
    roles = role_inventory[
        role_inventory["offense"].astype(str).eq("burglary")
        & role_inventory["role"].astype(str).isin(EVALUABLE_ROLES)
    ].copy()
    trainable = set(
        roles.loc[roles["role"].astype(str).isin(TRAINABLE_ROLES), "city_key"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    rows: list[pd.DataFrame] = []
    cache = prediction_cache[prediction_cache["offense"].astype(str).eq("burglary")].copy()
    cache["predicted_city_key"] = cache["predicted_city_key"].astype(str)
    cache["excluded_train_city_keys"] = cache["excluded_train_city_keys"].fillna("").astype(str)
    for role in roles[["city_key", "city_name", "jurisdiction_id", "role"]].drop_duplicates().itertuples(index=False):
        city_key = str(role.city_key)
        excluded_key = city_key if city_key in trainable else ""
        one = cache[
            cache["predicted_city_key"].eq(city_key)
            & cache["excluded_train_city_keys"].eq(_join_key((excluded_key,)))
        ].copy()
        if not one.empty:
            one["outer_role"] = str(role.role)
            rows.append(one)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _per_city_curve(pred: pd.DataFrame, tau_grid: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city_key, group in pred.groupby("predicted_city_key", dropna=False, sort=True):
        truth = _normalize(group["true_share"])
        incident_total = float(pd.to_numeric(group["incident_count"], errors="coerce").fillna(0.0).sum())
        for tau in tau_grid:
            share = _tempered_share(group["model_share"], group["predicted_log_ratio"], float(tau))
            rows.append(
                {
                    "city_key": str(city_key),
                    "city_name": str(group["city_name"].iloc[0]),
                    "jurisdiction_id": str(group["jurisdiction_id"].iloc[0]),
                    "role": str(group["outer_role"].iloc[0]) if "outer_role" in group.columns else str(group["inventory_role"].iloc[0]),
                    "tau": float(tau),
                    "tvd": float(0.5 * np.abs(share - truth).sum()),
                    "incident_total": incident_total,
                    "bg_count": int(len(group)),
                    "incident_nonzero_bg_count": int(pd.to_numeric(group["incident_count"], errors="coerce").fillna(0.0).gt(0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _incident_weighted(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.nansum(weights))
    if denom > 0.0:
        return float(np.nansum(values * weights) / denom)
    return float(np.nanmean(values))


def _summarize_curve(*, city_curve: pd.DataFrame, tau_grid: list[float], n_boot: int, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    pivot = city_curve.pivot_table(index="city_key", columns="tau", values="tvd", aggfunc="first")
    weights = (
        city_curve.groupby("city_key", dropna=False)["incident_total"]
        .first()
        .reindex(pivot.index)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    mat = pivot[[float(tau) for tau in tau_grid]].to_numpy(dtype=float)
    means = np.array([_incident_weighted(mat[:, idx], weights) for idx in range(mat.shape[1])], dtype=float)

    rng = np.random.default_rng(int(seed))
    n_cities = int(len(pivot.index))
    boot = np.empty((0, len(tau_grid)), dtype=float)
    if n_cities > 1 and n_boot > 1:
        boot_rows = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n_cities, size=n_cities)
            boot_rows.append([_incident_weighted(mat[idx, col], weights[idx]) for col in range(mat.shape[1])])
        boot = np.asarray(boot_rows, dtype=float)
    se = boot.std(axis=0, ddof=1) if len(boot) > 1 else np.zeros(len(tau_grid), dtype=float)
    ci_low = np.percentile(boot, 2.5, axis=0) if len(boot) > 1 else np.full(len(tau_grid), np.nan)
    ci_high = np.percentile(boot, 97.5, axis=0) if len(boot) > 1 else np.full(len(tau_grid), np.nan)

    best_idx = int(np.nanargmin(means))
    best_tau = float(tau_grid[best_idx])
    best_mean = float(means[best_idx])
    threshold = best_mean + float(se[best_idx])
    within = np.where(means <= threshold + 1e-15)[0]
    one_se_idx = int(within.min()) if len(within) else best_idx
    one_se_tau = float(tau_grid[one_se_idx])

    curve = []
    for idx, tau in enumerate(tau_grid):
        curve.append(
            {
                "tau": float(tau),
                "incident_weighted_mean_tvd": float(means[idx]),
                "city_bootstrap_se": float(se[idx]),
                "city_bootstrap_ci95_low": float(ci_low[idx]) if np.isfinite(ci_low[idx]) else None,
                "city_bootstrap_ci95_high": float(ci_high[idx]) if np.isfinite(ci_high[idx]) else None,
                "within_one_se_of_best": bool(means[idx] <= threshold + 1e-15),
            }
        )

    return curve, {
        "argmin_tau": best_tau,
        "argmin_tvd": best_mean,
        "argmin_se": float(se[best_idx]),
        "one_se_tau": one_se_tau,
        "one_se_tvd": float(means[one_se_idx]),
        "one_se_threshold": float(threshold),
        "one_se_parsimony": "smallest_tau_more_shrinkage",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=REPO_ROOT / "state" / "cache" / "nested_city_cv_prediction_cache_burglary_tau_2024.parquet",
    )
    parser.add_argument(
        "--role-inventory-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "city_role_inventory_2024.parquet",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "burglary_tau_calibration_2024.json",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    pred = pd.read_parquet(args.prediction_cache)
    roles = pd.read_parquet(args.role_inventory_path)
    outer = _outer_predictions(prediction_cache=pred, role_inventory=roles)
    if outer.empty:
        raise SystemExit("No burglary cold-start outer prediction rows found in prediction cache.")
    for col in ("true_share", "model_share", "predicted_log_ratio", "incident_count"):
        outer[col] = pd.to_numeric(outer[col], errors="coerce").fillna(0.0)
    city_curve = _per_city_curve(outer, TAU_GRID)
    curve, selection = _summarize_curve(
        city_curve=city_curve,
        tau_grid=TAU_GRID,
        n_boot=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    selected_tau = float(selection["one_se_tau"])
    city_meta = (
        city_curve[["city_key", "city_name", "jurisdiction_id", "role", "incident_total", "bg_count"]]
        .drop_duplicates("city_key")
        .sort_values("city_key", kind="mergesort")
    )
    output = {
        "year": int(args.year),
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "offense": "burglary",
        "harness": "nested_city_cv_harness cold-start outer predictions",
        "prediction_cache_path": str(args.prediction_cache),
        "role_inventory_path": str(args.role_inventory_path),
        "tau_grid": TAU_GRID,
        "selection_metric": "burglary incident-weighted held-out TVD",
        "selection_rule": "one_SE_smallest_tau_within_best_plus_best_tau_city_bootstrap_SE",
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "selection_blind_to_gradient_gate": True,
        "argmin_tau": selection["argmin_tau"],
        "argmin_tvd": selection["argmin_tvd"],
        "argmin_se": selection["argmin_se"],
        "one_se_tau": selection["one_se_tau"],
        "one_se_tvd": selection["one_se_tvd"],
        "one_se_threshold": selection["one_se_threshold"],
        "production_tau": selected_tau,
        "selected_tau_after_backstop": selected_tau,
        "curve": curve,
        "city_count": int(city_meta["city_key"].nunique()),
        "incident_total": float(city_meta["incident_total"].sum()),
        "bg_rows": int(len(outer)),
        "city_rows": _clean_json(city_meta.to_dict(orient="records")),
        "per_city_curve": _clean_json(city_curve.to_dict(orient="records")),
        "gradient_backstop": {
            "evaluated": False,
            "applied": False,
            "threshold_modeled_transfer_gradient": 1.45,
            "modeled_transfer_gradient": None,
            "direct_city_gradient": None,
            "pooled_gradient": None,
            "action": "pending_candidate_build",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_clean_json(output), indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({k: output[k] for k in ("one_se_tau", "argmin_tau", "one_se_tvd", "argmin_tvd", "city_count", "incident_total")}, indent=2))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
