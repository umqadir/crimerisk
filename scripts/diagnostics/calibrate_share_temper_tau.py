"""Calibrate the covariate-trust exponent tau_o for the unified posterior-share allocator.

Context (docs/STATE.md, estimator CORRECTION block + GPT-confirmed decision sequence). The fix for the
count<->index incoherence is to form the within-jurisdiction share as a tempered log-lift of the
covariate residual model over the offense-specific exposure baseline, using the residual model's
predicted log-ratio eta directly (avoids epsilon issues vs (q/e)**tau):

    p_i = normalize_j( model_share_i * exp(tau_o * predicted_log_ratio_i) )

model_share = exposure baseline e_i (bg_weight already offense-specific). predicted_log_ratio = eta_i =
the covariate model's predicted log-lift over e_i (already clipped in apply_city_residual_model). At
tau=0 the share collapses to the pure exposure baseline (bland); at tau=1 it reproduces the current
residual model exactly. tau_o in [0,1] is HOW MUCH of the covariate lift to trust, and it must be EARNED
on held-out cities, not set from incident support (which would zero it for the ~90% of mass outside
onboarded cities and make the national map bland).

Calibration (GPT-confirmed): primary objective is held-out within-city TVD; selection uses the
conservative ONE-SE rule (smallest tau whose held-out TVD is within one city-bootstrap SE of the min),
biased toward the smaller/safer tau when the curve is flat. A log-log regression slope is reported only
as a red-team cross-check. Scalar tau_o per offense (no regime split in this pilot).

The expensive step is the leave-one-city-out residual refits; their per-BG predictions are cached to
parquet so tau selection / tail diagnostics can be recomputed cheaply with --from-cache.

Run:  uv run python scripts/diagnostics/calibrate_share_temper_tau.py
      uv run python scripts/diagnostics/calibrate_share_temper_tau.py --from-cache   # reselect tau only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_residuals import (
    CityResidualConfig,
    apply_city_residual_model,
    fit_city_residual_model,
    prepare_city_residual_frame,
)
from crimerisk.allocation import (
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
    PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES,
    promoted_residual_extra_bg_feature_paths,
)
from crimerisk.paths import RepoPaths

OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]
EPS = 1e-12


def _tvd(pred: np.ndarray, truth: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)).sum())


def _tempered_share(e: np.ndarray, eta: np.ndarray, tau: float) -> np.ndarray:
    """normalize_j( e * exp(tau*eta) ); tau=0 -> exposure baseline, tau=1 -> residual model."""
    e = np.clip(np.asarray(e, dtype=float), 0.0, None)
    w = e * np.exp(float(tau) * np.asarray(eta, dtype=float))
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        n = len(w)
        return np.full(n, 1.0 / n) if n else w
    return w / s


def _holdout_predictions(*, paths: RepoPaths, year: int) -> pd.DataFrame:
    """Leave-one-city-out: per held-out validation city, fit on the rest and predict on the held-out city.

    Matches production residual training: next-phase validation surface, promoted case-type exclusions,
    promoted extra features. Returns per-BG rows with true_share, model_share (e), predicted_log_ratio
    (eta), incident weights, and the primary denominator (for low-D tail diagnostics).
    """
    city_shares = pd.read_parquet(
        paths.state_dir / "modeling" / f"next_phase_validation_city_incident_share_surface_{year}.parquet"
    )
    if "validation_case_type" in city_shares.columns:
        city_shares = city_shares[
            ~city_shares["validation_case_type"].astype(str).isin(set(PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES))
        ].copy()
    bg_prior = pd.read_parquet(paths.state_dir / "modeling" / f"bg_prior_long_{year}.parquet")
    bg_crosswalk = pd.read_parquet(paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet")
    extra_feature_paths = list(promoted_residual_extra_bg_feature_paths(paths))

    frame, feature_cols = prepare_city_residual_frame(
        paths=paths,
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=year,
        extra_feature_paths=extra_feature_paths,
        feature_policy_path=REPO_ROOT / DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
        exclude_feature_policy_classes=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
        exclude_feature_policy_classes_by_offense=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    )
    cfg = CityResidualConfig(
        extra_feature_paths=tuple(extra_feature_paths),
        feature_policy_path=REPO_ROOT / DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
        exclude_feature_policy_classes=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
        exclude_feature_policy_classes_by_offense=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    )

    holdout_keys = (
        frame[["city_name", "jurisdiction_id"]]
        .dropna(subset=["jurisdiction_id"])
        .drop_duplicates()
        .sort_values(["city_name", "jurisdiction_id"], kind="mergesort")
    )
    keep = [
        "city_name",
        "jurisdiction_id",
        "offense",
        "true_share",
        "model_share",
        "predicted_log_ratio",
        "residual_model_share",
        "incident_count",
        "incident_total",
    ]
    rows: list[pd.DataFrame] = []
    for h in holdout_keys.itertuples(index=False):
        jid = str(h.jurisdiction_id)
        train = frame[frame["jurisdiction_id"].astype(str) != jid].copy()
        test = frame[frame["jurisdiction_id"].astype(str) == jid].copy()
        if train.empty or test.empty:
            continue
        fitted = fit_city_residual_model(train, feature_cols=feature_cols, config=cfg)
        if fitted is None:
            continue
        test = apply_city_residual_model(test, fitted=fitted)
        cols = [c for c in keep if c in test.columns]
        rows.append(test[cols].copy())
        print(f"  held out {h.city_name} ({jid}) — {len(test)} BG rows", flush=True)
    pred = pd.concat(rows, ignore_index=True)
    for c in ["true_share", "model_share", "predicted_log_ratio", "residual_model_share", "incident_count", "incident_total"]:
        if c in pred.columns:
            pred[c] = pd.to_numeric(pred[c], errors="coerce").fillna(0.0)
    return pred


def _per_city_offense_tvd(pred: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    """TVD of the tempered share vs truth for each (held-out city, offense) at each tau."""
    rows: list[dict[str, object]] = []
    for (jid, off), g in pred.groupby(["jurisdiction_id", "offense"], dropna=False):
        truth = g["true_share"].to_numpy(dtype=float)
        if truth.sum() <= 0:
            continue
        truth = truth / truth.sum()
        e = g["model_share"].to_numpy(dtype=float)
        eta = g["predicted_log_ratio"].to_numpy(dtype=float)
        weight = float(g["incident_total"].iloc[0]) if "incident_total" in g.columns and len(g) else float(len(g))
        for tau in taus:
            rows.append(
                {
                    "jurisdiction_id": str(jid),
                    "offense": str(off),
                    "tau": float(tau),
                    "tvd": _tvd(_tempered_share(e, eta, tau), truth),
                    "incident_total": weight,
                    "n_bg": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def _select_tau(go: pd.DataFrame, taus: np.ndarray, *, n_boot: int = 400, seed: int = 12345) -> dict[str, object]:
    """One-SE conservative tau selection on incident-weighted held-out TVD (restricted to tau in [0,1])."""
    cities = go["jurisdiction_id"].drop_duplicates().to_numpy()
    pivot = go.pivot_table(index="jurisdiction_id", columns="tau", values="tvd", aggfunc="mean")
    wmap = go.groupby("jurisdiction_id")["incident_total"].first()
    w = wmap.reindex(pivot.index).to_numpy(dtype=float)
    tau_cols = np.array([float(c) for c in pivot.columns])

    def _wmean(mat_rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
        ws = weights.sum()
        if ws <= 0:
            return mat_rows.mean(axis=0)
        return (mat_rows * weights[:, None]).sum(axis=0) / ws

    mat = pivot.to_numpy(dtype=float)
    curve = _wmean(mat, w)  # incident-weighted mean TVD per tau
    curve_eq = mat.mean(axis=0)  # jurisdiction-weighted (equal) mean TVD per tau

    # restrict selection to tau in [0,1]
    sel = tau_cols <= 1.0 + 1e-9
    tau01 = tau_cols[sel]
    curve01 = curve[sel]
    argmin_idx = int(np.argmin(curve01))
    tau_argmin = float(tau01[argmin_idx])
    tvd_min = float(curve01[argmin_idx])

    # city-bootstrap SE of the min-TVD value
    rng = np.random.default_rng(seed)
    n = len(cities)
    boot_mins = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        bw = w[idx]
        bcurve = _wmean(mat[idx], bw)[sel]
        boot_mins.append(float(np.min(bcurve)))
    se = float(np.std(boot_mins, ddof=1)) if len(boot_mins) > 1 else 0.0

    # one-SE rule: smallest tau whose TVD <= tvd_min + se
    within = np.where(curve01 <= tvd_min + se)[0]
    tau_one_se = float(tau01[int(within.min())]) if within.size else tau_argmin

    return {
        "tau_argmin": tau_argmin,
        "tau_one_se": tau_one_se,
        "tvd_at_tau_argmin": tvd_min,
        "tvd_at_0_exposure": float(curve[np.isclose(tau_cols, 0.0)][0]),
        "tvd_at_1_residual": float(curve[np.isclose(tau_cols, 1.0)][0]) if np.isclose(tau_cols, 1.0).any() else float("nan"),
        "tvd_at_tau_one_se": float(curve01[int(within.min())]) if within.size else tvd_min,
        "bootstrap_se": se,
        "n_holdout_cities": int(n),
        "tvd_curve_incident_weighted": {f"{t:.4f}": float(v) for t, v in zip(tau_cols, curve)},
        "tvd_curve_jurisdiction_weighted": {f"{t:.4f}": float(v) for t, v in zip(tau_cols, curve_eq)},
    }


def _regression_beta(pred: pd.DataFrame, off: str) -> float | None:
    """Red-team cross-check: slope of log(true/e) on eta, pooled over held-out BGs (city-demeaned eta)."""
    gp = pred[pred["offense"] == off].copy()
    m = (gp["model_share"] > EPS) & (gp["true_share"] > EPS)
    if int(m.sum()) <= 10:
        return None
    gp = gp.loc[m].copy()
    gp["_y"] = np.log((gp["true_share"] / gp["model_share"]).to_numpy(dtype=float))
    gp["_x"] = gp["predicted_log_ratio"].to_numpy(dtype=float)
    # city fixed effects: demean within jurisdiction (GPT: avoid missing city-level normalization constants)
    gp["_xd"] = gp["_x"] - gp.groupby("jurisdiction_id")["_x"].transform("mean")
    gp["_yd"] = gp["_y"] - gp.groupby("jurisdiction_id")["_y"].transform("mean")
    if float(np.std(gp["_xd"])) <= 0:
        return None
    return float(np.polyfit(gp["_xd"].to_numpy(dtype=float), gp["_yd"].to_numpy(dtype=float), 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--tau-max", type=float, default=1.5)
    ap.add_argument("--tau-steps", type=int, default=61)
    ap.add_argument("--from-cache", action="store_true", help="Reuse cached holdout predictions; reselect tau only.")
    ap.add_argument("--pred-cache", type=Path, default=REPO_ROOT / "state" / "modeling" / "share_temper_tau_holdout_predictions_2024.parquet")
    ap.add_argument("--out-json", type=Path, default=REPO_ROOT / "state" / "modeling" / "share_temper_tau_calibration_2024.json")
    args = ap.parse_args()
    year = int(args.year)
    paths = RepoPaths.from_repo_root(REPO_ROOT)

    if args.from_cache and args.pred_cache.exists():
        print(f"[tau-calibration] loading cached holdout predictions: {args.pred_cache}", flush=True)
        pred = pd.read_parquet(args.pred_cache)
    else:
        print(f"[tau-calibration] leave-one-city-out holdout predictions (year={year})...", flush=True)
        pred = _holdout_predictions(paths=paths, year=year)
        args.pred_cache.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(args.pred_cache, index=False)
        print(f"[tau-calibration] cached predictions -> {args.pred_cache}", flush=True)
    print(f"[tau-calibration] {len(pred)} held-out BG rows across {pred['jurisdiction_id'].nunique()} cities", flush=True)

    taus = np.round(np.linspace(0.0, float(args.tau_max), int(args.tau_steps)), 4)
    grid = _per_city_offense_tvd(pred, taus)

    summary: dict[str, dict[str, object]] = {}
    for off in OFFENSES:
        go = grid[grid["offense"] == off]
        if go.empty:
            continue
        sel = _select_tau(go, taus)
        sel["regression_beta_eta_cityFE"] = _regression_beta(pred, off)
        sel["gain_vs_residual"] = (sel["tvd_at_1_residual"] - sel["tvd_at_tau_one_se"]) if np.isfinite(sel["tvd_at_1_residual"]) else None
        sel["gain_vs_exposure"] = sel["tvd_at_0_exposure"] - sel["tvd_at_tau_one_se"]
        summary[off] = sel

    out = {"year": year, "taus": [float(t) for t in taus], "selection_rule": "one_SE_conservative_incident_weighted_tau_in_0_1", "per_offense": summary}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    print()
    hdr = f"{'offense':22s} {'tau1SE':>6s} {'tauMin':>6s} {'TVD@0':>7s} {'TVD@1':>7s} {'TVD@t*':>7s} {'gain/res':>9s} {'beta':>6s} {'SE':>7s} {'cities':>6s}"
    print(hdr)
    for off in OFFENSES:
        if off not in summary:
            continue
        s = summary[off]
        b = f"{s['regression_beta_eta_cityFE']:.2f}" if s["regression_beta_eta_cityFE"] is not None else "  na"
        gr = s["gain_vs_residual"]
        gr_s = f"{gr:+.4f}" if gr is not None else "   na"
        print(
            f"{off:22s} {s['tau_one_se']:6.3f} {s['tau_argmin']:6.3f} {s['tvd_at_0_exposure']:7.4f} "
            f"{s['tvd_at_1_residual']:7.4f} {s['tvd_at_tau_one_se']:7.4f} {gr_s:>9s} {b:>6s} {s['bootstrap_se']:7.4f} {s['n_holdout_cities']:6d}"
        )
    print("\ntau1SE = conservative one-SE held-out tau (the value to deploy). tauMin = raw argmin.")
    print("TVD@0 = pure exposure baseline; TVD@1 = current full-trust residual model; TVD@t* at tau1SE.")
    print("gain/res > 0 means tempering beats the current residual model on held-out within-city TVD.")
    print(f"\nSaved: {args.out_json}\n       {args.pred_cache} (holdout predictions cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
