"""Evaluation primitives for shipped CrimeRisk expected-count surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OFFENSES = (
    "murder", "rape", "robbery", "aggravated_assault", "burglary", "larceny",
    "motor_vehicle_theft",
)
TRACT_OFFENSES = frozenset(("murder", "rape"))
PREDICTION_EPSILON = 1e-12


@dataclass(frozen=True)
class SurfaceArm:
    name: str
    block_group_path: Path
    tract_path: Path
    manifest_path: Path
    audit_path: Path | None = None


def footprint_block_groups(frame: pd.DataFrame, jurisdiction_id: str) -> pd.Index:
    """Return the full explicit BG footprint, including cells with zero truth."""
    jid = str(jurisdiction_id)
    if ":county:" in jid:
        county = jid.rsplit(":", 1)[-1].zfill(5)
        mask = frame["block_group_geoid"].astype("string").str.startswith(county)
    else:
        mask = frame["eb_jurisdiction_id"].astype("string").eq(jid)
    return pd.Index(frame.loc[mask, "block_group_geoid"].astype("string").str.zfill(12).unique())


def normalize_scores(values: pd.Series, *, epsilon: float = PREDICTION_EPSILON) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw).all() or bool((raw < 0.0).any()):
        raise ValueError("Prediction scores must be finite and nonnegative")
    smoothed = raw + float(epsilon)
    total = float(smoothed.sum())
    if total <= 0.0:
        raise ValueError("Prediction scores have no mass")
    return smoothed / total


def score_distribution(
    truth_counts: pd.Series,
    prediction_scores: pd.Series,
    *,
    support_ids: pd.Series,
    epsilon: float = PREDICTION_EPSILON,
) -> dict[str, float | int | str]:
    truth = pd.to_numeric(truth_counts, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(truth).all() or bool((truth < 0.0).any()) or float(truth.sum()) <= 0.0:
        raise ValueError("Truth counts must be finite, nonnegative, and have positive mass")
    truth_share = truth / truth.sum()
    pred_share = normalize_scores(prediction_scores, epsilon=epsilon)
    n = len(truth)
    k = max(1, int(np.ceil(0.1 * n)))
    order = pd.DataFrame(
        {"score": pred_share, "support_id": support_ids.astype(str).to_numpy(), "truth": truth_share}
    ).sort_values(["score", "support_id"], ascending=[False, True], kind="mergesort")
    rank = (
        float(spearmanr(pred_share, truth_share).statistic)
        if n > 1 and np.ptp(pred_share) > 0.0 and np.ptp(truth_share) > 0.0
        else float("nan")
    )
    return {
        "support_cells": int(n),
        "incidents": float(truth.sum()),
        "tvd": float(0.5 * np.abs(truth_share - pred_share).sum()),
        "cross_entropy": float(-(truth_share * np.log(pred_share)).sum()),
        "top_decile_capture": float(order.iloc[:k]["truth"].sum()),
        "spearman": rank,
        "top_decile_k": int(k),
        "top_decile_tie_rule": "score_desc_support_id_asc_exact_k",
        "prediction_epsilon_before_z": float(epsilon),
    }


def aggregate_to_support(frame: pd.DataFrame, offense: str) -> pd.DataFrame:
    """Aggregate a fixed BG footprint to the product's supported geography."""
    out = frame.copy()
    if offense not in TRACT_OFFENSES:
        out["support_id"] = out["block_group_geoid"].astype("string").str.zfill(12)
        return out
    out["support_id"] = out["block_group_geoid"].astype("string").str.zfill(12).str[:11]
    numeric = [column for column in out.columns if column.startswith("score__")]
    numeric.extend(["truth_count", "population"])
    return out.groupby("support_id", as_index=False)[numeric].sum()


def fully_contained_tracts(surface: pd.DataFrame, footprint: pd.Index) -> pd.Index:
    """Return tracts whose complete national BG membership lies in ``footprint``."""
    work = surface[["block_group_geoid", "tract_id"]].copy()
    work["block_group_geoid"] = work["block_group_geoid"].astype("string").str.zfill(12)
    work["tract_id"] = work["tract_id"].astype("string").str.zfill(11)
    total = work.groupby("tract_id")["block_group_geoid"].nunique()
    inside = work[work["block_group_geoid"].isin(footprint)].groupby("tract_id")["block_group_geoid"].nunique()
    return pd.Index(inside[inside.eq(total.reindex(inside.index))].index)


def bootstrap_summary(
    cells: pd.DataFrame,
    *,
    metric: str,
    cluster: str,
    iterations: int = 1000,
    seed: int = 20240912,
) -> dict[str, float | int | str | None]:
    work = cells.dropna(subset=[metric, cluster]).copy()
    groups = list(work.groupby(cluster, sort=True))
    if not groups:
        return {
            "clusters": 0, "mean": None, "se": None, "ci95_low": None,
            "ci95_high": None, "uncertainty_reason": "no_evaluable_clusters",
        }
    point = float(work[metric].mean())
    if len(groups) < 2:
        return {
            "clusters": len(groups), "mean": point, "se": None, "ci95_low": None,
            "ci95_high": None, "uncertainty_reason": "insufficient_independent_clusters",
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[j][1] for j in chosen], ignore_index=True)
        draws[i] = float(sample[metric].mean())
    return {
        "clusters": len(groups), "mean": point, "se": float(draws.std(ddof=1)),
        "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
        "uncertainty_reason": None,
    }
