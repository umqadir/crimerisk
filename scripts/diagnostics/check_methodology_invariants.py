"""Methodology regression gate for the CrimeRisk published index.

This is a RATCHET. It computes a small fixed set of methodology invariants from a
candidate block-group surface and compares them, per offense, against the best-known
values recorded in state/diagnostics/methodology_baseline.json. If any blocking
invariant REGRESSES past the baseline, the gate exits non-zero and promotion must stop.

Why this exists (see docs/METHODOLOGY-INVARIANTS.md): the share-EB reformulation
silently regressed the murder index from a clean max of ~2,413 to ~153,506 (a
tiny-denominator explosion) while that fact sat buried in a diagnostic table instead
of blocking the change. The lesson the owner drew is general: do not patch one noticed
symptom with one narrow fix and stack guardrails. A change that regresses a standing
invariant must FAIL loudly, and accepting the tradeoff must be a conscious, recorded act
(--update-baseline --rationale '...'), never a silent edit.

The gate deliberately uses a PER-OFFENSE ratchet rather than absolute thresholds, because
the shipped surface legitimately has very different maxima per offense (murder ~2,413 vs
larceny ~34,433 driven by real low-resident-premises hotspots). An absolute cap would be
exactly the kind of over-narrow guardrail this gate is meant to prevent.

Usage:
    # gate a candidate surface (default: the committed state/output BG core surface)
    python -m scripts.diagnostics.check_methodology_invariants \
        --surface state/tmp/<run>/crimerisk_block_group_2024_ags_core.parquet \
        --murder-gradient-chicago 21.4

    # consciously accept a tradeoff (re-seeds the baseline + appends rationale to history)
    python -m scripts.diagnostics.check_methodology_invariants \
        --surface <candidate> --update-baseline --rationale "why this regression is intended"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SURFACE = REPO_ROOT / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet"
DEFAULT_BASELINE = Path(__file__).resolve().parent / "methodology_baseline.json"

OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]

# Per-offense, per-surface metrics computed from the block-group parquet alone.
PER_OFFENSE_METRICS = [
    "max_index",
    "p999_index",
    "cells_gt_5000",
    "cells_gt_10000",
    "high_index_low_denominator_cells",
    "no_support_high_index_cells",
    "denom_weighted_mean_index",
    "published_fraction",
]
LOW_DENOMINATOR_TAIL_FLOOR = 10.0
HIGH_INDEX_TAIL_THRESHOLD = 5000.0


def _published_mask(df: pd.DataFrame, offense: str) -> pd.Series:
    col = f"index_publishable_{offense}"
    if col in df.columns:
        return df[col].astype("boolean").fillna(False).astype(bool)
    return pd.Series(True, index=df.index)


def compute_offense_metrics(df: pd.DataFrame, offense: str) -> dict[str, float]:
    idx = pd.to_numeric(df[f"index_{offense}"], errors="coerce")
    den_col = f"primary_denominator_{offense}"
    den = pd.to_numeric(df[den_col], errors="coerce") if den_col in df.columns else pd.Series(np.nan, index=df.index)
    pub = _published_mask(df, offense)

    idx_pub = idx[pub]
    valid = idx_pub.notna()
    idx_valid = idx_pub[valid]
    den_valid = den[pub][valid].clip(lower=0) if den.notna().any() else None

    if den_valid is not None and float(den_valid.sum()) > 0:
        wmean = float(np.average(idx_valid, weights=den_valid))
    else:
        wmean = float(idx_valid.mean()) if len(idx_valid) else float("nan")

    high_low_count = 0
    if den_valid is not None:
        high_low_count = int((idx_valid.gt(HIGH_INDEX_TAIL_THRESHOLD) & den_valid.le(LOW_DENOMINATOR_TAIL_FLOOR)).sum())

    # Tail sentinel: published cells with index>5000 AND the diagnostic EB low-denominator flag.
    # This stays aligned with the live output contract now that the published path is count-derived
    # and EB fields are diagnostics only.
    flag_col = f"diagnostic_eb_low_denominator_flag_{offense}"
    no_support_high = 0
    if flag_col in df.columns:
        flag_valid = pd.to_numeric(df[flag_col], errors="coerce").fillna(0.0)[pub][valid]
        no_support_high = int((idx_valid.gt(HIGH_INDEX_TAIL_THRESHOLD) & flag_valid.gt(0.0)).sum())

    return {
        "max_index": round(float(idx_valid.max()), 2) if len(idx_valid) else float("nan"),
        "cells_gt_5000": int((idx_valid > 5000).sum()),
        "cells_gt_10000": int((idx_valid > 10000).sum()),
        "high_index_low_denominator_cells": high_low_count,
        "no_support_high_index_cells": no_support_high,
        "denom_weighted_mean_index": round(wmean, 3),
        "published_fraction": round(float(pub.mean()), 5),
        "p999_index": round(float(idx_valid.quantile(0.999)), 2) if len(idx_valid) else float("nan"),
    }


def compute_surface_metrics(surface: Path) -> dict[str, Any]:
    df = pd.read_parquet(surface)
    return {off: compute_offense_metrics(df, off) for off in OFFENSES if f"index_{off}" in df.columns}


def _regressed(metric: str, baseline: float, candidate: float, rule: dict[str, Any]) -> tuple[bool, str]:
    """Return (is_regression, human-readable reason)."""
    if candidate is None or (isinstance(candidate, float) and np.isnan(candidate)):
        return True, "candidate value missing/NaN"
    direction = rule["direction"]
    tol_abs = float(rule.get("tol_abs", 0.0))
    if direction == "lower_is_better":
        tol_mult = float(rule.get("tol_mult", 1.0))
        ceiling = max(baseline * tol_mult, baseline + tol_abs)
        return (candidate > ceiling, f"<= {ceiling:.2f}")
    if direction == "higher_is_better":
        floor = baseline - tol_abs
        return (candidate < floor, f">= {floor:.4f}")
    if direction == "exact":
        target = float(rule.get("target", baseline))
        return (abs(candidate - target) > tol_abs, f"within +/-{tol_abs} of {target}")
    raise ValueError(f"unknown direction {direction!r}")


def evaluate(surface_metrics: dict[str, Any], baseline: dict[str, Any],
             murder_gradient_chicago: float | None) -> tuple[bool, list[dict[str, Any]]]:
    rules = baseline["rules"]
    rows: list[dict[str, Any]] = []
    ok = True

    for offense in OFFENSES:
        base = baseline["offenses"].get(offense)
        cand = surface_metrics.get(offense)
        if base is None or cand is None:
            continue
        for metric in PER_OFFENSE_METRICS:
            rule = rules.get(metric)
            if rule is None or not rule.get("blocking", False):
                continue
            if metric not in base or metric not in cand:
                continue  # metric absent from this baseline (e.g. pre-dates it); skip rather than crash
            b = float(base[metric])
            c = float(cand[metric])
            regressed, bound = _regressed(metric, b, c, rule)
            if regressed:
                ok = False
            rows.append({
                "scope": offense, "metric": metric, "baseline": b, "candidate": c,
                "bound": bound, "verdict": "REGRESSED" if regressed else "ok",
            })

    # Global: within-city murder gradient (owner priority) — only enforced if supplied.
    grad_rule = rules.get("murder_gradient_chicago_observed_positive")
    if grad_rule is not None:
        b = float(baseline["global"]["murder_gradient_chicago_observed_positive"])
        if murder_gradient_chicago is None:
            rows.append({
                "scope": "global", "metric": "murder_gradient_chicago_observed_positive",
                "baseline": b, "candidate": None, "bound": f">= {b - float(grad_rule['tol_abs'])}",
                "verdict": "NOT EVALUATED (supply --murder-gradient-chicago)",
            })
        else:
            regressed, bound = _regressed("murder_gradient_chicago_observed_positive", b,
                                          murder_gradient_chicago, grad_rule)
            if regressed and grad_rule.get("blocking", False):
                ok = False
            rows.append({
                "scope": "global", "metric": "murder_gradient_chicago_observed_positive",
                "baseline": b, "candidate": murder_gradient_chicago, "bound": bound,
                "verdict": "REGRESSED" if regressed else "ok",
            })
    return ok, rows


def print_table(rows: list[dict[str, Any]]) -> None:
    w = {"scope": 20, "metric": 34, "baseline": 14, "candidate": 14, "bound": 22, "verdict": 14}
    header = "".join(h.ljust(w[h]) for h in w)
    print(header)
    print("-" * len(header))
    for r in rows:
        cand = "n/a" if r["candidate"] is None else f"{r['candidate']:.4g}"
        line = (
            str(r["scope"]).ljust(w["scope"])
            + str(r["metric"]).ljust(w["metric"])
            + f"{r['baseline']:.4g}".ljust(w["baseline"])
            + cand.ljust(w["candidate"])
            + str(r["bound"]).ljust(w["bound"])
            + str(r["verdict"]).ljust(w["verdict"])
        )
        print(line)


def update_baseline(baseline_path: Path, baseline: dict[str, Any], surface_metrics: dict[str, Any],
                    surface: Path, murder_gradient_chicago: float | None, rationale: str) -> None:
    for offense, cand in surface_metrics.items():
        # Write every PER_OFFENSE_METRICS so newly-added metrics (e.g. no_support_high_index_cells)
        # flow through automatically rather than silently missing from the baseline.
        baseline["offenses"][offense] = {m: cand[m] for m in PER_OFFENSE_METRICS}
    if murder_gradient_chicago is not None:
        baseline["global"]["murder_gradient_chicago_observed_positive"] = murder_gradient_chicago
    baseline.setdefault("history", []).append({
        "date": "FILL-IN",  # stamp manually or via CI; Date.now is intentionally not assumed here
        "action": "update-baseline",
        "surface": str(surface.relative_to(REPO_ROOT)) if surface.is_relative_to(REPO_ROOT) else str(surface),
        "rationale": rationale,
    })
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", type=Path, default=DEFAULT_SURFACE,
                    help="block-group surface parquet to gate (default: committed state/output BG core)")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--murder-gradient-chicago", type=float, default=None,
                    help="Chicago observed-positive within-city murder tract gradient for the candidate (owner-priority invariant)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="consciously accept the candidate as the new ratchet (requires --rationale)")
    ap.add_argument("--rationale", type=str, default=None, help="why an intended regression is accepted")
    args = ap.parse_args(argv)

    if not args.surface.exists():
        print(f"ERROR: surface not found: {args.surface}", file=sys.stderr)
        return 2
    baseline = json.loads(args.baseline.read_text())
    surface_metrics = compute_surface_metrics(args.surface)

    print(f"Methodology regression gate")
    print(f"  surface : {args.surface}")
    print(f"  baseline: {args.baseline}\n")

    ok, rows = evaluate(surface_metrics, baseline, args.murder_gradient_chicago)
    print_table(rows)
    print()

    if args.update_baseline:
        if not args.rationale:
            print("ERROR: --update-baseline requires --rationale (accepting a tradeoff must be recorded).",
                  file=sys.stderr)
            return 2
        update_baseline(args.baseline, baseline, surface_metrics, args.surface,
                        args.murder_gradient_chicago, args.rationale)
        print(f"Baseline updated (rationale recorded). Set the 'date' field in the new history entry.")
        return 0

    regressions = [r for r in rows if r["verdict"] == "REGRESSED"]
    if regressions:
        print(f"GATE FAILED: {len(regressions)} invariant(s) regressed past the baseline.")
        print("Do NOT patch around this. Either fix the change so it clears the baseline, or — if the")
        print("tradeoff is genuinely intended — re-run with --update-baseline --rationale '<why>' AND")
        print("consult GPT Pro first on whether it signals a more fundamental issue (docs/METHODOLOGY-INVARIANTS.md).")
        return 1
    print("GATE PASSED: no methodology invariant regressed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
