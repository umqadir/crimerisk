"""Fail closed if Step 8 posterior city shares collapse to the old direct override."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_YEAR = 2024
DEFAULT_MATERIAL_TVD_THRESHOLD = 0.01
DEFAULT_UNDER_COUNT_THRESHOLD = 0.10
DEFAULT_ALPHA_COLLAPSE_THRESHOLD = 1e-6


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


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


def _distribution(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {key: None for key in ["min", "p05", "p25", "median", "p75", "p95", "max", "mean"]}
    quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": _finite(values.min()),
        "p05": _finite(quantiles.loc[0.05]),
        "p25": _finite(quantiles.loc[0.25]),
        "median": _finite(quantiles.loc[0.5]),
        "p75": _finite(quantiles.loc[0.75]),
        "p95": _finite(quantiles.loc[0.95]),
        "max": _finite(values.max()),
        "mean": _finite(values.mean()),
    }


def _example(df: pd.DataFrame, mask: pd.Series) -> dict[str, object] | None:
    candidates = df.loc[mask].copy()
    if candidates.empty:
        return None
    candidates["_sort"] = pd.to_numeric(
        candidates.get("posterior_prior_fraction"),
        errors="coerce",
    ).fillna(0.0)
    row = candidates.sort_values("_sort", ascending=False, kind="mergesort").iloc[0]
    keys = [
        "city_name",
        "jurisdiction_id",
        "state_fips",
        "offense",
        "feed_quality_count",
        "control_total",
        "feed_control_fraction",
        "missing_fraction",
        "match_rate",
        "volume_prior_fraction",
        "alpha",
        "posterior_prior_fraction",
        "tvd_posterior_vs_direct",
        "posterior_mass_in_zero_feed_bgs",
        "zero_feed_prior_positive_bg_count",
    ]
    return {key: _clean_json(row.get(key)) for key in keys}


def _resolve_diagnostics_path(args: argparse.Namespace) -> Path:
    if args.diagnostics_path is not None:
        return Path(args.diagnostics_path)
    if args.candidate_dir is None:
        raise ValueError("pass either --diagnostics-path or --candidate-dir")
    return Path(args.candidate_dir) / f"city_posterior_diagnostics_{int(args.year)}.parquet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-path", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--material-tvd-threshold", type=float, default=DEFAULT_MATERIAL_TVD_THRESHOLD)
    parser.add_argument("--under-count-threshold", type=float, default=DEFAULT_UNDER_COUNT_THRESHOLD)
    parser.add_argument("--alpha-collapse-threshold", type=float, default=DEFAULT_ALPHA_COLLAPSE_THRESHOLD)
    args = parser.parse_args(argv)

    diagnostics_path = _resolve_diagnostics_path(args)
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"posterior diagnostics not found: {diagnostics_path}")

    df = pd.read_parquet(diagnostics_path)
    alpha = pd.to_numeric(df.get("alpha"), errors="coerce").fillna(0.0)
    tvd = pd.to_numeric(df.get("tvd_posterior_vs_direct"), errors="coerce").fillna(0.0)
    missing = pd.to_numeric(df.get("missing_fraction"), errors="coerce").fillna(0.0)
    volume_prior = pd.to_numeric(df.get("volume_prior_fraction"), errors="coerce").fillna(0.0)
    zero_feed_mass = pd.to_numeric(df.get("posterior_mass_in_zero_feed_bgs"), errors="coerce").fillna(0.0)
    material = tvd.gt(float(args.material_tvd_threshold))
    under_counting = missing.gt(float(args.under_count_threshold))
    sparse_or_low_volume = volume_prior.gt(0.25)
    clean_dense = (
        missing.le(float(args.under_count_threshold))
        & volume_prior.le(0.05)
        & pd.to_numeric(df.get("match_rate"), errors="coerce").fillna(1.0).ge(0.95)
    )

    checks = {
        "has_active_groups": int(len(df)) > 0,
        "alpha_not_collapsed": bool(alpha.max() > float(args.alpha_collapse_threshold)) if len(alpha) else False,
        "has_material_group_changes": bool(material.any()),
        "under_counting_changes_when_present": bool((under_counting & material).any()) if bool(under_counting.any()) else True,
        "zero_feed_bgs_receive_prior_mass": bool(zero_feed_mass.gt(0.0).any()),
    }
    summary = {
        "ok": bool(all(checks.values())),
        "diagnostics_path": str(diagnostics_path),
        "active_groups": int(len(df)),
        "material_tvd_threshold": float(args.material_tvd_threshold),
        "alpha_distribution": _distribution(alpha),
        "posterior_prior_fraction_distribution": _distribution(df.get("posterior_prior_fraction", pd.Series(dtype=float))),
        "groups_materially_changed_vs_direct": int(material.sum()),
        "under_counting_groups": int(under_counting.sum()),
        "under_counting_groups_materially_changed": int((under_counting & material).sum()),
        "sparse_or_low_volume_groups": int(sparse_or_low_volume.sum()),
        "sparse_or_low_volume_groups_materially_changed": int((sparse_or_low_volume & material).sum()),
        "groups_with_zero_feed_prior_mass": int(zero_feed_mass.gt(0.0).sum()),
        "clean_dense_groups": int(clean_dense.sum()),
        "clean_dense_tvd_distribution": _distribution(tvd.loc[clean_dense]),
        "example_under_counting_group": _example(df, under_counting),
        "checks": checks,
    }
    text = json.dumps(_clean_json(summary), indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
