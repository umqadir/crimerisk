"""Build step-14 denominator-anchored BG prior candidate arms.

This script is candidate-only. It does not overwrite the canonical
state/modeling/bg_prior_long_2024.parquet unless explicitly pointed there.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crime import OFFENSES_7
from crimerisk.model_surface import ModelSurfaceConfig, build_model_surface, build_model_workload_plan
from crimerisk.paths import RepoPaths


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


def _prior_summary(prior: pd.DataFrame) -> dict[str, object]:
    weight = pd.to_numeric(prior["bg_weight"], errors="coerce").fillna(0.0)
    by_offense = (
        prior.assign(bg_weight=weight)
        .groupby("offense", dropna=False)["bg_weight"]
        .agg(["size", "sum", "min", "max"])
        .reset_index()
        .sort_values("offense", kind="mergesort")
    )
    return {
        "row_count": int(len(prior)),
        "bg_count": int(prior["bg_id"].astype("string").nunique()),
        "offenses": sorted(prior["offense"].astype(str).unique().tolist()),
        "bg_weight_sum": float(weight.sum()),
        "by_offense": _clean_json(by_offense.to_dict(orient="records")),
    }


def _feature_policy_summary(path: Path, excluded_classes: tuple[str, ...]) -> dict[str, object]:
    if not excluded_classes:
        return {"path": None, "excluded_classes": [], "excluded_feature_rows": 0}
    policy = pd.read_parquet(path, columns=["feature_column", "final_class", "feature_group"])
    classes = sorted({str(value) for value in excluded_classes})
    excluded = policy[policy["final_class"].astype("string").isin(classes)].copy()
    by_class = (
        excluded.groupby("final_class", dropna=False)
        .agg(feature_rows=("feature_column", "size"), groups=("feature_group", "nunique"))
        .reset_index()
        .sort_values("final_class", kind="mergesort")
    )
    return {
        "path": str(path),
        "excluded_classes": classes,
        "excluded_feature_rows": int(len(excluded)),
        "excluded_groups": sorted(excluded["feature_group"].dropna().astype(str).unique().tolist()),
        "by_class": _clean_json(by_class.to_dict(orient="records")),
    }


def _build_arm(
    *,
    paths: RepoPaths,
    label: str,
    config: ModelSurfaceConfig,
    prior_out: Path,
    meta_out: Path,
    summary_out: Path,
) -> dict[str, object]:
    print(f"[step14-priors] planning {label}", flush=True)
    plan = build_model_workload_plan(paths=paths, config=config)
    planned_fit_total = int(pd.to_numeric(plan["planned_fit_count"], errors="coerce").fillna(0).sum())
    print(f"[step14-priors] {label} planned_fit_total={planned_fit_total}", flush=True)

    started = time.perf_counter()

    def _progress(payload: dict[str, object]) -> None:
        if payload.get("event") != "offense_complete":
            return
        print(
            "[step14-priors] "
            f"{label} {payload.get('offense')} complete "
            f"elapsed={float(payload.get('elapsed_sec', 0.0)):.1f}s "
            f"weight_sum={float(payload.get('bg_weight_sum', 0.0)):.3f}",
            flush=True,
        )

    prior, diagnostics, meta = build_model_surface(
        paths=paths,
        config=config,
        progress_callback=_progress,
    )
    prior_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    prior.to_parquet(prior_out, index=False)
    meta.to_parquet(meta_out, index=False)
    if not diagnostics.empty:
        diagnostics_out = prior_out.with_name(prior_out.stem + "_diagnostics.parquet")
        diagnostics.to_parquet(diagnostics_out, index=False)
    else:
        diagnostics_out = None

    summary = {
        "label": label,
        "year": int(config.year),
        "created_at_utc": _now_iso(),
        "elapsed_sec": float(time.perf_counter() - started),
        "prior_anchor": str(config.prior_anchor),
        "feature_policy": _feature_policy_summary(
            Path(config.feature_policy_path),
            tuple(config.exclude_feature_policy_classes),
        ) if config.feature_policy_path is not None else {"path": None, "excluded_classes": []},
        "outputs": {
            "prior_path": str(prior_out),
            "feature_meta_path": str(meta_out),
            "diagnostics_path": str(diagnostics_out) if diagnostics_out is not None else None,
            "summary_path": str(summary_out),
        },
        "workload_plan": _clean_json(plan.to_dict(orient="records")),
        "prior": _prior_summary(prior),
        "feature_count": int(len(meta)),
        "retained_feature_groups": sorted(meta["feature_group"].dropna().astype(str).unique().tolist()),
    }
    summary_out.write_text(json.dumps(_clean_json(summary), indent=2, sort_keys=True, allow_nan=False))
    print(f"[step14-priors] wrote {label}: {prior_out}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--feature-policy-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "feature_transfer_policy_2024.parquet",
    )
    parser.add_argument(
        "--arm-a-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "bg_prior_long_2024_arm_a.parquet",
    )
    parser.add_argument(
        "--arm-b-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "bg_prior_long_2024_arm_b.parquet",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "step14_denominator_prior_candidates_2024.json",
    )
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    common = {
        "year": int(args.year),
        "compute_diagnostics": False,
        "prior_anchor": "offense_denominator",
    }
    arm_a_config = ModelSurfaceConfig(**common)
    arm_b_config = ModelSurfaceConfig(
        **common,
        feature_policy_path=Path(args.feature_policy_path),
        exclude_feature_policy_classes=("excluded_protected", "between_only"),
    )

    arm_a = _build_arm(
        paths=paths,
        label="arm_a_denominator_anchored_prior",
        config=arm_a_config,
        prior_out=Path(args.arm_a_out),
        meta_out=Path(args.arm_a_out).with_name(Path(args.arm_a_out).stem + "_features.parquet"),
        summary_out=Path(args.arm_a_out).with_suffix(".json"),
    )
    arm_b = _build_arm(
        paths=paths,
        label="arm_b_denominator_anchored_prior_feature_policy",
        config=arm_b_config,
        prior_out=Path(args.arm_b_out),
        meta_out=Path(args.arm_b_out).with_name(Path(args.arm_b_out).stem + "_features.parquet"),
        summary_out=Path(args.arm_b_out).with_suffix(".json"),
    )
    summary = {
        "year": int(args.year),
        "created_at_utc": _now_iso(),
        "candidate_arms": [arm_a, arm_b],
        "offenses": list(OFFENSES_7),
    }
    Path(args.summary_json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json_out).write_text(json.dumps(_clean_json(summary), indent=2, sort_keys=True, allow_nan=False))
    print(f"[step14-priors] wrote combined summary: {args.summary_json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

