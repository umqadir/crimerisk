"""Run resumable step-14 held-out TVD validation checkpoints.

Each config writes a distinct checkpoint JSON only after the nested-CV harness
has produced a complete h100 summary and diagnostics parquet. On restart, valid
checkpoints are skipped so an interruption costs at most the active config pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_AGGREGATES = (
    "overall_by_variant",
    "by_offense",
    "by_inventory_role",
    "by_sparsity_class",
)
REQUIRED_VARIANT = "nested_selected_policy"
HIST_MAX_ITER = 100


CONFIGS = {
    "baseline": {
        "label": "baseline",
        "bg_prior_path": REPO_ROOT / "state" / "modeling" / "bg_prior_long_2024.parquet",
        "checkpoint_path": REPO_ROOT / "state" / "modeling" / "step14_heldout_tvd_baseline.json",
        "summary_path": REPO_ROOT / "state" / "modeling" / "nested_city_cv_step14_baseline_h100.json",
        "diagnostics_path": REPO_ROOT / "state" / "modeling" / "nested_city_cv_step14_baseline_h100.parquet",
        "prediction_cache_path": REPO_ROOT / "state" / "cache" / "nested_city_cv_prediction_cache_step14_baseline_h100.parquet",
        "fallback_summary_paths": (
            REPO_ROOT / "state" / "modeling" / "nested_city_cv_2024.json",
        ),
    },
    "arm_a": {
        "label": "arm_a",
        "bg_prior_path": REPO_ROOT / "state" / "modeling" / "bg_prior_long_2024_arm_a.parquet",
        "checkpoint_path": REPO_ROOT / "state" / "modeling" / "step14_heldout_tvd_arm_a.json",
        "summary_path": REPO_ROOT / "state" / "modeling" / "nested_city_cv_step14_arm_a_h100.json",
        "diagnostics_path": REPO_ROOT / "state" / "modeling" / "nested_city_cv_step14_arm_a_h100.parquet",
        "prediction_cache_path": REPO_ROOT / "state" / "cache" / "nested_city_cv_prediction_cache_step14_arm_a_h100.parquet",
        "fallback_summary_paths": (),
    },
    "arm_b": {
        "label": "arm_b",
        "bg_prior_path": REPO_ROOT / "state" / "modeling" / "bg_prior_long_2024_arm_b.parquet",
        "checkpoint_path": REPO_ROOT / "state" / "modeling" / "step14_heldout_tvd_arm_b.json",
        "summary_path": REPO_ROOT / "state" / "modeling" / "nested_city_cv_step14_arm_b_h100.json",
        "diagnostics_path": REPO_ROOT / "state" / "modeling" / "nested_city_cv_step14_arm_b_h100.parquet",
        "prediction_cache_path": REPO_ROOT / "state" / "cache" / "nested_city_cv_prediction_cache_step14_arm_b_h100.parquet",
        "fallback_summary_paths": (),
    },
}


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
    if value is not None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _same_path(actual: str | None, expected: Path) -> bool:
    if not actual:
        return False
    try:
        return Path(actual).expanduser().resolve() == expected.expanduser().resolve()
    except OSError:
        return False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_summary(path: Path, *, bg_prior_path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing json"
    try:
        data = _read_json(path)
    except Exception as exc:  # noqa: BLE001 - validation should report parse failures.
        return False, f"invalid json: {exc}"

    hist_iter = data.get("residual_fit_config", {}).get("hist_max_iter")
    if int(hist_iter or -1) != HIST_MAX_ITER:
        return False, f"hist_max_iter={hist_iter!r}, expected {HIST_MAX_ITER}"
    prior = data.get("upstream_baseline_scope", {}).get("bg_prior_long_2024")
    if not _same_path(prior, bg_prior_path):
        return False, f"bg prior mismatch: {prior!r}"

    row_counts = data.get("row_counts", {})
    if int(row_counts.get("missing_evaluable_city_offense_cases", -1)) != 0:
        return False, "missing evaluable city-offense cases"
    if int(row_counts.get("diagnostic_rows", 0)) <= 0:
        return False, "no diagnostic rows"
    if REQUIRED_VARIANT not in set(row_counts.get("model_variants", [])):
        return False, f"missing {REQUIRED_VARIANT} variant"

    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict):
        return False, "missing aggregate"
    for key in REQUIRED_AGGREGATES:
        value = aggregate.get(key)
        if not isinstance(value, list) or not value:
            return False, f"missing aggregate.{key}"
        if not any(row.get("model_variant") == REQUIRED_VARIANT for row in value if isinstance(row, dict)):
            return False, f"aggregate.{key} lacks {REQUIRED_VARIANT}"

    diagnostics_path = data.get("diagnostics_path")
    if not diagnostics_path:
        return False, "missing diagnostics_path"
    diagnostics = Path(diagnostics_path)
    if not diagnostics.is_absolute():
        diagnostics = REPO_ROOT / diagnostics
    if not diagnostics.exists():
        return False, f"missing diagnostics parquet: {diagnostics}"
    try:
        metadata = pq.ParquetFile(diagnostics).metadata
    except Exception as exc:  # noqa: BLE001 - validation should report parquet failures.
        return False, f"invalid diagnostics parquet: {exc}"
    if int(metadata.num_rows) != int(row_counts.get("diagnostic_rows", -1)):
        return False, "diagnostics row count mismatch"
    return True, "ok"


def _validate_checkpoint(path: Path, *, bg_prior_path: Path) -> tuple[bool, str]:
    ok, reason = _validate_summary(path, bg_prior_path=bg_prior_path)
    if not ok:
        return ok, reason
    try:
        data = _read_json(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"invalid checkpoint json: {exc}"
    meta = data.get("step14_validation_checkpoint", {})
    if not isinstance(meta, dict):
        return False, "missing step14 checkpoint metadata"
    if meta.get("schema") != "step14_heldout_tvd_checkpoint_v1":
        return False, "unexpected checkpoint schema"
    return True, "ok"


def _valid_prediction_cache(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return int(pq.ParquetFile(path).metadata.num_rows) > 0
    except Exception:
        return False


def _write_checkpoint(*, summary_path: Path, checkpoint_path: Path, label: str, source: str) -> None:
    data = _read_json(summary_path)
    data["step14_validation_checkpoint"] = {
        "schema": "step14_heldout_tvd_checkpoint_v1",
        "label": label,
        "source": source,
        "checkpoint_path": str(checkpoint_path),
        "source_summary_json_path": str(summary_path),
        "written_at_utc": _now_iso(),
        "required_aggregate_breakdowns": list(REQUIRED_AGGREGATES),
        "recommendation_metric_variant": REQUIRED_VARIANT,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_clean_json(data), indent=2, sort_keys=True, allow_nan=False))
    tmp_path.replace(checkpoint_path)


def _run_harness(config: dict[str, Any], *, jobs: int) -> None:
    summary_path = Path(config["summary_path"])
    diagnostics_path = Path(config["diagnostics_path"])
    prediction_cache_path = Path(config["prediction_cache_path"])
    bg_prior_path = Path(config["bg_prior_path"])

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "diagnostics" / "nested_city_cv_harness.py"),
        "--hist-max-iter",
        str(HIST_MAX_ITER),
        "--jobs",
        str(jobs),
        "--prediction-cache",
        str(prediction_cache_path),
        "--diagnostics-out",
        str(diagnostics_path),
        "--summary-json-out",
        str(summary_path),
        "--bg-prior-path",
        str(bg_prior_path),
    ]
    if _valid_prediction_cache(prediction_cache_path):
        cmd.append("--reuse-prediction-cache")

    print(f"[step14-validation] running {config['label']}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_config(name: str, *, jobs: int) -> str:
    config = CONFIGS[name]
    bg_prior_path = Path(config["bg_prior_path"])
    checkpoint_path = Path(config["checkpoint_path"])
    summary_path = Path(config["summary_path"])

    ok, reason = _validate_checkpoint(checkpoint_path, bg_prior_path=bg_prior_path)
    if ok:
        print(f"[step14-validation] skip valid checkpoint: {checkpoint_path}", flush=True)
        return "skipped_checkpoint"
    print(f"[step14-validation] checkpoint not reusable for {name}: {reason}", flush=True)

    candidate_summaries = [summary_path, *config.get("fallback_summary_paths", ())]
    for candidate in candidate_summaries:
        summary_ok, summary_reason = _validate_summary(Path(candidate), bg_prior_path=bg_prior_path)
        if summary_ok:
            _write_checkpoint(
                summary_path=Path(candidate),
                checkpoint_path=checkpoint_path,
                label=str(config["label"]),
                source="existing_valid_h100_summary",
            )
            print(f"[step14-validation] checkpointed {name} from existing summary: {candidate}", flush=True)
            return "checkpointed_existing_summary"
        print(f"[step14-validation] summary not reusable for {name}: {candidate} ({summary_reason})", flush=True)

    _run_harness(config, jobs=jobs)
    summary_ok, summary_reason = _validate_summary(summary_path, bg_prior_path=bg_prior_path)
    if not summary_ok:
        raise RuntimeError(f"Completed harness output for {name} is invalid: {summary_reason}")
    _write_checkpoint(
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        label=str(config["label"]),
        source="fresh_harness_run",
    )
    print(f"[step14-validation] checkpointed {name} after fresh harness run: {checkpoint_path}", flush=True)
    return "fresh_harness_run"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIGS),
        default=tuple(CONFIGS),
    )
    args = parser.parse_args()

    if shutil.which("python") is None:
        raise SystemExit("python executable is unavailable")

    results = {}
    for name in args.configs:
        results[name] = run_config(name, jobs=max(1, int(args.jobs)))
    print(json.dumps(_clean_json({"results": results}), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
