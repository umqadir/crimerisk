#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crimerisk.eval import GoldEvaluation, GoldEvaluationConfig
from crimerisk.paths import RepoPaths


def _use_recorded_temporal_blends(evaluator: GoldEvaluation, recorded_run: str) -> None:
    """Temporal folds take the blend table a completed run recorded for the same fold.

    The temporal refit reads the blend source run's retained fold inputs; when those are gone
    (gold_b0's were deleted at parking), the table another run computed from them is the same
    table. Accepted only if that run's manifest hashes the file, every row matches the current
    blend base (offense, status, source run, source fold type, fit cities) and each refit row
    names the same source run and cutoff.
    """
    import json

    import pandas as pd

    from crimerisk.eval.gold import _sha256

    original = evaluator.fold_blend_table
    run_dir = evaluator.paths.state_dir / "eval" / recorded_run
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    if "completed_at_utc" not in manifest:
        raise RuntimeError(f"Recorded blend run {recorded_run!r} is not complete")
    records = {str(f["fold_id"]): f for f in manifest["folds"]}
    key = ["offense", "status", "source_run_id", "source_fold_type", "fit_cities"]

    def table(spec):
        if not str(spec.fold_type).startswith("temporal"):
            return original(spec)
        path = run_dir / "blend" / f"{spec.fold_id}.csv"
        blend = (records.get(spec.fold_id) or {}).get("blend_weights") or {}
        if not path.is_file() or blend.get("sha256") != _sha256(path):
            raise RuntimeError(f"{recorded_run} has no hash-matching blend table for {spec.fold_id}")
        recorded = pd.read_csv(path)
        _base_path, base = evaluator.blend_base()
        if not recorded[key].astype(str).equals(base[key].astype(str)):
            raise RuntimeError(f"{recorded_run}'s {spec.fold_id} blend was made from a different base")
        for _, row in recorded.iterrows():
            source = str(row["fold_weight_source"])
            if source != "identity" and source != (
                f"refit_truth_through:{int(spec.feed_year_end)}:{row['source_run_id']}"
            ):
                raise RuntimeError(f"Unexpected blend source {source!r} in {path}")
        print(f"{spec.fold_id}: blend table from {path} (recorded by {recorded_run})", flush=True)
        recorded["fold_id"] = spec.fold_id
        return recorded

    evaluator.fold_blend_table = table


def _folds(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ("spatial", "temporal", "model")
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.replace("|", ",").split(",") if part.strip())
    invalid = sorted(set(parsed) - {"spatial", "temporal", "model"})
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown fold groups: {', '.join(invalid)}")
    return tuple(dict.fromkeys(parsed))


def _cities(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.replace("|", ",").split(",") if part.strip())
    return tuple(dict.fromkeys(parsed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CrimeRisk held-out gold evaluation.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--folds", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--cities",
        action="append",
        default=None,
        help=(
            "Restrict the leave-one-city-out spatial folds to these feed-city keys, "
            "comma-separated or repeated. Default: every admitted feed city."
        ),
    )
    parser.add_argument(
        "--variant",
        default="",
        help="Name of the one candidate change this run measures; appended to the run id.",
    )
    parser.add_argument(
        "--variant-arg",
        action="append",
        default=[],
        help=(
            "Extra build-outputs argument defining the variant (a flag, value or config path); "
            "repeat in order, e.g. --variant-arg=--mixture-weights-path --variant-arg=configs/x.csv."
        ),
    )
    parser.add_argument(
        "--score-candidate",
        default="",
        help="Score the model (E4) fold on this existing candidate directory instead of building it.",
    )
    parser.add_argument(
        "--baseline-run",
        default="",
        help="Completed run whose ours arms are paired against this run's (candidate - baseline).",
    )
    parser.add_argument(
        "--no-ags",
        action="store_true",
        help="Skip the AGS 2022A tract comparator (no AGS cache or download needed).",
    )
    parser.add_argument(
        "--e4-panel",
        default="configs/e4_panel_v1.json",
        help="Frozen E4 panel manifest the model fold scores.",
    )
    parser.add_argument(
        "--recorded-temporal-blend-run",
        default="",
        help="Take temporal folds' blend tables from this completed run's recorded tables.",
    )
    parser.add_argument(
        "--posthoc",
        default="",
        help="Score this post-hoc arm (blend_xfit) from the completed run's retained inputs.",
    )
    args = parser.parse_args(argv)
    run_id = str(args.run_id)
    if args.variant and not run_id.endswith(f"__{args.variant}"):
        run_id = f"{run_id}__{args.variant}"
    config = GoldEvaluationConfig(
        run_id=run_id,
        folds=_folds(args.folds),
        bootstrap_iterations=int(args.bootstrap_iterations),
        dry_run=bool(args.dry_run),
        cities=_cities(args.cities),
        variant=str(args.variant),
        variant_args=tuple(args.variant_arg),
        model_candidate_dir=str(args.score_candidate),
        baseline_run=str(args.baseline_run),
        ags_comparator=not bool(args.no_ags),
        e4_panel=str(args.e4_panel),
    )
    evaluator = GoldEvaluation(paths=RepoPaths.from_repo_root(ROOT), config=config)
    if args.recorded_temporal_blend_run:
        _use_recorded_temporal_blends(evaluator, str(args.recorded_temporal_blend_run))
    if args.posthoc:
        evaluator.run_posthoc(str(args.posthoc))
    else:
        evaluator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
