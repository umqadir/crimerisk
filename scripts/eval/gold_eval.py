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
    args = parser.parse_args(argv)
    config = GoldEvaluationConfig(
        run_id=str(args.run_id),
        folds=_folds(args.folds),
        bootstrap_iterations=int(args.bootstrap_iterations),
        dry_run=bool(args.dry_run),
        cities=_cities(args.cities),
    )
    evaluator = GoldEvaluation(paths=RepoPaths.from_repo_root(ROOT), config=config)
    evaluator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
