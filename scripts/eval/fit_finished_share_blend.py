#!/usr/bin/env python
"""Fit the production finished-share blend weights (2025.2 candidate M1).

Reads a completed gold run's retained spatial (held-out city) primary inputs, fits the 0.1
simplex over (model, population, primary exposure) by mean TVD over every held-out city, and
writes the versioned config the build reads. Only offenses the pre-registered rule adopted
keep their fitted weights; every other offense is written as (1, 0, 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crimerisk.eval.gold import OFFENSES, _sha256, pooled_blend_weights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="gold_b0")
    parser.add_argument("--adopt", action="append", required=True, help="Offense the rule adopted.")
    parser.add_argument("--out", default=str(ROOT / "configs" / "finished_share_blend_v1.csv"))
    args = parser.parse_args(argv)
    run_dir = ROOT / "state" / "eval" / args.run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    if "completed_at_utc" not in manifest:
        raise RuntimeError(f"Run {args.run_id!r} is not complete")
    inputs = []
    for record in manifest["folds"]:
        if record["fold_type"] != "spatial":
            continue
        path = Path(record["input_path"])
        if _sha256(path) != record["input_sha256"]:
            raise RuntimeError(f"Retained input {path} changed since the run")
        inputs.append(pd.read_parquet(path))
    retained = pd.concat(inputs, ignore_index=True)
    retained = retained[retained["support_role"].eq("primary")]
    fitted = pooled_blend_weights(retained).set_index("offense")
    adopted = set(args.adopt)
    unknown = sorted(adopted - set(fitted.index))
    if unknown:
        raise ValueError(f"No fitted weights for {unknown}")
    rows = []
    for offense in OFFENSES:
        adopt = offense in adopted
        fit = fitted.loc[offense] if adopt else None
        rows.append(
            {
                "offense": offense,
                "weight_model": float(fit["weight_ours"]) if adopt else 1.0,
                "weight_population": float(fit["weight_population"]) if adopt else 0.0,
                "weight_primary_exposure": float(fit["weight_primary_exposure"]) if adopt else 0.0,
                "status": "adopted" if adopt else "not_adopted",
                "source_run_id": args.run_id,
                "source_fold_type": "spatial",
                "fit_cities": fit["fit_cities"] if fit is not None else "",
                "fit_mean_tvd": float(fit["fit_mean_tvd"]) if fit is not None else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
