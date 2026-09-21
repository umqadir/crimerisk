#!/usr/bin/env python3
"""Measure a Census-residential substitution with fixed v52 counts; not a release builder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from crimerisk.exposure_ensemble import (
    ENSEMBLE_OFFENSES, combine_person_legs, compose_larceny_hybrid,
    larceny_hybrid_weights, load_ensemble_weights, person_leg_weights, rescale_to_total,
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    weights = load_ensemble_weights(args.weights)
    per_offense = person_leg_weights(weights)
    hybrid = larceny_hybrid_weights(weights)
    normalizers = pd.read_parquet(args.normalizers)
    pop_col = f"population_{args.year}"
    columns = ["block_group_geoid", "tract_id", pop_col, "special_use_type"]
    columns += [f"{prefix}_{o}" for o in ENSEMBLE_OFFENSES for prefix in
                ("expected_count", "primary_denominator", "primary_national_rate_per_100k")]
    bg = pd.read_parquet(args.baseline, columns=columns).rename(columns={"block_group_geoid": "bg_id"})
    frame = bg.merge(normalizers, on="bg_id", validate="one_to_one", suffixes=("", "_legs"))
    if len(frame) != len(bg) or len(frame) != len(normalizers):
        raise ValueError("Baseline and normalizer universes differ")
    if not np.allclose(frame[pop_col], frame.population, rtol=0, atol=1e-6):
        raise ValueError("Normalizer population differs from release population")
    total = frame[pop_col].sum()
    original = {key: frame[f"{key}_leg"].to_numpy() for key in
                ("landscan_night", "landscan_day", "daytime_jobs")}
    candidate = {**original, "landscan_night": frame[pop_col].to_numpy()}
    summaries, tails = [], []
    output = frame[["bg_id", "tract_id", pop_col, "special_use_type"]].copy()
    for o in ENSEMBLE_OFFENSES:
        arms = {}
        for name, legs in (("v52_recomposed", original), ("census_residential", candidate)):
            values = combine_person_legs(legs, per_offense[o])
            if o == "larceny":
                values, _ = compose_larceny_hybrid(person=values, parts={
                    "destination_poi": frame.destination_poi_total.to_numpy(),
                    "retail_jobs": frame.lodes_retail_jobs.to_numpy(),
                    "vehicles": np.zeros(len(frame)),
                }, weights=hybrid)
            arms[name], _ = rescale_to_total(values, target_total=total)
        if not np.allclose(arms["v52_recomposed"], frame[f"primary_denominator_{o}"], rtol=1e-10, atol=1e-8):
            raise ValueError(f"Cannot reproduce baseline normalizer: {o}")
        output[f"candidate_denominator_{o}"] = arms["census_residential"]
        for name, denominator in arms.items():
            work = frame[["bg_id", "tract_id", pop_col, f"expected_count_{o}"]].copy()
            work["denominator"] = denominator
            if o in ("murder", "rape"):
                work = work.groupby("tract_id", as_index=False)[[pop_col, f"expected_count_{o}", "denominator"]].sum()
            valid = (work.denominator >= 50) & ((work[pop_col] >= 50) | (work.denominator >= 500))
            national_rate = float(frame[f"primary_national_rate_per_100k_{o}"].iloc[0])
            work["index"] = (1e7 * work[f"expected_count_{o}"] / work.denominator / national_rate).where(valid)
            work["arm"], work["offense"] = name, o
            work["support"] = "tract" if o in ("murder", "rape") else "block_group"
            inhabited = work[pop_col] >= 500
            summaries.append({"arm": name, "offense": o, "support": work.support.iloc[0],
                "normalizer_total": float(denominator.sum()),
                "inhabited_denominator_below_tenth_residents": int((inhabited & (work.denominator < .1 * work[pop_col])).sum()),
                "denominator_ineligible_population": float(work.loc[~valid, pop_col].sum()),
                "index_p99": work["index"].quantile(.99), "index_p999": work["index"].quantile(.999),
                "index_max": work["index"].max(), "index_above_12800": int((work["index"] > 12800).sum())})
            tails.append(work.nlargest(20, "index"))
    args.out.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    pd.concat(tails, ignore_index=True).to_csv(args.out / "tails.csv", index=False)
    output.to_parquet(args.out / "diagnostic_denominators.parquet", index=False)
    manifest = {"year": args.year, "analysis": "fixed_counts_residential_component_substitution",
        "claim": "arithmetic sensitivity only; no predictive gain, recalibrated weights, or production candidate claimed",
        "eligibility_scope": "denominator_only",
        "eligibility_rule": "counts and support rules held fixed; eligibility changes only through the candidate denominator",
        "substitution": "replace LandScan nighttime leg with release Census resident population nationwide; retain all weights, other legs and national-total normalization",
        "inputs": {str(p.resolve()): digest(p) for p in (args.baseline, args.normalizers, args.weights)},
        "script_sha256": digest(Path(__file__))}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
