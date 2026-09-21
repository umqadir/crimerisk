"""Audit the smoothed overlap partition before and after the complete evidence spine fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.allocation import (
    _agency_estimates_cache_path,
    _build_overlap_group_targets,
    _load_bg_crosswalk,
    _load_controls,
    _load_crosswalk,
    _statewide_overlap_crosswalk_rows,
    build_agency_preferred_observations,
)
from crimerisk.paths import RepoPaths


def _comparison(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ["state_fips", "offense", "group_kind", "group_id"]
    out = old[keys + ["target_count"]].rename(columns={"target_count": "old_target_count"}).merge(
        new[keys + ["target_count"]].rename(columns={"target_count": "new_target_count"}),
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    for column in ["old_target_count", "new_target_count"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["delta"] = out["new_target_count"] - out["old_target_count"]
    out["absolute_delta"] = out["delta"].abs()
    out["new_to_old_ratio"] = np.where(
        out["old_target_count"].gt(0.0),
        out["new_target_count"] / out["old_target_count"],
        np.nan,
    )
    return out.sort_values(
        ["state_fips", "offense", "absolute_delta"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--bg-prior-path", type=Path, default=None)
    parser.add_argument("--preferred-path", type=Path, default=None)
    parser.add_argument("--risk-signals-path", type=Path, default=None)
    parser.add_argument("--details-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    paths = RepoPaths.from_repo_root(root)
    controls = _load_controls(paths, year=args.year, surface="smoothed")
    crosswalk = _load_crosswalk(paths)
    overlap_crosswalk = _statewide_overlap_crosswalk_rows(crosswalk)
    bg_crosswalk = _load_bg_crosswalk(paths)
    prior_path = args.bg_prior_path or paths.state_dir / "modeling" / f"bg_prior_long_{args.year}_arm_b.parquet"
    bg_prior = pd.read_parquet(prior_path, columns=["bg_id", "tract_id", "state_fips", "offense", "bg_weight"])
    estimate_path = _agency_estimates_cache_path(paths, args.year)
    agency_estimates = pd.read_parquet(estimate_path)
    preferred = (
        pd.read_parquet(args.preferred_path)
        if args.preferred_path is not None
        else build_agency_preferred_observations(paths=paths, year=args.year)
    )[["ori9", "state_fips", "offense", "preferred_count"]].copy()

    preferred_keys = pd.MultiIndex.from_frame(
        preferred[["ori9", "state_fips", "offense"]].assign(
            state_fips=lambda x: x["state_fips"].astype("string").str.zfill(2)
        )
    )
    risk_signals = (
        pd.read_parquet(args.risk_signals_path)
        if args.risk_signals_path is not None
        else None
    )
    estimate_keys = pd.MultiIndex.from_frame(
        agency_estimates[["ori9", "state_fips", "offense"]].assign(
            state_fips=lambda x: x["state_fips"].astype("string").str.zfill(2)
        )
    )
    legacy_estimates = agency_estimates.loc[estimate_keys.isin(preferred_keys)].copy()

    common = {
        "paths": paths,
        "controls": controls,
        "year": args.year,
        "enable_county_anchoring": True,
        "bg_prior": bg_prior,
        "bg_crosswalk": bg_crosswalk,
        "unlocated_mass": True,
    }
    old = _build_overlap_group_targets(
        agency_estimates=legacy_estimates,
        preferred_observations=preferred,
        agency_risk_signals=risk_signals,
        **common,
    )
    new = _build_overlap_group_targets(
        agency_estimates=agency_estimates,
        preferred_observations=preferred,
        agency_risk_signals=risk_signals,
        **common,
    )
    details = _comparison(old, new)

    overlap_keys = overlap_crosswalk[["ori", "state_fips"]].drop_duplicates().rename(columns={"ori": "ori9"})
    admitted = agency_estimates.merge(overlap_keys, on=["ori9", "state_fips"], how="inner")
    missing = admitted.loc[
        ~pd.MultiIndex.from_frame(admitted[["ori9", "state_fips", "offense"]]).isin(preferred_keys)
    ].copy()
    old_sums = old.groupby(["state_fips", "offense"])["target_count"].sum()
    new_sums = new.groupby(["state_fips", "offense"])["target_count"].sum()
    aligned = old_sums.to_frame("old").join(new_sums.rename("new"), how="outer").fillna(0.0)
    summary = {
        "year": args.year,
        "agency_estimates_path": str(estimate_path),
        "bg_prior_path": str(prior_path),
        "preferred_overlap_oris": int(
            preferred.merge(overlap_keys, on=["ori9", "state_fips"], how="inner")["ori9"].nunique()
        ),
        "admitted_overlap_oris": int(admitted["ori9"].nunique()),
        "missing_current_agency_offense_rows": int(len(missing)),
        "missing_current_oris": int(missing["ori9"].nunique()),
        "missing_current_states": int(missing["state_fips"].nunique()),
        "missing_current_estimated_count": float(missing["estimated_count"].sum()),
        "affected_state_offense_controls": int(
            details.loc[details["absolute_delta"].gt(1e-8), ["state_fips", "offense"]]
            .drop_duplicates()
            .shape[0]
        ),
        "changed_groups": int(details["absolute_delta"].gt(1e-8).sum()),
        "max_group_absolute_delta": float(details["absolute_delta"].max()),
        "old_new_parent_max_abs_delta": float((aligned["new"] - aligned["old"]).abs().max()),
        "old_parent_total": float(old["target_count"].sum()),
        "new_parent_total": float(new["target_count"].sum()),
    }
    if args.details_out is not None:
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(args.details_out, index=False)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
