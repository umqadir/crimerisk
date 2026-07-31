from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crime.constraints import STATE_ABBR_TO_FIPS, load_fbi_cde_state_constraints
from crimerisk.paths import RepoPaths
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
)


YEAR = 2024
OFFENSES_7 = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]
CONTINENTAL_SCOPE_EXCLUDE = {"02", "15", "72"}
FIPS_TO_STATE_ABBR = {v: k for k, v in STATE_ABBR_TO_FIPS.items()}


def _load_product_scope_fips(paths: RepoPaths) -> set[str]:
    controls_path = paths.state_dir / "controls" / "jurisdiction_controls_2024.parquet"
    if controls_path.exists():
        controls = pd.read_parquet(controls_path, columns=["state_fips"])
        scope = set(controls["state_fips"].astype(str).str.zfill(2).dropna().unique().tolist())
        if scope:
            return scope
    return {fips for fips in STATE_ABBR_TO_FIPS.values() if fips not in CONTINENTAL_SCOPE_EXCLUDE}


def _build_repo_source_totals(paths: RepoPaths, product_scope_fips: set[str]) -> pd.DataFrame:
    obs = pd.read_parquet(paths.state_dir / "observations" / "agency_year_observations.parquet")
    obs = obs[
        obs["year"].astype(int).eq(YEAR)
        & obs["source"].isin([CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE, NIBRS_SOURCE])
        & obs["offense"].isin(OFFENSES_7)
    ].copy()
    obs["state_fips"] = obs["state_fips"].astype(str).str.zfill(2)
    obs["state_abbr"] = obs["state_fips"].map(FIPS_TO_STATE_ABBR)
    obs = obs[obs["state_fips"].isin(product_scope_fips)].copy()
    grouped = (
        obs.groupby(["state_fips", "state_abbr", "offense", "source"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["state_fips", "state_abbr", "offense"],
        columns="source",
        values="count",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    pivot = pivot.rename(
        columns={
            CIUS_SOURCE: "repo_cius_observed_total_2024",
            LOCAL_PUBLICATION_SOURCE: "repo_local_publication_observed_total_2024",
            STATE_PUBLICATION_SOURCE: "repo_state_publication_observed_total_2024",
            SUMMARY_SOURCE: "repo_srs_observed_total_2024",
            NIBRS_SOURCE: "repo_nibrs_observed_total_2024",
        }
    )
    for col in [
        "repo_cius_observed_total_2024",
        "repo_local_publication_observed_total_2024",
        "repo_state_publication_observed_total_2024",
        "repo_srs_observed_total_2024",
        "repo_nibrs_observed_total_2024",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["year"] = YEAR
    return pivot


def _build_repo_preferred_totals(paths: RepoPaths, product_scope_fips: set[str]) -> pd.DataFrame:
    controls = pd.read_parquet(paths.state_dir / "controls" / "jurisdiction_controls_2024.parquet")
    controls["state_fips"] = controls["state_fips"].astype(str).str.zfill(2)
    controls["state_abbr"] = controls["state_abbr"].astype(str).str.upper()
    controls = controls[
        controls["state_fips"].isin(product_scope_fips)
        & controls["offense"].isin(OFFENSES_7)
    ].copy()
    grouped = (
        controls.groupby(["state_fips", "state_abbr", "offense"], dropna=False)[
            ["reported_count_preferred", "adjusted_count_ags_core"]
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                "reported_count_preferred": "repo_preferred_observed_total_2024",
                "adjusted_count_ags_core": "repo_ags_core_adjusted_total_2024",
            }
        )
    )
    grouped["year"] = YEAR
    return grouped


def _build_fbi_cde_long(paths: RepoPaths, product_scope_fips: set[str]) -> pd.DataFrame:
    constraints = load_fbi_cde_state_constraints(
        csv_path=paths.data_dir / "FBI-CDE-Estimates-1979-2024" / "estimated_crimes_1979_2024.csv",
        year_start=YEAR,
        year_end=YEAR,
    ).df.copy()
    constraints["state_fips"] = constraints["state_fips"].astype(str).str.zfill(2)
    constraints = constraints[constraints["state_fips"].isin(product_scope_fips)].copy()
    long = constraints.melt(
        id_vars=["year", "state_fips", "population"],
        value_vars=OFFENSES_7,
        var_name="offense",
        value_name="fbi_cde_estimated_total_2024",
    )
    long["state_abbr"] = long["state_fips"].map(FIPS_TO_STATE_ABBR)
    return long[["year", "state_fips", "state_abbr", "offense", "fbi_cde_estimated_total_2024", "population"]].copy()


def main() -> None:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    product_scope_fips = _load_product_scope_fips(paths)

    repo = _build_repo_source_totals(paths, product_scope_fips)
    preferred = _build_repo_preferred_totals(paths, product_scope_fips)
    fbi = _build_fbi_cde_long(paths, product_scope_fips)

    merged = (
        repo.merge(preferred, on=["year", "state_fips", "state_abbr", "offense"], how="outer")
        .merge(fbi, on=["year", "state_fips", "state_abbr", "offense"], how="outer")
        .sort_values(["state_fips", "offense"], kind="mergesort")
        .reset_index(drop=True)
    )
    for col in [
        "repo_local_publication_observed_total_2024",
        "repo_srs_observed_total_2024",
        "repo_nibrs_observed_total_2024",
        "repo_preferred_observed_total_2024",
        "repo_ags_core_adjusted_total_2024",
        "fbi_cde_estimated_total_2024",
    ]:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce")

    merged["repo_srs_minus_fbi_cde"] = merged["repo_srs_observed_total_2024"] - merged["fbi_cde_estimated_total_2024"]
    merged["repo_nibrs_minus_fbi_cde"] = (
        merged["repo_nibrs_observed_total_2024"] - merged["fbi_cde_estimated_total_2024"]
    )
    merged["repo_preferred_minus_fbi_cde"] = (
        merged["repo_preferred_observed_total_2024"] - merged["fbi_cde_estimated_total_2024"]
    )
    merged["repo_ags_core_adjusted_minus_fbi_cde"] = (
        merged["repo_ags_core_adjusted_total_2024"] - merged["fbi_cde_estimated_total_2024"]
    )
    merged["repo_preferred_gap_below_fbi_cde"] = (-merged["repo_preferred_minus_fbi_cde"].clip(upper=0)).fillna(0.0)
    merged["repo_ags_core_gap_below_fbi_cde"] = (
        -merged["repo_ags_core_adjusted_minus_fbi_cde"].clip(upper=0)
    ).fillna(0.0)
    merged["gap_priority_score"] = merged["repo_preferred_gap_below_fbi_cde"] + merged["repo_ags_core_gap_below_fbi_cde"]

    out_dir = paths.review_analysis_dir / "source_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "state_source_benchmark_2024.csv"
    parquet_path = out_dir / "state_source_benchmark_2024.parquet"
    priority_csv_path = out_dir / "state_source_priority_2024.csv"
    priority_parquet_path = out_dir / "state_source_priority_2024.parquet"
    merged.to_csv(csv_path, index=False)
    merged.to_parquet(parquet_path, index=False)
    priority = (
        merged.sort_values(
            ["gap_priority_score", "repo_preferred_gap_below_fbi_cde", "repo_ags_core_gap_below_fbi_cde"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )
    priority["priority_rank"] = range(1, len(priority) + 1)
    priority.to_csv(priority_csv_path, index=False)
    priority.to_parquet(priority_parquet_path, index=False)
    print(
        {
            "rows": int(len(merged)),
            "csv_path": str(csv_path),
            "parquet_path": str(parquet_path),
            "priority_csv_path": str(priority_csv_path),
            "priority_parquet_path": str(priority_parquet_path),
        }
    )


if __name__ == "__main__":
    main()
