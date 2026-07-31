from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.crime.srs import SRS_OFFENSE_COLUMN_MAP, ensure_srs_month_year_parquet, ensure_srs_year_parquet
from crimerisk.paths import RepoPaths
from crimerisk.source_selection import build_agency_preferred_observations


def _build_one_year_spikes(paths: RepoPaths, *, year_start: int, year_end: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(int(year_start), int(year_end) + 1):
        preferred = build_agency_preferred_observations(paths=paths, year=year)
        if preferred.empty:
            continue
        preferred = preferred.copy()
        preferred["year"] = int(year)
        frames.append(preferred)
    if not frames:
        return pd.DataFrame()

    obs = pd.concat(frames, ignore_index=True)
    obs = obs[obs["offense"].isin(OFFENSES_7)].copy()
    agency_master = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=["ori9", "agency_name_raw", "agency_name_std", "population_latest_nibrs"],
    ).drop_duplicates(subset=["ori9"])
    obs = obs.merge(agency_master, on="ori9", how="left")
    obs["preferred_count"] = pd.to_numeric(obs["preferred_count"], errors="coerce").fillna(0.0)
    obs["preferred_months_reported"] = pd.to_numeric(obs["preferred_months_reported"], errors="coerce").fillna(0.0)
    obs["population"] = pd.to_numeric(obs["population_latest_nibrs"], errors="coerce")
    obs = obs.sort_values(["ori9", "offense", "year"], kind="mergesort")
    grouped = obs.groupby(["ori9", "offense"], dropna=False)
    obs["prev_count"] = grouped["preferred_count"].shift(1)
    obs["next_count"] = grouped["preferred_count"].shift(-1)
    obs["prev_source"] = grouped["preferred_source"].shift(1)
    obs["next_source"] = grouped["preferred_source"].shift(-1)
    obs["prev_year"] = grouped["year"].shift(1)
    obs["next_year"] = grouped["year"].shift(-1)
    obs["jump_ratio_prev"] = np.where(obs["prev_count"] > 0, obs["preferred_count"] / obs["prev_count"], np.nan)
    obs["drop_ratio_next"] = np.where(obs["preferred_count"] > 0, obs["next_count"] / obs["preferred_count"], np.nan)
    obs["per_100k"] = np.where(obs["population"] > 0, 1e5 * obs["preferred_count"] / obs["population"], np.nan)
    obs["count_to_pop"] = np.where(obs["population"] > 0, obs["preferred_count"] / obs["population"], np.nan)

    spikes = obs[
        obs["preferred_source"].eq("srs_return_a_annual")
        & obs["prev_count"].gt(0)
        & obs["next_count"].notna()
        & obs["jump_ratio_prev"].ge(20.0)
        & obs["drop_ratio_next"].le(0.1)
    ].copy()
    if spikes.empty:
        return spikes

    cols = [
        "ori9",
        "agency_name_raw",
        "agency_name_std",
        "state_abbr",
        "offense",
        "year",
        "population",
        "preferred_source",
        "preferred_months_reported",
        "preferred_observation_weight",
        "prev_year",
        "prev_source",
        "prev_count",
        "preferred_count",
        "next_year",
        "next_source",
        "next_count",
        "jump_ratio_prev",
        "drop_ratio_next",
        "per_100k",
        "count_to_pop",
    ]
    return spikes[cols].sort_values(
        ["jump_ratio_prev", "preferred_count", "per_100k"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_srs_month_count_mismatches(paths: RepoPaths, *, year_start: int, year_end: int) -> pd.DataFrame:
    srs_zip = paths.data_dir / "SRS-Kaplan-1960-2024" / "offenses_known_parquet_1960_2024_year.zip"
    srs_month_zip = paths.data_dir / "SRS-Kaplan-1960-2024" / "offenses_known_parquet_1960_2024_month.zip"
    annual_path = ensure_srs_year_parquet(zip_path=srs_zip, cache_dir=paths.cache_dir).parquet_path

    annual = duckdb.sql(
        f"""
        SELECT
          COALESCE(NULLIF(ori9, ''), ori || '00') AS ori9,
          CAST(year AS INTEGER) AS year,
          agency_name AS agency_name_raw,
          UPPER(state_abb) AS state_abbr,
          CAST(population AS DOUBLE) AS population,
          CAST(number_of_months_reported AS INTEGER) AS annual_months_reported_raw,
          CAST(number_of_months_missing AS INTEGER) AS annual_months_missing_raw,
          ({' + '.join(f'COALESCE(CAST({col} AS DOUBLE), 0.0)' for col in SRS_OFFENSE_COLUMN_MAP.values())}) AS annual_part1_total
        FROM read_parquet('{annual_path.as_posix()}')
        WHERE CAST(year AS INTEGER) BETWEEN {int(year_start)} AND {int(year_end)}
          AND COALESCE(NULLIF(ori9, ''), ori || '00') IS NOT NULL
        """
    ).df()

    monthly_frames: list[pd.DataFrame] = []
    monthly_part1_expr = " + ".join(f"COALESCE(CAST({col} AS DOUBLE), 0.0)" for col in SRS_OFFENSE_COLUMN_MAP.values())
    for year in range(int(year_start), int(year_end) + 1):
        month_path = ensure_srs_month_year_parquet(
            zip_path=srs_month_zip,
            year=year,
            cache_dir=paths.cache_dir,
        ).parquet_path
        monthly = duckdb.sql(
            f"""
            WITH base AS (
              SELECT
                COALESCE(NULLIF(ori9, ''), ori || '00') AS ori9,
                CAST(year AS INTEGER) AS year,
                ({monthly_part1_expr}) AS part1_total,
                CAST(COALESCE(month_missing, 0) AS INTEGER) AS month_missing
              FROM read_parquet('{month_path.as_posix()}')
              WHERE CAST(year AS INTEGER) = {int(year)}
                AND COALESCE(NULLIF(ori9, ''), ori || '00') IS NOT NULL
            )
            SELECT
              ori9,
              year,
              COUNT(*) AS monthly_row_count,
              SUM(CASE WHEN month_missing = 0 THEN 1 ELSE 0 END) AS monthly_non_missing_month_count,
              SUM(part1_total) AS monthly_part1_total,
              MAX(part1_total) AS max_month_part1_total
            FROM base
            GROUP BY 1, 2
            """
        ).df()
        monthly_frames.append(monthly)
    monthly_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()

    merged = annual.merge(monthly_df, on=["ori9", "year"], how="left")
    merged["annual_months_reported_raw"] = pd.to_numeric(merged["annual_months_reported_raw"], errors="coerce")
    merged["monthly_non_missing_month_count"] = pd.to_numeric(merged["monthly_non_missing_month_count"], errors="coerce")
    merged["month_count_delta"] = merged["monthly_non_missing_month_count"] - merged["annual_months_reported_raw"]
    merged["monthly_part1_total"] = pd.to_numeric(merged["monthly_part1_total"], errors="coerce")
    merged["annual_part1_total"] = pd.to_numeric(merged["annual_part1_total"], errors="coerce")
    merged["part1_diff_ratio"] = np.where(
        merged["annual_part1_total"].fillna(0.0).gt(0.0),
        (merged["monthly_part1_total"] - merged["annual_part1_total"]).abs() / merged["annual_part1_total"].clip(lower=1.0),
        np.nan,
    )
    mismatches = merged[
        merged["annual_months_reported_raw"].notna()
        & merged["monthly_non_missing_month_count"].notna()
        & merged["month_count_delta"].ne(0)
        & (
            merged["annual_part1_total"].fillna(0.0).gt(0.0)
            | merged["monthly_part1_total"].fillna(0.0).gt(0.0)
        )
    ].copy()
    if mismatches.empty:
        return mismatches
    cols = [
        "ori9",
        "agency_name_raw",
        "state_abbr",
        "year",
        "population",
        "annual_months_reported_raw",
        "annual_months_missing_raw",
        "monthly_non_missing_month_count",
        "monthly_row_count",
        "month_count_delta",
        "annual_part1_total",
        "monthly_part1_total",
        "max_month_part1_total",
        "part1_diff_ratio",
    ]
    return mismatches[cols].sort_values(
        ["month_count_delta", "part1_diff_ratio", "annual_part1_total"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument(
        "--spikes-out",
        type=str,
        default="state/review/analysis/panel_qa/srs_one_year_spike_candidates_2018_2024.parquet",
    )
    parser.add_argument(
        "--month-mismatch-out",
        type=str,
        default="state/review/analysis/panel_qa/srs_month_count_mismatches_2018_2024.parquet",
    )
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(Path.cwd())
    spikes_out = Path(args.spikes_out)
    mismatch_out = Path(args.month_mismatch_out)
    spikes_out.parent.mkdir(parents=True, exist_ok=True)
    mismatch_out.parent.mkdir(parents=True, exist_ok=True)

    spikes = _build_one_year_spikes(paths, year_start=args.year_start, year_end=args.year_end)
    mismatches = _build_srs_month_count_mismatches(paths, year_start=args.year_start, year_end=args.year_end)
    spikes.to_parquet(spikes_out, index=False)
    mismatches.to_parquet(mismatch_out, index=False)
    print(f"spike_rows: {len(spikes)}")
    print(f"mismatch_rows: {len(mismatches)}")
    print(f"Wrote {spikes_out}")
    print(f"Wrote {mismatch_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
