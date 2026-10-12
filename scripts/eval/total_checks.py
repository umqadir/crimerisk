"""End-to-end total checks: jurisdiction controls against counts that do not come from the build.

    uv run python scripts/eval/total_checks.py --year 2025 [--out-dir state/eval/total_checks_2025]

The spatial evaluation normalises every city to shares, so it cannot see a control that is
several times too low. This report puts each jurisdiction's accounting and final control next to

* the city feeds' own annual count for the year, mapped to FBI categories
  (`city_incident_reconciliation_<year>`; feed and FBI counting units differ by roughly a tenth),
* the jurisdiction's own admitted reporting history: the median of its complete admitted years
  (the clean rows the smoother reads), and
* the smoothing step itself (final control over accounting),

and flags the rows outside fixed bands. Writes `total_check_flags.csv` and `summary.json`.
Flags are an investigation list, not established errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crimerisk.jurisdiction_targets import admitted_full_year_history  # noqa: E402

FEED_BAND = (0.8, 1.25)
HISTORY_BAND = (0.5, 2.0)
SMOOTHING_BAND = (0.7, 1.4)
MIN_POPULATION = 50_000.0
MIN_COUNT = 20.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    year = int(args.year)
    out_dir = args.out_dir or ROOT / "state" / "eval" / f"total_checks_{year}"
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    controls_dir = ROOT / "state" / "controls"

    smoothed = pd.read_parquet(controls_dir / f"jurisdiction_controls_smoothed_{year}.parquet")
    frame = smoothed[
        ["jurisdiction_id", "jurisdiction_name", "jurisdiction_type", "state_abbr", "offense", "bucket_population",
         "accounting_count", "smoothed_count", "estimator", "fallback_reason", "clean_year_count"]
    ].copy()

    panel = pd.read_parquet(controls_dir / "jurisdiction_year_estimates.parquet")
    history = panel[pd.to_numeric(panel["year"], errors="coerce").lt(year)].copy()
    clean, complete = admitted_full_year_history(history)
    history = history.assign(_complete=complete)[clean]
    median = history.groupby(["jurisdiction_id", "offense"]).agg(
        history_median=("_complete", "median"), history_years=("year", "nunique"), history_last_year=("year", "max")
    ).reset_index()
    frame = frame.merge(median, on=["jurisdiction_id", "offense"], how="left")

    feeds = pd.read_parquet(ROOT / "state" / "modeling" / f"city_incident_reconciliation_{year}.parquet")
    feeds = feeds[pd.to_numeric(feeds["year"], errors="coerce").eq(year)]
    feeds = feeds.groupby(["jurisdiction_id", "offense"], as_index=False).agg(
        city_key=("city_key", "first"), feed_mapped_count=("mapped_offense_count", "sum")
    )
    frame = frame.merge(feeds, on=["jurisdiction_id", "offense"], how="left")

    frame["final_over_accounting"] = frame["smoothed_count"] / frame["accounting_count"].where(frame["accounting_count"].gt(0))
    frame["final_over_history"] = frame["smoothed_count"] / frame["history_median"].where(frame["history_median"].gt(0))
    frame["final_over_feed"] = frame["smoothed_count"] / frame["feed_mapped_count"].where(frame["feed_mapped_count"].gt(0))
    frame["accounting_over_feed"] = frame["accounting_count"] / frame["feed_mapped_count"].where(frame["feed_mapped_count"].gt(0))

    def outside(series: pd.Series, band: tuple[float, float]) -> pd.Series:
        return series.notna() & (series.lt(band[0]) | series.gt(band[1]))

    sizable = frame["bucket_population"].ge(MIN_POPULATION) & frame["jurisdiction_type"].eq("municipal")
    countable = frame[["accounting_count", "history_median"]].max(axis=1).ge(MIN_COUNT)
    frame["flag_feed"] = outside(frame["final_over_feed"], FEED_BAND)
    frame["flag_history"] = sizable & countable & outside(frame["final_over_history"], HISTORY_BAND)
    frame["flag_smoothing"] = sizable & countable & outside(frame["final_over_accounting"], SMOOTHING_BAND)
    flags = frame[frame[["flag_feed", "flag_history", "flag_smoothing"]].any(axis=1)].sort_values(
        ["flag_feed", "bucket_population"], ascending=[False, False]
    )
    flags.to_csv(out_dir / "total_check_flags.csv", index=False)
    summary = {
        "year": year,
        "bands": {"feed": FEED_BAND, "history": HISTORY_BAND, "smoothing": SMOOTHING_BAND},
        "screen": {"municipal_min_population": MIN_POPULATION, "min_count": MIN_COUNT},
        "feed_cells": int(frame["feed_mapped_count"].notna().sum()),
        "feed_flags": int(frame["flag_feed"].sum()),
        "history_flags": int(frame["flag_history"].sum()),
        "smoothing_flags": int(frame["flag_smoothing"].sum()),
        "history_flag_jurisdictions": int(frame.loc[frame["flag_history"], "jurisdiction_id"].nunique()),
        "smoothing_flag_jurisdictions": int(frame.loc[frame["flag_smoothing"], "jurisdiction_id"].nunique()),
        "feed_ratio_median": float(np.nanmedian(frame["final_over_feed"])) if frame["final_over_feed"].notna().any() else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
