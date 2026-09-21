"""Does the year after corroborate what an admission screen refused?

The holdout harness can say whether the repair ladder reconstructs a hidden row. It
cannot say whether refusing a *filed* row was right, because the filed row is the only
truth it has. The year after can.

If an agency's 2024 filing was a failure, its 2025 filing should come back up toward
the pre-2024 level. If the 2024 decline was real, 2025 should hold at or below it. So
for every agency-offense a screen refuses in the target year, this compares the filed
target-year count against the same agency's next-year count and its previous-year
count, and reports how the refused population splits.

The comparison uses data the screen cannot see: the screen runs on 2018-2024 and the
verdict comes from 2025. Nothing here is tuned; it is a precision check on a rule that
already exists.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.level_lane import apply_level_lane_admission  # noqa: E402
from crimerisk.paths import get_paths  # noqa: E402
from crimerisk.trend_fills import build_agency_trend_fill_panel  # noqa: E402


# A screened row needs a prior year big enough for a ratio to mean anything.
MIN_PRIOR_COUNT = 20.0
# "Came back up" and "stayed down" need a margin so ordinary noise is neither.
RECOVERY_RATIO = 1.25
PERSISTENCE_RATIO = 1.10


def next_year_counts(paths, *, year: int) -> pd.Series:
    observations = pd.read_parquet(
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        columns=["ori9", "year", "source", "offense", "count", "months_reported"],
    )
    following = observations[
        observations["year"].eq(int(year) + 1)
        & pd.to_numeric(observations["months_reported"], errors="coerce").ge(12.0)
    ]
    # Source selection's lane order, applied here so the corroborating number is the
    # one the lane would itself prefer.
    following = following.sort_values("source").drop_duplicates(
        ["ori9", "offense"], keep="first"
    )
    return following.set_index(["ori9", "offense"])["count"]


def screened_rows(paths, *, panel: pd.DataFrame, year: int, rules: str) -> pd.DataFrame:
    os.environ["CRIMERISK_LEVEL_RULES"] = rules
    disposition = apply_level_lane_admission(
        panel, paths=paths, target_year=int(year)
    ).disposition
    return disposition[
        disposition["year"].eq(int(year))
        & disposition["level_policy_resolution"].eq("repair_screened_vector")
    ].copy()


def evaluate(
    *, screened: pd.DataFrame, panel: pd.DataFrame, following: pd.Series, year: int
) -> dict[str, object]:
    history = panel[
        panel["usable_as_observed"].fillna(False).astype(bool)
        & pd.to_numeric(panel["preferred_months_reported"], errors="coerce").ge(12.0)
    ]
    prior = history[pd.to_numeric(history["year"], errors="coerce").eq(int(year) - 1)]
    prior_counts = prior.set_index(["ori9", "offense"])["preferred_count"]

    work = screened.copy()
    keys = pd.MultiIndex.from_arrays([work["ori9"], work["offense"]])
    work["prior_count"] = keys.map(prior_counts)
    work["next_count"] = keys.map(following)
    work["target_count"] = pd.to_numeric(
        work["level_lane_original_count"], errors="coerce"
    )
    testable = work.dropna(subset=["prior_count", "next_count", "target_count"])
    testable = testable[testable["prior_count"].ge(MIN_PRIOR_COUNT)]

    recovered = testable["next_count"].gt(testable["target_count"] * RECOVERY_RATIO)
    persisted = testable["next_count"].le(testable["target_count"] * PERSISTENCE_RATIO)
    dip = testable["target_count"].lt(
        0.8 * np.minimum(testable["prior_count"], testable["next_count"])
    )
    out: dict[str, object] = {
        "screened_rows": int(len(work)),
        "screened_agencies": int(work["ori9"].nunique()),
        "screened_count_mass": float(work["target_count"].sum()),
        "testable_rows": int(len(testable)),
        "target_year_is_a_dip_below_both_neighbours": int(dip.sum()),
        "next_year_recovers": int(recovered.sum()),
        "next_year_holds_at_or_below": int(persisted.sum()),
    }
    if len(testable):
        out["precision_next_year_recovers"] = round(float(recovered.mean()), 4)
        out["false_positive_next_year_holds"] = round(float(persisted.mean()), 4)
        out["reasons"] = (
            work["level_admission_reason"].astype("string").value_counts().to_dict()
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check refused rows against the following year's filing."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--panel-year-end", type=int, default=2025)
    parser.add_argument(
        "--panel-cache",
        type=Path,
        default=Path("state/eval/level_v1/cache/preferred_panel_2024.parquet"),
    )
    parser.add_argument(
        "--rules",
        nargs="*",
        default=[
            "own_history_admission_bound",
            "joint_vector_gate",
            "own_history_admission_bound,joint_vector_gate",
        ],
    )
    parser.add_argument("--out", type=Path, default=Path("state/eval/level_v1/next_year_corroboration.json"))
    args = parser.parse_args()

    paths = get_paths()
    cache = paths.repo_root / args.panel_cache
    if cache.exists():
        panel = pd.read_parquet(cache)
    else:
        panel = build_agency_trend_fill_panel(
            paths=paths, year_start=2018, year_end=int(args.panel_year_end)
        )
        panel = panel[pd.to_numeric(panel["year"], errors="coerce").le(int(args.year))]

    following = next_year_counts(paths, year=int(args.year))
    report = {
        "year": int(args.year),
        "corroborating_year": int(args.year) + 1,
        "min_prior_count": MIN_PRIOR_COUNT,
        "recovery_ratio": RECOVERY_RATIO,
        "persistence_ratio": PERSISTENCE_RATIO,
        "configurations": {},
    }
    for rules in args.rules:
        report["configurations"][rules] = evaluate(
            screened=screened_rows(paths, panel=panel, year=int(args.year), rules=rules),
            panel=panel,
            following=following,
            year=int(args.year),
        )
    text = json.dumps(report, indent=2, default=str)
    out = paths.repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
