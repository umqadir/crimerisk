"""First-look outlier screen — the anomalies a fresh viewer would hit while browsing.

Report-only (never a gate): ranks the shipped surface's most visually prominent
outliers so the supervisor can adjudicate each one as either genuinely reported
(stays dark, with the receipt) or a defect (blocks the release narrative until
fixed or explained). Motivated by the 2026-07-27 first-look failure: a fabricated
dark frontier county (Kenedy TX) was the first thing the user clicked.

Screens:
  A. Giant dark polygons — BGs with large land area and a high total index. These
     dominate low-zoom browsing far out of proportion to their population.
  B. Top BGs per offense index — the extreme tail a viewer reaches by clicking
     the darkest cells at high zoom.
  C. Implied-rate sanity — BGs whose expected counts imply absurd victimization
     ratios against their own stock (burglaries per household, vehicle thefts
     per vehicle).

Each row carries the routing facts needed to adjudicate (jurisdiction lane,
reliability, exposure) without re-deriving them.

Run:  uv run python scripts/diagnostics/first_look_outlier_screen.py
      [--surface state/output/crimerisk_block_group_2024_fbi_calibrated.parquet]
      [--out state/qa/first_look_outlier_screen.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

TOTAL_INDEX = "index_total_primary_event_weighted"

# Screen A: polygons big enough to dominate a state view, dark enough to draw a click.
GIANT_AREA_SQ_MI = 200.0
GIANT_INDEX_MIN = 300.0

# Screen B: per-offense tail depth.
TOP_N_PER_OFFENSE = 15

# Screen C: annual victimization ratios beyond which the number reads as absurd
# on its face (a third of all households burglarized, a fifth of vehicles stolen).
BURGLARY_PER_HOUSEHOLD_MAX = 0.30
MVT_PER_VEHICLE_MAX = 0.20

OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]

CONTEXT_COLS = [
    "block_group_geoid",
    "population_2024",
    "exposure_proxy_2024",
    "land_area_sq_mi",
    "urban_stratum",
    "eb_jurisdiction_id",
    "eb_jurisdiction_type",
    TOTAL_INDEX,
]


def screen_giant_dark_polygons(bg: pd.DataFrame) -> pd.DataFrame:
    hit = bg[
        (bg["land_area_sq_mi"] > GIANT_AREA_SQ_MI) & (bg[TOTAL_INDEX] > GIANT_INDEX_MIN)
    ].copy()
    hit["screen"] = "giant_dark_polygon"
    hit["screen_value"] = hit[TOTAL_INDEX]
    return hit.sort_values(TOTAL_INDEX, ascending=False)


def screen_top_offense_tails(bg: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for off in OFFENSES:
        col = f"index_{off}_primary"
        # Murder/rape publish at tract support: their BG index columns are all-null
        # by policy, and an all-null column has no tail to screen here.
        if col not in bg.columns or bg[col].notna().sum() == 0:
            continue
        top = bg.nlargest(TOP_N_PER_OFFENSE, col).copy()
        top["screen"] = f"top_index_{off}"
        top["screen_value"] = top[col]
        frames.append(top)
    return pd.concat(frames, ignore_index=False)


def screen_implied_ratios(bg: pd.DataFrame) -> pd.DataFrame:
    frames = []
    burg = bg[
        (bg["households_total"] >= 20)
        & (bg["expected_count_burglary"] / bg["households_total"] > BURGLARY_PER_HOUSEHOLD_MAX)
    ].copy()
    burg["screen"] = "burglaries_per_household"
    burg["screen_value"] = burg["expected_count_burglary"] / burg["households_total"]
    frames.append(burg)

    mvt = bg[
        (bg["vehicle_exposure_2024"] >= 20)
        & (
            bg["expected_count_motor_vehicle_theft"] / bg["vehicle_exposure_2024"]
            > MVT_PER_VEHICLE_MAX
        )
    ].copy()
    mvt["screen"] = "thefts_per_vehicle"
    mvt["screen_value"] = (
        mvt["expected_count_motor_vehicle_theft"] / mvt["vehicle_exposure_2024"]
    )
    frames.append(mvt)
    return pd.concat(frames, ignore_index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--surface",
        default=str(REPO / "state/output/crimerisk_block_group_2024_fbi_calibrated.parquet"),
    )
    ap.add_argument("--out", default=str(REPO / "state/qa/first_look_outlier_screen.csv"))
    args = ap.parse_args()

    cols = (
        CONTEXT_COLS
        + [f"index_{o}_primary" for o in OFFENSES]
        + [f"expected_count_{o}" for o in OFFENSES]
        + [f"reliability_tier_{o}" for o in OFFENSES]
        + ["households_total", "vehicle_exposure_2024"]
    )
    bg = pd.read_parquet(args.surface)
    bg = bg[[c for c in cols if c in bg.columns]]

    screens = pd.concat(
        [
            screen_giant_dark_polygons(bg),
            screen_top_offense_tails(bg),
            screen_implied_ratios(bg),
        ],
        ignore_index=True,
    )
    keep = ["screen", "screen_value"] + [c for c in bg.columns]
    screens = screens[keep].drop_duplicates(subset=["screen", "block_group_geoid"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    screens.to_csv(out, index=False)

    print(f"surface: {args.surface}")
    print(f"wrote {len(screens)} screen rows -> {out}")
    for name, grp in screens.groupby("screen"):
        print(f"  {name:32s} {len(grp):5d} rows   worst: {grp.screen_value.max():.1f}")
    lane = screens[screens.screen == "giant_dark_polygon"]["eb_jurisdiction_type"].value_counts()
    if len(lane):
        print("giant_dark_polygon by lane:", dict(lane))


if __name__ == "__main__":
    main()
