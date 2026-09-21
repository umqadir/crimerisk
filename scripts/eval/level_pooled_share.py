"""How much of the country a pooled silent-unit control speaks for.

`level_repair_mode = pooled_silent_unit` is the label the repair ladder writes when an
agency has no target-year row at all. The QA pass measured it on larceny: 34% of the
national count and 40% of the control population sat under such a control. This
reproduces that measurement from whatever ledger is on disk, so the same command run
before and after a data refresh gives both halves.

It also separates the label from what the ladder actually did. A jurisdiction carrying
the label is not necessarily sized from a peer pool: an agency with usable history
still gets its own decayed history, and only an agency with neither a current row nor
usable history falls through to a genuinely pooled number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import get_paths  # noqa: E402


POOLED_MODE = "pooled_silent_unit"


def measure(*, ledger: pd.DataFrame, controls: pd.DataFrame, offense: str) -> dict[str, object]:
    lane = ledger[ledger["offense"].astype("string").eq(offense)].copy()
    if lane.empty:
        return {"offense": offense, "rows": 0}
    lane["_pooled"] = lane["level_repair_mode"].astype("string").eq(POOLED_MODE) | lane[
        "ownership_class"
    ].astype("string").eq(POOLED_MODE)
    lane["_mass"] = pd.to_numeric(lane["final_control_mass"], errors="coerce").fillna(0.0)
    grouped = lane.groupby("jurisdiction_id", dropna=False).agg(
        pooled_mass=("_mass", lambda s: float(s[lane.loc[s.index, "_pooled"]].sum())),
        total_mass=("_mass", "sum"),
    ).reset_index()
    population = controls[controls["offense"].astype("string").eq(offense)][
        ["jurisdiction_id", "bucket_population"]
    ].drop_duplicates("jurisdiction_id")
    grouped = grouped.merge(population, on="jurisdiction_id", how="left")
    grouped["bucket_population"] = pd.to_numeric(
        grouped["bucket_population"], errors="coerce"
    ).fillna(0.0)
    share = grouped["pooled_mass"] / grouped["total_mass"].replace(0.0, np.nan)
    majority = grouped[share.fillna(0.0) > 0.5]
    total_population = float(grouped["bucket_population"].sum())
    return {
        "offense": offense,
        "jurisdictions": int(len(grouped)),
        "majority_pooled_jurisdictions": int(len(majority)),
        "majority_pooled_population": float(majority["bucket_population"].sum()),
        "control_population": total_population,
        "majority_pooled_population_share": (
            round(float(majority["bucket_population"].sum() / total_population), 4)
            if total_population
            else None
        ),
        "pooled_count_share": round(
            float(grouped["pooled_mass"].sum() / max(grouped["total_mass"].sum(), 1e-9)), 4
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Population and count share under a pooled silent-unit control."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = get_paths()
    ledger_path = args.ledger or (
        paths.state_dir / "controls" / f"level_lane_mass_ledger_{args.year}.parquet"
    )
    controls_path = args.controls or (
        paths.state_dir / "controls" / f"jurisdiction_controls_{args.year}.parquet"
    )
    ledger = pd.read_parquet(ledger_path)
    controls = pd.read_parquet(
        controls_path, columns=["jurisdiction_id", "offense", "bucket_population"]
    )
    offenses = sorted(set(ledger["offense"].astype(str)))
    report = {
        "year": int(args.year),
        "ledger": str(ledger_path),
        "by_offense": [measure(ledger=ledger, controls=controls, offense=o) for o in offenses],
        "agency_level_repair_modes": (
            ledger[ledger["ownership_class"].astype("string").eq("agency_level")]
            .groupby("level_repair_mode")["final_control_mass"]
            .agg(["size", "sum"])
            .round(1)
            .to_dict(orient="index")
        ),
    }
    text = json.dumps(report, indent=2, default=str)
    if args.out is not None:
        out = paths.repo_root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
