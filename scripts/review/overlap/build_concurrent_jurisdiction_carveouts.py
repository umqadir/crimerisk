"""Build configs/concurrent_jurisdiction_carveouts.csv from the Stage 2 displacement census.

Stage 2 fork ruling 1 (docs/STATE.md, "Stage 2 footprint batch IMPLEMENTED"): exclusive
displacement stays the default, but the counties where a REPORTING non-municipal remainder
agency loses more than half of its block-group exposure to displacing footprints revert to
shared-overlap treatment (no displacement) pending per-case PL-280 adjudication. Zeroing a
reporting sheriff's territory trades one distortion for another.

Selection, from analysis_scratch/stage2_fix_batch/displacement_county_concentration.csv:

    retained < 0.5                 -- displacement removes >50% of the county's remainder exposure
    remainder_agency_mass > 0      -- a remainder agency actually reports there

Counties failing only the second test (Todd SD, Oglala Lakota SD, Bennett SD, Buffalo SD,
Fremont WY) have no reporting remainder agency to distort and keep full displacement --
Todd SD is the batch's own worked example of displacement working.

Usage: uv run python scripts/review/overlap/build_concurrent_jurisdiction_carveouts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crimerisk.paths import get_paths

RETAINED_CEILING = 0.5
REVIEWER_NOTE = "concurrent_jurisdiction_unresolved"
SOURCE_ARTIFACT = "analysis_scratch/stage2_fix_batch/displacement_county_concentration.csv"

OUT_COLUMNS = [
    "state_fips",
    "county_fips",
    "county_geoid",
    "county_name",
    "reviewer_note",
    "remainder_exposure_before_displacement",
    "remainder_exposure_after_displacement",
    "remainder_exposure_retained_share",
    "reporting_remainder_agency_mass_2024",
    "source_artifact",
]


def build(repo_root: Path) -> pd.DataFrame:
    census = pd.read_csv(repo_root / SOURCE_ARTIFACT)
    census["county_geoid"] = census["county"].astype(str).str.zfill(5)
    census["retained"] = pd.to_numeric(census["retained"], errors="coerce")
    census["remainder_agency_mass"] = pd.to_numeric(
        census["remainder_agency_mass"], errors="coerce"
    ).fillna(0.0)
    selected = census[
        census["retained"].lt(RETAINED_CEILING) & census["remainder_agency_mass"].gt(0.0)
    ].copy()
    selected["state_fips"] = selected["county_geoid"].str.slice(0, 2)
    selected["county_fips"] = selected["county_geoid"].str.slice(2, 5)
    selected["reviewer_note"] = REVIEWER_NOTE
    selected["source_artifact"] = SOURCE_ARTIFACT
    out = selected.rename(
        columns={
            "before": "remainder_exposure_before_displacement",
            "after": "remainder_exposure_after_displacement",
            "retained": "remainder_exposure_retained_share",
            "remainder_agency_mass": "reporting_remainder_agency_mass_2024",
        }
    )[OUT_COLUMNS]
    return out.sort_values("remainder_exposure_retained_share", kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    paths = get_paths()
    out_path = args.out or (paths.repo_root / "configs" / "concurrent_jurisdiction_carveouts.csv")
    frame = build(paths.repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"wrote {len(frame)} carve-out counties -> {out_path}")
    print(frame[["county_geoid", "county_name", "remainder_exposure_retained_share"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
