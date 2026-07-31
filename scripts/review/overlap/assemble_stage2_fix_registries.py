"""Deterministically assemble the Stage 2 fix batch's registry files from base + generated parts.

Idempotent: always starts from the committed base snapshot in
`analysis_scratch/stage2_fix_batch/base/` and re-applies every addition, so re-running never
double-appends and the configs are reproducible from the evidence tables plus the generators.

Parts, in application order:
  1. state-police post/troop/district footprints   (build_state_police_post_footprints.py)
  2. tribal AIANNH footprints + Class D reroutes    (build_tribal_aiannh_footprints.py)
  3. fail-open custom footprints (6 ORIs)           (reviewed evidence table)
  4. municipal misresolution overrides (23 cases)   (reviewed evidence table)
  5. hand-reviewed footprint/plausibility rows      (reviewed evidence table)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crimerisk.allocation import OVERLAP_FOOTPRINT_OVERRIDE_COLUMNS
from crimerisk.paths import get_paths

LOCAL_COLUMNS = (
    "ori",
    "decision",
    "replacement_geo_type",
    "replacement_geoid",
    "replacement_jurisdiction_name",
    "confidence",
    "source_note",
    "reviewer_note",
)
CUSTOM_FOOTPRINT_COLUMNS = (
    "ori",
    "state_fips",
    "block_group_geoid",
    "weight_share",
    "bg_population_coverage_share",
    "weight_share_basis",
    "geometry_source_type",
    "geometry_source_ref",
    "footprint_note",
)


def _read(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[list(columns)]


def assemble(*, base_dir: Path, staging_dir: Path, out_dir: Path) -> dict[str, int]:
    generated = staging_dir / "generated"

    local = _read(base_dir / "local_resolution_overrides.csv", LOCAL_COLUMNS)
    local_parts = [
        local,
        _read(generated / "tribal_local_resolution_overrides.add.csv", LOCAL_COLUMNS),
        _read(staging_dir / "municipal_misresolution_overrides.csv", LOCAL_COLUMNS),
        _read(staging_dir / "reviewed_local_resolution_overrides.csv", LOCAL_COLUMNS),
    ]
    local = pd.concat(local_parts, ignore_index=True)
    local = local[local["ori"].astype(str).str.strip().ne("")]
    local = local.drop_duplicates("ori", keep="last").sort_values("ori").reset_index(drop=True)

    overrides = _read(base_dir / "overlap_footprint_overrides.csv", OVERLAP_FOOTPRINT_OVERRIDE_COLUMNS)
    override_parts = [
        overrides,
        _read(generated / "overlap_footprint_overrides.add.csv", OVERLAP_FOOTPRINT_OVERRIDE_COLUMNS),
        _read(generated / "tribal_overlap_footprint_overrides.add.csv", OVERLAP_FOOTPRINT_OVERRIDE_COLUMNS),
        _read(staging_dir / "reviewed_overlap_footprint_overrides.csv", OVERLAP_FOOTPRINT_OVERRIDE_COLUMNS),
    ]
    overrides = pd.concat(override_parts, ignore_index=True)
    overrides = overrides[overrides["ori"].astype(str).str.strip().ne("")]
    overrides = overrides.drop_duplicates("ori", keep="last").sort_values("ori").reset_index(drop=True)

    footprints = _read(base_dir / "overlap_custom_footprints.csv", CUSTOM_FOOTPRINT_COLUMNS)
    footprint_parts = [
        footprints,
        _read(generated / "overlap_custom_footprints.add.csv", CUSTOM_FOOTPRINT_COLUMNS),
        _read(generated / "tribal_overlap_custom_footprints.add.csv", CUSTOM_FOOTPRINT_COLUMNS),
        _read(staging_dir / "failopen_custom_footprints.csv", CUSTOM_FOOTPRINT_COLUMNS),
        _read(staging_dir / "reviewed_overlap_custom_footprints.csv", CUSTOM_FOOTPRINT_COLUMNS),
    ]
    # A later part REPLACES an ORI's footprint wholesale, it does not merge into it: a rebuilt
    # footprint normally has a different block-group SET (Harvard drops from 56 rows to 29), and
    # row-level dedup would leave the old rows behind and break the weights-sum-to-1 invariant.
    footprints = pd.DataFrame(columns=list(CUSTOM_FOOTPRINT_COLUMNS))
    for part in footprint_parts:
        part = part[part["ori"].astype(str).str.strip().ne("")]
        if part.empty:
            continue
        if not footprints.empty:
            footprints = footprints[~footprints["ori"].isin(set(part["ori"]))]
        footprints = pd.concat([footprints, part], ignore_index=True)
    footprints = footprints.drop_duplicates(
        ["ori", "state_fips", "block_group_geoid"], keep="last"
    ).reset_index(drop=True)

    # An override may only declare displacement if every one of its footprint rows carries a
    # positive coverage share; the loader enforces this too, but failing here names the fix.
    coverage_by_ori = footprints.groupby("ori")["bg_population_coverage_share"].apply(
        lambda values: all(str(v).strip() not in ("", "0", "0.0") for v in values)
    )
    declared = overrides["displaces_county_remainder"].astype(str).str.strip().str.upper().eq("TRUE")
    bad = overrides.loc[declared & ~overrides["ori"].map(coverage_by_ori).fillna(False), "ori"].tolist()
    if bad:
        overrides.loc[overrides["ori"].isin(bad), "displaces_county_remainder"] = ""
        print(
            "  note: cleared displaces_county_remainder for footprints with no positive resident "
            f"coverage: {bad}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    local.to_csv(out_dir / "local_resolution_overrides.csv", index=False)
    overrides.to_csv(out_dir / "overlap_footprint_overrides.csv", index=False)
    footprints.to_csv(out_dir / "overlap_custom_footprints.csv", index=False)
    return {
        "local_resolution_overrides": len(local),
        "overlap_footprint_overrides": len(overrides),
        "overlap_custom_footprints": len(footprints),
        "overlap_custom_footprint_oris": int(footprints["ori"].nunique()),
    }


def main() -> int:
    repo_root = get_paths().repo_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=repo_root / "analysis_scratch" / "stage2_fix_batch" / "base")
    parser.add_argument("--staging-dir", type=Path, default=repo_root / "analysis_scratch" / "stage2_fix_batch")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "configs")
    args = parser.parse_args()
    stats = assemble(base_dir=args.base_dir, staging_dir=args.staging_dir, out_dir=args.out_dir)
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
