#!/usr/bin/env python3
"""
Pull LODES workplace, residence, and commute-flow inputs at block group level.

LODES provides job counts and origin-destination flows at Census block level.
This script aggregates them to block groups for downstream daytime and commute features.

Usage:
    python scripts/pull/pull_lodes_block_groups.py

Output:
    data/LODES/parsed/lodes_wac_block_groups.parquet
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.covariates.lodes_flow import STATE_FIPS_TO_ABBR, build_state_block_group_lodes_frame
from crimerisk.paths import get_paths


def _migrate_legacy_lodes_cache(*, legacy_cache_dir: Path, canonical_cache_dir: Path) -> None:
    if not legacy_cache_dir.exists():
        return

    canonical_cache_dir.parent.mkdir(parents=True, exist_ok=True)

    if not canonical_cache_dir.exists():
        legacy_cache_dir.rename(canonical_cache_dir)
        return

    for legacy_path in sorted(legacy_cache_dir.rglob("*")):
        if not legacy_path.is_file():
            continue
        relative_path = legacy_path.relative_to(legacy_cache_dir)
        canonical_path = canonical_cache_dir / relative_path
        if canonical_path.exists():
            legacy_path.unlink()
            continue
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(canonical_path))

    for old_dir in sorted((p for p in legacy_cache_dir.rglob("*") if p.is_dir()), reverse=True):
        old_dir.rmdir()
    legacy_cache_dir.rmdir()


def _state_lodes_parquet_is_valid(*, path: Path, state_fips: str) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["bg_id", "tract_id", "state_fips", "lodes_year"])
    except Exception:
        return False
    if df.empty:
        return False
    state = str(state_fips).zfill(2)
    bg_prefix = df["bg_id"].astype(str).str.zfill(12).str.slice(0, 2)
    tract_prefix = df["tract_id"].astype(str).str.zfill(11).str.slice(0, 2)
    state_col = df["state_fips"].astype(str).str.zfill(2)
    if not (bg_prefix.eq(state) & tract_prefix.eq(state) & state_col.eq(state)).all():
        return False
    if df.duplicated(["bg_id", "tract_id", "state_fips"]).any():
        return False
    if "lodes_year" in df.columns and pd.to_numeric(df["lodes_year"], errors="coerce").nunique(dropna=True) > 1:
        return False
    return True


def main():
    paths = get_paths()
    lodes_dir = paths.data_dir / "LODES" / "raw" / "wac_2023"
    cache_dir = paths.cache_dir
    legacy_lodes_cache_dir = paths.data_dir / "cache" / "lodes"
    output_path = paths.data_dir / "LODES" / "parsed" / "lodes_wac_block_groups.parquet"
    state_output_dir = paths.data_dir / "LODES" / "state_block_groups"

    lodes_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_output_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_lodes_cache(
        legacy_cache_dir=legacy_lodes_cache_dir,
        canonical_cache_dir=cache_dir / "lodes",
    )

    all_dfs = []
    failed_states = []

    for i, (state_fips, state_abbr) in enumerate(sorted(STATE_FIPS_TO_ABBR.items())):
        print(f"[{i+1}/52] Processing {state_fips} ({state_abbr.upper()})...", end=" ", flush=True)
        state_output_path = state_output_dir / f"{state_fips}.parquet"

        if _state_lodes_parquet_is_valid(path=state_output_path, state_fips=state_fips):
            df = pd.read_parquet(state_output_path)
            print(f"using cached state parquet with {len(df):,} block groups")
            all_dfs.append(df)
            continue
        if state_output_path.exists():
            print("cached state parquet invalid; rebuilding", end=" ", flush=True)
            state_output_path.unlink()

        try:
            df, year_used = build_state_block_group_lodes_frame(
                state_fips=state_fips,
                lodes_dir=lodes_dir,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            print(f"FAILED ({type(exc).__name__}: {exc})")
            failed_states.append(state_fips)
            continue

        if df is not None:
            print(f"got {len(df):,} block groups (year {year_used})")
            df.to_parquet(state_output_path, index=False)
            all_dfs.append(df)
        else:
            print("FAILED")
            failed_states.append(state_fips)

    if not all_dfs:
        print("No data retrieved!")
        return 1

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["bg_id"] = combined["bg_id"].astype(str).str.zfill(12)
    combined["tract_id"] = combined["bg_id"].str[:11]
    combined["state_fips"] = combined["bg_id"].str[:2]

    combined.to_parquet(output_path, index=False)

    print(f"\nSaved {len(combined):,} block groups to {output_path}")
    print(f"Total jobs: {combined['jobs_wac'].sum():,}")
    print(f"States succeeded: {52 - len(failed_states)}")
    if failed_states:
        print(f"Failed states: {failed_states}")

    return 0


if __name__ == "__main__":
    exit(main())
