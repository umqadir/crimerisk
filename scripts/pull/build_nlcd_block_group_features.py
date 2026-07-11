#!/usr/bin/env python3
"""
Build block-group NLCD land-cover and impervious features for the active production universe.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.covariates.nlcd import NlcdBuildConfig, build_block_group_nlcd_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bg-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tiger_bg",
    )
    parser.add_argument(
        "--crosswalk-parquet",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument(
        "--state-cache-dir",
        type=Path,
        default=REPO_ROOT / "state" / "cache" / "NLCD",
    )
    parser.add_argument(
        "--state-out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "NLCD" / "state_block_groups",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "NLCD" / "block_group_nlcd_2023.parquet",
    )
    parser.add_argument("--land-cover-year", type=int, default=2023)
    parser.add_argument("--impervious-year", type=int, default=2023)
    parser.add_argument("--bbox-buffer-m", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    frame = build_block_group_nlcd_features(
        bg_dir=args.bg_dir,
        block_group_crosswalk_parquet=args.crosswalk_parquet,
        state_cache_dir=args.state_cache_dir,
        state_out_dir=args.state_out_dir,
        cfg=NlcdBuildConfig(
            land_cover_year=int(args.land_cover_year),
            impervious_year=int(args.impervious_year),
            bbox_buffer_m=float(args.bbox_buffer_m),
            timeout_seconds=int(args.timeout_seconds),
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"rows={len(frame):,} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
