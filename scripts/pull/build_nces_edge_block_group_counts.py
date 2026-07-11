#!/usr/bin/env python3
"""
Aggregate NCES EDGE school and postsecondary locations to Census block groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.covariates.nces_edge import build_block_group_education_anchor_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-school-parquet",
        type=Path,
        default=REPO_ROOT / "data" / "NCES-EDGE" / "parsed" / "public_school_locations_2425.parquet",
    )
    parser.add_argument(
        "--postsecondary-parquet",
        type=Path,
        default=REPO_ROOT / "data" / "NCES-EDGE" / "parsed" / "postsecondary_locations_2425.parquet",
    )
    parser.add_argument(
        "--bg-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tiger_bg",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "NCES-EDGE" / "parsed" / "block_group_education_anchors_2425.parquet",
    )
    args = parser.parse_args()

    frame = build_block_group_education_anchor_counts(
        public_school_parquet=args.public_school_parquet,
        postsecondary_parquet=args.postsecondary_parquet,
        bg_dir=args.bg_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"rows={len(frame):,} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
