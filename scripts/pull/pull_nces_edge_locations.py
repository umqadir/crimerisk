#!/usr/bin/env python3
"""
Download and normalize NCES EDGE school geocode files.

Outputs:
    data/NCES-EDGE/raw/*.zip
    data/NCES-EDGE/parsed/public_school_locations_2425.parquet
    data/NCES-EDGE/parsed/postsecondary_locations_2425.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.covariates.nces_edge import NCES_EDGE_DATASETS, download_nces_edge_zip, load_nces_edge_locations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "NCES-EDGE" / "raw",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "NCES-EDGE" / "parsed",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.parsed_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for key, dataset in NCES_EDGE_DATASETS.items():
        zip_path = download_nces_edge_zip(
            dataset=dataset,
            out_dir=args.raw_dir,
            overwrite=bool(args.overwrite),
        )
        frame = load_nces_edge_locations(zip_path=zip_path, dataset=dataset)
        out_path = args.parsed_dir / f"{key}_locations_2425.parquet"
        frame.to_parquet(out_path, index=False)
        written.append(out_path)
        print(f"{key}: {len(frame):,} rows -> {out_path}")

    combined = pd.concat([pd.read_parquet(path) for path in written], ignore_index=True)
    combined_out = args.parsed_dir / "education_anchor_locations_2425.parquet"
    combined.to_parquet(combined_out, index=False)
    print(f"combined: {len(combined):,} rows -> {combined_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
