from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.covariates.cms_hospitals import (  # noqa: E402
    CmsHospitalBuildConfig,
    build_cms_hospital_frame_by_block_group,
    download_cms_hospital_general_info,
)
from crimerisk.covariates.roads import infer_state_fips_from_bg_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CMS hospital-anchor block-group features from the official CMS hospital general information dataset.")
    parser.add_argument("--bg-dir", type=Path, default=REPO_ROOT / "data" / "tiger_bg")
    parser.add_argument("--raw-csv", type=Path, default=REPO_ROOT / "data" / "CMS-Hospital-General-Info" / "raw" / "Hospital_General_Information_latest.csv")
    parser.add_argument("--points-out", type=Path, default=REPO_ROOT / "data" / "CMS-Hospital-General-Info" / "parsed" / "cms_hospital_points.parquet")
    parser.add_argument("--state-out-dir", type=Path, default=REPO_ROOT / "data" / "CMS-Hospital-General-Info" / "parsed" / "state_block_groups")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "CMS-Hospital-General-Info" / "parsed" / "block_group_hospital_anchors.parquet")
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    raw_csv = args.raw_csv
    if not raw_csv.exists():
        raw_csv = download_cms_hospital_general_info(
            out_csv_path=raw_csv,
            cfg=CmsHospitalBuildConfig(timeout_seconds=int(args.timeout_seconds), batch_size=int(args.batch_size)),
        )

    states = [str(s).zfill(2) for s in args.states] if args.states else infer_state_fips_from_bg_dir(args.bg_dir, year=2020)
    points, bg = build_cms_hospital_frame_by_block_group(
        bg_dir=args.bg_dir,
        raw_csv_path=raw_csv,
        state_fips_values=states,
        cfg=CmsHospitalBuildConfig(timeout_seconds=int(args.timeout_seconds), batch_size=int(args.batch_size)),
    )

    args.points_out.parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(args.points_out, index=False)

    args.state_out_dir.mkdir(parents=True, exist_ok=True)
    for state_fips, frame in bg.groupby("state_fips", sort=True):
        frame.to_parquet(args.state_out_dir / f"{str(state_fips).zfill(2)}.parquet", index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bg.to_parquet(args.out, index=False)
    print(f"cms_hospital_points={len(points):,}")
    print(f"bg_rows={len(bg):,}")
    print(f"points_out={args.points_out}")
    print(f"bg_out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
