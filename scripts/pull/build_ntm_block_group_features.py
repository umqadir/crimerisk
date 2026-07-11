from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.covariates.roads import infer_state_fips_from_bg_dir  # noqa: E402
from crimerisk.covariates.transit import (  # noqa: E402
    NationalTransitMapBuildConfig,
    build_block_group_transit_stop_features,
    download_ntm_stops_shapefile,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build National Transit Map stop-access block-group features from the official NTM stops shapefile."
    )
    parser.add_argument("--bg-dir", type=Path, default=REPO_ROOT / "data" / "tiger_bg")
    parser.add_argument(
        "--raw-zip",
        type=Path,
        default=REPO_ROOT / "data" / "NTM" / "raw" / "national_transit_map_stops.zip",
    )
    parser.add_argument(
        "--state-out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "NTM" / "state_block_groups",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "NTM" / "parsed" / "block_group_transit_stops.parquet",
    )
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--bbox-padding-degrees", type=float, default=0.1)
    parser.add_argument("--projected-crs", type=str, default="EPSG:5070")
    parser.add_argument("--overwrite-download", action="store_true")
    args = parser.parse_args()

    raw_zip = download_ntm_stops_shapefile(
        out_zip_path=args.raw_zip,
        cfg=NationalTransitMapBuildConfig(
            timeout_seconds=int(args.timeout_seconds),
            projected_crs=str(args.projected_crs),
            bbox_padding_degrees=float(args.bbox_padding_degrees),
        ),
        overwrite=bool(args.overwrite_download),
    )

    states = [str(s).zfill(2) for s in args.states] if args.states else infer_state_fips_from_bg_dir(args.bg_dir, year=2020)
    frame = build_block_group_transit_stop_features(
        stops_zip_path=raw_zip,
        bg_dir=args.bg_dir,
        state_fips_values=states,
        cfg=NationalTransitMapBuildConfig(
            timeout_seconds=int(args.timeout_seconds),
            projected_crs=str(args.projected_crs),
            bbox_padding_degrees=float(args.bbox_padding_degrees),
        ),
    )

    args.state_out_dir.mkdir(parents=True, exist_ok=True)
    for state_fips, state_frame in frame.groupby("state_fips", sort=True):
        state_frame.to_parquet(args.state_out_dir / f"{str(state_fips).zfill(2)}.parquet", index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"bg_rows={len(frame):,}")
    print(f"raw_zip={raw_zip}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
