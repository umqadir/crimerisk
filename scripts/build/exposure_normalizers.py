from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.exposure_ensemble import (
    RESIDENTIAL_LEG_SOURCES,
    ExposureEnsembleConfig,
    write_v2_exposure_normalizers,
)
from crimerisk.paths import RepoPaths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Block-group opportunity normalizers for the v2 exposure lane: a per-offense convex "
            "ensemble over LandScan night, LandScan day and the LODES daytime jobs proxy, plus "
            "larceny's destination-POI/retail-jobs hybrid. Weights are frozen in "
            "configs/exposure_ensemble_weights_v1.csv and are read, never re-derived."
        )
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--residential-leg-source",
        choices=RESIDENTIAL_LEG_SOURCES,
        default="landscan_night",
    )
    parser.add_argument("--baseline-normalizers-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--enable-qcew-exposure-updating",
        action="store_true",
        help=(
            "PLAN item 6: move the LODES-derived legs to the target year with county QCEW "
            "employment ratios and publish the vintages. Default off; writes the _qcew artifact."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path, summary = write_v2_exposure_normalizers(
        paths=RepoPaths.from_repo_root(REPO_ROOT),
        out_path=args.out,
        config=ExposureEnsembleConfig(
            year=int(args.year),
            enable_qcew_exposure_updating=bool(args.enable_qcew_exposure_updating),
            residential_leg_source=str(args.residential_leg_source),
            baseline_normalizers_path=args.baseline_normalizers_path,
        ),
        force=bool(args.force),
    )
    print(f"wrote {out_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
