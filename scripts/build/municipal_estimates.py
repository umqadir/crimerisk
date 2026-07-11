from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths
from crimerisk.municipal_estimator import MunicipalEstimatorConfig, write_v2_municipal_estimates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "municipal_estimates_2024.parquet",
    )
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--target-year", type=int, default=2024)
    parser.add_argument(
        "--force-reporting-regimes-rebuild",
        action="store_true",
        help="Rebuild agency_year_reporting_regimes.parquet instead of reusing the cached artifact.",
    )
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    summary = write_v2_municipal_estimates(
        paths=paths,
        out_path=args.out,
        config=MunicipalEstimatorConfig(
            year_start=int(args.year_start),
            year_end=int(args.year_end),
            target_year=int(args.target_year),
            force_reporting_regimes_rebuild=bool(args.force_reporting_regimes_rebuild),
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
