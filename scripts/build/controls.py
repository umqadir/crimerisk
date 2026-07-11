from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths
from crimerisk.controls import ControlBuildConfig, write_v2_controls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--state-out",
        type=Path,
        default=REPO_ROOT / "state" / "controls" / "state_control_comparison.parquet",
    )
    parser.add_argument(
        "--jurisdiction-out",
        type=Path,
        default=REPO_ROOT / "state" / "controls" / "jurisdiction_controls_2024.parquet",
    )
    parser.add_argument(
        "--jurisdiction-year-estimates-out",
        type=Path,
        default=REPO_ROOT / "state" / "controls" / "jurisdiction_year_estimates.parquet",
    )
    parser.add_argument(
        "--force-reporting-regimes-rebuild",
        "--force-reporting-regimes",
        action="store_true",
        dest="force_reporting_regimes_rebuild",
    )
    parser.add_argument(
        "--force-municipal-estimates-rebuild",
        "--force-municipal-estimates",
        action="store_true",
        dest="force_municipal_estimates_rebuild",
    )
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    state_out, jurisdiction_out, jurisdiction_year_estimates_out = write_v2_controls(
        paths=paths,
        state_out_path=args.state_out,
        jurisdiction_out_path=args.jurisdiction_out,
        jurisdiction_year_estimates_out_path=args.jurisdiction_year_estimates_out,
        config=ControlBuildConfig(
            year=int(args.year),
            force_reporting_regimes_rebuild=bool(args.force_reporting_regimes_rebuild),
            force_municipal_estimates_rebuild=bool(args.force_municipal_estimates_rebuild),
        ),
    )
    print(f"wrote {state_out}")
    print(f"wrote {jurisdiction_out}")
    if jurisdiction_year_estimates_out is not None:
        print(f"wrote {jurisdiction_year_estimates_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
