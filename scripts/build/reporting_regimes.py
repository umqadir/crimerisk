from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import get_paths
from crimerisk.reporting_regimes import ReportingRegimeBuildConfig, write_v2_reporting_regimes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build agency-year reporting-regime artifact.")
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument(
        "--override-path",
        type=Path,
        default=Path("configs/reporting_regime_overrides.csv"),
    )
    parser.add_argument(
        "--source-override-path",
        type=Path,
        default=Path("configs/source_preference_overrides.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("state/modeling/agency_year_reporting_regimes.parquet"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    summary = write_v2_reporting_regimes(
        paths=paths,
        out_path=paths.repo_root / args.out,
        config=ReportingRegimeBuildConfig(
            year_start=args.year_start,
            year_end=args.year_end,
            override_path=(paths.repo_root / args.override_path),
            source_override_path=(paths.repo_root / args.source_override_path),
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {paths.repo_root / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
