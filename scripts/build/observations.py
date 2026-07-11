from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import get_paths
from crimerisk.observations import ObservationBuildConfig, write_v2_observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build agency-year and jurisdiction-year observations.")
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument(
        "--agency-out",
        type=Path,
        default=Path("state/observations/agency_year_observations.parquet"),
    )
    parser.add_argument(
        "--jurisdiction-out",
        type=Path,
        default=Path("state/observations/jurisdiction_year_observations.parquet"),
    )
    parser.add_argument(
        "--local-publication-input-path",
        type=Path,
        default=Path("state/modeling/inputs/local_publication_annual.parquet"),
    )
    parser.add_argument(
        "--state-publication-input-path",
        type=Path,
        default=Path("state/modeling/inputs/state_publication_annual.parquet"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    summary = write_v2_observations(
        paths=paths,
        agency_out_path=paths.repo_root / args.agency_out,
        jurisdiction_out_path=paths.repo_root / args.jurisdiction_out,
        config=ObservationBuildConfig(
            year_start=args.year_start,
            year_end=args.year_end,
            local_publication_input_path=paths.repo_root / args.local_publication_input_path,
            state_publication_input_path=paths.repo_root / args.state_publication_input_path,
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
