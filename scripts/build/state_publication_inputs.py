from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import get_paths
from crimerisk.state_publications import (
    get_v2_state_publication_input_path,
    write_v2_state_publication_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the canonical state-publication annual input surface."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--out",
        type=Path,
        default=get_v2_state_publication_input_path(get_paths()).relative_to(get_paths().repo_root),
    )
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    out_path = paths.repo_root / args.out
    summary = write_v2_state_publication_inputs(
        paths=paths,
        out_path=out_path,
        year=args.year,
        force_refresh=bool(args.force_refresh),
        max_workers=int(args.max_workers),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
