from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.local_publications import (
    get_v2_local_publication_input_path,
    promote_v2_local_publication_inputs,
)
from crimerisk.paths import get_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote reviewed municipal packet publication extracts into the canonical local-publication input surface."
    )
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument(
        "--out",
        type=Path,
        default=get_v2_local_publication_input_path(get_paths()).relative_to(get_paths().repo_root),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    out_path = paths.repo_root / args.out
    summary = promote_v2_local_publication_inputs(
        paths=paths,
        out_path=out_path,
        year_start=args.year_start,
        year_end=args.year_end,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
