from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_shares import (
    get_v2_city_incident_input_root,
    promote_v2_city_incident_inputs,
)
from crimerisk.paths import get_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote reviewed city packet gate files into the canonical city-incident input surface."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=get_v2_city_incident_input_root(get_paths()).relative_to(get_paths().repo_root),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    summary = promote_v2_city_incident_inputs(
        paths=paths,
        out_root=paths.repo_root / args.out_root,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
