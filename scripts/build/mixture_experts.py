from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.mixture_allocation import MixtureAllocationConfig, write_v2_mixture_experts
from crimerisk.paths import RepoPaths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Block-group expert table for the three-expert mixture allocator: the bounded "
            "downward GBM and the population-exposure expert. The mixture weights are frozen "
            "in configs/mixture_ship_weights_v2.csv and are read, never re-derived."
        )
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path, summary = write_v2_mixture_experts(
        paths=RepoPaths.from_repo_root(REPO_ROOT),
        out_path=args.out,
        config=MixtureAllocationConfig(year=int(args.year)),
        force=bool(args.force),
    )
    print(f"wrote {out_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
