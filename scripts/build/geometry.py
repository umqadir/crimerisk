from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths
from crimerisk.geometry import GeometryBuildConfig, write_v2_geometry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical V2 block and block-group jurisdiction crosswalks."
    )
    parser.add_argument(
        "--block-out",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument(
        "--block-group-out",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument("--force-rebuild", "--force", action="store_true", dest="force_rebuild")
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    block_out, block_group_out = write_v2_geometry(
        paths=paths,
        block_out_path=args.block_out,
        block_group_out_path=args.block_group_out,
        config=GeometryBuildConfig(),
        force_rebuild=bool(args.force_rebuild),
    )
    print(f"wrote {block_out}")
    print(f"wrote {block_group_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
