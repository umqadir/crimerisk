"""Write the optional EB shrinkage audit for a published block-group or tract surface."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.eb_audit import eb_shrinkage_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True, help="A candidate crimerisk_{block_group,tract}_<year>_*.parquet.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    surface = pd.read_parquet(args.surface)
    geo_id_col = "block_group_geoid" if "block_group_geoid" in surface.columns else "tract_id"
    audit = eb_shrinkage_audit(surface, geo_id_col=geo_id_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(audit):,} rows, {audit.shape[1]:,} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
