"""Rook adjacency between 2020 block groups, read by frontend/build/05b_exception_ledger.py.

Two block groups are neighbours when their boundaries share more than 1 metre of line (EPSG:5070),
so corner-only contacts are not edges. Reads data/tiger_bg/tl_2020_<st>_bg.zip, writes
state/geometry/bg_rook_edges_2020.parquet (left_geoid, right_geoid, shared_m; left < right).

    uv run python scripts/build/build_bg_rook_edges.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "state" / "geometry" / "bg_rook_edges_2020.parquet"
MIN_SHARED_M = 1.0


def main() -> int:
    files = sorted((REPO / "data" / "tiger_bg").glob("tl_2020_*_bg.zip"))
    if not files:
        raise SystemExit("no data/tiger_bg/tl_2020_*_bg.zip files")
    bg = pd.concat([gpd.read_file(f)[["GEOID", "geometry"]].to_crs(5070) for f in files], ignore_index=True)
    bg = gpd.GeoDataFrame(bg, geometry="geometry", crs=5070)
    geoms = bg.geometry.values
    left, right = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    keep = left < right
    left, right = left[keep], right[keep]
    edges = shapely.boundary(geoms)
    shared = np.zeros(len(left))
    for start in range(0, len(left), 200_000):
        stop = start + 200_000
        shared[start:stop] = shapely.length(shapely.intersection(edges[left[start:stop]], edges[right[start:stop]]))
    rook = shared > MIN_SHARED_M
    ids = bg["GEOID"].astype(str).to_numpy()
    lo = np.minimum(ids[left[rook]], ids[right[rook]])
    hi = np.maximum(ids[left[rook]], ids[right[rook]])
    out = pd.DataFrame({"left_geoid": lo, "right_geoid": hi, "shared_m": shared[rook]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["left_geoid", "right_geoid"]).to_parquet(OUT, index=False)
    print(f"{len(bg):,} block groups, {len(out):,} rook edges -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
