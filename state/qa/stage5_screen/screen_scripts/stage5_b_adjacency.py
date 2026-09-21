"""Build the national block-group adjacency list from TIGER 2020 BG shapefiles.

Written once to state/qa/stage5_screen/bg_adjacency_2020.parquet (working input, not a deliverable).
Rook+queen adjacency = polygons that intersect (share any boundary point). Cross-state
adjacency is included: all states are unioned before the spatial join.
"""
from __future__ import annotations

import glob
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path("/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone")
OUT = REPO / "state/qa/stage5_screen/bg_adjacency_2020.parquet"
EXCLUDE = {"02", "15", "72", "60", "66", "69", "78"}  # release-excluded / territories

if OUT.exists():
    print(f"{OUT} exists, skipping")
    raise SystemExit(0)

files = sorted(glob.glob(str(REPO / "data/tiger_bg/tl_2020_*_bg.zip")))
frames = []
t0 = time.time()
for f in files:
    st = Path(f).name.split("_")[2]
    if st in EXCLUDE:
        continue
    g = gpd.read_file(f, columns=["GEOID", "geometry"])
    g = g[["GEOID", "geometry"]].rename(columns={"GEOID": "bg"})
    frames.append(g)
    print(f"  {st} {len(g):,}  ({time.time()-t0:.0f}s)", flush=True)

gdf = pd.concat(frames, ignore_index=True)
gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=frames[0].crs)
gdf["bg"] = gdf["bg"].astype(str).str.zfill(12)
print(f"total polygons {len(gdf):,}  ({time.time()-t0:.0f}s)")

gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
pairs = gpd.sjoin(gdf, gdf, how="inner", predicate="intersects")
print(f"raw pairs {len(pairs):,}  ({time.time()-t0:.0f}s)")
adj = pd.DataFrame(
    {
        "bg": pairs["bg_left"].to_numpy(),
        "nb": gdf["bg"].to_numpy()[pairs["index_right"].to_numpy()],
    }
)
adj = adj[adj["bg"] != adj["nb"]].drop_duplicates().reset_index(drop=True)
adj.to_parquet(OUT, index=False)
print(f"wrote {OUT}  rows={len(adj):,}  ({time.time()-t0:.0f}s)")
print("mean degree", len(adj) / adj["bg"].nunique())
