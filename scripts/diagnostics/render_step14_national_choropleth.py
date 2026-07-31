"""National CONUS block-group choropleth, arm_b, QUANTILE color scale (mirrors the frontend),
with state borders drawn, to check for 'paint-by-state' slabs at national zoom."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from pathlib import Path

REPO = Path("/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone")
OUT = Path("/tmp/crimerisk_arm_compare"); OUT.mkdir(exist_ok=True)
FIELD = "index_total_part1_resident"
P = REPO/"state/candidates/step14-arm-b/crimerisk_block_group_2024_ags_core.parquet"

idx = pd.read_parquet(P, columns=["block_group_geoid", FIELD])
idx["block_group_geoid"] = idx["block_group_geoid"].astype(str).str.zfill(12)
idx = idx.set_index("block_group_geoid")[FIELD]

# load all BG geometry
import glob
geo = []
for z in sorted(glob.glob(str(REPO/"data/tiger_bg/tl_2020_*_bg.zip"))):
    ss = z.split("tl_2020_")[1][:2]
    if ss in {"02","15","60","66","69","72","78"}:  # drop AK/HI/territories for CONUS frame
        continue
    g = gpd.read_file(f"zip://{z}")[["GEOID","STATEFP","geometry"]]
    geo.append(g)
geo = pd.concat(geo, ignore_index=True)
geo = gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4269").to_crs(5070)  # Albers
geo["GEOID"] = geo["GEOID"].astype(str).str.zfill(12)
geo["_v"] = idx.reindex(geo["GEOID"]).values
print("rows:", len(geo), "published:", geo["_v"].notna().sum())

# quantile bins (deciles) like the frontend quantile colorway
vals = geo["_v"].dropna()
edges = np.unique(np.nanpercentile(vals, np.linspace(0,100,11)))
norm = BoundaryNorm(edges, ncolors=256)
states = geo.dissolve("STATEFP").boundary

fig, ax = plt.subplots(figsize=(20,12))
geo.plot(column="_v", ax=ax, cmap="magma_r", norm=norm, linewidth=0,
         missing_kwds={"color":"#e8e8e8"})
states.plot(ax=ax, color="#00d0ff", linewidth=0.5)
ax.set_axis_off()
ax.set_title("arm_b · index_total_part1_resident · block group · DECILE color scale · state borders cyan", fontsize=13)
plt.tight_layout()
f = OUT/"national_arm_b_quantile.png"; plt.savefig(f, dpi=95, bbox_inches="tight"); plt.close()
print("wrote", f)
PY = None
