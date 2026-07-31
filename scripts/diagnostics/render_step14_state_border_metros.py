"""Compare arm_b (no state FE) vs step10 baseline (with state FE) at state-border metros.
Renders block-group choropleths of index_total_part1_resident, state borders drawn thick,
so visible state-line cliffs (if any) are obvious. Output: /tmp/crimerisk_arm_compare/*.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path("/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone")
OUT = Path("/tmp/crimerisk_arm_compare"); OUT.mkdir(exist_ok=True)
FIELD = "index_total_part1_resident"
ARMS = {
    "baseline_stateFE": REPO/"state/candidates/step10-confidence-layer/crimerisk_block_group_2024_ags_core.parquet",
    "arm_b_no_stateFE": REPO/"state/candidates/step14-arm-b/crimerisk_block_group_2024_ags_core.parquet",
}
METROS = {
    "DC_MD_VA":      dict(states=["11","24","51"],       bbox=(-77.65,-76.85,38.70,39.15)),
    "KansasCity_MOKS":dict(states=["29","20"],           bbox=(-95.05,-94.30,38.80,39.45)),
    "Cincinnati_OHKYIN":dict(states=["39","21","18"],    bbox=(-84.85,-84.20,38.90,39.42)),
    "StLouis_MOIL":  dict(states=["29","17"],            bbox=(-90.65,-89.85,38.40,38.92)),
}

# load only the state BG geometries we need, once
need_states = sorted({s for m in METROS.values() for s in m["states"]})
geo = []
for ss in need_states:
    z = REPO/f"data/tiger_bg/tl_2020_{ss}_bg.zip"
    if z.exists():
        g = gpd.read_file(f"zip://{z}")[["GEOID","STATEFP","geometry"]]
        geo.append(g)
geo = pd.concat(geo, ignore_index=True)
geo = gpd.GeoDataFrame(geo, geometry="geometry", crs=geo.crs).to_crs(4326)
print("geometry rows:", len(geo))

# load both arms' index, keyed by block_group_geoid
idx = {}
for name, p in ARMS.items():
    df = pd.read_parquet(p, columns=["block_group_geoid", FIELD])
    df["block_group_geoid"] = df["block_group_geoid"].astype(str).str.zfill(12)
    idx[name] = df.set_index("block_group_geoid")[FIELD]
geo["GEOID"] = geo["GEOID"].astype(str).str.zfill(12)

for metro, cfg in METROS.items():
    lon0,lon1,lat0,lat1 = cfg["bbox"]
    sub = geo[geo["STATEFP"].isin(cfg["states"])].cx[lon0:lon1, lat0:lat1].copy()
    # shared robust color scale across both arms for this metro
    vals = pd.concat([idx["baseline_stateFE"].reindex(sub["GEOID"]),
                      idx["arm_b_no_stateFE"].reindex(sub["GEOID"])])
    vmin, vmax = np.nanpercentile(vals.dropna(), [4,96])
    state_borders = sub.dissolve("STATEFP").boundary
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    for ax, arm in zip(axes, ["baseline_stateFE","arm_b_no_stateFE"]):
        sub["_v"] = idx[arm].reindex(sub["GEOID"]).values
        sub.plot(column="_v", ax=ax, cmap="magma_r", vmin=vmin, vmax=vmax,
                 linewidth=0, edgecolor="none", missing_kwds={"color":"#dddddd"})
        state_borders.plot(ax=ax, color="cyan", linewidth=1.6)
        ax.set_xlim(lon0,lon1); ax.set_ylim(lat0,lat1); ax.set_axis_off()
        ax.set_title(f"{metro}  ·  {arm}\n{FIELD}  (state borders cyan)", fontsize=11)
    plt.tight_layout()
    f = OUT/f"{metro}_compare.png"; plt.savefig(f, dpi=110, bbox_inches="tight"); plt.close()
    print("wrote", f)
print("DONE")
