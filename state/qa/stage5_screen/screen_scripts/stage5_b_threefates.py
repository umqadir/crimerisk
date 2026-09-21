"""Stage 5 screen (b) addendum: the three-fates class at two strictnesses, plus the
Maine exemplar neighbourhood, rewritten over s5b_three_fates_triples.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone/state/qa/stage5_screen/screen_scripts")
from stage5_common import INDEX_BREAKS, OFFENSES_7, load_bg, write  # noqa: E402

REPO = Path("/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone")
adj = pd.read_parquet(REPO / "state/qa/stage5_screen/bg_adjacency_2020.parquet")
bg = load_bg().set_index("block_group_geoid")

LAYER = "index_total_primary_event_weighted"
idx = pd.to_numeric(bg[LAYER], errors="coerce")
cnt = pd.to_numeric(bg["expected_count_total"], errors="coerce").fillna(0.0)
band = pd.Series(np.nan, index=idx.index, dtype=float)
band.loc[idx.notna()] = np.searchsorted(np.array(INDEX_BREAKS), idx[idx.notna()].to_numpy(), side="right").astype(float)

info = pd.DataFrame(
    {
        "band": band,
        "idx": idx,
        "cnt": cnt,
        "land": pd.to_numeric(bg["land_area_sq_mi"], errors="coerce"),
        "pop": pd.to_numeric(bg["population_2024"], errors="coerce"),
        "expo": pd.to_numeric(bg["exposure_proxy_2024"], errors="coerce"),
    }
)

nb = adj.join(info.add_suffix("_nb"), on="nb")
g = nb.groupby("bg")
agg = pd.DataFrame(
    {
        "n_nb": g.size(),
        "n_grey_nb": g["band_nb"].apply(lambda s: int(s.isna().sum())),
        "nb_band_min": g["band_nb"].min(),
        "nb_band_max": g["band_nb"].max(),
        "nb_idx_min": g["idx_nb"].min(),
        "nb_idx_max": g["idx_nb"].max(),
        "nb_cnt_max": g["cnt_nb"].max(),
        "nb_land_max": g["land_nb"].max(),
        "nb_pop_max": g["pop_nb"].max(),
    }
).join(info, how="left")

self_band = agg["band"]
band_lo = pd.concat([agg["nb_band_min"], self_band], axis=1).min(axis=1)
band_hi = pd.concat([agg["nb_band_max"], self_band], axis=1).max(axis=1)
agg["neighbourhood_band_span"] = band_hi - band_lo
agg["neighbourhood_has_grey"] = (agg["n_grey_nb"] >= 1) | agg["band"].isna()
agg["neighbourhood_max_count"] = pd.concat([agg["nb_cnt_max"], agg["cnt"]], axis=1).max(axis=1)

STRICT = (
    agg["neighbourhood_has_grey"]
    & (band_lo <= 2)
    & (band_hi >= 7)
    & (agg["neighbourhood_max_count"] <= 25.0)
)
LOOSE = (
    agg["neighbourhood_has_grey"]
    & (agg["neighbourhood_band_span"] >= 4)
    & (agg["neighbourhood_max_count"] <= 50.0)
)
NO_GREY_CHASM = (~agg["neighbourhood_has_grey"]) & (agg["neighbourhood_band_span"] >= 5) & (
    agg["neighbourhood_max_count"] <= 25.0
)

out = agg.copy()
out["class_strict_three_fates"] = STRICT
out["class_loose_grey_plus_4band_span"] = LOOSE
out["class_no_grey_5band_chasm"] = NO_GREY_CHASM
out = out[STRICT | LOOSE | NO_GREY_CHASM].reset_index().rename(columns={"bg": "block_group_geoid"})
out["state_fips"] = out["block_group_geoid"].str.slice(0, 2)
out = out[
    [
        "block_group_geoid",
        "state_fips",
        "class_strict_three_fates",
        "class_loose_grey_plus_4band_span",
        "class_no_grey_5band_chasm",
        "n_nb",
        "n_grey_nb",
        "band",
        "idx",
        "cnt",
        "pop",
        "expo",
        "land",
        "nb_band_min",
        "nb_band_max",
        "nb_idx_min",
        "nb_idx_max",
        "nb_cnt_max",
        "nb_pop_max",
        "nb_land_max",
        "neighbourhood_band_span",
        "neighbourhood_max_count",
    ]
].sort_values(["state_fips", "block_group_geoid"])
write(out, "s5b_three_fates_triples.csv")

print("strict", int(STRICT.sum()), "loose", int(LOOSE.sum()), "no-grey chasm", int(NO_GREY_CHASM.sum()))
print("by state (loose):")
print(out[out["class_loose_grey_plus_4band_span"]].groupby("state_fips").size().sort_values(ascending=False).head(15).to_string())
print()
me = out[out["state_fips"] == "23"]
print(f"Maine rows in the class: {len(me)}")
print(me.head(20).to_string())
