"""Reweight the Ohio State University PD footprint onto a LandScan daytime basis.

Stage 2 fork ruling 4: OSU is the other campus custom footprint still allocating by LAND AREA
after Harvard was moved to LandScan USA 2021 daytime population. A campus is the case where an
area basis is worst: the acreage is athletics fields, parking and west-campus research land,
while the people are in the core academic and medical blocks. OSU's shipped area weights put
47.5% of OSUPD's mass on block group 390490011221 (LandScan day 17,867) and 10.0% on
390490011212 (LandScan day 37,537) -- the block group with the most modelled daytime population
in the whole footprint gets a fifth of the weight of one with half its daytime load.

Method, and how it differs from Harvard's:

    Harvard's weights are LandScan day summed over the 3-arcsecond raster cells whose centres
    fall inside (property polygon x block group). That needs the property polygon. OSU's
    footprint polygon is not stored in this repo -- it was built from live OSU ArcGIS campus-map
    layers -- so the polygon-level intersection cannot be reproduced here. What IS recoverable
    is the footprint's AREA inside each block group: the shipped `weight_share` is exactly that
    area, normalised. Combining it with the block group's LandScan daytime DENSITY gives the
    footprint's modelled daytime population per block group under a uniform-density assumption
    inside the block group:

        new_weight_i  proportional to  weight_share_i x landscan_day_pop_i / bg_land_area_i

    which is the same BASIS as Harvard (modelled daytime population, not acreage) at coarser
    resolution. The uniform-density step is the stated limitation; it is recorded in the
    footprint note rather than left for a reader to infer.

Block groups with zero modelled daytime population get no weight, exactly as in the Harvard
rebuild, so the footprint stops being paid for being large.

Usage: uv run python scripts/review/overlap/reweight_osu_footprint_landscan.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crimerisk.paths import get_paths

OSU_ORI = "OH0252700"
LANDSCAN_BG = Path("data/LandScan-USA/block_group_landscan_usa_2021.parquet")
GEOMETRY_SOURCE_TYPE = (
    "official_osu_campusmap_buildings_and_parking_layers_with_landscan_daytime_population"
)
NOTE_SUFFIX = (
    " REWEIGHTED (Stage 2 fork ruling 4, Stage 4/5 batch): weight = LandScan USA 2021 DAYTIME"
    " population (residents+workers+students+shoppers) attributed to the footprint's area inside"
    " each block group, i.e. the block group's LandScan day population times the footprint's"
    " share of that block group's 2020 TIGER land area. This replaces the pure land-area basis,"
    " which gave 47.5% of OSUPD's mass to block group 390490011221 (LandScan day 17,867) and"
    " 10.0% to 390490011212 (LandScan day 37,537). Block-group pieces with zero modelled daytime"
    " population get no weight, so athletics, parking and west-campus acreage is not paid for"
    " being large. LIMITATION: the OSU footprint polygon is not stored in this repo, so LandScan"
    " is applied at block-group resolution under a uniform-density assumption inside the block"
    " group rather than summed over raster cells inside (polygon x block group) as for Harvard."
)


def reweight(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_path = repo_root / "configs" / "overlap_custom_footprints.csv"
    footprints = pd.read_csv(config_path, dtype=str)
    numeric = pd.to_numeric(footprints["weight_share"], errors="coerce")
    osu = footprints["ori"].astype(str).eq(OSU_ORI)
    if not bool(osu.any()):
        raise ValueError(f"{OSU_ORI} has no rows in {config_path}")

    landscan = pd.read_parquet(repo_root / LANDSCAN_BG)
    landscan["bg"] = landscan["block_group_geoid"].astype(str).str.zfill(12)
    landscan = landscan.set_index("bg")["landscan_day_pop"]

    crosswalk = pd.read_parquet(
        repo_root / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        columns=["block_group_geoid", "total_aland20"],
    )
    crosswalk["bg"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    land = crosswalk.groupby("bg")["total_aland20"].max()

    rows = footprints.loc[osu].copy()
    rows["bg"] = rows["block_group_geoid"].astype(str).str.zfill(12)
    rows["area_share"] = numeric.loc[osu].to_numpy()
    rows["landscan_day_pop"] = rows["bg"].map(landscan)
    rows["bg_land_area_m2"] = rows["bg"].map(land)
    missing = rows["landscan_day_pop"].isna() | rows["bg_land_area_m2"].isna() | rows["bg_land_area_m2"].le(0)
    if bool(missing.any()):
        raise ValueError(
            "OSU footprint block groups missing LandScan or land area: "
            f"{rows.loc[missing, 'bg'].tolist()}"
        )
    rows["landscan_day_in_footprint"] = (
        rows["area_share"] * rows["landscan_day_pop"] / rows["bg_land_area_m2"]
    )
    total = float(rows["landscan_day_in_footprint"].sum())
    if total <= 0:
        raise ValueError("OSU footprint carries no modelled LandScan daytime population")
    rows["new_weight_share"] = rows["landscan_day_in_footprint"] / total
    # Absorb float residue into the largest weight so the loader's sum-to-1 assertion holds.
    largest = rows["new_weight_share"].idxmax()
    rows.loc[largest, "new_weight_share"] += 1.0 - float(rows["new_weight_share"].sum())

    updated = footprints.copy()
    updated.loc[osu, "weight_share"] = [f"{v:.15g}" for v in rows["new_weight_share"]]
    updated.loc[osu, "geometry_source_type"] = GEOMETRY_SOURCE_TYPE
    updated.loc[osu, "weight_share_basis"] = "activity_or_area"
    base_note = str(rows["footprint_note"].iloc[0])
    updated.loc[osu, "footprint_note"] = base_note.split(" REWEIGHTED (Stage 2 fork ruling 4")[0] + NOTE_SUFFIX
    return updated, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = get_paths()
    updated, rows = reweight(paths.repo_root)
    pd.set_option("display.width", 200)
    print(
        rows[["bg", "area_share", "landscan_day_pop", "new_weight_share"]]
        .sort_values("new_weight_share", ascending=False)
        .to_string(index=False)
    )
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    out = paths.repo_root / "configs" / "overlap_custom_footprints.csv"
    updated.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
