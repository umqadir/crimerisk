"""Build the reviewed Navajo Nation Police footprint across AZ, NM, and UT.

The output is a replacement slice for ``AZ0018900`` in
``configs/overlap_custom_footprints.csv``. Ramah Navajo is removed at Census-block
resolution because it has its own reporting ORI (``NM0170400``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import geopandas as gpd
import pandas as pd


STATES = {"04": "AZ", "35": "NM", "49": "UT"}
AIANNH_CODE = "2430"
RAMAH_AITSN_GEOID = "2430638"
ORI = "AZ0018900"


def _read_aiannh(zip_path: Path, state_fips: str, state_abbr: str) -> pd.DataFrame:
    member = f"BlockAssign_ST{state_fips}_{state_abbr}_AIANNH.txt"
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as stream:
        frame = pd.read_csv(stream, sep="|", dtype="string")
    frame.columns = [column.strip().lower() for column in frame.columns]
    return frame.rename(columns={"blockid": "block_geoid"})[
        ["block_geoid", "aiannhce", "comptyp"]
    ]


def _ramah_blocks(aitsn_path: Path, nm_tabblock_path: Path) -> set[str]:
    subdivisions = gpd.read_file(f"zip://{aitsn_path}")
    selected = subdivisions[subdivisions["GEOID"].astype("string").eq(RAMAH_AITSN_GEOID)]
    if len(selected) != 1:
        raise ValueError(f"Expected one Ramah AITSN polygon; found {len(selected)}")
    blocks = gpd.read_file(f"zip://{nm_tabblock_path}", columns=["GEOID20", "geometry"])
    polygon = selected.to_crs(blocks.crs).geometry.iloc[0]
    inside = blocks.geometry.representative_point().within(polygon)
    return set(blocks.loc[inside, "GEOID20"].astype("string"))


def build(
    *,
    baf_dir: Path,
    blocks_dir: Path,
    aitsn_path: Path,
    nm_tabblock_path: Path,
    existing_footprints_path: Path,
    output_path: Path,
    coverage_output_path: Path,
) -> pd.DataFrame:
    ramah = _ramah_blocks(aitsn_path, nm_tabblock_path)
    selected_frames: list[pd.DataFrame] = []
    bg_totals: list[pd.DataFrame] = []
    for state_fips, state_abbr in STATES.items():
        baf = _read_aiannh(
            baf_dir / f"BlockAssign_ST{state_fips}_{state_abbr}.zip",
            state_fips,
            state_abbr,
        )
        blocks = pd.read_parquet(
            blocks_dir / f"{state_fips}.parquet",
            columns=["block_geoid", "block_group_geoid", "pop20", "aland20"],
        )
        blocks["pop20"] = pd.to_numeric(blocks["pop20"], errors="coerce").fillna(0.0)
        blocks["aland20"] = pd.to_numeric(blocks["aland20"], errors="coerce").fillna(0.0)
        bg_totals.append(
            blocks.groupby("block_group_geoid", as_index=False).agg(
                bg_total_pop20=("pop20", "sum"),
                bg_total_land_m2=("aland20", "sum"),
            )
        )
        chosen = baf[baf["aiannhce"].eq(AIANNH_CODE) & baf["comptyp"].isin(["R", "T"])]
        chosen = chosen.merge(blocks, on="block_geoid", how="inner", validate="one_to_one")
        if state_fips == "35":
            chosen = chosen[~chosen["block_geoid"].isin(ramah)].copy()
        chosen["state_fips"] = state_fips
        selected_frames.append(chosen)

    selected = pd.concat(selected_frames, ignore_index=True)
    if selected.empty or float(selected["pop20"].sum()) <= 0:
        raise ValueError("Navajo service footprint has no resident population")
    by_bg = selected.groupby(["state_fips", "block_group_geoid"], as_index=False).agg(
        footprint_pop20=("pop20", "sum"),
        footprint_land_m2=("aland20", "sum"),
    )
    totals = pd.concat(bg_totals, ignore_index=True)
    by_bg = by_bg.merge(totals, on="block_group_geoid", how="left", validate="one_to_one")
    total_land = float(by_bg["footprint_land_m2"].sum())
    by_bg["weight_share"] = by_bg["footprint_land_m2"] / total_land
    by_bg["bg_service_population_coverage_share"] = (
        by_bg["footprint_pop20"] / by_bg["bg_total_pop20"].replace(0.0, pd.NA)
    ).fillna(0.0).clip(upper=1.0)
    by_bg["bg_land_area_coverage_share"] = (
        by_bg["footprint_land_m2"] / by_bg["bg_total_land_m2"].replace(0.0, pd.NA)
    ).clip(upper=1.0)
    if bool(by_bg["bg_land_area_coverage_share"].isna().any()):
        raise ValueError("Service footprint contains a block group with no Census land area")

    existing = pd.read_csv(existing_footprints_path, dtype="string")
    existing = existing.assign(
        block_group_geoid=lambda d: d["block_group_geoid"].str.zfill(12),
        declared_union=lambda d: pd.to_numeric(
            d["bg_population_coverage_share"], errors="coerce"
        ),
    )
    current_service_union = existing[existing["ori"].eq(ORI)][
        ["block_group_geoid", "declared_union"]
    ]
    other_union = (
        existing[~existing["ori"].eq(ORI)]
        .groupby("block_group_geoid", as_index=False)["declared_union"]
        .max()
        .rename(columns={"declared_union": "other_union"})
    )
    by_bg = by_bg.merge(current_service_union, on="block_group_geoid", how="left")
    by_bg = by_bg.merge(other_union, on="block_group_geoid", how="left")
    # This column is the already-reviewed UNION displacement fraction, not this service's own
    # population fraction. Preserve it for existing rows. New zero-resident sliver rows have
    # no service population to displace, and inherit only another footprint's declared union.
    by_bg["bg_population_coverage_share"] = by_bg["declared_union"].combine_first(
        by_bg["other_union"]
    ).fillna(0.0)

    largest = by_bg["weight_share"].idxmax()
    by_bg.loc[largest, "weight_share"] += 1.0 - float(by_bg["weight_share"].sum())
    source_ref = (
        "https://www2.census.gov/geo/docs/maps-data/data/baf2020/ | "
        "https://www2.census.gov/geo/tiger/TIGER2020/AITSN/tl_2020_us_aitsn.zip"
    )
    note = (
        "Service-wide Navajo Nation Police footprint: Navajo Nation Reservation and Trust Land "
        "(AIANNHCE 2430, COMPTYP R;T) in Arizona, New Mexico, and Utah, excluding Ramah Navajo "
        "Chapter (AITSN GEOID 2430638); allocation integrates the frozen block-group count "
        "prior over the within-block-group service land fraction."
    )
    output = pd.DataFrame(
        {
            "ori": ORI,
            "state_fips": by_bg["state_fips"],
            "block_group_geoid": by_bg["block_group_geoid"].astype("string").str.zfill(12),
            "weight_share": by_bg["weight_share"].map(lambda value: f"{float(value):.15g}"),
            "bg_population_coverage_share": by_bg["bg_population_coverage_share"].map(
                lambda value: f"{float(value):.15g}"
            ),
            "weight_share_basis": "service_area_prior",
            "geometry_source_type": "census_2020_baf_aiannh_excluding_aitsn",
            "geometry_source_ref": source_ref,
            "footprint_note": note,
        }
    ).sort_values(["state_fips", "block_group_geoid"], kind="mergesort")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    coverage = pd.DataFrame(
        {
            "service_scope_id": "navajo_nation_police",
            "state_fips": by_bg["state_fips"],
            "block_group_geoid": by_bg["block_group_geoid"].astype("string").str.zfill(12),
            "bg_service_population_coverage_share": by_bg[
                "bg_service_population_coverage_share"
            ].map(lambda value: f"{float(value):.15g}"),
            "bg_land_area_coverage_share": by_bg["bg_land_area_coverage_share"].map(
                lambda value: f"{float(value):.15g}"
            ),
            "coverage_basis": "census_2020_block_aiannh_land_fraction_excluding_ramah",
        }
    ).sort_values(["state_fips", "block_group_geoid"], kind="mergesort")
    coverage_output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_output_path, index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baf-dir", type=Path, required=True)
    parser.add_argument("--blocks-dir", type=Path, required=True)
    parser.add_argument("--aitsn-path", type=Path, required=True)
    parser.add_argument("--nm-tabblock-path", type=Path, required=True)
    parser.add_argument("--existing-footprints-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--coverage-output-path", type=Path, required=True)
    args = parser.parse_args()
    output = build(**vars(args))
    print(
        f"wrote {len(output)} rows; states={sorted(output['state_fips'].unique())}; "
        f"weight_sum={pd.to_numeric(output['weight_share']).sum():.15g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
