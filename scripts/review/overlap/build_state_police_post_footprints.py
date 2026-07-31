"""Generate state-police post/troop/district footprint rows for the overlap registries.

Stage 2 fix S2-4. `_state_police_county_subunit_mask` anchors a state-police post ORI to the
single county its post BUILDING sits in, so KY Post 13 (Hazard) put 100% of its mass in Perry
County when its published district is Breathitt/Knott/Leslie/Letcher/Perry -- Knott read
30/100k next to Perry at 947.

Inputs
------
`configs/state_police_post_county_footprints.csv`
    The transcribed evidence table: one row per ORI, the counties that post/district/troop
    serves, the official source URL, and the disposition. Built from
    `state/qa/stage2_screen/post_directories/` (official state directories, transcribed and
    escalated where the state does not publish at the granularity our ORIs need -- WV is at
    DISTRICT granularity per the supervisor ruling).

`state/geometry/blocks_by_state/<FF>.parquet`
    2020 blocks with `pop20` and the jurisdiction each block was assigned to. No spatial
    operations: the block -> jurisdiction assignment already exists.

Weights
-------
`weight_share` is proportional to each block group's **non-municipal 2020 population** inside
the served counties -- the ground no agency-bearing municipality covers, which is what a state
police post actually polices. It is NOT "unincorporated": in New England a town with no police
department is state-police primary territory and lands in the non-municipal remainder, which is
exactly the behaviour we want.

Where a served county has no non-municipal population at all (Virginia independent cities,
Rhode Island, Baltimore city), that county falls back to total population so the county is not
silently dropped from its own post's footprint. Every fallback is recorded in the row note.

Outputs (written to a staging directory, then merged into configs/ by the caller)
    <out_dir>/overlap_footprint_overrides.add.csv
    <out_dir>/overlap_custom_footprints.add.csv
    <out_dir>/state_police_post_footprint_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crimerisk.paths import get_paths

MULTI_COUNTY_DISPOSITION = "post_multi_county_footprint"
GEOMETRY_SOURCE_TYPE = "official_state_police_post_directory_plus_2020_block_assignment"


def _load_evidence(path: Path) -> pd.DataFrame:
    evidence = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "ori",
        "state_fips",
        "agency_name_std",
        "disposition",
        "granularity",
        "counties_served_fips",
        "counties_served_names",
        "current_anchor_county_fips",
        "directory_row",
        "source_url",
        "evidence_note",
    }
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"state police post evidence missing columns: {sorted(missing)}")
    evidence["state_fips"] = evidence["state_fips"].str.zfill(2)
    dupes = evidence.loc[evidence.duplicated("ori", keep=False), "ori"].tolist()
    if dupes:
        raise ValueError(f"duplicate ORIs in state police post evidence: {sorted(set(dupes))}")
    return evidence


def _block_population_by_bg(paths: Path, state_fips: str) -> pd.DataFrame:
    """Per (county, block group): total and non-municipal 2020 population."""
    blocks = pd.read_parquet(
        paths / "state" / "geometry" / "blocks_by_state" / f"{state_fips}.parquet",
        columns=["state_fips", "county_fips", "block_group_geoid", "pop20", "jurisdiction_type"],
    )
    blocks["county_geoid"] = (
        blocks["state_fips"].astype("string").str.zfill(2)
        + blocks["county_fips"].astype("string").str.zfill(3)
    )
    blocks["pop20"] = pd.to_numeric(blocks["pop20"], errors="coerce").fillna(0.0)
    blocks["nonmunicipal_pop20"] = blocks["pop20"].where(
        blocks["jurisdiction_type"].astype("string").ne("municipal"), 0.0
    )
    out = (
        blocks.groupby(["county_geoid", "block_group_geoid"], dropna=False, as_index=False)
        .agg(pop20=("pop20", "sum"), nonmunicipal_pop20=("nonmunicipal_pop20", "sum"))
    )
    out["state_fips"] = state_fips
    return out


def build(evidence_path: Path, out_dir: Path) -> dict[str, int]:
    paths = get_paths()
    evidence = _load_evidence(evidence_path)
    footprint_rows = evidence[evidence["disposition"].eq(MULTI_COUNTY_DISPOSITION)].copy()
    if footprint_rows.empty:
        raise ValueError("no post_multi_county_footprint rows in the evidence table")

    bg_by_state = {
        state: _block_population_by_bg(paths.repo_root, state)
        for state in sorted(footprint_rows["state_fips"].unique())
    }

    override_records: list[dict[str, object]] = []
    footprint_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for row in footprint_rows.itertuples(index=False):
        state_fips = str(row.state_fips).zfill(2)
        counties = [c for c in str(row.counties_served_fips).split(";") if c]
        if len(counties) < 1:
            raise ValueError(f"{row.ori}: multi-county disposition with no counties")
        bgs = bg_by_state[state_fips]
        served = bgs[bgs["county_geoid"].isin(counties)].copy()
        found_counties = sorted(served["county_geoid"].unique())
        if set(found_counties) != set(counties):
            raise ValueError(
                f"{row.ori}: counties with no block coverage: {sorted(set(counties) - set(found_counties))}"
            )

        # Per served county, use non-municipal population; fall back to total population for a
        # county that has none (otherwise the county silently leaves its own post's footprint).
        county_nonmunicipal = served.groupby("county_geoid")["nonmunicipal_pop20"].sum()
        fallback_counties = sorted(county_nonmunicipal[county_nonmunicipal.le(0.0)].index)
        served["exposure_pop"] = served["nonmunicipal_pop20"]
        if fallback_counties:
            fallback = served["county_geoid"].isin(fallback_counties)
            served.loc[fallback, "exposure_pop"] = served.loc[fallback, "pop20"]

        served = served[served["exposure_pop"].gt(0.0)].copy()
        total = float(served["exposure_pop"].sum())
        if total <= 0.0:
            raise ValueError(f"{row.ori}: served counties carry no 2020 population at all")
        served = served.sort_values("block_group_geoid", kind="mergesort").reset_index(drop=True)
        served["weight_share"] = served["exposure_pop"] / total
        # Absorb float residual into the largest row so the loader's 1e-6 sum check passes
        # without depending on how many rows there are.
        largest = int(served["weight_share"].idxmax())
        served.loc[largest, "weight_share"] += 1.0 - float(served["weight_share"].sum())

        basis = "nonmunicipal_pop20"
        note = (
            f"State-police {row.granularity} footprint for {row.agency_name_std}: "
            f"{len(counties)} served counties ({row.counties_served_names}) per "
            f"{row.directory_row or 'the state directory'}. weight_share proportional to "
            "2020 non-municipal block-group population (2020 block -> jurisdiction assignment, "
            "no spatial ops)."
        )
        if fallback_counties:
            basis = "nonmunicipal_pop20_with_total_pop_fallback"
            note += (
                " Counties with zero non-municipal population fell back to total population: "
                f"{';'.join(fallback_counties)}."
            )

        for bg_row in served.itertuples(index=False):
            footprint_records.append(
                {
                    "ori": row.ori,
                    "state_fips": state_fips,
                    "block_group_geoid": str(bg_row.block_group_geoid).zfill(12),
                    "weight_share": f"{float(bg_row.weight_share):.15g}",
                    "bg_population_coverage_share": "",
                    # weight_share IS a 2020 non-municipal resident-population share, so the
                    # allocator applies the same activity basis the county lanes use.
                    "weight_share_basis": "resident_population",
                    "geometry_source_type": GEOMETRY_SOURCE_TYPE,
                    "geometry_source_ref": row.source_url,
                    "footprint_note": note,
                }
            )

        override_records.append(
            {
                "ori": row.ori,
                "final_overlap_treatment": "localize_to_custom_footprint",
                "overlap_subtype_final": "other_special",
                "footprint_type": f"state_police_{row.granularity}_multi_county_nonmunicipal_footprint",
                "target_state_fips": state_fips,
                "target_county_fips": "",
                "target_place_fips": "",
                "target_jurisdiction_id": "",
                # State police are a genuine concurrent-jurisdiction overlap, not the primary
                # agency on non-municipal ground (the sheriff is), so this footprint is
                # additive and does NOT displace the county remainder.
                "displaces_county_remainder": "",
                "geometry_source_type": GEOMETRY_SOURCE_TYPE,
                "geometry_source_ref": row.source_url,
                "confidence": "0.90" if row.granularity == "post" else "0.80",
                "source_note": (
                    f"{row.directory_row or row.agency_name_std} serves "
                    f"{row.counties_served_names} ({row.source_url})."
                ),
                "reviewer_note": (
                    f"{row.evidence_note} Replaces the single-county post-building anchor "
                    f"({row.current_anchor_county_fips or 'none'}) with the published "
                    f"{row.granularity} footprint, restricted to non-municipal exposure."
                ),
            }
        )

        summary_records.append(
            {
                "ori": row.ori,
                "state_fips": state_fips,
                "agency_name_std": row.agency_name_std,
                "granularity": row.granularity,
                "n_counties": len(counties),
                "n_block_groups": int(len(served)),
                "weight_basis": basis,
                "fallback_counties": ";".join(fallback_counties),
                "footprint_pop20": float(served["pop20"].sum()),
                "footprint_nonmunicipal_pop20": float(served["nonmunicipal_pop20"].sum()),
                "anchor_county_weight_share": float(
                    served.loc[
                        served["county_geoid"].eq(str(row.current_anchor_county_fips)),
                        "weight_share",
                    ].sum()
                ),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(override_records).to_csv(out_dir / "overlap_footprint_overrides.add.csv", index=False)
    pd.DataFrame(footprint_records).to_csv(out_dir / "overlap_custom_footprints.add.csv", index=False)
    pd.DataFrame(summary_records).to_csv(out_dir / "state_police_post_footprint_summary.csv", index=False)
    return {
        "oris": len(override_records),
        "footprint_rows": len(footprint_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=get_paths().repo_root / "configs" / "state_police_post_county_footprints.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=get_paths().repo_root / "analysis_scratch" / "stage2_fix_batch" / "generated",
    )
    args = parser.parse_args()
    stats = build(args.evidence, args.out_dir)
    print(f"state police post footprints: {stats['oris']} ORIs, {stats['footprint_rows']} block-group rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
