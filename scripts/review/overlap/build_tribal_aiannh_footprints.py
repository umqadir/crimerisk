"""Generate AIANNH footprint rows for tribal law-enforcement agencies (Stage 2 Class D).

Mechanism being fixed
---------------------
LEAIC hands a tribal police department the `FPLACE` of its agency-seat CDP. The automatic local
lane takes any valid `place_fips` unconditionally, before the tribal branch of
`_infer_special_bucket` is reachable, so a whole reservation's crime lands on one small town --
Colville -> Nespelem town at 39x the national rate, Rocky Boy's -> Box Elder CDP at 52x -- and,
in the other direction, a reservation inside a city dilutes invisibly (Seminole Tribal FL -> the
whole of Hollywood city; Puyallup -> Tacoma).

Semantics: EXCLUSIVE, per Pro's Class D correction (docs/STATE.md, v20 program). A tribal PD is
the PRIMARY agency on its reservation, so the footprint DISPLACES the county remainder on that
ground rather than adding to it. Additive overlap for a primary agency double-counts and breaks
conservation. That is what `displaces_county_remainder` and `bg_population_coverage_share` carry.

`bg_population_coverage_share` is the share of a block group's 2020 population inside the UNION
of all displacing footprints, written identically on every displacing row that touches the block
group -- so production can take the max and get exact set semantics without knowing which
footprints overlap (24 footprints are shared by two ORIs each: duplicate tribal/BIA ORI pairs
and joint Oklahoma OTSAs).

Inputs
------
`configs/tribal_agency_aiannh_footprints.csv`
    Reviewed ORI -> AIANNH mapping: match_status, aiannh_codes, comptyp_included, match_basis,
    reviewer_note. Built against the 2020 TIGER AIANNH vocabulary.
`data/Census-BAF-2020/aiannh/BlockAssign_ST<FF>_<SS>_AIANNH.txt`
    2020 Census Block Assignment Files: BLOCKID | AIANNHCE | COMPTYP. No spatial operations.
`state/geometry/blocks_by_state/<FF>.parquet`
    2020 blocks with pop20/housing20/aland20 and their block group.

Outputs (staged; merged into configs/ by the caller)
    <out_dir>/tribal_local_resolution_overrides.add.csv
    <out_dir>/tribal_overlap_footprint_overrides.add.csv
    <out_dir>/tribal_overlap_custom_footprints.add.csv
    <out_dir>/tribal_footprint_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crimerisk.paths import get_paths

FOOTPRINT_STATUSES = ("matched", "matched_multi")
GEOMETRY_SOURCE_TYPE = "census_2020_block_assignment_file_aiannh"
GEOMETRY_SOURCE_REF = (

# The allocator needs to know what `weight_share` MEASURES, not which column produced it.
# A population or housing-unit share is a resident-exposure apportionment and is size-blind on
# its own, so the custom-footprint lane applies the same activity basis the county lanes use.
# A land-area share on a zero-resident reservation is an area apportionment, like an airfield,
# and is used verbatim. See VALID_CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASES in allocation.py.
WEIGHT_SHARE_BASIS_BY_BLOCK_BASIS = {
    "pop20": "resident_population",
    "housing20": "resident_population",
    "aland20": "activity_or_area",
}
    "https://www2.census.gov/geo/docs/maps-data/data/baf2020/ | "
    "https://www2.census.gov/geo/tiger/TIGER2020/AIANNH/tl_2020_us_aiannh.zip"
)
STATE_ABBR_BY_FIPS_PATH = "data/Census-BAF-2020/aiannh"


def _baf_path(repo_root: Path, state_fips: str, state_abbr: str) -> Path:
    return repo_root / STATE_ABBR_BY_FIPS_PATH / f"BlockAssign_ST{state_fips}_{state_abbr}_AIANNH.txt"


def _load_state_blocks(repo_root: Path, state_fips: str) -> pd.DataFrame:
    blocks = pd.read_parquet(
        repo_root / "state" / "geometry" / "blocks_by_state" / f"{state_fips}.parquet",
        columns=["block_geoid", "block_group_geoid", "pop20", "housing20", "aland20"],
    )
    for column in ("pop20", "housing20", "aland20"):
        blocks[column] = pd.to_numeric(blocks[column], errors="coerce").fillna(0.0)
    return blocks


def _load_state_baf(repo_root: Path, state_fips: str, state_abbr: str) -> pd.DataFrame:
    path = _baf_path(repo_root, state_fips, state_abbr)
    if not path.exists():
        return pd.DataFrame(columns=["block_geoid", "aiannhce", "comptyp"])
    baf = pd.read_csv(path, sep="|", dtype=str)
    baf.columns = [c.strip().lower() for c in baf.columns]
    return baf.rename(columns={"blockid": "block_geoid"})[["block_geoid", "aiannhce", "comptyp"]]


def build(mapping_path: Path, out_dir: Path, *, target_oris: set[str] | None = None) -> dict[str, int]:
    paths = get_paths()
    repo_root = paths.repo_root
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    required = {
        "ori",
        "state_fips",
        "state_abbr",
        "agency_name_std",
        "match_status",
        "aiannh_codes",
        "aiannh_names",
        "comptyp_included",
        "match_basis",
        "reviewer_note",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"tribal AIANNH mapping missing columns: {sorted(missing)}")
    mapping["state_fips"] = mapping["state_fips"].str.zfill(2)
    # `not_tribal` rows are the reviewed false positives of the audit's looser detector (Indian
    # River County FL, five California police-protection districts that carry LEAIC's
    # LG_POPULATION sentinel, BIA headquarters). They are not tribal agencies and must not be
    # rerouted or pinned -- the production `is_tribal_agency` flag already excludes them.
    mapping = mapping[mapping["match_status"].ne("not_tribal")].copy()
    if target_oris is not None:
        mapping = mapping[mapping["ori"].isin(target_oris)].copy()

    crosswalk = pd.read_parquet(
        repo_root / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        columns=["ori", "state_fips", "jurisdiction_id", "relationship_type"],
    )
    municipal_links = crosswalk[
        crosswalk["relationship_type"].astype("string").eq("exclusive")
        & crosswalk["jurisdiction_id"].astype("string").str.contains(":municipal:", na=False)
    ][["ori", "jurisdiction_id"]].rename(columns={"jurisdiction_id": "current_jurisdiction_id"})
    mapping = mapping.merge(municipal_links, on="ori", how="left")

    baf_cache: dict[str, pd.DataFrame] = {}
    block_cache: dict[str, pd.DataFrame] = {}

    def _footprint_blocks(state_fips: str, state_abbr: str, codes: list[str], comptyps: list[str]) -> pd.DataFrame:
        if state_fips not in baf_cache:
            baf_cache[state_fips] = _load_state_baf(repo_root, state_fips, state_abbr)
            block_cache[state_fips] = _load_state_blocks(repo_root, state_fips)
        baf = baf_cache[state_fips]
        if baf.empty:
            return pd.DataFrame(columns=["block_group_geoid", "pop20", "housing20", "aland20"])
        selected = baf[baf["aiannhce"].isin(codes)]
        if comptyps:
            selected = selected[selected["comptyp"].isin(comptyps)]
        if selected.empty:
            return pd.DataFrame(columns=["block_group_geoid", "pop20", "housing20", "aland20"])
        return selected.merge(block_cache[state_fips], on="block_geoid", how="inner")

    footprints: dict[str, pd.DataFrame] = {}
    weight_basis: dict[str, str] = {}
    is_otsa: dict[str, bool] = {}
    for row in mapping.itertuples(index=False):
        if row.match_status not in FOOTPRINT_STATUSES:
            continue
        is_otsa[row.ori] = "OTSA" in str(row.aiannh_names).upper()
        codes = [c.strip() for c in str(row.aiannh_codes).split(";") if c.strip()]
        comptyps = [c.strip() for c in str(row.comptyp_included).split(";") if c.strip()]
        if not codes:
            continue
        blocks = _footprint_blocks(row.state_fips, row.state_abbr, codes, comptyps)
        if blocks.empty:
            continue
        basis = "pop20"
        if float(blocks["pop20"].sum()) <= 0.0:
            # Reservations with zero 2020 residents (Narragansett RI, Snoqualmie WA) cannot be
            # population-weighted; fall back to housing units, then land area, and record it.
            basis = "housing20" if float(blocks["housing20"].sum()) > 0 else "aland20"
        if float(blocks[basis].sum()) <= 0.0:
            continue
        footprints[row.ori] = blocks
        weight_basis[row.ori] = basis

    # Union coverage of every DISPLACING footprint, per block group, on a population basis.
    #
    # Oklahoma OTSAs are excluded from displacement. An OTSA is a Census STATISTICAL area, not a
    # reservation -- eastern Oklahoma is almost entirely OTSA, they overlap each other, and they
    # are not a verified exclusive policing jurisdiction (Pro: "AIANNH is a defensible fallback
    # footprint, not verified jurisdiction; validate per case"). Displacing the county remainder
    # across the Creek OTSA's 813,184 residents on the strength of a 0-count tribal ORI would be
    # a far larger error than the one being fixed, so OTSA footprints stay ADDITIVE overlap and
    # carry the caveat in their reviewer_note.
    displacing_frames = [
        frame[["block_geoid", "block_group_geoid", "pop20"]]
        for ori, frame in footprints.items()
        if not is_otsa.get(ori, False)
    ]
    union_blocks = (
        pd.concat(displacing_frames, ignore_index=True).drop_duplicates("block_geoid")
        if displacing_frames
        else pd.DataFrame(columns=["block_geoid", "block_group_geoid", "pop20"])
    )
    covered_pop = (
        union_blocks.groupby("block_group_geoid", dropna=False)["pop20"].sum().rename("covered_pop20")
    )
    bg_total_pop: dict[str, float] = {}
    for state_fips, blocks in block_cache.items():
        totals = blocks.groupby("block_group_geoid", dropna=False)["pop20"].sum()
        bg_total_pop.update(totals.to_dict())
    # Only a POSITIVE coverage share is a displacement. A footprint that covers no residents
    # (Narragansett RI, Snoqualmie WA -- real reservations with zero 2020 population) takes over
    # no exposure, so it has nothing to displace and the column stays blank.
    coverage_share = {
        bg: min(1.0, float(value) / float(bg_total_pop.get(bg, 0.0)))
        for bg, value in covered_pop.items()
        if float(bg_total_pop.get(bg, 0.0)) > 0.0 and float(value) > 0.0
    }

    local_records: list[dict[str, object]] = []
    override_records: list[dict[str, object]] = []
    footprint_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for row in mapping.itertuples(index=False):
        on_municipal = bool(str(row.current_jurisdiction_id or "").strip())
        blocks = footprints.get(row.ori)
        if blocks is None and row.match_status == "ambiguous" and on_municipal:
            # BIA multi-tribe regional agencies (Miami, Pawnee, Northern Pueblos, Southern
            # Pueblos, Eastern/Western Nevada) serve many tribes whose areas we cannot attribute
            # individually. Leaving such an agency on its office city AFFIRMS a false claim about
            # where the crime happened; the statewide overlap says "somewhere in this state",
            # which is true. Declared explicitly rather than reached by omission.
            local_records.append(
                {
                    "ori": row.ori,
                    "decision": "reclassify_overlap",
                    "replacement_geo_type": "",
                    "replacement_geoid": "",
                    "replacement_jurisdiction_name": "",
                    "confidence": "medium",
                    "source_note": (
                        f"BIA multi-tribe regional agency; the automatic local lane had it on "
                        f"{row.current_jurisdiction_id} via its office city place_fips. {row.match_basis}"
                    ),
                    "reviewer_note": (
                        "Class D: routed off the office-city municipality. No single AIANNH area "
                        "can be attributed to a multi-tribe BIA agency, so the footprint is the "
                        f"declared statewide overlap. {row.reviewer_note}"
                    ),
                }
            )
            override_records.append(
                {
                    "ori": row.ori,
                    "final_overlap_treatment": "keep_statewide_overlap",
                    "overlap_subtype_final": "tribal",
                    "footprint_type": "statewide",
                    "target_state_fips": row.state_fips,
                    "target_county_fips": "",
                    "target_place_fips": "",
                    "target_jurisdiction_id": "",
                    "displaces_county_remainder": "",
                    "geometry_source_type": "reviewed_no_attributable_aiannh_area",
                    "geometry_source_ref": GEOMETRY_SOURCE_REF,
                    "confidence": "0.60",
                    "source_note": row.match_basis,
                    "reviewer_note": (
                        "DECLARED statewide overlap: multi-tribe BIA regional agency with no "
                        f"attributable single AIANNH footprint. {row.reviewer_note}"
                    ),
                }
            )
            summary_records.append(
                {
                    "ori": row.ori,
                    "state_abbr": row.state_abbr,
                    "agency_name_std": row.agency_name_std,
                    "match_status": row.match_status,
                    "disposition": "declared_statewide_overlap",
                    "n_block_groups": 0,
                    "footprint_pop20": 0.0,
                    "weight_basis": "",
                    "displaces_county_remainder": False,
                    "current_jurisdiction_id": row.current_jurisdiction_id,
                }
            )
            continue
        if blocks is None:
            # No defensible AIANNH footprint. If the automatic lane put this agency on a
            # municipality, PIN that placement explicitly so the Class D gate has a reviewed
            # decision to point at rather than failing the build on an unanswerable case.
            if on_municipal:
                geo_type, geoid = str(row.current_jurisdiction_id).split(":")[2:4]
                local_records.append(
                    {
                        "ori": row.ori,
                        "decision": "municipal_place" if geo_type == "place" else "municipal_cousub",
                        "replacement_geo_type": geo_type,
                        "replacement_geoid": geoid,
                        "replacement_jurisdiction_name": "",
                        "confidence": "low",
                        "source_note": (
                            f"Tribal agency with no usable 2020 AIANNH footprint "
                            f"(match_status={row.match_status}). {row.match_basis}"
                        ),
                        "reviewer_note": (
                            "DOCUMENTED LIMITATION: the existing agency-seat placement is "
                            "retained because no AIANNH area can be attributed to this agency "
                            f"from the 2020 vocabulary. {row.reviewer_note}"
                        ),
                    }
                )
                summary_records.append(
                    {
                        "ori": row.ori,
                        "state_abbr": row.state_abbr,
                        "agency_name_std": row.agency_name_std,
                        "match_status": row.match_status,
                        "disposition": "municipal_pin_no_footprint",
                        "n_block_groups": 0,
                        "footprint_pop20": 0.0,
                        "weight_basis": "",
                        "current_jurisdiction_id": row.current_jurisdiction_id,
                    }
                )
            continue

        basis = weight_basis[row.ori]
        by_bg = (
            blocks.groupby("block_group_geoid", dropna=False)
            .agg(weight=(basis, "sum"), pop20=("pop20", "sum"))
            .reset_index()
        )
        by_bg = by_bg[by_bg["weight"].gt(0.0)].sort_values("block_group_geoid", kind="mergesort")
        if by_bg.empty:
            continue
        total = float(by_bg["weight"].sum())
        by_bg["weight_share"] = by_bg["weight"] / total
        by_bg = by_bg.reset_index(drop=True)
        largest = int(by_bg["weight_share"].idxmax())
        by_bg.loc[largest, "weight_share"] += 1.0 - float(by_bg["weight_share"].sum())

        note = (
            f"AIANNH footprint for {row.agency_name_std}: areas {row.aiannh_names} "
            f"(AIANNHCE {row.aiannh_codes}, COMPTYP {row.comptyp_included}), 2020 Block "
            f"Assignment File joined to the 2020 block table, weights proportional to {basis}. "
            f"{row.match_basis}"
        )
        for bg_row in by_bg.itertuples(index=False):
            bg = str(bg_row.block_group_geoid).zfill(12)
            share = None if is_otsa.get(row.ori, False) else coverage_share.get(bg)
            footprint_records.append(
                {
                    "ori": row.ori,
                    "state_fips": row.state_fips,
                    "block_group_geoid": bg,
                    "weight_share": f"{float(bg_row.weight_share):.15g}",
                    "bg_population_coverage_share": "" if share is None else f"{float(share):.15g}",
                    # weight_share IS a 2020 resident-population share, so the allocator
                    # applies the same activity basis the county lanes use.
                    "weight_share_basis": WEIGHT_SHARE_BASIS_BY_BLOCK_BASIS[basis],
                    "geometry_source_type": GEOMETRY_SOURCE_TYPE,
                    "geometry_source_ref": GEOMETRY_SOURCE_REF,
                    "footprint_note": note,
                }
            )

        # A footprint whose block groups are all unpopulated cannot displace anything (there is
        # no resident exposure to take over), and an OTSA must not displace at all (see the union
        # comment above), so either way it stays additive rather than exclusive.
        displaces = (not is_otsa.get(row.ori, False)) and any(
            coverage_share.get(str(bg).zfill(12)) is not None
            for bg in by_bg["block_group_geoid"]
        )
        override_records.append(
            {
                "ori": row.ori,
                "final_overlap_treatment": "localize_to_custom_footprint",
                "overlap_subtype_final": "tribal",
                "footprint_type": "aiannh_reservation_and_trust_land_footprint",
                "target_state_fips": row.state_fips,
                "target_county_fips": "",
                "target_place_fips": "",
                "target_jurisdiction_id": "",
                "displaces_county_remainder": "TRUE" if displaces else "",
                "geometry_source_type": GEOMETRY_SOURCE_TYPE,
                "geometry_source_ref": GEOMETRY_SOURCE_REF,
                "confidence": "0.85" if row.match_status == "matched" else "0.75",
                "source_note": (
                    f"{row.agency_name_std} polices {row.aiannh_names}; footprint built from the "
                    "2020 Census Block Assignment File for those AIANNH areas."
                ),
                "reviewer_note": (
                    (
                        "EXCLUSIVE tribal footprint (Pro Class D correction: a primary agency's "
                        "footprint displaces the county remainder, never adds to it). "
                        if displaces
                        else "ADDITIVE tribal footprint -- NOT exclusive. "
                        + (
                            "Oklahoma OTSA: a Census statistical area, not a reservation, so it "
                            "does not displace the county remainder and the footprint population "
                            "far exceeds any plausible service population. "
                            if is_otsa.get(row.ori, False)
                            else "The footprint carries no 2020 resident population to take over. "
                        )
                    )
                    + f"{row.reviewer_note}"
                ),
            }
        )
        if on_municipal:
            local_records.append(
                {
                    "ori": row.ori,
                    "decision": "reclassify_overlap",
                    "replacement_geo_type": "",
                    "replacement_geoid": "",
                    "replacement_jurisdiction_name": "",
                    "confidence": "high",
                    "source_note": (
                        "LEAIC/FBI-roster tribal agency; the automatic local lane had it on "
                        f"{row.current_jurisdiction_id} via its agency-seat place_fips."
                    ),
                    "reviewer_note": (
                        "Class D: routed off the agency-seat municipality onto its AIANNH "
                        f"footprint ({row.aiannh_names}). {row.match_basis}"
                    ),
                }
            )
        summary_records.append(
            {
                "ori": row.ori,
                "state_abbr": row.state_abbr,
                "agency_name_std": row.agency_name_std,
                "match_status": row.match_status,
                "disposition": "aiannh_footprint",
                "n_block_groups": int(len(by_bg)),
                "footprint_pop20": float(by_bg["pop20"].sum()),
                "weight_basis": basis,
                "displaces_county_remainder": bool(displaces),
                "current_jurisdiction_id": row.current_jurisdiction_id,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(local_records).to_csv(out_dir / "tribal_local_resolution_overrides.add.csv", index=False)
    pd.DataFrame(override_records).to_csv(out_dir / "tribal_overlap_footprint_overrides.add.csv", index=False)
    pd.DataFrame(footprint_records).to_csv(out_dir / "tribal_overlap_custom_footprints.add.csv", index=False)
    pd.DataFrame(summary_records).to_csv(out_dir / "tribal_footprint_summary.csv", index=False)
    return {
        "footprint_oris": len(override_records),
        "footprint_rows": len(footprint_records),
        "local_override_rows": len(local_records),
        "displacing_block_groups": len(coverage_share),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = get_paths().repo_root
    parser.add_argument(
        "--mapping",
        type=Path,
        default=repo_root / "configs" / "tribal_agency_aiannh_footprints.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "analysis_scratch" / "stage2_fix_batch" / "generated",
    )
    parser.add_argument(
        "--municipal-links-only",
        action="store_true",
        help="restrict to tribal ORIs whose crosswalk link is currently exclusive-municipal "
        "(the audited Class D defect population)",
    )
    args = parser.parse_args()
    target: set[str] | None = None
    if args.municipal_links_only:
        from crimerisk.reference import tribal_agency_flag

        crosswalk = pd.read_parquet(
            repo_root / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
            columns=["ori", "jurisdiction_id", "relationship_type"],
        )
        municipal = set(
            crosswalk.loc[
                crosswalk["relationship_type"].astype("string").eq("exclusive")
                & crosswalk["jurisdiction_id"].astype("string").str.contains(":municipal:", na=False),
                "ori",
            ].astype(str)
        )
        agency_master = pd.read_parquet(
            repo_root / "state" / "reference" / "agency_master.parquet",
            columns=["ori9", "agency_name_std"],
        )
        flag = tribal_agency_flag(
            agency_master["ori9"], agency_master["agency_name_std"], paths=get_paths()
        )
        # Intersect with the PRODUCTION tribal flag so the generator can never reroute an agency
        # the build's own Class D gate does not consider tribal.
        target = municipal & set(agency_master.loc[flag, "ori9"].astype(str))
    stats = build(args.mapping, args.out_dir, target_oris=target)
    print(
        f"tribal footprints: {stats['footprint_oris']} ORIs, {stats['footprint_rows']} block-group rows, "
        f"{stats['local_override_rows']} local-resolution rows, "
        f"{stats['displacing_block_groups']} block groups with displacement"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
