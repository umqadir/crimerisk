from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.fbi_publications import norm_municipal_name
from crimerisk.source_selection import _globally_dead_observation_oris


# Evidence-based municipal relink rule for remainder-routed agencies.
#
# A remainder-routed agency (sole exclusive crosswalk row pointing at its
# state_nonmunicipal_remainder) is relinked municipal-exclusive to a place
# jurisdiction when ALL of the following hold:
#
# (a) independent evidence identifies the place as the agency's service area:
#     the FBI census linkage (agency_master place_fips, excluding the 99xxx
#     pseudo codes used for non-place county-equivalents) resolves to the
#     place, OR the agency's standardized name exactly matches the place
#     jurisdiction's standardized name uniquely within the state;
# (b) the agency's FBI service population is within [0.85, 1.20] of the
#     place population (2020 block population aggregated over the place's
#     current BG-assignment support -- a place with no BG support is not a
#     valid municipal target, the same normalization rule the overlap
#     allocator applies);
# (c) the place jurisdiction currently has no live totals source: every ORI
#     the crosswalk links to it is globally dead, entirely absent from the
#     observations panel, or carries no positive FBI service population.
#
# Carson City NV (NV0130000) is the flagship case: the sheriff's office is
# the consolidated municipality's primary law-enforcement surface but was
# auto-routed to the state remainder, leaving the Carson City place polygon
# sourced only from a defunct city PD ORI and a zero-population tribal ORI.
#
# This script is a review-time diagnostic only: it FLAGS candidates into the
# review queue. Applying a relink is a hand-authored override row in
# configs/local_resolution_overrides.csv (consumed by
# _apply_nonlocal_resolution_overrides at the reference stage), written with
# recorded qualitative evidence; ORIs already present there are excluded from
# detection.

SERVICE_POP_RATIO_MIN = 0.85
SERVICE_POP_RATIO_MAX = 1.20
REMAINDER_SUFFIX = ":state_nonmunicipal_remainder"
OVERRIDES_PATH = Path("configs/local_resolution_overrides.csv")


def _load_agency_service_population(obs: pd.DataFrame) -> pd.Series:
    """Latest positive FBI-reported service population per ORI."""
    pop = obs[["ori9", "year", "population"]].copy()
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop = pop[pop["population"].gt(0)]
    if pop.empty:
        return pd.Series(dtype=float)
    pop = pop.sort_values(["ori9", "year"], kind="mergesort")
    latest_year = pop.groupby("ori9")["year"].transform("max")
    pop = pop[pop["year"].eq(latest_year)]
    return pop.groupby("ori9")["population"].max()


def _load_place_population(repo_root: Path) -> pd.Series:
    """2020 block population over each municipal jurisdiction's BG support."""
    blocks = pd.read_parquet(
        repo_root / "state" / "geometry" / "block_to_jurisdiction_crosswalk.parquet",
        columns=["jurisdiction_id", "jurisdiction_type", "pop20"],
    )
    blocks = blocks[blocks["jurisdiction_type"].astype("string").eq("municipal")]
    return blocks.groupby("jurisdiction_id")["pop20"].sum()


def build_relink_rows(*, repo_root: Path) -> pd.DataFrame:
    agency_master = pd.read_parquet(
        repo_root / "state" / "reference" / "agency_master.parquet",
        columns=[
            "ori9",
            "state_fips",
            "state_abbr",
            "agency_name_std",
            "agency_type_norm",
            "place_fips",
            "census_name_std",
        ],
    )
    crosswalk = pd.read_parquet(
        repo_root / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    )
    jurisdiction_master = pd.read_parquet(
        repo_root / "state" / "reference" / "jurisdiction_master.parquet"
    )
    obs = pd.read_parquet(
        repo_root / "state" / "observations" / "agency_year_observations.parquet",
        columns=["ori9", "year", "count", "months_reported", "population"],
    )

    dead_oris = _globally_dead_observation_oris(
        obs[["ori9", "count", "months_reported"]]
    )
    observed_oris = set(obs["ori9"].dropna().astype(str))
    service_pop = _load_agency_service_population(obs)
    place_pop = _load_place_population(repo_root)

    def _ori_is_live_totals_source(ori: str) -> bool:
        ori = str(ori)
        if ori not in observed_oris or ori in dead_oris:
            return False
        return float(service_pop.get(ori, 0.0)) > 0.0

    # Universe: agencies whose entire crosswalk footprint is a single exclusive
    # row on the state nonmunicipal remainder, not already overridden.
    cw = crosswalk.copy()
    cw["ori"] = cw["ori"].astype("string")
    row_counts = cw.groupby("ori")["jurisdiction_id"].size()
    remainder_rows = cw[
        cw["jurisdiction_id"].astype(str).str.endswith(REMAINDER_SUFFIX)
        & cw["relationship_type"].eq("exclusive")
    ]
    sole_remainder_oris = {
        str(ori)
        for ori in remainder_rows["ori"]
        if int(row_counts.get(ori, 0)) == 1
    }

    overrides_path = repo_root / OVERRIDES_PATH
    existing_overrides = (
        pd.read_csv(overrides_path) if overrides_path.exists() else pd.DataFrame(columns=["ori"])
    )
    already_overridden = set(existing_overrides["ori"].dropna().astype(str))

    candidates = agency_master[
        agency_master["ori9"].astype(str).isin(sole_remainder_oris)
        & ~agency_master["ori9"].astype(str).isin(already_overridden)
    ].copy()
    candidates["state_fips"] = candidates["state_fips"].astype("string").str.zfill(2)

    # Municipal place jurisdictions and their standardized names.
    places = jurisdiction_master[
        jurisdiction_master["jurisdiction_type"].astype("string").eq("municipal")
        & jurisdiction_master["geo_type"].astype("string").eq("place")
        & jurisdiction_master["geoid"].notna()
    ][["jurisdiction_id", "state_fips", "geoid", "jurisdiction_name"]].copy()
    places["state_fips"] = places["state_fips"].astype("string").str.zfill(2)
    places["geoid"] = places["geoid"].astype("string").str.zfill(7)
    places["name_std"] = places["jurisdiction_name"].map(norm_municipal_name)
    place_by_geoid = places.set_index("geoid")

    # Unique standardized name -> place within each state (ambiguous names drop out).
    name_counts = places.groupby(["state_fips", "name_std"])["geoid"].nunique()
    unique_names = name_counts[name_counts.eq(1)].index
    name_lookup = (
        places.set_index(["state_fips", "name_std"])
        .loc[places.set_index(["state_fips", "name_std"]).index.isin(unique_names)]
    )

    linked_by_jurisdiction = (
        cw.groupby("jurisdiction_id")["ori"].apply(lambda s: sorted(set(s.dropna().astype(str))))
    )

    records: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        ori = str(row.ori9)
        agency_pop = float(service_pop.get(ori, 0.0))
        if agency_pop <= 0.0:
            continue

        # Evidence (a): census place linkage, then exact name match.
        target_geoid: str | None = None
        evidence: str | None = None
        place_fips = None if pd.isna(row.place_fips) else str(row.place_fips).zfill(5)
        if place_fips and not place_fips.startswith("99"):
            geoid = str(row.state_fips) + place_fips
            if geoid in place_by_geoid.index:
                target_geoid = geoid
                evidence = f"fbi_census_place_linkage place_fips={place_fips}"
        if target_geoid is None:
            name_std = norm_municipal_name(row.agency_name_std)
            key = (str(row.state_fips), name_std)
            if name_std and key in name_lookup.index:
                match = name_lookup.loc[key]
                target_geoid = str(match["geoid"])
                evidence = f"exact_name_match name_std={name_std}"
        if target_geoid is None:
            continue

        place_row = place_by_geoid.loc[target_geoid]
        jurisdiction_id = str(place_row["jurisdiction_id"])

        # Evidence (b): service population within band of the place population.
        target_place_pop = float(place_pop.get(jurisdiction_id, 0.0))
        if target_place_pop <= 0.0:
            continue
        ratio = agency_pop / target_place_pop
        if not (SERVICE_POP_RATIO_MIN <= ratio <= SERVICE_POP_RATIO_MAX):
            continue

        # Evidence (c): the place currently has no live totals source.
        linked = linked_by_jurisdiction.get(jurisdiction_id, [])
        if any(_ori_is_live_totals_source(l) for l in linked):
            continue

        records.append(
            {
                "ori": ori,
                "state_abbr": row.state_abbr,
                "agency_name_std": row.agency_name_std,
                "agency_type_norm": row.agency_type_norm,
                "jurisdiction_id": jurisdiction_id,
                "replacement_geoid": target_geoid,
                "replacement_jurisdiction_name": place_row["jurisdiction_name"],
                "evidence": evidence,
                "agency_service_population": agency_pop,
                "place_population": target_place_pop,
                "population_ratio": round(ratio, 4),
                "linked_oris_without_live_source": ",".join(linked),
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Flag remainder-routed agencies with evidence for a municipal-exclusive "
            "relink into the local-resolution review queue."
        )
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("state/review/queues/local_resolution/current/evidence_based_municipal_relink"),
    )
    args = parser.parse_args()

    detected = build_relink_rows(repo_root=REPO_ROOT)
    out_base = REPO_ROOT / args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    detected.to_parquet(out_base.with_suffix(".parquet"), index=False)
    detected.to_csv(out_base.with_suffix(".csv"), index=False)

    print(
        {
            "detected": int(len(detected)),
            "oris": detected["ori"].tolist() if not detected.empty else [],
            "jurisdictions": detected["jurisdiction_id"].tolist() if not detected.empty else [],
            "out_base": str(out_base),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
