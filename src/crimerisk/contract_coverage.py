"""FBI ``covered_by_ori`` service relationships.

The covering ORI owns the caseload.  The covered ORI contributes its municipal
footprint, but no second numerator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.jurisdiction_reference import ReferenceArtifacts


def covered_by_source_path(paths) -> Path:
    state_dir = Path(getattr(paths, "state_dir", Path(paths.repo_root) / "state"))
    return state_dir / "cache" / "nibrs_batch_header" / "nibrs_batch_header_1991_2024.parquet"


def load_covered_by_relationships(paths, *, year: int) -> pd.DataFrame:
    path = covered_by_source_path(paths)
    if not path.exists():
        return pd.DataFrame(
            columns=["ori9", "covered_by_ori", "covering_ori", "covered_population", "covering_service_population"]
        )
    frame = pd.read_parquet(path, columns=["ori", "year", "population", "covered_by_ori"])
    frame = frame[pd.to_numeric(frame["year"], errors="coerce").eq(int(year))].copy()
    frame["ori9"] = frame["ori"].astype("string").str.strip().str.upper()
    frame["covered_by_ori"] = frame["covered_by_ori"].astype("string").str.strip().str.upper()
    frame["covered_population"] = pd.to_numeric(frame["population"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame = frame[
        frame["ori9"].str.fullmatch(r"[A-Z]{2}[A-Z0-9]{7}", na=False)
        & frame["covered_by_ori"].str.fullmatch(r"[A-Z]{2}[A-Z0-9]{7}", na=False)
        & frame["ori9"].ne(frame["covered_by_ori"])
    ].copy()
    frame = frame.sort_values(["ori9", "covered_population"], ascending=[True, False]).drop_duplicates("ori9")
    direct = dict(zip(frame["ori9"], frame["covered_by_ori"], strict=False))

    def root(ori: str) -> str:
        seen: set[str] = set()
        current = ori
        while current in direct and current not in seen:
            seen.add(current)
            current = direct[current]
        return current

    frame["covering_ori"] = frame["ori9"].map(root).astype("string")
    population = (
        pd.read_parquet(path, columns=["ori", "year", "population"])
        .loc[lambda d: pd.to_numeric(d["year"], errors="coerce").eq(int(year))]
        .assign(
            covering_ori=lambda d: d["ori"].astype("string").str.strip().str.upper(),
            covering_service_population=lambda d: pd.to_numeric(d["population"], errors="coerce").fillna(0.0).clip(lower=0.0),
        )
        .sort_values("covering_service_population", ascending=False)
        .drop_duplicates("covering_ori")[["covering_ori", "covering_service_population"]]
    )
    return frame[["ori9", "covered_by_ori", "covering_ori", "covered_population"]].merge(
        population, on="covering_ori", how="left"
    )


def apply_contract_coverage(artifacts: ReferenceArtifacts, relationships: pd.DataFrame) -> ReferenceArtifacts:
    if relationships.empty:
        return artifacts
    crosswalk = artifacts.agency_to_jurisdiction_crosswalk.copy()
    crosswalk["ori"] = crosswalk["ori"].astype("string").str.upper()
    municipal_ids = set(
        artifacts.jurisdiction_master.loc[
            artifacts.jurisdiction_master["jurisdiction_type"].astype("string").eq("municipal"),
            "jurisdiction_id",
        ].astype(str)
    )
    original = crosswalk.copy()
    by_ori = {ori: grp.copy() for ori, grp in original.groupby("ori", sort=False)}
    replacement_rows: list[pd.DataFrame] = []
    changed_coverers: set[str] = set()
    covered_oris: set[str] = set()

    for coverer, group in relationships.groupby("covering_ori", sort=False):
        coverer = str(coverer)
        own = by_ori.get(coverer)
        if own is None or own.empty:
            continue
        usable = []
        for rel in group.itertuples(index=False):
            footprint = by_ori.get(str(rel.ori9))
            if footprint is None or footprint.empty:
                continue
            footprint = footprint[footprint["jurisdiction_id"].astype(str).isin(municipal_ids)].copy()
            if footprint.empty:
                continue
            usable.append((rel, footprint))
        if not usable:
            continue
        changed_coverers.add(coverer)
        service_pop = float(pd.to_numeric(group["covering_service_population"], errors="coerce").max())
        covered_pop = sum(float(rel.covered_population) for rel, _ in usable)
        own_pop = max(0.0, service_pop - covered_pop)
        covered_sizes = {str(rel.ori9): float(rel.covered_population) for rel, _ in usable}
        if service_pop <= 0.0 or own_pop + covered_pop <= 0.0:
            own_pop = 1.0
            covered_sizes = {str(rel.ori9): 1.0 for rel, _ in usable}
            covered_pop = float(sum(covered_sizes.values()))
        total = own_pop + covered_pop
        own_weight = pd.to_numeric(own["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        own_sum = float(own_weight.sum())
        if own_sum > 0 and own_pop > 0:
            own = own.copy()
            own["weight"] = own_weight / own_sum * (own_pop / total)
            own["relationship_type"] = "contract_covering_footprint"
            own["review_status"] = "fbi_linkage"
            own["resolution_source"] = "nibrs_covered_by_ori"
            replacement_rows.append(own)
        for rel, footprint in usable:
            covered_ori = str(rel.ori9)
            covered_oris.add(covered_ori)
            footprint_weight = pd.to_numeric(footprint["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
            footprint_sum = float(footprint_weight.sum())
            if footprint_sum <= 0:
                continue
            links = footprint.copy()
            links["ori"] = coverer
            links["weight"] = footprint_weight / footprint_sum * (covered_sizes[covered_ori] / total)
            links["relationship_type"] = "contract_covered_footprint"
            links["review_status"] = "fbi_linkage"
            links["resolution_source"] = "nibrs_covered_by_ori"
            links["source_table"] = "fbi_covered_by"
            replacement_rows.append(links)

    if not changed_coverers:
        return artifacts
    keep = ~crosswalk["ori"].isin(changed_coverers | covered_oris)
    pieces = [crosswalk.loc[keep], *replacement_rows]
    covered_rows = original[original["ori"].isin(covered_oris)].copy()
    covered_rows["weight"] = 0.0
    covered_rows["relationship_type"] = "covered_by_other_agency"
    covered_rows["review_status"] = "fbi_linkage"
    covered_rows["resolution_source"] = "nibrs_covered_by_ori"
    covered_rows["source_table"] = "fbi_covered_by"
    pieces.append(covered_rows)
    crosswalk = pd.concat(pieces, ignore_index=True)
    keys = [c for c in crosswalk.columns if c != "weight"]
    crosswalk = crosswalk.groupby(keys, dropna=False, as_index=False)["weight"].sum()

    # A covered town can resolve to the same Census jurisdiction as the coverer's
    # own footprint (for example, a department and a nominally covered unit that
    # both resolve to one incorporated place).  The service partition still needs
    # both population shares, but the crosswalk contract is one row per
    # (ORI, jurisdiction).  Coalesce those coincident positive contract links so
    # downstream controls cannot see the numerator twice.
    contract_mask = crosswalk["relationship_type"].isin(
        ["contract_covering_footprint", "contract_covered_footprint"]
    )
    contract_rows = crosswalk.loc[contract_mask].copy()
    if contract_rows.duplicated(["ori", "jurisdiction_id"], keep=False).any():
        contract_rows["_combined_weight"] = contract_rows.groupby(
            ["ori", "jurisdiction_id"], dropna=False
        )["weight"].transform("sum")
        contract_rows["_covered_priority"] = contract_rows["relationship_type"].eq(
            "contract_covered_footprint"
        )
        contract_rows = (
            contract_rows.sort_values(
                ["ori", "jurisdiction_id", "_covered_priority", "weight"],
                ascending=[True, True, False, False],
                kind="mergesort",
            )
            .drop_duplicates(["ori", "jurisdiction_id"], keep="first")
            .assign(weight=lambda d: d["_combined_weight"])
            .drop(columns=["_combined_weight", "_covered_priority"])
        )
        crosswalk = pd.concat([crosswalk.loc[~contract_mask], contract_rows], ignore_index=True)
    sums = crosswalk[crosswalk["weight"].gt(0)].groupby("ori")["weight"].sum()
    changed_sums = sums.reindex(list(changed_coverers)).fillna(0.0)
    if not np.allclose(changed_sums.to_numpy(), 1.0, atol=1e-9):
        bad = changed_sums[~np.isclose(changed_sums, 1.0, atol=1e-9)]
        raise ValueError(f"contract-coverage weights failed to partition covering agencies: {bad.head(20).to_dict()}")

    jurisdiction_master = artifacts.jurisdiction_master.copy()
    contracted_ids = set(
        original.loc[original["ori"].isin(covered_oris), "jurisdiction_id"].astype(str)
    ) & municipal_ids
    jurisdiction_master.loc[
        jurisdiction_master["jurisdiction_id"].astype(str).isin(contracted_ids), "is_contracted_place"
    ] = True
    return ReferenceArtifacts(
        full_local=artifacts.full_local,
        full_nonlocal=artifacts.full_nonlocal,
        jurisdiction_master=jurisdiction_master,
        agency_to_jurisdiction_crosswalk=crosswalk,
    )


def apply_footprint_reassignment_overrides(
    artifacts: ReferenceArtifacts, paths
) -> ReferenceArtifacts:
    """Move a reviewed reporter share from an unresolved pool to its real polygon."""
    path = Path(paths.repo_root) / "configs" / "footprint_reassignment_overrides.csv"
    if not path.exists():
        return artifacts
    rows = pd.read_csv(path, dtype="string")
    required = {
        "ori", "source_jurisdiction_id", "target_jurisdiction_id", "state_fips",
        "state_abbr", "jurisdiction_name", "geo_type", "geoid", "source_note",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Footprint reassignment overrides missing columns: {sorted(missing)}")
    if rows.duplicated(["ori", "target_jurisdiction_id"]).any():
        raise ValueError("Duplicate footprint reassignment override")

    crosswalk = artifacts.agency_to_jurisdiction_crosswalk.copy()
    master = artifacts.jurisdiction_master.copy()
    for row in rows.itertuples(index=False):
        mask = (
            crosswalk["ori"].astype("string").eq(str(row.ori))
            & crosswalk["jurisdiction_id"].astype("string").eq(
                str(row.source_jurisdiction_id)
            )
            & pd.to_numeric(crosswalk["weight"], errors="coerce").gt(0.0)
        )
        target_mask = (
            crosswalk["ori"].astype("string").eq(str(row.ori))
            & crosswalk["jurisdiction_id"].astype("string").eq(
                str(row.target_jurisdiction_id)
            )
            & pd.to_numeric(crosswalk["weight"], errors="coerce").gt(0.0)
        )
        if int(mask.sum()) == 0 and int(target_mask.sum()) == 1:
            continue
        if int(mask.sum()) != 1:
            raise ValueError(
                "Footprint reassignment must match exactly one positive source row: "
                f"{row.ori} {row.source_jurisdiction_id} matched {int(mask.sum())}"
            )
        crosswalk.loc[mask, "jurisdiction_id"] = str(row.target_jurisdiction_id)
        crosswalk.loc[mask, "relationship_type"] = "reviewed_covered_footprint"
        crosswalk.loc[mask, "review_status"] = "reviewed"
        crosswalk.loc[mask, "resolution_source"] = "footprint_reassignment_override"
        crosswalk.loc[mask, "source_table"] = "footprint_reassignment_overrides"
        if not master["jurisdiction_id"].astype("string").eq(
            str(row.target_jurisdiction_id)
        ).any():
            master = pd.concat(
                [
                    master,
                    pd.DataFrame(
                        [
                            {
                                "jurisdiction_id": str(row.target_jurisdiction_id),
                                "jurisdiction_type": "municipal",
                                "state_fips": str(row.state_fips).zfill(2),
                                "state_abbr": str(row.state_abbr),
                                "jurisdiction_name": str(row.jurisdiction_name),
                                "geometry_source": f"tiger_2020_{row.geo_type}",
                                "is_contracted_place": True,
                                "manual_review_flag": True,
                                "geo_type": str(row.geo_type),
                                "geoid": str(row.geoid),
                                "agency_count": 1,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    sums = crosswalk[crosswalk["weight"].gt(0)].groupby("ori")["weight"].sum()
    affected = sums.reindex(rows["ori"].astype(str).unique())
    if affected.isna().any() or not np.allclose(affected.to_numpy(), 1.0, atol=1e-9):
        raise ValueError(f"Footprint reassignment broke ORI weights: {affected.to_dict()}")
    return ReferenceArtifacts(
        full_local=artifacts.full_local,
        full_nonlocal=artifacts.full_nonlocal,
        jurisdiction_master=master.sort_values(
            ["state_fips", "jurisdiction_type", "jurisdiction_id"]
        ).reset_index(drop=True),
        agency_to_jurisdiction_crosswalk=crosswalk.sort_values(
            ["state_fips", "ori", "jurisdiction_id"]
        ).reset_index(drop=True),
    )
