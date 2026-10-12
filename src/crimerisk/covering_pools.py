"""Zero-total towns take their covering pool's rate (fix A, 2025.2).

A town whose smoothed control is zero for every offense is not a town without crime. Its
own ORI files nothing (a corroborated structural zero) or nothing usable, and the crimes
committed there are reported by whoever polices it: the county's state-police subunit or
sheriff, whose mass the allocator localizes to the county's nonmunicipal block groups, a
Pennsylvania State Police service unit, or a contracting city. Left alone, the town's block
groups carry zero and the covering agency's footprint carries the town's crimes on fewer
people (Chester MA against the one Hampden remainder block group: 375.8x).

The rule, per offense and covering pool: the pool's mass is re-spread over its donor
components (in their current shape) and the zero-total towns it covers (the pool's rate,
mass over donor exposure, times each recipient block group's exposure in the town). Only
the donors' mass on the pool's own ground is pooled. Every donor lineage keeps its own
total, so each control, each source and the nation are conserved exactly; only the
location of the pool's mass changes.

Covering pool, first match:
1. the contracting agency's municipal jurisdiction, where the FBI batch header names one
   (`covered_by_ori`, latest year at or before the target year);
2. the county's nonmunicipal pools (county-localized remainder and county-localized
   nonmunicipal overlap), which is where state-police subunit and sheriff mass sits;
3. a Pennsylvania State Police service jurisdiction in the county.
A town with none of the three keeps zero and is counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.contract_coverage import load_latest_covered_by_relationships
from crimerisk.crime import OFFENSES_7


COUNTY_POOL_TYPES = frozenset(
    {"localized_remainder_county_layer", "localized_overlap_county_nonmunicipal_layer"}
)
POLICE_SERVICE_MARKER = ":police_service:"
STATE_REMAINDER_SUFFIX = ":state_nonmunicipal_remainder"
RECIPIENT_COLUMN = "covering_pool_recipient_jurisdiction_id"
POOL_COLUMN = "covering_pool_id"
LINEAGE_COLUMNS = (
    "jurisdiction_id",
    "jurisdiction_type",
    "canonical_target_ori",
    "service_scope_id",
    "source_state_fips",
    "source_target_count",
)


COVERED_REPAIR_MODE = "covered_by_covering_agency"


def covering_pool_recipients(controls: pd.DataFrame) -> pd.Index:
    """The towns a covering pool spreads onto.

    A town is a recipient because of who polices it, not because a number happens to be
    zero: every offense row of its control carries the covered-by identity (its ORI is
    numerator-exclusive to the covering agency), or, failing any recorded relationship, its
    control is zero for all seven offenses (`zero_total_towns`). A covered town keeps no
    numerator of its own; one that does stops the build, because pooling onto it would count
    its crime twice.
    """
    municipal = controls[controls["jurisdiction_type"].astype("string").eq("municipal")]
    zero = zero_total_towns(controls)
    if "level_repair_mode" not in municipal.columns:
        return zero
    mode = municipal["level_repair_mode"].astype("string")
    covered_all = mode.eq(COVERED_REPAIR_MODE).fillna(False).groupby(
        municipal["jurisdiction_id"].astype("string")
    ).all()
    population = pd.to_numeric(municipal["bucket_population"], errors="coerce").fillna(0.0).groupby(
        municipal["jurisdiction_id"].astype("string")
    ).max()
    covered = covered_all[covered_all & population.reindex(covered_all.index).gt(0.0)].index
    count = pd.to_numeric(municipal["adjusted_count_ags_core"], errors="coerce").fillna(0.0).clip(lower=0.0)
    own = count.groupby(municipal["jurisdiction_id"].astype("string")).sum()
    keeping_mass = own.reindex(covered).gt(1e-9)
    if keeping_mass.any():
        raise ValueError(
            "covered towns still carry a numerator of their own: "
            + str(own.reindex(covered)[keeping_mass].head(10).to_dict())
        )
    return zero.union(pd.Index(covered, dtype=zero.dtype))


def zero_total_towns(controls: pd.DataFrame) -> pd.Index:
    """Municipal jurisdictions with population whose control is zero for all seven offenses."""
    municipal = controls[controls["jurisdiction_type"].astype("string").eq("municipal")]
    count = pd.to_numeric(municipal["adjusted_count_ags_core"], errors="coerce").fillna(0.0).clip(lower=0.0)
    population = pd.to_numeric(municipal["bucket_population"], errors="coerce").fillna(0.0)
    frame = pd.DataFrame(
        {"jurisdiction_id": municipal["jurisdiction_id"].astype("string"), "count": count, "pop": population}
    )
    totals = frame.groupby("jurisdiction_id").agg(count=("count", "sum"), pop=("pop", "max"))
    return totals.index[totals["count"].le(1e-9) & totals["pop"].gt(0.0)]


def assign_covering_pools(
    *,
    towns: pd.Index,
    recipients: pd.DataFrame,
    agency_crosswalk: pd.DataFrame,
    covered_by: pd.DataFrame,
    agency_types: pd.Series,
    donor_keys: pd.DataFrame,
) -> pd.DataFrame:
    """One covering pool per (town, county).

    `recipients` has (jurisdiction_id, bg_id); `donor_keys` has the distinct
    (jurisdiction_id, jurisdiction_type, county) of the candidate donor components.
    Returns (town, county, pool_kind, pool_id, donor_jurisdiction_ids).
    """
    crosswalk = agency_crosswalk.assign(
        ori=agency_crosswalk["ori"].astype("string").str.upper(),
        jurisdiction_id=agency_crosswalk["jurisdiction_id"].astype("string"),
    )
    covering = dict(
        zip(
            covered_by["ori9"].astype("string").str.upper(),
            covered_by["covering_ori"].astype("string").str.upper(),
            strict=False,
        )
    )
    town_oris = crosswalk[crosswalk["jurisdiction_id"].isin(set(towns))].groupby("jurisdiction_id")["ori"].agg(list)
    ori_jurisdictions = (
        crosswalk[crosswalk["jurisdiction_id"].str.contains(":municipal:", na=False)]
        .groupby("ori")["jurisdiction_id"]
        .agg(lambda values: sorted(set(values)))
    )
    county_pools = (
        donor_keys[donor_keys["jurisdiction_type"].isin(COUNTY_POOL_TYPES)]
        .groupby("county")["jurisdiction_id"]
        .agg(lambda values: sorted(set(values)))
    )
    service = donor_keys[donor_keys["jurisdiction_id"].str.contains(POLICE_SERVICE_MARKER, na=False)].copy()
    service["ori"] = service["jurisdiction_id"].str.rsplit(":", n=1).str[-1].str.upper()
    service = service[service["ori"].map(agency_types).eq("state_law_enforcement")]
    service_pools = service.groupby("county")["jurisdiction_id"].agg(lambda values: sorted(set(values)))

    pairs = recipients[["jurisdiction_id", "bg_id"]].assign(county=lambda d: d["bg_id"].str.slice(0, 5))
    pairs = pairs[["jurisdiction_id", "county"]].drop_duplicates()
    rows = []
    for town, county in pairs.itertuples(index=False):
        contract = sorted(
            {
                jurisdiction
                for ori in town_oris.get(town, [])
                if ori in covering and covering[ori] != ori
                for jurisdiction in ori_jurisdictions.get(covering[ori], [])
                if jurisdiction != town
            }
        )
        if contract:
            kind, donors = "contracting_agency", contract
        elif county in county_pools.index:
            kind, donors = "county_nonmunicipal_pool", county_pools[county]
        elif county in service_pools.index:
            kind, donors = "state_police_service", service_pools[county]
        else:
            kind, donors = "none", []
        rows.append(
            {
                "town": town,
                "county": county,
                "pool_kind": kind,
                # A contracting agency's pool is the agency's own jurisdiction wherever the
                # covered town lies; the other pools are one county's.
                "pool_id": (
                    (f"{kind}:" if kind == "contracting_agency" else f"{kind}:{county}:")
                    + "|".join(donors)
                    if donors
                    else pd.NA
                ),
                "donor_jurisdiction_ids": tuple(donors),
            }
        )
    return pd.DataFrame(rows, columns=["town", "county", "pool_kind", "pool_id", "donor_jurisdiction_ids"])


def spread_covering_pools(
    components: pd.DataFrame,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move each covering pool's rate onto the zero-total towns it covers (`_spread`).

    The component table's attrs carry DataFrames, which pandas cannot compare when it
    propagates attrs through groupby/concat; they are set aside for the computation and
    returned on the result.
    """
    saved = dict(components.attrs)
    components.attrs = {}
    try:
        result, audit = _spread(components, **kwargs)
    finally:
        components.attrs = saved
    result.attrs = saved
    return result, audit


def _spread(
    components: pd.DataFrame,
    *,
    controls: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    agency_crosswalk: pd.DataFrame,
    agency_types: pd.Series,
    covered_by: pd.DataFrame,
    denominators: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move each covering pool's rate onto the zero-total towns it covers.

    `denominators` (bg_id, offense, primary_denominator_raw) supplies block-group exposure
    when the components do not carry it.

    Returns the new component table (donor rows scaled, recipient rows appended) and an
    audit with one row per (pool, offense, recipient town).
    """
    audit_columns = [
        "town", "county", "pool_kind", "pool_id", "offense", "pool_mass", "donor_exposure",
        "pool_rate", "recipient_exposure", "moved_mass",
    ]
    towns = covering_pool_recipients(controls)
    if components.empty or len(towns) == 0:
        return components, pd.DataFrame(columns=audit_columns)
    out = components.copy()
    out["component_count"] = pd.to_numeric(out["component_count"], errors="coerce").fillna(0.0)
    share = bg_crosswalk.assign(
        bg_id=bg_crosswalk["block_group_geoid"].astype("string").str.zfill(12),
        jurisdiction_id=bg_crosswalk["jurisdiction_id"].astype("string"),
        allocation_share=pd.to_numeric(bg_crosswalk["allocation_share"], errors="coerce").fillna(0.0),
    )
    share = share.groupby(["bg_id", "jurisdiction_id"])["allocation_share"].sum()
    recipients = share.reset_index()
    recipients = recipients[recipients["jurisdiction_id"].isin(set(towns)) & recipients["allocation_share"].gt(0.0)]
    if recipients.empty:
        return components, pd.DataFrame(columns=audit_columns)

    bg = out["bg_id"].astype("string").str.zfill(12)
    county = bg.str.slice(0, 5)
    jurisdiction = out["jurisdiction_id"].astype("string")
    jtype = out["jurisdiction_type"].astype("string")
    # A pool counts only where it holds mass on its own ground: a county pool on block
    # groups with nonmunicipal (state remainder) share, which Pennsylvania's all-municipal
    # counties do not have, so their towns fall through to the state-police service unit.
    remainder_share = share.reindex(
        pd.MultiIndex.from_arrays([bg, bg.str.slice(0, 2) + STATE_REMAINDER_SUFFIX])
    ).fillna(0.0).to_numpy(dtype=float)
    positive = out["component_count"].to_numpy(dtype=float) > 0.0
    donor_candidate = pd.Series(
        positive
        & (
            (jtype.isin(COUNTY_POOL_TYPES).to_numpy() & (remainder_share > 0.0))
            | jtype.eq("municipal").to_numpy()
            | jurisdiction.str.contains(POLICE_SERVICE_MARKER, na=False).to_numpy()
        ),
        index=out.index,
    )
    donor_keys = pd.DataFrame(
        {"jurisdiction_id": jurisdiction[donor_candidate], "jurisdiction_type": jtype[donor_candidate], "county": county[donor_candidate]}
    ).drop_duplicates()
    pools = assign_covering_pools(
        towns=towns,
        recipients=recipients,
        agency_crosswalk=agency_crosswalk,
        covered_by=covered_by,
        agency_types=agency_types,
        donor_keys=donor_keys,
    )
    placed = pools[pools["pool_kind"].ne("none")]
    if placed.empty:
        return components, pd.DataFrame(columns=audit_columns)

    # Block-group denominators per offense, read off the components (a BG-level field).
    source = (
        denominators
        if denominators is not None
        else pd.DataFrame(
            {"bg_id": bg, "offense": out["offense"], "primary_denominator_raw": out["primary_denominator_raw"]}
        )
    )
    denominator = (
        pd.DataFrame(
            {
                "bg_id": source["bg_id"].astype("string").str.zfill(12),
                "offense": source["offense"].astype("string"),
                "d": pd.to_numeric(source["primary_denominator_raw"], errors="coerce").fillna(0.0).clip(lower=0.0),
            }
        )
        .groupby(["bg_id", "offense"])["d"]
        .max()
    )

    def crosswalk_share(bg_ids: pd.Series, jurisdiction_ids: pd.Series) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([bg_ids.astype("string"), jurisdiction_ids.astype("string")])
        return share.reindex(keys).fillna(0.0).to_numpy(dtype=float)

    new_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    lineage = [column for column in LINEAGE_COLUMNS if column in out.columns]
    counts = out["component_count"].to_numpy(dtype=float).copy()
    bg_values = bg.to_numpy()
    county_values = county.to_numpy()
    offense_values = out["offense"].astype("string").to_numpy()
    jurisdiction_values = jurisdiction.to_numpy()
    jtype_values = jtype.to_numpy()
    donor_ids = set().union(*placed["donor_jurisdiction_ids"].map(set))
    candidate_rows = np.flatnonzero(jurisdiction.isin(donor_ids).to_numpy())
    # A pool is its mass on its own ground: the donor rows on block groups its crosswalk
    # jurisdiction owns (the state remainder for the county pools), the same block groups its
    # exposure is read from. Mass a donor carries elsewhere (the county-wide share the state-
    # police footprint-coverage rule, B2, moves off a footprint) is neither pooled nor rescaled,
    # so the rate is mass over exposure on one and the same ground.
    candidate_crosswalk = np.where(
        np.isin(jtype_values[candidate_rows], list(COUNTY_POOL_TYPES)),
        pd.Series(bg_values[candidate_rows]).str.slice(0, 2).to_numpy() + STATE_REMAINDER_SUFFIX,
        jurisdiction_values[candidate_rows],
    )
    candidate_rows = candidate_rows[
        crosswalk_share(pd.Series(bg_values[candidate_rows]), pd.Series(candidate_crosswalk)) > 0.0
    ]
    rows_by_jurisdiction = pd.Series(candidate_rows).groupby(jurisdiction_values[candidate_rows]).indices
    rows_by_jurisdiction = {key: candidate_rows[value] for key, value in rows_by_jurisdiction.items()}
    lineage_frame = out[lineage]
    for pool_id, members in placed.groupby("pool_id", sort=True):
        donors = members["donor_jurisdiction_ids"].iloc[0]
        kind = str(members["pool_kind"].iloc[0])
        pool_county = str(members["county"].iloc[0])
        pool_rows = np.concatenate(
            [rows_by_jurisdiction[d] for d in donors if d in rows_by_jurisdiction] or [np.array([], dtype=int)]
        )
        if kind != "contracting_agency":
            pool_rows = pool_rows[county_values[pool_rows] == pool_county]
        member_recipients = recipients.merge(
            members[["town", "county"]].rename(columns={"town": "jurisdiction_id"}), on="jurisdiction_id"
        )
        member_recipients = member_recipients[
            member_recipients["bg_id"].str.slice(0, 5).eq(member_recipients["county"])
        ].reset_index(drop=True)
        for offense in OFFENSES_7:
            donor_index = pool_rows[offense_values[pool_rows] == offense]
            if len(donor_index) == 0:
                continue
            pool_mass = float(counts[donor_index].sum())
            if pool_mass <= 0.0:
                continue
            # The pool's exposure: each donor block group's denominator times the share of
            # it the pool's crosswalk jurisdiction owns (the state remainder for the county
            # pools), counted once per block group.
            donor_bg = pd.DataFrame(
                {
                    "bg_id": bg_values[donor_index],
                    "crosswalk_id": np.where(
                        np.isin(jtype_values[donor_index], list(COUNTY_POOL_TYPES)),
                        pd.Series(bg_values[donor_index]).str.slice(0, 2).to_numpy() + STATE_REMAINDER_SUFFIX,
                        jurisdiction_values[donor_index],
                    ),
                }
            ).drop_duplicates()
            donor_d = denominator.reindex(
                pd.MultiIndex.from_arrays([donor_bg["bg_id"], pd.Series(offense, index=donor_bg.index)])
            ).fillna(0.0).to_numpy(dtype=float)
            donor_exposure = float(
                (donor_d * crosswalk_share(donor_bg["bg_id"], donor_bg["crosswalk_id"])).sum()
            )
            if donor_exposure <= 0.0:
                continue
            rate = pool_mass / donor_exposure
            rec = member_recipients
            rec_d = denominator.reindex(
                pd.MultiIndex.from_arrays([rec["bg_id"], pd.Series(offense, index=rec.index)])
            ).fillna(0.0).to_numpy(dtype=float)
            exposure = rec_d * rec["allocation_share"].to_numpy(dtype=float)
            recipient_exposure = float(exposure.sum())
            if recipient_exposure <= 0.0:
                continue
            raw = rate * exposure
            scale = pool_mass / (pool_mass + float(raw.sum()))
            received = raw * scale
            # Recipient mass keeps its donor lineage, split by each lineage's share of the pool.
            lineage_mass = (
                lineage_frame.iloc[donor_index]
                .assign(_mass=counts[donor_index])
                .groupby(lineage, dropna=False)["_mass"]
                .sum()
            )
            lineage_mass = lineage_mass[lineage_mass.gt(0.0)]
            counts[donor_index] = counts[donor_index] * scale
            for key, mass in lineage_mass.items():
                key = key if isinstance(key, tuple) else (key,)
                fraction = float(mass) / pool_mass
                block = pd.DataFrame(
                    {
                        "state_fips": rec["bg_id"].str.slice(0, 2).to_numpy(),
                        "bg_id": rec["bg_id"].to_numpy(),
                        "tract_id": rec["bg_id"].str.slice(0, 11).to_numpy(),
                        "offense": offense,
                        "component_count": received * fraction,
                        RECIPIENT_COLUMN: rec["jurisdiction_id"].to_numpy(),
                        POOL_COLUMN: str(pool_id),
                    }
                )
                for column, value in zip(lineage, key, strict=True):
                    block[column] = value
                new_rows.append(block)
            by_town = pd.DataFrame(
                {"town": rec["jurisdiction_id"].to_numpy(), "exposure": exposure, "received": received}
            ).groupby("town").sum()
            for town, values in by_town.iterrows():
                audit_rows.append(
                    {
                        "town": town,
                        "county": pool_county,
                        "pool_kind": kind,
                        "pool_id": str(pool_id),
                        "offense": offense,
                        "pool_mass": pool_mass,
                        "donor_exposure": donor_exposure,
                        "pool_rate": rate,
                        "recipient_exposure": float(values["exposure"]),
                        "moved_mass": float(values["received"]),
                    }
                )
    out["component_count"] = counts
    if not new_rows:
        return components, pd.DataFrame(columns=audit_columns)
    appended = pd.concat(new_rows, ignore_index=True)
    for column in appended.columns:
        if column not in out.columns:
            out[column] = pd.NA
    result = pd.concat([out, appended.reindex(columns=out.columns)], ignore_index=True)
    before = components.groupby(components["offense"].astype("string"))["component_count"].sum()
    after = result.groupby(result["offense"].astype("string"))["component_count"].sum()
    if not np.allclose(before.reindex(after.index).to_numpy(float), after.to_numpy(float), rtol=1e-9, atol=1e-6):
        raise AssertionError("covering-pool spread did not conserve national mass per offense")
    audit = pd.DataFrame(audit_rows, columns=audit_columns)
    unplaced = pools[pools["pool_kind"].eq("none")]
    audit.attrs["towns_without_pool"] = sorted(set(unplaced["town"].astype(str)) - set(audit["town"].astype(str)))
    audit.attrs["zero_total_towns"] = int(len(towns))
    return result, audit


def spread_covering_pools_for_build(
    components: pd.DataFrame,
    *,
    paths,
    year: int,
    controls: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    agency_crosswalk: pd.DataFrame,
    agency_master: pd.DataFrame,
    denominators: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    agency_types = (
        agency_master.assign(ori9=agency_master["ori9"].astype("string").str.upper())
        .drop_duplicates("ori9")
        .set_index("ori9")["agency_type_norm"]
        .astype("string")
    )
    return spread_covering_pools(
        components,
        controls=controls,
        bg_crosswalk=bg_crosswalk,
        agency_crosswalk=agency_crosswalk,
        agency_types=agency_types,
        covered_by=load_latest_covered_by_relationships(paths, year=int(year)),
        denominators=denominators,
    )
