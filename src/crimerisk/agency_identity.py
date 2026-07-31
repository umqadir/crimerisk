"""Who is one agency -- identity resolution applied before source preference.

Source preference answers "which lane's number do we believe for this agency-year".
It cannot answer "is this one agency or two", and when it is handed the same agency
twice it dutifully believes both. Two deterministic identity rules run first, both
about identity rather than value:

**1. Cross-lane twins.** During the NIBRS transition an agency's NIBRS submissions
arrive under an ORI9 whose last two characters differ from the ORI9 the summary
lanes carry (`CA034550X` / `MA009629E` against `CA0345500` / `MA0096200`; every
non-`00` ORI9 suffix in the panel belongs to the NIBRS lane and to no other). Both
survive into the preferred panel, both are crosswalked at weight 1.0 and both
receive a target-year estimate, so the agency's mass is carried twice -- and the
NIBRS-side twin is also mis-footprinted, because its `agency_master` name comes
from the NIBRS batch header's `city_name`, which is the covering city rather than
the agency (Rancho Cordova's twin is named SACRAMENTO and lands on Sacramento
city; nine Orange County contract cities land on Santa Ana).

The witness that two ORIs carry one submission is an exact agreement of the whole
seven-offense count vector in a year with positive counts: the FBI's Return A "as
released" for these agency-years IS its own conversion of the same NIBRS
submission, so the agreement is that conversion, not a coincidence. A single-year,
single-offense agreement on a small count is not a witness by itself -- there are
only seven such vectors -- so that tier additionally requires the FBI's own agency
roster to corroborate by listing the NIBRS ORI and not listing the summary ORI at
all. Resolution is to the summary-lane `...00` ORI, which is the one whose name,
county and jurisdiction footprint are already resolved.

Groups that share an ORI7 stem but disagree on counts (the audit's a2/a3 classes)
are NOT identity-resolvable by any rule -- a campus PD inside a city's ORI block
looks exactly like a partial rendering of the city -- and are left for per-case
review. Those reviews came back, and `build_adjudicated_twin_ledger` below carries
their verdicts into the same ledger the rule builds, under one admission gate:
an adjudicated merge is applied only when it is MASS-NEUTRAL, i.e. every member of
the group already sits on the same jurisdiction footprint, so re-keying deduplicates
without relocating anything. Cases whose members sit on different footprints are
quarantined rather than applied, because the registry has no machine-readable
"which footprint survives" field and several of the reviewers said in prose that
the surviving footprint is NOT the canonical ORI's own crosswalk row (Yountville:
"the survivor has the WRONG footprint and the dead ORI has the RIGHT one"). Applying
those from `canonical_ori` alone would move a town's counts into a state remainder.
They belong to the Stage-2 footprint registries, one decision each.

**2. ORI succession.** An agency that changed ORI leaves a retired ORI behind with
several years of history and no current report. The fill ladder reads that history
and resurrects it every year, onto a place its live successor already covers in
full: Jacksonville FL0160200 was retired in 2021 in favour of FL0160000 and still
receives a 25,685-count fill, inflating the city's total by 77%. Every ORI in the
crosswalk resolves to exactly one jurisdiction at weight 1.0, so the rule is
exact: a municipal jurisdiction whose live agency reports the target year cannot
simultaneously be covered by an agency that stopped reporting years ago. The dead
ORI is superseded -- no estimate row at all, not a fill and not a silent-unit
imputation, because the territory is not silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.source_provenance import NIBRS_SOURCE
from crimerisk.stage1_adjudications import (
    Stage1AdjudicationError,
    load_twin_rulings,
)


CANONICAL_ORI9_SUFFIX = "00"
ORI7_STEM_LENGTH = 7

# A single-year, single-offense vector agreement is weak evidence on its own (only
# seven such vectors exist), so it is accepted only with the roster witness below.
TWIN_MIN_MATCH_YEARS = 2
TWIN_MIN_MATCHED_OFFENSE_CATEGORIES = 2

CROSS_LANE_TWIN_LEDGER_COLUMNS = [
    "variant_ori9",
    "canonical_ori9",
    "ori7_stem",
    "state_abbr",
    "match_years",
    "matched_offense_categories",
    "matched_max_total",
    "variant_on_fbi_roster",
    "canonical_on_fbi_roster",
    "identity_evidence",
]

ORI_SUCCESSION_LEDGER_COLUMNS = [
    "superseded_ori9",
    "successor_ori9",
    "jurisdiction_id",
    "state_fips",
    "superseded_last_usable_year",
    "successor_last_usable_year",
    "years_since_last_usable_report",
]


def load_fbi_roster_oris(paths, *, year: int) -> set[str]:
    """ORIs the FBI's own agency roster lists for `year`.

    The roster is the federal directory of agencies that exist, which is a
    different fact from whether an agency submitted data. It is the identity
    witness for cross-lane twins here and the defunct-vs-non-reporting split in
    `benchmark_imputation`; both read it through this one loader.
    """
    path = (
        paths.data_dir
        / f"FBI-CDE-Agency-Rosters-{int(year)}"
        / "parsed"
        / f"agency_rosters_{int(year)}.parquet"
    )
    if not path.exists():
        return set()
    roster = pd.read_parquet(path, columns=["ori"])
    return set(roster["ori"].astype("string").str.strip().str.upper().dropna())


def load_agency_jurisdiction_crosswalk(paths) -> pd.DataFrame:
    """ORI -> jurisdiction footprint, normalised, read through one loader.

    Every ORI in this table resolves to exactly one jurisdiction at weight 1.0; the
    weight column is kept because the succession rule's claim ("the footprint is
    covered in full") is about the weight, not merely about the pairing.
    """
    frame = pd.read_parquet(
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    ).rename(columns={"ori": "ori9"})
    frame["ori9"] = frame["ori9"].astype("string")
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
    frame["jurisdiction_id"] = frame["jurisdiction_id"].astype("string")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return frame[["ori9", "state_fips", "jurisdiction_id", "weight"]]


def _offense_signature_frame(observations: pd.DataFrame) -> pd.DataFrame:
    """One row per (ori9, year, source) carrying its seven-offense vector."""
    work = observations[["ori9", "year", "source", "offense", "count"]].copy()
    work["ori9"] = work["ori9"].astype("string")
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0.0)
    wide = (
        work.pivot_table(
            index=["ori9", "year", "source"],
            columns="offense",
            values="count",
            aggfunc="sum",
        )
        .reindex(columns=list(OFFENSES_7))
        .fillna(0.0)
        .reset_index()
    )
    offense_values = wide[list(OFFENSES_7)]
    wide["vector"] = offense_values.round(6).astype(str).agg("|".join, axis=1)
    wide["vector_total"] = offense_values.sum(axis=1)
    wide["vector_offense_categories"] = (offense_values > 0).sum(axis=1)
    wide["ori7_stem"] = wide["ori9"].str.slice(0, ORI7_STEM_LENGTH)
    wide["ori9_suffix"] = wide["ori9"].str.slice(ORI7_STEM_LENGTH)
    return wide


def build_cross_lane_twin_ledger(
    observations: pd.DataFrame,
    *,
    roster_oris: set[str],
) -> pd.DataFrame:
    """Resolve NIBRS-lane ORI9 variants to the summary-lane ORI they duplicate."""
    empty = pd.DataFrame(columns=CROSS_LANE_TWIN_LEDGER_COLUMNS)
    if observations.empty:
        return empty
    signatures = _offense_signature_frame(observations)
    state_by_ori = (
        observations[["ori9", "state_abbr"]]
        .assign(ori9=lambda frame: frame["ori9"].astype("string"))
        .dropna()
        .drop_duplicates(subset=["ori9"], keep="first")
        .set_index("ori9")["state_abbr"]
        .astype("string")
        .str.upper()
    )

    variants = signatures[
        signatures["source"].eq(NIBRS_SOURCE)
        & signatures["ori9_suffix"].ne(CANONICAL_ORI9_SUFFIX)
    ]
    canonicals = signatures[
        signatures["source"].ne(NIBRS_SOURCE)
        & signatures["ori9_suffix"].eq(CANONICAL_ORI9_SUFFIX)
    ]
    if variants.empty or canonicals.empty:
        return empty

    paired = variants[
        ["ori9", "ori7_stem", "year", "vector", "vector_total", "vector_offense_categories"]
    ].merge(
        canonicals[["ori9", "ori7_stem", "year", "vector"]].rename(
            columns={"ori9": "canonical_ori9", "vector": "canonical_vector"}
        ),
        on=["ori7_stem", "year"],
        how="inner",
    )
    if paired.empty:
        return empty
    paired["state_abbr"] = paired["ori9"].map(state_by_ori)
    paired["canonical_state_abbr"] = paired["canonical_ori9"].map(state_by_ori)
    matched = paired[
        paired["vector"].eq(paired["canonical_vector"])
        & paired["vector_total"].gt(0.0)
        & paired["state_abbr"].eq(paired["canonical_state_abbr"])
    ].drop_duplicates(subset=["ori9", "canonical_ori9", "year"])
    if matched.empty:
        return empty

    evidence = matched.groupby(["ori9", "canonical_ori9", "ori7_stem"], as_index=False).agg(
        match_years=("year", "nunique"),
        matched_offense_categories=("vector_offense_categories", "max"),
        matched_max_total=("vector_total", "max"),
    )
    evidence["state_abbr"] = evidence["ori9"].map(state_by_ori)
    roster = {str(ori).upper() for ori in roster_oris}
    evidence["variant_on_fbi_roster"] = evidence["ori9"].str.upper().isin(roster)
    evidence["canonical_on_fbi_roster"] = evidence["canonical_ori9"].str.upper().isin(roster)

    vector_witness = evidence["match_years"].ge(TWIN_MIN_MATCH_YEARS) | evidence[
        "matched_offense_categories"
    ].ge(TWIN_MIN_MATCHED_OFFENSE_CATEGORIES)
    roster_witness = evidence["variant_on_fbi_roster"] & ~evidence["canonical_on_fbi_roster"]
    evidence["identity_evidence"] = np.where(
        vector_witness,
        "offense_vector_identity",
        "single_count_vector_identity_with_fbi_roster_witness",
    )
    ledger = evidence[vector_witness | roster_witness].copy()
    if ledger.empty:
        return empty
    ledger = ledger.rename(columns={"ori9": "variant_ori9"})

    _assert_twin_resolution_is_a_function(ledger)
    return (
        ledger[CROSS_LANE_TWIN_LEDGER_COLUMNS]
        .sort_values(["state_abbr", "canonical_ori9", "variant_ori9"], kind="mergesort")
        .reset_index(drop=True)
    )


def _assert_twin_resolution_is_a_function(ledger: pd.DataFrame) -> None:
    """Fail closed if the rule would fold two live NIBRS ORIs into one identity.

    Resolution has to be a function from variant to canonical: a variant matching
    two `...00` ORIs, or two variants matching one `...00` ORI, means the stem
    block holds more than one agency and the vector witness is not enough to say
    which. That is an a2-class review case, not a merge.
    """
    variant_col = "variant_ori9" if "variant_ori9" in ledger.columns else "ori9"
    ambiguous_variant = ledger.groupby(variant_col)["canonical_ori9"].nunique()
    ambiguous_canonical = ledger.groupby("canonical_ori9")[variant_col].nunique()
    offenders = {
        "variants matching several canonical ORIs": sorted(
            ambiguous_variant[ambiguous_variant.gt(1)].index.astype(str)
        ),
        "canonical ORIs claimed by several variants": sorted(
            ambiguous_canonical[ambiguous_canonical.gt(1)].index.astype(str)
        ),
    }
    if any(offenders.values()):
        raise ValueError(
            "cross-lane twin resolution is not a function from variant to canonical "
            f"ORI: {offenders}"
        )


# Attributes that describe WHO the agency is rather than what it reported. After a
# twin merge these must be the canonical agency's, both because they are the resolved
# ones -- the NIBRS side's `agency_name_std` comes from the batch header's `city_name`,
# so Rancho Cordova's twin is named SACRAMENTO and Pontiac's is nameless -- and because
# downstream joins key on them: `reporting_regimes` aligns its five lanes on
# (`ori9`, year, offense, state, `agency_name_std`, `agency_type_norm`), so two
# identities under one ORI would split every one of that agency's regime rows in two.
AGENCY_IDENTITY_COLUMNS = (
    "state_fips",
    "state_abbr",
    "county_fips",
    "place_fips",
    "agency_name_raw",
    "agency_name_std",
    "agency_type_raw",
    "agency_type_norm",
    "crosswalk_agency_name",
    "census_name",
    "manual_review_flag",
    "population",
)


def apply_cross_lane_twin_ledger(
    observations: pd.DataFrame, ledger: pd.DataFrame
) -> pd.DataFrame:
    """Re-key resolved NIBRS-lane variants onto their canonical ORI, identity and all.

    Identity is a property of the agency, not of a year, so the whole variant
    series moves -- including the transition years whose vectors do not agree,
    which are the same agency's partial NIBRS coverage while Return A still
    carried its real submission. `ori7` moves with it so the panel's stem column
    keeps describing the row's own ORI, and the agency's descriptive attributes are
    resolved to one value per ORI, the canonical rows' where they have one.
    """
    if observations.empty or ledger.empty:
        return observations
    mapping = (
        ledger.set_index(ledger["variant_ori9"].astype("string"))["canonical_ori9"]
        .astype("string")
        .to_dict()
    )
    out = observations.copy()
    ori9 = out["ori9"].astype("string")
    resolved = ori9.map(mapping)
    merged = resolved.notna()
    if not merged.any():
        return observations
    out["ori9"] = ori9.where(~merged, resolved)
    if "ori7" in out.columns:
        out["ori7"] = (
            out["ori7"]
            .astype("string")
            .where(~merged, out.loc[:, "ori9"].astype("string").str.slice(0, ORI7_STEM_LENGTH))
        )

    affected = set(ledger["canonical_ori9"].astype(str))
    touched = out["ori9"].astype(str).isin(affected)
    if not touched.any():
        return out
    # Canonical rows first, then adopted ones, so `first` (which skips nulls) prefers
    # the canonical agency's own value and falls back to the variant's only where the
    # canonical never carried one.
    order = pd.DataFrame(
        {"ori9": out.loc[touched, "ori9"].astype(str), "_adopted": merged[touched].astype(int)}
    ).sort_values(["ori9", "_adopted"], kind="mergesort")
    for column in AGENCY_IDENTITY_COLUMNS:
        if column not in out.columns:
            continue
        values = out.loc[touched, column].reindex(order.index)
        canonical = values.groupby(order["ori9"].to_numpy()).first()
        replacement = out.loc[touched, "ori9"].astype(str).map(canonical)
        out.loc[touched, column] = replacement.where(replacement.notna(), out.loc[touched, column])
    return out


def _last_usable_year_by_ori(agency_panel: pd.DataFrame, *, target_year: int) -> pd.Series:
    """Latest year up to `target_year` in which the agency's chosen lane reported."""
    supported = (
        agency_panel["usable_as_observed"].fillna(False).astype(bool)
        | agency_panel["current_row_is_true_partial"].fillna(False).astype(bool)
    )
    years = pd.to_numeric(agency_panel["year"], errors="coerce")
    usable = agency_panel[supported & years.le(int(target_year))].copy()
    if usable.empty:
        return pd.Series(dtype="float64")
    usable["year"] = pd.to_numeric(usable["year"], errors="coerce")
    return usable.groupby(usable["ori9"].astype("string"))["year"].max()


def build_ori_succession_ledger(
    *,
    agency_panel: pd.DataFrame,
    agency_jurisdiction_crosswalk: pd.DataFrame,
    target_year: int,
    max_reference_age_years: int,
) -> pd.DataFrame:
    """Dead ORIs whose municipal footprint a live ORI already covers in full.

    `max_reference_age_years` is the same recency bound the fill ladder uses, so
    "dead" here means exactly "too stale to fill from" -- an agency that stopped
    reporting longer ago than the ladder will reach back. `live` means the
    successor reported the target year itself.
    """
    empty = pd.DataFrame(columns=ORI_SUCCESSION_LEDGER_COLUMNS)
    if agency_panel.empty or agency_jurisdiction_crosswalk.empty:
        return empty
    last_usable = _last_usable_year_by_ori(agency_panel, target_year=int(target_year))
    if last_usable.empty:
        return empty

    crosswalk = agency_jurisdiction_crosswalk.copy()
    ori_col = "ori9" if "ori9" in crosswalk.columns else "ori"
    crosswalk = crosswalk.rename(columns={ori_col: "ori9"})
    crosswalk["ori9"] = crosswalk["ori9"].astype("string")
    crosswalk["jurisdiction_id"] = crosswalk["jurisdiction_id"].astype("string")
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0)
    # Only exclusive municipal coverage can be superseded: an overlap layer or a
    # state nonmunicipal remainder legitimately hosts many agencies at once.
    crosswalk = crosswalk[
        crosswalk["jurisdiction_id"].str.contains(":municipal:", na=False)
        & crosswalk["weight"].ge(1.0)
    ].copy()
    if crosswalk.empty:
        return empty
    crosswalk["last_usable_year"] = crosswalk["ori9"].map(last_usable)

    stale_before = int(target_year) - int(max_reference_age_years)
    dead = crosswalk[
        crosswalk["last_usable_year"].notna() & crosswalk["last_usable_year"].lt(stale_before)
    ]
    live = crosswalk[crosswalk["last_usable_year"].eq(float(target_year))]
    if dead.empty or live.empty:
        return empty

    pairs = dead.merge(
        live[["ori9", "jurisdiction_id", "last_usable_year"]].rename(
            columns={"ori9": "successor_ori9", "last_usable_year": "successor_last_usable_year"}
        ),
        on="jurisdiction_id",
        how="inner",
    )
    pairs = pairs[pairs["ori9"].ne(pairs["successor_ori9"])]
    if pairs.empty:
        return empty
    pairs = pairs.rename(
        columns={"ori9": "superseded_ori9", "last_usable_year": "superseded_last_usable_year"}
    )
    pairs["years_since_last_usable_report"] = (
        int(target_year) - pairs["superseded_last_usable_year"]
    )
    return (
        pairs[ORI_SUCCESSION_LEDGER_COLUMNS]
        .sort_values(["state_fips", "jurisdiction_id", "superseded_ori9"], kind="mergesort")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# The adjudicated residue: the a2/a3 cases the two rules above declined to decide.
# ---------------------------------------------------------------------------

ADJUDICATED_TWIN_EVIDENCE = "adjudicated_case_review"
ADJUDICATED_SUCCESSION_REASON = "adjudicated_case_review"

ADJUDICATED_QUARANTINE_COLUMNS = [
    "case_id",
    "verdict",
    "canonical_ori9",
    "member_oris",
    "member_footprints",
    "reason",
]


def _assert_identity_resolution_is_single_valued(ledger: pd.DataFrame) -> None:
    """Every ORI resolves to exactly one identity -- the relaxed form of the rule's assertion.

    The rule's own `_assert_twin_resolution_is_a_function` additionally forbids TWO variants under
    one canonical, because a vector witness cannot say which of two live NIBRS ORIs it matched. An
    adjudicated ruling can: three Pennsylvania State Police keys collapsing onto one is a finding,
    not an ambiguity. What must still hold either way is that no ORI has two identities and no ORI
    is both a survivor and something that was collapsed -- those are contradictions, not findings.
    """
    if ledger.empty:
        return
    variants = ledger["variant_ori9"].astype(str)
    canonicals = ledger["canonical_ori9"].astype(str)
    ambiguous = sorted(
        ledger.groupby(variants)["canonical_ori9"]
        .nunique()
        .pipe(lambda counts: counts[counts.gt(1)])
        .index.astype(str)
    )
    both = sorted(set(variants) & set(canonicals))
    offenders = {
        "ORIs resolved to several canonical ORIs": ambiguous,
        "ORIs that are both collapsed and a survivor": both,
    }
    if any(offenders.values()):
        raise Stage1AdjudicationError(
            f"identity resolution is not single-valued: {offenders}"
        )


def _footprint_lookup(agency_jurisdiction_crosswalk: pd.DataFrame) -> pd.DataFrame:
    """ORI -> its single weight-1.0 footprint, as the identity rules see it."""
    crosswalk = agency_jurisdiction_crosswalk.copy()
    ori_col = "ori9" if "ori9" in crosswalk.columns else "ori"
    crosswalk = crosswalk.rename(columns={ori_col: "ori9"})
    crosswalk["ori9"] = crosswalk["ori9"].astype("string").str.upper()
    crosswalk["jurisdiction_id"] = crosswalk["jurisdiction_id"].astype("string")
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0)
    return crosswalk[["ori9", "state_fips", "jurisdiction_id", "weight"]].drop_duplicates(
        subset=["ori9"], keep="first"
    )


def _member_footprints(members, footprints: pd.DataFrame) -> dict[str, str | None]:
    indexed = footprints.set_index("ori9")
    out: dict[str, str | None] = {}
    for ori in members:
        out[ori] = (
            str(indexed.loc[ori, "jurisdiction_id"])
            if ori in indexed.index and pd.notna(indexed.loc[ori, "jurisdiction_id"])
            else None
        )
    return out


def _mass_neutral(members, footprints: dict[str, str | None]) -> tuple[bool, str]:
    """Whether collapsing `members` onto one key can move mass off its current footprint.

    Only members that actually carry mass are judged: a member absent from the panel, or with no
    crosswalk row, has no counts standing anywhere, so re-keying it cannot relocate any.
    """
    known = {ori: footprints.get(ori) for ori in members if footprints.get(ori) is not None}
    if not known:
        return False, "no member of the group that carries mass has a jurisdiction footprint"
    distinct = set(known.values())
    if len(distinct) > 1:
        return False, (
            "members sit on different jurisdiction footprints, so re-keying would relocate mass; "
            "the surviving footprint is a Stage-2 decision and the registry does not carry one"
        )
    return True, ""


def build_adjudicated_twin_ledger(
    twin_rulings: pd.DataFrame,
    *,
    observations: pd.DataFrame,
    agency_jurisdiction_crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`same_agency_merge` verdicts, as rows of the cross-lane twin ledger.

    Returns `(ledger, quarantined)`. The ledger has the same columns the rule produces, so
    `apply_cross_lane_twin_ledger` consumes both without knowing which is which -- the merge
    semantics are identical (whole series re-keyed, descriptive attributes resolved to the
    canonical's). Only the witness differs, and `identity_evidence` records that.
    """
    empty_ledger = pd.DataFrame(columns=CROSS_LANE_TWIN_LEDGER_COLUMNS)
    empty_quarantine = pd.DataFrame(columns=ADJUDICATED_QUARANTINE_COLUMNS)
    if twin_rulings.empty:
        return empty_ledger, empty_quarantine

    footprints = _footprint_lookup(agency_jurisdiction_crosswalk)
    present = set(observations["ori9"].astype("string").str.upper().dropna().unique())
    state_by_ori = (
        observations[["ori9", "state_abbr"]]
        .assign(ori9=lambda frame: frame["ori9"].astype("string").str.upper())
        .dropna()
        .drop_duplicates(subset=["ori9"], keep="first")
        .set_index("ori9")["state_abbr"]
        .astype("string")
        .str.upper()
    )

    rows, quarantine = [], []
    merges = twin_rulings[twin_rulings["verdict"].eq("same_agency_merge")]
    for ruling in merges.to_dict(orient="records"):
        members = list(ruling["ori_list"])
        canonical = ruling["canonical_ori"]
        member_fp = _member_footprints(members, footprints)
        ok, why = _mass_neutral([o for o in members if o in present], member_fp)
        if member_fp.get(canonical) is None:
            ok, why = False, "the canonical ORI has no jurisdiction footprint, so the merged " \
                             "agency's mass would have nowhere to land"
        if not ok:
            quarantine.append(
                {
                    "case_id": ruling["case_id"],
                    "verdict": ruling["verdict"],
                    "canonical_ori9": canonical,
                    "member_oris": ";".join(members),
                    "member_footprints": ";".join(
                        f"{ori}={member_fp[ori] or 'none'}" for ori in members
                    ),
                    "reason": why,
                }
            )
            continue
        for variant in members:
            if variant == canonical or variant not in present:
                continue
            rows.append(
                {
                    "variant_ori9": variant,
                    "canonical_ori9": canonical,
                    "ori7_stem": variant[:ORI7_STEM_LENGTH],
                    "state_abbr": state_by_ori.get(variant, ruling["state"]),
                    "match_years": pd.NA,
                    "matched_offense_categories": pd.NA,
                    "matched_max_total": pd.NA,
                    "variant_on_fbi_roster": pd.NA,
                    "canonical_on_fbi_roster": pd.NA,
                    "identity_evidence": ADJUDICATED_TWIN_EVIDENCE,
                }
            )
    ledger = pd.DataFrame(rows, columns=CROSS_LANE_TWIN_LEDGER_COLUMNS)
    # The rule's witness columns are empty here by construction -- an adjudicated ruling is not a
    # vector match -- but they are typed so concatenating the two ledgers keeps one dtype per column.
    for column in ("match_years", "matched_offense_categories"):
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce").astype("Int64")
    ledger["matched_max_total"] = pd.to_numeric(ledger["matched_max_total"], errors="coerce")
    for column in ("variant_on_fbi_roster", "canonical_on_fbi_roster"):
        ledger[column] = ledger[column].astype("boolean")
    _assert_identity_resolution_is_single_valued(ledger)
    return ledger, pd.DataFrame(quarantine, columns=ADJUDICATED_QUARANTINE_COLUMNS)


def build_adjudicated_succession_ledger(
    twin_rulings: pd.DataFrame,
    *,
    observations: pd.DataFrame,
    agency_jurisdiction_crosswalk: pd.DataFrame,
    target_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`superseded_ori` verdicts, as rows of the ORI succession ledger.

    Admitted under the rule's OWN precondition, applied to an adjudicated pair: the dead ORI's
    footprint has to be the one the survivor already covers. The rule states that as a shared
    `jurisdiction_id`, and it is the whole basis for dropping the dead ORI's estimate rather than
    filling them -- "the territory is not silent" is only true if the survivor is standing on it.
    """
    empty_ledger = pd.DataFrame(columns=ORI_SUCCESSION_LEDGER_COLUMNS)
    empty_quarantine = pd.DataFrame(columns=ADJUDICATED_QUARANTINE_COLUMNS)
    if twin_rulings.empty:
        return empty_ledger, empty_quarantine

    footprints = _footprint_lookup(agency_jurisdiction_crosswalk).set_index("ori9")
    present = set(observations["ori9"].astype("string").str.upper().dropna().unique())

    rows, quarantine = [], []
    for ruling in twin_rulings[twin_rulings["verdict"].eq("superseded_ori")].to_dict(
        orient="records"
    ):
        members = list(ruling["ori_list"])
        survivor = ruling["canonical_ori"]
        member_fp = _member_footprints(members, footprints.reset_index())
        dead = [ori for ori in members if ori != survivor and ori in present]
        mismatched = [
            ori for ori in dead if member_fp.get(ori) != member_fp.get(survivor)
        ]
        if member_fp.get(survivor) is None or mismatched or not dead:
            quarantine.append(
                {
                    "case_id": ruling["case_id"],
                    "verdict": ruling["verdict"],
                    "canonical_ori9": survivor,
                    "member_oris": ";".join(members),
                    "member_footprints": ";".join(
                        f"{ori}={member_fp[ori] or 'none'}" for ori in members
                    ),
                    "reason": (
                        "no dead ORI of this pair appears in the panel"
                        if not dead
                        else "the survivor has no jurisdiction footprint"
                        if member_fp.get(survivor) is None
                        else "the dead ORI's footprint is not the survivor's, so superseding it "
                        "would remove coverage rather than deduplicate it; the surviving "
                        "footprint is a Stage-2 decision and the registry does not carry one"
                    ),
                }
            )
            continue
        for ori in dead:
            rows.append(
                {
                    "superseded_ori9": ori,
                    "successor_ori9": survivor,
                    "jurisdiction_id": member_fp[survivor],
                    "state_fips": (
                        str(footprints.loc[survivor, "state_fips"])
                        if survivor in footprints.index
                        else pd.NA
                    ),
                    "superseded_last_usable_year": pd.NA,
                    "successor_last_usable_year": int(target_year),
                    "years_since_last_usable_report": pd.NA,
                }
            )
    return (
        pd.DataFrame(rows, columns=ORI_SUCCESSION_LEDGER_COLUMNS),
        pd.DataFrame(quarantine, columns=ADJUDICATED_QUARANTINE_COLUMNS),
    )


def combine_twin_ledgers(
    rule_ledger: pd.DataFrame, adjudicated: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Rule first, adjudicated rows for the survivors; fail closed only where they CONTRADICT.

    The two populations are not perfectly disjoint. A registry case can name several ORIs of which
    the rule already resolved one -- Mount Holyoke's group holds a `...0X` variant the vector witness
    reaches and a `...9E` variant it does not -- and there the rule and the reviewer AGREE. Those
    rows are dropped as already-done and counted, because re-adding them would put the same variant
    in the ledger twice.

    What does fail closed is a disagreement: the same variant resolved to two different canonical
    ORIs, or an ORI the registry calls the survivor that the rule calls a variant of something else.
    Those are not precedence questions -- one of the two is wrong about who the agency is, and
    picking a winner in code would bury that.
    """
    if adjudicated.empty:
        return rule_ledger, 0
    if rule_ledger.empty:
        _assert_identity_resolution_is_single_valued(adjudicated)
        return adjudicated, 0
    rule_map = dict(
        zip(
            rule_ledger["variant_ori9"].astype(str),
            rule_ledger["canonical_ori9"].astype(str),
            strict=True,
        )
    )
    rule_variants = set(rule_map)
    contradictions = [
        {"ori": variant, "rule_canonical": rule_map[variant], "adjudicated_canonical": canonical}
        for variant, canonical in zip(
            adjudicated["variant_ori9"].astype(str),
            adjudicated["canonical_ori9"].astype(str),
            strict=True,
        )
        if variant in rule_variants and rule_map[variant] != canonical
    ]
    contradictions += [
        {
            "ori": canonical,
            "rule_canonical": rule_map[canonical],
            "adjudicated_canonical": "(named as the survivor)",
        }
        for canonical in set(adjudicated["canonical_ori9"].astype(str))
        if canonical in rule_variants
    ]
    if contradictions:
        raise Stage1AdjudicationError(
            "the deterministic twin rule and the adjudicated twin registry disagree about who an "
            f"agency is: {contradictions}. One of the two is wrong; the rule's scope and the "
            "ruling both need re-reading, and neither should win by precedence."
        )
    already_resolved = adjudicated["variant_ori9"].astype(str).isin(rule_variants)
    combined = pd.concat([rule_ledger, adjudicated[~already_resolved]], ignore_index=True)
    _assert_identity_resolution_is_single_valued(combined)
    return combined, int(already_resolved.sum())


def assert_adjudicated_distinct_agencies_are_not_merged(
    ledger: pd.DataFrame, twin_rulings: pd.DataFrame
) -> None:
    """Fail closed when the merge machinery collapses a pair a reviewer ruled DISTINCT.

    `distinct_agencies` is the majority verdict in the registry (80 of 106) and it is the only one
    that produces no ledger row, so without this check it would be an unenforced opinion. It is a
    real test of the rule: the rule's witness is an exact seven-offense vector agreement, and a
    reviewer who saw the same evidence and said "two agencies" is asserting that no such agreement
    exists in a year with positive counts.
    """
    if ledger.empty or twin_rulings.empty:
        return
    resolved = {
        (str(variant), str(canonical))
        for variant, canonical in zip(
            ledger["variant_ori9"], ledger["canonical_ori9"], strict=True
        )
    }
    violations = []
    for ruling in twin_rulings[twin_rulings["verdict"].eq("distinct_agencies")].to_dict(
        orient="records"
    ):
        members = set(ruling["ori_list"])
        for variant, canonical in resolved:
            if variant in members and canonical in members:
                violations.append(
                    {
                        "case_id": ruling["case_id"],
                        "merged": f"{variant}->{canonical}",
                        "members": ";".join(sorted(members)),
                    }
                )
    if violations:
        raise Stage1AdjudicationError(
            "the twin ledger merges ORI pairs a reviewer ruled distinct_agencies: "
            f"{violations[:20]}"
        )


def resolve_agency_identity(
    observations: pd.DataFrame,
    *,
    paths,
    roster_oris: set[str],
    agency_jurisdiction_crosswalk: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the twin rule, then the adjudicated residue, and report what each one did.

    One entry point so no consumer can apply the rule and miss the registry, and so the counts land
    in the build summary rather than in a print nobody reads.
    """
    if agency_jurisdiction_crosswalk is None:
        agency_jurisdiction_crosswalk = load_agency_jurisdiction_crosswalk(paths)
    rule_ledger = build_cross_lane_twin_ledger(observations, roster_oris=roster_oris)
    twin_rulings = load_twin_rulings(paths)
    adjudicated, quarantined = build_adjudicated_twin_ledger(
        twin_rulings,
        observations=observations,
        agency_jurisdiction_crosswalk=agency_jurisdiction_crosswalk,
    )
    ledger, already_resolved_by_rule = combine_twin_ledgers(rule_ledger, adjudicated)
    assert_adjudicated_distinct_agencies_are_not_merged(ledger, twin_rulings)
    summary = {
        "twin_rule_variants_merged": int(len(rule_ledger)),
        "twin_adjudicated_rulings": int(len(twin_rulings)),
        "twin_adjudicated_merge_rulings": int(
            twin_rulings["verdict"].eq("same_agency_merge").sum()
        ),
        "twin_adjudicated_distinct_rulings": int(
            twin_rulings["verdict"].eq("distinct_agencies").sum()
        ),
        "twin_adjudicated_variants_merged": int(len(adjudicated)) - already_resolved_by_rule,
        "twin_adjudicated_variants_already_resolved_by_rule": already_resolved_by_rule,
        "twin_adjudicated_cases_applied": int(
            adjudicated["canonical_ori9"].nunique() if not adjudicated.empty else 0
        ),
        "twin_adjudicated_cases_quarantined": int(len(quarantined)),
        # A merge ruling that is admitted but whose collapsed members are absent from the panel
        # changes nothing. Counted rather than silently dropped: the registry contract treats an
        # inert row as a defect, and this is the honest place to see how many there are.
        "twin_adjudicated_cases_inert": int(
            twin_rulings["verdict"].eq("same_agency_merge").sum()
            - (adjudicated["canonical_ori9"].nunique() if not adjudicated.empty else 0)
            - len(quarantined)
        ),
        "twin_adjudicated_quarantined_cases": (
            quarantined["case_id"].tolist() if not quarantined.empty else []
        ),
    }
    if twin_rulings["verdict"].eq("same_agency_merge").any() and ledger.equals(rule_ledger):
        raise Stage1AdjudicationError(
            "the twin registry carries same_agency_merge rulings but none reached the ledger: "
            f"{summary}"
        )
    return (
        apply_cross_lane_twin_ledger(observations, ledger) if not ledger.empty else observations,
        summary,
    )


def resolve_ori_succession(
    *,
    paths,
    agency_panel: pd.DataFrame,
    agency_jurisdiction_crosswalk: pd.DataFrame,
    target_year: int,
    max_reference_age_years: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """The succession rule plus its adjudicated residue, as one ledger with counts."""
    rule_ledger = build_ori_succession_ledger(
        agency_panel=agency_panel,
        agency_jurisdiction_crosswalk=agency_jurisdiction_crosswalk,
        target_year=int(target_year),
        max_reference_age_years=int(max_reference_age_years),
    )
    twin_rulings = load_twin_rulings(paths)
    adjudicated, quarantined = build_adjudicated_succession_ledger(
        twin_rulings,
        observations=agency_panel,
        agency_jurisdiction_crosswalk=agency_jurisdiction_crosswalk,
        target_year=int(target_year),
    )
    already = set(rule_ledger["superseded_ori9"].astype(str)) if not rule_ledger.empty else set()
    new_rows = (
        adjudicated[~adjudicated["superseded_ori9"].astype(str).isin(already)]
        if not adjudicated.empty
        else adjudicated
    )
    ledger = (
        pd.concat([rule_ledger, new_rows], ignore_index=True)
        if not new_rows.empty
        else rule_ledger
    )
    summary = {
        "succession_rule_oris": int(len(rule_ledger)),
        "succession_adjudicated_rulings": int(twin_rulings["verdict"].eq("superseded_ori").sum()),
        "succession_adjudicated_oris_added": int(len(new_rows)),
        "succession_adjudicated_cases_quarantined": int(len(quarantined)),
        "succession_adjudicated_quarantined_cases": (
            quarantined["case_id"].tolist() if not quarantined.empty else []
        ),
    }
    return ledger, summary
