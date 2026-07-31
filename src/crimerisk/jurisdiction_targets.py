"""Stage 3 — jurisdiction targets, by consumption of the Stage-1 agency estimates.

This layer holds no estimator. Every target amount published here is a weighted sum of
Stage-1 per-agency estimates; the only things decided at jurisdiction level are which
jurisdictions exist, which agency owns which territory, and how the components are
labelled. The rule (adjudicated 2026-07-29 after an external design review, recorded in
`docs/STATE.md`):

1. **Skeleton first.** The control universe is the geometry / jurisdiction-ownership
   skeleton, never the estimate table. A jurisdiction that owns territory gets a row
   whether or not any agency reported for it, so a silent unit exists with zero
   pre-imputation mass and is visible to `benchmark_imputation` instead of vanishing
   before eligibility is evaluated. Absence from the estimate table must never mean
   absence from the control universe.

2. **Aggregation, not re-derivation.** The pre-imputation target of a jurisdiction is

       target = Σ over agencies ( stage-1 agency estimate × crosswalk weight )

   per state × offense × jurisdiction lane, for **all three lanes** — municipal, the
   state nonmunicipal remainder pool and the statewide overlap layer. There is no
   jurisdiction-level source preference, no jurisdiction-level usability rule and no
   jurisdiction-level fill ladder. Selection and filling are agency facts and neither
   commutes with aggregation: `fill(Σ agencies)` is not `Σ fill(agency)`, and one
   currently reporting agency can make a jurisdiction look current while another agency
   inside it is silent. Jurisdiction-level models may allocate these amounts spatially
   (Stage 4); they may never resize them.

3. **Provenance is composed, not chosen.** A jurisdiction whose agencies sit on
   different source lanes keeps the component masses of every lane and every estimate
   class. `preferred_source` and `dominant_reporting_regime` survive as descriptive
   labels for the published surface; nothing in this stage or downstream of it may make
   a selection or usability decision from them, because a dominant label conceals mixed
   provenance by construction.

4. **Fail closed on the conservation identities.** Crosswalk weights must partition each
   ORI; every positive agency estimate must land exactly once; agency-estimate mass must
   equal pre-imputation control mass per state × offense × lane; the row identity
   `target = reported + partial-year uplift + fill` must hold exactly; and a jurisdiction
   that owns no territory may not carry mass — it is enumerated in an explicit exclusion
   artifact rather than peer-filled or silently discarded.

What this replaced: `municipal_estimator.py` (a second source-preference rule family, a
second usability rule and a second fill ladder, run over jurisdiction rollups of the
observation panel and written onto the control unconditionally) and
`jurisdiction_estimator.py`'s generic jurisdiction ladder plus its two target-year
override patches. Between them they re-opened, at jurisdiction level, all five defect
classes Stage 1 had closed at the agency-year: partial years read as complete, fabricated
peer anchors, fills past the recency bound, per-offense lane forks, and silent agencies
re-animated by their own jurisdiction's history — the last of which locked the control
before `benchmark_imputation` could see the unit (the Jackson MS shape, now impossible by
construction). They also dropped 21,757 counts of overlap-layer agency mass that no
override ever reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.crime.municipal_totals import MunicipalTotalsConfig, _assign_pop_band
from crimerisk.paths import RepoPaths
from crimerisk.scope import PRODUCTION_SCOPE_EXCLUDE
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    default_conversion_status_from_source,
    raw_data_source_from_parts,
    reporting_mode_from_source,
    source_family_from_source,
    source_lane_from_source,
    source_origin_from_parts,
)
from crimerisk.trend_fills import (
    AGENCY_OBSERVED_ESTIMATE_SOURCE,
    LANE_GROUNDED_ESTIMATE_SOURCES,
    add_preferred_support_flags,
    build_agency_trend_fill_panel,
)


MUNICIPAL_TYPE = "municipal"
STATE_REMAINDER_TYPE = "state_nonmunicipal_remainder"
STATE_OVERLAP_TYPE = "statewide_overlap_layer"
JURISDICTION_LANES = (MUNICIPAL_TYPE, STATE_REMAINDER_TYPE, STATE_OVERLAP_TYPE)

# Lanes whose territory is the block-group crosswalk's own partition. The statewide
# overlap layer is deliberately not one of them: it is a state-level lane whose footprint
# is resolved at allocation time (overlap footprints and custom footprints), so its
# ownership test is "does any agency route here", not "does it hold block groups".
GEOMETRY_OWNED_LANES = (MUNICIPAL_TYPE, STATE_REMAINDER_TYPE)

PARTIAL_UPLIFT_ESTIMATE_SOURCE = "true_partial_month_ratio"

# The target-year estimate-source vocabulary of this stage. Each label describes the
# COMPOSITION of the mass on the row, not a rung of a ladder that no longer exists.
TARGET_SOURCE_OBSERVED = "agency_rollup_observed"
TARGET_SOURCE_PARTIAL_UPLIFT = "agency_rollup_partial_uplift"
TARGET_SOURCE_FILL = "agency_rollup_fill"
TARGET_SOURCE_NO_EVIDENCE = "no_agency_evidence"
HISTORY_SOURCE_REPORTED_ROLLUP = "agency_reported_rollup"

SOURCE_LANES = (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    NIBRS_SOURCE,
)

LANE_REPORTED_COLUMNS: dict[str, str] = {
    CIUS_SOURCE: "reported_count_cius",
    LOCAL_PUBLICATION_SOURCE: "reported_count_local_publication",
    STATE_PUBLICATION_SOURCE: "reported_count_state_publication",
    SUMMARY_SOURCE: "reported_count_srs",
    NIBRS_SOURCE: "reported_count_nibrs",
}

# Component provenance: how much of the row's TARGET came through each Stage-1 source
# lane. Kept per lane rather than collapsed to the dominant label, because the dominant
# label is exactly what hides a mixed-provenance jurisdiction.
LANE_TARGET_COLUMNS: dict[str, str] = {
    source: f"target_count_from_{source_lane_from_source(source)}"
    for source in SOURCE_LANES
}
# The residual bucket, so the components sum to the target exactly. It catches the one
# real case: an agency whose target-year estimate is a fill taken from its own history
# and which therefore has no target-year lane at all.
OTHER_LANE_TARGET_COLUMN = "target_count_from_reported_other"
LANE_TARGET_COMPONENT_COLUMNS = (*LANE_TARGET_COLUMNS.values(), OTHER_LANE_TARGET_COLUMN)

# Signed mass removed from the raw reported rollup by the ORI-succession ledger. A
# superseded ORI can still carry a published target-year row even though its successor
# covers the same agency and territory, so this component is negative by construction.
# It is derived from the ledger rather than from the row-identity residual: an unrelated
# leak must still fail closed instead of disappearing into a catch-all adjustment.
IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN = "identity_resolution_adjustment_count"

JURISDICTION_IDENTITY_COLUMNS = [
    "jurisdiction_id",
    "jurisdiction_type",
    "jurisdiction_name",
    "state_fips",
    "state_abbr",
    "geo_type",
    "geoid",
]

OWNERSHIP_EXCLUSION_COLUMNS = [
    "jurisdiction_id",
    "jurisdiction_type",
    "jurisdiction_name",
    "state_fips",
    "state_abbr",
    "geo_type",
    "geoid",
    "exclusion_reason",
    "crosswalk_agency_count",
    "crosswalk_agency_oris",
    "agency_estimate_mass",
]


@dataclass(frozen=True)
class JurisdictionTargetConfig:
    year_start: int = 2018
    target_year: int = 2024
    exclude_scope_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE))
    pop_bands: tuple[tuple[float, float, str], ...] = MunicipalTotalsConfig().pop_bands
    force_reporting_regimes_rebuild: bool = False


# --- inputs -----------------------------------------------------------------


def load_jurisdiction_master(paths: RepoPaths) -> pd.DataFrame:
    frame = pd.read_parquet(paths.state_dir / "reference" / "jurisdiction_master.parquet")
    frame["jurisdiction_id"] = frame["jurisdiction_id"].astype("string")
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
    frame["state_abbr"] = frame["state_abbr"].astype("string").str.upper()
    return frame


def load_crosswalk(paths: RepoPaths) -> pd.DataFrame:
    frame = pd.read_parquet(
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    ).rename(columns={"ori": "ori9"})
    frame["ori9"] = frame["ori9"].astype("string")
    frame["jurisdiction_id"] = frame["jurisdiction_id"].astype("string")
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    return frame


def load_block_group_exposure(paths: RepoPaths) -> pd.DataFrame:
    """Territory ownership and exposure for the two geometry-owned lanes, from the one
    crosswalk `benchmark_imputation` also sizes its units on."""
    frame = pd.read_parquet(
        paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        columns=["state_fips", "jurisdiction_id", "jurisdiction_type", "pop20"],
    )
    frame["jurisdiction_id"] = frame["jurisdiction_id"].astype("string")
    frame["pop20"] = pd.to_numeric(frame["pop20"], errors="coerce").fillna(0.0)
    return (
        frame.groupby("jurisdiction_id", dropna=False, as_index=False)["pop20"]
        .sum()
        .rename(columns={"pop20": "bucket_population"})
    )


def _assert_crosswalk_weights_partition_every_agency(crosswalk: pd.DataFrame) -> None:
    """Fail closed unless each ORI's weights sum to exactly one.

    Weights below one lose agency mass on its way into the surface and weights above one
    duplicate it; either way the aggregation identity this stage is built on stops being
    an identity. It has always held (26,767 ORIs, zero deviation) and was never asserted.
    """
    if crosswalk.empty:
        return
    sums = crosswalk.groupby("ori9", dropna=False)["weight"].sum()
    bad = sums[(sums - 1.0).abs() > 1e-9]
    if bad.empty:
        return
    raise ValueError(
        f"{len(bad)} ORI(s) have agency->jurisdiction crosswalk weights that do not sum "
        "to 1, so agency mass would be lost or duplicated on its way into a "
        f"jurisdiction target: {bad.head(20).to_dict()}"
    )


# --- the skeleton -----------------------------------------------------------


def build_jurisdiction_ownership(
    *,
    paths: RepoPaths,
    config: JurisdictionTargetConfig = JurisdictionTargetConfig(),
) -> pd.DataFrame:
    """One row per in-scope jurisdiction, saying whether it owns territory and how.

    This is the only place the control universe is decided, and it is decided from
    geometry and agency ownership alone — never from whether an estimate row happens to
    exist. That ordering is what keeps a silent unit in the control panel long enough for
    `benchmark_imputation` to size it.
    """
    master = load_jurisdiction_master(paths)
    master = master[
        ~master["state_abbr"].isin(set(config.exclude_scope_state_abbrs))
    ].copy()

    crosswalk = load_crosswalk(paths)
    _assert_crosswalk_weights_partition_every_agency(crosswalk)
    links = crosswalk[crosswalk["weight"].gt(0.0)]
    agency_counts = links.groupby("jurisdiction_id", dropna=False)["ori9"].agg(
        crosswalk_agency_count="nunique",
        crosswalk_agency_oris=lambda values: "|".join(sorted(str(v) for v in values.dropna().unique())),
    )
    master = master.merge(agency_counts, on="jurisdiction_id", how="left")
    master["crosswalk_agency_count"] = (
        pd.to_numeric(master["crosswalk_agency_count"], errors="coerce").fillna(0).astype(int)
    )
    master["crosswalk_agency_oris"] = master["crosswalk_agency_oris"].astype("string")

    exposure = load_block_group_exposure(paths)
    master = master.merge(exposure, on="jurisdiction_id", how="left")
    master["owns_block_group_geometry"] = master["bucket_population"].notna()
    master["bucket_population"] = (
        pd.to_numeric(master["bucket_population"], errors="coerce").fillna(0.0)
    )
    master["pop_band"] = master["bucket_population"].map(
        lambda value: _assign_pop_band(value, config.pop_bands)
    )

    geometry_lane = master["jurisdiction_type"].isin(GEOMETRY_OWNED_LANES)
    has_agency = master["crosswalk_agency_count"].gt(0)
    master["owns_territory"] = np.where(
        geometry_lane, master["owns_block_group_geometry"], has_agency
    )
    master["ownership_basis"] = np.where(
        geometry_lane, "block_group_geometry", "statewide_overlap_agencies"
    )
    master["exclusion_reason"] = np.select(
        [
            master["owns_territory"],
            geometry_lane & has_agency,
            geometry_lane & ~has_agency,
        ],
        [
            pd.NA,
            "no_block_group_geometry",
            "no_block_group_geometry_and_no_agency",
        ],
        default="no_crosswalked_agency",
    )
    master["exclusion_reason"] = master["exclusion_reason"].astype("string")
    return master.sort_values(
        ["state_fips", "jurisdiction_type", "jurisdiction_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_jurisdiction_control_skeleton(ownership: pd.DataFrame) -> pd.DataFrame:
    """Every territory-owning jurisdiction × every offense, and nothing else."""
    owning = ownership[ownership["owns_territory"]].copy()
    offenses = pd.DataFrame({"offense": list(OFFENSES_7)})
    skeleton = owning.merge(offenses, how="cross")
    columns = [
        *JURISDICTION_IDENTITY_COLUMNS,
        "offense",
        "bucket_population",
        "pop_band",
        "owns_block_group_geometry",
        "ownership_basis",
        "crosswalk_agency_count",
    ]
    return (
        skeleton[columns]
        .sort_values(
            ["state_fips", "jurisdiction_type", "jurisdiction_id", "offense"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_ownership_exclusions(
    *,
    ownership: pd.DataFrame,
    agency_estimates: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """The jurisdictions deliberately left out of the control universe, with the agency
    mass they would have carried.

    Explicitly excluded and enumerated, which is the only defensible treatment: a
    jurisdiction with no territory cannot be allocated onto the surface, and inventing a
    peer count for it (the deleted `peer_state_count_median` rung: 16,863 cells /
    134,209 counts, every one of them a jurisdiction absent from both geometry
    crosswalks) fabricates mass that then has nowhere to go.
    """
    excluded = ownership[~ownership["owns_territory"]].copy()
    if excluded.empty:
        return pd.DataFrame(columns=OWNERSHIP_EXCLUSION_COLUMNS)
    mass = _agency_mass_by_jurisdiction(
        agency_estimates=agency_estimates, crosswalk=crosswalk
    )
    excluded = excluded.merge(mass, on="jurisdiction_id", how="left")
    excluded["agency_estimate_mass"] = (
        pd.to_numeric(excluded["agency_estimate_mass"], errors="coerce").fillna(0.0)
    )
    return excluded[OWNERSHIP_EXCLUSION_COLUMNS].reset_index(drop=True)


def _agency_mass_by_jurisdiction(
    *, agency_estimates: pd.DataFrame, crosswalk: pd.DataFrame
) -> pd.DataFrame:
    if agency_estimates.empty:
        return pd.DataFrame(columns=["jurisdiction_id", "agency_estimate_mass"])
    merged = agency_estimates[["ori9", "estimated_count"]].merge(
        crosswalk[["ori9", "jurisdiction_id", "weight"]], on="ori9", how="inner"
    )
    merged["_mass"] = (
        pd.to_numeric(merged["estimated_count"], errors="coerce").fillna(0.0)
        * merged["weight"]
    )
    return (
        merged.groupby("jurisdiction_id", dropna=False, as_index=False)["_mass"]
        .sum()
        .rename(columns={"_mass": "agency_estimate_mass"})
    )


def _assert_every_agency_estimate_lands_on_the_skeleton(
    *,
    agency_estimates: pd.DataFrame,
    crosswalk: pd.DataFrame,
    skeleton: pd.DataFrame,
) -> None:
    """Fail closed if positive agency mass routes to a jurisdiction with no control row.

    The counterpart of the exclusion artifact: a jurisdiction may be left out of the
    control universe only while it carries nothing. The moment one of the geometry-less
    places acquires a reporting agency, this stops the build instead of dropping the
    mass into the lossy merge that used to swallow 138,530 counts a year.
    """
    if agency_estimates.empty:
        return
    mass = _agency_mass_by_jurisdiction(
        agency_estimates=agency_estimates, crosswalk=crosswalk
    )
    on_skeleton = set(skeleton["jurisdiction_id"].astype("string"))
    stranded = mass[
        ~mass["jurisdiction_id"].astype("string").isin(on_skeleton)
        & mass["agency_estimate_mass"].abs().gt(1e-9)
    ]
    if stranded.empty:
        return
    raise ValueError(
        f"{len(stranded)} jurisdiction(s) outside the control skeleton carry "
        f"{float(stranded['agency_estimate_mass'].sum()):.3f} counts of agency estimate "
        "that can never land on a control row: "
        + str(stranded.head(20).to_dict(orient="records"))
    )


# --- the aggregation --------------------------------------------------------


def _support_weight(allocated_count: pd.Series, weight: pd.Series) -> pd.Series:
    """Weight a jurisdiction's descriptive labels by the mass each agency contributes,
    falling back to the crosswalk weight where the whole cell is zero."""
    support = allocated_count.abs()
    return support.where(support.gt(0.0), weight)


def _dominant_label(
    *,
    frame: pd.DataFrame,
    key_cols: list[str],
    label_col: str,
    weight_col: str,
    output_col: str,
) -> pd.DataFrame:
    labels = frame[label_col].astype("string")
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    mask = labels.notna() & labels.ne("")
    if not mask.any():
        return pd.DataFrame(columns=[*key_cols, output_col])
    support = (
        frame.loc[mask, key_cols]
        .assign(_label=labels.loc[mask].to_numpy(), _weight=weights.loc[mask].to_numpy())
        .groupby([*key_cols, "_label"], dropna=False, as_index=False)["_weight"]
        .sum()
        .sort_values(
            [*key_cols, "_weight", "_label"],
            ascending=[True] * len(key_cols) + [False, True],
            kind="mergesort",
        )
    )
    support["_rank"] = support.groupby(key_cols, dropna=False).cumcount()
    return (
        support[support["_rank"].eq(0)][[*key_cols, "_label"]]
        .rename(columns={"_label": output_col})
        .reset_index(drop=True)
    )


def build_jurisdiction_reported_rollup(
    *,
    agency_panel: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Per (jurisdiction, year, offense): what the agencies of that jurisdiction reported,
    plus the descriptive provenance of that report.

    Reported mass only — no fills, no annualisation, no preference. Every number here is
    a weighted sum of the Stage-1 preferred observation the agency itself published.
    """
    key_cols = ["jurisdiction_id", "year", "offense"]
    if agency_panel.empty:
        return pd.DataFrame(columns=key_cols)

    merged = agency_panel.merge(
        crosswalk[["ori9", "jurisdiction_id", "weight"]], on="ori9", how="inner"
    )
    merged["preferred_count"] = pd.to_numeric(
        merged["preferred_count"], errors="coerce"
    ).fillna(0.0)
    merged["allocated_count"] = merged["preferred_count"] * merged["weight"]
    merged["support_weight"] = _support_weight(
        merged["allocated_count"], merged["weight"]
    )
    obs_weight = pd.to_numeric(merged["preferred_observation_weight"], errors="coerce")
    months = pd.to_numeric(merged["preferred_months_reported"], errors="coerce")
    merged["_obs_num"] = obs_weight.fillna(0.0) * merged["support_weight"]
    merged["_obs_den"] = merged["support_weight"].where(obs_weight.notna(), 0.0)
    merged["_month_num"] = months.fillna(0.0) * merged["support_weight"]
    merged["_month_den"] = merged["support_weight"].where(months.notna(), 0.0)

    grouped = merged.groupby(key_cols, dropna=False, as_index=False).agg(
        reported_count_preferred=("allocated_count", "sum"),
        contributing_agency_count=("ori9", "nunique"),
        _obs_num=("_obs_num", "sum"),
        _obs_den=("_obs_den", "sum"),
        _month_num=("_month_num", "sum"),
        _month_den=("_month_den", "sum"),
    )
    grouped["observation_weight_preferred"] = np.where(
        grouped["_obs_den"].gt(0.0), grouped["_obs_num"] / grouped["_obs_den"], np.nan
    )
    grouped["mean_months_reported_preferred"] = np.where(
        grouped["_month_den"].gt(0.0),
        grouped["_month_num"] / grouped["_month_den"],
        np.nan,
    )
    grouped = grouped.drop(columns=["_obs_num", "_obs_den", "_month_num", "_month_den"])

    for column in LANE_REPORTED_COLUMNS.values():
        lane_values = (
            pd.to_numeric(merged[column], errors="coerce")
            if column in merged.columns
            else pd.Series(np.nan, index=merged.index, dtype=float)
        )
        present = lane_values.notna().to_numpy()
        if not present.any():
            grouped[column] = np.nan
            continue
        lane = merged.loc[present, key_cols].copy()
        lane["_lane_count"] = (lane_values * merged["weight"]).to_numpy()[present]
        rolled = (
            lane.groupby(key_cols, dropna=False, as_index=False)["_lane_count"]
            .sum()
            .rename(columns={"_lane_count": column})
        )
        grouped = grouped.merge(rolled, on=key_cols, how="left")

    for label_col, output_col in (
        ("preferred_source", "preferred_source"),
        ("reporting_regime", "dominant_reporting_regime"),
    ):
        grouped = grouped.merge(
            _dominant_label(
                frame=merged,
                key_cols=key_cols,
                label_col=label_col,
                weight_col="support_weight",
                output_col=output_col,
            ),
            on=key_cols,
            how="left",
        )
    return grouped


def build_jurisdiction_target_components(
    *,
    agency_estimates: pd.DataFrame,
    crosswalk: pd.DataFrame,
    agency_target_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per (jurisdiction, offense): the target and its components, from agency estimates.

    Every column is a weighted sum over the agencies whose territory the jurisdiction
    owns. The three estimate classes are kept apart — the agency's own report, the
    annualisation of its own partial year, and a fill taken from its own recent history —
    because that decomposition IS the control's uplift / fill split, and deriving it here
    is what makes the row identity exact rather than inferred from a jurisdiction-level
    months column.
    """
    key_cols = ["jurisdiction_id", "offense"]
    columns = [
        *key_cols,
        "reported_count_from_estimates",
        "estimated_count",
        "partial_reporting_uplift_count",
        "current_year_fill_count",
        "observed_component_count",
        "partial_component_count",
        "fill_component_count",
        "estimating_agency_count",
        *LANE_TARGET_COMPONENT_COLUMNS,
    ]
    if agency_estimates.empty:
        return pd.DataFrame(columns=columns)

    estimates = agency_estimates
    if "preferred_source" not in estimates.columns and agency_target_panel is not None:
        # The Stage-1 estimate frame carries the amount, the target-year preferred panel
        # carries the lane it came from. Joined here so the row can publish which source
        # lane each component of its target arrived on.
        estimates = estimates.merge(
            agency_target_panel[["ori9", "offense", "preferred_source"]].drop_duplicates(
                subset=["ori9", "offense"]
            ),
            on=["ori9", "offense"],
            how="left",
        )
    merged = estimates.merge(
        crosswalk[["ori9", "jurisdiction_id", "weight"]], on="ori9", how="inner"
    )
    source = merged["agency_estimate_source"].astype("string").fillna("")
    weight = merged["weight"]
    estimated = pd.to_numeric(merged["estimated_count"], errors="coerce").fillna(0.0)
    reported = pd.to_numeric(merged["reported_count_current"], errors="coerce").fillna(0.0)
    adjustment = pd.to_numeric(
        merged["agency_adjustment_count"], errors="coerce"
    ).fillna(0.0)

    is_observed = source.eq(AGENCY_OBSERVED_ESTIMATE_SOURCE)
    is_partial = source.eq(PARTIAL_UPLIFT_ESTIMATE_SOURCE)
    is_fill = ~source.isin(LANE_GROUNDED_ESTIMATE_SOURCES)

    work = merged[key_cols].copy()
    work["reported_count_from_estimates"] = reported * weight
    work["estimated_count"] = estimated * weight
    work["partial_reporting_uplift_count"] = (adjustment * weight).where(is_partial, 0.0)
    work["current_year_fill_count"] = (adjustment * weight).where(is_fill, 0.0)
    work["observed_component_count"] = (estimated * weight).where(is_observed, 0.0)
    work["partial_component_count"] = (estimated * weight).where(is_partial, 0.0)
    work["fill_component_count"] = (estimated * weight).where(is_fill, 0.0)
    work["_ori9"] = merged["ori9"]

    grouped = work.groupby(key_cols, dropna=False, as_index=False).agg(
        reported_count_from_estimates=("reported_count_from_estimates", "sum"),
        estimated_count=("estimated_count", "sum"),
        partial_reporting_uplift_count=("partial_reporting_uplift_count", "sum"),
        current_year_fill_count=("current_year_fill_count", "sum"),
        observed_component_count=("observed_component_count", "sum"),
        partial_component_count=("partial_component_count", "sum"),
        fill_component_count=("fill_component_count", "sum"),
        estimating_agency_count=("_ori9", "nunique"),
    )

    if "preferred_source" in merged.columns:
        lane_frame = merged[key_cols].copy()
        lane_frame["_lane"] = merged["preferred_source"].astype("string").fillna("unknown")
        lane_frame["_mass"] = estimated * weight
        lane_mass = (
            lane_frame.groupby([*key_cols, "_lane"], dropna=False, as_index=False)["_mass"]
            .sum()
            .pivot_table(index=key_cols, columns="_lane", values="_mass", aggfunc="sum")
            .reset_index()
        )
        for source_name, column in LANE_TARGET_COLUMNS.items():
            if source_name in lane_mass.columns:
                lane_mass = lane_mass.rename(columns={source_name: column})
            else:
                lane_mass[column] = 0.0
        grouped = grouped.merge(
            lane_mass[[*key_cols, *LANE_TARGET_COLUMNS.values()]], on=key_cols, how="left"
        )
    for column in LANE_TARGET_COLUMNS.values():
        if column not in grouped.columns:
            grouped[column] = 0.0
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(0.0)
    grouped[OTHER_LANE_TARGET_COLUMN] = (
        grouped["estimated_count"] - grouped[list(LANE_TARGET_COLUMNS.values())].sum(axis=1)
    )
    _assert_lane_components_sum_to_the_target(grouped)
    return grouped[columns]


def build_identity_resolution_adjustments(
    *,
    agency_panel: pd.DataFrame,
    crosswalk: pd.DataFrame,
    succession_ledger: pd.DataFrame | None,
    target_year: int,
) -> pd.DataFrame:
    """Signed reported mass removed because a successor already carries the agency.

    The reported rollup deliberately remains the raw Stage-1 preferred observation.
    The estimate set, however, drops superseded ORIs to avoid double counting. Publishing
    the difference as a named, ledger-derived component keeps both facts visible and
    makes the target identity exact without exempting identity-resolved rows.
    """
    columns = [
        "jurisdiction_id",
        "year",
        "offense",
        IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN,
    ]
    if (
        agency_panel.empty
        or succession_ledger is None
        or succession_ledger.empty
    ):
        return pd.DataFrame(columns=columns)
    if "superseded_ori9" not in succession_ledger.columns:
        raise ValueError(
            "succession ledger is missing superseded_ori9, so identity-resolution "
            "adjustments cannot be derived"
        )

    superseded = set(
        succession_ledger["superseded_ori9"]
        .astype("string")
        .str.upper()
        .dropna()
    )
    work = agency_panel[
        agency_panel["year"].eq(int(target_year))
        & agency_panel["ori9"].astype("string").str.upper().isin(superseded)
    ].merge(
        crosswalk[["ori9", "jurisdiction_id", "weight"]],
        on="ori9",
        how="inner",
    )
    if work.empty:
        return pd.DataFrame(columns=columns)
    work[IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN] = -(
        pd.to_numeric(work["preferred_count"], errors="coerce").fillna(0.0)
        * pd.to_numeric(work["weight"], errors="coerce").fillna(0.0)
    )
    return (
        work.groupby(
            ["jurisdiction_id", "year", "offense"],
            dropna=False,
            as_index=False,
        )[IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN]
        .sum()
        .reindex(columns=columns)
    )


def _assert_lane_components_sum_to_the_target(components: pd.DataFrame) -> None:
    """The per-lane component masses must decompose the target exactly.

    Component provenance is only worth publishing if it is complete: a lane split that
    quietly loses mass is the dominant-label problem in another costume.
    """
    if components.empty:
        return
    residual = (
        components["estimated_count"]
        - components[list(LANE_TARGET_COMPONENT_COLUMNS)].sum(axis=1)
    ).abs()
    bad = components[residual.gt(1e-6)]
    if bad.empty:
        return
    raise ValueError(
        f"{len(bad)} jurisdiction target(s) whose source-lane components do not sum to "
        "the target: " + str(bad.head(20).to_dict(orient="records"))
    )


def _assert_row_identity(targets: pd.DataFrame) -> None:
    """The complete control identity must hold exactly on every row.

    Never asserted before this stage was rewritten, and it did not hold: 27 rows carried
    a target BELOW their own reported count (the municipal ladder's missing floor) and
    1,165 benchmark-imputed rows broke it by design. Here it is an identity of the
    aggregation itself, so a break means Stage 1 dropped reported mass that the estimate
    side never accounted for — for instance an ORI retired by identity resolution whose
    successor does not in fact cover it. Identity-resolved reported mass and benchmark
    imputation are explicit components, not exemptions.
    """
    if targets.empty:
        return
    reported = pd.to_numeric(targets["reported_count_preferred"], errors="coerce").fillna(0.0)
    uplift = pd.to_numeric(targets["partial_reporting_uplift_count"], errors="coerce").fillna(0.0)
    fill = pd.to_numeric(targets["current_year_fill_count"], errors="coerce").fillna(0.0)
    identity_resolution = pd.to_numeric(
        targets.get(
            IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN,
            pd.Series(0.0, index=targets.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    benchmark = pd.to_numeric(
        targets.get(
            "benchmark_imputed_count",
            pd.Series(0.0, index=targets.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    target_column = (
        "adjusted_count_ags_core"
        if "adjusted_count_ags_core" in targets.columns
        else "estimated_count"
    )
    target = pd.to_numeric(targets[target_column], errors="coerce").fillna(0.0)
    residual = (
        target - reported - identity_resolution - uplift - fill - benchmark
    ).abs()
    bad = targets[residual.gt(1e-6)]
    if bad.empty:
        return
    detail_columns = [
        "jurisdiction_id",
        "offense",
        "reported_count_preferred",
        IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN,
        "partial_reporting_uplift_count",
        "current_year_fill_count",
        "benchmark_imputed_count",
        target_column,
    ]
    raise ValueError(
        f"{len(bad)} jurisdiction target(s) violate "
        "target = reported + identity_resolution_adjustment + "
        "partial_reporting_uplift + current_year_fill + benchmark_imputed: "
        + str(
            bad[[column for column in detail_columns if column in bad.columns]]
            .head(20)
            .to_dict(orient="records")
        )
    )


def assert_agency_mass_equals_control_mass(
    *,
    agency_estimates: pd.DataFrame,
    crosswalk: pd.DataFrame,
    targets: pd.DataFrame,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Fail closed unless Σ(agency estimate × weight) == pre-imputation target, per
    state × offense × jurisdiction lane.

    The identity the whole restructure exists to create. Before it, municipal targets ran
    46,338 counts above the agency mass they claimed to aggregate (the municipal ladder's
    own extra fills) and the statewide overlap layer ran 21,757 counts below it (an
    override written for the remainder pool and never extended), and nothing in the build
    noticed either.
    """
    lane_key = ["state_fips", "jurisdiction_type", "offense"]
    control_side = (
        targets.groupby(lane_key, dropna=False, as_index=False)["estimated_count"]
        .sum()
        .rename(columns={"estimated_count": "control_mass"})
    )
    if agency_estimates.empty:
        agency_side = pd.DataFrame(columns=[*lane_key, "agency_mass"])
    else:
        merged = (
            agency_estimates[["ori9", "offense", "estimated_count"]]
            .merge(crosswalk[["ori9", "jurisdiction_id", "weight"]], on="ori9", how="inner")
            .merge(
                targets[
                    ["jurisdiction_id", "jurisdiction_type", "state_fips"]
                ].drop_duplicates(subset=["jurisdiction_id"]),
                on="jurisdiction_id",
                how="inner",
            )
        )
        merged["_mass"] = (
            pd.to_numeric(merged["estimated_count"], errors="coerce").fillna(0.0)
            * merged["weight"]
        )
        agency_side = (
            merged.groupby(lane_key, dropna=False, as_index=False)["_mass"]
            .sum()
            .rename(columns={"_mass": "agency_mass"})
        )
    reconciliation = control_side.merge(agency_side, on=lane_key, how="outer")
    for column in ("control_mass", "agency_mass"):
        reconciliation[column] = pd.to_numeric(
            reconciliation[column], errors="coerce"
        ).fillna(0.0)
    reconciliation["delta"] = (
        reconciliation["control_mass"] - reconciliation["agency_mass"]
    )
    bad = reconciliation[reconciliation["delta"].abs().gt(tolerance)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} state x offense x lane cell(s) where the pre-imputation control "
            "mass does not equal the agency-estimate mass it aggregates: "
            + str(bad.head(20).to_dict(orient="records"))
        )
    return reconciliation


# --- the assembled targets --------------------------------------------------


def _estimate_source_from_composition(targets: pd.DataFrame) -> pd.Series:
    has_agency = pd.to_numeric(
        targets["estimating_agency_count"], errors="coerce"
    ).fillna(0).gt(0)
    fill = pd.to_numeric(targets["current_year_fill_count"], errors="coerce").fillna(0.0)
    uplift = pd.to_numeric(
        targets["partial_reporting_uplift_count"], errors="coerce"
    ).fillna(0.0)
    return pd.Series(
        np.select(
            [~has_agency, fill.gt(1e-12), uplift.gt(1e-12)],
            [TARGET_SOURCE_NO_EVIDENCE, TARGET_SOURCE_FILL, TARGET_SOURCE_PARTIAL_UPLIFT],
            default=TARGET_SOURCE_OBSERVED,
        ),
        index=targets.index,
        dtype="string",
    )


def _confidence_from_source(source: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                source.isin([TARGET_SOURCE_OBSERVED, HISTORY_SOURCE_REPORTED_ROLLUP]),
                source.eq(TARGET_SOURCE_PARTIAL_UPLIFT),
            ],
            ["high", "medium"],
            default="low",
        ),
        index=source.index,
        dtype="string",
    )


def _attach_descriptive_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    source = out["preferred_source"].astype("string")
    out["preferred_source_family"] = source.map(source_family_from_source).astype("string")
    out["preferred_source_origin"] = source.map(
        lambda value: source_origin_from_parts(source=value) if pd.notna(value) else pd.NA
    ).astype("string")
    out["preferred_raw_data_source"] = source.map(
        lambda value: raw_data_source_from_parts(source=value) if pd.notna(value) else pd.NA
    ).astype("string")
    out["preferred_source_lane"] = source.map(source_lane_from_source).astype("string")
    out["preferred_reporting_mode"] = source.map(reporting_mode_from_source).astype("string")
    out["preferred_conversion_status"] = source.map(
        default_conversion_status_from_source
    ).astype("string")
    out["preferred_state_exception_flag"] = source.eq(STATE_PUBLICATION_SOURCE).fillna(False)
    out["preferred_cius_reference_flag"] = source.eq(CIUS_SOURCE).fillna(False)
    return out


def _attach_crosswalk_relationship(
    *, frame: pd.DataFrame, crosswalk: pd.DataFrame
) -> pd.DataFrame:
    """`relationship_type` / `overlap_subtype` are properties of the jurisdiction's own
    agency footprint, not of any source lane, so they are read straight off the crosswalk
    instead of being carried through a per-lane preference. A jurisdiction with no agency
    takes its lane's structural default."""
    out = frame.copy()
    links = crosswalk[crosswalk["weight"].gt(0.0)].copy()
    for label_col, output_col in (
        ("relationship_type", "relationship_type_preferred"),
        ("overlap_subtype", "overlap_subtype_preferred"),
    ):
        if label_col not in links.columns:
            out[output_col] = pd.NA
            continue
        dominant = _dominant_label(
            frame=links,
            key_cols=["jurisdiction_id"],
            label_col=label_col,
            weight_col="weight",
            output_col=output_col,
        )
        out = out.merge(dominant, on="jurisdiction_id", how="left")
        out[output_col] = out[output_col].astype("string")
    structural_default = np.where(
        out["jurisdiction_type"].eq(STATE_OVERLAP_TYPE), "overlap", "exclusive"
    )
    out["relationship_type_preferred"] = pd.Series(
        np.where(
            out["relationship_type_preferred"].isna(),
            structural_default,
            out["relationship_type_preferred"].to_numpy(),
        ),
        index=out.index,
        dtype="string",
    )
    return out


def build_jurisdiction_year_estimates(
    *,
    paths: RepoPaths,
    config: JurisdictionTargetConfig = JurisdictionTargetConfig(),
    agency_panel: pd.DataFrame | None = None,
    agency_estimates: pd.DataFrame,
    ownership: pd.DataFrame | None = None,
    succession_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The jurisdiction-year panel: reported rollup for every year, consumed agency
    estimates for the target year.

    History years carry the reported rollup and nothing else. There is no jurisdiction
    fill for a past year because there is no jurisdiction estimator: a year in which an
    agency did not report is a year with no jurisdiction-level evidence, and inventing
    one here would put the deleted ladder back in the reference set the target year is
    read against.
    """
    crosswalk = load_crosswalk(paths)
    if ownership is None:
        ownership = build_jurisdiction_ownership(paths=paths, config=config)
    skeleton = build_jurisdiction_control_skeleton(ownership)
    if agency_panel is None:
        agency_panel = build_agency_trend_fill_panel(
            paths=paths,
            year_start=int(config.year_start),
            year_end=int(config.target_year),
            force_reporting_regimes_rebuild=bool(config.force_reporting_regimes_rebuild),
            exclude_state_abbrs=tuple(config.exclude_scope_state_abbrs),
        )

    years = pd.DataFrame(
        {"year": list(range(int(config.year_start), int(config.target_year) + 1))}
    )
    panel = skeleton.merge(years, how="cross")

    rollup = build_jurisdiction_reported_rollup(
        agency_panel=agency_panel, crosswalk=crosswalk
    )
    panel = panel.merge(rollup, on=["jurisdiction_id", "year", "offense"], how="left")
    panel["reported_count_preferred"] = pd.to_numeric(
        panel["reported_count_preferred"], errors="coerce"
    ).fillna(0.0)
    panel["contributing_agency_count"] = (
        pd.to_numeric(panel["contributing_agency_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    components = build_jurisdiction_target_components(
        agency_estimates=agency_estimates,
        crosswalk=crosswalk,
        agency_target_panel=agency_panel[
            agency_panel["year"].eq(int(config.target_year))
        ]
        if not agency_panel.empty
        else None,
    )
    panel = panel.merge(
        components.assign(year=int(config.target_year)),
        on=["jurisdiction_id", "year", "offense"],
        how="left",
    ).reset_index(drop=True)
    identity_resolution = build_identity_resolution_adjustments(
        agency_panel=agency_panel,
        crosswalk=crosswalk,
        succession_ledger=succession_ledger,
        target_year=int(config.target_year),
    )
    panel = panel.merge(
        identity_resolution,
        on=["jurisdiction_id", "year", "offense"],
        how="left",
    ).reset_index(drop=True)
    is_target = panel["year"].eq(int(config.target_year))
    for column in [
        "reported_count_from_estimates",
        "estimated_count",
        IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN,
        "partial_reporting_uplift_count",
        "current_year_fill_count",
        "observed_component_count",
        "partial_component_count",
        "fill_component_count",
        *LANE_TARGET_COMPONENT_COLUMNS,
    ]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
    panel["estimating_agency_count"] = (
        pd.to_numeric(panel["estimating_agency_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # History years publish the reported rollup itself; the target year publishes the
    # consumed agency estimate. Both are sums of agency-level facts.
    panel.loc[~is_target, "estimated_count"] = panel.loc[
        ~is_target, "reported_count_preferred"
    ]
    panel.loc[~is_target, "observed_component_count"] = panel.loc[
        ~is_target, "reported_count_preferred"
    ]

    panel["estimate_source"] = pd.Series(
        HISTORY_SOURCE_REPORTED_ROLLUP, index=panel.index, dtype="string"
    )
    panel.loc[is_target, "estimate_source"] = _estimate_source_from_composition(
        panel.loc[is_target]
    )
    panel["estimate_confidence"] = _confidence_from_source(panel["estimate_source"])
    panel.loc[
        ~is_target & panel["contributing_agency_count"].eq(0), "estimate_confidence"
    ] = "low"
    # "Not the agencies' own reported number." Two ways to earn it: the row carries
    # uplift or fill mass, or it carries no agency evidence at all -- a skeleton row at
    # zero is the absence of an observation, not an observation of zero, and must never
    # reach a downstream model as training data.
    adjustment_mass = pd.to_numeric(
        panel["partial_reporting_uplift_count"], errors="coerce"
    ).fillna(0.0) + pd.to_numeric(panel["current_year_fill_count"], errors="coerce").fillna(0.0)
    has_evidence = np.where(
        is_target.to_numpy(),
        panel["estimating_agency_count"].gt(0).to_numpy(),
        panel["contributing_agency_count"].gt(0).to_numpy(),
    )
    panel["estimated_from_panel"] = adjustment_mass.gt(1e-12) | ~pd.Series(
        has_evidence, index=panel.index
    )

    panel = _attach_descriptive_provenance(panel)
    panel = _attach_crosswalk_relationship(frame=panel, crosswalk=crosswalk)
    panel = panel.reset_index(drop=True)
    panel["quality_tier_preferred"] = _quality_tier_from_months(
        panel["mean_months_reported_preferred"]
    )

    target_rows = panel[panel["year"].eq(int(config.target_year))]
    _assert_row_identity(target_rows)
    _assert_every_agency_estimate_lands_on_the_skeleton(
        agency_estimates=agency_estimates, crosswalk=crosswalk, skeleton=skeleton
    )
    assert_agency_mass_equals_control_mass(
        agency_estimates=agency_estimates,
        crosswalk=crosswalk,
        targets=target_rows,
    )
    return panel.sort_values(
        ["state_fips", "jurisdiction_type", "jurisdiction_id", "year", "offense"],
        kind="mergesort",
    ).reset_index(drop=True)


def _quality_tier_from_months(months: pd.Series) -> pd.Series:
    """The Stage-1 tier ladder, applied to the jurisdiction's own mass-weighted coverage.

    Read off the aggregate months rather than carried over from whichever lane a
    per-offense preference happened to pick: the shipped control panel had 289 rows /
    201,621 counts whose published quality tier came from a different lane than the count
    it describes, because the tier was written by a construction the count then overrode.
    """
    months_num = pd.to_numeric(months, errors="coerce")
    out = pd.Series("unknown", index=months_num.index, dtype="string")
    out.loc[months_num.between(1, 5, inclusive="both")] = "sparse"
    out.loc[months_num.between(6, 9, inclusive="both")] = "low"
    out.loc[months_num.between(10, 11, inclusive="both")] = "medium"
    out.loc[months_num >= 12] = "high"
    return out


def build_agency_target_panel_slice(
    *, agency_panel: pd.DataFrame, target_year: int
) -> pd.DataFrame:
    """The target-year preferred panel with the Stage-1 support flags attached, which is
    what `benchmark_imputation` reads to decide which agencies are silent."""
    if agency_panel.empty:
        return agency_panel
    target = agency_panel[agency_panel["year"].eq(int(target_year))].copy()
    if "usable_as_observed" in target.columns:
        return target
    return add_preferred_support_flags(target)


def write_jurisdiction_ownership_exclusions(
    exclusions: pd.DataFrame, *, paths: RepoPaths, year: int
) -> Path:
    path = (
        paths.state_dir
        / "controls"
        / f"jurisdiction_ownership_exclusions_{int(year)}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    exclusions.to_parquet(path, index=False)
    return path
