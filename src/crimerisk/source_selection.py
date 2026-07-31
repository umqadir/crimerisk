"""Choosing which lane speaks for an agency-year.

Selection is per (`ori9`, year), not per (`ori9`, offense). The panel keeps up to
five candidate lanes side by side and each of them renders the SAME agency-year;
picking a different renderer for each offense produced agency-years assembled out
of two incompatible accounts of the same twelve months -- 1,412 offenses read from
NIBRS as full-year observed zeros while 1,703 offenses of the same agency-years
were read from Return A and annualised by x12/months (3,213 rows over 459
agencies in 2024; 2,595 agencies carried more than one preferred source across
their offenses). Every per-offense rule that existed to patch that -- prefer NIBRS
when the Return A regime is `true_partial`, prefer NIBRS when Return A reports
exactly zero -- was a workaround for the missing agency-year decision, and all of
them are gone: with coherent zero and partial semantics in both lanes, a zero in
the chosen lane is a zero and a partial year in the chosen lane annualises.

The rule: publication lanes keep their standing as complete annual compilations
(CIUS, then local, then state); otherwise the lane that covered more of the year
wins, ties going to the heavier observation weight, then to the FBI's published
NIBRS tables corroborating the rollup, then to the standing lane order. A manual
`configs/source_preference_overrides.csv` entry names the lane for the whole
agency-year.

One bounded exception, annotated per row rather than silent: the publication
lanes do not always publish every offense -- CIUS withholds individual cells
("The FBI determined that the agency's data were overreported. Consequently,
those data are not included in this table"), the MS TOPS extract suppresses
columns -- and a withheld cell is missing, not zero. An offense the chosen lane
does not carry falls to the next lane in the same agency-year ranking and is
marked `preferred_source_is_lane_supplement`. The Return A and NIBRS lanes always
carry all seven offenses (asserted at panel build), so a supplement can only ever
arise under a publication lane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.paths import RepoPaths
from crimerisk.published_nibrs import (
    build_published_nibrs_corroboration_mask,
    load_published_nibrs_reference_counts,
)
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    CIUS_ORIGIN,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    assign_preferred_value,
    default_conversion_status_from_source,
    raw_data_source_from_parts,
    reporting_mode_from_source,
    source_family_from_source,
    source_lane_from_source,
    source_origin_from_parts,
)
from crimerisk.reporting_regimes import (
    ReportingRegimeBuildConfig,
    get_v2_reporting_regimes_path,
    reporting_regimes_artifact_is_current,
    write_v2_reporting_regimes,
)


def _normalize_optional_panel_schema(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    numeric_cols = [
        "reported_count_local_publication",
        "observation_weight_local_publication",
        "mean_months_reported_local_publication",
        "reported_count_state_publication",
        "observation_weight_state_publication",
        "mean_months_reported_state_publication",
        "published_nibrs_official_count",
    ]
    string_cols = [
        "conversion_status_local_publication",
        "conversion_status_state_publication",
        "published_nibrs_agency_type",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in string_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def _load_agency_year_observations(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    return pd.read_parquet(path)


def _load_reporting_regimes(
    paths: RepoPaths,
    *,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    path = get_v2_reporting_regimes_path(paths)
    config = ReportingRegimeBuildConfig()
    if not force_rebuild and reporting_regimes_artifact_is_current(
        paths, config=config, out_path=path
    ):
        return pd.read_parquet(path)
    write_v2_reporting_regimes(paths=paths, out_path=path, config=config)
    return pd.read_parquet(path)


def _globally_dead_observation_oris(agency_obs: pd.DataFrame) -> set[str]:
    if agency_obs.empty:
        return set()
    stats = agency_obs[["ori9", "count", "months_reported"]].copy()
    stats["count"] = pd.to_numeric(stats["count"], errors="coerce").fillna(0.0)
    stats["months_reported"] = pd.to_numeric(
        stats["months_reported"], errors="coerce"
    ).fillna(0.0)
    stats = (
        stats.groupby("ori9", dropna=False)
        .agg(total_count=("count", "sum"), max_months=("months_reported", "max"))
        .reset_index()
    )
    dead = stats[
        stats["ori9"].notna()
        & stats["total_count"].le(0.0)
        & stats["max_months"].le(0.0)
    ]["ori9"].astype(str)
    return set(dead.tolist())


def globally_dead_municipal_jurisdiction_ids(
    *,
    agency_obs: pd.DataFrame,
    agency_jurisdiction_crosswalk: pd.DataFrame,
    jurisdiction_master: pd.DataFrame,
) -> frozenset[str]:
    """Municipal jurisdictions that are not a valid totals source.

    A municipal jurisdiction is not a valid totals source when every ORI the
    agency-to-jurisdiction crosswalk links to it counts as dead: either
    globally dead (see `_globally_dead_observation_oris`: zero reported count
    AND zero reported months across all years) -- e.g. Patchogue Village NY's
    defunct NY0511000, a village PD with no current DCJS roster entry, now
    policed by Suffolk County PD -- or entirely absent from the observations
    panel (no rows at all, in any year, in any source: never had a reportable
    submission to begin with). Such a jurisdiction contributes no rows to the
    canonical preferred-observations panel and would otherwise publish
    all-zero controls with zero months reported: not a real zero-crime signal
    but a dead source.

    This is intentionally conservative: a jurisdiction with SOME nonzero
    reporting history in an earlier year (a live agency that simply isn't
    reporting the current target year) is left untouched here and remains a
    candidate for the existing masked-gap / trend-fill machinery instead of
    this re-routing rule.

    Callers should treat the returned jurisdiction_ids as invalid BG-assignment
    targets: BGs that would otherwise fall inside one of these jurisdictions'
    polygons must instead route to the covering geography (county remainder /
    state_nonmunicipal_remainder), exactly as if the place were unincorporated.
    """
    dead_oris = _globally_dead_observation_oris(agency_obs)
    observed_oris = (
        set(agency_obs["ori9"].dropna().astype(str)) if not agency_obs.empty else set()
    )
    municipal_ids = set(
        jurisdiction_master.loc[
            jurisdiction_master["jurisdiction_type"].astype("string").eq("municipal"),
            "jurisdiction_id",
        ].astype("string")
    )
    if not municipal_ids:
        return frozenset()
    cw = agency_jurisdiction_crosswalk.copy()
    ori_col = "ori9" if "ori9" in cw.columns else "ori"
    cw[ori_col] = cw[ori_col].astype("string")
    cw["jurisdiction_id"] = cw["jurisdiction_id"].astype("string")
    cw = cw[cw["jurisdiction_id"].isin(municipal_ids)]
    if cw.empty:
        return frozenset()
    linked_oris = cw.groupby("jurisdiction_id")[ori_col].apply(
        lambda oris: set(oris.dropna().astype(str))
    )
    invalid = linked_oris[
        linked_oris.apply(
            lambda oris: len(oris) > 0
            and all(ori in dead_oris or ori not in observed_oris for ori in oris)
        )
    ]
    return frozenset(invalid.index.astype(str))


PREFERRED_LANE_COLUMNS = {
    CIUS_SOURCE: ("reported_count_cius", "mean_months_reported_cius"),
    LOCAL_PUBLICATION_SOURCE: (
        "reported_count_local_publication",
        "mean_months_reported_local_publication",
    ),
    STATE_PUBLICATION_SOURCE: (
        "reported_count_state_publication",
        "mean_months_reported_state_publication",
    ),
    SUMMARY_SOURCE: ("reported_count_srs", "mean_months_reported_srs"),
    NIBRS_SOURCE: ("reported_count_nibrs", "mean_months_reported_nibrs"),
}

LANE_WEIGHT_COLUMNS = {
    CIUS_SOURCE: "observation_weight_cius",
    LOCAL_PUBLICATION_SOURCE: "observation_weight_local_publication",
    STATE_PUBLICATION_SOURCE: "observation_weight_state_publication",
    SUMMARY_SOURCE: "observation_weight_srs",
    NIBRS_SOURCE: "observation_weight_nibrs",
}

# Standing lane order, used as the final tiebreak and as the order in which a
# supplement looks for an offense the chosen lane does not publish.
LANE_STANDING_ORDER = (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    NIBRS_SOURCE,
)
# Lanes that publish a completed annual compilation rather than a monthly
# submission stream. Their standing above the two FBI lanes is the existing rule
# and is preserved: a published municipal total is a stronger statement about the
# year than either federal file's rendering of it.
PUBLICATION_LANES = frozenset(
    {CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE}
)

LANE_SELECTION_REASON_OVERRIDE = "manual_source_override"
LANE_SELECTION_REASON_ONLY_LANE = "only_lane_present"
LANE_SELECTION_REASON_PUBLICATION = "publication_lane_standing"
LANE_SELECTION_REASON_COVERAGE = "greater_month_coverage"
LANE_SELECTION_REASON_WEIGHT = "greater_observation_weight"
LANE_SELECTION_REASON_CORROBORATION = "published_nibrs_corroborates_the_rollup"
LANE_SELECTION_REASON_STANDING = "standing_lane_order"


def _agency_year_lane_signals(
    panel: pd.DataFrame, *, corroborated: pd.Series
) -> pd.DataFrame:
    """One row per (agency, candidate lane) with the lane's agency-year signals.

    Coverage and weight are agency-year properties on every lane (Return A and
    NIBRS derive them from the agency-year's month metadata, the publication lanes
    assert them), so the maximum over the agency's offenses is that property, not
    an aggregate of differing values. Corroboration is per offense by nature and
    counts for the agency-year when any offense is corroborated.
    """
    rows: list[pd.DataFrame] = []
    for lane_index, source in enumerate(LANE_STANDING_ORDER):
        count_col, months_col = PREFERRED_LANE_COLUMNS[source]
        weight_col = LANE_WEIGHT_COLUMNS[source]
        lane = pd.DataFrame(
            {
                "ori9": panel["ori9"],
                "source": source,
                "_present": pd.to_numeric(panel[count_col], errors="coerce").notna(),
                "_months": pd.to_numeric(panel[months_col], errors="coerce"),
                "_weight": pd.to_numeric(panel.get(weight_col), errors="coerce"),
                "_corroborated": (
                    corroborated if source == NIBRS_SOURCE else pd.Series(False, index=panel.index)
                ),
            }
        )
        lane["_standing"] = lane_index
        rows.append(lane)
    long = pd.concat(rows, ignore_index=True)
    signals = long.groupby(["ori9", "source"], dropna=False, as_index=False).agg(
        present=("_present", "max"),
        months=("_months", "max"),
        weight=("_weight", "max"),
        corroborated=("_corroborated", "max"),
        standing=("_standing", "min"),
    )
    signals = signals[signals["present"].fillna(False).astype(bool)].copy()
    signals["months"] = pd.to_numeric(signals["months"], errors="coerce").fillna(0.0)
    signals["weight"] = pd.to_numeric(signals["weight"], errors="coerce").fillna(0.0)
    signals["corroborated"] = signals["corroborated"].fillna(False).astype(bool)
    signals["is_publication"] = signals["source"].isin(PUBLICATION_LANES)
    return signals


def _rank_agency_year_lanes(
    signals: pd.DataFrame, *, override_lane: pd.Series
) -> pd.DataFrame:
    """Rank every present lane within its agency-year; rank 0 is the chosen lane."""
    ranked = signals.copy()
    override = ranked["ori9"].map(override_lane)
    ranked["is_override"] = override.notna() & override.eq(ranked["source"])
    ranked = ranked.sort_values(
        ["ori9", "is_override", "is_publication", "months", "weight", "corroborated", "standing"],
        ascending=[True, False, False, False, False, False, True],
        kind="mergesort",
    )
    ranked["lane_rank"] = ranked.groupby("ori9", sort=False).cumcount()
    return ranked.reset_index(drop=True)


def _lane_selection_reason(ranked: pd.DataFrame) -> pd.Series:
    """Why the winner beat the runner-up, recorded per agency-year."""
    winner = ranked[ranked["lane_rank"].eq(0)].set_index("ori9")
    runner = ranked[ranked["lane_rank"].eq(1)].set_index("ori9")
    reason = pd.Series(LANE_SELECTION_REASON_ONLY_LANE, index=winner.index, dtype="string")
    contested = winner.index.intersection(runner.index)
    if len(contested) == 0:
        return reason
    win = winner.loc[contested]
    run = runner.loc[contested]
    reason.loc[contested] = np.select(
        [
            win["is_override"].to_numpy(),
            (win["is_publication"] & ~run["is_publication"]).to_numpy(),
            win["months"].gt(run["months"]).to_numpy(),
            win["weight"].gt(run["weight"]).to_numpy(),
            (win["corroborated"] & ~run["corroborated"]).to_numpy(),
        ],
        [
            LANE_SELECTION_REASON_OVERRIDE,
            LANE_SELECTION_REASON_PUBLICATION,
            LANE_SELECTION_REASON_COVERAGE,
            LANE_SELECTION_REASON_WEIGHT,
            LANE_SELECTION_REASON_CORROBORATION,
        ],
        default=LANE_SELECTION_REASON_STANDING,
    )
    return reason


def _assert_one_preferred_row_per_agency_offense(panel: pd.DataFrame) -> None:
    """Fail closed if an agency-offense resolves to more than one preferred row.

    The preferred panel is one row per (`ori9`, offense) by construction: the five
    lane pivots are keyed (`ori9`, state, offense) and outer-joined. A duplicate means
    an ORI is carrying two identities -- two state codes, or the split identity a
    half-applied twin merge leaves behind -- and every downstream per-agency join
    would silently double that agency's mass.
    """
    if panel.empty:
        return
    duplicated = panel.duplicated(subset=["ori9", "offense"], keep=False)
    if not duplicated.any():
        return
    offenders = panel.loc[duplicated, ["ori9", "state_fips", "state_abbr", "offense"]]
    raise ValueError(
        f"{int(duplicated.sum())} preferred row(s) across "
        f"{offenders['ori9'].nunique()} agenc(ies) share an (ori9, offense) key: "
        + str(offenders.head(20).to_dict(orient="records"))
    )


def _assert_preferred_lane_unity(panel: pd.DataFrame) -> None:
    """Fail closed if an agency-year is assembled out of more than one lane.

    The chosen lane is a property of the agency-year, so every offense must read
    from it. The single annotated exception is a supplement: an offense the chosen
    lane does not publish at all, which the ledger column makes visible per row.
    """
    if panel.empty:
        return
    lanes_per_agency = panel.groupby("ori9", dropna=False)[
        "preferred_lane_for_agency_year"
    ].nunique()
    split = lanes_per_agency[lanes_per_agency.gt(1)]
    if not split.empty:
        raise ValueError(
            f"{len(split)} agency-year(s) resolved to more than one lane: "
            + str(sorted(split.index.astype(str))[:20])
        )
    unannotated = panel[
        ~panel["preferred_source"].eq(panel["preferred_lane_for_agency_year"])
        & ~panel["preferred_source_is_lane_supplement"].fillna(False).astype(bool)
    ]
    if not unannotated.empty:
        raise ValueError(
            f"{len(unannotated)} preferred row(s) read from a lane other than their "
            "agency-year's chosen lane without a supplement annotation: "
            + str(
                unannotated[
                    ["ori9", "offense", "preferred_source", "preferred_lane_for_agency_year"]
                ]
                .head(20)
                .to_dict(orient="records")
            )
        )


def _assert_preferred_metadata_matches_the_selected_lane(panel: pd.DataFrame) -> None:
    """Fail closed if a preferred count and its months come from different lanes.

    Attaching one lane's participation metadata to another lane's count is what turns a
    complete year into an apparent partial one and licenses a spurious annualization
    (Kenedy's 12-month NIBRS count read against the SRS lane's 2 months). The pairing is
    true by construction today; asserting it keeps it that way.
    """
    mismatched = pd.Series(False, index=panel.index)
    for source, (count_col, months_col) in PREFERRED_LANE_COLUMNS.items():
        selected = panel["preferred_source"].eq(source)
        if not selected.any():
            continue
        for preferred_col, lane_col in [
            ("preferred_count", count_col),
            ("preferred_months_reported", months_col),
        ]:
            preferred = pd.to_numeric(panel[preferred_col], errors="coerce")
            lane = pd.to_numeric(panel[lane_col], errors="coerce")
            mismatched |= selected & ~(
                (preferred == lane) | (preferred.isna() & lane.isna())
            )
    if not mismatched.any():
        return
    raise ValueError(
        f"{int(mismatched.sum())} preferred agency observation(s) carry count or months "
        "metadata from a lane other than the selected one: "
        + str(
            panel.loc[
                mismatched,
                [
                    "ori9",
                    "offense",
                    "preferred_source",
                    "preferred_count",
                    "preferred_months_reported",
                ],
            ]
            .head(20)
            .to_dict(orient="records")
        )
    )


def build_agency_preferred_observations(
    *,
    paths: RepoPaths,
    year: int,
    force_reporting_regimes_rebuild: bool = False,
) -> pd.DataFrame:
    all_agency_obs = _load_agency_year_observations(paths)
    dead_oris = _globally_dead_observation_oris(all_agency_obs)
    agency_obs = all_agency_obs[
        (all_agency_obs["year"].astype(int) == int(year))
        & all_agency_obs["source"].isin(
            [
                CIUS_SOURCE,
                LOCAL_PUBLICATION_SOURCE,
                STATE_PUBLICATION_SOURCE,
                SUMMARY_SOURCE,
                NIBRS_SOURCE,
            ]
        )
    ].copy()
    if dead_oris:
        agency_obs = agency_obs[
            ~agency_obs["ori9"].astype("string").isin(sorted(dead_oris))
        ].copy()
    if agency_obs.empty:
        return pd.DataFrame(
            columns=[
                "ori9",
                "state_fips",
                "state_abbr",
                "offense",
                "preferred_source",
                "preferred_source_family",
                "preferred_source_origin",
                "preferred_raw_data_source",
                "preferred_count",
                "preferred_observation_weight",
                "preferred_months_reported",
                "preferred_source_lane",
                "preferred_reporting_mode",
                "preferred_conversion_status",
                "preferred_state_exception_flag",
                "preferred_cius_reference_flag",
                "preferred_lane_for_agency_year",
                "preferred_lane_selection_reason",
                "preferred_source_is_lane_supplement",
                "reporting_regime",
                "preferred_source_by_regime",
                "reported_count_local_publication",
                "reported_count_srs",
                "reported_count_nibrs",
                "mean_months_reported_local_publication",
                "mean_months_reported_srs",
                "mean_months_reported_nibrs",
            ]
        )

    agency_obs["state_fips"] = agency_obs["state_fips"].astype("string").str.zfill(2)
    agency_obs["state_abbr"] = agency_obs["state_abbr"].astype("string").str.upper()
    key_cols = ["ori9", "state_fips", "state_abbr", "offense"]

    srs = agency_obs[agency_obs["source"].eq(SUMMARY_SOURCE)].rename(
        columns={
            "count": "reported_count_srs",
            "observation_weight": "observation_weight_srs",
            "months_reported": "mean_months_reported_srs",
            "conversion_status": "conversion_status_srs",
            "cius_reference_flag": "cius_reference_flag_summary",
        }
    )[
        key_cols
        + [
            "reported_count_srs",
            "observation_weight_srs",
            "mean_months_reported_srs",
            "conversion_status_srs",
            "cius_reference_flag_summary",
        ]
    ]
    cius = agency_obs[agency_obs["source"].eq(CIUS_SOURCE)].rename(
        columns={
            "count": "reported_count_cius",
            "observation_weight": "observation_weight_cius",
            "months_reported": "mean_months_reported_cius",
            "conversion_status": "conversion_status_cius",
            "cius_reference_flag": "cius_reference_flag_cius",
        }
    )[
        key_cols
        + [
            "reported_count_cius",
            "observation_weight_cius",
            "mean_months_reported_cius",
            "conversion_status_cius",
            "cius_reference_flag_cius",
        ]
    ]
    local_pub = agency_obs[agency_obs["source"].eq(LOCAL_PUBLICATION_SOURCE)].rename(
        columns={
            "count": "reported_count_local_publication",
            "observation_weight": "observation_weight_local_publication",
            "months_reported": "mean_months_reported_local_publication",
            "conversion_status": "conversion_status_local_publication",
        }
    )[
        key_cols
        + [
            "reported_count_local_publication",
            "observation_weight_local_publication",
            "mean_months_reported_local_publication",
            "conversion_status_local_publication",
        ]
    ]
    state_pub = agency_obs[agency_obs["source"].eq(STATE_PUBLICATION_SOURCE)].rename(
        columns={
            "count": "reported_count_state_publication",
            "observation_weight": "observation_weight_state_publication",
            "months_reported": "mean_months_reported_state_publication",
            "conversion_status": "conversion_status_state_publication",
        }
    )[
        key_cols
        + [
            "reported_count_state_publication",
            "observation_weight_state_publication",
            "mean_months_reported_state_publication",
            "conversion_status_state_publication",
        ]
    ]
    nibrs = agency_obs[agency_obs["source"].eq(NIBRS_SOURCE)].rename(
        columns={
            "count": "reported_count_nibrs",
            "observation_weight": "observation_weight_nibrs",
            "months_reported": "mean_months_reported_nibrs",
        }
    )[
        key_cols
        + [
            "reported_count_nibrs",
            "observation_weight_nibrs",
            "mean_months_reported_nibrs",
        ]
    ]
    panel = (
        cius.merge(local_pub, on=key_cols, how="outer")
        .merge(state_pub, on=key_cols, how="outer")
        .merge(srs, on=key_cols, how="outer")
        .merge(nibrs, on=key_cols, how="outer")
    )

    reporting_regimes = _load_reporting_regimes(
        paths,
        force_rebuild=force_reporting_regimes_rebuild,
    )
    regimes = reporting_regimes[reporting_regimes["year"].astype(int).eq(int(year))][
        [
            "ori9",
            "offense",
            "reporting_regime",
            "preferred_source_by_regime",
            "srs_months_reported",
            "nibrs_months_reported",
            "srs_observation_weight",
            "nibrs_observation_weight",
            "source_override_applied",
        ]
    ].copy()
    panel = panel.merge(regimes, on=["ori9", "offense"], how="left")
    panel = panel.merge(
        load_published_nibrs_reference_counts(paths, year=year),
        on=["ori9", "state_abbr", "offense"],
        how="left",
    )
    panel = _normalize_optional_panel_schema(panel)

    # The lane metadata columns the regime table carries duplicate the panel's own
    # per-lane months and weights; the panel's are used directly so the signals the
    # agency-year decision reads come from one place.
    nibrs_months = pd.to_numeric(
        panel["nibrs_months_reported"], errors="coerce"
    ).fillna(
        pd.to_numeric(panel["mean_months_reported_nibrs"], errors="coerce").fillna(0.0)
    )
    published_nibrs_supports_nibrs = (
        panel["reported_count_nibrs"].notna()
        & build_published_nibrs_corroboration_mask(
            nibrs_count=panel["reported_count_nibrs"],
            published_nibrs_count=panel["published_nibrs_official_count"],
            nibrs_months=nibrs_months,
            srs_count=panel["reported_count_srs"],
        )
    )
    manual_source_override = (
        panel["source_override_applied"].astype("boolean").fillna(False).astype(bool)
    )
    # An override names a lane for one (ori, year, offense) row; the lane it names is
    # a statement about the agency-year, so it selects for all seven offenses.
    override_lane = (
        panel.loc[manual_source_override, ["ori9", "preferred_source_by_regime"]]
        .dropna()
        .drop_duplicates(subset=["ori9"], keep="first")
        .set_index("ori9")["preferred_source_by_regime"]
        .astype("string")
    )

    signals = _agency_year_lane_signals(panel, corroborated=published_nibrs_supports_nibrs)
    ranked = _rank_agency_year_lanes(signals, override_lane=override_lane)
    chosen_lane = (
        ranked[ranked["lane_rank"].eq(0)].set_index("ori9")["source"].astype("string")
    )
    panel["preferred_lane_for_agency_year"] = panel["ori9"].map(chosen_lane).astype("string")
    panel["preferred_lane_selection_reason"] = (
        panel["ori9"].map(_lane_selection_reason(ranked)).astype("string")
    )

    # Read every offense from the chosen lane; where the chosen lane does not publish
    # an offense at all, walk the same agency-year ranking to the next lane that does.
    lane_rank_by_agency = {
        source: panel["ori9"].map(
            ranked[ranked["source"].eq(source)].set_index("ori9")["lane_rank"]
        )
        for source in LANE_STANDING_ORDER
    }
    preferred_source = pd.Series(pd.NA, index=panel.index, dtype="string")
    best_rank = pd.Series(np.inf, index=panel.index, dtype=float)
    for source in LANE_STANDING_ORDER:
        count_col, _ = PREFERRED_LANE_COLUMNS[source]
        rank = pd.to_numeric(lane_rank_by_agency[source], errors="coerce")
        candidate = panel[count_col].notna() & rank.notna() & rank.lt(best_rank)
        preferred_source = preferred_source.where(~candidate, source)
        best_rank = best_rank.where(~candidate, rank)
    panel["preferred_source"] = preferred_source
    panel["preferred_source_is_lane_supplement"] = ~panel["preferred_source"].eq(
        panel["preferred_lane_for_agency_year"]
    )
    panel = assign_preferred_value(
        panel,
        output_col="preferred_count",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "reported_count_cius",
            LOCAL_PUBLICATION_SOURCE: "reported_count_local_publication",
            STATE_PUBLICATION_SOURCE: "reported_count_state_publication",
            SUMMARY_SOURCE: "reported_count_srs",
            NIBRS_SOURCE: "reported_count_nibrs",
        },
    )
    panel = assign_preferred_value(
        panel,
        output_col="preferred_observation_weight",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "observation_weight_cius",
            LOCAL_PUBLICATION_SOURCE: "observation_weight_local_publication",
            STATE_PUBLICATION_SOURCE: "observation_weight_state_publication",
            SUMMARY_SOURCE: "observation_weight_srs",
            NIBRS_SOURCE: "observation_weight_nibrs",
        },
    )
    panel = assign_preferred_value(
        panel,
        output_col="preferred_months_reported",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "mean_months_reported_cius",
            LOCAL_PUBLICATION_SOURCE: "mean_months_reported_local_publication",
            STATE_PUBLICATION_SOURCE: "mean_months_reported_state_publication",
            SUMMARY_SOURCE: "mean_months_reported_srs",
            NIBRS_SOURCE: "mean_months_reported_nibrs",
        },
    )
    panel["preferred_source_lane"] = (
        panel["preferred_source"].map(source_lane_from_source).astype("string")
    )
    panel["preferred_reporting_mode"] = (
        panel["preferred_source"].map(reporting_mode_from_source).astype("string")
    )
    panel["preferred_conversion_status"] = (
        panel["preferred_source"]
        .map(default_conversion_status_from_source)
        .astype("string")
    )
    panel["conversion_status_nibrs"] = panel["preferred_conversion_status"]
    panel = assign_preferred_value(
        panel,
        output_col="preferred_conversion_status",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: "conversion_status_cius",
            LOCAL_PUBLICATION_SOURCE: "conversion_status_local_publication",
            STATE_PUBLICATION_SOURCE: "conversion_status_state_publication",
            SUMMARY_SOURCE: "conversion_status_srs",
            NIBRS_SOURCE: "conversion_status_nibrs",
        },
    )
    panel = panel.drop(columns=["conversion_status_nibrs"])
    panel["preferred_source_family"] = (
        panel["preferred_source"].map(source_family_from_source).astype("string")
    )
    panel["preferred_source_origin"] = [
        source_origin_from_parts(source=source) for source in panel["preferred_source"]
    ]
    panel["preferred_raw_data_source"] = [
        raw_data_source_from_parts(source=source)
        for source in panel["preferred_source"]
    ]
    panel["preferred_state_exception_flag"] = panel["preferred_source"].eq(
        STATE_PUBLICATION_SOURCE
    )
    panel["preferred_cius_reference_flag"] = panel["preferred_source_origin"].eq(
        CIUS_ORIGIN
    )
    _assert_one_preferred_row_per_agency_offense(panel)
    _assert_preferred_metadata_matches_the_selected_lane(panel)
    _assert_preferred_lane_unity(panel)
    return panel.sort_values(
        ["state_fips", "ori9", "offense"], kind="mergesort"
    ).reset_index(drop=True)
