"""Stage 1 identity resolution, coverage semantics and integrity post-conditions.

Companion to test_agency_zero_semantics.py: that file pins what a zero means, this
one pins who an agency is, how much of the year a lane covered, and what the panel
refuses to write.
"""

import numpy as np
import pandas as pd
import pytest

from crimerisk.agency_identity import (
    apply_cross_lane_twin_ledger,
    build_cross_lane_twin_ledger,
    build_ori_succession_ledger,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.observations import (
    _assert_fbi_lane_offense_sets_are_complete,
    _assert_no_negative_counts,
    _clamp_return_a_negative_counts,
)
from crimerisk.scope import PRODUCTION_SCOPE_EXCLUDE, production_scope_excluded
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
)
from crimerisk.source_selection import (
    LANE_SELECTION_REASON_COVERAGE,
    LANE_SELECTION_REASON_OVERRIDE,
    LANE_SELECTION_REASON_PUBLICATION,
    LANE_SELECTION_REASON_STANDING,
    _agency_year_lane_signals,
    _assert_one_preferred_row_per_agency_offense,
    _assert_preferred_lane_unity,
    _lane_selection_reason,
    _rank_agency_year_lanes,
)
from crimerisk.trend_fills import (
    _drop_silent_agency_estimates,
    _drop_superseded_ori_estimates,
    add_preferred_support_flags,
)


# --- cross-lane twin identity ------------------------------------------------


def _lane_rows(
    ori9: str, year: int, source: str, counts: dict[str, float], state_abbr: str = "CA"
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ori9": ori9,
            "year": year,
            "source": source,
            "offense": list(OFFENSES_7),
            "count": [float(counts.get(offense, 0.0)) for offense in OFFENSES_7],
            "state_abbr": state_abbr,
        }
    )


VECTOR = {"burglary": 12.0, "larceny": 30.0, "robbery": 3.0}


def test_twin_merge_resolves_a_nibrs_variant_onto_the_summary_ori():
    obs = pd.concat(
        [
            _lane_rows("CA0345500", 2024, CIUS_SOURCE, VECTOR),
            _lane_rows("CA034550X", 2024, NIBRS_SOURCE, VECTOR),
        ],
        ignore_index=True,
    )
    ledger = build_cross_lane_twin_ledger(obs, roster_oris=set())
    assert list(ledger["variant_ori9"]) == ["CA034550X"]
    assert list(ledger["canonical_ori9"]) == ["CA0345500"]
    assert ledger.loc[0, "identity_evidence"] == "offense_vector_identity"

    merged = apply_cross_lane_twin_ledger(obs, ledger)
    # One agency carrying both lanes, which is what lets preference choose once.
    assert set(merged["ori9"]) == {"CA0345500"}
    assert set(merged["source"]) == {CIUS_SOURCE, NIBRS_SOURCE}


def test_twin_merge_moves_the_whole_variant_series_including_disagreeing_years():
    # Identity is a property of the agency: the 2023 transition year, where NIBRS
    # coverage was partial and Return A still carried the real submission, moves too.
    obs = pd.concat(
        [
            _lane_rows("CA0345500", 2023, SUMMARY_SOURCE, {"larceny": 900.0}),
            _lane_rows("CA034550X", 2023, NIBRS_SOURCE, {"larceny": 4.0}),
            _lane_rows("CA0345500", 2024, SUMMARY_SOURCE, VECTOR),
            _lane_rows("CA034550X", 2024, NIBRS_SOURCE, VECTOR),
        ],
        ignore_index=True,
    )
    ledger = build_cross_lane_twin_ledger(obs, roster_oris=set())
    merged = apply_cross_lane_twin_ledger(obs, ledger)
    assert set(merged["ori9"]) == {"CA0345500"}
    assert len(merged[merged["year"].eq(2023) & merged["source"].eq(NIBRS_SOURCE)]) == 7


def test_twin_merge_resolves_the_agency_identity_not_only_the_key():
    # The NIBRS side's name comes from the batch header's covering city, so the twin of
    # Rancho Cordova is called SACRAMENTO. Carrying both identities under one ORI is
    # what splits every downstream join that keys on agency name.
    obs = pd.concat(
        [
            _lane_rows("CA0345500", 2024, CIUS_SOURCE, VECTOR),
            _lane_rows("CA034550X", 2024, NIBRS_SOURCE, VECTOR),
        ],
        ignore_index=True,
    )
    obs["agency_name_std"] = ["RANCHO CORDOVA"] * 7 + ["SACRAMENTO"] * 7
    obs["county_fips"] = [pd.NA] * 7 + ["067"] * 7
    obs["population"] = [np.nan] * 7 + [83622.0] * 7

    merged = apply_cross_lane_twin_ledger(
        obs, build_cross_lane_twin_ledger(obs, roster_oris=set())
    )
    assert set(merged["agency_name_std"]) == {"RANCHO CORDOVA"}
    # Where the canonical rows carry nothing, the variant's value survives rather than
    # being thrown away.
    assert set(merged["county_fips"]) == {"067"}
    assert set(merged["population"]) == {83622.0}


def test_twin_merge_leaves_distinct_agencies_in_one_ori_block_alone():
    # Same stem, different counts: a campus PD inside a city's ORI block is not the
    # city, and no vector agreement claims it is (the audit's a2 class).
    obs = pd.concat(
        [
            _lane_rows("CA0019900", 2024, SUMMARY_SOURCE, {"larceny": 400.0}),
            _lane_rows("CA0019945", 2024, NIBRS_SOURCE, {"larceny": 17.0}),
        ],
        ignore_index=True,
    )
    assert build_cross_lane_twin_ledger(obs, roster_oris=set()).empty


def test_single_count_agreement_needs_the_roster_witness():
    # One offense, one count, one year: there are only seven such vectors, so the
    # agreement is not evidence by itself.
    obs = pd.concat(
        [
            _lane_rows("TX1292300", 2022, SUMMARY_SOURCE, {"motor_vehicle_theft": 1.0}, "TX"),
            _lane_rows("TX129239E", 2022, NIBRS_SOURCE, {"motor_vehicle_theft": 1.0}, "TX"),
        ],
        ignore_index=True,
    )
    assert build_cross_lane_twin_ledger(obs, roster_oris=set()).empty
    # ... unless the FBI's own roster lists the NIBRS ORI and not the summary one.
    witnessed = build_cross_lane_twin_ledger(obs, roster_oris={"TX129239E"})
    assert list(witnessed["variant_ori9"]) == ["TX129239E"]
    assert (
        witnessed.loc[0, "identity_evidence"]
        == "single_count_vector_identity_with_fbi_roster_witness"
    )
    # Both listed means two agencies, and the coincidence stays a coincidence.
    assert build_cross_lane_twin_ledger(
        obs, roster_oris={"TX129239E", "TX1292300"}
    ).empty


def test_an_all_zero_year_is_not_identity_evidence():
    obs = pd.concat(
        [
            _lane_rows("CA0345500", 2024, SUMMARY_SOURCE, {}),
            _lane_rows("CA034550X", 2024, NIBRS_SOURCE, {}),
        ],
        ignore_index=True,
    )
    assert build_cross_lane_twin_ledger(obs, roster_oris=set()).empty


def test_twin_resolution_must_be_a_function_from_variant_to_canonical():
    # Two NIBRS variants matching one summary ORI means the stem block holds more than
    # one agency; the rule refuses rather than folding them together.
    obs = pd.concat(
        [
            _lane_rows("CA0345500", 2024, SUMMARY_SOURCE, VECTOR),
            _lane_rows("CA034550X", 2024, NIBRS_SOURCE, VECTOR),
            _lane_rows("CA034559E", 2024, NIBRS_SOURCE, VECTOR),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="not a function from variant to canonical"):
        build_cross_lane_twin_ledger(obs, roster_oris=set())


# --- ORI succession ----------------------------------------------------------


def _panel_rows(rows: list[tuple[str, int, bool]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["ori9", "year", "usable_as_observed"])
    frame["current_row_is_true_partial"] = False
    return frame


def _crosswalk(rows: list[tuple[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["ori9", "jurisdiction_id"])
    frame["weight"] = 1.0
    frame["state_fips"] = "12"
    return frame


def test_a_dead_ori_covered_by_a_live_successor_is_superseded():
    # Jacksonville: FL0160200 retired in 2021, FL0160000 covers the same city now.
    panel = _panel_rows(
        [
            ("FL0160200", 2019, True),
            ("FL0160200", 2020, True),
            ("FL0160000", 2023, True),
            ("FL0160000", 2024, True),
        ]
    )
    crosswalk = _crosswalk(
        [
            ("FL0160200", "12:municipal:place:1235000"),
            ("FL0160000", "12:municipal:place:1235000"),
        ]
    )
    ledger = build_ori_succession_ledger(
        agency_panel=panel,
        agency_jurisdiction_crosswalk=crosswalk,
        target_year=2024,
        max_reference_age_years=2,
    )
    assert list(ledger["superseded_ori9"]) == ["FL0160200"]
    assert list(ledger["successor_ori9"]) == ["FL0160000"]

    estimates = pd.DataFrame(
        {
            "ori9": ["FL0160200", "FL0160000"],
            "offense": ["burglary", "burglary"],
            "estimated_count": [3000.0, 4000.0],
        }
    )
    kept = _drop_superseded_ori_estimates(estimates, succession_ledger=ledger)
    assert set(kept["ori9"]) == {"FL0160000"}


def test_a_recently_reporting_ori_is_not_superseded():
    panel = _panel_rows([("A", 2023, True), ("B", 2024, True)])
    crosswalk = _crosswalk([("A", "12:municipal:place:1"), ("B", "12:municipal:place:1")])
    assert build_ori_succession_ledger(
        agency_panel=panel,
        agency_jurisdiction_crosswalk=crosswalk,
        target_year=2024,
        max_reference_age_years=2,
    ).empty


def test_overlap_layers_and_county_remainders_cannot_supersede():
    # A statewide overlap layer legitimately hosts many agencies at once, so shared
    # membership there says nothing about coverage.
    panel = _panel_rows([("A", 2019, True), ("B", 2024, True)])
    crosswalk = _crosswalk(
        [("A", "12:statewide_overlap_layer"), ("B", "12:statewide_overlap_layer")]
    )
    assert build_ori_succession_ledger(
        agency_panel=panel,
        agency_jurisdiction_crosswalk=crosswalk,
        target_year=2024,
        max_reference_age_years=2,
    ).empty


# --- partial-year coverage semantics ----------------------------------------


def _preferred(
    source: str, months: float, count: float, regime: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ori9": ["A"],
            "offense": ["burglary"],
            "preferred_source": [source],
            "preferred_count": [count],
            "preferred_months_reported": [months],
            "reporting_regime": [regime],
        }
    )


@pytest.mark.parametrize(
    "source", [SUMMARY_SOURCE, NIBRS_SOURCE, STATE_PUBLICATION_SOURCE, LOCAL_PUBLICATION_SOURCE]
)
def test_a_partial_year_is_partial_on_every_lane(source):
    flags = add_preferred_support_flags(_preferred(source, 5.0, 20.0, "true_partial"))
    assert bool(flags.loc[0, "current_row_is_true_partial"])
    assert not bool(flags.loc[0, "usable_as_observed"])


def test_a_zero_offense_inside_a_partial_year_is_still_partial():
    # The e2 class: the count>0 guard sent these zeros down the fill ladder instead of
    # leaving them zeros over the reported months.
    flags = add_preferred_support_flags(_preferred(NIBRS_SOURCE, 5.0, 0.0, "true_partial"))
    assert bool(flags.loc[0, "current_row_is_true_partial"])


def test_a_full_year_row_is_observed_not_partial():
    flags = add_preferred_support_flags(
        _preferred(NIBRS_SOURCE, 12.0, 20.0, "annual_only_but_usable")
    )
    assert bool(flags.loc[0, "usable_as_observed"])
    assert not bool(flags.loc[0, "current_row_is_true_partial"])


def test_a_twelve_month_nibrs_row_under_an_srs_partial_regime_stays_observed():
    # The chosen lane covered the whole year; the regime label describes Return A.
    flags = add_preferred_support_flags(_preferred(NIBRS_SOURCE, 12.0, 20.0, "true_partial"))
    assert bool(flags.loc[0, "usable_as_observed"])
    assert not bool(flags.loc[0, "current_row_is_true_partial"])


def test_a_lumpy_return_a_month_set_is_not_annualised():
    flags = add_preferred_support_flags(_preferred(SUMMARY_SOURCE, 5.0, 20.0, "lumpy_or_batched"))
    assert not bool(flags.loc[0, "current_row_is_true_partial"])
    assert bool(flags.loc[0, "usable_as_observed"])


def test_lumpiness_is_a_return_a_diagnostic_and_does_not_veto_another_lane():
    flags = add_preferred_support_flags(_preferred(NIBRS_SOURCE, 5.0, 20.0, "lumpy_or_batched"))
    assert bool(flags.loc[0, "current_row_is_true_partial"])


def test_structurally_missing_rows_are_neither_observed_nor_partial():
    flags = add_preferred_support_flags(
        _preferred(SUMMARY_SOURCE, 0.0, 0.0, "structurally_missing_or_unreliable")
    )
    assert not bool(flags.loc[0, "usable_as_observed"])
    assert not bool(flags.loc[0, "current_row_is_true_partial"])


# --- fill recency bound ------------------------------------------------------


def test_an_agency_past_the_recency_bound_gets_no_estimate_row():
    result = pd.DataFrame(
        {
            "ori9": ["STALE", "RECENT"],
            "offense": ["burglary", "burglary"],
            "reported_count_current": [0.0, 0.0],
            "estimated_count": [0.0, 7.0],
            "agency_estimate_source": [
                "no_recent_history_beyond_fill_recency_bound",
                "hist_median",
            ],
        }
    )
    assert set(_drop_silent_agency_estimates(result)["ori9"]) == {"RECENT"}


# --- per-agency-year lane unity ---------------------------------------------


def _unity_panel(
    sources: list[str], chosen: str, supplement: list[bool]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ori9": ["A"] * len(sources),
            "offense": list(OFFENSES_7)[: len(sources)],
            "preferred_source": sources,
            "preferred_lane_for_agency_year": [chosen] * len(sources),
            "preferred_source_is_lane_supplement": supplement,
        }
    )


def test_lane_unity_holds_when_every_offense_reads_the_chosen_lane():
    _assert_preferred_lane_unity(
        _unity_panel([CIUS_SOURCE, CIUS_SOURCE], CIUS_SOURCE, [False, False])
    )


def test_lane_unity_allows_an_annotated_supplement():
    # CIUS withheld one cell; that offense reads the next lane and says so.
    _assert_preferred_lane_unity(
        _unity_panel([CIUS_SOURCE, SUMMARY_SOURCE], CIUS_SOURCE, [False, True])
    )


def test_an_unannotated_second_lane_fails_closed():
    with pytest.raises(ValueError, match="without a supplement annotation"):
        _assert_preferred_lane_unity(
            _unity_panel([CIUS_SOURCE, SUMMARY_SOURCE], CIUS_SOURCE, [False, False])
        )


def test_a_split_agency_identity_in_the_preferred_panel_fails_closed():
    panel = pd.DataFrame(
        {
            "ori9": ["A", "A"],
            "state_fips": ["48", "48"],
            "state_abbr": ["TX", "TX"],
            "offense": ["burglary", "burglary"],
        }
    )
    with pytest.raises(ValueError, match=r"share an \(ori9, offense\) key"):
        _assert_one_preferred_row_per_agency_offense(panel)


def test_two_chosen_lanes_in_one_agency_year_fail_closed():
    panel = _unity_panel([CIUS_SOURCE, SUMMARY_SOURCE], CIUS_SOURCE, [False, False])
    panel.loc[1, "preferred_lane_for_agency_year"] = SUMMARY_SOURCE
    with pytest.raises(ValueError, match="more than one lane"):
        _assert_preferred_lane_unity(panel)


def _fork_panel(months: float, counts: dict[str, float]) -> pd.DataFrame:
    """The Okanogan shape: Return A and NIBRS render the same partial agency-year."""
    frame = pd.DataFrame({"ori9": "WA0240300", "offense": list(OFFENSES_7)})
    frame["reported_count_cius"] = np.nan
    frame["mean_months_reported_cius"] = np.nan
    frame["observation_weight_cius"] = np.nan
    frame["reported_count_local_publication"] = np.nan
    frame["mean_months_reported_local_publication"] = np.nan
    frame["observation_weight_local_publication"] = np.nan
    frame["reported_count_state_publication"] = np.nan
    frame["mean_months_reported_state_publication"] = np.nan
    frame["observation_weight_state_publication"] = np.nan
    values = [float(counts.get(offense, 0.0)) for offense in OFFENSES_7]
    for lane in ("srs", "nibrs"):
        frame[f"reported_count_{lane}"] = values
        frame[f"mean_months_reported_{lane}"] = months
        frame[f"observation_weight_{lane}"] = 0.5
    return frame


def test_a_partial_year_rendered_by_both_fbi_lanes_resolves_to_one_lane():
    # Omak WA 2024: nine months in both lanes, identical counts. The per-offense rule
    # sent murder and robbery (0) to NIBRS as full-year observed zeros while the five
    # positive offenses took Return A and were annualised x12/9 -- one agency-year, two
    # incompatible accounts of the same nine months.
    panel = _fork_panel(
        9.0,
        {"aggravated_assault": 2.0, "burglary": 27.0, "larceny": 58.0,
         "motor_vehicle_theft": 6.0, "rape": 2.0},
    )
    signals = _agency_year_lane_signals(
        panel, corroborated=pd.Series(False, index=panel.index)
    )
    ranked = _rank_agency_year_lanes(signals, override_lane=pd.Series(dtype="string"))
    winner = ranked[ranked["lane_rank"].eq(0)]
    assert list(winner["source"]) == [SUMMARY_SOURCE]
    # Equal coverage and weight, so the standing lane order decides -- and it decides
    # once, for all seven offenses including the zeros.
    assert (
        _lane_selection_reason(ranked).loc["WA0240300"] == LANE_SELECTION_REASON_STANDING
    )


def test_a_lane_with_more_month_coverage_wins_the_agency_year():
    panel = _fork_panel(9.0, {"larceny": 58.0})
    panel["mean_months_reported_nibrs"] = 12.0
    panel["observation_weight_nibrs"] = 1.0
    signals = _agency_year_lane_signals(
        panel, corroborated=pd.Series(False, index=panel.index)
    )
    ranked = _rank_agency_year_lanes(signals, override_lane=pd.Series(dtype="string"))
    assert list(ranked[ranked["lane_rank"].eq(0)]["source"]) == [NIBRS_SOURCE]
    assert (
        _lane_selection_reason(ranked).loc["WA0240300"] == LANE_SELECTION_REASON_COVERAGE
    )


def test_a_manual_source_override_names_the_lane_for_the_whole_agency_year():
    panel = _fork_panel(12.0, {"larceny": 58.0})
    signals = _agency_year_lane_signals(
        panel, corroborated=pd.Series(False, index=panel.index)
    )
    ranked = _rank_agency_year_lanes(
        signals, override_lane=pd.Series({"WA0240300": NIBRS_SOURCE}, dtype="string")
    )
    assert list(ranked[ranked["lane_rank"].eq(0)]["source"]) == [NIBRS_SOURCE]
    assert (
        _lane_selection_reason(ranked).loc["WA0240300"] == LANE_SELECTION_REASON_OVERRIDE
    )


def test_a_publication_lane_keeps_its_standing_over_a_fuller_fbi_year():
    panel = _fork_panel(12.0, {"larceny": 58.0})
    panel["reported_count_state_publication"] = 60.0
    panel["mean_months_reported_state_publication"] = 6.0
    panel["observation_weight_state_publication"] = 0.5
    signals = _agency_year_lane_signals(
        panel, corroborated=pd.Series(False, index=panel.index)
    )
    ranked = _rank_agency_year_lanes(signals, override_lane=pd.Series(dtype="string"))
    assert list(ranked[ranked["lane_rank"].eq(0)]["source"]) == [STATE_PUBLICATION_SOURCE]
    assert (
        _lane_selection_reason(ranked).loc["WA0240300"]
        == LANE_SELECTION_REASON_PUBLICATION
    )


# --- integrity post-conditions ----------------------------------------------


def _count_rows(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["ori9", "source", "count"])
    frame["year"] = 2019
    frame["offense"] = "rape"
    return frame


def test_return_a_negative_adjustments_are_clamped_and_recorded():
    clamped = _clamp_return_a_negative_counts(
        _count_rows([("A", SUMMARY_SOURCE, -1.0), ("B", SUMMARY_SOURCE, 4.0)])
    )
    assert list(clamped["count"]) == [0.0, 4.0]
    assert list(clamped["negative_count_clamped_amount"]) == [1.0, 0.0]
    _assert_no_negative_counts(clamped)


def test_a_negative_count_on_a_lane_without_an_adjustment_mechanism_fails_closed():
    with pytest.raises(ValueError, match="no documented adjustment mechanism"):
        _clamp_return_a_negative_counts(_count_rows([("A", CIUS_SOURCE, -1.0)]))


def test_a_negative_count_in_the_written_panel_fails_closed():
    with pytest.raises(ValueError, match="negative count"):
        _assert_no_negative_counts(_count_rows([("A", SUMMARY_SOURCE, -1.0)]))


def _offense_set_rows(source: str, offenses: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ori9": "A", "year": 2024, "source": source, "offense": offenses, "count": 0.0}
    )


def test_fbi_lanes_must_carry_every_offense():
    _assert_fbi_lane_offense_sets_are_complete(
        _offense_set_rows(NIBRS_SOURCE, list(OFFENSES_7))
    )
    with pytest.raises(ValueError, match="fewer than 7 offense rows"):
        _assert_fbi_lane_offense_sets_are_complete(
            _offense_set_rows(SUMMARY_SOURCE, list(OFFENSES_7)[:-1])
        )


def test_publication_lanes_may_publish_a_partial_offense_set():
    # A withheld CIUS cell is missing, not zero, so the row is simply absent and the
    # completeness assertion does not claim otherwise.
    _assert_fbi_lane_offense_sets_are_complete(
        _offense_set_rows(CIUS_SOURCE, list(OFFENSES_7)[:-1])
    )


def test_production_scope_names_the_excluded_populations():
    flags = production_scope_excluded(pd.Series(["CA", "PR", "GU", "AK", "HI", "NB"]))
    assert list(flags) == [False, True, True, True, True, False]
    # NB is two federal ORIs sitting in Nebraska under a retired state code, not a
    # territory: the dead-ORI predicate handles them, not the scope rule.
    assert "NB" not in PRODUCTION_SCOPE_EXCLUDE
    assert {"PR", "GU", "VI", "GM", "AK", "HI"} <= PRODUCTION_SCOPE_EXCLUDE
