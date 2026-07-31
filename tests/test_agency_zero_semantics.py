"""Zero-vs-missing semantics in the agency lane: NIBRS true zeros, silent agencies, and
the deterministic invariants that keep both from regressing."""

import numpy as np
import pandas as pd
import pytest

from crimerisk.crime import OFFENSES_7
from crimerisk.observations import _complete_nibrs_offense_rows_with_zeros
from crimerisk.source_provenance import NIBRS_SOURCE, SUMMARY_SOURCE
from crimerisk.source_selection import (
    _assert_preferred_metadata_matches_the_selected_lane,
)
from crimerisk.trend_fills import (
    _assert_estimates_finite_and_nonnegative,
    _assert_no_fill_where_the_chosen_lane_reported,
    _drop_silent_agency_estimates,
)


def _rollup_rows(rows: list[tuple[str, int, str, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "ori9",
            "year",
            "offense",
            "count",
            "offense_incident_months",
            "incident_months_any",
        ],
    )


def _submitted(rows: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ori9", "year"])


def test_nibrs_zero_emission_completes_the_offense_set():
    nibrs = _rollup_rows(
        [
            ("TX1310000", 2024, "motor_vehicle_theft", 3, 2, 2),
            ("TX1310000", 2023, "larceny", 5, 4, 4),
        ]
    )
    out = _complete_nibrs_offense_rows_with_zeros(
        nibrs, submitted_agency_years=_submitted([("TX1310000", 2024), ("TX1310000", 2023)])
    )

    assert len(out) == 2 * len(OFFENSES_7)
    for _key, grp in out.groupby(["ori9", "year"]):
        assert set(grp["offense"]) == set(OFFENSES_7)

    year_2024 = out[out["year"].eq(2024)].set_index("offense")
    # Present offenses are untouched.
    assert year_2024.loc["motor_vehicle_theft", "count"] == 3
    assert year_2024.loc["motor_vehicle_theft", "offense_incident_months"] == 2
    # Absent offenses become explicit zeros with the agency-year's own metadata.
    absent = year_2024.drop(index="motor_vehicle_theft")
    assert (absent["count"] == 0).all()
    assert (absent["offense_incident_months"] == 0).all()
    assert (absent["incident_months_any"] == 2).all()


def test_nibrs_zero_emission_covers_a_submitted_year_with_no_part_i_incident():
    # The b1 class: the batch header says the agency reported, the offense rollup has
    # nothing at all for it because no Part I incident occurred. That is an all-zero
    # year, not a missing one, and keying completion on the rollup lost 8,306 of them.
    nibrs = _rollup_rows([("TX1310000", 2024, "motor_vehicle_theft", 3, 2, 2)])
    out = _complete_nibrs_offense_rows_with_zeros(
        nibrs,
        submitted_agency_years=_submitted([("TX1310000", 2024), ("CA0190000", 2024)]),
    )
    silent = out[out["ori9"].eq("CA0190000")]
    assert set(silent["offense"]) == set(OFFENSES_7)
    assert (silent["count"] == 0).all()
    assert (silent["incident_months_any"] == 0).all()


def test_nibrs_zero_emission_does_not_invent_unsubmitted_agency_years():
    nibrs = _rollup_rows([("TX1310000", 2024, "motor_vehicle_theft", 3, 2, 2)])
    out = _complete_nibrs_offense_rows_with_zeros(
        nibrs, submitted_agency_years=_submitted([("TX1310000", 2024)])
    )
    # Neither the header nor the rollup mentions any other agency-year, so none appears:
    # an agency that did not submit must not become an all-zero reporter.
    assert set(zip(out["ori9"], out["year"])) == {("TX1310000", 2024)}


def test_nibrs_zero_emission_keeps_a_rollup_year_absent_from_the_header():
    # Incidents filed without a header record are still a submission; the completed
    # population is the union, so this agency-year is not dropped.
    nibrs = _rollup_rows([("TX1310000", 2024, "motor_vehicle_theft", 3, 2, 2)])
    out = _complete_nibrs_offense_rows_with_zeros(
        nibrs, submitted_agency_years=_submitted([])
    )
    assert set(zip(out["ori9"], out["year"])) == {("TX1310000", 2024)}
    assert len(out) == len(OFFENSES_7)


def test_nibrs_zero_emission_is_a_noop_on_a_complete_agency_year():
    nibrs = _rollup_rows(
        [("AA0000000", 2024, offense, 1, 1, 1) for offense in OFFENSES_7]
    )
    out = _complete_nibrs_offense_rows_with_zeros(
        nibrs, submitted_agency_years=_submitted([("AA0000000", 2024)])
    )
    assert len(out) == len(OFFENSES_7)
    assert (out["count"] == 1).all()


def _estimate_rows(rows: list[tuple[str, str, float, float, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "ori9",
            "offense",
            "reported_count_current",
            "estimated_count",
            "agency_estimate_source",
        ],
    )
    frame["state_fips"] = "48"
    frame["agency_adjustment_count"] = (
        frame["estimated_count"] - frame["reported_count_current"]
    ).clip(lower=0.0)
    frame["masked_gap_reclassified"] = False
    return frame


def test_silent_agencies_produce_no_estimate_row_at_all():
    result = _estimate_rows(
        [
            ("REPORTS", "burglary", 0.0, 0.0, "observed"),
            ("HAS_HISTORY", "burglary", 0.0, 7.0, "hist_median"),
            ("SILENT", "burglary", 0.0, 0.0, "current_count_no_history"),
        ]
    )
    out = _drop_silent_agency_estimates(result)

    # Absent, not present-with-zero: a fabricated zero anchor is as wrong as a peer one.
    assert "SILENT" not in set(out["ori9"])
    # A reported true zero is evidence and stays; so does a history-grounded fill.
    assert set(out["ori9"]) == {"REPORTS", "HAS_HISTORY"}


def test_silent_drop_keeps_a_no_history_row_that_still_reported_something():
    result = _estimate_rows(
        [("PARTIAL", "burglary", 4.0, 4.0, "current_count_no_history")]
    )
    assert set(_drop_silent_agency_estimates(result)["ori9"]) == {"PARTIAL"}


def _current_rows(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "ori9",
            "offense",
            "preferred_source",
            "preferred_months_reported",
            "preferred_count",
        ],
    )


def test_fill_on_a_full_year_lane_report_fails_closed():
    # The Kenedy shape: the chosen lane covered all twelve months, yet an offense of that
    # agency-year was filled from elsewhere.
    result = _estimate_rows([("TX1310000", "burglary", 0.0, 16.0, "hist_median")])
    current = _current_rows([("TX1310000", "burglary", NIBRS_SOURCE, 12.0, 0.0)])
    with pytest.raises(ValueError, match="preferred lane reported the target year"):
        _assert_no_fill_where_the_chosen_lane_reported(result, current_rows=current)


def test_fill_on_a_partial_year_lane_report_is_allowed():
    # One month of zeros is not a zero year, so a history fill here is legitimate.
    result = _estimate_rows([("AL0040200", "burglary", 0.0, 0.6, "trend_log_linear")])
    current = _current_rows([("AL0040200", "burglary", SUMMARY_SOURCE, 1.0, 0.0)])
    _assert_no_fill_where_the_chosen_lane_reported(result, current_rows=current)


def test_observed_and_annualized_rows_satisfy_the_lane_invariant():
    result = _estimate_rows(
        [
            ("A", "burglary", 3.0, 3.0, "observed"),
            ("B", "burglary", 3.0, 18.0, "true_partial_month_ratio"),
        ]
    )
    current = _current_rows(
        [
            ("A", "burglary", NIBRS_SOURCE, 12.0, 3.0),
            ("B", "burglary", SUMMARY_SOURCE, 12.0, 3.0),
        ]
    )
    _assert_no_fill_where_the_chosen_lane_reported(result, current_rows=current)


def test_non_finite_or_negative_estimates_fail_closed():
    bad = _estimate_rows([("A", "burglary", 0.0, np.nan, "hist_median")])
    with pytest.raises(ValueError, match="non-finite or negative"):
        _assert_estimates_finite_and_nonnegative(bad)

    negative = _estimate_rows([("A", "burglary", 0.0, -1.0, "hist_median")])
    with pytest.raises(ValueError, match="non-finite or negative"):
        _assert_estimates_finite_and_nonnegative(negative)


def _preferred_panel(months_from_srs: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ori9": ["TX1310000"],
            "offense": ["motor_vehicle_theft"],
            "preferred_source": [NIBRS_SOURCE],
            "preferred_count": [3.0],
            "preferred_months_reported": [2.0 if months_from_srs else 12.0],
            "reported_count_cius": [np.nan],
            "mean_months_reported_cius": [np.nan],
            "reported_count_local_publication": [np.nan],
            "mean_months_reported_local_publication": [np.nan],
            "reported_count_state_publication": [np.nan],
            "mean_months_reported_state_publication": [np.nan],
            "reported_count_srs": [3.0],
            "mean_months_reported_srs": [2.0],
            "reported_count_nibrs": [3.0],
            "mean_months_reported_nibrs": [12.0],
        }
    )


def test_preferred_metadata_must_come_from_the_selected_lane():
    _assert_preferred_metadata_matches_the_selected_lane(_preferred_panel(False))
    with pytest.raises(ValueError, match="metadata from a lane other than"):
        _assert_preferred_metadata_matches_the_selected_lane(_preferred_panel(True))
