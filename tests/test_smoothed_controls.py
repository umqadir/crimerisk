"""Surface 2 — the smoothed control lane (contract: SURFACE2_CONTROLS_CONTRACT.md).

Three things are load-bearing and are asserted here: the E1 estimator arithmetic
(hand-computed EWA + hierarchical shrink, and murder's flat-mean kernel), the fallback path
for units the estimator may not run on, and the schema/universe contract against the
accounting surface.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import _load_controls
from crimerisk.controls import CONTROL_KEY_COLUMNS
from crimerisk.paths import RepoPaths
from crimerisk.smoothed_controls import (
    COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR,
    DYNAMIC_SHRINK_ESTIMATOR,
    PRECISION_WEIGHTED_ESTIMATOR,
    E1_HALFLIFE_YEARS,
    E1_SHRINKAGE_K,
    ESTIMATE_DIAGNOSTIC_COLUMNS,
    FALLBACK_ACCOUNTING_ESTIMATOR,
    FLAT_MEAN_KERNEL,
    INFORMATION_POSTERIOR_ESTIMATOR,
    INSUFFICIENT_HISTORY_REASON,
    NO_EXPOSURE_REASON,
    NO_PEER_SUPPORT_REASON,
    SOURCE_REPLACEMENT_PROTECTED_REASON,
    STRUCTURAL_ZERO_PROTECTED_REASON,
    MURDER_INFORMATION_PRIOR_EXPOSURE,
    SMOOTHED_CONTROL_COLUMNS,
    SmoothedControlConfig,
    assert_smoothed_control_invariants,
    build_smoothed_controls,
    clean_panel_history,
    kernel_weights,
    series_kernel_levels,
    smoothed_controls_summary_path,
    write_v2_smoothed_controls,
)


CONFIG = SmoothedControlConfig(year=2024)


def test_allocator_can_select_validated_smoothed_controls(tmp_path: Path) -> None:
    paths = RepoPaths.from_repo_root(tmp_path)
    controls_dir = paths.state_dir / "controls"
    controls_dir.mkdir(parents=True)
    accounting = pd.DataFrame(
        [
            {
                "jurisdiction_id": "01:municipal:place:A",
                "offense": "robbery",
                "adjusted_count_ags_core": 10.0,
                "reported_count_preferred": 8.0,
            }
        ]
    )
    smoothed = pd.DataFrame(
        [
            {
                "jurisdiction_id": "01:municipal:place:A",
                "offense": "robbery",
                "accounting_count": 10.0,
                "smoothed_count": 12.5,
                "estimator": DYNAMIC_SHRINK_ESTIMATOR,
            }
        ]
    )
    accounting.to_parquet(controls_dir / "jurisdiction_controls_2024.parquet", index=False)
    smoothed.to_parquet(
        controls_dir / "jurisdiction_controls_smoothed_2024.parquet", index=False
    )

    loaded = _load_controls(paths, year=2024)

    assert loaded.loc[0, "adjusted_count_ags_core"] == pytest.approx(12.5)
    assert loaded.loc[0, "accounting_count"] == pytest.approx(10.0)
    assert loaded.loc[0, "reported_count_preferred"] == pytest.approx(8.0)
    assert loaded.loc[0, "control_surface"] == "smoothed"


def test_allocator_rejects_smoothed_controls_from_another_accounting_edition(
    tmp_path: Path,
) -> None:
    paths = RepoPaths.from_repo_root(tmp_path)
    controls_dir = paths.state_dir / "controls"
    controls_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "jurisdiction_id": "01:municipal:place:A",
                "offense": "robbery",
                "adjusted_count_ags_core": 10.0,
            }
        ]
    ).to_parquet(controls_dir / "jurisdiction_controls_2024.parquet", index=False)
    pd.DataFrame(
        [
            {
                "jurisdiction_id": "01:municipal:place:A",
                "offense": "robbery",
                "accounting_count": 9.0,
                "smoothed_count": 12.5,
                "estimator": DYNAMIC_SHRINK_ESTIMATOR,
            }
        ]
    ).to_parquet(controls_dir / "jurisdiction_controls_smoothed_2024.parquet", index=False)

    with pytest.raises(ValueError, match="different accounting edition"):
        _load_controls(paths, year=2024)


# --- fixtures ---------------------------------------------------------------


def _accounting_row(
    jurisdiction_id: str,
    offense: str,
    *,
    population: float,
    count: float,
    jurisdiction_type: str = "municipal",
    state_fips: str = "01",
) -> dict:
    return {
        "jurisdiction_id": jurisdiction_id,
        "jurisdiction_type": jurisdiction_type,
        "jurisdiction_name": jurisdiction_id,
        "state_fips": state_fips,
        "state_abbr": "AL",
        "geo_type": "place",
        "geoid": jurisdiction_id.split(":")[-1],
        "offense": offense,
        "bucket_population": population,
        "pop_band": "25k-50k",
        "adjusted_count_ags_core": count,
    }


def _panel_row(
    jurisdiction_id: str,
    offense: str,
    year: int,
    count: float | None,
    *,
    months: float = 12.0,
    weight: float = 1.0,
    fill: float = 0.0,
    partial: float = 0.0,
    population: float = 1000.0,
) -> dict:
    complete = months >= 12.0 and weight >= 1.0 and fill == 0.0 and partial == 0.0 and count is not None
    return {
        "jurisdiction_id": jurisdiction_id,
        "offense": offense,
        "year": year,
        "reported_count_preferred": count,
        # The panel carries the accounting-basis level too; only the target year's differs
        # from the reported one, and the summary's panel band reads it.
        "estimated_count": count,
        "mean_months_reported_preferred": months,
        "observation_weight_preferred": weight,
        "fill_component_count": fill,
        "partial_component_count": partial,
        # The adjudicated history's coverage: a row is a complete admitted year only when
        # it is a twelve-month, full-weight filing with no fill and no partial component.
        "full_year_admitted_count": count if complete else 0.0,
        "full_year_admitted_share": 1.0 if complete else 0.0,
        "bucket_population": population,
    }


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two units in one peer group.

    `A` carries a full burglary window and a full murder window; `B` carries a two-year
    burglary window of zeros (so it exercises `w = 0`, i.e. pure peer prediction) and a
    one-year murder window (the minimum history admitted by the E1 tournament).
    """
    accounting = pd.DataFrame(
        [
            _accounting_row("01:municipal:place:A", "burglary", population=1000.0, count=44.0),
            _accounting_row("01:municipal:place:A", "murder", population=1000.0, count=7.0),
            _accounting_row("01:municipal:place:B", "burglary", population=1000.0, count=3.0),
            _accounting_row("01:municipal:place:B", "murder", population=1000.0, count=5.0),
        ]
    )
    panel = pd.DataFrame(
        [
            # A/burglary: three clean years, plus three years excluded for three reasons.
            _panel_row("01:municipal:place:A", "burglary", 2019, 900.0, weight=0.5),
            _panel_row("01:municipal:place:A", "burglary", 2020, 900.0, fill=3.0),
            _panel_row("01:municipal:place:A", "burglary", 2021, 900.0, months=6.0),
            _panel_row("01:municipal:place:A", "burglary", 2022, 10.0),
            _panel_row("01:municipal:place:A", "burglary", 2023, 20.0),
            _panel_row("01:municipal:place:A", "burglary", 2024, 40.0),
            # A/murder: three clean years, plus one excluded for a partial component.
            _panel_row("01:municipal:place:A", "murder", 2021, 900.0, partial=1.0),
            _panel_row("01:municipal:place:A", "murder", 2022, 1.0),
            _panel_row("01:municipal:place:A", "murder", 2023, 2.0),
            _panel_row("01:municipal:place:A", "murder", 2024, 6.0),
            # B/burglary: two clean years of zeros.
            _panel_row("01:municipal:place:B", "burglary", 2023, 0.0),
            _panel_row("01:municipal:place:B", "burglary", 2024, 0.0),
            # B/murder: one clean year -> admitted and hierarchically shrunk.
            _panel_row("01:municipal:place:B", "murder", 2024, 4.0),
        ]
    )
    return accounting, panel


def _row(frame: pd.DataFrame, jurisdiction_id: str, offense: str) -> pd.Series:
    match = frame[frame["jurisdiction_id"].eq(jurisdiction_id) & frame["offense"].eq(offense)]
    assert len(match) == 1
    return match.iloc[0]


# --- the shipped constants are the tournament's ------------------------------


def test_shipped_constants_preserve_volume_e1_and_use_validated_murder_information_parameters():
    assert E1_SHRINKAGE_K == 0.5
    assert E1_HALFLIFE_YEARS["murder"] == 1.0
    assert MURDER_INFORMATION_PRIOR_EXPOSURE == 10_000.0
    assert {E1_HALFLIFE_YEARS[o] for o in ("rape", "robbery", "aggravated_assault")} == {2.0}
    assert {
        E1_HALFLIFE_YEARS[o] for o in ("burglary", "larceny", "motor_vehicle_theft")
    } == {1.0}
    # Round 3's trend multiplier was rejected; nothing in the shipped config may reinstate it.
    assert not any("trend" in field for field in SmoothedControlConfig.__dataclass_fields__)
    assert not any("trend" in column for column in SMOOTHED_CONTROL_COLUMNS)


def test_kernel_weights_halve_at_the_halflife_and_are_flat_for_the_flat_mean():
    ages = pd.Series([0.0, 1.0, 2.0, 4.0])
    ewa = kernel_weights(ages, pd.Series([2.0, 2.0, 2.0, 2.0]))
    assert ewa.tolist() == pytest.approx([1.0, 2.0**-0.5, 0.5, 0.25])
    flat = kernel_weights(ages, pd.Series([np.nan] * 4))
    assert flat.tolist() == [1.0, 1.0, 1.0, 1.0]


# --- the estimator arithmetic ------------------------------------------------


def test_dynamic_shrink_matches_the_hand_computation():
    """A: y=(10, 20, 40) for 2022-2024 at halflife 1.

    The E1 level (published diagnostics): weights (0.25, 0.5, 1.0); EWA 30; peer_rate 0.015 (peer
    prediction 15); w = 30/30.5. B: two clean zeros -> EWA 0, w = 0.
    """
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    a = _row(out, "01:municipal:place:A", "burglary")
    assert a["estimator"] == PRECISION_WEIGHTED_ESTIMATOR
    assert a["clean_year_count"] == 3
    assert a["latest_clean_year"] == 2024
    assert a["halflife_used"] == 1.0
    assert a["ewa_count"] == pytest.approx(30.0)
    assert a["peer_rate"] == pytest.approx(0.015)
    assert a["peer_predicted_count"] == pytest.approx(15.0)
    assert a["ewa_weight"] == pytest.approx(30.0 / 30.5)
    # The smoothed control is not the accounting control, and is not raked to it.
    assert a["accounting_count"] == 44.0

    b = _row(out, "01:municipal:place:B", "burglary")
    assert b["ewa_count"] == pytest.approx(0.0)
    assert b["ewa_weight"] == pytest.approx(0.0)


def test_precision_weighted_level_matches_the_hand_computation():
    """The target-year count weighted by its precision against the past-only forecast.

    Two units cannot support a signal variance (SIGNAL_VARIANCE_MIN_SUPPORT pairs), so the explicit
    fallback applies and is published: tau2 = 0 and the forecast keeps only its own variance, so
    the current year counts as one more year of the series.
    A 2024: own forecast from 2022-2023 at weights (0.25, 0.5): EWA = 12.5 / 0.75; the leave-one-out
    past-only peer is B's own forecast of 2024 (its 2023 zero), so peer = 0; s = EWA / (EWA + k);
    H = s EWA; var(H) = s^2 EWA (0.25^2 + 0.5^2) / 0.75^2; w = var(H) / (var(H) + H).
    B 2024: no earlier year of its own beyond a zero (EWA 0, s = 0): H is the peer, A's forecast
    rate times B's exposure, worth one year (var = H), so w = 1/2 and c = 0.
    """
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    a = _row(out, "01:municipal:place:A", "burglary")
    ewa = 12.5 / 0.75
    s = ewa / (ewa + 0.5)
    h = s * ewa
    var_h = s**2 * ewa * (0.25**2 + 0.5**2) / 0.75**2
    w = var_h / (var_h + h)
    assert a["signal_variance_level"] == "fallback_one_more_year"
    assert a["signal_relative_variance"] == pytest.approx(0.0)
    assert a["history_forecast_count"] == pytest.approx(h)
    assert a["history_forecast_variance"] == pytest.approx(var_h)
    assert a["history_forecast_years"] == 2
    assert a["current_year_count"] == pytest.approx(40.0)
    assert a["current_year_weight"] == pytest.approx(w)
    assert a["smoothed_count"] == pytest.approx(w * 40.0 + (1 - w) * h)

    b = _row(out, "01:municipal:place:B", "burglary")
    assert b["history_forecast_count"] == pytest.approx(ewa)
    assert b["current_year_weight"] == pytest.approx(0.5)
    assert b["smoothed_count"] == pytest.approx(0.5 * ewa)


def _precision_units(n_big: int, n_small: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    rows, units = [], []
    for unit, scale in [(f"big{i}", 1000.0) for i in range(n_big)] + [(f"small{i}", 2.0) for i in range(n_small)]:
        for i, year in enumerate(years):
            # Every series shares a +-10% year-to-year signal; the big ones are Poisson-precise.
            rows.append({"jurisdiction_id": unit, "offense": "robbery", "year": year,
                         "clean_count": scale * (1.1 if i % 2 else 0.9), "clean_exposure": 1000.0})
        units.append({"jurisdiction_id": unit, "offense": "robbery", "state_abbr": "AL", "state_fips": "01",
                      "jurisdiction_type": "municipal", "pop_band": "<5k", "bucket_population": 1000.0})
    return pd.DataFrame(units), pd.DataFrame(rows)


def test_precision_weight_rises_with_the_count_and_the_forecast_moves_to_the_state_level():
    from crimerisk.smoothed_controls import SIGNAL_VARIANCE_MIN_SUPPORT, precision_weighted_levels

    units, history = _precision_units(20, 20)
    config = SmoothedControlConfig(year=2024)
    out = precision_weighted_levels(units, history, config=config, state_year_totals=None).set_index("jurisdiction_id")
    assert set(out["signal_variance_level"]) == {"offense_band"}
    assert out["signal_variance_support"].min() >= SIGNAL_VARIANCE_MIN_SUPPORT
    assert out["signal_relative_variance"].min() > 0
    assert out.loc["big0", "current_year_weight"] > 0.9
    assert out.loc["small0", "current_year_weight"] < out.loc["big0", "current_year_weight"]

    # A state total that doubles from 2023 to 2024 doubles the history forecast at 2024.
    totals = pd.DataFrame({"state_abbr": "AL", "offense": "robbery", "year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
                           "cde_state_count": [100.0] * 6 + [200.0]})
    scaled = precision_weighted_levels(units, history, config=config, state_year_totals=totals).set_index("jurisdiction_id")
    assert scaled.loc["big0", "history_forecast_count"] == pytest.approx(2 * out.loc["big0", "history_forecast_count"], rel=1e-3)


def test_signal_variance_falls_back_up_the_hierarchy_and_publishes_its_support():
    from crimerisk.smoothed_controls import SIGNAL_VARIANCE_MIN_SUPPORT, precision_weighted_levels

    units, history = _precision_units(20, 20)
    # Two units in a band of their own: too few pairs there, so the offense level answers.
    units.loc[units["jurisdiction_id"].isin(["small0", "small1"]), "pop_band"] = "5k-10k"
    out = precision_weighted_levels(units, history, config=SmoothedControlConfig(year=2024), state_year_totals=None).set_index("jurisdiction_id")
    assert out.loc["small0", "signal_variance_level"] == "offense"
    assert out.loc["small0", "signal_variance_support"] > out.loc["big0", "signal_variance_support"] >= SIGNAL_VARIANCE_MIN_SUPPORT
    assert out.loc["big0", "signal_variance_level"] == "offense_band"

    # Earlier years never see the target year: adding a 2025 row changes no 2024 level.
    later = pd.concat([history, history[history["year"].eq(2024)].assign(year=2025, clean_count=99_999.0)], ignore_index=True)
    again = precision_weighted_levels(units, later[later["year"].le(2025)], config=SmoothedControlConfig(year=2024), state_year_totals=None)
    pd.testing.assert_series_equal(
        again.set_index("jurisdiction_id")["precision_weighted_count"], out["precision_weighted_count"], check_names=False)


def test_murder_uses_information_weighting_and_one_year_series_borrow_from_established_peers():
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    a = _row(out, "01:municipal:place:A", "murder")
    # A cannot use itself as a peer, and B's one clean year cannot define its peer.
    assert a["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert a["fallback_reason"] == NO_PEER_SUPPORT_REASON

    b = _row(out, "01:municipal:place:B", "murder")
    # Murder is precision-weighted against its information-posterior forecast; with no earlier
    # year and no signal variance this is exactly the information posterior.
    assert b["estimator"] == PRECISION_WEIGHTED_ESTIMATOR
    assert b["temporal_kernel"] == "ewa"
    assert b["halflife_used"] == pytest.approx(1.0)
    # A's weighted history is (.25*1 + .5*2 + 1*6) events over 1,750 person-years.
    peer_rate = 7.25 / 1_750.0
    assert b["peer_rate"] == pytest.approx(peer_rate)
    assert b["ewa_weight"] == pytest.approx(1_000.0 / 11_000.0)
    # The forecast's peer is past-only: A's weighted years before 2024, (.25*1 + .5*2) events over
    # 750 person-years. Under the explicit variance fallback the precision-weighted level is the
    # information posterior with that peer rate.
    past_rate = 1.25 / 750.0
    assert b["history_forecast_count"] == pytest.approx(past_rate * 1_000.0)
    assert b["smoothed_count"] == pytest.approx(
        (4.0 + MURDER_INFORMATION_PRIOR_EXPOSURE * past_rate)
        / (1_000.0 + MURDER_INFORMATION_PRIOR_EXPOSURE)
        * 1_000.0
    )


def test_unclean_panel_years_never_enter_the_window():
    accounting, panel = _fixture()
    history = clean_panel_history(panel, config=CONFIG)
    kept = history[history["jurisdiction_id"].eq("01:municipal:place:A") & history["offense"].eq("burglary")]
    assert sorted(kept["year"]) == [2022, 2023, 2024]
    # Partial weight, a fill component, and a short year are each excluded on their own.
    assert 900.0 not in set(history["clean_count"])

    levels = series_kernel_levels(history, config=CONFIG)
    a = levels[
        levels["jurisdiction_id"].eq("01:municipal:place:A") & levels["offense"].eq("burglary")
    ].iloc[0]
    assert a["clean_year_count"] == 3
    assert a["ewa_count"] == pytest.approx(30.0)


def test_negative_reported_counts_are_floored_before_the_kernel():
    accounting, panel = _fixture()
    panel = pd.concat(
        [panel, pd.DataFrame([_panel_row("01:municipal:place:B", "murder", 2023, -5.0)])],
        ignore_index=True,
    )
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    b = _row(out, "01:municipal:place:B", "murder")
    assert b["estimator"] == PRECISION_WEIGHTED_ESTIMATOR
    # The negative report is floored, then its exposure still counts as information.
    assert b["information_weighted_count"] == pytest.approx(4.0)
    assert b["information_weighted_exposure"] == pytest.approx(1_500.0)
    assert b["smoothed_count"] >= 0.0


# --- the fallback path -------------------------------------------------------


def test_one_clean_year_runs_the_tournament_estimator_instead_of_copying_accounting():
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    b = _row(out, "01:municipal:place:B", "murder")
    assert b["estimator"] == PRECISION_WEIGHTED_ESTIMATOR
    assert pd.isna(b["fallback_reason"])
    assert b["ewa_count"] == pytest.approx(4.0)
    peer_rate = 7.25 / 1_750.0
    assert b["peer_predicted_count"] == pytest.approx(peer_rate * 1_000.0)
    past_rate = 1.25 / 750.0
    assert b["smoothed_count"] == pytest.approx(
        (4.0 + MURDER_INFORMATION_PRIOR_EXPOSURE * past_rate)
        / (1_000.0 + MURDER_INFORMATION_PRIOR_EXPOSURE)
        * 1_000.0
    )
    assert b["smoothed_count"] != b["accounting_count"]
    assert b["clean_year_count"] == 1
    assert b["peer_group_id"] == "01:municipal:25k-50k"


def test_zero_exposure_falls_back_even_with_a_full_history():
    accounting, panel = _fixture()
    accounting.loc[
        accounting["jurisdiction_id"].eq("01:municipal:place:A")
        & accounting["offense"].eq("burglary"),
        "bucket_population",
    ] = 0.0
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    a = _row(out, "01:municipal:place:A", "burglary")
    assert a["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert a["fallback_reason"] == NO_EXPOSURE_REASON
    assert a["smoothed_count"] == a["accounting_count"] == 44.0
    # A dropped out of the peer group, so B now has no peer mass to borrow and lands on 0.
    b = _row(out, "01:municipal:place:B", "burglary")
    assert b["peer_rate"] == pytest.approx(0.0)
    assert b["smoothed_count"] == pytest.approx(0.0)


def test_a_unit_with_no_panel_history_at_all_falls_back():
    accounting, panel = _fixture()
    accounting = pd.concat(
        [accounting, pd.DataFrame([_accounting_row("01:municipal:place:C", "burglary", population=500.0, count=9.0)])],
        ignore_index=True,
    )
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    c = _row(out, "01:municipal:place:C", "burglary")
    assert c["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert c["fallback_reason"] == INSUFFICIENT_HISTORY_REASON
    assert c["clean_year_count"] == 0
    assert pd.isna(c["latest_clean_year"])
    assert c["smoothed_count"] == 9.0


def test_one_year_ordinary_unit_without_an_established_peer_falls_back():
    accounting = pd.DataFrame(
        [_accounting_row("04:municipal:place:solo", "robbery", population=7_500.0, count=2.0)]
    )
    panel = pd.DataFrame(
        [_panel_row("04:municipal:place:solo", "robbery", 2024, 2.0)]
    )

    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    row = out.iloc[0]

    assert row["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert row["fallback_reason"] == NO_PEER_SUPPORT_REASON
    assert row["smoothed_count"] == row["accounting_count"] == pytest.approx(2.0)
    assert row[ESTIMATE_DIAGNOSTIC_COLUMNS].isna().all()


def test_verified_replacements_and_structural_zeros_are_not_temporally_rewritten():
    accounting, panel = _fixture()
    accounting["level_repair_mode"] = "none"
    source_row = (
        accounting["jurisdiction_id"].eq("01:municipal:place:A")
        & accounting["offense"].eq("burglary")
    )
    zero_row = (
        accounting["jurisdiction_id"].eq("01:municipal:place:B")
        & accounting["offense"].eq("burglary")
    )
    accounting.loc[source_row, "level_repair_mode"] = "source_supported_replacement"
    accounting.loc[zero_row, "level_repair_mode"] = "locked_structural_zero"
    accounting.loc[zero_row, "adjusted_count_ags_core"] = 0.0

    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    source = _row(out, "01:municipal:place:A", "burglary")
    zero = _row(out, "01:municipal:place:B", "burglary")

    assert source["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert source["fallback_reason"] == SOURCE_REPLACEMENT_PROTECTED_REASON
    assert source["smoothed_count"] == source["accounting_count"] == 44.0
    assert zero["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert zero["fallback_reason"] == STRUCTURAL_ZERO_PROTECTED_REASON
    assert zero["smoothed_count"] == zero["accounting_count"] == 0.0


def test_one_year_units_receive_estimates_without_redefining_established_peer_rates():
    accounting, panel = _fixture()
    # The unit receives an estimate from one clean observation. It is not allowed to move
    # the established reference rate used by A and B.
    accounting = pd.concat(
        [accounting, pd.DataFrame([_accounting_row("01:municipal:place:D", "burglary", population=1000.0, count=5000.0)])],
        ignore_index=True,
    )
    panel = pd.concat(
        [panel, pd.DataFrame([_panel_row("01:municipal:place:D", "burglary", 2024, 5000.0)])],
        ignore_index=True,
    )
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    expected_peer_rate = (30.0 + 0.0) / 2000.0
    assert _row(out, "01:municipal:place:A", "burglary")["peer_rate"] == pytest.approx(expected_peer_rate)
    assert _row(out, "01:municipal:place:D", "burglary")["estimator"] == PRECISION_WEIGHTED_ESTIMATOR


def test_state_remainder_never_defines_a_municipal_peer_rate():
    accounting = pd.DataFrame(
        [
            _accounting_row(
                "09:municipal:place:A", "robbery", population=4_000.0, count=1.0,
                state_fips="09",
            ),
            _accounting_row(
                "09:municipal:place:B", "robbery", population=4_000.0, count=0.0,
                state_fips="09",
            ),
            _accounting_row(
                "09:municipal:place:new", "robbery", population=4_000.0, count=20.0,
                state_fips="09",
            ),
            _accounting_row(
                "09:state_nonmunicipal_remainder",
                "robbery",
                population=1_500.0,
                count=100.0,
                jurisdiction_type="state_nonmunicipal_remainder",
                state_fips="09",
            ),
        ]
    )
    accounting["pop_band"] = "<5k"
    panel = pd.DataFrame(
        [
            _panel_row("09:municipal:place:A", "robbery", 2023, 1.0),
            _panel_row("09:municipal:place:A", "robbery", 2024, 1.0),
            _panel_row("09:municipal:place:B", "robbery", 2023, 0.0),
            _panel_row("09:municipal:place:B", "robbery", 2024, 0.0),
            _panel_row("09:municipal:place:new", "robbery", 2024, 20.0),
            _panel_row("09:state_nonmunicipal_remainder", "robbery", 2023, 100.0),
            _panel_row("09:state_nonmunicipal_remainder", "robbery", 2024, 100.0),
        ]
    )

    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    municipal_peer_rate = 1.0 / 8_000.0
    for jurisdiction_id in (
        "09:municipal:place:A",
        "09:municipal:place:B",
        "09:municipal:place:new",
    ):
        row = _row(out, jurisdiction_id, "robbery")
        assert row["peer_rate"] == pytest.approx(municipal_peer_rate)
        assert row["peer_group_id"] == "09:municipal:<5k"

    remainder = _row(out, "09:state_nonmunicipal_remainder", "robbery")
    assert remainder["peer_rate"] == pytest.approx(100.0 / 1_500.0)
    assert remainder["smoothed_count"] == pytest.approx(100.0)
    assert remainder["peer_group_id"] == "09:state_nonmunicipal_remainder:<5k"


def test_incomplete_state_remainder_is_floored_by_coverage_and_reliable_peer_rate():
    accounting = pd.DataFrame(
        [
            _accounting_row(
                "01:municipal:place:A", "robbery", population=100_000.0, count=100.0
            ),
            _accounting_row(
                "01:state_nonmunicipal_remainder",
                "robbery",
                population=100_000.0,
                count=15.0,
                jurisdiction_type="state_nonmunicipal_remainder",
            ),
            _accounting_row(
                "01:municipal:place:large", "robbery", population=150_000.0, count=10_000.0
            ),
            _accounting_row(
                "02:municipal:place:B",
                "robbery",
                population=100_000.0,
                count=100.0,
                state_fips="02",
            ),
            _accounting_row(
                "02:state_nonmunicipal_remainder",
                "robbery",
                population=100_000.0,
                count=1.0,
                jurisdiction_type="state_nonmunicipal_remainder",
                state_fips="02",
            ),
            _accounting_row(
                "03:municipal:place:C",
                "robbery",
                population=100_000.0,
                count=100.0,
                state_fips="03",
            ),
            _accounting_row(
                "03:state_nonmunicipal_remainder",
                "robbery",
                population=100_000.0,
                count=100.0,
                jurisdiction_type="state_nonmunicipal_remainder",
                state_fips="03",
            ),
        ]
    )
    accounting.loc[
        accounting["jurisdiction_id"].eq("01:municipal:place:large"), "pop_band"
    ] = "100k-250k"
    panel = pd.DataFrame(
        [
            _panel_row("01:municipal:place:A", "robbery", 2023, 100.0),
            _panel_row("01:municipal:place:A", "robbery", 2024, 100.0),
            _panel_row("01:state_nonmunicipal_remainder", "robbery", 2023, 15.0),
            _panel_row("01:state_nonmunicipal_remainder", "robbery", 2024, 15.0),
            _panel_row("01:municipal:place:large", "robbery", 2023, 10_000.0),
            _panel_row("01:municipal:place:large", "robbery", 2024, 10_000.0),
            _panel_row("02:municipal:place:B", "robbery", 2023, 100.0),
            _panel_row("02:municipal:place:B", "robbery", 2024, 100.0),
            _panel_row("03:municipal:place:C", "robbery", 2023, 100.0),
            _panel_row("03:municipal:place:C", "robbery", 2024, 100.0),
            # A state remainder is a pooled residual rather than an ordinary unit. One
            # year cannot make it a recipient or a cross-state reference.
            _panel_row("03:state_nonmunicipal_remainder", "robbery", 2024, 100.0),
            _panel_row(
                "02:state_nonmunicipal_remainder",
                "robbery",
                2024,
                1.0,
                months=6.0,
                weight=0.5,
            ),
        ]
    )

    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    repaired = _row(out, "02:state_nonmunicipal_remainder", "robbery")
    one_year_remainder = _row(out, "03:state_nonmunicipal_remainder", "robbery")

    # The two-year supported state establishes the remainder:small-municipality rate ratio;
    # its 150,000-person city is deliberately excluded. State 03's one-year remainder
    # cannot borrow its municipality as a peer. State 02 has no complete panel year, so its
    # incomplete accounting residual is only a lower bound and the full-coverage peer
    # prediction is used.
    supported_remainder = _row(out, "01:state_nonmunicipal_remainder", "robbery")
    supported_municipal = _row(out, "01:municipal:place:A", "robbery")
    expected_ratio = (
        supported_remainder["smoothed_count"] / supported_remainder["bucket_population"]
    ) / (supported_municipal["smoothed_count"] / supported_municipal["bucket_population"])
    state_02_municipal = _row(out, "02:municipal:place:B", "robbery")
    expected_peer = expected_ratio * state_02_municipal["smoothed_count"]
    expected_blend = expected_peer
    assert repaired["estimator"] == COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR
    assert repaired["reporting_coverage_weight"] == pytest.approx(0.5)
    assert repaired["remainder_to_municipal_rate_ratio"] == pytest.approx(expected_ratio)
    assert repaired["coverage_municipal_reference_population"] == pytest.approx(100_000.0)
    assert repaired["coverage_municipal_reference_unit_count"] == pytest.approx(1.0)
    assert repaired["coverage_training_remainder_population"] == pytest.approx(100_000.0)
    assert repaired["coverage_training_remainder_unit_count"] == pytest.approx(1.0)
    assert repaired["coverage_training_municipal_population"] == pytest.approx(100_000.0)
    assert repaired["coverage_training_municipal_unit_count"] == pytest.approx(1.0)
    assert repaired["coverage_peer_predicted_count"] == pytest.approx(expected_peer)
    assert repaired["coverage_blended_count"] == pytest.approx(expected_blend)
    assert repaired["smoothed_count"] == pytest.approx(expected_blend)
    assert repaired["accounting_count"] == pytest.approx(1.0)
    assert pd.isna(repaired["fallback_reason"])
    assert repaired[ESTIMATE_DIAGNOSTIC_COLUMNS].isna().all()
    assert one_year_remainder["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert one_year_remainder["fallback_reason"] == INSUFFICIENT_HISTORY_REASON
    assert one_year_remainder["smoothed_count"] == pytest.approx(
        one_year_remainder["accounting_count"]
    )
    assert one_year_remainder[ESTIMATE_DIAGNOSTIC_COLUMNS].isna().all()
    assert_smoothed_control_invariants(smoothed=out, accounting=accounting, config=CONFIG)


def test_state_remainder_coverage_floor_never_removes_accounting_mass():
    accounting = pd.DataFrame(
        [
            _accounting_row(
                "01:municipal:place:A", "robbery", population=100_000.0, count=100.0
            ),
            _accounting_row(
                "01:state_nonmunicipal_remainder",
                "robbery",
                population=100_000.0,
                count=15.0,
                jurisdiction_type="state_nonmunicipal_remainder",
            ),
            _accounting_row(
                "02:municipal:place:B",
                "robbery",
                population=100_000.0,
                count=100.0,
                state_fips="02",
            ),
            _accounting_row(
                "02:state_nonmunicipal_remainder",
                "robbery",
                population=100_000.0,
                count=30.0,
                jurisdiction_type="state_nonmunicipal_remainder",
                state_fips="02",
            ),
        ]
    )
    panel = pd.DataFrame(
        [
            _panel_row("01:municipal:place:A", "robbery", 2023, 100.0),
            _panel_row("01:municipal:place:A", "robbery", 2024, 100.0),
            _panel_row("01:state_nonmunicipal_remainder", "robbery", 2023, 15.0),
            _panel_row("01:state_nonmunicipal_remainder", "robbery", 2024, 15.0),
            _panel_row("02:municipal:place:B", "robbery", 2023, 100.0),
            _panel_row("02:municipal:place:B", "robbery", 2024, 100.0),
            _panel_row(
                "02:state_nonmunicipal_remainder",
                "robbery",
                2024,
                30.0,
                months=6.0,
                weight=0.5,
            ),
        ]
    )

    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    retained = _row(out, "02:state_nonmunicipal_remainder", "robbery")

    assert retained["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert retained["smoothed_count"] == retained["accounting_count"] == 30.0


# --- schema, universe, invariants --------------------------------------------


def test_schema_and_universe_match_the_accounting_surface():
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    assert list(out.columns) == SMOOTHED_CONTROL_COLUMNS
    for column in CONTROL_KEY_COLUMNS:
        assert column in out.columns
    assert len(out) == len(accounting)
    assert not out.duplicated(["jurisdiction_id", "offense"]).any()
    assert set(map(tuple, out[["jurisdiction_id", "offense"]].to_numpy())) == set(
        map(tuple, accounting[["jurisdiction_id", "offense"]].to_numpy())
    )
    assert out["year"].eq(2024).all()
    assert out["smoothed_count"].ge(0.0).all()
    assert out["estimator"].notna().all()
    estimated = out["estimator"].isin(
        {DYNAMIC_SHRINK_ESTIMATOR, PRECISION_WEIGHTED_ESTIMATOR, INFORMATION_POSTERIOR_ESTIMATOR}
    )
    assert out.loc[estimated, "fallback_reason"].isna().all()
    assert out.loc[~estimated, "fallback_reason"].notna().all()
    assert_smoothed_control_invariants(smoothed=out, accounting=accounting, config=CONFIG)


def test_the_surface_conserves_nothing_to_the_accounting_year():
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    # By design: no raking step exists, so the totals differ.
    assert out["smoothed_count"].sum() != pytest.approx(out["accounting_count"].sum())


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda frame: frame.assign(
                smoothed_count=frame["smoothed_count"].where(
                    frame["estimator"].ne(FALLBACK_ACCOUNTING_ESTIMATOR), 0.0
                )
            ),
            id="fallback_value_diverges_from_accounting",
        ),
        pytest.param(
            lambda frame: frame.assign(
                smoothed_count=frame["smoothed_count"] * 1.05
            ),
            id="estimated_row_no_longer_recomposes",
        ),
        pytest.param(
            lambda frame: frame.assign(smoothed_count=-frame["smoothed_count"]),
            id="negative_counts",
        ),
        pytest.param(
            lambda frame: frame.assign(
                fallback_reason=frame["fallback_reason"].astype("string").where(
                    frame["estimator"].ne(FALLBACK_ACCOUNTING_ESTIMATOR), pd.NA
                )
            ),
            id="undocumented_fallback",
        ),
        pytest.param(lambda frame: frame.iloc[1:], id="missing_unit"),
    ],
)
def test_invariants_reject_a_broken_surface(mutate):
    accounting, panel = _fixture()
    accounting = pd.concat(
        [
            accounting,
            pd.DataFrame(
                [
                    _accounting_row(
                        "01:municipal:place:C", "burglary", population=500.0, count=9.0
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)
    with pytest.raises(ValueError):
        assert_smoothed_control_invariants(
            smoothed=mutate(out), accounting=accounting, config=CONFIG
        )


def test_duplicate_accounting_keys_are_refused():
    accounting, panel = _fixture()
    doubled = pd.concat([accounting, accounting.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_smoothed_controls(accounting=doubled, panel=panel, config=CONFIG)


def test_an_offense_without_a_configured_kernel_is_refused():
    accounting, panel = _fixture()
    accounting = pd.concat(
        [accounting, pd.DataFrame([_accounting_row("01:municipal:place:A", "arson", population=1000.0, count=1.0)])],
        ignore_index=True,
    )
    panel = pd.concat(
        [
            panel,
            pd.DataFrame(
                [
                    _panel_row("01:municipal:place:A", "arson", 2023, 1.0),
                    _panel_row("01:municipal:place:A", "arson", 2024, 2.0),
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="temporal kernel"):
        build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)


# --- the build entry point ---------------------------------------------------


def test_write_is_cached_on_content_and_refuses_to_rebuild_the_accounting_surface(tmp_path: Path):
    paths = RepoPaths.from_repo_root(tmp_path)
    controls_dir = paths.state_dir / "controls"
    controls_dir.mkdir(parents=True)
    accounting, panel = _fixture()
    accounting_path = controls_dir / "jurisdiction_controls_2024.parquet"

    with pytest.raises(FileNotFoundError, match="build-controls"):
        write_v2_smoothed_controls(paths=paths, config=CONFIG)

    accounting.to_parquet(accounting_path, index=False)
    panel.to_parquet(controls_dir / "jurisdiction_year_estimates.parquet", index=False)
    cde_path = (
        paths.data_dir
        / "FBI-CDE-Estimates-1979-2024"
        / "estimated_crimes_1979_2024.csv"
    )
    cde_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "year": year,
                "state_abbr": "AL",
                # Within the accounting-agreement tolerance of the fixture's 12 murders.
                "homicide": 12.0,
                "rape_revised": 20.0,
                "robbery": 30.0,
                "aggravated_assault": 40.0,
                "burglary": 50.0,
                "larceny": 60.0,
                "motor_vehicle_theft": 70.0,
            }
            for year in range(2018, 2025)
        ]
    ).to_csv(cde_path, index=False)
    # The state anchor's inputs: the controls stage's admission and mass ledger, the observations,
    # and the (empty) exception register. Missing ones are an error, not "no corrections".
    pd.DataFrame(columns=["ori9", "state_abbr", "year", "offense", "level_lane_hard_case_id", "level_repair_mode"]).to_parquet(
        controls_dir / "level_lane_admission_2024.parquet", index=False)
    pd.DataFrame({"ori9": pd.Series(dtype="string"), "year": pd.Series(dtype="int64"), "offense": pd.Series(dtype="string"),
                  "final_control_mass": pd.Series(dtype="float64")}).to_parquet(controls_dir / "level_lane_mass_ledger_2024.parquet", index=False)
    (paths.state_dir / "observations").mkdir(parents=True)
    pd.DataFrame({"ori9": pd.Series(dtype="string"), "year": pd.Series(dtype="int64"), "source": pd.Series(dtype="string"),
                  "offense": pd.Series(dtype="string"), "count": pd.Series(dtype="float64"), "months_reported": pd.Series(dtype="float64")}).to_parquet(
        paths.state_dir / "observations" / "agency_year_observations.parquet", index=False)
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "state_anchor_exceptions.csv").write_text("state_abbr,offense,effective_from_year,effective_to_year,evidence_source,note\n")
    accounting_bytes = accounting_path.read_bytes()

    out_path, summary = write_v2_smoothed_controls(paths=paths, config=CONFIG)
    assert out_path.exists()
    assert smoothed_controls_summary_path(paths, year=2024).exists()
    assert summary["units"] == len(accounting)
    assert summary["fallback_units"] == 1
    assert summary["negative_smoothed_counts"] == 0
    # Surface 1 is opened read-only; the build may not touch it.
    assert accounting_path.read_bytes() == accounting_bytes

    stamped = out_path.stat().st_mtime_ns
    _, cached = write_v2_smoothed_controls(paths=paths, config=CONFIG)
    assert out_path.stat().st_mtime_ns == stamped
    assert cached == summary

    # A changed input invalidates the content-hash stamp.
    moved = panel["jurisdiction_id"].eq("01:municipal:place:A") & panel["offense"].eq(
        "burglary"
    ) & panel["year"].eq(2024)
    panel.loc[moved, ["reported_count_preferred", "estimated_count", "full_year_admitted_count"]] = 80.0
    panel.to_parquet(controls_dir / "jurisdiction_year_estimates.parquet", index=False)
    _, rebuilt = write_v2_smoothed_controls(paths=paths, config=CONFIG)
    assert out_path.stat().st_mtime_ns != stamped
    assert (
        rebuilt["national_totals"]["burglary"]["precalibration_smoothed_total"]
        > summary["national_totals"]["burglary"]["precalibration_smoothed_total"]
    )
    assert rebuilt["national_totals"]["burglary"]["smoothed_total"] == pytest.approx(
        summary["national_totals"]["burglary"]["national_anchor_count"]
    )


def test_coverage_lift_is_capped_by_the_lane_own_agency_evidence(monkeypatch):
    """(e) A remainder whose agencies the ladder already estimated cannot be lifted again."""
    from crimerisk.smoothed_controls import _apply_state_remainder_coverage_floor

    def frames(evidence_agencies: int):
        out = pd.DataFrame(
            [
                # Two reliable remainders train the ratio.
                {"jurisdiction_id": "01:state_nonmunicipal_remainder", "jurisdiction_type": "state_nonmunicipal_remainder",
                 "state_fips": "01", "offense": "burglary", "bucket_population": 2_000_000.0,
                 "pop_band": "1m+", "smoothed_count": 4000.0, "estimator": "dynamic_shrink",
                 "fallback_reason": pd.NA, "clean_year_count": 5},
                {"jurisdiction_id": "02:state_nonmunicipal_remainder", "jurisdiction_type": "state_nonmunicipal_remainder",
                 "state_fips": "02", "offense": "burglary", "bucket_population": 2_000_000.0,
                 "pop_band": "1m+", "smoothed_count": 4000.0, "estimator": "dynamic_shrink",
                 "fallback_reason": pd.NA, "clean_year_count": 5},
                # The lane under test: a fallback accounting residual far below the peer rate.
                {"jurisdiction_id": "21:state_nonmunicipal_remainder", "jurisdiction_type": "state_nonmunicipal_remainder",
                 "state_fips": "21", "offense": "burglary", "bucket_population": 2_000_000.0,
                 "pop_band": "1m+", "smoothed_count": 1000.0, "estimator": "fallback_accounting",
                 "fallback_reason": "insufficient_clean_history", "clean_year_count": 0},
            ]
            + [
                {"jurisdiction_id": f"{fips}:municipal:x", "jurisdiction_type": "municipal",
                 "state_fips": fips, "offense": "burglary", "bucket_population": 1_000_000.0,
                 "pop_band": "25k-50k", "smoothed_count": 4000.0, "estimator": "dynamic_shrink",
                 "fallback_reason": pd.NA, "clean_year_count": 5}
                for fips in ("01", "02", "21")
            ]
        )
        panel = pd.DataFrame(
            [
                {"jurisdiction_id": jid, "offense": "burglary", "year": 2025,
                 "observation_weight_preferred": 0.0}
                for jid in out["jurisdiction_id"]
            ]
        )
        accounting = pd.DataFrame(
            [
                {"jurisdiction_id": "21:state_nonmunicipal_remainder", "offense": "burglary",
                 "crosswalk_agency_count": 100, "contributing_agency_count": 0,
                 "estimating_agency_count": evidence_agencies}
            ]
        )
        return out, panel, accounting

    def lifted(rules: str, evidence_agencies: int) -> float:
        monkeypatch.setenv("CRIMERISK_LEVEL_RULES", rules)
        out, panel, accounting = frames(evidence_agencies)
        result = _apply_state_remainder_coverage_floor(
            out,
            panel=panel,
            config=SmoothedControlConfig(year=2025),
            accounting_controls=accounting,
        )
        row = result[result["jurisdiction_id"].eq("21:state_nonmunicipal_remainder")]
        return float(row["smoothed_count"].iloc[0])

    uncapped = lifted("none", 90)
    assert uncapped > 1000.0  # the peer rate lifts it today
    capped = lifted("remainder_coverage_cap", 90)
    # 90 of 100 agencies already carry an estimate, so at most a 1/0.9 gross-up.
    assert capped == pytest.approx(1000.0 / 0.9, rel=1e-6)
    assert capped < uncapped
    # A lane whose agencies really are missing keeps its lift.
    assert lifted("remainder_coverage_cap", 2) == pytest.approx(uncapped, rel=1e-6)


def test_national_anchor_is_the_data_year_estimate_not_a_weighted_history():
    """The published estimand is the data year: a falling series must not be anchored high."""
    from crimerisk.smoothed_controls import (
        assert_national_anchor_agrees_with_accounting,
        national_anchor_from_cde_data_year,
    )

    config = SmoothedControlConfig(year=2025)
    history = pd.DataFrame(
        [
            {"offense": offense, "year": year, "cde_total": 100.0 - 5.0 * (year - 2018)}
            for offense, _ in config.halflife_years
            for year in range(2018, 2026)
        ]
    )
    anchor = national_anchor_from_cde_data_year(history, config=config).set_index("offense")
    assert anchor["national_anchor_count"].eq(65.0).all()
    with pytest.raises(ValueError, match="exactly one 2025"):
        national_anchor_from_cde_data_year(history[history["year"].lt(2025)], config=config)

    agree = {"national_totals": {"murder": {"accounting_over_anchor": 1.05}}}
    assert_national_anchor_agrees_with_accounting(agree)
    with pytest.raises(ValueError, match="disagree"):
        assert_national_anchor_agrees_with_accounting(
            {"national_totals": {"murder": {"accounting_over_anchor": 1.17}}}
        )


def test_state_anchor_is_cde_plus_documented_corrections_and_recorded_exceptions():
    from crimerisk.smoothed_controls import (
        STATE_ANCHOR_CDE_SOURCE,
        STATE_ANCHOR_CORRECTED_SOURCE,
        STATE_ANCHOR_EXCEPTION_SOURCE,
        _apply_state_anchor_calibration,
        documented_agency_corrections,
        state_anchor_from_cde_data_year,
    )

    accounting = pd.DataFrame(
        {
            "state_abbr": ["AA", "AA", "BB", "CC"],
            "offense": ["robbery"] * 4,
            "adjusted_count_ags_core": [60.0, 40.0, 150.0, 90.0],
        }
    )
    cde = pd.DataFrame({"state_abbr": ["AA", "BB", "CC"], "offense": ["robbery"] * 3, "cde_state_count": [110.0, 100.0, 50.0]})
    # BB: one agency under a documented restriction; the FBI used its filed 20, we carry 70.
    # An agency under a correction that the FBI never had (imputed) adds nothing.
    admission = pd.DataFrame({"ori9": ["BB0000100", "BB0000200"], "state_abbr": "BB", "year": 2025, "offense": "robbery",
                              "level_lane_hard_case_id": ["bb-city-2021", "bb-town-2022"]})
    ledger = pd.DataFrame({"ori9": ["BB0000100", "BB0000200"], "year": 2025, "offense": "robbery", "final_control_mass": [70.0, 9.0]})
    fbi = pd.DataFrame({"ori9": ["BB0000100"], "year": 2025, "source": "srs_return_a_annual", "offense": "robbery",
                        "count": [20.0], "months_reported": [12.0]})
    corrections = documented_agency_corrections(admission=admission, mass_ledger=ledger, fbi_observations=fbi, year=2025)
    # Both documented agencies are in the contribution ledger; only the FBI's twelve-month filer is
    # included (the other's CDE contribution is the FBI's own imputation, which stands).
    assert corrections["ori9"].tolist() == ["BB0000100", "BB0000200"]
    assert corrections["inclusion_status"].tolist() == ["fbi_used_own_12_month_report", "excluded_fbi_imputed_nonreporter"]
    assert corrections["correction"].tolist() == [50.0, 0.0]
    assert corrections["model_dependent"].tolist() == [True, True]  # no source-supported replacement mode
    exceptions = pd.DataFrame({"state_abbr": ["CC"], "offense": ["robbery"], "effective_from_year": ["2024"], "effective_to_year": ["2025"],
                               "evidence_source": ["state program not transmitting"], "note": [""]})
    anchor = state_anchor_from_cde_data_year(cde, accounting=accounting, corrections=corrections, exceptions=exceptions).set_index("state_abbr")
    assert anchor.loc["AA", "state_anchor_source"] == STATE_ANCHOR_CDE_SOURCE
    assert anchor.loc["AA", "state_anchor_count"] == 110.0
    # BB: 100 + (70 - 20), whatever the accounting total is (150 here, 1.5x CDE).
    assert anchor.loc["BB", "state_anchor_source"] == STATE_ANCHOR_CORRECTED_SOURCE
    assert anchor.loc["BB", "state_anchor_count"] == 150.0 - 0.0
    assert anchor.loc["CC", "state_anchor_source"] == STATE_ANCHOR_EXCEPTION_SOURCE
    assert anchor.loc["CC", "state_anchor_count"] == 90.0
    smoothed = pd.DataFrame(
        {"state_abbr": ["AA", "AA", "BB", "CC"], "offense": ["robbery"] * 4, "smoothed_count": [30.0, 70.0, 120.0, 80.0]}
    )
    raked = _apply_state_anchor_calibration(smoothed, state_anchor=anchor.reset_index())
    assert raked.groupby("state_abbr")["smoothed_count"].sum().to_dict() == {"AA": 110.0, "BB": 150.0, "CC": 90.0}
    assert raked["smoothed_count"].iloc[0] / raked["smoothed_count"].iloc[1] == 30.0 / 70.0


def test_state_anchor_moves_continuously_with_its_inputs():
    """No threshold: a small revision to accounting or to a corrected count moves the target by at most
    that revision (the old rule jumped from the CDE estimate to the accounting total at 1.2x)."""
    from crimerisk.smoothed_controls import state_anchor_from_cde_data_year

    cde = pd.DataFrame({"state_abbr": ["NE"], "offense": ["murder"], "cde_state_count": [46.0]})
    targets = []
    for accounting_total in (54.0, 55.0, 56.0):  # 1.17x, 1.20x, 1.22x CDE
        accounting = pd.DataFrame({"state_abbr": ["NE"], "offense": ["murder"], "adjusted_count_ags_core": [accounting_total]})
        targets.append(float(state_anchor_from_cde_data_year(cde, accounting=accounting)["state_anchor_count"].iloc[0]))
    assert targets == [46.0, 46.0, 46.0]
    for corrected in (60.0, 61.0):
        corrections = pd.DataFrame({"state_abbr": ["NE"], "offense": ["murder"], "ori9": ["NE0000100"],
                                    "correction": [corrected - 26.0]})
        accounting = pd.DataFrame({"state_abbr": ["NE"], "offense": ["murder"], "adjusted_count_ags_core": [80.0]})
        targets.append(float(state_anchor_from_cde_data_year(cde, accounting=accounting, corrections=corrections)["state_anchor_count"].iloc[0]))
    assert targets[-1] - targets[-2] == 1.0


def test_current_year_check_flags_a_large_unit_far_below_its_forecast_and_its_peers():
    from crimerisk.smoothed_controls import current_year_check

    out = pd.DataFrame({
        "jurisdiction_id": ["big-drop", "big-ok", "small-drop", "big-drop-confirmed", "p1", "p2", "p3", "p4"],
        "offense": "robbery", "pop_band": "100k-250k",
        "bucket_population": [200_000.0, 200_000.0, 20_000.0] + [200_000.0] * 5,
        "current_year_count": [40.0, 95.0, 4.0, 40.0, 100.0, 100.0, 100.0, 100.0],
        "history_forecast_count": [100.0] * 8,
    })
    confirmations = pd.DataFrame({"jurisdiction_id": ["big-drop-confirmed"], "offense": ["robbery"], "year": ["2025"],
                                  "verdict": ["confirmed_real"], "evidence_source": ["city annual report"], "note": [""]})
    status = current_year_check(out, confirmations=confirmations, year=2025).astype(object).where(lambda x: x.notna(), None).tolist()
    # Peer median ratio 0.975: 0.40 is below half of it and below half the forecast.
    assert status[0] == "unverified"
    assert status[1] is None and status[2] is None  # in line / under 25k residents
    assert status[3] == "confirmed"


def test_the_current_year_check_is_a_review_flag_and_never_moves_a_control():
    """A large unit's count far below its forecast is flagged, and the control is the ordinary
    precision-weighted blend either side of the flag's threshold (no cliff at the cutoff)."""
    years = list(range(2018, CONFIG.year + 1))
    rows, acc = [], []
    for i in range(40):
        jid = f"01:municipal:place:{i:05d}"
        for year in years:
            count = 10_000.0 if year < CONFIG.year else (4_999.0 if i == 0 else 5_000.0 if i == 1 else 10_000.0)
            rows.append(_panel_row(jid, "burglary", year, count, population=30_000.0))
        acc.append(_accounting_row(jid, "burglary", population=30_000.0, count=10_000.0 if i > 1 else (4_999.0 if i == 0 else 5_000.0)))
    out = build_smoothed_controls(accounting=pd.DataFrame(acc), panel=pd.DataFrame(rows), config=CONFIG).set_index("jurisdiction_id")
    flagged, unflagged = out.loc["01:municipal:place:00000"], out.loc["01:municipal:place:00001"]
    assert flagged["current_year_check"] == "unverified" and pd.isna(unflagged["current_year_check"])
    for unit in (flagged, unflagged):
        w = unit["current_year_weight"]
        assert 0.0 < w < 1.0
        assert unit["smoothed_count"] == pytest.approx(w * unit["current_year_count"] + (1 - w) * unit["history_forecast_count"])
    assert flagged["smoothed_count"] < unflagged["smoothed_count"]
    assert unflagged["smoothed_count"] - flagged["smoothed_count"] < 2.0


def test_a_confirmation_counts_only_with_the_confirmed_verdict(tmp_path):
    from crimerisk.smoothed_controls import current_year_check

    out = pd.DataFrame({
        "jurisdiction_id": ["drop", "p1", "p2", "p3"], "offense": "robbery", "pop_band": "100k-250k",
        "bucket_population": 200_000.0, "current_year_count": [40.0, 100.0, 100.0, 100.0],
        "history_forecast_count": 100.0,
    })
    wrong = pd.DataFrame({"jurisdiction_id": ["drop"], "offense": ["robbery"], "year": ["2025"],
                          "verdict": ["refuted"], "evidence_source": ["x"], "note": [""]})
    assert current_year_check(out, confirmations=wrong, year=2025).iloc[0] == "unverified"


def test_mislabelled_cde_state_row_is_resolved_by_signature_in_any_row_order():
    from crimerisk.smoothed_controls import _resolve_mislabelled_state_rows

    def row(state, year, scale):
        return {"state_abbr": state, "year": year, "homicide": 10 * scale, "rape_revised": 50 * scale, "robbery": 30 * scale,
                "aggravated_assault": 200 * scale, "burglary": 300 * scale, "larceny": 1000 * scale, "motor_vehicle_theft": 100 * scale}

    frame = pd.DataFrame([row("NC", 2021, 30), row("NC", 2023, 30), row("ND", 2021, 1), row("ND", 2023, 1),
                          row("NC", 2022, 1.05), row("NC", 2022, 31)])
    for order in (frame, frame.iloc[::-1]):
        out = _resolve_mislabelled_state_rows(order.reset_index(drop=True))
        nd = out[(out["state_abbr"] == "ND") & (out["year"] == 2022)]
        assert len(nd) == 1 and float(nd["larceny"].iloc[0]) == 1050.0
    ambiguous = frame.copy()
    ambiguous.loc[4, "larceny"] = 31000  # one offense now points the other way
    with pytest.raises(ValueError, match="signature"):
        _resolve_mislabelled_state_rows(ambiguous)


def test_a_contribution_needs_finite_mass_and_agreeing_fbi_sources():
    from crimerisk.smoothed_controls import StateAnchorError, documented_agency_corrections

    admission = pd.DataFrame({"ori9": ["BB0000100"], "state_abbr": "BB", "year": 2025, "offense": "robbery",
                              "level_lane_hard_case_id": ["bb-city"]})
    fbi = pd.DataFrame({"ori9": ["BB0000100"], "year": 2025, "source": "srs_return_a_annual", "offense": "robbery",
                        "count": [100.0], "months_reported": [12.0]})
    # A matching ledger row whose mass is missing is an error, not a corrected count of 0.
    null_mass = pd.DataFrame({"ori9": ["BB0000100"], "year": 2025, "offense": "robbery", "final_control_mass": [np.nan]})
    with pytest.raises(StateAnchorError):
        documented_agency_corrections(admission=admission, mass_ledger=null_mass, fbi_observations=fbi, year=2025)
    # FBI sources that disagree on the twelve-month count leave the contribution unidentified.
    ledger = pd.DataFrame({"ori9": ["BB0000100"], "year": 2025, "offense": "robbery", "final_control_mass": [150.0]})
    two = pd.concat([fbi, fbi.assign(source="cius_publication_annual", count=[90.0])], ignore_index=True)
    out = documented_agency_corrections(admission=admission, mass_ledger=ledger, fbi_observations=two, year=2025)
    assert out["inclusion_status"].tolist() == ["excluded_fbi_sources_disagree"]
    assert out["correction"].tolist() == [0.0]
    assert "srs_return_a_annual=100/12m" in out["fbi_source_evidence"].iloc[0]
    agree = documented_agency_corrections(admission=admission, mass_ledger=ledger,
                                          fbi_observations=pd.concat([fbi, fbi.assign(source="cius_publication_annual")]), year=2025)
    assert agree["inclusion_status"].tolist() == ["fbi_used_own_12_month_report"] and agree["correction"].tolist() == [50.0]


def test_own_history_stands_when_no_peer_exists():
    from crimerisk.smoothed_controls import precision_weighted_levels

    history = pd.DataFrame({"jurisdiction_id": "only", "offense": "robbery", "year": list(range(2018, 2025)),
                            "clean_count": 100.0, "clean_exposure": 1000.0})
    units = pd.DataFrame({"jurisdiction_id": ["only"], "offense": "robbery", "state_abbr": "AL", "state_fips": "01",
                          "jurisdiction_type": "municipal", "pop_band": "<5k", "bucket_population": 1000.0})
    out = precision_weighted_levels(units, history, config=SmoothedControlConfig(year=2024), state_year_totals=None).iloc[0]
    assert out["history_forecast_count"] == pytest.approx(100.0)
    assert out["precision_weighted_count"] == pytest.approx(100.0)


def test_peer_dispersion_enters_the_prior_variance_once():
    """A unit with no history of its own: V = tau2_peer H^2, not twice that."""
    from crimerisk.smoothed_controls import SIGNAL_VARIANCE_MIN_SUPPORT, precision_weighted_levels

    units, history = _precision_units(20, 0)
    rows = [history]
    extra = []
    # Newcomers: each unit's only clean year is its first, so every one is a peer-only forecast pair.
    for year in range(2019, 2025):
        for i in range(8):
            jid = f"new{year}_{i}"
            rows.append(pd.DataFrame([{"jurisdiction_id": jid, "offense": "robbery", "year": year,
                                       "clean_count": 1000.0 * (1.3 if i % 2 else 0.7), "clean_exposure": 1000.0}]))
            extra.append({"jurisdiction_id": jid, "offense": "robbery", "state_abbr": "AL", "state_fips": "01",
                          "jurisdiction_type": "municipal", "pop_band": "<5k", "bucket_population": 1000.0})
    history = pd.concat(rows, ignore_index=True)
    units = pd.concat([units, pd.DataFrame(extra)], ignore_index=True)
    out = precision_weighted_levels(units, history, config=SmoothedControlConfig(year=2024), state_year_totals=None).set_index("jurisdiction_id")
    new = out.loc["new2024_1"]
    assert new["signal_variance_level"] != "fallback_one_more_year"
    assert new["signal_variance_support"] >= SIGNAL_VARIANCE_MIN_SUPPORT
    assert new["history_forecast_variance"] == pytest.approx(0.0)
    h, tau2 = new["history_forecast_count"], new["signal_relative_variance"]
    assert tau2 > 0
    assert new["current_year_weight"] == pytest.approx(tau2 * h**2 / (tau2 * h**2 + h))
