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

    loaded = _load_controls(paths, year=2024, surface="smoothed")

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
        _load_controls(paths, year=2024, surface="smoothed")


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

    weights (2^-2, 2^-1, 2^0) = (0.25, 0.5, 1.0), sum 1.75; weighted sum 52.5; EWA 30.
    B: two clean zeros -> EWA 0. Peer group carries EWA 30 over 2,000 residents, so
    peer_rate = 0.015 and each unit's peer prediction is 15.
    A: w = 30/30.5, pred = (30*30 + 0.5*15)/30.5. B: w = 0, pred = 15.
    """
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    a = _row(out, "01:municipal:place:A", "burglary")
    assert a["estimator"] == DYNAMIC_SHRINK_ESTIMATOR
    assert a["clean_year_count"] == 3
    assert a["latest_clean_year"] == 2024
    assert a["halflife_used"] == 1.0
    assert a["ewa_count"] == pytest.approx(30.0)
    assert a["peer_rate"] == pytest.approx(0.015)
    assert a["peer_predicted_count"] == pytest.approx(15.0)
    assert a["ewa_weight"] == pytest.approx(30.0 / 30.5)
    assert a["smoothed_count"] == pytest.approx((30.0 * 30.0 + 0.5 * 15.0) / 30.5)
    # The smoothed control is not the accounting control, and is not raked to it.
    assert a["accounting_count"] == 44.0

    b = _row(out, "01:municipal:place:B", "burglary")
    assert b["ewa_count"] == pytest.approx(0.0)
    assert b["ewa_weight"] == pytest.approx(0.0)
    assert b["smoothed_count"] == pytest.approx(15.0)


def test_murder_uses_information_weighting_and_one_year_series_borrow_from_established_peers():
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    a = _row(out, "01:municipal:place:A", "murder")
    # A cannot use itself as a peer, and B's one clean year cannot define its peer.
    assert a["estimator"] == FALLBACK_ACCOUNTING_ESTIMATOR
    assert a["fallback_reason"] == NO_PEER_SUPPORT_REASON

    b = _row(out, "01:municipal:place:B", "murder")
    assert b["estimator"] == INFORMATION_POSTERIOR_ESTIMATOR
    assert b["temporal_kernel"] == "ewa"
    assert b["halflife_used"] == pytest.approx(1.0)
    # A's weighted history is (.25*1 + .5*2 + 1*6) events over 1,750 person-years.
    peer_rate = 7.25 / 1_750.0
    assert b["peer_rate"] == pytest.approx(peer_rate)
    assert b["ewa_weight"] == pytest.approx(1_000.0 / 11_000.0)
    assert b["smoothed_count"] == pytest.approx(
        (4.0 + MURDER_INFORMATION_PRIOR_EXPOSURE * peer_rate)
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
    assert b["estimator"] == INFORMATION_POSTERIOR_ESTIMATOR
    # The negative report is floored, then its exposure still counts as information.
    assert b["information_weighted_count"] == pytest.approx(4.0)
    assert b["information_weighted_exposure"] == pytest.approx(1_500.0)
    assert b["smoothed_count"] >= 0.0


# --- the fallback path -------------------------------------------------------


def test_one_clean_year_runs_the_tournament_estimator_instead_of_copying_accounting():
    accounting, panel = _fixture()
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=CONFIG)

    b = _row(out, "01:municipal:place:B", "murder")
    assert b["estimator"] == INFORMATION_POSTERIOR_ESTIMATOR
    assert pd.isna(b["fallback_reason"])
    assert b["ewa_count"] == pytest.approx(4.0)
    peer_rate = 7.25 / 1_750.0
    assert b["peer_predicted_count"] == pytest.approx(peer_rate * 1_000.0)
    assert b["smoothed_count"] == pytest.approx(
        (4.0 + MURDER_INFORMATION_PRIOR_EXPOSURE * peer_rate)
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
    assert _row(out, "01:municipal:place:D", "burglary")["estimator"] == DYNAMIC_SHRINK_ESTIMATOR


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
        {DYNAMIC_SHRINK_ESTIMATOR, INFORMATION_POSTERIOR_ESTIMATOR}
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
                "homicide": 10.0,
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
    panel.loc[moved, ["reported_count_preferred", "estimated_count"]] = 80.0
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
