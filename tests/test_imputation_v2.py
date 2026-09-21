"""Imputation v2: the three E5 pseudo-missingness fixes, and the legacy path they must
not touch.

E5 masked 664 units that DID report and scored the production imputation against the
report it was denied. Its verdict: the county-remainder lane is defensible (1.09x over
its applied stratum), the municipal lane over-states 2.18x because a
(state, lane, urbanicity) rate cell is set by its large cities and applied to its small
towns, and the named failure -- California City CA, 20,888 predicted against 292 -- is a
rate cell that had collapsed to 486 residents of remaining observed exposure being
applied to a 14,973-resident unit with nothing to detect it.

Asserted here: the band level's arithmetic and its degradation to the legacy cell rate;
the floor's two terms, its in-lane escalation, and its refusal terminal; the bounds
table's transcription, attachment and the fact that the central correction is attached
and NOT applied; and byte-safety of the legacy path against a verbatim transcription of
the pre-change expressions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.benchmark_imputation import (
    BOUNDS_ALL_OFFENSE_KEY,
    BOUNDS_BASIS_EXTRAPOLATED,
    BOUNDS_BASIS_IN_SUPPORT,
    CELL_EXPOSURE_FLOOR_ABSOLUTE,
    CELL_EXPOSURE_FLOOR_UNIT_MULTIPLE,
    COUNTY_UNIT_KIND,
    EMPIRICAL_BOUNDS_UNIT_COLUMNS,
    IMPUTATION_RULE_VERSION_V1,
    IMPUTATION_RULE_VERSION_V2,
    MUNICIPAL_POPULATION_BAND_EDGES,
    MUNICIPAL_POPULATION_BAND_LABELS,
    MUNICIPAL_UNIT_KIND,
    RATE_CELL_COLUMNS,
    RATE_LEVEL_BAND,
    RATE_LEVEL_CELL,
    RATE_LEVEL_NATIONAL,
    RATE_LEVEL_REFUSED,
    RATE_LEVEL_STATE,
    RATE_LEVEL_STATE_LANE,
    RATE_STRUCTURE_UNIT_COLUMNS,
    UNIT_COLUMNS,
    BenchmarkImputation,
    BenchmarkImputationConfig,
    _fit_rate_levels,
    _frozen_validation_folds,
    _unit_columns,
    assert_benchmark_imputation_invariants,
    attach_empirical_bounds,
    empirical_bounds_path,
    fit_pooled_rates,
    load_empirical_bounds,
    population_band,
    predict_pooled_rate,
    predict_pooled_rate_v2,
    select_band_pooling_constants,
    select_pooling_constants,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
E5_BOUNDS = (
    REPO_ROOT
    / "analysis_scratch"
    / "final_phase"
    / "e5_missingness"
    / "empirical_bounds_production_strata.csv"
)


class _Paths:
    repo_root = REPO_ROOT
    data_dir = REPO_ROOT / "data"
    state_dir = REPO_ROOT / "state"


# --- fixtures ---------------------------------------------------------------


def _municipal_panel() -> pd.DataFrame:
    """One state, one urbanicity, one lane: a big city and four small towns.

    The city's rate is twice the towns' and it carries 99% of the cell's exposure, so
    under the legacy cell rate every town is priced essentially at the city's rate --
    the E5 mechanism in miniature.
    """
    return pd.DataFrame(
        {
            "state_fips": ["01"] * 5,
            "lane": [MUNICIPAL_UNIT_KIND] * 5,
            "urbanicity": ["urban"] * 5,
            "unit_id": ["city", "t1", "t2", "t3", "t4"],
            "exposure_population": [400_000.0, 1_000.0, 1_000.0, 1_000.0, 1_000.0],
            "burglary": [80_000.0, 100.0, 100.0, 100.0, 100.0],
        }
    )


def _two_lane_panel() -> pd.DataFrame:
    """Both lanes in one state, with a deliberately collapsed municipal rural cell.

    Municipal urban 700,000 residents; municipal rural exactly E5's 486; county urban
    500,000; county rural 100,000 across two units. Every level the floor can climb to
    is therefore distinguishable from every other.
    """
    return pd.DataFrame(
        {
            "state_fips": ["01"] * 6,
            "lane": [MUNICIPAL_UNIT_KIND] * 3 + [COUNTY_UNIT_KIND] * 3,
            "urbanicity": ["urban", "urban", "rural", "urban", "rural", "rural"],
            "unit_id": ["m1", "m2", "m3", "c1", "c2", "c3"],
            "exposure_population": [500_000.0, 200_000.0, 486.0, 500_000.0, 60_000.0, 40_000.0],
            "burglary": [5_000.0, 2_000.0, 360.0, 1_500.0, 200.0, 150.0],
        }
    )


def _query(
    *,
    state_fips: str = "01",
    lane: str = MUNICIPAL_UNIT_KIND,
    urbanicity: str = "urban",
    population: float = 1_000.0,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "state_fips": [state_fips],
            "lane": [lane],
            "urbanicity": [urbanicity],
            "exposure_population": [float(population)],
        }
    )
    frame["population_band"] = population_band(frame["exposure_population"])
    return frame


# --- fix 1: size-aware municipal rates --------------------------------------


def test_population_bands_are_predeclared_and_left_closed():
    values = pd.Series([0.0, 2_499.0, 2_500.0, 9_999.0, 10_000.0, 49_999.0, 50_000.0, 1e9])
    bands = population_band(values).tolist()
    assert bands == [
        "<2.5k",
        "<2.5k",
        "2.5k-10k",
        "2.5k-10k",
        "10k-50k",
        "10k-50k",
        "50k+",
        "50k+",
    ]
    assert len(MUNICIPAL_POPULATION_BAND_LABELS) == len(MUNICIPAL_POPULATION_BAND_EDGES) + 1


def test_band_rate_is_the_gamma_poisson_posterior_toward_its_parent_cell():
    panel = _municipal_panel()
    K, K_band = 1e4, 1e3
    rates = _fit_rate_levels(
        panel, offense="burglary", pooling_constant=K, band_pooling_constant=K_band
    )
    # hand computation of the whole ladder
    y_total, e_total = 80_400.0, 404_000.0
    national = y_total / e_total
    state_rate = (y_total + K * national) / (e_total + K)
    cell_rate = (y_total + K * state_rate) / (e_total + K)
    small_band = ("01", MUNICIPAL_UNIT_KIND, "urban", "<2.5k")
    big_band = ("01", MUNICIPAL_UNIT_KIND, "urban", "50k+")
    assert rates["band"][small_band] == pytest.approx((400.0 + K_band * cell_rate) / (4_000.0 + K_band))
    assert rates["band"][big_band] == pytest.approx(
        (80_000.0 + K_band * cell_rate) / (400_000.0 + K_band)
    )


def test_the_band_level_separates_a_small_town_from_the_big_city_that_set_its_cell():
    panel = _municipal_panel()
    legacy = fit_pooled_rates(panel, offense="burglary", pooling_constant=1e4)
    banded = _fit_rate_levels(
        panel, offense="burglary", pooling_constant=1e4, band_pooling_constant=1e3
    )
    town = _query(population=1_000.0)
    legacy_rate = float(predict_pooled_rate(town, legacy)[0])
    band_rate = float(predict_pooled_rate_v2(town, banded)["pooled_rate"].iloc[0])
    # Towns report 0.10/person, the city 0.20. Legacy prices the town at the city's
    # 0.199; the band level moves it back toward its own peers.
    assert legacy_rate == pytest.approx(0.199, abs=1e-3)
    assert band_rate < legacy_rate
    assert band_rate == pytest.approx(banded["band"][("01", MUNICIPAL_UNIT_KIND, "urban", "<2.5k")])


def test_a_band_with_no_observed_data_reproduces_the_legacy_cell_rate_exactly():
    panel = _municipal_panel()
    legacy = fit_pooled_rates(panel, offense="burglary", pooling_constant=1e4)
    banded = _fit_rate_levels(
        panel, offense="burglary", pooling_constant=1e4, band_pooling_constant=1e3
    )
    # no observed municipal unit sits in 10k-50k, so a silent unit of that size falls
    # back to the parent cell -- degradation to v1, not to noise
    unit = _query(population=20_000.0)
    served = predict_pooled_rate_v2(unit, banded)
    assert served["rate_level"].iloc[0] == RATE_LEVEL_CELL
    assert float(served["pooled_rate"].iloc[0]) == float(predict_pooled_rate(unit, legacy)[0])


def test_the_band_level_never_touches_the_county_lane():
    panel = _two_lane_panel()
    legacy = fit_pooled_rates(panel, offense="burglary", pooling_constant=1e4)
    banded = _fit_rate_levels(
        panel, offense="burglary", pooling_constant=1e4, band_pooling_constant=1e2
    )
    assert legacy["cell"] == banded["cell"]
    assert all(key[1] == MUNICIPAL_UNIT_KIND for key in banded["band"])
    county = _query(lane=COUNTY_UNIT_KIND, urbanicity="rural", population=50_000.0)
    served = predict_pooled_rate_v2(county, banded)
    assert served["rate_level"].iloc[0] == RATE_LEVEL_CELL
    assert float(served["pooled_rate"].iloc[0]) == float(predict_pooled_rate(county, legacy)[0])


def test_band_pooling_shrinks_monotonically_toward_the_parent_cell():
    panel = _municipal_panel()
    key = ("01", MUNICIPAL_UNIT_KIND, "urban", "<2.5k")
    cell_key = ("01", MUNICIPAL_UNIT_KIND, "urban")
    rates = [
        _fit_rate_levels(
            panel, offense="burglary", pooling_constant=1e4, band_pooling_constant=k
        )
        for k in (1e1, 1e3, 1e5, 1e9)
    ]
    distances = [abs(r["band"][key] - r["cell"][cell_key]) for r in rates]
    assert distances == sorted(distances, reverse=True)
    assert rates[-1]["band"][key] == pytest.approx(rates[-1]["cell"][cell_key], rel=1e-4)


def test_band_constant_selection_is_a_profile_search_over_the_shared_grid():
    config = BenchmarkImputationConfig(
        year=2024, enable_size_aware_municipal_rates=True, validation_folds=2
    )
    observed = _selection_panel()
    parents, _ = select_pooling_constants(observed, config=config)
    frozen = dict(parents)
    chosen, table = select_band_pooling_constants(
        observed, config=config, pooling_constants=parents
    )
    assert parents == frozen, "the band pass must not mutate the parent selection"
    assert set(chosen) == set(frozen)
    assert set(table["band_pooling_constant"]) == {float(k) for k in config.pooling_constant_grid}
    assert all(value in set(config.pooling_constant_grid) for value in chosen.values())


def _selection_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for state in ("01", "02"):
        for i in range(40):
            population = float(rng.choice([800, 4_000, 20_000, 120_000]))
            for offense in ("murder", "rape", "robbery", "aggravated_assault", "burglary", "larceny", "motor_vehicle_theft"):
                rows.append(
                    {
                        "state_fips": state,
                        "lane": MUNICIPAL_UNIT_KIND if i % 2 else COUNTY_UNIT_KIND,
                        "urbanicity": ("urban", "suburban", "rural")[i % 3],
                        "unit_id": f"{state}:{i}",
                        "offense": offense,
                        "exposure_population": population,
                        offense: float(rng.poisson(population * 0.01)),
                    }
                )
    frame = pd.DataFrame(rows)
    for offense in ("murder", "rape", "robbery", "aggravated_assault", "burglary", "larceny", "motor_vehicle_theft"):
        frame[offense] = pd.to_numeric(frame[offense], errors="coerce").fillna(0.0)
    return frame


# --- fix 2: the cell-exposure floor -----------------------------------------


def test_the_floor_escalates_the_e5_failure_case_off_its_collapsed_cell():
    """486 residents of remaining observed exposure may not price a 14,973-person unit."""
    panel = _two_lane_panel()
    # K at the grid floor, which is where E5's masked build actually landed: the
    # collapsed cell then keeps 83% of the weight on its own 486 residents.
    rates = _fit_rate_levels(panel, offense="burglary", pooling_constant=1e2)
    unit = _query(urbanicity="rural", population=14_973.0)
    unguarded = predict_pooled_rate_v2(unit, rates)
    guarded = predict_pooled_rate_v2(
        unit,
        rates,
        floor_absolute=CELL_EXPOSURE_FLOOR_ABSOLUTE,
        floor_unit_multiple=CELL_EXPOSURE_FLOOR_UNIT_MULTIPLE,
    )
    assert unguarded["rate_level"].iloc[0] == RATE_LEVEL_CELL
    assert guarded["rate_level"].iloc[0] == RATE_LEVEL_STATE_LANE
    assert guarded["rate_escalation_reason"].iloc[0] == "cell_exposure_floor"
    # the collapsed cell prices at ~0.74/person; the in-lane parent is an order down
    assert float(guarded["pooled_rate"].iloc[0]) < float(unguarded["pooled_rate"].iloc[0]) / 5.0
    assert float(guarded["required_cell_exposure"].iloc[0]) == pytest.approx(
        max(CELL_EXPOSURE_FLOOR_ABSOLUTE, 5.0 * 14_973.0)
    )


def test_the_floor_is_the_max_of_its_absolute_and_relative_terms():
    panel = _two_lane_panel()
    rates = _fit_rate_levels(panel, offense="burglary", pooling_constant=1e4)
    # cell "01/municipal/rural" holds 486 residents; a 90-person unit clears the
    # relative term (5 x 90 = 450 < 486) but not the absolute one.
    tiny = predict_pooled_rate_v2(
        _query(urbanicity="rural", population=90.0),
        rates,
        floor_absolute=25_000.0,
        floor_unit_multiple=5.0,
    )
    assert tiny["rate_level"].iloc[0] == RATE_LEVEL_STATE_LANE
    relaxed = predict_pooled_rate_v2(
        _query(urbanicity="rural", population=90.0),
        rates,
        floor_absolute=100.0,
        floor_unit_multiple=5.0,
    )
    assert relaxed["rate_level"].iloc[0] == RATE_LEVEL_CELL
    # and the relative term binds on its own when the absolute one is slack
    relative = predict_pooled_rate_v2(
        _query(urbanicity="rural", population=200.0),
        rates,
        floor_absolute=100.0,
        floor_unit_multiple=5.0,
    )
    assert relative["rate_level"].iloc[0] == RATE_LEVEL_STATE_LANE


def test_escalation_stays_inside_the_unit_s_own_lane():
    """A rural county remainder may never be priced off the state's municipal crime."""
    panel = _two_lane_panel()
    rates = _fit_rate_levels(panel, offense="burglary", pooling_constant=1e4)
    # the county rural cell holds 100,000 residents, the county lane 600,000: a
    # 30,000-person remainder needs 150,000 and can only be served in-lane.
    served = predict_pooled_rate_v2(
        _query(lane=COUNTY_UNIT_KIND, urbanicity="rural", population=30_000.0),
        rates,
        floor_absolute=25_000.0,
        floor_unit_multiple=5.0,
    )
    assert served["rate_level"].iloc[0] == RATE_LEVEL_STATE_LANE
    assert float(served["pooled_rate"].iloc[0]) == pytest.approx(
        rates["state_lane"][("01", COUNTY_UNIT_KIND)]
    )
    assert float(served["pooled_rate"].iloc[0]) != pytest.approx(rates["state"]["01"])


def test_a_lane_below_the_absolute_floor_is_refused_rather_than_priced_elsewhere():
    """The terminal is a real terminal: no cross-lane, no cross-state, no national."""
    panel = pd.DataFrame(
        {
            "state_fips": ["01", "01", "02"],
            "lane": [MUNICIPAL_UNIT_KIND, MUNICIPAL_UNIT_KIND, MUNICIPAL_UNIT_KIND],
            "urbanicity": ["urban", "urban", "rural"],
            "unit_id": ["m1", "m2", "thin"],
            "exposure_population": [500_000.0, 200_000.0, 900.0],
            "burglary": [5_000.0, 2_000.0, 400.0],
        }
    )
    rates = _fit_rate_levels(panel, offense="burglary", pooling_constant=1e4)
    served = predict_pooled_rate_v2(
        _query(state_fips="02", urbanicity="rural", population=5_000.0),
        rates,
        floor_absolute=25_000.0,
        floor_unit_multiple=5.0,
    )
    assert served["rate_level"].iloc[0] == RATE_LEVEL_REFUSED
    assert bool(np.isnan(served["pooled_rate"].iloc[0]))
    assert served["rate_escalation_reason"].iloc[0] == "cell_exposure_floor"
    assert float(served["required_parent_exposure"].iloc[0]) == 25_000.0


def test_a_unit_is_never_refused_merely_for_being_large():
    """The relative term is a level-selection criterion and stops at the cell.

    E5's masked sample contains New York City, Phoenix and the Harris County TX
    remainder, each larger than a fifth of its whole lane. Requiring 5x the unit's
    population at the terminal level refused all of them -- the exact inverse of the
    fix, since a big-city PD going dark is what this imputation exists for.
    """
    panel = _two_lane_panel()
    rates = _fit_rate_levels(panel, offense="burglary", pooling_constant=1e4)
    giant = predict_pooled_rate_v2(
        _query(urbanicity="rural", population=5_000_000.0),
        rates,
        floor_absolute=25_000.0,
        floor_unit_multiple=5.0,
    )
    assert giant["rate_level"].iloc[0] == RATE_LEVEL_STATE_LANE
    assert float(giant["pooled_rate"].iloc[0]) == pytest.approx(
        rates["state_lane"][("01", MUNICIPAL_UNIT_KIND)]
    )
    # ... and it is still kept off the collapsed 486-resident cell it would have taken
    assert giant["rate_escalation_reason"].iloc[0] == "cell_exposure_floor"


def test_an_unknown_cell_with_the_floor_off_keeps_the_legacy_fall_through():
    panel = _two_lane_panel()
    rates = _fit_rate_levels(panel, offense="burglary", pooling_constant=1e4)
    unknown_state = _query(state_fips="99", population=1_000.0)
    served = predict_pooled_rate_v2(unknown_state, rates)
    assert served["rate_level"].iloc[0] == RATE_LEVEL_NATIONAL
    assert float(served["pooled_rate"].iloc[0]) == float(predict_pooled_rate(unknown_state, rates)[0])
    unknown_cell = _query(urbanicity="suburban", population=1_000.0)
    served = predict_pooled_rate_v2(unknown_cell, rates)
    assert served["rate_level"].iloc[0] == RATE_LEVEL_STATE
    assert float(served["pooled_rate"].iloc[0]) == float(predict_pooled_rate(unknown_cell, rates)[0])


def test_a_refused_unit_that_reaches_the_unit_table_fails_closed():
    imputation = _v2_imputation()
    imputation.units.loc[0, "rate_level"] = RATE_LEVEL_REFUSED
    with pytest.raises(ValueError, match="refused by the cell-exposure floor"):
        assert_benchmark_imputation_invariants(imputation)


def test_a_unit_recorded_as_refused_and_also_imputed_fails_closed():
    imputation = _v2_imputation(
        refusals=pd.DataFrame(
            {"unit_id": ["01:municipal:place:0000001"], "offense": ["burglary"], "exposure_population": [1_000.0]}
        )
    )
    with pytest.raises(ValueError, match="also carry imputed"):
        assert_benchmark_imputation_invariants(imputation)


def test_a_published_unit_with_no_rate_level_fails_closed():
    imputation = _v2_imputation()
    imputation.units.loc[0, "rate_level"] = pd.NA
    with pytest.raises(ValueError, match="carry no rate level"):
        assert_benchmark_imputation_invariants(imputation)


# --- fix 3: empirical bounds ------------------------------------------------


def test_the_shipped_v1_bounds_transcribe_the_e5_table_verbatim():
    shipped = pd.read_csv(empirical_bounds_path(_Paths))
    shipped = shipped[shipped["rule_version"].eq(IMPUTATION_RULE_VERSION_V1)]
    measured = pd.read_csv(E5_BOUNDS)
    key = ["stratum", "offense"]
    merged = measured.merge(shipped, on=key, how="outer", suffixes=("_e5", "_cfg"), indicator=True)
    assert set(merged["_merge"]) == {"both"}
    for column in ("n_units", "n", "agg_ratio", "median_ratio", "mult_lo80", "mult_hi80", "mult_lo95", "mult_hi95", "severe_under", "severe_over"):
        assert np.allclose(merged[f"{column}_e5"], merged[f"{column}_cfg"], rtol=0, atol=0), column


def test_the_bounds_table_carries_a_measured_row_set_for_both_rule_versions():
    for version in (IMPUTATION_RULE_VERSION_V1, IMPUTATION_RULE_VERSION_V2):
        bounds = load_empirical_bounds(_Paths, rule_version=version)
        assert set(bounds["lane"]) == {MUNICIPAL_UNIT_KIND, COUNTY_UNIT_KIND}
        assert (bounds["central_correction"] * bounds["agg_ratio"]).round(9).eq(1.0).all()


def _bounds_table() -> pd.DataFrame:
    rows = []
    for lane, stratum, pop_max in (
        (MUNICIPAL_UNIT_KIND, "municipal_<25k", 25_000.0),
        (COUNTY_UNIT_KIND, "county_2.5k-150k", 150_000.0),
    ):
        for offense, agg in (("burglary", 2.0), (BOUNDS_ALL_OFFENSE_KEY, 4.0)):
            rows.append(
                {
                    "rule_version": "test",
                    "stratum": stratum,
                    "lane": lane,
                    "pop_min": 0.0 if lane == MUNICIPAL_UNIT_KIND else 2_500.0,
                    "pop_max": pop_max,
                    "offense": offense,
                    "n_units": 10,
                    "n": 50,
                    "agg_ratio": agg,
                    "median_ratio": agg,
                    "mult_lo80": 0.25,
                    "mult_hi80": 2.0,
                    "mult_lo95": 0.1,
                    "mult_hi95": 5.0,
                    "severe_under": 0.1,
                    "severe_over": 0.4,
                    "source": "unit test",
                    "central_correction": 1.0 / agg,
                }
            )
    return pd.DataFrame(rows)


def _imputed_units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_kind": [MUNICIPAL_UNIT_KIND, MUNICIPAL_UNIT_KIND, COUNTY_UNIT_KIND],
            "unit_id": ["m1", "m2", "c1"],
            "offense": ["burglary", "larceny", "burglary"],
            "exposure_population": [1_000.0, 60_000.0, 40_000.0],
            "imputed_count": [10.0, 20.0, 30.0],
        }
    )


def test_bounds_attach_as_counts_that_bracket_the_point_estimate():
    attached = attach_empirical_bounds(_imputed_units(), bounds=_bounds_table())
    assert attached["bound_lo_80"].tolist() == [2.5, 5.0, 7.5]
    assert attached["bound_hi_80"].tolist() == [20.0, 40.0, 60.0]
    assert attached["bound_lo_95"].tolist() == [1.0, 2.0, 3.0]
    assert attached["bound_hi_95"].tolist() == [50.0, 100.0, 150.0]
    assert attached["imputed_count"].tolist() == [10.0, 20.0, 30.0]


def test_an_offense_without_its_own_row_takes_the_stratum_all_row():
    attached = attach_empirical_bounds(_imputed_units(), bounds=_bounds_table())
    # burglary has its own row (agg 2.0 -> correction 0.5); larceny falls back to ALL
    assert attached["central_correction"].tolist() == [0.5, 0.25, 0.5]


def test_a_unit_outside_the_measured_range_keeps_the_bounds_but_is_labelled():
    attached = attach_empirical_bounds(_imputed_units(), bounds=_bounds_table())
    assert attached["bounds_basis"].tolist() == [
        BOUNDS_BASIS_IN_SUPPORT,
        BOUNDS_BASIS_EXTRAPOLATED,  # 60k municipal unit, stratum measured under 25k
        BOUNDS_BASIS_IN_SUPPORT,
    ]
    assert attached["bounds_stratum"].tolist() == [
        "municipal_<25k",
        "municipal_<25k",
        "county_2.5k-150k",
    ]


def test_the_central_correction_is_attached_and_never_applied():
    units = _imputed_units()
    attached = attach_empirical_bounds(units, bounds=_bounds_table())
    assert attached["imputed_count"].tolist() == units["imputed_count"].tolist()
    assert attached["central_correction"].gt(0.0).all()


def test_the_bounds_loader_refuses_an_unknown_rule_version():
    with pytest.raises(ValueError, match="no rows for rule_version"):
        load_empirical_bounds(_Paths, rule_version="v99")


def test_the_bounds_loader_refuses_an_unordered_ladder(tmp_path, monkeypatch):
    table = _bounds_table().drop(columns=["central_correction"])
    table.loc[0, "mult_lo80"] = 9.0
    _write_and_expect(tmp_path, monkeypatch, table, "not ordered")


def test_the_bounds_loader_refuses_a_stratum_missing_its_all_row(tmp_path, monkeypatch):
    table = _bounds_table().drop(columns=["central_correction"])
    table = table[~(table["lane"].eq(COUNTY_UNIT_KIND) & table["offense"].eq(BOUNDS_ALL_OFFENSE_KEY))]
    _write_and_expect(tmp_path, monkeypatch, table, "no ALL fallback row")


def test_the_bounds_loader_refuses_a_duplicate_row(tmp_path, monkeypatch):
    table = _bounds_table().drop(columns=["central_correction"])
    table = pd.concat([table, table.head(1)], ignore_index=True)
    _write_and_expect(tmp_path, monkeypatch, table, "duplicate")


def test_the_bounds_loader_refuses_a_stratum_spanning_two_lanes(tmp_path, monkeypatch):
    table = _bounds_table().drop(columns=["central_correction"])
    table.loc[table.index[-1], "stratum"] = "municipal_<25k"
    _write_and_expect(tmp_path, monkeypatch, table, "more than one lane")


def _write_and_expect(tmp_path, monkeypatch, table: pd.DataFrame, message: str) -> None:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    table.to_csv(configs / "imputation_empirical_bounds.csv", index=False)

    class _Tmp:
        repo_root = tmp_path

    with pytest.raises(ValueError, match=message):
        load_empirical_bounds(_Tmp, rule_version="test")


def test_attached_bounds_that_are_out_of_order_fail_closed():
    imputation = _v2_imputation(with_bounds=True)
    imputation.units.loc[0, "bound_lo_80"] = 1e9
    with pytest.raises(ValueError, match="not ordered"):
        assert_benchmark_imputation_invariants(imputation)


# --- legacy byte-safety -----------------------------------------------------


def _legacy_fit_pooled_rates(panel: pd.DataFrame, *, offense: str, pooling_constant: float) -> dict:
    """Verbatim transcription of the pre-change fit_pooled_rates (commit 1e2ab8b)."""
    K = float(pooling_constant)
    exposure = pd.to_numeric(panel["exposure_population"], errors="coerce").fillna(0.0)
    counts = pd.to_numeric(panel[offense], errors="coerce").fillna(0.0)
    total_exposure = float(exposure.sum())
    national = float(counts.sum() / total_exposure) if total_exposure > 0 else 0.0

    work = panel[RATE_CELL_COLUMNS].copy()
    work["_y"] = counts.to_numpy(dtype=float)
    work["_E"] = exposure.to_numpy(dtype=float)
    state = work.groupby("state_fips", dropna=False).agg(y=("_y", "sum"), E=("_E", "sum"))
    state["rate"] = (state["y"] + K * national) / (state["E"] + K)
    cell = work.groupby(RATE_CELL_COLUMNS, dropna=False).agg(y=("_y", "sum"), E=("_E", "sum")).reset_index()
    cell = cell.merge(
        state["rate"].rename("state_rate"), left_on="state_fips", right_index=True, how="left"
    )
    cell["state_rate"] = pd.to_numeric(cell["state_rate"], errors="coerce").fillna(national)
    cell["rate"] = (cell["y"] + K * cell["state_rate"]) / (cell["E"] + K)
    return {
        "national": national,
        "state": {str(k): float(v) for k, v in state["rate"].items()},
        "cell": {
            tuple(str(part) for part in key): float(value)
            for key, value in zip(
                cell[RATE_CELL_COLUMNS].itertuples(index=False, name=None), cell["rate"], strict=True
            )
        },
    }


def _legacy_folds(observed: pd.DataFrame, *, seed: int, folds: int) -> np.ndarray:
    return (
        pd.util.hash_pandas_object(
            observed[["state_fips", "unit_id", "offense"]].astype("string"),
            index=False,
            hash_key=f"{int(seed):016d}"[-16:],
        ).to_numpy(dtype=np.uint64)
        % folds
    )


def test_the_legacy_rate_fit_is_byte_identical_to_the_pre_change_expression():
    rng = np.random.default_rng(7)
    panel = pd.DataFrame(
        {
            "state_fips": rng.choice(["01", "02", "03"], 400),
            "lane": rng.choice([MUNICIPAL_UNIT_KIND, COUNTY_UNIT_KIND], 400),
            "urbanicity": rng.choice(["urban", "suburban", "rural"], 400),
            "exposure_population": rng.integers(100, 500_000, 400).astype(float),
            "burglary": rng.integers(0, 5_000, 400).astype(float),
        }
    )
    for K in (1e2, 1e4, 3e5):
        new = fit_pooled_rates(panel, offense="burglary", pooling_constant=K)
        old = _legacy_fit_pooled_rates(panel, offense="burglary", pooling_constant=K)
        assert list(new) == list(old) == ["national", "state", "cell"]
        assert new["national"] == old["national"]
        assert new["state"] == old["state"]
        assert new["cell"] == old["cell"]


def test_the_validation_fold_assignment_is_byte_identical_to_the_pre_change_expression():
    observed = _selection_panel()
    config = BenchmarkImputationConfig(year=2024)
    assert np.array_equal(
        _frozen_validation_folds(observed, config=config),
        _legacy_folds(observed, seed=config.validation_seed, folds=config.validation_folds),
    )


def test_every_v2_flag_defaults_off_and_the_rule_version_follows_them():
    config = BenchmarkImputationConfig()
    assert config.enable_size_aware_municipal_rates is False
    assert config.enable_cell_exposure_floor is False
    assert config.attach_empirical_bounds is False
    assert config.uses_v2_rate_structure is False
    assert config.rate_rule_version == IMPUTATION_RULE_VERSION_V1
    assert config.bounds_rule_version == IMPUTATION_RULE_VERSION_V1
    for field in ("enable_size_aware_municipal_rates", "enable_cell_exposure_floor"):
        assert BenchmarkImputationConfig(**{field: True}).rate_rule_version == IMPUTATION_RULE_VERSION_V2
    # bounds alone do not move the estimate, so they do not re-version the rule
    assert (
        BenchmarkImputationConfig(attach_empirical_bounds=True).rate_rule_version
        == IMPUTATION_RULE_VERSION_V1
    )
    assert (
        BenchmarkImputationConfig(
            enable_cell_exposure_floor=True, empirical_bounds_rule_version="v1"
        ).bounds_rule_version
        == "v1"
    )


def test_the_unit_table_gains_no_column_with_the_flags_off():
    assert _unit_columns(BenchmarkImputationConfig()) == UNIT_COLUMNS
    assert _unit_columns(
        BenchmarkImputationConfig(enable_size_aware_municipal_rates=True)
    ) == UNIT_COLUMNS + RATE_STRUCTURE_UNIT_COLUMNS
    assert _unit_columns(
        BenchmarkImputationConfig(enable_cell_exposure_floor=True)
    ) == UNIT_COLUMNS + RATE_STRUCTURE_UNIT_COLUMNS
    assert _unit_columns(
        BenchmarkImputationConfig(attach_empirical_bounds=True)
    ) == UNIT_COLUMNS + EMPIRICAL_BOUNDS_UNIT_COLUMNS


def test_the_control_build_config_keeps_imputation_v2_off_by_default():
    from crimerisk.controls import ControlBuildConfig, _benchmark_imputation_config

    config = ControlBuildConfig()
    assert config.enable_imputation_v2 is False
    derived = _benchmark_imputation_config(config)
    assert derived.uses_v2_rate_structure is False
    assert derived.attach_empirical_bounds is False
    opted_in = _benchmark_imputation_config(ControlBuildConfig(enable_imputation_v2=True))
    assert opted_in.enable_size_aware_municipal_rates is True
    assert opted_in.enable_cell_exposure_floor is True
    assert opted_in.attach_empirical_bounds is True


# --- shared v2 imputation fixture -------------------------------------------


def _v2_imputation(
    *, refusals: pd.DataFrame | None = None, with_bounds: bool = False
) -> BenchmarkImputation:
    units = pd.DataFrame(
        {
            "year": [2024],
            "state_fips": ["01"],
            "state_abbr": ["AL"],
            "unit_kind": [MUNICIPAL_UNIT_KIND],
            "unit_id": ["01:municipal:place:0000001"],
            "county_geoid": [pd.NA],
            "offense": ["burglary"],
            "exposure_population": [1_000.0],
            "land_area_sq_mi": [2.0],
            "urbanicity": ["urban"],
            "pooled_rate": [0.01],
            "modeled_expected_count": [10.0],
            "modeled_variance": [10.0],
            "benchmark_scale": [1.0],
            "benchmark_weight": [0.001],
            "imputed_count": [10.0],
            "imputation_source": ["benchmarked_nonreporter_imputation"],
            "silent_agency_count": [1],
            "silent_agency_oris": ["AL0000100"],
            "population_band": ["<2.5k"],
            "rate_level": pd.array([RATE_LEVEL_BAND], dtype="string"),
            "rate_escalation_reason": pd.array([""], dtype="string"),
        }
    )
    if with_bounds:
        units = attach_empirical_bounds(units, bounds=_bounds_table())
    identity = pd.DataFrame(
        {
            "state_fips": ["01"],
            "offense": ["burglary"],
            "modeled_pool": [10.0],
            "imputed_total": [10.0],
            "conflict_kind": ["reconciled"],
        }
    )
    return BenchmarkImputation(
        units=units,
        state_identity=identity,
        validation=pd.DataFrame(),
        refusals=refusals if refusals is not None else pd.DataFrame(),
    )
