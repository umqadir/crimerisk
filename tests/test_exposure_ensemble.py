"""The v2 exposure normalizers (contract: EXPOSURE_ENSEMBLE_CONTRACT.md).

Six things are load-bearing and are asserted here: the frozen weight table is E3's output read in
rather than re-derived, the convex arithmetic matches hand-computed cases, the larceny hybrid is
invariant to the units of the opportunity surfaces it mixes in, the ensemble has no kink where the
hard max it replaces has one, every normalizer preserves the reference universe's resident total
while staying strictly positive wherever anybody is, and the legacy denominator path is
byte-identical when the flag is off.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN,
    AllocationBuildConfig,
    _offense_primary_normalizer,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.denominators import DENOMINATOR_SOURCE_COLUMNS, PRIMARY_DENOMINATOR_BY_OFFENSE
from crimerisk.exposure_ensemble import (
    CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE,
    CENSUS_RESIDENTIAL_NORMALIZER_VERSION,
    ENSEMBLE_OFFENSES,
    EXPOSURE_NORMALIZER_COLUMNS,
    EXPOSURE_NORMALIZER_VERSION,
    LANDSCAN_COVERAGE_REPAIR_COLUMN,
    LARCENY_HYBRID_PARTS,
    LARCENY_OPPORTUNITY_NORMALIZER_ID,
    LEGACY_PERSON_NORMALIZER_ID,
    NORMALIZER_SEMANTICS,
    PERSON_ENSEMBLE_NORMALIZER_ID,
    PERSON_LEGS,
    ExposureEnsembleConfig,
    ExposureEnsembleRuntime,
    assert_exposure_normalizer_invariants,
    attach_exposure_normalizers,
    build_exposure_normalizers,
    combine_person_legs,
    compose_larceny_hybrid,
    ensemble_recommendations_path,
    ensemble_weights_path,
    exposure_normalizers_path,
    larceny_hybrid_weights,
    load_ensemble_weights,
    normalizer_id_column,
    normalizer_id_for_offense,
    opportunity_normalizer_column,
    person_leg_weights,
    provisional_offenses,
    residential_leg_source,
    repair_landscan_coverage,
    rescale_to_total,
)
from crimerisk.paths import RepoPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
WEIGHTS = ensemble_weights_path(PATHS)
E3_DIR = REPO_ROOT / "analysis_scratch" / "final_phase" / "e3_exposure"


# --- fixtures ---------------------------------------------------------------


def _weight_rows(
    person: dict[str, tuple[float, float, float]] | None = None,
    hybrid: dict[str, float] | None = None,
) -> pd.DataFrame:
    person = person or {offense: (1.0, 0.0, 0.0) for offense in ENSEMBLE_OFFENSES}
    hybrid = hybrid or {"person_ensemble": 1.0, "destination_poi": 0.0, "retail_jobs": 0.0, "vehicles": 0.0}
    source = {
        "landscan_night": "landscan_night_pop",
        "landscan_day": "landscan_day_pop",
        "daytime_jobs": "daytime_population_jobs_proxy",
        "person_ensemble": "person_ens_v1_larceny",
        "destination_poi": "destination_poi_total",
        "retail_jobs": "lodes_retail_jobs",
        "vehicles": "vehicle_exposure_2024",
    }
    rows = [
        {
            "normalizer_id": PERSON_ENSEMBLE_NORMALIZER_ID,
            "offense": offense,
            "leg": leg,
            "surface_column": source[leg],
            "weight": weight,
            "provisional": "false",
            "note": "fixture",
        }
        for offense, weights in person.items()
        for leg, weight in zip(PERSON_LEGS, weights, strict=True)
    ]
    rows += [
        {
            "normalizer_id": LARCENY_OPPORTUNITY_NORMALIZER_ID,
            "offense": "larceny",
            "leg": part,
            "surface_column": source[part],
            "weight": hybrid[part],
            "provisional": "false",
            "note": "fixture",
        }
        for part in LARCENY_HYBRID_PARTS
    ]
    return pd.DataFrame(rows)


def _write_weights(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "weights.csv"
    frame.to_csv(path, index=False)
    return path


def _normalizer_table(
    *,
    bg_ids: list[str],
    population: list[float],
    night: list[float],
    day: list[float],
    jobs: list[float],
    repaired: list[bool] | None = None,
    normalizers: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "state_fips": pd.array([bg[:2] for bg in bg_ids], dtype="string"),
            "bg_id": pd.array(bg_ids, dtype="string"),
            "population": population,
            "landscan_night_leg": night,
            "landscan_day_leg": day,
            "daytime_jobs_leg": jobs,
            LANDSCAN_COVERAGE_REPAIR_COLUMN: repaired if repaired is not None else [False] * len(bg_ids),
            "destination_poi_total": [0.0] * len(bg_ids),
            "lodes_retail_jobs": [0.0] * len(bg_ids),
        }
    )
    for offense in ENSEMBLE_OFFENSES:
        column = opportunity_normalizer_column(offense)
        frame[column] = (normalizers or {}).get(offense, list(np.asarray(night, dtype=float)))
    return frame.reindex(columns=EXPOSURE_NORMALIZER_COLUMNS)


# --- the frozen weight table -------------------------------------------------


def test_shipped_person_weights_are_the_e3_recommendation_verbatim():
    """configs/exposure_ensemble_weights_v1.csv must still be E3's leave-one-source-out answer."""
    shipped = person_leg_weights(load_ensemble_weights(WEIGHTS))
    for source_path in (E3_DIR / "recommendations.csv", ensemble_recommendations_path(PATHS)):
        if not source_path.exists():
            pytest.skip(f"{source_path} absent")
        source = pd.read_csv(source_path).set_index("offense")
        for offense, (night, day, jobs) in shipped.items():
            row = source.loc[offense]
            assert (night, day, jobs) == (
                float(row["w_landscan_night"]),
                float(row["w_landscan_day"]),
                float(row["w_daytime_jobs"]),
            )


def test_shipped_larceny_hybrid_is_the_e3_pooled_best_combo():
    metadata_path = E3_DIR / "run_metadata.json"
    if not metadata_path.exists():
        pytest.skip("E3 run metadata absent")
    verdict = json.loads(metadata_path.read_text())["larceny_verdict"]
    expected = [float(value) for value in verdict["pooled_best_combo"][1:].split("_")]
    shipped = larceny_hybrid_weights(load_ensemble_weights(WEIGHTS))
    # E3 calls the first part `person`; this lane names it `person_ensemble` because it IS
    # person_ens_v1 evaluated at larceny's own leg weights. Same quantity, explicit name.
    assert verdict["parts"] == ["person", "destination_poi", "retail_jobs", "vehicles"]
    renamed = ["person_ensemble", *verdict["parts"][1:]]
    assert [shipped[part] for part in renamed] == expected
    # E3's `person` part is this lane's `person_ensemble`, and its person mix is larceny's own.
    assert person_leg_weights(load_ensemble_weights(WEIGHTS))["larceny"] == tuple(
        verdict["person_mix_used"]
    )


def test_shipped_table_is_a_convex_simplex_over_exactly_the_expected_cells():
    weights = load_ensemble_weights(WEIGHTS)
    person = weights[weights["normalizer_id"].eq(PERSON_ENSEMBLE_NORMALIZER_ID)]
    hybrid = weights[weights["normalizer_id"].eq(LARCENY_OPPORTUNITY_NORMALIZER_ID)]
    assert set(person["offense"]) == set(ENSEMBLE_OFFENSES)
    assert set(hybrid["leg"]) == set(LARCENY_HYBRID_PARTS)
    assert weights["weight"].min() >= 0.0
    sums = weights.groupby(["normalizer_id", "offense"])["weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)
    # Burglary and motor vehicle theft are absent by design: E3 keeps the premises denominator and
    # defers the vehicle one to its own experiment.
    assert "burglary" not in set(weights["offense"])
    assert "motor_vehicle_theft" not in set(weights["offense"])


def test_murder_and_rape_ship_flagged_provisional_with_a_reason():
    weights = load_ensemble_weights(WEIGHTS)
    assert provisional_offenses(weights) == ("murder", "rape")
    provisional = weights[weights["provisional"]]
    assert set(provisional["offense"]) == {"murder", "rape"}
    assert provisional["note"].str.contains("PROVISIONAL").all()
    assert not weights[~weights["provisional"]]["note"].str.contains("PROVISIONAL").any()


def test_weight_table_rejects_a_missing_cell(tmp_path):
    frame = _weight_rows()
    path = _write_weights(tmp_path, frame.iloc[1:])
    with pytest.raises(ValueError, match="must cover exactly"):
        load_ensemble_weights(path)


def test_weight_table_rejects_a_vector_that_is_not_convex(tmp_path):
    frame = _weight_rows(person={**{o: (1.0, 0.0, 0.0) for o in ENSEMBLE_OFFENSES}, "murder": (0.5, 0.2, 0.2)})
    path = _write_weights(tmp_path, frame)
    with pytest.raises(ValueError, match="does not sum to 1"):
        load_ensemble_weights(path)


def test_weight_table_rejects_a_negative_weight(tmp_path):
    frame = _weight_rows(person={**{o: (1.0, 0.0, 0.0) for o in ENSEMBLE_OFFENSES}, "robbery": (1.2, -0.2, 0.0)})
    path = _write_weights(tmp_path, frame)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        load_ensemble_weights(path)


def test_weight_table_rejects_duplicate_rows(tmp_path):
    frame = _weight_rows()
    path = _write_weights(tmp_path, pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="duplicate"):
        load_ensemble_weights(path)


def test_weight_table_rejects_a_leg_pointed_at_the_wrong_surface(tmp_path):
    frame = _weight_rows()
    frame.loc[frame["leg"].eq("landscan_night"), "surface_column"] = "population_2024"
    path = _write_weights(tmp_path, frame)
    with pytest.raises(ValueError, match="this lane reads"):
        load_ensemble_weights(path)


def test_weight_table_rejects_a_non_boolean_provisional_flag(tmp_path):
    frame = _weight_rows()
    frame.loc[0, "provisional"] = "maybe"
    path = _write_weights(tmp_path, frame)
    with pytest.raises(ValueError, match="non-boolean provisional"):
        load_ensemble_weights(path)


def test_missing_weight_table_is_a_hard_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="source of record"):
        load_ensemble_weights(tmp_path / "absent.csv")


# --- weight application arithmetic -------------------------------------------


def test_person_ensemble_matches_a_hand_computed_three_way_mix():
    legs = {
        "landscan_night": np.array([100.0, 0.0, 40.0]),
        "landscan_day": np.array([20.0, 300.0, 40.0]),
        "daytime_jobs": np.array([50.0, 500.0, 40.0]),
    }
    result = combine_person_legs(legs, (0.7, 0.1, 0.2))
    expected = np.array(
        [
            0.7 * 100.0 + 0.1 * 20.0 + 0.2 * 50.0,
            0.7 * 0.0 + 0.1 * 300.0 + 0.2 * 500.0,
            40.0,
        ]
    )
    assert np.allclose(result, expected, rtol=0.0, atol=1e-12)
    # A mix of three copies of one surface is that surface: the simplex is closed on the diagonal.
    assert result[2] == pytest.approx(40.0)


def test_person_ensemble_ignores_a_leg_at_weight_zero():
    """Murder ships at w_landscan_day = 0; the day surface must not touch its normalizer."""
    legs = {
        "landscan_night": np.array([100.0, 200.0]),
        "landscan_day": np.array([1.0, 2.0]),
        "daytime_jobs": np.array([300.0, 400.0]),
    }
    baseline = combine_person_legs(legs, (0.9, 0.0, 0.1))
    moved = combine_person_legs({**legs, "landscan_day": np.array([1e9, 1e9])}, (0.9, 0.0, 0.1))
    assert np.array_equal(baseline, moved)


def test_person_ensemble_rejects_weights_that_are_not_convex():
    legs = {leg: np.array([1.0]) for leg in PERSON_LEGS}
    with pytest.raises(ValueError, match="convex"):
        combine_person_legs(legs, (0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="person-leg weights"):
        combine_person_legs(legs, (0.5, 0.5))


def test_reference_rescale_preserves_every_ratio_and_hits_the_target_total():
    values = np.array([10.0, 30.0, 0.0, 60.0])
    scaled, scale = rescale_to_total(values, target_total=200.0)
    assert float(scaled.sum()) == pytest.approx(200.0)
    assert scale == pytest.approx(2.0)
    positive = values > 0
    assert np.allclose(scaled[positive] / values[positive], scale)


def test_reference_rescale_refuses_a_surface_with_no_mass():
    with pytest.raises(ValueError, match="not positive"):
        rescale_to_total(np.zeros(4), target_total=100.0)


# --- the larceny opportunity hybrid ------------------------------------------


def test_larceny_hybrid_matches_a_hand_computed_composition():
    person = np.array([100.0, 300.0, 600.0])          # total 1000
    poi = np.array([1.0, 4.0, 5.0])                    # total 10   -> scale 100
    retail = np.array([10.0, 10.0, 30.0])              # total 50   -> scale 20
    hybrid, scales = compose_larceny_hybrid(
        person=person,
        parts={"destination_poi": poi, "retail_jobs": retail, "vehicles": np.zeros(3)},
        weights={"person_ensemble": 0.5, "destination_poi": 0.3, "retail_jobs": 0.2, "vehicles": 0.0},
    )
    expected = 0.5 * person + 0.3 * (poi * 100.0) + 0.2 * (retail * 20.0)
    assert np.allclose(hybrid, expected, rtol=0.0, atol=1e-9)
    assert scales["destination_poi"] == pytest.approx(100.0)
    assert scales["retail_jobs"] == pytest.approx(20.0)
    # Conservation: every part carries the person ensemble's total, so the hybrid does too.
    assert float(hybrid.sum()) == pytest.approx(float(person.sum()))


def test_larceny_hybrid_is_invariant_to_the_units_of_its_opportunity_surfaces():
    """A POI count is not a person: the pre-mix rescale is what makes the blend meaningful."""
    person = np.array([100.0, 300.0, 600.0])
    parts = {
        "destination_poi": np.array([1.0, 4.0, 5.0]),
        "retail_jobs": np.array([10.0, 10.0, 30.0]),
        "vehicles": np.zeros(3),
    }
    weights = {"person_ensemble": 0.5, "destination_poi": 0.3, "retail_jobs": 0.2, "vehicles": 0.0}
    base, _ = compose_larceny_hybrid(person=person, parts=parts, weights=weights)
    rescaled_inputs = {name: values * 1000.0 for name, values in parts.items()}
    moved, _ = compose_larceny_hybrid(person=person, parts=rescaled_inputs, weights=weights)
    assert np.allclose(base, moved, rtol=1e-12, atol=1e-9)


def test_larceny_hybrid_does_not_read_the_rejected_vehicle_leg():
    person = np.array([100.0, 300.0, 600.0])
    weights = {"person_ensemble": 0.5, "destination_poi": 0.3, "retail_jobs": 0.2, "vehicles": 0.0}
    parts = {
        "destination_poi": np.array([1.0, 4.0, 5.0]),
        "retail_jobs": np.array([10.0, 10.0, 30.0]),
        "vehicles": np.zeros(3),
    }
    base, scales = compose_larceny_hybrid(person=person, parts=parts, weights=weights)
    moved, _ = compose_larceny_hybrid(
        person=person, parts={**parts, "vehicles": np.array([1e6, 0.0, 0.0])}, weights=weights
    )
    assert np.array_equal(base, moved)
    assert scales["vehicles"] == 0.0


def test_larceny_hybrid_refuses_a_person_ensemble_with_no_mass():
    with pytest.raises(ValueError, match="no mass"):
        compose_larceny_hybrid(
            person=np.zeros(3),
            parts={"destination_poi": np.ones(3), "retail_jobs": np.ones(3), "vehicles": np.zeros(3)},
            weights={"person_ensemble": 0.5, "destination_poi": 0.3, "retail_jobs": 0.2, "vehicles": 0.0},
        )


# --- LandScan coverage repair ------------------------------------------------


def test_coverage_repair_fires_only_where_landscan_carries_no_mass_at_all():
    night = np.array([0.0, 0.0, 5.0, 0.0])
    day = np.array([0.0, 7.0, 0.0, 0.0])
    fallback = np.array([120.0, 90.0, 80.0, 0.0])
    repaired_night, repaired_day, repaired = repair_landscan_coverage(
        night=night, day=day, fallback=fallback
    )
    assert repaired.tolist() == [True, False, False, True]
    assert repaired_night.tolist() == [120.0, 0.0, 5.0, 0.0]
    assert repaired_day.tolist() == [120.0, 7.0, 0.0, 0.0]


def test_coverage_repair_set_is_invariant_to_any_rescaling_of_the_legs():
    """The anti-kink property: no perturbation of the inputs moves a cell in or out of the set."""
    night = np.array([0.0, 3.0, 0.0, 11.0])
    day = np.array([0.0, 0.0, 2.0, 9.0])
    fallback = np.array([50.0, 60.0, 70.0, 80.0])
    _, _, baseline = repair_landscan_coverage(night=night, day=day, fallback=fallback)
    for night_factor in (0.8, 1.0, 1.2, 100.0):
        for day_factor in (0.8, 1.0, 1.2, 100.0):
            _, _, moved = repair_landscan_coverage(
                night=night * night_factor, day=day * day_factor, fallback=fallback
            )
            assert moved.tolist() == baseline.tolist()


def test_repair_keeps_the_robbery_mix_positive_where_landscan_has_a_hole():
    """Robbery ships at zero weight on the jobs leg, so an unrepaired hole would zero its rate."""
    night, day, _ = repair_landscan_coverage(
        night=np.array([0.0]), day=np.array([0.0]), fallback=np.array([420.0])
    )
    legs = {"landscan_night": night, "landscan_day": day, "daytime_jobs": np.array([420.0])}
    assert combine_person_legs(legs, (0.7, 0.3, 0.0))[0] == pytest.approx(420.0)


# --- continuity: the ensemble has no kink where the hard max has one ---------


def _hard_max(day: np.ndarray, jobs: np.ndarray) -> np.ndarray:
    return np.maximum(day, jobs)


def test_ensemble_responds_linearly_to_a_one_percent_perturbation():
    rng = np.random.default_rng(20260805)
    night = rng.uniform(1.0, 5_000.0, 500)
    day = rng.uniform(1.0, 5_000.0, 500)
    jobs = rng.uniform(1.0, 5_000.0, 500)
    legs = {"landscan_night": night, "landscan_day": day, "daytime_jobs": jobs}
    weights = (0.7, 0.1, 0.2)
    base = combine_person_legs(legs, weights)
    for index, leg in enumerate(PERSON_LEGS):
        up = combine_person_legs({**legs, leg: legs[leg] * 1.01}, weights)
        down = combine_person_legs({**legs, leg: legs[leg] * 0.99}, weights)
        # Exactly linear: the second difference is zero, so there is no kink anywhere.
        assert np.allclose(up + down, 2.0 * base, rtol=0.0, atol=1e-9)
        # And the response is the analytic one: w_k x leg_k x 1%.
        assert np.allclose(up - base, 0.01 * weights[index] * legs[leg], rtol=1e-12, atol=1e-9)
        # A 1% miss on one input can never move the normalizer by more than 1%.
        assert np.max(np.abs(up - base) / base) <= 0.01 + 1e-12


def test_hard_max_kinks_at_the_crossing_where_the_ensemble_does_not():
    # A sweep of `day` straight through `jobs`: the hard max's derivative jumps from 0 to 1 at the
    # crossing, which is the selection instability E3 measured (8-11% of block groups switch leg
    # under a 20% recalibration). The convex ensemble's derivative is its weight, everywhere.
    jobs = np.full(9, 100.0)
    day = np.linspace(80.0, 120.0, 9)
    night = np.full(9, 100.0)
    legs = {"landscan_night": night, "landscan_day": day, "daytime_jobs": jobs}
    weights = (0.7, 0.1, 0.2)

    hard_base = _hard_max(day, jobs)
    hard_up = _hard_max(day * 1.01, jobs)
    hard_sensitivity = (hard_up - hard_base) / (0.01 * day)
    assert hard_sensitivity.min() == pytest.approx(0.0)      # below the crossing: no response
    assert hard_sensitivity.max() == pytest.approx(1.0)      # above it: full response
    hard_down = _hard_max(day * 0.99, jobs)
    assert np.max(np.abs(hard_up + hard_down - 2.0 * hard_base)) > 0.0   # a kink

    base = combine_person_legs(legs, weights)
    up = combine_person_legs({**legs, "landscan_day": day * 1.01}, weights)
    down = combine_person_legs({**legs, "landscan_day": day * 0.99}, weights)
    sensitivity = (up - base) / (0.01 * day)
    assert np.allclose(sensitivity, weights[1])
    assert np.allclose(up + down, 2.0 * base, rtol=0.0, atol=1e-12)      # no kink


def test_hard_max_selection_flips_under_a_one_percent_perturbation_and_the_ensemble_has_no_selection():
    jobs = np.array([99.5, 100.0, 100.5])
    day = np.full(3, 100.0)
    baseline_selection = day > jobs
    moved_selection = day > (jobs * 1.01)
    assert (baseline_selection != moved_selection).any()
    # The ensemble has no argmax to flip: every leg enters at its frozen weight on every row.
    legs = {"landscan_night": np.full(3, 50.0), "landscan_day": day, "daytime_jobs": jobs}
    base = combine_person_legs(legs, (0.7, 0.1, 0.2))
    moved = combine_person_legs({**legs, "daytime_jobs": jobs * 1.01}, (0.7, 0.1, 0.2))
    assert np.allclose(moved - base, 0.2 * 0.01 * jobs)


# --- invariants ---------------------------------------------------------------


def test_invariants_accept_a_well_formed_table():
    frame = _normalizer_table(
        bg_ids=["010010001001", "010010001002"],
        population=[100.0, 200.0],
        night=[100.0, 200.0],
        day=[100.0, 200.0],
        jobs=[100.0, 200.0],
    )
    assert_exposure_normalizer_invariants(frame, reference_total=300.0)


def test_invariants_reject_a_normalizer_that_is_zero_where_people_live():
    frame = _normalizer_table(
        bg_ids=["010010001001", "010010001002"],
        population=[100.0, 200.0],
        night=[0.0, 200.0],
        day=[0.0, 200.0],
        jobs=[100.0, 200.0],
    )
    with pytest.raises(ValueError, match="resident population"):
        assert_exposure_normalizer_invariants(frame)


def test_invariants_reject_a_negative_or_nonfinite_normalizer():
    frame = _normalizer_table(
        bg_ids=["010010001001"], population=[10.0], night=[10.0], day=[10.0], jobs=[10.0]
    )
    frame.loc[0, opportunity_normalizer_column("murder")] = -1.0
    with pytest.raises(ValueError, match="negative"):
        assert_exposure_normalizer_invariants(frame)
    frame.loc[0, opportunity_normalizer_column("murder")] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        assert_exposure_normalizer_invariants(frame)


def test_invariants_reject_a_total_that_is_not_the_reference_universe_total():
    frame = _normalizer_table(
        bg_ids=["010010001001"], population=[10.0], night=[10.0], day=[10.0], jobs=[10.0]
    )
    with pytest.raises(ValueError, match="over the reference universe"):
        assert_exposure_normalizer_invariants(frame, reference_total=999.0)


def test_invariants_reject_duplicate_block_groups():
    frame = _normalizer_table(
        bg_ids=["010010001001", "010010001001"],
        population=[10.0, 10.0],
        night=[10.0, 10.0],
        day=[10.0, 10.0],
        jobs=[10.0, 10.0],
    )
    with pytest.raises(ValueError, match="duplicate"):
        assert_exposure_normalizer_invariants(frame)


def test_invariants_reject_a_repair_flag_whose_legs_are_not_the_person_proxy():
    frame = _normalizer_table(
        bg_ids=["010010001001"],
        population=[10.0],
        night=[3.0],
        day=[4.0],
        jobs=[10.0],
        repaired=[True],
    )
    with pytest.raises(ValueError, match="coverage repair"):
        assert_exposure_normalizer_invariants(frame)


def test_invariants_reject_a_normalizer_that_does_not_recompose_at_the_frozen_weights():
    weights = load_ensemble_weights(WEIGHTS)
    night, day, jobs = [10.0, 40.0], [20.0, 10.0], [30.0, 20.0]
    murder = [
        0.9 * night[row] + 0.0 * day[row] + 0.1 * jobs[row] for row in range(2)
    ]
    frame = _normalizer_table(
        bg_ids=["010010001001", "010010001002"],
        population=[10.0, 10.0],
        night=night,
        day=day,
        jobs=jobs,
        normalizers={
            offense: (murder if offense == "murder" else list(np.asarray(night, dtype=float)))
            for offense in ENSEMBLE_OFFENSES
        },
    )
    # murder recomposes; rape (0.7/0.1/0.2) does not, because it was filled with the night leg.
    with pytest.raises(ValueError, match="does not recompose"):
        assert_exposure_normalizer_invariants(frame, weights=weights)


def test_normalizer_table_schema_is_the_contract_schema():
    frame = _normalizer_table(
        bg_ids=["010010001001"], population=[1.0], night=[1.0], day=[1.0], jobs=[1.0]
    )
    assert list(frame.columns) == EXPOSURE_NORMALIZER_COLUMNS
    assert [opportunity_normalizer_column(offense) for offense in ENSEMBLE_OFFENSES] == [
        f"opportunity_normalizer_{offense}" for offense in ENSEMBLE_OFFENSES
    ]


def test_census_residential_arm_is_nationwide_and_retains_landscan(monkeypatch):
    frame = pd.DataFrame(
        {
            "state_fips": pd.array(["01", "01"], dtype="string"),
            "bg_id": pd.array(["010010001001", "010010001002"], dtype="string"),
            "population": [100.0, 300.0],
            "landscan_night_pop": [10.0, 0.0],
            "landscan_day_pop": [20.0, 0.0],
            "daytime_population_jobs_proxy": [40.0, 500.0],
            "lodes_retail_jobs": [1.0, 2.0],
            "destination_poi_total": [2.0, 1.0],
            "vehicle_exposure_2024": [0.0, 0.0],
            "exposure_proxy_2024": [40.0, 500.0],
        }
    )
    monkeypatch.setattr("crimerisk.exposure_ensemble._reference_frame", lambda **_: frame)
    normalizers, summary = build_exposure_normalizers(
        paths=PATHS,
        config=ExposureEnsembleConfig(
            year=2025,
            weights_path=WEIGHTS,
            residential_leg_source=CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE,
        ),
    )
    assert residential_leg_source(normalizers) == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
    assert normalizers["residential_leg"].tolist() == [100.0, 300.0]
    assert normalizers["raw_landscan_night_pop"].tolist() == [10.0, 0.0]
    assert normalizers["landscan_night_leg"].tolist() == [10.0, 500.0]
    assert summary["normalizer_version"] == CENSUS_RESIDENTIAL_NORMALIZER_VERSION
    assert summary["residential_leg"]["source"] == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
    assert summary["residential_leg"]["local_exceptions"] == []


def test_census_residential_arm_has_a_distinct_default_path():
    assert exposure_normalizers_path(PATHS, year=2025).name == (
        "bg_exposure_normalizers_2025.parquet"
    )
    assert exposure_normalizers_path(
        PATHS,
        year=2025,
        residential_leg_source=CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE,
    ).name == "bg_exposure_normalizers_2025_census_residential.parquet"


# --- naming and versioning ----------------------------------------------------


def test_field_names_and_ids_say_opportunity_normalized_intensity():
    assert NORMALIZER_SEMANTICS == "opportunity_normalized_intensity_not_person_time_risk"
    assert EXPOSURE_NORMALIZER_VERSION == "exposure_ensemble_v1"
    assert normalizer_id_for_offense("murder") == PERSON_ENSEMBLE_NORMALIZER_ID
    assert normalizer_id_for_offense("larceny") == LARCENY_OPPORTUNITY_NORMALIZER_ID
    assert normalizer_id_for_offense("burglary") == "premises_nnls_v1"
    assert normalizer_id_for_offense("motor_vehicle_theft") == "vehicle_exposure_v1"
    # Off the lane, the person offenses are honest about what they are: a hard max, v1.
    assert normalizer_id_for_offense("robbery", enabled=False) == LEGACY_PERSON_NORMALIZER_ID
    assert normalizer_id_for_offense("burglary", enabled=False) == "premises_nnls_v1"
    assert normalizer_id_column("robbery") == "primary_denominator_normalizer_id_robbery"
    with pytest.raises(KeyError):
        normalizer_id_for_offense("arson")


def test_ensemble_offenses_are_the_five_person_exposure_offenses():
    assert ENSEMBLE_OFFENSES == ("murder", "rape", "robbery", "aggravated_assault", "larceny")
    assert set(OFFENSES_7) - set(ENSEMBLE_OFFENSES) == {"burglary", "motor_vehicle_theft"}


# --- the allocation hook ------------------------------------------------------


def _denominator_frame(*, with_normalizers: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "bg_id": pd.array(["010010001001", "010010001002", "010010001003"], dtype="string"),
            "state_fips": pd.array(["01", "01", "01"], dtype="string"),
            "exposure_proxy_2024": [1_000.0, 0.0, np.nan],
            "households_total": [400.0, 0.0, 20.0],
            "burglary_premises_total": [700.0, 1.0, np.nan],
            "vehicle_exposure_2024": [800.0, -5.0, 3.0],
            "population_2024": [900.0, 0.0, 10.0],
        }
    )
    if with_normalizers:
        for index, offense in enumerate(ENSEMBLE_OFFENSES):
            frame[opportunity_normalizer_column(offense)] = [
                100.0 + index,
                200.0 + index,
                300.0 + index,
            ]
    return frame


def _legacy_primary_denominator(frame: pd.DataFrame, offense: str) -> pd.Series:
    """Verbatim transcription of the pre-change expression in `_finalize_output`."""
    denominator_type = PRIMARY_DENOMINATOR_BY_OFFENSE[offense]
    return (
        pd.to_numeric(frame[DENOMINATOR_SOURCE_COLUMNS[denominator_type]], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )


def test_default_allocation_config_leaves_the_exposure_ensemble_off():
    config = AllocationBuildConfig()
    assert config.enable_exposure_ensemble is False
    assert config.exposure_normalizers_path is None
    assert config.exposure_ensemble_weights_path is None


def test_legacy_denominator_is_byte_identical_when_the_ensemble_is_off():
    frame = _denominator_frame(with_normalizers=True)
    for offense in OFFENSES_7:
        value, normalizer_id = _offense_primary_normalizer(
            frame, offense=offense, enable_exposure_ensemble=False
        )
        pd.testing.assert_series_equal(
            value, _legacy_primary_denominator(frame, offense), check_exact=True, check_names=False
        )
        assert normalizer_id is None


def test_enabled_lane_publishes_the_named_normalizer_for_person_offenses_only():
    frame = _denominator_frame(with_normalizers=True)
    for offense in ENSEMBLE_OFFENSES:
        value, normalizer_id = _offense_primary_normalizer(
            frame, offense=offense, enable_exposure_ensemble=True
        )
        assert normalizer_id == normalizer_id_for_offense(offense)
        assert value.tolist() == frame[opportunity_normalizer_column(offense)].tolist()
    for offense in ("burglary", "motor_vehicle_theft"):
        value, normalizer_id = _offense_primary_normalizer(
            frame, offense=offense, enable_exposure_ensemble=True
        )
        assert normalizer_id == normalizer_id_for_offense(offense)
        pd.testing.assert_series_equal(
            value, _legacy_primary_denominator(frame, offense), check_exact=True, check_names=False
        )


def test_enabled_lane_fails_closed_when_the_normalizer_surface_is_missing():
    frame = _denominator_frame(with_normalizers=False)
    with pytest.raises(ValueError, match="build-exposure-normalizers"):
        _offense_primary_normalizer(frame, offense="robbery", enable_exposure_ensemble=True)


def test_enabled_lane_fails_closed_on_a_null_normalizer_row():
    frame = _denominator_frame(with_normalizers=True)
    frame.loc[1, opportunity_normalizer_column("robbery")] = np.nan
    with pytest.raises(ValueError, match="null rows"):
        _offense_primary_normalizer(frame, offense="robbery", enable_exposure_ensemble=True)


def _runtime(normalizers: pd.DataFrame) -> ExposureEnsembleRuntime:
    return ExposureEnsembleRuntime(
        normalizers=normalizers,
        weights=load_ensemble_weights(WEIGHTS),
        config=ExposureEnsembleConfig(year=2024),
    )


def _surface_row(
    bg: str, *, pop: float, exposure: float, normalizer: float, counts: float, with_normalizers: bool
) -> dict:
    row = {
        "block_group_geoid": bg,
        "tract_id": bg[:11],
        "state_fips": bg[:2],
        "population_2024": pop,
        "households_total": 120.0,
        "commercial_premises_total": 0.0,
        "destination_poi_total": 0.0,
        "daytime_population_jobs_proxy": pop,
        "landscan_day_pop": 0.0,
        "exposure_proxy_2024": exposure,
        "burglary_premises_total": 120.0,
        "aggregate_vehicles_total": pop * 0.8,
        "vehicle_exposure_2024": pop * 0.8,
        "land_area_sq_mi": 1.0,
        "eb_jurisdiction_id": "J1",
        "eb_jurisdiction_type": "municipal",
    }
    for offense in OFFENSES_7:
        row[f"expected_count_{offense}"] = counts
        row[f"footprint_derived_count_{offense}"] = 0.0
    # On a legacy build the covariate frame never carries these: the attach is driven by the same
    # flag, so the legacy surface must be reproduced without them.
    if with_normalizers:
        for offense in ENSEMBLE_OFFENSES:
            row[opportunity_normalizer_column(offense)] = normalizer
    return row


def _finalized(enabled: bool) -> pd.DataFrame:
    from crimerisk.allocation import _finalize_output

    rows = [
        _surface_row(
            f"01001000{index:04d}",
            pop=2_000.0,
            exposure=3_000.0,
            normalizer=2_400.0,
            counts=20.0,
            with_normalizers=enabled,
        )
        for index in range(30)
    ]
    return _finalize_output(
        pd.DataFrame(rows),
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(enable_exposure_ensemble=enabled),
    ).set_index("block_group_geoid")


def test_finalized_surface_divides_by_the_named_normalizer_and_says_which_one():
    surface = _finalized(enabled=True)
    cell = surface.loc["010010000000"]
    for offense in ENSEMBLE_OFFENSES:
        assert cell[f"primary_denominator_{offense}"] == pytest.approx(2_400.0)
        assert cell[f"rate_{offense}_primary"] == pytest.approx(1e5 * 20.0 / 2_400.0)
        assert cell[normalizer_id_column(offense)] == normalizer_id_for_offense(offense)
    # Burglary and MVT keep their own denominators, and say so.
    assert cell["primary_denominator_burglary"] == pytest.approx(120.0)
    assert cell["primary_denominator_motor_vehicle_theft"] == pytest.approx(1_600.0)
    assert cell[normalizer_id_column("burglary")] == "premises_nnls_v1"
    assert cell["opportunity_normalizer_semantics"] == NORMALIZER_SEMANTICS
    # The resident arm is untouched: it exists to be the ambient-blind comparator.
    assert cell["rate_larceny_resident"] == pytest.approx(1e5 * 20.0 / 2_000.0)


def test_finalized_legacy_surface_gains_no_columns_and_keeps_the_unchanged_offenses_identical():
    legacy = _finalized(enabled=False)
    enabled = _finalized(enabled=True)
    new_columns = set(enabled.columns) - set(legacy.columns)
    assert new_columns == {
        "opportunity_normalizer_semantics",
        # Not this lane's column, but this lane turns it on: the near-zero-resident opportunity
        # floor is the publication rule the rescaled denominator has to carry (see the contract),
        # and it is absent from a legacy build for the same reason everything else here is.
        ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN,
        *[normalizer_id_column(offense) for offense in OFFENSES_7],
        *[opportunity_normalizer_column(offense) for offense in ENSEMBLE_OFFENSES],
    }
    assert legacy["primary_denominator_larceny"].eq(3_000.0).all()   # the deployed hard max
    for offense in ("burglary", "motor_vehicle_theft"):
        for column in (f"primary_denominator_{offense}", f"rate_{offense}_primary", f"index_{offense}_primary"):
            pd.testing.assert_series_equal(legacy[column], enabled[column], check_exact=True)


def test_attach_fails_closed_on_a_published_universe_gap_but_tolerates_excluded_states():
    normalizers = _normalizer_table(
        bg_ids=["010010001001"], population=[10.0], night=[10.0], day=[10.0], jobs=[10.0]
    )
    covariates = pd.DataFrame(
        {
            "bg_id": pd.array(["010010001001", "020010001001"], dtype="string"),
            "state_fips": pd.array(["01", "02"], dtype="string"),
        }
    )
    attached = attach_exposure_normalizers(covariates, runtime=_runtime(normalizers))
    assert attached.loc[0, opportunity_normalizer_column("murder")] == 10.0
    assert pd.isna(attached.loc[1, opportunity_normalizer_column("murder")])

    covariates.loc[1, "state_fips"] = "01"
    with pytest.raises(ValueError, match="published universe"):
        attach_exposure_normalizers(covariates, runtime=_runtime(normalizers))
