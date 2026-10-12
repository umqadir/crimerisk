"""Cross-version comparisons re-score both versions on one frozen, keyed processed truth."""

from __future__ import annotations

import pandas as pd
import pytest

from crimerisk.eval.gold import processed_truth_sha256, processed_truth_table, rescore_on_truth


def _retained(fold: str, ids: list[str], truth: list[float], prediction: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": fold, "fold_type": "spatial", "city_name": "Austin", "jurisdiction_id": "48:municipal:place:4805000",
            "ags_uses_city_feed": True, "offense": "robbery", "support_role": "primary", "support": "block_group",
            "cell_scope": "primary", "support_id": ids, "truth_count": truth, "population": [1000.0] * len(ids),
            "support_lon": [-97.7 + 0.01 * i for i in range(len(ids))], "support_lat": [30.3] * len(ids),
            "prediction_ours": prediction,
        }
    )


def test_a_baseline_is_scored_on_this_runs_truth_and_a_different_support_is_never_paired():
    current = _retained("spatial_austin", ["a", "b", "c"], [5.0, 3.0, 2.0], [4.0, 4.0, 2.0])
    truth = processed_truth_table(current)
    # The baseline scored a different truth (feed pool grew) on the same support.
    baseline = _retained("spatial_austin", ["a", "b", "c"], [50.0, 1.0, 1.0], [6.0, 2.0, 2.0])
    cells, excluded = rescore_on_truth(truth, baseline, arm_column="prediction_ours", arm_name="baseline_ours")
    assert excluded.empty and len(cells) == 1
    # TVD against THIS run's truth (0.5, 0.3, 0.2) for the baseline prediction (0.6, 0.2, 0.2).
    assert cells["tvd"].iloc[0] == pytest.approx(0.1)
    assert cells["n_incidents"].iloc[0] == 10.0

    other_support = _retained("spatial_austin", ["a", "b", "d"], [5.0, 3.0, 2.0], [6.0, 2.0, 2.0])
    cells, excluded = rescore_on_truth(truth, other_support, arm_column="prediction_ours", arm_name="baseline_ours")
    assert cells.empty and list(excluded["reason"]) == ["support_ids_differ"]


def test_the_processed_truth_hash_moves_with_any_truth_value():
    current = _retained("spatial_austin", ["a", "b"], [5.0, 3.0], [1.0, 1.0])
    other = _retained("spatial_austin", ["a", "b"], [5.0, 4.0], [1.0, 1.0])
    assert processed_truth_sha256(processed_truth_table(current)) != processed_truth_sha256(processed_truth_table(other))


def test_the_processed_truth_hash_moves_with_population_and_coordinates():
    current = _retained("spatial_austin", ["a", "b"], [5.0, 3.0], [1.0, 1.0])
    reference = processed_truth_sha256(processed_truth_table(current))
    for column, value in (("population", 1001.0), ("support_lon", -97.0), ("support_lat", 30.31)):
        other = current.copy()
        other.loc[1, column] = value
        assert processed_truth_sha256(processed_truth_table(other)) != reference, column


def test_duplicate_prediction_keys_raise_instead_of_passing_the_support_guard():
    """The review's probe: three prediction values for two truth cells share the truth's ID set."""
    truth = processed_truth_table(_retained("spatial_austin", ["a", "b"], [5.0, 3.0], [1.0, 1.0]))
    duplicated = _retained("spatial_austin", ["a", "b", "b"], [5.0, 3.0, 3.0], [4.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="prediction table repeats 1"):
        rescore_on_truth(truth, duplicated, arm_column="prediction_ours", arm_name="baseline_ours")


def test_duplicate_truth_keys_raise():
    truth = _retained("spatial_austin", ["a", "a"], [5.0, 3.0], [1.0, 1.0])
    prediction = _retained("spatial_austin", ["a"], [5.0], [1.0])
    with pytest.raises(ValueError, match="truth table repeats 1"):
        rescore_on_truth(truth, prediction, arm_column="prediction_ours", arm_name="baseline_ours")
