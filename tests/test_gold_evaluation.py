from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.eval.gold import (
    GoldEvaluation,
    GoldEvaluationConfig,
    _aggregate_bg_counts_to_tracts,
    _log_width_requirements,
    _reuse_flags,
    _score_vector,
    _share_coverage,
)


def test_distribution_metrics_report_losses_without_filtering() -> None:
    truth = np.array([8.0, 2.0, 0.0])
    population = np.array([5.0, 3.0, 2.0])
    result = _score_vector(
        truth_count=truth,
        prediction=np.array([0.0, 2.0, 8.0]),
        support_ids=np.array(["a", "b", "c"]),
        population=population,
        distance_matrix=np.array(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
        ),
    )
    assert result["tvd"] > 0.5
    assert result["w1_spatial_skill"] < 0.0
    assert set(result) == {
        "tvd", "spearman", "top_decile_capture", "w1_spatial_skill"
    }


def test_undefined_zero_mass_baseline_reports_nan_metrics() -> None:
    result = _score_vector(
        truth_count=np.array([8.0, 2.0]),
        prediction=np.array([0.0, 0.0]),
        support_ids=np.array(["a", "b"]),
        population=np.array([5.0, 5.0]),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert set(result) == {
        "tvd", "spearman", "top_decile_capture", "w1_spatial_skill"
    }
    assert all(np.isnan(value) for value in result.values())


def test_share_coverage_uses_one_candidate_total_for_both_bounds() -> None:
    assert _share_coverage(
        np.array([6.0, 4.0]),
        np.array([5.0, 3.0]),
        np.array([7.0, 5.0]),
        np.array([6.0, 4.0]),
    ) == 1.0


def test_ours_tract_sums_block_groups_on_requested_ags_universe() -> None:
    frame = pd.DataFrame(
        {
            "tract_id": ["1", "1", "2", "3"],
            "expected_count_robbery": [1.25, 2.75, 4.0, 99.0],
        }
    )
    result = _aggregate_bg_counts_to_tracts(
        frame, offense="robbery", tract_ids=["1", "2", "4"]
    )
    assert result.to_dict() == {"1": 4.0, "2": 4.0, "4": 0.0}


def test_log_width_factor_is_one_when_truth_is_on_existing_bound() -> None:
    requirements = _log_width_requirements(
        truth_count=np.array([8.0, 2.0]),
        lower_count=np.array([4.0, 1.0]),
        upper_count=np.array([8.0, 4.0]),
        point_count=np.array([6.0, 4.0]),
    )
    assert np.isclose(requirements[0], 1.0)


def test_reuse_flags_name_every_governed_constant() -> None:
    flags = _reuse_flags("model", "rape")
    for name in (
        "mixture_weights_v3",
        "soft_shrinkage",
        "rape_triple",
        "exposure_ensemble_weights",
        "murder_K",
        "tau",
    ):
        assert f"{name}=" in flags


def test_results_are_long_form_with_city_cluster_intervals() -> None:
    evaluator = GoldEvaluation.__new__(GoldEvaluation)
    evaluator.config = GoldEvaluationConfig(
        run_id="test",
        bootstrap_iterations=50,
        bootstrap_seed=7,
    )
    cells = pd.DataFrame(
        {
            "fold_type": ["spatial", "spatial"],
            "offense": ["robbery", "robbery"],
            "arm": ["ours", "ours"],
            "city_name": ["a", "b"],
            "jurisdiction_id": ["a", "b"],
            "n_cells": [10, 20],
            "n_incidents": [100.0, 200.0],
            "reuse_flags": ["tau=not_seen", "tau=not_seen"],
            "tvd": [0.2, 0.4],
        }
    )
    result = evaluator._summarize(cells)
    row = result.iloc[0]
    assert row["metric"] == "tvd"
    assert np.isclose(row["estimate"], 0.3)
    assert row["n_cities"] == 2
    assert row["n_cells"] == 30
    assert row["n_incidents"] == 300.0


def test_results_split_ags_feed_and_feed_free_cities() -> None:
    evaluator = GoldEvaluation.__new__(GoldEvaluation)
    evaluator.config = GoldEvaluationConfig(
        run_id="test",
        bootstrap_iterations=10,
        bootstrap_seed=7,
    )
    cells = pd.DataFrame(
        {
            "fold_type": ["spatial", "spatial"],
            "offense": ["robbery", "robbery"],
            "arm": ["ours_tract", "ours_tract"],
            "ags_uses_city_feed": [False, True],
            "city_name": ["feed_free", "feed"],
            "jurisdiction_id": ["a", "b"],
            "n_cells": [10, 20],
            "n_incidents": [100.0, 200.0],
            "reuse_flags": ["tau=not_seen", "tau=not_seen"],
            "tvd": [0.2, 0.4],
        }
    )
    result = evaluator._summarize(cells)
    assert result["ags_uses_city_feed"].tolist() == [False, True]
    assert result["estimate"].tolist() == [0.2, 0.4]
