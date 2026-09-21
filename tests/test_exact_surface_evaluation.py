import numpy as np
import pandas as pd
import pytest

from crimerisk.exact_surface_evaluation import (
    aggregate_to_support,
    bootstrap_summary,
    footprint_block_groups,
    fully_contained_tracts,
    normalize_scores,
    score_distribution,
)


def test_footprint_includes_zero_truth_cells_for_municipal_and_county_cases() -> None:
    frame = pd.DataFrame(
        {
            "block_group_geoid": ["010010001001", "010010001002", "010030001001"],
            "eb_jurisdiction_id": ["01:municipal:place:0101000", "01:municipal:place:0101000", "other"],
        }
    )
    assert list(footprint_block_groups(frame, "01:municipal:place:0101000")) == [
        "010010001001", "010010001002"
    ]
    assert list(footprint_block_groups(frame, "01:county:01001")) == [
        "010010001001", "010010001002"
    ]


def test_rare_offense_aggregates_full_bg_footprint_to_tract() -> None:
    frame = pd.DataFrame(
        {
            "block_group_geoid": ["010010001001", "010010001002", "010010002001"],
            "truth_count": [1.0, 0.0, 2.0],
            "population": [10.0, 20.0, 30.0],
            "score__arm": [2.0, 3.0, 5.0],
        }
    )
    tract = aggregate_to_support(frame, "murder").set_index("support_id")
    assert len(tract) == 2
    assert tract.loc["01001000100", "truth_count"] == pytest.approx(1.0)
    assert tract.loc["01001000100", "score__arm"] == pytest.approx(5.0)


def test_fully_contained_tracts_excludes_city_boundary_tract() -> None:
    surface = pd.DataFrame(
        {
            "block_group_geoid": ["010010001001", "010010001002", "010010002001"],
            "tract_id": ["01001000100", "01001000100", "01001000200"],
        }
    )
    footprint = pd.Index(["010010001001", "010010002001"])
    assert list(fully_contained_tracts(surface, footprint)) == ["01001000200"]


def test_prediction_epsilon_is_added_before_normalization() -> None:
    got = normalize_scores(pd.Series([0.0, 1.0]), epsilon=0.5)
    assert got == pytest.approx(np.array([0.25, 0.75]))


def test_top_decile_ties_use_support_id_order_and_exact_k() -> None:
    result = score_distribution(
        pd.Series([1.0, 0.0, 0.0, 0.0]),
        pd.Series([1.0, 1.0, 1.0, 1.0]),
        support_ids=pd.Series(["b", "a", "c", "d"]),
    )
    assert result["top_decile_k"] == 1
    assert result["top_decile_capture"] == pytest.approx(0.0)
    assert result["top_decile_tie_rule"] == "score_desc_support_id_asc_exact_k"


def test_cluster_uncertainty_is_null_for_one_independent_cluster() -> None:
    cells = pd.DataFrame({"source": ["only", "only"], "tvd": [0.2, 0.4]})
    result = bootstrap_summary(cells, metric="tvd", cluster="source", iterations=100)
    assert result["clusters"] == 1
    assert result["mean"] == pytest.approx(0.3)
    assert result["se"] is None
    assert result["ci95_low"] is None
    assert result["ci95_high"] is None
    assert result["uncertainty_reason"] == "insufficient_independent_clusters"
