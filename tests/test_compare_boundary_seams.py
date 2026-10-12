import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts/diagnostics/compare_boundary_seams.py"
SPEC = importlib.util.spec_from_file_location("compare_boundary_seams", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rare_support_pairs_are_deduplicated_tract_pairs():
    bg_pairs = pd.DataFrame([
        {"bg_a": "010010001001", "bg_b": "020010002001", "boundary_tags": "metro"},
        {"bg_a": "010010001002", "bg_b": "020010002002", "boundary_tags": "rural"},
        {"bg_a": "010010001001", "bg_b": "010010001002", "boundary_tags": "within"},
    ])
    bg = pd.DataFrame({
        "block_group_geoid": ["010010001001", "010010001002", "020010002001", "020010002002"],
        "tract_id": ["01001000100", "01001000100", "02001000200", "02001000200"],
    })
    result = MODULE.derive_support_pairs(bg_pairs, bg, "tract")
    assert result[["support_a", "support_b"]].to_dict("records") == [
        {"support_a": "01001000100", "support_b": "02001000200"}
    ]
    assert result.iloc[0].boundary_tags == "metro;rural"


def test_pair_metrics_keep_zero_and_missing_out_of_log_ratios():
    assert MODULE.pair_metric(0, 0) == ("both_zero", None, None)
    assert MODULE.pair_metric(0, 2) == ("one_zero", None, None)
    assert MODULE.pair_metric(np.nan, 2) == ("missing", None, None)
    status, signed, absolute = MODULE.pair_metric(2, 8)
    assert status == "positive"
    assert np.isclose(signed, np.log(4))
    assert np.isclose(absolute, np.log(4))


def test_pair_plan_tags_cross_state_navajo_edges():
    adjacency = pd.DataFrame({"bg": ["040010001001"], "nb": ["350010001001"]})
    centroids = pd.DataFrame({
        "bg_id": ["040010001001", "350010001001"],
        "lon": [-109.1, -109.0], "lat": [35.0, 35.0],
    })
    bg = pd.DataFrame({
        "block_group_geoid": ["040010001001", "350010001001"],
        "tract_id": ["04001000100", "35001000100"],
        "state_fips": ["04", "35"],
        "eb_jurisdiction_id": ["az", "nm"],
        "urban_stratum": ["rural", "rural"],
    })
    plan = MODULE.build_bg_pair_plan(
        adjacency, centroids, bg, {"040010001001"}, rural_per_state_pair=1,
        navajo_ids={"040010001001"},
    )
    tags = set(plan.iloc[0].boundary_tags.split(";"))
    assert {"custom_tribal_perimeter", "cross_state_rural", "navajo_cross_state"} <= tags


def test_pairwise_index_decomposition_and_change_driver():
    pairs = pd.DataFrame([{
        "support_a": "a", "support_b": "b", "boundary_tags": "cross_state_rural"
    }])
    baseline = pd.DataFrame({
        "block_group_geoid": ["a", "b"],
        "expected_count_robbery": [2.0, 8.0],
        "primary_denominator_robbery": [10.0, 20.0],
        "index_robbery_primary": [20.0, 40.0],
    })
    candidate = baseline.copy()
    candidate["primary_denominator_robbery"] = [20.0, 20.0]
    candidate["index_robbery_primary"] = [10.0, 40.0]
    result = MODULE.evaluate_pairs(
        pairs, baseline, candidate, support="block_group", offense="robbery"
    ).iloc[0]
    assert result.baseline_decomposition_ok
    assert result.candidate_decomposition_ok
    assert result.numeric_change_driver == "denominator"
    assert np.isclose(result.change_index_signed_log_ratio, np.log(2))
    assert result.baseline_count_a == 2.0
    assert result.candidate_denominator_a == 20.0


def test_summary_reports_positive_zero_and_missing_counts_separately():
    details = pd.DataFrame({
        "support": ["block_group"] * 4,
        "offense": ["robbery"] * 4,
        "boundary_tags": ["rural"] * 4,
        "baseline_count_status": ["positive", "both_zero", "one_zero", "missing"],
        "baseline_count_abs_log_ratio": [1.0, None, None, None],
        "candidate_count_status": ["positive", "both_zero", "one_zero", "missing"],
        "candidate_count_abs_log_ratio": [2.0, None, None, None],
        **{
            f"{version}_{metric}_{suffix}": values
            for version in ("baseline", "candidate")
            for metric in ("denominator", "index")
            for suffix, values in (
                ("status", ["positive"] * 4),
                ("abs_log_ratio", [1.0] * 4),
            )
        },
    })
    row = MODULE.summarize(details).query(
        "version == 'baseline' and metric == 'count'"
    ).iloc[0]
    assert row.positive_pair_count == 1
    assert row.both_zero_pair_count == 1
    assert row.one_zero_pair_count == 1
    assert row.missing_pair_count == 1
