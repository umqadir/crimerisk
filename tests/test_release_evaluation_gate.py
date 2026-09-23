"""The release gate binds promotion to the release's own held-out numbers.

It used to require the measurement harness to recommend a particular next
workstream, which is a research conclusion and says nothing about whether this
release is any good. These tests pin what replaced it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"
_SPEC = importlib.util.spec_from_file_location("validate_release_outputs_under_test", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules["validate_release_outputs_under_test"] = validator
_SPEC.loader.exec_module(validator)


RUN_ID = "gold_test"
CANDIDATE = "vtest-2025"


def _results(overrides: dict[str, float] | None = None) -> pd.DataFrame:
    """A spatial TVD table where `ours` beats the population null everywhere."""
    beats = {
        "murder": (0.40, 0.60),
        "rape": (0.46, 0.48),
        "robbery": (0.41, 0.47),
        "aggravated_assault": (0.36, 0.45),
        "burglary": (0.29, 0.32),
        "larceny": (0.25, 0.38),
        "motor_vehicle_theft": (0.27, 0.29),
    }
    beats.update(overrides or {})
    rows = []
    for offense, (ours, null) in beats.items():
        for arm, estimate in (("ours", ours), ("population", null)):
            rows.append(
                {
                    "run_id": RUN_ID,
                    "fold_type": "spatial",
                    "offense": offense,
                    "arm": arm,
                    "metric": "tvd",
                    "estimate": estimate,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A self-contained release tree the check can be pointed at."""
    gold_dir = tmp_path / "state" / "eval" / RUN_ID
    gold_dir.mkdir(parents=True)
    (gold_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": RUN_ID, "config": {"year": 2025}})
    )
    (tmp_path / "state" / "candidates" / CANDIDATE).mkdir(parents=True)
    snapshot = tmp_path / "frontend" / "build" / "snapshot_config.env"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(
        "CRIMERISK_SNAPSHOT_SRC=state/candidates/"
        f"{CANDIDATE}/crimerisk_block_group_2025_ags_core.parquet\n"
    )

    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "RELEASE_GOLD_RUN_ID", RUN_ID)
    monkeypatch.setattr(validator, "RELEASE_GOLD_DIR", gold_dir)
    monkeypatch.setattr(validator, "RELEASE_CANDIDATE", CANDIDATE)
    monkeypatch.setattr(validator, "RELEASE_SNAPSHOT_CONFIG", snapshot)
    return gold_dir


def _run(gold_dir: Path, results: pd.DataFrame) -> tuple[list[str], dict]:
    results.to_csv(gold_dir / "results.csv", index=False)
    issues: list[str] = []
    summary = validator._check_release_evaluation(issues=issues)
    return issues, summary


def test_a_release_that_beats_the_population_null_everywhere_passes(release):
    issues, summary = _run(release, _results())
    assert issues == []
    assert summary["candidate_published"] == CANDIDATE
    assert all(row["beats_baseline"] for row in summary["spatial_tvd_vs_baseline"])


def test_losing_to_the_population_null_on_one_offense_fails_when_blocking(
    release, monkeypatch
):
    monkeypatch.setattr(validator, "RELEASE_GOLD_TVD_BLOCKING", True)
    issues, summary = _run(release, _results({"motor_vehicle_theft": (0.298, 0.287)}))
    assert summary["status"] == "blocking"
    assert len(issues) == 1
    assert "motor_vehicle_theft" in issues[0]
    losing = next(
        row
        for row in summary["spatial_tvd_vs_baseline"]
        if row["offense"] == "motor_vehicle_theft"
    )
    assert losing["beats_baseline"] is False


def test_a_loss_is_reported_not_raised_while_the_comparison_is_not_blocking(
    release, monkeypatch
):
    """2025.1.1: the loss is stated in the summary and is not an issue."""
    monkeypatch.setattr(validator, "RELEASE_GOLD_TVD_BLOCKING", False)
    issues, summary = _run(release, _results({"motor_vehicle_theft": (0.298, 0.287)}))
    assert issues == []
    assert summary["status"] == "reported"
    assert summary["passed"] is False
    assert len(summary["failures"]) == 1
    assert "motor_vehicle_theft" in summary["failures"][0]


def test_binding_failures_block_even_while_the_comparison_is_not_blocking(
    release, monkeypatch
):
    monkeypatch.setattr(validator, "RELEASE_GOLD_TVD_BLOCKING", False)
    results = _results()
    results["run_id"] = "gold_something_else"
    issues, _ = _run(release, results)
    assert any("release evaluation results name run ids" in issue for issue in issues)


def test_rape_is_reported_but_not_gated(release):
    """One eligible fold city is a report, not a test."""
    issues, summary = _run(release, _results({"rape": (0.50, 0.48)}))
    assert issues == []
    rape = next(
        row for row in summary["spatial_tvd_vs_baseline"] if row["offense"] == "rape"
    )
    assert rape["gated"] is False
    assert rape["beats_baseline"] is False


def test_a_missing_results_table_fails(release):
    issues: list[str] = []
    summary = validator._check_release_evaluation(issues=issues)
    assert summary["present"] is False
    assert any("missing release evaluation results" in issue for issue in issues)


def test_an_evaluation_of_another_candidate_fails(release, monkeypatch):
    snapshot = validator.RELEASE_SNAPSHOT_CONFIG
    snapshot.write_text(
        "CRIMERISK_SNAPSHOT_SRC=state/candidates/v99-2025/crimerisk_block_group_2025_ags_core.parquet\n"
    )
    issues, summary = _run(release, _results())
    assert summary["candidate_published"] == "v99-2025"
    assert any("published snapshot is built from" in issue for issue in issues)


def test_results_from_a_different_run_fail(release):
    results = _results()
    results["run_id"] = "gold_something_else"
    issues, _ = _run(release, results)
    assert any("release evaluation results name run ids" in issue for issue in issues)


def test_the_recommendation_string_is_no_longer_a_gate():
    source = SCRIPT.read_text()
    assert 'expected allocator_expansion_first' not in source
    assert '"recommended_next_workstream"' in source
