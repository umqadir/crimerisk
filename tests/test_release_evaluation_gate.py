"""The release gate binds promotion to the release's own held-out numbers.

It used to require the measurement harness to recommend a particular next
workstream, which is a research conclusion and says nothing about whether this
release is any good. These tests pin what replaced it.
"""

from __future__ import annotations

import hashlib
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(root: Path, name: str, *, content: str = "bg") -> Path:
    """A candidate directory: build manifest naming its run id, and the two published surfaces."""
    candidate = root / "state" / "candidates" / name
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "manifest.json").write_text(json.dumps({"run": {"run_id": name}}))
    (candidate / "crimerisk_block_group_2025_ags_core.parquet").write_text(content)
    (candidate / "crimerisk_tract_2025_ags_core.parquet").write_text(content + "-tract")
    return candidate


CODE = {
    "main.py": "# entry\n",
    "pyproject.toml": "[project]\n",
    "src/crimerisk/allocation.py": "# allocation\n",
    "src/crimerisk/eval/gold.py": "# gold\n",
    "scripts/eval/gold_eval.py": "# harness\n",
    "configs/service_relationships.csv": "provider\n",
}


def _code_tree(root: Path) -> None:
    for rel, text in CODE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def _dependencies(root: Path, candidate: Path) -> dict[str, str]:
    """What the gold harness records: code, configs and the candidate's surfaces."""
    files = [root / rel for rel in CODE] + [
        candidate / "crimerisk_block_group_2025_ags_core.parquet",
        candidate / "crimerisk_tract_2025_ags_core.parquet",
    ]
    return {str(path.relative_to(root)): _sha256(path) for path in files}


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A candidate plus the completed gold run whose model fold scored its bytes."""
    _code_tree(tmp_path)
    candidate = _candidate(tmp_path, CANDIDATE)
    gold_dir = tmp_path / "state" / "eval" / RUN_ID
    gold_dir.mkdir(parents=True)
    _write_manifest(gold_dir, candidate)
    snapshot = tmp_path / "frontend" / "build" / "snapshot_config.env"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(
        "CRIMERISK_SNAPSHOT_SRC=state/candidates/v1-2025/crimerisk_block_group_2025_ags_core.parquet\n"
    )
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "RELEASE_SNAPSHOT_CONFIG", snapshot)
    monkeypatch.setattr(validator, "YEAR", 2025)
    return gold_dir


def _write_manifest(gold_dir: Path, candidate: Path, *, results_sha256: str | None = None) -> None:
    (gold_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": gold_dir.name,
                "config": {"year": 2025},
                "completed_at_utc": "2026-09-25T00:00:00+00:00",
                "results_sha256": results_sha256,
                "dependencies": _dependencies(gold_dir.parents[2], candidate),
                "fold_ids": ["model_scoring_only"],
                "folds": [
                    {
                        "fold_id": "model_scoring_only",
                        "fold_type": "model",
                        "candidate_dir": str(candidate),
                        "surface_sha256": {
                            "block_group": _sha256(candidate / "crimerisk_block_group_2025_ags_core.parquet"),
                            "tract": _sha256(candidate / "crimerisk_tract_2025_ags_core.parquet"),
                        },
                    }
                ],
            }
        )
    )


def _candidate_dir(gold_dir: Path, name: str = CANDIDATE) -> Path:
    return gold_dir.parents[1] / "candidates" / name


def _run(gold_dir: Path, results: pd.DataFrame, *, output_dir: Path | None = None) -> tuple[list[str], dict]:
    results.to_csv(gold_dir / "results.csv", index=False)
    manifest = json.loads((gold_dir / "run_manifest.json").read_text())
    manifest["results_sha256"] = _sha256(gold_dir / "results.csv")
    (gold_dir / "run_manifest.json").write_text(json.dumps(manifest))
    issues: list[str] = []
    summary = validator._check_release_evaluation(
        output_dir=output_dir or _candidate_dir(gold_dir), issues=issues
    )
    return issues, summary


def test_a_release_that_beats_the_population_null_everywhere_passes(release):
    issues, summary = _run(release, _results())
    assert issues == []
    assert summary["candidate_id"] == CANDIDATE
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
    summary = validator._check_release_evaluation(output_dir=_candidate_dir(release), issues=issues)
    assert summary["present"] is False
    assert any("missing release evaluation results" in issue for issue in issues)


def test_the_gate_evaluates_the_candidate_under_validation_not_the_snapshot(release):
    issues, summary = _run(release, _results())
    assert issues == []
    assert summary["candidate_id"] == CANDIDATE
    assert summary["run_id"] == RUN_ID
    assert summary["snapshot_candidate"] == "v1-2025"


def test_a_candidate_no_gold_run_scored_fails(release):
    """Different bytes: the gold run scored another surface, even though names line up."""
    other = _candidate(release.parents[2], "vother-2025", content="other")
    issues, summary = _run(release, _results(), output_dir=other)
    assert summary["candidate_id"] == "vother-2025"
    assert summary["gold_runs_scoring_candidate"] == []
    assert any("no completed gold run" in issue for issue in issues)


def test_the_same_bytes_under_another_candidate_id_fail(release):
    copy = _candidate(release.parents[2], "vcopy-2025")
    issues, _ = _run(release, _results(), output_dir=copy)
    assert any("but the build manifest names 'vcopy-2025'" in issue for issue in issues)


def test_an_incomplete_gold_run_does_not_bind(release):
    manifest = json.loads((release / "run_manifest.json").read_text())
    del manifest["completed_at_utc"]
    (release / "run_manifest.json").write_text(json.dumps(manifest))
    _results().to_csv(release / "results.csv", index=False)
    issues: list[str] = []
    validator._check_release_evaluation(output_dir=_candidate_dir(release), issues=issues)
    assert any("no completed gold run" in issue for issue in issues)


def test_a_gold_run_from_the_pre_fold_scoping_harness_fails(release):
    manifest = json.loads((release / "run_manifest.json").read_text())
    del manifest["dependencies"]
    (release / "run_manifest.json").write_text(json.dumps(manifest))
    issues, _ = _run(release, _results())
    assert any("predates fold-scoped gold folds" in issue for issue in issues)


def test_results_edited_after_the_run_fail(release):
    issues, _ = _run(release, _results())
    assert issues == []
    edited = _results({"larceny": (0.20, 0.38)})
    edited.to_csv(release / "results.csv", index=False)
    issues = []
    validator._check_release_evaluation(output_dir=_candidate_dir(release), issues=issues)
    assert any("changed since run" in issue for issue in issues)


def test_a_build_without_a_candidate_run_id_fails(release, tmp_path):
    loose = tmp_path / "state" / "output"
    loose.mkdir(parents=True)
    (loose / "manifest.json").write_text(json.dumps({"run": {}}))
    issues: list[str] = []
    validator._check_release_evaluation(output_dir=loose, issues=issues)
    assert any("names no candidate run id" in issue for issue in issues)


def test_results_from_a_different_run_fail(release):
    results = _results()
    results["run_id"] = "gold_something_else"
    issues, _ = _run(release, results)
    assert any("release evaluation results name run ids" in issue for issue in issues)


def test_the_recommendation_string_is_no_longer_a_gate():
    source = SCRIPT.read_text()
    assert 'expected allocator_expansion_first' not in source
    assert '"recommended_next_workstream"' in source


def test_the_review_probe_does_not_bind(release):
    """A run that hashed one unrelated file binds nothing."""
    root = release.parents[2]
    (root / "notes.txt").write_text("unrelated")
    manifest = json.loads((release / "run_manifest.json").read_text())
    manifest["dependencies"] = {"notes.txt": _sha256(root / "notes.txt")}
    (release / "run_manifest.json").write_text(json.dumps(manifest))
    issues, summary = _run(release, _results())
    assert len(issues) == 1 and "(b) recorded digests omit 8 required file(s)" in issues[0]
    assert summary["binding_failures"] == [issues[0].split("candidate: ", 1)[1]]


def test_code_changed_after_the_run_unbinds_it(release):
    root = release.parents[2]
    (root / "src" / "crimerisk" / "allocation.py").write_text("# allocation, edited\n")
    issues, _ = _run(release, _results())
    assert len(issues) == 1
    assert "(a) 1 recorded digest(s) differ" in issues[0] and "src/crimerisk/allocation.py" in issues[0]


def test_code_added_after_the_run_is_required(release):
    root = release.parents[2]
    (root / "scripts" / "eval" / "new_harness.py").write_text("# new\n")
    (root / "configs" / "new.csv").write_text("x\n")
    issues, _ = _run(release, _results())
    assert len(issues) == 1 and "(b) recorded digests omit 2 required file(s)" in issues[0]
    assert "configs/new.csv" in issues[0] and "scripts/eval/new_harness.py" in issues[0]


def test_a_run_that_did_not_hash_the_candidate_surfaces_does_not_bind(release):
    manifest = json.loads((release / "run_manifest.json").read_text())
    manifest["dependencies"] = {
        k: v for k, v in manifest["dependencies"].items() if not k.startswith("state/candidates/")
    }
    (release / "run_manifest.json").write_text(json.dumps(manifest))
    issues, _ = _run(release, _results())
    assert len(issues) == 1 and "omit 2 required" in issues[0]
    assert "crimerisk_block_group_2025_ags_core.parquet" in issues[0]


def test_a_run_missing_a_planned_fold_does_not_bind(release):
    manifest = json.loads((release / "run_manifest.json").read_text())
    manifest["fold_ids"] = ["spatial_new_york", "model_scoring_only"]
    (release / "run_manifest.json").write_text(json.dumps(manifest))
    issues, _ = _run(release, _results())
    assert len(issues) == 1 and "(c) fold records ['model_scoring_only'] do not match" in issues[0]


def test_an_older_binding_run_is_preferred_to_a_later_unbound_one(release):
    later = release.parent / "gold_test_later"
    later.mkdir()
    manifest = json.loads((release / "run_manifest.json").read_text())
    manifest.update(run_id=later.name, completed_at_utc="2026-10-01T00:00:00+00:00", dependencies={"x": "0"})
    (later / "run_manifest.json").write_text(json.dumps(manifest))
    issues, summary = _run(release, _results())
    assert sorted(summary["gold_runs_scoring_candidate"]) == ["gold_test", "gold_test_later"]
    assert summary["run_id"] == RUN_ID and issues == []
