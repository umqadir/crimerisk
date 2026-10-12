"""A gold run records every dependency the release validator requires of it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from crimerisk.eval import GoldEvaluation, GoldEvaluationConfig
from crimerisk.eval import gold as gold_module
from crimerisk.paths import RepoPaths

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "validate_release_outputs.py"
_SPEC = importlib.util.spec_from_file_location("validate_release_outputs_gold_dependencies", SCRIPT)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


def test_the_gold_fingerprint_covers_the_validators_required_set(tmp_path, monkeypatch):
    paths = RepoPaths.from_repo_root(tmp_path)
    for rel in (
        "main.py", "pyproject.toml", "uv.lock", "src/crimerisk/allocation.py",
        "src/crimerisk/eval/gold.py", "scripts/eval/gold_eval.py", "scripts/eval/level_holdout.py",
        "configs/service_relationships.csv", "configs/city_feeds/austin.json",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rel)
    centroids = paths.data_dir / "tiger_bg" / "parsed"
    centroids.mkdir(parents=True)
    pd.DataFrame({"bg_id": ["480010001001"], "lon": [-97.7], "lat": [30.3]}).to_parquet(
        centroids / "bg_centroids.parquet", index=False
    )
    candidate = tmp_path / "state" / "candidates" / "vtest-2025"
    candidate.mkdir(parents=True)
    for geography in ("block_group", "tract"):
        (candidate / f"crimerisk_{geography}_2025_ags_core.parquet").write_text(geography)
    truth = pd.DataFrame({"city_key": ["austin"], "jurisdiction_id": ["austin"], "incident_count": [1.0]})
    monkeypatch.setattr(gold_module, "_active_truth", lambda paths, truth_path=None: (truth, ["austin"]))
    evaluation = GoldEvaluation(
        paths=paths,
        config=GoldEvaluationConfig(
            run_id="gold_t", folds=("model",), model_candidate_dir=str(candidate), ags_comparator=False
        ),
    )
    fingerprint = evaluation.dependency_fingerprint()

    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "YEAR", 2025)
    required = validator._gold_required_dependencies(candidate)
    assert "scripts/eval/gold_eval.py" in required
    assert "state/candidates/vtest-2025/crimerisk_tract_2025_ags_core.parquet" in required
    assert required <= set(fingerprint)
    assert validator._dependency_binding_failures(fingerprint, required, digests={}) == []
