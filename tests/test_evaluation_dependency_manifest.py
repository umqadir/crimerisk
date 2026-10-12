"""Evaluation manifests come from the stages' own dependency declarations, and every input moves them.

The 2025.3 review's probe changed a CDE burglary value from 100 to 1,000 and found the level
holdout's digest mapping unchanged: the hand-written list omitted the state-estimate file the
smoother reads. The manifest is now the declared dependency closure
(`crimerisk.eval.dependencies`), shared by the harnesses and the release validator. These tests
mutate one input of each kind and require the digest to move and the validator to reject the run
recorded before the change; the gold fingerprint is held to the same for its raw inputs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from crimerisk.eval import GoldEvaluation, GoldEvaluationConfig
from crimerisk.eval import gold as gold_module
from crimerisk.eval.dependencies import build_outputs_dependency_paths, level_holdout_dependency_paths
from crimerisk.paths import RepoPaths

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "validate_release_outputs_dependency_manifest", ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"
)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)
_HOLDOUT_SPEC = importlib.util.spec_from_file_location(
    "level_holdout_dependency_manifest_under_test", ROOT / "scripts" / "eval" / "level_holdout.py"
)
holdout = importlib.util.module_from_spec(_HOLDOUT_SPEC)
sys.modules["level_holdout_dependency_manifest_under_test"] = holdout
_HOLDOUT_SPEC.loader.exec_module(holdout)

YEAR, PANEL_YEAR_END = 2024, 2025
CDE = f"data/FBI-CDE-Estimates-1979-{YEAR}/estimated_crimes_1979_{YEAR}.csv"
CDE_TEXT = "year,state_abbr,population,burglary\n2024,TX,1000000,{burglary}\n"
# One input of every kind the review named, at the path its stage declares or its reader reads.
INPUTS = {
    "cde_state_estimates": CDE,
    "agency_crosswalk": "state/reference/agency_to_jurisdiction_crosswalk.parquet",
    "block_group_geometry_crosswalk": "state/geometry/block_group_to_jurisdiction_crosswalk.parquet",
    # A parse with no annual file: observations does not declare it, the offense-months reader
    # (and so every fold's masked-gap detector) still reads it.
    "return_a_monthly_profile": "data/FBI-UCR-Return-A-Parsed-2024-20250101/return_a_master_monthly_2024.parquet",
    "numerical_reference_jurisdiction_master": "state/reference/jurisdiction_master.parquet",
    "numerical_reference_agency_master": "state/reference/agency_master.parquet",
    "reporting_regimes": "state/modeling/agency_year_reporting_regimes.parquet",
    "nibrs_offense_table": f"data/FBI-NIBRS-Tables-{YEAR}/parsed/nibrs_offense_type_by_agency_{YEAR}.parquet",
    "raw_srs_monthly": "data/SRS-Kaplan-1960-2024/offenses_known_parquet_1960_2024_month.zip",
    "config_state_anchor_exceptions": "configs/state_anchor_exceptions.csv",
    "config_nested": "configs/stage1_adjudications/ori_successions.csv",
    "code_smoother": "src/crimerisk/smoothed_controls.py",
    "observations": "state/observations/agency_year_observations.parquet",
}
ARMS = ("corrupted", "refused", "silent")
CONFIG = holdout.HoldoutConfig(year=YEAR, folds=3, mask_share=0.34, seed=7, arms=ARMS)
ELIGIBLE = [f"XX{index:07d}" for index in range(12)]


def _edition(root: Path) -> RepoPaths:
    for key, rel in INPUTS.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CDE_TEXT.format(burglary=100) if rel == CDE else f"{key}\n")
    # geometry enumerates its per-state raw inputs from the jurisdiction master.
    pd.DataFrame({"jurisdiction_id": ["48:x"], "state_fips": ["48"], "state_abbr": ["TX"]}).to_parquet(
        root / INPUTS["numerical_reference_jurisdiction_master"], index=False
    )
    tiger = root / "data" / "tiger_tabblock20" / "tl_2020_48_tabblock20.zip"
    tiger.parent.mkdir(parents=True, exist_ok=True)
    tiger.write_bytes(b"blocks")
    (root / "scripts" / "eval").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "eval" / "level_holdout.py").write_text("# holdout\n")
    return RepoPaths.from_repo_root(root)


def _mutate(root: Path, rel: str) -> None:
    path = root / rel
    if rel == CDE:
        path.write_text(CDE_TEXT.format(burglary=1000))
    elif rel == INPUTS["numerical_reference_jurisdiction_master"]:
        pd.DataFrame({"jurisdiction_id": ["48:y"], "state_fips": ["48"], "state_abbr": ["TX"]}).to_parquet(
            path, index=False
        )
    else:
        path.write_text(path.read_text() + "changed\n")


def _keys(paths: RepoPaths, files: list[Path]) -> set[str]:
    root = Path(paths.repo_root).resolve()
    return {
        str(path.resolve().relative_to(root)) for path in files if path.resolve().is_relative_to(root)
    }


def _digests(paths: RepoPaths) -> dict[str, str]:
    digests = holdout.level_holdout_input_digests(
        paths, year=YEAR, panel_year_end=PANEL_YEAR_END, remainder=False
    )
    return {key: value for key, value in digests.items() if not Path(key).is_absolute()}


def _write_run(paths: RepoPaths) -> Path:
    plan = holdout.planned_masks(ELIGIBLE, config=CONFIG, separate_arms=False)
    rows = [
        {"ori9": ori, "offense": offense, "arm": arm, "fold": int(fold), "config": "production_default",
         "observed_count": 10.0, "predicted_count": 9.0, "population": 5_000.0,
         "agency_estimate_source": "decayed_own_history"}
        for arm, offenses in holdout.planned_offenses(ARMS).items()
        for fold, oris in plan[arm].items()
        for ori in oris
        for offense in offenses
    ]
    out_dir = Path(paths.state_dir) / "eval" / "level_v9"
    holdout.write_report(
        out_dir=out_dir,
        agency_scored=holdout.score(pd.DataFrame(rows)),
        remainder_scored=pd.DataFrame(),
        config=CONFIG,
        meta={
            "config_label": "production_default", "panel_year_end": PANEL_YEAR_END,
            "remainder_scored": False, "edition_inputs_sha256": _digests(paths),
            "planned_masks": plan, "planned_offenses": holdout.planned_offenses(ARMS),
        },
    )
    return out_dir


def test_the_level_manifest_is_the_declared_closure(tmp_path):
    paths = _edition(tmp_path)
    keys = _keys(paths, level_holdout_dependency_paths(paths, year=YEAR, panel_year_end=PANEL_YEAR_END, remainder=False))
    assert set(INPUTS.values()) <= keys
    assert "scripts/eval/level_holdout.py" in keys
    # The geometry stage is not run by a level fold; its raw inputs are not level inputs.
    assert "data/tiger_tabblock20/tl_2020_48_tabblock20.zip" not in keys


def test_state_controls_inputs_are_built_by_the_fold_not_read(tmp_path):
    """The smoother declares state/controls files the fold builds itself on an empty
    state/controls; only the remainder lane reads production's smoothed controls."""
    paths = _edition(tmp_path)
    controls = Path(paths.state_dir) / "controls"
    controls.mkdir(parents=True)
    for name in ("jurisdiction_year_estimates.parquet", f"jurisdiction_controls_smoothed_{YEAR}.parquet"):
        (controls / name).write_text(name)
    without = _keys(paths, level_holdout_dependency_paths(paths, year=YEAR, panel_year_end=PANEL_YEAR_END, remainder=False))
    with_remainder = _keys(paths, level_holdout_dependency_paths(paths, year=YEAR, panel_year_end=PANEL_YEAR_END, remainder=True))
    assert not any(key.startswith("state/controls/") for key in without)
    assert with_remainder - without == {f"state/controls/jurisdiction_controls_smoothed_{YEAR}.parquet"}


@pytest.mark.parametrize("kind", sorted(INPUTS))
def test_changing_any_one_input_moves_the_digest_and_unbinds_the_run(tmp_path, monkeypatch, kind):
    paths = _edition(tmp_path)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _write_run(paths)
    issues: list[str] = []
    assert validator._check_level_holdout(issues=issues)["run_dir"] == "state/eval/level_v9"
    assert issues == []

    before = _digests(paths)
    rel = INPUTS[kind]
    _mutate(tmp_path, rel)
    after = _digests(paths)
    assert {key for key in before if before[key] != after.get(key)} == {rel}

    issues = []
    summary = validator._check_level_holdout(issues=issues)
    assert summary["run_dir"] is None
    assert summary["unbound"]["level_v9"] == [f"(a) 1 recorded digest(s) differ from the current files: {rel}"]


def test_a_new_return_a_parse_is_required_of_an_older_run(tmp_path, monkeypatch):
    paths = _edition(tmp_path)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _write_run(paths)
    added = tmp_path / "data" / "FBI-UCR-Return-A-Parsed-2025-20260921" / "return_a_master_monthly_2025.parquet"
    added.parent.mkdir(parents=True)
    added.write_text("monthly 2025")
    issues: list[str] = []
    summary = validator._check_level_holdout(issues=issues)
    assert summary["unbound"]["level_v9"] == [
        "(b) recorded digests omit 1 required file(s): "
        "data/FBI-UCR-Return-A-Parsed-2025-20260921/return_a_master_monthly_2025.parquet"
    ]


# --- gold ---------------------------------------------------------------------------------------

# A raw file in a city's declared feed directory (`city_shares._city_impls`).
CITY_FEED = "data/city-incidents/nyc/incidents_2024.csv"


def _gold(tmp_path: Path, monkeypatch) -> tuple[GoldEvaluation, Path]:
    paths = _edition(tmp_path)
    centroids = paths.data_dir / gold_module.BG_CENTROIDS_RELPATH
    centroids.parent.mkdir(parents=True)
    pd.DataFrame({"bg_id": ["480010001001"], "lon": [-97.7], "lat": [30.3]}).to_parquet(centroids, index=False)
    feed = paths.data_dir / CITY_FEED.removeprefix("data/")
    feed.parent.mkdir(parents=True)
    feed.write_text("id,offense\n1,burglary\n")
    candidate = tmp_path / "state" / "candidates" / "vtest-2025"
    candidate.mkdir(parents=True)
    for geography in ("block_group", "tract"):
        (candidate / f"crimerisk_{geography}_{YEAR}_ags_core.parquet").write_text(geography)
    truth = pd.DataFrame({"city_key": ["austin"], "jurisdiction_id": ["austin"], "incident_count": [1.0]})
    monkeypatch.setattr(gold_module, "_active_truth", lambda paths, truth_path=None: (truth, ["austin"]))
    evaluation = GoldEvaluation(
        paths=paths,
        config=GoldEvaluationConfig(
            run_id="gold_t", year=YEAR, folds=("model",), model_candidate_dir=str(candidate),
            ags_comparator=False,
        ),
    )
    return evaluation, candidate


GOLD_RAW = (
    CDE,
    CITY_FEED,
    INPUTS["return_a_monthly_profile"],
    INPUTS["raw_srs_monthly"],
    INPUTS["nibrs_offense_table"],
    "data/tiger_tabblock20/tl_2020_48_tabblock20.zip",
    "data/tiger_bg/parsed/bg_centroids.parquet",
    INPUTS["block_group_geometry_crosswalk"],
)


def test_the_gold_fingerprint_hashes_the_raw_inputs_folds_rebuild_from(tmp_path, monkeypatch):
    evaluation, _candidate = _gold(tmp_path, monkeypatch)
    fingerprint = evaluation.dependency_fingerprint()
    assert set(GOLD_RAW) <= set(fingerprint)
    assert _keys(evaluation.paths, build_outputs_dependency_paths(evaluation.paths, year=YEAR)) <= set(fingerprint)


@pytest.mark.parametrize(
    "rel", [CDE, CITY_FEED, "data/tiger_tabblock20/tl_2020_48_tabblock20.zip", INPUTS["return_a_monthly_profile"]]
)
def test_a_raw_input_change_unbinds_the_gold_run(tmp_path, monkeypatch, rel):
    evaluation, candidate = _gold(tmp_path, monkeypatch)
    fingerprint = evaluation.dependency_fingerprint()
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "YEAR", YEAR)
    manifest = {
        "completed_at_utc": "2026-10-11T00:00:00+00:00", "fold_ids": ["model_scoring_only"],
        "folds": [{"fold_id": "model_scoring_only"}], "dependencies": fingerprint,
    }
    required = validator._gold_required_dependencies(candidate)
    assert set(GOLD_RAW) <= required
    assert validator._gold_binding_failures(manifest, required=required, digests={}) == []

    if rel.endswith(".zip"):
        (tmp_path / rel).write_bytes(b"blocks, revised")
    else:
        _mutate(tmp_path, rel)
    assert evaluation.dependency_fingerprint()[rel] != fingerprint[rel]
    assert validator._gold_binding_failures(manifest, required=required, digests={}) == [
        f"(a) 1 recorded digest(s) differ from the current files: {rel}"
    ]
    json.dumps(fingerprint)  # the manifest stays serialisable
