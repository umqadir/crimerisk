"""A level-holdout fold hands the state anchor its own admission, mass ledger and observations.

Production's smoother reads the controls stage's level-lane admission and mass ledger from
state/controls. A fold's state/controls is empty, so a fold that let the smoother default to
those files would either lose the documented corrections or (pointed at production) read the
edition's unmasked ledger. These tests pin that the fold path passes non-empty fold-local
anchor inputs explicitly, that masking reaches the admission frame as it reaches the panel,
and that a fold without a mass ledger fails instead of smoothing without corrections.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths  # noqa: E402
from crimerisk.smoothed_controls import StateAnchorError, load_anchor_inputs  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "level_holdout_anchor_under_test", REPO_ROOT / "scripts" / "eval" / "level_holdout.py"
)
holdout = importlib.util.module_from_spec(_SPEC)
sys.modules["level_holdout_anchor_under_test"] = holdout
_SPEC.loader.exec_module(holdout)

YEAR = 2024
FOLD_ORI = "XX0000001"
PROD_ORI = "PROD00001"


def _paths(tmp_path: Path) -> RepoPaths:
    paths = RepoPaths.from_repo_root(tmp_path / "repo")
    observations = paths.state_dir / "observations"
    observations.mkdir(parents=True)
    pd.DataFrame(
        {
            "ori9": [FOLD_ORI, FOLD_ORI, PROD_ORI],
            "year": [YEAR, YEAR - 1, YEAR],
            "source": "srs_return_a_annual",
            "offense": "burglary",
            "count": [40.0, 41.0, 7.0],
            "months_reported": 12.0,
        }
    ).to_parquet(observations / "agency_year_observations.parquet", index=False)
    # The edition's own anchor inputs, which no fold may read.
    controls = paths.state_dir / "controls"
    controls.mkdir(parents=True)
    pd.DataFrame(
        {"ori9": [PROD_ORI], "state_abbr": "XX", "year": YEAR, "offense": "burglary",
         "level_lane_hard_case_id": "prod_case", "level_repair_mode": "none"}
    ).to_parquet(controls / f"level_lane_admission_{YEAR}.parquet", index=False)
    pd.DataFrame(
        {"ori9": [PROD_ORI], "year": YEAR, "offense": "burglary", "final_control_mass": 9_999.0}
    ).to_parquet(controls / f"level_lane_mass_ledger_{YEAR}.parquet", index=False)
    return paths


def _fold_admission() -> pd.DataFrame:
    return pd.DataFrame(
        {"ori9": [FOLD_ORI], "state_abbr": "XX", "year": YEAR, "offense": "burglary",
         "level_lane_hard_case_id": "fold_case", "level_repair_mode": "source_supported_replacement"}
    )


def _fold_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {"ori9": [FOLD_ORI], "year": YEAR, "offense": "burglary", "final_control_mass": 55.0}
    )


@pytest.fixture
def stages(monkeypatch):
    """The controls stage and the smoother are stubbed; the fold wiring between them is real."""
    captured: dict[str, object] = {}

    def fake_bundle(*, paths, config, stage1):
        captured["bundle_paths"] = paths
        imputation = captured.get("imputation", SimpleNamespace(mass_ledger=_fold_ledger()))
        return pd.DataFrame(), None, pd.DataFrame(), None, imputation, stage1.agency_estimates

    def fake_smoothing(*, paths, config, accounting, panel, anchor_inputs):
        captured["smoothing_paths"] = paths
        captured["anchor_inputs"] = anchor_inputs
        # The fold's own state/controls holds no anchor artifacts: had the smoother been
        # left to read them, it would have had nothing (or production's) to read.
        captured["fold_controls_files"] = sorted(p.name for p in (Path(paths.state_dir) / "controls").iterdir())
        return pd.DataFrame({"jurisdiction_id": ["j"], "smoothed_count": [1.0]}), {}

    monkeypatch.setattr(holdout, "build_controls_bundle", fake_bundle)
    monkeypatch.setattr(holdout, "build_smoothed_controls_bundle", fake_smoothing)
    monkeypatch.setattr(holdout, "resolve_ori_succession", lambda **kwargs: (pd.DataFrame(), {}))
    monkeypatch.setattr(holdout, "load_agency_jurisdiction_crosswalk", lambda paths: pd.DataFrame())
    monkeypatch.setattr(
        holdout,
        "build_agency_target_panel_slice",
        lambda *, agency_panel, target_year: agency_panel[agency_panel["year"].eq(target_year)],
    )
    return captured


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {"ori9": [FOLD_ORI, FOLD_ORI], "year": [YEAR - 1, YEAR], "offense": "burglary",
         "preferred_count": [41.0, 40.0]}
    )


def test_the_fold_path_passes_non_empty_fold_local_anchor_inputs_to_the_smoother(tmp_path, stages):
    paths = _paths(tmp_path)
    work_dir = paths.state_dir / "eval" / "run" / "work" / "fold0"
    fold = holdout.fold_controls(
        paths,
        panel=_panel(),
        estimates=pd.DataFrame({"ori9": [FOLD_ORI], "offense": ["burglary"]}),
        admission=_fold_admission(),
        year=YEAR,
        work_dir=work_dir,
    )
    inputs = stages["anchor_inputs"]
    assert inputs is not None and inputs is fold.anchor_inputs
    assert not inputs.admission.empty and not inputs.mass_ledger.empty and not inputs.observations.empty
    pd.testing.assert_frame_equal(inputs.admission, _fold_admission())
    pd.testing.assert_frame_equal(inputs.mass_ledger, _fold_ledger())
    # Nothing of the edition's state/controls reached the anchor.
    assert PROD_ORI not in set(inputs.admission["ori9"]) | set(inputs.mass_ledger["ori9"])
    # Observations come from the fold's state (read through to the edition's input), target year only.
    assert set(inputs.observations["year"]) == {YEAR}
    assert set(inputs.observations.columns) == set(holdout.ANCHOR_OBSERVATION_COLUMNS)
    # The smoother ran on the fold's paths, whose controls directory has no anchor artifacts.
    smoothing_state = Path(stages["smoothing_paths"].state_dir)
    assert smoothing_state == work_dir / "state"
    assert not any(name.startswith("level_lane_") for name in stages["fold_controls_files"])
    with pytest.raises(StateAnchorError):
        load_anchor_inputs(stages["smoothing_paths"], year=YEAR)


def test_a_fold_without_a_mass_ledger_fails_instead_of_smoothing_without_corrections(tmp_path, stages):
    paths = _paths(tmp_path)
    stages["imputation"] = None
    with pytest.raises(ValueError, match="mass ledger"):
        holdout.fold_controls(
            paths,
            panel=_panel(),
            estimates=pd.DataFrame({"ori9": [FOLD_ORI], "offense": ["burglary"]}),
            admission=_fold_admission(),
            year=YEAR,
            work_dir=tmp_path / "work",
        )
    with pytest.raises(ValueError, match="admission"):
        holdout.fold_controls(
            paths,
            panel=_panel(),
            estimates=pd.DataFrame({"ori9": [FOLD_ORI], "offense": ["burglary"]}),
            admission=_fold_admission().iloc[0:0],
            year=YEAR,
            work_dir=tmp_path / "work2",
        )


def test_an_estimator_pass_masks_the_admission_like_the_panel(tmp_path, monkeypatch):
    silent, refused, kept = "XX0000002", "XX0000003", "XX0000004"
    oris = [silent, refused, kept]
    preferred = pd.DataFrame(
        {"ori9": oris * 2, "year": [YEAR - 1] * 3 + [YEAR] * 3, "offense": "burglary",
         "preferred_count": 10.0, "state_abbr": "XX"}
    )
    captured: dict[str, object] = {}

    def fake_admit(paths, *, panel, year):
        target = panel[panel["year"].eq(year)].assign(
            level_lane_hard_case_id="case", level_repair_mode="none",
            level1_admission_status="valid_complete_year",
        )
        return panel.copy(), target.reset_index(drop=True)

    def fake_fold_controls(paths, *, panel, estimates, admission, year, work_dir):
        captured["admission"] = admission
        return SimpleNamespace(smoothed=pd.DataFrame())

    monkeypatch.setattr(holdout, "admit", fake_admit)
    monkeypatch.setattr(
        holdout, "build_agency_allocation_target_estimates", lambda **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(holdout, "fold_controls", fake_fold_controls)
    holdout.estimate_pass(
        RepoPaths.from_repo_root(tmp_path),
        preferred=preferred,
        year=YEAR,
        masked_by_arm={"silent": {silent}, "refused": {refused}},
        work_dir=tmp_path / "work",
    )
    admission = captured["admission"].set_index("ori9")
    assert silent not in admission.index
    assert admission.loc[refused, "level1_admission_status"] == "coverage_defective"
    assert admission.loc[kept, "level1_admission_status"] == "valid_complete_year"


def _repo_with_inputs(tmp_path: Path) -> RepoPaths:
    root = tmp_path / "repo"
    paths = RepoPaths.from_repo_root(root)
    configs = root / "configs"
    (configs / "stage1_adjudications").mkdir(parents=True)
    for name in (
        "service_relationships.csv", "level_lane_admission_registry.csv", "level_lane_hard_evidence.csv",
        "reporting_regime_restrictions.csv", "overlap_custom_footprints.csv",
        "service_wide_agency_scopes.csv", "level_lane_structural_zero_screen.csv",
        "stage1_adjudications/twins_adjudicated.csv",
    ):
        (configs / name).write_text(f"{name}\n")
    source = root / "src" / "crimerisk"
    source.mkdir(parents=True)
    for name in ("controls.py", "smoothed_controls.py", "contract_coverage.py"):
        (source / name).write_text(f"# {name}\n")
    observations = paths.state_dir / "observations"
    observations.mkdir(parents=True)
    (observations / "agency_year_observations.parquet").write_bytes(b"observations")
    return paths


def test_the_input_digests_cover_code_configs_and_observations(tmp_path):
    paths = _repo_with_inputs(tmp_path)
    digests = holdout.level_holdout_input_digests(paths, year=YEAR, panel_year_end=2025, remainder=False)
    required = {
        "state/observations/agency_year_observations.parquet",
        "configs/service_relationships.csv",
        "configs/level_lane_admission_registry.csv",
        "configs/level_lane_hard_evidence.csv",
        "configs/reporting_regime_restrictions.csv",
        "configs/stage1_adjudications/twins_adjudicated.csv",
        "src/crimerisk/controls.py",
        "src/crimerisk/smoothed_controls.py",
        "src/crimerisk/contract_coverage.py",
    }
    assert required <= set(digests)
    # This script and the admission module are code the estimator runs.
    assert any(key.endswith("scripts/eval/level_holdout.py") for key in digests)
    assert any(key.endswith("crimerisk/level_lane.py") for key in digests)
    before = dict(digests)
    (paths.repo_root / "src" / "crimerisk" / "controls.py").write_text("# changed\n")
    assert holdout.level_holdout_input_digests(paths, year=YEAR, panel_year_end=2025, remainder=False)["src/crimerisk/controls.py"] != before["src/crimerisk/controls.py"]


def test_run_meta_records_arms_label_and_completion_as_the_last_write(tmp_path):
    out_dir = tmp_path / "run"
    config = holdout.HoldoutConfig(year=YEAR, folds=10, mask_share=0.1, seed=1, arms=("refused", "silent", "corrupted"))
    scored = holdout.score(
        pd.DataFrame(
            {"ori9": ["A", "B"], "offense": "burglary", "observed_count": [10.0, 20.0],
             "predicted_count": [11.0, 19.0], "population": [1_000.0, 30_000.0],
             "agency_estimate_source": "observed", "arm": ["refused", "silent"], "config": "l1"}
        )
    )
    holdout.write_report(
        out_dir=out_dir,
        agency_scored=scored,
        remainder_scored=pd.DataFrame(),
        config=config,
        meta={"config_label": "l1", "edition_inputs_sha256": {"configs/x.csv": "0" * 64}},
    )
    meta = json.loads((out_dir / "run_meta.json").read_text())
    assert meta["arms"] == ["refused", "silent", "corrupted"]
    assert meta["config_label"] == "l1"
    assert meta["edition_inputs_sha256"] == {"configs/x.csv": "0" * 64}
    assert meta["completed_at_utc"]
    meta_written = os.stat(out_dir / "run_meta.json").st_mtime_ns
    assert all(
        os.stat(path).st_mtime_ns <= meta_written for path in out_dir.iterdir() if path.name != "run_meta.json"
    )


def test_frame_difference_reports_rows_values_and_columns():
    production = pd.DataFrame({"j": ["a", "b"], "o": "x", "count": [1.0, 2.0], "source": ["s", "t"]})
    same = holdout.frame_difference(production.copy(), production, keys=["j", "o"])
    assert same["equivalent"] and same["max_abs_difference"] == 0.0
    fold = production.assign(count=[1.0, 2.5], source=["s", "u"])
    diff = holdout.frame_difference(fold, production, keys=["j", "o"])
    assert not diff["equivalent"]
    assert diff["numeric_differences"] == {"count": 0.5}
    assert diff["value_mismatches"] == {"source": 1}
    extra = pd.concat([production, production.iloc[[0]].assign(j="c")], ignore_index=True)
    assert holdout.frame_difference(extra, production, keys=["j", "o"])["rows_only_fold"] == 1


def test_the_equivalence_check_compares_every_stage_with_the_production_artifacts(tmp_path, monkeypatch):
    paths = RepoPaths.from_repo_root(tmp_path / "repo")
    controls_dir = paths.state_dir / "controls"
    controls_dir.mkdir(parents=True)
    year = 2025
    admission = _fold_admission().assign(year=year)
    history = pd.DataFrame({"ori9": [FOLD_ORI], "offense": "burglary", "year": [year - 1], "history_status": "admitted_complete"})
    jurisdiction_years = pd.DataFrame({"jurisdiction_id": ["j"], "offense": "burglary", "year": [year], "estimated_count": [5.0]})
    accounting = pd.DataFrame({"jurisdiction_id": ["j"], "offense": "burglary", "adjusted_count_ags_core": [5.0]})
    ledger = _fold_ledger().assign(year=year, jurisdiction_id="j", ownership_class="agency_level")
    smoothed = pd.DataFrame({"jurisdiction_id": ["j"], "offense": "burglary", "state_abbr": "XX",
                             "smoothed_count": [6.0], "state_anchor_count": [60.0], "state_anchor_source": "cde"})
    admission.to_parquet(controls_dir / f"level_lane_admission_{year}.parquet", index=False)
    history.to_parquet(controls_dir / f"reporting_history_{year}.parquet", index=False)
    jurisdiction_years.to_parquet(controls_dir / "jurisdiction_year_estimates.parquet", index=False)
    accounting.to_parquet(controls_dir / f"jurisdiction_controls_{year}.parquet", index=False)
    ledger.to_parquet(controls_dir / f"level_lane_mass_ledger_{year}.parquet", index=False)
    smoothed.to_parquet(controls_dir / f"jurisdiction_controls_smoothed_{year}.parquet", index=False)

    fold_smoothed = smoothed.assign(smoothed_count=[6.5])
    calls: dict[str, object] = {}

    def fake_pass(paths, *, preferred, year, masked_by_arm, work_dir):
        calls["masked_by_arm"] = masked_by_arm
        calls["work_dir"] = work_dir
        fold = SimpleNamespace(
            controls=accounting, jurisdiction_years=jurisdiction_years, mass_ledger=ledger,
            anchor_inputs=SimpleNamespace(mass_ledger=ledger), smoothed=fold_smoothed,
        )
        return SimpleNamespace(admitted=pd.DataFrame({"year": [year]}), admission=admission, estimates=None, controls=fold)

    def fake_history(history_frame, target_rows, *, paths, year):
        path = Path(paths.state_dir) / "controls" / f"reporting_history_{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        history_frame.to_parquet(path, index=False)
        return path

    monkeypatch.setattr(holdout, "reporting_regimes_artifact_is_current", lambda paths, config: True)
    monkeypatch.setattr(holdout, "build_preferred_panel", lambda paths, *, year, panel_year_end: (pd.DataFrame(), history))
    monkeypatch.setattr(holdout, "estimate_pass", fake_pass)
    monkeypatch.setattr(holdout, "write_reporting_history", fake_history)
    monkeypatch.setattr(holdout, "contribution_ledger", lambda inputs, *, year: inputs.mass_ledger.assign(state_abbr="XX"))
    monkeypatch.setattr(
        holdout, "load_anchor_inputs", lambda paths, *, year: SimpleNamespace(mass_ledger=pd.read_parquet(
            controls_dir / f"level_lane_mass_ledger_{year}.parquet"))
    )
    out_dir = paths.state_dir / "eval" / "equivalence"
    report = holdout.equivalence_check(paths, year=year, panel_year_end=year, out_dir=out_dir)
    assert calls["masked_by_arm"] == {}
    assert Path(calls["work_dir"]).is_relative_to(out_dir)
    assert set(report["components"]) == {
        "admission", "history", "jurisdiction_years", "accounting", "mass_ledger", "contributions",
        "state_targets", "smoothed",
    }
    assert not report["equivalent"]
    assert report["components"]["smoothed"]["numeric_differences"] == {"smoothed_count": 0.5}
    assert all(entry["equivalent"] for name, entry in report["components"].items() if name != "smoothed")
    assert (out_dir / f"equivalence_{year}.json").exists()
    # Production state is read, never written.
    assert sorted(p.name for p in controls_dir.iterdir()) == sorted([
        f"level_lane_admission_{year}.parquet", f"reporting_history_{year}.parquet",
        "jurisdiction_year_estimates.parquet", f"jurisdiction_controls_{year}.parquet",
        f"level_lane_mass_ledger_{year}.parquet", f"jurisdiction_controls_smoothed_{year}.parquet",
    ])
    (controls_dir / f"level_lane_mass_ledger_{year}.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="level_lane_mass_ledger"):
        holdout.equivalence_check(paths, year=year, panel_year_end=year, out_dir=out_dir)
