"""The level-lane holdout is a release metric bound to the edition's inputs, outputs and plan."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

from crimerisk.paths import RepoPaths

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"
_SPEC = importlib.util.spec_from_file_location("validate_release_outputs_level_holdout", SCRIPT)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)
_HOLDOUT_SPEC = importlib.util.spec_from_file_location(
    "level_holdout_release_metric_under_test", ROOT / "scripts" / "eval" / "level_holdout.py"
)
holdout = importlib.util.module_from_spec(_HOLDOUT_SPEC)
sys.modules["level_holdout_release_metric_under_test"] = holdout
_HOLDOUT_SPEC.loader.exec_module(holdout)

YEAR, PANEL_YEAR_END = 2024, 2025
OBSERVATIONS = "state/observations/agency_year_observations.parquet"
EDITION = {
    OBSERVATIONS: b"edition panel",
    "configs/service_relationships.csv": b"provider,beneficiary\n",
    "configs/level_lane_admission_registry.csv": b"case_id\n",
    "src/crimerisk/level_lane.py": b"# estimator\n",
    "src/crimerisk/controls.py": b"# controls\n",
    "scripts/eval/level_holdout.py": b"# holdout\n",
}
ARMS = ("corrupted", "refused", "silent")
CONFIG = holdout.HoldoutConfig(year=YEAR, folds=3, mask_share=0.34, seed=7, arms=ARMS)
ELIGIBLE = [f"XX{index:07d}" for index in range(12)]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _edition(root: Path) -> None:
    for rel, data in EDITION.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _digests(root: Path) -> dict[str, str]:
    """What the harness records for this edition, keyed as the validator keys it."""
    digests = holdout.level_holdout_input_digests(
        RepoPaths.from_repo_root(root), year=YEAR, panel_year_end=PANEL_YEAR_END, remainder=False
    )
    # The running script lives outside the fixture repo; the fixture's own copy stands for it.
    return {key: value for key, value in digests.items() if not Path(key).is_absolute()}


def _cases(plan: dict[str, dict[str, list[str]]], *, predicted: float = 9.0) -> pd.DataFrame:
    rows = [
        {
            "ori9": ori, "offense": offense, "arm": arm, "fold": int(fold), "config": "production_default",
            "observed_count": 10.0, "predicted_count": predicted, "population": 5_000.0,
            "agency_estimate_source": "decayed_own_history",
        }
        for arm, offenses in holdout.planned_offenses(ARMS).items()
        for fold, oris in plan[arm].items()
        for ori in oris
        for offense in offenses
    ]
    return holdout.score(pd.DataFrame(rows))


def _run(root: Path, name: str, *, cases: pd.DataFrame | None = None, **meta_overrides) -> Path:
    """A run written by the harness's own report writer, with the run_meta main() records."""
    plan = holdout.planned_masks(ELIGIBLE, config=CONFIG, separate_arms=False)
    out_dir = root / "state" / "eval" / name
    meta = {
        "config_label": "production_default",
        "append": False,
        "panel_year_end": PANEL_YEAR_END,
        "remainder_scored": False,
        "edition_inputs_sha256": _digests(root),
        "planned_masks": plan,
        "planned_offenses": holdout.planned_offenses(ARMS),
    }
    meta.update(meta_overrides)
    holdout.write_report(
        out_dir=out_dir,
        agency_scored=_cases(plan) if cases is None else cases,
        remainder_scored=pd.DataFrame(),
        config=CONFIG,
        meta=meta,
    )
    return out_dir


def _rewrite_meta(run: Path, **changes) -> None:
    meta = json.loads((run / "run_meta.json").read_text())
    meta.update(changes)
    (run / "run_meta.json").write_text(json.dumps(meta))


def _check() -> tuple[list[str], dict]:
    issues: list[str] = []
    summary = validator._check_level_holdout(issues=issues)
    return issues, summary


def test_the_plan_masks_every_arm():
    plan = holdout.planned_masks(ELIGIBLE, config=CONFIG, separate_arms=False)
    assert all(any(folds.values()) for folds in plan.values())
    masked = [ori for folds in plan.values() for oris in folds.values() for ori in oris]
    assert len(masked) == len(set(masked))


def test_no_run_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    issues, summary = _check()
    assert summary["run_dir"] is None
    assert issues and "no level-lane holdout" in issues[0]


def test_a_complete_run_on_the_current_edition_binds(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    run = _run(tmp_path, "level_v9")
    meta = json.loads((run / "run_meta.json").read_text())
    assert set(meta["outputs_sha256"]) == {
        "results.csv", "results.md", "agency_cases.parquet", "agency_cases_production_default.parquet"
    }
    issues, summary = _check()
    assert issues == []
    assert summary["run_dir"] == "state/eval/level_v9"
    assert sorted(row["arm"] for row in summary["overall"]) == sorted(ARMS)


def test_the_validators_required_set_is_what_the_harness_records(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    required = validator._level_holdout_required_inputs(
        year=YEAR, panel_year_end=PANEL_YEAR_END, remainder=False
    )
    assert required == set(_digests(tmp_path))
    assert set(EDITION) <= required


def test_the_review_probe_fails(tmp_path, monkeypatch):
    """One hash of an unrelated text file and one agency overall row used to pass."""
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    (tmp_path / "notes.txt").write_bytes(b"unrelated")
    probe = tmp_path / "state" / "eval" / "level_probe"
    probe.mkdir(parents=True)
    (probe / "run_meta.json").write_text(
        json.dumps({"edition_inputs_sha256": {"notes.txt": _sha(b"unrelated")}})
    )
    pd.DataFrame(
        {"lane": "agency", "stratum": "overall", "arm": ["silent"], "total_ratio": 0.99,
         "mean_poisson_deviance": 20.0, "median_abs_log_ratio": 0.3, "share_within_25pct": 0.45}
    ).to_csv(probe / "results.csv", index=False)
    issues, summary = _check()
    assert summary["run_dir"] is None
    assert len(issues) == 1 and "level_probe" in issues[0]
    reasons = summary["unbound"]["level_probe"]
    assert any(r.startswith("(b) run_meta lacks config.year") for r in reasons)
    assert any(r.startswith("(c) run_meta arms omit") for r in reasons)
    assert any("no agency overall row for arm 'corrupted'" in r for r in reasons)
    assert any(r.startswith("(d)") for r in reasons)
    assert "(e) run_meta records no output digests" in reasons
    assert "(f) run_meta records no planned masks" in reasons


def test_each_input_condition_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    digests = _digests(tmp_path)
    _rewrite_meta(_run(tmp_path, "level_a"), edition_inputs_sha256=dict(digests, **{OBSERVATIONS: _sha(b"older")}))
    partial = {k: v for k, v in digests.items() if k != "src/crimerisk/controls.py"}
    _rewrite_meta(_run(tmp_path, "level_b"), edition_inputs_sha256=partial)
    _rewrite_meta(_run(tmp_path, "level_d"), completed_at_utc=None)
    issues, summary = _check()
    assert summary["run_dir"] is None and issues
    unbound = summary["unbound"]
    assert [r[:3] for r in unbound["level_a"]] == ["(a)"] and OBSERVATIONS in unbound["level_a"][0]
    assert [r[:3] for r in unbound["level_b"]] == ["(b)"] and "src/crimerisk/controls.py" in unbound["level_b"][0]
    assert unbound["level_d"] == ["(d) run_meta has no completed_at_utc"]


def test_non_finite_overall_metrics_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    plan = holdout.planned_masks(ELIGIBLE, config=CONFIG, separate_arms=False)
    cases = _cases(plan)
    cases["observed_count"] = 0.0  # every total_ratio is 0/0
    _run(tmp_path, "level_c", cases=cases)
    _issues, summary = _check()
    assert summary["unbound"]["level_c"] == [
        f"(c) agency overall metrics for arm {arm!r} are not all finite" for arm in sorted(ARMS)
    ]


def test_a_code_change_unbinds_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    _run(tmp_path, "level_v9")
    (tmp_path / "src" / "crimerisk" / "level_lane.py").write_bytes(b"# estimator, edited\n")
    issues, summary = _check()
    assert summary["run_dir"] is None
    assert "(a)" in issues[0] and "src/crimerisk/level_lane.py" in issues[0]
    # A module added after the run is required too, nested packages included.
    (tmp_path / "src" / "crimerisk" / "level_lane.py").write_bytes(EDITION["src/crimerisk/level_lane.py"])
    (tmp_path / "src" / "crimerisk" / "eval").mkdir()
    (tmp_path / "src" / "crimerisk" / "eval" / "new_module.py").write_bytes(b"# new\n")
    issues, summary = _check()
    assert "(b)" in issues[0] and "src/crimerisk/eval/new_module.py" in issues[0]


def test_a_changed_case_file_or_result_unbinds_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    run = _run(tmp_path, "level_v9")
    cases = pd.read_parquet(run / "agency_cases_production_default.parquet")
    cases.loc[0, "predicted_count"] = 10.0
    cases.to_parquet(run / "agency_cases_production_default.parquet", index=False)
    _issues, summary = _check()
    assert summary["unbound"]["level_v9"] == [
        "(e) 1 output(s) differ from the digests run_meta recorded: agency_cases_production_default.parquet"
    ]
    run_b = _run(tmp_path, "level_b")
    results = pd.read_csv(run_b / "results.csv")
    results.loc[0, "share_within_25pct"] = 0.99
    results.to_csv(run_b / "results.csv", index=False)
    _issues, summary = _check()
    assert summary["unbound"]["level_b"] == [
        "(e) 1 output(s) differ from the digests run_meta recorded: results.csv"
    ]


def test_an_unscored_planned_case_unbinds_the_run(tmp_path, monkeypatch):
    """Finite overall rows for every arm, one planned agency-offense never scored."""
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    plan = holdout.planned_masks(ELIGIBLE, config=CONFIG, separate_arms=False)
    cases = _cases(plan)
    dropped = cases.iloc[0]
    _run(tmp_path, "level_missing", cases=cases.iloc[1:].reset_index(drop=True))
    nan_cases = cases.copy()
    nan_cases.loc[0, "predicted_count"] = float("nan")
    _run(tmp_path, "level_nan", cases=nan_cases)
    extra = pd.concat([cases, cases.iloc[[0]].assign(ori9="YY0000001")], ignore_index=True)
    _run(tmp_path, "level_extra", cases=extra)
    _issues, summary = _check()
    unbound = summary["unbound"]
    key = f"{dropped['arm']}/{dropped['fold']}/{dropped['ori9']}/{dropped['offense']}"
    for name in ("level_missing", "level_nan"):
        assert len(unbound[name]) == 1, unbound[name]
        assert unbound[name][0].startswith(f"(f) 1 of {len(cases)} planned case(s) were not scored")
        assert key in unbound[name][0]
    assert unbound["level_extra"] == [
        f"(f) 1 scored case(s) are outside the plan: {dropped['arm']}/{dropped['fold']}/YY0000001/{dropped['offense']}"
    ]


def test_a_plan_without_an_arm_or_offenses_unbinds_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    _edition(tmp_path)
    plan = holdout.planned_masks(ELIGIBLE, config=CONFIG, separate_arms=False)
    run = _run(tmp_path, "level_v9")
    _rewrite_meta(
        run,
        planned_masks={**plan, "silent": {}},
        planned_offenses={**holdout.planned_offenses(ARMS), "refused": ["murder"]},
    )
    _issues, summary = _check()
    reasons = summary["unbound"]["level_v9"]
    assert "(f) the plan masks no agency in arm 'silent'" in reasons
    assert "(f) arm 'refused' plans 1 of the seven offenses" in reasons
