"""Ladder-dropped agencies in the level holdout are scored by the production fallback.

An agency the repair ladder declines to estimate is not absent from the release: its
jurisdiction is sized without it by the controls stage (pooled silent-unit imputation) and
then smoothed. The holdout scores the dropped agency as that fold control. These tests pin
three things: the fold control is built from the masked panel the estimator saw, a panel
with no population column (the production panel's shape) still gives every dropped agency
with footprint population a positive prediction, and the fold's controls stage cannot write
into production state.
"""

from __future__ import annotations

import importlib.util
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
from crimerisk.smoothed_controls import build_smoothed_controls  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "level_holdout_under_test", REPO_ROOT / "scripts" / "eval" / "level_holdout.py"
)
holdout = importlib.util.module_from_spec(_SPEC)
sys.modules["level_holdout_under_test"] = holdout
_SPEC.loader.exec_module(holdout)

SOLE = "XX0000001"  # the only agency of a municipal jurisdiction
SHARED = "XX0000002"  # shares a jurisdiction with a reporter, custom footprint population
REPORTER = "XX0000003"
OVERLAY = "XX0000004"  # statewide overlap layer: no population anywhere
J_SOLE = "99:municipal:place:1"
J_SHARED = "99:municipal:place:2"
J_OVERLAY = "99:statewide_overlap_layer"
POPULATION = {J_SOLE: 1_000.0, J_SHARED: 10_000.0, J_OVERLAY: 0.0}


def _panel() -> pd.DataFrame:
    """Production-shaped: offense-centric, and no population column."""
    rows = []
    for ori in (SOLE, SHARED, REPORTER, OVERLAY):
        for year in (2021, 2022, 2023, 2024):
            rows.append(
                {
                    "ori9": ori,
                    "year": year,
                    "offense": "burglary",
                    "preferred_count": 40.0,
                    "preferred_months_reported": 12.0,
                    "usable_as_observed": True,
                    "current_row_is_true_partial": False,
                    "level1_admission_status": "valid_complete_year",
                    "state_fips": "99",
                }
            )
    return pd.DataFrame(rows)


def _accounting_row(jurisdiction_id: str, count: float) -> dict:
    return {
        "jurisdiction_id": jurisdiction_id,
        "jurisdiction_type": "municipal" if "municipal" in jurisdiction_id else "statewide_overlap_layer",
        "jurisdiction_name": jurisdiction_id,
        "state_fips": "99",
        "state_abbr": "XX",
        "geo_type": "place",
        "geoid": jurisdiction_id.split(":")[-1],
        "offense": "burglary",
        "bucket_population": POPULATION[jurisdiction_id],
        "pop_band": "under_25k",
        "adjusted_count_ags_core": count,
    }


def _history_row(jurisdiction_id: str, year: int, count: float) -> dict:
    return {
        "jurisdiction_id": jurisdiction_id,
        "offense": "burglary",
        "year": year,
        "reported_count_preferred": count,
        "estimated_count": count,
        "mean_months_reported_preferred": 12.0,
        "observation_weight_preferred": 1.0,
        "fill_component_count": 0.0,
        "partial_component_count": 0.0,
        "full_year_admitted_count": count,
        "full_year_admitted_share": 1.0,
        "bucket_population": POPULATION[jurisdiction_id],
    }


def _mass_ledger(estimates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ori9": estimates["ori9"].astype(str).to_numpy(),
            "year": 2024,
            "offense": estimates["offense"].astype(str).to_numpy(),
            "final_control_mass": 40.0,
        }
    )


def _write_observations(paths: RepoPaths) -> None:
    observations = paths.state_dir / "observations"
    observations.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ori9": [SOLE, SHARED, REPORTER, OVERLAY],
            "year": 2024,
            "source": "srs_return_a_annual",
            "offense": "burglary",
            "count": 40.0,
            "months_reported": 12.0,
        }
    ).to_parquet(observations / "agency_year_observations.parquet", index=False)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Everything upstream of the smoothing is stubbed; the smoothing itself is real.

    The stubbed controls stage sizes each jurisdiction from the agencies the estimator
    returned, as the real one does: a jurisdiction whose agencies were all dropped has a
    zero accounting count for the target year and its reported history before it.
    """
    captured: dict[str, object] = {}

    def fake_estimates(*, paths, year, agency_panel):
        current = agency_panel[agency_panel["year"].eq(year)]
        return pd.DataFrame(
            {
                "ori9": current["ori9"],
                "offense": current["offense"],
                "estimated_count": current["preferred_count"],
                "agency_estimate_source": "observed",
            }
        )

    def fake_bundle(*, paths, config, stage1):
        captured["stage1_panel"] = stage1.agency_panel
        captured["paths"] = paths
        reported = set(stage1.agency_estimates["ori9"])
        accounting = pd.DataFrame(
            [
                _accounting_row(J_SOLE, 40.0 if SOLE in reported else 0.0),
                _accounting_row(J_SHARED, 40.0 * len(reported & {SHARED, REPORTER})),
                _accounting_row(J_OVERLAY, 40.0 if OVERLAY in reported else 0.0),
            ]
        )
        history = pd.DataFrame(
            [
                _history_row(J_SOLE, year, 40.0) for year in (2021, 2022, 2023)
            ]
            + [_history_row(J_SHARED, year, 80.0) for year in (2021, 2022, 2023)]
        )
        imputation = SimpleNamespace(mass_ledger=_mass_ledger(stage1.agency_estimates))
        return accounting, None, history, None, imputation, stage1.agency_estimates

    def real_smoothing(*, paths, config, accounting, panel, anchor_inputs):
        captured["anchor_inputs"] = anchor_inputs
        return build_smoothed_controls(accounting=accounting, panel=panel, config=config), {}

    def fake_admit(paths, *, panel, year):
        return panel, panel[panel["year"].eq(year)].assign(level_lane_hard_case_id=pd.NA)

    monkeypatch.setattr(holdout, "admit", fake_admit)
    monkeypatch.setattr(holdout, "build_agency_allocation_target_estimates", fake_estimates)
    monkeypatch.setattr(holdout, "build_controls_bundle", fake_bundle)
    monkeypatch.setattr(holdout, "build_smoothed_controls_bundle", real_smoothing)
    monkeypatch.setattr(holdout, "resolve_ori_succession", lambda **kwargs: (pd.DataFrame(), {}))
    monkeypatch.setattr(holdout, "load_agency_jurisdiction_crosswalk", lambda paths: pd.DataFrame())
    monkeypatch.setattr(
        holdout,
        "build_agency_target_panel_slice",
        lambda *, agency_panel, target_year: agency_panel[agency_panel["year"].eq(target_year)],
    )
    paths = RepoPaths.from_repo_root(tmp_path / "repo")
    (paths.state_dir / "reference").mkdir(parents=True)
    _write_observations(paths)
    fallback = holdout.FallbackInputs(
        crosswalk=pd.DataFrame(
            {
                "ori9": [SOLE, SHARED, REPORTER, OVERLAY],
                "jurisdiction_id": [J_SOLE, J_SHARED, J_SHARED, J_OVERLAY],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        footprint_population=pd.DataFrame({"ori9": [SHARED], "_covered_population": [2_000.0]}),
        work_dir=tmp_path / "work",
    )
    return paths, fallback, captured


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ori9": ori,
                "offense": "burglary",
                "state_abbr": "XX",
                "state_fips": "99",
                "observed_count": 40.0,
                # The truth frame's population comes from the agency master; the overlap
                # agency has none, the shared one has none of its own (custom footprint).
                "population": {SOLE: 1_000.0, SHARED: 0.0, REPORTER: 8_000.0, OVERLAY: 0.0}[ori],
            }
            for ori in (SOLE, SHARED, REPORTER, OVERLAY)
        ]
    )


def _run(stubbed, masked: tuple[str, ...]) -> pd.DataFrame:
    paths, fallback, _captured = stubbed
    config = holdout.HoldoutConfig(year=2024, folds=1, mask_share=1.0, seed=1, arms=("silent",))
    panel = _panel()
    assert "population" not in panel.columns
    return holdout.run_agency_holdout_combined(
        paths=paths,
        panel=panel,
        preferred=panel,
        config=config,
        truth=_truth(),
        eligible_oris=list(masked),
        label="l1",
        fallback=fallback,
    )


def test_no_dropped_agency_with_footprint_population_scores_zero(stubbed) -> None:
    scored = _run(stubbed, (SOLE, SHARED, OVERLAY))
    dropped = scored[scored["dropped_by_ladder"]].set_index("ori9")
    assert set(dropped.index) == {SOLE, SHARED, OVERLAY}
    footprint = {SOLE: 1_000.0, SHARED: 2_000.0, OVERLAY: 0.0}
    for ori, population in footprint.items():
        if population > 0.0:
            assert dropped.loc[ori, "predicted_count"] > 0.0, ori
    # A statewide overlap layer has no population, so production gives it nothing.
    assert dropped.loc[OVERLAY, "predicted_count"] == 0.0
    assert dropped.loc[SOLE, "fallback_jurisdiction_ids"] == J_SOLE


def test_a_sole_agency_gets_its_whole_jurisdiction_and_a_shared_one_its_footprint_share(stubbed) -> None:
    scored = _run(stubbed, (SOLE, SHARED))
    dropped = scored[scored["dropped_by_ladder"]].set_index("ori9")["predicted_count"]
    paths, fallback, captured = stubbed
    # Reproduce the fold's smoothed controls from what the stage was handed.
    smoothed = holdout.fold_fallback_controls(
        paths,
        panel=captured["stage1_panel"],
        estimates=pd.DataFrame({"ori9": [REPORTER], "offense": ["burglary"]}),
        admission=captured["anchor_inputs"].admission,
        year=2024,
        work_dir=fallback.work_dir / "check",
    ).set_index("jurisdiction_id")["smoothed_count"]
    assert dropped[SOLE] == pytest.approx(smoothed[J_SOLE])
    assert dropped[SHARED] == pytest.approx(smoothed[J_SHARED] * 2_000.0 / 10_000.0)


def test_the_fold_control_is_built_from_the_masked_panel(stubbed) -> None:
    _run(stubbed, (SOLE,))
    _paths, _fallback, captured = stubbed
    stage_panel = captured["stage1_panel"]
    target = stage_panel[stage_panel["year"].eq(2024)]
    assert SOLE not in set(target["ori9"])
    assert SOLE in set(stage_panel.loc[stage_panel["year"].lt(2024), "ori9"])


def test_the_fold_controls_stage_writes_under_the_run_not_into_production(tmp_path) -> None:
    paths = RepoPaths.from_repo_root(tmp_path / "repo")
    (paths.state_dir / "controls").mkdir(parents=True)
    (paths.state_dir / "reference").mkdir(parents=True)
    production = paths.state_dir / "controls" / "benchmark_uncertainty_2024.csv"
    production.write_text("edition\n")
    (paths.state_dir / "reference" / "agency_master.parquet").write_text("master")

    (paths.state_dir / "eval").mkdir()
    isolated = holdout.isolated_state_paths(paths, paths.state_dir / "eval" / "level_v3" / "work")
    (isolated.state_dir / "controls" / "benchmark_uncertainty_2024.csv").write_text("fold\n")
    assert production.read_text() == "edition\n"
    assert not (isolated.state_dir / "eval").exists()
    assert (isolated.state_dir / "reference" / "agency_master.parquet").read_text() == "master"
    assert isolated.cache_dir == isolated.state_dir / "cache"
    assert isolated.review_analysis_dir == isolated.state_dir / "review" / "analysis"


def test_the_fold_drops_its_read_through_links_after_the_controls_stage(stubbed) -> None:
    _run(stubbed, (SOLE,))
    _paths, fallback, _captured = stubbed
    state = fallback.work_dir / "l1" / "state"
    assert (state / "controls").is_dir()
    assert not any(entry.is_symlink() for entry in state.iterdir())


def test_protected_agencies_include_the_rulings_that_name_them_in_an_oris_column(tmp_path) -> None:
    paths = RepoPaths.from_repo_root(tmp_path)
    configs = tmp_path / "configs" / "stage1_adjudications"
    configs.mkdir(parents=True)
    (tmp_path / "configs" / "level_lane_admission_registry.csv").write_text(
        "case_id,ori9,year\nx,AA0000001,2024\ny,AA0000009,2023\n"
    )
    (tmp_path / "configs" / "level_lane_hard_evidence.csv").write_text("case_id,ori9,year\n")
    (configs / "ori_successions.csv").write_text(
        "case_id,superseded_ori,successor_ori\ns,AA0000002,AA0000003\n"
    )
    (configs / "twins_adjudicated.csv").write_text(
        "case_id,oris,canonical_ori\nt,AA0000004;AA0000005,AA0000004\n"
    )
    (configs / "zero_missing_adjudicated.csv").write_text(
        "case_id,target_year,oris\nz,2024,AA0000006\nw,2023,AA0000010\n"
    )
    (configs / "token_reporters_adjudicated.csv").write_text(
        "case_id,target_year,oris\nk,2024,AA0000007\n"
    )
    assert holdout.protected_oris(paths, year=2024) == {
        "AA0000001", "AA0000002", "AA0000003", "AA0000004", "AA0000005", "AA0000006",
        "AA0000007",
    }
    (configs / "token_reporters_adjudicated.csv").unlink()
    with pytest.raises(FileNotFoundError, match="token_reporters_adjudicated"):
        holdout.protected_oris(paths, year=2024)

