"""Coverage scenarios (contract: COVERAGE_SCENARIOS_CONTRACT.md).

The gate is exercised against a MINIATURE BUT REAL EDITION -- the scenario tables are produced by
`build_coverage_scenarios`'s own `_build_state_deltas` and its own column tuples, so a passing test
means the gate accepts what the producer actually emits rather than what this file thinks it emits.

Each failure test then breaks exactly one thing and asserts the gate names it: an out-of-order
ladder in each direction, a correction that moved non-imputed mass, a lane split that stops adding
up, a state delta table that no longer reconstructs, an ALL row that is not its state's rows
summed, a share that is no longer the ratio of the counts beside it, an edition that ships the
tables without declaring them, and an edition that claims block-group scenarios. A gate that cannot
fail is not a gate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from crimerisk.crime import OFFENSES_7


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"
PRODUCER_PATH = REPO_ROOT / "scripts" / "release" / "build_coverage_scenarios.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load(VALIDATOR_PATH, "validate_release_outputs")
PRODUCER = _load(PRODUCER_PATH, "build_coverage_scenarios")
YEAR = VALIDATOR.YEAR


def _scenario_frame() -> pd.DataFrame:
    """Two states, every offense, with a real imputed component in one of them."""
    rows = []
    for state, abbr in (("42", "PA"), ("09", "CT")):
        for index, offense in enumerate(OFFENSES_7):
            observed = 1000.0 + 10.0 * index
            partial = 20.0
            fill = 50.0
            municipal = 12.0 if state == "42" else 0.0
            county = 8.0 if state == "42" else 0.0
            imputed = municipal + county
            # The municipal lane is centred by 1/2 and the county lane is not; the upper surface
            # prices both at 3x, which is the ordering the ladder needs.
            central_imputed = municipal * 0.5 + county
            upper_imputed = imputed * 3.0
            non_imputed = observed + partial + fill
            rows.append(
                {
                    "year": YEAR,
                    "state_fips": state,
                    "state_abbr": abbr,
                    "jurisdiction_id": f"{state}:{offense}:muni",
                    "jurisdiction_type": "municipal",
                    "offense": offense,
                    "count_published": non_imputed + imputed,
                    "count_observed": observed,
                    "count_central": non_imputed + central_imputed,
                    "count_upper": non_imputed + upper_imputed,
                    "observed_component_count": observed,
                    "partial_component_count": partial,
                    "fill_component_count": fill,
                    "benchmark_imputed_count": imputed,
                    "benchmark_imputed_municipal": municipal,
                    "benchmark_imputed_county": county,
                    "benchmark_imputed_central": central_imputed,
                    "benchmark_imputed_upper": upper_imputed,
                    "central_correction_applied": 0.5 if municipal else 1.0,
                    "scenario_dependent_count": (non_imputed + upper_imputed) - observed,
                }
            )
    return pd.DataFrame(rows)[list(PRODUCER.SCENARIO_COLUMNS)]


def _write_edition(
    tmp_path: Path,
    *,
    scenarios: pd.DataFrame | None = None,
    deltas: pd.DataFrame | None = None,
    declared: dict | None | str = "default",
) -> Path:
    edition_dir = tmp_path / "2024A-annual"
    scenario_dir = edition_dir / PRODUCER.SCENARIOS_DIRNAME
    scenario_dir.mkdir(parents=True, exist_ok=True)
    if scenarios is not None:
        scenarios.to_parquet(
            PRODUCER.coverage_scenarios_path(scenario_dir, year=YEAR), index=False
        )
    if deltas is not None:
        deltas.to_parquet(
            PRODUCER.coverage_scenario_state_deltas_path(scenario_dir, year=YEAR), index=False
        )
    if declared == "default":
        declared = {
            "version": PRODUCER.SCENARIO_VERSION,
            "support": "jurisdiction",
            "block_group_scenarios_published": False,
            "bounds_rule_version": "v1",
        }
    payload: dict = {"edition_id": "2024A-annual"}
    if declared is not None:
        payload["coverage_scenarios"] = declared
    (edition_dir / "edition.json").write_text(json.dumps(payload))
    return edition_dir


def _edition(tmp_path: Path, scenarios: pd.DataFrame | None = None, **kwargs) -> Path:
    scenarios = _scenario_frame() if scenarios is None else scenarios
    deltas = kwargs.pop("deltas", None)
    if deltas is None:
        deltas = PRODUCER._build_state_deltas(scenarios, year=YEAR)
    return _write_edition(tmp_path, scenarios=scenarios, deltas=deltas, **kwargs)


# --- the producer's own invariants ------------------------------------------


def test_the_producer_accepts_an_ordered_ladder():
    PRODUCER._assert_scenario_invariants(_scenario_frame())


def test_the_producer_rejects_central_below_observed():
    frame = _scenario_frame()
    frame.loc[0, "count_central"] = frame.loc[0, "count_observed"] - 1.0
    with pytest.raises(SystemExit, match="observed <= central"):
        PRODUCER._assert_scenario_invariants(frame)


def test_the_producer_rejects_upper_below_central():
    frame = _scenario_frame()
    frame.loc[0, "count_upper"] = frame.loc[0, "count_central"] - 1.0
    with pytest.raises(SystemExit, match="central <= upper"):
        PRODUCER._assert_scenario_invariants(frame)


def test_the_producer_rejects_a_correction_that_moved_non_imputed_mass():
    """The claim "central is the published surface, centred" is exactly this identity."""
    frame = _scenario_frame()
    frame.loc[0, "count_central"] = frame.loc[0, "count_central"] + 5.0
    with pytest.raises(SystemExit, match="non-imputed mass is not the published surface"):
        PRODUCER._assert_scenario_invariants(frame)


def test_the_state_delta_table_carries_an_all_row_per_state():
    deltas = PRODUCER._build_state_deltas(_scenario_frame(), year=YEAR)
    all_rows = deltas[deltas["offense"].eq(PRODUCER.ALL_OFFENSES_KEY)]
    assert sorted(all_rows["state_fips"]) == ["09", "42"]
    for state in ("09", "42"):
        per_offense = deltas[deltas["state_fips"].eq(state) & ~deltas["offense"].eq("ALL")]
        row = all_rows[all_rows["state_fips"].eq(state)].iloc[0]
        assert row["count_published"] == pytest.approx(per_offense["count_published"].sum())


def test_the_shares_are_ratios_of_the_counts_beside_them():
    deltas = PRODUCER._build_state_deltas(_scenario_frame(), year=YEAR)
    expected = deltas["benchmark_imputed_count"] / deltas["count_published"]
    assert deltas["imputed_share_of_published"].to_numpy() == pytest.approx(expected.to_numpy())


# --- the release gate --------------------------------------------------------


def test_the_gate_accepts_a_real_miniature_edition(tmp_path):
    issues: list[str] = []
    summary = VALIDATOR._check_edition_coverage_scenarios(
        edition_dir=_edition(tmp_path), issues=issues
    )
    assert issues == []
    assert summary["checked"] is True
    assert summary["jurisdiction_offense_rows"] == 14
    assert summary["max_abs_non_imputed_identity_delta"] == pytest.approx(0.0, abs=1e-9)


def test_the_gate_is_silent_for_an_edition_with_no_scenarios(tmp_path):
    edition_dir = _write_edition(tmp_path, declared=None)
    issues: list[str] = []
    summary = VALIDATOR._check_edition_coverage_scenarios(edition_dir=edition_dir, issues=issues)
    assert issues == []
    assert summary["checked"] is False


def test_the_gate_fails_tables_that_are_shipped_but_not_declared(tmp_path):
    edition_dir = _edition(tmp_path, declared=None)
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(edition_dir=edition_dir, issues=issues)
    assert any("declares no coverage_scenarios block" in issue for issue in issues)


def test_the_gate_fails_a_declaration_with_no_tables(tmp_path):
    edition_dir = _write_edition(tmp_path, scenarios=None, deltas=None)
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(edition_dir=edition_dir, issues=issues)
    assert any("is absent" in issue for issue in issues)


def test_the_gate_refuses_an_edition_claiming_block_group_scenarios(tmp_path):
    edition_dir = _edition(
        tmp_path,
        declared={
            "support": "jurisdiction",
            "block_group_scenarios_published": True,
        },
    )
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(edition_dir=edition_dir, issues=issues)
    assert any("block_group_scenarios_published=false" in issue for issue in issues)


def test_the_gate_catches_an_unordered_ladder(tmp_path):
    frame = _scenario_frame()
    frame.loc[0, "count_upper"] = frame.loc[0, "count_central"] - 1.0
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(
        edition_dir=_edition(tmp_path, scenarios=frame), issues=issues
    )
    assert any("upper < central" in issue for issue in issues)


def test_the_gate_catches_a_correction_that_moved_non_imputed_mass(tmp_path):
    frame = _scenario_frame()
    frame.loc[0, "count_central"] = frame.loc[0, "count_central"] + 5.0
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(
        edition_dir=_edition(tmp_path, scenarios=frame), issues=issues
    )
    assert any("non-imputed mass is not the published surface" in issue for issue in issues)


def test_the_gate_catches_a_lane_split_that_stops_adding_up(tmp_path):
    frame = _scenario_frame()
    frame.loc[0, "benchmark_imputed_municipal"] = frame.loc[0, "benchmark_imputed_municipal"] + 3.0
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(
        edition_dir=_edition(tmp_path, scenarios=frame), issues=issues
    )
    assert any("lane split does not reconstruct" in issue for issue in issues)


def test_the_gate_catches_a_state_delta_table_that_no_longer_reconstructs(tmp_path):
    scenarios = _scenario_frame()
    deltas = PRODUCER._build_state_deltas(scenarios, year=YEAR)
    per_offense = ~deltas["offense"].eq(PRODUCER.ALL_OFFENSES_KEY)
    deltas.loc[deltas.index[per_offense][0], "count_upper"] += 7.0
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(
        edition_dir=_edition(tmp_path, scenarios=scenarios, deltas=deltas), issues=issues
    )
    assert any("does not reconstruct from the jurisdiction table" in issue for issue in issues)


def test_the_gate_catches_an_all_row_that_is_not_its_states_rows_summed(tmp_path):
    scenarios = _scenario_frame()
    deltas = PRODUCER._build_state_deltas(scenarios, year=YEAR)
    all_rows = deltas["offense"].eq(PRODUCER.ALL_OFFENSES_KEY)
    deltas.loc[deltas.index[all_rows][0], "count_observed"] += 9.0
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(
        edition_dir=_edition(tmp_path, scenarios=scenarios, deltas=deltas), issues=issues
    )
    assert any("ALL-offense rows are not the state's offense rows summed" in issue for issue in issues)


def test_the_gate_catches_a_share_that_is_not_the_ratio_beside_it(tmp_path):
    scenarios = _scenario_frame()
    deltas = PRODUCER._build_state_deltas(scenarios, year=YEAR)
    deltas.loc[0, "imputed_share_of_published"] = 0.75
    issues: list[str] = []
    VALIDATOR._check_edition_coverage_scenarios(
        edition_dir=_edition(tmp_path, scenarios=scenarios, deltas=deltas), issues=issues
    )
    assert any("imputed_share_of_published is not the ratio" in issue for issue in issues)


def test_the_validator_transcribes_the_producers_column_contract():
    # The mirror is a transcription, not an import. It must be able to disagree with the producer,
    # and it must not disagree by accident.
    assert set(VALIDATOR.COVERAGE_SCENARIO_COLUMNS) <= set(PRODUCER.SCENARIO_COLUMNS)
    assert set(VALIDATOR.COVERAGE_SCENARIO_STATE_DELTA_COLUMNS) <= set(PRODUCER.STATE_DELTA_COLUMNS)
    assert VALIDATOR.COVERAGE_SCENARIOS_DIRNAME == PRODUCER.SCENARIOS_DIRNAME
    assert VALIDATOR.COVERAGE_SCENARIO_ALL_KEY == PRODUCER.ALL_OFFENSES_KEY
