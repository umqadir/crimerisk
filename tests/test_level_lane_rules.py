"""The five reviewed level-lane rules, each exercised against the defect it exists for."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crimerisk.crime import OFFENSES_7
from crimerisk.level_lane import (
    JOINT_VECTOR_MIN_BREACHES,
    LevelLaneConfigError,
    LevelLanePolicy,
    OWN_HISTORY_MIN_CLEAN_YEARS,
    _measure_own_history_log_change_bounds,
    apply_level_lane_admission,
    level_lane_policy,
)


# --- switches --------------------------------------------------------------------


def test_switches_default_on_and_none_turns_every_rule_off(monkeypatch):
    monkeypatch.delenv("CRIMERISK_LEVEL_RULES", raising=False)
    assert level_lane_policy() == LevelLanePolicy()
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "none")
    off = level_lane_policy()
    assert not any(getattr(off, name) for name in LevelLanePolicy.field_names())


def test_switches_select_one_rule_at_a_time(monkeypatch):
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "all,joint_vector_gate=0")
    policy = level_lane_policy()
    assert policy.own_history_admission_bound
    assert not policy.joint_vector_gate
    # A bare list names exactly the rules that apply.
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "joint_vector_gate")
    only_joint = level_lane_policy()
    assert only_joint.joint_vector_gate
    assert not only_joint.own_history_admission_bound
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "nonsense=1")
    with pytest.raises(LevelLaneConfigError, match="unknown level-lane rule switch"):
        level_lane_policy()


# --- (a) own-history bound -------------------------------------------------------


def _history(counts: dict[int, float], *, ori: str = "ZZ0000000", offense: str = "burglary"):
    return pd.DataFrame(
        [
            {
                "ori9": ori,
                "offense": offense,
                "year": year,
                "level_lane_original_count": value,
                "preferred_months_reported": 12.0,
                "usable_as_observed": True,
            }
            for year, value in counts.items()
        ]
    )


def test_own_bound_is_narrower_than_the_peer_envelope_for_a_steady_agency():
    panel = _history({2018: 364, 2019: 350, 2020: 330, 2021: 242, 2022: 242, 2023: 240, 2024: 335})
    bounds = _measure_own_history_log_change_bounds(panel, target_year=2025).set_index(
        ["ori9", "offense"]
    )
    row = bounds.loc[("ZZ0000000", "burglary")]
    assert row["level_own_clean_years"] == 7
    # Cicero's 2024->2025 burglary move, 335 to 91.
    cicero_move = float(np.log(91.0 / 335.0))
    assert cicero_move < row["level_own_log_change_lower"]
    # Its ordinary year-to-year moves stay inside.
    assert float(np.log(242.0 / 330.0)) > row["level_own_log_change_lower"]


def test_own_bound_is_withheld_below_three_clean_years():
    panel = _history({2023: 300, 2024: 320})
    bounds = _measure_own_history_log_change_bounds(panel, target_year=2025)
    assert bounds["level_own_clean_years"].max() < OWN_HISTORY_MIN_CLEAN_YEARS
    assert bounds["level_own_log_change_lower"].isna().all()


def test_nibrs_transition_years_are_dropped_from_the_own_bound():
    panel = _history({2018: 954, 2019: 940, 2020: 954, 2021: 488, 2022: 470, 2023: 469, 2024: 344})
    transitions = pd.Series({"ZZ0000000": 2021}, dtype="Int64")
    without = _measure_own_history_log_change_bounds(panel, target_year=2025)
    with_guard = _measure_own_history_log_change_bounds(
        panel, target_year=2025, nibrs_transition_years=transitions
    )
    # The 2020->2021 halving is the NIBRS start, not a real collapse: leaving it in
    # widens the band enough to admit almost anything afterwards.
    assert with_guard["level_own_log_change_lower"].iloc[0] > without[
        "level_own_log_change_lower"
    ].iloc[0]
    assert with_guard["level_own_clean_years"].iloc[0] == 3


# --- admission end to end --------------------------------------------------------


def _empty_configs(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    pd.DataFrame(
        columns=[
            "case_id", "ori9", "year", "offense", "adjudication", "reason_code",
            "reviewer", "evidence_note",
        ]
    ).to_csv(configs / "level_lane_admission_registry.csv", index=False)
    pd.DataFrame(
        columns=[
            "case_id", "ori9", "year", "offense", "level1_status", "level2_status",
            "reason_code", "repair_mode", "replacement_count", "external_check_status",
            "evidence_source",
        ]
    ).to_csv(configs / "level_lane_hard_evidence.csv", index=False)
    screen = tmp_path / "analysis_scratch" / "level_lane_screen"
    screen.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["ori9", "year"]).to_csv(
        screen / "structural_zero_corroborated_all.csv", index=False
    )
    return SimpleNamespace(repo_root=tmp_path, data_dir=tmp_path / "data")


# Cicero's filed vectors, 2018 through 2025 (state/controls, v52 surface).
CICERO = {
    "burglary": {2018: 364, 2019: 300, 2020: 280, 2021: 250, 2022: 242, 2023: 240, 2024: 335, 2025: 91},
    "robbery": {2018: 127, 2019: 120, 2020: 110, 2021: 104, 2022: 117, 2023: 67, 2024: 97, 2025: 49},
    "motor_vehicle_theft": {2018: 111, 2019: 120, 2020: 150, 2021: 240, 2022: 277, 2023: 273, 2024: 305, 2025: 175},
    "aggravated_assault": {2018: 149, 2019: 150, 2020: 152, 2021: 155, 2022: 156, 2023: 125, 2024: 164, 2025: 108},
    "murder": {2018: 4, 2019: 3, 2020: 4, 2021: 3, 2022: 3, 2023: 5, 2024: 4, 2025: 0},
    "larceny": {2018: 1295, 2019: 1100, 2020: 900, 2021: 700, 2022: 647, 2023: 469, 2024: 344, 2025: 528},
    "rape": {2018: 30, 2019: 28, 2020: 26, 2021: 27, 2022: 25, 2023: 24, 2024: 26, 2025: 22},
}


def _vector_rows(series: dict[str, dict[int, float]], *, ori: str, scale: float = 1.0):
    rows = []
    for offense, by_year in series.items():
        for year, value in by_year.items():
            rows.append(
                {
                    "ori9": ori,
                    "state_fips": "17",
                    "state_abbr": "IL",
                    "year": year,
                    "offense": offense,
                    "preferred_count": float(value) * scale,
                    "preferred_source": "cius_publication_annual",
                    "preferred_months_reported": 12.0,
                    "usable_as_observed": True,
                    "current_row_is_true_partial": False,
                    "population": 80_000.0,
                }
            )
    return rows


def _peer_rows(n: int = 12):
    """Steady peers, so the measured peer envelope is wide but not degenerate."""
    rows = []
    rng = np.random.default_rng(7)
    for index in range(n):
        ori = f"IL{index:07d}"
        for offense in OFFENSES_7:
            base = {"murder": 25.0, "rape": 60.0, "robbery": 150.0,
                    "aggravated_assault": 300.0, "burglary": 400.0,
                    "larceny": 1500.0, "motor_vehicle_theft": 250.0}[offense]
            for year in range(2018, 2026):
                # One peer really does collapse, which is what keeps the peer first
                # percentile wide enough to admit Cicero today.
                factor = 0.1 if (index == 0 and year == 2025) else float(
                    rng.uniform(0.9, 1.1)
                )
                rows.append(
                    {
                        "ori9": ori, "state_fips": "17", "state_abbr": "IL", "year": year,
                        "offense": offense, "preferred_count": base * factor,
                        "preferred_source": "cius_publication_annual",
                        "preferred_months_reported": 12.0, "usable_as_observed": True,
                        "current_row_is_true_partial": False, "population": 80_000.0,
                    }
                )
    return rows


def _admit(tmp_path, monkeypatch, rules: str):
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", rules)
    paths = _empty_configs(tmp_path)
    panel = pd.DataFrame(_vector_rows(CICERO, ori="IL0161200") + _peer_rows())
    admitted = apply_level_lane_admission(panel, paths=paths, target_year=2025).panel
    return admitted[
        admitted["year"].eq(2025) & admitted["ori9"].eq("IL0161200")
    ].set_index("offense")


def test_peer_percentile_alone_admits_the_collapsed_vector(tmp_path, monkeypatch):
    current = _admit(tmp_path, monkeypatch, "none")
    assert set(current["level1_admission_status"]) == {"valid_complete_year"}


def test_own_history_bound_refuses_the_collapsed_offenses(tmp_path, monkeypatch):
    current = _admit(tmp_path, monkeypatch, "own_history_admission_bound")
    assert current.loc["burglary", "level1_admission_status"] == "coverage_defective"
    assert current.loc["burglary", "level_admission_reason"] == "own_history_bound_breach"
    # Larceny rises; nothing about it is a collapse.
    assert current.loc["larceny", "level1_admission_status"] == "valid_complete_year"


def test_joint_vector_gate_refuses_the_whole_vector(tmp_path, monkeypatch):
    current = _admit(tmp_path, monkeypatch, "all")
    breaching = [
        offense
        for offense in OFFENSES_7
        if current.loc[offense, "level1_admission_status"] == "coverage_defective"
    ]
    assert len(breaching) >= JOINT_VECTOR_MIN_BREACHES
    assert int(current["level_joint_vector_breach_count"].max()) >= JOINT_VECTOR_MIN_BREACHES
    # The gate owns the vector, so even the offense that rose is refused with it.
    assert current.loc["larceny", "level1_admission_status"] == "coverage_defective"
    assert current.loc["larceny", "preferred_count"] == 0.0


def test_a_steady_vector_is_untouched_by_either_screen(tmp_path, monkeypatch):
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "all")
    paths = _empty_configs(tmp_path)
    steady = {
        offense: {year: value for year, value in CICERO[offense].items()}
        for offense in CICERO
    }
    for offense in steady:
        steady[offense][2025] = steady[offense][2024]
    panel = pd.DataFrame(_vector_rows(steady, ori="IL0161200") + _peer_rows())
    admitted = apply_level_lane_admission(panel, paths=paths, target_year=2025).panel
    current = admitted[
        admitted["year"].eq(2025) & admitted["ori9"].eq("IL0161200")
    ]
    assert set(current["level1_admission_status"]) == {"valid_complete_year"}


# --- (d) own history versus the pooled prior --------------------------------------


def test_long_clean_history_drops_the_pooled_pseudocount(monkeypatch):
    from crimerisk.trend_fills import (
        TrendFillLookup,
        build_agency_target_estimates_from_panel,
    )

    lookup = TrendFillLookup(state_map={}, national_map={})

    rows = []
    # The refused agency: seven steady years, and a target-year row the admission gate
    # has thrown out. Its own population is known, so the pooled prior is live.
    for offense in OFFENSES_7:
        rows.append(
            {
                "ori9": "FL0160000", "state_fips": "12", "state_abbr": "FL",
                "year": 2025, "offense": offense, "preferred_count": 0.0,
                "preferred_months_reported": 12.0, "usable_as_observed": False,
                "current_row_is_true_partial": False, "population": 1_000_000.0,
                "agency_type_norm": "local_police",
                "preferred_source": "srs_return_a_annual", "reporting_regime": "srs",
                "level1_admission_status": "coverage_defective",
                "level2_semantic_status": "semantically_unusable",
                "level_admission_reason": "own_history_bound_breach",
                "level_repair_mode": "decayed_own_history_or_pooled",
                "level_policy_resolution": "repair_ladder",
            }
        )
    for year in range(2018, 2025):
        for offense in OFFENSES_7:
            rows.append(
                {
                    "ori9": "FL0160000", "state_fips": "12", "state_abbr": "FL",
                    "year": year, "offense": offense, "preferred_count": 3000.0,
                    "preferred_months_reported": 12.0, "usable_as_observed": True,
                    "current_row_is_true_partial": False, "population": 1_000_000.0,
                    "agency_type_norm": "local_police",
                    "preferred_source": "srs_return_a_annual", "reporting_regime": "srs",
                    "level1_admission_status": "valid_complete_year",
                    "level2_semantic_status": "definitionally_complete",
                    "level_admission_reason": "selected_lane_supported",
                    "level_repair_mode": "none",
                    "level_policy_resolution": "not_applicable",
                }
            )
    # A pool of much smaller-rate peers that do file the target year.
    for index in range(6):
        for offense in OFFENSES_7:
            for year in range(2018, 2026):
                rows.append(
                    {
                        "ori9": f"FL{index:07d}", "state_fips": "12", "state_abbr": "FL",
                        "year": year, "offense": offense, "preferred_count": 100.0,
                        "preferred_months_reported": 12.0, "usable_as_observed": True,
                        "current_row_is_true_partial": False, "population": 1_000_000.0,
                        "agency_type_norm": "local_police",
                        "preferred_source": "srs_return_a_annual", "reporting_regime": "srs",
                        "level1_admission_status": "valid_complete_year",
                        "level2_semantic_status": "definitionally_complete",
                        "level_admission_reason": "selected_lane_supported",
                        "level_repair_mode": "none",
                        "level_policy_resolution": "not_applicable",
                    }
                )
    panel = pd.DataFrame(rows)

    def estimate(rules: str) -> float:
        monkeypatch.setenv("CRIMERISK_LEVEL_RULES", rules)
        result = build_agency_target_estimates_from_panel(
            panel, target_year=2025, trend_fill_lookup=lookup
        )
        return float(
            result[result["ori9"].eq("FL0160000") & result["offense"].eq("burglary")][
                "estimated_count"
            ].iloc[0]
        )

    blended = estimate("none")
    own_only = estimate("prefer_own_history_over_pool=1")
    assert own_only > blended
    assert own_only == pytest.approx(3000.0, rel=1e-6)


def test_nibrs_transition_guard_drops_pre_transition_fill_references(monkeypatch):
    """(c) A pre-transition year describes a different collection, not a higher level."""
    from crimerisk.trend_fills import (
        TrendFillLookup,
        build_agency_target_estimates_from_panel,
    )

    rows = []
    for year in range(2018, 2025):
        refused = year == 2024
        for offense in OFFENSES_7:
            rows.append(
                {
                    "ori9": "ZZ0000001", "state_fips": "99", "state_abbr": "ZZ",
                    "year": year, "offense": offense,
                    # The agency converted to NIBRS in 2021 and its recorded level
                    # halved with the definition, not with the crime.
                    "preferred_count": 0.0 if refused else (1000.0 if year < 2021 else 400.0),
                    "preferred_months_reported": 12.0,
                    "usable_as_observed": not refused,
                    "current_row_is_true_partial": False,
                    "population": 50_000.0,
                    "agency_type_norm": "local_police",
                    "preferred_source": "srs_return_a_annual",
                    "reporting_regime": "srs",
                    "level_nibrs_start_year": 2021,
                    "level1_admission_status": (
                        "coverage_defective" if refused else "valid_complete_year"
                    ),
                    "level2_semantic_status": (
                        "semantically_unusable" if refused else "definitionally_complete"
                    ),
                    "level_admission_reason": "holdout",
                    "level_repair_mode": (
                        "decayed_own_history_or_pooled" if refused else "none"
                    ),
                    "level_policy_resolution": (
                        "repair_ladder" if refused else "not_applicable"
                    ),
                }
            )
    panel = pd.DataFrame(rows)
    lookup = TrendFillLookup(state_map={}, national_map={})

    def estimate(rules: str) -> float:
        monkeypatch.setenv("CRIMERISK_LEVEL_RULES", rules)
        result = build_agency_target_estimates_from_panel(
            panel, target_year=2024, trend_fill_lookup=lookup
        )
        return float(
            result[result["offense"].eq("burglary")]["estimated_count"].iloc[0]
        )

    without_guard = estimate("none")
    with_guard = estimate("nibrs_transition_reference_guard")
    assert with_guard == pytest.approx(400.0, rel=1e-6)
    assert without_guard > with_guard


def test_partial_year_below_own_history_floor_goes_to_the_ladder(tmp_path, monkeypatch):
    """(f) A token partial year is a non-report, not a lower bound."""
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "partial_year_history_floor")
    paths = _empty_configs(tmp_path)
    rows = []
    # Seven clean years around 3,000 Part 1, then four months carrying eight offenses.
    for year in range(2018, 2025):
        for offense in OFFENSES_7:
            rows.append(
                {
                    "ori9": "FL0280000", "state_fips": "12", "state_abbr": "FL",
                    "year": year, "offense": offense, "preferred_count": 430.0,
                    "preferred_source": "srs_return_a_annual",
                    "preferred_months_reported": 12.0, "usable_as_observed": True,
                    "current_row_is_true_partial": False, "population": 95_570.0,
                }
            )
    for offense in OFFENSES_7:
        rows.append(
            {
                "ori9": "FL0280000", "state_fips": "12", "state_abbr": "FL",
                "year": 2025, "offense": offense,
                "preferred_count": 2.0 if offense == "larceny" else 1.0,
                "preferred_source": "srs_return_a_annual",
                "preferred_months_reported": 4.0, "usable_as_observed": False,
                "current_row_is_true_partial": True, "population": 95_570.0,
            }
        )
    panel = pd.DataFrame(rows + _peer_rows(4))
    admitted = apply_level_lane_admission(panel, paths=paths, target_year=2025).panel
    current = admitted[
        admitted["year"].eq(2025) & admitted["ori9"].eq("FL0280000")
    ]
    assert set(current["level1_admission_status"]) == {"coverage_defective"}
    assert set(current["level_admission_reason"]) == {
        "partial_year_below_own_history_floor"
    }
    assert set(current["level_repair_mode"]) == {"decayed_own_history_or_pooled"}
    assert current["preferred_count"].eq(0.0).all()

    # A partial year that annualises near the agency's own level is still a lower bound.
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "partial_year_history_floor")
    honest = [dict(r) for r in rows]
    for row in honest:
        if row["year"] == 2025:
            row["preferred_count"] = 140.0
    admitted = apply_level_lane_admission(
        pd.DataFrame(honest + _peer_rows(4)), paths=paths, target_year=2025
    ).panel
    current = admitted[
        admitted["year"].eq(2025) & admitted["ori9"].eq("FL0280000")
    ]
    assert set(current["level1_admission_status"]) == {"valid_partial_lower_bound"}
