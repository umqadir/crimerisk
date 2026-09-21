from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from crimerisk.crime import OFFENSES_7
from crimerisk.controls import assert_level_lane_mass_ledger, attach_level_lane_disclosures
from crimerisk.level_lane import (
    LevelLaneConfigError,
    apply_level_lane_admission,
    classify_controlled_corruption,
    load_admission_registry,
)


@pytest.fixture
def clean_vector() -> pd.Series:
    return pd.Series(
        {
            "murder": 12.0,
            "rape": 80.0,
            "robbery": 240.0,
            "aggravated_assault": 900.0,
            "burglary": 700.0,
            "larceny": 4200.0,
            "motor_vehicle_theft": 650.0,
        }
    )


@pytest.mark.parametrize("defect_class,fill", [("all_zero", 0.0), ("token_one", 1.0)])
def test_zero_and_token_corruption_is_reviewed_not_auto_repaired(clean_vector, defect_class, fill):
    result = classify_controlled_corruption(
        clean_vector, pd.Series(fill, index=OFFENSES_7), defect_class=defect_class
    )
    assert set(result["level1_admission_status"]) == {"unresolved_review"}
    assert set(result["action"]) == {"hold_unchanged"}
    assert result["repaired_count"].equals(result["corrupted_count"])


def test_mislabeled_partial_preserves_observed_lower_bound_on_common_basis(clean_vector):
    partial = clean_vector / 4.0
    result = classify_controlled_corruption(
        clean_vector,
        partial,
        defect_class="mislabeled_partial",
        measured_coverage_months=3,
    )
    assert set(result["level1_admission_status"]) == {"valid_partial_lower_bound"}
    assert set(result["action"]) == {"partial_missing_increment"}
    assert (result["repaired_count"] >= result["corrupted_count"]).all()
    assert (result["repaired_count"] == partial.reindex(OFFENSES_7).to_numpy() * 4.0).all()


@pytest.mark.parametrize(
    "defect_class,affected,expected_level2",
    [
        ("selective_channel_omission", {"burglary", "larceny", "motor_vehicle_theft"}, "channel_omitted"),
        ("chicago_category_omission", {"aggravated_assault"}, "wrong_category_mapping"),
    ],
)
def test_offense_channel_repairs_are_offense_specific(
    clean_vector, defect_class, affected, expected_level2
):
    corrupted = clean_vector.copy()
    corrupted.loc[list(affected)] = 0.0
    result = classify_controlled_corruption(
        clean_vector, corrupted, defect_class=defect_class, affected_offenses=affected
    ).set_index("offense")
    assert set(result.loc[list(affected), "level2_semantic_status"]) == {expected_level2}
    assert (result.loc[list(affected), "repaired_count"] == clean_vector.loc[list(affected)]).all()
    unaffected = sorted(set(OFFENSES_7) - affected)
    assert (result.loc[unaffected, "repaired_count"] == corrupted.loc[unaffected]).all()
    assert set(result.loc[unaffected, "action"]) == {"accept_unchanged"}


def test_documented_swap_reclassifies_and_conserves_pair(clean_vector):
    corrupted = clean_vector.copy()
    corrupted["murder"], corrupted["rape"] = clean_vector["rape"], clean_vector["murder"]
    result = classify_controlled_corruption(
        clean_vector,
        corrupted,
        defect_class="offense_swap",
        affected_offenses={"murder", "rape"},
    ).set_index("offense")
    pair = result.loc[["murder", "rape"]]
    assert set(pair["action"]) == {"swap_reclassification"}
    assert pair["repaired_count"].sum() == pair["corrupted_count"].sum()
    assert (pair["repaired_count"] == clean_vector.loc[["murder", "rape"]]).all()


@pytest.mark.parametrize("defect_class", ["duplicate_identity", "stale_source_precedence"])
def test_identity_and_precedence_failures_use_verified_source(clean_vector, defect_class):
    corrupted = clean_vector * 2.0 if defect_class == "duplicate_identity" else clean_vector * 0.4
    result = classify_controlled_corruption(clean_vector, corrupted, defect_class=defect_class)
    assert set(result["level1_admission_status"]) == {"source_identity_failure"}
    assert set(result["action"]) == {"source_supported_replacement"}
    assert (result.set_index("offense")["repaired_count"] == clean_vector).all()


def test_uniform_reduction_is_declared_undetectable_without_external_evidence(clean_vector):
    corrupted = clean_vector * 0.55
    internal = classify_controlled_corruption(clean_vector, corrupted, defect_class="uniform_factor")
    assert set(internal["detectability"]) == {"declared_internally_undetectable"}
    assert set(internal["action"]) == {"accept_unchanged"}
    external = classify_controlled_corruption(
        clean_vector, corrupted, defect_class="uniform_factor", external_comparator=True
    )
    assert set(external["level1_admission_status"]) == {"unresolved_review"}
    assert set(external["action"]) == {"hold_unchanged"}


def test_single_offense_collapse_does_not_change_neighbors(clean_vector):
    corrupted = clean_vector.copy()
    corrupted["larceny"] = 2.0
    result = classify_controlled_corruption(
        clean_vector,
        corrupted,
        defect_class="single_offense_collapse",
        affected_offenses={"larceny"},
    ).set_index("offense")
    assert result.loc["larceny", "action"] == "hold_unchanged"
    unaffected = [offense for offense in OFFENSES_7 if offense != "larceny"]
    assert set(result.loc[unaffected, "action"]) == {"accept_unchanged"}
    assert (result.loc[unaffected, "repaired_count"] == clean_vector.loc[unaffected]).all()


@pytest.mark.parametrize("defect_class,factor", [("genuine_spike", 5.0), ("genuine_decline", 0.15)])
def test_genuine_extreme_is_held_unchanged_not_repaired(clean_vector, defect_class, factor):
    corrupted = clean_vector * factor
    result = classify_controlled_corruption(clean_vector, corrupted, defect_class=defect_class)
    assert set(result["action"]) == {"hold_unchanged"}
    assert (result.set_index("offense")["repaired_count"] == corrupted).all()


def _registry_paths(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    return SimpleNamespace(repo_root=tmp_path), configs


def test_admission_registry_is_required_and_fail_closed(tmp_path):
    paths, configs = _registry_paths(tmp_path)
    with pytest.raises(LevelLaneConfigError, match="registry is missing"):
        load_admission_registry(paths)
    pd.DataFrame(
        [{
            "case_id": "bad", "ori9": "ZZ0000000", "year": 2024, "offense": "larceny",
            "adjudication": "auto_upward_repair", "reason_code": "bad", "reviewer": "test",
            "evidence_note": "invalid action",
        }]
    ).to_csv(configs / "level_lane_admission_registry.csv", index=False)
    with pytest.raises(LevelLaneConfigError, match="unknown adjudications"):
        load_admission_registry(paths)


def _spike_with_peer_rows(clean_vector: pd.Series) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year, factor in ((2023, 1.0), (2024, 5.0)):
        for offense, count in clean_vector.items():
            rows.append({
                "ori9": "ZZ0000000", "state_fips": "99", "state_abbr": "ZZ", "year": year,
                "offense": offense, "preferred_count": count * factor,
                "preferred_source": "state_publication_annual", "preferred_months_reported": 12.0,
                "usable_as_observed": True, "current_row_is_true_partial": False,
            })
    peer_vector = clean_vector.copy()
    peer_vector["murder"] = 20.0
    for year in (2023, 2024):
        for offense, count in peer_vector.items():
            rows.append({
                "ori9": "ZZ0000001", "state_fips": "99", "state_abbr": "ZZ", "year": year,
                "offense": offense, "preferred_count": count,
                "preferred_source": "state_publication_annual", "preferred_months_reported": 12.0,
                "usable_as_observed": True, "current_row_is_true_partial": False,
            })
    return rows


def test_empty_registry_resolves_soft_spike_per_offense(tmp_path, clean_vector):
    paths, configs = _registry_paths(tmp_path)
    pd.DataFrame(columns=[
        "case_id", "ori9", "year", "offense", "adjudication", "reason_code", "reviewer", "evidence_note",
    ]).to_csv(configs / "level_lane_admission_registry.csv", index=False)
    pd.DataFrame(columns=[
        "case_id", "ori9", "year", "offense", "level1_status", "level2_status", "reason_code",
        "repair_mode", "replacement_count", "external_check_status", "evidence_source",
    ]).to_csv(configs / "level_lane_hard_evidence.csv", index=False)
    structural = tmp_path / "analysis_scratch" / "level_lane_screen"
    structural.mkdir(parents=True)
    pd.DataFrame(columns=["ori9", "year"]).to_csv(
        structural / "structural_zero_corroborated_all.csv", index=False
    )
    rows = _spike_with_peer_rows(clean_vector)
    admitted = apply_level_lane_admission(pd.DataFrame(rows), paths=paths, target_year=2024)
    current = admitted.panel[
        admitted.panel["year"].eq(2024) & admitted.panel["ori9"].eq("ZZ0000000")
    ].set_index("offense")
    assert current.loc["murder", "level1_admission_status"] == "valid_complete_year"
    assert current.loc["murder", "level_policy_resolution"] == "admit_valid_complete_year"
    comparable = [offense for offense in OFFENSES_7 if offense != "murder"]
    assert set(current.loc[comparable, "level1_admission_status"]) == {"coverage_defective"}
    assert set(current.loc[comparable, "level_repair_mode"]) == {"decayed_own_history_or_pooled"}
    assert current.loc[comparable, "preferred_count"].eq(0.0).all()
    assert admitted.review_queue.empty


@pytest.mark.parametrize(
    "adjudication,expected_level1,expected_mode,expected_count",
    [
        ("accept_unchanged", "valid_complete_year", "none", 21000.0),
        ("refuse_repair", "coverage_defective", "decayed_own_history_or_pooled", 0.0),
        ("refuse_silent", "coverage_defective", "soft_benchmark_reconciliation", 0.0),
    ],
)
def test_registry_ruling_precedes_mechanical_policy(
    tmp_path,
    clean_vector,
    adjudication,
    expected_level1,
    expected_mode,
    expected_count,
):
    paths, configs = _registry_paths(tmp_path)
    pd.DataFrame([{
        "case_id": "reviewed-larceny", "ori9": "ZZ0000000", "year": 2024,
        "offense": "larceny", "adjudication": adjudication,
        "reason_code": "external_review", "reviewer": "tester",
        "evidence_note": "unit-test ruling",
    }]).to_csv(configs / "level_lane_admission_registry.csv", index=False)
    pd.DataFrame(columns=[
        "case_id", "ori9", "year", "offense", "level1_status", "level2_status", "reason_code",
        "repair_mode", "replacement_count", "external_check_status", "evidence_source",
    ]).to_csv(configs / "level_lane_hard_evidence.csv", index=False)
    structural = tmp_path / "analysis_scratch" / "level_lane_screen"
    structural.mkdir(parents=True)
    pd.DataFrame(columns=["ori9", "year"]).to_csv(
        structural / "structural_zero_corroborated_all.csv", index=False
    )
    admitted = apply_level_lane_admission(
        pd.DataFrame(_spike_with_peer_rows(clean_vector)), paths=paths, target_year=2024
    )
    row = admitted.panel[
        admitted.panel["year"].eq(2024)
        & admitted.panel["ori9"].eq("ZZ0000000")
        & admitted.panel["offense"].eq("larceny")
    ].iloc[0]
    assert row["level1_admission_status"] == expected_level1
    assert row["level_repair_mode"] == expected_mode
    assert row["preferred_count"] == expected_count
    assert row["level_lane_registry_adjudication"] == adjudication


def test_explicit_accept_survives_coagency_hard_evidence_that_clears_soft_signal(
    tmp_path,
    clean_vector,
):
    paths, configs = _registry_paths(tmp_path)
    pd.DataFrame([{
        "case_id": "reviewed-larceny", "ori9": "ZZ0000000", "year": 2024,
        "offense": "larceny", "adjudication": "accept_unchanged",
        "reason_code": "external_evidence_admit", "reviewer": "tester",
        "evidence_note": "review predates a co-agency hard-evidence replacement",
    }]).to_csv(configs / "level_lane_admission_registry.csv", index=False)
    pd.DataFrame([{
        "case_id": "hard-murder", "ori9": "ZZ0000000", "year": 2024,
        "offense": "murder", "level1_status": "valid_complete_year",
        "level2_status": "wrong_category_mapping",
        "reason_code": "official_agency_publication_contradiction",
        "repair_mode": "source_supported_replacement", "replacement_count": 11.0,
        "external_check_status": "contradiction_confirmed",
        "evidence_source": "unit-test source",
    }]).to_csv(configs / "level_lane_hard_evidence.csv", index=False)
    structural = tmp_path / "analysis_scratch" / "level_lane_screen"
    structural.mkdir(parents=True)
    pd.DataFrame(columns=["ori9", "year"]).to_csv(
        structural / "structural_zero_corroborated_all.csv", index=False
    )
    rows = []
    for year in (2023, 2024):
        for offense, count in clean_vector.items():
            rows.append({
                "ori9": "ZZ0000000", "state_fips": "99", "state_abbr": "ZZ",
                "year": year, "offense": offense, "preferred_count": count,
                "preferred_source": "state_publication_annual",
                "preferred_months_reported": 12.0, "usable_as_observed": True,
                "current_row_is_true_partial": False,
            })
    admitted = apply_level_lane_admission(
        pd.DataFrame(rows), paths=paths, target_year=2024
    ).panel
    current = admitted[admitted["year"].eq(2024)].set_index("offense")
    assert current.loc["murder", "level_repair_mode"] == "source_supported_replacement"
    assert current.loc["larceny", "level_policy_resolution"] == "registry_accept_unchanged"
    assert current.loc["larceny", "level_lane_registry_adjudication"] == "accept_unchanged"
    assert current.loc["larceny", "preferred_count"] == clean_vector["larceny"]


def test_soft_signal_does_not_relabel_hard_coverage_failure_as_held(tmp_path, clean_vector):
    paths, configs = _registry_paths(tmp_path)
    pd.DataFrame(columns=[
        "case_id", "ori9", "year", "offense", "adjudication", "reason_code", "reviewer", "evidence_note",
    ]).to_csv(configs / "level_lane_admission_registry.csv", index=False)
    pd.DataFrame(columns=[
        "case_id", "ori9", "year", "offense", "level1_status", "level2_status", "reason_code",
        "repair_mode", "replacement_count", "external_check_status", "evidence_source",
    ]).to_csv(configs / "level_lane_hard_evidence.csv", index=False)
    structural = tmp_path / "analysis_scratch" / "level_lane_screen"
    structural.mkdir(parents=True)
    pd.DataFrame(columns=["ori9", "year"]).to_csv(
        structural / "structural_zero_corroborated_all.csv", index=False
    )
    rows = []
    for year, factor, usable in ((2023, 1.0, True), (2024, 0.0, False)):
        for offense, count in clean_vector.items():
            rows.append({
                "ori9": "ZZ0000000", "state_fips": "99", "state_abbr": "ZZ", "year": year,
                "offense": offense, "preferred_count": count * factor,
                "preferred_source": "state_publication_annual", "preferred_months_reported": 12.0,
                "usable_as_observed": usable, "current_row_is_true_partial": False,
            })
    admitted = apply_level_lane_admission(pd.DataFrame(rows), paths=paths, target_year=2024)
    current = admitted.panel[admitted.panel["year"].eq(2024)]
    assert set(current["level1_admission_status"]) == {"coverage_defective"}
    assert set(current["level2_semantic_status"]) == {"semantically_unusable"}
    assert set(current["level_repair_mode"]) == {"decayed_own_history_or_pooled"}
    assert not current["level_review_hold"].any()


def test_near_zero_collapse_vs_own_history_routes_to_repair(tmp_path):
    paths, configs = _registry_paths(tmp_path)
    pd.DataFrame({"ori": ["ZZ0000000"]}).to_csv(
        configs / "overlap_custom_footprints.csv", index=False
    )
    pd.DataFrame(columns=[
        "case_id", "ori9", "year", "offense", "adjudication", "reason_code", "reviewer", "evidence_note",
    ]).to_csv(configs / "level_lane_admission_registry.csv", index=False)
    pd.DataFrame(columns=[
        "case_id", "ori9", "year", "offense", "level1_status", "level2_status", "reason_code",
        "repair_mode", "replacement_count", "external_check_status", "evidence_source",
    ]).to_csv(configs / "level_lane_hard_evidence.csv", index=False)
    structural = tmp_path / "analysis_scratch" / "level_lane_screen"
    structural.mkdir(parents=True)
    pd.DataFrame(columns=["ori9", "year"]).to_csv(
        structural / "structural_zero_corroborated_all.csv", index=False
    )
    historical = {
        "murder": 1.0,
        "rape": 2.0,
        "robbery": 1.0,
        "aggravated_assault": 12.0,
        "burglary": 4.0,
        "larceny": 8.0,
        "motor_vehicle_theft": 2.0,
    }
    current = {
        "murder": 0.0,
        "rape": 0.0,
        "robbery": 0.0,
        "aggravated_assault": 5.0,
        "burglary": 1.0,
        "larceny": 1.0,
        "motor_vehicle_theft": 0.0,
    }
    rows = []
    for year, vector in ((2022, historical), (2023, historical), (2024, current)):
        for offense, count in vector.items():
            rows.append({
                "ori9": "ZZ0000000", "state_fips": "99", "state_abbr": "ZZ", "year": year,
                "offense": offense, "preferred_count": count,
                "preferred_source": "nibrs_srs_equivalent_annual", "preferred_months_reported": 12.0,
                "usable_as_observed": True, "current_row_is_true_partial": False,
            })
    admitted = apply_level_lane_admission(pd.DataFrame(rows), paths=paths, target_year=2024)
    target = admitted.panel[admitted.panel["year"].eq(2024)]
    assert set(target["level1_admission_status"]) == {"coverage_defective"}
    assert set(target["level_admission_reason"]) == {"uncorroborated_near_zero_filed_vector"}
    assert set(target["level_repair_mode"]) == {"decayed_own_history_or_pooled"}
    assert target["preferred_count"].eq(0.0).all()


def _ledger_row(**overrides):
    row = {
        "ori9": "ZZ0000000",
        "year": 2024,
        "offense": "larceny",
        "jurisdiction_id": "99:state_nonmunicipal_remainder",
        "jurisdiction_link_weight": 1.0,
        "ownership_class": "agency_level",
        "level1_admission_status": "unresolved_review",
        "level2_semantic_status": "definitionally_complete",
        "level_admission_reason": "collapse_vs_history",
        "level_repair_mode": "unchanged_review_hold",
        "external_check_status": "unavailable",
        "admitted_input_mass": 10.0,
        "accepted_observed_mass": 0.0,
        "repair_mass": 0.0,
        "benchmark_mass": 0.0,
        "unresolved_mass": 10.0,
        "invalid_fragment_audit_mass": 0.0,
        "final_control_mass": 10.0,
        "consumption_status": "consumed",
    }
    row.update(overrides)
    return row


def test_ledger_rejects_held_mass_sized_above_admitted_input():
    ledger = pd.DataFrame([_ledger_row(unresolved_mass=12.0, final_control_mass=12.0)])
    controls = pd.DataFrame([{
        "jurisdiction_id": "99:state_nonmunicipal_remainder",
        "offense": "larceny",
        "adjusted_count_ags_core": 12.0,
    }])
    with pytest.raises(ValueError, match="unchanged review hold changed admitted input"):
        assert_level_lane_mass_ledger(ledger=ledger, controls=controls)


def test_benchmark_unit_stays_separate_and_drives_control_disclosure():
    held = _ledger_row()
    benchmark = _ledger_row(
        ori9="POOL::99:county:99001",
        ownership_class="pooled_silent_unit",
        level1_admission_status="coverage_defective",
        level2_semantic_status="semantically_unusable",
        level_admission_reason="pooled_silent_territory",
        level_repair_mode="soft_benchmark_reconciliation",
        external_check_status="cde_soft_benchmark",
        admitted_input_mass=0.0,
        benchmark_mass=5.0,
        unresolved_mass=0.0,
        final_control_mass=5.0,
    )
    ledger = pd.DataFrame([held, benchmark])
    controls = pd.DataFrame([{
        "jurisdiction_id": "99:state_nonmunicipal_remainder",
        "offense": "larceny",
        "state_fips": "99",
        "adjusted_count_ags_core": 15.0,
    }])
    assert_level_lane_mass_ledger(ledger=ledger, controls=controls)
    disclosed = attach_level_lane_disclosures(
        controls=controls,
        ledger=ledger,
        benchmark_imputation=SimpleNamespace(state_identity=pd.DataFrame()),
    ).iloc[0]
    assert disclosed["level_repair_mode"] == "soft_benchmark_reconciliation"
    assert disclosed["level_repair_share"] == pytest.approx(1.0 / 3.0)
    assert bool(disclosed["level_unresolved_review"])


def test_control_discloses_composite_without_dual_owning_a_component():
    held = _ledger_row()
    repair = _ledger_row(
        ori9="ZZ0000001",
        level1_admission_status="coverage_defective",
        level2_semantic_status="semantically_unusable",
        level_admission_reason="selected_lane_not_usable",
        level_repair_mode="decayed_own_history_or_pooled",
        admitted_input_mass=0.0,
        repair_mass=3.0,
        unresolved_mass=0.0,
        final_control_mass=3.0,
    )
    benchmark = _ledger_row(
        ori9="POOL::99:county:99001",
        ownership_class="pooled_silent_unit",
        level1_admission_status="coverage_defective",
        level2_semantic_status="semantically_unusable",
        level_admission_reason="pooled_silent_territory",
        level_repair_mode="soft_benchmark_reconciliation",
        external_check_status="cde_soft_benchmark",
        admitted_input_mass=0.0,
        benchmark_mass=5.0,
        unresolved_mass=0.0,
        final_control_mass=5.0,
    )
    ledger = pd.DataFrame([held, repair, benchmark])
    controls = pd.DataFrame([{
        "jurisdiction_id": "99:state_nonmunicipal_remainder",
        "offense": "larceny",
        "state_fips": "99",
        "adjusted_count_ags_core": 18.0,
    }])
    assert_level_lane_mass_ledger(ledger=ledger, controls=controls)
    disclosed = attach_level_lane_disclosures(
        controls=controls,
        ledger=ledger,
        benchmark_imputation=SimpleNamespace(state_identity=pd.DataFrame()),
    ).iloc[0]
    assert disclosed["level_repair_mode"] == "reason_coded_level_repair_plus_soft_benchmark"
    assert disclosed["level_repair_share"] == pytest.approx(8.0 / 18.0)
