"""Release-mirror tests for source-state service allocation ledgers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"


def _validator():
    spec = importlib.util.spec_from_file_location(
        "validate_release_outputs_service_transfer", VALIDATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = _validator()


def _ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": VALIDATOR.YEAR,
                "service_scope_id": "synthetic_service",
                "canonical_target_ori": "AA0000001",
                "offense": "robbery",
                "source_state_fips": "04",
                "allocation_state_fips": "04",
                "source_target_count": 100.0,
                "allocation_count": 60.0,
                "allocation_share": 0.6,
                "route_reason": "",
            },
            {
                "year": VALIDATOR.YEAR,
                "service_scope_id": "synthetic_service",
                "canonical_target_ori": "AA0000001",
                "offense": "robbery",
                "source_state_fips": "04",
                "allocation_state_fips": "35",
                "source_target_count": 100.0,
                "allocation_count": 30.0,
                "allocation_share": 0.3,
                "route_reason": "",
            },
            {
                "year": VALIDATOR.YEAR,
                "service_scope_id": "synthetic_service",
                "canonical_target_ori": "AA0000001",
                "offense": "robbery",
                "source_state_fips": "04",
                "allocation_state_fips": "49",
                "source_target_count": 100.0,
                "allocation_count": 10.0,
                "allocation_share": 0.1,
                "route_reason": "",
            },
        ]
    )


def _audit() -> pd.DataFrame:
    rows = []
    for state, counts in {"04": (35.0, 25.0), "35": (30.0,), "49": (10.0,)}.items():
        for index, count in enumerate(counts):
            rows.append(
                {
                    "state_fips": state,
                    "bg_id": f"{state}00100000{index + 1}",
                    "jurisdiction_type": "custom_footprint_overlap_layer",
                    "offense": "robbery",
                    "service_scope_id": "synthetic_service",
                    "canonical_target_ori": "AA0000001",
                    "source_state_fips": "04",
                    "source_target_count": 100.0,
                    "component_count_after": count,
                }
            )
    return pd.DataFrame(rows)


def _check(ledger: pd.DataFrame, audit: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    VALIDATOR._check_service_state_transfer_table(
        ledger=ledger, component_audit=audit, issues=issues
    )
    return issues


def test_within_state_and_import_rows_reconstruct_one_source_target():
    issues: list[str] = []
    summary = VALIDATOR._check_service_state_transfer_table(
        ledger=_ledger(), component_audit=_audit(), issues=issues
    )
    assert issues == []
    assert summary["ok"] is True
    assert summary["cross_state_rows"] == 2
    assert summary["national_source_target_total"] == pytest.approx(100.0)
    assert summary["national_allocation_total"] == pytest.approx(100.0)


def test_dropped_import_row_fails_source_and_component_reconciliation():
    issues = _check(_ledger().iloc[:2].copy(), _audit())
    assert any("source targets are not reconstructed" in issue for issue in issues)
    assert any("destination allocations do not match" in issue for issue in issues)


def test_duplicate_import_row_fails_even_when_identity_text_is_unchanged():
    ledger = pd.concat([_ledger(), _ledger().iloc[[1]]], ignore_index=True)
    issues = _check(ledger, _audit())
    assert any("duplicate allocation-key rows" in issue for issue in issues)


def test_component_audit_source_target_must_match_independent_ledger_target():
    audit = _audit()
    audit.loc[audit["state_fips"].eq("35"), "source_target_count"] = 99.0
    issues = _check(_ledger(), audit)
    assert any("component-audit source targets disagree" in issue for issue in issues)


def test_terminal_row_requires_matching_source_owned_unlocated_mass():
    ledger = _ledger().iloc[:2].copy()
    ledger.loc[1, "allocation_state_fips"] = pd.NA
    ledger.loc[1, "allocation_count"] = 40.0
    ledger.loc[1, "allocation_share"] = 0.4
    ledger.loc[1, "route_reason"] = "service_no_eligible_receiver"
    audit = _audit().loc[_audit()["state_fips"].eq("04")].copy()
    unlocated = pd.DataFrame(
        [{
            "source_state_fips": "04",
            "service_scope_id": "synthetic_service",
            "canonical_target_ori": "AA0000001",
            "offense": "robbery",
            "unlocated_count": 40.0,
            "route_reason": "service_no_eligible_receiver",
        }]
    )
    issues: list[str] = []
    summary = VALIDATOR._check_service_state_transfer_table(
        ledger=ledger,
        component_audit=audit,
        unlocated_mass=unlocated,
        issues=issues,
    )
    assert issues == []
    assert summary["terminal_unlocated_rows"] == 1

    unlocated.loc[0, "unlocated_count"] = 39.0
    assert any(
        "terminal allocations do not match" in issue
        for issue in _check_terminal(ledger, audit, unlocated)
    )


def _check_terminal(
    ledger: pd.DataFrame, audit: pd.DataFrame, unlocated: pd.DataFrame
) -> list[str]:
    issues: list[str] = []
    VALIDATOR._check_service_state_transfer_table(
        ledger=ledger,
        component_audit=audit,
        unlocated_mass=unlocated,
        issues=issues,
    )
    return issues


def test_published_geographic_totals_reconcile_to_source_state_control(tmp_path: Path):
    ledger = _ledger().iloc[:2].copy()
    ledger.loc[1, "allocation_count"] = 40.0
    ledger.loc[1, "allocation_share"] = 0.4
    ledger.to_parquet(
        tmp_path / f"allocation_service_state_transfer_{VALIDATOR.YEAR}.parquet",
        index=False,
    )
    rows = []
    for state, robbery in (("04", 60.0), ("35", 40.0)):
        row = {"state_fips": state}
        row.update({f"expected_count_{offense}": 0.0 for offense in VALIDATOR.OFFENSES_7})
        row["expected_count_robbery"] = robbery
        rows.append(row)
    surface = pd.DataFrame(rows)
    surface.to_parquet(
        tmp_path / f"crimerisk_block_group_{VALIDATOR.YEAR}_ags_core.parquet",
        index=False,
    )
    surface.to_parquet(
        tmp_path / f"crimerisk_tract_{VALIDATOR.YEAR}_ags_core.parquet",
        index=False,
    )
    controls = pd.DataFrame(
        [{"state_fips": "04", "offense": "robbery", "control_target": 100.0}]
    )
    contract = VALIDATOR.ReleaseContract(
        service_wide_custom_allocation_enabled=True,
        service_wide_custom_allocation_version="service_state_transfer_v1",
        service_wide_custom_scope_ids=("synthetic_service",),
        service_state_transfer_companion_table=(
            f"allocation_service_state_transfer_{VALIDATOR.YEAR}.parquet"
        ),
    )
    issues: list[str] = []
    summary = VALIDATOR._published_surface_state_total_reconciliation(
        output_dir=tmp_path,
        controls=controls,
        issues=issues,
        contract=contract,
    )
    assert issues == []
    assert summary["surfaces"]["block_group_ags_core"]["max_abs_delta"] == pytest.approx(0.0)
    assert summary["surfaces"]["tract_ags_core"]["max_abs_delta"] == pytest.approx(0.0)


def test_release_contract_fails_closed_when_declared_ledger_is_missing(tmp_path: Path):
    contract = VALIDATOR.ReleaseContract(
        manifest_present=True,
        service_wide_custom_allocation_enabled=True,
        service_wide_custom_allocation_version="service_state_transfer_v1",
        service_wide_custom_scope_ids=("synthetic_service",),
        service_state_transfer_companion_table=(
            f"allocation_service_state_transfer_{VALIDATOR.YEAR}.parquet"
        ),
    )
    issues: list[str] = []
    summary = VALIDATOR._check_release_contract(
        contract=contract, state_output_dir=tmp_path, issues=issues
    )
    assert summary["service_state_transfer"]["companion_table_present"] is False
    assert any(
        "service-wide allocation is declared but ledger" in issue for issue in issues
    )


def _component_reconciliation_fixture(tmp_path: Path, unlocated: pd.DataFrame):
    pd.DataFrame(
        [{
            "state_fips": "04",
            "source_state_fips": "04",
            "jurisdiction_id": "04:state_nonmunicipal_remainder",
            "jurisdiction_type": "state_nonmunicipal_remainder",
            "offense": "robbery",
            "component_count_before": 10.0,
            "component_count_after": 10.0,
            "service_scope_id": pd.NA,
        }]
    ).to_parquet(
        tmp_path / f"allocation_component_denominator_audit_{VALIDATOR.YEAR}.parquet",
        index=False,
    )
    unlocated.to_parquet(
        tmp_path / f"unlocated_mass_{VALIDATOR.YEAR}.parquet", index=False
    )
    controls = pd.DataFrame(
        [{
            "state_fips": "04",
            "jurisdiction_id": "04:state_nonmunicipal_remainder",
            "offense": "robbery",
            "reported_count_preferred": 10.0,
            "adjusted_count_ags_core": 10.0,
        }]
    )
    contract = VALIDATOR.ReleaseContract(service_wide_custom_allocation_enabled=True)
    issues: list[str] = []
    VALIDATOR._component_control_reconciliation(
        output_dir=tmp_path, controls=controls, issues=issues, contract=contract
    )
    return issues


def test_component_reconciliation_allows_ordinary_unlocated_without_source_state(tmp_path: Path):
    issues = _component_reconciliation_fixture(
        tmp_path,
        pd.DataFrame([{
            "jurisdiction_id": "04:state_nonmunicipal_remainder",
            "offense": "robbery",
            "unlocated_count": 0.0,
            "route_reason": "county_no_eligible_receiver",
        }]),
    )
    assert not any("missing source_state_fips" in issue for issue in issues)


def test_component_reconciliation_requires_source_state_on_service_terminal(tmp_path: Path):
    issues = _component_reconciliation_fixture(
        tmp_path,
        pd.DataFrame([{
            "jurisdiction_id": "AZ0018900",
            "offense": "robbery",
            "unlocated_count": 1.0,
            "route_reason": "service_no_eligible_receiver",
        }]),
    )
    assert any("missing source_state_fips" in issue for issue in issues)


def test_component_reconciliation_credits_valid_service_terminal_source_state(tmp_path: Path):
    issues = _component_reconciliation_fixture(
        tmp_path,
        pd.DataFrame([{
            "jurisdiction_id": "AZ0018900",
            "source_state_fips": "04",
            "offense": "robbery",
            "unlocated_count": 1.0,
            "route_reason": "service_no_eligible_receiver",
        }]),
    )
    assert not any("source_state_fips" in issue for issue in issues)


def _fraction_contract_fixture(tmp_path: Path) -> tuple[object, Path, Path]:
    service_path = tmp_path / "configs" / "service_wide_footprint_coverage.csv"
    ordinary_path = (
        tmp_path / "configs" / "overlap_custom_footprint_resident_coverage.csv"
    )
    footprint_path = tmp_path / "configs" / "overlap_custom_footprints.csv"
    scope_path = tmp_path / "configs" / "service_wide_agency_scopes.csv"
    policy_path = tmp_path / "configs" / "primary_service_response_policies.csv"
    service_path.parent.mkdir()
    pd.DataFrame(
        [{
            "service_scope_id": "scope",
            "state_fips": "04",
            "block_group_geoid": "040010000001",
            "bg_service_population_coverage_share": 0.0,
            "bg_land_area_coverage_share": 0.3,
            "coverage_basis": "synthetic_service_geometry",
        }]
    ).to_csv(service_path, index=False)
    pd.DataFrame(
        [{
            "ori": "AA0000001",
            "state_fips": "04",
            "block_group_geoid": "040010000001",
            "bg_responsibility_population_coverage_share": 0.5,
            "responsibility_fraction_basis": "synthetic_own_geometry",
        }]
    ).to_csv(ordinary_path, index=False)
    pd.DataFrame(
        [
            {
                "ori": "AA0000001",
                "state_fips": "04",
                "block_group_geoid": "040010000001",
                "weight_share_basis": "resident_population",
                "bg_population_coverage_share": 0.8,
            },
            {
                "ori": "BB0000001",
                "state_fips": "04",
                "block_group_geoid": "040010000001",
                "weight_share_basis": "service_area_prior",
                "bg_population_coverage_share": 0.0,
            },
        ]
    ).to_csv(footprint_path, index=False)
    pd.DataFrame(
        [{"service_scope_id": "scope", "canonical_target_ori": "BB0000001"}]
    ).to_csv(scope_path, index=False)
    pd.DataFrame(
        [{
            "service_scope_id": "scope",
            "primary_response_policy": (
                "displace_county_remainder_within_reviewed_carveout"
            ),
            "official_source_ref": "https://example.test/official",
            "evidence_artifact": "state/analysis/evidence.json",
            "evidence_sha256": "a" * 64,
        }]
    ).to_csv(policy_path, index=False)
    audit_path = tmp_path / f"allocation_component_denominator_audit_{VALIDATOR.YEAR}.parquet"
    pd.DataFrame(
        [{
            "bg_responsibility_population_coverage_share": 0.5,
            "responsibility_fraction_basis": "synthetic_own_geometry",
        }]
    ).to_parquet(audit_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "input_file_stats": {
            "service_wide_footprint_coverage": {"path": str(service_path)},
            "overlap_custom_footprint_resident_coverage": {"path": str(ordinary_path)},
            "overlap_custom_footprints": {"path": str(footprint_path)},
            "service_wide_agency_scopes": {"path": str(scope_path)},
            "primary_service_response_policies": {"path": str(policy_path)},
        },
        "summary": {
            "service_wide_custom_allocation": {
                "allocation_weight_basis": "count_prior_x_within_bg_land_fraction",
                "footprint_coverage_path": "configs/service_wide_footprint_coverage.csv",
                "footprint_coverage_sha256": VALIDATOR.hashlib.sha256(
                    service_path.read_bytes()
                ).hexdigest(),
                "primary_response_policy_path": (
                    "configs/primary_service_response_policies.csv"
                ),
                "primary_response_policy_sha256": VALIDATOR.hashlib.sha256(
                    policy_path.read_bytes()
                ).hexdigest(),
                "primary_response_scope_ids": ["scope"],
            },
            "ordinary_resident_custom_allocation": {
                "allocation_weight_basis": (
                    "count_prior_x_within_bg_responsibility_population_fraction"
                ),
                "footprint_coverage_path": (
                    "configs/overlap_custom_footprint_resident_coverage.csv"
                ),
                "footprint_coverage_sha256": VALIDATOR.hashlib.sha256(
                    ordinary_path.read_bytes()
                ).hexdigest(),
            },
        },
    }
    manifest_path.write_text(VALIDATOR.json.dumps(manifest))
    contract = VALIDATOR.ReleaseContract(
        manifest_path=manifest_path,
        manifest_present=True,
        service_wide_custom_allocation_enabled=True,
    )
    return contract, ordinary_path, manifest_path


def test_fraction_contract_accepts_bound_in_range_own_within_union(tmp_path: Path):
    contract, _ordinary_path, _manifest_path = _fraction_contract_fixture(tmp_path)
    issues: list[str] = []
    summary = VALIDATOR._check_custom_footprint_fraction_contract(
        contract=contract, state_output_dir=tmp_path, issues=issues
    )
    assert issues == []
    assert summary["ok"] is True
    assert summary["audit_responsibility_rows"] == 1


def test_fraction_contract_rejects_out_of_range_and_own_beyond_union(tmp_path: Path):
    contract, ordinary_path, manifest_path = _fraction_contract_fixture(tmp_path)
    service_path = tmp_path / "configs" / "service_wide_footprint_coverage.csv"
    ordinary = pd.read_csv(ordinary_path)
    ordinary.loc[0, "bg_responsibility_population_coverage_share"] = 1.1
    ordinary.to_csv(ordinary_path, index=False)
    manifest = VALIDATOR.json.loads(manifest_path.read_text())
    manifest["summary"]["ordinary_resident_custom_allocation"][
        "footprint_coverage_sha256"
    ] = VALIDATOR.hashlib.sha256(ordinary_path.read_bytes()).hexdigest()
    service = pd.read_csv(service_path)
    service.loc[0, "bg_service_population_coverage_share"] = 0.7
    service.to_csv(service_path, index=False)
    manifest["summary"]["service_wide_custom_allocation"][
        "footprint_coverage_sha256"
    ] = VALIDATOR.hashlib.sha256(service_path.read_bytes()).hexdigest()
    manifest_path.write_text(VALIDATOR.json.dumps(manifest))
    issues: list[str] = []
    VALIDATOR._check_custom_footprint_fraction_contract(
        contract=contract, state_output_dir=tmp_path, issues=issues
    )
    assert any("must be finite in (0, 1]" in issue for issue in issues)
    assert any("exceeds union geometry" in issue for issue in issues)
    assert any("service own-population fraction exceeds union" in issue for issue in issues)


@pytest.mark.parametrize("bad_union", ["", "inf", "-0.1", "1.1"])
def test_fraction_contract_rejects_invalid_service_union_coverage(
    tmp_path: Path, bad_union: object
):
    contract, _ordinary_path, _manifest_path = _fraction_contract_fixture(tmp_path)
    footprint_path = tmp_path / "configs" / "overlap_custom_footprints.csv"
    footprint = pd.read_csv(footprint_path, dtype="string")
    service = footprint["weight_share_basis"].eq("service_area_prior")
    footprint.loc[service, "bg_population_coverage_share"] = bad_union
    footprint.to_csv(footprint_path, index=False)

    issues: list[str] = []
    VALIDATOR._check_custom_footprint_fraction_contract(
        contract=contract, state_output_dir=tmp_path, issues=issues
    )
    assert any(
        "service union coverage must be explicit and finite in [0, 1]" in issue
        for issue in issues
    )
