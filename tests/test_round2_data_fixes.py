from __future__ import annotations

import pandas as pd

from crimerisk.contract_coverage import apply_contract_coverage
from crimerisk.jurisdiction_reference import ReferenceArtifacts
from scripts.diagnostics.validate_release_outputs import _component_control_reconciliation


def test_covering_agency_partitions_caseload_across_own_and_covered_footprints():
    crosswalk = pd.DataFrame(
        [
            {"ori": "HOST00000", "state_fips": "99", "state_abbr": "ZZ", "jurisdiction_id": "own", "relationship_type": "exclusive", "weight": 1.0, "review_status": "auto", "overlap_subtype": None, "geometry_hint": None, "resolution_source": "test", "source_table": "local_full"},
            {"ori": "COVERED01", "state_fips": "99", "state_abbr": "ZZ", "jurisdiction_id": "town1", "relationship_type": "exclusive", "weight": 1.0, "review_status": "auto", "overlap_subtype": None, "geometry_hint": None, "resolution_source": "test", "source_table": "local_full"},
            {"ori": "COVERED02", "state_fips": "99", "state_abbr": "ZZ", "jurisdiction_id": "town2", "relationship_type": "exclusive", "weight": 1.0, "review_status": "auto", "overlap_subtype": None, "geometry_hint": None, "resolution_source": "test", "source_table": "local_full"},
        ]
    )
    master = pd.DataFrame(
        {
            "jurisdiction_id": ["own", "town1", "town2"],
            "jurisdiction_type": ["municipal"] * 3,
            "is_contracted_place": [False] * 3,
        }
    )
    relationships = pd.DataFrame(
        {
            "ori9": ["COVERED01", "COVERED02"],
            "covered_by_ori": ["HOST00000", "HOST00000"],
            "covering_ori": ["HOST00000", "HOST00000"],
            "covered_population": [30.0, 20.0],
            "covering_service_population": [100.0, 100.0],
        }
    )
    empty = pd.DataFrame()
    artifacts = ReferenceArtifacts(empty, empty, master, crosswalk)
    out = apply_contract_coverage(artifacts, relationships)
    host = out.agency_to_jurisdiction_crosswalk.query("ori == 'HOST00000'").set_index("jurisdiction_id")
    assert host["weight"].to_dict() == {"own": 0.5, "town1": 0.3, "town2": 0.2}
    covered = out.agency_to_jurisdiction_crosswalk.query("ori != 'HOST00000'")
    assert covered["weight"].eq(0.0).all()
    assert covered["relationship_type"].eq("covered_by_other_agency").all()
    marked = out.jurisdiction_master.set_index("jurisdiction_id")["is_contracted_place"]
    assert bool(marked["town1"]) and bool(marked["town2"])


def test_coincident_contract_footprints_are_one_crosswalk_assignment():
    crosswalk = pd.DataFrame(
        [
            {"ori": "HOST00000", "state_fips": "99", "state_abbr": "ZZ", "jurisdiction_id": "same", "relationship_type": "exclusive", "weight": 1.0, "review_status": "auto", "overlap_subtype": None, "geometry_hint": None, "resolution_source": "test", "source_table": "local_full"},
            {"ori": "COVERED01", "state_fips": "99", "state_abbr": "ZZ", "jurisdiction_id": "same", "relationship_type": "exclusive", "weight": 1.0, "review_status": "auto", "overlap_subtype": None, "geometry_hint": None, "resolution_source": "test", "source_table": "local_full"},
        ]
    )
    master = pd.DataFrame(
        {"jurisdiction_id": ["same"], "jurisdiction_type": ["municipal"], "is_contracted_place": [False]}
    )
    relationships = pd.DataFrame(
        {
            "ori9": ["COVERED01"],
            "covered_by_ori": ["HOST00000"],
            "covering_ori": ["HOST00000"],
            "covered_population": [30.0],
            "covering_service_population": [100.0],
        }
    )
    artifacts = ReferenceArtifacts(pd.DataFrame(), pd.DataFrame(), master, crosswalk)
    out = apply_contract_coverage(artifacts, relationships).agency_to_jurisdiction_crosswalk
    host = out.query("ori == 'HOST00000'")
    assert len(host) == 1
    assert host.iloc[0]["jurisdiction_id"] == "same"
    assert host.iloc[0]["weight"] == 1.0


def test_component_control_reconciliation_separates_attribution_from_spatial_repair(tmp_path):
    components = pd.DataFrame(
        [
            {
                "state_fips": "55",
                "jurisdiction_id": "55:municipal:place:5500001",
                "jurisdiction_type": "municipal",
                "offense": "larceny",
                "component_count_before": 10.0,
                "component_count_after": 15.0,
            },
            {
                "state_fips": "55",
                "jurisdiction_id": "55:state_nonmunicipal_remainder:county:55001",
                "jurisdiction_type": "localized_remainder_county_layer",
                "offense": "larceny",
                "component_count_before": 20.0,
                "component_count_after": 15.0,
            },
        ]
    )
    components.to_parquet(tmp_path / "allocation_component_denominator_audit_2024.parquet", index=False)
    controls = pd.DataFrame(
        [
            {
                "state_fips": "55",
                "jurisdiction_id": "55:municipal:place:5500001",
                "offense": "larceny",
                "reported_count_preferred": 10.0,
                "adjusted_count_ags_core": 10.0,
            },
            {
                "state_fips": "55",
                "jurisdiction_id": "55:state_nonmunicipal_remainder",
                "offense": "larceny",
                "reported_count_preferred": 20.0,
                "adjusted_count_ags_core": 20.0,
            },
        ]
    )
    issues: list[str] = []

    summary = _component_control_reconciliation(output_dir=tmp_path, controls=controls, issues=issues)

    assert summary["ok"] is True
    assert issues == []
    assert summary["target_reconciliation_bad_rows"] == 0
    assert summary["post_repair_spatial_jurisdiction_diff_rows_nonblocking"] == 2
    assert summary["post_repair_state_offense_conservation_bad_rows"] == 0
