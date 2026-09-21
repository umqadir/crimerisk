from pathlib import Path

import pandas as pd
import pytest

from crimerisk.federal_ori_scope import (
    assert_nonzero_federal_oris_are_classified,
    build_federal_ori_mass_audit,
    exclude_national_hq_rows,
    identify_federal_oris,
    load_federal_ori_scope_registry,
    write_federal_ori_mass_audit,
)
from crimerisk.paths import RepoPaths, get_paths


def _registry(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ori9": ori9, "class": scope, "evidence": "test", "source": "test"}
            for ori9, scope in rows
        ],
        columns=["ori9", "class", "evidence", "source"],
    )


def test_federal_mass_audit_writes_under_runtime_state(tmp_path: Path):
    paths = RepoPaths.from_repo_root(tmp_path)
    out = write_federal_ori_mass_audit(
        pd.DataFrame({"ori9": ["US0000001"]}), paths=paths, year=2025
    )
    assert out == tmp_path / "state/analysis/dc_federal/federal_ori_2025_mass_audit.csv"
    assert out.is_file()


def test_registry_loader_is_strict_and_fail_closed(tmp_path: Path):
    paths = RepoPaths.from_repo_root(tmp_path)
    registry_path = tmp_path / "configs" / "federal_ori_scope.csv"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        "ori9,class,evidence,source\nDCFBIWA00,national_hq,test,test\n",
        encoding="utf-8",
    )
    loaded = load_federal_ori_scope_registry(paths)
    assert loaded[["ori9", "class"]].to_records(index=False).tolist() == [
        ("DCFBIWA00", "national_hq")
    ]

    registry_path.write_text(
        "ori9,class,evidence,source\nDCFBIWA00,unknown,test,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="installation or national_hq"):
        load_federal_ori_scope_registry(paths)


def test_unregistered_nonzero_federal_ori_fails_closed():
    census = identify_federal_oris(
        pd.DataFrame(
            {
                "ori9": ["ZZFBI0100"],
                "agency_name_raw": ["FBI TEST FIELD OFFICE"],
                "agency_type_raw": ["federal"],
            }
        )
    )
    estimates = pd.DataFrame(
        {
            "ori9": ["ZZFBI0100"],
            "offense": ["larceny"],
            "estimated_count": [1.0],
        }
    )
    with pytest.raises(ValueError, match="absent from configs/federal_ori_scope.csv"):
        assert_nonzero_federal_oris_are_classified(
            census=census,
            agency_estimates=estimates,
            registry=_registry([]),
        )


def test_dc_hq_is_excluded_but_installation_and_metro_controls_are_unchanged():
    registry = load_federal_ori_scope_registry(get_paths())
    rows = pd.DataFrame(
        {
            "ori9": ["DCFBIWA00", "VADPP0000", "DCMTP0000"],
            "offense": ["murder", "larceny", "robbery"],
            "estimated_count": [41.0, 31.0, 301.0],
        }
    )
    out = exclude_national_hq_rows(rows, registry=registry)
    assert "DCFBIWA00" not in set(out["ori9"])
    assert out[out["ori9"].eq("VADPP0000")].equals(rows.iloc[[1]])
    assert out[out["ori9"].eq("DCMTP0000")].equals(rows.iloc[[2]])


def test_mass_audit_names_the_existing_state_overlap_landing():
    census = pd.DataFrame(
        {
            "ori9": ["DCFBIWA00"],
            "agency_name": ["FEDERAL BUREAU OF INVESTIGATION"],
            "federal_census_signal": ["federal_ori_class_and_name"],
        }
    )
    estimates = pd.DataFrame(
        {
            "ori9": ["DCFBIWA00"],
            "offense": ["murder"],
            "estimated_count": [41.0],
            "reported_count_current": [41.0],
        }
    )
    crosswalk = pd.DataFrame(
        {
            "ori9": ["DCFBIWA00"],
            "state_fips": ["11"],
            "jurisdiction_id": ["11:statewide_overlap_layer"],
            "weight": [1.0],
        }
    )
    audit = build_federal_ori_mass_audit(
        census=census,
        agency_estimates=estimates,
        crosswalk=crosswalk,
        registry=_registry([("DCFBIWA00", "national_hq")]),
    )
    row = audit.iloc[0]
    assert row["jurisdiction_id"] == "11:statewide_overlap_layer"
    assert float(row["landed_estimated_count"]) == 41.0
    assert bool(row["excluded_from_jurisdiction_targets"])
