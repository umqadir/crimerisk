from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from crimerisk.allocation import _load_overlap_custom_footprints
from crimerisk.service_scopes import (
    attach_service_scope_columns,
    load_custom_footprint_population_by_target,
    load_service_wide_agency_scopes,
)


def _paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(repo_root=tmp_path, state_dir=tmp_path / "state")


def _write_scope(tmp_path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "service_scope_id",
        "canonical_target_ori",
        "source_state_fips",
        "geometry_contributor_ori",
        "allocation_scope",
        "official_source_ref",
        "evidence_artifact",
        "evidence_sha256",
        "scope_note",
    ]
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(
        tmp_path / "configs" / "service_wide_agency_scopes.csv", index=False
    )


def _write_coverage(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(
        tmp_path / "configs" / "service_wide_footprint_coverage.csv", index=False
    )


def _write_resident_coverage(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(
        tmp_path / "configs" / "overlap_custom_footprint_resident_coverage.csv",
        index=False,
    )


def _write_overrides(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["ori", "target_state_fips", "displaces_county_remainder"],
    ).to_csv(tmp_path / "configs" / "overlap_footprint_overrides.csv", index=False)


def test_scope_canonicalizes_geometry_contributors(tmp_path: Path) -> None:
    _write_scope(
        tmp_path,
        [
            {
                "service_scope_id": "system",
                "canonical_target_ori": "AA0000001",
                "source_state_fips": "01",
                "geometry_contributor_ori": contributor,
                "allocation_scope": "service_wide",
                "official_source_ref": "official",
                "evidence_artifact": "evidence.json",
                "evidence_sha256": "a" * 64,
                "scope_note": "reviewed",
            }
            for contributor in ("AA0000001", "BB0000002")
        ],
    )
    frame = pd.DataFrame(
        {
            "ori9": ["AA0000001", "BB0000002", "CC0000003"],
            "state_fips": ["01", "02", "03"],
        }
    )
    result = attach_service_scope_columns(frame, _paths(tmp_path))
    assert result["ori9"].tolist() == ["AA0000001", "AA0000001", "CC0000003"]
    assert result["allocation_scope"].tolist() == [
        "service_wide",
        "service_wide",
        "source_state",
    ]
    assert result["geometry_contributor_ori"].tolist() == [
        "AA0000001",
        "BB0000002",
        "CC0000003",
    ]


def test_scope_rejects_two_target_owners(tmp_path: Path) -> None:
    _write_scope(
        tmp_path,
        [
            {
                "service_scope_id": "system",
                "canonical_target_ori": owner,
                "source_state_fips": "01",
                "geometry_contributor_ori": contributor,
                "allocation_scope": "service_wide",
                "official_source_ref": "official",
                "evidence_artifact": "evidence.json",
                "evidence_sha256": "a" * 64,
                "scope_note": "reviewed",
            }
            for owner, contributor in (("AA0000001", "AA0000001"), ("BB0000002", "BB0000002"))
        ],
    )
    with pytest.raises(ValueError, match="exactly one canonical target"):
        load_service_wide_agency_scopes(_paths(tmp_path))


def test_population_uses_full_service_scope(tmp_path: Path) -> None:
    _write_overrides(
        tmp_path,
        [
            {
                "ori": "AA0000001",
                "target_state_fips": "01",
                "displaces_county_remainder": "TRUE",
            }
        ],
    )
    _write_scope(
        tmp_path,
        [
            {
                "service_scope_id": "nation",
                "canonical_target_ori": "AA0000001",
                "source_state_fips": "01",
                "geometry_contributor_ori": "AA0000001",
                "allocation_scope": "service_wide",
                "official_source_ref": "official",
                "evidence_artifact": "evidence.json",
                "evidence_sha256": "a" * 64,
                "scope_note": "reviewed",
            }
        ],
    )
    _write_coverage(
        tmp_path,
        [
            {
                "service_scope_id": "nation",
                "state_fips": state,
                "block_group_geoid": bg,
                "bg_service_population_coverage_share": population_share,
                "bg_land_area_coverage_share": land_share,
                "coverage_basis": "blocks",
            }
            for state, bg, population_share, land_share in (
                ("01", "010010001001", 1.0, 1.0),
                ("04", "040010001001", 0.5, 0.25),
            )
        ],
    )
    geometry = tmp_path / "state" / "geometry"
    geometry.mkdir(parents=True)
    pd.DataFrame(
        {
            "block_group_geoid": ["010010001001", "040010001001"],
            "total_pop20": [100.0, 80.0],
        }
    ).to_parquet(geometry / "block_group_to_jurisdiction_crosswalk.parquet", index=False)
    result = load_custom_footprint_population_by_target(_paths(tmp_path))
    assert result.to_dict(orient="records") == [
        {"ori9": "AA0000001", "_covered_population": 140.0}
    ]


def test_population_combines_ordinary_own_activity_legacy_and_service_owner(
    tmp_path: Path,
) -> None:
    _write_overrides(
        tmp_path,
        [
            {
                "ori": ori,
                "target_state_fips": state,
                "displaces_county_remainder": "TRUE",
            }
            for ori, state in (
                ("SS0000001", "01"),
                ("RR0000001", "04"),
                ("AA0000001", "03"),
            )
        ],
    )
    _write_scope(
        tmp_path,
        [
            {
                "service_scope_id": "nation",
                "canonical_target_ori": "SS0000001",
                "source_state_fips": "01",
                "geometry_contributor_ori": "SS0000001",
                "allocation_scope": "service_wide",
                "official_source_ref": "official",
                "evidence_artifact": "evidence.json",
                "evidence_sha256": "a" * 64,
                "scope_note": "reviewed",
            }
        ],
    )
    _write_coverage(
        tmp_path,
        [
            {
                "service_scope_id": "nation",
                "state_fips": "01",
                "block_group_geoid": "010010001001",
                "bg_service_population_coverage_share": 0.5,
                "bg_land_area_coverage_share": 0.3,
                "coverage_basis": "service_blocks",
            }
        ],
    )
    _write_resident_coverage(
        tmp_path,
        [
            {
                "ori": "RR0000001",
                "state_fips": "04",
                "block_group_geoid": "040010001001",
                "bg_responsibility_population_coverage_share": 0.25,
                "responsibility_fraction_basis": "resident_blocks",
            }
        ],
    )
    pd.DataFrame(
        [
            {
                "ori": "RR0000001",
                "state_fips": "04",
                "block_group_geoid": "040010001001",
                "weight_share_basis": "resident_population",
                "bg_population_coverage_share": 0.9,
            },
            {
                "ori": "AA0000001",
                "state_fips": "03",
                "block_group_geoid": "030010001001",
                "weight_share_basis": "activity_or_area",
                "bg_population_coverage_share": 0.4,
            },
        ]
    ).to_csv(tmp_path / "configs" / "overlap_custom_footprints.csv", index=False)
    geometry = tmp_path / "state" / "geometry"
    geometry.mkdir(parents=True)
    pd.DataFrame(
        {
            "block_group_geoid": [
                "010010001001",
                "040010001001",
                "030010001001",
            ],
            "total_pop20": [100.0, 200.0, 300.0],
        }
    ).to_parquet(geometry / "block_group_to_jurisdiction_crosswalk.parquet", index=False)

    result = load_custom_footprint_population_by_target(_paths(tmp_path)).set_index("ori9")
    assert result.loc["SS0000001", "_covered_population"] == pytest.approx(50.0)
    assert result.loc["RR0000001", "_covered_population"] == pytest.approx(50.0)
    assert result.loc["AA0000001", "_covered_population"] == pytest.approx(120.0)


def test_population_rejects_cross_lane_duplicate_target_bg(tmp_path: Path) -> None:
    _write_overrides(
        tmp_path,
        [
            {
                "ori": "AA0000001",
                "target_state_fips": "01",
                "displaces_county_remainder": "TRUE",
            }
        ],
    )
    _write_scope(tmp_path, [])
    _write_resident_coverage(
        tmp_path,
        [
            {
                "ori": "AA0000001",
                "state_fips": "01",
                "block_group_geoid": "010010001001",
                "bg_responsibility_population_coverage_share": 0.5,
                "responsibility_fraction_basis": "resident_blocks",
            }
        ],
    )
    pd.DataFrame(
        [
            {
                "ori": "AA0000001",
                "state_fips": "01",
                "block_group_geoid": "010010001001",
                "weight_share_basis": "activity_or_area",
                "bg_population_coverage_share": 0.5,
            }
        ]
    ).to_csv(tmp_path / "configs" / "overlap_custom_footprints.csv", index=False)
    geometry = tmp_path / "state" / "geometry"
    geometry.mkdir(parents=True)
    pd.DataFrame(
        {"block_group_geoid": ["010010001001"], "total_pop20": [100.0]}
    ).to_parquet(geometry / "block_group_to_jurisdiction_crosswalk.parquet", index=False)

    with pytest.raises(ValueError, match="duplicate a target/state/BG key"):
        load_custom_footprint_population_by_target(_paths(tmp_path))


def test_population_excludes_non_displacing_and_out_of_scope_owners(tmp_path: Path) -> None:
    _write_scope(tmp_path, [])
    _write_overrides(
        tmp_path,
        [
            {
                "ori": "AA0000001",
                "target_state_fips": "01",
                "displaces_county_remainder": "FALSE",
            },
            {
                "ori": "AK0000001",
                "target_state_fips": "02",
                "displaces_county_remainder": "TRUE",
            },
        ],
    )
    _write_resident_coverage(
        tmp_path,
        [
            {
                "ori": ori,
                "state_fips": state,
                "block_group_geoid": bg,
                "bg_responsibility_population_coverage_share": 0.5,
                "responsibility_fraction_basis": "resident_blocks",
            }
            for ori, state, bg in (
                ("AA0000001", "01", "010010001001"),
                ("AK0000001", "02", "020010001001"),
            )
        ],
    )
    geometry = tmp_path / "state" / "geometry"
    geometry.mkdir(parents=True)
    pd.DataFrame(
        {"block_group_geoid": ["010010001001"], "total_pop20": [100.0]}
    ).to_parquet(geometry / "block_group_to_jurisdiction_crosswalk.parquet", index=False)

    assert load_custom_footprint_population_by_target(_paths(tmp_path)).empty


def test_population_rejects_missing_in_scope_bg_population(tmp_path: Path) -> None:
    _write_scope(tmp_path, [])
    _write_overrides(
        tmp_path,
        [
            {
                "ori": "AA0000001",
                "target_state_fips": "01",
                "displaces_county_remainder": "TRUE",
            }
        ],
    )
    _write_resident_coverage(
        tmp_path,
        [
            {
                "ori": "AA0000001",
                "state_fips": "01",
                "block_group_geoid": "010010001001",
                "bg_responsibility_population_coverage_share": 0.5,
                "responsibility_fraction_basis": "resident_blocks",
            }
        ],
    )
    geometry = tmp_path / "state" / "geometry"
    geometry.mkdir(parents=True)
    pd.DataFrame(
        {"block_group_geoid": ["010010002001"], "total_pop20": [100.0]}
    ).to_parquet(geometry / "block_group_to_jurisdiction_crosswalk.parquet", index=False)

    with pytest.raises(ValueError, match="require finite BG population"):
        load_custom_footprint_population_by_target(_paths(tmp_path))


def test_loader_normalizes_service_scope_globally(tmp_path: Path) -> None:
    _write_scope(
        tmp_path,
        [
            {
                "service_scope_id": "nation",
                "canonical_target_ori": "AA0000001",
                "source_state_fips": "01",
                "geometry_contributor_ori": "AA0000001",
                "allocation_scope": "service_wide",
                "official_source_ref": "official",
                "evidence_artifact": "evidence.json",
                "evidence_sha256": "a" * 64,
                "scope_note": "reviewed",
            }
        ],
    )
    pd.DataFrame(
        {
            "ori": ["AA0000001", "AA0000001"],
            "state_fips": ["01", "02"],
            "block_group_geoid": ["010010001001", "020010001001"],
            "weight_share": [0.6, 0.4],
            "bg_population_coverage_share": [1.0, 0.0],
            "weight_share_basis": ["service_area_prior", "service_area_prior"],
            "geometry_source_type": ["census", "census"],
            "geometry_source_ref": ["official", "official"],
            "footprint_note": ["reviewed", "reviewed"],
        }
    ).to_csv(tmp_path / "configs" / "overlap_custom_footprints.csv", index=False)
    _write_coverage(
        tmp_path,
        [
            {
                "service_scope_id": "nation",
                "state_fips": state,
                "block_group_geoid": bg,
                "bg_service_population_coverage_share": population_share,
                "bg_land_area_coverage_share": land_share,
                "coverage_basis": "blocks",
            }
            for state, bg, population_share, land_share in (
                ("01", "010010001001", 1.0, 1.0),
                ("02", "020010001001", 0.0, 0.25),
            )
        ],
    )
    result = _load_overlap_custom_footprints(_paths(tmp_path))
    assert result["state_fips"].tolist() == ["01", "02"]
    assert result["source_state_fips"].tolist() == ["01", "01"]
    assert result["weight_share"].sum() == pytest.approx(1.0)
    assert result["bg_population_coverage_share"].tolist() == pytest.approx([1.0, 0.0])
    assert result["bg_land_area_coverage_share"].tolist() == pytest.approx([1.0, 0.25])


def test_navajo_and_ramah_shared_bg_carry_union_coverage() -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "overlap_custom_footprints.csv"
    footprints = pd.read_csv(config, dtype="string")
    shared = footprints[
        footprints["block_group_geoid"].astype("string").str.zfill(12).eq("350069458002")
        & footprints["ori"].isin(["AZ0018900", "NM0170400"])
    ]
    assert set(shared["ori"]) == {"AZ0018900", "NM0170400"}
    coverage = pd.to_numeric(shared["bg_population_coverage_share"])
    assert coverage.nunique() == 1
    assert coverage.iloc[0] == pytest.approx(0.868928296067849)


def test_navajo_service_coverage_retains_partial_and_zero_resident_geometry() -> None:
    root = Path(__file__).resolve().parents[1]
    coverage = pd.read_csv(
        root / "configs" / "service_wide_footprint_coverage.csv", dtype="string"
    )
    navajo = coverage[coverage["service_scope_id"].eq("navajo_nation_police")].copy()
    land = pd.to_numeric(navajo["bg_land_area_coverage_share"])
    population = pd.to_numeric(navajo["bg_service_population_coverage_share"])
    assert len(navajo) == 166
    assert (land < 1.0 - 1e-12).sum() == 59
    assert population.eq(0.0).sum() == 8
    tiny = navajo[navajo["block_group_geoid"].eq("040050013023")]
    assert float(tiny["bg_land_area_coverage_share"].iloc[0]) == pytest.approx(
        18467 / 652752029
    )


def test_navajo_and_ramah_own_population_fractions_reconcile_union() -> None:
    root = Path(__file__).resolve().parents[1]
    bg = "350069458002"
    service = pd.read_csv(
        root / "configs" / "service_wide_footprint_coverage.csv", dtype="string"
    )
    resident = pd.read_csv(
        root / "configs" / "overlap_custom_footprint_resident_coverage.csv", dtype="string"
    )
    footprints = pd.read_csv(root / "configs" / "overlap_custom_footprints.csv", dtype="string")
    navajo = float(
        service.loc[
            service["block_group_geoid"].eq(bg),
            "bg_service_population_coverage_share",
        ].iloc[0]
    )
    ramah = float(
        resident.loc[
            resident["ori"].eq("NM0170400") & resident["block_group_geoid"].eq(bg),
            "bg_responsibility_population_coverage_share",
        ].iloc[0]
    )
    union = float(
        footprints.loc[
            footprints["ori"].eq("NM0170400")
            & footprints["block_group_geoid"].eq(bg),
            "bg_population_coverage_share",
        ].iloc[0]
    )
    assert navajo + ramah == pytest.approx(union)


def test_ordinary_resident_sidecar_matches_every_declared_footprint_row() -> None:
    root = Path(__file__).resolve().parents[1]
    footprints = pd.read_csv(root / "configs" / "overlap_custom_footprints.csv", dtype="string")
    resident = footprints[footprints["weight_share_basis"].eq("resident_population")].copy()
    coverage = pd.read_csv(
        root / "configs" / "overlap_custom_footprint_resident_coverage.csv",
        dtype="string",
    )
    for frame in (resident, coverage):
        frame["state_fips"] = frame["state_fips"].str.zfill(2)
        frame["block_group_geoid"] = frame["block_group_geoid"].str.zfill(12)
    keys = ["ori", "state_fips", "block_group_geoid"]
    joined = resident.merge(coverage, on=keys, how="outer", validate="one_to_one", indicator=True)
    assert len(resident) == len(coverage) == 9786
    assert resident["ori"].nunique() == coverage["ori"].nunique() == 242
    assert joined["_merge"].eq("both").all()
    fraction = pd.to_numeric(joined["bg_responsibility_population_coverage_share"])
    assert fraction.between(0.0, 1.0, inclusive="right").all()
