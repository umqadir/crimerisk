"""Versioned editions: the id grammar, the two edition types, the layout, and GeoParquet.

The properties under test: an edition id parses into year, sequence and type or is rejected; the
quarterly-provisional type is DEFINED and refuses to build with its own stated reason; the layout
is fixed and self-describing; the manifest records both defined types plus a SHA-256 for every
file; and the GeoParquet writer joins real TIGER geometry, keeps the source CRS, and refuses to
publish a row without a boundary.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from crimerisk.editions import (
    ANNUAL,
    EDITION_LAYOUT_VERSION,
    EDITION_TYPES,
    GEOMETRY_COLUMN,
    PROVISIONAL_DIRECTORIES,
    PROVISIONAL_DIRNAME,
    QUARTERLY_PROVISIONAL,
    TIGER_CRS_EPSG,
    TIGER_SPECS,
    EditionId,
    EditionManifest,
    EditionTypeMismatch,
    EditionTypeNotBuildable,
    copy_into,
    edition_layout,
    editions_dir,
    file_record,
    parse_edition_id,
    require_buildable,
    require_edition_type,
    sha256,
    write_geoparquet,
)
from crimerisk.paths import RepoPaths

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
DC_BG_ZIP = PATHS.data_dir / "tiger_bg" / "tl_2020_11_bg.zip"


# --- the id grammar ----------------------------------------------------------------------------


def test_annual_edition_id_parses():
    edition = parse_edition_id("2024A-annual")
    assert edition == EditionId(year=2024, sequence="A", type_key="annual")
    assert str(edition) == "2024A-annual"
    assert edition.edition_type is ANNUAL


def test_quarterly_sequence_is_a_quarter_token():
    edition = parse_edition_id("2025Q3-quarterly-provisional")
    assert edition.year == 2025
    assert edition.sequence == "Q3"
    assert edition.edition_type is QUARTERLY_PROVISIONAL


@pytest.mark.parametrize(
    "value", ["2024-annual", "24A-annual", "2024A", "2024A-monthly", "2024Q5-quarterly-provisional"]
)
def test_bad_edition_ids_are_rejected(value: str):
    with pytest.raises(ValueError):
        parse_edition_id(value)


# --- the two edition types -----------------------------------------------------------------------


def test_two_edition_types_are_defined_and_both_build():
    assert set(EDITION_TYPES) == {"annual", "quarterly-provisional"}
    assert ANNUAL.buildable
    assert QUARTERLY_PROVISIONAL.buildable
    require_buildable(parse_edition_id("2024A-annual"))
    require_buildable(parse_edition_id("2025Q3-quarterly-provisional"))


def test_a_type_that_stops_being_buildable_refuses_with_its_own_reason():
    """The guard still exists and still names the reason; only the two shipped types satisfy it."""
    unbuildable = replace(ANNUAL, key="annual", buildable=False, note="the reason it cannot")
    with pytest.raises(EditionTypeNotBuildable) as excinfo:
        with mock.patch.dict(EDITION_TYPES, {"annual": unbuildable}):
            require_buildable(parse_edition_id("2024A-annual"))
    message = str(excinfo.value)
    assert "defined but not buildable" in message
    assert "the reason it cannot" in message


def test_neither_builder_will_build_the_other_types_edition():
    require_edition_type(parse_edition_id("2024A-annual"), ANNUAL.key, builder="package_edition.py")
    with pytest.raises(EditionTypeMismatch) as excinfo:
        require_edition_type(
            parse_edition_id("2025Q3-quarterly-provisional"),
            ANNUAL.key,
            builder="package_edition.py",
        )
    assert "package_edition.py builds 'annual' editions" in str(excinfo.value)


def test_provisional_type_keeps_its_disclaimers_now_that_it_builds():
    note = QUARTERLY_PROVISIONAL.note
    assert "preliminary" in note and "as-of" in note and "revision history" in note
    assert QUARTERLY_PROVISIONAL.cadence == "quarterly"
    assert "no finality claim" in note.lower()
    assert "not comparable to the annual accounting surface" in note


# --- layout ---------------------------------------------------------------------------------------


def test_layout_is_fixed_and_under_the_editions_root(tmp_path: Path):
    edition = parse_edition_id("2024A-annual")
    layout = edition_layout(PATHS, edition, root=tmp_path)
    assert layout.root == tmp_path / "2024A-annual"
    assert layout.surfaces_dir.name == "surfaces"
    assert layout.rollups_dir.name == "rollups"
    assert layout.dictionary_path.name == "FIELD_DICTIONARY.md"
    assert layout.manifest_path.name == "edition.json"
    assert layout.build_manifest_path.name == "build_manifest.json"
    assert layout.validation_summary_path.name == "validation_summary.json"
    layout.prepare()
    assert layout.surfaces_dir.is_dir() and layout.rollups_dir.is_dir()
    assert not layout.provisional_dir.exists()


def test_a_provisional_layout_does_not_advertise_rollups_it_never_computed(tmp_path: Path):
    edition = parse_edition_id("2025Q2-quarterly-provisional")
    layout = edition_layout(PATHS, edition, root=tmp_path)
    assert layout.root == tmp_path / "2025Q2-quarterly-provisional"
    assert layout.provisional_dir.name == PROVISIONAL_DIRNAME
    assert layout.readme_path.name == "README.md"
    layout.prepare(directories=PROVISIONAL_DIRECTORIES)
    assert layout.provisional_dir.is_dir()
    assert not layout.rollups_dir.exists()
    assert not layout.surfaces_dir.exists()


def test_default_editions_root_is_under_state():
    assert editions_dir(PATHS) == PATHS.state_dir / "editions"


# --- the manifest ------------------------------------------------------------------------------------


def test_manifest_records_both_defined_types_and_every_file(tmp_path: Path):
    edition = parse_edition_id("2024A-annual")
    layout = edition_layout(PATHS, edition, root=tmp_path)
    layout.prepare()
    payload_file = layout.rollups_dir / "example.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(payload_file, index=False)

    manifest = EditionManifest(
        edition_id=edition,
        layout=layout,
        input_dir=tmp_path / "input",
        payload={"files": [file_record(payload_file, root=layout.root)]},
    )
    written = json.loads(manifest.write().read_text())
    assert written["layout_version"] == EDITION_LAYOUT_VERSION
    assert written["edition_id"] == "2024A-annual"
    assert written["edition_type"]["key"] == "annual"
    assert set(written["defined_edition_types"]) == {"annual", "quarterly-provisional"}
    assert written["defined_edition_types"]["quarterly-provisional"]["buildable"] is True
    assert (
        written["defined_edition_types"]["quarterly-provisional"]["surface"]
        == "Surface 3 - provisional nowcast"
    )
    assert written["files"][0]["path"] == "rollups/example.parquet"
    assert len(written["files"][0]["sha256"]) == 64


def test_sha256_and_copy_are_content_faithful(tmp_path: Path):
    source = tmp_path / "a.txt"
    source.write_text("hello")
    destination = copy_into(source, tmp_path / "nested" / "b.txt")
    assert destination.read_text() == "hello"
    assert sha256(source) == sha256(destination)


# --- GeoParquet ------------------------------------------------------------------------------------


def test_tiger_specs_cover_both_published_surfaces():
    assert set(TIGER_SPECS) == {"block_group", "tract"}
    assert TIGER_SPECS["block_group"]["id_column"] == "block_group_geoid"
    assert TIGER_SPECS["tract"]["id_column"] == "tract_id"


@pytest.mark.skipif(not DC_BG_ZIP.exists(), reason="TIGER block group file not present")
def test_geoparquet_joins_real_geometry_and_keeps_the_source_crs(tmp_path: Path):
    import geopandas as gpd

    geoids = (
        gpd.read_file(f"zip://{DC_BG_ZIP}", columns=["GEOID"])["GEOID"].astype(str).head(20).tolist()
    )
    surface_path = tmp_path / "surface.parquet"
    pd.DataFrame(
        {"block_group_geoid": geoids, "expected_count_robbery": [1.0] * len(geoids)}
    ).to_parquet(surface_path, index=False)

    out_path = tmp_path / "surface.geo.parquet"
    record = write_geoparquet(
        paths=PATHS, surface_path=surface_path, geography="block_group", out_path=out_path
    )
    assert record["rows"] == len(geoids)
    assert record["crs"] == f"EPSG:{TIGER_CRS_EPSG}"

    frame = gpd.read_parquet(out_path)
    assert frame.crs.to_epsg() == TIGER_CRS_EPSG
    assert frame.geometry.name == GEOMETRY_COLUMN
    assert frame.geometry.notna().all()

    metadata = __import__("pyarrow.parquet", fromlist=["parquet"]).ParquetFile(
        str(out_path)
    ).schema_arrow.metadata
    geo = json.loads(metadata[b"geo"].decode())
    assert geo["primary_column"] == GEOMETRY_COLUMN
    assert geo["columns"][GEOMETRY_COLUMN]["encoding"] == "WKB"


@pytest.mark.skipif(not DC_BG_ZIP.exists(), reason="TIGER block group file not present")
def test_geoparquet_refuses_a_row_with_no_boundary(tmp_path: Path):
    surface_path = tmp_path / "surface.parquet"
    pd.DataFrame(
        {"block_group_geoid": ["110019999999"], "expected_count_robbery": [1.0]}
    ).to_parquet(surface_path, index=False)
    with pytest.raises(ValueError, match="no TIGER geometry"):
        write_geoparquet(
            paths=PATHS,
            surface_path=surface_path,
            geography="block_group",
            out_path=tmp_path / "out.geo.parquet",
        )
