"""The static lookup: sharding by GEOID prefix, exact round-trip, and the no-server contract.

The properties under test are the ones the construction promises a client: the shard filename is
derivable from the GEOID alone (so a lookup is one fetch and no index scan); every shard fits the
byte budget, with the prefix length MEASURED rather than assumed; the field list is the row's
positional schema and is never repeated; a value read back out of a shard is bit-for-bit the value
in the Parquet column; and a headline field a support does not carry is named absent rather than
emitted null.

The synthetic cases make the arithmetic and the sharding checkable by hand. The real-edition
round-trip below runs against `state/editions/2024A-annual` when it is present, and is the check
the brief asks for: five random GEOIDs per geography, every published field, exact equality.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

import pandas as pd
import pytest

from crimerisk.composites import COUNT_FIRST_AGGREGATE_INDEX_FIELDS
from crimerisk.crime import OFFENSES_7
from crimerisk.lookup import (
    CHECKSUMS_FILENAME,
    CLIENT_PATTERN,
    HEADLINE_FIELD_FAMILIES,
    LOOKUP_VERSION,
    MANIFEST_FILENAME,
    SHARD_BYTE_BUDGET,
    build_lookup_index,
    composite_fields,
    headline_fields,
    lookup_geography,
    lookup_geoid,
    published_fields,
    shard_path,
    sha256_text,
)
from crimerisk.rollups import ZCTA_ANTI_ZIP_CAVEAT

REPO_ROOT = Path(__file__).resolve().parents[1]
EDITION_DIR = REPO_ROOT / "state" / "editions" / "2024A-annual"

INDEX_FIELDS = [f"index_{offense}_primary" for offense in OFFENSES_7]
TIER_FIELDS = [f"reliability_tier_{offense}" for offense in OFFENSES_7]


def _county_table(rows: int = 40) -> pd.DataFrame:
    """A synthetic county rollup: five states, eight counties each, headline fields only."""
    records = []
    for index in range(rows):
        state = f"{(index % 5) + 1:02d}"
        county = f"{state}{index:03d}"
        record: dict[str, object] = {"county_geoid": county, "county_name": f"County {index}"}
        for offset, field in enumerate(INDEX_FIELDS):
            record[field] = 100.0 + index + offset / 7.0
        for offset, field in enumerate(COUNT_FIRST_AGGREGATE_INDEX_FIELDS):
            record[field] = 50.0 + index + offset / 3.0
        records.append(record)
    return pd.DataFrame(records)


def _block_group_table(rows: int = 30) -> pd.DataFrame:
    records = []
    for index in range(rows):
        geoid = f"01001{index:07d}"
        record: dict[str, object] = {"block_group_geoid": geoid, "special_use_type": "residential"}
        for field in INDEX_FIELDS:
            record[field] = float(index)
        for field in COUNT_FIRST_AGGREGATE_INDEX_FIELDS:
            record[field] = float(index) * 2.0
        for field in TIER_FIELDS:
            record[field] = "high"
        records.append(record)
    return pd.DataFrame(records)


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


# --- the field list ---------------------------------------------------------------------------


def test_headline_fields_are_indexes_composites_tiers_and_type():
    fields = headline_fields()
    assert fields[: len(INDEX_FIELDS)] == INDEX_FIELDS
    for field in composite_fields():
        assert field in fields
    for field in TIER_FIELDS:
        assert field in fields
    assert fields[-1] == "special_use_type"


def test_the_field_list_never_repeats_a_name():
    fields = headline_fields()
    assert len(set(fields)) == len(fields)
    for family in HEADLINE_FIELD_FAMILIES.values():
        assert len(set(family)) == len(family)


def test_composites_are_read_off_the_composite_lane_not_transcribed():
    assert composite_fields() == list(COUNT_FIRST_AGGREGATE_INDEX_FIELDS)


def test_a_support_that_carries_no_tier_names_it_absent_rather_than_publishing_null():
    geo = lookup_geography("county")
    columns = set(_county_table().columns)
    present, absent = published_fields(geo, columns)
    assert present[0] == "county_name"
    assert set(TIER_FIELDS).issubset(set(absent))
    assert "special_use_type" in absent
    assert not set(absent) & set(present)


def test_unknown_geography_is_rejected():
    with pytest.raises(KeyError):
        lookup_geography("precinct")


# --- sharding ---------------------------------------------------------------------------------


def test_shard_filename_is_derivable_from_the_geoid_alone(tmp_path: Path):
    table = _write(_county_table(), tmp_path / "county.parquet")
    manifest = build_lookup_index(
        edition_root=tmp_path, edition_id="2024A-annual", tables={"county": table}
    )
    entry = manifest["geographies"]["county"]
    assert shard_path(manifest, geography="county", geoid="01005") == Path("county") / (
        "01005"[: entry["prefix_length"]] + ".json"
    )
    lookup_dir = tmp_path / "lookup"
    assert (lookup_dir / shard_path(manifest, geography="county", geoid="01005")).exists()


def test_prefix_length_is_the_shortest_that_fits_the_budget(tmp_path: Path):
    table = _write(_county_table(rows=40), tmp_path / "county.parquet")
    manifest = build_lookup_index(
        edition_root=tmp_path, edition_id="2024A-annual", tables={"county": table}, budget=1200
    )
    entry = manifest["geographies"]["county"]
    assert entry["max_shard_bytes"] <= 1200
    # A tighter budget must push the prefix deeper, never drop a row.
    tighter = build_lookup_index(
        edition_root=tmp_path / "tight",
        edition_id="2024A-annual",
        tables={"county": table},
        budget=1000,
    )
    tight_entry = tighter["geographies"]["county"]
    assert tight_entry["prefix_length"] > entry["prefix_length"]
    assert tight_entry["rows"] == entry["rows"] == 40
    assert tight_entry["max_shard_bytes"] <= 1000


def test_every_shard_fits_the_budget_and_every_row_lands_in_one(tmp_path: Path):
    table = _write(_block_group_table(), tmp_path / "bg.parquet")
    manifest = build_lookup_index(
        edition_root=tmp_path, edition_id="2024A-annual", tables={"block_group": table}, budget=4000
    )
    lookup_dir = tmp_path / "lookup"
    seen: set[str] = set()
    for shard in sorted((lookup_dir / "block_group").glob("*.json")):
        assert shard.stat().st_size <= 4000
        payload = json.loads(shard.read_text())
        assert payload["shard"] == shard.stem
        for geoid in payload["rows"]:
            assert geoid.startswith(shard.stem)
            seen.add(geoid)
    assert len(seen) == int(manifest["geographies"]["block_group"]["rows"]) == 30


def test_the_manifest_states_the_scheme_the_client_needs(tmp_path: Path):
    table = _write(_county_table(), tmp_path / "county.parquet")
    build_lookup_index(edition_root=tmp_path, edition_id="2024A-annual", tables={"county": table})
    manifest = json.loads((tmp_path / "lookup" / MANIFEST_FILENAME).read_text())
    assert manifest["lookup_version"] == LOOKUP_VERSION
    assert manifest["edition_id"] == "2024A-annual"
    assert manifest["shard_byte_budget"] == SHARD_BYTE_BUDGET
    assert manifest["client_pattern"] == CLIENT_PATTERN
    entry = manifest["geographies"]["county"]
    assert entry["id_field"] == "county_geoid"
    assert entry["path_template"] == "county/{shard}.json"
    assert entry["prefix_length"] >= 1
    assert set(HEADLINE_FIELD_FAMILIES) == {"per_offense_index", "composite", "tier", "type"}


def test_every_shard_is_hashed_and_the_checksum_file_is_hashed_in_the_manifest(tmp_path: Path):
    table = _write(_county_table(), tmp_path / "county.parquet")
    manifest = build_lookup_index(
        edition_root=tmp_path, edition_id="2024A-annual", tables={"county": table}, budget=1200
    )
    checksums_path = tmp_path / "lookup" / CHECKSUMS_FILENAME
    checksums = json.loads(checksums_path.read_text())
    assert manifest["checksums"]["sha256"] == sha256_text(checksums_path.read_text())
    shards = sorted((tmp_path / "lookup" / "county").glob("*.json"))
    assert len(checksums["shards"]) == len(shards) == manifest["geographies"]["county"]["shards"]
    for shard in shards:
        record = checksums["shards"][f"county/{shard.name}"]
        assert record["sha256"] == sha256_text(shard.read_text())
        assert record["size_bytes"] == shard.stat().st_size


def test_a_duplicate_or_missing_geoid_is_a_build_error(tmp_path: Path):
    frame = _county_table(rows=4)
    frame.loc[1, "county_geoid"] = frame.loc[0, "county_geoid"]
    table = _write(frame, tmp_path / "dup.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        build_lookup_index(
            edition_root=tmp_path, edition_id="2024A-annual", tables={"county": table}
        )


def test_a_table_without_its_geoid_column_is_a_build_error(tmp_path: Path):
    frame = _county_table(rows=4).drop(columns=["county_geoid"])
    table = _write(frame, tmp_path / "nogeoid.parquet")
    with pytest.raises(KeyError, match="county_geoid"):
        build_lookup_index(
            edition_root=tmp_path, edition_id="2024A-annual", tables={"county": table}
        )


# --- the ZCTA caveat travels ---------------------------------------------------------------------


def test_every_zcta_shard_carries_the_anti_zip_caveat_verbatim(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "zcta5": [f"{10000 + index:05d}" for index in range(6)],
            **{field: [1.0] * 6 for field in INDEX_FIELDS},
            **{field: [2.0] * 6 for field in COUNT_FIRST_AGGREGATE_INDEX_FIELDS},
        }
    )
    table = _write(frame, tmp_path / "zcta.parquet")
    manifest = build_lookup_index(
        edition_root=tmp_path, edition_id="2024A-annual", tables={"zcta": table}
    )
    assert manifest["geographies"]["zcta"]["caveat"] == ZCTA_ANTI_ZIP_CAVEAT
    for shard in (tmp_path / "lookup" / "zcta").glob("*.json"):
        assert json.loads(shard.read_text())["caveat"] == ZCTA_ANTI_ZIP_CAVEAT


# --- round-trip -----------------------------------------------------------------------------------


def test_values_round_trip_exactly_including_nulls(tmp_path: Path):
    frame = _county_table(rows=6)
    frame.loc[2, INDEX_FIELDS[0]] = float("nan")
    frame.loc[3, "county_name"] = None
    # A value whose decimal expansion is not exact in binary: the shard must carry the same double.
    frame.loc[4, INDEX_FIELDS[1]] = 1.0 / 3.0
    frame.loc[5, INDEX_FIELDS[2]] = 0.1 + 0.2
    table = _write(frame, tmp_path / "county.parquet")
    manifest = build_lookup_index(
        edition_root=tmp_path, edition_id="2024A-annual", tables={"county": table}
    )
    fields = manifest["geographies"]["county"]["fields"]
    for _, row in frame.iterrows():
        got = lookup_geoid(tmp_path / "lookup", geography="county", geoid=str(row["county_geoid"]))
        assert got is not None
        for field in fields:
            value = row[field]
            if pd.isna(value):
                assert got[field] is None
            elif isinstance(value, str):
                assert got[field] == value
            else:
                assert got[field] == float(value)
                assert repr(got[field]) == repr(float(value))


def test_an_unknown_geoid_returns_nothing_rather_than_a_wrong_row(tmp_path: Path):
    table = _write(_county_table(), tmp_path / "county.parquet")
    build_lookup_index(edition_root=tmp_path, edition_id="2024A-annual", tables={"county": table})
    assert lookup_geoid(tmp_path / "lookup", geography="county", geoid="99999") is None


# --- the built edition ------------------------------------------------------------------------------

EDITION_LOOKUP = EDITION_DIR / "lookup"


def _edition_tables() -> dict[str, Path]:
    edition = json.loads((EDITION_DIR / "edition.json").read_text())
    tables: dict[str, Path] = {}
    rollups = json.loads((EDITION_DIR / "rollups" / "rollup_summary.json").read_text())
    for geography, entry in (rollups.get("geographies") or {}).items():
        tables[geography] = EDITION_DIR / "rollups" / Path(str(entry["path"])).name
    for record in edition.get("geoparquet") or []:
        tables[str(record["geography"])] = EDITION_DIR / str(record["path"])
    for geography, record in (edition.get("source_surfaces") or {}).items():
        tables.setdefault(str(geography), Path(str(record["path"])))
    return tables


@pytest.mark.skipif(
    not (EDITION_LOOKUP / MANIFEST_FILENAME).exists(), reason="2024A-annual lookup not built"
)
def test_built_edition_lookup_round_trips_five_random_geoids_per_geography():
    """The brief's check: five random GEOIDs per support, every field, exact equality."""
    manifest = json.loads((EDITION_LOOKUP / MANIFEST_FILENAME).read_text())
    tables = _edition_tables()
    rng = random.Random(20260806)
    checked = 0
    for geography, entry in sorted(manifest["geographies"].items()):
        path = tables[geography]
        fields = list(entry["fields"])
        frame = pd.read_parquet(path, columns=[entry["id_field"], *fields])
        for position in rng.sample(range(len(frame)), 5):
            row = frame.iloc[position]
            geoid = str(row[entry["id_field"]])
            got = lookup_geoid(EDITION_LOOKUP, geography=geography, geoid=geoid)
            assert got is not None, f"{geography} {geoid} is absent from its shard"
            assert list(got) == fields
            for field in fields:
                value = row[field]
                if pd.isna(value):
                    assert got[field] is None, f"{geography} {geoid} {field}"
                elif isinstance(value, str):
                    assert got[field] == value, f"{geography} {geoid} {field}"
                else:
                    assert got[field] == float(value), f"{geography} {geoid} {field}"
                    assert repr(got[field]) == repr(float(value)), f"{geography} {geoid} {field}"
            checked += 1
    assert checked == 5 * len(manifest["geographies"])


@pytest.mark.skipif(
    not (EDITION_LOOKUP / MANIFEST_FILENAME).exists(), reason="2024A-annual lookup not built"
)
def test_built_edition_shards_all_fit_the_budget():
    manifest = json.loads((EDITION_LOOKUP / MANIFEST_FILENAME).read_text())
    budget = int(manifest["shard_byte_budget"])
    assert int(manifest["totals"]["max_shard_bytes"]) <= budget
    for geography, entry in manifest["geographies"].items():
        shards = list((EDITION_LOOKUP / geography).glob("*.json"))
        assert len(shards) == int(entry["shards"])
        assert max(shard.stat().st_size for shard in shards) <= budget
