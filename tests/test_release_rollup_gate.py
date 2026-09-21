"""The release validator's rollup section: conservation, count-derivation, dictionary currency.

The gate is exercised against a MINIATURE BUT REAL EDITION -- the rollup table is produced by
`rollups.roll_up`, its conservation report by `rollups.conservation_report` and its field
dictionary by `field_dictionary.build_dictionary`, so a passing test means the gate accepts what
the producers actually emit rather than what this file thinks they emit.

Each failure test then breaks exactly one thing in that edition and asserts the gate names it:
a lost count, a residual that stops being named, a rate that is no longer the ratio of the sums,
an index that no longer divides by the national rate published beside it, a value published under
a false publication flag, a column the dictionary does not describe, and a dictionary that
describes a column the table no longer carries. A gate that cannot fail is not a gate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from crimerisk.composites import (
    COMMON_DENOMINATOR_COLUMN,
    CompositeRuntime,
    load_severity_weights,
    severity_weights_path,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.field_dictionary import SurfaceEntry, build_dictionary
from crimerisk.paths import RepoPaths
from crimerisk.rollups import (
    CBSA,
    COUNTY,
    conservation_report,
    coverage_universe,
    expected_count_column,
    primary_denominator_column,
    primary_national_rate_column,
    resident_national_rate_column,
    roll_up,
    rollup_filename,
    source_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
VALIDATOR_PATH = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"


def _validator():
    spec = importlib.util.spec_from_file_location("validate_release_outputs", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = _validator()


def test_promoted_manifest_selects_smoothed_allocation_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promoted release keeps its manifest under the public build filename."""
    output_dir = tmp_path / "state" / "output"
    controls_dir = tmp_path / "state" / "controls"
    output_dir.mkdir(parents=True)
    controls_dir.mkdir(parents=True)
    (output_dir / "crimerisk_output_build_2024.json").write_text(
        json.dumps({"resolved_config": {"control_surface": "smoothed"}})
    )
    pd.DataFrame(
        [
            {
                "state_fips": "01",
                "jurisdiction_id": "01:municipal:place:A",
                "offense": "robbery",
                "smoothed_count": 12.5,
            }
        ]
    ).to_parquet(
        controls_dir / "jurisdiction_controls_smoothed_2024.parquet", index=False
    )
    monkeypatch.setattr(VALIDATOR, "REPO_ROOT", tmp_path)

    issues: list[str] = []
    controls, summary = VALIDATOR._allocation_control_targets(
        output_dir=output_dir, issues=issues
    )

    assert issues == []
    assert summary["control_surface"] == "smoothed"
    assert summary["target_column"] == "smoothed_count"
    assert controls is not None
    assert controls.loc[0, "control_target"] == pytest.approx(12.5)

FLOORS = {
    "eb_hard_min_denominator": 1.0,
    "non_residential_household_floor": 10.0,
    "person_exposure_denominator_floor": 50.0,
    "mvt_vehicle_exposure_denominator_floor": 50.0,
    "zero_resident_opportunity_rate_floor": 500.0,
}
DENOMINATOR_TYPE = {
    "murder": "exposure",
    "rape": "exposure",
    "robbery": "exposure",
    "aggravated_assault": "exposure",
    "burglary": "premises",
    "larceny": "exposure",
    "motor_vehicle_theft": "vehicles",
}
NATIONAL_RATE = {offense: 100.0 + 10.0 * index for index, offense in enumerate(OFFENSES_7)}
RESIDENT_NATIONAL_RATE = {offense: 90.0 + 10.0 * index for index, offense in enumerate(OFFENSES_7)}
MINIMAL_MANIFEST = {
    "year": 2024,
    "resolved_config": {
        "exposure_ensemble": {
            "enabled": True,
            "semantics": "opportunity_normalized_intensity_not_person_time_risk",
            "normalizer_id_by_offense": {offense: f"{offense}_v1" for offense in OFFENSES_7},
        }
    },
}


def _surface() -> pd.DataFrame:
    rows = []
    for index, geoid in enumerate(
        ["010010201001", "010010201002", "010030301001", "040010101001", "040010101002"]
    ):
        row: dict[str, object] = {
            "block_group_geoid": geoid,
            "population_2024": 1000 + 100 * index,
            "households_total": 400 + 10 * index,
            "land_area_sq_mi": 2.0 + index,
            COMMON_DENOMINATOR_COLUMN: 1000 + 100 * index,
        }
        for offense_index, offense in enumerate(OFFENSES_7):
            row[expected_count_column(offense)] = 1.0 + index + offense_index
            row[primary_denominator_column(offense)] = 2000.0 + 100 * index + 10 * offense_index
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["block_group_geoid"] = frame["block_group_geoid"].astype("string")
    frame["state_fips"] = frame["block_group_geoid"].str.slice(0, 2)
    for column, value in FLOORS.items():
        frame[column] = float(value)
    for offense in OFFENSES_7:
        frame[f"primary_denominator_type_{offense}"] = DENOMINATOR_TYPE[offense]
        frame[primary_national_rate_column(offense)] = NATIONAL_RATE[offense]
        frame[resident_national_rate_column(offense)] = RESIDENT_NATIONAL_RATE[offense]
    return frame[source_columns()]


def _crosswalk(surface: pd.DataFrame, geo, *, drop: str | None = None) -> pd.DataFrame:
    ids = surface["block_group_geoid"]
    frame = pd.DataFrame(
        {
            "block_group_geoid": ids,
            geo.id_column: ids.str.slice(0, 5),
            "weight": 1.0,
        }
    )
    if geo.name_column is not None:
        frame[geo.name_column] = f"Test {geo.key}"
    if drop is not None:
        frame = frame[frame["block_group_geoid"] != drop].reset_index(drop=True)
    return frame


def _runtime() -> CompositeRuntime:
    path = severity_weights_path(PATHS)
    return CompositeRuntime(weights=load_severity_weights(path), weights_path=path, year=2024)


def _build_edition(root: Path, *, geo=COUNTY, drop_from_crosswalk: str | None = None) -> Path:
    """A real miniature edition: real rollup, real conservation report, real field dictionary."""
    edition_dir = root / "2024A-annual"
    (edition_dir / "rollups").mkdir(parents=True, exist_ok=True)

    surface = _surface()
    surface_path = edition_dir / "crimerisk_block_group_2024_ags_core.parquet"
    surface.to_parquet(surface_path, index=False)

    crosswalk = _crosswalk(surface, geo, drop=drop_from_crosswalk)
    rollup = roll_up(surface, geo=geo, crosswalk=crosswalk, composites=_runtime())
    rollup_path = edition_dir / "rollups" / rollup_filename(geo, year=2024)
    rollup.to_parquet(rollup_path, index=False)
    report = conservation_report(surface, geo=geo, crosswalk=crosswalk, rollup=rollup)

    universe = coverage_universe(surface)
    rollup_summary = {
        "version": "exact_rollups_v1",
        "year": 2024,
        "coverage_universe": universe,
        "geographies": {
            geo.key: {
                "path": str(rollup_path),
                "rows": int(len(rollup)),
                "columns": int(rollup.shape[1]),
                "source": geo.source,
                "overlap_weighted": not geo.nests_in_block_group,
                "conservation": report,
            }
        },
        "conserved": bool(report["conserved"]),
    }
    (edition_dir / "rollups" / "rollup_summary.json").write_text(
        json.dumps(rollup_summary, indent=2, sort_keys=True, default=str) + "\n"
    )

    markdown, _ = build_dictionary(
        surfaces=[
            SurfaceEntry(
                label="Block group surface",
                geography="block_group",
                path=surface_path,
                rows=int(len(surface)),
            ),
            SurfaceEntry(
                label=f"{geo.key.upper()} rollup",
                geography=geo.key,
                path=rollup_path,
                rows=int(len(rollup)),
            ),
        ],
        manifest=MINIMAL_MANIFEST,
        coverage_universe=universe,
        edition_id="2024A-annual",
        year=2024,
    )
    (edition_dir / "FIELD_DICTIONARY.md").write_text(markdown)

    (edition_dir / "edition.json").write_text(
        json.dumps(
            {
                "edition_id": "2024A-annual",
                "year": 2024,
                "target_year": 2024,
                "coverage_universe": universe,
                "source_surfaces": {"block_group": {"path": str(surface_path)}},
                "geoparquet": [],
                "rollups": {geo.key: {"rows": int(len(rollup)), "conserved": True}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return edition_dir


def _run(edition_dir: Path) -> tuple[dict, list[str]]:
    issues: list[str] = []
    summary = VALIDATOR._check_edition_rollups(edition_dir=edition_dir, issues=issues)
    return summary, issues


@pytest.fixture()
def edition(tmp_path: Path) -> Path:
    return _build_edition(tmp_path)


def _rollup_path(edition_dir: Path) -> Path:
    return edition_dir / "rollups" / rollup_filename(COUNTY, year=2024)


def _edit_rollup(edition_dir: Path, mutate) -> None:
    path = _rollup_path(edition_dir)
    frame = pd.read_parquet(path)
    mutate(frame)
    frame.to_parquet(path, index=False)


# --- the gate is only emitted when an edition is under validation --------------------------------


def test_build_summary_takes_an_optional_edition_dir_and_defaults_to_none():
    import inspect

    parameter = inspect.signature(VALIDATOR.build_summary).parameters["edition_dir"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_cli_exposes_edition_dir():
    source = VALIDATOR_PATH.read_text()
    assert '"--edition-dir"' in source
    assert '"--holdout-run"' in source
    assert "edition_dir=args.edition_dir" in source
    assert "holdout_run=bool(args.holdout_run)" in source


# --- the happy path ----------------------------------------------------------------------------


def test_a_real_miniature_edition_passes_the_gate(edition: Path):
    summary, issues = _run(edition)
    assert issues == []
    assert summary["present"] is True
    assert summary["edition_id"] == "2024A-annual"
    assert summary["geographies"] == ["county"]
    assert summary["conservation"]["county"]["max_abs_conservation_difference"] < 1e-9
    assert summary["rates_from_counts"]["county"]["publishable_lane_values_checked"] > 0
    assert summary["field_dictionary"]["surfaces"]["county"]["undocumented"] == []
    assert summary["field_dictionary"]["surfaces"]["block_group"]["stale"] == []


def test_the_gate_re_derives_conservation_against_the_edition_s_own_surface(edition: Path):
    summary, _ = _run(edition)
    entry = summary["conservation"]["county"]
    surface = pd.read_parquet(edition / "crimerisk_block_group_2024_ags_core.parquet")
    for offense in OFFENSES_7:
        national = float(surface[expected_count_column(offense)].sum())
        assert entry["offenses"][offense]["national"] == pytest.approx(national)
        assert entry["offenses"][offense]["named_residual"] == 0.0


def test_uncovered_mass_passes_only_while_it_stays_a_named_residual(tmp_path: Path):
    # CBSAs do not cover the country. A crosswalk that places four of five block groups leaves the
    # fifth's counts outside the metro/micro universe, and the edition conserves BECAUSE the
    # summary names that residual.
    edition = _build_edition(tmp_path, geo=CBSA, drop_from_crosswalk="040010101002")
    summary, issues = _run(edition)
    assert issues == []
    entry = summary["conservation"]["cbsa"]
    assert entry["block_groups_outside_universe"] == 1
    assert all(entry["offenses"][offense]["named_residual"] > 0.0 for offense in OFFENSES_7)


def test_a_geography_that_does_not_cover_the_universe_must_account_for_what_it_leaves_out(
    tmp_path: Path,
):
    edition = _build_edition(tmp_path, geo=CBSA)
    _, issues = _run(edition)
    assert any(
        "does not cover the universe but reports no block groups outside it" in issue
        for issue in issues
    )


# --- conservation failures ------------------------------------------------------------------------


def test_a_lost_count_fails_the_gate(edition: Path):
    _edit_rollup(edition, lambda frame: frame.__setitem__(expected_count_column("robbery"), 0.0))
    _, issues = _run(edition)
    assert any("expected_count_robbery does not conserve" in issue for issue in issues)


def test_a_residual_that_stops_being_named_fails_the_gate(tmp_path: Path):
    edition = _build_edition(tmp_path, drop_from_crosswalk="040010101002")
    path = edition / "rollups" / "rollup_summary.json"
    summary = json.loads(path.read_text())
    offenses = summary["geographies"]["county"]["conservation"]["offenses"]
    for offense in OFFENSES_7:
        offenses[offense]["outside_universe_total"] = 0.0
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _, issues = _run(edition)
    assert sum("does not conserve" in issue for issue in issues) == len(OFFENSES_7)


def test_a_summary_that_misstates_its_own_rollup_total_fails_the_gate(edition: Path):
    path = edition / "rollups" / "rollup_summary.json"
    summary = json.loads(path.read_text())
    summary["geographies"]["county"]["conservation"]["offenses"]["larceny"]["rollup_total"] = 1.0
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _, issues = _run(edition)
    assert any("but the published table sums to" in issue for issue in issues)


def test_a_universe_covering_geography_may_not_carry_a_residual(edition: Path):
    path = edition / "rollups" / "rollup_summary.json"
    summary = json.loads(path.read_text())
    summary["geographies"]["county"]["conservation"]["offenses"]["murder"][
        "outside_universe_total"
    ] = 5.0
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _, issues = _run(edition)
    assert any("covers the universe but names a" in issue for issue in issues)


def test_a_row_count_that_does_not_match_the_published_table_fails_the_gate(edition: Path):
    path = edition / "rollups" / "rollup_summary.json"
    summary = json.loads(path.read_text())
    summary["geographies"]["county"]["rows"] = 99
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _, issues = _run(edition)
    assert any("the summary claims 99 rows" in issue for issue in issues)


# --- the assertion set is the edition's own contract ----------------------------------------------


def test_two_disagreeing_declarations_fail_rather_than_one_being_chosen(edition: Path):
    path = edition / "edition.json"
    manifest = json.loads(path.read_text())
    manifest["rollups"]["zcta"] = {"rows": 0}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _, issues = _run(edition)
    assert any("cannot be validated against two different contracts" in issue for issue in issues)


def test_a_declared_rollup_whose_table_is_absent_fails_the_gate(edition: Path):
    _rollup_path(edition).unlink()
    _, issues = _run(edition)
    assert any("is not present" in issue for issue in issues)


def test_an_edition_without_a_rollup_summary_fails_the_gate(edition: Path):
    (edition / "rollups" / "rollup_summary.json").unlink()
    summary, issues = _run(edition)
    assert summary["present"] is False
    assert any("no rollups/rollup_summary.json" in issue for issue in issues)


# --- rates recomputed from counts ------------------------------------------------------------------


def test_a_rate_that_is_not_the_ratio_of_the_sums_fails_the_gate(edition: Path):
    _edit_rollup(edition, lambda frame: frame.__setitem__("rate_larceny_primary", 1.0))
    _, issues = _run(edition)
    assert any("rate_larceny_primary is not recomputable" in issue for issue in issues)


def test_an_index_that_does_not_divide_by_its_published_national_rate_fails_the_gate(edition: Path):
    def mutate(frame: pd.DataFrame) -> None:
        frame["index_burglary_primary"] = frame["index_burglary_primary"] * 1.01

    _edit_rollup(edition, mutate)
    _, issues = _run(edition)
    assert any("index_burglary_primary is not recomputable" in issue for issue in issues)


def test_a_resident_lane_rate_is_checked_against_the_common_denominator(edition: Path):
    _edit_rollup(edition, lambda frame: frame.__setitem__("rate_murder_resident", 7.0))
    _, issues = _run(edition)
    assert any("rate_murder_resident is not recomputable" in issue for issue in issues)


def test_a_withheld_value_must_be_null_not_a_number(edition: Path):
    def mutate(frame: pd.DataFrame) -> None:
        frame["primary_index_publishable_robbery"] = False

    _edit_rollup(edition, mutate)
    _, issues = _run(edition)
    assert any("a withheld value must be null" in issue for issue in issues)


def test_an_aggregate_that_is_not_the_sum_of_the_published_counts_fails_the_gate(edition: Path):
    _edit_rollup(edition, lambda frame: frame.__setitem__("expected_count_total", 1.0))
    _, issues = _run(edition)
    assert any("expected_count_total is not recomputable" in issue for issue in issues)


def test_a_density_that_is_not_counts_over_land_area_fails_the_gate(edition: Path):
    _edit_rollup(edition, lambda frame: frame.__setitem__("crime_density_murder", 3.0))
    _, issues = _run(edition)
    assert any("crime_density_murder is not recomputable" in issue for issue in issues)


# --- the field dictionary is current --------------------------------------------------------------


def test_a_published_column_the_dictionary_does_not_describe_fails_the_gate(edition: Path):
    text = (edition / "FIELD_DICTIONARY.md").read_text()
    text = "\n".join(
        line for line in text.splitlines() if not line.startswith("| `index_larceny_primary` |")
    )
    (edition / "FIELD_DICTIONARY.md").write_text(text + "\n")
    _, issues = _run(edition)
    assert any("are absent from the field dictionary" in issue for issue in issues)


def test_a_dictionary_describing_a_column_the_table_lost_fails_the_gate(edition: Path):
    _edit_rollup(edition, lambda frame: frame.drop(columns=["crime_density_total"], inplace=True))
    _, issues = _run(edition)
    assert any(
        "documents 1 columns the published table does not carry" in issue for issue in issues
    )


def test_a_dictionary_whose_declared_column_count_is_stale_fails_the_gate(edition: Path):
    path = edition / "FIELD_DICTIONARY.md"
    lines = path.read_text().splitlines()
    for position, line in enumerate(lines):
        if line.startswith("| COUNTY rollup | county |"):
            parts = line.split(" | ")
            parts[2] = "9999"
            lines[position] = " | ".join(parts)
    path.write_text("\n".join(lines) + "\n")
    _, issues = _run(edition)
    assert any("surface table claims 9999 columns" in issue for issue in issues)


def test_an_edition_with_no_dictionary_fails_the_gate(edition: Path):
    (edition / "FIELD_DICTIONARY.md").unlink()
    summary, issues = _run(edition)
    assert summary["field_dictionary"]["present"] is False
    assert any("ships no field dictionary" in issue for issue in issues)


def test_geometry_is_not_treated_as_an_undocumented_field(edition: Path, tmp_path: Path):
    # The packaging step adds `geometry`; the dictionary is generated from the surface without it,
    # and that difference must not read as staleness.
    documented, _ = VALIDATOR._documented_dictionary((edition / "FIELD_DICTIONARY.md").read_text())
    assert VALIDATOR.EDITION_GEOMETRY_COLUMN not in documented["block_group"]
    _, issues = _run(edition)
    assert not any("geometry" in issue for issue in issues)


# --- the built edition ------------------------------------------------------------------------------

BUILT_EDITION = REPO_ROOT / "state" / "editions" / "2024A-annual"


@pytest.mark.skipif(
    not (BUILT_EDITION / "rollups" / "rollup_summary.json").exists(),
    reason="2024A-annual is not packaged",
)
def test_the_packaged_2024a_annual_edition_passes_the_gate():
    summary, issues = _run(BUILT_EDITION)
    assert issues == []
    assert summary["geographies"] == ["cbsa", "county", "state", "zcta"]
    assert set(summary["field_dictionary"]["surfaces"]) == {
        "block_group",
        "tract",
        "county",
        "cbsa",
        "zcta",
        "state",
    }


def test_the_conservation_tolerance_is_a_floor_widened_by_ulp(edition: Path):
    # 1e-9 is the floor and binds on small totals; the ULP term only relaxes it where a float64
    # sum of that magnitude cannot be more exact than it already is.
    assert VALIDATOR._conservation_tolerance(1.0e4) == pytest.approx(1e-9)
    assert VALIDATOR._conservation_tolerance(5.4e6) > 1e-9
    shutil.rmtree(edition, ignore_errors=True)
