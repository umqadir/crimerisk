"""The generated field dictionary: every published column described, nothing hand-maintained.

The properties under test are the ones the generator exists to guarantee: every column of every
published surface resolves to a rule (an unmatched column is an error, not an omission); the
semantics strings come from the build's own manifest rather than a transcription; the observed
vocabulary of a small categorical is read off the surface; and the ZCTA anti-ZIP caveat appears
VERBATIM wherever a ZCTA field does.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crimerisk.composites import COMMON_DENOMINATOR_COLUMN, COMMON_DENOMINATOR_ID
from crimerisk.crime import OFFENSES_7
from crimerisk.field_dictionary import (
    DICTIONARY_VERSION,
    MAX_VOCABULARY_CARDINALITY,
    PROVENANCE,
    RULES,
    DictionaryContext,
    FieldSpec,
    SurfaceEntry,
    UndocumentedFieldError,
    _compiled_rules,
    build_dictionary,
    build_field_specs,
    read_schema,
    read_vocabularies,
    render_markdown,
    resolve_context,
    resolve_rule,
)
from crimerisk.rollups import ZCTA_ANTI_ZIP_CAVEAT

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = REPO_ROOT / "state" / "candidates" / "v2-evidence-chain"
BLOCK_GROUP_SURFACE = CANDIDATE_DIR / "crimerisk_block_group_2024_ags_core.parquet"
TRACT_SURFACE = CANDIDATE_DIR / "crimerisk_tract_2024_ags_core.parquet"

MANIFEST = {
    "year": 2024,
    "resolved_config": {
        "exposure_ensemble": {
            "enabled": True,
            "semantics": "opportunity_normalized_intensity_not_person_time_risk",
            "normalizer_version": "exposure_normalizers_v1",
            "normalizer_id_by_offense": {offense: f"{offense}_v1" for offense in OFFENSES_7},
        },
        "uncertainty_layer": {
            "enabled": True,
            "version": "uncertainty_layer_v1",
            "index_breaks": [12.5, 25.0, 100.0],
        },
        "special_use_taxonomy": {"enabled": True, "version": "special_use_taxonomy_v2"},
        "count_first_composites": {"enabled": True, "version": "count_first_composites_v1"},
    },
    "summary": {
        "count_first_composites": {"primary_severity_vector": {"vector_id": "test_vector_v1"}}
    },
}
UNIVERSE = {
    "description": "48 contiguous states + District of Columbia",
    "block_groups": 238193,
    "counties": 3108,
}


def _context() -> DictionaryContext:
    return resolve_context(MANIFEST)


# --- the rule table -------------------------------------------------------------------------


def test_every_rule_names_a_known_provenance():
    for rule in RULES:
        assert rule.provenance_key in PROVENANCE, rule.pattern


def test_offense_token_expands_to_exactly_the_seven_part_one_offences():
    rule = next(rule for rule in RULES if rule.pattern == "expected_count_{offense}")
    regex = rule.regex()
    for offense in OFFENSES_7:
        assert regex.match(f"expected_count_{offense}")
    assert not regex.match("expected_count_arson")
    assert not regex.match("expected_count_total")


def test_year_token_accepts_annual_fields_without_hardcoding_the_release_year():
    compiled = _compiled_rules()
    for column in (
        "population_2024",
        "population_2025",
        "exposure_proxy_2025",
        "vehicle_exposure_2025",
    ):
        resolve_rule(column, compiled)
    with pytest.raises(UndocumentedFieldError):
        resolve_rule("population_latest", compiled)


def test_decennial_group_quarters_inputs_are_documented():
    compiled = _compiled_rules()
    for column in (
        "special_use_gq_total_2020",
        "special_use_gq_institutional_2020",
        "special_use_gq_correctional_2020",
        "special_use_gq_juvenile_2020",
        "special_use_gq_nursing_2020",
        "special_use_gq_college_2020",
        "special_use_gq_military_2020",
        "special_use_gq_other_institutional_2020",
        "special_use_gq_other_noninstitutional_2020",
    ):
        resolve_rule(column, compiled)


def test_an_undescribed_column_is_an_error_not_an_omission():
    with pytest.raises(UndocumentedFieldError, match="matches no field-dictionary rule"):
        resolve_rule("some_new_field", _compiled_rules())


@pytest.mark.skipif(not BLOCK_GROUP_SURFACE.exists(), reason="candidate surface not present")
def test_every_published_block_group_column_is_described():
    compiled = _compiled_rules()
    for column in read_schema(BLOCK_GROUP_SURFACE):
        resolve_rule(column, compiled)


@pytest.mark.skipif(not TRACT_SURFACE.exists(), reason="candidate surface not present")
def test_every_published_tract_column_is_described():
    compiled = _compiled_rules()
    for column in read_schema(TRACT_SURFACE):
        resolve_rule(column, compiled)


# --- semantics come from the build ------------------------------------------------------------


def test_context_reads_the_builds_own_semantics_strings():
    context = _context()
    assert context.normalizer_semantics == "opportunity_normalized_intensity_not_person_time_risk"
    assert context.uncertainty_version == "uncertainty_layer_v1"
    assert context.severity_vector_id == "test_vector_v1"
    assert context.uncertainty_index_breaks == (12.5, 25.0, 100.0)
    assert "exposure_ensemble" in context.lanes


def test_normalizer_semantics_is_spliced_into_the_denominator_and_rate_fields():
    specs = build_field_specs(
        schema={
            "primary_denominator_robbery": "double",
            "primary_denominator_normalizer_id_robbery": "string",
            "rate_robbery_primary": "double",
            COMMON_DENOMINATOR_COLUMN: "int64",
        },
        vocabularies={},
        context=_context(),
    )
    by_name = {spec.name: spec for spec in specs}
    assert "robbery_v1" in by_name["primary_denominator_robbery"].semantics
    assert (
        "opportunity_normalized_intensity_not_person_time_risk"
        in by_name["rate_robbery_primary"].semantics
    )
    assert "robbery_v1" in by_name["primary_denominator_normalizer_id_robbery"].semantics
    assert COMMON_DENOMINATOR_ID in by_name[COMMON_DENOMINATOR_COLUMN].semantics


def test_index_bin_edges_come_from_the_manifest():
    specs = build_field_specs(
        schema={"displayed_index_bin_robbery": "double"}, vocabularies={}, context=_context()
    )
    assert "12.5, 25.0, 100.0" in specs[0].semantics


# --- vocabulary is read off the surface ----------------------------------------------------------


def test_small_categoricals_carry_their_observed_vocabulary(tmp_path: Path):
    path = tmp_path / "surface.parquet"
    pd.DataFrame(
        {
            "estimate_mode_robbery": ["count_derived", "special_use", "count_derived"],
            "level_provenance_text_robbery": [f"sentence {index}" for index in range(3)],
        }
    ).to_parquet(path, index=False)
    vocabularies = read_vocabularies(path, ["estimate_mode_robbery", "level_provenance_text_robbery"], max_cardinality=2)
    assert vocabularies["estimate_mode_robbery"] == ("count_derived", "special_use")
    assert "level_provenance_text_robbery" not in vocabularies


def test_vocabulary_is_rendered_into_the_semantics_cell():
    spec = FieldSpec(
        name="estimate_mode_robbery",
        dtype="string",
        group="per offence: publication",
        semantics="The estimand label.",
        provenance=PROVENANCE["allocation"],
        vocabulary=("count_derived", "special_use"),
    )
    assert "count_derived" in spec.semantics_cell()
    assert "special_use" in spec.semantics_cell()


def test_vocabulary_cardinality_cap_is_declared():
    assert MAX_VOCABULARY_CARDINALITY >= 4


# --- rendering ------------------------------------------------------------------------------------


def _rendered(tmp_path: Path) -> str:
    entries = [
        (
            SurfaceEntry(label="Block group surface", geography="block_group", path=tmp_path / "bg.parquet", rows=3),
            build_field_specs(
                schema={"block_group_geoid": "string", "expected_count_robbery": "double"},
                vocabularies={},
                context=_context(),
            ),
        ),
        (
            SurfaceEntry(label="ZCTA rollup", geography="zcta", path=tmp_path / "zcta.parquet", rows=2),
            build_field_specs(
                schema={"zcta5": "string", "expected_count_robbery": "double"},
                vocabularies={},
                context=_context(),
            ),
        ),
    ]
    return render_markdown(
        surfaces=entries, context=_context(), coverage_universe=UNIVERSE, edition_id="2024A-annual"
    )


def test_rendered_document_carries_the_zcta_caveat_verbatim_twice(tmp_path: Path):
    markdown = _rendered(tmp_path)
    assert markdown.count(ZCTA_ANTI_ZIP_CAVEAT) == 2
    assert "## ZCTA caveat" in markdown


def test_rendered_document_names_the_edition_universe_and_count_derived_rule(tmp_path: Path):
    markdown = _rendered(tmp_path)
    assert "2024A-annual" in markdown
    assert UNIVERSE["description"] in markdown
    assert "rate = 100000 * expected_count / denominator" in markdown
    assert DICTIONARY_VERSION in markdown


def test_rendered_document_lists_every_column_it_was_given(tmp_path: Path):
    markdown = _rendered(tmp_path)
    for column in ("block_group_geoid", "expected_count_robbery", "zcta5"):
        assert f"`{column}`" in markdown


def test_build_dictionary_end_to_end_on_small_surfaces(tmp_path: Path):
    bg_path = tmp_path / "bg.parquet"
    pd.DataFrame(
        {
            "block_group_geoid": ["010010201001"],
            "expected_count_robbery": [1.0],
            "estimate_mode_robbery": ["count_derived"],
        }
    ).to_parquet(bg_path, index=False)
    zcta_path = tmp_path / "zcta.parquet"
    pd.DataFrame({"zcta5": ["01001"], "expected_count_robbery": [1.0]}).to_parquet(
        zcta_path, index=False
    )
    markdown, summary = build_dictionary(
        surfaces=[
            SurfaceEntry(label="Block group surface", geography="block_group", path=bg_path, rows=1),
            SurfaceEntry(label="ZCTA rollup", geography="zcta", path=zcta_path, rows=1),
        ],
        manifest=MANIFEST,
        coverage_universe=UNIVERSE,
        edition_id="2024A-annual",
    )
    assert summary["documented_columns"] == 5
    assert summary["zcta_caveat_present"]
    assert summary["surfaces"]["zcta"]["columns"] == 2
    assert "count_derived" in markdown
