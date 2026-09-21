"""Typed special-use taxonomy: classification annotates rates but never suppresses them.

The properties under test are the ones the construction exists to guarantee: a 98-series code is a
warning flag and not a measurement. Publication is controlled by the active denominator and its
floor. The release validator re-derives every classification and gate independently.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    BURGLARY_PREMISES_DENOMINATOR_FLOOR,
    NON_RESIDENTIAL_HOUSEHOLD_FLOOR,
    PERSON_EXPOSURE_DENOMINATOR_FLOOR,
    SPECIAL_USE_TRACT_PREFIX,
    ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN,
    AllocationBuildConfig,
    _finalize_output,
)
from crimerisk.composites import burden_publishable
from crimerisk.crime import OFFENSES_7
from crimerisk.special_use import (
    COARSER_RECOMMENDATION_TYPES,
    DISPLAY_POLICY_BY_TYPE,
    EDUCATION_JOB_SHARE_MIN,
    EVIDENCE_CAMPUS,
    EVIDENCE_GROUP_QUARTERS_OTHER,
    EVIDENCE_INDUSTRIAL,
    EVIDENCE_NOT_CANDIDATE,
    EVIDENCE_PARK,
    EVIDENCE_PRISON,
    EVIDENCE_RESIDENTIAL_SUPPORT,
    EVIDENCE_TRANSIENT,
    EVIDENCE_UNKNOWN,
    GROUP_QUARTERS_POPULATION_SHARE_MIN,
    NLCD_OPEN_NATURAL_CLASSES,
    NLCD_VALID_LAND_COVER_CLASSES,
    OPEN_NATURAL_LAND_COVER_SHARE_MIN,
    POLICY_COUNT_AND_DENSITY_ONLY,
    POLICY_COUNT_AND_DENSITY_ONLY_FAIL_CLOSED,
    POLICY_EXPOSURE_AND_RESIDENT_RATE,
    POLICY_EXPOSURE_RATE_ONLY,
    POLICY_ORDINARY,
    POSTSECONDARY_ANCHOR_MIN,
    PRIMARY_RATE_ALLOWED_BY_TYPE,
    RESIDENT_RATE_ALLOWED_BY_TYPE,
    SPECIAL_USE_EMPLOYMENT_FLOOR,
    SPECIAL_USE_FEATURE_COLUMNS,
    SPECIAL_USE_HOUSEHOLD_FLOOR,
    SPECIAL_USE_OUTPUT_COLUMNS,
    SPECIAL_USE_PUBLISHED_COLUMNS,
    SPECIAL_USE_TAXONOMY_VERSION,
    SPECIAL_USE_TYPES,
    TRANSIENT_CONFIDENCE_TYPES,
    TYPE_CAMPUS_INSTITUTION,
    TYPE_GROUP_QUARTERS_OTHER,
    TYPE_INDUSTRIAL_EMPLOYMENT,
    TYPE_ORDINARY,
    TYPE_PARK_OPEN_SPACE,
    TYPE_PRISON_INSTITUTIONAL,
    TYPE_TRANSIENT_DESTINATION,
    TYPE_UNKNOWN_SPECIAL_USE,
    UNEXPLAINED_DAYTIME_PRESENCE_RATIO_MIN,
    apply_special_use_taxonomy,
    classify_special_use,
    special_use_diagnostics,
    special_use_features,
    summarize_special_use_taxonomy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / "analysis_scratch" / "final_phase" / "special_use"
CONTRACT_PATH = REPO_ROOT / "analysis_scratch" / "final_phase" / "SPECIAL_USE_TAXONOMY_CONTRACT.md"


def _validator_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"
    spec = importlib.util.spec_from_file_location("validate_release_outputs_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- thresholds: provenance, not invention --------------------------------------------------


def test_the_two_floors_are_the_pipelines_own_constants_reused():
    # The taxonomy mints no new residential floor and no new exposure floor; the consult calls
    # both provisional, and reusing them is what keeps the ordinary lane bit-identical.
    assert SPECIAL_USE_HOUSEHOLD_FLOOR == NON_RESIDENTIAL_HOUSEHOLD_FLOOR
    assert SPECIAL_USE_EMPLOYMENT_FLOOR == PERSON_EXPOSURE_DENOMINATOR_FLOOR


def test_majority_thresholds_are_majorities_and_the_land_cover_one_is_stricter():
    assert GROUP_QUARTERS_POPULATION_SHARE_MIN == 0.5
    assert EDUCATION_JOB_SHARE_MIN == 0.5
    assert POSTSECONDARY_ANCHOR_MIN == 1.0
    # Three quarters, not a majority: a cell that is half built is not open space.
    assert OPEN_NATURAL_LAND_COVER_SHARE_MIN > 0.5


def test_the_transient_threshold_sits_at_the_recorded_reference_p99():
    evidence = json.loads((EVIDENCE_DIR / "threshold_evidence.json").read_text())
    p99 = float(evidence["daytime_presence_ratio"]["quantiles"]["p99"])
    # Selected as the P99 of the national reference distribution, not by looking at a map.
    assert abs(UNEXPLAINED_DAYTIME_PRESENCE_RATIO_MIN - p99) < 0.05
    # And it is far above the middle of that distribution, which is what makes it "unexplained".
    assert UNEXPLAINED_DAYTIME_PRESENCE_RATIO_MIN > 4.0 * float(
        evidence["daytime_presence_ratio"]["quantiles"]["p50"]
    )


def test_open_natural_classes_exclude_every_built_class():
    # NLCD 22/23/24 are developed low/medium/high. A cell cannot be open space because it is built.
    assert 22 not in NLCD_OPEN_NATURAL_CLASSES
    assert 23 not in NLCD_OPEN_NATURAL_CLASSES
    assert 24 not in NLCD_OPEN_NATURAL_CLASSES
    assert 21 in NLCD_OPEN_NATURAL_CLASSES  # developed OPEN space is open space


def test_every_type_has_exactly_one_display_policy_and_one_decision_per_arm():
    assert set(DISPLAY_POLICY_BY_TYPE) == set(SPECIAL_USE_TYPES)
    assert set(PRIMARY_RATE_ALLOWED_BY_TYPE) == set(SPECIAL_USE_TYPES)
    assert set(RESIDENT_RATE_ALLOWED_BY_TYPE) == set(SPECIAL_USE_TYPES)
    for name in SPECIAL_USE_TYPES:
        assert PRIMARY_RATE_ALLOWED_BY_TYPE[name] is True
        assert RESIDENT_RATE_ALLOWED_BY_TYPE[name] is True
    assert COARSER_RECOMMENDATION_TYPES == frozenset()
    assert TRANSIENT_CONFIDENCE_TYPES == frozenset({TYPE_TRANSIENT_DESTINATION})


def test_the_display_policy_table_never_uses_type_as_a_publication_gate():
    assert DISPLAY_POLICY_BY_TYPE[TYPE_ORDINARY] == POLICY_ORDINARY
    for name in set(SPECIAL_USE_TYPES) - {TYPE_ORDINARY}:
        assert DISPLAY_POLICY_BY_TYPE[name] == POLICY_EXPOSURE_AND_RESIDENT_RATE
        assert PRIMARY_RATE_ALLOWED_BY_TYPE[name] is True
        assert RESIDENT_RATE_ALLOWED_BY_TYPE[name] is True


def test_published_columns_are_the_inputs_plus_the_outputs_with_no_duplicates():
    assert SPECIAL_USE_PUBLISHED_COLUMNS == (
        *SPECIAL_USE_FEATURE_COLUMNS,
        *SPECIAL_USE_OUTPUT_COLUMNS,
    )
    assert len(set(SPECIAL_USE_PUBLISHED_COLUMNS)) == len(SPECIAL_USE_PUBLISHED_COLUMNS)


def test_the_contract_records_the_naming_deviation_it_made():
    text = CONTRACT_PATH.read_text()
    assert "transient_destination" in text
    assert "airport_transit" in text  # the name it did NOT use, and why


# --- the cascade on constructed fixtures ----------------------------------------------------


def _cell(
    *,
    tract: str = "01001000100",
    population: float,
    households: float,
    acs_population: float | None = None,
    household_population: float | None = None,
    jobs: float = 0.0,
    education_jobs: float = 0.0,
    postsecondary: float = 0.0,
    open_natural_pixels: float = 0.0,
    classified_pixels: float = 1_000.0,
    landscan_day: float = 0.0,
    gq_type: str = "institutional",
) -> dict:
    acs = population if acs_population is None else acs_population
    household = population if household_population is None else household_population
    gq_total = max(float(acs) - float(household), 0.0)
    values = {
        "tract_id": tract,
        "population_2024": population,
        "households_total": households,
        "special_use_acs_population": acs,
        "special_use_household_population": household,
        "special_use_jobs_total": jobs,
        "special_use_jobs_education": education_jobs,
        "special_use_postsecondary_count": postsecondary,
        "special_use_open_natural_pixels": open_natural_pixels,
        "special_use_classified_pixels": classified_pixels,
        "landscan_day_pop": landscan_day,
        "special_use_gq_total_2020": gq_total,
        "special_use_gq_institutional_2020": gq_total if gq_type == "institutional" else 0.0,
        "special_use_gq_correctional_2020": gq_total if gq_type == "institutional" else 0.0,
        "special_use_gq_juvenile_2020": 0.0,
        "special_use_gq_nursing_2020": 0.0,
        "special_use_gq_college_2020": gq_total if gq_type == "college" else 0.0,
        "special_use_gq_military_2020": gq_total if gq_type == "other" else 0.0,
        "special_use_gq_other_institutional_2020": 0.0,
        "special_use_gq_other_noninstitutional_2020": 0.0,
    }
    return values


CASCADE_CASES: tuple[tuple[str, dict, str, str], ...] = (
    (
        "ordinary neighbourhood is never asked",
        _cell(population=2_000.0, households=700.0),
        TYPE_ORDINARY,
        EVIDENCE_NOT_CANDIDATE,
    ),
    (
        "98-series code with real residential support is released",
        _cell(tract="01001980100", population=4_610.0, households=1_800.0),
        TYPE_ORDINARY,
        EVIDENCE_RESIDENTIAL_SUPPORT,
    ),
    (
        "98-series code with majority group quarters is NOT released",
        _cell(
            tract="01001980100",
            population=3_000.0,
            households=1_000.0,
            household_population=0.0,
        ),
        TYPE_PRISON_INSTITUTIONAL,
        EVIDENCE_PRISON,
    ),
    (
        "campus: anchor plus dormitory population",
        _cell(
            population=1_500.0,
            households=0.0,
            household_population=0.0,
            jobs=300.0,
            postsecondary=1.0,
        ),
        TYPE_CAMPUS_INSTITUTION,
        EVIDENCE_CAMPUS,
    ),
    (
        "campus: Census college housing without a same-BG campus anchor",
        _cell(
            population=1_500.0,
            households=0.0,
            household_population=0.0,
            gq_type="college",
        ),
        TYPE_CAMPUS_INSTITUTION,
        EVIDENCE_CAMPUS,
    ),
    (
        "campus: anchor plus education-dominated employment",
        _cell(
            population=0.0,
            households=0.0,
            jobs=400.0,
            education_jobs=300.0,
            postsecondary=2.0,
            landscan_day=200.0,
        ),
        TYPE_CAMPUS_INSTITUTION,
        EVIDENCE_CAMPUS,
    ),
    (
        "prison: majority group quarters, no anchor",
        _cell(
            population=1_200.0,
            households=0.0,
            household_population=0.0,
            jobs=200.0,
            landscan_day=100.0,
        ),
        TYPE_PRISON_INSTITUTIONAL,
        EVIDENCE_PRISON,
    ),
    (
        "other group quarters are not labeled institutional",
        _cell(
            population=1_200.0,
            households=0.0,
            household_population=0.0,
            gq_type="other",
        ),
        TYPE_GROUP_QUARTERS_OTHER,
        EVIDENCE_GROUP_QUARTERS_OTHER,
    ),
    (
        "park: open and undeveloped land cover",
        _cell(population=0.0, households=0.0, open_natural_pixels=950.0),
        TYPE_PARK_OPEN_SPACE,
        EVIDENCE_PARK,
    ),
    (
        "transient destination: daytime presence beyond residents plus jobs",
        _cell(population=0.0, households=0.0, jobs=500.0, landscan_day=5_000.0),
        TYPE_TRANSIENT_DESTINATION,
        EVIDENCE_TRANSIENT,
    ),
    (
        "employment district: daytime presence explained by its jobs",
        _cell(population=0.0, households=0.0, jobs=500.0, landscan_day=400.0),
        TYPE_INDUSTRIAL_EMPLOYMENT,
        EVIDENCE_INDUSTRIAL,
    ),
    (
        "no resolving evidence fails closed",
        _cell(
            population=0.0,
            households=0.0,
            jobs=10.0,
            open_natural_pixels=100.0,
            landscan_day=5.0,
        ),
        TYPE_UNKNOWN_SPECIAL_USE,
        EVIDENCE_UNKNOWN,
    ),
)


@pytest.mark.parametrize(
    "case,expected_type,expected_evidence",
    [(row, expected, evidence) for _, row, expected, evidence in CASCADE_CASES],
    ids=[name for name, _, _, _ in CASCADE_CASES],
)
def test_the_cascade_assigns_the_type_and_names_the_clause(case, expected_type, expected_evidence):
    frame = pd.DataFrame([case])
    frame["special_use_tract_flag"] = frame["tract_id"].str.slice(5, 11).str.startswith(
        SPECIAL_USE_TRACT_PREFIX
    )
    typed = classify_special_use(frame, population_col="population_2024")
    assert typed["special_use_type"].iloc[0] == expected_type
    assert typed["special_use_type_evidence"].iloc[0] == expected_evidence
    assert typed["special_use_display_policy"].iloc[0] == DISPLAY_POLICY_BY_TYPE[expected_type]


def test_a_populated_ordinary_cell_with_group_quarters_is_never_pulled_into_the_taxonomy():
    # A dormitory-heavy neighbourhood with a real household base is not a candidate at all: the
    # taxonomy only ever looks at cells the Census flagged or that have no residential base.
    frame = pd.DataFrame(
        [_cell(population=4_000.0, households=500.0, household_population=0.0)]
    )
    typed = classify_special_use(frame, population_col="population_2024")
    assert typed["special_use_type"].iloc[0] == TYPE_ORDINARY
    assert bool(typed["special_use_candidate_flag"].iloc[0]) is False


def test_campus_outranks_prison_and_park_outranks_the_employment_rules():
    # A cell with BOTH an anchor and majority group quarters is a campus, not a prison.
    campus = pd.DataFrame(
        [
            _cell(
                population=2_000.0,
                households=0.0,
                household_population=0.0,
                postsecondary=1.0,
                jobs=900.0,
            )
        ]
    )
    assert (
        classify_special_use(campus, population_col="population_2024")["special_use_type"].iloc[0]
        == TYPE_CAMPUS_INSTITUTION
    )
    # A depot on a mostly-natural cell does not make it an employment district.
    park = pd.DataFrame(
        [
            _cell(
                population=0.0,
                households=0.0,
                jobs=800.0,
                open_natural_pixels=900.0,
                landscan_day=9_000.0,
            )
        ]
    )
    assert (
        classify_special_use(park, population_col="population_2024")["special_use_type"].iloc[0]
        == TYPE_PARK_OPEN_SPACE
    )


def test_absent_classification_inputs_are_marked_unknown_without_closing_denominator_valid_rates():
    frame = pd.DataFrame(
        [
            {"tract_id": "01001980100", "population_2024": 0.0, "households_total": 0.0},
            {"tract_id": "01001000200", "population_2024": 0.0, "households_total": 0.0},
        ]
    )
    typed = classify_special_use(frame, population_col="population_2024")
    assert list(typed["special_use_type"]) == [TYPE_UNKNOWN_SPECIAL_USE, TYPE_UNKNOWN_SPECIAL_USE]
    assert typed["special_use_primary_rate_allowed"].all()
    assert typed["special_use_resident_rate_allowed"].all()


def test_missing_land_cover_is_an_unknown_share_and_not_a_zero_one():
    # Zero classified pixels must not read as "0% open space, therefore fully developed"; the
    # share is NULL and the cell falls through to the fail-closed type.
    frame = pd.DataFrame(
        [_cell(population=0.0, households=0.0, open_natural_pixels=0.0, classified_pixels=0.0)]
    )
    diagnostics = special_use_diagnostics(frame, population_col="population_2024")
    assert bool(pd.isna(diagnostics["open_natural_share"].iloc[0]))
    typed = classify_special_use(frame, population_col="population_2024")
    assert typed["special_use_type"].iloc[0] == TYPE_UNKNOWN_SPECIAL_USE


def test_group_quarters_share_is_null_where_nobody_lives_and_bounded_where_they_do():
    frame = pd.DataFrame(
        [
            _cell(population=0.0, households=0.0, acs_population=0.0, household_population=0.0),
            # Household population above the ACS total (survey noise) must clip, never go negative.
            _cell(population=100.0, households=40.0, acs_population=100.0, household_population=120.0),
        ]
    )
    diagnostics = special_use_diagnostics(frame, population_col="population_2024")
    assert bool(pd.isna(diagnostics["group_quarters_share"].iloc[0]))
    assert float(diagnostics["group_quarters_share"].iloc[1]) == 0.0


# --- the gates ------------------------------------------------------------------------------


def test_the_arm_gates_never_close_a_rate_by_type():
    rows = []
    for _, case, expected_type, _ in CASCADE_CASES:
        rows.append({**case, "_expected_type": expected_type})
    frame = pd.DataFrame(rows)
    frame["special_use_tract_flag"] = frame["tract_id"].str.slice(5, 11).str.startswith(
        SPECIAL_USE_TRACT_PREFIX
    )
    typed = classify_special_use(frame, population_col="population_2024")
    for position, expected_type in enumerate(frame["_expected_type"]):
        assert PRIMARY_RATE_ALLOWED_BY_TYPE[expected_type] is True
        assert RESIDENT_RATE_ALLOWED_BY_TYPE[expected_type] is True
        assert bool(typed["special_use_primary_rate_allowed"].iloc[position]) is True
        assert bool(typed["special_use_resident_rate_allowed"].iloc[position]) is True


def test_a_campus_with_no_residents_is_left_to_the_denominator_floor():
    frame = pd.DataFrame(
        [_cell(population=0.0, households=0.0, jobs=400.0, education_jobs=300.0, postsecondary=1.0)]
    )
    typed = classify_special_use(frame, population_col="population_2024")
    assert typed["special_use_type"].iloc[0] == TYPE_CAMPUS_INSTITUTION
    assert bool(typed["special_use_primary_rate_allowed"].iloc[0]) is True
    assert bool(typed["special_use_resident_rate_allowed"].iloc[0]) is True


def test_the_transient_flag_rides_only_on_the_class_it_describes():
    rows = [case for _, case, _, _ in CASCADE_CASES]
    frame = pd.DataFrame(rows)
    frame["special_use_tract_flag"] = frame["tract_id"].str.slice(5, 11).str.startswith(
        SPECIAL_USE_TRACT_PREFIX
    )
    typed = classify_special_use(frame, population_col="population_2024")
    flagged = typed.loc[typed["special_use_transient_confidence_flag"], "special_use_type"]
    assert set(flagged) == {TYPE_TRANSIENT_DESTINATION}


# --- extensivity: one cascade, both geographies ---------------------------------------------


def test_every_classification_input_is_extensive():
    """Doubling the wide covariate frame's counts doubles every derived input.

    This is the property that makes a tract classify itself from the sum of its block groups
    through the same function, so it is asserted rather than assumed.
    """
    wide = pd.DataFrame(
        [
            {
                "total_population": 1_000.0,
                "households_total": 400.0,
                "avg_household_size_total": 2.5,
                "jobs_wac": 300.0,
                "lodes_cns15_share": 0.4,
                "postsecondary_count": 1.0,
                "nlcd_valid_pixel_count": 800.0,
                "census2020_gq_total": 100.0,
                "census2020_gq_institutional": 40.0,
                "census2020_gq_correctional": 20.0,
                "census2020_gq_juvenile": 5.0,
                "census2020_gq_nursing": 10.0,
                "census2020_gq_college": 50.0,
                "census2020_gq_military": 5.0,
                "census2020_gq_other_institutional": 5.0,
                "census2020_gq_other_noninstitutional": 5.0,
                **{f"nlcd_count_{code}": 10.0 for code in NLCD_OPEN_NATURAL_CLASSES},
            }
        ]
    )
    single = special_use_features(wide)
    doubled = special_use_features(
        wide.assign(
            total_population=wide["total_population"] * 2,
            households_total=wide["households_total"] * 2,
            jobs_wac=wide["jobs_wac"] * 2,
            postsecondary_count=wide["postsecondary_count"] * 2,
            nlcd_valid_pixel_count=wide["nlcd_valid_pixel_count"] * 2,
            **{
                column: wide[column] * 2
                for column in wide.columns
                if column.startswith("census2020_gq_")
            },
            **{f"nlcd_count_{code}": wide[f"nlcd_count_{code}"] * 2 for code in NLCD_OPEN_NATURAL_CLASSES},
        )
    )
    for column in SPECIAL_USE_FEATURE_COLUMNS:
        assert float(doubled[column].iloc[0]) == pytest.approx(2.0 * float(single[column].iloc[0]))
    # And the group-quarters population is total minus household, on the ACS basis throughout.
    assert float(single["special_use_household_population"].iloc[0]) == pytest.approx(1_000.0)
    assert float(single["special_use_open_natural_pixels"].iloc[0]) == pytest.approx(
        10.0 * len(NLCD_OPEN_NATURAL_CLASSES)
    )


def test_classified_land_pixels_reconstruct_from_the_frozen_count_columns():
    wide = pd.DataFrame(
        [
            {
                **{f"nlcd_count_{code}": float(code) for code in NLCD_VALID_LAND_COVER_CLASSES},
                "total_population": 0.0,
                "households_total": 0.0,
                "avg_household_size_total": 0.0,
                "jobs_wac": 0.0,
                "lodes_cns15_share": 0.0,
                "postsecondary_count": 0.0,
            }
        ]
    )

    features = special_use_features(wide)

    assert float(features["special_use_classified_pixels"].iloc[0]) == pytest.approx(
        sum(NLCD_VALID_LAND_COVER_CLASSES)
    )


def test_a_tract_classifies_itself_from_the_sum_of_its_block_groups():
    # Three block groups: a prison, and two empty slivers. The tract's own quantities are their
    # sums, and the tract answer is the cascade on those sums -- not a vote over the children.
    children = pd.DataFrame(
        [
            _cell(population=1_200.0, households=0.0, household_population=0.0, jobs=200.0),
            _cell(population=0.0, households=0.0, jobs=10.0, open_natural_pixels=100.0),
            _cell(population=0.0, households=0.0, jobs=5.0, open_natural_pixels=50.0),
        ]
    )
    child_types = classify_special_use(children, population_col="population_2024")[
        "special_use_type"
    ]
    assert list(child_types) == [
        TYPE_PRISON_INSTITUTIONAL,
        TYPE_UNKNOWN_SPECIAL_USE,
        TYPE_UNKNOWN_SPECIAL_USE,
    ]
    summed = children.drop(columns=["tract_id"]).sum().to_frame().T
    summed["tract_id"] = "01001000100"
    parent_type = classify_special_use(summed, population_col="population_2024")[
        "special_use_type"
    ].iloc[0]
    # Two of three children fail closed and the parent still resolves, because the parent's
    # group-quarters share is a real measured quantity at its own support.
    assert parent_type == TYPE_PRISON_INSTITUTIONAL


# --- the finalized surface, both geographies ------------------------------------------------


def _surface_row(
    geoid: str,
    *,
    population: float,
    households: float,
    counts: float = 12.0,
    exposure: float = 3_000.0,
    premises: float = 400.0,
    vehicles: float = 900.0,
    **special_use: float,
) -> dict:
    tract = geoid[:11]
    row = {
        "block_group_geoid": geoid,
        "tract_id": tract,
        "state_fips": geoid[:2],
        "population_2024": population,
        "households_total": households,
        "commercial_premises_total": 0.0,
        "destination_poi_total": 0.0,
        "daytime_population_jobs_proxy": population,
        "landscan_day_pop": special_use.get("landscan_day", 0.0),
        "exposure_proxy_2024": exposure,
        "burglary_premises_total": premises,
        "aggregate_vehicles_total": vehicles,
        "vehicle_exposure_2024": vehicles,
        "land_area_sq_mi": 1.0,
        "eb_jurisdiction_id": "J1",
        "eb_jurisdiction_type": "municipal",
    }
    cell = _cell(
        tract=tract,
        population=population,
        households=households,
        acs_population=special_use.get("acs_population", population),
        household_population=special_use.get("household_population", population),
        jobs=special_use.get("jobs", 0.0),
        education_jobs=special_use.get("education_jobs", 0.0),
        postsecondary=special_use.get("postsecondary", 0.0),
        open_natural_pixels=special_use.get("open_natural_pixels", 0.0),
        classified_pixels=special_use.get("classified_pixels", 1_000.0),
        landscan_day=special_use.get("landscan_day", 0.0),
        gq_type=special_use.get("gq_type", "institutional"),
    )
    for column in SPECIAL_USE_FEATURE_COLUMNS:
        row[column] = cell[column]
    for offense in OFFENSES_7:
        row[f"expected_count_{offense}"] = counts
        row[f"footprint_derived_count_{offense}"] = 0.0
    return row


TYPED_ROWS: dict[str, dict] = {
    TYPE_CAMPUS_INSTITUTION: _surface_row(
        "010010001011",
        population=1_500.0,
        households=0.0,
        household_population=0.0,
        jobs=300.0,
        postsecondary=1.0,
        exposure=1_800.0,
    ),
    TYPE_INDUSTRIAL_EMPLOYMENT: _surface_row(
        "010010001021",
        population=0.0,
        households=0.0,
        jobs=500.0,
        landscan_day=400.0,
        exposure=600.0,
        # Nobody lives here, so every offense's normalizer has to clear the near-zero-resident
        # opportunity floor before the TYPE's authorisation means anything. The shared 400-premises
        # default does not, and this fixture is about the type gate, not about the support floor --
        # `tests/test_zero_resident_opportunity_floor.py` owns that interaction.
        premises=800.0,
    ),
    TYPE_TRANSIENT_DESTINATION: _surface_row(
        "010010001031",
        population=0.0,
        households=0.0,
        jobs=500.0,
        landscan_day=5_000.0,
        exposure=600.0,
    ),
    TYPE_PARK_OPEN_SPACE: _surface_row(
        "010010001041",
        population=0.0,
        households=0.0,
        open_natural_pixels=950.0,
        exposure=200.0,
    ),
    TYPE_PRISON_INSTITUTIONAL: _surface_row(
        "010010001051",
        population=1_200.0,
        households=0.0,
        household_population=0.0,
        jobs=200.0,
        exposure=1_400.0,
    ),
    TYPE_GROUP_QUARTERS_OTHER: _surface_row(
        "010010001071",
        population=1_000.0,
        households=0.0,
        household_population=0.0,
        gq_type="other",
        exposure=1_200.0,
    ),
    TYPE_UNKNOWN_SPECIAL_USE: _surface_row(
        "010010001061",
        population=0.0,
        households=0.0,
        jobs=10.0,
        open_natural_pixels=100.0,
        exposure=200.0,
    ),
}


def _ordinary_rows(count: int = 24) -> list[dict]:
    return [
        _surface_row(
            f"010010002{index:02d}1",
            population=2_000.0,
            households=700.0,
            counts=20.0,
        )
        for index in range(count)
    ]


def _finalized(*, taxonomy: bool, geography: str = "block_group") -> pd.DataFrame:
    rows = [*_ordinary_rows(), *TYPED_ROWS.values()]
    frame = pd.DataFrame(rows)
    if geography == "tract":
        # One block group per tract, so the tract surface carries the same cells at its own
        # support and the cascade sees identical sums.
        frame = frame.drop(columns=["block_group_geoid", "eb_jurisdiction_id", "eb_jurisdiction_type"])
        frame["dominant_eb_jurisdiction_id"] = "J1"
        return _finalize_output(
            frame,
            geo_id_col="tract_id",
            population_col="population_2024",
            config=AllocationBuildConfig(enable_special_use_taxonomy=taxonomy),
            jurisdiction_col="dominant_eb_jurisdiction_id",
        ).set_index("tract_id")
    return _finalize_output(
        frame,
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(enable_special_use_taxonomy=taxonomy),
    ).set_index("block_group_geoid")


def _row_id(geography: str, geoid: str) -> str:
    return geoid[:11] if geography == "tract" else geoid


@pytest.mark.parametrize("geography", ["block_group", "tract"])
def test_the_typed_surface_uses_denominator_gates_after_type_authorisation(geography):
    surface = _finalized(taxonomy=True, geography=geography)
    for expected_type, row in TYPED_ROWS.items():
        key = _row_id(geography, row["block_group_geoid"])
        assert surface.loc[key, "special_use_type"] == expected_type
        primary = PRIMARY_RATE_ALLOWED_BY_TYPE[expected_type]
        resident = RESIDENT_RATE_ALLOWED_BY_TYPE[expected_type]
        for offense in OFFENSES_7:
            primary_value = surface.loc[key, f"index_{offense}_primary"]
            resident_value = surface.loc[key, f"index_{offense}_resident"]
            assert primary is not False
            assert resident is not False
            assert str(surface.loc[key, f"estimate_mode_{offense}"]) != "special_use"
            if pd.isna(primary_value):
                assert str(surface.loc[key, f"denominator_reason_{offense}"]) in {
                    "insufficient_exposure",
                    "non_residential",
                }
            if float(row["population_2024"]) >= PERSON_EXPOSURE_DENOMINATOR_FLOOR:
                assert pd.notna(resident_value), (expected_type, offense)
            else:
                assert pd.isna(resident_value), (expected_type, offense)


@pytest.mark.parametrize("geography", ["block_group", "tract"])
def test_no_closed_arm_is_ever_encoded_as_zero(geography):
    surface = _finalized(taxonomy=True, geography=geography)
    for row in TYPED_ROWS.values():
        key = _row_id(geography, row["block_group_geoid"])
        for offense in OFFENSES_7:
            for field in (
                f"index_{offense}_primary",
                f"rate_{offense}_primary",
                f"index_{offense}_resident",
                f"rate_{offense}_resident",
            ):
                value = surface.loc[key, field]
                assert pd.isna(value) or float(value) != 0.0, (key, field)


@pytest.mark.parametrize("geography", ["block_group", "tract"])
def test_counts_and_density_survive_every_type(geography):
    surface = _finalized(taxonomy=True, geography=geography)
    for row in TYPED_ROWS.values():
        key = _row_id(geography, row["block_group_geoid"])
        for offense in OFFENSES_7:
            assert float(surface.loc[key, f"expected_count_{offense}"]) > 0.0
            assert pd.notna(surface.loc[key, f"crime_density_{offense}"])
        assert pd.notna(surface.loc[key, "crime_density_total"])


@pytest.mark.parametrize("geography", ["block_group", "tract"])
def test_type_never_reports_special_use_as_the_reason_a_rate_is_closed(geography):
    surface = _finalized(taxonomy=True, geography=geography)
    for expected_type, row in TYPED_ROWS.items():
        key = _row_id(geography, row["block_group_geoid"])
        for offense in OFFENSES_7:
            mode = str(surface.loc[key, f"estimate_mode_{offense}"])
            assert mode != "special_use", (expected_type, offense)
            assert str(surface.loc[key, f"denominator_reason_{offense}"]) != "special_use"


def test_a_published_typed_cell_is_never_also_labelled_non_residential():
    surface = _finalized(taxonomy=True)
    key = TYPED_ROWS[TYPE_INDUSTRIAL_EMPLOYMENT]["block_group_geoid"]
    assert float(surface.loc[key, "households_total"]) < NON_RESIDENTIAL_HOUSEHOLD_FLOOR
    for offense in OFFENSES_7:
        assert str(surface.loc[key, f"estimate_mode_{offense}"]) == "count_derived"
        assert str(surface.loc[key, f"denominator_reason_{offense}"]) != "non_residential"


def test_the_burglary_premises_floor_stops_calling_itself_special_use():
    rows = [
        *_ordinary_rows(),
        _surface_row(
            "010010003001",
            population=2_000.0,
            households=700.0,
            premises=float(BURGLARY_PREMISES_DENOMINATOR_FLOOR) - 1.0,
        ),
    ]
    frame = pd.DataFrame(rows)
    legacy = _finalize_output(
        frame.copy(),
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(),
    ).set_index("block_group_geoid")
    typed = _finalize_output(
        frame.copy(),
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(enable_special_use_taxonomy=True),
    ).set_index("block_group_geoid")
    assert str(legacy.loc["010010003001", "estimate_mode_burglary"]) == "special_use"
    assert str(typed.loc["010010003001", "estimate_mode_burglary"]) == "insufficient_exposure"
    # The premises floor belongs to the premises-normalized arm. A valid resident
    # denominator remains publishable.
    assert str(typed.loc["010010003001", "resident_denominator_reason_burglary"]) == "publishable"
    assert pd.isna(typed.loc["010010003001", "index_burglary_primary"])
    assert pd.notna(typed.loc["010010003001", "index_burglary_resident"])


# --- legacy byte-safety ---------------------------------------------------------------------


def test_the_flag_off_surface_gains_no_column_from_this_lane():
    legacy = _finalized(taxonomy=False)
    typed = _finalized(taxonomy=True)
    for column in SPECIAL_USE_PUBLISHED_COLUMNS:
        assert column not in legacy.columns
        assert column in typed.columns
    assert "special_use_tract_flag" in legacy.columns  # the Census fact stays, unchanged
    # The floor column is not this lane's, but this lane turns it on: opening the opportunity arm
    # by TYPE is what created the residentless-publisher class the near-zero-resident opportunity
    # floor closes (see the contract's "Interaction with the rules this lane does not own").
    assert set(typed.columns) - set(legacy.columns) == {
        *SPECIAL_USE_PUBLISHED_COLUMNS,
        ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN,
    }


def test_the_flag_off_publication_masks_match_a_verbatim_transcription_of_the_old_rule():
    """Pre-change expression from `_finalize_output`, transcribed rather than imported."""
    legacy = _finalized(taxonomy=False)
    households = pd.to_numeric(legacy["households_total"], errors="coerce").fillna(0.0)
    residential_eligible = households.ge(float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR))
    special_use_tract = (
        legacy["tract_id"].astype("string").str.zfill(11).str.slice(5, 11).str.startswith("98", na=False)
    )
    for offense in OFFENSES_7:
        denominator = pd.to_numeric(legacy[f"primary_denominator_{offense}"], errors="coerce").fillna(0.0)
        if offense == "burglary":
            special_use_suppressed = special_use_tract | denominator.lt(
                float(BURGLARY_PREMISES_DENOMINATOR_FLOOR)
            )
        else:
            special_use_suppressed = special_use_tract
        if offense in {"murder", "rape", "robbery", "aggravated_assault", "larceny"}:
            insufficient = denominator.lt(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
        elif offense == "motor_vehicle_theft":
            insufficient = denominator.lt(50.0)
        else:
            insufficient = pd.Series(False, index=legacy.index)
        footprint = legacy[f"footprint_ambient_exposure_missing_{offense}"].fillna(False).astype(bool)
        expected = (
            residential_eligible
            & denominator.gt(0.0)
            & ~special_use_suppressed.fillna(False).astype(bool)
            & ~insufficient.fillna(False).astype(bool)
            & ~footprint
        )
        published = legacy[f"primary_index_publishable_{offense}"].fillna(False).astype(bool)
        assert np.array_equal(published.to_numpy(dtype=bool), expected.to_numpy(dtype=bool)), offense


def test_the_ordinary_rows_are_bit_identical_across_the_two_lanes():
    legacy = _finalized(taxonomy=False)
    typed = _finalized(taxonomy=True)
    ordinary = [row["block_group_geoid"] for row in _ordinary_rows()]
    # Every ordinary cell keeps its exact published point. The typed lane changes WHO publishes,
    # and adding publishers changes the national normalizer -- so this asserts the ordinary rows
    # are untouched only where the publishable set did not move, which for these fixtures is the
    # count and the denominator, not the index.
    for offense in OFFENSES_7:
        assert np.array_equal(
            legacy.loc[ordinary, f"expected_count_{offense}"].to_numpy(dtype=float),
            typed.loc[ordinary, f"expected_count_{offense}"].to_numpy(dtype=float),
        )
        assert np.array_equal(
            legacy.loc[ordinary, f"primary_denominator_{offense}"].to_numpy(dtype=float),
            typed.loc[ordinary, f"primary_denominator_{offense}"].to_numpy(dtype=float),
        )
        assert (
            legacy.loc[ordinary, f"estimate_mode_{offense}"].tolist()
            == typed.loc[ordinary, f"estimate_mode_{offense}"].tolist()
        )


# --- the release validator's independent mirror ---------------------------------------------


@pytest.mark.parametrize("geography", ["block_group", "tract"])
def test_the_release_validator_re_derives_every_gate_from_published_inputs(geography):
    validator = _validator_module()
    surface = _finalized(taxonomy=True, geography=geography).reset_index()
    assert validator._special_use_taxonomy_lane(surface) is True
    mirror = validator._expected_special_use_taxonomy(surface, population_col="population_2024")
    assert mirror["special_use_type"].tolist() == surface["special_use_type"].tolist()
    for column in (
        "special_use_primary_rate_allowed",
        "special_use_resident_rate_allowed",
        "special_use_candidate_flag",
    ):
        assert np.array_equal(
            mirror[column].to_numpy(dtype=bool),
            surface[column].fillna(False).astype(bool).to_numpy(dtype=bool),
        ), (geography, column)


def test_the_validator_mirror_reduces_to_the_legacy_rule_off_the_typed_lane():
    validator = _validator_module()
    legacy = _finalized(taxonomy=False).reset_index()
    assert validator._special_use_taxonomy_lane(legacy) is False
    assert validator._count_first_publishable(legacy).dtype == bool


def test_the_validator_expects_the_typed_columns_only_on_a_typed_surface():
    validator = _validator_module()
    typed_columns = set(
        validator._expected_columns(geography="block_group", special_use_taxonomy=True)
    )
    legacy_columns = set(
        validator._expected_columns(geography="block_group", special_use_taxonomy=False)
    )
    assert typed_columns - legacy_columns == {
        *SPECIAL_USE_PUBLISHED_COLUMNS,
        ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN,
    }


# --- count-first composites ------------------------------------------------------------------


def test_the_count_first_gate_reads_the_resident_arm_and_agrees_on_ordinary_cells():
    surface = _finalized(taxonomy=True)
    gate = burden_publishable(surface)
    ordinary = [row["block_group_geoid"] for row in _ordinary_rows()]
    assert bool(gate.loc[ordinary].all())
    for expected_type, row in TYPED_ROWS.items():
        expected = float(row["population_2024"]) >= PERSON_EXPOSURE_DENOMINATOR_FLOOR
        assert bool(gate.loc[row["block_group_geoid"]]) is expected, expected_type


def test_the_count_first_gate_is_unchanged_off_the_typed_lane():
    legacy = _finalized(taxonomy=False)
    households = pd.to_numeric(legacy["households_total"], errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(legacy["resident_secondary_denominator"], errors="coerce").fillna(0.0)
    special_use = legacy["special_use_tract_flag"].fillna(False).astype(bool)
    expected = (
        households.ge(float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR))
        & denominator.gt(0.0)
        & denominator.ge(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
        & ~special_use
    )
    assert np.array_equal(
        burden_publishable(legacy).to_numpy(dtype=bool), expected.to_numpy(dtype=bool)
    )


# --- config, manifest, reporting --------------------------------------------------------------


def test_the_lane_is_off_by_default():
    assert AllocationBuildConfig().enable_special_use_taxonomy is False


def test_the_cli_exposes_the_opt_in_flag():
    from crimerisk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["build-outputs", "--enable-special-use-taxonomy"])
    assert args.enable_special_use_taxonomy is True
    assert parser.parse_args(["build-outputs"]).enable_special_use_taxonomy is False


def test_the_summary_reports_the_taxonomy_only_when_it_ran():
    typed = _finalized(taxonomy=True)
    summary = summarize_special_use_taxonomy(typed)
    assert summary["enabled"] is True
    assert summary["version"] == SPECIAL_USE_TAXONOMY_VERSION
    assert set(summary["type_counts"]) == set(SPECIAL_USE_TYPES)
    assert sum(summary["type_counts"].values()) == len(typed)
    for expected_type in TYPED_ROWS:
        assert summary["type_counts"][expected_type] == 1
    assert summarize_special_use_taxonomy(_finalized(taxonomy=False)) == {"enabled": False}


def test_apply_special_use_taxonomy_attaches_exactly_the_output_columns():
    frame = pd.DataFrame([case for _, case, _, _ in CASCADE_CASES])
    frame["special_use_tract_flag"] = frame["tract_id"].str.slice(5, 11).str.startswith(
        SPECIAL_USE_TRACT_PREFIX
    )
    out = apply_special_use_taxonomy(frame, population_col="population_2024")
    assert set(out.columns) - set(frame.columns) == set(SPECIAL_USE_OUTPUT_COLUMNS)


# --- the audit artifact ------------------------------------------------------------------------


def test_the_audit_sample_covers_every_stratum_with_before_and_after_display_modes():
    delta = json.loads((EVIDENCE_DIR / "display_mode_delta.json").read_text())
    if delta.get("taxonomy_version") != SPECIAL_USE_TAXONOMY_VERSION:
        pytest.skip("special-use evidence awaits regeneration for the current taxonomy")
    audit = pd.read_csv(EVIDENCE_DIR / "audit_sample_tracts.csv", dtype={"tract_id": str})
    assert len(audit) >= 200
    strata = set(audit["audit_stratum"])
    assert strata == {
        "airports_transit_or_major_employment",
        "institutional_or_campus",
        "parks_or_industrial",
        "near_zero_population_ordinary_code",
        "fail_closed_unknown",
    }
    for offense in OFFENSES_7:
        assert f"display_mode_before_{offense}" in audit.columns
        assert f"display_mode_after_{offense}" in audit.columns
    # Every sampled tract carries the clause that assigned it, so classification quality is
    # reviewable without re-running anything.
    assert audit["special_use_type_evidence"].notna().all()
    assert set(audit["special_use_type"]).issubset(set(SPECIAL_USE_TYPES))


def test_the_recorded_delta_is_monotone_in_the_direction_the_taxonomy_claims():
    delta = json.loads((EVIDENCE_DIR / "display_mode_delta.json").read_text())
    if delta.get("taxonomy_version") != SPECIAL_USE_TAXONOMY_VERSION:
        pytest.skip("special-use evidence awaits regeneration for the current taxonomy")
    # The harness reproduced the deployed surface before measuring anything.
    for offense, agreed in delta["harness_agreement_with_promoted_surface"].items():
        assert agreed == delta["tracts"], offense
    assert delta["cells_suppressed_to_published_total"] > 100 * delta[
        "cells_published_to_suppressed_total"
    ]
    assert set(delta["type_counts"]) == set(SPECIAL_USE_TYPES)
    assert set(delta["block_group_type_counts"]) == set(SPECIAL_USE_TYPES)
