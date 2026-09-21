"""The near-zero-resident opportunity floor: an opportunity rate needs opportunity mass behind it.

The v2 lanes created a cell class the legacy lane could not reach. The typed taxonomy opens the
opportunity arm on `industrial_employment` and `campus_institution` by TYPE rather than by
household count, and the exposure ensemble stopped inflating the denominator by taking a max over
its legs -- so a block group with NO residents can clear the landed 50-unit exposure floor on a
denominator of 60-odd person-equivalents and paint the map from one modelled event. Measured on the
v2 candidate before this rule: 3,825 (BG x offense) cells published an opportunity index with zero
residents, 222 above the top legend break, the loudest a robbery index of 25,485 on a denominator of
66.5.

The rule under test: below the resident threshold the exposure floor already uses (50 residents),
publishing ANY primary opportunity-rate index additionally requires 500 units of that offense's own
normalizer. Below that the cell publishes counts and density -- the taxonomy's own park/prison
policy -- under the existing `insufficient_exposure` display semantics, with no doubt language and
no new display code. A cell with 50 or more residents is decided by the landed floor exactly as
before, and a build with both flags off never evaluates the rule at all.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    BURGLARY_PREMISES_DENOMINATOR_FLOOR,
    MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR,
    NON_RESIDENTIAL_HOUSEHOLD_FLOOR,
    PERSON_EXPOSURE_DENOMINATOR_FLOOR,
    ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR,
    ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN,
    AllocationBuildConfig,
    _finalize_output,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.exposure_ensemble import ENSEMBLE_OFFENSES, opportunity_normalizer_column
from crimerisk.special_use import SPECIAL_USE_FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPOSURE_CONTRACT = REPO_ROOT / "analysis_scratch" / "final_phase" / "EXPOSURE_ENSEMBLE_CONTRACT.md"
TAXONOMY_CONTRACT = (
    REPO_ROOT / "analysis_scratch" / "final_phase" / "SPECIAL_USE_TAXONOMY_CONTRACT.md"
)

# Every denominator on a fixture row is set to the same number, so one row exercises the rule on
# all seven offenses at once against whichever normalizer each offense actually uses.
BELOW = 200.0  # above every landed floor (50 person / 50 vehicle / 10 premises), below 500
ABOVE = 900.0


def _validator_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"
    spec = importlib.util.spec_from_file_location("validate_zero_resident_floor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    geoid: str,
    *,
    population: float,
    households: float,
    denominator: float,
    jobs: float = 500.0,
    counts: float = 12.0,
    with_normalizers: bool = False,
) -> dict:
    """One block group whose every offense normalizer equals `denominator`."""
    row = {
        "block_group_geoid": geoid,
        "tract_id": geoid[:11],
        "state_fips": geoid[:2],
        "population_2024": population,
        "households_total": households,
        "commercial_premises_total": 0.0,
        "destination_poi_total": 0.0,
        "daytime_population_jobs_proxy": population,
        "landscan_day_pop": 0.0,
        "exposure_proxy_2024": denominator,
        "burglary_premises_total": denominator,
        "aggregate_vehicles_total": denominator,
        "vehicle_exposure_2024": denominator,
        "land_area_sq_mi": 1.0,
        "eb_jurisdiction_id": "J1",
        "eb_jurisdiction_type": "municipal",
        # Classification inputs: a measurable workforce whose daytime presence its jobs explain, so
        # a residentless cell here types as `industrial_employment` and its opportunity arm is
        # opened by the taxonomy. That is the class the floor exists for.
        "special_use_acs_population": population,
        "special_use_household_population": population,
        "special_use_jobs_total": jobs,
        "special_use_jobs_education": 0.0,
        "special_use_postsecondary_count": 0.0,
        "special_use_open_natural_pixels": 0.0,
        "special_use_classified_pixels": 1_000.0,
        "special_use_gq_total_2020": 0.0,
        "special_use_gq_institutional_2020": 0.0,
        "special_use_gq_correctional_2020": 0.0,
        "special_use_gq_juvenile_2020": 0.0,
        "special_use_gq_nursing_2020": 0.0,
        "special_use_gq_college_2020": 0.0,
        "special_use_gq_military_2020": 0.0,
        "special_use_gq_other_institutional_2020": 0.0,
        "special_use_gq_other_noninstitutional_2020": 0.0,
    }
    for offense in OFFENSES_7:
        row[f"expected_count_{offense}"] = counts
        row[f"footprint_derived_count_{offense}"] = 0.0
    if with_normalizers:
        for offense in ENSEMBLE_OFFENSES:
            row[opportunity_normalizer_column(offense)] = denominator
    return row


# The cells the rule has to tell apart, plus enough ordinary neighbourhood to give the national
# rate something real to normalise on.
#
# Two families, because the two lanes reach the class by different routes. The ZERO_* pair carries
# no households at all: only the TYPED taxonomy opens its arm (the legacy household rule closes it
# outright, which is why the frozen surface never published these and the candidate did) -- this is
# the LAX / Bronx-industrial class verbatim. The LOW_* pair carries 40 households, so the household
# rule opens its arm on EVERY lane and the new floor is the only thing that can close it: that pair
# is what makes the rule visible on the ensemble-only lane and legacy-safe off both.
ZERO_TINY = "010010001011"  # no residents, no households, normalizer below 500
ZERO_LARGE = "010010001021"  # no residents, no households, normalizer above 500
LOW_RESIDENT = "010010001031"  # 49 residents, 40 households, normalizer below 500
FIFTY_RESIDENT = "010010001041"  # 50 residents, 40 households, same normalizer
ORDINARY = [f"010010002{index:02d}1" for index in range(24)]


def _rows(*, with_normalizers: bool) -> list[dict]:
    ordinary = [
        _row(
            geoid,
            population=2_000.0,
            households=700.0,
            denominator=3_000.0,
            counts=20.0,
            with_normalizers=with_normalizers,
        )
        for geoid in ORDINARY
    ]
    return [
        *ordinary,
        _row(ZERO_TINY, population=0.0, households=0.0, denominator=BELOW,
             with_normalizers=with_normalizers),
        _row(ZERO_LARGE, population=0.0, households=0.0, denominator=ABOVE,
             with_normalizers=with_normalizers),
        _row(LOW_RESIDENT, population=49.0, households=40.0, denominator=BELOW,
             with_normalizers=with_normalizers),
        _row(FIFTY_RESIDENT, population=50.0, households=40.0, denominator=BELOW,
             with_normalizers=with_normalizers),
    ]


def _finalized(
    *, taxonomy: bool = False, ensemble: bool = False, geography: str = "block_group"
) -> pd.DataFrame:
    frame = pd.DataFrame(_rows(with_normalizers=ensemble))
    config = AllocationBuildConfig(
        enable_special_use_taxonomy=taxonomy, enable_exposure_ensemble=ensemble
    )
    if geography == "tract":
        frame = frame.drop(
            columns=["block_group_geoid", "eb_jurisdiction_id", "eb_jurisdiction_type"]
        )
        frame["dominant_eb_jurisdiction_id"] = "J1"
        return _finalize_output(
            frame,
            geo_id_col="tract_id",
            population_col="population_2024",
            config=config,
            jurisdiction_col="dominant_eb_jurisdiction_id",
        ).set_index("tract_id")
    return _finalize_output(
        frame,
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=config,
    ).set_index("block_group_geoid")


LANES = [
    pytest.param({"taxonomy": True, "ensemble": False}, id="taxonomy_only"),
    pytest.param({"taxonomy": False, "ensemble": True}, id="ensemble_only"),
    pytest.param({"taxonomy": True, "ensemble": True}, id="both"),
]
# The residentless pair only has an open arm where the TYPE opens it, so the lanes that can express
# it are the taxonomy ones. Off the taxonomy the legacy household rule closes those cells for a
# different reason entirely -- asserted below rather than assumed, so this restriction stays honest
# if the household rule ever moves.
TAXONOMY_LANES = [param for param in LANES if param.values[0]["taxonomy"]]


# --- the constant ---------------------------------------------------------------------------


def test_the_floor_is_the_repos_own_support_convention_and_not_a_new_number():
    assert ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR == 500.0
    # It sits an order of magnitude above the landed floors it supplements rather than replacing
    # any of them: below 50 residents the cell needs ten times the exposure the floor asks for.
    assert ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR == 10.0 * PERSON_EXPOSURE_DENOMINATOR_FLOOR
    assert ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR > MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR
    assert ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR > BURGLARY_PREMISES_DENOMINATOR_FLOOR
    # The resident side of the predicate is the exposure floor's own threshold, reused.
    assert PERSON_EXPOSURE_DENOMINATOR_FLOOR == 50.0


def test_both_contracts_document_the_rule_and_the_constant():
    for contract in (EXPOSURE_CONTRACT, TAXONOMY_CONTRACT):
        text = contract.read_text()
        assert "ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR" in text, contract.name
        assert "500" in text, contract.name


# --- the rule, on every lane that turns it on -------------------------------------------------


@pytest.mark.parametrize("flags", TAXONOMY_LANES)
def test_a_residentless_cell_with_a_thin_normalizer_publishes_no_opportunity_index(flags):
    """The 3,825-cell class: the TYPE opened the arm, and 200 units of exposure cannot fill it."""
    surface = _finalized(**flags)
    cell = surface.loc[ZERO_TINY]
    assert cell["special_use_type"] == "industrial_employment"
    assert bool(cell["special_use_primary_rate_allowed"])  # the type says yes; the floor says no
    for offense in OFFENSES_7:
        assert pd.isna(cell[f"index_{offense}_primary"]), offense
        assert pd.isna(cell[f"rate_{offense}_primary"]), offense
        assert bool(cell[f"primary_index_suppressed_{offense}"]), offense
        assert not bool(cell[f"primary_index_publishable_{offense}"]), offense


@pytest.mark.parametrize("flags", TAXONOMY_LANES)
def test_a_residentless_cell_with_real_opportunity_mass_still_publishes(flags):
    """The rule closes a support gap, not the typed lane: 900 units publish exactly as intended."""
    surface = _finalized(**flags)
    cell = surface.loc[ZERO_LARGE]
    for offense in OFFENSES_7:
        assert pd.notna(cell[f"index_{offense}_primary"]), offense
        assert cell[f"rate_{offense}_primary"] == pytest.approx(
            1e5 * float(cell[f"expected_count_{offense}"]) / ABOVE
        ), offense
        assert cell[f"estimate_mode_{offense}"] == "count_derived", offense


@pytest.mark.parametrize("flags", TAXONOMY_LANES)
def test_the_suppressed_cell_keeps_its_count_and_its_density(flags):
    """The taxonomy's own park/prison policy: the count is real and conserved, the rate is not."""
    surface = _finalized(**flags)
    cell = surface.loc[ZERO_TINY]
    for offense in OFFENSES_7:
        assert float(cell[f"expected_count_{offense}"]) == pytest.approx(12.0), offense
        assert float(cell[f"crime_density_{offense}"]) > 0.0, offense
    assert float(cell["crime_density_total"]) > 0.0


@pytest.mark.parametrize("flags", LANES)
def test_the_floor_reports_insufficient_exposure_and_mints_no_new_vocabulary(flags):
    """The display code is the one the other exposure floors already use, on both vocabularies.

    The 49-resident cell carries households, so nothing else is competing for its status: the mode
    and the reason both name the floor. (On a cell with no households at all the pre-existing
    precedence still hands `denominator_reason` to `non_residential` and leaves `estimate_mode` on
    the floor -- an asymmetry this rule inherits rather than introduces, asserted below.)
    """
    surface = _finalized(**flags)
    cell = surface.loc[LOW_RESIDENT]
    for offense in OFFENSES_7:
        assert cell[f"estimate_mode_{offense}"] == "insufficient_exposure", offense
        assert cell[f"denominator_reason_{offense}"] == "insufficient_exposure", offense


@pytest.mark.parametrize("flags", TAXONOMY_LANES)
def test_the_landed_reason_precedence_is_inherited_and_not_rewritten(flags):
    """A residentless cell reports the denominator floor, not a household classification."""
    surface = _finalized(**flags)
    cell = surface.loc[ZERO_TINY]
    for offense in OFFENSES_7:
        assert cell[f"estimate_mode_{offense}"] == "insufficient_exposure", offense
        assert cell[f"denominator_reason_{offense}"] == "insufficient_exposure", offense


@pytest.mark.parametrize("flags", LANES)
def test_a_closed_rate_is_null_and_never_zero(flags):
    """No doubt language, and no zero standing in for an undefined rate."""
    surface = _finalized(**flags)
    for offense in OFFENSES_7:
        for column in (f"index_{offense}_primary", f"rate_{offense}_primary"):
            value = surface.loc[LOW_RESIDENT, column]
            assert pd.isna(value), (offense, column)


@pytest.mark.parametrize("flags", LANES)
def test_fifty_residents_is_the_line_and_the_landed_floor_decides_above_it(flags):
    """Two cells with IDENTICAL normalizers, 49 residents and 50, on either side of the threshold."""
    surface = _finalized(**flags)
    for offense in OFFENSES_7:
        assert pd.isna(surface.loc[LOW_RESIDENT, f"index_{offense}_primary"]), offense
        # 50 residents: the landed 50-unit floor is the only thing asked, and 200 clears it.
        assert pd.notna(surface.loc[FIFTY_RESIDENT, f"index_{offense}_primary"]), offense
        assert surface.loc[FIFTY_RESIDENT, f"estimate_mode_{offense}"] == "count_derived", offense


def test_off_the_taxonomy_the_residentless_pair_is_closed_by_the_household_rule_instead():
    """Why the residentless assertions above are parametrized on the taxonomy lanes only.

    Without the type gate, a cell with no households never had an open opportunity arm -- which is
    exactly why the frozen surface published none of these and the v2 candidate published 3,825.
    """
    for flags in ({"taxonomy": False, "ensemble": False}, {"taxonomy": False, "ensemble": True}):
        surface = _finalized(**flags)
        for geoid in (ZERO_TINY, ZERO_LARGE):
            for offense in OFFENSES_7:
                assert pd.isna(surface.loc[geoid, f"index_{offense}_primary"]), (flags, geoid)
                assert surface.loc[geoid, f"estimate_mode_{offense}"] == "non_residential", (
                    flags,
                    geoid,
                    offense,
                )


@pytest.mark.parametrize("flags", LANES)
def test_ordinary_neighbourhoods_are_untouched(flags):
    surface = _finalized(**flags)
    for offense in OFFENSES_7:
        assert surface.loc[ORDINARY, f"index_{offense}_primary"].notna().all(), offense


@pytest.mark.parametrize("flags", LANES)
def test_the_floor_is_published_on_the_frame_so_a_quoted_cell_is_re_derivable(flags):
    surface = _finalized(**flags)
    assert (
        surface[ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN]
        == float(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR)
    ).all()


def test_the_rule_holds_at_tract_support_through_the_same_code_path():
    surface = _finalized(taxonomy=True, ensemble=False, geography="tract")
    for offense in OFFENSES_7:
        assert pd.isna(surface.loc[ZERO_TINY[:11], f"index_{offense}_primary"]), offense
        assert pd.notna(surface.loc[ZERO_LARGE[:11], f"index_{offense}_primary"]), offense


# --- legacy byte-safety -----------------------------------------------------------------------


def test_the_flag_off_surface_gains_no_column_from_this_rule():
    legacy = _finalized(taxonomy=False, ensemble=False)
    assert ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN not in legacy.columns
    for flags in ({"taxonomy": True}, {"ensemble": True}):
        assert ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN in _finalized(**flags).columns


def test_the_flag_off_publication_masks_match_a_verbatim_transcription_of_the_old_rule():
    """Pre-change expression from `_finalize_output`, transcribed rather than imported."""
    legacy = _finalized(taxonomy=False, ensemble=False)
    households = pd.to_numeric(legacy["households_total"], errors="coerce").fillna(0.0)
    residential_eligible = households.ge(float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR))
    special_use_tract = (
        legacy["tract_id"].astype("string").str.zfill(11).str.slice(5, 11).str.startswith("98", na=False)
    )
    for offense in OFFENSES_7:
        denominator = pd.to_numeric(
            legacy[f"primary_denominator_{offense}"], errors="coerce"
        ).fillna(0.0)
        if offense == "burglary":
            special_use_suppressed = special_use_tract | denominator.lt(
                float(BURGLARY_PREMISES_DENOMINATOR_FLOOR)
            )
        else:
            special_use_suppressed = special_use_tract
        if offense in {"murder", "rape", "robbery", "aggravated_assault", "larceny"}:
            insufficient = denominator.lt(float(PERSON_EXPOSURE_DENOMINATOR_FLOOR))
        elif offense == "motor_vehicle_theft":
            insufficient = denominator.lt(float(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR))
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


def test_the_discriminating_cell_publishes_off_the_lane_and_is_closed_on_it():
    """40 households, 49 residents, a 200-unit normalizer.

    The household floor opens the arm on every lane and the cell types as `ordinary`, so nothing
    about the TYPE or the ensemble decides it -- only the new floor does. It publishes on the legacy
    lane exactly as it always did, which is the byte-safety claim stated as an outcome rather than
    as an absence, and it is closed on every lane that turns the floor on.
    """
    legacy = _finalized(taxonomy=False, ensemble=False)
    for offense in OFFENSES_7:
        assert pd.notna(legacy.loc[LOW_RESIDENT, f"index_{offense}_primary"]), offense
        assert legacy.loc[LOW_RESIDENT, f"estimate_mode_{offense}"] == "count_derived", offense
    for flags in ({"taxonomy": True}, {"ensemble": True}, {"taxonomy": True, "ensemble": True}):
        surface = _finalized(**flags)
        for offense in OFFENSES_7:
            assert pd.isna(surface.loc[LOW_RESIDENT, f"index_{offense}_primary"]), (flags, offense)
            assert (
                surface.loc[LOW_RESIDENT, f"estimate_mode_{offense}"] == "insufficient_exposure"
            ), (flags, offense)


# --- the release validator's independent mirror -----------------------------------------------


def test_the_validator_transcribes_the_same_constant():
    validator = _validator_module()
    assert validator.ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR == ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR
    assert (
        validator.ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN
        == ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN
    )


def test_the_validator_holds_the_rule_on_either_lane_and_neither_off_both():
    validator = _validator_module()
    assert validator.LEGACY_RELEASE_CONTRACT.zero_resident_opportunity_floor is False
    for lane in ("special_use_taxonomy", "exposure_ensemble"):
        contract = validator.ReleaseContract(lanes={lane})
        assert contract.zero_resident_opportunity_floor is True, lane


def test_the_validator_expects_the_floor_column_only_on_a_v2_surface():
    validator = _validator_module()
    legacy = set(validator._expected_columns(geography="block_group"))
    assert ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN not in legacy
    for lane in ("special_use_taxonomy", "exposure_ensemble"):
        columns = set(validator._expected_columns(geography="block_group", **{lane: True}))
        assert ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR_COLUMN in columns, lane


def test_the_validator_mirror_reproduces_the_producers_suppression_set():
    """The mirror recomputes the rule from published fields; it must reach the same cells."""
    validator = _validator_module()
    surface = _finalized(taxonomy=True, ensemble=False).reset_index()
    population = pd.to_numeric(surface["population_2024"], errors="coerce").fillna(0.0)
    below_resident_threshold = population.lt(
        float(validator.PERSON_EXPOSURE_DENOMINATOR_FLOOR)
    )
    for offense in OFFENSES_7:
        denominator = pd.to_numeric(
            surface[f"primary_denominator_{offense}"], errors="coerce"
        ).fillna(0.0)
        closed_by_the_floor = below_resident_threshold & denominator.lt(
            float(validator.ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR)
        )
        published = surface[f"primary_index_publishable_{offense}"].fillna(False).astype(bool)
        # Necessary direction: no cell the mirror closes may be published by the producer.
        assert not bool((closed_by_the_floor & published).any()), offense
        # And the rule is live rather than vacuous on this fixture.
        assert bool(closed_by_the_floor.any()), offense


def test_the_manifest_declares_the_policy_and_the_validator_checks_it():
    validator = _validator_module()
    resolved = {
        "special_use_taxonomy": {"enabled": True},
        "zero_resident_opportunity_rate_policy": {
            "enabled": True,
            "resident_threshold": float(PERSON_EXPOSURE_DENOMINATOR_FLOOR),
            "opportunity_normalizer_floor": float(ZERO_RESIDENT_OPPORTUNITY_RATE_FLOOR),
            "offenses": sorted(OFFENSES_7),
            "floor_estimate_mode": "insufficient_exposure",
        },
    }
    contract = validator.ReleaseContract(
        lanes={"special_use_taxonomy"},
        manifest_present=True,
        zero_resident_opportunity_policy=resolved["zero_resident_opportunity_rate_policy"],
    )
    issues: list[str] = []
    validator._check_release_contract(
        contract=contract, state_output_dir=REPO_ROOT / "state" / "output", issues=issues
    )
    assert not [issue for issue in issues if "zero_resident" in issue]

    # A build that drops the block, or moves the number, fails rather than exempting itself.
    for broken in ({}, {**resolved["zero_resident_opportunity_rate_policy"], "opportunity_normalizer_floor": 50.0}):
        broken_issues: list[str] = []
        validator._check_release_contract(
            contract=validator.ReleaseContract(
                lanes={"special_use_taxonomy"},
                manifest_present=True,
                zero_resident_opportunity_policy=broken,
            ),
            state_output_dir=REPO_ROOT / "state" / "output",
            issues=broken_issues,
        )
        assert [issue for issue in broken_issues if "zero_resident" in issue], broken


def test_the_taxonomy_feature_columns_are_all_present_on_the_fixture():
    """Guards the fixture itself: a missing classification input would fail the cell CLOSED and
    make the suppression assertions above pass for the wrong reason."""
    surface = _finalized(taxonomy=True)
    for column in SPECIAL_USE_FEATURE_COLUMNS:
        assert column in surface.columns
    assert surface.loc[ZERO_TINY, "special_use_type"] == "industrial_employment"
    assert surface.loc[ZERO_LARGE, "special_use_type"] == "industrial_employment"
    assert surface.loc[LOW_RESIDENT, "special_use_type"] == "ordinary"
    assert surface.loc[FIFTY_RESIDENT, "special_use_type"] == "ordinary"
