"""Stage 2 footprint resolution: who a footprint belongs to and what it may take over.

Pins the Stage 2 fix batch. Stage 1 pins who an agency IS; this file pins WHICH PIECE OF
GROUND its mass lands on, and the fail-closed rules that stop a footprint from being arrived
at by omission:

  * tribal agencies never reach a municipality through an automatic lane (Class D);
  * the tribal name test matches word boundaries, not substrings;
  * `localize_to_custom_footprint` with no footprint rows FAILS the build instead of silently
    spreading the agency over the whole state;
  * an EXCLUSIVE (remainder-displacing) footprint must say how much of each block group it
    takes over, and the county remainder loses exactly that much and no more;
  * a state-police county anchor spreads over non-municipal exposure, falling back to the
    whole county where there is none rather than stranding the mass.
"""

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    COUNTY_NONMUNICIPAL_OVERLAP_KIND,
    STATE_REMAINDER_TYPE,
    _apply_exclusive_footprint_displacement,
    _assert_custom_footprint_overrides_have_rows,
    _build_exclusive_footprint_displacement,
    _nonmunicipal_bg_exposure_share,
    _parse_registry_flag,
)
from crimerisk.jurisdiction_reference import assert_tribal_agencies_not_auto_placed
from crimerisk.reference import matches_tribal_name


# --- tribal identity and the name test ---------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "COLVILLE TRIBAL",
        "NAVAJO NATION",
        "SAC AND FOX TRIBE",
        "ISLETA PUEBLO",
        "TABLE MOUNTAIN RANCHERIA",
        "BIA LAW ENFORCEMENT",
    ],
)
def test_tribal_name_test_matches_real_tribal_agencies(name: str) -> None:
    assert matches_tribal_name(name)


@pytest.mark.parametrize(
    "name",
    [
        # The substring form matched all of these; each is a measured false hit from the
        # Stage 2 screen (74 false hits over all 26,767 links).
        "NATIONAL PARK SERVICE",
        "NATIONAL SECURITY AGENCY",
        "NATIONAL INSTITUTES OF HEALTH",
        "INDIANAPOLIS POLICE DEPARTMENT",
        "NATIONAL MONUMENT RANGERS",
    ],
)
def test_tribal_name_test_rejects_the_measured_false_hits(name: str) -> None:
    assert not matches_tribal_name(name)


# --- Class D gate: no automatic municipal placement of a tribal agency --------


def _local_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_tribal_gate_fails_the_build_on_an_automatic_municipal_placement() -> None:
    frame = _local_frame(
        [
            {
                "ori9": "WADI05700",
                "agency_name_std": "COLVILLE TRIBAL",
                "final_decision": "municipal_place",
                "resolved_geoid": "5348540",
                "resolution_source": "provisional_auto",
            }
        ]
    )
    with pytest.raises(ValueError, match="automatic lane"):
        assert_tribal_agencies_not_auto_placed(
            frame, tribal_oris={"WADI05700"}, licensed_oris=set()
        )


def test_tribal_gate_accepts_a_reviewed_registry_row() -> None:
    frame = _local_frame(
        [
            {
                "ori9": "WADI05700",
                "agency_name_std": "COLVILLE TRIBAL",
                "final_decision": "municipal_place",
                "resolved_geoid": "5348540",
                # The canonicalization passes rewrite resolution_source, so the licence is
                # keyed on registry membership, not on this column.
                "resolution_source": "cdp_cousub_canonicalized",
            }
        ]
    )
    assert_tribal_agencies_not_auto_placed(
        frame, tribal_oris={"WADI05700"}, licensed_oris={"WADI05700"}
    )


def test_tribal_gate_ignores_a_tribal_agency_routed_off_the_municipal_lane() -> None:
    frame = _local_frame(
        [
            {
                "ori9": "WADI05700",
                "agency_name_std": "COLVILLE TRIBAL",
                "final_decision": "reclassify_overlap",
                "resolved_geoid": None,
                "resolution_source": "local_resolution_override",
            }
        ]
    )
    assert_tribal_agencies_not_auto_placed(
        frame, tribal_oris={"WADI05700"}, licensed_oris=set()
    )


def test_tribal_gate_ignores_non_tribal_municipal_placements() -> None:
    frame = _local_frame(
        [
            {
                "ori9": "CA0071300",
                "agency_name_std": "KENSINGTON POLICE PROTECTION DISTRICT",
                "final_decision": "municipal_place",
                "resolved_geoid": "0637918",
                "resolution_source": "provisional_auto",
            }
        ]
    )
    # Kensington CA carries LEAIC's LG_POPULATION sentinel and is NOT tribal; the production
    # flag excludes it, so the gate must too.
    assert_tribal_agencies_not_auto_placed(frame, tribal_oris=set(), licensed_oris=set())


# --- the custom-footprint fail-open ------------------------------------------


def _overrides(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_custom_footprint_override_without_rows_fails_the_build() -> None:
    overrides = _overrides(
        [
            {
                "ori9": "NJ0073200",
                "final_overlap_treatment": "localize_to_custom_footprint",
                "displaces_county_remainder": False,
            }
        ]
    )
    with pytest.raises(ValueError, match="NJ0073200"):
        _assert_custom_footprint_overrides_have_rows(overrides, pd.DataFrame(columns=["ori9"]))


def test_declared_statewide_overlap_is_allowed_without_footprint_rows() -> None:
    overrides = _overrides(
        [
            {
                "ori9": "NJ0073200",
                "final_overlap_treatment": "keep_statewide_overlap",
                "displaces_county_remainder": False,
            }
        ]
    )
    _assert_custom_footprint_overrides_have_rows(overrides, pd.DataFrame(columns=["ori9"]))


def test_registry_flag_treats_blank_as_false_and_rejects_garbage() -> None:
    parsed = _parse_registry_flag(pd.Series(["TRUE", "", None, "false"]))
    assert parsed.tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        _parse_registry_flag(pd.Series(["maybe"]))


# --- EXCLUSIVE footprints displace the county remainder ----------------------


def _footprints(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _displacing_registry(coverage: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    overrides = _overrides(
        [
            {
                "ori9": "SD0600200",
                "final_overlap_treatment": "localize_to_custom_footprint",
                "displaces_county_remainder": True,
            }
        ]
    )
    footprints = _footprints(
        [
            {
                "ori9": "SD0600200",
                "state_fips": "46",
                "bg_id": "460071234001",
                "weight_share": 1.0,
                "bg_population_coverage_share": coverage,
            }
        ]
    )
    return overrides, footprints


def test_displacement_requires_a_bg_population_coverage_share() -> None:
    overrides, footprints = _displacing_registry(np.nan)
    with pytest.raises(ValueError, match="bg_population_coverage_share"):
        _build_exclusive_footprint_displacement(
            overrides=overrides, custom_footprints=footprints
        )


def test_displacement_is_the_union_not_the_sum_across_shared_footprints() -> None:
    """Two ORIs on one shared footprint must not displace twice.

    24 footprints are shared by two ORIs each (duplicate tribal/BIA ORI pairs and joint
    Oklahoma OTSAs); the coverage share is defined as the UNION so production takes the max.
    """
    overrides = _overrides(
        [
            {
                "ori9": "SD0600200",
                "final_overlap_treatment": "localize_to_custom_footprint",
                "displaces_county_remainder": True,
            },
            {
                "ori9": "SDDI06000",
                "final_overlap_treatment": "localize_to_custom_footprint",
                "displaces_county_remainder": True,
            },
        ]
    )
    footprints = _footprints(
        [
            {
                "ori9": ori,
                "state_fips": "46",
                "bg_id": "460071234001",
                "weight_share": 1.0,
                "bg_population_coverage_share": 0.6,
            }
            for ori in ("SD0600200", "SDDI06000")
        ]
    )
    displacement = _build_exclusive_footprint_displacement(
        overrides=overrides, custom_footprints=footprints
    )
    assert len(displacement) == 1
    assert displacement["displaced_share"].iloc[0] == pytest.approx(0.6)


def _bg_crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["46", "46", "46"],
            "block_group_geoid": ["460071234001", "460071234001", "460079999001"],
            "jurisdiction_id": [
                "46:municipal:place:4600100",
                "46:state_nonmunicipal_remainder",
                "46:state_nonmunicipal_remainder",
            ],
            "jurisdiction_type": ["municipal", STATE_REMAINDER_TYPE, STATE_REMAINDER_TYPE],
            "pop20": [400.0, 600.0, 1000.0],
            "allocation_share": [0.4, 0.6, 1.0],
        }
    )


def test_displacement_removes_exactly_the_covered_share_from_the_remainder() -> None:
    displacement = pd.DataFrame(
        {"state_fips": ["46"], "bg_id": ["460071234001"], "displaced_share": [0.5]}
    )
    out = _apply_exclusive_footprint_displacement(_bg_crosswalk(), displacement)
    remainder = out[out["jurisdiction_type"].eq(STATE_REMAINDER_TYPE)].set_index(
        "block_group_geoid"
    )
    # Half the covered block group's remainder exposure leaves; the untouched one is intact.
    assert remainder.loc["460071234001", "allocation_share"] == pytest.approx(0.3)
    assert remainder.loc["460079999001", "allocation_share"] == pytest.approx(1.0)
    # Municipal support is never touched by a reservation footprint.
    municipal = out[out["jurisdiction_type"].eq("municipal")]
    assert municipal["allocation_share"].iloc[0] == pytest.approx(0.4)


def test_fully_displaced_remainder_rows_are_dropped_not_left_at_zero() -> None:
    displacement = pd.DataFrame(
        {"state_fips": ["46"], "bg_id": ["460071234001"], "displaced_share": [1.0]}
    )
    out = _apply_exclusive_footprint_displacement(_bg_crosswalk(), displacement)
    remaining = set(
        out.loc[out["jurisdiction_type"].eq(STATE_REMAINDER_TYPE), "block_group_geoid"]
    )
    assert remaining == {"460079999001"}


def test_displacement_that_would_empty_a_state_remainder_fails_closed() -> None:
    displacement = pd.DataFrame(
        {
            "state_fips": ["46", "46"],
            "bg_id": ["460071234001", "460079999001"],
            "displaced_share": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="strand"):
        _apply_exclusive_footprint_displacement(_bg_crosswalk(), displacement)


# --- state-police non-municipal exposure ------------------------------------


def test_nonmunicipal_exposure_share_is_the_remainder_population_share() -> None:
    shares = _nonmunicipal_bg_exposure_share(_bg_crosswalk()).set_index("bg_id")
    assert shares.loc["460071234001", "nonmunicipal_share"] == pytest.approx(0.6)
    assert shares.loc["460079999001", "nonmunicipal_share"] == pytest.approx(1.0)


def test_nonmunicipal_exposure_share_omits_fully_municipal_block_groups() -> None:
    crosswalk = pd.DataFrame(
        {
            "state_fips": ["44"],
            "block_group_geoid": ["440070001001"],
            "jurisdiction_id": ["44:municipal:place:4400100"],
            "jurisdiction_type": ["municipal"],
            "pop20": [1000.0],
            "allocation_share": [1.0],
        }
    )
    shares = _nonmunicipal_bg_exposure_share(crosswalk)
    # Rhode Island, Virginia's independent cities and Baltimore city have no non-municipal
    # ground at all; the county spread must fall back rather than strand the mass, which the
    # allocator does by treating a zero non-municipal total as "use the whole county".
    assert shares.empty


def test_county_nonmunicipal_overlap_kind_is_distinct_from_county_overlap() -> None:
    assert COUNTY_NONMUNICIPAL_OVERLAP_KIND != "county_overlap"
