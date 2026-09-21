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
from pandas.testing import assert_frame_equal
import pytest
import crimerisk.allocation as allocation

from crimerisk.allocation import (
    COUNTY_NONMUNICIPAL_OVERLAP_KIND,
    STATE_REMAINDER_TYPE,
    _apply_exclusive_footprint_displacement,
    _assert_custom_footprint_overrides_have_rows,
    _build_overlap_evidence_spine,
    _build_exclusive_footprint_displacement,
    _nonmunicipal_bg_exposure_share,
    _parse_registry_flag,
    _reviewed_succession_county_anchor_mask,
    _smoothed_county_remainder_partition,
    _smoothed_overlap_partition,
    _state_police_county_subunit_mask,
    _statewide_overlap_crosswalk_rows,
)
from crimerisk.jurisdiction_reference import assert_tribal_agencies_not_auto_placed
from crimerisk.reference import matches_tribal_name


def test_smoothed_county_partition_gives_zero_year_county_a_risk_prior() -> None:
    controls = pd.DataFrame(
        {"state_fips": ["01"], "offense": ["robbery"], "state_target": [100.0]}
    )
    crosswalk = pd.DataFrame(
        {
            "state_fips": ["01", "01"],
            "block_group_geoid": ["010010001001", "010030001001"],
            "jurisdiction_type": [STATE_REMAINDER_TYPE, STATE_REMAINDER_TYPE],
            "allocation_share": [1.0, 1.0],
            "pop20": [100.0, 100.0],
        }
    )
    groups = pd.DataFrame(
        {
            "state_fips": ["01", "01"],
            "offense": ["robbery", "robbery"],
            "group_kind": ["county_remainder", "county_remainder"],
            "group_id": [
                "01:state_nonmunicipal_remainder:county:01001",
                "01:state_nonmunicipal_remainder:county:01003",
            ],
            "risk_signal_count": [0.0, 100.0],
            "weight": [1.0, 1.0],
        }
    )
    out = _smoothed_county_remainder_partition(
        remainder_controls=controls,
        agency_groups=groups,
        bg_crosswalk=crosswalk,
        bg_prior=None,
    )
    county = out[out["group_kind"].eq("county_remainder")].set_index("group_id")
    assert county.loc[
        "01:state_nonmunicipal_remainder:county:01001", "target_count"
    ] > 0.0
    assert county.loc[
        "01:state_nonmunicipal_remainder:county:01003", "target_count"
    ] > county.loc[
        "01:state_nonmunicipal_remainder:county:01001", "target_count"
    ]
    assert out["target_count"].sum() == pytest.approx(100.0)


def test_smoothed_overlap_partition_uses_multiyear_signal_and_conserves_parent() -> None:
    controls = pd.DataFrame(
        {"state_fips": ["01"], "offense": ["robbery"], "state_target": [100.0]}
    )
    groups = pd.DataFrame(
        {
            "state_fips": ["01", "01"],
            "offense": ["robbery", "robbery"],
            "group_kind": ["county_overlap", "county_overlap"],
            "group_id": ["01001", "01003"],
            "risk_signal_count": [25.0, 75.0],
            "weight": [1.0, 1.0],
        }
    )
    out = _smoothed_overlap_partition(
        overlap_controls=controls,
        agency_groups=groups,
    ).set_index("group_id")
    assert out.loc["01001", "target_count"] == pytest.approx(25.0)
    assert out.loc["01003", "target_count"] == pytest.approx(75.0)
    assert out["target_count"].sum() == pytest.approx(100.0)


def _overlap_crosswalk(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ori": [ori for ori, _weight in rows],
            "state_fips": ["01"] * len(rows),
            "jurisdiction_id": ["01:statewide_overlap_layer"] * len(rows),
            "weight": [weight for _ori, weight in rows],
            "geometry_hint": ["statewide"] * len(rows),
            "overlap_subtype": ["statewide"] * len(rows),
        }
    )


def test_overlap_evidence_spine_retains_historical_agency_missing_current_year() -> None:
    preferred = pd.DataFrame(
        {
            "ori9": ["CURRENT01"],
            "state_fips": ["01"],
            "offense": ["robbery"],
            "preferred_count": [10.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "ori9": ["CURRENT01", "HISTORY01"],
            "state_fips": ["01", "01"],
            "offense": ["robbery", "robbery"],
            "reported_count_current_supported": [10.0, 0.0],
            "agency_adjustment_count": [0.0, 30.0],
        }
    )
    out = _build_overlap_evidence_spine(
        preferred=preferred,
        agency_estimates=estimates,
        overlap_crosswalk=_overlap_crosswalk([("CURRENT01", 1.0), ("HISTORY01", 1.0)]),
    ).set_index("ori9")

    assert set(out.index) == {"CURRENT01", "HISTORY01"}
    assert bool(out.loc["HISTORY01", "agency_estimate_present"])
    assert not bool(out.loc["HISTORY01", "preferred_observation_present"])
    assert pd.isna(out.loc["HISTORY01", "preferred_count"])


def test_overlap_evidence_spine_is_identity_when_every_estimate_is_current() -> None:
    preferred = pd.DataFrame(
        {
            "ori9": ["A", "B"],
            "state_fips": ["01", "01"],
            "offense": ["burglary", "burglary"],
            "preferred_count": [4.0, 6.0],
        }
    )
    estimates = preferred[["ori9", "state_fips", "offense"]].assign(
        reported_count_current_supported=[4.0, 6.0], agency_adjustment_count=0.0
    )
    out = _build_overlap_evidence_spine(
        preferred=preferred,
        agency_estimates=estimates,
        overlap_crosswalk=_overlap_crosswalk([("A", 1.0), ("B", 1.0)]),
    )

    assert_frame_equal(
        out[["ori9", "preferred_count"]].sort_values("ori9").reset_index(drop=True),
        preferred[["ori9", "preferred_count"]].sort_values("ori9").reset_index(drop=True),
        check_dtype=False,
    )
    assert out["preferred_observation_present"].all()


def test_overlap_evidence_spine_distinguishes_reported_zero_from_missing() -> None:
    preferred = pd.DataFrame(
        {
            "ori9": ["ZERO"],
            "state_fips": ["01"],
            "offense": ["murder"],
            "preferred_count": [0.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "ori9": ["ZERO", "MISSING"],
            "state_fips": ["01", "01"],
            "offense": ["murder", "murder"],
        }
    )
    out = _build_overlap_evidence_spine(
        preferred=preferred,
        agency_estimates=estimates,
        overlap_crosswalk=_overlap_crosswalk([("ZERO", 1.0), ("MISSING", 1.0)]),
    ).set_index("ori9")

    assert out.loc["ZERO", "preferred_count"] == 0.0
    assert bool(out.loc["ZERO", "preferred_observation_present"])
    assert pd.isna(out.loc["MISSING", "preferred_count"])
    assert not bool(out.loc["MISSING", "preferred_observation_present"])


def test_smoothed_overlap_partition_respects_split_weights_and_unlocated_conservation() -> None:
    controls = pd.DataFrame(
        {"state_fips": ["01"], "offense": ["robbery"], "state_target": [120.0]}
    )
    groups = pd.DataFrame(
        {
            "state_fips": ["01", "01", "01"],
            "offense": ["robbery"] * 3,
            "group_kind": ["county_overlap", "county_overlap", "unlocated_overlap"],
            "group_id": ["01001", "01003", "01"],
            "risk_signal_count": [80.0, 80.0, 40.0],
            "weight": [0.25, 0.75, 1.0],
        }
    )
    out = _smoothed_overlap_partition(
        overlap_controls=controls,
        agency_groups=groups,
    ).set_index(["group_kind", "group_id"])

    assert out.loc[("county_overlap", "01001"), "target_count"] == pytest.approx(20.0)
    assert out.loc[("county_overlap", "01003"), "target_count"] == pytest.approx(60.0)
    assert out.loc[("unlocated_overlap", "01"), "target_count"] == pytest.approx(40.0)
    assert out["target_count"].sum() == pytest.approx(120.0)


def test_build_overlap_targets_integrates_missing_current_agency_estimate(monkeypatch) -> None:
    preferred = pd.DataFrame(
        {
            "ori9": ["CURRENT01"],
            "state_fips": ["01"],
            "offense": ["robbery"],
            "preferred_count": [10.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "ori9": ["CURRENT01", "HISTORY01"],
            "state_fips": ["01", "01"],
            "offense": ["robbery", "robbery"],
            "reported_count_current_supported": [10.0, 0.0],
            "agency_adjustment_count": [0.0, 30.0],
            "estimated_count": [10.0, 30.0],
        }
    )
    crosswalk = _overlap_crosswalk([("CURRENT01", 1.0), ("HISTORY01", 1.0)])
    crosswalk.loc[crosswalk["ori"].eq("HISTORY01"), ["geometry_hint", "overlap_subtype"]] = [
        "place",
        "local",
    ]
    master = pd.DataFrame(
        {
            "ori9": ["CURRENT01", "HISTORY01"],
            "state_fips": ["01", "01"],
            "state_abbr": ["AL", "AL"],
            "county_fips": [pd.NA, pd.NA],
            "place_fips": [pd.NA, "12345"],
            "agency_name_std": ["CURRENT", "HISTORICAL"],
            "agency_type_norm": ["other", "other"],
            "county_fips_source": [pd.NA, pd.NA],
        }
    )
    monkeypatch.setattr(allocation, "build_agency_preferred_observations", lambda **_kwargs: preferred)
    monkeypatch.setattr(allocation, "_load_crosswalk", lambda _paths: crosswalk)
    monkeypatch.setattr(allocation, "_load_agency_master", lambda _paths: master)
    monkeypatch.setattr(
        allocation,
        "_load_jurisdiction_master",
        lambda _paths: pd.DataFrame(
            {
                "jurisdiction_type": ["municipal"],
                "geo_type": ["place"],
                "geoid": ["0112345"],
                "jurisdiction_id": ["01:municipal:place:0112345"],
            }
        ),
    )
    monkeypatch.setattr(allocation, "_load_overlap_footprint_overrides", lambda _paths: pd.DataFrame())
    monkeypatch.setattr(allocation, "_load_overlap_custom_footprints", lambda _paths: pd.DataFrame())
    monkeypatch.setattr(allocation, "load_explicit_succession_rulings", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(
        allocation,
        "_build_agency_risk_signals",
        lambda **_kwargs: estimates[["ori9", "state_fips", "offense", "estimated_count"]]
        .rename(columns={"estimated_count": "risk_signal_count"})
        .assign(clean_year_count=1),
    )
    controls = pd.DataFrame(
        {
            "state_fips": ["01"],
            "offense": ["robbery"],
            "jurisdiction_type": ["statewide_overlap_layer"],
            "adjusted_count_ags_core": [80.0],
            "reported_count_preferred": [20.0],
            "control_surface": ["smoothed"],
        }
    )

    out = allocation._build_overlap_group_targets(
        paths=object(),
        controls=controls,
        year=2025,
        enable_county_anchoring=False,
        agency_estimates=estimates,
    )

    assert out["target_count"].sum() == pytest.approx(80.0)
    assert out.loc[out["group_kind"].eq("statewide_overlap"), "target_count"].sum() == pytest.approx(20.0)
    assert out.loc[out["group_kind"].eq("municipal_place_overlap"), "target_count"].sum() == pytest.approx(60.0)


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


def test_overlap_lane_follows_the_target_jurisdiction_not_relationship_label() -> None:
    crosswalk = pd.DataFrame(
        [
            {
                "ori": "MDMSP2100",
                "state_fips": "24",
                "jurisdiction_id": "24:statewide_overlap_layer",
                "relationship_type": "contract_covering_footprint",
                "weight": 0.975,
            },
            {
                "ori": "MDMSP2100",
                "state_fips": "24",
                "jurisdiction_id": "24:municipal:place:2447875",
                "relationship_type": "contract_covered_footprint",
                "weight": 0.025,
            },
        ]
    )
    selected = _statewide_overlap_crosswalk_rows(crosswalk)
    assert selected[["ori", "jurisdiction_id", "weight"]].to_dict("records") == [
        {
            "ori": "MDMSP2100",
            "jurisdiction_id": "24:statewide_overlap_layer",
            "weight": 0.975,
        }
    ]


def test_reviewed_reporter_migration_carries_the_county_anchor_forward() -> None:
    frame = pd.DataFrame(
        [
            {
                "ori9": "MDMSP2100",
                "state_abbr": "MD",
                "county_fips": "001",
            },
            {
                "ori9": "MDMSP2100",
                "state_abbr": "MD",
                "county_fips": "013",
            },
        ]
    )
    rulings = pd.DataFrame(
        [
            {
                "successor_ori": "MDMSP2100",
                "state": "MD",
                "county_fips": "001",
            }
        ]
    )
    assert _reviewed_succession_county_anchor_mask(frame, rulings).tolist() == [
        True,
        False,
    ]


def test_reviewed_county_identity_localizes_named_barrack_but_not_statewide_hq() -> None:
    frame = pd.DataFrame(
        {
            "agency_type_norm": ["state_law_enforcement", "state_law_enforcement"],
        }
    )
    names = pd.Series(["STATE POLICE NORTH EAST", "MSP STATEWIDE"])
    county = pd.Series([True, True])
    reviewed = pd.Series([True, True])
    assert _state_police_county_subunit_mask(
        frame,
        names,
        county,
        reviewed_county_identity=reviewed,
    ).tolist() == [True, False]


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
