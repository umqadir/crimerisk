from __future__ import annotations

import pandas as pd
import pytest

import crimerisk.allocation as allocation
from crimerisk.allocation import (
    STATE_REMAINDER_TYPE,
    _apply_allocation_primary_service_displacement,
    _apply_exclusive_footprint_displacement,
    _apply_model_only_allocation_envelopes,
    _apply_model_lane_shares,
    _assert_unlocated_mass_conservation,
    _build_allocation_exclusive_footprint_displacement,
    _build_exclusive_footprint_displacement,
    _build_jurisdiction_component_allocations,
    _build_service_state_transfer_ledger,
    _build_overlap_allocations,
    _custom_footprint_component_shares,
    _load_primary_service_response_policies,
    _redistribute_zero_target_components,
    _scale_components_by_source_state,
)
from crimerisk.paths import RepoPaths


def test_service_wide_footprint_normalizes_once_across_destination_states() -> None:
    frame = pd.DataFrame(
        [
            {
                "state_fips": "04",
                "source_state_fips": "04",
                "ori9": "AZ0018900",
                "canonical_target_ori": "AZ0018900",
                "service_scope_id": "navajo_nation_pd",
                "allocation_scope": "service_wide",
                "offense": "aggravated_assault",
                "weight_share_basis": "service_area_prior",
                "weight_share": 0.25,
                "bg_land_area_coverage_share": 0.25,
                "bg_weight": 100.0,
            },
            {
                "state_fips": "35",
                "source_state_fips": "04",
                "ori9": "AZ0018900",
                "canonical_target_ori": "AZ0018900",
                "service_scope_id": "navajo_nation_pd",
                "allocation_scope": "service_wide",
                "offense": "aggravated_assault",
                "weight_share_basis": "service_area_prior",
                "weight_share": 0.75,
                "bg_land_area_coverage_share": 0.75,
                "bg_weight": 100.0,
            },
        ]
    )

    out = _custom_footprint_component_shares(frame)

    assert out["component_share"].tolist() == pytest.approx([0.25, 0.75])
    assert out["component_share"].sum() == pytest.approx(1.0)


def test_service_area_prior_handles_whole_sliver_and_zero_population_members() -> None:
    frame = pd.DataFrame(
        [
            {
                "state_fips": "04",
                "source_state_fips": "04",
                "ori9": "AZ0018900",
                "canonical_target_ori": "AZ0018900",
                "service_scope_id": "navajo_nation_police",
                "allocation_scope": "service_wide",
                "offense": "larceny",
                "weight_share_basis": "service_area_prior",
                "weight_share": population_share,
                "bg_population_coverage_share": population_coverage,
                "bg_land_area_coverage_share": area_coverage,
                "bg_weight": prior,
            }
            for population_share, population_coverage, area_coverage, prior in (
                (0.9, 1.0, 1.0, 10.0),
                (0.1, 0.1, 0.1, 10.0),
                (0.0, 0.0, 0.5, 5.0),
            )
        ]
    )

    out = _custom_footprint_component_shares(frame)

    # Allocation uses expected-count prior × within-BG land coverage: 10, 1, 2.5.
    assert out["component_share"].tolist() == pytest.approx(
        [10.0 / 13.5, 1.0 / 13.5, 2.5 / 13.5]
    )
    assert out.iloc[2]["component_share"] > 0.0


def test_service_area_prior_fails_closed_without_own_service_area_coverage() -> None:
    frame = pd.DataFrame(
        [
            {
                "state_fips": "04",
                "source_state_fips": "04",
                "ori9": "AZ0018900",
                "canonical_target_ori": "AZ0018900",
                "service_scope_id": "navajo_nation_police",
                "allocation_scope": "service_wide",
                "offense": "larceny",
                "weight_share_basis": "service_area_prior",
                "weight_share": 1.0,
                "bg_weight": 2.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="require bg_land_area_coverage_share"):
        _custom_footprint_component_shares(frame)


def test_zero_resident_service_land_does_not_displace_the_county_remainder() -> None:
    overrides = pd.DataFrame(
        {
            "ori9": ["AZ0018900"],
            "displaces_county_remainder": [True],
        }
    )
    service_footprint = pd.DataFrame(
        {
            "ori9": ["AZ0018900"],
            "state_fips": ["35"],
            "bg_id": ["350010001001"],
            "allocation_scope": ["service_wide"],
            "bg_population_coverage_share": [0.0],
            "bg_land_area_coverage_share": [0.5],
        }
    )
    displacement = _build_exclusive_footprint_displacement(
        overrides=overrides,
        custom_footprints=service_footprint,
    )
    assert displacement.empty

    county_remainder = pd.DataFrame(
        {
            "state_fips": ["35"],
            "block_group_geoid": ["350010001001"],
            "jurisdiction_type": [STATE_REMAINDER_TYPE],
            "allocation_share": [0.8],
            "pop_share": [0.8],
        }
    )
    out = _apply_exclusive_footprint_displacement(county_remainder, displacement)
    pd.testing.assert_frame_equal(out, county_remainder)


def test_missing_union_displacement_fraction_remains_invalid_for_service_land() -> None:
    overrides = pd.DataFrame(
        {"ori9": ["AZ0018900"], "displaces_county_remainder": [True]}
    )
    service_footprint = pd.DataFrame(
        {
            "ori9": ["AZ0018900"],
            "state_fips": ["35"],
            "bg_id": ["350010001001"],
            "allocation_scope": ["service_wide"],
            "bg_population_coverage_share": [pd.NA],
            "bg_land_area_coverage_share": [0.5],
        }
    )
    with pytest.raises(ValueError, match="must carry bg_population_coverage_share"):
        _build_exclusive_footprint_displacement(
            overrides=overrides,
            custom_footprints=service_footprint,
        )


def test_primary_service_precedence_uses_own_fraction_inside_carveout() -> None:
    overrides = pd.DataFrame(
        {"ori9": ["AZ0018900"], "displaces_county_remainder": [True]}
    )
    footprint = pd.DataFrame(
        {
            "ori9": ["AZ0018900"],
            "state_fips": ["04"],
            "bg_id": ["040019440002"],
            "service_scope_id": ["navajo_nation_police"],
            "bg_population_coverage_share": [0.9],
            "bg_service_population_coverage_share": [0.4],
        }
    )
    base = _build_exclusive_footprint_displacement(
        overrides=overrides,
        custom_footprints=footprint,
        concurrent_jurisdiction_carveouts=pd.DataFrame({"county_geoid": ["04001"]}),
    )
    assert base.empty

    displacement = _build_allocation_exclusive_footprint_displacement(
        overrides=overrides,
        custom_footprints=footprint,
        concurrent_jurisdiction_carveouts=pd.DataFrame({"county_geoid": ["04001"]}),
        primary_response_policies=pd.DataFrame(
            {
                "service_scope_id": ["navajo_nation_police"],
                "primary_response_policy": [
                    "displace_county_remainder_within_reviewed_carveout"
                ],
            }
        ),
    )
    assert displacement[["state_fips", "bg_id"]].to_dict("records") == [
        {"state_fips": "04", "bg_id": "040019440002"}
    ]
    assert displacement.loc[0, "displaced_share"] == pytest.approx(0.4)


def test_primary_service_policy_loader_is_fail_closed(tmp_path) -> None:
    path = tmp_path / "configs" / "primary_service_response_policies.csv"
    path.parent.mkdir()
    valid = pd.DataFrame(
        [{
            "service_scope_id": "navajo_nation_police",
            "primary_response_policy": (
                "displace_county_remainder_within_reviewed_carveout"
            ),
            "official_source_ref": "https://example.test/official",
            "evidence_artifact": "state/analysis/evidence.json",
            "evidence_sha256": "a" * 64,
        }]
    )
    valid.to_csv(path, index=False)
    loaded = _load_primary_service_response_policies(
        RepoPaths.from_repo_root(tmp_path)
    )
    assert loaded["service_scope_id"].tolist() == ["navajo_nation_police"]

    pd.concat([valid, valid], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        _load_primary_service_response_policies(RepoPaths.from_repo_root(tmp_path))

    valid.assign(primary_response_policy="unknown").to_csv(path, index=False)
    with pytest.raises(ValueError, match="require a service scope"):
        _load_primary_service_response_policies(RepoPaths.from_repo_root(tmp_path))


def test_primary_service_policy_must_target_a_displacing_owner() -> None:
    footprint = pd.DataFrame(
        {
            "ori9": ["AZ0018900"],
            "state_fips": ["04"],
            "bg_id": ["040019440002"],
            "service_scope_id": ["navajo_nation_police"],
            "bg_population_coverage_share": [0.9],
            "bg_service_population_coverage_share": [0.4],
        }
    )
    with pytest.raises(ValueError, match="displaces county remainder"):
        _build_allocation_exclusive_footprint_displacement(
            overrides=pd.DataFrame(
                {"ori9": ["AZ0018900"], "displaces_county_remainder": [False]}
            ),
            custom_footprints=footprint,
            concurrent_jurisdiction_carveouts=pd.DataFrame(
                {"county_geoid": ["04001"]}
            ),
            primary_response_policies=pd.DataFrame(
                {
                    "service_scope_id": ["navajo_nation_police"],
                    "primary_response_policy": [
                        "displace_county_remainder_within_reviewed_carveout"
                    ],
                }
            ),
        )


def test_primary_service_precedence_accepts_explicit_zero_own_fraction() -> None:
    displacement = _build_allocation_exclusive_footprint_displacement(
        overrides=pd.DataFrame(
            {"ori9": ["AZ0018900"], "displaces_county_remainder": [True]}
        ),
        custom_footprints=pd.DataFrame(
            {
                "ori9": ["AZ0018900"],
                "state_fips": ["35"],
                "bg_id": ["350319440002"],
                "service_scope_id": ["navajo_nation_police"],
                "bg_population_coverage_share": [0.0],
                "bg_service_population_coverage_share": [0.0],
            }
        ),
        concurrent_jurisdiction_carveouts=pd.DataFrame(
            {"county_geoid": ["35031"]}
        ),
        primary_response_policies=pd.DataFrame(
            {
                "service_scope_id": ["navajo_nation_police"],
                "primary_response_policy": [
                    "displace_county_remainder_within_reviewed_carveout"
                ],
            }
        ),
    )
    assert displacement.empty


def test_final_component_support_applies_primary_service_precedence(
    monkeypatch, tmp_path
) -> None:
    crosswalk = pd.DataFrame(
        {
            "state_fips": ["04", "04", "04"],
            "block_group_geoid": [
                "040019440002", "040019440003", "040019440002"
            ],
            "jurisdiction_type": [
                STATE_REMAINDER_TYPE, STATE_REMAINDER_TYPE, "municipal"
            ],
            "jurisdiction_id": ["04:remainder", "04:remainder", "04:municipal"],
            "allocation_share": [0.6, 1.0, 0.4],
        }
    )
    monkeypatch.setattr(
        allocation, "_load_overlap_footprint_overrides",
        lambda _paths: pd.DataFrame(
            {"ori9": ["AZ0018900"], "displaces_county_remainder": [True]}
        ),
    )
    monkeypatch.setattr(
        allocation, "_load_overlap_custom_footprints",
        lambda _paths: pd.DataFrame(
            {
                "ori9": ["AZ0018900"],
                "state_fips": ["04"],
                "bg_id": ["040019440002"],
                "service_scope_id": ["navajo_nation_police"],
                "bg_population_coverage_share": [0.9],
                "bg_service_population_coverage_share": [0.5],
            }
        ),
    )
    monkeypatch.setattr(
        allocation, "_load_concurrent_jurisdiction_carveouts",
        lambda _paths: pd.DataFrame({"county_geoid": ["04001"]}),
    )
    monkeypatch.setattr(
        allocation, "_load_primary_service_response_policies",
        lambda _paths: pd.DataFrame(
            {
                "service_scope_id": ["navajo_nation_police"],
                "primary_response_policy": [
                    "displace_county_remainder_within_reviewed_carveout"
                ],
            }
        ),
    )

    out = _apply_allocation_primary_service_displacement(
        paths=RepoPaths.from_repo_root(tmp_path), bg_crosswalk=crosswalk
    )
    remainder = out[out["jurisdiction_type"].eq(STATE_REMAINDER_TYPE)].copy()
    assert remainder["allocation_share"].tolist() == pytest.approx([0.3, 1.0])
    municipal = out[out["jurisdiction_type"].eq("municipal")]
    assert municipal["allocation_share"].tolist() == pytest.approx([0.4])

    model_input = remainder.rename(columns={"block_group_geoid": "bg_id"}).assign(
        offense="aggravated_assault", bg_weight=1.0
    )
    model_output, mixture_summary = _apply_model_lane_shares(model_input, mixture=None)
    assert mixture_summary is None
    target = 13.0
    allocated = target * model_output["model_share"]
    assert allocated.tolist() == pytest.approx([3.0, 10.0])
    assert allocated.sum() == pytest.approx(target)
    without_policy = target * pd.Series([0.6, 1.0]) / 1.6
    assert allocated.iloc[1] > without_policy.iloc[1]


def test_final_component_builder_invokes_allocation_primary_service_policy(
    monkeypatch, tmp_path
) -> None:
    class PolicyReached(RuntimeError):
        pass

    monkeypatch.setattr(
        allocation, "_load_consolidated_agency_footprints", lambda _paths: pd.DataFrame()
    )

    def reached_policy(*, paths, bg_crosswalk):
        raise PolicyReached

    monkeypatch.setattr(
        allocation, "_apply_allocation_primary_service_displacement", reached_policy
    )
    with pytest.raises(PolicyReached):
        _build_jurisdiction_component_allocations(
            paths=RepoPaths.from_repo_root(tmp_path),
            bg_prior=pd.DataFrame(),
            bg_crosswalk=pd.DataFrame(),
            controls=pd.DataFrame(),
            year=2025,
        )


def test_overlapping_primary_services_inside_carveout_require_union_evidence() -> None:
    overrides = pd.DataFrame(
        {"ori9": ["AZ0018900", "AZ9999999"], "displaces_county_remainder": [True, True]}
    )
    footprint = pd.DataFrame(
        {
            "ori9": ["AZ0018900", "AZ9999999"],
            "state_fips": ["04", "04"],
            "bg_id": ["040019440002", "040019440002"],
            "service_scope_id": ["navajo_nation_police", "other_primary_service"],
            "bg_population_coverage_share": [0.9, 0.9],
            "bg_service_population_coverage_share": [0.4, 0.2],
        }
    )
    with pytest.raises(ValueError, match="explicit union coverage fraction"):
        _build_allocation_exclusive_footprint_displacement(
            overrides=overrides,
            custom_footprints=footprint,
            concurrent_jurisdiction_carveouts=pd.DataFrame({"county_geoid": ["04001"]}),
            primary_response_policies=pd.DataFrame(
                {
                    "service_scope_id": [
                        "navajo_nation_police", "other_primary_service"
                    ],
                    "primary_response_policy": [
                        "displace_county_remainder_within_reviewed_carveout",
                        "displace_county_remainder_within_reviewed_carveout",
                    ],
                }
            ),
        )


def test_source_state_calibration_scales_imports_by_owner_not_destination() -> None:
    components = pd.DataFrame(
        [
            {
                "state_fips": "35",
                "source_state_fips": "04",
                "bg_id": "350010001001",
                "tract_id": "35001000100",
                "offense": "robbery",
                "jurisdiction_type": "custom_footprint_overlap_layer",
                "component_count": 2.0,
            },
            {
                "state_fips": "35",
                "source_state_fips": "35",
                "bg_id": "350010001001",
                "tract_id": "35001000100",
                "offense": "robbery",
                "jurisdiction_type": "statewide_overlap_layer",
                "component_count": 3.0,
            },
        ]
    )

    out = _scale_components_by_source_state(
        components,
        {("04", "robbery"): 2.0, ("35", "robbery"): 3.0},
    )

    assert out["component_count"].tolist() == pytest.approx([4.0, 9.0])
    assert out["component_count"].sum() == pytest.approx(13.0)


def test_transfer_ledger_reconciles_final_component_mass() -> None:
    components = pd.DataFrame(
        [
            {
                "state_fips": destination,
                "source_state_fips": "04",
                "service_scope_id": "navajo_nation_pd",
                "canonical_target_ori": "AZ0018900",
                "offense": "robbery",
                "source_target_count": 10.0,
                "component_count": count,
            }
            for destination, count in (("04", 4.0), ("35", 3.0), ("49", 3.0))
        ]
    )

    out = _build_service_state_transfer_ledger(components, year=2025)

    assert out["allocation_state_fips"].tolist() == ["04", "35", "49"]
    assert out["allocation_count"].tolist() == pytest.approx([4.0, 3.0, 3.0])
    assert out["allocation_share"].tolist() == pytest.approx([0.4, 0.3, 0.3])
    assert set(out["year"]) == {2025}


def test_service_water_mass_only_routes_to_same_service_scope() -> None:
    common = {
        "state_fips": "35",
        "source_state_fips": "04",
        "tract_id": "35001000100",
        "jurisdiction_id": "AZ0018900",
        "jurisdiction_type": "custom_footprint_overlap_layer",
        "offense": "robbery",
        "canonical_target_ori": "AZ0018900",
        "source_target_count": 4.0,
    }
    components = pd.DataFrame(
        [
            {
                **common,
                "bg_id": "350010001001",
                "service_scope_id": "navajo_nation_pd",
                "component_count": 1.0,
                "primary_denominator_raw": 0.0,
                "land_area_sq_mi": 0.0,
            },
            {
                **common,
                "bg_id": "350010001002",
                "service_scope_id": "navajo_nation_pd",
                "component_count": 3.0,
                "primary_denominator_raw": 100.0,
                "land_area_sq_mi": 1.0,
            },
            {
                **common,
                "bg_id": "350010001003",
                "service_scope_id": "ramah_navajo_pd",
                "canonical_target_ori": "NM0170400",
                "jurisdiction_id": "NM0170400",
                "source_state_fips": "35",
                "source_target_count": 5.0,
                "component_count": 5.0,
                "primary_denominator_raw": 100.0,
                "land_area_sq_mi": 1.0,
            },
            {
                **common,
                "bg_id": "350010001004",
                "service_scope_id": pd.NA,
                "canonical_target_ori": "OUTSIDE",
                "jurisdiction_id": "OUTSIDE",
                "source_state_fips": "35",
                "source_target_count": 7.0,
                "component_count": 7.0,
                "primary_denominator_raw": 100.0,
                "land_area_sq_mi": 1.0,
            },
        ]
    )

    out, _, _ = _redistribute_zero_target_components(components, pd.DataFrame())
    counts = out.set_index("bg_id")["component_count"]

    assert counts["350010001001"] == 0.0
    assert counts["350010001002"] == pytest.approx(4.0)
    assert counts["350010001003"] == pytest.approx(5.0)
    assert counts["350010001004"] == pytest.approx(7.0)


def test_service_water_mass_backs_off_to_another_county_inside_same_scope() -> None:
    components = pd.DataFrame(
        [
            {
                "state_fips": "35",
                "source_state_fips": "04",
                "bg_id": "350010001001",
                "tract_id": "35001000100",
                "jurisdiction_id": "AZ0018900",
                "jurisdiction_type": "custom_footprint_overlap_layer",
                "offense": "robbery",
                "service_scope_id": "navajo_nation_police",
                "canonical_target_ori": "AZ0018900",
                "source_target_count": 4.0,
                "component_count": 1.0,
                "primary_denominator_raw": 0.0,
                "land_area_sq_mi": 0.0,
            },
            {
                "state_fips": "35",
                "source_state_fips": "04",
                "bg_id": "350030001001",
                "tract_id": "35003000100",
                "jurisdiction_id": "AZ0018900",
                "jurisdiction_type": "custom_footprint_overlap_layer",
                "offense": "robbery",
                "service_scope_id": "navajo_nation_police",
                "canonical_target_ori": "AZ0018900",
                "source_target_count": 4.0,
                "component_count": 3.0,
                "primary_denominator_raw": 100.0,
                "land_area_sq_mi": 1.0,
            },
        ]
    )

    out, _, _ = _redistribute_zero_target_components(components, pd.DataFrame())
    counts = out.set_index("bg_id")["component_count"]

    assert counts["350010001001"] == 0.0
    assert counts["350030001001"] == pytest.approx(4.0)


def test_entire_ineligible_service_routes_to_explicit_source_owned_unlocated() -> None:
    components = pd.DataFrame(
        [
            {
                "state_fips": state,
                "source_state_fips": "04",
                "bg_id": bg,
                "tract_id": bg[:11],
                "jurisdiction_id": "AZ0018900",
                "jurisdiction_type": "custom_footprint_overlap_layer",
                "offense": "robbery",
                "service_scope_id": "navajo_nation_police",
                "canonical_target_ori": "AZ0018900",
                "source_target_count": 4.0,
                "component_count": count,
                "primary_denominator_raw": 0.0,
                "land_area_sq_mi": 0.0,
            }
            for state, bg, count in (
                ("04", "040010001001", 1.0),
                ("35", "350030001001", 3.0),
            )
        ]
    )

    out, _, _ = _redistribute_zero_target_components(components, pd.DataFrame())
    unlocated = out.attrs["service_unlocated_mass"]
    ledger = _build_service_state_transfer_ledger(
        out, year=2025, service_unlocated=unlocated
    )

    assert out["component_count"].sum() == 0.0
    assert unlocated.to_dict(orient="records") == [
        {
            "source_state_fips": "04",
            "service_scope_id": "navajo_nation_police",
            "canonical_target_ori": "AZ0018900",
            "offense": "robbery",
            "source_target_count": 4.0,
            "unlocated_count": 4.0,
            "route_reason": "service_no_eligible_receiver",
        }
    ]
    assert len(ledger) == 1
    assert pd.isna(ledger.iloc[0]["allocation_state_fips"])
    assert ledger.iloc[0]["allocation_count"] == pytest.approx(4.0)
    assert ledger.iloc[0]["allocation_share"] == pytest.approx(1.0)


def test_source_control_conservation_accounts_for_service_exports_and_imports() -> None:
    surface = pd.DataFrame(
        [
            {"state_fips": "04", "expected_count_robbery": 4.0},
            {"state_fips": "35", "expected_count_robbery": 6.0},
        ]
    )
    for offense in allocation.OFFENSES_7:
        column = f"expected_count_{offense}"
        if column not in surface:
            surface[column] = 0.0
    controls = pd.DataFrame(
        [
            {"state_fips": "04", "offense": "robbery", "adjusted_count_ags_core": 10.0},
            {"state_fips": "35", "offense": "robbery", "adjusted_count_ags_core": 0.0},
        ]
    )
    unlocated = pd.DataFrame(
        columns=["state_fips", "offense", "unlocated_count"]
    )
    transfer = pd.DataFrame(
        [
            {
                "source_state_fips": "04",
                "allocation_state_fips": "04",
                "offense": "robbery",
                "allocation_count": 4.0,
            },
            {
                "source_state_fips": "04",
                "allocation_state_fips": "35",
                "offense": "robbery",
                "allocation_count": 6.0,
            },
        ]
    )

    result = _assert_unlocated_mass_conservation(
        surface=surface,
        controls=controls,
        unlocated=unlocated,
        service_state_transfer=transfer,
        year=2025,
    )

    assert result["max_abs_delta"] == 0.0


def test_envelope_conserves_service_mass_across_destination_states_without_leakage() -> None:
    def row(bg: str, count: float, denominator: float) -> dict[str, object]:
        return {
            "state_fips": bg[:2],
            "source_state_fips": "04",
            "bg_id": bg,
            "tract_id": bg[:11],
            "jurisdiction_id": "AZ0018900",
            "jurisdiction_type": "custom_footprint_overlap_layer",
            "offense": "robbery",
            "component_count": count,
            "component_share": count / 100.0,
            "city_incident_posterior_active": False,
            "primary_denominator_raw": denominator,
            "households_total": 100.0,
            "service_scope_id": "navajo_nation_police",
            "canonical_target_ori": "AZ0018900",
        }

    service = [
        row("040010001001", 90.0, 100.0),
        row("350010001001", 5.0, 1000.0),
        row("350010002001", 5.0, 1000.0),
    ]
    outside = {
        **row("350010003001", 7.0, 1000.0),
        "source_state_fips": "35",
        "jurisdiction_id": "35:statewide_overlap_layer",
        "jurisdiction_type": "statewide_overlap_layer",
        "service_scope_id": pd.NA,
        "canonical_target_ori": pd.NA,
    }

    out = _apply_model_only_allocation_envelopes(pd.DataFrame([*service, outside]))
    service_out = out[out["service_scope_id"].eq("navajo_nation_police").fillna(False)]
    outside_out = out[out["jurisdiction_id"].eq("35:statewide_overlap_layer")]

    assert service_out["component_count"].sum() == pytest.approx(100.0, abs=1e-12)
    assert outside_out["component_count"].iloc[0] == pytest.approx(7.0, abs=1e-12)


def test_service_wide_target_joins_once_and_allocates_over_all_states(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    targets = pd.DataFrame(
        [
            {
                "state_fips": "04",
                "offense": "robbery",
                "group_kind": "custom_footprint_overlap",
                "group_id": "AZ0018900",
                "target_count": 10.0,
            }
        ]
    )
    footprints = pd.DataFrame(
        [
            {
                "state_fips": state,
                "bg_id": bg,
                "ori9": "AZ0018900",
                "weight_share": share,
                "weight_share_basis": "service_area_prior",
                "bg_land_area_coverage_share": share,
                "allocation_scope": "service_wide",
                "service_scope_id": "navajo_nation_police",
                "canonical_target_ori": "AZ0018900",
                "source_state_fips": "04",
            }
            for state, bg, share in (
                ("04", "040010001001", 0.4),
                ("35", "350010001001", 0.3),
                ("49", "490010001001", 0.3),
            )
        ]
    )
    monkeypatch.setattr(allocation, "_build_overlap_group_targets", lambda **_: targets)
    monkeypatch.setattr(allocation, "_load_overlap_custom_footprints", lambda _: footprints)
    monkeypatch.setattr(allocation, "_load_overlap_footprint_overrides", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        allocation, "_load_concurrent_jurisdiction_carveouts", lambda _: pd.DataFrame()
    )
    prior = pd.DataFrame(
        [
            {
                "state_fips": state,
                "bg_id": bg,
                "tract_id": bg[:11],
                "offense": "robbery",
                "bg_weight": 1.0,
            }
            for state, bg in (
                ("04", "040010001001"),
                ("35", "350010001001"),
                ("49", "490010001001"),
            )
        ]
    )

    out = _build_overlap_allocations(
        paths=RepoPaths.from_repo_root(tmp_path),
        bg_prior=prior,
        bg_crosswalk=pd.DataFrame(),
        controls=pd.DataFrame(),
        year=2025,
    )

    assert out.groupby("state_fips")["component_count"].sum().to_dict() == pytest.approx(
        {"04": 4.0, "35": 3.0, "49": 3.0}
    )
    assert out["component_count"].sum() == pytest.approx(10.0)
    assert set(out["source_state_fips"]) == {"04"}


def test_ordinary_resident_target_conserves_source_state_without_population_squaring(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    targets = pd.DataFrame(
        [
            {
                "state_fips": "06",
                "offense": "robbery",
                "group_kind": "custom_footprint_overlap",
                "group_id": "CA0000000",
                "target_count": 10.0,
            }
        ]
    )
    footprints = pd.DataFrame(
        [
            {
                "state_fips": "06",
                "bg_id": bg,
                "ori9": "CA0000000",
                "weight_share": population_share,
                "weight_share_basis": "resident_population",
                "bg_responsibility_population_coverage_share": 1.0,
                "responsibility_fraction_basis": "test_full_bg_population",
                "allocation_scope": "source_state",
                "service_scope_id": pd.NA,
                "canonical_target_ori": "CA0000000",
                "source_state_fips": "06",
            }
            for bg, population_share in (
                ("060010001001", 0.1),
                ("060010001002", 0.9),
            )
        ]
    )
    monkeypatch.setattr(allocation, "_build_overlap_group_targets", lambda **_: targets)
    monkeypatch.setattr(allocation, "_load_overlap_custom_footprints", lambda _: footprints)
    monkeypatch.setattr(allocation, "_load_overlap_footprint_overrides", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        allocation, "_load_concurrent_jurisdiction_carveouts", lambda _: pd.DataFrame()
    )
    prior = pd.DataFrame(
        [
            {
                "state_fips": "06",
                "bg_id": bg,
                "tract_id": bg[:11],
                "offense": "robbery",
                "bg_weight": prior_count,
            }
            for bg, prior_count in (
                ("060010001001", 10.0),
                ("060010001002", 90.0),
            )
        ]
    )

    out = _build_overlap_allocations(
        paths=RepoPaths.from_repo_root(tmp_path),
        bg_prior=prior,
        bg_crosswalk=pd.DataFrame(),
        controls=pd.DataFrame(),
        year=2025,
    )

    assert out.sort_values("bg_id")["component_count"].tolist() == pytest.approx([1.0, 9.0])
    assert out["component_count"].sum() == pytest.approx(10.0)
    assert set(out["source_state_fips"]) == {"06"}
    assert out["bg_responsibility_population_coverage_share"].tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert set(out["responsibility_fraction_basis"]) == {"test_full_bg_population"}
