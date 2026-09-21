"""Per-ORI conservation on the custom-footprint overlap layer.

The property under test is the municipal lane's property, applied to footprints: the mass a
footprint places on block groups is the mass its OWN admitted control carries in
`level_lane_mass_ledger`, never a slice of the state overlap pool. The four cases the fixture
covers are the four the real surface contains -- an exclusively owned footprint, a footprint two
reporter keys both claim, a footprint member with no residents, and an ORI with ledger mass whose
footprint the block-group prior does not carry.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    CUSTOM_FOOTPRINT_STATUS_NO_PLACEABLE_SUPPORT,
    CUSTOM_FOOTPRINT_STATUS_PLACED,
    CUSTOM_FOOTPRINT_STATUS_SUPPRESSED_DUPLICATE,
    CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT,
    OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT,
    OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE,
    UNLOCATED_GROUP_KIND,
    UNLOCATED_ROUTES,
    _apply_footprint_ledger_rake,
    _build_footprint_mass_conservation,
    _canonical_footprint_owners,
    _custom_footprint_component_shares,
    _footprint_ledger_partition,
    _footprint_unlocated_routes,
)
from crimerisk.allocation import (
    FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR,
    FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR,
    footprint_mass_conservation_path,
)

STATE = "99"
OFFENSE = "larceny"

# Two block groups for the exclusive footprint, one of which has no residents at all; one block
# group two reporter keys share; one block group the prior does not carry.
BG_A1 = "990010001001"
BG_A2 = "990010001002"
BG_SHARED = "990010002001"
BG_ABSENT = "990010009001"


def _footprint_row(ori: str, bg: str, weight: float, responsibility: float = 1.0) -> dict:
    return {
        "ori9": ori,
        "state_fips": STATE,
        "bg_id": bg,
        "weight_share": weight,
        "bg_population_coverage_share": responsibility,
        "weight_share_basis": CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT,
        "geometry_source_type": "test",
        "geometry_source_ref": "test",
        "footprint_note": "test",
        "geometry_contributor_ori": ori,
        "service_scope_id": pd.NA,
        "canonical_target_ori": ori,
        "source_state_fips": STATE,
        "allocation_scope": "source_state",
        "bg_service_population_coverage_share": pd.NA,
        "bg_land_area_coverage_share": pd.NA,
        "coverage_basis": pd.NA,
        "bg_responsibility_population_coverage_share": responsibility,
        "responsibility_fraction_basis": "test_block_population",
    }


def _custom_footprints() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # EXCLUSIVE footprint: two block groups, the second with no residents.
            _footprint_row("AAA000001", BG_A1, 0.75),
            _footprint_row("AAA000001", BG_A2, 0.25),
            # SHARED footprint: one block group, byte-identical support under two reporter keys.
            _footprint_row("BBB000001", BG_SHARED, 1.0),
            _footprint_row("BBB000002", BG_SHARED, 1.0),
            # LEDGER WITHOUT PLACEABLE FOOTPRINT: the prior does not carry this block group.
            _footprint_row("CCC000001", BG_ABSENT, 1.0),
        ]
    )


def _ledger() -> pd.DataFrame:
    """Own admitted mass per ORI. The state overlap layer also holds an agency with no
    footprint at all, so the footprint ORIs are a strict subset of the pool."""
    return pd.DataFrame(
        [
            {"ori9": "AAA000001", "state_fips": STATE, "offense": OFFENSE, "ledger_control_mass": 100.0},
            {"ori9": "BBB000001", "state_fips": STATE, "offense": OFFENSE, "ledger_control_mass": 60.0},
            {"ori9": "BBB000002", "state_fips": STATE, "offense": OFFENSE, "ledger_control_mass": 40.0},
            {"ori9": "CCC000001", "state_fips": STATE, "offense": OFFENSE, "ledger_control_mass": 50.0},
            {"ori9": "ZZZ000001", "state_fips": STATE, "offense": OFFENSE, "ledger_control_mass": 250.0},
        ]
    )


def _bg_prior() -> pd.DataFrame:
    """The model prior. BG_A2 has no residents and therefore no model weight; BG_ABSENT is not
    in the prior at all, which is what makes CCC000001 unplaceable."""
    return pd.DataFrame(
        [
            {"state_fips": STATE, "bg_id": BG_A1, "tract_id": BG_A1[:11], "offense": OFFENSE, "bg_weight": 10.0},
            {"state_fips": STATE, "bg_id": BG_A2, "tract_id": BG_A2[:11], "offense": OFFENSE, "bg_weight": 0.0},
            {"state_fips": STATE, "bg_id": BG_SHARED, "tract_id": BG_SHARED[:11], "offense": OFFENSE, "bg_weight": 4.0},
        ]
    )


def _overlap_controls(state_target: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"state_fips": STATE, "offense": OFFENSE, "state_target": float(state_target)}]
    )


def _partition(state_target: float = 500.0) -> pd.DataFrame:
    return _footprint_ledger_partition(
        custom_footprints=_custom_footprints(),
        ledger=_ledger(),
        overlap_controls=_overlap_controls(state_target),
        bg_prior=_bg_prior(),
    )


def _status(partition: pd.DataFrame, ori: str) -> str:
    return str(partition.loc[partition["ori9"].eq(ori), "footprint_status"].iloc[0])


def _expected(partition: pd.DataFrame, ori: str) -> float:
    return float(partition.loc[partition["ori9"].eq(ori), "expected_placed_mass"].iloc[0])


# --- the canonical rule for a shared footprint ---------------------------------------------


def test_identical_footprints_resolve_to_the_reporter_that_carries_the_mass():
    owners = _canonical_footprint_owners(_custom_footprints(), _ledger())
    owners = owners.set_index("ori9")
    assert owners.loc["BBB000001", "canonical_target_ori"] == "BBB000001"
    assert owners.loc["BBB000002", "canonical_target_ori"] == "BBB000001"
    assert int(owners.loc["BBB000001", "shared_footprint_ori_count"]) == 2
    # A footprint nobody else claims is its own canonical owner.
    assert owners.loc["AAA000001", "canonical_target_ori"] == "AAA000001"
    assert int(owners.loc["AAA000001", "shared_footprint_ori_count"]) == 1


def test_the_canonical_choice_follows_the_ledger_and_not_the_ori_ordering():
    ledger = _ledger()
    # Flip which reporter key filed: the canonical owner must follow the mass, not the sort.
    ledger.loc[ledger["ori9"].eq("BBB000001"), "ledger_control_mass"] = 1.0
    owners = _canonical_footprint_owners(_custom_footprints(), ledger).set_index("ori9")
    assert owners.loc["BBB000001", "canonical_target_ori"] == "BBB000002"
    assert owners.loc["BBB000002", "canonical_target_ori"] == "BBB000002"


# --- the partition -------------------------------------------------------------------------


def test_every_footprint_ori_gets_one_of_three_statuses():
    partition = _partition()
    assert _status(partition, "AAA000001") == CUSTOM_FOOTPRINT_STATUS_PLACED
    assert _status(partition, "BBB000001") == CUSTOM_FOOTPRINT_STATUS_PLACED
    assert _status(partition, "BBB000002") == CUSTOM_FOOTPRINT_STATUS_SUPPRESSED_DUPLICATE
    assert _status(partition, "CCC000001") == CUSTOM_FOOTPRINT_STATUS_NO_PLACEABLE_SUPPORT
    # The pool member with no footprint never enters the table.
    assert "ZZZ000001" not in set(partition["ori9"])


def test_on_the_accounting_surface_the_expected_mass_is_the_ledger_control_exactly():
    partition = _partition(state_target=500.0).set_index("ori9")
    assert partition.loc["AAA000001", "control_surface_factor"] == pytest.approx(1.0)
    assert partition.loc["AAA000001", "surface_control_mass"] == pytest.approx(100.0)
    assert partition.loc["AAA000001", "expected_placed_mass"] == pytest.approx(100.0)
    assert partition.loc["BBB000001", "expected_placed_mass"] == pytest.approx(60.0)
    # The two statuses that do not place owe their mass to the unlocated table, so nothing is
    # expected on block groups; the mass itself is still carried on the row.
    assert partition.loc["BBB000002", "surface_control_mass"] == pytest.approx(40.0)
    assert partition.loc["BBB000002", "expected_placed_mass"] == pytest.approx(0.0)
    assert partition.loc["CCC000001", "surface_control_mass"] == pytest.approx(50.0)
    assert partition.loc["CCC000001", "expected_placed_mass"] == pytest.approx(0.0)


def test_a_smoothed_control_carries_the_ledger_share_not_the_raw_ledger_amount():
    """The released map divides a temporally smoothed state total. The ORI's own SHARE of the
    accounting pool is what transfers; the factor is the same for every ORI in the state."""
    partition = _partition(state_target=250.0).set_index("ori9")
    assert partition.loc["AAA000001", "control_surface_factor"] == pytest.approx(0.5)
    assert partition.loc["AAA000001", "ledger_share_of_state"] == pytest.approx(0.2)
    assert partition.loc["AAA000001", "expected_placed_mass"] == pytest.approx(50.0)


# --- the rake ------------------------------------------------------------------------------


def _grouped(state_target: float = 500.0) -> pd.DataFrame:
    """The pre-rake partition the cascade produces: one footprint group holding a wildly wrong
    slice of the state pool, one ordinary statewide group, and the seeded residual.

    The real frame also carries the rake bookkeeping the cascade built it with -- including a
    `state_target` column whose name collides with the control table the rake merges in.
    """
    return pd.DataFrame(
        [
            {"state_fips": STATE, "offense": OFFENSE, "group_kind": "custom_footprint_overlap", "group_id": "AAA000001", "target_count": 420.0, "state_target": state_target, "reported_count": 420.0},
            {"state_fips": STATE, "offense": OFFENSE, "group_kind": "county_overlap", "group_id": "99001", "target_count": 80.0, "state_target": state_target, "reported_count": 80.0},
            {"state_fips": STATE, "offense": OFFENSE, "group_kind": UNLOCATED_GROUP_KIND, "group_id": STATE, "target_count": 0.0, "state_target": state_target, "reported_count": 0.0},
        ]
    )


def _raked(state_target: float = 500.0) -> pd.DataFrame:
    return _apply_footprint_ledger_rake(
        _grouped(state_target),
        partition=_partition(state_target),
        overlap_controls=_overlap_controls(state_target),
        residual_kind=UNLOCATED_GROUP_KIND,
    ).set_index(["group_kind", "group_id"])


def test_the_rake_gives_each_placed_footprint_its_own_ledger_mass():
    raked = _raked()
    assert raked.loc[("custom_footprint_overlap", "AAA000001"), "target_count"] == pytest.approx(100.0)
    assert raked.loc[("custom_footprint_overlap", "BBB000001"), "target_count"] == pytest.approx(60.0)
    assert ("custom_footprint_overlap", "BBB000002") not in raked.index
    assert ("custom_footprint_overlap", "CCC000001") not in raked.index


def test_mass_that_cannot_be_placed_goes_to_the_unlocated_lane_and_not_to_silence():
    raked = _raked()
    # 40 from the suppressed duplicate plus 50 from the footprint with no placeable support.
    assert raked.loc[(UNLOCATED_GROUP_KIND, STATE), "target_count"] == pytest.approx(90.0)


def test_the_rake_still_partitions_the_state_control_exactly():
    for state_target in (500.0, 250.0, 1000.0):
        raked = _raked(state_target)
        assert float(raked["target_count"].sum()) == pytest.approx(state_target)


def test_every_other_group_keeps_its_relative_answer():
    """The rake takes mass off the footprints, not off any particular other lane: what is left
    is shared in the proportions the cascade already produced."""
    grouped = _grouped()
    grouped = pd.concat(
        [
            grouped,
            pd.DataFrame(
                [
                    {"state_fips": STATE, "offense": OFFENSE, "group_kind": "county_overlap", "group_id": "99003", "target_count": 0.0, "state_target": 500.0, "reported_count": 0.0},
                    {"state_fips": STATE, "offense": OFFENSE, "group_kind": "municipal_place_overlap", "group_id": "99:m", "target_count": 20.0, "state_target": 500.0, "reported_count": 20.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    raked = _apply_footprint_ledger_rake(
        grouped,
        partition=_partition(),
        overlap_controls=_overlap_controls(500.0),
        residual_kind=UNLOCATED_GROUP_KIND,
    ).set_index(["group_kind", "group_id"])
    county = float(raked.loc[("county_overlap", "99001"), "target_count"])
    place = float(raked.loc[("municipal_place_overlap", "99:m"), "target_count"])
    assert county / place == pytest.approx(80.0 / 20.0)


def test_the_unplaceable_mass_is_reported_under_its_own_routes():
    routes = _footprint_unlocated_routes(_partition()).set_index("overlap_resolution_route")
    assert routes.loc[OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE, "route_target_count"] == pytest.approx(40.0)
    assert routes.loc[OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT, "route_target_count"] == pytest.approx(50.0)
    # Both routes are declared unlocated, so the companion table can account for them.
    assert OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE in UNLOCATED_ROUTES
    assert OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT in UNLOCATED_ROUTES


# --- placement: the shares inside a footprint still sum to one ------------------------------


def test_a_zero_population_member_does_not_leak_mass_out_of_the_footprint():
    """BG_A2 carries no model weight. Its share is zero and BG_A1 takes the whole target, so the
    ORI still places exactly what it was raked to -- the member is empty, not the footprint."""
    footprints = _custom_footprints()
    merged = (
        _bg_prior()
        .merge(footprints[footprints["ori9"].eq("AAA000001")], on=["state_fips", "bg_id"], how="inner")
        .assign(target_count=100.0)
    )
    shares = _custom_footprint_component_shares(merged)
    assert float(shares["component_share"].sum()) == pytest.approx(1.0)
    placed = float((shares["component_share"] * shares["target_count"]).sum())
    assert placed == pytest.approx(100.0)
    assert float(shares.loc[shares["bg_id"].eq(BG_A2), "component_share"].iloc[0]) == pytest.approx(0.0)


def test_a_footprint_with_no_model_signal_anywhere_falls_back_to_its_declared_weights():
    footprints = _custom_footprints()
    prior = _bg_prior()
    prior["bg_weight"] = 0.0
    merged = (
        prior.merge(footprints[footprints["ori9"].eq("AAA000001")], on=["state_fips", "bg_id"], how="inner")
        .assign(target_count=100.0)
    )
    shares = _custom_footprint_component_shares(merged)
    assert float(shares["component_share"].sum()) == pytest.approx(1.0)
    assert float(shares.loc[shares["bg_id"].eq(BG_A1), "component_share"].iloc[0]) == pytest.approx(0.75)


# --- the audit table the release gate reads -------------------------------------------------


def _components(placed: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": STATE,
                "bg_id": BG_A1,
                "tract_id": BG_A1[:11],
                "jurisdiction_id": ori,
                "jurisdiction_type": "custom_footprint_overlap_layer",
                "offense": OFFENSE,
                "component_count": mass,
            }
            for ori, mass in placed.items()
        ]
    )


def test_the_conservation_table_reads_zero_error_on_a_conserved_surface():
    audit = _build_footprint_mass_conservation(
        _components({"AAA000001": 100.0, "BBB000001": 60.0}), _partition()
    ).set_index("ori9")
    assert float(audit.loc["AAA000001", "relative_error"]) == pytest.approx(0.0)
    assert float(audit.loc["BBB000001", "relative_error"]) == pytest.approx(0.0)
    assert float(audit.loc["BBB000002", "placed_mass"]) == pytest.approx(0.0)
    assert float(audit.loc["CCC000001", "placed_mass"]) == pytest.approx(0.0)
    assert float(audit["relative_error"].max()) == pytest.approx(0.0)


def test_the_conservation_table_catches_a_footprint_that_places_the_wrong_mass():
    audit = _build_footprint_mass_conservation(
        _components({"AAA000001": 474.0, "BBB000001": 60.0}), _partition()
    ).set_index("ori9")
    assert float(audit.loc["AAA000001", "relative_error"]) == pytest.approx(3.74)
    assert float(audit.loc["AAA000001", "placed_minus_expected"]) == pytest.approx(374.0)


def test_the_conservation_table_catches_a_suppressed_duplicate_that_placed_anything():
    audit = _build_footprint_mass_conservation(
        _components({"AAA000001": 100.0, "BBB000001": 60.0, "BBB000002": 40.0}), _partition()
    ).set_index("ori9")
    assert not np.isfinite(float(audit.loc["BBB000002", "relative_error"]))
    assert float(audit.loc["BBB000002", "placed_mass"]) == pytest.approx(40.0)


def test_the_residual_lane_is_seeded_where_the_partition_left_none():
    """Under the smoothed partition a state keeps only the groups its own agencies produced, so
    a state whose overlap agencies all resolved cleanly has no residual row. The suppressed
    duplicate on one of its reservations still has to land somewhere."""
    grouped = _grouped()
    grouped = grouped[~grouped["group_kind"].eq(UNLOCATED_GROUP_KIND)].copy()
    assert UNLOCATED_GROUP_KIND not in set(grouped["group_kind"])
    raked = _apply_footprint_ledger_rake(
        grouped,
        partition=_partition(),
        overlap_controls=_overlap_controls(500.0),
        residual_kind=UNLOCATED_GROUP_KIND,
    ).set_index(["group_kind", "group_id"])
    assert raked.loc[(UNLOCATED_GROUP_KIND, STATE), "target_count"] == pytest.approx(90.0)
    assert float(raked["target_count"].sum()) == pytest.approx(500.0)


# --- the release gate's two bands -------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"

OFFENSES = (
    "homicide",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
)


def _validator():
    spec = importlib.util.spec_from_file_location("validate_release_outputs_for_gate", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = _validator()


def test_the_per_offence_floor_is_two_whole_offences():
    """The one-offence floor caught VTVSP1600 robbery; the rate-ratio cap, not a bad rake."""
    assert FOOTPRINT_MASS_CONSERVATION_MIN_ABSOLUTE_ERROR == pytest.approx(2.0)


def _conservation_frame(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """(ori9, offense, ledger/expected mass, placed mass) -> the audited conservation table."""
    records = []
    for ori, offense, expected, placed in rows:
        relative = abs(placed - expected) / expected if expected > 0 else (
            np.inf if abs(placed) > 1e-9 else 0.0
        )
        records.append(
            {
                "ori9": ori,
                "state_fips": "50",
                "offense": offense,
                "footprint_status": CUSTOM_FOOTPRINT_STATUS_PLACED,
                "canonical_target_ori": ori,
                "shared_footprint_ori_count": 1,
                "ledger_control_mass": expected,
                "state_ledger_mass_total": expected,
                "ledger_share_of_state": 1.0,
                "control_surface_factor": 1.0,
                "surface_control_mass": expected,
                "expected_placed_mass": expected,
                "placed_mass": placed,
                "placed_minus_expected": placed - expected,
                "relative_error": relative,
            }
        )
    return pd.DataFrame.from_records(records)


def _check(tmp_path: Path, frame: pd.DataFrame) -> tuple[dict, list[str]]:
    frame.to_parquet(footprint_mass_conservation_path(tmp_path, year=VALIDATOR.YEAR), index=False)
    issues: list[str] = []
    summary = VALIDATOR._check_footprint_mass_conservation(output_dir=tmp_path, issues=issues)
    return summary, issues


def _vermont_barracks_rows(
    robbery_placed: float, *, other_total: float = 53.0
) -> list[tuple[str, str, float, float]]:
    """VTVSP1600 as the surface carries it: seven offences, 57 expected, robbery clipped."""
    rows = [("VTVSP1600", "robbery", 4.0, robbery_placed)]
    rows += [
        ("VTVSP1600", offense, other_total / 6.0, other_total / 6.0)
        for offense in OFFENSES
        if offense != "robbery"
    ]
    return rows


def test_the_vermont_robbery_row_no_longer_registers_at_all(tmp_path):
    """Ledger 4.0, placed 5.174: +1.19 offences on the row, 2.06% across the ORI."""
    summary, issues = _check(tmp_path, _conservation_frame(_vermont_barracks_rows(5.174)))
    assert summary["max_ori_relative_error"] == pytest.approx(1.174 / 57.0, rel=1e-3)
    assert summary["max_ori_relative_error"] < FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR
    assert summary["oris_over_threshold"] == 0
    assert summary["rows_over_threshold"] == 0
    assert summary["ok"] is True
    assert issues == []


def test_a_per_offence_row_over_the_floor_is_advisory_while_its_ori_passes(tmp_path):
    """3.0 offences on one row, still only 2.4% of the ORI's own control -- inside the gate."""
    summary, issues = _check(
        tmp_path, _conservation_frame(_vermont_barracks_rows(7.0, other_total=120.0))
    )
    assert summary["rows_over_threshold"] == 1
    assert summary["rows_over_threshold_advisory"] == 1
    assert summary["per_offense_check_blocking"] is False
    assert summary["rows_over_threshold_sample"][0]["offense"] == "robbery"
    assert summary["max_ori_relative_error"] < FOOTPRINT_MASS_CONSERVATION_MAX_RELATIVE_ERROR
    assert summary["oris_over_threshold"] == 0
    assert summary["ok"] is True
    assert issues == []


def test_a_broken_rake_still_blocks_and_the_per_offence_row_names_the_offence(tmp_path):
    rows = [("VTVSP1600", "robbery", 4.0, 40.0)]
    rows += [
        ("VTVSP1600", offense, 53.0 / 6.0, 53.0 / 6.0)
        for offense in OFFENSES
        if offense != "robbery"
    ]
    summary, issues = _check(tmp_path, _conservation_frame(rows))
    assert summary["oris_over_threshold"] == 1
    assert summary["rows_over_threshold"] == 1
    assert summary["per_offense_check_blocking"] is True
    assert summary["rows_over_threshold_advisory"] == 0
    assert summary["ok"] is False
    assert len(issues) == 2
    assert any("ORI/offense row(s)" in issue and "robbery" in issue for issue in issues)


def test_a_non_placing_footprint_that_placed_mass_still_blocks(tmp_path):
    frame = _conservation_frame(_vermont_barracks_rows(4.0))
    frame.loc[frame["offense"].eq("robbery"), "footprint_status"] = (
        CUSTOM_FOOTPRINT_STATUS_SUPPRESSED_DUPLICATE
    )
    summary, issues = _check(tmp_path, frame)
    assert summary["non_placing_rows_with_mass"] == 1
    assert summary["ok"] is False
    assert any("still placed mass" in issue for issue in issues)
