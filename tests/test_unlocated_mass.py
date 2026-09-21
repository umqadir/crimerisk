"""The unlocated-mass bucket (contract: UNLOCATED_MASS_CONTRACT.md).

Six things are load-bearing and are asserted here: the route partition says the right thing about
which statewide answers are findings and which are failures, the route decomposition reconstructs
the bucket or raises, the widened conservation identity is real (it fails when mass goes missing
and when a bucket is invented), the release mirror fails a half-migrated artifact in BOTH
directions, the mirror's widened identity accepts a conserving edition and rejects a short one,
and every path is unchanged when the flag is off.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    AllocationBuildConfig,
    OVERLAP_ROUTE_CROSSWALK_STATEWIDE_GEOMETRY,
    OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT,
    OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE,
    OVERLAP_ROUTE_RARE_OFFENSE_DEMOTION,
    OVERLAP_ROUTE_REGISTRY_HOLD,
    OVERLAP_ROUTE_REGISTRY_STATEWIDE,
    OVERLAP_ROUTE_SERVICE_NO_ELIGIBLE_RECEIVER,
    OVERLAP_ROUTE_UNATTRIBUTED_RESIDUAL,
    OVERLAP_ROUTE_UNRESOLVED_NO_TARGET,
    OVERLAP_ROUTE_UNSUPPORTED_COUNTY,
    STATE_OVERLAP_TYPE,
    UNLOCATED_GROUP_KIND,
    UNLOCATED_MASS_COLUMNS,
    UNLOCATED_ROUTES,
    _assert_unlocated_mass_conservation,
    _coalesce_overlap_rake_groups,
    _unlocated_route_detail,
    unlocated_mass_path,
)
from crimerisk.crime import OFFENSES_7


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"


def _validator():
    spec = importlib.util.spec_from_file_location("validate_release_outputs", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = _validator()
YEAR = VALIDATOR.YEAR


# --- the route partition ----------------------------------------------------


def test_the_flag_is_off_by_default():
    assert AllocationBuildConfig().enable_unlocated_mass is False


def test_a_declared_statewide_footprint_is_a_finding_and_stays_located():
    # The editorial content of the whole lane. A reviewer's `keep_statewide_overlap`, the
    # crosswalk's own statewide geometry hint, and the named rare-offense evidence rule all reach
    # the same rung, and none of them is a resolution failure.
    for route in (
        OVERLAP_ROUTE_REGISTRY_STATEWIDE,
        OVERLAP_ROUTE_CROSSWALK_STATEWIDE_GEOMETRY,
        OVERLAP_ROUTE_RARE_OFFENSE_DEMOTION,
    ):
        assert route not in UNLOCATED_ROUTES


def test_a_failure_to_resolve_is_unlocated():
    assert UNLOCATED_ROUTES == frozenset(
        {
            OVERLAP_ROUTE_REGISTRY_HOLD,
            OVERLAP_ROUTE_UNRESOLVED_NO_TARGET,
            OVERLAP_ROUTE_UNSUPPORTED_COUNTY,
            OVERLAP_ROUTE_UNATTRIBUTED_RESIDUAL,
            OVERLAP_ROUTE_SERVICE_NO_ELIGIBLE_RECEIVER,
            # A footprint ORI's own admitted mass that the per-ORI rake refuses to place: a
            # duplicate reporter key on a shared footprint, or a footprint the block-group prior
            # does not carry. Both are resolution failures, so both are published here.
            OVERLAP_ROUTE_FOOTPRINT_SUPPRESSED_DUPLICATE,
            OVERLAP_ROUTE_FOOTPRINT_NO_PLACEABLE_SUPPORT,
        }
    )


def test_the_validator_transcribes_the_same_routes():
    # The mirror is a transcription, not an import, so it can disagree with the producer. It must
    # not disagree by accident.
    assert sorted(VALIDATOR.UNLOCATED_MASS_ROUTES) == sorted(UNLOCATED_ROUTES)
    assert sorted(VALIDATOR.UNLOCATED_MASS_COLUMNS) == sorted(UNLOCATED_MASS_COLUMNS)
    assert VALIDATOR.UNLOCATED_MASS_JURISDICTION_TYPE == STATE_OVERLAP_TYPE


# --- the route decomposition ------------------------------------------------


def test_a_missing_lane_target_coalesces_with_an_existing_unlocated_group():
    """A zero-raw placeholder and its residual target must remain one join key."""
    adjustment = pd.DataFrame(
        [
            {
                "state_fips": "46",
                "offense": "aggravated_assault",
                "group_kind": UNLOCATED_GROUP_KIND,
                "group_id": "46",
                "adjustment_raw_count": 0.0,
                "adjustment_target_count": 0.0,
                "county_anchor_evidence_count": 0.0,
            },
            {
                "state_fips": "46",
                "offense": "aggravated_assault",
                "group_kind": UNLOCATED_GROUP_KIND,
                "group_id": "46",
                "adjustment_raw_count": 0.0,
                "adjustment_target_count": 204.1309102239078,
                "county_anchor_evidence_count": 0.0,
            },
        ]
    )
    out = _coalesce_overlap_rake_groups(
        adjustment,
        raw_column="adjustment_raw_count",
        target_column="adjustment_target_count",
    )
    assert len(out) == 1
    assert out.iloc[0]["adjustment_target_count"] == pytest.approx(204.1309102239078)


def _merged(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["state_fips"] = frame["state_fips"].astype("string")
    frame["offense"] = frame["offense"].astype("string")
    return frame


def _overlap_controls(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["state_fips"] = frame["state_fips"].astype("string")
    frame["offense"] = frame["offense"].astype("string")
    return frame


def test_the_route_detail_reconstructs_the_bucket_from_the_same_rake():
    merged = _merged(
        [
            {
                "state_fips": "42",
                "offense": "larceny",
                "group_kind": UNLOCATED_GROUP_KIND,
                "overlap_resolution_route": OVERLAP_ROUTE_UNRESOLVED_NO_TARGET,
                "reported_count_current_supported": 30.0,
                "agency_adjustment_count": 0.0,
            },
            {
                "state_fips": "42",
                "offense": "larceny",
                "group_kind": UNLOCATED_GROUP_KIND,
                "overlap_resolution_route": OVERLAP_ROUTE_REGISTRY_HOLD,
                "reported_count_current_supported": 10.0,
                "agency_adjustment_count": 0.0,
            },
            {
                "state_fips": "42",
                "offense": "larceny",
                "group_kind": "statewide_overlap",
                "overlap_resolution_route": OVERLAP_ROUTE_REGISTRY_STATEWIDE,
                "reported_count_current_supported": 60.0,
                "agency_adjustment_count": 0.0,
            },
        ]
    )
    controls = _overlap_controls(
        [
            {
                "state_fips": "42",
                "offense": "larceny",
                "state_target": 200.0,
                "state_reported_target": 200.0,
                "state_adjustment_target": 0.0,
            }
        ]
    )
    # 40 of the state's 100 raw counts are unlocated, so 40% of the 200-count target is.
    grouped = pd.DataFrame(
        [
            {
                "state_fips": "42",
                "offense": "larceny",
                "group_kind": UNLOCATED_GROUP_KIND,
                "group_id": "42",
                "target_count": 80.0,
            }
        ]
    )
    grouped["state_fips"] = grouped["state_fips"].astype("string")
    grouped["offense"] = grouped["offense"].astype("string")

    out = _unlocated_route_detail(
        merged=merged, grouped=grouped, overlap_controls=controls, residual_delta={}
    )
    assert list(out.columns) == list(UNLOCATED_MASS_COLUMNS)
    row = out.iloc[0]
    assert row["unlocated_count"] == pytest.approx(80.0)
    assert row["unlocated_count_" + OVERLAP_ROUTE_UNRESOLVED_NO_TARGET] == pytest.approx(60.0)
    assert row["unlocated_count_" + OVERLAP_ROUTE_REGISTRY_HOLD] == pytest.approx(20.0)
    assert row["jurisdiction_id"] == f"42:{STATE_OVERLAP_TYPE}"
    assert row["jurisdiction_type"] == STATE_OVERLAP_TYPE
    assert row["unlocated_share_of_control"] == pytest.approx(0.4)


def test_the_state_residual_is_attributed_in_full_and_never_apportioned():
    merged = _merged(
        [
            {
                "state_fips": "09",
                "offense": "murder",
                "group_kind": "statewide_overlap",
                "overlap_resolution_route": OVERLAP_ROUTE_REGISTRY_STATEWIDE,
                "reported_count_current_supported": 0.0,
                "agency_adjustment_count": 0.0,
            }
        ]
    )
    controls = _overlap_controls(
        [
            {
                "state_fips": "09",
                "offense": "murder",
                "state_target": 3.0,
                "state_reported_target": 3.0,
                "state_adjustment_target": 0.0,
            }
        ]
    )
    grouped = pd.DataFrame(
        [
            {
                "state_fips": "09",
                "offense": "murder",
                "group_kind": UNLOCATED_GROUP_KIND,
                "group_id": "09",
                "target_count": 3.0,
            }
        ]
    )
    grouped["state_fips"] = grouped["state_fips"].astype("string")
    grouped["offense"] = grouped["offense"].astype("string")
    out = _unlocated_route_detail(
        merged=merged,
        grouped=grouped,
        overlap_controls=controls,
        residual_delta={("09", "murder"): 3.0},
    )
    row = out.iloc[0]
    assert row["unlocated_count_" + OVERLAP_ROUTE_UNATTRIBUTED_RESIDUAL] == pytest.approx(3.0)
    assert row["unlocated_count_" + OVERLAP_ROUTE_UNRESOLVED_NO_TARGET] == pytest.approx(0.0)


def test_a_route_table_that_does_not_add_up_raises():
    """The mass that reached the bucket without a recorded route is exactly the bug this catches."""
    merged = _merged(
        [
            {
                "state_fips": "42",
                "offense": "larceny",
                "group_kind": "statewide_overlap",
                "overlap_resolution_route": OVERLAP_ROUTE_REGISTRY_STATEWIDE,
                "reported_count_current_supported": 100.0,
                "agency_adjustment_count": 0.0,
            }
        ]
    )
    controls = _overlap_controls(
        [
            {
                "state_fips": "42",
                "offense": "larceny",
                "state_target": 200.0,
                "state_reported_target": 200.0,
                "state_adjustment_target": 0.0,
            }
        ]
    )
    grouped = pd.DataFrame(
        [
            {
                "state_fips": "42",
                "offense": "larceny",
                "group_kind": UNLOCATED_GROUP_KIND,
                "group_id": "42",
                "target_count": 25.0,
            }
        ]
    )
    grouped["state_fips"] = grouped["state_fips"].astype("string")
    grouped["offense"] = grouped["offense"].astype("string")
    with pytest.raises(ValueError, match="does not reconstruct the bucket"):
        _unlocated_route_detail(
            merged=merged, grouped=grouped, overlap_controls=controls, residual_delta={}
        )


# --- the widened identity, at build time ------------------------------------


def _surface(published: dict[tuple[str, str], float]) -> pd.DataFrame:
    states = sorted({state for state, _offense in published})
    frame = pd.DataFrame({"state_fips": states})
    for offense in OFFENSES_7:
        frame[f"expected_count_{offense}"] = [
            published.get((state, offense), 0.0) for state in states
        ]
    return frame


def _controls(control: dict[tuple[str, str], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_fips": state, "offense": offense, "adjusted_count_ags_core": value}
            for (state, offense), value in control.items()
        ]
    )


def _unlocated(withheld: dict[tuple[str, str], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_fips": state, "offense": offense, "unlocated_count": value}
            for (state, offense), value in withheld.items()
        ]
    )


def test_published_plus_unlocated_equals_control():
    summary = _assert_unlocated_mass_conservation(
        surface=_surface({("42", "larceny"): 900.0}),
        controls=_controls({("42", "larceny"): 1000.0}),
        unlocated=_unlocated({("42", "larceny"): 100.0}),
        year=YEAR,
    )
    assert summary["identity"] == (
        "published_block_group_sum + unlocated + service_exports - service_imports "
        "== source_state_control"
    )
    assert summary["max_abs_delta"] == pytest.approx(0.0, abs=1e-9)
    assert summary["unlocated_total"] == pytest.approx(100.0)


def test_mass_that_left_the_surface_without_being_published_fails():
    with pytest.raises(ValueError, match="do not reconstruct the controls"):
        _assert_unlocated_mass_conservation(
            surface=_surface({("42", "larceny"): 900.0}),
            controls=_controls({("42", "larceny"): 1000.0}),
            unlocated=_unlocated({("42", "larceny"): 0.0}),
            year=YEAR,
        )


def test_a_bucket_the_surface_never_gave_up_fails():
    with pytest.raises(ValueError, match="do not reconstruct the controls"):
        _assert_unlocated_mass_conservation(
            surface=_surface({("42", "larceny"): 1000.0}),
            controls=_controls({("42", "larceny"): 1000.0}),
            unlocated=_unlocated({("42", "larceny"): 100.0}),
            year=YEAR,
        )


# --- the release mirror -----------------------------------------------------


def _companion_table() -> pd.DataFrame:
    rows = []
    for state in ("42", "09"):
        for offense in OFFENSES_7:
            unlocated = 100.0 if (state, offense) == ("42", "larceny") else 0.0
            control = 1000.0
            rows.append(
                {
                    "state_fips": state,
                    "jurisdiction_id": f"{state}:{STATE_OVERLAP_TYPE}",
                    "jurisdiction_type": STATE_OVERLAP_TYPE,
                    "offense": offense,
                    "unlocated_count": unlocated,
                    "control_count": control,
                    "unlocated_share_of_control": unlocated / control,
                    **{
                        f"unlocated_count_{route}": (
                            unlocated if route == OVERLAP_ROUTE_UNRESOLVED_NO_TARGET else 0.0
                        )
                        for route in sorted(UNLOCATED_ROUTES)
                    },
                }
            )
    return pd.DataFrame(rows)[list(UNLOCATED_MASS_COLUMNS)]


def _contract(*, enabled: bool, **overrides):
    if not enabled:
        return VALIDATOR.ReleaseContract(manifest_present=True, **overrides)
    payload = {
        "lanes": frozenset({"unlocated_mass"}),
        "manifest_present": True,
        "unlocated_mass_identity": VALIDATOR.UNLOCATED_MASS_IDENTITY,
        "unlocated_mass_companion_table": f"unlocated_mass_{YEAR}.parquet",
        "unlocated_mass_routes": tuple(sorted(UNLOCATED_ROUTES)),
        "unlocated_mass_semantics": "source_resolution",
    }
    payload.update(overrides)
    return VALIDATOR.ReleaseContract(**payload)


def test_the_mirror_accepts_a_well_formed_companion_table(tmp_path):
    _companion_table().to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    issues: list[str] = []
    summary = VALIDATOR._check_unlocated_mass(
        contract=_contract(enabled=True), state_output_dir=tmp_path, issues=issues
    )
    assert issues == []
    assert summary["unlocated_total"] == pytest.approx(100.0)
    assert summary["unlocated_by_route"][OVERLAP_ROUTE_UNRESOLVED_NO_TARGET] == pytest.approx(100.0)


def test_the_mirror_fails_a_declared_lane_with_no_table(tmp_path):
    issues: list[str] = []
    VALIDATOR._check_unlocated_mass(
        contract=_contract(enabled=True), state_output_dir=tmp_path, issues=issues
    )
    assert any("companion table" in issue and "absent" in issue for issue in issues)


def test_the_mirror_fails_a_table_with_no_declared_lane(tmp_path):
    _companion_table().to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    issues: list[str] = []
    VALIDATOR._check_unlocated_mass(
        contract=_contract(enabled=False), state_output_dir=tmp_path, issues=issues
    )
    assert any("half-migrated" in issue for issue in issues)


def test_the_mirror_refuses_doubt_semantics(tmp_path):
    """The bucket is a source-resolution statement; relabelling it is a display-contract breach."""
    _companion_table().to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    issues: list[str] = []
    VALIDATOR._check_unlocated_mass(
        contract=_contract(enabled=True, unlocated_mass_semantics="low_confidence"),
        state_output_dir=tmp_path,
        issues=issues,
    )
    assert any("semantics" in issue for issue in issues)


def test_the_mirror_fails_a_route_split_that_does_not_add_up(tmp_path):
    table = _companion_table()
    column = f"unlocated_count_{OVERLAP_ROUTE_UNRESOLVED_NO_TARGET}"
    table.loc[table[column].gt(0.0), column] = 40.0
    table.to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    issues: list[str] = []
    VALIDATOR._check_unlocated_mass(
        contract=_contract(enabled=True), state_output_dir=tmp_path, issues=issues
    )
    assert any("route decomposition" in issue for issue in issues)


def test_the_mirror_rejects_the_obsolete_four_route_manifest_contract(tmp_path):
    _companion_table().to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    obsolete_routes = tuple(
        route
        for route in sorted(UNLOCATED_ROUTES)
        if route != OVERLAP_ROUTE_SERVICE_NO_ELIGIBLE_RECEIVER
    )
    issues: list[str] = []
    VALIDATOR._check_unlocated_mass(
        contract=_contract(enabled=True, unlocated_mass_routes=obsolete_routes),
        state_output_dir=tmp_path,
        issues=issues,
    )
    assert any("manifest declares unlocated routes" in issue for issue in issues)


def test_the_mirror_fails_a_bucket_larger_than_its_control(tmp_path):
    table = _companion_table()
    table.loc[table["unlocated_count"].gt(0.0), "unlocated_count"] = 5000.0
    table.to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    issues: list[str] = []
    VALIDATOR._check_unlocated_mass(
        contract=_contract(enabled=True), state_output_dir=tmp_path, issues=issues
    )
    assert any("exceeds the control" in issue for issue in issues)


# --- the mirror's widened conservation identity -----------------------------


def _write_release_surfaces(directory: Path, published: dict[tuple[str, str], float]) -> None:
    surface = _surface(published)
    for name in (
        f"crimerisk_block_group_{YEAR}_ags_core.parquet",
        f"crimerisk_tract_{YEAR}_ags_core.parquet",
    ):
        surface.to_parquet(directory / name, index=False)


def test_the_release_identity_widens_by_exactly_the_companion_table(tmp_path):
    published = {("42", offense): (900.0 if offense == "larceny" else 1000.0) for offense in OFFENSES_7}
    published.update({("09", offense): 1000.0 for offense in OFFENSES_7})
    _write_release_surfaces(tmp_path, published)
    _companion_table().to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    controls = pd.DataFrame(
        [
            {
                "state_fips": state,
                "offense": offense,
                VALIDATOR.TOTAL_LANE_TARGET_COLUMN: 1000.0,
            }
            for state in ("42", "09")
            for offense in OFFENSES_7
        ]
    )
    issues: list[str] = []
    summary = VALIDATOR._published_surface_state_total_reconciliation(
        output_dir=tmp_path, controls=controls, issues=issues, contract=_contract(enabled=True)
    )
    assert issues == []
    assert summary["identity"] == VALIDATOR.UNLOCATED_MASS_IDENTITY
    assert summary["surfaces"]["block_group_ags_core"]["unlocated_total"] == pytest.approx(100.0)

    # The same surfaces under the LEGACY contract are short of their controls, which is the whole
    # reason the manifest has to declare the lane.
    legacy_issues: list[str] = []
    legacy = VALIDATOR._published_surface_state_total_reconciliation(
        output_dir=tmp_path, controls=controls, issues=legacy_issues
    )
    assert legacy["identity"] == VALIDATOR.LEGACY_MASS_IDENTITY
    assert legacy_issues


def test_the_release_identity_still_fails_when_the_bucket_does_not_cover_the_shortfall(tmp_path):
    published = {("42", offense): (500.0 if offense == "larceny" else 1000.0) for offense in OFFENSES_7}
    published.update({("09", offense): 1000.0 for offense in OFFENSES_7})
    _write_release_surfaces(tmp_path, published)
    _companion_table().to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    controls = pd.DataFrame(
        [
            {"state_fips": state, "offense": offense, VALIDATOR.TOTAL_LANE_TARGET_COLUMN: 1000.0}
            for state in ("42", "09")
            for offense in OFFENSES_7
        ]
    )
    issues: list[str] = []
    VALIDATOR._published_surface_state_total_reconciliation(
        output_dir=tmp_path, controls=controls, issues=issues, contract=_contract(enabled=True)
    )
    assert any("do not satisfy" in issue for issue in issues)


# --- byte-safety when the lane is off ---------------------------------------


def test_the_legacy_contract_reads_no_unlocated_lane(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"resolved_config": {}}))
    contract = VALIDATOR._load_release_contract(tmp_path)
    assert contract.unlocated_mass is False
    assert contract.unlocated_mass_identity is None


def test_a_manifest_that_declares_the_lane_is_read_as_declaring_it(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "resolved_config": {
                    "unlocated_mass": {
                        "enabled": True,
                        "identity": VALIDATOR.UNLOCATED_MASS_IDENTITY,
                        "companion_table": f"unlocated_mass_{YEAR}.parquet",
                        "unlocated_routes": sorted(UNLOCATED_ROUTES),
                        "semantics": "source_resolution",
                    }
                }
            }
        )
    )
    contract = VALIDATOR._load_release_contract(tmp_path)
    assert contract.unlocated_mass is True
    assert contract.unlocated_mass_identity == VALIDATOR.UNLOCATED_MASS_IDENTITY
    assert sorted(contract.unlocated_mass_routes) == sorted(UNLOCATED_ROUTES)


def test_the_off_lane_publishes_the_narrow_identity_string(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "resolved_config": {
                    "unlocated_mass": {
                        "enabled": False,
                        "identity": VALIDATOR.UNLOCATED_MASS_IDENTITY,
                    }
                }
            }
        )
    )
    contract = VALIDATOR._load_release_contract(tmp_path)
    issues: list[str] = []
    VALIDATOR._check_unlocated_mass(contract=contract, state_output_dir=tmp_path, issues=issues)
    assert any("the lane is off but the manifest declares the identity" in issue for issue in issues)


def test_share_is_the_ratio_of_the_counts_beside_it(tmp_path):
    table = _companion_table()
    table["unlocated_share_of_control"] = np.where(
        table["unlocated_count"].gt(0.0), 0.5, table["unlocated_share_of_control"]
    )
    table.to_parquet(unlocated_mass_path(tmp_path, year=YEAR), index=False)
    issues: list[str] = []
    VALIDATOR._check_unlocated_mass(
        contract=_contract(enabled=True), state_output_dir=tmp_path, issues=issues
    )
    assert any("not the ratio of the two counts" in issue for issue in issues)
