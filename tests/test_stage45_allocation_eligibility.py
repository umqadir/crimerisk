"""Stage 4/5 rule batch: the crosswalk share basis, the custom-footprint activity term, the
concurrent-jurisdiction carve-out, the ambient-blind footprint eligibility class, the transient
guard, the suppression vocabulary, and the one city-key vocabulary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO,
    AllocationBuildConfig,
    CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_ACTIVITY,
    CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT,
    CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_SERVICE_AREA_PRIOR,
    FOOTPRINT_DERIVED_MASS_SHARE_FLOOR,
    INSUFFICIENT_AMBIENT_EXPOSURE_REASON,
    MURDER_TRACT_POSTERIOR_PRIOR_INCIDENTS,
    MODEL_ONLY_MURDER_TRACT_RATE_RATIO_CAP,
    MODEL_ONLY_ROBBERY_BG_RATE_RATIO_CAP,
    OFFENSES_7,
    RARE_OFFENSE_TRACT_SUPPORT,
    TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO,
    _build_exclusive_footprint_displacement,
    _apply_model_only_allocation_envelopes,
    _calibrated_residual_policy_labels,
    _custom_footprint_component_shares,
    _expected_count_col,
    _finalize_output,
    _footprint_derived_count_col,
    _load_concurrent_jurisdiction_carveouts,
    _load_overlap_custom_footprints,
    _tract_incident_posterior,
)
from crimerisk.crosswalk_shares import (
    BELOW_FLOOR,
    CROSSWALK_MINIMUM_RECIPIENT_SHARE,
    DEGENERATE_BASIS,
    FLOOR_EXEMPT_COUNTY_REMAINDER,
    FLOOR_EXEMPT_JURISDICTION,
    RECIPIENT,
    ZERO_ON_BASIS,
    assert_allocation_shares_conserve,
    normalize_block_group_allocation_shares,
)
from crimerisk.city_incidents import _washington_dc_murder_history_only


def test_calibrated_residual_policy_labels_are_string_typed_when_empty():
    labels = _calibrated_residual_policy_labels(pd.Series(dtype=float))

    assert labels.empty
    assert str(labels.dtype) == "string"


def test_calibrated_residual_policy_labels_include_formatted_tau():
    labels = _calibrated_residual_policy_labels(pd.Series([0.25, 0.5]))

    assert labels.tolist() == ["calibrated_residual_tau0.25", "calibrated_residual_tau0.50"]


# ----------------------------------------------------------------- crosswalk share basis


def _crosswalk_row(bg, jurisdiction_id, *, pop20, housing20=0.0, blocks=1.0, aland20=1.0,
                   jurisdiction_type="municipal"):
    return {
        "state_fips": bg[:2],
        "block_group_geoid": bg,
        "jurisdiction_id": jurisdiction_id,
        "jurisdiction_type": jurisdiction_type,
        "pop20": pop20,
        "housing20": housing20,
        "blocks": blocks,
        "aland20": aland20,
    }


def _with_shares(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for raw, share in (
        ("pop20", "pop_share"),
        ("housing20", "housing_share"),
        ("blocks", "block_share"),
        ("aland20", "aland_share"),
    ):
        total = frame.groupby(["state_fips", "block_group_geoid"])[raw].transform("sum")
        frame[share] = np.where(total > 0, frame[raw] / total.replace(0, 1), 0.0)
    frame["allocation_share"] = np.nan
    return frame


def test_one_basis_per_block_group_and_zero_population_fragment_is_not_a_recipient():
    """The a6 mechanism: a populated fragment on pop_share plus a zero-population fragment on
    block_share, summed as if commensurable. Under one basis the zero-population fragment is out."""
    frame = _with_shares(
        [
            _crosswalk_row("010010000011", "J_POPULATED", pop20=1000.0, blocks=10.0),
            _crosswalk_row("010010000011", "J_EMPTY", pop20=0.0, blocks=5.0),
        ]
    )
    out = normalize_block_group_allocation_shares(frame)
    assert set(out["allocation_basis"]) == {"pop_share"}
    populated = out.loc[out["jurisdiction_id"].eq("J_POPULATED")].iloc[0]
    empty = out.loc[out["jurisdiction_id"].eq("J_EMPTY")].iloc[0]
    assert populated["allocation_share"] == pytest.approx(1.0)
    assert empty["allocation_share"] == 0.0
    assert empty["allocation_recipient_status"] == ZERO_ON_BASIS
    assert_allocation_shares_conserve(out)


def test_sliver_below_the_recipient_floor_routes_to_the_remaining_recipients():
    """The a7 mechanism (Lakewood CO at share 0.017 delivering 67.5% of a 1,025-person cell)."""
    frame = _with_shares(
        [
            _crosswalk_row("010010000011", "J_PARENT", pop20=1000.0),
            _crosswalk_row("010010000011", "J_SLIVER", pop20=16.0),
            # A second block group so J_SLIVER is not stranded by the never-strand guard.
            _crosswalk_row("010010000022", "J_SLIVER", pop20=900.0),
        ]
    )
    out = normalize_block_group_allocation_shares(frame)
    cell = out.loc[out["block_group_geoid"].eq("010010000011")].set_index("jurisdiction_id")
    assert cell.loc["J_SLIVER", "allocation_share_before_recipient_floor"] < CROSSWALK_MINIMUM_RECIPIENT_SHARE
    assert cell.loc["J_SLIVER", "allocation_recipient_status"] == BELOW_FLOOR
    assert cell.loc["J_SLIVER", "allocation_share"] == 0.0
    assert cell.loc["J_PARENT", "allocation_share"] == pytest.approx(1.0)
    assert cell.loc["J_PARENT", "allocation_recipient_status"] == RECIPIENT
    assert_allocation_shares_conserve(out)


def test_a_fragment_just_above_the_floor_stays_a_recipient():
    frame = _with_shares(
        [
            _crosswalk_row("010010000011", "J_PARENT", pop20=1000.0),
            _crosswalk_row("010010000011", "J_SMALL", pop20=30.0),
        ]
    )
    out = normalize_block_group_allocation_shares(frame)
    small = out.loc[out["jurisdiction_id"].eq("J_SMALL")].iloc[0]
    assert small["allocation_share_before_recipient_floor"] > CROSSWALK_MINIMUM_RECIPIENT_SHARE
    assert small["allocation_recipient_status"] == RECIPIENT
    assert small["allocation_share"] > 0.0


def test_the_floor_never_strands_a_jurisdiction_whose_only_support_is_a_sliver():
    frame = _with_shares(
        [
            _crosswalk_row("010010000011", "J_PARENT", pop20=1000.0),
            _crosswalk_row("010010000011", "J_ONLY_A_SLIVER", pop20=5.0),
        ]
    )
    out = normalize_block_group_allocation_shares(frame)
    sliver = out.loc[out["jurisdiction_id"].eq("J_ONLY_A_SLIVER")].iloc[0]
    assert sliver["allocation_recipient_status"] == FLOOR_EXEMPT_JURISDICTION
    assert sliver["allocation_share"] > 0.0
    assert_allocation_shares_conserve(out)


def test_the_floor_never_strands_a_county_remainder():
    rows = [
        _crosswalk_row("010010000011", "J_CITY", pop20=1000.0),
        _crosswalk_row(
            "010010000011", "01:state_nonmunicipal_remainder", pop20=4.0,
            jurisdiction_type="state_nonmunicipal_remainder",
        ),
        # The same remainder jurisdiction has real support in a DIFFERENT county, so the
        # jurisdiction-level exemption does not cover county 001.
        _crosswalk_row(
            "010030000011", "01:state_nonmunicipal_remainder", pop20=800.0,
            jurisdiction_type="state_nonmunicipal_remainder",
        ),
    ]
    out = normalize_block_group_allocation_shares(_with_shares(rows))
    remainder = out.loc[out["block_group_geoid"].eq("010010000011")
                        & out["jurisdiction_type"].eq("state_nonmunicipal_remainder")].iloc[0]
    assert remainder["allocation_recipient_status"] == FLOOR_EXEMPT_COUNTY_REMAINDER
    assert remainder["allocation_share"] > 0.0
    assert_allocation_shares_conserve(out)


def test_degenerate_block_group_falls_back_to_an_equal_split():
    frame = _with_shares(
        [
            _crosswalk_row("010010000011", "J_A", pop20=0.0, housing20=0.0, blocks=0.0, aland20=0.0),
            _crosswalk_row("010010000011", "J_B", pop20=0.0, housing20=0.0, blocks=0.0, aland20=0.0),
        ]
    )
    out = normalize_block_group_allocation_shares(frame)
    assert set(out["allocation_basis"]) == {DEGENERATE_BASIS}
    assert out["allocation_share"].tolist() == pytest.approx([0.5, 0.5])
    assert_allocation_shares_conserve(out)


def test_normalization_is_idempotent():
    frame = _with_shares(
        [
            _crosswalk_row("010010000011", "J_PARENT", pop20=1000.0),
            _crosswalk_row("010010000011", "J_SLIVER", pop20=16.0),
            _crosswalk_row("010010000022", "J_SLIVER", pop20=900.0),
        ]
    )
    once = normalize_block_group_allocation_shares(frame)
    twice = normalize_block_group_allocation_shares(
        once.drop(columns=["allocation_basis", "allocation_share_before_recipient_floor",
                           "allocation_recipient_status"])
    )
    assert twice["allocation_share"].tolist() == pytest.approx(once["allocation_share"].tolist())


def test_conservation_assertion_catches_a_broken_share_column():
    frame = _with_shares(
        [
            _crosswalk_row("010010000011", "J_A", pop20=600.0),
            _crosswalk_row("010010000011", "J_B", pop20=400.0),
        ]
    )
    out = normalize_block_group_allocation_shares(frame)
    out.loc[0, "allocation_share"] = 0.1
    with pytest.raises(ValueError, match="must sum to 1 per block group"):
        assert_allocation_shares_conserve(out)


# ------------------------------------------------- custom-footprint activity term


def _footprint_pool(basis: str, *, weights: list[float], bg_weights: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["06"] * len(weights),
            "ori9": ["CA0000000"] * len(weights),
            "offense": ["larceny"] * len(weights),
            "bg_id": [f"0600100000{i:02d}" for i in range(len(weights))],
            "weight_share": weights,
            "weight_share_basis": [basis] * len(weights),
            "bg_responsibility_population_coverage_share": weights,
            "responsibility_fraction_basis": ["test_within_bg_population"] * len(weights),
            "bg_weight": bg_weights,
        }
    )


def test_resident_basis_footprint_gets_the_activity_term():
    pool = _footprint_pool(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, weights=[0.5, 0.5], bg_weights=[300.0, 100.0]
    )
    out = _custom_footprint_component_shares(pool)
    assert out["component_share"].sum() == pytest.approx(1.0)
    # Equal within-BG responsibility, 3:1 activity -> 3:1 mass.
    assert out["component_share"].tolist() == pytest.approx([0.75, 0.25])


def test_resident_basis_does_not_square_unequal_full_bg_population() -> None:
    pool = _footprint_pool(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT,
        weights=[0.1, 0.9],
        bg_weights=[10.0, 90.0],
    )
    pool["bg_responsibility_population_coverage_share"] = [1.0, 1.0]

    out = _custom_footprint_component_shares(pool)

    # The expected-count prior already contains population/exposure. Two fully responsible BGs
    # retain its 1:9 ratio instead of multiplying by the footprint's 1:9 population share again.
    assert out["component_share"].tolist() == pytest.approx([0.1, 0.9])


def test_resident_basis_scales_partial_bg_by_ori_responsibility_fraction() -> None:
    pool = _footprint_pool(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT,
        weights=[0.5, 0.5],
        bg_weights=[10.0, 10.0],
    )
    pool["bg_responsibility_population_coverage_share"] = [0.25, 1.0]

    out = _custom_footprint_component_shares(pool)

    assert out["component_share"].tolist() == pytest.approx([0.2, 0.8])
    assert out["bg_responsibility_population_coverage_share"].tolist() == pytest.approx(
        [0.25, 1.0]
    )


def test_activity_basis_footprint_is_used_verbatim():
    pool = _footprint_pool(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_ACTIVITY, weights=[0.8, 0.2], bg_weights=[1.0, 900.0]
    )
    out = _custom_footprint_component_shares(pool)
    assert out["component_share"].tolist() == pytest.approx([0.8, 0.2])


def test_resident_basis_footprint_with_no_model_signal_uses_covered_population_fallback():
    pool = _footprint_pool(
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, weights=[0.7, 0.3], bg_weights=[0.0, 0.0]
    )
    out = _custom_footprint_component_shares(pool)
    assert out["component_share"].tolist() == pytest.approx([0.7, 0.3])


def test_promoted_custom_footprint_registry_declares_a_valid_basis_for_every_ori():
    from crimerisk.paths import get_paths

    footprints = _load_overlap_custom_footprints(get_paths())
    assert not footprints.empty
    assert set(footprints["weight_share_basis"]) <= {
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT,
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_ACTIVITY,
        CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_SERVICE_AREA_PRIOR,
    }
    per_pool = footprints.groupby(["ori9", "state_fips"])["weight_share_basis"].nunique()
    assert int(per_pool.max()) == 1


def test_custom_footprint_loader_rejects_an_unknown_basis(tmp_path):
    from crimerisk.paths import get_paths

    paths = get_paths()
    source = pd.read_csv(paths.repo_root / "configs" / "overlap_custom_footprints.csv")
    broken = source[source["ori"].eq(source["ori"].iloc[0])].copy()
    broken["weight_share_basis"] = "made_up_basis"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    broken.to_csv(config_dir / "overlap_custom_footprints.csv", index=False)

    class _Paths:
        repo_root = tmp_path

    with pytest.raises(ValueError, match="unknown weight_share_basis"):
        _load_overlap_custom_footprints(_Paths())


# --------------------------------------------- concurrent-jurisdiction carve-out


def test_promoted_carveout_registry_is_declared_unresolved_everywhere():
    from crimerisk.paths import get_paths

    carveouts = _load_concurrent_jurisdiction_carveouts(get_paths())
    assert not carveouts.empty
    assert set(carveouts["reviewer_note"]) == {"concurrent_jurisdiction_unresolved"}
    assert carveouts["remainder_exposure_retained_share"].max() < 0.5
    assert carveouts["reporting_remainder_agency_mass_2024"].min() > 0.0
    assert carveouts["county_geoid"].is_unique


def test_carveout_registry_rejects_any_other_reviewer_note(tmp_path):
    from crimerisk.paths import get_paths

    source = pd.read_csv(get_paths().repo_root / "configs" / "concurrent_jurisdiction_carveouts.csv")
    source.loc[0, "reviewer_note"] = "resolved_shared_jurisdiction"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    source.to_csv(config_dir / "concurrent_jurisdiction_carveouts.csv", index=False)

    class _Paths:
        repo_root = tmp_path

    with pytest.raises(ValueError, match="reviewer_note"):
        _load_concurrent_jurisdiction_carveouts(_Paths())


def _displacement_fixtures():
    overrides = pd.DataFrame(
        {
            "ori9": ["XX0000001"],
            "displaces_county_remainder": [True],
            "final_overlap_treatment": ["localize_to_custom_footprint"],
        }
    )
    footprints = pd.DataFrame(
        {
            "ori9": ["XX0000001", "XX0000001"],
            "state_fips": ["40", "40"],
            "bg_id": ["401130001001", "400010001001"],
            "weight_share": [0.5, 0.5],
            "bg_population_coverage_share": [0.9, 0.9],
        }
    )
    return overrides, footprints


def test_carveout_county_stops_displacing_and_other_counties_do_not():
    overrides, footprints = _displacement_fixtures()
    carveouts = pd.DataFrame({"county_geoid": ["40113"]})
    baseline = _build_exclusive_footprint_displacement(
        overrides=overrides, custom_footprints=footprints
    )
    assert set(baseline["bg_id"]) == {"401130001001", "400010001001"}
    carved = _build_exclusive_footprint_displacement(
        overrides=overrides,
        custom_footprints=footprints,
        concurrent_jurisdiction_carveouts=carveouts,
    )
    assert set(carved["bg_id"]) == {"400010001001"}


def test_a_carveout_county_no_footprint_touches_fails_closed():
    overrides, footprints = _displacement_fixtures()
    carveouts = pd.DataFrame({"county_geoid": ["40113", "99999"]})
    with pytest.raises(ValueError, match="no displacing footprint touches"):
        _build_exclusive_footprint_displacement(
            overrides=overrides,
            custom_footprints=footprints,
            concurrent_jurisdiction_carveouts=carveouts,
        )


# ------------------------------------------------------ Stage 5 eligibility + vocabulary


def _surface_row(bg, *, pop, households, exposure, counts, footprint_counts, landscan=0.0):
    row = {
        "block_group_geoid": bg,
        "tract_id": bg[:11],
        "state_fips": bg[:2],
        "population_2024": pop,
        "households_total": households,
        "commercial_premises_total": 0.0,
        "destination_poi_total": 0.0,
        "daytime_population_jobs_proxy": pop,
        "landscan_day_pop": landscan,
        "exposure_proxy_2024": exposure,
        "burglary_premises_total": max(households, 1.0),
        "aggregate_vehicles_total": max(pop * 0.8, 1.0),
        "vehicle_exposure_2024": max(pop * 0.8, 1.0),
        "land_area_sq_mi": 1.0,
        "eb_jurisdiction_id": "J1",
        "eb_jurisdiction_type": "municipal",
    }
    for offense in OFFENSES_7:
        row[_expected_count_col(offense)] = counts
        row[_footprint_derived_count_col(offense)] = footprint_counts
    return row


def _finalized_test_surface(*, census_residential: bool = False) -> pd.DataFrame:
    rows = [
        # 0: ambient-blind footprint cell -- majority footprint mass, no lift, huge implied rate
        _surface_row("010010000011", pop=281, households=120, exposure=281.0, counts=50.0, footprint_counts=49.5),
        # 1: same rate, but the exposure denominator DOES carry ambient lift
        _surface_row("010010000012", pop=281, households=120, exposure=3000.0, counts=50.0, footprint_counts=49.5, landscan=3000.0),
        # 2: footprint-derived but an ordinary rate
        _surface_row("010010000013", pop=281, households=120, exposure=281.0, counts=1.0, footprint_counts=0.99),
        # 3: footprint mass present but a minority of the cell
        _surface_row("010010000014", pop=281, households=120, exposure=281.0, counts=50.0, footprint_counts=10.0),
        # 4: non-residential
        _surface_row("010010000015", pop=20, households=2, exposure=20.0, counts=5.0, footprint_counts=0.0),
        # 5: transient -- 10x lift and a very high resident index, no footprint mass
        _surface_row("010010000016", pop=57, households=22, exposure=570.0, counts=30.0, footprint_counts=0.0),
        # 6: footprint-derived, no lift, and a POINT rate above 3x national on a count small
        #    enough that its Poisson interval reaches well below it -- the rare-offense noise
        #    case the lower bound is there to reject.
        _surface_row("010010000017", pop=281, households=120, exposure=281.0, counts=11.0, footprint_counts=11.0),
        # the ordinary country that sets the national rates
        *[
            _surface_row(f"0100100000{i:02d}", pop=2000, households=800, exposure=2000.0, counts=20.0, footprint_counts=0.0)
            for i in range(20, 60)
        ],
    ]
    if census_residential:
        for row in rows:
            for offense in ("murder", "rape", "robbery", "aggravated_assault", "larceny"):
                row[f"opportunity_normalizer_{offense}"] = row["population_2024"]
    return _finalize_output(
        pd.DataFrame(rows),
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(
            enable_exposure_ensemble=census_residential,
            exposure_residential_leg_source=(
                "census_release_population" if census_residential else "landscan_night"
            ),
        ),
    ).set_index("block_group_geoid")


def test_ambient_blind_footprint_only_suppresses_the_person_exposure_index():
    surface = _finalized_test_surface()
    cell = surface.loc["010010000011"]
    assert bool(cell["footprint_ambient_exposure_missing_larceny"]) is True
    assert cell["footprint_derived_count_share_larceny"] == pytest.approx(0.99)
    assert pd.isna(cell["index_larceny_primary"])
    assert pd.notna(cell["index_larceny_resident"])
    # Suppression propagates into the composite. The index-average scores are built from the
    # per-offense PRIMARY indexes, so a cell that refuses to show its own larceny index may not
    # publish a score that asserts one; substituting the parent tract's value is what let
    # Poarch Creek publish a composite of 1,878.8 beside a null larceny point. The resident
    # arm is untouched, because its denominator is fine (see the assertion above).
    assert pd.isna(cell["index_total_primary_event_weighted"])
    assert pd.notna(cell["index_total_part1_resident"])
    # Counts and density are never nulled by an eligibility rule.
    assert cell["expected_count_larceny"] == pytest.approx(50.0)
    assert cell["crime_density_larceny"] == pytest.approx(50.0)
    assert cell["estimate_mode_larceny"] == "insufficient_exposure"
    assert cell["denominator_reason_larceny"] == INSUFFICIENT_AMBIENT_EXPOSURE_REASON
    assert cell["resident_denominator_reason_larceny"] == "publishable"
    for offense in ("burglary", "motor_vehicle_theft"):
        assert bool(cell[f"footprint_ambient_exposure_missing_{offense}"]) is False
        assert pd.notna(cell[f"index_{offense}_primary"])
        assert pd.notna(cell[f"index_{offense}_resident"])


def test_census_residential_arm_does_not_apply_landscan_ambient_blind_gate():
    surface = _finalized_test_surface(census_residential=True)
    cell = surface.loc["010010000011"]
    for offense in ("murder", "rape", "robbery", "aggravated_assault", "larceny"):
        assert bool(cell[f"footprint_ambient_exposure_missing_{offense}"]) is False
        assert pd.notna(cell[f"index_{offense}_primary"])


def test_ambient_lift_or_an_ordinary_rate_or_a_minority_share_keeps_the_index():
    surface = _finalized_test_surface()
    for bg in ("010010000013", "010010000014"):
        cell = surface.loc[bg]
        assert bool(cell["footprint_ambient_exposure_missing_larceny"]) is False, bg
        assert cell["estimate_mode_larceny"] == "count_derived", bg
        assert not pd.isna(cell["index_larceny_primary"]), bg
    lifted = surface.loc["010010000012"]
    assert bool(lifted["footprint_ambient_exposure_missing_larceny"]) is False
    assert bool(lifted["transient_exposure_likely_larceny"]) is True
    assert lifted["estimate_mode_larceny"] == "count_derived"
    assert not pd.isna(lifted["index_larceny_primary"])


def test_a_count_the_interval_cannot_separate_cannot_carry_the_ambient_blind_claim():
    """Poisson lower bound. On the point count the rule fired on 66 murder and 38 rape cells
    whose median flagged count was 0.27 and 2.30: arithmetically above 3x national, evidence of
    nothing. Here the POINT ratio clears the threshold and the lower-bound ratio does not."""
    from crimerisk.allocation import _poisson_count_interval

    surface = _finalized_test_surface()
    cell = surface.loc["010010000017"]
    assert cell["footprint_derived_count_share_larceny"] == pytest.approx(1.0)
    assert float(cell["exposure_proxy_2024"]) <= float(cell["population_2024"])
    point_ratio = (
        1e5 * float(cell["expected_count_larceny"]) / float(cell["resident_secondary_denominator"])
    ) / float(cell["resident_national_rate_per_100k_larceny"])
    assert point_ratio >= AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO
    lower, _upper = _poisson_count_interval(pd.Series([float(cell["expected_count_larceny"])]))
    lower_ratio = (
        1e5 * float(lower.iloc[0]) / float(cell["resident_secondary_denominator"])
    ) / float(cell["resident_national_rate_per_100k_larceny"])
    assert lower_ratio < AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO
    assert bool(cell["footprint_ambient_exposure_missing_larceny"]) is False
    assert cell["estimate_mode_larceny"] == "count_derived"


def test_footprint_share_floor_and_rate_ratio_are_the_declared_constants():
    surface = _finalized_test_surface()
    minority = surface.loc["010010000014"]
    assert minority["footprint_derived_count_share_larceny"] <= FOOTPRINT_DERIVED_MASS_SHARE_FLOOR
    blind = surface.loc["010010000011"]
    national = float(blind["resident_national_rate_per_100k_larceny"])
    # Measured on the count's exact-Poisson LOWER bound, not the point count.
    from crimerisk.allocation import _poisson_count_interval

    lower, _upper = _poisson_count_interval(pd.Series([float(blind["expected_count_larceny"])]))
    implied = 1e5 * float(lower.iloc[0]) / float(blind["resident_secondary_denominator"])
    assert implied / national >= AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO


def test_denominator_reason_now_carries_non_residential_on_both_arms():
    surface = _finalized_test_surface()
    cell = surface.loc["010010000015"]
    assert bool(cell["non_residential_flag"]) is True
    assert cell["estimate_mode_larceny"] == "non_residential"
    assert cell["denominator_reason_larceny"] == "non_residential"
    assert cell["resident_denominator_reason_larceny"] == "non_residential"
    assert pd.isna(cell["index_larceny_primary"])
    assert cell["expected_count_larceny"] == pytest.approx(5.0)


def test_transient_warning_does_not_hide_valid_rates():
    surface = _finalized_test_surface()
    cell = surface.loc["010010000016"]
    ratio = float(cell["transient_exposure_daytime_to_resident_ratio"])
    assert ratio == pytest.approx(10.0)
    assert ratio >= TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO
    assert bool(cell["transient_exposure_likely_larceny"]) is True
    assert cell["estimate_mode_larceny"] == "count_derived"
    assert not pd.isna(cell["index_larceny_primary"])
    assert not pd.isna(cell["rate_larceny_primary"])
    assert not pd.isna(cell["index_larceny_resident"])
    assert not pd.isna(cell["rate_larceny_resident"])
    assert cell["expected_count_larceny"] == pytest.approx(30.0)
    assert cell["crime_density_larceny"] == pytest.approx(30.0)
    # This fixture's MVT denominator independently invalidates the full composite,
    # but its personal composite remains available and the advisory flag changes no arm.
    assert not pd.isna(cell["index_personal_part1_resident"])


def test_transient_guard_does_not_need_the_dead_households_term():
    """A non-residential cell publishes no index, so the flag stays False without the term."""
    surface = _finalized_test_surface()
    cell = surface.loc["010010000015"]
    assert bool(cell["non_residential_flag"]) is True
    assert bool(cell["transient_exposure_likely_larceny"]) is False


def _envelope_component(
    *,
    bg: str,
    jurisdiction: str,
    offense: str,
    count: float,
    denominator: float,
    active: bool = False,
) -> dict[str, object]:
    return {
        "state_fips": bg[:2],
        "bg_id": bg,
        "tract_id": bg[:11],
        "jurisdiction_id": jurisdiction,
        "jurisdiction_type": "municipal",
        "offense": offense,
        "component_count": count,
        "component_share": count / 100.0,
        "city_incident_posterior_active": active,
        "primary_denominator_raw": denominator,
        "households_total": 100.0,
    }


def test_model_only_robbery_envelope_caps_and_conserves_source_total():
    rows = [
        _envelope_component(
            bg="010010001001", jurisdiction="J1", offense="robbery",
            count=90.0, denominator=100.0,
        ),
        _envelope_component(
            bg="010010001002", jurisdiction="J1", offense="robbery",
            count=5.0, denominator=1000.0,
        ),
        _envelope_component(
            bg="010010001003", jurisdiction="J1", offense="robbery",
            count=5.0, denominator=1000.0,
        ),
    ]
    out = _apply_model_only_allocation_envelopes(pd.DataFrame(rows))
    assert out["component_count"].sum() == pytest.approx(100.0, abs=1e-12)
    counts = out.set_index("bg_id")["component_count"]
    rates = counts / pd.Series(
        {"010010001001": 100.0, "010010001002": 1000.0, "010010001003": 1000.0}
    )
    ratio = rates / rates.median()
    assert ratio.max() <= MODEL_ONLY_ROBBERY_BG_RATE_RATIO_CAP + 1e-12
    assert out["allocation_envelope_clipped_source"].sum() == 1
    assert out["allocation_envelope_redistribution_recipient"].sum() == 2


def test_robbery_envelope_conserves_canonical_overlap_and_city_controls_separately():
    rows = [
        _envelope_component(
            bg="550010001001", jurisdiction="55:municipal:place:5548000",
            offense="robbery", count=90.0, denominator=100.0,
        ),
        _envelope_component(
            bg="550010001002", jurisdiction="55:municipal:place:5548000",
            offense="robbery", count=5.0, denominator=1000.0,
        ),
        {
            **_envelope_component(
                bg="550010001003", jurisdiction="55:municipal:place:5548000",
                offense="robbery", count=5.0, denominator=1000.0,
            ),
            "jurisdiction_type": "localized_overlap_place_layer",
        },
    ]
    out = _apply_model_only_allocation_envelopes(pd.DataFrame(rows))
    municipal = out["jurisdiction_type"].eq("municipal")
    overlap = out["jurisdiction_type"].str.contains("overlap")
    assert out.loc[municipal, "component_count"].sum() == pytest.approx(95.0, abs=1e-12)
    assert out.loc[overlap, "component_count"].sum() == pytest.approx(5.0, abs=1e-12)


def test_murder_envelope_operates_at_tract_support_and_direct_rows_are_untouched():
    rows = [
        _envelope_component(
            bg="010010001001", jurisdiction="J1", offense="murder",
            count=90.0, denominator=100.0,
        ),
        _envelope_component(
            bg="010010001002", jurisdiction="J1", offense="murder",
            count=9.0, denominator=900.0,
        ),
        _envelope_component(
            bg="010010002001", jurisdiction="J1", offense="murder",
            count=0.5, denominator=1000.0,
        ),
        _envelope_component(
            bg="010010003001", jurisdiction="J1", offense="murder",
            count=0.5, denominator=1000.0,
        ),
        _envelope_component(
            bg="010010004001", jurisdiction="J2", offense="robbery",
            count=100.0, denominator=100.0, active=True,
        ),
    ]
    before = pd.DataFrame(rows)
    out = _apply_model_only_allocation_envelopes(before)
    murder = out[out["offense"].eq("murder")]
    assert murder["component_count"].sum() == pytest.approx(100.0, abs=1e-12)
    tract = murder.groupby("tract_id")["component_count"].sum()
    denom = pd.Series(
        {"01001000100": 1000.0, "01001000200": 1000.0, "01001000300": 1000.0}
    )
    ratio = (tract / denom) / (tract / denom).median()
    assert ratio.max() <= MODEL_ONLY_MURDER_TRACT_RATE_RATIO_CAP + 1e-12
    direct_before = before.loc[before["city_incident_posterior_active"], "component_count"].iloc[0]
    direct_after = out.loc[out["city_incident_posterior_active"], "component_count"].iloc[0]
    assert direct_after == direct_before


def test_rare_offense_tract_support_clears_the_transient_flag_it_can_no_longer_justify():
    from crimerisk.allocation import apply_rare_offense_tract_support

    surface = _finalized_test_surface().reset_index()
    tract_cols = ["tract_id", *[f"index_{offense}_primary" for offense in RARE_OFFENSE_TRACT_SUPPORT]]
    tract = surface[tract_cols].drop_duplicates("tract_id").copy()
    for offense in RARE_OFFENSE_TRACT_SUPPORT:
        surface[f"transient_exposure_likely_{offense}"] = True
    out = apply_rare_offense_tract_support(surface, tract)
    for offense in RARE_OFFENSE_TRACT_SUPPORT:
        assert not out[f"transient_exposure_likely_{offense}"].any()
        assert out[f"index_{offense}_primary"].isna().all()
        assert not out[f"index_publishable_{offense}"].any()
        assert not out[f"primary_index_publishable_{offense}"].any()
        assert not out[f"index_{offense}_resident_publishable"].any()
        assert out[f"primary_index_suppressed_{offense}"].all()
        assert out[f"index_{offense}_resident_suppressed"].all()


def test_rare_offense_tract_support_rebuilds_resident_aggregates_from_final_publishability():
    """A rare-only pre-policy suppression cannot strand a null resident aggregate."""
    from crimerisk.allocation import apply_rare_offense_tract_support

    surface = _finalized_test_surface().reset_index()
    target = surface["block_group_geoid"].eq("010010000013")
    assert target.sum() == 1
    assert all(
        bool(surface.loc[target, f"index_{offense}_resident_publishable"].iloc[0])
        for offense in OFFENSES_7
    )

    # Reproduce the standard-build edge: rape alone crosses a pre-policy suppression rule, so
    # the resident total and personal aggregate were originally computed as null on this row.
    surface.loc[target, "index_rape_resident"] = np.nan
    surface.loc[target, "index_rape_resident_publishable"] = False
    surface.loc[target, "index_total_part1_resident"] = np.nan
    surface.loc[target, "index_personal_part1_resident"] = np.nan

    tract_cols = ["tract_id", *[f"index_{offense}_primary" for offense in RARE_OFFENSE_TRACT_SUPPORT]]
    tract = surface[tract_cols].drop_duplicates("tract_id").copy()
    out = apply_rare_offense_tract_support(surface, tract)

    denominator = pd.to_numeric(out["resident_secondary_denominator"], errors="coerce").fillna(0.0)
    for field, offenses in (
        ("index_total_part1_resident", list(OFFENSES_7)),
        ("index_personal_part1_resident", ["murder", "rape", "robbery", "aggravated_assault"]),
    ):
        counts = sum(
            pd.to_numeric(out[f"expected_count_{offense}"], errors="coerce").fillna(0.0)
            for offense in offenses
        )
        publishable = denominator.gt(0.0)
        for offense in offenses:
            if offense not in RARE_OFFENSE_TRACT_SUPPORT:
                # Contract: a component suppressed only for cell-level denominator
                # invalidity keeps its real count and does not veto the aggregate.
                mode_exempt = (
                    out[f"estimate_mode_{offense}"]
                    .astype("string")
                    .isin({"insufficient_exposure", "vehicle_denominator_invalid"})
                    .fillna(False)
                    if f"estimate_mode_{offense}" in out.columns
                    else pd.Series(False, index=out.index)
                )
                publishable &= out[f"index_{offense}_resident_publishable"].astype(bool) | mode_exempt
        national_rate = 100_000.0 * counts.loc[publishable].sum() / denominator.loc[publishable].sum()
        expected = pd.Series(np.nan, index=out.index, dtype=float)
        expected.loc[publishable] = (
            100.0
            * (100_000.0 * counts.loc[publishable] / denominator.loc[publishable])
            / national_rate
        )
        pd.testing.assert_series_equal(out[field], expected, check_names=False, rtol=0.0, atol=1e-12)
        assert out.loc[target, field].notna().all()


# -------------------------------------------------------------- one city-key vocabulary


def test_city_key_aliases_resolve_to_one_canonical_key():
    from crimerisk.city_feed_quarantine import canonical_city_key, resolve_texture_key

    assert canonical_city_key("st_louis") == "st_louis_mo"
    assert canonical_city_key("st_louis_mo") == "st_louis_mo"
    assert canonical_city_key("oakland_california") == "oakland_ca"
    assert canonical_city_key("houston_texas") == "houston_tx"
    assert canonical_city_key("milwaukee_wi") == "milwaukee"
    assert resolve_texture_key("St. Louis") == "st_louis_mo"
    assert resolve_texture_key("Washington") == "washington_dc"


def test_unknown_city_key_fails_closed_rather_than_being_mangled():
    from crimerisk.city_feed_quarantine import canonical_city_key

    with pytest.raises(ValueError, match="Unknown city key"):
        canonical_city_key("springfield_confusion")
    # The permissive form exists only for read-only queries over an open-ended city list.
    assert canonical_city_key("springfield_confusion", strict=False) == "springfield_confusion"


def test_live_configs_speak_the_canonical_vocabulary():
    from crimerisk.city_feed_quarantine import (
        CANONICAL_CITY_KEYS,
        load_coordinate_quarantine,
        load_texture_policy,
    )

    quarantine = load_coordinate_quarantine()
    assert set(quarantine["city_key"]) <= CANONICAL_CITY_KEYS
    policy = load_texture_policy()
    assert set(policy["city_key"]) - {"*"} <= CANONICAL_CITY_KEYS
    # The texture policy's St. Louis row and the pipeline's St. Louis key are now the same key.
    assert "st_louis_mo" in set(policy["city_key"])


def test_quarantine_wiring_witness_fails_on_an_undeclared_city():
    from crimerisk.city_feed_quarantine import (
        assert_quarantine_wired,
        record_quarantine_applied,
        record_quarantine_not_applicable,
        reset_quarantine_application_log,
    )

    reset_quarantine_application_log()
    record_quarantine_applied("boston")
    record_quarantine_not_applicable("austin", reason="feed_carries_no_point_coordinates")
    log = assert_quarantine_wired(["boston", "austin"])
    assert log["boston"] == "applied"
    assert log["austin"].startswith("not_applicable")
    with pytest.raises(RuntimeError, match="not wired into every enabled city builder"):
        assert_quarantine_wired(["boston", "austin", "denver"])
    reset_quarantine_application_log()


def test_every_production_city_builder_routes_through_the_shared_quarantine_helper():
    """The b3 finding was 4 of 13. Checked structurally so a new builder cannot skip it."""
    import inspect

    from crimerisk import city_incidents
    from crimerisk.city_feed_quarantine import PRODUCTION_CITY_KEYS

    source = inspect.getsource(city_incidents)
    assert "_drop_quarantined_coordinates(" in source
    for city_key in PRODUCTION_CITY_KEYS:
        assert f'_drop_quarantined_coordinates(geocoded, city_key="{city_key}"' in source or (
            f'_drop_quarantined_coordinates(raw, city_key="{city_key}"' in source
        ), city_key


def test_washington_dc_loader_fetches_and_preserves_every_requested_year(tmp_path, monkeypatch):
    from crimerisk import city_incidents

    raw_columns = {
        "OBJECTID": ["1", "2"],
        "CCN": ["a", "b"],
        "REPORT_DAT": ["2020-01-01", "2020-01-02"],
        "START_DATE": ["2020-01-01", "2020-01-02"],
        "END_DATE": ["2020-01-01", "2020-01-02"],
        "OFFENSE": ["HOMICIDE", "ROBBERY"],
        "METHOD": ["GUN", "OTHERS"],
        "BLOCK": ["1 TEST ST", "2 TEST ST"],
        "BLOCK_GROUP": ["000100 1", "000100 1"],
        "CENSUS_TRACT": ["000100", "000100"],
        "LATITUDE": ["38.9", "38.9"],
        "LONGITUDE": ["-77.0", "-77.0"],
    }
    source_paths = {}
    for year in (2018, 2019, 2020):
        frame = pd.DataFrame(raw_columns)
        frame["REPORT_DAT"] = f"{year}-01-01"
        path = tmp_path / f"source_{year}.csv"
        frame.to_csv(path, index=False)
        source_paths[year] = path

    requested_years = []

    def fake_download(url, destination, *, force_download=False):
        year = int(str(url).split("crime-incidents-in-")[1].split(".")[0])
        requested_years.append(year)
        return source_paths[year]

    monkeypatch.setattr(city_incidents, "_download_file", fake_download)
    out = city_incidents._load_washington_dc_raw_snapshot(
        raw_dir=tmp_path / "raw",
        start_year=2018,
        end_year=2020,
    )

    assert requested_years == [2018, 2019, 2020]
    assert out.groupby("year").size().to_dict() == {2018: 2, 2019: 2, 2020: 2}
    assert len(out) == 6  # OBJECTID values may repeat in separate annual snapshots.


def test_st_louis_loader_retains_latest_verified_year_for_newer_edition(tmp_path, monkeypatch):
    from crimerisk import city_incidents

    raw_dir = tmp_path / "raw"
    cache_path = (
        raw_dir
        / "parsed"
        / f"st_louis_minimal_snapshot_{city_incidents.ST_LOUIS_LIVE_START_YEAR}_"
        f"{city_incidents.ST_LOUIS_LIVE_END_YEAR}.parquet"
    )
    cache_path.parent.mkdir(parents=True)
    cached = pd.DataFrame(
        {
            column: ["2024-01-01" if column == "IncidentDate" else "x"]
            for column in city_incidents.ST_LOUIS_MINIMAL_COLUMNS
        }
    )
    cached["source_month"] = 1
    cached["source_file_url"] = "https://example.invalid/verified-2024.csv"
    cached.to_parquet(cache_path, index=False)

    monkeypatch.setattr(
        city_incidents,
        "_download_st_louis_file",
        lambda *args, **kwargs: pytest.fail("verified cached year should be reused"),
    )
    out = city_incidents._load_st_louis_raw_snapshot(
        raw_dir=raw_dir,
        start_year=2018,
        end_year=2025,
    )

    assert len(out) == 1
    assert out.loc[0, "IncidentDate"] == "2024-01-01"


def test_washington_dc_history_expansion_is_murder_only():
    frame = pd.DataFrame(
        {
            "year": [2023, 2024, 2023, 2024],
            "offense": ["murder", "murder", "robbery", "robbery"],
        }
    )

    out = _washington_dc_murder_history_only(frame, target_year=2024)

    assert out[["year", "offense"]].to_records(index=False).tolist() == [
        (2023, "murder"),
        (2024, "murder"),
        (2024, "robbery"),
    ]


def test_murder_incident_posterior_updates_tracts_not_point_block_groups():
    frame = pd.DataFrame(
        {
            "jurisdiction_id": ["J", "J", "J", "J"],
            "state_fips": ["11"] * 4,
            "offense": ["murder"] * 4,
            "tract_id": ["11001000100", "11001000100", "11001000200", "11001000200"],
        }
    )
    prior = pd.Series([0.20, 0.30, 0.10, 0.40])
    # The incident happens in the first BG, but only its parent tract may receive evidence.
    direct = pd.Series([1.0, 0.0, 0.0, 0.0])
    out = _tract_incident_posterior(
        frame,
        direct_count=direct,
        prior_share=prior,
        prior_incidents=MURDER_TRACT_POSTERIOR_PRIOR_INCIDENTS,
    )

    expected_first_tract = (
        1.0 + MURDER_TRACT_POSTERIOR_PRIOR_INCIDENTS * 0.50
    ) / (1.0 + MURDER_TRACT_POSTERIOR_PRIOR_INCIDENTS)
    assert out["posterior_share"].sum() == pytest.approx(1.0)
    assert out.loc[[0, 1], "posterior_share"].sum() == pytest.approx(expected_first_tract)
    # Within the tract, the 40/60 prior split is preserved instead of learning the point BG.
    assert out.loc[0, "posterior_share"] / out.loc[1, "posterior_share"] == pytest.approx(2.0 / 3.0)
    assert out.loc[0, "direct_share"] / out.loc[1, "direct_share"] == pytest.approx(2.0 / 3.0)


# ----------------------------------------------------- compositional metadata alignment


def test_model_training_gate_rejects_a_lane_carried_quality_tier():
    from crimerisk.model_surface import _assert_quality_tier_is_recomputed_from_coverage

    good = pd.DataFrame(
        {
            "jurisdiction_id": ["J1", "J2", "J3"],
            "mean_months_reported_preferred": [12.0, 7.0, np.nan],
            "quality_tier_preferred": ["high", "low", "unknown"],
        }
    )
    _assert_quality_tier_is_recomputed_from_coverage(good, offense="larceny")

    bad = good.copy()
    # A tier carried over from a lane whose months do not describe the count it gates on.
    bad.loc[1, "quality_tier_preferred"] = "high"
    with pytest.raises(ValueError, match="not the tier re-derived"):
        _assert_quality_tier_is_recomputed_from_coverage(bad, offense="larceny")
