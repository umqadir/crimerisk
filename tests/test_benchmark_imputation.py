"""Benchmark-constrained imputation for silent-agency territory (Class A, v20).

Three things are load-bearing and are asserted here: the state x offense accounting
identity (imputed mass can never exceed the FBI benchmark residual, and a locked total
above the benchmark produces a conflict record rather than negative mass), the exact
municipal/county partition of the residual, and the survival of each county sub-target
into the allocation group targets.
"""

from pathlib import Path

import pandas as pd
import pytest

from crimerisk.allocation import (
    _assert_displaced_imputed_county_targets_survive,
    _assert_imputed_county_targets_survive,
    _assert_no_negative_group_targets,
    _partition_imputed_county_targets_by_support,
)
from crimerisk.benchmark_imputation import (
    BENCHMARK_IMPUTATION_SOURCE,
    COUNTY_UNIT_KIND,
    MUNICIPAL_UNIT_KIND,
    BenchmarkImputation,
    _assert_lane_partition,
    _urbanicity,
    apply_benchmark_imputation_to_controls,
    assert_benchmark_imputation_invariants,
    build_unit_exposure,
    county_remainder_imputed_targets,
    fit_pooled_rates,
    predict_pooled_rate,
)
from crimerisk.reference import (
    COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES,
    canonicalize_agency_county_fips,
)
from crimerisk.confidence import (
    MODELED_SOURCE_MODE,
    _append_confidence_tiers,
    _benchmark_imputed_share_wide,
    _normalize_component_audit,
)



class _NullPaths:
    """Paths with no roster/LEAIC/name inputs, so the canonicalization exercises only
    its deterministic steps (sentinel clearing and the retired-GEOID remap)."""

    repo_root = Path("/nonexistent")
    data_dir = Path("/nonexistent/data")


# --- the pooled rate model --------------------------------------------------


def _rate_panel() -> pd.DataFrame:
    """Two states, one lane, one urbanicity band. State 01 is dense in data and high
    rate; state 02 is thin and would otherwise take its own noisy rate."""
    return pd.DataFrame(
        {
            "state_fips": ["01", "01", "02"],
            "lane": [COUNTY_UNIT_KIND] * 3,
            "urbanicity": ["rural"] * 3,
            "exposure_population": [100_000.0, 100_000.0, 100.0],
            "burglary": [1_000.0, 1_000.0, 10.0],
        }
    )


def test_pooled_rate_shrinks_a_thin_cell_toward_the_national_rate():
    panel = _rate_panel()
    # National rate is ~1,005/100k; state 02's own rate is 10,000/100k on 100 people.
    weak = fit_pooled_rates(panel, offense="burglary", pooling_constant=1e5)
    strong_own_data = fit_pooled_rates(panel, offense="burglary", pooling_constant=1e0)
    state_02 = ("02", COUNTY_UNIT_KIND, "rural")
    assert weak["cell"][state_02] < strong_own_data["cell"][state_02]
    # With heavy pooling the thin cell is pulled essentially onto the national rate.
    assert weak["cell"][state_02] == pytest.approx(weak["national"], rel=0.05)
    # The data-rich state barely moves either way.
    state_01 = ("01", COUNTY_UNIT_KIND, "rural")
    assert weak["cell"][state_01] == pytest.approx(strong_own_data["cell"][state_01], rel=0.15)


def test_pooled_rate_prediction_falls_back_through_state_then_national():
    panel = _rate_panel()
    rates = fit_pooled_rates(panel, offense="burglary", pooling_constant=1e3)
    unseen = pd.DataFrame(
        {
            # An urbanicity band never observed in state 01 -> state rate;
            # a state never observed at all -> national rate.
            "state_fips": ["01", "99"],
            "lane": [COUNTY_UNIT_KIND, COUNTY_UNIT_KIND],
            "urbanicity": ["urban", "rural"],
        }
    )
    predicted = predict_pooled_rate(unseen, rates)
    assert predicted[0] == pytest.approx(rates["state"]["01"])
    assert predicted[1] == pytest.approx(rates["national"])


def test_urbanicity_bands_come_from_the_unit_density():
    band = _urbanicity(
        pd.Series([5_000.0, 500.0, 50.0, 10.0]),
        pd.Series([1.0, 1.0, 1.0, 0.0]),
    )
    assert list(band) == ["urban", "suburban", "rural", "rural"]


# --- the partition ----------------------------------------------------------


def _bg_crosswalk() -> pd.DataFrame:
    """One block group split between a municipal jurisdiction and the county remainder,
    plus one entirely-remainder block group in the same county."""
    return pd.DataFrame(
        {
            "state_fips": ["01", "01", "01"],
            "block_group_geoid": ["010010001001", "010010001001", "010010001002"],
            "jurisdiction_id": [
                "01:municipal:place:0100100",
                "01:state_nonmunicipal_remainder",
                "01:state_nonmunicipal_remainder",
            ],
            "jurisdiction_type": [
                "municipal",
                "state_nonmunicipal_remainder",
                "state_nonmunicipal_remainder",
            ],
            "aland20": [2_589_988.110336, 2_589_988.110336, 2_589_988.110336],
            "pop20": [800.0, 200.0, 400.0],
            "allocation_share": [0.8, 0.2, 1.0],
            "county_geoid": ["01001", "01001", "01001"],
        }
    )


def test_unit_exposure_partitions_population_between_the_two_lanes():
    units = build_unit_exposure(_bg_crosswalk())
    by_kind = units.set_index("unit_kind")["exposure_population"]
    assert by_kind[MUNICIPAL_UNIT_KIND] == pytest.approx(800.0)
    assert by_kind[COUNTY_UNIT_KIND] == pytest.approx(600.0)
    # Exactly the block groups' population, counted once.
    assert units["exposure_population"].sum() == pytest.approx(1_400.0)
    county = units[units["unit_kind"].eq(COUNTY_UNIT_KIND)].iloc[0]
    assert county["unit_id"] == "01:state_nonmunicipal_remainder:county:01001"


def test_lane_partition_failure_is_fail_closed():
    broken = _bg_crosswalk()
    broken.loc[0, "allocation_share"] = 0.5
    with pytest.raises(ValueError, match="do not partition"):
        _assert_lane_partition(broken)


# --- the accounting identity ------------------------------------------------


def _imputation(
    *,
    imputed: float,
    residual: float,
    conflict_kind: str = "benchmark_capped_below_model_pool",
    unit_kind: str = COUNTY_UNIT_KIND,
    unit_id: str = "01:state_nonmunicipal_remainder:county:01001",
) -> BenchmarkImputation:
    units = pd.DataFrame(
        {
            "year": [2024],
            "state_fips": ["01"],
            "state_abbr": ["AL"],
            "unit_kind": [unit_kind],
            "unit_id": [unit_id],
            "county_geoid": ["01001"],
            "offense": ["burglary"],
            "exposure_population": [1_000.0],
            "land_area_sq_mi": [10.0],
            "urbanicity": ["rural"],
            "pooled_rate": [0.01],
            "modeled_expected_count": [10.0],
            "benchmark_scale": [imputed / 10.0 if imputed else 0.0],
            "imputed_count": [imputed],
            "imputation_source": [BENCHMARK_IMPUTATION_SOURCE],
            "silent_agency_count": [1],
            "silent_agency_oris": ["AL0000100"],
        }
    )
    identity = pd.DataFrame(
        {
            "year": [2024],
            "state_fips": ["01"],
            "state_abbr": ["AL"],
            "offense": ["burglary"],
            "locked_observed_total": [100.0],
            "fbi_cde_estimated_total": [100.0 + residual],
            "benchmark_residual": [residual],
            "modeled_pool": [10.0],
            "benchmark_scale": [imputed / 10.0 if imputed else 0.0],
            "imputed_total": [imputed],
            "unused_benchmark_headroom": [max(0.0, residual - imputed)],
            "unfilled_modeled_pool": [max(0.0, 10.0 - imputed)],
            "silent_unit_count": [1],
            "silent_unit_population": [1_000.0],
            "conflict_kind": [conflict_kind],
        }
    )
    return BenchmarkImputation(units=units, state_identity=identity, validation=pd.DataFrame())


def test_imputed_mass_within_the_benchmark_residual_passes():
    assert_benchmark_imputation_invariants(_imputation(imputed=4.0, residual=4.0))


def test_imputed_mass_above_the_benchmark_residual_fails_closed():
    with pytest.raises(ValueError, match="exceeds the state x offense benchmark residual"):
        assert_benchmark_imputation_invariants(_imputation(imputed=9.0, residual=4.0))


def test_a_benchmark_conflict_cell_may_not_carry_imputed_mass():
    conflicted = _imputation(
        imputed=3.0, residual=0.0, conflict_kind="locked_exceeds_benchmark"
    )
    with pytest.raises(ValueError, match="benchmark-conflict"):
        assert_benchmark_imputation_invariants(conflicted)


def test_a_benchmark_conflict_cell_with_zero_imputation_is_the_expected_shape():
    # Locked observations above the benchmark are preserved and the residual is zero --
    # never negative imputation.
    conflicted = _imputation(
        imputed=0.0, residual=0.0, conflict_kind="locked_exceeds_benchmark"
    )
    conflicted.units.drop(conflicted.units.index, inplace=True)
    assert_benchmark_imputation_invariants(conflicted)
    assert conflicted.imputed_total == 0.0


def test_a_unit_may_not_be_claimed_by_both_lanes():
    county = _imputation(imputed=1.0, residual=5.0)
    municipal = _imputation(imputed=1.0, residual=5.0, unit_kind=MUNICIPAL_UNIT_KIND)
    # Different offenses, so only the lane collision fires (not the duplicate-row rule).
    municipal.units["offense"] = "larceny"
    identity = pd.concat(
        [county.state_identity, county.state_identity.assign(offense="larceny")],
        ignore_index=True,
    )
    both = BenchmarkImputation(
        units=pd.concat([county.units, municipal.units], ignore_index=True),
        state_identity=identity,
        validation=pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="claimed by both"):
        assert_benchmark_imputation_invariants(both)


# --- application to controls and to the group targets -----------------------


def _controls() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "jurisdiction_id": [
                "01:municipal:place:0100100",
                "01:municipal:place:0100200",
                "01:state_nonmunicipal_remainder",
            ],
            "jurisdiction_type": [
                "municipal",
                "municipal",
                "state_nonmunicipal_remainder",
            ],
            "state_fips": ["01", "01", "01"],
            "offense": ["burglary", "burglary", "burglary"],
            "adjusted_count_ags_core": [0.0, 250.0, 40.0],
            "estimated_count_ags_core": [0.0, 250.0, 40.0],
            "adjustment_total": [0.0, 0.0, 0.0],
            "estimate_source": ["hist_median", "observed_nibrs_rollup", "observed_srs"],
            "estimate_confidence": ["low", "high", "high"],
            "estimated_from_panel": [True, False, False],
        }
    )


def _units_for_controls() -> pd.DataFrame:
    base = _imputation(imputed=6.0, residual=100.0).units
    municipal = base.copy()
    municipal["unit_kind"] = MUNICIPAL_UNIT_KIND
    municipal["unit_id"] = "01:municipal:place:0100100"
    municipal["county_geoid"] = pd.NA
    municipal["imputed_count"] = 2.0
    return pd.concat([base, municipal], ignore_index=True)


def test_controls_application_credits_the_right_rows_and_leaves_observed_alone():
    out = apply_benchmark_imputation_to_controls(_controls(), units=_units_for_controls())
    by_id = out.set_index("jurisdiction_id")

    # The silent municipal jurisdiction takes its own sub-target...
    assert by_id.loc["01:municipal:place:0100100", "adjusted_count_ags_core"] == pytest.approx(2.0)
    assert by_id.loc["01:municipal:place:0100100", "estimate_source"] == BENCHMARK_IMPUTATION_SOURCE
    assert by_id.loc["01:municipal:place:0100100", "estimate_confidence"] == "low"
    # ...the county sub-target lands on the state remainder pool row (allocation splits
    # it back out per county from the same unit table)...
    assert by_id.loc["01:state_nonmunicipal_remainder", "adjusted_count_ags_core"] == pytest.approx(46.0)
    # ...and a locked observed jurisdiction is untouched, label included.
    assert by_id.loc["01:municipal:place:0100200", "adjusted_count_ags_core"] == pytest.approx(250.0)
    assert by_id.loc["01:municipal:place:0100200", "estimate_source"] == "observed_nibrs_rollup"
    assert not bool(by_id.loc["01:municipal:place:0100200", "benchmark_imputation_applied"])


def test_controls_application_is_a_noop_without_units():
    out = apply_benchmark_imputation_to_controls(
        _controls(), units=pd.DataFrame(columns=["unit_kind"])
    )
    assert out["adjusted_count_ags_core"].tolist() == [0.0, 250.0, 40.0]
    assert not out["benchmark_imputation_applied"].any()


def test_county_sub_targets_are_shaped_for_the_group_target_table():
    targets = county_remainder_imputed_targets(_units_for_controls())
    assert len(targets) == 1
    row = targets.iloc[0]
    assert row["group_id"] == "01:state_nonmunicipal_remainder:county:01001"
    assert row["imputed_target_count"] == pytest.approx(6.0)


def test_fully_displaced_county_sub_target_falls_back_without_losing_mass():
    targets = pd.concat(
        [
            county_remainder_imputed_targets(_units_for_controls()),
            county_remainder_imputed_targets(_units_for_controls()).assign(
                group_id="01:state_nonmunicipal_remainder:county:01003",
                county_geoid="01003",
                imputed_target_count=4.0,
            ),
        ],
        ignore_index=True,
    )
    localized, displaced = _partition_imputed_county_targets_by_support(
        targets,
        {("01", "01001")},
    )
    assert localized["imputed_target_count"].sum() == pytest.approx(6.0)
    assert displaced["imputed_target_count"].sum() == pytest.approx(4.0)
    delivered = pd.DataFrame(
        {
            "state_fips": ["01"],
            "offense": ["burglary"],
            "group_kind": ["residual_remainder"],
            "adjustment_target_count": [4.0],
        }
    )
    _assert_displaced_imputed_county_targets_survive(delivered, displaced)


def test_fully_displaced_county_sub_target_missing_from_residual_fails_closed():
    targets = county_remainder_imputed_targets(_units_for_controls())
    with pytest.raises(ValueError, match="did not survive"):
        _assert_displaced_imputed_county_targets_survive(
            pd.DataFrame(
                columns=[
                    "state_fips",
                    "offense",
                    "group_kind",
                    "adjustment_target_count",
                ]
            ),
            targets,
        )


def test_county_sub_target_absorbed_into_the_state_residual_fails_closed():
    targets = county_remainder_imputed_targets(_units_for_controls())
    # What the delta reconciliation would produce if the county row were dropped: the
    # mass reappears in the state residual group instead of the county's own group.
    smeared = pd.DataFrame(
        {
            "state_fips": ["01"],
            "offense": ["burglary"],
            "group_kind": ["residual_remainder"],
            "group_id": ["01:state_nonmunicipal_remainder:residual"],
            "target_count": [46.0],
        }
    )
    with pytest.raises(ValueError, match="did not survive"):
        _assert_imputed_county_targets_survive(smeared, targets)


def test_imputed_territory_reads_as_imputed_in_the_confidence_layer():
    # Provenance has to reach the published surface: a block group whose mass comes
    # from an imputed jurisdiction must carry the share, and a half-and-half block
    # group must carry half of it.
    units = _units_for_controls()
    component = _normalize_component_audit(
        pd.DataFrame(
            {
                "state_fips": ["01", "01", "01"],
                "bg_id": ["010010001001", "010010001001", "010010001002"],
                "tract_id": ["01001000100"] * 3,
                "jurisdiction_id": [
                    "01:state_nonmunicipal_remainder:county:01001",
                    "01:municipal:place:0100200",
                    "01:municipal:place:0100100",
                ],
                "offense": ["burglary"] * 3,
                "component_count_after": [10.0, 10.0, 5.0],
                "component_share": [0.5, 0.5, 1.0],
                "model_share": [0.5, 0.5, 1.0],
                "city_incident_posterior_active": [False] * 3,
            }
        )
    )
    shares = _benchmark_imputed_share_wide(
        component, units, geo_col="bg_id", out_geo_col="block_group_geoid"
    ).set_index("block_group_geoid")["benchmark_imputed_share_burglary"]
    assert shares["010010001001"] == pytest.approx(0.5)
    assert shares["010010001002"] == pytest.approx(1.0)


def test_confidence_tier_is_forced_low_on_imputed_territory():
    surface = pd.DataFrame(
        {
            "estimate_mode_burglary": ["count_derived", "count_derived"],
            "reliability_tier_burglary": ["high", "high"],
            "source_mode_burglary": [MODELED_SOURCE_MODE, MODELED_SOURCE_MODE],
            "domain_overlap_score_burglary": [1.0, 1.0],
            "benchmark_imputed_share_burglary": [0.0, 1.0],
        }
    )
    out = _append_confidence_tiers(surface)
    assert out["confidence_tier_burglary"].tolist() == ["high", "low"]
    assert "benchmarked_nonreporter_imputation" not in out["confidence_reasons_burglary"].iloc[0]
    assert "benchmarked_nonreporter_imputation" in out["confidence_reasons_burglary"].iloc[1]


def test_only_authoritative_county_placements_can_anchor_a_county():
    # The county-FIPS canonicalization resolves a county NAME for agencies the SRS
    # header never placed. That is enough to say which county's silence an agency
    # speaks to, but not to concentrate its crime into that county's unincorporated
    # remainder: Pontiac PD (MI631900X, roster-filled county) would otherwise put 1,619
    # crimes on Oakland County's 32,376-person remainder.
    assert "srs_agency_header" in COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES
    # The retired-FIPS remap is the SRS header's own placement forwarded to a live
    # GEOID (Shannon SD 46113 -> Oglala Lakota 46102), so it stays anchor-eligible.
    assert "retired_county_remap" in COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES
    assert "fbi_cde_agency_roster" not in COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES
    assert "leaic_crosswalk" not in COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES


def test_canonicalization_records_the_provenance_of_every_county_it_sets():
    master = pd.DataFrame(
        {
            "ori9": ["AA0000100", "SD0560000", "CZ0000500"],
            "state_fips": ["01", "46", "57"],
            # header-placed / retired GEOID / sentinel "no county"
            "county_fips": ["001", "113", "999"],
        }
    )
    out = canonicalize_agency_county_fips(master, paths=_NullPaths())
    by_ori = out.set_index("ori9")
    assert by_ori.loc["AA0000100", "county_fips_source"] == "srs_agency_header"
    assert by_ori.loc["SD0560000", "county_fips"] == "102"
    assert by_ori.loc["SD0560000", "county_fips_source"] == "retired_county_remap"
    # The sentinel is cleared, and a cleared county carries no provenance to anchor on.
    assert pd.isna(by_ori.loc["CZ0000500", "county_fips"])
    assert pd.isna(by_ori.loc["CZ0000500", "county_fips_source"])


def test_a_group_split_that_over_draws_the_control_fails_closed():
    # The exact v20 shape: the control carried the imputed mass, the observed split
    # normalized it across the reporting counties, the county sub-targets added it a
    # second time, and the delta reconciliation parked the -1,188 overdraft in the
    # residual group where the trailing clip(lower=0) would silently re-add it.
    over_drawn = pd.DataFrame(
        {
            "state_fips": ["18", "18"],
            "offense": ["larceny", "larceny"],
            "group_kind": ["county_remainder", "residual_remainder"],
            "group_id": [
                "18:state_nonmunicipal_remainder:county:18007",
                "18:state_nonmunicipal_remainder:residual",
            ],
            "target_count": [8308.02, -1187.98],
        }
    )
    with pytest.raises(ValueError, match="negative"):
        _assert_no_negative_group_targets(over_drawn)


def test_a_balanced_group_split_passes():
    balanced = pd.DataFrame(
        {
            "state_fips": ["18", "18"],
            "offense": ["larceny", "larceny"],
            "group_kind": ["county_remainder", "residual_remainder"],
            "group_id": [
                "18:state_nonmunicipal_remainder:county:18007",
                "18:state_nonmunicipal_remainder:residual",
            ],
            # Floating-point dust below the tolerance is hygiene, not an overdraft.
            "target_count": [7120.05, -1e-12],
        }
    )
    _assert_no_negative_group_targets(balanced)


def test_county_sub_target_that_survives_passes():
    targets = county_remainder_imputed_targets(_units_for_controls())
    intact = pd.DataFrame(
        {
            "state_fips": ["01", "01"],
            "offense": ["burglary", "burglary"],
            "group_kind": ["county_remainder", "residual_remainder"],
            "group_id": [
                "01:state_nonmunicipal_remainder:county:01001",
                "01:state_nonmunicipal_remainder:residual",
            ],
            "target_count": [6.0, 40.0],
        }
    )
    _assert_imputed_county_targets_survive(intact, targets)
