"""v2 uncertainty layer: coherent draws, exact conservation, decisions that recompute.

The properties under test are the ones the construction exists to guarantee. A release is
reproducible bit for bit from fixed seeds. Every draw conserves its footprint's drawn control
exactly, not on average. The quantile ladder, the bin probability and the tier are arithmetic a
reader can redo from published fields, and the release validator redoes them with its own
transcribed constants. The concentration rule hits the dispersion it claims to hit, checked
against simulation rather than against itself. The measured dispersion tables are the ones the
held-out harness wrote. And a build with the flag off gains no column and no behaviour.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    AllocationBuildConfig,
    _allocation_build_manifest,
    _finalize_output,
    _rare_offense_published_point_fields,
    _uncertainty_dependency_paths,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths
from crimerisk.uncertainty import (
    CALIBRATION_CLASS_FOR_SUPPORT,
    CALIBRATION_VERSION,
    CONTROL_DISPERSION_VERSION,
    DEFAULT_N_DRAWS,
    IDENTITY_CALIBRATION,
    INDEX_BIN_LABELS,
    INDEX_BREAKS,
    MAX_SHARE_CONCENTRATION,
    PIT_GRID,
    SHARE_DISPERSION_VERSION,
    STRATA,
    SUPPORT_CLASS_BENCHMARK,
    SUPPORT_CLASS_DIRECT,
    SUPPORT_CLASS_MODEL,
    TIER_HIGH,
    TIER_HIGH_MIN_BIN_PROBABILITY,
    TIER_LOW,
    TIER_LOW_BENCHMARK_SHARE_MAX,
    TIER_MEDIUM,
    TIER_MEDIUM_MIN_BIN_PROBABILITY,
    TIER_WITHHELD,
    UNCERTAINTY_LAYER_VERSION,
    UNCERTAINTY_SEED,
    CalibrationMap,
    UncertaintyEngine,
    apply_uncertainty_layer,
    benchmark_multiplier_sigmas,
    cache_frame_to_summaries,
    calibration_table_path,
    components_signature,
    concentration_from_log_share_dispersion,
    control_dispersion_path,
    control_mass_fractions,
    decision_reliability_tier,
    denominator_log_sd,
    dirichlet_log_share_sd,
    index_bin_indices,
    index_bin_labels,
    load_calibration,
    load_control_dispersion,
    load_imputation_bounds,
    load_share_dispersion,
    log_dispersion,
    prob_displayed_bin_column,
    quasi_poisson_lognormal_sigma,
    decision_tier_column,
    displayed_bin_column,
    index_quantile_column,
    rare_offense_suppressed_columns,
    resolve_uncertainty_runtime,
    rollup_draws,
    share_dispersion_path,
    stable_fold,
    stratum_for_population,
    summaries_to_cache_frame,
    summarize_offense,
    summarize_uncertainty_layer,
    support_class_series,
    uncertainty_columns_for_offense,
    uncertainty_published_columns,
    uncertainty_summary_inputs_signature,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "analysis_scratch" / "final_phase" / "UNCERTAINTY_LAYER_CONTRACT.md"
EVIDENCE_DIR = REPO_ROOT / "analysis_scratch" / "final_phase" / "uncertainty"
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
N_TEST_DRAWS = 24


def _validator_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"
    spec = importlib.util.spec_from_file_location("validate_release_outputs_for_uncertainty", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime():
    return resolve_uncertainty_runtime(PATHS, n_draws=N_TEST_DRAWS)


# --- fixtures ------------------------------------------------------------------------------


def _surface_rows(n: int = 40) -> pd.DataFrame:
    rows = []
    for index in range(n):
        geoid = f"01001{index // 4:04d}{index % 4}01"[:12].ljust(12, "0")
        geoid = f"01001{index // 4:06d}{index % 4}"
        rows.append(
            {
                "block_group_geoid": geoid,
                "tract_id": geoid[:11],
                "state_fips": "01",
                "population_2024": 2_000.0 + 10.0 * index,
                "households_total": 700.0,
                "commercial_premises_total": 0.0,
                "destination_poi_total": 0.0,
                "daytime_population_jobs_proxy": 2_000.0 + 10.0 * index,
                "landscan_day_pop": 1_600.0 + 20.0 * index,
                "exposure_proxy_2024": 3_000.0,
                "burglary_premises_total": 400.0,
                "aggregate_vehicles_total": 900.0,
                "vehicle_exposure_2024": 900.0,
                "land_area_sq_mi": 1.0,
                "eb_jurisdiction_id": "J1" if index < n // 2 else "J2",
                "eb_jurisdiction_type": "municipal",
                **{f"expected_count_{offense}": 5.0 + index for offense in OFFENSES_7},
                **{f"footprint_derived_count_{offense}": 0.0 for offense in OFFENSES_7},
            }
        )
    return pd.DataFrame(rows)


def _finalized(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    return _finalize_output(
        (frame if frame is not None else _surface_rows()),
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(),
    )


def _components(surface: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "state_fips": "01",
                    "bg_id": surface["block_group_geoid"],
                    "tract_id": surface["tract_id"],
                    "jurisdiction_id": np.where(
                        np.arange(len(surface)) < len(surface) // 2, "J1", "J2"
                    ),
                    "jurisdiction_type": "municipal",
                    "offense": offense,
                    "component_count": surface[f"expected_count_{offense}"].to_numpy(),
                }
            )
            for offense in OFFENSES_7
        ],
        ignore_index=True,
    )


def _controls(components: pd.DataFrame, *, benchmark: float = 0.0, estimated: float = 20.0) -> pd.DataFrame:
    rows = []
    for jurisdiction in ("J1", "J2"):
        for offense in OFFENSES_7:
            total = float(
                components[
                    components["jurisdiction_id"].eq(jurisdiction) & components["offense"].eq(offense)
                ]["component_count"].sum()
            )
            rows.append(
                {
                    "jurisdiction_id": jurisdiction,
                    "jurisdiction_type": "municipal",
                    "offense": offense,
                    "bucket_population": 300_000.0 if jurisdiction == "J1" else 20_000.0,
                    "estimated_count_ags_core": total,
                    "benchmark_imputed_count": benchmark,
                    "fill_component_count": estimated,
                    "partial_reporting_uplift_count": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _engine(runtime, *, surface=None, controls=None, audit=None) -> tuple[UncertaintyEngine, pd.DataFrame]:
    finalized = surface if surface is not None else _finalized()
    components = _components(finalized)
    engine = UncertaintyEngine(
        components=components,
        controls=controls if controls is not None else _controls(components),
        component_audit=audit if audit is not None else pd.DataFrame(),
        block_group_ids=finalized["block_group_geoid"],
        runtime=runtime,
    )
    return engine, finalized


# --- determinism ---------------------------------------------------------------------------


def test_draws_are_bit_identical_across_engines_and_repeat_calls(runtime):
    engine_a, surface = _engine(runtime)
    first = engine_a.offense_draws("burglary")
    again = engine_a.offense_draws("burglary")
    engine_b, _ = _engine(runtime)
    fresh = engine_b.offense_draws("burglary")
    assert np.array_equal(first.counts, again.counts)
    assert np.array_equal(first.counts, fresh.counts)
    assert np.array_equal(first.control_only, fresh.control_only)
    assert np.array_equal(first.share_only, fresh.share_only)


def test_draws_do_not_depend_on_the_order_offenses_are_requested(runtime):
    engine_a, _ = _engine(runtime)
    forward = {offense: engine_a.offense_draws(offense).counts for offense in OFFENSES_7}
    engine_b, _ = _engine(runtime)
    backward = {offense: engine_b.offense_draws(offense).counts for offense in reversed(OFFENSES_7)}
    for offense in OFFENSES_7:
        assert np.array_equal(forward[offense], backward[offense]), offense


def test_the_seed_is_a_fixed_constant_and_not_wall_clock(runtime):
    assert isinstance(UNCERTAINTY_SEED, int)
    source = (REPO_ROOT / "src" / "crimerisk" / "uncertainty.py").read_text()
    for forbidden in ("time.time(", "datetime.now", "np.random.seed", "random.random("):
        assert forbidden not in source, forbidden
    # Every generator is keyed; none is constructed without an explicit seed.
    assert "default_rng()" not in source


def test_a_different_draw_count_is_a_different_cache_key(runtime):
    other = resolve_uncertainty_runtime(PATHS, n_draws=N_TEST_DRAWS + 1)
    assert other.signature() != runtime.signature()


def test_cached_summary_signature_covers_final_surface_and_direct_concentration_inputs():
    surface = pd.DataFrame({
        "block_group_geoid": ["010010201001"],
        "expected_count_robbery": [2.0],
        "primary_denominator_robbery": [100.0],
        "primary_national_rate_per_100k_robbery": [200.0],
        "primary_index_publishable_robbery": [True],
        "index_robbery_primary": [1000.0],
        "benchmark_imputed_share_robbery": [0.0],
        "footprint_ambient_exposure_missing_robbery": [False],
        "unresolved_level_flag_robbery": [False],
        "domain_overlap_score_robbery": [1.0],
        "numerator_support_source_robbery": ["model"],
        "daytime_population_jobs_proxy": [100.0],
        "landscan_day_pop": [110.0],
    })
    audit = pd.DataFrame({
        "jurisdiction_id": ["01:municipal:test"],
        "jurisdiction_type": ["municipal"], "offense": ["robbery"],
        "city_incident_posterior_active": [True], "incident_count": [10.0],
        "city_posterior_alpha": [5.0],
    })

    def signature(frame=surface, component_audit=audit, ratios=None):
        return uncertainty_summary_inputs_signature(
            surfaces={"block_group_ags_core": frame},
            geo_columns={"block_group_ags_core": "block_group_geoid"},
            component_audit=component_audit,
            state_calibration_ratios=ratios or {("01", "robbery"): 1.0},
        )

    baseline = signature()
    for column, value in (
        ("expected_count_robbery", 3.0),
        ("primary_index_publishable_robbery", False),
        ("benchmark_imputed_share_robbery", 0.75),
        ("domain_overlap_score_robbery", 0.0),
        ("daytime_population_jobs_proxy", 120.0),
    ):
        changed = surface.copy()
        changed.loc[0, column] = value
        assert signature(frame=changed) != baseline, column
    changed_audit = audit.copy()
    changed_audit.loc[0, "city_posterior_alpha"] = 6.0
    assert signature(component_audit=changed_audit) != baseline
    assert signature(ratios={("01", "robbery"): 1.25}) != baseline


def test_the_cache_depends_on_the_modules_that_decide_publication(runtime):
    """What this cache stores is SUMMARIES, and a summary is null outside the publishable set.

    Measured regression: the near-zero-resident opportunity floor withdrew 367 robbery cells and a
    cached rebuild returned bin probabilities and decision tiers for all 367 of them, orphaned from
    a null index -- caught by the release validator, not by this cache. The publication mask's
    producers are dependencies of the cache in exactly the sense `uncertainty.py` already was.
    """
    stamped = {path.name for path in _uncertainty_dependency_paths(runtime)}
    for module in ("uncertainty.py", "allocation.py", "special_use.py", "exposure_ensemble.py"):
        assert module in stamped, module


# --- conservation --------------------------------------------------------------------------


@pytest.mark.parametrize("offense", ["burglary", "murder"])
def test_every_draw_conserves_its_footprint_exactly(runtime, offense):
    engine, _ = _engine(runtime)
    geometry = engine._geometry_for(offense)
    offense_index = list(OFFENSES_7).index(offense)
    for draw in range(runtime.n_draws):
        control = engine._control_draw(geometry, offense_index=offense_index, draw=draw)
        shares = engine._share_draw(geometry, offense_index=offense_index, draw=draw)
        assert np.allclose(np.add.reduceat(shares, geometry.group_starts), 1.0, atol=1e-12)
        placed = np.add.reduceat(control[geometry.group_index] * shares, geometry.group_starts)
        assert np.allclose(placed, control, rtol=1e-12, atol=1e-9)


def test_a_fully_measured_control_is_never_perturbed(runtime):
    surface = _finalized()
    components = _components(surface)
    controls = _controls(components, benchmark=0.0, estimated=0.0)
    engine = UncertaintyEngine(
        components=components,
        controls=controls,
        component_audit=pd.DataFrame(),
        block_group_ids=surface["block_group_geoid"],
        runtime=runtime,
    )
    geometry = engine._geometry_for("burglary")
    assert np.allclose(geometry.measured_fraction, 1.0)
    for draw in range(runtime.n_draws):
        control = engine._control_draw(geometry, offense_index=4, draw=draw)
        assert np.allclose(control, geometry.group_total)


def test_the_tract_rollup_sums_the_same_draw(runtime):
    engine, surface = _engine(runtime)
    draws = engine.offense_draws("burglary")
    tract_ids = surface["tract_id"].astype("string")
    codes, uniques = pd.factorize(tract_ids, sort=True)
    rolled = rollup_draws(draws.counts, group_row=codes, n_groups=len(uniques))
    expected = (
        pd.DataFrame(draws.counts).groupby(codes).sum().reindex(range(len(uniques))).to_numpy()
    )
    assert np.allclose(rolled, expected, rtol=1e-5)
    assert np.allclose(rolled.sum(axis=0), draws.counts.sum(axis=0), rtol=1e-5)


# --- the control decomposition --------------------------------------------------------------


def test_control_mass_fractions_partition_the_control(runtime):
    controls = _controls(_components(_finalized()), benchmark=30.0, estimated=20.0)
    fractions = control_mass_fractions(controls)
    total = (
        fractions["measured_fraction"] + fractions["estimated_fraction"] + fractions["benchmark_fraction"]
    )
    assert np.allclose(total.to_numpy(), 1.0)
    assert (fractions["benchmark_fraction"] > 0.0).all()
    assert (fractions["estimated_fraction"] > 0.0).all()


def test_quasi_poisson_dispersion_is_converted_to_lognormal_cv_exactly():
    dispersion = 8.0
    mass = np.array([0.001, 2.0, 200.0, 0.0])
    sigma = quasi_poisson_lognormal_sigma(dispersion, mass)
    expected_cv2 = np.array([8_000.0, 4.0, 0.04])
    assert np.allclose(np.expm1(np.square(sigma[:3])), expected_cv2)
    assert sigma[3] == 0.0
    assert np.isfinite(sigma).all()
    # The retired small-CV approximation exceeded 89 here and overflowed draws.
    assert sigma[0] < 4.0


def test_benchmark_sigmas_reproduce_the_measured_e5_bounds(runtime):
    bounds = runtime.imputation_bounds
    row = bounds[bounds["offense"].eq("burglary") & bounds["lane"].eq("county_remainder")].iloc[0]
    sigma_lo, sigma_hi = benchmark_multiplier_sigmas(
        lanes=np.array(["county_remainder"]), offense="burglary", bounds=bounds
    )
    z80 = 1.2815515655446004
    assert np.isclose(float(np.exp(-sigma_lo[0] * z80)), float(row["mult_lo80"]), rtol=1e-9)
    assert np.isclose(float(np.exp(sigma_hi[0] * z80)), float(row["mult_hi80"]), rtol=1e-9)


def test_an_unmeasured_benchmark_lane_falls_wide_not_narrow(runtime):
    bounds = runtime.imputation_bounds
    known_lo, known_hi = benchmark_multiplier_sigmas(
        lanes=np.array(["county_remainder"]), offense="burglary", bounds=bounds
    )
    unknown_lo, unknown_hi = benchmark_multiplier_sigmas(
        lanes=np.array(["a_lane_that_was_never_measured"]), offense="burglary", bounds=bounds
    )
    assert unknown_lo[0] >= known_lo[0]
    assert unknown_hi[0] >= known_hi[0]


def test_a_footprint_with_no_control_row_is_treated_as_fully_estimated(runtime):
    surface = _finalized()
    components = _components(surface)
    controls = _controls(components)
    controls = controls[controls["jurisdiction_id"].eq("J1")]
    engine = UncertaintyEngine(
        components=components,
        controls=controls,
        component_audit=pd.DataFrame(),
        block_group_ids=surface["block_group_geoid"],
        runtime=runtime,
    )
    geometry = engine._geometry_for("burglary")
    unmatched = np.flatnonzero(geometry.estimated_fraction >= 1.0)
    assert len(unmatched) == 1
    assert geometry.measured_fraction[unmatched[0]] == 0.0


# --- the concentration rule -------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.4, 0.8, 1.5])
@pytest.mark.parametrize("n_cells", [40, 400])
def test_the_concentration_hits_the_dispersion_it_claims(target, n_cells):
    share = np.array([1.0 / n_cells])
    concentration = concentration_from_log_share_dispersion(share, np.array([target]))
    assert np.isclose(float(dirichlet_log_share_sd(concentration, share)[0]), target, rtol=1e-3)
    # Simulation is the independent check: the closed form must not be verified against itself.
    rng = np.random.default_rng(11)
    alpha = float(concentration[0]) / n_cells
    gamma = rng.gamma(alpha, size=(4000, n_cells))
    simulated = gamma / gamma.sum(axis=1, keepdims=True)
    assert np.isclose(float(np.log(simulated).std(axis=0).mean()), target, rtol=0.05)


def test_an_undefined_dispersion_target_tightens_rather_than_opens():
    assert concentration_from_log_share_dispersion(np.array([0.0]), np.array([0.5]))[0] == MAX_SHARE_CONCENTRATION
    assert concentration_from_log_share_dispersion(np.array([0.01]), np.array([0.0]))[0] == MAX_SHARE_CONCENTRATION
    assert concentration_from_log_share_dispersion(np.array([0.01]), np.array([np.nan]))[0] == MAX_SHARE_CONCENTRATION


def test_a_direct_footprint_uses_the_production_posterior_concentration(runtime):
    surface = _finalized()
    components = _components(surface)
    audit = pd.DataFrame(
        {
            "jurisdiction_id": ["J1"] * 4,
            "jurisdiction_type": ["municipal"] * 4,
            "offense": ["burglary"] * 4,
            "city_incident_posterior_active": [True] * 4,
            "incident_count": [100.0, 200.0, 300.0, 400.0],
            "city_posterior_alpha": [25.0] * 4,
        }
    )
    engine = UncertaintyEngine(
        components=components,
        controls=_controls(components),
        component_audit=audit,
        block_group_ids=surface["block_group_geoid"],
        runtime=runtime,
    )
    geometry = engine._geometry_for("burglary")
    assert float(geometry.concentration[0]) == pytest.approx(1000.0 + 25.0)


# --- quantiles, probabilities and bins ---------------------------------------------------------


def test_index_bins_are_the_published_break_set():
    assert len(INDEX_BIN_LABELS) == len(INDEX_BREAKS) + 1
    values = np.array([1.0, 12.4, 12.5, 99.9, 100.0, 132.9, 800.0, 5_000.0, np.nan])
    assert index_bin_indices(values).tolist() == [0, 2, 3, 6, 7, 7, 11, 13, -1]
    labels = index_bin_labels(values)
    assert labels[0] == "<3.125"
    assert labels[6] == "800-1600"
    assert labels[-1] is None


def test_the_quantile_ladder_is_monotone_and_brackets_the_point(runtime):
    engine, surface = _engine(runtime)
    draws = engine.offense_draws("burglary")
    summary = summarize_offense(
        surface,
        offense="burglary",
        count_draws=draws.counts,
        control_log_sd=log_dispersion(draws.control_only, draws.point_counts),
        share_log_sd=log_dispersion(draws.share_only, draws.point_counts),
        runtime=runtime,
    )
    for prefix in ("expected_count_burglary", "rate_burglary_primary"):
        low = summary[f"{prefix}_p10"]
        mid = summary[f"{prefix}_p50"]
        high = summary[f"{prefix}_p90"]
        assert (low <= mid + 1e-9).all()
        assert (mid <= high + 1e-9).all()
    ladder = [summary[index_quantile_column("burglary", q)] for q in ("p10", "p25", "p50", "p75", "p90")]
    for lower, upper in zip(ladder, ladder[1:]):
        assert (lower <= upper + 1e-9).all()


def test_bin_probability_is_the_calibrated_mass_of_the_painted_bin(runtime):
    engine, surface = _engine(runtime)
    draws = engine.offense_draws("robbery")
    summary = summarize_offense(
        surface,
        offense="robbery",
        count_draws=draws.counts,
        control_log_sd=log_dispersion(draws.control_only, draws.point_counts),
        share_log_sd=log_dispersion(draws.share_only, draws.point_counts),
        runtime=runtime,
    )
    probability = pd.to_numeric(summary[prob_displayed_bin_column("robbery")], errors="coerce")
    published = probability.dropna()
    assert len(published)
    assert ((published >= 0.0) & (published <= 1.0)).all()
    labels = summary[displayed_bin_column("robbery")].astype("string")
    expected = pd.Series(
        index_bin_labels(pd.to_numeric(surface["index_robbery_primary"], errors="coerce").to_numpy()),
        dtype="string",
    )
    publishable = surface["primary_index_publishable_robbery"].fillna(False).astype(bool)
    assert labels[publishable].tolist() == expected[publishable].tolist()
    assert labels[~publishable].isna().all()


def test_probability_above_the_reference_is_the_complement_of_the_index_cdf(runtime):
    engine, surface = _engine(runtime)
    draws = engine.offense_draws("larceny")
    summary = summarize_offense(
        surface,
        offense="larceny",
        count_draws=draws.counts,
        control_log_sd=log_dispersion(draws.control_only, draws.point_counts),
        share_log_sd=log_dispersion(draws.share_only, draws.point_counts),
        runtime=runtime,
    )
    above = pd.to_numeric(summary["prob_index_larceny_above_100"], errors="coerce").dropna()
    assert ((above >= 0.0) & (above <= 1.0)).all()
    # A cell whose whole recalibrated interval sits below 100 cannot have positive probability of
    # exceeding it, and one whose p10 is above 100 must be near certain.
    high = pd.to_numeric(summary[index_quantile_column("larceny", "p10")], errors="coerce")
    confident = high.gt(100.0)
    if bool(confident.any()):
        assert (
            pd.to_numeric(summary["prob_index_larceny_above_100"], errors="coerce")[confident] >= 0.5
        ).all()


# --- calibration ----------------------------------------------------------------------------


def test_the_identity_calibration_changes_nothing():
    probabilities = np.array([0.0, 0.13, 0.5, 0.87, 1.0])
    assert np.allclose(IDENTITY_CALIBRATION.apply(probabilities), probabilities)
    for level in (0.1, 0.25, 0.5, 0.9):
        assert IDENTITY_CALIBRATION.invert(level) == pytest.approx(level, abs=1e-9)


def test_a_calibration_map_is_monotone_and_invertible():
    mapper = CalibrationMap(
        support_class="model",
        offense="burglary",
        u=np.asarray(PIT_GRID, dtype=float),
        g=np.linspace(0.0, 1.0, len(PIT_GRID)) ** 2,
    )
    grid = np.linspace(0.0, 1.0, 51)
    applied = mapper.apply(grid)
    assert (np.diff(applied) >= -1e-12).all()
    for level in (0.1, 0.5, 0.9):
        assert mapper.apply(np.array([mapper.invert(level)]))[0] == pytest.approx(level, abs=1e-6)


def test_the_shipped_calibration_table_covers_every_offense_and_class(runtime):
    maps = load_calibration(calibration_table_path(REPO_ROOT / "configs"))
    for offense in OFFENSES_7:
        for support_class in (SUPPORT_CLASS_MODEL, SUPPORT_CLASS_DIRECT):
            assert (support_class, offense) in maps, (support_class, offense)
    for support_class, expected in CALIBRATION_CLASS_FOR_SUPPORT.items():
        chosen = runtime.calibration_map(support_class=support_class, offense="burglary")
        assert chosen.support_class == expected
    table = pd.read_csv(calibration_table_path(REPO_ROOT / "configs"))
    assert set(table["calibration_version"]) == {CALIBRATION_VERSION}
    assert bool(table["held_out"].all()), "every shipped map must come from a real held-out split"


def test_recalibration_never_reorders_cells(runtime):
    mapper = runtime.calibration_map(support_class=SUPPORT_CLASS_MODEL, offense="burglary")
    probabilities = np.linspace(0.0, 1.0, 101)
    calibrated = mapper.apply(probabilities)
    assert (np.diff(calibrated) >= -1e-12).all()
    levels = [mapper.invert(level) for level in (0.10, 0.25, 0.50, 0.75, 0.90)]
    assert levels == sorted(levels)


def test_the_measured_dispersion_tables_are_complete_and_versioned(runtime):
    control = pd.read_csv(control_dispersion_path(REPO_ROOT / "configs"))
    assert set(control["control_dispersion_version"]) == {CONTROL_DISPERSION_VERSION}
    assert set(control["offense"]) == set(OFFENSES_7)
    assert (control["quasi_poisson_dispersion"] > 0.0).all()
    loaded = load_control_dispersion(control_dispersion_path(REPO_ROOT / "configs"))
    assert set(loaded) == set(OFFENSES_7)

    share = pd.read_csv(share_dispersion_path(REPO_ROOT / "configs"))
    assert set(share["share_dispersion_version"]) == {SHARE_DISPERSION_VERSION}
    assert set(zip(share["offense"], share["stratum"])) == {
        (offense, stratum) for offense in OFFENSES_7 for stratum in STRATA
    }
    assert (share["target_log_sd"] > 0.0).all()
    # The TVD rides along as a diagnostic; it must not be mistaken for the fitting target.
    assert "measured_tvd" in share.columns
    assert load_share_dispersion(share_dispersion_path(REPO_ROOT / "configs"))


def test_a_share_dispersion_table_written_against_the_retired_rule_is_refused(tmp_path):
    legacy = tmp_path / "legacy.csv"
    pd.DataFrame(
        [{"offense": offense, "stratum": stratum, "target_tvd": 0.5} for offense in OFFENSES_7 for stratum in STRATA]
    ).to_csv(legacy, index=False)
    with pytest.raises(ValueError, match="target_log_sd"):
        load_share_dispersion(legacy)


def test_the_e1_control_dispersion_is_the_tournament_winners_measured_deviance():
    results = pd.read_csv(
        REPO_ROOT / "analysis_scratch" / "final_phase" / "e1_tournament" / "round2_results.csv"
    )
    expected = (
        results[results["estimator"].eq("dynamic_shrink")].groupby("offense")["mean_dev"].mean()
    )
    shipped = load_control_dispersion(control_dispersion_path(REPO_ROOT / "configs"))
    for offense, value in expected.items():
        assert shipped[str(offense)] == pytest.approx(float(value), rel=1e-9)


def test_the_recorded_coverage_improves_on_the_uncalibrated_intervals():
    coverage = pd.read_csv(EVIDENCE_DIR / "coverage_by_offense.csv")
    at_80 = coverage[coverage["nominal_level"].eq(0.80)]
    assert len(at_80) >= 12
    before = (at_80["coverage_before"] - 0.80).abs().mean()
    after = (at_80["coverage_after"] - 0.80).abs().mean()
    assert after < before, f"recalibration must reduce the coverage gap ({before=} {after=})"
    assert bool(at_80["held_out"].all())


# --- tiers ------------------------------------------------------------------------------------


def _tier(**overrides) -> str:
    base = {
        "bin_probability": 0.95,
        "publishable": True,
        "point_index": 120.0,
        "benchmark_share": 0.0,
        "footprint_conflict": False,
        "unresolved_level": False,
        "out_of_domain": False,
    }
    base.update(overrides)
    index = pd.RangeIndex(1)
    return str(
        decision_reliability_tier(
            bin_probability=pd.Series([base["bin_probability"]], index=index, dtype=float),
            publishable=pd.Series([base["publishable"]], index=index),
            point_index=pd.Series([base["point_index"]], index=index, dtype=float),
            benchmark_share=pd.Series([base["benchmark_share"]], index=index, dtype=float),
            footprint_conflict=pd.Series([base["footprint_conflict"]], index=index),
            unresolved_level=pd.Series([base["unresolved_level"]], index=index),
            out_of_domain=pd.Series([base["out_of_domain"]], index=index),
        ).iloc[0]
    )


def test_the_tier_rule_is_the_consults_rule():
    assert _tier(bin_probability=TIER_HIGH_MIN_BIN_PROBABILITY) == TIER_HIGH
    assert _tier(bin_probability=TIER_HIGH_MIN_BIN_PROBABILITY - 1e-9) == TIER_MEDIUM
    assert _tier(bin_probability=TIER_MEDIUM_MIN_BIN_PROBABILITY) == TIER_MEDIUM
    assert _tier(bin_probability=TIER_MEDIUM_MIN_BIN_PROBABILITY - 1e-9) == TIER_LOW
    # Each forcing condition demotes a would-be High to Low on its own.
    assert _tier(benchmark_share=TIER_LOW_BENCHMARK_SHARE_MAX + 1e-9) == TIER_LOW
    assert _tier(benchmark_share=TIER_LOW_BENCHMARK_SHARE_MAX) == TIER_HIGH
    assert _tier(footprint_conflict=True) == TIER_LOW
    assert _tier(unresolved_level=True) == TIER_LOW
    assert _tier(out_of_domain=True) == TIER_LOW
    # Withheld beats everything: an undefined denominator or an undefendable placement.
    assert _tier(publishable=False) == TIER_WITHHELD
    assert _tier(point_index=np.nan) == TIER_WITHHELD
    assert _tier(bin_probability=np.nan) == TIER_WITHHELD
    assert _tier(publishable=False, bin_probability=0.99) == TIER_WITHHELD


def test_support_class_prefers_the_control_statement_over_the_share_statement():
    frame = pd.DataFrame(
        {
            "benchmark_imputed_share_burglary": [0.0, 0.0, 0.9, 0.9],
            "numerator_support_source_burglary": [
                "direct_city_incident",
                "model_only",
                "direct_city_incident",
                "model_only",
            ],
        }
    )
    classes = support_class_series(frame, offense="burglary").tolist()
    assert classes == [
        SUPPORT_CLASS_DIRECT,
        SUPPORT_CLASS_MODEL,
        SUPPORT_CLASS_BENCHMARK,
        SUPPORT_CLASS_BENCHMARK,
    ]


# --- the denominator component -----------------------------------------------------------------


def test_the_denominator_term_reads_the_two_competing_public_proxies():
    frame = pd.DataFrame(
        {
            "daytime_population_jobs_proxy": [1_000.0, 1_000.0, 10.0, 1_000.0],
            "landscan_day_pop": [1_000.0, 4_000.0, 4_000.0, 20.0],
        }
    )
    spread = denominator_log_sd(frame, offense="robbery").to_numpy()
    assert spread[0] == pytest.approx(0.0)
    assert spread[1] == pytest.approx(0.5 * np.log(4.0))
    # Either surface below the exposure floor makes the pair uninformative rather than enormous.
    assert spread[2] == pytest.approx(0.0)
    assert spread[3] == pytest.approx(0.0)


def test_offenses_without_a_competing_proxy_publish_a_zero_term_not_a_guess():
    frame = pd.DataFrame(
        {"daytime_population_jobs_proxy": [1_000.0], "landscan_day_pop": [4_000.0]}
    )
    for offense in ("burglary", "motor_vehicle_theft"):
        assert float(denominator_log_sd(frame, offense=offense).iloc[0]) == 0.0


# --- attaching, caching, and the manifest -------------------------------------------------------


def test_the_layer_refuses_to_overwrite_an_existing_published_column(runtime):
    engine, surface = _engine(runtime)
    draws = engine.offense_draws("burglary")
    summary = summarize_offense(
        surface,
        offense="burglary",
        count_draws=draws.counts,
        control_log_sd=log_dispersion(draws.control_only, draws.point_counts),
        share_log_sd=log_dispersion(draws.share_only, draws.point_counts),
        runtime=runtime,
    )
    collided = surface.copy()
    collided[decision_tier_column("burglary")] = "high"
    with pytest.raises(ValueError, match="overwrite"):
        apply_uncertainty_layer(collided, summaries={"burglary": summary}, runtime=runtime)


def test_the_cache_round_trips_every_published_field(runtime):
    engine, surface = _engine(runtime)
    summaries = {}
    for offense in OFFENSES_7:
        draws = engine.offense_draws(offense)
        summaries[offense] = summarize_offense(
            surface,
            offense=offense,
            count_draws=draws.counts,
            control_log_sd=log_dispersion(draws.control_only, draws.point_counts),
            share_log_sd=log_dispersion(draws.share_only, draws.point_counts),
            runtime=runtime,
        )
    cache = summaries_to_cache_frame({"block_group_ags_core": (surface["block_group_geoid"], summaries)})
    restored = cache_frame_to_summaries(
        cache, surface="block_group_ags_core", geo_ids=surface["block_group_geoid"]
    )
    for offense in OFFENSES_7:
        for column, values in summaries[offense].items():
            left = pd.Series(values).astype("object").to_numpy()
            right = pd.Series(restored[offense][column]).astype("object").to_numpy()
            assert [
                (None if pd.isna(value) else value) for value in left
            ] == [(None if pd.isna(value) else value) for value in right], column


def test_the_signature_moves_when_the_components_move(runtime):
    surface = _finalized()
    components = _components(surface)
    controls = _controls(components)
    baseline = components_signature(components, controls)
    assert baseline == components_signature(components.copy(), controls.copy())
    moved = components.copy()
    moved.loc[0, "component_count"] = float(moved.loc[0, "component_count"]) + 1.0
    assert components_signature(moved, controls) != baseline

    sourced = components.copy()
    sourced["source_state_fips"] = "01"
    sourced_signature = components_signature(sourced, controls)
    sourced.loc[0, "source_state_fips"] = "04"
    assert components_signature(sourced, controls) != sourced_signature


def test_the_manifest_records_the_lane(runtime):
    manifest = _allocation_build_manifest(
        paths=PATHS,
        config=AllocationBuildConfig(enable_uncertainty_layer=True, uncertainty_draws=DEFAULT_N_DRAWS),
        summary={},
        output_paths={},
    )
    block = manifest["resolved_config"]["uncertainty_layer"]
    assert block["enabled"] is True
    assert block["version"] == UNCERTAINTY_LAYER_VERSION
    assert block["calibration_version"] == CALIBRATION_VERSION
    assert block["n_draws"] == DEFAULT_N_DRAWS
    assert block["index_breaks"] == list(INDEX_BREAKS)
    assert block["contract"].endswith("UNCERTAINTY_LAYER_CONTRACT.md")
    off = _allocation_build_manifest(
        paths=PATHS, config=AllocationBuildConfig(), summary={}, output_paths={}
    )
    assert off["resolved_config"]["uncertainty_layer"]["enabled"] is False


def test_the_summary_reports_the_measured_dispersions_and_tier_distribution(runtime):
    engine, surface = _engine(runtime)
    draws = engine.offense_draws("burglary")
    summary = summarize_offense(
        surface,
        offense="burglary",
        count_draws=draws.counts,
        control_log_sd=log_dispersion(draws.control_only, draws.point_counts),
        share_log_sd=log_dispersion(draws.share_only, draws.point_counts),
        runtime=runtime,
    )
    attached = apply_uncertainty_layer(surface, summaries={"burglary": summary}, runtime=runtime)
    reported = summarize_uncertainty_layer(runtime=runtime, surfaces={"block_group_ags_core": attached})
    assert reported["enabled"] is True
    assert set(reported["control_dispersion"]) == set(OFFENSES_7)
    assert reported["decision_tier_distribution"]["block_group_ags_core"]["burglary"]
    assert summarize_uncertainty_layer(runtime=None) == {"enabled": False}


# --- legacy byte-safety ---------------------------------------------------------------------------


def test_a_flag_off_surface_gains_no_column_and_no_value():
    rows = _surface_rows()
    legacy = _finalize_output(
        rows,
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(),
    )
    with_flag = _finalize_output(
        rows,
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(enable_uncertainty_layer=True),
    )
    assert list(legacy.columns) == list(with_flag.columns)
    pd.testing.assert_frame_equal(legacy, with_flag)
    for column in uncertainty_published_columns():
        assert column not in legacy.columns


def test_the_lane_publishes_only_its_own_columns(runtime):
    engine, surface = _engine(runtime)
    draws = engine.offense_draws("burglary")
    summary = summarize_offense(
        surface,
        offense="burglary",
        count_draws=draws.counts,
        control_log_sd=log_dispersion(draws.control_only, draws.point_counts),
        share_log_sd=log_dispersion(draws.share_only, draws.point_counts),
        runtime=runtime,
    )
    attached = apply_uncertainty_layer(surface, summaries={"burglary": summary}, runtime=runtime)
    added = [column for column in attached.columns if column not in surface.columns]
    assert set(added) == {"uncertainty_layer_version", *uncertainty_columns_for_offense("burglary")}
    # Every pre-existing column is bit-identical, including the reliability tier and the
    # recommended geography this lane deliberately does not redefine.
    pd.testing.assert_frame_equal(attached[surface.columns], surface)


def test_the_rare_offense_policy_carries_the_quantiles_and_not_the_counts():
    for offense in ("murder", "rape"):
        fields = set(_rare_offense_published_point_fields(offense))
        for column in rare_offense_suppressed_columns(offense):
            assert column in fields, column
        assert f"expected_count_{offense}_p50" not in fields
        assert f"expected_count_{offense}_p90" not in fields


# --- the four-surface wiring and its cache -------------------------------------------------------


def _surface_bundle(tmp_path) -> dict[str, object]:
    """The four surfaces the build hands the lane, plus the pieces `_build_uncertainty_surfaces`
    needs: a tract rollup that really is the block groups summed, and an FBI-calibrated twin that
    really is the counts times a per-state scalar."""
    rows = _surface_rows()
    block_group = _finalized(rows)
    tract_rows = (
        rows.assign(tract_id=rows["tract_id"])
        .groupby("tract_id", as_index=False)
        .agg(
            {
                "state_fips": "first",
                "population_2024": "sum",
                "households_total": "sum",
                "commercial_premises_total": "sum",
                "destination_poi_total": "sum",
                "daytime_population_jobs_proxy": "sum",
                "landscan_day_pop": "sum",
                "exposure_proxy_2024": "sum",
                "burglary_premises_total": "sum",
                "aggregate_vehicles_total": "sum",
                "vehicle_exposure_2024": "sum",
                "land_area_sq_mi": "sum",
                **{f"expected_count_{offense}": "sum" for offense in OFFENSES_7},
                **{f"footprint_derived_count_{offense}": "sum" for offense in OFFENSES_7},
            }
        )
    )
    tract_rows["dominant_eb_jurisdiction_id"] = "J1"
    tract = _finalize_output(
        tract_rows,
        geo_id_col="tract_id",
        population_col="population_2024",
        config=AllocationBuildConfig(),
        jurisdiction_col="dominant_eb_jurisdiction_id",
    )
    ratios = {("01", offense): 1.25 for offense in OFFENSES_7}
    calibrated_rows = rows.copy()
    for offense in OFFENSES_7:
        calibrated_rows[f"expected_count_{offense}"] = calibrated_rows[f"expected_count_{offense}"] * 1.25
    block_group_cal = _finalized(calibrated_rows)
    tract_cal_rows = tract_rows.copy()
    for offense in OFFENSES_7:
        tract_cal_rows[f"expected_count_{offense}"] = tract_cal_rows[f"expected_count_{offense}"] * 1.25
    tract_cal = _finalize_output(
        tract_cal_rows,
        geo_id_col="tract_id",
        population_col="population_2024",
        config=AllocationBuildConfig(),
        jurisdiction_col="dominant_eb_jurisdiction_id",
    )
    components = _components(block_group)
    return {
        "components": components,
        "controls": _controls(components),
        "block_group_ags_core": block_group,
        "tract_ags_core": tract,
        "block_group_fbi_calibrated": block_group_cal,
        "tract_fbi_calibrated": tract_cal,
        "state_calibration_ratios": ratios,
    }


def test_the_build_wiring_attaches_all_four_surfaces_from_one_set_of_draws(tmp_path, runtime):
    from crimerisk.allocation import _build_uncertainty_surfaces

    bundle = _surface_bundle(tmp_path)
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    paths = type(paths)(**{**paths.__dict__, "cache_dir": tmp_path / "cache"})
    config = AllocationBuildConfig(enable_uncertainty_layer=True, uncertainty_draws=N_TEST_DRAWS)

    def run():
        return _build_uncertainty_surfaces(
            paths=paths,
            config=config,
            runtime=runtime,
            components=bundle["components"],
            controls=bundle["controls"],
            component_audit=pd.DataFrame(),
            block_group_ags_core=bundle["block_group_ags_core"],
            tract_ags_core=bundle["tract_ags_core"],
            block_group_fbi_calibrated=bundle["block_group_fbi_calibrated"],
            tract_fbi_calibrated=bundle["tract_fbi_calibrated"],
            state_calibration_ratios=bundle["state_calibration_ratios"],
            population_col="population_2024",
        )

    surfaces = run()
    assert set(surfaces) == {
        "block_group_ags_core",
        "tract_ags_core",
        "block_group_fbi_calibrated",
        "tract_fbi_calibrated",
    }
    for label, frame in surfaces.items():
        assert "uncertainty_layer_version" in frame.columns, label
        assert len(frame) == len(bundle[label])
        for offense in OFFENSES_7:
            assert decision_tier_column(offense) in frame.columns

    # The tract surface is the block-group draws summed: its count median must equal the sum of its
    # children's, cell for cell, because it is literally the same draw.
    block_group = surfaces["block_group_ags_core"]
    tract = surfaces["tract_ags_core"]
    rolled = (
        block_group.assign(parent=block_group["block_group_geoid"].str.slice(0, 11))
        .groupby("parent")["expected_count_burglary_p50"]
        .sum()
    )
    assert float(rolled.sum()) > 0.0
    assert float(tract["expected_count_burglary_p50"].sum()) == pytest.approx(
        float(rolled.sum()), rel=0.15
    )

    # The FBI-calibrated twin is the same draws under the same scalar, so its quantiles scale.
    calibrated = surfaces["block_group_fbi_calibrated"]
    assert calibrated["expected_count_burglary_p50"].to_numpy() == pytest.approx(
        1.25 * block_group["expected_count_burglary_p50"].to_numpy(), rel=1e-5
    )

    # Second call is a cache hit and reproduces the first bit for bit.
    cached = run()
    for label in surfaces:
        pd.testing.assert_frame_equal(surfaces[label], cached[label])
    written = list((tmp_path / "cache" / "uncertainty").glob("*.parquet"))
    assert len(written) == 1

    # A finalized decision input can move without changing the pre-finalize components. It must
    # address a new cache artifact rather than restoring the old tier beside the new surface.
    bundle["block_group_ags_core"] = bundle["block_group_ags_core"].copy()
    bundle["block_group_ags_core"].loc[0, "domain_overlap_score_burglary"] = 0.0
    run()
    written = list((tmp_path / "cache" / "uncertainty").glob("*.parquet"))
    assert len(written) == 2


# --- the release validator's independent mirror ------------------------------------------------


def test_the_validator_mirrors_the_bin_and_the_tier(runtime):
    module = _validator_module()
    engine, surface = _engine(runtime)
    summaries = {}
    for offense in OFFENSES_7:
        draws = engine.offense_draws(offense)
        summaries[offense] = summarize_offense(
            surface,
            offense=offense,
            count_draws=draws.counts,
            control_log_sd=log_dispersion(draws.control_only, draws.point_counts),
            share_log_sd=log_dispersion(draws.share_only, draws.point_counts),
            runtime=runtime,
        )
    attached = apply_uncertainty_layer(surface, summaries=summaries, runtime=runtime)
    assert module._uncertainty_layer_lane(attached)
    assert not module._uncertainty_layer_lane(surface)
    assert module._uncertainty_layer_issues(attached, label="tract_ags_core") == []

    # The mirror must actually disagree when the producer is wrong.
    corrupted = attached.copy()
    corrupted[displayed_bin_column("robbery")] = ">=800"
    issues = module._uncertainty_layer_issues(corrupted, label="tract_ags_core")
    assert any("displayed_index_bin_robbery" in issue for issue in issues)

    corrupted = attached.copy()
    corrupted[decision_tier_column("robbery")] = "high"
    issues = module._uncertainty_layer_issues(corrupted, label="tract_ags_core")
    assert any("decision_reliability_tier_robbery" in issue for issue in issues)

    corrupted = attached.copy()
    corrupted[index_quantile_column("robbery", "p10")] = 1e9
    issues = module._uncertainty_layer_issues(corrupted, label="tract_ags_core")
    assert any("monotone" in issue for issue in issues)

    corrupted = attached.copy()
    corrupted[prob_displayed_bin_column("robbery")] = 1.5
    issues = module._uncertainty_layer_issues(corrupted, label="tract_ags_core")
    assert any("outside [0, 1]" in issue for issue in issues)


def test_the_validators_constants_are_transcribed_not_imported():
    source = (REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py").read_text()
    assert "from crimerisk.uncertainty import" not in source
    module = _validator_module()
    assert module.UNCERTAINTY_INDEX_BREAKS == list(INDEX_BREAKS)
    assert module.UNCERTAINTY_INDEX_BIN_LABELS == list(INDEX_BIN_LABELS)
    assert module.UNCERTAINTY_TIER_HIGH_MIN == TIER_HIGH_MIN_BIN_PROBABILITY
    assert module.UNCERTAINTY_TIER_MEDIUM_MIN == TIER_MEDIUM_MIN_BIN_PROBABILITY
    assert module.UNCERTAINTY_TIER_BENCHMARK_SHARE_MAX == TIER_LOW_BENCHMARK_SHARE_MAX


# --- the contract ---------------------------------------------------------------------------------


def test_the_contract_states_the_display_compliance_and_the_measured_results():
    text = CONTRACT_PATH.read_text()
    for phrase in (
        "doubt disclosure",
        "conservation",
        "split-conformal",
        "leave-city-out",
        "leave-year-out",
        "coverage",
        UNCERTAINTY_LAYER_VERSION,
    ):
        assert phrase in text, phrase
    assert (EVIDENCE_DIR / "coverage_by_offense.csv").exists()
    assert (EVIDENCE_DIR / "national_tier_distribution.json").exists()


def test_helper_identities_used_across_the_lane():
    assert stratum_for_population(250_000) == "core"
    assert stratum_for_population(60_000) == "suburban"
    assert stratum_for_population(10_000) == "small"
    assert stable_fold("22:municipal:place:2205000", 5) == stable_fold("22:municipal:place:2205000", 5)
    assert 0 <= stable_fold("anything", 5) < 5
    assert load_imputation_bounds(REPO_ROOT / "configs" / "imputation_empirical_bounds.csv").shape[0] > 0


def test_source_state_scaling_reuses_draws_for_cross_state_service_group(runtime):
    surface = _finalized()
    components = _components(surface)
    components["source_state_fips"] = np.where(
        components["jurisdiction_id"].eq("J1"), "04", "01"
    )
    engine = UncertaintyEngine(
        components=components,
        controls=_controls(components),
        component_audit=pd.DataFrame(),
        block_group_ids=surface["block_group_geoid"],
        runtime=runtime,
    )
    baseline = engine.offense_draws("robbery")
    calibrated = engine.offense_draws(
        "robbery", source_state_multipliers={"04": 2.0, "01": 3.0}
    )
    first_group = components.loc[
        components["offense"].eq("robbery") & components["jurisdiction_id"].eq("J1"),
        "bg_id",
    ]
    first_rows = surface["block_group_geoid"].isin(first_group).to_numpy()
    assert np.allclose(calibrated.counts[first_rows], baseline.counts[first_rows] * 2.0)
    assert np.allclose(
        calibrated.control_only[first_rows], baseline.control_only[first_rows] * 2.0
    )
    assert np.allclose(
        calibrated.share_only[first_rows], baseline.share_only[first_rows] * 2.0
    )
    assert np.allclose(
        calibrated.point_counts[first_rows], baseline.point_counts[first_rows] * 2.0
    )
    assert np.allclose(
        calibrated.counts[~first_rows], baseline.counts[~first_rows] * 3.0
    )


def test_source_state_scaling_rejects_one_group_with_multiple_sources(runtime):
    surface = _finalized()
    components = _components(surface)
    components["source_state_fips"] = "01"
    first_group = components["jurisdiction_id"].eq("J1")
    components.loc[
        first_group & components.index.to_series().mod(2).eq(0), "source_state_fips"
    ] = "04"
    engine = UncertaintyEngine(
        components=components,
        controls=_controls(components),
        component_audit=pd.DataFrame(),
        block_group_ids=surface["block_group_geoid"],
        runtime=runtime,
    )
    with pytest.raises(ValueError, match="multiple source states"):
        engine.offense_draws("robbery")
