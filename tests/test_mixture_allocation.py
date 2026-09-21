"""The three-expert mixture allocator (contract: MIXTURE_ALLOCATOR_CONTRACT.md).

Five things are load-bearing and are asserted here: the frozen weight table is read and
validated rather than re-derived, the stratum and envelope rules are the tournament's, the
mixture arithmetic matches a hand-computed three-expert case, the model lane's prior information
mass survives the substitution, and the legacy path is byte-identical when the mixture is off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import AllocationBuildConfig, _apply_model_lane_shares
from crimerisk.crime import OFFENSES_7
from crimerisk.mixture_allocation import (
    CORE_POPULATION_FLOOR,
    CORE_STRATUM,
    EXPOSURE_BASIS_BY_OFFENSE,
    EXPOSURE_BASIS_POPULATION,
    MIXTURE_SHARE_AUDIT_COLUMNS,
    SMALL_STRATUM,
    STRATA,
    SUBURBAN_POPULATION_FLOOR,
    SUBURBAN_STRATUM,
    WEIGHT_COLUMNS,
    MixtureAllocationConfig,
    MixtureRuntime,
    apply_mixture_to_model_lane,
    assert_mixture_expert_invariants,
    assert_mixture_share_invariants,
    classify_stratum,
    classify_stratum_series,
    clip_to_envelope,
    exposure_basis_for_offense,
    exposure_expert_weights,
    load_opportunity_normalizers,
    load_ship_weights,
    mix_expert_shares,
    normalize_within_groups,
    ship_weights_path,
    summarize_mixture_shares,
    weight_lookup,
)
from crimerisk.paths import RepoPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
SHIP_WEIGHTS = ship_weights_path(PATHS)


# --- fixtures ---------------------------------------------------------------


def _weight_table(rows: list[dict] | None = None) -> pd.DataFrame:
    if rows is None:
        rows = [
            {"offense": offense, "stratum": stratum, "w_prior": 1.0, "w_gbm": 0.0, "w_exposure": 0.0}
            for offense in OFFENSES_7
            for stratum in STRATA
        ]
    return pd.DataFrame(rows)


def _write_weights(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "weights.csv"
    frame.to_csv(path, index=False)
    return path


def _experts(
    bg_ids: list[str],
    offense: str,
    *,
    population: list[float],
    gbm_weight: list[float],
    exposure_weight: list[float] | None = None,
) -> pd.DataFrame:
    exposure = exposure_weight if exposure_weight is not None else [max(value, 1.0) for value in population]
    return pd.DataFrame(
        {
            "state_fips": [bg[:2] for bg in bg_ids],
            "bg_id": pd.array(bg_ids, dtype="string"),
            "offense": pd.array([offense] * len(bg_ids), dtype="string"),
            "population": population,
            "exposure_weight": exposure,
            "exposure_basis": exposure_basis_for_offense(offense),
            "exposure_basis_row_fallback": [False] * len(bg_ids),
            "gbm_rate": [1.0] * len(bg_ids),
            "gbm_rate_clipped": [1.0] * len(bg_ids),
            "gbm_weight": gbm_weight,
            "envelope_lo": 0.0,
            "envelope_hi": 10.0,
        }
    )


def _support(bg_ids: list[str], offense: str, *, bg_weight: list[float], allocation: list[float] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "bg_id": bg_ids,
            "offense": offense,
            "jurisdiction_id": "01:municipal:place:0100100",
            "jurisdiction_type": "municipal",
            "state_fips": "01",
            "allocation_share": allocation if allocation is not None else [1.0] * len(bg_ids),
            "bg_weight": bg_weight,
        }
    )
    return frame


def _legacy_model_lane(merged: pd.DataFrame) -> pd.DataFrame:
    """The pre-change model-lane computation, transcribed verbatim as the byte-safety reference."""
    out = merged.copy()
    out["bg_weight"] = pd.to_numeric(out["bg_weight"], errors="coerce").fillna(0.0)
    out["allocation_share"] = pd.to_numeric(out["allocation_share"], errors="coerce").fillna(0.0)
    out["model_component_weight"] = out["bg_weight"] * out["allocation_share"]
    out["model_total"] = out.groupby(
        ["jurisdiction_id", "state_fips", "offense"],
        dropna=False,
    )["model_component_weight"].transform("sum")
    out["model_share"] = np.where(
        pd.to_numeric(out["model_total"], errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(out["model_component_weight"], errors="coerce").fillna(0.0)
        / pd.to_numeric(out["model_total"], errors="coerce").fillna(np.nan),
        0.0,
    )
    return out


def _runtime(experts: pd.DataFrame, weights: pd.DataFrame) -> MixtureRuntime:
    return MixtureRuntime(experts=experts, weights=weights, config=MixtureAllocationConfig(year=2024))


# --- the frozen weight table ------------------------------------------------


def test_shipped_weight_table_is_the_tournament_output_verbatim():
    """configs/mixture_ship_weights_v3.csv must still be the selection script's output.

    All 21 rows come from `14_reselect_weights_v3.py`'s full pass -- unlike v2, which carried 18
    rows verbatim from v1, nothing is carried here, because the exposure expert those weights
    multiply is a different vector now and a carried row would be a weight selected for a surface
    that no longer exists. A silent revert to a v2 triple is what this assertion catches.
    """
    corpus = REPO_ROOT / "analysis_scratch" / "final_phase" / "corpus_expansion"
    shipped = load_ship_weights(SHIP_WEIGHTS)
    source = pd.read_csv(corpus / "ship_weights_v3.csv")
    merged = shipped.merge(source, on=["offense", "stratum"], suffixes=("_cfg", "_src"))
    assert len(merged) == len(source) == 21
    for column in WEIGHT_COLUMNS:
        assert np.allclose(merged[f"{column}_cfg"], merged[f"{column}_src"], rtol=0.0, atol=0.0)
    assert set(source["selection_version"].astype(str)) == {"reselect_v3"}

    # 17 of 21 rows moved; the four that did not are genuine agreements between two different
    # objectives on two different experts, not a partial write.
    v2 = pd.read_csv(corpus / "ship_weights_v2.csv")
    both = shipped.merge(v2, on=["offense", "stratum"], suffixes=("_v3", "_v2"))
    moved = (
        both[[f"{c}_v3" for c in WEIGHT_COLUMNS]].to_numpy()
        != both[[f"{c}_v2" for c in WEIGHT_COLUMNS]].to_numpy()
    ).any(axis=1)
    assert int(moved.sum()) == 17


def test_shipped_weight_table_covers_every_offense_and_stratum_on_the_simplex():
    weights = load_ship_weights(SHIP_WEIGHTS)
    assert set(zip(weights["offense"], weights["stratum"], strict=True)) == {
        (offense, stratum) for offense in OFFENSES_7 for stratum in STRATA
    }
    assert np.allclose(weights[list(WEIGHT_COLUMNS)].sum(axis=1), 1.0, atol=1e-9)
    assert weights[list(WEIGHT_COLUMNS)].to_numpy().min() >= 0.0


def test_weight_lookup_returns_the_pooled_fallback_rows_as_shipped():
    """murder/small and rape/small have zero benchmark cells; they ship the offense-pool pick."""
    lookup = weight_lookup(load_ship_weights(SHIP_WEIGHTS))
    assert lookup[("murder", "small")] == (0.125, 0.875, 0.0)
    assert lookup[("rape", "small")] == (0.125, 0.5, 0.375)
    # Larceny's small and suburban strata were PRIOR-ONLY under v2 -- the mixture was the identity
    # there. Under the re-based expert they invert onto the opportunity normalizer, which is the
    # single largest weight movement in the table and the one to argue about first.
    assert lookup[("larceny", "suburban")] == (0.125, 0.0, 0.875)
    assert lookup[("larceny", "small")] == (0.0, 0.0, 1.0)


def test_weight_table_rejects_a_missing_cell(tmp_path):
    frame = _weight_table()
    path = _write_weights(tmp_path, frame[frame["stratum"] != SMALL_STRATUM])
    with pytest.raises(ValueError, match="must cover exactly"):
        load_ship_weights(path)


def test_weight_table_rejects_a_triple_that_is_not_convex(tmp_path):
    frame = _weight_table()
    frame.loc[0, "w_gbm"] = 0.5
    with pytest.raises(ValueError, match="does not sum to 1"):
        load_ship_weights(_write_weights(tmp_path, frame))


def test_weight_table_rejects_a_negative_weight(tmp_path):
    frame = _weight_table()
    frame.loc[0, ["w_prior", "w_gbm", "w_exposure"]] = [1.5, -0.5, 0.0]
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        load_ship_weights(_write_weights(tmp_path, frame))


def test_weight_table_rejects_duplicate_rows(tmp_path):
    frame = pd.concat([_weight_table(), _weight_table().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_ship_weights(_write_weights(tmp_path, frame))


def test_missing_weight_table_is_a_hard_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="source of record"):
        load_ship_weights(tmp_path / "absent.csv")


# --- stratum ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("population", "expected"),
    [
        (0.0, SMALL_STRATUM),
        (SUBURBAN_POPULATION_FLOOR - 1e-6, SMALL_STRATUM),
        (SUBURBAN_POPULATION_FLOOR, SUBURBAN_STRATUM),
        (CORE_POPULATION_FLOOR - 1e-6, SUBURBAN_STRATUM),
        (CORE_POPULATION_FLOOR, CORE_STRATUM),
        (3_000_000.0, CORE_STRATUM),
    ],
)
def test_stratum_boundaries_are_closed_below(population, expected):
    assert classify_stratum(population) == expected


def test_stratum_series_matches_the_scalar_rule_including_nan():
    values = pd.Series([np.nan, 0.0, 49_999.0, 50_000.0, 199_999.0, 200_000.0])
    series = classify_stratum_series(values)
    assert list(series) == [
        SMALL_STRATUM,
        SMALL_STRATUM,
        SMALL_STRATUM,
        SUBURBAN_STRATUM,
        SUBURBAN_STRATUM,
        CORE_STRATUM,
    ]


# --- envelope ---------------------------------------------------------------


def test_envelope_clips_both_tails():
    clipped = clip_to_envelope(np.array([-5.0, 0.5, 3.0, 99.0]), 1.0, 10.0)
    assert list(clipped) == [1.0, 1.0, 3.0, 10.0]


def test_envelope_rejects_an_inverted_or_nonfinite_range():
    with pytest.raises(ValueError, match="inverted"):
        clip_to_envelope(np.array([1.0]), 10.0, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        clip_to_envelope(np.array([1.0]), np.nan, 1.0)


def _all_offense_experts() -> pd.DataFrame:
    return pd.concat(
        [
            _experts(["010010001001"], offense, population=[100.0], gbm_weight=[100.0])
            for offense in OFFENSES_7
        ],
        ignore_index=True,
    )


def test_expert_invariants_accept_a_well_formed_table():
    assert_mixture_expert_invariants(_all_offense_experts())


def test_expert_invariants_reject_a_rate_outside_its_envelope():
    experts = _all_offense_experts()
    experts.loc[0, "gbm_rate_clipped"] = 99.0
    with pytest.raises(ValueError, match="outside its learned envelope"):
        assert_mixture_expert_invariants(experts)


def test_expert_invariants_reject_a_negative_weight():
    experts = _all_offense_experts()
    experts.loc[0, "gbm_weight"] = -1.0
    with pytest.raises(ValueError, match="negative gbm_weight"):
        assert_mixture_expert_invariants(experts)


def test_expert_invariants_reject_a_missing_offense():
    experts = _all_offense_experts()
    with pytest.raises(ValueError, match="expected all seven"):
        assert_mixture_expert_invariants(experts[experts["offense"].astype(str) != "murder"])


def test_expert_invariants_reject_duplicate_keys():
    experts = _all_offense_experts()
    with pytest.raises(ValueError, match="duplicate"):
        assert_mixture_expert_invariants(pd.concat([experts, experts.head(1)], ignore_index=True))


# --- normalisation and mixing ----------------------------------------------


def test_normalisation_sums_to_one_per_group_and_leaves_a_dead_group_at_zero():
    values = np.array([1.0, 3.0, 0.0, 0.0])
    codes = np.array([0, 0, 1, 1])
    shares = normalize_within_groups(values, codes, 2)
    assert shares.tolist() == [0.25, 0.75, 0.0, 0.0]


def test_normalisation_treats_negative_and_nonfinite_input_as_no_mass():
    shares = normalize_within_groups(np.array([-2.0, np.nan, 4.0]), np.array([0, 0, 0]), 1)
    assert shares.tolist() == [0.0, 0.0, 1.0]


def test_mixture_matches_a_hand_computed_three_expert_case():
    # Two block groups. prior = (0.8, 0.2), gbm = (0.5, 0.5), exposure = (0.25, 0.75);
    # weights (0.5, 0.25, 0.25) -> (0.5*0.8 + 0.25*0.5 + 0.25*0.25, ...) = (0.5875, 0.4125).
    share = mix_expert_shares(
        prior_share=np.array([0.8, 0.2]),
        gbm_share=np.array([0.5, 0.5]),
        exposure_share=np.array([0.25, 0.75]),
        w_prior=np.array([0.5, 0.5]),
        w_gbm=np.array([0.25, 0.25]),
        w_exposure=np.array([0.25, 0.25]),
        group_codes=np.array([0, 0]),
        n_groups=1,
    )
    assert np.allclose(share, [0.5875, 0.4125], rtol=0.0, atol=1e-12)
    assert share.sum() == pytest.approx(1.0)


def test_mixture_renormalises_when_one_expert_is_dead_over_the_footprint():
    """A footprint whose prior is all zero still gets a valid vector from the live experts."""
    share = mix_expert_shares(
        prior_share=np.array([0.0, 0.0]),
        gbm_share=np.array([0.5, 0.5]),
        exposure_share=np.array([0.2, 0.8]),
        w_prior=np.array([0.5, 0.5]),
        w_gbm=np.array([0.25, 0.25]),
        w_exposure=np.array([0.25, 0.25]),
        group_codes=np.array([0, 0]),
        n_groups=1,
    )
    assert share.sum() == pytest.approx(1.0)
    assert np.allclose(share, [0.35, 0.65])


# --- the model lane ---------------------------------------------------------


def test_legacy_model_lane_is_byte_identical_when_the_mixture_is_off():
    """The default path must produce the pre-change frame, column for column."""
    support = _support(
        ["010010001001", "010010001002", "010010001003"],
        "larceny",
        bg_weight=[3.0, 1.0, 0.0],
        allocation=[1.0, 0.5, 0.25],
    )
    support["bg_weight"] = pd.to_numeric(support["bg_weight"], errors="coerce").fillna(0.0)
    support["allocation_share"] = pd.to_numeric(support["allocation_share"], errors="coerce").fillna(0.0)
    expected = _legacy_model_lane(support)
    actual, summary = _apply_model_lane_shares(support.copy(), mixture=None)
    assert summary is None
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_default_allocation_config_leaves_the_mixture_off():
    assert AllocationBuildConfig().enable_mixture_allocation is False


def test_prior_only_weights_reproduce_the_legacy_share_exactly():
    """`w = (1, 0, 0)` must be the legacy allocator exactly, whatever the other experts say."""
    bg_ids = ["010010001001", "010010001002", "010010001003"]
    support = _support(bg_ids, "larceny", bg_weight=[3.0, 1.0, 4.0], allocation=[1.0, 0.5, 0.25])
    legacy = _legacy_model_lane(support)
    runtime = _runtime(
        _experts(bg_ids, "larceny", population=[10.0, 20.0, 30.0], gbm_weight=[5.0, 50.0, 500.0]),
        _weight_table(),
    )
    mixed, audit = apply_mixture_to_model_lane(legacy, runtime=runtime)
    assert set(audit["stratum"]) == {SMALL_STRATUM}
    assert np.allclose(mixed["model_share"], legacy["model_share"], rtol=0.0, atol=1e-12)


def test_no_shipped_row_is_prior_only_any_more():
    """v2 had two identity cells (larceny suburban/small); v3 has none.

    Recorded as an assertion because it is a real product change and not an accident of the
    selection: after the re-selection there is no offense x stratum where the shipped mixture
    reduces to the legacy allocator, so the model lane moves everywhere.
    """
    weights = load_ship_weights(SHIP_WEIGHTS)
    assert not weights["w_prior"].eq(1.0).any()


def test_mixture_preserves_the_prior_information_mass_and_the_share_identity():
    bg_ids = ["010010001001", "010010001002"]
    support = _legacy_model_lane(_support(bg_ids, "murder", bg_weight=[3.0, 1.0]))
    runtime = _runtime(
        _experts(bg_ids, "murder", population=[100.0, 900.0], gbm_weight=[10.0, 10.0]),
        load_ship_weights(SHIP_WEIGHTS),
    )
    mixed, _ = apply_mixture_to_model_lane(support, runtime=runtime)
    assert np.allclose(mixed["model_total"], support["model_total"], rtol=0.0, atol=0.0)
    assert mixed["model_component_weight"].sum() == pytest.approx(support["model_component_weight"].sum())
    assert np.allclose(
        mixed["model_component_weight"], mixed["model_share"] * mixed["model_total"], rtol=0.0, atol=1e-12
    )
    assert mixed["model_share"].sum() == pytest.approx(1.0)


def test_mixture_reads_the_stratum_from_the_footprint_weighted_population():
    """Halving a footprint's crosswalk share halves its population and can change its stratum."""
    bg_ids = ["010010001001", "010010001002"]
    experts = _experts(bg_ids, "robbery", population=[150_000.0, 150_000.0], gbm_weight=[1.0, 1.0])
    weights = load_ship_weights(SHIP_WEIGHTS)

    whole = _legacy_model_lane(_support(bg_ids, "robbery", bg_weight=[1.0, 1.0]))
    _, audit_whole = apply_mixture_to_model_lane(whole, runtime=_runtime(experts, weights))
    assert set(audit_whole["stratum"]) == {CORE_STRATUM}

    half = _legacy_model_lane(
        _support(bg_ids, "robbery", bg_weight=[1.0, 1.0], allocation=[0.5, 0.5])
    )
    _, audit_half = apply_mixture_to_model_lane(half, runtime=_runtime(experts, weights))
    assert set(audit_half["stratum"]) == {SUBURBAN_STRATUM}


def test_mixture_fails_closed_when_the_expert_table_misses_a_block_group():
    support = _legacy_model_lane(_support(["010010001001", "010010001002"], "burglary", bg_weight=[1.0, 1.0]))
    runtime = _runtime(
        _experts(["010010001001"], "burglary", population=[100.0], gbm_weight=[1.0]),
        load_ship_weights(SHIP_WEIGHTS),
    )
    with pytest.raises(ValueError, match="does not cover 1 model-lane support rows"):
        apply_mixture_to_model_lane(support, runtime=runtime)


def test_mixture_leaves_every_non_model_lane_column_untouched():
    bg_ids = ["010010001001", "010010001002"]
    support = _legacy_model_lane(_support(bg_ids, "rape", bg_weight=[2.0, 6.0]))
    support["city_posterior_share"] = [0.9, 0.1]
    runtime = _runtime(
        _experts(bg_ids, "rape", population=[500.0, 500.0], gbm_weight=[3.0, 1.0]),
        load_ship_weights(SHIP_WEIGHTS),
    )
    mixed, _ = apply_mixture_to_model_lane(support, runtime=runtime)
    for column in ("bg_weight", "allocation_share", "model_total", "city_posterior_share"):
        assert np.allclose(mixed[column], support[column], rtol=0.0, atol=0.0)
    assert not np.allclose(mixed["model_share"], support["model_share"])


# --- audit invariants -------------------------------------------------------


def _audit_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    bg_ids = ["010010001001", "010010001002", "010010001003"]
    support = _legacy_model_lane(_support(bg_ids, "motor_vehicle_theft", bg_weight=[1.0, 2.0, 3.0]))
    runtime = _runtime(
        _experts(bg_ids, "motor_vehicle_theft", population=[100.0, 200.0, 300.0], gbm_weight=[1.0, 1.0, 1.0]),
        load_ship_weights(SHIP_WEIGHTS),
    )
    _, audit = apply_mixture_to_model_lane(support, runtime=runtime)
    return audit, runtime.weights


def test_audit_carries_the_contract_schema_and_passes_its_invariants():
    audit, weights = _audit_fixture()
    assert list(audit.columns) == MIXTURE_SHARE_AUDIT_COLUMNS
    assert_mixture_share_invariants(audit=audit, weights=weights)


def test_invariants_reject_shares_that_do_not_sum_to_one():
    audit, weights = _audit_fixture()
    audit.loc[0, "model_share"] = audit.loc[0, "model_share"] + 0.1
    with pytest.raises(ValueError, match="do not sum to 1"):
        assert_mixture_share_invariants(audit=audit, weights=weights)


def test_invariants_reject_a_weight_that_is_not_the_frozen_table_entry():
    audit, weights = _audit_fixture()
    audit.loc[:, "w_prior"] = 0.3
    audit.loc[:, "w_gbm"] = 0.3
    audit.loc[:, "w_exposure"] = 0.4
    with pytest.raises(ValueError, match="not the frozen table entry"):
        assert_mixture_share_invariants(audit=audit, weights=weights)


def test_invariants_reject_a_share_that_does_not_recompose_from_its_experts():
    audit, weights = _audit_fixture()
    audit.loc[:, "prior_share"] = [0.9, 0.05, 0.05]
    with pytest.raises(ValueError, match="do not recompose"):
        assert_mixture_share_invariants(audit=audit, weights=weights)


def test_invariants_reject_a_negative_share():
    audit, weights = _audit_fixture()
    audit.loc[0, "model_share"] = -0.5
    with pytest.raises(ValueError, match="negative"):
        assert_mixture_share_invariants(audit=audit, weights=weights)


def test_summary_reports_the_divergence_from_the_legacy_lane():
    audit, _ = _audit_fixture()
    summary = summarize_mixture_shares(audit)
    assert summary["support_rows"] == 3
    assert summary["footprints"] == 1
    assert 0.0 <= summary["mean_tvd_vs_legacy"] <= 1.0
    assert set(summary["stratum_row_counts"]) <= set(STRATA)


# --- the re-based exposure expert -------------------------------------------
#
# CONSTANT_RATIFICATIONS.md, "Special-use rank regression — mechanism decision": expert `e` means
# "where the exposure is", and residential population is the wrong measurement of that. The five
# offenses the v2 exposure lane builds an opportunity normalizer for read it; burglary and motor
# vehicle theft keep population because their premises/vehicle surfaces live in the denominator
# lane, not in this artifact. The tests below pin the mechanism (mass lands on resident-less block
# groups), the two documented fallbacks, and the fail-closed check that stops weights selected for
# one expert being applied to the other.


def _normalizer_frame(bg_ids: list[str], **columns: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({"bg_id": bg_ids, **columns})
    return frame.set_index("bg_id")


def test_the_five_ensemble_offenses_read_their_normalizer_and_the_other_two_read_population():
    assert exposure_basis_for_offense("murder") == "person_ens_v1"
    assert exposure_basis_for_offense("rape") == "person_ens_v1"
    assert exposure_basis_for_offense("robbery") == "person_ens_v1"
    assert exposure_basis_for_offense("aggravated_assault") == "person_ens_v1"
    assert exposure_basis_for_offense("larceny") == "larceny_opp_v1"
    assert exposure_basis_for_offense("burglary") == EXPOSURE_BASIS_POPULATION
    assert exposure_basis_for_offense("motor_vehicle_theft") == EXPOSURE_BASIS_POPULATION
    assert set(EXPOSURE_BASIS_BY_OFFENSE) == set(OFFENSES_7)


def test_an_unknown_offense_has_no_exposure_basis():
    with pytest.raises(KeyError, match="unknown offense"):
        exposure_basis_for_offense("arson")


def test_the_expert_puts_real_mass_on_a_resident_less_block_group():
    """The mechanism the diagnosis measured, in one assertion.

    A block group with no residents and substantial daytime opportunity mass gets ~0 share from
    the population expert and a real share from the re-based one.
    """
    bg_ids = ["010010001001", "010010001002"]
    population = np.array([0.0, 1_000.0])
    normalizers = _normalizer_frame(bg_ids, opportunity_normalizer_robbery=[900.0, 1_100.0])
    exposure, basis, fallback = exposure_expert_weights(
        offense="robbery",
        population=population,
        bg_ids=pd.Series(bg_ids),
        normalizers=normalizers,
        config=MixtureAllocationConfig(year=2024),
    )
    assert basis == "person_ens_v1"
    assert not fallback.any()
    share = exposure / exposure.sum()
    assert share[0] == pytest.approx(0.45)
    population_share = np.maximum(population, 1.0) / np.maximum(population, 1.0).sum()
    assert population_share[0] < 1e-3          # what the retired expert said about the same cell


def test_burglary_and_mvt_ignore_the_normalizer_entirely():
    bg_ids = ["010010001001", "010010001002"]
    population = np.array([0.0, 1_000.0])
    normalizers = _normalizer_frame(bg_ids, opportunity_normalizer_robbery=[900.0, 1_100.0])
    for offense in ("burglary", "motor_vehicle_theft"):
        exposure, basis, fallback = exposure_expert_weights(
            offense=offense,
            population=population,
            bg_ids=pd.Series(bg_ids),
            normalizers=normalizers,
            config=MixtureAllocationConfig(year=2024),
        )
        assert basis == EXPOSURE_BASIS_POPULATION
        assert not fallback.any()
        assert np.array_equal(exposure, np.array([1.0, 1_000.0]))


@pytest.mark.parametrize("bad", [0.0, np.nan, -5.0])
def test_a_block_group_the_normalizer_cannot_speak_for_falls_back_to_population_and_is_flagged(bad):
    bg_ids = ["010010001001", "010010001002"]
    population = np.array([400.0, 1_000.0])
    normalizers = _normalizer_frame(bg_ids, opportunity_normalizer_murder=[bad, 1_100.0])
    exposure, _basis, fallback = exposure_expert_weights(
        offense="murder",
        population=population,
        bg_ids=pd.Series(bg_ids),
        normalizers=normalizers,
        config=MixtureAllocationConfig(year=2024),
    )
    assert fallback.tolist() == [True, False]
    assert exposure.tolist() == [400.0, 1_100.0]


def test_a_block_group_absent_from_the_normalizer_surface_falls_back_rather_than_zeroing():
    """An uncovered row must not become an exact zero: that would shrink it out of its footprint."""
    normalizers = _normalizer_frame(["010010001002"], opportunity_normalizer_larceny=[1_100.0])
    exposure, _basis, fallback = exposure_expert_weights(
        offense="larceny",
        population=np.array([400.0, 1_000.0]),
        bg_ids=pd.Series(["010010001001", "010010001002"]),
        normalizers=normalizers,
        config=MixtureAllocationConfig(year=2024),
    )
    assert fallback.tolist() == [True, False]
    assert exposure.tolist() == [400.0, 1_100.0]


def test_the_floor_applies_to_whichever_quantity_was_selected():
    normalizers = _normalizer_frame(["010010001001"], opportunity_normalizer_rape=[0.25])
    exposure, _basis, fallback = exposure_expert_weights(
        offense="rape",
        population=np.array([0.0]),
        bg_ids=pd.Series(["010010001001"]),
        normalizers=normalizers,
        config=MixtureAllocationConfig(year=2024),
    )
    # 0.25 is a real (if tiny) opportunity mass, so the normalizer is used and then floored --
    # not replaced by the population, which is itself zero here.
    assert fallback.tolist() == [False]
    assert exposure.tolist() == [1.0]


def test_a_missing_normalizer_column_is_a_hard_error():
    with pytest.raises(ValueError, match="missing opportunity_normalizer_murder"):
        exposure_expert_weights(
            offense="murder",
            population=np.array([100.0]),
            bg_ids=pd.Series(["010010001001"]),
            normalizers=_normalizer_frame(["010010001001"], opportunity_normalizer_rape=[1.0]),
            config=MixtureAllocationConfig(year=2024),
        )


def test_an_absent_normalizer_surface_is_a_hard_error_rather_than_a_silent_population_expert(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-exposure-normalizers"):
        load_opportunity_normalizers(
            paths=RepoPaths.from_repo_root(tmp_path), config=MixtureAllocationConfig(year=2024)
        )


def test_an_expert_table_built_on_the_retired_population_expert_fails_closed():
    experts = pd.concat(
        [
            _experts([f"0100100010{i:02d}"], offense, population=[100.0], gbm_weight=[100.0])
            for i, offense in enumerate(OFFENSES_7)
        ],
        ignore_index=True,
    )
    assert_mixture_expert_invariants(experts)          # the fixture declares the current basis
    experts.loc[experts["offense"] == "robbery", "exposure_basis"] = EXPOSURE_BASIS_POPULATION
    with pytest.raises(ValueError, match="declares exposure basis"):
        assert_mixture_expert_invariants(experts)


def test_a_row_fallback_on_a_population_basis_offense_is_impossible_and_rejected():
    experts = pd.concat(
        [
            _experts([f"0100100010{i:02d}"], offense, population=[100.0], gbm_weight=[100.0])
            for i, offense in enumerate(OFFENSES_7)
        ],
        ignore_index=True,
    )
    experts.loc[experts["offense"] == "burglary", "exposure_basis_row_fallback"] = True
    with pytest.raises(ValueError, match="row-level exposure fallback"):
        assert_mixture_expert_invariants(experts)
