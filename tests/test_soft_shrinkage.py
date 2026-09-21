"""Soft shrinkage replaces the hard envelope caps (contract: SOFT_SHRINKAGE_CONTRACT.md).

Six things are load-bearing and are asserted here: the compressor keeps ordering and never
truncates to a constant, it is the identity inside the envelope, the extrapolation term shrinks
toward the bound (which is the exposure-share null), the expert table's contract-aware invariants
replace containment with a real mirror rather than an exemption, a mode mismatch between the expert
table and the build fails closed in BOTH directions, and every path is byte-identical to the
pre-lane code when the flag is off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    AllocationBuildConfig,
    MODEL_ONLY_ALLOCATION_ENVELOPE_MODE_COLUMN,
    MODEL_ONLY_ROBBERY_BG_RATE_RATIO_CAP,
    _apply_model_only_allocation_envelopes,
    _softened_retained_rate_ratio,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.mixture_allocation import (
    ENVELOPE_MODE_SOFT_SHRINKAGE,
    SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT,
    SOFT_SHRINKAGE_NU,
    SOFT_SHRINKAGE_EXPERT_COLUMNS,
    MixtureAllocationConfig,
    _soft_shrinkage_identity,
    apply_envelope,
    assert_mixture_expert_invariants,
    clip_to_envelope,
    compress_log_excess,
    expert_table_is_soft_shrunk,
    exposure_basis_for_offense,
    extrapolation_flags,
    resolve_mixture_runtime,
    ship_weights_path,
    soft_shrink_to_envelope,
    soft_shrinkage_record,
)
from crimerisk.paths import RepoPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
LO, HI = 3.0, 50.0


# --- the compressor ---------------------------------------------------------


def test_compressor_is_the_identity_to_first_order_and_logarithmic_far_out():
    assert compress_log_excess(np.array([0.0]), nu=2.0) == pytest.approx(0.0)
    tiny = 1e-6
    assert compress_log_excess(np.array([tiny]), nu=2.0)[0] == pytest.approx(tiny, rel=1e-5)
    far = compress_log_excess(np.array([100.0]), nu=2.0)[0]
    assert far == pytest.approx(2.0 * np.log1p(50.0))
    assert far < 100.0


@pytest.mark.parametrize("nu", [0.0, -1.0, float("nan"), float("inf")])
def test_compressor_rejects_a_nu_that_is_not_finite_and_positive(nu):
    with pytest.raises(ValueError, match="nu"):
        compress_log_excess(np.array([1.0]), nu=nu)


def test_compressor_rejects_a_negative_excess():
    with pytest.raises(ValueError, match="non-negative"):
        compress_log_excess(np.array([-0.1]), nu=2.0)


def test_soft_shrinkage_leaves_the_inside_of_the_envelope_bit_identical():
    inside = np.array([3.0, 3.0000001, 10.0, 49.999, 50.0])
    out = soft_shrink_to_envelope(inside, LO, HI, nu=SOFT_SHRINKAGE_NU)
    assert np.array_equal(out, inside)


def test_soft_shrinkage_preserves_strict_ordering_where_the_clip_destroys_it():
    tail = np.array([60.0, 200.0, 2_000.0, 20_000.0])
    clipped = clip_to_envelope(tail, LO, HI)
    soft = soft_shrink_to_envelope(tail, LO, HI, nu=SOFT_SHRINKAGE_NU)
    # the clip collapses the whole tail onto one number...
    assert len(set(clipped.tolist())) == 1
    # ...the compressor keeps every distinction, strictly.
    assert np.all(np.diff(soft) > 0.0)
    # and it is still a shrinkage: above the bound, below the raw prediction.
    assert np.all(soft >= HI)
    assert np.all(soft < tail)


def test_soft_shrinkage_compresses_the_lower_tail_symmetrically():
    below = np.array([0.5, 1.0, 2.0])
    soft = soft_shrink_to_envelope(below, LO, HI, nu=SOFT_SHRINKAGE_NU)
    assert np.all(np.diff(soft) > 0.0)
    assert np.all(soft <= LO)
    assert np.all(soft > below)


def test_a_zero_prediction_stays_zero_rather_than_being_lifted_to_the_floor():
    assert soft_shrink_to_envelope(np.array([0.0]), LO, HI, nu=SOFT_SHRINKAGE_NU)[0] == 0.0
    assert clip_to_envelope(np.array([0.0]), LO, HI)[0] == LO


def test_a_small_nu_approaches_the_hard_clip_and_a_large_nu_approaches_no_shrinkage():
    tail = np.array([5_000.0])
    nearly_clipped = soft_shrink_to_envelope(tail, LO, HI, nu=1e-6)[0]
    nearly_untouched = soft_shrink_to_envelope(tail, LO, HI, nu=1e9)[0]
    assert nearly_clipped == pytest.approx(HI, rel=1e-3)
    assert nearly_untouched == pytest.approx(tail[0], rel=1e-3)


def test_extrapolation_weight_shrinks_the_excess_toward_the_bound():
    tail = np.array([5_000.0])
    flag = np.array([True])
    plain = soft_shrink_to_envelope(tail, LO, HI, nu=SOFT_SHRINKAGE_NU)[0]
    half = soft_shrink_to_envelope(
        tail, LO, HI, nu=SOFT_SHRINKAGE_NU, extrapolation_weight=0.5, extrapolating=flag
    )[0]
    none = soft_shrink_to_envelope(
        tail, LO, HI, nu=SOFT_SHRINKAGE_NU, extrapolation_weight=0.0, extrapolating=flag
    )[0]
    full = soft_shrink_to_envelope(
        tail, LO, HI, nu=SOFT_SHRINKAGE_NU, extrapolation_weight=1.0, extrapolating=flag
    )[0]
    assert none == plain
    assert HI < half < plain
    # weight 1 lands exactly on the bound -- a constant rate across the offense, which is
    # `gbm_share == exposure_share`: the exposure-share null the plan item names.
    assert full == pytest.approx(HI)


def test_extrapolation_weight_touches_only_the_flagged_rows():
    tail = np.array([5_000.0, 5_000.0])
    out = soft_shrink_to_envelope(
        tail, LO, HI, nu=SOFT_SHRINKAGE_NU, extrapolation_weight=0.75,
        extrapolating=np.array([True, False]),
    )
    assert out[0] < out[1]
    assert out[1] == soft_shrink_to_envelope(tail[1:], LO, HI, nu=SOFT_SHRINKAGE_NU)[0]


def test_soft_shrinkage_rejects_a_bad_envelope_weight_or_flag():
    with pytest.raises(ValueError, match="inverted"):
        soft_shrink_to_envelope(np.array([1.0]), 10.0, 1.0, nu=2.0)
    with pytest.raises(ValueError, match="non-finite"):
        soft_shrink_to_envelope(np.array([1.0]), 0.0, float("nan"), nu=2.0)
    with pytest.raises(ValueError, match="weight"):
        soft_shrink_to_envelope(np.array([1.0]), LO, HI, nu=2.0, extrapolation_weight=1.5)
    with pytest.raises(ValueError, match="shaped"):
        soft_shrink_to_envelope(
            np.array([1.0, 2.0]), LO, HI, nu=2.0, extrapolating=np.array([True])
        )


def test_apply_envelope_dispatches_on_the_config_and_the_hard_path_is_the_clip():
    rates = np.array([0.5, 10.0, 900.0])
    hard = apply_envelope(rates, LO, HI, config=MixtureAllocationConfig())
    assert np.array_equal(hard, clip_to_envelope(rates, LO, HI))
    soft = apply_envelope(
        rates, LO, HI, config=MixtureAllocationConfig(enable_soft_shrinkage=True)
    )
    assert np.array_equal(
        soft,
        soft_shrink_to_envelope(
            rates, LO, HI, nu=SOFT_SHRINKAGE_NU,
            extrapolation_weight=SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT,
        ),
    )


# --- the extrapolation hull proxy -------------------------------------------


def test_hull_flag_is_any_governed_feature_outside_the_training_range():
    features = np.array([[1.0, 2.0], [5.0, 2.0], [1.0, 9.0], [5.0, 9.0]])
    flag, count = extrapolation_flags(
        features, feature_low=np.array([0.0, 1.0]), feature_high=np.array([3.0, 5.0])
    )
    assert flag.tolist() == [False, True, True, True]
    assert count.tolist() == [0, 1, 1, 2]
    assert np.array_equal(flag, count > 0)


def test_a_missing_feature_is_not_an_extrapolation():
    flag, count = extrapolation_flags(
        np.array([[np.nan, 2.0]]), feature_low=np.array([0.0, 1.0]), feature_high=np.array([3.0, 5.0])
    )
    assert not bool(flag[0])
    assert int(count[0]) == 0


def test_hull_bounds_must_match_the_feature_matrix():
    with pytest.raises(ValueError, match="hull bounds"):
        extrapolation_flags(
            np.array([[1.0, 2.0]]), feature_low=np.array([0.0]), feature_high=np.array([3.0])
        )


# --- the expert table's contract-aware invariants ---------------------------


def _expert_frame(*, soft: bool, rate: list[float], bounded: list[float]) -> pd.DataFrame:
    n = len(rate)
    frame = pd.DataFrame(
        {
            "state_fips": ["01"] * n,
            "bg_id": pd.array([f"0100100000{i:02d}"[:12] for i in range(n)], dtype="string"),
            "offense": pd.array(["robbery"] * n, dtype="string"),
            "population": [100.0] * n,
            "exposure_weight": [100.0] * n,
            "exposure_basis": exposure_basis_for_offense("robbery"),
            "exposure_basis_row_fallback": [False] * n,
            "gbm_rate": rate,
            "gbm_rate_clipped": bounded,
            "gbm_weight": [value * 100.0 for value in bounded],
            "envelope_lo": LO,
            "envelope_hi": HI,
        }
    )
    frames = [frame]
    for offense in OFFENSES_7:
        if offense == "robbery":
            continue
        other = frame.copy()
        other["offense"] = pd.array([offense] * n, dtype="string")
        other["bg_id"] = pd.array([f"{offense[:2]}{i:010d}" for i in range(n)], dtype="string")
        other["exposure_basis"] = exposure_basis_for_offense(offense)
        frames.append(other)
    out = pd.concat(frames, ignore_index=True)
    if soft:
        out["envelope_mode"] = ENVELOPE_MODE_SOFT_SHRINKAGE
        out["extrapolation_flag"] = [False] * len(out)
        out["extrapolation_feature_count"] = np.zeros(len(out), dtype=np.int32)
    return out


def test_a_well_formed_soft_table_passes_and_declares_itself():
    raw = [0.0, 10.0, 5_000.0]
    frame = _expert_frame(
        soft=True,
        rate=raw,
        bounded=list(soft_shrink_to_envelope(np.array(raw), LO, HI, nu=SOFT_SHRINKAGE_NU)),
    )
    assert expert_table_is_soft_shrunk(frame)
    assert_mixture_expert_invariants(frame)


def test_the_hard_contract_still_rejects_a_rate_outside_the_envelope():
    frame = _expert_frame(soft=False, rate=[10.0, 900.0], bounded=[10.0, 900.0])
    with pytest.raises(ValueError, match="outside its learned envelope"):
        assert_mixture_expert_invariants(frame)


def test_the_soft_contract_rejects_moving_a_rate_that_was_inside_the_envelope():
    frame = _expert_frame(soft=True, rate=[10.0, 900.0], bounded=[9.0, 200.0])
    with pytest.raises(ValueError, match="inside its envelope"):
        assert_mixture_expert_invariants(frame)


def test_the_soft_contract_rejects_a_treatment_that_overshoots_its_bound():
    frame = _expert_frame(soft=True, rate=[900.0], bounded=[HI - 1.0])
    with pytest.raises(ValueError, match=r"above-envelope"):
        assert_mixture_expert_invariants(frame)
    frame = _expert_frame(soft=True, rate=[900.0], bounded=[1_000.0])
    with pytest.raises(ValueError, match=r"above-envelope"):
        assert_mixture_expert_invariants(frame)


def test_the_soft_contract_rejects_a_flag_that_disagrees_with_its_own_count():
    frame = _expert_frame(soft=True, rate=[10.0], bounded=[10.0])
    frame["extrapolation_flag"] = True
    with pytest.raises(ValueError, match="disagrees with its own count"):
        assert_mixture_expert_invariants(frame)


def test_a_partial_soft_column_set_is_a_hard_error():
    frame = _expert_frame(soft=True, rate=[10.0], bounded=[10.0]).drop(columns=["extrapolation_flag"])
    with pytest.raises(ValueError, match="partial soft-shrinkage column set"):
        expert_table_is_soft_shrunk(frame)


def test_an_unexpected_envelope_mode_is_a_hard_error():
    frame = _expert_frame(soft=True, rate=[10.0], bounded=[10.0])
    frame["envelope_mode"] = "clip_but_pretty"
    with pytest.raises(ValueError, match="unexpected envelope modes"):
        expert_table_is_soft_shrunk(frame)


@pytest.mark.parametrize("build_soft", [True, False])
def test_a_mode_mismatch_between_table_and_build_fails_closed_both_ways(tmp_path, build_soft):
    raw = [10.0, 900.0]
    table_soft = not build_soft
    frame = _expert_frame(
        soft=table_soft,
        rate=raw,
        bounded=(
            list(soft_shrink_to_envelope(np.array(raw), LO, HI, nu=SOFT_SHRINKAGE_NU))
            if table_soft
            else list(clip_to_envelope(np.array(raw), LO, HI))
        ),
    )
    path = tmp_path / "experts.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="asked for"):
        resolve_mixture_runtime(
            paths=PATHS,
            config=MixtureAllocationConfig(enable_soft_shrinkage=build_soft),
            experts_path=path,
            weights_path=ship_weights_path(PATHS),
        )


# --- freshness and the self-identifying record ------------------------------


def test_a_pre_lane_summary_reads_as_a_disabled_lane_so_legacy_artifacts_stay_current():
    assert _soft_shrinkage_identity(None) == _soft_shrinkage_identity(
        soft_shrinkage_record(MixtureAllocationConfig())
    )


def test_changing_a_constant_changes_the_freshness_identity():
    base = soft_shrinkage_record(MixtureAllocationConfig(enable_soft_shrinkage=True))
    moved = soft_shrinkage_record(
        MixtureAllocationConfig(enable_soft_shrinkage=True, soft_shrinkage_nu=SOFT_SHRINKAGE_NU + 1)
    )
    assert _soft_shrinkage_identity(base) != _soft_shrinkage_identity(moved)


def test_the_record_names_the_selected_constants_and_its_contract():
    record = soft_shrinkage_record(MixtureAllocationConfig(enable_soft_shrinkage=True))
    assert record["enabled"] is True
    assert record["envelope_mode"] == ENVELOPE_MODE_SOFT_SHRINKAGE
    assert record["nu"] == SOFT_SHRINKAGE_NU
    assert record["extrapolation_weight"] == SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT
    assert record["contract"].endswith("SOFT_SHRINKAGE_CONTRACT.md")


# --- the model-only rate-ratio caps (the second hard envelope) --------------


def test_the_retained_rate_ratio_is_untouched_at_or_below_the_cap():
    ratio = pd.Series([0.5, 1.0, 21.9, 22.0])
    out = _softened_retained_rate_ratio(ratio, cap_ratio=22.0, nu=SOFT_SHRINKAGE_NU)
    assert np.allclose(out.to_numpy(), ratio.to_numpy())


def test_the_retained_rate_ratio_compresses_above_the_cap_and_keeps_its_ordering():
    ratio = pd.Series([30.0, 100.0, 1_000.0])
    out = _softened_retained_rate_ratio(ratio, cap_ratio=22.0, nu=SOFT_SHRINKAGE_NU).to_numpy()
    assert np.all(out > 22.0)
    assert np.all(out < ratio.to_numpy())
    assert np.all(np.diff(out) > 0.0)


def _components() -> pd.DataFrame:
    """One robbery jurisdiction: a screaming tail unit, and two ordinary recipients."""
    rows = []
    for index, (bg, count, denominator) in enumerate(
        [("010010000001", 900.0, 1_000.0), ("010010000002", 10.0, 1_000.0), ("010010000003", 10.0, 1_000.0)]
    ):
        rows.append(
            {
                "bg_id": bg,
                "tract_id": bg[:11],
                "offense": "robbery",
                "jurisdiction_id": "01:municipal:place:0100100",
                "jurisdiction_type": "municipal",
                "state_fips": "01",
                "component_count": count,
                "component_share": count / 920.0,
                "primary_denominator_raw": denominator,
                "households_total": 400.0,
                "city_incident_posterior_active": False,
                "_index": index,
            }
        )
    return pd.DataFrame(rows).drop(columns=["_index"])


def test_the_hard_cap_path_is_byte_identical_to_the_default():
    components = _components()
    default = _apply_model_only_allocation_envelopes(components)
    explicit = _apply_model_only_allocation_envelopes(components, soft_shrinkage=False)
    pd.testing.assert_frame_equal(default, explicit)
    assert MODEL_ONLY_ALLOCATION_ENVELOPE_MODE_COLUMN not in default.columns
    assert AllocationBuildConfig().enable_soft_shrinkage is False
    assert AllocationBuildConfig().enable_imputation_v2 is False


def test_softening_removes_strictly_less_mass_from_the_tail_and_still_conserves():
    components = _components()
    hard = _apply_model_only_allocation_envelopes(components)
    soft = _apply_model_only_allocation_envelopes(
        components, soft_shrinkage=True, soft_shrinkage_nu=SOFT_SHRINKAGE_NU
    )
    assert hard["allocation_envelope_clipped_source"].any()
    assert soft["allocation_envelope_clipped_source"].any()
    tail = components["component_count"].idxmax()
    assert hard.loc[tail, "component_count"] < soft.loc[tail, "component_count"] < 900.0
    for frame in (hard, soft):
        assert frame["component_count"].sum() == pytest.approx(
            components["component_count"].sum()
        )
    assert set(soft[MODEL_ONLY_ALLOCATION_ENVELOPE_MODE_COLUMN]) == {ENVELOPE_MODE_SOFT_SHRINKAGE}
    assert MODEL_ONLY_ROBBERY_BG_RATE_RATIO_CAP == 22.0


def test_the_soft_columns_are_absent_from_a_hard_expert_build():
    frame = _expert_frame(soft=False, rate=[10.0], bounded=[10.0])
    assert not any(column in frame.columns for column in SOFT_SHRINKAGE_EXPERT_COLUMNS)
    assert expert_table_is_soft_shrunk(frame) is False
