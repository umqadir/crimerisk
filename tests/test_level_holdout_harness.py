"""The level holdout measures production's rules as configured.

- `mask_share` sets how many agencies each arm scores in the default (combined) runner.
- The remainder holdout uses production's eligibility, municipal reference pool and ratio,
  so its leave-one-state-out prediction is the prediction production makes for a state it
  did not learn from.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk import smoothed_controls  # noqa: E402
from crimerisk.paths import RepoPaths  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "level_holdout_harness_under_test", REPO_ROOT / "scripts" / "eval" / "level_holdout.py"
)
holdout = importlib.util.module_from_spec(_SPEC)
sys.modules["level_holdout_harness_under_test"] = holdout
_SPEC.loader.exec_module(holdout)


# --- #12 mask_share ---------------------------------------------------------------------------


def _config(share: float) -> object:
    return holdout.HoldoutConfig(
        year=2024, folds=10, mask_share=share, seed=20260921, arms=("refused", "silent", "corrupted")
    )


def test_mask_share_sets_the_share_each_arm_scores() -> None:
    eligible = [f"XX{index:07d}" for index in range(3000)]
    for share, passes in ((0.1, 1), (0.2, 2), (0.3, 3)):
        masks = holdout.combined_masks(eligible, config=_config(share))
        assert len(masks) == passes
        for arm in ("refused", "silent", "corrupted"):
            scored = set().union(*(pass_masks[arm] for pass_masks in masks))
            assert abs(len(scored) / len(eligible) - share) < 0.03, (share, arm, len(scored))
        every = [ori for pass_masks in masks for agencies in pass_masks.values() for ori in agencies]
        assert len(every) == len(set(every)), "an agency sits in two arms or two passes"
        # One fold per arm per pass: each estimator pass hides about 30% of agencies.
        for pass_masks in masks:
            hidden = sum(len(agencies) for agencies in pass_masks.values())
            assert abs(hidden / len(eligible) - 0.3) < 0.03


def test_a_mask_share_the_folds_cannot_hold_is_refused() -> None:
    with pytest.raises(ValueError, match="only 10 folds"):
        holdout.combined_masks(["XX0000001"], config=_config(0.4))


# --- #11 the remainder holdout is production's rule -----------------------------------------------


def _controls() -> pd.DataFrame:
    rows = []
    states = {"01": 1.0, "02": 1.5, "03": 0.8, "04": 1.2}
    for state, level in states.items():
        for index, (band, population) in enumerate((("<5k", 4_000.0), ("10k-25k", 20_000.0))):
            rows.append(
                {
                    "jurisdiction_id": f"{state}:municipal:{index}",
                    "state_fips": state,
                    "offense": "burglary",
                    "jurisdiction_type": "municipal",
                    "pop_band": band,
                    "bucket_population": population,
                    "smoothed_count": population * 0.004 * level,
                    "estimator": smoothed_controls.DYNAMIC_SHRINK_ESTIMATOR,
                    "fallback_reason": pd.NA,
                    "clean_year_count": 5,
                }
            )
        # Outside production's reference pool: a large city and a population-less overlay.
        rows.append(
            {
                "jurisdiction_id": f"{state}:municipal:big",
                "state_fips": state,
                "offense": "burglary",
                "jurisdiction_type": "municipal",
                "pop_band": "250k+",
                "bucket_population": 400_000.0,
                "smoothed_count": 400_000.0 * 0.02,
                "estimator": smoothed_controls.DYNAMIC_SHRINK_ESTIMATOR,
                "fallback_reason": pd.NA,
                "clean_year_count": 5,
            }
        )
        rows.append(
            {
                "jurisdiction_id": f"{state}:statewide_overlap_layer",
                "state_fips": state,
                "offense": "burglary",
                "jurisdiction_type": "statewide_overlap_layer",
                "pop_band": "0",
                "bucket_population": 0.0,
                "smoothed_count": 500.0,
                "estimator": smoothed_controls.FALLBACK_ACCOUNTING_ESTIMATOR,
                "fallback_reason": smoothed_controls.INSUFFICIENT_HISTORY_REASON,
                "clean_year_count": 0,
            }
        )
        rows.append(
            {
                "jurisdiction_id": f"{state}:remainder",
                "state_fips": state,
                "offense": "burglary",
                "jurisdiction_type": smoothed_controls.STATE_REMAINDER_TYPE,
                "pop_band": "250k+",
                "bucket_population": 300_000.0,
                "smoothed_count": 300_000.0 * 0.003 * level * (1.0 + 0.1 * int(state)),
                "estimator": smoothed_controls.DYNAMIC_SHRINK_ESTIMATOR,
                "fallback_reason": pd.NA,
                "clean_year_count": 5,
            }
        )
    # A remainder production would not learn from: one clean year.
    rows.append(
        {
            "jurisdiction_id": "05:remainder",
            "state_fips": "05",
            "offense": "burglary",
            "jurisdiction_type": smoothed_controls.STATE_REMAINDER_TYPE,
            "pop_band": "250k+",
            "bucket_population": 300_000.0,
            "smoothed_count": 9_000.0,
            "estimator": smoothed_controls.DYNAMIC_SHRINK_ESTIMATOR,
            "fallback_reason": pd.NA,
            "clean_year_count": 1,
        }
    )
    return pd.DataFrame(rows)


def test_the_remainder_holdout_predicts_what_production_would_for_an_unseen_state(tmp_path) -> None:
    controls = _controls()
    paths = RepoPaths.from_repo_root(tmp_path)
    (paths.state_dir / "controls").mkdir(parents=True)
    controls.assign(precalibration_smoothed_count=controls["smoothed_count"]).to_parquet(
        paths.state_dir / "controls" / "jurisdiction_controls_smoothed_2024.parquet", index=False
    )
    scored = holdout.run_remainder_holdout(paths, year=2024).set_index("state_fips")
    assert set(scored.index) == {"01", "02", "03", "04"}

    for state in ("01", "02", "03", "04"):
        # Production, with this state's remainder unlearnable (no history), predicts it from the
        # other states' ratio: that is the number the holdout must reproduce.
        hidden = controls.copy()
        row = hidden["jurisdiction_id"].eq(f"{state}:remainder")
        hidden.loc[row, "estimator"] = smoothed_controls.FALLBACK_ACCOUNTING_ESTIMATOR
        hidden.loc[row, "fallback_reason"] = smoothed_controls.INSUFFICIENT_HISTORY_REASON
        hidden.loc[row, "clean_year_count"] = 0
        hidden.loc[row, "smoothed_count"] = 1.0  # what little was reported: production lifts it
        panel = hidden[["jurisdiction_id", "offense"]].assign(year=2024, observation_weight_preferred=0.0)
        floored = smoothed_controls._apply_state_remainder_coverage_floor(
            hidden, panel=panel, config=smoothed_controls.SmoothedControlConfig(year=2024)
        ).set_index("jurisdiction_id")
        production = floored.loc[f"{state}:remainder", "coverage_peer_predicted_count"]
        assert np.isfinite(production)
        assert scored.loc[state, "predicted_count"] == pytest.approx(production, rel=1e-12)
