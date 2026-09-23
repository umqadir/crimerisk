"""The level-lane holdout must not let a hidden outcome inform its own prediction.

The harness scores a ladder-dropped agency as the pooled silent-unit control its
absence triggers. That pool is built from a panel, and if the panel is the unmasked
one the held-out agency is in its own donor pool: a large masked agency in a small
state pool then predicts itself almost one for one. These tests pin the invariance --
changing a hidden target-year count must not move the prediction for that agency --
on both the fallback itself and the runner that calls it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "level_holdout_under_test", REPO_ROOT / "scripts" / "eval" / "level_holdout.py"
)
holdout = importlib.util.module_from_spec(_SPEC)
sys.modules["level_holdout_under_test"] = holdout
_SPEC.loader.exec_module(holdout)


HELD_OUT = "XX0000001"
TRAINING = ("XX0000002", "XX0000003")


def _panel(held_out_count: float) -> pd.DataFrame:
    """One state, one offense: a large held-out agency and two small training reporters.

    The training reporters carry 100 per 100k between them. The held-out agency is ten
    times either of them, so if it reaches the pool it sets the pool's rate.
    """
    rows = [
        {
            "ori9": HELD_OUT,
            "year": 2024,
            "offense": "burglary",
            "preferred_count": float(held_out_count),
            "population": 100_000.0,
            "usable_as_observed": True,
            "level1_admission_status": "valid_complete_year",
            "state_fips": "99",
        }
    ]
    rows += [
        {
            "ori9": ori,
            "year": 2024,
            "offense": "burglary",
            "preferred_count": 50.0,
            "population": 50_000.0,
            "usable_as_observed": True,
            "level1_admission_status": "valid_complete_year",
            "state_fips": "99",
        }
        for ori in TRAINING
    ]
    return pd.DataFrame(rows)


def _dropped_row() -> pd.DataFrame:
    """The scored row of an agency the repair ladder declined to estimate."""
    return pd.DataFrame(
        [
            {
                "ori9": HELD_OUT,
                "offense": "burglary",
                "state_fips": "99",
                "observed_count": np.nan,
                "population": 100_000.0,
                "predicted_count": np.nan,
            }
        ]
    )


def _backfilled(panel: pd.DataFrame, *, exclude: set[str] | None) -> float:
    out = holdout.pooled_silent_unit_backfill(
        _dropped_row(), panel=panel, year=2024, exclude_oris=exclude
    )
    return float(out["predicted_count"].iloc[0])


def test_hidden_outcome_cannot_move_its_own_pooled_prediction():
    low = _backfilled(_panel(1_000.0), exclude={HELD_OUT})
    high = _backfilled(_panel(2_000.0), exclude={HELD_OUT})
    assert low == high


def test_the_pooled_prediction_is_the_training_reporters_rate():
    # 100 per 100k over the two training reporters, on the held-out agency's 100k.
    assert _backfilled(_panel(1_000.0), exclude={HELD_OUT}) == 100.0
    training_only = _panel(1_000.0)
    training_only = training_only[training_only["ori9"].ne(HELD_OUT)]
    assert _backfilled(training_only, exclude=None) == 100.0


def test_a_masked_panel_alone_already_withholds_the_held_out_row():
    """The `refused` and `silent` masks take the row out of the pool by themselves.

    The explicit exclusion is the belt to that pair of braces, and the brace the
    corrupted arm does not have.
    """
    silent = holdout.mask_panel(
        _panel(1_000.0), year=2024, masked_oris={HELD_OUT}, arm="silent"
    )
    refused = holdout.mask_panel(
        _panel(1_000.0), year=2024, masked_oris={HELD_OUT}, arm="refused"
    )
    assert _backfilled(silent, exclude=None) == 100.0
    assert _backfilled(refused, exclude=None) == 100.0


def test_the_unmasked_panel_is_what_leaks():
    """The defect this guards against, stated as the behaviour it produced.

    Kept as a test so that a future refactor cannot quietly reintroduce the leaky
    call by passing the unmasked panel: the numbers here are the ones the fallback
    returns when the held-out agency is left in its own pool.
    """
    assert _backfilled(_panel(1_000.0), exclude=None) == 550.0
    assert _backfilled(_panel(2_000.0), exclude=None) == 1050.0


def test_the_runner_backfills_against_the_fold_it_masked(monkeypatch):
    """End to end through `run_agency_holdout_combined`, with the estimator stubbed.

    The estimator returns nothing, so every masked row falls to the pooled control --
    which is the path the leak lived on. The runner must reach that control through
    the panel it admitted, not the one it was handed.
    """
    captured: dict[str, pd.DataFrame] = {}

    def fake_admit(paths, *, panel, year):
        captured["admitted"] = panel
        return panel

    def fake_estimates(*, paths, year, agency_panel):
        return pd.DataFrame(
            columns=["ori9", "offense", "estimated_count", "agency_estimate_source"]
        )

    monkeypatch.setattr(holdout, "admit_panel", fake_admit)
    monkeypatch.setattr(
        holdout, "build_agency_allocation_target_estimates", fake_estimates
    )

    config = holdout.HoldoutConfig(
        year=2024, folds=3, mask_share=1.0, seed=1, arms=("silent",)
    )
    truth = pd.DataFrame(
        [
            {
                "ori9": ori,
                "offense": "burglary",
                "state_abbr": "XX",
                "state_fips": "99",
                "observed_count": 1_000.0 if ori == HELD_OUT else 50.0,
                "population": 100_000.0 if ori == HELD_OUT else 50_000.0,
            }
            for ori in (HELD_OUT, *TRAINING)
        ]
    )

    predictions = {}
    for hidden in (1_000.0, 2_000.0):
        panel = _panel(hidden)
        scored = holdout.run_agency_holdout_combined(
            paths=None,
            panel=panel,
            preferred=panel,
            config=config,
            truth=truth,
            eligible_oris=[HELD_OUT, *TRAINING],
            label="invariance",
        )
        masked = scored[scored["ori9"].eq(HELD_OUT)]
        if masked.empty:
            continue
        predictions[hidden] = float(masked["predicted_count"].iloc[0])

    assert len(predictions) == 2, "the held-out agency was never in the masked fold"
    assert predictions[1_000.0] == predictions[2_000.0]
