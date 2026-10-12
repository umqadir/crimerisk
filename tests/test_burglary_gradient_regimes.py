"""The burglary commercial-share gradient blocks on modeled mass, reports direct-feed mass."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "validate_release_outputs.py"
_SPEC = importlib.util.spec_from_file_location("validate_release_outputs_gradient", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules["validate_release_outputs_gradient"] = validator
_SPEC.loader.exec_module(validator)


def _findings(direct, modeled):
    issues: list[str] = []
    reported = validator._burglary_commercial_gradient_findings(
        {"after_q5_q1_mean_direct": direct, "after_q5_q1_mean_modeled": modeled}, issues=issues
    )
    return issues, reported


def test_direct_city_gradient_outside_band_is_reported_not_blocking():
    issues, reported = _findings(1.3093348086682461, 1.15)
    assert issues == []
    assert len(reported) == 1 and "direct-city" in reported[0] and "1.309" in reported[0]


def test_modeled_transfer_gradient_outside_band_blocks():
    issues, reported = _findings(1.15, validator.BURGLARY_COMMERCIAL_GRADIENT_MODELED_MAX + 0.01)
    assert reported == []
    assert len(issues) == 1 and "modeled-transfer" in issues[0]


def test_missing_gradient_blocks_in_either_regime():
    issues, reported = _findings(None, None)
    assert reported == []
    assert len(issues) == 2 and all("missing from build manifest" in i for i in issues)


def test_both_inside_band_is_clean():
    assert _findings(1.1, 1.2) == ([], [])
