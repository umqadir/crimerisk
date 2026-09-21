from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnostics" / "build_feature_transfer_policy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_feature_transfer_policy_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_annual_acs_policy_groups_are_vintage_invariant() -> None:
    module = _load_module()
    for year in (2024, 2025, 2030):
        assert module._policy_group_for_feature(f"median_household_income_{year}") == "income_wealth"
        assert module._policy_group_for_feature(f"per_capita_income_{year}") == "income_wealth"
        assert module._policy_group_for_feature(f"median_home_value_{year}") == "income_wealth"
        assert module._policy_group_for_feature(f"median_rent_{year}") == "income_wealth"
        assert module._policy_group_for_feature(f"pop_{year}") == "population_density"
        assert module._policy_group_for_feature(f"pop_factor_{year}") == "population_density"
