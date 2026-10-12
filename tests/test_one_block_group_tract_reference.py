"""A tract with exactly one published block group publishes that block group's index.

Both supports index against one national reference per offense and measure, frozen before any
publication mask (tests/test_national_reference.py pins the producer side). The release validator
checks it on the published surfaces and persists the numbers in the validation summary.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "validate_release_outputs.py"
_SPEC = importlib.util.spec_from_file_location("validate_release_outputs_one_bg_tract", SCRIPT)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)

OFFENSES = validator.OFFENSES_7
RARE = set(validator.RARE_OFFENSE_TRACT_SUPPORT)
SINGLE_TRACT, SINGLE_BG = "01047956702", "010479567021"
PAIR_TRACT = "01047956800"


def _surfaces() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tract 01047956702 is one block group; tract 01047956800 has two."""
    bg = pd.DataFrame({"block_group_geoid": [SINGLE_BG, "010479568001", "010479568002"]})
    tract = pd.DataFrame({"tract_id": [SINGLE_TRACT, PAIR_TRACT]})
    for position, offense in enumerate(OFFENSES):
        for measure in ("primary", "resident"):
            column = f"index_{offense}_{measure}"
            value = 50.0 + 10.0 * position + (5.0 if measure == "resident" else 0.0)
            # The rare offenses' block-group point fields are null by policy.
            bg[column] = np.nan if offense in RARE else [value, value * 2.0, np.nan]
            tract[column] = [value, value * 1.7]
    return bg, tract


def _check(tmp_path: Path, bg: pd.DataFrame, tract: pd.DataFrame) -> tuple[list[str], dict]:
    bg.to_parquet(tmp_path / f"crimerisk_block_group_{validator.YEAR}_ags_core.parquet", index=False)
    tract.to_parquet(tmp_path / f"crimerisk_tract_{validator.YEAR}_ags_core.parquet", index=False)
    issues: list[str] = []
    summary = validator._check_one_block_group_tracts(output_dir=tmp_path, issues=issues)
    return issues, summary


def test_equal_indexes_pass_and_persist_their_numbers(tmp_path):
    issues, summary = _check(tmp_path, *_surfaces())
    assert issues == []
    assert summary["status"] == "checked"
    assert summary["one_block_group_tracts"] == 1
    # Five volume offenses x two measures; the two-block-group tract is not compared.
    assert summary["cells_compared"] == 10
    assert summary["cells_over_tolerance"] == 0
    assert summary["max_abs_difference"] == 0.0
    assert summary["by_measure"]["primary"]["cells_compared"] == 5
    assert set(summary["not_compared"]) == {
        f"index_{offense}_{measure}" for offense in RARE for measure in ("primary", "resident")
    }
    assert len(summary["worst"]) == 10
    assert {row["tract_id"] for row in summary["worst"]} == {SINGLE_TRACT}


def test_a_difference_beyond_tolerance_blocks_and_is_the_worst_cell(tmp_path):
    bg, tract = _surfaces()
    tract.loc[0, "index_larceny_resident"] = bg.loc[0, "index_larceny_resident"] * 1.001
    issues, summary = _check(tmp_path, bg, tract)
    assert len(issues) == 1 and issues[0].startswith("one_block_group_tract_reference: 1 of 10")
    assert summary["cells_over_tolerance"] == 1
    worst = summary["worst"][0]
    assert (worst["offense"], worst["measure"], worst["block_group_geoid"]) == ("larceny", "resident", SINGLE_BG)
    assert summary["max_abs_difference"] == pytest.approx(bg.loc[0, "index_larceny_resident"] * 0.001)
    assert worst["rel_difference"] == pytest.approx(0.001)
    assert summary["by_measure"]["resident"]["cells_over_tolerance"] == 1
    assert summary["by_measure"]["primary"]["cells_over_tolerance"] == 0


def test_a_difference_within_tolerance_is_reported_not_blocking(tmp_path):
    bg, tract = _surfaces()
    tract.loc[0, "index_burglary_primary"] = bg.loc[0, "index_burglary_primary"] * (1.0 + 1e-8)
    issues, summary = _check(tmp_path, bg, tract)
    assert issues == []
    assert summary["max_abs_difference"] > 0.0
    assert summary["worst"][0]["offense"] == "burglary"


def test_a_cell_published_on_one_support_only_is_counted_not_compared(tmp_path):
    bg, tract = _surfaces()
    tract.loc[0, "index_robbery_primary"] = np.nan
    issues, summary = _check(tmp_path, bg, tract)
    assert issues == []
    assert summary["cells_compared"] == 9
    assert summary["published_on_one_support_only"] == 1
    assert summary["published_on_one_support_only_sample"][0]["offense"] == "robbery"


def test_missing_surfaces_are_not_available(tmp_path):
    issues: list[str] = []
    summary = validator._check_one_block_group_tracts(output_dir=tmp_path, issues=issues)
    assert issues == [] and summary["status"] == "not_available"
