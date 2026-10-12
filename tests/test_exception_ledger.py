import importlib.util
import sys
from pathlib import Path

import pandas as pd

BUILD = Path(__file__).parents[1] / "frontend" / "build"
sys.path.insert(0, str(BUILD))
SPEC = importlib.util.spec_from_file_location("exception_ledger", BUILD / "05b_exception_ledger.py")
LEDGER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LEDGER)


def block_groups(rows):
    return pd.DataFrame(
        [
            {"block_group_geoid": g, "eb_jurisdiction_id": j, LEDGER.OVERALL_EXPOSURE: v}
            for g, j, v in rows
        ]
    )


def totals():
    return pd.DataFrame(
        {
            "level_repair_mode": ["none", "pooled_silent_unit"],
            "total_latest_clean_year": [2025, 2019],
            "total_clean_years": [8, 3],
        },
        index=["J1", "J2"],
    )


def test_ledger_keeps_only_published_cross_jurisdiction_pairs_at_or_above_the_threshold():
    bg = block_groups([
        ("A", "J1", 400.0),
        ("B", "J2", 100.0),   # A/B: ratio exactly 4 -> kept
        ("C", "J2", 101.0),   # A/C: 3.96 -> dropped
        ("D", "J1", 10.0),    # A/D: same jurisdiction -> dropped
        ("E", "J2", None),    # unpublished -> dropped
        ("F", "J2", 1.0),     # D/F: 10x -> kept
    ])
    edges = pd.DataFrame({
        "left_geoid": ["A", "B", "A", "A", "A", "D", "F"],
        "right_geoid": ["B", "A", "C", "D", "E", "F", "D"],   # B-A and F-D repeat A-B, D-F
    })
    out = LEDGER.ledger(bg, edges, {"J1": "Alpha city"}, totals())
    assert list(out.columns) == LEDGER.COLUMNS
    assert out[["block_group_a", "block_group_b", "ratio"]].values.tolist() == [
        ["D", "F", 10.0],
        ["A", "B", 4.0],
    ]
    ab = out.iloc[1]
    # The higher-index block group is side a.
    assert ab["overall_exposure_a"] == 400.0 and ab["overall_exposure_b"] == 100.0
    assert ab["jurisdiction_name_a"] == "Alpha city"
    assert ab["jurisdiction_name_b"] == "J2"          # no name on file: the id
    assert ab["level_repair_mode_b"] == "pooled_silent_unit"
    assert ab["total_latest_clean_year_b"] == 2019
    assert ab["total_clean_years_a"] == 8


def test_jurisdiction_totals_list_modes_once_and_take_the_weakest_offense():
    controls = pd.DataFrame({
        "jurisdiction_id": ["J1", "J1", "J1"],
        "offense": ["murder", "larceny", "robbery"],
        "level_repair_mode": ["none", "pooled_silent_unit", "none"],
    })
    smoothed = pd.DataFrame({
        "jurisdiction_id": ["J1", "J1"],
        "offense": ["murder", "larceny"],
        "latest_clean_year": [2025, 2021],
        "clean_year_count": [8, 2],
    })
    got = LEDGER.jurisdiction_totals(controls, smoothed)
    assert got.loc["J1"].tolist() == ["none;pooled_silent_unit", 2021, 2]


def test_statewide_remainder_gets_a_display_name():
    assert LEDGER.juris_display("17:state_nonmunicipal_remainder", {}) == "Unincorporated Illinois"
