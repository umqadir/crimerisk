"""An offense that stops arriving inside a twelve-month Return A year is a partial year.

Columbus GA (GA1060100) filed twelve months for 2025, but motor vehicle theft is zero from
June to December while every other offense keeps reporting (210 over January-May). The
Return A lane's nonzero-months field is the agency's Part I total, so only the per-offense
month profile from the master's monthly file shows the gap.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from crimerisk.observations import RETURN_A_OFFENSE_MONTHS_SOURCE, return_a_master_offense_months
from crimerisk.source_provenance import NIBRS_SOURCE
from crimerisk.trend_fills import build_masked_gap_flags

MVT_2025 = [31, 34, 53, 43, 49, 0, 0, 0, 0, 0, 0, 0]
LARCENY_2025 = [235, 242, 260, 297, 267, 277, 337, 88, 294, 266, 297, 292]


def _paths(tmp_path):
    data = tmp_path / "data" / "FBI-UCR-Return-A-Parsed-2025-20260921"
    data.mkdir(parents=True)
    rows = []
    for offense, months in (("motor_vehicle_theft", MVT_2025), ("larceny", LARCENY_2025)):
        for month, count in enumerate(months, start=1):
            rows.append({"ori7": "GA10601", "year": 2025, "month": month, "offense": offense,
                         "count": float(count), "reported": True, "population": 200622.0})
    pd.DataFrame(rows).to_parquet(data / "return_a_master_monthly_2025.parquet", index=False)
    reference = tmp_path / "state" / "reference"
    reference.mkdir(parents=True)
    pd.DataFrame({"ori9": ["GA1060100"], "ori7": ["GA10601"]}).to_parquet(reference / "agency_master.parquet", index=False)
    return SimpleNamespace(data_dir=tmp_path / "data", state_dir=tmp_path / "state")


def test_offense_month_profile_counts_nonzero_reported_months_per_offense(tmp_path):
    out = return_a_master_offense_months(_paths(tmp_path)).set_index("offense")
    assert set(out["ori9"]) == {"GA1060100"}
    assert out.loc["motor_vehicle_theft", "count"] == 210.0
    assert out.loc["motor_vehicle_theft", "nonzero_months"] == 5
    assert out.loc["motor_vehicle_theft", "months_reported"] == 12
    assert out.loc["larceny", "nonzero_months"] == 12
    assert set(out["source"]) == {RETURN_A_OFFENSE_MONTHS_SOURCE}


def test_masked_gap_reads_the_return_a_offense_profile_when_no_nibrs_row(tmp_path):
    paths = _paths(tmp_path)
    current = return_a_master_offense_months(paths)
    prior = pd.DataFrame({
        "ori9": "GA1060100", "year": 2024, "source": NIBRS_SOURCE, "offense": ["motor_vehicle_theft", "larceny"],
        "count": [495.0, 3289.0], "months_reported": 12.0, "months_missing": 0.0, "nonzero_months": 12.0,
        "population": 201061.0,
    })
    obs = pd.concat([prior, current], ignore_index=True)
    obs["year"] = obs["year"].astype("Int64")
    panel = pd.DataFrame({
        "ori9": "GA1060100", "offense": ["motor_vehicle_theft", "larceny"] * 2, "year": [2025, 2025, 2024, 2024],
        "preferred_count": [210.0, 3132.0, 495.0, 3289.0], "preferred_months_reported": 12.0, "usable_as_observed": True,
    })
    flags = build_masked_gap_flags(paths, target_year=2025, agency_panel=panel, obs=obs)
    assert flags["offense"].tolist() == ["motor_vehicle_theft"]
    row = flags.iloc[0]
    assert row["masked_gap_kind"] == "partial_months"
    assert row["masked_gap_effective_months"] == 5
