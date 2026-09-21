from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "next_phase_measurement.py"
SPEC = importlib.util.spec_from_file_location("next_phase_measurement_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_bg_population_frame = MODULE._bg_population_frame
_build_output_count_prior = MODULE._build_output_count_prior


def test_bg_population_frame_uses_requested_surface_and_year(tmp_path):
    surface = tmp_path / "candidate.parquet"
    pd.DataFrame(
        {
            "block_group_geoid": ["010010201001"],
            "state_fips": ["01"],
            "tract_id": ["01001020100"],
            "population_2024": [111.0],
            "population_2025": [222.0],
        }
    ).to_parquet(surface, index=False)

    actual = _bg_population_frame(surface, year=2025)

    assert actual.loc[0, "bg_id"] == "010010201001"
    assert actual.loc[0, "county_fips"] == "01001"
    assert actual.loc[0, "population"] == 222.0


def test_output_count_prior_reads_expected_counts_not_removed_legacy_counts():
    output = pd.DataFrame(
        {
            "bg_id": ["010010201001"],
            "tract_id": ["01001020100"],
            "state_fips": ["01"],
            "expected_count_murder": [0.25],
            "expected_count_robbery": [4.5],
        }
    )

    actual = _build_output_count_prior(output, offenses=["murder", "robbery", "burglary"])

    assert actual[["offense", "bg_weight"]].to_dict("records") == [
        {"offense": "murder", "bg_weight": 0.25},
        {"offense": "robbery", "bg_weight": 4.5},
    ]
