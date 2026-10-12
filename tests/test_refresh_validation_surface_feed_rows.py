"""The validation surface takes its feed-city rows from the live feed surface and nothing else."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build" / "refresh_validation_surface_feed_rows.py"
_SPEC = importlib.util.spec_from_file_location("refresh_validation_surface_under_test", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
refresh_module = importlib.util.module_from_spec(_SPEC)
sys.modules["refresh_validation_surface_under_test"] = refresh_module
_SPEC.loader.exec_module(refresh_module)

FEED_COLUMNS = [
    "city_name", "jurisdiction_id", "state_fips", "year", "offense",
    "block_group_geoid", "incident_count", "share_within_city", "geocode_quality_tier",
]
META = ["validation_case_type", "validation_source_name", "validation_source_url", "validation_case_notes"]


def _row(city, jid, count, case="promoted_city_control"):
    return [city, jid, "36", 2024, "robbery", "360610001001", count, 1.0, "reported_latlon",
            case, "existing_city_incident_share_surface" if case == "promoted_city_control" else "portal",
            "", "note"]


def test_feed_rows_replaced_validation_rows_carried(monkeypatch):
    monkeypatch.setattr(refresh_module, "load_texture_policy", lambda: pd.DataFrame())
    surface = pd.DataFrame(
        [
            _row("New York", "nyc", 3.0),
            _row("Charlotte", "clt", 7.0, case="municipal_validation_case"),
            _row("St. Louis", "stl", 5.0, case="municipal_validation_case"),
        ],
        columns=FEED_COLUMNS + META,
    )
    feed = pd.DataFrame(
        [r[:9] for r in (_row("New York", "nyc", 300.0), _row("St. Louis", "stl", 9.0))],
        columns=FEED_COLUMNS,
    )
    feed["city_key"] = ["new_york", "st_louis"]

    out = refresh_module.refresh(surface, feed).set_index("city_name")

    assert list(out.reset_index().columns) == list(surface.columns)
    assert out.loc["New York", "incident_count"] == 300.0
    assert out.loc["New York", "validation_case_type"] == "promoted_city_control"
    assert out.loc["Charlotte", "incident_count"] == 7.0
    # A city with its own validation feed keeps its validation rows.
    assert out.loc["St. Louis", "incident_count"] == 5.0
    assert out.loc["St. Louis", "validation_case_type"] == "municipal_validation_case"
