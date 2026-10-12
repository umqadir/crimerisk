"""The exact-point QA reads the live feed surface and each group's own year window."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "build_city_feed_exact_point_qa.py"
_SPEC = importlib.util.spec_from_file_location("build_city_feed_exact_point_qa_under_test", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
qa = importlib.util.module_from_spec(_SPEC)
sys.modules["build_city_feed_exact_point_qa_under_test"] = qa
_SPEC.loader.exec_module(qa)


def _surface(rows):
    return pd.DataFrame(rows, columns=["city_name", "offense", "incident_count", "year"])


def test_live_feed_rows_replace_the_frozen_copy_and_carry_their_years(tmp_path):
    validation = tmp_path / "validation.parquet"
    feed = tmp_path / "feed.parquet"
    _surface(
        [
            ("New York", "murder", 10.0, 2025),  # stale copy of the feed
            ("Charlotte", "burglary", 30.0, 2024),
            ("Charlotte", "burglary", 40.0, 2025),
        ]
    ).to_parquet(validation)
    _surface(
        [
            ("New York", "murder", 300.0, 2020),
            ("New York", "murder", 400.0, 2025),
        ]
    ).to_parquet(feed)

    groups = qa._load_active_groups(validation, feed).set_index(["city_name", "offense"])

    assert groups.loc[("New York", "murder"), "surface_located_incident_count"] == 700.0
    assert groups.loc[("New York", "murder"), "surface_years"] == (2020, 2025)
    assert groups.loc[("Charlotte", "burglary"), "surface_located_incident_count"] == 70.0
    assert groups.loc[("Charlotte", "burglary"), "surface_years"] == (2024, 2025)


def test_without_a_feed_surface_the_validation_surface_stands(tmp_path):
    validation = tmp_path / "validation.parquet"
    _surface([("New York", "murder", 10.0, 2025)]).to_parquet(validation)
    groups = qa._load_active_groups(validation, tmp_path / "absent.parquet")
    assert groups["surface_located_incident_count"].tolist() == [10.0]
