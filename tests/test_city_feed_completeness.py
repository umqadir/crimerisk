"""City feed completeness: a truncated pull fails the build instead of shipping as a quiet year."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crimerisk.city_incidents import (
    _check_nyc_report_years,
    _raise_on_socrata_error_trailer,
)
from crimerisk.city_shares import (
    CITY_YEAR_COLLAPSE_FRACTION,
    city_year_collapses,
    load_city_feed_year_exclusions,
)
from crimerisk.paths import RepoPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


def _surface(counts: dict[tuple[str, str, int], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city_key": city, "offense": offense, "year": year, "incident_count": value}
            for (city, offense, year), value in counts.items()
        ]
    )


def test_a_steady_series_passes():
    surface = _surface({("austin", "larceny", year): 1_000.0 + 50.0 * year % 7 for year in range(2018, 2026)})
    assert city_year_collapses(surface).empty


def test_a_collapsed_year_fails_and_is_named():
    counts = {("new_york", "larceny", year): 60_000.0 for year in range(2018, 2026)}
    counts[("new_york", "larceny", 2020)] = 72.0
    collapses = city_year_collapses(_surface(counts))
    assert collapses[["city_key", "offense", "year"]].to_dict("records") == [
        {"city_key": "new_york", "offense": "larceny", "year": 2020}
    ]


def test_a_missing_year_inside_the_span_counts_as_zero():
    counts = {("chicago", "robbery", year): 8_000.0 for year in (2018, 2019, 2021, 2022)}
    collapses = city_year_collapses(_surface(counts))
    assert collapses["year"].tolist() == [2020]
    assert collapses["incident_count"].tolist() == [0.0]


def test_the_threshold_is_a_fraction_of_the_series_median():
    counts = {("seattle", "murder", year): 60.0 for year in range(2018, 2024)}
    counts[("seattle", "murder", 2024)] = 60.0 * CITY_YEAR_COLLAPSE_FRACTION
    assert city_year_collapses(_surface(counts)).empty
    counts[("seattle", "murder", 2024)] = 60.0 * CITY_YEAR_COLLAPSE_FRACTION - 1.0
    assert city_year_collapses(_surface(counts))["year"].tolist() == [2024]


def test_a_single_year_series_is_never_a_collapse():
    assert city_year_collapses(_surface({("st_louis_mo", "burglary", 2024): 10.0})).empty


def _nyc(rows_by_year: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rpt_dt": [
                f"{year}-06-01T00:00:00.000" for year, n in rows_by_year.items() for _ in range(n)
            ]
        }
    )


def test_the_nyc_snapshot_must_cover_every_report_year_in_its_window():
    _check_nyc_report_years(_nyc({2018: 10, 2019: 9, 2020: 8}), start_year=2018, end_year=2020)
    with pytest.raises(RuntimeError, match="does not cover"):
        _check_nyc_report_years(_nyc({2018: 10, 2020: 8}), start_year=2018, end_year=2020)
    with pytest.raises(RuntimeError, match="does not cover"):
        _check_nyc_report_years(_nyc({2018: 10, 2019: 1, 2020: 8}), start_year=2018, end_year=2020)


def test_the_2026_03_22_snapshot_shape_is_refused():
    """2.0M rows of 2008-2012 plus one year of 2025, the cached file behind the 2025.1 NY feed."""
    with pytest.raises(RuntimeError, match="outside the window"):
        _check_nyc_report_years(
            _nyc({2009: 20, 2010: 20, 2021: 1, 2025: 20}), start_year=2018, end_year=2025
        )


def test_a_socrata_body_that_ends_in_an_error_object_is_rejected(tmp_path: Path):
    body = tmp_path / "rows.csv.tmp"
    body.write_text('CMPLNT_NUM,RPT_DT\n1,07/26/2012\n85943699,07/25/2012,17:00{\n  "error" : true,\n  "message" : "Internal error",\n  "status" : 500\n}\n')
    with pytest.raises(RuntimeError, match="server error body"):
        _raise_on_socrata_error_trailer(body, url="https://example.invalid/rows.csv")
    assert not body.exists()
    clean = tmp_path / "clean.csv.tmp"
    clean.write_text("cmplnt_num,rpt_dt\n1,2018-01-01T00:00:00.000\n")
    _raise_on_socrata_error_trailer(clean, url="https://example.invalid/rows.csv")
    assert clean.exists()


def test_every_year_exclusion_names_a_feed_city_offense_year_and_a_reason():
    table = load_city_feed_year_exclusions(RepoPaths.from_repo_root(REPO_ROOT))
    assert not table.duplicated(["city_key", "offense", "year"]).any()
    assert table["reason"].str.len().gt(20).all()
    assert set(table["offense"]) <= {
        "murder", "rape", "robbery", "aggravated_assault", "burglary", "larceny", "motor_vehicle_theft"
    }


def test_a_withheld_year_inside_the_span_is_not_a_collapse():
    counts = {("philadelphia", "robbery", year): 5_000.0 for year in (2018, 2020, 2021, 2022, 2025)}
    excluded = pd.DataFrame({"city_key": ["philadelphia"] * 3, "offense": ["robbery"] * 3, "year": [2019, 2023, 2024]})
    assert city_year_collapses(_surface(counts), excluded).empty
    assert city_year_collapses(_surface(counts))["year"].tolist() == [2019, 2023, 2024]
