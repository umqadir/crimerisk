from __future__ import annotations

from pathlib import Path

import pandas as pd

from crimerisk import allocation
from crimerisk.cli import build_parser
from crimerisk.research.m4_rolling_origin_strength import rolling_origin_posterior_strength
from crimerisk.paths import RepoPaths


def _merged(prior: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "jurisdiction_id": ["j"] * len(prior),
            "state_fips": ["06"] * len(prior),
            "offense": ["larceny"] * len(prior),
            "bg_id": [f"06000000000{i}" for i in range(len(prior))],
            "city_posterior_model_prior_share": prior,
            "city_incident_posterior_active": [True] * len(prior),
        }
    )


def _year_counts(by_year: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for year, counts in by_year.items():
        for i, count in enumerate(counts):
            rows.append(
                {
                    "city_name": "City",
                    "jurisdiction_id": "j",
                    "state_fips": "06",
                    "offense": "larceny",
                    "block_group_geoid": f"06000000000{i}",
                    "year": year,
                    "incident_count": count,
                }
            )
    return pd.DataFrame(rows)


def test_stable_history_learns_no_prior_weight() -> None:
    history = [60.0, 30.0, 10.0]
    learned = rolling_origin_posterior_strength(
        _merged([0.1, 0.1, 0.8]),
        year_counts=_year_counts({2019: history, 2020: history, 2021: history}),
        offenses=("larceny",),
    )
    assert len(learned) == 1
    assert learned["rolling_origin_alpha"].iloc[0] == 0.0
    assert learned["rolling_origin_origins"].iloc[0] == 2


def test_prior_that_predicts_the_next_year_gets_weight() -> None:
    # Each year's pattern flips, so the pooled history is a poor forecast and the prior,
    # which matches every target year, should dominate.
    learned = rolling_origin_posterior_strength(
        _merged([0.5, 0.5, 0.0]),
        year_counts=_year_counts({2019: [0.0, 0.0, 10.0], 2020: [5.0, 5.0, 0.0], 2021: [5.0, 5.0, 0.0]}),
        offenses=("larceny",),
    )
    assert learned["rolling_origin_alpha"].iloc[0] >= 1.0e3


def test_single_feed_year_and_other_offenses_get_no_row() -> None:
    merged = _merged([0.5, 0.5, 0.0])
    one_year = rolling_origin_posterior_strength(
        merged, year_counts=_year_counts({2021: [1.0, 2.0, 3.0]}), offenses=("larceny",)
    )
    assert one_year.empty
    other = rolling_origin_posterior_strength(
        merged,
        year_counts=_year_counts({2020: [1.0, 2.0, 3.0], 2021: [1.0, 2.0, 3.0]}),
        offenses=("robbery",),
    )
    assert other.empty


def test_year_counts_stop_at_the_feed_truncation(tmp_path: Path) -> None:
    surface = tmp_path / "city.parquet"
    frame = _year_counts({2020: [1.0, 2.0, 3.0], 2021: [1.0, 2.0, 3.0], 2022: [9.0, 9.0, 9.0]})
    frame["share_within_city"] = 0.0
    frame["geocode_quality_tier"] = "exact"
    frame.to_parquet(surface, index=False)
    counts = allocation._city_incident_year_counts(
        RepoPaths.from_repo_root(tmp_path), year=2021, path=surface
    )
    assert sorted(counts["year"].unique()) == [2020, 2021]
    pooled = allocation._load_city_incident_share_surface(
        RepoPaths.from_repo_root(tmp_path), year=2021, path=surface
    )
    assert pooled["incident_count"].sum() == counts["incident_count"].sum()


def test_the_release_build_has_no_m4_switch() -> None:
    args = build_parser().parse_args(["build-outputs"])
    assert not hasattr(args, "rolling_origin_posterior_offenses")
    assert "rolling_origin_posterior_offenses" not in allocation.AllocationBuildConfig.__dataclass_fields__
