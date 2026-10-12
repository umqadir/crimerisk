from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk import allocation
from crimerisk.cli import build_parser


def _frame() -> pd.DataFrame:
    # Two tracts, two block groups each; the model prior favours tract A.
    return pd.DataFrame(
        {
            "jurisdiction_id": ["j"] * 4,
            "state_fips": ["06"] * 4,
            "offense": ["murder"] * 4,
            "tract_id": ["06000000001", "06000000001", "06000000002", "06000000002"],
            "bg_id": ["060000000011", "060000000012", "060000000021", "060000000022"],
            "city_posterior_model_prior_share": [0.45, 0.45, 0.05, 0.05],
        }
    )


def _year_counts(rows: list[tuple[str, int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city_name": "City",
                "jurisdiction_id": "j",
                "state_fips": "06",
                "offense": offense,
                "block_group_geoid": bg,
                "year": year,
                "incident_count": count,
            }
            for offense, year, bg, count in rows
        ]
    )


def test_violent_prior_is_the_city_tract_share_and_strength_is_learned() -> None:
    rows = []
    for year in (2019, 2020, 2021):
        rows += [
            ("robbery", year, "060000000021", 30.0),
            ("aggravated_assault", year, "060000000022", 60.0),
            ("robbery", year, "060000000011", 10.0),
            ("murder", year, "060000000021", 2.0),
        ]
    learned = allocation._murder_violent_tract_prior(_frame(), year_counts=_year_counts(rows))
    assert learned["tract_prior_share"].round(6).tolist() == [0.1, 0.1, 0.9, 0.9]
    assert learned["rolling_origin_origins"].eq(2).all()
    assert learned["prior_incidents"].notna().all()


def test_no_violent_feed_keeps_the_b0_prior() -> None:
    rows = [("murder", 2020, "060000000021", 2.0), ("murder", 2021, "060000000021", 1.0)]
    learned = allocation._murder_violent_tract_prior(_frame(), year_counts=_year_counts(rows))
    assert learned["prior_incidents"].isna().all()
    assert learned["tract_prior_share"].isna().all()


def test_tract_posterior_with_violent_prior_conserves_and_spreads_by_model_prior() -> None:
    frame = _frame()
    direct = pd.Series([1.0, 0.0, 3.0, 0.0], index=frame.index)
    result = allocation._tract_incident_posterior(
        frame,
        direct_count=direct,
        prior_share=frame["city_posterior_model_prior_share"],
        prior_incidents=pd.Series(4.0, index=frame.index),
        tract_prior_share=pd.Series([0.1, 0.1, 0.9, 0.9], index=frame.index),
    )
    assert np.isclose(result["posterior_share"].sum(), 1.0)
    # Tract B: (3 + 4 * 0.9) / (4 + 4) = 0.825, split evenly by the model prior.
    assert np.allclose(result["posterior_share"].iloc[2:], [0.4125, 0.4125])


def test_cli_flag_defaults_on_with_b0_off_switch() -> None:
    assert build_parser().parse_args(["build-outputs"]).murder_violent_tract_prior is True
    assert (
        build_parser()
        .parse_args(["build-outputs", "--no-enable-murder-violent-tract-prior"])
        .murder_violent_tract_prior
        is False
    )
