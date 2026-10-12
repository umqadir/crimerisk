from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crimerisk import allocation
from crimerisk.denominators import calibrate_burglary_commercial_weight
from crimerisk.paths import RepoPaths


BLOCK_GROUPS_PER_CITY = 150
# Burglaries per destination POI, relative to one per household, in each synthetic city.
CITY_K = {"Chicago": 10.0, "New York": 40.0}
CITY_STATE = {"Chicago": "17", "New York": "36"}


def _surface_and_covariates(tmp_path):
    rng = np.random.default_rng(20260925)
    rows = []
    covariates = []
    for city_index, (city, k) in enumerate(CITY_K.items()):
        for bg_index in range(BLOCK_GROUPS_PER_CITY):
            geoid = f"{CITY_STATE[city]}{city_index:03d}{bg_index:07d}"
            households = float(rng.integers(50, 500))
            poi = float(rng.integers(0, 20))
            covariates.append(
                {
                    "bg_id": geoid,
                    "households_total": households,
                    "destination_poi_total": poi,
                    "lodes_retail_jobs": 0.0,
                    "lodes_industrial_jobs": 0.0,
                    "census2020_gq_total": 0.0,
                }
            )
            burglaries = households + k * poi
            # 2020 carries the held-out-year tail of New York: twice the POI rate.
            for year, scale in ((2019, 1.0), (2020, 1.0 if city == "Chicago" else 2.0)):
                rows.append(
                    {
                        "city_name": city,
                        "jurisdiction_id": f"{CITY_STATE[city]}:municipal:place:{city_index}",
                        "state_fips": CITY_STATE[city],
                        "year": year,
                        "offense": "burglary",
                        "block_group_geoid": geoid,
                        "incident_count": households + scale * k * poi if year == 2020 else burglaries,
                    }
                )
    surface_path = (
        tmp_path / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2025.parquet"
    )
    surface_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(surface_path, index=False)
    return RepoPaths.from_repo_root(tmp_path), pd.DataFrame(covariates)


def test_held_out_city_is_not_in_the_premises_fit(tmp_path):
    paths, covariates = _surface_and_covariates(tmp_path)

    pooled = calibrate_burglary_commercial_weight(paths=paths, bg_covariates=covariates, year=2025)
    held_out = calibrate_burglary_commercial_weight(
        paths=paths, bg_covariates=covariates, year=2025, exclude_city_keys=("new_york",)
    )

    assert pooled["calibration_city_count"] == 2
    assert pooled["k_destination_poi"] > CITY_K["Chicago"] + 1.0
    assert held_out["calibration_city_count"] == 1
    assert held_out["calibration_city_names"] == ["Chicago"]
    assert held_out["calibration_excluded_city_keys"] == ["new_york"]
    assert held_out["calibration_holdout_rows_dropped"] == 2 * BLOCK_GROUPS_PER_CITY
    assert held_out["k_destination_poi"] == pytest.approx(CITY_K["Chicago"], rel=1e-6)


def test_held_out_years_are_not_in_the_premises_fit(tmp_path):
    paths, covariates = _surface_and_covariates(tmp_path)

    truncated = calibrate_burglary_commercial_weight(
        paths=paths,
        bg_covariates=covariates,
        year=2025,
        exclude_city_keys=("chicago",),
        year_end=2019,
    )
    full_years = calibrate_burglary_commercial_weight(
        paths=paths, bg_covariates=covariates, year=2025, exclude_city_keys=("chicago",)
    )

    assert truncated["calibration_year_max"] == 2019
    assert truncated["k_destination_poi"] == pytest.approx(CITY_K["New York"], rel=1e-6)
    assert full_years["k_destination_poi"] == pytest.approx(1.5 * CITY_K["New York"], rel=1e-6)


def test_stage_12_covariates_pass_the_fold_holdout_to_the_calibration(tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    class Stop(Exception):
        pass

    def fake_add_offense_denominators(bg, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(allocation, "build_bg_feature_frame", lambda **_: pd.DataFrame())
    monkeypatch.setattr(allocation, "add_offense_denominators", fake_add_offense_denominators)
    with pytest.raises(Stop):
        allocation._load_bg_covariates(
            RepoPaths.from_repo_root(tmp_path),
            year=2025,
            exclude_feed_city_keys=("seattle",),
            feed_year_end=2021,
        )

    assert seen["burglary_calibration_exclude_city_keys"] == ("seattle",)
    assert seen["burglary_calibration_year_end"] == 2021
