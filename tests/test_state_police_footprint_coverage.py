"""B2 (2025.2): a state-police footprint keeps only the share of the agency's total its residents cover."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk import allocation
from crimerisk.allocation import (
    COUNTY_NONMUNICIPAL_OVERLAP_TYPE,
    _apply_state_police_footprint_coverage,
    _components_with_primary_denominator,
    _state_police_footprint_units,
)
from crimerisk.paths import get_paths

SP = "42:police_service:PA002SP00"


def _covariates(bg_ids: list[str], population: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "bg_id": pd.array(bg_ids, dtype="string"),
            "population": population,
            "daytime_population_jobs_proxy": population,
            "exposure_proxy_2024": population,
            "landscan_day_pop": population,
            "landscan_day_lifted_person_exposure": False,
            "land_area_sq_mi": 1.0,
        }
    )
    for column in (
        "person_exposure_before_hq_jobs_cap",
        "person_exposure_hq_jobs_cap",
        "person_exposure_hq_jobs_cap_candidate",
        "person_exposure_hq_jobs_capped",
        "households_total",
        "commercial_premises_total",
        "destination_poi_total",
        "lodes_manufacturing_jobs",
        "lodes_wholesale_jobs",
        "lodes_retail_jobs",
        "lodes_transport_warehouse_jobs",
        "lodes_industrial_jobs",
        "burglary_premises_total",
        "burglary_commercial_exposure_weight",
        "burglary_destination_poi_exposure_weight",
        "burglary_retail_jobs_exposure_weight",
        "burglary_industrial_jobs_exposure_weight",
        "aggregate_vehicles_total",
        "county_auto_commute_vehicle_share",
        "mvt_commuter_vehicle_proxy",
        "vehicle_exposure_2024",
    ):
        frame[column] = 0.0
    return frame


def _components() -> pd.DataFrame:
    rows = [
        # The state-police unit: two footprint block groups.
        ("420035638003", SP, "municipal", 60.0),
        ("420035638004", SP, "municipal", 40.0),
        # The rest of the county's allocation.
        ("420035638004", "42:municipal:a", "municipal", 100.0),
        ("420030001001", "42:municipal:a", "municipal", 200.0),
        ("420030002001", "42:municipal:b", "municipal", 700.0),
        # Another county: untouched.
        ("420050001001", "42:municipal:c", "municipal", 500.0),
    ]
    frame = pd.DataFrame(rows, columns=["bg_id", "jurisdiction_id", "jurisdiction_type", "component_count"])
    frame["state_fips"] = "42"
    frame["source_state_fips"] = "42"
    frame["tract_id"] = frame["bg_id"].str.slice(0, 11)
    frame["offense"] = "larceny"
    frame["component_share"] = frame["component_count"] / frame.groupby("jurisdiction_id")[
        "component_count"
    ].transform("sum")
    return frame


def _units(coverage: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "jurisdiction_id": [SP],
            "jurisdiction_type": ["municipal"],
            "county_geoid": ["42003"],
            "oris": ["PA002SP00"],
            "served_population": [1000.0],
            "footprint_population": [1000.0 * coverage],
            "coverage": [coverage],
        }
    )


def test_uncovered_share_spreads_over_the_county_and_every_total_is_conserved():
    covariates = _covariates(
        ["420035638003", "420035638004", "420030001001", "420030002001", "420050001001"],
        [200.0, 100.0, 1000.0, 3000.0, 2000.0],
    )
    components = _components_with_primary_denominator(_components(), covariates)
    out, summary = _apply_state_police_footprint_coverage(components, covariates, units=_units(0.1))
    count = out["component_count"].astype(float)
    by_jurisdiction = count.groupby(out["jurisdiction_id"]).sum()
    before = components.groupby("jurisdiction_id")["component_count"].sum()
    pd.testing.assert_series_equal(by_jurisdiction.sort_index(), before.sort_index(), check_names=False)
    assert np.isclose(count.sum(), components["component_count"].sum(), rtol=0, atol=1e-9)

    unit = out[out["jurisdiction_id"].eq(SP)].set_index("bg_id")["component_count"]
    # 90 moves, in the county's other shape: 100 / 200 / 700 over the three block groups.
    assert np.isclose(unit["420035638003"], 6.0)
    assert np.isclose(unit["420035638004"], 4.0 + 90.0 * 0.1)
    assert np.isclose(unit["420030001001"], 90.0 * 0.2)
    assert np.isclose(unit["420030002001"], 90.0 * 0.7)
    assert "420050001001" not in unit.index
    assert np.isclose(summary["moved_count_by_offense"]["larceny"], 90.0)
    shares = out.loc[out["jurisdiction_id"].eq(SP), "component_share"].astype(float)
    assert np.isclose(shares.sum(), 1.0)
    # New rows carry the published denominator columns the zero-target step reads.
    assert out["primary_denominator_raw"].notna().all()


def test_full_coverage_leaves_the_components_unchanged():
    components = _components()
    covariates = _covariates(["420035638003"], [1.0])
    out, summary = _apply_state_police_footprint_coverage(components, covariates, units=_units(1.0))
    pd.testing.assert_frame_equal(out, components)
    assert summary["units_below_full_coverage"] == 0


def test_units_measure_footprint_residents_against_served_population(monkeypatch):
    master = pd.DataFrame(
        {
            "ori9": ["PA002SP00", "NJ013SP00", "NJ099PD00"],
            "state_fips": ["42", "34", "34"],
            "county_fips": ["003", "025", "025"],
            "agency_type_norm": ["state_law_enforcement", "state_law_enforcement", "local_police"],
            "population_latest_nibrs": [2605.0, 40000.0, 5000.0],
        }
    )
    monkeypatch.setattr(allocation, "_load_agency_master", lambda paths: master)
    crosswalk = pd.DataFrame(
        {
            "state_fips": ["42", "42", "34", "34"],
            "block_group_geoid": ["420035638003", "420035638004", "340250001001", "340250001001"],
            "jurisdiction_id": [SP, SP, "34:state_nonmunicipal_remainder", "34:municipal:x"],
            "jurisdiction_type": ["municipal", "municipal", "state_nonmunicipal_remainder", "municipal"],
            "allocation_share": [1.0, 1.0, 0.25, 0.75],
        }
    )
    covariates = pd.DataFrame(
        {
            "bg_id": pd.array(["420035638003", "420035638004", "340250001001"], dtype="string"),
            "population": [197.0, 98.0, 80000.0],
        }
    )
    components = pd.DataFrame({"jurisdiction_id": [SP, "34025", "42:municipal:a"]})
    groups = pd.DataFrame({"state_fips": ["34"], "group_id": ["34025"], "ori9": ["NJ013SP00"]})
    units = _state_police_footprint_units(
        paths=get_paths(),
        components=components,
        bg_covariates=covariates,
        bg_crosswalk=crosswalk,
        county_group_oris=groups,
    ).set_index("jurisdiction_id")
    assert np.isclose(units.loc[SP, "footprint_population"], 295.0)
    assert np.isclose(units.loc[SP, "coverage"], 295.0 / 2605.0)
    assert units.loc[SP, "county_geoid"] == "42003"
    assert units.loc["34025", "jurisdiction_type"] == COUNTY_NONMUNICIPAL_OVERLAP_TYPE
    assert np.isclose(units.loc["34025", "footprint_population"], 20000.0)
    assert np.isclose(units.loc["34025", "coverage"], 0.5)
