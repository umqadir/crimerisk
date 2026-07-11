#!/usr/bin/env python3
"""
Pull ACS 5-year data at tract level for all US states.

Usage:
    python scripts/pull/pull_acs_tracts.py [--acs-year YEAR] [--api-key KEY] [--output-dir DIR]

Output:
    data/ACS-5yr-<start>-<end>/acs_tracts_full.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56", "72",
]


TRACT_VARIABLES = {
    "B01003_001E": "total_population",
    "B19301_001E": "per_capita_income",
    "B17001_002E": "poverty_count",
    "B15003_001E": "edu_pop_25plus",
    "B15003_017E": "edu_hs_diploma",
    "B15003_022E": "edu_bachelors",
    "B15003_023E": "edu_masters",
    "B15003_024E": "edu_professional",
    "B15003_025E": "edu_doctorate",
    "B23025_002E": "labor_force",
    "B23025_005E": "unemployed",
    "B25070_001E": "rent_burden_total",
    "B25070_007E": "rent_30_35pct",
    "B25070_008E": "rent_35_40pct",
    "B25070_009E": "rent_40_50pct",
    "B25070_010E": "rent_50plus_pct",
    "B08303_001E": "commute_total",
    "B08303_012E": "commute_45_59",
    "B08303_013E": "commute_60plus",
    "B08301_001E": "commute_mode_total",
    "B08301_003E": "commute_drive_alone",
    "B08301_004E": "commute_carpool",
    "B08301_010E": "commute_transit",
    "B08301_016E": "commute_taxi_ridehail",
    "B08301_018E": "commute_bicycle",
    "B08301_019E": "commute_walk",
    "B08301_021E": "commute_work_from_home",
    "B01001_001E": "pop_total",
    "B01001_003E": "male_under5",
    "B01001_004E": "male_5to9",
    "B01001_005E": "male_10to14",
    "B01001_006E": "male_15to17",
    "B01001_007E": "male_18to19",
    "B01001_008E": "male_20",
    "B01001_009E": "male_21",
    "B01001_010E": "male_22to24",
    "B01001_011E": "male_25to29",
    "B01001_012E": "male_30to34",
    "B01001_013E": "male_35to39",
    "B01001_014E": "male_40to44",
    "B01001_015E": "male_45to49",
    "B01001_016E": "male_50to54",
    "B01001_017E": "male_55to59",
    "B01001_018E": "male_60to61",
    "B01001_019E": "male_62to64",
    "B01001_020E": "male_65to66",
    "B01001_021E": "male_67to69",
    "B01001_022E": "male_70to74",
    "B01001_023E": "male_75to79",
    "B01001_024E": "male_80to84",
    "B01001_025E": "male_85plus",
    "B01002_001E": "median_age",
    "B01002_002E": "median_age_male",
    "B01002_003E": "median_age_female",
    "B11001_001E": "households_total",
    "B11001_002E": "hh_family",
    "B11001_007E": "hh_nonfamily",
    "B25002_001E": "housing_units_total",
    "B25002_002E": "housing_occupied",
    "B25002_003E": "housing_vacant",
    "B25003_001E": "tenure_total",
    "B25003_002E": "tenure_owner",
    "B25003_003E": "tenure_renter",
    "B25044_001E": "vehicle_total",
    "B25044_002E": "vehicle_owner_total",
    "B25044_003E": "owner_no_vehicle",
    "B25044_004E": "owner_1_vehicle",
    "B25044_005E": "owner_2_vehicle",
    "B25044_006E": "owner_3_vehicle",
    "B25044_007E": "owner_4_vehicle",
    "B25044_008E": "owner_5plus_vehicle",
    "B25044_009E": "vehicle_renter_total",
    "B25044_010E": "renter_no_vehicle",
    "B25044_011E": "renter_1_vehicle",
    "B25044_012E": "renter_2_vehicle",
    "B25044_013E": "renter_3_vehicle",
    "B25044_014E": "renter_4_vehicle",
    "B25044_015E": "renter_5plus_vehicle",
    "B25046_001E": "aggregate_vehicles_total",
    "B25046_002E": "aggregate_vehicles_owner",
    "B25046_003E": "aggregate_vehicles_renter",
    "B28002_001E": "internet_total",
    "B28002_002E": "internet_subscription_any",
    "B28002_003E": "internet_dialup_only",
    "B28002_004E": "internet_broadband_any",
    "B28002_005E": "internet_cellular_any",
    "B28002_006E": "internet_cellular_only",
    "B28002_007E": "internet_cable_fiber_dsl_any",
    "B28002_008E": "internet_cable_fiber_dsl_only",
    "B28002_009E": "internet_satellite_any",
    "B28002_010E": "internet_satellite_only",
    "B28002_011E": "internet_other_service_only",
    "B28002_012E": "internet_no_subscription",
    "B28002_013E": "internet_no_access",
    "B28003_001E": "computer_total",
    "B28003_002E": "computer_has_any",
    "B28003_004E": "computer_broadband",
    "B28003_005E": "computer_no_internet",
    "B28003_006E": "computer_no_device",
    "B07003_001E": "mobility_total",
    "B07003_004E": "mobility_same_house",
    "B07003_007E": "mobility_diff_house_same_county",
    "B07003_010E": "mobility_diff_county_same_state",
    "B07003_013E": "mobility_diff_state",
    "B07003_016E": "mobility_abroad",
}

MAYBE_TRACT_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B25064_001E": "median_rent",
    "B25077_001E": "median_home_value",
}

FAMILY_HH_VARIABLES = {
    "B11003_001E": "hh_family_total_b11003",
    "B11003_002E": "hh_family_married",
    "B11003_003E": "hh_family_married_children",
    "B11003_009E": "hh_family_male_no_spouse",
    "B11003_010E": "hh_family_male_no_spouse_children",
    "B11003_015E": "hh_family_female_no_spouse",
    "B11003_016E": "hh_family_female_no_spouse_children",
}


def fetch_tract_data(
    state_fips: str,
    variables: dict[str, str],
    acs_year: int,
    api_key: str | None = None,
) -> pd.DataFrame | None:
    acs_base_url = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    var_codes = list(variables.keys())
    chunk_size = 48
    all_data = None

    for i in range(0, len(var_codes), chunk_size):
        chunk_vars = var_codes[i:i + chunk_size]
        var_str = ",".join(chunk_vars)
        url = f"{acs_base_url}?get=NAME,{var_str}&for=tract:*&in=state:{state_fips}&in=county:*"
        if api_key:
            url += f"&key={api_key}"
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"  Error fetching state {state_fips} chunk {i}: {e}")
            return None
        if not data or len(data) < 2:
            print(f"  No data returned for state {state_fips}")
            return None
        headers = data[0]
        rows = data[1:]
        chunk_df = pd.DataFrame(rows, columns=headers)
        if all_data is None:
            all_data = chunk_df
        else:
            geo_cols = ["NAME", "state", "county", "tract"]
            all_data = all_data.merge(chunk_df, on=geo_cols, how="outer")
        time.sleep(0.1)

    return all_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull ACS tract data")
    parser.add_argument("--acs-year", type=int, default=2024)
    parser.add_argument("--api-key")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--states", nargs="*")
    args = parser.parse_args()

    acs_start_year = args.acs_year - 4
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/ACS-5yr-{acs_start_year}-{args.acs_year}")
    output_dir.mkdir(parents=True, exist_ok=True)
    states = args.states if args.states else STATE_FIPS
    variables = {**TRACT_VARIABLES, **FAMILY_HH_VARIABLES, **MAYBE_TRACT_VARIABLES}

    print(f"Pulling ACS {acs_start_year}-{args.acs_year} tract data...")
    print(f"Pulling {len(variables)} variables for {len(states)} states...")

    all_state_dfs: list[pd.DataFrame] = []
    failed_states: list[str] = []
    for i, state_fips in enumerate(states):
        print(f"[{i+1}/{len(states)}] Fetching state {state_fips}...", end=" ", flush=True)
        df = fetch_tract_data(state_fips, variables, args.acs_year, args.api_key)
        if df is None:
            print("FAILED")
            failed_states.append(state_fips)
            continue
        print(f"got {len(df)} tracts")
        all_state_dfs.append(df)

    if not all_state_dfs:
        print("No data fetched.")
        return 1

    combined = pd.concat(all_state_dfs, ignore_index=True)
    combined = combined.rename(columns=variables)
    combined["tract_id"] = (
        combined["state"].astype(str).str.zfill(2)
        + combined["county"].astype(str).str.zfill(3)
        + combined["tract"].astype(str).str.zfill(6)
    )
    combined["state_fips"] = combined["state"].astype(str).str.zfill(2)
    combined["county_fips"] = combined["county"].astype(str).str.zfill(3)

    value_cols = [c for c in combined.columns if c not in {"NAME", "state", "county", "tract", "tract_id", "state_fips", "county_fips"}]
    for col in value_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    output_path = output_dir / "acs_tracts_full.parquet"
    combined.to_parquet(output_path, index=False)
    print(f"\nSaved {len(combined)} tracts to {output_path}")
    print(f"States succeeded: {len(states) - len(failed_states)}")
    if failed_states:
        print(f"Failed states: {failed_states}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
