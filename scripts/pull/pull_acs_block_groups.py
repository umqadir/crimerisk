#!/usr/bin/env python3
"""
Pull ACS 5-year data at block group level for all US states.

This script fetches the same variables used in the tract-level covariate build,
but at block group resolution to support finer-grained crime risk modeling.

Usage:
    python scripts/pull/pull_acs_block_groups.py [--acs-year YEAR] [--api-key KEY] [--output-dir DIR]

Output:
    data/ACS-5yr-<start>-<end>/acs_block_groups.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# State FIPS codes (50 states + DC + PR)
STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56", "72",  # Include Puerto Rico
]

# Variables available at block group level
# NOTE: Some median variables are NOT available at BG level - we'll handle those separately
BG_VARIABLES = {
    # Population (B01003)
    "B01003_001E": "total_population",

    # Per capita income (B19301) - available at BG
    "B19301_001E": "per_capita_income",

    # Poverty (B17001)
    "B17001_002E": "poverty_count",

    # Education - B15003 (Educational Attainment 25+)
    "B15003_001E": "edu_pop_25plus",
    "B15003_017E": "edu_hs_diploma",
    "B15003_022E": "edu_bachelors",
    "B15003_023E": "edu_masters",
    "B15003_024E": "edu_professional",
    "B15003_025E": "edu_doctorate",

    # Employment (B23025)
    "B23025_002E": "labor_force",
    "B23025_005E": "unemployed",

    # Rent burden (B25070)
    "B25070_001E": "rent_burden_total",
    "B25070_007E": "rent_30_35pct",
    "B25070_008E": "rent_35_40pct",
    "B25070_009E": "rent_40_50pct",
    "B25070_010E": "rent_50plus_pct",

    # Commute time (B08303)
    "B08303_001E": "commute_total",
    "B08303_012E": "commute_45_59",
    "B08303_013E": "commute_60plus",

    # Commute mode (B08301)
    "B08301_001E": "commute_mode_total",
    "B08301_003E": "commute_drive_alone",
    "B08301_004E": "commute_carpool",
    "B08301_010E": "commute_transit",
    "B08301_016E": "commute_taxi_ridehail",
    "B08301_018E": "commute_bicycle",
    "B08301_019E": "commute_walk",
    "B08301_021E": "commute_work_from_home",

    # Commute mode (B08301)
    "B08301_001E": "commute_mode_total",
    "B08301_003E": "commute_drive_alone",
    "B08301_004E": "commute_carpool",
    "B08301_010E": "commute_transit",
    "B08301_016E": "commute_taxi_ridehail",
    "B08301_018E": "commute_bicycle",
    "B08301_019E": "commute_walk",
    "B08301_021E": "commute_work_from_home",

    # Age/Sex (B01001)
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

    # Median age (B01002) - check if available at BG
    "B01002_001E": "median_age",
    "B01002_002E": "median_age_male",
    "B01002_003E": "median_age_female",

    # Households (B11001)
    "B11001_001E": "households_total",
    "B11001_002E": "hh_family",
    "B11001_007E": "hh_nonfamily",

    # Housing units (B25002)
    "B25002_001E": "housing_units_total",
    "B25002_002E": "housing_occupied",
    "B25002_003E": "housing_vacant",

    # Tenure (B25003)
    "B25003_001E": "tenure_total",
    "B25003_002E": "tenure_owner",
    "B25003_003E": "tenure_renter",

    # Income distribution (B19001)
    "B19001_001E": "hh_income_total",
    "B19001_002E": "hh_income_lt10k",
    "B19001_003E": "hh_income_10_15k",
    "B19001_004E": "hh_income_15_20k",
    "B19001_005E": "hh_income_20_25k",
    "B19001_006E": "hh_income_25_30k",
    "B19001_007E": "hh_income_30_35k",
    "B19001_008E": "hh_income_35_40k",
    "B19001_009E": "hh_income_40_45k",
    "B19001_010E": "hh_income_45_50k",
    "B19001_011E": "hh_income_50_60k",
    "B19001_012E": "hh_income_60_75k",
    "B19001_013E": "hh_income_75_100k",
    "B19001_014E": "hh_income_100_125k",
    "B19001_015E": "hh_income_125_150k",
    "B19001_016E": "hh_income_150_200k",
    "B19001_017E": "hh_income_200k_plus",

    # Households with children (B11005)
    "B11005_001E": "hh_children_total_b11005",
    "B11005_002E": "hh_under18",
    "B11005_008E": "hh_under18_nonfamily",

    # SNAP / Food Stamps (B22010)
    "B22010_001E": "snap_total",
    "B22010_002E": "snap_received",

    # Vehicle availability by tenure (B25044) + aggregate vehicles (B25046)
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

    # Internet access and subscriptions (B28002)
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

    # Computer access (B28003)
    "B28003_001E": "computer_total",
    "B28003_002E": "computer_has_any",
    "B28003_004E": "computer_broadband",
    "B28003_005E": "computer_no_internet",
    "B28003_006E": "computer_no_device",

    # Units in structure (B25024)
    "B25024_001E": "structure_total",
    "B25024_002E": "structure_single_detached",
    "B25024_003E": "structure_single_attached",
    "B25024_004E": "structure_2_units",
    "B25024_005E": "structure_3_4_units",
    "B25024_006E": "structure_5_9_units",
    "B25024_007E": "structure_10_19_units",
    "B25024_008E": "structure_20_49_units",
    "B25024_009E": "structure_50plus_units",
    "B25024_010E": "structure_mobile_home",
    "B25024_011E": "structure_other",

    # Vacancy type (B25004)
    "B25004_001E": "vacancy_total_b25004",
    "B25004_002E": "vacancy_for_rent",
    "B25004_003E": "vacancy_rented_not_occupied",
    "B25004_004E": "vacancy_for_sale",
    "B25004_005E": "vacancy_sold_not_occupied",
    "B25004_006E": "vacancy_seasonal",
    "B25004_007E": "vacancy_migrant",
    "B25004_008E": "vacancy_other",

    # Year structure built (B25034)
    "B25034_001E": "structure_built_total",
    "B25034_002E": "structure_built_2020_plus",
    "B25034_003E": "structure_built_2010s",
    "B25034_004E": "structure_built_2000s",
    "B25034_005E": "structure_built_1990s",
    "B25034_006E": "structure_built_1980s",
    "B25034_007E": "structure_built_1970s",
    "B25034_008E": "structure_built_1960s",
    "B25034_009E": "structure_built_1950s",
    "B25034_010E": "structure_built_1940s",
    "B25034_011E": "structure_built_pre1940",

    # Average household size (B25010)
    "B25010_001E": "avg_household_size_total",
    "B25010_002E": "avg_household_size_owner",
    "B25010_003E": "avg_household_size_renter",

    # Occupants per room / overcrowding (B25014)
    "B25014_001E": "occupants_per_room_total",
    "B25014_002E": "occupants_per_room_owner_total",
    "B25014_005E": "occupants_per_room_owner_101_150",
    "B25014_006E": "occupants_per_room_owner_151_200",
    "B25014_007E": "occupants_per_room_owner_201_plus",
    "B25014_008E": "occupants_per_room_renter_total",
    "B25014_011E": "occupants_per_room_renter_101_150",
    "B25014_012E": "occupants_per_room_renter_151_200",
    "B25014_013E": "occupants_per_room_renter_201_plus",

    # Mobility (B07003)
    "B07003_001E": "mobility_total",
    "B07003_004E": "mobility_same_house",
    "B07003_007E": "mobility_diff_house_same_county",
    "B07003_010E": "mobility_diff_county_same_state",
    "B07003_013E": "mobility_diff_state",
    "B07003_016E": "mobility_abroad",
}

# Variables that may only be available at tract level (medians with small sample sizes)
# We'll try to pull these but expect some may fail
MAYBE_BG_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B25064_001E": "median_rent",
    "B25077_001E": "median_home_value",
}

# Family household details from B11003
FAMILY_HH_VARIABLES = {
    "B11003_001E": "hh_family_total_b11003",
    "B11003_002E": "hh_family_married",
    "B11003_003E": "hh_family_married_children",
    "B11003_009E": "hh_family_male_no_spouse",
    "B11003_010E": "hh_family_male_no_spouse_children",
    "B11003_015E": "hh_family_female_no_spouse",
    "B11003_016E": "hh_family_female_no_spouse_children",
}


def fetch_block_group_data(
    state_fips: str,
    variables: dict[str, str],
    acs_year: int,
    api_key: str | None = None,
) -> pd.DataFrame | None:
    """Fetch ACS data for all block groups in a state."""
    acs_base_url = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    var_codes = list(variables.keys())

    # Census API limits to 50 variables per request
    chunk_size = 48  # Leave room for geography identifiers
    all_data = None

    for i in range(0, len(var_codes), chunk_size):
        chunk_vars = var_codes[i:i + chunk_size]
        var_str = ",".join(chunk_vars)

        params = {
            "get": f"NAME,{var_str}",
            "for": "block group:*",
            "in": f"state:{state_fips}&in=county:*&in=tract:*",
        }
        if api_key:
            params["key"] = api_key

        url = f"{acs_base_url}?get=NAME,{var_str}&for=block%20group:*&in=state:{state_fips}&in=county:*&in=tract:*"
        if api_key:
            url += f"&key={api_key}"

        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching state {state_fips} chunk {i}: {e}")
            return None
        except ValueError as e:
            print(f"  JSON decode error for state {state_fips}: {e}")
            print(f"  Response: {resp.text[:500]}")
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
            # Merge on geography columns
            geo_cols = ["NAME", "state", "county", "tract", "block group"]
            all_data = all_data.merge(
                chunk_df,
                on=[c for c in geo_cols if c in chunk_df.columns],
                how="outer",
            )

        time.sleep(0.1)  # Rate limiting

    return all_data


def build_block_group_id(row: pd.Series) -> str:
    """Construct 12-digit block group GEOID."""
    return f"{row['state']}{row['county']}{row['tract']}{row['block group']}"


def main():
    parser = argparse.ArgumentParser(description="Pull ACS block group data")
    parser.add_argument(
        "--acs-year",
        type=int,
        default=2024,
        help="ACS endpoint year, e.g. 2024 for ACS 2020-2024 5-year.",
    )
    parser.add_argument("--api-key", help="Census API key (optional but recommended)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--states",
        nargs="*",
        help="Specific state FIPS codes to pull (default: all)",
    )
    args = parser.parse_args()

    acs_start_year = args.acs_year - 4
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/ACS-5yr-{acs_start_year}-{args.acs_year}")
    output_dir.mkdir(parents=True, exist_ok=True)

    states = args.states if args.states else STATE_FIPS

    # Combine all variables
    all_variables = {**BG_VARIABLES, **FAMILY_HH_VARIABLES}

    # Try to add median variables (may fail for some)
    all_variables.update(MAYBE_BG_VARIABLES)

    print(f"Pulling ACS {acs_start_year}-{args.acs_year} block group data...")
    print(f"Pulling {len(all_variables)} variables for {len(states)} states...")
    print(f"Variables: {len(BG_VARIABLES)} core + {len(FAMILY_HH_VARIABLES)} family + {len(MAYBE_BG_VARIABLES)} median (may fail)")

    all_state_dfs = []
    failed_states = []

    for i, state_fips in enumerate(states):
        print(f"[{i+1}/{len(states)}] Fetching state {state_fips}...", end=" ", flush=True)

        df = fetch_block_group_data(state_fips, all_variables, args.acs_year, args.api_key)

        if df is not None and len(df) > 0:
            print(f"got {len(df)} block groups")
            all_state_dfs.append(df)
        else:
            print("FAILED")
            failed_states.append(state_fips)

        time.sleep(0.2)  # Rate limiting between states

    if not all_state_dfs:
        print("No data retrieved!")
        return 1

    print(f"\nCombining {len(all_state_dfs)} states...")
    combined = pd.concat(all_state_dfs, ignore_index=True)

    # Build block group ID
    combined["bg_id"] = combined.apply(build_block_group_id, axis=1)

    # Also extract tract_id for joining
    combined["tract_id"] = combined["bg_id"].str[:11]
    combined["state_fips"] = combined["state"]
    combined["county_fips"] = combined["county"]

    # Rename columns to friendly names
    rename_map = {v: k for k, v in {**all_variables}.items()}
    # Invert: code -> name
    rename_map = {code: name for code, name in all_variables.items()}
    combined = combined.rename(columns=rename_map)

    # Convert numeric columns
    for col in all_variables.values():
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Reorder columns
    id_cols = ["bg_id", "tract_id", "state_fips", "county_fips", "NAME"]
    other_cols = [c for c in combined.columns if c not in id_cols and c not in ["state", "county", "tract", "block group"]]
    combined = combined[id_cols + sorted(other_cols)]

    # Save
    output_path = output_dir / "acs_block_groups.parquet"
    combined.to_parquet(output_path, index=False)
    print(f"\nSaved {len(combined)} block groups to {output_path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Total block groups: {len(combined)}")
    print(f"  States succeeded: {len(all_state_dfs)}")
    print(f"  States failed: {len(failed_states)}")
    if failed_states:
        print(f"  Failed states: {failed_states}")

    # Check for missing median variables
    print(f"\nMedian variable availability:")
    for var_name in MAYBE_BG_VARIABLES.values():
        if var_name in combined.columns:
            non_null = combined[var_name].notna().sum()
            pct = 100 * non_null / len(combined)
            print(f"  {var_name}: {non_null:,} non-null ({pct:.1f}%)")
        else:
            print(f"  {var_name}: NOT AVAILABLE at block group level")

    return 0


if __name__ == "__main__":
    exit(main())
