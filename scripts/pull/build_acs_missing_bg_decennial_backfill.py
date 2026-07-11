#!/usr/bin/env python3
"""
Build the decennial-2020 backfill table for published block groups absent from the ACS BG source.

The ACS 5-year BG release can drop legacy-2020 GEOIDs from its vocabulary entirely (observed:
ACS 2020-2024 omits 13 Suffolk County NY tracts plus 2 BGs of tract 36103146001 — 39 BGs whose
population is absent from ACS, not renumbered into successor tracts; TIGER2024 still carries the
same legacy geometry). Connecticut's planning-region recoding is a separate, purely administrative
1:1 relabel handled by the geometry remap in crimerisk.covariates.features and is excluded here.

For every block group in the published legacy-2020 universe (the jurisdiction crosswalk) whose
GEOID is absent from the ACS BG parquet — outside Connecticut — this script pulls 2020 Decennial
DHC counts (P1_001N total population; H1_001N housing units; H3_002N occupied / H3_003N vacant)
from the Census API and writes configs/acs_missing_bg_decennial_backfill.csv. The covariates
build injects these rows as the survey-base values BEFORE county-control scaling, and the release
validator fails closed on any published BG absent from ACS that is covered by neither the CT
relabel nor this table. Re-run this script whenever the ACS vintage or BG universe changes.

Usage:
    python scripts/pull/build_acs_missing_bg_decennial_backfill.py [--api-key KEY]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACS_BG_PARQUET = REPO_ROOT / "data" / "ACS-5yr-2020-2024" / "parsed" / "acs_block_groups.parquet"
DEFAULT_CROSSWALK_PARQUET = REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "configs" / "acs_missing_bg_decennial_backfill.csv"

# Connecticut's ACS planning-region vocabulary is handled by the geometry relabel in
# crimerisk.covariates.features (build_ct_bg_2023_to_2020_map), not by decennial backfill.
CT_STATE_FIPS = "09"

DHC_DATASET_URL = "https://api.census.gov/data/2020/dec/dhc"
DHC_VARIABLES = {
    "P1_001N": "population_dec2020",
    "H1_001N": "housing_units_dec2020",
    "H3_002N": "households_occupied_dec2020",
    "H3_003N": "housing_vacant_dec2020",
}


def fetch_dhc_county_block_groups(state_fips: str, county_fips: str, api_key: str | None) -> pd.DataFrame:
    var_str = ",".join(DHC_VARIABLES)
    url = (
        f"{DHC_DATASET_URL}?get={var_str}"
        f"&for=block%20group:*&in=state:{state_fips}&in=county:{county_fips}&in=tract:*"
    )
    if api_key:
        url += f"&key={api_key}"
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    else:
        raise SystemExit(f"ERROR: Census DHC API failed after retries for {state_fips}{county_fips}: {last_error}")
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["bg_id"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    for code in DHC_VARIABLES:
        df[code] = pd.to_numeric(df[code], errors="coerce")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acs-bg-parquet", type=Path, default=DEFAULT_ACS_BG_PARQUET)
    parser.add_argument("--crosswalk-parquet", type=Path, default=DEFAULT_CROSSWALK_PARQUET)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--api-key", default=os.environ.get("CENSUS_API_KEY"))
    args = parser.parse_args()

    acs_ids = set(
        pd.read_parquet(args.acs_bg_parquet, columns=["bg_id"])["bg_id"].astype(str).str.zfill(12)
    )
    universe = (
        pd.read_parquet(args.crosswalk_parquet, columns=["block_group_geoid"])["block_group_geoid"]
        .astype(str)
        .str.zfill(12)
        .drop_duplicates()
    )
    missing = universe[~universe.isin(acs_ids) & ~universe.str.startswith(CT_STATE_FIPS)].sort_values()
    print(f"Universe BGs: {len(universe)}; absent from ACS outside CT: {len(missing)}")
    by_state = missing.str.slice(0, 2).value_counts().sort_index()
    for state, n in by_state.items():
        print(f"  state {state}: {n} BGs")

    rows: list[pd.DataFrame] = []
    counties = sorted({(g[:2], g[2:5]) for g in missing})
    for state_fips, county_fips in counties:
        print(f"Fetching DHC 2020 block groups for {state_fips}{county_fips}...")
        dhc = fetch_dhc_county_block_groups(state_fips, county_fips, args.api_key)
        wanted = missing[missing.str.startswith(state_fips + county_fips)]
        found = dhc[dhc["bg_id"].isin(set(wanted))].copy()
        absent_from_dhc = sorted(set(wanted) - set(found["bg_id"]))
        if absent_from_dhc:
            raise SystemExit(
                f"ERROR: {len(absent_from_dhc)} BGs absent from both ACS and decennial DHC in "
                f"{state_fips}{county_fips} (no backfill source): {absent_from_dhc[:10]}"
            )
        rows.append(found)
        time.sleep(0.2)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["bg_id", *DHC_VARIABLES])
    out = out.rename(columns=DHC_VARIABLES)
    out["bg_id"] = out["bg_id"].astype(str).str.zfill(12)
    out["tract_id"] = out["bg_id"].str.slice(0, 11)
    out["state_fips"] = out["bg_id"].str.slice(0, 2)
    out["county_fips"] = out["bg_id"].str.slice(2, 5)
    out["source"] = "census_2020_dec_dhc_api"
    out = out[
        [
            "bg_id",
            "tract_id",
            "state_fips",
            "county_fips",
            "population_dec2020",
            "housing_units_dec2020",
            "households_occupied_dec2020",
            "housing_vacant_dec2020",
            "source",
        ]
    ].sort_values("bg_id")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} backfill rows to {args.output_csv}")
    print(
        f"Totals: population={out['population_dec2020'].sum():.0f}"
        f" housing_units={out['housing_units_dec2020'].sum():.0f}"
        f" occupied={out['households_occupied_dec2020'].sum():.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
