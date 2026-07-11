#!/usr/bin/env python3
"""
Extract block group centroids from TIGER shapefiles.

Reads the zip files in data/tiger_bg/ and extracts centroid coordinates
for use in bucket assignment.

Usage:
    python scripts/pull/extract_bg_centroids.py

Output:
    data/tiger_bg/parsed/bg_centroids.parquet
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd


def extract_centroids_from_zip(zip_path: Path) -> pd.DataFrame | None:
    """Extract block group centroids from a TIGER shapefile zip."""
    try:
        # Read shapefile directly from zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the .shp file
            shp_files = [n for n in zf.namelist() if n.endswith(".shp")]
            if not shp_files:
                print(f"  No .shp file in {zip_path.name}")
                return None

            shp_name = shp_files[0]

        # geopandas can read directly from zip
        gdf = gpd.read_file(f"zip://{zip_path}!{shp_name}")

        # Get centroids (in lat/lon)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        centroids = gdf.geometry.centroid

        df = pd.DataFrame({
            "bg_id": gdf["GEOID"].astype(str).str.zfill(12),
            "state_fips": gdf["STATEFP"].astype(str).str.zfill(2),
            "county_fips": gdf["COUNTYFP"].astype(str).str.zfill(3),
            "tract_fips": gdf["TRACTCE"].astype(str).str.zfill(6),
            "bg_fips": gdf["BLKGRPCE"].astype(str),
            "tract_id": gdf["GEOID"].astype(str).str[:11].str.zfill(11),
            "aland": gdf["ALAND"],  # Land area in sq meters
            "awater": gdf["AWATER"],  # Water area in sq meters
            "lon": centroids.x,
            "lat": centroids.y,
        })

        return df

    except Exception as e:
        print(f"  Error processing {zip_path.name}: {e}")
        return None


def main():
    tiger_dir = Path("data/tiger_bg")
    output_path = tiger_dir / "parsed" / "bg_centroids.parquet"

    if not tiger_dir.exists():
        print(f"Error: {tiger_dir} not found")
        return 1

    zip_files = sorted(tiger_dir.glob("tl_2020_*_bg.zip"))
    print(f"Found {len(zip_files)} TIGER block group shapefiles")

    all_dfs = []
    failed = []

    for i, zip_path in enumerate(zip_files):
        state_fips = zip_path.stem.split("_")[2]
        print(f"[{i+1}/{len(zip_files)}] Processing state {state_fips}...", end=" ", flush=True)

        df = extract_centroids_from_zip(zip_path)
        if df is not None:
            print(f"got {len(df)} block groups")
            all_dfs.append(df)
        else:
            print("FAILED")
            failed.append(state_fips)

    if not all_dfs:
        print("No data extracted!")
        return 1

    print(f"\nCombining {len(all_dfs)} states...")
    combined = pd.concat(all_dfs, ignore_index=True)

    # Sort by bg_id
    combined = combined.sort_values("bg_id").reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"\nSaved {len(combined)} block group centroids to {output_path}")

    # Summary stats
    print(f"\nSummary:")
    print(f"  Total block groups: {len(combined):,}")
    print(f"  Unique tracts: {combined['tract_id'].nunique():,}")
    print(f"  States: {combined['state_fips'].nunique()}")
    print(f"  Avg block groups per tract: {len(combined) / combined['tract_id'].nunique():.2f}")
    if failed:
        print(f"  Failed states: {failed}")

    return 0


if __name__ == "__main__":
    exit(main())
