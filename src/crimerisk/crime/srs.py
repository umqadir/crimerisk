from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.utils.zipfiles import extract_zip_member


SRS_OFFENSE_COLUMN_MAP: dict[str, str] = {
    "murder": "actual_murder",
    "rape": "actual_rape_total",
    "robbery": "actual_robbery_total",
    "aggravated_assault": "actual_assault_aggravated",
    "burglary": "actual_burglary_total",
    "larceny": "actual_theft_total",
    "motor_vehicle_theft": "actual_motor_vehicle_theft_total",
}


@dataclass(frozen=True)
class SrsExtract:
    parquet_path: Path


@dataclass(frozen=True)
class SrsMonthlyExtract:
    parquet_path: Path


def ensure_srs_year_parquet(
    *,
    zip_path: Path,
    cache_dir: Path,
) -> SrsExtract:
    member = "offenses_known_yearly_1960_2024.parquet"
    out_path = cache_dir / "offenses_known_year" / member
    extract_zip_member(zip_path, member, out_path)
    return SrsExtract(parquet_path=out_path)


def ensure_srs_month_year_parquet(
    *,
    zip_path: Path,
    year: int,
    cache_dir: Path,
) -> SrsMonthlyExtract:
    member = f"offenses_known_monthly_{year}.parquet"
    out_path = cache_dir / "offenses_known_month" / member
    extract_zip_member(zip_path, member, out_path)
    return SrsMonthlyExtract(parquet_path=out_path)


def load_srs_agency_year_part1(
    *,
    srs_parquet_path: Path,
    year_start: int,
    year_end: int,
    agency_types: tuple[str, ...] = ("local police department", "constable/marshal"),
) -> pd.DataFrame:
    offense_select = ",\n      ".join(
        f"CAST({col} AS DOUBLE) AS {offense}"
        for offense, col in SRS_OFFENSE_COLUMN_MAP.items()
        if offense in OFFENSES_7
    )
    agency_types_sql = ", ".join([f"'{t}'" for t in agency_types])
    query = f"""
    SELECT
      (ori || '00') AS ori9,
      ori AS ori7,
      ori9 AS kaplan_ori9,
      CAST(year AS INTEGER) AS year,
      fips_state_code AS state_fips,
      fips_county_code AS county_fips,
      fips_place_code AS place_fips,
      agency_name,
      agency_type,
      CAST(number_of_months_missing AS INTEGER) AS months_missing,
      CASE
        WHEN number_of_months_missing IS NULL THEN NULL
        ELSE GREATEST(0, LEAST(12, 12 - CAST(number_of_months_missing AS INTEGER)))
      END AS months_reported,
      {offense_select}
    FROM read_parquet('{srs_parquet_path.as_posix()}')
    WHERE CAST(year AS INTEGER) BETWEEN {year_start} AND {year_end}
      AND agency_type IN ({agency_types_sql})
    """
    df = duckdb.sql(query).df()
    df["ori9"] = df["ori9"].astype(str)
    df["ori7"] = df["ori7"].astype(str)
    df["kaplan_ori9"] = df["kaplan_ori9"].astype(str)
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(3)
    df["place_fips"] = df["place_fips"].astype(str).str.zfill(5)
    for offense in OFFENSES_7:
        df[offense] = pd.to_numeric(df[offense], errors="coerce")
    return df
