"""NIBRS-to-SRS-equivalent annual rollup.

Counts are derived from the raw NIBRS segments using the FBI's documented SRS scoring
rules (UCR Handbook), with rule variants selected empirically against the FBI's own
converted Return A rows on dual-reporting agencies (scripts/diagnostics/
tune_nibrs_srs_rules.py; docs/FBI-DATA-GUIDE.md section 4):

  murder               one per victim (victim segment); murder is never suppressed
  rape                 11A + 11B + 11C (revised definition), one per victim,
                       murder supersedes per victim
  robbery              one per incident after murder/rape hierarchy suppression
  aggravated assault   one per victim after murder/rape/robbery hierarchy
                       suppression
  burglary             one per incident, suppressed when the incident also contains a
                       higher Part-I category; premises multiplier (hotel rule) where
                       location is hotel/motel or rental storage facility, capped at
                       max(observed legitimate p95, 30), with 99 treated as a sentinel
  larceny              one per incident, suppressed when any higher Part I category
                       is present in the incident, including motor vehicle theft
  motor vehicle theft  one per stolen vehicle (property segment), suppressed when a
                       higher Part-I non-larceny category is present

Selection evidence: aggregate ratios vs converted Return A land within ~1-2% of 1.0
for 2023 and 2024 dual agencies (state/qa/nibrs_srs_rule_tuning.md). 2022 and earlier
run looser because mid-year NIBRS transitioners have full-year Return A rows but only
partial-year NIBRS segments — a composition effect, not a rule error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.utils.zipfiles import extract_zip_member


NIBRS_RAPE_CODES_SQL = (
    "'sex offenses - rape','sex offenses - sodomy',"
    "'sex offenses - sexual assault with an object'"
)

# Data Element 10 (number of premises entered) is only collected at these two
# location types; they are the only locations where the SRS hotel rule applies.
NIBRS_HOTEL_RULE_LOCATIONS_SQL = "'hotel/motel/etc.','rental storage facility'"


@dataclass(frozen=True)
class NibrsExtract:
    parquet_path: Path


@dataclass(frozen=True)
class NibrsBatchHeaderExtract:
    parquet_path: Path


def ensure_nibrs_offense_year_parquet(
    *,
    zip_path: Path,
    year: int,
    cache_dir: Path,
) -> NibrsExtract:
    member = f"nibrs_offense_segment_{year}.parquet"
    out_path = cache_dir / "nibrs_offense" / member
    extract_zip_member(zip_path, member, out_path)
    return NibrsExtract(parquet_path=out_path)


def ensure_nibrs_victim_year_parquet(
    *,
    zip_path: Path,
    year: int,
    cache_dir: Path,
) -> NibrsExtract:
    member = f"nibrs_victim_segment_{year}.parquet"
    out_path = cache_dir / "nibrs_victim" / member
    extract_zip_member(zip_path, member, out_path)
    return NibrsExtract(parquet_path=out_path)


def ensure_nibrs_property_year_parquet(
    *,
    zip_path: Path,
    year: int,
    cache_dir: Path,
) -> NibrsExtract:
    member = f"nibrs_property_segment_{year}.parquet"
    out_path = cache_dir / "nibrs_property" / member
    extract_zip_member(zip_path, member, out_path)
    return NibrsExtract(parquet_path=out_path)


def ensure_nibrs_batch_header_parquet(
    *,
    zip_path: Path,
    cache_dir: Path,
) -> NibrsBatchHeaderExtract:
    member = "nibrs_batch_header_1991_2024.parquet"
    out_path = cache_dir / "nibrs_batch_header" / member
    extract_zip_member(zip_path, member, out_path)
    return NibrsBatchHeaderExtract(parquet_path=out_path)


def srs_equivalent_incident_counts_sql(
    *,
    offense_parquet_path: Path,
    victim_parquet_path: Path,
    property_parquet_path: Path,
) -> str:
    """SQL selecting per-incident SRS-equivalent counts for the seven categories.

    Output columns: ori9, year, unique_incident_id, incident_month, then one count
    column per offense in OFFENSES_7. Incidents with no Part I content appear with
    all-zero counts so that month coverage over all NIBRS activity stays computable.
    """
    victim_codes_list = ",".join(
        f"lower(CAST(ucr_offense_code_{i} AS VARCHAR))" for i in range(1, 11)
    )
    return f"""
    WITH off AS (
      SELECT
        ori,
        CAST(year AS INTEGER) AS year,
        unique_incident_id,
        MAX(SUBSTR(incident_date, 1, 7)) AS incident_month,
        MAX(CASE WHEN lower(ucr_offense_code) = 'murder/nonnegligent manslaughter'
            THEN 1 ELSE 0 END) AS has_murder,
        MAX(CASE WHEN lower(ucr_offense_code) IN ({NIBRS_RAPE_CODES_SQL})
            THEN 1 ELSE 0 END) AS has_rape,
        MAX(CASE WHEN lower(ucr_offense_code) = 'robbery'
            THEN 1 ELSE 0 END) AS has_robbery,
        MAX(CASE WHEN lower(ucr_offense_code) = 'assault offenses - aggravated assault'
            THEN 1 ELSE 0 END) AS has_aggasslt,
        MAX(CASE WHEN lower(ucr_offense_code) = 'burglary/breaking and entering'
            THEN 1 ELSE 0 END) AS has_burglary,
        MAX(CASE WHEN lower(ucr_offense_code) LIKE 'larceny/theft offenses%'
            THEN 1 ELSE 0 END) AS has_larceny,
        MAX(CASE WHEN lower(ucr_offense_code) = 'motor vehicle theft'
            THEN 1 ELSE 0 END) AS has_mvt,
        MAX(CASE WHEN lower(ucr_offense_code) = 'burglary/breaking and entering'
                  AND lower(location_type) IN ({NIBRS_HOTEL_RULE_LOCATIONS_SQL})
            THEN TRY_CAST(number_of_premises_entered AS DOUBLE) END) AS hotel_rule_premises
      FROM read_parquet('{offense_parquet_path.as_posix()}')
      GROUP BY 1, 2, 3
    ),
    premises_cap AS (
      SELECT
        GREATEST(
          COALESCE(
            quantile_cont(TRY_CAST(number_of_premises_entered AS DOUBLE), 0.95)
              FILTER (
                WHERE lower(ucr_offense_code) = 'burglary/breaking and entering'
                  AND lower(location_type) IN ({NIBRS_HOTEL_RULE_LOCATIONS_SQL})
                  AND TRY_CAST(number_of_premises_entered AS DOUBLE) > 0
                  AND TRY_CAST(number_of_premises_entered AS DOUBLE) <> 99
              ),
            0.0
          ),
          30.0
        ) AS hotel_rule_premises_cap
      FROM read_parquet('{offense_parquet_path.as_posix()}')
    ),
    vic AS (
      SELECT ori, unique_incident_id,
        SUM(CASE WHEN list_contains(codes, 'murder/nonnegligent manslaughter')
            THEN 1 ELSE 0 END) AS murder_victims,
        SUM(CASE WHEN (list_contains(codes, 'sex offenses - rape')
                OR list_contains(codes, 'sex offenses - sodomy')
                OR list_contains(codes, 'sex offenses - sexual assault with an object'))
              AND NOT list_contains(codes, 'murder/nonnegligent manslaughter')
            THEN 1 ELSE 0 END) AS rape_victims,
        SUM(CASE WHEN list_contains(codes, 'assault offenses - aggravated assault')
              AND NOT list_contains(codes, 'murder/nonnegligent manslaughter')
              AND NOT (list_contains(codes, 'sex offenses - rape')
                OR list_contains(codes, 'sex offenses - sodomy')
                OR list_contains(codes, 'sex offenses - sexual assault with an object'))
            THEN 1 ELSE 0 END) AS aggasslt_victims
      FROM (
        SELECT ori, unique_incident_id, [{victim_codes_list}] AS codes
        FROM read_parquet('{victim_parquet_path.as_posix()}')
      )
      GROUP BY 1, 2
    ),
    veh AS (
      SELECT ori, unique_incident_id,
        MAX(TRY_CAST(number_of_stolen_motor_vehicles AS DOUBLE)) AS stolen_vehicles
      FROM read_parquet('{property_parquet_path.as_posix()}')
      GROUP BY 1, 2
    ),
    inc AS (
      SELECT
        o.*,
        v.unique_incident_id IS NOT NULL AS has_victim_rows,
        COALESCE(v.murder_victims, 0) AS murder_victims,
        COALESCE(v.rape_victims, 0) AS rape_victims,
        COALESCE(v.aggasslt_victims, 0) AS aggasslt_victims,
        veh.stolen_vehicles,
        CASE
          WHEN o.hotel_rule_premises IS NULL THEN NULL
          WHEN o.hotel_rule_premises = 99 THEN pc.hotel_rule_premises_cap
          ELSE LEAST(o.hotel_rule_premises, pc.hotel_rule_premises_cap)
        END AS hotel_rule_premises_capped
      FROM off o
      CROSS JOIN premises_cap pc
      LEFT JOIN vic v
        ON o.ori = v.ori AND o.unique_incident_id = v.unique_incident_id
      LEFT JOIN veh
        ON o.ori = veh.ori AND o.unique_incident_id = veh.unique_incident_id
    )
    SELECT
      ori AS ori9,
      year,
      unique_incident_id,
      incident_month,
      CASE WHEN has_murder = 1 THEN
        CASE WHEN has_victim_rows THEN GREATEST(murder_victims, 1) ELSE 1 END
        ELSE 0 END AS murder,
      CASE WHEN has_rape = 1 AND has_murder = 0 THEN
        CASE WHEN has_victim_rows THEN rape_victims ELSE 1 END
        ELSE 0 END AS rape,
      CASE WHEN has_robbery = 1 AND has_murder = 0 AND has_rape = 0
        THEN 1 ELSE 0 END AS robbery,
      CASE WHEN has_aggasslt = 1 AND has_murder = 0 AND has_rape = 0 AND has_robbery = 0 THEN
        CASE WHEN has_victim_rows THEN aggasslt_victims ELSE 1 END
        ELSE 0 END AS aggravated_assault,
      CASE WHEN has_burglary = 1 AND has_murder = 0 AND has_rape = 0
                AND has_robbery = 0 AND has_aggasslt = 0
        THEN CAST(GREATEST(COALESCE(hotel_rule_premises_capped, 0), 1) AS BIGINT)
        ELSE 0 END AS burglary,
      CASE WHEN has_larceny = 1 AND has_murder = 0 AND has_rape = 0
                AND has_robbery = 0 AND has_aggasslt = 0 AND has_burglary = 0
                AND has_mvt = 0
        THEN 1 ELSE 0 END AS larceny,
      CASE WHEN has_mvt = 1 AND has_murder = 0 AND has_rape = 0
                AND has_robbery = 0 AND has_aggasslt = 0 AND has_burglary = 0
        THEN CAST(GREATEST(COALESCE(stolen_vehicles, 0), 1) AS BIGINT)
        ELSE 0 END AS motor_vehicle_theft
    FROM inc
    """


def aggregate_nibrs_year_srs_equivalent(
    *,
    offense_parquet_path: Path,
    victim_parquet_path: Path,
    property_parquet_path: Path,
) -> pd.DataFrame:
    """Annual SRS-equivalent counts per agency and offense.

    Long output: ori9, year, offense, count, offense_incident_months,
    incident_months_any. Rows exist only for offenses with count > 0, matching the
    contract of the previous incident rollup; incident_months_any covers all NIBRS
    incidents for the agency-year (any offense type, Part I or not).
    """
    incident_sql = srs_equivalent_incident_counts_sql(
        offense_parquet_path=offense_parquet_path,
        victim_parquet_path=victim_parquet_path,
        property_parquet_path=property_parquet_path,
    )
    offense_selects = "\n      UNION ALL\n".join(
        f"""
      SELECT ori9, year, '{offense}' AS offense,
        SUM({offense}) AS count,
        COUNT(DISTINCT incident_month) FILTER (WHERE {offense} > 0) AS offense_incident_months
      FROM incident_counts
      GROUP BY 1, 2
      HAVING SUM({offense}) > 0
        """.strip()
        for offense in OFFENSES_7
    )
    query = f"""
    WITH incident_counts AS ({incident_sql}),
    months_any AS (
      SELECT ori9, year,
        COUNT(DISTINCT incident_month) AS incident_months_any
      FROM incident_counts
      WHERE incident_month IS NOT NULL
      GROUP BY 1, 2
    ),
    long AS (
      {offense_selects}
    )
    SELECT l.ori9, l.year, l.offense, l.count, l.offense_incident_months,
      m.incident_months_any
    FROM long l
    LEFT JOIN months_any m ON l.ori9 = m.ori9 AND l.year = m.year
    """
    long_df = duckdb.sql(query).df()
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "ori9",
                "year",
                "offense",
                "count",
                "offense_incident_months",
                "incident_months_any",
            ]
        )
    long_df["ori9"] = long_df["ori9"].astype(str)
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce").fillna(0).astype(int)
    return long_df
