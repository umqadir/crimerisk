"""Empirically select the NIBRS-to-SRS-equivalent counting-rule variants.

For each candidate variant of each offense's counting rule, recompute agency totals
from the raw NIBRS segments and compare against the FBI's converted Return A totals
on dual-reporting agencies. The variant whose aggregate ratio is closest to 1.0 (and
stable across years) is the one the production rollup implements.

Rule space (docs/FBI-DATA-GUIDE.md section 4; FBI UCR Handbook scoring rules):
  murder              per victim; variants: victim-row fallback handling
  rape                11A+11B+11C per victim, murder supersedes
  aggravated assault  per victim, murder supersedes; variants: also rape supersedes,
                      robbery-incident suppression
  robbery             per incident; variants: murder / murder+rape suppression
  burglary            per incident with premises multiplier; variants: gate on
                      hotel/motel, rental storage, both, none
  larceny             per incident; variant: full hierarchy suppression
  motor vehicle theft per stolen vehicle; variants: per incident, hierarchy suppression

Writes state/qa/nibrs_srs_rule_tuning.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import get_paths

YEARS = (2022, 2023, 2024)

PATHS = get_paths()
REGIMES = PATHS.state_dir / "modeling" / "agency_year_reporting_regimes.parquet"
REPORT_PATH = PATHS.state_dir / "qa" / "nibrs_srs_rule_tuning.md"

RAPE_CODES = (
    "'sex offenses - rape','sex offenses - sodomy','sex offenses - sexual assault with an object'"
)

VARIANT_COLUMNS = {
    "murder": ["murder_v"],
    "rape": ["rape_v"],
    "aggravated_assault": ["asslt_v_m", "asslt_v_mr", "asslt_v_mr_norob"],
    "robbery": ["rob_all", "rob_h_m", "rob_h_mr"],
    "burglary": ["burg_inc", "burg_hotel", "burg_storage", "burg_both", "burg_both_hsupp"],
    "larceny": ["larc_all", "larc_hsupp"],
    "motor_vehicle_theft": ["mvt_veh", "mvt_inc", "mvt_veh_hsupp"],
}


def per_ori_variant_counts(con: duckdb.DuckDBPyConnection, year: int) -> str:
    """Register a view of per-ORI counts for every rule variant; return view name."""
    off = PATHS.cache_dir / "nibrs_offense" / f"nibrs_offense_segment_{year}.parquet"
    vic = PATHS.cache_dir / "nibrs_victim" / f"nibrs_victim_segment_{year}.parquet"
    prop = PATHS.cache_dir / "nibrs_property" / f"nibrs_property_segment_{year}.parquet"
    codes_list = ",".join(
        f"lower(CAST(ucr_offense_code_{i} AS VARCHAR))" for i in range(1, 11)
    )
    view = f"variants_{year}"
    con.sql(f"""
    CREATE OR REPLACE VIEW {view} AS
    WITH off AS (
      SELECT ori, unique_incident_id,
        MAX(CASE WHEN lower(ucr_offense_code)='murder/nonnegligent manslaughter' THEN 1 ELSE 0 END) has_murder,
        MAX(CASE WHEN lower(ucr_offense_code) IN ({RAPE_CODES}) THEN 1 ELSE 0 END) has_rape,
        MAX(CASE WHEN lower(ucr_offense_code)='robbery' THEN 1 ELSE 0 END) has_robbery,
        MAX(CASE WHEN lower(ucr_offense_code)='assault offenses - aggravated assault' THEN 1 ELSE 0 END) has_aggasslt,
        MAX(CASE WHEN lower(ucr_offense_code)='burglary/breaking and entering' THEN 1 ELSE 0 END) has_burglary,
        MAX(CASE WHEN lower(ucr_offense_code) LIKE 'larceny/theft offenses%' THEN 1 ELSE 0 END) has_larceny,
        MAX(CASE WHEN lower(ucr_offense_code)='motor vehicle theft' THEN 1 ELSE 0 END) has_mvt,
        MAX(CASE WHEN lower(ucr_offense_code)='burglary/breaking and entering'
                  AND lower(location_type)='hotel/motel/etc.'
             THEN TRY_CAST(number_of_premises_entered AS DOUBLE) END) prem_hotel,
        MAX(CASE WHEN lower(ucr_offense_code)='burglary/breaking and entering'
                  AND lower(location_type)='rental storage facility'
             THEN TRY_CAST(number_of_premises_entered AS DOUBLE) END) prem_storage
      FROM read_parquet('{off.as_posix()}')
      GROUP BY 1, 2
    ),
    vic_rows AS (
      SELECT ori, unique_incident_id, [{codes_list}] codes
      FROM read_parquet('{vic.as_posix()}')
    ),
    vic AS (
      SELECT ori, unique_incident_id,
        SUM(CASE WHEN list_contains(codes,'murder/nonnegligent manslaughter') THEN 1 ELSE 0 END) murder_victims,
        SUM(CASE WHEN (list_contains(codes,'sex offenses - rape')
                OR list_contains(codes,'sex offenses - sodomy')
                OR list_contains(codes,'sex offenses - sexual assault with an object'))
              AND NOT list_contains(codes,'murder/nonnegligent manslaughter') THEN 1 ELSE 0 END) rape_victims,
        SUM(CASE WHEN list_contains(codes,'assault offenses - aggravated assault')
              AND NOT list_contains(codes,'murder/nonnegligent manslaughter') THEN 1 ELSE 0 END) asslt_victims_m,
        SUM(CASE WHEN list_contains(codes,'assault offenses - aggravated assault')
              AND NOT list_contains(codes,'murder/nonnegligent manslaughter')
              AND NOT (list_contains(codes,'sex offenses - rape')
                OR list_contains(codes,'sex offenses - sodomy')
                OR list_contains(codes,'sex offenses - sexual assault with an object'))
             THEN 1 ELSE 0 END) asslt_victims_mr
      FROM vic_rows
      GROUP BY 1, 2
    ),
    veh AS (
      SELECT ori, unique_incident_id,
        MAX(TRY_CAST(number_of_stolen_motor_vehicles AS DOUBLE)) nveh
      FROM read_parquet('{prop.as_posix()}')
      GROUP BY 1, 2
    ),
    inc AS (
      SELECT o.*,
        v.unique_incident_id IS NOT NULL AS has_victim_rows,
        COALESCE(v.murder_victims, 0) murder_victims,
        COALESCE(v.rape_victims, 0) rape_victims,
        COALESCE(v.asslt_victims_m, 0) asslt_victims_m,
        COALESCE(v.asslt_victims_mr, 0) asslt_victims_mr,
        veh.nveh
      FROM off o
      LEFT JOIN vic v ON o.ori = v.ori AND o.unique_incident_id = v.unique_incident_id
      LEFT JOIN veh ON o.ori = veh.ori AND o.unique_incident_id = veh.unique_incident_id
    )
    SELECT ori,
      SUM(CASE WHEN has_murder=1 THEN
        CASE WHEN has_victim_rows THEN GREATEST(murder_victims, 1) ELSE 1 END
        ELSE 0 END) murder_v,
      SUM(CASE WHEN has_rape=1 THEN
        CASE WHEN has_victim_rows THEN rape_victims ELSE 1 END
        ELSE 0 END) rape_v,
      SUM(CASE WHEN has_aggasslt=1 THEN
        CASE WHEN has_victim_rows THEN asslt_victims_m ELSE 1 END
        ELSE 0 END) asslt_v_m,
      SUM(CASE WHEN has_aggasslt=1 THEN
        CASE WHEN has_victim_rows THEN asslt_victims_mr ELSE 1 END
        ELSE 0 END) asslt_v_mr,
      SUM(CASE WHEN has_aggasslt=1 AND has_robbery=0 THEN
        CASE WHEN has_victim_rows THEN asslt_victims_mr ELSE 1 END
        ELSE 0 END) asslt_v_mr_norob,
      SUM(has_robbery) rob_all,
      SUM(CASE WHEN has_robbery=1 AND has_murder=0 THEN 1 ELSE 0 END) rob_h_m,
      SUM(CASE WHEN has_robbery=1 AND has_murder=0 AND has_rape=0 THEN 1 ELSE 0 END) rob_h_mr,
      SUM(has_burglary) burg_inc,
      SUM(CASE WHEN has_burglary=1 THEN GREATEST(COALESCE(prem_hotel, 1), 1) ELSE 0 END) burg_hotel,
      SUM(CASE WHEN has_burglary=1 THEN GREATEST(COALESCE(prem_storage, 1), 1) ELSE 0 END) burg_storage,
      SUM(CASE WHEN has_burglary=1 THEN
        GREATEST(COALESCE(prem_hotel, 0), COALESCE(prem_storage, 0), 1)
        ELSE 0 END) burg_both,
      SUM(CASE WHEN has_burglary=1 AND has_murder=0 AND has_rape=0 AND has_robbery=0 AND has_aggasslt=0 THEN
        GREATEST(COALESCE(prem_hotel, 0), COALESCE(prem_storage, 0), 1)
        ELSE 0 END) burg_both_hsupp,
      SUM(has_larceny) larc_all,
      SUM(CASE WHEN has_larceny=1 AND has_murder=0 AND has_rape=0 AND has_robbery=0
                AND has_aggasslt=0 AND has_burglary=0 THEN 1 ELSE 0 END) larc_hsupp,
      SUM(CASE WHEN has_mvt=1 THEN GREATEST(COALESCE(nveh, 1), 1) ELSE 0 END) mvt_veh,
      SUM(has_mvt) mvt_inc,
      SUM(CASE WHEN has_mvt=1 AND has_murder=0 AND has_rape=0 AND has_robbery=0
                AND has_aggasslt=0 AND has_burglary=0 AND has_larceny=0
           THEN GREATEST(COALESCE(nveh, 1), 1) ELSE 0 END) mvt_veh_hsupp
    FROM inc
    GROUP BY 1
    """)
    return view


con = duckdb.connect()
lines = [
    "# NIBRS SRS-equivalent rule tuning",
    "",
    f"Variant totals vs FBI-converted Return A on dual-reporting agencies, years {YEARS}.",
    "Ratio = Return A total / variant recomputation; closest to 1.0 wins.",
    "",
]

results: dict[str, dict[str, list[float]]] = {}
for year in YEARS:
    view = per_ori_variant_counts(con, year)
    # dual set: ORIs with both a populated Return A row and a NIBRS rollup row this year
    con.sql(f"""
    CREATE OR REPLACE VIEW dual_{year} AS
    SELECT DISTINCT ori9 FROM read_parquet('{REGIMES.as_posix()}')
    WHERE year={year} AND srs_count IS NOT NULL AND nibrs_count IS NOT NULL
    """)
    lines.append(f"## {year}")
    lines.append("")
    lines.append("| offense | variant | return_a_total | variant_total | ratio |")
    lines.append("| --- | --- | --- | --- | --- |")
    for offense, variants in VARIANT_COLUMNS.items():
        srs_total = con.sql(f"""
        SELECT SUM(srs_count) FROM read_parquet('{REGIMES.as_posix()}')
        WHERE year={year} AND offense='{offense}' AND srs_count IS NOT NULL
          AND ori9 IN (SELECT ori9 FROM dual_{year})
        """).fetchone()[0]
        for variant in variants:
            variant_total = con.sql(f"""
            SELECT SUM(v.{variant}) FROM {view} v
            WHERE v.ori IN (SELECT ori9 FROM dual_{year})
              AND v.ori IN (
                SELECT ori9 FROM read_parquet('{REGIMES.as_posix()}')
                WHERE year={year} AND offense='{offense}' AND srs_count IS NOT NULL
              )
            """).fetchone()[0]
            ratio = (srs_total / variant_total) if variant_total else float("nan")
            results.setdefault(offense, {}).setdefault(variant, []).append(ratio)
            lines.append(
                f"| {offense} | {variant} | {int(srs_total):,} | {int(variant_total):,} | {ratio:.4f} |"
            )
    lines.append("")

lines.append("## Mean absolute deviation from 1.0 across years")
lines.append("")
lines.append("| offense | variant | mean_abs_dev | per-year ratios |")
lines.append("| --- | --- | --- | --- |")
for offense, variants in results.items():
    best = min(variants, key=lambda v: sum(abs(r - 1) for r in variants[v]) / len(variants[v]))
    for variant, ratios in variants.items():
        mad = sum(abs(r - 1) for r in ratios) / len(ratios)
        marker = " **<- best**" if variant == best else ""
        lines.append(
            f"| {offense} | {variant} | {mad:.4f} | "
            + ", ".join(f"{r:.4f}" for r in ratios)
            + marker
            + " |"
        )

report = "\n".join(lines)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report)
print(report)
print(f"\nwrote {REPORT_PATH}")
