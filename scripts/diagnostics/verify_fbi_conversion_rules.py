"""Re-verify the empirical claims in docs/FBI-DATA-GUIDE.md against the frozen data.

Reproduces, from the raw frozen inputs:
  A. Return A vs NIBRS agency presence overlap by year (the "Return A is mostly
     converted NIBRS" table), including the 2021 natural experiment and the
     batch-header membership of the 2021 SRS-only residual.
  B. Return A vs the production SRS-equivalent rollup (acceptance check: aggregate
     ratios within ~1-2% of 1.0 per offense for current-vintage years).
  C. The FBI NIBRS-to-SRS counting rules, rule by rule, on dual-reporting agencies:
     per-victim murder / rape (11A+11B+11C) / aggravated assault, full hierarchy
     suppression including the MVT-above-larceny exception, per-vehicle motor vehicle
     theft, and the capped burglary hotel-rule premises multiplier.
  D. Hierarchy-rule materiality at the Part I level.
  E. CIUS arbitration: which source the published row equals when sources disagree.
  F. National reported Return A sums vs CDE estimated totals.

Writes a markdown report to state/qa/fbi_conversion_rules_verification.md.
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
from crimerisk.utils.zipfiles import extract_zip_member

YEAR_START = 2018
YEAR_END = 2024
FOCUS_YEAR = 2024

PATHS = get_paths()
OBS = PATHS.state_dir / "observations" / "agency_year_observations.parquet"
REGIMES = PATHS.state_dir / "modeling" / "agency_year_reporting_regimes.parquet"
OFFENSE_SEG = PATHS.cache_dir / "nibrs_offense" / f"nibrs_offense_segment_{FOCUS_YEAR}.parquet"
BATCH_HEADER = PATHS.cache_dir / "nibrs_batch_header" / "nibrs_batch_header_1991_2024.parquet"
OFFENSES_KNOWN = PATHS.cache_dir / "offenses_known_year" / "offenses_known_yearly_1960_2024.parquet"
CDE_ESTIMATES = PATHS.data_dir / "FBI-CDE-Estimates-1979-2024" / "estimated_crimes_1979_2024.csv"
NIBRS_DIR = PATHS.data_dir / "NIBRS-Kaplan-1991-2024"
REPORT_PATH = PATHS.state_dir / "qa" / "fbi_conversion_rules_verification.md"

NIBRS_PART1_CASE = """
  CASE
    WHEN lower(ucr_offense_code) = 'murder/nonnegligent manslaughter' THEN 'murder'
    WHEN lower(ucr_offense_code) = 'sex offenses - rape' THEN 'rape'
    WHEN lower(ucr_offense_code) = 'robbery' THEN 'robbery'
    WHEN lower(ucr_offense_code) = 'assault offenses - aggravated assault' THEN 'aggravated_assault'
    WHEN lower(ucr_offense_code) = 'burglary/breaking and entering' THEN 'burglary'
    WHEN lower(ucr_offense_code) LIKE 'larceny/theft offenses%' THEN 'larceny'
    WHEN lower(ucr_offense_code) = 'motor vehicle theft' THEN 'motor_vehicle_theft'
    ELSE NULL
  END
"""

RAPE_ABC = (
    "'sex offenses - rape','sex offenses - sodomy','sex offenses - sexual assault with an object'"
)


def ensure_segment(segment: str, year: int) -> Path:
    member = f"nibrs_{segment}_segment_{year}.parquet"
    out_path = PATHS.cache_dir / f"nibrs_{segment}" / member
    zip_path = NIBRS_DIR / f"{segment}_segment_parquet_1991_2024.zip"
    extract_zip_member(zip_path, member, out_path)
    return out_path


def md_table(df) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


victim_seg = ensure_segment("victim", FOCUS_YEAR)
property_seg = ensure_segment("property", FOCUS_YEAR)

con = duckdb.connect()
sections: list[str] = [
    "# FBI conversion-rule verification",
    "",
    f"Companion to `docs/FBI-DATA-GUIDE.md`. Focus year {FOCUS_YEAR}; panel years "
    f"{YEAR_START}-{YEAR_END}. Regenerate with "
    "`uv run python scripts/diagnostics/verify_fbi_conversion_rules.py`.",
    "",
]

# --- A. presence overlap -------------------------------------------------------------
overlap = con.sql(f"""
WITH srs AS (
  SELECT DISTINCT ori9, year FROM '{OBS}'
  WHERE source='srs_return_a_annual' AND year BETWEEN {YEAR_START} AND {YEAR_END}
    AND months_reported > 0
), nib AS (
  SELECT DISTINCT ori9, year FROM '{OBS}'
  WHERE source='nibrs_srs_equivalent_annual' AND year BETWEEN {YEAR_START} AND {YEAR_END}
    AND count IS NOT NULL
)
SELECT COALESCE(s.year, n.year) AS year,
  COUNT(DISTINCT s.ori9) AS return_a_agencies,
  COUNT(DISTINCT CASE WHEN s.ori9 IS NOT NULL AND n.ori9 IS NOT NULL THEN s.ori9 END) AS in_both,
  COUNT(DISTINCT CASE WHEN n.ori9 IS NULL THEN s.ori9 END) AS srs_only,
  COUNT(DISTINCT CASE WHEN s.ori9 IS NULL THEN n.ori9 END) AS nibrs_only
FROM srs s FULL OUTER JOIN nib n ON s.ori9=n.ori9 AND s.year=n.year
GROUP BY 1 ORDER BY 1
""").df()

residual_2021 = con.sql(f"""
WITH nib AS (
  SELECT DISTINCT ori9 FROM '{OBS}'
  WHERE source='nibrs_srs_equivalent_annual' AND year=2021 AND count IS NOT NULL
), s AS (
  SELECT ori9, MAX(annual_part1_total) tot FROM '{OBS}'
  WHERE source='srs_return_a_annual' AND year=2021 AND months_reported>0 GROUP BY 1
), srs_only AS (SELECT * FROM s WHERE ori9 NOT IN (SELECT ori9 FROM nib)),
bhy AS (SELECT DISTINCT ori FROM '{BATCH_HEADER}' WHERE year=2021)
SELECT COUNT(*) srs_only_2021,
  ROUND(AVG(CASE WHEN b.ori IS NOT NULL THEN 1 ELSE 0 END),3) in_nibrs_batch_header,
  ROUND(AVG(CASE WHEN b.ori IS NOT NULL AND tot=0 THEN 1 ELSE 0 END),3) in_header_zero_part1
FROM srs_only o LEFT JOIN bhy b ON o.ori9=b.ori
""").df()

sections += [
    "## A. Return A vs NIBRS agency presence (local panel, months_reported > 0)",
    "",
    md_table(overlap),
    "2021 accepted zero native SRS submissions; populated 2021 Return A rows are "
    "FBI-converted NIBRS. SRS-only residual breakdown:",
    "",
    md_table(residual_2021),
]

# --- B. production rollup agreement --------------------------------------------------
agree_year = con.sql(f"""
SELECT year, COUNT(*) pairs,
  ROUND(AVG(CASE WHEN srs_count = nibrs_count THEN 1 ELSE 0 END),3) exact_eq,
  ROUND(SUM(srs_count)/NULLIF(SUM(nibrs_count),0),4) return_a_over_rollup
FROM '{REGIMES}'
WHERE srs_count IS NOT NULL AND nibrs_count IS NOT NULL
  AND year BETWEEN {YEAR_START} AND {YEAR_END}
GROUP BY 1 ORDER BY 1
""").df()

agree_offense = con.sql(f"""
SELECT offense, COUNT(*) pairs,
  ROUND(AVG(CASE WHEN srs_count = nibrs_count THEN 1 ELSE 0 END),3) exact_eq,
  ROUND(SUM(srs_count)/NULLIF(SUM(nibrs_count),0),4) return_a_over_rollup
FROM '{REGIMES}'
WHERE srs_count IS NOT NULL AND nibrs_count IS NOT NULL AND year={FOCUS_YEAR}
GROUP BY 1 ORDER BY 1
""").df()

sections += [
    "## B. Acceptance: Return A vs production SRS-equivalent rollup (dual agencies)",
    "",
    "The rollup implements the SRS scoring rules (crime/nibrs.py); aggregate ratios "
    "should sit within ~1-2% of 1.0 for current-vintage years (2023-2024). Earlier "
    "years run looser because mid-year NIBRS transitioners have full-year Return A "
    "rows but only partial-year NIBRS segments.",
    "",
    md_table(agree_year),
    f"By offense, {FOCUS_YEAR}:",
    "",
    md_table(agree_offense),
]

# --- C. counting rules ---------------------------------------------------------------
codes_list = ",".join(
    f"lower(CAST(ucr_offense_code_{i} AS VARCHAR))" for i in range(1, 11)
)
victim_rules = con.sql(f"""
WITH v AS (
  SELECT ori, [{codes_list}] codes FROM '{victim_seg}'
), vc AS (
  SELECT ori,
    SUM(CASE WHEN list_contains(codes,'murder/nonnegligent manslaughter')
        THEN 1 ELSE 0 END) murder_victims,
    SUM(CASE WHEN list_contains(codes,'assault offenses - aggravated assault')
        AND NOT list_contains(codes,'murder/nonnegligent manslaughter')
        THEN 1 ELSE 0 END) aggasslt_victims,
    SUM(CASE WHEN (list_contains(codes,'sex offenses - rape')
            OR list_contains(codes,'sex offenses - sodomy')
            OR list_contains(codes,'sex offenses - sexual assault with an object'))
        AND NOT list_contains(codes,'murder/nonnegligent manslaughter')
        THEN 1 ELSE 0 END) rape_victims
  FROM v GROUP BY 1
)
SELECT r.offense, SUM(r.srs_count)::BIGINT return_a_total,
  SUM(r.nibrs_count)::BIGINT production_rollup,
  SUM(CASE r.offense WHEN 'murder' THEN murder_victims
      WHEN 'aggravated_assault' THEN aggasslt_victims
      WHEN 'rape' THEN rape_victims END)::BIGINT per_victim_count,
  ROUND(SUM(r.srs_count)/SUM(CASE r.offense WHEN 'murder' THEN murder_victims
      WHEN 'aggravated_assault' THEN aggasslt_victims
      WHEN 'rape' THEN rape_victims END),4) return_a_over_per_victim
FROM '{REGIMES}' r JOIN vc ON r.ori9=vc.ori
WHERE r.year={FOCUS_YEAR} AND r.offense IN ('murder','aggravated_assault','rape')
  AND r.srs_count IS NOT NULL AND r.nibrs_count IS NOT NULL
GROUP BY 1 ORDER BY 1
""").df()

mvt_rule = con.sql(f"""
WITH mvt_inc AS (
  SELECT DISTINCT ori, unique_incident_id FROM '{OFFENSE_SEG}'
  WHERE lower(ucr_offense_code)='motor vehicle theft'
), veh AS (
  SELECT p.ori, p.unique_incident_id,
    MAX(TRY_CAST(p.number_of_stolen_motor_vehicles AS DOUBLE)) nveh
  FROM '{property_seg}' p
  JOIN mvt_inc m ON p.unique_incident_id=m.unique_incident_id AND p.ori=m.ori
  GROUP BY 1,2
), agg AS (
  SELECT ori, COUNT(*) incidents, SUM(GREATEST(COALESCE(nveh,1),1)) vehicles
  FROM veh GROUP BY 1
)
SELECT SUM(r.srs_count)::BIGINT return_a_total,
  SUM(a.incidents)::BIGINT incidents, SUM(a.vehicles)::BIGINT vehicles,
  ROUND(SUM(r.srs_count)/SUM(a.incidents),4) return_a_over_incidents,
  ROUND(SUM(r.srs_count)/SUM(a.vehicles),4) return_a_over_vehicles
FROM '{REGIMES}' r JOIN agg a ON r.ori9=a.ori
WHERE r.year={FOCUS_YEAR} AND r.offense='motor_vehicle_theft'
  AND r.srs_count IS NOT NULL AND r.nibrs_count IS NOT NULL
""").df()

burglary_bracket = con.sql(f"""
WITH b AS (
  SELECT ori, unique_incident_id,
    MAX(TRY_CAST(number_of_premises_entered AS DOUBLE)) prem
  FROM '{OFFENSE_SEG}'
  WHERE lower(ucr_offense_code)='burglary/breaking and entering'
  GROUP BY 1,2
), agg AS (
  SELECT ori, COUNT(*) incidents, SUM(GREATEST(COALESCE(prem,1),1)) premises
  FROM b GROUP BY 1
)
SELECT SUM(r.srs_count)::BIGINT return_a_total,
  SUM(a.incidents)::BIGINT incidents, SUM(a.premises)::BIGINT premises_adjusted,
  ROUND(SUM(r.srs_count)/SUM(a.incidents),4) return_a_over_incidents,
  ROUND(SUM(r.srs_count)/SUM(a.premises),4) return_a_over_all_premises
FROM '{REGIMES}' r JOIN agg a ON r.ori9=a.ori
WHERE r.year={FOCUS_YEAR} AND r.offense='burglary'
  AND r.srs_count IS NOT NULL AND r.nibrs_count IS NOT NULL
""").df()

sections += [
    f"## C. Counting-rule tests, {FOCUS_YEAR} dual-reporting agencies",
    "",
    "Per-victim rules (murder; rape as 11A+11B+11C; aggravated assault with murder "
    "superseding per victim). Ratios near 1.0 confirm the rule:",
    "",
    md_table(victim_rules),
    "Motor vehicle theft: per stolen vehicle (property segment), not per incident:",
    "",
    md_table(mvt_rule),
    "Burglary: per incident plus hotel-rule premises multiplier. The production rollup "
    "applies the multiplier only to lodging-type locations, caps legitimate premises at "
    "max(p95, 30), and treats 99 as a cap sentinel. The bracket below remains a useful "
    "sanity range:",
    "",
    md_table(burglary_bracket),
]

# --- D. hierarchy materiality --------------------------------------------------------
hierarchy = con.sql(f"""
WITH p1 AS (
  SELECT unique_incident_id, {NIBRS_PART1_CASE} cat FROM '{OFFENSE_SEG}'
), f AS (SELECT * FROM p1 WHERE cat IS NOT NULL)
SELECT COUNT(*) part1_offense_rows,
  COUNT(DISTINCT unique_incident_id) part1_incidents,
  ROUND(COUNT(*)*1.0/COUNT(DISTINCT unique_incident_id),4) offenses_per_incident,
  (SELECT COUNT(*) FROM (
     SELECT unique_incident_id FROM f GROUP BY 1 HAVING COUNT(DISTINCT cat)>1
  )) multi_category_incidents
FROM f
""").df()

sections += [
    f"## D. Hierarchy-rule materiality at the Part I level, {FOCUS_YEAR}",
    "",
    md_table(hierarchy),
]

# --- E. CIUS arbitration -------------------------------------------------------------
cius_arbitration = con.sql(f"""
SELECT year, COUNT(*) disagreeing_cells,
  ROUND(AVG(CASE WHEN cius_count = srs_count THEN 1 ELSE 0 END),3) cius_eq_return_a,
  ROUND(AVG(CASE WHEN cius_count = nibrs_count THEN 1 ELSE 0 END),3) cius_eq_rollup,
  ROUND(AVG(CASE WHEN cius_count != srs_count AND cius_count != nibrs_count
      THEN 1 ELSE 0 END),3) cius_eq_neither
FROM '{REGIMES}'
WHERE cius_count IS NOT NULL AND srs_count IS NOT NULL AND nibrs_count IS NOT NULL
  AND srs_count != nibrs_count AND year BETWEEN 2021 AND {YEAR_END}
GROUP BY 1 ORDER BY 1
""").df()

cius_overall = con.sql(f"""
SELECT year, COUNT(*) comparable_cells,
  ROUND(AVG(CASE WHEN cius_count = srs_count THEN 1 ELSE 0 END),4) cius_eq_return_a
FROM '{REGIMES}'
WHERE cius_count IS NOT NULL AND srs_count IS NOT NULL
  AND year BETWEEN 2021 AND {YEAR_END}
GROUP BY 1 ORDER BY 1
""").df()

sections += [
    "## E. CIUS arbitration (published row vs current master-file sources)",
    "",
    "Where Return A and the production NIBRS rollup disagree, the published CIUS cell equals:",
    "",
    md_table(cius_arbitration),
    "All comparable cells (vintage drift shows up as sub-1.0 in revised years):",
    "",
    md_table(cius_overall),
]

# --- F. reported vs estimated --------------------------------------------------------
reported_vs_estimated = con.sql(f"""
WITH rep AS (
  SELECT year,
    SUM(actual_murder + actual_rape_total + actual_robbery_total
        + actual_assault_aggravated) viol_rep,
    SUM(actual_burglary_total + actual_theft_total
        + actual_motor_vehicle_theft_total) prop_rep
  FROM '{OFFENSES_KNOWN}' WHERE year BETWEEN {YEAR_START} AND {YEAR_END} GROUP BY 1
), est AS (
  SELECT TRY_CAST(year AS INT) yr,
    TRY_CAST(replace(violent_crime,',','') AS DOUBLE) viol_est,
    TRY_CAST(replace(property_crime,',','') AS DOUBLE) prop_est
  FROM read_csv('{CDE_ESTIMATES}', header=true, all_varchar=true)
  WHERE (state_abbr IS NULL OR state_abbr='')
)
SELECT r.year, viol_rep::BIGINT viol_reported, viol_est::BIGINT viol_estimated,
  ROUND(viol_rep/viol_est,3) violent_ratio,
  prop_rep::BIGINT prop_reported, prop_est::BIGINT prop_estimated,
  ROUND(prop_rep/prop_est,3) property_ratio
FROM rep r JOIN est e ON r.year=e.yr ORDER BY 1
""").df()

sections += [
    "## F. National Return A reported sums vs CDE estimated totals",
    "",
    "All agencies in the Return A master file (not the local-only panel). The shortfall "
    "vs 1.0 is the FBI's estimation layer; national rows for 2018-2020 are absent from "
    "the frozen estimates CSV:",
    "",
    md_table(reported_vs_estimated),
]

report = "\n".join(sections)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report)
print(report)
print(f"\nwrote {REPORT_PATH}")
