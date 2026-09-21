from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.crime.nibrs import ensure_nibrs_batch_header_parquet, ensure_nibrs_offense_year_parquet
from crimerisk.crime.srs import ensure_srs_month_year_parquet, ensure_srs_year_parquet
from crimerisk.paths import RepoPaths


@dataclass(frozen=True)
class ExplorationConfig:
    year_start: int = 2018
    year_end: int = 2024
    agency_types: tuple[str, ...] = ("local police department", "constable/marshal")
    leaic_agency_type_codes: tuple[str, ...] = ("0", "7")  # local police department, constable/marshal
    deep_dive_years: tuple[int, ...] = (2021, 2024)
    top_partial_agencies: int = 50
    quarterly_pattern_part1_min_total: float = 500.0
    quarterly_pattern_max_nonzero_months: int = 4
    quarterly_pattern_min_max_month_share: float = 0.4


def _months_bucket_sql(col: str) -> str:
    return f"""
      CASE
        WHEN {col} = 0 THEN '0'
        WHEN {col} BETWEEN 1 AND 9 THEN '1-9'
        WHEN {col} = 10 THEN '10'
        WHEN {col} = 11 THEN '11'
        WHEN {col} = 12 THEN '12'
        ELSE 'other'
      END
    """.strip()


def _srs_true_months_reported_sql() -> str:
    return """
      CASE
        WHEN number_of_months_missing IS NULL THEN NULL
        ELSE GREATEST(0, LEAST(12, 12 - CAST(number_of_months_missing AS INTEGER)))
      END
    """.strip()


def _md_table(df: pd.DataFrame, *, max_rows: int = 80) -> str:
    if df.empty:
        return "_(no rows)_\n"
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns.tolist()]

    def esc(v: object) -> str:
        s = "" if v is None else str(v)
        s = s.replace("\n", " ").replace("|", "\\|")
        return s

    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append("")
        lines.append("_(truncated)_")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _ensure_leaic_local_oris(
    con: duckdb.DuckDBPyConnection, *, leaic_tsv_path: Path, agency_type_codes: Iterable[str]
) -> pd.DataFrame:
    codes_sql = ", ".join([f"'{str(c)}'" for c in agency_type_codes])
    q = f"""
    SELECT DISTINCT
      ORI9 AS ori
    FROM read_csv_auto('{leaic_tsv_path.as_posix()}', delim='\t', all_varchar=true)
    WHERE AGCYTYPE IN ({codes_sql})
      AND ORI9 IS NOT NULL
    """
    df = con.sql(q).df()
    df["ori"] = df["ori"].astype(str)
    return df


def _ensure_distinct_oris(
    con: duckdb.DuckDBPyConnection, parquet_path: Path, *, local_oris_view: str | None = None
) -> pd.DataFrame:
    if local_oris_view is None:
        return con.sql(f"SELECT DISTINCT ori AS ori FROM read_parquet('{parquet_path.as_posix()}')").df()
    return con.sql(
        f"""
        SELECT DISTINCT n.ori AS ori
        FROM read_parquet('{parquet_path.as_posix()}') n
        JOIN {local_oris_view} l ON n.ori = l.ori
        """
    ).df()


def _ensure_distinct_oris_part1(
    con: duckdb.DuckDBPyConnection, parquet_path: Path, *, local_oris_view: str | None = None
) -> pd.DataFrame:
    # Kaplan NIBRS parquet uses textual ucr_offense_code strings (not legacy numeric codes).
    part1_where = """
      lower(ucr_offense_code) IN (
        'murder/nonnegligent manslaughter',
        'sex offenses - rape',
        'robbery',
        'assault offenses - aggravated assault',
        'burglary/breaking and entering',
        'motor vehicle theft'
      )
      OR lower(ucr_offense_code) LIKE 'larceny/theft offenses%'
    """.strip()
    if local_oris_view is None:
        return con.sql(
            f"SELECT DISTINCT ori AS ori FROM read_parquet('{parquet_path.as_posix()}') WHERE {part1_where}"
        ).df()
    return con.sql(
        f"""
        SELECT DISTINCT n.ori AS ori
        FROM read_parquet('{parquet_path.as_posix()}') n
        JOIN {local_oris_view} l ON n.ori = l.ori
        WHERE {part1_where}
        """
    ).df()


def build_srs_nibrs_exploration(
    *,
    paths: RepoPaths,
    out_dir: Path,
    cfg: ExplorationConfig = ExplorationConfig(),
    canonical_parquet: Path | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # Inputs
    srs_zip = paths.data_dir / "SRS-Kaplan-1960-2024" / "offenses_known_parquet_1960_2024_year.zip"
    nibrs_zip = paths.data_dir / "NIBRS-Kaplan-1991-2024" / "offense_segment_parquet_1991_2024.zip"
    nibrs_batch_header_zip = paths.data_dir / "NIBRS-Kaplan-1991-2024" / "batch_header_parquet_1991_2024.zip"
    leaic_tsv = paths.data_dir / "LEAIC-Crosswalk-ICPSR_35158" / "DS0001" / "35158-0001-Data.tsv"
    srs_parquet = ensure_srs_year_parquet(zip_path=srs_zip, cache_dir=paths.cache_dir).parquet_path
    batch_header_parquet = ensure_nibrs_batch_header_parquet(
        zip_path=nibrs_batch_header_zip,
        cache_dir=paths.cache_dir,
    ).parquet_path

    leaic_local_oris = _ensure_leaic_local_oris(
        con, leaic_tsv_path=leaic_tsv, agency_type_codes=cfg.leaic_agency_type_codes
    )
    con.register("leaic_local_oris", leaic_local_oris)

    agency_types_sql = ", ".join([f"'{t}'" for t in cfg.agency_types])
    offense_sum_sql = ", ".join([f"SUM(CAST({c} AS DOUBLE)) AS {o}" for o, c in _srs_offense_cols().items()])
    total_part1_sql = " + ".join([f"CAST({c} AS DOUBLE)" for c in _srs_offense_cols().values()])

    # 1) SRS months bucket summary by year
    srs_rows: list[pd.DataFrame] = []
    for year in range(cfg.year_start, cfg.year_end + 1):
        months_col = _srs_true_months_reported_sql()
        bucket = _months_bucket_sql(months_col)
        q = f"""
        SELECT
          {year} AS year,
          {bucket} AS months_bucket,
          COUNT(*) AS agencies,
          SUM(CAST(population AS DOUBLE)) AS population,
          {offense_sum_sql},
          SUM(({total_part1_sql})) AS part1_total
        FROM read_parquet('{srs_parquet.as_posix()}')
        WHERE CAST(year AS INTEGER) = {year}
          AND agency_type IN ({agency_types_sql})
        GROUP BY 1,2
        ORDER BY 1,2
        """
        srs_rows.append(con.sql(q).df())

    srs_summary = pd.concat(srs_rows, ignore_index=True)
    srs_summary_path = out_dir / "srs_months_bucket_summary.csv"
    srs_summary.to_csv(srs_summary_path, index=False)

    # 2) ORI overlap between SRS and NIBRS (offense segment), by year and SRS months bucket
    overlap_rows: list[pd.DataFrame] = []
    for year in range(cfg.year_start, cfg.year_end + 1):
        nibrs_parquet = ensure_nibrs_offense_year_parquet(zip_path=nibrs_zip, year=year, cache_dir=paths.cache_dir).parquet_path

        months_col = "CAST(number_of_months_reported AS INTEGER)"
        bucket = _months_bucket_sql(months_col)

        # Distinct ORIs in NIBRS (any offense) and Part I subset.
        nibrs_oris = _ensure_distinct_oris(con, nibrs_parquet, local_oris_view="leaic_local_oris")
        nibrs_oris_part1 = _ensure_distinct_oris_part1(con, nibrs_parquet, local_oris_view="leaic_local_oris")
        con.register("nibrs_oris", nibrs_oris)
        con.register("nibrs_oris_part1", nibrs_oris_part1)

        q = f"""
        WITH srs AS (
          SELECT
            (ori || '00') AS ori,
            {bucket} AS months_bucket
          FROM read_parquet('{srs_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
        )
        SELECT
          {year} AS year,
          months_bucket,
          COUNT(*) AS srs_agencies,
          SUM(CASE WHEN n.ori IS NOT NULL THEN 1 ELSE 0 END) AS srs_agencies_in_nibrs_any,
          SUM(CASE WHEN p.ori IS NOT NULL THEN 1 ELSE 0 END) AS srs_agencies_in_nibrs_part1,
          (SELECT COUNT(*) FROM nibrs_oris) AS nibrs_oris_any,
          (SELECT COUNT(*) FROM nibrs_oris_part1) AS nibrs_oris_part1
        FROM srs
        LEFT JOIN nibrs_oris n ON srs.ori = n.ori
        LEFT JOIN nibrs_oris_part1 p ON srs.ori = p.ori
        GROUP BY 1,2
        ORDER BY 1,2
        """
        overlap_rows.append(con.sql(q).df())
        con.unregister("nibrs_oris")
        con.unregister("nibrs_oris_part1")

    overlap = pd.concat(overlap_rows, ignore_index=True)
    overlap_path = out_dir / "srs_nibrs_ori_overlap_by_months_bucket.csv"
    overlap.to_csv(overlap_path, index=False)

    # 2a.1) NIBRS-local ORIs missing from SRS (diagnostic)
    nibrs_missing_in_srs_paths: dict[str, Path] = {}
    for year in cfg.deep_dive_years:
        if year < cfg.year_start or year > cfg.year_end:
            continue
        nibrs_parquet = ensure_nibrs_offense_year_parquet(zip_path=nibrs_zip, year=year, cache_dir=paths.cache_dir).parquet_path

        q = f"""
        WITH nibrs_local AS (
          SELECT DISTINCT n.ori AS fbi_ori9
          FROM read_parquet('{nibrs_parquet.as_posix()}') n
          JOIN leaic_local_oris l ON n.ori = l.ori
        ),
        srs_local AS (
          SELECT DISTINCT (ori || '00') AS fbi_ori9
          FROM read_parquet('{srs_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
        ),
        missing AS (
          SELECT n.fbi_ori9
          FROM nibrs_local n
          LEFT JOIN srs_local s USING (fbi_ori9)
          WHERE s.fbi_ori9 IS NULL
        ),
        leaic AS (
          SELECT
            ORI9 AS fbi_ori9,
            NAME AS agency_name,
            STATENAME AS state_name,
            COUNTYNAME AS county_name,
            LG_NAME AS local_gov_name,
            AGCYTYPE AS leaic_agcytype
          FROM read_csv_auto('{leaic_tsv.as_posix()}', delim='\\t', all_varchar=true)
        )
        SELECT
          m.fbi_ori9,
          l.agency_name,
          l.state_name,
          l.county_name,
          l.local_gov_name,
          l.leaic_agcytype
        FROM missing m
        LEFT JOIN leaic l USING (fbi_ori9)
        ORDER BY m.fbi_ori9
        """
        missing_df = con.sql(q).df()
        out_path = out_dir / f"nibrs_local_oris_missing_in_srs_{year}.csv"
        missing_df.to_csv(out_path, index=False)
        nibrs_missing_in_srs_paths[f"nibrs_missing_in_srs_{year}"] = out_path

    # 2b) Reporting-month comparison: canonical SRS true months vs NIBRS batch header number_of_months_reported
    reporting_rows: list[pd.DataFrame] = []
    for year in range(cfg.year_start, cfg.year_end + 1):
        months_col = _srs_true_months_reported_sql()
        bucket = _months_bucket_sql(months_col)
        q = f"""
        WITH srs AS (
          SELECT
            (ori || '00') AS fbi_ori9,
            {months_col} AS srs_months,
            {bucket} AS months_bucket
          FROM read_parquet('{srs_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
        ),
        bh AS (
          SELECT
            ori AS fbi_ori9,
            CAST(NULLIF(number_of_months_reported, '') AS INTEGER) AS nibrs_months
          FROM read_parquet('{batch_header_parquet.as_posix()}')
          WHERE year = {year}
        )
        SELECT
          {year} AS year,
          s.months_bucket,
          COUNT(*) AS srs_agencies,
          SUM(CASE WHEN b.fbi_ori9 IS NOT NULL THEN 1 ELSE 0 END) AS nibrs_batch_header_present,
          SUM(CASE WHEN b.nibrs_months IS NULL THEN 1 ELSE 0 END) AS nibrs_months_null,
          SUM(CASE WHEN b.nibrs_months = 0 THEN 1 ELSE 0 END) AS nibrs_months_0,
          SUM(CASE WHEN b.nibrs_months BETWEEN 1 AND 11 THEN 1 ELSE 0 END) AS nibrs_months_1_11,
          SUM(CASE WHEN b.nibrs_months = 12 THEN 1 ELSE 0 END) AS nibrs_months_12,
          SUM(CASE WHEN b.nibrs_months > 0 THEN 1 ELSE 0 END) AS nibrs_months_gt0,
          SUM(CASE WHEN b.nibrs_months > s.srs_months THEN 1 ELSE 0 END) AS nibrs_months_gt_srs
        FROM srs s
        LEFT JOIN bh b ON s.fbi_ori9 = b.fbi_ori9
        GROUP BY 1,2
        ORDER BY 1,2
        """
        reporting_rows.append(con.sql(q).df())

    reporting = pd.concat(reporting_rows, ignore_index=True)
    reporting_path = out_dir / "srs_vs_nibrs_reporting_months_by_bucket.csv"
    reporting.to_csv(reporting_path, index=False)

    # 2c) SRS monthly vs annual consistency (sum of 12 months vs annual totals).
    srs_monthly_zip = paths.data_dir / "SRS-Kaplan-1960-2024" / "offenses_known_parquet_1960_2024_month.zip"
    monthly_consistency_rows: list[pd.DataFrame] = []
    mismatch_rows: list[pd.DataFrame] = []
    tol = 0.5
    annual_cols = _srs_offense_cols()
    annual_select = ",\n          ".join([f"CAST({c} AS DOUBLE) AS {o}_annual" for o, c in annual_cols.items()])
    monthly_sum_select = ",\n          ".join([f"SUM(CAST({c} AS DOUBLE)) AS {o}_monthly_sum" for o, c in annual_cols.items()])
    any_mismatch_sql = " OR ".join([f"ABS({o}_monthly_sum - {o}_annual) > {tol}" for o in annual_cols])

    for year in range(cfg.year_start, cfg.year_end + 1):
        srs_month_parquet = ensure_srs_month_year_parquet(
            zip_path=srs_monthly_zip, year=year, cache_dir=paths.cache_dir
        ).parquet_path
        q = f"""
        WITH annual AS (
          SELECT
            (ori || '00') AS fbi_ori9,
            CAST(year AS INTEGER) AS year,
            {_srs_true_months_reported_sql()} AS months_reported,
            CAST(number_of_months_missing AS INTEGER) AS months_missing,
            CAST(population AS DOUBLE) AS population,
            {annual_select}
          FROM read_parquet('{srs_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
        ),
        monthly AS (
          SELECT
            (ori || '00') AS fbi_ori9,
            CAST(year AS INTEGER) AS year,
            {monthly_sum_select}
          FROM read_parquet('{srs_month_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
          GROUP BY 1,2
        ),
        joined AS (
          SELECT
            a.*,
            {", ".join([f"m.{o}_monthly_sum" for o in annual_cols])}
          FROM annual a
          LEFT JOIN monthly m USING (fbi_ori9, year)
        )
        SELECT
          {year} AS year,
          COUNT(*) AS agencies,
          SUM(CASE WHEN {any_mismatch_sql} THEN 1 ELSE 0 END) AS agencies_any_mismatch,
          {", ".join([f"SUM(CASE WHEN ABS({o}_monthly_sum - {o}_annual) > {tol} THEN 1 ELSE 0 END) AS {o}_mismatch" for o in annual_cols])}
        FROM joined
        """
        monthly_consistency_rows.append(con.sql(q).df())

        q_mismatch = f"""
        WITH annual AS (
          SELECT
            (ori || '00') AS fbi_ori9,
            CAST(year AS INTEGER) AS year,
            agency_name,
            state_abb,
            agency_type,
            {_srs_true_months_reported_sql()} AS months_reported,
            CAST(number_of_months_missing AS INTEGER) AS months_missing,
            CAST(population AS DOUBLE) AS population,
            {annual_select}
          FROM read_parquet('{srs_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
        ),
        monthly AS (
          SELECT
            (ori || '00') AS fbi_ori9,
            CAST(year AS INTEGER) AS year,
            {monthly_sum_select}
          FROM read_parquet('{srs_month_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
          GROUP BY 1,2
        )
        SELECT
          a.*,
          {", ".join([f"m.{o}_monthly_sum" for o in annual_cols])},
          {", ".join([f"(m.{o}_monthly_sum - a.{o}_annual) AS {o}_diff" for o in annual_cols])}
        FROM annual a
        JOIN monthly m USING (fbi_ori9, year)
        WHERE {any_mismatch_sql}
        ORDER BY population DESC
        LIMIT 500
        """
        mismatch_rows.append(con.sql(q_mismatch).df())

    monthly_consistency = pd.concat(monthly_consistency_rows, ignore_index=True)
    monthly_consistency_path = out_dir / "srs_monthly_vs_annual_consistency_summary.csv"
    monthly_consistency.to_csv(monthly_consistency_path, index=False)

    mismatches = pd.concat(mismatch_rows, ignore_index=True) if mismatch_rows else pd.DataFrame()
    mismatch_path = out_dir / "srs_monthly_vs_annual_mismatches.csv"
    mismatches.to_csv(mismatch_path, index=False)

    # 2d) Quarterly-pattern candidates in SRS monthly data (QA-only).
    quarterly_paths: dict[str, Path] = {}
    monthly_total_expr = " + ".join([f"CAST({c} AS DOUBLE)" for c in annual_cols.values()])
    annual_total_expr = " + ".join([f"CAST({c} AS DOUBLE)" for c in annual_cols.values()])
    for year in cfg.deep_dive_years:
        if year < cfg.year_start or year > cfg.year_end:
            continue
        srs_month_parquet = ensure_srs_month_year_parquet(
            zip_path=srs_monthly_zip, year=year, cache_dir=paths.cache_dir
        ).parquet_path
        q = f"""
        WITH annual AS (
          SELECT
            (ori || '00') AS fbi_ori9,
            CAST(year AS INTEGER) AS year,
            agency_name,
            state_abb,
            agency_type,
            {_srs_true_months_reported_sql()} AS months_reported,
            CAST(population AS DOUBLE) AS population,
            ({annual_total_expr}) AS part1_total_annual
          FROM read_parquet('{srs_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
        ),
        monthly AS (
          SELECT
            (ori || '00') AS fbi_ori9,
            CAST(year AS INTEGER) AS year,
            SUM(CASE WHEN ({monthly_total_expr}) > 0 THEN 1 ELSE 0 END) AS months_nonzero_part1,
            MAX(({monthly_total_expr})) AS max_month_part1,
            SUM(({monthly_total_expr})) AS part1_total_monthly_sum
          FROM read_parquet('{srs_month_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
          GROUP BY 1,2
        )
        SELECT
          a.fbi_ori9,
          a.agency_name,
          a.state_abb,
          a.agency_type,
          a.months_reported,
          a.population,
          a.part1_total_annual,
          m.months_nonzero_part1,
          m.max_month_part1,
          m.part1_total_monthly_sum,
          CASE WHEN a.part1_total_annual > 0 THEN (m.max_month_part1 / a.part1_total_annual) ELSE NULL END AS max_month_share
        FROM annual a
        LEFT JOIN monthly m USING (fbi_ori9, year)
        WHERE a.months_reported = 12
          AND a.part1_total_annual >= {float(cfg.quarterly_pattern_part1_min_total)}
          AND COALESCE(m.months_nonzero_part1, 0) <= {int(cfg.quarterly_pattern_max_nonzero_months)}
          AND COALESCE(CASE WHEN a.part1_total_annual > 0 THEN (m.max_month_part1 / a.part1_total_annual) ELSE NULL END, 0)
              >= {float(cfg.quarterly_pattern_min_max_month_share)}
        ORDER BY a.population DESC
        LIMIT 500
        """
        candidates = con.sql(q).df()
        out_path = out_dir / f"srs_quarterly_pattern_candidates_{year}.csv"
        candidates.to_csv(out_path, index=False)
        quarterly_paths[f"quarterly_candidates_{year}"] = out_path

    # 3) Deep dives for partial reporters (1–11 months) in selected years
    deep_paths: dict[str, Path] = {}
    for year in cfg.deep_dive_years:
        if year < cfg.year_start or year > cfg.year_end:
            continue

        nibrs_parquet = ensure_nibrs_offense_year_parquet(zip_path=nibrs_zip, year=year, cache_dir=paths.cache_dir).parquet_path
        bh_year = con.sql(
            f"""
            SELECT
              ori AS fbi_ori9,
              CAST(NULLIF(number_of_months_reported, '') AS INTEGER) AS nibrs_months_reported
            FROM read_parquet('{batch_header_parquet.as_posix()}')
            WHERE year = {year}
            """
        ).df()
        con.register("bh_year", bh_year)

        q_partial = f"""
        WITH srs AS (
          SELECT
            (ori || '00') AS fbi_ori9,
            ori9 AS kaplan_ori9,
            agency_name,
            state_abb,
            {_srs_true_months_reported_sql()} AS months_reported,
            CAST(population AS DOUBLE) AS population,
            {", ".join([f"CAST({c} AS DOUBLE) AS {o}" for o, c in _srs_offense_cols().items()])},
            ({total_part1_sql}) AS part1_total
          FROM read_parquet('{srs_parquet.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND agency_type IN ({agency_types_sql})
            AND ori IS NOT NULL
        )
        SELECT
          srs.*,
          b.nibrs_months_reported,
          CASE WHEN b.nibrs_months_reported IS NOT NULL AND b.nibrs_months_reported > 0 THEN 1 ELSE 0 END AS in_nibrs_reported
        FROM srs
        LEFT JOIN bh_year b ON srs.fbi_ori9 = b.fbi_ori9
        WHERE months_reported BETWEEN 1 AND 11
        ORDER BY population DESC
        LIMIT {int(cfg.top_partial_agencies)}
        """
        top_partial = con.sql(q_partial).df()

        # NIBRS offense rows for those top partial ORIs that have any NIBRS months reported.
        top_oris = top_partial.loc[top_partial["in_nibrs_reported"] == 1, "fbi_ori9"].dropna().astype(str).tolist()
        coverage = pd.DataFrame(columns=["fbi_ori9", "nibrs_offense_rows"])
        if top_oris:
            con.register("top_oris", pd.DataFrame({"fbi_ori9": top_oris}))
            q_cov = f"""
            SELECT
              t.fbi_ori9 AS fbi_ori9,
              COUNT(*) AS nibrs_offense_rows
            FROM read_parquet('{nibrs_parquet.as_posix()}') n
            JOIN top_oris t ON n.ori = t.fbi_ori9
            GROUP BY 1
            ORDER BY 1
            """
            coverage = con.sql(q_cov).df()
            con.unregister("top_oris")

        top_partial = top_partial.merge(coverage, on="fbi_ori9", how="left")
        out_path = out_dir / f"top_partial_reporters_{year}.csv"
        top_partial.to_csv(out_path, index=False)
        deep_paths[f"top_partial_{year}"] = out_path
        con.unregister("bh_year")

    # 4) SRS vs NIBRS-derived Part I comparisons using canonical parquet (if available)
    compare_paths: dict[str, Path] = {}
    compare_summary = None
    if canonical_parquet is not None and canonical_parquet.exists():
        canon = pd.read_parquet(canonical_parquet)
        canon["year"] = pd.to_numeric(canon["year"], errors="coerce").astype("Int64")
        canon["months_reported"] = pd.to_numeric(canon["months_reported"], errors="coerce").fillna(0).astype(int)
        canon = canon[(canon["year"] >= cfg.year_start) & (canon["year"] <= cfg.year_end)].copy()

        def bucket(m: int) -> str:
            if m == 0:
                return "0"
            if 1 <= m <= 9:
                return "1-9"
            if m == 10:
                return "10"
            if m == 11:
                return "11"
            if m == 12:
                return "12"
            return "other"

        canon["months_bucket"] = canon["months_reported"].map(bucket)
        for o in OFFENSES_7:
            canon[f"{o}_srs"] = pd.to_numeric(canon[f"{o}_srs"], errors="coerce")
            canon[f"{o}_nibrs"] = pd.to_numeric(canon[f"{o}_nibrs"], errors="coerce")

        rows: list[dict[str, Any]] = []
        for (year, months_bucket), grp in canon.groupby(["year", "months_bucket"], dropna=True):
            # Only compare where NIBRS-derived exists (i.e., ORI present in our Part I aggregation).
            has_nibrs = grp[[f"{o}_nibrs" for o in OFFENSES_7]].notna().any(axis=1)
            g = grp[has_nibrs].copy()
            row: dict[str, Any] = {"year": int(year), "months_bucket": str(months_bucket), "agencies_both": int(len(g))}
            for o in OFFENSES_7:
                srs_sum = float(pd.to_numeric(g[f"{o}_srs"], errors="coerce").fillna(0.0).sum())
                nibrs_sum = float(pd.to_numeric(g[f"{o}_nibrs"], errors="coerce").fillna(0.0).sum())
                row[f"srs_sum_{o}"] = srs_sum
                row[f"nibrs_sum_{o}"] = nibrs_sum
                row[f"ratio_nibrs_over_srs_{o}"] = (nibrs_sum / srs_sum) if srs_sum > 0 else np.nan
            rows.append(row)

        compare_summary = pd.DataFrame(rows).sort_values(["year", "months_bucket"]).reset_index(drop=True)
        out_path = out_dir / "srs_vs_nibrs_part1_comparison_by_months_bucket.csv"
        compare_summary.to_csv(out_path, index=False)
        compare_paths["srs_vs_nibrs_by_bucket"] = out_path

    # 5) Markdown narrative report
    report_lines: list[str] = []
    report_lines.append("# SRS vs NIBRS exploratory report (Kaplan 2018–2024)")
    report_lines.append("")
    report_lines.append("This report is intended for statistical review before we lock a single fixed canonicalization policy.")
    report_lines.append("")
    report_lines.append("## Questions this report helps answer")
    report_lines.append("- How common are partial-year SRS reporters (true months_reported 1–11) by year, and how much do they contribute to totals?")
    report_lines.append("- When SRS is partial, is NIBRS present for the same ORI-year (i.e., is substitution even possible)?")
    report_lines.append("- When both exist, how do NIBRS-derived Part I counts (SRS hierarchy conversion) compare to Kaplan’s SRS-style counts?")
    report_lines.append("")
    report_lines.append("## Key takeaways (preliminary)")
    report_lines.append(
        "- Raw Kaplan SRS annual `number_of_months_reported` is not used here as a true month count; this report uses canonical SRS months reported derived from `number_of_months_missing`."
    )
    report_lines.append("- ORI overlap between SRS and NIBRS is substantial, but not universal; some SRS agencies are not present in NIBRS (SRS-only reporters).")
    report_lines.append(
        f"- In overlap stats below, NIBRS ORIs are restricted to LEAIC `AGCYTYPE ∈ {list(cfg.leaic_agency_type_codes)}` to match the municipal local-PD/constable universe."
    )
    report_lines.append(
        "- NIBRS batch header `number_of_months_reported` is compared against canonical SRS true months reported, not against the misleading raw SRS annual `number_of_months_reported` field."
    )
    report_lines.append("- `months_reported = 0` SRS agencies can still carry non-zero population while having all-zero Part I counts; this is not evidence that the raw annual last-month code was a true completeness count.")
    report_lines.append(
        "- SRS annual totals match the sum of the 12 monthly rows exactly for the same agency-year (within tolerance) in 2018–2024 for our local-PD/constable universe."
    )
    report_lines.append("")
    report_lines.append("## SRS months_reported bucket summary")
    report_lines.append(f"- Output CSV: `{srs_summary_path}`")
    report_lines.append("")
    report_lines.append(_md_table(srs_summary))
    report_lines.append("")
    report_lines.append("## ORI overlap: SRS vs NIBRS offense segment")
    report_lines.append(f"- Output CSV: `{overlap_path}`")
    report_lines.append("")
    report_lines.append(_md_table(overlap))
    report_lines.append("")
    for year in cfg.deep_dive_years:
        key = f"nibrs_missing_in_srs_{year}"
        if key not in nibrs_missing_in_srs_paths:
            continue
        report_lines.append(f"### NIBRS-local ORIs missing from SRS ({year})")
        report_lines.append(f"- Output CSV: `{nibrs_missing_in_srs_paths[key]}`")
        report_lines.append("")
        report_lines.append(_md_table(pd.read_csv(nibrs_missing_in_srs_paths[key]).head(40)))
        report_lines.append("")
    report_lines.append("## Reporting months: SRS vs NIBRS batch header")
    report_lines.append(f"- Output CSV: `{reporting_path}`")
    report_lines.append("")
    report_lines.append(
        "NIBRS reporting months here come from Kaplan’s NIBRS `batch_header` file (`number_of_months_reported`). SRS reporting months here use the canonical true month count derived from `number_of_months_missing`, not the misleading raw SRS annual `number_of_months_reported` field."
    )
    report_lines.append("")
    report_lines.append(_md_table(reporting))
    report_lines.append("")
    report_lines.append("## SRS annual vs sum(monthly) consistency (QA-only)")
    report_lines.append(f"- Output CSV: `{monthly_consistency_path}`")
    report_lines.append(f"- Output CSV: `{mismatch_path}`")
    report_lines.append("")
    report_lines.append(
        "This compares annual Return A totals to the sum of all 12 monthly records for the same agency-year. It is intended as a backstop against misinterpreting monthly zeros as missingness (e.g., quarterly dumps)."
    )
    report_lines.append("")
    report_lines.append(_md_table(monthly_consistency))
    report_lines.append("")
    if not mismatches.empty:
        report_lines.append("### Sample mismatches (up to 25 rows)")
        report_lines.append(_md_table(mismatches.head(25)))
        report_lines.append("")

    for year in cfg.deep_dive_years:
        key = f"quarterly_candidates_{year}"
        if key not in quarterly_paths:
            continue
        report_lines.append(f"## QA: quarterly-pattern candidates in SRS monthly ({year})")
        report_lines.append(f"- Output CSV: `{quarterly_paths[key]}`")
        report_lines.append("")
        report_lines.append(
            "These are agencies with canonical `months_reported = 12` but very few months with non-zero Part I totals in the monthly file. "
            "This is a cadence/structure heuristic (e.g., possible quarterly dumps), not proof of missingness."
        )
        report_lines.append("")
        report_lines.append(_md_table(pd.read_csv(quarterly_paths[key]).head(25)))
        report_lines.append("")

    for year in cfg.deep_dive_years:
        key = f"top_partial_{year}"
        if key not in deep_paths:
            continue
        report_lines.append(f"## Deep dive: top partial reporters by population ({year})")
        report_lines.append(f"- Output CSV: `{deep_paths[key]}`")
        report_lines.append("")
        report_lines.append(
            "Fields include NIBRS batch header reporting months (`nibrs_months_reported`) and whether any NIBRS months were reported (`in_nibrs_reported`)."
        )
        report_lines.append("")
        report_lines.append(_md_table(pd.read_csv(deep_paths[key]).head(25)))
        report_lines.append("")

    if compare_summary is not None and compare_paths:
        report_lines.append("## SRS vs NIBRS-derived Part I comparison (agencies present in both)")
        report_lines.append(f"- Output CSV: `{compare_paths['srs_vs_nibrs_by_bucket']}`")
        report_lines.append("")
        report_lines.append(
            "NIBRS-derived counts here are produced by our current SRS-hierarchy conversion (1 Part I offense per incident, highest-ranked)."
        )
        report_lines.append("")
        report_lines.append(_md_table(compare_summary))
        report_lines.append("")
    else:
        report_lines.append("## SRS vs NIBRS-derived Part I comparison (skipped)")
        report_lines.append("- Canonical parquet not provided or missing; pass `--canonical` pointing to a canonical agency-year parquet to enable this section.")
        report_lines.append("")

    report_lines.append("## Next decision (to be made by statistical review)")
    report_lines.append("Choose a single fixed canonicalization policy for the production pipeline:")
    report_lines.append("- Trust Kaplan/FBI SRS-style counts as canonical and handle partial/missing months via a trend/imputation model (Workstream C1).")
    report_lines.append("- Substitute NIBRS-derived counts for partial reporters when NIBRS coverage is stronger, and use trend/imputation only for true non-reporters.")
    report_lines.append("- Use NIBRS-derived counts as canonical wherever NIBRS exists and use SRS only for SRS-only reporters (requires validating conversion alignment).")
    report_lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n")

    # Machine-readable manifest for the statistician.
    manifest = {
        "config": {
            "year_start": cfg.year_start,
            "year_end": cfg.year_end,
            "agency_types": list(cfg.agency_types),
            "leaic_agency_type_codes": list(cfg.leaic_agency_type_codes),
            "deep_dive_years": list(cfg.deep_dive_years),
            "top_partial_agencies": int(cfg.top_partial_agencies),
        },
        "outputs": {
            "report_md": str(report_path),
            "srs_months_bucket_summary": str(srs_summary_path),
            "srs_nibrs_ori_overlap_by_months_bucket": str(overlap_path),
            "srs_vs_nibrs_reporting_months_by_bucket": str(reporting_path),
            "srs_monthly_vs_annual_consistency_summary": str(monthly_consistency_path),
            "srs_monthly_vs_annual_mismatches": str(mismatch_path),
            **{k: str(v) for k, v in nibrs_missing_in_srs_paths.items()},
            **{k: str(v) for k, v in quarterly_paths.items()},
            **{k: str(v) for k, v in deep_paths.items()},
            **{k: str(v) for k, v in compare_paths.items()},
        },
        "inputs": {
            "srs_zip": str(srs_zip),
            "srs_monthly_zip": str(srs_monthly_zip),
            "nibrs_offense_zip": str(nibrs_zip),
            "nibrs_batch_header_zip": str(nibrs_batch_header_zip),
            "leaic_tsv": str(leaic_tsv),
            "canonical_parquet": str(canonical_parquet) if canonical_parquet is not None else None,
        },
    }
    _write_json(out_dir / "manifest.json", manifest)

    outputs: dict[str, Path] = {
        "report_md": report_path,
        "manifest_json": out_dir / "manifest.json",
        "srs_summary_csv": srs_summary_path,
        "overlap_csv": overlap_path,
        "reporting_months_csv": reporting_path,
        "monthly_consistency_summary_csv": monthly_consistency_path,
        "monthly_vs_annual_mismatches_csv": mismatch_path,
        **nibrs_missing_in_srs_paths,
        **quarterly_paths,
        **deep_paths,
        **compare_paths,
    }
    return outputs


def _srs_offense_cols() -> dict[str, str]:
    return {
        "murder": "actual_murder",
        "rape": "actual_rape_total",
        "robbery": "actual_robbery_total",
        "aggravated_assault": "actual_assault_aggravated",
        "burglary": "actual_burglary_total",
        "larceny": "actual_theft_total",
        "motor_vehicle_theft": "actual_motor_vehicle_theft_total",
    }
