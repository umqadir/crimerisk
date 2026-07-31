from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from crimerisk.agency_identity import (
    load_fbi_roster_oris,
    resolve_agency_identity,
)
from crimerisk.stage1_adjudications import registry_dependency_paths
from crimerisk.build_freshness import artifact_is_current
from crimerisk.crime import OFFENSES_7
from crimerisk.fbi_publications import (
    cius_local_rows_available,
    match_cius_municipal_rows_to_jurisdictions,
    norm_text,
    parse_cius_local_rows,
)
from crimerisk.crime.nibrs import (
    aggregate_nibrs_year_srs_equivalent,
    ensure_nibrs_batch_header_parquet,
    ensure_nibrs_offense_year_parquet,
    ensure_nibrs_property_year_parquet,
    ensure_nibrs_victim_year_parquet,
)
from crimerisk.crime.srs import (
    SRS_OFFENSE_COLUMN_MAP,
    ensure_srs_month_year_parquet,
    ensure_srs_year_parquet,
)
from crimerisk.local_publications import (
    get_v2_local_publication_input_path,
    load_local_publication_annual_ags_rows,
)
from crimerisk.paths import RepoPaths
from crimerisk.reference import canonicalize_agency_county_fips
from crimerisk.scope import production_scope_excluded
from crimerisk.reference_layers import (
    ReferenceLayerBuildConfig,
    get_v2_reference_output_paths,
    reference_layers_artifacts_are_current,
    write_v2_reference_layers,
)
from crimerisk.stage_locks import blockers_for_stage, stage_write_lock
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    NIBRS_SOURCE,
    SUMMARY_SOURCE,
    default_conversion_status_from_source,
    raw_data_source_from_parts,
    reporting_mode_from_source,
    source_family_from_source,
    source_lane_from_source,
    source_origin_from_parts,
)
from crimerisk.state_publications import (
    get_v2_state_publication_input_path,
    load_state_publication_annual_ags_rows,
)


@dataclass(frozen=True)
class ObservationBuildConfig:
    year_start: int = 2018
    year_end: int = 2024
    srs_lumpy_min_total: float = 500.0
    srs_lumpy_max_month_share: float = 0.4
    srs_lumpy_nonzero_months_max: int = 4
    srs_monthly_annual_mismatch_ratio: float = 0.25
    local_publication_input_path: Path | None = None
    state_publication_input_path: Path | None = None


AGENCY_OBSERVATION_ID_COLUMNS = {
    "source",
    "offense",
    "ori9",
    "ori7",
    "state_fips",
    "state_abbr",
    "county_fips",
    "place_fips",
}


def get_v2_agency_observations_path(paths: RepoPaths) -> Path:
    return paths.state_dir / "observations" / "agency_year_observations.parquet"


def get_v2_jurisdiction_observations_path(paths: RepoPaths) -> Path:
    return paths.state_dir / "observations" / "jurisdiction_year_observations.parquet"


def get_v2_observation_paths(paths: RepoPaths) -> tuple[Path, Path]:
    return get_v2_agency_observations_path(
        paths
    ), get_v2_jurisdiction_observations_path(paths)


def _resolve_observation_config(
    *,
    paths: RepoPaths,
    config: ObservationBuildConfig,
) -> ObservationBuildConfig:
    return ObservationBuildConfig(
        year_start=int(config.year_start),
        year_end=int(config.year_end),
        srs_lumpy_min_total=float(config.srs_lumpy_min_total),
        srs_lumpy_max_month_share=float(config.srs_lumpy_max_month_share),
        srs_lumpy_nonzero_months_max=int(config.srs_lumpy_nonzero_months_max),
        srs_monthly_annual_mismatch_ratio=float(
            config.srs_monthly_annual_mismatch_ratio
        ),
        local_publication_input_path=config.local_publication_input_path
        or get_v2_local_publication_input_path(paths),
        state_publication_input_path=config.state_publication_input_path
        or get_v2_state_publication_input_path(paths),
    )


def observation_dependency_paths(
    paths: RepoPaths,
    *,
    config: ObservationBuildConfig = ObservationBuildConfig(),
) -> list[Path]:
    resolved = _resolve_observation_config(paths=paths, config=config)
    cius_root = paths.data_dir / "FBI-CIUS-Annual"
    cius_sources = (
        sorted(c for c in cius_root.rglob("*") if c.is_file())
        if cius_root.exists()
        else []
    )
    dependencies = [
        *registry_dependency_paths(paths),
        paths.state_dir / "reference" / "agency_master.parquet",
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        paths.data_dir
        / "SRS-Kaplan-1960-2024"
        / "offenses_known_parquet_1960_2024_year.zip",
        paths.data_dir
        / "SRS-Kaplan-1960-2024"
        / "offenses_known_parquet_1960_2024_month.zip",
        paths.data_dir
        / "NIBRS-Kaplan-1991-2024"
        / "offense_segment_parquet_1991_2024.zip",
        paths.data_dir
        / "NIBRS-Kaplan-1991-2024"
        / "batch_header_parquet_1991_2024.zip",
        paths.repo_root / "src" / "crimerisk" / "crime" / "nibrs.py",
        resolved.local_publication_input_path,
        resolved.state_publication_input_path,
        Path(__file__),
        paths.repo_root / "src" / "crimerisk" / "fbi_publications.py",
        paths.repo_root / "src" / "crimerisk" / "local_publications.py",
        paths.repo_root / "src" / "crimerisk" / "state_publications.py",
        paths.repo_root / "src" / "crimerisk" / "source_provenance.py",
    ]
    dependencies.extend(cius_sources)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in dependencies:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def observations_artifacts_are_current(
    paths: RepoPaths,
    *,
    config: ObservationBuildConfig = ObservationBuildConfig(),
    agency_out_path: Path | None = None,
    jurisdiction_out_path: Path | None = None,
) -> bool:
    dependencies = observation_dependency_paths(paths, config=config)
    agency_path = agency_out_path or get_v2_agency_observations_path(paths)
    jurisdiction_path = jurisdiction_out_path or get_v2_jurisdiction_observations_path(
        paths
    )
    return artifact_is_current(agency_path, dependencies) and artifact_is_current(
        jurisdiction_path, dependencies
    )


def _quality_tier_from_months(months: pd.Series) -> pd.Series:
    months_num = pd.to_numeric(months, errors="coerce")
    out = pd.Series("unknown", index=months.index, dtype="object")
    out.loc[months_num.between(1, 5, inclusive="both")] = "sparse"
    out.loc[months_num.between(6, 9, inclusive="both")] = "low"
    out.loc[months_num.between(10, 11, inclusive="both")] = "medium"
    out.loc[months_num >= 12] = "high"
    return out


def _weight_from_tier(tier: pd.Series) -> pd.Series:
    return tier.map(
        {
            "high": 1.0,
            "medium": 0.8,
            "low": 0.5,
            "sparse": 0.25,
            "unknown": 0.1,
        }
    ).astype(float)


def _offense_long(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=[c for c in df.columns if c not in OFFENSES_7],
        value_vars=list(OFFENSES_7),
        var_name="offense",
        value_name="count",
    )
    long_df["source"] = source
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce")
    return long_df


def _load_agency_master(paths: RepoPaths) -> pd.DataFrame:
    agency_master_path = paths.state_dir / "reference" / "agency_master.parquet"
    available = pd.read_parquet(agency_master_path).columns.tolist()
    requested = [
        "ori9",
        "ori7",
        "state_fips",
        "state_abbr",
        "county_fips",
        "place_fips",
        "agency_name_raw",
        "agency_name_std",
        "agency_type_raw",
        "crosswalk_agency_name_std",
        "census_name_std",
        "agency_type_norm",
        "manual_review_flag",
        "population_latest_nibrs",
    ]
    agency_master = pd.read_parquet(
        agency_master_path,
        columns=[col for col in requested if col in available],
    )
    if "population_latest_nibrs" in agency_master.columns:
        agency_master = agency_master.rename(
            columns={"population_latest_nibrs": "population"}
        )
    elif "population" not in agency_master.columns:
        agency_master["population"] = pd.NA
    if "crosswalk_agency_name_std" in agency_master.columns:
        agency_master["crosswalk_agency_name"] = agency_master[
            "crosswalk_agency_name_std"
        ]
    elif "crosswalk_agency_name" not in agency_master.columns:
        agency_master["crosswalk_agency_name"] = pd.NA
    if "census_name_std" in agency_master.columns:
        agency_master["census_name"] = agency_master["census_name_std"]
    elif "census_name" not in agency_master.columns:
        agency_master["census_name"] = pd.NA
    return agency_master


def _build_agency_alias_frame(agency_master: pd.DataFrame) -> pd.DataFrame:
    alias_frames: list[pd.DataFrame] = []
    for alias_col in [
        "agency_name_std",
        "crosswalk_agency_name_std",
        "census_name_std",
    ]:
        part = agency_master[
            ["ori9", "state_abbr", "agency_type_norm", alias_col]
        ].rename(columns={alias_col: "publication_name_std"})
        part["publication_name_std"] = part["publication_name_std"].map(norm_text)
        alias_frames.append(part)
    aliases = pd.concat(alias_frames, ignore_index=True)
    aliases = (
        aliases[aliases["publication_name_std"].notna()]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return aliases


def _build_cius_summary_promotions(
    *,
    paths: RepoPaths,
    config: ObservationBuildConfig,
    srs_obs: pd.DataFrame,
    agency_master: pd.DataFrame,
) -> pd.DataFrame:
    promotions: list[pd.DataFrame] = []
    for year in range(int(config.year_start), int(config.year_end) + 1):
        if not cius_local_rows_available(paths, year=year):
            continue

        cius_rows = parse_cius_local_rows(paths, year=year)
        srs_year = srs_obs[srs_obs["year"].astype(int).eq(year)].copy()
        if srs_year.empty or cius_rows.empty:
            continue

        municipal = cius_rows[
            cius_rows["publication_collection"].eq("table8_city")
        ].copy()
        if not municipal.empty:
            jurisdiction_master = pd.read_parquet(
                paths.state_dir / "reference" / "jurisdiction_master.parquet",
                columns=[
                    "jurisdiction_id",
                    "state_abbr",
                    "jurisdiction_type",
                    "jurisdiction_name",
                ],
            )
            municipal = match_cius_municipal_rows_to_jurisdictions(
                cius_rows=municipal,
                jurisdictions=jurisdiction_master,
                agency_master=agency_master,
                agency_to_jurisdiction_crosswalk=pd.read_parquet(
                    paths.state_dir
                    / "reference"
                    / "agency_to_jurisdiction_crosswalk.parquet"
                ),
            )
            official_match_count = (
                municipal.groupby(["jurisdiction_id", "offense"], dropna=False)[
                    "official_count"
                ]
                .size()
                .rename("official_match_count")
                .reset_index()
            )
            municipal = municipal.merge(
                official_match_count,
                on=["jurisdiction_id", "offense"],
                how="left",
            )
            municipal = municipal[municipal["official_match_count"].eq(1)].copy()

            crosswalk = pd.read_parquet(
                paths.state_dir
                / "reference"
                / "agency_to_jurisdiction_crosswalk.parquet"
            )
            crosswalk = crosswalk.rename(columns={"ori": "ori9"})
            crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce")
            single_ori = (
                crosswalk.groupby("jurisdiction_id", dropna=False)
                .agg(
                    n_ori=("ori9", "nunique"),
                    max_weight=("weight", "max"),
                    ori9=("ori9", "first"),
                )
                .reset_index()
            )
            single_ori = single_ori[
                single_ori["n_ori"].eq(1) & single_ori["max_weight"].gt(0.999)
            ].copy()

            municipal = municipal.merge(
                single_ori[["jurisdiction_id", "ori9"]],
                on="jurisdiction_id",
                how="inner",
            )
            promoted = srs_year.merge(
                municipal[["ori9", "offense", "official_count"]].rename(
                    columns={"official_count": "count"}
                ),
                on=["ori9", "offense"],
                how="inner",
                suffixes=("", "_cius"),
            )
            promoted["count"] = promoted["count_cius"]
            promoted["annual_part1_total"] = promoted.groupby(["ori9", "year"])[
                "count"
            ].transform("sum")
            promoted["months_reported"] = 12.0
            promoted["months_missing"] = 0.0
            promoted["monthly_lumpiness_flag"] = False
            promoted["quality_tier"] = "high"
            promoted["observation_weight"] = 1.0
            promoted["monthly_part1_total"] = np.nan
            promoted["monthly_row_count"] = np.nan
            promoted["nonzero_months"] = np.nan
            promoted["max_month_share"] = np.nan
            promoted["annual_month_diff_ratio"] = np.nan
            promoted["source"] = CIUS_SOURCE
            promoted["cius_reference_flag"] = True
            promotions.append(promoted.drop(columns=["count_cius"]))

        agency_rows = cius_rows[
            cius_rows["publication_collection"].eq("table9_university")
        ].copy()
        if not agency_rows.empty:
            aliases = _build_agency_alias_frame(agency_master)
            agency_rows["match_collection"] = agency_rows["publication_collection"]
            allowed_types = {
                "table9_university": {"special_jurisdiction"},
            }
            alias_frames: list[pd.DataFrame] = []
            for collection, allowed in allowed_types.items():
                part = aliases[aliases["agency_type_norm"].isin(sorted(allowed))].copy()
                part["match_collection"] = collection
                alias_frames.append(part)
            agency_candidates = pd.concat(alias_frames, ignore_index=True)
            official_match_count = (
                agency_rows.groupby(
                    [
                        "match_collection",
                        "state_abbr",
                        "publication_name_std",
                        "offense",
                    ],
                    dropna=False,
                )["official_count"]
                .size()
                .rename("official_match_count")
                .reset_index()
            )
            agency_rows = agency_rows.merge(
                official_match_count,
                on=[
                    "match_collection",
                    "state_abbr",
                    "publication_name_std",
                    "offense",
                ],
                how="left",
            )
            agency_rows = agency_rows[agency_rows["official_match_count"].eq(1)].copy()
            agency_match_count = (
                agency_candidates.groupby(
                    ["match_collection", "state_abbr", "publication_name_std"],
                    dropna=False,
                )["ori9"]
                .nunique()
                .rename("repo_match_count")
                .reset_index()
            )
            agency_candidates = agency_candidates.merge(
                agency_match_count,
                on=["match_collection", "state_abbr", "publication_name_std"],
                how="left",
            )
            agency_candidates = agency_candidates[
                agency_candidates["repo_match_count"].eq(1)
            ].copy()
            agency_rows = agency_rows.merge(
                agency_candidates[
                    ["match_collection", "state_abbr", "publication_name_std", "ori9"]
                ],
                on=["match_collection", "state_abbr", "publication_name_std"],
                how="inner",
            )
            promoted = srs_year.merge(
                agency_rows[["ori9", "offense", "official_count"]].rename(
                    columns={"official_count": "count"}
                ),
                on=["ori9", "offense"],
                how="inner",
                suffixes=("", "_cius"),
            )
            promoted["count"] = promoted["count_cius"]
            promoted["annual_part1_total"] = promoted.groupby(["ori9", "year"])[
                "count"
            ].transform("sum")
            promoted["months_reported"] = 12.0
            promoted["months_missing"] = 0.0
            promoted["monthly_lumpiness_flag"] = False
            promoted["quality_tier"] = "high"
            promoted["observation_weight"] = 1.0
            promoted["monthly_part1_total"] = np.nan
            promoted["monthly_row_count"] = np.nan
            promoted["nonzero_months"] = np.nan
            promoted["max_month_share"] = np.nan
            promoted["annual_month_diff_ratio"] = np.nan
            promoted["source"] = CIUS_SOURCE
            promoted["cius_reference_flag"] = True
            promotions.append(promoted.drop(columns=["count_cius"]))

    if not promotions:
        return pd.DataFrame(columns=list(srs_obs.columns) + ["cius_reference_flag"])

    promoted_all = pd.concat(promotions, ignore_index=True)
    promoted_all = promoted_all.drop_duplicates(
        subset=["ori9", "year", "offense"], keep="last"
    )
    return promoted_all


def _build_srs_annual_observations(
    *,
    paths: RepoPaths,
    config: ObservationBuildConfig,
    agency_master: pd.DataFrame,
) -> pd.DataFrame:
    srs_zip = (
        paths.data_dir
        / "SRS-Kaplan-1960-2024"
        / "offenses_known_parquet_1960_2024_year.zip"
    )
    srs_parquet_path = ensure_srs_year_parquet(
        zip_path=srs_zip, cache_dir=paths.cache_dir
    ).parquet_path

    offense_select = ",\n      ".join(
        f"CAST({source_col} AS DOUBLE) AS {offense}"
        for offense, source_col in SRS_OFFENSE_COLUMN_MAP.items()
    )
    q = f"""
    SELECT
      COALESCE(NULLIF(ori9, ''), ori || '00') AS ori9,
      COALESCE(NULLIF(ori, ''), SUBSTR(COALESCE(NULLIF(ori9, ''), ori || '00'), 1, 7)) AS ori7,
      CAST(year AS INTEGER) AS year,
      LPAD(CAST(fips_state_code AS VARCHAR), 2, '0') AS state_fips,
      UPPER(state_abb) AS state_abbr,
      LPAD(CAST(fips_county_code AS VARCHAR), 3, '0') AS county_fips,
      LPAD(CAST(fips_place_code AS VARCHAR), 5, '0') AS place_fips,
      agency_name AS agency_name_raw,
      agency_type AS agency_type_raw,
      crosswalk_agency_name,
      census_name,
      CAST(population AS DOUBLE) AS population,
      CAST(number_of_months_missing AS INTEGER) AS months_missing,
      {offense_select}
    FROM read_parquet('{srs_parquet_path.as_posix()}')
    WHERE CAST(year AS INTEGER) BETWEEN {config.year_start} AND {config.year_end}
      AND COALESCE(NULLIF(ori9, ''), ori || '00') IS NOT NULL
    """
    annual_wide = duckdb.sql(q).df()
    annual_wide["annual_part1_total"] = annual_wide[list(OFFENSES_7)].sum(
        axis=1, min_count=1
    )

    monthly_metrics: list[pd.DataFrame] = []
    srs_month_zip = (
        paths.data_dir
        / "SRS-Kaplan-1960-2024"
        / "offenses_known_parquet_1960_2024_month.zip"
    )
    monthly_part1_expr = " + ".join(
        f"COALESCE(CAST({col} AS DOUBLE), 0.0)"
        for col in SRS_OFFENSE_COLUMN_MAP.values()
    )
    for year in range(config.year_start, config.year_end + 1):
        month_parquet_path = ensure_srs_month_year_parquet(
            zip_path=srs_month_zip,
            year=year,
            cache_dir=paths.cache_dir,
        ).parquet_path
        q_month = f"""
        WITH monthly AS (
          SELECT
            COALESCE(NULLIF(ori9, ''), ori || '00') AS ori9,
            CAST(year AS INTEGER) AS year,
            ({monthly_part1_expr}) AS part1_total,
            CAST(COALESCE(month_missing, 0) AS INTEGER) AS month_missing
          FROM read_parquet('{month_parquet_path.as_posix()}')
          WHERE CAST(year AS INTEGER) = {year}
            AND COALESCE(NULLIF(ori9, ''), ori || '00') IS NOT NULL
        )
        SELECT
          ori9,
          year,
          COUNT(*) AS monthly_row_count,
          SUM(part1_total) AS monthly_part1_total,
          SUM(CASE WHEN month_missing = 0 THEN 1 ELSE 0 END) AS non_missing_month_count,
          SUM(CASE WHEN part1_total > 0 THEN 1 ELSE 0 END) AS nonzero_months,
          MAX(part1_total) AS max_month_total
        FROM monthly
        GROUP BY 1, 2
        """
        monthly_metrics.append(duckdb.sql(q_month).df())

    monthly_df = (
        pd.concat(monthly_metrics, ignore_index=True)
        if monthly_metrics
        else pd.DataFrame()
    )
    srs = annual_wide.merge(monthly_df, on=["ori9", "year"], how="left")
    srs["months_missing"] = pd.to_numeric(srs["months_missing"], errors="coerce")
    srs["months_reported"] = pd.to_numeric(
        srs["non_missing_month_count"], errors="coerce"
    ).where(
        pd.to_numeric(srs["non_missing_month_count"], errors="coerce").notna(),
        (12 - pd.to_numeric(srs["months_missing"], errors="coerce")).clip(
            lower=0, upper=12
        ),
    )
    srs["monthly_part1_total"] = pd.to_numeric(
        srs["monthly_part1_total"], errors="coerce"
    )
    srs["non_missing_month_count"] = pd.to_numeric(
        srs["non_missing_month_count"], errors="coerce"
    )
    srs["nonzero_months"] = pd.to_numeric(srs["nonzero_months"], errors="coerce")
    srs["max_month_total"] = pd.to_numeric(srs["max_month_total"], errors="coerce")
    srs["max_month_share"] = np.where(
        pd.to_numeric(srs["monthly_part1_total"], errors="coerce").fillna(0) > 0,
        srs["max_month_total"] / srs["monthly_part1_total"],
        np.nan,
    )
    srs["annual_month_diff_ratio"] = np.where(
        pd.to_numeric(srs["annual_part1_total"], errors="coerce").fillna(0) > 0,
        (srs["monthly_part1_total"] - srs["annual_part1_total"]).abs()
        / srs["annual_part1_total"].clip(lower=1.0),
        np.nan,
    )
    srs["monthly_lumpiness_flag"] = (
        pd.to_numeric(srs["annual_part1_total"], errors="coerce").fillna(0)
        >= config.srs_lumpy_min_total
    ) & (
        (
            pd.to_numeric(srs["nonzero_months"], errors="coerce").fillna(12)
            <= config.srs_lumpy_nonzero_months_max
        )
        | (
            pd.to_numeric(srs["max_month_share"], errors="coerce").fillna(0)
            >= config.srs_lumpy_max_month_share
        )
        | (
            pd.to_numeric(srs["annual_month_diff_ratio"], errors="coerce").fillna(0)
            >= config.srs_monthly_annual_mismatch_ratio
        )
    )

    srs["quality_tier"] = _quality_tier_from_months(srs["months_reported"])
    srs.loc[
        srs["monthly_lumpiness_flag"] & srs["quality_tier"].eq("high"), "quality_tier"
    ] = "medium"
    srs.loc[
        srs["monthly_lumpiness_flag"] & srs["quality_tier"].eq("medium"), "quality_tier"
    ] = "low"
    srs["observation_weight"] = _weight_from_tier(srs["quality_tier"])

    srs_long = _offense_long(
        srs[
            [
                "ori9",
                "ori7",
                "year",
                "state_fips",
                "state_abbr",
                "county_fips",
                "place_fips",
                "agency_name_raw",
                "agency_type_raw",
                "crosswalk_agency_name",
                "census_name",
                "population",
                "months_reported",
                "months_missing",
                "monthly_lumpiness_flag",
                "quality_tier",
                "observation_weight",
                "annual_part1_total",
                "monthly_part1_total",
                "monthly_row_count",
                "nonzero_months",
                "max_month_share",
                "annual_month_diff_ratio",
                *OFFENSES_7,
            ]
        ],
        source=SUMMARY_SOURCE,
    )

    srs_long = srs_long.merge(
        agency_master,
        on="ori9",
        how="left",
        suffixes=("", "_agency_master"),
    )
    for col in ["state_fips", "state_abbr", "county_fips", "place_fips"]:
        agency_col = f"{col}_agency_master"
        if agency_col in srs_long.columns:
            srs_long[col] = srs_long[col].fillna(srs_long[agency_col])
    return srs_long


def _complete_nibrs_offense_rows_with_zeros(
    nibrs: pd.DataFrame, *, submitted_agency_years: pd.DataFrame
) -> pd.DataFrame:
    """Complete every submitted NIBRS agency-year to the full Part I offense set.

    Reporting in NIBRS is an agency-year property -- the batch header's months cover
    the whole submission, not one offense -- so an offense with no incidents in a year
    the agency submitted is a zero over exactly those months. That is the same
    semantics the SRS/Return A lane already carries explicitly, and it holds for partial
    years too: a 5-month submission's absent offenses are zeros over those 5 months, and
    zero uplifts to zero.

    The population that gets completed is the **batch header's** submitted agency-years,
    not the offense rollup's. Keying on the rollup (v19) silently excluded the agencies
    whose whole submission contained no Part I incident at all -- 8,306 agency-years
    over 2018-2024, 1,631 of them in 2024, including Los Angeles County's and San
    Bernardino County's sheriffs -- and downstream read that absence as missing data
    rather than as the all-zero year the header says it is. An agency-year present in
    the rollup but absent from the header (incidents filed without a header record)
    keeps its rows too: the union is the submitted population.

    `incident_months_any` is an agency-year property and carries onto every emitted row;
    `offense_incident_months` is 0 for an offense with no incidents by construction.
    """
    header_years = submitted_agency_years[["ori9", "year"]].drop_duplicates()
    rollup_years = (
        nibrs[["ori9", "year"]].drop_duplicates()
        if not nibrs.empty
        else pd.DataFrame(columns=["ori9", "year"])
    )
    agency_years = pd.concat([header_years, rollup_years], ignore_index=True).drop_duplicates(
        subset=["ori9", "year"]
    )
    if agency_years.empty:
        return nibrs
    incident_months = (
        nibrs[["ori9", "year", "incident_months_any"]].drop_duplicates(subset=["ori9", "year"])
        if not nibrs.empty
        else pd.DataFrame(columns=["ori9", "year", "incident_months_any"])
    )
    agency_years = agency_years.merge(incident_months, on=["ori9", "year"], how="left")
    agency_years["incident_months_any"] = pd.to_numeric(
        agency_years["incident_months_any"], errors="coerce"
    ).fillna(0)

    complete = agency_years.merge(
        pd.DataFrame({"offense": list(OFFENSES_7)}), how="cross"
    )
    counts = (
        nibrs.drop(columns=["incident_months_any"])
        if "incident_months_any" in nibrs.columns
        else nibrs
    )
    out = complete.merge(counts, on=["ori9", "year", "offense"], how="left")
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0).astype(int)
    out["offense_incident_months"] = pd.to_numeric(
        out["offense_incident_months"], errors="coerce"
    ).fillna(0)
    return out


def _build_nibrs_annual_observations(
    *,
    paths: RepoPaths,
    config: ObservationBuildConfig,
    agency_master: pd.DataFrame,
) -> pd.DataFrame:
    nibrs_zip = (
        paths.data_dir
        / "NIBRS-Kaplan-1991-2024"
        / "offense_segment_parquet_1991_2024.zip"
    )
    batch_header_zip = (
        paths.data_dir / "NIBRS-Kaplan-1991-2024" / "batch_header_parquet_1991_2024.zip"
    )
    batch_header_path = ensure_nibrs_batch_header_parquet(
        zip_path=batch_header_zip, cache_dir=paths.cache_dir
    ).parquet_path

    batch_header = duckdb.sql(
        f"""
        SELECT
          ori AS ori9,
          year,
          UPPER(state_abbreviation) AS header_state_abbr,
          CAST(NULLIF(number_of_months_reported, '') AS INTEGER) AS header_months_reported,
          CAST(population AS DOUBLE) AS header_population,
          agency_indicator,
          city_name,
          covered_by_ori
        FROM read_parquet('{batch_header_path.as_posix()}')
        WHERE year BETWEEN {config.year_start} AND {config.year_end}
        """
    ).df()

    nibrs_frames: list[pd.DataFrame] = []
    for year in range(config.year_start, config.year_end + 1):
        offense_parquet_path = ensure_nibrs_offense_year_parquet(
            zip_path=nibrs_zip,
            year=year,
            cache_dir=paths.cache_dir,
        ).parquet_path
        victim_parquet_path = ensure_nibrs_victim_year_parquet(
            zip_path=paths.data_dir
            / "NIBRS-Kaplan-1991-2024"
            / "victim_segment_parquet_1991_2024.zip",
            year=year,
            cache_dir=paths.cache_dir,
        ).parquet_path
        property_parquet_path = ensure_nibrs_property_year_parquet(
            zip_path=paths.data_dir
            / "NIBRS-Kaplan-1991-2024"
            / "property_segment_parquet_1991_2024.zip",
            year=year,
            cache_dir=paths.cache_dir,
        ).parquet_path
        nibrs_frames.append(
            aggregate_nibrs_year_srs_equivalent(
                offense_parquet_path=offense_parquet_path,
                victim_parquet_path=victim_parquet_path,
                property_parquet_path=property_parquet_path,
            )
        )

    nibrs = (
        pd.concat(nibrs_frames, ignore_index=True) if nibrs_frames else pd.DataFrame()
    )
    # The submitted population: header rows claiming at least one reported month.
    # A header row with no months is a registration, not a submission.
    submitted_agency_years = batch_header.loc[
        pd.to_numeric(batch_header["header_months_reported"], errors="coerce").fillna(0.0).gt(0.0),
        ["ori9", "year"],
    ].drop_duplicates()
    nibrs = _complete_nibrs_offense_rows_with_zeros(
        nibrs, submitted_agency_years=submitted_agency_years
    )
    nibrs = nibrs.merge(batch_header, on=["ori9", "year"], how="left")
    nibrs["months_reported"] = pd.to_numeric(
        nibrs["header_months_reported"], errors="coerce"
    ).astype(float)
    fill_mask = nibrs["months_reported"].isna() | (nibrs["months_reported"] <= 0)
    nibrs.loc[fill_mask, "months_reported"] = pd.to_numeric(
        nibrs.loc[fill_mask, "incident_months_any"], errors="coerce"
    )
    nibrs["months_missing"] = np.where(
        pd.to_numeric(nibrs["months_reported"], errors="coerce").notna(),
        12 - pd.to_numeric(nibrs["months_reported"], errors="coerce"),
        np.nan,
    )
    nibrs["monthly_lumpiness_flag"] = False
    nibrs["quality_tier"] = _quality_tier_from_months(nibrs["months_reported"])
    nibrs["observation_weight"] = _weight_from_tier(nibrs["quality_tier"])
    nibrs["population"] = pd.to_numeric(nibrs["header_population"], errors="coerce")

    nibrs["agency_name_raw"] = pd.NA
    nibrs["agency_type_raw"] = "nibrs_srs_equivalent_aggregate"
    nibrs["crosswalk_agency_name"] = pd.NA
    nibrs["census_name"] = pd.NA
    nibrs["annual_part1_total"] = nibrs.groupby(["ori9", "year"])["count"].transform(
        "sum"
    )
    nibrs["monthly_part1_total"] = pd.NA
    nibrs["monthly_row_count"] = pd.NA
    nibrs["nonzero_months"] = pd.to_numeric(
        nibrs["offense_incident_months"], errors="coerce"
    )
    nibrs["max_month_share"] = pd.NA
    nibrs["annual_month_diff_ratio"] = pd.NA
    nibrs["source"] = NIBRS_SOURCE

    nibrs = nibrs.merge(
        agency_master,
        on="ori9",
        how="left",
        suffixes=("", "_agency_master"),
    )

    for col in ["state_fips", "state_abbr", "county_fips", "place_fips"]:
        agency_col = f"{col}_agency_master"
        if agency_col in nibrs.columns:
            nibrs[col] = nibrs[col].fillna(nibrs[agency_col])

    return nibrs[
        [
            "ori9",
            "ori7",
            "year",
            "state_fips",
            "state_abbr",
            "county_fips",
            "place_fips",
            "agency_name_raw",
            "agency_type_raw",
            "crosswalk_agency_name",
            "census_name",
            "population",
            "months_reported",
            "months_missing",
            "monthly_lumpiness_flag",
            "quality_tier",
            "observation_weight",
            "annual_part1_total",
            "monthly_part1_total",
            "monthly_row_count",
            "nonzero_months",
            "max_month_share",
            "annual_month_diff_ratio",
            "offense",
            "count",
            "source",
            "agency_name_std",
            "agency_type_norm",
            "manual_review_flag",
        ]
    ]


def _normalize_agency_observation_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    string_cols = [
        "ori9",
        "ori7",
        "state_fips",
        "state_abbr",
        "county_fips",
        "place_fips",
        "agency_name_raw",
        "agency_name_std",
        "agency_type_raw",
        "agency_type_norm",
        "crosswalk_agency_name",
        "census_name",
        "manual_review_flag",
        "quality_tier",
        "offense",
        "source",
        "source_lane",
        "source_family",
        "source_origin",
        "reporting_mode",
        "conversion_status",
        "annual_batch_detector_reason",
        "raw_data_source",
    ]
    numeric_cols = [
        "year",
        "population",
        "months_reported",
        "months_missing",
        "observation_weight",
        "annual_part1_total",
        "monthly_part1_total",
        "monthly_row_count",
        "nonzero_months",
        "max_month_share",
        "annual_month_diff_ratio",
        "reported_months_original",
        "annual_batch_panel_median_full_year_total",
        "annual_batch_panel_max_full_year_total",
        "count",
    ]
    bool_cols = [
        "monthly_lumpiness_flag",
        "cius_reference_flag",
        "state_exception_flag",
        "annual_batch_detected",
        "annual_batch_absolute_total_flag",
        "annual_batch_panel_median_flag",
    ]
    for col in string_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].astype("boolean")
    return out


def _batch_detector_reason(
    *,
    absolute_flag: pd.Series,
    panel_flag: pd.Series,
) -> pd.Series:
    out = pd.Series(pd.NA, index=absolute_flag.index, dtype="object")
    out.loc[absolute_flag & panel_flag] = "absolute_total_floor_and_panel_median_ratio"
    out.loc[absolute_flag & ~panel_flag] = "absolute_total_floor"
    out.loc[~absolute_flag & panel_flag] = "panel_median_ratio"
    return out.astype("string")


def _apply_annual_batch_detector(
    observations: pd.DataFrame,
    *,
    config: ObservationBuildConfig,
) -> pd.DataFrame:
    """Promote one/two-month annual batch dumps to annual observations.

    The detector is source-agnostic and panel-relative. A row is annual-batched
    when its reported months are <= 2 and its annual total is either at least the
    legacy absolute floor or at least 60% of the agency's 2019-2023 full-year
    panel median.
    """
    out = observations.copy()
    out["reported_months_original"] = pd.to_numeric(out["months_reported"], errors="coerce")
    out["annual_batch_detected"] = False
    out["annual_batch_detector_reason"] = pd.NA
    out["annual_batch_panel_median_full_year_total"] = np.nan
    out["annual_batch_panel_max_full_year_total"] = np.nan
    out["annual_batch_absolute_total_flag"] = False
    out["annual_batch_panel_median_flag"] = False

    agency_year = out[
        ["ori9", "year", "source", "reported_months_original", "annual_part1_total"]
    ].drop_duplicates(["ori9", "year", "source"]).copy()
    agency_year["year"] = pd.to_numeric(agency_year["year"], errors="coerce")
    agency_year["reported_months_original"] = pd.to_numeric(
        agency_year["reported_months_original"], errors="coerce"
    )
    agency_year["annual_part1_total"] = pd.to_numeric(
        agency_year["annual_part1_total"], errors="coerce"
    )
    history = agency_year[
        agency_year["year"].between(2019, 2023, inclusive="both")
        & agency_year["reported_months_original"].ge(12.0)
        & agency_year["annual_part1_total"].gt(0.0)
    ].copy()
    panel = (
        history.groupby("ori9", dropna=False)["annual_part1_total"]
        .agg(
            annual_batch_panel_median_full_year_total="median",
            annual_batch_panel_max_full_year_total="max",
        )
        .reset_index()
    )
    out = out.merge(panel, on="ori9", how="left", suffixes=("", "_panel"))
    for col in [
        "annual_batch_panel_median_full_year_total",
        "annual_batch_panel_max_full_year_total",
    ]:
        panel_col = f"{col}_panel"
        if panel_col in out.columns:
            out[col] = pd.to_numeric(out[panel_col], errors="coerce").combine_first(
                pd.to_numeric(out[col], errors="coerce")
            )
            out = out.drop(columns=[panel_col])

    annual_total = pd.to_numeric(out["annual_part1_total"], errors="coerce").fillna(0.0)
    months = pd.to_numeric(out["reported_months_original"], errors="coerce")
    panel_median = pd.to_numeric(
        out["annual_batch_panel_median_full_year_total"], errors="coerce"
    )
    absolute_flag = annual_total.ge(float(config.srs_lumpy_min_total))
    panel_flag = panel_median.gt(0.0) & annual_total.ge(0.6 * panel_median)
    batch_mask = months.le(2.0) & annual_total.gt(0.0) & (absolute_flag | panel_flag)

    out["annual_batch_absolute_total_flag"] = absolute_flag & batch_mask
    out["annual_batch_panel_median_flag"] = panel_flag & batch_mask
    out["annual_batch_detected"] = batch_mask
    out.loc[batch_mask, "annual_batch_detector_reason"] = _batch_detector_reason(
        absolute_flag=absolute_flag,
        panel_flag=panel_flag,
    ).loc[batch_mask]
    out.loc[batch_mask, "months_reported"] = 12.0
    out.loc[batch_mask, "months_missing"] = 0.0
    out.loc[batch_mask, "monthly_lumpiness_flag"] = False
    out.loc[batch_mask, "quality_tier"] = "high"
    out.loc[batch_mask, "observation_weight"] = 1.0
    out.loc[batch_mask, "conversion_status"] = "batched_annual"
    return out


def _concat_agency_observation_frames(
    frames: list[pd.DataFrame],
    *,
    preserve_cols: set[str] | None = None,
) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    out = _normalize_agency_observation_schema(nonempty[0])
    for frame in nonempty[1:]:
        incoming = _normalize_agency_observation_schema(frame)
        out, incoming, common_cols = _prepare_concat_frames(
            out,
            incoming,
            preserve_cols=preserve_cols,
        )
        out = pd.concat([out, incoming], ignore_index=True, sort=False)
        out = out.reindex(columns=common_cols)
    return out.reset_index(drop=True)


# Return A carries unfounded-offense adjustments, so a year in which an agency
# unfounded more offences of a category than it reported can publish a small negative
# residual (29 rows over 2018-2024, -37 counts in total, all rape/MVT/burglary at -1
# to -4). A negative annual count is not a count: it is clamped to zero with the
# clamped amount recorded, and the post-condition below then forbids negatives in the
# panel outright. The clamp is licensed only on the Return A lane, where the
# adjustment mechanism is documented; a negative anywhere else is a parse error and
# fails the build.
NEGATIVE_COUNT_CLAMP_ELIGIBLE_SOURCES = frozenset({SUMMARY_SOURCE})


def _clamp_return_a_negative_counts(observations: pd.DataFrame) -> pd.DataFrame:
    """Clamp Return A's negative adjustment residues to zero, recording the amount."""
    out = observations.copy()
    counts = pd.to_numeric(out["count"], errors="coerce")
    negative = counts.lt(0.0).fillna(False)
    out["negative_count_clamped_amount"] = np.where(negative, -counts.fillna(0.0), 0.0)
    if not negative.any():
        return out
    ineligible = negative & ~out["source"].isin(NEGATIVE_COUNT_CLAMP_ELIGIBLE_SOURCES)
    if ineligible.any():
        raise ValueError(
            f"{int(ineligible.sum())} negative observation count(s) on a lane with no "
            "documented adjustment mechanism: "
            + str(
                out.loc[ineligible, ["ori9", "year", "source", "offense", "count"]]
                .head(20)
                .to_dict(orient="records")
            )
        )
    out.loc[negative, "count"] = 0.0
    return out


def _assert_no_negative_counts(observations: pd.DataFrame) -> None:
    """Fail closed on a negative count in the written panel."""
    counts = pd.to_numeric(observations["count"], errors="coerce")
    negative = counts.lt(0.0).fillna(False)
    if not negative.any():
        return
    raise ValueError(
        f"{int(negative.sum())} agency observation(s) carry a negative count: "
        + str(
            observations.loc[negative, ["ori9", "year", "source", "offense", "count"]]
            .head(20)
            .to_dict(orient="records")
        )
    )


# The two FBI lanes claim a complete offense set per agency-year: Return A emits an
# explicit row per Part I offense, and the NIBRS rollup is completed to all seven
# above. The publication lanes do not, and their absent offences are MISSING rather
# than zero -- CIUS withholds individual cells with the published footnote "The FBI
# determined that the agency's data were overreported. Consequently, those data are
# not included in this table", and the MS TOPS extract applies column suppression.
# The panel therefore simply has no row for them and the preference ladder falls to
# the next lane; see docs/PIPELINE.md Stage 1.
OFFENSE_COMPLETE_SOURCES = frozenset({SUMMARY_SOURCE, NIBRS_SOURCE})


def _assert_fbi_lane_offense_sets_are_complete(observations: pd.DataFrame) -> None:
    """Fail closed if a Return A or NIBRS agency-year is missing an offense row."""
    fbi = observations[observations["source"].isin(OFFENSE_COMPLETE_SOURCES)]
    if fbi.empty:
        return
    sizes = fbi.groupby(["ori9", "year", "source"], dropna=False)["offense"].nunique()
    incomplete = sizes[sizes.ne(len(OFFENSES_7))]
    if incomplete.empty:
        return
    raise ValueError(
        f"{len(incomplete)} FBI-lane agency-year(s) carry fewer than {len(OFFENSES_7)} "
        "offense rows, so an absent offense is indistinguishable from a zero: "
        + str(incomplete.head(20).to_dict())
    )


def _deduplicate_agency_source_candidates(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return observations.copy()
    key_cols = ["ori9", "year", "source", "offense"]
    out = observations.copy()
    out["_dedupe_observation_weight"] = pd.to_numeric(
        out["observation_weight"], errors="coerce"
    ).fillna(-1.0)
    out["_dedupe_months_reported"] = pd.to_numeric(
        out["months_reported"], errors="coerce"
    ).fillna(-1.0)
    out["_dedupe_count"] = pd.to_numeric(out["count"], errors="coerce").fillna(-1.0)
    out = (
        out.sort_values(
            [
                *key_cols,
                "_dedupe_observation_weight",
                "_dedupe_months_reported",
                "_dedupe_count",
            ],
            ascending=[True, True, True, True, False, False, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=key_cols, keep="first")
        .drop(
            columns=[
                "_dedupe_observation_weight",
                "_dedupe_months_reported",
                "_dedupe_count",
            ]
        )
        .reset_index(drop=True)
    )
    return out


def _drop_state_publication_non_reports(
    promotions: pd.DataFrame,
    *,
    srs_obs: pd.DataFrame,
) -> pd.DataFrame:
    """Drop state-publication agency-years that are a non-report, not a zero.

    State-publication sheets (FDLE FIBRS, NY DCJS) carry no months-reported
    field of their own, so an agency that simply did not submit for a year is
    indistinguishable on the sheet from an agency that genuinely reported zero
    for every offense: both arrive here as all-zero rows that would be
    promoted at months_reported=12 / quality high. The SRS Return A panel
    resolves the ambiguity -- it is the FBI-side source that can record a
    literal zero months (a NIBRS row's presence implies at least one reported
    month, and CIUS carries only completed annual compilations). An
    agency-year that is zero for every offense in the state publication AND
    shows zero SRS months reported for the same year (or is absent from the
    SRS panel entirely) is a non-report: it is dropped -- missing, not zero --
    so source selection falls back to the FBI lane and the existing gap-fill
    machinery engages instead of a fabricated full-year zero.
    """
    if promotions.empty:
        return promotions
    srs_months = srs_obs[["ori9", "year", "months_reported"]].copy()
    srs_months["months_reported"] = pd.to_numeric(
        srs_months["months_reported"], errors="coerce"
    ).fillna(0.0)
    srs_max_months = srs_months.groupby(
        [srs_months["ori9"].astype(str), srs_months["year"].astype(int)]
    )["months_reported"].max()

    counts = pd.to_numeric(promotions["count"], errors="coerce").fillna(0.0)
    agency_year_total = counts.groupby(
        [promotions["ori9"], promotions["year"]]
    ).transform("sum")
    lookup_keys = pd.MultiIndex.from_arrays(
        [promotions["ori9"].astype(str), promotions["year"].astype(int)]
    )
    months_for_rows = pd.Series(
        srs_max_months.reindex(lookup_keys).fillna(0.0).to_numpy(),
        index=promotions.index,
    )
    is_non_report = agency_year_total.le(0.0) & months_for_rows.le(0.0)
    if not is_non_report.any():
        return promotions
    return promotions.loc[~is_non_report].reset_index(drop=True)


def _prepare_concat_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    preserve_cols: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    preserve_cols = preserve_cols or set()
    common_cols = sorted(set(left.columns) | set(right.columns))
    left = left.reindex(columns=common_cols)
    right = right.reindex(columns=common_cols)

    drop_both: list[str] = []
    drop_left: list[str] = []
    drop_right: list[str] = []
    for col in common_cols:
        if col in preserve_cols:
            continue
        left_all_na = left[col].isna().all()
        right_all_na = right[col].isna().all()
        if left_all_na and right_all_na:
            drop_both.append(col)
        elif left_all_na:
            drop_left.append(col)
        elif right_all_na:
            drop_right.append(col)

    if drop_both or drop_left:
        left = left.drop(columns=drop_both + drop_left, errors="ignore")
    if drop_both or drop_right:
        right = right.drop(columns=drop_both + drop_right, errors="ignore")

    kept_cols = [col for col in common_cols if col not in set(drop_both)]
    return left, right, kept_cols


def build_agency_year_observations(
    *,
    paths: RepoPaths,
    config: ObservationBuildConfig = ObservationBuildConfig(),
) -> pd.DataFrame:
    agency_master = _load_agency_master(paths)
    agency_base = agency_master[
        [
            "ori9",
            "ori7",
            "state_fips",
            "state_abbr",
            "county_fips",
            "place_fips",
            "population",
            "agency_name_raw",
            "agency_name_std",
            "agency_type_raw",
            "agency_type_norm",
            "crosswalk_agency_name",
            "census_name",
            "manual_review_flag",
        ]
    ].drop_duplicates()
    srs_obs = _build_srs_annual_observations(
        paths=paths,
        config=config,
        agency_master=agency_master,
    )
    srs_obs["cius_reference_flag"] = False
    srs_obs["state_exception_flag"] = False
    summary_candidate_frames = [srs_obs]

    cius_promotions = _build_cius_summary_promotions(
        paths=paths,
        config=config,
        srs_obs=srs_obs,
        agency_master=agency_master,
    )
    if not cius_promotions.empty:
        cius_promotions = _normalize_agency_observation_schema(cius_promotions)
        summary_candidate_frames.append(cius_promotions)

    state_publication_input_path = (
        config.state_publication_input_path
        or get_v2_state_publication_input_path(paths)
    )
    # Load the full configured span, not just the target year: multi-year lanes
    # (NY DCJS 2021-2024) feed trend references and fill anchors, mirroring the
    # local-publication lane below.
    state_publication_promotions = load_state_publication_annual_ags_rows(
        paths=paths,
        year_start=config.year_start,
        year_end=config.year_end,
        source_path=state_publication_input_path,
        require_source_exists=True,
    )
    state_publication_promotions = _drop_state_publication_non_reports(
        state_publication_promotions,
        srs_obs=srs_obs,
    )
    if not state_publication_promotions.empty:
        state_publication_promotions = state_publication_promotions.merge(
            agency_base,
            on=["ori9", "state_abbr"],
            how="left",
        )
        state_publication_promotions["quarters_reported"] = pd.to_numeric(
            state_publication_promotions.get("quarters_reported"),
            errors="coerce",
        )
        source_months_reported = pd.to_numeric(
            state_publication_promotions.get("months_reported"),
            errors="coerce",
        )
        state_publication_promotions["months_reported"] = source_months_reported.where(
            source_months_reported.notna(),
            state_publication_promotions["quarters_reported"] * 3.0,
        ).where(
            source_months_reported.notna() | state_publication_promotions["quarters_reported"].notna(),
            12.0,
        )
        state_publication_promotions["months_reported"] = pd.to_numeric(
            state_publication_promotions["months_reported"], errors="coerce"
        ).clip(lower=1.0, upper=12.0)
        state_publication_promotions["months_missing"] = (
            12.0 - state_publication_promotions["months_reported"]
        ).clip(lower=0.0, upper=11.0)
        state_publication_promotions["monthly_lumpiness_flag"] = False
        state_publication_promotions["quality_tier"] = _quality_tier_from_months(
            state_publication_promotions["months_reported"]
        )
        state_publication_promotions["observation_weight"] = _weight_from_tier(
            state_publication_promotions["quality_tier"]
        )
        state_publication_promotions["annual_part1_total"] = (
            state_publication_promotions.groupby(["ori9", "year"])["count"].transform(
                "sum"
            )
        )
        state_publication_promotions["monthly_part1_total"] = np.nan
        state_publication_promotions["monthly_row_count"] = np.nan
        state_publication_promotions["nonzero_months"] = np.nan
        state_publication_promotions["max_month_share"] = np.nan
        state_publication_promotions["annual_month_diff_ratio"] = np.nan
        state_publication_promotions["cius_reference_flag"] = False
        state_publication_promotions["state_exception_flag"] = True
        state_publication_promotions = _normalize_agency_observation_schema(
            state_publication_promotions
        )
        summary_candidate_frames.append(state_publication_promotions)

    local_publication_input_path = (
        config.local_publication_input_path
        or get_v2_local_publication_input_path(paths)
    )
    local_publication_promotions = load_local_publication_annual_ags_rows(
        paths=paths,
        year_start=config.year_start,
        year_end=config.year_end,
        source_path=local_publication_input_path,
        require_source_exists=True,
    )
    if not local_publication_promotions.empty:
        local_publication_promotions = local_publication_promotions.merge(
            agency_base,
            on=["ori9", "state_abbr"],
            how="left",
        )
        local_publication_promotions["months_reported"] = 12.0
        local_publication_promotions["months_missing"] = 0.0
        local_publication_promotions["monthly_lumpiness_flag"] = False
        local_publication_promotions["quality_tier"] = _quality_tier_from_months(
            local_publication_promotions["months_reported"]
        )
        local_publication_promotions["observation_weight"] = _weight_from_tier(
            local_publication_promotions["quality_tier"]
        )
        local_publication_promotions["annual_part1_total"] = (
            local_publication_promotions.groupby(["ori9", "year"])["count"].transform(
                "sum"
            )
        )
        local_publication_promotions["monthly_part1_total"] = np.nan
        local_publication_promotions["monthly_row_count"] = np.nan
        local_publication_promotions["nonzero_months"] = np.nan
        local_publication_promotions["max_month_share"] = np.nan
        local_publication_promotions["annual_month_diff_ratio"] = np.nan
        local_publication_promotions["cius_reference_flag"] = False
        local_publication_promotions["state_exception_flag"] = False
        local_publication_promotions = _normalize_agency_observation_schema(
            local_publication_promotions
        )
        summary_candidate_frames.append(local_publication_promotions)

    summary_obs = _concat_agency_observation_frames(
        summary_candidate_frames,
        preserve_cols=AGENCY_OBSERVATION_ID_COLUMNS,
    )

    nibrs_obs = _build_nibrs_annual_observations(
        paths=paths,
        config=config,
        agency_master=agency_master,
    )
    nibrs_obs["cius_reference_flag"] = False
    nibrs_obs["state_exception_flag"] = False

    summary_obs = _normalize_agency_observation_schema(summary_obs)
    nibrs_obs = _normalize_agency_observation_schema(nibrs_obs)
    observations = _concat_agency_observation_frames(
        [summary_obs, nibrs_obs],
        preserve_cols=AGENCY_OBSERVATION_ID_COLUMNS,
    )
    observations = _deduplicate_agency_source_candidates(observations)

    # Identity before value: fold each NIBRS-lane ORI9 variant onto the summary-lane
    # ORI it duplicates, so source preference sees one agency carrying both lanes
    # rather than two agencies each carrying the same submission. Re-keying can put
    # two rows on one primary key, so the dedupe rule runs again over the result.
    # The deterministic rule runs first and the adjudicated a2/a3 residue lands on top
    # of it inside resolve_agency_identity, which fails closed if the two overlap.
    row_count_before_identity = len(observations)
    observations, identity_summary = resolve_agency_identity(
        observations,
        paths=paths,
        roster_oris=load_fbi_roster_oris(paths, year=int(config.year_end)),
    )
    if len(observations) != row_count_before_identity or identity_summary[
        "twin_adjudicated_variants_merged"
    ]:
        observations = _deduplicate_agency_source_candidates(observations)
    print(
        "build_v2_observations: identity resolution "
        f"{identity_summary['twin_rule_variants_merged']} rule variants + "
        f"{identity_summary['twin_adjudicated_variants_merged']} adjudicated variants merged; "
        f"{identity_summary['twin_adjudicated_cases_quarantined']} adjudicated cases quarantined",
        flush=True,
    )

    observations = _clamp_return_a_negative_counts(observations)
    observations["count"] = pd.to_numeric(
        observations["count"], errors="coerce"
    ).fillna(0.0)
    observations["year"] = pd.to_numeric(observations["year"], errors="coerce").astype(
        int
    )
    observations["ori9"] = observations["ori9"].astype(str)
    observations["months_reported"] = pd.to_numeric(
        observations["months_reported"], errors="coerce"
    )
    observations["months_missing"] = pd.to_numeric(
        observations["months_missing"], errors="coerce"
    )
    observations["monthly_lumpiness_flag"] = (
        observations["monthly_lumpiness_flag"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    observations["observation_weight"] = pd.to_numeric(
        observations["observation_weight"], errors="coerce"
    ).fillna(0.1)
    observations["quality_tier"] = observations["quality_tier"].fillna("unknown")
    observations["cius_reference_flag"] = (
        observations["cius_reference_flag"].astype("boolean").fillna(False).astype(bool)
    )
    observations["source_lane"] = (
        observations["source"].map(source_lane_from_source).fillna("reported_other")
    )
    observations["reporting_mode"] = (
        observations["source"].map(reporting_mode_from_source).fillna("other")
    )
    observations["conversion_status"] = (
        observations["source"]
        .map(default_conversion_status_from_source)
        .fillna("other")
    )
    observations = _apply_annual_batch_detector(observations, config=config)
    observations["source_family"] = (
        observations["source"].map(source_family_from_source).fillna("unknown")
    )
    observations["source_origin"] = [
        source_origin_from_parts(source=source) for source in observations["source"]
    ]
    default_raw_sources = pd.Series(
        [
            raw_data_source_from_parts(source=source)
            for source in observations["source"]
        ],
        index=observations.index,
        dtype="object",
    )
    existing_raw_sources = observations.get("raw_data_source")
    if existing_raw_sources is None:
        observations["raw_data_source"] = default_raw_sources
    else:
        observations["raw_data_source"] = existing_raw_sources.where(
            existing_raw_sources.astype("string").notna(),
            default_raw_sources,
        )
    if "state_exception_flag" in observations.columns:
        observations["state_exception_flag"] = (
            observations["state_exception_flag"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    else:
        observations["state_exception_flag"] = False

    for col in ["state_fips", "county_fips"]:
        observations[col] = (
            observations[col]
            .astype("string")
            .str.zfill(2 if col == "state_fips" else 3)
        )
    observations["place_fips"] = (
        observations["place_fips"].astype("string").str.zfill(5)
    )
    observations["state_abbr"] = observations["state_abbr"].astype("string").str.upper()
    observations = observations[observations["state_fips"].notna()].copy()

    # The same canonicalisation `agency_master` gets, applied to the panel's own
    # county column, which until now came straight off the SRS annual header and
    # still carried the 999 sentinel and retired GEOIDs (02261, 02270, 46113, 51515,
    # 57999). One implementation, one `county_fips_source` vocabulary, so a consumer
    # can gate on placement authority here exactly as it does on the master.
    observations = canonicalize_agency_county_fips(
        observations, paths=paths, roster_year=int(config.year_end)
    )
    observations["production_scope_excluded"] = production_scope_excluded(
        observations["state_abbr"]
    )

    _assert_no_negative_counts(observations)
    _assert_fbi_lane_offense_sets_are_complete(observations)

    observations = observations[
        [
            "ori9",
            "ori7",
            "year",
            "source",
            "source_family",
            "source_origin",
            "raw_data_source",
            "source_lane",
            "reporting_mode",
            "conversion_status",
            "state_exception_flag",
            "cius_reference_flag",
            "offense",
            "count",
            "observation_weight",
            "quality_tier",
            "months_reported",
            "months_missing",
            "monthly_lumpiness_flag",
            "reported_months_original",
            "annual_batch_detected",
            "annual_batch_detector_reason",
            "annual_batch_panel_median_full_year_total",
            "annual_batch_panel_max_full_year_total",
            "annual_batch_absolute_total_flag",
            "annual_batch_panel_median_flag",
            "annual_part1_total",
            "monthly_part1_total",
            "monthly_row_count",
            "nonzero_months",
            "max_month_share",
            "annual_month_diff_ratio",
            "negative_count_clamped_amount",
            "state_fips",
            "state_abbr",
            "county_fips",
            "county_fips_source",
            "place_fips",
            "production_scope_excluded",
            "population",
            "agency_name_raw",
            "agency_name_std",
            "agency_type_raw",
            "agency_type_norm",
            "crosswalk_agency_name",
            "census_name",
            "manual_review_flag",
        ]
    ].sort_values(["ori9", "year", "source", "offense"], kind="mergesort")

    observations = observations.reset_index(drop=True)
    # Carried on the frame so `write_v2_observations` can report what identity resolution did
    # without this builder having to change its return type for every consumer.
    observations.attrs["identity_resolution"] = identity_summary
    return observations


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_num = pd.to_numeric(values, errors="coerce")
    weights_num = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = values_num.notna() & weights_num.gt(0)
    if not mask.any():
        mask = values_num.notna()
        if not mask.any():
            return float("nan")
        return float(values_num.loc[mask].mean())
    return float(np.average(values_num.loc[mask], weights=weights_num.loc[mask]))


def _dominant_label(values: pd.Series, weights: pd.Series) -> str | None:
    labels = values.astype("string")
    weight_values = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = labels.notna() & labels.ne("")
    if not mask.any():
        return None
    support = (
        pd.DataFrame({"label": labels.loc[mask], "weight": weight_values.loc[mask]})
        .groupby("label", dropna=False)["weight"]
        .sum()
        .sort_values(ascending=False, kind="mergesort")
    )
    if support.empty:
        return None
    if len(support) == 1:
        return str(support.index[0])
    top = float(support.iloc[0])
    second = float(support.iloc[1])
    if second <= 0 or top >= second * 5:
        return str(support.index[0])
    return "mixed"


def _build_dominant_label_frame(
    *,
    merged: pd.DataFrame,
    key_cols: list[str],
    label_col: str,
    output_col: str,
) -> pd.DataFrame:
    labels = merged[label_col].astype("string")
    weights = pd.to_numeric(merged["support_weight"], errors="coerce").fillna(0.0)
    mask = labels.notna() & labels.ne("")
    if not mask.any():
        return pd.DataFrame(columns=[*key_cols, output_col])

    support = (
        merged.loc[mask, key_cols]
        .assign(_label=labels.loc[mask].values, _weight=weights.loc[mask].values)
        .groupby([*key_cols, "_label"], dropna=False, as_index=False)["_weight"]
        .sum()
        .sort_values(
            [*key_cols, "_weight", "_label"],
            ascending=[True] * len(key_cols) + [False, True],
            kind="mergesort",
        )
    )
    support["_rank"] = support.groupby(key_cols, dropna=False).cumcount()

    top = support[support["_rank"].eq(0)][[*key_cols, "_label", "_weight"]].rename(
        columns={"_label": "_top_label", "_weight": "_top_weight"}
    )
    second = support[support["_rank"].eq(1)][[*key_cols, "_weight"]].rename(
        columns={"_weight": "_second_weight"}
    )
    out = top.merge(second, on=key_cols, how="left")
    out[output_col] = np.where(
        out["_second_weight"].isna()
        | out["_second_weight"].le(0)
        | out["_top_weight"].ge(out["_second_weight"] * 5),
        out["_top_label"],
        "mixed",
    )
    return out[[*key_cols, output_col]]


def build_jurisdiction_year_observations(
    *,
    paths: RepoPaths,
    agency_year_observations: pd.DataFrame,
) -> pd.DataFrame:
    crosswalk_path = (
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    )
    jurisdiction_path = paths.state_dir / "reference" / "jurisdiction_master.parquet"

    crosswalk = pd.read_parquet(crosswalk_path).rename(columns={"ori": "ori9"})
    jurisdictions = pd.read_parquet(jurisdiction_path)

    merged = agency_year_observations.merge(crosswalk, on="ori9", how="inner")
    merged["allocated_count"] = pd.to_numeric(merged["count"], errors="coerce").fillna(
        0.0
    ) * pd.to_numeric(merged["weight"], errors="coerce").fillna(1.0)
    merged["support_weight"] = merged["allocated_count"].abs()
    merged["support_weight"] = merged["support_weight"].where(
        merged["support_weight"] > 0, merged["weight"].fillna(1.0)
    )

    key_cols = [
        "jurisdiction_id",
        "year",
        "source",
        "offense",
    ]

    grouped = merged.groupby(
        key_cols,
        dropna=False,
        as_index=False,
    ).agg(
        observed_count=("allocated_count", "sum"),
        contributing_agencies=("ori9", "nunique"),
        mean_months_reported=("months_reported", "mean"),
        max_months_reported=("months_reported", "max"),
        any_monthly_lumpiness_flag=("monthly_lumpiness_flag", "max"),
        any_manual_review_flag=("manual_review_flag", "max"),
        any_cius_reference_flag=("cius_reference_flag", "max"),
        any_crosswalk_review_flag=(
            "review_status",
            lambda s: (~s.fillna("auto").eq("auto")).any(),
        ),
        state_exception_flag=("state_exception_flag", "max"),
        weight_support=("support_weight", "sum"),
    )

    obs_weight_num = pd.to_numeric(merged["observation_weight"], errors="coerce")
    support_weight_num = pd.to_numeric(
        merged["support_weight"], errors="coerce"
    ).fillna(0.0)
    weighted_mask = obs_weight_num.notna() & support_weight_num.gt(0)
    weighted_base = merged[key_cols].copy()
    weighted_base["_weighted_numerator"] = obs_weight_num.where(
        weighted_mask, 0.0
    ) * support_weight_num.where(weighted_mask, 0.0)
    weighted_base["_weighted_denominator"] = support_weight_num.where(
        weighted_mask, 0.0
    )
    weighted_base["_fallback_value"] = obs_weight_num

    weighted = (
        weighted_base.groupby(key_cols, dropna=False, as_index=False)
        .agg(
            _weighted_numerator=("_weighted_numerator", "sum"),
            _weighted_denominator=("_weighted_denominator", "sum"),
            _fallback_value=("_fallback_value", "mean"),
        )
        .assign(
            observation_weight=lambda df: np.where(
                df["_weighted_denominator"].gt(0),
                df["_weighted_numerator"] / df["_weighted_denominator"],
                df["_fallback_value"],
            )
        )[[*key_cols, "observation_weight"]]
    )

    source_meta = (
        merged[
            key_cols
            + [
                "source_family",
                "source_origin",
                "raw_data_source",
                "source_lane",
                "reporting_mode",
            ]
        ]
        .drop_duplicates(subset=key_cols)
        .copy()
    )

    conversion_meta = _build_dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="conversion_status",
        output_col="conversion_status",
    )

    relationship_meta = _build_dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="relationship_type",
        output_col="relationship_type",
    )

    overlap_meta = _build_dominant_label_frame(
        merged=merged,
        key_cols=key_cols,
        label_col="overlap_subtype",
        output_col="overlap_subtype",
    )

    out = (
        grouped.merge(
            weighted,
            on=["jurisdiction_id", "year", "source", "offense"],
            how="left",
        )
        .merge(
            source_meta,
            on=["jurisdiction_id", "year", "source", "offense"],
            how="left",
        )
        .merge(
            conversion_meta,
            on=["jurisdiction_id", "year", "source", "offense"],
            how="left",
        )
        .merge(
            relationship_meta,
            on=["jurisdiction_id", "year", "source", "offense"],
            how="left",
        )
        .merge(
            overlap_meta,
            on=["jurisdiction_id", "year", "source", "offense"],
            how="left",
        )
        .merge(jurisdictions, on="jurisdiction_id", how="left")
    )

    out["quality_tier"] = pd.cut(
        out["observation_weight"],
        bins=[-np.inf, 0.15, 0.35, 0.65, 0.9, np.inf],
        labels=["unknown", "sparse", "low", "medium", "high"],
    ).astype("string")

    return out[
        [
            "jurisdiction_id",
            "jurisdiction_type",
            "jurisdiction_name",
            "state_fips",
            "state_abbr",
            "geo_type",
            "geoid",
            "year",
            "source",
            "source_family",
            "source_origin",
            "raw_data_source",
            "source_lane",
            "reporting_mode",
            "conversion_status",
            "state_exception_flag",
            "offense",
            "relationship_type",
            "overlap_subtype",
            "observed_count",
            "observation_weight",
            "quality_tier",
            "contributing_agencies",
            "mean_months_reported",
            "max_months_reported",
            "any_monthly_lumpiness_flag",
            "any_manual_review_flag",
            "any_cius_reference_flag",
            "any_crosswalk_review_flag",
            "geometry_source",
            "is_contracted_place",
            "manual_review_flag",
        ]
    ].sort_values(["jurisdiction_id", "year", "source", "offense"], kind="mergesort")


def write_v2_observations(
    *,
    paths: RepoPaths,
    agency_out_path: Path,
    jurisdiction_out_path: Path,
    config: ObservationBuildConfig = ObservationBuildConfig(),
    blocked_by: tuple[str, ...] | None = None,
    reference_ignore_blockers: tuple[str, ...] = (),
) -> dict[str, int]:
    with stage_write_lock(paths=paths, stage="observations", blocked_by=blocked_by):
        reference_outputs = get_v2_reference_output_paths(paths)
        reference_config = ReferenceLayerBuildConfig(
            year_start=int(config.year_start),
            year_end=int(config.year_end),
        )
        if not reference_layers_artifacts_are_current(
            paths=paths,
            config=reference_config,
            agency_master_path=reference_outputs["agency_master"],
            full_local_out_path=reference_outputs["full_local"],
            full_nonlocal_out_path=reference_outputs["full_nonlocal"],
            jurisdiction_out_path=reference_outputs["jurisdiction_master"],
            crosswalk_out_path=reference_outputs["crosswalk"],
        ):
            write_v2_reference_layers(
                paths=paths,
                full_local_out_path=reference_outputs["full_local"],
                full_nonlocal_out_path=reference_outputs["full_nonlocal"],
                jurisdiction_out_path=reference_outputs["jurisdiction_master"],
                crosswalk_out_path=reference_outputs["crosswalk"],
                config=reference_config,
                blocked_by=blockers_for_stage(
                    "reference_layers",
                    ignore=("observations", *reference_ignore_blockers),
                ),
            )

        agency_out_path.parent.mkdir(parents=True, exist_ok=True)
        jurisdiction_out_path.parent.mkdir(parents=True, exist_ok=True)

        print("build_v2_observations: building agency observations...", flush=True)
        agency_obs = build_agency_year_observations(paths=paths, config=config)
        print(
            f"build_v2_observations: agency observations ready ({len(agency_obs):,} rows)",
            flush=True,
        )
        print(
            "build_v2_observations: building jurisdiction observations...", flush=True
        )
        jurisdiction_obs = build_jurisdiction_year_observations(
            paths=paths, agency_year_observations=agency_obs
        )
        print(
            f"build_v2_observations: jurisdiction observations ready ({len(jurisdiction_obs):,} rows)",
            flush=True,
        )

        print(f"build_v2_observations: writing {agency_out_path}", flush=True)
        agency_obs.to_parquet(agency_out_path, index=False)
        print(f"build_v2_observations: writing {jurisdiction_out_path}", flush=True)
        jurisdiction_obs.to_parquet(jurisdiction_out_path, index=False)

        publication_partial_offense_sets = (
            agency_obs[~agency_obs["source"].isin(OFFENSE_COMPLETE_SOURCES)]
            .groupby(["ori9", "year", "source"], dropna=False)["offense"]
            .nunique()
        )
        return {
            "agency_rows": int(len(agency_obs)),
            "agency_oris": int(agency_obs["ori9"].nunique()),
            **{
                key: value
                for key, value in agency_obs.attrs.get("identity_resolution", {}).items()
                if not isinstance(value, list)
            },
            "agency_cius_reference_rows": (
                int(agency_obs["cius_reference_flag"].fillna(False).sum())
                if "cius_reference_flag" in agency_obs.columns
                else 0
            ),
            "negative_counts_clamped_rows": int(
                pd.to_numeric(agency_obs["negative_count_clamped_amount"], errors="coerce")
                .fillna(0.0)
                .gt(0.0)
                .sum()
            ),
            "production_scope_excluded_rows": int(
                agency_obs["production_scope_excluded"].fillna(False).sum()
            ),
            "publication_lane_partial_offense_agency_years": int(
                publication_partial_offense_sets.lt(len(OFFENSES_7)).sum()
            ),
            "jurisdiction_rows": int(len(jurisdiction_obs)),
            "jurisdictions": int(jurisdiction_obs["jurisdiction_id"].nunique()),
        }
