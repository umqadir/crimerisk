from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CountyPopulationEstimates:
    year: int
    df: pd.DataFrame  # columns: state_fips, county_fips, pop_<year>


def load_county_population_estimates(*, csv_path: Path, year: int = 2024) -> CountyPopulationEstimates:
    """Load county POPESTIMATE<year> controls from a Census PopEst county totals file.

    The vintage file carries one POPESTIMATE column per estimate year (the Vintage 2025
    file covers POPESTIMATE2020..POPESTIMATE2025), so a single file serves both the 2024
    and 2025 builds; a year the file does not cover fails loudly here rather than
    silently controlling on a different year.
    """
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, dtype=str, encoding="latin-1")
    estimate_col = f"POPESTIMATE{int(year)}"
    if estimate_col not in df.columns:
        available = ", ".join(sorted(col for col in df.columns if col.startswith("POPESTIMATE")))
        raise ValueError(
            f"County population estimates file {csv_path} has no {estimate_col} column "
            f"(available: {available}). Use a PopEst vintage that covers {int(year)}."
        )
    df = df[(df["COUNTY"] != "000") & (df["SUMLEV"] == "050")].copy()
    df["state_fips"] = df["STATE"].astype(str).str.zfill(2)
    df["county_fips"] = df["COUNTY"].astype(str).str.zfill(3)
    pop_col = f"pop_{int(year)}"
    df[pop_col] = pd.to_numeric(df[estimate_col], errors="coerce").astype("Int64")
    out = df[["state_fips", "county_fips", pop_col]].dropna(subset=[pop_col]).reset_index(drop=True)
    return CountyPopulationEstimates(year=int(year), df=out)


@dataclass(frozen=True)
class CountyPopulationProjection:
    target_year: int
    df: pd.DataFrame  # columns: state_fips, county_fips, pop_target


def project_county_population(
    *,
    csv_path: Path,
    target_year: int,
    base_year: int = 2024,
    start_year: int = 2020,
) -> CountyPopulationProjection:
    if target_year < base_year:
        raise ValueError("target_year must be >= base_year")
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, dtype=str, encoding="latin-1")
    df = df[(df["COUNTY"] != "000") & (df["SUMLEV"] == "050")].copy()
    df["state_fips"] = df["STATE"].astype(str).str.zfill(2)
    df["county_fips"] = df["COUNTY"].astype(str).str.zfill(3)

    pop_start = pd.to_numeric(df[f"POPESTIMATE{start_year}"], errors="coerce")
    pop_base = pd.to_numeric(df[f"POPESTIMATE{base_year}"], errors="coerce")

    years = base_year - start_year
    years_forward = target_year - base_year
    growth = pd.Series(1.0, index=df.index, dtype=float)
    valid = pop_start.notna() & pop_base.notna() & (pop_start > 0)
    growth.loc[valid] = (pop_base.loc[valid] / pop_start.loc[valid]) ** (1.0 / float(years))
    growth = growth.replace([float("inf"), float("-inf")], 1.0).fillna(1.0).clip(lower=0.0)

    pop_target = (pop_base * (growth**years_forward)).round()
    pop_target = pop_target.astype("Int64")
    out = pd.DataFrame(
        {
            "state_fips": df["state_fips"],
            "county_fips": df["county_fips"],
            "pop_target": pop_target,
        }
    ).dropna(subset=["pop_target"])

    return CountyPopulationProjection(target_year=target_year, df=out.reset_index(drop=True))


def align_tract_population_to_county_projection(
    *,
    acs_tract_df: pd.DataFrame,
    county_projection: CountyPopulationProjection,
    acs_population_col: str = "total_population",
) -> pd.DataFrame:
    required = {"tract_id", "state_fips", "county_fips", acs_population_col}
    missing = required - set(acs_tract_df.columns)
    if missing:
        raise ValueError(f"acs_tract_df missing columns: {sorted(missing)}")

    tracts = acs_tract_df.copy()
    tracts["state_fips"] = tracts["state_fips"].astype(str).str.zfill(2)
    tracts["county_fips"] = tracts["county_fips"].astype(str).str.zfill(3)
    tracts["acs_population"] = pd.to_numeric(tracts[acs_population_col], errors="coerce")

    county_sum = (
        tracts.groupby(["state_fips", "county_fips"])["acs_population"].sum(min_count=1).reset_index(name="acs_pop_sum")
    )
    proj = county_projection.df.copy()
    merged = county_sum.merge(proj, on=["state_fips", "county_fips"], how="left")
    merged["pop_factor_target"] = merged["pop_target"] / merged["acs_pop_sum"]
    merged["pop_factor_target"] = (
        merged["pop_factor_target"].replace([float("inf"), float("-inf")], pd.NA).fillna(1.0).clip(lower=0.0)
    )

    tracts = tracts.merge(
        merged[["state_fips", "county_fips", "pop_target", "pop_factor_target"]],
        on=["state_fips", "county_fips"],
        how="left",
    )
    col = f"population_{county_projection.target_year}"
    tracts[col] = (tracts["acs_population"] * tracts["pop_factor_target"]).round()
    tracts[col] = tracts[col].fillna(tracts["acs_population"]).astype("Int64")
    return tracts


def align_tract_population_to_county(
    *,
    acs_tract_df: pd.DataFrame,
    county_estimates: CountyPopulationEstimates,
    acs_population_col: str = "total_population",
) -> pd.DataFrame:
    required = {"tract_id", "state_fips", "county_fips", acs_population_col}
    missing = required - set(acs_tract_df.columns)
    if missing:
        raise ValueError(f"acs_tract_df missing columns: {sorted(missing)}")

    year = int(county_estimates.year)
    pop_col = f"pop_{year}"
    factor_col = f"pop_factor_{year}"
    population_col = f"population_{year}"

    tracts = acs_tract_df.copy()
    tracts["state_fips"] = tracts["state_fips"].astype(str).str.zfill(2)
    tracts["county_fips"] = tracts["county_fips"].astype(str).str.zfill(3)
    tracts["acs_population"] = pd.to_numeric(tracts[acs_population_col], errors="coerce")

    county_sum = (
        tracts.groupby(["state_fips", "county_fips"])["acs_population"].sum(min_count=1).reset_index(name="acs_pop_sum")
    )
    pop = county_estimates.df.copy()
    merged = county_sum.merge(pop, on=["state_fips", "county_fips"], how="left")
    merged[factor_col] = merged[pop_col] / merged["acs_pop_sum"]
    merged[factor_col] = (
        merged[factor_col].replace([float("inf"), float("-inf")], pd.NA).fillna(1.0).clip(lower=0.0)
    )

    tracts = tracts.merge(merged[["state_fips", "county_fips", pop_col, factor_col]], on=["state_fips", "county_fips"], how="left")
    tracts[population_col] = (tracts["acs_population"] * tracts[factor_col]).round()
    tracts[population_col] = tracts[population_col].fillna(tracts["acs_population"]).astype("Int64")

    return tracts


def align_block_group_population_to_county(
    *,
    acs_bg_df: pd.DataFrame,
    county_estimates: CountyPopulationEstimates,
    acs_population_col: str = "total_population",
) -> pd.DataFrame:
    required = {"bg_id", "state_fips", "county_fips", acs_population_col}
    missing = required - set(acs_bg_df.columns)
    if missing:
        raise ValueError(f"acs_bg_df missing columns: {sorted(missing)}")

    year = int(county_estimates.year)
    pop_col = f"pop_{year}"
    factor_col = f"pop_factor_{year}"
    population_col = f"population_{year}"

    bgs = acs_bg_df.copy()
    bgs["state_fips"] = bgs["state_fips"].astype(str).str.zfill(2)
    bgs["county_fips"] = bgs["county_fips"].astype(str).str.zfill(3)
    bgs["acs_population"] = pd.to_numeric(bgs[acs_population_col], errors="coerce")

    county_sum = (
        bgs.groupby(["state_fips", "county_fips"])["acs_population"].sum(min_count=1).reset_index(name="acs_pop_sum")
    )
    pop = county_estimates.df.copy()
    merged = county_sum.merge(pop, on=["state_fips", "county_fips"], how="left")
    merged[factor_col] = merged[pop_col] / merged["acs_pop_sum"]
    merged[factor_col] = (
        merged[factor_col].replace([float("inf"), float("-inf")], pd.NA).fillna(1.0).clip(lower=0.0)
    )

    bgs = bgs.merge(merged[["state_fips", "county_fips", pop_col, factor_col]], on=["state_fips", "county_fips"], how="left")
    bgs[population_col] = (bgs["acs_population"] * bgs[factor_col]).round()
    bgs[population_col] = bgs[population_col].fillna(bgs["acs_population"]).astype("Int64")
    return bgs
