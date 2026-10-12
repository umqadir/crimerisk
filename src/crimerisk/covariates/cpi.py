from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FRED_CPI_U_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"


@dataclass(frozen=True)
class CpiSeries:
    source_url: str
    df: pd.DataFrame  # columns: date, cpi


def load_cpi_u_series(*, cache_csv_path: Path, source_url: str = FRED_CPI_U_URL) -> CpiSeries:
    cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_csv_path.exists():
        df = pd.read_csv(cache_csv_path)
    else:
        df = pd.read_csv(source_url)
        df.to_csv(cache_csv_path, index=False)

    # FRED format variants:
    # - DATE, CPIAUCSL (fredgraph.csv)
    # - observation_date, CPIAUCSL (api/fred/series/observations CSV)
    if "DATE" in df.columns:
        df = df.rename(columns={"DATE": "date"}).copy()
    elif "observation_date" in df.columns:
        df = df.rename(columns={"observation_date": "date"}).copy()
    df = df.rename(columns={"CPIAUCSL": "cpi"}).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")
    df = df.dropna(subset=["date", "cpi"]).sort_values("date").reset_index(drop=True)
    return CpiSeries(source_url=source_url, df=df[["date", "cpi"]])


def inflation_factor(
    *,
    cpi: CpiSeries,
    from_year: int,
    to_year: int,
) -> float:
    df = cpi.df.copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["cpi"].mean()
    if from_year not in annual.index or to_year not in annual.index:
        raise ValueError(f"Missing CPI annual averages for {from_year} or {to_year}")
    return float(annual.loc[to_year] / annual.loc[from_year])
