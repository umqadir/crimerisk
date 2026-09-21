from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import urllib.error
import urllib.request

import pandas as pd


LODES8_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"


def _is_valid_gzip(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, out_path.open("wb") as f:
        f.write(resp.read())


def _lodes_url(state_abbr: str, year: int) -> str:
    st = state_abbr.lower()
    return f"{LODES8_BASE}/{st}/wac/{st}_wac_S000_JT00_{year}.csv.gz"


def _find_or_download_wac(
    *,
    state_abbr: str,
    local_dir: Path,
    cache_dir: Path,
    preferred_year: int = 2023,
    min_year: int = 2002,
) -> tuple[Path, int] | None:
    st = state_abbr.lower()
    local = local_dir / f"{st}_wac_S000_JT00_{preferred_year}.csv.gz"
    if local.exists() and _is_valid_gzip(local):
        return local, preferred_year

    # Prefer cached downloads if present.
    for year in range(preferred_year, min_year - 1, -1):
        cached = cache_dir / "lodes" / st / f"{st}_wac_S000_JT00_{year}.csv.gz"
        if cached.exists() and _is_valid_gzip(cached):
            return cached, year

    # Download the latest available year.
    for year in range(preferred_year, min_year - 1, -1):
        url = _lodes_url(state_abbr, year)
        cached = cache_dir / "lodes" / st / f"{st}_wac_S000_JT00_{year}.csv.gz"
        try:
            _download(url, cached)
        except urllib.error.HTTPError as e:
            if e.code in {403, 404}:
                continue
            raise
        if _is_valid_gzip(cached):
            return cached, year

    return None


def compute_lodes_wac_jobs_by_tract(
    *,
    lodes_dir: Path,
    cache_dir: Path,
    state_abbrs: Iterable[str] | None = None,
    preferred_year: int = 2023,
) -> pd.DataFrame:
    totals: dict[str, int] = defaultdict(int)
    year_used_by_tract: dict[str, int] = {}

    if state_abbrs is None:
        # Infer from existing local files (including potentially corrupted downloads).
        state_abbrs = sorted({p.name[:2] for p in lodes_dir.glob("*_wac_S000_JT00_*.csv.gz")})

    for state_abbr in state_abbrs:
        resolved = _find_or_download_wac(
            state_abbr=state_abbr,
            local_dir=lodes_dir,
            cache_dir=cache_dir,
            preferred_year=preferred_year,
        )
        if resolved is None:
            continue
        path, year_used = resolved
        for chunk in pd.read_csv(
            path,
            usecols=["w_geocode", "C000"],
            dtype={"w_geocode": str, "C000": "Int64"},
            chunksize=1_000_000,
        ):
            chunk = chunk.dropna(subset=["w_geocode", "C000"])
            tract_ids = chunk["w_geocode"].str.slice(0, 11)
            grouped = chunk.assign(tract_id=tract_ids).groupby("tract_id")["C000"].sum()
            for tract_id, value in grouped.items():
                tid = str(tract_id).zfill(11)
                totals[tid] += int(value)
                year_used_by_tract[tid] = int(year_used)

    df = pd.DataFrame({"tract_id": list(totals.keys()), "jobs_wac": list(totals.values())})
    df["tract_id"] = df["tract_id"].astype(str).str.zfill(11)
    df["jobs_wac_year"] = df["tract_id"].map(year_used_by_tract).astype("Int64")
    return df
