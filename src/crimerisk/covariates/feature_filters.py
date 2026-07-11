from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSelection:
    feature_cols: list[str]
    numeric_candidate_cols: list[str]
    protected_cols: list[str]
    count_like_cols: list[str]


_PROTECTED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, flags=re.IGNORECASE)
    for p in (
        r"(^|_)race($|_)",
        r"(^|_)ethnic($|_)",
        r"(^|_)hisp($|_)",
        r"(^|_)latino($|_)",
        r"(^|_)white($|_)",
        r"(^|_)black($|_)",
        r"(^|_)afric",
        r"(^|_)asian($|_)",
        r"(^|_)native($|_)",
        r"(^|_)indian($|_)",
        r"(^|_)alaska($|_)",
        r"(^|_)pacific($|_)",
        r"(^|_)islander($|_)",
        r"(^|_)language($|_)",
        r"(^|_)english($|_)",
        r"(^|_)spanish($|_)",
        r"(^|_)ancestr",
        r"(^|_)nativity($|_)",
        r"(^|_)foreign(_|$)",
        r"(^|_)citizen",
        r"(^|_)immigr",
        r"(^|_)born_abroad($|_)",
    )
)


def is_protected_column(col: str) -> bool:
    c = str(col)
    return any(p.search(c) is not None for p in _PROTECTED_PATTERNS)


_COUNT_PREFIXES: tuple[str, ...] = (
    "male_",
    "edu_",
    "rent_",
    "commute_",
    "hh_",
    "hh_income_",
    "housing_",
    "snap_",
    "structure_",
    "tenure_",
    "vehicle_",
    "owner_",
    "renter_",
    "aggregate_vehicles_",
    "internet_",
    "computer_",
    "mobility_",
    "vacancy_",
    "occupants_per_room_",
    "od_",
)
_COUNT_SUFFIXES: tuple[str, ...] = ("_count", "_total")
_ALLOW_SUFFIXES: tuple[str, ...] = ("_rate", "_share", "_per_capita", "_2024", "_2029")
_EXPLICIT_COUNT_COLS: set[str] = {
    "total_population",
    "poverty_count",
    "labor_force",
    "unemployed",
    "jobs_wac",
    "jobs_rac",
    "pop_total",
    "acs_population",
}
_EXPLICIT_NON_COUNT_COLS: set[str] = {
    "housing_density_sqkm",
}


def is_count_like_column(col: str) -> bool:
    c = str(col)
    if c in _EXPLICIT_NON_COUNT_COLS:
        return False
    if c in _EXPLICIT_COUNT_COLS:
        return True
    if c.endswith(_ALLOW_SUFFIXES):
        return False
    if c.endswith(_COUNT_SUFFIXES):
        return True
    return any(c.startswith(prefix) for prefix in _COUNT_PREFIXES)


def _drop_exact_duplicate_numeric_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if len(cols) <= 1:
        return cols
    keep: list[str] = []
    groups: dict[int, list[str]] = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").astype(float)
        key = int(pd.util.hash_pandas_object(series, index=False).sum())
        groups.setdefault(key, []).append(col)
    for group in groups.values():
        if len(group) == 1:
            keep.extend(group)
            continue
        ordered = sorted(group, key=lambda c: (len(str(c)), str(c)))
        representative = ordered[0]
        rep_series = pd.to_numeric(df[representative], errors="coerce").astype(float)
        keep.append(representative)
        for candidate in ordered[1:]:
            cand_series = pd.to_numeric(df[candidate], errors="coerce").astype(float)
            if not np.array_equal(rep_series.to_numpy(), cand_series.to_numpy(), equal_nan=True):
                keep.append(candidate)
    return sorted(set(keep))


def select_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    base_exclude: Iterable[str],
    allow_protected: bool = False,
    exclude_count_like: bool = True,
    min_non_null_share: float = 0.01,
    min_unique_non_null: int = 2,
    min_non_zero_share: float = 0.001,
) -> FeatureSelection:
    exclude = {str(c) for c in base_exclude}
    numeric_candidate_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    protected_cols = sorted([c for c in numeric_candidate_cols if is_protected_column(c)])
    count_like_cols = sorted([c for c in numeric_candidate_cols if is_count_like_column(c)])
    if allow_protected:
        feature_cols = sorted(numeric_candidate_cols)
    else:
        feature_cols = sorted([c for c in numeric_candidate_cols if c not in set(protected_cols)])
    if exclude_count_like:
        feature_cols = [c for c in feature_cols if c not in set(count_like_cols)]
    informative_cols: list[str] = []
    for col in feature_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if float(series.notna().mean()) < float(min_non_null_share):
            continue
        if int(series.nunique(dropna=True)) < int(min_unique_non_null):
            continue
        if float(series.fillna(0).ne(0).mean()) < float(min_non_zero_share):
            continue
        informative_cols.append(col)
    informative_cols = _drop_exact_duplicate_numeric_columns(df, informative_cols)
    return FeatureSelection(
        feature_cols=informative_cols,
        numeric_candidate_cols=sorted(numeric_candidate_cols),
        protected_cols=protected_cols,
        count_like_cols=count_like_cols,
    )
