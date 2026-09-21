from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from crimerisk.fbi_publications import STATE_NAME_TO_ABBR


_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_NON_STATE_ABBRS = {"AS", "CZ", "GU", "MP", "PR", "VI"}
ESTIMATE_COLUMNS = [
    "year",
    "state_abbr",
    "state_name",
    "population",
    "violent_crime",
    "homicide",
    "rape_legacy",
    "rape_revised",
    "robbery",
    "aggravated_assault",
    "property_crime",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
    "caveats",
]


def _norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return _NON_ALNUM_RE.sub("", str(value).strip().upper())


def _table5_member(zip_path: Path, *, year: int) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".xlsx", ".xls"))
            and "table_5" in name.lower()
            and str(int(year)) in name
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one CIUS Table 5 workbook for {year}, found {candidates}"
        )
    return candidates[0]


def parse_cius_table5_state_estimates(zip_path: Path, *, year: int) -> pd.DataFrame:
    """Parse final CIUS Table 5 state totals into the cumulative CDE schema."""

    member = _table5_member(zip_path, year=year)
    with zipfile.ZipFile(zip_path) as archive:
        raw = pd.read_excel(io.BytesIO(archive.read(member)), header=None)

    header_idx = None
    for idx in range(min(15, len(raw))):
        normalized = {_norm(value) for value in raw.iloc[idx].tolist()}
        if {"STATE", "POPULATION1", "VIOLENTCRIME", "PROPERTYCRIME"}.issubset(
            normalized
        ):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not locate the CIUS Table 5 header in {member}")

    columns = [_norm(value) or f"COL{idx}" for idx, value in enumerate(raw.iloc[header_idx])]
    table = raw.iloc[header_idx + 1 :].copy()
    table.columns = columns
    table["STATE"] = (
        table["STATE"]
        .astype("string")
        .ffill()
        .str.strip()
        .str.upper()
        .str.replace(r"\d+$", "", regex=True)
        .str.strip()
    )

    area_col = "AREA"
    if area_col not in table.columns:
        raise ValueError(f"CIUS Table 5 is missing the Area column: {member}")
    state_totals = table[
        table[area_col].astype("string").str.strip().isin({"State Total", "Total"})
    ].copy()
    state_totals["state_abbr"] = state_totals["STATE"].map(STATE_NAME_TO_ABBR)
    state_totals = state_totals[
        state_totals["state_abbr"].notna()
        & ~state_totals["state_abbr"].isin(_NON_STATE_ABBRS)
    ].copy()

    source_columns = {
        "POPULATION1": "population",
        "VIOLENTCRIME": "violent_crime",
        "MURDERANDNONNEGLIGENTMANSLAUGHTER": "homicide",
        "RAPE": "rape_revised",
        "ROBBERY": "robbery",
        "AGGRAVATEDASSAULT": "aggravated_assault",
        "PROPERTYCRIME": "property_crime",
        "BURGLARY": "burglary",
        "LARCENYTHEFT": "larceny",
        "MOTORVEHICLETHEFT": "motor_vehicle_theft",
    }
    missing = sorted(set(source_columns) - set(state_totals.columns))
    if missing:
        raise ValueError(f"CIUS Table 5 is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "year": int(year),
            "state_abbr": state_totals["state_abbr"].astype("string"),
            "state_name": state_totals["STATE"].str.title(),
        }
    )
    for source, target in source_columns.items():
        out[target] = pd.to_numeric(state_totals[source], errors="coerce")
    out["rape_legacy"] = pd.NA
    out["caveats"] = pd.NA
    out = out.reindex(columns=ESTIMATE_COLUMNS)

    if out["state_abbr"].duplicated().any():
        raise ValueError("CIUS Table 5 produced duplicate state totals")
    expected = set(STATE_NAME_TO_ABBR.values()) - _NON_STATE_ABBRS
    observed = set(out["state_abbr"].dropna().astype(str))
    if observed != expected:
        raise ValueError(
            f"CIUS Table 5 state coverage mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    if out[list(source_columns.values())].isna().any().any():
        raise ValueError("CIUS Table 5 contains a missing required state estimate")

    national = {
        "year": int(year),
        "state_abbr": pd.NA,
        "state_name": "United States Total",
        "rape_legacy": pd.NA,
        "caveats": pd.NA,
    }
    for target in source_columns.values():
        national[target] = float(out[target].sum())
    national_frame = pd.DataFrame(
        {
            column: pd.Series([national.get(column, pd.NA)], dtype=out[column].dtype)
            for column in ESTIMATE_COLUMNS
        }
    )
    out = pd.concat([national_frame, out], ignore_index=True)
    return out.reindex(columns=ESTIMATE_COLUMNS)


def append_cius_state_estimates(
    historical_csv: Path,
    cius_zip: Path,
    *,
    year: int,
) -> pd.DataFrame:
    historical = pd.read_csv(historical_csv, dtype="string")
    if list(historical.columns) != ESTIMATE_COLUMNS:
        raise ValueError(
            f"Unexpected historical estimate schema: {list(historical.columns)}"
        )
    historical_year = pd.to_numeric(historical["year"], errors="coerce")
    historical = historical[~historical_year.eq(int(year))].copy()
    current = parse_cius_table5_state_estimates(cius_zip, year=year)
    out = pd.concat([historical, current], ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out = out.sort_values(
        ["year", "state_abbr"], na_position="first", kind="mergesort"
    ).reset_index(drop=True)
    return out
