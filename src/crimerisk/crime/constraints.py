from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


STATE_ABBR_TO_FIPS: dict[str, str] = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


@dataclass(frozen=True)
class StateConstraints:
    df: pd.DataFrame


def _to_int(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"nan": None, "": None})
        .astype("Int64")
    )


def load_fbi_cde_state_constraints(
    *,
    csv_path: Path,
    year_start: int,
    year_end: int,
) -> StateConstraints:
    raw = pd.read_csv(csv_path, dtype=str)
    raw["year"] = _to_int(raw["year"]).astype(int)
    raw = raw[(raw["year"] >= year_start) & (raw["year"] <= year_end)].copy()

    raw["state_abbr"] = raw["state_abbr"].fillna("").str.strip()
    states = raw[raw["state_abbr"] != ""].copy()
    states["state_fips"] = states["state_abbr"].map(STATE_ABBR_TO_FIPS)
    states = states[states["state_fips"].notna()].copy()

    states["population"] = _to_int(states["population"])
    states["homicide"] = _to_int(states["homicide"])
    states["robbery"] = _to_int(states["robbery"])
    states["aggravated_assault"] = _to_int(states["aggravated_assault"])
    states["burglary"] = _to_int(states["burglary"])
    states["larceny"] = _to_int(states["larceny"])
    states["motor_vehicle_theft"] = _to_int(states["motor_vehicle_theft"])

    rape_revised = _to_int(states["rape_revised"])
    rape_legacy = _to_int(states["rape_legacy"])
    states["rape"] = rape_revised.fillna(rape_legacy)

    out = states[
        [
            "year",
            "state_fips",
            "population",
            "homicide",
            "rape",
            "robbery",
            "aggravated_assault",
            "burglary",
            "larceny",
            "motor_vehicle_theft",
        ]
    ].rename(columns={"homicide": "murder"})

    return StateConstraints(df=out)

