"""Parse the FBI NIBRS "Offense Type by Agency" publication xlsx tables.

The 2024 bundle holds one combined xlsx; the 2025 bundle splits it into a
State/Territory file and a Federal/Tribal file.  Every file has the same layout:
five title/header rows, then one row per agency with State, Agency Type, Agency
Name, Population and one column per NIBRS offense (footnote rows follow the data).

The output is one row per agency and Part-I offense.  Federal agencies, agencies
outside the 50 states + DC (e.g. Guam) and the AK/HI/PR production-scope exclusions
are dropped; Tribal agencies are kept.  Column headers carry footnote digits and
hyphenation breaks, which are stripped before matching.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.fbi_publications import PRODUCTION_SCOPE_EXCLUDE, STATE_NAME_TO_ABBR  # noqa: E402
from crimerisk.reference import _std_text  # noqa: E402

COLLECTION = "nibrs_offense_type_by_agency"
HEADER_ROWS = 6  # title rows + two header rows precede the data
# Exact source columns (footnote digits stripped).  Rape is the "Rape" column only
# (not Sodomy / Sexual Assault With an Object); larceny is the Larceny/Theft total.
OFFENSE_MAP = {
    "Murder and Nonnegligent Manslaughter": "murder",
    "Rape": "rape",
    "Robbery": "robbery",
    "Aggravated Assault": "aggravated_assault",
    "Burglary/ Breaking & Entering": "burglary",
    "Larceny/ Theft Offenses": "larceny",
    "Motor Vehicle Theft": "motor_vehicle_theft",
}
OUTPUT_COLUMNS = [
    "publication_collection", "state_abbr", "publication_agency_type", "publication_name_raw",
    "publication_name_std", "publication_population", "offense", "official_count",
]


def _clean_header(value: object) -> str:
    return re.sub(r"(?<=[a-z])\d+$", "", re.sub(r"\s+", " ", str(value)).strip())


def parse_workbook(data: bytes | Path) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(data) if isinstance(data, bytes) else data, header=None)
    top = raw.iloc[4].astype(str).map(_clean_header)
    sub = raw.iloc[5].astype(str).map(_clean_header)
    cols = {}
    for idx in range(4, raw.shape[1]):
        name = sub[idx] if sub[idx] != "nan" else top[idx]
        if name in OFFENSE_MAP:
            if OFFENSE_MAP[name] in cols:
                raise ValueError(f"duplicate offense column {name!r}")
            cols[OFFENSE_MAP[name]] = idx
    if len(cols) != len(OFFENSE_MAP):
        raise ValueError(f"missing offense columns: {set(OFFENSE_MAP.values()) - set(cols)}")
    body = raw.iloc[HEADER_ROWS:]
    body = body[body[1].notna()]  # footnote rows have no agency type
    state = body[0].astype(str).str.replace(r"(?<=[A-Za-z])\d+$", "", regex=True).str.strip().str.upper()
    out = []
    for offense, idx in cols.items():
        out.append(
            pd.DataFrame(
                {
                    "publication_collection": COLLECTION,
                    "state_abbr": state.map(STATE_NAME_TO_ABBR),
                    "publication_agency_type": body[1].astype(str).str.strip(),
                    "publication_name_raw": body[2],
                    "publication_population": pd.to_numeric(body[3], errors="coerce"),
                    "offense": offense,
                    "official_count": pd.to_numeric(body[idx], errors="raise"),
                }
            )
        )
    df = pd.concat(out, ignore_index=True)
    # Territories (Guam appears in 2025) are outside the CONUS + DC scope, as AK and HI are.
    df = df[df["state_abbr"].notna() & ~df["state_abbr"].isin({*PRODUCTION_SCOPE_EXCLUDE, "GU", "VI", "AS", "MP"})]
    df["publication_name_std"] = df["publication_name_raw"].map(_std_text)
    return df[OUTPUT_COLUMNS].astype({"publication_population": "float64", "official_count": "float64"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    base = REPO_ROOT / f"data/FBI-NIBRS-Tables-{args.year}"
    zip_path = args.zip or base / "raw/statesAndFederal.zip"
    out = args.out or base / f"parsed/nibrs_offense_type_by_agency_{args.year}.parquet"
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(n for n in zf.namelist() if n.lower().endswith(".xlsx")):
            frames.append(parse_workbook(zf.read(name)))
            print(f"{name}: {len(frames[-1])} rows")
    df = pd.concat(frames, ignore_index=True)
    key = ["state_abbr", "publication_agency_type", "publication_name_raw", "offense"]
    if df.duplicated(key).any():
        raise ValueError("duplicate agency/offense keys")
    # offense blocks in OFFENSE_MAP order, agencies in source order within each block
    df = df.sort_values("offense", kind="stable", key=lambda s: s.map(list(OFFENSE_MAP.values()).index))
    df = df.reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
