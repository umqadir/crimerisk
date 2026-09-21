"""Parse the CT Crime Insight town/offense list export for the DESPP lane.

The source export has four report-metadata lines followed by a CSV header.  Its
``Jurisdiction by Geography`` dimension deliberately mixes bare-town rollups,
municipal/campus/tribal agencies, and ``CSP - <Town>`` jurisdiction members.  The
DESPP lane uses only the CSP member for each registry town, then assigns the
registry's synthetic ``CTSP###00`` identifier.  Bare-town rows are retained only
for a fail-closed reconciliation.

The output is one row per registry unit and selected Part-I offense for 2024.
Blank measure cells are structural zeros in the complete export grid; commas in
thousands-formatted measures are removed before numeric conversion.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.ct_town_lane import (  # noqa: E402
    get_ct_town_parsed_path,
    load_ct_town_coverage_registry,
    validate_ct_town_parsed_rows,
)
from crimerisk.paths import RepoPaths  # noqa: E402


REPORT_TITLE = "Crimes by Offense and Town"
REPORT_DATA_SOURCE = "Data source: Connecticut_SS, Offense Data"
REPORT_JURISDICTION_SLICER = "Jurisdiction by Type: Connecticut"
SOURCE_COLUMNS = (
    "Offense Type",
    "Incident Date",
    "Jurisdiction by Geography",
    "Number of Crimes",
)

# These are source rollups/members, not fuzzy aliases.  Selecting one exact source
# row per output offense prevents additive double counting of rollups and children.
OFFENSE_MAP = {
    "Murder and Nonnegligent Manslaughter": "murder",
    "All Rape": "rape",
    "Robbery": "robbery",
    "Aggravated Assault": "aggravated_assault",
    "Burglary/Breaking & Entering": "burglary",
    "Larceny_Theft Offenses Total": "larceny",
    "Motor Vehicle Theft": "motor_vehicle_theft",
}

# NIBRS 23A-23H members represented by the source's larceny total rollup.
LARCENY_NIBRS_MEMBERS = (
    "Pocket-picking",
    "Purse-snatching",
    "Shoplifting",
    "Theft From Building",
    "Theft From Coin-Operated Machine or Device",
    "Theft From Motor Vehicle",
    "Theft of Motor Vehicle Parts or Accessories",
    "All Other Larceny",
)

# Source display aliases observed in the official export.  Keys are registry names.
CSP_TOWN_SOURCE_ALIASES = {"Barkhamsted": "Barkhamstead"}
BARE_TOWN_SOURCE_ALIASES = {"Windham": "Willimantic"}


def _read_report_metadata(raw_path: Path) -> list[list[str]]:
    with raw_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        return [next(reader, []) for _ in range(4)]


def read_ct_crime_insight_list_csv(raw_path: Path) -> pd.DataFrame:
    """Read and validate the fixed report/list-export envelope."""
    metadata = _read_report_metadata(raw_path)
    expected = (REPORT_TITLE, None, REPORT_DATA_SOURCE, REPORT_JURISDICTION_SLICER)
    observed = tuple(row[0].strip() if row else "" for row in metadata)
    if observed[0] != expected[0] or observed[2:] != expected[2:]:
        raise ValueError(
            "Unexpected CT Crime Insight report metadata: "
            f"expected title/data source/slicer {expected}, found {observed}"
        )
    if not observed[1].startswith("Current date:"):
        raise ValueError(f"Unexpected CT Crime Insight retrieval line: {observed[1]!r}")

    raw = pd.read_csv(raw_path, skiprows=4, dtype="string")
    missing = sorted(set(SOURCE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"CT Crime Insight export missing columns: {missing}")
    extra = [column for column in raw.columns if column not in SOURCE_COLUMNS]
    if extra and raw[extra].notna().any().any():
        raise ValueError(f"CT Crime Insight export has populated extra columns: {extra}")
    raw = raw[list(SOURCE_COLUMNS)].copy()
    if raw.duplicated(
        ["Offense Type", "Incident Date", "Jurisdiction by Geography"]
    ).any():
        raise ValueError("CT Crime Insight export has duplicate offense-year-geography rows")

    measure_text = raw["Number of Crimes"].fillna("").str.strip()
    normalized = measure_text.str.replace(",", "", regex=False)
    invalid = normalized.ne("") & ~normalized.str.fullmatch(r"\d+")
    if invalid.any():
        raise ValueError(
            "CT Crime Insight export has invalid Number of Crimes values: "
            + str(sorted(measure_text[invalid].unique().tolist())[:20])
        )
    # The export is a complete category × year × geography grid and represents
    # zeroes as blank cells.  Explicit zeroes and blanks therefore have one meaning.
    raw["count"] = pd.to_numeric(normalized.mask(normalized.eq(""), "0"), errors="raise")
    raw["year"] = pd.to_numeric(raw["Incident Date"], errors="raise").astype("Int64")
    return raw


def _source_town_name(town: str, *, csp: bool) -> str:
    aliases = CSP_TOWN_SOURCE_ALIASES if csp else BARE_TOWN_SOURCE_ALIASES
    return aliases.get(town, town)


def reconcile_ct_town_members(
    raw: pd.DataFrame,
    *,
    registry: pd.DataFrame,
    year: int = 2024,
) -> pd.DataFrame:
    """Compare each registry town's CSP member with its bare-town rollup."""
    selected = raw[
        raw["year"].eq(int(year)) & raw["Offense Type"].isin(OFFENSE_MAP)
    ].copy()
    selected["offense"] = selected["Offense Type"].map(OFFENSE_MAP)
    rows: list[dict[str, object]] = []
    for item in registry.itertuples(index=False):
        town = str(item.town)
        csp_geography = "CSP - " + _source_town_name(town, csp=True)
        bare_geography = _source_town_name(town, csp=False)
        for source_offense, offense in OFFENSE_MAP.items():
            csp = selected[
                selected["Jurisdiction by Geography"].eq(csp_geography)
                & selected["Offense Type"].eq(source_offense)
            ]
            bare = selected[
                selected["Jurisdiction by Geography"].eq(bare_geography)
                & selected["Offense Type"].eq(source_offense)
            ]
            if len(csp) != 1 or len(bare) != 1:
                raise ValueError(
                    "CT Crime Insight registry-town coverage is incomplete: "
                    f"town={town!r}, offense={source_offense!r}, "
                    f"CSP rows={len(csp)}, bare-town rows={len(bare)}"
                )
            csp_count = int(csp["count"].iloc[0])
            bare_count = int(bare["count"].iloc[0])
            rows.append(
                {
                    "town": town,
                    "coverage_type": item.coverage_type,
                    "current_master_municipal_cousub": item.current_master_municipal_cousub,
                    "reporting_unit": item.ct_despp_reporting_unit,
                    "source_offense": source_offense,
                    "offense": offense,
                    "csp_count": csp_count,
                    "bare_town_count": bare_count,
                    "bare_minus_csp": bare_count - csp_count,
                }
            )
    return pd.DataFrame(rows)


def parse_ct_despp_town_srs_csv(
    raw_path: Path,
    *,
    paths: RepoPaths,
    year: int = 2024,
) -> pd.DataFrame:
    raw = read_ct_crime_insight_list_csv(raw_path)
    registry = load_ct_town_coverage_registry(paths)
    reconciliation = reconcile_ct_town_members(raw, registry=registry, year=year)

    parsed = reconciliation[reconciliation["reporting_unit"].notna()].copy()
    parsed = parsed.rename(columns={"csp_count": "count"})
    parsed["year"] = int(year)
    parsed["definition_basis"] = (
        "CT Crime Insight offense-category count; CSP town jurisdiction member"
    )
    parsed = validate_ct_town_parsed_rows(parsed, registry=registry, year=year)
    return parsed[
        [
            "reporting_unit",
            "town",
            "year",
            "source_offense",
            "offense",
            "count",
            "definition_basis",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    out_path = args.out or get_ct_town_parsed_path(paths)
    parsed = parse_ct_despp_town_srs_csv(args.raw_csv, paths=paths, year=int(args.year))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(out_path, index=False)
    print(
        {
            "rows": len(parsed),
            "units": parsed["reporting_unit"].nunique(),
            "out": str(out_path),
        }
    )


if __name__ == "__main__":
    main()
