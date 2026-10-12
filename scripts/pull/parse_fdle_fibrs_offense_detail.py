"""Parse one year sheet of FDLE's FIBRS Offense Detail workbook into the offense-period extract.

Source: https://www.fdle.state.fl.us/CJAB/FIBRS ("FIBRS_Offense_Detail_<vintage>.xlsx").
Each ``FIBRS Offense <year>`` sheet has three header rows (title; offense names at the
first column of each 12-month block; ``<year> <Mon>`` month labels) and one agency row
per line from row 4: County, Agency Name, ``<year> Population Estimate``, then 37
offense blocks of 12 monthly counts.  The workbook carries no ORI and no YTD columns.

Output grain: one row per agency x detail offense x period (M01..M12 and a computed
YTD).  This is a port of the extract half of
``state/review/packets/source/states/fl/normalize_fdle_fibrs_2024.py`` (which made the
staged 2024 extract), parameterized by workbook and year; it reproduces that extract
exactly from ``FIBRS_Offense_Detail_2021-2026.xlsx``.  ``months_reported`` and
``quarters_reported`` are per agency x offense: the count of non-blank monthly cells
(a zero is a reported month) and the number of distinct quarters among them.  YTD is
the sum of the reported months, blank when no month is reported.

The 08-2026 workbook writes 0 into months an agency did not report, so a non-blank cell is
not FDLE's coverage claim. A month counts as reported for an agency only when the agency
has at least one offense of any of the 37 detail types that month; an agency-month that is
zero on every detail offense is a month not reported. An agency too small for an all-zero
month to say anything (fewer than ``ACTIVE_MONTH_MIN_ANNUAL`` offenses in the year across all
detail types) keeps the cell count.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from openpyxl import load_workbook


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_ABBR = "FL"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DETAIL_BLOCK_WIDTH = 12
# At 60 offenses a year across all 37 detail types an agency averages 5 a month, and a genuine
# all-zero month has probability about e^-5 < 1%.
ACTIVE_MONTH_MIN_ANNUAL = 60
FIRST_DATA_ROW = 4
# The offense that opens each bucket section in the workbook's block order.
SECTION_START_OFFENSES = {
    "Murder and Non-Negligent Manslaughter": "violent",
    "Burglary": "property",
    "Negligent Manslaughter": "other",
}
OFFENSE_NAME_CORRECTIONS = {"Prostitutition": "Prostitution"}
EXTRACT_COLUMNS = [
    "state_abbr",
    "year",
    "source_sheet",
    "source_row_number",
    "county",
    "agency_name",
    "agency_name_normalized",
    "population_estimate",
    "quarters_reported",
    "months_reported",
    "offense_level",
    "offense_bucket",
    "offense_name",
    "offense_name_source",
    "offense_slug",
    "period",
    "quarter",
    "month",
    "month_name",
    "incident_count",
    "annual_rate",
]


def default_out_path(year: int) -> Path:
    return REPO_ROOT / "data" / f"FDLE-FIBRS-{int(year)}" / "parsed" / f"fdle_fibrs_{int(year)}_offense_period_extract.csv"


def clean_string(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def parse_count(value):
    if value in (None, "", "--"):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    value = str(value).strip()
    if not value or value == "--":
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def slugify(value: str) -> str:
    value = OFFENSE_NAME_CORRECTIONS.get(value, value).lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def normalize_agency_name(value):
    """Alias key carried in the extract.  Kept byte-identical to the 2024 normalizer,
    including its unbounded ``" ra"`` replacement (``Ranches`` -> ``resident agencynches``);
    the pipeline matches on ``crimerisk.state_publications`` normalizers, not this column."""
    if not value:
        return ""
    value = value.lower()
    replacements = [
        ("&", " and "),
        ("department of law enforcement", "fdle"),
        ("dept. of law enforcement", "fdle"),
        ("department", "dept"),
        ("dept.", "dept"),
        ("police department", "police"),
        ("police dept.", "police"),
        ("police dept", "police"),
        (" sheriff's office", " sheriff"),
        (" sheriffs office", " sheriff"),
        (" sheriff office", " sheriff"),
        (" sheriff's", " sheriff"),
        (" sheriffs", " sheriff"),
        ("univ.", "university"),
        ("univ ", "university "),
        (" pd ", " police "),
        (" pd", " police"),
        (" ra", " resident agency"),
        ("tboc", " field office"),
        (":", " "),
        ("-", " "),
        ("/", " "),
        (",", " "),
    ]
    for old, new in replacements:
        value = value.replace(old, new)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _detail_blocks(header_offenses: tuple, header_months: tuple, year: int) -> list[dict]:
    blocks = []
    section = None
    for index in range(3, len(header_offenses)):
        name_source = clean_string(header_offenses[index])
        if name_source is None:
            continue
        section = SECTION_START_OFFENSES.get(name_source, section)
        if section is None:
            raise ValueError(f"Offense block {name_source!r} precedes the first section start")
        months = [clean_string(v) for v in header_months[index : index + DETAIL_BLOCK_WIDTH]]
        expected = [f"{int(year)} {m}" for m in MONTH_NAMES]
        if months != expected:
            raise ValueError(f"Block {name_source!r} month header {months} != {expected}")
        blocks.append(
            {
                "section": section,
                "start_index": index,
                "offense_name_source": name_source,
                "offense_name": OFFENSE_NAME_CORRECTIONS.get(name_source, name_source),
                "offense_slug": slugify(name_source),
            }
        )
    return blocks


def parse_fdle_fibrs_offense_detail(workbook_path: Path, year: int) -> list[dict]:
    sheet_name = f"FIBRS Offense {int(year)}"
    wb = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        ws = wb[sheet_name]
        rows = ws.iter_rows(values_only=True)
        next(rows)
        header_offenses = next(rows)
        header_months = next(rows)
        if clean_string(header_months[0]) != "County" or clean_string(header_months[1]) != "Agency Name":
            raise ValueError(f"Unexpected identifier header in {sheet_name}: {header_months[:3]}")
        if clean_string(header_offenses[2]) != f"{int(year)} Population Estimate":
            raise ValueError(f"Unexpected population header in {sheet_name}: {header_offenses[2]!r}")
        blocks = _detail_blocks(header_offenses, header_months, year)
        if len(blocks) != 37:
            raise ValueError(f"Expected 37 offense blocks in {sheet_name}, found {len(blocks)}")

        out: list[dict] = []
        for row_number, row in enumerate(rows, start=FIRST_DATA_ROW):
            county = clean_string(row[0])
            agency_name = clean_string(row[1])
            if county is None and agency_name is None:
                continue
            base = {
                "state_abbr": STATE_ABBR,
                "year": int(year),
                "source_sheet": sheet_name,
                "source_row_number": row_number,
                "county": county,
                "agency_name": agency_name,
                "agency_name_normalized": normalize_agency_name(agency_name),
                "population_estimate": parse_count(row[2]),
            }
            month_totals = [0] * DETAIL_BLOCK_WIDTH
            for block in blocks:
                start = block["start_index"]
                for i, value in enumerate(row[start : start + DETAIL_BLOCK_WIDTH]):
                    count = parse_count(value)
                    month_totals[i] += count or 0
            active_months = (
                {i for i, total in enumerate(month_totals, start=1) if total > 0}
                if sum(month_totals) >= ACTIVE_MONTH_MIN_ANNUAL
                else None
            )
            for block in blocks:
                start = block["start_index"]
                raw_months = list(row[start : start + DETAIL_BLOCK_WIDTH])
                month_indexes = [i for i, v in enumerate(raw_months, start=1) if v not in (None, "", "--")]
                if active_months is not None:
                    month_indexes = [i for i in month_indexes if i in active_months]
                counts = [parse_count(v) for v in raw_months]
                months_reported = len(month_indexes)
                quarters_reported = len({(m - 1) // 3 + 1 for m in month_indexes}) if month_indexes else None
                ytd = sum(c for c in counts if c is not None) if month_indexes else None
                block_fields = {
                    "quarters_reported": quarters_reported,
                    "months_reported": months_reported,
                    "offense_level": "detail_offense",
                    "offense_bucket": block["section"],
                    "offense_name": block["offense_name"],
                    "offense_name_source": block["offense_name_source"],
                    "offense_slug": block["offense_slug"],
                }
                for month_index, count in enumerate(counts, start=1):
                    out.append(
                        {
                            **base,
                            **block_fields,
                            "period": f"M{month_index:02d}",
                            "quarter": (month_index - 1) // 3 + 1,
                            "month": month_index,
                            "month_name": MONTH_NAMES[month_index - 1],
                            "incident_count": count,
                            "annual_rate": None,
                        }
                    )
                out.append(
                    {
                        **base,
                        **block_fields,
                        "period": "YTD",
                        "quarter": None,
                        "month": None,
                        "month_name": None,
                        "incident_count": ytd,
                        "annual_rate": None,
                    }
                )
        return out
    finally:
        wb.close()


def write_extract(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTRACT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: data/FDLE-FIBRS-<year>/parsed/...")
    args = parser.parse_args()
    out_path = args.out or default_out_path(args.year)
    rows = parse_fdle_fibrs_offense_detail(args.workbook, args.year)
    write_extract(rows, out_path)
    agencies = len({r["source_row_number"] for r in rows})
    print(f"Wrote {len(rows)} rows ({agencies} agencies) for {args.year}: {out_path}")


if __name__ == "__main__":
    main()
