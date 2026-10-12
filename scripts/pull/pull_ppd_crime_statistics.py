"""Pull the Philadelphia Police Department "Major Crimes as reported to PPD" year-to-date tables.

    uv run python scripts/pull/pull_ppd_crime_statistics.py [--dates 2024-12-31,2025-12-31] [--out data/PPD-Crime-Statistics]

Source: https://www.phillypolice.com/crime-data/crime-statistics/ (the page's own AJAX calls,
`get_crimedata_content` citywide and `get_crimedata_content_districtwise` per police district; the
site publishes district tables from 2023). Each table for date D carries the year-to-D counts for
D's year and the prior year. Writes `ppd_major_crimes.csv` (as_of_date, scope, district, year,
crime_type, count) and the raw HTML responses next to it.
"""

from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

PAGE = "https://www.phillypolice.com/crime-data/crime-statistics/"
AJAX = "https://www.phillypolice.com/wp-admin/admin-ajax.php"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/129.0.0.0 Safari/537.36")
# The 21 PPD patrol districts.
DISTRICT_IDS = (1, 2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 18, 19, 22, 24, 25, 26, 35, 39)
CRIME_ROW = re.compile(r"^[A-Z][A-Za-z /]+$")


def cells(text: str) -> list[str]:
    text = re.sub(r"<(script|style).*?</\1>", "", text, flags=re.S)
    out = [html.unescape(c).strip() for c in re.split(r"<[^>]+>", text)]
    return [c for c in out if c and c != "-->"]


def parse(text: str) -> tuple[list[int], list[tuple[str, int, int]]]:
    """(the two year-to-date years, [(crime_type, prior_year_count, year_count)])."""
    c = cells(text)
    years: list[int] = []
    rows = []
    i = 0
    while i < len(c):
        if c[i] == "% Change" and i + 3 < len(c) and c[i + 1].isdigit() and c[i + 2].isdigit() and len(c[i + 1]) == 4:
            years = [int(c[i + 1]), int(c[i + 2])]
            i += 3
            continue
        # A data row: label, 4 period counts, % change, prior YTD, current YTD, % change.
        if years and CRIME_ROW.match(c[i]) and i + 8 < len(c) and c[i + 6].replace(",", "").isdigit() \
                and c[i + 7].replace(",", "").isdigit():
            rows.append((c[i], int(c[i + 6].replace(",", "")), int(c[i + 7].replace(",", ""))))
            i += 9
            continue
        i += 1
    return years, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates", default="2024-12-31,2025-12-31")
    parser.add_argument("--out", default="data/PPD-Crime-Statistics")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    session.get(PAGE, timeout=60).raise_for_status()
    headers = {"X-Requested-With": "XMLHttpRequest", "Referer": PAGE, "Origin": "https://www.phillypolice.com"}
    records = []
    for date in [d for d in args.dates.split(",") if d]:
        targets = [("citywide", None)] + [("district", d) for d in DISTRICT_IDS]
        for scope, district in targets:
            data = {"action": "get_crimedata_content", "date": date} if district is None else \
                {"action": "get_crimedata_content_districtwise", "date": date, "id": str(district)}
            for attempt in range(6):
                response = session.post(AJAX, data=data, headers=headers, timeout=60)
                if response.status_code != 429:
                    break
                time.sleep(30 * (attempt + 1))
            response.raise_for_status()
            years, rows = parse(response.text)
            if not rows:
                continue
            name = f"{date}_{scope}{'' if district is None else f'_{district:02d}'}.html"
            (out / "raw" / name).write_text(response.text)
            for crime_type, prior, current in rows:
                for year, count in zip(years, (prior, current)):
                    records.append({"as_of_date": date, "scope": scope, "district": district or "", "year": year,
                                    "crime_type": crime_type, "count": count})
            time.sleep(3)
    table = pd.DataFrame(records)
    table.to_csv(out / "ppd_major_crimes.csv", index=False)
    print(table.groupby(["as_of_date", "scope"])["district"].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
