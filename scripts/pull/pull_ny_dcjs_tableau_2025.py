"""Pull and parse the DCJS "Index Crimes by Agency, 2010-2025" table for the NY DCJS lane.

data.ny.gov dataset ca8h-8gjq (the 2021-2024 source of the lane) stops at 2024. DCJS
publishes 2025 in the HistoricTable view of its Tableau workbook NYSIndexCrimesbyAgency
(criminaljustice.ny.gov/index-crime-reporting-agency-data-dashboard redirects there).
The view's CSV export is one row per (agency, county, measure, year); this script asks
for 2021-2025 with the view's Year filter, pivots it to the parsed-extract schema and
writes data/NY-DCJS-2025/parsed/ny_dcjs_index_crimes_2021_2025.csv: the 2021-2024 rows
of the existing extract copied unchanged, plus the 2025 rows from the export.

Same definition: the view's Data Notes define the seven Index offenses as the FBI does,
with Rape on the FBI's revised definition (DCJS from 2015), the definition of the
data.ny.gov dictionary. Fail-closed checks: the export's 2021-2024 counts must equal
the existing extract row for row; 2025 county totals must equal their agency sums; and
each row's violent/property/index subtotals must equal their offenses.

Conventions carried over from the data.ny.gov extract: the view's "Total" row is the
"County Total" row (months_reported blank); NYC boroughs publish one "<Borough> County
NYPD" row each, which data.ny.gov carries as the borough "County Total" (region "New
York City", months 12); agencies with 0 months reported (all measures blank) are
omitted, as data.ny.gov omits them; region is "New York City" for the five boroughs and
"Non-New York City" elsewhere. One agency is renamed back to its data.ny.gov name.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths  # noqa: E402
from crimerisk.state_publications import build_ny_dcjs_ori_match_frame  # noqa: E402

VIEW_BASE_URL = "https://mypublicdashboard.ny.gov/t/OJRP_PUBLIC/views/NYSIndexCrimesbyAgency"
LANDING_PAGE_URL = "https://www.criminaljustice.ny.gov/index-crime-reporting-agency-data-dashboard"
EXPORT_YEARS = (2021, 2022, 2023, 2024, 2025)
NEW_YEAR = 2025
RAW_FILES = {
    "historic_table_2021_2025.csv": f"{VIEW_BASE_URL}/HistoricTable.csv?Year={','.join(map(str, EXPORT_YEARS))}",
    "data_notes.png": f"{VIEW_BASE_URL}/DataNotes.png",
    "landing_page.png": f"{VIEW_BASE_URL}/LandingPage.png",
}
EXISTING_EXTRACT = Path("NY-DCJS-2024") / "parsed" / "ny_dcjs_index_crimes_2021_2024.csv"
OUT_DIR = Path("NY-DCJS-2025")
OUT_EXTRACT_NAME = "ny_dcjs_index_crimes_2021_2025.csv"

NYC_COUNTIES = ("Bronx", "Kings", "New York", "Queens", "Richmond")
MEASURE_MAP = {
    "Index Total": "total_index_crimes",
    "Violent Total": "violent",
    "Property Total": "property",
    "Murder": "murder",
    "Rape": "forcible_rape",
    "Robbery": "robbery",
    "Agg. Assault": "aggravated_assault",
    "Burglary": "burglary",
    "Larceny": "larceny",
    "MV Theft": "motor_vehicle_theft",
}
MONTHS_MEASURE = "Months Reported (Agcy)"
# Tableau name -> data.ny.gov name (same agency; the 2021-2024 counts are identical).
AGENCY_RENAMES = {("Albany", "SUNY Police - Albany Plaza"): "SUNY Police - SUNY Plaza"}
COLUMNS = [
    "state_abbr", "year", "county", "agency", "row_type", "region", "months_reported",
    "total_index_crimes", "violent", "property", "murder", "forcible_rape", "robbery",
    "aggravated_assault", "burglary", "larceny", "motor_vehicle_theft",
]
COUNT_COLUMNS = COLUMNS[7:]
SORT_KEYS = ["year", "county", "agency"]


def fetch_raw(raw_dir: Path) -> dict[str, dict[str, str]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_path = raw_dir / "retrieval.json"
    records = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name, url in RAW_FILES.items():
        path = raw_dir / name
        if not path.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
            records[name] = {
                "url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        records[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(records, indent=2) + "\n")
    return records


def parse_historic_table(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.apply(lambda col: col.str.strip())
    keys = ["County", "Agency", "Year"]
    months = raw[raw["Measure Names"].eq(MONTHS_MEASURE)].set_index(keys)["Measure Values"]
    wide = raw[raw["Measure Names"].isin(MEASURE_MAP)].pivot(index=keys, columns="Measure Names", values="Measure Values")
    wide = wide.rename(columns=MEASURE_MAP)
    wide["months_reported"] = months.reindex(wide.index)
    wide = wide.reset_index().rename(columns={"County": "county", "Agency": "agency", "Year": "year"})
    for col in COUNT_COLUMNS + ["months_reported"]:
        wide[col] = pd.to_numeric(wide[col].str.replace(",", "", regex=False), errors="coerce")
    wide["year"] = wide["year"].astype(int)

    # Agencies with 0 months reported carry blank measures; data.ny.gov omits them.
    zero_months = wide["months_reported"].eq(0)
    if wide.loc[zero_months, COUNT_COLUMNS].notna().any().any():
        raise SystemExit("0-month rows with published counts")
    wide = wide[~zero_months].copy()
    if wide[COUNT_COLUMNS].isna().any().any():
        raise SystemExit("blank counts outside 0-month rows")

    is_nyc = wide["county"].isin(NYC_COUNTIES)
    is_total = wide["agency"].eq("Total") | (is_nyc & wide["agency"].str.endswith(" County NYPD"))
    if is_nyc.sum() != is_total[is_nyc].sum():
        raise SystemExit("unexpected NYC agency rows")
    wide["row_type"] = "agency"
    wide.loc[is_total, "row_type"] = "county_total"
    wide.loc[is_total, "agency"] = "County Total"
    wide.loc[wide["agency"].eq("County Total") & ~is_nyc, "months_reported"] = float("nan")
    for (county, old), new in AGENCY_RENAMES.items():
        wide.loc[wide["county"].eq(county) & wide["agency"].eq(old), "agency"] = new
    wide["region"] = is_nyc.map({True: "New York City", False: "Non-New York City"})
    wide["state_abbr"] = "NY"
    for col in COUNT_COLUMNS:
        wide[col] = wide[col].astype("int64")
    return wide[COLUMNS].sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)


def check_against_existing(parsed: pd.DataFrame, existing: pd.DataFrame) -> dict[str, object]:
    prior = parsed[parsed["year"].lt(NEW_YEAR)]
    merged = existing.merge(prior, on=SORT_KEYS, how="outer", suffixes=("_existing", "_export"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise SystemExit(f"2021-2024 row sets differ:\n{merged[merged['_merge'] != 'both'][SORT_KEYS + ['_merge']]}")
    for col in COUNT_COLUMNS + ["row_type", "region"]:
        if not merged[f"{col}_existing"].equals(merged[f"{col}_export"]):
            raise SystemExit(f"2021-2024 {col} differs from the existing extract")
    a, b = merged["months_reported_existing"], merged["months_reported_export"]
    months_diff = merged[~((a == b) | (a.isna() & b.isna()))]
    return {
        "rows_compared": int(len(merged)),
        "count_differences": 0,
        "months_reported_differences": int(len(months_diff)),
        "months_reported_differences_detail": months_diff[SORT_KEYS + ["months_reported_existing", "months_reported_export"]]
        .to_dict(orient="records"),
    }


def check_new_year(rows: pd.DataFrame) -> None:
    violent = rows[["murder", "forcible_rape", "robbery", "aggravated_assault"]].sum(axis=1)
    prop = rows[["burglary", "larceny", "motor_vehicle_theft"]].sum(axis=1)
    if not (rows["violent"].eq(violent) & rows["property"].eq(prop) & rows["total_index_crimes"].eq(violent + prop)).all():
        raise SystemExit("2025 subtotal mismatch")
    agency = rows[rows["row_type"].eq("agency")].groupby("county")[COUNT_COLUMNS].sum()
    totals = rows[rows["row_type"].eq("county_total") & rows["region"].ne("New York City")].set_index("county")[COUNT_COLUMNS]
    if not totals.equals(agency.reindex(totals.index).astype("int64")):
        raise SystemExit("2025 county totals differ from agency sums")


def main() -> None:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    data_dir = paths.data_dir
    raw_dir = data_dir / OUT_DIR / "raw"
    parsed_dir = data_dir / OUT_DIR / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    retrieval = fetch_raw(raw_dir)
    parsed = parse_historic_table(raw_dir / "historic_table_2021_2025.csv")
    existing = pd.read_csv(data_dir / EXISTING_EXTRACT)
    prior_check = check_against_existing(parsed, existing)
    new_rows = parsed[parsed["year"].eq(NEW_YEAR)].reset_index(drop=True)
    check_new_year(new_rows)

    out_path = parsed_dir / OUT_EXTRACT_NAME
    pd.concat([existing, new_rows], ignore_index=True).to_csv(out_path, index=False)
    existing_text = (data_dir / EXISTING_EXTRACT).read_text()
    if not out_path.read_text().startswith(existing_text):
        raise SystemExit("2021-2024 rows were not copied byte-for-byte")

    new_agency = new_rows[new_rows["row_type"].eq("agency")]
    prior_2024 = existing[existing["year"].eq(NEW_YEAR - 1) & existing["row_type"].eq("agency")]
    pairs_2025 = set(map(tuple, new_agency[["county", "agency"]].to_numpy()))
    pairs_2024 = set(map(tuple, prior_2024[["county", "agency"]].to_numpy()))
    match = build_ny_dcjs_ori_match_frame(paths, new_agency[["county", "agency"]])
    provenance = {
        "parsed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "DCJS Tableau workbook NYSIndexCrimesbyAgency, view HistoricTable (Index Crimes by Agency, 2010-2025)",
        "landing_page_url": LANDING_PAGE_URL,
        "raw_files": retrieval,
        "dashboard_dates_as_shown": "Visualization Updated: August 2026; Current Data Last Updated: July 2026; "
        "Historic Data Last Updated: November 2025 (raw/landing_page.png)",
        "definition_finding": "Data Notes (raw/data_notes.png) define the seven Index offenses as the FBI does; Rape is "
        "the FBI revised definition (DCJS from 2015), the same text as the data.ny.gov dictionary in "
        "data/NY-DCJS-2024/raw/data_dictionary.pdf. Notes flag NYPD (2023+) and NYSP (2024-2025) as NIBRS reporters.",
        "rows_2021_2024": "copied unchanged from data/NY-DCJS-2024/parsed/ny_dcjs_index_crimes_2021_2024.csv",
        "export_vs_existing_2021_2024": prior_check,
        "agency_renames": {f"{c} | {old}": new for (c, old), new in AGENCY_RENAMES.items()},
        "rows_2025": int(len(new_rows)),
        "agency_rows_2025": int(len(new_agency)),
        "county_total_rows_2025": int(new_rows["row_type"].eq("county_total").sum()),
        "agency_pairs_2025_in_2024_extract": len(pairs_2025 & pairs_2024),
        "agency_pairs_2025_not_in_2024": sorted(" | ".join(p) for p in pairs_2025 - pairs_2024),
        "agency_pairs_2024_absent_2025": sorted(" | ".join(p) for p in pairs_2024 - pairs_2025),
        "ori_match_2025": {
            "matched_pairs": int(match["n_candidate_ori"].eq(1).sum()),
            "ambiguous_pairs": int(match["n_candidate_ori"].gt(1).sum()),
            "unmatched_pairs": int(match["n_candidate_ori"].eq(0).sum()),
        },
        "output": str(out_path.relative_to(REPO_ROOT)),
    }
    (parsed_dir / "ny_dcjs_pull_provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k != "export_vs_existing_2021_2024"}, indent=2, default=str))


if __name__ == "__main__":
    main()
