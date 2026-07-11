"""Pull, parse, and ORI-match the NY DCJS "Index Crimes by County and Agency" dataset.

CP1 of the NY DCJS onboarding brief (scratchpad/briefs/brief_X_ny_dcjs.md): pull the
official state-publication source (Socrata `ca8h-8gjq` on data.ny.gov) that DCJS
compiles from the ~500 NY police/sheriff departments before forwarding to the FBI,
parse it deterministically, verify the published county totals equal the sum of
their agency rows, map DCJS's index-crime columns to our seven canonical offenses,
and match DCJS agency names to `state/reference/agency_master.parquet` ORIs.

This mirrors the FDLE / MS TOPS `state_publication_annual` lane in
`src/crimerisk/state_publications.py` (raw cache under `data/<STATE>-<SOURCE>/raw/`,
deterministic parse under `.../parsed/`). The offense map and the DCJS-name ->
agency_master ORI matcher are canonical in `crimerisk.state_publications`
(`NY_DCJS_OFFENSE_MAP`, `build_ny_dcjs_ori_match_frame`) where the NY lane loader
(`load_ny_dcjs_annual_ags_rows`) also applies them at build time; this script imports
them so pull-time audit CSVs and the canonical lane can never disagree.

Rape-definition finding (must be established before offense mapping, per the brief):
the official DCJS data dictionary (fetched from the dataset's Socrata attachment,
cached under `data/NY-DCJS-2024/raw/`) defines "Rape" as "Penetration, no matter how
slight, of the vagina or anus with any body part or object, or oral penetration by a
sex organ of another person, without the consent of the victim." That is the FBI's
REVISED (2013+) UCR rape definition (NIBRS 11A rape + 11B sodomy + 11C sexual assault
with an object; see docs/FBI-DATA-GUIDE.md), the same definition our pipeline already
uses. It is NOT ambiguous, so `forcible_rape` maps directly to our `rape` offense
along with the other six -- no offense is left unmapped.

Outputs (all under `data/NY-DCJS-2024/`):
  raw/dataset_metadata.json                  Socrata view metadata (refresh date, row counts)
  raw/data_dictionary.pdf                     DCJS's official column data dictionary
  raw/index_crimes_<year>.json                One cached raw page-concatenated pull per year
  parsed/ny_dcjs_index_crimes_2021_2024.csv   Tidy one-row-per-(county,agency,year) extract
  parsed/ny_dcjs_county_total_consistency.csv Per county-year: published total vs agency sum
  parsed/ny_dcjs_ori_match.csv                Agency rows matched to ori9 (+ match tier)
  parsed/ny_dcjs_ori_unmatched.csv            Unmatched (county, agency) with row counts
  parsed/ny_dcjs_pull_provenance.json         Row counts, refresh date, match-rate summary
  parsed/ny_dcjs_sanity_top20_2024.csv        Top-20 2024 NY agencies: DCJS vs FBI-side (optional)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths  # noqa: E402
from crimerisk.state_publications import (  # noqa: E402
    NY_DCJS_OFFENSE_MAP,
    build_ny_dcjs_ori_match_frame,
)

DATASET_ID = "ca8h-8gjq"
SOCRATA_RESOURCE_URL = f"https://data.ny.gov/resource/{DATASET_ID}.json"
DATASET_METADATA_URL = f"https://data.ny.gov/api/views/{DATASET_ID}.json"
DATA_DICTIONARY_URL = (
    f"https://data.ny.gov/api/views/{DATASET_ID}/files/"
    "ad4d5543-71e4-4bbd-8c02-9cda59762dc1"
    "?download=true&filename=DCJS_IndexCrimesByCountyAndAgency_DataDictionary.pdf"
)
LANDING_PAGE_URL = "https://data.ny.gov/Public-Safety/Index-Crimes-by-County-and-Agency-Beginning-1990/ca8h-8gjq"
YEARS = (2021, 2022, 2023, 2024)
PAGE_LIMIT = 5000
STATE_ABBR = "NY"

OUT_DIR_NAME = "NY-DCJS-2024"
COUNTY_TOTAL_LABEL = "County Total"

# DCJS "Index Crimes by County and Agency" columns -> our seven canonical offenses.
# Canonical mapping lives in crimerisk.state_publications.NY_DCJS_OFFENSE_MAP; see the
# module docstring for the rape-definition finding that justifies mapping
# forcible_rape -> rape directly (confirmed non-ambiguous: DCJS uses the FBI's
# revised/current UCR rape definition, same as our pipeline).
OFFENSE_MAP = NY_DCJS_OFFENSE_MAP
RAW_OFFENSE_COLUMNS = list(OFFENSE_MAP.keys())
SUBTOTAL_COLUMNS = ["total_index_crimes", "violent", "property"]


# ---------------------------------------------------------------------------
# Fetch (raw cache)
# ---------------------------------------------------------------------------


def _get_json(url: str, *, params: dict | None = None, timeout: int = 60) -> object:
    last_error: Exception | None = None
    for attempt, sleep_sec in enumerate((0.0, 2.0, 5.0), start=1):
        if sleep_sec:
            time.sleep(sleep_sec)
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            print(f"  fetch attempt {attempt} failed for {url}: {exc}")
    raise RuntimeError(f"Failed to fetch {url} after retries") from last_error


def _get_bytes(url: str, *, timeout: int = 60) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def fetch_dataset_metadata(raw_dir: Path, *, force_refresh: bool = False) -> dict:
    path = raw_dir / "dataset_metadata.json"
    if path.exists() and not force_refresh:
        return json.loads(path.read_text())
    metadata = _get_json(DATASET_METADATA_URL)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def fetch_data_dictionary(raw_dir: Path, *, force_refresh: bool = False) -> Path:
    path = raw_dir / "data_dictionary.pdf"
    if path.exists() and not force_refresh:
        return path
    path.write_bytes(_get_bytes(DATA_DICTIONARY_URL))
    return path


def fetch_year_rows(raw_dir: Path, *, year: int, force_refresh: bool = False) -> list[dict]:
    """Fetch all rows for one year via $limit/$offset pagination and cache the raw pages."""
    path = raw_dir / f"index_crimes_{year}.json"
    if path.exists() and not force_refresh:
        return json.loads(path.read_text())

    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "$limit": PAGE_LIMIT,
            "$offset": offset,
            "$where": f"year='{year}'",
            "$order": "county,agency",
        }
        page = _get_json(SOCRATA_RESOURCE_URL, params=params)
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected Socrata payload shape for year={year} offset={offset}")
        rows.extend(page)
        if len(page) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT

    path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    return rows


# ---------------------------------------------------------------------------
# Parse (deterministic long extract)
# ---------------------------------------------------------------------------


def build_tidy_extract(years_rows: dict[int, list[dict]]) -> pd.DataFrame:
    frames = []
    for year, rows in years_rows.items():
        frame = pd.DataFrame(rows)
        frame["year"] = int(year)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)

    numeric_cols = RAW_OFFENSE_COLUMNS + SUBTOTAL_COLUMNS + ["months_reported"]
    for col in numeric_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["row_type"] = raw["agency"].map(
        lambda a: "county_total" if str(a).strip() == COUNTY_TOTAL_LABEL else "agency"
    )
    raw["state_abbr"] = STATE_ABBR

    ordered_cols = [
        "state_abbr",
        "year",
        "county",
        "agency",
        "row_type",
        "region",
        "months_reported",
        "total_index_crimes",
        "violent",
        "property",
    ] + RAW_OFFENSE_COLUMNS
    ordered_cols = [c for c in ordered_cols if c in raw.columns]
    return raw[ordered_cols].sort_values(["year", "county", "agency"], kind="mergesort").reset_index(drop=True)


def check_county_total_consistency(tidy: pd.DataFrame) -> pd.DataFrame:
    """Verify published 'County Total' rows equal the sum of that county-year's agency rows."""
    agency_rows = tidy[tidy["row_type"] == "agency"]
    county_rows = tidy[tidy["row_type"] == "county_total"]

    sum_cols = ["total_index_crimes"] + RAW_OFFENSE_COLUMNS
    agency_sums = agency_rows.groupby(["year", "county"], dropna=False)[sum_cols].sum().reset_index()
    agency_sums = agency_sums.rename(columns={c: f"{c}_agency_sum" for c in sum_cols})

    published = county_rows[["year", "county"] + sum_cols].rename(
        columns={c: f"{c}_published" for c in sum_cols}
    )

    check = published.merge(agency_sums, on=["year", "county"], how="outer", indicator=True)
    for col in sum_cols:
        check[f"{col}_diff"] = check[f"{col}_published"] - check[f"{col}_agency_sum"]
    # NYC's 5 boroughs (region == "New York City") publish only a "County Total" row
    # per year with NO agency-level breakout in this dataset (NYPD is not split into
    # per-borough agency rows here) -- that is a structural absence, not a consistency
    # violation, so it is flagged separately rather than counted as a mismatch.
    check["no_agency_breakdown_published"] = check["_merge"].eq("left_only")
    check["all_match"] = (
        (check[[f"{c}_diff" for c in sum_cols]].fillna(0).abs().sum(axis=1) == 0) & (check["_merge"] == "both")
    ) | check["no_agency_breakdown_published"]
    return check.sort_values(["year", "county"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ORI matching
#
# The matcher itself (DCJS-specific normalizer + tiered candidate scheme on top of
# the shared `_build_publication_alias_frame` machinery) is canonical in
# crimerisk.state_publications (`build_ny_dcjs_ori_match_frame`), where the NY lane
# also uses it at load time; this script just applies it to produce the audit CSVs.
# A pair is matched only when exactly one ori9 resolves at the best alias priority;
# multi-ORI ties are reported as ambiguous, never guessed.
# ---------------------------------------------------------------------------


def match_agencies_to_ori(tidy: pd.DataFrame, paths: RepoPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (matched_rows, unmatched_summary, ambiguous_summary)."""
    agency_rows = tidy[tidy["row_type"] == "agency"].copy()
    match_key = build_ny_dcjs_ori_match_frame(paths, agency_rows[["county", "agency"]])

    matched_keys = match_key[match_key["n_candidate_ori"] == 1][["county", "agency", "ori9", "match_tier"]]
    matched_rows = agency_rows.merge(matched_keys, on=["county", "agency"], how="inner")

    unmatched_keys = match_key[match_key["n_candidate_ori"] == 0][["county", "agency"]]
    row_counts = agency_rows.groupby(["county", "agency"]).size().rename("row_count").reset_index()
    unmatched_summary = unmatched_keys.merge(row_counts, on=["county", "agency"], how="left").sort_values(
        ["county", "agency"], kind="mergesort"
    ).reset_index(drop=True)

    ambiguous_keys = match_key[match_key["n_candidate_ori"] > 1][
        ["county", "agency", "n_candidate_ori", "candidate_oris", "match_tier"]
    ]
    ambiguous_summary = ambiguous_keys.merge(row_counts, on=["county", "agency"], how="left").sort_values(
        ["county", "agency"], kind="mergesort"
    ).reset_index(drop=True)

    return matched_rows, unmatched_summary, ambiguous_summary


# ---------------------------------------------------------------------------
# Sanity table: top-20 2024 NY agencies, DCJS vs FBI-side observed/adjusted
# ---------------------------------------------------------------------------


def build_sanity_table(matched_rows: pd.DataFrame, paths: RepoPaths, *, year: int = 2024) -> pd.DataFrame:
    dcjs_year = matched_rows[matched_rows["year"] == year][["ori9", "county", "agency", "total_index_crimes"]]
    dcjs_year = dcjs_year.rename(columns={"total_index_crimes": "dcjs_total"})
    top20 = dcjs_year.sort_values("dcjs_total", ascending=False, kind="mergesort").head(20).copy()

    # `agency_year_observations.parquet` carries the SAME count under multiple
    # `source` labels (e.g. srs_return_a_annual / nibrs_srs_equivalent_annual /
    # cius_publication_annual all repeating the identical figure for one
    # agency-year-offense) -- summing across sources would triple-count, so pick
    # one row per (ori9, year, offense) using a fixed source preference order.
    _SOURCE_PREFERENCE = [
        "srs_return_a_annual",
        "nibrs_srs_equivalent_annual",
        "cius_publication_annual",
        "local_publication_annual",
        "state_publication_annual",
    ]
    observations_path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    fbi_observed_total = pd.Series(dtype="float64")
    if observations_path.exists():
        obs = pd.read_parquet(observations_path, columns=["ori9", "year", "offense", "count", "source"])
        obs_year = obs[
            obs["ori9"].isin(top20["ori9"]) & obs["year"].eq(year) & obs["offense"].isin(RAW_OFFENSE_COLUMNS_MAPPED)
        ].copy()
        source_rank = {name: i for i, name in enumerate(_SOURCE_PREFERENCE)}
        obs_year["source_rank"] = obs_year["source"].map(source_rank).fillna(len(_SOURCE_PREFERENCE))
        obs_year = obs_year.sort_values(["ori9", "year", "offense", "source_rank"], kind="mergesort")
        obs_year = obs_year.drop_duplicates(subset=["ori9", "year", "offense"], keep="first")
        fbi_observed_total = obs_year.groupby("ori9")["count"].sum()
    top20["fbi_observed_total"] = top20["ori9"].map(fbi_observed_total).fillna(0)

    fbi_adjusted_total = pd.Series(dtype="float64")
    try:
        from crimerisk.trend_fills import build_agency_allocation_target_estimates

        estimates = build_agency_allocation_target_estimates(paths=paths, year=year)
        estimates_ny = estimates[estimates["ori9"].isin(top20["ori9"])]
        fbi_adjusted_total = estimates_ny.groupby("ori9")["estimated_count"].sum()
    except Exception as exc:  # pragma: no cover - defensive; sanity table is best-effort
        print(f"  (sanity table) could not compute FBI-side adjusted totals: {exc}")
    top20["fbi_adjusted_total"] = top20["ori9"].map(fbi_adjusted_total).fillna(0)

    top20["adjusted_vs_dcjs_ratio"] = (top20["fbi_adjusted_total"] / top20["dcjs_total"]).round(3)
    top20["observed_vs_dcjs_ratio"] = (top20["fbi_observed_total"] / top20["dcjs_total"]).round(3)
    return top20.reset_index(drop=True)


RAW_OFFENSE_COLUMNS_MAPPED = list(OFFENSE_MAP.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="*", type=int, default=list(YEARS))
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / OUT_DIR_NAME)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-sanity-table", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    raw_dir = out_dir / "raw"
    parsed_dir = out_dir / "parsed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    paths = RepoPaths.from_repo_root(REPO_ROOT)

    print("Fetching dataset metadata + data dictionary ...")
    metadata = fetch_dataset_metadata(raw_dir, force_refresh=args.force_refresh)
    fetch_data_dictionary(raw_dir, force_refresh=args.force_refresh)

    years_rows: dict[int, list[dict]] = {}
    for year in args.years:
        print(f"Fetching year {year} ...")
        years_rows[year] = fetch_year_rows(raw_dir, year=year, force_refresh=args.force_refresh)
        print(f"  {len(years_rows[year]):,} rows")

    print("Parsing tidy extract ...")
    tidy = build_tidy_extract(years_rows)
    tidy_path = parsed_dir / "ny_dcjs_index_crimes_2021_2024.csv"
    tidy.to_csv(tidy_path, index=False)
    print(f"  wrote {tidy_path} ({len(tidy):,} rows)")

    print("Checking county-total = sum(agency rows) consistency ...")
    consistency = check_county_total_consistency(tidy)
    consistency_path = parsed_dir / "ny_dcjs_county_total_consistency.csv"
    consistency.to_csv(consistency_path, index=False)
    no_breakdown = consistency[consistency["no_agency_breakdown_published"]]
    violations = consistency[~consistency["all_match"]]
    print(
        f"  {len(consistency)} county-years checked; {len(no_breakdown)} have no agency-level "
        f"breakdown published (NYC boroughs, expected); {len(violations)} genuine violations"
    )
    if len(violations):
        print(violations.to_string())

    print("Matching agencies to agency_master ORIs ...")
    matched_rows, unmatched_summary, ambiguous_summary = match_agencies_to_ori(tidy, paths)
    matched_path = parsed_dir / "ny_dcjs_ori_match.csv"
    unmatched_path = parsed_dir / "ny_dcjs_ori_unmatched.csv"
    ambiguous_path = parsed_dir / "ny_dcjs_ori_ambiguous.csv"
    matched_rows.to_csv(matched_path, index=False)
    unmatched_summary.to_csv(unmatched_path, index=False)
    ambiguous_summary.to_csv(ambiguous_path, index=False)

    total_agency_rows = int((tidy["row_type"] == "agency").sum())
    matched_row_count = int(len(matched_rows))
    unique_pairs = tidy[tidy["row_type"] == "agency"][["county", "agency"]].drop_duplicates()
    matched_pairs = matched_rows[["county", "agency"]].drop_duplicates()
    print(
        f"  matched {len(matched_pairs)}/{len(unique_pairs)} distinct (county, agency) pairs "
        f"({len(matched_pairs) / len(unique_pairs):.1%}); "
        f"{matched_row_count}/{total_agency_rows} agency-rows across {len(args.years)} years"
    )
    print(f"  ambiguous: {len(ambiguous_summary)} pairs; unmatched: {len(unmatched_summary)} pairs")

    sanity_path = None
    if not args.skip_sanity_table:
        print("Building sanity table (top-20 2024 NY agencies, DCJS vs FBI-side) ...")
        try:
            sanity = build_sanity_table(matched_rows, paths, year=2024)
            sanity_path = parsed_dir / "ny_dcjs_sanity_top20_2024.csv"
            sanity.to_csv(sanity_path, index=False)
            print(sanity.to_string())
        except Exception as exc:  # pragma: no cover - best-effort diagnostic output
            print(f"  sanity table failed: {exc}")

    provenance = {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "landing_page_url": LANDING_PAGE_URL,
        "dataset_publication_date_epoch": metadata.get("publicationDate"),
        "dataset_rows_updated_at_epoch": metadata.get("rowsUpdatedAt"),
        "dataset_view_last_modified_epoch": metadata.get("viewLastModified"),
        "years_pulled": list(args.years),
        "row_counts_by_year": {str(y): len(rows) for y, rows in years_rows.items()},
        "total_rows": int(sum(len(rows) for rows in years_rows.values())),
        "total_agency_rows": total_agency_rows,
        "total_county_total_rows": int((tidy["row_type"] == "county_total").sum()),
        "county_total_consistency_violations": int(len(violations)),
        "ori_match": {
            "distinct_county_agency_pairs": int(len(unique_pairs)),
            "matched_pairs": int(len(matched_pairs)),
            "match_rate": round(len(matched_pairs) / len(unique_pairs), 4),
            "ambiguous_pairs": int(len(ambiguous_summary)),
            "unmatched_pairs": int(len(unmatched_summary)),
            "matched_agency_rows": matched_row_count,
            "total_agency_rows": total_agency_rows,
        },
        "offense_map": OFFENSE_MAP,
        "rape_definition_finding": (
            "DCJS data dictionary (data/NY-DCJS-2024/raw/data_dictionary.pdf) defines Rape as "
            "'Penetration, no matter how slight, of the vagina or anus with any body part or "
            "object, or oral penetration by a sex organ of another person, without the consent "
            "of the victim' -- the FBI's revised (2013+) UCR rape definition, matching the "
            "definition our pipeline already uses. Not ambiguous; mapped directly."
        ),
        "outputs": {
            "tidy_extract": str(tidy_path),
            "county_total_consistency": str(consistency_path),
            "ori_matched": str(matched_path),
            "ori_unmatched": str(unmatched_path),
            "ori_ambiguous": str(ambiguous_path),
            "sanity_table": str(sanity_path) if sanity_path else None,
        },
    }
    provenance_path = parsed_dir / "ny_dcjs_pull_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=False))
    print(f"Wrote provenance: {provenance_path}")


if __name__ == "__main__":
    main()
