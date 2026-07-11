from __future__ import annotations

import csv
import io
import re
import ssl
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from crimerisk.fbi_publications import norm_municipal_name
from crimerisk.paths import RepoPaths
from crimerisk.reference import _std_text
from crimerisk.source_provenance import STATE_PUBLICATION_SOURCE


FDLE_PUBLICATION_RAW_DATA_SOURCE = "fdle_fibrs_offense_detail_ytd"
MS_TOPS_PUBLICATION_RAW_DATA_SOURCE = "ms_tops_offense_detail_annual"
NY_DCJS_PUBLICATION_RAW_DATA_SOURCE = "ny_dcjs_index_crimes_annual"
# Per-lane verified target-year coverage. A lane asked for a target year outside its set
# contributes zero rows for that year, and write_v2_state_publication_inputs warns loudly:
# jurisdictions in that state then source from the FBI lane (the standing
# partial-coverage-with-FBI-fallback contract; STATE.md scopes exactly this for FL/NY 2025).
# Extend a set only after the state's publication for the new year is parsed and verified.
FDLE_FIBRS_SUPPORTED_YEARS = (2024,)
MS_TOPS_SUPPORTED_YEARS = (2024,)
# NY DCJS supported years are NY_DCJS_YEARS (defined with the lane constants below).
STATE_PUBLICATION_COLUMNS = [
    "ori9",
    "year",
    "offense",
    "count",
    "quarters_reported",
    "months_reported",
    "state_abbr",
    "source",
    "raw_data_source",
]
COUNTY_NAME_LOOKUP_PATH = Path("configs/county_names_2020.csv")
_COUNTY_SUFFIX_RE = re.compile(r"\b(COUNTY|PARISH|BOROUGH|CENSUS AREA|CITY AND BOROUGH|MUNICIPALITY)\b")
MS_TOPS_BASE_URL = "https://mscrimestats.dps.ms.gov/public/View/dispview.aspx"
MS_TOPS_JURISDICTION_SELECTION = "MemberSelection_[Jurisdiction].[Jurisdiction Hierarchy]"
MS_TOPS_REPORT_SPECS = {
    105: {
        "report_family": "violent",
        "offense_map": {
            "Murder": "murder",
            "Non-Consensual Sex Offenses": "rape",
            "Robbery": "robbery",
            "Aggravated Assault": "aggravated_assault",
        },
    },
    115: {
        "report_family": "property",
        "offense_map": {
            "Burglary": "burglary",
            "Larceny - Theft": "larceny",
            "Motor Vehicle Theft": "motor_vehicle_theft",
        },
    },
}

# --- NY DCJS "Index Crimes by County and Agency" (Socrata ca8h-8gjq) --------------
# Pulled and parsed by scripts/pull/pull_ny_dcjs_index_crimes.py; this module only
# consumes the deterministic parsed extract. DCJS's "Rape" column uses the FBI's
# revised (2013+) UCR definition (verified against the dataset's official data
# dictionary, cached at data/NY-DCJS-2024/raw/data_dictionary.pdf; DCJS adopted the
# expanded definition for statistics from 2015 on), which matches our pipeline's
# definition, so all seven index columns map directly.
NY_DCJS_YEARS = (2021, 2022, 2023, 2024)
NY_DCJS_PARSED_EXTRACT_RELPATH = Path("NY-DCJS-2024") / "parsed" / "ny_dcjs_index_crimes_2021_2024.csv"
NY_DCJS_OFFENSE_MAP = {
    "murder": "murder",
    "forcible_rape": "rape",
    "robbery": "robbery",
    "aggravated_assault": "aggravated_assault",
    "burglary": "burglary",
    "larceny": "larceny",
    "motor_vehicle_theft": "motor_vehicle_theft",
}
NY_DCJS_NYC_REGION_LABEL = "New York City"
NY_DCJS_COUNTY_TOTAL_LABEL = "County Total"

_NY_DCJS_SHERIFF_S_OFFICE_RE = re.compile(r"\bSHERIFF S\b")
_NY_DCJS_SHERIFF_OFFICE_RE = re.compile(r"\bSHERIFF OFFICE\b")
_NY_DCJS_TYPE_WORD_RE = re.compile(r"\b(CITY|TOWN|VILLAGE)\b")
_NY_DCJS_MTA_SUFFIX_RE = re.compile(r"^(.*)\s+COUNTY MTA$")
_NY_DCJS_STATE_POLICE_SUFFIX_RE = re.compile(r"^(.*)\s+COUNTY STATE POLICE$")
_NY_DCJS_AND_RE = re.compile(r"\bAND\b")

_OFFENSE_MAP = {
    "murder_and_non_negligent_manslaughter": "murder",
    "forcible_sex_offenses": "rape",
    "robbery": "robbery",
    "aggravated_assault": "aggravated_assault",
    "burglary": "burglary",
    "motor_vehicle_theft": "motor_vehicle_theft",
}
_DROP_TOKENS_RE = re.compile(
    r"\b(DEPARTMENT|DEPT|OFFICE|OFFICES|BUREAU|DIVISION|UNIT|AGENCY)\b"
)
_DROP_CONTEXT_TOKENS_RE = re.compile(r"\b(CAMPUS)\b")
_SHERIFF_VARIANTS_RE = re.compile(r"\bSHERIFFS\b")
_COUNTY_VARIANTS_RE = re.compile(r"\bCNTY\b")
_STATUS_PREFIX_RE = re.compile(
    r"^(RECERT(?:\s*-\s*|\s+)|NOT REPORTING CERTIFIED(?:\s*-\s*|\s+)|NOT REPORTING CERITFIED(?:\s*-\s*|\s+))",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")


def get_v2_state_publication_input_path(paths: RepoPaths) -> Path:
    return paths.state_dir / "modeling" / "inputs" / "state_publication_annual.parquet"


def _empty_state_publication_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STATE_PUBLICATION_COLUMNS)


def _normalize_publication_match_name(value: object, *, fdle_variants: bool = False) -> str | None:
    text = _std_text(value)
    if not text:
        return None
    text = _STATUS_PREFIX_RE.sub("", text)
    if fdle_variants:
        text = re.sub(r"\b(CITY|TOWN|VILLAGE)\s+OF\b", " ", text)
        text = re.sub(r"\bFLORIDA\s+DLE\s+FIELD\s+OFFICE\b", "FDLE", text)
        text = re.sub(r"\bFL(?:ORIDA)?\s+DEPT\s+(?:OF\s+)?LAW\s+ENFORCEMENT\b", "FDLE", text)
        text = re.sub(r"\bDLE\b", "FDLE", text)
        text = re.sub(r"\bSHP\b", "FHP", text)
        text = re.sub(r"\bPD\b", "POLICE", text)
        text = re.sub(r"\bPOLIC\b", "POLICE", text)
        text = re.sub(r"\bUNIV\b", "UNIVERSITY", text)
        text = re.sub(r"\bFLA\b", "FLORIDA", text)
        text = re.sub(r"\bST\s+PETE\b", "ST PETERSBURG", text)
        text = re.sub(r"\bRA\b", " ", text)
        text = re.sub(r"\bTBOC\b", " ", text)
        text = re.sub(r"\bPUBLIC\s+SCHOOLS?\b", "SCHOOL", text)
        text = re.sub(r"\bSCHOOLS\b", "SCHOOL", text)
        text = re.sub(r"\bCOUNTY\s+SCHOOL\b", "SCHOOL", text)
    text = _COUNTY_VARIANTS_RE.sub("COUNTY", text)
    text = re.sub(r"\bMHP\b", "MISSISSIPPI HIGHWAY PATROL", text)
    text = re.sub(r"\bMS\b", "MISSISSIPPI", text)
    text = re.sub(r"\bCC\b", "COMMUNITY COLLEGE", text)
    text = re.sub(r"\bJR\b", "JUNIOR", text)
    text = text.replace("COMMNTY", "COMMUNITY")
    text = _SHERIFF_VARIANTS_RE.sub("SHERIFF", text)
    text = _DROP_TOKENS_RE.sub(" ", text)
    text = _DROP_CONTEXT_TOKENS_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return None
    return text.lower()


def normalize_publication_match_name(value: object) -> str | None:
    return _normalize_publication_match_name(value, fdle_variants=False)


def normalize_fdle_publication_match_name(value: object) -> str | None:
    return _normalize_publication_match_name(value, fdle_variants=True)


def normalize_county_match_name(value: object) -> str | None:
    text = _std_text(value)
    if not text:
        return None
    text = _COUNTY_SUFFIX_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text.lower() or None


def _load_county_name_lookup(paths: RepoPaths) -> pd.DataFrame:
    path = paths.repo_root / COUNTY_NAME_LOOKUP_PATH
    if not path.exists():
        return pd.DataFrame(columns=["state_abbr", "county_fips", "county_match_name"])
    counties = pd.read_csv(path, dtype=str)
    counties["state_fips"] = counties["state_fips"].astype("string").str.zfill(2)
    counties["county_fips"] = counties["county_fips"].astype("string").str.zfill(5)
    state_abbr = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=["state_fips", "state_abbr"],
    ).drop_duplicates()
    state_abbr["state_fips"] = state_abbr["state_fips"].astype("string").str.zfill(2)
    counties = counties.merge(
        state_abbr,
        on="state_fips",
        how="left",
    )
    counties["county_match_name"] = counties["county_name"].map(normalize_county_match_name)
    return counties[["state_abbr", "county_fips", "county_match_name"]].dropna().drop_duplicates()


def _build_publication_alias_frame(
    paths: RepoPaths,
    agency_master: pd.DataFrame,
    *,
    normalizer=normalize_publication_match_name,
    include_fdle_alias_variants: bool = False,
) -> pd.DataFrame:
    alias_frames: list[pd.DataFrame] = []
    canonical_agencies = agency_master[agency_master["ori7"].astype("string").notna()].copy()
    alias_priorities = {
        "crosswalk_agency_name_std": 0,
        "agency_name_raw": 5,
        "agency_name_std": 10,
        "census_name_std": 20,
    }
    for alias_col, alias_priority in alias_priorities.items():
        if alias_col not in canonical_agencies.columns:
            continue
        part = canonical_agencies[["ori9", "state_abbr", "agency_type_norm", alias_col]].rename(
            columns={alias_col: "alias_raw"}
        )
        part["alias"] = part["alias_raw"].map(normalizer)
        part["alias_source"] = alias_col
        part["alias_priority"] = alias_priority
        alias_frames.append(part[["ori9", "state_abbr", "agency_type_norm", "alias", "alias_source", "alias_priority"]])
        if include_fdle_alias_variants:
            county_police = part.copy()
            county_police["alias"] = county_police["alias"].astype("string").str.replace(
                r"\bcounty police$",
                "police",
                regex=True,
            )
            county_police = county_police[county_police["alias"].notna() & county_police["alias"].ne(part["alias"])]
            if not county_police.empty:
                county_police["alias_source"] = f"{alias_col}_fdle_county_police_variant"
                county_police["alias_priority"] = alias_priority + 1
                alias_frames.append(
                    county_police[
                        ["ori9", "state_abbr", "agency_type_norm", "alias", "alias_source", "alias_priority"]
                    ]
                )
            campus_police = part.copy()
            campus_police["alias"] = campus_police["alias"].astype("string").str.replace(
                r"^(.*)\b(st petersburg|st augustine|tampa|jacksonville|panama city)\s+police$",
                r"\1police \2",
                regex=True,
            )
            campus_police = campus_police[campus_police["alias"].notna() & campus_police["alias"].ne(part["alias"])]
            if not campus_police.empty:
                campus_police["alias_source"] = f"{alias_col}_fdle_campus_police_variant"
                campus_police["alias_priority"] = alias_priority + 1
                alias_frames.append(
                    campus_police[
                        ["ori9", "state_abbr", "agency_type_norm", "alias", "alias_source", "alias_priority"]
                    ]
                )

    crosswalk = pd.read_parquet(paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet")
    ori_col = "ori" if "ori" in crosswalk.columns else "ori9"
    if ori_col != "ori9":
        crosswalk = crosswalk.rename(columns={ori_col: "ori9"})
    eligible = crosswalk[
        crosswalk["relationship_type"].astype("string").eq("exclusive")
        & pd.to_numeric(crosswalk["weight"], errors="coerce").gt(0.999)
    ][["jurisdiction_id", "ori9"]].drop_duplicates()
    eligible = eligible.merge(
        canonical_agencies[["ori9"]].drop_duplicates(),
        on="ori9",
        how="inner",
    )

    jurisdictions = pd.read_parquet(
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        columns=["jurisdiction_id", "state_abbr", "jurisdiction_type", "jurisdiction_name"],
    )
    juris_alias = eligible.merge(jurisdictions, on="jurisdiction_id", how="left").merge(
        agency_master[["ori9", "agency_type_norm"]].drop_duplicates(), on="ori9", how="left"
    )
    juris_alias["jurisdiction_name_std"] = juris_alias["jurisdiction_name"].map(_std_text)
    juris_alias["jurisdiction_name_base"] = juris_alias["jurisdiction_name_std"].map(norm_municipal_name)

    generated_aliases: list[tuple[str, str, str, str | None, str, int]] = []
    for row in juris_alias.itertuples(index=False):
        candidates = {
            normalizer(getattr(row, "jurisdiction_name_std", None)),
            normalizer(getattr(row, "jurisdiction_name_base", None)),
        }
        base = getattr(row, "jurisdiction_name_base", None) or getattr(row, "jurisdiction_name_std", None)
        if isinstance(base, str) and base.strip():
            if row.agency_type_norm == "local_police":
                candidates.add(normalizer(f"{base} police department"))
                candidates.add(normalizer(f"{base} police"))
                county_base = re.sub(r"\bCOUNTY$", "", base).strip()
                if county_base and county_base != base:
                    candidates.add(normalizer(f"{county_base} police department"))
                    candidates.add(normalizer(f"{county_base} police"))
            elif row.agency_type_norm == "sheriff":
                candidates.add(normalizer(f"{base} sheriff's office"))
                candidates.add(normalizer(f"{base} sheriff office"))
                candidates.add(normalizer(f"{base} sheriff"))
        for alias in candidates:
            generated_aliases.append((row.ori9, row.state_abbr, row.agency_type_norm, alias, "jurisdiction_generated", 50))

    generated = pd.DataFrame(
        generated_aliases,
        columns=["ori9", "state_abbr", "agency_type_norm", "alias", "alias_source", "alias_priority"],
    )
    alias_frames.append(generated)
    aliases = pd.concat(alias_frames, ignore_index=True)
    aliases = aliases[aliases["alias"].notna()].drop_duplicates().reset_index(drop=True)
    return aliases


def _fetch_state_publication_export(url: str) -> str:
    last_error: Exception | None = None
    for attempt, timeout in enumerate((20, 30, 45), start=1):
        try:
            request = Request(url, method="POST", data=b"")
            ssl_context = ssl._create_unverified_context()
            with urlopen(request, timeout=timeout, context=ssl_context) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            time.sleep(float(attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch state publication export: {url}")


def _parse_ms_tops_csv_export(text: str) -> dict[str, object] | None:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 8:
        return None
    title = rows[0][0].strip() if rows and rows[0] else ""
    jurisdiction_line = rows[3][0].strip() if len(rows) > 3 and rows[3] else ""
    measures_line = rows[4][0].strip() if len(rows) > 4 and rows[4] else ""
    header = rows[5] if len(rows) > 5 else []
    if "2024" not in header:
        return None
    data_rows = [row for row in rows[7:] if any(str(cell).strip() for cell in row)]
    jurisdiction_name = jurisdiction_line.split(":", 1)[1].strip() if ":" in jurisdiction_line else jurisdiction_line
    return {
        "title": title,
        "jurisdiction_name": jurisdiction_name,
        "measures": measures_line.split(":", 1)[1].strip() if ":" in measures_line else measures_line,
        "header": header,
        "data_rows": data_rows,
    }


def _numeric_csv_cell(value: object) -> float | None:
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _ms_tops_raw_path(paths: RepoPaths, *, report_id: int, member: str) -> Path:
    return paths.data_dir / "MS-TOPS-2024" / "raw" / f"report_{int(report_id)}" / f"{member}.csv"


def _ensure_ms_tops_raw_export(
    paths: RepoPaths,
    *,
    report_id: int,
    member: str,
    force_refresh: bool = False,
) -> Path:
    raw_path = _ms_tops_raw_path(paths, report_id=report_id, member=member)
    if raw_path.exists() and not force_refresh:
        return raw_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "ReportId": str(int(report_id)),
        MS_TOPS_JURISDICTION_SELECTION: member,
        "download": "yes",
        "DownloadFormat": "CSV",
        "CsvDataFormat": "CSV_MD",
    }
    url = f"{MS_TOPS_BASE_URL}?{urlencode(params, quote_via=quote)}"
    raw_path.write_text(_fetch_state_publication_export(url), encoding="utf-8")
    return raw_path


def _build_ms_tops_candidate_frame(paths: RepoPaths) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    agency_master = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=[
            "ori9",
            "ori7",
            "state_abbr",
            "agency_type_norm",
            "agency_name_std",
            "crosswalk_agency_name_std",
            "census_name_std",
        ],
    )
    candidates = agency_master[
        agency_master["state_abbr"].astype("string").eq("MS") & agency_master["ori7"].astype("string").notna()
    ][["ori9", "state_abbr", "agency_type_norm"]].drop_duplicates(subset=["ori9"])
    aliases = _build_publication_alias_frame(paths, agency_master)
    aliases = aliases[aliases["state_abbr"].astype("string").eq("MS")].copy()
    alias_map = aliases.groupby("ori9", dropna=False)["alias"].agg(lambda s: set(str(v) for v in s if pd.notna(v))).to_dict()
    return candidates.reset_index(drop=True), alias_map


def _parse_ms_tops_member_report(
    paths: RepoPaths,
    *,
    ori9: str,
    report_id: int,
    alias_map: dict[str, set[str]],
    year: int,
    force_refresh: bool = False,
) -> list[dict[str, object]]:
    raw_path = _ensure_ms_tops_raw_export(
        paths,
        report_id=report_id,
        member=ori9,
        force_refresh=force_refresh,
    )
    parsed = _parse_ms_tops_csv_export(raw_path.read_text(encoding="utf-8"))
    if not parsed:
        return []
    reported_name = normalize_publication_match_name(parsed.get("jurisdiction_name"))
    valid_aliases = alias_map.get(ori9, set())
    if reported_name not in valid_aliases:
        return []
    header = [str(cell).strip() for cell in parsed["header"]]
    if str(year) not in header:
        return []
    year_idx = header.index(str(year))
    spec = MS_TOPS_REPORT_SPECS[int(report_id)]
    rows: list[dict[str, object]] = []
    for row in parsed["data_rows"]:
        label = str(row[0]).strip() if row else ""
        offense = spec["offense_map"].get(label)
        if not offense:
            continue
        count = _numeric_csv_cell(row[year_idx] if year_idx < len(row) else None)
        if count is None:
            continue
        rows.append(
            {
                "state_abbr": "MS",
                "year": int(year),
                "ori9": ori9,
                "report_family": spec["report_family"],
                "report_id": int(report_id),
                "reported_jurisdiction_name": parsed["jurisdiction_name"],
                "offense_name": label,
                "offense": offense,
                "incident_count": float(count),
            }
        )
    return rows


def ensure_ms_tops_annual_extract(
    *,
    paths: RepoPaths,
    year: int = 2024,
    max_workers: int = 4,
    force_refresh: bool = False,
) -> Path:
    out_path = paths.data_dir / "MS-TOPS-2024" / "parsed" / f"ms_tops_{int(year)}_offense_extract.csv"

    candidates, alias_map = _build_ms_tops_candidate_frame(paths)
    tasks = [
        (row.ori9, report_id)
        for row in candidates.itertuples(index=False)
        for report_id in sorted(MS_TOPS_REPORT_SPECS)
    ]
    extracted_rows: list[dict[str, object]] = []
    failed_tasks: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(
                _parse_ms_tops_member_report,
                paths,
                ori9=ori9,
                report_id=report_id,
                alias_map=alias_map,
                year=int(year),
                force_refresh=force_refresh,
            ): (ori9, report_id)
            for ori9, report_id in tasks
        }
        for future in as_completed(futures):
            ori9, report_id = futures[future]
            try:
                extracted_rows.extend(future.result())
            except Exception as exc:  # pragma: no cover - network path
                failed_tasks.append((ori9, report_id))

    final_failures: list[tuple[str, int, str]] = []
    for ori9, report_id in failed_tasks:
        try:
            extracted_rows.extend(
                _parse_ms_tops_member_report(
                    paths,
                    ori9=ori9,
                    report_id=report_id,
                    alias_map=alias_map,
                    year=int(year),
                    force_refresh=force_refresh,
                )
            )
        except Exception as exc:  # pragma: no cover - network path
            final_failures.append((ori9, report_id, str(exc)))
    if final_failures:
        samples = ", ".join(f"{ori9}:{report_id}" for ori9, report_id, _ in final_failures[:8])
        raise RuntimeError(
            f"Mississippi TOPS extraction still failed for {len(final_failures)} tasks after retry: {samples}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    extract = pd.DataFrame(extracted_rows)
    if extract.empty:
        extract = pd.DataFrame(
            columns=[
                "state_abbr",
                "year",
                "ori9",
                "report_family",
                "report_id",
                "reported_jurisdiction_name",
                "offense_name",
                "offense",
                "incident_count",
            ]
        )
    else:
        extract = (
            extract.groupby(
                ["state_abbr", "year", "ori9", "report_family", "report_id", "reported_jurisdiction_name", "offense_name", "offense"],
                dropna=False,
            )["incident_count"]
            .sum()
            .reset_index()
            .sort_values(["ori9", "report_family", "offense"], kind="mergesort")
            .reset_index(drop=True)
        )
    extract.to_csv(out_path, index=False)
    return out_path


def normalize_ny_dcjs_alias(value: object) -> str | None:
    """Normalizer applied to agency_master alias columns for NY DCJS matching."""
    text = _std_text(value)
    if not text:
        return None
    text = _NY_DCJS_SHERIFF_S_OFFICE_RE.sub("SHERIFF", text)
    text = _NY_DCJS_SHERIFF_OFFICE_RE.sub("SHERIFF", text)
    text = _SHERIFF_VARIANTS_RE.sub("SHERIFF", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text.lower() or None


def ny_dcjs_agency_match_candidates(value: object) -> tuple[list[str], list[str]]:
    """Tiered match candidates for a DCJS agency name.

    Tier A keeps the jurisdiction-type word (City/Town/Village) from the DCJS name;
    Tier B drops it and is consulted only when Tier A resolves nothing, so
    same-name town/village pairs (e.g. Fishkill Town PD vs Fishkill Vg PD) stay
    distinct wherever agency_master distinguishes them.
    """
    text = _std_text(value)
    if not text:
        return [], []
    text = re.sub(r"^SUNY POLICE\s*-\s*", "SUNY POLICE ", text)
    text = re.sub(r"\bVG\b", "VILLAGE", text)
    text = _NY_DCJS_AND_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()

    tier_a: set[str] = set()
    tier_b: set[str] = set()

    mta_match = _NY_DCJS_MTA_SUFFIX_RE.match(text)
    if mta_match:
        tier_a.add(f"nycmta {mta_match.group(1)} county".lower())
    state_police_match = _NY_DCJS_STATE_POLICE_SUFFIX_RE.match(text)
    if state_police_match:
        tier_b.add(f"{state_police_match.group(1)} state police".lower())

    for pd_expansion in ("POLICE DEPARTMENT", "POLICE"):
        expanded = re.sub(r"\bPD\b", pd_expansion, text)
        full = _NY_DCJS_SHERIFF_S_OFFICE_RE.sub("SHERIFF", expanded)
        full = _NY_DCJS_SHERIFF_OFFICE_RE.sub("SHERIFF", full)
        full = _SHERIFF_VARIANTS_RE.sub("SHERIFF", full)
        full = _SPACE_RE.sub(" ", full).strip()
        if full:
            tier_a.add(full.lower())

        dropped = _NY_DCJS_TYPE_WORD_RE.sub(" ", expanded)
        dropped = _NY_DCJS_SHERIFF_S_OFFICE_RE.sub("SHERIFF", dropped)
        dropped = _NY_DCJS_SHERIFF_OFFICE_RE.sub("SHERIFF", dropped)
        dropped = _SHERIFF_VARIANTS_RE.sub("SHERIFF", dropped)
        dropped = _SPACE_RE.sub(" ", dropped).strip()
        if dropped:
            tier_b.add(dropped.lower())

    return sorted(tier_a), sorted(tier_b)


def build_ny_dcjs_ori_match_frame(paths: RepoPaths, agency_pairs: pd.DataFrame) -> pd.DataFrame:
    """Match distinct DCJS (county, agency) name pairs to agency_master ori9 values.

    Only exact alias hits count; when candidates exist at multiple alias priorities
    (crosswalk name > raw name > std name > census name > jurisdiction-generated),
    the best priority wins. A pair is matched only when exactly one ori9 resolves at
    that best priority; multi-ORI ties are reported as ambiguous (n_candidate_ori > 1)
    and never guessed. If two distinct DCJS names resolve to the same ori9, all of
    that ORI's pairs are demoted to ambiguous as a conflict guard.
    """
    agency_master = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=[
            "ori9",
            "ori7",
            "state_abbr",
            "agency_type_norm",
            "agency_name_raw",
            "agency_name_std",
            "crosswalk_agency_name_std",
            "census_name_std",
        ],
    )
    aliases = _build_publication_alias_frame(paths, agency_master, normalizer=normalize_ny_dcjs_alias)
    aliases_ny = aliases[aliases["state_abbr"].astype("string").eq("NY")]
    alias_lookup: dict[str, list[tuple[str, int]]] = {}
    for alias, ori9, priority in zip(aliases_ny["alias"], aliases_ny["ori9"], aliases_ny["alias_priority"]):
        alias_lookup.setdefault(alias, []).append((str(ori9), int(priority)))

    def _resolve(candidates: list[str]) -> set[str]:
        pairs: list[tuple[str, int]] = []
        for candidate in candidates:
            pairs.extend(alias_lookup.get(candidate, []))
        if not pairs:
            return set()
        best_priority = min(priority for _, priority in pairs)
        return {ori9 for ori9, priority in pairs if priority == best_priority}

    records: list[dict[str, object]] = []
    for row in agency_pairs[["county", "agency"]].drop_duplicates().itertuples(index=False):
        tier_a, tier_b = ny_dcjs_agency_match_candidates(row.agency)
        oris = _resolve(tier_a)
        tier_used: str | None = "A"
        if not oris:
            oris = _resolve(tier_b)
            tier_used = "B"
        if not oris:
            tier_used = None
        records.append(
            {
                "county": row.county,
                "agency": row.agency,
                "match_tier": tier_used,
                "n_candidate_ori": len(oris),
                "ori9": sorted(oris)[0] if len(oris) == 1 else None,
                "candidate_oris": ",".join(sorted(oris)) if oris else None,
            }
        )
    match_frame = pd.DataFrame.from_records(records)

    matched = match_frame["ori9"].notna()
    ori_pair_counts = match_frame.loc[matched].groupby("ori9")["agency"].transform("size")
    conflicted = matched.copy()
    conflicted.loc[matched] = ori_pair_counts.gt(1)
    if conflicted.any():
        match_frame.loc[conflicted, "n_candidate_ori"] = 2
        match_frame.loc[conflicted, "ori9"] = None
        match_frame.loc[conflicted, "match_tier"] = None
    return match_frame


def load_ny_dcjs_annual_ags_rows(
    *,
    paths: RepoPaths,
    year: int = 2024,
) -> pd.DataFrame:
    """NY DCJS lane: one state_publication_annual row per matched agency-offense-year.

    Coverage model (documented in docs/STATE.md source contract): DCJS is preferred
    per-agency wherever a confident (unique, exact-alias) ORI match exists; agencies
    that are unmatched or ambiguous simply produce no rows here and keep their
    FBI-side sourcing — the same partial-coverage-with-FBI-fallback shape as the
    FDLE lane. NYC borough rows are excluded entirely (the dataset publishes only
    borough "County Total" rows with no agency split; NYPD keeps its FBI/CIUS
    sourcing).

    Months semantics: DCJS's published counts are accepted as the official annual
    figures at face value — the same acceptance stance as FDLE published figures —
    so emitted rows carry months_reported = 12 and are never re-annualized
    downstream. DCJS itself publishes full-year-scale figures for partial direct
    reporters where it can (SCPD 2024: 3/12 months direct, published 16,710 ≈ its
    trailing full years); for small agencies whose partial-month rows are plain
    partial sums, face-value acceptance understates slightly, and the deep
    collapse-shaped cases stay covered by the existing masked-gap detector. DCJS's
    own months_reported is preserved in the parsed extract for diagnostics.
    """
    if int(year) not in NY_DCJS_YEARS:
        return _empty_state_publication_frame()
    source_path = paths.data_dir / NY_DCJS_PARSED_EXTRACT_RELPATH
    if not source_path.exists():
        return _empty_state_publication_frame()

    raw = pd.read_csv(source_path)
    raw = raw[
        pd.to_numeric(raw["year"], errors="coerce").eq(int(year))
        & raw["row_type"].astype("string").eq("agency")
        & ~raw["region"].astype("string").eq(NY_DCJS_NYC_REGION_LABEL)
        & ~raw["agency"].astype("string").str.strip().eq(NY_DCJS_COUNTY_TOTAL_LABEL)
    ].copy()
    if raw.empty:
        return _empty_state_publication_frame()

    match_frame = build_ny_dcjs_ori_match_frame(paths, raw[["county", "agency"]])
    matched_keys = match_frame[match_frame["n_candidate_ori"].eq(1)][["county", "agency", "ori9"]]
    raw = raw.merge(matched_keys, on=["county", "agency"], how="inner")
    if raw.empty:
        return _empty_state_publication_frame()

    long = raw.melt(
        id_vars=["ori9", "year"],
        value_vars=list(NY_DCJS_OFFENSE_MAP.keys()),
        var_name="dcjs_offense",
        value_name="count",
    )
    long["offense"] = long["dcjs_offense"].map(NY_DCJS_OFFENSE_MAP)
    long["count"] = pd.to_numeric(long["count"], errors="coerce")
    long = long[long["offense"].notna() & long["count"].notna()]
    if long.empty:
        return _empty_state_publication_frame()

    grouped = (
        long.groupby(["ori9", "year", "offense"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    grouped["state_abbr"] = "NY"
    grouped["quarters_reported"] = pd.NA
    grouped["months_reported"] = 12.0
    grouped["source"] = STATE_PUBLICATION_SOURCE
    grouped["raw_data_source"] = NY_DCJS_PUBLICATION_RAW_DATA_SOURCE
    return grouped.reindex(columns=STATE_PUBLICATION_COLUMNS)


def load_fdle_annual_ags_rows(
    *,
    paths: RepoPaths,
    year: int = 2024,
    min_quarters_reported: int = 1,
) -> pd.DataFrame:
    if int(year) not in FDLE_FIBRS_SUPPORTED_YEARS:
        return _empty_state_publication_frame()
    source_path = paths.data_dir / f"FDLE-FIBRS-{int(year)}" / "parsed" / f"fdle_fibrs_{int(year)}_offense_period_extract.csv"
    if not source_path.exists():
        return _empty_state_publication_frame()

    raw = pd.read_csv(source_path)
    raw = raw[
        raw["year"].astype("Int64").eq(int(year))
        & raw["period"].astype("string").eq("YTD")
    ].copy()
    if "months_reported" not in raw.columns:
        raw["months_reported"] = pd.NA
    raw["months_reported"] = pd.to_numeric(raw["months_reported"], errors="coerce")
    raw["quarters_reported"] = pd.to_numeric(raw["quarters_reported"], errors="coerce")
    reported_quarters = raw["quarters_reported"].where(
        raw["quarters_reported"].notna(),
        np.ceil(raw["months_reported"] / 3.0),
    )
    raw = raw[reported_quarters.ge(min_quarters_reported)].copy()
    if raw.empty:
        return _empty_state_publication_frame()

    raw["match_name"] = raw["agency_name"].map(normalize_fdle_publication_match_name)
    raw["county_match_name"] = raw["county"].map(normalize_county_match_name)
    raw["offense"] = raw["offense_slug"].map(_OFFENSE_MAP)
    larceny = raw["offense_slug"].astype("string").str.startswith("larceny_")
    raw.loc[larceny, "offense"] = "larceny"
    raw = raw[raw["offense"].notna()].copy()
    if raw.empty:
        return _empty_state_publication_frame()

    agency_master = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=[
            "ori9",
            "ori7",
            "state_abbr",
            "agency_name_std",
            "crosswalk_agency_name_std",
            "census_name_std",
            "agency_type_norm",
            "county_fips",
            "latest_srs_year",
            "latest_srs_part1_total",
            "latest_nibrs_year",
            "population_latest_nibrs",
        ],
    )
    agency_rank_fields = agency_master[
        [
            "ori9",
            "state_abbr",
            "county_fips",
            "latest_srs_year",
            "latest_srs_part1_total",
            "latest_nibrs_year",
            "population_latest_nibrs",
        ]
    ].drop_duplicates(subset=["ori9"])
    aliases = _build_publication_alias_frame(
        paths,
        agency_master,
        normalizer=normalize_fdle_publication_match_name,
        include_fdle_alias_variants=True,
    )
    aliases = aliases.merge(agency_rank_fields, on=["ori9", "state_abbr"], how="left")
    county_lookup = _load_county_name_lookup(paths)
    raw = raw.merge(county_lookup, on=["state_abbr", "county_match_name"], how="left", suffixes=("", "_source"))
    raw = raw.rename(columns={"county_fips": "source_county_fips"})

    source_key_cols = [
        "state_abbr",
        "source_sheet",
        "source_row_number",
        "county",
        "agency_name",
        "match_name",
    ]
    candidate_match = raw[source_key_cols + ["source_county_fips", "population_estimate"]].drop_duplicates().merge(
        aliases.rename(columns={"alias": "match_name"}),
        on=["state_abbr", "match_name"],
        how="left",
    )
    candidate_match = candidate_match[candidate_match["ori9"].notna()].copy()
    if candidate_match.empty:
        return _empty_state_publication_frame()

    source_county_code = candidate_match["source_county_fips"].astype("string").str[-3:]
    candidate_county_code = candidate_match["county_fips"].astype("string").str.zfill(3)
    candidate_match["candidate_county_match"] = (
        source_county_code.notna()
        & candidate_county_code.notna()
        & source_county_code.eq(candidate_county_code)
    )
    has_county_match = candidate_match.groupby(source_key_cols, dropna=False)["candidate_county_match"].transform("any")
    candidate_match = candidate_match[~has_county_match | candidate_match["candidate_county_match"]].copy()
    candidate_match["alias_priority"] = pd.to_numeric(candidate_match["alias_priority"], errors="coerce").fillna(99)
    candidate_match["nibrs_year_gap"] = (
        int(year) - pd.to_numeric(candidate_match["latest_nibrs_year"], errors="coerce")
    ).abs().fillna(999)
    candidate_match["srs_year_gap"] = (
        int(year) - pd.to_numeric(candidate_match["latest_srs_year"], errors="coerce")
    ).abs().fillna(999)
    source_population = pd.to_numeric(candidate_match["population_estimate"], errors="coerce")
    candidate_population = pd.to_numeric(candidate_match["population_latest_nibrs"], errors="coerce")
    population_log_gap = pd.Series(999.0, index=candidate_match.index)
    population_mask = source_population.gt(0) & candidate_population.gt(0)
    population_log_gap.loc[population_mask] = np.abs(
        np.log(candidate_population.loc[population_mask] / source_population.loc[population_mask])
    )
    candidate_match["population_log_gap"] = population_log_gap
    candidate_match["part1_sort"] = -pd.to_numeric(candidate_match["latest_srs_part1_total"], errors="coerce").fillna(0.0)
    rank_cols = [
        "alias_priority",
        "nibrs_year_gap",
        "srs_year_gap",
        "population_log_gap",
        "part1_sort",
    ]
    candidate_match = candidate_match.sort_values(
        source_key_cols + rank_cols + ["ori9"],
        kind="mergesort",
    )
    best = candidate_match.groupby(source_key_cols, dropna=False, as_index=False).first()
    top_scores = candidate_match.merge(
        best[source_key_cols + rank_cols],
        on=source_key_cols + rank_cols,
        how="inner",
    )
    best_score_counts = (
        top_scores.groupby(source_key_cols, dropna=False)["ori9"].nunique().rename("best_score_count").reset_index()
    )
    best = best.merge(best_score_counts, on=source_key_cols, how="left")
    best = best[best["best_score_count"].eq(1)].copy()
    if best.empty:
        return _empty_state_publication_frame()
    raw = raw.merge(
        best[source_key_cols + ["ori9", "agency_type_norm"]],
        on=source_key_cols,
        how="inner",
    )

    grouped = (
        raw.groupby(["ori9", "state_abbr", "year", "offense"], dropna=False)
        .agg(
            count=("incident_count", "sum"),
            quarters_reported=("quarters_reported", "max"),
            months_reported=("months_reported", "max"),
        )
        .reset_index()
    )
    grouped["source"] = STATE_PUBLICATION_SOURCE
    grouped["raw_data_source"] = FDLE_PUBLICATION_RAW_DATA_SOURCE
    return grouped.reindex(columns=STATE_PUBLICATION_COLUMNS)


def load_ms_tops_annual_ags_rows(
    *,
    paths: RepoPaths,
    year: int = 2024,
    force_refresh: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    if int(year) not in MS_TOPS_SUPPORTED_YEARS:
        return _empty_state_publication_frame()
    source_path = ensure_ms_tops_annual_extract(
        paths=paths,
        year=year,
        max_workers=max_workers,
        force_refresh=force_refresh,
    )
    if not source_path.exists():
        return _empty_state_publication_frame()

    raw = pd.read_csv(source_path)
    if raw.empty:
        return _empty_state_publication_frame()
    grouped = (
        raw.groupby(["ori9", "state_abbr", "year", "offense"], dropna=False)["incident_count"]
        .sum()
        .rename("count")
        .reset_index()
    )
    grouped["quarters_reported"] = pd.NA
    grouped["months_reported"] = pd.NA
    grouped["source"] = STATE_PUBLICATION_SOURCE
    grouped["raw_data_source"] = MS_TOPS_PUBLICATION_RAW_DATA_SOURCE
    return grouped.reindex(columns=STATE_PUBLICATION_COLUMNS)


def _build_state_publication_annual_ags_rows(
    *,
    paths: RepoPaths,
    year: int = 2024,
    force_refresh: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    frames = [
        load_fdle_annual_ags_rows(paths=paths, year=year, min_quarters_reported=1),
        load_ms_tops_annual_ags_rows(
            paths=paths,
            year=year,
            force_refresh=force_refresh,
            max_workers=max_workers,
        ),
        load_ny_dcjs_annual_ags_rows(paths=paths, year=year),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_state_publication_frame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out[out["year"].eq(int(year))].copy()
    if out.empty:
        return _empty_state_publication_frame()
    out["year"] = out["year"].astype(int)
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["quarters_reported"] = pd.to_numeric(out["quarters_reported"], errors="coerce")
    out["months_reported"] = pd.to_numeric(out.get("months_reported"), errors="coerce")
    out["state_abbr"] = out["state_abbr"].astype("string").str.upper()
    out["offense"] = out["offense"].astype("string")
    out["ori9"] = out["ori9"].astype("string")
    out["raw_data_source"] = out["raw_data_source"].astype("string")
    return out.reindex(columns=STATE_PUBLICATION_COLUMNS).drop_duplicates(
        subset=["ori9", "year", "offense", "source", "raw_data_source"],
        keep="first",
    ).reset_index(drop=True)


def write_v2_state_publication_inputs(
    *,
    paths: RepoPaths,
    out_path: Path | None = None,
    year: int = 2024,
    force_refresh: bool = False,
    max_workers: int = 4,
) -> dict[str, object]:
    """Write the canonical state_publication_annual parquet.

    Multi-year contract: the parquet contains, per lane, every year that lane
    publishes and the pipeline consumes — FDLE and MS TOPS contribute the target
    year only; the NY DCJS lane contributes its full 2021→target span so trend
    fills and reference years for remaining NY gaps anchor on DCJS-quality
    history. Target-year-only consumers (reporting regimes, controls) keep
    filtering to their year; observations load the full span.

    Lane coverage: each lane has an enumerated supported-year set. A lane that does
    not cover the target year contributes zero target-year rows and triggers a loud
    warning here; the affected state's jurisdictions then source from the FBI lane
    (the standing partial-coverage fallback — STATE.md scopes exactly this for
    FL/NY at target year 2025). This writer never silently reuses a prior year.
    """
    lane_supported_years = {
        "FDLE FIBRS (FL)": tuple(FDLE_FIBRS_SUPPORTED_YEARS),
        "MS TOPS (MS)": tuple(MS_TOPS_SUPPORTED_YEARS),
        "NY DCJS (NY)": tuple(NY_DCJS_YEARS),
    }
    lanes_missing_target_year = sorted(
        lane for lane, supported in lane_supported_years.items() if int(year) not in supported
    )
    for lane in lanes_missing_target_year:
        warnings.warn(
            f"State-publication lane {lane} has no verified publication for target year {int(year)} "
            f"(supported years: {lane_supported_years[lane]}). The lane contributes no rows for "
            f"{int(year)}; that state's jurisdiction totals fall back to the FBI lane. Extend the "
            "lane's supported-year set only after the new publication is parsed and verified.",
            RuntimeWarning,
            stacklevel=2,
        )
    out_path = out_path or get_v2_state_publication_input_path(paths)
    years = sorted(set(int(y) for y in NY_DCJS_YEARS if int(y) <= int(year)) | {int(year)})
    frames = [
        _build_state_publication_annual_ags_rows(
            paths=paths,
            year=build_year,
            force_refresh=force_refresh,
            max_workers=max_workers,
        )
        for build_year in years
    ]
    frames = [frame for frame in frames if not frame.empty]
    rows = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else _empty_state_publication_frame()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_path, index=False)
    target_rows = rows[rows["year"].eq(int(year))] if not rows.empty else rows
    return {
        "rows": int(len(rows)),
        "agencies": int(rows["ori9"].nunique()) if not rows.empty else 0,
        "states": int(rows["state_abbr"].nunique()) if not rows.empty else 0,
        "years": years,
        "year": int(year),
        "target_year_rows": int(len(target_rows)),
        "lanes_missing_target_year": lanes_missing_target_year,
        "out_path": str(out_path),
    }


def load_state_publication_annual_ags_rows(
    *,
    paths: RepoPaths,
    year: int = 2024,
    year_start: int | None = None,
    year_end: int | None = None,
    source_path: Path | None = None,
    require_source_exists: bool = False,
) -> pd.DataFrame:
    """Load canonical state-publication rows.

    Default is single-year (`year`) for target-year consumers; pass
    `year_start`/`year_end` (both required together) to load a span — used by the
    observations build so multi-year lanes (NY DCJS 2021-2024) feed the panel.
    """
    if (year_start is None) != (year_end is None):
        raise ValueError("Pass both year_start and year_end, or neither.")
    source_path = source_path or get_v2_state_publication_input_path(paths)
    if not source_path.exists():
        if require_source_exists:
            raise FileNotFoundError(
                "Missing canonical state-publication input surface: "
                f"{source_path}. Run `python main.py build-state-publication-inputs` first."
            )
        return _empty_state_publication_frame()
    rows = pd.read_parquet(source_path).copy()
    if rows.empty:
        return _empty_state_publication_frame()
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce").astype("Int64")
    if year_start is not None:
        rows = rows[rows["year"].between(int(year_start), int(year_end))].copy()
    else:
        rows = rows[rows["year"].eq(int(year))].copy()
    if rows.empty:
        return _empty_state_publication_frame()
    rows["year"] = rows["year"].astype(int)
    rows["count"] = pd.to_numeric(rows["count"], errors="coerce")
    rows["quarters_reported"] = pd.to_numeric(rows.get("quarters_reported"), errors="coerce")
    rows["months_reported"] = pd.to_numeric(rows.get("months_reported"), errors="coerce")
    rows["state_abbr"] = rows["state_abbr"].astype("string").str.upper()
    rows["offense"] = rows["offense"].astype("string")
    rows["ori9"] = rows["ori9"].astype("string")
    rows["raw_data_source"] = rows["raw_data_source"].astype("string")
    return rows.reindex(columns=STATE_PUBLICATION_COLUMNS).drop_duplicates(
        subset=["ori9", "year", "offense", "source", "raw_data_source"],
        keep="first",
    ).reset_index(drop=True)
