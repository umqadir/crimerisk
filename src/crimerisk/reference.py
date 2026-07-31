from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.crime.nibrs import ensure_nibrs_batch_header_parquet
from crimerisk.crime.srs import ensure_srs_year_parquet
from crimerisk.paths import RepoPaths
from crimerisk.utils.hashing import sha256_file


_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9 ]+")

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
    "CZ": "57",
    "GU": "66",
    "PR": "72",
    "VI": "78",
    "NB": "31",
    "GM": "66",
}


# The external raw-data contract lives in crimerisk.required_inputs; the input
# manifest is generated from that single source of truth.


# --- county-FIPS canonicalization -------------------------------------------
# The SRS agency header carries the county code the agency was coded with when its
# ORI was issued, so a handful of ORIs still point at county FIPS the Census retired
# (and NIBRS-only agencies carry no county code at all). Both defects land in the same
# place: allocation._build_county_remainder_group_targets anchors remainder-lane
# agencies to `state_fips + county_fips`, so a retired or missing code silently fails
# the county join and the agency's territory drops back into the state residual pool.
RETIRED_COUNTY_GEOID_REMAP: dict[str, str] = {
    # Shannon County SD renamed Oglala Lakota County (2015).
    "46113": "46102",
    # Valdez-Cordova Census Area AK split into Chugach + Copper River (2019); the
    # three ORIs carrying it (Valdez, Cordova, Whittier) all sit in Chugach.
    "02261": "02063",
    # Wade Hampton Census Area AK renamed Kusilvak (2015).
    "02270": "02158",
    # Bedford city VA reverted to a town inside Bedford County (2013).
    "51515": "51019",
}
# 999 is the SRS sentinel for "no county", not a county. The Canal Zone (state 57)
# carries 57999 and has no successor geography at all.
SENTINEL_COUNTY_CODES: frozenset[str] = frozenset({"999"})

# Which county_fips provenances are strong enough to ANCHOR an agency's crime to a
# county remainder (allocation._build_county_remainder_group_targets).
#
# The SRS agency header is the FBI's own placement of the agency, and the retired-FIPS
# remap is that same placement forwarded to a live GEOID -- both are authoritative for
# where the agency's crime happened. The roster/LEAIC fills are NOT: they resolve a
# county NAME for an agency the header never placed, which is enough to say which
# county's silence an agency speaks to (imputation eligibility, dead/active predicates)
# but not enough to concentrate its crime into that county's unincorporated remainder.
#
# Verified 2026-07-28: of the 38 remainder-lane agencies the fills made county-anchorable
# with nonzero mass, 36 are roster type "City" whose real footprint is a municipality or a
# regional township force -- Pontiac MI (MI631900X, 1,619 crimes onto Oakland County's
# 32,376-person remainder = 5,001/100k, the v20 candidate's national #1 aggravated-assault
# block group) is the extreme. 24 of the 38 also carry a same-state ORI twin reporting
# identical counts, a pre-existing duplicate-ORI condition this scoping deliberately does
# not deepen.
COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES: frozenset[str] = frozenset(
    {"srs_agency_header", "retired_county_remap"}
)

# Tribal agency identity. Two witnesses, combined asymmetrically because they fail differently.
#
# The FBI CDE roster's own `agency_type_name == "Tribal"` (217 ORIs) is the FBI's declaration
# about the agency and is used on its own.
#
# LEAIC's `LG_POPULATION = 999999999` sentinel means "the local government this agency belongs
# to has no Census place population". That is true of tribes -- but MEASURED 2026-07-30 it is
# ALSO true of police-protection and community-services districts: Kensington CA, Broadmoor CA,
# Bear Valley CA, Lake Shastina CA and Stallion Springs CA all carry the sentinel and are not
# tribal (LEAIC likewise uses 888888888 for state parks and counties). So the sentinel alone is
# NOT a tribal witness and is required to agree with a word-boundary tribal name token.
#
# Verified against the Stage 2 screen: this definition flags 292 ORIs, 246 with a crosswalk
# link, 165 of them resolving to a municipal place/cousub -- the Class D defect population,
# with all 12 of the screen's named false positives excluded.
LEAIC_TRIBAL_LG_POPULATION_SENTINEL = "999999999"
FBI_ROSTER_TRIBAL_AGENCY_TYPE = "Tribal"
# Word boundaries, not substrings: the substring form matched NATIONAL PARK / NATIONAL SECURITY
# AGENCY / INDIANAPOLIS (Stage 2 screen c: 74 false hits over all links). Shared with
# jurisdiction_reference so there is one tribal name test in the codebase.
TRIBAL_NAME_TOKENS: tuple[str, ...] = (
    "TRIBAL",
    "TRIBE",
    "NATION",
    "PUEBLO",
    "RESERVATION",
    "RANCHERIA",
    "INDIAN",
    "BIA",
    "NAVAJO",
)
TRIBAL_NAME_RE = re.compile(r"\b(?:" + "|".join(TRIBAL_NAME_TOKENS) + r")\b")


def matches_tribal_name(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(TRIBAL_NAME_RE.search(_NON_ALNUM_RE.sub(" ", str(value).upper())))


def _leaic_tribal_witnesses(*, paths: RepoPaths) -> tuple[set[str], set[str]]:
    """(ORIs carrying the LG_POPULATION sentinel, ORIs whose LEAIC names read tribal)."""
    leaic_path = (
        paths.data_dir / "LEAIC-Crosswalk-ICPSR_35158" / "DS0001" / "35158-0001-Data.tsv"
    )
    if not leaic_path.exists():
        return set(), set()
    leaic = pd.read_csv(
        leaic_path, sep="\t", dtype=str, usecols=["ORI9", "LG_POPULATION", "NAME", "LG_NAME"]
    )
    ori = leaic["ORI9"].astype("string").str.strip().str.upper()
    sentinel = leaic["LG_POPULATION"].astype("string").str.strip().eq(
        LEAIC_TRIBAL_LG_POPULATION_SENTINEL
    )
    named = leaic["NAME"].map(matches_tribal_name) | leaic["LG_NAME"].map(matches_tribal_name)
    return (
        {value for value in ori[sentinel].dropna() if value},
        {value for value in ori[named].dropna() if value},
    )


def _fbi_roster_tribal_oris(*, paths: RepoPaths, roster_year: int) -> set[str]:
    roster_path = (
        paths.data_dir
        / "FBI-CDE-Agency-Rosters-2024"
        / "parsed"
        / f"agency_rosters_{int(roster_year)}.csv"
    )
    if not roster_path.exists():
        return set()
    roster = pd.read_csv(roster_path, dtype=str, usecols=["ori", "agency_type_name"])
    tribal = roster["agency_type_name"].astype("string").str.strip().eq(
        FBI_ROSTER_TRIBAL_AGENCY_TYPE
    )
    return {
        value
        for value in roster.loc[tribal, "ori"].astype("string").str.strip().str.upper().dropna()
        if value
    }


def tribal_agency_flag(
    ori: pd.Series,
    agency_name: pd.Series | None = None,
    *,
    paths: RepoPaths,
    roster_year: int = 2024,
) -> pd.Series:
    """Boolean tribal-agency flag, per the definition documented above.

    This is what gates the automatic agency-seat-place shortcut in `jurisdiction_reference`,
    which is the Class D mechanism: LEAIC hands a tribal PD the FPLACE of its agency-seat CDP,
    the automatic local lane takes any valid place_fips unconditionally, and a whole
    reservation's crime lands on one small town.
    """
    key = ori.astype("string").str.strip().str.upper()
    sentinel_oris, leaic_named_oris = _leaic_tribal_witnesses(paths=paths)
    roster_oris = _fbi_roster_tribal_oris(paths=paths, roster_year=roster_year)
    named = key.isin(leaic_named_oris)
    if agency_name is not None:
        named = named | agency_name.map(matches_tribal_name).fillna(False).astype(bool)
    return (key.isin(roster_oris) | (key.isin(sentinel_oris) & named)).fillna(False).astype(bool)


def _county_name_key(value: object) -> str:
    text = str(value or "").upper()
    text = _NON_ALNUM_RE.sub(" ", text).strip()
    for suffix in (
        "CITY AND BOROUGH",
        "COUNTY AND BOROUGH",
        "CENSUS AREA",
        "MUNICIPALITY",
        "COUNTY",
        "PARISH",
        "BOROUGH",
    ):
        if text.endswith(f" {suffix}"):
            text = text[: -len(suffix)].strip()
            break
    return _WS_RE.sub(" ", text)


def _county_name_to_geoid_lookup(*, paths: RepoPaths) -> dict[tuple[str, str], str]:
    path = paths.repo_root / "configs" / "county_names_2020.csv"
    if not path.exists():
        return {}
    names = pd.read_csv(path, dtype=str)
    return {
        (str(row.state_fips).zfill(2), _county_name_key(row.county_name)): str(
            row.county_fips
        ).zfill(5)
        for row in names.itertuples(index=False)
        if pd.notna(row.state_fips) and pd.notna(row.county_fips)
    }


def _roster_county_geoid_by_ori(*, paths: RepoPaths, year: int) -> pd.Series:
    """ORI -> county GEOID from the FBI CDE agency roster (`counties` name field).

    The roster is the FBI's own 2024 agency directory, so it covers NIBRS-only
    agencies the SRS header never carried. Where an agency lists several counties the
    first is taken: the county anchor is a single-county assignment by construction.
    Validated against the master's own SRS-derived county on the 16,507 ORIs that
    carry both: 97.7% agreement, and this pass only FILLS nulls, never overrides.
    """
    path = (
        paths.data_dir
        / f"FBI-CDE-Agency-Rosters-{int(year)}"
        / "parsed"
        / f"agency_rosters_{int(year)}.parquet"
    )
    if not path.exists():
        return pd.Series(dtype="string")
    roster = pd.read_parquet(path, columns=["ori", "state_abbr", "counties"])
    roster["ori"] = roster["ori"].astype("string").str.strip().str.upper()
    roster["state_fips"] = (
        roster["state_abbr"].astype("string").str.upper().map(STATE_ABBR_TO_FIPS)
    )
    roster["county_key"] = (
        roster["counties"].astype("string").str.split(",").str[0].map(_county_name_key)
    )
    lookup = _county_name_to_geoid_lookup(paths=paths)
    geoid = [
        lookup.get((str(state_fips), str(key)))
        if pd.notna(state_fips) and pd.notna(key)
        else None
        for state_fips, key in zip(roster["state_fips"], roster["county_key"], strict=True)
    ]
    roster["county_geoid"] = pd.Series(geoid, index=roster.index, dtype="string")
    resolved = roster[roster["ori"].notna() & roster["county_geoid"].notna()]
    return resolved.drop_duplicates(subset=["ori"], keep="first").set_index("ori")[
        "county_geoid"
    ]


def _leaic_county_geoid_by_ori(*, paths: RepoPaths) -> pd.Series:
    path = (
        paths.data_dir
        / "LEAIC-Crosswalk-ICPSR_35158"
        / "DS0001"
        / "35158-0001-Data.tsv"
    )
    if not path.exists():
        return pd.Series(dtype="string")
    leaic = pd.read_csv(path, sep="\t", dtype=str, usecols=["ORI9", "FIPS"])
    leaic["ORI9"] = leaic["ORI9"].astype("string").str.strip().str.upper()
    # LEAIC's FIPS column is the combined 5-digit state+county GEOID (FIPS_COUNTY is
    # the bare 3-digit county code).
    leaic["FIPS"] = leaic["FIPS"].astype("string").str.strip().str.zfill(5)
    resolved = leaic[
        leaic["ORI9"].notna()
        & leaic["FIPS"].notna()
        & leaic["FIPS"].str.fullmatch(r"\d{5}").fillna(False)
    ]
    return resolved.drop_duplicates(subset=["ORI9"], keep="first").set_index("ORI9")[
        "FIPS"
    ]


def canonicalize_agency_county_fips(
    df: pd.DataFrame, *, paths: RepoPaths, roster_year: int = 2024
) -> pd.DataFrame:
    """Make `state_fips + county_fips` a live 2020-vintage county GEOID wherever it can be.

    Three deterministic steps, in order, each recorded on the row so the pass is
    auditable downstream:
      1. sentinel county codes (999) are cleared -- they are "no county", not a county;
      2. retired county GEOIDs are remapped to their successor (RETIRED_COUNTY_GEOID_REMAP);
      3. still-missing county codes are filled from the FBI CDE agency roster, then LEAIC.
    Existing valid codes are never overridden.
    """
    out = df.copy()
    state = out["state_fips"].astype("string").str.zfill(2)
    county = out["county_fips"].astype("string").str.zfill(3)
    source = pd.Series("srs_agency_header", index=out.index, dtype="string")
    source = source.where(county.notna(), pd.NA)

    sentinel = county.isin(SENTINEL_COUNTY_CODES).fillna(False)
    county = county.mask(sentinel, pd.NA)
    source = source.mask(sentinel, pd.NA)

    geoid = state.fillna("") + county.fillna("")
    retired = county.notna() & geoid.isin(RETIRED_COUNTY_GEOID_REMAP)
    if retired.any():
        remapped = geoid[retired].map(RETIRED_COUNTY_GEOID_REMAP).astype("string")
        state = state.mask(retired, remapped.str.slice(0, 2))
        county = county.mask(retired, remapped.str.slice(2, 5))
        source = source.mask(retired, "retired_county_remap")

    ori = out["ori9"].astype("string").str.strip().str.upper()
    for label, lookup in (
        ("fbi_cde_agency_roster", _roster_county_geoid_by_ori(paths=paths, year=roster_year)),
        ("leaic_crosswalk", _leaic_county_geoid_by_ori(paths=paths)),
    ):
        missing = county.isna()
        if not missing.any() or lookup.empty:
            continue
        filled = ori.map(lookup).astype("string")
        # Only fill where the resolved county belongs to the agency's own state.
        usable = missing & filled.notna() & (filled.str.slice(0, 2) == state)
        if not usable.any():
            continue
        county = county.mask(usable, filled.str.slice(2, 5))
        source = source.mask(usable, label)

    out["county_fips"] = county
    out["state_fips"] = state
    out["county_fips_source"] = source
    return out


def _empty_agency_master_supplement() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ori9",
            "ori7",
            "state_fips",
            "state_abbr",
            "agency_name_raw",
            "agency_name_std",
            "agency_type_raw",
            "agency_type_norm",
            "source_presence_srs",
            "source_presence_nibrs",
            "srs_years",
            "nibrs_years",
            "srs_year_count",
            "nibrs_year_count",
            "latest_srs_year",
            "latest_nibrs_year",
            "latest_srs_part1_total",
            "latest_srs_months_reported",
            "latest_srs_months_missing",
            "population_latest_nibrs",
            "name_alias_group",
            "manual_review_flag",
        ]
    )


def _std_text(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "<na>"}:
        return None
    text = text.upper()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text or None


def _normalize_srs_agency_type(value: object) -> str:
    text = _std_text(value)
    if not text:
        return "unknown"
    if text == "LOCAL POLICE DEPARTMENT":
        return "local_police"
    if text == "CONSTABLE MARSHAL":
        return "constable_marshal"
    if text == "SHERIFFS OFFICE":
        return "sheriff"
    if text == "STATE LAW ENFORCEMENT AGENCY":
        return "state_law_enforcement"
    if text == "SPECIAL JURISDICTION":
        return "special_jurisdiction"
    return "unknown"


def _normalize_nibrs_agency_indicator(value: object) -> str:
    text = _std_text(value)
    if not text:
        return "unknown"
    mapping = {
        "CITY": "local_police",
        "COUNTY": "sheriff",
        "STATE POLICE": "state_law_enforcement",
        "STATE OTHER": "state_law_enforcement",
        "UNIVERSITY OR COLLEGE": "special_jurisdiction",
        "OTHER": "special_jurisdiction",
        "TRIBAL": "special_jurisdiction",
        "FEDERAL": "special_jurisdiction",
    }
    return mapping.get(text, "unknown")


def _zfill_nullable(series: pd.Series, width: int) -> pd.Series:
    return series.map(
        lambda value: None
        if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan", "<na>"}
        else str(value).zfill(width)
    )


def _parse_nullable_bool(value: object) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, bool)):
        return bool(value)
    if isinstance(value, float):
        if value in {0.0, 1.0}:
            return bool(int(value))
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "<na>"}:
        return None
    if text in {"true", "1", "1.0", "yes", "y"}:
        return True
    if text in {"false", "0", "0.0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean supplement value: {value!r}")


def _parse_nullable_numeric(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "<na>"}:
        return None
    number = float(text)
    return int(number) if number.is_integer() else number


def _load_agency_master_supplement(*, supplement_path: Path | None) -> pd.DataFrame:
    if supplement_path is None or not supplement_path.exists():
        return _empty_agency_master_supplement()

    supplement = pd.read_csv(supplement_path).copy()
    required = {"ori9", "agency_name_raw", "state_fips", "state_abbr"}
    missing = required - set(supplement.columns)
    if missing:
        raise ValueError(f"Agency master supplement missing columns: {sorted(missing)}")

    if supplement.empty:
        return _empty_agency_master_supplement()

    supplement["ori9"] = supplement["ori9"].astype(str).str.strip()
    if "ori7" in supplement.columns:
        supplement["ori7"] = supplement["ori7"].map(lambda value: None if pd.isna(value) else str(value).strip() or None)
    else:
        supplement["ori7"] = None
    missing_ori7 = supplement["ori7"].isna()
    supplement.loc[missing_ori7, "ori7"] = supplement.loc[missing_ori7, "ori9"].str[:7]

    supplement["state_fips"] = supplement["state_fips"].map(lambda value: None if pd.isna(value) else str(value).zfill(2))
    supplement["state_abbr"] = supplement["state_abbr"].map(_std_text)
    supplement["agency_name_raw"] = supplement["agency_name_raw"].map(
        lambda value: None if pd.isna(value) else str(value).strip() or None
    )
    if "agency_name_std" in supplement.columns:
        supplement["agency_name_std"] = supplement["agency_name_std"].map(_std_text)
    else:
        supplement["agency_name_std"] = supplement["agency_name_raw"].map(_std_text)
    if "agency_type_raw" in supplement.columns:
        supplement["agency_type_raw"] = supplement["agency_type_raw"].map(
            lambda value: None if pd.isna(value) else str(value).strip() or None
        )
    else:
        supplement["agency_type_raw"] = None
    if "agency_type_norm" in supplement.columns:
        supplement["agency_type_norm"] = supplement["agency_type_norm"].map(
            lambda value: None if pd.isna(value) else str(value).strip() or None
        )
    else:
        supplement["agency_type_norm"] = None

    for col in ("source_presence_srs", "source_presence_nibrs", "manual_review_flag"):
        if col in supplement.columns:
            supplement[col] = supplement[col].map(_parse_nullable_bool)
        else:
            supplement[col] = None

    numeric_cols = (
        "srs_year_count",
        "nibrs_year_count",
        "latest_srs_year",
        "latest_nibrs_year",
        "latest_srs_part1_total",
        "latest_srs_months_reported",
        "latest_srs_months_missing",
        "population_latest_nibrs",
    )
    for col in numeric_cols:
        if col in supplement.columns:
            supplement[col] = supplement[col].map(_parse_nullable_numeric)
        else:
            supplement[col] = None

    for col in ("srs_years", "nibrs_years", "name_alias_group"):
        if col in supplement.columns:
            supplement[col] = supplement[col].map(
                lambda value: None if pd.isna(value) else str(value).strip() or None
            )
        else:
            supplement[col] = None

    supplement = supplement.drop_duplicates(subset=["ori9"], keep="last").reset_index(drop=True)
    out = _empty_agency_master_supplement()
    for col in out.columns:
        if col in supplement.columns:
            out[col] = supplement[col]
    return out


def _apply_agency_master_supplement(df: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    if supplement.empty:
        return df

    out = df.copy()
    fill_cols = [col for col in supplement.columns if col != "ori9"]
    supplement_indexed = supplement.set_index("ori9")

    for col in fill_cols:
        if col not in out.columns:
            out[col] = None
        mapped = out["ori9"].map(supplement_indexed[col])
        updates = mapped.loc[out[col].isna() & mapped.notna()]
        if updates.empty:
            continue
        out.loc[updates.index, col] = updates

    missing = supplement[~supplement["ori9"].isin(out["ori9"])].copy()
    if not missing.empty:
        for col in out.columns:
            if col not in missing.columns:
                missing[col] = None
        out = pd.concat([out, missing[out.columns]], ignore_index=True)

    out["name_alias_group"] = out["agency_name_std"]
    out["manual_review_flag"] = out["agency_name_std"].isna() | out["state_fips"].isna()
    return out




def _load_srs_agency_presence(*, paths: RepoPaths, year_start: int, year_end: int) -> pd.DataFrame:
    srs_zip = paths.data_dir / "SRS-Kaplan-1960-2024" / "offenses_known_parquet_1960_2024_year.zip"
    extract = ensure_srs_year_parquet(zip_path=srs_zip, cache_dir=paths.cache_dir)
    query = f"""
    SELECT
      COALESCE(NULLIF(CAST(ori9 AS VARCHAR), ''), CAST(ori AS VARCHAR) || '00') AS ori9,
      COALESCE(NULLIF(CAST(ori AS VARCHAR), ''), SUBSTR(COALESCE(NULLIF(CAST(ori9 AS VARCHAR), ''), CAST(ori AS VARCHAR) || '00'), 1, 7)) AS ori7,
      CAST(year AS INTEGER) AS year,
      CAST(fips_state_code AS VARCHAR) AS state_fips,
      CAST(fips_county_code AS VARCHAR) AS county_fips,
      CAST(fips_place_code AS VARCHAR) AS place_fips,
      UPPER(CAST(state_abb AS VARCHAR)) AS state_abbr,
      agency_name,
      agency_type,
      census_name,
      crosswalk_agency_name,
      CAST(number_of_months_missing AS INTEGER) AS months_missing,
      CASE
        WHEN number_of_months_missing IS NULL THEN NULL
        ELSE GREATEST(0, LEAST(12, 12 - CAST(number_of_months_missing AS INTEGER)))
      END AS months_reported,
      COALESCE(CAST(actual_murder AS DOUBLE), 0.0)
        + COALESCE(CAST(actual_rape_total AS DOUBLE), 0.0)
        + COALESCE(CAST(actual_robbery_total AS DOUBLE), 0.0)
        + COALESCE(CAST(actual_assault_aggravated AS DOUBLE), 0.0)
        + COALESCE(CAST(actual_burglary_total AS DOUBLE), 0.0)
        + COALESCE(CAST(actual_theft_total AS DOUBLE), 0.0)
        + COALESCE(CAST(actual_motor_vehicle_theft_total AS DOUBLE), 0.0) AS part1_total_7
    FROM read_parquet('{extract.parquet_path.as_posix()}')
    WHERE CAST(year AS INTEGER) BETWEEN {int(year_start)} AND {int(year_end)}
    """
    df = duckdb.sql(query).df()
    if df.empty:
        return df
    df["ori9"] = df["ori9"].astype(str)
    df["ori7"] = df["ori7"].astype(str)
    df["state_fips"] = _zfill_nullable(df["state_fips"], 2)
    df["county_fips"] = _zfill_nullable(df["county_fips"], 3)
    df["place_fips"] = _zfill_nullable(df["place_fips"], 5)
    df["agency_name_std"] = df["agency_name"].map(_std_text)
    df["census_name_std"] = df["census_name"].map(_std_text)
    df["crosswalk_agency_name_std"] = df["crosswalk_agency_name"].map(_std_text)
    df["agency_type_norm"] = df["agency_type"].map(_normalize_srs_agency_type)
    return df


def _load_nibrs_agency_presence(*, paths: RepoPaths, year_start: int, year_end: int) -> pd.DataFrame:
    nibrs_zip = paths.data_dir / "NIBRS-Kaplan-1991-2024" / "batch_header_parquet_1991_2024.zip"
    extract = ensure_nibrs_batch_header_parquet(zip_path=nibrs_zip, cache_dir=paths.cache_dir)
    query = f"""
    SELECT
      CAST(ori AS VARCHAR) AS ori9,
      CAST(year AS INTEGER) AS year,
      CAST(state AS VARCHAR) AS state_name,
      UPPER(CAST(state_abbreviation AS VARCHAR)) AS state_abbr,
      CAST(city_name AS VARCHAR) AS city_name,
      CAST(agency_indicator AS VARCHAR) AS agency_indicator,
      CAST(agency_nibrs_flag AS VARCHAR) AS agency_nibrs_flag,
      CAST(number_of_months_reported AS INTEGER) AS months_reported_nibrs,
      CAST(population AS DOUBLE) AS population
    FROM read_parquet('{extract.parquet_path.as_posix()}')
    WHERE CAST(year AS INTEGER) BETWEEN {int(year_start)} AND {int(year_end)}
    """
    df = duckdb.sql(query).df()
    if df.empty:
        return df
    df["ori9"] = df["ori9"].astype(str)
    df["city_name_std"] = df["city_name"].map(_std_text)
    df["agency_indicator_std"] = df["agency_indicator"].map(_std_text)
    df["agency_type_norm_nibrs"] = df["agency_indicator"].map(_normalize_nibrs_agency_indicator)
    return df


def _collapse_years(years: list[int]) -> str:
    years = sorted({int(y) for y in years if pd.notna(y)})
    return ",".join(str(y) for y in years)


def build_agency_master(
    *,
    paths: RepoPaths,
    year_start: int = 2018,
    year_end: int = 2024,
    supplement_path: Path | None = None,
) -> pd.DataFrame:
    if supplement_path is None:
        default_supplement = paths.repo_root / "configs" / "agency_master_supplement.csv"
        supplement_path = default_supplement if default_supplement.exists() else None

    srs = _load_srs_agency_presence(paths=paths, year_start=year_start, year_end=year_end)
    nibrs = _load_nibrs_agency_presence(paths=paths, year_start=year_start, year_end=year_end)

    srs_grouped = pd.DataFrame(columns=["ori9"])
    if not srs.empty:
        srs_sorted = srs.sort_values(["ori9", "year"])
        srs_grouped = (
            srs_sorted.groupby("ori9", as_index=False)
            .agg(
                ori7=("ori7", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                state_fips=("state_fips", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                county_fips=("county_fips", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                place_fips=("place_fips", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                state_abbr_srs=("state_abbr", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                agency_name_raw_srs=("agency_name", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                agency_name_std_srs=("agency_name_std", lambda s: next((x for x in s if x), None)),
                census_name_std=("census_name_std", lambda s: next((x for x in s if x), None)),
                crosswalk_agency_name_std=("crosswalk_agency_name_std", lambda s: next((x for x in s if x), None)),
                agency_type_raw_srs=("agency_type", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                agency_type_norm_srs=("agency_type_norm", lambda s: next((x for x in s if x and x != "unknown"), "unknown")),
                srs_years=("year", lambda s: _collapse_years(list(s))),
                srs_year_count=("year", "nunique"),
                srs_max_months_reported=("months_reported", "max"),
                latest_srs_year=("year", "max"),
            )
        )
        srs_latest = (
            srs_sorted.sort_values(["ori9", "year"])
            .drop_duplicates(subset=["ori9"], keep="last")[["ori9", "part1_total_7", "months_reported", "months_missing"]]
            .rename(
                columns={
                    "part1_total_7": "latest_srs_part1_total",
                    "months_reported": "latest_srs_months_reported",
                    "months_missing": "latest_srs_months_missing",
                }
            )
        )
        srs_grouped = srs_grouped.merge(srs_latest, on="ori9", how="left")
        srs_grouped["source_presence_srs"] = srs_grouped["srs_year_count"].fillna(0).astype(int) > 0

    nibrs_grouped = pd.DataFrame(columns=["ori9"])
    if not nibrs.empty:
        nibrs_sorted = nibrs.sort_values(["ori9", "year"])
        nibrs_grouped = (
            nibrs_sorted.groupby("ori9", as_index=False)
            .agg(
                state_abbr_nibrs=("state_abbr", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                city_name_raw_nibrs=("city_name", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                city_name_std_nibrs=("city_name_std", lambda s: next((x for x in s if x), None)),
                agency_indicator_raw_nibrs=("agency_indicator", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), None)),
                agency_type_norm_nibrs=("agency_type_norm_nibrs", lambda s: next((x for x in s if x and x != "unknown"), "unknown")),
                nibrs_years=("year", lambda s: _collapse_years(list(s))),
                nibrs_year_count=("year", "nunique"),
                nibrs_max_months_reported=("months_reported_nibrs", "max"),
                population_latest_nibrs=("population", "max"),
                latest_nibrs_year=("year", "max"),
            )
        )
        nibrs_grouped["source_presence_nibrs"] = nibrs_grouped["nibrs_year_count"].fillna(0).astype(int) > 0

    ori_df = pd.DataFrame({"ori9": sorted(set(srs_grouped.get("ori9", pd.Series(dtype=str))) | set(nibrs_grouped.get("ori9", pd.Series(dtype=str))))})
    if ori_df.empty:
        return ori_df

    df = ori_df.merge(srs_grouped, on="ori9", how="left").merge(nibrs_grouped, on="ori9", how="left")
    df["state_abbr_srs"] = df["state_abbr_srs"].map(_std_text)
    df["state_abbr_nibrs"] = df["state_abbr_nibrs"].map(_std_text)
    df["state_abbr"] = df["state_abbr_srs"].fillna(df["state_abbr_nibrs"])
    ori_prefix = df["ori9"].astype(str).str[:2].str.upper()
    fill_abbr = df["state_abbr"].isna() & ori_prefix.isin(set(STATE_ABBR_TO_FIPS))
    df.loc[fill_abbr, "state_abbr"] = ori_prefix.loc[fill_abbr]
    df["state_fips"] = df["state_fips"].where(df["state_fips"].notna(), df["state_abbr"].map(STATE_ABBR_TO_FIPS))
    df["agency_name_raw"] = df["agency_name_raw_srs"].fillna(df["city_name_raw_nibrs"])
    df["agency_name_std"] = (
        df["agency_name_std_srs"]
        .fillna(df["crosswalk_agency_name_std"])
        .fillna(df["census_name_std"])
        .fillna(df["city_name_std_nibrs"])
    )
    df["agency_type_raw"] = df["agency_type_raw_srs"].fillna(df["agency_indicator_raw_nibrs"])
    df["agency_type_norm"] = df["agency_type_norm_srs"]
    needs_fill = df["agency_type_norm"].isna() | (df["agency_type_norm"] == "unknown")
    df.loc[needs_fill, "agency_type_norm"] = df.loc[needs_fill, "agency_type_norm_nibrs"]
    df["agency_type_norm"] = df["agency_type_norm"].fillna("unknown")
    df["source_presence_srs"] = df["source_presence_srs"].fillna(False).astype(bool)
    df["source_presence_nibrs"] = df["source_presence_nibrs"].fillna(False).astype(bool)
    df = _apply_agency_master_supplement(
        df,
        _load_agency_master_supplement(supplement_path=supplement_path),
    )
    df = canonicalize_agency_county_fips(df, paths=paths, roster_year=int(year_end))
    df["is_tribal_agency"] = tribal_agency_flag(
        df["ori9"],
        df["agency_name_std"],
        paths=paths,
        roster_year=int(year_end),
    )
    df["name_alias_group"] = df["agency_name_std"]
    df["manual_review_flag"] = df["agency_name_std"].isna() | df["state_fips"].isna()

    keep_cols = [
        "ori9",
        "ori7",
        "state_fips",
        "state_abbr",
        "county_fips",
        "place_fips",
        "agency_name_raw",
        "agency_name_std",
        "agency_name_std_srs",
        "census_name_std",
        "crosswalk_agency_name_std",
        "city_name_std_nibrs",
        "agency_type_raw",
        "agency_type_norm",
        "source_presence_srs",
        "source_presence_nibrs",
        "srs_years",
        "nibrs_years",
        "srs_year_count",
        "nibrs_year_count",
        "srs_max_months_reported",
        "nibrs_max_months_reported",
        "latest_srs_year",
        "latest_srs_part1_total",
        "latest_srs_months_reported",
        "latest_srs_months_missing",
        "latest_nibrs_year",
        "population_latest_nibrs",
        "name_alias_group",
        "manual_review_flag",
        "county_fips_source",
        "is_tribal_agency",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    df = df[keep_cols].sort_values("ori9").reset_index(drop=True)
    return df


def write_input_manifest(*, paths: RepoPaths, out_path: Path) -> Path:
    from crimerisk.required_inputs import write_required_input_manifest

    return write_required_input_manifest(repo_root=paths.repo_root, out_path=out_path)


def write_agency_master(
    *,
    paths: RepoPaths,
    out_path: Path,
    year_start: int = 2018,
    year_end: int = 2024,
    supplement_path: Path | None = None,
) -> Path:
    df = build_agency_master(
        paths=paths,
        year_start=year_start,
        year_end=year_end,
        supplement_path=supplement_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path
