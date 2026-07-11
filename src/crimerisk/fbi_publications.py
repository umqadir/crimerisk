from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.jurisdiction_reference import STATE_ABBR_BY_FIPS, STATE_NAME_BY_FIPS
from crimerisk.paths import RepoPaths
from crimerisk.reference import _std_text


STATE_NAME_TO_ABBR = {name.upper(): STATE_ABBR_BY_FIPS[fips] for fips, name in STATE_NAME_BY_FIPS.items()}
PRODUCTION_SCOPE_EXCLUDE = {"AK", "HI", "PR"}
MUNICIPAL_SUFFIX_RE = re.compile(
    r"\b(CITY|TOWN|VILLAGE|BOROUGH|MUNICIPALITY|TOWNSHIP|PLANTATION|METROPOLITAN GOVERNMENT|URBAN COUNTY|UNIFIED GOVERNMENT|BALANCE)\b$"
)
CIUS_CITY_FOOTNOTE_RE = re.compile(r"(?<=[A-Z])\d+$")
CIUS_OFFENSE_COLUMNS = {
    "Murder and\nnonnegligent\nmanslaughter": "murder",
    "Rape": "rape",
    "Robbery": "robbery",
    "Aggravated\nassault": "aggravated_assault",
    "Burglary": "burglary",
    "Larceny-\ntheft": "larceny",
    "Motor\nvehicle\ntheft": "motor_vehicle_theft",
}
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
CIUS_STATE_TYPE_SPLIT_RE = re.compile(r"\s*[–-]\s*")

CIUS_TABLE_SPECS = {
    "table8_city": {
        "member_tokens": ("table", "8", "city"),
        "glob_patterns": [
            "*Table*8*.xls*",
            "*table*8*.xls*",
            "*Offenses*Known*City*{year}*.xls*",
        ],
        "id_cols": ["State", "City"],
    },
    "table9_university": {
        "member_tokens": ("table", "9", "university"),
        "glob_patterns": [
            "*Table*9*.xls*",
            "*table*9*.xls*",
            "*University*College*{year}*.xls*",
        ],
        "id_cols": ["State", "University/College"],
    },
    "table11_state_tribal_other": {
        "member_tokens": ("table", "11", "tribal", "other"),
        "glob_patterns": [
            "*Table*11*.xls*",
            "*table*11*.xls*",
            "*State*Tribal*Other*{year}*.xls*",
        ],
        "id_cols": ["State", "State/Tribal/Other", "Agency", "Unit/Office"],
    },
}


def norm_text(value: object) -> str | None:
    return _std_text(value)


def norm_municipal_name(value: object) -> str | None:
    text = norm_text(value)
    if not text:
        return None
    prev = None
    while prev != text:
        prev = text
        text = MUNICIPAL_SUFFIX_RE.sub("", text).strip()
    return text or None


def norm_cius_city_name(value: object) -> str | None:
    text = norm_text(value)
    if not text:
        return None
    text = CIUS_CITY_FOOTNOTE_RE.sub("", text).strip()
    return text or None


def _scope_state_mask(series: pd.Series) -> pd.Series:
    return ~series.astype("string").str.upper().isin(PRODUCTION_SCOPE_EXCLUDE)


def _read_excel_from_zip(zip_path: Path, member: str, *, header) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        return pd.read_excel(io.BytesIO(zf.read(member)), header=header)


def _read_excel_path(path: Path, *, header) -> pd.DataFrame:
    return pd.read_excel(path, header=header)


def _norm_col(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return _NON_ALNUM_RE.sub("", text)


CIUS_OFFENSE_COLUMNS_NORM = {
    _norm_col(k): v for k, v in CIUS_OFFENSE_COLUMNS.items()
}
CIUS_OFFENSE_COLUMNS_NORM.update(
    {
        "LARCENYTHEFT": "larceny",
        "LARCENYTHEFTOFFENSES": "larceny",
        "MOTORVEHICLETHEFT": "motor_vehicle_theft",
    }
)


def _map_cius_offense_norm(norm: str) -> str | None:
    if not norm:
        return None
    if norm in CIUS_OFFENSE_COLUMNS_NORM:
        return CIUS_OFFENSE_COLUMNS_NORM[norm]
    if norm.startswith("RAPE"):
        return "rape"
    return None


def _extract_state_abbr_from_source_name(name: str) -> str | None:
    match = re.search(r"ENFORCEMENT_(.+?)_BY_", name.upper())
    if not match:
        return None
    state_name = match.group(1).replace("_", " ").strip()
    return STATE_NAME_TO_ABBR.get(state_name)


def _extract_state_name_from_title_rows(raw_headerless: pd.DataFrame) -> str | None:
    if raw_headerless.empty:
        return None
    for row_idx in range(min(4, len(raw_headerless))):
        first = raw_headerless.iloc[row_idx, 0]
        if first is None or pd.isna(first):
            continue
        text = str(first).strip().upper()
        if not text:
            continue
        text = re.sub(r"\d+$", "", text).strip()
        if text in STATE_NAME_TO_ABBR:
            return text
    return None


def _normalize_table11_type(value: object) -> str | None:
    text = norm_text(value)
    if not text:
        return None
    text = re.sub(r"\bAGENCIES\b", "AGENCY", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _load_cius_source_frame(source_kind: str, source_path: Path, member: str | None, *, table_key: str) -> pd.DataFrame:
    source_name = member or source_path.name
    state_abbr_from_name = _extract_state_abbr_from_source_name(source_name)
    headerless = (
        _read_excel_from_zip(source_path, str(member), header=None).copy()
        if source_kind == "zip"
        else _read_excel_path(source_path, header=None).copy()
    )
    state_name_from_header = _extract_state_name_from_title_rows(headerless)
    raw: pd.DataFrame | None = None
    norm_map: dict[str, str] = {}
    for header_idx in range(min(12, len(headerless))):
        header_vals = headerless.iloc[header_idx].tolist()
        current_norm = {str(idx): _norm_col(value) for idx, value in enumerate(header_vals)}
        offense_hits = sum(1 for value in set(current_norm.values()) if _map_cius_offense_norm(value) is not None)
        if table_key == "table8_city":
            has_city = any(value in current_norm.values() for value in ("CITY", "STATE"))
            ok = has_city and offense_hits >= 6
        elif table_key == "table9_university":
            ok = "UNIVERSITYCOLLEGE" in current_norm.values() and offense_hits >= 6
        else:
            has_state = "STATE" in current_norm.values()
            has_type = any(value in current_norm.values() for value in ("STATETRIBALOTHER", "AGENCYTYPE"))
            has_agency = any(value in current_norm.values() for value in ("AGENCY", "STATETRIBALOTHERAGENCIES"))
            ok = (
                offense_hits >= 6
                and (
                    (has_state and has_type and has_agency)
                    or (has_type and has_agency)
                    or (has_state and "STATETRIBALOTHERAGENCIES" in current_norm.values())
                )
            )
        if not ok:
            continue
        columns = [str(value).strip() if pd.notna(value) else f"Unnamed: {idx}" for idx, value in enumerate(header_vals)]
        trial = headerless.iloc[header_idx + 1 :].copy()
        trial.columns = columns
        trial = trial.dropna(how="all").reset_index(drop=True)
        raw = trial
        norm_map = {str(col): _norm_col(col) for col in trial.columns}
        break
    if raw is None:
        raise ValueError(f"Could not detect CIUS schema for {source_name}")

    rename_map: dict[str, str] = {}
    for col, norm in norm_map.items():
        if norm == "STATE":
            rename_map[col] = "State"
        elif norm == "CITY":
            rename_map[col] = "City"
        elif norm == "UNIVERSITYCOLLEGE":
            rename_map[col] = "University/College"
        elif norm == "STATETRIBALOTHER":
            rename_map[col] = "State/Tribal/Other"
        elif norm == "STATETRIBALOTHERAGENCIES":
            rename_map[col] = "Agency" if table_key == "table11_state_tribal_other" else "State/Tribal/Other"
        elif norm == "AGENCYTYPE":
            rename_map[col] = "State/Tribal/Other"
        elif norm == "AGENCY":
            rename_map[col] = "Agency"
        elif norm == "UNITOFFICE":
            rename_map[col] = "Unit/Office"
        elif _map_cius_offense_norm(norm) is not None:
            target_offense = _map_cius_offense_norm(norm)
            rename_map[col] = next(
                original for original, offense in CIUS_OFFENSE_COLUMNS.items()
                if offense == target_offense
            )
    raw = raw.rename(columns=rename_map)
    if table_key == "table8_city" and "City" not in raw.columns and "State" in raw.columns:
        # Legacy state-cut table 8 workbooks sometimes label the city-name column as "State".
        raw = raw.rename(columns={"State": "City"})
    if "State" not in raw.columns and state_abbr_from_name:
        reverse_state_name = {abbr: name.upper() for name, abbr in STATE_NAME_TO_ABBR.items()}
        raw["State"] = reverse_state_name.get(state_abbr_from_name, pd.NA)
    elif "State" not in raw.columns and state_name_from_header:
        raw["State"] = state_name_from_header
    if "State" in raw.columns:
        raw["State"] = raw["State"].astype("string").ffill()
    if table_key == "table11_state_tribal_other" and "State/Tribal/Other" not in raw.columns and "State" in raw.columns:
        state_values = raw["State"].astype("string").ffill()
        split = state_values.str.split(CIUS_STATE_TYPE_SPLIT_RE, n=1, expand=True)
        if split.shape[1] >= 2:
            raw["State"] = split[0]
            raw["State/Tribal/Other"] = split[1]
    if table_key == "table11_state_tribal_other" and "State/Tribal/Other" in raw.columns:
        raw["State/Tribal/Other"] = raw["State/Tribal/Other"].map(_normalize_table11_type)
    if table_key == "table11_state_tribal_other" and "Agency" in raw.columns:
        raw["Agency"] = raw["Agency"].astype("string").replace({"": pd.NA}).ffill()
    if table_key == "table11_state_tribal_other" and "Unit/Office" not in raw.columns:
        raw["Unit/Office"] = pd.NA
    if table_key == "table9_university" and "University/College" in raw.columns:
        raw["University/College"] = raw["University/College"].astype("string").replace({"": pd.NA}).ffill()
    return raw


def _ciustable_member_match(name: str, *, year: int, table_key: str) -> bool:
    lower = name.lower()
    if not lower.endswith((".xlsx", ".xls")):
        return False
    if str(year) not in lower:
        return False
    tokens = tuple(CIUS_TABLE_SPECS[table_key]["member_tokens"])
    return all(token in lower for token in tokens)


def _resolve_cius_table_sources(paths: RepoPaths, *, year: int, table_key: str) -> list[tuple[str, Path, str | None]]:
    raw_dir = paths.data_dir / "FBI-CIUS-Annual" / str(year) / "raw"
    spec = CIUS_TABLE_SPECS[table_key]
    out: list[tuple[str, Path, str | None]] = []
    seen: set[tuple[str, str]] = set()

    zip_path = raw_dir / f"offenses-known-to-le-{year}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            for member in sorted(zf.namelist()):
                if _ciustable_member_match(member, year=year, table_key=table_key):
                    key = ("zip", member)
                    if key not in seen:
                        out.append(("zip", zip_path, member))
                        seen.add(key)
    for pattern in spec["glob_patterns"]:
        matches = sorted(raw_dir.glob(str(pattern).format(year=year)))
        for match in matches:
            if not _ciustable_member_match(match.name, year=year, table_key=table_key):
                continue
            key = ("file", str(match))
            if key not in seen:
                out.append(("file", match, None))
                seen.add(key)
    return out


def cius_local_rows_available(paths: RepoPaths, *, year: int) -> bool:
    return all(_resolve_cius_table_sources(paths, year=year, table_key=table_key) for table_key in CIUS_TABLE_SPECS)


def parse_cius_local_rows(paths: RepoPaths, *, year: int) -> pd.DataFrame:
    output_cols = [
        "publication_collection",
        "publication_type",
        "state_abbr",
        "publication_name_raw",
        "publication_name_exact_std",
        "publication_name_std",
        "publication_unit_raw",
        "offense",
        "official_count",
    ]

    def _normalize_output(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reindex(columns=output_cols).copy()
        for col in output_cols:
            if col == "official_count":
                out[col] = pd.to_numeric(out[col], errors="coerce")
            else:
                out[col] = out[col].astype("string")
        return out

    frames: list[pd.DataFrame] = []
    for collection, spec in CIUS_TABLE_SPECS.items():
        sources = _resolve_cius_table_sources(paths, year=year, table_key=collection)
        if not sources:
            raise FileNotFoundError(f"Missing CIUS local table source for year={year}, collection={collection}")
        for source_kind, source_path, member in sources:
            raw = _load_cius_source_frame(source_kind, source_path, member, table_key=collection)
            raw["state_name"] = raw["State"].astype("string").str.strip().str.upper()
            raw["state_abbr"] = raw["state_name"].map(STATE_NAME_TO_ABBR)
            raw = raw[raw["state_abbr"].notna() & _scope_state_mask(raw["state_abbr"])].copy()
            value_cols = [col for col in CIUS_OFFENSE_COLUMNS.keys() if col in raw.columns]
            long = raw.melt(
                id_vars=list(spec["id_cols"]) + ["state_abbr"],
                value_vars=value_cols,
                var_name="official_offense_col",
                value_name="official_count",
            )
            long["official_count"] = pd.to_numeric(long["official_count"], errors="coerce")
            long = long[long["official_count"].notna()].copy()
            long["offense"] = long["official_offense_col"].map(CIUS_OFFENSE_COLUMNS)
            long["publication_collection"] = collection
            if collection == "table8_city":
                long["publication_name_raw"] = long["City"]
                long["publication_name_exact_std"] = long["City"].map(norm_cius_city_name)
                long["publication_name_std"] = long["City"].map(lambda value: norm_municipal_name(norm_cius_city_name(value)))
                long["publication_type"] = "municipal"
                long["publication_unit_raw"] = pd.NA
            elif collection == "table9_university":
                long["publication_name_raw"] = long["University/College"]
                long["publication_name_exact_std"] = long["University/College"].map(norm_text)
                long["publication_name_std"] = long["University/College"].map(norm_text)
                long["publication_type"] = "university_college"
                long["publication_unit_raw"] = pd.NA
            else:
                long["publication_name_raw"] = long["Agency"]
                long["publication_name_exact_std"] = long["Agency"].map(norm_text)
                long["publication_name_std"] = long["Agency"].map(norm_text)
                long["publication_type"] = long["State/Tribal/Other"].astype("string")
                long["publication_unit_raw"] = long["Unit/Office"].astype("string")
            frames.append(_normalize_output(long))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=output_cols)


def match_cius_municipal_rows_to_jurisdictions(
    cius_rows: pd.DataFrame,
    jurisdictions: pd.DataFrame,
    *,
    agency_master: pd.DataFrame | None = None,
    agency_to_jurisdiction_crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    municipal = cius_rows[cius_rows["publication_collection"].eq("table8_city")].copy()
    if municipal.empty:
        return pd.DataFrame(columns=list(cius_rows.columns) + ["jurisdiction_id"])

    muni = jurisdictions.copy()
    if "jurisdiction_type" in muni.columns:
        muni = muni[muni["jurisdiction_type"].eq("municipal")].copy()
    muni["publication_name_exact_std"] = muni["jurisdiction_name"].map(norm_text)
    muni["publication_name_std"] = muni["jurisdiction_name"].map(norm_municipal_name)
    muni = muni[["jurisdiction_id", "state_abbr", "publication_name_exact_std", "publication_name_std"]].drop_duplicates()

    municipal = municipal.reset_index(drop=True).copy()
    municipal["match_row_id"] = np.arange(len(municipal))

    exact_candidates = muni[
        muni["publication_name_exact_std"].notna()
    ][["jurisdiction_id", "state_abbr", "publication_name_exact_std"]].drop_duplicates()
    exact_counts = (
        exact_candidates.groupby(["state_abbr", "publication_name_exact_std"], dropna=False)["jurisdiction_id"]
        .nunique()
        .rename("repo_match_count")
        .reset_index()
    )
    exact_candidates = exact_candidates.merge(
        exact_counts,
        on=["state_abbr", "publication_name_exact_std"],
        how="left",
    )
    exact_candidates = exact_candidates[exact_candidates["repo_match_count"].eq(1)].copy()
    exact = municipal.merge(
        exact_candidates[["jurisdiction_id", "state_abbr", "publication_name_exact_std"]],
        on=["state_abbr", "publication_name_exact_std"],
        how="inner",
    )

    unresolved = municipal[~municipal["match_row_id"].isin(exact["match_row_id"])].copy()
    fallback = pd.DataFrame(columns=list(municipal.columns) + ["jurisdiction_id"])
    if not unresolved.empty:
        core_candidates = muni[
            muni["publication_name_std"].notna()
        ][["jurisdiction_id", "state_abbr", "publication_name_std"]].drop_duplicates()
        core_counts = (
            core_candidates.groupby(["state_abbr", "publication_name_std"], dropna=False)["jurisdiction_id"]
            .nunique()
            .rename("repo_match_count")
            .reset_index()
        )
        core_candidates = core_candidates.merge(
            core_counts,
            on=["state_abbr", "publication_name_std"],
            how="left",
        )
        core_candidates = core_candidates[core_candidates["repo_match_count"].eq(1)].copy()
        fallback = unresolved.merge(
            core_candidates[["jurisdiction_id", "state_abbr", "publication_name_std"]],
            on=["state_abbr", "publication_name_std"],
            how="inner",
        )

    agency_alias = pd.DataFrame(columns=list(municipal.columns) + ["jurisdiction_id"])
    unresolved = unresolved[~unresolved["match_row_id"].isin(fallback["match_row_id"])].copy()
    if (
        not unresolved.empty
        and agency_master is not None
        and agency_to_jurisdiction_crosswalk is not None
    ):
        alias_candidates = build_municipal_agency_alias_rows(
            agency_master=agency_master,
            agency_to_jurisdiction_crosswalk=agency_to_jurisdiction_crosswalk,
            jurisdictions=jurisdictions,
        )
        if not alias_candidates.empty:
            agency_alias = unresolved.merge(
                alias_candidates[["state_abbr", "publication_name_exact_std", "jurisdiction_id"]],
                on=["state_abbr", "publication_name_exact_std"],
                how="inner",
            )

    matched = pd.concat([exact, fallback, agency_alias], ignore_index=True)
    if matched.empty:
        return pd.DataFrame(columns=list(cius_rows.columns) + ["jurisdiction_id"])
    return matched.drop_duplicates(subset=["match_row_id"], keep="first").drop(columns=["match_row_id"])


def build_municipal_agency_alias_rows(
    *,
    agency_master: pd.DataFrame,
    agency_to_jurisdiction_crosswalk: pd.DataFrame,
    jurisdictions: pd.DataFrame,
) -> pd.DataFrame:
    municipal_ids = set(
        jurisdictions.loc[jurisdictions["jurisdiction_type"].eq("municipal"), "jurisdiction_id"].astype(str)
    )
    crosswalk = agency_to_jurisdiction_crosswalk.rename(columns={"ori": "ori9"}).copy()
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce")
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).isin(municipal_ids) & crosswalk["weight"].gt(0.999)].copy()
    if crosswalk.empty:
        return pd.DataFrame(columns=["state_abbr", "publication_name_exact_std", "jurisdiction_id"])
    crosswalk = crosswalk.groupby("ori9", dropna=False).filter(lambda g: g["jurisdiction_id"].nunique() == 1)
    crosswalk = crosswalk[["ori9", "jurisdiction_id"]].drop_duplicates()
    if crosswalk.empty:
        return pd.DataFrame(columns=["state_abbr", "publication_name_exact_std", "jurisdiction_id"])

    aliases = agency_master.merge(crosswalk, on="ori9", how="inner")
    alias_frames: list[pd.DataFrame] = []
    for alias_col in ["agency_name_std", "crosswalk_agency_name_std", "census_name_std"]:
        if alias_col not in aliases.columns:
            continue
        part = aliases[["state_abbr", "jurisdiction_id", alias_col]].rename(
            columns={alias_col: "publication_name_exact_std"}
        )
        part["publication_name_exact_std"] = part["publication_name_exact_std"].map(norm_text)
        alias_frames.append(part)
    if not alias_frames:
        return pd.DataFrame(columns=["state_abbr", "publication_name_exact_std", "jurisdiction_id"])
    out = pd.concat(alias_frames, ignore_index=True)
    out = out[out["publication_name_exact_std"].notna()].drop_duplicates().reset_index(drop=True)
    unique_counts = (
        out.groupby(["state_abbr", "publication_name_exact_std"], dropna=False)["jurisdiction_id"]
        .nunique()
        .rename("repo_match_count")
        .reset_index()
    )
    out = out.merge(unique_counts, on=["state_abbr", "publication_name_exact_std"], how="left")
    out = out[out["repo_match_count"].eq(1)].drop(columns=["repo_match_count"]).reset_index(drop=True)
    return out
