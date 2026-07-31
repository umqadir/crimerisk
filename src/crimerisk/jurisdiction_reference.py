from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crimerisk.paths import get_paths
from crimerisk.jurisdiction_review import _load_all_tiger_lookups
from crimerisk.reference import _std_text, matches_tribal_name


STATE_NAME_BY_FIPS: dict[str, str] = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "57": "Canal Zone",
    "66": "Guam",
    "72": "Puerto Rico",
    "78": "U.S. Virgin Islands",
}

STATE_ABBR_ALIAS_TO_FIPS: dict[str, str] = {
    "NB": "31",
    "GM": "66",
    "GU": "66",
    "VI": "78",
    "CZ": "57",
}

STATE_ABBR_BY_FIPS: dict[str, str] = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    "57": "CZ",
    "66": "GU",
    "72": "PR",
    "78": "VI",
}

_TRANSIT_KEYWORDS = (
    " TRANSIT",
    " METRO ",
    " BART",
    " RAIL",
    " SUBWAY",
    " RAPID TRANSIT",
)
_TRANSPORT_HUB_KEYWORDS = (
    " AIRPORT",
    " AVIATION",
    " HARBOR",
    " PORT AUTHORITY",
    " PORT ",
    " SEAPORT",
)
# Tribal overlap-SUBTYPE inference. The name test itself lives in `reference.matches_tribal_name`
# (word boundaries, one definition in the codebase) because the same tokens also form half the
# tribal-agency identity witness. Tribal ROUTING is driven by `reference.tribal_agency_flag`, not
# by a name: a name test is not a defensible identity witness (Stage 2 screen c measured 74 false
# hits / 49 misses for the substring form this replaced -- `" NATION"` matched NATIONAL SECURITY
# AGENCY, NATIONAL INSTITUTES OF HEALTH and, because the tribal branch below is tested BEFORE the
# statewide branch, every NATIONAL PARK force too, while `" INDIAN"` matched INDIANAPOLIS).
_matches_tribal_name = matches_tribal_name


_CAMPUS_KEYWORDS = (
    " UNIVERSITY",
    " COLLEGE",
    " CAMPUS",
    " SCHOOL",
    " TECHNICAL",
)
_LOCAL_SPECIAL_KEYWORDS = (
    " HOUSING",
    " PARK POLICE",
    " PARK PD",
    " PARKS AND REC",
    " PARK AUTHORITY",
)
_OTHER_SPECIAL_KEYWORDS = (
    " TASK FORCE",
    " NARCOTICS",
    " STRIKE FORCE",
    " DETECTIVE",
    " BUREAU",
)
_STATEWIDE_KEYWORDS = (
    " STATE TROOPER",
    " STATE POLICE",
    " DEPARTMENT OF PUBLIC SAFETY",
    " HIGHWAY POLICE",
    " HIGHWAY PATROL",
    " FBI",
    " DEA",
    " ATF",
    " BORDER PATROL",
    " CUSTOMS",
    " POSTAL INSPECTION",
    " NATIONAL PARK",
    " NATIONAL MONUMENT",
    " NATIONAL MEMORIAL",
    " NATIONAL HISTORIC",
    " NATIONAL RECREATION",
    " STATE PARK",
    " GAME FISH",
    " FISH AND GAME",
    " NATURAL RESOURCE",
    " WILDLIFE",
    " GAME COMMISSION",
    " ALCOHOLIC BEVERAGE",
    " LIQUEFIED",
    " U S CUSTOMS",
    " US CUSTOMS",
    " U S BORDER",
    " US BORDER",
)


@dataclass(frozen=True)
class ReferenceArtifacts:
    full_local: pd.DataFrame
    full_nonlocal: pd.DataFrame
    jurisdiction_master: pd.DataFrame
    agency_to_jurisdiction_crosswalk: pd.DataFrame


def _load_municipal_geometry_overrides(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "jurisdiction_id",
                "decision",
                "replacement_geo_type",
                "replacement_geoid",
                "replacement_jurisdiction_name",
                "reason",
                "sources",
            ]
        )
    if path.suffix.lower() == ".parquet":
        overrides = pd.read_parquet(path).copy()
    else:
        overrides = pd.read_csv(path).copy()
    required = {
        "jurisdiction_id",
        "decision",
        "replacement_geo_type",
        "replacement_geoid",
        "replacement_jurisdiction_name",
        "reason",
        "sources",
    }
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(f"Municipal geometry overrides missing columns: {sorted(missing)}")
    overrides["jurisdiction_id"] = overrides["jurisdiction_id"].astype("string")
    overrides["decision"] = overrides["decision"].astype("string")
    dupes = overrides["jurisdiction_id"].value_counts()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        raise ValueError(f"Duplicate jurisdiction_id in municipal geometry overrides: {dupes.to_dict()}")
    valid_decisions = {"map_to_place", "map_to_cousub", "exclude"}
    invalid = overrides.loc[~overrides["decision"].isin(valid_decisions), ["jurisdiction_id", "decision"]]
    if not invalid.empty:
        raise ValueError(f"Invalid municipal geometry override decisions: {invalid.to_dict(orient='records')}")
    mapped = overrides["decision"].isin(["map_to_place", "map_to_cousub"])
    if overrides.loc[mapped, "replacement_geo_type"].isna().any():
        raise ValueError("Mapped municipal geometry overrides must include replacement_geo_type")
    if overrides.loc[mapped, "replacement_geoid"].isna().any():
        raise ValueError("Mapped municipal geometry overrides must include replacement_geoid")
    if overrides.loc[mapped, "replacement_jurisdiction_name"].isna().any():
        raise ValueError("Mapped municipal geometry overrides must include replacement_jurisdiction_name")
    overrides["replacement_geo_type"] = overrides["replacement_geo_type"].astype("string").replace(
        {"county_subdivision": "cousub"}
    )

    def _normalize_override_geoid(geo_type: object, geoid: object) -> str | None:
        if pd.isna(geoid):
            return None
        text = str(geoid).strip()
        if "." in text:
            whole, frac = text.split(".", 1)
            if frac.strip("0") == "":
                text = whole
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return None
        if str(geo_type) == "place":
            return digits[-7:].zfill(7)
        if str(geo_type) == "cousub":
            return digits[-10:].zfill(10)
        return digits

    overrides["replacement_geoid"] = [
        _normalize_override_geoid(geo_type, geoid)
        for geo_type, geoid in zip(
            overrides["replacement_geo_type"],
            overrides["replacement_geoid"],
            strict=False,
        )
    ]
    return overrides


def _load_local_resolution_overrides(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "ori9",
                "decision",
                "replacement_geo_type",
                "replacement_geoid",
                "replacement_jurisdiction_name",
                "confidence",
                "source_note",
                "reviewer_note",
            ]
        )
    overrides = pd.read_csv(path).copy()
    required = {
        "ori",
        "decision",
        "replacement_geo_type",
        "replacement_geoid",
        "replacement_jurisdiction_name",
        "confidence",
        "source_note",
        "reviewer_note",
    }
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(f"Local resolution overrides missing columns: {sorted(missing)}")
    overrides["ori9"] = overrides["ori"].astype("string").str.strip()
    dupes = overrides["ori9"].value_counts()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        raise ValueError(f"Duplicate ori in local resolution overrides: {dupes.to_dict()}")
    valid_decisions = {"municipal_place", "municipal_cousub", "reclassify_nonmunicipal", "reclassify_overlap"}
    invalid = overrides.loc[~overrides["decision"].isin(valid_decisions), ["ori9", "decision"]]
    if not invalid.empty:
        raise ValueError(f"Invalid local resolution override decisions: {invalid.to_dict(orient='records')}")
    overrides["replacement_geo_type"] = overrides["replacement_geo_type"].astype("string").replace(
        {"county_subdivision": "cousub", "municipal_place": "place", "municipal_cousub": "cousub"}
    )

    def _normalize_override_geoid(geo_type: object, geoid: object) -> str | None:
        if pd.isna(geoid):
            return None
        text = str(geoid).strip()
        if "." in text:
            whole, frac = text.split(".", 1)
            if frac.strip("0") == "":
                text = whole
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return None
        if str(geo_type) == "place":
            return digits[-7:].zfill(7)
        if str(geo_type) == "cousub":
            return digits[-10:].zfill(10)
        return digits

    overrides["replacement_geoid"] = [
        _normalize_override_geoid(geo_type, geoid)
        for geo_type, geoid in zip(
            overrides["replacement_geo_type"],
            overrides["replacement_geoid"],
            strict=False,
        )
    ]

    municipal = overrides["decision"].isin(["municipal_place", "municipal_cousub"])
    if overrides.loc[municipal, "replacement_geo_type"].isna().any():
        raise ValueError("Municipal local-resolution overrides must include replacement_geo_type")
    if overrides.loc[municipal, "replacement_geoid"].isna().any():
        raise ValueError("Municipal local-resolution overrides must include replacement_geoid")
    if overrides.loc[municipal, "replacement_jurisdiction_name"].isna().any():
        raise ValueError("Municipal local-resolution overrides must include replacement_jurisdiction_name")
    return overrides


def _tribal_oris_from_agency_master(agency_master: pd.DataFrame) -> set[str]:
    """Tribal ORI set, preferring the flag the agency master already carries."""
    if "is_tribal_agency" in agency_master.columns:
        flag = agency_master["is_tribal_agency"].fillna(False).astype(bool)
        return {
            str(value).strip().upper()
            for value in agency_master.loc[flag, "ori9"]
            if pd.notna(value)
        }
    raise ValueError(
        "agency_master is missing is_tribal_agency; rebuild it (reference.write_agency_master) "
        "before building the reference layers -- the Class D gate depends on it."
    )


def _municipal_placement_licensed_oris(local_override_path: Path | None) -> set[str]:
    """ORIs a reviewed registry row is allowed to place on a municipal jurisdiction."""
    overrides = _load_local_resolution_overrides(local_override_path)
    if overrides.empty:
        return set()
    return {
        str(value).strip().upper()
        for value in overrides["ori9"]
        if pd.notna(value)
    }


def assert_tribal_agencies_not_auto_placed(
    full_local: pd.DataFrame,
    *,
    tribal_oris: set[str],
    licensed_oris: set[str],
) -> None:
    """FAIL CLOSED when an automatic lane places a tribal agency on a municipality.

    This is the Class D gate. LEAIC gives a tribal police department the FPLACE of its
    agency-seat CDP, and the automatic local lanes (`provisional_local_agency_matches`, and
    rung (b) of `_build_unassigned_fallbacks`) take a valid place_fips unconditionally -- so
    a whole reservation's crime lands on one small town. Measured 2026-07-29: 171 tribal
    links resolved exclusively to a municipal place/cousub, 169 of them `provisional_auto`
    (Stage 2 screen S2-3), producing e.g. Colville -> Nespelem town at 19.7x the national
    rate and, in the other direction, Seminole Tribal -> Hollywood city (751 counts
    invisibly diluted into a 153k city).

    The rule is therefore: a tribal agency may sit on a municipal jurisdiction ONLY because
    a reviewed registry row says so. `licensed_oris` is the union of the ORIs named in
    `configs/local_resolution_overrides.csv` and `configs/municipal_geometry_overrides.csv`;
    the licence is keyed on registry membership rather than on `resolution_source` because
    the PA/CDP canonicalization passes downstream rewrite that column.

    Everything else must be routed off the municipal lane (normally `reclassify_overlap`
    plus an AIANNH footprint in the overlap footprint registries).
    """
    if full_local.empty or not tribal_oris:
        return
    ori = full_local["ori9"].astype("string").str.strip().str.upper()
    municipal = full_local["final_decision"].astype("string").isin(
        ["municipal_place", "municipal_cousub"]
    )
    offending = full_local.loc[
        municipal & ori.isin(tribal_oris) & ~ori.isin(licensed_oris)
    ]
    if offending.empty:
        return
    detail = offending[["ori9", "agency_name_std", "final_decision", "resolved_geoid", "resolution_source"]]
    raise ValueError(
        "Tribal agencies placed on a municipal jurisdiction by an automatic lane "
        f"({len(offending)} ORIs). Each needs a reviewed row in "
        "configs/local_resolution_overrides.csv (reclassify_overlap plus an AIANNH "
        "footprint, or an explicit municipal_place pin with a reviewer_note saying why). "
        f"Offending rows: {detail.to_dict(orient='records')}"
    )


def _apply_local_resolution_overrides(frame: pd.DataFrame, override_path: Path | None) -> pd.DataFrame:
    overrides = _load_local_resolution_overrides(override_path)
    if overrides.empty:
        return frame
    merged = frame.merge(
        overrides[
            [
                "ori9",
                "decision",
                "replacement_geo_type",
                "replacement_geoid",
                "replacement_jurisdiction_name",
            ]
        ],
        on="ori9",
        how="left",
    )
    has_override = merged["decision"].notna()
    if not has_override.any():
        return frame
    municipal = has_override & merged["decision"].isin(["municipal_place", "municipal_cousub"])
    if municipal.any():
        merged.loc[municipal, "resolved_geo_type"] = merged.loc[municipal, "replacement_geo_type"]
        merged.loc[municipal, "resolved_geoid"] = merged.loc[municipal, "replacement_geoid"]
        merged.loc[municipal, "resolved_label"] = merged.loc[municipal, "replacement_jurisdiction_name"]
        merged.loc[municipal, "has_final_municipal_geoid"] = True
    nonmunicipal = has_override & merged["decision"].eq("reclassify_nonmunicipal")
    if nonmunicipal.any():
        merged.loc[nonmunicipal, "resolved_geo_type"] = None
        merged.loc[nonmunicipal, "resolved_geoid"] = None
        merged.loc[nonmunicipal, "resolved_label"] = None
        merged.loc[nonmunicipal, "has_final_municipal_geoid"] = False
    overlap = has_override & merged["decision"].eq("reclassify_overlap")
    if overlap.any():
        merged.loc[overlap, "resolved_geo_type"] = None
        merged.loc[overlap, "resolved_geoid"] = None
        merged.loc[overlap, "resolved_label"] = None
        merged.loc[overlap, "has_final_municipal_geoid"] = False
    merged.loc[has_override, "final_decision"] = merged.loc[has_override, "decision"]
    merged.loc[has_override, "manual_review_flag"] = True
    merged.loc[has_override, "resolution_source"] = "local_resolution_override"
    merged.loc[has_override, "fallback_applied"] = False
    merged.loc[has_override, "match_method"] = "local_resolution_override"
    return merged.drop(
        columns=[
            "decision",
            "replacement_geo_type",
            "replacement_geoid",
            "replacement_jurisdiction_name",
        ],
        errors="ignore",
    )


def _apply_nonlocal_resolution_overrides(frame: pd.DataFrame, override_path: Path | None) -> pd.DataFrame:
    overrides = _load_local_resolution_overrides(override_path)
    if overrides.empty:
        return frame
    merged = frame.merge(
        overrides[
            [
                "ori9",
                "decision",
                "replacement_geo_type",
                "replacement_geoid",
                "replacement_jurisdiction_name",
            ]
        ],
        on="ori9",
        how="left",
    )
    has_override = merged["decision"].notna()
    if not has_override.any():
        return frame

    municipal = has_override & merged["decision"].isin(["municipal_place", "municipal_cousub"])
    if municipal.any():
        merged.loc[municipal, "final_bucket_decision"] = "municipal_exclusive"
        merged.loc[municipal, "final_geometry_hint"] = merged.loc[municipal, "replacement_geo_type"]
        merged.loc[municipal, "final_overlap_subtype"] = None
        merged.loc[municipal, "override_geo_type"] = merged.loc[municipal, "replacement_geo_type"]
        merged.loc[municipal, "override_geoid"] = merged.loc[municipal, "replacement_geoid"]
        merged.loc[municipal, "override_label"] = merged.loc[municipal, "replacement_jurisdiction_name"]

    nonmunicipal = has_override & merged["decision"].eq("reclassify_nonmunicipal")
    if nonmunicipal.any():
        merged.loc[nonmunicipal, "final_bucket_decision"] = "nonmunicipal_exclusive"
        merged.loc[nonmunicipal, "final_geometry_hint"] = "county_remainder"
        merged.loc[nonmunicipal, "final_overlap_subtype"] = None

    overlap = has_override & merged["decision"].eq("reclassify_overlap")
    if overlap.any():
        merged.loc[overlap, "final_bucket_decision"] = "statewide_overlap"
        merged.loc[overlap, "final_geometry_hint"] = None
        merged.loc[overlap, "final_overlap_subtype"] = None

    merged["confidence"] = merged["confidence"].astype("string")
    merged["reason"] = merged["reason"].astype("string")
    merged.loc[has_override, "manual_review_flag"] = True
    merged.loc[has_override, "resolution_source"] = "local_resolution_override"
    merged.loc[has_override, "confidence"] = merged.loc[has_override, "confidence"].fillna("high")
    merged.loc[has_override, "reason"] = merged.loc[has_override, "reason"].fillna(
        "Promoted reviewed local-resolution override into nonlocal routing."
    )
    return merged.drop(
        columns=[
            "decision",
            "replacement_geo_type",
            "replacement_geoid",
            "replacement_jurisdiction_name",
        ],
        errors="ignore",
    )


def _build_state_map(agency_master: pd.DataFrame) -> dict[str, str]:
    state_map: dict[str, str] = {}
    valid = agency_master[["state_fips", "state_abbr"]].dropna().copy()
    valid["state_fips"] = valid["state_fips"].astype(str).str.zfill(2)
    valid["state_abbr"] = valid["state_abbr"].astype(str).str.upper()
    valid = valid[valid["state_abbr"].isin(STATE_ABBR_BY_FIPS.values())]
    for row in valid.itertuples(index=False):
        state_map[row.state_abbr] = row.state_fips
    state_map.update(STATE_ABBR_ALIAS_TO_FIPS)
    return state_map


def _normalize_state_fields(
    df: pd.DataFrame,
    *,
    state_map: dict[str, str],
    geoid_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "state_abbr" in out.columns:
        out["state_abbr"] = out["state_abbr"].astype(str).str.upper()
        out.loc[out["state_abbr"].isin(["NONE", "NAN"]), "state_abbr"] = None
    if "state_fips" in out.columns:
        out["state_fips"] = out["state_fips"].map(
            lambda v: None if pd.isna(v) or str(v).strip().lower() in {"none", "nan", ""} else str(v).zfill(2)
        )
    if geoid_col and geoid_col in out.columns:
        geoid_mask = out["state_fips"].isna() & out[geoid_col].notna()
        out.loc[geoid_mask, "state_fips"] = out.loc[geoid_mask, geoid_col].astype(str).str[:2]
    if "state_abbr" in out.columns and "state_fips" in out.columns:
        canonical_fips = out["state_abbr"].map(state_map)
        abbr_known = out["state_abbr"].notna() & canonical_fips.notna()
        out.loc[abbr_known, "state_fips"] = canonical_fips.loc[abbr_known]
        fill_fips = out["state_fips"].isna() & out["state_abbr"].notna()
        out.loc[fill_fips, "state_fips"] = out.loc[fill_fips, "state_abbr"].map(state_map)
        fill_abbr = out["state_abbr"].isna() & out["state_fips"].notna()
        out.loc[fill_abbr, "state_abbr"] = out.loc[fill_abbr, "state_fips"].map(STATE_ABBR_BY_FIPS)
        alias_fix = out["state_abbr"].isin(STATE_ABBR_ALIAS_TO_FIPS)
        out.loc[alias_fix, "state_fips"] = out.loc[alias_fix, "state_abbr"].map(state_map)
        out.loc[alias_fix, "state_abbr"] = out.loc[alias_fix, "state_fips"].map(STATE_ABBR_BY_FIPS)
    return out


def _canonicalize_municipal_geoid_prefix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not {"resolved_geo_type", "resolved_geoid", "state_fips"}.issubset(out.columns):
        return out
    municipal = out["resolved_geo_type"].isin(["place", "cousub"]) & out["resolved_geoid"].notna() & out["state_fips"].notna()
    geoid_text = out["resolved_geoid"].astype(str)
    digit_mask = municipal & geoid_text.str.fullmatch(r"\d+").fillna(False)
    mismatch = digit_mask & geoid_text.str[:2].ne(out["state_fips"])
    out.loc[mismatch, "resolved_geoid"] = (
        out.loc[mismatch, "state_fips"].astype(str) + geoid_text.loc[mismatch].str[2:]
    )
    return out


def _drop_invalid_ori_rows(df: pd.DataFrame, *, ori_col: str) -> pd.DataFrame:
    out = df.copy()
    text = out[ori_col].astype(str).str.strip().str.upper()
    keep = out[ori_col].notna() & ~text.isin(["", "NONE", "NAN", "<NA>"])
    return out.loc[keep].copy()


def _valid_place_fips(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return False
    text = text.zfill(5)
    return text.isdigit() and text != "00000" and not text.startswith("99")


def _build_tiger_exact_lookups() -> tuple[dict[tuple[str, str], tuple[str, str]], dict[tuple[str, str, str], tuple[str, str]]]:
    tiger = _load_all_tiger_lookups(paths=get_paths()).copy()
    if tiger.empty:
        return {}, {}

    tiger["resolved_geo_type"] = tiger["geo_type"].replace({"county_subdivision": "cousub"})
    tiger["state_fips"] = tiger["state_fips"].astype(str).str.zfill(2)
    tiger["geoid"] = tiger["geoid"].astype(str)
    tiger["namelsad_std"] = tiger["namelsad"].map(_std_text)
    if "name" in tiger.columns:
        tiger["name_std"] = tiger["name"].map(_std_text)
    else:
        tiger["name_std"] = tiger["namelsad_std"]

    place_by_geoid: dict[tuple[str, str], tuple[str, str]] = {}
    for row in tiger[tiger["resolved_geo_type"].eq("place")].drop_duplicates(["state_fips", "geoid"]).itertuples(index=False):
        place_by_geoid[(row.state_fips, row.geoid)] = (row.geoid, row.namelsad)

    exact_name_lookup: dict[tuple[str, str, str], tuple[str, str]] = {}
    for geo_type in ["place", "cousub"]:
        subset = tiger[tiger["resolved_geo_type"].eq(geo_type)].copy()
        if subset.empty:
            continue
        label_keys = pd.concat(
            [
                subset[["state_fips", "resolved_geo_type", "geoid", "namelsad", "namelsad_std"]].rename(
                    columns={"namelsad_std": "lookup_key"}
                ),
                subset[["state_fips", "resolved_geo_type", "geoid", "namelsad", "name_std"]].rename(
                    columns={"name_std": "lookup_key"}
                ),
            ],
            ignore_index=True,
        ).dropna(subset=["lookup_key"]).drop_duplicates()
        unique_keys = (
            label_keys.groupby(["state_fips", "resolved_geo_type", "lookup_key"])["geoid"]
            .nunique()
            .rename("n")
            .reset_index()
        )
        label_keys = label_keys.merge(
            unique_keys[unique_keys["n"] == 1],
            on=["state_fips", "resolved_geo_type", "lookup_key"],
            how="inner",
        )
        for row in label_keys.drop_duplicates(["state_fips", "resolved_geo_type", "lookup_key"]).itertuples(index=False):
            exact_name_lookup[(row.state_fips, row.resolved_geo_type, row.lookup_key)] = (row.geoid, row.namelsad)

    return place_by_geoid, exact_name_lookup


def _text_candidates(row: pd.Series) -> list[str]:
    candidates: list[str] = []
    for col in ["agency_name_std", "agency_name_raw", "crosswalk_agency_name_std", "census_name_std"]:
        text = _std_text(row.get(col))
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def _infer_special_bucket(text_candidates: list[str], agency_type_norm: str | None) -> tuple[str, str | None, str | None]:
    text = " | ".join(f" {value} " for value in text_candidates)
    if any(keyword in text for keyword in _TRANSPORT_HUB_KEYWORDS):
        return "localized_special_overlap", "transport_hub", "facility_or_authority_footprint"
    if any(keyword in text for keyword in _TRANSIT_KEYWORDS):
        return "localized_special_overlap", "transit", "network_or_system_footprint"
    if _matches_tribal_name(text):
        return "localized_special_overlap", "tribal", "tribal_or_reservation_footprint"
    if any(keyword in text for keyword in _CAMPUS_KEYWORDS):
        return "localized_special_overlap", "campus", "campus_footprint"
    if any(keyword in text for keyword in _LOCAL_SPECIAL_KEYWORDS):
        return "localized_special_overlap", "local_special", "facility_or_authority_footprint"
    if any(keyword in text for keyword in _OTHER_SPECIAL_KEYWORDS):
        return "localized_special_overlap", "other_special", "network_or_multiagency_footprint"
    if agency_type_norm == "state_law_enforcement" or any(keyword in text for keyword in _STATEWIDE_KEYWORDS):
        return "statewide_overlap", None, "statewide"
    if agency_type_norm == "special_jurisdiction":
        return "statewide_overlap", None, "statewide"
    return "statewide_overlap", None, "statewide"


def _resolve_unassigned_municipal(
    row: pd.Series,
    *,
    place_by_geoid: dict[tuple[str, str], tuple[str, str]],
    exact_name_lookup: dict[tuple[str, str, str], tuple[str, str]],
) -> tuple[str, str, str] | None:
    state_fips = row.get("state_fips")
    if pd.isna(state_fips) or not state_fips:
        return None
    state_fips = str(state_fips).zfill(2)

    place_fips = row.get("place_fips")
    if _valid_place_fips(place_fips):
        geoid = state_fips + str(place_fips).zfill(5)
        match = place_by_geoid.get((state_fips, geoid))
        if match is not None:
            return ("place", match[0], match[1])

    candidates = _text_candidates(row)
    state_abbr = str(row.get("state_abbr") or "").upper()
    local_terms = (" TOWNSHIP", " BOROUGH", " TOWN", " CITY", " VILLAGE")
    for text in candidates:
        if state_abbr == "PA" and any(term in text for term in local_terms):
            match = exact_name_lookup.get((state_fips, "cousub", text))
            if match is not None:
                return ("cousub", match[0], match[1])
        match = exact_name_lookup.get((state_fips, "place", text))
        if match is not None:
            return ("place", match[0], match[1])
        if state_abbr == "PA":
            match = exact_name_lookup.get((state_fips, "cousub", text))
            if match is not None:
                return ("cousub", match[0], match[1])
    return None


def _build_unassigned_fallbacks(
    *,
    agency_master: pd.DataFrame,
    full_local: pd.DataFrame,
    full_nonlocal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned_oris = set(full_local["ori9"]) | set(full_nonlocal["ori9"])
    unassigned = agency_master[~agency_master["ori9"].isin(assigned_oris)].copy()
    if unassigned.empty:
        return pd.DataFrame(columns=full_local.columns), pd.DataFrame(columns=full_nonlocal.columns)

    state_map = _build_state_map(agency_master)
    unassigned = _normalize_state_fields(unassigned, state_map=state_map)
    place_by_geoid, exact_name_lookup = _build_tiger_exact_lookups()

    local_rows: list[dict[str, object]] = []
    nonlocal_rows: list[dict[str, object]] = []
    for row in unassigned.to_dict(orient="records"):
        row_series = pd.Series(row)
        agency_type_norm = str(row.get("agency_type_norm") or "unknown")

        if agency_type_norm == "sheriff":
            nonlocal_rows.append(
                {
                    "ori9": row.get("ori9"),
                    "ori7": row.get("ori7"),
                    "state_fips": row.get("state_fips"),
                    "state_abbr": row.get("state_abbr"),
                    "agency_name_raw": row.get("agency_name_raw"),
                    "agency_name_std": row.get("agency_name_std"),
                    "agency_type_norm": agency_type_norm,
                    "latest_srs_part1_total": row.get("latest_srs_part1_total"),
                    "manual_review_flag": True,
                    "final_bucket_decision": "nonmunicipal_exclusive",
                    "final_overlap_subtype": None,
                    "final_geometry_hint": "county_remainder",
                    "resolution_source": "agency_master_fallback",
                    "confidence": None,
                    "reason": "Fallback county-remainder assignment for previously unassigned sheriff-style agency.",
                    "sources": None,
                    "needs_high_review": False,
                }
            )
            continue

        texts = _text_candidates(row_series)
        joined_text = " | ".join(f" {value} " for value in texts)
        explicit_special = (
            agency_type_norm in {"special_jurisdiction", "state_law_enforcement"}
            # The Class D gate at rung (b): a LEAIC/roster-identified tribal agency never
            # takes the agency-seat place shortcut. Without this the shortcut fires before
            # the tribal branch of `_infer_special_bucket` is reachable at all.
            or bool(row.get("is_tribal_agency") is True)
            or _matches_tribal_name(joined_text)
            or any(
                keyword in joined_text
                for keyword in (
                    _TRANSPORT_HUB_KEYWORDS
                    + _TRANSIT_KEYWORDS
                    + _CAMPUS_KEYWORDS
                    + _LOCAL_SPECIAL_KEYWORDS
                    + _OTHER_SPECIAL_KEYWORDS
                    + _STATEWIDE_KEYWORDS
                )
            )
        )
        municipal = None
        if not explicit_special:
            municipal = _resolve_unassigned_municipal(
                row_series,
                place_by_geoid=place_by_geoid,
                exact_name_lookup=exact_name_lookup,
            )
        if municipal is not None:
            geo_type, geoid, label = municipal
            local_rows.append(
                {
                    "ori9": row.get("ori9"),
                    "ori7": row.get("ori7"),
                    "state_fips": row.get("state_fips"),
                    "state_abbr": row.get("state_abbr"),
                    "agency_name_raw": row.get("agency_name_raw"),
                    "agency_name_std": row.get("agency_name_std"),
                    "agency_type_norm": agency_type_norm,
                    "final_decision": "municipal_place" if geo_type == "place" else "municipal_cousub",
                    "resolved_geo_type": geo_type,
                    "resolved_geoid": geoid,
                    "resolved_label": label,
                    "manual_review_flag": True,
                    "resolution_source": "agency_master_fallback",
                    "fallback_applied": True,
                    "has_final_municipal_geoid": True,
                    "match_method": "agency_master_fallback",
                }
            )
            continue

        is_local_like = agency_type_norm in {"local_police", "constable_marshal"}
        special_bucket, special_subtype, geometry_hint = _infer_special_bucket(texts, agency_type_norm)

        if is_local_like and special_subtype is None and special_bucket == "statewide_overlap":
            local_rows.append(
                {
                    "ori9": row.get("ori9"),
                    "ori7": row.get("ori7"),
                    "state_fips": row.get("state_fips"),
                    "state_abbr": row.get("state_abbr"),
                    "agency_name_raw": row.get("agency_name_raw"),
                    "agency_name_std": row.get("agency_name_std"),
                    "agency_type_norm": agency_type_norm,
                    "final_decision": "reclassify_nonmunicipal",
                    "resolved_geo_type": None,
                    "resolved_geoid": None,
                    "resolved_label": None,
                    "manual_review_flag": True,
                    "resolution_source": "agency_master_fallback",
                    "fallback_applied": True,
                    "has_final_municipal_geoid": False,
                    "match_method": "agency_master_fallback",
                }
            )
            continue

        nonlocal_rows.append(
            {
                "ori9": row.get("ori9"),
                "ori7": row.get("ori7"),
                "state_fips": row.get("state_fips"),
                "state_abbr": row.get("state_abbr"),
                "agency_name_raw": row.get("agency_name_raw"),
                "agency_name_std": row.get("agency_name_std"),
                "agency_type_norm": agency_type_norm,
                "latest_srs_part1_total": row.get("latest_srs_part1_total"),
                "manual_review_flag": True,
                "final_bucket_decision": special_bucket,
                "final_overlap_subtype": special_subtype,
                "final_geometry_hint": geometry_hint,
                "resolution_source": "agency_master_fallback",
                "confidence": None,
                "reason": "Fallback overlap assignment for previously unassigned agency outside the resolved local or sheriff/default pools.",
                "sources": None,
                "needs_high_review": False,
            }
        )

    fallback_local = pd.DataFrame(local_rows)
    fallback_nonlocal = pd.DataFrame(nonlocal_rows)
    for col in full_local.columns:
        if col not in fallback_local.columns:
            fallback_local[col] = None
    for col in full_nonlocal.columns:
        if col not in fallback_nonlocal.columns:
            fallback_nonlocal[col] = None
    return fallback_local[full_local.columns], fallback_nonlocal[full_nonlocal.columns]


def _enrich_from_agency_master(
    resolved_df: pd.DataFrame,
    *,
    agency_master: pd.DataFrame,
) -> pd.DataFrame:
    if resolved_df.empty or agency_master.empty:
        return resolved_df

    lookup_cols = [
        "ori9",
        "ori7",
        "state_fips",
        "state_abbr",
        "agency_name_raw",
        "agency_name_std",
        "agency_type_norm",
    ]
    lookup = (
        agency_master[lookup_cols]
        .drop_duplicates(subset=["ori9"], keep="last")
        .set_index("ori9")
    )
    out = resolved_df.copy()
    for col in lookup_cols[1:]:
        if col not in out.columns:
            out[col] = None
        mapped = out["ori9"].map(lookup[col])
        if col == "agency_type_norm":
            missing_mask = out[col].isna() | out[col].astype("string").eq("unknown")
        else:
            missing_mask = out[col].isna()
        updates = mapped.loc[missing_mask & mapped.notna()]
        if updates.empty:
            continue
        out.loc[updates.index, col] = updates
    return out


def _build_full_local_resolution(
    *,
    provisional_local_path: Path,
    resolved_local_tail_path: Path,
    agency_master_path: Path,
    local_override_path: Path | None = None,
) -> pd.DataFrame:
    agency_master = pd.read_parquet(agency_master_path)
    state_map = _build_state_map(agency_master)

    provisional = pd.read_parquet(provisional_local_path).copy()
    provisional = _drop_invalid_ori_rows(provisional, ori_col="ori9")
    provisional = _normalize_state_fields(provisional, state_map=state_map, geoid_col="geoid")
    provisional["resolved_geo_type"] = provisional["geo_type"].replace({"county_subdivision": "cousub"})
    provisional["resolved_geoid"] = provisional["geoid"].astype(str)
    provisional["resolved_label"] = provisional["namelsad"]
    provisional["final_decision"] = provisional["resolved_geo_type"].map(
        {"place": "municipal_place", "cousub": "municipal_cousub"}
    )
    provisional["resolution_source"] = "provisional_auto"
    provisional["manual_review_flag"] = provisional["match_method"].eq("place_preferred")
    provisional["fallback_applied"] = False
    provisional["has_final_municipal_geoid"] = True
    provisional = _canonicalize_municipal_geoid_prefix(provisional)
    keep_cols = [
        "ori9",
        "ori7",
        "state_fips",
        "state_abbr",
        "agency_name_raw",
        "agency_name_std",
        "agency_type_norm",
        "final_decision",
        "resolved_geo_type",
        "resolved_geoid",
        "resolved_label",
        "manual_review_flag",
        "resolution_source",
        "fallback_applied",
        "has_final_municipal_geoid",
        "match_method",
    ]
    for col in keep_cols:
        if col not in provisional.columns:
            provisional[col] = None
    provisional = provisional[
        keep_cols
    ].copy()

    resolved_tail = pd.read_parquet(resolved_local_tail_path).copy()
    resolved_tail = _drop_invalid_ori_rows(resolved_tail, ori_col="ori9")
    resolved_tail = _normalize_state_fields(
        resolved_tail,
        state_map=state_map,
        geoid_col="resolved_geoid",
    )
    resolved_tail["resolved_geo_type"] = resolved_tail["resolved_geo_type"].replace(
        {"municipal_place": "place", "municipal_cousub": "cousub", "county_subdivision": "cousub"}
    )
    keep_cols = [
        "ori9",
        "ori7",
        "state_fips",
        "state_abbr",
        "agency_name_raw",
        "agency_name_std",
        "agency_type_norm",
        "final_decision",
        "resolved_geo_type",
        "resolved_geoid",
        "resolved_label",
        "manual_review_flag",
        "resolution_source",
        "fallback_applied",
        "has_final_municipal_geoid",
        "match_method",
    ]
    for col in keep_cols:
        if col not in resolved_tail.columns:
            resolved_tail[col] = None
    resolved_tail = _canonicalize_municipal_geoid_prefix(resolved_tail)
    resolved_tail = resolved_tail[keep_cols].copy()

    out = pd.concat(
        [
            provisional[~provisional["ori9"].isin(set(resolved_tail["ori9"]))],
            resolved_tail,
        ],
        ignore_index=True,
    )
    out = _apply_local_resolution_overrides(out, local_override_path)
    out = _reconcile_local_municipal_geography(out)
    out = out.sort_values("ori9").reset_index(drop=True)
    return out


def _label_key(label: object, state_name: str | None, state_abbr: str | None) -> str | None:
    text = _std_text(label)
    if not text:
        return None
    for suffix in filter(None, [state_name, state_abbr]):
        suffix_std = _std_text(suffix)
        if not suffix_std:
            continue
        if text.endswith(" " + suffix_std):
            text = text[: -(len(suffix_std) + 1)].strip()
    return text


def _reconcile_local_municipal_geography(full_local: pd.DataFrame) -> pd.DataFrame:
    paths = get_paths()
    tiger = _load_all_tiger_lookups(paths=paths).copy()
    if tiger.empty:
        return full_local

    tiger["resolved_geo_type"] = tiger["geo_type"].replace({"county_subdivision": "cousub"})
    tiger["canonical_label"] = tiger["namelsad"]
    tiger["canonical_label_std"] = tiger["namelsad"].map(_std_text)
    tiger["label_key"] = [
        _label_key(label, STATE_NAME_BY_FIPS.get(state_fips), STATE_ABBR_BY_FIPS.get(state_fips))
        for label, state_fips in zip(tiger["namelsad"], tiger["state_fips"], strict=False)
    ]
    unique_label_lookup = (
        tiger.groupby(["state_fips", "resolved_geo_type", "label_key"], dropna=False)["geoid"]
        .nunique()
        .rename("n")
        .reset_index()
    )
    unique_label_lookup = unique_label_lookup[unique_label_lookup["n"] == 1].drop(columns="n")
    unique_label_lookup = unique_label_lookup.merge(
        tiger[
            [
                "state_fips",
                "resolved_geo_type",
                "label_key",
                "geoid",
                "canonical_label",
            ]
        ].drop_duplicates(),
        on=["state_fips", "resolved_geo_type", "label_key"],
        how="left",
    )
    geoid_lookup = tiger[
        ["state_fips", "resolved_geo_type", "geoid", "canonical_label", "canonical_label_std"]
    ].drop_duplicates()

    out = full_local.copy()
    municipal = out["final_decision"].isin(["municipal_place", "municipal_cousub"])
    out["resolved_label_key"] = [
        _label_key(label, STATE_NAME_BY_FIPS.get(state_fips), state_abbr)
        for label, state_fips, state_abbr in zip(
            out["resolved_label"], out["state_fips"], out["state_abbr"], strict=False
        )
    ]

    out = out.merge(
        geoid_lookup.rename(
            columns={
                "canonical_label": "canonical_label_current",
                "canonical_label_std": "canonical_label_std_current",
            }
        ),
        left_on=["state_fips", "resolved_geo_type", "resolved_geoid"],
        right_on=["state_fips", "resolved_geo_type", "geoid"],
        how="left",
    )
    out = out.merge(
        unique_label_lookup.rename(
            columns={
                "geoid": "label_match_geoid",
                "canonical_label": "label_match_label",
            }
        ),
        left_on=["state_fips", "resolved_geo_type", "resolved_label_key"],
        right_on=["state_fips", "resolved_geo_type", "label_key"],
        how="left",
    )

    mismatch = municipal & out["label_match_geoid"].notna() & out["label_match_geoid"].ne(out["resolved_geoid"])
    out.loc[mismatch, "resolved_geoid"] = out.loc[mismatch, "label_match_geoid"]
    out.loc[mismatch, "resolved_label"] = out.loc[mismatch, "label_match_label"]

    out = out.drop(columns=["geoid", "label_key"], errors="ignore")
    out = out.merge(
        geoid_lookup.rename(
            columns={
                "canonical_label": "canonical_label_final",
                "canonical_label_std": "canonical_label_std_final",
            }
        ),
        left_on=["state_fips", "resolved_geo_type", "resolved_geoid"],
        right_on=["state_fips", "resolved_geo_type", "geoid"],
        how="left",
    )
    canonical_fill = municipal & out["canonical_label_final"].notna()
    out.loc[canonical_fill, "resolved_label"] = out.loc[canonical_fill, "canonical_label_final"]
    return out.drop(
        columns=[
            "resolved_label_key",
            "canonical_label_current",
            "canonical_label_std_current",
            "label_match_geoid",
            "label_match_label",
            "canonical_label_final",
            "canonical_label_std_final",
            "geoid",
        ],
        errors="ignore",
    )


def _build_full_nonlocal_resolution(
    *,
    nonlocal_auto_path: Path,
    nonlocal_resolved_path: Path,
    agency_master_path: Path,
) -> pd.DataFrame:
    agency_master = pd.read_parquet(agency_master_path)
    state_map = _build_state_map(agency_master)

    auto_df = pd.read_parquet(nonlocal_auto_path).copy()
    auto_df = _drop_invalid_ori_rows(auto_df, ori_col="ori9")
    auto_df = _normalize_state_fields(auto_df, state_map=state_map)
    auto_df["final_bucket_decision"] = auto_df["suggested_bucket_class"]
    auto_df["final_overlap_subtype"] = None
    auto_df["final_geometry_hint"] = auto_df["final_bucket_decision"].map(
        {
            "nonmunicipal_exclusive": "county_remainder",
            "statewide_overlap": "statewide",
        }
    )
    auto_df["resolution_source"] = "auto_default"
    auto_df["confidence"] = None
    auto_df["reason"] = auto_df["final_bucket_decision"].map(
        {
            "nonmunicipal_exclusive": "Sheriff-style or countywide general-purpose agency routed to the state nonmunicipal remainder.",
            "statewide_overlap": "State law-enforcement agency routed to the statewide overlap layer.",
        }
    )
    auto_df["sources"] = None
    auto_df["needs_high_review"] = False

    keep_cols = [
        "ori9",
        "ori7",
        "state_fips",
        "state_abbr",
        "agency_name_raw",
        "agency_name_std",
        "agency_type_norm",
        "latest_srs_part1_total",
        "manual_review_flag",
        "final_bucket_decision",
        "final_overlap_subtype",
        "final_geometry_hint",
        "resolution_source",
        "confidence",
        "reason",
        "sources",
        "needs_high_review",
    ]
    for col in keep_cols:
        if col not in auto_df.columns:
            auto_df[col] = None
    auto_df = auto_df[keep_cols].copy()

    resolved = pd.read_parquet(nonlocal_resolved_path).copy()
    resolved = _drop_invalid_ori_rows(resolved, ori_col="ori9")
    resolved = _normalize_state_fields(resolved, state_map=state_map)
    for col in keep_cols:
        if col not in resolved.columns:
            resolved[col] = None
    resolved = resolved[keep_cols].copy()

    out = pd.concat(
        [
            auto_df[~auto_df["ori9"].isin(set(resolved["ori9"]))],
            resolved,
        ],
        ignore_index=True,
    )
    out = out.sort_values("ori9").reset_index(drop=True)
    return out


def _group_municipal_jurisdiction_rows(municipal: pd.DataFrame) -> pd.DataFrame:
    if municipal.empty:
        return pd.DataFrame(
            columns=[
                "jurisdiction_id",
                "jurisdiction_type",
                "state_fips",
                "state_abbr",
                "jurisdiction_name",
                "geometry_source",
                "is_contracted_place",
                "manual_review_flag",
                "geo_type",
                "geoid",
                "agency_count",
            ]
        )
    grouped = (
        municipal.groupby(
            ["state_fips", "state_abbr", "resolved_geo_type", "resolved_geoid", "resolved_label"],
            dropna=False,
            as_index=False,
        )
        .agg(
            manual_review_flag=("manual_review_flag", "max"),
            is_contracted_place=("is_contracted_place", "max"),
            geometry_source=("geometry_source", lambda s: next((x for x in s if pd.notna(x)), None)),
            agency_count=("ori9", "nunique"),
        )
    )
    grouped["jurisdiction_id"] = (
        grouped["state_fips"].astype(str)
        + ":municipal:"
        + grouped["resolved_geo_type"].astype(str)
        + ":"
        + grouped["resolved_geoid"].astype(str)
    )
    grouped["jurisdiction_type"] = "municipal"
    grouped["jurisdiction_name"] = grouped["resolved_label"]
    grouped["geo_type"] = grouped["resolved_geo_type"]
    grouped["geoid"] = grouped["resolved_geoid"]
    return grouped[
        [
            "jurisdiction_id",
            "jurisdiction_type",
            "state_fips",
            "state_abbr",
            "jurisdiction_name",
            "geometry_source",
            "is_contracted_place",
            "manual_review_flag",
            "geo_type",
            "geoid",
            "agency_count",
        ]
    ]


def _municipal_jurisdiction_rows(full_local: pd.DataFrame) -> pd.DataFrame:
    municipal = full_local[
        full_local["final_decision"].isin(["municipal_place", "municipal_cousub"])
    ].copy()
    municipal["jurisdiction_type"] = "municipal"
    municipal["geometry_source"] = municipal["resolved_geo_type"].map(
        {"place": "tiger_2020_place", "cousub": "tiger_2020_cousub"}
    )
    municipal["is_contracted_place"] = False
    return _group_municipal_jurisdiction_rows(municipal)


def _nonlocal_municipal_jurisdiction_rows(full_nonlocal: pd.DataFrame) -> pd.DataFrame:
    municipal = full_nonlocal[full_nonlocal["final_bucket_decision"].eq("municipal_exclusive")].copy()
    if municipal.empty:
        return _group_municipal_jurisdiction_rows(municipal)
    municipal["resolved_geo_type"] = municipal["override_geo_type"]
    municipal["resolved_geoid"] = municipal["override_geoid"]
    municipal["resolved_label"] = municipal["override_label"]
    municipal["geometry_source"] = municipal["resolved_geo_type"].map(
        {"place": "tiger_2020_place", "cousub": "tiger_2020_cousub"}
    )
    municipal["is_contracted_place"] = False
    return _group_municipal_jurisdiction_rows(municipal)


def _canonicalize_pa_equivalent_place_cousub_rows(full_local: pd.DataFrame) -> pd.DataFrame:
    out = full_local.copy()
    municipal = out["final_decision"].isin(["municipal_place", "municipal_cousub"])
    muni = out.loc[municipal].copy()
    if muni.empty:
        return out

    muni["state_abbr"] = muni["state_abbr"].astype("string").str.upper()
    muni = muni[muni["state_abbr"].eq("PA")].copy()
    if muni.empty:
        return out

    muni["resolved_geo_type"] = muni["resolved_geo_type"].astype("string")
    muni["resolved_geoid"] = muni["resolved_geoid"].astype("string")
    muni["name_root"] = muni["resolved_label"].map(_std_text)
    muni["local_code"] = muni["resolved_geoid"].str[-5:]

    paired = (
        muni.groupby(["name_root", "local_code"], dropna=False)
        .agg(
            place_geoid=("resolved_geoid", lambda s: next((x for x, gt in zip(s, muni.loc[s.index, "resolved_geo_type"], strict=False) if gt == "place"), None)),
            place_label=("resolved_label", lambda s: next((x for x, gt in zip(s, muni.loc[s.index, "resolved_geo_type"], strict=False) if gt == "place"), None)),
            cousub_geoid=("resolved_geoid", lambda s: next((x for x, gt in zip(s, muni.loc[s.index, "resolved_geo_type"], strict=False) if gt == "cousub"), None)),
            cousub_label=("resolved_label", lambda s: next((x for x, gt in zip(s, muni.loc[s.index, "resolved_geo_type"], strict=False) if gt == "cousub"), None)),
            place_rows=("resolved_geo_type", lambda s: int((s == "place").sum())),
            cousub_rows=("resolved_geo_type", lambda s: int((s == "cousub").sum())),
        )
        .reset_index()
    )
    remap: dict[tuple[str | None, str | None], tuple[str, str, str]] = {
        (row.name_root, row.local_code): (row.cousub_geoid, row.cousub_label, "pa_equivalent_cousub_canonicalized")
        for row in paired[
            paired["place_rows"].gt(0)
            & paired["cousub_rows"].gt(0)
            & paired["place_geoid"].notna()
            & paired["cousub_geoid"].notna()
        ].itertuples(index=False)
    }
    out["resolved_geo_type"] = out["resolved_geo_type"].astype("string")
    out["resolved_geoid"] = out["resolved_geoid"].astype("string")
    out["state_abbr"] = out["state_abbr"].astype("string").str.upper()
    mask = (
        out["state_abbr"].eq("PA")
        & out["final_decision"].eq("municipal_place")
        & out["resolved_geoid"].notna()
    )
    if not mask.any():
        return out

    name_root = out.loc[mask, "resolved_label"].map(_std_text)
    local_code = out.loc[mask, "resolved_geoid"].str[-5:]
    key_series = pd.Series(list(zip(name_root.tolist(), local_code.tolist(), strict=False)), index=out.index[mask])

    unresolved_keys = [key for key in key_series.tolist() if key not in remap]
    if unresolved_keys:
        _, exact_name_lookup = _build_tiger_exact_lookups()
        for key in unresolved_keys:
            label_std, code = key
            if not label_std or not code:
                continue
            tiger_match = exact_name_lookup.get(("42", "cousub", label_std))
            if tiger_match is None:
                continue
            geoid, label = tiger_match
            if str(geoid)[-5:] != str(code):
                continue
            remap[key] = (str(geoid), str(label), "pa_equivalent_cousub_tiger_canonicalized")

    has_remap = pd.Series([k in remap for k in key_series.tolist()], index=out.index[mask])
    if not has_remap.any():
        return out

    affected = out.index[mask][has_remap.to_numpy()]
    mapped = [remap[k] for k in key_series.loc[affected].tolist()]
    out.loc[affected, "resolved_geo_type"] = "cousub"
    out.loc[affected, "resolved_geoid"] = [m[0] for m in mapped]
    out.loc[affected, "resolved_label"] = [m[1] for m in mapped]
    out.loc[affected, "final_decision"] = "municipal_cousub"
    out.loc[affected, "manual_review_flag"] = True
    out.loc[affected, "resolution_source"] = [m[2] for m in mapped]
    return out


def _canonicalize_cdp_place_to_cousub_rows(full_local: pd.DataFrame) -> pd.DataFrame:
    out = full_local.copy()
    municipal = out["final_decision"].isin(["municipal_place", "municipal_cousub"])
    muni = out.loc[municipal].copy()
    if muni.empty:
        return out

    muni["resolved_geo_type"] = muni["resolved_geo_type"].astype("string")
    muni["resolved_geoid"] = muni["resolved_geoid"].astype("string")
    muni["jurisdiction_label_std"] = muni["resolved_label"].map(_std_text)
    muni["is_cdp_place"] = muni["resolved_geo_type"].eq("place") & muni["jurisdiction_label_std"].astype("string").str.endswith(" CDP")

    def base_name(value: object) -> str | None:
        text = _std_text(value)
        if text is None:
            return None
        for suffix in [" CDP", " TOWN", " CITY", " BOROUGH", " TOWNSHIP", " VILLAGE"]:
            if text.endswith(suffix):
                return text[: -len(suffix)].strip()
        return text

    muni["base_name"] = muni["resolved_label"].map(base_name)
    grouped = (
        muni.groupby(["state_abbr", "base_name"], dropna=False)
        .agg(
            cdp_place_geoid=(
                "resolved_geoid",
                lambda s: next(
                    (
                        x
                        for x, is_cdp in zip(s, muni.loc[s.index, "is_cdp_place"], strict=False)
                        if is_cdp
                    ),
                    None,
                ),
            ),
            cdp_place_label=(
                "resolved_label",
                lambda s: next(
                    (
                        x
                        for x, is_cdp in zip(s, muni.loc[s.index, "is_cdp_place"], strict=False)
                        if is_cdp
                    ),
                    None,
                ),
            ),
            cousub_geoid=(
                "resolved_geoid",
                lambda s: next(
                    (
                        x
                        for x, gt in zip(s, muni.loc[s.index, "resolved_geo_type"], strict=False)
                        if gt == "cousub"
                    ),
                    None,
                ),
            ),
            cousub_label=(
                "resolved_label",
                lambda s: next(
                    (
                        x
                        for x, gt in zip(s, muni.loc[s.index, "resolved_geo_type"], strict=False)
                        if gt == "cousub"
                    ),
                    None,
                ),
            ),
            cdp_place_rows=("is_cdp_place", "sum"),
            cousub_rows=("resolved_geo_type", lambda s: int((s == "cousub").sum())),
        )
        .reset_index()
    )
    grouped = grouped[
        grouped["cdp_place_rows"].gt(0)
        & grouped["cousub_rows"].gt(0)
        & grouped["cdp_place_geoid"].notna()
        & grouped["cousub_geoid"].notna()
    ].copy()
    if grouped.empty:
        return out

    remap = {
        (row.state_abbr, row.base_name): (row.cousub_geoid, row.cousub_label)
        for row in grouped.itertuples(index=False)
    }
    out["state_abbr"] = out["state_abbr"].astype("string").str.upper()
    out["resolved_geo_type"] = out["resolved_geo_type"].astype("string")
    out["resolved_label_std"] = out["resolved_label"].map(_std_text)
    out["base_name"] = out["resolved_label"].map(base_name)
    mask = (
        out["final_decision"].eq("municipal_place")
        & out["resolved_label_std"].astype("string").str.endswith(" CDP")
        & out["base_name"].notna()
    )
    if not mask.any():
        return out.drop(columns=["resolved_label_std", "base_name"])

    keys = list(zip(out.loc[mask, "state_abbr"].tolist(), out.loc[mask, "base_name"].tolist(), strict=False))
    mapped_mask = pd.Series([k in remap for k in keys], index=out.index[mask])
    if mapped_mask.any():
        affected = out.index[mask][mapped_mask.to_numpy()]
        mapped = [remap[k] for k in pd.Series(keys, index=out.index[mask]).loc[affected].tolist()]
        out.loc[affected, "resolved_geo_type"] = "cousub"
        out.loc[affected, "resolved_geoid"] = [m[0] for m in mapped]
        out.loc[affected, "resolved_label"] = [m[1] for m in mapped]
        out.loc[affected, "final_decision"] = "municipal_cousub"
        out.loc[affected, "manual_review_flag"] = True
        out.loc[affected, "resolution_source"] = "cdp_cousub_canonicalized"
    return out.drop(columns=["resolved_label_std", "base_name"])


def _apply_municipal_geometry_overrides(
    full_local: pd.DataFrame,
    *,
    override_path: Path | None,
) -> pd.DataFrame:
    overrides = _load_municipal_geometry_overrides(override_path)
    if overrides.empty:
        return full_local

    out = full_local.copy()
    municipal = out["final_decision"].isin(["municipal_place", "municipal_cousub"])
    out["source_jurisdiction_id"] = None
    out.loc[municipal, "source_jurisdiction_id"] = (
        out.loc[municipal, "state_fips"].astype(str)
        + ":municipal:"
        + out.loc[municipal, "resolved_geo_type"].astype(str)
        + ":"
        + out.loc[municipal, "resolved_geoid"].astype(str)
    )

    tiger = _load_all_tiger_lookups(paths=get_paths()).copy()
    tiger["resolved_geo_type"] = tiger["geo_type"].replace({"county_subdivision": "cousub"})
    tiger["state_fips"] = tiger["state_fips"].astype(str).str.zfill(2)
    tiger["geoid"] = tiger["geoid"].astype(str)
    tiger_lookup = {
        (row.state_fips, row.resolved_geo_type, row.geoid): row.namelsad
        for row in tiger.drop_duplicates(["state_fips", "resolved_geo_type", "geoid"]).itertuples(index=False)
    }

    merged = out.merge(
        overrides.rename(
            columns={
                "jurisdiction_id": "override_jurisdiction_id",
                "decision": "override_decision",
                "replacement_geo_type": "override_geo_type",
                "replacement_geoid": "override_geoid",
                "replacement_jurisdiction_name": "override_label",
            }
        ),
        left_on="source_jurisdiction_id",
        right_on="override_jurisdiction_id",
        how="left",
    )
    has_override = merged["override_decision"].notna()
    if not has_override.any():
        return out.drop(columns=["source_jurisdiction_id"], errors="ignore")

    mapped = has_override & merged["override_decision"].isin(["map_to_place", "map_to_cousub"])
    if mapped.any():
        for row in merged.loc[mapped, ["state_fips", "override_geo_type", "override_geoid", "source_jurisdiction_id"]].drop_duplicates().itertuples(index=False):
            key = (str(row.state_fips).zfill(2), str(row.override_geo_type), str(row.override_geoid))
            if key not in tiger_lookup:
                raise ValueError(
                    f"Municipal geometry override points to missing TIGER geometry: {row.source_jurisdiction_id} -> {key}"
                )
        merged.loc[mapped, "resolved_geo_type"] = merged.loc[mapped, "override_geo_type"]
        merged.loc[mapped, "resolved_geoid"] = merged.loc[mapped, "override_geoid"]
        merged.loc[mapped, "resolved_label"] = [
            tiger_lookup[
                (
                    str(state_fips).zfill(2),
                    str(geo_type),
                    str(geoid),
                )
            ]
            for state_fips, geo_type, geoid in zip(
                merged.loc[mapped, "state_fips"],
                merged.loc[mapped, "override_geo_type"],
                merged.loc[mapped, "override_geoid"],
                strict=False,
            )
        ]
        merged.loc[mapped, "final_decision"] = merged.loc[mapped, "override_geo_type"].map(
            {"place": "municipal_place", "cousub": "municipal_cousub"}
        )
        merged.loc[mapped, "manual_review_flag"] = True
        merged.loc[mapped, "resolution_source"] = "municipal_geometry_override"
        merged.loc[mapped, "fallback_applied"] = False
        merged.loc[mapped, "has_final_municipal_geoid"] = True
        merged.loc[mapped, "match_method"] = "municipal_geometry_override"

    excluded = has_override & merged["override_decision"].eq("exclude")
    if excluded.any():
        merged.loc[excluded, "final_decision"] = "exclude_legacy_municipal"
        merged.loc[excluded, "resolved_geo_type"] = None
        merged.loc[excluded, "resolved_geoid"] = None
        merged.loc[excluded, "resolved_label"] = None
        merged.loc[excluded, "manual_review_flag"] = True
        merged.loc[excluded, "resolution_source"] = "municipal_geometry_override_exclude"
        merged.loc[excluded, "fallback_applied"] = False
        merged.loc[excluded, "has_final_municipal_geoid"] = False
        merged.loc[excluded, "match_method"] = "municipal_geometry_override"

    return merged.drop(
        columns=[
            "source_jurisdiction_id",
            "override_jurisdiction_id",
            "override_decision",
            "override_geo_type",
            "override_geoid",
            "override_label",
            "reason",
            "sources",
        ],
        errors="ignore",
    )


def _state_layer_rows(state_fips_values: list[str], layer_type: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state_fips in state_fips_values:
        state_name = STATE_NAME_BY_FIPS.get(state_fips, state_fips)
        state_abbr = STATE_ABBR_BY_FIPS.get(state_fips)
        if layer_type == "state_nonmunicipal_remainder":
            jurisdiction_id = f"{state_fips}:state_nonmunicipal_remainder"
            jurisdiction_name = f"{state_name} nonmunicipal remainder"
            geometry_source = "statewide_synthetic_remainder"
        elif layer_type == "statewide_overlap_layer":
            jurisdiction_id = f"{state_fips}:statewide_overlap_layer"
            jurisdiction_name = f"{state_name} statewide overlap layer"
            geometry_source = "statewide_synthetic_overlap"
        else:
            raise ValueError(layer_type)
        rows.append(
            {
                "jurisdiction_id": jurisdiction_id,
                "jurisdiction_type": layer_type,
                "state_fips": state_fips,
                "state_abbr": state_abbr,
                "jurisdiction_name": jurisdiction_name,
                "geometry_source": geometry_source,
                "is_contracted_place": False,
                "manual_review_flag": False,
                "geo_type": None,
                "geoid": None,
                "agency_count": None,
            }
        )
    return pd.DataFrame(rows)


def build_jurisdiction_master(
    *,
    full_local: pd.DataFrame,
    full_nonlocal: pd.DataFrame,
) -> pd.DataFrame:
    local_states = set(full_local["state_fips"].dropna().astype(str))
    nonlocal_states = set(full_nonlocal["state_fips"].dropna().astype(str))
    state_fips_values = sorted(local_states | nonlocal_states)
    municipal = pd.concat(
        [
            _municipal_jurisdiction_rows(full_local),
            _nonlocal_municipal_jurisdiction_rows(full_nonlocal),
        ],
        ignore_index=True,
    )
    if not municipal.empty:
        municipal = (
            municipal.groupby(
                [
                    "jurisdiction_id",
                    "jurisdiction_type",
                    "state_fips",
                    "state_abbr",
                    "jurisdiction_name",
                    "geo_type",
                    "geoid",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                geometry_source=("geometry_source", lambda s: next((x for x in s if pd.notna(x)), None)),
                is_contracted_place=("is_contracted_place", "max"),
                manual_review_flag=("manual_review_flag", "max"),
                agency_count=("agency_count", "sum"),
            )
        )
    nonmunicipal = _state_layer_rows(state_fips_values, "state_nonmunicipal_remainder")
    overlap = _state_layer_rows(state_fips_values, "statewide_overlap_layer")
    out = pd.concat([municipal, nonmunicipal, overlap], ignore_index=True)
    out = out.sort_values(["state_fips", "jurisdiction_type", "jurisdiction_id"]).reset_index(drop=True)
    return out


def _review_status(resolution_source: str | None, manual_review_flag: object, fallback_applied: object) -> str:
    if bool(fallback_applied) or resolution_source == "agency_master_fallback":
        return "fallback"
    if resolution_source in {"second_pass", "first_pass_retained", "mini_second_pass", "high_second_pass"}:
        return "reviewed"
    if bool(manual_review_flag):
        return "flagged_auto"
    return "auto"


def build_agency_to_jurisdiction_crosswalk(
    *,
    full_local: pd.DataFrame,
    full_nonlocal: pd.DataFrame,
) -> pd.DataFrame:
    local = full_local.copy()
    local["relationship_type"] = None
    local["jurisdiction_id"] = None

    municipal = local["final_decision"].isin(["municipal_place", "municipal_cousub"])
    local.loc[municipal, "relationship_type"] = "exclusive"
    local.loc[municipal, "jurisdiction_id"] = (
        local.loc[municipal, "state_fips"].astype(str)
        + ":municipal:"
        + local.loc[municipal, "resolved_geo_type"].astype(str)
        + ":"
        + local.loc[municipal, "resolved_geoid"].astype(str)
    )

    local_nonmun = local["final_decision"].eq("reclassify_nonmunicipal")
    local.loc[local_nonmun, "relationship_type"] = local.loc[local_nonmun, "fallback_applied"].map(
        lambda v: "unresolved" if bool(v) else "exclusive"
    )
    local.loc[local_nonmun, "jurisdiction_id"] = (
        local.loc[local_nonmun, "state_fips"].astype(str) + ":state_nonmunicipal_remainder"
    )

    local_overlap = local["final_decision"].eq("reclassify_overlap")
    local.loc[local_overlap, "relationship_type"] = "overlap"
    local.loc[local_overlap, "jurisdiction_id"] = (
        local.loc[local_overlap, "state_fips"].astype(str) + ":statewide_overlap_layer"
    )

    local["review_status"] = [
        _review_status(rs, mr, fb)
        for rs, mr, fb in zip(
            local["resolution_source"],
            local["manual_review_flag"],
            local["fallback_applied"],
            strict=False,
        )
    ]
    local["weight"] = 1.0
    local["overlap_subtype"] = None
    local["geometry_hint"] = None
    local["source_table"] = "local_full"
    local_crosswalk = local[
        [
            "ori9",
            "state_fips",
            "state_abbr",
            "jurisdiction_id",
            "relationship_type",
            "weight",
            "review_status",
            "overlap_subtype",
            "geometry_hint",
            "resolution_source",
            "source_table",
        ]
    ].rename(columns={"ori9": "ori"})

    nonlocal_df = full_nonlocal.copy()
    nonlocal_df["relationship_type"] = None
    nonlocal_df["jurisdiction_id"] = None

    municipal_nonlocal = nonlocal_df["final_bucket_decision"].eq("municipal_exclusive")
    nonlocal_df.loc[municipal_nonlocal, "relationship_type"] = "exclusive"
    nonlocal_df.loc[municipal_nonlocal, "jurisdiction_id"] = (
        nonlocal_df.loc[municipal_nonlocal, "state_fips"].astype(str)
        + ":municipal:"
        + nonlocal_df.loc[municipal_nonlocal, "override_geo_type"].astype(str)
        + ":"
        + nonlocal_df.loc[municipal_nonlocal, "override_geoid"].astype(str)
    )

    nonmun = nonlocal_df["final_bucket_decision"].eq("nonmunicipal_exclusive")
    nonlocal_df.loc[nonmun, "relationship_type"] = "exclusive"
    nonlocal_df.loc[nonmun, "jurisdiction_id"] = (
        nonlocal_df.loc[nonmun, "state_fips"].astype(str) + ":state_nonmunicipal_remainder"
    )

    overlap = nonlocal_df["final_bucket_decision"].isin(["statewide_overlap", "localized_special_overlap"])
    nonlocal_df.loc[overlap, "relationship_type"] = "overlap"
    nonlocal_df.loc[overlap, "jurisdiction_id"] = (
        nonlocal_df.loc[overlap, "state_fips"].astype(str) + ":statewide_overlap_layer"
    )

    nonlocal_df["review_status"] = [
        _review_status(rs, mr, False)
        for rs, mr in zip(nonlocal_df["resolution_source"], nonlocal_df["manual_review_flag"], strict=False)
    ]
    nonlocal_df["weight"] = 1.0
    nonlocal_df["overlap_subtype"] = nonlocal_df["final_overlap_subtype"]
    nonlocal_df["geometry_hint"] = nonlocal_df["final_geometry_hint"]
    nonlocal_df["source_table"] = "nonlocal_final"
    nonlocal_crosswalk = nonlocal_df[
        [
            "ori9",
            "state_fips",
            "state_abbr",
            "jurisdiction_id",
            "relationship_type",
            "weight",
            "review_status",
            "overlap_subtype",
            "geometry_hint",
            "resolution_source",
            "source_table",
        ]
    ].rename(columns={"ori9": "ori"})

    out = pd.concat([local_crosswalk, nonlocal_crosswalk], ignore_index=True)
    out = out[out["state_fips"].notna() & out["jurisdiction_id"].notna()].copy()
    out = out.drop_duplicates(subset=["ori", "jurisdiction_id"], keep="last")
    out = out.sort_values(["state_fips", "ori", "jurisdiction_id"]).reset_index(drop=True)
    return out


def build_reference_artifacts(
    *,
    agency_master_path: Path,
    provisional_local_path: Path,
    resolved_local_tail_path: Path,
    nonlocal_final_path: Path,
    nonlocal_auto_path: Path,
    municipal_override_path: Path | None = None,
    local_override_path: Path | None = None,
) -> ReferenceArtifacts:
    agency_master = pd.read_parquet(agency_master_path)
    full_local = _build_full_local_resolution(
        provisional_local_path=provisional_local_path,
        resolved_local_tail_path=resolved_local_tail_path,
        agency_master_path=agency_master_path,
        local_override_path=local_override_path,
    )
    full_nonlocal = _build_full_nonlocal_resolution(
        nonlocal_auto_path=nonlocal_auto_path,
        nonlocal_resolved_path=nonlocal_final_path,
        agency_master_path=agency_master_path,
    )
    full_nonlocal = _apply_nonlocal_resolution_overrides(full_nonlocal, local_override_path)
    fallback_local, fallback_nonlocal = _build_unassigned_fallbacks(
        agency_master=agency_master,
        full_local=full_local,
        full_nonlocal=full_nonlocal,
    )
    if not fallback_local.empty:
        full_local = pd.concat([full_local, fallback_local], ignore_index=True).drop_duplicates(subset=["ori9"], keep="last")
        full_local = _apply_local_resolution_overrides(full_local, local_override_path)
        full_local = _reconcile_local_municipal_geography(full_local).sort_values("ori9").reset_index(drop=True)
    full_local = _enrich_from_agency_master(full_local, agency_master=agency_master)
    full_local = _canonicalize_pa_equivalent_place_cousub_rows(full_local).sort_values("ori9").reset_index(drop=True)
    full_local = _canonicalize_cdp_place_to_cousub_rows(full_local).sort_values("ori9").reset_index(drop=True)
    full_local = _apply_municipal_geometry_overrides(
        full_local,
        override_path=municipal_override_path,
    ).sort_values("ori9").reset_index(drop=True)
    if not fallback_nonlocal.empty:
        full_nonlocal = (
            pd.concat([full_nonlocal, fallback_nonlocal], ignore_index=True)
            .drop_duplicates(subset=["ori9"], keep="last")
            .sort_values("ori9")
            .reset_index(drop=True)
        )
        full_nonlocal = _apply_nonlocal_resolution_overrides(full_nonlocal, local_override_path)
    full_nonlocal = _enrich_from_agency_master(full_nonlocal, agency_master=agency_master)
    assert_tribal_agencies_not_auto_placed(
        full_local,
        tribal_oris=_tribal_oris_from_agency_master(agency_master),
        licensed_oris=_municipal_placement_licensed_oris(local_override_path),
    )
    jurisdiction_master = build_jurisdiction_master(
        full_local=full_local,
        full_nonlocal=full_nonlocal,
    )
    crosswalk = build_agency_to_jurisdiction_crosswalk(
        full_local=full_local,
        full_nonlocal=full_nonlocal,
    )
    return ReferenceArtifacts(
        full_local=full_local,
        full_nonlocal=full_nonlocal,
        jurisdiction_master=jurisdiction_master,
        agency_to_jurisdiction_crosswalk=crosswalk,
    )
