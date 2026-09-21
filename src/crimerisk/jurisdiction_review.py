from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import shapefile  # pyshp

from crimerisk.paths import RepoPaths
from crimerisk.utils.zipfiles import extract_zip_all
from crimerisk.reference import build_agency_master, _std_text


LOCAL_TYPES = {"local_police", "constable_marshal"}
NONLOCAL_DEFAULT_BUCKET = {
    "sheriff": "nonmunicipal_exclusive",
    "state_law_enforcement": "statewide_overlap",
    "special_jurisdiction": "manual_overlap_review",
    "unknown": "manual_type_review",
}

_AUTOROUTE_STOP_TOKENS = {
    "POLICE",
    "DEPARTMENT",
    "DEPT",
    "PUBLIC",
    "SAFETY",
    "SHERIFF",
    "OFFICE",
    "CITY",
    "COUNTY",
    "TOWN",
    "TOWNSHIP",
    "VILLAGE",
    "METRO",
    "METROPOLITAN",
    "DISTRICT",
    "DIVISION",
    "OF",
    "PD",
}
_TOKEN_RE = re.compile(r"[A-Z0-9]+")


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _name_tokens(value: object) -> set[str]:
    text = "" if pd.isna(value) else str(value).upper()
    return {
        token
        for token in _TOKEN_RE.findall(text)
        if token and token not in _AUTOROUTE_STOP_TOKENS
    }


def _flag_county_style_place_autoroutes(review: pd.DataFrame) -> pd.DataFrame:
    if review.empty:
        return review
    out = review.copy()
    place_fips = out.get("place_fips", pd.Series(index=out.index, dtype="object")).astype("string").str.zfill(5)
    place_name_std = out.get("name", pd.Series(index=out.index, dtype="object")).map(_std_text)
    city_alias_std = out.get("city_name_std_nibrs", pd.Series(index=out.index, dtype="object")).map(_std_text)
    agency_tokens = out.get("agency_name_std", pd.Series(index=out.index, dtype="object")).map(_name_tokens)
    place_tokens = out.get("name", pd.Series(index=out.index, dtype="object")).map(_name_tokens)
    token_overlap = pd.Series(
        [
            len((agency if isinstance(agency, set) else set()) & (place if isinstance(place, set) else set()))
            for agency, place in zip(agency_tokens, place_tokens, strict=False)
        ],
        index=out.index,
        dtype="int64",
    )
    county_like_name_signal = pd.Series(False, index=out.index)
    for col in ["agency_name_std", "agency_name_std_srs", "census_name_std", "crosswalk_agency_name_std"]:
        county_like_name_signal = county_like_name_signal | out.get(
            col,
            pd.Series(index=out.index, dtype="object"),
        ).fillna("").astype(str).str.contains(r"\b(?:COUNTY|PARISH|BOROUGH)\b")
    county_style = (
        out.get("agency_type_norm", pd.Series(index=out.index, dtype="object")).eq("local_police")
        & out.get("agency_name_std", pd.Series(index=out.index, dtype="object")).fillna("").astype(str).str.contains(r"\bCOUNTY\b")
    )
    out["place_preferred_token_overlap_count"] = np.where(
        out.get("match_method", pd.Series(index=out.index, dtype="object")).eq("place_preferred"),
        token_overlap,
        0,
    )
    out["zero_overlap_place_preferred_flag"] = (
        out.get("match_method", pd.Series(index=out.index, dtype="object")).eq("place_preferred")
        & token_overlap.eq(0)
    )
    out["county_like_name_signal"] = county_like_name_signal
    out["pseudo_place_fips_flag"] = place_fips.str.startswith("99").fillna(False)
    out["county_place_alias_risk_flag"] = (
        out.get("geo_type", pd.Series(index=out.index, dtype="object")).eq("place")
        & out["pseudo_place_fips_flag"].eq(True)
        & out["county_like_name_signal"].eq(True)
    )
    suspicious = (
        out.get("match_status", pd.Series(index=out.index, dtype="object")).eq("auto_matched")
        & out.get("geo_type", pd.Series(index=out.index, dtype="object")).eq("place")
        & county_style
        & out["pseudo_place_fips_flag"].eq(True)
        & (token_overlap.eq(0) | city_alias_std.eq(place_name_std))
    )
    out.loc[suspicious, "match_status"] = "unresolved"
    out.loc[suspicious, "match_method"] = "county_style_invalid_place_code"
    return out


def _load_tiger_lookup(*, zip_dir: Path, pattern: str, kind: str, cache_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for zip_path in sorted(zip_dir.glob(pattern)):
        out_dir = cache_dir / "v2_tiger_lookup" / zip_path.stem
        extract_zip_all(zip_path, out_dir)
        shp_files = list(out_dir.glob("*.shp"))
        if len(shp_files) != 1:
            raise RuntimeError(f"Expected 1 shapefile in {out_dir}, found {len(shp_files)}")
        reader = shapefile.Reader(str(shp_files[0]))
        field_names = [f[0] for f in reader.fields[1:]]
        idx = {name: field_names.index(name) for name in field_names}
        code_field = "PLACEFP" if kind == "place" else "COUSUBFP"
        for rec in reader.iterRecords():
            state_fips = str(rec[idx["STATEFP"]]).zfill(2)
            code = str(rec[idx[code_field]]).zfill(5)
            geoid = str(rec[idx["GEOID"]])
            name = str(rec[idx["NAME"]])
            namelsad = str(rec[idx["NAMELSAD"]])
            rows.append(
                {
                    "state_fips": state_fips,
                    "geo_type": kind,
                    "code": code,
                    "geoid": geoid,
                    "name": name,
                    "namelsad": namelsad,
                    "name_std": _std_text(name),
                    "namelsad_std": _std_text(namelsad),
                }
            )
    return pd.DataFrame(rows)


def _load_all_tiger_lookups(*, paths: RepoPaths) -> pd.DataFrame:
    places = _load_tiger_lookup(
        zip_dir=paths.data_dir / "tiger_places",
        pattern="tl_2020_*_place.zip",
        kind="place",
        cache_dir=paths.cache_dir,
    )
    cousubs = _load_tiger_lookup(
        zip_dir=paths.data_dir / "tiger_cousub",
        pattern="tl_2020_*_cousub.zip",
        kind="cousub",
        cache_dir=paths.cache_dir,
    )
    lookup = pd.concat([places, cousubs], ignore_index=True)
    if lookup.empty:
        return lookup
    lookup["candidate_label"] = (
        lookup["geo_type"].astype(str)
        + ":"
        + lookup["geoid"].astype(str)
        + ":"
        + lookup["namelsad"].astype(str)
    )
    return lookup


def _alias_rows(local_df: pd.DataFrame) -> pd.DataFrame:
    alias_cols = [
        "agency_name_std",
        "agency_name_std_srs",
        "census_name_std",
        "crosswalk_agency_name_std",
        "city_name_std_nibrs",
    ]
    existing = [c for c in alias_cols if c in local_df.columns]
    melted = local_df[["ori9", "state_fips", *existing]].melt(
        id_vars=["ori9", "state_fips"],
        value_vars=existing,
        value_name="alias_std",
    )
    melted = melted.drop(columns=["variable"])
    melted = melted.dropna(subset=["alias_std"]).drop_duplicates()
    return melted


def _summarize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["ori9", "candidate_count", "candidate_summary"])
    grouped = (
        candidates.sort_values(["ori9", "geo_type", "geoid"])
        .groupby("ori9", as_index=False)
        .agg(
            candidate_count=("geoid", "nunique"),
            candidate_summary=("candidate_label", lambda s: " | ".join(list(pd.unique(s))[:5])),
        )
    )
    return grouped


def _resolve_candidate_row(candidates: pd.DataFrame, alias_matches: pd.DataFrame, *, default_status: str) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["ori9", "match_status", "match_method"])
    candidate_counts = candidates.groupby("ori9")["geoid"].nunique().rename("candidate_count").reset_index()
    resolved = candidate_counts.copy()
    resolved["match_status"] = default_status
    resolved["match_method"] = default_status

    one_candidate = candidate_counts[candidate_counts["candidate_count"] == 1]["ori9"]
    if not one_candidate.empty:
        resolved.loc[resolved["ori9"].isin(one_candidate), "match_status"] = "auto_matched"
        resolved.loc[resolved["ori9"].isin(one_candidate), "match_method"] = "single_candidate"

    if not alias_matches.empty:
        alias_counts = alias_matches.groupby("ori9")["geoid"].nunique().rename("alias_candidate_count").reset_index()
        resolved = resolved.merge(alias_counts, on="ori9", how="left")
        resolved["alias_candidate_count"] = resolved["alias_candidate_count"].fillna(0).astype(int)
        resolved.loc[
            (resolved["candidate_count"] > 1) & (resolved["alias_candidate_count"] == 1),
            "match_status",
        ] = "auto_matched"
        resolved.loc[
            (resolved["candidate_count"] > 1) & (resolved["alias_candidate_count"] == 1),
            "match_method",
        ] = "alias_resolved"
        resolved.loc[
            (resolved["candidate_count"] > 1) & (resolved["alias_candidate_count"] > 1),
            "match_status",
        ] = "ambiguous"
        resolved.loc[
            (resolved["candidate_count"] > 1) & (resolved["alias_candidate_count"] > 1),
            "match_method",
        ] = "ambiguous_aliases"
        resolved = resolved.drop(columns=["alias_candidate_count"])

    place_counts = (
        candidates.assign(is_place=candidates["geo_type"].eq("place").astype(int))
        .groupby("ori9", as_index=False)
        .agg(place_candidate_count=("is_place", "sum"))
    )
    resolved = resolved.merge(place_counts, on="ori9", how="left")
    place_prefer_mask = (resolved["candidate_count"] > 1) & (resolved["place_candidate_count"] == 1)
    resolved.loc[place_prefer_mask, "match_status"] = "auto_matched"
    resolved.loc[place_prefer_mask, "match_method"] = "place_preferred"
    resolved = resolved.drop(columns=["place_candidate_count"])

    chosen_direct = candidates.sort_values(["ori9", "geo_type", "geoid"]).drop_duplicates(subset=["ori9"], keep="first")
    chosen_alias = alias_matches.sort_values(["ori9", "geo_type", "geoid"]).drop_duplicates(subset=["ori9"], keep="first")
    chosen_place = (
        candidates[candidates["geo_type"].eq("place")]
        .sort_values(["ori9", "geoid"])
        .drop_duplicates(subset=["ori9"], keep="first")
        .rename(
            columns={
                "geo_type": "geo_type_place",
                "code": "code_place",
                "geoid": "geoid_place",
                "name": "name_place",
                "namelsad": "namelsad_place",
                "candidate_label": "candidate_label_place",
            }
        )
    )
    chosen = (
        resolved[["ori9", "match_status"]]
        .merge(chosen_direct, on="ori9", how="left", suffixes=("", "_direct"))
        .merge(chosen_alias, on="ori9", how="left", suffixes=("_direct", "_alias"))
        .merge(chosen_place, on="ori9", how="left")
    )

    out = pd.DataFrame(
        {
            "ori9": resolved["ori9"],
            "match_status": resolved["match_status"],
            "match_method": resolved["match_method"],
        }
    )
    for col in ["geo_type", "code", "geoid", "name", "namelsad", "candidate_label"]:
        direct_col = f"{col}_direct"
        alias_col = f"{col}_alias"
        place_col = f"{col}_place"
        if direct_col in chosen.columns:
            out[col] = chosen[direct_col]
        if alias_col in chosen.columns:
            use_alias = out["match_status"].eq("auto_matched") & chosen[alias_col].notna()
            out.loc[use_alias, col] = chosen.loc[use_alias, alias_col]
        if place_col in chosen.columns:
            use_place = out["match_method"].eq("place_preferred") & chosen[place_col].notna()
            out.loc[use_place, col] = chosen.loc[use_place, place_col]
    return out


def build_jurisdiction_review(
    *,
    paths: RepoPaths,
    agency_master_path: Path | None = None,
    year_start: int = 2018,
    year_end: int = 2024,
) -> dict[str, pd.DataFrame]:
    if agency_master_path is not None and agency_master_path.exists():
        agency_master = pd.read_parquet(agency_master_path)
    else:
        agency_master = build_agency_master(paths=paths, year_start=year_start, year_end=year_end)

    tiger_lookup = _load_all_tiger_lookups(paths=paths)

    local = agency_master[agency_master["agency_type_norm"].isin(LOCAL_TYPES)].copy()
    local["review_priority"] = (
        _series_or_default(local, "latest_srs_part1_total") * 1000.0
        + _series_or_default(local, "population_latest_nibrs")
    )

    code_candidates = local.merge(
        tiger_lookup,
        left_on=["state_fips", "place_fips"],
        right_on=["state_fips", "code"],
        how="left",
    )
    code_candidates = code_candidates[code_candidates["geoid"].notna()].copy()

    aliases = _alias_rows(local)
    code_alias_matches = code_candidates.merge(
        aliases,
        left_on=["ori9", "state_fips"],
        right_on=["ori9", "state_fips"],
        how="left",
    )
    code_alias_matches = code_alias_matches[
        (code_alias_matches["alias_std"] == code_alias_matches["name_std"])
        | (code_alias_matches["alias_std"] == code_alias_matches["namelsad_std"])
    ].copy()
    code_resolved = _resolve_candidate_row(code_candidates, code_alias_matches, default_status="code_ambiguous")
    code_summary = _summarize_candidates(code_candidates)

    matched_after_code = set(code_resolved.loc[code_resolved["match_status"] == "auto_matched", "ori9"])
    needs_name = local[~local["ori9"].isin(matched_after_code)].copy()

    name_candidates_name = needs_name.merge(
        aliases,
        on=["ori9", "state_fips"],
        how="left",
    ).merge(
        tiger_lookup,
        left_on=["state_fips", "alias_std"],
        right_on=["state_fips", "name_std"],
        how="left",
    )
    name_candidates_name = name_candidates_name[name_candidates_name["geoid"].notna()].copy()

    name_candidates_namelsad = needs_name.merge(
        aliases,
        on=["ori9", "state_fips"],
        how="left",
    ).merge(
        tiger_lookup,
        left_on=["state_fips", "alias_std"],
        right_on=["state_fips", "namelsad_std"],
        how="left",
    )
    name_candidates_namelsad = name_candidates_namelsad[name_candidates_namelsad["geoid"].notna()].copy()

    name_candidates = pd.concat([name_candidates_name, name_candidates_namelsad], ignore_index=True)
    if not name_candidates.empty:
        keep_cols = [c for c in ["ori9", "geo_type", "code", "geoid", "name", "namelsad", "candidate_label"] if c in name_candidates.columns]
        name_candidates = name_candidates[keep_cols].drop_duplicates()
    name_resolved = _resolve_candidate_row(name_candidates, name_candidates, default_status="name_ambiguous")
    name_summary = _summarize_candidates(name_candidates)

    review = local.merge(code_resolved, on="ori9", how="left", suffixes=("", "_code"))
    review = review.merge(code_summary, on="ori9", how="left")

    need_name_mask = review["match_status"].isna() | review["match_status"].ne("auto_matched")
    review_name = review.loc[need_name_mask, ["ori9"]].merge(name_resolved, on="ori9", how="left")
    review = review.merge(review_name, on="ori9", how="left", suffixes=("", "_name"))
    review = review.merge(name_summary.add_suffix("_name"), left_on="ori9", right_on="ori9_name", how="left")

    use_name = review["match_status"].ne("auto_matched") & review["match_status_name"].notna()
    review.loc[use_name, "match_status"] = review.loc[use_name, "match_status_name"]
    review.loc[use_name, "match_method"] = review.loc[use_name, "match_method_name"]
    for col in ["geo_type", "code", "geoid", "name", "namelsad", "candidate_label"]:
        name_col = f"{col}_name"
        if name_col in review.columns:
            review.loc[use_name, col] = review.loc[use_name, name_col]
    review.loc[use_name & review["candidate_count_name"].notna(), "candidate_count"] = review.loc[
        use_name & review["candidate_count_name"].notna(), "candidate_count_name"
    ]
    review.loc[use_name & review["candidate_summary_name"].notna(), "candidate_summary"] = review.loc[
        use_name & review["candidate_summary_name"].notna(), "candidate_summary_name"
    ]

    review["match_status"] = review["match_status"].fillna("unresolved")
    review["match_method"] = review["match_method"].fillna("unresolved")
    review = _flag_county_style_place_autoroutes(review)
    review["provisional_jurisdiction_id"] = None
    auto_mask = review["match_status"].eq("auto_matched") & review["geoid"].notna()
    review.loc[auto_mask, "provisional_jurisdiction_id"] = (
        review.loc[auto_mask, "state_fips"].astype(str)
        + ":"
        + review.loc[auto_mask, "geo_type"].astype(str)
        + ":"
        + review.loc[auto_mask, "geoid"].astype(str)
    )

    review = review.sort_values(["match_status", "review_priority", "state_fips", "agency_name_std"], ascending=[True, False, True, True]).reset_index(drop=True)
    auto_mask = review["match_status"].eq("auto_matched") & review["geoid"].notna()

    provisional_matches = review.loc[
        auto_mask,
        [
            "ori9",
            "state_fips",
            "state_abbr",
            "place_fips",
            "agency_name_raw",
            "agency_name_std",
            "agency_type_norm",
            "geo_type",
            "code",
            "geoid",
            "name",
            "namelsad",
            "provisional_jurisdiction_id",
            "match_status",
            "match_method",
        ],
    ].copy()
    place_preferred_audit = provisional_matches[provisional_matches["match_method"].eq("place_preferred")].copy()

    unresolved_local = review[review["match_status"].isin(["unresolved", "code_ambiguous", "name_ambiguous", "ambiguous"])].copy()

    nonlocal_review = agency_master[~agency_master["agency_type_norm"].isin(LOCAL_TYPES)].copy()
    nonlocal_review["suggested_bucket_class"] = nonlocal_review["agency_type_norm"].map(NONLOCAL_DEFAULT_BUCKET).fillna("manual_type_review")
    nonlocal_review["review_priority"] = (
        _series_or_default(nonlocal_review, "latest_srs_part1_total") * 1000.0
        + _series_or_default(nonlocal_review, "population_latest_nibrs")
    )
    nonlocal_review = nonlocal_review.sort_values(["suggested_bucket_class", "review_priority", "state_fips", "agency_name_std"], ascending=[True, False, True, True]).reset_index(drop=True)
    nonlocal_manual = nonlocal_review[
        nonlocal_review["suggested_bucket_class"].isin(["manual_overlap_review", "manual_type_review"])
    ].copy()
    nonlocal_auto = nonlocal_review[
        nonlocal_review["suggested_bucket_class"].isin(["nonmunicipal_exclusive", "statewide_overlap"])
    ].copy()

    return {
        "agency_jurisdiction_review": review,
        "provisional_local_agency_matches": provisional_matches,
        "local_agency_place_preferred_audit": place_preferred_audit,
        "local_agency_manual_review": unresolved_local,
        "nonmunicipal_and_overlap_review": nonlocal_review,
        "nonmunicipal_auto_defaults": nonlocal_auto,
        "nonmunicipal_manual_review": nonlocal_manual,
    }


def write_jurisdiction_review(
    *,
    paths: RepoPaths,
    out_dir: Path,
    agency_master_path: Path | None = None,
    year_start: int = 2018,
    year_end: int = 2024,
) -> dict[str, Path]:
    outputs = build_jurisdiction_review(
        paths=paths,
        agency_master_path=agency_master_path,
        year_start=year_start,
        year_end=year_end,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, df in outputs.items():
        parquet_path = out_dir / f"{name}.parquet"
        csv_path = out_dir / f"{name}.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        written[f"{name}_parquet"] = parquet_path
        written[f"{name}_csv"] = csv_path
    return written
