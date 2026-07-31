from __future__ import annotations

from pathlib import Path
import io
import sys
import zipfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.fbi_publications import (
    build_municipal_agency_alias_rows,
    norm_municipal_name,
    norm_text,
    parse_cius_local_rows,
)
from crimerisk.paths import RepoPaths
from crimerisk.jurisdiction_reference import STATE_ABBR_BY_FIPS, STATE_NAME_BY_FIPS
from crimerisk.source_selection import build_agency_preferred_observations


YEAR = 2024
STATE_NAME_TO_ABBR = {name.upper(): STATE_ABBR_BY_FIPS[fips] for fips, name in STATE_NAME_BY_FIPS.items()}
PRODUCTION_SCOPE_EXCLUDE = {"AK", "HI", "PR"}
OFFENSES_7 = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]


def _read_excel_from_zip(zip_path: Path, member: str, *, header) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        return pd.read_excel(io.BytesIO(zf.read(member)), header=header)


def _scope_state_mask(series: pd.Series) -> pd.Series:
    return ~series.astype("string").str.upper().isin(PRODUCTION_SCOPE_EXCLUDE)


def _flatten_nibrs_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flat = []
    for left, right in df.columns:
        ltxt = "" if pd.isna(left) or str(left).startswith("Unnamed:") else str(left).strip()
        rtxt = "" if pd.isna(right) or str(right).startswith("Unnamed:") else str(right).strip()
        flat.append("__".join([x for x in [ltxt, rtxt] if x]))
    df.columns = flat
    return df


def _parse_published_nibrs_rows(paths: RepoPaths) -> pd.DataFrame:
    zip_path = paths.data_dir / "FBI-NIBRS-Tables-2024" / "raw" / "statesAndFederal.zip"
    with zipfile.ZipFile(zip_path) as zf:
        member = zf.namelist()[0]
        raw = pd.read_excel(io.BytesIO(zf.read(member)), header=[4, 5])
    raw = _flatten_nibrs_columns(raw)
    raw = raw.rename(columns=lambda c: str(c).strip())
    raw["state_name"] = raw["State"].astype("string").str.strip().str.upper()
    raw["state_abbr"] = raw["state_name"].map(STATE_NAME_TO_ABBR)
    raw = raw[raw["state_abbr"].notna() & _scope_state_mask(raw["state_abbr"])].copy()
    offense_cols = {
        "Crimes Against Persons__Murder and\nNonnegligent\nManslaughter": "murder",
        "Crimes Against Persons__Rape": "rape",
        "Crimes Against Property__Robbery": "robbery",
        "Crimes Against Persons__Aggravated\nAssault": "aggravated_assault",
        "Crimes Against Property__Burglary/\nBreaking &\nEntering": "burglary",
        "Crimes Against Property__Larceny/\nTheft\nOffenses": "larceny",
        "Crimes Against Property__Motor\nVehicle\nTheft": "motor_vehicle_theft",
    }
    long = raw.melt(
        id_vars=["state_abbr", "Agency Type", "Agency Name", "Population1"],
        value_vars=list(offense_cols.keys()),
        var_name="official_offense_col",
        value_name="official_count",
    )
    long["official_count"] = pd.to_numeric(long["official_count"], errors="coerce")
    long = long[long["official_count"].notna()].copy()
    long["offense"] = long["official_offense_col"].map(offense_cols)
    long["publication_name_std"] = long["Agency Name"].map(norm_text)
    long["publication_collection"] = "nibrs_offense_type_by_agency"
    return long.rename(
        columns={
            "Agency Type": "publication_agency_type",
            "Agency Name": "publication_name_raw",
            "Population1": "publication_population",
        }
    )[
        [
            "publication_collection",
            "state_abbr",
            "publication_agency_type",
            "publication_name_raw",
            "publication_name_std",
            "publication_population",
            "offense",
            "official_count",
        ]
    ].copy()


def _build_municipal_repo_rows(paths: RepoPaths) -> pd.DataFrame:
    controls = pd.read_parquet(
        paths.state_dir / "controls" / f"jurisdiction_controls_{YEAR}.parquet",
        columns=["jurisdiction_id", "state_abbr", "jurisdiction_type", "offense", "reported_count_preferred"],
    )
    controls = controls[controls["jurisdiction_type"].eq("municipal")].copy()
    jm = pd.read_parquet(
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        columns=["jurisdiction_id", "state_abbr", "jurisdiction_type", "geo_type", "jurisdiction_name"],
    )
    jm = jm[jm["jurisdiction_type"].eq("municipal")].copy()
    jm["publication_name_std"] = jm["jurisdiction_name"].map(norm_municipal_name)
    repo = controls.merge(
        jm[["jurisdiction_id", "state_abbr", "geo_type", "jurisdiction_name", "publication_name_std"]],
        on=["jurisdiction_id", "state_abbr"],
        how="left",
    )
    repo["repo_entity_type"] = "municipal"
    repo["repo_display_name"] = repo["jurisdiction_name"]
    repo = repo[
        [
            "jurisdiction_id",
            "state_abbr",
            "geo_type",
            "publication_name_std",
            "repo_entity_type",
            "repo_display_name",
            "offense",
            "reported_count_preferred",
        ]
    ].copy()

    agency_master = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=[
            "ori9",
            "state_abbr",
            "agency_name_std",
            "crosswalk_agency_name_std",
            "census_name_std",
        ],
    )
    crosswalk = pd.read_parquet(paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet")
    alias_rows = build_municipal_agency_alias_rows(
        agency_master=agency_master,
        agency_to_jurisdiction_crosswalk=crosswalk,
        jurisdictions=jm,
    )
    if not alias_rows.empty:
        alias_repo = controls.merge(
            alias_rows.rename(columns={"publication_name_exact_std": "publication_name_std"}),
            on=["jurisdiction_id", "state_abbr"],
            how="inner",
        )
        alias_repo["geo_type"] = pd.NA
        alias_repo["repo_entity_type"] = "municipal"
        alias_repo["repo_display_name"] = alias_repo["jurisdiction_id"]
        alias_repo = alias_repo[
            [
                "jurisdiction_id",
                "state_abbr",
                "geo_type",
                "publication_name_std",
                "repo_entity_type",
                "repo_display_name",
                "offense",
                "reported_count_preferred",
            ]
        ].copy()
        repo = pd.concat([repo, alias_repo], ignore_index=True).drop_duplicates()

    return repo


def _agency_alias_frame(paths: RepoPaths) -> pd.DataFrame:
    agency_master = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=[
            "ori9",
            "state_abbr",
            "agency_name_std",
            "crosswalk_agency_name_std",
            "census_name_std",
            "agency_type_norm",
        ],
    )
    alias_rows = []
    for alias_col in ["agency_name_std", "crosswalk_agency_name_std", "census_name_std"]:
        part = agency_master[["ori9", "state_abbr", "agency_type_norm", alias_col]].rename(columns={alias_col: "publication_name_std"})
        part["alias_source"] = alias_col
        alias_rows.append(part)
    aliases = pd.concat(alias_rows, ignore_index=True)
    aliases["publication_name_std"] = aliases["publication_name_std"].map(norm_text)
    aliases = aliases[aliases["publication_name_std"].notna()].drop_duplicates().reset_index(drop=True)
    return aliases


def _build_agency_repo_rows(paths: RepoPaths) -> pd.DataFrame:
    preferred = build_agency_preferred_observations(paths=paths, year=YEAR)
    preferred = preferred[
        [
            "ori9",
            "state_abbr",
            "offense",
            "preferred_count",
            "preferred_source",
            "reporting_regime",
            "preferred_cius_reference_flag",
        ]
    ].copy()
    aliases = _agency_alias_frame(paths)
    return preferred.merge(aliases, on=["ori9", "state_abbr"], how="left")


def _candidate_type_mask(series: pd.Series, allowed: set[str] | None) -> pd.Series:
    if not allowed:
        return pd.Series(True, index=series.index)
    return series.astype("string").isin(sorted(allowed))


def _match_publication_rows(
    official: pd.DataFrame,
    repo: pd.DataFrame,
    *,
    allowed_types_by_collection: dict[str, set[str] | None],
    repo_id_col: str,
    repo_name_col: str,
    repo_count_col: str,
) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    group_col = "match_collection" if "match_collection" in official.columns else "publication_collection"
    for collection, official_group in official.groupby(group_col, dropna=False):
        allowed = allowed_types_by_collection.get(str(collection))
        repo_group = repo[_candidate_type_mask(repo["agency_type_norm"], allowed)].copy() if "agency_type_norm" in repo.columns else repo.copy()
        merged = official_group.merge(
            repo_group,
            on=["state_abbr", "publication_name_std", "offense"],
            how="left",
            suffixes=("", "_repo"),
        )
        if repo_id_col in merged.columns:
            match_counts = (
                merged.groupby(
                    [group_col, "state_abbr", "publication_name_std", "offense"],
                    dropna=False,
                )[repo_id_col]
                .nunique(dropna=True)
                .rename("repo_match_count")
                .reset_index()
            )
            merged = merged.merge(
                match_counts,
                on=[group_col, "state_abbr", "publication_name_std", "offense"],
                how="left",
            )
            merged["repo_match_count"] = pd.to_numeric(merged["repo_match_count"], errors="coerce").fillna(0).astype(int)
            merged["match_status"] = "no_repo_match"
            merged.loc[merged["repo_match_count"].eq(1), "match_status"] = "matched_unique"
            merged.loc[merged["repo_match_count"].gt(1), "match_status"] = "ambiguous_repo_match"
            merged.loc[merged["match_status"].ne("matched_unique"), repo_id_col] = pd.NA
            merged.loc[merged["match_status"].ne("matched_unique"), repo_name_col] = pd.NA
            merged.loc[merged["match_status"].ne("matched_unique"), repo_count_col] = pd.NA
            merged = merged.drop_duplicates(
                subset=[group_col, "state_abbr", "publication_name_raw", "publication_name_std", "offense", repo_id_col],
                keep="first",
            ).reset_index(drop=True)
            out_frames.append(merged)
    out = pd.concat(out_frames, ignore_index=True)
    out["official_minus_repo"] = pd.to_numeric(out["official_count"], errors="coerce") - pd.to_numeric(
        out[repo_count_col], errors="coerce"
    )
    out["abs_official_minus_repo"] = out["official_minus_repo"].abs()
    return out


def _build_cius_validation(paths: RepoPaths, cius_rows: pd.DataFrame) -> pd.DataFrame:
    municipal_rows = cius_rows[cius_rows["publication_collection"].eq("table8_city")].copy()
    other_rows = cius_rows[cius_rows["publication_collection"].isin(["table9_university", "table11_state_tribal_other"])].copy()
    other_rows["match_collection"] = other_rows["publication_collection"]
    state_mask = other_rows["publication_type"].astype("string").str.contains("state agency", case=False, na=False)
    tribal_mask = other_rows["publication_type"].astype("string").str.contains("tribal", case=False, na=False)
    other_mask = other_rows["publication_collection"].eq("table11_state_tribal_other") & ~(state_mask | tribal_mask)
    other_rows.loc[state_mask, "match_collection"] = "table11_state_tribal_other::state"
    other_rows.loc[tribal_mask, "match_collection"] = "table11_state_tribal_other::tribal"
    other_rows.loc[other_mask, "match_collection"] = "table11_state_tribal_other::other"

    municipal_repo = _build_municipal_repo_rows(paths).rename(
        columns={
            "jurisdiction_id": "repo_entity_id",
            "repo_display_name": "repo_entity_name",
            "reported_count_preferred": "repo_count",
        }
    )
    municipal_validation = _match_publication_rows(
        municipal_rows,
        municipal_repo,
        allowed_types_by_collection={"table8_city": None},
        repo_id_col="repo_entity_id",
        repo_name_col="repo_entity_name",
        repo_count_col="repo_count",
    )
    municipal_validation["repo_entity_kind"] = "jurisdiction"

    agency_repo = _build_agency_repo_rows(paths).rename(
        columns={
            "ori9": "repo_entity_id",
            "publication_name_std": "repo_publication_name_std",
            "preferred_count": "repo_count",
            "agency_type_norm": "repo_agency_type_norm",
        }
    )
    agency_repo["publication_name_std"] = agency_repo["repo_publication_name_std"]
    agency_repo["repo_entity_name"] = agency_repo["repo_publication_name_std"]
    agency_validation = _match_publication_rows(
        other_rows,
        agency_repo,
        allowed_types_by_collection={
            "table9_university": {"special_jurisdiction"},
            "table11_state_tribal_other::state": {"state_law_enforcement"},
            "table11_state_tribal_other::tribal": {"special_jurisdiction"},
            "table11_state_tribal_other::other": {"special_jurisdiction", "unknown"},
        },
        repo_id_col="repo_entity_id",
        repo_name_col="repo_entity_name",
        repo_count_col="repo_count",
    )
    agency_validation["repo_entity_kind"] = "agency"

    out = pd.concat([municipal_validation, agency_validation], ignore_index=True, sort=False)
    return out[
        [
            "publication_collection",
            "publication_type",
            "state_abbr",
            "publication_name_raw",
            "publication_name_std",
            "publication_unit_raw",
            "offense",
            "official_count",
            "match_status",
            "repo_match_count",
            "repo_entity_kind",
            "repo_entity_id",
            "repo_entity_name",
            "repo_count",
            "official_minus_repo",
            "abs_official_minus_repo",
        ]
    ].copy()


def _build_published_nibrs_validation(paths: RepoPaths, nibrs_rows: pd.DataFrame) -> pd.DataFrame:
    nibrs_rows = nibrs_rows.copy()
    nibrs_rows["match_collection"] = (
        "nibrs_offense_type_by_agency::" + nibrs_rows["publication_agency_type"].astype("string").fillna("unknown")
    )
    repo = _build_agency_repo_rows(paths).rename(
        columns={
            "ori9": "repo_entity_id",
            "publication_name_std": "repo_publication_name_std",
            "preferred_count": "repo_count",
            "agency_type_norm": "repo_agency_type_norm",
            "preferred_source": "repo_preferred_source",
            "reporting_regime": "repo_reporting_regime",
            "preferred_cius_reference_flag": "repo_preferred_cius_reference_flag",
        }
    )
    repo["publication_name_std"] = repo["repo_publication_name_std"]
    repo["repo_entity_name"] = repo["repo_publication_name_std"]
    out = _match_publication_rows(
        nibrs_rows,
        repo,
        allowed_types_by_collection={
            "nibrs_offense_type_by_agency::City": {"local_police", "constable_marshal"},
            "nibrs_offense_type_by_agency::Metropolitan County": {"sheriff"},
            "nibrs_offense_type_by_agency::Nonmetropolitan County": {"sheriff"},
            "nibrs_offense_type_by_agency::University or College": {"special_jurisdiction"},
            "nibrs_offense_type_by_agency::State Police": {"state_law_enforcement"},
            "nibrs_offense_type_by_agency::Tribal": {"special_jurisdiction"},
            "nibrs_offense_type_by_agency::Other": {"special_jurisdiction", "state_law_enforcement", "unknown"},
            "nibrs_offense_type_by_agency::Federal": set(),
        },
        repo_id_col="repo_entity_id",
        repo_name_col="repo_entity_name",
        repo_count_col="repo_count",
    )
    return out[
        [
            "publication_collection",
            "state_abbr",
            "publication_agency_type",
            "publication_name_raw",
            "publication_name_std",
            "publication_population",
            "offense",
            "official_count",
            "match_status",
            "repo_match_count",
            "repo_entity_id",
            "repo_entity_name",
            "repo_count",
            "repo_preferred_source",
            "repo_reporting_regime",
            "repo_preferred_cius_reference_flag",
            "official_minus_repo",
            "abs_official_minus_repo",
        ]
    ].copy()


def _write_summary(df: pd.DataFrame, *, out_path: Path, group_cols: list[str]) -> None:
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=("offense", "size"),
            official_total=("official_count", "sum"),
            repo_total=("repo_count", "sum"),
            mean_abs_gap=("abs_official_minus_repo", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_path, index=False)


def main() -> None:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    cius_rows = parse_cius_local_rows(paths, year=YEAR)
    nibrs_rows = _parse_published_nibrs_rows(paths)

    cius_parsed_dir = paths.data_dir / "FBI-CIUS-Annual" / str(YEAR) / "parsed"
    nibrs_parsed_dir = paths.data_dir / "FBI-NIBRS-Tables-2024" / "parsed"
    cius_parsed_dir.mkdir(parents=True, exist_ok=True)
    nibrs_parsed_dir.mkdir(parents=True, exist_ok=True)
    cius_rows.to_parquet(cius_parsed_dir / "cius_local_rows_2024.parquet", index=False)
    cius_rows.to_csv(cius_parsed_dir / "cius_local_rows_2024.csv", index=False)
    nibrs_rows.to_parquet(nibrs_parsed_dir / "nibrs_offense_type_by_agency_2024.parquet", index=False)
    nibrs_rows.to_csv(nibrs_parsed_dir / "nibrs_offense_type_by_agency_2024.csv", index=False)

    cius_validation = _build_cius_validation(paths, cius_rows)
    nibrs_validation = _build_published_nibrs_validation(paths, nibrs_rows)

    out_dir = paths.review_analysis_dir / "source_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    cius_validation.to_parquet(out_dir / "cius_local_validation_2024.parquet", index=False)
    cius_validation.to_csv(out_dir / "cius_local_validation_2024.csv", index=False)
    nibrs_validation.to_parquet(out_dir / "published_nibrs_agency_validation_2024.parquet", index=False)
    nibrs_validation.to_csv(out_dir / "published_nibrs_agency_validation_2024.csv", index=False)
    _write_summary(
        cius_validation,
        out_path=out_dir / "cius_local_validation_summary_2024.csv",
        group_cols=["publication_collection", "match_status", "offense"],
    )
    _write_summary(
        nibrs_validation,
        out_path=out_dir / "published_nibrs_agency_validation_summary_2024.csv",
        group_cols=["publication_agency_type", "match_status", "offense"],
    )
    _write_summary(
        nibrs_validation[
            ~nibrs_validation["repo_preferred_cius_reference_flag"].astype("boolean").fillna(False).astype(bool)
        ].copy(),
        out_path=out_dir / "published_nibrs_agency_validation_summary_excluding_cius_reference_2024.csv",
        group_cols=["publication_agency_type", "match_status", "offense"],
    )
    print(
        {
            "cius_rows": int(len(cius_rows)),
            "nibrs_rows": int(len(nibrs_rows)),
            "cius_validation_rows": int(len(cius_validation)),
            "nibrs_validation_rows": int(len(nibrs_validation)),
            "out_dir": str(out_dir),
        }
    )


if __name__ == "__main__":
    main()
