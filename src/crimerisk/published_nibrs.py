from __future__ import annotations

import warnings

import pandas as pd

from crimerisk.paths import RepoPaths
from crimerisk.reference import _std_text


# Years for which a parsed FBI published-NIBRS tables bundle
# (data/FBI-NIBRS-Tables-<year>/parsed/nibrs_offense_type_by_agency_<year>.parquet)
# exists and is verified. An unsupported year yields an empty corroboration frame
# with a loud warning (corroboration is optional metadata, not a build input), never
# a silent reuse of another year's tables. FBI 2025 tables do not exist yet
# (STATE.md 2025 refresh availability facts, 2026-07-10).
PUBLISHED_NIBRS_SUPPORTED_YEARS = (2024,)

PUBLISHED_NIBRS_ALLOWED_TYPES_BY_COLLECTION: dict[str, set[str]] = {
    "nibrs_offense_type_by_agency::City": {"local_police", "constable_marshal"},
    "nibrs_offense_type_by_agency::Metropolitan County": {"sheriff"},
    "nibrs_offense_type_by_agency::Nonmetropolitan County": {"sheriff"},
    "nibrs_offense_type_by_agency::University or College": {"special_jurisdiction"},
    "nibrs_offense_type_by_agency::State Police": {"state_law_enforcement"},
    "nibrs_offense_type_by_agency::Tribal": {"special_jurisdiction"},
    "nibrs_offense_type_by_agency::Other": {"special_jurisdiction", "state_law_enforcement", "unknown"},
    "nibrs_offense_type_by_agency::Federal": set(),
}


def build_published_nibrs_corroboration_mask(
    *,
    nibrs_count: pd.Series,
    published_nibrs_count: pd.Series,
    nibrs_months: pd.Series | None = None,
    srs_count: pd.Series | None = None,
    veto_mask: pd.Series | None = None,
) -> pd.Series:
    nibrs_count_num = pd.to_numeric(nibrs_count, errors="coerce")
    published_nibrs_num = pd.to_numeric(published_nibrs_count, errors="coerce")
    mask = (
        nibrs_count_num.notna()
        & published_nibrs_num.notna()
        & nibrs_count_num.eq(published_nibrs_num)
    )
    if nibrs_months is not None:
        nibrs_months_num = pd.to_numeric(nibrs_months, errors="coerce")
        mask = mask & nibrs_months_num.ge(12)
    if srs_count is not None:
        srs_count_num = pd.to_numeric(srs_count, errors="coerce")
        mask = mask & (srs_count_num.isna() | ~srs_count_num.eq(published_nibrs_num))
    if veto_mask is not None:
        veto = pd.Series(veto_mask, index=mask.index).fillna(False).astype(bool)
        mask = mask & ~veto
    return mask.fillna(False)


def _build_agency_alias_frame(paths: RepoPaths) -> pd.DataFrame:
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
    alias_rows: list[pd.DataFrame] = []
    for alias_col in ["agency_name_std", "crosswalk_agency_name_std", "census_name_std"]:
        part = agency_master[["ori9", "state_abbr", "agency_type_norm", alias_col]].rename(
            columns={alias_col: "publication_name_std"}
        )
        alias_rows.append(part)
    aliases = pd.concat(alias_rows, ignore_index=True)
    aliases["state_abbr"] = aliases["state_abbr"].astype("string").str.upper()
    aliases["publication_name_std"] = aliases["publication_name_std"].map(_std_text)
    return aliases[aliases["publication_name_std"].notna()].drop_duplicates().reset_index(drop=True)


def load_published_nibrs_reference_counts(
    paths: RepoPaths,
    *,
    year: int,
) -> pd.DataFrame:
    columns = [
        "ori9",
        "state_abbr",
        "offense",
        "published_nibrs_official_count",
        "published_nibrs_agency_type",
    ]
    if int(year) not in PUBLISHED_NIBRS_SUPPORTED_YEARS:
        # Callers legitimately probe historical panel years (2018..2023) that never had a
        # parsed bundle; those stay silent, matching the pre-parameterization behavior.
        # Only a year BEYOND the supported window (the refresh direction) warns loudly.
        if int(year) > max(PUBLISHED_NIBRS_SUPPORTED_YEARS):
            warnings.warn(
                f"No parsed FBI published-NIBRS tables bundle for {int(year)} "
                f"(supported years: {PUBLISHED_NIBRS_SUPPORTED_YEARS}); published-NIBRS corroboration "
                f"is empty for this year. Parse data/FBI-NIBRS-Tables-{int(year)} and extend "
                "PUBLISHED_NIBRS_SUPPORTED_YEARS once the FBI publishes it.",
                RuntimeWarning,
                stacklevel=2,
            )
        return pd.DataFrame(columns=columns)

    parsed_path = (
        paths.data_dir
        / f"FBI-NIBRS-Tables-{int(year)}"
        / "parsed"
        / f"nibrs_offense_type_by_agency_{int(year)}.parquet"
    )
    if not parsed_path.exists():
        return pd.DataFrame(columns=columns)

    official = pd.read_parquet(
        parsed_path,
        columns=[
            "state_abbr",
            "publication_agency_type",
            "publication_name_std",
            "offense",
            "official_count",
        ],
    ).copy()
    official["state_abbr"] = official["state_abbr"].astype("string").str.upper()
    official["publication_name_std"] = official["publication_name_std"].map(_std_text)
    official["official_count"] = pd.to_numeric(official["official_count"], errors="coerce")
    official = official[official["official_count"].notna() & official["publication_name_std"].notna()].copy()
    official["match_collection"] = (
        "nibrs_offense_type_by_agency::"
        + official["publication_agency_type"].astype("string").fillna("unknown")
    )

    aliases = _build_agency_alias_frame(paths)
    matched_frames: list[pd.DataFrame] = []
    for collection, official_group in official.groupby("match_collection", dropna=False):
        allowed = PUBLISHED_NIBRS_ALLOWED_TYPES_BY_COLLECTION.get(str(collection), set())
        if not allowed:
            continue
        repo_group = aliases[aliases["agency_type_norm"].isin(sorted(allowed))].copy()
        merged = official_group.merge(
            repo_group,
            on=["state_abbr", "publication_name_std"],
            how="left",
        )
        match_counts = (
            merged.groupby(
                ["match_collection", "state_abbr", "publication_name_std", "offense"],
                dropna=False,
            )["ori9"]
            .nunique(dropna=True)
            .rename("repo_match_count")
            .reset_index()
        )
        merged = merged.merge(
            match_counts,
            on=["match_collection", "state_abbr", "publication_name_std", "offense"],
            how="left",
        )
        matched = merged[merged["repo_match_count"].eq(1)].copy()
        if matched.empty:
            continue
        matched_frames.append(
            matched[
                [
                    "ori9",
                    "state_abbr",
                    "offense",
                    "publication_agency_type",
                    "official_count",
                ]
            ].rename(
                columns={
                    "publication_agency_type": "published_nibrs_agency_type",
                    "official_count": "published_nibrs_official_count",
                }
            )
        )
    if not matched_frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(matched_frames, ignore_index=True)
    return out.drop_duplicates(subset=["ori9", "state_abbr", "offense"], keep="first").reset_index(drop=True)
