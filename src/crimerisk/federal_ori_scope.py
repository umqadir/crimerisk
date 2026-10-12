"""Federal-ORI scope registry and fail-closed jurisdiction-target exclusion.

Federal UCR rows are not ordinary state/local police observations. Installation
agencies police a mappable footprint and remain eligible for jurisdiction targets;
department-wide, service-wide, and investigative rollups do not identify where their
incidents occurred and are excluded below the national scale.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from crimerisk.paths import RepoPaths


INSTALLATION = "installation"
NATIONAL_HQ = "national_hq"
FEDERAL_SCOPE_CLASSES = {INSTALLATION, NATIONAL_HQ}
REGISTRY_COLUMNS = ["ori9", "class", "evidence", "source"]

# Legacy UCR federal ORIs use agency mnemonics in the ORI stem instead of the
# state/local numeric geography pattern. Most mnemonic stems require the federal-name
# guard because state programs also use alphabetic stems. The DOI/BIA ``DI`` class is
# itself a census signal; its reservation and park agencies are classified as fixed
# installations in the registry.
_FEDERAL_ORI_STEM_PREFIXES = (
    "ATF",
    "CBP",
    "CIA",
    "DEA",
    "D0A",
    "D0L",
    "D0T",
    "DOD",
    "DOJ",
    "DOS",
    "DOT",
    "FBI",
    "GSA",
    "HHS",
    "INS",
    "MC",
    "NAV",
    "PO",
    "SS",
    "TIX",
    "TSA",
    "USA",
    "USC",
    "USM",
    "USN",
    "VA",
    "ZPP",
)

_FEDERAL_NAME_RE = re.compile(
    r"(?:^|\b)(?:"
    r"BUREAU OF ATF|FEDERAL BUREAU|FEDERAL DEPOSIT INS|"
    r"U\s*S\s+(?:DEPT|DEPARTMENT|DOI|DOE|CUSTOMS|BORDER|MARSHALS|POSTAL|"
    r"SECRET SERVICE|PARK POLICE|FOREST SER|GENERAL SERVICES|MARINE CORPS|"
    r"NAVY)|"
    r"UNITED STATES (?:AIR FORCE|ARMY|DEPARTMENT|FOREST SER|GENERAL "
    r"SERVICES|MARINE CORPS|NAVY|POSTAL SER|TREASURY)|"
    r"NATIONAL INSTITUTE OF HE|CIA SECURITY|PENTAGON FORCE|"
    r"DEFENSE INTELLIGENCE|PENSION BENEFIT GUARANTY|"
    r"ARCHITECT OF THE CAPITOL|DEA(?:\b|,)|FBI(?:\b|,)"
    r")"
)


def federal_ori_scope_registry_path(paths: RepoPaths) -> Path:
    return paths.repo_root / "configs" / "federal_ori_scope.csv"


def load_federal_ori_scope_registry(
    paths: RepoPaths, *, path: Path | None = None
) -> pd.DataFrame:
    """Load and strictly validate the federal-ORI scope registry."""
    registry_path = path or federal_ori_scope_registry_path(paths)
    if not registry_path.exists():
        raise FileNotFoundError(
            "federal ORI scope registry is required for jurisdiction controls: "
            f"{registry_path}"
        )
    frame = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    missing = sorted(set(REGISTRY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"federal ORI scope registry is missing required columns {missing}: "
            f"{registry_path}"
        )
    frame = frame[REGISTRY_COLUMNS].copy()
    for column in REGISTRY_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip()
    frame["ori9"] = frame["ori9"].str.upper()
    bad_ori = frame[~frame["ori9"].str.fullmatch(r"[A-Z0-9]{9}", na=False)]
    if not bad_ori.empty:
        raise ValueError(
            "federal ORI scope registry has invalid ori9 values: "
            + str(bad_ori["ori9"].head(20).tolist())
        )
    duplicated = frame[frame["ori9"].duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError(
            "federal ORI scope registry has duplicate ori9 rows: "
            + str(sorted(duplicated["ori9"].unique())[:20])
        )
    bad_class = frame[~frame["class"].isin(FEDERAL_SCOPE_CLASSES)]
    if not bad_class.empty:
        raise ValueError(
            "federal ORI scope registry class must be installation or national_hq: "
            + str(bad_class[["ori9", "class"]].head(20).to_dict(orient="records"))
        )
    empty_evidence = frame[frame["evidence"].eq("") | frame["source"].eq("")]
    if not empty_evidence.empty:
        raise ValueError(
            "federal ORI scope registry requires non-empty evidence and source for "
            "every ORI: "
            + str(empty_evidence["ori9"].head(20).tolist())
        )
    return frame.sort_values("ori9", kind="mergesort").reset_index(drop=True)


def _first_text(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    out = pd.Series("", index=frame.index, dtype="string")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").fillna("").str.strip()
        out = out.where(out.ne(""), values)
    return out


def identify_federal_oris(frame: pd.DataFrame) -> pd.DataFrame:
    """Identify federal ORIs independently of the classification registry.

    Signals come from an explicit federal agency type, the DOI/BIA ``DI`` ORI class,
    the conjunction of another legacy federal ORI stem and a federal agency name, or a
    federal agency name on a numeric-looking legacy ORI.
    """
    if frame.empty:
        return pd.DataFrame(columns=["ori9", "agency_name", "federal_census_signal"])
    work = frame.copy()
    ori_col = "ori9" if "ori9" in work.columns else "ori"
    if ori_col not in work.columns:
        raise ValueError("federal ORI census input requires ori9 or ori")
    work["ori9"] = work[ori_col].astype("string").str.strip().str.upper()
    work = work[work["ori9"].str.fullmatch(r"[A-Z0-9]{9}", na=False)].copy()
    work["agency_name"] = _first_text(
        work,
        (
            "agency_name_std",
            "agency_name_raw",
            "agency_name",
            "crosswalk_agency_name",
            "publication_name_std",
            "publication_name_raw",
        ),
    ).str.upper()
    agency_type = _first_text(
        work,
        (
            "agency_type_raw",
            "agency_type_name",
            "publication_agency_type",
            "agency_type_norm",
        ),
    ).str.lower()
    type_signal = agency_type.eq("federal")
    stem = work["ori9"].str.slice(2, 9)
    ori_signal = stem.str.startswith(_FEDERAL_ORI_STEM_PREFIXES)
    doi_bia_class_signal = stem.str.startswith("DI")
    name_signal = work["agency_name"].str.contains(_FEDERAL_NAME_RE, na=False)
    # A handful of published federal agencies retain numeric-looking legacy ORIs.
    # Their federal names are the independent signal the ORI stem cannot provide.
    federal_name_signal = name_signal & ~work["ori9"].str.startswith("VI")
    keep = (
        type_signal
        | doi_bia_class_signal
        | (ori_signal & name_signal)
        | federal_name_signal
    )
    federal = work.loc[keep, ["ori9", "agency_name"]].copy()
    federal["federal_census_signal"] = pd.Series(
        pd.NA, index=federal.index, dtype="string"
    )
    federal.loc[type_signal.loc[keep], "federal_census_signal"] = "agency_type_federal"
    federal.loc[
        federal["federal_census_signal"].isna()
        & doi_bia_class_signal.loc[keep],
        "federal_census_signal",
    ] = "federal_ori_class"
    federal.loc[
        federal["federal_census_signal"].isna()
        & (ori_signal & name_signal).loc[keep],
        "federal_census_signal",
    ] = "federal_ori_class_and_name"
    federal["federal_census_signal"] = federal[
        "federal_census_signal"
    ].fillna("federal_agency_name")
    return (
        federal.sort_values(["ori9", "federal_census_signal"], kind="mergesort")
        .drop_duplicates("ori9", keep="first")
        .reset_index(drop=True)
    )


def build_federal_ori_census(
    *, paths: RepoPaths, agency_panel: pd.DataFrame
) -> pd.DataFrame:
    """Union the roster, agency master, and live panel federal signals."""
    frames: list[pd.DataFrame] = []
    master_path = paths.state_dir / "reference" / "agency_master.parquet"
    if master_path.exists():
        frames.append(pd.read_parquet(master_path))
    panel_year = (
        int(pd.to_numeric(agency_panel.get("year"), errors="coerce").max())
        if not agency_panel.empty and "year" in agency_panel.columns
        else 2024
    )
    roster_path = (
        paths.data_dir
        / f"FBI-CDE-Agency-Rosters-{panel_year}"
        / "parsed"
        / f"agency_rosters_{panel_year}.parquet"
    )
    if roster_path.exists():
        frames.append(pd.read_parquet(roster_path).rename(columns={"ori": "ori9"}))
    if not agency_panel.empty:
        frames.append(agency_panel)
    census_parts = [identify_federal_oris(frame) for frame in frames]
    census_parts = [frame for frame in census_parts if not frame.empty]
    if not census_parts:
        return pd.DataFrame(columns=["ori9", "agency_name", "federal_census_signal"])
    census = pd.concat(census_parts, ignore_index=True)
    census["_has_name"] = census["agency_name"].astype("string").fillna("").ne("")
    return (
        census.sort_values(
            ["ori9", "_has_name", "federal_census_signal"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("ori9", keep="first")
        .drop(columns="_has_name")
        .reset_index(drop=True)
    )


def assert_nonzero_federal_oris_are_classified(
    *,
    census: pd.DataFrame,
    agency_estimates: pd.DataFrame,
    registry: pd.DataFrame,
    tolerance: float = 1e-9,
) -> None:
    """Fail closed when a federal ORI has target-year mass but no registry row."""
    if census.empty or agency_estimates.empty:
        return
    mass = (
        agency_estimates.assign(
            _mass=pd.to_numeric(
                agency_estimates["estimated_count"], errors="coerce"
            ).fillna(0.0)
        )
        .groupby("ori9", dropna=False, as_index=False)["_mass"]
        .sum()
    )
    active = census.merge(mass, on="ori9", how="inner")
    registered = set(registry["ori9"].astype("string"))
    missing = active[active["_mass"].gt(tolerance) & ~active["ori9"].isin(registered)]
    if missing.empty:
        return
    raise ValueError(
        f"{len(missing)} federal ORI(s) have nonzero 2024 jurisdiction-target mass but "
        "are absent from configs/federal_ori_scope.csv; classify each as installation "
        "or national_hq before rebuilding controls: "
        + str(
            missing[["ori9", "agency_name", "federal_census_signal", "_mass"]]
            .head(50)
            .to_dict(orient="records")
        )
    )


def exclude_national_hq_rows(
    frame: pd.DataFrame, *, registry: pd.DataFrame
) -> pd.DataFrame:
    """Remove national-HQ ORIs; installation rows pass through byte-for-byte."""
    if frame.empty:
        return frame.copy()
    excluded = set(
        registry.loc[registry["class"].eq(NATIONAL_HQ), "ori9"].astype("string")
    )
    return frame[~frame["ori9"].astype("string").str.upper().isin(excluded)].copy()


def build_federal_ori_mass_audit(
    *,
    census: pd.DataFrame,
    agency_estimates: pd.DataFrame,
    crosswalk: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Every federal ORI/offense with 2024 mass, its class, and current landing."""
    columns = [
        "ori9",
        "agency_name",
        "federal_census_signal",
        "class",
        "evidence",
        "source",
        "offense",
        "estimated_count",
        "reported_count_current",
        "state_fips",
        "jurisdiction_id",
        "landing_weight",
        "landed_estimated_count",
        "excluded_from_jurisdiction_targets",
    ]
    if census.empty or agency_estimates.empty:
        return pd.DataFrame(columns=columns)
    estimates = agency_estimates.copy()
    estimates["estimated_count"] = pd.to_numeric(
        estimates["estimated_count"], errors="coerce"
    ).fillna(0.0)
    estimates = estimates[estimates["estimated_count"].gt(1e-9)]
    estimates = estimates.merge(census, on="ori9", how="inner")
    estimates = estimates.merge(registry, on="ori9", how="left", validate="many_to_one")
    links = crosswalk.copy()
    if "ori9" not in links.columns and "ori" in links.columns:
        links = links.rename(columns={"ori": "ori9"})
    links = links.rename(
        columns={"weight": "landing_weight", "state_fips": "landing_state_fips"}
    )
    audit = estimates.merge(
        links[["ori9", "landing_state_fips", "jurisdiction_id", "landing_weight"]],
        on="ori9",
        how="left",
    )
    audit["state_fips"] = audit["landing_state_fips"].astype("string").str.zfill(2)
    audit["landing_weight"] = pd.to_numeric(
        audit["landing_weight"], errors="coerce"
    ).fillna(0.0)
    audit["landed_estimated_count"] = (
        audit["estimated_count"] * audit["landing_weight"]
    )
    audit["excluded_from_jurisdiction_targets"] = audit["class"].eq(NATIONAL_HQ)
    return audit.reindex(columns=columns).sort_values(
        ["ori9", "offense", "jurisdiction_id"], kind="mergesort"
    ).reset_index(drop=True)


def write_federal_ori_mass_audit(
    audit: pd.DataFrame, *, paths: RepoPaths, year: int
) -> Path:
    out = (
        paths.state_dir
        / "analysis"
        / "dc_federal"
        / f"federal_ori_{int(year)}_mass_audit.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    return out
