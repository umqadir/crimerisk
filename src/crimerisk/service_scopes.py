from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SERVICE_WIDE_SCOPE_COLUMNS = (
    "service_scope_id",
    "canonical_target_ori",
    "source_state_fips",
    "geometry_contributor_ori",
    "allocation_scope",
    "official_source_ref",
    "evidence_artifact",
    "evidence_sha256",
    "scope_note",
)

SERVICE_WIDE_FOOTPRINT_COVERAGE_COLUMNS = (
    "service_scope_id",
    "state_fips",
    "block_group_geoid",
    "bg_service_population_coverage_share",
    "bg_land_area_coverage_share",
    "coverage_basis",
)

RESIDENT_FOOTPRINT_COVERAGE_COLUMNS = (
    "ori",
    "state_fips",
    "block_group_geoid",
    "bg_responsibility_population_coverage_share",
    "responsibility_fraction_basis",
)


def service_wide_scope_path(paths: object) -> Path:
    return Path(paths.repo_root) / "configs" / "service_wide_agency_scopes.csv"


def service_wide_scope_dependency_paths(paths: object) -> list[Path]:
    return [
        service_wide_scope_path(paths),
        Path(paths.repo_root) / "configs" / "service_wide_footprint_coverage.csv",
        Path(paths.repo_root) / "configs" / "overlap_custom_footprint_resident_coverage.csv",
        Path(__file__),
    ]


def load_resident_footprint_coverage(paths: object) -> pd.DataFrame:
    path = Path(paths.repo_root) / "configs" / "overlap_custom_footprint_resident_coverage.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(RESIDENT_FOOTPRINT_COVERAGE_COLUMNS))
    coverage = pd.read_csv(path, dtype="string").fillna("")
    missing = set(RESIDENT_FOOTPRINT_COVERAGE_COLUMNS) - set(coverage.columns)
    if missing:
        raise ValueError(f"Resident footprint coverage missing columns: {sorted(missing)}")
    coverage = coverage[list(RESIDENT_FOOTPRINT_COVERAGE_COLUMNS)].copy()
    coverage["ori"] = coverage["ori"].str.strip().str.upper()
    coverage["state_fips"] = coverage["state_fips"].str.strip().str.zfill(2)
    coverage["block_group_geoid"] = coverage["block_group_geoid"].str.strip().str.zfill(12)
    coverage["responsibility_fraction_basis"] = coverage[
        "responsibility_fraction_basis"
    ].str.strip()
    blank = coverage[["ori", "responsibility_fraction_basis"]].eq("").any(axis=1)
    bad_bg = ~coverage["block_group_geoid"].str.fullmatch(r"\d{12}").fillna(False)
    fraction = pd.to_numeric(
        coverage["bg_responsibility_population_coverage_share"], errors="coerce"
    )
    bad_fraction = fraction.isna() | fraction.le(0.0) | fraction.gt(1.0 + 1e-9)
    if bool((blank | bad_bg | bad_fraction).any()):
        raise ValueError("Resident footprint coverage contains invalid identity or fraction rows")
    coverage["bg_responsibility_population_coverage_share"] = fraction
    if bool(coverage.duplicated(["ori", "state_fips", "block_group_geoid"]).any()):
        raise ValueError("Duplicate resident footprint coverage rows")
    return coverage


def load_service_wide_footprint_coverage(paths: object) -> pd.DataFrame:
    path = Path(paths.repo_root) / "configs" / "service_wide_footprint_coverage.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(SERVICE_WIDE_FOOTPRINT_COVERAGE_COLUMNS))
    coverage = pd.read_csv(path, dtype="string").fillna("")
    missing = set(SERVICE_WIDE_FOOTPRINT_COVERAGE_COLUMNS) - set(coverage.columns)
    if missing:
        raise ValueError(f"Service-wide footprint coverage missing columns: {sorted(missing)}")
    coverage = coverage[list(SERVICE_WIDE_FOOTPRINT_COVERAGE_COLUMNS)].copy()
    coverage["service_scope_id"] = coverage["service_scope_id"].str.strip()
    coverage["state_fips"] = coverage["state_fips"].str.strip().str.zfill(2)
    coverage["block_group_geoid"] = coverage["block_group_geoid"].str.strip().str.zfill(12)
    coverage["coverage_basis"] = coverage["coverage_basis"].str.strip()
    if bool(coverage[["service_scope_id", "coverage_basis"]].eq("").any(axis=None)):
        raise ValueError("Service-wide footprint coverage identity fields may not be blank")
    bad_bg = ~coverage["block_group_geoid"].str.fullmatch(r"\d{12}").fillna(False)
    if bool(bad_bg.any()):
        raise ValueError("Service-wide footprint coverage carries malformed block group GEOIDs")
    for column, allow_zero in (
        ("bg_service_population_coverage_share", True),
        ("bg_land_area_coverage_share", False),
    ):
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce")
        lower_bad = coverage[column].lt(0.0) if allow_zero else coverage[column].le(0.0)
        bad = coverage[column].isna() | lower_bad | coverage[column].gt(1.0 + 1e-9)
        if bool(bad.any()):
            interval = "[0, 1]" if allow_zero else "(0, 1]"
            raise ValueError(f"{column} must be finite and lie in {interval}")
    if bool(coverage.duplicated(["service_scope_id", "block_group_geoid"]).any()):
        raise ValueError("Duplicate service-wide footprint coverage rows")
    return coverage


def load_service_wide_agency_scopes(paths: object) -> pd.DataFrame:
    path = service_wide_scope_path(paths)
    if not path.exists():
        return pd.DataFrame(columns=list(SERVICE_WIDE_SCOPE_COLUMNS))
    scopes = pd.read_csv(path, dtype="string").fillna("")
    missing = set(SERVICE_WIDE_SCOPE_COLUMNS) - set(scopes.columns)
    if missing:
        raise ValueError(f"Service-wide agency scopes missing columns: {sorted(missing)}")
    scopes = scopes[list(SERVICE_WIDE_SCOPE_COLUMNS)].copy()
    for column in (
        "service_scope_id",
        "canonical_target_ori",
        "geometry_contributor_ori",
        "allocation_scope",
    ):
        scopes[column] = scopes[column].str.strip()
    scopes["canonical_target_ori"] = scopes["canonical_target_ori"].str.upper()
    scopes["geometry_contributor_ori"] = scopes["geometry_contributor_ori"].str.upper()
    scopes["source_state_fips"] = scopes["source_state_fips"].str.strip().str.zfill(2)
    if bool(scopes[list(SERVICE_WIDE_SCOPE_COLUMNS[:5])].eq("").any(axis=None)):
        raise ValueError("Service-wide agency scope identity fields may not be blank")
    if not scopes["allocation_scope"].eq("service_wide").all():
        raise ValueError("service_wide_agency_scopes.csv only accepts allocation_scope=service_wide")
    if bool(scopes.duplicated("geometry_contributor_ori", keep=False).any()):
        raise ValueError("A geometry contributor may belong to only one service-wide scope")
    identity = scopes.groupby("service_scope_id", dropna=False).agg(
        canonical_targets=("canonical_target_ori", "nunique"),
        source_states=("source_state_fips", "nunique"),
    )
    bad = identity[(identity["canonical_targets"] != 1) | (identity["source_states"] != 1)]
    if not bad.empty:
        raise ValueError(
            "Each service-wide scope must have exactly one canonical target and source state: "
            f"{bad.reset_index().to_dict(orient='records')}"
        )
    return scopes


def attach_service_scope_columns(footprints: pd.DataFrame, paths: object) -> pd.DataFrame:
    """Attach reviewed service identity while retaining each geometry contributor."""
    out = footprints.copy()
    out["geometry_contributor_ori"] = out["ori9"].astype("string").str.upper()
    scopes = load_service_wide_agency_scopes(paths)
    if scopes.empty:
        out["service_scope_id"] = pd.NA
        out["canonical_target_ori"] = out["ori9"]
        out["source_state_fips"] = out["state_fips"]
        out["allocation_scope"] = "source_state"
        return out
    mapping = scopes[
        [
            "geometry_contributor_ori",
            "service_scope_id",
            "canonical_target_ori",
            "source_state_fips",
            "allocation_scope",
        ]
    ]
    out = out.merge(mapping, on="geometry_contributor_ori", how="left", validate="many_to_one")
    service = out["service_scope_id"].astype("string").fillna("").str.strip().ne("")
    out["canonical_target_ori"] = out["canonical_target_ori"].where(service, out["ori9"])
    out["source_state_fips"] = out["source_state_fips"].where(service, out["state_fips"])
    out["allocation_scope"] = out["allocation_scope"].where(service, "source_state")
    out.loc[service, "ori9"] = out.loc[service, "canonical_target_ori"]
    if "bg_id" in out.columns:
        resident_coverage = load_resident_footprint_coverage(paths).rename(
            columns={"ori": "geometry_contributor_ori", "block_group_geoid": "bg_id"}
        )
        if not resident_coverage.empty:
            out = out.merge(
                resident_coverage,
                on=["geometry_contributor_ori", "state_fips", "bg_id"],
                how="left",
                validate="many_to_one",
            )
        else:
            out["bg_responsibility_population_coverage_share"] = pd.NA
            out["responsibility_fraction_basis"] = pd.NA
        coverage = load_service_wide_footprint_coverage(paths).rename(
            columns={"block_group_geoid": "bg_id"}
        )
        if not coverage.empty:
            out = out.merge(
                coverage,
                on=["service_scope_id", "state_fips", "bg_id"],
                how="left",
                validate="many_to_one",
            )
        else:
            for column in (
                "bg_service_population_coverage_share",
                "bg_land_area_coverage_share",
                "coverage_basis",
            ):
                out[column] = pd.NA
        service = out["allocation_scope"].astype("string").eq("service_wide")
        required = [
            "bg_service_population_coverage_share",
            "bg_land_area_coverage_share",
            "coverage_basis",
        ]
        missing_coverage = service & out[required].isna().any(axis=1)
        if bool(missing_coverage.any()):
            raise ValueError(
                "Service-wide footprint rows require exact per-BG population and land coverage"
            )
    return out


def load_custom_footprint_population_by_target(paths: object) -> pd.DataFrame:
    """Return each reporting target's own covered 2020 population."""
    bg_path = Path(paths.state_dir) / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    override_path = Path(paths.repo_root) / "configs" / "overlap_footprint_overrides.csv"
    if not bg_path.exists() or not override_path.exists():
        return pd.DataFrame(columns=["ori9", "_covered_population"])
    overrides = pd.read_csv(
        override_path,
        usecols=["ori", "target_state_fips", "displaces_county_remainder"],
        dtype="string",
    )
    overrides["ori"] = overrides["ori"].str.strip().str.upper()
    overrides["target_state_fips"] = overrides["target_state_fips"].str.strip().str.zfill(2)
    eligible_oris = set(
        overrides.loc[
            overrides["displaces_county_remainder"].str.strip().str.upper().eq("TRUE")
            & ~overrides["target_state_fips"].isin({"02", "15"}),
            "ori",
        ]
    )

    population_rows: list[pd.DataFrame] = []
    resident = load_resident_footprint_coverage(paths)
    if not resident.empty:
        resident = resident[
            resident["ori"].isin(eligible_oris)
            & ~resident["state_fips"].isin({"02", "15"})
        ].copy()
        population_rows.append(
            resident.rename(
                columns={
                    "ori": "ori9",
                    "block_group_geoid": "bg_id",
                    "bg_responsibility_population_coverage_share": "coverage",
                }
            )[["ori9", "state_fips", "bg_id", "coverage"]]
        )

    footprint_path = Path(paths.repo_root) / "configs" / "overlap_custom_footprints.csv"
    if footprint_path.exists():
        activity = pd.read_csv(
            footprint_path,
            usecols=[
                "ori",
                "state_fips",
                "block_group_geoid",
                "weight_share_basis",
                "bg_population_coverage_share",
            ],
            dtype="string",
        )
        activity = activity[
            activity["weight_share_basis"].eq("activity_or_area")
            & activity["ori"].str.strip().str.upper().isin(eligible_oris)
            & ~activity["state_fips"].str.strip().str.zfill(2).isin({"02", "15"})
        ].rename(
            columns={
                "ori": "ori9",
                "block_group_geoid": "bg_id",
                "bg_population_coverage_share": "coverage",
            }
        )
        activity["coverage"] = pd.to_numeric(activity["coverage"], errors="coerce")
        activity = activity[activity["coverage"].notna()].copy()
        if not activity.empty:
            population_rows.append(
                activity.groupby(
                    ["ori9", "state_fips", "bg_id"], as_index=False
                )["coverage"].max()
            )

    service = load_service_wide_footprint_coverage(paths)
    scopes = load_service_wide_agency_scopes(paths)
    if not service.empty and not scopes.empty:
        owners = scopes[
            ["service_scope_id", "canonical_target_ori"]
        ].drop_duplicates()
        owners = owners[owners["canonical_target_ori"].isin(eligible_oris)].copy()
        service = service.merge(
            owners, on="service_scope_id", how="left", validate="many_to_one"
        )
        service = service[
            service["canonical_target_ori"].notna()
            & ~service["state_fips"].isin({"02", "15"})
        ].rename(
            columns={
                "block_group_geoid": "bg_id",
                "canonical_target_ori": "ori9",
                "bg_service_population_coverage_share": "coverage",
            }
        )
        population_rows.append(
            service[["ori9", "state_fips", "bg_id", "coverage"]]
        )

    if not population_rows:
        return pd.DataFrame(columns=["ori9", "_covered_population"])
    footprint = pd.concat(population_rows, ignore_index=True)
    footprint["ori9"] = footprint["ori9"].astype("string").str.strip().str.upper()
    footprint["state_fips"] = footprint["state_fips"].astype("string").str.zfill(2)
    footprint["bg_id"] = footprint["bg_id"].astype("string").str.zfill(12)
    if bool(footprint.duplicated(["ori9", "state_fips", "bg_id"]).any()):
        raise ValueError("Custom footprint population lanes duplicate a target/state/BG key")

    bg_population = pd.read_parquet(
        bg_path, columns=["block_group_geoid", "total_pop20"]
    ).rename(columns={"block_group_geoid": "bg_id", "total_pop20": "bg_population"})
    bg_population["bg_id"] = bg_population["bg_id"].astype("string").str.zfill(12)
    bg_population = bg_population.drop_duplicates("bg_id")
    footprint = footprint.merge(
        bg_population, on="bg_id", how="left", validate="many_to_one"
    )
    population = pd.to_numeric(footprint["bg_population"], errors="coerce")
    invalid_population = population.isna() | ~np.isfinite(population) | population.lt(0.0)
    if bool(invalid_population.any()):
        raise ValueError("In-scope custom footprint population keys require finite BG population")
    return (
        footprint.assign(
            _covered_population=lambda d: d["coverage"]
            * pd.to_numeric(d["bg_population"], errors="coerce")
        )
        .groupby("ori9", as_index=False)["_covered_population"]
        .sum()
    )
