"""Target-year Pennsylvania primary police-service ownership."""

from __future__ import annotations

from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

from crimerisk.jurisdiction_reference import ReferenceArtifacts


PA_STATE_FIPS = "42"
PA_STATE_ABBR = "PA"
PA_SERVICE_SOURCE_URL = (
    "https://services.arcgis.com/nx9zcV4VjX0IwxpW/arcgis/rest/services/"
    "Police_Jurisdictions/FeatureServer/0"
)
PA_MCD_BAF_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/baf2020/"
    "BlockAssign_ST42_PA.zip"
)
PA_STATE_REMAINDER_ID = "42:state_nonmunicipal_remainder"
PA_STATEWIDE_OVERLAP_ID = "42:statewide_overlap_layer"
# These are rejected historical/CDP identities, not 2024 primary patrol providers.
# Keep the allowlist explicit so a newly orphaned local ORI fails closed instead of
# being silently discarded or spread across Pennsylvania.
PA_REVIEWED_NONPROVIDER_ORIS = frozenset(
    {
        "PA0064000",  # Temple
        "PA0211100",  # West Fairview
        "PA0234100",  # Boothwyn
        "PA0260600",  # Republic
    }
)
PA_SERVICE_REGISTRY_COLUMNS = [
    "municipality_geoid",
    "county_name",
    "municipality_name",
    "police_service_type",
    "provider_name",
    "provider_ori",
    "service_area_id",
    "admitted_concurrent_oris",
    "adjudicated_alias_oris",
    "target_year_disposition",
    "municipality_population",
    "source_url",
    "source_modified_utc",
]
PA_MCD_OVERRIDE_COLUMNS = [
    "mcd_geoid_2020",
    "disposition",
    "source_municipality_geoid",
    "provider_ori",
    "evidence",
]


def pa_police_service_registry_path(paths) -> Path:
    return Path(paths.repo_root) / "configs" / "pa_police_service_coverage.csv"


def pa_police_service_mcd_overrides_path(paths) -> Path:
    return Path(paths.repo_root) / "configs" / "pa_police_service_mcd_overrides.csv"


def pa_mcd_baf_path(paths) -> Path:
    return Path(paths.data_dir) / "Census-BAF-2020" / "BlockAssign_ST42_PA.zip"


def pa_police_service_dependency_paths(paths) -> list[Path]:
    return [
        pa_police_service_registry_path(paths),
        pa_police_service_mcd_overrides_path(paths),
        Path(__file__),
    ]


def _ori_set(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip().upper() for part in str(value).split("|") if part.strip()}


def _valid_ori(ori: str) -> bool:
    return bool(pd.Series([ori]).str.fullmatch(r"PA[A-Z0-9]{7}").iloc[0])


def load_pa_police_service_registry(paths) -> pd.DataFrame:
    path = pa_police_service_registry_path(paths)
    if not path.exists():
        return pd.DataFrame(columns=PA_SERVICE_REGISTRY_COLUMNS)
    frame = pd.read_csv(path, dtype="string")
    missing = set(PA_SERVICE_REGISTRY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Pennsylvania service registry missing {sorted(missing)}")
    frame = frame[PA_SERVICE_REGISTRY_COLUMNS].copy()
    frame["municipality_geoid"] = frame["municipality_geoid"].str.strip().str.zfill(10)
    if not frame["municipality_geoid"].str.fullmatch(r"42\d{8}", na=False).all():
        raise ValueError("Pennsylvania service registry has an invalid municipality GEOID")
    if frame.duplicated("municipality_geoid").any():
        raise ValueError("Pennsylvania service registry has duplicate municipality GEOIDs")
    frame["provider_ori"] = frame["provider_ori"].str.strip().str.upper()
    if not frame["provider_ori"].map(_valid_ori).all():
        raise ValueError("Pennsylvania service registry has an invalid provider ORI")
    expected_service = "42:police_service:" + frame["provider_ori"]
    if not frame["service_area_id"].astype(str).eq(expected_service.astype(str)).all():
        raise ValueError("Pennsylvania service-area IDs do not match their canonical provider")
    for column in ("admitted_concurrent_oris", "adjudicated_alias_oris"):
        bad = sorted(
            ori
            for value in frame[column]
            for ori in _ori_set(value)
            if not _valid_ori(ori)
        )
        if bad:
            raise ValueError(f"Pennsylvania registry has invalid {column}: {bad}")
    frame["municipality_population"] = pd.to_numeric(
        frame["municipality_population"], errors="coerce"
    ).fillna(0.0)
    if not frame["source_url"].astype(str).eq(PA_SERVICE_SOURCE_URL).all():
        raise ValueError("Pennsylvania service registry has an unexpected source URL")
    if frame["source_modified_utc"].isna().any():
        raise ValueError("Pennsylvania service registry has no source-modified timestamp")
    return frame


def load_pa_police_service_mcd_map(paths) -> pd.DataFrame:
    registry = load_pa_police_service_registry(paths)
    out = registry[["municipality_geoid", "service_area_id"]].rename(
        columns={"municipality_geoid": "mcd_geoid_2020"}
    )
    override_path = pa_police_service_mcd_overrides_path(paths)
    overrides = pd.read_csv(override_path, dtype="string")
    missing = set(PA_MCD_OVERRIDE_COLUMNS) - set(overrides.columns)
    if missing:
        raise ValueError(f"Pennsylvania MCD overrides missing {sorted(missing)}")
    overrides = overrides[PA_MCD_OVERRIDE_COLUMNS].copy()
    overrides["mcd_geoid_2020"] = overrides["mcd_geoid_2020"].str.zfill(10)
    if overrides.duplicated("mcd_geoid_2020").any():
        raise ValueError("Pennsylvania MCD override ledger has duplicate GEOIDs")
    service_by_mcd = dict(zip(out["mcd_geoid_2020"], out["service_area_id"]))
    rows: list[dict[str, str]] = []
    for row in overrides.itertuples(index=False):
        source_geoid = "" if pd.isna(row.source_municipality_geoid) else str(row.source_municipality_geoid).zfill(10)
        provider_ori = "" if pd.isna(row.provider_ori) else str(row.provider_ori).strip().upper()
        if source_geoid:
            service_id = service_by_mcd.get(source_geoid)
            if service_id is None:
                raise ValueError(f"PA MCD override source is absent from registry: {source_geoid}")
        elif provider_ori and _valid_ori(provider_ori):
            service_id = f"42:police_service:{provider_ori}"
        else:
            raise ValueError(f"PA MCD override has no valid target: {row.mcd_geoid_2020}")
        rows.append({"mcd_geoid_2020": str(row.mcd_geoid_2020), "service_area_id": service_id})
    out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
    if out.duplicated("mcd_geoid_2020").any():
        raise ValueError("Pennsylvania service MCD map has duplicate GEOIDs")
    return out.sort_values("mcd_geoid_2020", kind="mergesort").reset_index(drop=True)


def ensure_pa_mcd_baf(paths) -> Path:
    path = pa_mcd_baf_path(paths)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".zip.download")
    with requests.get(PA_MCD_BAF_URL, timeout=120, stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(path)
    return path


def read_pa_mcd_baf(paths) -> pd.DataFrame:
    path = ensure_pa_mcd_baf(paths)
    member = "BlockAssign_ST42_PA_MCD.txt"
    with ZipFile(path) as archive:
        with archive.open(member) as raw:
            frame = pd.read_csv(TextIOWrapper(raw, encoding="utf-8"), sep="|", dtype="string")
    frame["block_geoid"] = frame["BLOCKID"].str.zfill(15)
    frame["mcd_geoid_2020"] = "42" + frame["COUNTYFP"].str.zfill(3) + frame["COUSUBFP"].str.zfill(5)
    if frame.duplicated("block_geoid").any():
        raise ValueError("Pennsylvania MCD BAF has duplicate block IDs")
    return frame[["block_geoid", "mcd_geoid_2020"]]


def _membership(registry: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    positive: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for service_id, group in registry.groupby("service_area_id", sort=False):
        canonical = set(group["provider_ori"].astype(str))
        concurrent = set().union(*(_ori_set(value) for value in group["admitted_concurrent_oris"]))
        alias_set = set().union(*(_ori_set(value) for value in group["adjudicated_alias_oris"]))
        for ori in canonical | concurrent:
            prior = positive.get(ori)
            if prior is not None and prior != service_id:
                raise ValueError(f"PA reporting ORI {ori} belongs to multiple service areas")
            positive[ori] = str(service_id)
        for ori in alias_set:
            prior = aliases.get(ori)
            if prior is not None and prior != service_id:
                raise ValueError(f"PA alias ORI {ori} belongs to multiple service areas")
            aliases[ori] = str(service_id)
    conflict = sorted(set(positive) & set(aliases))
    if conflict:
        raise ValueError(f"PA ORIs are both admitted and aliases: {conflict}")
    return positive, aliases


def _crosswalk_row(columns: list[str], *, ori: str, jurisdiction_id: str, weight: float, relationship: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "ori": ori,
            "state_fips": PA_STATE_FIPS,
            "state_abbr": PA_STATE_ABBR,
            "jurisdiction_id": jurisdiction_id,
            "relationship_type": relationship,
            "review_status": "official_service_map",
            "overlap_subtype": pd.NA,
            "geometry_hint": "census_2020_baf_mcd",
            "resolution_source": "pccd_service_map_target_year",
            "source_table": "pa_police_service_coverage",
            "weight": float(weight),
        }]
    ).reindex(columns=columns)


def _route_pa_nonprimary_agencies(
    *,
    crosswalk: pd.DataFrame,
    agency: pd.DataFrame,
) -> pd.DataFrame:
    """Remove positive mass from PA's geometryless legacy remainder.

    The PCCD service registry is the exclusive primary-patrol partition.  County
    sheriffs, county detectives, and Allegheny County Police are additive county
    agencies, so they use the existing county-localized overlap lane.  Reviewed
    historical/CDP reporter identities with no 2024 primary footprint remain as
    zero-weight audit links.  Nothing is allowed to fall through statewide.
    """
    out = crosswalk.copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    agency_index = agency.drop_duplicates("ori9").set_index("ori9")
    remainder = (
        out["state_fips"].astype("string").str.zfill(2).eq(PA_STATE_FIPS)
        & out["jurisdiction_id"].astype(str).eq(PA_STATE_REMAINDER_ID)
        & out["weight"].gt(0.0)
    )
    if not remainder.any():
        return out

    residual_oris = set(out.loc[remainder, "ori"].astype(str))
    absent = sorted(residual_oris - set(agency_index.index.astype(str)))
    if absent:
        raise ValueError(f"PA remainder ORIs are absent from agency master: {absent}")

    county_oris: set[str] = set()
    nonprovider_oris: set[str] = set()
    unresolved: list[dict[str, str]] = []
    for ori in sorted(residual_oris):
        row = agency_index.loc[ori]
        agency_type = str(row.get("agency_type_norm", ""))
        agency_name = str(row.get("agency_name_std", "")).upper()
        county = "" if pd.isna(row.get("county_fips")) else str(row.get("county_fips")).zfill(3)
        is_county_agency = (
            agency_type == "sheriff"
            or (agency_type == "special_jurisdiction" and "COUNTY" in agency_name)
            or ori == "PA0022800"
        )
        if is_county_agency and county.isdigit() and len(county) == 3:
            county_oris.add(ori)
        elif ori in PA_REVIEWED_NONPROVIDER_ORIS:
            nonprovider_oris.add(ori)
        else:
            unresolved.append(
                {
                    "ori": ori,
                    "agency_name": agency_name,
                    "agency_type": agency_type,
                    "county_fips": county,
                }
            )
    if unresolved:
        raise ValueError(
            "PA primary-service rewrite left unreviewed positive remainder ORIs: "
            f"{unresolved}"
        )

    county_mask = remainder & out["ori"].astype(str).isin(county_oris)
    out.loc[county_mask, "jurisdiction_id"] = PA_STATEWIDE_OVERLAP_ID
    out.loc[county_mask, "relationship_type"] = "overlap"
    out.loc[county_mask, "review_status"] = "reviewed"
    out.loc[county_mask, "overlap_subtype"] = "county_agency"
    out.loc[county_mask, "geometry_hint"] = "county"
    out.loc[county_mask, "resolution_source"] = "pa_nonprimary_county_agency"
    out.loc[county_mask, "source_table"] = "pa_police_service_coverage"

    nonprovider_mask = remainder & out["ori"].astype(str).isin(nonprovider_oris)
    out.loc[nonprovider_mask, "weight"] = 0.0
    out.loc[nonprovider_mask, "relationship_type"] = "reviewed_nonprovider_identity"
    out.loc[nonprovider_mask, "review_status"] = "reviewed"
    out.loc[nonprovider_mask, "resolution_source"] = "pa_no_target_year_primary_footprint"
    out.loc[nonprovider_mask, "source_table"] = "pa_police_service_coverage"

    positive_remainder = (
        out["state_fips"].astype("string").str.zfill(2).eq(PA_STATE_FIPS)
        & out["jurisdiction_id"].astype(str).eq(PA_STATE_REMAINDER_ID)
        & pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).gt(0.0)
    )
    if positive_remainder.any():
        raise ValueError(
            "PA service partition retains positive geometryless remainder links: "
            f"{sorted(out.loc[positive_remainder, 'ori'].astype(str).unique())}"
        )
    return out


def apply_pa_police_service_coverage(artifacts: ReferenceArtifacts, *, paths) -> ReferenceArtifacts:
    registry = load_pa_police_service_registry(paths)
    if registry.empty:
        return artifacts

    crosswalk = artifacts.agency_to_jurisdiction_crosswalk.copy()
    crosswalk["ori"] = crosswalk["ori"].astype("string").str.upper()
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0)
    master = artifacts.jurisdiction_master.copy()
    agency = pd.read_parquet(Path(paths.state_dir) / "reference" / "agency_master.parquet")
    agency = agency.drop_duplicates("ori9")
    known_oris = set(agency["ori9"].astype(str))
    agency_name = dict(zip(agency["ori9"].astype(str), agency["agency_name_std"].astype(str)))
    agency_type = dict(zip(agency["ori9"].astype(str), agency["agency_type_norm"].astype(str)))

    positive_members, aliases = _membership(registry)
    unknown = sorted((set(positive_members) | set(aliases)) - known_oris)
    if unknown:
        raise ValueError(f"PA service registry ORIs are absent from agency master: {unknown}")

    old_pa_municipal = master[
        master["state_fips"].astype("string").str.zfill(2).eq(PA_STATE_FIPS)
        & master["jurisdiction_type"].astype(str).eq("municipal")
    ].copy()
    old_ids = set(old_pa_municipal["jurisdiction_id"].astype(str))
    master = master[~master["jurisdiction_id"].astype(str).isin(old_ids)].copy()

    service = registry.sort_values(
        ["service_area_id", "municipality_population"], ascending=[True, False]
    ).drop_duplicates("service_area_id")
    service_master = pd.DataFrame({
        "jurisdiction_id": service["service_area_id"].astype(str),
        "jurisdiction_type": "municipal",
        "state_fips": PA_STATE_FIPS,
        "state_abbr": PA_STATE_ABBR,
        "jurisdiction_name": service["provider_ori"].map(agency_name),
        "geo_type": "police_service",
        "geoid": service["provider_ori"].astype(str),
        "geometry_source": "census_2020_baf_mcd+pccd_police_jurisdictions",
        "is_contracted_place": True,
        "manual_review_flag": True,
        "agency_count": service["service_area_id"].map(
            pd.Series(positive_members).value_counts()
        ).fillna(0).astype(int),
    }).reindex(columns=master.columns)
    master = pd.concat([master, service_master], ignore_index=True)

    old_positive = crosswalk[
        crosswalk["jurisdiction_id"].astype(str).isin(old_ids) & crosswalk["weight"].gt(0.0)
    ].copy()
    covered_candidates = set(
        old_positive.loc[
            old_positive["ori"].map(agency_type).eq("local_police"), "ori"
        ].astype(str)
    )
    covered_oris = covered_candidates - set(positive_members) - set(aliases)

    affected = set(positive_members) | set(aliases) | covered_oris
    kept = crosswalk[
        ~crosswalk["jurisdiction_id"].astype(str).isin(old_ids)
        & ~crosswalk["ori"].astype(str).isin(affected)
    ].copy()
    columns = list(crosswalk.columns)
    rows: list[pd.DataFrame] = [kept]
    for ori, service_id in sorted(positive_members.items()):
        rows.append(_crosswalk_row(columns, ori=ori, jurisdiction_id=service_id, weight=1.0, relationship="official_police_service_footprint"))
    for ori, service_id in sorted(aliases.items()):
        rows.append(_crosswalk_row(columns, ori=ori, jurisdiction_id=service_id, weight=0.0, relationship="superseded_reporting_alias"))

    service_by_old_id = dict(
        zip(
            "42:municipal:cousub:" + registry["municipality_geoid"].astype(str),
            registry["service_area_id"].astype(str),
        )
    )
    for ori in sorted(covered_oris):
        targets = sorted({
            service_by_old_id[jurisdiction_id]
            for jurisdiction_id in old_positive.loc[
                old_positive["ori"].astype(str).eq(ori), "jurisdiction_id"
            ].astype(str)
            if jurisdiction_id in service_by_old_id
        })
        for service_id in targets or ["42:state_nonmunicipal_remainder"]:
            rows.append(_crosswalk_row(columns, ori=ori, jurisdiction_id=service_id, weight=0.0, relationship="covered_or_superseded_provider"))

    crosswalk = pd.concat(rows, ignore_index=True).reindex(columns=columns)
    crosswalk = _route_pa_nonprimary_agencies(crosswalk=crosswalk, agency=agency)
    positive = crosswalk[pd.to_numeric(crosswalk["weight"], errors="coerce").gt(0.0)]
    sums = positive.groupby("ori", dropna=False)["weight"].sum().reindex(positive_members)
    bad = sums[sums.isna() | ~np.isclose(sums, 1.0, atol=1e-9)]
    if not bad.empty:
        raise ValueError(f"PA service provider crosswalk weights do not sum to one: {bad.to_dict()}")
    dangling = set(crosswalk["jurisdiction_id"].astype(str)) - set(master["jurisdiction_id"].astype(str))
    if dangling:
        raise ValueError(f"PA service rewrite left dangling jurisdiction IDs: {sorted(dangling)[:20]}")

    return ReferenceArtifacts(
        full_local=artifacts.full_local,
        full_nonlocal=artifacts.full_nonlocal,
        jurisdiction_master=master.sort_values(
            ["state_fips", "jurisdiction_type", "jurisdiction_id"], kind="mergesort"
        ).reset_index(drop=True),
        agency_to_jurisdiction_crosswalk=crosswalk.sort_values(
            ["state_fips", "ori", "jurisdiction_id"], kind="mergesort"
        ).reset_index(drop=True),
    )
