"""Connecticut DESPP town-publication routing and CSP overlap residualization.

DESPP town reporting units (``CTSP###00``) are state-publication identifiers, not
FBI ORIs.  The registry maps each unit to one 2020 Connecticut town county
subdivision.  The DESPP publication is the footprint-complete town source, so it
owns every registry town for which it publishes a unit.  A same-town FBI agency
row is submission evidence, not a second footprint numerator.

``CTCSP0000`` is an FBI rendering of overlapping Connecticut State Police mass.
Once town-publication rows are present, carrying both the full aggregate and the
town rows would double count them.  The supported overlap mechanism therefore
subtracts the covered DESPP town mass, offense by offense, from ``CTCSP0000``
before jurisdiction aggregation and overlap allocation.  Any nonnegative residual
remains on the statewide-overlap lane.  The conservation audit exposes the FBI
aggregate, DESPP town mass, retained residual, and publication-vs-FBI excess; it
never silently forces unlike publications to agree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths


CT_STATE_FIPS = "09"
CT_STATE_ABBR = "CT"
CTCSP_AGGREGATE_ORI = "CTCSP0000"
CT_DESPP_RAW_DATA_SOURCE = "ct_despp_town"
CT_TOWN_REGISTRY_RELPATH = Path("configs/ct_town_coverage.csv")
CT_TOWN_PARSED_RELPATH = Path("CT-DESPP-2024") / "parsed" / "ct_despp_town_srs_2024.csv"
CT_TOWN_COVERAGE_TYPES = frozenset({"resident_trooper", "state_police_general"})
CT_TOWN_UNIT_RE = re.compile(r"^CTSP\d{3}00$")

CT_TOWN_REGISTRY_COLUMNS = (
    "town",
    "cousub_geoid_2020",
    "county_fips_2020",
    "coverage_type",
    "troop",
    "ct_despp_reporting_unit",
    "despp_population_2024",
    "despp_total_nibrs_crimes_2024",
    "current_master_municipal_cousub",
    "coverage_source_locator",
    "coverage_source_url",
    "geoid_source_url",
    "review_note",
)


def get_ct_town_registry_path(paths: RepoPaths) -> Path:
    return paths.repo_root / CT_TOWN_REGISTRY_RELPATH


def get_ct_town_parsed_path(paths: RepoPaths) -> Path:
    return paths.data_dir / CT_TOWN_PARSED_RELPATH


def load_ct_town_coverage_registry(
    paths: RepoPaths,
    *,
    require_exists: bool = True,
) -> pd.DataFrame:
    """Load the CT town registry and fail closed on identity/geography defects."""
    path = get_ct_town_registry_path(paths)
    if not path.exists():
        if require_exists:
            raise FileNotFoundError(path)
        return pd.DataFrame(columns=CT_TOWN_REGISTRY_COLUMNS)

    registry = pd.read_csv(path, dtype="string").reindex(
        columns=CT_TOWN_REGISTRY_COLUMNS
    )
    missing_columns = [
        column
        for column in CT_TOWN_REGISTRY_COLUMNS
        if column not in pd.read_csv(path, nrows=0).columns
    ]
    if missing_columns:
        raise ValueError(f"CT town registry missing columns: {missing_columns}")

    registry["town"] = registry["town"].str.strip()
    registry["cousub_geoid_2020"] = (
        registry["cousub_geoid_2020"].str.strip().str.zfill(10)
    )
    registry["county_fips_2020"] = registry["county_fips_2020"].str.strip().str.zfill(5)
    registry["coverage_type"] = registry["coverage_type"].str.strip().str.lower()
    registry["ct_despp_reporting_unit"] = (
        registry["ct_despp_reporting_unit"].str.strip().str.upper()
    )
    registry["current_master_municipal_cousub"] = (
        registry["current_master_municipal_cousub"].str.strip().str.lower()
    )

    null_identity = registry[
        registry["town"].isna() | registry["cousub_geoid_2020"].isna()
    ]
    if not null_identity.empty:
        raise ValueError(
            "CT town registry has null town/GEOID identities: "
            + str(null_identity.index.tolist()[:20])
        )
    for column in ("town", "cousub_geoid_2020"):
        duplicated = registry[registry.duplicated(column, keep=False)]
        if not duplicated.empty:
            raise ValueError(
                f"CT town registry has duplicate {column}: "
                + str(sorted(duplicated[column].dropna().unique().tolist())[:20])
            )

    invalid_geoids = registry[
        ~registry["cousub_geoid_2020"].str.match(r"^09\d{8}$", na=False)
        | ~registry["county_fips_2020"].str.match(r"^09\d{3}$", na=False)
        | ~registry["cousub_geoid_2020"].str[:5].eq(registry["county_fips_2020"])
    ]
    if not invalid_geoids.empty:
        raise ValueError(
            "CT town registry has malformed or county-inconsistent GEOIDs: "
            + str(
                invalid_geoids[["town", "cousub_geoid_2020", "county_fips_2020"]]
                .head(20)
                .to_dict(orient="records")
            )
        )

    invalid_coverage = registry[~registry["coverage_type"].isin(CT_TOWN_COVERAGE_TYPES)]
    if not invalid_coverage.empty:
        raise ValueError(
            "CT town registry has unknown coverage_type: "
            + str(sorted(invalid_coverage["coverage_type"].dropna().unique().tolist()))
        )
    invalid_existing = registry[
        ~registry["current_master_municipal_cousub"].isin(["yes", "no"])
    ]
    if not invalid_existing.empty:
        raise ValueError(
            "CT town registry current_master_municipal_cousub must be yes/no"
        )

    units = registry[registry["ct_despp_reporting_unit"].notna()].copy()
    invalid_units = units[
        ~units["ct_despp_reporting_unit"].str.match(CT_TOWN_UNIT_RE, na=False)
    ]
    if not invalid_units.empty:
        raise ValueError(
            "CT town registry has malformed DESPP reporting units: "
            + str(
                invalid_units[["town", "ct_despp_reporting_unit"]]
                .head(20)
                .to_dict(orient="records")
            )
        )
    duplicated_units = units[units.duplicated("ct_despp_reporting_unit", keep=False)]
    if not duplicated_units.empty:
        raise ValueError(
            "CT town registry has duplicate DESPP reporting units: "
            + str(
                sorted(duplicated_units["ct_despp_reporting_unit"].unique().tolist())[
                    :20
                ]
            )
        )
    return registry.sort_values("cousub_geoid_2020", kind="mergesort").reset_index(
        drop=True
    )


def ct_town_lane_registry_rows(registry: pd.DataFrame) -> pd.DataFrame:
    """Rows licensed for the footprint-complete DESPP town lane."""
    if registry.empty:
        return registry.copy()
    return registry[registry["ct_despp_reporting_unit"].notna()].copy()


def ct_town_jurisdiction_rows(registry: pd.DataFrame) -> pd.DataFrame:
    """One municipal county-subdivision jurisdiction row for every registry town."""
    if registry.empty:
        return pd.DataFrame()
    has_town_unit = registry["ct_despp_reporting_unit"].notna()
    out = pd.DataFrame(
        {
            "jurisdiction_id": (
                CT_STATE_FIPS
                + ":municipal:cousub:"
                + registry["cousub_geoid_2020"].astype("string")
            ),
            "jurisdiction_type": "municipal",
            "state_fips": CT_STATE_FIPS,
            "state_abbr": CT_STATE_ABBR,
            "jurisdiction_name": registry["town"].astype("string") + " town",
            "geo_type": "cousub",
            "geoid": registry["cousub_geoid_2020"].astype("string"),
            "geometry_source": "ct_despp_town_registry",
            "is_contracted_place": False,
            "manual_review_flag": True,
            "agency_count": has_town_unit.astype(int),
        }
    )
    return out


def ct_town_crosswalk_rows(registry: pd.DataFrame) -> pd.DataFrame:
    """Exclusive synthetic-unit crosswalk rows for lane-eligible DESPP towns."""
    lane = ct_town_lane_registry_rows(registry)
    if lane.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "ori": lane["ct_despp_reporting_unit"].astype("string"),
            "state_fips": CT_STATE_FIPS,
            "state_abbr": CT_STATE_ABBR,
            "jurisdiction_id": (
                CT_STATE_FIPS
                + ":municipal:cousub:"
                + lane["cousub_geoid_2020"].astype("string")
            ),
            "relationship_type": "exclusive",
            "weight": 1.0,
            "review_status": "reviewed",
            "overlap_subtype": pd.NA,
            "geometry_hint": "ct_despp_town_cousub",
            "resolution_source": "ct_town_coverage_registry",
            "source_table": "state_publication_ct_despp_town",
        }
    )


def ct_town_agency_metadata_rows(registry: pd.DataFrame) -> pd.DataFrame:
    """Observation metadata for DESPP units, which are deliberately absent from FBI master."""
    lane = ct_town_lane_registry_rows(registry)
    if lane.empty:
        return pd.DataFrame()
    population = pd.to_numeric(lane["despp_population_2024"], errors="coerce")
    return pd.DataFrame(
        {
            "ori9": lane["ct_despp_reporting_unit"].astype("string"),
            "ori7": pd.NA,
            "state_fips": CT_STATE_FIPS,
            "state_abbr": CT_STATE_ABBR,
            "county_fips": lane["county_fips_2020"].astype("string"),
            "place_fips": pd.NA,
            "population": population,
            "agency_name_raw": lane["town"].astype("string") + " DESPP town unit",
            "agency_name_std": lane["town"].astype("string").str.upper()
            + " DESPP TOWN UNIT",
            "agency_type_raw": "ct_despp_state_publication_unit",
            "agency_type_norm": "state_publication_unit",
            "crosswalk_agency_name": lane["town"].astype("string"),
            "census_name": lane["town"].astype("string") + " town",
            "manual_review_flag": True,
        }
    )


def validate_ct_town_parsed_rows(
    rows: pd.DataFrame,
    *,
    registry: pd.DataFrame,
    year: int = 2024,
) -> pd.DataFrame:
    """Validate canonical parsed rows against the registry and seven-offense contract."""
    required = {"reporting_unit", "town", "year", "offense", "count"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"CT DESPP parsed extract missing columns: {missing}")
    out = rows.copy()
    out["reporting_unit"] = out["reporting_unit"].astype("string").str.upper()
    out["town"] = out["town"].astype("string").str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["offense"] = out["offense"].astype("string")
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out = out[out["year"].eq(int(year))].copy()
    if out.empty:
        raise ValueError(f"CT DESPP parsed extract has no rows for {int(year)}")
    if out["count"].isna().any() or out["count"].lt(0).any():
        raise ValueError("CT DESPP parsed extract has null or negative counts")
    if not out["offense"].isin(OFFENSES_7).all():
        unknown = sorted(out.loc[~out["offense"].isin(OFFENSES_7), "offense"].unique())
        raise ValueError(f"CT DESPP parsed extract has unknown offenses: {unknown}")
    if out.duplicated(["reporting_unit", "year", "offense"]).any():
        raise ValueError("CT DESPP parsed extract has duplicate unit-year-offense rows")

    lane = ct_town_lane_registry_rows(registry)
    registered = registry[registry["ct_despp_reporting_unit"].notna()].copy()
    registry_units = set(registered["ct_despp_reporting_unit"].astype(str))
    parsed_units = set(out["reporting_unit"].astype(str))
    unknown_units = sorted(parsed_units - registry_units)
    if unknown_units:
        raise ValueError(
            f"CT DESPP parsed extract contains units absent from registry: {unknown_units[:20]}"
        )
    required_lane_units = set(lane["ct_despp_reporting_unit"].astype(str))
    missing_units = sorted(required_lane_units - parsed_units)
    if missing_units:
        raise ValueError(
            f"CT DESPP parsed extract is missing registry units: {missing_units[:20]}"
        )

    expected_names = registered.set_index("ct_despp_reporting_unit")["town"].astype(
        "string"
    )
    actual_names = out[["reporting_unit", "town"]].drop_duplicates()
    actual_names["expected_town"] = actual_names["reporting_unit"].map(expected_names)
    bad_names = actual_names[~actual_names["town"].eq(actual_names["expected_town"])]
    if not bad_names.empty:
        raise ValueError(
            "CT DESPP parsed town names do not match registry: "
            + str(bad_names.head(20).to_dict(orient="records"))
        )

    offense_sets = out.groupby("reporting_unit")["offense"].agg(set)
    bad_sets = offense_sets[offense_sets.map(lambda values: values != set(OFFENSES_7))]
    if not bad_sets.empty:
        raise ValueError(
            "CT DESPP parsed units do not carry exactly seven offenses: "
            + str({key: sorted(value) for key, value in bad_sets.head(20).items()})
        )
    return out.sort_values(
        ["reporting_unit", "year", "offense"], kind="mergesort"
    ).reset_index(drop=True)


def residualize_ctcsp_overlap(
    *,
    agency_panel: pd.DataFrame,
    agency_estimates: pd.DataFrame,
    registry: pd.DataFrame,
    target_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Subtract covered CT DESPP town mass from CTCSP0000 before overlap allocation."""
    panel = agency_panel.copy()
    estimates = agency_estimates.copy()
    audit_columns = [
        "year",
        "offense",
        "fbi_ctcsp_count_before",
        "despp_town_count",
        "ctcsp_residual_after",
        "ct_state_lane_total_before",
        "ct_state_lane_total_after",
        "publication_minus_fbi_excess",
        "conservation_delta",
    ]
    if panel.empty:
        return panel, estimates, pd.DataFrame(columns=audit_columns)

    lane_units = set(
        ct_town_lane_registry_rows(registry)["ct_despp_reporting_unit"]
        .dropna()
        .astype(str)
    )
    target = panel[panel["year"].eq(int(target_year))].copy()
    published = target[
        target["ori9"].astype("string").isin(lane_units)
        & target.get("preferred_raw_data_source", pd.Series(pd.NA, index=target.index))
        .astype("string")
        .eq(CT_DESPP_RAW_DATA_SOURCE)
    ]
    if published.empty:
        return panel, estimates, pd.DataFrame(columns=audit_columns)

    town_mass = (
        published.groupby("offense", as_index=True)["preferred_count"]
        .sum()
        .reindex(OFFENSES_7)
    )
    if town_mass.isna().any():
        raise ValueError(
            "CT DESPP town lane is present but does not carry every Part I offense"
        )
    aggregate = target[target["ori9"].astype("string").eq(CTCSP_AGGREGATE_ORI)]
    aggregate_mass = (
        aggregate.groupby("offense", as_index=True)["preferred_count"]
        .sum()
        .reindex(OFFENSES_7)
    )
    if aggregate_mass.isna().any():
        raise ValueError(
            "CT DESPP town lane is present but CTCSP0000 lacks a comparable offense"
        )
    residual = (aggregate_mass - town_mass).clip(lower=0.0)

    panel_aggregate = panel["year"].eq(int(target_year)) & panel["ori9"].astype(
        "string"
    ).eq(CTCSP_AGGREGATE_ORI)
    panel.loc[panel_aggregate, "preferred_count"] = panel.loc[
        panel_aggregate, "offense"
    ].map(residual)

    estimate_aggregate = estimates["ori9"].astype("string").eq(CTCSP_AGGREGATE_ORI)
    estimates.loc[estimate_aggregate, "estimated_count"] = estimates.loc[
        estimate_aggregate, "offense"
    ].map(residual)
    if "reported_count_current" in estimates.columns:
        estimates.loc[estimate_aggregate, "reported_count_current"] = estimates.loc[
            estimate_aggregate, "offense"
        ].map(residual)
    if "agency_adjustment_count" in estimates.columns:
        estimates.loc[estimate_aggregate, "agency_adjustment_count"] = 0.0
    if "reported_count_current_supported" in estimates.columns:
        estimates.loc[estimate_aggregate, "reported_count_current_supported"] = estimates.loc[
            estimate_aggregate, "offense"
        ].map(residual)
    component_columns = [
        column
        for column in ("accepted_observed_mass", "repair_mass", "unresolved_mass")
        if column in estimates.columns
    ]
    if component_columns:
        component_total = estimates.loc[estimate_aggregate, component_columns].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0).sum(axis=1)
        residual_values = pd.to_numeric(
            estimates.loc[estimate_aggregate, "estimated_count"], errors="coerce"
        ).fillna(0.0)
        scale = residual_values / component_total.replace(0.0, pd.NA)
        scale = scale.fillna(0.0)
        estimates.loc[estimate_aggregate, component_columns] = (
            estimates.loc[estimate_aggregate, component_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .mul(scale, axis=0)
        )

    publication_excess = (town_mass - aggregate_mass).clip(lower=0.0)
    before = aggregate_mass
    after = town_mass + residual
    audit = pd.DataFrame(
        {
            "year": int(target_year),
            "offense": list(OFFENSES_7),
            "fbi_ctcsp_count_before": aggregate_mass.to_numpy(dtype=float),
            "despp_town_count": town_mass.to_numpy(dtype=float),
            "ctcsp_residual_after": residual.to_numpy(dtype=float),
            "ct_state_lane_total_before": before.to_numpy(dtype=float),
            "ct_state_lane_total_after": after.to_numpy(dtype=float),
            "publication_minus_fbi_excess": publication_excess.to_numpy(dtype=float),
        }
    )
    audit["conservation_delta"] = (
        audit["ct_state_lane_total_after"] - audit["ct_state_lane_total_before"]
    )
    mismatch = (
        audit["conservation_delta"] - audit["publication_minus_fbi_excess"]
    ).abs()
    if mismatch.gt(1e-9).any():
        raise AssertionError(
            "CT DESPP/CTCSP residual conservation identity failed: "
            + str(audit[mismatch.gt(1e-9)].to_dict(orient="records"))
        )
    return panel, estimates, audit[audit_columns]
