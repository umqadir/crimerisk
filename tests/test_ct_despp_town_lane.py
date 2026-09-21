from __future__ import annotations

import pandas as pd
import pytest

from scripts.pull.parse_ct_despp_town_srs import (
    OFFENSE_MAP as CT_DESPP_OFFENSE_MAP,
    parse_ct_despp_town_srs_csv,
    read_ct_crime_insight_list_csv,
    reconcile_ct_town_members,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.ct_town_lane import (
    CT_DESPP_RAW_DATA_SOURCE,
    ct_town_crosswalk_rows,
    ct_town_jurisdiction_rows,
    ct_town_lane_registry_rows,
    load_ct_town_coverage_registry,
    residualize_ctcsp_overlap,
    validate_ct_town_parsed_rows,
)
from crimerisk.jurisdiction_targets import build_jurisdiction_target_components
from crimerisk.paths import RepoPaths, get_paths
from crimerisk.state_publications import load_ct_despp_town_annual_ags_rows


CT_RAW_RELPATH = (
    "CT-DESPP-2024/raw/ct_crimes_by_offense_and_town_2021_2026_list.csv"
)


def test_promoted_ct_registry_is_complete_and_fail_closed() -> None:
    registry = load_ct_town_coverage_registry(get_paths())
    lane = ct_town_lane_registry_rows(registry)
    assert len(registry) == 79
    assert len(lane) == 77
    assert registry["cousub_geoid_2020"].nunique() == 79
    assert lane["ct_despp_reporting_unit"].nunique() == 77

    jurisdictions = ct_town_jurisdiction_rows(registry)
    andover = jurisdictions[jurisdictions["geoid"].eq("0901301080")].iloc[0]
    assert andover["jurisdiction_id"] == "09:municipal:cousub:0901301080"
    assert andover["geo_type"] == "cousub"

    crosswalk = ct_town_crosswalk_rows(registry)
    andover_link = crosswalk[crosswalk["ori"].eq("CTSP00100")].iloc[0]
    assert andover_link["jurisdiction_id"] == andover["jurisdiction_id"]
    assert andover_link["relationship_type"] == "exclusive"
    assert andover_link["weight"] == 1.0
    assert "CTSP00600" in set(crosswalk["ori"])


def _complete_parsed_fixture(registry: pd.DataFrame) -> pd.DataFrame:
    registered = registry[registry["ct_despp_reporting_unit"].notna()][
        ["ct_despp_reporting_unit", "town"]
    ].rename(columns={"ct_despp_reporting_unit": "reporting_unit"})
    rows = registered.merge(pd.DataFrame({"offense": list(OFFENSES_7)}), how="cross")
    rows["year"] = 2024
    rows["count"] = 0.0
    # Distinctive synthetic value exercises routing without impersonating source data.
    rows.loc[
        rows["reporting_unit"].eq("CTSP01900") & rows["offense"].eq("larceny"),
        "count",
    ] = 37.0
    return rows


def test_resident_trooper_synthetic_count_reaches_town_control() -> None:
    registry = load_ct_town_coverage_registry(get_paths())
    parsed = validate_ct_town_parsed_rows(
        _complete_parsed_fixture(registry), registry=registry, year=2024
    )
    brooklyn_count = parsed.loc[
        parsed["reporting_unit"].eq("CTSP01900") & parsed["offense"].eq("larceny"),
        "count",
    ].item()
    estimates = pd.DataFrame(
        {
            "ori9": ["CTSP01900"],
            "offense": ["larceny"],
            "estimated_count": [brooklyn_count],
            "reported_count_current": [brooklyn_count],
            "agency_adjustment_count": [0.0],
            "agency_estimate_source": ["observed"],
            "preferred_source": ["state_publication_annual"],
        }
    )
    crosswalk = ct_town_crosswalk_rows(registry).rename(columns={"ori": "ori9"})
    controls = build_jurisdiction_target_components(
        agency_estimates=estimates,
        crosswalk=crosswalk,
    )
    brooklyn = controls[
        controls["jurisdiction_id"].eq("09:municipal:cousub:0901509190")
        & controls["offense"].eq("larceny")
    ].iloc[0]
    assert brooklyn["estimated_count"] == 37.0
    assert brooklyn["target_count_from_reported_state_publication"] == 37.0


def test_unknown_despp_unit_fails_closed() -> None:
    registry = load_ct_town_coverage_registry(get_paths())
    parsed = _complete_parsed_fixture(registry)
    parsed.loc[parsed["reporting_unit"].eq("CTSP01900"), "reporting_unit"] = "CTSP99900"
    with pytest.raises(ValueError, match="units absent from registry"):
        validate_ct_town_parsed_rows(parsed, registry=registry, year=2024)


def test_missing_ct_despp_extract_fails_closed(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="required state-publication lane"):
        load_ct_despp_town_annual_ags_rows(
            paths=RepoPaths.from_repo_root(tmp_path), year=2024
        )


def test_real_crime_insight_export_covers_and_reconciles_all_registry_towns() -> None:
    paths = get_paths()
    registry = load_ct_town_coverage_registry(paths)
    raw = read_ct_crime_insight_list_csv(paths.data_dir / CT_RAW_RELPATH)
    assert raw.loc[
        raw["Offense Type"].eq("All Offense Types")
        & raw["year"].eq(2024)
        & raw["Jurisdiction by Geography"].eq("Bridgeport"),
        "count",
    ].item() == 5917
    assert raw.loc[
        raw["Offense Type"].eq("Larceny_Theft Offenses Total")
        & raw["year"].eq(2024)
        & raw["Jurisdiction by Geography"].eq("CSP - East Lyme"),
        "count",
    ].item() == 0
    reconciliation = reconcile_ct_town_members(raw, registry=registry, year=2024)

    assert reconciliation["town"].nunique() == 79
    assert len(reconciliation) == 79 * len(OFFENSES_7)
    assert set(reconciliation["offense"]) == set(OFFENSES_7)
    exact = reconciliation.groupby("town")["bare_minus_csp"].apply(
        lambda values: values.eq(0).all()
    )
    assert exact.sum() == 75
    assert set(exact.index[~exact]) == {"East Lyme", "Mansfield", "Montville", "Windham"}

    parsed = parse_ct_despp_town_srs_csv(
        paths.data_dir / CT_RAW_RELPATH, paths=paths, year=2024
    )
    assert len(parsed) == 77 * len(OFFENSES_7)
    assert parsed["reporting_unit"].nunique() == 77
    assert set(parsed["source_offense"]) == set(CT_DESPP_OFFENSE_MAP)


def test_real_brooklyn_golden_vector_reaches_town_control() -> None:
    paths = get_paths()
    rows = load_ct_despp_town_annual_ags_rows(paths=paths, year=2024)
    brooklyn = rows[rows["ori9"].eq("CTSP01900")].set_index("offense")["count"]
    assert brooklyn.to_dict() == {
        "aggravated_assault": 2,
        "burglary": 5,
        "larceny": 21,
        "motor_vehicle_theft": 2,
        "murder": 0,
        "rape": 1,
        "robbery": 1,
    }

    estimates = pd.DataFrame(
        {
            "ori9": brooklyn.index.map(lambda _: "CTSP01900"),
            "offense": brooklyn.index,
            "estimated_count": brooklyn.to_numpy(dtype=float),
            "reported_count_current": brooklyn.to_numpy(dtype=float),
            "agency_adjustment_count": 0.0,
            "agency_estimate_source": "observed",
            "preferred_source": "state_publication_annual",
        }
    )
    crosswalk = ct_town_crosswalk_rows(
        load_ct_town_coverage_registry(paths)
    ).rename(columns={"ori": "ori9"})
    controls = build_jurisdiction_target_components(
        agency_estimates=estimates,
        crosswalk=crosswalk,
    )
    routed = controls[
        controls["jurisdiction_id"].eq("09:municipal:cousub:0901509190")
    ].set_index("offense")
    assert routed["estimated_count"].to_dict() == brooklyn.astype(float).to_dict()
    assert (
        routed["target_count_from_reported_state_publication"].to_dict()
        == brooklyn.astype(float).to_dict()
    )


def test_real_ctcsp_residualization_conserves_the_fbi_vector() -> None:
    paths = get_paths()
    registry = load_ct_town_coverage_registry(paths)
    published = load_ct_despp_town_annual_ags_rows(paths=paths, year=2024)
    ctcsp = {
        "murder": 8.0,
        "rape": 61.0,
        "robbery": 25.0,
        "aggravated_assault": 69.0,
        "burglary": 128.0,
        "larceny": 897.0,
        "motor_vehicle_theft": 263.0,
    }
    panel = published.rename(columns={"count": "preferred_count"})[
        ["ori9", "year", "offense", "preferred_count", "raw_data_source"]
    ].rename(columns={"raw_data_source": "preferred_raw_data_source"})
    panel = pd.concat(
        [
            panel,
            pd.DataFrame(
                {
                    "ori9": "CTCSP0000",
                    "year": 2024,
                    "offense": list(ctcsp),
                    "preferred_count": list(ctcsp.values()),
                    "preferred_raw_data_source": "kaplan_openicpsr_srs_return_a",
                }
            ),
        ],
        ignore_index=True,
    )
    estimates = panel.rename(
        columns={"preferred_count": "estimated_count"}
    )[["ori9", "offense", "estimated_count"]].copy()
    estimates["reported_count_current"] = estimates["estimated_count"]
    estimates["agency_adjustment_count"] = 0.0

    _, _, audit = residualize_ctcsp_overlap(
        agency_panel=panel,
        agency_estimates=estimates,
        registry=registry,
        target_year=2024,
    )
    assert audit["despp_town_count"].sum() == 1278.0
    assert audit["fbi_ctcsp_count_before"].sum() == 1451.0
    assert audit["ctcsp_residual_after"].sum() == 184.0
    assert audit["ct_state_lane_total_after"].sum() == 1462.0
    assert audit["publication_minus_fbi_excess"].sum() == 11.0
    assert audit["conservation_delta"].sum() == 11.0


def test_ctcsp_residualization_asserts_publication_difference_conservation() -> None:
    registry = load_ct_town_coverage_registry(get_paths())
    unit = "CTSP01900"
    panel_rows = []
    estimate_rows = []
    for offense_index, offense in enumerate(OFFENSES_7, start=1):
        town_count = float(offense_index * 3)
        fbi_count = float(offense_index * 5)
        if offense == "larceny":
            town_count = 37.0
            fbi_count = 30.0  # Publication excess remains explicit, never hidden.
        panel_rows.extend(
            [
                {
                    "ori9": unit,
                    "year": 2024,
                    "offense": offense,
                    "preferred_count": town_count,
                    "preferred_raw_data_source": CT_DESPP_RAW_DATA_SOURCE,
                },
                {
                    "ori9": "CTCSP0000",
                    "year": 2024,
                    "offense": offense,
                    "preferred_count": fbi_count,
                    "preferred_raw_data_source": "kaplan_openicpsr_srs_return_a",
                },
            ]
        )
        estimate_rows.extend(
            [
                {
                    "ori9": unit,
                    "offense": offense,
                    "estimated_count": town_count,
                    "reported_count_current": town_count,
                    "agency_adjustment_count": 0.0,
                    "accepted_observed_mass": town_count,
                    "repair_mass": 0.0,
                    "unresolved_mass": 0.0,
                },
                {
                    "ori9": "CTCSP0000",
                    "offense": offense,
                    "estimated_count": fbi_count,
                    "reported_count_current": fbi_count,
                    "agency_adjustment_count": 0.0,
                    "accepted_observed_mass": fbi_count,
                    "repair_mass": 0.0,
                    "unresolved_mass": 0.0,
                },
            ]
        )
    panel, estimates, audit = residualize_ctcsp_overlap(
        agency_panel=pd.DataFrame(panel_rows),
        agency_estimates=pd.DataFrame(estimate_rows),
        registry=registry[registry["ct_despp_reporting_unit"].eq(unit)],
        target_year=2024,
    )
    larceny = audit[audit["offense"].eq("larceny")].iloc[0]
    assert larceny["fbi_ctcsp_count_before"] == 30.0
    assert larceny["despp_town_count"] == 37.0
    assert larceny["ctcsp_residual_after"] == 0.0
    assert larceny["publication_minus_fbi_excess"] == 7.0
    assert larceny["conservation_delta"] == 7.0
    assert (
        audit["conservation_delta"] - audit["publication_minus_fbi_excess"]
    ).abs().max() == 0.0
    assert (
        panel.loc[
            panel["ori9"].eq("CTCSP0000") & panel["offense"].eq("murder"),
            "preferred_count",
        ].item()
        == 2.0
    )
    assert (
        estimates.loc[
            estimates["ori9"].eq("CTCSP0000") & estimates["offense"].eq("murder"),
            "estimated_count",
        ].item()
        == 2.0
    )
    assert (
        estimates.loc[
            estimates["ori9"].eq("CTCSP0000") & estimates["offense"].eq("murder"),
            ["accepted_observed_mass", "repair_mass", "unresolved_mass"],
        ].sum(axis=1).item()
        == 2.0
    )
