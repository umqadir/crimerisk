from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from crimerisk.observations import (
    _match_cius_table11_agencies,
    _promote_cius_agency_counts,
)
from crimerisk.fbi_publications import match_cius_municipal_rows_to_jurisdictions
from crimerisk.jurisdiction_reference import _infer_special_bucket
from crimerisk.reference import _merge_current_roster_identity
from crimerisk.reporting_regimes import (
    ReportingRegimeBuildConfig,
    _srs_month_detail_years,
)
from crimerisk.source_selection import _load_reporting_regimes
from crimerisk.trend_fills import build_agency_trend_fill_panel
from crimerisk.source_provenance import CIUS_SOURCE


def test_cius_promotion_does_not_require_same_year_return_a_row():
    counts = pd.DataFrame(
        {
            "ori9": ["XX0000100", "XX0000100"],
            "offense": ["murder", "robbery"],
            "official_count": [2, 11],
            "publication_population": [12345, 12345],
        }
    )
    agency_master = pd.DataFrame(
        {
            "ori9": ["XX0000100"],
            "ori7": ["XX00001"],
            "state_fips": ["99"],
            "state_abbr": ["XX"],
            "county_fips": ["001"],
            "place_fips": ["00001"],
            "agency_name_raw": ["Example Police Department"],
            "agency_name_std": ["EXAMPLE POLICE DEPARTMENT"],
            "agency_type_raw": ["City"],
            "agency_type_norm": ["local police department"],
            "crosswalk_agency_name": ["EXAMPLE"],
            "census_name": ["EXAMPLE"],
            "manual_review_flag": [False],
            "population": [12000],
        }
    )

    promoted = _promote_cius_agency_counts(
        counts,
        agency_master=agency_master,
        year=2025,
    )

    assert promoted["year"].eq(2025).all()
    assert promoted["source"].eq(CIUS_SOURCE).all()
    assert promoted["months_reported"].eq(12).all()
    assert promoted["population"].eq(12345).all()
    assert promoted["annual_part1_total"].eq(13).all()
    assert dict(zip(promoted["offense"], promoted["count"])) == {
        "murder": 2,
        "robbery": 11,
    }


def _one_cius_city(name: str, *, population: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "publication_collection": ["table8_city"],
            "state_abbr": ["MI"],
            "publication_name_raw": [name],
            "publication_name_exact_std": [name.upper()],
            "publication_name_std": [name.upper().removesuffix(" TOWNSHIP")],
            "publication_population": [population],
            "offense": ["robbery"],
            "official_count": [3],
        }
    )


def test_cius_typed_township_match_does_not_collapse_to_same_name_village():
    jurisdictions = pd.DataFrame(
        {
            "jurisdiction_id": [
                "26:municipal:place:2647560",
                "26:municipal:cousub:2602147600",
            ],
            "state_abbr": ["MI", "MI"],
            "jurisdiction_type": ["municipal", "municipal"],
            "jurisdiction_name": ["Lincoln village", "Lincoln charter township"],
        }
    )

    matched = match_cius_municipal_rows_to_jurisdictions(
        _one_cius_city("Lincoln Township"), jurisdictions
    )

    assert matched.loc[0, "jurisdiction_id"] == "26:municipal:cousub:2602147600"


def test_cius_current_roster_disambiguates_same_name_townships():
    jurisdictions = pd.DataFrame(
        {
            "jurisdiction_id": [
                "26:municipal:cousub:2614349840",
                "26:municipal:cousub:2612549820",
            ],
            "state_abbr": ["MI", "MI"],
            "jurisdiction_type": ["municipal", "municipal"],
            "jurisdiction_name": ["Lyon township", "Lyon charter township"],
        }
    )
    agency_master = pd.DataFrame(
        {
            "ori9": ["MI631860X"],
            "state_abbr": ["MI"],
            "source_presence_roster": [True],
            "roster_municipal_name_std": ["LYON TOWNSHIP"],
        }
    )
    crosswalk = pd.DataFrame(
        {
            "ori": ["MI631860X"],
            "jurisdiction_id": ["26:municipal:cousub:2612549820"],
            "weight": [1.0],
        }
    )

    matched = match_cius_municipal_rows_to_jurisdictions(
        _one_cius_city("Lyon Township", population=27456),
        jurisdictions,
        agency_master=agency_master,
        agency_to_jurisdiction_crosswalk=crosswalk,
    )

    assert matched.loc[0, "jurisdiction_id"] == "26:municipal:cousub:2612549820"


def test_cius_explicit_township_never_falls_through_to_same_name_city():
    jurisdictions = pd.DataFrame(
        {
            "jurisdiction_id": ["26:municipal:place:2600001"],
            "state_abbr": ["MI"],
            "jurisdiction_type": ["municipal"],
            "jurisdiction_name": ["Example city"],
        }
    )

    matched = match_cius_municipal_rows_to_jurisdictions(
        _one_cius_city("Example Township"), jurisdictions
    )

    assert matched.empty


def test_unresolved_special_agency_is_not_declared_statewide():
    bucket, subtype, geometry = _infer_special_bucket(
        ["CITY OF LAS VEGAS DEPARTMENT OF PUBLIC SAFETY"],
        "special_jurisdiction",
    )

    assert bucket == "localized_special_overlap"
    assert subtype is None
    assert geometry is None


def test_state_agency_field_office_is_not_declared_statewide_from_class_alone():
    bucket, subtype, geometry = _infer_special_bucket(
        ["COMMERCIAL ENFORCEMENT DIVISION HATTIESBURG"],
        "state_law_enforcement",
    )

    assert bucket == "localized_special_overlap"
    assert subtype is None
    assert geometry is None


def test_explicit_state_police_name_retains_statewide_footprint():
    bucket, subtype, geometry = _infer_special_bucket(
        ["NEW YORK STATE POLICE HEADQUARTERS"],
        "state_law_enforcement",
    )

    assert bucket == "statewide_overlap"
    assert subtype is None
    assert geometry == "statewide"


def test_target_year_roster_extends_historical_agency_identity(tmp_path):
    roster_dir = tmp_path / "data" / "FBI-CDE-Agency-Rosters-2025" / "parsed"
    roster_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ori": ["XX0010200"],
            "state_abbr": ["IA"],
            "agency_name": ["Lake View Police Department"],
            "agency_type_name": ["City"],
            "counties": ["SAC"],
        }
    ).to_parquet(roster_dir / "agency_rosters_2025.parquet", index=False)
    historical = pd.DataFrame(
        {
            "ori9": ["IA0010100"],
            "state_abbr": ["IA"],
            "state_fips": ["19"],
            "agency_name_raw": ["Existing"],
            "agency_name_std": ["EXISTING"],
            "agency_type_raw": ["City"],
            "agency_type_norm": ["local_police"],
            "source_presence_srs": [True],
            "source_presence_nibrs": [False],
        }
    )

    merged = _merge_current_roster_identity(
        historical,
        paths=SimpleNamespace(data_dir=tmp_path / "data"),
        year=2025,
        append_roster_oris={"XX0010200"},
    ).set_index("ori9")

    assert set(merged.index) == {"IA0010100", "XX0010200"}
    assert merged.loc["XX0010200", "roster_municipal_name_std"] == "LAKE VIEW"
    assert merged.loc["XX0010200", "agency_type_norm"] == "local_police"
    assert bool(merged.loc["XX0010200", "source_presence_roster"])
    assert not bool(merged.loc["XX0010200", "source_presence_srs"])


def test_reporting_regimes_only_request_years_in_packaged_kaplan_archive():
    config = ReportingRegimeBuildConfig(year_start=2023, year_end=2025)

    assert list(_srs_month_detail_years(config)) == [2023, 2024]


def test_source_selection_requests_the_target_year_reporting_artifact(
    tmp_path, monkeypatch
):
    state_dir = tmp_path / "state"
    artifact = state_dir / "modeling" / "agency_year_reporting_regimes.parquet"
    artifact.parent.mkdir(parents=True)
    pd.DataFrame({"year": [2025]}).to_parquet(artifact, index=False)
    seen = {}

    def current(paths, *, config, out_path):
        seen["year_end"] = config.year_end
        seen["out_path"] = out_path
        return True

    monkeypatch.setattr(
        "crimerisk.source_selection.reporting_regimes_artifact_is_current",
        current,
    )

    out = _load_reporting_regimes(
        SimpleNamespace(state_dir=state_dir), target_year=2025
    )

    assert seen == {"year_end": 2025, "out_path": artifact}
    assert out.loc[0, "year"] == 2025


def test_trend_panel_reuses_one_reporting_artifact_through_target_year(monkeypatch):
    calls = []

    def preferred(*, paths, year, reporting_year_end, force_reporting_regimes_rebuild):
        calls.append((year, reporting_year_end))
        return pd.DataFrame()

    monkeypatch.setattr(
        "crimerisk.trend_fills.build_agency_preferred_observations", preferred
    )

    out = build_agency_trend_fill_panel(
        paths=SimpleNamespace(), year_start=2023, year_end=2025
    )

    assert out.empty
    assert calls == [(2023, 2025), (2024, 2025), (2025, 2025)]


def test_table11_exact_unique_agency_name_is_promoted():
    rows = pd.DataFrame(
        {
            "publication_collection": ["table11_state_tribal_other"],
            "state_abbr": ["TX"],
            "publication_name_std": ["TEXAS PARK POLICE"],
            "publication_unit_raw": [pd.NA],
            "offense": ["robbery"],
            "official_count": [4],
        }
    )
    master = pd.DataFrame(
        {
            "ori9": ["TX0000100"],
            "state_abbr": ["TX"],
            "agency_type_norm": ["state_law_enforcement"],
            "agency_name_std": ["TEXAS PARK POLICE"],
            "crosswalk_agency_name_std": [pd.NA],
            "census_name_std": [pd.NA],
        }
    )

    matched = _match_cius_table11_agencies(rows, agency_master=master)

    assert matched.loc[0, "ori9"] == "TX0000100"


def test_table11_md_state_police_county_unit_uses_reviewed_successor_identity():
    rows = pd.DataFrame(
        {
            "publication_collection": ["table11_state_tribal_other"],
            "state_abbr": ["MD"],
            "publication_name_std": ["STATE POLICE"],
            "publication_unit_raw": ["Anne Arundel County"],
            "offense": ["robbery"],
            "official_count": [9],
        }
    )
    master = pd.DataFrame(
        {
            "ori9": ["MDMSP5200"],
            "state_abbr": ["MD"],
            "agency_type_norm": ["state_law_enforcement"],
            "agency_name_std": ["STATE POLICE GLEN BURNIE"],
            "crosswalk_agency_name_std": [pd.NA],
            "census_name_std": [pd.NA],
        }
    )
    successions = pd.DataFrame(
        {
            "case_id": ["mdsp-ori-migration-anne-arundel"],
            "successor_ori": ["MDMSP5200"],
        }
    )

    matched = _match_cius_table11_agencies(
        rows,
        agency_master=master,
        explicit_successions=successions,
    )

    assert matched.loc[0, "ori9"] == "MDMSP5200"
