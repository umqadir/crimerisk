from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.model_surface import (
    MIN_POSITIVE_PRIOR_RATE_PER_100K,
    _merge_frozen_structural_feature_cache,
    _strictly_positive_rate_from_log1p_prediction,
)
from crimerisk.confidence import _level_provenance_text
from crimerisk.geometry import _load_state_municipal_polygons
from crimerisk.level_lane import _attach_population_for_plausibility, _soft_review_keys
from crimerisk.jurisdiction_reference import (
    _apply_municipal_geometry_overrides,
    _assert_no_institutional_municipal_exclusives,
    _reclassify_nongovernment_census_geographies,
)
from crimerisk.paths import get_paths


def _municipal_override_input(*, state_abbr: str, state_fips: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ori9": [f"{state_abbr}0010000"],
            "state_abbr": [state_abbr],
            "state_fips": [state_fips],
            "final_decision": ["municipal_place"],
            "resolved_geo_type": ["place"],
            "resolved_geoid": [f"{state_fips}99999"],
            "resolved_label": ["Original place"],
            "manual_review_flag": [False],
            "resolution_source": ["test"],
            "fallback_applied": [False],
            "has_final_municipal_geoid": [True],
            "match_method": ["test"],
        }
    )


def _write_municipal_override(path, *, jurisdiction_id: str, state_fips: str) -> None:
    pd.DataFrame(
        {
            "jurisdiction_id": [jurisdiction_id],
            "decision": ["map_to_cousub"],
            "replacement_geo_type": ["cousub"],
            "replacement_geoid": [f"{state_fips}5555555"],
            "replacement_jurisdiction_name": ["Replacement"],
            "reason": ["test"],
            "sources": ["test"],
        }
    ).to_csv(path, index=False)


def test_out_of_scope_municipal_override_does_not_require_missing_geometry(
    tmp_path, monkeypatch
) -> None:
    frame = _municipal_override_input(state_abbr="AK", state_fips="02")
    override_path = tmp_path / "overrides.csv"
    _write_municipal_override(
        override_path,
        jurisdiction_id="02:municipal:place:0299999",
        state_fips="02",
    )
    monkeypatch.setattr(
        "crimerisk.jurisdiction_reference._load_all_tiger_lookups",
        lambda paths: pd.DataFrame(
            columns=["state_fips", "geo_type", "geoid", "namelsad"]
        ),
    )

    out = _apply_municipal_geometry_overrides(frame, override_path=override_path)

    assert out.loc[0, "resolved_geoid"] == "0299999"
    assert out.loc[0, "resolution_source"] == "test"


def test_in_scope_municipal_override_still_requires_target_geometry(
    tmp_path, monkeypatch
) -> None:
    frame = _municipal_override_input(state_abbr="TX", state_fips="48")
    override_path = tmp_path / "overrides.csv"
    _write_municipal_override(
        override_path,
        jurisdiction_id="48:municipal:place:4899999",
        state_fips="48",
    )
    monkeypatch.setattr(
        "crimerisk.jurisdiction_reference._load_all_tiger_lookups",
        lambda paths: pd.DataFrame(
            columns=["state_fips", "geo_type", "geoid", "namelsad"]
        ),
    )

    with np.testing.assert_raises_regex(ValueError, "missing TIGER geometry"):
        _apply_municipal_geometry_overrides(frame, override_path=override_path)


def test_negative_log_rate_predictions_keep_strictly_positive_support() -> None:
    rates = _strictly_positive_rate_from_log1p_prediction(
        np.array([-100.0, -1.0, 0.0, np.log1p(12.5)])
    )
    assert np.all(rates > 0.0)
    assert rates[0] == rates[1] == MIN_POSITIVE_PRIOR_RATE_PER_100K
    np.testing.assert_allclose(rates[-1], 12.5, rtol=1e-12)


def test_frozen_structural_feature_cache_only_restores_missing_columns(tmp_path) -> None:
    cache_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "bg_id": ["1", "2"],
            "fresh_feature": [100.0, 200.0],
            "cleaned_feature": [3.0, 4.0],
        }
    ).to_parquet(cache_path, index=False)
    bg = pd.DataFrame(
        {
            "bg_id": ["000000000001", "000000000002"],
            "fresh_feature": [10.0, 20.0],
        }
    )

    out = _merge_frozen_structural_feature_cache(bg, feature_cache_path=cache_path)

    assert out["fresh_feature"].tolist() == [10.0, 20.0]
    assert out["cleaned_feature"].tolist() == [3.0, 4.0]


def test_frozen_structural_feature_cache_requires_complete_bg_coverage(tmp_path) -> None:
    cache_path = tmp_path / "features.parquet"
    pd.DataFrame({"bg_id": ["1"], "cleaned_feature": [3.0]}).to_parquet(
        cache_path,
        index=False,
    )
    bg = pd.DataFrame({"bg_id": ["000000000001", "000000000002"]})

    with np.testing.assert_raises_regex(ValueError, "does not cover"):
        _merge_frozen_structural_feature_cache(bg, feature_cache_path=cache_path)


def test_level_provenance_cannot_call_imputed_or_unknown_rows_observed() -> None:
    assert "modeled benchmark" in _level_provenance_text(
        "valid_complete_year", "none", False, 0.2
    )
    assert "statewide residual" in _level_provenance_text(
        "<NA>", "none", False, 0.0
    )


def test_implausibly_low_complete_vector_is_held_for_review() -> None:
    offenses = [
        "murder",
        "rape",
        "robbery",
        "aggravated_assault",
        "burglary",
        "larceny",
        "motor_vehicle_theft",
    ]
    panel = pd.DataFrame(
        [
            {
                "ori9": "ZZ0000001",
                "year": year,
                "offense": offense,
                "preferred_count": count,
                "usable_as_observed": True,
                "population": 50_000,
            }
            for year, count in [(2023, 100.0), (2024, 1.0)]
            for offense in offenses
        ]
    )
    held = _soft_review_keys(panel, year=2024, protected=set())
    assert held.loc[0, "ori9"] == "ZZ0000001"
    assert "implausibly_low_part1_rate" in held.loc[0, "soft_reason_codes"]


def test_plausibility_population_is_loaded_from_agency_master(tmp_path) -> None:
    reference = tmp_path / "state" / "reference"
    reference.mkdir(parents=True)
    pd.DataFrame(
        {"ori9": ["ZZ0010000"], "population_latest_nibrs": [12_345.0]}
    ).to_parquet(reference / "agency_master.parquet", index=False)
    paths = type("Paths", (), {"state_dir": tmp_path / "state"})()
    panel = pd.DataFrame({"ori9": ["zz0010000"]})
    enriched = _attach_population_for_plausibility(panel, paths=paths)
    assert enriched.loc[0, "population"] == 12_345.0


def test_ccd_is_rejected_as_a_municipal_jurisdiction() -> None:
    row = pd.DataFrame(
        {
            "final_decision": ["municipal_cousub"],
            "resolved_label": ["Augusta CCD"],
            "resolved_geo_type": ["cousub"],
            "resolved_geoid": ["1324590168"],
            "manual_review_flag": [False],
            "resolution_source": ["provisional_auto"],
            "has_final_municipal_geoid": [True],
            "match_method": ["name"],
        }
    )
    out = _reclassify_nongovernment_census_geographies(row)
    assert out.loc[0, "final_decision"] == "reclassify_nonmunicipal"
    assert pd.isna(out.loc[0, "resolved_geoid"])


def test_reviewed_cdp_service_area_is_retained() -> None:
    row = pd.DataFrame(
        {
            "final_decision": ["municipal_place"],
            "resolved_label": ["Ocean Pines CDP"],
            "resolved_geo_type": ["place"],
            "resolved_geoid": ["2458275"],
            "manual_review_flag": [True],
            "resolution_source": ["local_resolution_override"],
            "has_final_municipal_geoid": [True],
            "match_method": ["manual_override"],
        }
    )
    out = _reclassify_nongovernment_census_geographies(row)
    assert out.loc[0, "final_decision"] == "municipal_place"
    assert out.loc[0, "resolved_geoid"] == "2458275"


def test_school_police_cannot_be_an_exclusive_municipal_reporter() -> None:
    crosswalk = pd.DataFrame(
        {
            "ori": ["GA0000001"],
            "jurisdiction_id": ["13:municipal:place:1304000"],
            "relationship_type": ["exclusive"],
        }
    )
    agencies = pd.DataFrame(
        {
            "ori9": ["GA0000001"],
            "agency_name_raw": ["Example Public Schools"],
            "agency_type_norm": ["local_police"],
        }
    )
    with np.testing.assert_raises(ValueError):
        _assert_no_institutional_municipal_exclusives(crosswalk, agencies)


def test_louisville_consolidated_geometry_includes_old_city_piece() -> None:
    paths = get_paths()
    master = pd.read_parquet(paths.state_dir / "reference" / "jurisdiction_master.parquet")
    polys = _load_state_municipal_polygons(paths, "21", master)
    louisville = polys[
        polys["jurisdiction_id"].eq("21:municipal:place:2148006")
    ]
    assert {"2148006", "2148000"}.issubset(set(louisville["geoid"].astype(str)))


def test_pine_township_replaces_unlocalized_northern_regional_share() -> None:
    paths = get_paths()
    crosswalk = pd.read_parquet(
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    )
    pine = crosswalk[
        crosswalk["ori"].eq("PA0021B00")
        & crosswalk["jurisdiction_id"].eq("42:police_service:PA0021B00")
    ]
    assert len(pine) == 1
    assert pine.iloc[0]["weight"] > 0
    registry = pd.read_csv(
        paths.repo_root / "configs" / "pa_police_service_coverage.csv", dtype="string"
    )
    pine_mcd = registry[registry["municipality_geoid"].eq("4200360272")]
    assert len(pine_mcd) == 1
    assert pine_mcd.iloc[0]["service_area_id"] == "42:police_service:PA0021B00"


def test_standing_rock_has_an_exclusive_aiannh_footprint() -> None:
    paths = get_paths()
    override = pd.read_csv(
        paths.repo_root / "configs/overlap_footprint_overrides.csv", dtype="string"
    )
    row = override[override["ori"].eq("NDDI00300")].iloc[0]
    assert row["final_overlap_treatment"] == "localize_to_custom_footprint"
    assert row["displaces_county_remainder"] == "TRUE"
    footprint = pd.read_csv(
        paths.repo_root / "configs/overlap_custom_footprints.csv", dtype="string"
    )
    footprint = footprint[footprint["ori"].eq("NDDI00300")]
    assert set(footprint["block_group_geoid"].str.zfill(12)) == {
        "380859408001",
        "380859408002",
        "380859409001",
        "380859409002",
    }
    np.testing.assert_allclose(
        pd.to_numeric(footprint["weight_share"]).sum(), 1.0, atol=1e-12
    )


def test_ramah_navajo_uses_the_census_tribal_subdivision_footprint() -> None:
    paths = get_paths()
    override = pd.read_csv(
        paths.repo_root / "configs/overlap_footprint_overrides.csv", dtype="string"
    )
    row = override[override["ori"].eq("NM0170400")].iloc[0]
    assert row["footprint_type"] == "aitsn_tribal_subdivision_footprint"
    assert row["geometry_source_type"] == "census_2020_tiger_aitsn"
    footprint = pd.read_csv(
        paths.repo_root / "configs/overlap_custom_footprints.csv", dtype="string"
    )
    footprint = footprint[footprint["ori"].eq("NM0170400")].copy()
    assert set(footprint["block_group_geoid"].str.zfill(12)) == {
        "350069458001",
        "350069458002",
    }
    np.testing.assert_allclose(
        pd.to_numeric(footprint["weight_share"]),
        [1064 / 1613, 549 / 1613],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pd.to_numeric(footprint["bg_population_coverage_share"]),
        [1064 / 1291, 1127 / 1297],
        atol=1e-12,
    )
    responsibility = pd.read_csv(
        paths.repo_root
        / "configs"
        / "overlap_custom_footprint_resident_coverage.csv",
        dtype="string",
    )
    responsibility = responsibility[responsibility["ori"].eq("NM0170400")]
    np.testing.assert_allclose(
        pd.to_numeric(
            responsibility["bg_responsibility_population_coverage_share"]
        ),
        [1064 / 1291, 549 / 1297],
        atol=1e-12,
    )


def test_round3_truth_panel_is_unique_and_contains_all_named_headliners() -> None:
    panel = pd.read_csv(get_paths().repo_root / "configs/round3_truth_panel.csv")
    assert len(panel) == 58
    assert not panel["case_id"].duplicated().any()
    assert {
        "augusta-ccd",
        "louisville-old-city-piece",
        "ct-litchfield",
        "pine-township-bg-1",
        "pine-ridge-footprint",
        "standing-rock-footprint",
        "navajo-az-nm-seam",
        "wichita-repair",
        "nyc-overlap",
        "exact-zero-population",
        "popup-imputed-observed",
        "manhattan-hole-east-harlem",
    }.issubset(set(panel["case_id"]))
