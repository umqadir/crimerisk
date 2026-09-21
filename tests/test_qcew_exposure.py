"""QCEW exposure updating (contract: QCEW_EXPOSURE_CONTRACT.md).

Six things are load-bearing and are asserted here: the factor is a plain employment ratio and
nothing cleverer, the clip band catches a disclosure artifact and says which county it caught, the
coverage ladder falls county -> state -> identity for a real reason (Connecticut's 2024
planning-region renumbering) and never silently, the scaled legs move only their LODES-derived
term, the artifact publishes both vintages and its tier, and the flag-off artifact is byte- and
schema-identical to the one this lane did not exist for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.exposure_ensemble import (
    EXPOSURE_NORMALIZER_COLUMNS,
    NORMALIZER_SEMANTICS,
    QCEW_UPDATED_NORMALIZER_SEMANTICS,
    QCEW_UPDATING_COLUMNS,
    ExposureEnsembleConfig,
    ExposureEnsembleRuntime,
    assert_exposure_normalizer_invariants,
    exposure_normalizer_columns,
    exposure_normalizer_dependency_paths,
    exposure_normalizers_path,
    exposure_normalizers_summary_path,
    is_qcew_updated,
    load_ensemble_weights,
    normalizer_semantics,
    ensemble_weights_path,
    summarize_exposure_normalizers,
)
from crimerisk.paths import RepoPaths
from crimerisk.qcew_exposure import (
    LODES_VINTAGE_YEARS,
    MODELED_PLACE_EXPOSURE_PROXY,
    QCEW_RETAIL_INDUSTRY_CODE,
    QCEW_SCALE_VERSION,
    QCEW_TOTAL_INDUSTRY_CODE,
    RETAIL_LEG,
    SCALE_FACTOR_CEILING,
    SCALE_FACTOR_FLOOR,
    SCALE_TIERS,
    TIER_COUNTY,
    TIER_IDENTITY,
    TIER_STATE,
    TOTAL_LEG,
    QcewScalingConfig,
    assert_scale_factor_invariants,
    attach_scale_factors,
    build_county_scale_factors,
    clip_to_band,
    county_geoid_from_bg,
    qcew_annual_path,
    qcew_annual_url,
    qcew_employment,
    scale_daytime_jobs_leg,
    scale_factor_from_totals,
    scale_retail_leg,
    summarize_applied_factors,
    summarize_scale_factors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)


# --- fixtures ---------------------------------------------------------------


QCEW_HEADER = (
    '"area_fips","own_code","industry_code","agglvl_code","size_code","year","qtr",'
    '"disclosure_code","annual_avg_estabs","annual_avg_emplvl"\n'
)


def _qcew_csv(path: Path, rows: list[tuple[str, int, str, int, str, float]]) -> Path:
    """Write a minimal QCEW annual slice: (area, own, industry, agglvl, disclosure, emplvl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [QCEW_HEADER]
    for area, own, industry, agglvl, disclosure, emplvl in rows:
        lines.append(
            f'"{area}","{own}","{industry}","{agglvl}","0","2024","A","{disclosure}",1,{emplvl}\n'
        )
    path.write_text("".join(lines))
    return path


def _write_year(base: Path, *, year: int, total: dict[str, float], retail: dict[str, float]) -> None:
    """One (year) pair of QCEW files, county rows plus the state rows summed from them."""
    for industry, agglvl_county, agglvl_state, values in (
        (QCEW_TOTAL_INDUSTRY_CODE, 70, 50, total),
        (QCEW_RETAIL_INDUSTRY_CODE, 74, 54, retail),
    ):
        rows: list[tuple[str, int, str, int, str, float]] = []
        state_totals: dict[str, float] = {}
        for area, value in values.items():
            rows.append((area, 5, industry, agglvl_county, "", float(value)))
            state_totals[area[:2]] = state_totals.get(area[:2], 0.0) + float(value)
        for state, value in state_totals.items():
            rows.append((f"{state}000", 5, industry, agglvl_state, "", float(value)))
        slug = "10" if industry == QCEW_TOTAL_INDUSTRY_CODE else "44_45"
        _qcew_csv(base / "qcew" / "raw" / f"qcew_{year}_a_industry_{slug}.csv", rows)


def _paths_with_qcew(tmp_path: Path, *, source_year: int = 2023) -> RepoPaths:
    """A repo whose QCEW cache holds a hand-built two-year fixture."""
    paths = RepoPaths.from_repo_root(tmp_path)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    # 06001 grows 20%, 06002 shrinks 20%, 36001 is flat, 09001 exists only in the source year
    # (the Connecticut case), 06003 has a suppressed retail cell in the target year.
    _write_year(
        paths.data_dir,
        year=source_year,
        total={"06001": 1000.0, "06002": 1000.0, "06003": 500.0, "36001": 100.0, "09001": 700.0},
        retail={"06001": 100.0, "06002": 200.0, "06003": 50.0, "36001": 10.0, "09001": 70.0},
    )
    _write_year(
        paths.data_dir,
        year=2024,
        total={"06001": 1200.0, "06002": 800.0, "06003": 500.0, "36001": 100.0, "09110": 770.0},
        retail={"06001": 110.0, "06002": 200.0, "06003": 0.0, "36001": 10.0, "09110": 77.0},
    )
    return paths


# --- reading QCEW -----------------------------------------------------------


def test_open_data_urls_carry_the_underscore_sector_slug():
    # BLS's own path vocabulary: the retail sector is `44-45` in the data and `44_45` in the URL.
    assert qcew_annual_url(year=2024, industry_code=QCEW_TOTAL_INDUSTRY_CODE).endswith(
        "/2024/a/industry/10.csv"
    )
    assert qcew_annual_url(year=2024, industry_code=QCEW_RETAIL_INDUSTRY_CODE).endswith(
        "/2024/a/industry/44_45.csv"
    )
    with pytest.raises(KeyError):
        qcew_annual_url(year=2024, industry_code="23")


def test_county_employment_sums_ownership_and_drops_statewide_residuals(tmp_path: Path):
    path = _qcew_csv(
        tmp_path / "q.csv",
        [
            ("06001", 5, "44-45", 74, "", 100.0),
            ("06001", 3, "44-45", 74, "", 5.0),
            ("06999", 5, "44-45", 74, "", 40.0),  # statewide "unknown or undefined"
            ("C0600", 5, "44-45", 74, "", 900.0),  # an MSA, not a county
            ("06000", 5, "44-45", 54, "", 105.0),
        ],
    )
    counties = qcew_employment(path, industry_code=QCEW_RETAIL_INDUSTRY_CODE, level="county")
    assert counties.to_dict() == {"06001": 105.0}
    states = qcew_employment(path, industry_code=QCEW_RETAIL_INDUSTRY_CODE, level="state")
    assert states.to_dict() == {"06": 105.0}


def test_missing_qcew_file_names_the_pull_script(tmp_path: Path):
    paths = RepoPaths.from_repo_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="pull_qcew_county_employment"):
        qcew_employment(
            qcew_annual_path(paths, year=2024, industry_code=QCEW_TOTAL_INDUSTRY_CODE),
            industry_code=QCEW_TOTAL_INDUSTRY_CODE,
            level="county",
        )


def test_a_file_without_the_expected_aggregation_level_is_rejected(tmp_path: Path):
    path = _qcew_csv(tmp_path / "q.csv", [("06000", 5, "10", 50, "", 1000.0)])
    with pytest.raises(ValueError, match="annual industry slice"):
        qcew_employment(path, industry_code=QCEW_TOTAL_INDUSTRY_CODE, level="county")


# --- factor math ------------------------------------------------------------


def test_factor_is_the_plain_employment_ratio():
    source = pd.Series({"06001": 1000.0, "06002": 1000.0, "36001": 100.0})
    target = pd.Series({"06001": 1200.0, "06002": 800.0, "36001": 100.0})
    factors = scale_factor_from_totals(source, target)
    assert factors["06001"] == pytest.approx(1.2)
    assert factors["06002"] == pytest.approx(0.8)
    assert factors["36001"] == pytest.approx(1.0)


def test_a_zero_on_either_side_is_undefined_rather_than_zero_or_infinite():
    # QCEW publishes a suppressed cell as 0. A ratio built on it is a disclosure artifact, not an
    # employment collapse, so it must not survive as a number.
    source = pd.Series({"a": 0.0, "b": 100.0, "c": 0.0})
    target = pd.Series({"a": 100.0, "b": 0.0, "c": 0.0})
    factors = scale_factor_from_totals(source, target)
    assert factors.isna().all()


def test_clip_band_moves_only_out_of_band_entries_and_reports_which():
    values = pd.Series({"a": 0.002, "b": 1.05, "c": 3.4, "d": np.nan})
    clipped, moved = clip_to_band(values)
    assert clipped["a"] == pytest.approx(SCALE_FACTOR_FLOOR)
    assert clipped["b"] == pytest.approx(1.05)
    assert clipped["c"] == pytest.approx(SCALE_FACTOR_CEILING)
    assert moved.tolist() == [True, False, True, False]


def test_the_band_brackets_unity_and_a_band_that_does_not_is_rejected():
    assert SCALE_FACTOR_FLOOR < 1.0 < SCALE_FACTOR_CEILING
    with pytest.raises(ValueError, match="bracket 1.0"):
        QcewScalingConfig(floor=1.1, ceiling=2.0)
    with pytest.raises(ValueError, match="bracket 1.0"):
        QcewScalingConfig(floor=0.0, ceiling=2.0)


def test_county_factors_use_the_county_rung_where_both_years_have_it(tmp_path: Path):
    factors = build_county_scale_factors(paths=_paths_with_qcew(tmp_path), source_year=2023)
    total = factors[factors["leg"].eq(TOTAL_LEG)].set_index("county_geoid")
    assert total.loc["06001", "factor"] == pytest.approx(1.2)
    assert total.loc["06001", "tier"] == TIER_COUNTY
    assert total.loc["06002", "factor"] == pytest.approx(0.8)
    assert total.loc["36001", "factor"] == pytest.approx(1.0)


def test_a_renumbered_county_falls_to_its_state_rung_rather_than_to_unity(tmp_path: Path):
    # Connecticut in miniature: 09001 exists in 2023 and not in 2024, and the state ratio (1.1)
    # is well defined across the renumbering while no county ratio is.
    factors = build_county_scale_factors(paths=_paths_with_qcew(tmp_path), source_year=2023)
    total = factors[factors["leg"].eq(TOTAL_LEG)].set_index("county_geoid")
    assert total.loc["09001", "tier"] == TIER_STATE
    assert total.loc["09001", "factor"] == pytest.approx(1.1)


def test_a_suppressed_retail_cell_falls_to_the_state_rung(tmp_path: Path):
    factors = build_county_scale_factors(paths=_paths_with_qcew(tmp_path), source_year=2023)
    retail = factors[factors["leg"].eq(RETAIL_LEG)].set_index("county_geoid")
    assert retail.loc["06003", "tier"] == TIER_STATE
    # 06 retail: 350 -> 310 once the suppressed county reads 0, which is the state rung's own
    # number. The point is that the county's own 50 -> 0 never becomes a factor.
    assert retail.loc["06003", "factor"] == pytest.approx(310.0 / 350.0)


def test_scale_factor_invariants_reject_an_out_of_band_or_mislabelled_table(tmp_path: Path):
    factors = build_county_scale_factors(paths=_paths_with_qcew(tmp_path), source_year=2023)
    assert_scale_factor_invariants(factors)

    broken = factors.copy()
    broken.loc[0, "factor"] = 9.0
    with pytest.raises(ValueError, match="outside the band"):
        assert_scale_factor_invariants(broken)

    broken = factors.copy()
    broken.loc[0, "tier"] = "vibes"
    with pytest.raises(ValueError, match="unknown tiers"):
        assert_scale_factor_invariants(broken)

    broken = factors.copy()
    broken.loc[0, "tier"] = TIER_IDENTITY
    broken.loc[0, "factor"] = 1.3
    with pytest.raises(ValueError, match="identity-tier"):
        assert_scale_factor_invariants(broken)


def test_summary_reports_the_distribution_and_names_every_clipped_county(tmp_path: Path):
    factors = build_county_scale_factors(paths=_paths_with_qcew(tmp_path), source_year=2023)
    factors.loc[factors["county_geoid"].eq("06001") & factors["leg"].eq(TOTAL_LEG), "raw_factor"] = 5.0
    factors.loc[factors["county_geoid"].eq("06001") & factors["leg"].eq(TOTAL_LEG), "factor"] = SCALE_FACTOR_CEILING
    factors.loc[factors["county_geoid"].eq("06001") & factors["leg"].eq(TOTAL_LEG), "clipped"] = True
    summary = summarize_scale_factors(factors)
    entry = summary["by_source_year"]["2023"][TOTAL_LEG]
    assert entry["out_of_band_counties"] == 1
    assert entry["out_of_band_county_geoids"] == ["06001"]
    assert set(entry["tier_counts"]) == set(SCALE_TIERS)
    assert summary["scale_version"] == QCEW_SCALE_VERSION
    assert summary["ags_values_used"] is False


# --- attachment and the coverage ladder -------------------------------------


def _factor_table() -> pd.DataFrame:
    rows = []
    for leg, factor in ((TOTAL_LEG, 1.2), (RETAIL_LEG, 0.9)):
        rows.append(
            {
                "county_geoid": "06001",
                "source_year": 2023,
                "target_year": 2024,
                "leg": leg,
                "source_employment": 1000.0,
                "target_employment": 1000.0 * factor,
                "raw_factor": factor,
                "factor": factor,
                "tier": TIER_COUNTY,
                "clipped": False,
            }
        )
    return pd.DataFrame(rows)


def test_attach_keys_on_county_and_vintage_together():
    frame = pd.DataFrame({"bg_id": ["060010001001", "060010001002"]})
    applied = attach_scale_factors(
        frame, factors=_factor_table(), lodes_vintage=pd.Series([2023, 2021])
    )
    assert applied[f"qcew_{TOTAL_LEG}_scale_factor"].tolist() == [1.2, 1.0]
    # The 2021 row has no 2021 factor in the table, so it must not borrow the 2023 one.
    assert applied[f"qcew_{TOTAL_LEG}_scale_tier"].tolist() == [TIER_COUNTY, TIER_IDENTITY]
    assert applied["lodes_vintage"].tolist() == [2023, 2021]
    assert applied["qcew_scale_vintage"].tolist() == [2024, 2024]


def test_a_block_group_with_no_vintage_takes_the_identity_rung():
    frame = pd.DataFrame({"bg_id": ["060010001001"]})
    applied = attach_scale_factors(
        frame, factors=_factor_table(), lodes_vintage=pd.Series([np.nan])
    )
    assert applied[f"qcew_{TOTAL_LEG}_scale_factor"].tolist() == [1.0]
    assert applied[f"qcew_{RETAIL_LEG}_scale_tier"].tolist() == [TIER_IDENTITY]


def test_county_geoid_is_the_first_five_digits_of_a_zero_filled_bg_id():
    assert county_geoid_from_bg(pd.Series(["60010001001"])).tolist() == ["06001"]


# --- leg arithmetic ---------------------------------------------------------


def test_only_the_net_commuter_term_moves():
    # 500 residents, 300 workplace jobs, 100 resident jobs. At f = 1.2 the leg is
    # 500 + 1.2 * 200 = 740, not 1.2 * 700 = 840: the resident term is an ACS quantity already at
    # the release vintage and an employment ratio has no business multiplying it.
    leg = scale_daytime_jobs_leg(
        population=np.array([500.0]),
        jobs_wac=np.array([300.0]),
        jobs_rac=np.array([100.0]),
        factor=np.array([1.2]),
    )
    assert leg.tolist() == [pytest.approx(740.0)]


def test_the_identity_factor_reproduces_the_deployed_leg_exactly():
    population = np.array([500.0, 100.0, 0.0])
    wac = np.array([300.0, 10.0, 0.0])
    rac = np.array([100.0, 400.0, 0.0])
    deployed = np.maximum(np.clip(population + wac - rac, 0.0, None), population)
    scaled = scale_daytime_jobs_leg(
        population=population, jobs_wac=wac, jobs_rac=rac, factor=np.ones(3)
    )
    assert scaled.tolist() == deployed.tolist()


def test_the_leg_keeps_its_resident_floor_in_a_bedroom_community():
    # A shrinking-employment factor in a net-exporting block group cannot push the leg below the
    # people who live there; the floor is the deployed leg's own and it is kept.
    leg = scale_daytime_jobs_leg(
        population=np.array([500.0]),
        jobs_wac=np.array([10.0]),
        jobs_rac=np.array([400.0]),
        factor=np.array([1.5]),
    )
    assert leg.tolist() == [500.0]


def test_the_retail_leg_is_a_straight_scaling_that_cannot_go_negative():
    scaled = scale_retail_leg(
        retail_jobs=np.array([100.0, 0.0, 50.0]), factor=np.array([0.9, 1.5, 1.0])
    )
    assert scaled.tolist() == [pytest.approx(90.0), 0.0, pytest.approx(50.0)]


# --- schema, semantics and flag-off safety ----------------------------------


def test_semantics_string_gains_the_modeled_place_exposure_proxy_only_when_scaled():
    assert normalizer_semantics(qcew_updated=False) == NORMALIZER_SEMANTICS
    assert MODELED_PLACE_EXPOSURE_PROXY not in NORMALIZER_SEMANTICS
    assert MODELED_PLACE_EXPOSURE_PROXY in QCEW_UPDATED_NORMALIZER_SEMANTICS
    assert NORMALIZER_SEMANTICS in QCEW_UPDATED_NORMALIZER_SEMANTICS
    assert normalizer_semantics(qcew_updated=True) == QCEW_UPDATED_NORMALIZER_SEMANTICS


def test_flag_is_default_off_and_adds_no_columns_when_off():
    config = ExposureEnsembleConfig()
    assert config.enable_qcew_exposure_updating is False
    assert exposure_normalizer_columns(qcew_updated=False) == list(EXPOSURE_NORMALIZER_COLUMNS)
    assert exposure_normalizer_columns(qcew_updated=True) == [
        *EXPOSURE_NORMALIZER_COLUMNS,
        *QCEW_UPDATING_COLUMNS,
    ]


def test_the_two_modes_are_different_artifacts_not_different_vintages_of_one():
    legacy = exposure_normalizers_path(PATHS, year=2024)
    updated = exposure_normalizers_path(PATHS, year=2024, qcew_updated=True)
    assert legacy != updated
    assert updated.name == "bg_exposure_normalizers_2024_qcew.parquet"
    assert exposure_normalizers_summary_path(PATHS, year=2024, qcew_updated=True).name.endswith(
        "_qcew.summary.json"
    )


def test_qcew_files_join_the_dependency_set_only_when_the_flag_is_on():
    off = exposure_normalizer_dependency_paths(
        PATHS, config=ExposureEnsembleConfig(year=2024)
    )
    on = exposure_normalizer_dependency_paths(
        PATHS, config=ExposureEnsembleConfig(year=2024, enable_qcew_exposure_updating=True)
    )
    added = [path for path in on if path not in off]
    assert added, "the QCEW inputs must be part of the freshness stamp when the lane reads them"
    assert all("qcew" in str(path) for path in added)
    # Every declared LODES vintage plus the target year, both industries.
    assert len(added) == 2 * len({*LODES_VINTAGE_YEARS, 2024})


# --- artifact invariants ----------------------------------------------------


def _normalizer_frame(*, qcew: bool) -> pd.DataFrame:
    from crimerisk.exposure_ensemble import (
        ENSEMBLE_OFFENSES,
        LANDSCAN_COVERAGE_REPAIR_COLUMN,
        opportunity_normalizer_column,
    )

    frame = pd.DataFrame(
        {
            "state_fips": ["06", "06"],
            "bg_id": ["060010001001", "060010001002"],
            "population": [100.0, 200.0],
            "landscan_night_leg": [100.0, 200.0],
            "landscan_day_leg": [100.0, 200.0],
            "daytime_jobs_leg": [100.0, 200.0],
            LANDSCAN_COVERAGE_REPAIR_COLUMN: [False, False],
            "destination_poi_total": [1.0, 2.0],
            "lodes_retail_jobs": [5.0, 6.0],
        }
    )
    for offense in ENSEMBLE_OFFENSES:
        frame[opportunity_normalizer_column(offense)] = [100.0, 200.0]
    if qcew:
        frame["lodes_vintage"] = pd.array([2023, 2023], dtype="Int64")
        frame["qcew_scale_vintage"] = [2024, 2024]
        frame[f"qcew_{TOTAL_LEG}_scale_factor"] = [1.01, 1.0]
        frame[f"qcew_{RETAIL_LEG}_scale_factor"] = [0.99, 1.0]
        frame[f"qcew_{TOTAL_LEG}_scale_tier"] = [TIER_COUNTY, TIER_IDENTITY]
        frame[f"qcew_{RETAIL_LEG}_scale_tier"] = [TIER_COUNTY, TIER_IDENTITY]
    return frame


def test_a_table_is_self_identifying_and_a_half_written_schema_is_rejected():
    assert is_qcew_updated(_normalizer_frame(qcew=False)) is False
    assert is_qcew_updated(_normalizer_frame(qcew=True)) is True
    partial = _normalizer_frame(qcew=True).drop(columns=[f"qcew_{RETAIL_LEG}_scale_tier"])
    with pytest.raises(ValueError, match="partial QCEW provenance schema"):
        is_qcew_updated(partial)


def test_invariants_accept_a_well_formed_qcew_table_and_reject_the_broken_cases():
    assert_exposure_normalizer_invariants(_normalizer_frame(qcew=True))

    broken = _normalizer_frame(qcew=True)
    broken[f"qcew_{TOTAL_LEG}_scale_factor"] = [4.0, 1.0]
    with pytest.raises(ValueError, match="outside the band"):
        assert_exposure_normalizer_invariants(broken)

    broken = _normalizer_frame(qcew=True)
    broken[f"qcew_{RETAIL_LEG}_scale_tier"] = ["guessed", TIER_IDENTITY]
    with pytest.raises(ValueError, match="unknown qcew retail tier"):
        assert_exposure_normalizer_invariants(broken)

    broken = _normalizer_frame(qcew=True)
    broken[f"qcew_{TOTAL_LEG}_scale_factor"] = [1.01, 1.4]  # identity tier, non-identity factor
    with pytest.raises(ValueError, match="identity-tier"):
        assert_exposure_normalizer_invariants(broken)

    broken = _normalizer_frame(qcew=True)
    broken["lodes_vintage"] = pd.array([2019, 2023], dtype="Int64")
    with pytest.raises(ValueError, match="undeclared LODES vintage"):
        assert_exposure_normalizer_invariants(broken)

    broken = _normalizer_frame(qcew=True)
    broken["qcew_scale_vintage"] = [2024, 2023]
    with pytest.raises(ValueError, match="exactly one QCEW scale vintage"):
        assert_exposure_normalizer_invariants(broken)

    broken = _normalizer_frame(qcew=True)
    broken["lodes_vintage"] = pd.array([None, 2023], dtype="Int64")
    with pytest.raises(ValueError, match="no LODES vintage"):
        assert_exposure_normalizer_invariants(broken)


def test_a_vintage_less_block_group_may_sit_on_the_identity_rung():
    frame = _normalizer_frame(qcew=True)
    frame["lodes_vintage"] = pd.array([2023, None], dtype="Int64")
    assert_exposure_normalizer_invariants(frame)


def test_manifest_block_records_the_vintages_and_the_band():
    runtime = ExposureEnsembleRuntime(
        normalizers=_normalizer_frame(qcew=True),
        weights=load_ensemble_weights(ensemble_weights_path(PATHS)),
        config=ExposureEnsembleConfig(year=2024, enable_qcew_exposure_updating=True),
    )
    summary = summarize_exposure_normalizers(runtime.normalizers, runtime=runtime)
    assert summary["semantics"] == QCEW_UPDATED_NORMALIZER_SEMANTICS
    block = summary["qcew_exposure_updating"]
    assert block["enabled"] is True
    assert block["scale_version"] == QCEW_SCALE_VERSION
    assert block["band"] == [SCALE_FACTOR_FLOOR, SCALE_FACTOR_CEILING]
    assert block["lodes_vintages"] == {"2023": 2}
    assert block["qcew_scale_vintage"] == 2024
    assert block["legs"][TOTAL_LEG]["tier_block_groups"][TIER_COUNTY] == 1


def test_manifest_block_says_off_for_a_legacy_surface():
    runtime = ExposureEnsembleRuntime(
        normalizers=_normalizer_frame(qcew=False),
        weights=load_ensemble_weights(ensemble_weights_path(PATHS)),
        config=ExposureEnsembleConfig(year=2024),
    )
    summary = summarize_exposure_normalizers(runtime.normalizers, runtime=runtime)
    assert summary["semantics"] == NORMALIZER_SEMANTICS
    assert summary["qcew_exposure_updating"] == {"enabled": False}


def test_applied_summary_counts_every_tier(tmp_path: Path):
    applied = attach_scale_factors(
        pd.DataFrame({"bg_id": ["060010001001", "060010001002"]}),
        factors=_factor_table(),
        lodes_vintage=pd.Series([2023, np.nan]),
    )
    summary = summarize_applied_factors(applied)
    assert summary["block_groups"] == 2
    assert summary["lodes_vintages"] == {"2023": 1}
    assert summary["legs"][RETAIL_LEG]["tier_block_groups"] == {
        TIER_COUNTY: 1,
        TIER_STATE: 0,
        TIER_IDENTITY: 1,
    }
