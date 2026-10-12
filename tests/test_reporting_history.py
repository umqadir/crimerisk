"""One adjudicated reporting history: a year refused anywhere is refused everywhere."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crimerisk.crime import OFFENSES_7
from crimerisk.jurisdiction_targets import (
    FULL_YEAR_ADMITTED_MIN_SHARE,
    build_jurisdiction_reported_rollup,
)
from crimerisk.paths import RepoPaths
from crimerisk.reporting_history import (
    ADMITTED_COMPLETE,
    ADMITTED_REPLACEMENT,
    REJECTED,
    adjudicate_history_years,
)
from crimerisk.smoothed_controls import SmoothedControlConfig, clean_panel_history

BASE = {
    "murder": 25.0, "rape": 80.0, "robbery": 230.0, "aggravated_assault": 640.0,
    "burglary": 600.0, "larceny": 3800.0, "motor_vehicle_theft": 560.0,
}
# Savannah-Chatham GA0250300's 2024 CIUS row: twelve months, full weight, and a vector
# that is not a year of policing for 240,000 residents.
SAVANNAH_2024 = {
    "murder": 23.0, "rape": 25.0, "robbery": 0.0, "aggravated_assault": 1.0,
    "burglary": 0.0, "larceny": 0.0, "motor_vehicle_theft": 0.0,
}


def _configs(tmp_path, hard_rows: list[dict] | None = None) -> RepoPaths:
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    pd.DataFrame(
        columns=["case_id", "ori9", "year", "offense", "adjudication", "reason_code", "reviewer", "evidence_note"]
    ).to_csv(configs / "level_lane_admission_registry.csv", index=False)
    pd.DataFrame(
        hard_rows or [],
        columns=[
            "case_id", "ori9", "year", "offense", "level1_status", "level2_status", "reason_code",
            "repair_mode", "replacement_count", "external_check_status", "evidence_source",
        ],
    ).to_csv(configs / "level_lane_hard_evidence.csv", index=False)
    pd.DataFrame(columns=["ori9", "year"]).to_csv(
        configs / "level_lane_structural_zero_screen.csv", index=False
    )
    observations = tmp_path / "state" / "observations"
    observations.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ori9": pd.Series(dtype="string"), "year": pd.Series(dtype=int), "source": pd.Series(dtype="string"),
            "offense": pd.Series(dtype="string"), "count": pd.Series(dtype=float),
            "months_reported": pd.Series(dtype=float), "months_missing": pd.Series(dtype=float),
            "nonzero_months": pd.Series(dtype=float), "population": pd.Series(dtype=float),
        }
    ).to_parquet(observations / "agency_year_observations.parquet", index=False)
    return RepoPaths.from_repo_root(tmp_path)


def _row(ori: str, year: int, offense: str, count: float, *, months: float = 12.0, population: float = 240_000.0):
    return {
        "ori9": ori, "state_fips": "13", "state_abbr": "GA", "year": year, "offense": offense,
        "preferred_count": float(count), "preferred_source": "cius_publication_annual",
        "preferred_months_reported": months, "preferred_observation_weight": 1.0 if months >= 12 else months / 12.0,
        "usable_as_observed": months >= 12, "current_row_is_true_partial": 0 < months < 12,
        "population": population, "reporting_regime": "annual_only_but_usable",
    }


def _panel(*, target_vector: dict[int, dict[str, float]] | None = None) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    for year in range(2018, 2026):
        vector = (target_vector or {}).get(year)
        for offense in OFFENSES_7:
            value = vector[offense] if vector is not None else BASE[offense] * float(rng.uniform(0.95, 1.05))
            rows.append(_row("GA0250300", year, offense, value))
    # Steady peers keep the measured envelopes defined.
    for index in range(10):
        for year in range(2018, 2026):
            for offense in OFFENSES_7:
                rows.append(_row(f"GA{index:07d}", year, offense, BASE[offense] * float(rng.uniform(0.9, 1.1)), population=200_000.0))
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _no_stage1_registries(monkeypatch):
    """The stage-1 registries are year-scoped production configs; none applies here."""
    import crimerisk.stage1_adjudications as stage1

    monkeypatch.setattr(stage1, "build_usability_directives", lambda paths, *, target_year: pd.DataFrame())


def test_a_history_year_the_lane_refuses_is_missing_not_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("CRIMERISK_LEVEL_RULES", raising=False)
    paths = _configs(tmp_path)
    raw = _panel(target_vector={2024: SAVANNAH_2024})
    adjudicated, history = adjudicate_history_years(raw, paths=paths, target_year=2025)

    row = history[history["ori9"].eq("GA0250300") & history["year"].eq(2024)].set_index("offense")
    assert set(row["history_status"]) == {REJECTED}
    assert row.loc["larceny", "history_original_count"] == 0.0
    assert row.loc["rape", "history_original_count"] == 25.0
    clean = history[history["ori9"].eq("GA0250300") & history["year"].lt(2024)]
    assert set(clean["history_status"]) == {ADMITTED_COMPLETE}

    panel_2024 = adjudicated[adjudicated["ori9"].eq("GA0250300") & adjudicated["year"].eq(2024)]
    assert not panel_2024["usable_as_observed"].any()
    # The target year is left to the target-year path, untouched.
    target = adjudicated[adjudicated["year"].eq(2025)].sort_values(["ori9", "offense"]).reset_index(drop=True)
    expected = raw[raw["year"].eq(2025)].sort_values(["ori9", "offense"]).reset_index(drop=True)
    assert np.allclose(target["preferred_count"], expected["preferred_count"])

    # Downstream: the jurisdiction-year rollup and the smoother read the same refusal.
    crosswalk = pd.DataFrame({"ori9": ["GA0250300"], "jurisdiction_id": ["13:municipal:place:1369000"], "weight": [1.0]})
    rollup = build_jurisdiction_reported_rollup(agency_panel=adjudicated, crosswalk=crosswalk)
    larceny = rollup[rollup["offense"].eq("larceny")].set_index("year")
    assert larceny.loc[2024, "full_year_admitted_share"] == 0.0
    assert larceny.loc[2023, "full_year_admitted_share"] == 1.0
    panel = larceny.reset_index().assign(offense="larceny", bucket_population=240_000.0)
    used = clean_panel_history(panel, config=SmoothedControlConfig(year=2025))
    assert 2024 not in set(used["year"])
    assert used["clean_count"].min() > 3000.0


def test_a_documented_replacement_is_the_admitted_history_value(tmp_path, monkeypatch):
    monkeypatch.delenv("CRIMERISK_LEVEL_RULES", raising=False)
    paths = _configs(
        tmp_path,
        hard_rows=[{
            "case_id": "aa-04b", "ori9": "GA0250300", "year": 2023, "offense": "aggravated_assault",
            "level1_status": "valid_complete_year", "level2_status": "wrong_category_mapping",
            "reason_code": "category_omitted", "repair_mode": "source_supported_replacement",
            "replacement_count": 700, "external_check_status": "contradiction_confirmed",
            "evidence_source": "test",
        }],
    )
    raw = _panel()
    raw.loc[raw["ori9"].eq("GA0250300") & raw["year"].eq(2023) & raw["offense"].eq("aggravated_assault"), "preferred_count"] = 90.0
    adjudicated, history = adjudicate_history_years(raw, paths=paths, target_year=2025)
    row = history[history["ori9"].eq("GA0250300") & history["year"].eq(2023) & history["offense"].eq("aggravated_assault")].iloc[0]
    assert row["history_status"] == ADMITTED_REPLACEMENT
    assert row["history_admitted_count"] == 700.0
    assert row["history_original_count"] == 90.0
    panel_row = adjudicated[adjudicated["ori9"].eq("GA0250300") & adjudicated["year"].eq(2023) & adjudicated["offense"].eq("aggravated_assault")].iloc[0]
    assert bool(panel_row["usable_as_observed"]) and panel_row["preferred_count"] == 700.0


def _two_agency_panel(*, twin_counts: dict[int, float], twin_months: dict[int, float]) -> pd.DataFrame:
    rows = []
    for year in range(2018, 2026):
        rows.append(_row("GA1060100", year, "motor_vehicle_theft", 500.0))
        rows.append(
            _row("GA1060500", year, "motor_vehicle_theft", twin_counts.get(year, 0.0), months=twin_months.get(year, 0.0), population=0.0)
        )
    panel = pd.DataFrame(rows)
    panel["usable_as_observed"] = panel["preferred_months_reported"].ge(12.0)
    return panel


def test_a_silent_agency_with_no_mass_does_not_disqualify_a_complete_city_year():
    """Columbus GA: the second ORI never files; the old mean-months test (11.97 < 12) threw
    away every complete year the city filed."""
    panel = _two_agency_panel(twin_counts={}, twin_months={})
    crosswalk = pd.DataFrame({
        "ori9": ["GA1060100", "GA1060500"], "jurisdiction_id": ["13:municipal:place:1319000"] * 2, "weight": [1.0, 1.0],
    })
    rollup = build_jurisdiction_reported_rollup(agency_panel=panel, crosswalk=crosswalk)
    assert rollup["full_year_admitted_share"].eq(1.0).all()
    used = clean_panel_history(rollup.assign(bucket_population=200_000.0), config=SmoothedControlConfig(year=2025))
    assert len(used) == 8


def test_a_missing_agency_that_carries_mass_disqualifies_the_year():
    # The twin files 300 a year except 2022, when it files nothing.
    counts = {year: 300.0 for year in range(2018, 2026) if year != 2022}
    months = {year: 12.0 for year in counts}
    panel = _two_agency_panel(twin_counts=counts, twin_months=months)
    crosswalk = pd.DataFrame({
        "ori9": ["GA1060100", "GA1060500"], "jurisdiction_id": ["13:municipal:place:1319000"] * 2, "weight": [1.0, 1.0],
    })
    rollup = build_jurisdiction_reported_rollup(agency_panel=panel, crosswalk=crosswalk).set_index("year")
    assert rollup.loc[2022, "full_year_admitted_share"] == pytest.approx(500.0 / 800.0)
    assert rollup.loc[2022, "full_year_admitted_share"] < FULL_YEAR_ADMITTED_MIN_SHARE
    assert rollup.loc[2021, "full_year_admitted_share"] == 1.0
    used = clean_panel_history(rollup.reset_index().assign(bucket_population=200_000.0), config=SmoothedControlConfig(year=2025))
    assert 2022 not in set(used["year"])
    assert used["clean_count"].eq(800.0).all()


# --- reporting-regime restrictions -------------------------------------------------


def _restrictions(tmp_path, rows: list[dict]) -> None:
    from crimerisk.level_lane import REGIME_RESTRICTION_COLUMNS, REGIME_RESTRICTIONS_FILENAME

    pd.DataFrame(rows, columns=REGIME_RESTRICTION_COLUMNS).to_csv(
        tmp_path / "configs" / REGIME_RESTRICTIONS_FILENAME, index=False
    )


CHICAGO_AA = {
    "restriction_id": "aa-04b-omitted", "ori9": "GA0250300", "offense": "aggravated_assault",
    "effective_from_year": "2021", "lifted_from_year": "", "level1_status": "valid_complete_year",
    "level2_status": "wrong_category_mapping", "reason_code": "category_omitted",
    "repair_mode": "decayed_own_history_or_pooled", "external_check_status": "contradiction_confirmed",
    "evidence_source": "test",
}


def test_a_restriction_carries_forward_into_every_later_year(tmp_path, monkeypatch):
    from crimerisk.level_lane import apply_level_lane_admission, load_hard_evidence

    monkeypatch.delenv("CRIMERISK_LEVEL_RULES", raising=False)
    paths = _configs(tmp_path)
    _restrictions(tmp_path, [CHICAGO_AA])
    for year in (2021, 2024, 2025, 2031):
        rows = load_hard_evidence(paths, year=year)
        assert list(rows["case_id"]) == ["aa-04b-omitted"] and int(rows["year"].iloc[0]) == year
    assert load_hard_evidence(paths, year=2020).empty

    raw = _panel()
    adjudicated, history = adjudicate_history_years(raw, paths=paths, target_year=2025)
    aa = history[history["ori9"].eq("GA0250300") & history["offense"].eq("aggravated_assault")].set_index("year")
    assert set(aa.loc[2021:2024, "history_status"]) == {REJECTED}
    assert set(aa.loc[2018:2020, "history_status"]) == {ADMITTED_COMPLETE}
    other = history[history["ori9"].eq("GA0250300") & history["offense"].eq("larceny")]
    assert set(other["history_status"]) == {ADMITTED_COMPLETE}
    # The build year needs no new row: the target-year admission refuses it too.
    target = apply_level_lane_admission(adjudicated, paths=paths, target_year=2025).panel
    row = target[target["ori9"].eq("GA0250300") & target["year"].eq(2025) & target["offense"].eq("aggravated_assault")].iloc[0]
    assert row["level2_semantic_status"] == "wrong_category_mapping"
    assert not bool(row["usable_as_observed"])


def test_a_lifted_restriction_stops_at_its_lift_year_and_a_year_row_wins(tmp_path, monkeypatch):
    from crimerisk.level_lane import load_hard_evidence

    monkeypatch.delenv("CRIMERISK_LEVEL_RULES", raising=False)
    paths = _configs(
        tmp_path,
        hard_rows=[{
            "case_id": "aa-2023-replacement", "ori9": "GA0250300", "year": 2023, "offense": "aggravated_assault",
            "level1_status": "valid_complete_year", "level2_status": "wrong_category_mapping",
            "reason_code": "category_omitted", "repair_mode": "source_supported_replacement",
            "replacement_count": 650, "external_check_status": "contradiction_confirmed", "evidence_source": "test",
        }],
    )
    _restrictions(tmp_path, [{**CHICAGO_AA, "lifted_from_year": "2025"}])
    assert list(load_hard_evidence(paths, year=2023)["case_id"]) == ["aa-2023-replacement"]
    assert list(load_hard_evidence(paths, year=2024)["case_id"]) == ["aa-04b-omitted"]
    assert load_hard_evidence(paths, year=2025).empty


def test_restrictions_refuse_replacements_and_overlaps(tmp_path):
    from crimerisk.level_lane import LevelLaneConfigError, load_regime_restrictions

    paths = _configs(tmp_path)
    _restrictions(tmp_path, [{**CHICAGO_AA, "repair_mode": "source_supported_replacement"}])
    with pytest.raises(LevelLaneConfigError, match="cannot carry a replacement"):
        load_regime_restrictions(paths, year=2025)
    _restrictions(tmp_path, [CHICAGO_AA, {**CHICAGO_AA, "restriction_id": "vector", "offense": "*", "effective_from_year": "2023"}])
    with pytest.raises(LevelLaneConfigError, match="overlapping"):
        load_regime_restrictions(paths, year=2025)


def test_lifted_and_open_restrictions_load_together_and_may_follow_each_other(tmp_path):
    from crimerisk.level_lane import load_regime_restrictions

    paths = _configs(tmp_path)
    _restrictions(tmp_path, [
        {**CHICAGO_AA, "lifted_from_year": "2023"},
        {**CHICAGO_AA, "restriction_id": "later", "effective_from_year": "2023"},
        {**CHICAGO_AA, "restriction_id": "other-ori", "ori9": "IL0000001"},
    ])
    assert sorted(load_regime_restrictions(paths, year=2024)["case_id"]) == ["later", "other-ori"]


def test_fill_rows_of_an_ori_whose_territory_moved_are_dropped():
    from crimerisk.trend_fills import _drop_moved_territory_fill_estimates

    estimates = pd.DataFrame({
        "ori9": ["AL0640700", "AL0640700", "AL0640000"],
        "offense": ["larceny", "burglary", "larceny"],
        "agency_estimate_source": ["decayed_own_history", "covered_by_covering_agency", "observed"],
        "estimated_count": [3.0, 0.0, 900.0],
    })
    out = _drop_moved_territory_fill_estimates(estimates, covered_oris={"AL0640700"})
    # The covered town keeps no fill; its covered marker and the provider's own report stay.
    assert out["agency_estimate_source"].tolist() == ["covered_by_covering_agency", "observed"]


def _coverage_rows(rows):
    from crimerisk.jurisdiction_targets import _full_year_coverage

    frame = pd.DataFrame(rows)
    frame = frame.assign(jurisdiction_id="01:municipal:place:X", offense="robbery", weight=1.0,
                         preferred_observation_weight=1.0)
    out = _full_year_coverage(frame)
    return out.set_index("year")


def test_full_year_coverage_of_a_year_never_depends_on_later_years():
    # A files 100 every year; B never files before 2025, then files 0, 5, 6 or 100 in 2025.
    base = [{"ori9": "A", "year": y, "preferred_count": 100.0, "preferred_months_reported": 12.0, "usable_as_observed": True}
            for y in (2023, 2024, 2025)]
    shares = []
    for later in (0.0, 5.0, 6.0, 100.0):
        rows = base + [{"ori9": "B", "year": y, "preferred_count": 0.0, "preferred_months_reported": 0.0, "usable_as_observed": False}
                       for y in (2023, 2024)]
        rows += [{"ori9": "B", "year": 2025, "preferred_count": later, "preferred_months_reported": 12.0, "usable_as_observed": True}]
        out = _coverage_rows(rows)
        shares.append((float(out.loc[2024, "full_year_admitted_share"]), int(out.loc[2024, "full_year_silent_agency_count"])))
    # B had filed nothing in or before 2024: silent, so 2024 is A's complete year whatever B does later.
    assert shares == [(1.0, 1)] * 4


def test_unknown_missing_mass_is_not_zero():
    rows = [{"ori9": "A", "year": 2024, "preferred_count": 100.0, "preferred_months_reported": 12.0, "usable_as_observed": True},
            # B filed a partial year in 2023 and nothing in 2024: its 2024 mass is unknown.
            {"ori9": "B", "year": 2023, "preferred_count": 30.0, "preferred_months_reported": 4.0, "usable_as_observed": False},
            {"ori9": "B", "year": 2024, "preferred_count": 0.0, "preferred_months_reported": 0.0, "usable_as_observed": False}]
    out = _coverage_rows(rows)
    assert int(out.loc[2024, "full_year_unknown_agency_count"]) == 1
    assert pd.isna(out.loc[2024, "full_year_admitted_share"])
    # With a complete 2023, B's 2024 mass is estimated from the past (100 + 40 expected; share 100/140).
    rows[1] = {"ori9": "B", "year": 2023, "preferred_count": 40.0, "preferred_months_reported": 12.0, "usable_as_observed": True}
    out = _coverage_rows(rows)
    assert float(out.loc[2024, "full_year_missing_expected_count"]) == 40.0
    assert abs(float(out.loc[2024, "full_year_admitted_share"]) - 100.0 / 140.0) < 1e-12
