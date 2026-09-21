"""Exact rollups: counts summed, rates and indexes recomputed, conservation enforced.

The properties under test are the ones the construction exists to guarantee: a rolled-up rate is
the ratio of the sums and never the average of the ratios; the rolled-up index divides by the same
stored national normalizer the block-group surface published under; counts conserve exactly (with
uncovered mass reported rather than dropped); overlap weights over a split block group sum to one;
publication gates are reapplied at the rollup's own support instead of inherited; and the ZCTA
lane carries the anti-ZIP caveat verbatim.

Built on a synthetic surface so the arithmetic is checkable by hand. The real-surface conservation
run lives in `scripts/release/package_edition.py`, which fails closed on a miss.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crimerisk.composites import (
    COMMON_DENOMINATOR_COLUMN,
    COUNT_FIRST_AGGREGATE_INDEX_FIELDS,
    CompositeRuntime,
    load_severity_weights,
    severity_weights_path,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths
from crimerisk.rollups import (
    CBSA,
    COUNTY,
    DEFAULT_ZERO_RESIDENT_RESIDENT_THRESHOLD,
    FLOOR_COLUMNS,
    PERSONAL_OFFENSES,
    PROPERTY_OFFENSES,
    ROLLUP_GEOGRAPHIES,
    ROLLUP_VERSION,
    STATE,
    ZCTA,
    ZCTA_ANTI_ZIP_CAVEAT,
    assert_conservation,
    conservation_report,
    coverage_universe,
    expected_count_column,
    geography,
    national_rates,
    primary_denominator_column,
    primary_national_rate_column,
    resident_national_rate_column,
    roll_up,
    rollup_filename,
    source_columns,
    zero_resident_resident_threshold,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
RATE_PER_100K = 100000.0

FLOORS = {
    "eb_hard_min_denominator": 1.0,
    "non_residential_household_floor": 10.0,
    "person_exposure_denominator_floor": 50.0,
    "mvt_vehicle_exposure_denominator_floor": 50.0,
    "zero_resident_opportunity_rate_floor": 500.0,
}
DENOMINATOR_TYPE = {
    "murder": "exposure",
    "rape": "exposure",
    "robbery": "exposure",
    "aggravated_assault": "exposure",
    "burglary": "premises",
    "larceny": "exposure",
    "motor_vehicle_theft": "vehicles",
}
NATIONAL_RATE = {offense: 100.0 + 10.0 * index for index, offense in enumerate(OFFENSES_7)}
RESIDENT_NATIONAL_RATE = {offense: 90.0 + 10.0 * index for index, offense in enumerate(OFFENSES_7)}


def _runtime() -> CompositeRuntime:
    path = severity_weights_path(PATHS)
    return CompositeRuntime(weights=load_severity_weights(path), weights_path=path, year=2024)


def _surface(rows: list[dict[str, object]]) -> pd.DataFrame:
    """A synthetic block-group surface carrying exactly the columns a rollup reads."""
    frame = pd.DataFrame(rows)
    frame["block_group_geoid"] = frame["block_group_geoid"].astype("string")
    frame["state_fips"] = frame["block_group_geoid"].str.slice(0, 2)
    for column, value in FLOORS.items():
        frame[column] = float(value)
    for offense in OFFENSES_7:
        frame[f"primary_denominator_type_{offense}"] = DENOMINATOR_TYPE[offense]
        frame[primary_national_rate_column(offense)] = NATIONAL_RATE[offense]
        frame[resident_national_rate_column(offense)] = RESIDENT_NATIONAL_RATE[offense]
    return frame[source_columns()]


def _default_rows() -> list[dict[str, object]]:
    rows = []
    for index, geoid in enumerate(
        ["010010201001", "010010201002", "010030301001", "040010101001"]
    ):
        row: dict[str, object] = {
            "block_group_geoid": geoid,
            "population_2024": 1000 + 100 * index,
            "households_total": 400 + 10 * index,
            "land_area_sq_mi": 2.0 + index,
            COMMON_DENOMINATOR_COLUMN: 1000 + 100 * index,
        }
        for offense_index, offense in enumerate(OFFENSES_7):
            row[expected_count_column(offense)] = 1.0 + index + offense_index
            row[primary_denominator_column(offense)] = 2000.0 + 100 * index + 10 * offense_index
        rows.append(row)
    return rows


def _crosswalk(surface: pd.DataFrame, geo, weights: dict[str, list[tuple[str, float]]] | None = None):
    if weights is None:
        ids = surface["block_group_geoid"]
        if geo.key == COUNTY.key:
            frame = pd.DataFrame(
                {
                    "block_group_geoid": ids,
                    "county_geoid": ids.str.slice(0, 5),
                    "county_name": "Test County",
                    "weight": 1.0,
                }
            )
        else:
            frame = pd.DataFrame(
                {
                    "block_group_geoid": ids,
                    "state_fips": ids.str.slice(0, 2),
                    "state_abbr": "XX",
                    "weight": 1.0,
                }
            )
        return frame
    records = []
    for block_group, parts in weights.items():
        for unit, weight in parts:
            records.append(
                {"block_group_geoid": block_group, geo.id_column: unit, "weight": float(weight)}
            )
    return pd.DataFrame(records)


# --- the geography table -------------------------------------------------------------------


def test_rollup_geographies_are_the_four_product_supports():
    assert [geo.key for geo in ROLLUP_GEOGRAPHIES] == ["county", "cbsa", "zcta", "state"]
    assert geography("county") is COUNTY
    with pytest.raises(KeyError):
        geography("precinct")


def test_only_zcta_is_overlap_weighted_and_only_zcta_carries_a_caveat():
    for geo in ROLLUP_GEOGRAPHIES:
        assert geo.nests_in_block_group == (geo.key != ZCTA.key)
        assert (geo.caveat is not None) == (geo.key == ZCTA.key)
    assert ZCTA.caveat == ZCTA_ANTI_ZIP_CAVEAT


def test_county_and_state_cover_the_universe_cbsa_and_zcta_do_not():
    assert COUNTY.covers_universe and STATE.covers_universe
    assert not CBSA.covers_universe and not ZCTA.covers_universe


def test_zcta_caveat_says_zctas_are_not_zip_codes():
    assert ZCTA_ANTI_ZIP_CAVEAT.startswith("ZCTAs are not ZIP Codes.")
    assert "must not be joined to ZIP-coded records" in ZCTA_ANTI_ZIP_CAVEAT
    assert "Census Bureau" in ZCTA_ANTI_ZIP_CAVEAT


def test_rollup_filename_is_geography_and_year_keyed():
    assert rollup_filename(COUNTY, year=2024) == "crimerisk_county_2024_rollup.parquet"


def test_population_column_tracks_the_annual_surface_year():
    assert "population_2025" in source_columns(year=2025)
    assert "population_2024" not in source_columns(year=2025)

    surface = _surface(_default_rows()).rename(columns={"population_2024": "population_2025"})
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    assert "population_2025" in rollup.columns
    assert "population_2024" not in rollup.columns
    assert float(rollup["population_2025"].sum()) == pytest.approx(
        float(surface["population_2025"].sum())
    )


# --- rates are the ratio of the sums --------------------------------------------------------


def test_rolled_up_rate_is_the_ratio_of_sums_not_the_average_of_ratios():
    surface = _surface(_default_rows())
    crosswalk = _crosswalk(surface, COUNTY)
    rollup = roll_up(surface, geo=COUNTY, crosswalk=crosswalk, composites=_runtime())

    members = surface[surface["block_group_geoid"].str.startswith("01001")]
    for offense in OFFENSES_7:
        counts = float(members[expected_count_column(offense)].sum())
        denominator = float(members[primary_denominator_column(offense)].sum())
        row = rollup[rollup["county_geoid"] == "01001"].iloc[0]
        expected_rate = RATE_PER_100K * counts / denominator
        assert row[f"rate_{offense}_primary"] == pytest.approx(expected_rate)

        average_of_ratios = float(
            (
                RATE_PER_100K
                * members[expected_count_column(offense)]
                / members[primary_denominator_column(offense)]
            ).mean()
        )
        assert row[f"rate_{offense}_primary"] != pytest.approx(average_of_ratios)


def test_rolled_up_index_divides_by_the_stored_national_normalizer():
    surface = _surface(_default_rows())
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    for offense in OFFENSES_7:
        assert (rollup[primary_national_rate_column(offense)] == NATIONAL_RATE[offense]).all()
        recomputed = (
            100.0 * rollup[f"rate_{offense}_primary"] / rollup[primary_national_rate_column(offense)]
        )
        pd.testing.assert_series_equal(
            recomputed, rollup[f"index_{offense}_primary"], check_names=False
        )


def test_resident_lane_recomputes_over_the_common_denominator():
    surface = _surface(_default_rows())
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    for offense in OFFENSES_7:
        expected = (
            RATE_PER_100K
            * rollup[expected_count_column(offense)]
            / rollup[COMMON_DENOMINATOR_COLUMN]
        )
        pd.testing.assert_series_equal(
            expected, rollup[f"rate_{offense}_resident"], check_names=False
        )


def test_aggregate_counts_are_the_sums_of_the_published_offence_counts():
    surface = _surface(_default_rows())
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    for label, offenses in (
        ("personal", PERSONAL_OFFENSES),
        ("property", PROPERTY_OFFENSES),
        ("total", tuple(OFFENSES_7)),
    ):
        expected = sum(rollup[expected_count_column(offense)] for offense in offenses)
        pd.testing.assert_series_equal(
            expected, rollup[expected_count_column(label)], check_names=False
        )


def test_crime_density_is_counts_over_rolled_up_land_area():
    surface = _surface(_default_rows())
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    expected = rollup[expected_count_column("total")] / rollup["land_area_sq_mi"]
    pd.testing.assert_series_equal(expected, rollup["crime_density_total"], check_names=False)


def test_count_first_composites_are_recomputed_at_the_rollup_support():
    surface = _surface(_default_rows())
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    for column in COUNT_FIRST_AGGREGATE_INDEX_FIELDS:
        assert column in rollup.columns
    # The harm burden publishes at tract support and coarser; a county is coarser.
    assert rollup["index_harm_burden_resident"].notna().any()


# --- conservation ------------------------------------------------------------------------------


def test_nesting_geographies_conserve_counts_exactly():
    surface = _surface(_default_rows())
    for geo in (COUNTY, STATE):
        crosswalk = _crosswalk(surface, geo)
        rollup = roll_up(surface, geo=geo, crosswalk=crosswalk, composites=_runtime())
        report = conservation_report(surface, geo=geo, crosswalk=crosswalk, rollup=rollup)
        assert report["conserved"]
        assert report["block_groups_outside_universe"] == 0
        for offense in OFFENSES_7:
            entry = report["offenses"][offense]
            assert entry["difference"] == pytest.approx(0.0, abs=1e-9)
            assert entry["outside_universe_total"] == 0.0
        assert_conservation(report)


def test_uncovered_block_groups_are_reported_not_dropped_silently():
    surface = _surface(_default_rows())
    partial = _crosswalk(surface, COUNTY)
    partial = partial[partial["block_group_geoid"] != "040010101001"]
    rollup = roll_up(surface, geo=COUNTY, crosswalk=partial, composites=_runtime())
    report = conservation_report(surface, geo=COUNTY, crosswalk=partial, rollup=rollup)
    assert report["block_groups_outside_universe"] == 1
    for offense in OFFENSES_7:
        entry = report["offenses"][offense]
        assert entry["outside_universe_total"] > 0.0
        assert entry["rollup_total"] + entry["outside_universe_total"] == pytest.approx(
            entry["block_group_total"]
        )
    assert report["conserved"]


def test_assert_conservation_raises_on_a_lost_count():
    surface = _surface(_default_rows())
    crosswalk = _crosswalk(surface, COUNTY)
    rollup = roll_up(surface, geo=COUNTY, crosswalk=crosswalk, composites=_runtime())
    rollup.loc[0, expected_count_column("robbery")] = 0.0
    report = conservation_report(surface, geo=COUNTY, crosswalk=crosswalk, rollup=rollup)
    assert not report["conserved"]
    with pytest.raises(AssertionError, match="does not conserve"):
        assert_conservation(report)


# --- overlap weighting ----------------------------------------------------------------------


def test_overlap_weighted_split_conserves_and_splits_by_weight():
    surface = _surface(_default_rows())
    weights = {
        "010010201001": [("11111", 0.25), ("22222", 0.75)],
        "010010201002": [("11111", 1.0)],
        "010030301001": [("22222", 1.0)],
        "040010101001": [("33333", 1.0)],
    }
    crosswalk = _crosswalk(surface, ZCTA, weights=weights)
    rollup = roll_up(surface, geo=ZCTA, crosswalk=crosswalk, composites=_runtime())
    report = conservation_report(surface, geo=ZCTA, crosswalk=crosswalk, rollup=rollup)
    assert report["conserved"]

    split_count = float(surface.loc[0, expected_count_column("robbery")])
    whole_count = float(surface.loc[1, expected_count_column("robbery")])
    row = rollup[rollup["zcta5"] == "11111"].iloc[0]
    assert row[expected_count_column("robbery")] == pytest.approx(
        0.25 * split_count + whole_count
    )
    assert row["block_group_weight_sum"] == pytest.approx(1.25)
    assert row["block_group_parts"] == 2


def test_zcta_rollup_carries_the_caveat_on_every_row():
    surface = _surface(_default_rows())
    weights = {geoid: [("99999", 1.0)] for geoid in surface["block_group_geoid"]}
    rollup = roll_up(
        surface, geo=ZCTA, crosswalk=_crosswalk(surface, ZCTA, weights=weights), composites=_runtime()
    )
    assert (rollup["geography_caveat"] == ZCTA_ANTI_ZIP_CAVEAT).all()


# --- publication gates are reapplied, not inherited ------------------------------------------


def test_a_unit_below_the_household_floor_publishes_counts_but_no_rate():
    rows = _default_rows()
    for row in rows[:2]:
        row["households_total"] = 2
    surface = _surface(rows)
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    row = rollup[rollup["county_geoid"] == "01001"].iloc[0]
    assert row[expected_count_column("robbery")] > 0.0
    assert not bool(row["primary_index_publishable_robbery"])
    assert pd.isna(row["rate_robbery_primary"])
    assert pd.isna(row["index_robbery_primary"])


def test_a_unit_below_the_exposure_floor_is_suppressed_for_that_offence_only():
    rows = _default_rows()
    rows[3][primary_denominator_column("robbery")] = 10.0
    surface = _surface(rows)
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    row = rollup[rollup["county_geoid"] == "04001"].iloc[0]
    assert not bool(row["primary_index_publishable_robbery"])
    assert bool(row["primary_index_publishable_larceny"])


def test_near_zero_resident_units_need_the_larger_opportunity_denominator():
    rows = _default_rows()
    rows[3]["population_2024"] = 5
    rows[3][primary_denominator_column("robbery")] = 200.0
    rows[3][primary_denominator_column("larceny")] = 900.0
    surface = _surface(rows)
    rollup = roll_up(
        surface, geo=COUNTY, crosswalk=_crosswalk(surface, COUNTY), composites=_runtime()
    )
    row = rollup[rollup["county_geoid"] == "04001"].iloc[0]
    assert not bool(row["primary_index_publishable_robbery"])
    assert bool(row["primary_index_publishable_larceny"])


def test_rare_offences_publish_at_every_rollup_support():
    surface = _surface(_default_rows())
    for geo in (COUNTY, STATE):
        rollup = roll_up(
            surface, geo=geo, crosswalk=_crosswalk(surface, geo), composites=_runtime()
        )
        for offense in ("murder", "rape"):
            assert rollup[f"index_{offense}_primary"].notna().all()


# --- surface-derived constants ---------------------------------------------------------------


def test_floors_and_national_rates_are_read_off_the_surface():
    surface = _surface(_default_rows())
    rates = national_rates(surface)
    assert rates["primary"] == NATIONAL_RATE
    assert rates["resident"] == RESIDENT_NATIONAL_RATE
    for column in FLOOR_COLUMNS:
        assert column in surface.columns


def test_a_varying_national_rate_is_a_build_error():
    surface = _surface(_default_rows())
    surface.loc[0, primary_national_rate_column("robbery")] = 1.0
    with pytest.raises(ValueError, match="not a single surface-wide value"):
        national_rates(surface)


def test_zero_resident_threshold_comes_from_the_build_manifest():
    assert zero_resident_resident_threshold(None) == DEFAULT_ZERO_RESIDENT_RESIDENT_THRESHOLD
    manifest = {
        "resolved_config": {"zero_resident_opportunity_rate_policy": {"resident_threshold": 75.0}}
    }
    assert zero_resident_resident_threshold(manifest) == 75.0


def test_coverage_universe_is_named_from_the_surface_not_asserted():
    surface = _surface(_default_rows())
    universe = coverage_universe(surface)
    assert "Alaska" in universe["description"] and "Hawaii" in universe["description"]
    assert universe["block_groups"] == len(surface)
    assert universe["counties"] == 3
    assert universe["state_fips"] == ["01", "04"]


def test_rollup_version_is_stamped():
    assert ROLLUP_VERSION == "exact_rollups_v1"
