"""v2 count-first composites: event burden and harm burden from counts under one denominator.

The properties under test are the ones the construction exists to guarantee: the composites are
functions of published counts and the common denominator and of NOTHING else (no per-offense
denominator, no estimate mode, no publishability flag); the harm index is absent at block-group
support; the severity vector is versioned, cited, scale-free and sensitivity-tested; and the
legacy composite lane is byte-identical when the flag is off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import (
    AGGREGATE_INDEX_FIELDS,
    HARM_WEIGHTS,
    PERSONAL_OFFENSES as ALLOCATION_PERSONAL_OFFENSES,
    PROPERTY_OFFENSES as ALLOCATION_PROPERTY_OFFENSES,
    AllocationBuildConfig,
    apply_rare_offense_tract_support,
)
from crimerisk.composites import (
    COMMON_DENOMINATOR_COLUMN,
    COMPOSITE_VERSION,
    COUNT_FIRST_AGGREGATE_INDEX_FIELDS,
    EVENT_BURDEN_COLUMN,
    HARM_BURDEN_COLUMN,
    HARM_WEIGHTED_COUNT_COLUMN,
    INDEX_MAP_BREAKS,
    LEGACY_AGGREGATE_INDEX_FIELDS,
    PERSONAL_BURDEN_COLUMN,
    PERSONAL_OFFENSES,
    PROPERTY_BURDEN_COLUMN,
    PROPERTY_OFFENSES,
    RENAMED_COMPOSITE_FIELDS,
    SUPERSEDED_COMPOSITE_FIELDS,
    CompositeRuntime,
    aggregate_index_fields,
    apply_count_first_composites,
    burden_publishable,
    compare_severity_vectors,
    composite_normalizers,
    count_first_index,
    harm_burden_index,
    load_severity_weights,
    map_bin,
    multi_offense_score_column,
    primary_vector_id,
    severity_sensitivity,
    severity_vector,
    severity_weights_path,
    summarize_count_first_composites,
    vector_ids,
    vector_metadata,
    volume_only_vector,
    weighted_count,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
WEIGHTS_PATH = severity_weights_path(PATHS)
RATE_PER_100K = 100000.0


def _weights() -> pd.DataFrame:
    return load_severity_weights(WEIGHTS_PATH)


def _runtime() -> CompositeRuntime:
    return CompositeRuntime(weights=_weights(), weights_path=WEIGHTS_PATH, year=2024)


# --- the versioned severity vector ---------------------------------------------------------


def test_the_shipped_primary_vector_is_the_vector_the_surface_already_published():
    """Moving the harm weights into configs/ must not move a single number: the legacy harm field
    is reproducible only if the committed primary vector IS `allocation.HARM_WEIGHTS`."""
    weights = _weights()
    assert primary_vector_id(weights) == "cchi_sentencing_days_v1"
    assert severity_vector(weights, "cchi_sentencing_days_v1") == dict(HARM_WEIGHTS)


def test_the_alternative_vector_is_the_cited_published_table():
    """McCollister, French & Fang (2010) Table 5, total cost per offense, 2008 USD. Transcribed
    values, not remembered ones -- a wrong number here is a wrong citation."""
    weights = _weights()
    metadata = vector_metadata(weights, "mccollister_social_cost_2008usd_v1")
    assert metadata["role"] == "alternative"
    assert metadata["unit"] == "usd_2008_per_offense"
    assert "McCollister" in str(metadata["source"])
    assert "doi:10.1016/j.drugalcdep.2009.12.002" in str(metadata["source"])
    assert metadata["weights"] == {
        "murder": 8_982_907.0,
        "rape": 240_776.0,
        "robbery": 42_310.0,
        "aggravated_assault": 107_020.0,
        "burglary": 6_462.0,
        "larceny": 3_532.0,
        "motor_vehicle_theft": 10_772.0,
    }


def test_the_two_published_vectors_disagree_about_the_ordering_of_violence():
    """The sensitivity result is only informative if the alternative is genuinely a different
    normative claim. It is: the CCHI puts robbery above aggravated assault, the cost vector puts
    aggravated assault above robbery."""
    weights = _weights()
    cchi = severity_vector(weights, "cchi_sentencing_days_v1")
    cost = severity_vector(weights, "mccollister_social_cost_2008usd_v1")
    assert cchi["robbery"] > cchi["aggravated_assault"]
    assert cost["robbery"] < cost["aggravated_assault"]


def test_the_table_names_one_primary_and_at_least_one_alternative():
    weights = _weights()
    assert vector_ids(weights, role="primary") == ("cchi_sentencing_days_v1",)
    assert len(vector_ids(weights, role="alternative")) >= 1
    assert vector_ids(weights, role="anchor") == ("uniform_event_v1",)
    for vector_id in vector_ids(weights):
        assert set(severity_vector(weights, vector_id)) == set(OFFENSES_7)


def _write_weights(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "severity_weights_test.csv"
    frame.to_csv(path, index=False)
    return path


def test_absent_severity_table_fails_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="source of record"):
        load_severity_weights(tmp_path / "nope.csv")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.drop(columns=["unit"]), "missing columns"),
        (lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True), "duplicate"),
        (lambda f: f.assign(weight=f["weight"].astype(object).mask(f.index == 0, "x")), "non-numeric"),
        (lambda f: f.assign(weight=f["weight"].mask(f.index == 0, 0.0)), "non-positive"),
        (lambda f: f.assign(weight=f["weight"].mask(f.index == 0, -1.0)), "non-positive"),
        (lambda f: f.assign(role=f["role"].mask(f.index == 0, "headline")), "unknown roles"),
        (lambda f: f.iloc[1:], "exactly the seven Part-I offenses"),
        (
            lambda f: f.assign(offense=f["offense"].mask(f.index == 0, "arson")),
            "exactly the seven Part-I offenses",
        ),
        (
            lambda f: f.assign(role=f["role"].mask(f["vector_id"].eq("uniform_event_v1"), "primary")),
            "exactly one primary vector",
        ),
        (
            lambda f: f.assign(
                source=f["source"].mask(f.index == 0, "somewhere else")
            ),
            "distinct source values",
        ),
    ],
)
def test_a_corrupted_severity_table_fails_the_build_rather_than_reweighting_the_country(
    tmp_path: Path, mutate, message: str
):
    frame = pd.read_csv(WEIGHTS_PATH)
    with pytest.raises(ValueError, match=message):
        load_severity_weights(_write_weights(tmp_path, mutate(frame)))


def test_a_table_with_no_alternative_vector_refuses_to_publish_a_harm_composite(tmp_path: Path):
    frame = pd.read_csv(WEIGHTS_PATH)
    frame = frame[~frame["role"].eq("alternative")]
    with pytest.raises(ValueError, match="no alternative vector"):
        load_severity_weights(_write_weights(tmp_path, frame))


def test_the_offense_groups_mirror_the_allocation_lists():
    assert list(PERSONAL_OFFENSES) == list(ALLOCATION_PERSONAL_OFFENSES)
    assert list(PROPERTY_OFFENSES) == list(ALLOCATION_PROPERTY_OFFENSES)


# --- the formulas, on a hand-computed fixture ----------------------------------------------


def _hand_frame() -> pd.DataFrame:
    """Two publishable cells with hand-computed totals, plus three that must not publish."""
    counts = {
        "murder": [1.0, 0.0, 5.0, 5.0, 5.0],
        "rape": [2.0, 0.0, 5.0, 5.0, 5.0],
        "robbery": [3.0, 0.0, 5.0, 5.0, 5.0],
        "aggravated_assault": [4.0, 0.0, 5.0, 5.0, 5.0],
        "burglary": [5.0, 0.0, 5.0, 5.0, 5.0],
        "larceny": [6.0, 100.0, 5.0, 5.0, 5.0],
        "motor_vehicle_theft": [7.0, 0.0, 5.0, 5.0, 5.0],
    }
    frame = pd.DataFrame(
        {
            COMMON_DENOMINATOR_COLUMN: [1_000.0, 2_000.0, 40.0, 1_000.0, 1_000.0],
            "households_total": [400.0, 500.0, 30.0, 5.0, 400.0],
            "special_use_tract_flag": [False, False, False, False, True],
        }
    )
    for offense, values in counts.items():
        frame[f"expected_count_{offense}"] = values
    return frame


def test_event_burden_is_the_summed_count_rate_indexed_to_its_own_reference_rate():
    frame = _hand_frame()
    published = count_first_index(
        counts=weighted_count(frame, {offense: 1.0 for offense in OFFENSES_7}),
        denominator=frame[COMMON_DENOMINATOR_COLUMN],
        publishable=burden_publishable(frame),
    )
    # 28 events over 1,000 residents and 100 over 2,000; reference 128/3,000.
    assert published["reference_rate_per_100k"] == pytest.approx(RATE_PER_100K * 128.0 / 3_000.0)
    index = pd.Series(published["index"])
    assert index.iloc[0] == pytest.approx(65.625)
    assert index.iloc[1] == pytest.approx(117.1875)
    assert index.iloc[2:].isna().all()


def test_harm_burden_is_the_same_form_with_severity_weights():
    frame = _hand_frame()
    vector = severity_vector(_weights(), "cchi_sentencing_days_v1")
    # 1x5475 + 2x1825 + 3x365 + 4x180 + 5x90 + 6x7 + 7x30
    assert weighted_count(frame, vector).iloc[0] == pytest.approx(11_642.0)
    assert weighted_count(frame, vector).iloc[1] == pytest.approx(700.0)
    published = harm_burden_index(frame, vector=vector)
    reference = RATE_PER_100K * (11_642.0 + 700.0) / 3_000.0
    assert published["reference_rate_per_100k"] == pytest.approx(reference)
    index = pd.Series(published["index"])
    assert index.iloc[0] == pytest.approx(100.0 * (RATE_PER_100K * 11_642.0 / 1_000.0) / reference)
    assert index.iloc[1] == pytest.approx(100.0 * (RATE_PER_100K * 700.0 / 2_000.0) / reference)


def test_the_harm_index_is_invariant_to_any_positive_rescaling_of_the_vector():
    """Which is why a sentencing-days vector and a 2008-dollars vector are comparable at all."""
    frame = _hand_frame()
    vector = severity_vector(_weights(), "cchi_sentencing_days_v1")
    scaled = {offense: 3.7 * weight for offense, weight in vector.items()}
    base = pd.Series(harm_burden_index(frame, vector=vector)["index"])
    rescaled = pd.Series(harm_burden_index(frame, vector=scaled)["index"])
    pd.testing.assert_series_equal(base, rescaled, check_exact=False, rtol=1e-12)


def test_the_uniform_anchor_vector_reproduces_the_event_burden_exactly():
    frame = _hand_frame()
    uniform = severity_vector(_weights(), "uniform_event_v1")
    harm = pd.Series(harm_burden_index(frame, vector=uniform)["index"])
    event = pd.Series(
        count_first_index(
            counts=weighted_count(frame, {offense: 1.0 for offense in OFFENSES_7}),
            denominator=frame[COMMON_DENOMINATOR_COLUMN],
            publishable=burden_publishable(frame),
        )["index"]
    )
    pd.testing.assert_series_equal(harm, event, check_exact=False, rtol=1e-12)


def test_personal_and_property_subtotals_use_the_same_construction():
    frame = _hand_frame()
    applied = apply_count_first_composites(frame, runtime=_runtime(), support="tract")
    personal = weighted_count(frame, {offense: 1.0 for offense in PERSONAL_OFFENSES})
    assert personal.iloc[0] == pytest.approx(10.0)
    expected = pd.Series(
        count_first_index(
            counts=personal,
            denominator=frame[COMMON_DENOMINATOR_COLUMN],
            publishable=burden_publishable(frame),
        )["index"]
    )
    pd.testing.assert_series_equal(
        applied[PERSONAL_BURDEN_COLUMN], expected, check_names=False, rtol=1e-12
    )


def test_the_publication_rule_is_the_common_denominator_and_nothing_else():
    frame = _hand_frame()
    publishable = burden_publishable(frame)
    # below the 50-resident floor / below the household floor / special-use tract
    assert publishable.tolist() == [True, True, False, False, False]


def test_the_burdens_never_read_a_per_offense_opportunity_denominator_or_display_mode():
    """The count-first property, asserted directly: scramble every per-offense OPPORTUNITY
    denominator and display mode on the frame and the composites do not move. This is the
    cross-family rule -- a below-floor vehicle exposure says nothing about resident population."""
    frame = _hand_frame()
    for offense in OFFENSES_7:
        frame[f"primary_denominator_{offense}"] = 1_000.0
        frame[f"estimate_mode_{offense}"] = "published"
        frame[f"index_{offense}_resident_publishable"] = True
    before = apply_count_first_composites(frame, runtime=_runtime(), support="tract")

    scrambled = frame.copy()
    for offense in OFFENSES_7:
        scrambled[f"primary_denominator_{offense}"] = 0.0
        scrambled[f"estimate_mode_{offense}"] = "vehicle_denominator_invalid"
        scrambled[f"primary_index_suppressed_{offense}"] = True
    scrambled["primary_denominator_invalid_motor_vehicle_theft"] = True
    after = apply_count_first_composites(scrambled, runtime=_runtime(), support="tract")

    for field in (EVENT_BURDEN_COLUMN, PERSONAL_BURDEN_COLUMN, PROPERTY_BURDEN_COLUMN, HARM_BURDEN_COLUMN):
        pd.testing.assert_series_equal(before[field], after[field], check_exact=True)
    assert before[EVENT_BURDEN_COLUMN].notna().sum() == 2


def test_a_burden_is_null_where_its_own_component_is_suppressed_on_the_resident_arm():
    """The within-family rule: a composite may not assert a component the surface suppresses on
    the SAME denominator the composite divides by."""
    frame = _hand_frame()
    for offense in OFFENSES_7:
        frame[f"index_{offense}_resident_publishable"] = True
    published = apply_count_first_composites(frame, runtime=_runtime(), support="tract")
    assert published[EVENT_BURDEN_COLUMN].notna().sum() == 2

    suppressed = frame.copy()
    suppressed["index_larceny_resident_publishable"] = False
    out = apply_count_first_composites(suppressed, runtime=_runtime(), support="tract")
    # Larceny is a property offense, so it vetoes the event, property and harm burdens and
    # leaves the personal burden alone: each composite is gated by its OWN component set.
    assert out[EVENT_BURDEN_COLUMN].isna().all()
    assert out[PROPERTY_BURDEN_COLUMN].isna().all()
    assert out[HARM_BURDEN_COLUMN].isna().all()
    pd.testing.assert_series_equal(
        published[PERSONAL_BURDEN_COLUMN], out[PERSONAL_BURDEN_COLUMN], check_exact=True
    )


def test_the_index_average_composites_go_null_when_a_component_point_is_suppressed():
    """`multi_offense_relative_score_*` averages per-offense PRIMARY indexes, so a suppressed
    primary point must null it rather than being replaced by the parent tract's value."""
    from crimerisk.allocation import _finalize_output

    rows = [
        _surface_row(f"01001000{index:04d}", pop=2_000.0, households=700.0, counts=20.0, vehicles=1_600.0)
        for index in range(30)
    ]
    # One cell with real counts and a real resident population whose PERSON EXPOSURE is below the
    # publication floor: its larceny/robbery/assault points are suppressed, its counts are not.
    low_exposure = _surface_row(
        "010010001999", pop=2_000.0, households=700.0, counts=20.0, vehicles=1_600.0, exposure=10.0
    )
    # The published person exposure is the max of the LODES proxy and the selected exposure, so
    # both have to be below the floor for the cell to be the one the real surface produces.
    low_exposure["daytime_population_jobs_proxy"] = 0.0
    rows.append(low_exposure)
    finalized = _finalize_output(
        pd.DataFrame(rows),
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(enable_count_first_composites=True),
        composites=_runtime(),
    ).set_index("block_group_geoid")
    row = finalized.loc["010010001999"]
    assert str(row["estimate_mode_larceny"]) == "insufficient_exposure"
    assert pd.isna(row["index_larceny_primary"])
    for column in RENAMED_COMPOSITE_FIELDS.values():
        assert pd.isna(row[column]), column
    # The cells whose own points are all publishable are untouched.
    healthy = finalized.loc["010010000000"]
    for column in RENAMED_COMPOSITE_FIELDS.values():
        assert pd.notna(healthy[column]), column


# --- support policy ------------------------------------------------------------------------


def test_the_harm_index_is_absent_at_block_group_and_present_at_tract():
    frame = _hand_frame()
    bg = apply_count_first_composites(frame, runtime=_runtime(), support="block_group")
    tract = apply_count_first_composites(frame, runtime=_runtime(), support="tract")
    assert bg[HARM_BURDEN_COLUMN].isna().all()
    assert tract[HARM_BURDEN_COLUMN].notna().sum() == 2
    # The count is a linear combination of counts this surface already publishes, so it carries
    # at both supports -- it is the index, not the arithmetic, that the rule is about.
    pd.testing.assert_series_equal(
        bg[HARM_WEIGHTED_COUNT_COLUMN], tract[HARM_WEIGHTED_COUNT_COLUMN], check_exact=True
    )
    # Every other burden is unaffected by support.
    for field in (EVENT_BURDEN_COLUMN, PERSONAL_BURDEN_COLUMN, PROPERTY_BURDEN_COLUMN):
        pd.testing.assert_series_equal(bg[field], tract[field], check_exact=True)


# --- map bins and the severity sensitivity matrix -------------------------------------------


def test_map_bin_ordinals_are_half_open_intervals_over_the_published_breaks():
    values = pd.Series([0.0, 12.4, 12.5, 99.9, 100.0, 799.0, 800.0, 5_000.0, np.nan])
    assert map_bin(values).tolist()[:-1] == [0.0, 2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 13.0]
    assert pd.isna(map_bin(values).iloc[-1])


def test_comparison_metrics_are_exact_on_a_constructed_pair():
    base = pd.Series(float(value) for value in range(10, 110, 10))  # 10 rows, strictly increasing
    identical = base.copy()
    metrics = compare_severity_vectors(base, identical)
    assert metrics["rows_compared"] == 10
    assert metrics["spearman_rho"] == pytest.approx(1.0)
    assert metrics["top_decile_overlap"] == pytest.approx(1.0)
    assert metrics["share_moving_1plus_bins"] == pytest.approx(0.0)
    assert metrics["share_moving_2plus_bins"] == pytest.approx(0.0)

    reversed_metrics = compare_severity_vectors(base, base.iloc[::-1].reset_index(drop=True))
    assert reversed_metrics["spearman_rho"] == pytest.approx(-1.0)
    assert reversed_metrics["top_decile_overlap"] == pytest.approx(0.0)

    # 10 rows, all at bin 4 ([75, 100)); the alternative puts two of them three bins up.
    left = pd.Series([80.0] * 10)
    right = pd.Series([80.0] * 8 + [500.0, 900.0])
    shifted = compare_severity_vectors(left, right)
    assert shifted["share_moving_2plus_bins"] == pytest.approx(0.2)
    assert shifted["max_abs_bin_shift"] == pytest.approx(5.0)


def test_the_sensitivity_matrix_prices_every_alternative_and_the_rare_offense_dependence():
    frame = _hand_frame()
    result = severity_sensitivity(frame, weights=_weights(), label="tract_test", support="tract")
    comparisons = set(result["comparison_vector_id"])
    assert "mccollister_social_cost_2008usd_v1" in comparisons
    assert "uniform_event_v1" in comparisons
    assert "cchi_sentencing_days_v1_volume_only" in comparisons
    assert set(result["comparison_role"]) == {"alternative", "anchor", "diagnostic"}
    assert (result["published_rows"] == 2).all()
    assert (result["primary_vector_id"] == "cchi_sentencing_days_v1").all()
    # The fixture is 5 murders/rapes against 100 larcenies in the other cell, so the rare share
    # is large; the point of the column is that it is measured and published, not that it is small.
    assert (result["primary_rare_offense_harm_mass_share"] > 0.0).all()
    assert result["primary_rare_offense_harm_mass_share"].nunique() == 1


def test_the_volume_only_diagnostic_vector_zeroes_exactly_the_rare_offenses():
    vector = severity_vector(_weights(), "cchi_sentencing_days_v1")
    volume_only = volume_only_vector(vector)
    assert volume_only["murder"] == 0.0
    assert volume_only["rape"] == 0.0
    for offense in ("robbery", "aggravated_assault", "burglary", "larceny", "motor_vehicle_theft"):
        assert volume_only[offense] == vector[offense]


# --- the finalized surface, both lanes ------------------------------------------------------


def _surface_row(
    bg: str,
    *,
    pop: float,
    households: float,
    counts: float,
    vehicles: float,
    exposure: float | None = None,
) -> dict:
    row = {
        "block_group_geoid": bg,
        "tract_id": bg[:11],
        "state_fips": bg[:2],
        "population_2024": pop,
        "households_total": households,
        "commercial_premises_total": 0.0,
        "destination_poi_total": 0.0,
        "daytime_population_jobs_proxy": pop,
        "landscan_day_pop": 0.0,
        "exposure_proxy_2024": pop * 1.5 if exposure is None else exposure,
        "burglary_premises_total": max(households, 1.0),
        "aggregate_vehicles_total": vehicles,
        "vehicle_exposure_2024": vehicles,
        "land_area_sq_mi": 1.0,
        "eb_jurisdiction_id": "J1",
        "eb_jurisdiction_type": "municipal",
    }
    for offense in OFFENSES_7:
        row[f"expected_count_{offense}"] = counts
        row[f"footprint_derived_count_{offense}"] = 0.0
    return row


def _finalized(*, count_first: bool, low_population_row: bool = False) -> pd.DataFrame:
    from crimerisk.allocation import _finalize_output

    rows = [
        _surface_row(f"01001000{index:04d}", pop=2_000.0, households=700.0, counts=20.0, vehicles=1_600.0)
        for index in range(30)
    ]
    if low_population_row:
        # Both denominators below their floors, which is the real-surface case: the per-offense
        # points are suppressed with estimate_mode = insufficient_exposure, the legacy aggregate's
        # exemption for that mode then lets a 40-resident cell publish a resident-denominator
        # aggregate, and the count-first rule does not, because 40 residents is exactly the
        # denominator it would be dividing by.
        rows.append(
            _surface_row(
                "010010001999", pop=40.0, households=25.0, counts=1.0, vehicles=30.0, exposure=40.0
            )
        )
    return _finalize_output(
        pd.DataFrame(rows),
        geo_id_col="block_group_geoid",
        population_col="population_2024",
        config=AllocationBuildConfig(enable_count_first_composites=count_first),
        composites=_runtime() if count_first else None,
    ).set_index("block_group_geoid")


def _finalized_tract(*, count_first: bool) -> pd.DataFrame:
    from crimerisk.allocation import _finalize_output

    rows = [
        _surface_row(f"0100100{index:05d}", pop=4_000.0, households=1_400.0, counts=40.0, vehicles=3_200.0)
        for index in range(20)
    ]
    frame = pd.DataFrame(rows).rename(columns={"block_group_geoid": "drop_me"})
    frame["tract_id"] = [f"0100100{index:04d}" for index in range(20)]
    frame = frame.drop(columns=["drop_me"])
    frame["dominant_eb_jurisdiction_id"] = "J1"
    return _finalize_output(
        frame,
        geo_id_col="tract_id",
        population_col="population_2024",
        config=AllocationBuildConfig(enable_count_first_composites=count_first),
        jurisdiction_col="dominant_eb_jurisdiction_id",
        composites=_runtime() if count_first else None,
    ).set_index("tract_id")


def test_default_allocation_config_leaves_the_composite_lane_off():
    config = AllocationBuildConfig()
    assert config.enable_count_first_composites is False
    assert config.severity_weights_path is None


def test_the_lane_fails_closed_when_the_flag_is_set_without_a_resolved_vector():
    from crimerisk.allocation import _finalize_output

    with pytest.raises(ValueError, match="fails closed"):
        _finalize_output(
            pd.DataFrame([_surface_row("010010001001", pop=2_000.0, households=700.0, counts=20.0, vehicles=1_600.0)]),
            geo_id_col="block_group_geoid",
            population_col="population_2024",
            config=AllocationBuildConfig(enable_count_first_composites=True),
        )


def test_the_legacy_composite_lane_is_byte_identical_when_the_flag_is_off():
    """Every column of the flag-off surface, composites included, must be what it was before this
    lane existed. The composite fields are recomputed here from a verbatim transcription of the
    published contract rather than trusted."""
    legacy = _finalized(count_first=False)
    assert list(aggregate_index_fields(count_first=False)) == list(AGGREGATE_INDEX_FIELDS)
    for field in AGGREGATE_INDEX_FIELDS:
        assert field in legacy.columns
    for field in COUNT_FIRST_AGGREGATE_INDEX_FIELDS:
        assert field not in legacy.columns
    assert HARM_WEIGHTED_COUNT_COLUMN not in legacy.columns

    # index_total_harm, transcribed: harm-weighted counts over person exposure, one normalization.
    denominator = pd.to_numeric(legacy["exposure_proxy_2024"], errors="coerce").fillna(0.0).clip(lower=0.0)
    publishable = (
        pd.to_numeric(legacy["households_total"], errors="coerce").fillna(0.0).ge(10.0)
        & denominator.gt(0.0)
        & ~legacy["special_use_tract_flag"].fillna(False).astype(bool)
        & ~denominator.lt(50.0)
    )
    counts = sum(
        float(HARM_WEIGHTS[offense])
        * pd.to_numeric(legacy[f"expected_count_{offense}"], errors="coerce").fillna(0.0).clip(lower=0.0)
        for offense in OFFENSES_7
    )
    reference = (
        RATE_PER_100K * float(counts[publishable].sum()) / float(denominator[publishable].sum())
    )
    expected = pd.Series(np.nan, index=legacy.index, dtype=float)
    expected.loc[publishable] = (
        100.0 * (RATE_PER_100K * counts[publishable] / denominator[publishable]) / reference
    )
    pd.testing.assert_series_equal(
        legacy["index_total_harm"], expected, check_names=False, rtol=1e-12
    )


def test_the_count_first_lane_swaps_the_composite_set_and_changes_nothing_else():
    legacy = _finalized(count_first=False)
    enabled = _finalized(count_first=True)
    assert set(enabled.columns) - set(legacy.columns) == {
        HARM_WEIGHTED_COUNT_COLUMN,
        *COUNT_FIRST_AGGREGATE_INDEX_FIELDS,
    }
    assert set(legacy.columns) - set(enabled.columns) == set(LEGACY_AGGREGATE_INDEX_FIELDS)
    shared = [column for column in legacy.columns if column in enabled.columns]
    for column in shared:
        pd.testing.assert_series_equal(legacy[column], enabled[column], check_exact=True)


def test_the_index_average_composites_are_renamed_and_not_recomputed():
    legacy = _finalized(count_first=False)
    enabled = _finalized(count_first=True)
    for legacy_field, new_field in RENAMED_COMPOSITE_FIELDS.items():
        assert multi_offense_score_column(legacy_field, count_first=True) == new_field
        assert multi_offense_score_column(legacy_field, count_first=False) == legacy_field
        pd.testing.assert_series_equal(
            legacy[legacy_field], enabled[new_field], check_names=False, check_exact=True
        )


def test_the_event_burden_is_the_number_the_legacy_total_already_published():
    """Where both lanes publish, the count-first event burden IS `index_total_part1_resident`.
    The v2 field is not a new estimate; it is the same estimate under a coherent publication rule
    and a name that says what the denominator is."""
    legacy = _finalized(count_first=False)
    enabled = _finalized(count_first=True)
    pd.testing.assert_series_equal(
        legacy["index_total_part1_resident"],
        enabled[EVENT_BURDEN_COLUMN],
        check_names=False,
        rtol=1e-9,
    )
    assert SUPERSEDED_COMPOSITE_FIELDS["index_total_part1_resident"] == EVENT_BURDEN_COLUMN


def test_both_resident_composite_lanes_enforce_the_resident_denominator_floor():
    """Neither resident-denominator composite publishes below the resident floor."""
    legacy = _finalized(count_first=False, low_population_row=True)
    enabled = _finalized(count_first=True, low_population_row=True)
    low = "010010001999"
    assert pd.isna(legacy.loc[low, "index_total_part1_resident"])
    assert pd.isna(enabled.loc[low, EVENT_BURDEN_COLUMN])
    assert pd.isna(enabled.loc[low, HARM_BURDEN_COLUMN])
    assert int(legacy["index_total_part1_resident"].notna().sum()) == len(legacy) - 1
    assert int(enabled[EVENT_BURDEN_COLUMN].notna().sum()) == len(enabled) - 1


def test_the_harm_index_publishes_on_the_finalized_tract_surface_only():
    bg = _finalized(count_first=True)
    tract = _finalized_tract(count_first=True)
    assert bg[HARM_BURDEN_COLUMN].isna().all()
    assert bg[HARM_WEIGHTED_COUNT_COLUMN].gt(0.0).all()
    assert tract[HARM_BURDEN_COLUMN].notna().all()


def test_the_burdens_survive_the_rare_offense_tract_support_policy_untouched():
    """The fold-then-compose ordering question, answered structurally: nothing in the rare-offense
    publication policy touches a count or the resident denominator, so no override machinery is
    needed for a composite that reads only those."""
    surface = _finalized(count_first=True).reset_index()
    tract_cols = ["tract_id", *[f"index_{offense}_primary" for offense in ("murder", "rape")]]
    tract = surface[tract_cols].drop_duplicates("tract_id").copy()
    out = apply_rare_offense_tract_support(surface, tract, count_first_composites=True)
    for field in (
        EVENT_BURDEN_COLUMN,
        PERSONAL_BURDEN_COLUMN,
        PROPERTY_BURDEN_COLUMN,
        HARM_BURDEN_COLUMN,
        HARM_WEIGHTED_COUNT_COLUMN,
    ):
        pd.testing.assert_series_equal(surface[field], out[field], check_exact=True)
    # The murder/rape points are still withdrawn at block group, and the renamed relative scores
    # still take their rare terms at tract support.
    assert out["index_murder_primary"].isna().all()
    assert out["recommended_display_geography_murder"].eq("tract").all()
    assert out["recommended_display_geography_rape"].eq("tract").all()
    assert RENAMED_COMPOSITE_FIELDS["index_total_equal_offense"] in out.columns
    assert "index_total_equal_offense" not in out.columns


def test_rare_offense_native_tract_support_is_not_relabelled_as_county_support():
    tract = _finalized_tract(count_first=True)
    assert tract["recommended_display_geography_murder"].eq("tract").all()
    assert tract["recommended_display_geography_rape"].eq("tract").all()


def test_the_manifest_records_the_vector_the_paths_and_the_superseded_fields():
    summary = summarize_count_first_composites(runtime=_runtime())
    assert summary["enabled"] is True
    assert summary["version"] == COMPOSITE_VERSION
    assert summary["harm_published_support"] == "tract"
    assert summary["primary_severity_vector"]["vector_id"] == "cchi_sentencing_days_v1"
    assert [entry["vector_id"] for entry in summary["alternative_severity_vectors"]]
    assert summary["superseded_fields"]["index_total_harm"] == HARM_BURDEN_COLUMN
    assert summary["ags_values_used"] is False
    assert summary["map_breaks"] == list(INDEX_MAP_BREAKS)

    normalizers = composite_normalizers(_hand_frame(), runtime=_runtime(), support="tract")
    harm = normalizers["fields"][HARM_BURDEN_COLUMN]
    assert harm["severity_vector_id"] == "cchi_sentencing_days_v1"
    assert harm["published_support"] == "tract"
    assert normalizers["fields"][EVENT_BURDEN_COLUMN]["expected_count_total"] == pytest.approx(128.0)


# --- the release validator's mirror ---------------------------------------------------------


def test_the_release_validator_mirror_reproduces_the_published_composites():
    """The mirror is independent code with its own copies of the floors; it must land on the same
    numbers from published fields alone, at both geographies."""
    from scripts.diagnostics.validate_release_outputs import (
        _count_first_expected,
        _count_first_publishable,
        _weighted_count_expected,
    )

    for surface in (_finalized(count_first=True), _finalized_tract(count_first=True)):
        frame = surface.reset_index()
        publishable = _count_first_publishable(frame)
        pd.testing.assert_series_equal(
            publishable, burden_publishable(frame), check_names=False, check_exact=True
        )
        expected, _reference = _count_first_expected(
            frame, weights={offense: 1.0 for offense in OFFENSES_7}, publishable=publishable
        )
        pd.testing.assert_series_equal(
            frame[EVENT_BURDEN_COLUMN], expected, check_names=False, rtol=1e-12
        )
        vector = severity_vector(_weights(), "cchi_sentencing_days_v1")
        pd.testing.assert_series_equal(
            frame[HARM_WEIGHTED_COUNT_COLUMN],
            _weighted_count_expected(frame, vector),
            check_names=False,
            rtol=1e-12,
        )


def test_the_release_validator_knows_both_composite_lanes():
    from scripts.diagnostics.validate_release_outputs import (
        _count_first_composite_lane,
        _expected_columns,
    )

    assert _count_first_composite_lane(_finalized(count_first=True).reset_index()) is True
    assert _count_first_composite_lane(_finalized(count_first=False).reset_index()) is False
    legacy_columns = set(_expected_columns(geography="tract", count_first_composites=False))
    v2_columns = set(_expected_columns(geography="tract", count_first_composites=True))
    assert set(LEGACY_AGGREGATE_INDEX_FIELDS) <= legacy_columns
    assert set(COUNT_FIRST_AGGREGATE_INDEX_FIELDS) | {HARM_WEIGHTED_COUNT_COLUMN} <= v2_columns
    assert not (set(LEGACY_AGGREGATE_INDEX_FIELDS) & v2_columns)
