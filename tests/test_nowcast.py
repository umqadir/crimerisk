"""Surface 3 — the provisional nowcast: the factor, the damping, the overlay, and the edition.

The properties under test are the ones a reader of a provisional number is entitled to rely on:

* the factor divides the SAME source by itself, over the SAME months, so it measures crime change
  rather than the difference between two data-production processes or between two seasons;
* the coverage damping is real -- a poorly observed state visibly does not move as far, and a
  state at zero coverage does not move at all;
* the block-group artifact is a SCALAR overlay: it cannot re-rank anything inside a state, which
  is the honest limit of a state-level input and is asserted here rather than promised in prose;
* the published factor reproduces the published numbers at both supports, and the gate that
  checks it can fail;
* the edition type builds, is labelled provisional in every table name, carries the source's own
  vintage, and refuses to be built by the annual builder.

The end-to-end test runs the real builder script against a synthetic three-state world, so a
passing suite means the script emits what these tests describe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.crime import OFFENSES_7
from crimerisk.editions import (
    ANNUAL,
    EDITION_TYPES,
    QUARTERLY_PROVISIONAL,
    EditionTypeMismatch,
    parse_edition_id,
    require_buildable,
    require_edition_type,
)
from crimerisk.nowcast import (
    CDE_DIAGNOSTIC_SLUGS,
    CDE_SUMMARIZED_SLUG_BY_OFFENSE,
    NO_BASE_MASS_BASIS,
    PROVISIONAL_LABEL,
    NowcastConfig,
    annual_expected_count_column,
    apply_factors_to_block_groups,
    apply_factors_to_controls,
    block_group_factor_column,
    block_group_overlay_column,
    cde_reference_ratios,
    cde_vintage,
    conservation_report,
    damping_report,
    diagnostic_rollup_factors,
    factor_distribution,
    national_comparability,
    provisional_factors,
    read_cde_state_monthly,
    state_provisional_totals,
)
from crimerisk.paths import RepoPaths

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = RepoPaths.from_repo_root(REPO_ROOT)
# --- synthetic source -----------------------------------------------------------------------


def make_panel(
    cells: dict[tuple[str, str, int], tuple[list[float], float]],
    *,
    us_rate: dict[tuple[str, int], list[float]] | None = None,
    max_data_date: str = "06/2026",
    last_refresh_date: str = "06/15/2026",
) -> pd.DataFrame:
    """A CDE-shaped monthly panel.

    `cells` maps (state, slug, year) -> (monthly counts, pct_population_coverage). Months are
    1..len(counts), so a short list is a partial year.
    """
    records = []
    for (state, slug, year), (counts, coverage) in cells.items():
        for index, count in enumerate(counts, start=1):
            rates = (us_rate or {}).get((slug, year))
            records.append(
                {
                    "state": state,
                    "offense_slug": slug,
                    "month": index,
                    "year": year,
                    "count": float(count),
                    "pct_population_coverage": float(coverage),
                    "us_rate": float(rates[index - 1]) if rates else float(count) / 100.0,
                    "max_data_date": max_data_date,
                    "last_refresh_date": last_refresh_date,
                }
            )
    return pd.DataFrame.from_records(records)


def flat(total: float, months: int = 12) -> list[float]:
    return [total / months] * months


# --- the offence mapping --------------------------------------------------------------------


def test_every_part_one_offence_has_its_own_slug_and_murder_is_homicide():
    assert set(CDE_SUMMARIZED_SLUG_BY_OFFENSE) == set(OFFENSES_7)
    assert CDE_SUMMARIZED_SLUG_BY_OFFENSE["murder"] == "homicide"
    assert len(set(CDE_SUMMARIZED_SLUG_BY_OFFENSE.values())) == len(OFFENSES_7)


def test_rollup_slugs_are_never_an_offence_factor():
    """A CDE rollup covers a different offence set; borrowing its factor would import that."""
    assert set(CDE_DIAGNOSTIC_SLUGS).isdisjoint(CDE_SUMMARIZED_SLUG_BY_OFFENSE.values())
    panel = make_panel(
        {
            ("AL", "homicide", 2024): (flat(120), 100.0),
            ("AL", "homicide", 2025): (flat(60), 100.0),
            ("AL", "violent-crime", 2024): (flat(1200), 100.0),
            ("AL", "violent-crime", 2025): (flat(2400), 100.0),
        }
    )
    factors = provisional_factors(panel)
    assert set(factors["cde_offense_slug"]) == {"homicide"}
    # The violent-crime slug doubled; murder halved. The murder factor follows murder.
    assert factors.loc[factors["offense"] == "murder", "provisional_factor"].iloc[0] == pytest.approx(0.6)

    diagnostics = diagnostic_rollup_factors(panel)
    assert diagnostics["applied"].eq(False).all()
    assert diagnostics["factor_raw"].iloc[0] == pytest.approx(2.0)


# --- the factor -----------------------------------------------------------------------------


def test_factor_is_target_over_base_from_the_same_source():
    panel = make_panel(
        {
            ("AL", "burglary", 2024): (flat(1000), 100.0),
            ("AL", "burglary", 2025): (flat(900), 100.0),
        }
    )
    factors = provisional_factors(panel)
    row = factors.iloc[0]
    assert row["base_count"] == pytest.approx(1000.0)
    assert row["target_count"] == pytest.approx(900.0)
    assert row["factor_raw"] == pytest.approx(0.9)
    assert row["provisional_factor"] == pytest.approx(0.9)
    assert row["months_used"] == 12
    assert row["month_span"] == "01-12"


def test_a_partial_target_year_is_compared_against_the_same_months_of_the_base_year():
    """Seasonality, not change, is what a full-year denominator would measure."""
    winter = [100.0, 100.0, 100.0]
    summer = [300.0] * 9
    panel = make_panel(
        {
            ("AL", "larceny", 2024): (winter + summer, 100.0),
            ("AL", "larceny", 2025): (winter, 100.0),
        }
    )
    factors = provisional_factors(panel)
    row = factors.iloc[0]
    assert row["months_used"] == 3
    assert row["month_span"] == "01-03"
    assert row["base_count"] == pytest.approx(300.0)
    assert row["factor_raw"] == pytest.approx(1.0)
    # Against the whole of 2024 the ratio would have been 300/3000 = 0.1, i.e. a 90% "drop"
    # that is nothing but the missing summer.
    assert row["provisional_factor"] == pytest.approx(1.0)


def test_no_base_mass_means_no_change_rather_than_an_invented_ratio():
    panel = make_panel(
        {
            ("WY", "homicide", 2024): (flat(0), 100.0),
            ("WY", "homicide", 2025): (flat(12), 100.0),
        }
    )
    row = provisional_factors(panel).iloc[0]
    assert row["factor_basis"] == NO_BASE_MASS_BASIS
    assert row["provisional_factor"] == pytest.approx(1.0)


# --- the damping ----------------------------------------------------------------------------


def test_low_coverage_states_shrink_visibly_toward_one():
    """The spec's headline requirement: a poorly observed state does not move as far."""
    panel = make_panel(
        {
            ("AL", "robbery", 2024): (flat(1000), 100.0),
            ("AL", "robbery", 2025): (flat(800), 100.0),
            ("MS", "robbery", 2024): (flat(1000), 50.0),
            ("MS", "robbery", 2025): (flat(800), 50.0),
            ("LA", "robbery", 2024): (flat(1000), 0.0),
            ("LA", "robbery", 2025): (flat(800), 0.0),
        }
    )
    factors = provisional_factors(panel).set_index("state_abbr")
    assert factors["factor_raw"].tolist() == pytest.approx([0.8, 0.8, 0.8])

    # Identical measured change, three different credibilities.
    assert factors.loc["AL", "provisional_factor"] == pytest.approx(0.8)
    assert factors.loc["MS", "provisional_factor"] == pytest.approx(0.9)
    assert factors.loc["LA", "provisional_factor"] == pytest.approx(1.0)

    # "Visibly": a sub-70%-coverage state retains less than 70% of the measured deviation.
    config = NowcastConfig()
    low = factors[factors["coverage_pct"] < config.low_coverage_pct]
    assert set(low.index) == {"MS", "LA"}
    retained = (low["provisional_factor"] - 1.0).abs() / (low["factor_raw"] - 1.0).abs()
    assert (retained < config.low_coverage_pct / 100.0).all()

    report = damping_report(provisional_factors(panel))
    assert report["low_coverage_rows"] == 2
    assert report["low_coverage_states"] == ["LA", "MS"]
    assert report["low_coverage_max_shrink_ratio"] == pytest.approx(0.5)
    assert report["weight_equals_shrink_where_unclipped"] is True


def test_the_clip_bounds_a_one_year_change_and_is_flagged():
    panel = make_panel(
        {
            ("AL", "homicide", 2024): (flat(1000), 100.0),
            ("AL", "homicide", 2025): (flat(100), 100.0),
            ("MS", "homicide", 2024): (flat(100), 100.0),
            ("MS", "homicide", 2025): (flat(1000), 100.0),
        }
    )
    factors = provisional_factors(panel).set_index("state_abbr")
    assert factors.loc["AL", "provisional_factor"] == pytest.approx(0.6)
    assert factors.loc["MS", "provisional_factor"] == pytest.approx(1.5)
    assert factors["clipped"].all()

    distribution = factor_distribution(factors.reset_index())
    assert distribution[0]["offense"] == "murder"
    assert distribution[0]["clipped"] == 2
    assert distribution[0]["factor_min"] == pytest.approx(0.6)
    assert distribution[0]["factor_max"] == pytest.approx(1.5)


# --- application ----------------------------------------------------------------------------


def _controls(states: dict[str, dict[str, float]], *, year: int = 2024) -> pd.DataFrame:
    records = []
    for state, offenses in states.items():
        for index, (offense, count) in enumerate(offenses.items()):
            records.append(
                {
                    "jurisdiction_id": f"{state}:municipal:place:{index}",
                    "jurisdiction_type": "municipal",
                    "jurisdiction_name": f"{state} city {index}",
                    "state_fips": {"AL": "01", "MS": "28", "WY": "56"}[state],
                    "state_abbr": state,
                    "geo_type": "place",
                    "geoid": f"{index:07d}",
                    "offense": offense,
                    "year": year,
                    "bucket_population": 10_000.0,
                    "smoothed_count": float(count),
                }
            )
    return pd.DataFrame.from_records(records)


def test_controls_are_multiplied_by_their_state_offence_factor():
    panel = make_panel(
        {
            ("AL", "burglary", 2024): (flat(1000), 100.0),
            ("AL", "burglary", 2025): (flat(500), 100.0),
        }
    )
    factors = provisional_factors(panel)
    controls = _controls({"AL": {"burglary": 40.0}})
    provisional = apply_factors_to_controls(controls, factors)
    assert provisional["annual_smoothed_count"].iloc[0] == pytest.approx(40.0)
    assert provisional["provisional_factor"].iloc[0] == pytest.approx(0.6)  # clipped from 0.5
    assert provisional["provisional_count"].iloc[0] == pytest.approx(24.0)
    assert provisional["is_provisional"].all()


def test_a_control_with_no_factor_is_refused_rather_than_published_unscaled():
    panel = make_panel(
        {
            ("AL", "burglary", 2024): (flat(1000), 100.0),
            ("AL", "burglary", 2025): (flat(900), 100.0),
        }
    )
    factors = provisional_factors(panel)
    controls = _controls({"AL": {"burglary": 40.0}, "WY": {"burglary": 5.0}})
    with pytest.raises(ValueError, match="no provisional factor"):
        apply_factors_to_controls(controls, factors)


def _surface(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """(block_group_geoid, state_fips, murder, larceny) -> a minimal annual surface."""
    frame = pd.DataFrame(
        rows, columns=["block_group_geoid", "state_fips", "murder", "larceny"]
    )
    return frame.rename(
        columns={
            "murder": annual_expected_count_column("murder"),
            "larceny": annual_expected_count_column("larceny"),
        }
    )


OVERLAY_OFFENSES = ("murder", "larceny")


def _two_state_factors() -> pd.DataFrame:
    panel = make_panel(
        {
            ("AL", "homicide", 2024): (flat(1000), 100.0),
            ("AL", "homicide", 2025): (flat(900), 100.0),
            ("AL", "larceny", 2024): (flat(1000), 100.0),
            ("AL", "larceny", 2025): (flat(1100), 100.0),
            ("MS", "homicide", 2024): (flat(1000), 100.0),
            ("MS", "homicide", 2025): (flat(1200), 100.0),
            ("MS", "larceny", 2024): (flat(1000), 100.0),
            ("MS", "larceny", 2025): (flat(800), 100.0),
        }
    )
    return provisional_factors(panel)


def test_the_block_group_overlay_is_a_scalar_and_re_ranks_nothing():
    factors = _two_state_factors()
    surface = _surface(
        [
            ("010010001001", "01", 3.0, 100.0),
            ("010010001002", "01", 1.0, 400.0),
            ("010010001003", "01", 9.0, 250.0),
            ("280010001001", "28", 2.0, 50.0),
            ("280010001002", "28", 5.0, 90.0),
        ]
    )
    overlay = apply_factors_to_block_groups(
        surface,
        factors,
        state_abbr_by_fips={"01": "AL", "28": "MS"},
        offenses=OVERLAY_OFFENSES,
    )
    for offense in OVERLAY_OFFENSES:
        annual = surface[annual_expected_count_column(offense)]
        moved = overlay[block_group_overlay_column(offense)]
        # Within a state every block group is multiplied by the same number ...
        for state in ("01", "28"):
            mask = overlay["state_fips"] == state
            ratio = (moved[mask] / annual[mask]).round(12).unique()
            assert len(ratio) == 1
        # ... so the within-state ordering is identical to the annual surface's.
        for state in ("01", "28"):
            mask = (overlay["state_fips"] == state).to_numpy()
            assert list(np.argsort(annual[mask].to_numpy())) == list(
                np.argsort(moved[mask].to_numpy())
            )
    assert overlay["is_provisional"].all()
    assert overlay[block_group_factor_column("murder")].nunique() == 2


def test_a_block_group_in_a_state_with_no_factor_is_refused():
    factors = _two_state_factors()
    surface = _surface([("560010001001", "56", 1.0, 1.0)])
    with pytest.raises(ValueError, match="no provisional factor"):
        apply_factors_to_block_groups(
            surface,
            factors,
            state_abbr_by_fips={"56": "WY"},
            offenses=OVERLAY_OFFENSES,
        )


# --- conservation ---------------------------------------------------------------------------


def _overlay_and_state_tables():
    factors = _two_state_factors()
    controls = _controls(
        {"AL": {"murder": 10.0, "larceny": 500.0}, "MS": {"murder": 20.0, "larceny": 300.0}}
    )
    state_table = state_provisional_totals(apply_factors_to_controls(controls, factors))
    surface = _surface(
        [
            ("010010001001", "01", 3.0, 100.0),
            ("010010001002", "01", 1.0, 400.0),
            ("280010001001", "28", 2.0, 50.0),
        ]
    )
    overlay = apply_factors_to_block_groups(
        surface, factors, state_abbr_by_fips={"01": "AL", "28": "MS"}, offenses=OVERLAY_OFFENSES
    )
    return overlay, state_table


def test_conservation_is_exact_and_the_footprint_ratio_is_untouched():
    overlay, state_table = _overlay_and_state_tables()
    report = conservation_report(
        block_group_overlay=overlay, state_provisional=state_table, offenses=OVERLAY_OFFENSES
    )
    assert report["conserved"] is True
    assert report["max_block_group_factor_error"] < 1e-12
    assert report["max_jurisdiction_factor_error"] < 1e-12
    assert report["alignment_unchanged"] is True
    assert report["cells"] == 4


def test_conservation_fails_when_the_two_artifacts_are_scaled_differently():
    """A gate that cannot fail is not a gate."""
    overlay, state_table = _overlay_and_state_tables()
    tampered = overlay.copy()
    tampered.loc[0, block_group_overlay_column("murder")] *= 1.5
    report = conservation_report(
        block_group_overlay=tampered, state_provisional=state_table, offenses=OVERLAY_OFFENSES
    )
    assert report["conserved"] is False
    assert report["block_group_factor_exact"] is False
    assert report["alignment_unchanged"] is False


# --- national comparability -----------------------------------------------------------------


def test_national_comparability_passes_when_the_factors_track_the_source():
    factors = _two_state_factors()
    controls = _controls({"AL": {"murder": 1000.0}, "MS": {"murder": 1000.0}})
    state_table = state_provisional_totals(apply_factors_to_controls(controls, factors))
    # AL 0.9, MS 1.2 on equal weights -> implied 1.05; hand the same number to the reference.
    references = {"murder": {"national": 1.05, "scope_matched": 1.05}}
    report = national_comparability(state_table, references)
    assert report["ok"] is True
    row = report["offenses"][0]
    assert row["implied_ratio"] == pytest.approx(1.05)
    assert row["gap_pct_of_reference"] == pytest.approx(0.0, abs=1e-9)


def test_national_comparability_flags_a_factor_field_applied_to_the_wrong_weights():
    factors = _two_state_factors()
    controls = _controls({"AL": {"murder": 1000.0}, "MS": {"murder": 1000.0}})
    state_table = state_provisional_totals(apply_factors_to_controls(controls, factors))
    references = {"murder": {"national": 0.80, "scope_matched": 0.80}}
    report = national_comparability(state_table, references)
    assert report["ok"] is False
    assert report["offenses"][0]["within_tolerance"] is False


def test_both_references_come_from_the_same_source_and_the_same_months():
    panel = make_panel(
        {
            ("AL", "homicide", 2024): (flat(600), 100.0),
            ("AL", "homicide", 2025): (flat(300), 100.0),
            ("MS", "homicide", 2024): (flat(400), 100.0),
            ("MS", "homicide", 2025): (flat(300), 100.0),
        },
        us_rate={("homicide", 2024): flat(12.0), ("homicide", 2025): flat(6.0)},
    )
    references = cde_reference_ratios(panel)
    # scope-matched is the count aggregate of the two states pulled: 600/1000.
    assert references["murder"]["scope_matched"] == pytest.approx(0.6)
    # the source's own national series is its own statement, and here it disagrees.
    assert references["murder"]["national"] == pytest.approx(0.5)


# --- vintage --------------------------------------------------------------------------------


def test_vintage_carries_the_sources_own_as_of_stamp():
    panel = make_panel(
        {
            ("AL", "homicide", 2024): (flat(600), 100.0),
            ("AL", "homicide", 2025): ([50.0, 50.0, 50.0], 100.0),
        },
        max_data_date="04/2026",
        last_refresh_date="04/15/2026",
    )
    vintage = cde_vintage(panel, path=Path("/tmp/panel.parquet"))
    assert vintage["max_data_date"] == "04/2026"
    assert vintage["last_refresh_date"] == "04/15/2026"
    assert vintage["months_by_year"][2025] == [1, 2, 3]
    assert vintage["state_count"] == 1


def test_a_panel_with_two_vintages_is_refused():
    panel = make_panel(
        {
            ("AL", "homicide", 2024): (flat(600), 100.0),
            ("AL", "homicide", 2025): (flat(600), 100.0),
        }
    )
    panel.loc[0, "max_data_date"] = "01/2026"
    with pytest.raises(ValueError, match="exactly one source vintage"):
        cde_vintage(panel)


def test_a_panel_missing_a_required_column_is_refused(tmp_path: Path):
    path = tmp_path / "panel.parquet"
    pd.DataFrame({"state": ["AL"], "count": [1]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not a CDE state-monthly panel"):
        read_cde_state_monthly(path)


# --- the edition type -----------------------------------------------------------------------


def test_the_provisional_type_builds_and_keeps_its_disclaimers():
    assert set(EDITION_TYPES) == {"annual", "quarterly-provisional"}
    assert QUARTERLY_PROVISIONAL.buildable
    require_buildable(parse_edition_id("2025Q2-quarterly-provisional"))
    note = QUARTERLY_PROVISIONAL.note
    assert "preliminary" in note and "as-of" in note and "revision history" in note
    assert "no finality claim" in note.lower()
    assert "never supersedes" in note


def test_neither_builder_will_build_the_other_types_edition():
    with pytest.raises(EditionTypeMismatch, match="package_edition.py"):
        require_edition_type(
            parse_edition_id("2025Q2-quarterly-provisional"),
            ANNUAL.key,
            builder="package_edition.py",
        )
    with pytest.raises(EditionTypeMismatch, match="build_provisional_edition.py"):
        require_edition_type(
            parse_edition_id("2024A-annual"),
            QUARTERLY_PROVISIONAL.key,
            builder="build_provisional_edition.py",
        )


def test_the_provisional_label_says_all_three_things():
    lowered = PROVISIONAL_LABEL.lower()
    assert "provisional" in lowered
    assert "revised continuously" in lowered
    assert "not comparable to the annual accounting surface" in lowered


# --- the real acquired panel ------------------------------------------------------------------

CDE_PANEL = REPO_ROOT / "analysis_scratch" / "final_phase" / "fbi_2025" / (
    "state_month_offense_2025.parquet"
)


@pytest.mark.skipif(not CDE_PANEL.exists(), reason="acquired CDE panel not present")
def test_the_real_panel_produces_one_factor_per_state_and_offence():
    panel = read_cde_state_monthly(CDE_PANEL)
    factors = provisional_factors(panel)
    assert set(factors["offense"]) == set(OFFENSES_7)
    assert len(factors) == factors["state_abbr"].nunique() * len(OFFENSES_7)
    assert factors["provisional_factor"].between(0.6, 1.5).all()
    assert factors["months_used"].between(1, 12).all()
    assert (factors["credibility_weight"] <= 1.0).all()
    assert damping_report(factors)["weight_equals_shrink_where_unclipped"] is True


@pytest.mark.skipif(not CDE_PANEL.exists(), reason="acquired CDE panel not present")
def test_no_real_state_is_below_the_low_coverage_line_so_that_test_is_synthetic():
    """Why `test_low_coverage_states_shrink_visibly_toward_one` uses a made-up world.

    Every state in the acquired vintage reports above 70% population coverage, so the damping's
    behaviour at low coverage cannot be exercised on the real panel. If a future vintage drops a
    state below the line, this test fails and says so -- which is the point: the surface would
    then be publishing a damped factor for a real state and that fact should not go unnoticed.
    """
    report = damping_report(provisional_factors(read_cde_state_monthly(CDE_PANEL)))
    assert report["low_coverage_rows"] == 0, (
        f"states below {report['low_coverage_pct']}% coverage appeared in this vintage: "
        f"{report['low_coverage_states']}"
    )
    assert report["min_coverage_pct"] > 70.0
