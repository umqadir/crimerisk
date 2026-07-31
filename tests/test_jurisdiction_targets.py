"""Stage 3 — the jurisdiction-target consumption layer.

Four things are load-bearing and are pinned here: the control universe is the ownership
skeleton and not the estimate table; every target amount is a weighted sum of Stage-1
agency estimates with the components preserved; the conservation identities fail closed;
and Jackson MS -- the worked case that showed a jurisdiction-level fill locking a control
before benchmark eligibility was evaluated -- can no longer happen by construction.
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from crimerisk.benchmark_imputation import (
    COUNTY_UNIT_KIND,
    MUNICIPAL_UNIT_KIND,
    BenchmarkImputationConfig,
    _assert_every_silent_unit_lands_on_exactly_one_empty_control_row,
    build_silent_agency_ledger,
)
from crimerisk.jurisdiction_targets import (
    IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN,
    JurisdictionTargetConfig,
    LANE_TARGET_COLUMNS,
    OTHER_LANE_TARGET_COLUMN,
    _assert_crosswalk_weights_partition_every_agency,
    _assert_every_agency_estimate_lands_on_the_skeleton,
    _assert_row_identity,
    assert_agency_mass_equals_control_mass,
    build_jurisdiction_control_skeleton,
    build_jurisdiction_ownership,
    build_jurisdiction_target_components,
    build_jurisdiction_year_estimates,
    build_ownership_exclusions,
    load_crosswalk,
)
from crimerisk.paths import RepoPaths, get_paths
from crimerisk.stage1_adjudications import config_dir
from crimerisk.source_provenance import NIBRS_SOURCE, SUMMARY_SOURCE
from crimerisk.trend_fills import (
    FILL_MAX_REFERENCE_AGE_YEARS,
    TrendFillLookup,
    build_agency_target_estimates_from_panel,
    build_ori_succession_ledger,
)
from crimerisk.crime import OFFENSES_7


JACKSON_ORI = "MS0250100"
JACKSON_JURISDICTION = "28:municipal:place:2836000"
TARGET_YEAR = 2024


# --- fixtures ---------------------------------------------------------------


def _write_state_tree(
    tmp_path: Path,
    *,
    jurisdictions: pd.DataFrame,
    crosswalk: pd.DataFrame,
    block_groups: pd.DataFrame,
    observations: pd.DataFrame | None = None,
    agency_master: pd.DataFrame | None = None,
    roster: pd.DataFrame | None = None,
) -> RepoPaths:
    paths = RepoPaths.from_repo_root(tmp_path)
    stage1_config = config_dir(paths)
    stage1_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config_dir(get_paths()), stage1_config)
    (paths.state_dir / "reference").mkdir(parents=True, exist_ok=True)
    (paths.state_dir / "geometry").mkdir(parents=True, exist_ok=True)
    (paths.state_dir / "observations").mkdir(parents=True, exist_ok=True)
    jurisdictions.to_parquet(paths.state_dir / "reference" / "jurisdiction_master.parquet")
    crosswalk.to_parquet(
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    )
    block_groups.to_parquet(
        paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    )
    if observations is not None:
        observations.to_parquet(
            paths.state_dir / "observations" / "agency_year_observations.parquet"
        )
    if agency_master is not None:
        agency_master.to_parquet(paths.state_dir / "reference" / "agency_master.parquet")
    if roster is not None:
        roster_dir = paths.data_dir / f"FBI-CDE-Agency-Rosters-{TARGET_YEAR}" / "parsed"
        roster_dir.mkdir(parents=True, exist_ok=True)
        roster.to_parquet(roster_dir / f"agency_rosters_{TARGET_YEAR}.parquet")
    return paths


def _jurisdictions(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "jurisdiction_id": jid,
                "jurisdiction_type": jtype,
                "state_fips": state_fips,
                "state_abbr": state_abbr,
                "jurisdiction_name": jid,
                "geo_type": "place",
                "geoid": jid.split(":")[-1],
            }
            for jid, jtype, state_fips, state_abbr in rows
        ]
    )


def _crosswalk(rows: list[tuple[str, str, str, float]], *, relationship: str = "exclusive") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ori": ori,
                "state_fips": state_fips,
                "state_abbr": "MS",
                "jurisdiction_id": jid,
                "relationship_type": relationship,
                "weight": weight,
                "overlap_subtype": pd.NA,
            }
            for ori, state_fips, jid, weight in rows
        ]
    )


def _block_groups(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "state_fips": state_fips,
                "block_group_geoid": bg,
                "jurisdiction_id": jid,
                "jurisdiction_type": jtype,
                "aland20": 1.0e7,
                "pop20": pop,
                "allocation_share": 1.0,
            }
            for state_fips, bg, jid, jtype, pop in [
                (r[0], f"{i:012d}", r[1], r[2], r[3]) for i, r in enumerate(rows, start=1)
            ]
        ]
    )
    if frame.empty:
        frame = pd.DataFrame(
            {
                "state_fips": pd.Series(dtype="string"),
                "block_group_geoid": pd.Series(dtype="string"),
                "jurisdiction_id": pd.Series(dtype="string"),
                "jurisdiction_type": pd.Series(dtype="string"),
                "aland20": pd.Series(dtype="float64"),
                "pop20": pd.Series(dtype="float64"),
                "allocation_share": pd.Series(dtype="float64"),
            }
        )
    return frame


def _estimate_rows(rows: list[tuple[str, str, float, float, str]]) -> pd.DataFrame:
    """(ori9, offense, reported, estimated, source) -> an agency estimate frame."""
    return pd.DataFrame(
        [
            {
                "ori9": ori,
                "state_fips": "28",
                "offense": offense,
                "reported_count_current": reported,
                "estimated_count": estimated,
                "agency_adjustment_count": max(0.0, estimated - reported),
                "agency_estimate_source": source,
                "preferred_source": SUMMARY_SOURCE,
            }
            for ori, offense, reported, estimated, source in rows
        ]
    )


def _panel_rows(rows: list[tuple[str, int, str, float, float, str, str]]) -> pd.DataFrame:
    """(ori9, year, offense, count, months, preferred_source, regime)."""
    frame = pd.DataFrame(
        [
            {
                "ori9": ori,
                "state_fips": "28",
                "state_abbr": "MS",
                "year": year,
                "offense": offense,
                "preferred_count": count,
                "preferred_months_reported": months,
                "preferred_observation_weight": 1.0 if months >= 12 else 0.5,
                "preferred_source": source,
                "reporting_regime": regime,
            }
            for ori, year, offense, count, months, source, regime in rows
        ]
    )
    frame["usable_as_observed"] = frame["reporting_regime"].ne(
        "structurally_missing_or_unreliable"
    ) & ~frame["reporting_regime"].eq("true_partial")
    frame["current_row_is_true_partial"] = frame["reporting_regime"].eq("true_partial")
    frame["reported_count_srs"] = frame["preferred_count"].where(
        frame["preferred_source"].eq(SUMMARY_SOURCE)
    )
    frame["reported_count_nibrs"] = frame["preferred_count"].where(
        frame["preferred_source"].eq(NIBRS_SOURCE)
    )
    return frame


# --- the skeleton -----------------------------------------------------------


def test_a_jurisdiction_whose_agency_is_silent_still_gets_a_control_row(tmp_path):
    """The naive-join failure mode: absence from the estimate table must not mean absence
    from the control universe, or the silent unit vanishes before imputation."""
    paths = _write_state_tree(
        tmp_path,
        jurisdictions=_jurisdictions([(JACKSON_JURISDICTION, "municipal", "28", "MS")]),
        crosswalk=_crosswalk([(JACKSON_ORI, "28", JACKSON_JURISDICTION, 1.0)]),
        block_groups=_block_groups([("28", JACKSON_JURISDICTION, "municipal", 153_701.0)]),
    )
    ownership = build_jurisdiction_ownership(paths=paths)
    skeleton = build_jurisdiction_control_skeleton(ownership)
    assert set(skeleton["jurisdiction_id"]) == {JACKSON_JURISDICTION}
    assert len(skeleton) == len(OFFENSES_7)


def test_a_geometry_less_jurisdiction_is_excluded_and_enumerated(tmp_path):
    paths = _write_state_tree(
        tmp_path,
        jurisdictions=_jurisdictions(
            [
                ("28:municipal:place:0000001", "municipal", "28", "MS"),
                ("28:municipal:place:0000002", "municipal", "28", "MS"),
            ]
        ),
        crosswalk=_crosswalk(
            [
                ("MS0000001", "28", "28:municipal:place:0000001", 1.0),
                ("MS0000002", "28", "28:municipal:place:0000002", 1.0),
            ]
        ),
        block_groups=_block_groups(
            [("28", "28:municipal:place:0000001", "municipal", 1000.0)]
        ),
    )
    ownership = build_jurisdiction_ownership(paths=paths)
    skeleton = build_jurisdiction_control_skeleton(ownership)
    assert set(skeleton["jurisdiction_id"]) == {"28:municipal:place:0000001"}

    exclusions = build_ownership_exclusions(
        ownership=ownership,
        agency_estimates=_estimate_rows([("MS0000002", "larceny", 0.0, 0.0, "observed")]),
        crosswalk=load_crosswalk(paths),
    )
    assert list(exclusions["jurisdiction_id"]) == ["28:municipal:place:0000002"]
    assert list(exclusions["exclusion_reason"]) == ["no_block_group_geometry"]
    assert float(exclusions["agency_estimate_mass"].iloc[0]) == 0.0


def test_agency_mass_routed_to_an_excluded_jurisdiction_fails_closed(tmp_path):
    """A geometry-less place is excluded only while it carries nothing. The moment it
    acquires a reporting agency the mass has nowhere to land and the build must stop --
    the lossy left merge it replaced swallowed 138,530 counts a year in silence."""
    paths = _write_state_tree(
        tmp_path,
        jurisdictions=_jurisdictions(
            [
                ("28:municipal:place:0000001", "municipal", "28", "MS"),
                ("28:municipal:place:0000002", "municipal", "28", "MS"),
            ]
        ),
        crosswalk=_crosswalk(
            [
                ("MS0000001", "28", "28:municipal:place:0000001", 1.0),
                ("MS0000002", "28", "28:municipal:place:0000002", 1.0),
            ]
        ),
        block_groups=_block_groups(
            [("28", "28:municipal:place:0000001", "municipal", 1000.0)]
        ),
    )
    skeleton = build_jurisdiction_control_skeleton(
        build_jurisdiction_ownership(paths=paths)
    )
    with pytest.raises(ValueError, match="outside the control skeleton"):
        _assert_every_agency_estimate_lands_on_the_skeleton(
            agency_estimates=_estimate_rows(
                [("MS0000002", "larceny", 12.0, 12.0, "observed")]
            ),
            crosswalk=load_crosswalk(paths),
            skeleton=skeleton,
        )


def test_the_statewide_overlap_layer_owns_territory_through_its_agencies(tmp_path):
    """The overlap layer holds no block groups by design; its ownership test is the
    agency crosswalk, which is why its target can be the same weighted agency sum as the
    other two lanes instead of a ladder over whoever happened to report."""
    paths = _write_state_tree(
        tmp_path,
        jurisdictions=_jurisdictions(
            [("28:statewide_overlap_layer", "statewide_overlap_layer", "28", "MS")]
        ),
        crosswalk=_crosswalk(
            [("MS0250999", "28", "28:statewide_overlap_layer", 1.0)],
            relationship="overlap",
        ),
        block_groups=_block_groups([]),
    )
    ownership = build_jurisdiction_ownership(paths=paths)
    assert bool(ownership["owns_territory"].iloc[0])
    assert ownership["ownership_basis"].iloc[0] == "statewide_overlap_agencies"


def test_crosswalk_weights_that_do_not_partition_an_agency_fail_closed():
    crosswalk = pd.DataFrame(
        {
            "ori9": ["A", "A", "B"],
            "jurisdiction_id": ["j1", "j2", "j3"],
            "weight": [0.5, 0.3, 1.0],
        }
    )
    with pytest.raises(ValueError, match="do not sum"):
        _assert_crosswalk_weights_partition_every_agency(crosswalk)


# --- the aggregation --------------------------------------------------------


def test_the_target_is_the_weighted_sum_of_the_agency_estimates():
    crosswalk = _crosswalk(
        [("A", "28", "j1", 0.6), ("A2", "28", "j1", 1.0)]
    ).rename(columns={"ori": "ori9"})
    estimates = _estimate_rows(
        [("A", "larceny", 100.0, 100.0, "observed"), ("A2", "larceny", 10.0, 25.0, "hist_median")]
    )
    components = build_jurisdiction_target_components(
        agency_estimates=estimates, crosswalk=crosswalk
    )
    row = components[components["offense"].eq("larceny")].iloc[0]
    assert float(row["estimated_count"]) == pytest.approx(100.0 * 0.6 + 25.0)
    assert float(row["reported_count_from_estimates"]) == pytest.approx(100.0 * 0.6 + 10.0)
    assert float(row["current_year_fill_count"]) == pytest.approx(15.0)
    assert float(row["partial_reporting_uplift_count"]) == 0.0


def test_a_partial_year_annualisation_lands_as_uplift_not_as_fill():
    """The remainder pools published 75,865 counts of genuine partial-year uplift as
    `current_year_fill` because the control flag keyed on an unweighted jurisdiction
    months column. Consumed from the agency estimate class, the split is exact."""
    crosswalk = _crosswalk([("A", "28", "j1", 1.0)]).rename(columns={"ori": "ori9"})
    estimates = _estimate_rows(
        [("A", "larceny", 60.0, 120.0, "true_partial_month_ratio")]
    )
    row = build_jurisdiction_target_components(
        agency_estimates=estimates, crosswalk=crosswalk
    ).iloc[0]
    assert float(row["partial_reporting_uplift_count"]) == pytest.approx(60.0)
    assert float(row["current_year_fill_count"]) == 0.0
    assert float(row["partial_component_count"]) == pytest.approx(120.0)
    assert float(row["fill_component_count"]) == 0.0


def test_mixed_agency_lanes_keep_their_component_masses():
    """A dominant label conceals mixed provenance by construction, so the components are
    published beside it and no decision is made from the label."""
    crosswalk = _crosswalk(
        [("A", "28", "j1", 1.0), ("B", "28", "j1", 1.0)]
    ).rename(columns={"ori": "ori9"})
    estimates = _estimate_rows(
        [("A", "larceny", 90.0, 90.0, "observed"), ("B", "larceny", 10.0, 10.0, "observed")]
    )
    estimates.loc[estimates["ori9"].eq("B"), "preferred_source"] = NIBRS_SOURCE
    row = build_jurisdiction_target_components(
        agency_estimates=estimates, crosswalk=crosswalk
    ).iloc[0]
    assert float(row[LANE_TARGET_COLUMNS[SUMMARY_SOURCE]]) == pytest.approx(90.0)
    assert float(row[LANE_TARGET_COLUMNS[NIBRS_SOURCE]]) == pytest.approx(10.0)
    assert float(row[OTHER_LANE_TARGET_COLUMN]) == pytest.approx(0.0)
    assert int(row["estimating_agency_count"]) == 2


def test_an_agency_with_no_target_year_lane_lands_in_the_residual_component():
    """A fill taken from the agency's own history has no target-year lane at all. It goes
    to the residual bucket so the components still sum to the target exactly, instead of
    disappearing from the provenance."""
    crosswalk = _crosswalk([("MS0000001", "28", "j1", 1.0)]).rename(columns={"ori": "ori9"})
    estimates = _estimate_rows(
        [("MS0000001", "larceny", 0.0, 40.0, "hist_median")]
    ).drop(columns=["preferred_source"])
    row = build_jurisdiction_target_components(
        agency_estimates=estimates,
        crosswalk=crosswalk,
        agency_target_panel=_panel_rows(
            [("MS0009999", TARGET_YEAR, "larceny", 5.0, 12.0, SUMMARY_SOURCE, "full_monthly")]
        ),
    ).iloc[0]
    assert float(row[OTHER_LANE_TARGET_COLUMN]) == pytest.approx(40.0)
    assert float(row["estimated_count"]) == pytest.approx(40.0)


def test_component_lanes_are_read_from_the_target_year_panel():
    """Production shape: the Stage-1 estimate frame carries the amount and the preferred
    panel carries the lane. If the join is skipped the component columns silently read
    zero and mixed provenance is invisible again."""
    crosswalk = _crosswalk(
        [("MS0000001", "28", "j1", 1.0), ("MS0000002", "28", "j1", 1.0)]
    ).rename(columns={"ori": "ori9"})
    estimates = _estimate_rows(
        [
            ("MS0000001", "larceny", 90.0, 90.0, "observed"),
            ("MS0000002", "larceny", 10.0, 10.0, "observed"),
        ]
    ).drop(columns=["preferred_source"])
    panel = _panel_rows(
        [
            ("MS0000001", TARGET_YEAR, "larceny", 90.0, 12.0, SUMMARY_SOURCE, "full_monthly"),
            ("MS0000002", TARGET_YEAR, "larceny", 10.0, 12.0, NIBRS_SOURCE, "annual_only_but_usable"),
        ]
    )
    row = build_jurisdiction_target_components(
        agency_estimates=estimates, crosswalk=crosswalk, agency_target_panel=panel
    ).iloc[0]
    assert float(row[LANE_TARGET_COLUMNS[SUMMARY_SOURCE]]) == pytest.approx(90.0)
    assert float(row[LANE_TARGET_COLUMNS[NIBRS_SOURCE]]) == pytest.approx(10.0)


def test_the_row_identity_is_exact_and_a_break_fails_closed():
    good = pd.DataFrame(
        {
            "jurisdiction_id": ["j1"],
            "offense": ["larceny"],
            "reported_count_preferred": [60.0],
            "partial_reporting_uplift_count": [30.0],
            "current_year_fill_count": [10.0],
            "estimated_count": [100.0],
        }
    )
    _assert_row_identity(good)
    bad = good.assign(estimated_count=[105.0])
    with pytest.raises(ValueError, match="violate"):
        _assert_row_identity(bad)


def test_identity_resolved_reported_mass_is_a_named_ledger_backed_component(tmp_path):
    jurisdiction_id = "28:statewide_overlap_layer"
    paths = _write_state_tree(
        tmp_path,
        jurisdictions=_jurisdictions(
            [(jurisdiction_id, "statewide_overlap_layer", "28", "MS")]
        ),
        crosswalk=_crosswalk(
            [
                ("MS0000001", "28", jurisdiction_id, 1.0),
                ("MS0000002", "28", jurisdiction_id, 1.0),
            ],
            relationship="overlap",
        ),
        block_groups=_block_groups([]),
    )
    panel = _panel_rows(
        [
            (
                "MS0000001",
                TARGET_YEAR,
                "larceny",
                30.0,
                12.0,
                SUMMARY_SOURCE,
                "full_monthly",
            ),
            (
                "MS0000002",
                TARGET_YEAR,
                "larceny",
                30.0,
                12.0,
                SUMMARY_SOURCE,
                "full_monthly",
            ),
        ]
    )
    estimates = _estimate_rows(
        [("MS0000002", "larceny", 30.0, 30.0, "observed")]
    )
    succession = pd.DataFrame(
        {
            "superseded_ori9": ["MS0000001"],
            "successor_ori9": ["MS0000002"],
        }
    )

    with pytest.raises(ValueError, match="violate"):
        build_jurisdiction_year_estimates(
            paths=paths,
            agency_panel=panel,
            agency_estimates=estimates,
        )

    out = build_jurisdiction_year_estimates(
        paths=paths,
        agency_panel=panel,
        agency_estimates=estimates,
        succession_ledger=succession,
    )
    row = out[
        out["year"].eq(TARGET_YEAR) & out["offense"].eq("larceny")
    ].iloc[0]
    assert float(row["reported_count_preferred"]) == pytest.approx(60.0)
    assert float(row[IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN]) == pytest.approx(-30.0)
    assert float(row["estimated_count"]) == pytest.approx(30.0)


def test_row_identity_includes_benchmark_and_identity_resolution_components():
    row = pd.DataFrame(
        {
            "jurisdiction_id": ["j1"],
            "offense": ["larceny"],
            "reported_count_preferred": [100.0],
            IDENTITY_RESOLUTION_ADJUSTMENT_COLUMN: [-30.0],
            "partial_reporting_uplift_count": [10.0],
            "current_year_fill_count": [5.0],
            "benchmark_imputed_count": [15.0],
            "adjusted_count_ags_core": [100.0],
        }
    )
    _assert_row_identity(row)
    with pytest.raises(ValueError, match="violate"):
        _assert_row_identity(row.assign(adjusted_count_ags_core=101.0))


def test_agency_mass_equals_control_mass_per_state_offense_and_lane():
    crosswalk = _crosswalk(
        [("A", "28", "j1", 1.0), ("B", "28", "j2", 1.0)]
    ).rename(columns={"ori": "ori9"})
    estimates = _estimate_rows(
        [("A", "larceny", 10.0, 10.0, "observed"), ("B", "larceny", 5.0, 5.0, "observed")]
    )
    targets = pd.DataFrame(
        {
            "jurisdiction_id": ["j1", "j2"],
            "jurisdiction_type": ["municipal", "statewide_overlap_layer"],
            "state_fips": ["28", "28"],
            "offense": ["larceny", "larceny"],
            "estimated_count": [10.0, 5.0],
        }
    )
    reconciliation = assert_agency_mass_equals_control_mass(
        agency_estimates=estimates, crosswalk=crosswalk, targets=targets
    )
    assert float(reconciliation["delta"].abs().max()) == pytest.approx(0.0)

    # The statewide overlap layer used to take the generic jurisdiction ladder over the
    # reporters-only aggregate and drop every contributing agency's fill: 21,757 counts
    # across 42 states, invisible because nothing compared the two sides.
    dropped = targets.assign(estimated_count=[10.0, 0.0])
    with pytest.raises(ValueError, match="does not equal the agency-estimate mass"):
        assert_agency_mass_equals_control_mass(
            agency_estimates=estimates, crosswalk=crosswalk, targets=dropped
        )


def test_history_years_carry_the_reported_rollup_and_no_fill(tmp_path):
    paths = _write_state_tree(
        tmp_path,
        jurisdictions=_jurisdictions([("28:municipal:place:0000001", "municipal", "28", "MS")]),
        crosswalk=_crosswalk([("MS0000001", "28", "28:municipal:place:0000001", 1.0)]),
        block_groups=_block_groups([("28", "28:municipal:place:0000001", "municipal", 5000.0)]),
    )
    panel = _panel_rows(
        [
            ("MS0000001", 2023, "larceny", 80.0, 12.0, SUMMARY_SOURCE, "full_monthly"),
            ("MS0000001", 2024, "larceny", 90.0, 12.0, SUMMARY_SOURCE, "full_monthly"),
        ]
    )
    estimates = _estimate_rows([("MS0000001", "larceny", 90.0, 90.0, "observed")])
    out = build_jurisdiction_year_estimates(
        paths=paths,
        config=JurisdictionTargetConfig(year_start=2023, target_year=TARGET_YEAR),
        agency_panel=panel,
        agency_estimates=estimates,
    )
    history = out[out["year"].eq(2023) & out["offense"].eq("larceny")].iloc[0]
    target = out[out["year"].eq(2024) & out["offense"].eq("larceny")].iloc[0]
    assert float(history["estimated_count"]) == pytest.approx(80.0)
    assert history["estimate_source"] == "agency_reported_rollup"
    assert float(history["current_year_fill_count"]) == 0.0
    assert float(target["estimated_count"]) == pytest.approx(90.0)
    assert target["estimate_source"] == "agency_rollup_observed"
    # An offense with no agency evidence exists on the skeleton at zero, labelled.
    empty = out[out["year"].eq(2024) & out["offense"].eq("murder")].iloc[0]
    assert float(empty["estimated_count"]) == 0.0
    assert empty["estimate_source"] == "no_agency_evidence"
    # A skeleton row at zero is the absence of an observation, not an observation of
    # zero: it must not read as the agencies' own reported number downstream.
    assert bool(empty["estimated_from_panel"])
    assert not bool(target["estimated_from_panel"])


# --- Jackson MS, and the ordering that makes it impossible ------------------


def _jackson_paths(tmp_path: Path) -> RepoPaths:
    """Jackson MS as the audit traced it: MS0250100 is the only agency of
    28:municipal:place:2836000 at weight 1.0, it last reported usably in 2019, and the
    jurisdiction holds 153,701 residents of exposure."""
    observations = pd.DataFrame(
        {
            "ori9": [JACKSON_ORI, JACKSON_ORI],
            "count": [3553.0, 0.0],
            "months_reported": [12.0, 0.0],
        }
    )
    return _write_state_tree(
        tmp_path,
        jurisdictions=_jurisdictions([(JACKSON_JURISDICTION, "municipal", "28", "MS")]),
        crosswalk=_crosswalk([(JACKSON_ORI, "28", JACKSON_JURISDICTION, 1.0)]),
        block_groups=_block_groups([("28", JACKSON_JURISDICTION, "municipal", 153_701.0)]),
        observations=observations,
        agency_master=pd.DataFrame(
            {
                "ori9": [JACKSON_ORI],
                "state_fips": ["28"],
                "county_fips": ["049"],
                "agency_type_norm": ["local_police"],
            }
        ),
        roster=pd.DataFrame({"ori": [JACKSON_ORI]}),
    )


def _jackson_panel() -> pd.DataFrame:
    return _panel_rows(
        [
            (JACKSON_ORI, 2019, "larceny", 3553.0, 12.0, SUMMARY_SOURCE, "full_monthly"),
            (
                JACKSON_ORI,
                2024,
                "larceny",
                0.0,
                0.0,
                SUMMARY_SOURCE,
                "structurally_missing_or_unreliable",
            ),
        ]
    )


def test_jackson_ms_takes_no_pre_imputation_jurisdiction_fill(tmp_path):
    """GOLDEN. The agency estimator refuses a 2019 reference as past the two-year recency
    bound and emits no row at all. Before the restructure the municipal ladder filled
    3,552.9 larceny from that same reference, wrote it onto the control unconditionally,
    and the control's locked mass then excluded the unit from benchmark imputation. There
    is no jurisdiction ladder left to do that: the target is the agency sum, and the agency
    sum of nothing is zero."""
    paths = _jackson_paths(tmp_path)
    panel = _jackson_panel()
    estimates = build_agency_target_estimates_from_panel(
        panel,
        target_year=TARGET_YEAR,
        trend_fill_lookup=TrendFillLookup(state_map={}, national_map={}),
        max_reference_age_years=FILL_MAX_REFERENCE_AGE_YEARS,
    )
    assert estimates[estimates["ori9"].eq(JACKSON_ORI)].empty

    targets = build_jurisdiction_year_estimates(
        paths=paths,
        config=JurisdictionTargetConfig(year_start=2019, target_year=TARGET_YEAR),
        agency_panel=panel,
        agency_estimates=estimates,
    )
    jackson = targets[
        targets["jurisdiction_id"].eq(JACKSON_JURISDICTION)
        & targets["year"].eq(TARGET_YEAR)
        & targets["offense"].eq("larceny")
    ]
    assert len(jackson) == 1
    assert float(jackson["estimated_count"].iloc[0]) == 0.0
    assert float(jackson["current_year_fill_count"].iloc[0]) == 0.0
    assert jackson["estimate_source"].iloc[0] == "no_agency_evidence"


def test_jackson_ms_is_an_eligible_silent_unit_with_an_empty_control_row(tmp_path):
    """GOLDEN, continued: silent in the agency ledger, and the control row that will carry
    its imputed mass exists and is empty."""
    paths = _jackson_paths(tmp_path)
    panel = _jackson_panel()
    estimates = build_agency_target_estimates_from_panel(
        panel,
        target_year=TARGET_YEAR,
        trend_fill_lookup=TrendFillLookup(state_map={}, national_map={}),
        max_reference_age_years=FILL_MAX_REFERENCE_AGE_YEARS,
    )
    target_panel = panel[panel["year"].eq(TARGET_YEAR)]
    ledger = build_silent_agency_ledger(
        paths=paths,
        config=BenchmarkImputationConfig(year=TARGET_YEAR),
        agency_preferred=target_panel,
        agency_estimates=estimates,
    )
    jackson = ledger[ledger["ori9"].eq(JACKSON_ORI)].iloc[0]
    assert not bool(jackson["is_supported"])
    assert not bool(jackson["is_fill_covered"])
    assert not bool(jackson["is_dead"])
    assert bool(jackson["is_eligible_silent"])

    silent = pd.DataFrame(
        {
            "unit_id": [JACKSON_JURISDICTION],
            "unit_kind": [MUNICIPAL_UNIT_KIND],
            "state_fips": ["28"],
            "exposure_population": [153_701.0],
            "has_control_row": [True],
            "locked_total": [0.0],
        }
    )
    _assert_every_silent_unit_lands_on_exactly_one_empty_control_row(silent)


def test_a_silent_unit_whose_control_already_carries_mass_fails_closed():
    """The Jackson defect stated as a post-condition: an eligible silent unit with locked
    pre-imputation mass means a fourth sizing lane has re-opened."""
    occupied = pd.DataFrame(
        {
            "unit_id": [JACKSON_JURISDICTION],
            "unit_kind": [MUNICIPAL_UNIT_KIND],
            "state_fips": ["28"],
            "exposure_population": [153_701.0],
            "has_control_row": [True],
            "locked_total": [3552.9],
        }
    )
    with pytest.raises(ValueError, match="already carry pre-imputation"):
        _assert_every_silent_unit_lands_on_exactly_one_empty_control_row(occupied)


def test_a_silent_unit_with_no_control_row_fails_closed():
    stranded = pd.DataFrame(
        {
            "unit_id": ["28:state_nonmunicipal_remainder:county:28049"],
            "unit_kind": [COUNTY_UNIT_KIND],
            "state_fips": ["28"],
            "exposure_population": [10_000.0],
            "has_control_row": [False],
            "locked_total": [0.0],
        }
    )
    with pytest.raises(ValueError, match="no control row"):
        _assert_every_silent_unit_lands_on_exactly_one_empty_control_row(stranded)


def test_a_superseded_ori_is_excluded_from_the_silent_ledger(tmp_path):
    """`build_benchmark_imputation` never passed a succession ledger, so `is_superseded`
    was dead in the production path and a retired ORI still on the FBI roster was excluded
    only as a side effect of its control row being empty. The ledger now reaches here."""
    paths = _jackson_paths(tmp_path)
    panel = _jackson_panel()
    estimates = build_agency_target_estimates_from_panel(
        panel,
        target_year=TARGET_YEAR,
        trend_fill_lookup=TrendFillLookup(state_map={}, national_map={}),
        max_reference_age_years=FILL_MAX_REFERENCE_AGE_YEARS,
    )
    target_panel = panel[panel["year"].eq(TARGET_YEAR)]
    succession = pd.DataFrame(
        {
            "superseded_ori9": [JACKSON_ORI],
            "successor_ori9": ["MS0250200"],
            "jurisdiction_id": [JACKSON_JURISDICTION],
        }
    )
    without = build_silent_agency_ledger(
        paths=paths,
        config=BenchmarkImputationConfig(year=TARGET_YEAR),
        agency_preferred=target_panel,
        agency_estimates=estimates,
    )
    with_ledger = build_silent_agency_ledger(
        paths=paths,
        config=BenchmarkImputationConfig(year=TARGET_YEAR),
        agency_preferred=target_panel,
        agency_estimates=estimates,
        succession_ledger=succession,
    )
    assert bool(without[without["ori9"].eq(JACKSON_ORI)]["is_eligible_silent"].iloc[0])
    assert bool(with_ledger[with_ledger["ori9"].eq(JACKSON_ORI)]["is_superseded"].iloc[0])
    assert not bool(
        with_ledger[with_ledger["ori9"].eq(JACKSON_ORI)]["is_eligible_silent"].iloc[0]
    )


def test_the_succession_ledger_the_controls_build_passes_is_the_one_stage_1_uses():
    """One ledger, computed once: identity resolution has to reach the fill lane and the
    benchmark lane or the retired ORI is resurrected by whichever one missed it."""
    panel = _panel_rows(
        [
            ("MS0000001", 2019, "larceny", 100.0, 12.0, SUMMARY_SOURCE, "full_monthly"),
            ("MS0000002", 2024, "larceny", 120.0, 12.0, SUMMARY_SOURCE, "full_monthly"),
        ]
    )
    crosswalk = pd.DataFrame(
        {
            "ori9": ["MS0000001", "MS0000002"],
            "state_fips": ["28", "28"],
            "jurisdiction_id": ["28:municipal:place:0000001"] * 2,
            "weight": [1.0, 1.0],
        }
    )
    ledger = build_ori_succession_ledger(
        agency_panel=panel,
        agency_jurisdiction_crosswalk=crosswalk,
        target_year=TARGET_YEAR,
        max_reference_age_years=FILL_MAX_REFERENCE_AGE_YEARS,
    )
    assert list(ledger["superseded_ori9"]) == ["MS0000001"]
    assert list(ledger["successor_ori9"]) == ["MS0000002"]


def test_no_estimate_source_label_is_used_to_pick_a_source():
    """The composition labels describe the mass on the row; they are not rungs of a
    ladder and nothing selects from them. Pinned so a future change cannot quietly turn
    the dominant label back into a decision input."""
    module = __import__("crimerisk.jurisdiction_targets", fromlist=["*"])
    text = Path(module.__file__).read_text()
    assert "initialize_preferred_source" not in text
    assert "prefer_nibrs" not in text
    assert "usable_as_observed" not in text.split("def build_agency_target_panel_slice")[0]
