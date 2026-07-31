"""The Stage-1 ad-hoc adjudication registries, and what each consumption point does with them.

Three sites consume them and each is tested twice: once against the LIVE shipped registry, so a
registry edit that breaks the contract fails here rather than in a build, and once against a
hand-built frame, so the fail-closed behaviour is exercised without needing the real data.
"""
from __future__ import annotations

import shutil

import pandas as pd
import pytest

from crimerisk.agency_identity import (
    CROSS_LANE_TWIN_LEDGER_COLUMNS,
    assert_adjudicated_distinct_agencies_are_not_merged,
    build_adjudicated_succession_ledger,
    build_adjudicated_twin_ledger,
    combine_twin_ledgers,
)
from crimerisk.benchmark_imputation import _supported_ori_set
from crimerisk.paths import get_paths
from crimerisk.reporting_regimes import STAGE1_ADJUDICATED_MISSING_REASON
from crimerisk.source_provenance import NIBRS_SOURCE
from crimerisk.stage1_adjudications import (
    ACTION_DIRECTIVE,
    VALID_TWIN_VERDICTS,
    VALID_ZERO_TOKEN_ACTIONS,
    VALID_ZERO_TOKEN_VERDICTS,
    Stage1AdjudicationError,
    build_token_reporter_directives,
    build_usability_directives,
    build_zero_missing_directives,
    config_dir,
    load_token_reporter_rulings,
    load_twin_rulings,
    load_zero_missing_rulings,
)
from crimerisk.trend_fills import (
    STAGE1_ADJUDICATED_LADDER_KIND,
    STAGE1_ADJUDICATED_PARTIAL_KIND,
    _assert_no_fill_where_the_chosen_lane_reported,
    add_preferred_support_flags,
    apply_stage1_adjudicated_usability,
    stage1_adjudicated_estimate_kinds,
)

TARGET_YEAR = 2024


# ---------------------------------------------------------------- helpers
def _paths_with_configs(tmp_path, *, edit=None):
    """A RepoPaths stand-in whose configs/ is a copy of the shipped one, optionally edited."""
    source = config_dir(get_paths())
    destination = tmp_path / "configs" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if edit is not None:
        edit(destination)

    class _Paths:
        repo_root = tmp_path

    return _Paths()


def _rewrite(path, transform):
    path.write_text(transform(path.read_text()))


def _twin_rulings(rows):
    """rows: (case_id, verdict, canonical, [oris])"""
    return pd.DataFrame(
        [
            {
                "case_id": case_id,
                "state": "XX",
                "oris": ";".join(oris),
                "ori_list": list(oris),
                "verdict": verdict,
                "canonical_ori": canonical,
                "downstream_action": (
                    "keep_all_oris" if verdict == "distinct_agencies" else "merge_dedupe"
                ),
                "confidence": "high",
                "needs_review": "0",
                "source_registry": "test",
            }
            for case_id, verdict, canonical, oris in rows
        ]
    )


def _observations(oris):
    return pd.DataFrame(
        {"ori9": list(oris), "state_abbr": ["XX"] * len(oris), "year": [TARGET_YEAR] * len(oris)}
    )


def _crosswalk(pairs):
    return pd.DataFrame(
        [
            {
                "ori9": ori,
                "state_fips": "01",
                "jurisdiction_id": jurisdiction,
                "weight": 1.0,
            }
            for ori, jurisdiction in pairs
        ]
    )


def _directives(rows):
    """rows: (ori9, directive, months, source_registry)"""
    return pd.DataFrame(
        [
            {
                "ori9": ori,
                "year": TARGET_YEAR,
                "directive": directive,
                "believable_months": months,
                "verdict": "token_reporting_flag",
                "case_id": f"c-{ori}",
                "source_registry": registry,
                "confidence": "high",
                "needs_review": "0",
            }
            for ori, directive, months, registry in rows
        ]
    )


def _panel(rows):
    """rows: (ori9, year, months, count, usable, true_partial)"""
    return pd.DataFrame(
        [
            {
                "ori9": ori,
                "state_fips": "01",
                "offense": "burglary",
                "year": year,
                "preferred_count": count,
                "preferred_months_reported": months,
                "usable_as_observed": usable,
                "current_row_is_true_partial": partial,
            }
            for ori, year, months, count, usable, partial in rows
        ]
    )


# ---------------------------------------------------------------- the loader, live registry
def test_live_registries_declare_only_known_verdicts_and_actions():
    paths = get_paths()
    twins = load_twin_rulings(paths)
    zeros = load_zero_missing_rulings(paths, target_year=TARGET_YEAR)
    tokens = load_token_reporter_rulings(paths, target_year=TARGET_YEAR)
    assert not twins.empty and not zeros.empty and not tokens.empty
    assert set(twins["verdict"]) <= VALID_TWIN_VERDICTS
    for frame in (zeros, tokens):
        assert set(frame["verdict"]) <= VALID_ZERO_TOKEN_VERDICTS
        assert set(frame["downstream_action"]) <= VALID_ZERO_TOKEN_ACTIONS
        assert set(frame["target_year"].astype(int)) == {TARGET_YEAR}


def test_live_registries_produce_one_directive_per_agency_year():
    directives = build_usability_directives(get_paths(), target_year=TARGET_YEAR)
    assert not directives.empty
    assert not directives.duplicated(["ori9", "year"]).any()
    assert set(directives["directive"]) == {"reads_missing", "partial_year"}
    partial = directives[directives["directive"].eq("partial_year")]
    months = pd.to_numeric(partial["believable_months"])
    assert months.between(1, 11).all(), "a partial-year directive must carry believable months"


def test_benchmark_supported_predicate_consumes_stage1_usability_directives():
    paths = get_paths()
    directives = build_usability_directives(paths, target_year=TARGET_YEAR)
    reads_missing = directives[directives["directive"].eq("reads_missing")].iloc[0]["ori9"]
    partial = directives[directives["directive"].eq("partial_year")].iloc[0]["ori9"]
    ordinary = "ZZ9999999"
    panel = pd.DataFrame(
        {
            "ori9": [reads_missing, partial, ordinary],
            "usable_as_observed": [True, False, True],
            "current_row_is_true_partial": [False, True, False],
        }
    )

    supported = _supported_ori_set(
        panel,
        paths=paths,
        target_year=TARGET_YEAR,
    )

    assert reads_missing not in supported
    assert partial in supported
    assert ordinary in supported


def test_every_action_in_the_live_registries_maps_to_a_declared_directive():
    paths = get_paths()
    actions = set(load_zero_missing_rulings(paths, target_year=TARGET_YEAR)["downstream_action"])
    actions |= set(load_token_reporter_rulings(paths, target_year=TARGET_YEAR)["downstream_action"])
    assert actions <= set(ACTION_DIRECTIVE)


def test_a_registry_without_a_provenance_header_fails_closed(tmp_path):
    def strip_header(directory):
        _rewrite(
            directory / "zero_missing_adjudicated.csv",
            lambda text: "\n".join(
                line for line in text.splitlines() if not line.startswith("#")
            )
            + "\n",
        )

    paths = _paths_with_configs(tmp_path, edit=strip_header)
    with pytest.raises(Stage1AdjudicationError, match="provenance header is missing keys"):
        load_zero_missing_rulings(paths, target_year=TARGET_YEAR)


def test_a_hand_edited_row_count_fails_closed(tmp_path):
    def drop_a_row(directory):
        path = directory / "token_reporters_adjudicated.csv"
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:-1]))

    paths = _paths_with_configs(tmp_path, edit=drop_a_row)
    with pytest.raises(Stage1AdjudicationError, match="rows_written"):
        load_token_reporter_rulings(paths, target_year=TARGET_YEAR)


def test_an_unknown_verdict_fails_closed(tmp_path):
    def bad_verdict(directory):
        _rewrite(
            directory / "zero_missing_adjudicated.csv",
            lambda text: text.replace("misread_missing", "probably_fine", 1),
        )

    paths = _paths_with_configs(tmp_path, edit=bad_verdict)
    with pytest.raises(Stage1AdjudicationError, match="unknown verdicts"):
        load_zero_missing_rulings(paths, target_year=TARGET_YEAR)


def test_applying_a_registry_to_another_year_fails_closed(tmp_path):
    paths = _paths_with_configs(tmp_path)
    with pytest.raises(Stage1AdjudicationError, match="adjudicated for 2024 but the build asked"):
        load_zero_missing_rulings(paths, target_year=2023)


def test_a_partial_year_verdict_without_believable_months_degrades_to_the_ladder(tmp_path):
    """No month count means no ratio, so the year goes to the ladder rather than get one invented."""

    def blank_the_months(directory):
        path = directory / "token_reporters_adjudicated.csv"
        text = path.read_text()
        header, body = text.split("case_id,", 1)
        frame = pd.read_csv(
            pd.io.common.StringIO("case_id," + body), dtype="string"
        )
        frame["believable_months"] = pd.NA
        out = header + frame.to_csv(index=False)
        path.write_text(out)

    paths = _paths_with_configs(tmp_path, edit=blank_the_months)
    directives = build_token_reporter_directives(paths, target_year=TARGET_YEAR)
    assert not directives["directive"].eq("partial_year").any()
    assert directives["directive"].eq("reads_missing").sum() > 0


# ---------------------------------------------------------------- SITE 1: identity
def test_a_mass_neutral_adjudicated_merge_is_applied():
    rulings = _twin_rulings([("case-a", "same_agency_merge", "AA0000100", ["AA0000100", "AA000019E"])])
    ledger, quarantined = build_adjudicated_twin_ledger(
        rulings,
        observations=_observations(["AA0000100", "AA000019E"]),
        agency_jurisdiction_crosswalk=_crosswalk(
            [("AA0000100", "01:municipal:place:0100100"),
             ("AA000019E", "01:municipal:place:0100100")]
        ),
    )
    assert quarantined.empty
    assert ledger["variant_ori9"].tolist() == ["AA000019E"]
    assert ledger["canonical_ori9"].tolist() == ["AA0000100"]
    assert ledger["identity_evidence"].tolist() == ["adjudicated_case_review"]


def test_a_merge_that_would_relocate_mass_is_quarantined_not_applied():
    """The registry has no surviving-footprint field, so a merge across footprints is not applied."""
    rulings = _twin_rulings([("case-b", "same_agency_merge", "BB0000100", ["BB0000100", "BB0000200"])])
    ledger, quarantined = build_adjudicated_twin_ledger(
        rulings,
        observations=_observations(["BB0000100", "BB0000200"]),
        agency_jurisdiction_crosswalk=_crosswalk(
            [("BB0000100", "01:statewide_overlap_layer"),
             ("BB0000200", "01:municipal:place:0100200")]
        ),
    )
    assert ledger.empty
    assert quarantined["case_id"].tolist() == ["case-b"]
    assert "different jurisdiction footprints" in quarantined["reason"].iloc[0]


def test_a_merge_whose_canonical_has_no_footprint_is_quarantined():
    rulings = _twin_rulings([("case-c", "same_agency_merge", "CC0000100", ["CC0000100", "CC000019E"])])
    ledger, quarantined = build_adjudicated_twin_ledger(
        rulings,
        observations=_observations(["CC0000100", "CC000019E"]),
        agency_jurisdiction_crosswalk=_crosswalk([("CC000019E", "01:municipal:place:0100100")]),
    )
    assert ledger.empty
    assert quarantined["case_id"].tolist() == ["case-c"]


def test_the_rule_and_the_registry_agreeing_is_deduped_not_double_applied():
    rule = pd.DataFrame(
        [
            {
                "variant_ori9": "DD000019E",
                "canonical_ori9": "DD0000100",
                "ori7_stem": "DD00001",
                "state_abbr": "XX",
                "match_years": 3,
                "matched_offense_categories": 4,
                "matched_max_total": 12.0,
                "variant_on_fbi_roster": True,
                "canonical_on_fbi_roster": False,
                "identity_evidence": "offense_vector_identity",
            }
        ],
        columns=CROSS_LANE_TWIN_LEDGER_COLUMNS,
    )
    adjudicated = rule.assign(identity_evidence="adjudicated_case_review")
    combined, already = combine_twin_ledgers(rule, adjudicated)
    assert already == 1
    assert len(combined) == 1
    assert combined["identity_evidence"].tolist() == ["offense_vector_identity"]


def test_the_rule_and_the_registry_disagreeing_fails_closed():
    rule = pd.DataFrame(
        [
            {
                "variant_ori9": "EE000019E",
                "canonical_ori9": "EE0000100",
                "ori7_stem": "EE00001",
                "state_abbr": "XX",
                "match_years": 3,
                "matched_offense_categories": 4,
                "matched_max_total": 12.0,
                "variant_on_fbi_roster": True,
                "canonical_on_fbi_roster": False,
                "identity_evidence": "offense_vector_identity",
            }
        ],
        columns=CROSS_LANE_TWIN_LEDGER_COLUMNS,
    )
    adjudicated = rule.assign(canonical_ori9="EE0000200", identity_evidence="adjudicated_case_review")
    with pytest.raises(Stage1AdjudicationError, match="disagree about who an agency is"):
        combine_twin_ledgers(rule, adjudicated)


def test_merging_a_pair_a_reviewer_ruled_distinct_fails_closed():
    ledger = pd.DataFrame(
        [
            {
                "variant_ori9": "FF0000200",
                "canonical_ori9": "FF0000100",
                "ori7_stem": "FF00002",
                "state_abbr": "XX",
                "match_years": 2,
                "matched_offense_categories": 2,
                "matched_max_total": 4.0,
                "variant_on_fbi_roster": True,
                "canonical_on_fbi_roster": False,
                "identity_evidence": "offense_vector_identity",
            }
        ],
        columns=CROSS_LANE_TWIN_LEDGER_COLUMNS,
    )
    rulings = _twin_rulings([("case-d", "distinct_agencies", "", ["FF0000100", "FF0000200"])])
    with pytest.raises(Stage1AdjudicationError, match="ruled distinct_agencies"):
        assert_adjudicated_distinct_agencies_are_not_merged(ledger, rulings)


def test_an_adjudicated_succession_needs_the_survivor_standing_on_the_dead_footprint():
    shared = _crosswalk(
        [("GG0000100", "01:municipal:place:0100100"),
         ("GG0000200", "01:municipal:place:0100100")]
    )
    split = _crosswalk(
        [("GG0000100", "01:municipal:place:0100100"),
         ("GG0000200", "01:state_nonmunicipal_remainder")]
    )
    rulings = _twin_rulings([("case-e", "superseded_ori", "GG0000200", ["GG0000100", "GG0000200"])])
    observations = _observations(["GG0000100", "GG0000200"])

    ledger, quarantined = build_adjudicated_succession_ledger(
        rulings,
        observations=observations,
        agency_jurisdiction_crosswalk=shared,
        target_year=TARGET_YEAR,
    )
    assert quarantined.empty
    assert ledger["superseded_ori9"].tolist() == ["GG0000100"]
    assert ledger["successor_ori9"].tolist() == ["GG0000200"]

    ledger, quarantined = build_adjudicated_succession_ledger(
        rulings,
        observations=observations,
        agency_jurisdiction_crosswalk=split,
        target_year=TARGET_YEAR,
    )
    assert ledger.empty
    assert quarantined["case_id"].tolist() == ["case-e"]


def test_the_live_twin_registry_partitions_into_applied_quarantined_or_absent():
    """The shipped registry's own split, so a footprint edit cannot change it silently.

    Note the panel is read from the BUILT artifact, which is already merged, so the collapsed ORIs
    are gone and the admitted set reads as empty here. What is asserted is the partition and the
    reasons: every merge ruling is applied, quarantined with a stated reason, or absent from the
    panel -- never silently dropped.
    """
    paths = get_paths()
    observations = pd.read_parquet(
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        columns=["ori9", "state_abbr", "year"],
    )
    from crimerisk.agency_identity import load_agency_jurisdiction_crosswalk

    crosswalk = load_agency_jurisdiction_crosswalk(paths)
    rulings = load_twin_rulings(paths)
    ledger, quarantined = build_adjudicated_twin_ledger(
        rulings, observations=observations, agency_jurisdiction_crosswalk=crosswalk
    )
    merges = rulings[rulings["verdict"].eq("same_agency_merge")]
    applied_cases = {
        ruling["case_id"]
        for ruling in merges.to_dict(orient="records")
        if ruling["canonical_ori"] in set(ledger["canonical_ori9"].astype(str))
    }
    quarantined_cases = set(quarantined["case_id"])
    assert not applied_cases & quarantined_cases
    assert quarantined["reason"].str.len().gt(0).all()
    assert quarantined_cases <= set(merges["case_id"])

    _, succession_quarantined = build_adjudicated_succession_ledger(
        rulings,
        observations=observations,
        agency_jurisdiction_crosswalk=crosswalk,
        target_year=TARGET_YEAR,
    )
    assert "a3-CA-yountville-vs-yountville-police-department" in set(
        succession_quarantined["case_id"]
    ), "the Yountville succession moves a town's counts into a state remainder and must not apply"


# ---------------------------------------------------------------- SITE 2: panel semantics
def test_a_structurally_missing_regime_is_not_usable_as_observed():
    """The link the zero registry rides: the regime value the adjudication writes is what
    `add_preferred_support_flags` reads as "not a measured year"."""
    preferred = pd.DataFrame(
        {
            "preferred_count": [4.0, 4.0],
            "preferred_months_reported": [12.0, 12.0],
            "reporting_regime": ["annual_only_but_usable", "structurally_missing_or_unreliable"],
            "preferred_source": [NIBRS_SOURCE, NIBRS_SOURCE],
        }
    )
    out = add_preferred_support_flags(preferred)
    assert out["usable_as_observed"].tolist() == [True, False]
    assert out["current_row_is_true_partial"].tolist() == [False, False]


def test_the_built_regime_artifact_carries_the_adjudicated_reason_for_exactly_the_ruled_zeros():
    paths = get_paths()
    path = paths.state_dir / "modeling" / "agency_year_reporting_regimes.parquet"
    if not path.exists():
        pytest.skip("reporting regimes have not been built in this tree")
    regimes = pd.read_parquet(path, columns=["ori9", "year", "reporting_regime", "regime_reason"])
    marked = regimes[regimes["regime_reason"].eq(STAGE1_ADJUDICATED_MISSING_REASON)]
    directives = build_zero_missing_directives(paths, target_year=TARGET_YEAR)
    expected = set(directives.loc[directives["directive"].eq("reads_missing"), "ori9"])
    stands = set(directives.loc[directives["directive"].eq("stands"), "ori9"])
    assert set(marked["ori9"]) == expected
    assert set(marked["year"].astype(int)) == {TARGET_YEAR}
    assert marked["reporting_regime"].eq("structurally_missing_or_unreliable").all()
    assert not stands & set(marked["ori9"]), "a genuine_zero verdict must leave the panel alone"


# ---------------------------------------------------------------- SITE 3: estimator usability
def test_a_reads_missing_directive_sends_the_year_to_the_ladder():
    panel = _panel([("HH0000100", TARGET_YEAR, 12.0, 1.0, True, False)])
    out, counts = apply_stage1_adjudicated_usability(
        panel,
        target_year=TARGET_YEAR,
        directives=_directives([("HH0000100", "reads_missing", None, "token_reporters_adjudicated.csv")]),
    )
    assert out["usable_as_observed"].tolist() == [False]
    assert out["current_row_is_true_partial"].tolist() == [False]
    assert out["preferred_months_reported"].tolist() == [12.0]
    assert counts["stage1_adjudicated_rows_to_ladder"] == 1
    assert counts["stage1_adjudicated_rows_to_partial"] == 0
    assert counts["stage1_adjudicated_flipped_only_inside_the_estimator"] == 0


def test_a_partial_year_directive_becomes_a_true_partial_at_the_believable_months():
    panel = _panel([("II0000100", TARGET_YEAR, 12.0, 16.0, True, False)])
    out, counts = apply_stage1_adjudicated_usability(
        panel,
        target_year=TARGET_YEAR,
        directives=_directives([("II0000100", "partial_year", 3.0, "token_reporters_adjudicated.csv")]),
    )
    assert out["usable_as_observed"].tolist() == [False]
    assert out["current_row_is_true_partial"].tolist() == [True]
    assert out["preferred_months_reported"].tolist() == [3.0]
    assert counts["stage1_adjudicated_rows_to_partial"] == 1


def test_only_the_target_year_is_touched():
    panel = _panel(
        [
            ("JJ0000100", TARGET_YEAR - 1, 12.0, 40.0, True, False),
            ("JJ0000100", TARGET_YEAR, 12.0, 1.0, True, False),
        ]
    )
    out, _ = apply_stage1_adjudicated_usability(
        panel,
        target_year=TARGET_YEAR,
        directives=_directives([("JJ0000100", "reads_missing", None, "token_reporters_adjudicated.csv")]),
    )
    assert out["usable_as_observed"].tolist() == [True, False]


def test_a_zero_registry_verdict_that_never_reached_the_regime_site_fails_closed():
    """The zero registry lands at the regime site; if the estimator still sees the row as observed
    the regime build did not run, and flipping the flag here would hide that."""
    panel = _panel([("KK0000100", TARGET_YEAR, 12.0, 0.0, True, False)])
    with pytest.raises(Stage1AdjudicationError, match="reporting-regime site did not apply"):
        apply_stage1_adjudicated_usability(
            panel,
            target_year=TARGET_YEAR,
            directives=_directives(
                [("KK0000100", "reads_missing", None, "state/qa/zero_missing_adjudicated.csv")]
            ),
        )


def test_an_already_unusable_zero_registry_row_passes():
    panel = _panel([("LL0000100", TARGET_YEAR, 12.0, 0.0, False, False)])
    out, counts = apply_stage1_adjudicated_usability(
        panel,
        target_year=TARGET_YEAR,
        directives=_directives(
            [("LL0000100", "reads_missing", None, "state/qa/zero_missing_adjudicated.csv")]
        ),
    )
    assert out["usable_as_observed"].tolist() == [False]
    assert counts["stage1_adjudicated_agency_years_matched"] == 1


def test_estimate_kind_labels_split_partial_from_ladder():
    kinds = stage1_adjudicated_estimate_kinds(
        _directives(
            [
                ("MM0000100", "reads_missing", None, "tokens"),
                ("MM0000200", "partial_year", 4.0, "tokens"),
                ("MM0000300", "partial_year", None, "tokens"),
            ]
        )
    )
    labels = dict(zip(kinds["ori9"], kinds["stage1_adjudicated_kind"], strict=True))
    assert labels["MM0000100"] == STAGE1_ADJUDICATED_LADDER_KIND
    assert labels["MM0000200"] == STAGE1_ADJUDICATED_PARTIAL_KIND
    assert labels["MM0000300"] == STAGE1_ADJUDICATED_LADDER_KIND


def _estimate_rows(rows):
    return pd.DataFrame(
        [
            {
                "ori9": ori,
                "offense": "burglary",
                "reported_count_current": reported,
                "estimated_count": estimated,
                "agency_estimate_source": source,
                "masked_gap_reclassified": False,
                "stage1_adjudicated_kind": kind,
            }
            for ori, reported, estimated, source, kind in rows
        ]
    )


def _current_rows(rows):
    return pd.DataFrame(
        [
            {
                "ori9": ori,
                "offense": "burglary",
                "preferred_source": NIBRS_SOURCE,
                "preferred_months_reported": 12.0,
                "preferred_count": count,
            }
            for ori, count in rows
        ]
    )


def test_the_fill_invariant_exempts_a_row_a_reviewer_ruled_not_credible():
    """The twelve-month header is the thing under review, so it cannot also be the premise."""
    adjudicated = _estimate_rows(
        [("NN0000100", 1.0, 97.5, "hist_median", STAGE1_ADJUDICATED_LADDER_KIND)]
    )
    current = _current_rows([("NN0000100", 1.0)])
    _assert_no_fill_where_the_chosen_lane_reported(adjudicated, current_rows=current)

    unreviewed = _estimate_rows([("NN0000200", 1.0, 97.5, "hist_median", None)])
    with pytest.raises(ValueError, match="preferred lane reported the target year"):
        _assert_no_fill_where_the_chosen_lane_reported(
            unreviewed, current_rows=_current_rows([("NN0000200", 1.0)])
        )
