"""New-source onboarding: QA on the pre-filter funnel, and the seeded stratified evaluation seal."""

from __future__ import annotations

import sys

import pandas as pd

from crimerisk.paths import get_repo_root

sys.path.insert(0, str(get_repo_root() / "scripts" / "agents"))

import onboard_feed  # noqa: E402


def test_qa_uses_the_funnel_and_reconciles_the_footprint(tmp_path, monkeypatch):
    xw = tmp_path / "xw.parquet"
    pd.DataFrame({"block_group_geoid": ["010010001001"], "jurisdiction_id": ["town"]}).to_parquet(xw)
    monkeypatch.setattr(onboard_feed, "CROSSWALK", xw)
    # 10 located larceny incidents in 2019; the funnel says 40 in-year rows existed before the
    # coordinate filter, 10 landed in a block group and 9 are in the footprint after de-duplication.
    incidents = pd.DataFrame({"year": [2019] * 10, "offense": ["larceny"] * 10, "latitude": [33.0 + i / 100 for i in range(10)],
                              "longitude": [-86.0] * 10, "block_group_geoid": ["010010001001"] * 10})
    funnel = pd.DataFrame([{"offense": "larceny", "year": "2019", "in_years": 40, "located": 12, "in_state": 10, "in_footprint": 10,
                            "deduplicated": 9}])
    official = pd.DataFrame({"year": [2019], "offense": ["larceny"], "official_total": [30.0], "official_source": ["srs"]})
    table, exceptions = onboard_feed.qa("town_x", None, incidents, "town", funnel_table=funnel, official=official, first_year=2019)
    row = table.iloc[0]
    assert (row["mapped_rows"], row["located_rows"], row["footprint_rows"]) == (40, 10, 9)
    assert row["match_share"] == 0.25
    assert abs(row["reconciliation_relative_delta"] - 0.7) < 1e-12
    checks = set(zip(exceptions["offense"], exceptions["check"]))
    assert ("larceny", "match_share") in checks and ("larceny", "reconciliation") in checks
    # Every other offense has no located row in 2019.
    assert {o for o, c in checks if c == "year_coverage"} == set(onboard_feed.cc.OFFENSES_7) - {"larceny"}


def test_seal_draw_is_seeded_stratified_and_one_third():
    cities = [{"city_key": f"c{i:02d}", "state_fips": fips, "population": pop}
              for i, (fips, pop) in enumerate([("06", 300000), ("06", 90000), ("04", 120000), ("53", 50000), ("37", 500000),
                                               ("37", 60000), ("13", 150000), ("48", 80000), ("17", 200000), ("39", 70000),
                                               ("42", 300000), ("36", 40000)])]
    sealed = onboard_feed.seal_draw(cities, 20260927)
    assert sealed == onboard_feed.seal_draw(list(reversed(cities)), 20260927)
    assert len(sealed) == 4
    regions = {onboard_feed.CENSUS_REGION[c["state_fips"]] for c in cities if c["city_key"] in sealed}
    assert len(regions) >= 3
    assert sealed != onboard_feed.seal_draw(cities, 1)


def test_official_comparator_is_an_admitted_count(tmp_path, monkeypatch):
    from crimerisk.state_publications import FDLE_PUBLICATION_RAW_DATA_SOURCE

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "level_lane_hard_evidence.csv").write_text(
        "case_id,ori9,year,offense,level1_status,level2_status,reason_code,repair_mode,replacement_count,external_check_status,evidence_source\n"
        "xx-murder,FL0000001,2025,murder,valid_complete_year,channel_omitted,undercount,source_supported_replacement,12,contradiction_confirmed,test\n"
        "xx-mvt,FL0000001,2025,motor_vehicle_theft,valid_complete_year,definitionally_complete,checked,none,,consistent,test\n")
    (configs / "reporting_regime_restrictions.csv").write_text(
        "restriction_id,ori9,offense,effective_from_year,lifted_from_year,level1_status,level2_status,reason_code,repair_mode,"
        "external_check_status,evidence_source\n"
        "xx-larceny,FL0000001,larceny,2024,,coverage_defective,semantically_unusable,incomplete_vector,decayed_own_history,"
        "contradiction_confirmed,test\n")
    (configs / "level_lane_admission_registry.csv").write_text(
        "case_id,ori9,year,offense,adjudication,reason_code,reviewer,evidence_note\n"
        "xx-robbery,FL0000001,2025,robbery,refuse_repair,implausible,test,test\n")

    class Paths:
        repo_root = tmp_path

    def row(offense, source, raw, count, weight=1.0):
        return {"ori9": "FL0000001", "year": 2025, "offense": offense, "source": source, "raw_data_source": raw, "count": count,
                "observation_weight": weight}

    srs = ("srs_return_a_annual", "kaplan_openicpsr_srs_return_a")
    fdle = ("state_publication_annual", FDLE_PUBLICATION_RAW_DATA_SOURCE)
    obs = pd.DataFrame([
        row("rape", *fdle, 900), row("rape", *srs, 50),             # FDLE rape is a definition mismatch
        row("larceny", *fdle, 5000), row("larceny", *srs, 4000),    # restricted from 2024: no comparator
        row("murder", *srs, 30),                                    # replaced by the hard-evidence count
        row("robbery", *srs, 70),                                   # refused by the registry
        row("motor_vehicle_theft", *srs, 80),                       # accepted unchanged
        row("burglary", *fdle, 45), row("burglary", *srs, 40),      # no ruling: weight, then the higher count
    ])
    path = tmp_path / "agency.parquet"
    obs.to_parquet(path)
    monkeypatch.setattr(onboard_feed, "AGENCY_OBSERVATIONS", path)
    out = onboard_feed.official_totals({"ori9": "FL0000001"}, [], paths=Paths).set_index("offense")
    assert out["official_total"].to_dict() == {"burglary": 45.0, "motor_vehicle_theft": 80.0, "murder": 12.0, "rape": 50.0}
    assert out.loc["murder", "official_source"] == "level_lane_replacement:xx-murder"
    assert out.loc["rape", "official_source"] == "srs_return_a_annual"
