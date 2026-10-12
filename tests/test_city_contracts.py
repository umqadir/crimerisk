"""Contract runner semantics and the replay tolerance check (crimerisk.city_contracts)."""

from __future__ import annotations

import pandas as pd

from crimerisk import city_contracts as cc


def _cond(field, op, value="", values=(), case_sensitive=False):
    return {"field": field, "op": op, "value": value, "values": list(values), "case_sensitive": case_sensitive}


CONTRACT = {
    "city_key": "testville",
    "mapping": {
        "id_fields": ["case"],
        "date_field": "occurred",
        "date_format": "auto",
        "year_field": "",
        "latitude_field": "lat",
        "longitude_field": "lon",
        "require_coordinates": True,
        "dedupe_on_id": True,
        "offense_rules": [
            {"offense": "murder", "conditions": [_cond("type", "eq", "MURDER")],
             "exclude_conditions": [_cond("desc", "contains", "NEGLIGENT")]},
            {"offense": "larceny", "conditions": [_cond("type", "in", values=["LARCENY", "Theft"], case_sensitive=True)],
             "exclude_conditions": []},
        ],
    },
}

RAW = pd.DataFrame(
    {
        "case": ["1", "2", "2", "3", "4", "5", "6", "7"],
        "occurred": ["2018-03-01", "2019-05-02", "2019-05-02", "2017-12-31", "2020-01-01", "2020-06-01", "2021-01-01", "2021-02-01"],
        "type": ["MURDER", "LARCENY", "LARCENY", "LARCENY", "MURDER", "theft", "Theft", "LARCENY"],
        "desc": ["", "", "", "", "CRIM NEGLIGENT HOMICIDE", "", "", ""],
        "lat": [40.1, 40.2, 40.2, 40.3, 40.4, 40.5, 0.0, None],
        "lon": [-75.1, -75.2, -75.2, -75.3, -75.4, -75.5, 0.0, -75.7],
    }
)


def test_run_contract_applies_rules_filters_and_dedupe():
    out = cc.run_contract(CONTRACT, RAW, year_start=2018, year_end=2024)
    # 3 is outside the years, 4 is excluded as negligent, 5 fails the case-sensitive match,
    # 6 is at (0, 0), 7 has no latitude, the second row of 2 is a duplicate id.
    assert out["incident_id"].tolist() == ["1", "2"]
    assert out["offense"].tolist() == ["murder", "larceny"]
    assert out["year"].tolist() == [2018, 2019]


def test_run_contract_epoch_ms_and_year_field():
    raw = RAW.assign(occurred=pd.to_datetime(RAW["occurred"]).astype("int64") // 10**6)
    out = cc.run_contract(CONTRACT, raw, year_start=2018, year_end=2024)
    assert out["year"].tolist() == [2018, 2019]
    contract = {**CONTRACT, "mapping": {**CONTRACT["mapping"], "year_field": "yr"}}
    out = cc.run_contract(contract, RAW.assign(yr=2020), year_start=2020, year_end=2020)
    assert out["incident_id"].tolist() == ["1", "2", "3"] and set(out["year"]) == {2020}


def _incidents(n_per_cell: dict[tuple[int, str, str], int]) -> pd.DataFrame:
    rows = [
        {"year": y, "offense": o, "block_group_geoid": bg}
        for (y, o, bg), n in n_per_cell.items()
        for _ in range(n)
    ]
    return pd.DataFrame(rows)


def test_replay_check_tolerances():
    oracle = _incidents({(2018, "larceny", "360290001001"): 600, (2018, "larceny", "360290001002"): 400, (2018, "murder", "360290001001"): 3})
    same = oracle.copy()
    _, summary = cc.replay_check(same, oracle)
    assert summary["pass"] and summary["bg_assignment_distance"] == 0.0
    # 1% of 1000 larceny rows and the one-incident floor on murder both pass.
    near = _incidents({(2018, "larceny", "360290001001"): 609, (2018, "larceny", "360290001002"): 400, (2018, "murder", "360290001001"): 4})
    counts, summary = cc.replay_check(near, oracle)
    assert summary["counts_pass"]
    # Moving 20 of 1003 incidents to another block group breaks the 0.5% assignment bound.
    moved = _incidents({(2018, "larceny", "360290001001"): 580, (2018, "larceny", "360290001002"): 420, (2018, "murder", "360290001001"): 3})
    _, summary = cc.replay_check(moved, oracle)
    assert summary["counts_pass"] and not summary["bg_pass"] and not summary["pass"]
    far = _incidents({(2018, "larceny", "360290001001"): 700, (2018, "larceny", "360290001002"): 400, (2018, "murder", "360290001001"): 3})
    counts, summary = cc.replay_check(far, oracle)
    assert not summary["counts_pass"]
    assert counts.loc[counts["offense"].eq("larceny"), "delta"].item() == 100


def test_truth_only_contracts_do_not_enter_stage7(tmp_path):
    class Paths:
        repo_root = tmp_path

    (tmp_path / "configs" / "city_feeds").mkdir(parents=True)
    (tmp_path / "configs" / "city_feeds" / "testville.json").write_text("{}")
    (tmp_path / "configs" / "city_offense_admissions.csv").write_text(
        "city_key,offense,admission_mode\ntestville,larceny,truth_only\nother,larceny,share_only\n"
    )
    assert cc.contract_share_cities(Paths) == []
    (tmp_path / "configs" / "city_offense_admissions.csv").write_text(
        "city_key,offense,admission_mode\ntestville,larceny,share_only\n"
    )
    assert cc.contract_share_cities(Paths) == ["testville"]


def _fake_block_groups(incidents, state_fips, tiger_dir, **_):
    # One block group per point, west of -75.45 only (the rest fall outside the contract's states).
    out = incidents.copy()
    out["block_group_geoid"] = [f"42101000{int(-lon * 100) % 1000:03d}1" if lon < -75.45 else pd.NA for lon in out["longitude"]]
    out["block_group_geoid"] = out["block_group_geoid"].astype("string")
    return out


def test_funnel_counts_every_stage_before_filtering(monkeypatch):
    monkeypatch.setattr(cc, "assign_block_groups", _fake_block_groups)
    contract = {**CONTRACT, "geography": {"state_fips": ["42"], "jurisdiction_ids": ["town"]}}
    raw = pd.concat([RAW, pd.DataFrame({"case": ["8", "9"], "occurred": ["not a date", "2019-01-01"], "type": ["LARCENY", "ARSON"],
                                        "desc": ["", ""], "lat": [40.8, 40.9], "lon": [-75.8, -75.9]})], ignore_index=True)
    crosswalk = pd.DataFrame({"block_group_geoid": ["421010005501", "421010005601", "421010007501"],
                              "jurisdiction_id": ["town", "town", "elsewhere"]})
    table = cc.funnel(contract, raw, year_start=2018, year_end=2024, tiger_dir=None, crosswalk=crosswalk)
    total = table[table["offense"].eq("all")].iloc[0]
    # 10 rows; 4 (negligent), 5 (case-sensitive miss) and 9 (arson) map to nothing; 8 has no date;
    # 3 is 2017; 6 is at (0, 0) and 7 has no latitude; 1 and the two rows of 2 fall outside the states.
    assert [int(total[s]) for s in cc.FUNNEL_STAGES] == [10, 7, 6, 5, 3, 0, 0, 0]
    larceny = table[table["offense"].eq("larceny") & table["year"].eq("all")].iloc[0]
    assert (int(larceny["mapped"]), int(larceny["dated"]), int(larceny["in_years"]), int(larceny["located"])) == (6, 5, 4, 2)
    # Shifted west, 1 and 2 land in the scope's block groups (6 moves off (0, 0) but stays outside
    # the states); the two rows of 2 share an id and count once.
    shifted = raw.assign(lon=raw["lon"] - 0.4)
    table = cc.funnel(contract, shifted, year_start=2018, year_end=2024, tiger_dir=None, crosswalk=crosswalk)
    total = table[table["offense"].eq("all")].iloc[0]
    assert [int(total[s]) for s in cc.FUNNEL_STAGES] == [10, 7, 6, 5, 4, 3, 3, 2]
    yearly = table[table["year"].eq("2019") & table["offense"].eq("larceny")].iloc[0]
    assert (int(yearly["in_years"]), int(yearly["in_footprint"]), int(yearly["deduplicated"])) == (2, 2, 1)
    # run_contract keeps its own filters: the unmapped, undated, out-of-year and unlocated rows go.
    assert cc.run_contract(contract, raw, year_start=2018, year_end=2024)["incident_id"].tolist() == ["1", "2"]


def test_dedup_unit_is_declared_per_contract_and_legacy_contracts_keep_their_key(monkeypatch, capsys):
    import pytest

    # Incident A has two rows that map to larceny (two 23x codes) and one burglary row; B has one
    # larceny row repeated under one victim/offense id.
    raw = pd.DataFrame({"inc": ["A", "A", "A", "B", "B"], "row_id": ["a1", "a2", "a3", "b1", "b1"],
                        "occurred": ["2019-01-01"] * 5, "type": ["23F", "23H", "220", "23F", "23F"],
                        "lat": [40.0] * 5, "lon": [-75.6] * 5})
    rules = [{"offense": "larceny", "conditions": [_cond("type", "in", values=["23F", "23H"])], "exclude_conditions": []},
             {"offense": "burglary", "conditions": [_cond("type", "eq", "220")], "exclude_conditions": []}]
    base = {k: v for k, v in CONTRACT["mapping"].items() if k != "dedupe_on_id"} | {"offense_rules": rules}

    def run(id_fields, **extra):
        contract = {**CONTRACT, "mapping": {**base, "id_fields": id_fields, **extra}}
        return cc.run_contract(contract, raw, year_start=2018, year_end=2024).groupby("offense").size().to_dict()

    assert run(["inc"], dedup_unit="row") == {"larceny": 4, "burglary": 1}
    assert run(["inc"], dedup_unit="incident_offense") == {"larceny": 2, "burglary": 1}
    assert run(["row_id"], dedup_unit="victim_offense") == {"larceny": 3, "burglary": 1}
    # Legacy: no dedup_unit, dedupe_on_id absent (true): one row per raw key, its first offense.
    assert cc.dedup_unit({"mapping": base}) == ("raw_id", False)
    assert run(["inc"]) == {"larceny": 2}
    assert cc.dedup_unit({"mapping": {**base, "dedupe_on_id": False}}) == ("row", False)
    assert cc.dedup_unit({"mapping": {**base, "dedup_unit": "incident_offense"}}) == ("incident_offense", True)
    with pytest.raises(ValueError, match="not both"):
        cc.dedup_unit({"mapping": {**base, "dedup_unit": "row", "dedupe_on_id": False}})
    with pytest.raises(ValueError, match="raw_id"):
        cc.dedup_unit({"mapping": {**base, "dedup_unit": "raw_id"}})
    # The funnel's deduplicated stage counts the same unit.
    monkeypatch.setattr(cc, "assign_block_groups", _fake_block_groups)
    contract = {**CONTRACT, "mapping": {**base, "id_fields": ["inc"], "dedup_unit": "incident_offense"},
                "geography": {"state_fips": ["42"], "jurisdiction_ids": ["town"]}}
    table = cc.funnel(contract, raw, year_start=2018, year_end=2024, tiger_dir=None,
                      crosswalk=_fake_block_groups(raw.rename(columns={"lat": "latitude", "lon": "longitude"}), None, None)
                      .assign(jurisdiction_id="town")[["block_group_geoid", "jurisdiction_id"]])
    total = table[table["offense"].eq("all")].iloc[0]
    assert (int(total["in_footprint"]), int(total["deduplicated"])) == (5, 3)


def test_contract_stage7_quarantine_and_reconciliation(tmp_path, monkeypatch):
    from crimerisk import city_feed_quarantine as q

    contract = {**CONTRACT, "city_name": "Testville", "sources": [{"name": "feed", "source_url": "u", "slug": "t"}],
                "geography": {"state_fips": ["42"], "jurisdiction_ids": ["town"]}}
    raw = RAW.assign(lon=RAW["lon"] - 0.4)
    monkeypatch.setattr(cc, "load_contract", lambda path: contract)
    monkeypatch.setattr(cc, "download", lambda *a, **k: raw)
    monkeypatch.setattr(cc, "assign_block_groups", _fake_block_groups)
    # Rows 1 and 2 land in 421010005501 / 421010005601 (town); a quarantine row at row 1's point
    # removes it before the block-group join.
    monkeypatch.setattr(q, "load_coordinate_quarantine", lambda *a, **k: pd.DataFrame(
        {"city_key": ["testville"], "offense": ["any"], "lat": ["40.1"], "lon": ["-75.5"], "radius_m": ["30"]}))
    monkeypatch.setattr(q, "CANONICAL_CITY_KEYS", q.CANONICAL_CITY_KEYS | {"testville"})
    xw = tmp_path / "xw.parquet"
    pd.DataFrame({"block_group_geoid": ["421010005501", "421010005601"], "jurisdiction_id": ["town", "town"]}).to_parquet(xw)
    kwargs = dict(contract_path=tmp_path / "testville.json", cache_dir=tmp_path, tiger_dir=tmp_path, bg_crosswalk_parquet=xw,
                  start_year=2018, end_year=2024)
    q.reset_quarantine_application_log()
    surface = cc._contract_share_surface(**kwargs)
    assert surface["incident_count"].sum() == 1 and surface["offense"].tolist() == ["larceny"]
    assert q.quarantine_application_log() == {"testville": q.QUARANTINE_APPLIED}
    recon = cc._contract_reconciliation(**kwargs)
    row = recon[recon["offense"].eq("larceny")].iloc[0]
    assert (row["mapped_offense_count"], row["matched_offense_count"], row["final_share_count"], row["state_fips"]) == (1, 1, 1, "42")
    assert set(recon["offense"]) == {"larceny"}


def test_contract_file_stems_are_canonical_city_keys():
    from crimerisk import city_feed_quarantine as q

    assert q.CONTRACT_CITY_KEYS and all(q.canonical_city_key(key) == key for key in q.CONTRACT_CITY_KEYS)


def test_packet_gates_must_match_share_admissions(tmp_path, monkeypatch):
    import pytest

    from crimerisk import city_shares

    monkeypatch.setattr(city_shares, "require_approved", lambda path: None)

    class Paths:
        repo_root = tmp_path

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "city_offense_admissions.csv").write_text(
        "city_key,offense,admission_mode\nfeedtown,robbery,share_only\nfeedtown,rape,rejected\ntruthtown,robbery,truth_only\n")
    city_shares.check_gates_match_share_admissions(Paths, ["feedtown", "truthtown"], {"feedtown": {"robbery"}, "truthtown": {"murder"}})
    with pytest.raises(RuntimeError, match="feedtown"):
        city_shares.check_gates_match_share_admissions(Paths, ["feedtown"], {"feedtown": {"robbery", "rape"}})
    with pytest.raises(RuntimeError, match="feedtown"):
        city_shares.check_gates_match_share_admissions(Paths, ["feedtown"], {})
