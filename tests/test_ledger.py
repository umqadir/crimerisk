"""Decision-ledger integrity: every governed unit approved, hand edits fail closed, hashes stable."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

from crimerisk import ledger
from crimerisk.level_lane import load_admission_registry
from crimerisk.paths import RepoPaths, get_repo_root

REPO = get_repo_root()
sys.path.insert(0, str(REPO / "scripts" / "agents"))

import common  # noqa: E402
import ledger as ledger_cli  # noqa: E402


def test_every_governed_unit_has_an_approved_row():
    assert ledger.check_all(REPO) == []


def test_ledger_integrity():
    frame = ledger.read_ledger(REPO)
    assert list(frame.columns) == list(ledger.LEDGER_COLUMNS)
    assert ledger.integrity_issues(frame) == []
    effective = ledger.effective_rows(frame)
    approved = effective[effective["human_approval"].isin(ledger.APPROVED)]
    assert not approved.duplicated(["target_config", "target_key"]).any()
    for target in set(approved["target_config"]) - {"code"}:
        assert (REPO / target).exists(), target


def _tmp_repo(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "configs" / "ledger").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "level_lane_admission_registry.csv", tmp_path / "configs")
    monkeypatch.setattr(ledger, "get_repo_root", lambda: tmp_path)
    units = ledger.payload_units(tmp_path / "configs" / "level_lane_admission_registry.csv", tmp_path)
    with open(tmp_path / ledger.LEDGER_RELPATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ledger.LEDGER_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for n, (key, sha) in enumerate(sorted(units.items()), 1):
            row = {c: "" for c in ledger.LEDGER_COLUMNS}
            row.update(decision_id=f"Dtest-legacy-{n:05d}", kind="legacy", target_config="configs/level_lane_admission_registry.csv",
                       target_key=key, payload_sha256=sha, human_approval="legacy", status="active")
            writer.writerow(row)
    return tmp_path


def test_hand_edited_governed_row_fails_the_loader(tmp_path, monkeypatch):
    root = _tmp_repo(tmp_path, monkeypatch)
    paths = RepoPaths.from_repo_root(root)
    assert len(load_admission_registry(paths)) > 0
    target = root / "configs" / "level_lane_admission_registry.csv"
    lines = target.read_text().splitlines(keepends=True)
    lines[1] = lines[1].replace("accept_unchanged", "accept_unchanged_edited", 1)
    target.write_text("".join(lines))
    with pytest.raises(ledger.LedgerError, match="no approved row"):
        load_admission_registry(paths)
    assert ledger.check_all(root)


def test_superseding_approved_row_admits_the_new_payload(tmp_path, monkeypatch):
    root = _tmp_repo(tmp_path, monkeypatch)
    target = root / "configs" / "level_lane_admission_registry.csv"
    lines = target.read_text().splitlines(keepends=True)
    lines[1] = lines[1].replace("accept_unchanged", "accept_unchanged_edited", 1)
    target.write_text("".join(lines))
    missing = ledger.unapproved_units(target, root)
    assert len(missing) == 1
    frame = ledger.read_ledger(root)
    old = frame[frame["target_key"].eq(missing[0])].iloc[0]
    row = old.to_dict()
    row.update(decision_id="Dtest-level_admission-00001", kind="level_admission",
               payload_sha256=ledger.payload_units(target, root)[missing[0]], human_approval="approved",
               approved_by="test", supersedes=old["decision_id"])
    with open(root / ledger.LEDGER_RELPATH, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=list(ledger.LEDGER_COLUMNS), lineterminator="\n").writerow(row)
    assert ledger.unapproved_units(target, root) == []
    frame = ledger.read_ledger(root)
    assert old["decision_id"] not in set(ledger.effective_rows(frame)["decision_id"])


def test_pending_row_does_not_approve(tmp_path, monkeypatch):
    root = _tmp_repo(tmp_path, monkeypatch)
    path = root / ledger.LEDGER_RELPATH
    text = path.read_text().replace(",legacy,,,,,active", ",pending,,,,,active")
    assert ",pending," in text
    path.write_text(text)
    assert ledger.unapproved_units(root / "configs" / "level_lane_admission_registry.csv", root)


def test_payload_hash_ignores_column_and_row_order_only(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "get_repo_root", lambda: tmp_path)
    (tmp_path / "configs").mkdir()
    target = tmp_path / "configs" / "overlap_custom_footprints.csv"
    target.write_text("ori,block_group_geoid,weight_share\nX1,010010201001,0.5\nX1,010010201002,0.5\nX2,010010201003,1\n")
    base = ledger.payload_units(target, tmp_path)
    assert set(base) == {"ori=X1", "ori=X2"}
    target.write_text("weight_share,ori,block_group_geoid\n0.5,X1,010010201002\n1,X2,010010201003\n0.5,X1,010010201001\n")
    assert ledger.payload_units(target, tmp_path) == base
    target.write_text("ori,block_group_geoid,weight_share\nX1,010010201001,0.5\nX1,010010201002,0.50\nX2,010010201003,1\n")
    changed = ledger.payload_units(target, tmp_path)
    assert changed["ori=X1"] != base["ori=X1"] and changed["ori=X2"] == base["ori=X2"]


def test_files_outside_the_checkout_are_not_governed(tmp_path):
    assert ledger.governed_relpath(tmp_path / "configs" / "reporting_regime_overrides.csv", REPO) is None
    assert ledger.governed_relpath(REPO / "configs" / "reporting_regime_overrides.csv", REPO) == "configs/reporting_regime_overrides.csv"
    assert ledger.governed_relpath(REPO / "configs" / "city_feeds" / "tacoma.json", REPO) == "configs/city_feeds/*.json"


def test_recover_provenance_reads_writes_not_outputs(tmp_path):
    row = "AZ0018900,navajo_nation_police,some,long,enough,payload,row"
    claude = tmp_path / "s.jsonl"
    events = [
        {"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-test", "content": [
            {"type": "tool_use", "name": "Write", "input": {"file_path": "x.csv", "content": f"h\n{row}\n"}}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": row}]}},
    ]
    claude.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    writes = list(ledger_cli._jsonl_writes(claude))
    assert [(no, model) for no, model, _, _ in writes] == [(1, "claude-opus-test")]
    assert row in ledger_cli._written_lines(writes[0][3])
    codex = tmp_path / "c.jsonl"
    events = [
        {"type": "turn_context", "payload": {"model": "gpt-test", "effort": "medium"}},
        {"type": "event_msg", "payload": {"type": "item_completed", "item": {
            "type": "CommandExecution", "command": ["/bin/zsh", "-lc", f"cat >> x.csv <<'EOF'\n{row}\nEOF"], "stdout": "ignored"}}},
    ]
    codex.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    writes = list(ledger_cli._jsonl_writes(codex))
    assert [(model, effort) for _, model, effort, _ in writes] == [("codex gpt-test", "medium")]
    assert row in ledger_cli._written_lines(writes[0][3])


def test_upsert_csv_unit_keeps_other_rows(tmp_path):
    target = tmp_path / "a.csv"
    target.write_text("city_key,offense,v\na,x,1\nb,x,2\nb,y,3\n")
    common.upsert_csv_unit(target, ["city_key", "offense"], {"city_key": "b", "offense": "x"}, [{"city_key": "b", "offense": "x", "v": "9"}])
    assert target.read_text() == "city_key,offense,v\na,x,1\nb,x,9\nb,y,3\n"
    common.upsert_csv_unit(target, ["city_key", "offense"], {"city_key": "c", "offense": "x"}, [{"city_key": "c", "offense": "x", "v": "4"}])
    assert target.read_text().endswith("c,x,4\n")


def test_upsert_csv_unit_adds_only_declared_columns(tmp_path):
    target = tmp_path / "a.csv"
    target.write_text("city_key,offense,v\na,x,1\nb,x,2\n")
    with pytest.raises(ValueError):
        common.upsert_csv_unit(target, ["city_key", "offense"], {"city_key": "b", "offense": "x"},
                               [{"city_key": "b", "offense": "x", "v": "9", "w": "5"}])
    common.upsert_csv_unit(target, ["city_key", "offense"], {"city_key": "b", "offense": "x"},
                           [{"city_key": "b", "offense": "x", "v": "9", "w": "5"}], add_columns=["w"])
    assert target.read_text() == "city_key,offense,v,w\na,x,1,\nb,x,9,5\n"


def test_gold_replay_scoring(tmp_path):
    import adjudicate

    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([
        {"case_id": "a", "verdict": "token_reporting_flag (1 month)"},
        {"case_id": "b", "verdict": "misread_missing"},
        {"case_id": "c", "verdict": "unclear_escalate"},
    ]))
    base = {"case_class": "k", "rule_echo": False}
    results = [
        dict(base, case_id="a", draft_verdict="token_report_treat_as_partial", reviewer_own_verdict="token_report_treat_as_partial",
             disposition="token_report_treat_as_partial"),
        dict(base, case_id="b", draft_verdict="true_zero", reviewer_own_verdict="publication_gap_treat_as_missing", disposition="escalate"),
        dict(base, case_id="c", draft_verdict="escalate", reviewer_own_verdict="escalate", disposition="escalate"),
    ]
    summary = adjudicate.score(results, gold)
    assert summary["draft_agree"] == 2 and summary["reviewer_own_agree"] == 3
    assert summary["final_escalated"] == 2 and summary["final_decided"] == 1 and summary["final_decided_agree"] == 1
    assert summary["final_agree_incl_escalations"] == 2
