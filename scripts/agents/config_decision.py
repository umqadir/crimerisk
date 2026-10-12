"""Governed configuration decisions through drafter and reviewer, ending in ledger proposals.

    uv run python scripts/agents/config_decision.py --queue <cases.json> --run <run_id> [--mode session]
    uv run python scripts/agents/config_decision.py ingest --run <run_id>

`cases.json` is a list of cases, each {case_id, case_class, kind (a ledger kind), edition,
units: [{target_config, key_columns, key_values, op ("upsert_csv_unit" | "delete_csv_unit"),
proposed_rows}], question, evidence_text, evidence_files}. One case is one drafter and one
reviewer reading over all of its units; each accepted unit becomes its own proposal. Per case the runner writes the packet
to `state/review/packets/config_unit/<case_id>/packet.json`, runs D (column and key checks) ->
O draft -> R review -> one revise round, and writes `proposal.json` for every case the reviewer
accepts. `scripts/agents/ledger.py approve --run <run_id>` then writes the units and the ledger
rows. A case the reviewer does not accept gets no approvable proposal.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import REPO_ROOT, RUNS_DIR, SKILLS_DIR  # noqa: E402

SKILL = SKILLS_DIR / "adjudicate"
KIND = "config_unit"


def _header(target_config: str) -> list[str]:
    path = REPO_ROOT / target_config
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return next(csv.reader(io.StringIO(lines[0]))) if lines else []


def _check_rows(unit: dict, rows: list[dict]) -> list[str]:
    header = _header(unit["target_config"])
    issues = []
    if unit["op"] == "delete_csv_unit" and rows:
        issues.append("a delete unit carries rows")
    if unit["op"] == "upsert_csv_unit" and not rows:
        issues.append("an upsert unit carries no rows")
    for i, row in enumerate(rows):
        extra = sorted(set(row) - set(header))
        missing = sorted(set(header) - set(row))
        if extra or missing:
            issues.append(f"row {i}: columns extra={extra} missing={missing}")
        for key in unit["key_columns"]:
            if str(row.get(key, "")) != str(unit["key_values"][key]):
                issues.append(f"row {i}: key {key}={row.get(key)!r} is not the unit's {unit['key_values'][key]!r}")
        if any(not isinstance(v, str) for v in row.values()):
            issues.append(f"row {i}: every value must be a string")
    return issues


def _target_key(unit: dict) -> str:
    return ";".join(f"{k}={unit['key_values'][k]}" for k in unit["key_columns"])


def _case_text(case: dict) -> str:
    lines = [f"QUESTION: {case['question']}", ""]
    for index, unit in enumerate(case["units"]):
        header = _header(unit["target_config"])
        lines += [
            f"UNIT {index}: {unit['target_config']} (columns: {', '.join(header)})",
            f"  key: {json.dumps(unit['key_values'])}  operation: {unit['op']}",
            "  proposed rows:",
            json.dumps(unit["proposed_rows"], indent=1),
            "",
        ]
    lines += ["EVIDENCE:", case["evidence_text"]]
    return "\n".join(lines)


def run_case(case: dict, run_dir: Path) -> dict:
    case_id = case["case_id"]
    for index, unit in enumerate(case["units"]):
        issues = _check_rows(unit, unit["proposed_rows"])
        if issues:
            raise SystemExit(f"{case_id} unit {index}: proposed rows fail the D checks: {issues}")
    packet_dir = REPO_ROOT / "state" / "review" / "packets" / KIND / case_id
    # Evidence is copied into the packet so the packet is self-contained and hashable in-repo.
    (packet_dir / "evidence").mkdir(parents=True, exist_ok=True)
    evidence = []
    for source in case.get("evidence_files", []):
        source = Path(source) if Path(source).is_absolute() else REPO_ROOT / source
        target = packet_dir / "evidence" / source.name
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
        evidence.append(target)
    packet_sha = common.write_packet(packet_dir, case_text=_case_text(case), signals={}, evidence=evidence)
    out_dir = run_dir / case_id
    values = {"packet_dir": common.rel(packet_dir), "case_id": case_id, "case_class": case["case_class"]}
    ruling_schema = SKILL / "schemas" / "config_ruling.json"
    review_schema = SKILL / "schemas" / "config_review.json"
    template = SKILL / "prompts" / f"{KIND}.md"
    review_template = SKILL / "prompts" / f"review_{KIND}.md"
    add_dirs = (packet_dir,)

    def render(path: Path, **extra: str) -> str:
        text = path.read_text()
        for key, value in {**values, **extra}.items():
            text = text.replace("{" + key + "}", value)
        return text

    draft_prompt = render(template)
    draft = common.run_agent("drafter", draft_prompt, schema=ruling_schema, out_dir=out_dir, template=template,
                             add_dirs=add_dirs, tag="draft")
    review = common.run_agent("reviewer", render(review_template, draft_json=json.dumps(draft["output"], indent=1)),
                              schema=review_schema, out_dir=out_dir, template=review_template,
                              add_dirs=add_dirs, tag="review")
    final_draft, final_review, rounds = draft, review, 1
    if review["output"]["verdict"] == "revise":
        revise = render(SKILL / "prompts" / "revise.md", draft_json=json.dumps(draft["output"], indent=1),
                        review_json=json.dumps(review["output"], indent=1))
        final_draft = common.run_agent("drafter", draft_prompt + "\n\n" + revise, schema=ruling_schema,
                                       out_dir=out_dir, template=template, add_dirs=add_dirs, tag="draft2")
        final_review = common.run_agent(
            "reviewer", render(review_template, draft_json=json.dumps(final_draft["output"], indent=1)),
            schema=review_schema, out_dir=out_dir, template=review_template, add_dirs=add_dirs, tag="review2")
        rounds = 2
    output = final_draft["output"]
    accepted = final_review["output"]["verdict"] == "accept" and output["verdict"] == "propose"
    drafted = {int(u["unit_index"]): u["rows"] for u in output.get("units", [])} if accepted else {}
    unit_issues = {}
    dropped = set()
    for index, unit in enumerate(case["units"]):
        if accepted:
            rows = drafted.get(index, [])
            if unit["op"] == "upsert_csv_unit" and not rows:
                # The drafter dropped this unit (the evidence does not support it); the
                # reviewer accepted that. Nothing is written for it.
                dropped.add(index)
                continue
            issues = _check_rows(unit, rows)
            if index not in drafted:
                issues.append("the draft returned no entry for this unit")
            if issues:
                unit_issues[index] = issues
    approvable = accepted and not unit_issues
    result = {
        "case_id": case_id, "case_class": case["case_class"], "packet_sha256": packet_sha,
        "draft_verdict": draft["output"]["verdict"], "final_draft_verdict": output["verdict"],
        "review_verdict": final_review["output"]["verdict"], "rounds": rounds,
        "disposition": output["verdict"] if approvable else "escalate", "unit_issues": unit_issues,
        "dropped_units": sorted(dropped),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for index, unit in enumerate(case["units"]):
        proposal = {
            "edition": case.get("edition", common.EDITION),
            "kind": case["kind"],
            "subject": common.subject_from_key(_target_key(unit)),
            "target_config": unit["target_config"],
            "target_key": _target_key(unit),
            "op": unit["op"],
            "key_columns": unit["key_columns"],
            "key_values": unit["key_values"],
            "rows": drafted.get(index, []),
            "packet_path": common.rel(packet_dir),
            "packet_sha256": packet_sha,
            "evidence_refs": ";".join(output.get("evidence_refs", []))[:2000],
            "rule_applied": case.get("rule_applied", ""),
            "checks": "D: columns and unit key; R: reviewer accept",
            "affects_published": "yes",
            "drafter_model": ";".join(final_draft["models"]),
            "drafter_effort": final_draft["effort"],
            "draft_sha256": final_draft["output_sha256"],
            "reviewer_model": ";".join(final_review["models"]),
            "review_verdict": final_review["output"]["verdict"],
            "review_sha256": final_review["output_sha256"],
            "prompt_sha256": final_draft["prompt_sha256"],
            "cli_version": final_draft.get("cli_version", ""),
            "approvable": approvable and index not in dropped,
        }
        if index in dropped:
            proposal["held_for_owner"] = "dropped by the drafter: evidence does not support the unit"
        path = out_dir / f"proposal_{index:02d}.json"
        if not (path.exists() and json.loads(path.read_text()).get("ledger_decision_id")):
            path.write_text(json.dumps(proposal, indent=1, sort_keys=True) + "\n")
    return result


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["ingest"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", required=True)
        run_id = parser.parse_args(argv[1:]).run
        saved = common.load_state(RUNS_DIR / run_id)
        if "argv" not in saved:
            raise SystemExit(f"run {run_id} has no saved arguments")
        argv = saved["argv"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=["headless", "session"], default="headless")
    args = parser.parse_args(argv)
    common.set_mode(args.mode)
    cases = json.loads((args.queue if args.queue.is_absolute() else REPO_ROOT / args.queue).read_text())
    run_dir = RUNS_DIR / args.run
    state = common.load_state(run_dir)
    state["argv"] = argv
    results, pending, failures = [], {}, {}
    for case in cases:
        try:
            results.append(run_case(case, run_dir))
        except common.PendingTask as task:
            pending[case["case_id"]] = str(task)
        except common.AgentError as exc:
            failures[case["case_id"]] = str(exc)
    state["cases"] = {r["case_id"]: r["disposition"] for r in results}
    state["pending_tasks"] = dict(sorted(pending.items()))
    state["failures"] = failures
    common.save_state(run_dir, state)
    if pending:
        print(f"run {args.run}: {len(results)} done, {len(pending)} task(s) waiting for a session subagent, {len(failures)} failed")
        for task in pending.values():
            print(f"  TASK {task}")
        print(f"after the outputs are written: uv run python scripts/agents/config_decision.py ingest --run {args.run}")
        return 2
    lines = [f"# Approval summary: {args.run}", "", "| case | disposition | review | rounds |", "|---|---|---|---|"]
    lines += [f"| {r['case_id']} | {r['disposition']} | {r['review_verdict']} | {r['rounds']} |" for r in sorted(results, key=lambda r: r["case_id"])]
    (run_dir / "approval_summary.md").write_text("\n".join(lines) + "\n")
    print(f"run {args.run}: {len(results)} done, {len(failures)} failed")
    for case_id, err in failures.items():
        print(f"  FAILED {case_id}: {err[:300]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
