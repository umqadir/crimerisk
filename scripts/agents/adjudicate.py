"""Adjudicate a queue of cases through drafter and reviewer (design section 6.5).

    uv run python scripts/agents/adjudicate.py --queue <csv> --kind agency_record --run <run_id> [--jobs N]
    uv run python scripts/agents/adjudicate.py ... --dry-run
    uv run python scripts/agents/adjudicate.py ... --gold <gold.json>     # score the run against a gold set
    uv run python scripts/agents/adjudicate.py ... --mode session         # write task.md files instead of calling claude -p
    uv run python scripts/agents/adjudicate.py ingest --run <run_id>      # session mode: validate outputs, continue

Queue columns: `case_id`, `case_class`, `packet_source` (`<json file>#<case_id>`; the file holds a
list of {case_id, class, evidence}). Per case the runner writes the packet to
`state/review/packets/<kind>/<case_id>/packet.json` and the agent records to
`state/review/runs/<run_id>/<case_id>/`. D rules -> O draft -> R review -> (one revise round) ->
final disposition; anything not `accept` after the revise round is an escalation for the owner.
Runs are resumable: finished agent calls are read back from disk.

Modes: `headless` calls `claude -p --agent <name>` per step. `session` writes each step's task to
`state/review/runs/<run_id>/<case_id>/<step>.task.md`; a Claude Code session runs every task with one
subagent that writes `<step>.output.json`, then `ingest` validates the outputs against the schema and
advances each case to its next step (see .claude/skills/adjudicate/SKILL.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import REPO_ROOT, RUNS_DIR, SKILLS_DIR  # noqa: E402

SKILL = SKILLS_DIR / "adjudicate"
SIGNALS_MARKER = "COMPUTED SIGNALS"

# Packet vocabulary -> gold vocabulary (state/qa/review_tier_trial/gold.json). Fixed before any
# model run; the replay is a measurement, not a tuning target.
GOLD_MAP = {
    "same_agency_dedupe": "same_agency_merge",
    "distinct_agencies_keep_both": "distinct_agencies",
    "superseded_ori": "superseded_ori",
    "true_zero": "genuine_zero_year",
    "publication_gap_treat_as_missing": "misread_missing",
    "partial_year_coverage": "partial_year_coverage",
    "credible_low_crime": "genuine_low_crime",
    "token_report_treat_as_partial": "token_reporting_flag",
    "structural_zero_treat_as_missing": "misread_missing",
    "data_error": "data_error",
    "defunct_drop": "defunct_no_fill",
    "superseded_by": "defunct_no_fill",
    "absorbed_by": "defunct_no_fill",
    "dormant_but_extant": "plausibly_active",
    "escalate": "unclear_escalate",
}


def load_queue(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def packet_for(kind: str, entry: dict) -> tuple[Path, str]:
    source, case_id = entry["packet_source"].split("#", 1)
    cases = json.loads((REPO_ROOT / source).read_text())
    case = next(c for c in cases if c["case_id"] == case_id)
    text = case["evidence"]
    head, sep, tail = text.partition(SIGNALS_MARKER)
    signals = {"label": "precomputed hints; not evidence on their own", "text": (sep + tail).strip()} if sep else {}
    packet_dir = REPO_ROOT / "state" / "review" / "packets" / kind / case_id
    sha = common.write_packet(packet_dir, case_text=head.strip(), signals=signals, evidence=[REPO_ROOT / source])
    return packet_dir, sha


def render(template: str, **values: str) -> str:
    out = (SKILL / "prompts" / template).read_text()
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def run_case(kind: str, run_dir: Path, entry: dict, dry_run: bool) -> dict:
    case_id, case_class = entry["case_id"], entry["case_class"]
    packet_dir, packet_sha = packet_for(kind, entry)
    out_dir = run_dir / case_id
    ruling_schema = SKILL / "schemas" / "ruling.json"
    review_schema = SKILL / "schemas" / "review.json"
    common_values = {"packet_dir": common.rel(packet_dir), "case_id": case_id, "case_class": case_class}
    draft_prompt = render(f"{kind}.md", **common_values)
    if dry_run:
        print(f"--- {case_id} ({case_class}) packet {common.rel(packet_dir)} sha {packet_sha[:12]}\n{draft_prompt}")
        return {"case_id": case_id, "dry_run": True}
    add = (packet_dir,)
    template = SKILL / "prompts" / f"{kind}.md"
    draft = common.run_agent("drafter", draft_prompt, schema=ruling_schema, out_dir=out_dir, template=template, add_dirs=add, tag="draft")
    review_prompt = render(f"review_{kind}.md", **common_values, draft_json=json.dumps(draft["output"], indent=1))
    review = common.run_agent("reviewer", review_prompt, schema=review_schema, out_dir=out_dir,
                              template=SKILL / "prompts" / f"review_{kind}.md", add_dirs=add, tag="review")
    final_draft, final_review, rounds = draft, review, 1
    if review["output"]["verdict"] == "revise":
        revise_prompt = draft_prompt + "\n\n" + render(
            "revise.md", case_id=case_id,
            draft_json=json.dumps(draft["output"], indent=1), review_json=json.dumps(review["output"], indent=1))
        final_draft = common.run_agent("drafter", revise_prompt, schema=ruling_schema, out_dir=out_dir, template=template, add_dirs=add, tag="draft2")
        review2_prompt = render(f"review_{kind}.md", **common_values, draft_json=json.dumps(final_draft["output"], indent=1))
        final_review = common.run_agent("reviewer", review2_prompt, schema=review_schema, out_dir=out_dir,
                                        template=SKILL / "prompts" / f"review_{kind}.md", add_dirs=add, tag="review2")
        rounds = 2
    accepted = final_review["output"]["verdict"] == "accept"
    result = {
        "case_id": case_id,
        "case_class": case_class,
        "packet_sha256": packet_sha,
        "draft_verdict": draft["output"]["verdict"],
        "final_draft_verdict": final_draft["output"]["verdict"],
        "review_verdict": final_review["output"]["verdict"],
        "reviewer_own_verdict": final_review["output"]["own_verdict"],
        "rule_echo": final_review["output"]["rule_echo"],
        "rounds": rounds,
        "disposition": final_draft["output"]["verdict"] if accepted else "escalate",
        "drafter_models": final_draft["models"],
        "reviewer_models": final_review["models"],
        "prompt_sha256": final_draft["prompt_sha256"],
        "cli_version": final_draft["cli_version"],
        "mode": final_draft.get("mode", "headless"),
        "cost_usd": sum((r.get("cost_usd") or 0) for r in (draft, review) + ((final_draft, final_review) if rounds == 2 else ())),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def score(results: list[dict], gold_path: Path) -> dict:
    """Agreement with a gold set on the primary verdict, through GOLD_MAP."""
    gold = {g["case_id"]: g["verdict"].split(" ")[0] for g in json.loads(gold_path.read_text())}
    rows = []
    for r in results:
        if r["case_id"] not in gold:
            continue
        rows.append({
            "case_id": r["case_id"],
            "case_class": r["case_class"],
            "gold": gold[r["case_id"]],
            "draft": GOLD_MAP[r["draft_verdict"]],
            "reviewer_own": GOLD_MAP[r["reviewer_own_verdict"]],
            "final": GOLD_MAP[r["disposition"]],
        })
    n = len(rows)
    decided = [x for x in rows if x["final"] != "unclear_escalate"]
    summary = {
        "cases": n,
        "draft_agree": sum(x["draft"] == x["gold"] for x in rows),
        "reviewer_own_agree": sum(x["reviewer_own"] == x["gold"] for x in rows),
        "final_escalated": n - len(decided),
        "final_decided": len(decided),
        "final_decided_agree": sum(x["final"] == x["gold"] for x in decided),
        "final_agree_incl_escalations": sum(x["final"] == x["gold"] for x in rows),
        "by_class": {},
        "rows": rows,
    }
    for cls in sorted({x["case_class"] for x in rows}):
        sub = [x for x in rows if x["case_class"] == cls]
        dec = [x for x in sub if x["final"] != "unclear_escalate"]
        summary["by_class"][cls] = {
            "cases": len(sub),
            "draft_agree": sum(x["draft"] == x["gold"] for x in sub),
            "final_decided": len(dec),
            "final_decided_agree": sum(x["final"] == x["gold"] for x in dec),
        }
    return summary


def approval_summary(run_id: str, results: list[dict]) -> str:
    lines = [f"# Approval summary: {run_id}", "", f"Cases: {len(results)}", ""]
    lines += ["| disposition | cases |", "|---|---|"]
    for verdict, n in Counter(r["disposition"] for r in results).most_common():
        lines.append(f"| {verdict} | {n} |")
    esc = [r for r in results if r["disposition"] == "escalate"]
    lines += ["", f"## Escalations ({len(esc)})", ""]
    lines += [f"- {r['case_id']}: draft {r['final_draft_verdict']}, review {r['review_verdict']}, reviewer own {r['reviewer_own_verdict']}" for r in esc]
    accepted = sorted((r for r in results if r["disposition"] != "escalate"), key=lambda r: r["case_id"])
    sample = random.Random(20260924).sample(accepted, min(20, len(accepted)))
    lines += ["", "## Seeded sample of accepted cases to read in full (seed 20260924)", ""]
    lines += [f"- {r['case_id']}: {r['disposition']} (state/review/runs/{run_id}/{r['case_id']}/)" for r in sorted(sample, key=lambda r: r["case_id"])]
    return "\n".join(lines) + "\n"


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["ingest"]:
        ingest = argparse.ArgumentParser()
        ingest.add_argument("--run", required=True)
        run_id = ingest.parse_args(argv[1:]).run
        saved = common.load_state(RUNS_DIR / run_id)
        if "argv" not in saved:
            raise SystemExit(f"run {run_id} has no saved arguments")
        argv = saved["argv"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=["agency_record"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=["headless", "session"], default="headless")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gold", type=Path)
    args = parser.parse_args(argv)
    common.set_mode(args.mode)
    queue = load_queue(args.queue)
    run_dir = RUNS_DIR / args.run
    jobs = max(1, min(args.jobs, 4))
    state = common.load_state(run_dir)
    if not args.dry_run:
        state["argv"] = [a for a in argv if a != "--dry-run"]
    results: list[dict] = []
    failures: dict[str, str] = {}

    pending: dict[str, str] = {}

    def one(entry: dict) -> None:
        try:
            results.append(run_case(args.kind, run_dir, entry, args.dry_run))
        except common.PendingTask as task:
            pending[entry["case_id"]] = str(task)
        except common.AgentError as exc:
            failures[entry["case_id"]] = str(exc)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, queue))
    if args.dry_run:
        return 0
    state["cases"] = {r["case_id"]: r["disposition"] for r in results}
    state["failures"] = failures
    state["pending_tasks"] = dict(sorted(pending.items()))
    common.save_state(run_dir, state)
    if pending:
        print(f"run {args.run}: {len(results)} done, {len(pending)} task(s) waiting for a session subagent, {len(failures)} failed")
        for case_id, task in sorted(pending.items()):
            print(f"  TASK {task}")
        for case_id, err in sorted(failures.items()):
            print(f"  FAILED {case_id}: {err[:300]}")
        print(f"after the outputs are written: uv run python scripts/agents/adjudicate.py ingest --run {args.run}")
        return 2
    results.sort(key=lambda r: r["case_id"])
    (run_dir / "approval_summary.md").write_text(approval_summary(args.run, results))
    print(f"run {args.run}: {len(results)} done, {len(failures)} failed")
    for case_id, err in sorted(failures.items()):
        print(f"  FAILED {case_id}: {err[:200]}")
    if args.gold:
        summary = score(results, args.gold)
        (run_dir / "gold_score.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    print(f"pending for the owner: uv run python scripts/agents/ledger.py summary --run {args.run}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
