"""Ledger I/O: status, summary, approve, reject, check, backfill, recover-provenance.

    uv run python scripts/agents/ledger.py status
    uv run python scripts/agents/ledger.py summary --run <run_id>
    uv run python scripts/agents/ledger.py approve --run <run_id> --approved-by <who> [--cases a,b] [--note ...]
    uv run python scripts/agents/ledger.py reject  --run <run_id> --approved-by <who> --cases a,b --note ...
    uv run python scripts/agents/ledger.py check
    uv run python scripts/agents/ledger.py backfill
    uv run python scripts/agents/ledger.py recover-provenance --logs <dir> [--logs <dir> ...]

`approve` is the only writer of a governed config: it writes the approved payload units of a run
and appends their ledger rows, then prints the commit command. A run's approvable units are the
`proposal.json` files under `state/review/runs/<run_id>/<case_id>/`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import REPO_ROOT, RUNS_DIR, core  # noqa: E402

LEGACY_EDITION = "2025.1"
OFFLINE = "unknown (logs offline)"
LEGACY_NOTE = "legacy backfill: payload as shipped in 2025.1; provenance per inventory section 3"


# ---------------------------------------------------------------- status / check


def cmd_status(_args) -> int:
    ledger = common.read_ledger()
    eff = core.effective_rows(ledger)
    print(f"ledger rows: {len(ledger)} (effective {len(eff)})")
    for column in ("kind", "human_approval", "edition"):
        print(f"  by {column}: {dict(Counter(eff[column]))}")
    pending = []
    for proposal in sorted(RUNS_DIR.glob("*/*/proposal*.json")):
        p = json.loads(proposal.read_text())
        if not p.get("ledger_decision_id"):
            pending.append(proposal.parent)
    print(f"pending proposals: {len(pending)}")
    for run in sorted({p.parent.name for p in pending}):
        n = sum(1 for p in pending if p.parent.name == run)
        print(f"  {run}: {n}  (uv run python scripts/agents/ledger.py summary --run {run})")
    return 0


def cmd_check(_args) -> int:
    issues = core.check_all(REPO_ROOT)
    for issue in issues:
        print(issue)
    print("ledger check:", "FAIL" if issues else "pass")
    return 1 if issues else 0


# ---------------------------------------------------------------- summary / approve / reject


def _proposals(run_id: str, cases: list[str] | None) -> list[tuple[Path, dict]]:
    run_dir = RUNS_DIR / run_id
    out = []
    for path in sorted(run_dir.glob("*/proposal*.json")):
        if cases and path.parent.name not in cases:
            continue
        out.append((path, json.loads(path.read_text())))
    if not out:
        raise SystemExit(f"no proposals under {run_dir}")
    return out


def cmd_summary(args) -> int:
    path = RUNS_DIR / args.run / "approval_summary.md"
    if not path.exists():
        raise SystemExit(f"{path} does not exist; the runner writes it at the end of a run")
    print(path.read_text())
    return 0


def _apply(proposal: dict) -> None:
    target = REPO_ROOT / proposal["target_config"]
    op = proposal["op"]
    if op == "write_json":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(proposal["payload"], indent=2, sort_keys=False) + "\n", encoding="utf-8")
    elif op == "upsert_csv_unit":
        common.upsert_csv_unit(target, proposal["key_columns"], proposal["key_values"], proposal["rows"],
                               add_columns=proposal.get("add_columns"))
    elif op == "delete_csv_unit":
        common.upsert_csv_unit(target, proposal["key_columns"], proposal["key_values"], [])
    else:
        raise ValueError(f"unknown op {op}")


def _ledger_row(proposal: dict, decision_id: str, *, approval: str, approved_by: str, note: str, payload_sha: str) -> dict:
    fields = {c: proposal.get(c, "") for c in core.LEDGER_COLUMNS}
    fields.update(
        decision_id=decision_id,
        payload_sha256=payload_sha,
        human_approval=approval,
        approved_by=approved_by,
        approved_at=common.now_iso(),
        approval_note=note,
        status="active",
    )
    return common.blank_row(**fields)


def _record(args, approval: str) -> int:
    proposals = _proposals(args.run, args.cases.split(",") if args.cases else None)
    todo = [(p, d) for p, d in proposals if not d.get("ledger_decision_id")]
    if approval == "approved":
        todo = [(p, d) for p, d in todo if d.get("approvable", True)]
    rows = []
    by_kind: dict[str, list] = defaultdict(list)
    for path, proposal in todo:
        by_kind[proposal["kind"]].append((path, proposal))
    for kind, items in by_kind.items():
        ids = common.next_decision_ids(kind, items[0][1].get("edition", common.EDITION), len(items))
        for (path, proposal), decision_id in zip(items, ids):
            effective = core.effective_rows(common.read_ledger())
            effective = effective[effective["human_approval"].isin(core.APPROVED)]  # the payload state being replaced
            prior = effective[
                effective["target_config"].eq(proposal["target_config"])
                & effective["target_key"].eq(proposal["target_key"])
            ]["decision_id"].tolist()
            if approval == "approved":
                _apply(proposal)
                units = core.payload_units(REPO_ROOT / proposal["target_config"], REPO_ROOT)
                sha = core.REMOVED_SHA256 if proposal["op"] == "delete_csv_unit" else units[proposal["target_key"]]
            else:
                sha = proposal.get("payload_sha256", "")
            row = _ledger_row(proposal, decision_id, approval=approval, approved_by=args.approved_by, note=args.note, payload_sha=sha)
            if approval == "approved" and prior:
                row["supersedes"] = prior[-1]
            common.append_ledger_rows([row])
            proposal["ledger_decision_id"] = decision_id
            path.write_text(json.dumps(proposal, indent=1, sort_keys=True) + "\n")
            rows.append(row)
    print(f"{approval}: {len(rows)} unit(s) from run {args.run}")
    if approval == "approved" and rows:
        targets = sorted({r["target_config"] for r in rows})
        print("commit with:")
        print(f"  git add {core.LEDGER_RELPATH} " + " ".join(t for t in targets if not t.startswith("state/")))
        print(f"  git commit -m 'Approve run {args.run}: {len(rows)} ledger rows'")
    return 0


def cmd_approve(args) -> int:
    return _record(args, "approved")


def cmd_reject(args) -> int:
    return _record(args, "rejected")


# ---------------------------------------------------------------- backfill


def _stage1_arms() -> tuple[dict[str, dict[str, str]], str]:
    path = REPO_ROOT / "state/qa/stage1_adhoc_review/rulings_full.csv"
    import pandas as pd

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    active = frame[frame["active"].str.lower().isin({"true", "1", "yes"})]
    return {r["case_id"]: r for r in active.to_dict("records")}, common.local_ref(path)


ARM_MODEL = {
    "opus": ("claude-opus (subagent; exact model id not recorded)", ""),
    "sol": ("codex gpt-5.6-sol", ""),
    "luna": ("codex gpt-5.6-luna", ""),
    "gold": ("gold pilot ruling (adjudicator not recorded)", ""),
}


def _compiled_keys(pattern: str, columns: list[str]) -> dict[str, tuple[set, str]]:
    import pandas as pd

    out = {}
    for path in sorted(REPO_ROOT.glob(pattern)):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        keys = set(map(tuple, frame[columns].astype(str).values.tolist()))
        out[path.parts[-3]] = (keys, common.local_ref(path))
    return out


def _legacy_provenance(rel: str, units: dict[str, str]) -> dict[str, dict[str, str]]:
    """Per target_key: ledger fields evidenced by inventory section 3. Missing fields stay offline-unknown."""
    import pandas as pd

    prov: dict[str, dict[str, str]] = {}
    if rel.startswith("configs/stage1_adjudications/") and not rel.endswith("ori_successions.csv"):
        arms, ref = _stage1_arms()
        for key in units:
            case_id = key.split("=", 1)[1]
            ruling = arms.get(case_id)
            if ruling:
                model, _ = ARM_MODEL.get(ruling["arm"], (OFFLINE, ""))
                prov[key] = {"drafter_model": model, "evidence_refs": ref,
                             "approval_note": f"{LEGACY_NOTE}; rulings_full arm={ruling['arm']} source={ruling['source']} agent={ruling['agent'] or 'n/a'}"}
    elif rel == "configs/level_lane_admission_registry.csv":
        frame = pd.read_csv(REPO_ROOT / rel, dtype=str, keep_default_na=False)
        for r in frame.to_dict("records"):
            prov[f"case_id={r['case_id']}"] = {"reviewer_model": f"{r['reviewer']} (agent identity not recorded)"}
    elif rel == "configs/level_lane_structural_zero_screen.csv":
        ref = "analysis_scratch/level_lane_screen/structural_zero_corroborated_all.csv@5228ef6b3b473461237ad7e0e78c77720daee99fd4aa1dd1e2897c90aec82410"
        for key in units:
            prov[key] = {
                "drafter_model": "none (rule)",
                "evidence_refs": ref,
                "rule_applied": "structural_zero_corroborated: 2024 Part I <= 1 with >= 11 months; >= 2 historical "
                "preferred-value years; historical median <= 1 and max <= 3 (level_lane_screen/01_build_level_lane_screen.py)",
                "approval_note": f"{LEGACY_NOTE}; moved from the untracked analysis_scratch screen (ori9 and rule columns)",
            }
    elif rel == "configs/reporting_regime_overrides.csv":
        runs = _compiled_keys("state/review/analysis/reporting/*/compiled/reporting_regime_accepted_overrides.parquet", ["ori", "year", "offense"])
        for key in units:
            parts = dict(p.split("=", 1) for p in key.split(";"))
            k = (parts["ori"], parts["year"], parts["offense"])
            hits = [ref for keys, ref in runs.values() if k in keys]
            if hits:
                prov[key] = {"drafter_model": "codex gpt-5.4-mini", "drafter_effort": "medium", "evidence_refs": ";".join(hits)}
    elif rel == "configs/overlap_footprint_overrides.csv":
        runs = _compiled_keys("state/review/analysis/overlap/*/compiled/overlap_localization_accepted_supported.parquet", ["ori"])
        for key in units:
            k = (key.split("=", 1)[1],)
            mini = [ref for run, (keys, ref) in runs.items() if k in keys and run.endswith("_mini")]
            big = [ref for run, (keys, ref) in runs.items() if k in keys and run.endswith("_gpt54")]
            if mini:
                prov[key] = {"drafter_model": "codex gpt-5.4-mini", "drafter_effort": "medium", "evidence_refs": ";".join(mini + big)}
                if big:
                    prov[key]["reviewer_model"] = "codex gpt-5.4 (verify run)"
            elif big:
                prov[key] = {"drafter_model": "codex gpt-5.4", "drafter_effort": "high", "evidence_refs": ";".join(big)}
    elif rel.endswith("local_queue_resolved_final.parquet"):
        frame = pd.read_parquet(REPO_ROOT / rel, columns=["ori9", "resolution_source"])
        ref = common.local_ref(REPO_ROOT / rel)
        for ori, source in frame.itertuples(index=False):
            key = f"ori9={ori}"
            if source in {"first_pass", "first_pass_retained"}:
                prov[key] = {"drafter_model": "codex gpt-5.4-mini", "drafter_effort": "medium", "evidence_refs": ref}
            elif source == "second_pass":
                prov[key] = {"drafter_model": "codex gpt-5.4", "drafter_effort": "high", "evidence_refs": ref}
            else:
                prov[key] = {"drafter_model": "none (rule)", "rule_applied": f"resolution_source={source}", "evidence_refs": ref}
    elif rel.endswith("nonmunicipal_special_resolved_final.parquet"):
        frame = pd.read_parquet(REPO_ROOT / rel, columns=["ori9", "resolution_source", "baseline_rule_name"])
        ref = common.local_ref(REPO_ROOT / rel)
        for ori, source, rule in frame.itertuples(index=False):
            key = f"ori9={ori}"
            if source == "deterministic_baseline":
                prov[key] = {"drafter_model": "none (rule)", "rule_applied": f"deterministic_baseline:{rule}", "evidence_refs": ref}
            elif source == "mini_second_pass":
                prov[key] = {"drafter_model": "codex gpt-5.4-mini", "drafter_effort": "medium", "evidence_refs": ref}
            else:
                prov[key] = {"drafter_model": "unknown (high-review model not in logs)", "evidence_refs": ref}
    return prov


def cmd_backfill(_args) -> int:
    ledger = common.read_ledger()
    covered = set(zip(ledger["target_config"], ledger["target_key"]))
    rows = []
    for path in core.governed_files(REPO_ROOT):
        rel = common.rel(path)
        units = core.payload_units(path, REPO_ROOT)
        prov = _legacy_provenance(rel, units)
        for key, sha in units.items():
            if (rel, key) in covered:
                continue
            fields = dict(
                kind="legacy", subject=common.subject_from_key(key), target_config=rel, target_key=key,
                payload_sha256=sha, edition=LEGACY_EDITION, drafter_model=OFFLINE, reviewer_model=OFFLINE,
                human_approval="legacy", approved_by=OFFLINE, approval_note=LEGACY_NOTE,
            )
            fields.update(prov.get(key, {}))
            rows.append(fields)
    ids = common.next_decision_ids("legacy", LEGACY_EDITION, len(rows))
    out = [common.blank_row(decision_id=i, **f) for i, f in zip(ids, rows)]
    common.append_ledger_rows(out)
    counts = Counter(r["target_config"] for r in out)
    evidenced = Counter(r["target_config"] for r in out if not r["drafter_model"].startswith("unknown") or not r["reviewer_model"].startswith("unknown"))
    print(f"backfill: {len(out)} legacy rows")
    for target in sorted(counts):
        print(f"  {target}: {counts[target]} rows, {evidenced[target]} with evidenced model or rule")
    return 0


# ---------------------------------------------------------------- recover-provenance

WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"}
MIN_LINE = 24


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _written_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.rstrip("\r")
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
        out.append(line)
    return out


def _jsonl_writes(path: Path):
    """Yield (line_no, model, effort, text) for every write-type tool input in a Claude or Codex JSONL log.

    Claude: assistant `tool_use` inputs of Write/Edit/MultiEdit/NotebookEdit/Bash, with the message's
    model. Codex: `item_completed` CommandExecution commands and FileChange contents, with the model
    and effort of the latest `turn_context`. Tool outputs are never read as writes.
    """
    model = effort = ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for no, raw in enumerate(fh, 1):
            if '"tool_use"' not in raw and '"CommandExecution"' not in raw and '"FileChange"' not in raw \
                    and '"turn_context"' not in raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            if event.get("type") == "turn_context" and payload.get("model"):
                model, effort = f"codex {payload['model']}", str(payload.get("effort") or "")
                continue
            message = event.get("message") if isinstance(event.get("message"), dict) else None
            if message and message.get("role") == "assistant":
                for item in message.get("content") or []:
                    if isinstance(item, dict) and item.get("type") == "tool_use" and item.get("name") in WRITE_TOOLS:
                        yield no, message.get("model", ""), "", "\n".join(_strings(item.get("input")))
            elif payload.get("type") == "item_completed":
                item = payload.get("item") or {}
                if item.get("type") == "CommandExecution":
                    yield no, model, effort, "\n".join(_strings(item.get("command")))
                elif item.get("type") == "FileChange":
                    yield no, model, effort, "\n".join(_strings(item.get("changes")))


def _text_log_writes(path: Path):
    """Codex CLI text logs: `model:` header, `apply patch` sections carry the written lines."""
    model = ""
    section = None
    buf: list[str] = []
    start = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for no, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if line.startswith("model: ") and not model:
                model = f"codex {line[7:].strip()}"
            if line in {"apply patch", "exec", "codex", "user", "thinking"} or line.startswith("tokens used"):
                if section == "apply patch" and buf:
                    yield start, model, "", "\n".join(buf)
                section, buf, start = line, [], no
                continue
            if section == "apply patch":
                buf.append(line)
    if section == "apply patch" and buf:
        yield start, model, "", "\n".join(buf)


def _raw_unit_lines(path: Path, key_columns: tuple[str, ...]) -> dict[str, list[str]]:
    """target_key -> the verbatim CSV lines of that unit as they stand in the file."""
    import csv

    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    header = next(csv.reader([lines[0]]))
    idx = [header.index(c) for c in key_columns]
    out: dict[str, list[str]] = defaultdict(list)
    for raw in lines[1:]:
        if not raw.strip():
            continue
        row = next(csv.reader([raw]))
        if len(row) != len(header):
            continue  # a quoted multi-line row; no verbatim single-line match is possible
        key = ";".join(f"{c}={row[i]}" for c, i in zip(key_columns, idx))
        out[key].append(raw)
    return out


def cmd_recover_provenance(args) -> int:
    ledger = common.read_ledger()
    effective = core.effective_rows(ledger)
    open_model = effective["drafter_model"].str.startswith("unknown") | effective["drafter_model"].str.contains("not recorded")
    legacy = effective[effective["human_approval"].eq("legacy") & open_model]
    targets: dict[str, list[tuple[str, str]]] = defaultdict(list)  # verbatim line -> [(config, key)]
    unit_lines: dict[tuple[str, str], list[str]] = {}
    for config in sorted(set(legacy["target_config"])):
        spec = core.GOVERNED.get(config)
        if spec is None or spec.kind != "csv":
            continue
        keys = set(legacy.loc[legacy["target_config"].eq(config), "target_key"])
        for key, raws in _raw_unit_lines(REPO_ROOT / config, spec.key_columns).items():
            if key in keys:
                unit_lines[(config, key)] = raws
                for raw in raws:
                    if len(raw) >= MIN_LINE:
                        targets[raw].append((config, key))
    # stage-1 subagent ids: an agent transcript named agent-<id>.jsonl gives that subagent's model
    arms, _ = _stage1_arms()
    agent_ids = {}
    for config, key in zip(legacy["target_config"], legacy["target_key"]):
        if config.startswith("configs/stage1_adjudications/") and key.startswith("case_id="):
            ruling = arms.get(key.split("=", 1)[1])
            if ruling and ruling.get("agent"):
                agent_ids.setdefault(ruling["agent"], []).append((config, key))

    seen: dict[str, dict[tuple[str, str], set]] = defaultdict(lambda: defaultdict(set))  # raw -> unit -> {(model, ev)}
    agent_hits: dict[tuple[str, str], set] = defaultdict(set)
    scanned = Counter()
    files: list[Path] = []
    for d in args.logs:
        d = Path(d).expanduser()
        if d.is_file():
            files.append(d)
        else:
            files.extend(sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in {".jsonl", ".log"}))
    home = str(Path.home())

    def shown(path: Path) -> str:
        text = str(path)
        if text.startswith(home + "/"):
            return "~" + text[len(home):]
        try:
            return common.rel(path)
        except ValueError:  # logs on an external drive
            return text

    for path in files:
        scanned[path.suffix] += 1
        if path.suffix == ".jsonl" and path.stem.startswith("agent-") and path.stem[6:] in agent_ids:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for no, raw in enumerate(fh, 1):
                    m = re.search(r'"model":"(claude-[^"]+)"', raw)
                    if m:
                        for unit in agent_ids[path.stem[6:]]:
                            agent_hits[unit].add((f"{m.group(1)}|", f"{shown(path)}:{no}"))
                        break
        writes = _jsonl_writes(path) if path.suffix == ".jsonl" else _text_log_writes(path)
        for no, model, effort, text in writes:
            for line in _written_lines(text):
                for unit in targets.get(line, ()):
                    seen[line][unit].add((f"{model or 'unknown model'}|{effort}", f"{shown(path)}:{no}"))

    recovered: dict[tuple[str, str], tuple[str, str]] = {}
    for unit, raws in unit_lines.items():
        models, evidence = set(), []
        complete = True
        for raw in raws:
            hits = seen.get(raw, {}).get(unit)
            if not hits:
                complete = False
                break
            models |= {m for m, _ in hits}
            evidence.extend(sorted(e for _, e in hits)[:1])
        if complete and len(models) == 1 and raws:
            recovered[unit] = (models.pop(), ";".join(sorted(set(evidence))[:3]))
    for unit, hits in agent_hits.items():
        models = {m for m, _ in hits}
        if len(models) == 1 and unit not in recovered:
            recovered[unit] = (models.pop(), ";".join(sorted(e for _, e in hits)[:3]))

    rows = []
    old = effective.set_index(["target_config", "target_key"])
    ids = common.next_decision_ids("legacy", LEGACY_EDITION, len(recovered))
    for (unit, (model_effort, evidence)), decision_id in zip(sorted(recovered.items()), ids):
        model, effort = model_effort.split("|", 1)
        prior = old.loc[[unit]].iloc[-1]  # a unit can carry several effective rows; take the latest
        fields = {c: prior[c] for c in core.LEDGER_COLUMNS if c in prior.index}
        fields.update(decision_id=decision_id, target_config=unit[0], target_key=unit[1], drafter_model=model,
                      drafter_effort=effort or prior["drafter_effort"],
                      evidence_refs=";".join(x for x in (prior["evidence_refs"], f"log:{evidence}") if x),
                      approval_note=f"{prior['approval_note']}; drafter recovered by exact log match",
                      supersedes=prior["decision_id"], status="active")
        rows.append(common.blank_row(**fields))
    if rows and not args.dry_run:
        common.append_ledger_rows(rows)
    by_log = Counter(r["evidence_refs"].split("log:")[1].split(":")[0] for r in rows)
    print(f"scanned {sum(scanned.values())} log files {dict(scanned)}; candidate units {len(unit_lines)}, "
          f"stage-1 agent ids {len(agent_ids)}")
    print(f"recovered drafter model for {len(rows)} legacy unit(s); approver recovered for 0 "
          "(no log records a per-row approval)")
    for log, n in by_log.most_common():
        print(f"  {n:5d}  {log}")
    for config, n in Counter(r["target_config"] for r in rows).most_common():
        print(f"  {config}: {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("check")
    sub.add_parser("backfill")
    p = sub.add_parser("summary")
    p.add_argument("--run", required=True)
    for name in ("approve", "reject"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--approved-by", required=True)
        p.add_argument("--cases", default="")
        p.add_argument("--note", default="")
    p = sub.add_parser("recover-provenance")
    p.add_argument("--logs", action="append", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    handler = {
        "status": cmd_status, "check": cmd_check, "backfill": cmd_backfill, "summary": cmd_summary,
        "approve": cmd_approve, "reject": cmd_reject, "recover-provenance": cmd_recover_provenance,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
