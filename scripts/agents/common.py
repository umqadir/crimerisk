"""Shared pieces of the agent runners: hashing, ledger I/O, packets and the headless claude call.

Only the runners write files. Agents return one JSON object on stdout (validated against the
kind's schema by the CLI's `--json-schema`) and never edit the repository.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crimerisk import ledger as core  # noqa: E402

RUNS_DIR = REPO_ROOT / "state" / "review" / "runs"
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
EDITION = "2025.3"

# Model policy (design section 1): Opus high drafts and reviews, Sonnet medium transcribes.
AGENT_MODELS = {
    "drafter": ("opus", "high"),
    "reviewer": ("opus", "high"),
    "transcriber": ("sonnet", "medium"),
}
AGENT_TOOLS = {
    "drafter": "Read Grep Glob",
    "reviewer": "Read Grep Glob Bash(uv run python scripts/agents/query.py *)",
    "transcriber": "Read Grep Glob",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def local_ref(path: Path) -> str:
    """Evidence ref in the ledger's local form `path@sha256`."""
    return f"{rel(path)}@{sha256_file(path)}"


# ---------------------------------------------------------------- ledger I/O


def read_ledger():
    return core.read_ledger(REPO_ROOT)


def next_decision_ids(kind: str, edition: str, count: int) -> list[str]:
    ledger = read_ledger()
    prefix = f"D{edition}-{kind}-"
    used = [int(d[len(prefix):]) for d in ledger["decision_id"] if d.startswith(prefix)]
    start = max(used, default=0) + 1
    return [f"{prefix}{n:05d}" for n in range(start, start + count)]


def blank_row(**values: str) -> dict[str, str]:
    row = {column: "" for column in core.LEDGER_COLUMNS}
    unknown = set(values) - set(row)
    if unknown:
        raise KeyError(f"not ledger columns: {sorted(unknown)}")
    row.update({k: "" if v is None else str(v) for k, v in values.items()})
    row.setdefault("status", "active")
    if not row["status"]:
        row["status"] = "active"
    return row


def append_ledger_rows(rows: list[dict[str, str]]) -> None:
    """Append rows to configs/ledger/decisions.csv. The file is never rewritten."""
    path = core.ledger_path(REPO_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(core.LEDGER_COLUMNS), lineterminator="\n")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in core.LEDGER_COLUMNS})


def subject_from_key(target_key: str) -> str:
    parts = dict(p.split("=", 1) for p in target_key.split(";"))
    if "case_id" in parts:
        return f"case:{parts['case_id']}"
    if "city_key" in parts:
        tail = f":{parts['offense']}" if "offense" in parts else ""
        return f"city:{parts['city_key']}{tail}"
    ori = parts.get("ori") or parts.get("ori9")
    if ori:
        tail = "".join(f":{parts[k]}" for k in ("year", "offense") if k in parts)
        return f"ori:{ori}{tail}"
    if "jurisdiction_id" in parts:
        return f"jurisdiction:{parts['jurisdiction_id']}"
    if "county_geoid" in parts:
        return f"county:{parts['county_geoid']}"
    if "service_scope_id" in parts:
        return f"service_scope:{parts['service_scope_id']}"
    return target_key


# ---------------------------------------------------------------- governed payload writes


def upsert_csv_unit(
    path: Path, key_columns: list[str], key_values: dict[str, str], rows: list[dict[str, str]],
    *, add_columns: list[str] | None = None,
) -> None:
    """Replace every row of one unit in a governed CSV (or append the unit). Other rows keep their values.

    `add_columns` (declared by the proposal) appends those columns to the header, blank on every other
    row; any other column not in the header is an error."""
    header: list[str]
    body: list[dict[str, str]]
    leading: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        while lines and lines[0].startswith("#"):
            leading.append(lines.pop(0))
        reader = csv.DictReader(lines)
        header = list(reader.fieldnames or [])
        body = [dict(r) for r in reader]
    else:
        header = list(rows[0].keys())
        body = []
    for column in add_columns or []:
        if column not in header:
            header.append(column)
    missing = [c for r in rows for c in r if c not in header]
    if missing:
        raise ValueError(f"{path}: proposal columns not in header: {sorted(set(missing))}")

    def is_unit(r: dict[str, str]) -> bool:
        return all(r.get(k, "") == key_values[k] for k in key_columns)

    kept = [r for r in body if not is_unit(r)]
    positions = [i for i, r in enumerate(body) if is_unit(r)]
    insert_at = positions[0] if positions else len(kept)
    new_rows = [{c: r.get(c, "") for c in header} for r in rows]
    out = kept[:insert_at] + new_rows + kept[insert_at:]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(out)
    path.write_text("".join(leading) + buf.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------- packets


def write_packet(packet_dir: Path, *, case_text: str, signals: dict, evidence: list[Path]) -> str:
    """Write packet.json deterministically and return its sha256.

    `signals` are precomputed hints, kept apart from the evidence and labeled as such
    (design rule 2). Evidence files are listed with their hashes.
    """
    packet_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(
        ({"path": rel(p), "sha256": sha256_file(p)} for p in evidence),
        key=lambda e: e["path"],
    )
    packet = {"case_text": case_text, "signals": signals, "evidence": entries}
    text = json.dumps(packet, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    (packet_dir / "packet.json").write_text(text, encoding="utf-8")
    return sha256_text(text)


# ---------------------------------------------------------------- headless agent call


def cli_version() -> str:
    return subprocess.run(["claude", "--version"], capture_output=True, text=True, check=True).stdout.strip()


def prompt_sha256(agent: str, template: Path | None, rendered: str, schema: Path) -> str:
    h = hashlib.sha256()
    h.update((AGENTS_DIR / f"{agent}.md").read_bytes())
    h.update(template.read_bytes() if template else b"")
    h.update(rendered.encode("utf-8"))
    h.update(schema.read_bytes())
    return h.hexdigest()


class AgentError(RuntimeError):
    pass


class PendingTask(Exception):
    """Session mode: the task file is written and waits for a session subagent's output."""


# Execution mode (design section 6.4 plus the session mode):
#   headless: `claude -p --agent <name>` subprocess per call (needs a logged-in CLI).
#   session:  the runner writes <tag>.task.md; a Claude Code session runs each task with one
#             subagent (opus-high for drafter/reviewer, sonnet-mechanical for transcriber) that
#             writes <tag>.output.json; rerunning the runner (`ingest`) validates and continues.
MODE = {"name": "headless"}
SESSION_MODELS = {
    "drafter": "claude-opus-5-5 (session subagent opus-high)",
    "reviewer": "claude-opus-5-5 (session subagent opus-high)",
    "transcriber": "claude-sonnet (session subagent sonnet-mechanical)",
}
AUTH_HINTS = ("credit balance", "failed to authenticate", "invalid api key", "oauth", "not logged in", "authentication")


def set_mode(name: str) -> None:
    if name not in {"headless", "session"}:
        raise ValueError(name)
    MODE["name"] = name


def agent_system_prompt(agent: str) -> str:
    text = (AGENTS_DIR / f"{agent}.md").read_text()
    return text.split("---", 2)[2].strip() if text.startswith("---") else text


def validate(obj: object, schema: dict, where: str = "$") -> list[str]:
    """Minimal JSON-schema check for the subset the skill schemas use."""
    errors: list[str] = []
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "number": (int, float), "integer": int}
    if kind and not isinstance(obj, types[kind]) or (kind in {"number", "integer"} and isinstance(obj, bool)):
        return [f"{where}: expected {kind}"]
    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{where}: {obj!r} not in enum")
    if kind == "object":
        for key in schema.get("required", []):
            if key not in obj:
                errors.append(f"{where}: missing {key}")
        props = schema.get("properties", {})
        for key, value in obj.items():
            if key in props:
                errors += validate(value, props[key], f"{where}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{where}: unexpected {key}")
    if kind == "array" and "items" in schema:
        for i, item in enumerate(obj):
            errors += validate(item, schema["items"], f"{where}[{i}]")
    return errors


def _record(agent: str, output: dict, models: list[str], rendered: str, template: Path | None, schema: Path, **extra) -> dict:
    return {
        "agent": agent,
        "mode": MODE["name"],
        "output": output,
        "models": models,
        "effort": AGENT_MODELS[agent][1],
        "prompt_sha256": prompt_sha256(agent, template, rendered, schema),
        "output_sha256": sha256_text(canonical_json(output)),
        **extra,
    }


def run_agent(
    agent: str,
    rendered: str,
    *,
    schema: Path,
    out_dir: Path,
    template: Path | None = None,
    add_dirs: tuple[Path, ...] = (),
    extra_tools: str = "",
    tag: str | None = None,
    retries: int = 1,
) -> dict:
    """Run one agent step and return {output, models, mode, prompt_sha256, ...}.

    A record already at <out_dir>/<tag>.json is returned as is (resumable runs). Headless mode
    keeps the stream-json transcript at <tag>.transcript.jsonl. Session mode writes <tag>.task.md
    and raises PendingTask until <tag>.output.json exists and validates against the schema.
    """
    tag = tag or agent
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / f"{tag}.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    schema_obj = json.loads(schema.read_text())
    if MODE["name"] == "session":
        output_path = out_dir / f"{tag}.output.json"
        task_path = out_dir / f"{tag}.task.md"
        if output_path.exists():
            try:
                output = json.loads(output_path.read_text())
            except json.JSONDecodeError as exc:
                raise AgentError(f"{rel(output_path)} is not JSON: {exc}") from exc
            errors = validate(output, schema_obj)
            if errors:
                raise AgentError(f"{rel(output_path)} fails {rel(schema)}: {errors[:5]}")
            record = _record(agent, output, [SESSION_MODELS[agent]], rendered, template, schema,
                             cli_version=f"session ({cli_version()})", task=rel(task_path), output_file=rel(output_path))
            record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
            return record
        task_path.write_text(
            f"# Task: {agent} ({tag})\n\n"
            f"Act as the `{agent}` agent. Its instructions:\n\n{agent_system_prompt(agent)}\n\n"
            f"## Case\n\n{rendered}\n\n"
            f"## Output\n\nWrite exactly one JSON object matching the JSON schema `{rel(schema)}` to "
            f"`{rel(output_path)}` (repository root: {REPO_ROOT}). Write no other file.\n",
            encoding="utf-8",
        )
        raise PendingTask(rel(task_path))
    model, effort = AGENT_MODELS[agent]
    tools = " ".join(t for t in (AGENT_TOOLS[agent], extra_tools) if t)
    cmd = [
        "claude", "-p",
        "--agent", agent,
        "--model", model,
        "--effort", effort,
        "--output-format", "stream-json", "--verbose",
        "--json-schema", json.dumps(schema_obj),
        "--permission-mode", "dontAsk",
        "--allowedTools", tools,
        "--no-session-persistence",
    ]
    for d in add_dirs:
        cmd += ["--add-dir", str(d)]
    transcript = out_dir / f"{tag}.transcript.jsonl"
    last_error = ""
    for _attempt in range(retries + 1):
        started = time.time()
        proc = subprocess.run(cmd, input=rendered, capture_output=True, text=True, cwd=REPO_ROOT)
        transcript.write_text(proc.stdout, encoding="utf-8")
        result = None
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                result = event
        if result and not result.get("is_error") and result.get("structured_output") is not None:
            record = _record(agent, result["structured_output"], sorted((result.get("modelUsage") or {}).keys()),
                             rendered, template, schema, cli_version=cli_version(),
                             cost_usd=result.get("total_cost_usd"), duration_s=round(time.time() - started, 1),
                             num_turns=result.get("num_turns"), transcript=rel(transcript))
            record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
            return record
        last_error = str((result or {}).get("result") or proc.stderr[-2000:] or "no result event")
        if any(h in last_error.lower() for h in AUTH_HINTS):
            raise AgentError(
                f"{agent}: the claude CLI could not authenticate ({last_error[:200]}). Run `claude auth login` "
                "once in a terminal, or use --mode session."
            )
        if "rate limit" in last_error.lower() or "usage limit" in last_error.lower():
            raise AgentError(f"{agent}: account limit: {last_error}")
    raise AgentError(f"{agent} failed after {retries + 1} attempt(s): {last_error[:2000]}")


# ---------------------------------------------------------------- run state


def load_state(run_dir: Path) -> dict:
    path = run_dir / "state.json"
    return json.loads(path.read_text()) if path.exists() else {"cases": {}}


def save_state(run_dir: Path, state: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
