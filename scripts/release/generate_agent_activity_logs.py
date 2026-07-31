from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project agent-activity logs from ~/.codex.")
    parser.add_argument(
        "--repo-cwd",
        default=str(Path.cwd()),
        help="Exact session cwd to treat as the project root.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the generated CSV and README files should be written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Parallel worker count for scanning session files.",
    )
    return parser.parse_args()


def iter_session_files(codex_home: Path):
    for root in [
        codex_home / "sessions",
        codex_home / "archived_sessions",
        codex_home / "sessions_archive",
    ]:
        if not root.exists():
            continue
        if root.name == "sessions":
            yield from root.rglob("*.jsonl")
        else:
            yield from root.glob("*.jsonl")


def excerpt(text: str | None, limit: int = 900) -> str:
    if text is None:
        return ""
    flat = re.sub(r"\s+", " ", str(text).replace("\r", " ").replace("\n", " ").strip())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 4] + " ..."


def normalize_template(text: str) -> str:
    text = excerpt(text, limit=1200)
    text = re.sub(r"/Users/[^\s,;:()]+", "<PATH>", text)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27}\b", "<UUID>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b01[0-9a-z]{24}\b", "<ID>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<DATE>", text)
    text = re.sub(r"\b\d+\b", "<N>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_source(meta_payload: dict) -> tuple[str, str, str, str, str]:
    source = meta_payload.get("source")
    if isinstance(source, dict):
        spawn = ((source.get("subagent") or {}).get("thread_spawn") or {})
        return (
            "subagent",
            json.dumps(source, ensure_ascii=False),
            spawn.get("agent_nickname", ""),
            spawn.get("agent_role", ""),
            spawn.get("parent_thread_id", ""),
        )
    if source == "exec":
        return ("exec", "exec", "", "", "")
    return ("main", str(source or ""), "", "", "")


def is_actual_codex_exec_launch(command: str) -> bool:
    if "codex exec" not in command:
        return False
    stripped = command.strip()
    if re.search(r"(^|[;&|()\s])codex exec\s+--help(\s|$)", stripped):
        return False
    if not re.search(r"(^|[;&|()\s])codex exec(\s|$)", stripped):
        return False
    if (
        re.search(r"\b(ps|pgrep)\b", stripped)
        and re.search(r"\b(rg|grep)\b", stripped)
        and stripped.count("codex exec") == 1
        and not stripped.lstrip().startswith("codex exec")
    ):
        return False
    return True


def launch_kind(command: str) -> str:
    if "--output-schema" in command:
        return "schema_worker"
    if "output_last_message.py" in command or "--output-last-message" in command:
        return "output_last_message_worker"
    if 'Return exactly {"ok":true}' in command:
        return "probe"
    return "generic"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def scan_session_file(path: Path) -> dict | None:
    meta_payload: dict | None = None
    meta_timestamp = ""
    meta_chain_depth = 0
    model = ""
    effort = ""
    first_prompt_timestamp = ""
    first_prompt_text = ""
    last_prompt_timestamp = ""
    last_prompt_text = ""
    function_calls: list[dict] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue

            event_type = event.get("type")
            if event_type == "session_meta":
                meta_chain_depth += 1
                if meta_payload is None:
                    meta_payload = event.get("payload") or {}
                    meta_timestamp = event.get("timestamp", "")
            elif event_type == "turn_context" and not model:
                payload = event.get("payload") or {}
                model = payload.get("model", "")
                effort = payload.get("effort", "")
            elif event_type == "user_message":
                payload = event.get("payload") or {}
                text = payload.get("text")
                if text and text.strip():
                    if not first_prompt_text:
                        first_prompt_timestamp = event.get("timestamp", "")
                        first_prompt_text = text
                    last_prompt_timestamp = event.get("timestamp", "")
                    last_prompt_text = text
            elif event_type == "response_item":
                payload = event.get("payload") or {}
                if payload.get("type") == "message" and payload.get("role") == "user":
                    parts = payload.get("content") or []
                    texts = [
                        part.get("text", "")
                        for part in parts
                        if isinstance(part, dict) and part.get("type") == "input_text" and part.get("text")
                    ]
                    text = "\n".join(texts).strip()
                    if text:
                        if not first_prompt_text:
                            first_prompt_timestamp = event.get("timestamp", "")
                            first_prompt_text = text
                        last_prompt_timestamp = event.get("timestamp", "")
                        last_prompt_text = text
                elif payload.get("type") == "function_call":
                    name = payload.get("name")
                    if name not in {"spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent", "exec_command"}:
                        continue
                    function_calls.append(
                        {
                            "timestamp": event.get("timestamp", ""),
                            "name": name,
                            "call_id": payload.get("call_id", ""),
                            "arguments": ensure_dict(payload.get("arguments") or {}),
                            "output": ensure_dict(payload.get("output") or {}),
                        }
                    )

    if meta_payload is None:
        return None

    return {
        "meta_payload": meta_payload,
        "meta_timestamp": meta_timestamp,
        "meta_chain_depth": meta_chain_depth,
        "model": model,
        "effort": effort,
        "first_prompt_timestamp": first_prompt_timestamp,
        "first_prompt_text": first_prompt_text,
        "last_prompt_timestamp": last_prompt_timestamp,
        "last_prompt_text": last_prompt_text,
        "function_calls": function_calls,
        "session_file": str(path),
    }


def scan_session_file_str(path_str: str) -> dict | None:
    return scan_session_file(Path(path_str))


def main() -> None:
    args = parse_args()
    repo_cwd = args.repo_cwd
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codex_home = Path.home() / ".codex"

    sessions: list[dict] = []
    excluded_subdir_rows: list[dict] = []

    session_files = [str(path) for path in iter_session_files(codex_home)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        scanned_iter = executor.map(scan_session_file_str, session_files, chunksize=16)
        for scanned in scanned_iter:
            if scanned is None:
                continue
            meta_payload = scanned["meta_payload"]
            cwd = meta_payload.get("cwd", "")
            if cwd != repo_cwd and not cwd.startswith(repo_cwd + "/"):
                continue

            if cwd == repo_cwd:
                source_kind, source_text, nickname, role, parent_thread_id = classify_source(meta_payload)
                sessions.append(
                    {
                        "timestamp": scanned["meta_timestamp"],
                        "session_id": meta_payload.get("id", ""),
                        "forked_from_id": meta_payload.get("forked_from_id", ""),
                        "cwd": cwd,
                        "source_kind": source_kind,
                        "source": source_text,
                        "agent_nickname": nickname,
                        "agent_role": role,
                        "parent_thread_id": parent_thread_id,
                        "model": scanned["model"],
                        "reasoning_effort": scanned["effort"],
                        "cli_version": meta_payload.get("cli_version", ""),
                        "session_file": scanned["session_file"],
                        "meta_chain_depth": str(scanned["meta_chain_depth"]),
                        "_first_prompt_timestamp": scanned["first_prompt_timestamp"],
                        "_first_prompt_text": scanned["first_prompt_text"],
                        "_last_prompt_timestamp": scanned["last_prompt_timestamp"],
                        "_last_prompt_text": scanned["last_prompt_text"],
                        "_function_calls": scanned["function_calls"],
                    }
                )
            else:
                excluded_subdir_rows.append(
                    {
                        "timestamp": scanned["meta_timestamp"],
                        "session_id": meta_payload.get("id", ""),
                        "cwd": cwd,
                        "model": scanned["model"],
                        "reasoning_effort": scanned["effort"],
                        "session_file": scanned["session_file"],
                    }
                )

    sessions.sort(key=lambda row: (row["timestamp"], row["session_id"]))
    excluded_subdir_rows.sort(key=lambda row: (row["timestamp"], row["session_id"]))

    main_sessions = [row for row in sessions if row["source_kind"] == "main"]
    subagent_sessions = [row for row in sessions if row["source_kind"] == "subagent"]
    exec_sessions = [row for row in sessions if row["source_kind"] == "exec"]

    session_fields = [
        "timestamp",
        "session_id",
        "forked_from_id",
        "cwd",
        "source_kind",
        "source",
        "agent_nickname",
        "agent_role",
        "model",
        "reasoning_effort",
        "cli_version",
        "session_file",
        "meta_chain_depth",
    ]
    write_csv(out_dir / "repo_root_sessions.csv", session_fields, [{k: row[k] for k in session_fields} for row in sessions])
    write_csv(out_dir / "main_sessions.csv", session_fields, [{k: row[k] for k in session_fields} for row in main_sessions])
    write_csv(
        out_dir / "excluded_subdirectory_sessions.csv",
        ["timestamp", "session_id", "cwd", "model", "reasoning_effort", "session_file"],
        excluded_subdir_rows,
    )

    control_rows: list[dict] = []
    for session in main_sessions:
        for function_call in session["_function_calls"]:
            tool_name = function_call["name"]
            if tool_name not in {"spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent"}:
                continue
            arguments = function_call["arguments"]
            output = function_call["output"]
            control_rows.append(
                {
                    "timestamp": function_call["timestamp"],
                    "main_session_id": session["session_id"],
                    "main_session_model": session["model"],
                    "main_session_effort": session["reasoning_effort"],
                    "tool_name": tool_name,
                    "call_id": function_call["call_id"],
                    "target": arguments.get("target", ""),
                    "targets": json.dumps(arguments.get("targets", ""), ensure_ascii=False) if "targets" in arguments else "",
                    "timeout_ms": arguments.get("timeout_ms", ""),
                    "interrupt": arguments.get("interrupt", ""),
                    "agent_type": arguments.get("agent_type", ""),
                    "requested_model": arguments.get("model", ""),
                    "requested_reasoning_effort": arguments.get("reasoning_effort", ""),
                    "fork_context": arguments.get("fork_context", ""),
                    "message_excerpt": excerpt(arguments.get("message", "")),
                    "output_excerpt": excerpt(json.dumps(output, ensure_ascii=False) if output else ""),
                    "spawned_agent_id": output.get("agent_id", "") if isinstance(output, dict) else "",
                    "spawned_agent_nickname": output.get("nickname", "") if isinstance(output, dict) else "",
                }
            )

    control_rows.sort(key=lambda row: (row["timestamp"], row["tool_name"], row["call_id"], row["main_session_id"]))
    seen_control = set()
    deduped_control_rows: list[dict] = []
    for row in control_rows:
        dedupe_key = row["call_id"] or (
            row["timestamp"],
            row["tool_name"],
            row["main_session_id"],
            row["message_excerpt"],
            row["target"],
            row["targets"],
        )
        if dedupe_key in seen_control:
            continue
        seen_control.add(dedupe_key)
        deduped_control_rows.append(row)

    write_csv(
        out_dir / "built_in_multiagent_tool_calls.csv",
        [
            "timestamp",
            "main_session_id",
            "main_session_model",
            "main_session_effort",
            "tool_name",
            "call_id",
            "target",
            "targets",
            "timeout_ms",
            "interrupt",
            "agent_type",
            "requested_model",
            "requested_reasoning_effort",
            "fork_context",
            "message_excerpt",
            "output_excerpt",
            "spawned_agent_id",
            "spawned_agent_nickname",
        ],
        deduped_control_rows,
    )

    exec_launch_rows: list[dict] = []
    for session in main_sessions:
        for function_call in session["_function_calls"]:
            if function_call["name"] != "exec_command":
                continue
            arguments = function_call["arguments"]
            command = arguments.get("cmd", "")
            if not is_actual_codex_exec_launch(command):
                continue
            model_match = re.search(r"--model\s+([^\s]+)", command)
            effort_match = re.search(r'reasoning_effort\s*=\s*"([^"]+)"', command)
            output_match = re.search(r"(?:-o|--output(?:-path)?)\s+(\S+)", command)
            exec_launch_rows.append(
                {
                    "timestamp": function_call["timestamp"],
                    "main_session_id": session["session_id"],
                    "main_session_model": session["model"],
                    "main_session_effort": session["reasoning_effort"],
                    "launch_kind": launch_kind(command),
                    "inferred_model": model_match.group(1) if model_match else "",
                    "inferred_reasoning_effort": effort_match.group(1) if effort_match else "",
                    "inferred_output_path": output_match.group(1) if output_match else "",
                    "command": command,
                    "command_excerpt": excerpt(command, limit=500),
                }
            )
    exec_launch_rows.sort(key=lambda row: (row["timestamp"], row["main_session_id"], row["command"]))
    write_csv(
        out_dir / "codex_exec_launch_commands.csv",
        [
            "timestamp",
            "main_session_id",
            "main_session_model",
            "main_session_effort",
            "launch_kind",
            "inferred_model",
            "inferred_reasoning_effort",
            "inferred_output_path",
            "command",
            "command_excerpt",
        ],
        exec_launch_rows,
    )

    exec_task_rows: list[dict] = []
    for session in exec_sessions:
        if not session["_last_prompt_text"]:
            continue
        exec_task_rows.append(
            {
                "timestamp": session["_last_prompt_timestamp"] or session["timestamp"],
                "exec_session_id": session["session_id"],
                "model": session["model"],
                "reasoning_effort": session["reasoning_effort"],
                "task_excerpt": excerpt(session["_last_prompt_text"]),
                "session_file": session["session_file"],
            }
        )
    exec_task_rows.sort(key=lambda row: (row["timestamp"], row["exec_session_id"]))
    write_csv(
        out_dir / "codex_exec_worker_tasks.csv",
        ["timestamp", "exec_session_id", "model", "reasoning_effort", "task_excerpt", "session_file"],
        exec_task_rows,
    )

    dispatch_rows = [
        row
        for row in deduped_control_rows
        if row["tool_name"] in {"spawn_agent", "send_input"} and row["message_excerpt"]
    ]
    dispatch_template_groups: dict[tuple[str, str], dict] = {}
    for row in dispatch_rows:
        key = (row["tool_name"], normalize_template(row["message_excerpt"]))
        group = dispatch_template_groups.setdefault(
            key,
            {
                "tool_name": row["tool_name"],
                "normalized_message": key[1],
                "count": 0,
                "example_message": row["message_excerpt"],
                "requested_model_modes": Counter(),
                "requested_effort_modes": Counter(),
                "agent_type_modes": Counter(),
            },
        )
        group["count"] += 1
        if row["requested_model"]:
            group["requested_model_modes"][row["requested_model"]] += 1
        if row["requested_reasoning_effort"]:
            group["requested_effort_modes"][row["requested_reasoning_effort"]] += 1
        if row["agent_type"]:
            group["agent_type_modes"][row["agent_type"]] += 1

    dispatch_template_rows = []
    for group in dispatch_template_groups.values():
        dispatch_template_rows.append(
            {
                "tool_name": group["tool_name"],
                "count": group["count"],
                "requested_model_modes": ", ".join(
                    f"{value}:{count}" for value, count in group["requested_model_modes"].most_common()
                ),
                "requested_effort_modes": ", ".join(
                    f"{value}:{count}" for value, count in group["requested_effort_modes"].most_common()
                ),
                "agent_type_modes": ", ".join(
                    f"{value}:{count}" for value, count in group["agent_type_modes"].most_common()
                ),
                "normalized_message": group["normalized_message"],
                "example_message": group["example_message"],
            }
        )
    dispatch_template_rows.sort(key=lambda row: (-row["count"], row["tool_name"], row["normalized_message"]))
    write_csv(
        out_dir / "built_in_multiagent_dispatch_templates.csv",
        [
            "tool_name",
            "count",
            "requested_model_modes",
            "requested_effort_modes",
            "agent_type_modes",
            "normalized_message",
            "example_message",
        ],
        dispatch_template_rows,
    )

    exec_template_groups: dict[str, dict] = {}
    for row in exec_task_rows:
        normalized = normalize_template(row["task_excerpt"])
        group = exec_template_groups.setdefault(
            normalized,
            {
                "count": 0,
                "example_message": row["task_excerpt"],
                "model_modes": Counter(),
                "effort_modes": Counter(),
            },
        )
        group["count"] += 1
        if row["model"]:
            group["model_modes"][row["model"]] += 1
        if row["reasoning_effort"]:
            group["effort_modes"][row["reasoning_effort"]] += 1

    exec_template_rows = []
    for normalized, group in exec_template_groups.items():
        exec_template_rows.append(
            {
                "count": group["count"],
                "model_modes": ", ".join(f"{value}:{count}" for value, count in group["model_modes"].most_common()),
                "effort_modes": ", ".join(
                    f"{value}:{count}" for value, count in group["effort_modes"].most_common()
                ),
                "normalized_message": normalized,
                "example_message": group["example_message"],
            }
        )
    exec_template_rows.sort(key=lambda row: (-row["count"], row["normalized_message"]))
    write_csv(
        out_dir / "codex_exec_worker_task_templates.csv",
        ["count", "model_modes", "effort_modes", "normalized_message", "example_message"],
        exec_template_rows,
    )

    control_counter = Counter(row["tool_name"] for row in deduped_control_rows)
    subagent_role_counter = Counter(row["agent_type"] for row in dispatch_rows if row["agent_type"])
    subagent_model_counter = Counter(
        (row["requested_model"], row["requested_reasoning_effort"])
        for row in dispatch_rows
        if row["requested_model"] or row["requested_reasoning_effort"]
    )
    exec_launch_counter = Counter(row["launch_kind"] for row in exec_launch_rows)
    exec_model_counter = Counter((row["model"], row["reasoning_effort"]) for row in exec_task_rows)
    counts_by_day: dict[str, Counter] = defaultdict(Counter)
    for row in sessions:
        counts_by_day[(row["timestamp"] or "")[:10]][row["source_kind"]] += 1

    lines = [
        "# Agent Activity Logs",
        "",
        f"These logs were generated directly from `~/.codex` and filtered to sessions whose `cwd` exactly equals `{repo_cwd}`. That strict filter captures the main project and excludes unrelated work nested under the repo path.",
        "",
        "The only excluded subdirectory sessions were under `streeteasy-enhanced-extension/sqft-from-photos`, which sits inside the repo tree but is a different project. Those excluded rows are listed in `excluded_subdirectory_sessions.csv`.",
        "",
        "## Scope",
        f"- Repo-root session span: `{sessions[0]['timestamp']}` to `{sessions[-1]['timestamp']}`",
        f"- Repo-root sessions captured: `{len(sessions)}`",
        f"- Main sessions: `{len(main_sessions)}`",
        f"- Built-in subagent sessions: `{len(subagent_sessions)}`",
        f"- `codex exec` worker sessions: `{len(exec_sessions)}`",
        f"- Excluded subdirectory sessions: `{len(excluded_subdir_rows)}`",
        "",
        "## Files",
        "- `repo_root_sessions.csv`: every repo-root session that matched the filter, with source classification and first-turn model context.",
        "- `main_sessions.csv`: the main-thread subset only.",
        "- `built_in_multiagent_tool_calls.csv`: the authoritative built-in multiagent control-plane log, deduplicated by `call_id` across forked main sessions.",
        "- `built_in_multiagent_dispatch_templates.csv`: normalized first-dispatch templates for `spawn_agent` and `send_input`, grouped so filename- or shard-only variants collapse into counted schemas.",
        "- `codex_exec_launch_commands.csv`: top-level main-session `exec_command` calls that actually launched `codex exec`. Process-inspection commands that only mentioned `codex exec` inside quoted `ps` or `rg` patterns were excluded.",
        "- `codex_exec_worker_tasks.csv`: the first persisted task message for each `codex exec` worker session.",
        "- `codex_exec_worker_task_templates.csv`: normalized first-task templates for `codex exec` workers, grouped so shard- or filename-only variants collapse into counted schemas.",
        "- `excluded_subdirectory_sessions.csv`: sessions under repo subdirectories that were intentionally excluded from the project logs.",
        "",
        "## Built-in Multiagent",
    ]
    for tool_name in ["spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent"]:
        if control_counter.get(tool_name):
            lines.append(f"- `{tool_name}` calls: `{control_counter[tool_name]}`")
    lines.extend(
        [
            f"- Actual built-in task dispatches (`spawn_agent` + `send_input`): `{control_counter.get('spawn_agent', 0) + control_counter.get('send_input', 0)}`",
            f"- Distinct normalized dispatch templates: `{len(dispatch_template_rows)}`",
            "- Requested roles on dispatches:",
        ]
    )
    for role, count in sorted(subagent_role_counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  - `{role}`: `{count}`")
    lines.append("- Requested model/effort pairs on dispatches:")
    for (model, effort), count in sorted(subagent_model_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        lines.append(f"  - `{model}` / `{effort}`: `{count}`")

    lines.extend(
        [
            "",
            "## `codex exec`",
            f"- Top-level actual `codex exec` launch commands captured: `{len(exec_launch_rows)}`",
            f"- Persisted `codex exec` worker sessions captured: `{len(exec_task_rows)}`",
            f"- Distinct normalized worker-task templates: `{len(exec_template_rows)}`",
            "- Launch kinds:",
        ]
    )
    for kind, count in sorted(exec_launch_counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  - `{kind}`: `{count}`")
    lines.append("- Worker model/effort pairs from persisted worker sessions:")
    for (model, effort), count in sorted(exec_model_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        lines.append(f"  - `{model}` / `{effort}`: `{count}`")

    lines.extend(
        [
            "",
            "## Source Audit And Exclusions",
            "- `~/.codex/sessions`, `~/.codex/archived_sessions`, and `~/.codex/sessions_archive` were used as the authoritative source because they preserve session metadata, turn context, tool calls, and child-session identities.",
            "- `~/.codex/history.jsonl` was checked but excluded from the final logs because it records user text history, not a reliable execution or control-plane record.",
            "- `~/.codex/session_index.jsonl` was checked but excluded because it is an index summary, not a richer source than the session JSONL files already scanned.",
            "- `~/.codex/shell_snapshots/` was checked but excluded because it stores shell environment snapshots rather than a trustworthy invocation log; it is useful for environment reconstruction, not for counting agent or swarm invocations.",
            "",
            "## Interpretation Notes",
            "- The authoritative built-in multiagent record is `built_in_multiagent_tool_calls.csv`, because it is extracted from the main-thread control plane after deduplicating replayed history across forked main sessions.",
            "- For built-in multiagent work, the first important message is already present in the main-thread dispatch record. The template file therefore groups normalized `spawn_agent` and `send_input` messages rather than replaying child sessions.",
            "- The worker-session log captures `codex exec` runs even when they were launched by a Python or shell wrapper, as long as those workers persisted as `exec` sessions in `~/.codex`.",
            "- The launch-command log is narrower: it only captures wrapper shells that were recoverable as top-level main-session `exec_command` calls.",
            "- Batch-level swarm grouping is only partially recoverable from `~/.codex`. Worker sessions are authoritative for what ran. Exact parent shell batches are only recoverable when the top-level main session persisted the launch command.",
            "",
            "## Session Counts By Day",
        ]
    )
    for day in sorted(counts_by_day):
        parts = []
        for source_kind in ["main", "subagent", "exec"]:
            if counts_by_day[day].get(source_kind):
                parts.append(f"{source_kind}={counts_by_day[day][source_kind]}")
        lines.append(f"- `{day}`: " + ", ".join(parts))

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
