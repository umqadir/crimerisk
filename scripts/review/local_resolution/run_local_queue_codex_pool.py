from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from queue import Empty, Queue
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


TZ = ZoneInfo("America/New_York")
USAGE_RETRY_RE = re.compile(r"try again at (\d{1,2}:\d{2} [AP]M)", re.IGNORECASE)
DEFAULT_START_BATCH = 26
DEFAULT_END_BATCH = 35
DEFAULT_CONCURRENCY = 2
DEFAULT_RESULTS_DIR = "results_codex_exec_trial_026_035"
DEFAULT_LOGS_DIR = "logs_codex_exec_trial_026_035"
DEFAULT_STATE_FILE = "codex_exec_trial_026_035_state.json"
DEFAULT_RUN_ROOT = "state/review/runs/local_resolution/local_queue_clean_run"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().with_name("local_queue_output_schema.json")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"

LOCAL_FIRST_PASS_PROMPT = """You are classifying one 5-case local-queue batch for V2 jurisdiction semantics.

Classify each case into exactly one of:
- municipal_place
- municipal_cousub
- reclassify_nonmunicipal
- reclassify_overlap
- exclude
- escalate

Rules:
- The primary partition is ordinary municipal jurisdictions plus ordinary nonmunicipal remainder geography.
- Use reclassify_nonmunicipal for ordinary county police or sheriff-style county remainder forces.
- Use reclassify_overlap for special, sovereign, tribal, campus, transit, airport, port, housing, regional, joint, or consolidated footprints that are not ordinary primary partition jurisdictions.
- Contract-policed but separately tracked municipalities are still municipal.
- Do not exclude merely because a place later reorganized.
- Do not assume two rows with the same city name are duplicates; distinct ORIs may map differently. Verify each ORI independently.
- Use provided exact_identity_rows and candidate_geographies as deterministic lookup inputs.
- Do not spend time reading local files to rediscover identity or candidate geometry already present in the case payload.
- Do not inspect prior run outputs, logs, cached browser artifacts, or other local files for answers.
- Browse official public web sources only as needed to resolve jurisdiction semantics.
- Prefer 1-2 official sources unless genuinely ambiguous.
- If municipal, resolved_geo_type should be place or cousub, with the best resolved_geoid and resolved_label you can support. Otherwise set those fields to null.
- Keep reason to one sentence.
- Return only JSON matching the required schema.
"""

LOCAL_SECOND_PASS_PROMPT = """You are re-reviewing one 5-case local-queue second-pass batch for V2 jurisdiction semantics.

Classify each case into exactly one of:
- municipal_place
- municipal_cousub
- reclassify_nonmunicipal
- reclassify_overlap
- exclude
- escalate

Rules:
- The primary partition is ordinary municipal jurisdictions plus ordinary nonmunicipal remainder geography.
- Use reclassify_nonmunicipal for ordinary county police or sheriff-style county remainder forces.
- Use reclassify_overlap for special, sovereign, tribal, campus, transit, airport, port, housing, regional, joint, or consolidated footprints that are not ordinary primary partition jurisdictions.
- Contract-policed but separately tracked municipalities are still municipal.
- Do not exclude merely because a place later reorganized.
- Do not assume two rows with the same city name are duplicates; distinct ORIs may map differently. Verify each ORI independently.
- Use provided exact_identity_rows, candidate_geographies, and first_pass_result as inputs.
- Treat first_pass_result as advisory only; override it when the payload or official public sources support a better answer.
- Do not spend time reading local files to rediscover identity or candidate geometry already present in the case payload.
- Do not inspect prior run outputs, logs, cached browser artifacts, or other local files for answers.
- Prefer resolving from deterministic payload inputs when they clearly identify a place, town, borough, township, village, city, tribal jurisdiction, county force, or special footprint.
- Do not spend time hunting FBI/CDE APIs, GitHub mirrors, or unofficial ORI rosters; use official local government, sheriff, tribal, county, university, or Census sources and escalate if they do not confirm the mapping quickly.
- If exact_identity_rows and candidate_geographies do not support a municipal assignment, and official public sources still do not identify a specific jurisdiction, prefer escalate over a guessed municipal assignment.
- Browse official public web sources only as needed to resolve jurisdiction semantics.
- Prefer 1-2 official sources unless genuinely ambiguous.
- If municipal, resolved_geo_type should be place or cousub, with the best resolved_geoid and resolved_label you can support. Otherwise set those fields to null.
- Keep reason to one sentence.
- Return only JSON matching the required schema.
"""


PA_OVERLAY = """
Pennsylvania-specific note:
- In Pennsylvania, borough, township, and city governments often resolve to municipal_cousub rather than municipal_place; use county context and official government structure.
"""


@dataclass
class BatchPaths:
    batch_num: int
    input_path: Path
    result_path: Path
    raw_result_path: Path
    log_path: Path

    @property
    def raw_tmp_path(self) -> Path:
        return self.raw_result_path.with_suffix(".tmp.json")


def now_iso() -> str:
    return datetime.now(TZ).isoformat()


def build_prompt(batch_rows: list[dict[str, Any]], *, prompt_kind: str) -> str:
    has_pa = any((row.get("state_abbr") or "").lower() == "pa" for row in batch_rows)
    if prompt_kind == "local_first_pass":
        base_prompt = LOCAL_FIRST_PASS_PROMPT
    elif prompt_kind == "local_second_pass":
        base_prompt = LOCAL_SECOND_PASS_PROMPT
    else:
        raise ValueError(f"unsupported prompt_kind={prompt_kind}")

    parts = [base_prompt.strip()]
    if has_pa:
        parts.append(PA_OVERLAY.strip())
    parts.append("Cases:")
    parts.append(json.dumps(batch_rows, ensure_ascii=True))
    return "\n\n".join(parts) + "\n"


def parse_retry_at(stderr: str) -> str | None:
    match = USAGE_RETRY_RE.search(stderr)
    if not match:
        return None
    time_part = match.group(1).upper()
    now_local = datetime.now(TZ)
    parsed = datetime.strptime(time_part, "%I:%M %p").replace(
        year=now_local.year,
        month=now_local.month,
        day=now_local.day,
        tzinfo=TZ,
    )
    if parsed <= now_local:
        parsed += timedelta(days=1)
    return parsed.isoformat()


def normalize_output(
    raw_result_path: Path,
    result_path: Path,
    expected_case_ids: set[str],
) -> None:
    raw_payload = json.loads(raw_result_path.read_text())
    results = canonicalize_results(raw_payload["results"])
    validate_result_payload(results, expected_case_ids)
    tmp_result = result_path.with_suffix(".tmp.json")
    tmp_result.write_text(json.dumps(results, indent=2) + "\n")
    tmp_result.replace(result_path)


def canonicalize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in results:
        row = dict(row)
        geo_type = row.get("resolved_geo_type")
        if isinstance(geo_type, str):
            geo_key = geo_type.strip().lower()
            if geo_key in {"municipal_place", "place"}:
                row["resolved_geo_type"] = "place"
            elif geo_key in {"municipal_cousub", "cousub", "county_subdivision"}:
                row["resolved_geo_type"] = "cousub"
        geoid = row.get("resolved_geoid")
        if isinstance(geoid, str) and row.get("resolved_geo_type") in {"place", "cousub"}:
            suffix = geoid.rsplit("US", 1)[-1] if "US" in geoid else geoid
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if row["resolved_geo_type"] == "place" and len(digits) >= 7:
                row["resolved_geoid"] = digits[-7:]
            elif row["resolved_geo_type"] == "cousub" and len(digits) >= 10:
                row["resolved_geoid"] = digits[-10:]
        out.append(row)
    return out


def validate_result_payload(payload: Any, expected_case_ids: set[str]) -> None:
    if not isinstance(payload, list):
        raise ValueError("result payload is not a list")
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("result row is not an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError("missing string case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id}")
        seen.add(case_id)
    if seen != expected_case_ids:
        missing = sorted(expected_case_ids - seen)
        extra = sorted(seen - expected_case_ids)
        raise ValueError(f"case_id mismatch missing={missing} extra={extra}")


def validate_result(result_path: Path, expected_case_ids: set[str]) -> None:
    payload = json.loads(result_path.read_text())
    validate_result_payload(payload, expected_case_ids)


def try_recover_completed_output(paths: BatchPaths, expected_case_ids: set[str]) -> bool:
    for candidate in (paths.raw_tmp_path, paths.raw_result_path):
        if not candidate.exists():
            continue
        try:
            normalize_output(candidate, paths.result_path, expected_case_ids)
            if candidate == paths.raw_tmp_path:
                shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
            validate_result(paths.result_path, expected_case_ids)
            return True
        except Exception:
            continue
    return False


def run_one_batch(
    repo_root: Path,
    worker_cwd: Path,
    schema_path: Path,
    paths: BatchPaths,
    timeout_seconds: int,
    *,
    model: str,
    reasoning_effort: str,
    prompt_kind: str,
) -> dict[str, Any]:
    batch_rows = json.loads(paths.input_path.read_text())
    expected_case_ids = {str(row["case_id"]) for row in batch_rows}
    prompt = build_prompt(batch_rows, prompt_kind=prompt_kind)
    paths.result_path.parent.mkdir(parents=True, exist_ok=True)
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    if paths.raw_tmp_path.exists():
        paths.raw_tmp_path.unlink()
    cmd = [
        "codex",
        "exec",
        "-",
        "--sandbox",
        "read-only",
        "--cd",
        str(worker_cwd),
        "--model",
        model,
        "--ephemeral",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(paths.raw_tmp_path),
        "--config",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--config",
        'web_search="live"',
        "--config",
        'approval_policy="never"',
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=worker_cwd,
            env=os.environ.copy(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        stdout = exc.stdout or ""
        retry_at = parse_retry_at(stderr)
        recovered = try_recover_completed_output(paths, expected_case_ids)
        paths.log_path.write_text(
            "\n".join(
                [
                    f"command: {' '.join(cmd)}",
                    "returncode: timeout",
                    "----- stdout -----",
                    stdout,
                    "----- stderr -----",
                    stderr,
                ]
            )
        )
        return {
            "status": "completed" if recovered else ("rate_limited" if retry_at else "timeout"),
            "retry_at": retry_at,
            "elapsed_seconds": round(time.time() - start, 2),
            "error": None if recovered else "timeout",
        }

    paths.log_path.write_text(
        "\n".join(
            [
                f"command: {' '.join(cmd)}",
                f"returncode: {proc.returncode}",
                "----- stdout -----",
                proc.stdout,
                "----- stderr -----",
                proc.stderr,
            ]
        )
    )

    retry_at = parse_retry_at(proc.stderr)
    if retry_at:
        if try_recover_completed_output(paths, expected_case_ids):
            return {
                "status": "completed",
                "retry_at": retry_at,
                "elapsed_seconds": round(time.time() - start, 2),
                "error": None,
            }
        return {
            "status": "rate_limited",
            "retry_at": retry_at,
            "elapsed_seconds": round(time.time() - start, 2),
            "error": "usage_cap",
        }

    if proc.returncode != 0:
        if try_recover_completed_output(paths, expected_case_ids):
            return {
                "status": "completed",
                "retry_at": None,
                "elapsed_seconds": round(time.time() - start, 2),
                "error": None,
            }
        return {
            "status": "error",
            "retry_at": None,
            "elapsed_seconds": round(time.time() - start, 2),
            "error": f"codex exec failed ({proc.returncode})",
        }

    shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
    normalize_output(paths.raw_result_path, paths.result_path, expected_case_ids)
    validate_result(paths.result_path, expected_case_ids)
    return {
        "status": "completed",
        "retry_at": None,
        "elapsed_seconds": round(time.time() - start, 2),
        "error": None,
    }


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        if path.exists():
            self.state = json.loads(path.read_text())
        else:
            self.state = {
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "global": {
                    "next_retry_at": None,
                },
                "runner": {
                    "lock_path": None,
                },
                "batches": {},
            }
            self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2) + "\n")

    def ensure_batch(self, batch_num: int, result_path: Path, log_path: Path) -> None:
        key = f"{batch_num:03d}"
        with self.lock:
            self.state["batches"].setdefault(
                key,
                {
                    "status": "pending",
                    "attempts": 0,
                    "result_path": str(result_path),
                    "log_path": str(log_path),
                    "updated_at": now_iso(),
                    "started_at": None,
                    "completed_at": None,
                    "last_error": None,
                    "last_retry_at": None,
                    "elapsed_seconds": None,
                },
            )
            self.state["updated_at"] = now_iso()
            self._flush()

    def mark_existing_complete(self, batch_num: int) -> None:
        key = f"{batch_num:03d}"
        with self.lock:
            entry = self.state["batches"][key]
            entry["status"] = "completed"
            entry["updated_at"] = now_iso()
            entry["completed_at"] = now_iso()
            entry["last_error"] = None
            self.state["updated_at"] = now_iso()
            self._flush()

    def start_attempt(self, batch_num: int) -> None:
        key = f"{batch_num:03d}"
        with self.lock:
            entry = self.state["batches"][key]
            entry["status"] = "running"
            entry["attempts"] += 1
            entry["started_at"] = now_iso()
            entry["updated_at"] = now_iso()
            self.state["updated_at"] = now_iso()
            self._flush()

    def finish_attempt(self, batch_num: int, result: dict[str, Any]) -> None:
        key = f"{batch_num:03d}"
        with self.lock:
            entry = self.state["batches"][key]
            entry["status"] = result["status"]
            entry["updated_at"] = now_iso()
            if result["status"] == "completed":
                entry["completed_at"] = now_iso()
            entry["last_error"] = result["error"]
            entry["last_retry_at"] = result["retry_at"]
            entry["elapsed_seconds"] = result["elapsed_seconds"]
            if result["status"] == "rate_limited":
                current = self.state["global"]["next_retry_at"]
                if current is None or result["retry_at"] < current:
                    self.state["global"]["next_retry_at"] = result["retry_at"]
            self.state["updated_at"] = now_iso()
            self._flush()

    def reset_running_to_pending(self) -> None:
        with self.lock:
            for entry in self.state["batches"].values():
                if entry["status"] == "running":
                    entry["status"] = "pending"
                    entry["last_error"] = "interrupted_runner"
                    entry["updated_at"] = now_iso()
            self.state["updated_at"] = now_iso()
            self._flush()

    def clear_global_retry_at(self) -> None:
        with self.lock:
            self.state["global"]["next_retry_at"] = None
            self.state["updated_at"] = now_iso()
            self._flush()

    def pending_batches(self, batch_nums: list[int]) -> list[int]:
        pending = []
        for batch_num in batch_nums:
            key = f"{batch_num:03d}"
            status = self.state["batches"][key]["status"]
            if status != "completed":
                pending.append(batch_num)
        return pending

    def next_retry_at(self) -> str | None:
        return self.state["global"]["next_retry_at"]

    def summary(self, batch_nums: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for batch_num in batch_nums:
            key = f"{batch_num:03d}"
            status = self.state["batches"][key]["status"]
            counts[status] = counts.get(status, 0) + 1
        return counts


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextmanager
def runner_lock(lock_path: Path, force: bool = False):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text())
            pid = int(payload["pid"])
        except Exception:
            pid = None
        if pid and pid_is_alive(pid) and not force:
            raise RuntimeError(f"runner already active under pid {pid}")
        lock_path.unlink(missing_ok=True)
    payload = {"pid": os.getpid(), "started_at": now_iso()}
    lock_path.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def build_paths(run_root: Path, results_dir: str, logs_dir: str, batch_num: int) -> BatchPaths:
    return BatchPaths(
        batch_num=batch_num,
        input_path=run_root / f"batch_{batch_num:03d}.json",
        result_path=run_root / results_dir / f"batch_{batch_num:03d}_results.json",
        raw_result_path=run_root / results_dir / f"batch_{batch_num:03d}_results.raw.json",
        log_path=run_root / logs_dir / f"batch_{batch_num:03d}.log",
    )


def print_controls() -> None:
    print(
        "[runner] controls: `s` = status, `r` = retry now after switching account/adding usage, "
        "`q` = graceful stop after current work, `qq` = stop immediately",
        flush=True,
    )


def command_loop(cmd_queue: "Queue[str]") -> None:
    while True:
        try:
            line = input().strip().lower()
        except EOFError:
            return
        cmd_queue.put(line)


def drain_commands(cmd_queue: "Queue[str]") -> list[str]:
    cmds: list[str] = []
    while True:
        try:
            cmds.append(cmd_queue.get_nowait())
        except Empty:
            return cmds


def prompt_lock_recovery(lock_path: Path, pid: int | None) -> str:
    pid_text = f" under pid {pid}" if pid else ""
    while True:
        print(
            f"\n[runner] another runner appears to own this queue{pid_text}.\n"
            f"Lock file: {lock_path}\n"
            "Options:\n"
            "  [Enter] quit and keep the existing runner/lock\n"
            "  f       force takeover if you know the old runner is dead or abandoned\n",
            flush=True,
        )
        choice = input("> ").strip().lower()
        if choice == "":
            return "quit"
        if choice == "f":
            return "force"


def format_summary(
    state: StateStore,
    batch_nums: list[int],
    active_batches: set[int],
) -> str:
    summary = state.summary(batch_nums)
    parts = [f"{k}={summary[k]}" for k in sorted(summary)]
    active = ",".join(f"{b:03d}" for b in sorted(active_batches)) or "-"
    next_retry_at = state.next_retry_at()
    retry_text = next_retry_at or "-"
    return (
        f"[runner] summary {' '.join(parts)} | active={active} | next_retry_at={retry_text}"
    )


def wait_until_retry_or_command(
    iso_ts: str,
    cmd_queue: "Queue[str]",
) -> str:
    target = datetime.fromisoformat(iso_ts)
    last_reported: int | None = None
    while True:
        for cmd in drain_commands(cmd_queue):
            if cmd in {"r", "retry"}:
                return "retry_now"
            if cmd in {"q", "quit"}:
                return "graceful_quit"
            if cmd == "qq":
                return "immediate_quit"

        now = datetime.now(TZ)
        remaining = int((target - now).total_seconds())
        if remaining <= 0:
            return "time_reached"
        if last_reported is None or remaining <= 10 or remaining // 30 != last_reported // 30:
            print(
                f"[runner] usage-capped; auto-resuming in {remaining}s "
                f"(type `r` to retry now or `q` to stop)",
                flush=True,
            )
            last_reported = remaining
        time.sleep(1)


def sleep_until(iso_ts: str) -> None:
    target = datetime.fromisoformat(iso_ts)
    last_reported: int | None = None
    while True:
        now = datetime.now(TZ)
        remaining = int((target - now).total_seconds())
        if remaining <= 0:
            return
        if last_reported is None or remaining <= 10 or remaining // 30 != last_reported // 30:
            print(
                f"[runner] usage-capped; auto-resuming in {remaining}s",
                flush=True,
            )
            last_reported = remaining
        time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--start-batch", type=int, default=DEFAULT_START_BATCH)
    parser.add_argument("--end-batch", type=int, default=DEFAULT_END_BATCH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--prompt-kind", default="local_first_pass")
    parser.add_argument("--auto-wait-for-usage", action="store_true")
    parser.add_argument("--clear-retry-at-start", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--force-if-stale-lock", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    run_root = (repo_root / args.run_root).resolve()
    worker_cwd = run_root / "_worker_cwd"
    worker_cwd.mkdir(parents=True, exist_ok=True)
    schema_path = (repo_root / args.schema_path).resolve()
    state = StateStore(run_root / args.state_file)
    lock_path = run_root / (args.state_file + ".lock")
    cmd_queue: "Queue[str]" = Queue()
    interactive = sys.stdin.isatty()
    stop_after_active = False
    immediate_stop = False
    active_batches: set[int] = set()

    batch_nums = list(range(args.start_batch, args.end_batch + 1))
    try:
        lock_cm = runner_lock(lock_path, force=args.force_if_stale_lock)
        lock_cm.__enter__()
    except RuntimeError as exc:
        if not sys.stdin.isatty():
            raise
        pid = None
        match = re.search(r"pid (\d+)", str(exc))
        if match:
            pid = int(match.group(1))
        action = prompt_lock_recovery(lock_path, pid)
        if action == "quit":
            return 3
        lock_cm = runner_lock(lock_path, force=True)
        lock_cm.__enter__()

    try:
        if interactive:
            print_controls()
            threading.Thread(target=command_loop, args=(cmd_queue,), daemon=True).start()
        state.reset_running_to_pending()
        for batch_num in batch_nums:
            paths = build_paths(run_root, args.results_dir, args.logs_dir, batch_num)
            state.ensure_batch(batch_num, paths.result_path, paths.log_path)
            expected_case_ids = {
                str(row["case_id"]) for row in json.loads(paths.input_path.read_text())
            }
            if paths.result_path.exists():
                try:
                    validate_result(paths.result_path, expected_case_ids)
                    state.mark_existing_complete(batch_num)
                    continue
                except Exception:
                    pass
            if try_recover_completed_output(paths, expected_case_ids):
                state.mark_existing_complete(batch_num)

        if args.clear_retry_at_start:
            state.clear_global_retry_at()

        if args.status_only:
            print(json.dumps({
                "summary": state.summary(batch_nums),
                "next_retry_at": state.next_retry_at(),
                "state_file": str(run_root / args.state_file),
            }, indent=2))
            return 0

        while True:
            if interactive:
                for cmd in drain_commands(cmd_queue):
                    if cmd in {"s", "status"}:
                        print(format_summary(state, batch_nums, active_batches), flush=True)
                    elif cmd in {"q", "quit"}:
                        stop_after_active = True
                        print("[runner] will stop after current active batches finish", flush=True)
                    elif cmd == "qq":
                        immediate_stop = True
                        print("[runner] immediate stop requested", flush=True)
                    elif cmd in {"r", "retry"}:
                        print("[runner] no usage wait active; ignoring `r`", flush=True)

            if immediate_stop:
                return 130

            pending = state.pending_batches(batch_nums)
            if not pending:
                print("[runner] all requested batches completed", flush=True)
                return 0
            if stop_after_active:
                print("[runner] stopped before launching new work; rerun the same command to resume", flush=True)
                return 2

            next_retry_at = state.next_retry_at()
            if next_retry_at:
                if args.auto_wait_for_usage:
                    sleep_until(next_retry_at)
                    state.clear_global_retry_at()
                elif sys.stdin.isatty():
                    action = wait_until_retry_or_command(next_retry_at, cmd_queue)
                    if action == "time_reached":
                        state.clear_global_retry_at()
                    elif action == "retry_now":
                        state.clear_global_retry_at()
                    elif action == "graceful_quit":
                        print("[runner] quitting during usage wait; rerun the same command to resume", flush=True)
                        return 2
                    else:
                        return 130
                else:
                    print(f"[runner] usage cap detected; rerun after {next_retry_at}", flush=True)
                    return 2

            active: dict[Future[dict[str, Any]], int] = {}
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                pending_iter = iter(pending)

                def submit_more() -> None:
                    while len(active) < args.concurrency and not stop_after_active and not immediate_stop:
                        try:
                            batch_num = next(pending_iter)
                        except StopIteration:
                            return
                        paths = build_paths(run_root, args.results_dir, args.logs_dir, batch_num)
                        state.start_attempt(batch_num)
                        active_batches.add(batch_num)
                        print(f"[runner] starting batch {batch_num:03d}", flush=True)
                        future = pool.submit(
                            run_one_batch,
                            repo_root,
                            worker_cwd,
                            schema_path,
                            paths,
                            args.timeout_seconds,
                            model=args.model,
                            reasoning_effort=args.reasoning_effort,
                            prompt_kind=args.prompt_kind,
                        )
                        active[future] = batch_num

                submit_more()
                usage_cap_hit = False
                while active:
                    if interactive:
                        for cmd in drain_commands(cmd_queue):
                            if cmd in {"s", "status"}:
                                print(format_summary(state, batch_nums, active_batches), flush=True)
                            elif cmd in {"q", "quit"}:
                                stop_after_active = True
                                print("[runner] will stop after current active batches finish", flush=True)
                            elif cmd == "qq":
                                immediate_stop = True
                                print("[runner] immediate stop requested", flush=True)
                            elif cmd in {"r", "retry"}:
                                print("[runner] retry-now only applies during a usage wait", flush=True)
                    if immediate_stop:
                        return 130
                    done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        batch_num = active.pop(future)
                        active_batches.discard(batch_num)
                        result = future.result()
                        state.finish_attempt(batch_num, result)
                        print(
                            f"[runner] batch {batch_num:03d} -> {result['status']} "
                            f"({result['elapsed_seconds']}s)",
                            flush=True,
                        )
                        if result["status"] == "rate_limited":
                            usage_cap_hit = True
                    if stop_after_active and not active:
                        print("[runner] stopped after active batches finished; rerun the same command to resume", flush=True)
                        return 2
                    if usage_cap_hit:
                        continue
                    submit_more()
                if usage_cap_hit:
                    continue
    finally:
        lock_cm.__exit__(None, None, None)


if __name__ == "__main__":
    sys.exit(main())
