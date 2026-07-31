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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Any
from zoneinfo import ZoneInfo


TZ = ZoneInfo("America/New_York")
USAGE_RETRY_RE = re.compile(r"try again at (\d{1,2}:\d{2} [AP]M)", re.IGNORECASE)


LOCAL_SECOND_PASS_PROMPT = """You are classifying one 5-case local second-pass batch for V2 jurisdiction semantics.

These are hard cases from the first-pass local queue. Each case includes `first_pass` as advisory context only; override it when the case payload or official sources support a better answer.

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
- Do not assume repeated names are duplicates; verify each ORI independently.
- Use provided exact_identity_rows and candidate_geographies as deterministic lookup inputs.
- Do not spend time reading local files to rediscover identity or candidate geometry already present in the case payload.
- Browse official public web sources only as needed to resolve jurisdiction semantics.
- Prefer resolving the case rather than escalating; use escalate only if you cannot defend any classification from the provided inputs and official sources.
- If municipal, resolved_geo_type must be place or cousub and resolved_geoid/resolved_label must be populated.
- If no defensible municipal target exists and the record does not clearly represent a special footprint, reclassify_nonmunicipal is an acceptable conservative fallback for an unresolved low-support local-like record.
- Keep reason to one sentence.
- Return only JSON matching the required schema.
"""

PA_OVERLAY = """
Pennsylvania-specific note:
- In Pennsylvania, borough, township, and city governments often resolve to municipal_cousub rather than municipal_place; use county context and official government structure.
"""

NONLOCAL_SECOND_PASS_PROMPT = """You are classifying one 5-case nonlocal/special-agency second-pass batch for V2 bucket semantics.

These are materially important special-agency or unknown cases. Each case includes `baseline_classification` as advisory context only; override it when the case payload or official sources support a better answer.

Classify each case into exactly one of:
- nonmunicipal_exclusive
- statewide_overlap
- localized_special_overlap
- exclude

Rules:
- Use nonmunicipal_exclusive only for ordinary county remainder or sheriff-style general-purpose nonmunicipal geography.
- Use localized_special_overlap for bounded or networked special footprints such as campus, transit, airport, port, tribal, housing, park, or other authority footprints.
- Use statewide_overlap for state police/highway patrol or unresolved special agencies that should remain in the statewide overlap layer.
- Preserve special agencies in overlap rather than pushing them into nonmunicipal remainder without evidence.
- If localized_special_overlap, set overlap_subtype to one of campus, transit, transport_hub, tribal, local_special, other_special and provide the best geometry_hint you can support.
- If not localized_special_overlap, set overlap_subtype to null. geometry_hint may be null or a short descriptor like statewide or county_remainder.
- Use provided exact_identity_rows as deterministic lookup inputs.
- Treat baseline_classification as advisory context; if it already matches the clear official semantics, keep it.
- Do not spend time reading local files to rediscover identity already present in the case payload.
- Do not spend time hunting FBI/CDE APIs, GitHub mirrors, or unofficial ORI rosters; use official public agency or government sources and keep the fallback bucket coherent when exact identity is still unclear.
- Browse official public web sources only as needed to resolve bucket semantics.
- Prefer 1-2 official sources unless genuinely ambiguous.
- Keep reason to one sentence.
- Return only JSON matching the required schema.
"""


@dataclass(frozen=True)
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


def build_prompt(task_type: str, batch_rows: list[dict[str, Any]]) -> str:
    if task_type == "local_second_pass":
        parts = [LOCAL_SECOND_PASS_PROMPT.strip()]
        if any((row.get("state_abbr") or "").upper() == "PA" for row in batch_rows):
            parts.append(PA_OVERLAY.strip())
    elif task_type == "nonlocal_second_pass":
        parts = [NONLOCAL_SECOND_PASS_PROMPT.strip()]
    else:
        raise ValueError(f"Unsupported task_type={task_type}")
    parts.append("Cases:")
    parts.append(json.dumps(batch_rows, ensure_ascii=True))
    return "\n\n".join(parts) + "\n"


def validate_local_results(payload: Any, expected_case_ids: set[str]) -> None:
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
        if row.get("decision") in {"municipal_place", "municipal_cousub"}:
            if row.get("resolved_geo_type") not in {"place", "cousub", "municipal_place", "municipal_cousub", "county_subdivision"}:
                raise ValueError(f"municipal row missing usable resolved_geo_type for {case_id}")
        seen.add(case_id)
    if seen != expected_case_ids:
        missing = sorted(expected_case_ids - seen)
        extra = sorted(seen - expected_case_ids)
        raise ValueError(f"case_id mismatch missing={missing} extra={extra}")


def canonicalize_local_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def validate_nonlocal_results(payload: Any, expected_case_ids: set[str]) -> None:
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
        if row.get("bucket_decision") == "localized_special_overlap" and not row.get("overlap_subtype"):
            raise ValueError(f"localized_special_overlap missing overlap_subtype for {case_id}")
        seen.add(case_id)
    if seen != expected_case_ids:
        missing = sorted(expected_case_ids - seen)
        extra = sorted(seen - expected_case_ids)
        raise ValueError(f"case_id mismatch missing={missing} extra={extra}")


def canonicalize_nonlocal_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in results:
        row = dict(row)
        if row.get("bucket_decision") != "localized_special_overlap":
            row["overlap_subtype"] = None
        out.append(row)
    return out


def normalize_output(task_type: str, raw_result_path: Path, result_path: Path, expected_case_ids: set[str]) -> None:
    raw_payload = json.loads(raw_result_path.read_text())
    results = raw_payload["results"]
    if task_type == "local_second_pass":
        results = canonicalize_local_results(results)
        validate_local_results(results, expected_case_ids)
    elif task_type == "nonlocal_second_pass":
        results = canonicalize_nonlocal_results(results)
        validate_nonlocal_results(results, expected_case_ids)
    else:
        raise ValueError(f"Unsupported task_type={task_type}")
    tmp_result = result_path.with_suffix(".tmp.json")
    tmp_result.write_text(json.dumps(results, indent=2) + "\n")
    tmp_result.replace(result_path)


def validate_existing_result(task_type: str, result_path: Path, expected_case_ids: set[str]) -> None:
    payload = json.loads(result_path.read_text())
    if task_type == "local_second_pass":
        validate_local_results(payload, expected_case_ids)
    else:
        validate_nonlocal_results(payload, expected_case_ids)


def try_recover_completed_output(task_type: str, paths: BatchPaths, expected_case_ids: set[str]) -> bool:
    for candidate in (paths.raw_tmp_path, paths.raw_result_path):
        if not candidate.exists():
            continue
        try:
            normalize_output(task_type, candidate, paths.result_path, expected_case_ids)
            if candidate == paths.raw_tmp_path:
                shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
            validate_existing_result(task_type, paths.result_path, expected_case_ids)
            return True
        except Exception:
            continue
    return False


def run_one_batch(
    task_type: str,
    worker_cwd: Path,
    schema_path: Path,
    model: str,
    reasoning_effort: str,
    paths: BatchPaths,
    timeout_seconds: int,
) -> dict[str, Any]:
    batch_rows = json.loads(paths.input_path.read_text())
    expected_case_ids = {str(row["case_id"]) for row in batch_rows}
    prompt = build_prompt(task_type, batch_rows)
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
        recovered = try_recover_completed_output(task_type, paths, expected_case_ids)
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
        if try_recover_completed_output(task_type, paths, expected_case_ids):
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
        if try_recover_completed_output(task_type, paths, expected_case_ids):
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
    normalize_output(task_type, paths.raw_result_path, paths.result_path, expected_case_ids)
    validate_existing_result(task_type, paths.result_path, expected_case_ids)
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
                "global": {"next_retry_at": None},
                "runner": {"lock_path": None},
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
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started_at": now_iso()}, indent=2) + "\n")
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


def format_summary(state: StateStore, batch_nums: list[int], active_batches: set[int]) -> str:
    summary = state.summary(batch_nums)
    parts = [f"{k}={summary[k]}" for k in sorted(summary)]
    active = ",".join(f"{b:03d}" for b in sorted(active_batches)) or "-"
    retry_text = state.next_retry_at() or "-"
    return f"[runner] summary {' '.join(parts)} | active={active} | next_retry_at={retry_text}"


def sleep_until(iso_ts: str) -> None:
    target = datetime.fromisoformat(iso_ts)
    while True:
        remaining = int((target - datetime.now(TZ)).total_seconds())
        if remaining <= 0:
            return
        print(f"[runner] usage-capped; auto-resuming in {remaining}s", flush=True)
        time.sleep(min(30, max(1, remaining)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", required=True, choices=["local_second_pass", "nonlocal_second_pass"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--start-batch", type=int, default=None)
    parser.add_argument("--end-batch", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--state-file", default="runner_state.json")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--auto-wait-for-usage", action="store_true")
    parser.add_argument("--clear-retry-at-start", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--force-if-stale-lock", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    run_root = args.run_root.resolve()
    worker_cwd = run_root / "_worker_cwd"
    worker_cwd.mkdir(parents=True, exist_ok=True)
    schema_path = repo_root / "scripts" / (
        "local_queue_output_schema.json" if args.task_type == "local_second_pass" else "nonlocal_queue_output_schema.json"
    )
    state = StateStore(run_root / args.state_file)
    lock_path = run_root / (args.state_file + ".lock")

    batch_files = sorted(run_root.glob("batch_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.json files found under {run_root}")
    all_batch_nums = [int(p.stem.split("_")[1]) for p in batch_files]
    start_batch = args.start_batch if args.start_batch is not None else min(all_batch_nums)
    end_batch = args.end_batch if args.end_batch is not None else max(all_batch_nums)
    batch_nums = list(range(start_batch, end_batch + 1))

    cmd_queue: "Queue[str]" = Queue()
    interactive = sys.stdin.isatty()
    stop_after_active = False
    immediate_stop = False
    active_batches: set[int] = set()

    with runner_lock(lock_path, force=args.force_if_stale_lock):
        if interactive:
            print_controls()
            threading.Thread(target=command_loop, args=(cmd_queue,), daemon=True).start()
        state.reset_running_to_pending()
        for batch_num in batch_nums:
            paths = build_paths(run_root, args.results_dir, args.logs_dir, batch_num)
            state.ensure_batch(batch_num, paths.result_path, paths.log_path)
            expected_case_ids = {str(row["case_id"]) for row in json.loads(paths.input_path.read_text())}
            if paths.result_path.exists():
                try:
                    validate_existing_result(args.task_type, paths.result_path, expected_case_ids)
                    state.mark_existing_complete(batch_num)
                    continue
                except Exception:
                    pass
            if try_recover_completed_output(args.task_type, paths, expected_case_ids):
                state.mark_existing_complete(batch_num)

        if args.clear_retry_at_start:
            state.clear_global_retry_at()

        if args.status_only:
            print(
                json.dumps(
                    {
                        "summary": state.summary(batch_nums),
                        "next_retry_at": state.next_retry_at(),
                        "state_file": str(run_root / args.state_file),
                    },
                    indent=2,
                )
            )
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
                        state.clear_global_retry_at()
                        print("[runner] retry-now requested", flush=True)

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
                            args.task_type,
                            worker_cwd,
                            schema_path,
                            args.model,
                            args.reasoning_effort,
                            paths,
                            args.timeout_seconds,
                        )
                        active[future] = batch_num

                submit_more()
                usage_cap_hit = False
                while active:
                    done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        batch_num = active.pop(future)
                        active_batches.discard(batch_num)
                        result = future.result()
                        state.finish_attempt(batch_num, result)
                        print(
                            f"[runner] batch {batch_num:03d} -> {result['status']} ({result['elapsed_seconds']}s)",
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


if __name__ == "__main__":
    sys.exit(main())
