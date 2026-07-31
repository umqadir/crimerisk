from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


TZ = ZoneInfo("America/New_York")
USAGE_RETRY_RE = re.compile(r"try again at (\d{1,2}:\d{2} [AP]M)", re.IGNORECASE)

DEFAULT_BATCH_SIZE = 4
DEFAULT_CONCURRENCY = 4
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_RUN_ROOT = "state/review/runs/municipal/unsupported_municipal_geometry_run"
DEFAULT_QUEUE = "state/review/queues/municipal/unsupported_municipal_geometry_queue.parquet"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().with_name("municipal_geometry_resolution_output_schema.json")


PROMPT = """You are resolving a small batch of unsupported municipal geometry cases for V2.

Each case currently has:
- a positive 2024 municipal control total
- an agency/jurisdiction mapping in the reference layer
- no defensible block-group support in the current geometry/reference build

Your task is to decide whether the case should:
- map_to_place
- map_to_cousub
- exclude
- escalate

Rules:
- Do not invent geography.
- Do not snap to nearby places, nearby tracts, or nearby block groups.
- Only map a case if official public sources support a current Census/TIGER municipal geography or a clear current successor geography.
- Prefer official sources: Census geography pages/files, state/local government pages, municipal charters, county/state reorganization records, or other official government sources.
- Use exclude when the record is a dissolved, annexed, legacy, or otherwise non-current municipality with no separate current geography that should carry its own 2024 municipal surface.
- Use escalate only if you cannot defend either a current mapped geography or exclusion from official sources.
- If decision is map_to_place or map_to_cousub, replacement_geo_type / replacement_geoid / replacement_jurisdiction_name must be populated.
- If decision is exclude or escalate, replacement fields must be null.
- Keep reason to one sentence.
- Prefer 1-3 official sources.
- Return only JSON matching the required schema.

Cases:
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


def validate_payload(payload: Any, expected_case_ids: set[str]) -> None:
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
        decision = row.get("decision")
        if decision in {"map_to_place", "map_to_cousub"}:
            expected_type = "place" if decision == "map_to_place" else "cousub"
            if row.get("replacement_geo_type") != expected_type:
                raise ValueError(f"{case_id} missing replacement_geo_type={expected_type}")
            if not row.get("replacement_geoid") or not row.get("replacement_jurisdiction_name"):
                raise ValueError(f"{case_id} missing replacement fields")
        elif decision in {"exclude", "escalate"}:
            if row.get("replacement_geo_type") is not None or row.get("replacement_geoid") is not None or row.get("replacement_jurisdiction_name") is not None:
                raise ValueError(f"{case_id} exclusion/escalation should not set replacement fields")
        else:
            raise ValueError(f"{case_id} invalid decision")
        seen.add(case_id)
    if seen != expected_case_ids:
        missing = sorted(expected_case_ids - seen)
        extra = sorted(seen - expected_case_ids)
        raise ValueError(f"case_id mismatch missing={missing} extra={extra}")


def canonicalize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in results:
        row = dict(row)
        if row.get("decision") == "map_to_place":
            row["replacement_geo_type"] = "place"
        elif row.get("decision") == "map_to_cousub":
            row["replacement_geo_type"] = "cousub"
        geoid = row.get("replacement_geoid")
        if isinstance(geoid, str) and row.get("replacement_geo_type") == "place":
            digits = "".join(ch for ch in geoid if ch.isdigit())
            if len(digits) >= 7:
                row["replacement_geoid"] = digits[-7:]
        elif isinstance(geoid, str) and row.get("replacement_geo_type") == "cousub":
            digits = "".join(ch for ch in geoid if ch.isdigit())
            if len(digits) >= 10:
                row["replacement_geoid"] = digits[-10:]
        out.append(row)
    return out


def normalize_output(raw_result_path: Path, result_path: Path, expected_case_ids: set[str]) -> None:
    raw_payload = json.loads(raw_result_path.read_text())
    results = canonicalize_results(raw_payload["results"])
    validate_payload(results, expected_case_ids)
    tmp_result = result_path.with_suffix(".tmp.json")
    tmp_result.write_text(json.dumps(results, indent=2) + "\n")
    tmp_result.replace(result_path)


def validate_existing_result(result_path: Path, expected_case_ids: set[str]) -> None:
    payload = json.loads(result_path.read_text())
    validate_payload(payload, expected_case_ids)


def try_recover_completed_output(paths: BatchPaths, expected_case_ids: set[str]) -> bool:
    for candidate in (paths.raw_tmp_path, paths.raw_result_path):
        if not candidate.exists():
            continue
        try:
            normalize_output(candidate, paths.result_path, expected_case_ids)
            if candidate == paths.raw_tmp_path:
                shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
            validate_existing_result(paths.result_path, expected_case_ids)
            return True
        except Exception:
            continue
    return False


def try_write_raw_from_stdout(stdout: str, raw_path: Path) -> bool:
    text = (stdout or "").strip()
    if not text:
        return False
    try:
        payload = json.loads(text)
    except Exception:
        return False
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2) + "\n")
    return True


def run_one_batch(
    *,
    worker_cwd: Path,
    schema_path: Path,
    model: str,
    reasoning_effort: str,
    paths: BatchPaths,
    timeout_seconds: int,
) -> dict[str, Any]:
    batch_rows = json.loads(paths.input_path.read_text())
    expected_case_ids = {str(row["case_id"]) for row in batch_rows}
    prompt = PROMPT + json.dumps(batch_rows, ensure_ascii=True) + "\n"
    paths.result_path.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_result_path.parent.mkdir(parents=True, exist_ok=True)
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
            return {"status": "completed", "retry_at": retry_at, "elapsed_seconds": round(time.time() - start, 2), "error": None}
        return {"status": "rate_limited", "retry_at": retry_at, "elapsed_seconds": round(time.time() - start, 2), "error": "usage_cap"}

    if proc.returncode != 0:
        if try_recover_completed_output(paths, expected_case_ids):
            return {"status": "completed", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": None}
        return {"status": "error", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": f"codex exec failed ({proc.returncode})"}

    if not paths.raw_tmp_path.exists() and not paths.raw_result_path.exists():
        try_write_raw_from_stdout(proc.stdout, paths.raw_result_path)
    if paths.raw_tmp_path.exists():
        shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
    if not paths.raw_result_path.exists():
        return {
            "status": "error",
            "retry_at": None,
            "elapsed_seconds": round(time.time() - start, 2),
            "error": "missing_raw_result",
        }
    normalize_output(paths.raw_result_path, paths.result_path, expected_case_ids)
    validate_existing_result(paths.result_path, expected_case_ids)
    return {"status": "completed", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": None}


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        if path.exists():
            self.state = json.loads(path.read_text())
        else:
            self.state = {"created_at": now_iso(), "updated_at": now_iso(), "global": {"next_retry_at": None}, "batches": {}}
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
            entry["completed_at"] = now_iso()
            entry["updated_at"] = now_iso()
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

    def pending_batches(self, batch_nums: list[int]) -> list[int]:
        return [b for b in batch_nums if self.state["batches"][f"{b:03d}"]["status"] != "completed"]

    def next_retry_at(self) -> str | None:
        return self.state["global"]["next_retry_at"]

    def clear_global_retry_at(self) -> None:
        with self.lock:
            self.state["global"]["next_retry_at"] = None
            self.state["updated_at"] = now_iso()
            self._flush()

    def summary(self, batch_nums: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for batch_num in batch_nums:
            status = self.state["batches"][f"{batch_num:03d}"]["status"]
            counts[status] = counts.get(status, 0) + 1
        return counts


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextmanager
def runner_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text())
            pid = int(payload["pid"])
        except Exception:
            pid = None
        if pid and pid_is_alive(pid):
            raise RuntimeError(f"runner already active under pid {pid}")
        lock_path.unlink(missing_ok=True)
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started_at": now_iso()}, indent=2) + "\n")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def build_batches(queue_path: Path, batch_dir: Path, *, batch_size: int) -> list[Path]:
    queue = pd.read_parquet(queue_path)
    queue = queue[queue["review_disposition"].eq("manual_review_required")].copy()
    queue = queue.reset_index(drop=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(0, len(queue), batch_size):
        batch_num = idx // batch_size + 1
        batch_rows = []
        for _, row in queue.iloc[idx : idx + batch_size].iterrows():
            payload = row.to_dict()
            payload["case_id"] = str(row["jurisdiction_id"])
            batch_rows.append(payload)
        path = batch_dir / f"batch_{batch_num:03d}.json"
        path.write_text(json.dumps(batch_rows, indent=2) + "\n")
        paths.append(path)
    return paths


def paths_for(batch_num: int, run_root: Path) -> BatchPaths:
    return BatchPaths(
        batch_num=batch_num,
        input_path=run_root / "batches" / f"batch_{batch_num:03d}.json",
        result_path=run_root / "results" / f"batch_{batch_num:03d}.json",
        raw_result_path=run_root / "raw_results" / f"batch_{batch_num:03d}.json",
        log_path=run_root / "logs" / f"batch_{batch_num:03d}.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Codex exec worker pool for unsupported municipal geometry review.")
    parser.add_argument("--queue", type=Path, default=Path(DEFAULT_QUEUE))
    parser.add_argument("--run-root", type=Path, default=Path(DEFAULT_RUN_ROOT))
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--end-batch", type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    run_root = (repo_root / args.run_root).resolve()
    schema_path = (repo_root / args.schema_path).resolve()
    queue_path = (repo_root / args.queue).resolve()
    batch_paths = build_batches(queue_path, run_root / "batches", batch_size=args.batch_size)
    total_batches = len(batch_paths)
    end_batch = args.end_batch or total_batches
    batch_nums = list(range(args.start_batch, end_batch + 1))

    state = StateStore(run_root / "state.json")
    for batch_num in batch_nums:
        bp = paths_for(batch_num, run_root)
        state.ensure_batch(batch_num, bp.result_path, bp.log_path)
        if bp.result_path.exists():
            batch_rows = json.loads(bp.input_path.read_text())
            expected_case_ids = {str(row["case_id"]) for row in batch_rows}
            try:
                validate_existing_result(bp.result_path, expected_case_ids)
                state.mark_existing_complete(batch_num)
            except Exception:
                pass

    with runner_lock(run_root / "runner.lock"):
        while True:
            pending = state.pending_batches(batch_nums)
            if not pending:
                break
            retry_at = state.next_retry_at()
            if retry_at:
                retry_dt = datetime.fromisoformat(retry_at)
                now_dt = datetime.now(TZ)
                if retry_dt > now_dt:
                    time.sleep(min((retry_dt - now_dt).total_seconds(), 60))
                    continue
                state.clear_global_retry_at()

            futures: dict[Future[dict[str, Any]], int] = {}
            launch = pending[: args.concurrency]
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for batch_num in launch:
                    bp = paths_for(batch_num, run_root)
                    state.start_attempt(batch_num)
                    futures[
                        pool.submit(
                            run_one_batch,
                            worker_cwd=repo_root,
                            schema_path=schema_path,
                            model=args.model,
                            reasoning_effort=args.reasoning_effort,
                            paths=bp,
                            timeout_seconds=args.timeout_seconds,
                        )
                    ] = batch_num

                while futures:
                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        batch_num = futures.pop(future)
                        result = future.result()
                        state.finish_attempt(batch_num, result)

    summary = state.summary(batch_nums)
    print(json.dumps({"run_root": str(run_root), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
