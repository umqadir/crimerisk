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
DEFAULT_CONCURRENCY = 3
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_PROXY_MODEL = "gpt-5.4-mini"
DEFAULT_PROXY_REASONING_EFFORT = "medium"
DEFAULT_FOOTPRINT_MODEL = "gpt-5.4"
DEFAULT_FOOTPRINT_REASONING_EFFORT = "high"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().with_name("overlap_localization_review_output_schema.json")
DEFAULT_PROXY_QUEUE = "state/review/queues/overlap/overlap_localization_proxy_pilot.parquet"
DEFAULT_FOOTPRINT_QUEUE = "state/review/queues/overlap/overlap_localization_footprint_pilot.parquet"
DEFAULT_PROXY_RUN_ROOT = "state/review/runs/overlap/overlap_localization_proxy_run"
DEFAULT_FOOTPRINT_RUN_ROOT = "state/review/runs/overlap/overlap_localization_footprint_run"

PROXY_PROMPT = """You are resolving a small batch of overlap-localization review cases for V2.

Your job is to return:
- review_status: resolved or needs_escalation
- recommended_final_overlap_treatment

If review_status=resolved, allowed treatments are only:
- localize_to_place
- localize_to_county
- keep_statewide_overlap

Rules:
- Prefer official public sources when you need to verify service structure or footprint.
- Do not invent geography.
- Do not assume a place anchor is trustworthy just because place_fips exists; pseudo-place codes and network/system hints are weak anchors.
- For campus or institutional agencies, explicitly check whether the department serves multiple campuses, municipalities, or off-campus facilities before choosing a place proxy.
- If official sources indicate multi-campus or multi-city service, do not collapse to the branded/main campus place. Use county only if the full service area is still cleanly within one county; otherwise escalate.
- Use county only when it is a defensible sub-state proxy.
- Keep statewide overlap when the local proxy would be misleading.
- If the case appears to really require localize_to_custom_footprint or absorb_into_primary_jurisdiction, set review_status=needs_escalation instead of forcing a proxy answer.
- If localize_to_place, target_state_fips and target_place_fips must be populated.
- If localize_to_county, target_state_fips and target_county_fips must be populated.
- If keep_statewide_overlap, target geography fields should be null.
- Keep source_note to one short sentence.
- Prefer 1-3 official sources.
- Return only JSON matching the required schema.

Cases:
"""

FOOTPRINT_PROMPT = """You are resolving a small batch of high-impact overlap-localization review cases for V2.

Your job is to return:
- review_status: resolved or needs_escalation
- recommended_final_overlap_treatment

If review_status=resolved, allowed treatments are:
- absorb_into_primary_jurisdiction
- localize_to_place
- localize_to_county
- localize_to_custom_footprint
- keep_statewide_overlap
- exclude_or_hold

Rules:
- Use actual service structure and footprint logic when a generic place/county proxy would be materially misleading.
- Prefer official public sources when identifying facility, campus, transit, airport, port, tribal, or authority footprints.
- Do not invent geography.
- Use localize_to_custom_footprint when the correct treatment clearly depends on a real footprint rather than a generic place/county proxy.
- Use absorb_into_primary_jurisdiction only when the overlap should be represented as part of the primary jurisdiction system rather than a separate overlap layer.
- If you still cannot defend a treatment from the case payload plus official sources, set review_status=needs_escalation.
- If localize_to_place, target_state_fips and target_place_fips must be populated.
- If localize_to_county, target_state_fips and target_county_fips must be populated.
- If absorb_into_primary_jurisdiction and you can identify a target jurisdiction, populate target_jurisdiction_id.
- If localize_to_custom_footprint, footprint_type and geometry_source_type should be populated, and geometry_source_ref should identify the footprint source as concretely as possible.
- Keep source_note to one short sentence.
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


def build_prompt(review_kind: str, batch_rows: list[dict[str, Any]]) -> str:
    if review_kind == "proxy":
        base = PROXY_PROMPT
    elif review_kind == "footprint":
        base = FOOTPRINT_PROMPT
    else:
        raise ValueError(f"Unsupported review_kind={review_kind}")
    return base + json.dumps(batch_rows, ensure_ascii=True) + "\n"


def canonicalize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in results:
        item = dict(row)
        if "recommended_final_overlap_treatment" not in item and "final_overlap_treatment" in item:
            item["recommended_final_overlap_treatment"] = item.get("final_overlap_treatment")
        if "review_status" not in item:
            item["review_status"] = "needs_escalation" if bool(item.get("requires_escalation")) else "resolved"
        for key in ("target_state_fips", "target_county_fips", "target_place_fips"):
            value = item.get(key)
            if isinstance(value, str):
                digits = "".join(ch for ch in value if ch.isdigit())
                if key == "target_state_fips" and len(digits) >= 2:
                    item[key] = digits[-2:]
                elif key == "target_county_fips" and len(digits) >= 3:
                    item[key] = digits[-3:]
                elif key == "target_place_fips" and len(digits) >= 5:
                    item[key] = digits[-5:]
        if item.get("review_status") == "needs_escalation":
            item["recommended_final_overlap_treatment"] = None
            item["target_state_fips"] = None
            item["target_county_fips"] = None
            item["target_place_fips"] = None
            item["target_jurisdiction_id"] = None
        else:
            item["escalation_reason"] = None
        out.append(item)
    return out


def validate_payload(payload: Any, expected_case_ids: set[str], *, review_kind: str) -> None:
    if not isinstance(payload, list):
        raise ValueError("result payload is not a list")
    seen: set[str] = set()
    proxy_allowed = {"localize_to_place", "localize_to_county", "keep_statewide_overlap"}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("result row is not an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError("missing string case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id}")
        seen.add(case_id)

        if row.get("ori") != case_id:
            raise ValueError(f"{case_id} ori mismatch")
        review_status = row.get("review_status")
        if review_status not in {"resolved", "needs_escalation"}:
            raise ValueError(f"{case_id} invalid review_status")
        treatment = row.get("recommended_final_overlap_treatment")
        if review_status == "resolved":
            if review_kind == "proxy" and treatment not in proxy_allowed:
                raise ValueError(f"{case_id} invalid proxy treatment {treatment}")
            if review_kind == "footprint" and treatment not in {
                "absorb_into_primary_jurisdiction",
                "localize_to_place",
                "localize_to_county",
                "localize_to_custom_footprint",
                "keep_statewide_overlap",
                "exclude_or_hold",
            }:
                raise ValueError(f"{case_id} invalid footprint treatment {treatment}")
        else:
            if treatment is not None:
                raise ValueError(f"{case_id} escalation should not set recommended_final_overlap_treatment")
        if not isinstance(row.get("sources"), list):
            raise ValueError(f"{case_id} missing sources list")
        if not isinstance(row.get("confidence"), (int, float)):
            raise ValueError(f"{case_id} missing numeric confidence")
        if review_status == "needs_escalation" and not row.get("escalation_reason"):
            raise ValueError(f"{case_id} escalation missing escalation_reason")

        state_fips = row.get("target_state_fips")
        county_fips = row.get("target_county_fips")
        place_fips = row.get("target_place_fips")
        if treatment == "localize_to_place":
            if not (isinstance(state_fips, str) and len(state_fips) == 2 and isinstance(place_fips, str) and len(place_fips) == 5):
                raise ValueError(f"{case_id} place localization missing target fips")
        elif treatment == "localize_to_county":
            if not (isinstance(state_fips, str) and len(state_fips) == 2 and isinstance(county_fips, str) and len(county_fips) == 3):
                raise ValueError(f"{case_id} county localization missing target fips")
        elif treatment == "localize_to_custom_footprint":
            if review_kind != "footprint":
                raise ValueError(f"{case_id} custom footprint not allowed in proxy mode")
            if not row.get("footprint_type"):
                raise ValueError(f"{case_id} custom footprint missing footprint_type")
        elif treatment == "absorb_into_primary_jurisdiction":
            if review_kind != "footprint":
                raise ValueError(f"{case_id} absorb not allowed in proxy mode")
        elif treatment in {"keep_statewide_overlap", "exclude_or_hold"}:
            pass
        elif treatment is None and review_status == "needs_escalation":
            pass
        else:
            raise ValueError(f"{case_id} invalid treatment {treatment}")

    if seen != expected_case_ids:
        missing = sorted(expected_case_ids - seen)
        extra = sorted(seen - expected_case_ids)
        raise ValueError(f"case_id mismatch missing={missing} extra={extra}")


def normalize_output(raw_result_path: Path, result_path: Path, expected_case_ids: set[str], *, review_kind: str) -> None:
    raw_payload = json.loads(raw_result_path.read_text())
    results = canonicalize_results(raw_payload["results"])
    validate_payload(results, expected_case_ids, review_kind=review_kind)
    tmp_result = result_path.with_suffix(".tmp.json")
    tmp_result.write_text(json.dumps(results, indent=2) + "\n")
    tmp_result.replace(result_path)


def validate_existing_result(result_path: Path, expected_case_ids: set[str], *, review_kind: str) -> None:
    payload = json.loads(result_path.read_text())
    validate_payload(payload, expected_case_ids, review_kind=review_kind)


def try_recover_completed_output(paths: BatchPaths, expected_case_ids: set[str], *, review_kind: str) -> bool:
    for candidate in (paths.raw_tmp_path, paths.raw_result_path):
        if not candidate.exists():
            continue
        try:
            normalize_output(candidate, paths.result_path, expected_case_ids, review_kind=review_kind)
            if candidate == paths.raw_tmp_path:
                shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
            validate_existing_result(paths.result_path, expected_case_ids, review_kind=review_kind)
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
    review_kind: str,
    paths: BatchPaths,
    timeout_seconds: int,
) -> dict[str, Any]:
    batch_rows = json.loads(paths.input_path.read_text())
    expected_case_ids = {str(row["case_id"]) for row in batch_rows}
    prompt = build_prompt(review_kind, batch_rows)
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
        recovered = try_recover_completed_output(paths, expected_case_ids, review_kind=review_kind)
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
        if try_recover_completed_output(paths, expected_case_ids, review_kind=review_kind):
            return {"status": "completed", "retry_at": retry_at, "elapsed_seconds": round(time.time() - start, 2), "error": None}
        return {"status": "rate_limited", "retry_at": retry_at, "elapsed_seconds": round(time.time() - start, 2), "error": "usage_cap"}

    if proc.returncode != 0:
        if try_recover_completed_output(paths, expected_case_ids, review_kind=review_kind):
            return {"status": "completed", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": None}
        return {"status": "error", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": f"codex exec failed ({proc.returncode})"}

    if not paths.raw_tmp_path.exists() and not paths.raw_result_path.exists():
        try_write_raw_from_stdout(proc.stdout, paths.raw_result_path)
    if paths.raw_tmp_path.exists():
        shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
    if not paths.raw_result_path.exists():
        return {"status": "error", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": "missing_raw_result"}

    normalize_output(paths.raw_result_path, paths.result_path, expected_case_ids, review_kind=review_kind)
    validate_existing_result(paths.result_path, expected_case_ids, review_kind=review_kind)
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

    def attempts_for(self, batch_num: int) -> int:
        return int(self.state["batches"][f"{batch_num:03d}"].get("attempts", 0))


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


def _read_queue(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def build_batches(queue_path: Path, batch_dir: Path, *, batch_size: int) -> list[Path]:
    queue = _read_queue(queue_path).copy().reset_index(drop=True)
    if "case_id" not in queue.columns:
        queue["case_id"] = queue["ori"].astype("string")
    batch_dir.mkdir(parents=True, exist_ok=True)
    for stale in batch_dir.glob("batch_*.json"):
        stale.unlink()
    paths: list[Path] = []
    for idx in range(0, len(queue), batch_size):
        batch_num = idx // batch_size + 1
        batch_rows = []
        for _, row in queue.iloc[idx : idx + batch_size].iterrows():
            payload = row.where(pd.notna(row), None).to_dict()
            payload["case_id"] = str(payload["case_id"])
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
    parser = argparse.ArgumentParser(description="Run Codex exec worker pool for overlap localization review.")
    parser.add_argument("--review-kind", choices=["proxy", "footprint"], required=True)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--end-batch", type=int)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    default_queue = Path(DEFAULT_PROXY_QUEUE if args.review_kind == "proxy" else DEFAULT_FOOTPRINT_QUEUE)
    default_run_root = Path(DEFAULT_PROXY_RUN_ROOT if args.review_kind == "proxy" else DEFAULT_FOOTPRINT_RUN_ROOT)
    default_model = DEFAULT_PROXY_MODEL if args.review_kind == "proxy" else DEFAULT_FOOTPRINT_MODEL
    default_reasoning_effort = (
        DEFAULT_PROXY_REASONING_EFFORT if args.review_kind == "proxy" else DEFAULT_FOOTPRINT_REASONING_EFFORT
    )
    queue_path = (repo_root / (args.queue or default_queue)).resolve()
    run_root = (repo_root / (args.run_root or default_run_root)).resolve()
    schema_path = (repo_root / args.schema_path).resolve()
    model = args.model or default_model
    reasoning_effort = args.reasoning_effort or default_reasoning_effort

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
                validate_existing_result(bp.result_path, expected_case_ids, review_kind=args.review_kind)
                state.mark_existing_complete(batch_num)
            except Exception:
                pass

    if args.status_only:
        print(
            json.dumps(
                {
                    "review_kind": args.review_kind,
                    "queue": str(queue_path),
                    "run_root": str(run_root),
                    "summary": state.summary(batch_nums),
                },
                indent=2,
            )
        )
        return

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
            launch = [
                batch_num
                for batch_num in pending
                if state.attempts_for(batch_num) < int(args.max_attempts)
            ][: args.concurrency]
            if not launch:
                break
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for batch_num in launch:
                    bp = paths_for(batch_num, run_root)
                    state.start_attempt(batch_num)
                    futures[
                        pool.submit(
                            run_one_batch,
                            worker_cwd=repo_root,
                            schema_path=schema_path,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            review_kind=args.review_kind,
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
    print(
        json.dumps(
            {
                "review_kind": args.review_kind,
                "queue": str(queue_path),
                "run_root": str(run_root),
                "model": model,
                "reasoning_effort": reasoning_effort,
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
