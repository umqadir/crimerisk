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
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().with_name("reporting_regime_review_output_schema.json")
DEFAULT_QUEUE = "state/review/queues/reporting/reporting_regime_lumpy_pilot.parquet"
DEFAULT_RUN_ROOT = "state/review/runs/reporting/reporting_regime_homogeneous_lumpy_run"
DEFAULT_NIBRS_ONLY_QUEUE = "state/review/queues/reporting/reporting_regime_nibrs_only_pilot.parquet"
DEFAULT_NIBRS_ONLY_RUN_ROOT = "state/review/runs/reporting/reporting_regime_nibrs_only_run"
ALLOWED_REVIEW_STATUSES = {"resolved", "needs_escalation"}
ALLOWED_FINAL_REGIMES = {
    "full_monthly",
    "true_partial",
    "lumpy_or_batched",
    "annual_only_but_usable",
    "structurally_missing_or_unreliable",
}
ALLOWED_EVIDENCE_TYPES = {
    "queue_payload_lumpiness_signal",
    "queue_payload_nibrs_annual_support",
    "queue_payload_month_mask_mismatch",
    "queue_payload_annual_only_support",
    "official_monthly_publication",
    "official_open_data_portal",
    "official_methodology_or_policy",
    "official_agency_statement",
    "mixed_payload_and_official_source",
}

HOMOGENEOUS_LUMPY_PROMPT = """You are reviewing a small batch of reporting-regime cases for V2.

Every case in this batch comes from the homogeneous lumpy pilot lane:
- queue review_lane is srs_lumpiness_signal
- queue reporting_regime is currently lumpy_or_batched
- the automated flag was triggered by highly concentrated annual SRS totals

Return one result per case with:
- review_status: resolved or needs_escalation
- final_reporting_regime
- evidence_type

Allowed final_reporting_regime values for resolved cases:
- lumpy_or_batched
- annual_only_but_usable
- true_partial
- full_monthly
- structurally_missing_or_unreliable

Allowed evidence_type values:
- queue_payload_lumpiness_signal
- queue_payload_nibrs_annual_support
- queue_payload_month_mask_mismatch
- queue_payload_annual_only_support
- official_monthly_publication
- official_open_data_portal
- official_methodology_or_policy
- official_agency_statement
- mixed_payload_and_official_source

Rules:
- Treat the queue payload as the primary evidence. Do not browse unless the payload is insufficient to justify a reclassification.
- Keep lumpy_or_batched when the payload still looks like batched or highly concentrated monthly reporting.
- Use annual_only_but_usable when annual counts seem usable but the payload does not support a true lumpy monthly interpretation.
- Use true_partial only when there is defensible evidence of genuine partial monthly reporting rather than batching.
- Use full_monthly only when there is defensible evidence of actual full monthly reporting.
- Use structurally_missing_or_unreliable only when the data should not be treated as usable observed support.
- If you cannot defend a resolved classification from the payload plus any official sources, set review_status=needs_escalation.
- For resolved cases, final_reporting_regime, evidence_type, source_note, and reviewer_note must be populated, and escalation_reason must be null.
- For escalations, final_reporting_regime and evidence_type must be null, escalation_reason must be populated, and other narrative fields may be null.
- Keep source_note to one short sentence.
- reviewer_note should be short and concrete.
- Use [] for sources when queue payload alone is sufficient.
- Prefer at most 2 official sources.
- Return only JSON matching the required schema.

Cases:
"""

NIBRS_ONLY_PROMPT = """You are reviewing a small batch of reporting-regime cases for V2.

Every case in this batch comes from the NIBRS-only pilot lane:
- queue review_lane is nibrs_only
- SRS is absent or unusable for the offense-year
- the current automated classification usually treats these cases as annual_only_but_usable using NIBRS annual incident support

Return one result per case with:
- review_status: resolved or needs_escalation
- final_reporting_regime
- evidence_type

Allowed final_reporting_regime values for resolved cases:
- lumpy_or_batched
- annual_only_but_usable
- true_partial
- full_monthly
- structurally_missing_or_unreliable

Allowed evidence_type values:
- queue_payload_lumpiness_signal
- queue_payload_nibrs_annual_support
- queue_payload_month_mask_mismatch
- queue_payload_annual_only_support
- official_monthly_publication
- official_open_data_portal
- official_methodology_or_policy
- official_agency_statement
- mixed_payload_and_official_source

Rules:
- Treat the queue payload as the primary evidence. Do not browse unless the payload is insufficient to justify a reclassification.
- Use annual_only_but_usable when NIBRS annual support appears usable but there is no defensible evidence of true monthly observation.
- Prefer evidence_type=queue_payload_nibrs_annual_support when the queue payload alone is sufficient for that conclusion.
- Use full_monthly only when there is defensible evidence that the source is actually a monthly-usable NIBRS/open-data feed rather than an annual-only extraction.
- Use true_partial only when there is defensible evidence of genuine partial monthly reporting.
- Use structurally_missing_or_unreliable only when the NIBRS support should not be treated as usable observed support.
- lumpy_or_batched should be rare in this lane; use it only when the payload or official source clearly shows batched monthly-like reporting rather than annual support.
- If you cannot defend a resolved classification from the payload plus any official sources, set review_status=needs_escalation.
- For resolved cases, final_reporting_regime, evidence_type, source_note, and reviewer_note must be populated, and escalation_reason must be null.
- For escalations, final_reporting_regime and evidence_type must be null, escalation_reason must be populated, and other narrative fields may be null.
- Keep source_note to one short sentence.
- reviewer_note should be short and concrete.
- Use [] for sources when queue payload alone is sufficient.
- Prefer at most 2 official sources.
- Return only JSON matching the required schema.

Cases:
"""


@dataclass(frozen=True)
class LaneConfig:
    lane: str
    queue: str
    run_root: str
    prompt: str
    model: str
    reasoning_effort: str
    expected_review_lane: str
    expected_reporting_regime: str


LANE_CONFIGS: dict[str, LaneConfig] = {
    "homogeneous_lumpy": LaneConfig(
        lane="homogeneous_lumpy",
        queue=DEFAULT_QUEUE,
        run_root=DEFAULT_RUN_ROOT,
        prompt=HOMOGENEOUS_LUMPY_PROMPT,
        model=DEFAULT_MODEL,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        expected_review_lane="srs_lumpiness_signal",
        expected_reporting_regime="lumpy_or_batched",
    ),
    "nibrs_only": LaneConfig(
        lane="nibrs_only",
        queue=DEFAULT_NIBRS_ONLY_QUEUE,
        run_root=DEFAULT_NIBRS_ONLY_RUN_ROOT,
        prompt=NIBRS_ONLY_PROMPT,
        model=DEFAULT_MODEL,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        expected_review_lane="nibrs_only",
        expected_reporting_regime="annual_only_but_usable",
    ),
}


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


def _none_if_blank(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def build_prompt(lane_config: LaneConfig, batch_rows: list[dict[str, Any]]) -> str:
    return lane_config.prompt + json.dumps(batch_rows, ensure_ascii=True) + "\n"


def canonicalize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in results:
        item = dict(row)
        if "final_reporting_regime" not in item and "recommended_final_reporting_regime" in item:
            item["final_reporting_regime"] = item.get("recommended_final_reporting_regime")
        item["case_id"] = str(item.get("case_id", ""))
        item["ori"] = str(item.get("ori", ""))
        offense = item.get("offense")
        if offense is not None:
            item["offense"] = str(offense)
        year = item.get("year")
        if isinstance(year, str) and year.strip():
            try:
                item["year"] = int(year)
            except ValueError:
                pass
        if "review_status" not in item:
            item["review_status"] = "resolved" if item.get("final_reporting_regime") else "needs_escalation"
        for key in ("final_reporting_regime", "evidence_type", "source_note", "reviewer_note", "escalation_reason"):
            item[key] = _none_if_blank(item.get(key))
        confidence = item.get("confidence")
        if isinstance(confidence, str) and confidence.strip():
            try:
                item["confidence"] = float(confidence)
            except ValueError:
                pass
        sources = item.get("sources")
        if sources is None:
            item["sources"] = []
        if item.get("review_status") == "needs_escalation":
            item["final_reporting_regime"] = None
            item["evidence_type"] = None
            item["source_note"] = item.get("source_note")
        else:
            item["escalation_reason"] = None
        out.append(item)
    return out


def validate_payload(
    payload: Any,
    expected_cases: dict[str, tuple[str, int, str]],
    *,
    lane_config: LaneConfig,
) -> None:
    if not isinstance(payload, list):
        raise ValueError("result payload is not a list")
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("result row is not an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("missing string case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id}")
        if case_id not in expected_cases:
            raise ValueError(f"unexpected case_id {case_id}")
        seen.add(case_id)

        expected_ori, expected_year, expected_offense = expected_cases[case_id]
        if row.get("ori") != expected_ori:
            raise ValueError(f"{case_id} ori mismatch")
        if row.get("year") != expected_year:
            raise ValueError(f"{case_id} year mismatch")
        if row.get("offense") != expected_offense:
            raise ValueError(f"{case_id} offense mismatch")

        review_status = row.get("review_status")
        if review_status not in ALLOWED_REVIEW_STATUSES:
            raise ValueError(f"{case_id} invalid review_status")
        final_reporting_regime = row.get("final_reporting_regime")
        evidence_type = row.get("evidence_type")
        if review_status == "resolved":
            if final_reporting_regime not in ALLOWED_FINAL_REGIMES:
                raise ValueError(f"{case_id} invalid final_reporting_regime {final_reporting_regime}")
            if evidence_type not in ALLOWED_EVIDENCE_TYPES:
                raise ValueError(f"{case_id} invalid evidence_type {evidence_type}")
            if not isinstance(row.get("source_note"), str) or not row.get("source_note", "").strip():
                raise ValueError(f"{case_id} missing source_note")
            if not isinstance(row.get("reviewer_note"), str) or not row.get("reviewer_note", "").strip():
                raise ValueError(f"{case_id} missing reviewer_note")
            if row.get("escalation_reason") is not None:
                raise ValueError(f"{case_id} resolved row must not set escalation_reason")
        else:
            if final_reporting_regime is not None:
                raise ValueError(f"{case_id} escalation should not set final_reporting_regime")
            if evidence_type is not None:
                raise ValueError(f"{case_id} escalation should not set evidence_type")
            if not isinstance(row.get("escalation_reason"), str) or not row.get("escalation_reason", "").strip():
                raise ValueError(f"{case_id} escalation missing escalation_reason")

        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"{case_id} missing numeric confidence")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"{case_id} confidence outside [0, 1]")
        sources = row.get("sources")
        if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
            raise ValueError(f"{case_id} missing sources list")

    if seen != set(expected_cases):
        missing = sorted(set(expected_cases) - seen)
        extra = sorted(seen - set(expected_cases))
        raise ValueError(f"case_id mismatch missing={missing} extra={extra}")


def normalize_output(
    raw_result_path: Path,
    result_path: Path,
    expected_cases: dict[str, tuple[str, int, str]],
    *,
    lane_config: LaneConfig,
) -> None:
    raw_payload = json.loads(raw_result_path.read_text())
    results = canonicalize_results(raw_payload["results"])
    validate_payload(results, expected_cases, lane_config=lane_config)
    tmp_result = result_path.with_suffix(".tmp.json")
    tmp_result.write_text(json.dumps(results, indent=2) + "\n")
    tmp_result.replace(result_path)


def validate_existing_result(
    result_path: Path,
    expected_cases: dict[str, tuple[str, int, str]],
    *,
    lane_config: LaneConfig,
) -> None:
    payload = json.loads(result_path.read_text())
    validate_payload(payload, expected_cases, lane_config=lane_config)


def try_recover_completed_output(
    paths: BatchPaths,
    expected_cases: dict[str, tuple[str, int, str]],
    *,
    lane_config: LaneConfig,
) -> bool:
    for candidate in (paths.raw_tmp_path, paths.raw_result_path):
        if not candidate.exists():
            continue
        try:
            normalize_output(candidate, paths.result_path, expected_cases, lane_config=lane_config)
            if candidate == paths.raw_tmp_path:
                shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
            validate_existing_result(paths.result_path, expected_cases, lane_config=lane_config)
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
    lane_config: LaneConfig,
    model: str,
    reasoning_effort: str,
    paths: BatchPaths,
    timeout_seconds: int,
) -> dict[str, Any]:
    batch_rows = json.loads(paths.input_path.read_text())
    expected_cases = {
        str(row["case_id"]): (str(row["ori"]), int(row["year"]), str(row["offense"]))
        for row in batch_rows
    }
    prompt = build_prompt(lane_config, batch_rows)
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
        recovered = try_recover_completed_output(paths, expected_cases, lane_config=lane_config)
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
        if try_recover_completed_output(paths, expected_cases, lane_config=lane_config):
            return {"status": "completed", "retry_at": retry_at, "elapsed_seconds": round(time.time() - start, 2), "error": None}
        return {"status": "rate_limited", "retry_at": retry_at, "elapsed_seconds": round(time.time() - start, 2), "error": "usage_cap"}

    if proc.returncode != 0:
        if try_recover_completed_output(paths, expected_cases, lane_config=lane_config):
            return {"status": "completed", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": None}
        return {"status": "error", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": f"codex exec failed ({proc.returncode})"}

    if not paths.raw_tmp_path.exists() and not paths.raw_result_path.exists():
        try_write_raw_from_stdout(proc.stdout, paths.raw_result_path)
    if paths.raw_tmp_path.exists():
        shutil.move(str(paths.raw_tmp_path), str(paths.raw_result_path))
    if not paths.raw_result_path.exists():
        return {"status": "error", "retry_at": None, "elapsed_seconds": round(time.time() - start, 2), "error": "missing_raw_result"}

    normalize_output(paths.raw_result_path, paths.result_path, expected_cases, lane_config=lane_config)
    validate_existing_result(paths.result_path, expected_cases, lane_config=lane_config)
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


def _validate_queue(queue: pd.DataFrame, lane_config: LaneConfig) -> pd.DataFrame:
    required = {"ori", "year", "offense", "review_lane", "reporting_regime"}
    missing = sorted(required - set(queue.columns))
    if missing:
        raise ValueError(f"Queue missing columns: {missing}")
    out = queue.copy().reset_index(drop=True)
    out["ori"] = out["ori"].astype("string")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["offense"] = out["offense"].astype("string")
    out["review_lane"] = out["review_lane"].astype("string")
    out["reporting_regime"] = out["reporting_regime"].astype("string")
    if "case_id" not in out.columns:
        out["case_id"] = out["ori"] + ":" + out["year"].astype("string") + ":" + out["offense"]
    out["case_id"] = out["case_id"].astype("string")
    invalid_lane = out.loc[out["review_lane"].ne(lane_config.expected_review_lane), ["case_id", "review_lane"]]
    if not invalid_lane.empty:
        raise ValueError(f"Queue contains unexpected review_lane values: {invalid_lane.head(10).to_dict(orient='records')}")
    invalid_regime = out.loc[
        out["reporting_regime"].ne(lane_config.expected_reporting_regime),
        ["case_id", "reporting_regime"],
    ]
    if not invalid_regime.empty:
        raise ValueError(
            f"Queue contains unexpected reporting_regime values: {invalid_regime.head(10).to_dict(orient='records')}"
        )
    dupes = out.loc[out.duplicated(["case_id"], keep=False), ["case_id", "ori", "year", "offense"]]
    if not dupes.empty:
        raise ValueError(f"Queue contains duplicate case_id values: {dupes.head(10).to_dict(orient='records')}")
    if out["year"].isna().any():
        raise ValueError("Queue contains non-numeric year values")
    return out


def build_batches(queue_path: Path, batch_dir: Path, *, batch_size: int, lane_config: LaneConfig) -> list[Path]:
    queue = _validate_queue(_read_queue(queue_path), lane_config)
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
            payload["ori"] = str(payload["ori"])
            payload["year"] = int(payload["year"])
            payload["offense"] = str(payload["offense"])
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


def _expected_cases_for_batch(batch_path: Path) -> dict[str, tuple[str, int, str]]:
    batch_rows = json.loads(batch_path.read_text())
    return {
        str(row["case_id"]): (str(row["ori"]), int(row["year"]), str(row["offense"]))
        for row in batch_rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Codex exec worker pool for reporting-regime review.")
    parser.add_argument("--lane", choices=sorted(LANE_CONFIGS), default="homogeneous_lumpy")
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
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    lane_config = LANE_CONFIGS[args.lane]
    queue_path = (repo_root / (args.queue or Path(lane_config.queue))).resolve()
    run_root = (repo_root / (args.run_root or Path(lane_config.run_root))).resolve()
    schema_path = (repo_root / args.schema_path).resolve()
    model = args.model or lane_config.model
    reasoning_effort = args.reasoning_effort or lane_config.reasoning_effort

    batch_paths = build_batches(queue_path, run_root / "batches", batch_size=args.batch_size, lane_config=lane_config)
    total_batches = len(batch_paths)
    end_batch = args.end_batch or total_batches
    batch_nums = list(range(args.start_batch, end_batch + 1))

    state = StateStore(run_root / "state.json")
    for batch_num in batch_nums:
        bp = paths_for(batch_num, run_root)
        state.ensure_batch(batch_num, bp.result_path, bp.log_path)
        if bp.result_path.exists():
            expected_cases = _expected_cases_for_batch(bp.input_path)
            try:
                validate_existing_result(bp.result_path, expected_cases, lane_config=lane_config)
                state.mark_existing_complete(batch_num)
            except Exception:
                pass

    summary_payload = {
        "lane": lane_config.lane,
        "queue": str(queue_path),
        "run_root": str(run_root),
        "total_batches": total_batches,
        "selected_batches": batch_nums,
        "summary": state.summary(batch_nums),
    }
    if args.status_only or args.prepare_only:
        print(json.dumps(summary_payload, indent=2))
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
                            lane_config=lane_config,
                            model=model,
                            reasoning_effort=reasoning_effort,
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
                "lane": lane_config.lane,
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
