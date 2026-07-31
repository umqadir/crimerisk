from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from pathlib import Path


BASE_PROMPT = """You are classifying one 5-case local-queue batch for V2 jurisdiction semantics.

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
- Browse official public web sources only as needed to resolve jurisdiction semantics.
- Prefer 1-2 official sources unless genuinely ambiguous.
- If municipal, return the best resolved_geo_type, resolved_geoid, and resolved_label you can support. Otherwise set those fields to null.
- Keep reason to one sentence.
- Return only a JSON array matching the required schema.
"""


PA_OVERLAY = """
Pennsylvania-specific note:
- In Pennsylvania, borough, township, and city governments often resolve to municipal_cousub rather than municipal_place; use county context and official government structure.
"""


def build_prompt(batch_rows: list[dict]) -> str:
    has_pa = any((row.get("state_abbr") or "").lower() == "pa" for row in batch_rows)
    parts = [BASE_PROMPT.strip()]
    if has_pa:
        parts.append(PA_OVERLAY.strip())
    parts.append("Cases:")
    parts.append(json.dumps(batch_rows, ensure_ascii=True))
    return "\n\n".join(parts) + "\n"


def run_batch(
    repo_root: Path,
    schema_path: Path,
    batch_path: Path,
    out_path: Path,
    log_path: Path,
    timeout_seconds: int,
) -> tuple[str, int]:
    batch_rows = json.loads(batch_path.read_text())
    prompt = build_prompt(batch_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out_path = out_path.with_suffix(".raw.json")
    cmd = [
        "codex",
        "exec",
        "-",
        "--sandbox",
        "read-only",
        "--cd",
        str(repo_root),
        "--model",
        "gpt-5.4-mini",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(raw_out_path),
        "--config",
        'model_reasoning_effort="medium"',
        "--config",
        'web_search="live"',
        "--config",
        'approval_policy="never"',
    ]
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=repo_root,
        env=env,
        timeout=timeout_seconds,
    )
    log_path.write_text(
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
    return batch_path.stem, proc.returncode


def validate_json(out_path: Path) -> None:
    payload = json.loads(out_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"expected list output in {out_path}")


def normalize_output(raw_out_path: Path, out_path: Path) -> None:
    raw_payload = json.loads(raw_out_path.read_text())
    results = raw_payload["results"]
    out_path.write_text(json.dumps(results, indent=2) + "\n")


def worker_loop(
    repo_root: Path,
    schema_path: Path,
    jobs: "queue.Queue[tuple[Path, Path, Path]]",
    failures: list[str],
    lock: threading.Lock,
    timeout_seconds: int,
) -> None:
    while True:
        try:
            batch_path, out_path, log_path = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            _, returncode = run_batch(
                repo_root,
                schema_path,
                batch_path,
                out_path,
                log_path,
                timeout_seconds,
            )
            raw_out_path = out_path.with_suffix(".raw.json")
            if returncode != 0:
                with lock:
                    failures.append(f"{batch_path.name}: codex exec failed ({returncode})")
            else:
                normalize_output(raw_out_path, out_path)
                validate_json(out_path)
        except subprocess.TimeoutExpired:
            with lock:
                failures.append(f"{batch_path.name}: codex exec timed out")
        except Exception as exc:  # noqa: BLE001
            with lock:
                failures.append(f"{batch_path.name}: {exc}")
        finally:
            jobs.task_done()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-batch", type=int, required=True)
    parser.add_argument("--end-batch", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    run_root = repo_root / "state" / "review" / "runs" / "local_resolution" / "local_queue_clean_run"
    results_root = run_root / "results_cli_trial"
    logs_root = run_root / "logs_cli_trial"
    schema_path = Path(__file__).resolve().with_name("local_queue_output_schema.json")

    jobs: "queue.Queue[tuple[Path, Path, Path]]" = queue.Queue()
    for batch_num in range(args.start_batch, args.end_batch + 1):
        batch_path = run_root / f"batch_{batch_num:03d}.json"
        out_path = results_root / f"batch_{batch_num:03d}_results.json"
        log_path = logs_root / f"batch_{batch_num:03d}.log"
        jobs.put((batch_path, out_path, log_path))

    failures: list[str] = []
    lock = threading.Lock()
    threads = []
    for _ in range(args.concurrency):
        thread = threading.Thread(
            target=worker_loop,
            args=(repo_root, schema_path, jobs, failures, lock, args.timeout_seconds),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
