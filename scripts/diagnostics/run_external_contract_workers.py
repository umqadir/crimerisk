from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


PROMPT_TEMPLATE = """You are an external Codex exec worker in the CrimeRisk repo.

Task: produce one declarative generic-city onboarding contract for {city_name}, 2024.

Hard constraints:
- Do not edit files.
- Do not run broad repo-wide grep/find over the whole tree. Read only the schema/runner if needed,
  and use short targeted `uv run python` snippets for local controls/bounds. Do not use bare
  `python`.
- Use online research and local repo files as needed, but keep source discovery bounded. Prefer the
  first official runnable feed that has date, offense/category/code, stable row identity, and point
  coordinates or ArcGIS geometry.
- This is a share-lane contract only. Do not promote raw feed counts as annual totals.
- Output final response as JSON only, matching scripts/diagnostics/generic_city_contract_schema.json
  exactly.

City:
- city_key: {city_key}
- city_name: {city_name}
- expected state: {state_abbr}
- source hint, if useful: {source_hint}

Contract requirements:
- Determine the repo jurisdiction_id and state_fips from state/controls/jurisdiction_controls_2024.parquet.
- Compute broad latitude/longitude bounds from local TIGER/crosswalk data or a reliable official
  boundary source. The runner will spatially join to municipal block groups.
- source.portal_type must be one of: socrata, arcgis_feature_service, csv.
- For Socrata, api_endpoint should be the /resource/<id>.json endpoint and where/order/select
  should be valid SoQL.
- For ArcGIS, api_endpoint should be a FeatureServer layer endpoint or /query endpoint; where/order
  should be valid ArcGIS query parameters. The runner exposes geometry as __arcgis_latitude and
  __arcgis_longitude and handles epoch-ms date fields.
- For CSV, api_endpoint should be a direct CSV URL; where/order can be empty strings.
- Keep select fields minimal: stable id, date, offense/category/code, coordinate or geometry fields,
  and small useful context fields. Avoid large text blobs.
- Always include mapping.id_fields. Use [] when mapping.id_field alone is sufficient.
- Every condition object must include field, op, value, and values. Use value "" when op uses values;
  use values [] when op uses value.
- offense_rules should map only source categories you can defend to the seven canonical offenses:
  murder, rape, robbery, aggravated_assault, burglary, larceny, motor_vehicle_theft. Missing rape is
  acceptable and should be documented; the runner will reject no-data offenses.
- If no all-offense point feed exists, a partial official point feed is acceptable, but note the
  coverage limitation clearly and only include defensible offense rules.
- Include worker_judgment_notes with source-selection rationale, partial-coverage risks, and fields
  used for identity/date/geography/offense.
- Include sources_checked with URLs for pages/APIs inspected.
"""


def _parse_usage(jsonl_path: Path) -> tuple[str | None, dict[str, object] | None, str | None]:
    thread_id = None
    usage = None
    failure = None
    if not jsonl_path.exists():
        return thread_id, usage, failure
    for line in jsonl_path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        elif event.get("type") == "turn.completed":
            usage = event.get("usage", {})
        elif event.get("type") in {"turn.failed", "error"}:
            failure = json.dumps(event.get("error") or event.get("message"))[:500]
    return thread_id, usage, failure


def _run_worker(
    target: dict[str, str],
    *,
    output_root: Path,
    timeout_seconds: int,
    model: str,
    reasoning_effort: str,
) -> dict[str, object]:
    prompts_dir = output_root / "prompts"
    logs_dir = output_root / "worker_logs"
    outputs_dir = output_root / "worker_outputs"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    city_key = target["city_key"]
    prompt_path = prompts_dir / f"{city_key}.txt"
    jsonl_path = logs_dir / f"{city_key}.jsonl"
    stderr_path = logs_dir / f"{city_key}.stderr"
    output_path = outputs_dir / f"{city_key}_contract.json"
    prompt_path.write_text(PROMPT_TEMPLATE.format(**target))

    cmd = [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--sandbox",
        "danger-full-access",
        "-c",
        'approval_policy="never"',
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-C",
        str(REPO_ROOT),
        "-m",
        model,
        "--output-schema",
        "scripts/diagnostics/generic_city_contract_schema.json",
        "-o",
        str(output_path),
        "-",
    ]
    start = time.perf_counter()
    status = "unknown"
    return_code = None
    with prompt_path.open("rb") as stdin, jsonl_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            return_code = proc.wait(timeout=timeout_seconds)
            status = "completed" if return_code == 0 and output_path.exists() and output_path.stat().st_size > 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            return_code = proc.returncode
    wall_seconds = time.perf_counter() - start
    thread_id, usage, failure = _parse_usage(jsonl_path)
    stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
    old_error = "failed to initialize in-process app-server" in stderr_text or "Operation not permitted" in stderr_text
    return {
        "city_key": city_key,
        "city_name": target["city_name"],
        "state_abbr": target["state_abbr"],
        "status": status,
        "return_code": return_code,
        "timeout_seconds": timeout_seconds,
        "wall_seconds": wall_seconds,
        "thread_id": thread_id,
        "input_tokens": (usage or {}).get("input_tokens"),
        "cached_input_tokens": (usage or {}).get("cached_input_tokens"),
        "output_tokens": (usage or {}).get("output_tokens"),
        "reasoning_output_tokens": (usage or {}).get("reasoning_output_tokens"),
        "total_tokens": ((usage or {}).get("input_tokens") or 0) + ((usage or {}).get("output_tokens") or 0) if usage else None,
        "prompt_path": str(prompt_path),
        "jsonl_path": str(jsonl_path),
        "stderr_path": str(stderr_path),
        "output_path": str(output_path) if output_path.exists() else "",
        "stderr_has_old_appserver_error": old_error,
        "stderr_excerpt": stderr_text[:300],
        "failure_excerpt": failure or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    args = parser.parse_args()

    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    with args.target_csv.open(newline="") as handle:
        targets = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                _run_worker,
                target,
                output_root=output_root,
                timeout_seconds=args.timeout_seconds,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
            for target in targets
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    rows.sort(key=lambda row: str(row["city_key"]))
    output_root.mkdir(parents=True, exist_ok=True)
    pd = __import__("pandas")
    pd.DataFrame(rows).to_csv(output_root / "external_swarm_worker_cost_runtime.csv", index=False)
    summary = {
        "target_count": len(targets),
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "timeout": sum(row["status"] == "timeout" for row in rows),
        "old_appserver_error_rows": sum(bool(row["stderr_has_old_appserver_error"]) for row in rows),
    }
    (output_root / "external_swarm_worker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
