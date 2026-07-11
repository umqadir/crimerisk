from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crime import OFFENSES_7  # noqa: E402


PILOT_CITIES = ("dallas", "tucson", "charlotte", "san_diego", "montgomery_county_md")
RECONCILIATION_TOLERANCE = 0.10
MIN_GEOCODE_MATCH_SHARE = 0.80
COUNTY_CASE_TYPES = {"suburban_county_validation_case"}


_VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "build_next_phase_validation_shares",
    REPO_ROOT / "scripts" / "diagnostics" / "build_next_phase_validation_shares.py",
)
if _VALIDATION_SPEC is None or _VALIDATION_SPEC.loader is None:
    raise RuntimeError("Could not load build_next_phase_validation_shares.py")
validation = importlib.util.module_from_spec(_VALIDATION_SPEC)
_VALIDATION_SPEC.loader.exec_module(validation)


CITY_BUILDERS: dict[str, Callable[..., tuple[pd.DataFrame, pd.DataFrame]]] = {
    "dallas": validation._build_dallas_surface,
    "tucson": validation._build_tucson_surface,
    "charlotte": validation._build_charlotte_surface,
    "san_diego": validation._build_san_diego_surface,
    "montgomery_county_md": validation._build_montgomery_county_surface,
}


def _json_safe(value: object) -> object:
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _read_controls(year: int) -> pd.DataFrame:
    controls = pd.read_parquet(REPO_ROOT / "state" / "controls" / f"jurisdiction_controls_{year}.parquet")
    controls = controls[controls["offense"].isin(OFFENSES_7)].copy()
    controls["jurisdiction_id"] = controls["jurisdiction_id"].astype("string")
    controls["offense"] = controls["offense"].astype("string")
    controls["official_total"] = pd.to_numeric(controls["adjusted_count_ags_core"], errors="coerce").fillna(0.0)
    keep = [
        "jurisdiction_id",
        "offense",
        "official_total",
        "preferred_source",
        "quality_tier_preferred",
        "estimated_from_panel",
        "needs_current_year_fill",
    ]
    return controls[keep].copy()


def _run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, object]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "wall_seconds": float(time.perf_counter() - started),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _status_from_reconciliation(
    *,
    city_key: str,
    surface: pd.DataFrame,
    reconciliation: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    if surface.empty and reconciliation.empty:
        return pd.DataFrame()

    if reconciliation.empty:
        recon = (
            surface.groupby(["city_name", "jurisdiction_id", "state_fips", "year", "offense"], dropna=False)[
                "incident_count"
            ]
            .sum()
            .rename("matched_share_rows")
            .reset_index()
        )
        recon["raw_offense_rows"] = recon["matched_share_rows"]
        recon["match_share"] = 1.0
    else:
        recon = reconciliation.copy()
    recon["city_key"] = city_key
    recon["jurisdiction_id"] = recon["jurisdiction_id"].astype("string")
    recon["offense"] = recon["offense"].astype("string")
    for col in ["raw_offense_rows", "matched_share_rows", "point_candidate_rows", "source_block_rows"]:
        if col in recon.columns:
            recon[col] = pd.to_numeric(recon[col], errors="coerce").fillna(0.0)
    if "match_share" not in recon.columns:
        recon["match_share"] = np.where(
            pd.to_numeric(recon.get("raw_offense_rows"), errors="coerce").fillna(0.0).gt(0),
            pd.to_numeric(recon.get("matched_share_rows"), errors="coerce").fillna(0.0)
            / pd.to_numeric(recon.get("raw_offense_rows"), errors="coerce").fillna(np.nan),
            np.nan,
        )
    status = recon.merge(controls, on=["jurisdiction_id", "offense"], how="left")
    status["official_total_present"] = status["official_total"].notna()
    status["feed_total_for_reconciliation"] = pd.to_numeric(
        status.get("raw_offense_rows"),
        errors="coerce",
    ).fillna(0.0)
    status["reconciliation_abs_delta"] = (
        status["feed_total_for_reconciliation"] - status["official_total"].fillna(0.0)
    ).abs()
    status["reconciliation_relative_delta"] = np.where(
        status["official_total"].fillna(0.0).gt(0),
        status["reconciliation_abs_delta"] / status["official_total"].replace({0: np.nan}),
        np.nan,
    )
    status["reconciles_within_10pct"] = (
        status["official_total_present"]
        & status["reconciliation_relative_delta"].le(RECONCILIATION_TOLERANCE)
    )
    status["share_geography_pass"] = pd.to_numeric(status["match_share"], errors="coerce").ge(
        MIN_GEOCODE_MATCH_SHARE
    )
    status["share_sum_pass"] = True
    if not surface.empty:
        sums = (
            surface.groupby(["jurisdiction_id", "year", "offense"], dropna=False)["share_within_city"]
            .sum()
            .rename("share_sum")
            .reset_index()
        )
        sums["jurisdiction_id"] = sums["jurisdiction_id"].astype("string")
        sums["offense"] = sums["offense"].astype("string")
        status = status.merge(sums, on=["jurisdiction_id", "year", "offense"], how="left")
        status["share_sum_pass"] = pd.to_numeric(status["share_sum"], errors="coerce").sub(1.0).abs().le(1e-6)
    else:
        status["share_sum"] = np.nan
        status["share_sum_pass"] = False
    status["total_lane_disposition"] = np.where(status["reconciles_within_10pct"], "feed_total_candidate", "official_control")
    status["share_lane_disposition"] = np.where(
        status["official_total_present"] & status["share_geography_pass"] & status["share_sum_pass"],
        "admit_share_only",
        "reject_share",
    )
    status["admitted_flag"] = status["share_lane_disposition"].eq("admit_share_only")
    status["reconciliation_disposition"] = np.where(
        status["reconciles_within_10pct"],
        "reconciled_within_strict_10pct",
        "unreconciled_keep_share_only",
    )
    status.loc[~status["official_total_present"], "reconciliation_disposition"] = "missing_official_total_reject"
    status["qa_status"] = np.where(status["admitted_flag"], "pass_share_only", "reject")
    status["reviewer_notes"] = np.where(
        status["reconciles_within_10pct"],
        "Incident feed is within strict pilot tolerance, but pilot keeps official controls as total lane.",
        "Incident feed is admitted for shares only; official controls remain the total lane.",
    )
    status.loc[~status["admitted_flag"], "reviewer_notes"] = (
        "Rejected from admitted measurement; keep packet as pilot QA evidence and do not use rows in validation."
    )
    return status.sort_values(["city_key", "offense"], kind="mergesort").reset_index(drop=True)


def _write_packet(
    *,
    city_key: str,
    run_id: str,
    surface: pd.DataFrame,
    reconciliation: pd.DataFrame,
    status: pd.DataFrame,
    packet_root: Path,
    raw_dir: Path,
    worker_runtime: dict[str, object],
) -> dict[str, object]:
    packet_dir = packet_root / city_key
    packet_dir.mkdir(parents=True, exist_ok=True)
    source_name = ""
    source_url = ""
    city_name = ""
    jurisdiction_id = ""
    state_fips = ""
    validation_case_type = ""
    if not status.empty:
        row = status.iloc[0]
        source_name = str(row.get("validation_source_name", ""))
        source_url = str(row.get("validation_source_url", ""))
        city_name = str(row.get("city_name", ""))
        jurisdiction_id = str(row.get("jurisdiction_id", ""))
        validation_case_type = str(row.get("validation_case_type", ""))
    if not surface.empty:
        first = surface.iloc[0]
        city_name = city_name or str(first.get("city_name", ""))
        jurisdiction_id = jurisdiction_id or str(first.get("jurisdiction_id", ""))
        state_fips = str(first.get("state_fips", ""))
        validation_case_type = validation_case_type or str(first.get("validation_case_type", ""))
        source_name = source_name or str(first.get("validation_source_name", ""))
        source_url = source_url or str(first.get("validation_source_url", ""))

    source_candidate = pd.DataFrame(
        [
            {
                "case_id": f"{run_id}:{city_key}",
                "city_key": city_key,
                "jurisdiction_id": jurisdiction_id,
                "city_name": city_name,
                "state_abbr": "",
                "source_name": source_name,
                "source_url": source_url,
                "portal_type": "pilot_validation_feed",
                "coverage_start_year": 2024,
                "coverage_end_year": 2024,
                "years_usable": 1,
                "offense_fields_present": "source-specific canonical mapper in build_next_phase_validation_shares.py",
                "date_field": "source-specific",
                "location_fields_present": "lat/lon, census block, or source-specific point fields",
                "latlon_present": True,
                "address_present": None,
                "block_group_join_ready": bool(not surface.empty),
                "geocode_quality_tier": ";".join(sorted(surface["geocode_quality_tier"].dropna().astype(str).unique()))
                if not surface.empty
                else "",
                "dedupe_key_available": True,
                "offense_crosswalk_complexity": "source-specific",
                "recommended_disposition": "ready_now" if bool(status["admitted_flag"].any()) else "insufficient_quality",
                "analyst_notes": "Gate-1 pilot packet. Incident feed is share lane only unless strict reconciliation clears.",
                "sources": source_url,
            }
        ]
    )
    offense_rows = []
    for offense in OFFENSES_7:
        row = status[status["offense"].astype(str).eq(offense)].head(1)
        offense_rows.append(
            {
                "city_key": city_key,
                "offense": offense,
                "production_ready": "yes" if not row.empty and bool(row.iloc[0].get("admitted_flag")) else "no",
                "city_share_integration_status": "pilot_share_only" if not row.empty and bool(row.iloc[0].get("admitted_flag")) else "pilot_reject",
                "total_lane": "official_control",
                "feed_total_candidate": "yes" if not row.empty and bool(row.iloc[0].get("reconciles_within_10pct")) else "no",
                "notes": "" if not row.empty else "No mapped source rows for this offense.",
            }
        )
    offense_status = pd.DataFrame(offense_rows)
    packet_status = pd.DataFrame(
        [
            {
                "city_key": city_key,
                "city_name": city_name,
                "jurisdiction_id": jurisdiction_id,
                "validation_case_type": validation_case_type,
                "production_ready": "partial" if bool(status["admitted_flag"].any()) else "no",
                "city_share_integration_status": "pilot_share_only",
                "run_id": run_id,
            }
        ]
    )
    checklist = pd.DataFrame(
        [
            {"check": "official_total_lane_present", "status": "pass" if status["official_total_present"].all() else "fail"},
            {"check": "share_geography_gate", "status": "pass" if status["share_geography_pass"].all() else "fail"},
            {"check": "share_sum_gate", "status": "pass" if status["share_sum_pass"].all() else "fail"},
            {"check": "strict_reconciliation_recorded", "status": "pass"},
            {"check": "incident_feed_not_used_as_total", "status": "pass"},
        ]
    )
    qa_summary = {
        "run_id": run_id,
        "city_key": city_key,
        "city_name": city_name,
        "jurisdiction_id": jurisdiction_id,
        "validation_case_type": validation_case_type,
        "rows_in_surface": int(len(surface)),
        "offense_rows": int(len(status)),
        "admitted_offense_rows": int(status["admitted_flag"].sum()) if not status.empty else 0,
        "strict_reconciliation_tolerance": RECONCILIATION_TOLERANCE,
        "min_geocode_match_share": MIN_GEOCODE_MATCH_SHARE,
        "schema_pass": bool(not surface.empty and not status.empty),
        "share_geography_pass_all": bool(status["share_geography_pass"].all()) if not status.empty else False,
        "share_sum_pass_all": bool(status["share_sum_pass"].all()) if not status.empty else False,
        "official_total_present_all": bool(status["official_total_present"].all()) if not status.empty else False,
        "admission_status": "pass_share_only" if bool(status["admitted_flag"].any()) else "reject",
        "worker_runtime": worker_runtime,
    }
    manifest = {
        "run_id": run_id,
        "city_key": city_key,
        "city_name": city_name,
        "jurisdiction_id": jurisdiction_id,
        "source_name": source_name,
        "source_url": source_url,
        "raw_cache_dir": str(raw_dir),
        "builder": CITY_BUILDERS[city_key].__name__,
        "contract": "Gate-1 pilot: official totals lane plus incident share lane.",
        "incident_feed_total_policy": "share_only_unless_strict_reconciliation_clears",
        "packet_dir": str(packet_dir),
        "worker_runtime": worker_runtime,
    }
    source_candidate.to_csv(packet_dir / "source_candidate.csv", index=False)
    status.to_csv(packet_dir / "reconciliation_summary.csv", index=False)
    offense_status.to_csv(packet_dir / "packet_offense_status.csv", index=False)
    packet_status.to_csv(packet_dir / "packet_status.csv", index=False)
    checklist.to_csv(packet_dir / "packet_checklist.csv", index=False)
    surface.to_parquet(packet_dir / "city_share_surface.parquet", index=False)
    surface.to_parquet(packet_dir / "normalized_incidents.parquet", index=False)
    (packet_dir / "qa_summary.json").write_text(json.dumps(_json_safe(qa_summary), indent=2, sort_keys=True))
    (packet_dir / "packet_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    (packet_dir / "research_findings.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return qa_summary


def _write_digest(
    *,
    out_path: Path,
    run_id: str,
    status: pd.DataFrame,
    worker_runtime: pd.DataFrame,
    totals_metrics: pd.DataFrame | None,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        text = frame.copy()
        for col in text.columns:
            text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
        headers = [str(col) for col in text.columns]
        rows = text.astype(str).values.tolist()
        def clean(value: str) -> str:
            return value.replace("|", "\\|").replace("\n", " ")
        header_line = "| " + " | ".join(clean(value) for value in headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows]
        return "\n".join([header_line, sep_line, *body])

    admitted = status[status["admitted_flag"]].copy() if not status.empty else pd.DataFrame()
    rejected = status[~status["admitted_flag"]].copy() if not status.empty else pd.DataFrame()
    municipal = status[~status.get("validation_case_type", pd.Series("", index=status.index)).isin(COUNTY_CASE_TYPES)].copy() if not status.empty else pd.DataFrame()
    county = status[status.get("validation_case_type", pd.Series("", index=status.index)).isin(COUNTY_CASE_TYPES)].copy() if not status.empty else pd.DataFrame()
    lines = [
        f"# Gate-1 Pilot Supervisor Digest",
        "",
        f"- Run ID: `{run_id}`",
        f"- Pilot rows: {len(status)} city/offense rows",
        f"- Admitted share-only rows: {len(admitted)}",
        f"- Rejected rows: {len(rejected)}",
        f"- Municipal rows: {len(municipal)}",
        f"- Suburban county stress-case rows: {len(county)}",
        f"- Strict reconciliation tolerance: {RECONCILIATION_TOLERANCE:.0%}",
        "",
        "## Accepted Rows",
        "",
    ]
    if admitted.empty:
        lines.append("No rows admitted.")
    else:
        view = admitted[
            [
                "city_key",
                "offense",
                "official_total",
                "feed_total_for_reconciliation",
                "reconciliation_relative_delta",
                "match_share",
                "reconciliation_disposition",
            ]
        ].copy()
        lines.append(markdown_table(view))
    lines.extend(["", "## Rejected Rows", ""])
    if rejected.empty:
        lines.append("No rows rejected.")
    else:
        view = rejected[
            [
                "city_key",
                "offense",
                "official_total",
                "feed_total_for_reconciliation",
                "match_share",
                "qa_status",
                "reconciliation_disposition",
            ]
        ].copy()
        lines.append(markdown_table(view))
    lines.extend(["", "## Largest Uncertainties", ""])
    if status.empty:
        lines.append("- No status rows were produced.")
    else:
        largest = status.assign(
            abs_gap=lambda df: pd.to_numeric(df["reconciliation_relative_delta"], errors="coerce").abs()
        ).sort_values("abs_gap", ascending=False).head(10)
        for row in largest.to_dict(orient="records"):
            gap = row.get("reconciliation_relative_delta")
            gap_text = "n/a" if pd.isna(gap) else f"{float(gap):.1%}"
            lines.append(
                f"- `{row['city_key']}` `{row['offense']}`: feed/control gap {gap_text}; "
                f"match share {float(row.get('match_share', 0.0) or 0.0):.1%}; "
                f"{row.get('reconciliation_disposition', '')}."
            )
    lines.extend(["", "## Ranked Packet Audit Sample", ""])
    if status.empty:
        lines.append("- No packets available.")
    else:
        audit = (
            status.assign(
                priority_score=lambda df: (
                    pd.to_numeric(df["official_total"], errors="coerce").fillna(0.0)
                    * (1.0 - pd.to_numeric(df["match_share"], errors="coerce").fillna(0.0)).clip(lower=0.0)
                    + pd.to_numeric(df["reconciliation_relative_delta"], errors="coerce").abs().fillna(0.0) * 1000.0
                )
            )
            .sort_values("priority_score", ascending=False)
            .drop_duplicates("city_key")
        )
        for row in audit.head(8).to_dict(orient="records"):
            lines.append(
                f"- `{row['city_key']}`: audit `{REPO_ROOT / 'state' / 'review' / 'packets' / 'city' / str(row['city_key'])}` "
                f"because representative offense `{row['offense']}` has high reconciliation/geography risk."
            )
    lines.extend(["", "## Worker Cost / Runtime", ""])
    if worker_runtime.empty:
        lines.append("No worker runtime rows emitted.")
    else:
        lines.append(markdown_table(worker_runtime))
    if totals_metrics is not None and not totals_metrics.empty:
        lines.extend(["", "## Totals Diagnosis Metrics", ""])
        lines.append(markdown_table(totals_metrics.head(30)))
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Gate-1 gap-closing pilot for selected cities.")
    parser.add_argument("--run-id", default=time.strftime("gate1_pilot_%Y%m%d_%H%M%S"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--city-key", action="append", choices=PILOT_CITIES, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--run-measurement", action="store_true")
    args = parser.parse_args()

    run_id = str(args.run_id)
    city_keys = tuple(args.city_key or PILOT_CITIES)
    run_dir = REPO_ROOT / "state" / "orchestration" / "gap_closing_runs" / run_id
    raw_dir = REPO_ROOT / "state" / "modeling" / "inputs" / "gap_closing_pilot" / run_id
    packet_root = REPO_ROOT / "state" / "review" / "packets" / "city"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    controls = _read_controls(int(args.year))
    surfaces: list[pd.DataFrame] = []
    reconciliations: list[pd.DataFrame] = []
    statuses: list[pd.DataFrame] = []
    runtimes: list[dict[str, object]] = []
    qa_payloads: list[dict[str, object]] = []

    for city_key in city_keys:
        started = time.perf_counter()
        builder = CITY_BUILDERS[city_key]
        surface, reconciliation = builder(
            year=int(args.year),
            raw_dir=raw_dir,
            bg_crosswalk_path=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
            force_refresh=bool(args.force_refresh),
        )
        wall = float(time.perf_counter() - started)
        runtime = {
            "run_id": run_id,
            "worker_id": f"local_deterministic_{city_key}",
            "city_key": city_key,
            "worker_type": "local_deterministic_builder",
            "model": "",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "token_note": "No LLM tokens: deterministic Python city builder.",
            "wall_seconds": wall,
            "status": "completed" if not surface.empty else "empty_surface",
        }
        status = _status_from_reconciliation(
            city_key=city_key,
            surface=surface,
            reconciliation=reconciliation,
            controls=controls,
        )
        qa_payloads.append(
            _write_packet(
                city_key=city_key,
                run_id=run_id,
                surface=surface,
                reconciliation=reconciliation,
                status=status,
                packet_root=packet_root,
                raw_dir=raw_dir,
                worker_runtime=runtime,
            )
        )
        runtimes.append(runtime)
        if not surface.empty and not status.empty:
            admitted_keys = set(
                status.loc[status["admitted_flag"], ["jurisdiction_id", "year", "offense"]]
                .astype(str)
                .agg("|".join, axis=1)
                .tolist()
            )
            surface_keys = surface[["jurisdiction_id", "year", "offense"]].astype(str).agg("|".join, axis=1)
            admitted_surface = surface[surface_keys.isin(admitted_keys)].copy()
            if not admitted_surface.empty:
                surfaces.append(admitted_surface)
        if not reconciliation.empty:
            reconciliations.append(reconciliation.assign(city_key=city_key))
        if not status.empty:
            statuses.append(status)

    base = pd.read_parquet(REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet").copy()
    base_meta = validation._case_metadata_for_existing(base)
    base = base.merge(base_meta, on=["city_name", "jurisdiction_id", "state_fips"], how="left")
    validation_ids = {
        str(value)
        for surface in surfaces
        for value in surface["jurisdiction_id"].dropna().astype(str).unique().tolist()
    }
    combined = base[~base["jurisdiction_id"].astype(str).isin(validation_ids)].copy() if validation_ids else base
    if surfaces:
        combined = pd.concat([combined, *surfaces], ignore_index=True, sort=False)
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")
    combined["block_group_geoid"] = combined["block_group_geoid"].astype("string").str.zfill(12)
    combined["state_fips"] = combined["state_fips"].astype("string").str.zfill(2)
    status_all = pd.concat(statuses, ignore_index=True, sort=False) if statuses else pd.DataFrame()
    reconciliation_all = (
        pd.concat(reconciliations, ignore_index=True, sort=False)
        if reconciliations
        else pd.DataFrame()
    )
    runtime_df = pd.DataFrame(runtimes)

    city_surface_out = run_dir / f"pilot_city_incident_share_surface_{args.year}.parquet"
    status_out = run_dir / "city_offense_status.csv"
    reconciliation_out = run_dir / "pilot_reconciliation.csv"
    runtime_out = run_dir / "worker_cost_runtime.csv"
    combined.to_parquet(city_surface_out, index=False)
    status_all.to_csv(status_out, index=False)
    reconciliation_all.to_csv(reconciliation_out, index=False)
    runtime_df.to_csv(runtime_out, index=False)

    measurement_payload: dict[str, object] = {}
    totals_metrics: pd.DataFrame | None = None
    if args.run_measurement:
        env = dict(**{k: v for k, v in dict().items()})
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/diagnostics/next_phase_measurement.py",
            "--year",
            str(args.year),
            "--city-shares-path",
            str(city_surface_out),
            "--cv-predictions-out",
            str(run_dir / "jurisdiction_cv_predictions.parquet"),
            "--cv-metrics-out",
            str(run_dir / "jurisdiction_cv_prediction_metrics.parquet"),
            "--allocation-diagnostics-out",
            str(run_dir / "measurement_spine_allocation_diagnostics.parquet"),
            "--allocation-summary-out",
            str(run_dir / "measurement_spine_allocation_summary.csv"),
            "--error-budget-out",
            str(run_dir / "error_budget_city_offense.csv"),
            "--decision-table-out",
            str(run_dir / "measurement_spine_decision_table.csv"),
            "--summary-json-out",
            str(run_dir / "next_phase_measurement_summary.json"),
        ]
        command_payload = _run_cmd(
            cmd,
            cwd=REPO_ROOT,
            env={"UV_CACHE_DIR": "/private/tmp/uv-cache", **dict(__import__("os").environ)},
        )
        measurement_payload["next_phase_measurement_command"] = command_payload
        if int(command_payload["returncode"]) != 0:
            raise SystemExit(json.dumps(_json_safe(measurement_payload), indent=2))
        totals_metrics = pd.read_csv(run_dir / "measurement_spine_decision_table.csv")
        totals_metrics.to_csv(run_dir / "totals_diagnosis_metrics.csv", index=False)
    else:
        prior_metrics = REPO_ROOT / "materials" / "tables" / "measurement_spine_decision_table.csv"
        if prior_metrics.exists():
            totals_metrics = pd.read_csv(prior_metrics)
            totals_metrics.to_csv(run_dir / "totals_diagnosis_metrics.csv", index=False)

    run_manifest = {
        "run_id": run_id,
        "year": int(args.year),
        "city_keys": list(city_keys),
        "strict_reconciliation_tolerance": RECONCILIATION_TOLERANCE,
        "min_geocode_match_share": MIN_GEOCODE_MATCH_SHARE,
        "official_total_policy": "Federal controls satisfy total lane; local publications optional corroboration.",
        "share_policy": "Admit as share_only if geography/share QA passes, even when reconciliation is unresolved.",
        "county_case_policy": "Montgomery County, MD remains a suburban-county stress case and is not mixed into municipal conclusions.",
        "outputs": {
            "city_surface": str(city_surface_out),
            "city_offense_status": str(status_out),
            "reconciliation": str(reconciliation_out),
            "worker_cost_runtime": str(runtime_out),
            "digest": str(run_dir / "gate1_supervisor_digest.md"),
            "totals_diagnosis_metrics": str(run_dir / "totals_diagnosis_metrics.csv"),
        },
        "packet_root": str(packet_root),
        "raw_dir": str(raw_dir),
        "qa_payloads": qa_payloads,
        "measurement": measurement_payload,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(_json_safe(run_manifest), indent=2, sort_keys=True))
    _write_digest(
        out_path=run_dir / "gate1_supervisor_digest.md",
        run_id=run_id,
        status=status_all,
        worker_runtime=runtime_df,
        totals_metrics=totals_metrics,
    )
    print(json.dumps(_json_safe(run_manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
