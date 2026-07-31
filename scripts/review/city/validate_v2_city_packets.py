from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths


def _norm(value: object) -> str:
    return str(value).strip().lower()


def main() -> int:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    packet_root = paths.review_packets_dir / "city"
    source_cfg = REPO_ROOT / "configs" / "city_incident_sources.csv"
    summary_path = packet_root / "city_packet_status_summary.csv"

    if not packet_root.exists():
        raise FileNotFoundError(f"Missing city packet root: {packet_root}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing packet summary: {summary_path}")

    summary = pd.read_csv(summary_path, dtype=str).fillna("")
    source_cfg_df = pd.read_csv(source_cfg, dtype=str).fillna("") if source_cfg.exists() else pd.DataFrame()

    issues: list[dict[str, object]] = []
    for row in summary.to_dict(orient="records"):
        city_key = row.get("city_key", "")
        if city_key.startswith("_"):
            continue
        packet_dir = Path(row.get("packet_dir", ""))
        production_ready = _norm(row.get("production_ready", ""))
        integration_status = _norm(row.get("city_share_integration_status", ""))
        central_disposition = _norm(row.get("central_recommended_disposition", ""))

        required = {
            "packet_manifest.json": (packet_dir / "packet_manifest.json").exists(),
            "packet_checklist.csv": (packet_dir / "packet_checklist.csv").exists(),
            "packet_status.csv": (packet_dir / "packet_status.csv").exists(),
            "source_candidate.csv": (packet_dir / "source_candidate.csv").exists(),
        }
        for name, exists in required.items():
            if not exists:
                issues.append(
                    {
                        "city_key": city_key,
                        "severity": "error",
                        "issue_type": "missing_required_packet_artifact",
                        "artifact": name,
                        "detail": f"{packet_dir / name} is missing",
                    }
                )

        if central_disposition == "ready_now":
            if not (packet_dir / "research_findings.json").exists():
                issues.append(
                    {
                        "city_key": city_key,
                        "severity": "error",
                        "issue_type": "ready_city_missing_research_findings",
                        "artifact": "research_findings.json",
                        "detail": "Central source inventory marks city ready_now, but packet research findings are missing.",
                    }
                )
            if not (packet_dir / "reconciliation_summary.csv").exists():
                issues.append(
                    {
                        "city_key": city_key,
                        "severity": "error",
                        "issue_type": "ready_city_missing_reconciliation",
                        "artifact": "reconciliation_summary.csv",
                        "detail": "Central source inventory marks city ready_now, but packet reconciliation summary is missing.",
                    }
                )

        if production_ready in {"yes", "true", "1", "partial"} and integration_status in {"active", "approved", "live", "offense_selective"}:
            if integration_status == "offense_selective" and not (packet_dir / "packet_offense_status.csv").exists():
                issues.append(
                    {
                        "city_key": city_key,
                        "severity": "error",
                        "issue_type": "offense_selective_missing_offense_gate",
                        "artifact": "packet_offense_status.csv",
                        "detail": "Offense-selective integration requires packet_offense_status.csv.",
                    }
                )

        if not source_cfg_df.empty and city_key and city_key not in set(source_cfg_df["city_key"].astype(str)):
            issues.append(
                {
                    "city_key": city_key,
                    "severity": "warning",
                    "issue_type": "packet_missing_from_central_source_inventory",
                    "artifact": "configs/city_incident_sources.csv",
                    "detail": "Packet exists but city_key is absent from central source inventory.",
                }
            )

    out_dir = paths.review_packets_dir / "city"
    out_csv = out_dir / "city_packet_validation.csv"
    out_json = out_dir / "city_packet_validation_summary.json"

    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        issues_df = pd.DataFrame(columns=["city_key", "severity", "issue_type", "artifact", "detail"])
    issues_df.to_csv(out_csv, index=False)

    summary_payload = {
        "rows": int(len(summary)),
        "issue_count": int(len(issues_df)),
        "error_count": int((issues_df["severity"] == "error").sum()) if not issues_df.empty else 0,
        "warning_count": int((issues_df["severity"] == "warning").sum()) if not issues_df.empty else 0,
        "out_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary_payload, indent=2))
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
