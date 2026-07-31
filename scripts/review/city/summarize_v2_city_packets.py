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


def main() -> int:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    packet_root = paths.review_packets_dir / "city"
    source_inventory_path = REPO_ROOT / "configs" / "city_incident_sources.csv"
    source_inventory = (
        pd.read_csv(source_inventory_path, dtype=str).fillna("")
        if source_inventory_path.exists()
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for status_path in sorted(packet_root.glob("*/packet_status.csv")):
        packet_dir = status_path.parent
        city_key = packet_dir.name
        status = pd.read_csv(status_path, dtype=str).fillna("")
        offense_status_path = packet_dir / "packet_offense_status.csv"
        offense_status = (
            pd.read_csv(offense_status_path, dtype=str).fillna("")
            if offense_status_path.exists()
            else pd.DataFrame()
        )
        central_source = (
            source_inventory.loc[source_inventory["city_key"].eq(city_key)].head(1)
            if not source_inventory.empty and "city_key" in source_inventory.columns
            else pd.DataFrame()
        )
        packet_source_path = packet_dir / "source_candidate.csv"
        packet_source = (
            pd.read_csv(packet_source_path, dtype=str).fillna("")
            if packet_source_path.exists()
            else pd.DataFrame()
        )
        has_research_findings = int((packet_dir / "research_findings.json").exists())
        has_reconciliation_summary = int((packet_dir / "reconciliation_summary.csv").exists())
        has_packet_offense_status = int(offense_status_path.exists())
        offense_ready = []
        if not offense_status.empty and {"offense", "production_ready", "city_share_integration_status"}.issubset(offense_status.columns):
            ready_mask = offense_status["production_ready"].astype(str).str.lower().isin({"yes", "true", "1", "partial"})
            active_mask = offense_status["city_share_integration_status"].astype(str).str.lower().isin({"active", "approved", "live", "offense_selective"})
            offense_ready = sorted(
                {
                    str(v).strip()
                    for v in offense_status.loc[ready_mask & active_mask, "offense"].tolist()
                    if str(v).strip()
                }
            )
        row = {
            "city_key": city_key,
            "packet_dir": str(packet_dir),
            "packet_status": status.iloc[0]["packet_status"] if not status.empty and "packet_status" in status.columns else "",
            "current_owner": status.iloc[0]["current_owner"] if not status.empty and "current_owner" in status.columns else "",
            "production_ready": status.iloc[0]["production_ready"] if not status.empty and "production_ready" in status.columns else "",
            "city_share_integration_status": status.iloc[0]["city_share_integration_status"] if not status.empty and "city_share_integration_status" in status.columns else "",
            "reconciliation_status": status.iloc[0]["reconciliation_status"] if not status.empty and "reconciliation_status" in status.columns else "",
            "has_research_findings": has_research_findings,
            "has_reconciliation_summary": has_reconciliation_summary,
            "has_packet_offense_status": has_packet_offense_status,
            "offense_selective_ready_count": len(offense_ready),
            "offense_selective_ready_offenses": "|".join(offense_ready),
            "central_recommended_disposition": central_source.iloc[0]["recommended_disposition"] if not central_source.empty and "recommended_disposition" in central_source.columns else "",
            "packet_recommended_disposition": packet_source.iloc[0]["recommended_disposition"] if not packet_source.empty and "recommended_disposition" in packet_source.columns else "",
            "source_name": central_source.iloc[0]["source_name"] if not central_source.empty and "source_name" in central_source.columns else "",
            "state_abbr": central_source.iloc[0]["state_abbr"] if not central_source.empty and "state_abbr" in central_source.columns else "",
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["state_abbr", "city_key"], kind="mergesort")
    out_path = packet_root / "city_packet_status_summary.csv"
    summary.to_csv(out_path, index=False)
    print(json.dumps({"rows": int(len(summary)), "out_path": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
