from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths


def _write_packet_reconciliation_summary(packet_dir: Path, recon_dir: Path, city_key: str) -> None:
    recon_path = recon_dir / f"{city_key}_reconciliation.csv"
    if not recon_path.exists():
        return
    df = pd.read_csv(recon_path)
    if df.empty:
        return
    if "published_count" in df.columns:
        published = df[df["published_count"].notna()].copy()
    else:
        published = pd.DataFrame()
    base = published if not published.empty else df.copy()
    base["year_numeric"] = pd.to_numeric(base.get("year"), errors="coerce")
    latest_year = base["year_numeric"].dropna().max() if not base.empty else None
    if latest_year == latest_year:
        base = base[base["year_numeric"].eq(latest_year)].copy()
    keep = [
        "offense",
        "mapped_offense_count",
        "geocoded_offense_count",
        "matched_offense_count",
        "final_share_count",
        "published_count",
        "published_source_name",
        "published_source_url",
        "published_comparison_quality",
        "published_notes",
        "diff_final_vs_published",
        "pct_diff_final_vs_published",
    ]
    cols = [c for c in keep if c in base.columns]
    if cols:
        base[cols].to_csv(packet_dir / "reconciliation_summary.csv", index=False)


def _write_packet_research_findings(packet_dir: Path, row: dict[str, object], recon_dir: Path) -> None:
    findings_path = packet_dir / "research_findings.json"
    if findings_path.exists():
        return
    recon_path = recon_dir / f"{row.get('city_key', '')}_reconciliation.csv"
    latest_year = None
    comparability_notes: list[dict[str, object]] = []
    if recon_path.exists():
        recon = pd.read_csv(recon_path)
        if not recon.empty:
            year_series = pd.to_numeric(recon.get("year"), errors="coerce")
            latest_year = int(year_series.dropna().max()) if not year_series.dropna().empty else None
            latest = recon[year_series.eq(latest_year)].copy() if latest_year is not None else recon.copy()
            for _, rec in latest.iterrows():
                comparability_notes.append(
                    {
                        "offense": rec.get("offense"),
                        "published_comparison_quality": rec.get("published_comparison_quality"),
                        "published_count": rec.get("published_count"),
                        "final_share_count": rec.get("final_share_count"),
                        "pct_diff_final_vs_published": rec.get("pct_diff_final_vs_published"),
                    }
                )
    payload = {
        "city_key": row.get("city_key", ""),
        "city_name": row.get("city_name", ""),
        "jurisdiction_id": row.get("jurisdiction_id", ""),
        "analysis_date": pd.Timestamp.utcnow().date().isoformat(),
        "source_contract": {
            "recommended_source_name": row.get("source_name", ""),
            "recommended_source_url": row.get("source_url", ""),
            "portal_type": row.get("portal_type", ""),
            "coverage_start_year": row.get("coverage_start_year", ""),
            "coverage_end_year": row.get("coverage_end_year", ""),
            "years_usable": row.get("years_usable", ""),
            "geocode_quality_tier": row.get("geocode_quality_tier", ""),
            "recommended_disposition": row.get("recommended_disposition", ""),
        },
        "published_reconciliation": {
            "latest_reconciled_year": latest_year,
            "reconciliation_summary_file": "reconciliation_summary.csv",
            "packet_local_summary_generated_from": str(recon_path) if recon_path.exists() else "",
            "latest_year_offense_notes": comparability_notes,
        },
        "judgment": {
            "recommended_disposition": row.get("recommended_disposition", ""),
            "note": "Packet-local research summary generated from the active central city source inventory and stored reconciliation artifact.",
        },
    }
    findings_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    source_cfg = REPO_ROOT / "configs" / "city_incident_sources.csv"
    priority_cfg = REPO_ROOT / "configs" / "city_incident_priority.csv"
    packet_root = paths.review_packets_dir / "city"
    recon_dir = paths.review_analysis_dir / "city_reconciliation"
    if not source_cfg.exists() or not priority_cfg.exists() or not packet_root.exists():
        print({"updated_packets": 0, "reason": "missing_config_or_packet_root"})
        return 0

    sources = pd.read_csv(source_cfg, dtype=str).fillna("")
    priority = pd.read_csv(priority_cfg, dtype=str).fillna("")
    merged = priority.merge(
        sources,
        on=["city_key", "jurisdiction_id", "city_name", "state_abbr"],
        how="outer",
        suffixes=("_priority", "_source"),
    )
    merged["priority_rank_numeric"] = pd.to_numeric(merged.get("priority_rank", ""), errors="coerce")
    merged = merged.sort_values(["priority_bucket", "priority_rank_numeric", "city_key", "source_name"], kind="mergesort")

    manifest_rows: list[dict[str, object]] = []
    updated_packets = 0
    for row in merged.to_dict(orient="records"):
        city_key = str(row.get("city_key", "")).strip()
        if not city_key:
            continue
        packet_dir = packet_root / city_key
        if not packet_dir.exists():
            continue

        source_candidate_path = packet_dir / "source_candidate.csv"
        if not source_candidate_path.exists():
            source_row = pd.DataFrame([row]).drop(columns=["priority_rank_numeric"], errors="ignore")
            source_row.to_csv(source_candidate_path, index=False)

        _write_packet_reconciliation_summary(packet_dir, recon_dir, city_key)
        _write_packet_research_findings(packet_dir, row, recon_dir)

        manifest = {
            "city_key": city_key,
            "city_name": row.get("city_name", ""),
            "jurisdiction_id": row.get("jurisdiction_id", ""),
            "state_abbr": row.get("state_abbr", ""),
            "priority_bucket": row.get("priority_bucket", ""),
            "priority_rank": row.get("priority_rank", ""),
            "basis": row.get("basis", ""),
            "notes": row.get("notes", ""),
            "source_name": row.get("source_name", ""),
            "source_url": row.get("source_url", ""),
            "portal_type": row.get("portal_type", ""),
            "coverage_start_year": row.get("coverage_start_year", ""),
            "coverage_end_year": row.get("coverage_end_year", ""),
            "years_usable": row.get("years_usable", ""),
            "recommended_disposition": row.get("recommended_disposition", ""),
            "geocode_quality_tier": row.get("geocode_quality_tier", ""),
            "packet_dir": str(packet_dir),
            "required_artifacts": [
                "source_candidate.csv",
                "packet_status.csv",
                "packet_checklist.csv",
                "research_findings.json",
                "reconciliation_summary.csv",
            ],
        }
        (packet_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2))
        manifest_rows.append({k: v for k, v in manifest.items() if k != "required_artifacts"})
        updated_packets += 1

    if manifest_rows:
        manifest_path = packet_root / "city_packet_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print({"updated_packets": updated_packets, "packet_root": str(packet_root)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
