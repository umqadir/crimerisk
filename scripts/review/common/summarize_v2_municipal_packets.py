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


REQUIRED_PACKET_FILES = [
    "packet_manifest.json",
    "research_findings.json",
    "sources.csv",
    "recommendation.csv",
]


def _list_extract_files(packet_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in packet_dir.glob("*.csv")
        if path.name not in {"recommendation.csv", "sources.csv"}
    )


def _normalize_yes_no(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return "yes"
    if text in {"no", "false", "0"}:
        return "no"
    return text


def _load_live_non_observed_2024(paths: RepoPaths) -> pd.DataFrame:
    panel_path = paths.state_dir / "controls" / "jurisdiction_year_estimates.parquet"
    panel = pd.read_parquet(
        panel_path,
        columns=[
            "jurisdiction_id",
            "jurisdiction_name",
            "state_abbr",
            "year",
            "estimated_count",
            "usable_as_observed",
            "estimate_source",
        ],
    )
    panel = panel[pd.to_numeric(panel["year"], errors="coerce").eq(2024)].copy()
    panel["estimated_count"] = pd.to_numeric(panel["estimated_count"], errors="coerce").fillna(0.0)
    panel["usable_as_observed"] = panel["usable_as_observed"].fillna(False).astype(bool)
    non_observed = panel[~panel["usable_as_observed"]].copy()
    if non_observed.empty:
        return pd.DataFrame(
            columns=[
                "jurisdiction_id",
                "jurisdiction_name",
                "state_abbr",
                "non_observed_count_2024",
                "top_estimate_source_2024",
            ]
        )
    top_source = (
        non_observed.groupby(["jurisdiction_id", "estimate_source"], dropna=False)["estimated_count"]
        .sum()
        .reset_index()
        .sort_values(["jurisdiction_id", "estimated_count"], ascending=[True, False], kind="mergesort")
        .drop_duplicates(["jurisdiction_id"])
        .rename(columns={"estimate_source": "top_estimate_source_2024"})
    )
    summary = (
        non_observed.groupby(["jurisdiction_id", "jurisdiction_name", "state_abbr"], dropna=False)["estimated_count"]
        .sum()
        .reset_index()
        .rename(columns={"estimated_count": "non_observed_count_2024"})
    )
    return summary.merge(top_source[["jurisdiction_id", "top_estimate_source_2024"]], on="jurisdiction_id", how="left")


def main() -> int:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    packet_root = paths.review_packets_dir / "municipal_targets"
    impact = _load_live_non_observed_2024(paths)

    rows: list[dict[str, object]] = []
    for packet_dir in sorted(path for path in packet_root.iterdir() if path.is_dir()):
        manifest_path = packet_dir / "packet_manifest.json"
        recommendation_path = packet_dir / "recommendation.csv"
        findings_path = packet_dir / "research_findings.json"
        sources_path = packet_dir / "sources.csv"

        manifest: dict[str, object] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        recommendation = (
            pd.read_csv(recommendation_path, dtype=str).fillna("")
            if recommendation_path.exists()
            else pd.DataFrame()
        )

        missing_outputs = [name for name in REQUIRED_PACKET_FILES if not (packet_dir / name).exists()]
        packet_status = "review_complete" if not missing_outputs else "missing_required_outputs"

        row = {
            "case_key": (
                recommendation.iloc[0].get("case_key", "")
                if not recommendation.empty
                else str(manifest.get("case_key", packet_dir.name))
            ),
            "jurisdiction_id": (
                recommendation.iloc[0].get("jurisdiction_id", "")
                if not recommendation.empty
                else str(manifest.get("jurisdiction_id", ""))
            ),
            "jurisdiction_name": (
                recommendation.iloc[0].get("jurisdiction_name", "")
                if not recommendation.empty
                else str(manifest.get("jurisdiction_name", ""))
            ),
            "state_abbr": (
                recommendation.iloc[0].get("state_abbr", "")
                if not recommendation.empty
                else str(manifest.get("state_abbr", ""))
            ),
            "packet_dir": str(packet_dir.resolve()),
            "packet_status": packet_status,
            "missing_required_outputs": "|".join(missing_outputs),
            "has_manifest": int(manifest_path.exists()),
            "has_research_findings": int(findings_path.exists()),
            "has_sources": int(sources_path.exists()),
            "has_recommendation": int(recommendation_path.exists()),
            "review_goal": str(manifest.get("review_goal", "")),
            "recommended_disposition": recommendation.iloc[0].get("recommended_disposition", "") if not recommendation.empty else "",
            "production_ready": _normalize_yes_no(recommendation.iloc[0].get("production_ready", "")) if not recommendation.empty else "",
            "confidence": recommendation.iloc[0].get("confidence", "") if not recommendation.empty else "",
            "summary": recommendation.iloc[0].get("summary", "") if not recommendation.empty else "",
        }
        extract_files = _list_extract_files(packet_dir)
        row["has_promotable_extract"] = int(bool(extract_files))
        row["extract_files"] = "|".join(extract_files)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.merge(
            impact,
            on=["jurisdiction_id", "jurisdiction_name", "state_abbr"],
            how="left",
        )
        summary["non_observed_count_2024"] = pd.to_numeric(
            summary.get("non_observed_count_2024"), errors="coerce"
        ).fillna(0.0)
        summary = summary.sort_values(
            ["production_ready", "has_promotable_extract", "non_observed_count_2024", "state_abbr", "jurisdiction_name"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )

    out_path = packet_root / "packet_status_summary.csv"
    summary.to_csv(out_path, index=False)
    print(json.dumps({"rows": int(len(summary)), "out_path": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
