from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_batches(batch_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(batch_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        for row in payload:
            item = dict(row)
            item["_batch_file"] = path.name
            rows.append(item)
    return pd.DataFrame(rows)


def _load_results(result_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(result_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        for row in payload:
            item = dict(row)
            item["_result_file"] = path.name
            rows.append(item)
    return pd.DataFrame(rows)


def summarize(run_root: Path, lane: str) -> dict[str, object]:
    lane_root = run_root / lane
    batches = _load_batches(lane_root / "batches")
    results = _load_results(lane_root / "results")
    if batches.empty or results.empty:
        return {"lane": lane, "run_root": str(lane_root), "rows": 0}

    merged = batches.merge(results, on="case_id", how="inner", suffixes=("_input", "_result"))
    if "overlap_weight_2024_input" in merged.columns:
        merged["overlap_weight_2024"] = pd.to_numeric(merged["overlap_weight_2024_input"], errors="coerce").fillna(0.0)
    elif "overlap_weight_2024" in merged.columns:
        merged["overlap_weight_2024"] = pd.to_numeric(merged["overlap_weight_2024"], errors="coerce").fillna(0.0)
    else:
        merged["overlap_weight_2024"] = 0.0

    summary = {
        "lane": lane,
        "run_root": str(lane_root),
        "rows": int(len(merged)),
        "treatment_counts": merged["final_overlap_treatment"].value_counts(dropna=False).to_dict(),
        "escalation_counts": merged["requires_escalation"].value_counts(dropna=False).to_dict(),
        "total_weight_reviewed": float(merged["overlap_weight_2024"].sum()),
        "weight_by_treatment": merged.groupby("final_overlap_treatment", dropna=False)["overlap_weight_2024"].sum().to_dict(),
        "top_rows": merged.sort_values("overlap_weight_2024", ascending=False)[
            [
                "case_id",
                "ori_input",
                "state_abbr_input",
                "agency_name_std_input",
                "overlap_subtype_input",
                "geometry_hint_input",
                "overlap_weight_2024",
                "final_overlap_treatment",
                "requires_escalation",
                "confidence",
                "reviewer_note",
            ]
        ].head(15).to_dict(orient="records"),
    }
    summary_path = lane_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize overlap localization swarm results for one lane.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lane", choices=["proxy", "footprint"], required=True)
    args = parser.parse_args()
    summary = summarize(args.run_root, args.lane)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
