from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SUPPORTED_ALLOCATOR_TREATMENTS = {
    "localize_to_place",
    "localize_to_county",
    "keep_statewide_overlap",
}


def _load_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected payload shape in {path}")
        for row in payload:
            item = dict(row)
            item["result_file"] = str(path)
            rows.append(item)
    return pd.DataFrame(rows)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path.with_suffix(".parquet"), index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Flatten and summarize overlap localization run outputs.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    results_dir = run_root / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    results = _load_results(results_dir)
    if results.empty:
        raise ValueError(f"No overlap localization results found in {results_dir}")

    if "recommended_final_overlap_treatment" not in results.columns and "final_overlap_treatment" in results.columns:
        results["recommended_final_overlap_treatment"] = results["final_overlap_treatment"]
    if "review_status" not in results.columns:
        if "requires_escalation" in results.columns:
            results["review_status"] = results["requires_escalation"].fillna(False).map(
                {True: "needs_escalation", False: "resolved"}
            )
        else:
            results["review_status"] = "resolved"
    if "requires_escalation" not in results.columns:
        results["requires_escalation"] = results["review_status"].eq("needs_escalation")
    else:
        results["requires_escalation"] = results["requires_escalation"].fillna(False).astype(bool)

    results["is_allocator_supported"] = results["recommended_final_overlap_treatment"].isin(SUPPORTED_ALLOCATOR_TREATMENTS)
    results["is_directly_ingestible"] = results["is_allocator_supported"] & (~results["requires_escalation"])

    out_dir = run_root / "postprocessed"
    _write_frame(results, out_dir / "all_results")
    _write_frame(results.loc[~results["requires_escalation"]].copy(), out_dir / "accepted_results")
    _write_frame(results.loc[results["requires_escalation"]].copy(), out_dir / "escalation_queue")
    _write_frame(results.loc[results["is_directly_ingestible"]].copy(), out_dir / "allocator_ready_results")

    summary = {
        "run_root": str(run_root),
        "rows": int(len(results)),
        "review_status_counts": results["review_status"].value_counts(dropna=False).to_dict(),
        "treatment_counts": results["recommended_final_overlap_treatment"].value_counts(dropna=False).to_dict(),
        "escalation_count": int(results["requires_escalation"].sum()),
        "allocator_ready_count": int(results["is_directly_ingestible"].sum()),
        "allocator_ready_treatment_counts": (
            results.loc[results["is_directly_ingestible"], "recommended_final_overlap_treatment"]
            .value_counts(dropna=False)
            .to_dict()
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
