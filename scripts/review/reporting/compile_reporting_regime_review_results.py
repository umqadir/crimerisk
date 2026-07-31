from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SUPPORTED_REPORTING_REGIMES = {
    "full_monthly",
    "true_partial",
    "lumpy_or_batched",
    "annual_only_but_usable",
    "structurally_missing_or_unreliable",
}


def _load_queue(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        queue = pd.read_csv(path)
    else:
        queue = pd.read_parquet(path)
    if "case_id" not in queue.columns:
        queue["case_id"] = (
            queue["ori"].astype("string")
            + ":"
            + pd.to_numeric(queue["year"], errors="coerce").astype("Int64").astype("string")
            + ":"
            + queue["offense"].astype("string")
        )
    queue["case_id"] = queue["case_id"].astype("string")
    queue["ori"] = queue["ori"].astype("string")
    queue["year"] = pd.to_numeric(queue["year"], errors="coerce").astype("Int64")
    queue["offense"] = queue["offense"].astype("string")
    return queue


def _load_batch_inputs(batch_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(batch_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a batch list")
        for row in payload:
            item = dict(row)
            item["case_id"] = str(item["case_id"])
            item["ori"] = str(item["ori"])
            item["year"] = int(item["year"])
            item["offense"] = str(item["offense"])
            item["batch_file"] = path.name
            rows.append(item)
    return pd.DataFrame(rows)


def _filter_to_completed_batches(batch_inputs: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if batch_inputs.empty or results.empty:
        return batch_inputs.iloc[0:0].copy()
    completed_batches = set(results["batch_file"].astype("string").dropna().tolist())
    if not completed_batches:
        return batch_inputs.iloc[0:0].copy()
    return batch_inputs.loc[batch_inputs["batch_file"].astype("string").isin(completed_batches)].copy()


def _load_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a normalized result list")
        for row in payload:
            item = dict(row)
            item["batch_file"] = path.name
            rows.append(item)
    if not rows:
        return pd.DataFrame(
            columns=[
                "case_id",
                "ori",
                "year",
                "offense",
                "review_status",
                "final_reporting_regime",
                "evidence_type",
                "source_note",
                "confidence",
                "reviewer_note",
                "sources",
                "escalation_reason",
                "batch_file",
            ]
        )
    df = pd.DataFrame(rows)
    df["case_id"] = df["case_id"].astype("string")
    df["ori"] = df["ori"].astype("string")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["offense"] = df["offense"].astype("string")
    dupes = df.loc[df.duplicated(["case_id"], keep=False), ["case_id", "batch_file"]]
    if not dupes.empty:
        raise ValueError(f"Duplicate normalized result rows: {dupes.to_dict(orient='records')}")
    return df


def _to_override_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ori"] = out["ori"].astype("string")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["offense"] = out["offense"].astype("string")
    cols = [
        "ori",
        "year",
        "offense",
        "final_reporting_regime",
        "evidence_type",
        "source_note",
        "confidence",
        "reviewer_note",
    ]
    return out[cols].drop_duplicates(["ori", "year", "offense"], keep="last").copy()


def _merge_overrides(existing: pd.DataFrame, accepted: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return accepted.copy()
    merged = existing.copy()
    merged["ori"] = merged["ori"].astype("string")
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("Int64")
    merged["offense"] = merged["offense"].astype("string")
    accepted = accepted.copy()
    accepted["ori"] = accepted["ori"].astype("string")
    accepted["year"] = pd.to_numeric(accepted["year"], errors="coerce").astype("Int64")
    accepted["offense"] = accepted["offense"].astype("string")
    merged = merged[
        ~merged.set_index(["ori", "year", "offense"]).index.isin(accepted.set_index(["ori", "year", "offense"]).index)
    ]
    merged = pd.concat([merged, accepted], ignore_index=True)
    return merged.sort_values(["ori", "year", "offense"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile reporting-regime swarm results into accepted overrides and follow-up.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--write-overrides", action="store_true")
    parser.add_argument("--override-path", type=Path, default=Path("configs/reporting_regime_overrides.csv"))
    args = parser.parse_args()

    queue_path = (REPO_ROOT / args.queue).resolve()
    run_root = (REPO_ROOT / args.run_root).resolve()
    if args.out_dir is None:
        out_dir = (
            REPO_ROOT
            / "state"
            / "artifacts"
            / "v2"
            / "review"
            / "analysis"
            / "reporting"
            / run_root.name
            / "compiled"
        ).resolve()
    else:
        out_dir = (REPO_ROOT / args.out_dir).resolve()
    override_path = (REPO_ROOT / args.override_path).resolve()

    queue = _load_queue(queue_path)
    results = _load_results(run_root / "results")
    if results.empty:
        raise SystemExit("No normalized result files found.")
    batch_inputs = _load_batch_inputs(run_root / "batches")
    if batch_inputs.empty:
        raise SystemExit("No batch input files found.")
    total_batch_count = int(batch_inputs["batch_file"].astype("string").nunique())
    batch_inputs = _filter_to_completed_batches(batch_inputs, results)
    if batch_inputs.empty:
        raise SystemExit("No completed batch inputs matched normalized result files.")
    completed_batch_count = int(batch_inputs["batch_file"].astype("string").nunique())

    batch_keys = batch_inputs[["case_id", "ori", "year", "offense"]].drop_duplicates()
    if len(batch_keys) != len(batch_inputs):
        raise ValueError("Duplicate cases found in batch inputs.")

    merged = batch_inputs.merge(
        results,
        on=["case_id", "ori", "year", "offense"],
        how="left",
        suffixes=("", "_result"),
    )
    if merged["review_status"].isna().any():
        missing = merged.loc[merged["review_status"].isna(), ["case_id", "batch_file"]]
        raise ValueError(f"Missing normalized result rows for some batch inputs: {missing.to_dict(orient='records')}")

    queued = queue[["case_id", "ori", "year", "offense"]].drop_duplicates()
    merged = queued.merge(merged, on=["case_id", "ori", "year", "offense"], how="inner")
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)

    accepted = merged[
        merged["review_status"].eq("resolved")
        & merged["final_reporting_regime"].isin(SUPPORTED_REPORTING_REGIMES)
        & (merged["confidence"] >= float(args.min_confidence))
    ].copy()
    needs_followup = merged.loc[~merged.index.isin(accepted.index)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "reporting_regime_compiled_results.parquet", index=False)
    accepted.to_parquet(out_dir / "reporting_regime_accepted_overrides.parquet", index=False)
    needs_followup.to_parquet(out_dir / "reporting_regime_needs_followup.parquet", index=False)

    if args.write_overrides:
        override_rows = _to_override_rows(accepted)
        if override_path.exists():
            existing = pd.read_csv(override_path)
        else:
            existing = pd.DataFrame(columns=override_rows.columns)
        updated = _merge_overrides(existing, override_rows)
        updated.to_csv(override_path, index=False)

    summary = {
        "results_rows": int(len(merged)),
        "accepted_rows": int(len(accepted)),
        "needs_followup_rows": int(len(needs_followup)),
        "completed_batch_count": completed_batch_count,
        "skipped_unfinished_batch_count": int(total_batch_count - completed_batch_count),
        "accepted_regime_counts": accepted["final_reporting_regime"].value_counts(dropna=False).to_dict(),
        "accepted_evidence_counts": accepted["evidence_type"].value_counts(dropna=False).to_dict(),
        "needs_followup_status_counts": needs_followup["review_status"].value_counts(dropna=False).to_dict(),
        "needs_followup_regime_counts": needs_followup["final_reporting_regime"].value_counts(dropna=False).to_dict(),
    }
    print(summary)


if __name__ == "__main__":
    main()
