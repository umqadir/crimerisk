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


SUPPORTED_TREATMENTS = {
    "localize_to_place",
    "localize_to_county",
    "keep_statewide_overlap",
    "localize_to_custom_footprint",
    "absorb_into_primary_jurisdiction",
}


def _load_queue(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        queue = pd.read_csv(path)
    else:
        queue = pd.read_parquet(path)
    if "case_id" not in queue.columns and "ori" in queue.columns:
        queue = queue.copy()
        queue["case_id"] = queue["ori"].astype("string")
    if "case_id" in queue.columns:
        queue["case_id"] = queue["case_id"].astype("string")
    if "ori" in queue.columns:
        queue["ori"] = queue["ori"].astype("string")
    return queue


def _load_batch_inputs(batch_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(batch_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a batch list")
        for row in payload:
            item = dict(row)
            if "case_id" not in item and "ori" in item:
                item["case_id"] = item["ori"]
            item["case_id"] = str(item["case_id"])
            if "ori" in item:
                item["ori"] = str(item["ori"])
            item["batch_file"] = path.name
            rows.append(item)
    return pd.DataFrame(rows)


def _load_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a normalized result list")
        for row in payload:
            row = dict(row)
            row["batch_file"] = path.name
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "case_id",
                "ori",
                "review_status",
                "recommended_final_overlap_treatment",
                "overlap_subtype_final",
                "footprint_type",
                "target_state_fips",
                "target_county_fips",
                "target_place_fips",
                "target_jurisdiction_id",
                "geometry_source_type",
                "geometry_source_ref",
                "confidence",
                "source_note",
                "reviewer_note",
                "sources",
                "escalation_reason",
                "batch_file",
            ]
        )
    df = pd.DataFrame(rows)
    df["case_id"] = df["case_id"].astype("string")
    df["ori"] = df["ori"].astype("string")
    return df


def _to_override_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ori"] = out["ori"].astype("string")
    out["target_state_fips"] = out["target_state_fips"].astype("string")
    out["target_county_fips"] = out["target_county_fips"].astype("string")
    out["target_place_fips"] = out["target_place_fips"].astype("string")
    cols = [
        "ori",
        "recommended_final_overlap_treatment",
        "overlap_subtype_final",
        "footprint_type",
        "target_state_fips",
        "target_county_fips",
        "target_place_fips",
        "target_jurisdiction_id",
        "geometry_source_type",
        "geometry_source_ref",
        "confidence",
        "source_note",
        "reviewer_note",
    ]
    renamed = out[cols].rename(columns={"recommended_final_overlap_treatment": "final_overlap_treatment"})
    return renamed.drop_duplicates("ori", keep="last").copy()


def _merge_overrides(existing: pd.DataFrame, accepted: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return accepted.copy()
    merged = existing.copy()
    merged["ori"] = merged["ori"].astype("string")
    accepted = accepted.copy()
    accepted["ori"] = accepted["ori"].astype("string")
    merged = merged[~merged["ori"].isin(accepted["ori"])]
    merged = pd.concat([merged, accepted], ignore_index=True)
    return merged.sort_values(["ori"], kind="mergesort").reset_index(drop=True)


def _load_custom_footprint_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    custom = pd.read_csv(path)
    if custom.empty or "ori" not in custom.columns:
        return set()
    custom["ori"] = custom["ori"].astype("string").str.strip()
    state_series = (
        custom["state_fips"].astype("string").str.strip().str.zfill(2)
        if "state_fips" in custom.columns
        else pd.Series(pd.NA, index=custom.index, dtype="string")
    )
    return {
        (str(ori), str(state_fips))
        for ori, state_fips in zip(custom["ori"], state_series)
        if pd.notna(ori) and pd.notna(state_fips)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile overlap localization swarm results into accepted and deferred outputs.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--write-overrides", action="store_true")
    parser.add_argument("--override-path", type=Path, default=Path("configs/overlap_footprint_overrides.csv"))
    parser.add_argument("--custom-footprint-path", type=Path, default=Path("configs/overlap_custom_footprints.csv"))
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
            / "overlap"
            / run_root.name
            / "compiled"
        ).resolve()
    else:
        out_dir = (REPO_ROOT / args.out_dir).resolve()
    override_path = (REPO_ROOT / args.override_path).resolve()
    custom_footprint_path = (REPO_ROOT / args.custom_footprint_path).resolve()

    queue = _load_queue(queue_path)
    results = _load_results(run_root / "results")
    if results.empty:
        raise SystemExit("No normalized result files found.")
    batch_inputs = _load_batch_inputs(run_root / "batches")
    if batch_inputs.empty:
        raise SystemExit("No batch input files found.")

    base = batch_inputs if not batch_inputs.empty else queue
    merged = base.merge(results, on=["case_id", "ori"], how="inner", suffixes=("", "_result"))
    if "state_fips" in merged.columns:
        merged["state_fips"] = merged["state_fips"].astype("string").str.strip().str.zfill(2)
    merged["target_jurisdiction_id"] = merged["target_jurisdiction_id"].astype("string").str.strip()
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)
    custom_footprint_keys = _load_custom_footprint_keys(custom_footprint_path)

    base_accept_mask = (
        merged["review_status"].eq("resolved")
        & (merged["confidence"] >= float(args.min_confidence))
        & merged["recommended_final_overlap_treatment"].isin(SUPPORTED_TREATMENTS)
    )
    treatment = merged["recommended_final_overlap_treatment"].astype("string")
    always_supported_mask = treatment.isin(
        {"localize_to_place", "localize_to_county", "keep_statewide_overlap"}
    )
    custom_supported_mask = pd.Series(False, index=merged.index)
    if custom_footprint_keys and "state_fips" in merged.columns:
        custom_supported_mask = (
            treatment.eq("localize_to_custom_footprint")
            & merged.apply(
                lambda row: (str(row["ori"]), str(row["state_fips"])) in custom_footprint_keys,
                axis=1,
            )
        )
    absorb_supported_mask = (
        treatment.eq("absorb_into_primary_jurisdiction")
        & merged["target_jurisdiction_id"].notna()
        & merged["target_jurisdiction_id"].ne("")
    )
    accepted = merged[base_accept_mask & (always_supported_mask | custom_supported_mask | absorb_supported_mask)].copy()
    needs_followup = merged.loc[~merged.index.isin(accepted.index)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "overlap_localization_compiled_results.parquet", index=False)
    accepted.to_parquet(out_dir / "overlap_localization_accepted_supported.parquet", index=False)
    needs_followup.to_parquet(out_dir / "overlap_localization_needs_followup.parquet", index=False)

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
        "accepted_treatment_counts": accepted["recommended_final_overlap_treatment"].value_counts(dropna=False).to_dict(),
        "needs_followup_status_counts": needs_followup["review_status"].value_counts(dropna=False).to_dict(),
        "needs_followup_treatment_counts": needs_followup["recommended_final_overlap_treatment"].value_counts(dropna=False).to_dict(),
    }
    print(summary)


if __name__ == "__main__":
    main()
