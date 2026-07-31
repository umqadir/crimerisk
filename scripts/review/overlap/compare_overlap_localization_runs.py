from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


COMPARE_COLS = [
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
    "escalation_reason",
]


def _load_run(run_root: Path) -> pd.DataFrame:
    path = run_root / "postprocessed" / "all_results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing postprocessed results: {path}")
    return pd.read_parquet(path).copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two overlap localization runs by case_id.")
    parser.add_argument("--left-run", type=Path, required=True)
    parser.add_argument("--right-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    left_raw = _load_run(args.left_run.resolve())
    right_raw = _load_run(args.right_run.resolve())
    left = left_raw.rename(columns={c: f"{c}_left" for c in COMPARE_COLS if c in left_raw.columns})
    right = right_raw.rename(columns={c: f"{c}_right" for c in COMPARE_COLS if c in right_raw.columns})

    merged = left.merge(
        right,
        on=["case_id", "ori"],
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
    )
    for col in COMPARE_COLS:
        left_col = f"{col}_left"
        right_col = f"{col}_right"
        if left_col in merged.columns and right_col in merged.columns:
            merged[f"{col}_match"] = merged[left_col].fillna("<NA>").astype(str).eq(merged[right_col].fillna("<NA>").astype(str))

    merged["overall_match"] = True
    for col in COMPARE_COLS:
        match_col = f"{col}_match"
        if match_col in merged.columns:
            merged["overall_match"] &= merged[match_col].fillna(False)

    out_base = args.out.resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_base.with_suffix(".parquet"), index=False)
    merged.to_csv(out_base.with_suffix(".csv"), index=False)

    summary = {
        "left_run": str(args.left_run.resolve()),
        "right_run": str(args.right_run.resolve()),
        "rows": int(len(merged)),
        "overall_match_count": int(merged["overall_match"].sum()),
        "overall_mismatch_count": int((~merged["overall_match"]).sum()),
        "merge_indicator": merged["_merge"].value_counts(dropna=False).to_dict(),
    }
    out_base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
