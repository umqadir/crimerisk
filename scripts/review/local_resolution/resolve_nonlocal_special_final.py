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

from crimerisk.nonlocal_resolution import resolve_nonlocal_queue_final


BASELINE_PATH = (
    REPO_ROOT
    / "state" / "review"
    / "runs"
    / "local_resolution"
    / "nonlocal_second_pass"
    / "nonlocal_deterministic_baseline.parquet"
)
REVIEW_RESULTS_DIR = (
    REPO_ROOT
    / "state" / "review"
    / "runs"
    / "local_resolution"
    / "nonlocal_second_pass"
    / "results_full_v1"
)
HIGH_REVIEW_RESULTS_PATH = (
    REPO_ROOT
    / "state" / "review"
    / "runs"
    / "local_resolution"
    / "nonlocal_second_pass"
    / "high_review_batch_results.json"
)
OUT_CSV = REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "nonmunicipal_special_resolved_final.csv"
OUT_PARQUET = REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "nonmunicipal_special_resolved_final.parquet"
HIGH_REVIEW_CSV = (
    REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "nonmunicipal_special_high_review.csv"
)
HIGH_REVIEW_PARQUET = (
    REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "nonmunicipal_special_high_review.parquet"
)


def _stringify_list_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ["sources", "review_sources"]:
        if col in frame.columns:
            frame[col] = frame[col].map(
                lambda v: json.dumps(v) if isinstance(v, list) else (None if pd.isna(v) else str(v))
            )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=Path, default=BASELINE_PATH)
    parser.add_argument("--review-results-dir", type=Path, default=REVIEW_RESULTS_DIR)
    parser.add_argument("--high-review-results-path", type=Path, default=HIGH_REVIEW_RESULTS_PATH)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-parquet", type=Path, default=OUT_PARQUET)
    parser.add_argument("--high-review-csv", type=Path, default=HIGH_REVIEW_CSV)
    parser.add_argument("--high-review-parquet", type=Path, default=HIGH_REVIEW_PARQUET)
    args = parser.parse_args()

    final, high_review = resolve_nonlocal_queue_final(
        baseline_path=args.baseline_path,
        review_results_dir=args.review_results_dir,
        high_review_results_path=args.high_review_results_path,
    )
    final = _stringify_list_columns(final)
    high_review = _stringify_list_columns(high_review)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out_csv, index=False)
    final.to_parquet(args.out_parquet, index=False)
    high_review.to_csv(args.high_review_csv, index=False)
    high_review.to_parquet(args.high_review_parquet, index=False)

    summary = {
        "rows": int(len(final)),
        "final_bucket_counts": final["final_bucket_decision"].value_counts(dropna=False).to_dict(),
        "resolution_source_counts": final["resolution_source"].value_counts(dropna=False).to_dict(),
        "needs_high_review_rows": int(final["needs_high_review"].fillna(False).sum()),
        "high_review_bucket_counts": high_review["final_bucket_decision"].value_counts(dropna=False).to_dict(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
