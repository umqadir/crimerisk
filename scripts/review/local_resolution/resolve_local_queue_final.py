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

from crimerisk.paths import get_paths
from crimerisk.local_resolution import resolve_local_queue_final


LOCAL_QUEUE_PATH = REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "local_agency_manual_review.parquet"
FIRST_PASS_RESULTS_PATH = (
    REPO_ROOT
    / "state" / "review"
    / "runs"
    / "local_resolution"
    / "local_queue_clean_run"
    / "local_queue_full_v1_results.csv"
)
SECOND_PASS_RESULTS_DIR = (
    REPO_ROOT
    / "state" / "review"
    / "runs"
    / "local_resolution"
    / "local_queue_second_pass"
    / "results_full_v1"
)
OUT_CSV = REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "local_queue_resolved_final.csv"
OUT_PARQUET = REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "local_queue_resolved_final.parquet"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-queue", type=Path, default=LOCAL_QUEUE_PATH)
    parser.add_argument("--first-pass-results", type=Path, default=FIRST_PASS_RESULTS_PATH)
    parser.add_argument("--second-pass-results-dir", type=Path, default=SECOND_PASS_RESULTS_DIR)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-parquet", type=Path, default=OUT_PARQUET)
    args = parser.parse_args()

    paths = get_paths()
    final = resolve_local_queue_final(
        local_queue_path=args.local_queue,
        first_pass_results_path=args.first_pass_results,
        second_pass_results_dir=args.second_pass_results_dir,
        paths=paths,
    )
    for col in ["sources", "first_sources", "second_sources"]:
        if col in final.columns:
            final[col] = final[col].map(
                lambda v: json.dumps(v) if isinstance(v, list) else (None if pd.isna(v) else str(v))
            )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out_csv, index=False)
    final.to_parquet(args.out_parquet, index=False)
    summary = {
        "rows": int(len(final)),
        "final_decision_counts": final["final_decision"].value_counts().to_dict(),
        "fallback_rows": int(final["fallback_applied"].fillna(False).sum()),
        "manual_review_flag_rows": int(final["manual_review_flag"].fillna(False).sum()),
        "municipal_rows_missing_geoid": int(
            (
                final["final_decision"].isin(["municipal_place", "municipal_cousub"])
                & final["resolved_geoid"].isna()
            ).sum()
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
