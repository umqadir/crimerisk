from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import get_paths
from crimerisk.review_batches import build_local_enriched_cases, write_case_batches


RUN_ROOT = REPO_ROOT / "state" / "review" / "runs" / "local_resolution" / "local_queue_clean_run"
LOCAL_QUEUE_PATH = REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "local_agency_manual_review.parquet"
LEAIC_PATH = REPO_ROOT / "data" / "LEAIC-Crosswalk-ICPSR_35158" / "DS0001" / "35158-0001-Data.tsv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RUN_ROOT)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--local-queue", type=Path, default=LOCAL_QUEUE_PATH)
    parser.add_argument("--leaic", type=Path, default=LEAIC_PATH)
    args = parser.parse_args()

    paths = get_paths()
    cases = build_local_enriched_cases(
        local_queue_path=args.local_queue,
        leaic_path=args.leaic,
        paths=paths,
    )
    write_case_batches(cases, args.out_dir, args.batch_size)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "batch_size": args.batch_size,
                "batches": (len(cases) + args.batch_size - 1) // args.batch_size,
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
