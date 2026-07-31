from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths  # noqa: E402
from crimerisk.review_batches import (  # noqa: E402
    build_local_second_pass_cases,
    write_batched_cases,
)

LOCAL_QUEUE_PATH = REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "local_agency_manual_review.parquet"
FIRST_PASS_RESULTS_PATH = (
    REPO_ROOT / "state" / "review" / "runs" / "local_resolution" / "local_queue_clean_run" / "local_queue_full_v1_results.csv"
)
LEAIC_PATH = REPO_ROOT / "data" / "LEAIC-Crosswalk-ICPSR_35158" / "DS0001" / "35158-0001-Data.tsv"
DEFAULT_OUT_DIR = REPO_ROOT / "state" / "review" / "runs" / "local_resolution" / "local_queue_second_pass"


def build_cases() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    queue = pd.read_parquet(LOCAL_QUEUE_PATH).reset_index(drop=True)
    first_pass = pd.read_csv(FIRST_PASS_RESULTS_PATH)
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    cases = build_local_second_pass_cases(
        local_queue_path=LOCAL_QUEUE_PATH,
        first_pass_results_path=FIRST_PASS_RESULTS_PATH,
        leaic_path=LEAIC_PATH,
        paths=paths,
    )

    state_key = queue["state_abbr"].fillna("").astype(str)
    name_key = queue["agency_name_std"].fillna("").astype(str)
    repeat_key = state_key + "|" + name_key
    repeat_counts = repeat_key.value_counts(dropna=False).to_dict()

    merged = queue.merge(first_pass, on="ori9", how="left", suffixes=("", "_first"))

    is_escalate = merged["decision"].eq("escalate")
    is_low_conf = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0) < 0.8
    is_municipal = merged["decision"].isin(["municipal_place", "municipal_cousub"])
    has_null_geoid = merged["resolved_geoid"].isna() | (merged["resolved_geoid"].astype(str).str.strip() == "")
    is_repeat_cluster = (repeat_key.map(repeat_counts).fillna(0).astype(int) > 1) & name_key.ne("")

    second_pass_mask = is_escalate | is_low_conf | (is_municipal & has_null_geoid) | is_repeat_cluster
    second_pass_df = merged.loc[second_pass_mask].copy()
    second_pass_df["flag_escalate"] = is_escalate.loc[second_pass_df.index].astype(bool)
    second_pass_df["flag_low_confidence"] = is_low_conf.loc[second_pass_df.index].astype(bool)
    second_pass_df["flag_municipal_missing_geoid"] = (is_municipal & has_null_geoid).loc[
        second_pass_df.index
    ].astype(bool)
    second_pass_df["flag_repeat_name_cluster"] = is_repeat_cluster.loc[second_pass_df.index].astype(bool)

    audit_df = second_pass_df[
        [
            "ori9",
            "state_abbr",
            "agency_name_std",
            "agency_type_norm",
            "match_status",
            "match_method",
            "candidate_summary",
            "latest_srs_part1_total",
            "review_priority",
            "decision",
            "confidence",
            "resolved_geo_type",
            "resolved_geoid",
            "resolved_label",
        ]
    ].copy()
    audit_df["repeat_cluster_size"] = repeat_key.loc[second_pass_df.index].map(repeat_counts).astype(int)
    audit_df["flag_escalate"] = second_pass_df["flag_escalate"].to_numpy()
    audit_df["flag_low_confidence"] = second_pass_df["flag_low_confidence"].to_numpy()
    audit_df["flag_missing_first_pass_result"] = second_pass_df["decision"].isna().to_numpy()
    audit_df["flag_municipal_missing_geoid"] = second_pass_df["flag_municipal_missing_geoid"].to_numpy()
    audit_df["flag_repeat_name_cluster"] = second_pass_df["flag_repeat_name_cluster"].to_numpy()
    return cases, audit_df.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    cases, audit_df = build_cases()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_batched_cases(cases, out_dir, args.batch_size)
    audit_df.to_parquet(out_dir / "local_second_pass_queue.parquet", index=False)
    audit_df.to_csv(out_dir / "local_second_pass_queue.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "cases": len(cases),
                "batch_size": args.batch_size,
                "batches": (len(cases) + args.batch_size - 1) // args.batch_size,
                "flag_counts": {
                    "first_pass_escalate": int(audit_df["flag_escalate"].sum()),
                    "first_pass_low_confidence": int(audit_df["flag_low_confidence"].sum()),
                    "missing_first_pass_result": int(audit_df["flag_missing_first_pass_result"].sum()),
                    "municipal_missing_geoid": int(audit_df["flag_municipal_missing_geoid"].sum()),
                    "repeat_name_cluster": int(audit_df["flag_repeat_name_cluster"].sum()),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"out_dir": str(out_dir), "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
