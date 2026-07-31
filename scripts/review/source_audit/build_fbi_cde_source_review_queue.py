from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_queue(*, repo_root: Path, top_n: int) -> dict[str, str | int]:
    validation_path = (
        repo_root
        / "state"
        / "review"
        / "analysis"
        / "source_audit"
        / "fbi_cde_roster_validation"
        / "ori_source_validation_2024.parquet"
    )
    state_priority_path = (
        repo_root
        / "state" / "review"
        / "analysis"
        / "source_audit"
        / "state_source_priority_2024.parquet"
    )
    out_dir = repo_root / "state" / "review" / "queues" / "source_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    val = pd.read_parquet(validation_path).copy()
    val["repo_preferred_support_count_2024"] = pd.to_numeric(
        val["repo_preferred_support_count_2024"], errors="coerce"
    ).fillna(0.0)
    val = val[val["mismatch_reason"].astype(str).ne("aligned")].copy()

    lane = pd.Series("other", index=val.index, dtype="string")
    lane.loc[
        val["mismatch_reason"].astype(str).str.contains("official_nibrs_no_repo_nibrs|official_start_by_2024_but_repo_no_nibrs", regex=True, na=False)
    ] = "official_nibrs_repo_no_nibrs"
    lane.loc[val["official_repo_presence_flag"].astype(str).eq("present_but_no_2024_observation")] = "present_but_no_2024_observation"
    lane.loc[val["mismatch_reason"].astype(str).str.contains("missing_in_repo_master", na=False)] = "missing_in_repo_master"
    val["review_lane"] = lane
    val = val[~val["official_repo_transition_flag"].astype(str).eq("official_nibrs_start_after_2024")].copy()

    state_priority = pd.read_parquet(state_priority_path).copy()
    state_priority["state_abbr"] = state_priority["state_abbr"].astype("string").str.upper()
    priority_col = "gap_priority_score" if "gap_priority_score" in state_priority.columns else "priority_score_abs"
    if priority_col not in state_priority.columns:
        state_priority[priority_col] = 0.0
    state_priority[priority_col] = pd.to_numeric(
        state_priority[priority_col], errors="coerce"
    ).fillna(0.0)
    state_summary = (
        state_priority.groupby("state_abbr", dropna=False)
        .agg(
            state_priority_score_abs=(priority_col, "max"),
        )
        .reset_index()
    )
    val = val.merge(state_summary, on="state_abbr", how="left")
    val["state_priority_score_abs"] = pd.to_numeric(val["state_priority_score_abs"], errors="coerce").fillna(0.0)

    val = val.sort_values(
        ["repo_preferred_support_count_2024", "state_priority_score_abs", "state_abbr", "ori9"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    val["priority_rank"] = range(1, len(val) + 1)
    if top_n > 0:
        val = val.head(int(top_n)).copy()

    keep_cols = [
        "priority_rank",
        "review_lane",
        "state_abbr",
        "state_fips",
        "ori9",
        "agency_name",
        "agency_type_name",
        "official_is_nibrs",
        "official_effective_nibrs_2024",
        "official_nibrs_start_date",
        "repo_has_srs_2024",
        "repo_has_nibrs_2024",
        "repo_preferred_source_any_offense_2024",
        "repo_reporting_regime_mix_2024",
        "repo_preferred_support_count_2024",
        "official_repo_presence_flag",
        "official_repo_transition_flag",
        "mismatch_reason",
        "state_priority_score_abs",
    ]
    out = val[keep_cols].copy()
    parquet_path = out_dir / "fbi_cde_source_review_queue.parquet"
    csv_path = out_dir / "fbi_cde_source_review_queue.csv"
    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)
    return {
        "rows": int(len(out)),
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an official FBI/CDE-driven source review queue for 2024.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()
    print(build_queue(repo_root=args.repo_root.resolve(), top_n=int(args.top_n)))


if __name__ == "__main__":
    main()
