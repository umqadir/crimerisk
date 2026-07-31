from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_resolved_override_oris(repo_root: Path) -> set[str]:
    path = repo_root / "configs" / "local_resolution_overrides.csv"
    if not path.exists():
        return set()
    overrides = pd.read_csv(path).copy()
    if "ori" not in overrides.columns:
        return set()
    return set(overrides["ori"].astype("string").str.strip().dropna())


def build_queue(*, repo_root: Path, top_n: int) -> pd.DataFrame:
    queue = pd.read_parquet(
        repo_root / "state" / "review" / "queues" / "local_resolution" / "local_agency_manual_review.parquet"
    ).copy()
    risk_mask = pd.Series(False, index=queue.index)
    for col in ["zero_overlap_place_preferred_flag", "county_place_alias_risk_flag"]:
        if col in queue.columns:
            risk_mask = risk_mask | queue[col].fillna(False)
    focus = queue[risk_mask].copy()
    resolved_override_oris = _load_resolved_override_oris(repo_root)
    if resolved_override_oris and "ori9" in focus.columns:
        focus = focus.loc[~focus["ori9"].astype("string").str.strip().isin(resolved_override_oris)].copy()
    focus = focus.sort_values(
        ["review_priority", "state_abbr", "agency_name_std"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(int(top_n))
    return focus.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a focused high-support local reference-risk queue for worker review.")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("state/review/queues/local_resolution/current/reference_risk_focus_queue_2024"),
    )
    args = parser.parse_args()

    queue = build_queue(repo_root=REPO_ROOT, top_n=int(args.top_n))
    out_base = REPO_ROOT / args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    queue.to_parquet(out_base.with_suffix(".parquet"), index=False)
    queue.to_csv(out_base.with_suffix(".csv"), index=False)
    print(
        {
            "rows": int(len(queue)),
            "unique_oris": int(queue["ori9"].nunique()),
            "top_n": int(args.top_n),
            "out_base": str(out_base),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
