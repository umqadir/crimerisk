from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PRODUCTION_SCOPE_EXCLUDE = {"AK", "AS", "CZ", "GU", "HI", "MP", "PR", "VI"}
STRUCTURAL_SERVICE_TYPES = {"constable_marshal"}
STRUCTURAL_SERVICE_NAME_RE = re.compile(
    r"\b(?:CAPITOL POLICE|EMERSON COLLEGE|MIDDLESEX MET DIST COMM|FAIRMONT PARK|CEDAR POINT)\b"
)


def _structural_service_name_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.contains(STRUCTURAL_SERVICE_NAME_RE)


def build_queue(*, repo_root: Path, year: int) -> pd.DataFrame:
    review = pd.read_csv(
        repo_root / "state" / "review" / "queues" / "local_resolution" / "agency_jurisdiction_review.csv"
    )
    regimes = pd.read_parquet(repo_root / "state" / "modeling" / "agency_year_reporting_regimes.parquet")
    review["state_abbr"] = review["state_abbr"].astype("string").str.upper()
    regimes["state_abbr"] = regimes["state_abbr"].astype("string").str.upper()
    review = review[~review["state_abbr"].isin(PRODUCTION_SCOPE_EXCLUDE)].copy()
    regimes = regimes[~regimes["state_abbr"].isin(PRODUCTION_SCOPE_EXCLUDE)].copy()

    type_mask = review["agency_type_norm"].astype("string").isin(STRUCTURAL_SERVICE_TYPES)
    name_mask = _structural_service_name_mask(review["agency_name_std"])
    queue = review[type_mask | name_mask].copy()
    reg_year = regimes[regimes["year"].astype(int).eq(int(year))].copy()
    support = (
        reg_year.groupby("ori9", dropna=False)
        .agg(
            total_support_2024=("support_weight", "sum"),
            offense_count_2024=("offense", "nunique"),
            dominant_preferred_source_2024=(
                "preferred_source_by_regime",
                lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else pd.NA,
            ),
        )
        .reset_index()
    )
    queue = queue.merge(support, on="ori9", how="left")
    queue["total_support_2024"] = pd.to_numeric(queue["total_support_2024"], errors="coerce").fillna(0.0)
    queue = queue.sort_values(
        ["total_support_2024", "state_abbr", "agency_name_std"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    cols = [
        "ori9",
        "state_fips",
        "state_abbr",
        "agency_name_std",
        "agency_type_norm",
        "county_fips",
        "place_fips",
        "name",
        "namelsad",
        "provisional_jurisdiction_id",
        "match_method",
        "candidate_summary",
        "manual_review_flag",
        "total_support_2024",
        "offense_count_2024",
        "dominant_preferred_source_2024",
    ]
    for col in cols:
        if col not in queue.columns:
            queue[col] = pd.NA
    return queue[cols].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a structured-service queue for constable/marshal and similar non-primary-patrol local rows.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("state/review/queues/local_resolution/current/structural_service_queue_2024"),
    )
    args = parser.parse_args()

    queue = build_queue(repo_root=REPO_ROOT, year=int(args.year))
    out_base = REPO_ROOT / args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    queue.to_parquet(out_base.with_suffix(".parquet"), index=False)
    queue.to_csv(out_base.with_suffix(".csv"), index=False)
    print(
        {
            "year": int(args.year),
            "rows": int(len(queue)),
            "unique_oris": int(queue["ori9"].nunique()),
            "total_support_2024": float(queue["total_support_2024"].sum()),
            "out_base": str(out_base),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
