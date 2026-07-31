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
STRUCTURAL_SERVICE_EXCLUDE_TYPES = {"constable_marshal"}
STRUCTURAL_SERVICE_NAME_RE = re.compile(
    r"\b(?:CAPITOL POLICE|EMERSON COLLEGE|MIDDLESEX MET DIST COMM|FAIRMONT PARK|CEDAR POINT)\b"
)


def _load_resolved_override_oris(repo_root: Path) -> set[str]:
    path = repo_root / "configs" / "local_resolution_overrides.csv"
    if not path.exists():
        return set()
    overrides = pd.read_csv(path)
    if "ori" not in overrides.columns:
        return set()
    return {
        str(value).strip()
        for value in overrides["ori"].dropna().astype(str)
        if str(value).strip()
    }


def _load_live_crosswalk_status(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    crosswalk = pd.read_parquet(path)
    cols = ["ori", "jurisdiction_id", "relationship_type", "review_status", "resolution_source"]
    return (
        crosswalk[cols]
        .drop_duplicates(subset=["ori"], keep="first")
        .rename(columns={"ori": "ori9", "jurisdiction_id": "final_jurisdiction_id"})
    )


def _structural_service_name_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.contains(STRUCTURAL_SERVICE_NAME_RE)


def build_queue(*, repo_root: Path, year: int) -> pd.DataFrame:
    review = pd.read_csv(
        repo_root / "state" / "review" / "queues" / "local_resolution" / "agency_jurisdiction_review.csv"
    )
    regimes = pd.read_parquet(repo_root / "state" / "modeling" / "agency_year_reporting_regimes.parquet")
    crosswalk = _load_live_crosswalk_status(repo_root)
    review["state_abbr"] = review["state_abbr"].astype("string").str.upper()
    regimes["state_abbr"] = regimes["state_abbr"].astype("string").str.upper()
    review = review[~review["state_abbr"].isin(PRODUCTION_SCOPE_EXCLUDE)].copy()
    regimes = regimes[~regimes["state_abbr"].isin(PRODUCTION_SCOPE_EXCLUDE)].copy()

    if "county_place_alias_risk_flag" not in review.columns:
        raise ValueError(
            "agency_jurisdiction_review.csv is missing county_place_alias_risk_flag; rebuild jurisdiction review first"
        )

    risk = review[review["county_place_alias_risk_flag"].fillna(False)].copy()
    risk = risk[~risk["agency_type_norm"].astype("string").isin(STRUCTURAL_SERVICE_EXCLUDE_TYPES)].copy()
    risk = risk[~_structural_service_name_mask(risk["agency_name_std"])].copy()
    resolved_oris = _load_resolved_override_oris(repo_root)
    if resolved_oris:
        risk = risk[~risk["ori9"].astype("string").isin(resolved_oris)].copy()
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
    queue = risk.merge(support, on="ori9", how="left")
    queue = queue.merge(crosswalk, on="ori9", how="left")
    keep_mask = (
        queue["relationship_type"].eq("unresolved")
        | queue["resolution_source"].eq("provisional_auto")
    )
    queue = queue[keep_mask].copy()
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
        "city_name_std_nibrs",
        "name",
        "namelsad",
        "provisional_jurisdiction_id",
        "final_jurisdiction_id",
        "candidate_summary",
        "match_status",
        "match_method",
        "manual_review_flag",
        "county_like_name_signal",
        "pseudo_place_fips_flag",
        "relationship_type",
        "review_status",
        "resolution_source",
        "total_support_2024",
        "offense_count_2024",
        "dominant_preferred_source_2024",
    ]
    for col in cols:
        if col not in queue.columns:
            queue[col] = pd.NA
    return queue[cols].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build queue for county-like agencies auto-routed into municipal places via pseudo place codes.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("state/review/queues/local_resolution/current/county_place_alias_risk_queue_2024"),
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
