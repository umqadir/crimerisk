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


STOP_TOKENS = {
    "POLICE",
    "DEPARTMENT",
    "DEPT",
    "PUBLIC",
    "SAFETY",
    "SHERIFF",
    "OFFICE",
    "TOWNSHIP",
    "CITY",
    "COUNTY",
    "VILLAGE",
    "TOWN",
    "METRO",
    "METROPOLITAN",
    "UNIVERSITY",
    "CAMPUS",
    "STATE",
    "DISTRICT",
}


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


def _tokens(value: object) -> set[str]:
    text = "" if pd.isna(value) else str(value).upper()
    return {token for token in re.findall(r"[A-Z0-9]+", text) if token and token not in STOP_TOKENS}


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

    place_preferred = review[
        review["match_method"].isin(["place_preferred", "zero_overlap_place_preferred", "county_style_invalid_place_code"])
    ].copy()
    place_preferred = place_preferred[
        ~place_preferred["agency_type_norm"].astype("string").isin(STRUCTURAL_SERVICE_EXCLUDE_TYPES)
    ].copy()
    place_preferred = place_preferred[
        ~_structural_service_name_mask(place_preferred["agency_name_std"])
    ].copy()
    resolved_oris = _load_resolved_override_oris(repo_root)
    if resolved_oris:
        place_preferred = place_preferred[~place_preferred["ori9"].astype("string").isin(resolved_oris)].copy()
    place_preferred["agency_tokens"] = place_preferred["agency_name_std"].map(_tokens)
    place_preferred["place_tokens"] = place_preferred["name"].map(_tokens)
    if "place_preferred_token_overlap_count" in place_preferred.columns:
        place_preferred["token_overlap_count"] = pd.to_numeric(
            place_preferred["place_preferred_token_overlap_count"],
            errors="coerce",
        ).fillna(0).astype(int)
    else:
        place_preferred["token_overlap_count"] = place_preferred.apply(
            lambda row: len(row["agency_tokens"] & row["place_tokens"]),
            axis=1,
        )
    zero_overlap = place_preferred[place_preferred["token_overlap_count"].eq(0)].copy()

    reg_year = regimes[regimes["year"].astype(int).eq(int(year))].copy()
    support = (
        reg_year.groupby("ori9", dropna=False)
        .agg(
            total_support_2024=("support_weight", "sum"),
            offense_count_2024=("offense", "nunique"),
            dominant_preferred_source_2024=("preferred_source_by_regime", lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else pd.NA),
        )
        .reset_index()
    )

    queue = zero_overlap.merge(support, on="ori9", how="left")
    queue = queue.merge(crosswalk, on="ori9", how="left")
    keep_mask = (
        queue["relationship_type"].eq("unresolved")
        | queue["resolution_source"].eq("provisional_auto")
    )
    queue = queue[keep_mask].copy()
    queue["total_support_2024"] = pd.to_numeric(queue["total_support_2024"], errors="coerce").fillna(0.0)
    queue["agency_tokens_str"] = queue["agency_tokens"].map(lambda s: ",".join(sorted(s)))
    queue["place_tokens_str"] = queue["place_tokens"].map(lambda s: ",".join(sorted(s)))
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
        "name",
        "namelsad",
        "provisional_jurisdiction_id",
        "final_jurisdiction_id",
        "candidate_summary",
        "county_like_name_signal",
        "pseudo_place_fips_flag",
        "agency_tokens_str",
        "place_tokens_str",
        "token_overlap_count",
        "relationship_type",
        "review_status",
        "resolution_source",
        "total_support_2024",
        "offense_count_2024",
        "dominant_preferred_source_2024",
        "manual_review_flag",
    ]
    for col in cols:
        if col not in queue.columns:
            queue[col] = pd.NA
    return queue[cols].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build queue for zero-overlap place_preferred local matches.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("state/review/queues/local_resolution/current/zero_overlap_place_preferred_queue_2024"),
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
