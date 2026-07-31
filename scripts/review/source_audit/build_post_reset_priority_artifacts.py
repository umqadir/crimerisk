from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
)


YEAR = 2024


def _read_zero_overlap_place_preferred(paths: RepoPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = (
        paths.state_dir / "review"
        / "local_resolution"
        / f"zero_overlap_place_preferred_queue_{YEAR}.parquet"
    )
    if not path.exists():
        empty_jurisdiction = pd.DataFrame(columns=["repo_entity_id", "reference_risk_zero_overlap_place_preferred"])
        empty_ori = pd.DataFrame(columns=["ori9", "reference_risk_zero_overlap_place_preferred"])
        return empty_jurisdiction, empty_ori
    queue = pd.read_parquet(path)
    current_path = paths.state_dir / "reference" / "local_agency_resolved_full.parquet"
    if current_path.exists():
        current = pd.read_parquet(
            current_path,
            columns=["ori9", "state_fips", "resolved_geo_type", "resolved_geoid", "final_decision"],
        )
        current = current[current["ori9"].isin(queue["ori9"])].copy()
        current = current[current["final_decision"].isin(["municipal_place", "municipal_cousub"])].copy()
        current["repo_entity_id"] = (
            current["state_fips"].astype("string").str.zfill(2)
            + ":municipal:"
            + current["resolved_geo_type"].astype("string")
            + ":"
            + current["resolved_geoid"].astype("string")
        )
        by_jurisdiction = current[["repo_entity_id"]].drop_duplicates()
        by_ori = current[["ori9"]].drop_duplicates()
    else:
        by_jurisdiction = (
            queue[["provisional_jurisdiction_id"]]
            .drop_duplicates()
            .rename(columns={"provisional_jurisdiction_id": "repo_entity_id"})
        )
        split = by_jurisdiction["repo_entity_id"].astype("string").str.split(":", expand=True)
        if split.shape[1] >= 3:
            canonical = split[0].astype("string") + ":municipal:" + split[1].astype("string") + ":" + split[2].astype("string")
            by_jurisdiction = pd.concat(
                [
                    by_jurisdiction,
                    pd.DataFrame({"repo_entity_id": canonical}),
                ],
                ignore_index=True,
            ).drop_duplicates()
        by_ori = queue[["ori9"]].drop_duplicates()
    by_jurisdiction["reference_risk_zero_overlap_place_preferred"] = True
    by_ori["reference_risk_zero_overlap_place_preferred"] = True
    return by_jurisdiction, by_ori


def _read_matched_cius(paths: RepoPaths) -> pd.DataFrame:
    path = paths.review_analysis_dir / "source_audit" / f"cius_local_validation_{YEAR}.parquet"
    df = pd.read_parquet(path)
    df = df[df["match_status"].eq("matched_unique")].copy()
    for col in ["repo_count", "official_count", "official_minus_repo", "abs_official_minus_repo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    lane_map = {
        "table8_city": "municipal_summary_reference_tail",
        "table9_university": "university_summary_reference_tail",
        "table11_state_tribal_other": "state_tribal_other_validation_tail",
    }
    df["priority_lane"] = df["publication_collection"].map(lane_map).fillna("other")
    zero_overlap_by_jurisdiction, _ = _read_zero_overlap_place_preferred(paths)
    df = df.merge(zero_overlap_by_jurisdiction, on="repo_entity_id", how="left")
    df["reference_risk_zero_overlap_place_preferred"] = (
        df["reference_risk_zero_overlap_place_preferred"].astype("boolean").fillna(False).astype(bool)
    )
    return df.sort_values(
        ["abs_official_minus_repo", "publication_collection", "state_abbr", "publication_name_raw", "offense"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _read_observation_sources(paths: RepoPaths) -> pd.DataFrame:
    obs = pd.read_parquet(
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        columns=["ori9", "year", "source", "offense", "count"],
    )
    obs = obs[obs["year"].astype(int).eq(YEAR)].copy()
    wide = (
        obs.pivot_table(index=["ori9", "offense"], columns="source", values="count", aggfunc="sum")
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide


def _read_matched_nibrs(paths: RepoPaths) -> pd.DataFrame:
    path = (
        paths.state_dir / "review"
        / "source_audit"
        / f"published_nibrs_agency_validation_{YEAR}.parquet"
    )
    df = pd.read_parquet(path)
    df = df[df["match_status"].eq("matched_unique")].copy()
    for col in ["repo_count", "official_count", "official_minus_repo", "abs_official_minus_repo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.rename(columns={"repo_entity_id": "ori9"})

    src = _read_observation_sources(paths)
    df = df.merge(src, on=["ori9", "offense"], how="left")
    _, zero_overlap_by_ori = _read_zero_overlap_place_preferred(paths)
    df = df.merge(zero_overlap_by_ori, on="ori9", how="left")
    df["reference_risk_zero_overlap_place_preferred"] = (
        df["reference_risk_zero_overlap_place_preferred"].astype("boolean").fillna(False).astype(bool)
    )
    for col in [CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE, NIBRS_SOURCE]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["abs_err_cius"] = (df["official_count"] - df[CIUS_SOURCE]).abs()
    df["abs_err_local_publication"] = (df["official_count"] - df[LOCAL_PUBLICATION_SOURCE]).abs()
    df["abs_err_state_publication"] = (df["official_count"] - df[STATE_PUBLICATION_SOURCE]).abs()
    df["abs_err_srs"] = (df["official_count"] - df[SUMMARY_SOURCE]).abs()
    df["abs_err_nibrs"] = (df["official_count"] - df[NIBRS_SOURCE]).abs()
    df["nibrs_closer_than_srs"] = (
        df[NIBRS_SOURCE].notna()
        & df["abs_err_nibrs"].lt(df["abs_err_srs"])
    )
    df["improvement_if_switch_to_nibrs"] = (df["abs_err_srs"] - df["abs_err_nibrs"]).fillna(0.0)
    df["non_cius_nibrs_switch_candidate"] = (
        df["repo_preferred_source"].eq("srs_return_a_annual")
        & df["nibrs_closer_than_srs"]
    )
    return df.sort_values(
        [
            "non_cius_nibrs_switch_candidate",
            "improvement_if_switch_to_nibrs",
            "abs_official_minus_repo",
            "state_abbr",
            "publication_name_raw",
            "offense",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _read_state_priority(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "controls" / "state_control_comparison.parquet"
    df = pd.read_parquet(path)
    for col in ["ags_core_adjusted_total", "fbi_cde_estimated_total", "internal_srs_total", "internal_nibrs_total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["gap_vs_fbi_cde"] = df["ags_core_adjusted_total"] - df["fbi_cde_estimated_total"]
    df["abs_gap_vs_fbi_cde"] = df["gap_vs_fbi_cde"].abs()
    df["relative_gap_vs_fbi_cde"] = df["gap_vs_fbi_cde"] / df["fbi_cde_estimated_total"].where(
        df["fbi_cde_estimated_total"].ne(0),
        other=pd.NA,
    )
    return df.sort_values(
        ["abs_gap_vs_fbi_cde", "state_abbr", "offense"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _summary(cius: pd.DataFrame, nibrs: pd.DataFrame, state: pd.DataFrame) -> dict[str, object]:
    cius_city = cius[cius["publication_collection"].eq("table8_city")].copy()
    cius_state_other = cius[cius["publication_collection"].eq("table11_state_tribal_other")].copy()
    nibrs_switch = nibrs[nibrs["non_cius_nibrs_switch_candidate"]].copy()
    nibrs_switch_clean = nibrs_switch[~nibrs_switch["reference_risk_zero_overlap_place_preferred"]].copy()
    nibrs_switch_reference_risk = nibrs_switch[nibrs_switch["reference_risk_zero_overlap_place_preferred"]].copy()
    return {
        "year": YEAR,
        "cius_top_city_entities": cius_city.head(10)[
            ["state_abbr", "publication_name_raw", "offense", "repo_entity_name", "official_count", "repo_count", "official_minus_repo"]
        ].to_dict(orient="records"),
        "cius_top_city_reference_risk_entities": cius_city[
            cius_city["reference_risk_zero_overlap_place_preferred"]
        ].head(10)[
            ["state_abbr", "publication_name_raw", "offense", "repo_entity_name", "official_count", "repo_count", "official_minus_repo"]
        ].to_dict(orient="records"),
        "cius_top_state_other_entities": cius_state_other.head(10)[
            ["state_abbr", "publication_name_raw", "offense", "repo_entity_name", "official_count", "repo_count", "official_minus_repo"]
        ].to_dict(orient="records"),
        "nibrs_top_non_cius_switch_candidates": nibrs_switch_clean.head(10)[
            [
                "state_abbr",
                "publication_name_raw",
                "offense",
                "repo_reporting_regime",
                "repo_count",
                "official_count",
                LOCAL_PUBLICATION_SOURCE,
                STATE_PUBLICATION_SOURCE,
                SUMMARY_SOURCE,
                NIBRS_SOURCE,
                "improvement_if_switch_to_nibrs",
            ]
        ].to_dict(orient="records"),
        "nibrs_top_reference_risk_candidates": nibrs_switch_reference_risk.head(10)[
            [
                "state_abbr",
                "publication_name_raw",
                "offense",
                "repo_reporting_regime",
                "repo_count",
                "official_count",
                LOCAL_PUBLICATION_SOURCE,
                STATE_PUBLICATION_SOURCE,
                SUMMARY_SOURCE,
                NIBRS_SOURCE,
                "improvement_if_switch_to_nibrs",
            ]
        ].to_dict(orient="records"),
        "state_top_gap_rows": state.head(10)[
            ["state_abbr", "offense", "ags_core_adjusted_total", "fbi_cde_estimated_total", "gap_vs_fbi_cde"]
        ].to_dict(orient="records"),
    }


def main() -> int:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    out_dir = paths.review_analysis_dir / "source_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    cius = _read_matched_cius(paths)
    nibrs = _read_matched_nibrs(paths)
    state = _read_state_priority(paths)
    summary = _summary(cius, nibrs, state)

    cius_base = out_dir / f"post_reset_cius_priority_{YEAR}"
    cius_city_base = out_dir / f"post_reset_cius_city_priority_{YEAR}"
    cius_state_other_base = out_dir / f"post_reset_cius_state_tribal_other_priority_{YEAR}"
    nibrs_base = out_dir / f"post_reset_published_nibrs_priority_{YEAR}"
    nibrs_switch_base = out_dir / f"post_reset_nibrs_switch_candidates_{YEAR}"
    state_base = out_dir / f"post_reset_state_priority_{YEAR}"
    summary_path = out_dir / f"post_reset_priority_summary_{YEAR}.json"

    cius_city = cius[cius["publication_collection"].eq("table8_city")].copy()
    cius_state_other = cius[cius["publication_collection"].eq("table11_state_tribal_other")].copy()
    nibrs_switch = nibrs[nibrs["non_cius_nibrs_switch_candidate"]].copy()

    cius.to_parquet(cius_base.with_suffix(".parquet"), index=False)
    cius.to_csv(cius_base.with_suffix(".csv"), index=False)
    cius_city.to_parquet(cius_city_base.with_suffix(".parquet"), index=False)
    cius_city.to_csv(cius_city_base.with_suffix(".csv"), index=False)
    cius_state_other.to_parquet(cius_state_other_base.with_suffix(".parquet"), index=False)
    cius_state_other.to_csv(cius_state_other_base.with_suffix(".csv"), index=False)
    nibrs.to_parquet(nibrs_base.with_suffix(".parquet"), index=False)
    nibrs.to_csv(nibrs_base.with_suffix(".csv"), index=False)
    nibrs_switch.to_parquet(nibrs_switch_base.with_suffix(".parquet"), index=False)
    nibrs_switch.to_csv(nibrs_switch_base.with_suffix(".csv"), index=False)
    state.to_parquet(state_base.with_suffix(".parquet"), index=False)
    state.to_csv(state_base.with_suffix(".csv"), index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        {
            "year": YEAR,
            "cius_rows": int(len(cius)),
            "nibrs_rows": int(len(nibrs)),
            "nibrs_switch_candidates": int(nibrs["non_cius_nibrs_switch_candidate"].sum()),
            "state_rows": int(len(state)),
            "summary_path": str(summary_path),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
