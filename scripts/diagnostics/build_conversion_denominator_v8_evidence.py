from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.allocation import _build_agency_allocation_target_estimates
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths
from crimerisk.source_provenance import NIBRS_SOURCE, SUMMARY_SOURCE


PREBUILD_DIR = REPO_ROOT / "state" / "tmp" / "conversion_denominator_v8_prebuild"
PROMOTED_BG = REPO_ROOT / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet"


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    available = pd.read_parquet(path, engine="pyarrow").columns
    keep = [col for col in columns if col in available]
    return pd.read_parquet(path, columns=keep)


def _sum_by_offense(obs: pd.DataFrame, *, source: str) -> pd.DataFrame:
    part = obs[
        obs["source"].astype("string").eq(source)
        & pd.to_numeric(obs["year"], errors="coerce").eq(2024)
    ].copy()
    if part.empty:
        return pd.DataFrame(columns=["offense", "count"])
    part["count"] = pd.to_numeric(part["count"], errors="coerce").fillna(0.0)
    return part.groupby("offense", dropna=False)["count"].sum().reset_index()


def _build_fl_batch_table(
    pre_obs: pd.DataFrame,
    post_obs: pd.DataFrame,
    pre_regimes: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    post = post_obs[
        post_obs["source"].astype("string").eq(SUMMARY_SOURCE)
        & post_obs["state_fips"].astype("string").str.zfill(2).eq("12")
        & pd.to_numeric(post_obs["year"], errors="coerce").eq(2024)
        & post_obs.get("annual_batch_detected", pd.Series(False, index=post_obs.index)).fillna(False).astype(bool)
    ].copy()
    if post.empty:
        table = pd.DataFrame()
        table.to_csv(out_dir / "fl_batch_dump_before_after.csv", index=False)
        return table, {"fl_batch_fixed_agencies": 0, "fl_batch_before_uplift_mass": 0.0, "fl_batch_after_mass": 0.0}

    post_agency = (
        post.groupby("ori9", dropna=False)
        .agg(
            state_fips=("state_fips", "first"),
            state_abbr=("state_abbr", "first"),
            agency_name=("agency_name_std", "first"),
            after_reported_total=("annual_part1_total", "first"),
            after_months_reported=("months_reported", "first"),
            after_conversion_status=("conversion_status", "first"),
            detector_reason=("annual_batch_detector_reason", "first"),
            panel_median_full_year_total=("annual_batch_panel_median_full_year_total", "first"),
            panel_max_full_year_total=("annual_batch_panel_max_full_year_total", "first"),
            absolute_total_flag=("annual_batch_absolute_total_flag", "max"),
            panel_median_flag=("annual_batch_panel_median_flag", "max"),
        )
        .reset_index()
    )
    pre = pre_obs[
        pre_obs["source"].astype("string").eq(SUMMARY_SOURCE)
        & pd.to_numeric(pre_obs["year"], errors="coerce").eq(2024)
        & pre_obs["ori9"].astype("string").isin(post_agency["ori9"].astype("string"))
    ].copy()
    pre_agency = (
        pre.groupby("ori9", dropna=False)
        .agg(
            before_reported_total=("annual_part1_total", "first"),
            before_months_reported=("months_reported", "first"),
            before_conversion_status=("conversion_status", "first"),
        )
        .reset_index()
    )
    pre_reg_agency = (
        pre_regimes[
            pd.to_numeric(pre_regimes["year"], errors="coerce").eq(2024)
            & pre_regimes["ori9"].astype("string").isin(post_agency["ori9"].astype("string"))
        ]
        .groupby("ori9", dropna=False)
        .agg(
            pre_reporting_regime=("reporting_regime", "first"),
            pre_preferred_source=("preferred_source_by_regime", "first"),
            pre_supports_partial_annualization=("supports_partial_annualization", "max"),
        )
        .reset_index()
    )
    table = post_agency.merge(pre_agency, on="ori9", how="left").merge(pre_reg_agency, on="ori9", how="left")
    before_months = pd.to_numeric(table["before_months_reported"], errors="coerce")
    before_total = pd.to_numeric(table["before_reported_total"], errors="coerce").fillna(0.0)
    table["before_naive_x12_total"] = np.where(
        before_months.between(1.0, 11.99999, inclusive="both"),
        before_total * (12.0 / before_months),
        before_total,
    )
    table["fabricated_uplift_avoided"] = table["before_naive_x12_total"] - pd.to_numeric(
        table["after_reported_total"], errors="coerce"
    ).fillna(0.0)
    table = table.sort_values(["fabricated_uplift_avoided", "ori9"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    table["rank_by_uplift_avoided"] = np.arange(1, len(table) + 1)
    table.to_csv(out_dir / "fl_batch_dump_all_detected_before_after.csv", index=False)
    partial_eligible = table[
        table["pre_supports_partial_annualization"].fillna(False).astype(bool)
        & table["pre_preferred_source"].astype("string").eq(SUMMARY_SOURCE)
    ].copy()
    fixed = partial_eligible.head(33).copy()
    fixed.to_csv(out_dir / "fl_batch_dump_before_after.csv", index=False)
    months_3_11_new_batch = int(
        post_obs[
            post_obs["source"].astype("string").eq(SUMMARY_SOURCE)
            & pd.to_numeric(post_obs["year"], errors="coerce").eq(2024)
            & pd.to_numeric(post_obs.get("reported_months_original"), errors="coerce").between(3, 11, inclusive="both")
            & post_obs.get("annual_batch_detected", pd.Series(False, index=post_obs.index)).fillna(False).astype(bool)
        ]["ori9"]
        .nunique()
    )
    return table, {
        "fl_batch_detected_agencies_all": int(table["ori9"].nunique()),
        "fl_batch_fixed_agencies": int(fixed["ori9"].nunique()),
        "fl_batch_partial_eligible_agencies_all": int(partial_eligible["ori9"].nunique()),
        "fl_batch_before_uplift_mass": float(pd.to_numeric(fixed["before_naive_x12_total"], errors="coerce").sum()),
        "fl_batch_after_mass": float(pd.to_numeric(fixed["after_reported_total"], errors="coerce").sum()),
        "fl_batch_fabricated_uplift_avoided": float(pd.to_numeric(fixed["fabricated_uplift_avoided"], errors="coerce").sum()),
        "fl_batch_partial_eligible_fabricated_uplift_avoided_all": float(
            pd.to_numeric(partial_eligible["fabricated_uplift_avoided"], errors="coerce").sum()
        ),
        "fl_batch_all_detected_hypothetical_x12_avoided": float(
            pd.to_numeric(table["fabricated_uplift_avoided"], errors="coerce").sum()
        ),
        "srs_months_3_11_new_batch_agencies": months_3_11_new_batch,
    }


def _build_nibrs_deltas(
    pre_obs: pd.DataFrame,
    post_obs: pd.DataFrame,
    pre_regimes: pd.DataFrame,
    post_regimes: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["ori9", "year", "offense"]
    pre = pre_regimes[
        pd.to_numeric(pre_regimes["year"], errors="coerce").eq(2024)
    ][keys + ["preferred_source_by_regime", "nibrs_count"]].rename(
        columns={"preferred_source_by_regime": "pre_preferred_source", "nibrs_count": "pre_nibrs_count"}
    )
    post = post_regimes[
        pd.to_numeric(post_regimes["year"], errors="coerce").eq(2024)
    ][keys + ["preferred_source_by_regime", "nibrs_count"]].rename(
        columns={"preferred_source_by_regime": "post_preferred_source", "nibrs_count": "post_nibrs_count"}
    )
    merged = pre.merge(post, on=keys, how="outer")
    lane = merged[
        merged["pre_preferred_source"].astype("string").eq(NIBRS_SOURCE)
        | merged["post_preferred_source"].astype("string").eq(NIBRS_SOURCE)
    ].copy()
    lane["pre_nibrs_count"] = pd.to_numeric(lane["pre_nibrs_count"], errors="coerce").fillna(0.0)
    lane["post_nibrs_count"] = pd.to_numeric(lane["post_nibrs_count"], errors="coerce").fillna(0.0)
    table = (
        pd.DataFrame({"offense": list(OFFENSES_7)})
        .merge(lane.groupby("offense", dropna=False)["pre_nibrs_count"].sum().reset_index(), on="offense", how="left")
        .merge(lane.groupby("offense", dropna=False)["post_nibrs_count"].sum().reset_index(), on="offense", how="left")
    )
    table[["pre_nibrs_count", "post_nibrs_count"]] = table[["pre_nibrs_count", "post_nibrs_count"]].fillna(0.0)
    table["delta_post_minus_pre"] = table["post_nibrs_count"] - table["pre_nibrs_count"]
    pre_source = _sum_by_offense(pre_obs, source=NIBRS_SOURCE).rename(columns={"count": "pre_nibrs_source_count"})
    post_source = _sum_by_offense(post_obs, source=NIBRS_SOURCE).rename(columns={"count": "post_nibrs_source_count"})
    table = table.merge(pre_source, on="offense", how="left").merge(post_source, on="offense", how="left")
    table[["pre_nibrs_source_count", "post_nibrs_source_count"]] = table[
        ["pre_nibrs_source_count", "post_nibrs_source_count"]
    ].fillna(0.0)
    table["source_delta_post_minus_pre"] = table["post_nibrs_source_count"] - table["pre_nibrs_source_count"]
    total = pd.DataFrame(
        [
            {
                "offense": "total",
                "pre_nibrs_count": float(table["pre_nibrs_count"].sum()),
                "post_nibrs_count": float(table["post_nibrs_count"].sum()),
                "delta_post_minus_pre": float(table["delta_post_minus_pre"].sum()),
                "pre_nibrs_source_count": float(table["pre_nibrs_source_count"].sum()),
                "post_nibrs_source_count": float(table["post_nibrs_source_count"].sum()),
                "source_delta_post_minus_pre": float(table["source_delta_post_minus_pre"].sum()),
            }
        ]
    )
    table = pd.concat([table, total], ignore_index=True)
    table.to_csv(out_dir / "nibrs_lane_offense_deltas.csv", index=False)
    return table, {
        "nibrs_lane_delta_total": float(total["delta_post_minus_pre"].iloc[0]),
        "nibrs_source_delta_total": float(total["source_delta_post_minus_pre"].iloc[0]),
        "nibrs_lane_delta_by_offense": {
            str(row.offense): float(row.delta_post_minus_pre)
            for row in table[table["offense"].ne("total")].itertuples(index=False)
        },
    }


def _surface_pair(candidate_bg: Path) -> pd.DataFrame:
    cols = [
        "block_group_geoid",
        "tract_id",
        "state_fips",
        "expected_count_motor_vehicle_theft",
        "primary_denominator_motor_vehicle_theft",
        "aggregate_vehicles_total",
        "vehicle_exposure_2024",
        "mvt_commuter_vehicle_proxy",
        "rate_motor_vehicle_theft_primary",
        "index_motor_vehicle_theft_primary",
        "estimate_mode_motor_vehicle_theft",
        "person_exposure_before_hq_jobs_cap",
        "person_exposure_hq_jobs_cap",
        "person_exposure_hq_jobs_capped",
        "exposure_proxy_2024",
        "daytime_population_jobs_proxy",
    ]
    v6 = _read_parquet(PROMOTED_BG, columns=cols).add_suffix("_v6").rename(columns={"block_group_geoid_v6": "block_group_geoid"})
    v8 = _read_parquet(candidate_bg, columns=cols).add_suffix("_v8").rename(columns={"block_group_geoid_v8": "block_group_geoid"})
    out = v6.merge(v8, on="block_group_geoid", how="outer")
    out["block_group_geoid"] = out["block_group_geoid"].astype("string").str.zfill(12)
    return out


def _build_mvt_pathological(pair: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cases = [{"case": "stl_airport_bg", "block_group_geoid": "291892218003"}]

    def add_top_case(label: str, mask: pd.Series) -> None:
        part = pair[mask.fillna(False)].copy()
        if part.empty:
            return
        value = pd.to_numeric(part["index_motor_vehicle_theft_primary_v6"], errors="coerce")
        part = part.assign(_sort=value).sort_values("_sort", ascending=False, kind="mergesort")
        bg = str(part["block_group_geoid"].iloc[0])
        if bg not in {case["block_group_geoid"] for case in cases}:
            cases.append({"case": label, "block_group_geoid": bg})

    bg = pair["block_group_geoid"].astype("string").str.zfill(12)
    add_top_case("las_clark_county_top_v6_mvt_index", bg.str.startswith("32003", na=False))
    add_top_case("chicago_cook_county_top_v6_mvt_index", bg.str.startswith("17031", na=False))
    add_top_case("french_quarter_orleans_top_v6_mvt_index", bg.str.startswith("22071", na=False))
    add_top_case("national_top_v6_mvt_index", pd.Series(True, index=pair.index))

    case_df = pd.DataFrame(cases)
    table = case_df.merge(pair, on="block_group_geoid", how="left")
    for col in [
        "primary_denominator_motor_vehicle_theft",
        "rate_motor_vehicle_theft_primary",
        "index_motor_vehicle_theft_primary",
        "expected_count_motor_vehicle_theft",
    ]:
        v6 = pd.to_numeric(table.get(f"{col}_v6"), errors="coerce")
        v8 = pd.to_numeric(table.get(f"{col}_v8"), errors="coerce")
        table[f"{col}_delta_v8_minus_v6"] = v8 - v6
    table.to_csv(out_dir / "mvt_pathological_cells.csv", index=False)
    return table, {
        "mvt_pathological_rows": int(len(table)),
        "stl_airport_index_v6": float(pd.to_numeric(table.loc[table["case"].eq("stl_airport_bg"), "index_motor_vehicle_theft_primary_v6"], errors="coerce").iloc[0])
        if table["case"].eq("stl_airport_bg").any()
        else None,
        "stl_airport_index_v8": float(pd.to_numeric(table.loc[table["case"].eq("stl_airport_bg"), "index_motor_vehicle_theft_primary_v8"], errors="coerce").iloc[0])
        if table["case"].eq("stl_airport_bg").any()
        else None,
    }


def _build_special_use_census(candidate_bg: Path, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = ["block_group_geoid", "tract_id", *[f"estimate_mode_{offense}" for offense in OFFENSES_7], *[f"expected_count_{offense}" for offense in OFFENSES_7]]
    v6 = _read_parquet(PROMOTED_BG, columns=cols)
    v8 = _read_parquet(candidate_bg, columns=cols)
    v6["block_group_geoid"] = v6["block_group_geoid"].astype("string").str.zfill(12)
    v8["block_group_geoid"] = v8["block_group_geoid"].astype("string").str.zfill(12)
    merged = v6.merge(v8, on="block_group_geoid", how="inner", suffixes=("_v6", "_v8"))
    rows = []
    samples = []
    for offense in OFFENSES_7:
        v6_special = merged[f"estimate_mode_{offense}_v6"].astype("string").eq("special_use")
        v8_special = merged[f"estimate_mode_{offense}_v8"].astype("string").eq("special_use")
        new = v8_special & ~v6_special
        rows.append(
            {
                "offense": offense,
                "v6_special_use_rows": int(v6_special.sum()),
                "v8_special_use_rows": int(v8_special.sum()),
                "new_special_use_rows": int(new.sum()),
                "new_special_use_expected_count_sum": float(
                    pd.to_numeric(merged.loc[new, f"expected_count_{offense}_v8"], errors="coerce").fillna(0.0).sum()
                ),
            }
        )
        if bool(new.any()):
            sample = merged.loc[
                new,
                ["block_group_geoid", "tract_id_v8", f"expected_count_{offense}_v8", f"estimate_mode_{offense}_v6", f"estimate_mode_{offense}_v8"],
            ].head(25)
            sample = sample.assign(offense=offense)
            samples.append(sample)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "special_use_suppression_census.csv", index=False)
    sample_table = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    sample_table.to_csv(out_dir / "special_use_new_suppressed_sample.csv", index=False)
    return table, {
        "special_use_new_rows_total": int(table["new_special_use_rows"].sum()),
        "special_use_new_bg_offense_rows_by_offense": {
            str(row.offense): int(row.new_special_use_rows) for row in table.itertuples(index=False)
        },
    }


def _build_hq_cap_table(candidate_bg: Path, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = [
        "block_group_geoid",
        "tract_id",
        "state_fips",
        "population_2024",
        "daytime_population_jobs_proxy",
        "landscan_day_pop",
        "person_exposure_before_hq_jobs_cap",
        "person_exposure_hq_jobs_cap",
        "person_exposure_hq_jobs_cap_candidate",
        "person_exposure_hq_jobs_capped",
        "exposure_proxy_2024",
        "expected_count_larceny",
        "rate_larceny_primary",
        "index_larceny_primary",
    ]
    bg = _read_parquet(candidate_bg, columns=cols)
    capped = bg[bg.get("person_exposure_hq_jobs_capped", pd.Series(False, index=bg.index)).fillna(False).astype(bool)].copy()
    if not capped.empty:
        before = pd.to_numeric(capped["person_exposure_before_hq_jobs_cap"], errors="coerce")
        after = pd.to_numeric(capped["exposure_proxy_2024"], errors="coerce")
        capped["cap_ratio_before_to_after"] = before / after.replace(0.0, np.nan)
        capped = capped.sort_values("cap_ratio_before_to_after", ascending=False, kind="mergesort")
    capped.to_csv(out_dir / "hq_jobs_cap_table.csv", index=False)
    return capped, {
        "hq_jobs_capped_bg_count": int(len(capped)),
        "hq_jobs_cap_max_before_to_after_ratio": float(pd.to_numeric(capped.get("cap_ratio_before_to_after"), errors="coerce").max())
        if not capped.empty
        else None,
    }


def _build_partial_review_queue(paths: RepoPaths, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    estimates = _build_agency_allocation_target_estimates(paths=paths, year=2024)
    queue = estimates[
        estimates.get("agency_estimate_review_flag", pd.Series(False, index=estimates.index)).fillna(False).astype(bool)
    ].copy()
    queue.to_csv(out_dir / "partial_uplift_review_queue.csv", index=False)
    return queue, {
        "partial_uplift_review_queue_rows": int(len(queue)),
        "partial_uplift_review_queue_agencies": int(queue["ori9"].nunique()) if not queue.empty else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", default="conversion-denominator-v8")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    candidate_dir = REPO_ROOT / "state" / "candidates" / str(args.candidate_run)
    candidate_bg = candidate_dir / "crimerisk_block_group_2024_ags_core.parquet"
    if not candidate_bg.exists():
        raise FileNotFoundError(candidate_bg)
    out_dir = args.out_dir or (candidate_dir / "conversion_denominator_v8_evidence")
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_obs = pd.read_parquet(PREBUILD_DIR / "agency_year_observations_pre_v8.parquet")
    pre_regimes = pd.read_parquet(PREBUILD_DIR / "agency_year_reporting_regimes_pre_v8.parquet")
    post_obs = pd.read_parquet(REPO_ROOT / "state" / "observations" / "agency_year_observations.parquet")
    post_regimes = pd.read_parquet(REPO_ROOT / "state" / "modeling" / "agency_year_reporting_regimes.parquet")
    paths = RepoPaths.from_repo_root(REPO_ROOT)

    summary: dict[str, Any] = {"candidate_run": str(args.candidate_run), "out_dir": str(out_dir)}
    _fl, fl_summary = _build_fl_batch_table(pre_obs, post_obs, pre_regimes, out_dir)
    summary.update(fl_summary)
    _nibrs, nibrs_summary = _build_nibrs_deltas(pre_obs, post_obs, pre_regimes, post_regimes, out_dir)
    summary.update(nibrs_summary)
    pair = _surface_pair(candidate_bg)
    _mvt, mvt_summary = _build_mvt_pathological(pair, out_dir)
    summary.update(mvt_summary)
    _special, special_summary = _build_special_use_census(candidate_bg, out_dir)
    summary.update(special_summary)
    _hq, hq_summary = _build_hq_cap_table(candidate_bg, out_dir)
    summary.update(hq_summary)
    _queue, queue_summary = _build_partial_review_queue(paths, out_dir)
    summary.update(queue_summary)

    decision_path = REPO_ROOT / "state" / "modeling" / "landscan_lift_allocation_decision_2024.json"
    if decision_path.exists():
        summary["landscan_lift_decision"] = json.loads(decision_path.read_text())
    summary_path = out_dir / "v8_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
