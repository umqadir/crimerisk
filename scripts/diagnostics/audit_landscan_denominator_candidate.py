#!/usr/bin/env python3
"""Audit the Q2 LandScan day denominator floor-lift candidate."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_OUTPUT_DIR = REPO_ROOT / "state" / "output"
LANDSCAN_BG = REPO_ROOT / "data" / "LandScan-USA" / "block_group_landscan_usa_2021.parquet"
PATHOLOGICAL_CELLS = REPO_ROOT / "state" / "modeling" / "landscan_pathological_cells.csv"
TRUTH_SURFACE = REPO_ROOT / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet"
TIGER_BG_DIR = REPO_ROOT / "data" / "tiger_bg"

RATE_PER_100K = 100_000.0
PERSON_FLOOR = 50.0
COUNT_DERIVED_TOLERANCE = 1e-9
PATHOLOGICAL_INDEX_TOLERANCE = 0.05
PERSON_EXPOSURE_OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "larceny",
]
OFFENSES_7 = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]
STATIC_PRIMARY_OFFENSES = ["burglary", "motor_vehicle_theft"]
PRIMARY_SIX_SPOT_GROUPS = {
    "times_square",
    "mount_rainier",
    "south_bay_boston",
    "lv_strip",
    "easton_columbus",
}
RELEASE_EXCLUDED_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NA or pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    number = _to_float(value)
    if number is None:
        return "NA"
    if abs(number) >= 1000:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "NA"
    return f"{number:,.0f}"


def _round_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(6)
    return out


def _available_columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[column].fillna(default).astype(bool)


def _read_bg_surface(path: Path) -> pd.DataFrame:
    base_cols = [
        "block_group_geoid",
        "state_fips",
        "population_2024",
        "tract_id",
        "daytime_population_jobs_proxy",
        "landscan_day_pop",
        "exposure_proxy_2024",
        "landscan_day_lifted_person_exposure",
        "households_total",
        "commercial_premises_total",
        "aggregate_vehicles_total",
        "non_residential_flag",
        "special_use_tract_flag",
        "expected_count_total",
        "index_total_primary_event_weighted",
    ]
    offense_cols: list[str] = []
    for offense in OFFENSES_7:
        offense_cols.extend(
            [
                f"expected_count_{offense}",
                f"primary_denominator_{offense}",
                f"primary_denominator_raw_{offense}",
                f"primary_national_rate_per_100k_{offense}",
                f"rate_{offense}_primary",
                f"index_{offense}_primary",
                f"estimate_mode_{offense}",
                f"transient_exposure_likely_{offense}",
            ]
        )
    available = _available_columns(path)
    columns = [col for col in dict.fromkeys([*base_cols, *offense_cols]) if col in available]
    out = pd.read_parquet(path, columns=columns)
    out["block_group_geoid"] = out["block_group_geoid"].astype(str).str.zfill(12)
    if "state_fips" in out.columns:
        out["state_fips"] = out["state_fips"].astype(str).str.zfill(2)
    return out


def _read_landscan() -> pd.DataFrame:
    land = pd.read_parquet(LANDSCAN_BG, columns=["block_group_geoid", "landscan_day_pop", "landscan_night_pop"])
    land["block_group_geoid"] = land["block_group_geoid"].astype(str).str.zfill(12)
    for col in ["landscan_day_pop", "landscan_night_pop"]:
        land[col] = pd.to_numeric(land[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return land


def _merge_landscan_for_baseline(baseline: pd.DataFrame, land: pd.DataFrame) -> pd.DataFrame:
    out = baseline.drop(columns=[col for col in ["landscan_day_pop"] if col in baseline.columns]).merge(
        land[["block_group_geoid", "landscan_day_pop", "landscan_night_pop"]],
        on="block_group_geoid",
        how="left",
    )
    out[["landscan_day_pop", "landscan_night_pop"]] = out[["landscan_day_pop", "landscan_night_pop"]].fillna(0.0)
    out["q2_exposure_proxy_2024"] = np.maximum(
        _numeric(out, "exposure_proxy_2024"),
        _numeric(out, "landscan_day_pop").where(_numeric(out, "landscan_day_pop").gt(0.0), 0.0),
    )
    out["q2_landscan_day_lifted_person_exposure"] = _numeric(out, "landscan_day_pop").gt(
        _numeric(out, "exposure_proxy_2024")
    )
    return out


def _event_weights(df: pd.DataFrame) -> dict[str, float]:
    totals = {
        offense: float(_numeric(df, f"expected_count_{offense}").clip(lower=0.0).sum())
        for offense in OFFENSES_7
    }
    total = float(sum(totals.values()))
    if total <= 0.0:
        return {offense: float("nan") for offense in OFFENSES_7}
    return {offense: totals[offense] / total for offense in OFFENSES_7}


def _candidate_publishable(df: pd.DataFrame, offense: str, denominator: pd.Series) -> pd.Series:
    if offense in PERSON_EXPOSURE_OFFENSES:
        mode = df[f"estimate_mode_{offense}"].astype("string")
        base_ok = mode.isin(["count_derived", "insufficient_exposure"])
        return base_ok & denominator.gt(0.0) & denominator.ge(PERSON_FLOOR)
    current_index = pd.to_numeric(df[f"index_{offense}_primary"], errors="coerce")
    return current_index.notna()


def _compute_fixed_q2_indexes(df: pd.DataFrame) -> dict[str, Any]:
    denominator = _numeric(df, "q2_exposure_proxy_2024")
    weights = _event_weights(df)
    component_values: dict[str, pd.Series] = {}
    component_rates: dict[str, pd.Series] = {}
    offense_summary: dict[str, Any] = {}

    for offense in OFFENSES_7:
        if offense in PERSON_EXPOSURE_OFFENSES:
            publishable = _candidate_publishable(df, offense, denominator)
            counts = _numeric(df, f"expected_count_{offense}").clip(lower=0.0)
            denom_sum = float(denominator.loc[publishable].sum())
            count_sum = float(counts.loc[publishable].sum())
            national_rate = RATE_PER_100K * count_sum / denom_sum if denom_sum > 0.0 else float("nan")
            rate = pd.Series(np.nan, index=df.index, dtype=float)
            index = pd.Series(np.nan, index=df.index, dtype=float)
            rate.loc[publishable] = RATE_PER_100K * counts.loc[publishable] / denominator.loc[publishable]
            if np.isfinite(national_rate) and national_rate > 0.0:
                index.loc[publishable] = 100.0 * rate.loc[publishable] / national_rate
            component_values[offense] = index.replace([np.inf, -np.inf], np.nan)
            component_rates[offense] = rate.replace([np.inf, -np.inf], np.nan)
            offense_summary[offense] = {
                "candidate_national_rate_per_100k": _to_float(national_rate),
                "published_rows": int(publishable.sum()),
                "denominator_total": _to_float(denom_sum),
                "expected_count_total": _to_float(count_sum),
            }
        else:
            index = pd.to_numeric(df[f"index_{offense}_primary"], errors="coerce")
            rate = pd.to_numeric(df[f"rate_{offense}_primary"], errors="coerce")
            component_values[offense] = index
            component_rates[offense] = rate
            published = index.notna()
            offense_summary[offense] = {
                "candidate_national_rate_per_100k": _to_float(
                    df[f"primary_national_rate_per_100k_{offense}"].dropna().iloc[0]
                )
                if f"primary_national_rate_per_100k_{offense}" in df.columns
                and df[f"primary_national_rate_per_100k_{offense}"].notna().any()
                else None,
                "published_rows": int(published.sum()),
                "denominator_total": _to_float(_numeric(df, f"primary_denominator_{offense}").loc[published].sum()),
                "expected_count_total": _to_float(_numeric(df, f"expected_count_{offense}").loc[published].sum()),
            }

    matrix = np.vstack([component_values[offense].to_numpy(dtype=float) for offense in OFFENSES_7])
    weight_values = np.array([float(weights[offense]) for offense in OFFENSES_7], dtype=float)
    valid_rows = np.isfinite(matrix).all(axis=0)
    valid_weights = np.isfinite(weight_values) & (weight_values > 0.0)
    composite = np.full(len(df), np.nan, dtype=float)
    weight_sum = float(weight_values[valid_weights].sum())
    if weight_sum > 0.0:
        composite[valid_rows] = np.dot(weight_values, matrix[:, valid_rows]) / weight_sum

    return {
        "index_total_primary_event_weighted_q2_fixed_count": pd.Series(composite, index=df.index, dtype=float).replace(
            [np.inf, -np.inf], np.nan
        ),
        "component_indexes": component_values,
        "component_rates": component_rates,
        "event_weights": weights,
        "offense_summary": offense_summary,
    }


def _max_abs_pair_delta(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    lvals = pd.to_numeric(left, errors="coerce").replace([np.inf, -np.inf], np.nan)
    rvals = pd.to_numeric(right, errors="coerce").replace([np.inf, -np.inf], np.nan)
    both_null = lvals.isna() & rvals.isna()
    null_mismatch = int((lvals.isna() ^ rvals.isna()).sum())
    comparable = ~(both_null | lvals.isna() | rvals.isna())
    max_abs = float((lvals.loc[comparable] - rvals.loc[comparable]).abs().max()) if comparable.any() else 0.0
    return max_abs, null_mismatch


def _published_rate_gradient(
    truth: pd.DataFrame,
    surface: pd.DataFrame,
    *,
    label: str,
    use_truth_incidents: bool,
) -> list[dict[str, Any]]:
    columns = [
        "block_group_geoid",
        "households_total",
        "commercial_premises_total",
    ]
    for offense in PERSON_EXPOSURE_OFFENSES:
        columns.extend(
            [
                f"expected_count_{offense}",
                f"primary_denominator_{offense}",
                f"rate_{offense}_primary",
                f"index_{offense}_primary",
            ]
        )
    surface_cols = [col for col in dict.fromkeys(columns) if col in surface.columns]
    merged = truth.merge(surface[surface_cols], on="block_group_geoid", how="left")
    households = _numeric(merged, "households_total")
    premises = _numeric(merged, "commercial_premises_total")
    denom = households + premises
    merged["commercial_share"] = (premises / denom.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    merged["incident_count"] = pd.to_numeric(merged["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)

    rows: list[dict[str, Any]] = []
    for offense in PERSON_EXPOSURE_OFFENSES:
        base = merged[merged["offense"].eq(offense)].copy()
        base = base[base["commercial_share"].notna()]
        if len(base) < 10:
            continue
        ranks = base["commercial_share"].rank(method="first")
        try:
            base["commercial_quintile"] = pd.qcut(ranks, 5, labels=False) + 1
        except ValueError:
            continue
        count_col = "incident_count" if use_truth_incidents else f"expected_count_{offense}"
        denom_col = f"primary_denominator_{offense}"
        base["denominator"] = pd.to_numeric(base[denom_col], errors="coerce")
        base["count_for_rate"] = pd.to_numeric(base[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if not use_truth_incidents:
            published = pd.to_numeric(base[f"rate_{offense}_primary"], errors="coerce").notna()
            base = base[published]
        base = base[base["denominator"].gt(0.0)]
        if base.empty:
            continue
        q1 = base[base["commercial_quintile"].eq(1)]
        q5 = base[base["commercial_quintile"].eq(5)]
        q1_denom = float(q1["denominator"].sum())
        q5_denom = float(q5["denominator"].sum())
        q1_count = float(q1["count_for_rate"].sum())
        q5_count = float(q5["count_for_rate"].sum())
        q1_rate = RATE_PER_100K * q1_count / q1_denom if q1_denom > 0.0 else float("nan")
        q5_rate = RATE_PER_100K * q5_count / q5_denom if q5_denom > 0.0 else float("nan")
        rows.append(
            {
                "family": label,
                "offense": offense,
                "source": "direct_truth_incidents" if use_truth_incidents else "published_expected_counts",
                "rows": int(len(base)),
                "cities": int(base["city_name"].nunique()),
                "q1_rows": int(len(q1)),
                "q5_rows": int(len(q5)),
                "q1_commercial_share_mean": _to_float(q1["commercial_share"].mean()),
                "q5_commercial_share_mean": _to_float(q5["commercial_share"].mean()),
                "q1_count": _to_float(q1_count),
                "q5_count": _to_float(q5_count),
                "q1_denominator_sum": _to_float(q1_denom),
                "q5_denominator_sum": _to_float(q5_denom),
                "q1_rate_per_100k": _to_float(q1_rate),
                "q5_rate_per_100k": _to_float(q5_rate),
                "q5_over_q1_gradient": _to_float(q5_rate / q1_rate if q1_rate > 0.0 else float("nan")),
            }
        )
    return rows


def _truth_gradient(baseline: pd.DataFrame, candidate: pd.DataFrame, out_csv: Path) -> dict[str, Any]:
    if not TRUTH_SURFACE.exists():
        return {"available": False, "reason": f"Missing {TRUTH_SURFACE.relative_to(REPO_ROOT)}"}
    truth = pd.read_parquet(TRUTH_SURFACE)
    truth["block_group_geoid"] = truth["block_group_geoid"].astype(str).str.zfill(12)
    truth = truth[truth["offense"].isin(PERSON_EXPOSURE_OFFENSES)].copy()
    rows: list[dict[str, Any]] = []
    rows.extend(
        _published_rate_gradient(
            truth,
            baseline,
            label="current_promoted_surface",
            use_truth_incidents=True,
        )
    )
    rows.extend(
        _published_rate_gradient(
            truth,
            candidate,
            label="q2_landscan_candidate",
            use_truth_incidents=True,
        )
    )
    rows.extend(
        _published_rate_gradient(
            truth,
            baseline,
            label="current_promoted_surface",
            use_truth_incidents=False,
        )
    )
    rows.extend(
        _published_rate_gradient(
            truth,
            candidate,
            label="q2_landscan_candidate",
            use_truth_incidents=False,
        )
    )
    gradient = pd.DataFrame(rows)
    _round_for_csv(gradient).to_csv(out_csv, index=False)

    comparisons: list[dict[str, Any]] = []
    for source, group in gradient.groupby("source", sort=True):
        for offense in PERSON_EXPOSURE_OFFENSES:
            current = group[
                group["family"].eq("current_promoted_surface") & group["offense"].eq(offense)
            ]["q5_over_q1_gradient"]
            candidate_vals = group[
                group["family"].eq("q2_landscan_candidate") & group["offense"].eq(offense)
            ]["q5_over_q1_gradient"]
            if current.empty or candidate_vals.empty:
                continue
            current_value = float(current.iloc[0])
            candidate_value = float(candidate_vals.iloc[0])
            current_distance = abs(math.log(max(current_value, 1e-12))) if np.isfinite(current_value) else float("nan")
            candidate_distance = (
                abs(math.log(max(candidate_value, 1e-12))) if np.isfinite(candidate_value) else float("nan")
            )
            comparisons.append(
                {
                    "source": source,
                    "offense": offense,
                    "current_gradient": _to_float(current_value),
                    "candidate_gradient": _to_float(candidate_value),
                    "candidate_minus_current": _to_float(candidate_value - current_value),
                    "current_log_distance_from_1": _to_float(current_distance),
                    "candidate_log_distance_from_1": _to_float(candidate_distance),
                    "no_degradation": bool(
                        np.isfinite(current_distance)
                        and np.isfinite(candidate_distance)
                        and candidate_distance <= current_distance + 1e-9
                    ),
                    "near_unchanged": bool(
                        np.isfinite(current_value)
                        and np.isfinite(candidate_value)
                        and abs(candidate_value - current_value) <= 0.02
                    ),
                }
            )
    return {
        "available": True,
        "truth_surface_path": str(TRUTH_SURFACE.relative_to(REPO_ROOT)),
        "truth_rows_person_exposure": int(len(truth)),
        "covered_truth_cities_available": int(truth["city_name"].nunique()),
        "covered_truth_city_names": sorted(truth["city_name"].dropna().unique().tolist()),
        "csv": str(out_csv.relative_to(REPO_ROOT)),
        "rows": gradient.to_dict(orient="records"),
        "comparison": comparisons,
        "direct_truth_no_degradation_count": int(
            sum(row["no_degradation"] for row in comparisons if row["source"] == "direct_truth_incidents")
        ),
        "direct_truth_near_unchanged_count": int(
            sum(row["near_unchanged"] for row in comparisons if row["source"] == "direct_truth_incidents")
        ),
    }


def _suppressed_to_published(baseline: pd.DataFrame, candidate: pd.DataFrame, out_csv: Path) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    merged = baseline.merge(
        candidate,
        on="block_group_geoid",
        how="inner",
        suffixes=("_current", "_candidate"),
    )
    lift_col = (
        "landscan_day_lifted_person_exposure_candidate"
        if "landscan_day_lifted_person_exposure_candidate" in merged.columns
        else "landscan_day_lifted_person_exposure"
    )
    for offense in PERSON_EXPOSURE_OFFENSES:
        old_mode = merged[f"estimate_mode_{offense}_current"].astype("string")
        new_mode = merged[f"estimate_mode_{offense}_candidate"].astype("string")
        lifted = merged[lift_col].fillna(False).astype(bool)
        mask = old_mode.eq("insufficient_exposure") & new_mode.eq("count_derived") & lifted
        if not mask.any():
            continue
        part = pd.DataFrame(
            {
                "offense": offense,
                "block_group_geoid": merged.loc[mask, "block_group_geoid"],
                "state_fips": merged.loc[mask, "state_fips_candidate"],
                "population_2024": merged.loc[mask, "population_2024_candidate"],
                "current_exposure_proxy_2024": merged.loc[mask, "exposure_proxy_2024_current"],
                "daytime_population_jobs_proxy": merged.loc[mask, "daytime_population_jobs_proxy_candidate"],
                "landscan_day_pop": merged.loc[mask, "landscan_day_pop_candidate"],
                "candidate_exposure_proxy_2024": merged.loc[mask, "exposure_proxy_2024_candidate"],
                "expected_count": merged.loc[mask, f"expected_count_{offense}_candidate"],
                "old_estimate_mode": old_mode.loc[mask],
                "new_estimate_mode": new_mode.loc[mask],
                "old_primary_denominator": merged.loc[mask, f"primary_denominator_{offense}_current"],
                "new_primary_denominator": merged.loc[mask, f"primary_denominator_{offense}_candidate"],
                "new_rate_primary": merged.loc[mask, f"rate_{offense}_primary_candidate"],
                "new_index_primary": merged.loc[mask, f"index_{offense}_primary_candidate"],
                "new_transient_exposure_likely": merged.loc[mask, f"transient_exposure_likely_{offense}_candidate"],
                "landscan_day_lifted_person_exposure": lifted.loc[mask],
            }
        )
        rows.append(part)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out = out.sort_values(["offense", "state_fips", "block_group_geoid"], kind="mergesort")
    else:
        out = pd.DataFrame(
            columns=[
                "offense",
                "block_group_geoid",
                "state_fips",
                "population_2024",
                "current_exposure_proxy_2024",
                "daytime_population_jobs_proxy",
                "landscan_day_pop",
                "candidate_exposure_proxy_2024",
                "expected_count",
                "old_estimate_mode",
                "new_estimate_mode",
                "old_primary_denominator",
                "new_primary_denominator",
                "new_rate_primary",
                "new_index_primary",
                "new_transient_exposure_likely",
                "landscan_day_lifted_person_exposure",
            ]
        )
    _round_for_csv(out).to_csv(out_csv, index=False)
    return {
        "csv": str(out_csv.relative_to(REPO_ROOT)),
        "rows": int(len(out)),
        "unique_block_groups": int(out["block_group_geoid"].nunique()) if len(out) else 0,
        "by_offense": out.groupby("offense").size().astype(int).to_dict() if len(out) else {},
    }


def _pathological_before_after(
    pathological: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    q2: dict[str, Any],
    out_csv: Path,
) -> dict[str, Any]:
    base = baseline.set_index("block_group_geoid", drop=False)
    cand = candidate.set_index("block_group_geoid", drop=False)
    q2_total_by_bg = pd.Series(
        q2["index_total_primary_event_weighted_q2_fixed_count"].to_numpy(dtype=float),
        index=baseline["block_group_geoid"].astype(str).str.zfill(12),
    )
    q2_index_by_bg = {
        offense: pd.Series(
            q2["component_indexes"][offense].to_numpy(dtype=float),
            index=baseline["block_group_geoid"].astype(str).str.zfill(12),
        )
        for offense in PERSON_EXPOSURE_OFFENSES
    }
    rows: list[dict[str, Any]] = []
    for _, spot in pathological.iterrows():
        bg_id = str(spot["block_group_geoid"]).zfill(12)
        b = base.loc[bg_id] if bg_id in base.index else pd.Series(dtype=object)
        c = cand.loc[bg_id] if bg_id in cand.index else pd.Series(dtype=object)
        row: dict[str, Any] = {
            "spot_group": spot.get("spot_group"),
            "spot_label": spot.get("spot_label"),
            "block_group_geoid": bg_id,
            "q2_primary_six_cell": bool(spot.get("spot_group") in PRIMARY_SIX_SPOT_GROUPS),
            "selection_note": spot.get("selection_note"),
            "state_fips": c.get("state_fips", b.get("state_fips", spot.get("state_fips"))),
            "population_2024": c.get("population_2024", b.get("population_2024", np.nan)),
            "baseline_exposure_proxy_2024": b.get("exposure_proxy_2024", np.nan),
            "landscan_day_pop": c.get("landscan_day_pop", b.get("landscan_day_pop", np.nan)),
            "candidate_daytime_population_jobs_proxy": c.get("daytime_population_jobs_proxy", np.nan),
            "candidate_exposure_proxy_2024": c.get("exposure_proxy_2024", np.nan),
            "candidate_landscan_day_lifted_person_exposure": c.get(
                "landscan_day_lifted_person_exposure", False
            ),
            "q1_pilot_D_amb": spot.get("D_amb"),
            "q1_pilot_index_total_D_amb": spot.get("index_total_primary_event_weighted_D_amb"),
            "baseline_index_total_primary_event_weighted": b.get("index_total_primary_event_weighted", np.nan),
            "q2_pred_index_total_primary_event_weighted": q2_total_by_bg.get(bg_id, np.nan),
            "candidate_index_total_primary_event_weighted": c.get(
                "index_total_primary_event_weighted", np.nan
            ),
        }
        row["delta_candidate_minus_q2_pred_total"] = (
            row["candidate_index_total_primary_event_weighted"] - row["q2_pred_index_total_primary_event_weighted"]
            if pd.notna(row["candidate_index_total_primary_event_weighted"])
            and pd.notna(row["q2_pred_index_total_primary_event_weighted"])
            else np.nan
        )
        row["delta_q2_pred_minus_q1_D_amb_total"] = (
            row["q2_pred_index_total_primary_event_weighted"] - row["q1_pilot_index_total_D_amb"]
            if pd.notna(row["q2_pred_index_total_primary_event_weighted"])
            and pd.notna(row["q1_pilot_index_total_D_amb"])
            else np.nan
        )
        for offense in PERSON_EXPOSURE_OFFENSES:
            row[f"expected_count_{offense}"] = c.get(
                f"expected_count_{offense}", b.get(f"expected_count_{offense}", np.nan)
            )
            row[f"baseline_primary_denominator_{offense}"] = b.get(f"primary_denominator_{offense}", np.nan)
            row[f"candidate_primary_denominator_{offense}"] = c.get(f"primary_denominator_{offense}", np.nan)
            row[f"baseline_index_{offense}_primary"] = b.get(f"index_{offense}_primary", np.nan)
            row[f"q1_pilot_index_{offense}_D_amb"] = spot.get(f"index_{offense}_primary_D_amb")
            row[f"q2_pred_index_{offense}_primary"] = q2_index_by_bg[offense].get(bg_id, np.nan)
            row[f"candidate_index_{offense}_primary"] = c.get(f"index_{offense}_primary", np.nan)
            row[f"delta_candidate_minus_q2_pred_{offense}"] = (
                row[f"candidate_index_{offense}_primary"] - row[f"q2_pred_index_{offense}_primary"]
                if pd.notna(row[f"candidate_index_{offense}_primary"])
                and pd.notna(row[f"q2_pred_index_{offense}_primary"])
                else np.nan
            )
            row[f"candidate_estimate_mode_{offense}"] = c.get(f"estimate_mode_{offense}", None)
        rows.append(row)

    out = pd.DataFrame(rows)
    _round_for_csv(out).to_csv(out_csv, index=False)
    compare_cols = ["delta_candidate_minus_q2_pred_total"] + [
        f"delta_candidate_minus_q2_pred_{offense}" for offense in PERSON_EXPOSURE_OFFENSES
    ]
    primary = out[out["q2_primary_six_cell"]].copy()
    max_abs_delta = float(primary[compare_cols].abs().max().max()) if not primary.empty else 0.0
    null_mismatches = 0
    for col in compare_cols:
        predicted_col = col.replace("delta_candidate_minus_q2_pred_", "q2_pred_index_")
        if col == "delta_candidate_minus_q2_pred_total":
            predicted_col = "q2_pred_index_total_primary_event_weighted"
            actual_col = "candidate_index_total_primary_event_weighted"
        else:
            offense = col.replace("delta_candidate_minus_q2_pred_", "")
            actual_col = f"candidate_index_{offense}_primary"
            predicted_col = f"q2_pred_index_{offense}_primary"
        if actual_col in primary.columns and predicted_col in primary.columns:
            null_mismatches += int(primary[actual_col].isna().ne(primary[predicted_col].isna()).sum())
    return {
        "csv": str(out_csv.relative_to(REPO_ROOT)),
        "rows": int(len(out)),
        "primary_six_rows": int(primary.shape[0]),
        "primary_six_max_abs_candidate_minus_fixed_q2_prediction": _to_float(max_abs_delta),
        "primary_six_null_mismatches": int(null_mismatches),
        "primary_six_passed_rounding_tolerance": bool(
            null_mismatches == 0 and max_abs_delta <= PATHOLOGICAL_INDEX_TOLERANCE
        ),
    }


def _state_lift_summary(candidate: pd.DataFrame, out_csv: Path) -> dict[str, Any]:
    work = candidate.copy()
    work["landscan_day_lifted_person_exposure"] = _bool_series(work, "landscan_day_lifted_person_exposure")
    work["lift_amount"] = (_numeric(work, "exposure_proxy_2024") - _numeric(work, "daytime_population_jobs_proxy")).clip(
        lower=0.0
    )
    grouped = (
        work.groupby("state_fips", dropna=False)
        .agg(
            block_groups=("block_group_geoid", "size"),
            lifted_block_groups=("landscan_day_lifted_person_exposure", "sum"),
            exposure_proxy_sum=("exposure_proxy_2024", "sum"),
            jobs_exposure_sum=("daytime_population_jobs_proxy", "sum"),
            landscan_day_pop_sum=("landscan_day_pop", "sum"),
            lift_amount_sum=("lift_amount", "sum"),
        )
        .reset_index()
    )
    grouped["lifted_share"] = grouped["lifted_block_groups"] / grouped["block_groups"]
    grouped = grouped.sort_values(["lifted_block_groups", "lift_amount_sum"], ascending=[False, False], kind="mergesort")
    _round_for_csv(grouped).to_csv(out_csv, index=False)
    return {
        "csv": str(out_csv.relative_to(REPO_ROOT)),
        "rows": int(len(grouped)),
        "top_states_by_lifted_bg": grouped.head(10).to_dict(orient="records"),
    }


def _render_maps(candidate: pd.DataFrame, candidate_dir: Path) -> dict[str, Any]:
    import warnings

    warnings.filterwarnings("ignore")
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, Normalize

    value_col = "index_total_primary_event_weighted"
    values = candidate[["block_group_geoid", "state_fips", value_col, "landscan_day_lifted_person_exposure"]].copy()
    values["block_group_geoid"] = values["block_group_geoid"].astype(str).str.zfill(12)
    values = values.set_index("block_group_geoid")

    def read_states(states: list[str]) -> Any:
        frames = []
        for state in states:
            zip_path = TIGER_BG_DIR / f"tl_2020_{state}_bg.zip"
            if not zip_path.exists():
                continue
            g = gpd.read_file(f"zip://{zip_path}")
            geoid_col = "GEOID" if "GEOID" in g.columns else "GEOID20"
            state_col = "STATEFP" if "STATEFP" in g.columns else "STATEFP20"
            g = g[[geoid_col, state_col, "geometry"]].rename(
                columns={geoid_col: "block_group_geoid", state_col: "state_fips"}
            )
            g["block_group_geoid"] = g["block_group_geoid"].astype(str).str.zfill(12)
            g["state_fips"] = g["state_fips"].astype(str).str.zfill(2)
            frames.append(g)
        if not frames:
            return gpd.GeoDataFrame(columns=["block_group_geoid", "state_fips", "geometry"], geometry="geometry", crs="EPSG:4269")
        geo = pd.concat(frames, ignore_index=True)
        geo = gpd.GeoDataFrame(geo, geometry="geometry", crs=frames[0].crs or "EPSG:4269")
        if geo.crs is None:
            geo = geo.set_crs("EPSG:4269")
        return geo

    def render(
        *,
        name: str,
        states: list[str],
        bbox: tuple[float, float, float, float] | None,
        title: str,
        figsize: tuple[float, float],
        dpi: int,
    ) -> dict[str, Any]:
        geo = read_states(states)
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            if geo.crs is None:
                geo = geo.set_crs("EPSG:4269")
            geo = geo.to_crs("EPSG:4326")
            geo = geo.cx[minx:maxx, miny:maxy].copy()
        geo["_v"] = pd.to_numeric(values[value_col].reindex(geo["block_group_geoid"]), errors="coerce").to_numpy()
        geo["_lifted"] = (
            values["landscan_day_lifted_person_exposure"].reindex(geo["block_group_geoid"]).fillna(False).astype(bool).to_numpy()
        )
        projected = geo.to_crs(5070)
        vals = projected["_v"].dropna()
        if vals.empty:
            norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            edges = np.unique(np.nanpercentile(vals, np.linspace(0, 100, 11)))
            norm = BoundaryNorm(edges, ncolors=256) if len(edges) > 2 else Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
        fig, ax = plt.subplots(figsize=figsize)
        projected.plot(
            column="_v",
            ax=ax,
            cmap="magma_r",
            norm=norm,
            linewidth=0,
            missing_kwds={"color": "#e8e8e8"},
        )
        if not projected.empty:
            boundaries = projected.dissolve("state_fips").boundary
            boundaries.plot(ax=ax, color="#111111", linewidth=0.35)
            lifted = projected[projected["_lifted"]]
            if not lifted.empty and bbox is not None:
                lifted.boundary.plot(ax=ax, color="#00a6d6", linewidth=0.3, alpha=0.75)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12)
        out_path = candidate_dir / name
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "geometry_rows": int(len(projected)),
            "published_rows": int(vals.size),
            "lifted_rows_in_view": int(projected["_lifted"].sum()) if "_lifted" in projected else 0,
            "bytes": int(out_path.stat().st_size),
        }

    states = sorted(str(x).zfill(2) for x in candidate["state_fips"].dropna().unique())
    states = [state for state in states if state not in RELEASE_EXCLUDED_STATE_FIPS]
    renders = {
        "national": render(
            name="render_national_index_total_primary_event_weighted.png",
            states=states,
            bbox=None,
            title="ambient-denominator-v7 - national BG total primary event-weighted index",
            figsize=(20, 12),
            dpi=95,
        ),
        "boston_metro": render(
            name="render_boston_metro_index_total_primary_event_weighted.png",
            states=["25"],
            bbox=(-71.35, 42.15, -70.85, 42.55),
            title="ambient-denominator-v7 - Boston metro BG total index",
            figsize=(10, 9),
            dpi=140,
        ),
        "rainier": render(
            name="render_rainier_index_total_primary_event_weighted.png",
            states=["53"],
            bbox=(-122.25, 46.55, -121.25, 47.20),
            title="ambient-denominator-v7 - Mount Rainier BG total index",
            figsize=(10, 9),
            dpi=140,
        ),
    }
    try:
        from PIL import Image

        for spec in renders.values():
            path = REPO_ROOT / spec["path"]
            with Image.open(path) as image:
                arr = np.asarray(image.convert("RGB"))
                spec["pixel_min"] = int(arr.min())
                spec["pixel_max"] = int(arr.max())
                spec["nonblank"] = bool(arr.max() > arr.min())
    except Exception as exc:  # pragma: no cover - image inspection is best-effort.
        for spec in renders.values():
            spec["nonblank_check_error"] = str(exc)
    return renders


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(_fmt(value, 3))
            else:
                cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_markdown(report: dict[str, Any]) -> str:
    sanity = report["sanity_gates"]
    lifts = report["lift_summary"]
    suppressed = report["suppressed_to_published"]
    path = report["pathological_cells"]
    truth = report["truth_gradient"]
    static = report["static_offense_unchanged"]

    top_states = [
        {
            "state": row.get("state_fips"),
            "lifted": _fmt_int(row.get("lifted_block_groups")),
            "share": _fmt(row.get("lifted_share"), 4),
            "lift_amount": _fmt_int(row.get("lift_amount_sum")),
        }
        for row in report["lift_by_state"].get("top_states_by_lifted_bg", [])
    ]

    truth_rows = []
    for row in truth.get("comparison", []):
        if row.get("source") != "direct_truth_incidents":
            continue
        truth_rows.append(
            {
                "offense": row.get("offense"),
                "current": _fmt(row.get("current_gradient"), 4),
                "candidate": _fmt(row.get("candidate_gradient"), 4),
                "near": row.get("near_unchanged"),
                "no_degradation": row.get("no_degradation"),
            }
        )

    lines = [
        "# LandScan Day Denominator Candidate Audit",
        "",
        f"- Created: {report['created_at_utc']}",
        f"- Candidate: `{report['candidate_dir']}`",
        f"- Validator ok: {report['validator_ok']}",
        f"- Hard issues: {len(report['hard_issues'])}",
        "",
        "## Denominator Gates",
        "",
        f"- Lifted BGs: {_fmt_int(lifts['lifted_block_groups'])} of {_fmt_int(lifts['block_groups'])} ({_fmt(lifts['lifted_share'], 4)}).",
        f"- Rows with LandScan day <= jobs exposure but lifted: {_fmt_int(sanity['lifts_where_landscan_day_not_greater_than_jobs'])}.",
        f"- BG max-formula mismatches: {_fmt_int(sanity['bg_exposure_max_formula_mismatch_rows'])}; lift-flag mismatches: {_fmt_int(sanity['bg_lift_flag_mismatch_rows'])}.",
        f"- LandScan day zero rows: {_fmt_int(sanity['zero_landscan_day_rows'])}; zero-day rows with residents: {_fmt_int(sanity['zero_landscan_day_rows_with_population'])}.",
        f"- Static offense denominator max deltas: burglary {_fmt(static['burglary']['max_abs_primary_denominator_delta'], 12)}, MVT {_fmt(static['motor_vehicle_theft']['max_abs_primary_denominator_delta'], 12)}.",
        "",
        "## Lift By State",
        "",
        _markdown_table(
            top_states,
            [
                ("State", "state"),
                ("Lifted BGs", "lifted"),
                ("Share", "share"),
                ("Lift Amount", "lift_amount"),
            ],
        ),
        "",
        "## Suppressed To Published",
        "",
        f"- Rows clearing `insufficient_exposure` via LandScan day lift: {_fmt_int(suppressed['rows'])}; unique BGs: {_fmt_int(suppressed['unique_block_groups'])}.",
        f"- By offense: {suppressed['by_offense']}.",
        f"- CSV: `{suppressed['csv']}`.",
        "",
        "## Pathological Cells",
        "",
        f"- Primary six rows: {path['primary_six_rows']}; max abs candidate-vs-fixed-Q2-prediction delta: {_fmt(path['primary_six_max_abs_candidate_minus_fixed_q2_prediction'], 6)}; null mismatches: {path['primary_six_null_mismatches']}.",
        f"- CSV: `{path['csv']}`.",
        "",
        "## Truth Gradient",
        "",
        f"- Truth rows: {_fmt_int(truth.get('truth_rows_person_exposure'))}; covered cities: {_fmt_int(truth.get('covered_truth_cities_available'))}.",
        f"- Direct-truth near-unchanged offenses: {truth.get('direct_truth_near_unchanged_count')} of {len(PERSON_EXPOSURE_OFFENSES)}.",
        f"- Direct-truth no-degradation offenses: {truth.get('direct_truth_no_degradation_count')} of {len(PERSON_EXPOSURE_OFFENSES)}.",
        "",
        _markdown_table(
            truth_rows,
            [
                ("Offense", "offense"),
                ("Current Q5/Q1", "current"),
                ("Candidate Q5/Q1", "candidate"),
                ("Near", "near"),
                ("No Degradation", "no_degradation"),
            ],
        ),
        "",
        f"- CSV: `{truth.get('csv', 'NA')}`.",
        "",
        "## Renders",
        "",
    ]
    for name, spec in report["renders"].items():
        lines.append(
            f"- {name}: `{spec['path']}` ({_fmt_int(spec.get('bytes'))} bytes, nonblank={spec.get('nonblank')}, lifted rows in view={_fmt_int(spec.get('lifted_rows_in_view'))})."
        )
    if report["hard_issues"]:
        lines.extend(["", "## Hard Issues", ""])
        lines.extend(f"- {issue}" for issue in report["hard_issues"])
    return "\n".join(lines) + "\n"


def audit(candidate_dir: Path, *, render: bool = True) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    baseline_path = STATE_OUTPUT_DIR / "crimerisk_block_group_2024_ags_core.parquet"
    candidate_path = candidate_dir / "crimerisk_block_group_2024_ags_core.parquet"
    validation_path = candidate_dir / "validation_summary.json"
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    baseline = _read_bg_surface(baseline_path)
    candidate = _read_bg_surface(candidate_path)
    land = _read_landscan()
    baseline_q2 = _merge_landscan_for_baseline(baseline, land)

    candidate = candidate.sort_values("block_group_geoid", kind="mergesort").reset_index(drop=True)
    baseline_q2 = baseline_q2.sort_values("block_group_geoid", kind="mergesort").reset_index(drop=True)
    if not baseline_q2["block_group_geoid"].equals(candidate["block_group_geoid"]):
        raise RuntimeError("Baseline and candidate block-group IDs do not align after sorting.")
    q2 = _compute_fixed_q2_indexes(baseline_q2)

    jobs = _numeric(candidate, "daytime_population_jobs_proxy")
    land_day = _numeric(candidate, "landscan_day_pop").clip(lower=0.0)
    expected_exposure = np.maximum(jobs.to_numpy(dtype=float), land_day.where(land_day.gt(0.0), 0.0).to_numpy(dtype=float))
    exposure = _numeric(candidate, "exposure_proxy_2024")
    expected_lift = land_day.where(land_day.gt(0.0), 0.0).gt(jobs)
    observed_lift = _bool_series(candidate, "landscan_day_lifted_person_exposure")

    formula_mismatch = np.abs(exposure.to_numpy(dtype=float) - expected_exposure) > 1e-9
    lift_flag_mismatch = observed_lift.ne(expected_lift)
    lifts_where_not_greater = observed_lift & ~land_day.gt(jobs)
    exposure_below_jobs = exposure.lt(jobs - 1e-9)
    exposure_below_landscan = exposure.lt(land_day.where(land_day.gt(0.0), 0.0) - 1e-9)
    zero_landscan = land_day.le(0.0)
    population = _numeric(candidate, "population_2024")

    lift_amount = (exposure - jobs).clip(lower=0.0)
    lift_summary = {
        "block_groups": int(len(candidate)),
        "lifted_block_groups": int(observed_lift.sum()),
        "lifted_share": _to_float(observed_lift.mean()),
        "lift_amount_sum": _to_float(lift_amount.sum()),
        "lift_amount_mean_lifted": _to_float(lift_amount.loc[observed_lift].mean()),
        "lift_amount_median_lifted": _to_float(lift_amount.loc[observed_lift].median()),
        "lift_amount_p95_lifted": _to_float(lift_amount.loc[observed_lift].quantile(0.95)),
        "lift_amount_max": _to_float(lift_amount.max()),
        "landscan_day_pop_sum": _to_float(land_day.sum()),
        "jobs_exposure_sum": _to_float(jobs.sum()),
        "final_exposure_sum": _to_float(exposure.sum()),
    }

    static_unchanged: dict[str, Any] = {}
    for offense in STATIC_PRIMARY_OFFENSES:
        denom_max_abs, denom_null_mismatch = _max_abs_pair_delta(
            baseline_q2[f"primary_denominator_{offense}"],
            candidate[f"primary_denominator_{offense}"],
        )
        index_max_abs, index_null_mismatch = _max_abs_pair_delta(
            baseline_q2[f"index_{offense}_primary"],
            candidate[f"index_{offense}_primary"],
        )
        static_unchanged[offense] = {
            "max_abs_primary_denominator_delta": _to_float(denom_max_abs),
            "primary_denominator_null_mismatches": int(denom_null_mismatch),
            "max_abs_index_delta": _to_float(index_max_abs),
            "index_null_mismatches": int(index_null_mismatch),
        }

    count_deltas: dict[str, Any] = {}
    for offense in OFFENSES_7:
        max_abs, null_mismatch = _max_abs_pair_delta(
            baseline_q2[f"expected_count_{offense}"],
            candidate[f"expected_count_{offense}"],
        )
        count_deltas[offense] = {"max_abs_delta": _to_float(max_abs), "null_mismatches": int(null_mismatch)}

    candidate_q2_total = candidate["index_total_primary_event_weighted"]
    predicted_q2_total = q2["index_total_primary_event_weighted_q2_fixed_count"]
    total_pred_delta, total_pred_null_mismatch = _max_abs_pair_delta(candidate_q2_total, predicted_q2_total)
    offense_pred_delta: dict[str, Any] = {}
    for offense in PERSON_EXPOSURE_OFFENSES:
        max_abs, null_mismatch = _max_abs_pair_delta(
            candidate[f"index_{offense}_primary"],
            q2["component_indexes"][offense],
        )
        offense_pred_delta[offense] = {"max_abs_delta": _to_float(max_abs), "null_mismatches": int(null_mismatch)}

    suppressed = _suppressed_to_published(
        baseline_q2,
        candidate,
        candidate_dir / "landscan_suppressed_to_published.csv",
    )
    pathological = pd.read_csv(PATHOLOGICAL_CELLS, dtype={"block_group_geoid": str})
    path_report = _pathological_before_after(
        pathological,
        baseline_q2,
        candidate,
        q2,
        candidate_dir / "landscan_pathological_before_after.csv",
    )
    state_lift = _state_lift_summary(candidate, candidate_dir / "landscan_lift_by_state.csv")
    truth = _truth_gradient(
        baseline_q2,
        candidate,
        candidate_dir / "landscan_truth_gradient_final_published_rates.csv",
    )

    validator_ok = None
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validator_ok = bool(validation.get("ok"))

    hard_issues: list[str] = []
    if int(formula_mismatch.sum()):
        hard_issues.append(f"BG exposure max-formula mismatches: {int(formula_mismatch.sum())}")
    if int(lift_flag_mismatch.sum()):
        hard_issues.append(f"LandScan lift-flag mismatches: {int(lift_flag_mismatch.sum())}")
    if int(lifts_where_not_greater.sum()):
        hard_issues.append(
            f"Rows lifted even though LandScan day is not greater than jobs exposure: {int(lifts_where_not_greater.sum())}"
        )
    if bool(exposure_below_jobs.any()) or bool(exposure_below_landscan.any()):
        hard_issues.append(
            "Final exposure shrank below a source denominator "
            f"(below jobs={int(exposure_below_jobs.sum())}, below LandScan day={int(exposure_below_landscan.sum())})"
        )
    if lift_summary["lifted_block_groups"] < 10_000 or lift_summary["lifted_block_groups"] > 150_000:
        hard_issues.append(
            "National lifted-BG count is outside the expected tens-of-thousands band: "
            f"{lift_summary['lifted_block_groups']}"
        )
    for offense, metrics in static_unchanged.items():
        if (
            (metrics["max_abs_primary_denominator_delta"] or 0.0) > COUNT_DERIVED_TOLERANCE
            or metrics["primary_denominator_null_mismatches"]
        ):
            hard_issues.append(f"{offense} primary denominator changed unexpectedly: {metrics}")
    if path_report["primary_six_null_mismatches"] or (
        path_report["primary_six_max_abs_candidate_minus_fixed_q2_prediction"] or 0.0
    ) > PATHOLOGICAL_INDEX_TOLERANCE:
        hard_issues.append(
            "Primary six pathological cells do not reproduce the fixed-count Q2 prediction within "
            f"{PATHOLOGICAL_INDEX_TOLERANCE}: {path_report}"
        )
    if total_pred_null_mismatch or total_pred_delta > 1e-6:
        hard_issues.append(
            "Candidate total primary event-weighted index differs from fixed-count Q2 prediction "
            f"(max_abs={total_pred_delta}, null_mismatch={total_pred_null_mismatch})"
        )
    if validator_ok is False:
        hard_issues.append("Full validator summary is not ok.")

    renders: dict[str, Any] = {}
    if render:
        renders = _render_maps(candidate, candidate_dir)
        for name, spec in renders.items():
            if spec.get("bytes", 0) <= 0 or spec.get("nonblank") is False:
                hard_issues.append(f"Render {name} appears blank or empty: {spec}")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir.relative_to(REPO_ROOT)),
        "baseline_surface": str(baseline_path.relative_to(REPO_ROOT)),
        "candidate_surface": str(candidate_path.relative_to(REPO_ROOT)),
        "landscan_bg_parquet": str(LANDSCAN_BG.relative_to(REPO_ROOT)),
        "landscan_source": {
            "product": "LandScan USA 2021",
            "producer": "Oak Ridge National Laboratory",
            "license": "CC BY 4.0",
            "denominator_policy": "For murder, rape, robbery, aggravated_assault, and larceny, exposure_proxy_2024 = max(daytime_population_jobs_proxy, positive landscan_day_pop).",
            "tourist_limitation": "LandScan USA metadata excludes transitory populations such as business travelers and tourists; visitor-heavy areas may still overstate per-person risk and are flagged.",
        },
        "validator_ok": validator_ok,
        "lift_summary": lift_summary,
        "sanity_gates": {
            "bg_exposure_max_formula_mismatch_rows": int(formula_mismatch.sum()),
            "bg_lift_flag_mismatch_rows": int(lift_flag_mismatch.sum()),
            "lifts_where_landscan_day_not_greater_than_jobs": int(lifts_where_not_greater.sum()),
            "final_exposure_below_jobs_rows": int(exposure_below_jobs.sum()),
            "final_exposure_below_positive_landscan_day_rows": int(exposure_below_landscan.sum()),
            "zero_landscan_day_rows": int(zero_landscan.sum()),
            "zero_landscan_day_rows_with_population": int((zero_landscan & population.gt(0.0)).sum()),
            "zero_landscan_day_rows_with_positive_jobs_exposure": int((zero_landscan & jobs.gt(0.0)).sum()),
        },
        "static_offense_unchanged": static_unchanged,
        "expected_count_deltas": count_deltas,
        "fixed_q2_prediction_reconciliation": {
            "total_primary_event_weighted_max_abs_delta": _to_float(total_pred_delta),
            "total_primary_event_weighted_null_mismatches": int(total_pred_null_mismatch),
            "person_offense_index_delta": offense_pred_delta,
            "offense_normalizers": q2["offense_summary"],
            "event_weights": q2["event_weights"],
        },
        "suppressed_to_published": suppressed,
        "pathological_cells": path_report,
        "lift_by_state": state_lift,
        "truth_gradient": truth,
        "renders": renders,
        "hard_issues": hard_issues,
    }
    report_json = candidate_dir / "landscan_denominator_audit.json"
    report_md = candidate_dir / "landscan_denominator_audit.md"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_md.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=REPO_ROOT / "state" / "candidates" / "ambient-denominator-v7",
        help="Candidate output directory to audit.",
    )
    parser.add_argument("--skip-renders", action="store_true", help="Skip PNG render generation.")
    args = parser.parse_args(argv)
    report = audit(args.candidate_dir, render=not args.skip_renders)
    print(
        json.dumps(
            {
                "candidate_dir": report["candidate_dir"],
                "validator_ok": report["validator_ok"],
                "lifted_block_groups": report["lift_summary"]["lifted_block_groups"],
                "suppressed_to_published_rows": report["suppressed_to_published"]["rows"],
                "pathological_primary_six_max_abs_delta": report["pathological_cells"][
                    "primary_six_max_abs_candidate_minus_fixed_q2_prediction"
                ],
                "truth_direct_near_unchanged_count": report["truth_gradient"].get(
                    "direct_truth_near_unchanged_count"
                ),
                "hard_issues": report["hard_issues"],
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return 1 if report["hard_issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
