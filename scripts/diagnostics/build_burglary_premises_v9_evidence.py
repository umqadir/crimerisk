from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_RUN = "burglary-premises-v9"
DEFAULT_BASELINE_DIR = REPO_ROOT / "state" / "output"
GATE_DERIVATION_PATH = REPO_ROOT / "state" / "modeling" / "burglary_gate_ceiling_derivation.json"
BURGLARY_TAU_CALIBRATION_PATH = REPO_ROOT / "state" / "modeling" / "burglary_tau_calibration_2024.json"
WAREHOUSE_TARGET_COUNT = 612
WAREHOUSE_MAX_DESTINATION_POIS = 2

SURFACES = {
    "block_group_ags_core": "crimerisk_block_group_2024_ags_core.parquet",
    "tract_ags_core": "crimerisk_tract_2024_ags_core.parquet",
    "block_group_fbi_calibrated": "crimerisk_block_group_2024_fbi_calibrated.parquet",
    "tract_fbi_calibrated": "crimerisk_tract_2024_fbi_calibrated.parquet",
}
NON_BURGLARY_OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "larceny",
    "motor_vehicle_theft",
]
NAMED_BLOCK_GROUPS = [
    {
        "label": "Walmart Festus BG",
        "block_group_geoid": "290997007005",
        "lookup_source": "Local Overture point WALMART #000069 / Walmart Grocery Pickup near (-90.3880, 38.2121), spatially joined to TIGER 2020 BG.",
    },
    {
        "label": "Mall of America BG",
        "block_group_geoid": "270530251002",
        "lookup_source": "Local Overture Mall of America food-court/interior points near (-93.2426, 44.8555), spatially joined to TIGER 2020 BG.",
    },
    {
        "label": "Ontario CA warehouse BG",
        "block_group_geoid": "060710022064",
        "lookup_source": "Supervisor-specified audit cell.",
    },
    {
        "label": "Vegas Strip Bellagio BG",
        "block_group_geoid": "320030067001",
        "lookup_source": "Local Overture Bellagio Hotel & Casino point near (-115.1765, 36.1129), spatially joined to TIGER 2020 BG.",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _available_columns(path: Path) -> list[str]:
    return list(pd.read_parquet(path, engine="pyarrow").columns)


def _read_surface(path: Path, columns: list[str]) -> pd.DataFrame:
    available = set(_available_columns(path))
    keep = [col for col in columns if col in available]
    return pd.read_parquet(path, columns=keep)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fold_ranges(calibration: dict[str, Any]) -> dict[str, dict[str, float]]:
    folds = calibration.get("fold_safe", {}).get("folds", [])
    out: dict[str, dict[str, float]] = {}
    for key in ("k_destination_poi", "k_retail_jobs", "k_industrial_jobs"):
        values = [float(fold[key]) for fold in folds if fold.get(key) is not None]
        if not values:
            continue
        series = pd.Series(values, dtype=float)
        out[key] = {
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
        }
    return out


def _build_calibration_artifacts(manifest: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    calibration = manifest["summary"]["burglary_commercial_calibration"]
    ranges = _fold_ranges(calibration)
    summary = {
        "denominator_form": calibration.get("denominator_form"),
        "denominator_formula": calibration.get("denominator_formula"),
        "k_vector": calibration.get("k_vector"),
        "coefficients": calibration.get("coefficients"),
        "fold_count": calibration.get("fold_safe", {}).get("fold_count"),
        "fold_k_ranges": ranges,
        "calibration_rows": calibration.get("calibration_rows"),
        "calibration_block_groups": calibration.get("calibration_block_groups"),
        "calibration_city_count": calibration.get("calibration_city_count"),
        "calibration_truth_city_count": calibration.get("calibration_truth_city_count"),
        "calibration_year_min": calibration.get("calibration_year_min"),
        "calibration_year_max": calibration.get("calibration_year_max"),
        "nnls_residual_norm": calibration.get("nnls_residual_norm"),
        "single_term_nnls_residual_norm": calibration.get("single_term_nnls_residual_norm"),
        "nnls_residual_delta_vs_single_term": calibration.get("nnls_residual_delta_vs_single_term"),
        "beats_single_term_nnls_residual": calibration.get("beats_single_term_nnls_residual"),
    }
    if not summary["beats_single_term_nnls_residual"]:
        raise SystemExit("STOP: multi-term burglary NNLS residual does not beat the single-term form")
    _write_json(evidence_dir / "calibration_summary.json", summary)

    def k_range_columns(k_field: str) -> dict[str, float | None]:
        stats = ranges.get(k_field, {})
        return {f"fold_{stat}": stats.get(stat) for stat in ("min", "p25", "median", "p75", "max")}

    pd.DataFrame(
        [
            {
                "term": "destination_poi_total",
                "k": calibration.get("k_vector", {}).get("k_destination_poi"),
                "coefficient": calibration.get("coefficients", {}).get("destination_poi_total"),
                **k_range_columns("k_destination_poi"),
            },
            {
                "term": "lodes_retail_jobs",
                "k": calibration.get("k_vector", {}).get("k_retail_jobs"),
                "coefficient": calibration.get("coefficients", {}).get("lodes_retail_jobs"),
                **k_range_columns("k_retail_jobs"),
            },
            {
                "term": "lodes_industrial_jobs",
                "k": calibration.get("k_vector", {}).get("k_industrial_jobs"),
                "coefficient": calibration.get("coefficients", {}).get("lodes_industrial_jobs"),
                **k_range_columns("k_industrial_jobs"),
            },
        ]
    ).to_csv(evidence_dir / "calibration_table.csv", index=False)
    return summary


def _build_gradient_artifacts(manifest: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    gradients = manifest["summary"]["burglary_commercial_gradient"]
    rows: list[dict[str, Any]] = []
    quintile_rows: list[dict[str, Any]] = []
    for surface, payload in gradients.items():
        rows.extend(
            [
                {
                    "surface": surface,
                    "regime": "pooled_report_only",
                    "after_q5_q1_mean": payload.get("after_q5_q1_mean"),
                    "before_q5_q1_mean": payload.get("before_q5_q1_mean"),
                    "rows": payload.get("rows"),
                    "ok": payload.get("ok"),
                },
                {
                    "surface": surface,
                    "regime": "direct_city_incident",
                    "after_q5_q1_mean": payload.get("after_q5_q1_mean_direct"),
                    "before_q5_q1_mean": None,
                    "rows": sum(q.get("rows", 0) for q in payload.get("regime_quintiles", {}).get("direct_city_incident", [])),
                    "ok": payload.get("ok"),
                },
                {
                    "surface": surface,
                    "regime": "modeled_transfer",
                    "after_q5_q1_mean": payload.get("after_q5_q1_mean_modeled"),
                    "before_q5_q1_mean": None,
                    "rows": sum(q.get("rows", 0) for q in payload.get("regime_quintiles", {}).get("modeled_transfer", [])),
                    "ok": payload.get("ok"),
                },
            ]
        )
        for quintile in payload.get("quintiles", []):
            quintile_rows.append({"surface": surface, "regime": "pooled_report_only", **quintile})
        for regime, qs in payload.get("regime_quintiles", {}).items():
            for quintile in qs:
                quintile_rows.append({"surface": surface, "regime": regime, **quintile})

    gradient_df = pd.DataFrame(rows)
    quintile_df = pd.DataFrame(quintile_rows)
    gradient_df.to_csv(evidence_dir / "gradient_splits.csv", index=False)
    quintile_df.to_csv(evidence_dir / "gradient_quintiles.csv", index=False)

    gate = _read_json(GATE_DERIVATION_PATH)
    summary = {
        "candidate_gradient_splits": rows,
        "gate_derivation_path": str(GATE_DERIVATION_PATH.relative_to(REPO_ROOT)),
        "gate_derivation": gate,
    }
    _write_json(evidence_dir / "gradient_splits.json", summary)
    return summary


def _column_values_equal(left: pd.Series, right: pd.Series) -> tuple[bool, int, float | None]:
    if pd.api.types.is_bool_dtype(left) or pd.api.types.is_bool_dtype(right):
        null_mismatch = int((left.isna() != right.isna()).sum())
        l_bool = left.fillna(False).astype(bool)
        r_bool = right.fillna(False).astype(bool)
        mismatch = int((l_bool != r_bool).sum()) + null_mismatch
        return mismatch == 0, mismatch, None
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        l = pd.to_numeric(left, errors="coerce")
        r = pd.to_numeric(right, errors="coerce")
        null_mismatch = int((l.isna() != r.isna()).sum())
        diff = (l - r).abs()
        max_abs = float(diff.max(skipna=True)) if diff.notna().any() else 0.0
        mismatch = int((diff.fillna(0.0) != 0.0).sum()) + null_mismatch
        return mismatch == 0, mismatch, max_abs
    l_obj = left.astype("string").fillna("<NA>")
    r_obj = right.astype("string").fillna("<NA>")
    mismatch = int((l_obj != r_obj).sum())
    return mismatch == 0, mismatch, None


def _offense_columns(columns: list[str], offense: str) -> list[str]:
    tokens = [offense]
    if offense == "motor_vehicle_theft":
        tokens.append("mvt")
    return [
        col
        for col in columns
        if any(token in col for token in tokens)
        and "burglary" not in col
        and not col.startswith("expected_count_total")
        and not col.startswith("expected_count_property")
        and not col.startswith("expected_count_personal")
    ]


def _build_non_burglary_identity(
    baseline_dir: Path,
    candidate_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for surface, filename in SURFACES.items():
        baseline_path = baseline_dir / filename
        candidate_path = candidate_dir / filename
        baseline_cols = _available_columns(baseline_path)
        candidate_cols = _available_columns(candidate_path)
        key_col = "block_group_geoid" if filename.startswith("crimerisk_block_group") else "tract_id"
        for offense in NON_BURGLARY_OFFENSES:
            cols = sorted(set(_offense_columns(baseline_cols, offense)).intersection(_offense_columns(candidate_cols, offense)))
            if key_col not in cols:
                cols = [key_col, *cols]
            baseline = pd.read_parquet(baseline_path, columns=cols).sort_values(key_col).reset_index(drop=True)
            candidate = pd.read_parquet(candidate_path, columns=cols).sort_values(key_col).reset_index(drop=True)
            if not baseline[key_col].astype("string").equals(candidate[key_col].astype("string")):
                failure = {
                    "surface": surface,
                    "offense": offense,
                    "column": key_col,
                    "mismatched_values": None,
                    "max_abs_delta": None,
                    "reason": "key order/value mismatch",
                }
                failures.append(failure)
                records.append(failure)
                continue
            compared = 0
            for col in cols:
                if col == key_col:
                    continue
                equal, mismatch_count, max_abs = _column_values_equal(baseline[col], candidate[col])
                record = {
                    "surface": surface,
                    "offense": offense,
                    "column": col,
                    "rows": int(len(baseline)),
                    "mismatched_values": int(mismatch_count),
                    "max_abs_delta": max_abs,
                    "identical": bool(equal),
                }
                records.append(record)
                compared += 1
                if not equal:
                    failures.append(record)
            records.append(
                {
                    "surface": surface,
                    "offense": offense,
                    "column": "__offense_column_count__",
                    "rows": int(len(baseline)),
                    "mismatched_values": 0,
                    "max_abs_delta": 0.0,
                    "identical": True,
                    "compared_columns": compared,
                }
            )
    pd.DataFrame(records).to_csv(evidence_dir / "byte_identity_non_burglary.csv", index=False)
    summary = {
        "ok": not failures,
        "baseline_dir": str(baseline_dir.relative_to(REPO_ROOT)),
        "candidate_dir": str(candidate_dir.relative_to(REPO_ROOT)),
        "surfaces": list(SURFACES),
        "offenses": NON_BURGLARY_OFFENSES,
        "record_count": len(records),
        "details_csv": "byte_identity_non_burglary.csv",
        "failure_count": len(failures),
        "failures": failures[:25],
    }
    _write_json(evidence_dir / "byte_identity_non_burglary.json", summary)
    if failures:
        raise SystemExit(f"STOP: non-burglary offense columns changed ({len(failures)} failures)")
    return summary


def _select_warehouse_threshold(df: pd.DataFrame) -> tuple[float, bool]:
    eligible = df[
        df["v8_index_burglary_primary"].notna()
        & df["destination_poi_total"].le(WAREHOUSE_MAX_DESTINATION_POIS)
        & df["lodes_industrial_jobs"].gt(0)
    ].copy()
    exact: list[float] = []
    for threshold in sorted(float(v) for v in eligible["lodes_industrial_jobs"].dropna().unique()):
        if int(eligible["lodes_industrial_jobs"].ge(threshold).sum()) == WAREHOUSE_TARGET_COUNT:
            exact.append(threshold)
    if exact:
        return max(exact), True
    ranked = eligible.sort_values("lodes_industrial_jobs", ascending=False)
    if len(ranked) < WAREHOUSE_TARGET_COUNT:
        raise ValueError("not enough eligible warehouse-like BGs to reproduce audit census")
    return float(ranked.iloc[WAREHOUSE_TARGET_COUNT - 1]["lodes_industrial_jobs"]), False


def _build_warehouse_artifacts(
    baseline_dir: Path,
    candidate_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    cols = [
        "block_group_geoid",
        "state_fips",
        "households_total",
        "commercial_premises_total",
        "destination_poi_total",
        "lodes_retail_jobs",
        "lodes_industrial_jobs",
        "primary_denominator_burglary",
        "index_burglary_primary",
        "rate_burglary_primary",
        "expected_count_burglary",
        "estimate_mode_burglary",
    ]
    baseline_path = baseline_dir / SURFACES["block_group_ags_core"]
    candidate_path = candidate_dir / SURFACES["block_group_ags_core"]
    baseline = _read_surface(baseline_path, ["block_group_geoid", "index_burglary_primary", "rate_burglary_primary", "primary_denominator_burglary", "expected_count_burglary"]).rename(
        columns={
            "index_burglary_primary": "v8_index_burglary_primary",
            "rate_burglary_primary": "v8_rate_burglary_primary",
            "primary_denominator_burglary": "v8_primary_denominator_burglary",
            "expected_count_burglary": "v8_expected_count_burglary",
        }
    )
    candidate = _read_surface(candidate_path, cols).rename(
        columns={
            "index_burglary_primary": "v9_index_burglary_primary",
            "rate_burglary_primary": "v9_rate_burglary_primary",
            "primary_denominator_burglary": "v9_primary_denominator_burglary",
            "expected_count_burglary": "v9_expected_count_burglary",
        }
    )
    joined = baseline.merge(candidate, on="block_group_geoid", how="inner")
    for col in ("destination_poi_total", "lodes_retail_jobs", "lodes_industrial_jobs"):
        joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0.0)
    threshold, exact = _select_warehouse_threshold(joined)
    warehouse_mask = joined["destination_poi_total"].le(WAREHOUSE_MAX_DESTINATION_POIS) & joined[
        "lodes_industrial_jobs"
    ].ge(threshold)
    published_v8 = joined["v8_index_burglary_primary"].notna()
    published_v9 = joined["v9_index_burglary_primary"].notna()
    warehouse = joined.loc[warehouse_mask & published_v8 & published_v9].copy()
    warehouse.to_csv(evidence_dir / "warehouse_class_rows.csv", index=False)

    def median(series: pd.Series) -> float | None:
        value = pd.to_numeric(series, errors="coerce").median(skipna=True)
        return float(value) if pd.notna(value) else None

    v8_overall = median(joined.loc[published_v8, "v8_index_burglary_primary"])
    v9_overall = median(joined.loc[published_v9, "v9_index_burglary_primary"])
    v8_class = median(warehouse["v8_index_burglary_primary"])
    v9_class = median(warehouse["v9_index_burglary_primary"])
    before_gap = abs(v8_class - v8_overall) if v8_class is not None and v8_overall is not None else None
    after_gap = abs(v9_class - v9_overall) if v9_class is not None and v9_overall is not None else None
    summary = {
        "class_definition": {
            "lodes_industrial_jobs_threshold": threshold,
            "max_destination_poi_total": WAREHOUSE_MAX_DESTINATION_POIS,
            "target_count": WAREHOUSE_TARGET_COUNT,
            "exact_threshold_count_match": exact,
        },
        "class_count": int(len(warehouse)),
        "v8_overall_published_median_index": v8_overall,
        "v8_class_median_index": v8_class,
        "v8_class_minus_overall": None if v8_class is None or v8_overall is None else v8_class - v8_overall,
        "v9_overall_published_median_index": v9_overall,
        "v9_class_median_index": v9_class,
        "v9_class_minus_overall": None if v9_class is None or v9_overall is None else v9_class - v9_overall,
        "absolute_gap_before": before_gap,
        "absolute_gap_after": after_gap,
        "gap_improvement_toward_overall": None if before_gap is None or after_gap is None else before_gap - after_gap,
    }
    pd.DataFrame([summary | summary["class_definition"]]).drop(columns=["class_definition"]).to_csv(
        evidence_dir / "warehouse_class_before_after.csv",
        index=False,
    )
    _write_json(evidence_dir / "warehouse_class_before_after.json", summary)
    if int(len(warehouse)) != WAREHOUSE_TARGET_COUNT:
        raise SystemExit(f"STOP: warehouse audit census is {len(warehouse)}, expected {WAREHOUSE_TARGET_COUNT}")
    if summary["gap_improvement_toward_overall"] is not None and summary["gap_improvement_toward_overall"] < 0:
        raise SystemExit("STOP: warehouse audit class moved farther from the overall burglary median")
    return summary


def _build_named_cells(
    baseline_dir: Path,
    candidate_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    baseline_cols = [
        "block_group_geoid",
        "population_2024",
        "households_total",
        "commercial_premises_total",
        "primary_denominator_burglary",
        "expected_count_burglary",
        "rate_burglary_primary",
        "index_burglary_primary",
        "estimate_mode_burglary",
        "source_mode_burglary",
    ]
    candidate_cols = [
        *baseline_cols,
        "destination_poi_total",
        "lodes_retail_jobs",
        "lodes_industrial_jobs",
        "burglary_destination_poi_exposure_weight",
        "burglary_retail_jobs_exposure_weight",
        "burglary_industrial_jobs_exposure_weight",
    ]
    baseline = _read_surface(baseline_dir / SURFACES["block_group_ags_core"], baseline_cols).rename(
        columns={
            col: f"v8_{col}"
            for col in baseline_cols
            if col != "block_group_geoid"
        }
    )
    candidate = _read_surface(candidate_dir / SURFACES["block_group_ags_core"], candidate_cols).rename(
        columns={
            col: f"v9_{col}"
            for col in candidate_cols
            if col != "block_group_geoid"
        }
    )
    lookup = pd.DataFrame(NAMED_BLOCK_GROUPS)
    table = lookup.merge(baseline, on="block_group_geoid", how="left").merge(candidate, on="block_group_geoid", how="left")
    table.to_csv(evidence_dir / "named_cells.csv", index=False)
    payload = {"rows": table.to_dict(orient="records")}
    _write_json(evidence_dir / "named_cells.json", payload)
    return payload


def _build_heldout_artifact(
    baseline_dir: Path,
    candidate_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    tau = _read_json(BURGLARY_TAU_CALIBRATION_PATH)
    baseline = _read_surface(
        baseline_dir / SURFACES["block_group_ags_core"],
        ["block_group_geoid", "expected_count_burglary"],
    ).rename(columns={"expected_count_burglary": "v8_expected_count_burglary"})
    candidate = _read_surface(
        candidate_dir / SURFACES["block_group_ags_core"],
        ["block_group_geoid", "expected_count_burglary"],
    ).rename(columns={"expected_count_burglary": "v9_expected_count_burglary"})
    joined = baseline.merge(candidate, on="block_group_geoid", how="inner")
    diff = (
        pd.to_numeric(joined["v9_expected_count_burglary"], errors="coerce")
        - pd.to_numeric(joined["v8_expected_count_burglary"], errors="coerce")
    ).abs()
    selected_tau = tau.get("one_se_tau", tau.get("production_tau", tau.get("selected_tau_after_backstop")))
    selected_curve = next(
        (
            row
            for row in tau.get("curve", [])
            if selected_tau is not None and abs(float(row.get("tau", -999.0)) - float(selected_tau)) < 1e-12
        ),
        {},
    )
    summary = {
        "tau_artifact": str(BURGLARY_TAU_CALIBRATION_PATH.relative_to(REPO_ROOT)),
        "selected_tau": selected_tau,
        "one_se_tvd": tau.get("one_se_tvd"),
        "one_se_threshold": tau.get("one_se_threshold"),
        "selected_tau_city_bootstrap_se": selected_curve.get("city_bootstrap_se"),
        "argmin_tau": tau.get("argmin_tau"),
        "argmin_tvd": tau.get("argmin_tvd"),
        "argmin_se": tau.get("argmin_se"),
        "expected_count_burglary_v8_v9_max_abs_delta": float(diff.max(skipna=True)),
        "expected_count_burglary_v8_v9_changed_rows": int(diff.fillna(0.0).ne(0.0).sum()),
        "denominator_only_change_interpretation": "Held-out burglary allocation TVD is unchanged because expected burglary counts/shares are byte-identical; v9 changes the burglary publication denominator, rates, and indexes only.",
    }
    _write_json(evidence_dir / "heldout_burglary_allocation_quality.json", summary)
    if summary["expected_count_burglary_v8_v9_changed_rows"]:
        raise SystemExit("STOP: burglary expected counts changed, so held-out allocation quality must be rerun")
    return summary


def build_evidence(candidate_run: str, baseline_dir: Path) -> dict[str, Any]:
    candidate_dir = REPO_ROOT / "state" / "candidates" / candidate_run
    evidence_dir = candidate_dir / "burglary_premises_v9_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(candidate_dir / "manifest.json")

    summary = {
        "candidate_run": candidate_run,
        "baseline_dir": str(baseline_dir.relative_to(REPO_ROOT)),
        "candidate_dir": str(candidate_dir.relative_to(REPO_ROOT)),
        "calibration": _build_calibration_artifacts(manifest, evidence_dir),
        "gradients": _build_gradient_artifacts(manifest, evidence_dir),
        "non_burglary_identity": _build_non_burglary_identity(baseline_dir, candidate_dir, evidence_dir),
        "warehouse_class": _build_warehouse_artifacts(baseline_dir, candidate_dir, evidence_dir),
        "named_cells": _build_named_cells(baseline_dir, candidate_dir, evidence_dir),
        "heldout_burglary_allocation_quality": _build_heldout_artifact(baseline_dir, candidate_dir, evidence_dir),
    }
    _write_json(evidence_dir / "v9_evidence_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    args = parser.parse_args()
    print(json.dumps(build_evidence(args.candidate_run, args.baseline_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
