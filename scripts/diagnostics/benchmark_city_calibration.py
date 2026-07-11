from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_share_benchmark import build_city_share_diagnostics, build_city_share_truth_model_frame, weighted_mean


def _group_calibration_summary(
    diagnostics: pd.DataFrame,
    *,
    group_col: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_value, group in diagnostics.groupby(group_col, dropna=False, sort=True):
        rows.append(
            {
                str(group_col): str(group_value),
                "rows": int(len(group)),
                "incident_total": float(pd.to_numeric(group["incident_total"], errors="coerce").fillna(0.0).sum()),
                "baseline_weighted_total_variation_distance_mean": weighted_mean(
                    group, "baseline_total_variation_distance", "incident_total"
                ),
                "calibrated_weighted_total_variation_distance_mean": weighted_mean(
                    group, "calibrated_total_variation_distance", "incident_total"
                ),
                "weighted_tvd_delta": weighted_mean(group, "tvd_delta", "incident_total"),
                "baseline_weighted_pearson_share_mean": weighted_mean(
                    group, "baseline_pearson_share", "incident_total"
                ),
                "calibrated_weighted_pearson_share_mean": weighted_mean(
                    group, "calibrated_pearson_share", "incident_total"
                ),
                "baseline_weighted_spearman_share_mean": weighted_mean(
                    group, "baseline_spearman_share", "incident_total"
                ),
                "calibrated_weighted_spearman_share_mean": weighted_mean(
                    group, "calibrated_spearman_share", "incident_total"
                ),
                "baseline_weighted_top10_capture_mean": weighted_mean(
                    group, "baseline_top_10pct_true_mass_in_model_top_10pct", "incident_total"
                ),
                "calibrated_weighted_top10_capture_mean": weighted_mean(
                    group, "calibrated_top_10pct_true_mass_in_model_top_10pct", "incident_total"
                ),
                "weighted_fallback_incident_share_mean": weighted_mean(
                    group, "fallback_incident_share", "incident_total"
                ),
                "improved_tvd_rows": int(pd.to_numeric(group["tvd_delta"], errors="coerce").lt(0).sum()),
                "worsened_tvd_rows": int(pd.to_numeric(group["tvd_delta"], errors="coerce").gt(0).sum()),
            }
        )
    return rows


def build_city_calibration_benchmark(
    *,
    city_shares: pd.DataFrame,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    year: int,
    min_train_rows_per_offense: int = 100,
    min_unique_prediction_values: int = 8,
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = build_city_share_truth_model_frame(
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=int(year),
    )
    if merged.empty:
        return pd.DataFrame(), {"year": int(year), "rows": 0}

    frame = merged.copy()
    frame["row_weight"] = (
        pd.to_numeric(frame["incident_total"], errors="coerce").fillna(0.0)
        / frame.groupby(["city_name", "jurisdiction_id", "state_fips", "offense"], dropna=False)["bg_id"].transform("size")
        .replace(0, np.nan)
    ).fillna(0.0)

    holdout_keys = (
        frame[["city_name", "jurisdiction_id"]]
        .dropna(subset=["jurisdiction_id"])
        .drop_duplicates()
        .sort_values(["city_name", "jurisdiction_id"], kind="mergesort")
    )
    rows: list[pd.DataFrame] = []

    for holdout in holdout_keys.itertuples(index=False):
        holdout_city = str(holdout.city_name)
        holdout_jurisdiction_id = str(holdout.jurisdiction_id)
        train = frame[frame["jurisdiction_id"].astype(str) != holdout_jurisdiction_id].copy()
        test = frame[frame["jurisdiction_id"].astype(str) == holdout_jurisdiction_id].copy()
        if train.empty or test.empty:
            continue

        test["calibrated_unscaled_share"] = np.nan
        for offense in sorted(test["offense"].astype(str).dropna().unique().tolist()):
            train_offense = train[train["offense"].astype(str) == offense].copy()
            test_offense_mask = test["offense"].astype(str) == offense
            if train_offense.empty or not bool(test_offense_mask.any()):
                continue
            x_train = pd.to_numeric(train_offense["model_share"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            y_train = pd.to_numeric(train_offense["true_share"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            w_train = pd.to_numeric(train_offense["row_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            if (
                len(train_offense) < int(min_train_rows_per_offense)
                or int(pd.Series(x_train).nunique()) < int(min_unique_prediction_values)
                or float(np.max(y_train, initial=0.0)) <= 0.0
            ):
                test.loc[test_offense_mask, "calibrated_unscaled_share"] = pd.to_numeric(
                    test.loc[test_offense_mask, "model_share"], errors="coerce"
                ).fillna(0.0)
                continue

            calibrator = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
            calibrator.fit(x_train, y_train, sample_weight=w_train if np.any(w_train > 0) else None)
            x_test = pd.to_numeric(test.loc[test_offense_mask, "model_share"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            pred = np.asarray(calibrator.predict(x_test), dtype=float)
            pred = np.clip(pred, 0.0, None)
            test.loc[test_offense_mask, "calibrated_unscaled_share"] = pred

        test["calibrated_total"] = test.groupby(
            ["city_name", "jurisdiction_id", "state_fips", "offense"],
            dropna=False,
        )["calibrated_unscaled_share"].transform("sum")
        test["calibrated_share"] = np.where(
            pd.to_numeric(test["calibrated_total"], errors="coerce").fillna(0.0) > 0,
            pd.to_numeric(test["calibrated_unscaled_share"], errors="coerce").fillna(0.0)
            / pd.to_numeric(test["calibrated_total"], errors="coerce").fillna(np.nan),
            pd.to_numeric(test["model_share"], errors="coerce").fillna(0.0),
        )

        baseline_diag, _ = build_city_share_diagnostics(test, predicted_share_col="model_share")
        calibrated_diag, _ = build_city_share_diagnostics(test, predicted_share_col="calibrated_share")
        comparison = baseline_diag.merge(
            calibrated_diag,
            on=[
                "city_name",
                "jurisdiction_id",
                "state_fips",
                "offense",
                "incident_total",
                "primary_geocode_quality_tier",
                "fallback_incident_share",
                "offense_volume_band",
            ],
            how="inner",
            suffixes=("_baseline", "_calibrated"),
        )
        if comparison.empty:
            continue
        comparison["holdout_city_name"] = holdout_city
        comparison["holdout_jurisdiction_id"] = holdout_jurisdiction_id
        comparison["baseline_total_variation_distance"] = pd.to_numeric(
            comparison["total_variation_distance_baseline"], errors="coerce"
        )
        comparison["calibrated_total_variation_distance"] = pd.to_numeric(
            comparison["total_variation_distance_calibrated"], errors="coerce"
        )
        comparison["baseline_share_rmse"] = pd.to_numeric(comparison["share_rmse_baseline"], errors="coerce")
        comparison["calibrated_share_rmse"] = pd.to_numeric(comparison["share_rmse_calibrated"], errors="coerce")
        comparison["baseline_pearson_share"] = pd.to_numeric(comparison["pearson_share_baseline"], errors="coerce")
        comparison["calibrated_pearson_share"] = pd.to_numeric(comparison["pearson_share_calibrated"], errors="coerce")
        comparison["baseline_spearman_share"] = pd.to_numeric(comparison["spearman_share_baseline"], errors="coerce")
        comparison["calibrated_spearman_share"] = pd.to_numeric(comparison["spearman_share_calibrated"], errors="coerce")
        comparison["baseline_top_10pct_true_mass_in_model_top_10pct"] = pd.to_numeric(
            comparison["top_10pct_true_mass_in_model_top_10pct_baseline"],
            errors="coerce",
        )
        comparison["calibrated_top_10pct_true_mass_in_model_top_10pct"] = pd.to_numeric(
            comparison["top_10pct_true_mass_in_model_top_10pct_calibrated"],
            errors="coerce",
        )
        comparison["tvd_delta"] = (
            comparison["calibrated_total_variation_distance"] - comparison["baseline_total_variation_distance"]
        )
        comparison["share_rmse_delta"] = comparison["calibrated_share_rmse"] - comparison["baseline_share_rmse"]
        comparison["top10_capture_delta"] = (
            comparison["calibrated_top_10pct_true_mass_in_model_top_10pct"]
            - comparison["baseline_top_10pct_true_mass_in_model_top_10pct"]
        )
        rows.append(
            comparison[
                [
                    "holdout_city_name",
                    "city_name",
                    "holdout_jurisdiction_id",
                    "jurisdiction_id",
                    "state_fips",
                    "offense",
                    "incident_total",
                    "primary_geocode_quality_tier",
                    "fallback_incident_share",
                    "offense_volume_band",
                    "baseline_total_variation_distance",
                    "calibrated_total_variation_distance",
                    "baseline_share_rmse",
                    "calibrated_share_rmse",
                    "baseline_pearson_share",
                    "calibrated_pearson_share",
                    "baseline_spearman_share",
                    "calibrated_spearman_share",
                    "baseline_top_10pct_true_mass_in_model_top_10pct",
                    "calibrated_top_10pct_true_mass_in_model_top_10pct",
                    "tvd_delta",
                    "share_rmse_delta",
                    "top10_capture_delta",
                ]
            ].copy()
        )

    diagnostics = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if diagnostics.empty:
        return diagnostics, {"year": int(year), "rows": 0}

    summary: dict[str, object] = {
        "year": int(year),
        "rows": int(len(diagnostics)),
        "holdout_city_count": int(diagnostics["holdout_city_name"].astype(str).nunique()),
        "incident_total": float(pd.to_numeric(diagnostics["incident_total"], errors="coerce").fillna(0.0).sum()),
        "split_mode": "leave_one_city_out",
        "holdout_unit": "jurisdiction_id",
        "calibration_family": "isotonic_regression",
        "baseline_weighted_total_variation_distance_mean": weighted_mean(
            diagnostics, "baseline_total_variation_distance", "incident_total"
        ),
        "calibrated_weighted_total_variation_distance_mean": weighted_mean(
            diagnostics, "calibrated_total_variation_distance", "incident_total"
        ),
        "baseline_weighted_share_rmse_mean": weighted_mean(diagnostics, "baseline_share_rmse", "incident_total"),
        "calibrated_weighted_share_rmse_mean": weighted_mean(diagnostics, "calibrated_share_rmse", "incident_total"),
        "baseline_weighted_pearson_share_mean": weighted_mean(
            diagnostics, "baseline_pearson_share", "incident_total"
        ),
        "calibrated_weighted_pearson_share_mean": weighted_mean(
            diagnostics, "calibrated_pearson_share", "incident_total"
        ),
        "baseline_weighted_spearman_share_mean": weighted_mean(
            diagnostics, "baseline_spearman_share", "incident_total"
        ),
        "calibrated_weighted_spearman_share_mean": weighted_mean(
            diagnostics, "calibrated_spearman_share", "incident_total"
        ),
        "baseline_weighted_top10_capture_mean": weighted_mean(
            diagnostics, "baseline_top_10pct_true_mass_in_model_top_10pct", "incident_total"
        ),
        "calibrated_weighted_top10_capture_mean": weighted_mean(
            diagnostics, "calibrated_top_10pct_true_mass_in_model_top_10pct", "incident_total"
        ),
        "weighted_tvd_delta": weighted_mean(diagnostics, "tvd_delta", "incident_total"),
        "weighted_share_rmse_delta": weighted_mean(diagnostics, "share_rmse_delta", "incident_total"),
        "weighted_top10_capture_delta": weighted_mean(diagnostics, "top10_capture_delta", "incident_total"),
        "improved_tvd_rows": int(pd.to_numeric(diagnostics["tvd_delta"], errors="coerce").lt(0).sum()),
        "worsened_tvd_rows": int(pd.to_numeric(diagnostics["tvd_delta"], errors="coerce").gt(0).sum()),
    }
    by_holdout_city = []
    for holdout_city, subset in diagnostics.groupby("holdout_city_name", dropna=False):
        by_holdout_city.append(
            {
                "holdout_city_name": str(holdout_city),
                "rows": int(len(subset)),
                "incident_total": float(pd.to_numeric(subset["incident_total"], errors="coerce").fillna(0.0).sum()),
                "baseline_weighted_total_variation_distance_mean": weighted_mean(
                    subset, "baseline_total_variation_distance", "incident_total"
                ),
                "calibrated_weighted_total_variation_distance_mean": weighted_mean(
                    subset, "calibrated_total_variation_distance", "incident_total"
                ),
                "weighted_tvd_delta": weighted_mean(subset, "tvd_delta", "incident_total"),
                "baseline_weighted_pearson_share_mean": weighted_mean(
                    subset, "baseline_pearson_share", "incident_total"
                ),
                "calibrated_weighted_pearson_share_mean": weighted_mean(
                    subset, "calibrated_pearson_share", "incident_total"
                ),
            }
        )
    summary["by_holdout_city"] = sorted(by_holdout_city, key=lambda row: row["holdout_city_name"])
    summary["by_primary_geocode_quality_tier"] = _group_calibration_summary(
        diagnostics,
        group_col="primary_geocode_quality_tier",
    )
    summary["by_offense_volume_band"] = _group_calibration_summary(
        diagnostics,
        group_col="offense_volume_band",
    )
    return diagnostics, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--city-shares-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet",
    )
    parser.add_argument(
        "--bg-prior-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "bg_prior_long_2024.parquet",
    )
    parser.add_argument(
        "--bg-crosswalk-path",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument("--diagnostics-out", type=Path, default=None)
    parser.add_argument("--summary-json-out", type=Path, default=None)
    parser.add_argument("--min-train-rows-per-offense", type=int, default=100)
    parser.add_argument("--min-unique-prediction-values", type=int, default=8)
    args = parser.parse_args()

    year = int(args.year)
    if args.diagnostics_out is None:
        args.diagnostics_out = REPO_ROOT / "state" / "modeling" / f"city_calibration_benchmark_{year}.parquet"
    if args.summary_json_out is None:
        args.summary_json_out = REPO_ROOT / "state" / "modeling" / f"city_calibration_benchmark_{year}.json"

    city_shares = pd.read_parquet(args.city_shares_path)
    bg_prior = pd.read_parquet(args.bg_prior_path)
    bg_crosswalk = pd.read_parquet(args.bg_crosswalk_path)
    diagnostics, summary = build_city_calibration_benchmark(
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=year,
        min_train_rows_per_offense=int(args.min_train_rows_per_offense),
        min_unique_prediction_values=int(args.min_unique_prediction_values),
    )
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(args.diagnostics_out, index=False)
    args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.diagnostics_out}")
    print(f"Wrote {args.summary_json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
