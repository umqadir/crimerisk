from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_share_benchmark import build_city_share_diagnostics, build_city_share_truth_model_frame, weighted_mean
from crimerisk.city_residuals import CityResidualConfig, apply_city_residual_model, fit_city_residual_model, prepare_city_residual_frame
from crimerisk.allocation import (
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
)
from crimerisk.paths import RepoPaths


def _group_residual_summary(
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
                "residual_weighted_total_variation_distance_mean": weighted_mean(
                    group, "residual_total_variation_distance", "incident_total"
                ),
                "weighted_tvd_delta": weighted_mean(group, "tvd_delta", "incident_total"),
                "baseline_weighted_pearson_share_mean": weighted_mean(
                    group, "baseline_pearson_share", "incident_total"
                ),
                "residual_weighted_pearson_share_mean": weighted_mean(
                    group, "residual_pearson_share", "incident_total"
                ),
                "baseline_weighted_spearman_share_mean": weighted_mean(
                    group, "baseline_spearman_share", "incident_total"
                ),
                "residual_weighted_spearman_share_mean": weighted_mean(
                    group, "residual_spearman_share", "incident_total"
                ),
                "baseline_weighted_top10_capture_mean": weighted_mean(
                    group, "baseline_top_10pct_true_mass_in_model_top_10pct", "incident_total"
                ),
                "residual_weighted_top10_capture_mean": weighted_mean(
                    group, "residual_top_10pct_true_mass_in_model_top_10pct", "incident_total"
                ),
                "weighted_fallback_incident_share_mean": weighted_mean(
                    group, "fallback_incident_share", "incident_total"
                ),
                "improved_tvd_rows": int((pd.to_numeric(group["tvd_delta"], errors="coerce") < 0).sum()),
                "worsened_tvd_rows": int((pd.to_numeric(group["tvd_delta"], errors="coerce") > 0).sum()),
            }
        )
    return rows


def build_city_residual_benchmark(
    *,
    paths: RepoPaths,
    city_shares: pd.DataFrame,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    year: int,
    hist_learning_rate: float,
    hist_max_depth: int,
    hist_max_iter: int,
    hist_min_samples_leaf: int,
    hist_l2_regularization: float,
    extra_feature_paths: list[Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = build_city_share_truth_model_frame(
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=year,
    )
    if merged.empty:
        empty = pd.DataFrame(
            columns=[
                "holdout_city_name",
                "city_name",
                "jurisdiction_id",
                "state_fips",
                "offense",
                "incident_total",
                "baseline_total_variation_distance",
                "residual_total_variation_distance",
                "baseline_share_rmse",
                "residual_share_rmse",
                "baseline_pearson_share",
                "residual_pearson_share",
                "baseline_spearman_share",
                "residual_spearman_share",
                "baseline_top_10pct_true_mass_in_model_top_10pct",
                "residual_top_10pct_true_mass_in_model_top_10pct",
                "tvd_delta",
                "share_rmse_delta",
                "top10_capture_delta",
            ]
        )
        return empty, {"year": int(year), "rows": 0}

    frame, feature_cols = prepare_city_residual_frame(
        paths=paths,
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=year,
        extra_feature_paths=extra_feature_paths,
        feature_policy_path=REPO_ROOT / DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
        exclude_feature_policy_classes=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
        exclude_feature_policy_classes_by_offense=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    )
    if frame.empty:
        return pd.DataFrame(), {"year": int(year), "rows": 0}
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

        fitted = fit_city_residual_model(
            train,
            feature_cols=feature_cols,
            config=CityResidualConfig(
                hist_learning_rate=float(hist_learning_rate),
                hist_max_depth=int(hist_max_depth),
                hist_max_iter=int(hist_max_iter),
                hist_min_samples_leaf=int(hist_min_samples_leaf),
                hist_l2_regularization=float(hist_l2_regularization),
                extra_feature_paths=tuple(extra_feature_paths or ()),
                feature_policy_path=REPO_ROOT / DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
                exclude_feature_policy_classes=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
                exclude_feature_policy_classes_by_offense=DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
            ),
        )
        if fitted is None:
            continue
        test = apply_city_residual_model(
            test,
            fitted=fitted,
        )

        baseline_diag, _ = build_city_share_diagnostics(test, predicted_share_col="model_share")
        residual_diag, _ = build_city_share_diagnostics(test, predicted_share_col="residual_model_share")
        comparison = baseline_diag.merge(
            residual_diag,
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
            suffixes=("_baseline", "_residual"),
        )
        if comparison.empty:
            continue
        comparison["holdout_city_name"] = holdout_city
        comparison["holdout_jurisdiction_id"] = holdout_jurisdiction_id
        comparison["baseline_total_variation_distance"] = pd.to_numeric(
            comparison["total_variation_distance_baseline"],
            errors="coerce",
        )
        comparison["residual_total_variation_distance"] = pd.to_numeric(
            comparison["total_variation_distance_residual"],
            errors="coerce",
        )
        comparison["baseline_share_rmse"] = pd.to_numeric(comparison["share_rmse_baseline"], errors="coerce")
        comparison["residual_share_rmse"] = pd.to_numeric(comparison["share_rmse_residual"], errors="coerce")
        comparison["baseline_pearson_share"] = pd.to_numeric(comparison["pearson_share_baseline"], errors="coerce")
        comparison["residual_pearson_share"] = pd.to_numeric(comparison["pearson_share_residual"], errors="coerce")
        comparison["baseline_spearman_share"] = pd.to_numeric(comparison["spearman_share_baseline"], errors="coerce")
        comparison["residual_spearman_share"] = pd.to_numeric(comparison["spearman_share_residual"], errors="coerce")
        comparison["baseline_top_10pct_true_mass_in_model_top_10pct"] = pd.to_numeric(
            comparison["top_10pct_true_mass_in_model_top_10pct_baseline"],
            errors="coerce",
        )
        comparison["residual_top_10pct_true_mass_in_model_top_10pct"] = pd.to_numeric(
            comparison["top_10pct_true_mass_in_model_top_10pct_residual"],
            errors="coerce",
        )
        comparison["tvd_delta"] = (
            comparison["residual_total_variation_distance"] - comparison["baseline_total_variation_distance"]
        )
        comparison["share_rmse_delta"] = comparison["residual_share_rmse"] - comparison["baseline_share_rmse"]
        comparison["top10_capture_delta"] = (
            comparison["residual_top_10pct_true_mass_in_model_top_10pct"]
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
                    "residual_total_variation_distance",
                    "baseline_share_rmse",
                    "residual_share_rmse",
                    "baseline_pearson_share",
                    "residual_pearson_share",
                    "baseline_spearman_share",
                    "residual_spearman_share",
                    "baseline_top_10pct_true_mass_in_model_top_10pct",
                    "residual_top_10pct_true_mass_in_model_top_10pct",
                    "tvd_delta",
                    "share_rmse_delta",
                    "top10_capture_delta",
                ]
            ].copy()
        )

    diagnostics = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if diagnostics.empty:
        return diagnostics, {"year": int(year), "rows": 0}
    if "validation_case_type" in city_shares.columns:
        case_meta = (
            city_shares[["jurisdiction_id", "validation_case_type"]]
            .dropna(subset=["jurisdiction_id"])
            .drop_duplicates("jurisdiction_id")
            .copy()
        )
        diagnostics = diagnostics.merge(case_meta, on="jurisdiction_id", how="left")

    diagnostics = diagnostics.sort_values(
        ["holdout_city_name", "holdout_jurisdiction_id", "city_name", "offense"],
        kind="mergesort",
    ).reset_index(drop=True)
    by_holdout_city: list[dict[str, object]] = []
    holdout_groups = diagnostics.groupby(
        ["holdout_city_name", "holdout_jurisdiction_id"],
        dropna=False,
        sort=True,
    )
    for (holdout_city_name, holdout_jurisdiction_id), group in holdout_groups:
        group = group.copy()
        by_holdout_city.append(
            {
                "holdout_city_name": str(holdout_city_name),
                "holdout_jurisdiction_id": str(holdout_jurisdiction_id),
                "rows": int(len(group)),
                "incident_total": float(pd.to_numeric(group["incident_total"], errors="coerce").fillna(0.0).sum()),
                "baseline_weighted_total_variation_distance_mean": weighted_mean(
                    group,
                    "baseline_total_variation_distance",
                    "incident_total",
                ),
                "residual_weighted_total_variation_distance_mean": weighted_mean(
                    group,
                    "residual_total_variation_distance",
                    "incident_total",
                ),
                "weighted_tvd_delta": weighted_mean(group, "tvd_delta", "incident_total"),
                "baseline_weighted_pearson_share_mean": weighted_mean(
                    group,
                    "baseline_pearson_share",
                    "incident_total",
                ),
                "residual_weighted_pearson_share_mean": weighted_mean(
                    group,
                    "residual_pearson_share",
                    "incident_total",
                ),
                "baseline_weighted_spearman_share_mean": weighted_mean(
                    group,
                    "baseline_spearman_share",
                    "incident_total",
                ),
                "residual_weighted_spearman_share_mean": weighted_mean(
                    group,
                    "residual_spearman_share",
                    "incident_total",
                ),
                "baseline_weighted_top10_capture_mean": weighted_mean(
                    group,
                    "baseline_top_10pct_true_mass_in_model_top_10pct",
                    "incident_total",
                ),
                "residual_weighted_top10_capture_mean": weighted_mean(
                    group,
                    "residual_top_10pct_true_mass_in_model_top_10pct",
                    "incident_total",
                ),
                "weighted_top10_capture_delta": weighted_mean(group, "top10_capture_delta", "incident_total"),
                "improved_tvd_rows": int((pd.to_numeric(group["tvd_delta"], errors="coerce") < 0).sum()),
                "worsened_tvd_rows": int((pd.to_numeric(group["tvd_delta"], errors="coerce") > 0).sum()),
            }
        )
    summary = {
        "year": int(year),
        "rows": int(len(diagnostics)),
        "split_mode": "leave_one_city_out",
        "holdout_unit": "jurisdiction_id",
        "holdout_city_count": int(diagnostics["holdout_jurisdiction_id"].nunique()),
        "train_city_count_per_fold": int(max(diagnostics["holdout_jurisdiction_id"].nunique() - 1, 0)),
        "test_city_count_per_fold": 1,
        "incident_total": float(pd.to_numeric(diagnostics["incident_total"], errors="coerce").fillna(0.0).sum()),
        "extra_bg_feature_paths": [str(path) for path in (extra_feature_paths or [])],
        "extra_bg_feature_count": int(len(extra_feature_paths or [])),
        "feature_column_count": int(len(feature_cols)),
        "baseline_weighted_total_variation_distance_mean": weighted_mean(
            diagnostics,
            "baseline_total_variation_distance",
            "incident_total",
        ),
        "residual_weighted_total_variation_distance_mean": weighted_mean(
            diagnostics,
            "residual_total_variation_distance",
            "incident_total",
        ),
        "baseline_weighted_share_rmse_mean": weighted_mean(diagnostics, "baseline_share_rmse", "incident_total"),
        "residual_weighted_share_rmse_mean": weighted_mean(diagnostics, "residual_share_rmse", "incident_total"),
        "baseline_weighted_pearson_share_mean": weighted_mean(
            diagnostics,
            "baseline_pearson_share",
            "incident_total",
        ),
        "residual_weighted_pearson_share_mean": weighted_mean(
            diagnostics,
            "residual_pearson_share",
            "incident_total",
        ),
        "baseline_weighted_spearman_share_mean": weighted_mean(
            diagnostics,
            "baseline_spearman_share",
            "incident_total",
        ),
        "residual_weighted_spearman_share_mean": weighted_mean(
            diagnostics,
            "residual_spearman_share",
            "incident_total",
        ),
        "baseline_weighted_top10_capture_mean": weighted_mean(
            diagnostics,
            "baseline_top_10pct_true_mass_in_model_top_10pct",
            "incident_total",
        ),
        "residual_weighted_top10_capture_mean": weighted_mean(
            diagnostics,
            "residual_top_10pct_true_mass_in_model_top_10pct",
            "incident_total",
        ),
        "weighted_tvd_delta": weighted_mean(diagnostics, "tvd_delta", "incident_total"),
        "weighted_share_rmse_delta": weighted_mean(diagnostics, "share_rmse_delta", "incident_total"),
        "weighted_top10_capture_delta": weighted_mean(diagnostics, "top10_capture_delta", "incident_total"),
        "improved_tvd_rows": int((pd.to_numeric(diagnostics["tvd_delta"], errors="coerce") < 0).sum()),
        "worsened_tvd_rows": int((pd.to_numeric(diagnostics["tvd_delta"], errors="coerce") > 0).sum()),
        "by_holdout_city": by_holdout_city,
        "by_primary_geocode_quality_tier": _group_residual_summary(
            diagnostics,
            group_col="primary_geocode_quality_tier",
        ),
        "by_offense_volume_band": _group_residual_summary(
            diagnostics,
            group_col="offense_volume_band",
        ),
    }
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
    parser.add_argument("--hist-learning-rate", type=float, default=0.03)
    parser.add_argument("--hist-max-depth", type=int, default=5)
    parser.add_argument("--hist-max-iter", type=int, default=500)
    parser.add_argument("--hist-min-samples-leaf", type=int, default=20)
    parser.add_argument("--hist-l2-regularization", type=float, default=1.0)
    parser.add_argument(
        "--extra-bg-features-path",
        type=Path,
        action="append",
        default=[],
        help="Optional BG feature parquet to merge into the residual benchmark frame before feature selection.",
    )
    parser.add_argument(
        "--exclude-validation-case-type",
        action="append",
        default=[],
        help=(
            "Optional validation_case_type value to exclude from the city residual holdout benchmark. "
            "May be repeated; useful for keeping county-only allocation checks out of city conclusions."
        ),
    )
    parser.add_argument(
        "--allow-canonical-overwrite",
        action="store_true",
        help="Allow non-default residual benchmark configurations to overwrite canonical residual benchmark artifacts.",
    )
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary-csv-out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    year = int(args.year)
    canonical_city_shares_path = REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet"
    canonical_bg_prior_path = REPO_ROOT / "state" / "modeling" / f"bg_prior_long_{year}.parquet"
    canonical_bg_crosswalk_path = REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    canonical_diagnostics_out = REPO_ROOT / "state" / "modeling" / f"city_residual_benchmark_{year}.parquet"
    canonical_summary_json_out = REPO_ROOT / "state" / "modeling" / f"city_residual_benchmark_{year}.json"
    if args.diagnostics_out is None:
        args.diagnostics_out = canonical_diagnostics_out
    if args.summary_json_out is None:
        args.summary_json_out = canonical_summary_json_out

    experimental_config = any(
        [
            float(args.hist_learning_rate) != 0.03,
            int(args.hist_max_depth) != 5,
            int(args.hist_max_iter) != 500,
            int(args.hist_min_samples_leaf) != 20,
            float(args.hist_l2_regularization) != 1.0,
            bool(args.extra_bg_features_path),
            bool(args.exclude_validation_case_type),
            Path(args.city_shares_path).resolve() != canonical_city_shares_path.resolve(),
            Path(args.bg_prior_path).resolve() != canonical_bg_prior_path.resolve(),
            Path(args.bg_crosswalk_path).resolve() != canonical_bg_crosswalk_path.resolve(),
        ]
    )
    canonical_targets = {
        canonical_diagnostics_out.resolve(),
        canonical_summary_json_out.resolve(),
    }
    requested_targets = {
        Path(args.diagnostics_out).resolve(),
        Path(args.summary_json_out).resolve(),
    }
    if experimental_config and not bool(args.allow_canonical_overwrite):
        overlapping = sorted(str(path) for path in requested_targets & canonical_targets)
        if overlapping:
            raise SystemExit(
                "Refusing to overwrite canonical residual benchmark artifacts with a non-default "
                "benchmark configuration. Pass explicit scratch output paths or use "
                "--allow-canonical-overwrite if you are intentionally promoting the experiment. "
                f"Overlapping targets: {overlapping}"
            )

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    city_shares = pd.read_parquet(args.city_shares_path)
    excluded_case_types = {str(value) for value in args.exclude_validation_case_type if str(value).strip()}
    if excluded_case_types:
        if "validation_case_type" not in city_shares.columns:
            raise SystemExit(
                "--exclude-validation-case-type was supplied, but the city shares file has no "
                "validation_case_type column."
            )
        city_shares = city_shares[
            ~city_shares["validation_case_type"].astype("string").fillna("").isin(excluded_case_types)
        ].copy()
    bg_prior = pd.read_parquet(args.bg_prior_path)
    bg_crosswalk = pd.read_parquet(args.bg_crosswalk_path)
    diagnostics, summary = build_city_residual_benchmark(
        paths=paths,
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=year,
        hist_learning_rate=float(args.hist_learning_rate),
        hist_max_depth=int(args.hist_max_depth),
        hist_max_iter=int(args.hist_max_iter),
        hist_min_samples_leaf=int(args.hist_min_samples_leaf),
        hist_l2_regularization=float(args.hist_l2_regularization),
        extra_feature_paths=list(args.extra_bg_features_path),
    )
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(args.diagnostics_out, index=False)
    args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_csv_out is not None:
        args.summary_csv_out.parent.mkdir(parents=True, exist_ok=True)
        summary_rows = [
            {
                "summary_level": "overall",
                "group": "all",
                "rows": summary.get("rows"),
                "incident_total": summary.get("incident_total"),
                "baseline_weighted_total_variation_distance_mean": summary.get(
                    "baseline_weighted_total_variation_distance_mean"
                ),
                "residual_weighted_total_variation_distance_mean": summary.get(
                    "residual_weighted_total_variation_distance_mean"
                ),
                "weighted_tvd_delta": summary.get("weighted_tvd_delta"),
                "improved_tvd_rows": summary.get("improved_tvd_rows"),
                "worsened_tvd_rows": summary.get("worsened_tvd_rows"),
            }
        ]
        for entry in summary.get("by_holdout_city", []):
            row = dict(entry)
            row["summary_level"] = "holdout_city"
            row["group"] = row.get("holdout_city_name")
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(args.summary_csv_out, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
