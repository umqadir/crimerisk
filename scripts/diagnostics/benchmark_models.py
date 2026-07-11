from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_share_benchmark import build_city_share_diagnostics, build_city_share_truth_model_frame
from crimerisk.crime import OFFENSES_7
from crimerisk.crosswalk_shares import normalize_block_group_allocation_shares
from crimerisk.paths import RepoPaths
from crimerisk.model_surface import MODEL_FEATURE_GROUPS, ModelSurfaceConfig, build_model_surface, build_model_workload_plan


def _append_progress_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _training_row_policy_name(*, exclude_estimated: bool, high_confidence_only: bool) -> str:
    if high_confidence_only:
        return "high_confidence_only"
    if exclude_estimated:
        return "observed_only"
    return "include_estimated"


def _load_city_share_benchmark_inputs(
    *,
    year: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, object]]:
    city_share_path = REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet"
    bg_crosswalk_path = REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    metadata: dict[str, object] = {
        "city_share_surface_path": str(city_share_path),
        "bg_crosswalk_path": str(bg_crosswalk_path),
        "city_share_surface_exists": bool(city_share_path.exists()),
        "bg_crosswalk_exists": bool(bg_crosswalk_path.exists()),
        "snapshot_year": int(year),
    }
    if not city_share_path.exists() or not bg_crosswalk_path.exists():
        return None, None, metadata
    metadata["city_share_surface_mtime"] = float(city_share_path.stat().st_mtime)
    metadata["bg_crosswalk_mtime"] = float(bg_crosswalk_path.stat().st_mtime)
    city_shares = pd.read_parquet(city_share_path)
    bg_crosswalk = normalize_block_group_allocation_shares(pd.read_parquet(bg_crosswalk_path))
    metadata["city_share_surface_row_count"] = int(len(city_shares))
    metadata["bg_crosswalk_row_count"] = int(len(bg_crosswalk))
    return city_shares, bg_crosswalk, metadata


def _city_share_summary(
    *,
    bg_prior: pd.DataFrame,
    year: int,
    city_shares: pd.DataFrame | None,
    bg_crosswalk: pd.DataFrame | None,
) -> dict[str, float | int | None]:
    if city_shares is None or bg_crosswalk is None:
        return {
            "city_share_rows": 0,
            "city_share_city_count": 0,
            "city_share_weighted_total_variation_distance_mean": None,
            "city_share_weighted_pearson_share_mean": None,
            "city_share_weighted_spearman_share_mean": None,
        }
    merged = build_city_share_truth_model_frame(
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=int(year),
    )
    diagnostics, summary = build_city_share_diagnostics(merged, predicted_share_col="model_share")
    return {
        "city_share_rows": int(summary.get("rows", 0) or 0),
        "city_share_city_count": int(summary.get("city_count", 0) or 0),
        "city_share_weighted_total_variation_distance_mean": summary.get("weighted_total_variation_distance_mean"),
        "city_share_weighted_pearson_share_mean": summary.get("weighted_pearson_share_mean"),
        "city_share_weighted_spearman_share_mean": summary.get("weighted_spearman_share_mean"),
    }


def _summarize_variant(
    *,
    diagnostics: pd.DataFrame,
    config: ModelSurfaceConfig,
    variant_name: str,
    city_share_summary: dict[str, float | int | None] | None = None,
) -> dict[str, object]:
    property_offenses = {"burglary", "larceny", "motor_vehicle_theft"}
    personal_offenses = set(OFFENSES_7) - property_offenses
    row: dict[str, object] = {
        "variant_name": str(variant_name),
        "model_family": str(config.model_family),
        "training_row_policy": _training_row_policy_name(
            exclude_estimated=bool(config.exclude_estimated_from_panel_from_training),
            high_confidence_only=bool(config.high_confidence_training_only),
        ),
        "state_fixed_effects": bool(config.use_state_fixed_effects),
        "sparse_offense_pooling_strategy": str(config.sparse_offense_pooling_strategy),
        "exclude_feature_groups": list(config.exclude_feature_groups),
        "overall_train_r2_log_rate_mean": float(diagnostics["train_r2_log_rate"].mean()),
        "overall_cv_r2_log_rate_mean": float(diagnostics["cv_r2_log_rate"].mean()),
        "overall_cv_r2_rate_mean": float(diagnostics["cv_r2_rate"].mean()),
        "overall_cv_rmse_log_rate_mean": float(diagnostics["cv_rmse_log_rate"].mean()),
        "overall_leave_state_out_cv_r2_log_rate_mean": float(diagnostics["leave_state_out_cv_r2_log_rate"].mean()),
        "overall_leave_state_out_cv_r2_rate_mean": float(diagnostics["leave_state_out_cv_r2_rate"].mean()),
        "overall_leave_large_city_out_cv_r2_log_rate_mean": float(diagnostics["leave_large_city_out_cv_r2_log_rate"].mean()),
        "overall_leave_large_city_out_cv_r2_rate_mean": float(diagnostics["leave_large_city_out_cv_r2_rate"].mean()),
        "overall_leave_cbsa_out_cv_r2_log_rate_mean": float(diagnostics["leave_cbsa_out_cv_r2_log_rate"].mean()),
        "overall_leave_cbsa_out_cv_r2_rate_mean": float(diagnostics["leave_cbsa_out_cv_r2_rate"].mean()),
        "property_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "cv_r2_log_rate"].mean()),
        "property_leave_state_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_state_out_cv_r2_log_rate"].mean()),
        "property_leave_large_city_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_large_city_out_cv_r2_log_rate"].mean()),
        "property_leave_cbsa_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_cbsa_out_cv_r2_log_rate"].mean()),
        "personal_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "cv_r2_log_rate"].mean()),
        "personal_leave_state_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_state_out_cv_r2_log_rate"].mean()),
        "personal_leave_large_city_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_large_city_out_cv_r2_log_rate"].mean()),
        "personal_leave_cbsa_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_cbsa_out_cv_r2_log_rate"].mean()),
    }
    if city_share_summary is not None:
        row.update(city_share_summary)
    return row


def main() -> int:
    default_config = ModelSurfaceConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--min-training-population", type=int, default=int(default_config.min_training_population))
    parser.add_argument("--model-family", choices=["hist_gbm", "ridge", "elastic_net", "monotone_gam"], default=str(default_config.model_family))
    parser.add_argument("--ridge-alpha", type=float, default=float(default_config.ridge_alpha))
    parser.add_argument("--elastic-net-alpha", type=float, default=float(default_config.elastic_net_alpha))
    parser.add_argument("--elastic-net-l1-ratio", type=float, default=float(default_config.elastic_net_l1_ratio))
    parser.add_argument("--elastic-net-max-iter", type=int, default=int(default_config.elastic_net_max_iter))
    parser.add_argument("--monotone-gam-lam", type=float, default=float(default_config.monotone_gam_lam))
    parser.add_argument("--monotone-gam-n-splines", type=int, default=int(default_config.monotone_gam_n_splines))
    parser.add_argument("--monotone-gam-max-iter", type=int, default=int(default_config.monotone_gam_max_iter))
    parser.add_argument("--hist-learning-rate", type=float, default=float(default_config.hist_learning_rate))
    parser.add_argument("--hist-max-depth", type=int, default=int(default_config.hist_max_depth))
    parser.add_argument("--hist-max-iter", type=int, default=int(default_config.hist_max_iter))
    parser.add_argument("--hist-min-samples-leaf", type=int, default=int(default_config.hist_min_samples_leaf))
    parser.add_argument("--hist-l2-regularization", type=float, default=float(default_config.hist_l2_regularization))
    parser.add_argument(
        "--disable-state-fixed-effects",
        action="store_true",
        help="Benchmark without state fixed effects.",
    )
    parser.add_argument(
        "--high-confidence-training-only",
        action="store_true",
        help="Restrict training rows to the highest-confidence observed controls.",
    )
    parser.add_argument(
        "--exclude-feature-group",
        action="append",
        default=[],
        choices=list(MODEL_FEATURE_GROUPS),
        help="Exclude a named feature group from the benchmark feature stack.",
    )
    parser.add_argument(
        "--include-estimated-training-rows",
        action="store_true",
        help="Include retained panel-estimated control rows in the training sample.",
    )
    parser.add_argument(
        "--sparse-offense-pooling-strategy",
        choices=["none", "empirical_bayes_group_rate"],
        default=str(default_config.sparse_offense_pooling_strategy),
        help="Optional benchmark-only sparse-offense target pooling strategy.",
    )
    parser.add_argument(
        "--extra-bg-features-path",
        type=Path,
        action="append",
        default=[],
        help="Optional BG feature parquet to merge into the jurisdiction benchmark frame before feature selection.",
    )
    parser.add_argument(
        "--allow-canonical-overwrite",
        action="store_true",
        help="Allow non-default benchmark configurations to overwrite the canonical benchmark artifacts.",
    )
    parser.add_argument("--diagnostics-out", type=Path, default=None)
    parser.add_argument("--meta-out", type=Path, default=None)
    parser.add_argument("--summary-json-out", type=Path, default=None)
    parser.add_argument("--family-ladder-out", type=Path, default=None)
    parser.add_argument("--family-ladder-json-out", type=Path, default=None)
    parser.add_argument("--provenance-sensitivity-out", type=Path, default=None)
    parser.add_argument("--provenance-sensitivity-json-out", type=Path, default=None)
    parser.add_argument("--feature-ablation-out", type=Path, default=None)
    parser.add_argument("--feature-ablation-json-out", type=Path, default=None)
    parser.add_argument("--sparse-pooling-out", type=Path, default=None)
    parser.add_argument("--sparse-pooling-json-out", type=Path, default=None)
    parser.add_argument("--transit-sensitivity-out", type=Path, default=None)
    parser.add_argument("--transit-sensitivity-json-out", type=Path, default=None)
    parser.add_argument(
        "--transit-features-path",
        type=Path,
        default=None,
        help="Optional block-group transit feature parquet for the transit sensitivity benchmark lane.",
    )
    parser.add_argument(
        "--latest-log-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "benchmark_latest.log",
    )
    args = parser.parse_args()
    year = int(args.year)
    canonical_diagnostics_out = REPO_ROOT / "state" / "modeling" / f"jurisdiction_model_benchmark_{year}.parquet"
    canonical_meta_out = REPO_ROOT / "state" / "modeling" / f"jurisdiction_model_features_{year}.parquet"
    canonical_summary_json_out = REPO_ROOT / "state" / "modeling" / f"jurisdiction_model_benchmark_{year}.json"
    canonical_latest_log_out = REPO_ROOT / "state" / "modeling" / "benchmark_latest.log"
    if args.diagnostics_out is None:
        args.diagnostics_out = canonical_diagnostics_out
    if args.meta_out is None:
        args.meta_out = canonical_meta_out
    if args.summary_json_out is None:
        args.summary_json_out = canonical_summary_json_out

    experimental_config = any(
        [
            int(args.min_training_population) != int(default_config.min_training_population),
            str(args.model_family) != str(default_config.model_family),
            float(args.ridge_alpha) != float(default_config.ridge_alpha),
            float(args.elastic_net_alpha) != float(default_config.elastic_net_alpha),
            float(args.elastic_net_l1_ratio) != float(default_config.elastic_net_l1_ratio),
            int(args.elastic_net_max_iter) != int(default_config.elastic_net_max_iter),
            float(args.monotone_gam_lam) != float(default_config.monotone_gam_lam),
            int(args.monotone_gam_n_splines) != int(default_config.monotone_gam_n_splines),
            int(args.monotone_gam_max_iter) != int(default_config.monotone_gam_max_iter),
            float(args.hist_learning_rate) != float(default_config.hist_learning_rate),
            int(args.hist_max_depth) != int(default_config.hist_max_depth),
            int(args.hist_max_iter) != int(default_config.hist_max_iter),
            int(args.hist_min_samples_leaf) != int(default_config.hist_min_samples_leaf),
            float(args.hist_l2_regularization) != float(default_config.hist_l2_regularization),
            bool(args.disable_state_fixed_effects),
            bool(args.high_confidence_training_only),
            bool(args.include_estimated_training_rows),
            str(args.sparse_offense_pooling_strategy) != str(default_config.sparse_offense_pooling_strategy),
            bool(args.exclude_feature_group),
            bool(args.extra_bg_features_path),
        ]
    )
    canonical_targets = {
        canonical_diagnostics_out.resolve(),
        canonical_meta_out.resolve(),
        canonical_summary_json_out.resolve(),
        canonical_latest_log_out.resolve(),
    }
    requested_targets = {
        Path(args.diagnostics_out).resolve(),
        Path(args.meta_out).resolve(),
        Path(args.summary_json_out).resolve(),
        Path(args.latest_log_out).resolve(),
    }
    if experimental_config and not bool(args.allow_canonical_overwrite):
        overlapping = sorted(str(path) for path in requested_targets & canonical_targets)
        if overlapping:
            raise SystemExit(
                "Refusing to overwrite canonical benchmark artifacts with a non-default benchmark "
                "configuration. Pass explicit scratch output paths or use --allow-canonical-overwrite "
                f"if you are intentionally promoting the experiment. Overlapping targets: {overlapping}"
            )

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    frozen_city_shares, frozen_bg_crosswalk, city_share_snapshot = _load_city_share_benchmark_inputs(
        year=year,
    )
    base_config_kwargs = {
        "year": year,
        "min_training_population": int(args.min_training_population),
        "model_family": str(args.model_family),
        "ridge_alpha": float(args.ridge_alpha),
        "elastic_net_alpha": float(args.elastic_net_alpha),
        "elastic_net_l1_ratio": float(args.elastic_net_l1_ratio),
        "elastic_net_max_iter": int(args.elastic_net_max_iter),
        "monotone_gam_lam": float(args.monotone_gam_lam),
        "monotone_gam_n_splines": int(args.monotone_gam_n_splines),
        "monotone_gam_max_iter": int(args.monotone_gam_max_iter),
        "hist_learning_rate": float(args.hist_learning_rate),
        "hist_max_depth": int(args.hist_max_depth),
        "hist_max_iter": int(args.hist_max_iter),
        "hist_min_samples_leaf": int(args.hist_min_samples_leaf),
        "hist_l2_regularization": float(args.hist_l2_regularization),
        "use_state_fixed_effects": not bool(args.disable_state_fixed_effects),
        "exclude_estimated_from_panel_from_training": not bool(args.include_estimated_training_rows),
        "high_confidence_training_only": bool(args.high_confidence_training_only),
        "sparse_offense_pooling_strategy": str(args.sparse_offense_pooling_strategy),
        "exclude_feature_groups": tuple(sorted(set(args.exclude_feature_group))),
    }
    base_config = ModelSurfaceConfig(**base_config_kwargs)
    if args.latest_log_out.exists():
        args.latest_log_out.unlink()
    base_plan = build_model_workload_plan(
        paths=paths,
        config=base_config,
        extra_bg_feature_paths=list(args.extra_bg_features_path),
    )
    _append_progress_log(
        args.latest_log_out,
        {
            "event": "benchmark_plan",
            "variant_name": "base_config",
            "training_row_policy": _training_row_policy_name(
                exclude_estimated=bool(base_config.exclude_estimated_from_panel_from_training),
                high_confidence_only=bool(base_config.high_confidence_training_only),
            ),
            "state_fixed_effects": bool(base_config.use_state_fixed_effects),
            "sparse_offense_pooling_strategy": str(base_config.sparse_offense_pooling_strategy),
            "exclude_feature_groups": sorted({str(group) for group in base_config.exclude_feature_groups}),
            "extra_bg_feature_paths": [str(path) for path in args.extra_bg_features_path],
            "planned_fit_total": int(pd.to_numeric(base_plan["planned_fit_count"], errors="coerce").fillna(0).sum()),
            "workload_plan": base_plan.to_dict(orient="records"),
        },
    )
    base_started_at = time.perf_counter()
    base_completed_fit_count = 0

    def _base_progress_callback(payload: dict[str, object]) -> None:
        nonlocal base_completed_fit_count
        event = dict(payload)
        event["variant_name"] = "base_config"
        event["elapsed_sec_total"] = float(time.perf_counter() - base_started_at)
        if event.get("event") == "offense_complete":
            base_completed_fit_count += int(event.get("planned_fit_count", 0) or 0)
            event["completed_fit_count"] = int(base_completed_fit_count)
            event["planned_fit_total"] = int(pd.to_numeric(base_plan["planned_fit_count"], errors="coerce").fillna(0).sum())
            event["remaining_fit_count"] = int(event["planned_fit_total"]) - int(base_completed_fit_count)
        _append_progress_log(args.latest_log_out, event)

    bg_prior, diagnostics, model_meta = build_model_surface(
        paths=paths,
        config=base_config,
        extra_bg_feature_paths=list(args.extra_bg_features_path),
        progress_callback=_base_progress_callback,
    )

    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(args.diagnostics_out, index=False)
    model_meta.to_parquet(args.meta_out, index=False)

    city_share_summary = _city_share_summary(
        bg_prior=bg_prior,
        year=year,
        city_shares=frozen_city_shares,
        bg_crosswalk=frozen_bg_crosswalk,
    )
    property_offenses = {"burglary", "larceny", "motor_vehicle_theft"}
    personal_offenses = set(OFFENSES_7) - property_offenses
    summary = {
        "year": year,
        "min_training_population": int(args.min_training_population),
        "model_family": str(diagnostics["model_family"].iloc[0]) if "model_family" in diagnostics.columns and len(diagnostics) else "unknown",
        "training_row_policy": _training_row_policy_name(
            exclude_estimated=not bool(args.include_estimated_training_rows),
            high_confidence_only=bool(args.high_confidence_training_only),
        ),
        "exclude_estimated_from_panel_from_training": not bool(args.include_estimated_training_rows),
        "high_confidence_training_only": bool(args.high_confidence_training_only),
        "sparse_offense_pooling_strategy": str(args.sparse_offense_pooling_strategy),
        "state_fixed_effects": not bool(args.disable_state_fixed_effects),
        "exclude_feature_groups": sorted(set(args.exclude_feature_group)),
        "extra_bg_feature_paths": [str(path) for path in args.extra_bg_features_path],
        "extra_bg_feature_count": int(len(args.extra_bg_features_path)),
        "city_share_benchmark_input_snapshot": city_share_snapshot,
        **city_share_summary,
        "overall_train_r2_log_rate_mean": float(diagnostics["train_r2_log_rate"].mean()),
        "overall_cv_r2_log_rate_mean": float(diagnostics["cv_r2_log_rate"].mean()),
        "overall_cv_r2_rate_mean": float(diagnostics["cv_r2_rate"].mean()),
        "overall_cv_rmse_log_rate_mean": float(diagnostics["cv_rmse_log_rate"].mean()),
        "overall_leave_state_out_cv_r2_log_rate_mean": float(diagnostics["leave_state_out_cv_r2_log_rate"].mean()),
        "overall_leave_state_out_cv_r2_rate_mean": float(diagnostics["leave_state_out_cv_r2_rate"].mean()),
        "overall_leave_state_out_cv_rmse_log_rate_mean": float(diagnostics["leave_state_out_cv_rmse_log_rate"].mean()),
        "overall_leave_large_city_out_cv_r2_log_rate_mean": float(diagnostics["leave_large_city_out_cv_r2_log_rate"].mean()),
        "overall_leave_large_city_out_cv_r2_rate_mean": float(diagnostics["leave_large_city_out_cv_r2_rate"].mean()),
        "overall_leave_large_city_out_cv_rmse_log_rate_mean": float(diagnostics["leave_large_city_out_cv_rmse_log_rate"].mean()),
        "overall_leave_cbsa_out_cv_r2_log_rate_mean": float(diagnostics["leave_cbsa_out_cv_r2_log_rate"].mean()),
        "overall_leave_cbsa_out_cv_r2_rate_mean": float(diagnostics["leave_cbsa_out_cv_r2_rate"].mean()),
        "overall_leave_cbsa_out_cv_rmse_log_rate_mean": float(diagnostics["leave_cbsa_out_cv_rmse_log_rate"].mean()),
        "property_train_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "train_r2_log_rate"].mean()),
        "property_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "cv_r2_log_rate"].mean()),
        "property_cv_r2_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "cv_r2_rate"].mean()),
        "property_leave_state_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_state_out_cv_r2_log_rate"].mean()),
        "property_leave_state_out_cv_r2_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_state_out_cv_r2_rate"].mean()),
        "property_leave_large_city_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_large_city_out_cv_r2_log_rate"].mean()),
        "property_leave_large_city_out_cv_r2_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_large_city_out_cv_r2_rate"].mean()),
        "property_leave_cbsa_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_cbsa_out_cv_r2_log_rate"].mean()),
        "property_leave_cbsa_out_cv_r2_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(property_offenses), "leave_cbsa_out_cv_r2_rate"].mean()),
        "personal_train_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "train_r2_log_rate"].mean()),
        "personal_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "cv_r2_log_rate"].mean()),
        "personal_cv_r2_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "cv_r2_rate"].mean()),
        "personal_leave_state_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_state_out_cv_r2_log_rate"].mean()),
        "personal_leave_state_out_cv_r2_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_state_out_cv_r2_rate"].mean()),
        "personal_leave_large_city_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_large_city_out_cv_r2_log_rate"].mean()),
        "personal_leave_large_city_out_cv_r2_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_large_city_out_cv_r2_rate"].mean()),
        "personal_leave_cbsa_out_cv_r2_log_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_cbsa_out_cv_r2_log_rate"].mean()),
        "personal_leave_cbsa_out_cv_r2_rate_mean": float(diagnostics.loc[diagnostics["offense"].isin(personal_offenses), "leave_cbsa_out_cv_r2_rate"].mean()),
        "per_offense": {
            str(row.offense): {
                "train_r2_log_rate": float(row.train_r2_log_rate),
                "cv_r2_rate": float(row.cv_r2_rate),
                "cv_r2_log_rate": float(row.cv_r2_log_rate),
                "cv_rmse_log_rate": float(row.cv_rmse_log_rate),
                "leave_state_out_cv_r2_rate": float(row.leave_state_out_cv_r2_rate),
                "leave_state_out_cv_r2_log_rate": float(row.leave_state_out_cv_r2_log_rate),
                "leave_state_out_cv_rmse_log_rate": float(row.leave_state_out_cv_rmse_log_rate),
                "leave_large_city_out_cv_r2_rate": float(row.leave_large_city_out_cv_r2_rate),
                "leave_large_city_out_cv_r2_log_rate": float(row.leave_large_city_out_cv_r2_log_rate),
                "leave_large_city_out_cv_rmse_log_rate": float(row.leave_large_city_out_cv_rmse_log_rate),
                "leave_large_city_out_holdout_group_count": int(row.leave_large_city_out_holdout_group_count),
                "leave_cbsa_out_cv_r2_rate": float(row.leave_cbsa_out_cv_r2_rate),
                "leave_cbsa_out_cv_r2_log_rate": float(row.leave_cbsa_out_cv_r2_log_rate),
                "leave_cbsa_out_cv_rmse_log_rate": float(row.leave_cbsa_out_cv_rmse_log_rate),
                "leave_cbsa_out_holdout_group_count": int(row.leave_cbsa_out_holdout_group_count),
                "training_rows": int(row.training_rows),
                "calibration": (
                    json.loads(row.calibration_json)
                    if hasattr(row, "calibration_json") and pd.notna(row.calibration_json)
                    else []
                ),
                "leave_state_out_calibration": (
                    json.loads(row.leave_state_out_calibration_json)
                    if hasattr(row, "leave_state_out_calibration_json") and pd.notna(row.leave_state_out_calibration_json)
                    else []
                ),
                "residual_by_state": (
                    json.loads(row.residual_by_state_json)
                    if hasattr(row, "residual_by_state_json") and pd.notna(row.residual_by_state_json)
                    else []
                ),
                "leave_state_out_residual_by_state": (
                    json.loads(row.leave_state_out_residual_by_state_json)
                    if hasattr(row, "leave_state_out_residual_by_state_json") and pd.notna(row.leave_state_out_residual_by_state_json)
                    else []
                ),
                "residual_by_pop_band": (
                    json.loads(row.residual_by_pop_band_json)
                    if hasattr(row, "residual_by_pop_band_json") and pd.notna(row.residual_by_pop_band_json)
                    else []
                ),
                "leave_state_out_residual_by_pop_band": (
                    json.loads(row.leave_state_out_residual_by_pop_band_json)
                    if hasattr(row, "leave_state_out_residual_by_pop_band_json") and pd.notna(row.leave_state_out_residual_by_pop_band_json)
                    else []
                ),
                "leave_large_city_out_calibration": (
                    json.loads(row.leave_large_city_out_calibration_json)
                    if hasattr(row, "leave_large_city_out_calibration_json") and pd.notna(row.leave_large_city_out_calibration_json)
                    else []
                ),
                "leave_large_city_out_residual_by_state": (
                    json.loads(row.leave_large_city_out_residual_by_state_json)
                    if hasattr(row, "leave_large_city_out_residual_by_state_json") and pd.notna(row.leave_large_city_out_residual_by_state_json)
                    else []
                ),
                "leave_large_city_out_residual_by_pop_band": (
                    json.loads(row.leave_large_city_out_residual_by_pop_band_json)
                    if hasattr(row, "leave_large_city_out_residual_by_pop_band_json") and pd.notna(row.leave_large_city_out_residual_by_pop_band_json)
                    else []
                ),
                "leave_cbsa_out_calibration": (
                    json.loads(row.leave_cbsa_out_calibration_json)
                    if hasattr(row, "leave_cbsa_out_calibration_json") and pd.notna(row.leave_cbsa_out_calibration_json)
                    else []
                ),
                "leave_cbsa_out_residual_by_state": (
                    json.loads(row.leave_cbsa_out_residual_by_state_json)
                    if hasattr(row, "leave_cbsa_out_residual_by_state_json") and pd.notna(row.leave_cbsa_out_residual_by_state_json)
                    else []
                ),
                "leave_cbsa_out_residual_by_pop_band": (
                    json.loads(row.leave_cbsa_out_residual_by_pop_band_json)
                    if hasattr(row, "leave_cbsa_out_residual_by_pop_band_json") and pd.notna(row.leave_cbsa_out_residual_by_pop_band_json)
                    else []
                ),
            }
            for row in diagnostics.itertuples(index=False)
        },
    }

    def _run_variant(variant_name: str, **overrides: object) -> dict[str, object]:
        print(f"[benchmark_models] running variant: {variant_name}", file=sys.stderr, flush=True)
        variant_extra_feature_paths = overrides.pop("extra_bg_feature_paths", list(args.extra_bg_features_path))
        variant_kwargs = dict(base_config_kwargs)
        variant_kwargs.update(overrides)
        variant_config = ModelSurfaceConfig(**variant_kwargs)
        variant_plan = build_model_workload_plan(
            paths=paths,
            config=variant_config,
            extra_bg_feature_paths=list(variant_extra_feature_paths),
        )
        planned_fit_total = int(pd.to_numeric(variant_plan["planned_fit_count"], errors="coerce").fillna(0).sum())
        _append_progress_log(
            args.latest_log_out,
            {
                "event": "benchmark_plan",
                "variant_name": str(variant_name),
                "training_row_policy": _training_row_policy_name(
                    exclude_estimated=bool(variant_config.exclude_estimated_from_panel_from_training),
                    high_confidence_only=bool(variant_config.high_confidence_training_only),
                ),
                "state_fixed_effects": bool(variant_config.use_state_fixed_effects),
                "sparse_offense_pooling_strategy": str(variant_config.sparse_offense_pooling_strategy),
                "exclude_feature_groups": sorted({str(group) for group in variant_config.exclude_feature_groups}),
                "extra_bg_feature_paths": [str(path) for path in variant_extra_feature_paths],
                "planned_fit_total": planned_fit_total,
                "workload_plan": variant_plan.to_dict(orient="records"),
            },
        )
        variant_started_at = time.perf_counter()
        variant_completed_fit_count = 0

        def _variant_progress_callback(payload: dict[str, object]) -> None:
            nonlocal variant_completed_fit_count
            event = dict(payload)
            event["variant_name"] = str(variant_name)
            event["elapsed_sec_total"] = float(time.perf_counter() - variant_started_at)
            if event.get("event") == "offense_complete":
                variant_completed_fit_count += int(event.get("planned_fit_count", 0) or 0)
                event["completed_fit_count"] = int(variant_completed_fit_count)
                event["planned_fit_total"] = int(planned_fit_total)
                event["remaining_fit_count"] = int(planned_fit_total) - int(variant_completed_fit_count)
            _append_progress_log(args.latest_log_out, event)

        if variant_config == base_config and [str(path) for path in variant_extra_feature_paths] == [str(path) for path in args.extra_bg_features_path]:
            variant_bg_prior = bg_prior
            variant_diagnostics = diagnostics
        else:
            variant_bg_prior, variant_diagnostics, _ = build_model_surface(
                paths=paths,
                config=variant_config,
                extra_bg_feature_paths=list(variant_extra_feature_paths),
                progress_callback=_variant_progress_callback,
            )
        variant_city_summary = _city_share_summary(
            bg_prior=variant_bg_prior,
            year=year,
            city_shares=frozen_city_shares,
            bg_crosswalk=frozen_bg_crosswalk,
        )
        row = _summarize_variant(
            diagnostics=variant_diagnostics,
            config=variant_config,
            variant_name=variant_name,
            city_share_summary=variant_city_summary,
        )
        row["extra_bg_feature_paths"] = [str(path) for path in variant_extra_feature_paths]
        row["extra_bg_feature_count"] = int(len(variant_extra_feature_paths))
        return row

    def _write_rows(
        *,
        rows: list[dict[str, object]],
        parquet_path: Path | None,
        json_path: Path | None,
    ) -> None:
        if parquet_path is not None:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps({"year": year, "rows": rows}, indent=2, sort_keys=True))

    def _run_variant_group(
        variant_specs: list[tuple[str, dict[str, object]]],
        *,
        parquet_path: Path | None,
        json_path: Path | None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for variant_name, overrides in variant_specs:
            rows.append(_run_variant(variant_name, **overrides))
            _write_rows(rows=rows, parquet_path=parquet_path, json_path=json_path)
            gc.collect()
        return rows

    family_ladder_rows: list[dict[str, object]] = []
    if args.family_ladder_out is not None or args.family_ladder_json_out is not None:
        family_ladder_rows = _run_variant_group(
            [
                ("hist_gbm_observed_only", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=())),
                ("ridge_observed_only", dict(model_family="ridge", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=())),
                ("elastic_net_observed_only", dict(model_family="elastic_net", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=())),
                ("monotone_gam_observed_only", dict(model_family="monotone_gam", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=False, exclude_feature_groups=())),
            ],
            parquet_path=args.family_ladder_out,
            json_path=args.family_ladder_json_out,
        )
        summary["family_ladder"] = family_ladder_rows

    provenance_rows: list[dict[str, object]] = []
    if args.provenance_sensitivity_out is not None or args.provenance_sensitivity_json_out is not None:
        provenance_rows = _run_variant_group(
            [
                ("hist_gbm_observed_only", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=())),
                ("hist_gbm_include_estimated", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=False, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=())),
                ("hist_gbm_high_confidence_only", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=True, use_state_fixed_effects=True, exclude_feature_groups=())),
            ],
            parquet_path=args.provenance_sensitivity_out,
            json_path=args.provenance_sensitivity_json_out,
        )
        summary["provenance_sensitivity"] = provenance_rows

    ablation_rows: list[dict[str, object]] = []
    if args.feature_ablation_out is not None or args.feature_ablation_json_out is not None:
        ablation_rows = _run_variant_group(
            [
                ("hist_gbm_all_features", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=())),
                ("hist_gbm_no_state_fixed_effects", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=False, exclude_feature_groups=())),
                ("hist_gbm_no_activity_exposure", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=("activity_exposure",))),
                ("hist_gbm_no_roads_transport", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=("roads_transport",))),
                ("hist_gbm_no_land_cover", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=("land_cover",))),
                ("hist_gbm_no_institutional_anchors", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=("institutional_anchors",))),
                ("hist_gbm_no_socioeconomic_household", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, use_state_fixed_effects=True, exclude_feature_groups=("socioeconomic_household",))),
            ],
            parquet_path=args.feature_ablation_out,
            json_path=args.feature_ablation_json_out,
        )
        summary["feature_ablation"] = ablation_rows

    sparse_pooling_rows: list[dict[str, object]] = []
    if args.sparse_pooling_out is not None or args.sparse_pooling_json_out is not None:
        sparse_pooling_rows = _run_variant_group(
            [
                ("hist_gbm_observed_only", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, sparse_offense_pooling_strategy="none", use_state_fixed_effects=True, exclude_feature_groups=())),
                ("hist_gbm_observed_only_sparse_pooling", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, sparse_offense_pooling_strategy="empirical_bayes_group_rate", use_state_fixed_effects=True, exclude_feature_groups=())),
            ],
            parquet_path=args.sparse_pooling_out,
            json_path=args.sparse_pooling_json_out,
        )
        summary["sparse_offense_pooling"] = sparse_pooling_rows

    transit_rows: list[dict[str, object]] = []
    if (args.transit_sensitivity_out is not None or args.transit_sensitivity_json_out is not None) and args.transit_features_path is not None:
        transit_rows = _run_variant_group(
            [
                ("hist_gbm_observed_only", dict(model_family="hist_gbm", exclude_estimated_from_panel_from_training=True, high_confidence_training_only=False, sparse_offense_pooling_strategy="none", use_state_fixed_effects=True, exclude_feature_groups=())),
                (
                    "hist_gbm_observed_only_plus_transit",
                    dict(
                        model_family="hist_gbm",
                        exclude_estimated_from_panel_from_training=True,
                        high_confidence_training_only=False,
                        sparse_offense_pooling_strategy="none",
                        use_state_fixed_effects=True,
                        exclude_feature_groups=(),
                        extra_bg_feature_paths=[args.transit_features_path],
                    ),
                ),
            ],
            parquet_path=args.transit_sensitivity_out,
            json_path=args.transit_sensitivity_json_out,
        )
        summary["transit_sensitivity"] = transit_rows
    args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _append_progress_log(
        args.latest_log_out,
        {
            "event": "benchmark_summary",
            "summary_json_out": str(args.summary_json_out),
            "summary": summary,
        },
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
