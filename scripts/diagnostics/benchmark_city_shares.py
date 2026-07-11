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

from crimerisk.city_share_benchmark import build_city_share_diagnostics, build_city_share_truth_model_frame
from crimerisk.model_surface import ModelSurfaceConfig, build_model_surface
from crimerisk.paths import RepoPaths


def _load_simple_bg_baseline_weights() -> pd.DataFrame:
    acs = pd.read_parquet(
        REPO_ROOT / "data" / "ACS-5yr-2020-2024" / "parsed" / "acs_block_groups.parquet",
        columns=["bg_id", "state_fips", "total_population", "housing_units_total"],
    )
    lodes = pd.read_parquet(
        REPO_ROOT / "data" / "LODES" / "parsed" / "lodes_wac_block_groups.parquet",
        columns=["bg_id", "state_fips", "jobs_wac", "od_same_bg_jobs", "od_workplace_same_tract_jobs"],
    )
    nlcd = pd.read_parquet(
        REPO_ROOT / "data" / "NLCD" / "parsed" / "block_group_nlcd_2023.parquet",
        columns=[
            "bg_id",
            "state_fips",
            "nlcd_count_21",
            "nlcd_count_22",
            "nlcd_count_23",
            "nlcd_count_24",
            "nlcd_impervious_mean",
        ],
    )

    frame = acs.merge(lodes, on=["bg_id", "state_fips"], how="outer").merge(nlcd, on=["bg_id", "state_fips"], how="outer")
    frame["bg_id"] = frame["bg_id"].astype("string").str.zfill(12)
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)

    population = pd.to_numeric(frame["total_population"], errors="coerce").fillna(0.0).clip(lower=0.0)
    housing = pd.to_numeric(frame["housing_units_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
    jobs = pd.to_numeric(frame["jobs_wac"], errors="coerce").fillna(0.0).clip(lower=0.0)
    local_jobs = pd.to_numeric(frame["od_same_bg_jobs"], errors="coerce").fillna(0.0).clip(lower=0.0)
    same_tract_jobs = pd.to_numeric(frame["od_workplace_same_tract_jobs"], errors="coerce").fillna(0.0).clip(lower=0.0)
    developed_land = (
        pd.to_numeric(frame["nlcd_count_21"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["nlcd_count_22"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["nlcd_count_23"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["nlcd_count_24"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    impervious = pd.to_numeric(frame["nlcd_impervious_mean"], errors="coerce").fillna(0.0).clip(lower=0.0)

    frame["population_weight"] = population
    frame["housing_weight"] = housing
    frame["jobs_weight"] = jobs
    frame["developed_land_weight"] = developed_land
    frame["population_plus_jobs_weight"] = population + jobs
    frame["local_activity_weight"] = population + jobs + local_jobs + same_tract_jobs
    frame["impervious_area_weight"] = developed_land * impervious

    return frame[
        [
            "bg_id",
            "state_fips",
            "population_weight",
            "housing_weight",
            "jobs_weight",
            "developed_land_weight",
            "population_plus_jobs_weight",
            "local_activity_weight",
            "impervious_area_weight",
        ]
    ].drop_duplicates(["bg_id", "state_fips"])


def _build_simple_bg_prior(
    *,
    baseline_weights: pd.DataFrame,
    weight_col: str,
    offenses: list[str],
) -> pd.DataFrame:
    base = baseline_weights[["bg_id", "state_fips", weight_col]].copy()
    base["bg_weight"] = pd.to_numeric(base[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    base = base.drop(columns=[weight_col])
    frames = [base.assign(offense=str(offense)) for offense in offenses]
    if not frames:
        return pd.DataFrame(columns=["bg_id", "state_fips", "offense", "bg_weight"])
    return pd.concat(frames, ignore_index=True)


def _summarize_benchmark_row(model_name: str, summary: dict[str, object]) -> dict[str, object]:
    return {
        "model_name": str(model_name),
        "rows": int(summary.get("rows", 0) or 0),
        "city_count": int(summary.get("city_count", 0) or 0),
        "incident_total": float(summary.get("incident_total", 0.0) or 0.0),
        "weighted_total_variation_distance_mean": summary.get("weighted_total_variation_distance_mean"),
        "weighted_share_rmse_mean": summary.get("weighted_share_rmse_mean"),
        "weighted_pearson_share_mean": summary.get("weighted_pearson_share_mean"),
        "weighted_spearman_share_mean": summary.get("weighted_spearman_share_mean"),
        "weighted_top_10pct_true_mass_in_model_top_10pct_mean": summary.get(
            "weighted_top_10pct_true_mass_in_model_top_10pct_mean"
        ),
    }


def build_city_share_benchmark(
    *,
    city_shares: pd.DataFrame,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = build_city_share_truth_model_frame(
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=year,
    )
    diagnostics, summary = build_city_share_diagnostics(merged, predicted_share_col="model_share")
    summary["year"] = int(year)
    return diagnostics, summary


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
        "--include-estimated-training-rows",
        action="store_true",
        help="Include retained panel-estimated control rows in the bg-prior training sample.",
    )
    parser.add_argument(
        "--city-shares-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet",
    )
    parser.add_argument(
        "--bg-prior-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--bg-crosswalk-path",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument(
        "--force-bg-prior-rebuild",
        action="store_true",
        help="Recompute the bg_prior_long artifact from the live model surface before benchmarking.",
    )
    parser.add_argument(
        "--allow-canonical-overwrite",
        action="store_true",
        help="Allow non-default benchmark configurations to overwrite canonical city-benchmark or bg-prior artifacts.",
    )
    parser.add_argument("--diagnostics-out", type=Path, default=None)
    parser.add_argument("--summary-json-out", type=Path, default=None)
    parser.add_argument("--baseline-ladder-out", type=Path, default=None)
    parser.add_argument("--baseline-ladder-json-out", type=Path, default=None)
    args = parser.parse_args()

    year = int(args.year)
    canonical_bg_prior_path = (
        REPO_ROOT / "state" / "modeling" / f"bg_prior_long_{year}.parquet"
    )
    canonical_diagnostics_out = (
        REPO_ROOT / "state" / "modeling" / f"city_share_benchmark_{year}.parquet"
    )
    canonical_summary_json_out = (
        REPO_ROOT / "state" / "modeling" / f"city_share_benchmark_{year}.json"
    )
    canonical_baseline_ladder_out = (
        REPO_ROOT / "state" / "modeling" / f"city_share_baseline_ladder_{year}.parquet"
    )
    canonical_baseline_ladder_json_out = (
        REPO_ROOT / "state" / "modeling" / f"city_share_baseline_ladder_{year}.json"
    )
    if args.bg_prior_path is None:
        args.bg_prior_path = canonical_bg_prior_path
    if args.diagnostics_out is None:
        args.diagnostics_out = canonical_diagnostics_out
    if args.summary_json_out is None:
        args.summary_json_out = canonical_summary_json_out
    if args.baseline_ladder_out is None:
        args.baseline_ladder_out = canonical_baseline_ladder_out
    if args.baseline_ladder_json_out is None:
        args.baseline_ladder_json_out = canonical_baseline_ladder_json_out

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
            bool(args.include_estimated_training_rows),
        ]
    )
    if experimental_config and not bool(args.force_bg_prior_rebuild):
        raise SystemExit(
            "Non-default city-share benchmark configurations must rebuild the bg_prior surface in-process. "
            "Pass --force-bg-prior-rebuild and a scratch --bg-prior-path, or use canonical defaults."
        )
    canonical_targets = {
        canonical_bg_prior_path.resolve(),
        canonical_diagnostics_out.resolve(),
        canonical_summary_json_out.resolve(),
    }
    requested_targets = {
        Path(args.bg_prior_path).resolve(),
        Path(args.diagnostics_out).resolve(),
        Path(args.summary_json_out).resolve(),
        Path(args.baseline_ladder_out).resolve(),
        Path(args.baseline_ladder_json_out).resolve(),
    }
    if experimental_config and not bool(args.allow_canonical_overwrite):
        overlapping = sorted(str(path) for path in requested_targets & canonical_targets)
        if overlapping:
            raise SystemExit(
                "Refusing to overwrite canonical city-share benchmark artifacts with a non-default "
                "model configuration. Pass explicit scratch output paths or use "
                "--allow-canonical-overwrite if you are intentionally promoting the experiment. "
                f"Overlapping targets: {overlapping}"
            )

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    if bool(args.force_bg_prior_rebuild) or not args.bg_prior_path.exists():
        bg_prior, _, _ = build_model_surface(
            paths=paths,
            config=ModelSurfaceConfig(
                year=year,
                min_training_population=int(args.min_training_population),
                model_family=str(args.model_family),
                ridge_alpha=float(args.ridge_alpha),
                elastic_net_alpha=float(args.elastic_net_alpha),
                elastic_net_l1_ratio=float(args.elastic_net_l1_ratio),
                elastic_net_max_iter=int(args.elastic_net_max_iter),
                monotone_gam_lam=float(args.monotone_gam_lam),
                monotone_gam_n_splines=int(args.monotone_gam_n_splines),
                monotone_gam_max_iter=int(args.monotone_gam_max_iter),
                hist_learning_rate=float(args.hist_learning_rate),
                hist_max_depth=int(args.hist_max_depth),
                hist_max_iter=int(args.hist_max_iter),
                hist_min_samples_leaf=int(args.hist_min_samples_leaf),
                hist_l2_regularization=float(args.hist_l2_regularization),
                compute_diagnostics=False,
                exclude_estimated_from_panel_from_training=not bool(args.include_estimated_training_rows),
            ),
        )
        args.bg_prior_path.parent.mkdir(parents=True, exist_ok=True)
        bg_prior.to_parquet(args.bg_prior_path, index=False)
    city_shares = pd.read_parquet(args.city_shares_path)
    bg_prior = pd.read_parquet(args.bg_prior_path)
    bg_crosswalk = pd.read_parquet(args.bg_crosswalk_path)
    diagnostics, summary = build_city_share_benchmark(
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=year,
    )
    baseline_weights = _load_simple_bg_baseline_weights()
    offenses = sorted(
        {
            str(offense).strip()
            for offense in city_shares["offense"].astype("string").dropna().tolist()
            if str(offense).strip()
        }
    )
    ladder_specs = [
        ("hist_gbm" if str(args.model_family) == "hist_gbm" else str(args.model_family), bg_prior),
        ("population_share", _build_simple_bg_prior(baseline_weights=baseline_weights, weight_col="population_weight", offenses=offenses)),
        ("housing_share", _build_simple_bg_prior(baseline_weights=baseline_weights, weight_col="housing_weight", offenses=offenses)),
        ("jobs_share", _build_simple_bg_prior(baseline_weights=baseline_weights, weight_col="jobs_weight", offenses=offenses)),
        (
            "population_plus_jobs_share",
            _build_simple_bg_prior(baseline_weights=baseline_weights, weight_col="population_plus_jobs_weight", offenses=offenses),
        ),
        (
            "local_activity_share",
            _build_simple_bg_prior(baseline_weights=baseline_weights, weight_col="local_activity_weight", offenses=offenses),
        ),
        (
            "developed_land_share",
            _build_simple_bg_prior(baseline_weights=baseline_weights, weight_col="developed_land_weight", offenses=offenses),
        ),
        (
            "impervious_area_share",
            _build_simple_bg_prior(baseline_weights=baseline_weights, weight_col="impervious_area_weight", offenses=offenses),
        ),
    ]
    ladder_frames: list[pd.DataFrame] = []
    ladder_summary_rows: list[dict[str, object]] = []
    for model_name, candidate_bg_prior in ladder_specs:
        candidate_diag, candidate_summary = build_city_share_benchmark(
            city_shares=city_shares,
            bg_prior=candidate_bg_prior,
            bg_crosswalk=bg_crosswalk,
            year=year,
        )
        candidate_diag = candidate_diag.assign(model_name=str(model_name))
        ladder_frames.append(candidate_diag)
        ladder_summary_rows.append(_summarize_benchmark_row(str(model_name), candidate_summary))
    summary["exclude_estimated_from_panel_from_training"] = not bool(args.include_estimated_training_rows)
    summary["baseline_ladder"] = ladder_summary_rows
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_ladder_out.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_ladder_json_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(args.diagnostics_out, index=False)
    pd.concat(ladder_frames, ignore_index=True).to_parquet(args.baseline_ladder_out, index=False)
    args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    args.baseline_ladder_json_out.write_text(json.dumps({"year": year, "rows": ladder_summary_rows}, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
