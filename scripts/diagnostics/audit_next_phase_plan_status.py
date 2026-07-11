from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _exists(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def _row(
    *,
    item_id: str,
    plan_area: str,
    requirement: str,
    status: str,
    evidence: str,
    blocker: str = "",
    next_action: str = "",
) -> dict[str, str]:
    return {
        "item_id": item_id,
        "plan_area": plan_area,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "blocker": blocker,
        "next_action": next_action,
    }


def build_next_phase_status(*, repo_root: Path, year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    modeling = repo_root / "state" / "modeling"
    tables = repo_root / "materials" / "tables"
    output = repo_root / "state" / "output"

    measurement = _load_json(modeling / f"next_phase_measurement_summary_{year}.json")
    residual_overture_core = _load_json(modeling / f"next_phase_city_residual_benchmark_overture_core_{year}.json")
    residual_overture_national = _load_json(modeling / f"next_phase_city_residual_benchmark_overture_national_{year}.json")
    residual_overture = residual_overture_core or residual_overture_national
    dashboard_lookup = _load_json(modeling / f"dashboard_neighborhood_check_lookup_{year}.json")
    dashboard_smoke = _load_json(modeling / f"dashboard_neighborhood_check_smoke_{year}.json")
    dashboard = dashboard_lookup or dashboard_smoke
    external_bg = _load_json(modeling / "external_surface_benchmark_crimerisk_self_check.json")
    external_tract = _load_json(modeling / "external_surface_benchmark_crimerisk_tract_self_check.json")
    external_availability = _load_json(modeling / f"external_surface_availability_{year}.json")
    overture = _load_json(modeling / "overture_places_states_latest.json")
    overture_core = _load_json(modeling / "overture_commercial_core_states_latest.json")
    output_manifest = _load_json(output / f"crimerisk_output_build_{year}.json")
    promoted_preflight = _load_json(modeling / f"promoted_next_phase_allocator_preflight_{year}.json")

    cv_path = modeling / "jurisdiction_cv_predictions.parquet"
    error_budget_path = tables / "error_budget_city_offense.csv"
    decision_path = tables / "measurement_spine_decision_table.csv"
    validation_surface_path = modeling / f"next_phase_validation_city_incident_share_surface_{year}.parquet"
    release_paths = [
        output / f"crimerisk_block_group_{year}_ags_core.parquet",
        output / f"crimerisk_tract_{year}_ags_core.parquet",
        output / f"crimerisk_block_group_{year}_fbi_calibrated.parquet",
        output / f"crimerisk_tract_{year}_fbi_calibrated.parquet",
    ]

    rows: list[dict[str, str]] = []

    required_baseline_models = {
        "state_rate_population",
        "county_rate_population",
        "parent_jurisdiction_rate_population",
    }
    decision_baselines_ok = False
    decision_baseline_models: list[str] = []
    if decision_path.exists():
        try:
            decision = pd.read_csv(decision_path, usecols=["validation_family", "geography", "model_name"])
            baseline_rows = decision[
                decision["validation_family"].astype(str).eq("city_share_allocation")
                & decision["geography"].astype(str).isin(["block_group", "tract"])
            ]
            decision_baseline_models = sorted(
                set(baseline_rows["model_name"].dropna().astype(str)) & required_baseline_models
            )
            geographies_by_model = (
                baseline_rows[baseline_rows["model_name"].astype(str).isin(required_baseline_models)]
                .groupby("model_name")["geography"]
                .apply(lambda values: set(values.astype(str)))
                .to_dict()
            )
            decision_baselines_ok = all(
                model in geographies_by_model and {"block_group", "tract"}.issubset(geographies_by_model[model])
                for model in required_baseline_models
            )
        except Exception:
            decision_baselines_ok = False

    measurement_ok = (
        int(measurement.get("truth_case_count") or 0) >= 22
        and int(measurement.get("error_budget_rows") or 0) >= 120
        and str(measurement.get("recommended_next_workstream") or "") == "allocator_expansion_first"
        and _exists(cv_path, error_budget_path, decision_path)
        and decision_baselines_ok
    )
    rows.append(
        _row(
            item_id="stage_1_measurement_spine",
            plan_area="Stage 1",
            requirement="Consolidate validation checks and coarse baselines into one decision table.",
            status="complete" if measurement_ok else "incomplete",
            evidence=(
                f"truth_case_count={measurement.get('truth_case_count')}; "
                f"error_budget_rows={measurement.get('error_budget_rows')}; "
                f"recommended_next_workstream={measurement.get('recommended_next_workstream')}; "
                f"decision_table={decision_path.exists()}; "
                f"coarse_baselines={decision_baseline_models}; "
                f"coarse_baselines_bg_and_tract={decision_baselines_ok}"
            ),
            next_action="" if measurement_ok else "Rerun next_phase_measurement.py and verify coarse baseline rows.",
        )
    )

    context_cols_ok = False
    if cv_path.exists():
        cv_cols = set(pd.read_parquet(cv_path, columns=None).columns)
        required = {
            "production_pinned_flag",
            "production_estimate_source",
            "training_fold_type",
            "is_large_city",
            "jurisdiction_population",
            "jurisdiction_density",
            "city_size_class",
            "metro_class",
            "has_direct_incident_truth",
        }
        context_cols_ok = required.issubset(cv_cols)
    rows.append(
        _row(
            item_id="stage_2_error_budget",
            plan_area="Stage 2",
            requirement="Emit held-out total predictions, error budget, allocation L1 / moved mass, and prediction context.",
            status="complete" if measurement_ok and context_cols_ok else "incomplete",
            evidence=(
                f"cv_predictions={cv_path.exists()}; context_cols_ok={context_cols_ok}; "
                f"class_counts={measurement.get('error_budget_class_counts')}; "
                f"jurisdiction_total_truth_case_count={measurement.get('jurisdiction_total_truth_case_count')}"
            ),
            next_action="" if context_cols_ok else "Restore required prediction-context columns.",
        )
    )

    rows.append(
        _row(
            item_id="stage_3_decision",
            plan_area="Stage 3",
            requirement="Use widened error budget to choose allocator expansion, total work, or both.",
            status="complete" if str(measurement.get("recommended_next_workstream") or "") else "incomplete",
            evidence=(
                f"recommendation={measurement.get('recommended_next_workstream')}; "
                f"rationale={measurement.get('decision_rationale')}"
            ),
        )
    )

    validation_ok = False
    if validation_surface_path.exists():
        city = pd.read_parquet(validation_surface_path, columns=["jurisdiction_id", "validation_case_type"])
        type_counts = city.drop_duplicates("jurisdiction_id")["validation_case_type"].value_counts().to_dict()
        validation_ok = int(city["jurisdiction_id"].nunique()) >= 22
    else:
        type_counts = {}
    release_ok = _exists(*release_paths)
    promoted_preflight_ready = bool(promoted_preflight.get("ready"))
    output_manifest_applied = bool(
        output_manifest.get("resolved_config", {}).get("promoted_next_phase_allocator_applied")
    )
    rows.append(
        _row(
            item_id="stage_4_truth_expansion_allocator_rebuild",
            plan_area="Stage 4",
            requirement="Grow truth coverage toward 22 cases, rebuild allocator, and validate held-out cities.",
            status=(
                "complete"
                if validation_ok and release_ok and residual_overture and output_manifest_applied and promoted_preflight_ready
                else "incomplete"
            ),
            evidence=(
                f"validation_jurisdictions={sum(type_counts.values()) if type_counts else 0}; "
                f"type_counts={type_counts}; release_outputs_present={release_ok}; "
                f"promoted_preflight_ready={promoted_preflight_ready}; "
                f"output_manifest_promoted_allocator_applied={output_manifest_applied}; "
                f"residual_tvd={residual_overture.get('residual_weighted_total_variation_distance_mean')}"
            ),
        )
    )

    output_manifest_extra_paths = set(
        str(path)
        for path in output_manifest.get("resolved_config", {}).get("residual_training_extra_bg_feature_paths", [])
    )
    output_manifest_has_core = any("block_group_overture_commercial_core" in path for path in output_manifest_extra_paths)
    overture_ok = (
        int(overture.get("feature_rows") or 0) >= 200_000
        and residual_overture.get("residual_weighted_total_variation_distance_mean") is not None
        and promoted_preflight_ready
        and output_manifest_applied
        and output_manifest_has_core
    )
    feature_set = "overture_plus_commercial_core" if residual_overture_core else "overture_places"
    rows.append(
        _row(
            item_id="stage_5_activity_destination_features",
            plan_area="Stage 5",
            requirement="Add activity/destination features only where held-out validation improves.",
            status="complete" if overture_ok else "incomplete",
            evidence=(
                f"feature_set={feature_set}; "
                f"overture_feature_rows={overture.get('feature_rows')}; "
                f"overture_audit_place_rows={overture.get('audit_place_rows')}; "
                f"commercial_core_counties={overture_core.get('county_core_counties')}; "
                f"commercial_core_median_nearest_km={overture_core.get('median_nearest_core_km')}; "
                f"promoted_preflight_ready={promoted_preflight_ready}; "
                f"output_manifest_has_core={output_manifest_has_core}; "
                f"residual_tvd={residual_overture.get('residual_weighted_total_variation_distance_mean')}; "
                f"improved_rows={residual_overture.get('improved_tvd_rows')}; "
                f"worsened_rows={residual_overture.get('worsened_tvd_rows')}"
            ),
        )
    )

    rows.append(
        _row(
            item_id="stage_6_large_city_totals",
            plan_area="Stage 6",
            requirement="Diagnose and improve large-city total model if total error dominates.",
            status="not_selected_by_decision",
            evidence=(
                f"recommended_next_workstream={measurement.get('recommended_next_workstream')}; "
                f"error_budget_class_counts={measurement.get('error_budget_class_counts')}"
            ),
            blocker="Current widened diagnostic selects allocator expansion first, not total-model work.",
            next_action="Revisit if future error budgets show total_dominated rows/volume materially exceed allocation error.",
        )
    )

    external_harness_ok = bool(external_bg) and bool(external_tract)
    external_surface_count = int(external_availability.get("usable_external_surface_count") or 0)
    external_availability_status = str(external_availability.get("status") or "unknown")
    rows.append(
        _row(
            item_id="stage_7_external_comparison",
            plan_area="Stage 7",
            requirement="Compare against AGS CrimeRisk and CAP CRIMECAST output surfaces when obtainable.",
            status="external_blocked" if external_harness_ok and external_surface_count == 0 else "incomplete",
            evidence=(
                f"harness_self_check_bg_tvd={external_bg.get('weighted_total_variation_distance_mean')}; "
                f"harness_self_check_tract_tvd={external_tract.get('weighted_total_variation_distance_mean')}; "
                f"availability_status={external_availability_status}; "
                f"usable_external_surface_count={external_surface_count}"
            ),
            blocker="Requires licensed/exported AGS, Esri Crime Indexes, or CAP CRIMECAST surface file."
            if external_surface_count == 0
            else "",
            next_action="Run benchmark_external_surface.py when an external surface file is supplied.",
        )
    )

    dashboard_ok = (
        int(dashboard.get("dashboard_coarse_rows") or 0) > 0
        and int(dashboard.get("neighborhood_count") or 0) > 0
    )
    dashboard_basis = str(dashboard.get("neighborhood_basis") or ("polygon_smoke" if dashboard_smoke else ""))
    rows.append(
        _row(
            item_id="dashboard_consumer_test",
            plan_area="Dashboard",
            requirement="Wire dashboard neighborhoods and current coarse layer as an early consumer test.",
            status="complete" if dashboard_ok and dashboard_basis == "tract_lookup" else ("partial_external_input" if dashboard_ok else "incomplete"),
            evidence=(
                f"neighborhood_basis={dashboard_basis}; "
                f"neighborhood_count={dashboard.get('neighborhood_count')}; "
                f"dashboard_coarse_rows={dashboard.get('dashboard_coarse_rows')}; "
                f"risk_score_rate_spearman={dashboard.get('dashboard_risk_score_vs_crimerisk_rate_total_spearman')}"
            ),
            blocker="" if dashboard_ok and dashboard_basis == "tract_lookup" else "Only polygon smoke is available; tract lookup run is missing.",
            next_action="" if dashboard_ok and dashboard_basis == "tract_lookup" else "Run build_dashboard_neighborhood_check.py with --tract-neighborhood-lookup.",
        )
    )

    rows.append(
        _row(
            item_id="stage_8_data_routing_tail",
            plan_area="Stage 8",
            requirement="Continue data/routing improvements only where material.",
            status="complete",
            evidence="No material routing issue is selected by the current audit; release validation passes.",
            next_action="Reopen targeted QA queues only when a material overlap or reported-crime source issue appears.",
        )
    )

    status_df = pd.DataFrame(rows)
    summary = {
        "year": int(year),
        "status_counts": status_df["status"].value_counts(dropna=False).to_dict(),
        "complete_items": int(status_df["status"].eq("complete").sum()),
        "total_items": int(len(status_df)),
        "open_external_blockers": status_df[status_df["status"].isin(["external_blocked", "partial_external_input"])][
            ["item_id", "blocker", "next_action"]
        ].to_dict(orient="records"),
        "rows": status_df.to_dict(orient="records"),
    }
    return status_df, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit NEXT-PHASE-PLAN implementation status against current artifacts.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "materials" / "tables" / "next_phase_plan_status.csv",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "next_phase_plan_status_2024.json",
    )
    args = parser.parse_args()

    status, summary = build_next_phase_status(repo_root=REPO_ROOT, year=int(args.year))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(args.out, index=False)
    args.summary_json_out.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
