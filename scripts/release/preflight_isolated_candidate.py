"""Fail closed unless an isolated candidate can reuse every selected upstream artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


class _ConfigCaptured(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _writable_parent_probe(parent: Path) -> bool:
    """Confirm a derived-output parent is writable without retaining an artifact."""
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=parent, prefix=".preflight-", delete=True):
        pass
    return True


def _validator_city_exact_point_qa(
    *, source_root: Path, runtime_root: Path, year: int
) -> dict[str, object]:
    """Run the release validator's exact QA check against runtime-rooted inputs."""
    script = source_root / "scripts" / "diagnostics" / "validate_release_outputs.py"
    spec = importlib.util.spec_from_file_location("isolated_preflight_release_validator", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load release validator: {script}")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    overlay = runtime_root / "repo-overlay"
    validator.REPO_ROOT = overlay
    validator._apply_target_year(year)
    validator.REPO_CITY_EXACT_POINT_EXCEPTIONS = (
        overlay / "configs" / "city_feed_exact_point_exceptions.csv"
    )
    issues: list[str] = []
    summary = validator._check_city_feed_exact_point_tripwire(issues=issues)
    return {"ok": summary.get("ok") is True, "issues": issues, "summary": summary}


def _effective_config_parity(*, paths, launch: dict[str, object]) -> dict[str, object]:
    """Resolve the exact CLI config without entering the build and compare it with v52."""
    from crimerisk import cli
    from crimerisk.allocation import resolve_allocation_build_config

    captured: dict[str, object] = {}

    def capture_write(**kwargs):
        captured.update(kwargs)
        raise _ConfigCaptured

    command = list(launch["candidate_command"])
    parsed = cli.build_parser().parse_args(command)
    original_get_paths = cli.get_paths
    original_write = cli.write_v2_outputs
    cli.get_paths = lambda: paths
    cli.write_v2_outputs = capture_write
    try:
        try:
            parsed.func(parsed)
        except _ConfigCaptured:
            pass
    finally:
        cli.get_paths = original_get_paths
        cli.write_v2_outputs = original_write
    config = resolve_allocation_build_config(paths, config=captured["config"])
    baseline_manifest_path = Path(
        str(
            launch.get("baseline_manifest_path")
            or (
                Path(str(launch["canonical_root"]))
                / "state"
                / "candidates"
                / str(launch["baseline_run_id"])
                / "manifest.json"
            )
        )
    )
    baseline = json.loads(baseline_manifest_path.read_text())["resolved_config"]
    actual = {
        "control_surface": config.control_surface,
        "force_controls_rebuild": config.force_controls_rebuild,
        "force_reporting_regimes_rebuild": config.force_reporting_regimes_rebuild,
        "force_geometry_rebuild": config.force_geometry_rebuild,
        "force_bg_prior_rebuild": config.force_bg_prior_rebuild,
        "force_city_incident_share_rebuild": config.force_city_incident_share_rebuild,
        "force_city_incident_source_refresh": config.force_city_incident_source_refresh,
        "use_promoted_next_phase_allocator": config.use_promoted_next_phase_allocator,
        "model_surface_prior_anchor": config.model_surface_prior_anchor,
        "model_surface_exclude_feature_policy_classes": list(
            config.model_surface_exclude_feature_policy_classes
        ),
        "residual_training_exclude_validation_case_types": list(
            config.residual_training_exclude_validation_case_types
        ),
        "residual_exclude_feature_policy_classes": list(
            config.residual_exclude_feature_policy_classes
        ),
        "residual_exclude_feature_policy_classes_by_offense": {
            offense: list(classes)
            for offense, classes in config.residual_exclude_feature_policy_classes_by_offense
        },
        "residual_transfer_tau_by_offense": dict(config.residual_transfer_tau_by_offense),
        "rare_offense_information_constant_by_offense": dict(
            config.rare_offense_information_constant_by_offense
        ),
        "eb_alpha_by_offense": dict(config.eb_alpha_by_offense),
        "eb_hard_min_denominator": config.eb_hard_min_denominator,
        "city_posterior_reconciliation_tolerance": config.city_posterior_reconciliation_tolerance,
        "city_posterior_alpha_floor": config.city_posterior_alpha_floor,
        "city_posterior_alpha_volume_incidents": config.city_posterior_alpha_volume_incidents,
        "city_posterior_alpha_max_prior_fraction": config.city_posterior_alpha_max_prior_fraction,
        "enable_county_anchoring": config.enable_county_anchoring,
        "enable_mixture_allocation": config.enable_mixture_allocation,
        "enable_exposure_ensemble": config.enable_exposure_ensemble,
        "enable_count_first_composites": config.enable_count_first_composites,
        "enable_special_use_taxonomy": config.enable_special_use_taxonomy,
        "enable_uncertainty_layer": config.enable_uncertainty_layer,
        "uncertainty_draws": config.uncertainty_draws,
        "enable_soft_shrinkage": config.enable_soft_shrinkage,
        "soft_shrinkage_nu": config.soft_shrinkage_nu,
        "soft_shrinkage_extrapolation_weight": config.soft_shrinkage_extrapolation_weight,
        "enable_imputation_v2": config.enable_imputation_v2,
        "enable_unlocated_mass": config.enable_unlocated_mass,
    }
    expected = {
        "control_surface": baseline["control_surface"],
        "force_controls_rebuild": baseline["force_controls_rebuild"],
        "force_reporting_regimes_rebuild": baseline["force_reporting_regimes_rebuild"],
        "force_geometry_rebuild": baseline["force_geometry_rebuild"],
        "force_bg_prior_rebuild": baseline["force_bg_prior_rebuild"],
        "force_city_incident_share_rebuild": baseline["force_city_incident_share_rebuild"],
        "force_city_incident_source_refresh": baseline["force_city_incident_source_refresh"],
        "use_promoted_next_phase_allocator": baseline["use_promoted_next_phase_allocator"],
        "model_surface_prior_anchor": baseline["model_surface_prior_anchor"],
        "model_surface_exclude_feature_policy_classes": baseline[
            "model_surface_exclude_feature_policy_classes"
        ],
        "residual_training_exclude_validation_case_types": baseline[
            "residual_training_exclude_validation_case_types"
        ],
        "residual_exclude_feature_policy_classes": baseline[
            "residual_exclude_feature_policy_classes"
        ],
        "residual_exclude_feature_policy_classes_by_offense": baseline[
            "residual_exclude_feature_policy_classes_by_offense"
        ],
        "residual_transfer_tau_by_offense": baseline["residual_transfer_tau_by_offense"],
        "rare_offense_information_constant_by_offense": baseline[
            "rare_offense_tract_information_shrinkage"
        ]["information_constant_by_offense"],
        "eb_alpha_by_offense": baseline["eb_alpha_by_offense"],
        "eb_hard_min_denominator": baseline["eb_hard_min_denominator"],
        "city_posterior_reconciliation_tolerance": baseline[
            "city_posterior_reconciliation_tolerance"
        ],
        "city_posterior_alpha_floor": baseline["city_posterior_alpha_floor"],
        "city_posterior_alpha_volume_incidents": baseline[
            "city_posterior_alpha_volume_incidents"
        ],
        "city_posterior_alpha_max_prior_fraction": baseline[
            "city_posterior_alpha_max_prior_fraction"
        ],
        "enable_county_anchoring": True,
        "enable_mixture_allocation": baseline["mixture_allocation"]["enabled"],
        "enable_exposure_ensemble": baseline["exposure_ensemble"]["enabled"],
        "enable_count_first_composites": baseline["count_first_composites"]["enabled"],
        "enable_special_use_taxonomy": baseline["special_use_taxonomy"]["enabled"],
        "enable_uncertainty_layer": baseline["uncertainty_layer"]["enabled"],
        "uncertainty_draws": baseline["uncertainty_layer"]["n_draws"],
        "enable_soft_shrinkage": baseline["soft_shrinkage"]["enabled"],
        "soft_shrinkage_nu": baseline["soft_shrinkage"]["nu"],
        "soft_shrinkage_extrapolation_weight": baseline["soft_shrinkage"][
            "extrapolation_weight"
        ],
        "enable_imputation_v2": baseline["imputation_v2"]["enabled"],
        "enable_unlocated_mass": baseline["unlocated_mass"]["enabled"],
    }
    differences = {
        key: {"baseline": expected[key], "candidate": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    allowed = dict(launch.get("allowed_config_differences") or {})
    unexpected = {
        key: value
        for key, value in differences.items()
        if key not in allowed or actual[key] != allowed[key]
    }
    return {
        "ok": not unexpected,
        "baseline_run_id": launch["baseline_run_id"],
        "actual": actual,
        "expected": expected,
        "differences": differences,
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "path_roles_sha_bound": [
            "bg_prior",
            "model_and_residual_feature_policy",
            "residual_training_city_shares",
            "residual_training_extra_features",
            "mixture_experts_and_weights",
            "exposure_normalizers_and_weights",
            "direct_city_share_surface",
            "murder_and_burglary_calibration",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve()
    runtime_root = args.runtime_root.resolve()
    sys.path.insert(0, str(source_root / "src"))

    from run_isolated_candidate import _verify_source_identity

    _verify_source_identity(source_root, runtime_root)
    from crimerisk.build_freshness import artifact_is_current
    from crimerisk.city_shares import _city_dependency_paths, _city_impls
    from crimerisk.controls import controls_artifacts_are_current
    from crimerisk.geometry import GeometryBuildConfig, geometry_artifacts_are_current
    from crimerisk.observations import ObservationBuildConfig, observations_artifacts_are_current
    from crimerisk.paths import RepoPaths
    from crimerisk.reference_layers import (
        ReferenceLayerBuildConfig,
        reference_layers_artifacts_are_current,
    )
    from crimerisk.reporting_regimes import (
        ReportingRegimeBuildConfig,
        reporting_regimes_artifact_is_current,
    )
    from crimerisk.smoothed_controls import smoothed_controls_artifact_is_current
    from crimerisk.allocation import (
        _agency_estimates_cache_path,
        _agency_estimates_dependency_paths,
        _load_overlap_custom_footprints,
    )
    from crimerisk.service_scopes import load_custom_footprint_population_by_target

    launch = json.loads((runtime_root / "launch.json").read_text())
    bindings = json.loads((runtime_root / "arm_input_bindings.json").read_text())
    state_dir = runtime_root / "state"
    review_dir = state_dir / "review"
    paths = replace(
        RepoPaths.from_repo_root(runtime_root / "repo-overlay"),
        state_dir=state_dir,
        data_dir=runtime_root / "data",
        cache_dir=state_dir / "cache",
        archive_dir=runtime_root / "archive",
        review_dir=review_dir,
        review_support_dir=review_dir / "support",
        review_queues_dir=review_dir / "queues",
        review_packets_dir=review_dir / "packets",
        review_analysis_dir=review_dir / "analysis",
        review_runs_dir=review_dir / "runs",
    )
    year = int(launch["year"])
    declared_rebuilt = set(launch.get("declared_rebuilt_derived_roles") or [])
    rebuild_service_controls = bool(declared_rebuilt)
    controls_current = controls_artifacts_are_current(
        paths,
        year=year,
        state_out_path=state_dir / "controls" / "state_control_comparison.parquet",
        jurisdiction_out_path=state_dir / "controls" / f"jurisdiction_controls_{year}.parquet",
        jurisdiction_year_estimates_out_path=state_dir
        / "controls"
        / "jurisdiction_year_estimates.parquet",
    )
    agency_estimates_current = artifact_is_current(
        _agency_estimates_cache_path(paths, year),
        _agency_estimates_dependency_paths(paths, year=year),
    )
    structural_zero_path = (
        paths.repo_root
        / "analysis_scratch"
        / "level_lane_screen"
        / "structural_zero_corroborated_all.csv"
    )
    footprints = _load_overlap_custom_footprints(paths)
    service_rows = footprints[
        footprints["allocation_scope"].astype("string").eq("service_wide")
    ]
    ordinary_resident_rows = footprints[
        footprints["allocation_scope"].astype("string").eq("source_state")
        & footprints["weight_share_basis"].astype("string").eq("resident_population")
    ]
    service_population = load_custom_footprint_population_by_target(paths)
    covered_population_owners = set(service_population["ori9"].astype(str))
    ordinary_eligible_owners = covered_population_owners - {"AZ0018900"}
    service_population_value = float(
        service_population.loc[
            service_population["ori9"].eq("AZ0018900"), "_covered_population"
        ].sum()
    )
    checks = {
        "selected_input_sha256_equal": bool(bindings.get("all_selected_inputs_equal")),
        "candidate_id_unused": not (state_dir / "candidates" / str(launch["run_id"])).exists(),
        "reference": reference_layers_artifacts_are_current(
            paths=paths,
            config=ReferenceLayerBuildConfig(year_start=2018, year_end=year),
        ),
        "observations": observations_artifacts_are_current(
            paths, config=ObservationBuildConfig(year_start=2018, year_end=year)
        ),
        "reporting": reporting_regimes_artifact_is_current(
            paths, config=ReportingRegimeBuildConfig(year_start=2018, year_end=year)
        ),
        "geometry": geometry_artifacts_are_current(
            paths,
            block_out_path=state_dir / "geometry" / "block_to_jurisdiction_crosswalk.parquet",
            block_group_out_path=state_dir
            / "geometry"
            / "block_group_to_jurisdiction_crosswalk.parquet",
            config=GeometryBuildConfig(),
        ),
        "controls_fresh_or_declared_rebuild": (
            not controls_current if rebuild_service_controls else controls_current
        ),
        "agency_estimates_fresh_or_declared_rebuild": (
            not agency_estimates_current if rebuild_service_controls else agency_estimates_current
        ),
        "structural_zero_input_present": structural_zero_path.is_file(),
        "federal_mass_audit_output_writable": _writable_parent_probe(
            state_dir / "analysis" / "dc_federal"
        ),
        "service_scope_contract": bool(
            len(service_rows) == 166
            and set(service_rows["state_fips"].astype(str)) == {"04", "35", "49"}
            and service_rows["weight_share_basis"].eq("service_area_prior").all()
            and service_rows["bg_land_area_coverage_share"].notna().all()
            and service_rows["bg_service_population_coverage_share"].eq(0.0).sum() == 8
            and abs(service_population_value - 163545.0) <= 1e-9
        ),
        "ordinary_resident_responsibility_contract": bool(
            len(ordinary_resident_rows) == 9786
            and ordinary_resident_rows["ori9"].nunique() == 242
            and len(service_population) == 138
            and len(ordinary_eligible_owners) == 137
            and ordinary_eligible_owners
            <= set(ordinary_resident_rows["ori9"].astype(str))
            and service_population["_covered_population"].gt(0.0).all()
            and abs(float(service_population["_covered_population"].sum()) - 984915.0)
            <= 1e-9
            and ordinary_resident_rows[
                "bg_responsibility_population_coverage_share"
            ].notna().all()
            and ordinary_resident_rows["responsibility_fraction_basis"].notna().all()
        ),
        "city_incident_shares": artifact_is_current(
            state_dir / "modeling" / "city_incident_share_surface.parquet",
            _city_dependency_paths(paths, _city_impls(paths)),
        ),
    }
    effective_config = _effective_config_parity(paths=paths, launch=launch)
    checks["effective_config_matches_baseline"] = bool(effective_config["ok"])
    city_exact_point_qa = _validator_city_exact_point_qa(
        source_root=source_root, runtime_root=runtime_root, year=year
    )
    checks["validator_city_exact_point_qa"] = bool(city_exact_point_qa["ok"])
    report = {
        "ok": all(checks.values()),
        "checks": checks,
        "effective_config_parity": effective_config,
        "validator_city_exact_point_qa": city_exact_point_qa,
        "derived_rebuild_plan": {
            "declared_roles": sorted(declared_rebuilt),
            "agency_estimates_current_before_build": agency_estimates_current,
            "controls_current_before_build": controls_current,
            "smoothed_controls_current_before_build": smoothed_controls_artifact_is_current(
                paths, year=year
            ),
            "structural_zero_path": str(structural_zero_path),
            "structural_zero_sha256": (
                _sha256(structural_zero_path) if structural_zero_path.is_file() else None
            ),
        },
        "footprint_scope_inputs": {
            "overlap_custom_footprints": {
                "path": str(paths.repo_root / "configs" / "overlap_custom_footprints.csv"),
                "sha256": _sha256(
                    paths.repo_root / "configs" / "overlap_custom_footprints.csv"
                ),
            },
            "service_wide_footprint_coverage": {
                "path": str(
                    paths.repo_root / "configs" / "service_wide_footprint_coverage.csv"
                ),
                "sha256": _sha256(
                    paths.repo_root / "configs" / "service_wide_footprint_coverage.csv"
                ),
            },
            "ordinary_resident_footprint_coverage": {
                "path": str(
                    paths.repo_root
                    / "configs"
                    / "overlap_custom_footprint_resident_coverage.csv"
                ),
                "sha256": _sha256(
                    paths.repo_root
                    / "configs"
                    / "overlap_custom_footprint_resident_coverage.csv"
                ),
            },
            "service_population_2020": service_population_value,
            "eligible_population_target_count": len(service_population),
            "eligible_ordinary_owner_count": len(ordinary_eligible_owners),
            "eligible_population_total_2020": float(
                service_population["_covered_population"].sum()
            ),
        },
        "absent_source_features": bindings.get("absent_source_features", []),
        "launch_command": launch["full_candidate_command"],
    }
    report_path = runtime_root / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
