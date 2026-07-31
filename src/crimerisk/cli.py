from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from crimerisk.paths import get_paths
from crimerisk.allocation import (
    AllocationBuildConfig,
    DEFAULT_MODEL_SURFACE_EXCLUDE_FEATURE_POLICY_CLASSES,
    DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH,
    DEFAULT_MODEL_SURFACE_PRIOR_ANCHOR,
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    DEFAULT_RESIDUAL_FEATURE_POLICY_PATH,
    DEFAULT_RESIDUAL_TRANSFER_TAU_BY_OFFENSE,
    promoted_next_phase_allocator_preflight,
    promoted_next_phase_allocator_required_paths,
    write_v2_outputs,
)
from crimerisk.candidates import (
    candidate_run_manifest_metadata,
    resolve_candidate_output_run,
)
from crimerisk.city_shares import (
    CityIncidentShareBuildConfig,
    get_v2_city_incident_input_root,
    promote_v2_city_incident_inputs,
    write_v2_city_incident_shares,
)
from crimerisk.controls import ControlBuildConfig, write_v2_controls
from crimerisk.geometry import GeometryBuildConfig, write_v2_geometry
from crimerisk.local_publications import (
    get_v2_local_publication_input_path,
    promote_v2_local_publication_inputs,
)
from crimerisk.observations import ObservationBuildConfig, write_v2_observations
from crimerisk.reporting_regimes import ReportingRegimeBuildConfig, write_v2_reporting_regimes
from crimerisk.reference_layers import (
    ReferenceInputPromotionConfig,
    ReferenceLayerBuildConfig,
    promote_v2_reference_inputs,
    write_v2_reference_layers,
)
from crimerisk.state_publications import (
    get_v2_state_publication_input_path,
    write_v2_state_publication_inputs,
)
from crimerisk.stage_locks import StageLockError
from crimerisk.reference import write_agency_master, write_input_manifest
from crimerisk.jurisdiction_review import write_jurisdiction_review


def _parse_residual_transfer_tau(values: list[str]) -> tuple[tuple[str, float], ...]:
    tau_by_offense = {offense: float(tau) for offense, tau in DEFAULT_RESIDUAL_TRANSFER_TAU_BY_OFFENSE}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--residual-transfer-tau values must use offense=value")
        offense, raw_tau = value.split("=", 1)
        offense = offense.strip()
        if offense not in tau_by_offense:
            raise argparse.ArgumentTypeError(f"unknown offense in --residual-transfer-tau: {offense!r}")
        tau = float(raw_tau)
        if tau < 0.0 or tau > 1.0:
            raise argparse.ArgumentTypeError("--residual-transfer-tau values must be in [0, 1]")
        tau_by_offense[offense] = tau
    return tuple((offense, tau_by_offense[offense]) for offense, _tau in DEFAULT_RESIDUAL_TRANSFER_TAU_BY_OFFENSE)


def _release_output_default_paths(year: int) -> dict[str, Path]:
    """Final release/candidate output paths for a build year (the `{year}` pattern used
    for intermediates in _cmd_v2_build_release, extended to the four final outputs)."""
    return {
        "bg_ags_core_out": Path(f"state/output/crimerisk_block_group_{int(year)}_ags_core.parquet"),
        "tract_ags_core_out": Path(f"state/output/crimerisk_tract_{int(year)}_ags_core.parquet"),
        "bg_fbi_out": Path(f"state/output/crimerisk_block_group_{int(year)}_fbi_calibrated.parquet"),
        "tract_fbi_out": Path(f"state/output/crimerisk_tract_{int(year)}_fbi_calibrated.parquet"),
    }


def _resolve_release_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    defaults = _release_output_default_paths(int(args.year))
    return {
        key: Path(getattr(args, key)) if getattr(args, key) else default
        for key, default in defaults.items()
    }


def _cmd_v2_input_manifest(args: argparse.Namespace) -> int:
    paths = get_paths()
    out_path = Path(args.out)
    write_input_manifest(paths=paths, out_path=out_path)
    print(f"Wrote {out_path}")
    return 0


def _cmd_v2_agency_master(args: argparse.Namespace) -> int:
    paths = get_paths()
    out_path = Path(args.out)
    write_agency_master(
        paths=paths,
        out_path=out_path,
        year_start=args.year_start,
        year_end=args.year_end,
    )
    print(f"Wrote {out_path}")
    return 0


def _cmd_v2_jurisdiction_review(args: argparse.Namespace) -> int:
    paths = get_paths()
    out_dir = Path(args.out_dir)
    agency_master_path = Path(args.agency_master) if args.agency_master else None
    written = write_jurisdiction_review(
        paths=paths,
        out_dir=out_dir,
        agency_master_path=agency_master_path,
        year_start=args.year_start,
        year_end=args.year_end,
    )
    for name, out_path in written.items():
        print(f"{name}: {out_path}")
    return 0


def _cmd_v2_build_observations(args: argparse.Namespace) -> int:
    paths = get_paths()
    summary = write_v2_observations(
        paths=paths,
        agency_out_path=Path(args.agency_out),
        jurisdiction_out_path=Path(args.jurisdiction_out),
        config=ObservationBuildConfig(
            year_start=args.year_start,
            year_end=args.year_end,
            local_publication_input_path=Path(args.local_publication_input_path),
            state_publication_input_path=Path(args.state_publication_input_path),
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


def _cmd_v2_build_controls(args: argparse.Namespace) -> int:
    paths = get_paths()
    jurisdiction_out_path = (
        Path(args.jurisdiction_out)
        if args.jurisdiction_out
        else Path(f"state/controls/jurisdiction_controls_{int(args.year)}.parquet")
    )
    state_out, jurisdiction_out, jurisdiction_year_estimates_out = write_v2_controls(
        paths=paths,
        state_out_path=Path(args.state_out),
        jurisdiction_out_path=jurisdiction_out_path,
        jurisdiction_year_estimates_out_path=Path(args.jurisdiction_year_estimates_out),
        config=ControlBuildConfig(
            year=int(args.year),
            force_reporting_regimes_rebuild=bool(args.force_reporting_regimes_rebuild),
        ),
    )
    print(f"Wrote {state_out}")
    print(f"Wrote {jurisdiction_out}")
    if jurisdiction_year_estimates_out is not None:
        print(f"Wrote {jurisdiction_year_estimates_out}")
    return 0


def _cmd_v2_build_reporting_regimes(args: argparse.Namespace) -> int:
    paths = get_paths()
    summary = write_v2_reporting_regimes(
        paths=paths,
        out_path=Path(args.out),
        config=ReportingRegimeBuildConfig(
            year_start=args.year_start,
            year_end=args.year_end,
            override_path=Path(args.override_path) if args.override_path else None,
            source_override_path=Path(args.source_override_path) if args.source_override_path else None,
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {Path(args.out)}")
    return 0


def _cmd_v2_build_reference_layers(args: argparse.Namespace) -> int:
    paths = get_paths()
    summary = write_v2_reference_layers(
        paths=paths,
        full_local_out_path=Path(args.full_local_out),
        full_nonlocal_out_path=Path(args.full_nonlocal_out),
        jurisdiction_out_path=Path(args.jurisdiction_out),
        crosswalk_out_path=Path(args.crosswalk_out),
        config=ReferenceLayerBuildConfig(
            year_start=args.year_start,
            year_end=args.year_end,
            agency_master_path=Path(args.agency_master),
            provisional_local_path=Path(args.provisional_local_path),
            resolved_local_tail_path=Path(args.resolved_local_tail_path),
            nonlocal_final_path=Path(args.nonlocal_final_path),
            nonlocal_auto_path=Path(args.nonlocal_auto_path),
            municipal_override_path=Path(args.municipal_override_path) if args.municipal_override_path else None,
            local_override_path=Path(args.local_override_path) if args.local_override_path else None,
            agency_master_supplement_path=Path(args.agency_master_supplement_path)
            if args.agency_master_supplement_path
            else None,
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {Path(args.full_local_out)}")
    print(f"Wrote {Path(args.full_nonlocal_out)}")
    print(f"Wrote {Path(args.jurisdiction_out)}")
    print(f"Wrote {Path(args.crosswalk_out)}")
    return 0


def _cmd_v2_promote_reference_inputs(args: argparse.Namespace) -> int:
    paths = get_paths()
    summary = promote_v2_reference_inputs(
        paths=paths,
        config=ReferenceInputPromotionConfig(
            provisional_local_source_path=Path(args.provisional_local_source_path),
            resolved_local_tail_source_path=Path(args.resolved_local_tail_source_path),
            nonlocal_final_source_path=Path(args.nonlocal_final_source_path),
            nonlocal_auto_source_path=Path(args.nonlocal_auto_source_path),
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


def _cmd_v2_promote_local_publications(args: argparse.Namespace) -> int:
    paths = get_paths()
    summary = promote_v2_local_publication_inputs(
        paths=paths,
        out_path=Path(args.out),
        year_start=args.year_start,
        year_end=args.year_end,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {Path(args.out)}")
    return 0


def _cmd_v2_build_state_publication_inputs(args: argparse.Namespace) -> int:
    paths = get_paths()
    summary = write_v2_state_publication_inputs(
        paths=paths,
        out_path=Path(args.out),
        year=int(args.year),
        force_refresh=bool(args.force_refresh),
        max_workers=int(args.max_workers),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {Path(args.out)}")
    return 0


def _cmd_v2_build_geometry(args: argparse.Namespace) -> int:
    paths = get_paths()
    block_out, block_group_out = write_v2_geometry(
        paths=paths,
        block_out_path=Path(args.block_out),
        block_group_out_path=Path(args.block_group_out),
        config=GeometryBuildConfig(),
        force_rebuild=bool(args.force_rebuild),
    )
    print(f"Wrote {block_out}")
    print(f"Wrote {block_group_out}")
    return 0


def _cmd_v2_build_city_incident_shares(args: argparse.Namespace) -> int:
    paths = get_paths()
    summary = write_v2_city_incident_shares(
        paths=paths,
        out_path=Path(args.out),
        config=CityIncidentShareBuildConfig(
            year_start=int(args.year_start),
            year_end=int(args.year_end),
            force_rebuild=bool(args.force_rebuild),
            force_source_refresh=bool(args.force_source_refresh),
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


def _cmd_v2_promote_city_incident_inputs(args: argparse.Namespace) -> int:
    paths = get_paths()
    summary = promote_v2_city_incident_inputs(
        paths=paths,
        out_root=Path(args.out_root),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


def _cmd_v2_build_outputs(args: argparse.Namespace) -> int:
    paths = get_paths()
    candidate = None
    run_metadata = None
    emit_fbi_calibrated = bool(args.emit_fbi_calibrated)
    resolved_output_paths = _resolve_release_output_paths(args)
    block_group_ags_core_out = resolved_output_paths["bg_ags_core_out"]
    tract_ags_core_out = resolved_output_paths["tract_ags_core_out"]
    block_group_fbi_out = resolved_output_paths["bg_fbi_out"]
    tract_fbi_out = resolved_output_paths["tract_fbi_out"]
    build_manifest_out = Path(args.build_manifest_out) if args.build_manifest_out else None
    if args.candidate_run is not None:
        candidate_run_id = None if args.candidate_run == "auto" else str(args.candidate_run)
        candidate = resolve_candidate_output_run(
            paths=paths,
            year=int(args.year),
            run_id=candidate_run_id,
        )
        emit_fbi_calibrated = True
        block_group_ags_core_out = candidate.block_group_ags_core_out
        tract_ags_core_out = candidate.tract_ags_core_out
        block_group_fbi_out = candidate.block_group_fbi_out
        tract_fbi_out = candidate.tract_fbi_out
        build_manifest_out = candidate.build_manifest_out
        run_metadata = candidate_run_manifest_metadata(
            paths=paths,
            candidate=candidate,
            argv=list(sys.argv),
        )
    residual_exclude_classes = tuple(
        str(value) for value in args.residual_exclude_feature_policy_class
    )
    residual_exclude_classes_by_offense = tuple(
        (str(offense), residual_exclude_classes)
        for offense, _classes in DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE
    )
    summary = write_v2_outputs(
        paths=paths,
        block_group_ags_core_out=block_group_ags_core_out,
        tract_ags_core_out=tract_ags_core_out,
        block_group_fbi_out=block_group_fbi_out if emit_fbi_calibrated else None,
        tract_fbi_out=tract_fbi_out if emit_fbi_calibrated else None,
        build_manifest_out=build_manifest_out,
        run_metadata=run_metadata,
        config=AllocationBuildConfig(
            year=int(args.year),
            force_controls_rebuild=bool(args.force_controls_rebuild),
            force_reporting_regimes_rebuild=bool(args.force_reporting_regimes_rebuild),
            force_geometry_rebuild=bool(args.force_geometry_rebuild),
            force_bg_prior_rebuild=bool(args.force_bg_prior_rebuild),
            force_city_incident_share_rebuild=bool(args.force_city_incident_share_rebuild),
            force_city_incident_source_refresh=bool(args.force_city_incident_source_refresh),
            use_promoted_next_phase_allocator=bool(args.use_promoted_next_phase_allocator),
            residual_training_city_shares_path=(
                Path(args.residual_training_city_shares_path)
                if args.residual_training_city_shares_path
                else None
            ),
            residual_training_exclude_validation_case_types=tuple(
                str(value) for value in args.residual_training_exclude_validation_case_type
            ),
            residual_training_extra_bg_feature_paths=tuple(
                Path(value) for value in args.residual_training_extra_bg_features_path
            ),
            bg_prior_path=Path(args.bg_prior_path) if args.bg_prior_path else None,
            model_surface_prior_anchor=str(args.model_surface_prior_anchor),
            model_surface_feature_policy_path=(
                Path(args.model_surface_feature_policy_path)
                if args.model_surface_feature_policy_path
                else None
            ),
            model_surface_exclude_feature_policy_classes=tuple(
                str(value) for value in args.model_surface_exclude_feature_policy_class
            ),
            residual_feature_policy_path=(
                Path(args.residual_feature_policy_path)
                if args.residual_feature_policy_path
                else None
            ),
            residual_exclude_feature_policy_classes=residual_exclude_classes,
            residual_exclude_feature_policy_classes_by_offense=residual_exclude_classes_by_offense,
            residual_transfer_tau_by_offense=_parse_residual_transfer_tau(
                [str(value) for value in args.residual_transfer_tau]
            ),
            enable_county_anchoring=bool(args.enable_county_anchoring),
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    if candidate is not None:
        print(f"candidate_run_id: {candidate.run_id}")
        print(f"candidate_dir: {candidate.candidate_dir}")
        print(f"candidate_validation_summary_path: {candidate.validation_summary_out}")
    print(f"Wrote {block_group_ags_core_out}")
    print(f"Wrote {tract_ags_core_out}")
    if emit_fbi_calibrated:
        print(f"Wrote {block_group_fbi_out}")
        print(f"Wrote {tract_fbi_out}")
    return 0


def _cmd_v2_build_release(args: argparse.Namespace) -> int:
    paths = get_paths()

    reference_promotion = promote_v2_reference_inputs(paths=paths)
    print("reference_inputs_promoted:")
    for key, value in reference_promotion.items():
        print(f"  {key}: {value}")

    reference_summary = write_v2_reference_layers(
        paths=paths,
        full_local_out_path=paths.state_dir / "reference" / "local_agency_resolved_full.parquet",
        full_nonlocal_out_path=paths.state_dir / "reference" / "nonlocal_agency_resolved_full.parquet",
        jurisdiction_out_path=paths.state_dir / "reference" / "jurisdiction_master.parquet",
        crosswalk_out_path=paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        config=ReferenceLayerBuildConfig(
            year_start=2018,
            year_end=int(args.year),
            agency_master_path=paths.state_dir / "reference" / "agency_master.parquet",
        ),
    )
    print("reference_layers_built:")
    for key, value in reference_summary.items():
        print(f"  {key}: {value}")

    state_publication_summary = write_v2_state_publication_inputs(
        paths=paths,
        out_path=get_v2_state_publication_input_path(paths),
        year=int(args.year),
        force_refresh=bool(args.force_state_publication_refresh),
        max_workers=int(args.max_workers),
    )
    print("state_publication_inputs_built:")
    for key, value in state_publication_summary.items():
        print(f"  {key}: {value}")

    local_publication_summary = promote_v2_local_publication_inputs(
        paths=paths,
        out_path=get_v2_local_publication_input_path(paths),
        year_start=2018,
        year_end=int(args.year),
    )
    print("local_publications_promoted:")
    for key, value in local_publication_summary.items():
        print(f"  {key}: {value}")

    observation_summary = write_v2_observations(
        paths=paths,
        agency_out_path=paths.state_dir / "observations" / "agency_year_observations.parquet",
        jurisdiction_out_path=paths.state_dir / "observations" / "jurisdiction_year_observations.parquet",
        config=ObservationBuildConfig(
            year_start=2018,
            year_end=int(args.year),
            local_publication_input_path=get_v2_local_publication_input_path(paths),
            state_publication_input_path=get_v2_state_publication_input_path(paths),
        ),
    )
    print("observations_built:")
    for key, value in observation_summary.items():
        print(f"  {key}: {value}")

    city_promotion_summary = promote_v2_city_incident_inputs(
        paths=paths,
        out_root=get_v2_city_incident_input_root(paths),
    )
    print("city_incident_inputs_promoted:")
    for key, value in city_promotion_summary.items():
        print(f"  {key}: {value}")

    reporting_summary = write_v2_reporting_regimes(
        paths=paths,
        out_path=paths.state_dir / "modeling" / "agency_year_reporting_regimes.parquet",
        config=ReportingRegimeBuildConfig(
            year_start=2018,
            year_end=int(args.year),
        ),
    )
    print("reporting_regimes_built:")
    for key, value in reporting_summary.items():
        print(f"  {key}: {value}")

    state_control_out, jurisdiction_control_out, jurisdiction_year_estimates_out = write_v2_controls(
        paths=paths,
        state_out_path=paths.state_dir / "controls" / "state_control_comparison.parquet",
        jurisdiction_out_path=paths.state_dir / "controls" / f"jurisdiction_controls_{int(args.year)}.parquet",
        jurisdiction_year_estimates_out_path=paths.state_dir / "controls" / "jurisdiction_year_estimates.parquet",
        config=ControlBuildConfig(
            year=int(args.year),
            force_reporting_regimes_rebuild=False,
        ),
    )
    print("controls_built:")
    print(f"  state_out: {state_control_out}")
    print(f"  jurisdiction_out: {jurisdiction_control_out}")
    if jurisdiction_year_estimates_out is not None:
        print(f"  jurisdiction_year_estimates_out: {jurisdiction_year_estimates_out}")

    block_out, block_group_out = write_v2_geometry(
        paths=paths,
        block_out_path=paths.state_dir / "geometry" / "block_to_jurisdiction_crosswalk.parquet",
        block_group_out_path=paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        config=GeometryBuildConfig(),
        force_rebuild=True,
    )
    print("geometry_built:")
    print(f"  block_out: {block_out}")
    print(f"  block_group_out: {block_group_out}")

    city_share_summary = write_v2_city_incident_shares(
        paths=paths,
        out_path=paths.state_dir / "modeling" / "city_incident_share_surface.parquet",
        config=CityIncidentShareBuildConfig(
            year_start=2018,
            year_end=int(args.year),
            force_rebuild=True,
            force_source_refresh=bool(args.force_city_incident_source_refresh),
        ),
    )
    print("city_incident_shares_built:")
    for key, value in city_share_summary.items():
        print(f"  {key}: {value}")

    resolved_output_paths = _resolve_release_output_paths(args)
    output_summary = write_v2_outputs(
        paths=paths,
        block_group_ags_core_out=resolved_output_paths["bg_ags_core_out"],
        tract_ags_core_out=resolved_output_paths["tract_ags_core_out"],
        block_group_fbi_out=resolved_output_paths["bg_fbi_out"] if args.emit_fbi_calibrated else None,
        tract_fbi_out=resolved_output_paths["tract_fbi_out"] if args.emit_fbi_calibrated else None,
        build_manifest_out=Path(args.build_manifest_out) if args.build_manifest_out else None,
        config=AllocationBuildConfig(
            year=int(args.year),
            force_controls_rebuild=False,
            force_reporting_regimes_rebuild=False,
            force_geometry_rebuild=False,
            force_bg_prior_rebuild=True,
            force_city_incident_share_rebuild=False,
            force_city_incident_source_refresh=False,
            use_promoted_next_phase_allocator=bool(args.use_promoted_next_phase_allocator),
        ),
    )
    print("release_outputs_built:")
    for key, value in output_summary.items():
        print(f"  {key}: {value}")
    print(f"Wrote {resolved_output_paths['bg_ags_core_out']}")
    print(f"Wrote {resolved_output_paths['tract_ags_core_out']}")
    if args.emit_fbi_calibrated:
        print(f"Wrote {resolved_output_paths['bg_fbi_out']}")
        print(f"Wrote {resolved_output_paths['tract_fbi_out']}")
    return 0


def _cmd_v2_benchmark_suite(args: argparse.Namespace) -> int:
    script_root = Path(__file__).resolve().parents[2] / "scripts" / "diagnostics"
    benchmark_scripts = (
        "benchmark_models.py",
        "benchmark_city_shares.py",
        "benchmark_city_calibration.py",
        "benchmark_city_residuals.py",
    )
    missing_scripts = [name for name in benchmark_scripts if not (script_root / name).exists()]
    if missing_scripts:
        print(
            "benchmark-suite requires the full development workspace; the benchmark "
            "scripts are not included in the submission package:"
        )
        for name in missing_scripts:
            print(f"  missing: scripts/diagnostics/{name}")
        print(
            "Compact benchmark outputs are already included under validation/ and "
            "materials/tables/."
        )
        return 1
    paths = get_paths()
    year = int(args.year)

    def _year_default(value: str | None, template: str) -> Path:
        return Path(value) if value else Path(template.format(year=year))

    use_promoted_residual = bool(args.use_promoted_next_phase_allocator)
    promoted_paths = promoted_next_phase_allocator_required_paths(paths, year=year)
    if use_promoted_residual:
        preflight = promoted_next_phase_allocator_preflight(paths, year=year)
        if not bool(preflight.get("ready")):
            print(
                "benchmark-suite promoted residual benchmark requires promoted next-phase allocator inputs; "
                "missing:"
            )
            for missing_path in preflight.get("missing_required_paths", []):
                print(f"  missing: {missing_path}")
            print("Pass --no-promoted-next-phase-allocator to run the legacy residual benchmark.")
            return 1
    city_residual_diagnostics_out = _year_default(
        args.city_residual_diagnostics_out,
        "state/modeling/next_phase_city_residual_benchmark_overture_core_{year}.parquet"
        if use_promoted_residual
        else "state/modeling/city_residual_benchmark_{year}.parquet",
    )
    city_residual_summary_json_out = _year_default(
        args.city_residual_summary_json_out,
        "state/modeling/next_phase_city_residual_benchmark_overture_core_{year}.json"
        if use_promoted_residual
        else "state/modeling/city_residual_benchmark_{year}.json",
    )
    city_residual_summary_csv_out = Path(args.city_residual_summary_csv_out) if args.city_residual_summary_csv_out else (
        Path("materials/tables/next_phase_city_residual_overture_core_summary.csv")
        if use_promoted_residual
        else None
    )
    residual_command = [
        sys.executable,
        str(script_root / "benchmark_city_residuals.py"),
        "--year",
        str(int(args.year)),
        "--diagnostics-out",
        str(city_residual_diagnostics_out),
        "--summary-json-out",
        str(city_residual_summary_json_out),
    ]
    if city_residual_summary_csv_out is not None:
        residual_command.extend(["--summary-csv-out", str(city_residual_summary_csv_out)])
    if use_promoted_residual:
        residual_command.extend(
            [
                "--city-shares-path",
                str(promoted_paths["residual_training_city_shares"]),
                "--exclude-validation-case-type",
                "suburban_county_validation_case",
                "--extra-bg-features-path",
                str(promoted_paths["overture_places_bg_features"]),
                "--extra-bg-features-path",
                str(promoted_paths["overture_commercial_core_bg_features"]),
            ]
        )
    commands = [
        [
            sys.executable,
            str(script_root / "benchmark_models.py"),
            "--year",
            str(year),
            "--family-ladder-out",
            str(_year_default(args.family_ladder_out, "state/modeling/jurisdiction_model_family_ladder_{year}.parquet")),
            "--family-ladder-json-out",
            str(_year_default(args.family_ladder_json_out, "state/modeling/jurisdiction_model_family_ladder_{year}.json")),
            "--provenance-sensitivity-out",
            str(_year_default(args.provenance_sensitivity_out, "state/modeling/jurisdiction_model_provenance_sensitivity_{year}.parquet")),
            "--provenance-sensitivity-json-out",
            str(_year_default(args.provenance_sensitivity_json_out, "state/modeling/jurisdiction_model_provenance_sensitivity_{year}.json")),
            "--feature-ablation-out",
            str(_year_default(args.feature_ablation_out, "state/modeling/jurisdiction_model_feature_ablation_{year}.parquet")),
            "--feature-ablation-json-out",
            str(_year_default(args.feature_ablation_json_out, "state/modeling/jurisdiction_model_feature_ablation_{year}.json")),
            "--sparse-pooling-out",
            str(_year_default(args.sparse_pooling_out, "state/modeling/jurisdiction_model_sparse_pooling_{year}.parquet")),
            "--sparse-pooling-json-out",
            str(_year_default(args.sparse_pooling_json_out, "state/modeling/jurisdiction_model_sparse_pooling_{year}.json")),
            "--transit-sensitivity-out",
            str(_year_default(args.transit_sensitivity_out, "state/modeling/jurisdiction_model_transit_sensitivity_{year}.parquet")),
            "--transit-sensitivity-json-out",
            str(_year_default(args.transit_sensitivity_json_out, "state/modeling/jurisdiction_model_transit_sensitivity_{year}.json")),
        ],
        [
            sys.executable,
            str(script_root / "benchmark_city_shares.py"),
            "--year",
            str(year),
            "--diagnostics-out",
            str(_year_default(args.city_share_diagnostics_out, "state/modeling/city_share_benchmark_{year}.parquet")),
            "--summary-json-out",
            str(_year_default(args.city_share_summary_json_out, "state/modeling/city_share_benchmark_{year}.json")),
            "--baseline-ladder-out",
            str(_year_default(args.city_share_baseline_ladder_out, "state/modeling/city_share_baseline_ladder_{year}.parquet")),
            "--baseline-ladder-json-out",
            str(_year_default(args.city_share_baseline_ladder_json_out, "state/modeling/city_share_baseline_ladder_{year}.json")),
        ],
        [
            sys.executable,
            str(script_root / "benchmark_city_calibration.py"),
            "--year",
            str(year),
            "--diagnostics-out",
            str(_year_default(args.city_calibration_diagnostics_out, "state/modeling/city_calibration_benchmark_{year}.parquet")),
            "--summary-json-out",
            str(_year_default(args.city_calibration_summary_json_out, "state/modeling/city_calibration_benchmark_{year}.json")),
        ],
        residual_command,
    ]
    transit_features_path = Path(args.transit_features_path) if args.transit_features_path else None
    if transit_features_path is not None and transit_features_path.exists():
        commands[0].extend(
            [
                "--transit-features-path",
                str(transit_features_path),
            ]
        )
    for cmd in commands:
        completed = subprocess.run(cmd, check=False)
        if int(completed.returncode) != 0:
            return int(completed.returncode)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crimerisk",
        description="Active build and maintenance commands for the current CrimeRisk product surface.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p_manifest = sub.add_parser(
        "build-input-manifest",
        help="Write the raw-input manifest.",
    )
    p_manifest.add_argument(
        "--out",
        type=str,
        default="state/reference/input_manifest.json",
    )
    p_manifest.set_defaults(func=_cmd_v2_input_manifest)

    p_agency = sub.add_parser(
        "build-agency-master",
        help="Build the agency master from SRS + NIBRS.",
    )
    p_agency.add_argument("--year-start", type=int, default=2018)
    p_agency.add_argument("--year-end", type=int, default=2024)
    p_agency.add_argument(
        "--out",
        type=str,
        default="state/reference/agency_master.parquet",
    )
    p_agency.set_defaults(func=_cmd_v2_agency_master)

    p_review = sub.add_parser(
        "build-jurisdiction-review",
        help="Build provisional local matches and analyst review packs for jurisdiction mapping.",
    )
    p_review.add_argument("--year-start", type=int, default=2018)
    p_review.add_argument("--year-end", type=int, default=2024)
    p_review.add_argument(
        "--agency-master",
        type=str,
        default="state/reference/agency_master.parquet",
        help="Existing agency master parquet. Rebuilt on the fly if missing.",
    )
    p_review.add_argument(
        "--out-dir",
        type=str,
        default="state/review/queues/local_resolution",
    )
    p_review.set_defaults(func=_cmd_v2_jurisdiction_review)

    p_obs = sub.add_parser(
        "build-observations",
        help="Build source-separated agency and jurisdiction observations.",
    )
    p_obs.add_argument("--year-start", type=int, default=2018)
    p_obs.add_argument("--year-end", type=int, default=2024)
    p_obs.add_argument(
        "--agency-out",
        type=str,
        default="state/observations/agency_year_observations.parquet",
    )
    p_obs.add_argument(
        "--jurisdiction-out",
        type=str,
        default="state/observations/jurisdiction_year_observations.parquet",
    )
    p_obs.add_argument(
        "--local-publication-input-path",
        type=str,
        default=str(get_v2_local_publication_input_path(get_paths())),
    )
    p_obs.add_argument(
        "--state-publication-input-path",
        type=str,
        default=str(get_v2_state_publication_input_path(get_paths())),
    )
    p_obs.set_defaults(func=_cmd_v2_build_observations)

    p_promote_local_pub = sub.add_parser(
        "promote-local-publications",
        help="Promote reviewed municipal packet publication extracts into the canonical local-publication input surface.",
    )
    p_promote_local_pub.add_argument("--year-start", type=int, default=2018)
    p_promote_local_pub.add_argument("--year-end", type=int, default=2024)
    p_promote_local_pub.add_argument(
        "--out",
        type=str,
        default=str(get_v2_local_publication_input_path(get_paths())),
    )
    p_promote_local_pub.set_defaults(func=_cmd_v2_promote_local_publications)

    p_state_pub = sub.add_parser(
        "build-state-publication-inputs",
        help="Build the canonical state-publication annual input surface used by build-observations.",
    )
    p_state_pub.add_argument("--year", type=int, default=2024)
    p_state_pub.add_argument(
        "--out",
        type=str,
        default=str(get_v2_state_publication_input_path(get_paths())),
    )
    p_state_pub.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refresh raw state-publication source caches while rebuilding the canonical input surface.",
    )
    p_state_pub.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum worker threads for Mississippi TOPS raw extraction refresh.",
    )
    p_state_pub.set_defaults(func=_cmd_v2_build_state_publication_inputs)

    p_ref = sub.add_parser(
        "build-reference-layers",
        help="Build resolved local/nonlocal reference layers, jurisdiction master, and agency crosswalk.",
    )
    p_ref.add_argument("--year-start", type=int, default=2018)
    p_ref.add_argument("--year-end", type=int, default=2024)
    p_ref.add_argument(
        "--agency-master",
        type=str,
        default="state/reference/agency_master.parquet",
    )
    p_ref.add_argument(
        "--agency-master-supplement-path",
        type=str,
        default="configs/agency_master_supplement.csv",
    )
    p_ref.add_argument(
        "--provisional-local-path",
        type=str,
        default="state/reference/inputs/provisional_local_agency_matches.parquet",
    )
    p_ref.add_argument(
        "--resolved-local-tail-path",
        type=str,
        default="state/reference/inputs/local_queue_resolved_final.parquet",
    )
    p_ref.add_argument(
        "--nonlocal-final-path",
        type=str,
        default="state/reference/inputs/nonmunicipal_special_resolved_final.parquet",
    )
    p_ref.add_argument(
        "--nonlocal-auto-path",
        type=str,
        default="state/reference/inputs/nonmunicipal_auto_defaults.parquet",
    )
    p_ref.add_argument(
        "--municipal-override-path",
        type=str,
        default="configs/municipal_geometry_overrides.csv",
    )
    p_ref.add_argument(
        "--local-override-path",
        type=str,
        default="configs/local_resolution_overrides.csv",
    )
    p_ref.add_argument(
        "--full-local-out",
        type=str,
        default="state/reference/local_agency_resolved_full.parquet",
    )
    p_ref.add_argument(
        "--full-nonlocal-out",
        type=str,
        default="state/reference/nonlocal_agency_resolved_full.parquet",
    )
    p_ref.add_argument(
        "--jurisdiction-out",
        type=str,
        default="state/reference/jurisdiction_master.parquet",
    )
    p_ref.add_argument(
        "--crosswalk-out",
        type=str,
        default="state/reference/agency_to_jurisdiction_crosswalk.parquet",
    )
    p_ref.set_defaults(func=_cmd_v2_build_reference_layers)

    p_promote_ref = sub.add_parser(
        "promote-reference-inputs",
        help="Promote reviewed local-resolution queue outputs into the canonical reference input surface.",
    )
    p_promote_ref.add_argument(
        "--provisional-local-source-path",
        type=str,
        default="state/review/queues/local_resolution/provisional_local_agency_matches.parquet",
    )
    p_promote_ref.add_argument(
        "--resolved-local-tail-source-path",
        type=str,
        default="state/review/queues/local_resolution/local_queue_resolved_final.parquet",
    )
    p_promote_ref.add_argument(
        "--nonlocal-final-source-path",
        type=str,
        default="state/review/queues/local_resolution/nonmunicipal_special_resolved_final.parquet",
    )
    p_promote_ref.add_argument(
        "--nonlocal-auto-source-path",
        type=str,
        default="state/review/queues/local_resolution/nonmunicipal_auto_defaults.parquet",
    )
    p_promote_ref.set_defaults(func=_cmd_v2_promote_reference_inputs)

    p_regimes = sub.add_parser(
        "build-reporting-regimes",
        help="Build agency-year reporting-regime classifications.",
    )
    p_regimes.add_argument("--year-start", type=int, default=2018)
    p_regimes.add_argument("--year-end", type=int, default=2024)
    p_regimes.add_argument(
        "--override-path",
        type=str,
        default="configs/reporting_regime_overrides.csv",
    )
    p_regimes.add_argument(
        "--source-override-path",
        type=str,
        default="configs/source_preference_overrides.csv",
    )
    p_regimes.add_argument(
        "--out",
        type=str,
        default="state/modeling/agency_year_reporting_regimes.parquet",
    )
    p_regimes.set_defaults(func=_cmd_v2_build_reporting_regimes)

    p_ctrl = sub.add_parser(
        "build-controls",
        help="Build 2024 jurisdiction controls and state control comparison tables.",
    )
    p_ctrl.add_argument("--year", type=int, default=2024)
    p_ctrl.add_argument(
        "--state-out",
        type=str,
        default="state/controls/state_control_comparison.parquet",
    )
    p_ctrl.add_argument(
        "--jurisdiction-out",
        type=str,
        default=None,
        help="Defaults to state/controls/jurisdiction_controls_<year>.parquet.",
    )
    p_ctrl.add_argument(
        "--jurisdiction-year-estimates-out",
        type=str,
        default="state/controls/jurisdiction_year_estimates.parquet",
    )
    p_ctrl.add_argument(
        "--force-reporting-regimes-rebuild",
        action="store_true",
        help="Rebuild agency_year_reporting_regimes.parquet instead of reusing the cached artifact.",
    )
    p_ctrl.set_defaults(func=_cmd_v2_build_controls)

    p_geo = sub.add_parser(
        "build-geometry",
        help="Build block and block-group jurisdiction crosswalks.",
    )
    p_geo.add_argument(
        "--block-out",
        type=str,
        default="state/geometry/block_to_jurisdiction_crosswalk.parquet",
    )
    p_geo.add_argument(
        "--block-group-out",
        type=str,
        default="state/geometry/block_group_to_jurisdiction_crosswalk.parquet",
    )
    p_geo.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild per-state block assignment caches instead of reusing the cached blocks_by_state artifacts.",
    )
    p_geo.set_defaults(func=_cmd_v2_build_geometry)

    p_city = sub.add_parser(
        "promote-city-incident-inputs",
        help="Promote reviewed city packet support files into the canonical city-incident input surface.",
    )
    p_city.add_argument(
        "--out-root",
        type=str,
        default=str(get_v2_city_incident_input_root(get_paths())),
    )
    p_city.set_defaults(func=_cmd_v2_promote_city_incident_inputs)

    p_city = sub.add_parser(
        "build-city-incident-shares",
        help="Build the canonical city incident share surface used by build-outputs.",
    )
    p_city.add_argument("--year-start", type=int, default=2018)
    p_city.add_argument("--year-end", type=int, default=2024)
    p_city.add_argument(
        "--out",
        type=str,
        default="state/modeling/city_incident_share_surface.parquet",
    )
    p_city.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the city incident share surface instead of reusing the cached artifact.",
    )
    p_city.add_argument(
        "--force-source-refresh",
        action="store_true",
        help="Refresh raw city-source caches before rebuilding the city incident share surface.",
    )
    p_city.set_defaults(func=_cmd_v2_build_city_incident_shares)

    p_out = sub.add_parser(
        "build-outputs",
        help="Build ags_core outputs, with optional fbi_calibrated derivative outputs.",
    )
    p_out.add_argument("--year", type=int, default=2024)
    p_out.add_argument(
        "--bg-ags-core-out",
        type=str,
        default=None,
        help="Defaults to state/output/crimerisk_block_group_<year>_ags_core.parquet.",
    )
    p_out.add_argument(
        "--tract-ags-core-out",
        type=str,
        default=None,
        help="Defaults to state/output/crimerisk_tract_<year>_ags_core.parquet.",
    )
    p_out.add_argument(
        "--bg-fbi-out",
        type=str,
        default=None,
        help="Defaults to state/output/crimerisk_block_group_<year>_fbi_calibrated.parquet.",
    )
    p_out.add_argument(
        "--tract-fbi-out",
        type=str,
        default=None,
        help="Defaults to state/output/crimerisk_tract_<year>_fbi_calibrated.parquet.",
    )
    p_out.add_argument(
        "--force-controls-rebuild",
        action="store_true",
        help="Rebuild jurisdiction controls and jurisdiction_year_estimates before allocating outputs.",
    )
    p_out.add_argument(
        "--force-reporting-regimes-rebuild",
        action="store_true",
        help="When rebuilding controls from build-outputs, also rebuild agency_year_reporting_regimes.parquet.",
    )
    p_out.add_argument(
        "--force-geometry-rebuild",
        action="store_true",
        help="Rebuild block and block-group jurisdiction crosswalks before allocating outputs.",
    )
    p_out.add_argument(
        "--force-bg-prior-rebuild",
        action="store_true",
        help="Rebuild the cached bg_prior_long modeling surface before allocating outputs.",
    )
    p_out.add_argument(
        "--bg-prior-path",
        type=str,
        default=None,
        help=(
            "Optional cached bg_prior_long parquet to read or rebuild. Defaults to the production "
            "arm-B prior path when the model-surface config matches arm B."
        ),
    )
    p_out.add_argument(
        "--model-surface-prior-anchor",
        type=str,
        default=DEFAULT_MODEL_SURFACE_PRIOR_ANCHOR,
        choices=("resident_population", "offense_denominator"),
        help="Model-surface prior anchor used when generating bg_prior_long.",
    )
    p_out.add_argument(
        "--model-surface-feature-policy-path",
        type=str,
        default=str(DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH),
        help="Feature transfer policy parquet used by the production model surface.",
    )
    p_out.add_argument(
        "--model-surface-exclude-feature-policy-class",
        action="append",
        default=list(DEFAULT_MODEL_SURFACE_EXCLUDE_FEATURE_POLICY_CLASSES),
        help=(
            "Feature-transfer final_class to exclude from the production model surface. "
            "May be repeated."
        ),
    )
    p_out.add_argument(
        "--residual-feature-policy-path",
        type=str,
        default=str(DEFAULT_RESIDUAL_FEATURE_POLICY_PATH),
        help="Feature transfer policy parquet used by the within-allocation residual model.",
    )
    p_out.add_argument(
        "--residual-exclude-feature-policy-class",
        action="append",
        default=list(DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES),
        help=(
            "Feature-transfer final_class to exclude from within-allocation residual features. "
            "May be repeated."
        ),
    )
    p_out.add_argument(
        "--residual-transfer-tau",
        action="append",
        default=[],
        metavar="OFFENSE=TAU",
        help=(
            "Override uncovered-area residual transfer tau for one offense. "
            "Tau must be in [0, 1]. May be repeated."
        ),
    )
    p_out.add_argument(
        "--force-city-incident-share-rebuild",
        action="store_true",
        help="Rebuild city_incident_share_surface.parquet before allocating outputs.",
    )
    p_out.add_argument(
        "--force-city-incident-source-refresh",
        action="store_true",
        help="Refresh raw city-source caches before rebuilding city_incident_share_surface.parquet.",
    )
    p_out.add_argument(
        "--residual-training-city-shares-path",
        type=str,
        default=None,
        help=(
            "Optional city-share surface used only to train the residual allocator. "
            "Direct city incident overrides still come from city_incident_share_surface.parquet."
        ),
    )
    p_out.add_argument(
        "--residual-training-exclude-validation-case-type",
        action="append",
        default=[],
        help=(
            "validation_case_type value to exclude from the optional residual-training surface. "
            "May be repeated."
        ),
    )
    p_out.add_argument(
        "--residual-training-extra-bg-features-path",
        action="append",
        default=[],
        help=(
            "Optional BG feature parquet used only by the residual allocator model. "
            "May be repeated. It does not alter the jurisdiction-total model surface."
        ),
    )
    p_out.add_argument(
        "--no-promoted-next-phase-allocator",
        action="store_false",
        dest="use_promoted_next_phase_allocator",
        help=(
            "Disable automatic use of the promoted next-phase residual allocator inputs "
            "when those artifacts are present."
        ),
    )
    p_out.add_argument(
        "--emit-fbi-calibrated",
        action="store_true",
        help="Also write the optional fbi_calibrated derivative outputs.",
    )
    p_out.add_argument(
        "--no-county-anchoring",
        action="store_false",
        dest="enable_county_anchoring",
        help=(
            "Disable allocation-local county anchoring for state nonmunicipal remainder "
            "and county-evidenced overlap rows."
        ),
    )
    p_out.add_argument(
        "--candidate-run",
        nargs="?",
        const="auto",
        default=None,
        metavar="RUN_ID",
        help=(
            "Write the full output set to state/candidates/<RUN_ID>/ instead of state/output. "
            "Omit RUN_ID to generate a sortable UTC timestamp plus short git SHA."
        ),
    )
    p_out.add_argument(
        "--build-manifest-out",
        type=str,
        default=None,
        help=(
            "Optional JSON sidecar recording the resolved allocator configuration, key input file "
            "stats, and output file stats."
        ),
    )
    p_out.set_defaults(func=_cmd_v2_build_outputs)

    p_release = sub.add_parser(
        "build-release",
        help="Rebuild the released product from included raw data plus reviewed input surfaces.",
    )
    p_release.add_argument("--year", type=int, default=2024)
    p_release.add_argument(
        "--bg-ags-core-out",
        type=str,
        default=None,
        help="Defaults to state/output/crimerisk_block_group_<year>_ags_core.parquet.",
    )
    p_release.add_argument(
        "--tract-ags-core-out",
        type=str,
        default=None,
        help="Defaults to state/output/crimerisk_tract_<year>_ags_core.parquet.",
    )
    p_release.add_argument(
        "--bg-fbi-out",
        type=str,
        default=None,
        help="Defaults to state/output/crimerisk_block_group_<year>_fbi_calibrated.parquet.",
    )
    p_release.add_argument(
        "--tract-fbi-out",
        type=str,
        default=None,
        help="Defaults to state/output/crimerisk_tract_<year>_fbi_calibrated.parquet.",
    )
    p_release.add_argument(
        "--force-state-publication-refresh",
        action="store_true",
        help="Refresh raw state-publication source caches while rebuilding the state-publication input surface.",
    )
    p_release.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum worker threads for Mississippi TOPS raw extraction refresh.",
    )
    p_release.add_argument(
        "--force-city-incident-source-refresh",
        action="store_true",
        help="Refresh raw city-source caches while rebuilding city shares.",
    )
    p_release.add_argument(
        "--no-promoted-next-phase-allocator",
        action="store_false",
        dest="use_promoted_next_phase_allocator",
        help=(
            "Disable automatic use of the promoted next-phase residual allocator inputs "
            "when those artifacts are present."
        ),
    )
    p_release.add_argument(
        "--emit-fbi-calibrated",
        action="store_true",
        help="Also write the optional fbi_calibrated derivative outputs.",
    )
    p_release.add_argument(
        "--build-manifest-out",
        type=str,
        default=None,
        help=(
            "Optional JSON sidecar recording the resolved allocator configuration, key input file "
            "stats, and output file stats."
        ),
    )
    p_release.set_defaults(func=_cmd_v2_build_release)

    p_bench = sub.add_parser(
        "benchmark-suite",
        help="Run the canonical benchmark suite, including the promoted leave-one-city-out residual allocator benchmark by default.",
    )
    p_bench.add_argument("--year", type=int, default=2024)
    p_bench.add_argument(
        "--family-ladder-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_family_ladder_<year>.parquet.",
    )
    p_bench.add_argument(
        "--family-ladder-json-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_family_ladder_<year>.json.",
    )
    p_bench.add_argument(
        "--provenance-sensitivity-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_provenance_sensitivity_<year>.parquet.",
    )
    p_bench.add_argument(
        "--provenance-sensitivity-json-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_provenance_sensitivity_<year>.json.",
    )
    p_bench.add_argument(
        "--feature-ablation-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_feature_ablation_<year>.parquet.",
    )
    p_bench.add_argument(
        "--feature-ablation-json-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_feature_ablation_<year>.json.",
    )
    p_bench.add_argument(
        "--sparse-pooling-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_sparse_pooling_<year>.parquet.",
    )
    p_bench.add_argument(
        "--sparse-pooling-json-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_sparse_pooling_<year>.json.",
    )
    p_bench.add_argument(
        "--transit-sensitivity-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_transit_sensitivity_<year>.parquet.",
    )
    p_bench.add_argument(
        "--transit-sensitivity-json-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/jurisdiction_model_transit_sensitivity_<year>.json.",
    )
    p_bench.add_argument(
        "--transit-features-path",
        type=str,
        default="data/NTM/parsed/block_group_transit_stops.parquet",
    )
    p_bench.add_argument(
        "--city-share-diagnostics-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/city_share_benchmark_<year>.parquet.",
    )
    p_bench.add_argument(
        "--city-share-summary-json-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/city_share_benchmark_<year>.json.",
    )
    p_bench.add_argument(
        "--city-share-baseline-ladder-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/city_share_baseline_ladder_<year>.parquet.",
    )
    p_bench.add_argument(
        "--city-share-baseline-ladder-json-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/city_share_baseline_ladder_<year>.json.",
    )
    p_bench.add_argument(
        "--city-residual-diagnostics-out",
        type=str,
        default=None,
        help=(
            "Residual benchmark parquet output. Defaults to the promoted next-phase residual "
            "benchmark path unless --no-promoted-next-phase-allocator is supplied."
        ),
    )
    p_bench.add_argument(
        "--city-residual-summary-json-out",
        type=str,
        default=None,
        help=(
            "Residual benchmark JSON output. Defaults to the promoted next-phase residual "
            "benchmark path unless --no-promoted-next-phase-allocator is supplied."
        ),
    )
    p_bench.add_argument(
        "--city-residual-summary-csv-out",
        type=str,
        default=None,
        help=(
            "Optional residual benchmark CSV summary output. Defaults to the promoted next-phase "
            "summary CSV when the promoted residual benchmark is enabled."
        ),
    )
    p_bench.add_argument(
        "--city-calibration-diagnostics-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/city_calibration_benchmark_<year>.parquet.",
    )
    p_bench.add_argument(
        "--city-calibration-summary-json-out",
        type=str,
        default=None,
        help="Defaults to state/modeling/city_calibration_benchmark_<year>.json.",
    )
    p_bench.add_argument(
        "--no-promoted-next-phase-allocator",
        action="store_false",
        dest="use_promoted_next_phase_allocator",
        help=(
            "Run the legacy residual benchmark instead of the promoted next-phase residual "
            "benchmark lane."
        ),
    )
    p_bench.set_defaults(func=_cmd_v2_benchmark_suite)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except StageLockError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
