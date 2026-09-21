"""Prepare a copy-on-write CrimeRisk runtime and an OS-sandboxed launch command.

Mutable state inputs are cloned on the same APFS volume.  Raw data is exposed by
symlink, while the generated sandbox profile denies every write to the canonical
checkout.  Candidate outputs therefore stay under the isolated runtime root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
MUTABLE_STATE_DIRS = (
    "controls",
    "geometry",
    "modeling",
    "observations",
    "raw",
    "reference",
)
CACHE_DIRS = (
    "allocation",
    "cpi",
    "mixture",
    "nibrs_batch_header",
    "roads",
)
MIN_FREE_BYTES = 20 * 1024**3
ESTIMATED_BUILD_PEAK_BYTES = 4 * 1024**3
SOURCE_IDENTITY_SCOPES = (
    "src",
    "configs",
    "scripts/build/exposure_normalizers.py",
    "scripts/diagnostics/validate_release_outputs.py",
    "scripts/release/freeze_baseline.py",
    "scripts/release/preflight_isolated_candidate.py",
    "scripts/release/prepare_isolated_runtime.py",
    "scripts/release/run_isolated_candidate.py",
    "main.py",
    "pyproject.toml",
    "uv.lock",
)
SOURCE_IDENTITY_EXCLUDES = {
    # Standalone evaluator development does not enter the candidate build lineage.
    "src/crimerisk/exact_surface_evaluation.py",
}
ARM_INPUT_PATHS = (
    "state/modeling/bg_prior_long_2025_arm_b.parquet",
    "state/modeling/next_phase_validation_city_incident_share_surface_2025.parquet",
    "state/modeling/feature_transfer_policy_2025.parquet",
    "state/modeling/bg_mixture_experts_2025.parquet",
    "state/modeling/bg_exposure_normalizers_2025.parquet",
    "state/modeling/city_incident_share_surface.parquet",
    "state/modeling/city_residual_prediction_surface_2025.parquet",
    "state/controls/jurisdiction_controls_smoothed_2025.parquet",
    "state/controls/state_control_comparison.parquet",
    "state/observations/agency_year_observations.parquet",
    "state/geometry/block_group_to_jurisdiction_crosswalk.parquet",
    "state/reference/agency_to_jurisdiction_crosswalk.parquet",
    "state/reference/agency_master.parquet",
    "state/modeling/burglary_tau_calibration_2025.json",
    "state/modeling/murder_tract_posterior_calibration_2025.json",
    "state/modeling/next_phase_measurement_summary_2025.json",
    "state/modeling/dashboard_neighborhood_check_lookup_2025.json",
    "state/modeling/external_surface_availability_2025.json",
    "state/qa/city_feed_exact_point_concentration_2025.csv",
    "data/tiger_bg/parsed/bg_centroids.parquet",
    "data/Census-PopEst-2020-2025/co-est2025-alldata.csv",
    "data/ACS-5yr-2020-2024/parsed/acs_block_groups.parquet",
    "archive/2026-09-cull/configs/city_feed_exact_point_exceptions.csv",
    "configs/mixture_ship_weights_v3.csv",
    "configs/exposure_ensemble_weights_v1.csv",
)
ABSENT_COMPACTED_FEATURE_PATHS = (
    "data/roads/parsed/block_group_road_metrics.parquet",
    "data/HPMS/parsed/block_group_hpms_2024.parquet",
    "data/NCES-EDGE/parsed/block_group_education_anchors_2425.parquet",
    "data/CMS-Hospital-General-Info/parsed/block_group_hospital_anchors.parquet",
    "data/NLCD/parsed/block_group_nlcd_2023.parquet",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(source_root: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_IDENTITY_SCOPES,
        ],
        check=True,
        capture_output=True,
    )
    relative_paths = [
        raw.decode()
        for raw in sorted(filter(None, result.stdout.split(b"\0")))
        if raw.decode() not in SOURCE_IDENTITY_EXCLUDES
    ]
    return _source_identity_from_records(
        source_root, ({"path": relative} for relative in relative_paths)
    )


def _source_identity_from_records(
    source_root: Path, input_records: object
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for input_record in input_records:  # type: ignore[union-attr]
        relative = str(input_record["path"])
        path = source_root / relative
        if not path.is_file():
            raise SystemExit(f"source identity file is missing: {path}")
        digest = _sha256(path)
        size = path.stat().st_size
        records.append({"path": relative, "size_bytes": size, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\n")
    return {
        "algorithm": "sha256(path\\0sha256\\0size\\n)",
        "scopes": list(SOURCE_IDENTITY_SCOPES),
        "file_count": len(records),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }


def _git_metadata(source_root: Path) -> dict[str, object]:
    def capture(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = capture("status", "--porcelain=v1", "--untracked-files=normal") or ""
    tracked = capture("status", "--porcelain=v1", "--untracked-files=no") or ""
    return {
        "head_sha": capture("rev-parse", "HEAD"),
        "short_sha": capture("rev-parse", "--short=12", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status.splitlines()),
        "tracked_dirty": bool(tracked.splitlines()),
        "status_porcelain": status.splitlines(),
    }


def _clone(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"refusing to replace existing runtime path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-cR", str(source), str(destination)], check=True)


def _quote_sandbox(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _write_arm_input_bindings(
    *,
    source_root: Path,
    canonical_root: Path,
    runtime_root: Path,
    excluded_roles: frozenset[str] = frozenset(),
) -> Path:
    records: list[dict[str, object]] = []
    for relative in ARM_INPUT_PATHS:
        if relative in excluded_roles:
            continue
        canonical = canonical_root / relative
        runtime = (
            source_root / relative
            if relative.startswith("configs/")
            else runtime_root / relative
        )
        record: dict[str, object] = {
            "role": relative,
            "canonical_path": str(canonical),
            "runtime_path": str(runtime),
            "canonical_exists": canonical.is_file(),
            "runtime_exists": runtime.is_file(),
        }
        if canonical.is_file() and runtime.is_file():
            canonical_sha = _sha256(canonical)
            runtime_sha = _sha256(runtime)
            record.update(
                {
                    "canonical_sha256": canonical_sha,
                    "runtime_sha256": runtime_sha,
                    "sha256_equal": canonical_sha == runtime_sha,
                    "size_bytes": runtime.stat().st_size,
                }
            )
        else:
            record["sha256_equal"] = False
        records.append(record)
    absent = [
        {
            "path": relative,
            "canonical_exists": (canonical_root / relative).exists(),
            "runtime_exists": (runtime_root / relative).exists(),
            "dependency_role": "compacted_into_frozen_model_inputs_not_selected_directly",
        }
        for relative in ABSENT_COMPACTED_FEATURE_PATHS
    ]
    payload = {
        "contract": "selected_upstream_artifacts_are_byte_identical_apfs_clones",
        "all_selected_inputs_equal": all(bool(row["sha256_equal"]) for row in records),
        "selected_inputs": records,
        "absent_source_features": absent,
    }
    path = runtime_root / "arm_input_bindings.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["all_selected_inputs_equal"]:
        raise SystemExit(f"selected arm input binding failed: {path}")
    return path


def _latest_uncertainty_cache(canonical_root: Path, year: int) -> list[Path]:
    root = canonical_root / "state" / "cache" / "uncertainty"
    candidates = sorted(
        root.glob(f"uncertainty_cells_{year}_*.parquet"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        return []
    selected = candidates[0]
    paths = [selected]
    sidecar = Path(f"{selected}.deps.json")
    if sidecar.exists():
        paths.append(sidecar)
    return paths


def _rebase_dependency_sidecars(
    *, source_root: Path, canonical_root: Path, runtime_root: Path
) -> dict[str, object]:
    """Rebase cloned absolute dependency keys without changing their recorded digests.

    Freshness then verifies runtime clones and the read-only worktree by content. A mismatched
    source or input still invalidates the cache; path relocation alone does not.
    """
    rewrites = 0
    hash_memo: dict[Path, str] = {}
    sidecar_records: list[dict[str, object]] = []

    def file_sha(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in hash_memo:
            hash_memo[resolved] = _sha256(resolved)
        return hash_memo[resolved]

    prefixes = (
        (canonical_root / "state", runtime_root / "state"),
        (canonical_root, source_root),
    )
    for sidecar in sorted((runtime_root / "state").rglob("*.deps.json")):
        original_sha256 = _sha256(sidecar)
        payload = json.loads(sidecar.read_text())
        dependencies = payload.get("dependencies")
        if not isinstance(dependencies, dict):
            continue
        rebased: dict[str, object] = {}
        changed = False
        mappings: list[dict[str, object]] = []
        for raw_path, recorded_digest in dependencies.items():
            path = Path(str(raw_path))
            mapped = path
            # Runtime data entries are symlinks. Path.resolve(), which freshness uses for its
            # dependency keys, therefore remains the canonical data path.
            try:
                path.relative_to(canonical_root / "data")
                is_canonical_data = True
            except ValueError:
                is_canonical_data = False
            if is_canonical_data:
                rebased[str(path)] = recorded_digest
                continue
            for old_root, new_root in prefixes:
                try:
                    relative = path.relative_to(old_root)
                except ValueError:
                    continue
                candidate = new_root / relative
                record: dict[str, object] = {
                    "source_path": str(path),
                    "mapped_path": str(candidate),
                    "recorded_blake2b16": recorded_digest,
                    "source_exists": path.is_file(),
                    "mapped_exists": candidate.is_file(),
                }
                if path.is_file() and candidate.is_file():
                    source_sha256 = file_sha(path)
                    mapped_sha256 = file_sha(candidate)
                    record.update(
                        {
                            "source_sha256": source_sha256,
                            "mapped_sha256": mapped_sha256,
                            "sha256_equal": source_sha256 == mapped_sha256,
                        }
                    )
                    # Only a byte-identical relocation is admissible. A changed dependency stays
                    # keyed to its original path and the normal freshness check will reject reuse.
                    if source_sha256 == mapped_sha256:
                        mapped = candidate
                else:
                    record["sha256_equal"] = False
                mappings.append(record)
                break
            rebased[str(mapped)] = recorded_digest
            changed |= str(mapped) != str(path)
        if changed:
            payload["dependencies"] = rebased
            sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            rewrites += 1
        if mappings:
            sidecar_records.append(
                {
                    "runtime_sidecar": str(sidecar),
                    "canonical_sidecar": str(
                        canonical_root / sidecar.relative_to(runtime_root)
                    ),
                    "original_sha256": original_sha256,
                    "rewritten_sha256": _sha256(sidecar),
                    "changed": changed,
                    "mappings": mappings,
                }
            )
    audit = {
        "contract": "path_relocation_only_byte_identical_targets",
        "rewritten_sidecar_count": rewrites,
        "sidecars_with_relocation_candidates": len(sidecar_records),
        "sidecars": sidecar_records,
    }
    (runtime_root / "dependency_sidecar_rebase.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SCRIPT_REPO_ROOT)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run-id", default="v52-2025")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--rebuild-service-controls",
        action="store_true",
        help="declare that service-scope changes rebuild agency estimates and controls",
    )
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    canonical_root = args.canonical_root.resolve()
    runtime_root = args.runtime_root.resolve()
    if runtime_root.exists() and any(runtime_root.iterdir()):
        raise SystemExit(f"runtime root exists and is not empty: {runtime_root}")
    if canonical_root == runtime_root or canonical_root in runtime_root.parents:
        raise SystemExit("runtime root must be outside the canonical checkout")
    if source_root == runtime_root or source_root in runtime_root.parents:
        raise SystemExit("runtime root must be outside the source checkout")
    for required in (source_root / "src", canonical_root / "data", canonical_root / "state"):
        if not required.exists():
            raise SystemExit(f"missing required path: {required}")
    for candidate_root in (canonical_root, source_root, runtime_root):
        candidate = candidate_root / "state" / "candidates" / args.run_id
        if candidate.exists() or candidate.is_symlink():
            raise SystemExit(f"candidate id already exists: {candidate}")

    disk = shutil.disk_usage(runtime_root.parent if runtime_root.parent.exists() else Path("/"))
    if disk.free - ESTIMATED_BUILD_PEAK_BYTES < MIN_FREE_BYTES:
        raise SystemExit(
            "insufficient free space: "
            f"{disk.free / 1024**3:.1f} GiB free, estimated build peak "
            f"{ESTIMATED_BUILD_PEAK_BYTES / 1024**3:.1f} GiB, "
            f"minimum reserve {MIN_FREE_BYTES / 1024**3:.1f} GiB"
        )

    runtime_root.mkdir(parents=True, exist_ok=True)
    original_source_identity = _source_identity(source_root)
    original_git_metadata = _git_metadata(source_root)
    snapshot_root = runtime_root / "source-snapshot"
    snapshot_root.mkdir()
    for relative in SOURCE_IDENTITY_SCOPES:
        source = source_root / relative
        if source.exists():
            _clone(source, snapshot_root / relative)
    snapshot_identity = _source_identity_from_records(
        snapshot_root, original_source_identity["files"]
    )
    if snapshot_identity["aggregate_sha256"] != original_source_identity["aggregate_sha256"]:
        raise SystemExit("source snapshot differs from captured source identity")
    data_dir = runtime_root / "data"
    data_dir.mkdir()
    for source in sorted((canonical_root / "data").iterdir()):
        (data_dir / source.name).symlink_to(source, target_is_directory=source.is_dir())

    # RepoPaths.repo_root must still support the few build-time repo-relative lookups. Symlinks
    # preserve the resolved source/config dependency keys, expose canonical analysis inputs
    # read-only, and keep the historical repo_root/data lookup on the explicit data boundary.
    overlay_root = runtime_root / "repo-overlay"
    overlay_root.mkdir()
    for name, target in (
        ("src", source_root / "src"),
        ("configs", source_root / "configs"),
        ("data", data_dir),
        ("analysis_scratch", canonical_root / "analysis_scratch"),
    ):
        (overlay_root / name).symlink_to(target, target_is_directory=True)

    state_dir = runtime_root / "state"
    state_dir.mkdir()
    logical_clone_bytes = 0
    for name in MUTABLE_STATE_DIRS:
        source = canonical_root / "state" / name
        if source.exists():
            logical_clone_bytes += sum(
                path.stat().st_size for path in source.rglob("*") if path.is_file()
            )
            _clone(source, state_dir / name)
    qa_dir = state_dir / "qa"
    qa_dir.mkdir()
    qa_tripwire = canonical_root / "state" / "qa" / f"city_feed_exact_point_concentration_{args.year}.csv"
    if not qa_tripwire.is_file():
        raise SystemExit(f"missing validator QA tripwire: {qa_tripwire}")
    qa_tripwire_stamp = Path(f"{qa_tripwire}.deps.json")
    if not qa_tripwire_stamp.is_file():
        raise SystemExit(f"missing validator QA dependency stamp: {qa_tripwire_stamp}")
    for source in (qa_tripwire, qa_tripwire_stamp):
        logical_clone_bytes += source.stat().st_size
        _clone(source, qa_dir / source.name)
    cache_dir = state_dir / "cache"
    cache_dir.mkdir()
    for name in CACHE_DIRS:
        source = canonical_root / "state" / "cache" / name
        if source.exists():
            logical_clone_bytes += sum(
                path.stat().st_size for path in source.rglob("*") if path.is_file()
            )
            _clone(source, cache_dir / name)
    uncertainty_dir = cache_dir / "uncertainty"
    uncertainty_dir.mkdir()
    uncertainty_files = _latest_uncertainty_cache(canonical_root, args.year)
    for source in uncertainty_files:
        logical_clone_bytes += source.stat().st_size
        _clone(source, uncertainty_dir / source.name)
    for name in ("candidates", "locks", "logs"):
        (state_dir / name).mkdir()
    (overlay_root / "state").symlink_to(state_dir, target_is_directory=True)
    dependency_sidecar_rebase = _rebase_dependency_sidecars(
        source_root=source_root,
        canonical_root=canonical_root,
        runtime_root=runtime_root,
    )
    rebuilt_derived_roles = frozenset(
        {
            "state/controls/jurisdiction_controls_smoothed_2025.parquet",
            "state/controls/state_control_comparison.parquet",
        }
        if args.rebuild_service_controls
        else ()
    )
    arm_input_bindings_path = _write_arm_input_bindings(
        source_root=source_root,
        canonical_root=canonical_root,
        runtime_root=runtime_root,
        excluded_roles=rebuilt_derived_roles,
    )
    (runtime_root / "archive").mkdir()
    tmp_dir = runtime_root / "tmp"
    tmp_dir.mkdir()
    user_cache_dir = runtime_root / "user-cache"
    user_cache_dir.mkdir()
    matplotlib_dir = runtime_root / "matplotlib"
    matplotlib_dir.mkdir()

    profile = runtime_root / "sandbox.sb"
    profile.write_text(
        "\n".join(
            (
                "(version 1)",
                "(allow default)",
                f'(deny file-write* (subpath "{_quote_sandbox(canonical_root)}"))',
                f'(deny file-write* (subpath "{_quote_sandbox(source_root)}"))',
                f'(deny file-write* (subpath "{_quote_sandbox(snapshot_root)}"))',
                f'(deny file-write* (subpath "{_quote_sandbox(overlay_root)}"))',
                "",
            )
        )
    )
    python = canonical_root / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"missing canonical Python environment: {python}")
    # Execute from the captured source path so absolute dependency-stamp keys remain identical to
    # v52. The sandbox makes it read-only and the runner verifies the scoped content hashes before
    # every command. The clone above is the immutable audit copy, not a path-key-changing runtime.
    runner = source_root / "scripts" / "release" / "run_isolated_candidate.py"
    base_command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile),
        "/usr/bin/env",
        f"TMPDIR={tmp_dir}",
        f"XDG_CACHE_HOME={user_cache_dir}",
        f"MPLCONFIGDIR={matplotlib_dir}",
        "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={source_root / 'src'}",
        str(python),
        str(runner),
        "--source-root",
        str(source_root),
        "--runtime-root",
        str(runtime_root),
        "--",
    ]
    baseline_manifest_path = (
        canonical_root / "state" / "candidates" / args.baseline_run_id / "manifest.json"
    )
    if not baseline_manifest_path.exists():
        raise SystemExit(f"missing baseline manifest: {baseline_manifest_path}")
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    candidate_command = list((baseline_manifest.get("run") or {}).get("argv") or [])
    if candidate_command and candidate_command[0].endswith("main.py"):
        candidate_command.pop(0)
    forbidden = {
        "--force-controls-rebuild",
        "--force-reporting-regimes-rebuild",
        "--force-geometry-rebuild",
        "--force-bg-prior-rebuild",
        "--force-city-incident-share-rebuild",
        "--force-city-incident-source-refresh",
    }
    present_forbidden = sorted(forbidden.intersection(candidate_command))
    if present_forbidden:
        raise SystemExit(
            f"baseline command contains unsafe rebuild/refresh flags: {present_forbidden}"
        )
    try:
        candidate_index = candidate_command.index("--candidate-run")
    except ValueError as exc:
        raise SystemExit("baseline command has no --candidate-run") from exc
    if candidate_index + 1 >= len(candidate_command):
        raise SystemExit("baseline command has no candidate run id value")
    candidate_command[candidate_index + 1] = args.run_id
    if args.rebuild_service_controls:
        candidate_command.append("--force-controls-rebuild")
    candidate_command.extend(
        [
            "--bg-prior-path",
            str(state_dir / "modeling" / f"bg_prior_long_{args.year}_arm_b.parquet"),
            "--model-surface-feature-policy-path",
            str(state_dir / "modeling" / f"feature_transfer_policy_{args.year}.parquet"),
            "--residual-feature-policy-path",
            str(state_dir / "modeling" / f"feature_transfer_policy_{args.year}.parquet"),
            "--residual-training-city-shares-path",
            str(
                state_dir
                / "modeling"
                / f"next_phase_validation_city_incident_share_surface_{args.year}.parquet"
            ),
            "--residual-training-extra-bg-features-path",
            str(
                data_dir
                / "Overture-Places"
                / "parsed"
                / "block_group_overture_places_states_latest.parquet"
            ),
            "--residual-training-extra-bg-features-path",
            str(
                data_dir
                / "Overture-Places"
                / "parsed"
                / "block_group_overture_commercial_core_states_latest.parquet"
            ),
            "--mixture-experts-path",
            str(state_dir / "modeling" / f"bg_mixture_experts_{args.year}.parquet"),
            "--mixture-weights-path",
            str(source_root / "configs" / "mixture_ship_weights_v3.csv"),
            "--exposure-normalizers-path",
            str(state_dir / "modeling" / f"bg_exposure_normalizers_{args.year}.parquet"),
        ]
    )
    for case_type in (baseline_manifest.get("resolved_config") or {}).get(
        "residual_training_exclude_validation_case_types", []
    ):
        candidate_command.extend(
            ["--residual-training-exclude-validation-case-type", str(case_type)]
        )
    full_candidate_command = [*base_command, *candidate_command]
    launch = runtime_root / "launch.json"
    launch.write_text(
        json.dumps(
            {
                "original_source_root": str(source_root),
                "source_root": str(source_root),
                "source_snapshot_root": str(snapshot_root),
                "canonical_root": str(canonical_root),
                "runtime_root": str(runtime_root),
                "repo_overlay_root": str(overlay_root),
                "run_id": args.run_id,
                "baseline_run_id": args.baseline_run_id,
                "baseline_manifest_path": str(baseline_manifest_path),
                "year": args.year,
                "free_bytes_before_prepare": disk.free,
                "estimated_build_peak_bytes": ESTIMATED_BUILD_PEAK_BYTES,
                "minimum_free_reserve_bytes": MIN_FREE_BYTES,
                "logical_clone_bytes": logical_clone_bytes,
                "raw_input_symlink_count": len(list(data_dir.iterdir())),
                "uncertainty_cache_files": [str(path) for path in uncertainty_files],
                "rebased_dependency_sidecars": dependency_sidecar_rebase[
                    "rewritten_sidecar_count"
                ],
                "dependency_sidecar_rebase_audit": str(
                    runtime_root / "dependency_sidecar_rebase.json"
                ),
                "arm_input_bindings": str(arm_input_bindings_path),
                "declared_rebuilt_derived_roles": sorted(rebuilt_derived_roles),
                "allowed_config_differences": (
                    {"force_controls_rebuild": True}
                    if args.rebuild_service_controls
                    else {}
                ),
                "source_identity": original_source_identity,
                "original_source_identity": original_source_identity,
                "original_git_metadata": original_git_metadata,
                "base_command": base_command,
                "candidate_command": candidate_command,
                "full_candidate_command": full_candidate_command,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"prepared {runtime_root}")
    print(f"logical clone bytes: {logical_clone_bytes}")
    print(f"free before prepare: {disk.free}")
    print("candidate command prefix:")
    print(shlex.join(base_command))
    print("exact baseline-equivalent candidate launch:")
    print(shlex.join(full_candidate_command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
