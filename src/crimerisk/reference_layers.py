from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crimerisk.build_freshness import artifact_is_current
from crimerisk.jurisdiction_reference import ReferenceArtifacts, build_reference_artifacts
from crimerisk.paths import RepoPaths
from crimerisk.reference import write_agency_master
from crimerisk.stage_locks import stage_write_lock


@dataclass(frozen=True)
class ReferenceLayerBuildConfig:
    year_start: int = 2018
    year_end: int = 2024
    agency_master_path: Path | None = None
    provisional_local_path: Path | None = None
    resolved_local_tail_path: Path | None = None
    nonlocal_final_path: Path | None = None
    nonlocal_auto_path: Path | None = None
    municipal_override_path: Path | None = None
    local_override_path: Path | None = None
    agency_master_supplement_path: Path | None = None


@dataclass(frozen=True)
class ReferenceInputPromotionConfig:
    provisional_local_source_path: Path | None = None
    resolved_local_tail_source_path: Path | None = None
    nonlocal_final_source_path: Path | None = None
    nonlocal_auto_source_path: Path | None = None


def _canonical_reference_input_paths(paths: RepoPaths) -> dict[str, Path]:
    reference_inputs_root = paths.state_dir / "reference" / "inputs"
    return {
        "provisional_local": reference_inputs_root / "provisional_local_agency_matches.parquet",
        "resolved_local_tail": reference_inputs_root / "local_queue_resolved_final.parquet",
        "nonlocal_final": reference_inputs_root / "nonmunicipal_special_resolved_final.parquet",
        "nonlocal_auto": reference_inputs_root / "nonmunicipal_auto_defaults.parquet",
    }


def get_v2_reference_output_paths(paths: RepoPaths) -> dict[str, Path]:
    reference_root = paths.state_dir / "reference"
    return {
        "agency_master": reference_root / "agency_master.parquet",
        "full_local": reference_root / "local_agency_resolved_full.parquet",
        "full_nonlocal": reference_root / "nonlocal_agency_resolved_full.parquet",
        "jurisdiction_master": reference_root / "jurisdiction_master.parquet",
        "crosswalk": reference_root / "agency_to_jurisdiction_crosswalk.parquet",
    }


def _resolve_reference_layer_config(
    *,
    paths: RepoPaths,
    config: ReferenceLayerBuildConfig,
) -> ReferenceLayerBuildConfig:
    config_root = paths.repo_root / "configs"
    reference_root = paths.state_dir / "reference"
    reference_inputs = _canonical_reference_input_paths(paths)

    return ReferenceLayerBuildConfig(
        year_start=int(config.year_start),
        year_end=int(config.year_end),
        agency_master_path=config.agency_master_path or (reference_root / "agency_master.parquet"),
        provisional_local_path=config.provisional_local_path or reference_inputs["provisional_local"],
        resolved_local_tail_path=config.resolved_local_tail_path or reference_inputs["resolved_local_tail"],
        nonlocal_final_path=config.nonlocal_final_path or reference_inputs["nonlocal_final"],
        nonlocal_auto_path=config.nonlocal_auto_path or reference_inputs["nonlocal_auto"],
        municipal_override_path=config.municipal_override_path or (config_root / "municipal_geometry_overrides.csv"),
        local_override_path=config.local_override_path or (config_root / "local_resolution_overrides.csv"),
        agency_master_supplement_path=config.agency_master_supplement_path or (config_root / "agency_master_supplement.csv"),
    )


def reference_layer_dependency_paths(
    *,
    paths: RepoPaths,
    config: ReferenceLayerBuildConfig = ReferenceLayerBuildConfig(),
) -> list[Path]:
    resolved = _resolve_reference_layer_config(paths=paths, config=config)
    dependencies = [
        resolved.provisional_local_path,
        resolved.resolved_local_tail_path,
        resolved.nonlocal_final_path,
        resolved.nonlocal_auto_path,
        resolved.municipal_override_path,
        resolved.local_override_path,
        resolved.agency_master_supplement_path,
        Path(__file__),
        paths.repo_root / "src" / "crimerisk" / "reference.py",
        paths.repo_root / "src" / "crimerisk" / "jurisdiction_reference.py",
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in dependencies:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def reference_layers_artifacts_are_current(
    *,
    paths: RepoPaths,
    config: ReferenceLayerBuildConfig = ReferenceLayerBuildConfig(),
    agency_master_path: Path | None = None,
    full_local_out_path: Path | None = None,
    full_nonlocal_out_path: Path | None = None,
    jurisdiction_out_path: Path | None = None,
    crosswalk_out_path: Path | None = None,
) -> bool:
    canonical_outputs = get_v2_reference_output_paths(paths)
    resolved = _resolve_reference_layer_config(paths=paths, config=config)
    dependencies = reference_layer_dependency_paths(paths=paths, config=config)
    outputs = [
        agency_master_path or resolved.agency_master_path,
        full_local_out_path or canonical_outputs["full_local"],
        full_nonlocal_out_path or canonical_outputs["full_nonlocal"],
        jurisdiction_out_path or canonical_outputs["jurisdiction_master"],
        crosswalk_out_path or canonical_outputs["crosswalk"],
    ]
    return all(artifact_is_current(path, dependencies) for path in outputs)


def _resolve_reference_input_promotion_config(
    *,
    paths: RepoPaths,
    config: ReferenceInputPromotionConfig,
) -> ReferenceInputPromotionConfig:
    review_root = paths.review_queues_dir / "local_resolution"
    return ReferenceInputPromotionConfig(
        provisional_local_source_path=config.provisional_local_source_path
        or (review_root / "provisional_local_agency_matches.parquet"),
        resolved_local_tail_source_path=config.resolved_local_tail_source_path
        or (review_root / "local_queue_resolved_final.parquet"),
        nonlocal_final_source_path=config.nonlocal_final_source_path
        or (review_root / "nonmunicipal_special_resolved_final.parquet"),
        nonlocal_auto_source_path=config.nonlocal_auto_source_path
        or (review_root / "nonmunicipal_auto_defaults.parquet"),
    )


def _iter_reference_input_promotions(
    *,
    paths: RepoPaths,
    config: ReferenceInputPromotionConfig,
) -> list[tuple[str, Path, Path]]:
    canonical_inputs = _canonical_reference_input_paths(paths)
    return [
        (
            "provisional_local_agency_matches.parquet",
            canonical_inputs["provisional_local"],
            config.provisional_local_source_path,
        ),
        (
            "local_queue_resolved_final.parquet",
            canonical_inputs["resolved_local_tail"],
            config.resolved_local_tail_source_path,
        ),
        (
            "nonmunicipal_special_resolved_final.parquet",
            canonical_inputs["nonlocal_final"],
            config.nonlocal_final_source_path,
        ),
        (
            "nonmunicipal_auto_defaults.parquet",
            canonical_inputs["nonlocal_auto"],
            config.nonlocal_auto_source_path,
        ),
    ]


def promote_v2_reference_inputs(
    *,
    paths: RepoPaths,
    config: ReferenceInputPromotionConfig = ReferenceInputPromotionConfig(),
) -> dict[str, object]:
    resolved = _resolve_reference_input_promotion_config(paths=paths, config=config)
    copied: dict[str, str] = {}
    for output_name, canonical_path, source_path in _iter_reference_input_promotions(paths=paths, config=resolved):
        if not source_path.exists():
            raise FileNotFoundError(
                "Missing review resolution input required to promote canonical reference input: "
                f"{source_path}"
            )
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, canonical_path)
        copied[output_name] = str(canonical_path)
    return copied


def _require_reference_inputs(
    *,
    paths: RepoPaths,
    config: ReferenceLayerBuildConfig,
) -> None:
    canonical_paths = set(_canonical_reference_input_paths(paths).values())
    for active_path in [
        config.provisional_local_path,
        config.resolved_local_tail_path,
        config.nonlocal_final_path,
        config.nonlocal_auto_path,
    ]:
        if active_path.exists():
            continue
        if active_path in canonical_paths:
            raise FileNotFoundError(
                "Missing canonical reference input surface: "
                f"{active_path}. Run `python main.py promote-reference-inputs` first."
            )
        raise FileNotFoundError(f"Missing reference input path: {active_path}")


def build_v2_reference_layers(
    *,
    paths: RepoPaths,
    config: ReferenceLayerBuildConfig = ReferenceLayerBuildConfig(),
) -> ReferenceArtifacts:
    resolved = _resolve_reference_layer_config(paths=paths, config=config)
    _require_reference_inputs(paths=paths, config=resolved)
    write_agency_master(
        paths=paths,
        out_path=resolved.agency_master_path,
        year_start=resolved.year_start,
        year_end=resolved.year_end,
        supplement_path=resolved.agency_master_supplement_path,
    )
    return build_reference_artifacts(
        agency_master_path=resolved.agency_master_path,
        provisional_local_path=resolved.provisional_local_path,
        resolved_local_tail_path=resolved.resolved_local_tail_path,
        nonlocal_final_path=resolved.nonlocal_final_path,
        nonlocal_auto_path=resolved.nonlocal_auto_path,
        municipal_override_path=resolved.municipal_override_path,
        local_override_path=resolved.local_override_path,
    )


def write_v2_reference_layers(
    *,
    paths: RepoPaths,
    full_local_out_path: Path,
    full_nonlocal_out_path: Path,
    jurisdiction_out_path: Path,
    crosswalk_out_path: Path,
    config: ReferenceLayerBuildConfig = ReferenceLayerBuildConfig(),
    blocked_by: tuple[str, ...] | None = None,
) -> dict[str, object]:
    with stage_write_lock(paths=paths, stage="reference_layers", blocked_by=blocked_by):
        artifacts = build_v2_reference_layers(paths=paths, config=config)

        outputs: list[tuple[pd.DataFrame, Path]] = [
            (artifacts.full_local, full_local_out_path),
            (artifacts.full_nonlocal, full_nonlocal_out_path),
            (artifacts.jurisdiction_master, jurisdiction_out_path),
            (artifacts.agency_to_jurisdiction_crosswalk, crosswalk_out_path),
        ]
        for frame, path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)

        agency_master = pd.read_parquet(_resolve_reference_layer_config(paths=paths, config=config).agency_master_path)
        crosswalk = artifacts.agency_to_jurisdiction_crosswalk
        jurisdiction_master = artifacts.jurisdiction_master
        return {
            "full_local_rows": int(len(artifacts.full_local)),
            "full_nonlocal_rows": int(len(artifacts.full_nonlocal)),
            "jurisdiction_master_rows": int(len(jurisdiction_master)),
            "jurisdiction_type_counts": jurisdiction_master["jurisdiction_type"].value_counts().to_dict(),
            "crosswalk_rows": int(len(crosswalk)),
            "crosswalk_relationship_counts": crosswalk["relationship_type"].value_counts().to_dict(),
            "crosswalk_review_status_counts": crosswalk["review_status"].value_counts().to_dict(),
            "agency_master_rows": int(len(agency_master)),
            "crosswalk_unique_ori": int(crosswalk["ori"].nunique()),
            "missing_crosswalk_ori": int(agency_master["ori9"].nunique() - crosswalk["ori"].nunique()),
        }
