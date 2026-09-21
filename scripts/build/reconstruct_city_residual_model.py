from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.allocation import (  # noqa: E402
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES,
    DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE,
    PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES,
    _apply_exclusive_footprint_displacement,
    _build_exclusive_footprint_displacement,
    _load_bg_crosswalk,
    _load_city_incident_share_surface,
    _load_concurrent_jurisdiction_carveouts,
    _load_overlap_custom_footprints,
    _load_overlap_footprint_overrides,
    _regular_residual_prior_for_burglary_only_variant,
    promoted_residual_extra_bg_feature_paths,
    promoted_residual_training_city_shares_path,
)
from crimerisk.city_residuals import (  # noqa: E402
    CityResidualConfig,
    city_residual_model_code_identity,
    city_residual_fitted_model_path,
    city_residual_prediction_surface_path,
    fit_city_residual_model_from_truth,
    load_city_residual_prediction_surface,
    verify_city_residual_model_reproduction,
    write_city_residual_fitted_model,
)
from crimerisk.model_surface import bg_feature_dependency_paths  # noqa: E402
from crimerisk.paths import RepoPaths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct and verify the fitted city residual model."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(args.repo_root.resolve())
    paths = replace(
        paths,
        state_dir=args.state_dir.resolve(),
        cache_dir=args.state_dir.resolve() / "cache",
    )
    year = int(args.year)
    training_path = promoted_residual_training_city_shares_path(paths, year=year)
    prior_path = paths.state_dir / "modeling" / f"bg_prior_long_{year}_arm_b.parquet"
    policy_path = paths.state_dir / "modeling" / f"feature_transfer_policy_{year}.parquet"
    prediction_path = city_residual_prediction_surface_path(paths, year=year)
    extra_paths = promoted_residual_extra_bg_feature_paths(paths)

    training = _load_city_incident_share_surface(
        paths,
        year=year,
        path=training_path,
        exclude_validation_case_types=PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES,
    )
    bg_prior = pd.read_parquet(prior_path)
    bg_crosswalk = _load_bg_crosswalk(paths)
    displacement = _build_exclusive_footprint_displacement(
        overrides=_load_overlap_footprint_overrides(paths),
        custom_footprints=_load_overlap_custom_footprints(paths),
        concurrent_jurisdiction_carveouts=_load_concurrent_jurisdiction_carveouts(paths),
    )
    if not displacement.empty:
        bg_crosswalk = _apply_exclusive_footprint_displacement(bg_crosswalk, displacement)
    burglary_prior = _regular_residual_prior_for_burglary_only_variant(
        paths=paths,
        bg_prior=bg_prior,
        year=year,
    )
    config = CityResidualConfig(
        extra_feature_paths=tuple(extra_paths),
        feature_policy_path=policy_path,
        exclude_feature_policy_classes=tuple(
            DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES
        ),
        exclude_feature_policy_classes_by_offense=tuple(
            DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE
        ),
    )
    fitted = fit_city_residual_model_from_truth(
        paths=paths,
        city_shares=training.rename(columns={"block_group_geoid": "bg_id"}),
        bg_prior=burglary_prior if burglary_prior is not None else bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=year,
        config=config,
        burglary_bg_prior=bg_prior if burglary_prior is not None else None,
    )
    if fitted is None:
        raise ValueError("Frozen residual training inputs produced no fitted model")
    predictions = load_city_residual_prediction_surface(prediction_path)
    input_paths = {
        "training_city_shares": training_path,
        "bg_prior": prior_path,
        "bg_crosswalk": paths.state_dir
        / "geometry"
        / "block_group_to_jurisdiction_crosswalk.parquet",
        "feature_policy": policy_path,
        "promoted_prediction_surface": prediction_path,
        "promoted_prediction_manifest": prediction_path.with_suffix(
            prediction_path.suffix + ".manifest.json"
        ),
        "overlap_footprint_overrides": paths.repo_root
        / "configs"
        / "overlap_footprint_overrides.csv",
        "overlap_custom_footprints": paths.repo_root
        / "configs"
        / "overlap_custom_footprints.csv",
        "concurrent_jurisdiction_carveouts": paths.repo_root
        / "configs"
        / "concurrent_jurisdiction_carveouts.csv",
    }
    input_paths.update(
        {f"extra_feature_{index}": path for index, path in enumerate(extra_paths)}
    )
    input_paths.update(
        {
            f"bg_feature_dependency_{index}": path
            for index, path in enumerate(bg_feature_dependency_paths(paths, year=year))
            if path.exists()
        }
    )
    try:
        reproduction = verify_city_residual_model_reproduction(
            paths=paths,
            year=year,
            fitted=fitted,
            predictions=predictions,
            tolerance=1e-8,
            chunk_size=int(args.chunk_size),
        )
    except Exception as exc:
        quarantine_path = (
            paths.state_dir
            / "analysis"
            / "residual-model-reconstruction"
            / f"city_residual_fitted_model_{year}.unaccepted.joblib"
        )
        write_city_residual_fitted_model(
            path=quarantine_path,
            fitted=fitted,
            input_paths=input_paths,
            reproduction={
                "status": "failed",
                "rows_expected": int(len(predictions)),
                "tolerance": 1e-8,
                "error": str(exc),
            },
        )
        raise RuntimeError(
            f"Residual reconstruction failed; unaccepted fitted object preserved at {quarantine_path}"
        ) from exc
    manifest = write_city_residual_fitted_model(
        path=city_residual_fitted_model_path(paths, year=year),
        fitted=fitted,
        input_paths=input_paths,
        reproduction=reproduction,
        code_identity=city_residual_model_code_identity(
            {
                "city_residuals": paths.repo_root / "src" / "crimerisk" / "city_residuals.py",
                "allocation": paths.repo_root / "src" / "crimerisk" / "allocation.py",
            }
        ),
    )
    print(manifest["artifact"])
    print(manifest["artifact_sha256"])
    print(reproduction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
