from pathlib import Path

from crimerisk.allocation import (
    AllocationBuildConfig,
    promoted_residual_extra_bg_feature_paths,
    resolve_allocation_build_config,
)
from crimerisk.paths import RepoPaths


def test_promoted_allocation_uses_target_year_feature_policy(tmp_path: Path) -> None:
    paths = RepoPaths.from_repo_root(tmp_path)
    promoted = paths.state_dir / "modeling" / "next_phase_validation_city_incident_share_surface_2025.parquet"
    promoted.parent.mkdir(parents=True, exist_ok=True)
    promoted.touch()
    for path in promoted_residual_extra_bg_feature_paths(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    resolved = resolve_allocation_build_config(paths, config=AllocationBuildConfig(year=2025))

    expected = Path("state/modeling/feature_transfer_policy_2025.parquet")
    assert resolved.model_surface_feature_policy_path == expected
    assert resolved.residual_feature_policy_path == expected
