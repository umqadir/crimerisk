from __future__ import annotations

import pytest

from scripts.release.promote_candidate import (
    _manifest_year,
    _required_artifacts,
)


def test_promotion_artifact_names_follow_target_year() -> None:
    artifacts = _required_artifacts(2025)

    assert "crimerisk_block_group_2025_ags_core.parquet" in artifacts
    assert "crimerisk_tract_2025_ags_core.parquet" in artifacts
    assert "diagnostics/state_cde_core_comparison_2025.parquet" in artifacts
    assert "unlocated_mass_2025.parquet" in artifacts
    assert all("2024" not in name for name in artifacts)


def test_promotion_year_comes_from_candidate_manifest() -> None:
    assert _manifest_year({"year": 2025}) == 2025
    assert _manifest_year({"year": "2025"}) == 2025


@pytest.mark.parametrize("manifest", [{}, {"year": None}, {"year": "bad"}, {"year": 1999}])
def test_invalid_candidate_manifest_year_is_rejected(manifest: dict[str, object]) -> None:
    with pytest.raises(SystemExit):
        _manifest_year(manifest)
