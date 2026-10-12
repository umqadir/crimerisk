from __future__ import annotations

from crimerisk import allocation, smoothed_controls
from crimerisk.allocation import AllocationBuildConfig
from crimerisk.paths import RepoPaths


def test_output_dependency_rebuilds_geometry_before_controls(tmp_path, monkeypatch):
    paths = RepoPaths.from_repo_root(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(allocation, "write_v2_geometry", lambda **kwargs: calls.append("geometry"))
    monkeypatch.setattr(allocation, "write_v2_controls", lambda **kwargs: calls.append("controls"))
    monkeypatch.setattr(
        allocation,
        "_ensure_controls_dependencies",
        lambda **kwargs: calls.append("controls_dependencies"),
    )
    monkeypatch.setattr(
        allocation,
        "write_v2_city_incident_shares",
        lambda **kwargs: calls.append("city_incident_shares"),
    )
    monkeypatch.setattr(allocation, "_controls_imputation_lane_differs", lambda **kwargs: False)
    monkeypatch.setattr(allocation, "controls_artifacts_are_current", lambda *args, **kwargs: True)
    monkeypatch.setattr(allocation, "geometry_artifacts_are_current", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        smoothed_controls, "smoothed_controls_artifact_is_current", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(
        smoothed_controls,
        "write_v2_smoothed_controls",
        lambda **kwargs: calls.append("smoothed_controls"),
    )

    allocation._ensure_output_dependencies(
        paths=paths,
        config=AllocationBuildConfig(
            year=2025,
            force_geometry_rebuild=True,
            force_controls_rebuild=True,
        ),
    )

    assert calls == [
        "controls_dependencies",
        "geometry",
        "controls",
        "smoothed_controls",
        "city_incident_shares",
    ]
