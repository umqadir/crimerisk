from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crimerisk import allocation, city_shares
from crimerisk.allocation import AllocationBuildConfig
from crimerisk.build_freshness import artifact_is_current, write_dependency_stamp
from crimerisk.cli import build_parser
from crimerisk.confidence import _load_residual_training_city_shares
from crimerisk.paths import RepoPaths


def _city_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city_name": ["Austin", "Austin", "Boston", "Boston"],
            "jurisdiction_id": ["a", "a", "b", "b"],
            "state_fips": ["48", "48", "25", "25"],
            "year": [2021, 2024, 2021, 2024],
            "offense": ["robbery"] * 4,
            "block_group_geoid": ["1", "1", "2", "2"],
            "incident_count": [1.0, 2.0, 3.0, 4.0],
            "share_within_city": [1.0] * 4,
            "geocode_quality_tier": ["exact"] * 4,
            "validation_case_type": ["promoted_city_control"] * 4,
        }
    )


def test_build_outputs_cli_accepts_repeatable_city_exclusions_and_feed_year_end() -> None:
    args = build_parser().parse_args(
        [
            "build-outputs",
            "--exclude-feed-city",
            "austin",
            "--exclude-feed-city",
            "boston",
            "--feed-year-end",
            "2021",
        ]
    )
    assert args.exclude_feed_city == ["austin", "boston"]
    assert args.feed_year_end == 2021


def test_allocation_loader_excludes_city_and_truncates_year(tmp_path: Path) -> None:
    paths = RepoPaths.from_repo_root(tmp_path)
    surface = tmp_path / "city.parquet"
    _city_rows().to_parquet(surface, index=False)

    loaded = allocation._load_city_incident_share_surface(
        paths,
        year=2021,
        path=surface,
        exclude_feed_city_keys=("austin",),
    )

    assert set(loaded["city_name"]) == {"Boston"}
    assert loaded["pooled_source_year_max"].eq(2021).all()


def test_confidence_training_loader_uses_same_city_and_year_holdout(tmp_path: Path) -> None:
    surface = tmp_path / "training.parquet"
    _city_rows().to_parquet(surface, index=False)

    loaded = _load_residual_training_city_shares(
        path=surface,
        exclude_validation_case_types=(),
        exclude_feed_city_keys=("boston",),
        year_end=2021,
    )

    assert set(loaded["city_name"]) == {"Austin"}
    assert loaded["year"].max() == 2021


def test_city_share_cache_key_includes_year_range_and_exclusions(tmp_path: Path) -> None:
    artifact = tmp_path / "city.parquet"
    dependency = tmp_path / "input.csv"
    artifact.write_bytes(b"surface")
    dependency.write_text("input\n")
    parameters = {
        "year_start": 2018,
        "year_end": 2021,
        "exclude_city_keys": ["austin"],
    }
    write_dependency_stamp(artifact, [dependency], parameters=parameters)

    assert artifact_is_current(artifact, [dependency], parameters=parameters)
    assert not artifact_is_current(
        artifact,
        [dependency],
        parameters={**parameters, "year_end": 2023},
    )
    assert not artifact_is_current(
        artifact,
        [dependency],
        parameters={**parameters, "exclude_city_keys": ["boston"]},
    )
    stamp = json.loads((tmp_path / "city.parquet.deps.json").read_text())
    assert stamp["parameters"] == parameters


def test_output_dependency_build_threads_holdout_config(tmp_path: Path, monkeypatch) -> None:
    paths = RepoPaths.from_repo_root(tmp_path)
    captured: list[city_shares.CityIncidentShareBuildConfig] = []
    monkeypatch.setattr(allocation, "_ensure_controls_dependencies", lambda **kwargs: None)
    monkeypatch.setattr(allocation, "controls_artifacts_are_current", lambda *args, **kwargs: True)
    monkeypatch.setattr(allocation, "geometry_artifacts_are_current", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        allocation,
        "write_v2_city_incident_shares",
        lambda **kwargs: captured.append(kwargs["config"]),
    )

    allocation._ensure_output_dependencies(
        paths=paths,
        config=AllocationBuildConfig(
            year=2025,
            feed_year_end=2023,
            exclude_feed_city_keys=("austin",),
            use_promoted_next_phase_allocator=False,
        ),
    )

    assert len(captured) == 1
    assert captured[0].year_end == 2023
    assert captured[0].exclude_city_keys == ("austin",)


def test_filter_excluded_city_keys_supports_live_city_key_column() -> None:
    frame = _city_rows().assign(city_key=["austin", "austin", "boston", "boston"])
    filtered = city_shares.filter_excluded_city_keys(
        frame,
        exclude_city_keys=("austin",),
    )
    assert set(filtered["city_key"]) == {"boston"}
