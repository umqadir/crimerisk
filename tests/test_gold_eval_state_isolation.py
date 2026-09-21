"""An evaluation run reads shared state; it does not write it.

Every gold fold truncates the feed year, drops a feed city, or both, and the build it launches
used to write the resulting share surface and reconciliation tables straight into
`state/modeling`. That is how the released
`state/modeling/city_incident_share_surface.parquet` ended up holding a temporal fold's
year_end=2023 copy. Folds now carry a run-scoped `--feed-inputs-dir`, and the harness verifies
at the end of the run that the shared city-feed artifacts are byte-identical to what it found.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from crimerisk import city_shares
from crimerisk.allocation import (
    AllocationBuildConfig,
    city_incident_reconciliation_dir,
    city_incident_reconciliation_path,
    city_incident_share_surface_path,
)
from crimerisk.city_shares import CityIncidentShareBuildConfig, write_v2_city_incident_shares
from crimerisk.cli import build_parser
from crimerisk.eval import GoldEvaluation, GoldEvaluationConfig
from crimerisk.eval import gold as gold_module
from crimerisk.paths import RepoPaths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth_rows(years: tuple[int, ...] = (2018, 2021, 2023, 2024)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city_name": ["Austin"] * len(years),
            "city_key": ["austin"] * len(years),
            "jurisdiction_id": ["tx-austin"] * len(years),
            "state_fips": ["48"] * len(years),
            "year": list(years),
            "offense": ["robbery"] * len(years),
            "block_group_geoid": ["480010001001"] * len(years),
            "incident_count": [float(index + 1) for index in range(len(years))],
            "share_within_city": [1.0] * len(years),
            "geocode_quality_tier": ["exact"] * len(years),
        }
    )


@pytest.fixture
def world(tmp_path: Path) -> dict[str, object]:
    """A repo tree carrying released city-feed artifacts and nothing else."""
    paths = RepoPaths.from_repo_root(tmp_path)
    modeling = paths.state_dir / "modeling"
    modeling.mkdir(parents=True)
    production = modeling / "city_incident_share_surface.parquet"
    _truth_rows().to_parquet(production, index=False)
    combined = modeling / "city_incident_reconciliation_2025.parquet"
    _truth_rows().to_parquet(combined, index=False)
    # A model-side artifact with no city-feed input: reported if it moves, never fatal.
    model_side = modeling / "bg_mixture_shares_2025.parquet"
    _truth_rows().to_parquet(model_side, index=False)
    centroids = paths.data_dir / "tiger_bg" / "parsed"
    centroids.mkdir(parents=True)
    pd.DataFrame(
        {"bg_id": ["480010001001"], "lon": [-97.74], "lat": [30.27]}
    ).to_parquet(centroids / "bg_centroids.parquet", index=False)
    reconciliation = paths.review_analysis_dir / "city_reconciliation"
    reconciliation.mkdir(parents=True)
    _truth_rows().to_parquet(reconciliation / "austin_reconciliation.parquet", index=False)
    released = [production, combined, reconciliation / "austin_reconciliation.parquet"]
    return {
        "paths": paths,
        "production": production,
        "combined": combined,
        "model_side": model_side,
        "reconciliation": reconciliation,
        "released_sha256": {str(path): _sha256(path) for path in released},
    }


def _evaluation(world: dict[str, object], **overrides) -> GoldEvaluation:
    config = GoldEvaluationConfig(run_id="isolation", **overrides)
    return GoldEvaluation(paths=world["paths"], config=config)


@pytest.fixture(autouse=True)
def _stub_truth(monkeypatch):
    monkeypatch.setattr(
        gold_module, "_active_truth", lambda paths, truth_path=None: (_truth_rows(), ["austin"])
    )


# --- the fold build is told where to put its own feed artifacts --------------------------------


def test_every_fold_build_is_given_a_run_scoped_feed_inputs_directory(world) -> None:
    evaluation = _evaluation(world)
    specs = evaluation.fold_specs()
    assert specs, "no fold specs to check"
    seen: set[str] = set()
    for spec in specs:
        command = evaluation.build_command(spec)
        assert "--feed-inputs-dir" in command
        value = Path(command[command.index("--feed-inputs-dir") + 1])
        assert value == evaluation.inputs_dir / spec.fold_id
        assert evaluation.run_dir in value.parents
        seen.add(str(value))
    assert len(seen) == len(specs), "folds must not share a feed inputs directory"


def test_the_cli_option_routes_every_feed_artifact_out_of_state_modeling(world) -> None:
    paths: RepoPaths = world["paths"]
    fold_dir = paths.state_dir / "eval" / "isolation" / "inputs" / "temporal_through_2023"
    args = build_parser().parse_args(
        ["build-outputs", "--feed-year-end", "2023", "--feed-inputs-dir", str(fold_dir)]
    )
    assert args.feed_inputs_dir == str(fold_dir)

    config = AllocationBuildConfig(year=2025, feed_year_end=2023, feed_inputs_dir=fold_dir)
    assert city_incident_share_surface_path(paths, feed_inputs_dir=config.feed_inputs_dir) == (
        fold_dir / "city_incident_share_surface.parquet"
    )
    assert city_incident_reconciliation_dir(paths, feed_inputs_dir=config.feed_inputs_dir) == (
        fold_dir / "city_reconciliation"
    )
    assert city_incident_reconciliation_path(
        paths, year=2023, feed_inputs_dir=config.feed_inputs_dir
    ) == (fold_dir / "city_incident_reconciliation_2023.parquet")

    # Unset, every one of them still resolves to the released location.
    production = replace(config, feed_inputs_dir=None)
    assert city_incident_share_surface_path(
        paths, feed_inputs_dir=production.feed_inputs_dir
    ) == (paths.state_dir / "modeling" / "city_incident_share_surface.parquet")


# --- a dry run plus one synthetic fold leaves the released surface byte-identical --------------


def test_a_dry_run_and_one_synthetic_fold_leave_the_released_artifacts_untouched(
    world, monkeypatch, capsys
) -> None:
    paths: RepoPaths = world["paths"]
    before = dict(world["released_sha256"])

    evaluation = _evaluation(world, dry_run=True, folds=("temporal",))
    assert evaluation.run().empty
    payload = json.loads(capsys.readouterr().out)
    assert payload and all("--feed-inputs-dir" in row["command"] for row in payload)

    # The fold build, reduced to the one thing it used to do to shared state.
    spec = next(s for s in evaluation.fold_specs() if s.fold_id == "temporal_through_2023")
    fold_inputs = evaluation.fold_inputs_dir(spec)
    truncated = _truth_rows(years=(2018, 2021, 2023))
    monkeypatch.setattr(city_shares, "_city_impls", lambda paths: {})
    monkeypatch.setattr(
        city_shares, "_load_enabled_city_order", lambda paths, exclude_city_keys=(): ["austin"]
    )
    monkeypatch.setattr(city_shares, "_city_dependency_paths", lambda paths, impls: [])
    monkeypatch.setattr(
        city_shares,
        "build_city_incident_share_surface",
        lambda *, paths, config: (truncated, ["austin"], {"austin": truncated}),
    )
    summary = write_v2_city_incident_shares(
        paths=paths,
        out_path=city_incident_share_surface_path(paths, feed_inputs_dir=fold_inputs),
        reconciliation_dir=city_incident_reconciliation_dir(paths, feed_inputs_dir=fold_inputs),
        combined_reconciliation_path=city_incident_reconciliation_path(
            paths, year=2023, feed_inputs_dir=fold_inputs
        ),
        config=CityIncidentShareBuildConfig(year_start=2018, year_end=2023, force_rebuild=True),
    )
    assert summary["used_cache"] is False

    # The fold's truncated world exists, under the run.
    fold_surface = fold_inputs / "city_incident_share_surface.parquet"
    assert fold_surface.is_file()
    assert int(pd.read_parquet(fold_surface)["year"].max()) == 2023
    assert (fold_inputs / "city_incident_reconciliation_2023.parquet").is_file()
    assert (fold_inputs / "city_reconciliation" / "austin_reconciliation.parquet").is_file()

    # The released world is byte-for-byte what it was.
    after = {path: _sha256(Path(path)) for path in before}
    assert after == before
    assert int(pd.read_parquet(world["production"])["year"].max()) == 2024


# --- and the harness says so if some other stage reaches back in -------------------------------


def test_the_guard_fails_loudly_when_a_released_feed_artifact_changes(world) -> None:
    evaluation = _evaluation(world)
    protected = evaluation._fingerprint(evaluation.protected_state_paths())
    observed = evaluation._fingerprint(evaluation.observed_state_paths())
    assert str(world["production"]) in protected
    assert str(world["model_side"]) in observed
    assert str(world["model_side"]) not in protected

    _truth_rows(years=(2018, 2021, 2023)).to_parquet(world["production"], index=False)
    with pytest.raises(RuntimeError, match="must only read"):
        evaluation._assert_state_unchanged(protected, observed)


def test_the_guard_reports_a_model_side_rebuild_without_failing(world, capsys) -> None:
    evaluation = _evaluation(world)
    protected = evaluation._fingerprint(evaluation.protected_state_paths())
    observed = evaluation._fingerprint(evaluation.observed_state_paths())

    _truth_rows(years=(2018,)).to_parquet(world["model_side"], index=False)
    report = evaluation._assert_state_unchanged(protected, observed)
    assert report["observed"]["changed"] == [str(world["model_side"])]
    assert report["protected"]["changed"] == []
    assert "WARNING" in capsys.readouterr().out


def test_an_unchanged_run_reports_nothing_changed(world) -> None:
    evaluation = _evaluation(world)
    protected = evaluation._fingerprint(evaluation.protected_state_paths())
    observed = evaluation._fingerprint(evaluation.observed_state_paths())
    report = evaluation._assert_state_unchanged(protected, observed)
    assert not any(report["protected"].values())
    assert not any(report["observed"].values())
    assert report["protected_paths_checked"] == len(protected)
