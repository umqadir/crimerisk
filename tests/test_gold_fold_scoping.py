"""What a gold fold may see, and what a resumed run must find unchanged.

- The finished-share blend: a feed fold builds with weights fitted without what it scores
  (cross-fitted without the held-out city; refitted on truth through the cutoff).
- The stage-10b prior: a held-out fold builds its own, scoped like its stage-12 build.
- Resume: refused unless code and numerical dependencies have the same content.
- The AGS comparator is optional; the E4 panel is a digest-checked manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.crime import OFFENSES_7
from crimerisk.eval import GoldEvaluation, GoldEvaluationConfig
from crimerisk.eval import gold as gold_module
from crimerisk.eval.gold import FoldSpec, load_e4_panel, pooled_blend_weights, truth_through_year
from crimerisk.paths import RepoPaths

CITIES = {"austin": "Austin", "boston": "Boston", "chicago": "Chicago"}


def _truth() -> pd.DataFrame:
    rows = []
    for key, name in CITIES.items():
        for year in (2020, 2022, 2024):
            for index, bg in enumerate(("480010001001", "480010001002", "480010001003")):
                rows.append(
                    {
                        "city_key": key,
                        "city_name": name,
                        "jurisdiction_id": key,
                        "year": year,
                        "offense": "larceny",
                        "block_group_geoid": bg,
                        # Pre-2022 truth looks like the model; 2024 looks like exposure.
                        "incident_count": float([6, 1, 1][index] if year < 2022 else [1, 1, 6][index]),
                    }
                )
    return pd.DataFrame(rows)


def _retained() -> pd.DataFrame:
    rows = []
    for name in CITIES.values():
        for index, bg in enumerate(("480010001001", "480010001002", "480010001003")):
            rows.append(
                {
                    "fold_id": f"spatial_{name.lower()}",
                    "fold_type": "spatial",
                    "city_name": name,
                    "offense": "larceny",
                    "support_role": "primary",
                    "support": "block_group",
                    "support_id": bg,
                    "truth_count": 0.0,
                    "prediction_ours": [6.0, 1.0, 1.0][index],
                    "prediction_population": 1.0,
                    "prediction_primary_exposure": [1.0, 1.0, 6.0][index],
                }
            )
    return pd.DataFrame(rows)


def _blend_config(**larceny) -> pd.DataFrame:
    rows = []
    for offense in OFFENSES_7:
        adopted = offense == "larceny"
        rows.append(
            {
                "offense": offense,
                "weight_model": larceny.get("model", 0.8) if adopted else 1.0,
                "weight_population": larceny.get("population", 0.1) if adopted else 0.0,
                "weight_primary_exposure": larceny.get("exposure", 0.1) if adopted else 0.0,
                "status": "adopted" if adopted else "not_adopted",
                "source_run_id": "src",
                "source_fold_type": "spatial",
                "fit_cities": ";".join(CITIES.values()) if adopted else "",
                "fit_mean_tvd": 0.2 if adopted else np.nan,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def world(tmp_path: Path, monkeypatch) -> RepoPaths:
    paths = RepoPaths.from_repo_root(tmp_path)
    centroids = paths.data_dir / "tiger_bg" / "parsed"
    centroids.mkdir(parents=True)
    pd.DataFrame(
        {
            "bg_id": ["480010001001", "480010001002", "480010001003", "480010002001"],
            "lon": [-97.74, -97.73, -97.72, -97.70],
            "lat": [30.27, 30.28, 30.29, 30.30],
        }
    ).to_parquet(centroids / "bg_centroids.parquet", index=False)
    truth = _truth()
    monkeypatch.setattr(
        gold_module, "_active_truth", lambda paths, truth_path=None: (truth, list(CITIES))
    )
    (paths.repo_root / "configs").mkdir()
    _blend_config().to_csv(paths.repo_root / "configs" / "finished_share_blend_v1.csv", index=False)

    source = paths.state_dir / "eval" / "src"
    (source / "inputs").mkdir(parents=True)
    records = []
    for name, frame in _retained().groupby("city_name"):
        path = source / "inputs" / f"spatial_{name}.parquet"
        frame.to_parquet(path, index=False)
        records.append(
            {"fold_type": "spatial", "input_path": str(path), "input_sha256": gold_module._sha256(path)}
        )
    truth_path = source / "inputs" / "truth.parquet"
    truth.to_parquet(truth_path, index=False)
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "completed_at_utc": "x",
                "folds": records,
                "inputs": {"city_truth": {"path": str(truth_path), "sha256": gold_module._sha256(truth_path)}},
            }
        )
    )
    xfit_dir = source / "posthoc" / "blend_xfit"
    xfit_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "fold_type": "spatial",
                "offense": "larceny",
                "city_name": name,
                "weight_ours": 0.9 if name == "Austin" else 0.6,
                "weight_population": 0.0,
                "weight_primary_exposure": 0.1 if name == "Austin" else 0.4,
            }
            for name in ("Austin", "Boston")
        ]
    ).to_csv(xfit_dir / "weights.csv", index=False)
    (xfit_dir / "manifest.json").write_text(
        json.dumps({"weights_sha256": gold_module._sha256(xfit_dir / "weights.csv")})
    )
    return paths


def _evaluation(paths: RepoPaths, **overrides) -> GoldEvaluation:
    return GoldEvaluation(paths=paths, config=GoldEvaluationConfig(run_id=overrides.pop("run_id", "f1"), **overrides))


def _spec(evaluation: GoldEvaluation, fold_id: str) -> FoldSpec:
    return next(spec for spec in evaluation.fold_specs() if spec.fold_id == fold_id)


def _larceny(table: pd.DataFrame) -> tuple[float, float, float]:
    row = table[table["offense"].eq("larceny")].iloc[0]
    return (row["weight_model"], row["weight_population"], row["weight_primary_exposure"])


# --- #8 the blend a fold builds with -------------------------------------------------------------


def test_a_spatial_fold_uses_the_weights_cross_fitted_without_its_city(world) -> None:
    evaluation = _evaluation(world)
    spec = _spec(evaluation, "spatial_austin")
    table = evaluation.fold_blend_table(spec)
    assert _larceny(table) == (0.9, 0.0, 0.1)
    assert table.loc[table["offense"].eq("larceny"), "fold_weight_source"].item().startswith(
        "cross_fitted_without:Austin"
    )
    others = table[~table["offense"].eq("larceny")]
    assert (others["weight_model"] == 1.0).all() and (others["fold_weight_source"] == "identity").all()
    command = evaluation.build_command(spec)
    assert command[command.index("--finished-share-blend-path") + 1] == str(
        evaluation.run_dir / "blend" / "spatial_austin.csv"
    )


def test_a_fit_city_without_cross_fitted_weights_is_refused(world) -> None:
    evaluation = _evaluation(world)
    with pytest.raises(ValueError, match="fitted on Chicago"):
        evaluation.fold_blend_table(_spec(evaluation, "spatial_chicago"))


def test_a_temporal_fold_refits_on_truth_through_its_cutoff_only(world) -> None:
    evaluation = _evaluation(world)
    spec = _spec(evaluation, "temporal_through_2021")
    table = evaluation.fold_blend_table(spec)
    retained = _retained()
    expected = pooled_blend_weights(truth_through_year(retained, _truth(), year_end=2021))
    row = expected.set_index("offense").loc["larceny"]
    assert _larceny(table) == (row["weight_ours"], row["weight_population"], row["weight_primary_exposure"])
    # Pre-cutoff truth matches the model, so the refit keeps it; the production weights, fitted
    # on every year, would not be this.
    assert _larceny(table) == (1.0, 0.0, 0.0)
    all_years = pooled_blend_weights(truth_through_year(retained, _truth(), year_end=2025))
    assert all_years.set_index("offense").loc["larceny", "weight_ours"] < 1.0


def test_temporal_weights_do_not_move_when_truth_after_the_cutoff_moves(world, monkeypatch) -> None:
    before = _evaluation(world).fold_blend_table(_spec(_evaluation(world), "temporal_through_2021"))
    source = world.state_dir / "eval" / "src"
    manifest = json.loads((source / "run_manifest.json").read_text())
    truth = _truth()
    truth.loc[truth["year"].gt(2021), "incident_count"] *= 50.0
    truth_path = Path(manifest["inputs"]["city_truth"]["path"])
    truth.to_parquet(truth_path, index=False)
    manifest["inputs"]["city_truth"]["sha256"] = gold_module._sha256(truth_path)
    (source / "run_manifest.json").write_text(json.dumps(manifest))
    after = _evaluation(world).fold_blend_table(_spec(_evaluation(world), "temporal_through_2021"))
    assert _larceny(after) == _larceny(before)


def test_a_variant_blend_is_fold_scoped_not_passed_through(world) -> None:
    variant_path = world.repo_root / "configs" / "blend_variant.csv"
    _blend_config(model=0.7, population=0.1, exposure=0.2).to_csv(variant_path, index=False)
    evaluation = _evaluation(
        world,
        run_id="f1__blend",
        variant="blend",
        variant_args=("--finished-share-blend-path", "configs/blend_variant.csv"),
    )
    spec = _spec(evaluation, "spatial_austin")
    command = evaluation.build_command(spec)
    assert "configs/blend_variant.csv" not in command
    assert command.count("--finished-share-blend-path") == 1
    assert evaluation.blend_base()[0] == variant_path


def test_the_model_fold_and_an_identity_blend_take_no_fold_table(world) -> None:
    evaluation = _evaluation(world)
    assert evaluation.fold_blend_path(_spec(evaluation, "model_fold0")) is None
    _blend_config(model=1.0, population=0.0, exposure=0.0).to_csv(
        world.repo_root / "configs" / "finished_share_blend_v1.csv", index=False
    )
    assert evaluation.fold_blend_path(_spec(evaluation, "spatial_austin")) is None
    assert "--finished-share-blend-path" not in evaluation.build_command(_spec(evaluation, "spatial_austin"))


# --- #9 the fold's own stage-10b prior ----------------------------------------------------------


def test_held_out_folds_build_and_read_a_fold_scoped_prior(world) -> None:
    evaluation = _evaluation(world)
    spatial = _spec(evaluation, "spatial_boston")
    prior = evaluation.prior_command(spatial)
    assert prior[prior.index("--exclude-feed-city") + 1] == "boston"
    assert prior[prior.index("--out") + 1] == str(evaluation.fold_prior_path(spatial))
    build = evaluation.build_command(spatial)
    assert build[build.index("--bg-prior-path") + 1] == str(evaluation.fold_prior_path(spatial))

    temporal = _spec(evaluation, "temporal_through_2023")
    prior = evaluation.prior_command(temporal)
    assert "--exclude-feed-city" not in prior
    assert prior[prior.index("--feed-year-end") + 1] == "2023"

    model = _spec(evaluation, "model_fold0")
    assert evaluation.prior_command(model) is None
    assert "--bg-prior-path" not in evaluation.build_command(model)
    with pytest.raises(ValueError, match="harness-controlled"):
        _evaluation(world, run_id="x__p", variant="p", variant_args=("--bg-prior-path", "a.parquet"))


# --- #13 resume binds to content --------------------------------------------------------------


def _stub_run(monkeypatch, world: RepoPaths) -> list[str]:
    ran: list[str] = []
    monkeypatch.setattr(GoldEvaluation, "pin_truth", lambda self, prior_manifest: {"city_truth": {"sha256": "x"}})

    def fold(self, spec, *, ags_path):
        ran.append(spec.fold_id)
        assert ags_path is None
        return {**spec.__dict__, "wall_seconds": 1.0}, pd.DataFrame({"x": [1]})

    monkeypatch.setattr(GoldEvaluation, "_run_fold", fold)
    monkeypatch.setattr(
        GoldEvaluation,
        "write_results",
        lambda self, out_dir, cells: {"results": _csv(out_dir / "results.csv")},
    )

    def no_ags(paths):
        raise AssertionError("the AGS comparator is off")

    monkeypatch.setattr(gold_module, "ensure_ags_tract_cache", no_ags)
    return ran


def _csv(path: Path) -> Path:
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    return path


def test_resume_is_refused_when_code_or_state_content_changed(world, monkeypatch) -> None:
    code = world.repo_root / "src" / "crimerisk" / "allocation.py"
    code.parent.mkdir(parents=True)
    code.write_text("A = 1\n")
    state = world.state_dir / "modeling" / "bg_prior_long_2025_arm_b.parquet"
    state.parent.mkdir(parents=True)
    pd.DataFrame({"a": [1.0]}).to_parquet(state, index=False)
    config = dict(folds=("spatial",), cities=("austin", "boston"), ags_comparator=False)
    evaluation = _evaluation(world, **config)
    fingerprint = evaluation.dependency_fingerprint()
    assert {"src/crimerisk/allocation.py", "state/modeling/bg_prior_long_2025_arm_b.parquet",
            "configs/finished_share_blend_v1.csv", "state/eval/src/posthoc/blend_xfit/weights.csv"} <= set(fingerprint)

    ran = _stub_run(monkeypatch, world)
    manifest_path = evaluation.run_dir / "run_manifest.json"
    evaluation.run_dir.mkdir(parents=True)

    def interrupted(dependencies) -> None:
        manifest_path.write_text(
            json.dumps(
                {
                    "config": json.loads(json.dumps(gold_module.asdict(evaluation.config))),
                    "folds": [],
                    **({"dependencies": dependencies} if dependencies is not None else {}),
                }
            )
        )

    interrupted(None)
    with pytest.raises(RuntimeError, match="no dependency fingerprint"):
        evaluation.run()

    interrupted(fingerprint)
    code.write_text("A = 2\n")
    with pytest.raises(RuntimeError, match="src/crimerisk/allocation.py"):
        evaluation.run()

    code.write_text("A = 1\n")
    pd.DataFrame({"a": [2.0]}).to_parquet(state, index=False)
    with pytest.raises(RuntimeError, match="bg_prior_long_2025_arm_b"):
        evaluation.run()
    assert ran == []

    pd.DataFrame({"a": [1.0]}).to_parquet(state, index=False)
    evaluation.run()
    assert ran == ["spatial_austin", "spatial_boston"]
    assert json.loads(manifest_path.read_text())["dependencies"] == fingerprint


# --- AGS optional; E4 panel from a manifest -------------------------------------------------------


def _surface(path: Path, *, tract: bool) -> None:
    ids = ["48001000100"] if tract else ["480010001001", "480010001002", "480010001003"]
    frame = pd.DataFrame({"tract_id": ["48001000100"] * len(ids)}) if not tract else pd.DataFrame(
        {"tract_id": ids}
    )
    if not tract:
        frame["block_group_geoid"] = ids
        frame["eb_jurisdiction_id"] = "austin"
    frame["population_2025"] = 100.0 * len(ids) if tract else 100.0
    for offense in OFFENSES_7:
        base = np.arange(1, len(ids) + 1, dtype=float)
        for column in (
            f"expected_count_{offense}", f"primary_denominator_{offense}",
            f"expected_count_{offense}_p10", f"expected_count_{offense}_p50",
            f"expected_count_{offense}_p90", f"index_{offense}_primary_p25",
            f"index_{offense}_primary_p75", f"rate_{offense}_primary_ci95_lower",
            f"rate_{offense}_primary_ci95_upper",
        ):
            frame[column] = base
        frame[f"primary_national_rate_per_100k_{offense}"] = 100.0
        frame[f"uncertainty_support_class_{offense}"] = "direct"
    frame.to_parquet(path, index=False)


def test_feed_folds_score_every_other_arm_without_the_ags_comparator(world, tmp_path) -> None:
    evaluation = _evaluation(world, ags_comparator=False)
    bg_path, tract_path = tmp_path / "bg.parquet", tmp_path / "tract.parquet"
    _surface(bg_path, tract=False)
    _surface(tract_path, tract=True)
    spec = _spec(evaluation, "spatial_austin")
    truth = evaluation.truth[evaluation.truth["city_key"].eq("austin")]
    scores = evaluation._score_feed_candidate(
        spec, bg_path=bg_path, tract_path=tract_path, truth=truth, city_keys=["austin"], ags_path=None
    )
    assert set(scores["arm"]) == {"ours", "population", "primary_exposure", "ours_tract"}
    retained = pd.concat(evaluation._fold_input_parts, ignore_index=True)
    assert "prediction_ags_2022a_tract" not in retained.columns


def test_the_e4_panel_is_read_from_its_manifest_and_digest_checked(world) -> None:
    source = world.repo_root / "panel"
    source.mkdir()
    files = {}
    for name in ("truth", "registry", "frozen_cells"):
        path = source / f"{name}.csv"
        path.write_text(f"{name}\n")
        files[name] = {"path": f"panel/{name}.csv", "sha256": gold_module._sha256(path)}
    files["frozen_cells"]["variant"] = "base"
    manifest = {
        "panel_id": "p1",
        **files,
        "multi_jurisdiction_sources": ["lasd"],
        "cell_rules": {"min_tracts": 20, "min_kept_incidents": 50.0, "min_kept_share": 0.5},
        "expected": {"frozen_cells": 1},
    }
    (world.repo_root / "configs" / "p1.json").write_text(json.dumps(manifest))
    panel = load_e4_panel(world.repo_root, "configs/p1.json")
    assert panel["truth"]["resolved_path"] == source / "truth.csv"
    (source / "registry.csv").write_text("edited\n")
    with pytest.raises(ValueError, match="does not match"):
        load_e4_panel(world.repo_root, "configs/p1.json")


def test_the_shipped_e4_panel_names_the_frozen_benchmark() -> None:
    root = Path(__file__).resolve().parents[1]
    panel = json.loads((root / "configs" / "e4_panel_v1.json").read_text())
    assert panel["expected"]["scored_cells"] == 190
    assert panel["expected"]["scored_jurisdictions"] == 40
    assert panel["frozen_cells"]["variant"] == "base"
