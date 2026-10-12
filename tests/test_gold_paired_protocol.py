"""The 2025.2 gold protocol: paired matched-set tables, pinned and checked truth, a fixed
fold set, named variants, scoring-only E4, post-hoc arms and the E4 validity labels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk.eval import GoldEvaluation, GoldEvaluationConfig
from crimerisk.eval import gold as gold_module
from crimerisk.eval.gold import (
    BLEND_GRID,
    FoldSpec,
    _e4_spill_keys,
    _log_score,
    _reuse_flags,
    assert_truth_complete,
    cross_fitted_blend,
    truth_completeness,
)
from crimerisk.paths import RepoPaths


def _evaluator(**overrides) -> GoldEvaluation:
    evaluator = GoldEvaluation.__new__(GoldEvaluation)
    evaluator.config = GoldEvaluationConfig(
        run_id=overrides.pop("run_id", "test"), bootstrap_iterations=200, **overrides
    )
    return evaluator


def _cell(city: str, arm: str, tvd: float, **extra) -> dict[str, object]:
    return {
        "fold_id": f"spatial_{city}",
        "fold_type": "spatial",
        "city_name": city,
        "jurisdiction_id": city,
        "ags_uses_city_feed": False,
        "offense": "robbery",
        "arm": arm,
        "support": "block_group",
        "n_cells": 10,
        "n_incidents": 100.0,
        "cell_scope": "primary",
        "reuse_flags": "tau=not_seen",
        "tvd": tvd,
        "log_score": -tvd,
        "top_decile_capture": 1.0 - tvd,
        "w1_spatial_skill": 0.5 - tvd,
        "spearman": 0.5,
        **extra,
    }


# --- paired, matched-set summaries ----------------------------------------------------------


def test_arms_are_compared_on_the_cities_where_every_arm_is_defined() -> None:
    cells = pd.DataFrame(
        [
            _cell("a", "ours", 0.2), _cell("a", "population", 0.4), _cell("a", "past_counts", 0.3),
            _cell("b", "ours", 0.3), _cell("b", "population", 0.5), _cell("b", "past_counts", 0.2),
            # City c has no past counts: it leaves the whole block, not just that arm.
            _cell("c", "ours", 0.9), _cell("c", "population", 0.1),
            _cell("c", "past_counts", float("nan")),
        ]
    )
    arms, differences = _evaluator()._paired_summary(cells)
    tvd = arms[arms["metric"].eq("tvd")].set_index("arm")
    assert set(tvd["n_cities"]) == {2}
    assert set(tvd["cities"]) == {"a;b"}
    assert math.isclose(tvd.loc["ours", "estimate"], 0.25)
    assert math.isclose(tvd.loc["population", "estimate"], 0.45)
    delta = differences[
        differences["metric"].eq("tvd")
        & differences["arm"].eq("ours")
        & differences["reference_arm"].eq("population")
    ].iloc[0]
    assert math.isclose(delta["estimate"], -0.2)


def test_every_arm_and_difference_in_a_block_share_one_set_of_resample_indices() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for index in range(8):
        base = float(rng.uniform(0.2, 0.6))
        rows += [_cell(f"c{index}", "ours", base), _cell(f"c{index}", "population", base + 0.1)]
    arms, differences = _evaluator()._paired_summary(pd.DataFrame(rows))
    tvd = arms[arms["metric"].eq("tvd")].set_index("arm")
    delta = differences[
        differences["metric"].eq("tvd") & differences["arm"].eq("ours")
    ].iloc[0]
    # A constant per-city gap has a degenerate paired interval, and each arm's interval is the
    # other's shifted by exactly that gap: the same draws were used for both.
    assert math.isclose(delta["ci95_lower"], -0.1) and math.isclose(delta["ci95_upper"], -0.1)
    assert math.isclose(tvd.loc["population", "ci95_lower"] - tvd.loc["ours", "ci95_lower"], 0.1)
    assert math.isclose(tvd.loc["population", "ci95_upper"] - tvd.loc["ours", "ci95_upper"], 0.1)


def test_primary_rows_pool_all_cities_and_ags_rows_keep_feed_free_cities() -> None:
    cells = pd.DataFrame(
        [
            _cell("free", "ours", 0.2), _cell("free", "population", 0.3),
            _cell("fed", "ours", 0.4, ags_uses_city_feed=True),
            _cell("fed", "population", 0.5, ags_uses_city_feed=True),
            _cell("free", "ours_tract", 0.2), _cell("free", "ags_2022a_tract", 0.25),
            _cell("fed", "ours_tract", 0.4, ags_uses_city_feed=True),
            _cell("fed", "ags_2022a_tract", 0.1, ags_uses_city_feed=True),
        ]
    )
    arms, _differences = _evaluator()._paired_summary(cells)
    tvd = arms[arms["metric"].eq("tvd")]
    primary = tvd[tvd["family"].eq("primary")]
    assert set(primary["cities"]) == {"fed;free"} and set(primary["city_set"]) == {"all"}
    ags = tvd[tvd["family"].eq("ags_tract")]
    assert set(ags["cities"]) == {"free"} and set(ags["city_set"]) == {"ags_feed_free"}


def test_candidate_minus_baseline_is_a_paired_difference() -> None:
    cells = pd.DataFrame(
        [
            _cell("a", "ours", 0.20), _cell("a", "baseline_ours", 0.25),
            _cell("b", "ours", 0.30), _cell("b", "baseline_ours", 0.33),
            _cell("a", "population", 0.4), _cell("b", "population", 0.5),
        ]
    )
    _arms, differences = _evaluator()._paired_summary(cells)
    row = differences[
        differences["metric"].eq("tvd") & differences["reference_arm"].eq("baseline_ours")
    ].iloc[0]
    assert row["arm"] == "ours"
    assert math.isclose(row["estimate"], -0.04)


def test_results_markdown_leads_with_paired_tables() -> None:
    evaluator = _evaluator()
    cells = pd.DataFrame(
        [
            _cell("a", "ours", 0.2), _cell("a", "population", 0.4),
            _cell("b", "ours", 0.3), _cell("b", "population", 0.5),
        ]
    )
    arms, differences = evaluator._paired_summary(cells)
    text = evaluator._markdown(arms, differences, pd.DataFrame(columns=[
        "offense", "uncertainty_support_class", "nominal", "empirical_coverage",
        "coverage_ci95_lower", "coverage_ci95_upper", "log_width_factor",
        "factor_ci95_lower", "factor_ci95_upper", "n_cities", "n_cells", "n_incidents",
    ]), pd.DataFrame())
    sections = [line for line in text.splitlines() if line.startswith("## ")]
    assert sections[0] == "## Ours minus each baseline arm"
    assert sections.index("## Matched-set arm estimates") > sections.index(
        "## Ours tract minus AGS 2022A (ags_uses_city_feed=false)"
    )
    assert "tuning corpus for ours; untouched for AGS" in text


# --- rare-offense log score -----------------------------------------------------------------


def test_log_score_is_mean_log_share_per_incident_with_a_uniform_floor() -> None:
    truth = np.array([3.0, 1.0, 0.0, 0.0])
    prediction = np.array([1.0, 0.0, 0.0, 0.0])
    floored = 0.99 * np.array([1.0, 0.0, 0.0, 0.0]) + 0.01 / 4
    expected = (3 * np.log(floored[0]) + np.log(floored[1])) / 4
    assert math.isclose(_log_score(truth, prediction), expected)
    assert math.isnan(_log_score(truth, np.zeros(4)))


# --- truth completeness ---------------------------------------------------------------------


def _reconciliation(counts: dict[tuple[str, int, str], float], raw: dict[tuple[str, int], float]):
    return pd.DataFrame(
        [
            {
                "city_key": city,
                "year": year,
                "offense": offense,
                "final_share_count": value,
                "raw_year_total": raw[(city, year)],
                "published_count": np.nan,
                "published_comparison_quality": None,
            }
            for (city, year, offense), value in counts.items()
        ]
    )


def _truth(counts: dict[tuple[str, int, str], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city_key": city, "year": year, "offense": offense, "incident_count": value}
            for (city, year, offense), value in counts.items()
        ]
    )


def test_a_collapsed_feed_year_fails_the_run_even_with_no_truth_rows_for_it() -> None:
    recon = {("ny", 2018, "murder"): 0.0, ("ny", 2024, "murder"): 20.0, ("ny", 2025, "murder"): 274.0}
    raw = {("ny", 2018): 78.0, ("ny", 2024): 8051.0, ("ny", 2025): 569300.0}
    truth = _truth({("ny", 2024, "murder"): 20.0, ("ny", 2025, "murder"): 274.0})
    table = truth_completeness(truth, _reconciliation(recon, raw))
    status = table.set_index("year")["status"]
    assert status[2018] == "collapse_raw_year"
    assert status[2024] == "collapse_raw_year"
    assert status[2025] == "ok"
    with pytest.raises(ValueError, match="ny collapse_raw_year years 2018,2024"):
        assert_truth_complete(table)


def test_an_offense_collapse_inside_a_healthy_year_fails() -> None:
    recon = {("a", 2023, "larceny"): 10.0, ("a", 2024, "larceny"): 5000.0,
             ("a", 2023, "burglary"): 900.0, ("a", 2024, "burglary"): 1000.0}
    raw = {("a", 2023): 9000.0, ("a", 2024): 9000.0}
    truth = _truth({("a", 2023, "larceny"): 10.0, ("a", 2024, "larceny"): 5000.0,
                    ("a", 2023, "burglary"): 900.0, ("a", 2024, "burglary"): 1000.0})
    table = truth_completeness(truth, _reconciliation(recon, raw))
    failed = table[table["status"].ne("ok")]
    assert failed[["year", "offense", "status"]].to_dict("records") == [
        {"year": 2023, "offense": "larceny", "status": "collapse_offense_year"}
    ]


def test_offenses_and_years_left_out_by_admission_do_not_fail() -> None:
    # Washington-shaped: non-murder admitted for the latest year only.
    recon = {("dc", 2018, "larceny"): 25000.0, ("dc", 2025, "larceny"): 17000.0,
             ("dc", 2018, "murder"): 160.0, ("dc", 2025, "murder"): 127.0,
             ("dc", 2018, "rape"): 279.0, ("dc", 2025, "rape"): 105.0}
    raw = {("dc", 2018): 33846.0, ("dc", 2025): 24149.0}
    truth = _truth({("dc", 2025, "larceny"): 17000.0, ("dc", 2018, "murder"): 160.0,
                    ("dc", 2025, "murder"): 127.0})
    table = truth_completeness(truth, _reconciliation(recon, raw))
    assert set(table.loc[table["offense"].eq("rape"), "status"]) == {"not_admitted"}
    assert set(table.loc[table["offense"].ne("rape"), "status"]) == {"ok"}
    assert_truth_complete(table)


def test_truth_that_disagrees_with_its_reconciliation_fails() -> None:
    recon = {("a", 2024, "robbery"): 1000.0}
    table = truth_completeness(
        _truth({("a", 2024, "robbery"): 400.0}), _reconciliation(recon, {("a", 2024): 5000.0})
    )
    assert table["status"].tolist() == ["truth_reconciliation_mismatch"]


# --- fold set is fixed; variants; scoring-only ----------------------------------------------


@pytest.fixture
def world(tmp_path: Path, monkeypatch) -> RepoPaths:
    paths = RepoPaths.from_repo_root(tmp_path)
    centroids = paths.data_dir / "tiger_bg" / "parsed"
    centroids.mkdir(parents=True)
    pd.DataFrame({"bg_id": ["480010001001"], "lon": [-97.74], "lat": [30.27]}).to_parquet(
        centroids / "bg_centroids.parquet", index=False
    )
    truth = pd.DataFrame(
        {
            "city_key": ["austin", "boston", "chicago"],
            "city_name": ["Austin", "Boston", "Chicago"],
            "jurisdiction_id": ["a", "b", "c"],
            "year": [2024, 2024, 2024],
            "offense": ["robbery"] * 3,
            "block_group_geoid": ["480010001001"] * 3,
            "incident_count": [3.0, 2.0, 1.0],
        }
    )
    monkeypatch.setattr(
        gold_module, "_active_truth", lambda paths, truth_path=None: (truth, ["austin", "boston", "chicago"])
    )
    return paths


def test_a_slow_spatial_fold_does_not_shrink_the_fold_set(world, monkeypatch) -> None:
    evaluation = GoldEvaluation(paths=world, config=GoldEvaluationConfig(run_id="slow", folds=("spatial",)))
    ags = world.state_dir / "ags.parquet"
    ags.parent.mkdir(parents=True, exist_ok=True)
    ags.write_bytes(b"ags")
    monkeypatch.setattr(gold_module, "ensure_ags_tract_cache", lambda paths: ags)
    monkeypatch.setattr(GoldEvaluation, "pin_truth", lambda self, prior_manifest: {"city_truth": {"sha256": "x"}})
    ran: list[str] = []

    def slow_fold(self, spec, *, ags_path):
        ran.append(spec.fold_id)
        record = {**spec.__dict__, "build_wall_seconds": 10_000.0, "wall_seconds": 10_000.0}
        return record, pd.DataFrame([_cell(str(spec.exclude_city_key), "ours", 0.2)])

    monkeypatch.setattr(GoldEvaluation, "_run_fold", slow_fold)
    monkeypatch.setattr(
        GoldEvaluation,
        "write_results",
        lambda self, out_dir, cells: {"results": _write(out_dir / "results.csv")},
    )
    evaluation.run()
    assert ran == ["spatial_austin", "spatial_boston", "spatial_chicago"]
    manifest = json.loads((evaluation.run_dir / "run_manifest.json").read_text())
    assert "spatial_reduction" not in manifest
    assert manifest["fold_ids"] == ran
    assert not hasattr(gold_module, "BUILD_TIMEOUT_REDUCTION_SECONDS")


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"run_id": ["slow"]}).to_csv(path, index=False)
    return path


def test_a_variant_is_named_in_the_run_id_and_appended_to_every_fold_build(world) -> None:
    config_file = world.repo_root / "configs" / "weights.csv"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("w\n1\n")
    config = GoldEvaluationConfig(
        run_id="B1__mixw",
        variant="mixw",
        variant_args=("--mixture-weights-path", "configs/weights.csv"),
    )
    evaluation = GoldEvaluation(paths=world, config=config)
    for spec in evaluation.fold_specs():
        if spec.candidate_dir is None:
            assert evaluation.build_command(spec)[-2:] == [
                "--mixture-weights-path", "configs/weights.csv"
            ]
    record = evaluation.variant_record()
    assert record["name"] == "mixw"
    assert set(record["file_sha256"]) == {"configs/weights.csv"}

    with pytest.raises(ValueError, match="must end in '__mixw'"):
        GoldEvaluation(paths=world, config=GoldEvaluationConfig(run_id="B1", variant="mixw"))
    with pytest.raises(ValueError, match="harness-controlled"):
        GoldEvaluation(
            paths=world,
            config=GoldEvaluationConfig(
                run_id="B1__x", variant="x", variant_args=("--feed-year-end=2021",)
            ),
        )


def test_scoring_only_reads_an_existing_candidate_and_leaves_it_in_place(world, monkeypatch) -> None:
    candidate = world.state_dir / "candidates" / "published"
    candidate.mkdir(parents=True)
    files = [
        candidate / "crimerisk_block_group_2025_ags_core.parquet",
        candidate / "crimerisk_tract_2025_ags_core.parquet",
        candidate / "manifest.json",
    ]
    for path in files:
        path.write_text("x")
    evaluation = GoldEvaluation(
        paths=world,
        config=GoldEvaluationConfig(
            run_id="e4", folds=("model",), model_candidate_dir=str(candidate)
        ),
    )
    (spec,) = evaluation.fold_specs()
    assert spec.fold_type == "model" and spec.candidate_dir == str(candidate.resolve())
    assert evaluation.dry_run_payload()[0]["command"] is None

    def no_build(*args, **kwargs):
        raise AssertionError("scoring-only must not build")

    def score(self, spec, *, bg_path, tract_path, ags_path, manifest_path):
        assert bg_path == files[0].resolve() and manifest_path == files[2].resolve()
        self._fold_input_parts = [pd.DataFrame({"x": [1]})]
        return pd.DataFrame([_cell("src", "ours", 0.3)])

    monkeypatch.setattr(gold_module.subprocess, "run", no_build)
    monkeypatch.setattr(GoldEvaluation, "_score_model_candidate", score)
    evaluation.scored_dir.mkdir(parents=True)
    evaluation.inputs_dir.mkdir(parents=True)
    record, _scores = evaluation._run_fold(spec, ags_path=world.state_dir / "ags.parquet")
    assert all(path.exists() for path in files)
    assert record["command"] is None
    assert set(record["surface_sha256"]) == {"block_group", "tract"}
    assert record["manifest_path"] == str(files[2].resolve())


# --- post-hoc cross-fitted blend ------------------------------------------------------------


def _retained(truths: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for city, truth in truths.items():
        for index, value in enumerate(truth):
            rows.append(
                {
                    "fold_id": f"spatial_{city}",
                    "fold_type": "spatial",
                    "city_name": city,
                    "offense": "robbery",
                    "support_id": f"{city}{index}",
                    "truth_count": value,
                    "population": [1.0, 1.0, 1.0][index],
                    "prediction_ours": [5.0, 1.0, 1.0][index],
                    "prediction_population": [1.0, 1.0, 1.0][index],
                    "prediction_primary_exposure": [1.0, 1.0, 5.0][index],
                }
            )
    return pd.DataFrame(rows)


def test_blend_weights_for_a_city_never_see_that_city() -> None:
    truths = {"a": [8.0, 1.0, 1.0], "b": [1.0, 1.0, 8.0], "c": [3.0, 3.0, 3.0]}
    _predictions, weights = cross_fitted_blend(_retained(truths))
    before = weights.set_index("city_name")
    changed = dict(truths, a=[1.0, 8.0, 1.0])
    _predictions, weights = cross_fitted_blend(_retained(changed))
    after = weights.set_index("city_name")
    columns = ["weight_ours", "weight_population", "weight_primary_exposure"]
    assert before.loc["a", columns].tolist() == after.loc["a", columns].tolist()
    assert before.loc["a", "training_cities"] == "b;c"


def test_blend_grid_is_the_tenth_simplex_starting_at_ours() -> None:
    assert len(BLEND_GRID) == 66
    assert BLEND_GRID[0] == (1.0, 0.0, 0.0)
    assert all(math.isclose(sum(point), 1.0) for point in BLEND_GRID)


def test_blend_predictions_are_the_weighted_component_shares() -> None:
    predictions, weights = cross_fitted_blend(
        _retained({"a": [8.0, 1.0, 1.0], "b": [7.0, 2.0, 1.0]})
    )
    row = weights.set_index("city_name").loc["a"]
    a = predictions[predictions["city_name"].eq("a")]
    expected = (
        row["weight_ours"] * np.array([5.0, 1.0, 1.0]) / 7.0
        + row["weight_population"] * np.array([1.0, 1.0, 1.0]) / 3.0
        + row["weight_primary_exposure"] * np.array([1.0, 1.0, 5.0]) / 7.0
    )
    assert np.allclose(a["prediction_blend_xfit"].to_numpy(float), expected)


# --- E4 validity labels ---------------------------------------------------------------------


def test_e4_spill_cells_are_a_department_outside_its_home_jurisdiction() -> None:
    widened = pd.DataFrame(
        {
            "source": ["detroit", "detroit", "lasd", "lasd"],
            "jurisdiction_id": ["detroit_city", "warren", "remainder", "contract_city"],
            "incident_count": [1000.0, 5.0, 900.0, 100.0],
        }
    )
    assert _e4_spill_keys(widened, multi_jurisdiction_sources=("lasd",)) == {("detroit", "warren")}


def test_residual_training_reuse_flag_marks_model_fold_jurisdictions_only() -> None:
    assert "residual_training_city=seen" in _reuse_flags(
        "model", "robbery", residual_training_seen=True
    )
    assert "residual_training_city=not_seen" in _reuse_flags(
        "spatial", "robbery", residual_training_seen=True
    )


def test_fold_specs_hold_spatial_and_model_feeds_through_the_build_year(world) -> None:
    evaluation = GoldEvaluation(paths=world, config=GoldEvaluationConfig(run_id="specs"))
    specs = {spec.fold_id: spec for spec in evaluation.fold_specs()}
    assert specs["spatial_austin"] == FoldSpec(
        "spatial_austin", "spatial", exclude_city_key="austin", feed_year_end=2025
    )
    assert specs["model_fold0"].feed_year_end == 2025
    assert specs["temporal_through_2021"].feed_year_end == 2021


def test_truth_is_pinned_only_after_it_passes_and_resumes_only_on_the_same_bytes(world) -> None:
    evaluation = GoldEvaluation(paths=world, config=GoldEvaluationConfig(run_id="pin"))
    modeling = world.state_dir / "modeling"
    modeling.mkdir(parents=True)
    recon_path = modeling / "city_incident_reconciliation_2025.parquet"
    raw = {(city, year): 1000.0 for city in ("austin", "boston", "chicago") for year in (2023, 2024)}
    counts = {
        (city, year, "robbery"): value
        for city, value in zip(("austin", "boston", "chicago"), (3.0, 2.0, 1.0))
        for year in (2023, 2024)
    }
    collapsed = {**raw, ("chicago", 2024): 10.0}
    _reconciliation(counts, collapsed).to_parquet(recon_path, index=False)
    with pytest.raises(ValueError, match="chicago collapse_raw_year"):
        evaluation.pin_truth(prior_manifest=None)
    assert not evaluation.truth_snapshot_path.exists()

    _reconciliation(counts, raw).to_parquet(recon_path, index=False)
    inputs = evaluation.pin_truth(prior_manifest=None)
    assert evaluation.truth_snapshot_path.exists()
    assert inputs["city_truth"]["year_end"] == 2025
    assert inputs["city_reconciliation"]["path"] == str(recon_path)
    with pytest.raises(RuntimeError, match="no longer matches"):
        evaluation.pin_truth(prior_manifest={"inputs": {"city_truth": {"sha256": "0" * 64}}})


def test_posthoc_blend_scores_from_retained_inputs_without_touching_the_run(world, monkeypatch) -> None:
    evaluation = GoldEvaluation(paths=world, config=GoldEvaluationConfig(run_id="b0"))
    evaluation.inputs_dir.mkdir(parents=True)
    truths = {"a": [8.0, 1.0, 1.0], "b": [7.0, 2.0, 1.0], "c": [6.0, 3.0, 1.0]}
    retained = _retained(truths).assign(
        support_role="primary",
        support_lon=lambda d: -97.0 - 0.01 * d.index.to_series().mod(3).to_numpy(),
        support_lat=30.0,
    )
    records = []
    for city, frame in retained.groupby("city_name"):
        path = evaluation.inputs_dir / f"spatial_{city}.parquet"
        frame.to_parquet(path, index=False)
        records.append(
            {
                "fold_id": f"spatial_{city}",
                "fold_type": "spatial",
                "scoring_schema_version": 5,
                "input_path": str(path),
                "input_sha256": gold_module._sha256(path),
            }
        )
    cells = pd.DataFrame(
        [_cell(city, arm, 0.3) for city in truths for arm in ("ours", "population")]
        # The E4 fold repeats a source across jurisdictions; post-hoc scoring must not trip on it.
        + [
            _cell("e4", "ours", 0.3, fold_id="model_scoring_only", fold_type="model",
                  jurisdiction_id=jurisdiction, cell_scope=scope)
            for jurisdiction, scope in (("j1", "primary"), ("j2", "secondary_e4_spill"))
        ]
    )
    cells.to_csv(evaluation.run_dir / "cell_scores.csv", index=False)
    (evaluation.run_dir / "run_manifest.json").write_text(
        json.dumps({"completed_at_utc": "x", "folds": records})
    )
    before = (evaluation.run_dir / "cell_scores.csv").read_bytes()
    monkeypatch.setattr(
        GoldEvaluation,
        "_calibration_results",
        lambda self: pd.DataFrame(columns=[
            "offense", "uncertainty_support_class", "nominal", "empirical_coverage",
            "coverage_ci95_lower", "coverage_ci95_upper", "log_width_factor",
            "factor_ci95_lower", "factor_ci95_upper", "n_cities", "n_cells", "n_incidents",
        ]),
    )
    results = evaluation.run_posthoc("blend_xfit")
    out = evaluation.run_dir / "posthoc" / "blend_xfit"
    blend = pd.read_csv(out / "cell_scores.csv")
    assert set(blend["arm"]) == {"blend_xfit"} and len(blend) == 3
    assert set(results.loc[results["metric"].eq("tvd"), "arm"]) == {
        "ours", "blend_xfit", "population"
    }
    differences = pd.read_csv(out / "differences.csv")
    assert {"blend_xfit"} <= set(differences["arm"])
    assert (evaluation.run_dir / "cell_scores.csv").read_bytes() == before
    assert json.loads((out / "manifest.json").read_text())["arm"] == "blend_xfit"
