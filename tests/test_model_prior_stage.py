from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crimerisk import allocation, model_surface
from crimerisk.allocation import AllocationBuildConfig, _build_bg_prior_long
from crimerisk.model_surface import (
    JurisdictionBaggedRegressor,
    ModelSurfaceConfig,
    _bag_membership,
    _build_model,
    model_prior_catalog_path,
    model_prior_is_current,
    model_prior_path,
    write_model_prior,
)
from crimerisk.paths import RepoPaths


def _fake_build(calls):
    def build(*, paths, config, **_):
        calls.append(config)
        prior = pd.DataFrame(
            {"bg_id": ["010010201001"], "offense": ["burglary"], "bg_weight": [float(len(calls))]}
        )
        catalog = pd.DataFrame(
            {"offense": ["burglary"], "feature_column": ["jobs_density_sqkm"], "excluded_by_feature_policy": [False]}
        )
        return prior, pd.DataFrame(), catalog

    return build


@pytest.fixture
def repo(tmp_path, monkeypatch):
    calls: list[ModelSurfaceConfig] = []
    monkeypatch.setattr(model_surface, "build_model_surface", _fake_build(calls))
    paths = RepoPaths.from_repo_root(tmp_path)
    controls = paths.state_dir / "controls" / "jurisdiction_controls_2024.parquet"
    controls.parent.mkdir(parents=True)
    pd.DataFrame({"jurisdiction_id": ["a"], "value": [1.0]}).to_parquet(controls, index=False)
    policy = paths.state_dir / "modeling" / "feature_transfer_policy_2024.parquet"
    policy.parent.mkdir(parents=True)
    pd.DataFrame({"feature_column": ["jobs_density_sqkm"], "final_class": ["between_and_within"]}).to_parquet(
        policy, index=False
    )
    config = ModelSurfaceConfig(
        year=2024,
        compute_diagnostics=False,
        prior_anchor="offense_denominator",
        feature_policy_path=policy,
        exclude_feature_policy_classes=("between_only", "excluded_protected"),
    )
    return paths, config, controls, calls


def test_prior_is_stamped_and_goes_stale_on_input_or_configuration(repo):
    paths, config, controls, calls = repo
    prior_path = model_prior_path(paths, year=2024)

    write_model_prior(paths=paths, config=config, prior_path=prior_path)
    assert model_prior_catalog_path(prior_path).exists()
    assert model_prior_is_current(paths, config=config, prior_path=prior_path)

    _, summary = write_model_prior(paths=paths, config=config, prior_path=prior_path)
    assert summary["reused"] is True
    assert len(calls) == 1

    with_proxy = replace(
        config, exclude_feature_policy_classes=(*config.exclude_feature_policy_classes, "proxy_review")
    )
    assert not model_prior_is_current(paths, config=with_proxy, prior_path=prior_path)

    pd.DataFrame({"jurisdiction_id": ["a"], "value": [2.0]}).to_parquet(controls, index=False)
    assert not model_prior_is_current(paths, config=config, prior_path=prior_path)


def test_stage_12_refuses_a_prior_it_cannot_prove_current(repo):
    paths, _config, controls, calls = repo
    build = AllocationBuildConfig(year=2024)
    prior_path = model_prior_path(paths, year=2024)

    # A prior with no stamp, as the pre-2025.2 cache was.
    pd.DataFrame({"bg_id": ["010010201001"], "offense": ["burglary"], "bg_weight": [9.0]}).to_parquet(
        prior_path, index=False
    )
    with pytest.raises(RuntimeError, match="build-model-prior"):
        _build_bg_prior_long(paths, config=build)

    rebuilt = _build_bg_prior_long(paths, config=replace(build, force_bg_prior_rebuild=True))
    assert rebuilt["bg_weight"].tolist() == [1.0]
    assert _build_bg_prior_long(paths, config=build)["bg_weight"].tolist() == [1.0]

    pd.DataFrame({"jurisdiction_id": ["a"], "value": [3.0]}).to_parquet(controls, index=False)
    with pytest.raises(RuntimeError, match="stale"):
        _build_bg_prior_long(paths, config=build)
    assert len(calls) == 1
    assert allocation._is_default_step14_arm_b_model_surface(calls[0])


def test_the_prior_stamp_binds_its_feature_catalog_and_every_setting(repo):
    paths, config, _controls, calls = repo
    prior_path = model_prior_path(paths, year=2024)
    write_model_prior(paths=paths, config=config, prior_path=prior_path)
    assert model_prior_is_current(paths, config=config, prior_path=prior_path)

    # A numerical setting the old parameter record did not carry.
    assert not model_prior_is_current(paths, config=replace(config, hist_max_depth=6), prior_path=prior_path)
    assert not model_prior_is_current(
        paths, config=replace(config, use_state_fixed_effects=False), prior_path=prior_path
    )
    # The bagging count, fraction and seed.
    for change in ({"hist_bag_count": 1}, {"hist_bag_fraction": 0.7}, {"hist_bag_seed": 1}):
        assert not model_prior_is_current(paths, config=replace(config, **change), prior_path=prior_path)
    # compute_diagnostics does not change the prior.
    assert model_prior_is_current(paths, config=replace(config, compute_diagnostics=True), prior_path=prior_path)

    # A replaced catalog invalidates the prior it was written with.
    catalog = model_prior_catalog_path(prior_path)
    pd.DataFrame(
        {"offense": ["burglary"], "feature_column": ["median_rent"], "excluded_by_feature_policy": [False]}
    ).to_parquet(catalog, index=False)
    assert not model_prior_is_current(paths, config=config, prior_path=prior_path)


def test_a_fold_prior_is_its_own_artifact(repo):
    paths, config, _controls, calls = repo
    fold = replace(config, exclude_feed_city_keys=("denver",), feed_year_end=2021)
    fold_path = paths.state_dir / "eval" / "fold" / "bg_prior_long_2024_fold.parquet"
    write_model_prior(paths=paths, config=fold, prior_path=fold_path)
    assert calls[-1].exclude_feed_city_keys == ("denver",) and calls[-1].feed_year_end == 2021
    assert model_prior_is_current(paths, config=fold, prior_path=fold_path)
    assert not model_prior_is_current(paths, config=config, prior_path=fold_path)
    # A cutoff at the build year is no cutoff.
    production_path = model_prior_path(paths, year=2024)
    write_model_prior(paths=paths, config=config, prior_path=production_path)
    assert model_prior_is_current(paths, config=replace(config, feed_year_end=2024), prior_path=production_path)


def test_a_fold_build_refuses_the_release_prior(repo):
    paths, _config, _controls, _calls = repo
    with pytest.raises(RuntimeError, match="reads its own prior"):
        _build_bg_prior_long(paths, config=AllocationBuildConfig(year=2024, exclude_feed_city_keys=("denver",)))


def _training_matrix(n: int = 400) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n)},
        index=pd.Index([f"j{i:04d}" for i in range(n)], name="jurisdiction_id"),
    )
    return x, x["a"].to_numpy() * 2.0 + rng.normal(scale=0.1, size=n)


def test_the_default_prior_model_is_ten_keyed_shallow_fits_on_80_percent_subsamples():
    model = _build_model(ModelSurfaceConfig())
    assert isinstance(model, JurisdictionBaggedRegressor)
    assert (model.count, model.fraction, model.seed) == (10, 0.8, 20260926)
    inner = model.estimator.get_params()
    assert (inner["max_depth"], inner["min_samples_leaf"], inner["l2_regularization"]) == (3, 100, 10.0)
    assert not isinstance(_build_model(ModelSurfaceConfig(hist_bag_count=1)), JurisdictionBaggedRegressor)

    x, y = _training_matrix()
    fitted = JurisdictionBaggedRegressor(
        _build_model(ModelSurfaceConfig(hist_bag_count=1, hist_max_iter=20)), count=3, fraction=0.8, seed=7
    ).fit(x, y)
    members = [np.asarray(m.predict(x)) for m in fitted.estimators_]
    assert np.allclose(fitted.predict(x), np.mean(members, axis=0))
    sizes = [int(_bag_membership(x.index, seed=7, member=k, fraction=0.8).sum()) for k in range(3)]
    assert [m.n_features_in_ for m in fitted.estimators_] == [2, 2, 2]
    assert all(280 <= size <= 360 for size in sizes) and len(set(sizes)) > 1


def test_bag_membership_is_a_function_of_each_jurisdiction_only():
    x, _ = _training_matrix()
    full = _bag_membership(x.index, seed=20260926, member=3, fraction=0.8)
    kept = x.index[5:]
    assert np.array_equal(_bag_membership(kept, seed=20260926, member=3, fraction=0.8), full[5:])
    assert not np.array_equal(full, _bag_membership(x.index, seed=20260926, member=4, fraction=0.8))
    assert not np.array_equal(full, _bag_membership(x.index, seed=1, member=3, fraction=0.8))


def test_bagging_refuses_duplicate_keys():
    x, y = _training_matrix(20)
    x.index = pd.Index(["same"] * 20)
    with pytest.raises(ValueError, match="unique"):
        _build_model(ModelSurfaceConfig(hist_max_iter=5)).fit(x, y)
