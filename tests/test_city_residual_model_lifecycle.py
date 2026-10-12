"""The residual model lifecycle: training inputs -> fitted artifact -> predictions (review #2)."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from crimerisk.city_residuals import (
    CityResidualConfig,
    CityResidualFittedModel,
    apply_city_residual_feature_policy,
    apply_city_residual_model,
    resolve_city_residual_model,
)
from crimerisk.model_surface import _build_feature_selection


def test_feature_policy_carries_semantic_class_across_release_year(tmp_path) -> None:
    policy_path = tmp_path / "feature_policy.parquet"
    pd.DataFrame(
        {
            "feature_column": ["median_rent_2024", "feature_x"],
            "final_class": ["proxy_review", "between_only"],
            "feature_group": ["socioeconomic_household", "test"],
        }
    ).to_parquet(policy_path, index=False)

    selected, application = apply_city_residual_feature_policy(
        ["median_rent_2025", "feature_x"],
        feature_policy_path=policy_path,
        exclude_feature_policy_classes=("between_only",),
    )

    assert selected == ["median_rent_2025"]
    assert application["annual_policy_aliases"] == {
        "median_rent_2025": "median_rent_2024"
    }


def test_raw_release_population_is_not_a_model_feature() -> None:
    frame = pd.DataFrame(
        {
            "population_2025": [100.0, 200.0, 300.0],
            "feature_signal": [0.1, 0.3, 0.2],
        }
    )

    selected = _build_feature_selection(frame)

    assert "population_2025" not in selected
    assert "feature_signal" in selected


def _fitted_constant_residual() -> CityResidualFittedModel:
    model = DummyRegressor(strategy="constant", constant=0.25)
    model.fit([[0.0, 0.5, -0.6931471805599453, 1.0]], [0.25])
    return CityResidualFittedModel(
        model=model,
        burglary_model=None,
        feature_cols=("feature_x",),
        offense_cols=("offense_robbery",),
        config=CityResidualConfig(),
        training_row_count=1,
        training_city_count=1,
        training_incident_total=1.0,
    )


def _binding(tmp_path):
    source = tmp_path / "training.parquet"
    source.write_bytes(b"training labels")
    prior = tmp_path / "prior.parquet"
    prior.write_bytes(b"prior")
    return [source, prior], {"year": 2025, "exclude_feed_city_keys": [], "feed_year_end": None}


def test_the_artifact_is_fitted_once_and_then_read(tmp_path) -> None:
    dependencies, parameters = _binding(tmp_path)
    model_path = tmp_path / "city_residual_model_2025.joblib"
    fits = []

    def fit():
        fits.append(1)
        return _fitted_constant_residual()

    fitted, record = resolve_city_residual_model(
        path=model_path, dependencies=dependencies, parameters=parameters, fit=fit, allow_fit=True
    )
    assert record["fitted_in_this_run"] and len(fits) == 1
    again, record = resolve_city_residual_model(
        path=model_path, dependencies=dependencies, parameters=parameters, fit=fit, allow_fit=False
    )
    assert not record["fitted_in_this_run"] and len(fits) == 1
    assert isinstance(again, CityResidualFittedModel)
    frame = pd.DataFrame(
        {
            "jurisdiction_id": ["j", "j"],
            "state_fips": ["01", "01"],
            "offense": ["robbery", "robbery"],
            "model_share": [0.25, 0.75],
            "feature_x": [1.0, 2.0],
        }
    )
    scored = apply_city_residual_model(frame, fitted=again)
    assert scored["predicted_log_ratio"].eq(0.25).all()
    assert scored["residual_model_share"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("change", ["training_labels", "year", "fold_scope"])
def test_a_changed_binding_is_refused_without_a_refit(tmp_path, change) -> None:
    dependencies, parameters = _binding(tmp_path)
    model_path = tmp_path / "city_residual_model_2025.joblib"
    resolve_city_residual_model(
        path=model_path,
        dependencies=dependencies,
        parameters=parameters,
        fit=_fitted_constant_residual,
        allow_fit=True,
    )
    if change == "training_labels":
        dependencies[0].write_bytes(b"changed training labels")
    elif change == "year":
        parameters = {**parameters, "year": 2024}
    else:
        parameters = {**parameters, "exclude_feed_city_keys": ["denver"]}
    with pytest.raises(RuntimeError, match="stale"):
        resolve_city_residual_model(
            path=model_path,
            dependencies=dependencies,
            parameters=parameters,
            fit=_fitted_constant_residual,
            allow_fit=False,
        )


def test_a_replaced_artifact_is_refused(tmp_path) -> None:
    dependencies, parameters = _binding(tmp_path)
    model_path = tmp_path / "city_residual_model_2025.joblib"
    resolve_city_residual_model(
        path=model_path,
        dependencies=dependencies,
        parameters=parameters,
        fit=_fitted_constant_residual,
        allow_fit=True,
    )
    model_path.write_bytes(model_path.read_bytes() + b"\0")
    with pytest.raises(RuntimeError, match="stale"):
        resolve_city_residual_model(
            path=model_path,
            dependencies=dependencies,
            parameters=parameters,
            fit=_fitted_constant_residual,
            allow_fit=False,
        )


def _residual_training(n_cities: int = 12, rows_per_city: int = 40):
    import numpy as np

    rng = np.random.default_rng(0)
    groups = pd.Series(np.repeat([f"city_{k:02d}" for k in range(n_cities)], rows_per_city))
    x = pd.DataFrame({"feature_x": rng.normal(size=len(groups)), "feature_y": rng.normal(size=len(groups))})
    y = 0.5 * x["feature_x"].to_numpy() + rng.normal(scale=0.1, size=len(groups))
    w = rng.uniform(0.5, 2.0, size=len(groups))
    return x, y, w, groups


def test_the_residual_model_is_the_mean_of_city_keyed_bagged_fits() -> None:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    from crimerisk.city_residuals import CityBaggedRegressor, _fit_residual_regressor

    x, y, w, groups = _residual_training()
    config = CityResidualConfig(hist_max_iter=20)
    bagged = _fit_residual_regressor(x, y, w, groups=groups, config=config)
    assert isinstance(bagged, CityBaggedRegressor)
    assert len(bagged.estimators_) == config.bag_count == 10
    assert all(0 < len(cities) < 12 for cities in bagged.bag_cities_)
    assert len(set(bagged.bag_cities_)) > 1
    manual = np.mean([model.predict(x) for model in bagged.estimators_], axis=0)
    assert np.allclose(bagged.predict(x), manual)
    again = _fit_residual_regressor(x, y, w, groups=groups, config=config)
    assert np.array_equal(again.predict(x), bagged.predict(x))

    single = _fit_residual_regressor(x, y, w, groups=groups, config=CityResidualConfig(hist_max_iter=20, bag_count=1))
    assert isinstance(single, HistGradientBoostingRegressor)


def test_dropping_a_city_leaves_every_other_citys_bags_unchanged() -> None:
    from crimerisk.city_residuals import _fit_residual_regressor

    x, y, w, groups = _residual_training()
    config = CityResidualConfig(hist_max_iter=5)
    full = _fit_residual_regressor(x, y, w, groups=groups, config=config)
    keep = groups.ne("city_03").to_numpy()
    dropped = _fit_residual_regressor(
        x.loc[keep].reset_index(drop=True), y[keep], w[keep], groups=groups[keep].reset_index(drop=True), config=config
    )
    for with_city, without_city in zip(full.bag_cities_, dropped.bag_cities_):
        assert tuple(city for city in with_city if city != "city_03") == without_city


def test_a_bagged_residual_artifact_is_read_back_with_the_same_predictions(tmp_path) -> None:
    import numpy as np

    from crimerisk.city_residuals import CityBaggedRegressor, _fit_residual_regressor, design_city_residual_matrix

    raw, y, w, groups = _residual_training()
    frame = pd.DataFrame(
        {
            "jurisdiction_id": groups.to_numpy(),
            "state_fips": "01",
            "offense": "robbery",
            "model_share": 1.0 / 40,
            "feature_x": raw["feature_x"].to_numpy(),
            "feature_y": raw["feature_y"].to_numpy(),
        }
    )
    feature_cols, offense_cols = ("feature_x", "feature_y"), ("offense_robbery",)
    x = design_city_residual_matrix(frame, feature_cols=feature_cols, offense_cols=offense_cols)
    config = CityResidualConfig(hist_max_iter=20)
    fitted = CityResidualFittedModel(
        model=_fit_residual_regressor(x, y, w, groups=groups, config=config),
        burglary_model=None,
        feature_cols=feature_cols,
        offense_cols=offense_cols,
        config=config,
        training_row_count=len(x),
        training_city_count=int(groups.nunique()),
        training_incident_total=1.0,
    )
    dependencies, parameters = _binding(tmp_path)
    model_path = tmp_path / "city_residual_model_2025.joblib"
    resolve_city_residual_model(
        path=model_path, dependencies=dependencies, parameters=parameters, fit=lambda: fitted, allow_fit=True
    )
    again, record = resolve_city_residual_model(
        path=model_path, dependencies=dependencies, parameters=parameters, fit=lambda: fitted, allow_fit=False
    )
    assert not record["fitted_in_this_run"] and isinstance(again.model, CityBaggedRegressor)
    before = apply_city_residual_model(frame, fitted=fitted)
    after = apply_city_residual_model(frame, fitted=again)
    assert before["predicted_log_ratio"].abs().gt(0).any()
    assert np.array_equal(before["predicted_log_ratio"].to_numpy(), after["predicted_log_ratio"].to_numpy())
    assert np.allclose(after.groupby("jurisdiction_id")["residual_model_share"].sum(), 1.0)
