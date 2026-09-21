from __future__ import annotations

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from crimerisk.city_residuals import (
    CityResidualConfig,
    CityResidualFittedModel,
    apply_city_residual_feature_policy,
    apply_city_residual_prediction_surface,
    load_city_residual_fitted_model,
    verify_city_residual_model_reproduction,
    write_city_residual_fitted_model,
)
from crimerisk.paths import RepoPaths
from crimerisk.model_surface import _build_feature_selection


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bg_id": ["010010201001", "010010201002"],
            "state_fips": ["01", "01"],
            "jurisdiction_id": ["01:municipal:test", "01:municipal:test"],
            "offense": ["robbery", "robbery"],
            "model_share": [0.25, 0.75],
        }
    )


def _predictions() -> pd.DataFrame:
    out = _frame().rename(columns={"model_share": "model_share_at_fit"})
    out["predicted_log_ratio"] = [0.0, 1.0]
    return out


def test_promoted_residual_predictions_are_normalized_within_control() -> None:
    out = apply_city_residual_prediction_surface(_frame(), predictions=_predictions())
    assert out["residual_model_share"].sum() == pytest.approx(1.0)
    assert out.loc[1, "residual_model_share"] > out.loc[1, "model_share"]


def test_promoted_residual_predictions_fail_on_model_share_drift() -> None:
    frame = _frame()
    frame.loc[0, "model_share"] = 0.30
    with pytest.raises(ValueError, match="do not match the current model-share input"):
        apply_city_residual_prediction_surface(frame, predictions=_predictions())


def test_promoted_residual_predictions_fail_on_missing_component() -> None:
    with pytest.raises(ValueError, match="does not cover the allocation frame"):
        apply_city_residual_prediction_surface(_frame(), predictions=_predictions().iloc[:1])


def test_promoted_residual_predictions_allow_unmodeled_zero_share_component() -> None:
    frame = _frame()
    frame.loc[1, "model_share"] = 0.0
    out = apply_city_residual_prediction_surface(frame, predictions=_predictions().iloc[:1])
    assert out.loc[1, "predicted_log_ratio"] == 0.0
    assert out.loc[1, "residual_model_share"] == 0.0


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


def test_reconstructed_model_must_reproduce_every_promoted_prediction(
    tmp_path, monkeypatch
) -> None:
    predictions = _predictions()
    predictions["predicted_log_ratio"] = 0.25
    features = _frame()[["bg_id", "state_fips"]].copy()
    features["feature_x"] = [1.0, 2.0]
    monkeypatch.setattr(
        "crimerisk.city_residuals._load_city_residual_bg_features",
        lambda **_: (features, ["feature_x"]),
    )

    result = verify_city_residual_model_reproduction(
        paths=RepoPaths.from_repo_root(tmp_path),
        year=2025,
        fitted=_fitted_constant_residual(),
        predictions=predictions,
        chunk_size=1,
    )
    assert result == {
        "rows": 2,
        "max_abs_log_ratio_delta": 0.0,
        "tolerance": 1e-8,
    }

    predictions.loc[1, "predicted_log_ratio"] = 0.3
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_city_residual_model_reproduction(
            paths=RepoPaths.from_repo_root(tmp_path),
            year=2025,
            fitted=_fitted_constant_residual(),
            predictions=predictions,
            chunk_size=1,
        )

    duplicate = pd.concat([predictions.iloc[:1], predictions.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate keys"):
        verify_city_residual_model_reproduction(
            paths=RepoPaths.from_repo_root(tmp_path),
            year=2025,
            fitted=_fitted_constant_residual(),
            predictions=duplicate,
        )


def test_serialized_reconstruction_fails_closed_when_an_input_changes(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"frozen-input")
    model_path = tmp_path / "model.joblib"
    write_city_residual_fitted_model(
        path=model_path,
        fitted=_fitted_constant_residual(),
        input_paths={"source": source},
        reproduction={"rows": 2, "max_abs_log_ratio_delta": 0.0, "tolerance": 1e-8},
    )
    fitted, manifest = load_city_residual_fitted_model(model_path)
    assert isinstance(fitted, CityResidualFittedModel)
    assert manifest["reproduction"]["rows"] == 2

    manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
    original_manifest = manifest_path.read_text()
    manifest_path.write_text(original_manifest.replace(manifest["python_version"], "0.0"))
    with pytest.raises(ValueError, match="environment binding changed"):
        load_city_residual_fitted_model(model_path)
    manifest_path.write_text(original_manifest)

    source.write_bytes(b"changed-input")
    with pytest.raises(ValueError, match="input binding changed"):
        load_city_residual_fitted_model(model_path)
