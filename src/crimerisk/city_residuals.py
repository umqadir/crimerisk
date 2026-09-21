from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from crimerisk.city_share_benchmark import build_city_share_truth_model_frame
from crimerisk.crime import OFFENSES_7
from crimerisk.model_surface import (
    _build_feature_selection,
    annual_feature_key,
    build_bg_feature_frame,
    burglary_exposure_duplicate_columns,
    merge_extra_bg_features,
)
from crimerisk.paths import RepoPaths


@dataclass(frozen=True)
class CityResidualConfig:
    hist_learning_rate: float = 0.03
    hist_max_depth: int = 5
    hist_max_iter: int = 500
    hist_min_samples_leaf: int = 20
    hist_l2_regularization: float = 1.0
    log_ratio_clip: float = 8.0
    extra_feature_paths: tuple[Path, ...] = ()
    feature_policy_path: Path | None = None
    exclude_feature_policy_classes: tuple[str, ...] = ()
    exclude_feature_policy_classes_by_offense: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class CityResidualFittedModel:
    model: HistGradientBoostingRegressor | None
    burglary_model: HistGradientBoostingRegressor | None
    feature_cols: tuple[str, ...]
    offense_cols: tuple[str, ...]
    config: CityResidualConfig
    training_row_count: int
    training_city_count: int
    training_incident_total: float
    candidate_feature_cols: tuple[str, ...] = ()
    burglary_feature_cols: tuple[str, ...] = ()
    burglary_excluded_feature_cols: tuple[str, ...] = ()
    burglary_feature_neutral_values: dict[str, float] | None = None
    feature_policy_application: dict[str, object] | None = None


CITY_RESIDUAL_FEATURE_POLICY_ATTR = "city_residual_feature_policy_application"
CITY_RESIDUAL_PREDICTION_KEY_COLUMNS = [
    "bg_id",
    "state_fips",
    "jurisdiction_id",
    "offense",
]
CITY_RESIDUAL_PREDICTION_COLUMNS = [
    *CITY_RESIDUAL_PREDICTION_KEY_COLUMNS,
    "model_share_at_fit",
    "predicted_log_ratio",
]

RESIDUAL_MODEL_CODE_IDENTITY_NODES = {
    "city_residuals": (
        "CityResidualConfig",
        "CityResidualFittedModel",
        "_normalize_policy_classes",
        "_normalize_policy_classes_by_offense",
        "_application_selected_cols_for_offense",
        "apply_city_residual_feature_policy",
        "_load_city_residual_bg_features",
        "design_city_residual_matrix",
        "attach_city_residual_features",
        "prepare_city_residual_frame",
        "fit_city_residual_model",
        "fit_city_residual_model_from_truth",
        "apply_city_residual_model",
    ),
    "allocation": (
        "DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES",
        "DEFAULT_RESIDUAL_EXCLUDE_FEATURE_POLICY_CLASSES_BY_OFFENSE",
        "PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES",
        "promoted_residual_training_city_shares_path",
        "promoted_residual_extra_bg_feature_paths",
        "_load_bg_crosswalk",
        "_load_city_incident_share_surface",
        "_load_overlap_footprint_overrides",
        "_load_overlap_custom_footprints",
        "_build_exclusive_footprint_displacement",
        "_apply_exclusive_footprint_displacement",
        "_regular_residual_prior_for_burglary_only_variant",
    ),
}


def city_residual_model_code_identity(source_paths: dict[str, Path]) -> dict[str, object]:
    """Hash only definitions that determine residual fitting or replay."""
    records: dict[str, dict[str, str]] = {}
    aggregate = hashlib.sha256()
    for role, names in RESIDUAL_MODEL_CODE_IDENTITY_NODES.items():
        source = Path(source_paths[role])
        tree = ast.parse(source.read_text(), filename=str(source))
        nodes: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nodes[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        nodes[target.id] = node
        missing = sorted(set(names) - set(nodes))
        if missing:
            raise ValueError(f"Residual code identity nodes missing from {role}: {missing}")
        role_records: dict[str, str] = {}
        for name in names:
            canonical = ast.dump(nodes[name], annotate_fields=True, include_attributes=False)
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            role_records[name] = digest
            aggregate.update(role.encode() + b"\0" + name.encode() + b"\0" + digest.encode() + b"\n")
        records[role] = role_records
    return {
        "schema_version": "city_residual_model_code_identity_v1",
        "aggregate_sha256": aggregate.hexdigest(),
        "sources": {role: str(Path(path).resolve()) for role, path in source_paths.items()},
        "nodes": records,
    }


def city_residual_fitted_model_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "modeling" / f"city_residual_fitted_model_{int(year)}.joblib"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_city_residual_model_reproduction(
    *,
    paths: RepoPaths,
    year: int,
    fitted: CityResidualFittedModel,
    predictions: pd.DataFrame,
    tolerance: float = 1e-8,
    chunk_size: int = 50_000,
) -> dict[str, object]:
    """Prove that a deterministic refit reproduces the promoted scoring surface."""
    if bool(predictions[CITY_RESIDUAL_PREDICTION_KEY_COLUMNS].isna().any().any()):
        raise ValueError("Residual reconstruction predictions contain null keys")
    if bool(predictions.duplicated(CITY_RESIDUAL_PREDICTION_KEY_COLUMNS).any()):
        raise ValueError("Residual reconstruction predictions contain duplicate keys")
    required_features = tuple(sorted(set(fitted.feature_cols) | set(fitted.burglary_feature_cols)))
    bg, _ = _load_city_residual_bg_features(
        paths=paths,
        year=int(year),
        extra_feature_paths=list(fitted.config.extra_feature_paths),
        feature_policy_path=fitted.config.feature_policy_path,
        exclude_feature_policy_classes=tuple(fitted.config.exclude_feature_policy_classes),
        exclude_feature_policy_classes_by_offense=fitted.config.exclude_feature_policy_classes_by_offense,
    )
    if bool(bg.duplicated(["bg_id", "state_fips"]).any()):
        raise ValueError("Residual reconstruction feature frame has duplicate block-group keys")
    bg = bg[["bg_id", "state_fips", *required_features]]
    max_delta = 0.0
    compared = 0
    for start in range(0, len(predictions), max(1, int(chunk_size))):
        chunk = predictions.iloc[start : start + max(1, int(chunk_size))].copy()
        expected = pd.to_numeric(chunk.pop("predicted_log_ratio"), errors="raise").to_numpy(dtype=float)
        chunk["model_share"] = pd.to_numeric(chunk.pop("model_share_at_fit"), errors="raise")
        chunk = chunk.merge(bg, on=["bg_id", "state_fips"], how="left", validate="many_to_one")
        scored = apply_city_residual_model(chunk, fitted=fitted)
        actual = pd.to_numeric(scored["predicted_log_ratio"], errors="raise").to_numpy(dtype=float)
        if len(actual) != len(expected):
            raise ValueError("Residual reconstruction changed promoted prediction row coverage")
        if len(actual):
            max_delta = max(max_delta, float(np.max(np.abs(actual - expected))))
        compared += len(actual)
    if compared != len(predictions):
        raise ValueError("Residual reconstruction did not score every promoted prediction row")
    if max_delta > float(tolerance):
        raise ValueError(
            "Deterministic residual reconstruction does not reproduce the promoted surface; "
            f"max_abs_log_ratio_delta={max_delta:.12g} tolerance={float(tolerance):.12g}"
        )
    return {
        "rows": int(compared),
        "max_abs_log_ratio_delta": float(max_delta),
        "tolerance": float(tolerance),
    }


def write_city_residual_fitted_model(
    *,
    path: Path,
    fitted: CityResidualFittedModel,
    input_paths: dict[str, Path],
    reproduction: dict[str, object],
    code_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Serialize a reproduced estimator with replay-critical lineage."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, output, compress=3)
    def _json_value(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [_json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _json_value(item) for key, item in value.items()}
        return value

    manifest = {
        "artifact": str(output.resolve()),
        "artifact_sha256": _sha256(output),
        "python_version": platform.python_version(),
        "numpy_version": importlib.metadata.version("numpy"),
        "pandas_version": importlib.metadata.version("pandas"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "feature_cols": list(fitted.feature_cols),
        "burglary_feature_cols": list(fitted.burglary_feature_cols),
        "offense_cols": list(fitted.offense_cols),
        "config": {
            field: _json_value(value)
            for field, value in fitted.config.__dict__.items()
        },
        "training_row_count": int(fitted.training_row_count),
        "training_city_count": int(fitted.training_city_count),
        "training_incident_total": float(fitted.training_incident_total),
        "reproduction": dict(reproduction),
        "inputs": {
            role: {"path": str(Path(source).resolve()), "sha256": _sha256(Path(source))}
            for role, source in sorted(input_paths.items())
        },
    }
    if code_identity is not None:
        manifest["code_identity"] = code_identity
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_city_residual_fitted_model(path: Path) -> tuple[CityResidualFittedModel, dict[str, object]]:
    """Load a reproduced model only while its artifact and input bindings remain exact."""
    artifact = Path(path)
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if not artifact.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Reconstructed residual model or manifest is missing: {artifact}")
    manifest = json.loads(manifest_path.read_text())
    if str(manifest.get("artifact_sha256", "")) != _sha256(artifact):
        raise ValueError("Reconstructed residual model SHA does not match its manifest")
    expected_environment = {
        "python_version": platform.python_version(),
        "numpy_version": importlib.metadata.version("numpy"),
        "pandas_version": importlib.metadata.version("pandas"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
    }
    for field, actual in expected_environment.items():
        if str(manifest.get(field, "")) != actual:
            raise ValueError(f"Reconstructed residual model environment binding changed: {field}")
    reproduction = manifest.get("reproduction")
    if not isinstance(reproduction, dict) or int(reproduction.get("rows", 0)) <= 0:
        raise ValueError("Reconstructed residual model lacks a full-surface reproduction record")
    if float(reproduction.get("max_abs_log_ratio_delta", np.inf)) > float(
        reproduction.get("tolerance", 0.0)
    ):
        raise ValueError("Reconstructed residual model failed its recorded reproduction tolerance")
    for role, record in dict(manifest.get("inputs", {})).items():
        source = Path(str(record.get("path", "")))
        if not source.exists() or str(record.get("sha256", "")) != _sha256(source):
            raise ValueError(f"Reconstructed residual model input binding changed: {role}")
    recorded_code_identity = manifest.get("code_identity")
    if recorded_code_identity is not None:
        sources = {
            role: Path(path)
            for role, path in dict(recorded_code_identity.get("sources", {})).items()
        }
        current_code_identity = city_residual_model_code_identity(sources)
        if current_code_identity["aggregate_sha256"] != recorded_code_identity.get(
            "aggregate_sha256"
        ):
            raise ValueError("Reconstructed residual model code identity changed")
    fitted = joblib.load(artifact)
    if not isinstance(fitted, CityResidualFittedModel):
        raise TypeError("Reconstructed residual artifact has the wrong object type")
    return fitted, manifest


def city_residual_prediction_surface_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "modeling" / f"city_residual_prediction_surface_{int(year)}.parquet"


def city_residual_prediction_manifest_path(path: Path) -> Path:
    prediction_path = Path(path)
    return prediction_path.with_suffix(prediction_path.suffix + ".manifest.json")


def load_city_residual_prediction_manifest(path: Path) -> dict[str, object]:
    manifest_path = city_residual_prediction_manifest_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"City residual prediction manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = manifest.get("feature_policy_application")
    if not isinstance(policy, dict) or not policy:
        raise ValueError(
            "City residual prediction manifest does not preserve the fitted feature-policy application"
        )
    return manifest


def load_city_residual_prediction_surface(path: Path) -> pd.DataFrame:
    """Load a promoted component-level residual prediction surface.

    Output assembly consumes this frozen model product instead of refitting the shared
    nationwide residual model.  The stored model share binds the predictions to the model
    prior and jurisdiction geometry on which they were validated.
    """
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"City residual prediction surface not found: {prediction_path}")
    out = pd.read_parquet(prediction_path, columns=CITY_RESIDUAL_PREDICTION_COLUMNS)
    missing = sorted(set(CITY_RESIDUAL_PREDICTION_COLUMNS) - set(out.columns))
    if missing:
        raise ValueError(f"City residual prediction surface is missing columns: {missing}")
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["jurisdiction_id"] = out["jurisdiction_id"].astype("string")
    out["offense"] = out["offense"].astype("string")
    if bool(out[CITY_RESIDUAL_PREDICTION_KEY_COLUMNS].isna().any().any()):
        raise ValueError("City residual prediction surface has null key fields")
    if bool(out.duplicated(CITY_RESIDUAL_PREDICTION_KEY_COLUMNS).any()):
        raise ValueError("City residual prediction surface has duplicate component keys")
    numeric = out[["model_share_at_fit", "predicted_log_ratio"]].apply(pd.to_numeric, errors="coerce")
    if bool(~np.isfinite(numeric.to_numpy(dtype=float)).all()):
        raise ValueError("City residual prediction surface has non-finite predictions")
    if bool(numeric["model_share_at_fit"].lt(0.0).any()):
        raise ValueError("City residual prediction surface has a negative model share")
    out[["model_share_at_fit", "predicted_log_ratio"]] = numeric
    return out


def apply_city_residual_prediction_surface(
    frame: pd.DataFrame,
    *,
    predictions: pd.DataFrame,
    group_cols: list[str] | tuple[str, ...] = ("jurisdiction_id", "state_fips", "offense"),
    model_share_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Apply promoted residual predictions, failing on any input-contract drift."""
    out = frame.copy()
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["jurisdiction_id"] = out["jurisdiction_id"].astype("string")
    out["offense"] = out["offense"].astype("string")
    if bool(out.duplicated(CITY_RESIDUAL_PREDICTION_KEY_COLUMNS).any()):
        raise ValueError("Residual application frame has duplicate component keys")

    out["_prediction_row_id"] = np.arange(len(out), dtype=np.int64)
    out = out.merge(
        predictions[CITY_RESIDUAL_PREDICTION_COLUMNS],
        on=CITY_RESIDUAL_PREDICTION_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    ).sort_values("_prediction_row_id", kind="mergesort")
    # Match the fitted-model path: missing model shares are zero-mass support rows.
    current_share = pd.to_numeric(out["model_share"], errors="coerce").fillna(0.0)
    out["model_share"] = current_share
    missing_prediction = out["predicted_log_ratio"].isna() | out["model_share_at_fit"].isna()
    neutral_zero_share = missing_prediction & current_share.fillna(np.inf).eq(0.0)
    out.loc[neutral_zero_share, "model_share_at_fit"] = 0.0
    out.loc[neutral_zero_share, "predicted_log_ratio"] = 0.0
    missing_prediction = out["predicted_log_ratio"].isna() | out["model_share_at_fit"].isna()
    if bool(missing_prediction.any()):
        missing = out.loc[
            missing_prediction,
            [*CITY_RESIDUAL_PREDICTION_KEY_COLUMNS, "model_share"],
        ].head(100).to_dict("records")
        raise ValueError(f"Promoted residual prediction surface does not cover the allocation frame: {missing}")

    fitted_share = pd.to_numeric(out["model_share_at_fit"], errors="coerce")
    share_delta = (current_share - fitted_share).abs()
    if bool(current_share.isna().any()) or bool(share_delta.gt(float(model_share_tolerance)).any()):
        raise ValueError(
            "Promoted residual predictions do not match the current model-share input; "
            f"max_abs_delta={float(share_delta.max()):.12g}"
        )

    out["predicted_log_ratio"] = pd.to_numeric(out["predicted_log_ratio"], errors="raise")
    out["residual_model_unscaled_share"] = current_share * np.exp(out["predicted_log_ratio"])
    out["residual_model_total"] = out.groupby(list(group_cols), dropna=False)[
        "residual_model_unscaled_share"
    ].transform("sum")
    out["residual_model_share"] = np.where(
        out["residual_model_total"].gt(0.0),
        out["residual_model_unscaled_share"] / out["residual_model_total"].replace(0.0, np.nan),
        current_share,
    )
    return out.drop(columns=["_prediction_row_id", "model_share_at_fit"])


def _normalize_policy_classes(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _normalize_policy_classes_by_offense(
    *,
    default_classes: tuple[str, ...] | list[str] | set[str] = (),
    by_offense: tuple[tuple[str, tuple[str, ...]], ...] | dict[str, object] | None = None,
) -> dict[str, tuple[str, ...]]:
    default = _normalize_policy_classes(tuple(default_classes))
    out = {str(offense): default for offense in OFFENSES_7}
    if by_offense:
        items = by_offense.items() if isinstance(by_offense, dict) else by_offense
        for offense, classes in items:
            if isinstance(classes, str):
                raw_classes = (classes,)
            elif classes is None:
                raw_classes = ()
            else:
                raw_classes = tuple(classes)
            out[str(offense)] = _normalize_policy_classes(raw_classes)
    return out


def _application_selected_cols_for_offense(
    application: dict[str, object] | None,
    *,
    offense: str,
    fallback_cols: list[str] | tuple[str, ...],
) -> list[str]:
    if isinstance(application, dict):
        by_offense = application.get("selected_feature_cols_by_offense")
        if isinstance(by_offense, dict) and str(offense) in by_offense:
            return [str(col) for col in by_offense[str(offense)]]
    return [str(col) for col in fallback_cols]


def apply_city_residual_feature_policy(
    feature_cols: list[str] | tuple[str, ...],
    *,
    feature_policy_path: Path | None,
    exclude_feature_policy_classes: tuple[str, ...] | list[str] | set[str] = (),
    exclude_feature_policy_classes_by_offense: tuple[tuple[str, tuple[str, ...]], ...] | dict[str, object] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """Apply the step-13 feature-transfer policy to residual allocator features.

    The residual allocator is fail-closed: once a policy file is configured, every
    candidate feature column must be represented in the policy artifact.
    """

    candidate_cols = [str(col) for col in feature_cols]
    excluded_classes = _normalize_policy_classes(tuple(exclude_feature_policy_classes))
    excluded_classes_by_offense = _normalize_policy_classes_by_offense(
        default_classes=excluded_classes,
        by_offense=exclude_feature_policy_classes_by_offense,
    )
    base_application: dict[str, object] = {
        "path": str(feature_policy_path) if feature_policy_path is not None else None,
        "exclude_final_classes": list(excluded_classes),
        "exclude_final_classes_by_offense": {
            offense: list(classes) for offense, classes in sorted(excluded_classes_by_offense.items())
        },
        "policy_enforced_offenses": [
            offense for offense, classes in sorted(excluded_classes_by_offense.items()) if classes
        ],
        "policy_carveout_offenses": [
            offense for offense, classes in sorted(excluded_classes_by_offense.items()) if not classes
        ],
        "candidate_feature_count": int(len(candidate_cols)),
        "candidate_feature_cols": candidate_cols,
        "selected_feature_count": int(len(candidate_cols)),
        "excluded_feature_count": 0,
        "unmapped_feature_count": 0,
        "annual_policy_alias_count": 0,
        "annual_policy_aliases": {},
        "selected_feature_cols": candidate_cols,
        "excluded_feature_cols": [],
        "selected_feature_cols_by_offense": {
            offense: candidate_cols for offense in sorted(excluded_classes_by_offense)
        },
        "excluded_feature_cols_by_offense": {
            offense: [] for offense in sorted(excluded_classes_by_offense)
        },
        "excluded_feature_cols_by_class": {},
        "excluded_feature_cols_by_group": [],
        "excluded_feature_cols_by_group_by_offense": {
            offense: [] for offense in sorted(excluded_classes_by_offense)
        },
        "candidate_feature_class_counts": {},
        "selected_feature_class_counts": {},
        "selected_feature_class_counts_by_offense": {},
        "retained_proxy_review_feature_count": 0,
        "retained_proxy_review_feature_cols": [],
        "retained_proxy_review_feature_cols_by_offense": {
            offense: [] for offense in sorted(excluded_classes_by_offense)
        },
    }
    if feature_policy_path is None:
        return candidate_cols, base_application

    path = Path(feature_policy_path)
    if not path.exists():
        raise FileNotFoundError(f"City residual feature transfer policy parquet not found: {path}")

    policy = pd.read_parquet(path)
    required = {"feature_column", "final_class"}
    missing_required = sorted(required - set(policy.columns))
    if missing_required:
        raise ValueError(f"City residual feature policy {path} is missing required columns: {missing_required}")
    if "feature_group" not in policy.columns:
        policy["feature_group"] = "unknown"

    policy = policy[["feature_column", "final_class", "feature_group"]].copy()
    policy["feature_column"] = policy["feature_column"].astype("string")
    policy = policy[policy["feature_column"].notna()].copy()
    policy["feature_column"] = policy["feature_column"].astype(str)
    policy["final_class"] = policy["final_class"].astype("string").fillna("").astype(str)
    policy["feature_group"] = policy["feature_group"].astype("string").fillna("unknown").astype(str)

    class_conflicts = (
        policy.groupby("feature_column", dropna=False)["final_class"]
        .nunique(dropna=False)
        .reset_index(name="class_count")
    )
    conflicting_cols = sorted(class_conflicts.loc[class_conflicts["class_count"].gt(1), "feature_column"].tolist())
    if conflicting_cols:
        raise ValueError(
            "City residual feature policy has conflicting final_class rows for feature columns: "
            f"{conflicting_cols}"
        )
    policy = policy.drop_duplicates("feature_column", keep="first")
    policy_by_feature = policy.set_index("feature_column")

    policy["annual_feature_key"] = policy["feature_column"].map(annual_feature_key)
    annual_conflicts = (
        policy.groupby("annual_feature_key", dropna=False)
        .agg(
            final_class_count=("final_class", "nunique"),
            feature_group_count=("feature_group", "nunique"),
        )
        .reset_index()
    )
    conflicting_annual_keys = sorted(
        annual_conflicts.loc[
            annual_conflicts["final_class_count"].gt(1)
            | annual_conflicts["feature_group_count"].gt(1),
            "annual_feature_key",
        ].tolist()
    )
    if conflicting_annual_keys:
        raise ValueError(
            "City residual feature policy has conflicting annual aliases: "
            f"{conflicting_annual_keys}"
        )
    policy_by_annual_key = policy.drop_duplicates("annual_feature_key", keep="first").set_index(
        "annual_feature_key"
    )

    policy_source_feature: dict[str, str] = {}
    for col in candidate_cols:
        if col in policy_by_feature.index:
            policy_source_feature[col] = col
            continue
        key = annual_feature_key(col)
        if key in policy_by_annual_key.index:
            policy_source_feature[col] = str(policy_by_annual_key.at[key, "feature_column"])

    unmapped = sorted(col for col in candidate_cols if col not in policy_source_feature)
    if unmapped:
        raise ValueError(
            "City residual feature policy is missing selected feature columns; fail-closed unmapped columns: "
            f"{unmapped}"
        )

    final_class = {
        col: str(policy_by_feature.at[policy_source_feature[col], "final_class"])
        for col in candidate_cols
    }
    feature_group = {
        col: str(policy_by_feature.at[policy_source_feature[col], "feature_group"])
        for col in candidate_cols
    }
    selected_by_offense: dict[str, list[str]] = {}
    excluded_by_offense: dict[str, list[str]] = {}
    excluded_union: set[str] = set()
    selected_union: set[str] = set()
    selected_class_counts_by_offense: dict[str, dict[str, int]] = {}
    retained_proxy_by_offense: dict[str, list[str]] = {}
    excluded_group_by_offense: dict[str, list[dict[str, object]]] = {}
    for offense, offense_excluded_classes in sorted(excluded_classes_by_offense.items()):
        offense_excluded_set = {
            col
            for col in candidate_cols
            if final_class.get(col) in set(offense_excluded_classes)
        }
        offense_selected = [col for col in candidate_cols if col not in offense_excluded_set]
        offense_excluded = sorted(offense_excluded_set)
        selected_by_offense[offense] = offense_selected
        excluded_by_offense[offense] = offense_excluded
        excluded_union.update(offense_excluded)
        selected_union.update(offense_selected)
        selected_class_counts_by_offense[offense] = dict(
            sorted(Counter(final_class[col] for col in offense_selected).items())
        )
        retained_proxy_by_offense[offense] = sorted(
            col for col in offense_selected if final_class.get(col) == "proxy_review"
        )
        by_group_map: dict[tuple[str, str], list[str]] = {}
        for col in offense_excluded:
            key = (final_class[col], feature_group[col])
            by_group_map.setdefault(key, []).append(col)
        excluded_group_by_offense[offense] = [
            {
                "final_class": class_name,
                "feature_group": group_name,
                "feature_count": int(len(cols)),
                "feature_cols": sorted(cols),
            }
            for (class_name, group_name), cols in sorted(by_group_map.items())
        ]

    selected_cols = [col for col in candidate_cols if col in selected_union]
    excluded_cols = sorted(excluded_union)
    excluded_by_class: dict[str, list[str]] = {}
    for col in excluded_cols:
        excluded_by_class.setdefault(final_class[col], []).append(col)
    excluded_by_group_map: dict[tuple[str, str], list[str]] = {}
    for col in excluded_cols:
        key = (final_class[col], feature_group[col])
        excluded_by_group_map.setdefault(key, []).append(col)
    proxy_review_cols = sorted(col for col in selected_cols if final_class.get(col) == "proxy_review")

    application = {
        **base_application,
        "selected_feature_count": int(len(selected_cols)),
        "excluded_feature_count": int(len(excluded_cols)),
        "unmapped_feature_count": 0,
        "annual_policy_alias_count": int(
            sum(col != source for col, source in policy_source_feature.items())
        ),
        "annual_policy_aliases": {
            col: source
            for col, source in sorted(policy_source_feature.items())
            if col != source
        },
        "selected_feature_cols": selected_cols,
        "excluded_feature_cols": excluded_cols,
        "selected_feature_cols_by_offense": selected_by_offense,
        "excluded_feature_cols_by_offense": excluded_by_offense,
        "excluded_feature_cols_by_class": {
            key: sorted(values) for key, values in sorted(excluded_by_class.items())
        },
        "excluded_feature_cols_by_group": [
            {
                "final_class": final_class,
                "feature_group": feature_group,
                "feature_count": int(len(cols)),
                "feature_cols": sorted(cols),
            }
            for (final_class, feature_group), cols in sorted(excluded_by_group_map.items())
        ],
        "excluded_feature_cols_by_group_by_offense": excluded_group_by_offense,
        "candidate_feature_class_counts": dict(sorted(Counter(final_class.values()).items())),
        "selected_feature_class_counts": dict(
            sorted(Counter(final_class[col] for col in selected_cols).items())
        ),
        "selected_feature_class_counts_by_offense": selected_class_counts_by_offense,
        "retained_proxy_review_feature_count": int(len(proxy_review_cols)),
        "retained_proxy_review_feature_cols": proxy_review_cols,
        "retained_proxy_review_feature_cols_by_offense": retained_proxy_by_offense,
    }
    return selected_cols, application


def _load_city_residual_bg_features(
    *,
    paths: RepoPaths,
    year: int,
    extra_feature_paths: list[Path] | None = None,
    feature_policy_path: Path | None = None,
    exclude_feature_policy_classes: tuple[str, ...] = (),
    exclude_feature_policy_classes_by_offense: tuple[tuple[str, tuple[str, ...]], ...] | dict[str, object] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    bg = build_bg_feature_frame(paths=paths, year=int(year))
    bg = merge_extra_bg_features(bg, extra_feature_paths=extra_feature_paths)
    feature_cols = _build_feature_selection(bg)
    feature_cols, policy_application = apply_city_residual_feature_policy(
        feature_cols,
        feature_policy_path=feature_policy_path,
        exclude_feature_policy_classes=exclude_feature_policy_classes,
        exclude_feature_policy_classes_by_offense=exclude_feature_policy_classes_by_offense,
    )
    bg = bg[["bg_id", "state_fips", *feature_cols]].copy()
    bg["bg_id"] = bg["bg_id"].astype("string").str.zfill(12)
    bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    bg.attrs[CITY_RESIDUAL_FEATURE_POLICY_ATTR] = policy_application
    return bg, feature_cols


def design_city_residual_matrix(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | tuple[str, ...],
    offense_cols: list[str] | tuple[str, ...] | None = None,
    burglary_excluded_feature_cols: list[str] | tuple[str, ...] = (),
    burglary_feature_neutral_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    base = df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).copy()
    excluded = [col for col in burglary_excluded_feature_cols if col in base.columns]
    if excluded:
        burglary = df["offense"].astype(str).eq("burglary") if "offense" in df.columns else pd.Series(False, index=df.index)
        neutral_values = burglary_feature_neutral_values or {}
        for col in excluded:
            neutral = neutral_values.get(col)
            if neutral is None or not np.isfinite(float(neutral)):
                series = pd.to_numeric(base[col], errors="coerce")
                neutral = float(series.median()) if bool(series.notna().any()) else 0.0
            base.loc[burglary, col] = float(neutral)
    base["model_share"] = pd.to_numeric(df["model_share"], errors="coerce").fillna(0.0)
    base["log_model_share"] = np.log(base["model_share"].clip(lower=1e-12))
    offense = pd.get_dummies(df["offense"].astype(str), prefix="offense", dtype=float)
    if offense_cols is None:
        offense_cols = offense.columns.tolist()
    for col in offense_cols:
        if col not in offense.columns:
            offense[col] = 0.0
    return pd.concat([base.reset_index(drop=True), offense[list(offense_cols)].reset_index(drop=True)], axis=1)


def attach_city_residual_features(
    frame: pd.DataFrame,
    *,
    paths: RepoPaths,
    year: int,
    feature_cols: list[str] | tuple[str, ...] | None = None,
    extra_feature_paths: list[Path] | None = None,
    feature_policy_path: Path | None = None,
    exclude_feature_policy_classes: tuple[str, ...] = (),
    exclude_feature_policy_classes_by_offense: tuple[tuple[str, tuple[str, ...]], ...] | dict[str, object] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if feature_cols is None:
        bg, feature_cols = _load_city_residual_bg_features(
            paths=paths,
            year=int(year),
            extra_feature_paths=extra_feature_paths,
            feature_policy_path=feature_policy_path,
            exclude_feature_policy_classes=exclude_feature_policy_classes,
            exclude_feature_policy_classes_by_offense=exclude_feature_policy_classes_by_offense,
        )
    else:
        bg, _ = _load_city_residual_bg_features(
            paths=paths,
            year=int(year),
            extra_feature_paths=extra_feature_paths,
            feature_policy_path=feature_policy_path,
            exclude_feature_policy_classes=exclude_feature_policy_classes,
            exclude_feature_policy_classes_by_offense=exclude_feature_policy_classes_by_offense,
        )
        feature_cols, explicit_policy_application = apply_city_residual_feature_policy(
            list(feature_cols),
            feature_policy_path=feature_policy_path,
            exclude_feature_policy_classes=exclude_feature_policy_classes,
            exclude_feature_policy_classes_by_offense=exclude_feature_policy_classes_by_offense,
        )
        bg = bg[["bg_id", "state_fips", *feature_cols]].copy()
        bg.attrs[CITY_RESIDUAL_FEATURE_POLICY_ATTR] = explicit_policy_application
    out = frame.copy()
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out = out.merge(bg, on=["bg_id", "state_fips"], how="left")
    if CITY_RESIDUAL_FEATURE_POLICY_ATTR in bg.attrs:
        out.attrs[CITY_RESIDUAL_FEATURE_POLICY_ATTR] = bg.attrs[CITY_RESIDUAL_FEATURE_POLICY_ATTR]
    return out, list(feature_cols)


def prepare_city_residual_frame(
    *,
    paths: RepoPaths,
    city_shares: pd.DataFrame,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    year: int,
    extra_feature_paths: list[Path] | None = None,
    feature_policy_path: Path | None = None,
    exclude_feature_policy_classes: tuple[str, ...] = (),
    exclude_feature_policy_classes_by_offense: tuple[tuple[str, tuple[str, ...]], ...] | dict[str, object] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    merged = build_city_share_truth_model_frame(
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=int(year),
    )
    if merged.empty:
        return merged, []

    frame, feature_cols = attach_city_residual_features(
        merged,
        paths=paths,
        year=int(year),
        extra_feature_paths=extra_feature_paths,
        feature_policy_path=feature_policy_path,
        exclude_feature_policy_classes=exclude_feature_policy_classes,
        exclude_feature_policy_classes_by_offense=exclude_feature_policy_classes_by_offense,
    )
    frame["group_bg_count"] = frame.groupby(
        ["city_name", "jurisdiction_id", "state_fips", "offense"],
        dropna=False,
    )["bg_id"].transform("size")
    frame["row_weight"] = (
        pd.to_numeric(frame["incident_total"], errors="coerce").fillna(0.0)
        / pd.to_numeric(frame["group_bg_count"], errors="coerce").replace(0, np.nan)
    ).fillna(0.0)
    eps = 1e-12
    frame["target_log_ratio"] = np.log(
        (
            pd.to_numeric(frame["true_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
            + eps
        )
        / (
            pd.to_numeric(frame["model_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
            + eps
        )
    ).clip(-12.0, 12.0)
    return frame, feature_cols


def fit_city_residual_model(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str] | tuple[str, ...],
    config: CityResidualConfig = CityResidualConfig(),
    burglary_frame: pd.DataFrame | None = None,
) -> CityResidualFittedModel | None:
    if frame.empty or not feature_cols:
        return None
    policy_application = frame.attrs.get(CITY_RESIDUAL_FEATURE_POLICY_ATTR)
    if (
        not isinstance(policy_application, dict)
        or list(policy_application.get("selected_feature_cols", [])) != [str(col) for col in feature_cols]
    ):
        feature_cols, policy_application = apply_city_residual_feature_policy(
            list(feature_cols),
            feature_policy_path=config.feature_policy_path,
            exclude_feature_policy_classes=config.exclude_feature_policy_classes,
            exclude_feature_policy_classes_by_offense=config.exclude_feature_policy_classes_by_offense,
        )
    candidate_feature_cols = tuple(str(col) for col in policy_application.get("candidate_feature_cols", feature_cols))
    general_offenses = [offense for offense in OFFENSES_7 if offense != "burglary"]
    general_selected_sets = [
        set(_application_selected_cols_for_offense(policy_application, offense=offense, fallback_cols=feature_cols))
        for offense in general_offenses
    ]
    general_selected = set.intersection(*general_selected_sets) if general_selected_sets else set(feature_cols)
    general_feature_cols = [col for col in candidate_feature_cols if col in general_selected]
    burglary_feature_cols = _application_selected_cols_for_offense(
        policy_application,
        offense="burglary",
        fallback_cols=candidate_feature_cols,
    )
    duplicate_cols = burglary_exposure_duplicate_columns(config.feature_policy_path)
    burglary_excluded_feature_cols = tuple(col for col in burglary_feature_cols if str(col) in duplicate_cols)
    burglary_feature_neutral_values: dict[str, float] = {}
    offense_cols = tuple(sorted(f"offense_{offense}" for offense in frame["offense"].astype(str).sort_values().unique().tolist()))
    general_training = frame[~frame["offense"].astype(str).eq("burglary")].copy()
    model: HistGradientBoostingRegressor | None = None
    if not general_training.empty and general_feature_cols:
        x_train = design_city_residual_matrix(general_training, feature_cols=general_feature_cols, offense_cols=offense_cols)
        y_train = pd.to_numeric(general_training["target_log_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w_train = pd.to_numeric(general_training["row_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(config.hist_learning_rate),
            max_depth=int(config.hist_max_depth),
            max_iter=int(config.hist_max_iter),
            min_samples_leaf=int(config.hist_min_samples_leaf),
            l2_regularization=float(config.hist_l2_regularization),
            random_state=0,
        )
        model.fit(x_train, y_train, sample_weight=w_train if np.any(w_train > 0) else None)
    burglary_model = None
    burglary_training = burglary_frame if burglary_frame is not None else frame
    burglary_training = burglary_training[burglary_training["offense"].astype(str).eq("burglary")].copy()
    if not burglary_training.empty and burglary_feature_cols:
        for col in burglary_excluded_feature_cols:
            series = (
                pd.to_numeric(burglary_training[col], errors="coerce")
                if col in burglary_training.columns
                else pd.Series(dtype=float)
            )
            burglary_feature_neutral_values[col] = float(series.median()) if bool(series.notna().any()) else 0.0
        x_burglary = design_city_residual_matrix(
            burglary_training,
            feature_cols=burglary_feature_cols,
            offense_cols=offense_cols,
            burglary_excluded_feature_cols=burglary_excluded_feature_cols,
            burglary_feature_neutral_values=burglary_feature_neutral_values,
        )
        y_burglary = pd.to_numeric(
            burglary_training["target_log_ratio"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        w_burglary = pd.to_numeric(
            burglary_training["row_weight"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        burglary_model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(config.hist_learning_rate),
            max_depth=int(config.hist_max_depth),
            max_iter=int(config.hist_max_iter),
            min_samples_leaf=int(config.hist_min_samples_leaf),
            l2_regularization=float(config.hist_l2_regularization),
            random_state=0,
        )
        if not x_burglary.empty:
            burglary_model.fit(
                x_burglary,
                y_burglary,
                sample_weight=w_burglary if np.any(w_burglary > 0) else None,
            )
        else:
            burglary_model = None
    training_city_count = 0
    if "jurisdiction_id" in frame.columns:
        training_city_count = int(frame["jurisdiction_id"].astype(str).nunique())
    training_incident_total = 0.0
    if "incident_total" in frame.columns:
        incident_totals = (
            frame[["jurisdiction_id", "state_fips", "offense", "incident_total"]]
            .drop_duplicates()
            .get("incident_total", pd.Series(dtype=float))
        )
        training_incident_total = float(pd.to_numeric(incident_totals, errors="coerce").fillna(0.0).sum())
    return CityResidualFittedModel(
        model=model,
        burglary_model=burglary_model,
        feature_cols=tuple(general_feature_cols),
        offense_cols=offense_cols,
        config=config,
        training_row_count=int(len(frame)),
        training_city_count=training_city_count,
        training_incident_total=training_incident_total,
        candidate_feature_cols=tuple(candidate_feature_cols),
        burglary_feature_cols=tuple(burglary_feature_cols),
        burglary_excluded_feature_cols=burglary_excluded_feature_cols,
        burglary_feature_neutral_values=burglary_feature_neutral_values,
        feature_policy_application=policy_application,
    )


def fit_city_residual_model_from_truth(
    *,
    paths: RepoPaths,
    city_shares: pd.DataFrame,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    year: int,
    config: CityResidualConfig = CityResidualConfig(),
    burglary_bg_prior: pd.DataFrame | None = None,
) -> CityResidualFittedModel | None:
    frame, feature_cols = prepare_city_residual_frame(
        paths=paths,
        city_shares=city_shares,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        year=int(year),
        extra_feature_paths=list(config.extra_feature_paths),
        feature_policy_path=config.feature_policy_path,
        exclude_feature_policy_classes=tuple(config.exclude_feature_policy_classes),
        exclude_feature_policy_classes_by_offense=config.exclude_feature_policy_classes_by_offense,
    )
    burglary_frame = None
    if burglary_bg_prior is not None:
        burglary_frame, _ = prepare_city_residual_frame(
            paths=paths,
            city_shares=city_shares,
            bg_prior=burglary_bg_prior,
            bg_crosswalk=bg_crosswalk,
            year=int(year),
            extra_feature_paths=list(config.extra_feature_paths),
            feature_policy_path=config.feature_policy_path,
            exclude_feature_policy_classes=tuple(config.exclude_feature_policy_classes),
            exclude_feature_policy_classes_by_offense=config.exclude_feature_policy_classes_by_offense,
        )
    return fit_city_residual_model(
        frame,
        feature_cols=feature_cols,
        config=config,
        burglary_frame=burglary_frame,
    )


def apply_city_residual_model(
    frame: pd.DataFrame,
    *,
    fitted: CityResidualFittedModel | None,
    group_cols: list[str] | tuple[str, ...] = ("jurisdiction_id", "state_fips", "offense"),
) -> pd.DataFrame:
    out = frame.copy()
    out["model_share"] = pd.to_numeric(out.get("model_share"), errors="coerce").fillna(0.0)
    if fitted is None or out.empty:
        out["residual_model_share"] = out["model_share"]
        return out

    required_feature_cols = sorted(set(fitted.feature_cols) | set(fitted.burglary_feature_cols))
    missing = [col for col in required_feature_cols if col not in out.columns]
    if missing:
        raise ValueError(f"City residual application frame is missing feature columns: {sorted(missing)}")

    burglary = out["offense"].astype(str).eq("burglary") if "offense" in out.columns else pd.Series(False, index=out.index)
    pred_log_ratio = np.zeros(len(out), dtype=float)
    non_burglary = ~burglary
    if fitted.model is not None and fitted.feature_cols and bool(non_burglary.any()):
        non_burglary_mask = non_burglary.to_numpy(dtype=bool)
        x = design_city_residual_matrix(
            out,
            feature_cols=fitted.feature_cols,
            offense_cols=fitted.offense_cols,
        )
        # ``x`` has a reset (0-based) index, so index it positionally with the numpy
        # mask, matching the burglary branch below. Using the boolean Series
        # ``non_burglary`` (carrying ``out``'s original, possibly non-contiguous index
        # from a filtered frame) would be unalignable against ``x``.
        pred_log_ratio[non_burglary_mask] = np.asarray(
            fitted.model.predict(x.loc[non_burglary_mask]),
            dtype=float,
        )
    if fitted.burglary_model is not None and fitted.burglary_feature_cols and bool(burglary.any()):
        burglary_mask = burglary.to_numpy(dtype=bool)
        x_burglary = design_city_residual_matrix(
            out,
            feature_cols=fitted.burglary_feature_cols,
            offense_cols=fitted.offense_cols,
            burglary_excluded_feature_cols=fitted.burglary_excluded_feature_cols,
            burglary_feature_neutral_values=fitted.burglary_feature_neutral_values,
        )
        pred_log_ratio[burglary_mask] = np.asarray(
            fitted.burglary_model.predict(x_burglary.loc[burglary_mask]),
            dtype=float,
        )
    pred_log_ratio = np.clip(pred_log_ratio, -float(fitted.config.log_ratio_clip), float(fitted.config.log_ratio_clip))
    out["predicted_log_ratio"] = pred_log_ratio
    out["residual_model_unscaled_share"] = out["model_share"] * np.exp(pred_log_ratio)
    out["residual_model_total"] = out.groupby(list(group_cols), dropna=False)["residual_model_unscaled_share"].transform("sum")
    out["residual_model_share"] = np.where(
        pd.to_numeric(out["residual_model_total"], errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(out["residual_model_unscaled_share"], errors="coerce").fillna(0.0)
        / pd.to_numeric(out["residual_model_total"], errors="coerce").fillna(np.nan),
        out["model_share"],
    )
    return out
