from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_share_benchmark import (  # noqa: E402
    build_city_share_diagnostics,
    build_city_share_truth_model_frame,
    weighted_mean,
)
from crimerisk.crime import OFFENSES_7  # noqa: E402
from crimerisk.crosswalk_shares import normalize_block_group_allocation_shares  # noqa: E402
from crimerisk.model_surface import (  # noqa: E402
    LEAVE_LARGE_CITY_OUT_MIN_POPULATION,
    ModelSurfaceConfig,
    _build_model,
    _build_offense_training_state,
    _build_sparse_offense_pooling_target,
    _prepare_model_surface_context,
)
from crimerisk.paths import RepoPaths  # noqa: E402


DEFAULT_SPLIT_MODES = ("kfold", "leave_large_city_out", "leave_one_city_out")
DOMINANCE_RATIO = 1.5
TOTAL_DOMINATED_RELATIVE_ERROR_MIN = 0.25
ALLOCATION_DOMINATED_TVD_MIN = 0.20
LOW_ERROR_RELATIVE_ERROR_MAX = 0.20
LOW_ERROR_TVD_MAX = 0.20
OFFENSE_COUNT_FLOORS = {
    "murder": 200.0,
    "rape": 200.0,
    "robbery": 75.0,
    "aggravated_assault": 75.0,
    "burglary": 75.0,
    "larceny": 200.0,
    "motor_vehicle_theft": 75.0,
}


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _num(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    try:
        out = float(parsed)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype("boolean").fillna(False).astype(bool)


def _selected_group_splits(
    *,
    group_labels: np.ndarray,
    selected_groups: Iterable[str],
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    labels = pd.Series(group_labels).astype("string").fillna("unknown").astype(str).to_numpy()
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    for group in selected_groups:
        group_text = str(group)
        train_idx = np.flatnonzero(labels != group_text)
        test_idx = np.flatnonzero(labels == group_text)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        out.append((train_idx, test_idx, group_text))
    return out


def _city_size_class(population: object) -> str:
    pop = _num(population, default=float("nan"))
    if not np.isfinite(pop) or pop <= 0:
        return "unknown"
    if pop < 100_000:
        return "small"
    if pop < 250_000:
        return "mid_size"
    if pop < 500_000:
        return "large_sub_500k"
    if pop < 1_000_000:
        return "large_500k_1m"
    return "mega_1m_plus"


def _metro_class(row: pd.Series) -> str:
    cbsa = row.get("dominant_cbsa_code")
    share = _num(row.get("dominant_cbsa_share"), default=0.0)
    if pd.isna(cbsa) or not str(cbsa).strip() or str(cbsa).strip().lower() in {"nan", "none", "<na>", "unknown"}:
        return "nonmetro_or_unknown"
    if share >= 0.75:
        return "cbsa_dominant"
    if share > 0:
        return "cbsa_mixed"
    return "nonmetro_or_unknown"


def _jurisdiction_density_context(bg_crosswalk: pd.DataFrame) -> pd.DataFrame:
    bg = normalize_block_group_allocation_shares(bg_crosswalk.copy())
    bg["aland20"] = pd.to_numeric(bg.get("aland20"), errors="coerce").fillna(0.0).clip(lower=0.0)
    bg["allocation_share"] = pd.to_numeric(bg["allocation_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    area = (
        bg.assign(weighted_aland20=lambda df: df["aland20"] * df["allocation_share"])
        .groupby("jurisdiction_id", dropna=False)["weighted_aland20"]
        .sum()
        .rename("jurisdiction_land_area_sqkm")
        .reset_index()
    )
    area["jurisdiction_land_area_sqkm"] = pd.to_numeric(
        area["jurisdiction_land_area_sqkm"],
        errors="coerce",
    ).fillna(0.0) / 1_000_000.0
    return area


def _build_split_plan(
    *,
    split_mode: str,
    x_train: pd.DataFrame,
    state_fips_train: np.ndarray,
    large_city_group_labels: np.ndarray,
    large_city_eligible: Iterable[str],
    truth_city_eligible: Iterable[str],
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    if split_mode == "kfold":
        splitter = KFold(n_splits=min(5, len(x_train)), shuffle=True, random_state=0)
        return [
            (train_idx, test_idx, f"kfold_{fold_idx}", f"kfold_{fold_idx}")
            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x_train), start=1)
        ]
    if split_mode == "leave_state_out":
        states = pd.Series(state_fips_train).astype("string").fillna("unknown").astype(str).to_numpy()
        groups = sorted(pd.unique(states).tolist())
        return [
            (train_idx, test_idx, f"state_{group}", group)
            for train_idx, test_idx, group in _selected_group_splits(
                group_labels=states,
                selected_groups=groups,
            )
        ]
    if split_mode == "leave_large_city_out":
        return [
            (train_idx, test_idx, f"jurisdiction_{group}", group)
            for train_idx, test_idx, group in _selected_group_splits(
                group_labels=large_city_group_labels,
                selected_groups=large_city_eligible,
            )
        ]
    if split_mode in {"leave_truth_city_out", "leave_one_city_out"}:
        return [
            (train_idx, test_idx, f"city_{group}", group)
            for train_idx, test_idx, group in _selected_group_splits(
                group_labels=large_city_group_labels,
                selected_groups=truth_city_eligible,
            )
        ]
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def build_jurisdiction_cv_predictions(
    *,
    paths: RepoPaths,
    config: ModelSurfaceConfig,
    truth_city_jurisdiction_ids: Iterable[str],
    split_modes: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls, bg_crosswalk, _, training, feature_cols, train_state_cols, base_x, _, _ = _prepare_model_surface_context(
        paths=paths,
        config=config,
        extra_bg_feature_paths=None,
    )
    truth_case_types = {}
    if isinstance(truth_city_jurisdiction_ids, pd.DataFrame):
        truth_city_ids = sorted(
            {
                str(value)
                for value in truth_city_jurisdiction_ids["jurisdiction_id"].dropna().astype(str).tolist()
                if str(value)
            }
        )
        truth_case_types = (
            truth_city_jurisdiction_ids[["jurisdiction_id", "validation_case_type"]]
            .dropna(subset=["jurisdiction_id"])
            .drop_duplicates("jurisdiction_id")
            .set_index("jurisdiction_id")["validation_case_type"]
            .astype(str)
            .to_dict()
            if "validation_case_type" in truth_city_jurisdiction_ids.columns
            else {}
        )
    else:
        truth_city_ids = sorted({str(value) for value in truth_city_jurisdiction_ids if str(value)})
    density_context = _jurisdiction_density_context(bg_crosswalk)
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for offense in OFFENSES_7:
        offense_state = _build_offense_training_state(
            training=training,
            controls=controls,
            base_x=base_x,
            config=config,
            offense=str(offense),
            feature_cols=feature_cols,
            train_state_cols=train_state_cols,
        )
        frame = offense_state["frame"].copy()
        counts = offense_state["counts"]
        pop = offense_state["pop"]
        raw_rate = offense_state["raw_rate"]
        observed_only_mask = offense_state["observed_only_mask"]
        train_mask = offense_state["train_mask"]
        train_mask_np = offense_state["train_mask_np"]
        x_train = offense_state["x_train"]
        model_feature_cols = offense_state["model_feature_cols"]
        state_fips_train = offense_state["state_fips_train"]
        bucket_population_train = offense_state["bucket_population_train"]
        large_city_group_labels = offense_state["large_city_group_labels"]
        large_city_eligible = offense_state["large_city_eligible"]

        fit_rate, pooling_meta = _build_sparse_offense_pooling_target(
            frame=frame,
            counts=counts,
            pop=pop,
            offense=str(offense),
            config=config,
            observed_only_mask=observed_only_mask,
        )
        y_fit = np.log1p(fit_rate)
        y_eval = np.log1p(raw_rate)
        train_index = np.flatnonzero(train_mask_np)

        train_meta_cols = [
            "jurisdiction_id",
            "jurisdiction_name",
            "jurisdiction_type",
            "state_fips",
            "dominant_county_fips",
            "dominant_county_share",
            "dominant_cbsa_code",
            "dominant_cbsa_share",
            "bucket_population",
            "adjusted_count_ags_core",
            "preferred_source",
            "quality_tier_preferred",
            "estimate_confidence",
            "needs_current_year_fill",
            "estimated_from_panel",
        ]
        train_meta = frame.loc[train_mask, train_meta_cols].reset_index(drop=True).copy()
        train_meta = train_meta.merge(density_context, on="jurisdiction_id", how="left")
        train_meta["bucket_population"] = pd.to_numeric(train_meta["bucket_population"], errors="coerce").fillna(0.0)
        train_meta["jurisdiction_population"] = train_meta["bucket_population"]
        train_meta["jurisdiction_land_area_sqkm"] = pd.to_numeric(
            train_meta["jurisdiction_land_area_sqkm"],
            errors="coerce",
        ).fillna(0.0)
        train_meta["jurisdiction_density"] = np.where(
            train_meta["jurisdiction_land_area_sqkm"].gt(0),
            train_meta["jurisdiction_population"] / train_meta["jurisdiction_land_area_sqkm"],
            np.nan,
        )
        train_meta["production_pinned_flag"] = (
            ~_bool_series(train_meta["needs_current_year_fill"])
            & ~_bool_series(train_meta["estimated_from_panel"])
        )
        train_meta["production_estimate_source"] = train_meta["preferred_source"].astype("string")
        train_meta["is_large_city"] = (
            train_meta["jurisdiction_type"].eq("municipal")
            & train_meta["jurisdiction_population"].ge(float(LEAVE_LARGE_CITY_OUT_MIN_POPULATION))
        )
        train_meta["city_size_class"] = train_meta["jurisdiction_population"].map(_city_size_class)
        train_meta["metro_class"] = train_meta.apply(_metro_class, axis=1)
        train_meta["has_direct_incident_truth"] = train_meta["jurisdiction_id"].astype(str).isin(set(truth_city_ids))
        train_meta["validation_case_type"] = (
            train_meta["jurisdiction_id"].astype(str).map(truth_case_types).fillna("none")
        )
        train_meta["actual_count"] = pd.to_numeric(
            train_meta["adjusted_count_ags_core"],
            errors="coerce",
        ).fillna(0.0)
        train_meta["actual_rate_per_100k"] = np.where(
            train_meta["bucket_population"].gt(0),
            train_meta["actual_count"] / train_meta["bucket_population"] * 1e5,
            0.0,
        )

        for split_mode in split_modes:
            split_plan = _build_split_plan(
                split_mode=str(split_mode),
                x_train=x_train,
                state_fips_train=state_fips_train,
                large_city_group_labels=large_city_group_labels,
                large_city_eligible=large_city_eligible,
                truth_city_eligible=truth_city_ids,
            )
            if not split_plan:
                continue
            pred_log = np.full(len(x_train), np.nan, dtype=float)
            fold_parts: list[pd.DataFrame] = []
            for train_idx, test_idx, fold_id, holdout_group in split_plan:
                model = _build_model(config, feature_names=model_feature_cols)
                model.fit(x_train.iloc[train_idx].replace([np.inf, -np.inf], np.nan), y_fit[train_index[train_idx]])
                fold_pred_log = np.asarray(
                    model.predict(x_train.iloc[test_idx].replace([np.inf, -np.inf], np.nan)),
                    dtype=float,
                )
                pred_log[test_idx] = fold_pred_log
                part = train_meta.iloc[test_idx].copy()
                part["year"] = int(config.year)
                part["offense"] = str(offense)
                part["split_mode"] = str(split_mode)
                part["training_fold_type"] = str(split_mode)
                part["fold_id"] = str(fold_id)
                part["holdout_group"] = str(holdout_group)
                part["prediction_training_rows"] = int(len(train_idx))
                part["prediction_test_rows"] = int(len(test_idx))
                part["predicted_log_rate"] = fold_pred_log
                part["actual_log_rate"] = y_eval[train_index[test_idx]]
                fold_parts.append(part)
            if not fold_parts:
                continue
            split_predictions = pd.concat(fold_parts, ignore_index=True)
            split_predictions["predicted_log_rate"] = pd.to_numeric(
                split_predictions["predicted_log_rate"],
                errors="coerce",
            ).clip(lower=0.0, upper=12.0)
            split_predictions["predicted_rate_per_100k"] = np.expm1(split_predictions["predicted_log_rate"]).clip(
                lower=0.0
            )
            split_predictions["predicted_count"] = (
                split_predictions["predicted_rate_per_100k"]
                / 1e5
                * pd.to_numeric(split_predictions["bucket_population"], errors="coerce").fillna(0.0)
            )
            split_predictions["absolute_count_error"] = (
                pd.to_numeric(split_predictions["predicted_count"], errors="coerce").fillna(0.0)
                - pd.to_numeric(split_predictions["actual_count"], errors="coerce").fillna(0.0)
            ).abs()
            split_predictions["absolute_rate_error_per_100k"] = (
                pd.to_numeric(split_predictions["predicted_rate_per_100k"], errors="coerce").fillna(0.0)
                - pd.to_numeric(split_predictions["actual_rate_per_100k"], errors="coerce").fillna(0.0)
            ).abs()
            rows.append(split_predictions)

            mask = np.isfinite(y_eval[train_index]) & np.isfinite(pred_log)
            if int(mask.sum()) >= 5:
                actual_rate = np.expm1(y_eval[train_index][mask])
                predicted_rate = np.expm1(np.clip(pred_log[mask], 0.0, 12.0))
                metric_rows.append(
                    {
                        "year": int(config.year),
                        "offense": str(offense),
                        "split_mode": str(split_mode),
                        "rows": int(mask.sum()),
                        "holdout_group_count": int(len(split_plan)),
                        "r2_log_rate": float(r2_score(y_eval[train_index][mask], pred_log[mask])),
                        "r2_rate": float(r2_score(actual_rate, predicted_rate)),
                        "rmse_log_rate": float(np.sqrt(np.mean((y_eval[train_index][mask] - pred_log[mask]) ** 2))),
                        "mean_absolute_count_error": float(
                            pd.to_numeric(split_predictions["absolute_count_error"], errors="coerce")
                            .fillna(0.0)
                            .mean()
                        ),
                        "sparse_pooling_strategy": str(pooling_meta.get("strategy", "none")),
                        "sparse_pooling_applied_training_rows": int(pooling_meta.get("applied_rows", 0) or 0),
                    }
                )

    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["split_mode", "offense", "fold_id", "jurisdiction_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    if not metrics.empty:
        metrics = metrics.sort_values(["split_mode", "offense"], kind="mergesort").reset_index(drop=True)
    return predictions, metrics


def _bg_population_frame(paths: RepoPaths, *, year: int) -> pd.DataFrame:
    output = pd.read_parquet(
        paths.state_dir / "output" / f"crimerisk_block_group_{int(year)}_ags_core.parquet",
        columns=["block_group_geoid", "state_fips", "tract_id", f"population_{int(year)}"],
    )
    output = output.rename(columns={"block_group_geoid": "bg_id", f"population_{int(year)}": "population"})
    output["bg_id"] = output["bg_id"].astype("string").str.zfill(12)
    output["tract_id"] = output["tract_id"].astype("string").str.zfill(11)
    output["state_fips"] = output["state_fips"].astype("string").str.zfill(2)
    output["county_fips"] = output["bg_id"].str.slice(0, 5)
    output["population"] = pd.to_numeric(output["population"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return output


def _county_rate_lookup(paths: RepoPaths, *, year: int, bg_population: pd.DataFrame) -> pd.DataFrame:
    config = ModelSurfaceConfig(year=int(year), compute_diagnostics=False)
    controls, bg_crosswalk, bg, training, feature_cols, _, _, _, _ = _prepare_model_surface_context(
        paths=paths,
        config=config,
        extra_bg_feature_paths=None,
    )
    del bg_crosswalk, bg, feature_cols
    controls = controls[controls["jurisdiction_type"].isin(["municipal", "state_nonmunicipal_remainder"])].copy()
    controls["adjusted_count_ags_core"] = pd.to_numeric(
        controls["adjusted_count_ags_core"],
        errors="coerce",
    ).fillna(0.0)
    county_jurisdictions = training[["jurisdiction_id", "dominant_county_fips"]].copy()
    county_jurisdictions["dominant_county_fips"] = (
        county_jurisdictions["dominant_county_fips"].astype("string").str.zfill(5)
    )
    county_counts = controls.merge(county_jurisdictions, on="jurisdiction_id", how="left")
    county_counts = (
        county_counts.dropna(subset=["dominant_county_fips"])
        .groupby(["dominant_county_fips", "offense"], dropna=False)["adjusted_count_ags_core"]
        .sum()
        .rename("county_count")
        .reset_index()
        .rename(columns={"dominant_county_fips": "county_fips"})
    )
    county_pop = (
        bg_population.groupby("county_fips", dropna=False)["population"]
        .sum()
        .rename("county_population")
        .reset_index()
    )
    county = county_counts.merge(county_pop, on="county_fips", how="left")
    county["county_rate_per_person"] = np.where(
        pd.to_numeric(county["county_population"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(county["county_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(county["county_population"], errors="coerce").fillna(np.nan),
        0.0,
    )
    return county[["county_fips", "offense", "county_rate_per_person"]].copy()


def _state_rate_lookup(paths: RepoPaths, *, year: int, bg_population: pd.DataFrame) -> pd.DataFrame:
    controls = pd.read_parquet(paths.state_dir / "controls" / f"jurisdiction_controls_{int(year)}.parquet")
    controls = controls[controls["jurisdiction_type"].isin(["municipal", "state_nonmunicipal_remainder"])].copy()
    controls["state_fips"] = controls["state_fips"].astype("string").str.zfill(2)
    controls["adjusted_count_ags_core"] = pd.to_numeric(
        controls["adjusted_count_ags_core"],
        errors="coerce",
    ).fillna(0.0)
    state_counts = (
        controls.groupby(["state_fips", "offense"], dropna=False)["adjusted_count_ags_core"]
        .sum()
        .rename("state_count")
        .reset_index()
    )
    state_pop = (
        bg_population.groupby("state_fips", dropna=False)["population"].sum().rename("state_population").reset_index()
    )
    state = state_counts.merge(state_pop, on="state_fips", how="left")
    state["state_rate_per_person"] = np.where(
        pd.to_numeric(state["state_population"], errors="coerce").fillna(0.0).gt(0),
        pd.to_numeric(state["state_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(state["state_population"], errors="coerce").fillna(np.nan),
        0.0,
    )
    return state[["state_fips", "offense", "state_rate_per_person"]].copy()


def _build_long_prior_from_bg_values(
    *,
    bg_values: pd.DataFrame,
    value_col: str,
    offenses: Iterable[str],
) -> pd.DataFrame:
    base = bg_values[["bg_id", "tract_id", "state_fips", value_col]].copy()
    base["bg_weight"] = pd.to_numeric(base[value_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    base = base.drop(columns=[value_col])
    return pd.concat([base.assign(offense=str(offense)) for offense in offenses], ignore_index=True)


def build_allocation_measurement_tables(
    *,
    paths: RepoPaths,
    year: int,
    city_shares: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    bg_prior: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bg_population = _bg_population_frame(paths, year=year)
    output = pd.read_parquet(paths.state_dir / "output" / f"crimerisk_block_group_{int(year)}_ags_core.parquet")
    output = output.rename(columns={"block_group_geoid": "bg_id"}).copy()
    output["bg_id"] = output["bg_id"].astype("string").str.zfill(12)
    output["state_fips"] = output["state_fips"].astype("string").str.zfill(2)
    output["tract_id"] = output["tract_id"].astype("string").str.zfill(11)

    offenses = sorted(
        {
            str(offense)
            for offense in city_shares.loc[pd.to_numeric(city_shares["year"], errors="coerce").eq(int(year)), "offense"]
            .astype("string")
            .dropna()
            .tolist()
        }
    )
    output_long = pd.concat(
        [
            output[["bg_id", "tract_id", "state_fips", f"count_{offense}"]]
            .rename(columns={f"count_{offense}": "bg_weight"})
            .assign(offense=str(offense))
            for offense in offenses
            if f"count_{offense}" in output.columns
        ],
        ignore_index=True,
    )

    state_rates = _state_rate_lookup(paths, year=year, bg_population=bg_population)
    state_rate_values = (
        bg_population[["bg_id", "tract_id", "state_fips", "population"]]
        .merge(state_rates, on="state_fips", how="left")
        .assign(bg_weight=lambda df: pd.to_numeric(df["population"], errors="coerce").fillna(0.0) * pd.to_numeric(df["state_rate_per_person"], errors="coerce").fillna(0.0))
    )[["bg_id", "tract_id", "state_fips", "offense", "bg_weight"]]

    county_rates = _county_rate_lookup(paths, year=year, bg_population=bg_population)
    county_rate_values = (
        bg_population[["bg_id", "tract_id", "state_fips", "county_fips", "population"]]
        .merge(county_rates, on="county_fips", how="left")
        .assign(bg_weight=lambda df: pd.to_numeric(df["population"], errors="coerce").fillna(0.0) * pd.to_numeric(df["county_rate_per_person"], errors="coerce").fillna(0.0))
    )[["bg_id", "tract_id", "state_fips", "offense", "bg_weight"]]

    population_prior = _build_long_prior_from_bg_values(
        bg_values=bg_population,
        value_col="population",
        offenses=offenses,
    )

    candidate_priors = [
        ("ags_core_output", output_long),
        ("hist_gbm_bg_prior", bg_prior[["bg_id", "tract_id", "state_fips", "offense", "bg_weight"]].copy()),
        ("state_rate_population", state_rate_values),
        ("county_rate_population", county_rate_values),
        ("parent_jurisdiction_rate_population", population_prior),
    ]

    normalized_crosswalk = normalize_block_group_allocation_shares(bg_crosswalk.copy())
    county_case_mask = (
        city_shares.get("validation_case_type", pd.Series("", index=city_shares.index))
        .astype("string")
        .fillna("")
        .eq("suburban_county_validation_case")
    )
    standard_city_shares = city_shares[~county_case_mask].copy()
    county_city_shares = city_shares[county_case_mask].copy()
    county_crosswalk = pd.DataFrame(
        columns=["state_fips", "block_group_geoid", "jurisdiction_id", "jurisdiction_type", "allocation_share"]
    )
    if not county_city_shares.empty:
        county_crosswalk = (
            county_city_shares[["state_fips", "block_group_geoid", "jurisdiction_id"]]
            .drop_duplicates()
            .copy()
        )
        county_crosswalk["state_fips"] = county_crosswalk["state_fips"].astype("string").str.zfill(2)
        county_crosswalk["block_group_geoid"] = county_crosswalk["block_group_geoid"].astype("string").str.zfill(12)
        county_crosswalk["jurisdiction_type"] = "suburban_county_validation_case"
        county_crosswalk["allocation_share"] = 1.0

    bg_frames: list[pd.DataFrame] = []
    tract_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for model_name, prior in candidate_priors:
        bg_frame_parts: list[pd.DataFrame] = []
        if not standard_city_shares.empty:
            bg_frame_parts.append(
                build_city_share_truth_model_frame(
                    city_shares=standard_city_shares,
                    bg_prior=prior,
                    bg_crosswalk=normalized_crosswalk,
                    year=year,
                )
            )
        if not county_city_shares.empty and not county_crosswalk.empty:
            bg_frame_parts.append(
                build_city_share_truth_model_frame(
                    city_shares=county_city_shares,
                    bg_prior=prior,
                    bg_crosswalk=county_crosswalk,
                    year=year,
                )
            )
        bg_frame = pd.concat(bg_frame_parts, ignore_index=True) if bg_frame_parts else pd.DataFrame()
        bg_diag, bg_summary = build_city_share_diagnostics(bg_frame, predicted_share_col="model_share")
        bg_diag = bg_diag.assign(model_name=model_name, geography="block_group")
        bg_frames.append(bg_diag)
        summary_rows.append(_summarize_allocation_row(model_name, "block_group", bg_summary))

        tract_prior = (
            prior.groupby(["tract_id", "state_fips", "offense"], dropna=False)["bg_weight"]
            .sum()
            .rename("bg_weight")
            .reset_index()
            .rename(columns={"tract_id": "bg_id"})
        )
        tract_crosswalk_standard = (
            normalized_crosswalk.assign(tract_id=normalized_crosswalk["block_group_geoid"].astype("string").str.zfill(12).str.slice(0, 11))
            .groupby(["state_fips", "tract_id", "jurisdiction_id", "jurisdiction_type"], dropna=False)["allocation_share"]
            .sum()
            .rename("allocation_share")
            .reset_index()
            .rename(columns={"tract_id": "block_group_geoid"})
        )
        tract_city_shares_standard = (
            standard_city_shares.assign(
                tract_id=standard_city_shares["block_group_geoid"].astype("string").str.zfill(12).str.slice(0, 11)
            )
            .groupby(["city_name", "jurisdiction_id", "state_fips", "year", "offense", "tract_id", "geocode_quality_tier"], dropna=False)["incident_count"]
            .sum()
            .rename("incident_count")
            .reset_index()
            .rename(columns={"tract_id": "block_group_geoid"})
        )
        tract_frame_parts: list[pd.DataFrame] = []
        if not tract_city_shares_standard.empty:
            tract_frame_parts.append(
                build_city_share_truth_model_frame(
                    city_shares=tract_city_shares_standard,
                    bg_prior=tract_prior,
                    bg_crosswalk=tract_crosswalk_standard,
                    year=year,
                )
            )
        if not county_city_shares.empty and not county_crosswalk.empty:
            tract_crosswalk_county = (
                county_crosswalk.assign(
                    tract_id=county_crosswalk["block_group_geoid"].astype("string").str.zfill(12).str.slice(0, 11)
                )
                .groupby(["state_fips", "tract_id", "jurisdiction_id", "jurisdiction_type"], dropna=False)[
                    "allocation_share"
                ]
                .sum()
                .rename("allocation_share")
                .reset_index()
                .rename(columns={"tract_id": "block_group_geoid"})
            )
            tract_city_shares_county = (
                county_city_shares.assign(
                    tract_id=county_city_shares["block_group_geoid"].astype("string").str.zfill(12).str.slice(0, 11)
                )
                .groupby(
                    ["city_name", "jurisdiction_id", "state_fips", "year", "offense", "tract_id", "geocode_quality_tier"],
                    dropna=False,
                )["incident_count"]
                .sum()
                .rename("incident_count")
                .reset_index()
                .rename(columns={"tract_id": "block_group_geoid"})
            )
            tract_frame_parts.append(
                build_city_share_truth_model_frame(
                    city_shares=tract_city_shares_county,
                    bg_prior=tract_prior,
                    bg_crosswalk=tract_crosswalk_county,
                    year=year,
                )
            )
        tract_frame = pd.concat(tract_frame_parts, ignore_index=True) if tract_frame_parts else pd.DataFrame()
        tract_diag, tract_summary = build_city_share_diagnostics(tract_frame, predicted_share_col="model_share")
        tract_diag = tract_diag.assign(model_name=model_name, geography="tract")
        tract_frames.append(tract_diag)
        summary_rows.append(_summarize_allocation_row(model_name, "tract", tract_summary))

    return (
        pd.concat(bg_frames, ignore_index=True) if bg_frames else pd.DataFrame(),
        pd.concat(tract_frames, ignore_index=True) if tract_frames else pd.DataFrame(),
        pd.DataFrame(summary_rows),
    )


def _summarize_allocation_row(model_name: str, geography: str, summary: dict[str, object]) -> dict[str, object]:
    return {
        "validation_family": "city_share_allocation",
        "geography": str(geography),
        "model_name": str(model_name),
        "rows": int(summary.get("rows", 0) or 0),
        "city_count": int(summary.get("city_count", 0) or 0),
        "incident_total": float(summary.get("incident_total", 0.0) or 0.0),
        "weighted_total_variation_distance_mean": summary.get("weighted_total_variation_distance_mean"),
        "weighted_share_rmse_mean": summary.get("weighted_share_rmse_mean"),
        "weighted_pearson_share_mean": summary.get("weighted_pearson_share_mean"),
        "weighted_spearman_share_mean": summary.get("weighted_spearman_share_mean"),
        "weighted_top_10pct_true_mass_in_model_top_10pct_mean": summary.get(
            "weighted_top_10pct_true_mass_in_model_top_10pct_mean"
        ),
        "metric_primary": "weighted_total_variation_distance_mean",
        "lower_is_better": True,
    }


def _summarize_residual_allocator_rows(*, year: int, residual_diagnostics: pd.DataFrame) -> list[dict[str, object]]:
    if residual_diagnostics.empty:
        return []

    rows: list[dict[str, object]] = []

    def append_row(label_col: str, label_value: str, group: pd.DataFrame) -> None:
        rows.append(
            {
                "year": int(year),
                "validation_family": "city_share_residual_holdout",
                "geography": "block_group",
                "model_name": "residual_allocator_leave_one_city_out",
                label_col: label_value,
                "rows": int(len(group)),
                "holdout_group_count": int(group["holdout_jurisdiction_id"].astype(str).nunique())
                if "holdout_jurisdiction_id" in group.columns
                else None,
                "incident_total": float(
                    pd.to_numeric(group["incident_total"], errors="coerce").fillna(0.0).sum()
                ),
                "baseline_weighted_total_variation_distance_mean": weighted_mean(
                    group,
                    "baseline_total_variation_distance",
                    "incident_total",
                ),
                "residual_weighted_total_variation_distance_mean": weighted_mean(
                    group,
                    "residual_total_variation_distance",
                    "incident_total",
                ),
                "weighted_tvd_delta": weighted_mean(group, "tvd_delta", "incident_total"),
                "baseline_weighted_pearson_share_mean": weighted_mean(
                    group,
                    "baseline_pearson_share",
                    "incident_total",
                ),
                "residual_weighted_pearson_share_mean": weighted_mean(
                    group,
                    "residual_pearson_share",
                    "incident_total",
                ),
                "baseline_weighted_top10_capture_mean": weighted_mean(
                    group,
                    "baseline_top_10pct_true_mass_in_model_top_10pct",
                    "incident_total",
                ),
                "residual_weighted_top10_capture_mean": weighted_mean(
                    group,
                    "residual_top_10pct_true_mass_in_model_top_10pct",
                    "incident_total",
                ),
                "improved_tvd_rows": int((pd.to_numeric(group["tvd_delta"], errors="coerce") < 0).sum()),
                "worsened_tvd_rows": int((pd.to_numeric(group["tvd_delta"], errors="coerce") > 0).sum()),
                "metric_primary": "residual_weighted_total_variation_distance_mean",
                "lower_is_better": True,
            }
        )

    append_row("summary_scope", "overall", residual_diagnostics)
    if "validation_case_type" in residual_diagnostics.columns:
        for case_type, group in residual_diagnostics.groupby("validation_case_type", dropna=False, sort=True):
            append_row("validation_case_type", str(case_type), group)
    return rows


def _classification(row: pd.Series) -> str:
    reported = _num(row.get("reported_count"), default=0.0)
    total_l1_error = _num(row.get("total_l1_error"), default=np.nan)
    allocation_l1_error = _num(row.get("allocation_l1_error"), default=np.nan)
    tvd = _num(row.get("allocation_tvd"), default=np.nan)
    offense = str(row.get("offense", "")).strip()
    floor = float(OFFENSE_COUNT_FLOORS.get(offense, 100.0))
    if reported < floor:
        return "too_sparse"
    if not np.isfinite(total_l1_error) or not np.isfinite(allocation_l1_error) or not np.isfinite(tvd):
        return "missing_prediction_or_allocation"
    relative_total_error = total_l1_error / reported if reported > 0 else np.inf
    if relative_total_error < LOW_ERROR_RELATIVE_ERROR_MAX and tvd < LOW_ERROR_TVD_MAX:
        return "low_error"
    if (
        total_l1_error > DOMINANCE_RATIO * allocation_l1_error
        and relative_total_error > TOTAL_DOMINATED_RELATIVE_ERROR_MIN
    ):
        return "total_dominated"
    if allocation_l1_error > DOMINANCE_RATIO * total_l1_error and tvd > ALLOCATION_DOMINATED_TVD_MIN:
        return "allocation_dominated"
    return "mixed"


def build_error_budget(
    *,
    year: int,
    city_shares: pd.DataFrame,
    cv_predictions: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    allocation_diagnostics: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    error_city_shares = city_shares.copy()
    if "validation_case_type" in error_city_shares.columns:
        error_city_shares = error_city_shares[
            ~error_city_shares["validation_case_type"]
            .astype("string")
            .fillna("")
            .eq("suburban_county_validation_case")
        ].copy()
    truth_keys = (
        error_city_shares[pd.to_numeric(error_city_shares["year"], errors="coerce").eq(int(year))]
        .groupby(["city_name", "jurisdiction_id", "state_fips", "offense"], dropna=False)["incident_count"]
        .sum()
        .rename("incident_total")
        .reset_index()
    )
    meta_cols = [
        "city_name",
        "jurisdiction_id",
        "state_fips",
        "validation_case_type",
        "validation_source_name",
        "validation_source_url",
        "validation_case_notes",
    ]
    available_meta_cols = [col for col in meta_cols if col in city_shares.columns]
    if {"city_name", "jurisdiction_id", "state_fips"}.issubset(available_meta_cols):
        truth_meta = (
            error_city_shares[available_meta_cols]
            .drop_duplicates(["city_name", "jurisdiction_id", "state_fips"])
            .copy()
        )
    else:
        truth_meta = truth_keys[["city_name", "jurisdiction_id", "state_fips"]].drop_duplicates().copy()
    for col in ["validation_case_type", "validation_source_name", "validation_source_url", "validation_case_notes"]:
        if col not in truth_meta.columns:
            truth_meta[col] = ""
    controls = controls[controls["offense"].isin(OFFENSES_7)].copy()
    controls["reported_count"] = pd.to_numeric(controls["adjusted_count_ags_core"], errors="coerce").fillna(0.0)
    totals = controls[
        [
            "jurisdiction_id",
            "offense",
            "reported_count",
            "preferred_source",
            "quality_tier_preferred",
            "estimated_from_panel",
            "needs_current_year_fill",
        ]
    ].copy()
    cv = cv_predictions[cv_predictions["split_mode"].isin(["leave_one_city_out", "leave_truth_city_out"])].copy()
    cv = cv[["jurisdiction_id", "offense", "predicted_count", "predicted_rate_per_100k", "fold_id"]].rename(
        columns={
            "predicted_count": "heldout_predicted_count",
            "predicted_rate_per_100k": "heldout_predicted_rate_per_100k",
            "fold_id": "heldout_fold_id",
        }
    )
    residual = residual_diagnostics[
        [
            "holdout_city_name",
            "holdout_jurisdiction_id",
            "jurisdiction_id",
            "state_fips",
            "offense",
            "baseline_total_variation_distance",
            "residual_total_variation_distance",
            "baseline_pearson_share",
            "residual_pearson_share",
            "baseline_top_10pct_true_mass_in_model_top_10pct",
            "residual_top_10pct_true_mass_in_model_top_10pct",
            "primary_geocode_quality_tier",
            "fallback_incident_share",
            "offense_volume_band",
        ]
    ].copy()
    residual = residual.rename(
        columns={
            "holdout_city_name": "residual_holdout_city_name",
            "holdout_jurisdiction_id": "residual_holdout_jurisdiction_id",
        }
    )
    allocation_fallback = allocation_diagnostics[
        allocation_diagnostics["model_name"].astype(str).eq("ags_core_output")
        & allocation_diagnostics["geography"].astype(str).eq("block_group")
    ].copy()
    allocation_fallback = allocation_fallback[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "offense",
            "total_variation_distance",
            "pearson_share",
            "top_10pct_true_mass_in_model_top_10pct",
            "primary_geocode_quality_tier",
            "fallback_incident_share",
            "offense_volume_band",
        ]
    ].rename(
        columns={
            "total_variation_distance": "fallback_allocator_total_variation_distance",
            "pearson_share": "fallback_allocator_pearson_share",
            "top_10pct_true_mass_in_model_top_10pct": "fallback_allocator_top_10pct_true_mass_in_model_top_10pct",
            "primary_geocode_quality_tier": "fallback_primary_geocode_quality_tier",
            "fallback_incident_share": "fallback_allocator_incident_share",
            "offense_volume_band": "fallback_offense_volume_band",
        }
    )
    budget = (
        truth_keys.merge(totals, on=["jurisdiction_id", "offense"], how="left")
        .merge(cv, on=["jurisdiction_id", "offense"], how="left")
        .merge(residual, on=["jurisdiction_id", "state_fips", "offense"], how="left")
        .merge(allocation_fallback, on=["city_name", "jurisdiction_id", "state_fips", "offense"], how="left")
        .merge(truth_meta, on=["city_name", "jurisdiction_id", "state_fips"], how="left")
    )
    budget["reported_count"] = pd.to_numeric(budget["reported_count"], errors="coerce").fillna(0.0)
    budget["heldout_predicted_count"] = pd.to_numeric(
        budget["heldout_predicted_count"],
        errors="coerce",
    )
    budget["residual_total_variation_distance"] = pd.to_numeric(
        budget["residual_total_variation_distance"],
        errors="coerce",
    )
    budget["baseline_total_variation_distance"] = pd.to_numeric(
        budget["baseline_total_variation_distance"],
        errors="coerce",
    )
    budget["fallback_allocator_total_variation_distance"] = pd.to_numeric(
        budget["fallback_allocator_total_variation_distance"],
        errors="coerce",
    )
    budget["allocation_tvd"] = (
        budget["residual_total_variation_distance"]
        .fillna(budget["fallback_allocator_total_variation_distance"])
        .fillna(budget["baseline_total_variation_distance"])
    )
    budget["allocation_diagnostic_source"] = np.select(
        [
            budget["residual_total_variation_distance"].notna(),
            budget["fallback_allocator_total_variation_distance"].notna(),
            budget["baseline_total_variation_distance"].notna(),
        ],
        [
            "residual_leave_one_city_out",
            "ags_core_output_current_output_fallback",
            "baseline_prior_from_residual_benchmark",
        ],
        default="missing",
    )
    for target, fallback_col in [
        ("primary_geocode_quality_tier", "fallback_primary_geocode_quality_tier"),
        ("fallback_incident_share", "fallback_allocator_incident_share"),
        ("offense_volume_band", "fallback_offense_volume_band"),
    ]:
        budget[target] = budget[target].fillna(budget[fallback_col])
    budget["total_l1_error"] = (budget["heldout_predicted_count"] - budget["reported_count"]).abs()
    budget["total_relative_error"] = np.where(
        budget["reported_count"].gt(0),
        budget["total_l1_error"] / budget["reported_count"],
        np.nan,
    )
    budget["allocation_l1_error"] = (
        2.0
        * budget["reported_count"]
        * budget["allocation_tvd"]
    )
    budget["allocation_moved_mass"] = budget["reported_count"] * budget["allocation_tvd"]
    budget["error_budget_l1_total"] = (
        budget["total_l1_error"].fillna(0.0) + budget["allocation_l1_error"].fillna(0.0)
    )
    budget["total_l1_error_share"] = np.where(
        budget["error_budget_l1_total"].gt(0),
        budget["total_l1_error"] / budget["error_budget_l1_total"],
        np.nan,
    )
    budget["allocation_l1_error_share"] = np.where(
        budget["error_budget_l1_total"].gt(0),
        budget["allocation_l1_error"] / budget["error_budget_l1_total"],
        np.nan,
    )
    case_types = sorted(
        {
            str(value)
            for value in budget.get("validation_case_type", pd.Series(dtype="object")).dropna().unique().tolist()
            if str(value).strip()
        }
    )
    if any(case_type != "promoted_city_control" for case_type in case_types):
        budget["diagnostic_scope"] = "expanded_validation_truth_cases_pinned_control_cold_start"
    else:
        budget["diagnostic_scope"] = "current_12_truth_cities_large_city_pinned_control_cold_start"
    budget["classification_rule"] = (
        "too_sparse if reported count is below offense floor "
        f"{OFFENSE_COUNT_FLOORS}; low_error if total relative error < "
        f"{LOW_ERROR_RELATIVE_ERROR_MAX:g} and TVD < {LOW_ERROR_TVD_MAX:g}; "
        f"total_dominated if total_l1_error > {DOMINANCE_RATIO:g} * allocation_l1_error "
        f"and total relative error > {TOTAL_DOMINATED_RELATIVE_ERROR_MIN:g}; "
        f"allocation_dominated if allocation_l1_error > {DOMINANCE_RATIO:g} * total_l1_error "
        f"and TVD > {ALLOCATION_DOMINATED_TVD_MIN:g}; otherwise mixed"
    )
    budget["error_dominance_class"] = budget.apply(_classification, axis=1)
    return budget.sort_values(["city_name", "offense"], kind="mergesort").reset_index(drop=True)


def _build_decision_table(
    *,
    year: int,
    allocation_summary: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    error_budget: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not allocation_summary.empty:
        rows.extend(allocation_summary.assign(year=int(year)).to_dict(orient="records"))
    if not cv_metrics.empty:
        for row in cv_metrics.to_dict(orient="records"):
            rows.append(
                {
                    "year": int(year),
                    "validation_family": "jurisdiction_total_cv",
                    "geography": "jurisdiction",
                    "model_name": f"hist_gbm_{row['split_mode']}",
                    "offense": row["offense"],
                    "rows": row["rows"],
                    "holdout_group_count": row["holdout_group_count"],
                    "r2_log_rate": row["r2_log_rate"],
                    "r2_rate": row["r2_rate"],
                    "rmse_log_rate": row["rmse_log_rate"],
                    "mean_absolute_count_error": row["mean_absolute_count_error"],
                    "metric_primary": "r2_log_rate",
                    "lower_is_better": False,
                }
            )
    rows.extend(_summarize_residual_allocator_rows(year=year, residual_diagnostics=residual_diagnostics))
    if not error_budget.empty:
        for group_name, group in error_budget.groupby("error_dominance_class", dropna=False, sort=True):
            rows.append(
                {
                    "year": int(year),
                    "validation_family": "error_budget",
                    "geography": "city_offense",
                    "model_name": "heldout_total_plus_residual_allocator",
                    "error_dominance_class": str(group_name),
                    "rows": int(len(group)),
                    "reported_count_total": float(pd.to_numeric(group["reported_count"], errors="coerce").fillna(0.0).sum()),
                    "incident_total": float(pd.to_numeric(group["incident_total"], errors="coerce").fillna(0.0).sum()),
                    "weighted_total_l1_error_mean": weighted_mean(group, "total_l1_error", "reported_count"),
                    "weighted_allocation_l1_error_mean": weighted_mean(group, "allocation_l1_error", "reported_count"),
                    "weighted_allocation_moved_mass_mean": weighted_mean(
                        group,
                        "allocation_moved_mass",
                        "reported_count",
                    ),
                    "weighted_total_l1_error_share_mean": weighted_mean(group, "total_l1_error_share", "reported_count"),
                    "weighted_allocation_l1_error_share_mean": weighted_mean(
                        group,
                        "allocation_l1_error_share",
                        "reported_count",
                    ),
                    "metric_primary": "weighted_total_l1_error_share_mean",
                    "lower_is_better": None,
                }
            )
        rows.append(_build_next_workstream_decision_row(year=year, error_budget=error_budget))
    return pd.DataFrame(rows)


def _build_next_workstream_decision_row(*, year: int, error_budget: pd.DataFrame) -> dict[str, object]:
    material = error_budget[
        ~error_budget["error_dominance_class"].astype("string").fillna("").isin(
            ["too_sparse", "missing_prediction_or_allocation"]
        )
    ].copy()
    if material.empty:
        recommendation = "collect_more_validation_truth"
        rationale = "No material error-budget rows were available after sparse/missing exclusions."
        allocation_rows = total_rows = mixed_rows = 0
        allocation_weight = total_weight = mixed_weight = 0.0
    else:
        classes = material["error_dominance_class"].astype("string").fillna("unknown")
        weights = pd.to_numeric(material["reported_count"], errors="coerce").fillna(0.0)
        allocation_rows = int(classes.eq("allocation_dominated").sum())
        total_rows = int(classes.eq("total_dominated").sum())
        mixed_rows = int(classes.eq("mixed").sum())
        allocation_weight = float(weights[classes.eq("allocation_dominated")].sum())
        total_weight = float(weights[classes.eq("total_dominated")].sum())
        mixed_weight = float(weights[classes.eq("mixed")].sum())
        if allocation_rows >= max(total_rows * 2, 1) and allocation_weight >= max(total_weight * 2.0, 1.0):
            recommendation = "allocator_expansion_first"
            rationale = (
                "Allocation-dominated rows and reported-count weight materially exceed "
                "total-dominated rows after sparse rows are excluded."
            )
        elif total_rows >= max(allocation_rows * 2, 1) and total_weight >= max(allocation_weight * 2.0, 1.0):
            recommendation = "large_city_total_work_first"
            rationale = (
                "Total-dominated rows and reported-count weight materially exceed "
                "allocation-dominated rows after sparse rows are excluded."
            )
        else:
            recommendation = "allocator_and_total_work_in_parallel"
            rationale = (
                "Neither allocation nor total error clears a two-to-one dominance threshold "
                "on both rows and reported-count weight."
            )
    return {
        "year": int(year),
        "validation_family": "next_phase_workstream_decision",
        "geography": "city_offense",
        "model_name": "heldout_total_plus_allocator_error_budget",
        "metric_primary": "error_dominance_class",
        "lower_is_better": None,
        "recommended_next_workstream": recommendation,
        "decision_rationale": rationale,
        "allocation_dominated_rows_material": allocation_rows,
        "total_dominated_rows_material": total_rows,
        "mixed_rows_material": mixed_rows,
        "allocation_dominated_reported_count_material": allocation_weight,
        "total_dominated_reported_count_material": total_weight,
        "mixed_reported_count_material": mixed_weight,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the next-phase measurement spine and city/offense error-budget artifacts.",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--split-mode",
        action="append",
        choices=["kfold", "leave_state_out", "leave_large_city_out", "leave_one_city_out", "leave_truth_city_out"],
        default=None,
        help="CV split mode to emit. May be repeated. Defaults to kfold, leave_large_city_out, and leave_one_city_out.",
    )
    parser.add_argument(
        "--city-shares-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet",
    )
    parser.add_argument(
        "--bg-crosswalk-path",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument(
        "--bg-prior-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--residual-diagnostics-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cv-predictions-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cv-metrics-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--allocation-diagnostics-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--allocation-summary-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--error-budget-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--decision-table-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    year = int(args.year)
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    bg_prior_path = args.bg_prior_path or (REPO_ROOT / "state" / "modeling" / f"bg_prior_long_{year}.parquet")
    residual_path = args.residual_diagnostics_path or (
        REPO_ROOT / "state" / "modeling" / f"city_residual_benchmark_{year}.parquet"
    )
    cv_predictions_out = args.cv_predictions_out or (
        REPO_ROOT / "state" / "modeling" / "jurisdiction_cv_predictions.parquet"
    )
    cv_metrics_out = args.cv_metrics_out or (
        REPO_ROOT / "state" / "modeling" / "jurisdiction_cv_prediction_metrics.parquet"
    )
    allocation_diagnostics_out = args.allocation_diagnostics_out or (
        REPO_ROOT / "state" / "modeling" / f"measurement_spine_allocation_diagnostics_{year}.parquet"
    )
    allocation_summary_out = args.allocation_summary_out or (
        REPO_ROOT / "materials" / "tables" / "measurement_spine_allocation_summary.csv"
    )
    error_budget_out = args.error_budget_out or (
        REPO_ROOT / "materials" / "tables" / "error_budget_city_offense.csv"
    )
    decision_table_out = args.decision_table_out or (
        REPO_ROOT / "materials" / "tables" / "measurement_spine_decision_table.csv"
    )
    summary_json_out = args.summary_json_out or (
        REPO_ROOT / "state" / "modeling" / f"next_phase_measurement_summary_{year}.json"
    )
    split_modes = tuple(
        dict.fromkeys(
            "leave_one_city_out" if str(value) == "leave_truth_city_out" else str(value)
            for value in (args.split_mode or DEFAULT_SPLIT_MODES)
        )
    )

    city_shares = pd.read_parquet(args.city_shares_path)
    city_shares["jurisdiction_id"] = city_shares["jurisdiction_id"].astype("string")
    bg_crosswalk = pd.read_parquet(args.bg_crosswalk_path)
    bg_prior = pd.read_parquet(bg_prior_path)
    residual_diagnostics = pd.read_parquet(residual_path)
    controls = pd.read_parquet(paths.state_dir / "controls" / f"jurisdiction_controls_{year}.parquet")

    truth_city_meta_cols = ["jurisdiction_id"]
    if "validation_case_type" in city_shares.columns:
        truth_city_meta_cols.append("validation_case_type")
    truth_city_ids = (
        city_shares.loc[pd.to_numeric(city_shares["year"], errors="coerce").eq(year), truth_city_meta_cols]
        .dropna(subset=["jurisdiction_id"])
        .drop_duplicates("jurisdiction_id")
        .copy()
    )
    if "validation_case_type" in truth_city_ids.columns:
        truth_city_ids_for_cv = truth_city_ids[
            ~truth_city_ids["validation_case_type"].astype("string").fillna("").eq("suburban_county_validation_case")
        ].copy()
    else:
        truth_city_ids_for_cv = truth_city_ids
    cv_predictions, cv_metrics = build_jurisdiction_cv_predictions(
        paths=paths,
        config=ModelSurfaceConfig(year=year),
        truth_city_jurisdiction_ids=truth_city_ids_for_cv,
        split_modes=split_modes,
    )
    bg_diag, tract_diag, allocation_summary = build_allocation_measurement_tables(
        paths=paths,
        year=year,
        city_shares=city_shares,
        bg_crosswalk=bg_crosswalk,
        bg_prior=bg_prior,
    )
    allocation_diagnostics = pd.concat([bg_diag, tract_diag], ignore_index=True)
    error_budget = build_error_budget(
        year=year,
        city_shares=city_shares,
        cv_predictions=cv_predictions,
        residual_diagnostics=residual_diagnostics,
        allocation_diagnostics=allocation_diagnostics,
        controls=controls,
    )
    decision_table = _build_decision_table(
        year=year,
        allocation_summary=allocation_summary,
        cv_metrics=cv_metrics,
        residual_diagnostics=residual_diagnostics,
        error_budget=error_budget,
    )

    for path in [
        cv_predictions_out,
        cv_metrics_out,
        allocation_diagnostics_out,
        allocation_summary_out,
        error_budget_out,
        decision_table_out,
        summary_json_out,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    cv_predictions.to_parquet(cv_predictions_out, index=False)
    cv_metrics.to_parquet(cv_metrics_out, index=False)
    allocation_diagnostics.to_parquet(allocation_diagnostics_out, index=False)
    allocation_summary.to_csv(allocation_summary_out, index=False)
    error_budget.to_csv(error_budget_out, index=False)
    decision_table.to_csv(decision_table_out, index=False)
    decision_rows = (
        decision_table[decision_table["validation_family"].astype(str).eq("next_phase_workstream_decision")]
        if "validation_family" in decision_table.columns
        else pd.DataFrame()
    )
    decision_row = decision_rows.iloc[0].to_dict() if not decision_rows.empty else {}

    summary = {
        "year": year,
        "split_modes": list(split_modes),
        "truth_case_count": int(truth_city_ids["jurisdiction_id"].nunique()),
        "jurisdiction_total_truth_case_count": int(truth_city_ids_for_cv["jurisdiction_id"].nunique()),
        "truth_city_count": int(truth_city_ids_for_cv["jurisdiction_id"].nunique()),
        "truth_case_type_counts": truth_city_ids["validation_case_type"].value_counts(dropna=False).to_dict()
        if "validation_case_type" in truth_city_ids.columns
        else {},
        "cv_prediction_rows": int(len(cv_predictions)),
        "cv_metric_rows": int(len(cv_metrics)),
        "allocation_diagnostic_rows": int(len(allocation_diagnostics)),
        "allocation_summary_rows": int(len(allocation_summary)),
        "error_budget_rows": int(len(error_budget)),
        "decision_table_rows": int(len(decision_table)),
        "recommended_next_workstream": decision_row.get("recommended_next_workstream"),
        "decision_rationale": decision_row.get("decision_rationale"),
        "residual_allocator_summary": decision_table[
            decision_table["validation_family"].astype(str).eq("city_share_residual_holdout")
        ].to_dict(orient="records")
        if "validation_family" in decision_table.columns
        else [],
        "outputs": {
            "cv_predictions": str(cv_predictions_out),
            "cv_metrics": str(cv_metrics_out),
            "residual_diagnostics_input": str(residual_path),
            "allocation_diagnostics": str(allocation_diagnostics_out),
            "allocation_summary": str(allocation_summary_out),
            "error_budget": str(error_budget_out),
            "decision_table": str(decision_table_out),
        },
        "error_budget_class_counts": error_budget["error_dominance_class"].value_counts(dropna=False).to_dict()
        if "error_dominance_class" in error_budget.columns
        else {},
        "allocation_summary": allocation_summary.to_dict(orient="records"),
    }
    summary_json_out.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
