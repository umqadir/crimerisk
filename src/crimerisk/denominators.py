from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths


BURGLARY_COMMERCIAL_WEIGHT_FALLBACK = 41.0
LANDSCAN_BG_RELATIVE_PATH = Path("LandScan-USA") / "block_group_landscan_usa_2021.parquet"
LANDSCAN_DAY_POP_COLUMN = "landscan_day_pop"
LANDSCAN_SOURCE_YEAR = 2021
BURGLARY_NNLS_TERM_COLUMNS = (
    "households_total",
    "destination_poi_total",
    "lodes_retail_jobs",
    "lodes_industrial_jobs",
)
BURGLARY_TERM_TO_K_FIELD = {
    "destination_poi_total": "k_destination_poi",
    "lodes_retail_jobs": "k_retail_jobs",
    "lodes_industrial_jobs": "k_industrial_jobs",
}
BURGLARY_LODES_COMPONENT_COLUMNS = {
    "lodes_manufacturing_jobs": "CNS05",
    "lodes_wholesale_jobs": "CNS06",
    "lodes_retail_jobs": "CNS07",
    "lodes_transport_warehouse_jobs": "CNS08",
}
PERSON_EXPOSURE_DENOMINATOR_OFFENSES = frozenset(
    ("murder", "rape", "robbery", "aggravated_assault", "larceny")
)
PRIMARY_DENOMINATOR_BY_OFFENSE = {
    "murder": "exposure",
    "rape": "exposure",
    "robbery": "exposure",
    "aggravated_assault": "exposure",
    "burglary": "premises",
    "larceny": "exposure",
    "motor_vehicle_theft": "vehicles",
}
DENOMINATOR_SOURCE_COLUMNS = {
    "exposure": "exposure_proxy_2024",
    "households": "households_total",
    "premises": "burglary_premises_total",
    "vehicles": "vehicle_exposure_2024",
    # "resident" is currently unmapped by any offense in PRIMARY_DENOMINATOR_BY_OFFENSE
    # (dead entry). The published population column is year-derived (population_<year>),
    # so if an offense is ever pointed at "resident", this value must become year-derived
    # too or it will KeyError on any non-2024 build.
    "resident": "population_2024",
}


def offense_denominator_column(offense: str) -> str:
    offense_key = str(offense)
    if offense_key not in PRIMARY_DENOMINATOR_BY_OFFENSE:
        raise KeyError(f"Unknown offense for denominator lookup: {offense!r}")
    return f"offense_denominator_{offense_key}"


def _numeric(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(float(default))


def _county_auto_commute_share(out: pd.DataFrame) -> pd.Series:
    county_geoid = out["bg_id"].astype("string").str.zfill(12).str.slice(0, 5)
    commute_total = _numeric(out, "commute_mode_total").clip(lower=0.0)
    auto_commute = (
        _numeric(out, "commute_drive_alone").clip(lower=0.0)
        + _numeric(out, "commute_carpool").clip(lower=0.0)
    )
    county = (
        pd.DataFrame(
            {
                "county_geoid": county_geoid,
                "auto_commute": auto_commute,
                "commute_total": commute_total,
            }
        )
        .groupby("county_geoid", dropna=False)[["auto_commute", "commute_total"]]
        .sum()
        .reset_index()
    )
    county["county_auto_commute_vehicle_share"] = np.where(
        county["commute_total"].gt(0.0),
        county["auto_commute"] / county["commute_total"],
        np.nan,
    )
    share = pd.DataFrame({"county_geoid": county_geoid}, index=out.index).merge(
        county[["county_geoid", "county_auto_commute_vehicle_share"]],
        on="county_geoid",
        how="left",
    )["county_auto_commute_vehicle_share"]
    bg_share = np.where(commute_total.gt(0.0), auto_commute / commute_total.replace(0.0, np.nan), np.nan)
    national_share = (
        float(auto_commute.sum() / commute_total.sum()) if float(commute_total.sum()) > 0.0 else 0.0
    )
    return (
        pd.to_numeric(share, errors="coerce")
        .combine_first(pd.Series(bg_share, index=out.index, dtype=float))
        .fillna(float(national_share))
        .clip(lower=0.0, upper=1.0)
    )


def _population_for_denominators(bg: pd.DataFrame, *, year: int) -> pd.Series:
    candidates = [
        f"population_{int(year)}",
        "population_for_weights",
        "total_population",
        "pop20",
    ]
    out: pd.Series | None = None
    for column in candidates:
        if column not in bg.columns:
            continue
        values = pd.to_numeric(bg[column], errors="coerce")
        out = values if out is None else out.combine_first(values)
    if out is None:
        out = pd.Series(0.0, index=bg.index, dtype=float)
    return out.fillna(0.0).clip(lower=0.0)


def _landscan_bg_path(paths: RepoPaths) -> Path:
    return paths.data_dir / LANDSCAN_BG_RELATIVE_PATH


def _attach_landscan_day_population(bg: pd.DataFrame, *, paths: RepoPaths) -> pd.DataFrame:
    out = bg.copy()
    if LANDSCAN_DAY_POP_COLUMN in out.columns and pd.to_numeric(
        out[LANDSCAN_DAY_POP_COLUMN],
        errors="coerce",
    ).notna().any():
        out[LANDSCAN_DAY_POP_COLUMN] = (
            pd.to_numeric(out[LANDSCAN_DAY_POP_COLUMN], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        return out
    out = out.drop(columns=[LANDSCAN_DAY_POP_COLUMN], errors="ignore")

    path = _landscan_bg_path(paths)
    if not path.exists():
        raise FileNotFoundError(
            "Missing LandScan USA block-group denominator extract: "
            f"{path}. Build or place the Q1 LandScan USA 2021 aggregation parquet before output builds."
        )
    landscan = pd.read_parquet(
        path,
        columns=["block_group_geoid", LANDSCAN_DAY_POP_COLUMN],
    ).copy()
    landscan["bg_id"] = landscan["block_group_geoid"].astype("string").str.zfill(12)
    landscan[LANDSCAN_DAY_POP_COLUMN] = (
        pd.to_numeric(landscan[LANDSCAN_DAY_POP_COLUMN], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    landscan = (
        landscan.groupby("bg_id", dropna=False)[LANDSCAN_DAY_POP_COLUMN]
        .sum()
        .reset_index()
    )
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    out = out.merge(landscan, on="bg_id", how="left")
    out[LANDSCAN_DAY_POP_COLUMN] = (
        pd.to_numeric(out[LANDSCAN_DAY_POP_COLUMN], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    return out


def _attach_commercial_premises(bg: pd.DataFrame, *, paths: RepoPaths) -> pd.DataFrame:
    out = bg.copy()
    if "commercial_premises_total" in out.columns and pd.to_numeric(
        out["commercial_premises_total"],
        errors="coerce",
    ).notna().any():
        out["commercial_premises_total"] = (
            pd.to_numeric(out["commercial_premises_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        out["destination_poi_total"] = out["commercial_premises_total"]
        return out
    out = out.drop(columns=["commercial_premises_total"], errors="ignore")

    overture_places_path = (
        paths.data_dir / "Overture-Places" / "parsed" / "block_group_overture_places_states_latest.parquet"
    )
    if not overture_places_path.exists():
        out["commercial_premises_total"] = 0.0
        out["destination_poi_total"] = 0.0
        return out

    commercial = pd.read_parquet(
        overture_places_path,
        columns=["bg_id", "overture_consumer_destination_any_count"],
    ).copy()
    commercial["bg_id"] = commercial["bg_id"].astype("string").str.zfill(12)
    commercial["commercial_premises_total"] = (
        pd.to_numeric(commercial["overture_consumer_destination_any_count"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    commercial = (
        commercial.groupby("bg_id", dropna=False)["commercial_premises_total"]
        .sum()
        .reset_index()
    )
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    out = out.merge(commercial, on="bg_id", how="left")
    out["commercial_premises_total"] = (
        pd.to_numeric(out.get("commercial_premises_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    out["destination_poi_total"] = out["commercial_premises_total"]
    return out


def _attach_burglary_lodes_job_terms(bg: pd.DataFrame, *, paths: RepoPaths) -> pd.DataFrame:
    out = bg.copy()
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)

    target_cols = list(BURGLARY_LODES_COMPONENT_COLUMNS.keys())
    if all(col in out.columns for col in target_cols):
        for col in target_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        out["lodes_industrial_jobs"] = (
            out["lodes_manufacturing_jobs"]
            + out["lodes_wholesale_jobs"]
            + out["lodes_transport_warehouse_jobs"]
        )
        return out

    for col in [*target_cols, "lodes_industrial_jobs"]:
        out = out.drop(columns=[col], errors="ignore")

    lodes_path = paths.data_dir / "LODES" / "parsed" / "lodes_wac_block_groups.parquet"
    if not lodes_path.exists():
        for col in target_cols:
            out[col] = 0.0
        out["lodes_industrial_jobs"] = 0.0
        return out

    columns = ["bg_id", *BURGLARY_LODES_COMPONENT_COLUMNS.values()]
    lodes = pd.read_parquet(lodes_path, columns=columns).copy()
    lodes["bg_id"] = lodes["bg_id"].astype("string").str.zfill(12)
    for raw_col in BURGLARY_LODES_COMPONENT_COLUMNS.values():
        lodes[raw_col] = pd.to_numeric(lodes[raw_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    lodes = lodes.groupby("bg_id", dropna=False)[list(BURGLARY_LODES_COMPONENT_COLUMNS.values())].sum().reset_index()
    rename_map = {raw: target for target, raw in BURGLARY_LODES_COMPONENT_COLUMNS.items()}
    lodes = lodes.rename(columns=rename_map)
    out = out.merge(lodes, on="bg_id", how="left")
    for col in target_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["lodes_industrial_jobs"] = (
        out["lodes_manufacturing_jobs"]
        + out["lodes_wholesale_jobs"]
        + out["lodes_transport_warehouse_jobs"]
    )
    return out


def _read_share_surface(path: Path, *, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.DataFrame(columns=columns)


def _load_burglary_calibration_rows(
    *,
    paths: RepoPaths,
    year: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = ["city_name", "jurisdiction_id", "state_fips", "year", "offense", "block_group_geoid", "incident_count"]
    primary_path = paths.state_dir / "modeling" / f"next_phase_validation_city_incident_share_surface_{int(year)}.parquet"
    source_paths: list[str] = []
    appended_paths: list[str] = []
    appended_city_names: set[str] = set()
    appended_jurisdiction_ids: set[str] = set()
    appended_rows = 0

    if primary_path.exists():
        surface = _read_share_surface(primary_path, columns=columns)
        source_paths.append(str(primary_path))
    else:
        legacy_path = paths.state_dir / "modeling" / "city_incident_share_surface.parquet"
        surface = _read_share_surface(legacy_path, columns=columns) if legacy_path.exists() else pd.DataFrame(columns=columns)
        if legacy_path.exists():
            source_paths.append(str(legacy_path))

    for col in columns:
        if col not in surface.columns:
            surface[col] = pd.NA
    truth_city_names = set(surface["city_name"].dropna().astype(str).unique().tolist())
    truth_jurisdiction_ids = set(surface["jurisdiction_id"].dropna().astype(str).unique().tolist())

    # The assembled next-phase validation surface already folds every gate-17
    # share-admitted packet city into itself (geocoding-integrity-v10, single re-derived
    # path in build_next_phase_validation_shares), so it is the single source of burglary
    # calibration truth: no packet surface is appended here. The admission file is kept
    # only as a provenance reference in the returned metadata.
    admission_path = paths.repo_root / "configs" / "gate17_city_offense_admissions.csv"

    direct = surface[surface["offense"].astype("string").eq("burglary")][columns].copy()
    direct["block_group_geoid"] = direct["block_group_geoid"].astype("string").str.zfill(12)
    direct["incident_count"] = pd.to_numeric(direct["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)

    metadata = {
        "calibration_path": str(primary_path if primary_path.exists() else paths.state_dir / "modeling" / "city_incident_share_surface.parquet"),
        "calibration_paths": source_paths,
        "calibration_truth_surface": (
            "next_phase_validation_plus_gate17_share_admitted_packets"
            if primary_path.exists()
            else "city_incident_share_surface"
        ),
        "calibration_admission_path": str(admission_path) if admission_path.exists() else None,
        "calibration_truth_city_count": int(len(truth_jurisdiction_ids)),
        "calibration_truth_city_names": sorted(truth_city_names),
        "calibration_appended_city_count": int(len(appended_jurisdiction_ids)),
        "calibration_appended_city_names": sorted(appended_city_names),
        "calibration_appended_paths": appended_paths,
        "calibration_appended_rows": int(appended_rows),
        "source_year_pooling": "sum incident_count over all source-year block-group rows before NNLS",
    }
    return direct, metadata


def _fit_burglary_premises_nnls(
    *,
    direct: pd.DataFrame,
    covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], float, dict[str, object]]:
    observed = (
        direct.groupby("block_group_geoid", dropna=False)["incident_count"]
        .sum()
        .rename("observed_burglary_count")
        .reset_index()
    )
    calibration = observed.merge(
        covariates[["block_group_geoid", *BURGLARY_NNLS_TERM_COLUMNS]],
        on="block_group_geoid",
        how="inner",
    )
    for col in ("observed_burglary_count", *BURGLARY_NNLS_TERM_COLUMNS):
        calibration[col] = pd.to_numeric(calibration[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    calibration = calibration[
        calibration["observed_burglary_count"].ge(0.0)
        & calibration[list(BURGLARY_NNLS_TERM_COLUMNS)].sum(axis=1).gt(0.0)
    ].copy()
    x = calibration[list(BURGLARY_NNLS_TERM_COLUMNS)].to_numpy(dtype=float)
    y = pd.to_numeric(calibration["observed_burglary_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    coef, residual_norm = nnls(x, y)
    coefficients = {term: float(value) for term, value in zip(BURGLARY_NNLS_TERM_COLUMNS, coef, strict=True)}
    household_coef = float(coefficients["households_total"])
    k_vector = {
        k_field: (
            float(coefficients[term] / household_coef)
            if household_coef > 0.0 and np.isfinite(coefficients[term])
            else float("nan")
        )
        for term, k_field in BURGLARY_TERM_TO_K_FIELD.items()
    }

    old_x = calibration[["households_total", "destination_poi_total"]].to_numpy(dtype=float)
    old_coef, old_residual_norm = nnls(old_x, y)
    old_household_coef = float(old_coef[0])
    old_destination_coef = float(old_coef[1])
    old_form = {
        "terms": ["households_total", "destination_poi_total"],
        "household_coef": old_household_coef,
        "destination_poi_coef": old_destination_coef,
        "commercial_coef": old_destination_coef,
        "k_destination_poi": (
            old_destination_coef / old_household_coef if old_household_coef > 0.0 else float("nan")
        ),
        "k_commercial": (
            old_destination_coef / old_household_coef if old_household_coef > 0.0 else float("nan")
        ),
        "nnls_residual_norm": float(old_residual_norm),
        "comparison_basis": "same calibration rows as four-term NNLS",
    }
    return calibration, coefficients, k_vector, float(residual_norm), old_form


def _fit_burglary_commercial_nnls(
    *,
    direct: pd.DataFrame,
    covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float, float, float]:
    cov = covariates.copy()
    if "destination_poi_total" not in cov.columns:
        cov["destination_poi_total"] = pd.to_numeric(
            cov.get("commercial_premises_total"), errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
    for col in ("lodes_retail_jobs", "lodes_industrial_jobs"):
        if col not in cov.columns:
            cov[col] = 0.0
    calibration, coefficients, k_vector, residual_norm, _old_form = _fit_burglary_premises_nnls(
        direct=direct,
        covariates=cov,
    )
    return (
        calibration,
        float(coefficients["households_total"]),
        float(coefficients["destination_poi_total"]),
        float(k_vector["k_destination_poi"]),
        float(residual_norm),
    )


def _build_burglary_fold_safe_calibration(
    *,
    direct: pd.DataFrame,
    covariates: pd.DataFrame,
) -> dict[str, object]:
    if not {"jurisdiction_id", "city_name"}.issubset(direct.columns):
        return {"fold_count": 0, "folds": []}
    cities = (
        direct[["jurisdiction_id", "city_name"]]
        .dropna(subset=["jurisdiction_id"])
        .drop_duplicates()
        .sort_values(["city_name", "jurisdiction_id"], kind="mergesort")
    )
    folds: list[dict[str, object]] = []
    for row in cities.itertuples(index=False):
        heldout_jurisdiction_id = str(row.jurisdiction_id)
        heldout_city_name = str(row.city_name)
        train = direct[direct["jurisdiction_id"].astype(str).ne(heldout_jurisdiction_id)].copy()
        if train.empty:
            continue
        try:
            fold_calibration, coefficients, k_vector, residual_norm, old_form = _fit_burglary_premises_nnls(
                direct=train,
                covariates=covariates,
            )
        except Exception:
            continue
        years = pd.to_numeric(train["year"], errors="coerce").dropna()
        folds.append(
            {
                "heldout_city_name": heldout_city_name,
                "heldout_jurisdiction_id": heldout_jurisdiction_id,
                "training_city_count": int(train["jurisdiction_id"].nunique(dropna=True)),
                "training_rows": int(len(train)),
                "training_block_groups": int(len(fold_calibration)),
                "training_year_min": int(years.min()) if not years.empty else None,
                "training_year_max": int(years.max()) if not years.empty else None,
                "household_coef": float(coefficients["households_total"]),
                "destination_poi_coef": float(coefficients["destination_poi_total"]),
                "commercial_coef": float(coefficients["destination_poi_total"]),
                "retail_jobs_coef": float(coefficients["lodes_retail_jobs"]),
                "industrial_jobs_coef": float(coefficients["lodes_industrial_jobs"]),
                "coefficients": coefficients,
                "k_destination_poi": float(k_vector["k_destination_poi"]),
                "k_retail_jobs": float(k_vector["k_retail_jobs"]),
                "k_industrial_jobs": float(k_vector["k_industrial_jobs"]),
                "k_commercial": float(k_vector["k_destination_poi"]),
                "k_vector": k_vector,
                "nnls_residual_norm": float(residual_norm),
                "single_term_nnls_residual_norm": old_form.get("nnls_residual_norm"),
            }
        )
    k_values = pd.to_numeric(pd.Series([fold["k_destination_poi"] for fold in folds]), errors="coerce").dropna()
    summary: dict[str, object] = {
        "criterion": "per-held-out-burglary-city NNLS on all other calibration cities",
        "fold_count": int(len(folds)),
        "folds": folds,
    }
    if not k_values.empty:
        summary.update(
            {
                "k_min": float(k_values.min()),
                "k_p25": float(k_values.quantile(0.25)),
                "k_median": float(k_values.median()),
                "k_p75": float(k_values.quantile(0.75)),
                "k_max": float(k_values.max()),
            }
        )
    k_ranges: dict[str, dict[str, float]] = {}
    for k_field in ("k_destination_poi", "k_retail_jobs", "k_industrial_jobs"):
        values = pd.to_numeric(pd.Series([fold.get(k_field) for fold in folds]), errors="coerce").dropna()
        if values.empty:
            continue
        k_ranges[k_field] = {
            "min": float(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "max": float(values.max()),
        }
    if k_ranges:
        summary["k_ranges"] = k_ranges
    return summary


def calibrate_burglary_commercial_weight(
    *,
    paths: RepoPaths,
    bg_covariates: pd.DataFrame,
    year: int = 2024,
    override: float | None = None,
) -> dict[str, object]:
    if override is not None:
        weight = max(float(override), 0.0)
        return {
            "k_commercial": weight,
            "k_destination_poi": weight,
            "k_retail_jobs": 0.0,
            "k_industrial_jobs": 0.0,
            "k_vector": {
                "k_destination_poi": weight,
                "k_retail_jobs": 0.0,
                "k_industrial_jobs": 0.0,
            },
            "denominator_form": "manual_destination_poi_only_override",
            "calibration_source": "manual_config_override",
            "calibration_path": None,
            "calibration_rows": 0,
            "calibration_block_groups": 0,
            "calibration_city_count": 0,
            "calibration_year_min": None,
            "calibration_year_max": None,
            "household_coef": float("nan"),
            "commercial_coef": float("nan"),
            "destination_poi_coef": float("nan"),
            "retail_jobs_coef": 0.0,
            "industrial_jobs_coef": 0.0,
            "coefficients": {},
            "fold_safe": {"fold_count": 0, "folds": []},
            "used_fallback": False,
            "caveat": "Manual override supplied through AllocationBuildConfig.burglary_commercial_weight.",
        }

    fallback = {
        "k_commercial": float(BURGLARY_COMMERCIAL_WEIGHT_FALLBACK),
        "k_destination_poi": float(BURGLARY_COMMERCIAL_WEIGHT_FALLBACK),
        "k_retail_jobs": 0.0,
        "k_industrial_jobs": 0.0,
        "k_vector": {
            "k_destination_poi": float(BURGLARY_COMMERCIAL_WEIGHT_FALLBACK),
            "k_retail_jobs": 0.0,
            "k_industrial_jobs": 0.0,
        },
        "denominator_form": "fallback_destination_poi_only",
        "calibration_source": "fallback_lead_nnls_prior",
        "calibration_path": None,
        "calibration_rows": 0,
        "calibration_block_groups": 0,
        "calibration_city_count": 0,
        "calibration_year_min": None,
        "calibration_year_max": None,
        "household_coef": float("nan"),
        "commercial_coef": float("nan"),
        "destination_poi_coef": float("nan"),
        "retail_jobs_coef": 0.0,
        "industrial_jobs_coef": 0.0,
        "coefficients": {},
        "fold_safe": {"fold_count": 0, "folds": []},
        "used_fallback": True,
        "caveat": (
            "Covered-city raw burglary incident calibration was unavailable or ill-conditioned; "
            "using the lead-confirmed NNLS prior weight."
        ),
    }

    try:
        direct, source_metadata = _load_burglary_calibration_rows(paths=paths, year=int(year))
    except Exception:
        return fallback
    if direct.empty:
        return fallback
    cov = bg_covariates[
        ["bg_id", "households_total", "destination_poi_total", "lodes_retail_jobs", "lodes_industrial_jobs"]
    ].copy()
    cov["block_group_geoid"] = cov["bg_id"].astype("string").str.zfill(12)
    for col in BURGLARY_NNLS_TERM_COLUMNS:
        cov[col] = pd.to_numeric(cov[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    try:
        calibration, coefficients, k_vector, residual_norm, old_form = _fit_burglary_premises_nnls(
            direct=direct,
            covariates=cov,
        )
    except Exception:
        return {
            **fallback,
            **source_metadata,
            "calibration_rows": int(len(direct)),
            "calibration_block_groups": 0,
            "calibration_city_count": int(direct["jurisdiction_id"].nunique(dropna=True)),
        }
    if len(calibration) < 100:
        return {
            **fallback,
            **source_metadata,
            "calibration_rows": int(len(direct)),
            "calibration_block_groups": int(len(calibration)),
            "calibration_city_count": int(direct["jurisdiction_id"].nunique(dropna=True)),
        }
    household_coef = float(coefficients["households_total"])
    if (
        not np.isfinite(household_coef)
        or household_coef <= 0.0
        or any(not np.isfinite(value) for value in k_vector.values())
    ):
        return {
            **fallback,
            **source_metadata,
            "calibration_rows": int(len(direct)),
            "calibration_block_groups": int(len(calibration)),
            "calibration_city_count": int(direct["jurisdiction_id"].nunique(dropna=True)),
            "household_coef": household_coef,
            "commercial_coef": coefficients.get("destination_poi_total"),
            "destination_poi_coef": coefficients.get("destination_poi_total"),
            "retail_jobs_coef": coefficients.get("lodes_retail_jobs"),
            "industrial_jobs_coef": coefficients.get("lodes_industrial_jobs"),
            "coefficients": coefficients,
        }
    if not np.isfinite(k_vector["k_destination_poi"]) or k_vector["k_destination_poi"] <= 0.0:
        return {
            **fallback,
            **source_metadata,
            "calibration_rows": int(len(direct)),
            "calibration_block_groups": int(len(calibration)),
            "calibration_city_count": int(direct["jurisdiction_id"].nunique(dropna=True)),
            "household_coef": household_coef,
            "commercial_coef": coefficients.get("destination_poi_total"),
            "destination_poi_coef": coefficients.get("destination_poi_total"),
            "retail_jobs_coef": coefficients.get("lodes_retail_jobs"),
            "industrial_jobs_coef": coefficients.get("lodes_industrial_jobs"),
            "coefficients": coefficients,
        }

    years = pd.to_numeric(direct["year"], errors="coerce").dropna()
    fold_safe = _build_burglary_fold_safe_calibration(direct=direct, covariates=cov)
    residual_delta = float(residual_norm) - float(old_form["nnls_residual_norm"])
    return {
        "k_commercial": float(k_vector["k_destination_poi"]),
        "k_destination_poi": float(k_vector["k_destination_poi"]),
        "k_retail_jobs": float(k_vector["k_retail_jobs"]),
        "k_industrial_jobs": float(k_vector["k_industrial_jobs"]),
        "k_vector": k_vector,
        "denominator_form": "households_destination_poi_retail_industrial_jobs_nnls",
        "denominator_formula": (
            "households_total + k_destination_poi * destination_poi_total + "
            "k_retail_jobs * lodes_retail_jobs + k_industrial_jobs * "
            "(lodes_manufacturing_jobs + lodes_wholesale_jobs + lodes_transport_warehouse_jobs)"
        ),
        "denominator_terms": list(BURGLARY_NNLS_TERM_COLUMNS),
        "calibration_source": "covered_city_direct_burglary_incident_nnls",
        **source_metadata,
        "calibration_rows": int(len(direct)),
        "calibration_block_groups": int(len(calibration)),
        "calibration_city_count": int(direct["jurisdiction_id"].nunique(dropna=True)),
        "calibration_city_names": sorted(direct["city_name"].dropna().astype(str).unique().tolist()),
        "calibration_year_min": int(years.min()) if not years.empty else None,
        "calibration_year_max": int(years.max()) if not years.empty else None,
        "household_coef": float(coefficients["households_total"]),
        "commercial_coef": float(coefficients["destination_poi_total"]),
        "destination_poi_coef": float(coefficients["destination_poi_total"]),
        "retail_jobs_coef": float(coefficients["lodes_retail_jobs"]),
        "industrial_jobs_coef": float(coefficients["lodes_industrial_jobs"]),
        "coefficients": coefficients,
        "nnls_residual_norm": float(residual_norm),
        "single_term_comparison": old_form,
        "single_term_nnls_residual_norm": float(old_form["nnls_residual_norm"]),
        "nnls_residual_delta_vs_single_term": residual_delta,
        "beats_single_term_nnls_residual": bool(residual_delta < 0.0),
        "fold_safe": fold_safe,
        "used_fallback": False,
        "caveat": None,
    }


def add_offense_denominators(
    bg: pd.DataFrame,
    *,
    paths: RepoPaths,
    year: int,
    burglary_commercial_weight: float | None = None,
    apply_landscan_day_floor: bool = True,
) -> pd.DataFrame:
    out = bg.copy()
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    if "tract_id" in out.columns:
        out["tract_id"] = out["tract_id"].astype("string").str.zfill(11)
    if "state_fips" in out.columns:
        out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)

    population = _population_for_denominators(out, year=int(year))
    population_year_col = f"population_{int(year)}"
    out["population"] = population
    if population_year_col in out.columns:
        out[population_year_col] = pd.to_numeric(out[population_year_col], errors="coerce").fillna(population).clip(lower=0.0)
    else:
        out[population_year_col] = population
    out["daytime_population_jobs_proxy"] = (
        population
        + _numeric(out, "jobs_wac").clip(lower=0.0)
        - _numeric(out, "jobs_rac").clip(lower=0.0)
    ).clip(lower=0.0)
    out["daytime_population_jobs_proxy"] = np.maximum(
        pd.to_numeric(out["daytime_population_jobs_proxy"], errors="coerce").fillna(0.0),
        population,
    )
    jobs_wac = _numeric(out, "jobs_wac").clip(lower=0.0)
    jobs_exposure = pd.to_numeric(out["daytime_population_jobs_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if apply_landscan_day_floor:
        out = _attach_landscan_day_population(out, paths=paths)
        landscan_day = pd.to_numeric(out[LANDSCAN_DAY_POP_COLUMN], errors="coerce").fillna(0.0).clip(lower=0.0)
        landscan_positive = landscan_day.where(landscan_day.gt(0.0), 0.0)
        out["exposure_proxy_2024"] = np.maximum(jobs_exposure, landscan_positive)
        out["landscan_day_lifted_person_exposure"] = landscan_positive.gt(jobs_exposure)
    else:
        if LANDSCAN_DAY_POP_COLUMN not in out.columns:
            out[LANDSCAN_DAY_POP_COLUMN] = 0.0
        out[LANDSCAN_DAY_POP_COLUMN] = pd.to_numeric(
            out[LANDSCAN_DAY_POP_COLUMN], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        out["exposure_proxy_2024"] = jobs_exposure
        out["landscan_day_lifted_person_exposure"] = False
        landscan_day = pd.to_numeric(out[LANDSCAN_DAY_POP_COLUMN], errors="coerce").fillna(0.0).clip(lower=0.0)
    person_exposure_before_hq_cap = pd.to_numeric(out["exposure_proxy_2024"], errors="coerce").fillna(0.0).clip(lower=0.0)
    hq_cap_base = np.maximum(landscan_day, population)
    hq_cap_value = 3.0 * pd.Series(hq_cap_base, index=out.index, dtype=float).clip(lower=0.0)
    hq_cap_candidate = (
        jobs_wac.ge(5000.0)
        & (population + jobs_wac).gt(hq_cap_value)
        & hq_cap_value.gt(0.0)
    )
    hq_cap_applied = hq_cap_candidate & person_exposure_before_hq_cap.gt(hq_cap_value)
    out["person_exposure_before_hq_jobs_cap"] = person_exposure_before_hq_cap
    out["person_exposure_hq_jobs_cap"] = hq_cap_value
    out["person_exposure_hq_jobs_cap_candidate"] = hq_cap_candidate
    out["person_exposure_hq_jobs_capped"] = hq_cap_applied
    out.loc[hq_cap_applied, "exposure_proxy_2024"] = hq_cap_value.loc[hq_cap_applied]
    out["households_total"] = _numeric(out, "households_total").clip(lower=0.0)

    aggregate_vehicles = pd.to_numeric(out.get("aggregate_vehicles_total"), errors="coerce")
    vehicle_bins = [
        ("owner_1_vehicle", 1.0),
        ("renter_1_vehicle", 1.0),
        ("owner_2_vehicle", 2.0),
        ("renter_2_vehicle", 2.0),
        ("owner_3_vehicle", 3.0),
        ("renter_3_vehicle", 3.0),
        ("owner_4_vehicle", 4.0),
        ("renter_4_vehicle", 4.0),
        ("owner_5plus_vehicle", 5.0),
        ("renter_5plus_vehicle", 5.0),
    ]
    aggregate_vehicles_fallback = pd.Series(0.0, index=out.index, dtype=float)
    for col, vehicles_per_household in vehicle_bins:
        aggregate_vehicles_fallback += vehicles_per_household * _numeric(out, col).clip(lower=0.0)
    out["aggregate_vehicles_total"] = aggregate_vehicles.where(
        aggregate_vehicles.ge(0.0),
        aggregate_vehicles_fallback,
    ).fillna(aggregate_vehicles_fallback).clip(lower=0.0)
    out["county_auto_commute_vehicle_share"] = _county_auto_commute_share(out)
    out["mvt_commuter_vehicle_proxy"] = (
        jobs_wac * pd.to_numeric(out["county_auto_commute_vehicle_share"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    out["vehicle_exposure_2024"] = (
        pd.to_numeric(out["aggregate_vehicles_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
        + pd.to_numeric(out["mvt_commuter_vehicle_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    out = _attach_commercial_premises(out, paths=paths)
    out = _attach_burglary_lodes_job_terms(out, paths=paths)
    calibration = calibrate_burglary_commercial_weight(
        paths=paths,
        bg_covariates=out,
        year=int(year),
        override=burglary_commercial_weight,
    )
    destination_weight = float(calibration.get("k_destination_poi", calibration["k_commercial"]))
    retail_weight = float(calibration.get("k_retail_jobs", 0.0))
    industrial_weight = float(calibration.get("k_industrial_jobs", 0.0))
    out["burglary_premises_total"] = (
        pd.to_numeric(out["households_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
        + destination_weight * pd.to_numeric(out["destination_poi_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
        + retail_weight * pd.to_numeric(out["lodes_retail_jobs"], errors="coerce").fillna(0.0).clip(lower=0.0)
        + industrial_weight * pd.to_numeric(out["lodes_industrial_jobs"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    out["burglary_commercial_exposure_weight"] = destination_weight
    out["burglary_destination_poi_exposure_weight"] = destination_weight
    out["burglary_retail_jobs_exposure_weight"] = retail_weight
    out["burglary_industrial_jobs_exposure_weight"] = industrial_weight

    for offense in OFFENSES_7:
        denominator_type = PRIMARY_DENOMINATOR_BY_OFFENSE[offense]
        source_col = DENOMINATOR_SOURCE_COLUMNS[denominator_type]
        out[offense_denominator_column(offense)] = (
            pd.to_numeric(out[source_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        )

    out.attrs["burglary_commercial_calibration"] = calibration
    return out
