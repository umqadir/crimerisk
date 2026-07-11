from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from crimerisk.covariates.feature_filters import select_numeric_feature_columns
from crimerisk.covariates.features import CovariateBuildConfig, build_block_group_covariates
from crimerisk.crosswalk_shares import normalize_block_group_allocation_shares
from crimerisk.crime import OFFENSES_7
from crimerisk.denominators import add_offense_denominators, offense_denominator_column
from crimerisk.monotone_gam import MonotoneGAMRegressor, monotone_gam_feature_columns
from crimerisk.paths import RepoPaths
from crimerisk.qa.model_diagnostics import cross_validated_diagnostics

LEAVE_LARGE_CITY_OUT_MIN_POPULATION = 500_000
LEAVE_CBSA_OUT_DOMINANT_SHARE_MIN = 0.75
LEAVE_CBSA_OUT_MIN_JURISDICTIONS = 10
LEAVE_CBSA_OUT_MIN_POPULATION = 500_000
LEAVE_CBSA_OUT_RELAXED_MIN_JURISDICTIONS = 5
LEAVE_CBSA_OUT_RELAXED_MIN_POPULATION = 250_000
MODEL_FEATURE_GROUPS = (
    "activity_exposure",
    "roads_transport",
    "land_cover",
    "institutional_anchors",
    "socioeconomic_household",
)
SPARSE_OFFENSES_DEFAULT = ("murder", "rape")


@dataclass(frozen=True)
class ModelSurfaceConfig:
    year: int = 2024
    min_training_population: int = 10_000
    model_family: str = "hist_gbm"
    ridge_alpha: float = 1.0
    elastic_net_alpha: float = 0.001
    elastic_net_l1_ratio: float = 0.5
    elastic_net_max_iter: int = 5000
    monotone_gam_lam: float = 0.6
    monotone_gam_n_splines: int = 12
    monotone_gam_max_iter: int = 200
    hist_learning_rate: float = 0.03
    hist_max_depth: int = 5
    hist_max_iter: int = 500
    hist_min_samples_leaf: int = 20
    hist_l2_regularization: float = 1.0
    use_state_fixed_effects: bool = True
    compute_diagnostics: bool = True
    exclude_estimated_from_panel_from_training: bool = True
    high_confidence_training_only: bool = False
    exclude_feature_groups: tuple[str, ...] = ()
    sparse_offense_pooling_strategy: str = "none"
    sparse_offense_pooling_offenses: tuple[str, ...] = SPARSE_OFFENSES_DEFAULT
    prior_anchor: str = "resident_population"
    feature_policy_path: Path | None = None
    exclude_feature_policy_classes: tuple[str, ...] = ()
    burglary_commercial_weight: float | None = None


BURGLARY_RAW_COMMERCIAL_EXPOSURE_POLICY_GROUP = "activity_exposure_commercial"
BURGLARY_EXPOSURE_DUPLICATE_LABEL = "mass-duplicate"
BURGLARY_RESIDUAL_LABEL = "residual"

# Pinned covariate vintages (v1.0 mechanism freeze: NOT bumped for the 2025 refresh).
# Single point of control for the vintage strings that were previously duplicated
# between bg_feature_dependency_paths and build_bg_feature_frame; the HPMS extract
# in particular was dependency-tracked off the build year while the builder read the
# pinned 2024 extract -- both now key off HPMS_BG_VINTAGE_YEAR.
ACS_BUNDLE_DIRNAME = "ACS-5yr-2020-2024"
HPMS_BG_VINTAGE_YEAR = 2024
NLCD_BG_VINTAGE_YEAR = 2023


def _base_exclude_feature_cols() -> set[str]:
    return {
        "bg_id",
        "tract_id",
        "state_fips",
        "county_fips",
        "NAME",
        "median_household_income",
        "per_capita_income",
        "median_rent",
        "median_home_value",
        "pop20",
        "housing20",
        "aland20",
        "awater20",
        "population_2024",
        "lodes_year",
        "population_for_weights",
        "housing_for_weights",
        "lodes_cs02_share",
        "tenure_renter_share",
        "hh_nonfamily_share",
        "hh_no_computer_share",
        "housing_vacant_share",
        "net_commuter_jobs_per_capita",
        "drivable_road_total_length_km",
        "hh_internet_subscription_share",
        "public_school_present",
        "postsecondary_present",
    }


def _feature_group_for_column(column: str) -> str:
    col = str(column).strip().lower()
    if (
        col.startswith("lodes_")
        or col.startswith("jobs_")
        or col.startswith("log_jobs_")
        or col.startswith("daytime_population_jobs_proxy")
        or col.startswith("log_daytime_population_jobs_proxy")
        or col.startswith("workplace_")
        or col.startswith("resident_same_bg")
        or col.startswith("resident_jobs")
        or col.startswith("commute_")
        or col.startswith("overture_")
        or col.startswith("log_overture_")
        or col.startswith("nearest_overture_")
        or col in {"employed_per_capita", "labor_force_per_capita"}
    ):
        return "activity_exposure"
    if (
        col.startswith("hpms_")
        or col.startswith("transit_")
        or col.startswith("fixed_guideway_")
        or col.startswith("nearest_transit_")
        or col.startswith("nearest_fixed_guideway_")
        or col.endswith("_road_length_km")
        or col in {
            "road_total_length_km",
            "internal_census_road_length_km",
            "limited_access_road_share",
        }
    ):
        return "roads_transport"
    if col.startswith("nlcd_") or col in {"land_area_sqkm", "log_aland20"}:
        return "land_cover"
    if col in {
        "education_anchor_density_sqkm",
        "public_school_density_sqkm",
        "postsecondary_density_sqkm",
        "nearest_acute_care_hospital_km",
        "nearest_emergency_hospital_km",
    }:
        return "institutional_anchors"
    return "socioeconomic_household"


def bg_feature_dependency_paths(paths: RepoPaths, *, year: int) -> list[Path]:
    module_root = Path(__file__).resolve().parent
    dependency_paths: list[Path] = [
        paths.data_dir / ACS_BUNDLE_DIRNAME / "parsed" / "acs_block_groups.parquet",
        paths.data_dir / ACS_BUNDLE_DIRNAME / "parsed" / "acs_tracts_full.parquet",
        paths.data_dir / "LODES" / "parsed" / "lodes_wac_block_groups.parquet",
        paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        paths.cache_dir / "cpi" / "CPIAUCSL.csv",
        paths.data_dir / "Census-PopEst-2020-2025" / "co-est2025-alldata.csv",
        paths.data_dir / "roads" / "parsed" / "block_group_road_metrics.parquet",
        paths.cache_dir / "roads" / "block_group_road_metrics_from_states.parquet",
        paths.data_dir / "HPMS" / "parsed" / f"block_group_hpms_{HPMS_BG_VINTAGE_YEAR}.parquet",
        paths.cache_dir / "HPMS" / f"block_group_hpms_{HPMS_BG_VINTAGE_YEAR}_from_states.parquet",
        paths.data_dir / "NTM" / "parsed" / "block_group_transit_stops.parquet",
        paths.data_dir / "NCES-EDGE" / "parsed" / "block_group_education_anchors_2425.parquet",
        paths.data_dir / "CMS-Hospital-General-Info" / "parsed" / "block_group_hospital_anchors.parquet",
        paths.data_dir / "NLCD" / "parsed" / f"block_group_nlcd_{NLCD_BG_VINTAGE_YEAR}.parquet",
        paths.data_dir / "tiger_bg" / "tl_2023_09_bg.zip",
        paths.data_dir / "tiger_bg" / "tl_2020_09_bg.zip",
        paths.repo_root / "configs" / "acs_missing_bg_decennial_backfill.csv",
        module_root / "model_surface.py",
        module_root / "covariates" / "features.py",
        module_root / "covariates" / "feature_filters.py",
        module_root / "covariates" / "lodes_flow.py",
        module_root / "covariates" / "population.py",
    ]
    dependency_paths.extend(sorted((paths.data_dir / "roads" / "state_block_groups").glob("*.parquet")))
    dependency_paths.extend(sorted((paths.data_dir / "HPMS" / "state_block_groups").glob("*.parquet")))
    dependency_paths.extend(sorted((paths.data_dir / "NLCD" / "state_block_groups").glob("*.parquet")))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in dependency_paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _merge_ntm_transit_features(bg: pd.DataFrame, *, ntm_bg_parquet: Path) -> pd.DataFrame:
    if not ntm_bg_parquet.exists():
        return bg
    ntm_cols = [
        "bg_id",
        "state_fips",
        "transit_stop_count",
        "fixed_guideway_stop_count",
        "transit_mode_count_bg",
        "nearest_transit_stop_km",
        "nearest_fixed_guideway_stop_km",
        "transit_stop_present",
        "fixed_guideway_stop_present",
    ]
    ntm = pd.read_parquet(ntm_bg_parquet, columns=ntm_cols).copy()
    ntm["bg_id"] = ntm["bg_id"].astype("string").str.zfill(12)
    ntm["state_fips"] = ntm["state_fips"].astype("string").str.zfill(2)
    for col in ntm_cols:
        if col in {"bg_id", "state_fips"}:
            continue
        ntm[col] = pd.to_numeric(ntm[col], errors="coerce")

    out = bg.merge(ntm.drop_duplicates(["bg_id", "state_fips"]), on=["bg_id", "state_fips"], how="left")
    area_sqkm = pd.to_numeric(out.get("aland20"), errors="coerce").fillna(0.0) / 1_000_000.0
    area_sqkm = area_sqkm.where(area_sqkm > 0)
    out["transit_stop_density_sqkm"] = (
        pd.to_numeric(out.get("transit_stop_count"), errors="coerce").fillna(0.0) / area_sqkm
    ).replace([np.inf, -np.inf], np.nan)
    out["fixed_guideway_stop_density_sqkm"] = (
        pd.to_numeric(out.get("fixed_guideway_stop_count"), errors="coerce").fillna(0.0) / area_sqkm
    ).replace([np.inf, -np.inf], np.nan)
    return out


def _weighted_mean_by_group(
    df: pd.DataFrame, *, group_col: str, weight_col: str, value_cols: list[str]
) -> pd.DataFrame:
    g = df[[group_col, weight_col, *value_cols]].copy()
    g[weight_col] = pd.to_numeric(g[weight_col], errors="coerce").fillna(0.0)
    out = {group_col: []}
    for c in value_cols:
        out[c] = []

    for key, grp in g.groupby(group_col, sort=False):
        w = grp[weight_col].to_numpy(dtype=float)
        out[group_col].append(key)
        for c in value_cols:
            x = pd.to_numeric(grp[c], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
            denom = w[mask].sum()
            if denom <= 0:
                out[c].append(np.nan)
            else:
                out[c].append(float(np.sum(x[mask] * w[mask]) / denom))
    return pd.DataFrame(out)


def build_bg_feature_frame(*, paths: RepoPaths, year: int) -> pd.DataFrame:
    roads_bg_parquet = paths.data_dir / "roads" / "parsed" / "block_group_road_metrics.parquet"
    if not roads_bg_parquet.exists():
        state_roads_dir = paths.data_dir / "roads" / "state_block_groups"
        state_parts = sorted(state_roads_dir.glob("*.parquet"))
        if state_parts:
            roads_bg_parquet = paths.cache_dir / "roads" / "block_group_road_metrics_from_states.parquet"
            roads_bg_parquet.parent.mkdir(parents=True, exist_ok=True)
            if (not roads_bg_parquet.exists()) or any(part.stat().st_mtime > roads_bg_parquet.stat().st_mtime for part in state_parts):
                pd.concat([pd.read_parquet(part) for part in state_parts], ignore_index=True).to_parquet(
                    roads_bg_parquet,
                    index=False,
                )
    hpms_bg_parquet = paths.data_dir / "HPMS" / "parsed" / f"block_group_hpms_{HPMS_BG_VINTAGE_YEAR}.parquet"
    if not hpms_bg_parquet.exists():
        state_hpms_dir = paths.data_dir / "HPMS" / "state_block_groups"
        state_parts = sorted(state_hpms_dir.glob("*.parquet"))
        if state_parts:
            hpms_bg_parquet = paths.cache_dir / "HPMS" / f"block_group_hpms_{HPMS_BG_VINTAGE_YEAR}_from_states.parquet"
            hpms_bg_parquet.parent.mkdir(parents=True, exist_ok=True)
            if (not hpms_bg_parquet.exists()) or any(part.stat().st_mtime > hpms_bg_parquet.stat().st_mtime for part in state_parts):
                pd.concat([pd.read_parquet(part) for part in state_parts], ignore_index=True).to_parquet(
                    hpms_bg_parquet,
                    index=False,
                )

    bg = build_block_group_covariates(
        acs_bg_parquet=paths.data_dir / ACS_BUNDLE_DIRNAME / "parsed" / "acs_block_groups.parquet",
        lodes_bg_parquet=paths.data_dir / "LODES" / "parsed" / "lodes_wac_block_groups.parquet",
        block_group_crosswalk_parquet=paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        cpi_cache_csv=paths.cache_dir / "cpi" / "CPIAUCSL.csv",
        county_pop_csv=paths.data_dir / "Census-PopEst-2020-2025" / "co-est2025-alldata.csv",
        acs_tract_parquet=paths.data_dir / ACS_BUNDLE_DIRNAME / "parsed" / "acs_tracts_full.parquet",
        roads_bg_parquet=roads_bg_parquet if roads_bg_parquet.exists() else None,
        hpms_bg_parquet=hpms_bg_parquet if hpms_bg_parquet.exists() else None,
        education_anchor_bg_parquet=(
            paths.data_dir / "NCES-EDGE" / "parsed" / "block_group_education_anchors_2425.parquet"
        ),
        medical_anchor_bg_parquet=(
            paths.data_dir / "CMS-Hospital-General-Info" / "parsed" / "block_group_hospital_anchors.parquet"
        ),
        nlcd_bg_parquet=paths.data_dir / "NLCD" / "parsed" / f"block_group_nlcd_{NLCD_BG_VINTAGE_YEAR}.parquet",
        ct_bg_2023_zip=paths.data_dir / "tiger_bg" / "tl_2023_09_bg.zip",
        ct_bg_2020_zip=paths.data_dir / "tiger_bg" / "tl_2020_09_bg.zip",
        acs_missing_bg_backfill_csv=paths.repo_root / "configs" / "acs_missing_bg_decennial_backfill.csv",
        cfg=CovariateBuildConfig(estimate_year=int(year)),
    )
    bg["bg_id"] = bg["bg_id"].astype("string").str.zfill(12)
    bg["tract_id"] = bg["tract_id"].astype("string").str.zfill(11)
    bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    population_col = f"population_{int(year)}"
    bg["population_for_weights"] = (
        pd.to_numeric(bg.get(population_col), errors="coerce")
        .fillna(pd.to_numeric(bg.get("total_population"), errors="coerce"))
        .fillna(pd.to_numeric(bg.get("pop20"), errors="coerce"))
        .fillna(0.0)
        .clip(lower=0.0)
    )
    bg["housing_for_weights"] = (
        pd.to_numeric(bg.get("housing_units_total"), errors="coerce")
        .fillna(pd.to_numeric(bg.get("housing20"), errors="coerce"))
        .fillna(0.0)
        .clip(lower=0.0)
    )
    bg = _merge_ntm_transit_features(
        bg,
        ntm_bg_parquet=paths.data_dir / "NTM" / "parsed" / "block_group_transit_stops.parquet",
    )
    return bg.copy()


def merge_extra_bg_features(bg: pd.DataFrame, *, extra_feature_paths: list[Path] | None = None) -> pd.DataFrame:
    if not extra_feature_paths:
        return bg
    out = bg.copy()
    for path in extra_feature_paths:
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"Extra BG feature parquet not found: {path}")
        extra = pd.read_parquet(path)
        merge_keys = [col for col in ["bg_id", "tract_id", "state_fips"] if col in extra.columns and col in out.columns]
        if "bg_id" not in merge_keys or "state_fips" not in merge_keys:
            raise ValueError(f"Extra BG feature parquet must include bg_id and state_fips: {path}")
        extra = extra.copy()
        for col in merge_keys:
            width = 12 if col == "bg_id" else 11 if col == "tract_id" else 2
            extra[col] = extra[col].astype("string").str.zfill(width)
        keep_cols = [col for col in extra.columns if col not in set(merge_keys)]
        out = out.merge(
            extra[merge_keys + keep_cols].drop_duplicates(merge_keys),
            on=merge_keys,
            how="left",
        )
    return out


def _build_feature_selection(
    bg: pd.DataFrame,
    *,
    exclude_feature_groups: tuple[str, ...] = (),
) -> list[str]:
    selection = select_numeric_feature_columns(
        bg,
        base_exclude=_base_exclude_feature_cols(),
        allow_protected=False,
        exclude_count_like=True,
    )
    excluded_groups = {str(group).strip() for group in exclude_feature_groups if str(group).strip()}
    feature_cols = [
        c for c in selection.feature_cols
        if bg[c].notna().any()
        and _feature_group_for_column(c) not in excluded_groups
    ]
    return sorted(feature_cols)


def _load_feature_policy_excluded_columns(config: ModelSurfaceConfig) -> set[str]:
    if config.feature_policy_path is None or not tuple(config.exclude_feature_policy_classes):
        return set()
    path = Path(config.feature_policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature transfer policy parquet not found: {path}")
    policy = pd.read_parquet(path, columns=["feature_column", "final_class"])
    excluded_classes = {
        str(value).strip()
        for value in config.exclude_feature_policy_classes
        if str(value).strip()
    }
    if not excluded_classes:
        return set()
    return set(
        policy.loc[
            policy["final_class"].astype("string").isin(excluded_classes),
            "feature_column",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )


def _burglary_exposure_duplicate_feature_classification(
    column: object,
    feature_group: object,
    source: object,
) -> tuple[str, str]:
    col = str(column).strip().lower()
    group = str(feature_group).strip().lower()
    src = str(source).strip().lower()
    if group == BURGLARY_RAW_COMMERCIAL_EXPOSURE_POLICY_GROUP:
        if col.startswith("nearest_overture_") or col.startswith("log_overture_nearest_"):
            return BURGLARY_RESIDUAL_LABEL, "Nearest commercial/core distance is accessibility, not premises mass."
        if col.endswith("_share") or col.endswith("_ratio") or col.endswith("_per_capita"):
            return BURGLARY_RESIDUAL_LABEL, "Commercial/workplace ratio or mix; D_burglary already carries exposure volume."
        if col in {"log_jobs_rac"}:
            return BURGLARY_RESIDUAL_LABEL, "Resident-worker scale is not workplace or commercial-premise exposure mass."
        if src == "base_covariate" and (
            col.endswith("_density_sqkm")
            or col
            in {
                "log_daytime_population_jobs_proxy",
                "log_daytime_population_jobs_proxy_density_sqkm",
                "log_jobs_density_sqkm",
                "log_jobs_wac",
            }
        ):
            return BURGLARY_EXPOSURE_DUPLICATE_LABEL, "Raw workplace/commercial volume or density is already represented in D_burglary."
        if col.startswith("log_overture_") and col.endswith("_density_sqkm"):
            return BURGLARY_EXPOSURE_DUPLICATE_LABEL, "Raw Overture commercial category density duplicates commercial exposure in D_burglary."
        if col.startswith("overture_") and (
            col.endswith("_density_sqkm")
            or col.endswith("_present")
            or col.endswith("_within_1km")
            or col.endswith("_within_5km")
            or col.endswith("_core_flag")
            or col.endswith("_core_score")
            or col.endswith("commercial_core_score")
        ):
            return BURGLARY_EXPOSURE_DUPLICATE_LABEL, "Overture commercial presence, radius count, or core intensity is exposure mass already in D_burglary."
        return BURGLARY_RESIDUAL_LABEL, "Commercial/workplace field is ratio, mix, or accessibility rather than raw exposure mass."
    if group == "lodes_industry_activity":
        return BURGLARY_RESIDUAL_LABEL, "LODES industry shares are workplace mix, not raw workplace volume."
    if group in {"lodes_age", "lodes_education", "lodes_earnings"}:
        return BURGLARY_RESIDUAL_LABEL, "LODES worker shares describe workforce composition, not raw workplace volume."
    if group == "land_cover":
        if col in {"nlcd_count_21", "nlcd_count_22", "nlcd_count_23", "nlcd_count_24"}:
            return BURGLARY_EXPOSURE_DUPLICATE_LABEL, "Raw developed-land pixel count is built-premise mass already carried by D_burglary."
        if col in {"nlcd_descriptor_count_1", "nlcd_descriptor_count_2"}:
            return BURGLARY_EXPOSURE_DUPLICATE_LABEL, "Raw road/urban impervious descriptor count is developed activity mass already carried by D_burglary."
        return BURGLARY_RESIDUAL_LABEL, "Land-cover share, area, or non-developed class describes land-use character rather than premise volume."
    if group == "roads_transport":
        if col in {"hpms_aadt_km", "hpms_lane_aadt_km", "hpms_truck_aadt_km"}:
            return BURGLARY_EXPOSURE_DUPLICATE_LABEL, "Traffic-volume kilometers are commercial/activity volume, not residual network structure."
        if col.endswith("_length_km"):
            return BURGLARY_EXPOSURE_DUPLICATE_LABEL, "Raw road length is network volume that proxies activity mass already captured by D_burglary."
        return BURGLARY_RESIDUAL_LABEL, "Transport share, average intensity, stop access, or distance describes network structure/accessibility."
    if group == "population_density":
        if col in {
            "housing_density_sqkm",
            "log_housing_density_sqkm",
            "population_density_sqkm",
            "log_population_density_sqkm",
            "pop_2024",
            "log_total_population",
        }:
            return BURGLARY_EXPOSURE_DUPLICATE_LABEL, "Raw resident or housing volume/density duplicates residential exposure in D_burglary."
        return BURGLARY_RESIDUAL_LABEL, "Population field is growth, occupancy, or ratio rather than raw residential exposure mass."
    return BURGLARY_RESIDUAL_LABEL, "Feature is composition, mix, accessibility, or context rather than D_burglary exposure volume."


def classify_burglary_exposure_duplicate_features(feature_policy_path: Path | None) -> pd.DataFrame:
    if feature_policy_path is None:
        return pd.DataFrame(columns=["feature", "classification", "rationale"])
    path = Path(feature_policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature transfer policy parquet not found: {path}")
    policy = pd.read_parquet(path)
    required = {"feature_column", "feature_group"}
    if not required.issubset(policy.columns):
        return pd.DataFrame(columns=["feature", "classification", "rationale"])
    source_col = "feature_source" if "feature_source" in policy.columns else "source"
    if source_col not in policy.columns:
        policy[source_col] = ""
    rows: list[dict[str, object]] = []
    for _, row in policy.iterrows():
        classification, rationale = _burglary_exposure_duplicate_feature_classification(
            row.get("feature_column"),
            row.get("feature_group"),
            row.get(source_col),
        )
        rows.append(
            {
                "feature": str(row.get("feature_column")),
                "classification": classification,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def burglary_exposure_duplicate_columns(feature_policy_path: Path | None) -> set[str]:
    if feature_policy_path is None:
        return set()
    path = Path(feature_policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature transfer policy parquet not found: {path}")
    policy = pd.read_parquet(path)
    required = {"feature_column", "feature_group"}
    if not required.issubset(policy.columns):
        return set()
    source_col = "feature_source" if "feature_source" in policy.columns else "source"
    if source_col not in policy.columns:
        policy[source_col] = ""
    duplicate_mask = policy.apply(
        lambda row: _burglary_exposure_duplicate_feature_classification(
            row.get("feature_column"),
            row.get("feature_group"),
            row.get(source_col),
        )[0]
        == BURGLARY_EXPOSURE_DUPLICATE_LABEL,
        axis=1,
    )
    return set(
        policy.loc[duplicate_mask, "feature_column"]
        .dropna()
        .astype(str)
        .tolist()
    )


def burglary_raw_commercial_exposure_duplicate_columns(feature_policy_path: Path | None) -> set[str]:
    return burglary_exposure_duplicate_columns(feature_policy_path)


def _apply_feature_policy_exclusions(feature_cols: list[str], excluded_columns: set[str]) -> list[str]:
    if not excluded_columns:
        return list(feature_cols)
    return [col for col in feature_cols if str(col) not in excluded_columns]


def _feature_cols_for_offense(
    feature_cols: list[str],
    *,
    offense: str,
    burglary_exposure_duplicate_columns: set[str],
    denominator_anchor: bool,
) -> list[str]:
    if not denominator_anchor or str(offense) != "burglary" or not burglary_exposure_duplicate_columns:
        return list(feature_cols)
    return [col for col in feature_cols if str(col) not in burglary_exposure_duplicate_columns]


def build_model_feature_columns(
    bg: pd.DataFrame,
    *,
    exclude_feature_groups: tuple[str, ...] = (),
) -> list[str]:
    return _build_feature_selection(bg, exclude_feature_groups=exclude_feature_groups)


def _normalize_county_fips_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    if "." in text:
        parts = [part.strip() for part in text.split(".") if part.strip()]
        if len(parts) >= 2:
            state_digits = re.sub(r"\D", "", parts[0])
            county_digits = re.sub(r"\D", "", parts[1])
            if state_digits and county_digits:
                return f"{int(state_digits):02d}{int(county_digits):03d}"
    digits = re.sub(r"\D", "", text)
    if not digits:
        return pd.NA
    if len(digits) <= 5:
        return digits.zfill(5)
    if len(digits) >= 5:
        return f"{int(digits[:2]):02d}{int(digits[-3:]):03d}"
    return pd.NA


def _load_county_to_cbsa_delineation(paths: RepoPaths) -> pd.DataFrame:
    csv_path = paths.repo_root / "configs" / "county_to_cbsa_2023.csv"
    xlsx_path = paths.data_dir / "CBSA-Delineation" / "list1_2023.xlsx"
    if csv_path.exists():
        out = pd.read_csv(csv_path, dtype=str).fillna("").copy()
        required = {"county_fips", "cbsa_code", "cbsa_title"}
        missing_cols = sorted(required - set(out.columns))
        if missing_cols:
            raise ValueError(f"Tracked CBSA lookup missing columns: {missing_cols}")
        out["county_fips"] = out["county_fips"].map(_normalize_county_fips_value).astype("string")
        out["cbsa_code"] = out["cbsa_code"].astype("string").str.strip()
        out["cbsa_title"] = out["cbsa_title"].astype("string").str.strip()
        out = out[out["county_fips"].notna() & out["cbsa_code"].notna()].copy()
        return out[["county_fips", "cbsa_code", "cbsa_title"]].drop_duplicates().reset_index(drop=True)
    if not xlsx_path.exists():
        return pd.DataFrame(columns=["county_fips", "cbsa_code", "cbsa_title"])
    df = pd.read_excel(xlsx_path, header=2)
    rename_map = {
        "CBSA Code": "cbsa_code",
        "CBSA Title": "cbsa_title",
        "FIPS State Code": "state_fips",
        "FIPS County Code": "county_fips_suffix",
    }
    missing_cols = [col for col in rename_map if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CBSA delineation file missing columns: {missing_cols}")
    out = df.rename(columns=rename_map)[list(rename_map.values())].copy()
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["county_fips_suffix"] = out["county_fips_suffix"].astype("string").str.zfill(3)
    out["county_fips"] = out["state_fips"] + out["county_fips_suffix"]
    out["cbsa_code"] = out["cbsa_code"].astype("string").str.strip()
    out["cbsa_title"] = out["cbsa_title"].astype("string").str.strip()
    out = out[out["county_fips"].notna() & out["cbsa_code"].notna()].copy()
    return out[["county_fips", "cbsa_code", "cbsa_title"]].drop_duplicates().reset_index(drop=True)


def _dominant_weighted_label_frame(
    merged: pd.DataFrame,
    *,
    label_col: str,
    output_col: str,
    share_col: str,
) -> pd.DataFrame:
    work = merged[["jurisdiction_id", "aggregation_weight", label_col]].copy()
    work["aggregation_weight"] = pd.to_numeric(work["aggregation_weight"], errors="coerce").fillna(0.0)
    work[label_col] = work[label_col].astype("string")
    work = work[work[label_col].notna() & work["aggregation_weight"].gt(0)].copy()
    if work.empty:
        return pd.DataFrame(columns=["jurisdiction_id", output_col, share_col])
    grouped = (
        work.groupby(["jurisdiction_id", label_col], dropna=False)["aggregation_weight"]
        .sum()
        .rename("label_weight")
        .reset_index()
    )
    totals = (
        grouped.groupby("jurisdiction_id", dropna=False)["label_weight"]
        .sum()
        .rename("jurisdiction_weight_total")
        .reset_index()
    )
    grouped = grouped.merge(totals, on="jurisdiction_id", how="left")
    grouped["label_share"] = np.where(
        pd.to_numeric(grouped["jurisdiction_weight_total"], errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(grouped["label_weight"], errors="coerce").fillna(0.0)
        / pd.to_numeric(grouped["jurisdiction_weight_total"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    grouped = grouped.sort_values(
        ["jurisdiction_id", "label_weight", label_col],
        ascending=[True, False, True],
        kind="mergesort",
    )
    dominant = grouped.drop_duplicates(["jurisdiction_id"], keep="first").copy()
    dominant = dominant.rename(columns={label_col: output_col, "label_share": share_col})
    return dominant[["jurisdiction_id", output_col, share_col]].reset_index(drop=True)


def _select_cbsa_holdout_groups(
    *,
    training: pd.DataFrame,
    train_mask: pd.Series,
) -> list[str]:
    eligible_rows = training.loc[
        train_mask
        & training["jurisdiction_type"].eq("municipal")
        & training["dominant_cbsa_code"].astype("string").notna()
        & pd.to_numeric(training["dominant_cbsa_share"], errors="coerce").fillna(0.0).ge(
            float(LEAVE_CBSA_OUT_DOMINANT_SHARE_MIN)
        ),
        ["dominant_cbsa_code", "bucket_population"],
    ].copy()
    if eligible_rows.empty:
        return []
    eligible_rows["dominant_cbsa_code"] = eligible_rows["dominant_cbsa_code"].astype("string").astype(str)
    eligible_rows["bucket_population"] = pd.to_numeric(
        eligible_rows["bucket_population"],
        errors="coerce",
    ).fillna(0.0)
    summary = (
        eligible_rows.groupby("dominant_cbsa_code", dropna=False)
        .agg(
            jurisdiction_count=("dominant_cbsa_code", "size"),
            total_population=("bucket_population", "sum"),
        )
        .reset_index()
    )
    summary = summary[
        summary["jurisdiction_count"].ge(int(LEAVE_CBSA_OUT_MIN_JURISDICTIONS))
        & summary["total_population"].ge(float(LEAVE_CBSA_OUT_MIN_POPULATION))
    ].copy()
    if not summary.empty:
        return sorted(summary["dominant_cbsa_code"].astype(str).tolist())
    relaxed = (
        eligible_rows.groupby("dominant_cbsa_code", dropna=False)
        .agg(
            jurisdiction_count=("dominant_cbsa_code", "size"),
            total_population=("bucket_population", "sum"),
        )
        .reset_index()
    )
    relaxed = relaxed[
        relaxed["jurisdiction_count"].ge(int(LEAVE_CBSA_OUT_RELAXED_MIN_JURISDICTIONS))
        & relaxed["total_population"].ge(float(LEAVE_CBSA_OUT_RELAXED_MIN_POPULATION))
    ].copy()
    if not relaxed.empty:
        return sorted(relaxed["dominant_cbsa_code"].astype(str).tolist())
    fallback = (
        eligible_rows.groupby("dominant_cbsa_code", dropna=False)
        .agg(
            jurisdiction_count=("dominant_cbsa_code", "size"),
            total_population=("bucket_population", "sum"),
        )
        .reset_index()
        .sort_values(
            ["jurisdiction_count", "total_population", "dominant_cbsa_code"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(10)
    )
    return sorted(fallback["dominant_cbsa_code"].astype(str).tolist())


def _sparse_pooling_pop_band(pop: object) -> str:
    value = float(pd.to_numeric(pop, errors="coerce")) if pd.notna(pop) else float("nan")
    if not np.isfinite(value) or value <= 0:
        return "unknown"
    if value < 25_000:
        return "10k-25k"
    if value < 50_000:
        return "25k-50k"
    if value < 100_000:
        return "50k-100k"
    if value < 250_000:
        return "100k-250k"
    return "250k+"


def _build_sparse_offense_pooling_target(
    *,
    frame: pd.DataFrame,
    counts: pd.Series,
    pop: pd.Series,
    offense: str,
    config: ModelSurfaceConfig,
    observed_only_mask: pd.Series,
) -> tuple[np.ndarray, dict[str, object]]:
    raw_rate = np.where(pop > 0, counts / pop * 1e5, 0.0)
    pooling_strategy = str(config.sparse_offense_pooling_strategy).strip().lower()
    allowed_offenses = {str(value).strip() for value in config.sparse_offense_pooling_offenses if str(value).strip()}
    if pooling_strategy == "none" or offense not in allowed_offenses:
        return raw_rate, {
            "strategy": "none",
            "applied_rows": 0,
            "prior_source_counts": {},
        }
    if pooling_strategy != "empirical_bayes_group_rate":
        raise ValueError(f"Unsupported sparse_offense_pooling_strategy: {config.sparse_offense_pooling_strategy}")

    work = frame[["state_fips", "jurisdiction_type"]].copy()
    work["counts"] = pd.to_numeric(counts, errors="coerce").fillna(0.0)
    work["bucket_population"] = pd.to_numeric(pop, errors="coerce").fillna(0.0)
    work["pool_pop_band"] = work["bucket_population"].map(_sparse_pooling_pop_band)

    prior_base = work.loc[observed_only_mask & work["bucket_population"].gt(0)].copy()
    if prior_base.empty:
        return raw_rate, {
            "strategy": "empirical_bayes_group_rate",
            "applied_rows": 0,
            "prior_source_counts": {},
        }

    def _summary(group_cols: list[str], source_name: str, *, min_rows: int) -> pd.DataFrame:
        if group_cols:
            grouped = (
                prior_base.groupby(group_cols, dropna=False)
                .agg(
                    prior_row_count=("counts", "size"),
                    prior_total_count=("counts", "sum"),
                    prior_total_population=("bucket_population", "sum"),
                    prior_equiv_population=("bucket_population", "median"),
                )
                .reset_index()
            )
        else:
            grouped = pd.DataFrame(
                [
                    {
                        "prior_row_count": int(len(prior_base)),
                        "prior_total_count": float(pd.to_numeric(prior_base["counts"], errors="coerce").fillna(0.0).sum()),
                        "prior_total_population": float(
                            pd.to_numeric(prior_base["bucket_population"], errors="coerce").fillna(0.0).sum()
                        ),
                        "prior_equiv_population": float(
                            pd.to_numeric(prior_base["bucket_population"], errors="coerce").fillna(0.0).median()
                        ),
                    }
                ]
            )
        grouped = grouped[grouped["prior_row_count"].ge(int(min_rows))].copy()
        grouped = grouped[pd.to_numeric(grouped["prior_total_population"], errors="coerce").fillna(0.0).gt(0)].copy()
        if grouped.empty:
            return grouped
        grouped["prior_rate_per_person"] = (
            pd.to_numeric(grouped["prior_total_count"], errors="coerce").fillna(0.0)
            / pd.to_numeric(grouped["prior_total_population"], errors="coerce").fillna(np.nan)
        ).clip(lower=0.0)
        grouped["prior_equiv_population"] = (
            pd.to_numeric(grouped["prior_equiv_population"], errors="coerce")
            .fillna(pd.to_numeric(grouped["prior_total_population"], errors="coerce").fillna(0.0))
            .clip(lower=10_000.0, upper=100_000.0)
        )
        grouped["prior_source"] = source_name
        return grouped[group_cols + ["prior_rate_per_person", "prior_equiv_population", "prior_source"]]

    prior_state_pop = _summary(
        ["state_fips", "jurisdiction_type", "pool_pop_band"],
        "state_jurisdiction_type_pop_band",
        min_rows=5,
    )
    prior_state = _summary(
        ["state_fips", "jurisdiction_type"],
        "state_jurisdiction_type",
        min_rows=10,
    )
    prior_jurisdiction_type = _summary(
        ["jurisdiction_type"],
        "national_jurisdiction_type",
        min_rows=20,
    )
    prior_national = _summary([], "national", min_rows=20)

    enriched = work.copy()
    enriched["prior_rate_per_person"] = np.nan
    enriched["prior_equiv_population"] = np.nan
    enriched["prior_source"] = pd.Series(pd.NA, index=enriched.index, dtype="string")

    def _fill_prior(prior_frame: pd.DataFrame, merge_cols: list[str]) -> None:
        nonlocal enriched
        if prior_frame.empty:
            return
        merged = enriched.merge(prior_frame, on=merge_cols, how="left", suffixes=("", "_candidate"))
        missing = merged["prior_rate_per_person"].isna() & pd.to_numeric(
            merged["prior_rate_per_person_candidate"], errors="coerce"
        ).notna()
        if missing.any():
            merged.loc[missing, "prior_rate_per_person"] = pd.to_numeric(
                merged.loc[missing, "prior_rate_per_person_candidate"], errors="coerce"
            )
            merged.loc[missing, "prior_equiv_population"] = pd.to_numeric(
                merged.loc[missing, "prior_equiv_population_candidate"], errors="coerce"
            )
            merged.loc[missing, "prior_source"] = merged.loc[missing, "prior_source_candidate"].astype("string")
        enriched = merged.drop(
            columns=[
                "prior_rate_per_person_candidate",
                "prior_equiv_population_candidate",
                "prior_source_candidate",
            ],
            errors="ignore",
        )

    _fill_prior(prior_state_pop, ["state_fips", "jurisdiction_type", "pool_pop_band"])
    _fill_prior(prior_state, ["state_fips", "jurisdiction_type"])
    _fill_prior(prior_jurisdiction_type, ["jurisdiction_type"])
    if not prior_national.empty:
        national_row = prior_national.iloc[0]
        missing = enriched["prior_rate_per_person"].isna()
        if missing.any():
            enriched.loc[missing, "prior_rate_per_person"] = float(
                pd.to_numeric(national_row["prior_rate_per_person"], errors="coerce")
            )
            enriched.loc[missing, "prior_equiv_population"] = float(
                pd.to_numeric(national_row["prior_equiv_population"], errors="coerce")
            )
            enriched.loc[missing, "prior_source"] = str(national_row["prior_source"])

    prior_rate = pd.to_numeric(enriched["prior_rate_per_person"], errors="coerce").fillna(np.nan).to_numpy(dtype=float)
    prior_pop = pd.to_numeric(enriched["prior_equiv_population"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_counts = pd.to_numeric(counts, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_pop = pd.to_numeric(pop, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pooled_rate = raw_rate.copy()
    apply_mask = np.isfinite(prior_rate) & np.isfinite(raw_pop) & (raw_pop > 0) & np.isfinite(prior_pop) & (prior_pop > 0)
    pooled_rate[apply_mask] = (
        raw_counts[apply_mask] + prior_rate[apply_mask] * prior_pop[apply_mask]
    ) / (raw_pop[apply_mask] + prior_pop[apply_mask]) * 1e5
    return pooled_rate, {
        "strategy": pooling_strategy,
        "applied_rows": int(apply_mask.sum()),
        "prior_source_counts": {
            str(key): int(value)
            for key, value in enriched.loc[apply_mask, "prior_source"].astype("string").value_counts(dropna=False).to_dict().items()
        },
    }


def _encode_state(df: pd.DataFrame, *, cols: list[str] | None = None) -> pd.DataFrame:
    state_dummies = pd.get_dummies(df["state_fips"].astype(str), prefix="state", dtype=float)
    if cols is None:
        return state_dummies
    for col in cols:
        if col not in state_dummies.columns:
            state_dummies[col] = 0.0
    return state_dummies[cols]


def _model_feature_columns(
    config: ModelSurfaceConfig,
    *,
    feature_cols: list[str],
    state_cols: list[str],
) -> list[str]:
    if config.model_family == "monotone_gam":
        return monotone_gam_feature_columns(feature_cols)
    return list(feature_cols) + list(state_cols)


def _build_design_matrices(
    *,
    bg: pd.DataFrame,
    training: pd.DataFrame,
    feature_cols: list[str],
    config: ModelSurfaceConfig,
    excluded_policy_columns: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base_x = training[feature_cols].replace([np.inf, -np.inf], np.nan)
    bg_x = bg[feature_cols].replace([np.inf, -np.inf], np.nan)

    effective_use_state_fixed_effects = bool(config.use_state_fixed_effects) and config.model_family != "monotone_gam"
    train_state_cols: list[str] = []
    if effective_use_state_fixed_effects:
        state_train = _encode_state(training)
        train_state_cols = [
            col for col in state_train.columns.tolist()
            if str(col) not in excluded_policy_columns
        ]
        if train_state_cols:
            base_x = pd.concat(
                [base_x.reset_index(drop=True), state_train[train_state_cols].reset_index(drop=True)],
                axis=1,
            )
            bg_x = pd.concat(
                [
                    bg_x.reset_index(drop=True),
                    _encode_state(bg, cols=train_state_cols).reset_index(drop=True),
                ],
                axis=1,
            )

    return base_x, bg_x, train_state_cols


def _build_model(
    config: ModelSurfaceConfig,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
):
    if config.model_family == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(config.ridge_alpha), random_state=0)),
            ]
        )
    if config.model_family == "elastic_net":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "elastic_net",
                    ElasticNet(
                        alpha=float(config.elastic_net_alpha),
                        l1_ratio=float(config.elastic_net_l1_ratio),
                        max_iter=int(config.elastic_net_max_iter),
                        random_state=0,
                    ),
                ),
            ]
        )
    if config.model_family == "monotone_gam":
        if not feature_names:
            raise ValueError("monotone_gam requires feature_names")
        return MonotoneGAMRegressor(
            feature_names=list(feature_names),
            lam=float(config.monotone_gam_lam),
            n_splines=int(config.monotone_gam_n_splines),
            max_iter=int(config.monotone_gam_max_iter),
        )
    if config.model_family == "hist_gbm":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(config.hist_learning_rate),
            max_depth=int(config.hist_max_depth),
            max_iter=int(config.hist_max_iter),
            min_samples_leaf=int(config.hist_min_samples_leaf),
            l2_regularization=float(config.hist_l2_regularization),
            random_state=0,
        )
    raise ValueError(f"Unsupported model_family: {config.model_family}")


def build_jurisdiction_model_training_frame(
    *,
    bg: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: list[str],
    county_to_cbsa: pd.DataFrame | None = None,
    aggregation_weight_col: str = "population_for_weights",
) -> pd.DataFrame:
    main_types = {"municipal", "state_nonmunicipal_remainder"}
    main_crosswalk = normalize_block_group_allocation_shares(
        bg_crosswalk[bg_crosswalk["jurisdiction_type"].isin(main_types)].copy()
    )
    merged = main_crosswalk.merge(
        bg,
        left_on=["block_group_geoid", "state_fips"],
        right_on=["bg_id", "state_fips"],
        how="left",
    )
    merged["aggregation_weight"] = (
        pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0)
        * pd.to_numeric(merged[aggregation_weight_col], errors="coerce").fillna(0.0)
    )
    fallback = pd.to_numeric(merged["allocation_share"], errors="coerce").fillna(0.0)
    merged.loc[merged["aggregation_weight"].le(0), "aggregation_weight"] = fallback.loc[merged["aggregation_weight"].le(0)]
    merged["county_fips"] = merged["bg_id"].astype("string").str.zfill(12).str.slice(0, 5)
    if county_to_cbsa is not None and not county_to_cbsa.empty:
        merged = merged.merge(county_to_cbsa, on="county_fips", how="left")

    features = _weighted_mean_by_group(
        merged,
        group_col="jurisdiction_id",
        weight_col="aggregation_weight",
        value_cols=feature_cols,
    )
    pops = (
        merged.groupby("jurisdiction_id", dropna=False)["aggregation_weight"]
        .sum()
        .rename("bucket_population")
        .reset_index()
    )
    id_cols = controls[controls["jurisdiction_type"].isin(main_types)][
        ["jurisdiction_id", "jurisdiction_name", "state_fips", "jurisdiction_type"]
    ].drop_duplicates()
    training = features.merge(pops, on="jurisdiction_id", how="left").merge(id_cols, on="jurisdiction_id", how="left")
    dominant_county = _dominant_weighted_label_frame(
        merged,
        label_col="county_fips",
        output_col="dominant_county_fips",
        share_col="dominant_county_share",
    )
    training = training.merge(dominant_county, on="jurisdiction_id", how="left")
    if county_to_cbsa is not None and not county_to_cbsa.empty and "cbsa_code" in merged.columns:
        dominant_cbsa = _dominant_weighted_label_frame(
            merged,
            label_col="cbsa_code",
            output_col="dominant_cbsa_code",
            share_col="dominant_cbsa_share",
        )
        training = training.merge(dominant_cbsa, on="jurisdiction_id", how="left")
    if "dominant_cbsa_code" not in training.columns:
        training["dominant_cbsa_code"] = pd.Series(pd.NA, index=training.index, dtype="string")
    if "dominant_cbsa_share" not in training.columns:
        training["dominant_cbsa_share"] = np.nan
    return training


def _prepare_model_surface_context(
    *,
    paths: RepoPaths,
    config: ModelSurfaceConfig,
    extra_bg_feature_paths: list[Path] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    controls = pd.read_parquet(paths.state_dir / "controls" / f"jurisdiction_controls_{int(config.year)}.parquet")
    bg_crosswalk = pd.read_parquet(paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet")
    bg_crosswalk["state_fips"] = bg_crosswalk["state_fips"].astype("string").str.zfill(2)
    bg = build_bg_feature_frame(paths=paths, year=int(config.year))
    base_bg_cols = set(bg.columns)
    bg = merge_extra_bg_features(bg, extra_feature_paths=extra_bg_feature_paths)
    extra_feature_cols = sorted(set(bg.columns) - base_bg_cols)
    scoped_states = set(bg_crosswalk["state_fips"].dropna().astype("string").str.zfill(2).unique().tolist())
    scoped_bg_ids = set(bg_crosswalk["block_group_geoid"].dropna().astype("string").str.zfill(12).unique().tolist())
    bg = bg[
        bg["state_fips"].isin(scoped_states)
        & bg["bg_id"].isin(scoped_bg_ids)
    ].copy()
    feature_cols = _build_feature_selection(
        bg,
        exclude_feature_groups=tuple(config.exclude_feature_groups),
    )
    excluded_policy_columns = _load_feature_policy_excluded_columns(config)
    feature_cols = _apply_feature_policy_exclusions(feature_cols, excluded_policy_columns)
    if str(config.prior_anchor).strip().lower() == "offense_denominator":
        bg = add_offense_denominators(
            bg,
            paths=paths,
            year=int(config.year),
            burglary_commercial_weight=config.burglary_commercial_weight,
            apply_landscan_day_floor=False,
        )
    county_to_cbsa = _load_county_to_cbsa_delineation(paths)

    training = build_jurisdiction_model_training_frame(
        bg=bg,
        bg_crosswalk=bg_crosswalk,
        controls=controls,
        feature_cols=feature_cols,
        county_to_cbsa=county_to_cbsa,
    )
    base_x, bg_x, train_state_cols = _build_design_matrices(
        bg=bg,
        training=training,
        feature_cols=feature_cols,
        config=config,
        excluded_policy_columns=excluded_policy_columns,
    )

    return controls, bg_crosswalk, bg, training, feature_cols, train_state_cols, base_x, bg_x, extra_feature_cols


def _build_offense_training_state(
    *,
    training: pd.DataFrame,
    controls: pd.DataFrame,
    base_x: pd.DataFrame,
    config: ModelSurfaceConfig,
    offense: str,
    feature_cols: list[str],
    train_state_cols: list[str],
) -> dict[str, object]:
    target = controls[
        (controls["jurisdiction_type"].isin(["municipal", "state_nonmunicipal_remainder"]))
        & (controls["offense"].eq(offense))
    ][
        [
            "jurisdiction_id",
            "adjusted_count_ags_core",
            "needs_current_year_fill",
            "estimated_from_panel",
            "estimate_confidence",
            "quality_tier_preferred",
            "needs_partial_reporting_uplift",
            "preferred_source",
        ]
    ].copy()
    dup_mask = target.duplicated(subset=["jurisdiction_id"], keep=False)
    if dup_mask.any():
        sample_ids = (
            target.loc[dup_mask, "jurisdiction_id"]
            .astype("string")
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Duplicate jurisdiction controls detected for model training "
            f"({offense}; duplicate_ids_sample={sample_ids})"
        )
    target["adjusted_count_ags_core"] = pd.to_numeric(target["adjusted_count_ags_core"], errors="coerce").fillna(0.0)
    target["needs_current_year_fill"] = target["needs_current_year_fill"].eq(True)
    frame = training.merge(target, on="jurisdiction_id", how="left")
    counts = pd.to_numeric(frame["adjusted_count_ags_core"], errors="coerce").fillna(0.0)
    pop = pd.to_numeric(frame["bucket_population"], errors="coerce").fillna(0.0)
    raw_rate = np.where(pop > 0, counts / pop * 1e5, 0.0)
    needs_fill = frame["needs_current_year_fill"].eq(True)
    estimated_from_panel = frame["estimated_from_panel"].eq(True)
    estimate_confidence = frame["estimate_confidence"].astype("string").str.lower()
    quality_tier = frame["quality_tier_preferred"].astype("string").str.lower()
    needs_partial_uplift = frame["needs_partial_reporting_uplift"].eq(True)
    observed_only_mask = (~needs_fill) & (~estimated_from_panel)
    train_mask = (
        pd.to_numeric(frame["bucket_population"], errors="coerce").fillna(0.0) >= int(config.min_training_population)
    ) & (~needs_fill)
    if config.exclude_estimated_from_panel_from_training:
        train_mask = train_mask & (~estimated_from_panel)
    if config.high_confidence_training_only:
        train_mask = (
            train_mask
            & estimate_confidence.eq("high")
            & quality_tier.eq("high")
            & (~needs_partial_uplift)
        )
    train_mask_np = train_mask.fillna(False).to_numpy(dtype=bool)
    if int(train_mask.sum()) < 100:
        raise RuntimeError(f"Too few municipal jurisdictions above training threshold for {offense}: {int(train_mask.sum())}")

    x_train = base_x.loc[train_mask].copy()
    model_feature_cols = _model_feature_columns(
        config,
        feature_cols=feature_cols,
        state_cols=train_state_cols,
    )
    state_fips_train = frame.loc[train_mask, "state_fips"].astype(str).to_numpy()
    bucket_population_train = pop.loc[train_mask].astype(float)
    large_city_group_labels = training.loc[train_mask, "jurisdiction_id"].astype("string").astype(str).to_numpy()
    large_city_eligible = (
        training.loc[
            train_mask
            & training["jurisdiction_type"].eq("municipal")
            & pd.to_numeric(training["bucket_population"], errors="coerce").fillna(0.0).ge(
                float(LEAVE_LARGE_CITY_OUT_MIN_POPULATION)
            ),
            "jurisdiction_id",
        ]
        .astype("string")
        .dropna()
        .astype(str)
        .unique()
    )
    cbsa_group_labels = (
        training.loc[train_mask, "dominant_cbsa_code"].astype("string").fillna("unknown").astype(str).to_numpy()
    )
    cbsa_eligible = _select_cbsa_holdout_groups(
        training=training,
        train_mask=train_mask,
    )
    leave_state_holdout_count = int(len(pd.unique(state_fips_train))) if len(pd.unique(state_fips_train)) >= 2 else 0
    cv_fold_count = int(min(5, int(train_mask.sum()))) if int(train_mask.sum()) >= 2 else 0
    planned_fit_count = int(
        1
        + cv_fold_count
        + leave_state_holdout_count
        + len(large_city_eligible)
        + len(cbsa_eligible)
    )
    return {
        "frame": frame,
        "counts": counts,
        "pop": pop,
        "raw_rate": raw_rate,
        "needs_fill": needs_fill,
        "estimated_from_panel": estimated_from_panel,
        "estimate_confidence": estimate_confidence,
        "quality_tier": quality_tier,
        "needs_partial_uplift": needs_partial_uplift,
        "observed_only_mask": observed_only_mask,
        "train_mask": train_mask,
        "train_mask_np": train_mask_np,
        "x_train": x_train,
        "model_feature_cols": model_feature_cols,
        "state_fips_train": state_fips_train,
        "bucket_population_train": bucket_population_train,
        "large_city_group_labels": large_city_group_labels,
        "large_city_eligible": large_city_eligible,
        "cbsa_group_labels": cbsa_group_labels,
        "cbsa_eligible": cbsa_eligible,
        "leave_state_holdout_count": leave_state_holdout_count,
        "cv_fold_count": cv_fold_count,
        "planned_fit_count": planned_fit_count,
    }


def build_model_workload_plan(
    *,
    paths: RepoPaths,
    config: ModelSurfaceConfig = ModelSurfaceConfig(),
    extra_bg_feature_paths: list[Path] | None = None,
) -> pd.DataFrame:
    controls, bg_crosswalk, bg, training, feature_cols, train_state_cols, base_x, _, _ = _prepare_model_surface_context(
        paths=paths,
        config=config,
        extra_bg_feature_paths=extra_bg_feature_paths,
    )
    county_to_cbsa = _load_county_to_cbsa_delineation(paths)
    excluded_policy_columns = _load_feature_policy_excluded_columns(config)
    denominator_anchor = str(config.prior_anchor).strip().lower() == "offense_denominator"
    burglary_duplicate_columns = burglary_exposure_duplicate_columns(
        config.feature_policy_path
    )
    rows: list[dict[str, object]] = []
    for offense in OFFENSES_7:
        offense_feature_cols = _feature_cols_for_offense(
            feature_cols,
            offense=str(offense),
            burglary_exposure_duplicate_columns=burglary_duplicate_columns,
            denominator_anchor=denominator_anchor,
        )
        offense_training = training
        offense_base_x = base_x
        offense_train_state_cols = train_state_cols
        if denominator_anchor:
            offense_training = build_jurisdiction_model_training_frame(
                bg=bg,
                bg_crosswalk=bg_crosswalk,
                controls=controls,
                feature_cols=offense_feature_cols,
                county_to_cbsa=county_to_cbsa,
                aggregation_weight_col=offense_denominator_column(offense),
            )
            offense_base_x, _, offense_train_state_cols = _build_design_matrices(
                bg=bg,
                training=offense_training,
                feature_cols=offense_feature_cols,
                config=config,
                excluded_policy_columns=excluded_policy_columns,
            )
        offense_state = _build_offense_training_state(
            training=offense_training,
            controls=controls,
            base_x=offense_base_x,
            config=config,
            offense=offense,
            feature_cols=offense_feature_cols,
            train_state_cols=offense_train_state_cols,
        )
        rows.append(
            {
                "offense": str(offense),
                "training_rows": int(pd.to_numeric(offense_state["train_mask"], errors="coerce").sum()),
                "feature_count": int(len(offense_state["model_feature_cols"])),
                "cv_fold_count": int(offense_state["cv_fold_count"]),
                "leave_state_out_holdout_group_count": int(offense_state["leave_state_holdout_count"]),
                "leave_large_city_out_holdout_group_count": int(len(offense_state["large_city_eligible"])),
                "leave_cbsa_out_holdout_group_count": int(len(offense_state["cbsa_eligible"])),
                "planned_fit_count": int(offense_state["planned_fit_count"]),
            }
        )
    return pd.DataFrame(rows).sort_values("offense", kind="mergesort").reset_index(drop=True)


def build_model_surface(
    *,
    paths: RepoPaths,
    config: ModelSurfaceConfig = ModelSurfaceConfig(),
    extra_bg_feature_paths: list[Path] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    controls, bg_crosswalk, bg, training, feature_cols, train_state_cols, base_x, bg_x, extra_feature_cols = _prepare_model_surface_context(
        paths=paths,
        config=config,
        extra_bg_feature_paths=extra_bg_feature_paths,
    )
    effective_use_state_fixed_effects = (
        bool(config.use_state_fixed_effects)
        and config.model_family != "monotone_gam"
        and bool(train_state_cols)
    )
    county_to_cbsa = _load_county_to_cbsa_delineation(paths)
    excluded_policy_columns = _load_feature_policy_excluded_columns(config)
    denominator_anchor = str(config.prior_anchor).strip().lower() == "offense_denominator"
    burglary_duplicate_columns = burglary_exposure_duplicate_columns(
        config.feature_policy_path
    )

    diag_rows: list[dict[str, object]] = []
    prior_frames: list[pd.DataFrame] = []
    model_meta_rows: list[dict[str, object]] = []
    for offense in OFFENSES_7:
        offense_started_at = time.perf_counter()
        offense_feature_cols = _feature_cols_for_offense(
            feature_cols,
            offense=str(offense),
            burglary_exposure_duplicate_columns=burglary_duplicate_columns,
            denominator_anchor=denominator_anchor,
        )
        offense_training = training
        offense_base_x = base_x
        offense_bg_x = bg_x
        offense_train_state_cols = train_state_cols
        if denominator_anchor:
            offense_training = build_jurisdiction_model_training_frame(
                bg=bg,
                bg_crosswalk=bg_crosswalk,
                controls=controls,
                feature_cols=offense_feature_cols,
                county_to_cbsa=county_to_cbsa,
                aggregation_weight_col=offense_denominator_column(offense),
            )
            offense_base_x, offense_bg_x, offense_train_state_cols = _build_design_matrices(
                bg=bg,
                training=offense_training,
                feature_cols=offense_feature_cols,
                config=config,
                excluded_policy_columns=excluded_policy_columns,
            )
        offense_state = _build_offense_training_state(
            training=offense_training,
            controls=controls,
            base_x=offense_base_x,
            config=config,
            offense=offense,
            feature_cols=offense_feature_cols,
            train_state_cols=offense_train_state_cols,
        )
        frame = offense_state["frame"]
        counts = offense_state["counts"]
        pop = offense_state["pop"]
        raw_rate = offense_state["raw_rate"]
        needs_fill = offense_state["needs_fill"]
        estimated_from_panel = offense_state["estimated_from_panel"]
        estimate_confidence = offense_state["estimate_confidence"]
        quality_tier = offense_state["quality_tier"]
        needs_partial_uplift = offense_state["needs_partial_uplift"]
        observed_only_mask = offense_state["observed_only_mask"]
        train_mask = offense_state["train_mask"]
        train_mask_np = offense_state["train_mask_np"]
        x_train = offense_state["x_train"]
        model_feature_cols = offense_state["model_feature_cols"]
        state_fips_train = offense_state["state_fips_train"]
        bucket_population_train = offense_state["bucket_population_train"]
        large_city_group_labels = offense_state["large_city_group_labels"]
        large_city_eligible = offense_state["large_city_eligible"]
        cbsa_group_labels = offense_state["cbsa_group_labels"]
        cbsa_eligible = offense_state["cbsa_eligible"]
        leave_state_holdout_count = int(offense_state["leave_state_holdout_count"])
        cv_fold_count = int(offense_state["cv_fold_count"])
        planned_fit_count = int(offense_state["planned_fit_count"])
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "offense_start",
                    "offense": str(offense),
                    "training_rows": int(train_mask.sum()),
                    "feature_count": int(len(model_feature_cols)),
                    "cv_fold_count": cv_fold_count,
                    "leave_state_out_holdout_group_count": leave_state_holdout_count,
                    "leave_large_city_out_holdout_group_count": int(len(large_city_eligible)),
                    "leave_cbsa_out_holdout_group_count": int(len(cbsa_eligible)),
                    "planned_fit_count": planned_fit_count,
                }
            )
        fit_rate, pooling_meta = _build_sparse_offense_pooling_target(
            frame=frame,
            counts=counts,
            pop=pop,
            offense=offense,
            config=config,
            observed_only_mask=observed_only_mask,
        )
        y_fit = np.log1p(fit_rate)
        y_eval = np.log1p(raw_rate)
        model = _build_model(config, feature_names=model_feature_cols)
        model.fit(x_train, y_fit[train_mask_np])
        train_pred = np.asarray(model.predict(x_train), dtype=float)
        if config.compute_diagnostics:
            cv = cross_validated_diagnostics(
                x=x_train,
                y_log_rate=y_fit[train_mask_np],
                y_eval_log_rate=y_eval[train_mask_np],
                state_fips=state_fips_train,
                bucket_population=bucket_population_train,
                model_factory=lambda: _build_model(config, feature_names=model_feature_cols),
            )
            leave_state_out_cv = cross_validated_diagnostics(
                x=x_train,
                y_log_rate=y_fit[train_mask_np],
                y_eval_log_rate=y_eval[train_mask_np],
                state_fips=state_fips_train,
                bucket_population=bucket_population_train,
                group_labels=state_fips_train,
                split_mode="leave_group_out",
                model_factory=lambda: _build_model(config, feature_names=model_feature_cols),
            )
            leave_large_city_out_cv = cross_validated_diagnostics(
                x=x_train,
                y_log_rate=y_fit[train_mask_np],
                y_eval_log_rate=y_eval[train_mask_np],
                state_fips=state_fips_train,
                bucket_population=bucket_population_train,
                group_labels=large_city_group_labels,
                eligible_holdout_groups=large_city_eligible,
                split_mode="leave_selected_groups_out",
                model_factory=lambda: _build_model(config, feature_names=model_feature_cols),
            )
            leave_cbsa_out_cv = cross_validated_diagnostics(
                x=x_train,
                y_log_rate=y_fit[train_mask_np],
                y_eval_log_rate=y_eval[train_mask_np],
                state_fips=state_fips_train,
                bucket_population=bucket_population_train,
                group_labels=cbsa_group_labels,
                eligible_holdout_groups=cbsa_eligible,
                split_mode="leave_selected_groups_out",
                model_factory=lambda: _build_model(config, feature_names=model_feature_cols),
            )
            diag_rows.append(
                {
                    "offense": offense,
                    "training_rows": int(train_mask.sum()),
                    "feature_count": int(len(model_feature_cols)),
                    "model_family": str(config.model_family),
                    "prior_anchor": str(config.prior_anchor),
                    "feature_policy_path": str(config.feature_policy_path) if config.feature_policy_path is not None else None,
                    "exclude_feature_policy_classes": json.dumps(
                        sorted({str(group) for group in config.exclude_feature_policy_classes})
                    ),
                    "state_fixed_effects": bool(effective_use_state_fixed_effects),
                    "high_confidence_training_only": bool(config.high_confidence_training_only),
                    "sparse_offense_pooling_strategy": str(config.sparse_offense_pooling_strategy),
                    "exclude_feature_groups": json.dumps(sorted({str(group) for group in config.exclude_feature_groups})),
                    "excluded_current_year_fill_rows": int(frame["needs_current_year_fill"].fillna(False).astype(bool).sum()),
                    "excluded_estimated_from_panel_rows": int(estimated_from_panel.fillna(False).astype(bool).sum()),
                    "train_r2_log_rate": float(r2_score(y_eval[train_mask_np], train_pred)),
                    "sparse_pooling_applied_training_rows": int(pooling_meta.get("applied_rows", 0)),
                    "sparse_pooling_prior_source_counts_json": json.dumps(pooling_meta.get("prior_source_counts", {})),
                    "cv_r2_log_rate": float(cv.cv_r2_log_rate),
                    "cv_r2_rate": float(cv.cv_r2_rate),
                    "cv_rmse_log_rate": float(cv.cv_rmse_log_rate),
                    "cv_fold_count": int(cv_fold_count),
                    "leave_state_out_holdout_group_count": int(leave_state_holdout_count),
                    "leave_state_out_cv_r2_log_rate": float(leave_state_out_cv.cv_r2_log_rate),
                    "leave_state_out_cv_r2_rate": float(leave_state_out_cv.cv_r2_rate),
                    "leave_state_out_cv_rmse_log_rate": float(leave_state_out_cv.cv_rmse_log_rate),
                    "leave_large_city_out_cv_r2_log_rate": float(leave_large_city_out_cv.cv_r2_log_rate),
                    "leave_large_city_out_cv_r2_rate": float(leave_large_city_out_cv.cv_r2_rate),
                    "leave_large_city_out_cv_rmse_log_rate": float(leave_large_city_out_cv.cv_rmse_log_rate),
                    "leave_large_city_out_holdout_group_count": int(len(large_city_eligible)),
                    "leave_cbsa_out_cv_r2_log_rate": float(leave_cbsa_out_cv.cv_r2_log_rate),
                    "leave_cbsa_out_cv_r2_rate": float(leave_cbsa_out_cv.cv_r2_rate),
                    "leave_cbsa_out_cv_rmse_log_rate": float(leave_cbsa_out_cv.cv_rmse_log_rate),
                    "leave_cbsa_out_holdout_group_count": int(len(cbsa_eligible)),
                    "planned_fit_count": int(planned_fit_count),
                    "calibration_json": json.dumps(cv.calibration),
                    "residual_by_state_json": json.dumps(cv.residual_by_state),
                    "residual_by_pop_band_json": json.dumps(cv.residual_by_pop_band),
                    "leave_state_out_calibration_json": json.dumps(leave_state_out_cv.calibration),
                    "leave_state_out_residual_by_state_json": json.dumps(leave_state_out_cv.residual_by_state),
                    "leave_state_out_residual_by_pop_band_json": json.dumps(leave_state_out_cv.residual_by_pop_band),
                    "leave_large_city_out_calibration_json": json.dumps(leave_large_city_out_cv.calibration),
                    "leave_large_city_out_residual_by_state_json": json.dumps(leave_large_city_out_cv.residual_by_state),
                    "leave_large_city_out_residual_by_pop_band_json": json.dumps(leave_large_city_out_cv.residual_by_pop_band),
                    "leave_cbsa_out_calibration_json": json.dumps(leave_cbsa_out_cv.calibration),
                    "leave_cbsa_out_residual_by_state_json": json.dumps(leave_cbsa_out_cv.residual_by_state),
                    "leave_cbsa_out_residual_by_pop_band_json": json.dumps(leave_cbsa_out_cv.residual_by_pop_band),
                }
            )

        pred_log_rate = np.asarray(model.predict(offense_bg_x), dtype=float)
        pred_log_rate = np.clip(pred_log_rate, 0.0, 12.0)
        pred_rate = np.expm1(pred_log_rate)
        pred_rate = np.clip(pred_rate, 0.0, None)
        exposure_col = offense_denominator_column(offense) if denominator_anchor else "population_for_weights"
        exposure = pd.to_numeric(bg[exposure_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred_count = pred_rate / 1e5 * exposure
        prior = bg[["bg_id", "tract_id", "state_fips"]].copy()
        prior["offense"] = offense
        prior["bg_weight"] = np.clip(pred_count, 0.0, None)
        prior_frames.append(prior)
        if progress_callback is not None:
            progress_payload: dict[str, object] = {
                "event": "offense_complete",
                "offense": str(offense),
                "elapsed_sec": float(time.perf_counter() - offense_started_at),
                    "training_rows": int(train_mask.sum()),
                    "planned_fit_count": int(planned_fit_count),
                    "bg_weight_sum": float(np.clip(pred_count, 0.0, None).sum()),
                    "prior_anchor": str(config.prior_anchor),
                    "denominator_column": str(exposure_col),
                }
            if config.compute_diagnostics and diag_rows:
                last_row = diag_rows[-1]
                progress_payload.update(
                    {
                        "cv_r2_log_rate": float(last_row["cv_r2_log_rate"]),
                        "leave_state_out_cv_r2_log_rate": float(last_row["leave_state_out_cv_r2_log_rate"]),
                        "leave_large_city_out_cv_r2_log_rate": float(last_row["leave_large_city_out_cv_r2_log_rate"]),
                        "leave_cbsa_out_cv_r2_log_rate": float(last_row["leave_cbsa_out_cv_r2_log_rate"]),
                    }
                )
            progress_callback(progress_payload)
        for col in model_feature_cols:
            model_meta_rows.append(
                {
                    "offense": str(offense),
                    "feature_column": str(col),
                    "feature_group": _feature_group_for_column(col),
                    "feature_source": "extra_bg_feature" if col in set(extra_feature_cols) else "base_covariate",
                    "prior_anchor": str(config.prior_anchor),
                    "feature_policy_path": str(config.feature_policy_path) if config.feature_policy_path is not None else None,
                    "excluded_by_feature_policy": str(col) in excluded_policy_columns,
                    "excluded_for_burglary_exposure_duplicate": (
                        str(offense) == "burglary" and str(col) in burglary_duplicate_columns
                    ),
                    "excluded_for_burglary_raw_commercial_exposure_duplicate": (
                        str(offense) == "burglary" and str(col) in burglary_duplicate_columns
                    ),
                }
            )
        for col in train_state_cols:
            model_meta_rows.append(
                {
                    "offense": str(offense),
                    "feature_column": str(col),
                    "feature_group": "state_fixed_effect",
                    "feature_source": "state_fixed_effect",
                    "prior_anchor": str(config.prior_anchor),
                    "feature_policy_path": str(config.feature_policy_path) if config.feature_policy_path is not None else None,
                    "excluded_by_feature_policy": False,
                    "excluded_for_burglary_exposure_duplicate": False,
                    "excluded_for_burglary_raw_commercial_exposure_duplicate": False,
                }
            )

    diagnostics = pd.DataFrame(diag_rows)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values("offense", kind="mergesort").reset_index(drop=True)
    bg_prior_long = pd.concat(prior_frames, ignore_index=True)
    model_meta = pd.DataFrame(model_meta_rows)
    return bg_prior_long, diagnostics, model_meta
