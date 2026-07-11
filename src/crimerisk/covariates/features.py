from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import re

from crimerisk.covariates.cpi import CpiSeries, inflation_factor, load_cpi_u_series
from crimerisk.covariates.lodes_flow import (
    LODES_FLOW_RAW_COLS,
    WAC_KEEP_COLS,
    apply_lodes_flow_daytime_features,
    compute_lodes_flow_frame_by_tract,
)
from crimerisk.covariates.population import (
    CountyPopulationEstimates,
    align_block_group_population_to_county,
    align_tract_population_to_county,
    load_county_population_estimates,
)

LODES_COMPONENT_RE = re.compile(r"^(CA\d{2}|CE\d{2}|CNS\d{2}|CD\d{2}|CS\d{2}|CFA\d{2}|CFS\d{2})$")


@dataclass(frozen=True)
class CovariateBuildConfig:
    acs_money_base_year: int = 2023
    estimate_year: int = 2024


def _clean_acs_medians(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sentinel_cols = [
        "median_household_income",
        "per_capita_income",
        "median_rent",
        "median_home_value",
        "median_age",
        "median_age_male",
        "median_age_female",
    ]
    for col in sentinel_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] < 0, col] = np.nan
    return out


def _inflate_money_columns(
    df: pd.DataFrame,
    *,
    cpi_cache_csv: Path,
    cfg: CovariateBuildConfig,
) -> pd.DataFrame:
    out = df.copy()
    cpi = load_cpi_u_series(cache_csv_path=cpi_cache_csv)
    factor = inflation_factor(cpi=cpi, from_year=cfg.acs_money_base_year, to_year=cfg.estimate_year)
    for col in ["median_household_income", "per_capita_income", "median_rent", "median_home_value"]:
        if col in out.columns:
            out[f"{col}_{cfg.estimate_year}"] = (pd.to_numeric(out[col], errors="coerce") * factor).round()
    return out


def _apply_common_derived_covariates(df: pd.DataFrame) -> pd.DataFrame:
    acs = df.copy()
    derived: dict[str, pd.Series] = {}
    numeric_cache: dict[str, pd.Series] = {}

    def _has(col: str) -> bool:
        return col in derived or col in acs.columns

    def _has_all(cols: list[str] | tuple[str, ...] | set[str]) -> bool:
        return all(_has(col) for col in cols)

    def _series(col: str) -> pd.Series | None:
        if col in derived:
            return derived[col]
        if col in numeric_cache:
            return numeric_cache[col]
        if col not in acs.columns:
            return None
        numeric_cache[col] = pd.to_numeric(acs[col], errors="coerce")
        return numeric_cache[col]

    def _clip_nonnegative(series: pd.Series) -> pd.Series:
        return pd.Series(series, index=acs.index).clip(lower=0.0)

    def _bounded_ratio(numer: pd.Series, denom: pd.Series, *, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
        numer_vals = pd.to_numeric(numer, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        denom_vals = pd.to_numeric(denom, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        values = np.full(len(acs.index), np.nan, dtype=float)
        valid = np.isfinite(numer_vals) & np.isfinite(denom_vals) & (denom_vals > 0)
        np.divide(numer_vals, denom_vals, out=values, where=valid)
        return pd.Series(values, index=acs.index, dtype=float).clip(lower=lower, upper=upper)

    def _unbounded_ratio(numer: pd.Series, denom: pd.Series, *, lower: float | None = None) -> pd.Series:
        numer_vals = pd.to_numeric(numer, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        denom_vals = pd.to_numeric(denom, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        values = np.full(len(acs.index), np.nan, dtype=float)
        valid = np.isfinite(numer_vals) & np.isfinite(denom_vals) & (denom_vals > 0)
        np.divide(numer_vals, denom_vals, out=values, where=valid)
        out = pd.Series(values, index=acs.index, dtype=float)
        if lower is not None:
            out = out.clip(lower=lower)
        return out

    if _has_all({"hh_family", "hh_family_married", "hh_family_male_no_spouse", "hh_family_female_no_spouse"}):
        other = _series("hh_family") - _series("hh_family_married") - _series("hh_family_male_no_spouse") - _series("hh_family_female_no_spouse")
        derived["hh_family_other"] = _clip_nonnegative(other)
    if _has_all({"hh_family_married", "hh_family_married_children"}):
        married_no_children = (
            _series("hh_family_married")
            - _series("hh_family_married_children")
        )
        derived["hh_family_married_no_children"] = _clip_nonnegative(married_no_children)

    pop = _series("total_population")
    if pop is None:
        pop = pd.Series(np.nan, index=acs.index, dtype=float)
    if _has("poverty_count"):
        derived["poverty_rate"] = _unbounded_ratio(_series("poverty_count"), pop)
    if _has_all({"unemployed", "labor_force"}):
        derived["unemployment_rate"] = _unbounded_ratio(_series("unemployed"), _series("labor_force"))
    if _has("jobs_wac"):
        derived["jobs_per_capita"] = _unbounded_ratio(_series("jobs_wac"), pop)

    if _has("jobs_per_capita"):
        jobs_pc = _series("jobs_per_capita").clip(lower=0.0)
        derived["log_jobs_per_capita"] = np.log1p(jobs_pc)

    lodes_component_cols = [c for c in acs.columns if LODES_COMPONENT_RE.match(str(c))]
    if _has("jobs_wac") and lodes_component_cols:
        jobs_total = _series("jobs_wac")
        for col in lodes_component_cols:
            numer = _series(col)
            derived[f"lodes_{str(col).lower()}_share"] = _bounded_ratio(numer, jobs_total)

    def share(numer: str, denom: str, out_col: str) -> None:
        if not _has(numer) or not _has(denom):
            return
        derived[out_col] = _bounded_ratio(_series(numer), _series(denom))

    for col in ["edu_hs_diploma", "edu_bachelors", "edu_masters", "edu_professional", "edu_doctorate"]:
        share(col, "edu_pop_25plus", f"{col}_share")

    for col in ["rent_30_35pct", "rent_35_40pct", "rent_40_50pct", "rent_50plus_pct"]:
        share(col, "rent_burden_total", f"{col}_share")

    share("commute_45_59", "commute_total", "commute_45_59_share")
    share("commute_60plus", "commute_total", "commute_60plus_share")
    for col in [
        "commute_drive_alone",
        "commute_carpool",
        "commute_transit",
        "commute_taxi_ridehail",
        "commute_bicycle",
        "commute_walk",
        "commute_work_from_home",
    ]:
        share(col, "commute_mode_total", f"{col}_share")
    if _has_all({"commute_45_59", "commute_60plus", "commute_total"}):
        derived["commute_45plus_share"] = _bounded_ratio(
            _series("commute_45_59") + _series("commute_60plus"),
            _series("commute_total"),
        )

    income_bins = {
        "hh_income_lt25k_share": ["hh_income_lt10k", "hh_income_10_15k", "hh_income_15_20k", "hh_income_20_25k"],
        "hh_income_25_50k_share": ["hh_income_25_30k", "hh_income_30_35k", "hh_income_35_40k", "hh_income_40_45k", "hh_income_45_50k"],
        "hh_income_50_100k_share": ["hh_income_50_60k", "hh_income_60_75k", "hh_income_75_100k"],
        "hh_income_100_200k_share": ["hh_income_100_125k", "hh_income_125_150k", "hh_income_150_200k"],
        "hh_income_200k_plus_share": ["hh_income_200k_plus"],
    }
    for out_col, cols in income_bins.items():
        if _has_all(cols) and _has("hh_income_total"):
            numer = sum((_series(c) for c in cols), pd.Series(0.0, index=acs.index))
            derived[out_col] = _bounded_ratio(numer, _series("hh_income_total"))

    for col in [
        "hh_family",
        "hh_family_married",
        "hh_family_married_children",
        "hh_family_male_no_spouse_children",
        "hh_family_female_no_spouse_children",
        "hh_family_married_no_children",
        "hh_family_other",
        "hh_family_male_no_spouse",
        "hh_family_female_no_spouse",
        "hh_nonfamily",
    ]:
        share(col, "households_total", f"{col}_share")

    if _has_all({
        "hh_family_married_children",
        "hh_family_male_no_spouse_children",
        "hh_family_female_no_spouse_children",
    }):
        family_with_children = (
            _series("hh_family_married_children")
            + _series("hh_family_male_no_spouse_children")
            + _series("hh_family_female_no_spouse_children")
        )
        derived["hh_family_with_children"] = _clip_nonnegative(family_with_children)
        single_parent_with_children = (
            _series("hh_family_male_no_spouse_children")
            + _series("hh_family_female_no_spouse_children")
        )
        derived["hh_family_single_parent_with_children"] = _clip_nonnegative(single_parent_with_children)
        share("hh_family_with_children", "households_total", "hh_family_with_children_share")
        share("hh_family_with_children", "hh_family", "hh_family_with_children_within_family_share")
        share(
            "hh_family_single_parent_with_children",
            "households_total",
            "hh_family_single_parent_with_children_share",
        )
    share("hh_under18", "households_total", "hh_under18_share")
    share("hh_under18_nonfamily", "households_total", "hh_under18_nonfamily_share")

    share("housing_occupied", "housing_units_total", "housing_occupied_share")
    share("housing_vacant", "housing_units_total", "housing_vacant_share")
    share("tenure_owner", "tenure_total", "tenure_owner_share")
    share("tenure_renter", "tenure_total", "tenure_renter_share")
    if _has_all({"owner_no_vehicle", "renter_no_vehicle"}):
        derived["hh_no_vehicle"] = _clip_nonnegative(_series("owner_no_vehicle") + _series("renter_no_vehicle"))
        share("hh_no_vehicle", "vehicle_total", "hh_no_vehicle_share")
    if _has_all({"owner_1_vehicle", "renter_1_vehicle"}):
        derived["hh_1_vehicle"] = _clip_nonnegative(_series("owner_1_vehicle") + _series("renter_1_vehicle"))
        share("hh_1_vehicle", "vehicle_total", "hh_1_vehicle_share")
    if _has_all({"owner_2_vehicle", "renter_2_vehicle"}):
        derived["hh_2_vehicle"] = _clip_nonnegative(_series("owner_2_vehicle") + _series("renter_2_vehicle"))
        share("hh_2_vehicle", "vehicle_total", "hh_2_vehicle_share")
    if _has_all({"owner_3_vehicle", "renter_3_vehicle"}):
        derived["hh_3_vehicle"] = _clip_nonnegative(_series("owner_3_vehicle") + _series("renter_3_vehicle"))
        share("hh_3_vehicle", "vehicle_total", "hh_3_vehicle_share")
    if _has_all({"owner_4_vehicle", "owner_5plus_vehicle", "renter_4_vehicle", "renter_5plus_vehicle"}):
        derived["hh_4plus_vehicle"] = _clip_nonnegative(
            _series("owner_4_vehicle")
            + _series("owner_5plus_vehicle")
            + _series("renter_4_vehicle")
            + _series("renter_5plus_vehicle")
        )
        share("hh_4plus_vehicle", "vehicle_total", "hh_4plus_vehicle_share")
    if _has_all({"hh_2_vehicle", "hh_3_vehicle", "hh_4plus_vehicle"}):
        derived["hh_2plus_vehicle"] = _clip_nonnegative(
            _series("hh_2_vehicle") + _series("hh_3_vehicle") + _series("hh_4plus_vehicle")
        )
        share("hh_2plus_vehicle", "vehicle_total", "hh_2plus_vehicle_share")
    if _has_all({"hh_3_vehicle", "hh_4plus_vehicle"}):
        derived["hh_3plus_vehicle"] = _clip_nonnegative(_series("hh_3_vehicle") + _series("hh_4plus_vehicle"))
        share("hh_3plus_vehicle", "vehicle_total", "hh_3plus_vehicle_share")
    share("owner_no_vehicle", "vehicle_owner_total", "owner_no_vehicle_share")
    share("renter_no_vehicle", "vehicle_renter_total", "renter_no_vehicle_share")
    if _has_all({"owner_2_vehicle", "owner_3_vehicle", "owner_4_vehicle", "owner_5plus_vehicle"}):
        derived["owner_2plus_vehicle"] = _clip_nonnegative(
            _series("owner_2_vehicle")
            + _series("owner_3_vehicle")
            + _series("owner_4_vehicle")
            + _series("owner_5plus_vehicle")
        )
        share("owner_2plus_vehicle", "vehicle_owner_total", "owner_2plus_vehicle_share")
    if _has_all({"renter_2_vehicle", "renter_3_vehicle", "renter_4_vehicle", "renter_5plus_vehicle"}):
        derived["renter_2plus_vehicle"] = _clip_nonnegative(
            _series("renter_2_vehicle")
            + _series("renter_3_vehicle")
            + _series("renter_4_vehicle")
            + _series("renter_5plus_vehicle")
        )
        share("renter_2plus_vehicle", "vehicle_renter_total", "renter_2plus_vehicle_share")
    if _has("aggregate_vehicles_total"):
        derived["avg_vehicles_per_household"] = _unbounded_ratio(
            _series("aggregate_vehicles_total"),
            _series("vehicle_total"),
            lower=0.0,
        )
    if _has("aggregate_vehicles_owner"):
        derived["avg_vehicles_owner"] = _unbounded_ratio(
            _series("aggregate_vehicles_owner"),
            _series("vehicle_owner_total"),
            lower=0.0,
        )
    if _has("aggregate_vehicles_renter"):
        derived["avg_vehicles_renter"] = _unbounded_ratio(
            _series("aggregate_vehicles_renter"),
            _series("vehicle_renter_total"),
            lower=0.0,
        )
    share("snap_received", "snap_total", "snap_received_share")
    share("internet_subscription_any", "internet_total", "hh_internet_subscription_share")
    share("internet_dialup_only", "internet_total", "hh_dialup_only_share")
    share("internet_broadband_any", "internet_total", "hh_broadband_any_share")
    share("internet_cellular_any", "internet_total", "hh_cellular_plan_share")
    share("internet_cellular_only", "internet_total", "hh_cellular_only_share")
    share("internet_cable_fiber_dsl_any", "internet_total", "hh_wired_broadband_share")
    share("internet_cable_fiber_dsl_only", "internet_total", "hh_wired_broadband_only_share")
    share("internet_satellite_any", "internet_total", "hh_satellite_share")
    share("internet_satellite_only", "internet_total", "hh_satellite_only_share")
    share("internet_other_service_only", "internet_total", "hh_other_service_only_share")
    share("internet_no_subscription", "internet_total", "hh_access_without_subscription_share")
    share("internet_no_access", "internet_total", "hh_no_internet_share")
    share("computer_has_any", "computer_total", "hh_computer_access_share")
    share("computer_broadband", "computer_total", "hh_computer_broadband_share")
    share("computer_no_internet", "computer_total", "hh_computer_no_internet_share")
    share("computer_no_device", "computer_total", "hh_no_computer_share")
    share("labor_force", "total_population", "labor_force_per_capita")
    share("unemployed", "total_population", "unemployed_per_capita")
    if _has_all({"labor_force", "unemployed"}):
        derived["employed_count"] = _clip_nonnegative(_series("labor_force") - _series("unemployed"))
        share("employed_count", "total_population", "employed_per_capita")
    share("households_total", "housing_units_total", "households_per_housing_unit")
    share("housing_occupied", "total_population", "occupied_units_per_capita")
    share("housing_vacant", "total_population", "vacant_units_per_capita")
    share("tenure_owner", "total_population", "owner_units_per_capita")
    share("tenure_renter", "total_population", "renter_units_per_capita")

    for col in [
        "structure_single_detached",
        "structure_single_attached",
        "structure_mobile_home",
        "structure_other",
    ]:
        share(col, "structure_total", f"{col}_share")
    if _has_all({
        "structure_2_units",
        "structure_3_4_units",
        "structure_5_9_units",
        "structure_10_19_units",
        "structure_20_49_units",
        "structure_50plus_units",
    }):
        derived["structure_multifamily_small"] = _clip_nonnegative(_series("structure_2_units") + _series("structure_3_4_units"))
        derived["structure_multifamily_medium"] = _clip_nonnegative(_series("structure_5_9_units") + _series("structure_10_19_units"))
        derived["structure_multifamily_large"] = _clip_nonnegative(_series("structure_20_49_units") + _series("structure_50plus_units"))
        share("structure_multifamily_small", "structure_total", "structure_multifamily_small_share")
        share("structure_multifamily_medium", "structure_total", "structure_multifamily_medium_share")
        share("structure_multifamily_large", "structure_total", "structure_multifamily_large_share")

    for col in [
        "vacancy_for_rent",
        "vacancy_rented_not_occupied",
        "vacancy_for_sale",
        "vacancy_sold_not_occupied",
        "vacancy_seasonal",
        "vacancy_migrant",
        "vacancy_other",
    ]:
        share(col, "vacancy_total_b25004", f"{col}_share")

    age_built_bins = {
        "structure_built_2000_plus_share": [
            "structure_built_2020_plus",
            "structure_built_2010s",
            "structure_built_2000s",
        ],
        "structure_built_1980_1999_share": [
            "structure_built_1990s",
            "structure_built_1980s",
        ],
        "structure_built_1960_1979_share": [
            "structure_built_1970s",
            "structure_built_1960s",
        ],
        "structure_built_pre1960_share": [
            "structure_built_1950s",
            "structure_built_1940s",
            "structure_built_pre1940",
        ],
    }
    for out_col, cols in age_built_bins.items():
        if _has_all(cols) and _has("structure_built_total"):
            numer = sum((_series(c) for c in cols), pd.Series(0.0, index=acs.index))
            derived[out_col] = _bounded_ratio(numer, _series("structure_built_total"))

    if _has_all({
        "occupants_per_room_owner_101_150",
        "occupants_per_room_owner_151_200",
        "occupants_per_room_owner_201_plus",
    }):
        derived["occupants_per_room_owner_overcrowded"] = _clip_nonnegative(
            _series("occupants_per_room_owner_101_150")
            + _series("occupants_per_room_owner_151_200")
            + _series("occupants_per_room_owner_201_plus")
        )
        share(
            "occupants_per_room_owner_overcrowded",
            "occupants_per_room_owner_total",
            "owner_overcrowded_share",
        )
    if _has_all({
        "occupants_per_room_renter_101_150",
        "occupants_per_room_renter_151_200",
        "occupants_per_room_renter_201_plus",
    }):
        derived["occupants_per_room_renter_overcrowded"] = _clip_nonnegative(
            _series("occupants_per_room_renter_101_150")
            + _series("occupants_per_room_renter_151_200")
            + _series("occupants_per_room_renter_201_plus")
        )
        share(
            "occupants_per_room_renter_overcrowded",
            "occupants_per_room_renter_total",
            "renter_overcrowded_share",
        )
    if _has_all({"occupants_per_room_owner_overcrowded", "occupants_per_room_renter_overcrowded"}):
        derived["occupants_per_room_overcrowded"] = _clip_nonnegative(
            _series("occupants_per_room_owner_overcrowded")
            + _series("occupants_per_room_renter_overcrowded")
        )
        share("occupants_per_room_overcrowded", "housing_occupied", "overcrowded_occupied_share")

    for col in [
        "mobility_same_house",
        "mobility_diff_house_same_county",
        "mobility_diff_county_same_state",
        "mobility_diff_state",
        "mobility_abroad",
    ]:
        share(col, "mobility_total", f"{col}_share")

    male_under18 = ["male_under5", "male_5to9", "male_10to14", "male_15to17"]
    male_18_24 = ["male_18to19", "male_20", "male_21", "male_22to24"]
    male_15_24 = ["male_15to17", "male_18to19", "male_20", "male_21", "male_22to24"]
    male_18_29 = ["male_18to19", "male_20", "male_21", "male_22to24", "male_25to29"]
    male_18_34 = ["male_18to19", "male_20", "male_21", "male_22to24", "male_25to29", "male_30to34"]
    male_25_34 = ["male_25to29", "male_30to34"]
    male_25_44 = ["male_25to29", "male_30to34", "male_35to39", "male_40to44"]
    male_35_54 = ["male_35to39", "male_40to44", "male_45to49", "male_50to54"]
    male_45_64 = ["male_45to49", "male_50to54", "male_55to59", "male_60to61", "male_62to64"]
    male_65plus = ["male_65to66", "male_67to69", "male_70to74", "male_75to79", "male_80to84", "male_85plus"]
    for name, cols in [
        ("male_under18", male_under18),
        ("male_15_24", male_15_24),
        ("male_18_24", male_18_24),
        ("male_18_29", male_18_29),
        ("male_18_34", male_18_34),
        ("male_25_34", male_25_34),
        ("male_25_44", male_25_44),
        ("male_35_54", male_35_54),
        ("male_45_64", male_45_64),
        ("male_65plus", male_65plus),
    ]:
        if _has_all(cols) and _has("total_population"):
            numer = sum((_series(c) for c in cols), pd.Series(0.0, index=acs.index))
            derived[f"{name}_share"] = _bounded_ratio(numer, _series("total_population"))

    for col in ["total_population", "housing_units_total", "households_total", "jobs_wac", "aland20"]:
        if _has(col):
            derived[f"log_{col}"] = np.log1p(_series(col).clip(lower=0.0))

    if _has("aland20"):
        area_sqkm = _series("aland20") / 1_000_000.0
        derived["land_area_sqkm"] = area_sqkm
        if _has("total_population"):
            derived["population_density_sqkm"] = _unbounded_ratio(_series("total_population"), area_sqkm, lower=0.0)
        if _has("housing_units_total"):
            derived["housing_density_sqkm"] = _unbounded_ratio(_series("housing_units_total"), area_sqkm, lower=0.0)
        if _has("jobs_wac"):
            derived["jobs_density_sqkm"] = _unbounded_ratio(_series("jobs_wac"), area_sqkm, lower=0.0)
        if _has("public_school_count"):
            derived["public_school_density_sqkm"] = _unbounded_ratio(_series("public_school_count"), area_sqkm, lower=0.0)
        if _has("postsecondary_count"):
            derived["postsecondary_density_sqkm"] = _unbounded_ratio(_series("postsecondary_count"), area_sqkm, lower=0.0)
        if _has("education_anchor_total"):
            derived["education_anchor_density_sqkm"] = _unbounded_ratio(
                _series("education_anchor_total"), area_sqkm, lower=0.0
            )
        for col in ["population_density_sqkm", "housing_density_sqkm", "jobs_density_sqkm"]:
            if _has(col):
                derived[col] = _series(col).replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
                derived[f"log_{col}"] = np.log1p(derived[col])

    if derived:
        acs = pd.concat([acs, pd.DataFrame(derived, index=acs.index)], axis=1, copy=False)

    acs = apply_lodes_flow_daytime_features(acs)
    acs = acs.drop(
        columns=[
            "hh_family_with_children",
            "hh_family_single_parent_with_children",
            "employed_count",
            "structure_multifamily_small",
            "structure_multifamily_medium",
            "structure_multifamily_large",
            "occupants_per_room_owner_overcrowded",
            "occupants_per_room_renter_overcrowded",
            "occupants_per_room_overcrowded",
            *sorted(LODES_FLOW_RAW_COLS - {"jobs_rac"}),
        ],
        errors="ignore",
    ).copy()
    return acs


def build_ct_bg_2023_to_2020_map(
    *,
    ct_bg_2023_zip: Path,
    ct_bg_2020_zip: Path,
) -> dict[str, str]:
    """Map Connecticut ACS planning-region (2023-vintage) BG GEOIDs to legacy 2020 BG GEOIDs.

    The recoding is a purely administrative 1:1 relabel over identical BG geometry, so the
    representative point of each 2023 BG falls within exactly one 2020 BG.
    """
    import geopandas as gpd

    old = gpd.read_file(ct_bg_2020_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "bg2020"})
    new = gpd.read_file(ct_bg_2023_zip)[["GEOID", "geometry"]].rename(columns={"GEOID": "bg2023"})
    pts = new.copy()
    pts["geometry"] = pts.geometry.representative_point()
    mapping = gpd.sjoin(pts, old, how="left", predicate="within")[["bg2023", "bg2020"]].drop_duplicates()
    mapping = mapping.dropna(subset=["bg2020"]).copy()
    return dict(zip(mapping["bg2023"].astype(str), mapping["bg2020"].astype(str)))


def _normalize_connecticut_bg_ids(
    df: pd.DataFrame,
    *,
    ct_bg_2023_zip: Path | None,
    ct_bg_2020_zip: Path | None,
) -> pd.DataFrame:
    if ct_bg_2023_zip is None or ct_bg_2020_zip is None or not ct_bg_2023_zip.exists() or not ct_bg_2020_zip.exists():
        return df
    out = df.copy()
    ct_mask = out["state_fips"].astype(str).str.zfill(2).eq("09")
    if not ct_mask.any():
        return out

    bg_map = build_ct_bg_2023_to_2020_map(ct_bg_2023_zip=ct_bg_2023_zip, ct_bg_2020_zip=ct_bg_2020_zip)
    if not bg_map:
        return out

    old_bg = out.loc[ct_mask, "bg_id"].astype(str)
    mapped_bg = old_bg.map(bg_map).fillna(old_bg)
    out.loc[ct_mask, "bg_id"] = mapped_bg
    out.loc[ct_mask, "tract_id"] = mapped_bg.str.slice(0, 11)
    out.loc[ct_mask, "county_fips"] = mapped_bg.str.slice(2, 5)
    return out


def _normalize_connecticut_tract_ids(
    df: pd.DataFrame,
    *,
    ct_bg_2023_zip: Path | None,
    ct_bg_2020_zip: Path | None,
) -> pd.DataFrame:
    """Relabel CT ACS planning-region tract keys to the legacy 2020 tract GEOIDs.

    The BG frame is relabeled by _normalize_connecticut_bg_ids; any tract-keyed ACS frame merged
    against it (e.g. the tract-context features) must be relabeled with the SAME map or the CT
    merge keys can never intersect (planning-region counties 110-190 vs legacy counties 001-015).
    The tract map is derived from the shared BG map by truncation and must be one-to-one per 2023
    tract; a violation means the external TIGER vocabulary drifted, so fail loudly.
    """
    if ct_bg_2023_zip is None or ct_bg_2020_zip is None or not ct_bg_2023_zip.exists() or not ct_bg_2020_zip.exists():
        return df
    out = df.copy()
    ct_mask = out["state_fips"].astype(str).str.zfill(2).eq("09")
    if not ct_mask.any():
        return out

    bg_map = build_ct_bg_2023_to_2020_map(ct_bg_2023_zip=ct_bg_2023_zip, ct_bg_2020_zip=ct_bg_2020_zip)
    by_2023_tract: dict[str, set[str]] = {}
    for bg2023, bg2020 in bg_map.items():
        by_2023_tract.setdefault(bg2023[:11], set()).add(bg2020[:11])
    non_functional = {k: sorted(v) for k, v in by_2023_tract.items() if len(v) > 1}
    if non_functional:
        raise ValueError(
            "CT 2023->2020 tract relabel is not one-to-one (TIGER vocabulary drift); "
            f"offending 2023 tracts: {non_functional}"
        )
    tract_map = {k: next(iter(v)) for k, v in by_2023_tract.items()}
    old_tract = out.loc[ct_mask, "tract_id"].astype(str)
    mapped_tract = old_tract.map(tract_map).fillna(old_tract)
    out.loc[ct_mask, "tract_id"] = mapped_tract
    out.loc[ct_mask, "county_fips"] = mapped_tract.str.slice(2, 5)
    return out


def _backfill_acs_missing_block_groups(
    df: pd.DataFrame,
    *,
    backfill_csv: Path | None,
) -> pd.DataFrame:
    """Append survey-base rows for published BGs the ACS vintage dropped from its vocabulary.

    The ACS 2020-2024 5-year release omits some legacy-2020 BG GEOIDs entirely (observed: 39
    Suffolk County NY BGs, ~74k residents, absent from ACS — not renumbered; TIGER2024 keeps the
    legacy geometry). Their 2020 Decennial DHC counts (from
    scripts/pull/build_acs_missing_bg_decennial_backfill.py) are injected here as the base
    population/household values, BEFORE county-control scaling, so the county growth factor
    recomputes over the full universe and county totals stay on the POPEST controls.
    """
    if backfill_csv is None or not backfill_csv.exists():
        return df
    backfill = pd.read_csv(backfill_csv, dtype={"bg_id": str})
    if backfill.empty:
        return df
    backfill["bg_id"] = backfill["bg_id"].astype(str).str.zfill(12)
    existing = set(df["bg_id"].astype(str).str.zfill(12))
    backfill = backfill[~backfill["bg_id"].isin(existing)].copy()
    if backfill.empty:
        return df
    rows = pd.DataFrame(
        {
            "bg_id": backfill["bg_id"],
            "tract_id": backfill["bg_id"].str.slice(0, 11),
            "state_fips": backfill["bg_id"].str.slice(0, 2),
            "county_fips": backfill["bg_id"].str.slice(2, 5),
            "total_population": pd.to_numeric(backfill["population_dec2020"], errors="coerce"),
            "households_total": pd.to_numeric(backfill["households_occupied_dec2020"], errors="coerce"),
            "housing_units_total": pd.to_numeric(backfill["housing_units_dec2020"], errors="coerce"),
            "housing_occupied": pd.to_numeric(backfill["households_occupied_dec2020"], errors="coerce"),
            "housing_vacant": pd.to_numeric(backfill["housing_vacant_dec2020"], errors="coerce"),
        }
    )
    rows = rows.reindex(columns=df.columns)
    return pd.concat([df, rows], ignore_index=True)


def _dedupe_block_groups(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["bg_id", "tract_id", "state_fips", "county_fips"]
    if df.duplicated(subset=key_cols).sum() == 0:
        return df
    agg: dict[str, str] = {}
    for col in df.columns:
        if col in key_cols:
            continue
        agg[col] = "max" if pd.api.types.is_numeric_dtype(df[col]) else "first"
    return df.groupby(key_cols, dropna=False, as_index=False).agg(agg)


def build_tract_covariates(
    *,
    acs_tract_parquet: Path,
    lodes_dir: Path,
    county_pop_csv: Path,
    cpi_cache_csv: Path,
    cdc_places_parquet: Path | None = None,
    fema_nri_parquet: Path | None = None,
    cfg: CovariateBuildConfig = CovariateBuildConfig(),
) -> pd.DataFrame:
    acs = pd.read_parquet(acs_tract_parquet)
    acs["tract_id"] = acs["tract_id"].astype(str).str.zfill(11)
    acs["state_fips"] = acs["state_fips"].astype(str).str.zfill(2)
    acs["county_fips"] = acs["county_fips"].astype(str).str.zfill(3)
    acs = _clean_acs_medians(acs)

    county_est = load_county_population_estimates(csv_path=county_pop_csv, year=int(cfg.estimate_year))
    acs = align_tract_population_to_county(
        acs_tract_df=acs, county_estimates=county_est, acs_population_col="total_population"
    )
    acs = _inflate_money_columns(acs, cpi_cache_csv=cpi_cache_csv, cfg=cfg)

    jobs = compute_lodes_flow_frame_by_tract(lodes_dir=lodes_dir, cache_dir=cpi_cache_csv.parent.parent)
    jobs["state_fips"] = jobs["state_fips"].astype(str).str.zfill(2)
    acs = acs.merge(jobs, on=["tract_id", "state_fips"], how="left")
    acs["jobs_wac"] = pd.to_numeric(acs["jobs_wac"], errors="coerce").fillna(0).astype("Int64")
    if "jobs_rac" in acs.columns:
        acs["jobs_rac"] = pd.to_numeric(acs["jobs_rac"], errors="coerce").fillna(0).astype("Int64")
    for col in sorted(LODES_FLOW_RAW_COLS - {"jobs_rac"}):
        if col in acs.columns:
            acs[col] = pd.to_numeric(acs[col], errors="coerce").fillna(0).astype("Int64")

    # Optional CDC PLACES health outcomes data
    if cdc_places_parquet is not None:
        cdc = pd.read_parquet(cdc_places_parquet)
        cdc = cdc.rename(columns={"tractfips": "tract_id"}).copy()
        cdc["tract_id"] = cdc["tract_id"].astype(str).str.zfill(11)
        acs = acs.merge(cdc.drop(columns=["totalpopulation"], errors="ignore"), on="tract_id", how="left")

    # Optional FEMA NRI natural hazard risk data
    if fema_nri_parquet is not None:
        fema = pd.read_parquet(fema_nri_parquet)
        fema = fema.rename(columns={"TRACTFIPS": "tract_id"}).copy()
        fema["tract_id"] = fema["tract_id"].astype(str).str.zfill(11)
        acs = acs.merge(fema, on="tract_id", how="left")
    return _apply_common_derived_covariates(acs)


def build_block_group_covariates(
    *,
    acs_bg_parquet: Path,
    lodes_bg_parquet: Path,
    block_group_crosswalk_parquet: Path,
    cpi_cache_csv: Path,
    county_pop_csv: Path | None = None,
    acs_tract_parquet: Path | None = None,
    roads_bg_parquet: Path | None = None,
    hpms_bg_parquet: Path | None = None,
    education_anchor_bg_parquet: Path | None = None,
    medical_anchor_bg_parquet: Path | None = None,
    nlcd_bg_parquet: Path | None = None,
    ct_bg_2023_zip: Path | None = None,
    ct_bg_2020_zip: Path | None = None,
    acs_missing_bg_backfill_csv: Path | None = None,
    cfg: CovariateBuildConfig = CovariateBuildConfig(),
) -> pd.DataFrame:
    acs = pd.read_parquet(acs_bg_parquet)
    acs["bg_id"] = acs["bg_id"].astype(str).str.zfill(12)
    acs["tract_id"] = acs["tract_id"].astype(str).str.zfill(11)
    acs["state_fips"] = acs["state_fips"].astype(str).str.zfill(2)
    acs["county_fips"] = acs["county_fips"].astype(str).str.zfill(3)
    acs = _clean_acs_medians(acs)
    # BGs the ACS vintage dropped from its vocabulary get decennial-2020 base values injected
    # before county-control scaling so the county factor recomputes over the full universe.
    acs = _backfill_acs_missing_block_groups(acs, backfill_csv=acs_missing_bg_backfill_csv)
    if county_pop_csv is not None and county_pop_csv.exists():
        county_est = load_county_population_estimates(csv_path=county_pop_csv, year=int(cfg.estimate_year))
        acs = align_block_group_population_to_county(
            acs_bg_df=acs,
            county_estimates=county_est,
            acs_population_col="total_population",
        )
    # CT ACS uses planning-region county codes; align population on those before remapping
    # BG/tract GEOIDs back to the legacy 2020 codes used by the crime crosswalks.
    acs = _normalize_connecticut_bg_ids(
        acs,
        ct_bg_2023_zip=ct_bg_2023_zip,
        ct_bg_2020_zip=ct_bg_2020_zip,
    )
    acs = _inflate_money_columns(acs, cpi_cache_csv=cpi_cache_csv, cfg=cfg)

    lodes = pd.read_parquet(lodes_bg_parquet)
    lodes["bg_id"] = lodes["bg_id"].astype(str).str.zfill(12)
    lodes["tract_id"] = lodes["tract_id"].astype(str).str.zfill(11)
    lodes["state_fips"] = lodes["state_fips"].astype(str).str.zfill(2)
    lodes = lodes.loc[lodes["bg_id"].str.slice(0, 2).eq(lodes["state_fips"])].copy()
    if "lodes_year" in lodes.columns:
        lodes["lodes_year"] = pd.to_numeric(lodes["lodes_year"], errors="coerce")
        lodes = (
            lodes.sort_values(["bg_id", "tract_id", "state_fips", "lodes_year"], kind="stable")
            .drop_duplicates(["bg_id", "tract_id", "state_fips"], keep="last")
            .copy()
        )
    else:
        lodes = lodes.drop_duplicates(["bg_id", "tract_id", "state_fips"]).copy()
    lodes["jobs_wac"] = pd.to_numeric(lodes["jobs_wac"], errors="coerce").fillna(0.0)

    bg_geo = pd.read_parquet(block_group_crosswalk_parquet)[
        ["state_fips", "block_group_geoid", "total_aland20", "total_awater20", "total_pop20", "total_housing20"]
    ].drop_duplicates()
    bg_geo["block_group_geoid"] = bg_geo["block_group_geoid"].astype(str).str.zfill(12)
    bg_geo["tract_id"] = bg_geo["block_group_geoid"].str.slice(0, 11)
    bg_geo["state_fips"] = bg_geo["state_fips"].astype(str).str.zfill(2)
    bg_geo = bg_geo.rename(
        columns={
            "block_group_geoid": "bg_id",
            "total_aland20": "aland20",
            "total_awater20": "awater20",
            "total_pop20": "pop20",
            "total_housing20": "housing20",
        }
    )

    tract_context: pd.DataFrame | None = None
    if acs_tract_parquet is not None and acs_tract_parquet.exists():
        tract_acs = pd.read_parquet(acs_tract_parquet)
        tract_acs["tract_id"] = tract_acs["tract_id"].astype(str).str.zfill(11)
        tract_acs["state_fips"] = tract_acs["state_fips"].astype(str).str.zfill(2)
        tract_acs["county_fips"] = tract_acs["county_fips"].astype(str).str.zfill(3)
        # Same CT relabel as the BG frame: without it the CT tract-context merge keys never
        # intersect (planning-region vs legacy county codes) and all CT BGs get NaN context.
        tract_acs = _normalize_connecticut_tract_ids(
            tract_acs,
            ct_bg_2023_zip=ct_bg_2023_zip,
            ct_bg_2020_zip=ct_bg_2020_zip,
        )
        tract_acs = _clean_acs_medians(tract_acs)
        tract_acs = _inflate_money_columns(tract_acs, cpi_cache_csv=cpi_cache_csv, cfg=cfg)
        tract_acs = _apply_common_derived_covariates(tract_acs)
        tract_keep_cols = [
            "tract_id",
            "state_fips",
            "county_fips",
            "poverty_rate",
            "mobility_same_house_share",
            "mobility_diff_house_same_county_share",
            "mobility_diff_county_same_state_share",
            "mobility_diff_state_share",
            "mobility_abroad_share",
        ]
        tract_keep_cols = [c for c in tract_keep_cols if c in tract_acs.columns]
        tract_context = tract_acs[tract_keep_cols].rename(
            columns={
                "poverty_rate": "tract_poverty_rate",
                "mobility_same_house_share": "tract_mobility_same_house_share",
                "mobility_diff_house_same_county_share": "tract_mobility_diff_house_same_county_share",
                "mobility_diff_county_same_state_share": "tract_mobility_diff_county_same_state_share",
                "mobility_diff_state_share": "tract_mobility_diff_state_share",
                "mobility_abroad_share": "tract_mobility_abroad_share",
            }
        )
        # The CT relabel can collapse tract keys (two 2023 water tracts share one 2020 tract);
        # dedupe so the BG merge below cannot fan out rows.
        tract_context_keys = ["tract_id", "state_fips", "county_fips"]
        if tract_context.duplicated(subset=tract_context_keys).any():
            tract_context = tract_context.groupby(tract_context_keys, dropna=False, as_index=False).max()

    keep_lodes_cols = [
        c
        for c in lodes.columns
        if c in {"bg_id", "tract_id", "state_fips", "jobs_wac", "jobs_rac", "lodes_year", *LODES_FLOW_RAW_COLS}
        or str(c) in set(WAC_KEEP_COLS)
    ]
    acs = acs.merge(
        lodes[keep_lodes_cols],
        on=["bg_id", "tract_id", "state_fips"],
        how="left",
    ).merge(
        bg_geo,
        on=["bg_id", "tract_id", "state_fips"],
        how="left",
    )
    if tract_context is not None:
        acs = acs.merge(
            tract_context,
            on=["tract_id", "state_fips", "county_fips"],
            how="left",
        )
    if roads_bg_parquet is not None and roads_bg_parquet.exists():
        roads = pd.read_parquet(roads_bg_parquet)
        roads["bg_id"] = roads["bg_id"].astype(str).str.zfill(12)
        roads["tract_id"] = roads["tract_id"].astype(str).str.zfill(11)
        roads["state_fips"] = roads["state_fips"].astype(str).str.zfill(2)
        road_keep_cols = [
            c
            for c in roads.columns
            if c in {"bg_id", "tract_id", "state_fips"}
            or str(c).endswith("_length_km")
            or str(c).endswith("_road_share")
        ]
        acs = acs.merge(
            roads[road_keep_cols].drop_duplicates(["bg_id", "tract_id", "state_fips"]),
            on=["bg_id", "tract_id", "state_fips"],
            how="left",
        )
    if hpms_bg_parquet is not None and hpms_bg_parquet.exists():
        hpms = pd.read_parquet(hpms_bg_parquet)
        hpms["bg_id"] = hpms["bg_id"].astype(str).str.zfill(12)
        hpms["tract_id"] = hpms["tract_id"].astype(str).str.zfill(11)
        hpms["state_fips"] = hpms["state_fips"].astype(str).str.zfill(2)
        keep_cols = [
            c
            for c in hpms.columns
            if c in {"bg_id", "tract_id", "state_fips"}
            or str(c).startswith("hpms_")
        ]
        acs = acs.merge(
            hpms[keep_cols].drop_duplicates(["bg_id", "tract_id", "state_fips"]),
            on=["bg_id", "tract_id", "state_fips"],
            how="left",
        )
    if education_anchor_bg_parquet is not None and education_anchor_bg_parquet.exists():
        education = pd.read_parquet(education_anchor_bg_parquet)
        education["bg_id"] = education["bg_id"].astype(str).str.zfill(12)
        education["tract_id"] = education["tract_id"].astype(str).str.zfill(11)
        education["state_fips"] = education["state_fips"].astype(str).str.zfill(2)
        if "county_fips" in education.columns:
            education["county_fips"] = education["county_fips"].astype(str).str.zfill(3)
        keep_cols = [
            c
            for c in education.columns
            if c in {"bg_id", "tract_id", "state_fips"}
            or c in {
                "public_school_count",
                "postsecondary_count",
                "education_anchor_total",
                "public_school_present",
                "postsecondary_present",
            }
        ]
        acs = acs.merge(
            education[keep_cols].drop_duplicates(["bg_id", "tract_id", "state_fips"]),
            on=["bg_id", "tract_id", "state_fips"],
            how="left",
        )
    if medical_anchor_bg_parquet is not None and medical_anchor_bg_parquet.exists():
        medical = pd.read_parquet(medical_anchor_bg_parquet)
        medical["bg_id"] = medical["bg_id"].astype(str).str.zfill(12)
        medical["tract_id"] = medical["tract_id"].astype(str).str.zfill(11)
        medical["state_fips"] = medical["state_fips"].astype(str).str.zfill(2)
        keep_cols = [
            c
            for c in medical.columns
            if c in {"bg_id", "tract_id", "state_fips"}
            or c in {
                "acute_care_hospital_count",
                "emergency_hospital_count",
                "nearest_acute_care_hospital_km",
                "nearest_emergency_hospital_km",
            }
        ]
        acs = acs.merge(
            medical[keep_cols].drop_duplicates(["bg_id", "tract_id", "state_fips"]),
            on=["bg_id", "tract_id", "state_fips"],
            how="left",
        )
    if nlcd_bg_parquet is not None and nlcd_bg_parquet.exists():
        nlcd = pd.read_parquet(nlcd_bg_parquet)
        nlcd["bg_id"] = nlcd["bg_id"].astype(str).str.zfill(12)
        nlcd["tract_id"] = nlcd["tract_id"].astype(str).str.zfill(11)
        nlcd["state_fips"] = nlcd["state_fips"].astype(str).str.zfill(2)
        keep_cols = [
            c
            for c in nlcd.columns
            if c in {"bg_id", "tract_id", "state_fips"}
            or c.startswith("nlcd_")
        ]
        acs = acs.merge(
            nlcd[keep_cols].drop_duplicates(["bg_id", "tract_id", "state_fips"]),
            on=["bg_id", "tract_id", "state_fips"],
            how="left",
        )

    acs["jobs_wac"] = pd.to_numeric(acs["jobs_wac"], errors="coerce").fillna(0.0)
    if "jobs_rac" in acs.columns:
        acs["jobs_rac"] = pd.to_numeric(acs["jobs_rac"], errors="coerce").fillna(0.0)
    for col in sorted(LODES_FLOW_RAW_COLS - {"jobs_rac"}):
        if col in acs.columns:
            acs[col] = pd.to_numeric(acs[col], errors="coerce").fillna(0.0)
    acs["aland20"] = pd.to_numeric(acs["aland20"], errors="coerce")
    acs["awater20"] = pd.to_numeric(acs["awater20"], errors="coerce")
    acs = _dedupe_block_groups(acs)
    acs = _apply_common_derived_covariates(acs)
    raw_lodes_component_cols = [
        c for c in acs.columns
        if str(c) in set(WAC_KEEP_COLS)
    ]
    if raw_lodes_component_cols:
        acs = acs.drop(columns=raw_lodes_component_cols, errors="ignore")
    return acs


