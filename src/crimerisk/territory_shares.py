"""Offense-specific territory shares for `territory_prior` custom footprints.

A territory footprint row carries, per located exposure leg, the share of the block group's
leg that lies on the agency's territory blocks (`territory_share_<leg>`, written by
scripts/build/01_tribal_territory_footprints.py). The allocator combines them per offense with
the offense's published denominator: each leg's weight is the leg's contribution to that block
group's denominator (the normalizer's own coefficient times the block group's leg level), so a
leg the block group does not have gets weight 0 and the weights renormalise over the legs it
has. The combined share is the part of the block group's offense denominator on the territory.
Land share applies only where the block group has no denominator leg at all.

In the allocator (src/crimerisk/allocation.py `_territory_part_weights`) the weight is
s x (BG prior x s) + (1 - s) x (the footprint's prior-implied rate x `territory_offense_exposure`),
with s the combined share; `territory_offense_exposure`, sum(level x share), is the offense's
denominator on the territory part.

| Offense | Legs (coefficient x block-group level) |
|---|---|
| murder, rape, robbery, aggravated assault | person ensemble: residents, LandScan day, daytime jobs (configs/exposure_ensemble_weights_v1.csv) |
| larceny | 0.5 x person ensemble + 0.3 x destination POI + 0.2 x retail jobs, POI and retail rescaled as in larceny_opp_v1 |
| burglary | households (housing units located) + k_poi POI + k_retail retail + k_industrial industrial jobs + k_gq group quarters (residents located) |
| motor vehicle theft | household vehicles (housing units located) + commuter vehicles (WAC jobs located); where the presence floor binds, person presence (LandScan day or daytime jobs, the larger) |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.exposure_ensemble import (
    LARCENY_OPPORTUNITY_NORMALIZER_ID,
    LARCENY_PERSON_PART,
    PERSON_ENSEMBLE_NORMALIZER_ID,
)

TERRITORY_LEGS: tuple[str, ...] = (
    "residents",
    "landscan_day",
    "daytime_jobs",
    "destination_poi",
    "retail_jobs",
    "industrial_jobs",
    "wac_jobs",
    "housing_units",
)
TERRITORY_LAND_SHARE_COLUMN = "territory_share_land"
TERRITORY_SHARE_COLUMNS: tuple[str, ...] = (
    *(f"territory_share_{leg}" for leg in TERRITORY_LEGS),
    TERRITORY_LAND_SHARE_COLUMN,
)
TERRITORY_OFFENSE_SHARE_COLUMN = "territory_offense_share"
TERRITORY_OFFENSE_SHARE_BASIS_COLUMN = "territory_offense_share_basis"
# sum(level x share): the offense's denominator on the territory part, in the denominator's own
# linear form (a per-offense constant away from the published normalizer).
TERRITORY_OFFENSE_EXPOSURE_COLUMN = "territory_offense_exposure"
PERSON_LEG_BY_ENSEMBLE_LEG = {
    "landscan_night": "residents",
    "landscan_day": "landscan_day",
    "daytime_jobs": "daytime_jobs",
}
LEVEL_COLUMNS = ("bg_id", "offense", "leg", "level")


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"territory leg levels need block-group column {column!r}")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)


def _first(frame: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return _num(frame, column)
    raise ValueError(f"territory leg levels need one of {columns}")


def territory_leg_levels(
    bg_covariates: pd.DataFrame,
    *,
    weights: pd.DataFrame,
    normalizers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long (bg_id, offense, leg, level): each leg's contribution to the BG's offense denominator.

    `bg_covariates` is the stage-12 covariate frame (`_load_bg_covariates`); `normalizers` the
    stage-10 exposure normalizer table, whose legs (coverage repair included) define the person
    denominators. Without it the covariate frame's unrepaired legs stand in. `weights` is the
    exposure ensemble weight table. Only block groups in `bg_covariates` are returned.
    """
    cov = bg_covariates.copy()
    cov["bg_id"] = cov["bg_id"].astype("string").str.zfill(12)
    person_frame = cov[["bg_id"]].copy()
    if normalizers is not None:
        norm = normalizers.copy()
        norm["bg_id"] = norm["bg_id"].astype("string").str.zfill(12)
        person_frame = person_frame.merge(
            norm[["bg_id", "residential_leg", "landscan_day_leg", "daytime_jobs_leg",
                  "landscan_coverage_repaired"]],
            on="bg_id", how="left", validate="one_to_one",
        )
        residents = _num(person_frame, "residential_leg")
        day = _num(person_frame, "landscan_day_leg")
        jobs = _num(person_frame, "daytime_jobs_leg")
        repaired = person_frame["landscan_coverage_repaired"].fillna(False).astype(bool)
        universe = norm
    else:
        residents = _num(cov, "population")
        day = _first(cov, "landscan_day_pop")
        jobs = _num(cov, "daytime_population_jobs_proxy")
        repaired = pd.Series(False, index=cov.index)
        universe = None

    person = weights[weights["normalizer_id"].eq(PERSON_ENSEMBLE_NORMALIZER_ID)]
    hybrid = weights[weights["normalizer_id"].eq(LARCENY_OPPORTUNITY_NORMALIZER_ID)]
    hybrid_weight = {str(r.leg): float(r.weight) for r in hybrid.itertuples()}
    frames: list[pd.DataFrame] = []

    def emit(offense: str, leg: str, level: pd.Series) -> None:
        frames.append(
            pd.DataFrame(
                {"bg_id": cov["bg_id"].to_numpy(), "offense": offense, "leg": leg,
                 "level": np.asarray(level, dtype=float)}
            )
        )

    def person_legs(offense: str, scale: float) -> None:
        rows = person[person["offense"].eq(offense)]
        w = {str(r.leg): float(r.weight) * scale for r in rows.itertuples()}
        emit(offense, "residents", w["landscan_night"] * residents)
        # A LandScan coverage hole publishes the daytime-jobs proxy in the day leg's place.
        emit(offense, "landscan_day", w["landscan_day"] * day.where(~repaired, 0.0))
        emit(offense, "daytime_jobs", w["daytime_jobs"] * jobs + w["landscan_day"] * day.where(repaired, 0.0))

    for offense in ("murder", "rape", "robbery", "aggravated_assault"):
        person_legs(offense, 1.0)

    # Larceny: the opportunity parts are rescaled to the person ensemble's universe total.
    person_part = hybrid_weight[LARCENY_PERSON_PART]
    person_legs("larceny", person_part)
    rows = person[person["offense"].eq("larceny")]
    lw = {str(r.leg): float(r.weight) for r in rows.itertuples()}
    if universe is not None:
        person_total = float(
            (lw["landscan_night"] * _num(universe, "residential_leg")
             + lw["landscan_day"] * _num(universe, "landscan_day_leg")
             + lw["daytime_jobs"] * _num(universe, "daytime_jobs_leg")).sum()
        )
        poi_total = float(_num(universe, "destination_poi_total").sum())
        retail_total = float(_num(universe, "lodes_retail_jobs").sum())
    else:
        person_total = float((lw["landscan_night"] * residents + lw["landscan_day"] * day
                              + lw["daytime_jobs"] * jobs).sum())
        poi_total = float(_num(cov, "destination_poi_total").sum())
        retail_total = float(_num(cov, "lodes_retail_jobs").sum())
    poi_scale = person_total / poi_total if poi_total > 0 else 0.0
    retail_scale = person_total / retail_total if retail_total > 0 else 0.0
    emit("larceny", "destination_poi",
         hybrid_weight["destination_poi"] * poi_scale * _num(cov, "destination_poi_total"))
    emit("larceny", "retail_jobs",
         hybrid_weight["retail_jobs"] * retail_scale * _num(cov, "lodes_retail_jobs"))

    # Burglary premises: the stage-12 NNLS coefficients ride on every covariate row.
    emit("burglary", "housing_units", _num(cov, "households_total"))
    emit("burglary", "destination_poi",
         _num(cov, "burglary_destination_poi_exposure_weight") * _num(cov, "destination_poi_total"))
    emit("burglary", "retail_jobs",
         _num(cov, "burglary_retail_jobs_exposure_weight") * _num(cov, "lodes_retail_jobs"))
    emit("burglary", "industrial_jobs",
         _num(cov, "burglary_industrial_jobs_exposure_weight") * _num(cov, "lodes_industrial_jobs"))
    emit("burglary", "residents",
         _num(cov, "burglary_group_quarters_exposure_weight") * _num(cov, "census2020_gq_total"))

    # Motor vehicle theft: household plus commuter vehicles, or the presence floor where it binds.
    household_vehicles = _num(cov, "aggregate_vehicles_total")
    commuter_vehicles = _num(cov, "mvt_commuter_vehicle_proxy")
    published = _num(cov, "vehicle_exposure_2024")
    floored = published > (household_vehicles + commuter_vehicles) * (1.0 + 1e-9) + 1e-9
    day_presence = _num(cov, "landscan_day_pop") >= _num(cov, "daytime_population_jobs_proxy")
    emit("motor_vehicle_theft", "housing_units", household_vehicles.where(~floored, 0.0))
    emit("motor_vehicle_theft", "wac_jobs", commuter_vehicles.where(~floored, 0.0))
    emit("motor_vehicle_theft", "landscan_day", published.where(floored & day_presence, 0.0))
    emit("motor_vehicle_theft", "daytime_jobs", published.where(floored & ~day_presence, 0.0))

    out = pd.concat(frames, ignore_index=True)
    missing = set(OFFENSES_7) - set(out["offense"])
    if missing:
        raise ValueError(f"territory leg levels missing offenses {sorted(missing)}")
    return out[list(LEVEL_COLUMNS)]


def attach_territory_offense_shares(rows: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Add `territory_offense_share` to footprint rows that carry `bg_id`, `offense` and the
    per-leg shares: sum(level x share) / sum(level) over the legs whose share is defined; the
    land share where the block group has no positive leg. `territory_offense_exposure` is the
    numerator, sum(level x share): the offense's denominator on the territory part."""
    out = rows.copy()
    out["_row"] = np.arange(len(out))
    long = out[["_row", "bg_id", "offense", *TERRITORY_SHARE_COLUMNS]].melt(
        id_vars=["_row", "bg_id", "offense", TERRITORY_LAND_SHARE_COLUMN],
        value_vars=[f"territory_share_{leg}" for leg in TERRITORY_LEGS],
        var_name="leg",
        value_name="share",
    )
    long["leg"] = long["leg"].str.removeprefix("territory_share_")
    long["bg_id"] = long["bg_id"].astype("string").str.zfill(12)
    level = levels.copy()
    level["bg_id"] = level["bg_id"].astype("string").str.zfill(12)
    long = long.merge(level, on=["bg_id", "offense", "leg"], how="left", validate="many_to_one")
    share = pd.to_numeric(long["share"], errors="coerce")
    weight = pd.to_numeric(long["level"], errors="coerce").fillna(0.0).clip(lower=0.0)
    used = weight.gt(0.0) & share.notna()
    long["_num"] = np.where(used, weight * share.fillna(0.0).clip(0.0, 1.0), 0.0)
    long["_den"] = np.where(used, weight, 0.0)
    sums = long.groupby("_row")[["_num", "_den"]].sum()
    land = out.set_index("_row")[TERRITORY_LAND_SHARE_COLUMN]
    land = pd.to_numeric(land, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    den = sums["_den"].reindex(out["_row"]).fillna(0.0).to_numpy()
    num = sums["_num"].reindex(out["_row"]).fillna(0.0).to_numpy()
    out[TERRITORY_OFFENSE_SHARE_COLUMN] = np.where(
        den > 0.0, num / np.where(den > 0.0, den, 1.0), land.to_numpy()
    )
    out[TERRITORY_OFFENSE_SHARE_BASIS_COLUMN] = np.where(den > 0.0, "denominator_legs", "land")
    out[TERRITORY_OFFENSE_EXPOSURE_COLUMN] = num
    return out.drop(columns="_row")
