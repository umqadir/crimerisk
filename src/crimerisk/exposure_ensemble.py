"""v2 exposure normalizers: per-offense convex ensembles over the person-exposure surfaces.

The deployed person denominator is a HARD MAX,
`exposure_proxy_2024 = HQcap(max(landscan_day_pop, max(population + jobs_wac - jobs_rac, population)))`.
E3 (`analysis_scratch/final_phase/e3_exposure/`) measured it against 20 police incident feeds and
replaced it with a **per-offense convex ensemble** over three person-scale legs -- LandScan night,
LandScan day, and the LODES daytime-jobs proxy -- plus, for larceny only, an opportunity hybrid
that adds destination POI and retail jobs. Weights are frozen in
`configs/exposure_ensemble_weights_v1.csv`; the contract is
`analysis_scratch/final_phase/EXPOSURE_ENSEMBLE_CONTRACT.md`.

Three rules are load-bearing and enforced rather than trusted:

* **These are opportunity-normalized intensities, not person-time risk.** A larceny normalizer
  that carries retail floorspace is not a headcount, and E3's own caveat is that adopting it means
  a mall stops looking high-risk *per unit of exposure*. Every artifact this module writes carries
  the normalizer's id and that semantics string, and every rate regenerates from a count plus a
  named, versioned normalizer.
* **Weights are read, never re-derived.** Selection happened once, leave-one-source-out, against
  incident feeds. A new weight vector is a new versioned file, never an in-place edit.
* **Each normalizer preserves a reference-universe total.** Every named normalizer is rescaled by
  a single positive scalar so its total over the reference universe equals that universe's
  resident population. A global scalar cannot change any within-jurisdiction share or rank -- it
  fixes the LEVEL the rate is quoted in, and nothing else.

No AGS value enters anywhere: E3's truth was police incident feeds and its features were the
pipeline's own exposure surfaces plus ACS/LODES/Overture.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.build_freshness import artifact_is_current, write_dependency_stamp
from crimerisk.crime import OFFENSES_7
from crimerisk.denominators import (
    LANDSCAN_DAY_POP_COLUMN,
    LANDSCAN_NIGHT_POP_COLUMN,
    PERSON_EXPOSURE_DENOMINATOR_OFFENSES,
)
from crimerisk.paths import RepoPaths
from crimerisk.qcew_exposure import (
    LODES_VINTAGE_YEARS,
    MODELED_PLACE_EXPOSURE_PROXY,
    QCEW_SCALE_VERSION,
    RETAIL_LEG,
    TOTAL_LEG,
    QcewScalingConfig,
    attach_scale_factors,
    build_scale_factor_table,
    qcew_annual_path,
    scale_daytime_jobs_leg,
    scale_retail_leg,
    summarize_applied_factors,
    summarize_scale_factors,
)
from crimerisk.qcew_exposure import (
    LEG_INDUSTRY_CODE as QCEW_LEG_INDUSTRY_CODE,
    SCALED_LEGS as QCEW_SCALED_LEGS,
    SCALE_FACTOR_CEILING as QCEW_SCALE_FACTOR_CEILING,
    SCALE_FACTOR_FLOOR as QCEW_SCALE_FACTOR_FLOOR,
    SCALE_TIERS as QCEW_SCALE_TIERS,
    TIER_IDENTITY as QCEW_TIER_IDENTITY,
)


# --- naming and versioning ---------------------------------------------------------------

EXPOSURE_NORMALIZER_VERSION = "exposure_ensemble_v1"
CENSUS_RESIDENTIAL_NORMALIZER_VERSION = "exposure_ensemble_v2_census_residential"
PERSON_ENSEMBLE_NORMALIZER_ID = "person_ens_v1"
LARCENY_OPPORTUNITY_NORMALIZER_ID = "larceny_opp_v1"
BURGLARY_PREMISES_NORMALIZER_ID = "premises_nnls_v1"
MVT_VEHICLE_NORMALIZER_ID = "vehicle_exposure_v1"
LEGACY_PERSON_NORMALIZER_ID = "person_hard_max_v1"

# Said in the artifact, not only in the docs: what these denominators are and are not.
NORMALIZER_SEMANTICS = "opportunity_normalized_intensity_not_person_time_risk"
# PLAN item 6's rename. Once the LODES legs carry a QCEW-updated level, the lane is no longer
# reporting an observed workplace count at its own vintage; it is reporting a MODEL of where
# exposure is in the target year, and the semantics string says so before anything else does.
QCEW_UPDATED_NORMALIZER_SEMANTICS = (
    f"{MODELED_PLACE_EXPOSURE_PROXY}__{NORMALIZER_SEMANTICS}"
)


def normalizer_semantics(*, qcew_updated: bool = False) -> str:
    return QCEW_UPDATED_NORMALIZER_SEMANTICS if qcew_updated else NORMALIZER_SEMANTICS

# The published normalizer per offense when the lane is enabled. Burglary keeps its premises
# denominator (E3: the deployed form wins on both criteria) and motor vehicle theft keeps its
# vehicle denominator (E3 flagged it for its own experiment; changing it here is out of scope).
NORMALIZER_ID_BY_OFFENSE: dict[str, str] = {
    "murder": PERSON_ENSEMBLE_NORMALIZER_ID,
    "rape": PERSON_ENSEMBLE_NORMALIZER_ID,
    "robbery": PERSON_ENSEMBLE_NORMALIZER_ID,
    "aggravated_assault": PERSON_ENSEMBLE_NORMALIZER_ID,
    "larceny": LARCENY_OPPORTUNITY_NORMALIZER_ID,
    "burglary": BURGLARY_PREMISES_NORMALIZER_ID,
    "motor_vehicle_theft": MVT_VEHICLE_NORMALIZER_ID,
}

# Offenses whose published denominator this lane actually replaces.
ENSEMBLE_OFFENSES: tuple[str, ...] = tuple(
    offense for offense in OFFENSES_7 if offense in PERSON_EXPOSURE_DENOMINATOR_OFFENSES
)

WEIGHTS_FILENAME = "exposure_ensemble_weights_v1.csv"
RECOMMENDATIONS_FILENAME = "exposure_ensemble_recommendations_v1.csv"
WEIGHT_TABLE_COLUMNS = ("normalizer_id", "offense", "leg", "surface_column", "weight", "provisional", "note")

# E3 `PERSON_LEGS`, in the experiment's own order. The mix vector in recommendations.csv is read
# in this order and nowhere else.
PERSON_LEGS: tuple[str, ...] = ("landscan_night", "landscan_day", "daytime_jobs")
PERSON_LEG_SOURCE_COLUMNS: dict[str, str] = {
    "landscan_night": LANDSCAN_NIGHT_POP_COLUMN,
    "landscan_day": LANDSCAN_DAY_POP_COLUMN,
    "daytime_jobs": "daytime_population_jobs_proxy",
}
PERSON_LEG_FRAME_COLUMNS: dict[str, str] = {
    "landscan_night": "landscan_night_leg",
    "landscan_day": "landscan_day_leg",
    "daytime_jobs": "daytime_jobs_leg",
}

# E3 `HYBRID_PARTS`. `person_ensemble` is `person_ens_v1` evaluated at larceny's own leg weights;
# the other three are opportunity surfaces rescaled to the person ensemble's reference-universe
# total before mixing, which is what makes a convex mix of persons and POI counts coherent.
LARCENY_HYBRID_PARTS: tuple[str, ...] = ("person_ensemble", "destination_poi", "retail_jobs", "vehicles")
LARCENY_PERSON_PART = "person_ensemble"
LARCENY_HYBRID_SOURCE_COLUMNS: dict[str, str] = {
    "destination_poi": "destination_poi_total",
    "retail_jobs": "lodes_retail_jobs",
    "vehicles": "vehicle_exposure_2024",
}

LANDSCAN_COVERAGE_REPAIR_COLUMN = "landscan_coverage_repaired"
LANDSCAN_NIGHT_RESIDENTIAL_SOURCE = "landscan_night"
CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE = "census_release_population"
RESIDENTIAL_LEG_SOURCES = (
    LANDSCAN_NIGHT_RESIDENTIAL_SOURCE,
    CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE,
)
CENSUS_RESIDENTIAL_COLUMNS = [
    "raw_landscan_night_pop",
    "residential_leg",
    "residential_leg_source",
]


def _validate_residential_leg_source(source: str) -> str:
    value = str(source)
    if value not in RESIDENTIAL_LEG_SOURCES:
        raise ValueError(
            f"residential_leg_source must be one of {RESIDENTIAL_LEG_SOURCES}, got {value!r}"
        )
    return value


def exposure_normalizer_version(*, residential_leg_source: str) -> str:
    source = _validate_residential_leg_source(residential_leg_source)
    if source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE:
        return CENSUS_RESIDENTIAL_NORMALIZER_VERSION
    return EXPOSURE_NORMALIZER_VERSION


def opportunity_normalizer_column(offense: str) -> str:
    offense_key = str(offense)
    if offense_key not in OFFENSES_7:
        raise KeyError(f"Unknown offense for normalizer lookup: {offense!r}")
    return f"opportunity_normalizer_{offense_key}"


def normalizer_id_column(offense: str) -> str:
    return f"primary_denominator_normalizer_id_{str(offense)}"


def normalizer_id_for_offense(offense: str, *, enabled: bool = True) -> str:
    offense_key = str(offense)
    if offense_key not in NORMALIZER_ID_BY_OFFENSE:
        raise KeyError(f"Unknown offense for normalizer lookup: {offense!r}")
    if enabled:
        return NORMALIZER_ID_BY_OFFENSE[offense_key]
    if offense_key in PERSON_EXPOSURE_DENOMINATOR_OFFENSES:
        return LEGACY_PERSON_NORMALIZER_ID
    return NORMALIZER_ID_BY_OFFENSE[offense_key]


EXPOSURE_NORMALIZER_COLUMNS: list[str] = [
    "state_fips",
    "bg_id",
    "population",
    *PERSON_LEG_FRAME_COLUMNS.values(),
    LANDSCAN_COVERAGE_REPAIR_COLUMN,
    "destination_poi_total",
    "lodes_retail_jobs",
    *[f"opportunity_normalizer_{offense}" for offense in ENSEMBLE_OFFENSES],
]

# Published only when the QCEW updating lane is on, so a flag-off artifact keeps exactly the
# schema it had before this lane existed. `lodes_vintage` is per block group because the prepared
# LODES extract is not one vintage: 2023 for 49 states, 2021 for Michigan, 2016 for Alaska.
QCEW_UPDATING_COLUMNS: list[str] = [
    "lodes_vintage",
    "qcew_scale_vintage",
    *[f"qcew_{leg}_scale_factor" for leg in QCEW_SCALED_LEGS],
    *[f"qcew_{leg}_scale_tier" for leg in QCEW_SCALED_LEGS],
]


def exposure_normalizer_columns(
    *,
    qcew_updated: bool = False,
    residential_leg_source: str = LANDSCAN_NIGHT_RESIDENTIAL_SOURCE,
) -> list[str]:
    source = _validate_residential_leg_source(residential_leg_source)
    columns = list(EXPOSURE_NORMALIZER_COLUMNS)
    if source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE:
        columns = [*columns, *CENSUS_RESIDENTIAL_COLUMNS]
    if qcew_updated:
        columns = [*columns, *QCEW_UPDATING_COLUMNS]
    return columns


@dataclass(frozen=True)
class ExposureEnsembleConfig:
    """Structural constants only -- nothing here is fitted at build time."""

    year: int = 2024
    weights_path: Path | None = None
    # Mirrors allocation.RELEASE_EXCLUDED_STATE_FIPS. The reference universe whose resident
    # population total each normalizer preserves is the published one.
    excluded_state_fips: tuple[str, ...] = ("02", "15", "72")
    # PLAN item 6. Default off, like every other v2 lane: with it off the normalizer artifact is
    # byte-identical to the one this flag did not exist for.
    enable_qcew_exposure_updating: bool = False
    # Explicit sensitivity arm. The default preserves the v52 artifact path, schema, and math.
    # The opt-in arm uses the release Census population in the frozen nighttime weight slot,
    # nationwide, while retaining the observed LandScan values in separate artifact columns.
    residential_leg_source: str = LANDSCAN_NIGHT_RESIDENTIAL_SOURCE
    baseline_normalizers_path: Path | None = None
    qcew: QcewScalingConfig = QcewScalingConfig()


# --- paths -------------------------------------------------------------------------------


def ensemble_weights_path(paths: RepoPaths) -> Path:
    return paths.repo_root / "configs" / WEIGHTS_FILENAME


def ensemble_recommendations_path(paths: RepoPaths) -> Path:
    return paths.repo_root / "configs" / RECOMMENDATIONS_FILENAME


def exposure_normalizers_path(
    paths: RepoPaths,
    *,
    year: int,
    qcew_updated: bool = False,
    residential_leg_source: str = LANDSCAN_NIGHT_RESIDENTIAL_SOURCE,
) -> Path:
    # A QCEW-updated surface is a different artifact, not a newer one: freshness here is
    # content-addressed over input FILES, and toggling a flag changes no file, so sharing one path
    # would let a stale stamp hand back the wrong surface.
    source = _validate_residential_leg_source(residential_leg_source)
    suffix = (
        "_census_residential"
        if source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
        else ""
    )
    if qcew_updated:
        suffix += "_qcew"
    return paths.state_dir / "modeling" / f"bg_exposure_normalizers_{int(year)}{suffix}.parquet"


def exposure_normalizers_summary_path(
    paths: RepoPaths,
    *,
    year: int,
    qcew_updated: bool = False,
    residential_leg_source: str = LANDSCAN_NIGHT_RESIDENTIAL_SOURCE,
) -> Path:
    return exposure_normalizers_path(
        paths,
        year=year,
        qcew_updated=qcew_updated,
        residential_leg_source=residential_leg_source,
    ).with_suffix(".summary.json")


def qcew_dependency_paths(paths: RepoPaths, *, config: ExposureEnsembleConfig) -> list[Path]:
    """The QCEW annual files the lane can read, over every LODES vintage plus the target year.

    Listed unconditionally over the known vintage set rather than discovered from the extract:
    `existing_dependency_paths` drops the absent ones, and a dependency list that changes with the
    data it is stamping cannot detect its own inputs appearing.
    """
    years = sorted({*LODES_VINTAGE_YEARS, int(config.qcew.target_year)})
    return [
        qcew_annual_path(paths, year=year, industry_code=industry_code)
        for year in years
        for industry_code in (QCEW_LEG_INDUSTRY_CODE[leg] for leg in QCEW_SCALED_LEGS)
    ]


def exposure_normalizer_dependency_paths(
    paths: RepoPaths, *, config: ExposureEnsembleConfig
) -> list[Path]:
    # Imported lazily: model_surface pulls in the whole covariate stack and this list is also read
    # by callers that never build the frame.
    from crimerisk.denominators import LANDSCAN_BG_RELATIVE_PATH
    from crimerisk.model_surface import bg_feature_dependency_paths

    residential_source = _validate_residential_leg_source(config.residential_leg_source)
    upstream = (
        [
            config.baseline_normalizers_path
            or exposure_normalizers_path(paths, year=int(config.year)),
        ]
        if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
        else bg_feature_dependency_paths(paths, year=int(config.year))
    )
    surface_inputs = (
        []
        if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
        else [
            paths.data_dir
            / "Overture-Places"
            / "parsed"
            / "block_group_overture_places_states_latest.parquet",
            paths.data_dir / "LODES" / "parsed" / "lodes_wac_block_groups.parquet",
        ]
    )
    return [
        *upstream,
        paths.data_dir / LANDSCAN_BG_RELATIVE_PATH,
        *surface_inputs,
        config.weights_path or ensemble_weights_path(paths),
        *(
            qcew_dependency_paths(paths, config=config)
            if bool(config.enable_qcew_exposure_updating)
            else []
        ),
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("qcew_exposure.py"),
    ]


def exposure_normalizers_artifact_is_current(
    paths: RepoPaths, *, config: ExposureEnsembleConfig, out_path: Path | None = None
) -> bool:
    qcew_updated = bool(config.enable_qcew_exposure_updating)
    residential_source = _validate_residential_leg_source(config.residential_leg_source)
    artifact = out_path or exposure_normalizers_path(
        paths,
        year=int(config.year),
        qcew_updated=qcew_updated,
        residential_leg_source=residential_source,
    )
    summary_path = artifact.with_suffix(".summary.json")
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    if summary.get("normalizer_version") != exposure_normalizer_version(
        residential_leg_source=residential_source
    ):
        return False
    if (summary.get("residential_leg") or {}).get("source") != residential_source:
        return False
    return artifact_is_current(artifact, exposure_normalizer_dependency_paths(paths, config=config))


# --- the frozen weight table -------------------------------------------------------------


def load_ensemble_weights(path: Path) -> pd.DataFrame:
    """Read the frozen weight table and refuse anything that is not a convex simplex point.

    This is a read, not a selection. E3 selected these leave-one-source-out, once. Validation
    exists so a corrupted or hand-edited table fails the build instead of silently reshaping
    every published rate in the country.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is the exposure-ensemble weight source of record and is absent. Its person "
            f"legs are a transcription of analysis_scratch/final_phase/e3_exposure/"
            f"recommendations.csv and its larceny hybrid of that run's pooled_best_combo."
        )
    frame = pd.read_csv(path)
    missing = [column for column in WEIGHT_TABLE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"exposure ensemble weight table {path} is missing columns {missing}")
    for column in ("normalizer_id", "offense", "leg", "surface_column"):
        frame[column] = frame[column].astype(str).str.strip()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").astype(float)
    frame["provisional"] = (
        frame["provisional"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )
    if frame["provisional"].isna().any():
        raise ValueError(f"exposure ensemble weight table {path} carries a non-boolean provisional flag")
    frame["provisional"] = frame["provisional"].astype(bool)

    if frame.duplicated(["normalizer_id", "offense", "leg"]).any():
        raise ValueError(
            f"exposure ensemble weight table {path} carries duplicate (normalizer_id, offense, leg) rows"
        )
    if frame["weight"].isna().any():
        raise ValueError(f"exposure ensemble weight table {path} carries a non-numeric weight")
    if frame["weight"].lt(0.0).any() or frame["weight"].gt(1.0).any():
        raise ValueError(f"exposure ensemble weight table {path} carries a weight outside [0, 1]")

    expected_person = {
        (PERSON_ENSEMBLE_NORMALIZER_ID, offense, leg)
        for offense in ENSEMBLE_OFFENSES
        for leg in PERSON_LEGS
    }
    expected_hybrid = {
        (LARCENY_OPPORTUNITY_NORMALIZER_ID, "larceny", part) for part in LARCENY_HYBRID_PARTS
    }
    actual = set(zip(frame["normalizer_id"], frame["offense"], frame["leg"], strict=True))
    expected = expected_person | expected_hybrid
    if actual != expected:
        raise ValueError(
            f"exposure ensemble weight table {path} must cover exactly {len(expected)} "
            f"(normalizer_id, offense, leg) cells: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected"
        )

    sums = frame.groupby(["normalizer_id", "offense"])["weight"].sum()
    off_simplex = sums[(sums - 1.0).abs() > 1e-9]
    if not off_simplex.empty:
        raise ValueError(
            f"exposure ensemble weight table {path} carries a weight vector that does not sum to 1: "
            f"{off_simplex.to_dict()}"
        )
    for row in frame.itertuples():
        expected_column = (
            PERSON_LEG_SOURCE_COLUMNS.get(str(row.leg))
            if str(row.normalizer_id) == PERSON_ENSEMBLE_NORMALIZER_ID
            else LARCENY_HYBRID_SOURCE_COLUMNS.get(str(row.leg))
        )
        if expected_column is not None and str(row.surface_column) != expected_column:
            raise ValueError(
                f"exposure ensemble weight table {path} points leg {row.leg!r} at "
                f"{row.surface_column!r}; this lane reads {expected_column!r}"
            )
    return frame.sort_values(["normalizer_id", "offense", "leg"], kind="mergesort").reset_index(drop=True)


def person_leg_weights(weights: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Per offense, the (night, day, jobs) triple in `PERSON_LEGS` order."""
    person = weights[weights["normalizer_id"].eq(PERSON_ENSEMBLE_NORMALIZER_ID)]
    lookup = {
        (str(row.offense), str(row.leg)): float(row.weight) for row in person.itertuples()
    }
    return {
        offense: tuple(lookup[(offense, leg)] for leg in PERSON_LEGS)  # type: ignore[misc]
        for offense in ENSEMBLE_OFFENSES
    }


def larceny_hybrid_weights(weights: pd.DataFrame) -> dict[str, float]:
    hybrid = weights[weights["normalizer_id"].eq(LARCENY_OPPORTUNITY_NORMALIZER_ID)]
    return {str(row.leg): float(row.weight) for row in hybrid.itertuples()}


def provisional_offenses(weights: pd.DataFrame) -> tuple[str, ...]:
    provisional = weights.loc[weights["provisional"], "offense"].astype(str).unique().tolist()
    return tuple(sorted(provisional))


# --- surface arithmetic ------------------------------------------------------------------


def _nonnegative(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def repair_landscan_coverage(
    *,
    night: np.ndarray,
    day: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Substitute the pipeline's own person proxy where LandScan carries no mass at all.

    A cell with residents (or workers) and *zero LandScan mass on both surfaces* is a LandScan
    coverage hole, not a statement that nobody is there -- and robbery's adopted mix puts all its
    weight on the two LandScan legs, so an unrepaired hole would publish a zero denominator for a
    populated block group. Both legs take `daytime_population_jobs_proxy`, which is the nearest
    person-scale surface the pipeline has and is floored at resident population.

    The repair set is defined by `night + day == 0`, which is invariant to any positive rescaling
    of either leg: no perturbation of the inputs can move a cell in or out of it. That is what
    separates this from the hard max it replaces, whose selection set moves with a 20% recalibration
    of either surface.
    """
    night_values = _nonnegative(night)
    day_values = _nonnegative(day)
    fallback_values = _nonnegative(fallback)
    repaired = (night_values + day_values) <= 0.0
    return (
        np.where(repaired, fallback_values, night_values),
        np.where(repaired, fallback_values, day_values),
        repaired,
    )


def combine_person_legs(legs: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    """The 3-way convex ensemble, in level space, in `PERSON_LEGS` order."""
    if len(weights) != len(PERSON_LEGS):
        raise ValueError(f"expected {len(PERSON_LEGS)} person-leg weights, got {len(weights)}")
    if abs(float(sum(weights)) - 1.0) > 1e-9:
        raise ValueError(f"person-leg weights {weights} are not a convex combination")
    total = np.zeros(len(next(iter(legs.values()))), dtype=float)
    for weight, leg in zip(weights, PERSON_LEGS, strict=True):
        if weight == 0.0:
            continue
        total = total + float(weight) * _nonnegative(legs[leg])
    return total


def rescale_to_total(values: np.ndarray, *, target_total: float) -> tuple[np.ndarray, float]:
    """One positive scalar. It changes the level a rate is quoted in and nothing else."""
    array = _nonnegative(values)
    total = float(array.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("cannot rescale a surface whose reference-universe total is not positive")
    scale = float(target_total) / total
    return array * scale, scale


def compose_larceny_hybrid(
    *,
    person: np.ndarray,
    parts: dict[str, np.ndarray],
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """0.5 person ensemble + 0.3 destination POI + 0.2 retail jobs, dimensionally reconciled.

    A POI count is not a person, so each opportunity part is first rescaled to the person
    ensemble's reference-universe total. With every part carrying the same total and the weights
    summing to 1, the hybrid's total equals the person ensemble's -- the conservation property the
    final reference-total rescale then only has to apply once.
    """
    person_values = _nonnegative(person)
    person_total = float(person_values.sum())
    if person_total <= 0.0:
        raise ValueError("larceny person ensemble has no mass over the reference universe")
    hybrid = float(weights[LARCENY_PERSON_PART]) * person_values
    part_scales: dict[str, float] = {LARCENY_PERSON_PART: 1.0}
    for part in LARCENY_HYBRID_PARTS:
        if part == LARCENY_PERSON_PART:
            continue
        weight = float(weights[part])
        if weight == 0.0:
            part_scales[part] = 0.0
            continue
        scaled, scale = rescale_to_total(parts[part], target_total=person_total)
        part_scales[part] = scale
        hybrid = hybrid + weight * scaled
    return hybrid, part_scales


# --- the build ---------------------------------------------------------------------------


def _reference_frame(*, paths: RepoPaths, config: ExposureEnsembleConfig) -> pd.DataFrame:
    """The pipeline's own denominator surfaces, built the way the allocation build builds them."""
    from crimerisk.denominators import add_offense_denominators, attach_landscan_night_population
    from crimerisk.model_surface import build_bg_feature_frame

    residential_source = _validate_residential_leg_source(config.residential_leg_source)
    if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE:
        if config.enable_qcew_exposure_updating:
            raise ValueError(
                "Census residential substitution over a QCEW-updated base requires an explicit "
                "versioned base artifact and is not enabled in this arm"
            )
        baseline_path = config.baseline_normalizers_path or exposure_normalizers_path(
            paths, year=int(config.year)
        )
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Census residential arm requires frozen baseline normalizers: {baseline_path}"
            )
        baseline = pd.read_parquet(baseline_path)
        assert_exposure_normalizer_invariants(baseline)
        bg = baseline[
            [
                "state_fips",
                "bg_id",
                "population",
                "landscan_night_leg",
                "landscan_day_leg",
                "daytime_jobs_leg",
                LANDSCAN_COVERAGE_REPAIR_COLUMN,
                "destination_poi_total",
                "lodes_retail_jobs",
            ]
        ].copy()
        bg = attach_landscan_night_population(bg, paths=paths)
        bg[LANDSCAN_DAY_POP_COLUMN] = bg["landscan_day_leg"]
        bg["daytime_population_jobs_proxy"] = bg["daytime_jobs_leg"]
        bg["vehicle_exposure_2024"] = 0.0
        bg["exposure_proxy_2024"] = bg["population"]
        bg.attrs["frozen_baseline_normalizers_path"] = str(baseline_path)
        return bg

    bg = add_offense_denominators(
        build_bg_feature_frame(paths=paths, year=int(config.year)),
        paths=paths,
        year=int(config.year),
    )
    # Restricted to the published universe BEFORE the coverage check: the LandScan USA extract is
    # CONUS+DC (238,193 block groups) and the raw feature frame is not, so the 4,142-row gap is
    # release scope, not missing data. The reference universe whose resident total each normalizer
    # preserves is this one.
    excluded = {str(value).zfill(2) for value in config.excluded_state_fips}
    bg = bg[~bg["state_fips"].astype("string").str.zfill(2).isin(excluded)].reset_index(drop=True)
    bg = attach_landscan_night_population(bg, paths=paths)
    missing_night = int(pd.to_numeric(bg[LANDSCAN_NIGHT_POP_COLUMN], errors="coerce").isna().sum())
    if missing_night:
        raise ValueError(
            f"{missing_night} block groups in the published universe have no LandScan row; the "
            "night leg cannot be distinguished from a genuinely empty cell, so this lane fails "
            "closed rather than filling zeros."
        )
    return bg


def _resolve_qcew_update(
    bg: pd.DataFrame, *, paths: RepoPaths, config: ExposureEnsembleConfig
) -> pd.DataFrame:
    """The per-block-group factor frame, or the fail-closed error that says which input is missing."""
    if "lodes_year" not in bg.columns:
        raise ValueError(
            "the QCEW exposure-updating lane needs the per-block-group LODES vintage "
            "(`lodes_year`) and the feature frame does not carry it; it fails closed rather than "
            "assuming one vintage for the whole country."
        )
    vintage = pd.to_numeric(bg["lodes_year"], errors="coerce")
    # A block group with no LODES row has no LODES-derived mass to move, so its factor is the
    # identity and that is a fact rather than a fallback. Asserted, not assumed: if a vintage-less
    # cell ever carries jobs or retail, the lane would be silently declining to update real mass.
    unvintaged = vintage.isna()
    if unvintaged.any():
        carried = sum(
            float(_nonnegative(bg.loc[unvintaged, column]).sum())
            for column in ("jobs_wac", "jobs_rac", "lodes_retail_jobs")
        )
        if carried > 0.0:
            raise ValueError(
                f"{int(unvintaged.sum())} block groups carry LODES mass with no LODES vintage; "
                "the scale factor is undefined and this lane will not guess one."
            )
    present = {int(value) for value in vintage.dropna().unique()}
    unknown = sorted(present - set(LODES_VINTAGE_YEARS))
    if unknown:
        raise ValueError(
            f"the prepared LODES extract carries vintage(s) {unknown} that this lane has no "
            f"declared QCEW year for. Add them to qcew_exposure.LODES_VINTAGE_YEARS and run "
            f"`python scripts/pull/pull_qcew_county_employment.py`."
        )
    factors = build_scale_factor_table(
        paths=paths, source_years=tuple(sorted(present)), config=config.qcew
    )
    return attach_scale_factors(
        bg[["bg_id"]], factors=factors, lodes_vintage=vintage, config=config.qcew
    )


def build_exposure_normalizers(
    *, paths: RepoPaths, config: ExposureEnsembleConfig = ExposureEnsembleConfig()
) -> tuple[pd.DataFrame, dict]:
    weights = load_ensemble_weights(config.weights_path or ensemble_weights_path(paths))
    leg_weights = person_leg_weights(weights)
    hybrid_weights = larceny_hybrid_weights(weights)
    bg = _reference_frame(paths=paths, config=config)
    qcew_updated = bool(config.enable_qcew_exposure_updating)
    residential_source = _validate_residential_leg_source(config.residential_leg_source)

    population = _nonnegative(bg["population"])
    reference_total = float(population.sum())
    if reference_total <= 0.0:
        raise ValueError("reference universe has no resident population")

    raw_night = _nonnegative(bg[LANDSCAN_NIGHT_POP_COLUMN])
    raw_day = _nonnegative(bg[LANDSCAN_DAY_POP_COLUMN])
    retail = _nonnegative(bg["lodes_retail_jobs"])
    applied: pd.DataFrame | None = None
    if qcew_updated:
        # The two LODES-derived legs, and only those, move to the target year. LandScan supplies
        # its own vintage and the resident term is already at the release vintage, so neither is
        # touched: this lane updates LODES levels, it does not update the exposure surface.
        applied = _resolve_qcew_update(bg, paths=paths, config=config)
        jobs = scale_daytime_jobs_leg(
            population=population,
            jobs_wac=_nonnegative(bg["jobs_wac"]),
            jobs_rac=_nonnegative(bg["jobs_rac"]),
            factor=applied[f"qcew_{TOTAL_LEG}_scale_factor"].to_numpy(dtype=float),
        )
        retail = scale_retail_leg(
            retail_jobs=retail,
            factor=applied[f"qcew_{RETAIL_LEG}_scale_factor"].to_numpy(dtype=float),
        )
    else:
        jobs = _nonnegative(bg["daytime_population_jobs_proxy"])
    if "frozen_baseline_normalizers_path" in bg.attrs:
        night = _nonnegative(bg["landscan_night_leg"])
        day = _nonnegative(bg["landscan_day_leg"])
        repaired = bg[LANDSCAN_COVERAGE_REPAIR_COLUMN].fillna(False).astype(bool).to_numpy()
    else:
        night, day, repaired = repair_landscan_coverage(
            night=raw_night, day=raw_day, fallback=jobs
        )
    selected_residential = (
        population
        if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
        else night
    )
    legs = {
        "landscan_night": selected_residential,
        "landscan_day": day,
        "daytime_jobs": jobs,
    }

    out = pd.DataFrame(
        {
            "state_fips": bg["state_fips"].astype("string").str.zfill(2),
            "bg_id": bg["bg_id"].astype("string").str.zfill(12),
            "population": population,
            "landscan_night_leg": night,
            "landscan_day_leg": day,
            "daytime_jobs_leg": jobs,
            LANDSCAN_COVERAGE_REPAIR_COLUMN: repaired,
            "destination_poi_total": _nonnegative(bg["destination_poi_total"]),
            "lodes_retail_jobs": retail,
        }
    )
    if applied is not None:
        for column in QCEW_UPDATING_COLUMNS:
            out[column] = applied[column].to_numpy()
    if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE:
        out["raw_landscan_night_pop"] = raw_night
        out["residential_leg"] = selected_residential
        out["residential_leg_source"] = residential_source

    baseline_derived = "frozen_baseline_normalizers_path" in bg.attrs
    deployed = None if baseline_derived else _nonnegative(bg["exposure_proxy_2024"])
    offense_summary: dict[str, dict[str, float | int | str | list[float] | None]] = {}
    for offense in ENSEMBLE_OFFENSES:
        person = combine_person_legs(legs, leg_weights[offense])
        part_scales: dict[str, float] = {}
        if offense == "larceny":
            unscaled, part_scales = compose_larceny_hybrid(
                person=person,
                # Read from `out`, not `bg`: the retail part is the published leg, which is the
                # QCEW-updated one when the lane is on. Reading `bg` here would scale the leg in
                # the artifact and leave larceny's hybrid on the unscaled vintage.
                parts={
                    part: _nonnegative(out[column] if column in out.columns else bg[column])
                    for part, column in LARCENY_HYBRID_SOURCE_COLUMNS.items()
                },
                weights=hybrid_weights,
            )
        else:
            unscaled = person
        scaled, scale = rescale_to_total(unscaled, target_total=reference_total)
        out[opportunity_normalizer_column(offense)] = scaled
        positive = scaled > 0.0
        offense_summary[offense] = {
            "normalizer_id": normalizer_id_for_offense(offense),
            "person_leg_weights": [float(value) for value in leg_weights[offense]],
            "unscaled_total": float(unscaled.sum()),
            "reference_total_scale": float(scale),
            "zero_normalizer_block_groups": int((~positive).sum()),
            "zero_normalizer_with_population": int(((~positive) & (population > 0.0)).sum()),
            "below_publication_floor_block_groups": int((scaled < 50.0).sum()),
            "min_positive": float(scaled[positive].min()) if positive.any() else float("nan"),
            "median_ratio_to_deployed": (
                None
                if deployed is None
                else float(np.median(scaled[deployed > 0.0] / deployed[deployed > 0.0]))
            ),
        }
        if part_scales:
            offense_summary[offense]["hybrid_part_weights"] = {
                part: float(hybrid_weights[part]) for part in LARCENY_HYBRID_PARTS
            }
            offense_summary[offense]["hybrid_part_reference_scales"] = {
                part: float(value) for part, value in part_scales.items()
            }

    out = out.reindex(
        columns=exposure_normalizer_columns(
            qcew_updated=qcew_updated,
            residential_leg_source=residential_source,
        )
    )
    out = out.sort_values("bg_id", kind="mergesort").reset_index(drop=True)
    assert_exposure_normalizer_invariants(out, reference_total=reference_total, weights=weights)

    summary = {
        "year": int(config.year),
        "normalizer_version": exposure_normalizer_version(
            residential_leg_source=residential_source
        ),
        "semantics": normalizer_semantics(qcew_updated=qcew_updated),
        "provenance": "E3 exposure ensemble (analysis_scratch/final_phase/e3_exposure/01_ensemble_test.py)",
        "contract": "analysis_scratch/final_phase/EXPOSURE_ENSEMBLE_CONTRACT.md",
        "weights_path": str(config.weights_path or ensemble_weights_path(paths)),
        "frozen_baseline_normalizers_path": bg.attrs.get(
            "frozen_baseline_normalizers_path"
        ),
        "provisional_offenses": list(provisional_offenses(weights)),
        "block_groups": int(len(out)),
        "excluded_state_fips": list(config.excluded_state_fips),
        "reference_universe": "published block-group universe of the edition (CONUS + DC)",
        "reference_total_policy": "each normalizer is rescaled so its reference-universe total equals resident population",
        "residential_leg": {
            "source": residential_source,
            "selected_column": (
                "residential_leg"
                if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
                else "landscan_night_leg"
            ),
            "frozen_weight_slot": "landscan_night",
            "nationwide": True,
            "local_exceptions": [],
            "release_population_column": "population",
            "raw_landscan_column": (
                "raw_landscan_night_pop"
                if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
                else None
            ),
            "ambient_blind_custom_footprint_gate": (
                "not_applied_to_person_offenses_in_census_residential_arm"
                if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
                else "applied_by_existing_landscan_person_exposure_policy"
            ),
        },
        "reference_resident_population_total": reference_total,
        "deployed_person_exposure_total": (
            None if deployed is None else float(deployed.sum())
        ),
        "landscan_coverage_repaired_block_groups": int(repaired.sum()),
        "landscan_coverage_repaired_with_population": int((repaired & (population > 0.0)).sum()),
        "legacy_below_publication_floor_block_groups": (
            None if deployed is None else int((deployed < 50.0).sum())
        ),
        "unchanged_offenses": {
            "burglary": BURGLARY_PREMISES_NORMALIZER_ID,
            "motor_vehicle_theft": MVT_VEHICLE_NORMALIZER_ID,
        },
        "offenses": offense_summary,
        "qcew_exposure_updating": _qcew_summary_block(
            paths=paths, config=config, applied=applied
        ),
        "ags_values_used": False,
    }
    return out, summary


def _qcew_summary_block(
    *,
    paths: RepoPaths,
    config: ExposureEnsembleConfig,
    applied: pd.DataFrame | None,
) -> dict:
    """PLAN item 6's provenance block: what was scaled, from what vintage, by how much."""
    if applied is None:
        return {"enabled": False}
    vintages = tuple(
        sorted({int(value) for value in applied["lodes_vintage"].dropna().astype(int).unique()})
    )
    factors = build_scale_factor_table(
        paths=paths, source_years=vintages, config=config.qcew
    )
    return {
        "enabled": True,
        "scale_version": QCEW_SCALE_VERSION,
        "contract": "analysis_scratch/final_phase/QCEW_EXPOSURE_CONTRACT.md",
        "semantics": normalizer_semantics(qcew_updated=True),
        "scaled_legs": {
            TOTAL_LEG: "daytime_jobs_leg (LODES net-commuter term)",
            RETAIL_LEG: "lodes_retail_jobs (LODES CNS07; larceny hybrid leg)",
        },
        "lodes_vintages": list(vintages),
        "qcew_source_paths": [
            str(path)
            for path in qcew_dependency_paths(paths, config=config)
            if path.exists()
        ],
        "factor_distribution": summarize_scale_factors(factors, config=config.qcew),
        "applied": summarize_applied_factors(applied),
    }


def is_qcew_updated(normalizers: pd.DataFrame) -> bool:
    """A QCEW-updated table is self-identifying: it carries the provenance columns, or it is not one."""
    present = [column for column in QCEW_UPDATING_COLUMNS if column in normalizers.columns]
    if present and len(present) != len(QCEW_UPDATING_COLUMNS):
        missing = sorted(set(QCEW_UPDATING_COLUMNS) - set(present))
        raise ValueError(
            f"exposure normalizer table carries a partial QCEW provenance schema; missing {missing}"
        )
    return bool(present)


def residential_leg_source(normalizers: pd.DataFrame) -> str:
    """Read the residential source from the artifact schema and fail on partial provenance."""
    present = [column for column in CENSUS_RESIDENTIAL_COLUMNS if column in normalizers.columns]
    if not present:
        return LANDSCAN_NIGHT_RESIDENTIAL_SOURCE
    if len(present) != len(CENSUS_RESIDENTIAL_COLUMNS):
        missing = sorted(set(CENSUS_RESIDENTIAL_COLUMNS) - set(present))
        raise ValueError(
            f"exposure normalizer table carries partial residential provenance; missing {missing}"
        )
    sources = normalizers["residential_leg_source"].astype(str).unique().tolist()
    if sources != [CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE]:
        raise ValueError(
            "exposure normalizer table carries unexpected residential source values: "
            f"{sorted(sources)}"
        )
    return CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE


def _assert_qcew_columns(normalizers: pd.DataFrame, *, tolerance: float = 1e-12) -> None:
    if not is_qcew_updated(normalizers):
        return
    vintage = pd.to_numeric(normalizers["lodes_vintage"], errors="coerce")
    unknown_vintage = sorted(set(vintage.dropna().astype(int).unique()) - set(LODES_VINTAGE_YEARS))
    if unknown_vintage:
        raise ValueError(
            f"exposure normalizer table carries undeclared LODES vintage(s) {unknown_vintage}"
        )
    scale_vintage = pd.to_numeric(normalizers["qcew_scale_vintage"], errors="coerce")
    if scale_vintage.isna().any() or scale_vintage.nunique() != 1:
        raise ValueError(
            "exposure normalizer table must carry exactly one QCEW scale vintage; the target year "
            "is a property of the edition, not of a block group"
        )
    for leg in QCEW_SCALED_LEGS:
        factor = pd.to_numeric(normalizers[f"qcew_{leg}_scale_factor"], errors="coerce")
        if factor.isna().any() or not np.isfinite(factor.to_numpy(dtype=float)).all():
            raise ValueError(f"exposure normalizer table carries a nonfinite qcew {leg} factor")
        if (
            factor.lt(QCEW_SCALE_FACTOR_FLOOR - tolerance).any()
            or factor.gt(QCEW_SCALE_FACTOR_CEILING + tolerance).any()
        ):
            raise ValueError(
                f"exposure normalizer table carries a qcew {leg} factor outside the band "
                f"[{QCEW_SCALE_FACTOR_FLOOR}, {QCEW_SCALE_FACTOR_CEILING}]"
            )
        tiers = normalizers[f"qcew_{leg}_scale_tier"].astype(str)
        unknown_tier = sorted(set(tiers) - set(QCEW_SCALE_TIERS))
        if unknown_tier:
            raise ValueError(
                f"exposure normalizer table carries unknown qcew {leg} tier(s) {unknown_tier}"
            )
        identity = tiers.eq(QCEW_TIER_IDENTITY)
        if identity.any() and not np.allclose(factor[identity].to_numpy(dtype=float), 1.0):
            raise ValueError(
                f"exposure normalizer table has an identity-tier qcew {leg} row whose factor is "
                "not 1.0"
            )
        # A block group with no LODES row has no vintage to scale FROM, so it must sit on the
        # identity rung. Anything else would be a factor applied to an unknown baseline.
        unvintaged_scaled = vintage.isna() & ~identity
        if unvintaged_scaled.any():
            raise ValueError(
                f"exposure normalizer table scales the {leg} leg on "
                f"{int(unvintaged_scaled.sum())} block groups that carry no LODES vintage"
            )


def assert_exposure_normalizer_invariants(
    normalizers: pd.DataFrame,
    *,
    reference_total: float | None = None,
    weights: pd.DataFrame | None = None,
    tolerance: float = 1e-9,
) -> None:
    missing = sorted(set(EXPOSURE_NORMALIZER_COLUMNS) - set(normalizers.columns))
    if missing:
        raise ValueError(f"exposure normalizer table is missing columns {missing}")
    if normalizers["bg_id"].duplicated().any():
        raise ValueError("exposure normalizer table carries duplicate bg_id keys")
    _assert_qcew_columns(normalizers)
    residential_source = residential_leg_source(normalizers)

    population = _nonnegative(normalizers["population"])
    numeric_columns = [
        "population",
        *PERSON_LEG_FRAME_COLUMNS.values(),
        "destination_poi_total",
        "lodes_retail_jobs",
        *[opportunity_normalizer_column(offense) for offense in ENSEMBLE_OFFENSES],
    ]
    if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE:
        numeric_columns.extend(["raw_landscan_night_pop", "residential_leg"])
    for column in numeric_columns:
        values = pd.to_numeric(normalizers[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"exposure normalizer table carries a nonfinite {column}")
        if values.lt(-tolerance).any():
            raise ValueError(f"exposure normalizer table carries a negative {column}")

    repaired = normalizers[LANDSCAN_COVERAGE_REPAIR_COLUMN].fillna(False).astype(bool).to_numpy()
    if repaired.any():
        night = _nonnegative(normalizers["landscan_night_leg"])[repaired]
        day = _nonnegative(normalizers["landscan_day_leg"])[repaired]
        jobs = _nonnegative(normalizers["daytime_jobs_leg"])[repaired]
        if not (np.allclose(night, jobs, atol=tolerance) and np.allclose(day, jobs, atol=tolerance)):
            raise ValueError(
                "exposure normalizer table flags a LandScan coverage repair whose legs are not the "
                "daytime person proxy"
            )

    if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE:
        residential = _nonnegative(normalizers["residential_leg"])
        raw_night = _nonnegative(normalizers["raw_landscan_night_pop"])
        if not np.allclose(residential, population, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Census residential leg must equal release resident population nationwide"
            )
        unrepaired = ~repaired
        if unrepaired.any() and not np.allclose(
            _nonnegative(normalizers["landscan_night_leg"])[unrepaired],
            raw_night[unrepaired],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "Census residential artifact does not retain raw LandScan night values"
            )

    for offense in ENSEMBLE_OFFENSES:
        values = _nonnegative(normalizers[opportunity_normalizer_column(offense)])
        # Strict positivity wherever anybody is there. A normalizer that reports no exposure for a
        # populated block group would suppress that cell's rate on a data artifact.
        dead_with_people = (values <= 0.0) & (population > 0.0)
        if dead_with_people.any():
            raise ValueError(
                f"exposure normalizer {offense} is zero on {int(dead_with_people.sum())} block "
                "groups that have resident population"
            )
        if reference_total is not None:
            total = float(values.sum())
            if abs(total - float(reference_total)) > 1e-6 * float(reference_total):
                raise ValueError(
                    f"exposure normalizer {offense} totals {total:.1f} over the reference universe, "
                    f"expected the resident total {float(reference_total):.1f}"
                )

    if weights is not None:
        legs = {
            name: _nonnegative(normalizers[column])
            for name, column in PERSON_LEG_FRAME_COLUMNS.items()
        }
        if residential_source == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE:
            legs["landscan_night"] = _nonnegative(normalizers["residential_leg"])
        leg_weights = person_leg_weights(weights)
        for offense in ENSEMBLE_OFFENSES:
            if offense == "larceny":
                continue
            recomposed = combine_person_legs(legs, leg_weights[offense])
            published = _nonnegative(normalizers[opportunity_normalizer_column(offense)])
            scale = float(published.sum()) / float(recomposed.sum())
            if not np.allclose(published, recomposed * scale, rtol=1e-9, atol=1e-6):
                raise ValueError(
                    f"exposure normalizer {offense} does not recompose from its published legs at "
                    "the frozen weights"
                )


def write_v2_exposure_normalizers(
    *,
    paths: RepoPaths,
    out_path: Path | None = None,
    config: ExposureEnsembleConfig = ExposureEnsembleConfig(),
    force: bool = False,
) -> tuple[Path, dict]:
    qcew_updated = bool(config.enable_qcew_exposure_updating)
    residential_source = _validate_residential_leg_source(config.residential_leg_source)
    artifact = out_path or exposure_normalizers_path(
        paths,
        year=int(config.year),
        qcew_updated=qcew_updated,
        residential_leg_source=residential_source,
    )
    summary_path = artifact.with_suffix(".summary.json")
    if not force and exposure_normalizers_artifact_is_current(paths, config=config, out_path=artifact):
        return artifact, json.loads(summary_path.read_text())
    normalizers, summary = build_exposure_normalizers(paths=paths, config=config)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    normalizers.to_parquet(artifact, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_dependency_stamp(artifact, exposure_normalizer_dependency_paths(paths, config=config))
    return artifact, summary


def load_exposure_normalizers(
    paths: RepoPaths,
    *,
    year: int,
    path: Path | None = None,
    qcew_updated: bool = False,
    residential_source: str = LANDSCAN_NIGHT_RESIDENTIAL_SOURCE,
) -> pd.DataFrame:
    source = _validate_residential_leg_source(residential_source)
    artifact = path or exposure_normalizers_path(
        paths,
        year=int(year),
        qcew_updated=bool(qcew_updated),
        residential_leg_source=source,
    )
    if not artifact.exists():
        raise FileNotFoundError(
            f"{artifact} is required by the v2 exposure-ensemble path and is absent. Run "
            f"`build-exposure-normalizers` first."
        )
    normalizers = pd.read_parquet(artifact)
    normalizers["bg_id"] = normalizers["bg_id"].astype("string").str.zfill(12)
    assert_exposure_normalizer_invariants(normalizers)
    return normalizers


# --- application to the rate/index lane ---------------------------------------------------


@dataclass(frozen=True)
class ExposureEnsembleRuntime:
    """Everything the allocation build needs to publish ensemble normalizers, resolved once."""

    normalizers: pd.DataFrame
    weights: pd.DataFrame
    config: ExposureEnsembleConfig

    @property
    def offenses(self) -> tuple[str, ...]:
        return ENSEMBLE_OFFENSES

    @property
    def columns(self) -> list[str]:
        return [opportunity_normalizer_column(offense) for offense in ENSEMBLE_OFFENSES]

    @property
    def qcew_updated(self) -> bool:
        """Read off the loaded surface, not off the config.

        The build that consumes a normalizer artifact does not have to know which flag produced it;
        an artifact whose levels were moved to the target year says so in its own schema, and the
        manifest quotes the artifact rather than the caller's intent.
        """
        return is_qcew_updated(self.normalizers)

    @property
    def residential_leg_source(self) -> str:
        return residential_leg_source(self.normalizers)

    @property
    def normalizer_version(self) -> str:
        return exposure_normalizer_version(
            residential_leg_source=self.residential_leg_source
        )


def resolve_exposure_ensemble_runtime(
    *,
    paths: RepoPaths,
    config: ExposureEnsembleConfig = ExposureEnsembleConfig(),
    normalizers_path: Path | None = None,
    weights_path: Path | None = None,
) -> ExposureEnsembleRuntime:
    return ExposureEnsembleRuntime(
        normalizers=load_exposure_normalizers(
            paths,
            year=int(config.year),
            path=normalizers_path,
            qcew_updated=bool(config.enable_qcew_exposure_updating),
            residential_source=config.residential_leg_source,
        ),
        weights=load_ensemble_weights(
            weights_path or config.weights_path or ensemble_weights_path(paths)
        ),
        config=config,
    )


def attach_exposure_normalizers(
    bg_covariates: pd.DataFrame, *, runtime: ExposureEnsembleRuntime
) -> pd.DataFrame:
    """Join the named normalizer surfaces onto the block-group covariate frame, failing closed.

    A block group with no normalizer row would fall through to a zero denominator and silently
    lose its rate, so a gap raises instead. Coverage is required over the published universe only:
    the covariate frame is built on the raw block-group feature frame, which carries the states
    the release excludes (AK/HI/PR), and those rows are dropped by the crosswalk join downstream.
    """
    out = bg_covariates.copy()
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    surface = runtime.normalizers[["bg_id", *runtime.columns]].copy()
    surface["bg_id"] = surface["bg_id"].astype("string").str.zfill(12)
    merged = out.merge(surface, on="bg_id", how="left", validate="one_to_one")
    excluded = {str(value).zfill(2) for value in runtime.config.excluded_state_fips}
    in_scope = ~merged["state_fips"].astype("string").str.zfill(2).isin(excluded)
    gaps = merged[runtime.columns].isna().any(axis=1) & in_scope
    if bool(gaps.any()):
        raise ValueError(
            f"{int(gaps.sum())} block groups in the published universe have no exposure "
            "normalizer row; the v2 lane fails closed rather than publishing a zero denominator."
        )
    return merged


def _qcew_manifest_block(runtime: ExposureEnsembleRuntime) -> dict:
    """The vintage/provenance record the manifest carries, read off the normalizer surface."""
    if not runtime.qcew_updated:
        return {"enabled": False}
    normalizers = runtime.normalizers
    return {
        "enabled": True,
        "scale_version": QCEW_SCALE_VERSION,
        "contract": "analysis_scratch/final_phase/QCEW_EXPOSURE_CONTRACT.md",
        "scaled_legs": list(QCEW_SCALED_LEGS),
        "band": [QCEW_SCALE_FACTOR_FLOOR, QCEW_SCALE_FACTOR_CEILING],
        **summarize_applied_factors(normalizers),
    }


def summarize_exposure_normalizers(
    frame: pd.DataFrame, *, runtime: ExposureEnsembleRuntime
) -> dict:
    """What the build applied, for the manifest. A record, not a second derivation."""
    qcew_updated = runtime.qcew_updated
    summary: dict[str, object] = {
        "enabled": True,
        "normalizer_version": runtime.normalizer_version,
        "semantics": normalizer_semantics(qcew_updated=qcew_updated),
        "residential_leg": {
            "source": runtime.residential_leg_source,
            "selected_column": (
                "residential_leg"
                if runtime.residential_leg_source
                == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
                else "landscan_night_leg"
            ),
            "frozen_weight_slot": "landscan_night",
            "nationwide": True,
            "local_exceptions": [],
            "release_population_column": "population",
            "raw_landscan_column": (
                "raw_landscan_night_pop"
                if runtime.residential_leg_source
                == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
                else None
            ),
            "ambient_blind_custom_footprint_gate": (
                "not_applied_to_person_offenses_in_census_residential_arm"
                if runtime.residential_leg_source
                == CENSUS_RELEASE_POPULATION_RESIDENTIAL_SOURCE
                else "applied_by_existing_landscan_person_exposure_policy"
            ),
        },
        "provisional_offenses": list(provisional_offenses(runtime.weights)),
        "rows": int(len(frame)),
        "qcew_exposure_updating": _qcew_manifest_block(runtime),
        "offenses": {},
    }
    for offense in OFFENSES_7:
        entry: dict[str, object] = {"normalizer_id": normalizer_id_for_offense(offense)}
        column = opportunity_normalizer_column(offense)
        if column in frame.columns:
            values = _nonnegative(frame[column])
            entry["total"] = float(values.sum())
            entry["zero_rows"] = int((values <= 0.0).sum())
        summary["offenses"][offense] = entry  # type: ignore[index]
    return summary
