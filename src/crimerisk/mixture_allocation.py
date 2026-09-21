"""The three-expert mixture allocator (v2 model lane).

`allocation.py` builds one within-jurisdiction share vector per `(jurisdiction_id, state_fips,
offense)` and rakes it to that jurisdiction's official total. The legacy vector is the
incident-trained cold-start prior alone. This module replaces that vector -- and only that
vector -- with a convex mixture of three experts, selected per offense x jurisdiction-size
stratum by the E2 tournament:

* **b** -- the incident-trained prior (`bg_weight`, already in the merged support),
* **g** -- a bounded downward GBM over the governed covariates, trained here at build time on
  FBI-panel jurisdiction rates only and clipped into a learned intensity envelope,
* **e** -- exposure: where the offense's opportunity mass is.

EXPERT `e` IS NOT RESIDENTIAL POPULATION. The E2 tournament built it as `norm(pop.clip(1))`, and
`certification/special_use_rank_diagnosis.md` measured what that costs: over the 32 audited
special-use-heavy benchmark cells the exposure expert put **0.0009** of its mass on non-ordinary
block groups against a truth of **0.2406**, because parks, industrial tracts, prisons, campuses and
transient destinations are precisely the block groups whose residential headcount is ~0. Mixing it
in was therefore pure dilution of the one expert that carries any special-use signal, and it
accounted for 77% of the mixture lane's rank loss in that stratum. The expert is now normalised
over the v2 **opportunity normalizer** for the offense (`exposure_ensemble`'s `person_ens_v1` /
`larceny_opp_v1`, QCEW-scaled when that lane is on) -- the surface the repo already builds to
answer exactly the question the expert's name asks. `exposure_basis` on the expert table names
which surface each row was built from, and a build that is handed a table built on the other basis
fails closed, because the shipped weights were selected for a specific expert.

Provenance is `analysis_scratch/final_phase/e2_allocation/07_final_mixture.py` (architecture,
adopted per PLAN.md Amendment 3 items 1-2), `08_ship_weights.py` (the first full-benchmark weight
selection), `10_rape_reselect.py` (rape, re-selected on three metrics after the E2 scorecard) and
`14_reselect_weights_v3.py` (the one final re-selection under the re-based exposure expert, every
offense, nested leave-one-source-out on a composite of TVD + Spearman + exact spatial skill). The
shipped table is frozen at `configs/mixture_ship_weights_v3.csv` and each row records which
selection produced it in `selection_version`. The contract for this lane is
`analysis_scratch/final_phase/MIXTURE_ALLOCATOR_CONTRACT.md`.

Two rules are load-bearing and are enforced rather than trusted:

* **Weights are read, never re-derived.** Selection happened once, against a frozen benchmark
  registry. A build step that re-selected weights would put selection inside the shipped
  artifact, which is exactly the benchmark drift Amendment 3 item 2 froze the registry to stop.
* **No AGS value enters anywhere.** The GBM trains on the jurisdiction-year panel; the weights
  came from a tournament scored against local incident feeds. Nothing in this lane reads an AGS
  surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.build_freshness import artifact_is_current, write_dependency_stamp
from crimerisk.crime import OFFENSES_7
from crimerisk.exposure_ensemble import (
    ENSEMBLE_OFFENSES,
    exposure_normalizers_path,
    normalizer_id_for_offense,
    opportunity_normalizer_column,
)
from crimerisk.paths import RepoPaths


# --- frozen constants (E2 tournament outputs; none is fitted at build time) ---------------

# 07_final_mixture.py, `stratum()`: predeclared jurisdiction-size strata.
CORE_POPULATION_FLOOR = 200_000.0
SUBURBAN_POPULATION_FLOOR = 50_000.0

CORE_STRATUM = "core"
SUBURBAN_STRATUM = "suburban"
SMALL_STRATUM = "small"
STRATA = (CORE_STRATUM, SUBURBAN_STRATUM, SMALL_STRATUM)

WEIGHT_COLUMNS = ("w_prior", "w_gbm", "w_exposure")
SHIP_WEIGHTS_FILENAME = "mixture_ship_weights_v3.csv"

# 07_final_mixture.py, the GBM expert. Reproduced exactly, including the seed.
GBM_MAX_ITER = 150
GBM_MAX_DEPTH = 4
GBM_RANDOM_STATE = 0
ENVELOPE_PERCENTILES = (0.5, 99.5)

# `cb["pop"].clip(lower=1.0)` -- the exposure floor the tournament applied to both the exposure
# and GBM experts. It is NOT applied to the stratum population, which is a real headcount. The
# re-based exposure expert keeps the same device on its own quantity: a block group with no
# opportunity mass gets the floor rather than an exact zero, which is what stops a footprint whose
# normalizer is degenerate everywhere from having no exposure expert at all.
EXPOSURE_FLOOR = 1.0

# --- the exposure expert's basis (special-use rank regression, CONSTANT_RATIFICATIONS.md) -----
#
# `e` means "where the exposure is". For the five offenses whose opportunity normalizer the v2
# exposure lane actually builds, that surface IS the measurement, and it is what the expert reads.
# Burglary and motor vehicle theft keep the residential-population basis: their published
# normalizers (`premises_nnls_v1`, `vehicle_exposure_v1`) are assembled inside `allocation.py` from
# denominator inputs this module does not read, so wiring them is a separate change with its own
# evidence, not something to infer here. That fallback is per OFFENSE and is declared on every row.
#
# There is a second, per-ROW fallback with a different cause: a block group the normalizer artifact
# does not cover, or covers with a non-finite or non-positive value. The artifact is built on the
# same published universe as the feature frame, so this is expected to be empty; it is handled and
# counted rather than assumed away, because an uncovered row would otherwise become an exact zero
# and shrink that block group out of its own footprint silently.
EXPOSURE_BASIS_POPULATION = "population"
EXPOSURE_BASIS_COLUMN = "exposure_basis"
EXPOSURE_BASIS_FALLBACK_COLUMN = "exposure_basis_row_fallback"


def exposure_basis_for_offense(offense: str) -> str:
    """The surface the exposure expert is normalised over for this offense."""
    key = str(offense)
    if key not in OFFENSES_7:
        raise KeyError(f"unknown offense for the exposure expert basis: {offense!r}")
    if key in ENSEMBLE_OFFENSES:
        return normalizer_id_for_offense(key, enabled=True)
    return EXPOSURE_BASIS_POPULATION


EXPOSURE_BASIS_BY_OFFENSE: dict[str, str] = {
    offense: exposure_basis_for_offense(offense) for offense in OFFENSES_7
}

# --- soft shrinkage (PLAN.md item 5) -------------------------------------------------------
#
# A hard clip states something the model is not entitled to state: that every prediction above the
# training envelope is EXACTLY the envelope. It destroys the ordering among precisely the cells the
# ecological expert is least sure about, and it does so discontinuously in the derivative. The soft
# replacement keeps the ordering and compresses the magnitude. On the log scale, for a prediction r
# above the bound `hi`,
#
#     excess = log(r / hi)                    >= 0
#     r'     = hi * exp(nu * log1p(excess / nu))
#
# and symmetrically below `lo`. `nu * log1p(x/nu)` is the t-like compressor: it is the identity to
# first order at x -> 0 (predictions just outside the envelope are barely touched), grows like
# nu*log(x) far out (a heavy tail is compressed, never truncated), is strictly increasing (ordering
# is preserved exactly), and recovers the two limiting behaviours the plan contrasts -- nu -> inf is
# no shrinkage at all, nu -> 0 is the hard clip. Inside the envelope the prediction is untouched.
#
# EXTRAPOLATION. The second half of the plan item is transfer distance -> shrinkage toward the
# exposure-share null. The hull proxy is per-feature: a block group whose governed feature vector
# leaves the training rows' own [P0.5, P99.5] range on ANY governed feature is flagged, and its
# compressed excess is further shrunk by `extrapolation_weight`. Shrinking the excess to zero puts
# the cell exactly on the bound, and the bound is a CONSTANT rate across the offense -- a footprint
# whose GBM rates are constant has `gbm_share == exposure_share` identically. So shrinking the
# excess is shrinking toward the exposure-share null, which is the null the plan names.
#
# Both constants were selected on the E2 SELECTION folds only (never the report folds), by nested
# leave-source-out over benchmark_registry_v1: all 20 folds independently chose nu=2 from {2,3,5}
# and weight=0.75 from {0.25,0.5,0.75}. That is the CORNER of both grids -- the most clip-like arm
# available -- and it is recorded as a boundary selection, not as evidence that the compressor's
# shape is optimal: where this benchmark speaks at all it speaks faintly in favour of more
# compression, because it cannot pay for a fatter tail. Run and numbers in
# analysis_scratch/final_phase/SOFT_SHRINKAGE_CONTRACT.md; disposition in CONSTANT_RATIFICATIONS.md.
SOFT_SHRINKAGE_NU = 2.0
SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT = 0.75
# The hull proxy reuses the envelope's own percentiles rather than introducing a second pair: the
# question "is this cell outside what the panel exhibited" is the same question in feature space as
# in rate space, and one number is one number.
EXTRAPOLATION_HULL_PERCENTILES = (0.5, 99.5)
ENVELOPE_MODE_HARD_CLIP = "hard_clip"
ENVELOPE_MODE_SOFT_SHRINKAGE = "soft_shrinkage"

# The jurisdiction-panel target screen (07's `_e2_prep.load_prep`). Deliberately looser than the
# smoothed-control lane's clean-row rule: this is the screen the tournament selected under.
PANEL_YEAR_START = 2018
MIN_TARGET_YEARS = 3
MIN_TARGET_POPULATION = 500.0

FEATURE_CATALOG_FILENAME = "bg_prior_long_{year}_arm_b_features.parquet"
PRIOR_FILENAME = "bg_prior_long_{year}_arm_b.parquet"

MIXTURE_EXPERT_COLUMNS = [
    "state_fips",
    "bg_id",
    "offense",
    "population",
    "exposure_weight",
    EXPOSURE_BASIS_COLUMN,
    EXPOSURE_BASIS_FALLBACK_COLUMN,
    "gbm_rate",
    "gbm_rate_clipped",
    "gbm_weight",
    "envelope_lo",
    "envelope_hi",
]

# Written ONLY when the soft-shrinkage lane is on, so a hard-clip expert table is byte-identical to
# what it was before this lane existed. `gbm_rate_clipped` keeps its name in both modes -- it is
# "the rate after the envelope treatment", and `envelope_mode` is what says which treatment that
# was, so a reader never has to guess from the values.
SOFT_SHRINKAGE_EXPERT_COLUMNS = [
    "envelope_mode",
    "extrapolation_flag",
    "extrapolation_feature_count",
]

MIXTURE_SHARE_AUDIT_COLUMNS = [
    "state_fips",
    "jurisdiction_id",
    "jurisdiction_type",
    "bg_id",
    "offense",
    "stratum",
    "footprint_population",
    "w_prior",
    "w_gbm",
    "w_exposure",
    "prior_share",
    "gbm_share",
    "exposure_share",
    "model_share_legacy",
    "model_share",
]

# The share-lane grouping key, mirrored from allocation.CITY_POSTERIOR_GROUP_COLS. Duplicated
# rather than imported to keep this module free of a cycle back into allocation.py.
MIXTURE_GROUP_COLUMNS = ["jurisdiction_id", "state_fips", "offense"]


@dataclass(frozen=True)
class MixtureAllocationConfig:
    """Every knob is an E2 output or a structural constant -- none is fitted at build time."""

    year: int = 2024
    panel_year_start: int = PANEL_YEAR_START
    min_target_years: int = MIN_TARGET_YEARS
    min_target_population: float = MIN_TARGET_POPULATION
    envelope_percentiles: tuple[float, float] = ENVELOPE_PERCENTILES
    gbm_max_iter: int = GBM_MAX_ITER
    gbm_max_depth: int = GBM_MAX_DEPTH
    gbm_random_state: int = GBM_RANDOM_STATE
    exposure_floor: float = EXPOSURE_FLOOR
    core_population_floor: float = CORE_POPULATION_FLOOR
    suburban_population_floor: float = SUBURBAN_POPULATION_FLOOR
    # PLAN.md item 5. OFF by default: the promoted chain and the v2 candidate keep the hard clip
    # until the owner promotes this lane. See
    # analysis_scratch/final_phase/SOFT_SHRINKAGE_CONTRACT.md.
    enable_soft_shrinkage: bool = False
    soft_shrinkage_nu: float = SOFT_SHRINKAGE_NU
    soft_shrinkage_extrapolation_weight: float = SOFT_SHRINKAGE_EXTRAPOLATION_WEIGHT
    extrapolation_hull_percentiles: tuple[float, float] = EXTRAPOLATION_HULL_PERCENTILES
    weights_path: Path | None = None
    # The exposure expert's basis. NOT a lane flag: the opportunity normalizer is what the expert
    # means, so there is no "off" position -- these two only say WHICH normalizer artifact to read.
    # `enable_qcew_exposure_updating` mirrors the exposure lane's own flag, because a QCEW-updated
    # surface is a different artifact at a different path, not a newer one.
    exposure_normalizers_path: Path | None = None
    enable_qcew_exposure_updating: bool = False
    # State FIPS outside the published CONUS+DC scope. Mirrors
    # allocation.RELEASE_EXCLUDED_STATE_FIPS; the jurisdiction panel is built on the published
    # universe so the fitted model is the one the tournament fitted.
    excluded_state_fips: tuple[str, ...] = ("02", "15", "72")


# --- paths -------------------------------------------------------------------------------


def ship_weights_path(paths: RepoPaths) -> Path:
    return paths.repo_root / "configs" / SHIP_WEIGHTS_FILENAME


def mixture_experts_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "modeling" / f"bg_mixture_experts_{int(year)}.parquet"


def mixture_experts_summary_path(paths: RepoPaths, *, year: int) -> Path:
    return mixture_experts_path(paths, year=year).with_suffix(".summary.json")


def mixture_shares_audit_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "modeling" / f"bg_mixture_shares_{int(year)}.parquet"


def jurisdiction_feature_panel_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.cache_dir / "mixture" / f"jurisdiction_feature_panel_{int(year)}.parquet"


def feature_catalog_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "modeling" / FEATURE_CATALOG_FILENAME.format(year=int(year))


def jurisdiction_year_estimates_path(paths: RepoPaths) -> Path:
    return paths.state_dir / "controls" / "jurisdiction_year_estimates.parquet"


def block_group_crosswalk_path(paths: RepoPaths) -> Path:
    return paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"


def resolve_exposure_normalizers_path(
    paths: RepoPaths, *, config: MixtureAllocationConfig
) -> Path:
    if config.exposure_normalizers_path is not None:
        return Path(config.exposure_normalizers_path)
    return exposure_normalizers_path(
        paths,
        year=int(config.year),
        qcew_updated=bool(config.enable_qcew_exposure_updating),
    )


def mixture_experts_dependency_paths(
    paths: RepoPaths, *, config: MixtureAllocationConfig
) -> list[Path]:
    # Imported lazily: model_surface pulls in the covariate stack, and this list is also read by
    # callers that never build the frame.
    from crimerisk.model_surface import bg_feature_dependency_paths

    return [
        *bg_feature_dependency_paths(paths, year=int(config.year)),
        block_group_crosswalk_path(paths),
        jurisdiction_year_estimates_path(paths),
        feature_catalog_path(paths, year=int(config.year)),
        config.weights_path or ship_weights_path(paths),
        # The exposure expert reads this surface, so a rebuilt normalizer is a stale expert table.
        resolve_exposure_normalizers_path(paths, config=config),
        Path(__file__).resolve(),
    ]


def mixture_experts_artifact_is_current(
    paths: RepoPaths, *, config: MixtureAllocationConfig, out_path: Path | None = None
) -> bool:
    artifact = out_path or mixture_experts_path(paths, year=int(config.year))
    summary_path = mixture_experts_summary_path(paths, year=int(config.year))
    if not summary_path.exists():
        return False
    # The envelope mode and its constants are configuration, not input files, so the dependency
    # stamp cannot see them: a table built under the hard clip would otherwise satisfy a
    # soft-shrinkage request and the lane would silently not be on.
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    recorded = summary.get("soft_shrinkage")
    if _soft_shrinkage_identity(recorded) != _soft_shrinkage_identity(soft_shrinkage_record(config)):
        return False
    # Same reasoning for the exposure expert: which normalizer surface the expert reads is
    # configuration, and a summary written before this lane existed declares no basis at all,
    # which is exactly a table built on the retired population expert.
    exposure = summary.get("exposure_expert") or {}
    if exposure.get("basis_by_offense") != dict(EXPOSURE_BASIS_BY_OFFENSE):
        return False
    if bool(exposure.get("qcew_updated")) != bool(config.enable_qcew_exposure_updating):
        return False
    return artifact_is_current(artifact, mixture_experts_dependency_paths(paths, config=config))


def _soft_shrinkage_identity(record: dict | None) -> tuple:
    """The part of the soft-shrinkage record that changes the numbers.

    A `None` record is a pre-lane summary, which is exactly a disabled lane -- so an untouched
    hard-clip artifact stays current under the default config and is never needlessly rebuilt.
    """
    if not record:
        return (False, None, None, None)
    if not bool(record.get("enabled")):
        return (False, None, None, None)
    return (
        True,
        float(record.get("nu", float("nan"))),
        float(record.get("extrapolation_weight", float("nan"))),
        tuple(float(value) for value in record.get("extrapolation_hull_percentiles", ())),
    )


# --- the frozen weight table -------------------------------------------------------------


def load_ship_weights(path: Path) -> pd.DataFrame:
    """Read the frozen weight table and refuse anything that is not a convex simplex point.

    This is a read, not a selection. The file is `08_ship_weights.py`'s output carried verbatim;
    validation here exists so a corrupted or hand-edited table fails the build instead of
    silently reshaping the country's allocation.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is the mixture weight source of record and is absent. It is a committed "
            f"copy of analysis_scratch/final_phase/corpus_expansion/ship_weights_v3.csv."
        )
    frame = pd.read_csv(path)
    missing = [column for column in ("offense", "stratum", *WEIGHT_COLUMNS) if column not in frame.columns]
    if missing:
        raise ValueError(f"mixture weight table {path} is missing columns {missing}")
    frame["offense"] = frame["offense"].astype(str).str.strip()
    frame["stratum"] = frame["stratum"].astype(str).str.strip()
    for column in WEIGHT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)

    if frame.duplicated(["offense", "stratum"]).any():
        raise ValueError(f"mixture weight table {path} carries duplicate (offense, stratum) rows")
    expected = {(offense, stratum) for offense in OFFENSES_7 for stratum in STRATA}
    actual = set(zip(frame["offense"], frame["stratum"], strict=True))
    if actual != expected:
        raise ValueError(
            f"mixture weight table {path} must cover exactly {len(expected)} (offense, stratum) "
            f"cells: {len(expected - actual)} missing, {len(actual - expected)} unexpected"
        )

    weights = frame[list(WEIGHT_COLUMNS)]
    if weights.isna().any().any():
        raise ValueError(f"mixture weight table {path} carries a non-numeric weight")
    if weights.lt(0.0).any().any() or weights.gt(1.0).any().any():
        raise ValueError(f"mixture weight table {path} carries a weight outside [0, 1]")
    if weights.sum(axis=1).sub(1.0).abs().gt(1e-9).any():
        raise ValueError(f"mixture weight table {path} carries a weight triple that does not sum to 1")
    return frame.sort_values(["offense", "stratum"], kind="mergesort").reset_index(drop=True)


def weight_lookup(weights: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float]]:
    return {
        (str(row.offense), str(row.stratum)): (float(row.w_prior), float(row.w_gbm), float(row.w_exposure))
        for row in weights.itertuples()
    }


# --- stratum, envelope, normalisation ----------------------------------------------------


def classify_stratum(population: float) -> str:
    """`core >= 200k > suburban >= 50k > small` (07_final_mixture.py, `stratum()`)."""
    value = float(population) if np.isfinite(population) else 0.0
    if value >= CORE_POPULATION_FLOOR:
        return CORE_STRATUM
    if value >= SUBURBAN_POPULATION_FLOOR:
        return SUBURBAN_STRATUM
    return SMALL_STRATUM


def classify_stratum_series(population: pd.Series, *, config: MixtureAllocationConfig | None = None) -> pd.Series:
    cfg = config or MixtureAllocationConfig()
    value = pd.to_numeric(population, errors="coerce").fillna(0.0).astype(float)
    return pd.Series(
        np.where(
            value.ge(float(cfg.core_population_floor)),
            CORE_STRATUM,
            np.where(value.ge(float(cfg.suburban_population_floor)), SUBURBAN_STRATUM, SMALL_STRATUM),
        ),
        index=value.index,
        dtype=object,
    ).astype("string")


def clip_to_envelope(rates: np.ndarray, low: float, high: float) -> np.ndarray:
    """The bounded-downward constraint: the ecological expert may shape, never assert an
    intensity outside the range the jurisdiction panel exhibits."""
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"non-finite intensity envelope [{low}, {high}]")
    if high < low:
        raise ValueError(f"inverted intensity envelope [{low}, {high}]")
    return np.clip(np.asarray(rates, dtype=float), float(low), float(high))


def compress_log_excess(excess_log: np.ndarray, *, nu: float) -> np.ndarray:
    """`nu * log1p(excess / nu)` -- the t-like compressor, on a non-negative log excess.

    Strictly increasing, identity to first order at 0, `nu*log(excess)` far out. `nu -> 0` is the
    hard clip and `nu -> inf` is no shrinkage, so the single knob spans exactly the two behaviours
    the plan item contrasts.
    """
    value = float(nu)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"soft-shrinkage nu must be finite and positive, got {nu!r}")
    excess = np.asarray(excess_log, dtype=float)
    if np.any(excess < 0.0):
        raise ValueError("compress_log_excess expects a non-negative log excess")
    return value * np.log1p(excess / value)


def soft_shrink_to_envelope(
    rates: np.ndarray,
    low: float,
    high: float,
    *,
    nu: float,
    extrapolation_weight: float = 0.0,
    extrapolating: np.ndarray | None = None,
) -> np.ndarray:
    """The soft replacement for `clip_to_envelope`: compress the excess, never truncate it.

    Inside `[low, high]` the input is returned unchanged, bit for bit. Outside it the log-scale
    excess over the nearer bound is compressed, and an extrapolating row's compressed excess is
    further scaled by `1 - extrapolation_weight` -- weight 1 lands the row exactly on the bound (the
    constant rate that reproduces the exposure-share null), weight 0 leaves the compressor alone.

    A zero prediction under a positive `low` maps to zero rather than up to `low`: the compressor is
    the limit of a strictly increasing map and an expert that predicts no intensity at all is not
    evidence for the panel minimum. The hard clip's opposite choice is the one being replaced.
    """
    lo, hi = float(low), float(high)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"non-finite intensity envelope [{low}, {high}]")
    if hi < lo:
        raise ValueError(f"inverted intensity envelope [{low}, {high}]")
    weight = float(extrapolation_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"extrapolation weight must lie in [0, 1], got {extrapolation_weight!r}")

    work = np.asarray(rates, dtype=float)
    out = work.copy()
    scale = np.ones(work.shape, dtype=float)
    if extrapolating is not None:
        flag = np.asarray(extrapolating, dtype=bool)
        if flag.shape != work.shape:
            raise ValueError("extrapolation flag must be shaped like the rate vector")
        scale = np.where(flag, 1.0 - weight, 1.0)

    if hi > 0.0:
        above = work > hi
        if np.any(above):
            excess = compress_log_excess(np.log(work[above] / hi), nu=nu)
            out[above] = hi * np.exp(scale[above] * excess)
    if lo > 0.0:
        below = (work < lo) & (work > 0.0)
        if np.any(below):
            excess = compress_log_excess(np.log(lo / work[below]), nu=nu)
            out[below] = lo * np.exp(-scale[below] * excess)
        out[(work <= 0.0)] = 0.0
    return out


def apply_envelope(
    rates: np.ndarray,
    low: float,
    high: float,
    *,
    config: MixtureAllocationConfig,
    extrapolating: np.ndarray | None = None,
) -> np.ndarray:
    """One call site for the envelope treatment, so the two modes can never diverge."""
    if not bool(config.enable_soft_shrinkage):
        return clip_to_envelope(rates, low, high)
    return soft_shrink_to_envelope(
        rates,
        low,
        high,
        nu=float(config.soft_shrinkage_nu),
        extrapolation_weight=float(config.soft_shrinkage_extrapolation_weight),
        extrapolating=extrapolating,
    )


def extrapolation_flags(
    features: np.ndarray, *, feature_low: np.ndarray, feature_high: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (flag, count of governed features outside their training range).

    The hull proxy is deliberately per-feature rather than a convex hull or a Mahalanobis shell: the
    governed feature set runs to ~180 columns, where a convex hull is both intractable and
    degenerate (almost every point is a vertex), and a distance in a fitted covariance would be a
    second fitted object needing its own validation. "Outside the range the training rows exhibited,
    on at least one governed covariate" is a statement about the training data alone.
    """
    matrix = np.asarray(features, dtype=float)
    low = np.asarray(feature_low, dtype=float)
    high = np.asarray(feature_high, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != low.size or low.size != high.size:
        raise ValueError("extrapolation hull bounds do not match the feature matrix")
    outside = np.zeros(matrix.shape, dtype=bool)
    finite = np.isfinite(matrix)
    np.logical_and(finite, (matrix < low) | (matrix > high), out=outside)
    count = outside.sum(axis=1).astype(np.int32)
    return count > 0, count


def normalize_within_groups(values: np.ndarray, group_codes: np.ndarray, n_groups: int) -> np.ndarray:
    """`v / Σv` per group; a group whose mass is zero (or non-finite) keeps zeros.

    Mirrors `_e2_prep.norm`, which returns its input untouched when the total is not positive --
    for a non-negative input that is the zero vector either way.
    """
    work = np.asarray(values, dtype=float)
    work = np.where(np.isfinite(work), work, 0.0)
    work = np.clip(work, 0.0, None)
    totals = np.bincount(group_codes, weights=work, minlength=n_groups)
    denominator = np.where(totals > 0.0, totals, 1.0)[group_codes]
    return np.where(np.take(totals, group_codes) > 0.0, work / denominator, 0.0)


def _group_codes(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Dense integer codes for a composite key, in first-appearance order.

    A joined string key rather than a MultiIndex: `factorize` never has to compare tuples, the
    codes are guaranteed non-negative (so `bincount` is safe), and the result does not depend on
    the dtype of any individual key column. `\\x1f` is the ASCII unit separator, which cannot
    occur in a jurisdiction id, state FIPS, or offense label.
    """
    key = frame[columns[0]].astype(str)
    for column in columns[1:]:
        key = key + "\x1f" + frame[column].astype(str)
    return pd.factorize(key, sort=False)


def mix_expert_shares(
    *,
    prior_share: np.ndarray,
    gbm_share: np.ndarray,
    exposure_share: np.ndarray,
    w_prior: np.ndarray,
    w_gbm: np.ndarray,
    w_exposure: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """`norm(w_b·s_b + w_g·s_g + w_e·s_e)`, exactly 07's `norm(w[0]*s_b + w[1]*s_g + w[2]*s_e)`.

    The blend of three vectors that each already sum to 1 sums to 1 whenever the weights do, so
    the outer normalisation is a no-op except where an expert is degenerate over the footprint
    (all-zero prior, zero exposure) -- which is exactly where it is needed.
    """
    blended = (
        np.asarray(w_prior, dtype=float) * np.asarray(prior_share, dtype=float)
        + np.asarray(w_gbm, dtype=float) * np.asarray(gbm_share, dtype=float)
        + np.asarray(w_exposure, dtype=float) * np.asarray(exposure_share, dtype=float)
    )
    return normalize_within_groups(blended, group_codes, n_groups)


# --- the GBM expert ----------------------------------------------------------------------


# The tournament screened on the parquet file's *arrow* type names
# (`_e2_prep._feat_cols_from_schema`: double / float / int64 / int32). In memory those arrive as
# either numpy or pandas-nullable dtypes depending on the column's null pattern, and both must be
# admitted or the model trains on a different feature set. Booleans, unsigned, and narrow integer
# types are excluded here exactly as the arrow screen excluded them.
GOVERNED_FEATURE_DTYPES = frozenset(
    {"float64", "float32", "int64", "int32", "Float64", "Float32", "Int64", "Int32"}
)


def governed_feature_columns(*, catalog: pd.DataFrame, feature_frame_columns: list[str], dtypes: pd.Series) -> list[str]:
    """The governed, numeric covariates in feature-frame column order.

    Order is taken from the frame's own column order rather than from a sorted set, because the
    tournament read it from the parquet schema and a tree model is not order-invariant. Governed
    = the no-redlining feature policy already applied to the incident prior.
    """
    allowed = set(
        catalog.loc[~catalog["excluded_by_feature_policy"].fillna(False), "feature_column"].astype(str)
    )
    return [
        column
        for column in feature_frame_columns
        if column in allowed and str(dtypes[column]) in GOVERNED_FEATURE_DTYPES
    ]


def build_jurisdiction_feature_panel(
    *, bg: pd.DataFrame, feature_columns: list[str], jurisdiction_column: str = "eb_jurisdiction_id"
) -> pd.DataFrame:
    """Population-weighted, per-column NaN-skipping mean of the covariates, plus `log_pop`.

    Transcribed from `07_final_mixture.py`'s prep (`_e2_prep.load_prep`, `wmean`) rather than
    re-expressed as a vectorised groupby: a tree model is sensitive to the last bits of a feature
    value, and a different summation order is a different model.
    """

    def wmean(frame: pd.DataFrame) -> pd.Series:
        x = frame[feature_columns].to_numpy(dtype=float)
        ww = frame["pop"].to_numpy(dtype=float)[:, None]
        ok = ~np.isnan(x)
        num = np.nansum(np.where(ok, x, 0.0) * ww, axis=0)
        den = (ok * ww).sum(axis=0)
        with np.errstate(invalid="ignore"):
            return pd.Series(np.where(den > 0, num / den, np.nan), index=feature_columns)

    panel = bg.groupby(jurisdiction_column).apply(wmean, include_groups=False)
    panel["log_pop"] = np.log1p(bg.groupby(jurisdiction_column)["pop"].sum())
    return panel


def build_jurisdiction_targets(
    panel: pd.DataFrame, *, config: MixtureAllocationConfig
) -> pd.DataFrame:
    """Mean clean-row reported rate per (jurisdiction, offense) -- the GBM's only target.

    FBI-panel quantities exclusively. No incident-feed truth, so the fitted share carries no
    leakage from the benchmark the mixture weights were selected on.
    """
    required = [
        "jurisdiction_id",
        "offense",
        "year",
        "bucket_population",
        "reported_count_preferred",
        "mean_months_reported_preferred",
        "fill_component_count",
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"jurisdiction-year panel is missing columns {missing}")
    years = list(range(int(config.panel_year_start), int(config.year) + 1))
    clean = panel[
        panel["reported_count_preferred"].notna()
        & panel["mean_months_reported_preferred"].ge(12.0)
        & panel["fill_component_count"].fillna(0).eq(0.0)
        & panel["year"].isin(years)
    ]
    targets = (
        clean.groupby(["jurisdiction_id", "offense"])
        .agg(
            mean_count=("reported_count_preferred", "mean"),
            popj=("bucket_population", "first"),
            nyears=("year", "nunique"),
        )
        .reset_index()
    )
    targets = targets[
        targets["nyears"].ge(int(config.min_target_years))
        & targets["popj"].gt(float(config.min_target_population))
    ]
    targets = targets.copy()
    targets["rate"] = targets["mean_count"] / targets["popj"] * 1e5
    return targets


@dataclass(frozen=True)
class OffenseGBM:
    """A fitted offense model plus everything prediction needs to be reproducible."""

    offense: str
    model: object
    scaler: object
    envelope_lo: float
    envelope_hi: float
    column_median: np.ndarray
    n_train_rows: int
    n_train_jurisdictions: int
    # The per-feature training hull proxy. Computed unconditionally (it is two percentiles over a
    # matrix that is already in hand) and consumed only when the soft-shrinkage lane is on, so the
    # hard-clip path fits exactly the same model it fitted before.
    feature_lo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_hi: np.ndarray = field(default_factory=lambda: np.zeros(0))


def train_offense_gbm(
    *,
    offense: str,
    targets: pd.DataFrame,
    jurisdiction_features: pd.DataFrame,
    x_columns: list[str],
    config: MixtureAllocationConfig,
) -> OffenseGBM:
    """`HistGradientBoostingRegressor(150, depth 4, seed 0)` on pop-weighted `log1p(rate)`.

    Transcribed from 07_final_mixture.py's fit loop, including the NaN-complete-row screen, the
    scaler, the sample weighting by jurisdiction population, and the percentile envelope.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    frame = targets[targets["offense"] == offense]
    train = frame.merge(jurisdiction_features, left_on="jurisdiction_id", right_index=True, how="inner")
    X = train[x_columns].to_numpy(dtype=float)
    keep = ~np.isnan(X).any(axis=1)
    X, rates = X[keep], train["rate"].to_numpy()[keep]
    popw = train["popj"].to_numpy()[keep]
    if X.shape[0] == 0:
        raise ValueError(f"no complete training rows for offense {offense}")
    scaler = StandardScaler()
    gbm = HistGradientBoostingRegressor(
        max_iter=int(config.gbm_max_iter),
        max_depth=int(config.gbm_max_depth),
        random_state=int(config.gbm_random_state),
    ).fit(scaler.fit_transform(X), np.log1p(rates), sample_weight=popw)
    low, high = np.percentile(rates, list(config.envelope_percentiles))
    feature_lo, feature_hi = np.percentile(X, list(config.extrapolation_hull_percentiles), axis=0)
    return OffenseGBM(
        offense=str(offense),
        model=gbm,
        scaler=scaler,
        envelope_lo=float(low),
        envelope_hi=float(high),
        column_median=np.nanmedian(X, axis=0),
        n_train_rows=int(X.shape[0]),
        n_train_jurisdictions=int(train["jurisdiction_id"].nunique()),
        feature_lo=np.asarray(feature_lo, dtype=float),
        feature_hi=np.asarray(feature_hi, dtype=float),
    )


def predict_offense_rates(
    *, fitted: OffenseGBM, bg: pd.DataFrame, x_columns: list[str]
) -> np.ndarray:
    """Per-block-group predicted rate, NaNs filled with the training-row column median.

    Rows are independent under this imputation, so predicting the whole universe at once is
    identical to 07's per-jurisdiction prediction -- verified by the port-fidelity check.
    """
    Xb = bg[x_columns].to_numpy(dtype=float)
    index = np.where(np.isnan(Xb))
    Xb[index] = np.take(fitted.column_median, index[1])
    return np.expm1(fitted.model.predict(fitted.scaler.transform(Xb))).clip(min=0.0)


# --- the exposure expert -----------------------------------------------------------------


def load_opportunity_normalizers(
    *, paths: RepoPaths, config: MixtureAllocationConfig
) -> pd.DataFrame:
    """`bg_id` -> the per-offense opportunity mass, indexed and zero-padded.

    Read rather than rebuilt, for the same reason the weight table is: the normalizers were
    selected once by E3 against incident feeds, and a build step that re-derived them would put
    selection inside the shipped artifact. Absent artifact is a hard error -- the expert is DEFINED
    over this surface now, and falling back to population everywhere would silently hand the
    weights an expert they were not selected for.
    """
    path = resolve_exposure_normalizers_path(paths, config=config)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} carries the opportunity normalizers the mixture's exposure expert is "
            f"normalised over and is absent. Run `build-exposure-normalizers"
            f"{' --enable-qcew-exposure-updating' if config.enable_qcew_exposure_updating else ''}`"
            f" first."
        )
    columns = ["bg_id", *(opportunity_normalizer_column(offense) for offense in ENSEMBLE_OFFENSES)]
    frame = pd.read_parquet(path, columns=columns)
    frame["bg_id"] = frame["bg_id"].astype(str).str.zfill(12)
    if frame["bg_id"].duplicated().any():
        raise ValueError(f"opportunity normalizer surface {path} carries duplicate block groups")
    return frame.set_index("bg_id")


def exposure_expert_weights(
    *,
    offense: str,
    population: np.ndarray,
    bg_ids: pd.Series,
    normalizers: pd.DataFrame,
    config: MixtureAllocationConfig,
) -> tuple[np.ndarray, str, np.ndarray]:
    """`(exposure_weight, basis, row_fallback_flag)` for one offense over the block-group basis.

    The floor is applied last, to whichever quantity was selected, so the expert's shape is the
    normalizer's shape and the floor stays what it was: a guard against a degenerate footprint,
    not a contribution to the ordering.
    """
    floor = float(config.exposure_floor)
    pop = np.asarray(population, dtype=float)
    basis = exposure_basis_for_offense(offense)
    if basis == EXPOSURE_BASIS_POPULATION:
        return np.maximum(pop, floor), basis, np.zeros(pop.shape, dtype=bool)

    column = opportunity_normalizer_column(offense)
    if column not in normalizers.columns:
        raise ValueError(
            f"opportunity normalizer surface is missing {column}, which the exposure expert for "
            f"{offense} is normalised over"
        )
    values = pd.to_numeric(
        normalizers[column].reindex(pd.Index(bg_ids, name="bg_id")), errors="coerce"
    ).to_numpy(dtype=float)
    fallback = ~(np.isfinite(values) & (values > 0.0))
    selected = np.where(fallback, pop, values)
    return np.maximum(selected, floor), basis, fallback


# --- expert-table build ------------------------------------------------------------------


def build_block_group_basis(
    *, paths: RepoPaths, config: MixtureAllocationConfig
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """The published block-group universe with covariates, population, and its jurisdiction.

    Reconstructed from production inputs only -- the feature frame plus the crosswalk's dominant
    block-group assignment, minus the out-of-scope states. That reconstruction is exactly the
    published universe (verified: 238,193 block groups, identical membership), which is how this
    lane reproduces the tournament's fit without reading `state/output/`.
    """
    from crimerisk.model_surface import build_bg_feature_frame

    catalog = pd.read_parquet(feature_catalog_path(paths, year=int(config.year)))
    features = build_bg_feature_frame(paths=paths, year=int(config.year))
    feature_columns = governed_feature_columns(
        catalog=catalog,
        feature_frame_columns=list(features.columns),
        dtypes=features.dtypes,
    )
    if not feature_columns:
        raise ValueError("no governed numeric covariates resolved for the mixture GBM")

    population_column = f"population_{int(config.year)}"
    if population_column not in features.columns:
        raise ValueError(f"block-group feature frame is missing {population_column}")

    bg = features[["bg_id", *feature_columns]].copy()
    bg["bg_id"] = bg["bg_id"].astype(str).str.zfill(12)
    universe = features[["bg_id", population_column]].copy()
    universe["bg_id"] = universe["bg_id"].astype(str).str.zfill(12)

    # Lazy: allocation.py imports this module, so the reverse import happens at call time.
    from crimerisk.allocation import _dominant_bg_jurisdiction, _load_bg_crosswalk

    dominant = _dominant_bg_jurisdiction(_load_bg_crosswalk(paths))
    dominant["block_group_geoid"] = dominant["block_group_geoid"].astype(str).str.zfill(12)
    universe = universe.merge(
        dominant, left_on="bg_id", right_on="block_group_geoid", how="inner", validate="one_to_one"
    )
    universe["state_fips"] = universe["bg_id"].str.slice(0, 2)
    universe = universe[~universe["state_fips"].isin(set(config.excluded_state_fips))]

    bg = bg.merge(
        universe[["block_group_geoid", "eb_jurisdiction_id", population_column, "state_fips"]],
        left_on="bg_id",
        right_on="block_group_geoid",
        how="inner",
        suffixes=("", "_universe"),
    )
    population_source = (
        f"{population_column}_universe" if f"{population_column}_universe" in bg.columns else population_column
    )
    bg["pop"] = pd.to_numeric(bg[population_source], errors="coerce").fillna(0.0).clip(lower=0.0)
    bg["log_pop"] = np.log1p(bg["pop"])
    return bg, feature_columns, [*feature_columns, "log_pop"]


def _load_or_build_feature_panel(
    *,
    paths: RepoPaths,
    config: MixtureAllocationConfig,
    bg: pd.DataFrame,
    feature_columns: list[str],
    force: bool,
) -> pd.DataFrame:
    cache_path = jurisdiction_feature_panel_path(paths, year=int(config.year))
    dependencies = mixture_experts_dependency_paths(paths, config=config)
    if not force and artifact_is_current(cache_path, dependencies):
        panel = pd.read_parquet(cache_path).set_index("eb_jurisdiction_id")
        if list(panel.columns) == [*feature_columns, "log_pop"]:
            return panel
    panel = build_jurisdiction_feature_panel(bg=bg, feature_columns=feature_columns)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.rename_axis("eb_jurisdiction_id").reset_index().to_parquet(cache_path, index=False)
    write_dependency_stamp(cache_path, dependencies)
    return panel


def build_mixture_experts(
    *, paths: RepoPaths, config: MixtureAllocationConfig = MixtureAllocationConfig(), force: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Fit the seven offense GBMs and emit the per-block-group expert table."""
    bg, feature_columns, x_columns = build_block_group_basis(paths=paths, config=config)
    panel = _load_or_build_feature_panel(
        paths=paths, config=config, bg=bg, feature_columns=feature_columns, force=force
    )
    targets = build_jurisdiction_targets(
        pd.read_parquet(jurisdiction_year_estimates_path(paths)), config=config
    )
    normalizers = load_opportunity_normalizers(paths=paths, config=config)

    soft = bool(config.enable_soft_shrinkage)
    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, dict[str, float | int]] = {}
    for offense in OFFENSES_7:
        fitted = train_offense_gbm(
            offense=offense,
            targets=targets,
            jurisdiction_features=panel,
            x_columns=x_columns,
            config=config,
        )
        rate = predict_offense_rates(fitted=fitted, bg=bg, x_columns=x_columns)
        flag, flag_count = extrapolation_flags(
            bg[x_columns].to_numpy(dtype=float),
            feature_low=fitted.feature_lo,
            feature_high=fitted.feature_hi,
        )
        bounded = apply_envelope(
            rate,
            fitted.envelope_lo,
            fitted.envelope_hi,
            config=config,
            extrapolating=flag if soft else None,
        )
        population = bg["pop"].to_numpy(dtype=float)
        # The GBM predicts a per-RESIDENT rate on a population-weighted jurisdiction panel, so the
        # count it implies is that rate times the residents. Expert `g` therefore keeps the
        # population exposure it was fitted against; only expert `e` is re-based.
        population_exposure = np.maximum(population, float(config.exposure_floor))
        exposure, basis, row_fallback = exposure_expert_weights(
            offense=offense,
            population=population,
            bg_ids=bg["bg_id"],
            normalizers=normalizers,
            config=config,
        )
        frame = pd.DataFrame(
            {
                "state_fips": bg["state_fips"].astype("string").to_numpy(),
                "bg_id": bg["bg_id"].astype("string").to_numpy(),
                "offense": offense,
                "population": population,
                "exposure_weight": exposure,
                EXPOSURE_BASIS_COLUMN: basis,
                EXPOSURE_BASIS_FALLBACK_COLUMN: row_fallback,
                "gbm_rate": rate,
                "gbm_rate_clipped": bounded,
                "gbm_weight": bounded * population_exposure,
                "envelope_lo": fitted.envelope_lo,
                "envelope_hi": fitted.envelope_hi,
            }
        )
        if soft:
            frame["envelope_mode"] = ENVELOPE_MODE_SOFT_SHRINKAGE
            frame["extrapolation_flag"] = flag
            frame["extrapolation_feature_count"] = flag_count
        frames.append(frame)
        above = rate > fitted.envelope_hi
        below = rate < fitted.envelope_lo
        diagnostics[offense] = {
            "train_rows": fitted.n_train_rows,
            "train_jurisdictions": fitted.n_train_jurisdictions,
            "envelope_lo": fitted.envelope_lo,
            "envelope_hi": fitted.envelope_hi,
            "clipped_low_block_groups": int(np.sum(below)),
            "clipped_high_block_groups": int(np.sum(above)),
            "median_predicted_rate": float(np.median(rate)),
            "exposure_basis": basis,
            "exposure_basis_row_fallbacks": int(np.sum(row_fallback)),
            # The number the diagnosis was about: how much of the expert's own mass lands where
            # the residential headcount is zero. Under the population basis this is ~0 by
            # construction; under the normalizer it is the reason the lane changed.
            "exposure_mass_share_in_zero_population_block_groups": (
                float(exposure[population <= 0.0].sum() / exposure.sum())
                if exposure.sum() > 0.0
                else 0.0
            ),
        }
        if soft:
            outside = above | below
            diagnostics[offense].update(
                {
                    "extrapolating_block_groups": int(np.sum(flag)),
                    "extrapolating_outside_envelope_block_groups": int(np.sum(flag & outside)),
                    # What the excess would have been under the clip, and what it is instead. The
                    # honest size of the lane, per offense, in one pair of numbers.
                    "hard_clipped_mass_removed": float(
                        np.sum((rate - clip_to_envelope(rate, fitted.envelope_lo, fitted.envelope_hi))[above])
                    ),
                    "soft_shrunk_mass_removed": float(np.sum((rate - bounded)[above])),
                    "max_retained_rate_ratio_above_envelope": (
                        float(np.max(bounded[above] / fitted.envelope_hi))
                        if bool(above.any()) and fitted.envelope_hi > 0.0
                        else 1.0
                    ),
                }
            )

    experts = (
        pd.concat(frames, ignore_index=True)
        .reindex(columns=MIXTURE_EXPERT_COLUMNS + (SOFT_SHRINKAGE_EXPERT_COLUMNS if soft else []))
        .sort_values(["offense", "bg_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    experts["offense"] = experts["offense"].astype("string")
    assert_mixture_expert_invariants(experts)

    summary = {
        "year": int(config.year),
        "provenance": "E2 final mixture (analysis_scratch/final_phase/e2_allocation/07_final_mixture.py)",
        "block_groups": int(bg["bg_id"].nunique()),
        "jurisdictions_in_feature_panel": int(len(panel)),
        "governed_feature_columns": int(len(feature_columns)),
        "model_columns": int(len(x_columns)),
        "gbm": {
            "estimator": "HistGradientBoostingRegressor",
            "max_iter": int(config.gbm_max_iter),
            "max_depth": int(config.gbm_max_depth),
            "random_state": int(config.gbm_random_state),
            "target": "log1p(mean clean-row reported rate per 100k)",
            "sample_weight": "bucket_population",
            "scaler": "StandardScaler",
        },
        "target_screen": {
            "years": [int(config.panel_year_start), int(config.year)],
            "min_years": int(config.min_target_years),
            "min_population": float(config.min_target_population),
            "rows": int(len(targets)),
        },
        "envelope_percentiles": list(config.envelope_percentiles),
        "offenses": diagnostics,
        "weights_path": str(config.weights_path or ship_weights_path(paths)),
        # Self-identifying, like the soft-shrinkage record below: an expert table says which
        # surface its exposure arm measures, so the build that consumes it can refuse a table
        # built for different weights than the ones it is about to apply.
        "exposure_expert": {
            "basis_by_offense": dict(EXPOSURE_BASIS_BY_OFFENSE),
            "normalizers_path": str(resolve_exposure_normalizers_path(paths, config=config)),
            "qcew_updated": bool(config.enable_qcew_exposure_updating),
            "population_fallback_offenses": [
                offense
                for offense, basis in EXPOSURE_BASIS_BY_OFFENSE.items()
                if basis == EXPOSURE_BASIS_POPULATION
            ],
            "floor": float(config.exposure_floor),
            "diagnosis": "analysis_scratch/final_phase/certification/special_use_rank_diagnosis.md",
        },
        "ags_values_used": False,
        # Self-identifying, like every other v2 lane: an expert table produced under the soft
        # envelope says so, and says under which constants, so the build that consumes it can
        # refuse a table that was not built under the lane it was asked to run.
        "soft_shrinkage": soft_shrinkage_record(config),
    }
    return experts, summary


def soft_shrinkage_record(config: MixtureAllocationConfig) -> dict:
    return {
        "enabled": bool(config.enable_soft_shrinkage),
        "envelope_mode": (
            ENVELOPE_MODE_SOFT_SHRINKAGE if config.enable_soft_shrinkage else ENVELOPE_MODE_HARD_CLIP
        ),
        "compressor": "r' = bound * exp(scale * nu * log1p(|log(r/bound)| / nu))",
        "nu": float(config.soft_shrinkage_nu),
        "extrapolation_weight": float(config.soft_shrinkage_extrapolation_weight),
        "extrapolation_hull_percentiles": list(config.extrapolation_hull_percentiles),
        "contract": "analysis_scratch/final_phase/SOFT_SHRINKAGE_CONTRACT.md",
    }


def assert_mixture_expert_invariants(experts: pd.DataFrame, *, tolerance: float = 1e-9) -> None:
    missing = sorted(set(MIXTURE_EXPERT_COLUMNS) - set(experts.columns))
    if missing:
        raise ValueError(f"mixture expert table is missing columns {missing}")
    if experts.duplicated(["bg_id", "offense"]).any():
        raise ValueError("mixture expert table carries duplicate (bg_id, offense) keys")
    offenses = set(experts["offense"].astype(str))
    if offenses != set(OFFENSES_7):
        raise ValueError(f"mixture expert table covers offenses {sorted(offenses)}, expected all seven")
    for column in ("population", "exposure_weight", "gbm_rate", "gbm_rate_clipped", "gbm_weight"):
        value = pd.to_numeric(experts[column], errors="coerce")
        if value.isna().any() or not np.isfinite(value.to_numpy(dtype=float)).all():
            raise ValueError(f"mixture expert table carries a nonfinite {column}")
        if value.lt(-tolerance).any():
            raise ValueError(f"mixture expert table carries a negative {column}")
    if experts["exposure_weight"].lt(EXPOSURE_FLOOR - tolerance).any():
        raise ValueError("mixture expert table carries an exposure weight below the floor")
    assert_exposure_basis(experts)
    bounded = pd.to_numeric(experts["gbm_rate_clipped"], errors="coerce")
    raw = pd.to_numeric(experts["gbm_rate"], errors="coerce")
    low = pd.to_numeric(experts["envelope_lo"], errors="coerce")
    high = pd.to_numeric(experts["envelope_hi"], errors="coerce")

    if not expert_table_is_soft_shrunk(experts):
        if bounded.lt(low - tolerance).any() or bounded.gt(high + tolerance).any():
            raise ValueError("mixture expert table carries a GBM rate outside its learned envelope")
        return

    # Under the soft envelope containment is exactly the property being given up, so the mirror is
    # the property that replaced it: the treatment moves every row TOWARD its bound and never past
    # it, never past the raw prediction, and never at all inside the envelope.
    inside = raw.between(low - tolerance, high + tolerance)
    if bounded[inside].sub(raw[inside]).abs().gt(tolerance).any():
        raise ValueError("soft-shrunk expert table moved a GBM rate that was inside its envelope")
    above = raw.gt(high + tolerance)
    if above.any() and (bounded[above].lt(high - tolerance).any() or bounded[above].gt(raw[above] + tolerance).any()):
        raise ValueError("soft-shrunk expert table moved an above-envelope rate outside [bound, raw]")
    below = raw.lt(low - tolerance)
    if below.any() and (bounded[below].gt(low + tolerance).any() or bounded[below].lt(raw[below] - tolerance).any()):
        raise ValueError("soft-shrunk expert table moved a below-envelope rate outside [raw, bound]")
    flag = experts["extrapolation_flag"]
    if flag.isna().any():
        raise ValueError("soft-shrunk expert table carries a null extrapolation flag")
    count = pd.to_numeric(experts["extrapolation_feature_count"], errors="coerce")
    if count.isna().any() or count.lt(0).any():
        raise ValueError("soft-shrunk expert table carries an invalid extrapolation feature count")
    if not flag.astype(bool).eq(count.gt(0)).all():
        raise ValueError("soft-shrunk expert table's extrapolation flag disagrees with its own count")


def assert_exposure_basis(experts: pd.DataFrame) -> None:
    """Every offense's rows declare the basis this code builds that offense's expert on.

    The weights are selected FOR an expert. A table whose exposure arm is residential population
    scored under weights chosen for the opportunity normalizer is a silent misallocation -- the
    share vector still sums to 1 either way -- so the mismatch is an assertion, exactly like the
    envelope-mode check.
    """
    declared = (
        experts.groupby(experts["offense"].astype(str))[EXPOSURE_BASIS_COLUMN]
        .agg(lambda values: sorted(set(values.astype(str))))
        .to_dict()
    )
    for offense, bases in sorted(declared.items()):
        expected = exposure_basis_for_offense(offense)
        if bases != [expected]:
            raise ValueError(
                f"mixture expert table declares exposure basis {bases} for {offense}, but this "
                f"build's exposure expert is normalised over {expected!r}; rebuild it with "
                f"`build-mixture-experts --force`."
            )
    flag = experts[EXPOSURE_BASIS_FALLBACK_COLUMN]
    if flag.isna().any():
        raise ValueError("mixture expert table carries a null exposure-basis fallback flag")
    population_basis = experts[EXPOSURE_BASIS_COLUMN].astype(str).eq(EXPOSURE_BASIS_POPULATION)
    if flag.astype(bool).to_numpy()[population_basis.to_numpy()].any():
        raise ValueError(
            "mixture expert table marks a row-level exposure fallback on an offense whose basis "
            "is already population, which cannot happen"
        )


def expert_table_is_soft_shrunk(experts: pd.DataFrame) -> bool:
    """The table's own declaration, cross-checked against its columns.

    A half-written table -- the mode column without the flag columns, or flag columns without the
    mode -- fails here rather than quietly selecting whichever invariant set happens to pass.
    """
    present = [column for column in SOFT_SHRINKAGE_EXPERT_COLUMNS if column in experts.columns]
    if not present:
        return False
    if len(present) != len(SOFT_SHRINKAGE_EXPERT_COLUMNS):
        raise ValueError(
            f"mixture expert table carries a partial soft-shrinkage column set: {sorted(present)}"
        )
    modes = set(experts["envelope_mode"].astype(str))
    if modes != {ENVELOPE_MODE_SOFT_SHRINKAGE}:
        raise ValueError(f"mixture expert table declares unexpected envelope modes {sorted(modes)}")
    return True


def write_v2_mixture_experts(
    *,
    paths: RepoPaths,
    out_path: Path | None = None,
    config: MixtureAllocationConfig = MixtureAllocationConfig(),
    force: bool = False,
) -> tuple[Path, dict]:
    artifact = out_path or mixture_experts_path(paths, year=int(config.year))
    summary_path = mixture_experts_summary_path(paths, year=int(config.year))
    if not force and mixture_experts_artifact_is_current(paths, config=config, out_path=artifact):
        return artifact, json.loads(summary_path.read_text())
    experts, summary = build_mixture_experts(paths=paths, config=config, force=force)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    experts.to_parquet(artifact, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_dependency_stamp(artifact, mixture_experts_dependency_paths(paths, config=config))
    return artifact, summary


def load_mixture_experts(paths: RepoPaths, *, year: int, path: Path | None = None) -> pd.DataFrame:
    artifact = path or mixture_experts_path(paths, year=int(year))
    if not artifact.exists():
        raise FileNotFoundError(
            f"{artifact} is required by the v2 mixture allocation path and is absent. Run "
            f"`build-mixture-experts` first."
        )
    experts = pd.read_parquet(artifact)
    experts["bg_id"] = experts["bg_id"].astype("string").str.zfill(12)
    experts["offense"] = experts["offense"].astype("string")
    assert_mixture_expert_invariants(experts)
    return experts


# --- application to the model lane -------------------------------------------------------


@dataclass(frozen=True)
class MixtureRuntime:
    """Everything the allocation build needs to apply the mixture, resolved once per build."""

    experts: pd.DataFrame
    weights: pd.DataFrame
    config: MixtureAllocationConfig
    audit_path: Path | None = None

    @property
    def lookup(self) -> dict[tuple[str, str], tuple[float, float, float]]:
        return weight_lookup(self.weights)


def resolve_mixture_runtime(
    *,
    paths: RepoPaths,
    config: MixtureAllocationConfig = MixtureAllocationConfig(),
    experts_path: Path | None = None,
    weights_path: Path | None = None,
    audit_path: Path | None = None,
) -> MixtureRuntime:
    experts = load_mixture_experts(paths, year=int(config.year), path=experts_path)
    # Fail closed on a mode mismatch in EITHER direction. A build asked for the soft envelope but
    # handed a hard-clipped table would publish the caps it was told to replace; a build asked for
    # the caps but handed a soft table would publish a lane nobody enabled. Both are silent under a
    # share vector that sums to 1, which is why this is an assertion and not a warning.
    table_is_soft = expert_table_is_soft_shrunk(experts)
    if table_is_soft != bool(config.enable_soft_shrinkage):
        wanted = "soft shrinkage" if config.enable_soft_shrinkage else "the hard clip"
        found = "soft shrinkage" if table_is_soft else "the hard clip"
        raise ValueError(
            f"the mixture expert table was built under {found} but this build asked for {wanted}; "
            f"rebuild it with `build-mixture-experts"
            f"{' --enable-soft-shrinkage' if config.enable_soft_shrinkage else ''} --force`."
        )
    return MixtureRuntime(
        experts=experts,
        weights=load_ship_weights(weights_path or config.weights_path or ship_weights_path(paths)),
        config=config,
        audit_path=audit_path,
    )


def apply_mixture_to_model_lane(
    merged: pd.DataFrame, *, runtime: MixtureRuntime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace the model-lane share vector with the three-expert mixture.

    Returns the frame with `model_share` (and the `model_total`-preserving
    `model_component_weight`) replaced, plus the audit record of what was applied. Nothing else
    on the frame is touched: the city-incident lane, the crosswalk shares, the prior, and the
    group's prior information mass all survive unchanged.
    """
    required = ["bg_id", "offense", "allocation_share", "bg_weight", "model_share", "model_total", *MIXTURE_GROUP_COLUMNS]
    missing = [column for column in required if column not in merged.columns]
    if missing:
        raise ValueError(f"model-lane frame is missing columns {missing} required by the mixture")

    out = merged.copy()
    key_bg = out["bg_id"].astype("string").str.zfill(12)
    key_offense = out["offense"].astype("string")

    experts = runtime.experts[["bg_id", "offense", "population", "exposure_weight", "gbm_weight"]]
    joined = pd.DataFrame({"bg_id": key_bg, "offense": key_offense}, index=out.index).merge(
        experts, on=["bg_id", "offense"], how="left", validate="many_to_one"
    )
    joined.index = out.index
    unmatched = int(joined["exposure_weight"].isna().sum())
    if unmatched:
        # Fail closed. A missing expert row would silently zero two of the three experts for
        # that block group and shrink it out of its own footprint, which is a defect that a
        # share vector summing to 1 would otherwise hide.
        sample = sorted(set(key_bg[joined["exposure_weight"].isna()].head(5).astype(str)))
        raise ValueError(
            f"mixture expert table does not cover {unmatched} model-lane support rows "
            f"(e.g. block groups {sample}); rebuild it with `build-mixture-experts`."
        )

    allocation = pd.to_numeric(out["allocation_share"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    population_bg = pd.to_numeric(joined["population"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    exposure_bg = pd.to_numeric(joined["exposure_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    gbm_bg = pd.to_numeric(joined["gbm_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    prior_component = (
        pd.to_numeric(out["bg_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float) * allocation
    )

    group_codes, group_index = _group_codes(out, MIXTURE_GROUP_COLUMNS)
    n_groups = len(group_index)

    prior_share = normalize_within_groups(prior_component, group_codes, n_groups)
    gbm_share = normalize_within_groups(gbm_bg * allocation, group_codes, n_groups)
    exposure_share = normalize_within_groups(exposure_bg * allocation, group_codes, n_groups)

    # The stratum reads the real headcount: the exposure floor is an expert-construction device
    # (07's `pop.clip(lower=1)`), not a population, and must not inflate a footprint.
    footprint_population = np.bincount(group_codes, weights=population_bg * allocation, minlength=n_groups)
    strata = classify_stratum_series(pd.Series(footprint_population), config=runtime.config).to_numpy()
    row_stratum = np.take(strata, group_codes)

    lookup = runtime.lookup
    offense_values = key_offense.to_numpy()
    weights = np.array(
        [lookup[(str(offense), str(stratum))] for offense, stratum in zip(offense_values, row_stratum, strict=True)],
        dtype=float,
    )
    w_prior, w_gbm, w_exposure = weights[:, 0], weights[:, 1], weights[:, 2]

    share = mix_expert_shares(
        prior_share=prior_share,
        gbm_share=gbm_share,
        exposure_share=exposure_share,
        w_prior=w_prior,
        w_gbm=w_gbm,
        w_exposure=w_exposure,
        group_codes=group_codes,
        n_groups=n_groups,
    )

    legacy_share = pd.to_numeric(out["model_share"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model_total = pd.to_numeric(out["model_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["model_share"] = share
    # `model_total` is the prior information mass I that sets the rare-offense shrinkage weight
    # I/(I+K). Re-expressing the component at the mixture's shape leaves Σ per group -- and so I,
    # and so every K already calibrated against it -- exactly invariant.
    out["model_component_weight"] = share * model_total

    audit = pd.DataFrame(
        {
            "state_fips": out["state_fips"].astype("string").to_numpy(),
            "jurisdiction_id": out["jurisdiction_id"].astype("string").to_numpy(),
            "jurisdiction_type": (
                out["jurisdiction_type"].astype("string").to_numpy()
                if "jurisdiction_type" in out.columns
                else pd.array([pd.NA] * len(out), dtype="string")
            ),
            "bg_id": key_bg.to_numpy(),
            "offense": offense_values,
            "stratum": row_stratum,
            "footprint_population": np.take(footprint_population, group_codes),
            "w_prior": w_prior,
            "w_gbm": w_gbm,
            "w_exposure": w_exposure,
            "prior_share": prior_share,
            "gbm_share": gbm_share,
            "exposure_share": exposure_share,
            "model_share_legacy": legacy_share,
            "model_share": share,
        }
    ).reindex(columns=MIXTURE_SHARE_AUDIT_COLUMNS)
    return out, audit


def assert_mixture_share_invariants(
    *, audit: pd.DataFrame, weights: pd.DataFrame, tolerance: float = 1e-9
) -> None:
    """The contract's invariants, in code (MIXTURE_ALLOCATOR_CONTRACT.md)."""
    missing = sorted(set(MIXTURE_SHARE_AUDIT_COLUMNS) - set(audit.columns))
    if missing:
        raise ValueError(f"mixture share audit is missing columns {missing}")

    share = pd.to_numeric(audit["model_share"], errors="coerce")
    if share.isna().any() or not np.isfinite(share.to_numpy(dtype=float)).all():
        raise ValueError("mixture shares carry a nonfinite value")
    if share.lt(-tolerance).any():
        raise ValueError("mixture shares carry a negative value")

    totals = share.groupby(
        [audit["jurisdiction_id"].astype(str), audit["state_fips"].astype(str), audit["offense"].astype(str)],
        dropna=False,
    ).sum()
    off_simplex = totals[(totals.sub(1.0).abs() > 1e-9) & (totals.abs() > 1e-9)]
    if not off_simplex.empty:
        raise ValueError(
            f"mixture shares do not sum to 1 within {len(off_simplex)} footprints, "
            f"e.g. {off_simplex.head(5).to_dict()}"
        )

    strata = set(audit["stratum"].astype(str))
    if not strata.issubset(set(STRATA)):
        raise ValueError(f"mixture share audit carries unknown strata {sorted(strata - set(STRATA))}")

    lookup = weight_lookup(weights)
    expected = np.array(
        [
            lookup[(str(offense), str(stratum))]
            for offense, stratum in zip(
                audit["offense"].astype(str), audit["stratum"].astype(str), strict=True
            )
        ],
        dtype=float,
    )
    published = audit[list(WEIGHT_COLUMNS)].to_numpy(dtype=float)
    if not np.allclose(published, expected, rtol=0.0, atol=tolerance):
        raise ValueError("mixture share audit publishes weights that are not the frozen table entry")

    blended = (
        published[:, 0] * pd.to_numeric(audit["prior_share"], errors="coerce").to_numpy(dtype=float)
        + published[:, 1] * pd.to_numeric(audit["gbm_share"], errors="coerce").to_numpy(dtype=float)
        + published[:, 2] * pd.to_numeric(audit["exposure_share"], errors="coerce").to_numpy(dtype=float)
    )
    group_codes, group_index = _group_codes(audit, ["jurisdiction_id", "state_fips", "offense"])
    recomposed = normalize_within_groups(blended, group_codes, len(group_index))
    if not np.allclose(recomposed, share.to_numpy(dtype=float), rtol=0.0, atol=1e-9):
        raise ValueError("mixture shares do not recompose from their published expert shares")


def summarize_mixture_shares(audit: pd.DataFrame) -> dict:
    share = pd.to_numeric(audit["model_share"], errors="coerce").astype(float)
    legacy = pd.to_numeric(audit["model_share_legacy"], errors="coerce").astype(float)
    group_keys = [
        audit["jurisdiction_id"].astype(str),
        audit["state_fips"].astype(str),
        audit["offense"].astype(str),
    ]
    tvd = (share - legacy).abs().groupby(group_keys, dropna=False).sum().mul(0.5)
    return {
        "support_rows": int(len(audit)),
        "footprints": int(tvd.shape[0]),
        "stratum_row_counts": {
            str(key): int(value) for key, value in audit["stratum"].astype(str).value_counts().items()
        },
        "mean_tvd_vs_legacy": float(tvd.mean()) if len(tvd) else 0.0,
        "median_tvd_vs_legacy": float(tvd.median()) if len(tvd) else 0.0,
        "max_tvd_vs_legacy": float(tvd.max()) if len(tvd) else 0.0,
        "unchanged_footprints": int((tvd <= 1e-12).sum()),
    }
