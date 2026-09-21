"""v2 uncertainty layer: coherent draws over controls and shares, stored as decisions.

`REVIEW_SOL_NEUTRAL.md` sec.7 is blunt about what the deployed surface's Poisson interval does and
does not answer. A Garwood interval around a point mean describes event-process variation. It says
nothing about the uncertainty in the jurisdiction CONTROL, nothing about ALLOCATION error when a
share model is transferred to a city it never saw, and nothing about the DENOMINATOR. The held-out
TVDs make the size of the gap concrete: a 0.30-0.40 TVD means roughly a third of the predicted
incident mass would have to move between block groups to reproduce a held-out city.

This lane implements sec.7's recommendation. Treat the count as `C = T * S`, decompose on the log
scale, `Var(log C) ~ Var(log T) + Var(log S)` (plus `Var(log E)` once a rate is formed), and
realise the decomposition as ~100 DETERMINISTIC draws per release from which only DECISIONS are
stored: quantiles, the probability the index exceeds the national reference, the probability the
cell belongs in the map bin it is painted, and a reliability tier keyed to that probability.

Five rules are load-bearing and enforced rather than trusted:

* **Conservation holds in every draw, not on average.** A draw perturbs the control for a
  jurisdiction-offense footprint and then draws a share vector that sums to one WITHIN that
  footprint, so the footprint's mass in draw d is exactly its drawn control. Nothing is
  renormalised afterwards and no draw can move mass across a footprint boundary.
* **Every dispersion is measured on a held-out design, and each cites its own.** The control's
  estimated mass takes E1's rolling-origin quasi-Poisson dispersion
  (`configs/uncertainty_control_dispersion_v1.csv`, from the tournament's 2022-2024 folds); its
  benchmark-imputed mass takes E5's pseudo-missingness multiplicative bounds
  (`configs/imputation_empirical_bounds.csv`); the share vector's concentration is set so its
  expected TVD reproduces the out-of-fold benchmark TVD E2 actually measured
  (`configs/uncertainty_share_dispersion_v1.csv`); and the whole predictive distribution is then
  recalibrated against measured coverage on held-out places
  (`configs/uncertainty_calibration_v1.csv`). No dispersion in this file was chosen to make a map
  look right.
* **The estimand is the structural expected count, not a realized one.** "The probability the cell
  truly belongs in the displayed bin" is a question about the underlying quantity, so the draws
  carry estimation uncertainty and NOT the Poisson realization layer. The existing
  `*_ci95_*` fields answer the realized-count question and are untouched; the two coexist because
  they are two different questions, and the contract says which is which.
* **Recalibration is monotone.** The calibration table is the empirical CDF of the probability
  integral transform on held-out cells. Applying it reparameterises the predictive CDF and can
  therefore rescale confidence but can never reorder cells, change a point estimate, or move a
  published count.
* **Doubt is data, never paint.** Every field here is a popup/metadata field. This module contains
  no display rule that mutes, hatches, greys or otherwise marks a cell as probably wrong, and it
  does not touch `index_*`, `rate_*`, `expected_count_*`, `estimate_mode_*`, `reliability_tier_*`
  or `recommended_display_geography_*`. The owner's display contract is that source disclosure is
  fine and doubt disclosure is not.

No AGS value enters anywhere: every dispersion traces to this pipeline's own held-out experiments
against police incident feeds and FBI panel data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.build_freshness import artifact_is_current, write_dependency_stamp
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths


# --- naming and versioning ---------------------------------------------------------------

UNCERTAINTY_LAYER_VERSION = "uncertainty_layer_v1"
CALIBRATION_VERSION = "uncertainty_calibration_v1"
SHARE_DISPERSION_VERSION = "uncertainty_share_dispersion_v1"
CONTROL_DISPERSION_VERSION = "uncertainty_control_dispersion_v1"

CONTROL_DISPERSION_FILENAME = "uncertainty_control_dispersion_v1.csv"
SHARE_DISPERSION_FILENAME = "uncertainty_share_dispersion_v1.csv"
CALIBRATION_FILENAME = "uncertainty_calibration_v1.csv"
IMPUTATION_BOUNDS_FILENAME = "imputation_empirical_bounds.csv"

# Deterministic by construction. Every stream is `default_rng([SEED, offense_index, draw_index,
# stream_index])`, so a draw is a function of its coordinates and never of call order, thread
# count or iteration order. There is no `Date.now`, no entropy source and no global seeding.
UNCERTAINTY_SEED = 20260805
DEFAULT_N_DRAWS = 100

# Support classes. `benchmark` is a control statement (the unit never reported and its level was
# imputed); `direct` and `model` are share statements (whether the cell's own incidents are
# admitted). They are ordered so the control statement wins: a benchmark-imputed unit has no
# incident feed by construction.
SUPPORT_CLASS_BENCHMARK = "benchmark"
SUPPORT_CLASS_DIRECT = "direct"
SUPPORT_CLASS_MODEL = "model"
SUPPORT_CLASSES: tuple[str, ...] = (SUPPORT_CLASS_DIRECT, SUPPORT_CLASS_MODEL, SUPPORT_CLASS_BENCHMARK)
# A benchmark unit's SHARE comes from the same transfer as any other model-only cell, so it reads
# the model class's calibration map. Only its CONTROL is different, and that difference is carried
# by the E5 bounds in the draw itself.
CALIBRATION_CLASS_FOR_SUPPORT: dict[str, str] = {
    SUPPORT_CLASS_DIRECT: SUPPORT_CLASS_DIRECT,
    SUPPORT_CLASS_MODEL: SUPPORT_CLASS_MODEL,
    SUPPORT_CLASS_BENCHMARK: SUPPORT_CLASS_MODEL,
}

# Jurisdiction-size strata, predeclared in PLAN.md Amendment 3 item 1 and reused verbatim so the
# share dispersion is read at the same strata E2 selected its weights at. This module mints no
# stratum of its own.
STRATUM_CORE = "core"
STRATUM_SUBURBAN = "suburban"
STRATUM_SMALL = "small"
STRATA: tuple[str, ...] = (STRATUM_CORE, STRATUM_SUBURBAN, STRATUM_SMALL)
STRATUM_CORE_POPULATION_MIN = 200_000.0
STRATUM_SUBURBAN_POPULATION_MIN = 50_000.0

# The published map's break set, carried to the map through the edition manifest, where the
# release validator checks it against this tuple. (The retired frontend inlined the same list as
# a literal `INDEX_BREAKS`.) The ramp interpolates continuously between these stops; a reader
# nonetheless distinguishes
# colour by which SEGMENT a value sits in, and the segment is what "the displayed bin" means here.
# Both ends are open, exactly as the legend says.
INDEX_BREAKS: tuple[float, ...] = (
    3.125,
    6.25,
    12.5,
    25.0,
    50.0,
    75.0,
    100.0,
    133.0,
    200.0,
    400.0,
    800.0,
    1600.0,
    3200.0,
    6400.0,
    12800.0,
)
INDEX_BIN_LABELS: tuple[str, ...] = (
    "<3.125",
    "3.125-6.25",
    "6.25-12.5",
    "12.5-25",
    "25-50",
    "50-75",
    "75-100",
    "100-133",
    "133-200",
    "200-400",
    "400-800",
    "800-1600",
    "1600-3200",
    "3200-6400",
    "6400-12800",
    ">=12800",
)
NATIONAL_REFERENCE_INDEX = 100.0

# The Dirichlet's log-scale dispersion is EXACT, not asymptotic:
#   Var(log s_g) = trigamma(A * p_g) - trigamma(A),
# verified against simulation to three decimals. Solving that for A is how the concentration is
# set, and the bracket below only has to contain the root.
MIN_SHARE_CONCENTRATION = 1e-3
MAX_SHARE_CONCENTRATION = 1e12
CONCENTRATION_BISECTION_STEPS = 80

# WHY NOT MATCH THE BENCHMARK TVD DIRECTLY. The obvious reading of sec.7 is to set the Dirichlet's
# concentration so its expected TVD from the point shares equals the TVD E2 measured out of fold.
# That was implemented first and MEASURED, and it does not work: E2's out-of-fold cell-level TVDs
# are 0.39-0.92 by offense and stratum, and a Dirichlet can only reach a TVD that large by
# collapsing -- at the concentration that gets closest (and it saturates near 0.64, never reaching
# 0.87) the per-cell Var(log s) is about 24, i.e. a factor of 140 either side. That is not an
# uncertainty statement, it is a degenerate one.
#
# The mechanism is the reason. TVD is an AGGREGATE MISALLOCATION statistic: it is large when a
# share model puts mass systematically in the wrong places, which is a correlated failure across
# cells. A Dirichlet is INDEPENDENT jitter per cell, and the only way independent jitter reproduces
# a correlated failure's L1 distance is by making each cell's jitter enormous. So the target is the
# per-cell quantity that identifies a per-cell spread: the measured log-scale dispersion of the
# share residual on held-out cities, in excess of Poisson. The TVD stays in the evidence as the
# statistic that motivated the lane and as the record of why this route was not taken.
SHARE_DISPERSION_TARGET_COLUMN = "target_log_sd"

# The probability grid the calibration map is stored on. Coarse on purpose: the calibration half
# holds tens of city-offence cells per offence once cells are equalised, and a finer grid would
# report noise as structure.
PIT_GRID: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)

# Reliability tiers, sec.7's rule verbatim.
TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"
TIER_WITHHELD = "withheld"
TIERS: tuple[str, ...] = (TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_WITHHELD)
TIER_HIGH_MIN_BIN_PROBABILITY = 0.80
TIER_MEDIUM_MIN_BIN_PROBABILITY = 0.60
TIER_LOW_BENCHMARK_SHARE_MAX = 0.50

# The two competing public person-exposure proxies this pipeline already publishes on every cell.
# sec.7 asks for the denominator's uncertainty to come from "competing public proxy surfaces or an
# empirically calibrated error distribution"; these are the competing surfaces, and treating them
# as one standard deviation either side of their geometric mean introduces no free constant.
PERSON_PROXY_COLUMNS: tuple[str, str] = ("daytime_population_jobs_proxy", "landscan_day_pop")
# Below the person-exposure floor the cell publishes no per-person rate at all, so the pair is not
# read there: the log ratio of two sub-floor surfaces is noise, not disagreement.
PERSON_PROXY_MIN_EXPOSURE = 50.0


# --- published columns -------------------------------------------------------------------

UNCERTAINTY_VERSION_COLUMN = "uncertainty_layer_version"


def count_quantile_column(offense: str, quantile: str) -> str:
    return f"expected_count_{offense}_{quantile}"


def rate_quantile_column(offense: str, quantile: str) -> str:
    return f"rate_{offense}_primary_{quantile}"


def index_quantile_column(offense: str, quantile: str) -> str:
    return f"index_{offense}_primary_{quantile}"


def prob_above_reference_column(offense: str) -> str:
    return f"prob_index_{offense}_above_100"


def displayed_bin_column(offense: str) -> str:
    return f"displayed_index_bin_{offense}"


def prob_displayed_bin_column(offense: str) -> str:
    return f"prob_displayed_index_bin_{offense}"


def decision_tier_column(offense: str) -> str:
    return f"decision_reliability_tier_{offense}"


def support_class_column(offense: str) -> str:
    return f"uncertainty_support_class_{offense}"


def control_log_sd_column(offense: str) -> str:
    return f"uncertainty_control_log_sd_{offense}"


def share_log_sd_column(offense: str) -> str:
    return f"uncertainty_share_log_sd_{offense}"


def denominator_log_sd_column(offense: str) -> str:
    return f"uncertainty_denominator_log_sd_{offense}"


COUNT_QUANTILES: tuple[str, ...] = ("p10", "p50", "p90")
RATE_QUANTILES: tuple[str, ...] = ("p10", "p50", "p90")
INDEX_QUANTILES: tuple[str, ...] = ("p10", "p25", "p50", "p75", "p90")
QUANTILE_LEVELS: dict[str, float] = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}


def uncertainty_columns_for_offense(offense: str) -> tuple[str, ...]:
    return (
        *(count_quantile_column(offense, q) for q in COUNT_QUANTILES),
        *(rate_quantile_column(offense, q) for q in RATE_QUANTILES),
        *(index_quantile_column(offense, q) for q in INDEX_QUANTILES),
        prob_above_reference_column(offense),
        displayed_bin_column(offense),
        prob_displayed_bin_column(offense),
        decision_tier_column(offense),
        support_class_column(offense),
        control_log_sd_column(offense),
        share_log_sd_column(offense),
        denominator_log_sd_column(offense),
    )


def uncertainty_published_columns() -> tuple[str, ...]:
    columns: list[str] = [UNCERTAINTY_VERSION_COLUMN]
    for offense in OFFENSES_7:
        columns.extend(uncertainty_columns_for_offense(offense))
    return tuple(columns)


# The rate and index quantiles are per-offense POINT payload in exactly the sense the rare-offense
# tract-support policy means: a reader would take them as this block group's own value for murder.
# The COUNT quantiles are not -- the count is conserved and published at block group already.
def rare_offense_suppressed_columns(offense: str) -> tuple[str, ...]:
    return (
        *(rate_quantile_column(offense, q) for q in RATE_QUANTILES),
        *(index_quantile_column(offense, q) for q in INDEX_QUANTILES),
        prob_above_reference_column(offense),
        displayed_bin_column(offense),
        prob_displayed_bin_column(offense),
        decision_tier_column(offense),
    )


# --- small deterministic helpers ----------------------------------------------------------


def stable_fold(key: str, n_folds: int) -> int:
    """Process-stable fold assignment. Python's `hash()` is salted per interpreter, so the E2
    nested protocol's `hash(jid) % 5` is not reproducible across runs; this is."""
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % int(n_folds)


def stratum_for_population(population: float) -> str:
    value = float(population) if np.isfinite(population) else 0.0
    if value >= STRATUM_CORE_POPULATION_MIN:
        return STRATUM_CORE
    if value >= STRATUM_SUBURBAN_POPULATION_MIN:
        return STRATUM_SUBURBAN
    return STRATUM_SMALL


def dirichlet_log_share_sd(concentration: np.ndarray, share: np.ndarray) -> np.ndarray:
    """`sd(log s_g)` for a Dirichlet with total concentration A and mean share p_g.

    Exact: `Var(log s_g) = trigamma(A p_g) - trigamma(A)`, since `s_g` is the ratio of one Gamma to
    the sum and the log of a Gamma has trigamma variance. Used both to SET the concentration and,
    in the tests, to check the setting against simulation.
    """
    from scipy.special import polygamma

    alpha = np.clip(np.asarray(concentration, dtype=float) * np.asarray(share, dtype=float), 1e-12, None)
    total = np.clip(np.asarray(concentration, dtype=float), 1e-12, None)
    return np.sqrt(np.clip(polygamma(1, alpha) - polygamma(1, total), 0.0, None))


def concentration_from_log_share_dispersion(
    median_share: np.ndarray, target_log_sd: np.ndarray
) -> np.ndarray:
    """Dirichlet total concentration whose median cell has the MEASURED log-share dispersion.

    Monotone decreasing in A, so a plain bisection on the exact trigamma identity is both simplest
    and safest -- no closed form to be wrong about, no asymptotic regime to leave. The measured
    dispersion is anchored at the group's MEDIAN share; the gradient across cell sizes (a small
    block group is less certain than a large one) is then the Dirichlet's own and is not imposed.
    """
    share = np.asarray(median_share, dtype=float)
    target = np.asarray(target_log_sd, dtype=float)
    low = np.full(share.shape, MIN_SHARE_CONCENTRATION, dtype=float)
    high = np.full(share.shape, MAX_SHARE_CONCENTRATION, dtype=float)
    for _ in range(CONCENTRATION_BISECTION_STEPS):
        middle = np.sqrt(low * high)
        wider = dirichlet_log_share_sd(middle, share) > target
        low = np.where(wider, middle, low)
        high = np.where(wider, high, middle)
    concentration = np.sqrt(low * high)
    # A group with no mass, or a target that is not a positive number, gets the tightest
    # admissible concentration rather than the loosest: an undefined target must not open the
    # interval, and such a group publishes no rate anyway.
    undefined = ~np.isfinite(share) | (share <= 0.0) | ~np.isfinite(target) | (target <= 0.0)
    return np.where(undefined, MAX_SHARE_CONCENTRATION, concentration)


def index_bin_indices(values: np.ndarray) -> np.ndarray:
    """Which segment of the published break set a value paints in. NaN -> -1."""
    array = np.asarray(values, dtype=float)
    bins = np.searchsorted(np.asarray(INDEX_BREAKS, dtype=float), array, side="right")
    return np.where(np.isfinite(array), bins, -1).astype(np.int64)


def index_bin_labels(values: np.ndarray) -> np.ndarray:
    indices = index_bin_indices(values)
    labels = np.asarray(INDEX_BIN_LABELS, dtype=object)
    out = np.full(len(indices), None, dtype=object)
    valid = indices >= 0
    out[valid] = labels[indices[valid]]
    return out


def _monotone(values: np.ndarray) -> np.ndarray:
    """Strictly increasing copy, so `np.interp` has a well-defined inverse at ties."""
    increasing = np.maximum.accumulate(np.asarray(values, dtype=float))
    return increasing + np.arange(len(increasing), dtype=float) * 1e-12


# --- calibration map ----------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationMap:
    """Empirical CDF of the held-out probability integral transform, as (u, G(u)) on PIT_GRID.

    `apply` turns a raw predictive probability into a calibrated one; `invert` turns a nominal
    quantile level into the level the draws must be read at to deliver it. Both are monotone, so
    neither can reorder cells.
    """

    support_class: str
    offense: str
    u: np.ndarray
    g: np.ndarray

    def apply(self, probability: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(np.clip(probability, 0.0, 1.0), self.u, self.g), 0.0, 1.0)

    def invert(self, level: float) -> float:
        return float(np.clip(np.interp(float(level), _monotone(self.g), self.u), 0.0, 1.0))


IDENTITY_CALIBRATION = CalibrationMap(
    support_class="identity",
    offense="all",
    u=np.asarray(PIT_GRID, dtype=float),
    g=np.asarray(PIT_GRID, dtype=float),
)


# --- configuration -------------------------------------------------------------------------


def control_dispersion_path(config_dir: Path) -> Path:
    return Path(config_dir) / CONTROL_DISPERSION_FILENAME


def share_dispersion_path(config_dir: Path) -> Path:
    return Path(config_dir) / SHARE_DISPERSION_FILENAME


def calibration_table_path(config_dir: Path) -> Path:
    return Path(config_dir) / CALIBRATION_FILENAME


def imputation_bounds_path(config_dir: Path) -> Path:
    return Path(config_dir) / IMPUTATION_BOUNDS_FILENAME


def load_control_dispersion(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    missing = sorted(set(OFFENSES_7) - set(frame["offense"].astype(str)))
    if missing:
        raise ValueError(f"{path}: missing control dispersion for {missing}")
    return {
        str(row.offense): float(row.quasi_poisson_dispersion)
        for row in frame.itertuples()
        if str(row.offense) in set(OFFENSES_7)
    }


def load_share_dispersion(path: Path) -> dict[tuple[str, str], float]:
    frame = pd.read_csv(path)
    if SHARE_DISPERSION_TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"{path}: expected a {SHARE_DISPERSION_TARGET_COLUMN} column. A table written against "
            "the retired TVD-matching rule cannot be read by the log-dispersion rule; regenerate "
            "it with scripts/diagnostics/uncertainty_calibration.py --write-configs."
        )
    table = {
        (str(row.offense), str(row.stratum)): float(getattr(row, SHARE_DISPERSION_TARGET_COLUMN))
        for row in frame.itertuples()
    }
    missing = [
        (offense, stratum)
        for offense in OFFENSES_7
        for stratum in STRATA
        if (offense, stratum) not in table
    ]
    if missing:
        raise ValueError(f"{path}: missing share dispersion for {missing}")
    return table


def load_calibration(path: Path) -> dict[tuple[str, str], CalibrationMap]:
    frame = pd.read_csv(path)
    maps: dict[tuple[str, str], CalibrationMap] = {}
    for (support_class, offense), group in frame.groupby(["support_class", "offense"], sort=True):
        ordered = group.sort_values("u", kind="mergesort")
        maps[(str(support_class), str(offense))] = CalibrationMap(
            support_class=str(support_class),
            offense=str(offense),
            u=ordered["u"].to_numpy(dtype=float),
            g=np.clip(ordered["g"].to_numpy(dtype=float), 0.0, 1.0),
        )
    return maps


def load_imputation_bounds(path: Path, *, rule_version: str = "v2") -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["rule_version"].astype(str).eq(str(rule_version))].copy()
    if frame.empty:
        raise ValueError(f"{path}: no rows for rule_version={rule_version!r}")
    return frame


@dataclass(frozen=True)
class UncertaintyRuntime:
    version: str
    n_draws: int
    seed: int
    control_dispersion: dict[str, float]
    share_dispersion: dict[tuple[str, str], float]
    calibration: dict[tuple[str, str], CalibrationMap]
    imputation_bounds: pd.DataFrame
    config_paths: tuple[Path, ...]

    def calibration_map(self, *, support_class: str, offense: str) -> CalibrationMap:
        key = (CALIBRATION_CLASS_FOR_SUPPORT.get(support_class, SUPPORT_CLASS_MODEL), str(offense))
        return self.calibration.get(key, IDENTITY_CALIBRATION)

    def signature(self) -> str:
        digest = hashlib.blake2b(digest_size=8)
        digest.update(self.version.encode())
        digest.update(f"{int(self.n_draws)}|{int(self.seed)}".encode())
        for path in self.config_paths:
            resolved = Path(path)
            if resolved.exists():
                digest.update(resolved.read_bytes())
        return digest.hexdigest()


def resolve_uncertainty_runtime(
    paths: RepoPaths,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = UNCERTAINTY_SEED,
    control_dispersion_file: Path | None = None,
    share_dispersion_file: Path | None = None,
    calibration_file: Path | None = None,
) -> UncertaintyRuntime:
    config_dir = paths.repo_root / "configs"
    control_path = Path(control_dispersion_file) if control_dispersion_file else control_dispersion_path(config_dir)
    share_path = Path(share_dispersion_file) if share_dispersion_file else share_dispersion_path(config_dir)
    calibration_path = Path(calibration_file) if calibration_file else calibration_table_path(config_dir)
    bounds_path = imputation_bounds_path(config_dir)
    missing = [path for path in (control_path, share_path, calibration_path, bounds_path) if not path.exists()]
    if missing:
        listed = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "The uncertainty layer is enabled but its measured dispersion tables are missing:\n"
            f"{listed}\n"
            "Regenerate them with scripts/diagnostics/uncertainty_calibration.py --write-configs. "
            "This lane fails closed rather than substituting an unmeasured dispersion."
        )
    return UncertaintyRuntime(
        version=UNCERTAINTY_LAYER_VERSION,
        n_draws=int(n_draws),
        seed=int(seed),
        control_dispersion=load_control_dispersion(control_path),
        share_dispersion=load_share_dispersion(share_path),
        calibration=load_calibration(calibration_path),
        imputation_bounds=load_imputation_bounds(bounds_path),
        config_paths=(control_path, share_path, calibration_path, bounds_path),
    )


# --- control mass decomposition ------------------------------------------------------------

CONTROL_HARD_CLASS = "measured"
CONTROL_ESTIMATED_CLASS = "estimated_from_history"
CONTROL_BENCHMARK_CLASS = "benchmark_imputed"
CONTROL_MASS_CLASSES: tuple[str, ...] = (
    CONTROL_HARD_CLASS,
    CONTROL_ESTIMATED_CLASS,
    CONTROL_BENCHMARK_CLASS,
)


def control_mass_fractions(controls: pd.DataFrame) -> pd.DataFrame:
    """Split each jurisdiction-offense control into measured / estimated / benchmark-imputed mass.

    `estimated_count_ags_core` reconciles to
    `observed + partial + fill + benchmark_imputed`, and the estimated PART of the partial
    component is its uplift, so the measured mass is everything the agencies actually reported and
    the estimated mass is exactly what the panel had to supply. That is the split the perturbation
    needs: only supplied mass carries estimation error.
    """
    frame = controls.copy()

    def numeric(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(0.0, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)

    total = numeric("estimated_count_ags_core")
    benchmark = numeric("benchmark_imputed_count")
    estimated = numeric("fill_component_count") + numeric("partial_reporting_uplift_count")
    estimated = np.minimum(estimated, np.maximum(total - benchmark, 0.0))
    denominator = total.where(total.gt(0.0), np.nan)
    out = pd.DataFrame(
        {
            "jurisdiction_id": frame["jurisdiction_id"].astype("string"),
            "offense": frame["offense"].astype("string"),
            "bucket_population": pd.to_numeric(frame.get("bucket_population"), errors="coerce").fillna(0.0),
            "jurisdiction_type": frame.get(
                "jurisdiction_type", pd.Series("municipal", index=frame.index)
            ).astype("string"),
            "benchmark_fraction": (benchmark / denominator).fillna(0.0).clip(0.0, 1.0),
            "estimated_fraction": (pd.Series(estimated, index=frame.index) / denominator)
            .fillna(0.0)
            .clip(0.0, 1.0),
        }
    )
    out["measured_fraction"] = (
        1.0 - out["benchmark_fraction"] - out["estimated_fraction"]
    ).clip(lower=0.0)
    return out.drop_duplicates(["jurisdiction_id", "offense"], keep="first").reset_index(drop=True)


def _benchmark_bound_lookup(bounds: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """(lane, offense) -> (lo80, hi80) multiplicative bounds from the E5 pseudo-missingness run."""
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in bounds.itertuples():
        lookup[(str(row.lane), str(row.offense))] = (float(row.mult_lo80), float(row.mult_hi80))
    return lookup


def benchmark_multiplier_sigmas(
    *,
    lanes: np.ndarray,
    offense: str,
    bounds: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-lognormal sigmas that reproduce E5's measured 80% multiplicative interval.

    The measured bounds are strongly asymmetric (a masked small municipality is under-stated far
    more often than it is over-stated), so a symmetric sigma would misdescribe them. Two sigmas
    keep the measurement intact: `sigma_lo` from the lower bound, `sigma_hi` from the upper.
    A lane with no measured row of its own falls back to the WIDEST lane on record, because an
    unmeasured lane is less known and not more.
    """
    lookup = _benchmark_bound_lookup(bounds)
    z80 = 1.2815515655446004  # the 90th percentile of the standard normal
    widest_lo, widest_hi = 1.0, 1.0
    for (lane, off), (lo, hi) in lookup.items():
        if off != offense:
            continue
        widest_lo = min(widest_lo, lo)
        widest_hi = max(widest_hi, hi)
    sigma_lo = np.zeros(len(lanes), dtype=float)
    sigma_hi = np.zeros(len(lanes), dtype=float)
    for index, lane in enumerate(lanes):
        lo, hi = lookup.get((str(lane), offense), (widest_lo, widest_hi))
        sigma_lo[index] = max(-np.log(max(lo, 1e-9)), 0.0) / z80
        sigma_hi[index] = max(np.log(max(hi, 1e-9)), 0.0) / z80
    return sigma_lo, sigma_hi


BENCHMARK_LANE_MUNICIPAL = "municipal_jurisdiction"
BENCHMARK_LANE_COUNTY = "county_remainder"


def benchmark_lane_for_jurisdiction_type(jurisdiction_type: pd.Series) -> pd.Series:
    kind = jurisdiction_type.astype("string").fillna("")
    return pd.Series(
        np.where(kind.str.contains("remainder", na=False), BENCHMARK_LANE_COUNTY, BENCHMARK_LANE_MUNICIPAL),
        index=jurisdiction_type.index,
        dtype="string",
    )


# --- the draw engine -------------------------------------------------------------------------


@dataclass
class _OffenseGeometry:
    """Everything one offense's draws need, laid out for vectorised group arithmetic."""

    bg_row: np.ndarray            # component -> row in the block-group frame
    group_index: np.ndarray       # component -> group ordinal (sorted, contiguous)
    group_starts: np.ndarray      # reduceat offsets
    source_state: np.ndarray      # group -> control/calibration state
    component_count: np.ndarray   # point allocation per component
    group_total: np.ndarray       # point control per group
    shares: np.ndarray            # component share within its group
    concentration: np.ndarray     # Dirichlet total per group
    measured_fraction: np.ndarray
    estimated_fraction: np.ndarray
    benchmark_fraction: np.ndarray
    estimated_sigma: np.ndarray
    benchmark_sigma_lo: np.ndarray
    benchmark_sigma_hi: np.ndarray


@dataclass
class OffenseDraws:
    """Per-block-group draws for one offense: the full draw and its two single-component twins.

    The twins exist so `Var(log C) ~ Var(log T) + Var(log S)` is PUBLISHED rather than asserted.
    They are the same draw with one component held at its point value, so the three matrices are
    coherent draw by draw and their log dispersions decompose the full one.
    """

    counts: np.ndarray            # (n_bg, n_draws) float32, control and share both moving
    control_only: np.ndarray      # (n_bg, n_draws) float32, shares held at the point vector
    share_only: np.ndarray        # (n_bg, n_draws) float32, controls held at the point total
    point_counts: np.ndarray      # (n_bg,)


def quasi_poisson_lognormal_sigma(
    dispersion: float, estimated_mass: np.ndarray
) -> np.ndarray:
    """Convert quasi-Poisson relative variance to a lognormal log-scale spread.

    For ``Var(Y) = phi * E[Y]``, the squared coefficient of variation is
    ``phi / E[Y]``.  A lognormal multiplier with log-scale standard deviation
    ``sigma`` has ``CV^2 = exp(sigma^2) - 1``.  Solving that identity avoids the
    small-CV approximation ``sigma = sqrt(phi / mass)``, which diverges for tiny
    estimated controls and can overflow otherwise valid Monte Carlo draws.
    """
    mass = np.asarray(estimated_mass, dtype=float)
    relative_variance = np.divide(
        float(dispersion),
        mass,
        out=np.zeros_like(mass, dtype=float),
        where=mass > 0.0,
    )
    return np.sqrt(np.log1p(relative_variance))


def log_dispersion(draws: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Standard deviation of `log(draw)` where the point value is positive; NaN where it is not.

    A cell with no expected mass has no log scale to be uncertain on, and reporting 0.0 there
    would read as "known exactly" rather than "not defined".
    """
    values = np.asarray(draws, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(values > 0.0, np.log(values), np.nan)
    positive = np.asarray(point, dtype=float) > 0.0
    out = np.full(values.shape[0], np.nan, dtype=float)
    if positive.any():
        block = logs[positive]
        out[positive] = np.nanstd(np.where(np.isfinite(block), block, np.nan), axis=1)
    return out


class UncertaintyEngine:
    """Builds the draws once and answers every surface from them.

    The four published surfaces (block group / tract, AGS-core / FBI-calibrated) are the SAME
    draws: the tract surface is the block-group draws summed within tract, and the calibrated twin
    is the same draws times the same per-state scalar its counts were multiplied by. Drawing them
    separately would let the four surfaces disagree about the same cell.
    """

    def __init__(
        self,
        *,
        components: pd.DataFrame,
        controls: pd.DataFrame,
        component_audit: pd.DataFrame,
        block_group_ids: pd.Series,
        runtime: UncertaintyRuntime,
    ) -> None:
        self.runtime = runtime
        self.block_group_ids = pd.Index(
            pd.Series(block_group_ids).astype("string").str.zfill(12), name="block_group_geoid"
        )
        self._row_of_bg = pd.Series(
            np.arange(len(self.block_group_ids), dtype=np.int64), index=self.block_group_ids
        )
        self._controls = control_mass_fractions(controls)
        self._components = self._prepare_components(components)
        self._direct_concentration = self._prepare_direct_concentration(component_audit)
        self._geometry: dict[str, _OffenseGeometry] = {}

    # -- preparation -----------------------------------------------------------------

    def _prepare_components(self, components: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "bg_id",
            "jurisdiction_id",
            "jurisdiction_type",
            "offense",
            "component_count",
        ]
        if "source_state_fips" in components.columns:
            columns.append("source_state_fips")
        frame = components[columns].copy()
        frame["bg_id"] = frame["bg_id"].astype("string").str.zfill(12)
        frame["jurisdiction_id"] = frame["jurisdiction_id"].astype("string")
        frame["jurisdiction_type"] = frame["jurisdiction_type"].astype("string")
        frame["offense"] = frame["offense"].astype("string")
        frame["component_count"] = (
            pd.to_numeric(frame["component_count"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        geographic_state = frame["bg_id"].str.slice(0, 2)
        if "source_state_fips" not in frame.columns:
            frame["source_state_fips"] = geographic_state
        else:
            source = frame["source_state_fips"].astype("string").str.strip()
            frame["source_state_fips"] = source.where(
                source.ne("") & source.notna(), geographic_state
            )
        frame["source_state_fips"] = (
            frame["source_state_fips"].astype("string").str.zfill(2)
        )
        frame["_row"] = frame["bg_id"].map(self._row_of_bg)
        # A component whose block group is not on the published frame (excluded state, dropped
        # geography) contributes to no published cell; it is dropped here rather than silently
        # scattered onto row 0.
        frame = frame[frame["_row"].notna()].copy()
        frame["_row"] = frame["_row"].astype(np.int64)
        return frame

    def _prepare_direct_concentration(self, component_audit: pd.DataFrame) -> pd.DataFrame:
        """Dirichlet-multinomial concentration for a footprint whose own incidents are admitted.

        This is the production posterior's own pseudo-count total -- admitted incidents plus the
        prior mass the posterior actually used -- read off the audit rather than re-derived, so the
        draw is the posterior the surface was built from and not a second opinion about it.
        """
        columns = {"jurisdiction_id", "jurisdiction_type", "offense"}
        if component_audit is None or component_audit.empty or not columns.issubset(component_audit.columns):
            return pd.DataFrame(
                columns=["jurisdiction_id", "jurisdiction_type", "offense", "direct_concentration"]
            )
        frame = component_audit.copy()
        active = (
            frame["city_incident_posterior_active"].astype("boolean").fillna(False).astype(bool)
            if "city_incident_posterior_active" in frame.columns
            else pd.Series(False, index=frame.index)
        )
        frame = frame[active]
        if frame.empty:
            return pd.DataFrame(
                columns=["jurisdiction_id", "jurisdiction_type", "offense", "direct_concentration"]
            )
        frame["_incidents"] = pd.to_numeric(frame.get("incident_count"), errors="coerce").fillna(0.0).clip(lower=0.0)
        frame["_alpha"] = pd.to_numeric(frame.get("city_posterior_alpha"), errors="coerce").fillna(0.0).clip(lower=0.0)
        grouped = frame.groupby(["jurisdiction_id", "jurisdiction_type", "offense"], dropna=False).agg(
            incidents=("_incidents", "sum"), alpha=("_alpha", "max")
        ).reset_index()
        grouped["direct_concentration"] = grouped["incidents"] + grouped["alpha"]
        grouped["jurisdiction_id"] = grouped["jurisdiction_id"].astype("string")
        grouped["jurisdiction_type"] = grouped["jurisdiction_type"].astype("string")
        grouped["offense"] = grouped["offense"].astype("string")
        return grouped[["jurisdiction_id", "jurisdiction_type", "offense", "direct_concentration"]]

    def _geometry_for(self, offense: str) -> _OffenseGeometry:
        cached = self._geometry.get(offense)
        if cached is not None:
            return cached
        frame = self._components[self._components["offense"].eq(offense)].copy()
        frame = frame.sort_values(
            ["jurisdiction_id", "jurisdiction_type", "bg_id"], kind="mergesort"
        )
        keys = pd.MultiIndex.from_arrays(
            [frame["jurisdiction_id"].to_numpy(), frame["jurisdiction_type"].to_numpy()]
        )
        codes, uniques = pd.factorize(keys, sort=False)
        group_index = codes.astype(np.int64)
        n_groups = int(len(uniques))
        counts = frame["component_count"].to_numpy(dtype=float)
        group_total = np.bincount(group_index, weights=counts, minlength=n_groups)
        shares = np.divide(
            counts,
            group_total[group_index],
            out=np.zeros(len(counts), dtype=float),
            where=group_total[group_index] > 0.0,
        )
        starts = np.zeros(n_groups, dtype=np.int64)
        if len(group_index):
            boundaries = np.flatnonzero(np.diff(group_index)) + 1
            starts = np.concatenate([[0], boundaries]).astype(np.int64)

        group_frame = pd.DataFrame(
            {
                "jurisdiction_id": pd.Series(
                    [key[0] for key in uniques], dtype="string"
                ),
                "jurisdiction_type": pd.Series(
                    [key[1] for key in uniques], dtype="string"
                ),
            }
        )
        source_cardinality = frame.groupby(group_index, sort=False)[
            "source_state_fips"
        ].nunique()
        if bool(source_cardinality.gt(1).any()):
            raise ValueError(
                "uncertainty layer: one allocation group carries multiple source states; "
                "source-state calibration would be incoherent"
            )
        source_state = (
            frame.groupby(group_index, sort=False)["source_state_fips"]
            .first()
            .to_numpy(dtype=object)
        )
        group_frame["offense"] = offense
        merged = group_frame.merge(
            self._controls[self._controls["offense"].eq(offense)],
            on=["jurisdiction_id", "offense"],
            how="left",
            suffixes=("", "_control"),
        )
        # A footprint with no control row of its own is a modelled sub-target by construction
        # (county remainders, overlap layers). Fail WIDE: treat its whole mass as estimated rather
        # than as measured, which is the reading that cannot understate what is unknown.
        unmatched = merged["measured_fraction"].isna()
        merged["measured_fraction"] = merged["measured_fraction"].fillna(0.0)
        merged["estimated_fraction"] = merged["estimated_fraction"].fillna(0.0)
        merged["benchmark_fraction"] = merged["benchmark_fraction"].fillna(0.0)
        merged.loc[unmatched, "estimated_fraction"] = 1.0
        merged["bucket_population"] = merged["bucket_population"].fillna(0.0)

        dispersion = float(self.runtime.control_dispersion[str(offense)])
        estimated_mass = (
            merged["estimated_fraction"].to_numpy(dtype=float) * group_total
        )
        estimated_sigma = quasi_poisson_lognormal_sigma(dispersion, estimated_mass)
        lanes = benchmark_lane_for_jurisdiction_type(
            merged["jurisdiction_type"]
        ).to_numpy()
        sigma_lo, sigma_hi = benchmark_multiplier_sigmas(
            lanes=lanes, offense=str(offense), bounds=self.runtime.imputation_bounds
        )

        direct = merged.merge(
            self._direct_concentration,
            on=["jurisdiction_id", "jurisdiction_type", "offense"],
            how="left",
        )
        strata = [
            stratum_for_population(value)
            for value in merged["bucket_population"].to_numpy()
        ]
        # The group's median POSITIVE share, which is what the measured dispersion is anchored at.
        median_share = np.zeros(n_groups, dtype=float)
        for group in range(n_groups):
            start = starts[group]
            stop = starts[group + 1] if group + 1 < n_groups else len(shares)
            block = shares[start:stop]
            positive = block[block > 0.0]
            median_share[group] = float(np.median(positive)) if positive.size else 0.0
        target_log_sd = np.array(
            [
                self.runtime.share_dispersion[(str(offense), stratum)]
                for stratum in strata
            ],
            dtype=float,
        )
        concentration = concentration_from_log_share_dispersion(
            median_share, target_log_sd
        )
        # A footprint whose own incidents are admitted uses the production posterior's OWN
        # pseudo-count total instead: that is a measured Dirichlet-multinomial, not a transferred
        # one, and substituting the model-transfer dispersion there would overstate what is
        # unknown about a city that reported its own incidents.
        direct_values = pd.to_numeric(
            direct["direct_concentration"], errors="coerce"
        ).to_numpy(dtype=float)
        has_direct = np.isfinite(direct_values) & (direct_values > 0.0)
        concentration = np.where(has_direct, direct_values, concentration)

        geometry = _OffenseGeometry(
            bg_row=frame["_row"].to_numpy(dtype=np.int64),
            group_index=group_index,
            group_starts=starts,
            source_state=source_state,
            component_count=counts,
            group_total=group_total,
            shares=shares,
            concentration=concentration,
            measured_fraction=merged["measured_fraction"].to_numpy(dtype=float),
            estimated_fraction=merged["estimated_fraction"].to_numpy(dtype=float),
            benchmark_fraction=merged["benchmark_fraction"].to_numpy(dtype=float),
            estimated_sigma=estimated_sigma,
            benchmark_sigma_lo=sigma_lo,
            benchmark_sigma_hi=sigma_hi,
        )
        self._geometry[offense] = geometry
        return geometry

    # -- the draws --------------------------------------------------------------------

    def _control_draw(self, geometry: _OffenseGeometry, *, offense_index: int, draw: int) -> np.ndarray:
        """Median-preserving multiplicative perturbation of the ESTIMATED mass only.

        `T_draw = T * (f_measured + f_estimated * m_est + f_benchmark * m_bench)`. A control that
        is entirely an admitted full-year report is therefore unmoved: for the annual accounting
        surface the report IS the measurement, and inventing spread around it would be doubt
        disclosure with extra steps. The risk surface, whose estimand is next year, is where E1's
        rolling-origin dispersion applies to the whole control; that is a different lane.
        """
        rng = np.random.default_rng([self.runtime.seed, int(offense_index), int(draw), 1])
        n_groups = len(geometry.group_total)
        z_estimated = rng.standard_normal(n_groups)
        estimated_multiplier = np.exp(geometry.estimated_sigma * z_estimated)
        z_benchmark = rng.standard_normal(n_groups)
        sigma = np.where(z_benchmark < 0.0, geometry.benchmark_sigma_lo, geometry.benchmark_sigma_hi)
        benchmark_multiplier = np.exp(sigma * z_benchmark)
        factor = (
            geometry.measured_fraction
            + geometry.estimated_fraction * estimated_multiplier
            + geometry.benchmark_fraction * benchmark_multiplier
        )
        return geometry.group_total * np.clip(factor, 0.0, None)

    def _share_draw(self, geometry: _OffenseGeometry, *, offense_index: int, draw: int) -> np.ndarray:
        """Dirichlet share vector, normalised WITHIN the footprint so the draw conserves exactly."""
        rng = np.random.default_rng([self.runtime.seed, int(offense_index), int(draw), 2])
        alpha = geometry.concentration[geometry.group_index] * geometry.shares
        positive = alpha > 0.0
        gamma = np.zeros(len(alpha), dtype=float)
        if positive.any():
            gamma[positive] = rng.gamma(shape=alpha[positive])
        if not len(gamma):
            return gamma
        totals = np.add.reduceat(gamma, geometry.group_starts)
        denominator = totals[geometry.group_index]
        # A footprint whose every component drew exactly zero (possible only when the whole
        # footprint has zero mass) keeps the point shares, which are themselves zero.
        return np.divide(
            gamma,
            denominator,
            out=geometry.shares.copy(),
            where=denominator > 0.0,
        )

    def offense_draws(
        self,
        offense: str,
        *,
        source_state_multipliers: dict[str, float] | None = None,
    ) -> OffenseDraws:
        """Return draws, optionally scaled by each allocation group's source state.

        Repeating this call with multipliers uses the same deterministic random streams as the
        unscaled call. This lets a calibrated companion reuse the exact AGS-core control/share
        draws while a service-wide component placed over a state line retains its source state's
        calibration factor.
        """
        geometry = self._geometry_for(offense)
        offense_index = list(OFFENSES_7).index(str(offense))
        n_bg = len(self.block_group_ids)
        n_draws = int(self.runtime.n_draws)
        counts = np.zeros((n_bg, n_draws), dtype=np.float32)
        control_only = np.zeros((n_bg, n_draws), dtype=np.float32)
        share_only = np.zeros((n_bg, n_draws), dtype=np.float32)
        group_scale = np.ones(len(geometry.group_total), dtype=float)
        if source_state_multipliers is not None:
            group_scale = np.array(
                [
                    source_state_multipliers.get(str(state), 1.0)
                    for state in geometry.source_state
                ],
                dtype=float,
            )
            if not bool(np.isfinite(group_scale).all()) or bool(
                (group_scale < 0.0).any()
            ):
                raise ValueError(
                    "source-state calibration multipliers must be finite and non-negative"
                )
        point = np.bincount(
            geometry.bg_row,
            weights=geometry.component_count * group_scale[geometry.group_index],
            minlength=n_bg,
        )
        for draw in range(n_draws):
            control = (
                self._control_draw(geometry, offense_index=offense_index, draw=draw)
                * group_scale
            )
            shares = self._share_draw(geometry, offense_index=offense_index, draw=draw)
            group_of = geometry.group_index
            # The published interval: both components move together in the same draw.
            counts[:, draw] = np.bincount(
                geometry.bg_row, weights=control[group_of] * shares, minlength=n_bg
            )
            # The two single-component twins from the SAME draw, so the decomposition the
            # contract states is the one the artifact publishes.
            control_only[:, draw] = np.bincount(
                geometry.bg_row,
                weights=control[group_of] * geometry.shares,
                minlength=n_bg,
            )
            share_only[:, draw] = np.bincount(
                geometry.bg_row,
                weights=geometry.group_total[group_of] * group_scale[group_of] * shares,
                minlength=n_bg,
            )
        return OffenseDraws(
            counts=counts,
            control_only=control_only,
            share_only=share_only,
            point_counts=point,
        )


def rollup_draws(draws: np.ndarray, *, group_row: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum a draw matrix within groups, draw by draw.

    Aggregation happens WITHIN a draw and never across draws: the tract's uncertainty is the
    uncertainty of the sum, not the sum of the uncertainties, and its block groups share the same
    control error rather than averaging it away.
    """
    out = np.zeros((int(n_groups), draws.shape[1]), dtype=np.float32)
    for column in range(draws.shape[1]):
        out[:, column] = np.bincount(
            group_row, weights=draws[:, column].astype(np.float64), minlength=int(n_groups)
        )
    return out


# --- turning draws into decisions ------------------------------------------------------------


def denominator_log_sd(frame: pd.DataFrame, *, offense: str) -> pd.Series:
    """Log-scale spread of the person denominator, from the two competing public proxy surfaces.

    Only the person-denominator offenses have two competing public surfaces on the frame. Burglary
    premises and the motor-vehicle-theft vehicle denominator have exactly one construction each, so
    their term is 0.0 -- PUBLISHED as 0.0, so a reader sees the component is absent instead of
    assuming it was included.
    """
    from crimerisk.denominators import PERSON_EXPOSURE_DENOMINATOR_OFFENSES

    zero = pd.Series(0.0, index=frame.index, dtype=float)
    if str(offense) not in set(PERSON_EXPOSURE_DENOMINATOR_OFFENSES):
        return zero
    left_col, right_col = PERSON_PROXY_COLUMNS
    if left_col not in frame.columns or right_col not in frame.columns:
        return zero
    left = pd.to_numeric(frame[left_col], errors="coerce")
    right = pd.to_numeric(frame[right_col], errors="coerce")
    usable = left.ge(PERSON_PROXY_MIN_EXPOSURE) & right.ge(PERSON_PROXY_MIN_EXPOSURE)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = 0.5 * (np.log(left.where(usable)) - np.log(right.where(usable))).abs()
    return spread.fillna(0.0).replace([np.inf, -np.inf], 0.0)


def support_class_series(frame: pd.DataFrame, *, offense: str) -> pd.Series:
    """Which held-out design speaks for this cell. Read off published columns only."""
    benchmark_share = (
        pd.to_numeric(frame[f"benchmark_imputed_share_{offense}"], errors="coerce").fillna(0.0)
        if f"benchmark_imputed_share_{offense}" in frame.columns
        else pd.Series(0.0, index=frame.index, dtype=float)
    )
    direct = (
        frame[f"numerator_support_source_{offense}"].astype("string").eq("direct_city_incident")
        if f"numerator_support_source_{offense}" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    out = pd.Series(SUPPORT_CLASS_MODEL, index=frame.index, dtype="string")
    out.loc[direct.fillna(False)] = SUPPORT_CLASS_DIRECT
    out.loc[benchmark_share.gt(TIER_LOW_BENCHMARK_SHARE_MAX)] = SUPPORT_CLASS_BENCHMARK
    return out


def decision_reliability_tier(
    *,
    bin_probability: pd.Series,
    publishable: pd.Series,
    point_index: pd.Series,
    benchmark_share: pd.Series,
    footprint_conflict: pd.Series,
    unresolved_level: pd.Series,
    out_of_domain: pd.Series,
) -> pd.Series:
    """sec.7's rule, verbatim, on published facts.

    High needs the displayed-bin probability at or above 0.80 AND an adequate denominator AND no
    major footprint or source conflict; Medium is 0.60-0.80; Low is below that, or a benchmark
    share above one half, or an unresolved footprint, or extrapolation outside validation support;
    Withheld is a denominator that is undefined or a placement that is not defensible here.
    """
    index = bin_probability.index
    probability = pd.to_numeric(bin_probability, errors="coerce")
    published = pd.Series(publishable, index=index).fillna(False).astype(bool)
    point = pd.to_numeric(point_index, errors="coerce")
    withheld = ~published | point.isna() | probability.isna()
    forced_low = (
        pd.to_numeric(benchmark_share, errors="coerce").fillna(0.0).gt(TIER_LOW_BENCHMARK_SHARE_MAX)
        | pd.Series(footprint_conflict, index=index).fillna(False).astype(bool)
        | pd.Series(unresolved_level, index=index).fillna(False).astype(bool)
        | pd.Series(out_of_domain, index=index).fillna(False).astype(bool)
    )
    tier = pd.Series(TIER_LOW, index=index, dtype="string")
    tier.loc[~withheld & ~forced_low & probability.ge(TIER_MEDIUM_MIN_BIN_PROBABILITY)] = TIER_MEDIUM
    tier.loc[~withheld & ~forced_low & probability.ge(TIER_HIGH_MIN_BIN_PROBABILITY)] = TIER_HIGH
    tier.loc[withheld] = TIER_WITHHELD
    return tier


def _calibrated_quantiles(
    draws: np.ndarray,
    *,
    levels: dict[str, float],
    calibration: dict[str, CalibrationMap],
    support: np.ndarray,
) -> dict[str, np.ndarray]:
    """Quantiles read at the RECALIBRATED level, per support class."""
    out = {name: np.full(draws.shape[0], np.nan, dtype=float) for name in levels}
    n_draws = draws.shape[1]
    with np.errstate(invalid="ignore"):
        for support_class in np.unique(support):
            rows = np.flatnonzero(support == support_class)
            if not len(rows):
                continue
            mapper = calibration.get(str(support_class), IDENTITY_CALIBRATION)
            # One sort per block rather than one partition per level: identical linear-interpolation
            # quantiles (numpy's default method), an order of magnitude cheaper at release scale.
            ordered = np.sort(draws[rows], axis=1)
            for name, level in levels.items():
                position = mapper.invert(level) * (n_draws - 1)
                lower = int(np.floor(position))
                upper = min(lower + 1, n_draws - 1)
                fraction = position - lower
                out[name][rows] = ordered[:, lower] + fraction * (ordered[:, upper] - ordered[:, lower])
    return out


def _calibrated_probability_below(
    draws: np.ndarray,
    thresholds: np.ndarray,
    *,
    calibration: dict[str, CalibrationMap],
    support: np.ndarray,
) -> np.ndarray:
    """`G(F(t))`: the empirical share of draws below `t`, recalibrated."""
    raw = np.mean(draws <= thresholds[:, None], axis=1)
    out = np.empty(len(raw), dtype=float)
    for support_class in np.unique(support):
        rows = np.flatnonzero(support == support_class)
        if not len(rows):
            continue
        mapper = calibration.get(str(support_class), IDENTITY_CALIBRATION)
        out[rows] = mapper.apply(raw[rows])
    return out


def summarize_offense(
    frame: pd.DataFrame,
    *,
    offense: str,
    count_draws: np.ndarray,
    control_log_sd: np.ndarray,
    share_log_sd: np.ndarray,
    runtime: UncertaintyRuntime,
) -> dict[str, pd.Series]:
    """Every published field for one offense on one surface, from that surface's own draws."""
    index = frame.index
    n_draws = count_draws.shape[1]
    support = support_class_series(frame, offense=offense)
    calibration = {
        support_class: runtime.calibration_map(support_class=support_class, offense=offense)
        for support_class in SUPPORT_CLASSES
    }
    support_values = support.to_numpy(dtype=object)

    denominator = (
        pd.to_numeric(frame[f"primary_denominator_{offense}"], errors="coerce")
        if f"primary_denominator_{offense}" in frame.columns
        else pd.Series(np.nan, index=index, dtype=float)
    )
    national_rate = (
        pd.to_numeric(frame[f"primary_national_rate_per_100k_{offense}"], errors="coerce")
        if f"primary_national_rate_per_100k_{offense}" in frame.columns
        else pd.Series(np.nan, index=index, dtype=float)
    )
    publishable = (
        frame[f"primary_index_publishable_{offense}"].fillna(False).astype(bool)
        if f"primary_index_publishable_{offense}" in frame.columns
        else pd.Series(False, index=index)
    )
    point_index = (
        pd.to_numeric(frame[f"index_{offense}_primary"], errors="coerce")
        if f"index_{offense}_primary" in frame.columns
        else pd.Series(np.nan, index=index, dtype=float)
    )

    log_sd_denominator = denominator_log_sd(frame, offense=offense)
    # Median-preserving lognormal on the denominator, drawn independently of the count: the two
    # surfaces that measure exposure are not the surfaces that place incidents.
    rng = np.random.default_rng([runtime.seed, list(OFFENSES_7).index(str(offense)), 0, 3])
    denominator_noise = rng.standard_normal((len(index), n_draws))
    denominator_draws = denominator.to_numpy(dtype=float)[:, None] * np.exp(
        log_sd_denominator.to_numpy(dtype=float)[:, None] * denominator_noise
    )

    counts = np.asarray(count_draws, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_draws = np.where(
            denominator_draws > 0.0, 1e5 * counts / denominator_draws, np.float32("nan")
        ).astype(np.float32)
        national = national_rate.to_numpy(dtype=float)[:, None]
        index_draws = np.where(
            national > 0.0, 100.0 * rate_draws / national, np.float32("nan")
        ).astype(np.float32)
    unpublishable = ~publishable.to_numpy(dtype=bool)
    rate_draws[unpublishable] = np.nan
    index_draws[unpublishable] = np.nan

    out: dict[str, pd.Series] = {}
    for name, values in _calibrated_quantiles(
        counts,
        levels={q: QUANTILE_LEVELS[q] for q in COUNT_QUANTILES},
        calibration=calibration,
        support=support_values,
    ).items():
        out[count_quantile_column(offense, name)] = pd.Series(values, index=index, dtype=float)
    for name, values in _calibrated_quantiles(
        rate_draws,
        levels={q: QUANTILE_LEVELS[q] for q in RATE_QUANTILES},
        calibration=calibration,
        support=support_values,
    ).items():
        out[rate_quantile_column(offense, name)] = pd.Series(values, index=index, dtype=float)
    index_quantiles = _calibrated_quantiles(
        index_draws,
        levels={q: QUANTILE_LEVELS[q] for q in INDEX_QUANTILES},
        calibration=calibration,
        support=support_values,
    )
    for name, values in index_quantiles.items():
        out[index_quantile_column(offense, name)] = pd.Series(values, index=index, dtype=float)

    reference = np.full(len(index), NATIONAL_REFERENCE_INDEX, dtype=float)
    below_reference = _calibrated_probability_below(
        index_draws, reference, calibration=calibration, support=support_values
    )
    above = np.where(publishable.to_numpy(dtype=bool), 1.0 - below_reference, np.nan)
    out[prob_above_reference_column(offense)] = pd.Series(above, index=index, dtype=float)

    bins = index_bin_indices(point_index.to_numpy(dtype=float))
    breaks = np.asarray(INDEX_BREAKS, dtype=float)
    lower_edge = np.where(bins > 0, breaks[np.clip(bins - 1, 0, len(breaks) - 1)], -np.inf)
    upper_edge = np.where(bins < len(breaks), breaks[np.clip(bins, 0, len(breaks) - 1)], np.inf)
    below_upper = _calibrated_probability_below(
        index_draws, upper_edge, calibration=calibration, support=support_values
    )
    below_lower = _calibrated_probability_below(
        index_draws, lower_edge, calibration=calibration, support=support_values
    )
    bin_probability = np.where(
        publishable.to_numpy(dtype=bool) & (bins >= 0),
        np.clip(below_upper - below_lower, 0.0, 1.0),
        np.nan,
    )
    out[displayed_bin_column(offense)] = pd.Series(
        index_bin_labels(point_index.to_numpy(dtype=float)), index=index, dtype="string"
    ).where(publishable, pd.NA)
    out[prob_displayed_bin_column(offense)] = pd.Series(bin_probability, index=index, dtype=float)

    benchmark_share = (
        pd.to_numeric(frame[f"benchmark_imputed_share_{offense}"], errors="coerce").fillna(0.0)
        if f"benchmark_imputed_share_{offense}" in frame.columns
        else pd.Series(0.0, index=index, dtype=float)
    )
    footprint_conflict = (
        frame[f"footprint_ambient_exposure_missing_{offense}"].astype("boolean").fillna(False).astype(bool)
        if f"footprint_ambient_exposure_missing_{offense}" in frame.columns
        else pd.Series(False, index=index)
    )
    unresolved_level = (
        frame[f"unresolved_level_flag_{offense}"].astype("boolean").fillna(False).astype(bool)
        if f"unresolved_level_flag_{offense}" in frame.columns
        else pd.Series(False, index=index)
    )
    from crimerisk.confidence import DOMAIN_LOW_CUTOFF

    out_of_domain = (
        pd.to_numeric(frame[f"domain_overlap_score_{offense}"], errors="coerce").lt(DOMAIN_LOW_CUTOFF)
        if f"domain_overlap_score_{offense}" in frame.columns
        else pd.Series(False, index=index)
    ).fillna(False)

    out[decision_tier_column(offense)] = decision_reliability_tier(
        bin_probability=pd.Series(bin_probability, index=index, dtype=float),
        publishable=publishable,
        point_index=point_index,
        benchmark_share=benchmark_share,
        footprint_conflict=footprint_conflict,
        unresolved_level=unresolved_level,
        out_of_domain=out_of_domain,
    )
    out[support_class_column(offense)] = support
    out[control_log_sd_column(offense)] = pd.Series(control_log_sd, index=index, dtype=float)
    out[share_log_sd_column(offense)] = pd.Series(share_log_sd, index=index, dtype=float)
    out[denominator_log_sd_column(offense)] = log_sd_denominator.astype(float)
    return out


# --- summary for the build manifest -------------------------------------------------------


def summarize_uncertainty_layer(
    *, runtime: UncertaintyRuntime | None, surfaces: dict[str, pd.DataFrame] | None = None
) -> dict[str, object]:
    if runtime is None:
        return {"enabled": False}
    summary: dict[str, object] = {
        "enabled": True,
        "version": runtime.version,
        "n_draws": int(runtime.n_draws),
        "seed": int(runtime.seed),
        "calibration_version": CALIBRATION_VERSION,
        "share_dispersion_version": SHARE_DISPERSION_VERSION,
        "control_dispersion_version": CONTROL_DISPERSION_VERSION,
        "estimand": "structural expected count for the edition year; the realization layer stays in the ci95 fields",
        "control_dispersion": {k: float(v) for k, v in sorted(runtime.control_dispersion.items())},
        "share_dispersion": {
            f"{offense}|{stratum}": float(value)
            for (offense, stratum), value in sorted(runtime.share_dispersion.items())
        },
        "index_breaks": list(INDEX_BREAKS),
        "index_bin_labels": list(INDEX_BIN_LABELS),
        "tier_thresholds": {
            "high_min_displayed_bin_probability": float(TIER_HIGH_MIN_BIN_PROBABILITY),
            "medium_min_displayed_bin_probability": float(TIER_MEDIUM_MIN_BIN_PROBABILITY),
            "low_max_benchmark_share": float(TIER_LOW_BENCHMARK_SHARE_MAX),
        },
        "config_paths": [str(path) for path in runtime.config_paths],
        "contract": "analysis_scratch/final_phase/UNCERTAINTY_LAYER_CONTRACT.md",
    }
    if surfaces:
        distribution: dict[str, dict[str, dict[str, int]]] = {}
        for label, frame in surfaces.items():
            per_offense: dict[str, dict[str, int]] = {}
            for offense in OFFENSES_7:
                column = decision_tier_column(offense)
                if column not in frame.columns:
                    continue
                counts = frame[column].astype("string").value_counts(dropna=False)
                per_offense[offense] = {str(k): int(v) for k, v in counts.items()}
            distribution[label] = per_offense
        summary["decision_tier_distribution"] = distribution
    return summary


# --- cache --------------------------------------------------------------------------------


def uncertainty_cache_path(paths: RepoPaths, *, year: int, signature: str) -> Path:
    return paths.cache_dir / "uncertainty" / f"uncertainty_cells_{int(year)}_{signature}.parquet"


def components_signature(components: pd.DataFrame, controls: pd.DataFrame) -> str:
    """Content address of the two frames the draws are a function of.

    The components frame is in memory, not on disk, so freshness cannot be a file mtime: it is a
    digest of the values themselves. That is strictly tighter than a dependency-path stamp -- a
    rebuild that produces identical components reuses the cache, and one that does not, cannot.
    """
    digest = hashlib.blake2b(digest_size=16)
    for column in ("bg_id", "jurisdiction_id", "jurisdiction_type", "offense"):
        values = components[column].astype("string").fillna("").to_numpy(dtype=object)
        digest.update(
            hashlib.blake2b("\x1f".join(values).encode(), digest_size=16).digest()
        )
    if "source_state_fips" in components.columns:
        source_state = (
            components["source_state_fips"]
            .astype("string")
            .fillna("")
            .to_numpy(dtype=object)
        )
        digest.update(
            hashlib.blake2b("\x1f".join(source_state).encode(), digest_size=16).digest()
        )
    digest.update(
        np.ascontiguousarray(
            pd.to_numeric(components["component_count"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        ).tobytes()
    )
    fractions = control_mass_fractions(controls)
    for column in ("jurisdiction_id", "offense"):
        values = fractions[column].astype("string").fillna("").to_numpy(dtype=object)
        digest.update(
            hashlib.blake2b("\x1f".join(values).encode(), digest_size=16).digest()
        )
    for column in (
        "measured_fraction",
        "estimated_fraction",
        "benchmark_fraction",
        "bucket_population",
    ):
        digest.update(
            np.ascontiguousarray(fractions[column].to_numpy(dtype=np.float64)).tobytes()
        )
    return digest.hexdigest()


def uncertainty_summary_inputs_signature(
    *,
    surfaces: dict[str, pd.DataFrame],
    geo_columns: dict[str, str],
    component_audit: pd.DataFrame,
    state_calibration_ratios: dict[tuple[str, str], float],
) -> str:
    """Content address every non-draw input used to build cached published summaries.

    ``components_signature`` identifies the Monte Carlo allocation inputs.  The cached artifact
    also stores decisions calculated from the finalized surface and direct-posterior
    concentration, plus calibrated-twin results.  Those inputs can change while the pre-finalize
    component table does not, so they require their own digest.
    """
    digest = hashlib.blake2b(digest_size=16)
    per_offense_prefixes = (
        "expected_count_{offense}",
        "primary_denominator_{offense}",
        "primary_national_rate_per_100k_{offense}",
        "primary_index_publishable_{offense}",
        "index_{offense}_primary",
        "benchmark_imputed_share_{offense}",
        "footprint_ambient_exposure_missing_{offense}",
        "unresolved_level_flag_{offense}",
        "domain_overlap_score_{offense}",
        "numerator_support_source_{offense}",
    )
    for label in sorted(surfaces):
        frame = surfaces[label]
        candidates = [geo_columns[label], *PERSON_PROXY_COLUMNS]
        candidates.extend(
            template.format(offense=offense)
            for offense in OFFENSES_7
            for template in per_offense_prefixes
        )
        columns = list(dict.fromkeys(column for column in candidates if column in frame.columns))
        digest.update(label.encode())
        digest.update("\x1f".join(columns).encode())
        digest.update(
            pd.util.hash_pandas_object(frame[columns], index=False, categorize=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        )

    digest.update("component_audit".encode())
    audit_keys = ["jurisdiction_id", "jurisdiction_type", "offense"]
    required_audit = {
        *audit_keys, "city_incident_posterior_active", "incident_count", "city_posterior_alpha"
    }
    if required_audit.issubset(component_audit.columns):
        active = component_audit["city_incident_posterior_active"].astype("boolean").fillna(False)
        direct = component_audit.loc[active, audit_keys].copy()
        direct["incidents"] = (
            pd.to_numeric(component_audit.loc[active, "incident_count"], errors="coerce")
            .fillna(0.0).clip(lower=0.0)
        )
        direct["alpha"] = (
            pd.to_numeric(component_audit.loc[active, "city_posterior_alpha"], errors="coerce")
            .fillna(0.0).clip(lower=0.0)
        )
        direct = (
            direct.groupby(audit_keys, dropna=False, sort=True)
            .agg(incidents=("incidents", "sum"), alpha=("alpha", "max"))
            .reset_index()
        )
        digest.update(
            pd.util.hash_pandas_object(
                direct, index=False, categorize=True
            ).to_numpy(dtype=np.uint64).tobytes()
        )
    for (state, offense), ratio in sorted(state_calibration_ratios.items()):
        digest.update(f"{state}\x1f{offense}\x1f{float(ratio):.17g}".encode())
    return digest.hexdigest()


def read_cached_cells(path: Path, *, dependency_paths: list[Path]) -> pd.DataFrame | None:
    if not path.exists() or not artifact_is_current(path, dependency_paths):
        return None
    return pd.read_parquet(path)


def write_cached_cells(path: Path, frame: pd.DataFrame, *, dependency_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    write_dependency_stamp(path, dependency_paths)


CACHE_SURFACE_COLUMN = "uncertainty_surface"
CACHE_GEO_COLUMN = "uncertainty_geo_id"


def summaries_to_cache_frame(
    bundle: dict[str, tuple[pd.Series, dict[str, dict[str, pd.Series]]]],
) -> pd.DataFrame:
    """One wide cache frame for every surface, published column names kept verbatim.

    Wide rather than long on purpose: the cached values ARE the published fields, so a reviewer
    reading the cache reads the same column names as the artifact, and restoring it is a merge
    rather than a pivot that could reorder or rename anything.
    """
    frames: list[pd.DataFrame] = []
    for surface, (geo_ids, summaries) in sorted(bundle.items()):
        frame = pd.DataFrame({CACHE_GEO_COLUMN: pd.Series(geo_ids).astype("string").to_numpy()})
        frame[CACHE_SURFACE_COLUMN] = surface
        for offense in sorted(summaries):
            for column, values in summaries[offense].items():
                frame[column] = np.asarray(values)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cache_frame_to_summaries(
    cache: pd.DataFrame, *, surface: str, geo_ids: pd.Series
) -> dict[str, dict[str, pd.Series]]:
    """Restore one surface's summaries from the cache, aligned to the surface's own row order."""
    subset = cache[cache[CACHE_SURFACE_COLUMN].astype("string").eq(surface)]
    keyed = subset.set_index(subset[CACHE_GEO_COLUMN].astype("string"))
    aligned = keyed.reindex(pd.Series(geo_ids).astype("string").to_numpy())
    summaries: dict[str, dict[str, pd.Series]] = {}
    for offense in OFFENSES_7:
        fields: dict[str, pd.Series] = {}
        for column in uncertainty_columns_for_offense(offense):
            if column not in aligned.columns:
                continue
            fields[column] = pd.Series(
                aligned[column].to_numpy(), index=pd.RangeIndex(len(aligned))
            )
        if fields:
            summaries[offense] = fields
    return summaries


def apply_uncertainty_layer(
    frame: pd.DataFrame,
    *,
    summaries: dict[str, dict[str, pd.Series]],
    runtime: UncertaintyRuntime,
) -> pd.DataFrame:
    """Attach the published fields. Nothing existing is overwritten -- asserted, not assumed."""
    out = frame.copy()
    out[UNCERTAINTY_VERSION_COLUMN] = runtime.version
    for offense, fields in summaries.items():
        for column, values in fields.items():
            if column in frame.columns:
                raise ValueError(
                    f"uncertainty layer would overwrite an existing published column: {column!r}. "
                    "This lane adds fields beside the point payload and never redefines one."
                )
            out[column] = values.to_numpy()
    return out
