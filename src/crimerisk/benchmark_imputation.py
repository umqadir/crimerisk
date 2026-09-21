"""Soft benchmark reconciliation for silent-agency territory (level-lane v2).

The mechanism this exists to fix (diagnosed 2026-07-28, docs/STATE.md): jurisdiction
controls are built bottom-up from reporting agencies, so a fully silent agency
contributes nothing to any target and its territory's target is exactly zero -- 150 of
235 state x offense residual cells, 3.4M people of unincorporated county remainder plus
276k people inside municipal jurisdictions whose PD never reports. v19 correctly refused
to fabricate an agency-level anchor for those agencies; this module supplies the missing
mass at the only level where an external benchmark exists, and never as an agency
estimate.

The design (sol-reviewed, adjudicated; analysis_scratch/sol_v20_review.txt section 1):

The CDE state estimate is a noisy aggregate observation, never a hard ceiling. Accepted
local evidence is locked; offense-granular silent-unit priors are reconciled with an
empirically estimated benchmark variance. Positive-exposure silent units are not zeroed
when accepted local mass exceeds CDE. Exact-CDE scaling is diagnostics only.

Historical design notes below describe the superseded v20 hard-residual mechanism and
are retained only for archaeological context.

1. SUPERSEDED accounting identity per state x offense.

       M_so = max(0, CDE_so - sum of locked jurisdiction targets in state s)

   `CDE_so` is the FBI's own published state estimate, which imputes non-reporting
   agencies by population, agency type and geography -- the documented federal
   estimation program. Locked observed totals are never modified. When the locked
   total already exceeds the benchmark the residual is zero and the cell is recorded
   as a benchmark conflict, never as negative imputation. No headroom scavenging, no
   legacy pool-adjustment budgets: identical missing territory gets an identical
   estimand regardless of which bucket happens to hold mass.

2. SHARES FROM A PARTIALLY-POOLED RATE MODEL, not raw exposure. Exposure share alone
   asserts one latent rate for every silent unit in a state. Instead each unit takes
   the Gamma-Poisson posterior mean rate of its (state, lane, urbanicity) cell,
   shrunk toward the state rate and then the national rate with a single pooling
   constant K per offense chosen by masked validation on observed units:

       r_cell = (sum y_cell + K r_state) / (sum E_cell + K)
       T_u    = E_u r_u * scale_so,   scale_so = min(1, M_so / sum_d E_d r_d)

   The scale factor is capped at 1 deliberately. M_so is an UPPER BOUND on missing
   mass -- it also absorbs definitional, vintage and rounding differences against the
   FBI series -- while E_u r_u is the modelled expectation for the territory. Taking
   the smaller of the two keeps the state total at or below the benchmark (invariant)
   AND keeps every imputed unit at or below its own modelled rate, so a state whose
   benchmark headroom dwarfs its silent territory (Louisiana: 6,562 counts of headroom
   over 6,379 silent residents) cannot be forced to publish an absurd rate. The unused
   headroom is reported, not absorbed.

3. THE UNITS PARTITION THE TERRITORY EXACTLY. A silent municipal PD's territory gets
   its mass through its own municipal jurisdiction cell; the county remainder pool
   covers only what is left. That partition is inherited from the block-group
   crosswalk, whose `allocation_share` sums to one per block group across municipal
   jurisdictions and the state nonmunicipal remainder, so the two lanes' exposures are
   disjoint by construction and are asserted to be so.

4. ACTIVE AND PRIMARY BEFORE IMPUTING. A unit is only eligible if it still has an
   identified silent agency that is (a) of a primary policing type, (b) not dead --
   the existing zero-count/zero-months predicate, RELAXED by the FBI's own 2024 agency
   roster, because an agency the FBI still lists is a non-reporter rather than a
   defunct one -- (c) not a consolidated-footprint principal, and (d) not routed
   elsewhere by a local-resolution override. Territory with no identified silent
   primary agency (Connecticut's county remainders, Clark NV and Salt Lake UT under
   consolidated footprints) is left alone: absence of an agency is not evidence of
   missing crime.

5. PROVENANCE. Imputed mass is labelled `benchmarked_nonreporter_imputation` on the
   control row and carried per unit x offense into the published surface's confidence
   metadata. No agency-level estimate row is created anywhere.

Kentucky is excluded from imputation this release: its benchmark gap is implicated in a
consolidated-agency footprint error that needs its own diagnosis, and forcing a positive
residual into geography before that is resolved would bake the error in. Its gap is
emitted in the conflict report instead.

IMPUTATION v2 (E5 pseudo-missingness fixes, default OFF)
--------------------------------------------------------
E5 masked 664 units that DID report, ran this function unmodified on the doctored
inputs, and scored what it predicted for them. Two verdicts, both empirical:

  * the county-remainder lane is defensible as accounting (1.09x aggregate over its
    applied stratum, no size gradient); it is left alone;
  * the municipal lane over-states mass 2.18x in the strata production actually applies
    it to, because a (state, lane, urbanicity) rate cell is SET BY THE LARGE CITIES in
    it and then APPLIED TO SMALL TOWNS. The named failure -- California City CA,
    predicted 20,888 counts against 292 -- is the same defect at its limit: a rate cell
    that collapsed to 486 residents of remaining observed exposure was applied to a
    14,973-resident unit, undetected.

Three fixes, each behind its own default-off flag on `BenchmarkImputationConfig`:

  1. `enable_size_aware_municipal_rates` -- population enters the municipal rate
     structure as a predeclared band (<2.5k / 2.5k-10k / 10k-50k / 50k+), partially
     pooled toward its parent (state, lane, urbanicity) cell with its own Gamma-Poisson
     constant K_band chosen out of sample by the same masked-validation criterion, with
     the parent constant held at its legacy selection. A band with no data of its own
     reproduces the legacy cell rate exactly, so the structure degrades to v1 rather
     than to noise. The county lane has no band level.
  2. `enable_cell_exposure_floor` -- a unit may not take a rate from a level whose
     REMAINING OBSERVED exposure is below max(absolute floor, multiple x unit
     population). On breach the unit escalates band -> cell -> state; if even the state
     level cannot clear the floor the unit is REFUSED (recorded, never imputed) rather
     than served a rate from somewhere it does not belong.
  3. `attach_empirical_bounds` -- E5's per-stratum multiplicative bounds are attached to
     every imputed unit as `bound_lo_80/hi_80/lo_95/hi_95` counts plus the mass-centring
     `central_correction`. The correction is ATTACHED, NOT APPLIED: whether the published
     point estimate moves by it is a promotion-time decision for the owner.

Contract: analysis_scratch/final_phase/IMPUTATION_V2_CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.agency_identity import (
    load_agency_jurisdiction_crosswalk as _load_agency_jurisdiction_crosswalk,
    load_fbi_roster_oris as _fbi_roster_oris,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths
from crimerisk.stage1_adjudications import build_usability_directives
from crimerisk.trend_fills import LANE_GROUNDED_ESTIMATE_SOURCES


BENCHMARK_IMPUTATION_SOURCE = "benchmarked_nonreporter_imputation"
MUNICIPAL_UNIT_KIND = "municipal_jurisdiction"
COUNTY_UNIT_KIND = "county_remainder"
STATE_REMAINDER_TYPE = "state_nonmunicipal_remainder"
STATE_REMAINDER_SUFFIX = ":state_nonmunicipal_remainder"

# Agency types that can be primary for a piece of territory. Special-jurisdiction
# (campus/transit/park) and state-police agencies are real, but they are not the
# primary police force for the county remainder or a municipality, so their silence is
# not evidence that the territory's crime is missing.
PRIMARY_AGENCY_TYPES = frozenset({"sheriff", "local_police", "constable_marshal"})

# Urbanicity from the unit's own residential density (persons per square mile of land),
# the standard census-adjacent cut. Computed from the same block-group crosswalk that
# supplies exposure, so no new input and no county-level smearing.
SQ_METERS_PER_SQ_MILE = 2589988.110336
URBANICITY_URBAN_MIN_DENSITY = 1000.0
URBANICITY_SUBURBAN_MIN_DENSITY = 100.0

FILL_MASS_BASELINE_FILENAME = "agency_fill_mass_baseline.json"
BENCHMARK_IMPUTED_MASS_METRIC = "benchmark_imputed_mass"

CDE_OFFENSE_COLUMNS: dict[str, str] = {
    "murder": "homicide",
    "rape": "rape_revised",
    "robbery": "robbery",
    "aggravated_assault": "aggravated_assault",
    "burglary": "burglary",
    "larceny": "larceny",
    "motor_vehicle_theft": "motor_vehicle_theft",
}

RATE_CELL_COLUMNS = ["state_fips", "lane", "urbanicity"]

# --- imputation v2 (E5) -----------------------------------------------------

POPULATION_BAND_COLUMN = "population_band"
RATE_BAND_COLUMNS = [*RATE_CELL_COLUMNS, POPULATION_BAND_COLUMN]

# Predeclared, not learned. Left-closed cuts on the unit's own exposure population:
# [0, 2.5k) [2.5k, 10k) [10k, 50k) [50k, inf). E5's silent municipal population is
# 220 units under 2.5k, 39 in 2.5k-10k, 8 in 10k-50k, so the first two bands carry
# production's mass and the upper two exist so the cell's large units stop setting
# the small units' rate.
MUNICIPAL_POPULATION_BAND_EDGES: tuple[float, ...] = (2_500.0, 10_000.0, 50_000.0)
MUNICIPAL_POPULATION_BAND_LABELS: tuple[str, ...] = ("<2.5k", "2.5k-10k", "10k-50k", "50k+")

# The floor a rate level's REMAINING OBSERVED exposure must clear before a unit may be
# served from it: max(absolute, multiple x unit population). Justification in the
# contract; both terms are config fields so the choice is auditable, not buried.
CELL_EXPOSURE_FLOOR_ABSOLUTE = 25_000.0
CELL_EXPOSURE_FLOOR_UNIT_MULTIPLE = 5.0

RATE_LEVEL_BAND = "band"
RATE_LEVEL_CELL = "cell"
RATE_LEVEL_STATE_LANE = "state_lane"
RATE_LEVEL_STATE = "state"
RATE_LEVEL_NATIONAL = "national"
RATE_LEVEL_REFUSED = "refused"
ESCALATION_REASON_FLOOR = "cell_exposure_floor"
ESCALATION_REASON_NO_LEVEL = "level_absent"
REFUSAL_STATUS_CELL_EXPOSURE_FLOOR = "refused_cell_exposure_floor"

IMPUTATION_RULE_VERSION_V1 = "v1"
IMPUTATION_RULE_VERSION_V2 = "v2"
EMPIRICAL_BOUNDS_FILENAME = "imputation_empirical_bounds.csv"
IMPUTATION_V2_CONTRACT = "analysis_scratch/final_phase/IMPUTATION_V2_CONTRACT.md"

# Columns the v2 lanes add to the unit table. With every flag off the unit table keeps
# exactly UNIT_COLUMNS, which is the legacy byte-safety guarantee.
RATE_STRUCTURE_UNIT_COLUMNS = [
    POPULATION_BAND_COLUMN,
    "rate_level",
    "rate_escalation_reason",
]
EMPIRICAL_BOUNDS_UNIT_COLUMNS = [
    "bounds_rule_version",
    "bounds_stratum",
    "bounds_basis",
    "bound_lo_80",
    "bound_hi_80",
    "bound_lo_95",
    "bound_hi_95",
    "central_correction",
]
REFUSAL_COLUMNS = [
    "year",
    "state_fips",
    "state_abbr",
    "unit_kind",
    "unit_id",
    "county_geoid",
    "offense",
    "exposure_population",
    "urbanicity",
    POPULATION_BAND_COLUMN,
    "required_cell_exposure",
    "required_parent_exposure",
    "available_parent_exposure",
    "refusal_status",
]

UNIT_COLUMNS = [
    "year",
    "state_fips",
    "state_abbr",
    "unit_kind",
    "unit_id",
    "county_geoid",
    "offense",
    "exposure_population",
    "land_area_sq_mi",
    "urbanicity",
    "pooled_rate",
    "modeled_expected_count",
    "modeled_variance",
    "benchmark_scale",
    "benchmark_weight",
    "imputed_count",
    "imputation_source",
    "silent_agency_count",
    "silent_agency_oris",
]

STATE_IDENTITY_COLUMNS = [
    "year",
    "state_fips",
    "state_abbr",
    "offense",
    "locked_observed_total",
    "fbi_cde_estimated_total",
    "benchmark_residual",
    "modeled_pool",
    "modeled_pool_variance",
    "benchmark_relative_uncertainty",
    "reporting_coverage_fraction",
    "benchmark_standard_error",
    "benchmark_variance",
    "benchmark_weight",
    "benchmark_scale",
    "imputed_total",
    "posterior_total",
    "standardized_disagreement",
    "unused_benchmark_headroom",
    "unfilled_modeled_pool",
    "silent_unit_count",
    "silent_unit_population",
    "conflict_kind",
]


@dataclass(frozen=True)
class BenchmarkImputationConfig:
    year: int = 2024
    # Kentucky is deferred to its own diagnosis this release (see the module docstring).
    excluded_state_abbrs: tuple[str, ...] = ("KY",)
    pooling_constant_grid: tuple[float, ...] = (
        1e2,
        3e2,
        1e3,
        3e3,
        1e4,
        3e4,
        1e5,
        3e5,
        1e6,
    )
    validation_folds: int = 5
    validation_seed: int = 20240728
    enforce_mass_ratchet: bool = False

    # --- imputation v2 (E5), every flag default OFF -------------------------
    enable_size_aware_municipal_rates: bool = False
    enable_cell_exposure_floor: bool = False
    attach_empirical_bounds: bool = False
    municipal_population_band_edges: tuple[float, ...] = MUNICIPAL_POPULATION_BAND_EDGES
    cell_exposure_floor_absolute: float = CELL_EXPOSURE_FLOOR_ABSOLUTE
    cell_exposure_floor_unit_multiple: float = CELL_EXPOSURE_FLOOR_UNIT_MULTIPLE
    # None means "derive from the rate-structure flags": the bounds were measured
    # against a specific rule, so a v2 rate structure must not be quoted against v1's
    # bounds. An explicit string pins the table for a diagnostic run.
    empirical_bounds_rule_version: str | None = None

    @property
    def rate_rule_version(self) -> str:
        """Which measured rule the published rates correspond to."""
        changed = bool(self.enable_size_aware_municipal_rates) or bool(
            self.enable_cell_exposure_floor
        )
        return IMPUTATION_RULE_VERSION_V2 if changed else IMPUTATION_RULE_VERSION_V1

    @property
    def bounds_rule_version(self) -> str:
        return str(self.empirical_bounds_rule_version or self.rate_rule_version)

    @property
    def uses_v2_rate_structure(self) -> bool:
        return bool(self.enable_size_aware_municipal_rates) or bool(
            self.enable_cell_exposure_floor
        )


@dataclass(frozen=True)
class BenchmarkImputation:
    units: pd.DataFrame
    state_identity: pd.DataFrame
    validation: pd.DataFrame
    mass_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, object] = field(default_factory=dict)
    band_validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    refusals: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def imputed_total(self) -> float:
        if self.units.empty:
            return 0.0
        return float(
            pd.to_numeric(self.units["imputed_count"], errors="coerce").fillna(0.0).sum()
        )


# --- artifact paths ---------------------------------------------------------


def benchmark_imputation_units_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "controls" / f"benchmark_imputation_units_{int(year)}.parquet"


def benchmark_imputation_conflicts_path(paths: RepoPaths, *, year: int) -> Path:
    return (
        paths.state_dir / "controls" / f"benchmark_imputation_conflicts_{int(year)}.parquet"
    )


def benchmark_imputation_diagnostics_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "controls" / f"benchmark_imputation_{int(year)}.json"


def empty_benchmark_imputation_units() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIT_COLUMNS)


def load_benchmark_imputation_units(paths: RepoPaths, *, year: int) -> pd.DataFrame:
    path = benchmark_imputation_units_path(paths, year=int(year))
    if not path.exists():
        return empty_benchmark_imputation_units()
    return pd.read_parquet(path)


# --- inputs -----------------------------------------------------------------


def _load_block_group_crosswalk(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    frame = pd.read_parquet(
        path,
        columns=[
            "state_fips",
            "block_group_geoid",
            "jurisdiction_id",
            "jurisdiction_type",
            "aland20",
            "pop20",
            "allocation_share",
        ],
    )
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
    frame["block_group_geoid"] = frame["block_group_geoid"].astype("string").str.zfill(12)
    frame["county_geoid"] = frame["block_group_geoid"].str.slice(0, 5)
    for col in ("aland20", "pop20", "allocation_share"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame


def _load_agency_master(paths: RepoPaths) -> pd.DataFrame:
    frame = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=["ori9", "state_fips", "county_fips", "agency_type_norm"],
    )
    frame["ori9"] = frame["ori9"].astype("string")
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
    frame["county_fips"] = frame["county_fips"].astype("string").str.zfill(3)
    return frame


def _globally_dead_or_absent_oris(paths: RepoPaths, *, all_oris: set[str]) -> set[str]:
    """The existing dead-ORI predicate: zero counts AND zero months in every year, or
    absent from the observations panel entirely (the v17 placeholder-ORI broadening)."""
    obs = pd.read_parquet(
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        columns=["ori9", "count", "months_reported"],
    )
    obs["ori9"] = obs["ori9"].astype("string")
    obs["count"] = pd.to_numeric(obs["count"], errors="coerce").fillna(0.0)
    obs["months_reported"] = pd.to_numeric(obs["months_reported"], errors="coerce").fillna(0.0)
    stats = obs.groupby("ori9", dropna=False).agg(
        total_count=("count", "sum"), max_months=("months_reported", "max")
    )
    dead = set(
        stats[stats["total_count"].le(0.0) & stats["max_months"].le(0.0)].index.astype(str)
    )
    return dead | (set(all_oris) - set(stats.index.astype(str)))


def _fill_covered_oris(agency_estimates: pd.DataFrame) -> set[tuple[str, str]]:
    """ORIs whose target-year mass comes from the agency estimator's fill ladder.

    The partition this enforces: an agency's territory is sized EITHER by the agency
    itself (its own report, or its own recent history through a fill) OR by the state
    benchmark (this module), never by both. Before the fill recency bound the two
    populations overlapped -- a chronically absent agency both received a stale
    own-history fill and looked silent here -- and only the unit-level "already has
    locked mass" test kept the mass from being counted twice. That test is a
    consequence, not the rule; this is the rule.
    """
    if agency_estimates.empty or "agency_estimate_source" not in agency_estimates.columns:
        return set()
    filled = agency_estimates[
        ~agency_estimates["agency_estimate_source"].isin(LANE_GROUNDED_ESTIMATE_SOURCES)
    ]
    return set(zip(filled["ori9"].astype("string"), filled["offense"].astype("string"), strict=True))


def _config_ori_set(paths: RepoPaths, filename: str) -> set[str]:
    path = paths.repo_root / "configs" / filename
    if not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str)
    if "ori" not in frame.columns:
        return set()
    return set(frame["ori"].astype("string").str.strip().str.upper().dropna())


# --- exposure and units -----------------------------------------------------


def _urbanicity(population: pd.Series, land_area_sq_mi: pd.Series) -> pd.Series:
    pop = pd.to_numeric(population, errors="coerce").fillna(0.0)
    area = pd.to_numeric(land_area_sq_mi, errors="coerce").fillna(0.0)
    area_values = area.to_numpy(dtype=float)
    density = np.divide(
        pop.to_numpy(dtype=float),
        area_values,
        out=np.zeros(len(area_values), dtype=float),
        where=area_values > 0.0,
    )
    return pd.Series(
        np.select(
            [density >= URBANICITY_URBAN_MIN_DENSITY, density >= URBANICITY_SUBURBAN_MIN_DENSITY],
            ["urban", "suburban"],
            default="rural",
        ),
        index=pop.index,
        dtype="string",
    )


def build_unit_exposure(bg_crosswalk: pd.DataFrame) -> pd.DataFrame:
    """One row per imputation-eligible territory unit with its exposure.

    Municipal jurisdictions and per-county state-remainder cells partition the block
    groups exactly (the crosswalk's allocation_share sums to 1 per block group across
    the two lanes), which is what makes the municipal/county partition exact.
    """
    municipal = (
        bg_crosswalk[bg_crosswalk["jurisdiction_type"].astype("string").eq("municipal")]
        .groupby(["state_fips", "jurisdiction_id"], dropna=False, as_index=False)
        .agg(exposure_population=("pop20", "sum"), aland=("aland20", "sum"))
        .rename(columns={"jurisdiction_id": "unit_id"})
    )
    municipal["unit_kind"] = MUNICIPAL_UNIT_KIND
    municipal["lane"] = MUNICIPAL_UNIT_KIND
    municipal["county_geoid"] = pd.Series(pd.NA, index=municipal.index, dtype="string")

    county = (
        bg_crosswalk[
            bg_crosswalk["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
        ]
        .groupby(["state_fips", "county_geoid"], dropna=False, as_index=False)
        .agg(exposure_population=("pop20", "sum"), aland=("aland20", "sum"))
    )
    county["unit_kind"] = COUNTY_UNIT_KIND
    county["lane"] = COUNTY_UNIT_KIND
    county["unit_id"] = (
        county["state_fips"].astype("string")
        + STATE_REMAINDER_SUFFIX
        + ":county:"
        + county["county_geoid"].astype("string")
    )

    units = pd.concat([municipal, county], ignore_index=True)
    units["land_area_sq_mi"] = (
        pd.to_numeric(units["aland"], errors="coerce").fillna(0.0) / SQ_METERS_PER_SQ_MILE
    )
    units["urbanicity"] = _urbanicity(units["exposure_population"], units["land_area_sq_mi"])
    return units.drop(columns=["aland"])[
        [
            "state_fips",
            "unit_kind",
            "lane",
            "unit_id",
            "county_geoid",
            "exposure_population",
            "land_area_sq_mi",
            "urbanicity",
        ]
    ]


def _supported_ori_set(
    agency_preferred: pd.DataFrame,
    *,
    paths: RepoPaths,
    target_year: int,
) -> set[tuple[str, str]]:
    """ORIs with a target-year observation the pipeline treats as real reporting.

    Identical predicate to the one the agency estimator uses (usable_as_observed, or a
    true-partial year), including the Stage-1 adjudicated usability directives the estimator
    applies on its own panel copy. A twelve-month report of zero is NOT silent: v19 settled
    that true zeros are data.
    """
    directives = build_usability_directives(paths, target_year=int(target_year))
    if agency_preferred.empty:
        return set()
    usable = agency_preferred.get("usable_as_observed")
    partial = agency_preferred.get("current_row_is_true_partial")
    mask = pd.Series(False, index=agency_preferred.index)
    if usable is not None:
        mask = mask | pd.Series(usable, index=agency_preferred.index).fillna(False).astype(bool)
    if partial is not None:
        mask = mask | pd.Series(partial, index=agency_preferred.index).fillna(False).astype(bool)
    supported_frame = agency_preferred.loc[mask, ["ori9", "offense"]].copy()
    supported = set(zip(supported_frame["ori9"].astype("string").str.upper(), supported_frame["offense"].astype("string"), strict=True))
    reads_missing = set(
        directives.loc[
            directives["directive"].eq("reads_missing"), "ori9"
        ].astype("string").str.upper()
    )
    protected = set(
        agency_preferred.loc[
            agency_preferred.get(
                "level1_admission_status",
                pd.Series("", index=agency_preferred.index),
            ).astype("string").eq("corroborated_structural_zero"),
            "ori9",
        ].astype("string").str.upper()
    )
    return {(ori, offense) for ori, offense in supported if ori not in reads_missing or ori in protected}


def build_silent_agency_ledger(
    *,
    paths: RepoPaths,
    config: BenchmarkImputationConfig,
    agency_preferred: pd.DataFrame,
    agency_estimates: pd.DataFrame | None = None,
    succession_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (ORI, territory unit) with the active-and-primary screens applied."""
    crosswalk = _load_agency_jurisdiction_crosswalk(paths)
    master = _load_agency_master(paths)
    ledger = crosswalk.merge(
        master[["ori9", "county_fips", "agency_type_norm"]], on="ori9", how="left"
    )
    all_oris = set(master["ori9"].astype(str)) | set(ledger["ori9"].astype(str))
    dead = _globally_dead_or_absent_oris(paths, all_oris=all_oris)
    roster = _fbi_roster_oris(paths, year=int(config.year))
    consolidated = _config_ori_set(paths, "consolidated_agency_footprints.csv")
    overridden = _config_ori_set(paths, "local_resolution_overrides.csv")
    supported = _supported_ori_set(
        agency_preferred,
        paths=paths,
        target_year=int(config.year),
    )
    fill_covered = _fill_covered_oris(
        agency_estimates if agency_estimates is not None else pd.DataFrame()
    )
    superseded = (
        set(succession_ledger["superseded_ori9"].astype("string"))
        if succession_ledger is not None and not succession_ledger.empty
        else set()
    )

    ledger = ledger.merge(pd.DataFrame({"offense": list(OFFENSES_7)}), how="cross")
    is_remainder = ledger["jurisdiction_id"].str.endswith(STATE_REMAINDER_SUFFIX, na=False)
    county_geoid = ledger["state_fips"].astype("string") + ledger["county_fips"].astype("string")
    ledger["unit_kind"] = np.where(is_remainder, COUNTY_UNIT_KIND, MUNICIPAL_UNIT_KIND)
    ledger["unit_id"] = np.where(
        is_remainder,
        ledger["state_fips"].astype("string") + STATE_REMAINDER_SUFFIX + ":county:" + county_geoid,
        ledger["jurisdiction_id"].astype("string"),
    )
    ledger["county_geoid"] = np.where(is_remainder, county_geoid, pd.NA)

    pair_keys = list(zip(ledger["ori9"].astype(str), ledger["offense"].astype(str), strict=True))
    ledger["is_supported"] = [key in supported for key in pair_keys]
    ledger["is_fill_covered"] = [key in fill_covered for key in pair_keys]
    # Defunct vs merely non-reporting, in the roster's terms. An agency the FBI still
    # lists is a non-reporter whose territory is genuinely uncovered. One it no longer
    # lists, and for which the agency estimator found neither a current report nor a
    # reference inside its recency bound (`FILL_MAX_REFERENCE_AGE_YEARS`), is defunct:
    # its territory belongs to whoever polices it now, not to an imputation sized on
    # its behalf. That second arm routes the 779 off-roster stale agencies out of both
    # lanes instead of leaving them fillable forever.
    off_roster = ~ledger["ori9"].isin(roster)
    ledger["is_stale_reporter"] = ~ledger["is_supported"] & ~ledger["is_fill_covered"]
    ledger["is_dead"] = off_roster & (ledger["ori9"].isin(dead) | ledger["is_stale_reporter"])
    ledger["is_nonreporting_but_rostered"] = ledger["ori9"].isin(dead) & ledger["ori9"].isin(roster)
    ledger["is_consolidated"] = ledger["ori9"].isin(consolidated)
    ledger["is_override_routed"] = ledger["ori9"].isin(overridden)
    ledger["is_primary_type"] = ledger["agency_type_norm"].astype("string").isin(PRIMARY_AGENCY_TYPES)
    ledger["is_superseded"] = ledger["ori9"].isin(superseded)
    ledger["is_eligible_silent"] = (
        ~ledger["is_supported"]
        & ~ledger["is_dead"]
        & ~ledger["is_fill_covered"]
        & ~ledger["is_superseded"]
        & ~ledger["is_consolidated"]
        & ~ledger["is_override_routed"]
        & ledger["is_primary_type"]
    )
    _assert_agency_mass_flows_through_one_lane(ledger)
    return ledger


def _assert_agency_mass_flows_through_one_lane(ledger: pd.DataFrame) -> None:
    """Fail closed if an ORI is sized by the agency estimator AND by the benchmark.

    The three lanes an agency's territory can be sized through -- its own report, its
    own recent history through a fill, the state benchmark through this module -- are
    mutually exclusive by construction, and this says so out loud so that a change to
    either side's predicate cannot silently open an overlap.
    """
    overlapping = ledger[
        ledger["is_eligible_silent"] & (ledger["is_supported"] | ledger["is_fill_covered"])
    ]
    if overlapping.empty:
        return
    raise ValueError(
        f"{overlapping[['ori9', 'offense']].drop_duplicates().shape[0]} agency-offense cell(s) are both sized by the agency "
        "estimator and treated as silent territory for benchmark imputation: "
        + str(sorted(overlapping["ori9"].astype(str).unique())[:20])
    )


def _assert_every_silent_unit_lands_on_exactly_one_empty_control_row(
    silent: pd.DataFrame,
) -> None:
    """Fail closed on the two properties the Jackson MS defect broke.

    (1) **Exactly one landing row.** A silent unit whose control row does not exist would
    lose its whole sub-target at the landing merge. Under the ownership skeleton the
    municipal units ARE the block-group crosswalk's municipal jurisdictions and every one
    of them has a control row by construction, so a failure here means the skeleton and
    the exposure crosswalk have drifted apart.

    (2) **Zero pre-imputation locked mass.** This is the property that used to be the
    eligibility RULE (`locked_total <= 1e-9`) and is now a post-condition of it. It held
    only by accident before: the municipal ladder filled Jackson MS from a 2019 reference
    the agency estimator had already refused as stale, the fill locked the control, and
    the unit was excluded here -- the pipeline's own definition of silence disagreeing
    with itself across a stage boundary. With targets aggregated from agency estimates a
    unit whose every agency is silent can only be empty, and if it ever is not, that is a
    fourth sizing lane re-opening and the build must stop.
    """
    if silent.empty:
        return
    duplicate_key = ["unit_id", "offense"] if "offense" in silent.columns else ["unit_id"]
    duplicated = int(silent.duplicated(subset=duplicate_key).sum())
    if duplicated:
        raise ValueError(
            f"{duplicated} eligible silent unit(s) appear more than once in the unit "
            "table, so their imputed mass would land on more than one control row"
        )
    missing = silent[~silent["has_control_row"].astype(bool)]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} eligible silent unit(s) have no control row to land on: "
            + str(
                missing[["unit_id", "unit_kind", "state_fips", "exposure_population"]]
                .head(20)
                .to_dict(orient="records")
            )
        )
    locked_column = "locked_count" if "locked_count" in silent.columns else "locked_total"
    locked = pd.to_numeric(silent[locked_column], errors="coerce").fillna(0.0)
    occupied = silent[locked.gt(1e-9)]
    if not occupied.empty:
        raise ValueError(
            f"{len(occupied)} eligible silent unit(s) already carry pre-imputation "
            "locked mass; a jurisdiction-level fill has locked a control before "
            "benchmark eligibility was evaluated: "
            + str(
                occupied[[c for c in ["unit_id", "unit_kind", "state_fips", "offense", locked_column] if c in occupied.columns]]
                .head(20)
                .to_dict(orient="records")
            )
        )


def _locked_unit_targets(controls: pd.DataFrame) -> pd.DataFrame:
    """Locked target per (unit, offense), for both lanes.

    Municipal units read their jurisdiction control directly. The county lane has no
    per-county control row -- the state remainder pool is a single jurisdiction -- so
    county exposure is matched against the pool later; here only the municipal side is
    keyed by unit.
    """
    municipal = controls[controls["jurisdiction_type"].astype("string").eq("municipal")][
        ["jurisdiction_id", "state_fips", "offense", "adjusted_count_ags_core"]
    ].copy()
    municipal["unit_id"] = municipal["jurisdiction_id"].astype("string")
    municipal["state_fips"] = municipal["state_fips"].astype("string").str.zfill(2)
    municipal["locked_count"] = (
        pd.to_numeric(municipal["adjusted_count_ags_core"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    return municipal[["unit_id", "state_fips", "offense", "locked_count"]]


def _county_lane_locked_counts(
    *,
    agency_estimates: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Mass currently attributed to each county remainder cell.

    This is the same quantity allocation._build_county_remainder_group_targets splits
    the state remainder pool by: per-agency target-year estimate times crosswalk
    weight, grouped by the agency's county. Used as the county lane's observed `y` in
    the rate model and as the "is this cell already covered" test.
    """
    if agency_estimates.empty:
        return pd.DataFrame(columns=["unit_id", "state_fips", "offense", "locked_count"])
    remainder = ledger[ledger["unit_kind"].eq(COUNTY_UNIT_KIND)][
        ["ori9", "state_fips", "unit_id", "weight", "offense"]
    ]
    est = agency_estimates.copy()
    est["ori9"] = est["ori9"].astype("string")
    est["state_fips"] = est["state_fips"].astype("string").str.zfill(2)
    est["offense"] = est["offense"].astype("string")
    merged = est.merge(remainder, on=["ori9", "state_fips", "offense"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["unit_id", "state_fips", "offense", "locked_count"])
    merged["locked_count"] = (
        pd.to_numeric(merged["estimated_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        * merged["weight"]
    )
    return (
        merged.groupby(["unit_id", "state_fips", "offense"], dropna=False, as_index=False)[
            "locked_count"
        ]
        .sum()
    )


# --- pooled rate model ------------------------------------------------------


def population_band(
    population: pd.Series, *, edges: tuple[float, ...] = MUNICIPAL_POPULATION_BAND_EDGES
) -> pd.Series:
    """Predeclared left-closed population bands: [0, e0) [e0, e1) ... [e_last, inf).

    Labels are derived from the edges rather than looked up, so a config that moves an
    edge cannot silently keep a label that no longer describes the cut.
    """
    values = pd.to_numeric(population, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cuts = [float(edge) for edge in edges]
    if list(cuts) == list(MUNICIPAL_POPULATION_BAND_EDGES):
        labels = list(MUNICIPAL_POPULATION_BAND_LABELS)
    else:
        labels = [f"<{cuts[0]:g}"]
        labels += [f"{lo:g}-{hi:g}" for lo, hi in zip(cuts[:-1], cuts[1:], strict=True)]
        labels += [f"{cuts[-1]:g}+"]
    index = np.searchsorted(np.asarray(cuts, dtype=float), values, side="right")
    return pd.Series(
        [labels[int(i)] for i in index], index=population.index, dtype="string"
    )


def _fit_rate_levels(
    panel: pd.DataFrame,
    *,
    offense: str,
    pooling_constant: float,
    band_pooling_constant: float | None = None,
    band_edges: tuple[float, ...] = MUNICIPAL_POPULATION_BAND_EDGES,
) -> dict:
    """The full rate ladder plus the exposure behind every level.

    `fit_pooled_rates` is this function with the extra levels stripped, so the legacy
    national/state/cell arithmetic has exactly one implementation and cannot drift from
    the v2 path.
    """
    K = float(pooling_constant)
    exposure = pd.to_numeric(panel["exposure_population"], errors="coerce").fillna(0.0)
    counts = pd.to_numeric(panel[offense], errors="coerce").fillna(0.0)
    total_exposure = float(exposure.sum())
    national = float(counts.sum() / total_exposure) if total_exposure > 0 else 0.0

    work = panel[RATE_CELL_COLUMNS].copy()
    work["_y"] = counts.to_numpy(dtype=float)
    work["_E"] = exposure.to_numpy(dtype=float)
    state = work.groupby("state_fips", dropna=False).agg(y=("_y", "sum"), E=("_E", "sum"))
    state["rate"] = (state["y"] + K * national) / (state["E"] + K)
    cell = work.groupby(RATE_CELL_COLUMNS, dropna=False).agg(y=("_y", "sum"), E=("_E", "sum")).reset_index()
    cell = cell.merge(
        state["rate"].rename("state_rate"), left_on="state_fips", right_index=True, how="left"
    )
    cell["state_rate"] = pd.to_numeric(cell["state_rate"], errors="coerce").fillna(national)
    cell["rate"] = (cell["y"] + K * cell["state_rate"]) / (cell["E"] + K)
    cell_keys = list(cell[RATE_CELL_COLUMNS].itertuples(index=False, name=None))
    cell_keys = [tuple(str(part) for part in key) for key in cell_keys]

    # (state, lane): the escalation target the cell-exposure floor climbs to. It is a
    # SIBLING of the cell, shrunk from the state rate with the same K -- the fitted
    # parent chain (national -> state -> cell) is untouched, so v1's published rates are
    # unchanged wherever the floor does not fire. It exists because escalating a rural
    # county remainder to the all-lane state rate would price it off municipal crime,
    # which is the very substitution this item is fixing in the other direction.
    lane = work.groupby(["state_fips", "lane"], dropna=False).agg(y=("_y", "sum"), E=("_E", "sum")).reset_index()
    lane_parent = np.array(
        [
            float(state["rate"].get(str(key), national))
            for key in lane["state_fips"].astype("string")
        ],
        dtype=float,
    )
    lane["rate"] = (lane["y"].to_numpy(dtype=float) + K * lane_parent) / (
        lane["E"].to_numpy(dtype=float) + K
    )
    lane_keys = [
        tuple(str(part) for part in key)
        for key in lane[["state_fips", "lane"]].itertuples(index=False, name=None)
    ]

    rates: dict = {
        "national": national,
        "state": {str(k): float(v) for k, v in state["rate"].items()},
        "cell": dict(zip(cell_keys, (float(v) for v in cell["rate"]), strict=True)),
        "state_lane": dict(zip(lane_keys, (float(v) for v in lane["rate"]), strict=True)),
        "national_exposure": total_exposure,
        "state_exposure": {str(k): float(v) for k, v in state["E"].items()},
        "cell_exposure": dict(zip(cell_keys, (float(v) for v in cell["E"]), strict=True)),
        "state_lane_exposure": dict(zip(lane_keys, (float(v) for v in lane["E"]), strict=True)),
        "pooling_constant": K,
        "band_pooling_constant": None,
    }
    if band_pooling_constant is None:
        rates["band"] = {}
        rates["band_exposure"] = {}
        return rates

    # The band level exists for the municipal lane only: E5 found the county-remainder
    # lane unbiased with no size gradient, so it is not restructured.
    K_band = float(band_pooling_constant)
    banded = work[work["lane"].astype("string").eq(MUNICIPAL_UNIT_KIND)].copy()
    if banded.empty:
        rates["band"] = {}
        rates["band_exposure"] = {}
        rates["band_pooling_constant"] = K_band
        return rates
    banded[POPULATION_BAND_COLUMN] = population_band(banded["_E"], edges=band_edges)
    band = (
        banded.groupby(RATE_BAND_COLUMNS, dropna=False)
        .agg(y=("_y", "sum"), E=("_E", "sum"))
        .reset_index()
    )
    parent_keys = [
        tuple(str(part) for part in key)
        for key in band[RATE_CELL_COLUMNS].itertuples(index=False, name=None)
    ]
    parent_rate = np.array(
        [rates["cell"].get(key, rates["state"].get(key[0], national)) for key in parent_keys],
        dtype=float,
    )
    band["rate"] = (band["y"].to_numpy(dtype=float) + K_band * parent_rate) / (
        band["E"].to_numpy(dtype=float) + K_band
    )
    band_keys = [
        tuple(str(part) for part in key)
        for key in band[RATE_BAND_COLUMNS].itertuples(index=False, name=None)
    ]
    rates["band"] = dict(zip(band_keys, (float(v) for v in band["rate"]), strict=True))
    rates["band_exposure"] = dict(zip(band_keys, (float(v) for v in band["E"]), strict=True))
    rates["band_pooling_constant"] = K_band
    return rates


def fit_pooled_rates(panel: pd.DataFrame, *, offense: str, pooling_constant: float) -> dict:
    """Two-level Gamma-Poisson posterior means: cell shrunk to state shrunk to national.

    K is a pseudo-exposure: a cell with K person-years of its own data weighs its own
    rate and its parent's equally. One K per offense, chosen out of sample.
    """
    full = _fit_rate_levels(panel, offense=offense, pooling_constant=pooling_constant)
    return {"national": full["national"], "state": full["state"], "cell": full["cell"]}


def predict_pooled_rate(frame: pd.DataFrame, rates: dict) -> np.ndarray:
    keys = [
        tuple(str(part) for part in key)
        for key in frame[RATE_CELL_COLUMNS].itertuples(index=False, name=None)
    ]
    cell = rates["cell"]
    state = rates["state"]
    national = float(rates["national"])
    return np.array(
        [cell.get(key, state.get(key[0], national)) for key in keys],
        dtype=float,
    )


def predict_pooled_rate_v2(
    frame: pd.DataFrame,
    rates: dict,
    *,
    floor_absolute: float | None = None,
    floor_unit_multiple: float | None = None,
) -> pd.DataFrame:
    """Serve each unit from the finest rate level whose own support clears the floor.

    With the floor ON the ladder is band -> cell -> (state, lane).

    The floor is a LEVEL-SELECTION rule, and its two terms do different jobs. To be
    served from the band or the cell, a level's REMAINING OBSERVED exposure must clear
    `max(floor_absolute, floor_unit_multiple x unit population)` -- the relative term
    says "do not price a unit off peers its own size or smaller when a better-supported
    level exists", which is E5's failure case exactly (486 residents pricing 14,973).
    At the TERMINAL level, `(state, lane)`, there is no coarser in-lane level to prefer,
    so the comparative term has nothing to say and only `floor_absolute` applies. That
    asymmetry is deliberate: applying the relative term at the terminal would refuse the
    LARGEST units -- on E5's masked sample it refused New York City, Phoenix and the
    Harris County TX remainder, whose own populations exceed a fifth of their entire
    lane -- which inverts the fix. A big-city PD going dark is precisely what this
    imputation exists for; refusing to price it would be the worst available failure.

    Escalation stops at (state, lane) in BOTH directions. It never falls through to the
    all-lane state rate, because that would price a rural county remainder off municipal
    crime; and it never falls through to the national rate, because that would price a
    unit off other states' crime while claiming the state benchmark's authority. A unit
    whose own state x lane holds less than `floor_absolute` observed residents is
    REFUSED (`rate_level == "refused"`), recorded, and never imputed.

    With the floor OFF the ladder is the legacy fall-through -- cell -> state ->
    national -- with the band level in front of it, so enabling size-aware rates alone
    changes the rate structure and nothing about the fallback.
    """
    has_bands = bool(rates.get("band"))
    band_keys: list[tuple[str, ...]] | None = None
    if has_bands and POPULATION_BAND_COLUMN in frame.columns:
        band_keys = [
            tuple(str(part) for part in key)
            for key in frame[RATE_BAND_COLUMNS].itertuples(index=False, name=None)
        ]
    cell_keys = [
        tuple(str(part) for part in key)
        for key in frame[RATE_CELL_COLUMNS].itertuples(index=False, name=None)
    ]
    unit_exposure = (
        pd.to_numeric(frame["exposure_population"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if floor_absolute is None:
        required = np.zeros(len(frame), dtype=float)
        terminal_required = 0.0
        floor_on = False
    else:
        required = np.maximum(
            float(floor_absolute), float(floor_unit_multiple or 0.0) * unit_exposure
        )
        terminal_required = float(floor_absolute)
        floor_on = True

    lane_keys = [
        (str(state), str(lane))
        for state, lane in zip(frame["state_fips"], frame["lane"], strict=True)
    ]
    national = float(rates["national"])
    band_rate, band_exposure = rates.get("band", {}), rates.get("band_exposure", {})
    cell_rate, cell_exposure = rates["cell"], rates.get("cell_exposure", {})
    lane_rate = rates.get("state_lane", {})
    lane_exposure = rates.get("state_lane_exposure", {})
    state_rate = rates["state"]

    out_rate = np.zeros(len(frame), dtype=float)
    out_level = np.empty(len(frame), dtype=object)
    out_reason = np.empty(len(frame), dtype=object)
    out_parent_exposure = np.zeros(len(frame), dtype=float)

    for i in range(len(frame)):
        need = float(required[i])
        reason = ""
        cell_key = cell_keys[i]
        lane_key = lane_keys[i]
        out_parent_exposure[i] = float(lane_exposure.get(lane_key, 0.0))
        if band_keys is not None:
            band_key = band_keys[i]
            if band_key in band_rate:
                if not floor_on or float(band_exposure.get(band_key, 0.0)) >= need:
                    out_rate[i] = band_rate[band_key]
                    out_level[i] = RATE_LEVEL_BAND
                    out_reason[i] = ""
                    continue
                reason = ESCALATION_REASON_FLOOR
        if cell_key in cell_rate:
            if not floor_on or float(cell_exposure.get(cell_key, 0.0)) >= need:
                out_rate[i] = cell_rate[cell_key]
                out_level[i] = RATE_LEVEL_CELL
                out_reason[i] = reason
                continue
            reason = ESCALATION_REASON_FLOOR
        elif not reason:
            reason = ESCALATION_REASON_NO_LEVEL
        if floor_on:
            # In-lane parent, then refusal. No cross-lane and no cross-state fallback.
            # Only the absolute term applies here: see the docstring on why the
            # relative term is a level-selection criterion and stops at the cell.
            if lane_key in lane_rate and float(lane_exposure.get(lane_key, 0.0)) >= terminal_required:
                out_rate[i] = lane_rate[lane_key]
                out_level[i] = RATE_LEVEL_STATE_LANE
                out_reason[i] = reason
                continue
            out_rate[i] = np.nan
            out_level[i] = RATE_LEVEL_REFUSED
            out_reason[i] = reason or ESCALATION_REASON_FLOOR
            continue
        state_key = cell_key[0]
        if state_key in state_rate:
            out_rate[i] = state_rate[state_key]
            out_level[i] = RATE_LEVEL_STATE
            out_reason[i] = reason
            continue
        out_rate[i] = national
        out_level[i] = RATE_LEVEL_NATIONAL
        out_reason[i] = reason or ESCALATION_REASON_NO_LEVEL

    return pd.DataFrame(
        {
            "pooled_rate": out_rate,
            "rate_level": pd.array(out_level, dtype="string"),
            "rate_escalation_reason": pd.array(out_reason, dtype="string"),
            "required_cell_exposure": required,
            "required_parent_exposure": np.full(len(frame), terminal_required, dtype=float),
            "available_parent_exposure": out_parent_exposure,
        },
        index=frame.index,
    )


def _poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    mu = np.clip(expected, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(observed > 0.0, observed * np.log(observed / mu), 0.0)
    return float(2.0 * np.sum(term - (observed - mu)))


def _frozen_validation_folds(
    observed: pd.DataFrame, *, config: BenchmarkImputationConfig
) -> np.ndarray:
    """The fold assignment both K-selection passes share, so the band constant is chosen
    against the same out-of-sample split the parent constant was."""
    return (
        pd.util.hash_pandas_object(
            observed[["state_fips", "unit_id", "offense"]].astype("string"),
            index=False,
            hash_key=f"{int(config.validation_seed):016d}"[-16:],
        ).to_numpy(dtype=np.uint64)
        % int(config.validation_folds)
    )


def select_pooling_constants(
    observed: pd.DataFrame, *, config: BenchmarkImputationConfig
) -> tuple[dict[str, float], pd.DataFrame]:
    """Masked validation: hide a fold of the observed units, predict them from the model
    fitted on the rest, keep the K with the lowest out-of-sample Poisson deviance."""
    folds = int(config.validation_folds)
    observed = observed.assign(_frozen_fold=_frozen_validation_folds(observed, config=config))
    rows: list[dict[str, float]] = []
    for pooling_constant in config.pooling_constant_grid:
        record: dict[str, float] = {"pooling_constant": float(pooling_constant)}
        for offense in OFFENSES_7:
            record[offense] = 0.0
            offense_work = observed[observed["offense"].eq(offense)].copy()
            offense_work["_fold"] = offense_work["_frozen_fold"]
            for fold in range(folds):
                train = offense_work[offense_work["_fold"].ne(fold)]
                test = offense_work[offense_work["_fold"].eq(fold)]
                if train.empty or test.empty:
                    continue
                exposure = pd.to_numeric(test["exposure_population"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                rates = fit_pooled_rates(train, offense=offense, pooling_constant=pooling_constant)
                expected = predict_pooled_rate(test, rates) * exposure
                record[offense] += _poisson_deviance(
                    pd.to_numeric(test[offense], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                    expected,
                )
        record["total"] = float(sum(record[offense] for offense in OFFENSES_7))
        rows.append(record)
    validation = pd.DataFrame(rows)
    chosen = {
        offense: float(validation.loc[validation[offense].idxmin(), "pooling_constant"])
        for offense in OFFENSES_7
    }
    return chosen, validation


def select_band_pooling_constants(
    observed: pd.DataFrame,
    *,
    config: BenchmarkImputationConfig,
    pooling_constants: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Choose K_band per offense with the parent constant held at its legacy selection.

    A profile (coordinate) search, not a joint one, and deliberately so: holding
    K_parent fixed at the value `select_pooling_constants` already chose leaves the
    national/state/cell ladder -- and therefore the whole county-remainder lane -- bit
    for bit what it is today, so the band level is the only thing this pass can move.
    Criterion, folds and grid are the parent pass's.
    """
    folds = int(config.validation_folds)
    observed = observed.assign(_frozen_fold=_frozen_validation_folds(observed, config=config))
    observed = observed.assign(
        **{
            POPULATION_BAND_COLUMN: population_band(
                observed["exposure_population"], edges=tuple(config.municipal_population_band_edges)
            )
        }
    )
    floor_absolute = (
        float(config.cell_exposure_floor_absolute) if config.enable_cell_exposure_floor else None
    )
    rows: list[dict[str, float]] = []
    for band_constant in config.pooling_constant_grid:
        record: dict[str, float] = {"band_pooling_constant": float(band_constant)}
        for offense in OFFENSES_7:
            record[offense] = 0.0
            offense_work = observed[observed["offense"].eq(offense)]
            for fold in range(folds):
                train = offense_work[offense_work["_frozen_fold"].ne(fold)]
                test = offense_work[offense_work["_frozen_fold"].eq(fold)]
                if train.empty or test.empty:
                    continue
                rates = _fit_rate_levels(
                    train,
                    offense=offense,
                    pooling_constant=float(pooling_constants[offense]),
                    band_pooling_constant=float(band_constant),
                    band_edges=tuple(config.municipal_population_band_edges),
                )
                served = predict_pooled_rate_v2(
                    test,
                    rates,
                    floor_absolute=floor_absolute,
                    floor_unit_multiple=float(config.cell_exposure_floor_unit_multiple),
                )
                # Refused rows carry no prediction at any K_band, so dropping them
                # compares the grid on an identical row set.
                scored = served["pooled_rate"].notna().to_numpy()
                if not scored.any():
                    continue
                exposure = (
                    pd.to_numeric(test["exposure_population"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)[scored]
                )
                expected = served["pooled_rate"].to_numpy(dtype=float)[scored] * exposure
                record[offense] += _poisson_deviance(
                    pd.to_numeric(test[offense], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)[scored],
                    expected,
                )
        record["total"] = float(sum(record[offense] for offense in OFFENSES_7))
        rows.append(record)
    validation = pd.DataFrame(rows)
    chosen = {
        offense: float(
            validation.loc[validation[offense].idxmin(), "band_pooling_constant"]
        )
        for offense in OFFENSES_7
    }
    return chosen, validation


# --- empirical bounds (E5) --------------------------------------------------


EMPIRICAL_BOUNDS_COLUMNS = [
    "rule_version",
    "stratum",
    "lane",
    "pop_min",
    "pop_max",
    "offense",
    "n_units",
    "n",
    "agg_ratio",
    "median_ratio",
    "mult_lo80",
    "mult_hi80",
    "mult_lo95",
    "mult_hi95",
    "severe_under",
    "severe_over",
    "source",
]
BOUNDS_ALL_OFFENSE_KEY = "ALL"
BOUNDS_BASIS_IN_SUPPORT = "in_support"
BOUNDS_BASIS_EXTRAPOLATED = "extrapolated"


def empirical_bounds_path(paths: RepoPaths) -> Path:
    return paths.repo_root / "configs" / EMPIRICAL_BOUNDS_FILENAME


def load_empirical_bounds(paths: RepoPaths, *, rule_version: str) -> pd.DataFrame:
    """The measured per-stratum multiplicative bounds for one rule version.

    Refuses anything it cannot use: a missing table, a missing version, a stratum that
    is not exactly one lane, a bound ladder that is not ordered, or a non-positive
    multiplier. Bounds that silently degrade are worse than no bounds.
    """
    path = empirical_bounds_path(paths)
    if not path.exists():
        raise FileNotFoundError(
            f"empirical imputation bounds table missing at {path}; it is a committed "
            "config, not a build product"
        )
    frame = pd.read_csv(path)
    missing = [c for c in EMPIRICAL_BOUNDS_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    frame["rule_version"] = frame["rule_version"].astype("string").str.strip()
    selected = frame[frame["rule_version"].eq(str(rule_version))].copy()
    if selected.empty:
        raise ValueError(
            f"{path} carries no rows for rule_version={rule_version!r}; available: "
            + str(sorted(frame["rule_version"].dropna().unique()))
        )
    for column in ("stratum", "lane", "offense", "source"):
        selected[column] = selected[column].astype("string").str.strip()
    numeric = ["pop_min", "pop_max", "agg_ratio", "median_ratio", "mult_lo80", "mult_hi80", "mult_lo95", "mult_hi95"]
    for column in numeric:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected[numeric].isna().to_numpy().any():
        raise ValueError(f"{path} has non-numeric or missing bound values for {rule_version}")
    lanes_per_stratum = selected.groupby("stratum")["lane"].nunique()
    if int((lanes_per_stratum > 1).sum()):
        raise ValueError(
            f"{path}: stratum(s) span more than one lane: "
            + str(sorted(lanes_per_stratum[lanes_per_stratum > 1].index))
        )
    ordered = (
        selected["mult_lo95"].le(selected["mult_lo80"])
        & selected["mult_lo80"].le(selected["mult_hi80"])
        & selected["mult_hi80"].le(selected["mult_hi95"])
    )
    if not bool(ordered.all()):
        raise ValueError(
            f"{path}: bound ladder is not ordered lo95 <= lo80 <= hi80 <= hi95 for "
            + str(selected.loc[~ordered, ["stratum", "offense"]].to_dict(orient="records")[:10])
        )
    if not bool(selected[["mult_lo95", "agg_ratio"]].gt(0.0).to_numpy().all()):
        raise ValueError(f"{path}: non-positive multiplier or aggregate ratio for {rule_version}")
    duplicated = selected.duplicated(subset=["stratum", "offense"]).sum()
    if int(duplicated):
        raise ValueError(f"{path}: {int(duplicated)} duplicate (stratum, offense) row(s) for {rule_version}")
    for lane in selected["lane"].unique():
        lane_rows = selected[selected["lane"].eq(lane)]
        if BOUNDS_ALL_OFFENSE_KEY not in set(lane_rows["offense"]):
            raise ValueError(f"{path}: lane {lane!r} has no {BOUNDS_ALL_OFFENSE_KEY} fallback row")
    selected["central_correction"] = 1.0 / selected["agg_ratio"]
    return selected.reset_index(drop=True)


def attach_empirical_bounds(units: pd.DataFrame, *, bounds: pd.DataFrame) -> pd.DataFrame:
    """Attach the measured uncertainty envelope to every imputed unit.

    Multiplicative on the published point estimate: E5 scored `actual / predicted` on
    its masked sample, so `bound_lo_80 = imputed_count x mult_lo80` is the count the
    10th-percentile masked unit turned out to have. `central_correction` is the
    mass-centring multiplier `1 / aggregate ratio` -- ATTACHED, NEVER APPLIED here. A
    unit whose population falls outside the stratum's validated range keeps the bounds
    but is labelled `extrapolated`, because pretending the range is wider than it was
    measured on is the failure mode this whole item exists to stop.
    """
    out = units.copy()
    if out.empty:
        for column in EMPIRICAL_BOUNDS_UNIT_COLUMNS:
            if column not in out.columns:
                out[column] = pd.Series(dtype="float64" if column.startswith(("bound_", "central_")) else "string")
        return out
    lane = np.where(
        out["unit_kind"].astype("string").eq(COUNTY_UNIT_KIND), COUNTY_UNIT_KIND, MUNICIPAL_UNIT_KIND
    )
    exposure = pd.to_numeric(out["exposure_population"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    offense = out["offense"].astype("string").to_numpy()
    by_key = {
        (str(row.lane), str(row.offense)): row for row in bounds.itertuples(index=False)
    }
    missing_lanes = sorted({str(value) for value in lane} - {key[0] for key in by_key})
    if missing_lanes:
        raise ValueError(
            f"empirical bounds table has no rows for imputation lane(s) {missing_lanes}"
        )
    fields = ["mult_lo80", "mult_hi80", "mult_lo95", "mult_hi95"]
    picked = [
        by_key.get((str(lane[i]), str(offense[i])), by_key[(str(lane[i]), BOUNDS_ALL_OFFENSE_KEY)])
        for i in range(len(out))
    ]
    imputed = pd.to_numeric(out["imputed_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["bounds_rule_version"] = pd.array([str(row.rule_version) for row in picked], dtype="string")
    out["bounds_stratum"] = pd.array([str(row.stratum) for row in picked], dtype="string")
    out["bounds_basis"] = pd.array(
        [
            BOUNDS_BASIS_IN_SUPPORT
            if float(row.pop_min) <= exposure[i] < float(row.pop_max)
            else BOUNDS_BASIS_EXTRAPOLATED
            for i, row in enumerate(picked)
        ],
        dtype="string",
    )
    for field_name, column in zip(fields, ("bound_lo_80", "bound_hi_80", "bound_lo_95", "bound_hi_95"), strict=True):
        out[column] = imputed * np.array([float(getattr(row, field_name)) for row in picked], dtype=float)
    out["central_correction"] = np.array(
        [float(row.central_correction) for row in picked], dtype=float
    )
    return out


# --- the build --------------------------------------------------------------


def _cde_state_benchmark_long(cde_estimates: pd.DataFrame) -> pd.DataFrame:
    value_columns = [c for c in CDE_OFFENSE_COLUMNS.values() if c in cde_estimates.columns]
    long = cde_estimates.melt(
        id_vars=["state_abbr"],
        value_vars=value_columns,
        var_name="_cde_offense",
        value_name="fbi_cde_estimated_total",
    )
    reverse = {v: k for k, v in CDE_OFFENSE_COLUMNS.items()}
    long["offense"] = long["_cde_offense"].map(reverse).astype("string")
    long["state_abbr"] = long["state_abbr"].astype("string").str.upper()
    long["fbi_cde_estimated_total"] = pd.to_numeric(
        long["fbi_cde_estimated_total"], errors="coerce"
    )
    return long[["state_abbr", "offense", "fbi_cde_estimated_total"]].dropna(subset=["offense"])


def _unit_columns(config: BenchmarkImputationConfig) -> list[str]:
    """The unit table's schema for a given config.

    With every v2 flag off this is exactly `UNIT_COLUMNS`: the legacy artifact gains no
    column, which is the byte-safety guarantee the promoted chain relies on.
    """
    columns = list(UNIT_COLUMNS)
    if config.uses_v2_rate_structure:
        columns += list(RATE_STRUCTURE_UNIT_COLUMNS)
    if config.attach_empirical_bounds:
        columns += list(EMPIRICAL_BOUNDS_UNIT_COLUMNS)
    return columns


def _assert_lane_partition(bg_crosswalk: pd.DataFrame) -> None:
    """The municipal and county-remainder lanes must partition every block group.

    This is the load-bearing fact behind "a silent municipal PD's territory gets its
    mass through its own municipal cell OR the county pool, never both".
    """
    if bg_crosswalk.empty:
        return
    shares = bg_crosswalk.groupby("block_group_geoid", dropna=False)["allocation_share"].sum()
    worst = float((shares - 1.0).abs().max())
    if worst > 1e-6:
        raise ValueError(
            "block-group allocation shares do not partition the municipal and county "
            f"remainder lanes; max |sum-1| = {worst:.3e}"
        )


def build_benchmark_imputation(
    *,
    paths: RepoPaths,
    controls: pd.DataFrame,
    cde_estimates: pd.DataFrame,
    agency_preferred: pd.DataFrame,
    agency_estimates: pd.DataFrame,
    succession_ledger: pd.DataFrame | None = None,
    config: BenchmarkImputationConfig = BenchmarkImputationConfig(),
) -> BenchmarkImputation:
    year = int(config.year)
    controls = controls.copy()
    controls["state_fips"] = controls["state_fips"].astype("string").str.zfill(2)
    controls["state_abbr"] = controls["state_abbr"].astype("string").str.upper()
    controls["offense"] = controls["offense"].astype("string")

    bg_crosswalk = _load_block_group_crosswalk(paths)
    _assert_lane_partition(bg_crosswalk)
    units = build_unit_exposure(bg_crosswalk)
    ledger = build_silent_agency_ledger(
        paths=paths,
        config=config,
        agency_preferred=agency_preferred,
        agency_estimates=agency_estimates,
        succession_ledger=succession_ledger,
    )

    locked = pd.concat(
        [
            _locked_unit_targets(controls),
            _county_lane_locked_counts(agency_estimates=agency_estimates, ledger=ledger),
        ],
        ignore_index=True,
    )
    locked_wide = (
        locked.pivot_table(
            index=["unit_id", "state_fips"],
            columns="offense",
            values="locked_count",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()
        if not locked.empty
        else pd.DataFrame(columns=["unit_id", "state_fips"])
    )
    units = units.merge(locked_wide, on=["unit_id", "state_fips"], how="left")
    for offense in OFFENSES_7:
        if offense not in units.columns:
            units[offense] = 0.0
        units[offense] = pd.to_numeric(units[offense], errors="coerce").fillna(0.0).clip(lower=0.0)
    units["locked_total"] = units[list(OFFENSES_7)].sum(axis=1)
    units["exposure_population"] = pd.to_numeric(
        units["exposure_population"], errors="coerce"
    ).fillna(0.0)

    # Preserve the offense dimension through eligibility. One usable offense must not
    # make an agency supported for the other six.
    weighted_links = ledger[pd.to_numeric(ledger["weight"], errors="coerce").fillna(0.0).gt(0.0)]
    status = ledger.groupby(["unit_id", "offense"], dropna=False).agg(
        has_supported_agency=("is_supported", "max"),
        has_eligible_silent_agency=("is_eligible_silent", "max"),
    ).reset_index()
    fill_status = weighted_links.groupby(["unit_id", "offense"], dropna=False)["is_fill_covered"].max().rename("has_fill_covered_agency").reset_index()
    status = status.merge(fill_status, on=["unit_id", "offense"], how="left")
    silent_names = ledger[ledger["is_eligible_silent"]].groupby(["unit_id", "offense"], dropna=False)["ori9"].agg(
        silent_agency_oris=lambda values: "|".join(sorted(str(v) for v in values.dropna().unique())),
        silent_agency_count="nunique",
    ).reset_index()
    status = status.merge(silent_names, on=["unit_id", "offense"], how="left")
    # A unit can only be given a target if some control row can carry it: its own for a
    # municipal jurisdiction, the state remainder pool row for a county cell. Municipal
    # polygons that exist in geometry but never entered the control panel are recorded
    # and skipped rather than silently losing their sub-target downstream.
    municipal_control_ids = set(
        controls.loc[
            controls["jurisdiction_type"].astype("string").eq("municipal"), "jurisdiction_id"
        ].astype("string")
    )
    remainder_control_states = set(
        controls.loc[
            controls["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE), "state_fips"
        ].astype("string")
    )
    units["has_control_row"] = np.where(
        units["unit_kind"].eq(MUNICIPAL_UNIT_KIND),
        units["unit_id"].isin(municipal_control_ids),
        units["state_fips"].isin(remainder_control_states),
    )
    unit_offense = units.merge(pd.DataFrame({"offense": list(OFFENSES_7)}), how="cross")
    unit_offense = unit_offense.merge(status, on=["unit_id", "offense"], how="left")
    for column in ("has_supported_agency", "has_fill_covered_agency", "has_eligible_silent_agency"):
        unit_offense[column] = unit_offense[column].astype("boolean").fillna(False).astype(bool)
    unit_offense["silent_agency_count"] = pd.to_numeric(unit_offense["silent_agency_count"], errors="coerce").fillna(0).astype(int)
    unit_offense["silent_agency_oris"] = unit_offense["silent_agency_oris"].astype("string")
    unit_offense["locked_count"] = [float(row[offense]) for offense, row in zip(unit_offense["offense"], unit_offense.to_dict(orient="records"), strict=True)]

    # Zero is valid training evidence when its offense is supported.
    observed = unit_offense[
        unit_offense["has_supported_agency"] & unit_offense["exposure_population"].gt(0.0)
    ].copy()
    # Territory whose every agency is silent, and whose silence is attributable to an
    # identified, active, primary agency -- the imputation units. Stated over the agency
    # ledger's own dispositions rather than over "the control row happens to be empty":
    # the empty control row is the CONSEQUENCE of every agency being silent, and asserted
    # below to be one, not the rule.
    silent_candidates = unit_offense[
        ~unit_offense["has_supported_agency"]
        & ~unit_offense["has_fill_covered_agency"]
        & unit_offense["has_eligible_silent_agency"]
        & unit_offense["exposure_population"].gt(0.0)
    ].copy()
    # Territory an agency estimate already sizes in part while another agency inside it
    # is silent. Not imputed: the exposure cannot be split between the two agencies, so
    # imputing the whole unit would double the covered part. Counted, not hidden.
    partially_covered = unit_offense[
        ~unit_offense["has_supported_agency"]
        & unit_offense["has_fill_covered_agency"]
        & unit_offense["has_eligible_silent_agency"]
        & unit_offense["exposure_population"].gt(0.0)
    ].copy()

    state_abbr_by_fips = (
        controls[["state_fips", "state_abbr"]]
        .dropna()
        .drop_duplicates(subset=["state_fips"], keep="first")
        .set_index("state_fips")["state_abbr"]
    )
    for frame in (observed, silent_candidates, partially_covered):
        frame["state_abbr"] = frame["state_fips"].map(state_abbr_by_fips).astype("string")
    in_scope_states = set(state_abbr_by_fips.values)
    silent = silent_candidates[
        silent_candidates["state_abbr"].isin(in_scope_states)
    ].copy()
    partially_covered = partially_covered[
        partially_covered["state_abbr"].isin(in_scope_states)
    ].copy()
    excluded_states = {str(s).upper() for s in config.excluded_state_abbrs}
    silent["state_excluded"] = silent["state_abbr"].astype("string").isin(excluded_states)
    observed = observed[observed["state_abbr"].notna()].copy()
    _assert_every_silent_unit_lands_on_exactly_one_empty_control_row(silent)

    band_edges = tuple(float(edge) for edge in config.municipal_population_band_edges)
    uses_v2 = bool(config.uses_v2_rate_structure)
    if uses_v2:
        observed = observed.assign(
            **{POPULATION_BAND_COLUMN: population_band(observed["exposure_population"], edges=band_edges)}
        )
        silent = silent.assign(
            **{POPULATION_BAND_COLUMN: population_band(silent["exposure_population"], edges=band_edges)}
        )

    pooling_constants, validation = select_pooling_constants(observed, config=config)
    band_pooling_constants: dict[str, float] = {}
    band_validation = pd.DataFrame()
    if config.enable_size_aware_municipal_rates:
        band_pooling_constants, band_validation = select_band_pooling_constants(
            observed, config=config, pooling_constants=pooling_constants
        )
    rates_by_offense = {
        offense: _fit_rate_levels(
            observed[observed["offense"].eq(offense)],
            offense=offense,
            pooling_constant=pooling_constants[offense],
            band_pooling_constant=(
                float(band_pooling_constants[offense])
                if config.enable_size_aware_municipal_rates
                else None
            ),
            band_edges=band_edges,
        )
        for offense in OFFENSES_7
    }
    floor_absolute = (
        float(config.cell_exposure_floor_absolute) if config.enable_cell_exposure_floor else None
    )

    eligible = silent[~silent["state_excluded"]].copy()
    long_rows: list[pd.DataFrame] = []
    refusal_rows: list[pd.DataFrame] = []
    for offense in OFFENSES_7:
        eligible_offense = eligible[eligible["offense"].eq(offense)].copy()
        if eligible_offense.empty:
            continue
        extra: dict[str, np.ndarray] = {}
        if uses_v2:
            served = predict_pooled_rate_v2(
                eligible_offense,
                rates_by_offense[offense],
                floor_absolute=floor_absolute,
                floor_unit_multiple=float(config.cell_exposure_floor_unit_multiple),
            )
            refused = served["rate_level"].eq(RATE_LEVEL_REFUSED).to_numpy()
            if refused.any():
                refusal_rows.append(
                    eligible_offense.loc[refused]
                    .assign(
                        year=year,
                        offense=offense,
                        required_cell_exposure=served.loc[refused, "required_cell_exposure"].to_numpy(),
                        required_parent_exposure=served.loc[refused, "required_parent_exposure"].to_numpy(),
                        available_parent_exposure=served.loc[refused, "available_parent_exposure"].to_numpy(),
                        refusal_status=REFUSAL_STATUS_CELL_EXPOSURE_FLOOR,
                    )
                    .reindex(columns=REFUSAL_COLUMNS)
                )
            eligible_offense = eligible_offense.loc[~refused]
            served = served.loc[~refused]
            if eligible_offense.empty:
                continue
            rate = served["pooled_rate"].to_numpy(dtype=float)
            extra = {
                POPULATION_BAND_COLUMN: eligible_offense[POPULATION_BAND_COLUMN].to_numpy(),
                "rate_level": served["rate_level"].to_numpy(),
                "rate_escalation_reason": served["rate_escalation_reason"].to_numpy(),
            }
        else:
            rate = predict_pooled_rate(eligible_offense, rates_by_offense[offense])
        long_rows.append(
            pd.DataFrame(
                {
                    "state_fips": eligible_offense["state_fips"].to_numpy(),
                    "state_abbr": eligible_offense["state_abbr"].to_numpy(),
                    "unit_kind": eligible_offense["unit_kind"].to_numpy(),
                    "unit_id": eligible_offense["unit_id"].to_numpy(),
                    "county_geoid": eligible_offense["county_geoid"].to_numpy(),
                    "exposure_population": eligible_offense["exposure_population"].to_numpy(dtype=float),
                    "land_area_sq_mi": eligible_offense["land_area_sq_mi"].to_numpy(dtype=float),
                    "urbanicity": eligible_offense["urbanicity"].to_numpy(),
                    "silent_agency_count": eligible_offense["silent_agency_count"].to_numpy(),
                    "silent_agency_oris": eligible_offense["silent_agency_oris"].to_numpy(),
                    "offense": offense,
                    "pooled_rate": rate,
                    "modeled_expected_count": rate
                    * eligible_offense["exposure_population"].to_numpy(dtype=float),
                    "modeled_variance": np.maximum(
                        rate * eligible_offense["exposure_population"].to_numpy(dtype=float),
                        np.finfo(float).eps,
                    ),
                    **extra,
                }
            )
        )
    unit_columns = _unit_columns(config)
    unit_long = (
        pd.concat(long_rows, ignore_index=True)
        if long_rows
        else pd.DataFrame(columns=unit_columns)
    )
    refusals = (
        pd.concat(refusal_rows, ignore_index=True)
        if refusal_rows
        else pd.DataFrame(columns=REFUSAL_COLUMNS)
    )

    identity = _build_state_identity(
        paths=paths,
        controls=controls,
        cde_estimates=cde_estimates,
        agency_preferred=agency_preferred,
        unit_long=unit_long,
        silent=silent,
        year=year,
        excluded_states=excluded_states,
    )

    if not unit_long.empty:
        unit_long = unit_long.merge(
            identity[["state_fips", "offense", "benchmark_scale", "benchmark_weight"]],
            on=["state_fips", "offense"],
            how="left",
        )
        unit_long["benchmark_scale"] = pd.to_numeric(
            unit_long["benchmark_scale"], errors="coerce"
        ).fillna(0.0)
        unit_long["benchmark_weight"] = pd.to_numeric(unit_long["benchmark_weight"], errors="coerce").fillna(0.0)
        unit_long["imputed_count"] = (
            unit_long["modeled_expected_count"] * unit_long["benchmark_scale"]
        )
        unit_long["year"] = year
        unit_long["imputation_source"] = BENCHMARK_IMPUTATION_SOURCE
        unit_long = unit_long[unit_long["imputed_count"].gt(0.0)].copy()
    bounds = pd.DataFrame()
    if config.attach_empirical_bounds:
        bounds = load_empirical_bounds(paths, rule_version=config.bounds_rule_version)
        unit_long = attach_empirical_bounds(unit_long, bounds=bounds)
    unit_long = unit_long.reindex(columns=unit_columns)

    diagnostics = _build_diagnostics(
        paths=paths,
        config=config,
        ledger=ledger,
        observed=observed,
        silent=silent,
        partially_covered=partially_covered,
        units=unit_long,
        identity=identity,
        pooling_constants=pooling_constants,
        validation=validation,
        band_pooling_constants=band_pooling_constants,
        band_validation=band_validation,
        refusals=refusals,
        bounds=bounds,
    )
    imputation = BenchmarkImputation(
        units=unit_long.sort_values(["state_fips", "unit_id", "offense"], kind="mergesort").reset_index(drop=True),
        state_identity=identity.reindex(columns=STATE_IDENTITY_COLUMNS)
        .sort_values(["state_fips", "offense"], kind="mergesort")
        .reset_index(drop=True),
        validation=validation,
        diagnostics=diagnostics,
        band_validation=band_validation,
        refusals=refusals.reset_index(drop=True),
    )
    assert_benchmark_imputation_invariants(imputation)
    if config.enforce_mass_ratchet:
        check_benchmark_mass_against_baseline(imputation, paths=paths)
    return imputation


def _build_state_identity(
    *,
    paths: RepoPaths,
    controls: pd.DataFrame,
    cde_estimates: pd.DataFrame,
    agency_preferred: pd.DataFrame,
    unit_long: pd.DataFrame,
    silent: pd.DataFrame,
    year: int,
    excluded_states: set[str],
) -> pd.DataFrame:
    locked_state = (
        controls.groupby(["state_fips", "state_abbr", "offense"], dropna=False, as_index=False)[
            "adjusted_count_ags_core"
        ]
        .sum()
        .rename(columns={"adjusted_count_ags_core": "locked_observed_total"})
    )
    benchmark = _cde_state_benchmark_long(cde_estimates)
    identity = locked_state.merge(benchmark, on=["state_abbr", "offense"], how="left")
    identity["benchmark_residual"] = (
        pd.to_numeric(identity["fbi_cde_estimated_total"], errors="coerce")
        - pd.to_numeric(identity["locked_observed_total"], errors="coerce")
    )

    pool = (
        unit_long.groupby(["state_fips", "offense"], dropna=False, as_index=False)[
            "modeled_expected_count"
        ]
        .sum()
        .rename(columns={"modeled_expected_count": "modeled_pool"})
        if not unit_long.empty
        else pd.DataFrame(columns=["state_fips", "offense", "modeled_pool"])
    )
    identity = identity.merge(pool, on=["state_fips", "offense"], how="left")
    identity["modeled_pool"] = pd.to_numeric(identity["modeled_pool"], errors="coerce").fillna(0.0)
    variance = (
        unit_long.groupby(["state_fips", "offense"], dropna=False, as_index=False)["modeled_variance"]
        .sum().rename(columns={"modeled_variance": "modeled_pool_variance"})
        if not unit_long.empty else pd.DataFrame(columns=["state_fips", "offense", "modeled_pool_variance"])
    )
    identity = identity.merge(variance, on=["state_fips", "offense"], how="left")
    identity["modeled_pool_variance"] = pd.to_numeric(identity["modeled_pool_variance"], errors="coerce").fillna(0.0)

    # No release-vintage series is present in-repo. Estimate offense uncertainty from
    # the specified CA/NY/CT fallback: summed high-coverage preferred state panels
    # versus CDE, then inflate each state by its roster reporting-coverage fraction.
    comparison_states = {"CA", "NY", "CT"}
    verified_replacement = agency_preferred.get(
        "level_repair_mode", pd.Series("", index=agency_preferred.index)
    ).astype("string").isin({"source_supported_replacement", "swap_reclassification"})
    supported_for_comparison = (
        agency_preferred["usable_as_observed"].fillna(False).astype(bool)
        | verified_replacement
    )
    panel = agency_preferred[
        agency_preferred["state_abbr"].astype("string").isin(comparison_states)
        & supported_for_comparison
    ].copy()
    replacement = pd.to_numeric(
        panel.get("level_lane_replacement_count", pd.Series(np.nan, index=panel.index)),
        errors="coerce",
    )
    panel["_comparison_count"] = replacement.where(
        panel.get("level_repair_mode", pd.Series("", index=panel.index)).astype("string").isin(
            {"source_supported_replacement", "swap_reclassification"}
        ),
        pd.to_numeric(panel["preferred_count"], errors="coerce").fillna(0.0),
    )
    panel_totals = panel.groupby(["state_abbr", "offense"], dropna=False)["_comparison_count"].sum()
    cde_long = _cde_state_benchmark_long(cde_estimates).set_index(["state_abbr", "offense"])["fbi_cde_estimated_total"]
    uncertainty_rows = []
    for offense in OFFENSES_7:
        errors = []
        for state in sorted(comparison_states):
            local = panel_totals.get((state, offense), np.nan)
            benchmark_value = cde_long.get((state, offense), np.nan)
            if pd.notna(local) and pd.notna(benchmark_value) and float(benchmark_value) > 0:
                errors.append(float(np.log((float(local) + 0.5) / (float(benchmark_value) + 0.5))))
        relative = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
        uncertainty_rows.append({"offense": offense, "benchmark_relative_uncertainty": relative, "comparison_state_count": len(errors)})
    uncertainty = pd.DataFrame(uncertainty_rows)
    if uncertainty["benchmark_relative_uncertainty"].isna().any():
        raise ValueError("benchmark uncertainty could not be estimated for every offense from CA/NY/CT")
    roster_path = (
        paths.data_dir
        / f"FBI-CDE-Agency-Rosters-{int(year)}"
        / "parsed"
        / f"agency_rosters_{int(year)}.csv"
    )
    roster = pd.read_csv(roster_path, dtype="string")
    roster_counts = roster.groupby("state_abbr")["ori"].nunique()
    supported_counts = agency_preferred[
        agency_preferred["usable_as_observed"].fillna(False).astype(bool)
        | agency_preferred.get(
            "current_row_is_true_partial", pd.Series(False, index=agency_preferred.index)
        ).fillna(False).astype(bool)
        | verified_replacement
    ].groupby("state_abbr")["ori9"].nunique()
    coverage = (supported_counts / roster_counts).clip(lower=1e-6, upper=1.0)
    identity = identity.merge(uncertainty[["offense", "benchmark_relative_uncertainty"]], on="offense", how="left")
    identity["reporting_coverage_fraction"] = identity["state_abbr"].map(coverage).fillna(float(coverage.median())).clip(lower=1e-6, upper=1.0)
    benchmark_value = pd.to_numeric(identity["fbi_cde_estimated_total"], errors="coerce").fillna(0.0)
    rel = pd.to_numeric(identity["benchmark_relative_uncertainty"], errors="coerce")
    identity["benchmark_variance"] = benchmark_value + (rel * benchmark_value) ** 2 / identity["reporting_coverage_fraction"]
    identity["benchmark_standard_error"] = np.sqrt(identity["benchmark_variance"].clip(lower=0.0))

    unit_stats = (
        silent.groupby(["state_fips", "offense"], dropna=False)
        .agg(silent_unit_count=("unit_id", "nunique"), silent_unit_population=("exposure_population", "sum"))
        .reset_index()
        if not silent.empty
        else pd.DataFrame(columns=["state_fips", "offense", "silent_unit_count", "silent_unit_population"])
    )
    identity = identity.merge(unit_stats, on=["state_fips", "offense"], how="left")
    identity["silent_unit_count"] = pd.to_numeric(identity["silent_unit_count"], errors="coerce").fillna(0).astype(int)
    identity["silent_unit_population"] = pd.to_numeric(
        identity["silent_unit_population"], errors="coerce"
    ).fillna(0.0)

    residual = identity["benchmark_residual"].to_numpy(dtype=float)
    modeled = identity["modeled_pool"].to_numpy(dtype=float)
    model_variance = identity["modeled_pool_variance"].to_numpy(dtype=float)
    benchmark_variance = identity["benchmark_variance"].to_numpy(dtype=float)
    denom = model_variance + benchmark_variance
    weight = np.divide(model_variance, denom, out=np.zeros_like(model_variance), where=denom > 0)
    posterior_silent = modeled + weight * (residual - modeled)
    # When accepted local mass already exceeds CDE, the aggregate contains no evidence
    # that a particular silent territory is zero. Retain the model-prior component not
    # absorbed by the benchmark weight.
    posterior_silent = np.where(residual <= 0.0, modeled * (1.0 - weight), posterior_silent)
    posterior_silent = np.where(modeled > 0.0, np.maximum(posterior_silent, np.finfo(float).eps), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.divide(posterior_silent, modeled, out=np.zeros_like(modeled), where=modeled > 0.0)
    identity["benchmark_weight"] = weight
    identity["benchmark_scale"] = scale
    identity["imputed_total"] = posterior_silent
    identity["posterior_total"] = pd.to_numeric(identity["locked_observed_total"], errors="coerce").fillna(0.0) + posterior_silent
    identity["standardized_disagreement"] = np.divide(
        pd.to_numeric(identity["locked_observed_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float) + modeled - benchmark_value.to_numpy(dtype=float),
        np.sqrt(np.maximum(denom, np.finfo(float).eps)),
    )
    identity["unused_benchmark_headroom"] = np.maximum(0.0, residual - posterior_silent)
    identity["unfilled_modeled_pool"] = np.maximum(0.0, modeled - posterior_silent)

    locked_exceeds = pd.to_numeric(identity["fbi_cde_estimated_total"], errors="coerce").lt(
        pd.to_numeric(identity["locked_observed_total"], errors="coerce")
    )
    no_benchmark = pd.to_numeric(identity["fbi_cde_estimated_total"], errors="coerce").isna()
    state_excluded = identity["state_abbr"].astype("string").isin(excluded_states)
    identity["conflict_kind"] = np.select(
        [
            no_benchmark,
            state_excluded,
            locked_exceeds,
            identity["modeled_pool"].le(0.0),
            identity["unused_benchmark_headroom"].gt(1e-9),
            identity["unfilled_modeled_pool"].gt(1e-9),
        ],
        [
            "benchmark_unavailable",
            "state_excluded_from_imputation",
            "locked_exceeds_benchmark",
            "no_eligible_silent_territory",
            "benchmark_above_soft_posterior",
            "benchmark_below_model_prior",
        ],
        default="reconciled",
    )
    identity["year"] = int(year)
    uncertainty_path = paths.state_dir / "controls" / f"benchmark_uncertainty_{int(year)}.csv"
    uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
    uncertainty.assign(
        method="CA_NY_CT_high_coverage_log_dispersion_plus_roster_coverage",
        vintage_basis="genuine_revision_vintages_unavailable_in_repo",
    ).to_csv(uncertainty_path, index=False)
    return identity


def _build_diagnostics(
    *,
    paths: RepoPaths,
    config: BenchmarkImputationConfig,
    ledger: pd.DataFrame,
    observed: pd.DataFrame,
    silent: pd.DataFrame,
    partially_covered: pd.DataFrame,
    units: pd.DataFrame,
    identity: pd.DataFrame,
    pooling_constants: dict[str, float],
    validation: pd.DataFrame,
    band_pooling_constants: dict[str, float] | None = None,
    band_validation: pd.DataFrame | None = None,
    refusals: pd.DataFrame | None = None,
    bounds: pd.DataFrame | None = None,
) -> dict[str, object]:
    unsupported = ledger[~ledger["is_supported"]]
    screens = {
        "silent_agency_links": int(len(unsupported)),
        "excluded_dead_or_absent": int(unsupported["is_dead"].sum()),
        "nonreporting_but_in_fbi_roster": int(unsupported["is_nonreporting_but_rostered"].sum()),
        "excluded_consolidated_footprint": int(unsupported["is_consolidated"].sum()),
        "excluded_local_resolution_override": int(unsupported["is_override_routed"].sum()),
        "excluded_non_primary_agency_type": int((~unsupported["is_primary_type"]).sum()),
        "eligible_silent_agency_links": int(unsupported["is_eligible_silent"].sum()),
    }
    by_kind = (
        silent.groupby("unit_kind")
        .agg(units=("unit_id", "nunique"), population=("exposure_population", "sum"))
        .to_dict(orient="index")
        if not silent.empty
        else {}
    )
    return {
        "year": int(config.year),
        "imputation_source": BENCHMARK_IMPUTATION_SOURCE,
        "excluded_state_abbrs": list(config.excluded_state_abbrs),
        "pooling_constants": {k: float(v) for k, v in pooling_constants.items()},
        "validation_folds": int(config.validation_folds),
        "validation_seed": int(config.validation_seed),
        "rate_model_fit_units": int(len(observed)),
        # Units holding both a silent primary agency and an agency the fill ladder
        # already sizes. Not imputable without splitting exposure between the two, so
        # they are recorded here rather than silently dropped by an emptiness test.
        "partially_covered_units_not_imputed": int(len(partially_covered)),
        "partially_covered_unit_population": float(
            pd.to_numeric(partially_covered["exposure_population"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        if not partially_covered.empty
        else 0.0,
        "silent_units": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_kind.items()},
        "agency_screens": screens,
        "national_imputed_mass": float(
            pd.to_numeric(units["imputed_count"], errors="coerce").fillna(0.0).sum()
        )
        if not units.empty
        else 0.0,
        "national_modeled_pool": float(identity["modeled_pool"].sum()),
        "national_benchmark_residual": float(identity["benchmark_residual"].sum()),
        "conflict_kind_counts": identity["conflict_kind"].value_counts().to_dict(),
        "validation_table": validation.to_dict(orient="records"),
        "benchmark_reconciliation": BENCHMARK_RECONCILIATION_NOTES,
        # Self-identifying: an artifact built on any v2 imputation lane says so in its
        # own diagnostics, so a candidate can never be mistaken for a legacy build.
        "imputation_v2": _imputation_v2_record(
            paths=paths,
            config=config,
            units=units,
            band_pooling_constants=band_pooling_constants or {},
            band_validation=band_validation if band_validation is not None else pd.DataFrame(),
            refusals=refusals if refusals is not None else pd.DataFrame(),
            bounds=bounds if bounds is not None else pd.DataFrame(),
        ),
    }


def _imputation_v2_record(
    *,
    paths: RepoPaths,
    config: BenchmarkImputationConfig,
    units: pd.DataFrame,
    band_pooling_constants: dict[str, float],
    band_validation: pd.DataFrame,
    refusals: pd.DataFrame,
    bounds: pd.DataFrame,
) -> dict[str, object]:
    record: dict[str, object] = {
        "rate_rule_version": config.rate_rule_version,
        "contract": IMPUTATION_V2_CONTRACT,
        "size_aware_municipal_rates": {
            "enabled": bool(config.enable_size_aware_municipal_rates),
            "lane": MUNICIPAL_UNIT_KIND,
            "population_band_edges": [float(e) for e in config.municipal_population_band_edges],
            "band_pooling_constants": {k: float(v) for k, v in band_pooling_constants.items()},
            "selection": "profile search on K_band with the legacy K held fixed; same folds, grid and out-of-sample Poisson deviance",
        },
        "cell_exposure_floor": {
            "enabled": bool(config.enable_cell_exposure_floor),
            "absolute": float(config.cell_exposure_floor_absolute),
            "unit_population_multiple": float(config.cell_exposure_floor_unit_multiple),
            "rule": (
                "band/cell exposure >= max(absolute, multiple x unit population); "
                "escalate band -> cell -> (state, lane); the terminal (state, lane) "
                "level requires the absolute term only; refuse above (state, lane)"
            ),
            "refused_unit_offense_cells": int(len(refusals)),
            "refused_units": int(refusals["unit_id"].nunique()) if not refusals.empty else 0,
            "refused_population": float(
                pd.to_numeric(refusals["exposure_population"], errors="coerce").fillna(0.0).sum()
            )
            if not refusals.empty
            else 0.0,
            "refusals": refusals.head(500).to_dict(orient="records") if not refusals.empty else [],
        },
        "empirical_bounds": {
            "enabled": bool(config.attach_empirical_bounds),
            "rule_version": config.bounds_rule_version if config.attach_empirical_bounds else None,
            "path": str(empirical_bounds_path(paths)),
            "central_correction_applied_to_point_estimate": False,
            "strata": sorted(bounds["stratum"].dropna().unique().tolist()) if not bounds.empty else [],
        },
    }
    if config.uses_v2_rate_structure and not units.empty and "rate_level" in units.columns:
        record["rate_level_counts"] = {
            str(k): int(v) for k, v in units["rate_level"].value_counts().items()
        }
        record["rate_escalation_counts"] = {
            str(k): int(v)
            for k, v in units["rate_escalation_reason"].fillna("").value_counts().items()
        }
    if not band_validation.empty:
        record["band_validation_table"] = band_validation.to_dict(orient="records")
    return record


# Explicit reconciliation of our controls against the FBI CDE state series. Recorded
# rather than fudged: every line here is a difference we know about and have decided
# how to treat.
BENCHMARK_RECONCILIATION_NOTES: dict[str, str] = {
    "year": (
        "The CDE estimated-crimes bundle is a cumulative 1979-2024 series; only the "
        "row for the target data year is read, matching the controls' reference year."
    ),
    "offense_definitions": (
        "murder = CDE homicide (murder and nonnegligent manslaughter); rape = "
        "rape_revised (the legacy rape column is empty from 2017 onward and is never "
        "read); the remaining five map one-to-one. Our own rape series is the revised "
        "definition throughout, so the pairing is definitional, not approximate."
    ),
    "coverage": (
        "The CDE state row estimates the state's whole resident population. Our locked "
        "total is the sum of every in-scope jurisdiction control (municipal + state "
        "nonmunicipal remainder + statewide overlap), which is the same universe. Two "
        "known asymmetries, both left in the residual rather than adjusted for: tribal "
        "agencies, which the FBI reports in a separate table and may or may not fold "
        "into the state figure, and federal agencies inside a state, which we exclude "
        "from municipal territory but which can appear in either series."
    ),
    "scope": (
        "AK, HI and the territories are out of production scope on both sides: the "
        "controls exclude them and the CDE frame handed in here is filtered the same "
        "way, so no state is compared against a partial counterpart."
    ),
    "vintage": (
        "CDE estimates are revised in later releases; this is the vintage shipped with "
        "the target year's bundle in data/. Because the residual is capped by the "
        "modelled pool, a later revision moves imputed mass smoothly rather than "
        "producing a discontinuity."
    ),
    "residual_interpretation": (
        "M_so is an upper bound on missing mass, not a measurement of it: it also "
        "absorbs rounding in the published CDE figures and any of our own partial-year "
        "uplift that runs above the FBI's estimate. That is exactly why the imputed "
        "amount is min(model expectation, benchmark residual)."
    ),
}


# --- invariants and ratchet -------------------------------------------------


def assert_benchmark_imputation_invariants(imputation: BenchmarkImputation) -> None:
    """Fail closed on the v2 soft-reconciliation properties."""
    units = imputation.units
    identity = imputation.state_identity

    # A benchmark conflict remains modeled: local mass is never destroyed and positive
    # silent priors are never mechanically zeroed by CDE exhaustion.
    conflicted = identity[identity["conflict_kind"].eq("locked_exceeds_benchmark")]
    if not conflicted.empty:
        keys = set(zip(conflicted["state_fips"], conflicted["offense"], strict=True))
        offending = (
            units[
                [
                    (state, offense) in keys
                    for state, offense in zip(units["state_fips"], units["offense"], strict=True)
                ]
            ]
            if not units.empty
            else units
        )
        if (pd.to_numeric(conflicted["modeled_pool"], errors="coerce").gt(0.0) & pd.to_numeric(conflicted["imputed_total"], errors="coerce").le(0.0)).any():
            raise ValueError(
                "positive silent prior was zeroed solely because locked local mass exceeds CDE"
            )

    if not units.empty:
        values = pd.to_numeric(units["imputed_count"], errors="coerce")
        bad = units[~np.isfinite(values) | values.lt(0.0)]
        if not bad.empty:
            raise ValueError(
                f"{len(bad)} benchmark-imputed target(s) are non-finite or negative: "
                + str(bad[["unit_id", "offense", "imputed_count"]].head(20).to_dict(orient="records"))
            )

        # Unit mass must equal the state/offense posterior silent total.
        per_cell = units.groupby(["state_fips", "offense"], dropna=False)["imputed_count"].sum()
        expected = identity.set_index(["state_fips", "offense"])["imputed_total"]
        residual = (per_cell - expected.reindex(per_cell.index).fillna(0.0)).abs().max()
        if pd.notna(residual) and float(residual) > 1e-6:
            raise ValueError(
                "benchmark unit mass does not equal the soft posterior silent total by "
                f"{float(residual):.6f}"
            )

        # (2) The partition: no unit appears twice for one offense, and the municipal
        # and county lanes never claim the same unit id.
        duplicated = units.duplicated(subset=["unit_id", "offense"]).sum()
        if int(duplicated) > 0:
            raise ValueError(
                f"{int(duplicated)} benchmark-imputation unit(s) appear more than once for an offense"
            )
        lanes = units.groupby("unit_id")["unit_kind"].nunique()
        if int((lanes > 1).sum()) > 0:
            raise ValueError(
                "benchmark-imputation unit(s) claimed by both the municipal and county "
                f"remainder lanes: {sorted(lanes[lanes > 1].index)[:10]}"
            )

        # (3) v2 rate structure: every published unit was served by a named level, and a
        # refused unit never reaches the unit table.
        if "rate_level" in units.columns:
            served = units["rate_level"].astype("string")
            unserved = units[served.isna() | served.eq("")]
            if not unserved.empty:
                raise ValueError(
                    f"{len(unserved)} imputed unit(s) carry no rate level: "
                    + str(unserved[["unit_id", "offense"]].head(10).to_dict(orient="records"))
                )
            refused = units[served.eq(RATE_LEVEL_REFUSED)]
            if not refused.empty:
                raise ValueError(
                    f"{len(refused)} unit(s) refused by the cell-exposure floor were "
                    "published anyway: "
                    + str(refused[["unit_id", "offense"]].head(10).to_dict(orient="records"))
                )
            if not imputation.refusals.empty:
                published = set(
                    zip(units["unit_id"].astype(str), units["offense"].astype(str), strict=True)
                )
                collided = [
                    key
                    for key in zip(
                        imputation.refusals["unit_id"].astype(str),
                        imputation.refusals["offense"].astype(str),
                        strict=True,
                    )
                    if key in published
                ]
                if collided:
                    raise ValueError(
                        f"{len(collided)} refused unit-offense cell(s) also carry imputed "
                        f"mass: {collided[:10]}"
                    )

        # (4) Empirical bounds, when attached, bracket the point estimate in order.
        if "bound_lo_80" in units.columns:
            lo95 = pd.to_numeric(units["bound_lo_95"], errors="coerce")
            lo80 = pd.to_numeric(units["bound_lo_80"], errors="coerce")
            hi80 = pd.to_numeric(units["bound_hi_80"], errors="coerce")
            hi95 = pd.to_numeric(units["bound_hi_95"], errors="coerce")
            ordered = lo95.le(lo80) & lo80.le(hi80) & hi80.le(hi95)
            if not bool(ordered.fillna(False).all()):
                raise ValueError(
                    "attached empirical bounds are not ordered lo95 <= lo80 <= hi80 <= hi95 "
                    f"on {int((~ordered.fillna(False)).sum())} unit-offense cell(s)"
                )
            correction = pd.to_numeric(units["central_correction"], errors="coerce")
            if not bool((correction.gt(0.0) & np.isfinite(correction)).all()):
                raise ValueError("attached central_correction is non-positive or non-finite")


def check_benchmark_mass_against_baseline(
    imputation: BenchmarkImputation, *, paths: RepoPaths
) -> dict[str, float]:
    """Ratchet the national imputed total on the existing fill-mass baseline contract.

    Same file, same tolerance, same rule as the agency fill ratchet: only increases
    beyond tolerance fail, and accepting a genuine increase means editing the baseline
    deliberately with a rationale in docs/STATE.md.
    """
    measured = {BENCHMARK_IMPUTED_MASS_METRIC: float(imputation.imputed_total)}
    baseline_path = paths.repo_root / "configs" / FILL_MASS_BASELINE_FILENAME
    if not baseline_path.exists():
        return measured
    baseline = json.loads(baseline_path.read_text())
    if BENCHMARK_IMPUTED_MASS_METRIC not in baseline:
        return measured
    tolerance = float(baseline["tolerance_fraction"])
    limit = float(baseline[BENCHMARK_IMPUTED_MASS_METRIC]) * (1.0 + tolerance)
    if measured[BENCHMARK_IMPUTED_MASS_METRIC] > limit:
        raise ValueError(
            f"benchmark-imputed mass regressed beyond the {tolerance:.0%} tolerance in "
            f"{baseline_path}: measured={measured[BENCHMARK_IMPUTED_MASS_METRIC]:.1f} "
            f"baseline={float(baseline[BENCHMARK_IMPUTED_MASS_METRIC]):.1f}"
        )
    return measured


# --- consumers --------------------------------------------------------------


def apply_benchmark_imputation_to_controls(
    controls: pd.DataFrame, *, units: pd.DataFrame
) -> pd.DataFrame:
    """Add imputed mass to the control rows that own the territory.

    Municipal units take their imputed count on their own control row. County remainder
    units have no control row of their own -- the state nonmunicipal remainder pool is
    one jurisdiction -- so their mass is added to the pool row and split back out per
    county by allocation.py from the same unit table. Locked observed rows are never
    touched: only rows whose current target is zero receive mass.
    """
    out = controls.copy()
    for column, default in (
        ("benchmark_imputed_count", 0.0),
        ("benchmark_imputation_applied", False),
    ):
        if column not in out.columns:
            out[column] = default
    if units is None or units.empty:
        return out

    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["offense"] = out["offense"].astype("string")
    work = units.copy()
    work["state_fips"] = work["state_fips"].astype("string").str.zfill(2)
    work["offense"] = work["offense"].astype("string")

    municipal = (
        work[work["unit_kind"].eq(MUNICIPAL_UNIT_KIND)]
        .groupby(["unit_id", "offense"], dropna=False, as_index=False)["imputed_count"]
        .sum()
        .rename(columns={"unit_id": "jurisdiction_id"})
    )
    county_pool = (
        work[work["unit_kind"].eq(COUNTY_UNIT_KIND)]
        .groupby(["state_fips", "offense"], dropna=False, as_index=False)["imputed_count"]
        .sum()
    )
    county_pool["jurisdiction_id"] = (
        county_pool["state_fips"].astype("string") + STATE_REMAINDER_SUFFIX
    )
    increments = pd.concat(
        [
            municipal[["jurisdiction_id", "offense", "imputed_count"]],
            county_pool[["jurisdiction_id", "offense", "imputed_count"]],
        ],
        ignore_index=True,
    )
    increments = increments.groupby(["jurisdiction_id", "offense"], dropna=False, as_index=False)[
        "imputed_count"
    ].sum()

    out["jurisdiction_id"] = out["jurisdiction_id"].astype("string")
    out = out.merge(increments, on=["jurisdiction_id", "offense"], how="left")
    added = pd.to_numeric(out["imputed_count"], errors="coerce").fillna(0.0)
    applied = added.gt(0.0)
    out["benchmark_imputed_count"] = (
        pd.to_numeric(out["benchmark_imputed_count"], errors="coerce").fillna(0.0) + added
    )
    out["benchmark_imputation_applied"] = (
        out["benchmark_imputation_applied"].astype("boolean").fillna(False).astype(bool) | applied
    )
    for column in ("adjusted_count_ags_core", "estimated_count_ags_core"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) + added
    if "adjustment_total" in out.columns:
        out["adjustment_total"] = (
            pd.to_numeric(out["adjustment_total"], errors="coerce").fillna(0.0) + added
        )
    if "estimate_source" in out.columns:
        out.loc[applied, "estimate_source"] = BENCHMARK_IMPUTATION_SOURCE
    if "estimate_confidence" in out.columns:
        out.loc[applied, "estimate_confidence"] = "low"
    if "estimated_from_panel" in out.columns:
        out.loc[applied, "estimated_from_panel"] = True
    if "needs_current_year_fill" in out.columns:
        out.loc[applied, "needs_current_year_fill"] = True

    # Mass conservation from the unit table into the controls. A unit whose control row
    # does not exist (a municipal jurisdiction present in geometry but absent from the
    # control panel) would silently drop its sub-target here, so the totals are
    # compared rather than assumed.
    expected = float(pd.to_numeric(work["imputed_count"], errors="coerce").fillna(0.0).sum())
    landed = float(added.sum())
    if abs(expected - landed) > 1e-6:
        missing = increments.merge(
            out[["jurisdiction_id", "offense"]].drop_duplicates(),
            on=["jurisdiction_id", "offense"],
            how="left",
            indicator=True,
        )
        missing = missing[missing["_merge"].eq("left_only")]
        raise ValueError(
            f"benchmark-imputed mass did not land on the controls: expected {expected:.6f}, "
            f"applied {landed:.6f}; {len(missing)} target row(s) absent from the control "
            f"panel, e.g. {missing.head(10).to_dict(orient='records')}"
        )
    return out.drop(columns=["imputed_count"], errors="ignore")


def county_remainder_imputed_targets(units: pd.DataFrame) -> pd.DataFrame:
    """Per-county imputed sub-targets, shaped for allocation's group-target table."""
    columns = ["state_fips", "offense", "group_id", "county_geoid", "imputed_target_count"]
    if units is None or units.empty:
        return pd.DataFrame(columns=columns)
    county = units[units["unit_kind"].eq(COUNTY_UNIT_KIND)].copy()
    if county.empty:
        return pd.DataFrame(columns=columns)
    county["state_fips"] = county["state_fips"].astype("string").str.zfill(2)
    county["offense"] = county["offense"].astype("string")
    county["group_id"] = county["unit_id"].astype("string")
    county["county_geoid"] = county["county_geoid"].astype("string")
    county["imputed_target_count"] = (
        pd.to_numeric(county["imputed_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    return (
        county.groupby(
            ["state_fips", "offense", "group_id", "county_geoid"], dropna=False, as_index=False
        )["imputed_target_count"]
        .sum()
    )


def benchmark_imputation_refusals_path(paths: RepoPaths, *, year: int) -> Path:
    return (
        paths.state_dir / "controls" / f"benchmark_imputation_refusals_{int(year)}.parquet"
    )


def write_benchmark_imputation_artifacts(
    imputation: BenchmarkImputation, *, paths: RepoPaths, year: int
) -> tuple[Path, Path, Path]:
    units_path = benchmark_imputation_units_path(paths, year=int(year))
    conflicts_path = benchmark_imputation_conflicts_path(paths, year=int(year))
    diagnostics_path = benchmark_imputation_diagnostics_path(paths, year=int(year))
    units_path.parent.mkdir(parents=True, exist_ok=True)
    # Refusals are a v2-only product: with the floor off there are none and the legacy
    # artifact set is untouched.
    if not imputation.refusals.empty:
        imputation.refusals.to_parquet(
            benchmark_imputation_refusals_path(paths, year=int(year)), index=False
        )
    imputation.units.to_parquet(units_path, index=False)
    imputation.state_identity.to_parquet(conflicts_path, index=False)
    diagnostics_path.write_text(json.dumps(imputation.diagnostics, indent=2, default=str))
    return units_path, conflicts_path, diagnostics_path
