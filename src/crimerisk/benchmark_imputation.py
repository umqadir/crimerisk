"""Benchmark-constrained imputation for silent-agency territory (Class A, v20).

The mechanism this exists to fix (diagnosed 2026-07-28, docs/STATE.md): jurisdiction
controls are built bottom-up from reporting agencies, so a fully silent agency
contributes nothing to any target and its territory's target is exactly zero -- 150 of
235 state x offense residual cells, 3.4M people of unincorporated county remainder plus
276k people inside municipal jurisdictions whose PD never reports. v19 correctly refused
to fabricate an agency-level anchor for those agencies; this module supplies the missing
mass at the only level where an external benchmark exists, and never as an agency
estimate.

The design (sol-reviewed, adjudicated; analysis_scratch/sol_v20_review.txt section 1):

1. ONE ACCOUNTING IDENTITY per state x offense.

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
    "benchmark_scale",
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
    "benchmark_scale",
    "imputed_total",
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
    enforce_mass_ratchet: bool = True


@dataclass(frozen=True)
class BenchmarkImputation:
    units: pd.DataFrame
    state_identity: pd.DataFrame
    validation: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)

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


def _fill_covered_oris(agency_estimates: pd.DataFrame) -> set[str]:
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
    return set(filled["ori9"].astype("string").dropna())


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
) -> set[str]:
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
    supported = set(
        agency_preferred.loc[mask, "ori9"].astype("string").str.upper().dropna()
    )
    reads_missing = set(
        directives.loc[
            directives["directive"].eq("reads_missing"), "ori9"
        ].astype("string").str.upper()
    )
    return supported - reads_missing


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

    is_remainder = ledger["jurisdiction_id"].str.endswith(STATE_REMAINDER_SUFFIX, na=False)
    county_geoid = ledger["state_fips"].astype("string") + ledger["county_fips"].astype("string")
    ledger["unit_kind"] = np.where(is_remainder, COUNTY_UNIT_KIND, MUNICIPAL_UNIT_KIND)
    ledger["unit_id"] = np.where(
        is_remainder,
        ledger["state_fips"].astype("string") + STATE_REMAINDER_SUFFIX + ":county:" + county_geoid,
        ledger["jurisdiction_id"].astype("string"),
    )
    ledger["county_geoid"] = np.where(is_remainder, county_geoid, pd.NA)

    ledger["is_supported"] = ledger["ori9"].isin(supported)
    ledger["is_fill_covered"] = ledger["ori9"].isin(fill_covered)
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
        f"{overlapping['ori9'].nunique()} agenc(ies) are both sized by the agency "
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
    duplicated = int(silent.duplicated(subset=["unit_id"]).sum())
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
    locked = pd.to_numeric(silent["locked_total"], errors="coerce").fillna(0.0)
    occupied = silent[locked.gt(1e-9)]
    if not occupied.empty:
        raise ValueError(
            f"{len(occupied)} eligible silent unit(s) already carry pre-imputation "
            "locked mass; a jurisdiction-level fill has locked a control before "
            "benchmark eligibility was evaluated: "
            + str(
                occupied[["unit_id", "unit_kind", "state_fips", "locked_total"]]
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
        ["ori9", "state_fips", "unit_id", "weight"]
    ]
    est = agency_estimates.copy()
    est["ori9"] = est["ori9"].astype("string")
    est["state_fips"] = est["state_fips"].astype("string").str.zfill(2)
    est["offense"] = est["offense"].astype("string")
    merged = est.merge(remainder, on=["ori9", "state_fips"], how="inner")
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


def fit_pooled_rates(panel: pd.DataFrame, *, offense: str, pooling_constant: float) -> dict:
    """Two-level Gamma-Poisson posterior means: cell shrunk to state shrunk to national.

    K is a pseudo-exposure: a cell with K person-years of its own data weighs its own
    rate and its parent's equally. One K per offense, chosen out of sample.
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
    return {
        "national": national,
        "state": {str(k): float(v) for k, v in state["rate"].items()},
        "cell": {
            tuple(str(part) for part in key): float(value)
            for key, value in zip(
                cell[RATE_CELL_COLUMNS].itertuples(index=False, name=None), cell["rate"], strict=True
            )
        },
    }


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


def _poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    mu = np.clip(expected, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(observed > 0.0, observed * np.log(observed / mu), 0.0)
    return float(2.0 * np.sum(term - (observed - mu)))


def select_pooling_constants(
    observed: pd.DataFrame, *, config: BenchmarkImputationConfig
) -> tuple[dict[str, float], pd.DataFrame]:
    """Masked validation: hide a fold of the observed units, predict them from the model
    fitted on the rest, keep the K with the lowest out-of-sample Poisson deviance."""
    rng = np.random.default_rng(int(config.validation_seed))
    folds = int(config.validation_folds)
    assignment = rng.integers(0, folds, size=len(observed))
    work = observed.assign(_fold=assignment)
    rows: list[dict[str, float]] = []
    for pooling_constant in config.pooling_constant_grid:
        record: dict[str, float] = {"pooling_constant": float(pooling_constant)}
        for offense in OFFENSES_7:
            record[offense] = 0.0
        for fold in range(folds):
            train = work[work["_fold"].ne(fold)]
            test = work[work["_fold"].eq(fold)]
            if train.empty or test.empty:
                continue
            exposure = pd.to_numeric(test["exposure_population"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for offense in OFFENSES_7:
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

    # Unit-level dispositions are read off the AGENCY ledger, never off control mass.
    # That ordering is the structural part of the fix: eligibility is decided before any
    # mass is looked at, so no jurisdiction-level number can lock a unit out of the
    # benchmark lane. Links at zero weight route no mass and therefore say nothing about
    # whether the territory is covered.
    weighted_links = ledger[pd.to_numeric(ledger["weight"], errors="coerce").fillna(0.0).gt(0.0)]
    unit_supported = (
        ledger.groupby("unit_id", dropna=False)["is_supported"].max().astype("boolean")
    )
    unit_fill_covered = (
        weighted_links.groupby("unit_id", dropna=False)["is_fill_covered"]
        .max()
        .astype("boolean")
    )
    unit_eligible_silent = (
        ledger.groupby("unit_id", dropna=False)["is_eligible_silent"].max().astype("boolean")
    )
    silent_oris = (
        ledger[ledger["is_eligible_silent"]]
        .groupby("unit_id", dropna=False)["ori9"]
        .apply(lambda values: "|".join(sorted(str(v) for v in values.dropna().unique())))
    )
    silent_counts = ledger[ledger["is_eligible_silent"]].groupby("unit_id", dropna=False)["ori9"].nunique()
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
    units["has_supported_agency"] = units["unit_id"].map(unit_supported).fillna(False).astype(bool)
    units["has_fill_covered_agency"] = (
        units["unit_id"].map(unit_fill_covered).fillna(False).astype(bool)
    )
    units["has_eligible_silent_agency"] = (
        units["unit_id"].map(unit_eligible_silent).fillna(False).astype(bool)
    )
    units["silent_agency_count"] = units["unit_id"].map(silent_counts).fillna(0).astype(int)
    units["silent_agency_oris"] = units["unit_id"].map(silent_oris).astype("string")

    # Territory the pipeline currently observes -- the fit sample for the rate model.
    observed = units[
        units["has_supported_agency"]
        & units["exposure_population"].gt(0.0)
        & units["locked_total"].gt(0.0)
    ].copy()
    # Territory whose every agency is silent, and whose silence is attributable to an
    # identified, active, primary agency -- the imputation units. Stated over the agency
    # ledger's own dispositions rather than over "the control row happens to be empty":
    # the empty control row is the CONSEQUENCE of every agency being silent, and asserted
    # below to be one, not the rule.
    silent_candidates = units[
        ~units["has_supported_agency"]
        & ~units["has_fill_covered_agency"]
        & units["has_eligible_silent_agency"]
        & units["exposure_population"].gt(0.0)
    ].copy()
    # Territory an agency estimate already sizes in part while another agency inside it
    # is silent. Not imputed: the exposure cannot be split between the two agencies, so
    # imputing the whole unit would double the covered part. Counted, not hidden.
    partially_covered = units[
        ~units["has_supported_agency"]
        & units["has_fill_covered_agency"]
        & units["has_eligible_silent_agency"]
        & units["exposure_population"].gt(0.0)
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

    pooling_constants, validation = select_pooling_constants(observed, config=config)
    rates_by_offense = {
        offense: fit_pooled_rates(
            observed, offense=offense, pooling_constant=pooling_constants[offense]
        )
        for offense in OFFENSES_7
    }

    eligible = silent[~silent["state_excluded"]].copy()
    long_rows: list[pd.DataFrame] = []
    for offense in OFFENSES_7:
        if eligible.empty:
            continue
        rate = predict_pooled_rate(eligible, rates_by_offense[offense])
        long_rows.append(
            pd.DataFrame(
                {
                    "state_fips": eligible["state_fips"].to_numpy(),
                    "state_abbr": eligible["state_abbr"].to_numpy(),
                    "unit_kind": eligible["unit_kind"].to_numpy(),
                    "unit_id": eligible["unit_id"].to_numpy(),
                    "county_geoid": eligible["county_geoid"].to_numpy(),
                    "exposure_population": eligible["exposure_population"].to_numpy(dtype=float),
                    "land_area_sq_mi": eligible["land_area_sq_mi"].to_numpy(dtype=float),
                    "urbanicity": eligible["urbanicity"].to_numpy(),
                    "silent_agency_count": eligible["silent_agency_count"].to_numpy(),
                    "silent_agency_oris": eligible["silent_agency_oris"].to_numpy(),
                    "offense": offense,
                    "pooled_rate": rate,
                    "modeled_expected_count": rate
                    * eligible["exposure_population"].to_numpy(dtype=float),
                }
            )
        )
    unit_long = (
        pd.concat(long_rows, ignore_index=True)
        if long_rows
        else pd.DataFrame(columns=UNIT_COLUMNS)
    )

    identity = _build_state_identity(
        controls=controls,
        cde_estimates=cde_estimates,
        unit_long=unit_long,
        silent=silent,
        year=year,
        excluded_states=excluded_states,
    )

    if not unit_long.empty:
        unit_long = unit_long.merge(
            identity[["state_fips", "offense", "benchmark_scale"]],
            on=["state_fips", "offense"],
            how="left",
        )
        unit_long["benchmark_scale"] = pd.to_numeric(
            unit_long["benchmark_scale"], errors="coerce"
        ).fillna(0.0)
        unit_long["imputed_count"] = (
            unit_long["modeled_expected_count"] * unit_long["benchmark_scale"]
        )
        unit_long["year"] = year
        unit_long["imputation_source"] = BENCHMARK_IMPUTATION_SOURCE
        unit_long = unit_long[unit_long["imputed_count"].gt(0.0)].copy()
    unit_long = unit_long.reindex(columns=UNIT_COLUMNS)

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
    )
    imputation = BenchmarkImputation(
        units=unit_long.sort_values(["state_fips", "unit_id", "offense"], kind="mergesort").reset_index(drop=True),
        state_identity=identity.reindex(columns=STATE_IDENTITY_COLUMNS)
        .sort_values(["state_fips", "offense"], kind="mergesort")
        .reset_index(drop=True),
        validation=validation,
        diagnostics=diagnostics,
    )
    assert_benchmark_imputation_invariants(imputation)
    if config.enforce_mass_ratchet:
        check_benchmark_mass_against_baseline(imputation, paths=paths)
    return imputation


def _build_state_identity(
    *,
    controls: pd.DataFrame,
    cde_estimates: pd.DataFrame,
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
    ).clip(lower=0.0)

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

    unit_stats = (
        silent.groupby("state_fips", dropna=False)
        .agg(silent_unit_count=("unit_id", "nunique"), silent_unit_population=("exposure_population", "sum"))
        .reset_index()
        if not silent.empty
        else pd.DataFrame(columns=["state_fips", "silent_unit_count", "silent_unit_population"])
    )
    identity = identity.merge(unit_stats, on="state_fips", how="left")
    identity["silent_unit_count"] = pd.to_numeric(identity["silent_unit_count"], errors="coerce").fillna(0).astype(int)
    identity["silent_unit_population"] = pd.to_numeric(
        identity["silent_unit_population"], errors="coerce"
    ).fillna(0.0)

    residual = identity["benchmark_residual"].to_numpy(dtype=float)
    modeled = identity["modeled_pool"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(modeled > 0.0, np.minimum(1.0, residual / np.where(modeled > 0.0, modeled, 1.0)), 0.0)
    identity["benchmark_scale"] = scale
    identity["imputed_total"] = modeled * scale
    identity["unused_benchmark_headroom"] = np.maximum(0.0, residual - identity["imputed_total"].to_numpy(dtype=float))
    identity["unfilled_modeled_pool"] = np.maximum(0.0, modeled - identity["imputed_total"].to_numpy(dtype=float))

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
            "headroom_exceeds_model_pool",
            "benchmark_capped_below_model_pool",
        ],
        default="reconciled",
    )
    identity["year"] = int(year)
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
    }


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
    """Fail closed on the three properties the design is accountable for."""
    units = imputation.units
    identity = imputation.state_identity

    # (3) The conflict path is never silent: a state x offense whose locked total
    # exceeds the benchmark must carry a conflict record and zero imputed mass. Checked
    # first because it is the most specific diagnosis of an over-imputation.
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
        if float(conflicted["imputed_total"].abs().max()) > 1e-9 or (
            not offending.empty
            and float(pd.to_numeric(offending["imputed_count"], errors="coerce").fillna(0.0).sum())
            > 1e-9
        ):
            raise ValueError(
                "benchmark-conflict state x offense cells carry imputed mass; the "
                "residual must be zero when locked observations exceed the benchmark"
            )

    if not units.empty:
        values = pd.to_numeric(units["imputed_count"], errors="coerce")
        bad = units[~np.isfinite(values) | values.lt(0.0)]
        if not bad.empty:
            raise ValueError(
                f"{len(bad)} benchmark-imputed target(s) are non-finite or negative: "
                + str(bad[["unit_id", "offense", "imputed_count"]].head(20).to_dict(orient="records"))
            )

        # (1) The accounting identity: imputed mass per state x offense never exceeds
        # the benchmark residual.
        per_cell = units.groupby(["state_fips", "offense"], dropna=False)["imputed_count"].sum()
        limits = identity.set_index(["state_fips", "offense"])["benchmark_residual"]
        overrun = (per_cell - limits.reindex(per_cell.index).fillna(0.0)).max()
        if pd.notna(overrun) and float(overrun) > 1e-6:
            raise ValueError(
                "benchmark-imputed mass exceeds the state x offense benchmark residual by "
                f"{float(overrun):.6f}"
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


def write_benchmark_imputation_artifacts(
    imputation: BenchmarkImputation, *, paths: RepoPaths, year: int
) -> tuple[Path, Path, Path]:
    units_path = benchmark_imputation_units_path(paths, year=int(year))
    conflicts_path = benchmark_imputation_conflicts_path(paths, year=int(year))
    diagnostics_path = benchmark_imputation_diagnostics_path(paths, year=int(year))
    units_path.parent.mkdir(parents=True, exist_ok=True)
    imputation.units.to_parquet(units_path, index=False)
    imputation.state_identity.to_parquet(conflicts_path, index=False)
    diagnostics_path.write_text(json.dumps(imputation.diagnostics, indent=2, default=str))
    return units_path, conflicts_path, diagnostics_path
