"""Surface 2 — temporally smoothed jurisdiction controls (current annual risk).

The product carries two control surfaces over one spine (`PLAN.md`, target architecture):

* Surface 1, the annual accounting control (`controls.py` ->
  `jurisdiction_controls_<year>.parquet`), is estimated *reported* offenses for one calendar
  year under exact conservation. It is never smoothed and this module never rebuilds it --
  it is opened read-only, which is the mechanism that keeps the frozen edition
  byte-identical.
* Surface 2, built here, is the level a jurisdiction-offense carries into a 12-month risk
  window. It borrows across the jurisdiction-year panel with offense-specific persistence
  and shrinks small series toward a peer rate, so a single year's reporting accident cannot
  move it the way it moves the accounting row.

The two disagree by construction and neither is a correction of the other. Surface 2 is
never raked to the accounting year.  Its offense totals are reconciled once to an independent
national anchor built from FBI CDE state estimates with the same E1 temporal kernel.  The
offense-wide factor cannot alter any within-offense relative index or jurisdiction boundary;
it restores absolute-count scale and the offense mix used by composites.

Estimator provenance is the E1 rolling-origin tournament
(`analysis_scratch/final_phase/e1_tournament/`): round 1 selected the per-offense temporal
kernel, round 2 added hierarchical shrinkage over it and beat the round-1 winner on every
offense, round 3's trend multiplier was rejected and is deliberately absent here. The
contract for this artifact is `analysis_scratch/final_phase/SURFACE2_CONTROLS_CONTRACT.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.build_freshness import artifact_is_current, write_dependency_stamp
from crimerisk.controls import (
    CDE_OFFENSE_MAP,
    CONTROL_KEY_COLUMNS,
    _cde_estimates_path,
)
from crimerisk.level_lane import level_lane_policy
from crimerisk.paths import RepoPaths
from crimerisk.scope import PRODUCTION_SCOPE_EXCLUDE
from crimerisk.stage_locks import blockers_for_stage, stage_write_lock


# The volume-offense kernels remain the E1 tournament winners. Murder was re-estimated under
# an information-keyed exposure posterior (murder_unified_estimator/01_control_estimator_cv.py):
# a one-year half-life and 10,000 prior person-years passed the untouched 2023-2024 gate.
E1_HALFLIFE_YEARS: dict[str, float | None] = {
    "murder": 1.0,
    "rape": 2.0,
    "robbery": 2.0,
    "aggravated_assault": 2.0,
    "burglary": 1.0,
    "larceny": 1.0,
    "motor_vehicle_theft": 1.0,
}

# E1 round 2 (`round2_summary.md`): k selected on the 2021 fold ONLY, over the grid
# [0.5, 2, 5, 10, 25, 50, 100]; every offense chose 0.5. Reported scores are the 2022-2024
# folds, so selection never touched the reported test years.
E1_SHRINKAGE_K = 0.5
MURDER_INFORMATION_PRIOR_EXPOSURE = 10_000.0

# A unit with one clean year can receive the E1 estimate, but it cannot define the
# production peer baseline.  This separates recipient eligibility from reference
# eligibility: newly introduced source-publication units borrow from established local
# series without moving the estimates of those established series themselves.
PEER_REFERENCE_MIN_CLEAN_YEARS = 2

DYNAMIC_SHRINK_ESTIMATOR = "dynamic_shrink"
INFORMATION_POSTERIOR_ESTIMATOR = "information_posterior"
FALLBACK_ACCOUNTING_ESTIMATOR = "fallback_accounting"
COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR = "coverage_adjusted_remainder"
NATIONAL_ANCHOR_SOURCE = "fbi_cde_state_estimates_e1_kernel"
IDENTITY_ANCHOR_SOURCE = "identity_no_external_anchor"

EWA_KERNEL = "ewa"
FLAT_MEAN_KERNEL = "flat_mean"

INSUFFICIENT_HISTORY_REASON = "insufficient_clean_history"
NO_EXPOSURE_REASON = "no_exposure"
NO_PEER_SUPPORT_REASON = "no_peer_support"
SOURCE_REPLACEMENT_PROTECTED_REASON = "source_supported_replacement_protected"
STRUCTURAL_ZERO_PROTECTED_REASON = "locked_structural_zero_protected"

SOURCE_REPLACEMENT_REPAIR_MODE = "source_supported_replacement"
STRUCTURAL_ZERO_REPAIR_MODE = "locked_structural_zero"

ACCOUNTING_COUNT_COLUMN = "adjusted_count_ags_core"
STATE_REMAINDER_TYPE = "state_nonmunicipal_remainder"
REMAINDER_PEER_MIN_POPULATION = 100_000.0
# A single clean year is enough to estimate an ordinary local unit with hierarchical
# shrinkage, matching E1.  It is not enough to define the cross-state reference ratio
# used to repair a different state's undercovered pooled remainder.  Keep that reference
# pool on the independently stricter support rule so the local eligibility repair cannot
# cascade into unrelated state controls.
REMAINDER_MIN_CLEAN_YEARS = 2
REMAINDER_MUNICIPAL_PEER_POP_BANDS = (
    "<5k",
    "5k-10k",
    "10k-25k",
    "25k-50k",
)

# Diagnostics that describe THE ESTIMATE. They are null on a fallback row, because no
# estimate was produced -- publishing an offense-level constant there would imply one was.
ESTIMATE_DIAGNOSTIC_COLUMNS = [
    "temporal_kernel",
    "halflife_used",
    "shrinkage_k",
    "ewa_weight",
    "ewa_count",
    "information_weighted_count",
    "information_weighted_exposure",
    "prior_exposure",
    "peer_rate",
    "peer_predicted_count",
]

# A pooled state remainder is not a normal jurisdiction. In states where its current
# reporting coverage is incomplete, the accounting residual can be close to zero even
# though the row owns millions of residents. Surface 1 must retain that accounting answer;
# Surface 2 floors it with a reporting-coverage blend toward the rate observed in
# well-supported state remainders. These fields make that repair fully recomposable.
# (e) The lift a coverage shortfall can justify is bounded by the shortfall itself.
# A remainder whose agencies were already filled upstream has no coverage left to
# repair, and lifting it again pays for the same missing agencies twice. The floor
# keeps a lane with almost no covered agency from implying an unbounded gross-up.
COVERAGE_EVIDENCE_FLOOR = 0.05

COVERAGE_REMAINDER_DIAGNOSTIC_COLUMNS = [
    "reporting_coverage_weight",
    "remainder_to_municipal_rate_ratio",
    "coverage_municipal_reference_population",
    "coverage_municipal_reference_unit_count",
    "coverage_training_remainder_population",
    "coverage_training_remainder_unit_count",
    "coverage_training_municipal_population",
    "coverage_training_municipal_unit_count",
    "coverage_peer_predicted_count",
    "coverage_blended_count",
    "coverage_evidence_share",
    "coverage_evidence_capped_count",
    "coverage_rate_band_ceiling_count",
]

NATIONAL_CALIBRATION_COLUMNS = [
    "precalibration_smoothed_count",
    "national_anchor_count",
    "national_calibration_factor",
    "national_anchor_source",
]

# Descriptors of THE UNIT. Well defined whether or not the estimator ran.
UNIT_DIAGNOSTIC_COLUMNS = [
    "peer_group_id",
    "clean_year_count",
    "latest_clean_year",
]

SMOOTHED_CONTROL_COLUMNS = [
    *CONTROL_KEY_COLUMNS,
    "year",
    "bucket_population",
    "pop_band",
    "accounting_count",
    "smoothed_count",
    "estimator",
    "fallback_reason",
    *NATIONAL_CALIBRATION_COLUMNS,
    *ESTIMATE_DIAGNOSTIC_COLUMNS,
    *COVERAGE_REMAINDER_DIAGNOSTIC_COLUMNS,
    *UNIT_DIAGNOSTIC_COLUMNS,
]

PANEL_CLEAN_ROW_COLUMNS = [
    "jurisdiction_id",
    "offense",
    "year",
    "reported_count_preferred",
    "mean_months_reported_preferred",
    "observation_weight_preferred",
    "fill_component_count",
    "partial_component_count",
    "bucket_population",
]


@dataclass(frozen=True)
class SmoothedControlConfig:
    """Every knob is an E1 output or a structural minimum -- none is fitted to the result."""

    year: int = 2024
    panel_year_start: int = 2018
    # E1's rolling-origin tournament admitted any series with at least one clean
    # historical observation.  A one-year series still has a defined own-series level;
    # the hierarchical peer term is what keeps that single observation from becoming an
    # exact small-area risk estimate.  No recency, size, or outcome-conditioned screen
    # sits on top of it.
    min_clean_years: int = 1
    shrinkage_k: float = E1_SHRINKAGE_K
    halflife_years: tuple[tuple[str, float | None], ...] = tuple(E1_HALFLIFE_YEARS.items())
    # Width of the reported sanity band, ending at `year`.
    sanity_band_years: int = 4
    exclude_scope_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE))

    def halflife_map(self) -> dict[str, float | None]:
        return {offense: halflife for offense, halflife in self.halflife_years}

    def band_years(self) -> tuple[int, ...]:
        return tuple(range(int(self.year) - int(self.sanity_band_years) + 1, int(self.year) + 1))


# --- paths and freshness ----------------------------------------------------


def smoothed_controls_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "controls" / f"jurisdiction_controls_smoothed_{int(year)}.parquet"


def smoothed_controls_summary_path(paths: RepoPaths, *, year: int) -> Path:
    return smoothed_controls_path(paths, year=year).with_suffix(".summary.json")


def accounting_controls_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "controls" / f"jurisdiction_controls_{int(year)}.parquet"


def jurisdiction_year_estimates_path(paths: RepoPaths) -> Path:
    return paths.state_dir / "controls" / "jurisdiction_year_estimates.parquet"


def smoothed_controls_dependency_paths(paths: RepoPaths, *, year: int) -> list[Path]:
    return [
        jurisdiction_year_estimates_path(paths),
        accounting_controls_path(paths, year=year),
        _cde_estimates_path(paths, year=int(year)),
        Path(__file__),
    ]


def smoothed_controls_artifact_is_current(
    paths: RepoPaths, *, year: int, out_path: Path | None = None
) -> bool:
    artifact = out_path or smoothed_controls_path(paths, year=year)
    if not smoothed_controls_summary_path(paths, year=year).exists():
        return False
    return artifact_is_current(artifact, smoothed_controls_dependency_paths(paths, year=year))


# --- the estimator ----------------------------------------------------------


def clean_panel_history(panel: pd.DataFrame, *, config: SmoothedControlConfig) -> pd.DataFrame:
    """The usable history rows, on E1's clean-row definition.

    A year enters a unit's window only when the jurisdiction filed a full twelve months at
    full observation weight with no fill and no partial component. That restriction is also
    what makes the smoothed level a coverage-complete quantity without any explicit coverage
    adjustment: every year averaged is a complete filing.
    """
    missing = sorted(set(PANEL_CLEAN_ROW_COLUMNS) - set(panel.columns))
    if missing:
        raise ValueError(f"jurisdiction-year panel is missing columns {missing}")
    work = panel.loc[:, PANEL_CLEAN_ROW_COLUMNS].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    in_window = work["year"].ge(int(config.panel_year_start)) & work["year"].le(int(config.year))
    clean = (
        in_window.fillna(False).astype(bool)
        & work["reported_count_preferred"].notna()
        & pd.to_numeric(work["mean_months_reported_preferred"], errors="coerce").ge(12.0)
        & pd.to_numeric(work["observation_weight_preferred"], errors="coerce").ge(1.0)
        & pd.to_numeric(work["fill_component_count"], errors="coerce").fillna(0.0).eq(0.0)
        & pd.to_numeric(work["partial_component_count"], errors="coerce").fillna(0.0).eq(0.0)
    ).astype(bool)
    history = work.loc[
        clean,
        ["jurisdiction_id", "offense", "year", "bucket_population"],
    ].copy()
    history["year"] = history["year"].astype(int)
    history["clean_count"] = (
        pd.to_numeric(work.loc[clean, "reported_count_preferred"], errors="coerce")
        .astype(float)
        .clip(lower=0.0)
    )
    history["clean_exposure"] = (
        pd.to_numeric(history["bucket_population"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    history = history.drop(columns="bucket_population")
    return history.reset_index(drop=True)


def kernel_weights(ages: pd.Series, halflife: pd.Series) -> pd.Series:
    """`0.5 ** (age / halflife)`, and a flat 1.0 wherever the kernel is the flat mean.

    Weights are normalised by their own sum downstream, so a flat 1.0 reproduces E1's
    `hist.mean(axis=1)` exactly.
    """
    age = pd.to_numeric(ages, errors="coerce").astype(float)
    life = pd.to_numeric(halflife, errors="coerce").astype(float)
    weight = np.where(life.isna(), 1.0, np.power(0.5, age / life.where(life.notna(), 1.0)))
    return pd.Series(weight, index=age.index, dtype=float)


def national_e1_anchor_from_cde_history(
    cde_history: pd.DataFrame,
    *,
    config: SmoothedControlConfig,
) -> pd.DataFrame:
    """One independent CONUS+DC offense total under the selected E1 kernel."""
    required = {"offense", "year", "cde_total"}
    missing = sorted(required - set(cde_history.columns))
    if missing:
        raise ValueError(f"CDE history is missing columns {missing}")
    history = cde_history.copy()
    history["year"] = pd.to_numeric(history["year"], errors="coerce").astype("Int64")
    history["cde_total"] = pd.to_numeric(history["cde_total"], errors="coerce")
    history = history[
        history["year"].ge(int(config.panel_year_start))
        & history["year"].le(int(config.year))
    ].copy()
    expected_years = set(range(int(config.panel_year_start), int(config.year) + 1))
    rows: list[dict[str, float | str]] = []
    for offense, halflife in config.halflife_years:
        part = history[history["offense"].astype(str).eq(str(offense))].sort_values("year")
        actual_years = set(part["year"].dropna().astype(int))
        if actual_years != expected_years:
            raise ValueError(
                f"CDE national anchor for {offense} has years {sorted(actual_years)}, "
                f"expected {sorted(expected_years)}"
            )
        if part["cde_total"].isna().any() or part["cde_total"].lt(0.0).any():
            raise ValueError(f"CDE national anchor for {offense} has invalid totals")
        ages = int(config.year) - part["year"].astype(int)
        lives = pd.Series(
            np.nan if halflife is None else float(halflife),
            index=part.index,
            dtype=float,
        )
        weights = kernel_weights(ages, lives)
        anchor = float(np.average(part["cde_total"].to_numpy(dtype=float), weights=weights))
        rows.append({"offense": str(offense), "national_anchor_count": anchor})
    return pd.DataFrame(rows)


def _apply_national_anchor_calibration(
    out: pd.DataFrame,
    *,
    national_anchor: pd.DataFrame | None,
    source: str,
) -> pd.DataFrame:
    """Reconcile each offense to one ex-ante national anchor with a uniform factor."""
    work = out.copy()
    work["precalibration_smoothed_count"] = pd.to_numeric(
        work["smoothed_count"], errors="coerce"
    ).astype(float)
    totals = (
        work.groupby("offense", dropna=False, as_index=False)[
            "precalibration_smoothed_count"
        ]
        .sum()
        .rename(columns={"precalibration_smoothed_count": "precalibration_total"})
    )
    if national_anchor is None:
        anchor = totals[["offense", "precalibration_total"]].rename(
            columns={"precalibration_total": "national_anchor_count"}
        )
        source = IDENTITY_ANCHOR_SOURCE
    else:
        required = {"offense", "national_anchor_count"}
        missing = sorted(required - set(national_anchor.columns))
        if missing:
            raise ValueError(f"national anchor is missing columns {missing}")
        anchor = national_anchor[["offense", "national_anchor_count"]].copy()
        if anchor.duplicated("offense").any():
            raise ValueError("national anchor has duplicate offenses")
        expected = set(work["offense"].astype(str))
        actual = set(anchor["offense"].astype(str))
        if actual != expected:
            raise ValueError(
                "national anchor offense universe differs from controls: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    anchor["offense"] = anchor["offense"].astype("string")
    anchor["national_anchor_count"] = pd.to_numeric(
        anchor["national_anchor_count"], errors="coerce"
    )
    calibration = totals.merge(anchor, on="offense", how="left", validate="one_to_one")
    invalid = (
        calibration["national_anchor_count"].isna()
        | calibration["national_anchor_count"].lt(0.0)
        | calibration["precalibration_total"].le(0.0)
    )
    if invalid.any():
        raise ValueError(
            "national anchor calibration has invalid totals: "
            f"{calibration.loc[invalid].to_dict(orient='records')}"
        )
    calibration["national_calibration_factor"] = (
        calibration["national_anchor_count"] / calibration["precalibration_total"]
    )
    work = work.merge(
        calibration[
            ["offense", "national_anchor_count", "national_calibration_factor"]
        ],
        on="offense",
        how="left",
        validate="many_to_one",
    )
    work["national_anchor_source"] = str(source)
    work["smoothed_count"] = (
        work["precalibration_smoothed_count"]
        * work["national_calibration_factor"]
    )
    return work


def series_kernel_levels(
    history: pd.DataFrame, *, config: SmoothedControlConfig
) -> pd.DataFrame:
    """Per unit: the kernel's own-series level, its window size, and its newest year."""
    halflife_map = config.halflife_map()
    work = history.copy()
    unknown = sorted(set(work["offense"].astype(str)) - set(halflife_map))
    if unknown:
        raise ValueError(f"no E1 temporal kernel configured for offenses {unknown}")
    work["_halflife"] = work["offense"].astype(str).map(halflife_map).astype(float)
    work["_weight"] = kernel_weights(int(config.year) - work["year"], work["_halflife"])
    work["_weighted_count"] = work["_weight"] * work["clean_count"]
    work["_weighted_exposure"] = work["_weight"] * work["clean_exposure"]
    grouped = work.groupby(["jurisdiction_id", "offense"], dropna=False, as_index=False).agg(
        _weight_sum=("_weight", "sum"),
        _weighted_count_sum=("_weighted_count", "sum"),
        information_weighted_exposure=("_weighted_exposure", "sum"),
        clean_year_count=("year", "size"),
        latest_clean_year=("year", "max"),
    )
    grouped["ewa_count"] = np.where(
        grouped["_weight_sum"] > 0.0,
        grouped["_weighted_count_sum"] / grouped["_weight_sum"].replace(0.0, np.nan),
        np.nan,
    )
    grouped["information_weighted_count"] = grouped["_weighted_count_sum"]
    return grouped[
        [
            "jurisdiction_id",
            "offense",
            "ewa_count",
            "information_weighted_count",
            "information_weighted_exposure",
            "clean_year_count",
            "latest_clean_year",
        ]
    ]


def _peer_rates(eligible: pd.DataFrame) -> pd.DataFrame:
    """Exposure-weighted peer EWA rate per (offense, state, unit class, population band).

    A unit is a member of its own peer group, as in E1: a group of one shrinks toward
    itself, which is a no-op rather than a fabricated neighbour.

    Municipal patrol controls and pooled state remainders are different estimands.  A
    tiny remainder must never define the peer rate for ordinary towns in the same
    population band (the concrete failure was Connecticut's 1,486-person residual).
    """
    grouped = eligible.groupby(
        ["offense", "state_fips", "jurisdiction_type", "pop_band"],
        dropna=False,
        as_index=False,
    ).agg(_peer_ewa_sum=("ewa_count", "sum"), _peer_pop_sum=("bucket_population", "sum"))
    grouped["peer_rate"] = np.where(
        grouped["_peer_pop_sum"] > 0.0,
        grouped["_peer_ewa_sum"] / grouped["_peer_pop_sum"].replace(0.0, np.nan),
        np.nan,
    )
    return grouped[
        ["offense", "state_fips", "jurisdiction_type", "pop_band", "peer_rate"]
    ]


def _murder_leave_one_out_peer_rates(
    recipients: pd.DataFrame,
    *,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Peer rates for the murder information posterior, excluding the recipient itself."""
    keys = [
        "jurisdiction_id",
        "offense",
        "state_fips",
        "jurisdiction_type",
        "pop_band",
    ]
    out = recipients.loc[:, keys].copy().reset_index(drop=True)
    out["_row_id"] = np.arange(len(out), dtype=np.int64)
    self_mass = reference[
        [
            "jurisdiction_id",
            "offense",
            "information_weighted_count",
            "information_weighted_exposure",
        ]
    ].rename(
        columns={
            "information_weighted_count": "_self_count",
            "information_weighted_exposure": "_self_exposure",
        }
    )
    out = out.merge(self_mass, on=["jurisdiction_id", "offense"], how="left", validate="one_to_one")
    out[["_self_count", "_self_exposure"]] = out[["_self_count", "_self_exposure"]].fillna(0.0)
    out["peer_rate"] = np.nan

    ladders = (
        ["offense", "state_fips", "jurisdiction_type", "pop_band"],
        ["offense", "state_fips", "jurisdiction_type"],
        ["offense", "jurisdiction_type", "pop_band"],
        ["offense", "jurisdiction_type"],
    )
    for group_keys in ladders:
        totals = reference.groupby(group_keys, as_index=False, dropna=False).agg(
            _group_count=("information_weighted_count", "sum"),
            _group_exposure=("information_weighted_exposure", "sum"),
        )
        joined = out[["_row_id", *group_keys]].merge(
            totals,
            on=group_keys,
            how="left",
            validate="many_to_one",
        ).sort_values("_row_id", kind="mergesort")
        numerator = pd.to_numeric(joined["_group_count"], errors="coerce").fillna(0.0) - out["_self_count"]
        denominator = pd.to_numeric(joined["_group_exposure"], errors="coerce").fillna(0.0) - out["_self_exposure"]
        candidate = numerator / denominator.where(denominator.gt(0.0))
        take = out["peer_rate"].isna() & candidate.notna() & candidate.ge(0.0)
        out.loc[take, "peer_rate"] = candidate.loc[take]
    return out[["jurisdiction_id", "offense", "peer_rate"]]


def _apply_state_remainder_coverage_floor(
    out: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    config: SmoothedControlConfig,
    accounting_controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Repair undercovered pooled state remainders without changing Surface 1.

    The comparison rate is learned only from state remainders for which the ordinary
    multi-year estimator ran. For each offense it is the pooled remainder rate divided by
    the pooled rate of municipalities below 50,000 residents in those same states. The
    small-jurisdiction reference avoids transferring large-city composition into rural
    nonmunicipal territory. A fallback remainder is blended toward that state-specific
    prediction by its missing current-year reporting share. The result is a floor:
    observed/accounting mass is never removed.
    """
    work = out.copy()
    for column in COVERAGE_REMAINDER_DIAGNOSTIC_COLUMNS:
        work[column] = np.nan

    municipal = (
        work[
            work["jurisdiction_type"].astype("string").eq("municipal")
            & work["pop_band"].astype("string").isin(REMAINDER_MUNICIPAL_PEER_POP_BANDS)
        ]
        .groupby(["state_fips", "offense"], dropna=False, as_index=False)
        .agg(
            municipal_smoothed_count=("smoothed_count", "sum"),
            municipal_population=("bucket_population", "sum"),
            municipal_unit_count=("jurisdiction_id", "nunique"),
        )
    )
    municipal["municipal_rate"] = np.where(
        municipal["municipal_population"].gt(0.0),
        municipal["municipal_smoothed_count"]
        / municipal["municipal_population"].replace(0.0, np.nan),
        np.nan,
    )

    remainder = work[
        work["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
    ][
        [
            "jurisdiction_id",
            "state_fips",
            "offense",
            "bucket_population",
            "smoothed_count",
            "estimator",
            "fallback_reason",
            "clean_year_count",
        ]
    ].merge(municipal, on=["state_fips", "offense"], how="left", validate="many_to_one")
    if remainder.empty:
        return work

    reliable = remainder[
        remainder["estimator"].astype("string").isin(
            {DYNAMIC_SHRINK_ESTIMATOR, INFORMATION_POSTERIOR_ESTIMATOR}
        )
        & pd.to_numeric(remainder["clean_year_count"], errors="coerce").ge(
            REMAINDER_MIN_CLEAN_YEARS
        )
        & pd.to_numeric(remainder["bucket_population"], errors="coerce").ge(
            REMAINDER_PEER_MIN_POPULATION
        )
        & pd.to_numeric(remainder["municipal_population"], errors="coerce").gt(0.0)
    ].copy()
    if reliable.empty:
        return work

    peer = (
        reliable.groupby("offense", dropna=False, as_index=False)
        .agg(
            reliable_remainder_count=("smoothed_count", "sum"),
            coverage_training_remainder_population=("bucket_population", "sum"),
            coverage_training_remainder_unit_count=("jurisdiction_id", "nunique"),
            reliable_municipal_count=("municipal_smoothed_count", "sum"),
            coverage_training_municipal_population=("municipal_population", "sum"),
            coverage_training_municipal_unit_count=("municipal_unit_count", "sum"),
        )
    )
    peer["remainder_rate"] = (
        peer["reliable_remainder_count"]
        / peer["coverage_training_remainder_population"].replace(0.0, np.nan)
    )
    peer["municipal_rate"] = (
        peer["reliable_municipal_count"]
        / peer["coverage_training_municipal_population"].replace(0.0, np.nan)
    )
    peer["remainder_to_municipal_rate_ratio"] = (
        peer["remainder_rate"] / peer["municipal_rate"].replace(0.0, np.nan)
    )
    remainder = remainder.merge(
        peer[
            [
                "offense",
                "remainder_to_municipal_rate_ratio",
                "coverage_training_remainder_population",
                "coverage_training_remainder_unit_count",
                "coverage_training_municipal_population",
                "coverage_training_municipal_unit_count",
            ]
        ],
        on="offense",
        how="left",
        validate="many_to_one",
    )

    current_coverage = panel[
        pd.to_numeric(panel["year"], errors="coerce").eq(int(config.year))
    ][["jurisdiction_id", "offense", "observation_weight_preferred"]].copy()
    if current_coverage.duplicated(["jurisdiction_id", "offense"]).any():
        raise ValueError(
            "jurisdiction-year panel has duplicate current-year jurisdiction/offense rows"
        )
    remainder = remainder.merge(
        current_coverage,
        on=["jurisdiction_id", "offense"],
        how="left",
        validate="one_to_one",
    )
    remainder["reporting_coverage_weight"] = (
        pd.to_numeric(remainder["observation_weight_preferred"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    remainder["coverage_municipal_reference_population"] = pd.to_numeric(
        remainder["municipal_population"], errors="coerce"
    )
    remainder["coverage_municipal_reference_unit_count"] = pd.to_numeric(
        remainder["municipal_unit_count"], errors="coerce"
    )
    remainder["coverage_peer_predicted_count"] = (
        pd.to_numeric(remainder["bucket_population"], errors="coerce").fillna(0.0)
        * pd.to_numeric(remainder["municipal_rate"], errors="coerce")
        * pd.to_numeric(
            remainder["remainder_to_municipal_rate_ratio"], errors="coerce"
        )
    )
    accounting = pd.to_numeric(remainder["smoothed_count"], errors="coerce")
    coverage = remainder["reporting_coverage_weight"]
    partial_history_blend = (
        coverage * accounting
        + (1.0 - coverage) * remainder["coverage_peer_predicted_count"]
    )
    # With no complete year in the panel, the accounting residual is an observed
    # lower bound for an incompletely covered statewide pool, not a competing
    # full-coverage estimate. Letting its partial-reporting weight pull down the
    # peer prediction reproduces the missing-agency artifact (most visibly in PA).
    # A row with one complete year still gets the conservative coverage blend.
    no_complete_year = pd.to_numeric(
        remainder["clean_year_count"], errors="coerce"
    ).fillna(0.0).eq(0.0)
    remainder["coverage_blended_count"] = np.where(
        no_complete_year,
        remainder["coverage_peer_predicted_count"],
        partial_history_blend,
    )
    # (e) Two bounds on the lift, both read off the lane rather than assumed.
    #
    # The first is the lane's own coverage evidence. A state remainder whose agencies
    # were already estimated by the agency ladder is not an under-covered observation
    # waiting to be grossed up -- the missing agencies have been paid for once already,
    # and paying again is the Owsley seam. The gross-up is therefore capped at the
    # reciprocal of the share of the lane's crosswalked agencies that either reported
    # or carry an agency-level estimate.
    #
    # The second is the ratio the estimator itself records. `remainder_to_municipal_
    # rate_ratio` is the rate relationship the reliable remainders actually exhibit;
    # it has been written to the artifact all along and never enforced. A lifted lane
    # may not imply a remainder rate above its own state's municipal rate times that
    # ratio.
    policy_switches = level_lane_policy()
    evidence_share = pd.Series(np.nan, index=remainder.index, dtype=float)
    if accounting_controls is not None and not accounting_controls.empty:
        columns = [
            column
            for column in (
                "jurisdiction_id",
                "offense",
                "crosswalk_agency_count",
                "contributing_agency_count",
                "estimating_agency_count",
                "observed_component_count",
                "partial_component_count",
                "adjusted_count_ags_core",
            )
            if column in accounting_controls.columns
        ]
        if {"jurisdiction_id", "offense", "crosswalk_agency_count"}.issubset(columns):
            evidence = accounting_controls[columns].drop_duplicates(
                ["jurisdiction_id", "offense"]
            )
            remainder = remainder.merge(
                evidence, on=["jurisdiction_id", "offense"], how="left"
            )
            crosswalked = pd.to_numeric(
                remainder["crosswalk_agency_count"], errors="coerce"
            )
            covered = pd.concat(
                [
                    pd.to_numeric(
                        remainder.get(
                            "contributing_agency_count",
                            pd.Series(0.0, index=remainder.index),
                        ),
                        errors="coerce",
                    ),
                    pd.to_numeric(
                        remainder.get(
                            "estimating_agency_count",
                            pd.Series(0.0, index=remainder.index),
                        ),
                        errors="coerce",
                    ),
                ],
                axis=1,
            ).max(axis=1)
            agency_share = (covered / crosswalked.where(crosswalked.gt(0.0))).clip(
                lower=0.0, upper=1.0
            )
            # The lift acts on mass, so the share of the lane's accounting mass that
            # is grounded in a report rather than filled is the more direct measure.
            grounded = pd.to_numeric(
                remainder.get(
                    "observed_component_count", pd.Series(np.nan, index=remainder.index)
                ),
                errors="coerce",
            ).fillna(0.0) + pd.to_numeric(
                remainder.get(
                    "partial_component_count", pd.Series(np.nan, index=remainder.index)
                ),
                errors="coerce",
            ).fillna(0.0)
            lane_total = pd.to_numeric(
                remainder.get(
                    "adjusted_count_ags_core", pd.Series(np.nan, index=remainder.index)
                ),
                errors="coerce",
            )
            mass_share = (grounded / lane_total.where(lane_total.gt(0.0))).clip(
                lower=0.0, upper=1.0
            )
            # Whichever measure says the lane is better covered wins: an up-fill has to
            # justify itself against the best evidence the lane has, not the weakest.
            evidence_share = pd.concat([agency_share, mass_share], axis=1).max(axis=1)
    remainder["coverage_evidence_share"] = evidence_share
    remainder["coverage_evidence_capped_count"] = accounting * (
        1.0 / evidence_share.clip(lower=COVERAGE_EVIDENCE_FLOOR)
    )
    remainder["coverage_rate_band_ceiling_count"] = (
        pd.to_numeric(remainder["bucket_population"], errors="coerce")
        * pd.to_numeric(remainder["municipal_rate"], errors="coerce")
        * pd.to_numeric(remainder["remainder_to_municipal_rate_ratio"], errors="coerce")
    )
    candidate = np.maximum(accounting, remainder["coverage_blended_count"])
    if policy_switches.remainder_coverage_cap:
        evidence_cap = pd.to_numeric(
            remainder["coverage_evidence_capped_count"], errors="coerce"
        )
        band_cap = pd.to_numeric(
            remainder["coverage_rate_band_ceiling_count"], errors="coerce"
        )
        ceiling = pd.concat([evidence_cap, band_cap], axis=1).min(axis=1)
        candidate = np.maximum(
            accounting, pd.Series(candidate, index=remainder.index).where(
                ceiling.isna(), np.minimum(candidate, ceiling.fillna(np.inf))
            )
        )
    apply = (
        remainder["estimator"].astype("string").eq(FALLBACK_ACCOUNTING_ESTIMATOR)
        & remainder["fallback_reason"].astype("string").isin(
            {INSUFFICIENT_HISTORY_REASON, NO_EXPOSURE_REASON, NO_PEER_SUPPORT_REASON}
        )
        & pd.to_numeric(remainder["bucket_population"], errors="coerce").gt(0.0)
        & np.isfinite(candidate)
        & candidate.gt(accounting + 1e-9)
    )
    if not bool(apply.any()):
        return work

    repaired = remainder.loc[
        apply,
        [
            "jurisdiction_id",
            "offense",
            "reporting_coverage_weight",
            "remainder_to_municipal_rate_ratio",
            "coverage_municipal_reference_population",
            "coverage_municipal_reference_unit_count",
            "coverage_training_remainder_population",
            "coverage_training_remainder_unit_count",
            "coverage_training_municipal_population",
            "coverage_training_municipal_unit_count",
            "coverage_peer_predicted_count",
            "coverage_blended_count",
            "coverage_evidence_share",
            "coverage_evidence_capped_count",
            "coverage_rate_band_ceiling_count",
        ],
    ].copy()
    repaired["smoothed_count"] = candidate.loc[apply].to_numpy(dtype=float)
    repaired_index = repaired.set_index(["jurisdiction_id", "offense"])
    work_index = pd.MultiIndex.from_frame(work[["jurisdiction_id", "offense"]])
    matched = work_index.isin(repaired_index.index)
    lookup = repaired_index.reindex(work_index[matched])
    work.loc[matched, "smoothed_count"] = lookup["smoothed_count"].to_numpy(dtype=float)
    work.loc[matched, "estimator"] = COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR
    work.loc[matched, "fallback_reason"] = pd.NA
    for column in COVERAGE_REMAINDER_DIAGNOSTIC_COLUMNS:
        work.loc[matched, column] = lookup[column].to_numpy(dtype=float)
    return work


def build_smoothed_controls(
    *,
    accounting: pd.DataFrame,
    panel: pd.DataFrame,
    config: SmoothedControlConfig = SmoothedControlConfig(),
    national_anchor: pd.DataFrame | None = None,
    national_anchor_source: str = NATIONAL_ANCHOR_SOURCE,
) -> pd.DataFrame:
    """Surface 2 for every Surface-1 control unit: a validated estimator or coded fallback.

    Pure over frames -- no repo paths, no IO -- so the estimator is testable against a
    hand-computed case.
    """
    carried = [*CONTROL_KEY_COLUMNS, "bucket_population", "pop_band", ACCOUNTING_COUNT_COLUMN]
    missing = [column for column in carried if column not in accounting.columns]
    if missing:
        raise ValueError(f"accounting controls are missing columns {missing}")
    if accounting.duplicated(["jurisdiction_id", "offense"]).any():
        raise ValueError("accounting controls carry duplicate (jurisdiction_id, offense) keys")

    out = accounting.loc[:, carried].copy().reset_index(drop=True)
    repair_mode = accounting.get(
        "level_repair_mode", pd.Series(pd.NA, index=accounting.index, dtype="string")
    ).astype("string").reset_index(drop=True)
    source_replacement = repair_mode.eq(SOURCE_REPLACEMENT_REPAIR_MODE).fillna(False)
    structural_zero = repair_mode.eq(STRUCTURAL_ZERO_REPAIR_MODE).fillna(False)
    protected_accounting = source_replacement | structural_zero
    out["accounting_count"] = pd.to_numeric(
        out[ACCOUNTING_COUNT_COLUMN], errors="coerce"
    ).astype(float)
    if out["accounting_count"].isna().any():
        raise ValueError("accounting controls carry a non-numeric adjusted_count_ags_core")
    out = out.drop(columns=[ACCOUNTING_COUNT_COLUMN])
    out["bucket_population"] = pd.to_numeric(out["bucket_population"], errors="coerce").fillna(0.0)

    levels = series_kernel_levels(
        clean_panel_history(panel, config=config), config=config
    )
    out = out.merge(levels, on=["jurisdiction_id", "offense"], how="left", validate="one_to_one")
    out["clean_year_count"] = (
        pd.to_numeric(out["clean_year_count"], errors="coerce").fillna(0).astype(int)
    )
    out["latest_clean_year"] = pd.to_numeric(out["latest_clean_year"], errors="coerce").astype(
        "Int64"
    )
    out["peer_group_id"] = (
        out["state_fips"].astype("string").fillna("")
        + ":"
        + out["jurisdiction_type"].astype("string").fillna("")
        + ":"
        + out["pop_band"].astype("string").fillna("")
    )

    required_clean_years = pd.Series(int(config.min_clean_years), index=out.index, dtype=int)
    is_state_remainder = out["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
    required_clean_years.loc[is_state_remainder] = REMAINDER_MIN_CLEAN_YEARS
    has_history = out["clean_year_count"].ge(required_clean_years) & out["ewa_count"].notna()
    has_peer_history = (
        out["clean_year_count"].ge(PEER_REFERENCE_MIN_CLEAN_YEARS)
        & out["ewa_count"].notna()
    )
    has_exposure = out["bucket_population"].gt(0.0)
    peer_supported_type = out["jurisdiction_type"].astype("string").isin(
        {"municipal", STATE_REMAINDER_TYPE}
    )
    eligible = has_history & has_exposure & peer_supported_type
    peer_eligible = has_peer_history & has_exposure & peer_supported_type
    is_murder = out["offense"].astype("string").eq("murder")

    peer = _peer_rates(
        out.loc[
            peer_eligible & ~is_murder,
            [
                "offense",
                "state_fips",
                "jurisdiction_type",
                "pop_band",
                "ewa_count",
                "bucket_population",
            ],
        ]
    )
    out = out.merge(
        peer,
        on=["offense", "state_fips", "jurisdiction_type", "pop_band"],
        how="left",
        validate="many_to_one",
    )
    murder_peer = _murder_leave_one_out_peer_rates(
        out.loc[
            eligible & is_murder,
            [
                "jurisdiction_id",
                "offense",
                "state_fips",
                "jurisdiction_type",
                "pop_band",
            ],
        ],
        reference=out.loc[
            peer_eligible & is_murder,
            [
                "jurisdiction_id",
                "offense",
                "state_fips",
                "jurisdiction_type",
                "pop_band",
                "information_weighted_count",
                "information_weighted_exposure",
            ],
        ],
    ).rename(columns={"peer_rate": "murder_peer_rate"})
    out = out.merge(
        murder_peer,
        on=["jurisdiction_id", "offense"],
        how="left",
        validate="one_to_one",
    )
    out.loc[is_murder, "peer_rate"] = out.loc[is_murder, "murder_peer_rate"]
    out = out.drop(columns="murder_peer_rate")

    murder_exposure = pd.to_numeric(
        out["information_weighted_exposure"], errors="coerce"
    ).fillna(0.0)
    murder_count = pd.to_numeric(
        out["information_weighted_count"], errors="coerce"
    ).fillna(0.0)
    out.loc[is_murder, "ewa_count"] = (
        murder_count.loc[is_murder]
        / murder_exposure.loc[is_murder].where(murder_exposure.loc[is_murder].gt(0.0))
        * out.loc[is_murder, "bucket_population"]
    )
    out["peer_predicted_count"] = out["peer_rate"] * out["bucket_population"]

    halflife_map = config.halflife_map()
    out["halflife_used"] = out["offense"].astype(str).map(halflife_map).astype(float)
    out["temporal_kernel"] = pd.Series(
        np.where(out["halflife_used"].isna(), FLAT_MEAN_KERNEL, EWA_KERNEL), index=out.index
    ).astype("string")
    out["shrinkage_k"] = float(config.shrinkage_k)
    out["prior_exposure"] = np.nan
    out.loc[is_murder, "shrinkage_k"] = float(MURDER_INFORMATION_PRIOR_EXPOSURE)
    out.loc[is_murder, "prior_exposure"] = float(MURDER_INFORMATION_PRIOR_EXPOSURE)
    out["ewa_weight"] = out["ewa_count"] / (out["ewa_count"] + float(config.shrinkage_k))
    out.loc[is_murder, "ewa_weight"] = (
        murder_exposure.loc[is_murder]
        / (murder_exposure.loc[is_murder] + float(MURDER_INFORMATION_PRIOR_EXPOSURE))
    )

    prediction = (
        out["ewa_weight"] * out["ewa_count"]
        + (1.0 - out["ewa_weight"]) * out["peer_predicted_count"]
    )
    out.loc[
        ~is_murder,
        ["information_weighted_count", "information_weighted_exposure", "prior_exposure"],
    ] = np.nan
    usable = pd.Series(np.isfinite(prediction.to_numpy(dtype=float)), index=out.index)
    estimated = (
        eligible & out["peer_rate"].notna() & usable & ~protected_accounting
    ).astype(bool)

    out["smoothed_count"] = np.where(estimated, prediction, out["accounting_count"]).astype(float)
    out["estimator"] = pd.Series(FALLBACK_ACCOUNTING_ESTIMATOR, index=out.index, dtype="string")
    out.loc[estimated & ~is_murder, "estimator"] = DYNAMIC_SHRINK_ESTIMATOR
    out.loc[estimated & is_murder, "estimator"] = INFORMATION_POSTERIOR_ESTIMATOR
    # First applicable reason wins, in contract order.
    out["fallback_reason"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[source_replacement, "fallback_reason"] = SOURCE_REPLACEMENT_PROTECTED_REASON
    out.loc[structural_zero, "fallback_reason"] = STRUCTURAL_ZERO_PROTECTED_REASON
    unclassified_fallback = ~estimated & out["fallback_reason"].isna()
    out.loc[unclassified_fallback & ~has_history, "fallback_reason"] = INSUFFICIENT_HISTORY_REASON
    out.loc[
        unclassified_fallback & has_history & ~has_exposure, "fallback_reason"
    ] = NO_EXPOSURE_REASON
    out.loc[~estimated & out["fallback_reason"].isna(), "fallback_reason"] = NO_PEER_SUPPORT_REASON
    out.loc[~estimated, "temporal_kernel"] = pd.NA
    for column in ESTIMATE_DIAGNOSTIC_COLUMNS:
        if column == "temporal_kernel":
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
        out.loc[~estimated, column] = np.nan
    out["peer_group_id"] = out["peer_group_id"].astype("string")
    out = _apply_state_remainder_coverage_floor(
        out, panel=panel, config=config, accounting_controls=accounting
    )
    out = _apply_national_anchor_calibration(
        out,
        national_anchor=national_anchor,
        source=national_anchor_source,
    )
    out["year"] = int(config.year)

    return (
        out.reindex(columns=SMOOTHED_CONTROL_COLUMNS)
        .sort_values(
            ["state_fips", "jurisdiction_type", "jurisdiction_id", "offense"], kind="mergesort"
        )
        .reset_index(drop=True)
    )


def assert_smoothed_control_invariants(
    *,
    smoothed: pd.DataFrame,
    accounting: pd.DataFrame,
    config: SmoothedControlConfig = SmoothedControlConfig(),
    tolerance: float = 1e-9,
) -> None:
    """The contract's invariants, in code (SURFACE2_CONTROLS_CONTRACT.md)."""
    missing_columns = sorted(set(SMOOTHED_CONTROL_COLUMNS) - set(smoothed.columns))
    if missing_columns:
        raise ValueError(f"smoothed controls are missing columns {missing_columns}")

    key = ["jurisdiction_id", "offense"]
    if smoothed.duplicated(key).any():
        raise ValueError("smoothed controls carry duplicate (jurisdiction_id, offense) keys")
    expected = set(map(tuple, accounting[key].astype(str).to_numpy()))
    actual = set(map(tuple, smoothed[key].astype(str).to_numpy()))
    if expected != actual:
        raise ValueError(
            f"smoothed control universe differs from the accounting universe: "
            f"{len(expected - actual)} missing, {len(actual - expected)} extra"
        )

    value = pd.to_numeric(smoothed["smoothed_count"], errors="coerce")
    if value.isna().any() or not np.isfinite(value.to_numpy()).all():
        raise ValueError("smoothed controls carry a nonfinite smoothed_count")
    if value.lt(-tolerance).any():
        raise ValueError("smoothed controls carry a negative smoothed_count")
    precalibration = pd.to_numeric(
        smoothed["precalibration_smoothed_count"], errors="coerce"
    )
    anchor_count = pd.to_numeric(smoothed["national_anchor_count"], errors="coerce")
    calibration_factor = pd.to_numeric(
        smoothed["national_calibration_factor"], errors="coerce"
    )
    if (
        precalibration.isna().any()
        or precalibration.lt(-tolerance).any()
        or anchor_count.isna().any()
        or anchor_count.lt(-tolerance).any()
        or calibration_factor.isna().any()
        or calibration_factor.le(0.0).any()
    ):
        raise ValueError("smoothed controls carry invalid national calibration fields")
    recomposed_calibrated = precalibration * calibration_factor
    if (recomposed_calibrated - value).abs().gt(1e-6).any():
        raise ValueError("smoothed controls do not recompose from the national calibration")
    calibration_nunique = smoothed.assign(
        _factor=calibration_factor, _anchor=anchor_count
    ).groupby("offense", dropna=False)[["_factor", "_anchor"]].nunique(dropna=False)
    if calibration_nunique.gt(1).any().any():
        raise ValueError("national calibration fields are not constant within offense")
    calibrated_totals = value.groupby(smoothed["offense"].astype(str)).sum()
    published_anchors = anchor_count.groupby(smoothed["offense"].astype(str)).first()
    if (calibrated_totals - published_anchors).abs().gt(1e-6).any():
        raise ValueError("smoothed offense totals do not equal their national anchors")
    if not smoothed["national_anchor_source"].astype("string").isin(
        {NATIONAL_ANCHOR_SOURCE, IDENTITY_ANCHOR_SOURCE}
    ).all():
        raise ValueError("smoothed controls carry an unknown national anchor source")

    estimators = smoothed["estimator"].astype("string")
    if not estimators.isin(
        {
            DYNAMIC_SHRINK_ESTIMATOR,
            INFORMATION_POSTERIOR_ESTIMATOR,
            FALLBACK_ACCOUNTING_ESTIMATOR,
            COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR,
        }
    ).all():
        raise ValueError("smoothed controls carry an unknown estimator label")
    fallback = estimators.eq(FALLBACK_ACCOUNTING_ESTIMATOR)
    dynamic = estimators.eq(DYNAMIC_SHRINK_ESTIMATOR)
    information_posterior = estimators.eq(INFORMATION_POSTERIOR_ESTIMATOR)
    estimated = dynamic | information_posterior
    coverage_adjusted = estimators.eq(COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR)
    reasons = smoothed["fallback_reason"].astype("string")
    if reasons[fallback].isna().any():
        raise ValueError("a fallback row carries no documented fallback reason")
    if reasons[~fallback].notna().any():
        raise ValueError("an estimated row carries a fallback reason")
    if not reasons[fallback].isin(
        {
            INSUFFICIENT_HISTORY_REASON,
            NO_EXPOSURE_REASON,
            NO_PEER_SUPPORT_REASON,
            SOURCE_REPLACEMENT_PROTECTED_REASON,
            STRUCTURAL_ZERO_PROTECTED_REASON,
        }
    ).all():
        raise ValueError("smoothed controls carry an undocumented fallback reason")

    accounting_value = pd.to_numeric(smoothed["accounting_count"], errors="coerce")
    if (precalibration[fallback] - accounting_value[fallback]).abs().gt(tolerance).any():
        raise ValueError("a fallback row does not carry the accounting value before calibration")
    if smoothed.loc[fallback, ESTIMATE_DIAGNOSTIC_COLUMNS].notna().any().any():
        raise ValueError("a fallback row publishes estimate diagnostics for an estimate that never ran")
    if smoothed.loc[fallback, COVERAGE_REMAINDER_DIAGNOSTIC_COLUMNS].notna().any().any():
        raise ValueError("a fallback row publishes coverage diagnostics for an estimate that never ran")
    if smoothed.loc[estimated, COVERAGE_REMAINDER_DIAGNOSTIC_COLUMNS].notna().any().any():
        raise ValueError("a dynamic estimate carries state-remainder coverage diagnostics")

    weight = pd.to_numeric(smoothed.loc[estimated, "ewa_weight"], errors="coerce")
    if weight.isna().any() or weight.lt(-tolerance).any() or weight.gt(1.0 + tolerance).any():
        raise ValueError("an estimated row carries an ewa_weight outside [0, 1]")
    halflife_map = config.halflife_map()
    expected_halflife = smoothed.loc[estimated, "offense"].astype(str).map(halflife_map).astype(float)
    published_halflife = pd.to_numeric(smoothed.loc[estimated, "halflife_used"], errors="coerce")
    if not published_halflife.isna().equals(expected_halflife.isna()):
        raise ValueError("an estimated row publishes the wrong temporal kernel family")
    if (published_halflife.fillna(0.0) - expected_halflife.fillna(0.0)).abs().gt(tolerance).any():
        raise ValueError("an estimated row publishes a half-life the offense was not configured with")
    recomposed = (
        weight * pd.to_numeric(smoothed.loc[estimated, "ewa_count"], errors="coerce")
        + (1.0 - weight) * pd.to_numeric(smoothed.loc[estimated, "peer_predicted_count"], errors="coerce")
    )
    if (recomposed - precalibration[estimated]).abs().gt(1e-6).any():
        raise ValueError("an estimated row does not recompose from its published components")

    if not smoothed.loc[information_posterior, "offense"].astype("string").eq("murder").all():
        raise ValueError("an information posterior estimate is not murder")
    information = smoothed.loc[
        information_posterior,
        ["information_weighted_count", "information_weighted_exposure", "prior_exposure"],
    ].apply(pd.to_numeric, errors="coerce")
    if (
        information.isna().any().any()
        or information["information_weighted_count"].lt(-tolerance).any()
        or information["information_weighted_exposure"].le(0.0).any()
        or information["prior_exposure"].le(0.0).any()
    ):
        raise ValueError("a murder information posterior carries invalid information diagnostics")

    if not smoothed.loc[coverage_adjusted, "jurisdiction_type"].astype("string").eq(
        STATE_REMAINDER_TYPE
    ).all():
        raise ValueError("a coverage-adjusted estimate is not a state remainder")
    if smoothed.loc[coverage_adjusted, ESTIMATE_DIAGNOSTIC_COLUMNS].notna().any().any():
        raise ValueError("a coverage-adjusted remainder carries dynamic-estimator diagnostics")
    coverage_weight = pd.to_numeric(
        smoothed.loc[coverage_adjusted, "reporting_coverage_weight"], errors="coerce"
    )
    coverage_peer = pd.to_numeric(
        smoothed.loc[coverage_adjusted, "coverage_peer_predicted_count"], errors="coerce"
    )
    coverage_blend = pd.to_numeric(
        smoothed.loc[coverage_adjusted, "coverage_blended_count"], errors="coerce"
    )
    coverage_reference = smoothed.loc[
        coverage_adjusted,
        [
            "remainder_to_municipal_rate_ratio",
            "coverage_municipal_reference_population",
            "coverage_municipal_reference_unit_count",
            "coverage_training_remainder_population",
            "coverage_training_remainder_unit_count",
            "coverage_training_municipal_population",
            "coverage_training_municipal_unit_count",
        ],
    ].apply(pd.to_numeric, errors="coerce")
    coverage_accounting = accounting_value[coverage_adjusted]
    if (
        coverage_weight.isna().any()
        or coverage_weight.lt(-tolerance).any()
        or coverage_weight.gt(1.0 + tolerance).any()
        or coverage_peer.isna().any()
        or coverage_peer.lt(-tolerance).any()
        or coverage_reference.isna().any().any()
        or coverage_reference.le(0.0).any().any()
    ):
        raise ValueError("a coverage-adjusted remainder carries invalid diagnostics")
    partial_history_blend = (
        coverage_weight * coverage_accounting
        + (1.0 - coverage_weight) * coverage_peer
    )
    no_complete_year = pd.to_numeric(
        smoothed.loc[coverage_adjusted, "clean_year_count"], errors="coerce"
    ).fillna(0.0).eq(0.0)
    expected_blend = pd.Series(
        np.where(no_complete_year, coverage_peer, partial_history_blend),
        index=coverage_blend.index,
        dtype=float,
    )
    if (expected_blend - coverage_blend).abs().gt(1e-6).any():
        raise ValueError("a coverage-adjusted remainder does not recompose its blend")
    expected_coverage_value = np.maximum(coverage_accounting, coverage_blend)
    if level_lane_policy().remainder_coverage_cap:
        evidence_cap = pd.to_numeric(
            smoothed.loc[coverage_adjusted, "coverage_evidence_capped_count"],
            errors="coerce",
        )
        band_cap = pd.to_numeric(
            smoothed.loc[coverage_adjusted, "coverage_rate_band_ceiling_count"],
            errors="coerce",
        )
        ceiling = pd.concat([evidence_cap, band_cap], axis=1).min(axis=1)
        expected_coverage_value = np.maximum(
            coverage_accounting,
            np.minimum(expected_coverage_value, ceiling.fillna(np.inf)),
        )
    if (expected_coverage_value - precalibration[coverage_adjusted]).abs().gt(1e-6).any():
        raise ValueError("a coverage-adjusted remainder does not recompose its value")


# --- sanity bands and the build summary -------------------------------------


def _load_cde_history(paths: RepoPaths, *, config: SmoothedControlConfig) -> pd.DataFrame:
    """CONUS+DC FBI CDE estimated totals over the E1 history, long by offense.

    The CDE series is coverage-complete on the FBI's own estimation, which makes it the
    like-for-like comparator for a coverage-complete smoothed level -- the panel's own
    pre-target years carry no repair or imputation and are a reported-basis lower bound.
    """
    path = _cde_estimates_path(paths, year=int(config.year))
    if not path.exists():
        return pd.DataFrame(columns=["offense", "year", "cde_total"])
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    numeric_year = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame[
        numeric_year.ge(int(config.panel_year_start))
        & numeric_year.le(int(config.year))
    ].copy()
    frame["state_abbr"] = frame["state_abbr"].astype("string").str.strip().str.upper()
    frame = frame[
        frame["state_abbr"].notna()
        & ~frame["state_abbr"].isin({"<NA>", "NAN", *config.exclude_scope_state_abbrs})
    ]
    rows = []
    for offense, column in CDE_OFFENSE_MAP.items():
        if column not in frame.columns:
            continue
        totals = (
            pd.to_numeric(
                frame[column].astype(str).str.replace(",", "").str.strip(), errors="coerce"
            )
            .groupby(frame["year"].astype(int))
            .sum()
        )
        rows.extend(
            {"offense": offense, "year": int(year), "cde_total": float(total)}
            for year, total in totals.items()
        )
    return pd.DataFrame(rows, columns=["offense", "year", "cde_total"])


def _load_cde_band(paths: RepoPaths, *, config: SmoothedControlConfig) -> pd.DataFrame:
    history = _load_cde_history(paths, config=config)
    return history[history["year"].isin(config.band_years())].reset_index(drop=True)


def summarize_smoothed_controls(
    *,
    smoothed: pd.DataFrame,
    panel: pd.DataFrame,
    cde_band: pd.DataFrame | None = None,
    config: SmoothedControlConfig = SmoothedControlConfig(),
) -> dict:
    """Artifact stats plus both sanity bands. Excursions are reported, never suppressed."""
    estimator = smoothed["estimator"].astype("string")
    fallback = estimator.eq(FALLBACK_ACCOUNTING_ESTIMATOR)
    coverage_adjusted = estimator.eq(COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR)
    accounting_total = float(pd.to_numeric(smoothed["accounting_count"], errors="coerce").sum())

    panel_totals = (
        pd.to_numeric(panel["estimated_count"], errors="coerce")
        .groupby([panel["offense"].astype(str), pd.to_numeric(panel["year"], errors="coerce").astype(int)])
        .sum()
        .unstack()
    )
    band_years = [year for year in config.band_years() if year in panel_totals.columns]
    cde = cde_band if cde_band is not None else pd.DataFrame(columns=["offense", "year", "cde_total"])
    cde_totals = (
        cde.pivot_table(index="offense", columns="year", values="cde_total", aggfunc="sum")
        if not cde.empty
        else pd.DataFrame()
    )

    national: dict[str, dict[str, float | bool | None]] = {}
    for offense, group in smoothed.groupby(smoothed["offense"].astype(str)):
        entry: dict[str, float | bool | None] = {
            "smoothed_total": float(pd.to_numeric(group["smoothed_count"], errors="coerce").sum()),
            "precalibration_smoothed_total": float(
                pd.to_numeric(
                    group["precalibration_smoothed_count"], errors="coerce"
                ).sum()
            ),
            "national_anchor_count": float(
                pd.to_numeric(group["national_anchor_count"], errors="coerce").iloc[0]
            ),
            "national_calibration_factor": float(
                pd.to_numeric(
                    group["national_calibration_factor"], errors="coerce"
                ).iloc[0]
            ),
            "accounting_total": float(pd.to_numeric(group["accounting_count"], errors="coerce").sum()),
        }
        entry["smoothed_over_accounting"] = (
            entry["smoothed_total"] / entry["accounting_total"] if entry["accounting_total"] else None
        )
        if offense in panel_totals.index and band_years:
            row = panel_totals.loc[offense, band_years]
            entry["panel_band_min"] = float(row.min())
            entry["panel_band_max"] = float(row.max())
            entry["within_panel_band"] = bool(
                entry["panel_band_min"] <= entry["smoothed_total"] <= entry["panel_band_max"]
            )
            entry["panel_mean_all_years"] = float(panel_totals.loc[offense].mean())
        if not cde_totals.empty and offense in cde_totals.index:
            row = cde_totals.loc[offense].dropna()
            if not row.empty:
                entry["cde_band_min"] = float(row.min())
                entry["cde_band_max"] = float(row.max())
                entry["cde_band_mean"] = float(row.mean())
                entry["within_cde_band"] = bool(
                    entry["cde_band_min"] <= entry["smoothed_total"] <= entry["cde_band_max"]
                )
        national[offense] = entry

    return {
        "year": int(config.year),
        "estimator": DYNAMIC_SHRINK_ESTIMATOR,
        "estimator_source": "E1 round 2 (analysis_scratch/final_phase/e1_tournament)",
        "national_anchor_source": NATIONAL_ANCHOR_SOURCE,
        "shrinkage_k": float(config.shrinkage_k),
        "min_clean_years": int(config.min_clean_years),
        "peer_reference_min_clean_years": int(PEER_REFERENCE_MIN_CLEAN_YEARS),
        "remainder_min_clean_years": int(REMAINDER_MIN_CLEAN_YEARS),
        "remainder_municipal_peer_pop_bands": list(
            REMAINDER_MUNICIPAL_PEER_POP_BANDS
        ),
        "halflife_years": {offense: halflife for offense, halflife in config.halflife_years},
        "units": int(len(smoothed)),
        "estimated_units": int((~fallback).sum()),
        "coverage_adjusted_remainder_units": int(coverage_adjusted.sum()),
        "coverage_adjusted_remainder_added_count": float(
            (
                pd.to_numeric(
                    smoothed.loc[coverage_adjusted, "smoothed_count"], errors="coerce"
                )
                - pd.to_numeric(
                    smoothed.loc[coverage_adjusted, "accounting_count"], errors="coerce"
                )
            ).sum()
        ),
        "fallback_units": int(fallback.sum()),
        "fallback_share": float(fallback.mean()) if len(smoothed) else 0.0,
        "fallback_reason_counts": {
            str(reason): int(count)
            for reason, count in smoothed.loc[fallback, "fallback_reason"].value_counts().items()
        },
        "fallback_share_by_jurisdiction_type": {
            str(kind): float(share)
            for kind, share in fallback.groupby(smoothed["jurisdiction_type"].astype(str)).mean().items()
        },
        "fallback_accounting_mass_share": (
            float(pd.to_numeric(smoothed.loc[fallback, "accounting_count"], errors="coerce").sum() / accounting_total)
            if accounting_total
            else 0.0
        ),
        "negative_smoothed_counts": int(
            pd.to_numeric(smoothed["smoothed_count"], errors="coerce").lt(0.0).sum()
        ),
        "sanity_band_years": list(config.band_years()),
        "national_totals": national,
    }


# --- build entry point ------------------------------------------------------


def build_smoothed_controls_bundle(
    *, paths: RepoPaths, config: SmoothedControlConfig = SmoothedControlConfig()
) -> tuple[pd.DataFrame, dict]:
    accounting_path = accounting_controls_path(paths, year=int(config.year))
    panel_path = jurisdiction_year_estimates_path(paths)
    for path, hint in (
        (accounting_path, "build-controls"),
        (panel_path, "build-controls"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is required by the smoothed-control lane and is absent. Run "
                f"`{hint}` first; this build never rebuilds the accounting surface, so that "
                f"edition stays byte-identical."
            )
    accounting = pd.read_parquet(accounting_path)
    panel = pd.read_parquet(panel_path)
    cde_history = _load_cde_history(paths, config=config)
    national_anchor = national_e1_anchor_from_cde_history(
        cde_history,
        config=config,
    )
    national_anchor = national_anchor[
        national_anchor["offense"].astype(str).isin(set(accounting["offense"].astype(str)))
    ].reset_index(drop=True)
    smoothed = build_smoothed_controls(
        accounting=accounting,
        panel=panel,
        config=config,
        national_anchor=national_anchor,
    )
    assert_smoothed_control_invariants(
        smoothed=smoothed, accounting=accounting, config=config
    )
    summary = summarize_smoothed_controls(
        smoothed=smoothed,
        panel=panel,
        cde_band=cde_history[cde_history["year"].isin(config.band_years())],
        config=config,
    )
    return smoothed, summary


def write_v2_smoothed_controls(
    *,
    paths: RepoPaths,
    out_path: Path | None = None,
    config: SmoothedControlConfig = SmoothedControlConfig(),
    force: bool = False,
) -> tuple[Path, dict]:
    artifact = out_path or smoothed_controls_path(paths, year=int(config.year))
    summary_path = smoothed_controls_summary_path(paths, year=int(config.year))
    if not force and smoothed_controls_artifact_is_current(
        paths, year=int(config.year), out_path=artifact
    ):
        return artifact, json.loads(summary_path.read_text())

    # This build reads the controls stage and writes a sibling artifact the outputs stage
    # does not consume, so an active outputs build is not a blocker; an active upstream
    # build is, because it would move the inputs underneath the read.
    with stage_write_lock(
        paths=paths,
        stage="controls",
        blocked_by=blockers_for_stage("controls", ignore=("outputs",)),
    ):
        smoothed, summary = build_smoothed_controls_bundle(paths=paths, config=config)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        smoothed.to_parquet(artifact, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        write_dependency_stamp(
            artifact, smoothed_controls_dependency_paths(paths, year=int(config.year))
        )
    return artifact, summary
