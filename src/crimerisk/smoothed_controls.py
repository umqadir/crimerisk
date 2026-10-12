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

from dataclasses import dataclass, replace
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
from crimerisk.jurisdiction_targets import FULL_YEAR_ADMITTED_MIN_SHARE, admitted_full_year_history
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
# 2025.3 (owner item B, pm/e3/evidence/s6e_*): the target year's own clean count weighted by its
# precision against the forecast from the years before it (see `precision_weighted_levels`).
PRECISION_WEIGHTED_ESTIMATOR = "precision_weighted_current_year"
# The signal variance is estimated from forecasts that rest on at least two earlier panel years.
SIGNAL_VARIANCE_MIN_FORECAST_YEARS = 2
# Current-year check (2025.3): the precision weighting trusts an admitted target-year count, so a
# unit of at least this many residents whose clean target-year count is below half its own
# state-level history forecast, and below half of what its peers' current/forecast ratio implies
# (offense x population band median), is "unverified" unless a ledger row confirms the change
# (configs/current_year_confirmations.csv). The flag is for review only (2025.3 review 2): it changes no
# control, and the release validator lists every unverified cell. A count found defective is refused
# through a reporting-regime restriction or hard-evidence row, which the adjudicated history consumes.
# The factor 2 is the total-check history band (scripts/eval/total_checks.py).
# A drop must also be improbable as noise: the count is in the lower 1-in-1,000 tail of a negative
# binomial with mean forecast x peer ratio and variance m + tau2 m^2 (the unit's own signal variance),
# so a town's zero murders against a forecast of 4 is not flagged.
CURRENT_YEAR_CHECK_MIN_POPULATION = 25_000.0
CURRENT_YEAR_CHECK_FACTOR = 0.5
CURRENT_YEAR_CHECK_TAIL = 0.001
CURRENT_YEAR_CONFIRMATIONS_RELPATH = "configs/current_year_confirmations.csv"
CURRENT_YEAR_CONFIRMATION_COLUMNS = ["jurisdiction_id", "offense", "year", "verdict", "evidence_source", "note"]
CURRENT_YEAR_UNVERIFIED = "unverified"
CURRENT_YEAR_CONFIRMED = "confirmed"
# The one verdict a confirmation row may carry; any other value is refused when the file is read.
CURRENT_YEAR_CONFIRMED_VERDICT = "confirmed_real"
INFORMATION_POSTERIOR_ESTIMATOR = "information_posterior"
FALLBACK_ACCOUNTING_ESTIMATOR = "fallback_accounting"
COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR = "coverage_adjusted_remainder"
NATIONAL_ANCHOR_SOURCE = "fbi_cde_state_estimates_data_year"
# The accounting controls and the FBI's data-year estimate are two measurements of the same
# national level: accounting is bottom-up (reports plus repair and imputation), the CDE is the
# FBI's own coverage-complete estimate. The build stops if they disagree by more than this.
NATIONAL_ANCHOR_ACCOUNTING_TOLERANCE = 0.10
IDENTITY_ANCHOR_SOURCE = "identity_no_external_anchor"
# 2025.2 (rc4) state reconciliation, pre-registered in docs/NOTES.md: each state-offense total is raked
# to the FBI CDE data-year state estimate, except where our own accounting total for the state
# disagrees with that estimate by more than this factor (the FBI imputation is then in doubt and the
# accounting total is the target). The national total is the sum of the state targets.
# 2025.3 (owner item A) replaced that switch. A state's target is the CDE estimate plus, for every
# agency whose target-year count the adjudicated history corrects with evidence (a regime restriction
# or a source-supported replacement, `level_lane_hard_case_id`) and whose own count the FBI used, our
# corrected count minus the count the FBI used. An agency the FBI imputed keeps the FBI's imputation.
# Where the CDE state estimate itself is documented as broken, a recorded exception
# (configs/state_anchor_exceptions.csv, governed) makes the accounting total the target.
# Disagreement size alone never changes the target, so no input revision moves it discontinuously.
STATE_ANCHOR_CDE_SOURCE = "fbi_cde_state_estimate_data_year"
STATE_ANCHOR_CORRECTED_SOURCE = "fbi_cde_state_estimate_plus_documented_corrections"
STATE_ANCHOR_EXCEPTION_SOURCE = "accounting_state_total_recorded_cde_exception"
STATE_ANCHOR_IDENTITY_SOURCE = "identity_no_state_anchor"
STATE_ANCHOR_EXCEPTIONS_RELPATH = "configs/state_anchor_exceptions.csv"
STATE_ANCHOR_EXCEPTION_COLUMNS = ["state_abbr", "offense", "effective_from_year", "effective_to_year", "evidence_source", "note"]
# The FBI lanes whose count is the agency's own report as the FBI holds it, in order of preference.
FBI_USED_COUNT_SOURCES = ("srs_return_a_annual", "nibrs_srs_equivalent_annual", "cius_publication_annual")

EWA_KERNEL = "ewa"
FLAT_MEAN_KERNEL = "flat_mean"

INSUFFICIENT_HISTORY_REASON = "insufficient_clean_history"
NO_EXPOSURE_REASON = "no_exposure"
NO_PEER_SUPPORT_REASON = "no_peer_support"
SOURCE_REPLACEMENT_PROTECTED_REASON = "source_supported_replacement_protected"
STRUCTURAL_ZERO_PROTECTED_REASON = "locked_structural_zero_protected"
# A jurisdiction whose agency is numerator-exclusive to a covering agency keeps no level of
# its own: its own past filings are not evidence of crime the covering agency now reports, and
# the covering pool spreads that agency's total onto it (covering_pools).
COVERED_PROTECTED_REASON = "covered_by_provider_protected"

SOURCE_REPLACEMENT_REPAIR_MODE = "source_supported_replacement"
STRUCTURAL_ZERO_REPAIR_MODE = "locked_structural_zero"
COVERED_REPAIR_MODE = "covered_by_covering_agency"

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

PRECISION_DIAGNOSTIC_COLUMNS = [
    "history_forecast_count",
    "history_forecast_variance",
    "history_forecast_years",
    "current_year_count",
    "current_year_weight",
    "signal_relative_variance",
    "signal_variance_level",
    "signal_variance_support",
]


def current_year_check(out: pd.DataFrame, *, confirmations: pd.DataFrame | None, year: int) -> pd.Series:
    """"unverified" / "confirmed" / <NA> per row of the precision-weighted controls (see
    CURRENT_YEAR_CHECK_FACTOR). Peers: the median current/forecast ratio of the offense and
    population band over every precision-weighted unit with both.

    A review flag only: it never changes a control. The release report lists the unverified
    cells; a count found defective is refused through a reporting-regime restriction or a
    hard-evidence row, which the adjudicated history (and so every later year and every
    evaluation fold) consumes. A confirmation counts only with verdict CURRENT_YEAR_CONFIRMED_VERDICT."""
    current = pd.to_numeric(out["current_year_count"], errors="coerce")
    forecast = pd.to_numeric(out["history_forecast_count"], errors="coerce")
    ratio = current / forecast.where(forecast.gt(0))
    peer = ratio.groupby([out["offense"].astype(str), out["pop_band"].astype(str)]).transform("median")
    population = pd.to_numeric(out["bucket_population"], errors="coerce")
    from scipy.stats import nbinom, poisson

    mean = (forecast * peer.clip(upper=1.0)).to_numpy(float)
    tau2 = pd.to_numeric(out.get("signal_relative_variance", 0.0), errors="coerce")
    tau2 = (tau2 if isinstance(tau2, pd.Series) else pd.Series(tau2, index=out.index)).fillna(0.0).to_numpy(float)
    count = current.fillna(0.0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.where(tau2 > 0, 1.0 / np.where(tau2 > 0, tau2, 1.0), np.inf)
        prob = np.where(np.isfinite(size), size / (size + mean), 1.0)
        tail = np.where(np.isfinite(size), nbinom.cdf(count, size, prob), poisson.cdf(count, mean))
    improbable = pd.Series(np.nan_to_num(tail, nan=1.0) < CURRENT_YEAR_CHECK_TAIL, index=out.index)
    low = (
        population.ge(CURRENT_YEAR_CHECK_MIN_POPULATION)
        & ratio.lt(CURRENT_YEAR_CHECK_FACTOR)
        & ratio.lt(CURRENT_YEAR_CHECK_FACTOR * peer)
        & improbable
    ).fillna(False)
    status = pd.Series(pd.NA, index=out.index, dtype="string")
    status[low] = CURRENT_YEAR_UNVERIFIED
    if confirmations is not None and not confirmations.empty:
        c = confirmations[pd.to_numeric(confirmations["year"], errors="coerce").eq(int(year))
                          & confirmations["verdict"].astype("string").eq(CURRENT_YEAR_CONFIRMED_VERDICT).fillna(False)]
        keys = set(zip(c["jurisdiction_id"].astype(str), c["offense"].astype(str), strict=True))
        confirmed = pd.Series([(j, o) in keys for j, o in zip(out["jurisdiction_id"].astype(str), out["offense"].astype(str), strict=True)], index=out.index)
        status[low & confirmed] = CURRENT_YEAR_CONFIRMED
    return status


def load_current_year_confirmations(paths: RepoPaths) -> pd.DataFrame:
    path = Path(paths.repo_root) / CURRENT_YEAR_CONFIRMATIONS_RELPATH
    if not path.exists():
        return pd.DataFrame(columns=CURRENT_YEAR_CONFIRMATION_COLUMNS)
    from crimerisk.ledger import require_approved

    require_approved(path)
    frame = pd.read_csv(path, dtype="string").reindex(columns=CURRENT_YEAR_CONFIRMATION_COLUMNS)
    bad = frame[~frame["verdict"].astype("string").eq(CURRENT_YEAR_CONFIRMED_VERDICT).fillna(False)]
    if not bad.empty:
        raise ValueError(f"{CURRENT_YEAR_CONFIRMATIONS_RELPATH}: verdict must be {CURRENT_YEAR_CONFIRMED_VERDICT!r}; "
                         f"rows {bad[['jurisdiction_id', 'offense', 'year', 'verdict']].to_dict(orient='records')}")
    return frame

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
    "history_forecast_count",
    "history_forecast_variance",
    "history_forecast_years",
    "current_year_count",
    "current_year_weight",
    "signal_relative_variance",
    "signal_variance_support",
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
    "state_anchor_count",
    "state_calibration_factor",
    "state_anchor_source",
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
    "signal_variance_level",
    "current_year_check",
]

PANEL_CLEAN_ROW_COLUMNS = [
    "jurisdiction_id",
    "offense",
    "year",
    "full_year_admitted_count",
    "full_year_admitted_share",
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
        paths.state_dir / "controls" / f"level_lane_admission_{int(year)}.parquet",
        paths.state_dir / "controls" / f"level_lane_mass_ledger_{int(year)}.parquet",
        Path(paths.repo_root) / STATE_ANCHOR_EXCEPTIONS_RELPATH,
        Path(paths.repo_root) / CURRENT_YEAR_CONFIRMATIONS_RELPATH,
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
    """The usable history rows.

    The panel's agency rows come from the one adjudicated reporting history
    (`reporting_history`): a year the level lane refuses is missing for its agency, never a
    filed zero. A jurisdiction-year enters a unit's window when complete, admitted
    twelve-month filings carry at least `FULL_YEAR_ADMITTED_MIN_SHARE` of the jurisdiction's
    expected mass that year (`full_year_admitted_share`, `admitted_full_year_history`), and its
    count is their mass divided
    by that share, so every year averaged is a coverage-complete level. The share is read off
    mass, not off a mean of months: a silent twin agency with nothing to report does not
    disqualify a city's complete filing, and a missing agency that carries real mass does.
    """
    missing = sorted(set(PANEL_CLEAN_ROW_COLUMNS) - set(panel.columns))
    if missing:
        raise ValueError(f"jurisdiction-year panel is missing columns {missing}")
    work = panel.loc[:, PANEL_CLEAN_ROW_COLUMNS].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    in_window = work["year"].ge(int(config.panel_year_start)) & work["year"].le(int(config.year))
    admitted, complete = admitted_full_year_history(work)
    clean = (in_window.fillna(False).astype(bool) & admitted).astype(bool)
    history = work.loc[
        clean,
        ["jurisdiction_id", "offense", "year", "bucket_population"],
    ].copy()
    history["year"] = history["year"].astype(int)
    history["clean_count"] = complete[clean].astype(float).clip(lower=0.0)
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


def national_anchor_from_cde_data_year(
    cde_history: pd.DataFrame,
    *,
    config: SmoothedControlConfig,
) -> pd.DataFrame:
    """The FBI CDE CONUS+DC estimate for the data year itself, per offense.

    The published estimand is offenses in the data year. The temporal kernel may shape how
    that total is spread across jurisdictions; it must not set the national level, which a
    history-weighted anchor did (murder 16,353 against a 2025 estimate of 14,023).
    """
    required = {"offense", "year", "cde_total"}
    missing = sorted(required - set(cde_history.columns))
    if missing:
        raise ValueError(f"CDE history is missing columns {missing}")
    history = cde_history.copy()
    history["year"] = pd.to_numeric(history["year"], errors="coerce").astype("Int64")
    history["cde_total"] = pd.to_numeric(history["cde_total"], errors="coerce")
    rows: list[dict[str, float | str]] = []
    for offense, _halflife in config.halflife_years:
        part = history[
            history["offense"].astype(str).eq(str(offense)) & history["year"].eq(int(config.year))
        ]
        if len(part) != 1:
            raise ValueError(
                f"CDE national anchor for {offense} needs exactly one {int(config.year)} total, "
                f"found {len(part)}"
            )
        total = float(part["cde_total"].iloc[0])
        if not np.isfinite(total) or total < 0.0:
            raise ValueError(f"CDE national anchor for {offense} has an invalid total")
        rows.append({"offense": str(offense), "national_anchor_count": total})
    return pd.DataFrame(rows)


def assert_national_anchor_agrees_with_accounting(
    summary: dict, *, tolerance: float = NATIONAL_ANCHOR_ACCOUNTING_TOLERANCE
) -> None:
    """Fail the build when the accounting total and the data-year anchor disagree."""
    breaches = {
        offense: entry.get("accounting_over_anchor")
        for offense, entry in summary["national_totals"].items()
        if entry.get("accounting_over_anchor") is None
        or abs(float(entry["accounting_over_anchor"]) - 1.0) > float(tolerance)
    }
    if breaches:
        raise ValueError(
            f"national anchor and accounting controls disagree beyond {tolerance:.0%}: {breaches}"
        )


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


CONTRIBUTION_LEDGER_COLUMNS = [
    "state_abbr", "offense", "ori9", "case_id", "fbi_source", "fbi_months_reported", "fbi_used_count",
    "counting_unit", "fbi_source_evidence", "inclusion_status", "replacement_basis", "measured_or_imputed",
    "model_dependent", "corrected_count", "correction",
]
FBI_SOURCE_COUNTING_UNIT = {
    "srs_return_a_annual": "fbi_return_a_offense_count",
    "nibrs_srs_equivalent_annual": "fbi_nibrs_srs_equivalent_offense_count",
    "cius_publication_annual": "fbi_cius_published_offense_count",
}
INCLUDED_FBI_OWN_REPORT = "fbi_used_own_12_month_report"
EXCLUDED_FBI_PARTIAL = "excluded_fbi_imputed_partial_reporter"
EXCLUDED_FBI_NO_REPORT = "excluded_fbi_imputed_nonreporter"
# The FBI's own sources for the agency-year (Return A, NIBRS SRS-equivalent, CIUS publication) give
# different twelve-month counts: the contribution inside the published estimate is not identified, so
# no correction is made (a fixed source priority is not evidence of which count the FBI used).
EXCLUDED_FBI_SOURCES_DISAGREE = "excluded_fbi_sources_disagree"
MEASURED_REPLACEMENT = "measured_external_replacement"
MODEL_REPAIR = "model_repair_of_documented_defect"


class StateAnchorError(ValueError):
    """An anchor input is missing or inconsistent; the anchor is never silently substituted."""


def documented_agency_corrections(
    *, admission: pd.DataFrame, mass_ledger: pd.DataFrame, fbi_observations: pd.DataFrame, year: int
) -> pd.DataFrame:
    """The state-anchor contribution ledger: every agency-offense with an evidence-backed correction in
    the target year (`level_lane_hard_case_id`), with the FBI's own count for it and our corrected count.

    Inclusion: the FBI used the agency's own filed count only for a twelve-month filer; a partial
    reporter's or non-reporter's contribution to the CDE estimate is the FBI's imputation, which
    stands (the row is kept, excluded, correction 0). Basis: a source-supported replacement is a
    measured external count; any other repair of a documented-defective filing is model-derived and
    flagged `model_dependent`. An included row without control mass in the ledger is an error, never a
    zero. `fbi_observations` carries ori9, year, source, offense, count, months_reported.

    Identification: `fbi_source_evidence` lists every FBI source for the agency-offense-year with its
    count and months. A row is included only when every FBI source with twelve months gives the same
    count (the agency's own report, the same in the Return A file, the NIBRS conversion and the CIUS
    agency table); otherwise it is excluded as unidentified. The published FBI imputations for partial
    and non-reporters are held fixed: this is a first-order correction of identified own-report
    contributions, not a re-estimate of the FBI's imputation.
    """
    if admission.empty:
        return pd.DataFrame(columns=CONTRIBUTION_LEDGER_COLUMNS)
    adm = admission[pd.to_numeric(admission["year"], errors="coerce").eq(int(year)) & admission["level_lane_hard_case_id"].notna()]
    adm = adm.assign(ori9=adm["ori9"].astype("string").str.upper(), offense=adm["offense"].astype("string"))
    adm = adm.drop_duplicates(["ori9", "offense"])
    mode = adm.get("level_repair_mode", pd.Series(pd.NA, index=adm.index)).astype("string")
    adm = adm.assign(
        case_id=adm["level_lane_hard_case_id"].astype("string"),
        replacement_basis=np.where(mode.eq(SOURCE_REPLACEMENT_REPAIR_MODE).fillna(False), MEASURED_REPLACEMENT, MODEL_REPAIR),
    )[["state_abbr", "ori9", "offense", "case_id", "replacement_basis"]]
    led = mass_ledger[pd.to_numeric(mass_ledger["year"], errors="coerce").eq(int(year))]
    corrected = (
        led.assign(ori9=led["ori9"].astype("string").str.upper(), offense=led["offense"].astype("string"))
        .groupby(["ori9", "offense"], as_index=False)["final_control_mass"].sum(min_count=1)
        .rename(columns={"final_control_mass": "corrected_count"})
    )
    fbi = fbi_observations[
        pd.to_numeric(fbi_observations["year"], errors="coerce").eq(int(year))
        & fbi_observations["source"].isin(FBI_USED_COUNT_SOURCES)
        & pd.to_numeric(fbi_observations["count"], errors="coerce").notna()
    ].copy()
    fbi["ori9"] = fbi["ori9"].astype("string").str.upper()
    fbi["offense"] = fbi["offense"].astype("string")
    fbi["fbi_months_reported"] = pd.to_numeric(fbi["months_reported"], errors="coerce").fillna(0.0)
    fbi["_rank"] = fbi["source"].map({source: rank for rank, source in enumerate(FBI_USED_COUNT_SOURCES)})
    fbi = fbi.sort_values(["ori9", "offense", "_rank"])
    full = fbi[fbi["fbi_months_reported"].ge(12.0)]
    disagree = full.groupby(["ori9", "offense"])["count"].agg(lambda c: pd.to_numeric(c, errors="coerce").nunique() > 1)
    disagree = disagree.rename("_sources_disagree").reset_index()
    evidence = fbi.groupby(["ori9", "offense"], as_index=False).apply(
        lambda d: pd.Series({"fbi_source_evidence": ";".join(
            f"{src}={float(cnt):g}/{float(mon):g}m" for src, cnt, mon in zip(d["source"], d["count"], d["fbi_months_reported"]))}),
        include_groups=False,
    ) if not fbi.empty else pd.DataFrame(columns=["ori9", "offense", "fbi_source_evidence"])
    fbi = fbi.drop_duplicates(["ori9", "offense"])
    fbi = fbi[["ori9", "offense", "source", "fbi_months_reported", "count"]].rename(
        columns={"source": "fbi_source", "count": "fbi_used_count"})
    out = adm.merge(fbi, on=["ori9", "offense"], how="left").merge(corrected, on=["ori9", "offense"], how="left")
    out = out.merge(evidence, on=["ori9", "offense"], how="left").merge(disagree, on=["ori9", "offense"], how="left")
    months = pd.to_numeric(out["fbi_months_reported"], errors="coerce").fillna(0.0)
    sources_disagree = out["_sources_disagree"].astype("boolean").fillna(False).astype(bool)
    out["inclusion_status"] = np.select(
        [out["fbi_source"].notna() & months.ge(12.0) & sources_disagree,
         out["fbi_source"].notna() & months.ge(12.0), out["fbi_source"].notna() & months.gt(0.0)],
        [EXCLUDED_FBI_SOURCES_DISAGREE, INCLUDED_FBI_OWN_REPORT, EXCLUDED_FBI_PARTIAL],
        default=EXCLUDED_FBI_NO_REPORT,
    )
    out["counting_unit"] = out["fbi_source"].map(FBI_SOURCE_COUNTING_UNIT)
    out["measured_or_imputed"] = np.where(out["replacement_basis"].eq(MEASURED_REPLACEMENT), "measured", "imputed")
    out["model_dependent"] = out["replacement_basis"].eq(MODEL_REPAIR)
    included = out["inclusion_status"].eq(INCLUDED_FBI_OWN_REPORT)
    unmatched = included & out["corrected_count"].isna()
    if unmatched.any():
        raise StateAnchorError(
            "documented corrections without control mass in the level-lane mass ledger: "
            + str(out.loc[unmatched, ["ori9", "offense", "case_id"]].to_dict(orient="records"))
        )
    out["fbi_used_count"] = pd.to_numeric(out["fbi_used_count"], errors="coerce").astype(float)
    out["corrected_count"] = pd.to_numeric(out["corrected_count"], errors="coerce").astype(float)
    nonfinite = included & ~(np.isfinite(out["corrected_count"]) & np.isfinite(out["fbi_used_count"]))
    if nonfinite.any():
        raise StateAnchorError(
            "documented corrections with a missing or non-finite corrected or FBI count: "
            + str(out.loc[nonfinite, ["ori9", "offense", "case_id", "corrected_count", "fbi_used_count"]].to_dict(orient="records"))
        )
    out["correction"] = np.where(included, out["corrected_count"] - out["fbi_used_count"], 0.0)
    return out[CONTRIBUTION_LEDGER_COLUMNS].sort_values(["state_abbr", "offense", "ori9"]).reset_index(drop=True)


def load_state_anchor_exceptions(paths: RepoPaths, *, year: int) -> pd.DataFrame:
    """Recorded CDE breaks in force for `year` (governed config). Every row carries both years: an
    exception is approved for the years its evidence covers and expires after effective_to_year."""
    path = Path(paths.repo_root) / STATE_ANCHOR_EXCEPTIONS_RELPATH
    if not path.exists():
        raise StateAnchorError(f"{path} is absent; an empty file means no exceptions")
    from crimerisk.ledger import require_approved

    require_approved(path)
    frame = pd.read_csv(path, dtype="string").reindex(columns=STATE_ANCHOR_EXCEPTION_COLUMNS)
    start = pd.to_numeric(frame["effective_from_year"], errors="coerce")
    end = pd.to_numeric(frame["effective_to_year"], errors="coerce")
    bad = frame[["state_abbr", "offense", "evidence_source"]].isna().any(axis=1) | start.isna() | end.isna() | end.lt(start)
    if bad.any():
        raise StateAnchorError(f"{path} rows need state, offense, evidence and both years (from <= to): {frame[bad].to_dict(orient='records')}")
    if frame.duplicated(["state_abbr", "offense"]).any():
        raise StateAnchorError(f"{path} has duplicate state-offense rows")
    return frame[start.le(int(year)) & end.ge(int(year))].reset_index(drop=True)


def state_anchor_from_cde_data_year(
    cde_states: pd.DataFrame,
    *,
    accounting: pd.DataFrame,
    corrections: pd.DataFrame | None = None,
    exceptions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per state and offense: the data-year target the smoothed levels are raked to.

    target = CDE state estimate + sum of the included corrections of the contribution ledger
    (`documented_agency_corrections`); a recorded exception takes the accounting total. Also reported:
    `state_anchor_count_external_only`, the target with the model-dependent corrections left out (the
    sensitivity of the anchor to the model it constrains). A state without a CDE estimate, or a
    negative target, is an error, not a substitution. Continuous in every input.
    `cde_states` holds state_abbr, offense, cde_state_count (FBI CDE estimate for the data year);
    `accounting` the Surface-1 rows (state_abbr, offense, adjusted_count_ags_core).
    """
    acc = (
        accounting.assign(
            _count=pd.to_numeric(accounting[ACCOUNTING_COUNT_COLUMN], errors="coerce").fillna(0.0)
        )
        .groupby(["state_abbr", "offense"], as_index=False)["_count"]
        .sum()
        .rename(columns={"_count": "accounting_state_count"})
    )
    acc["state_abbr"] = acc["state_abbr"].astype("string")
    acc["offense"] = acc["offense"].astype("string")
    cde = cde_states[["state_abbr", "offense", "cde_state_count"]].copy()
    cde["state_abbr"] = cde["state_abbr"].astype("string")
    cde["offense"] = cde["offense"].astype("string")
    out = acc.merge(cde, on=["state_abbr", "offense"], how="left", validate="one_to_one")
    cde_count = pd.to_numeric(out["cde_state_count"], errors="coerce")
    if cde_count.isna().any():
        raise StateAnchorError(
            "no CDE state estimate for " + str(out.loc[cde_count.isna(), ["state_abbr", "offense"]].to_dict(orient="records"))
        )
    for column in ("documented_correction_count", "model_dependent_correction_count"):
        out[column] = 0.0
    out["documented_correction_agencies"] = 0
    if corrections is not None and not corrections.empty:
        c = corrections.assign(
            state_abbr=corrections["state_abbr"].astype("string"), offense=corrections["offense"].astype("string"),
            _included=corrections.get("inclusion_status", pd.Series(INCLUDED_FBI_OWN_REPORT, index=corrections.index)).eq(INCLUDED_FBI_OWN_REPORT),
            _model=corrections.get("model_dependent", pd.Series(False, index=corrections.index)).astype(bool),
        )
        c["_corr"] = pd.to_numeric(c["correction"], errors="coerce").fillna(0.0).where(c["_included"], 0.0)
        c["_model_corr"] = c["_corr"].where(c["_model"], 0.0)
        summed = c[c["_included"]].groupby(["state_abbr", "offense"], as_index=False).agg(
            _a=("_corr", "sum"), _m=("_model_corr", "sum"), _n=("ori9", "nunique"))
        out = out.merge(summed, on=["state_abbr", "offense"], how="left")
        out["documented_correction_count"] = out["_a"].fillna(0.0)
        out["model_dependent_correction_count"] = out["_m"].fillna(0.0)
        out["documented_correction_agencies"] = out["_n"].fillna(0).astype(int)
        out = out.drop(columns=["_a", "_m", "_n"])
    excepted = pd.Series(False, index=out.index)
    if exceptions is not None and not exceptions.empty:
        keys = set(zip(exceptions["state_abbr"].astype(str).str.upper(), exceptions["offense"].astype(str), strict=True))
        excepted = pd.Series([(str(s), str(o)) in keys for s, o in zip(out["state_abbr"], out["offense"], strict=True)], index=out.index)
    corrected = cde_count + out["documented_correction_count"]
    external = cde_count + out["documented_correction_count"] - out["model_dependent_correction_count"]
    negative = ~excepted & (corrected.lt(0.0) | external.lt(0.0))
    if negative.any():
        raise StateAnchorError("negative corrected state target: " + str(out.loc[negative, ["state_abbr", "offense"]].to_dict(orient="records")))
    out["state_anchor_count"] = np.where(excepted, out["accounting_state_count"], corrected)
    out["state_anchor_count_external_only"] = np.where(excepted, out["accounting_state_count"], external)
    out["state_anchor_source"] = np.select(
        [excepted, out["documented_correction_agencies"].gt(0)],
        [STATE_ANCHOR_EXCEPTION_SOURCE, STATE_ANCHOR_CORRECTED_SOURCE],
        default=STATE_ANCHOR_CDE_SOURCE,
    )
    out["accounting_over_cde_state"] = out["accounting_state_count"] / cde_count.where(cde_count.gt(0.0))
    return out


def _apply_state_anchor_calibration(
    out: pd.DataFrame,
    *,
    state_anchor: pd.DataFrame | None,
) -> pd.DataFrame:
    """Rake the nationally calibrated levels within each state to the state target (exact)."""
    work = out.copy()
    if state_anchor is None:
        work["state_anchor_count"] = (
            pd.to_numeric(work["smoothed_count"], errors="coerce")
            .groupby([work["state_abbr"], work["offense"]])
            .transform("sum")
        )
        work["state_calibration_factor"] = 1.0
        work["state_anchor_source"] = STATE_ANCHOR_IDENTITY_SOURCE
        return work
    anchor = state_anchor[["state_abbr", "offense", "state_anchor_count", "state_anchor_source"]].copy()
    anchor["state_abbr"] = anchor["state_abbr"].astype("string")
    anchor["offense"] = anchor["offense"].astype("string")
    work["_state"] = work["state_abbr"].astype("string")
    work["_offense"] = work["offense"].astype("string")
    totals = (
        work.groupby(["_state", "_offense"], as_index=False)["smoothed_count"]
        .sum()
        .rename(columns={"_state": "state_abbr", "_offense": "offense", "smoothed_count": "_state_total"})
    )
    calibration = totals.merge(anchor, on=["state_abbr", "offense"], how="left", validate="one_to_one")
    if calibration["state_anchor_count"].isna().any():
        missing = calibration.loc[calibration["state_anchor_count"].isna(), ["state_abbr", "offense"]]
        raise ValueError(f"state anchor missing for {missing.to_dict(orient='records')[:10]}")
    bad = calibration["_state_total"].le(0.0) & calibration["state_anchor_count"].gt(0.0)
    if bad.any():
        raise ValueError(
            f"state anchor needs mass where the smoothed levels have none: {calibration.loc[bad].to_dict(orient='records')[:10]}"
        )
    calibration["state_calibration_factor"] = np.where(
        calibration["_state_total"].gt(0.0),
        calibration["state_anchor_count"] / calibration["_state_total"].where(calibration["_state_total"].gt(0.0)),
        1.0,
    )
    work = work.merge(
        calibration[["state_abbr", "offense", "state_anchor_count", "state_calibration_factor", "state_anchor_source"]]
        .rename(columns={"state_abbr": "_state", "offense": "_offense"}),
        on=["_state", "_offense"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["_state", "_offense"])
    work["smoothed_count"] = work["smoothed_count"] * work["state_calibration_factor"]
    return work


def history_at_year_level(
    history: pd.DataFrame, *, year: int, state_of: pd.Series, state_year_totals: pd.DataFrame | None
) -> pd.DataFrame:
    """Each clean count moved to `year`'s state level, c_t * S_year / S_t (S: FBI CDE state estimate
    for the offense), so a kernel over the past averages the unit's share of its state total. A
    missing state total leaves the count unscaled."""
    if state_year_totals is None or state_year_totals.empty:
        return history
    totals = state_year_totals[["state_abbr", "offense", "year", "cde_state_count"]].copy()
    totals["year"] = pd.to_numeric(totals["year"], errors="coerce").astype(int)
    h = history[["jurisdiction_id", "offense", "year"]].assign(state_abbr=history["jurisdiction_id"].map(state_of).astype("string"))
    h["offense"] = h["offense"].astype("string")
    totals["state_abbr"] = totals["state_abbr"].astype("string")
    totals["offense"] = totals["offense"].astype("string")
    at_t = h.merge(totals, on=["state_abbr", "offense", "year"], how="left")["cde_state_count"].to_numpy(float)
    target = totals[totals["year"].eq(int(year))][["state_abbr", "offense", "cde_state_count"]]
    at_year = h.merge(target, on=["state_abbr", "offense"], how="left")["cde_state_count"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where((at_t > 0) & np.isfinite(at_year), at_year / at_t, 1.0)
    out = history.copy()
    out["clean_count"] = history["clean_count"].to_numpy(float) * factor
    return out


def history_forecasts(
    history: pd.DataFrame,
    *,
    year: int,
    config: SmoothedControlConfig,
    state_of: pd.Series,
    state_year_totals: pd.DataFrame | None,
) -> pd.DataFrame:
    """Per unit: the kernel level of the clean years before `year` at `year`'s state level.

    `forecast_ewa` = sum w_t f_t c_t / sum w_t with f_t = S_year / S_t; `forecast_variance_factor` =
    sum w_t^2 f_t / (sum w_t)^2, so the Poisson variance of the scaled forecast is
    forecast_ewa x factor (each c_t has variance lambda_t = forecast / f_t, scaled by f_t^2);
    `forecast_years` counts the clean years it rests on."""
    columns = ["jurisdiction_id", "offense", "forecast_ewa", "forecast_variance_factor", "forecast_years"]
    past = history[history["year"].lt(int(year))]
    if past.empty:
        return pd.DataFrame(columns=columns)
    scaled = history_at_year_level(past, year=int(year), state_of=state_of, state_year_totals=state_year_totals)
    factor = (scaled["clean_count"].to_numpy(float) / past["clean_count"].to_numpy(float).clip(min=1e-12))
    factor = np.where(past["clean_count"].to_numpy(float) > 0, factor, 1.0)
    life = past["offense"].astype(str).map(config.halflife_map()).astype(float)
    work = scaled.assign(_w=kernel_weights(int(year) - past["year"], life).to_numpy(), _f=factor)
    work["_wc"] = work["_w"] * work["clean_count"]
    work["_w2f"] = work["_w"] ** 2 * work["_f"]
    g = work.groupby(["jurisdiction_id", "offense"], as_index=False).agg(
        _s=("_w", "sum"), _sc=("_wc", "sum"), _s2f=("_w2f", "sum"), forecast_years=("year", "nunique")
    )
    g["forecast_ewa"] = g["_sc"] / g["_s"]
    g["forecast_variance_factor"] = g["_s2f"] / g["_s"] ** 2
    return g[columns]


def _forecast_count(forecast_ewa: pd.Series, peer_predicted: pd.Series) -> pd.Series:
    """The history forecast shrunk to the peer prediction exactly as the E1 level is."""
    weight = forecast_ewa / (forecast_ewa + E1_SHRINKAGE_K)
    return (weight * forecast_ewa + (1.0 - weight) * peer_predicted).where(forecast_ewa.notna(), peer_predicted)


PEER_GROUP_KEYS = ["offense", "state_fips", "jurisdiction_type", "pop_band"]
NATIONAL_PEER_KEYS = ["offense", "jurisdiction_type", "pop_band"]
# A variance estimated from fewer forecast/observation pairs than this falls back up the hierarchy
# (offense x band -> offense -> all offenses); the level used and its support are published.
SIGNAL_VARIANCE_MIN_SUPPORT = 30


def _loo_rate(frame: pd.DataFrame, value: str, exposure: str, keys: list[str]) -> pd.Series:
    """Leave-one-out pooled rate of `value` per `exposure` within `keys` (NaN when no other unit)."""
    v = pd.to_numeric(frame[value], errors="coerce").fillna(0.0)
    e = pd.to_numeric(frame[exposure], errors="coerce").fillna(0.0)
    has = pd.to_numeric(frame[value], errors="coerce").notna() & e.gt(0)
    v, e = v.where(has, 0.0), e.where(has, 0.0)
    gv = v.groupby([frame[k] for k in keys]).transform("sum") - v
    ge = e.groupby([frame[k] for k in keys]).transform("sum") - e
    return (gv / ge.where(ge.gt(0))).astype(float)


def _forecast_frame(
    units: pd.DataFrame,
    history: pd.DataFrame,
    *,
    year: int,
    config: SmoothedControlConfig,
    state_of: pd.Series,
    state_year_totals: pd.DataFrame | None,
) -> pd.DataFrame:
    """Per unit: the forecast H of `year` from the years before it, and the pieces of its variance.

    Past-only throughout: the peer rate is the leave-one-out pool of the other units' own forecasts
    of the same year (state x type x band, else national type x band), never the current year.
    Non-murder: H = s ewa + (1 - s) peer, s = ewa / (ewa + k) as E1, own variance s^2 ewa x factor
    (`history_forecasts`); a unit without earlier clean years rests on its peer alone. Murder: the
    information posterior of the earlier years with the past-only leave-one-out peer rate, variance
    H E_T / (sum w E + k).
    """
    keys = ["jurisdiction_id", "offense"]
    out = units[[*keys, "state_fips", "jurisdiction_type", "pop_band", "bucket_population"]].copy()
    exposure = pd.to_numeric(out["bucket_population"], errors="coerce")
    plain_mask = out["offense"].astype(str).ne("murder")
    plain = out[plain_mask].merge(
        history_forecasts(history, year=year, config=config, state_of=state_of, state_year_totals=state_year_totals),
        on=keys, how="left",
    )
    plain["_rate_value"] = plain["forecast_ewa"]
    peer_rate = _loo_rate(plain, "_rate_value", "bucket_population", PEER_GROUP_KEYS)
    peer_rate = peer_rate.fillna(_loo_rate(plain, "_rate_value", "bucket_population", NATIONAL_PEER_KEYS))
    plain["peer_forecast"] = peer_rate * pd.to_numeric(plain["bucket_population"], errors="coerce")
    ewa = plain["forecast_ewa"]
    s_w = (ewa / (ewa + E1_SHRINKAGE_K)).where(ewa.notna(), 0.0)
    # No peer (no other unit with a forecast in the group or nationally): the own history stands alone.
    s_w = s_w.where(plain["peer_forecast"].notna() | ewa.isna(), 1.0)
    plain["H"] = (s_w * ewa.fillna(0.0) + (1.0 - s_w) * plain["peer_forecast"].fillna(0.0)).where(
        ewa.notna() | plain["peer_forecast"].notna())
    plain["own_variance"] = (s_w**2 * ewa.fillna(0.0) * plain["forecast_variance_factor"].fillna(0.0))
    plain["peer_weight"] = 1.0 - s_w
    plain["has_history"] = ewa.notna()
    plain["forecast_years"] = plain["forecast_years"].fillna(0).astype(int)
    murder = out[~plain_mask].copy()
    if not murder.empty:
        past = history[history["year"].lt(int(year))]
        levels = series_kernel_levels(past, config=replace(config, year=int(year))) if not past.empty else pd.DataFrame(
            columns=[*keys, "information_weighted_count", "information_weighted_exposure", "clean_year_count"])
        murder = murder.merge(levels[[*keys, "information_weighted_count", "information_weighted_exposure", "clean_year_count"]], on=keys, how="left")
        rate = _loo_rate(murder, "information_weighted_count", "information_weighted_exposure", PEER_GROUP_KEYS)
        rate = rate.fillna(_loo_rate(murder, "information_weighted_count", "information_weighted_exposure", NATIONAL_PEER_KEYS))
        e_t = pd.to_numeric(murder["bucket_population"], errors="coerce")
        iwc = pd.to_numeric(murder["information_weighted_count"], errors="coerce").fillna(0.0)
        iwe = pd.to_numeric(murder["information_weighted_exposure"], errors="coerce").fillna(0.0)
        k = float(MURDER_INFORMATION_PRIOR_EXPOSURE)
        murder["peer_forecast"] = rate * e_t
        murder["H"] = (iwc + k * rate) / (iwe + k) * e_t
        murder["own_variance"] = murder["H"] * e_t / (iwe + k)
        murder["peer_weight"] = 0.0  # the prior is inside the posterior variance
        murder["has_history"] = iwe.gt(0)
        murder["forecast_years"] = pd.to_numeric(murder["clean_year_count"], errors="coerce").fillna(0).astype(int)
    columns = [*keys, "pop_band", "H", "own_variance", "peer_weight", "peer_forecast", "has_history", "forecast_years"]
    return pd.concat([plain[columns], murder.reindex(columns=columns)], ignore_index=True)


def _signal_variance_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Method-of-moments tau2 = sum((c - H)^2 - H - var(H)) / sum(H^2) per offense x band, offense, all,
    with the number of pairs behind each."""
    pairs = pairs.assign(_num=(pairs["clean_count"] - pairs["H"]) ** 2 - pairs["H"] - pairs["var_H"], _den=pairs["H"] ** 2)
    tables = []
    for level, keys in (("offense_band", ["offense", "pop_band"]), ("offense", ["offense"]), ("all", [])):
        if keys:
            g = pairs.groupby(keys, as_index=False, dropna=False).agg(_num=("_num", "sum"), _den=("_den", "sum"), support=("_num", "size"))
        else:
            g = pd.DataFrame({"_num": [pairs["_num"].sum()], "_den": [pairs["_den"].sum()], "support": [len(pairs)]})
        g["tau2"] = (g["_num"] / g["_den"].where(g["_den"].gt(0))).clip(lower=0.0)
        g["level"] = level
        tables.append(g.drop(columns=["_num", "_den"]))
    return tables


def _resolve_tau2(frame: pd.DataFrame, tables: list[pd.DataFrame], *, prefix: str) -> pd.DataFrame:
    """Attach tau2 from the finest level with at least SIGNAL_VARIANCE_MIN_SUPPORT pairs."""
    band, offense, overall = tables
    out = frame.merge(band.rename(columns={"tau2": "_t_b", "support": "_n_b"}).drop(columns="level"), on=["offense", "pop_band"], how="left")
    out = out.merge(offense.rename(columns={"tau2": "_t_o", "support": "_n_o"}).drop(columns="level"), on=["offense"], how="left")
    t_all, n_all = (float(overall["tau2"].iloc[0]) if len(overall) and pd.notna(overall["tau2"].iloc[0]) else np.nan,
                    int(overall["support"].iloc[0]) if len(overall) else 0)
    use_b = out["_n_b"].fillna(0).ge(SIGNAL_VARIANCE_MIN_SUPPORT) & out["_t_b"].notna()
    use_o = ~use_b & out["_n_o"].fillna(0).ge(SIGNAL_VARIANCE_MIN_SUPPORT) & out["_t_o"].notna()
    use_a = ~use_b & ~use_o & (n_all >= SIGNAL_VARIANCE_MIN_SUPPORT) & np.isfinite(t_all)
    out[f"{prefix}_tau2"] = np.select([use_b, use_o, use_a], [out["_t_b"], out["_t_o"], t_all], default=np.nan)
    out[f"{prefix}_tau2_level"] = np.select([use_b, use_o, use_a], ["offense_band", "offense", "all"], default="unsupported")
    out[f"{prefix}_tau2_support"] = np.select([use_b, use_o, use_a], [out["_n_b"].fillna(0), out["_n_o"].fillna(0), n_all], default=0).astype(int)
    return out.drop(columns=["_t_b", "_n_b", "_t_o", "_n_o"])


def precision_weighted_levels(
    units: pd.DataFrame,
    history: pd.DataFrame,
    *,
    config: SmoothedControlConfig,
    state_year_totals: pd.DataFrame | None,
) -> pd.DataFrame:
    """The year-T level: the clean target-year count weighted by its precision against the forecast.

        level = w c_T + (1 - w) H,   w = V / (V + H),   V = tau2 H^2 + var(H)

    H is the past-only forecast of T (`_forecast_frame`); var(H) is its own Poisson variance. tau2 is the
    relative variance of the true level around an own-history forecast beyond Poisson and tau2_peer around a
    peer-only forecast; each enters V once (tau2 for a unit with history, tau2_peer for one without), as it
    was fitted. Both are method-of-moments estimates over every earlier unit-year whose forecast rests on at least
    SIGNAL_VARIANCE_MIN_FORECAST_YEARS clean years (own) or on none (peer), taken from the finest of
    offense x band, offense, all offenses with SIGNAL_VARIANCE_MIN_SUPPORT pairs. Where no level has
    that support the fallback is explicit and published (`signal_variance_level` =
    fallback_one_more_year): the current year then counts as one more year of the series. Holding H and the variances fixed, w depends on
    the expected count scale H, not on the realized c_T: a large expected count makes c_T's Poisson
    noise small next to real change. Without a clean target year the level is the forecast.
    `units` carries jurisdiction_id, offense, state_abbr, state_fips, jurisdiction_type, pop_band,
    bucket_population.
    """
    state_of = units.drop_duplicates("jurisdiction_id").set_index("jurisdiction_id")["state_abbr"].astype("string")
    keys = ["jurisdiction_id", "offense"]
    own_rows, peer_rows = [], []
    for year in range(int(config.panel_year_start) + 1, int(config.year)):
        f = _forecast_frame(units, history, year=year, config=config, state_of=state_of, state_year_totals=state_year_totals)
        f = f.merge(history[history["year"].eq(year)][[*keys, "clean_count"]], on=keys, how="inner")
        f = f[f["H"].gt(0)]
        own = f[f["has_history"].fillna(False).astype(bool) & f["forecast_years"].ge(SIGNAL_VARIANCE_MIN_FORECAST_YEARS)]
        own_rows.append(own.assign(var_H=own["own_variance"])[["offense", "pop_band", "clean_count", "H", "var_H"]])
        peer = f[~f["has_history"].fillna(False).astype(bool)]
        peer_rows.append(peer.assign(var_H=0.0)[["offense", "pop_band", "clean_count", "H", "var_H"]])
    empty = pd.DataFrame(columns=["offense", "pop_band", "clean_count", "H", "var_H"])
    own_pairs = pd.concat([r for r in own_rows if not r.empty] or [empty], ignore_index=True)
    peer_pairs = pd.concat([r for r in peer_rows if not r.empty] or [empty], ignore_index=True)
    out = _forecast_frame(units, history, year=int(config.year), config=config, state_of=state_of, state_year_totals=state_year_totals)
    out = out.merge(history[history["year"].eq(int(config.year))][[*keys, "clean_count"]], on=keys, how="left")
    out = _resolve_tau2(out, _signal_variance_table(own_pairs), prefix="signal")
    out = _resolve_tau2(out, _signal_variance_table(peer_pairs), prefix="peer_signal")
    has_history = out["has_history"].fillna(False).astype(bool)
    h = pd.to_numeric(out["H"], errors="coerce")
    peer = pd.to_numeric(out["peer_forecast"], errors="coerce").fillna(0.0)
    # Each fitted dispersion enters once, as it was fitted: tau2 (own) around forecasts whose var_H was
    # the own Poisson variance only, tau2_peer around peer-only forecasts with var_H = 0. So V is
    # tau2 H^2 + own variance for a unit with history and tau2_peer H^2 for one without.
    # Explicit fallback where the applicable level has no SIGNAL_VARIANCE_MIN_SUPPORT pairs (never a
    # zero variance): the forecast keeps its own sampling variance and the peer part is worth one year
    # (variance = its own Poisson level), so the current year counts as one more year.
    own_signal = out["signal_tau2"].where(has_history, out["peer_signal_tau2"])
    fallback = own_signal.isna()
    peer_var = (out["peer_weight"] * peer).where(fallback, 0.0)
    var_h = out["own_variance"].fillna(0.0) + peer_var.fillna(0.0)
    peer_fallback = out["peer_signal_tau2"].isna()
    own_signal = own_signal.fillna(0.0)
    prior_var = own_signal * h**2 + var_h
    level = out["signal_tau2_level"].where(has_history, out["peer_signal_tau2_level"])
    out["signal_tau2_level"] = level.where(~fallback, "fallback_one_more_year")
    out["peer_signal_tau2_level"] = out["peer_signal_tau2_level"].where(~peer_fallback, "fallback_one_more_year")
    out["history_forecast_count"] = h
    out["history_forecast_variance"] = var_h
    out["signal_relative_variance"] = own_signal
    out["signal_variance_level"] = out["signal_tau2_level"]
    out["signal_variance_support"] = out["signal_tau2_support"].where(out["has_history"].fillna(False).astype(bool), out["peer_signal_tau2_support"])
    out["history_forecast_years"] = out["forecast_years"]
    out["current_year_count"] = out["clean_count"]
    # No forecast at all (no earlier clean year and no peer with one): the level is the current count.
    out.loc[h.isna(), "signal_tau2_level"] = "no_forecast"
    out["signal_variance_level"] = out["signal_tau2_level"]
    weight = (prior_var / (prior_var + h)).where(h.gt(0), 1.0).fillna(1.0)
    out["current_year_weight"] = weight.where(out["current_year_count"].notna(), 0.0)
    out["precision_weighted_count"] = (
        out["current_year_weight"] * out["current_year_count"].fillna(0.0) + (1.0 - out["current_year_weight"]) * h.fillna(0.0)
    ).where(h.notna() | out["current_year_count"].notna())
    return out[[*keys, *PRECISION_DIAGNOSTIC_COLUMNS, "precision_weighted_count"]]


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


def remainder_municipal_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Per state and offense, the small-municipality reference the remainder rule compares to.

    Municipal jurisdictions in `REMAINDER_MUNICIPAL_PEER_POP_BANDS` only: their smoothed count,
    bucket population, unit count and rate.
    """
    municipal = (
        frame[
            frame["jurisdiction_type"].astype("string").eq("municipal")
            & frame["pop_band"].astype("string").isin(REMAINDER_MUNICIPAL_PEER_POP_BANDS)
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
    return municipal


def reliable_remainder_rows(remainder: pd.DataFrame) -> pd.DataFrame:
    """The state remainders the remainder-to-municipal ratio is learned from.

    `remainder` is state-remainder rows joined to `remainder_municipal_reference`. Reliable:
    estimated by the ordinary multi-year estimator on at least `REMAINDER_MIN_CLEAN_YEARS`
    clean years, at least `REMAINDER_PEER_MIN_POPULATION` residents, and a municipal reference.
    """
    return remainder[
        remainder["estimator"].astype("string").isin(
            {DYNAMIC_SHRINK_ESTIMATOR, PRECISION_WEIGHTED_ESTIMATOR, INFORMATION_POSTERIOR_ESTIMATOR}
        )
        & pd.to_numeric(remainder["clean_year_count"], errors="coerce").ge(
            REMAINDER_MIN_CLEAN_YEARS
        )
        & pd.to_numeric(remainder["bucket_population"], errors="coerce").ge(
            REMAINDER_PEER_MIN_POPULATION
        )
        & pd.to_numeric(remainder["municipal_population"], errors="coerce").gt(0.0)
    ].copy()


def remainder_peer_ratio(reliable: pd.DataFrame) -> pd.DataFrame:
    """Per offense, the pooled remainder rate over the pooled small-municipal rate."""
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
    return peer


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

    municipal = remainder_municipal_reference(work)

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

    reliable = reliable_remainder_rows(remainder)
    if reliable.empty:
        return work

    peer = remainder_peer_ratio(reliable)
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
    state_anchor: pd.DataFrame | None = None,
    state_year_totals: pd.DataFrame | None = None,
    current_year_confirmations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Surface 2 for every Surface-1 control unit: a validated estimator or coded fallback.

    `state_year_totals` (state_abbr, offense, year, cde_state_count) puts the history forecast of
    the precision-weighted estimator at the target year's state level; without it the forecast is
    the plain kernel level.

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
    covered = repair_mode.eq(COVERED_REPAIR_MODE).fillna(False)
    protected_accounting = source_replacement | structural_zero | covered
    out["accounting_count"] = pd.to_numeric(
        out[ACCOUNTING_COUNT_COLUMN], errors="coerce"
    ).astype(float)
    if out["accounting_count"].isna().any():
        raise ValueError("accounting controls carry a non-numeric adjusted_count_ags_core")
    out = out.drop(columns=[ACCOUNTING_COUNT_COLUMN])
    out["bucket_population"] = pd.to_numeric(out["bucket_population"], errors="coerce").fillna(0.0)

    history = clean_panel_history(panel, config=config)
    levels = series_kernel_levels(history, config=config)
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
    out.loc[estimated & is_murder, "estimator"] = INFORMATION_POSTERIOR_ESTIMATOR
    out.loc[estimated & ~is_murder, "estimator"] = DYNAMIC_SHRINK_ESTIMATOR
    precise = estimated.astype(bool)
    for column in PRECISION_DIAGNOSTIC_COLUMNS:
        out[column] = pd.Series(pd.NA, index=out.index, dtype="string") if column == "signal_variance_level" else np.nan
    if precise.any():
        # Forecasts, leave-one-out peers and the signal variance are taken over every unit that can
        # be a peer (exposure, a peer-supported type), not only the units being estimated.
        pool = has_exposure & peer_supported_type
        weighted = precision_weighted_levels(
            out.loc[pool, ["jurisdiction_id", "offense", "state_abbr", "state_fips", "jurisdiction_type", "pop_band", "bucket_population"]],
            history,
            config=config,
            state_year_totals=state_year_totals,
        ).set_index(["jurisdiction_id", "offense"])
        index = pd.MultiIndex.from_frame(out.loc[precise, ["jurisdiction_id", "offense"]])
        for column in PRECISION_DIAGNOSTIC_COLUMNS:
            values = weighted[column].reindex(index)
            out.loc[precise, column] = values.to_numpy() if column == "signal_variance_level" else values.to_numpy(float)
        level = weighted["precision_weighted_count"].reindex(index).to_numpy(float)
        has_level = np.isfinite(level)
        rows = out.index[precise]
        out.loc[rows[has_level], "smoothed_count"] = level[has_level]
        out.loc[rows[has_level], "estimator"] = PRECISION_WEIGHTED_ESTIMATOR
        # Neither a forecast nor a clean current year: the E1 estimate stands (its label is kept).
        for column in PRECISION_DIAGNOSTIC_COLUMNS:
            out.loc[rows[~has_level], column] = pd.NA if column == "signal_variance_level" else np.nan
    out["current_year_check"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if precise.any():
        out.loc[precise, "current_year_check"] = current_year_check(
            out.loc[precise], confirmations=current_year_confirmations, year=int(config.year)
        )
    # First applicable reason wins, in contract order.
    out["fallback_reason"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[source_replacement, "fallback_reason"] = SOURCE_REPLACEMENT_PROTECTED_REASON
    out.loc[structural_zero, "fallback_reason"] = STRUCTURAL_ZERO_PROTECTED_REASON
    out.loc[covered, "fallback_reason"] = COVERED_PROTECTED_REASON
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
    out = _apply_state_anchor_calibration(out, state_anchor=state_anchor)
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
    state_factor = pd.to_numeric(smoothed["state_calibration_factor"], errors="coerce")
    if state_factor.isna().any() or state_factor.lt(0.0).any():
        raise ValueError("smoothed controls carry invalid state calibration factors")
    recomposed_calibrated = precalibration * calibration_factor * state_factor
    if (recomposed_calibrated - value).abs().gt(1e-6).any():
        raise ValueError("smoothed controls do not recompose from the national and state calibration")
    state_keys = [smoothed["state_abbr"].astype(str), smoothed["offense"].astype(str)]
    state_totals = value.groupby(state_keys).sum()
    state_anchors = pd.to_numeric(smoothed["state_anchor_count"], errors="coerce").groupby(state_keys).first()
    if (state_totals - state_anchors).abs().gt(1e-6 * state_anchors.abs().clip(lower=1.0)).any():
        raise ValueError("smoothed state totals do not equal their state anchors")
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
            PRECISION_WEIGHTED_ESTIMATOR,
            INFORMATION_POSTERIOR_ESTIMATOR,
            FALLBACK_ACCOUNTING_ESTIMATOR,
            COVERAGE_ADJUSTED_REMAINDER_ESTIMATOR,
        }
    ).all():
        raise ValueError("smoothed controls carry an unknown estimator label")
    fallback = estimators.eq(FALLBACK_ACCOUNTING_ESTIMATOR)
    dynamic = estimators.eq(DYNAMIC_SHRINK_ESTIMATOR)
    precise = estimators.eq(PRECISION_WEIGHTED_ESTIMATOR)
    information_posterior = estimators.eq(INFORMATION_POSTERIOR_ESTIMATOR)
    estimated = dynamic | precise | information_posterior
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
            COVERED_PROTECTED_REASON,
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
    shrunk = estimated & ~precise
    recomposed = (
        weight[shrunk[estimated]] * pd.to_numeric(smoothed.loc[shrunk, "ewa_count"], errors="coerce")
        + (1.0 - weight[shrunk[estimated]]) * pd.to_numeric(smoothed.loc[shrunk, "peer_predicted_count"], errors="coerce")
    )
    if (recomposed - precalibration[shrunk]).abs().gt(1e-6).any():
        raise ValueError("an estimated row does not recompose from its published components")
    current_weight = pd.to_numeric(smoothed.loc[precise, "current_year_weight"], errors="coerce")
    if current_weight.isna().any() or current_weight.lt(-tolerance).any() or current_weight.gt(1.0 + tolerance).any():
        raise ValueError("a precision-weighted row carries a current-year weight outside [0, 1]")
    weighted = (
        current_weight * pd.to_numeric(smoothed.loc[precise, "current_year_count"], errors="coerce").fillna(0.0)
        + (1.0 - current_weight) * pd.to_numeric(smoothed.loc[precise, "history_forecast_count"], errors="coerce")
    )
    if (weighted - precalibration[precise]).abs().gt(1e-6).any():
        raise ValueError("a precision-weighted row does not recompose from its forecast and current year")

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


@dataclass(frozen=True)
class AnchorInputs:
    """The controls-stage frames the state anchor reads: the level-lane admission rows, the mass
    ledger, and the agency-year observations. A level-holdout fold passes its own (fold-local) frames;
    the production build reads the controls stage's artifacts."""

    admission: pd.DataFrame
    mass_ledger: pd.DataFrame
    observations: pd.DataFrame


def load_anchor_inputs(paths: RepoPaths, *, year: int) -> AnchorInputs:
    controls = Path(paths.state_dir) / "controls"
    admission_path = controls / f"level_lane_admission_{int(year)}.parquet"
    ledger_path = controls / f"level_lane_mass_ledger_{int(year)}.parquet"
    observations_path = Path(paths.state_dir) / "observations" / "agency_year_observations.parquet"
    missing = [str(p) for p in (admission_path, ledger_path, observations_path) if not p.exists()]
    if missing:
        raise StateAnchorError(f"state anchor inputs are missing: {missing}; run build-controls first")
    return AnchorInputs(
        admission=pd.read_parquet(admission_path, columns=["ori9", "state_abbr", "year", "offense", "level_lane_hard_case_id", "level_repair_mode"]),
        mass_ledger=pd.read_parquet(ledger_path, columns=["ori9", "year", "offense", "final_control_mass"]),
        observations=pd.read_parquet(
            observations_path, columns=["ori9", "year", "source", "offense", "count", "months_reported"],
            filters=[("year", "==", int(year))],
        ),
    )


def contribution_ledger(inputs: AnchorInputs, *, year: int) -> pd.DataFrame:
    return documented_agency_corrections(
        admission=inputs.admission, mass_ledger=inputs.mass_ledger, fbi_observations=inputs.observations, year=int(year)
    )


def _resolve_mislabelled_state_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Relabel a duplicate state-year row whose counts belong to the state missing that year.

    For every (state, year) with two rows while exactly one other state lacks that year, each row is
    compared, offense by offense in log scale, with the two states' own adjacent years. A row is
    relabelled only when one row matches the missing state and the other matches the labelled state on
    every offense; otherwise the duplicate is an error. Independent of row order.
    """
    offense_columns = [c for c in CDE_OFFENSE_MAP.values() if c in frame.columns]
    values = frame[offense_columns].apply(lambda c: pd.to_numeric(c.astype(str).str.replace(",", "").str.strip(), errors="coerce"))
    year = pd.to_numeric(frame["year"], errors="coerce")
    out = frame.copy()
    states = set(frame["state_abbr"].dropna().astype(str))
    keys = frame.assign(_year=year).groupby(["state_abbr", "_year"]).size()
    for (state, y), n in keys[keys.gt(1)].items():
        if n != 2:
            raise ValueError(f"CDE estimates: {n} rows for {state} {int(y)}")
        present = set(frame.loc[year.eq(y), "state_abbr"].astype(str))
        missing = sorted(states - present)
        if len(missing) != 1:
            raise ValueError(f"CDE estimates: duplicate {state} {int(y)} rows with {len(missing)} states missing that year")
        other = missing[0]

        def neighbours(st: str) -> np.ndarray:
            rows = values[frame["state_abbr"].astype(str).eq(st) & year.isin([y - 1, y + 1])]
            return np.log(rows.mean(axis=0).to_numpy(float).clip(min=1.0))

        own, theirs = neighbours(str(state)), neighbours(other)
        idx = frame.index[frame["state_abbr"].astype(str).eq(str(state)) & year.eq(y)]
        logs = {i: np.log(values.loc[i].to_numpy(float).clip(min=1.0)) for i in idx}
        to_other = [i for i in idx if (np.abs(logs[i] - theirs) < np.abs(logs[i] - own)).all()]
        to_own = [i for i in idx if (np.abs(logs[i] - own) < np.abs(logs[i] - theirs)).all()]
        if len(to_other) != 1 or len(to_own) != 1 or to_other[0] == to_own[0]:
            raise ValueError(f"CDE estimates: cannot attribute the duplicate {state} {int(y)} rows by signature")
        out.loc[to_other[0], "state_abbr"] = other
    return out


def _load_cde_state_years(paths: RepoPaths, *, config: SmoothedControlConfig) -> pd.DataFrame:
    """FBI CDE state estimates for every panel year, long (state_abbr, offense, year, cde_state_count).

    Duplicate state-year rows are resolved by signature (`_resolve_mislabelled_state_rows`): the
    1979-2025 file labels North Dakota's 2022 row "NC" (NC has two 2022 rows, ND none).
    """
    path = _cde_estimates_path(paths, year=int(config.year))
    if not path.exists():
        return pd.DataFrame(columns=["state_abbr", "offense", "year", "cde_state_count"])
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    frame["state_abbr"] = frame["state_abbr"].astype("string").str.strip().str.upper()
    frame = _resolve_mislabelled_state_rows(frame)
    year = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame[year.ge(int(config.panel_year_start)) & year.le(int(config.year))].copy()
    frame = frame[frame["state_abbr"].notna() & ~frame["state_abbr"].isin({"<NA>", "NAN", *config.exclude_scope_state_abbrs})]
    rows = []
    for offense, column in CDE_OFFENSE_MAP.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        rows.append(pd.DataFrame({"state_abbr": frame["state_abbr"].astype(str).to_numpy(), "offense": offense,
                                  "year": pd.to_numeric(frame["year"], errors="coerce").astype(int).to_numpy(),
                                  "cde_state_count": values.to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["state_abbr", "offense", "year", "cde_state_count"])


def _load_cde_states_data_year(paths: RepoPaths, *, config: SmoothedControlConfig) -> pd.DataFrame:
    """FBI CDE state estimates for the data year, long by state and offense (CONUS + DC)."""
    path = _cde_estimates_path(paths, year=int(config.year))
    if not path.exists():
        return pd.DataFrame(columns=["state_abbr", "offense", "cde_state_count"])
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame[pd.to_numeric(frame["year"], errors="coerce").eq(int(config.year))].copy()
    frame["state_abbr"] = frame["state_abbr"].astype("string").str.strip().str.upper()
    frame = frame[
        frame["state_abbr"].notna()
        & ~frame["state_abbr"].isin({"<NA>", "NAN", *config.exclude_scope_state_abbrs})
    ]
    rows = []
    for offense, column in CDE_OFFENSE_MAP.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        rows.extend(
            {"state_abbr": str(st), "offense": offense, "cde_state_count": float(v)}
            for st, v in zip(frame["state_abbr"], values, strict=True)
        )
    return pd.DataFrame(rows, columns=["state_abbr", "offense", "cde_state_count"])


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
        entry["accounting_over_anchor"] = (
            entry["accounting_total"] / entry["national_anchor_count"]
            if entry["national_anchor_count"]
            else None
        )
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
        "estimator": PRECISION_WEIGHTED_ESTIMATOR,
        "estimator_source": "E1 round 2 kernel and peer shrinkage as the history forecast; precision weighting of the target year (2025.3, pm/e3/evidence/s6e_*)",
        "national_anchor_source": NATIONAL_ANCHOR_SOURCE,
        "shrinkage_k": float(config.shrinkage_k),
        "min_clean_years": int(config.min_clean_years),
        "full_year_admitted_min_share": float(FULL_YEAR_ADMITTED_MIN_SHARE),
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
    *,
    paths: RepoPaths,
    config: SmoothedControlConfig = SmoothedControlConfig(),
    accounting: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
    anchor_inputs: AnchorInputs | None = None,
) -> tuple[pd.DataFrame, dict]:
    """`accounting`, `panel` and `anchor_inputs` default to the controls stage's artifacts; the level
    holdout passes a fold's own frames (all three, or the anchor would read production's ledger)."""
    if accounting is None or panel is None:
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
        accounting = pd.read_parquet(accounting_path) if accounting is None else accounting
        panel = pd.read_parquet(panel_path) if panel is None else panel
    cde_history = _load_cde_history(paths, config=config)
    national_anchor = national_anchor_from_cde_data_year(
        cde_history,
        config=config,
    )
    national_anchor = national_anchor[
        national_anchor["offense"].astype(str).isin(set(accounting["offense"].astype(str)))
    ].reset_index(drop=True)
    if anchor_inputs is None:
        anchor_inputs = load_anchor_inputs(paths, year=int(config.year))
    contributions = contribution_ledger(anchor_inputs, year=int(config.year))
    state_anchor = state_anchor_from_cde_data_year(
        _load_cde_states_data_year(paths, config=config),
        accounting=accounting,
        corrections=contributions,
        exceptions=load_state_anchor_exceptions(paths, year=int(config.year)),
    )
    # The national level is the sum of the state targets; the data-year CDE national estimate
    # stays the comparator the accounting-agreement check reads.
    summed = state_anchor.groupby("offense", as_index=False)["state_anchor_count"].sum()
    national_anchor = national_anchor.drop(columns="national_anchor_count").merge(
        summed.rename(columns={"state_anchor_count": "national_anchor_count"}), on="offense", how="left"
    )
    smoothed = build_smoothed_controls(
        accounting=accounting,
        panel=panel,
        config=config,
        national_anchor=national_anchor,
        state_anchor=state_anchor,
        state_year_totals=_load_cde_state_years(paths, config=config),
        current_year_confirmations=load_current_year_confirmations(paths),
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
    summary["state_anchor_contributions"] = contributions.astype(object).where(contributions.notna(), None).to_dict(orient="records")
    sensitivity = state_anchor[state_anchor["documented_correction_agencies"].gt(0) | state_anchor["state_anchor_source"].eq(STATE_ANCHOR_EXCEPTION_SOURCE)]
    summary["state_anchor_sensitivity"] = sensitivity[
        ["state_abbr", "offense", "cde_state_count", "documented_correction_count", "model_dependent_correction_count",
         "state_anchor_count", "state_anchor_count_external_only", "state_anchor_source"]
    ].astype(object).to_dict(orient="records")
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
        assert_national_anchor_agrees_with_accounting(summary)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        smoothed.to_parquet(artifact, index=False)
        pd.DataFrame(summary["state_anchor_contributions"], columns=CONTRIBUTION_LEDGER_COLUMNS).to_csv(
            artifact.parent / f"state_anchor_contributions_{int(config.year)}.csv", index=False)
        pd.DataFrame(summary["state_anchor_sensitivity"]).to_csv(
            artifact.parent / f"state_anchor_sensitivity_{int(config.year)}.csv", index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        write_dependency_stamp(
            artifact, smoothed_controls_dependency_paths(paths, year=int(config.year))
        )
    return artifact, summary
