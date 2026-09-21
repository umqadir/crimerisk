"""Held-out evaluation of the level lane's imputation path.

What it measures
----------------
Take every agency-offense that filed a complete target year, hide it, and make the
production repair path guess it back. Nothing about the estimator is re-implemented
here: the masked panel goes through `build_agency_allocation_target_estimates`, the same
function allocation and jurisdiction controls call, so the rungs exercised are the real
ones -- decayed own history blended against the pooled silent-unit prior, the peer rate
that backs that prior, and the no-history fallback.

Two masks, because the lane fails in two different ways:

`refused`  the row is present and the admission gate refuses it (Cicero's shape). The
           ladder sees a target-year row that is not usable and falls to history.
`silent`   the agency-year is absent entirely (Jacksonville's and Kansas City's shape).
           The ladder sees no current row at all; an agency with no usable history and
           no explicit footprint gets no estimate row, which is what sends its
           jurisdiction to a pooled silent-unit control downstream.
`corrupted` the row is present, wrong, and syntactically perfect (Cicero's shape). Four
           offenses are collapsed together and the vector is handed to the admission
           gate intact. This is the only arm that can score an admission rule: hiding a
           clean row measures the repair ladder, not the decision to distrust a filed
           one.

The state-remainder lane is scored separately and on its own terms: leave-one-state-out
on the national remainder-to-municipal rate ratio, predicting each state's remainder
count from the other states' ratio and its own municipal rate.

This is a yardstick. Nothing in it is fitted, and no rule may be tuned against it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crime import OFFENSES_7  # noqa: E402
from crimerisk.level_lane import apply_level_lane_admission  # noqa: E402
from crimerisk.paths import get_paths  # noqa: E402
from crimerisk.trend_fills import (  # noqa: E402
    apply_masked_gap_reclassification,
    apply_stage1_adjudicated_usability,
    build_agency_allocation_target_estimates,
    build_agency_trend_fill_panel,
    build_masked_gap_flags,
)
from crimerisk.stage1_adjudications import build_usability_directives  # noqa: E402


SIZE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("00_under_2500", 0.0, 2_500.0),
    ("01_2500_10k", 2_500.0, 10_000.0),
    ("02_10k_25k", 10_000.0, 25_000.0),
    ("03_25k_100k", 25_000.0, 100_000.0),
    ("04_100k_250k", 100_000.0, 250_000.0),
    ("05_250k_plus", 250_000.0, float("inf")),
)

ARMS = ("refused", "silent", "corrupted")

# The corrupted arm reproduces the shape the admission gate misses: several offenses
# collapsing together by an amount no single peer-percentile test objects to. The
# factor and the offense set are fixed, so the arm is the same experiment in every
# configuration; they are a defect description, not a tuned parameter.
CORRUPTION_FACTOR = 0.35
CORRUPTED_OFFENSES = ("burglary", "robbery", "aggravated_assault", "motor_vehicle_theft")


@dataclass(frozen=True)
class HoldoutConfig:
    year: int
    folds: int
    mask_share: float
    seed: int
    arms: tuple[str, ...]


def _size_band(population: pd.Series) -> pd.Series:
    values = pd.to_numeric(population, errors="coerce").fillna(0.0)
    out = pd.Series("unknown", index=values.index, dtype=object)
    for label, low, high in SIZE_BANDS:
        out.loc[values.ge(low) & values.lt(high)] = label
    return out


def _stable_fold(ori9: pd.Series, *, folds: int, seed: int) -> pd.Series:
    def one(value: object) -> int:
        digest = hashlib.blake2b(
            f"{seed}:{value}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") % max(int(folds), 1)

    return ori9.astype(str).map(one)


# ---------------------------------------------------------------------------
# panel construction
# ---------------------------------------------------------------------------


def load_preferred_panel(
    paths, *, year: int, panel_year_end: int, cache_dir: Path, refresh: bool
) -> pd.DataFrame:
    """The preferred-observation panel, before admission. Cached across rule configs.

    The panel is always built to the edition's own last year, never to the holdout
    target year: the reporting-regime stage rebuilds the observation panel to whatever
    year it is asked for, so asking it for 2024 would silently drop the edition's 2025
    rows from `state/observations`. The span is trimmed to the target year afterwards.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"preferred_panel_{year}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    panel = build_agency_trend_fill_panel(
        paths=paths, year_start=2018, year_end=max(int(year), int(panel_year_end))
    )
    panel = panel[pd.to_numeric(panel["year"], errors="coerce").le(int(year))].copy()
    panel.to_parquet(cache_path, index=False)
    return panel


def admit_panel(paths, *, panel: pd.DataFrame, year: int) -> pd.DataFrame:
    """Admission, masked-gap reclassification and adjudicated usability, in order.

    The same sequence `build_agency_allocation_target_estimates` runs when it is handed
    a raw panel. Running it here, per rule configuration, keeps the admission rules
    (a)-(c) inside what the harness measures while the panel build above stays shared.
    """
    admitted = apply_level_lane_admission(
        panel, paths=paths, target_year=int(year)
    ).panel
    masked_gap_flags = build_masked_gap_flags(
        paths, target_year=int(year), agency_panel=admitted
    )
    admitted = apply_masked_gap_reclassification(
        admitted, target_year=int(year), masked_gap_flags=masked_gap_flags
    )
    admitted, _counts = apply_stage1_adjudicated_usability(
        admitted,
        target_year=int(year),
        directives=build_usability_directives(paths, target_year=int(year)),
    )
    return admitted


@contextlib.contextmanager
def _rules(value: str):
    """Run a block under an explicit rule configuration, then put the old one back."""
    previous = os.environ.get("CRIMERISK_LEVEL_RULES")
    os.environ["CRIMERISK_LEVEL_RULES"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CRIMERISK_LEVEL_RULES", None)
        else:
            os.environ["CRIMERISK_LEVEL_RULES"] = previous


def truth_rows(panel: pd.DataFrame, *, year: int, paths_state_dir: str = "state") -> pd.DataFrame:
    target = panel[pd.to_numeric(panel["year"], errors="coerce").eq(int(year))].copy()
    months = pd.to_numeric(target["preferred_months_reported"], errors="coerce")
    keep = (
        target["usable_as_observed"].fillna(False).astype(bool)
        & ~target["current_row_is_true_partial"].fillna(False).astype(bool)
        & months.ge(12.0)
        & target["level1_admission_status"].astype("string").eq("valid_complete_year")
        & target["level2_semantic_status"].astype("string").eq("definitionally_complete")
        & pd.to_numeric(target["preferred_count"], errors="coerce").notna()
        & target["offense"].isin(OFFENSES_7)
    )
    value_column = (
        "level_lane_original_count"
        if "level_lane_original_count" in target.columns
        else "preferred_count"
    )
    out = target.loc[keep, ["ori9", "offense", "state_abbr", "state_fips", value_column]].copy()
    out = out.rename(columns={value_column: "observed_count"})
    population = (
        target.groupby("ori9", dropna=False)["population"].max()
        if "population" in target.columns
        else pd.Series(dtype=float)
    )
    out["population"] = out["ori9"].map(population)
    # The preferred-observation panel is offense-centric and does not carry population,
    # so the size band comes from the canonical agency master, exactly as the level
    # lane's own plausibility screen resolves it.
    master_path = Path(paths_state_dir) / "reference" / "agency_master.parquet"
    if master_path.exists():
        master = pd.read_parquet(
            master_path, columns=["ori9", "population_latest_nibrs"]
        ).drop_duplicates("ori9")
        resolved = out["ori9"].astype(str).map(
            master.set_index("ori9")["population_latest_nibrs"]
        )
        out["population"] = pd.to_numeric(out["population"], errors="coerce").fillna(
            pd.to_numeric(resolved, errors="coerce")
        )
    # Only agencies whose whole seven-offense vector is a clean complete year are
    # masked: hiding one offense of a filed vector is not a case the lane ever sees.
    complete = out.groupby("ori9")["offense"].nunique().eq(len(OFFENSES_7))
    out = out[out["ori9"].map(complete).fillna(False)]
    return out.reset_index(drop=True)


PROTECTED_CONFIGS: tuple[tuple[str, bool], ...] = (
    ("level_lane_admission_registry.csv", True),
    ("level_lane_hard_evidence.csv", True),
    ("level_lane_truth_panel.csv", True),
    # The succession registry verifies that a successor ORI carries a target-year
    # report before it may suppress the retired one. Masking either side of a
    # migration fails that gate, which is the registry doing its job, so both sides
    # are out of scope for the holdout.
    ("stage1_adjudications/ori_successions.csv", False),
    ("stage1_adjudications/twins_adjudicated.csv", False),
    ("stage1_adjudications/zero_missing_adjudicated.csv", False),
    ("stage1_adjudications/token_reporters_adjudicated.csv", False),
)

ORI_COLUMN_CANDIDATES = (
    "ori9", "ori", "superseded_ori", "successor_ori", "superseded_ori9",
    "successor_ori9", "retired_ori9", "live_ori9", "primary_ori9", "primary_ori",
    "variant_ori9", "variant_ori", "old_ori9", "new_ori9", "twin_ori", "twin_ori9",
)


def protected_oris(paths, *, year: int) -> set[str]:
    """ORIs an explicit ruling names; never masked.

    A registry entry is a statement about a specific agency-year, and several of them
    fail closed when the row they name is not there. Hiding one measures the registry,
    not the lane.
    """
    out: set[str] = set()
    for name, year_scoped in PROTECTED_CONFIGS:
        path = Path(paths.repo_root) / "configs" / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype="string", comment="#")
        if year_scoped and "year" in frame.columns:
            frame = frame[pd.to_numeric(frame["year"], errors="coerce").eq(int(year))]
        for column in ORI_COLUMN_CANDIDATES:
            if column in frame.columns:
                out |= set(frame[column].dropna().astype(str).str.upper().str.strip())
    return {ori for ori in out if ori and ori.lower() != "nan"}


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------


REFUSAL_FIELDS = {
    "level1_admission_status": "coverage_defective",
    "level2_semantic_status": "semantically_unusable",
    "level_admission_reason": "holdout_masked",
    "level_repair_mode": "decayed_own_history_or_pooled",
    "level_policy_resolution": "repair_ladder",
}


def mask_panel(
    panel: pd.DataFrame, *, year: int, masked_oris: set[str], arm: str
) -> pd.DataFrame:
    target = pd.to_numeric(panel["year"], errors="coerce").eq(int(year))
    hit = target & panel["ori9"].astype(str).isin(masked_oris)
    if arm == "silent":
        return panel.loc[~hit].reset_index(drop=True)
    out = panel.copy()
    for column, value in REFUSAL_FIELDS.items():
        if column in out.columns:
            out.loc[hit, column] = value
    out.loc[hit, "usable_as_observed"] = False
    out.loc[hit, "current_row_is_true_partial"] = False
    out.loc[hit, "preferred_count"] = 0.0
    return out


def corrupt_panel(
    panel: pd.DataFrame, *, year: int, masked_oris: set[str]
) -> pd.DataFrame:
    """Collapse four offenses of the target-year vector, leaving it otherwise intact."""
    out = panel.copy()
    target = pd.to_numeric(out["year"], errors="coerce").eq(int(year))
    hit = (
        target
        & out["ori9"].astype(str).isin(masked_oris)
        & out["offense"].astype("string").isin(CORRUPTED_OFFENSES)
    )
    out.loc[hit, "preferred_count"] = (
        pd.to_numeric(out.loc[hit, "preferred_count"], errors="coerce") * CORRUPTION_FACTOR
    ).round()
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _poisson_deviance(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    y = np.clip(observed.astype(float), 0.0, None)
    mu = np.clip(predicted.astype(float), 1e-9, None)
    term = np.zeros_like(y)
    positive = y > 0
    term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    return 2.0 * (term - (y - mu))


def score(frame: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(frame["observed_count"], errors="coerce").to_numpy(dtype=float)
    mu = pd.to_numeric(frame["predicted_count"], errors="coerce").to_numpy(dtype=float)
    out = frame.copy()
    out["poisson_deviance"] = _poisson_deviance(y, mu)
    out["abs_log_ratio"] = np.abs(np.log((mu + 0.5) / (y + 0.5)))
    # Within +-25% is asked of the value the lane is trying to reproduce. A zero-count
    # truth has no 25% band, so it is scored on an absolute half-count tolerance and
    # reported separately through `observed_positive`.
    within = np.where(
        y > 0,
        (mu >= 0.75 * y) & (mu <= 1.25 * y),
        mu <= 0.5,
    )
    out["within_25pct"] = within
    out["observed_positive"] = y > 0
    return out


def summarise(scored: pd.DataFrame, *, by: list[str]) -> pd.DataFrame:
    grouped = scored.groupby(by, dropna=False)
    out = grouped.agg(
        cases=("observed_count", "size"),
        observed_total=("observed_count", "sum"),
        predicted_total=("predicted_count", "sum"),
        mean_poisson_deviance=("poisson_deviance", "mean"),
        median_abs_log_ratio=("abs_log_ratio", "median"),
        share_within_25pct=("within_25pct", "mean"),
        observed_positive_cases=("observed_positive", "sum"),
    ).reset_index()
    out["total_ratio"] = out["predicted_total"] / out["observed_total"].replace(0.0, np.nan)
    return out


# ---------------------------------------------------------------------------
# agency lane
# ---------------------------------------------------------------------------


def _score_masked(
    *,
    truth: pd.DataFrame,
    estimates: pd.DataFrame,
    masked: set[str],
    arm: str,
    fold: int,
    label: str,
) -> pd.DataFrame:
    scope = truth[truth["ori9"].isin(masked)]
    if arm == "corrupted":
        scope = scope[scope["offense"].isin(CORRUPTED_OFFENSES)]
    joined = scope.merge(
        estimates[["ori9", "offense", "estimated_count", "agency_estimate_source"]],
        on=["ori9", "offense"],
        how="left",
    )
    # An agency the ladder declines to estimate is not absent from the release: its
    # jurisdiction falls to a pooled silent-unit control downstream. It is scored as
    # the pooled prior it triggers, so the arm's number is the number the published
    # surface would carry.
    joined["dropped_by_ladder"] = joined["estimated_count"].isna()
    joined["agency_estimate_source"] = (
        joined["agency_estimate_source"].astype("string").fillna("dropped_to_pooled_silent_unit")
    )
    joined["predicted_count"] = pd.to_numeric(joined["estimated_count"], errors="coerce")
    joined["arm"] = arm
    joined["fold"] = fold
    joined["config"] = label
    return joined


def run_agency_holdout_combined(
    *,
    paths,
    panel: pd.DataFrame,
    preferred: pd.DataFrame,
    config: HoldoutConfig,
    truth: pd.DataFrame,
    eligible_oris: list[str],
    label: str,
) -> pd.DataFrame:
    """One estimator pass, three arms, disjoint agency sets.

    The estimator is the whole cost of a run and it re-estimates every agency whatever
    is masked, so running the arms on separate folds of one panel measures the same
    thing three times more cheaply. The arms stay independent because no agency is in
    two of them, and every configuration masks exactly the same agencies, so the
    comparison between configurations is unaffected by the sharing.
    """
    folds = _stable_fold(pd.Series(eligible_oris), folds=config.folds, seed=config.seed)
    fold_of = dict(zip(eligible_oris, folds.tolist()))
    arm_folds = {arm: index for index, arm in enumerate(config.arms)}
    masked_by_arm = {
        arm: {ori for ori, value in fold_of.items() if value == fold}
        for arm, fold in arm_folds.items()
    }
    started = time.time()
    work = preferred
    if "corrupted" in masked_by_arm:
        work = corrupt_panel(
            work, year=config.year, masked_oris=masked_by_arm["corrupted"]
        )
    admitted = admit_panel(paths, panel=work, year=config.year)
    if "refused" in masked_by_arm:
        admitted = mask_panel(
            admitted, year=config.year, masked_oris=masked_by_arm["refused"], arm="refused"
        )
    if "silent" in masked_by_arm:
        admitted = mask_panel(
            admitted, year=config.year, masked_oris=masked_by_arm["silent"], arm="silent"
        )
    estimates = build_agency_allocation_target_estimates(
        paths=paths, year=int(config.year), agency_panel=admitted
    )
    results = [
        _score_masked(
            truth=truth,
            estimates=estimates,
            masked=masked,
            arm=arm,
            fold=arm_folds[arm],
            label=label,
        )
        for arm, masked in masked_by_arm.items()
        if masked
    ]
    print(
        f"  [{label}] combined arms {list(masked_by_arm)} "
        f"masked={sum(len(v) for v in masked_by_arm.values())} agencies "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def run_agency_holdout(
    *,
    paths,
    panel: pd.DataFrame,
    preferred: pd.DataFrame,
    config: HoldoutConfig,
    truth: pd.DataFrame,
    eligible_oris: list[str],
    label: str,
) -> pd.DataFrame:
    folds = _stable_fold(pd.Series(eligible_oris), folds=config.folds, seed=config.seed)
    fold_of = dict(zip(eligible_oris, folds.tolist()))
    n_folds = max(1, int(round(config.folds * config.mask_share)))
    results: list[pd.DataFrame] = []
    for arm in config.arms:
        for fold in range(n_folds):
            masked = {ori for ori, value in fold_of.items() if value == fold}
            if not masked:
                continue
            started = time.time()
            if arm == "corrupted":
                # Corruption happens before admission, because whether the gate sees
                # through it is the thing being measured.
                masked_panel = admit_panel(
                    paths,
                    panel=corrupt_panel(preferred, year=config.year, masked_oris=masked),
                    year=config.year,
                )
            else:
                masked_panel = mask_panel(
                    panel, year=config.year, masked_oris=masked, arm=arm
                )
            estimates = build_agency_allocation_target_estimates(
                paths=paths, year=int(config.year), agency_panel=masked_panel
            )
            scope = truth[truth["ori9"].isin(masked)]
            if arm == "corrupted":
                scope = scope[scope["offense"].isin(CORRUPTED_OFFENSES)]
            joined = scope.merge(
                estimates[
                    ["ori9", "offense", "estimated_count", "agency_estimate_source"]
                ],
                on=["ori9", "offense"],
                how="left",
            )
            # An agency the ladder declines to estimate is not absent from the release:
            # its jurisdiction falls to a pooled silent-unit control downstream. Score
            # it as the pooled prior it triggers, so the arm's number is the number the
            # published surface would carry.
            joined["dropped_by_ladder"] = joined["estimated_count"].isna()
            joined["agency_estimate_source"] = (
                joined["agency_estimate_source"]
                .astype("string")
                .fillna("dropped_to_pooled_silent_unit")
            )
            joined["predicted_count"] = pd.to_numeric(
                joined["estimated_count"], errors="coerce"
            )
            joined["arm"] = arm
            joined["fold"] = fold
            joined["config"] = label
            results.append(joined)
            print(
                f"  [{label}] arm={arm} fold={fold} masked_agencies={len(masked)} "
                f"rows={len(joined)} dropped={int(joined['dropped_by_ladder'].sum())} "
                f"{time.time() - started:.0f}s",
                flush=True,
            )
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def pooled_silent_unit_backfill(
    scored: pd.DataFrame, *, panel: pd.DataFrame, year: int
) -> pd.DataFrame:
    """Give a ladder-dropped agency the pooled rate its jurisdiction would inherit.

    The rate is the target-year complete-reporter rate for the agency's own state, type
    and offense -- the same pool the ladder itself uses -- so a dropped row is scored
    against the control its absence actually produces rather than silently excluded.
    """
    out = scored.copy()
    missing = out["predicted_count"].isna()
    if not missing.any():
        return out
    target = panel[pd.to_numeric(panel["year"], errors="coerce").eq(int(year))]
    pool = target[
        target["usable_as_observed"].fillna(False).astype(bool)
        & target["level1_admission_status"].astype("string").eq("valid_complete_year")
    ].copy()
    pool["_count"] = pd.to_numeric(pool["preferred_count"], errors="coerce").fillna(0.0)
    pool["_population"] = pd.to_numeric(
        pool.get("population", pd.Series(0.0, index=pool.index)), errors="coerce"
    ).fillna(0.0)
    by_state = pool.groupby(["state_fips", "offense"], dropna=False).agg(
        y=("_count", "sum"), e=("_population", "sum")
    )
    by_state["rate"] = by_state["y"] / by_state["e"].replace(0.0, np.nan)
    national = pool.groupby("offense", dropna=False).agg(
        y=("_count", "sum"), e=("_population", "sum")
    )
    national["rate"] = national["y"] / national["e"].replace(0.0, np.nan)
    keys = pd.MultiIndex.from_arrays(
        [
            out.loc[missing, "state_fips"].astype("string").str.zfill(2),
            out.loc[missing, "offense"].astype("string"),
        ]
    )
    rate = pd.Series(by_state["rate"].reindex(keys).to_numpy(), index=out.index[missing])
    rate = rate.fillna(out.loc[missing, "offense"].map(national["rate"]))
    out.loc[missing, "predicted_count"] = (
        rate.fillna(0.0) * pd.to_numeric(out.loc[missing, "population"], errors="coerce").fillna(0.0)
    )
    return out


# ---------------------------------------------------------------------------
# state remainder lane
# ---------------------------------------------------------------------------


def run_remainder_holdout(paths, *, year: int) -> pd.DataFrame:
    """Leave-one-state-out on the remainder-to-municipal rate ratio.

    The production estimator learns one national ratio per offense from the remainder
    lanes it considers reliable, then lifts every under-covered lane to that ratio times
    its own state's municipal rate. Refit the ratio without a state and ask how well it
    reproduces that state's own reliable remainder count.
    """
    path = Path(paths.state_dir) / "controls" / f"jurisdiction_controls_smoothed_{year}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    needed = {
        "jurisdiction_id",
        "state_fips",
        "offense",
        "jurisdiction_type",
        "bucket_population",
        "smoothed_count",
        "estimator",
    }
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    frame["jurisdiction_type"] = frame["jurisdiction_type"].astype("string")
    remainder = frame[frame["jurisdiction_type"].eq("state_nonmunicipal_remainder")].copy()
    municipal = frame[~frame["jurisdiction_type"].eq("state_nonmunicipal_remainder")].copy()
    if remainder.empty or municipal.empty:
        return pd.DataFrame()
    muni = (
        municipal.groupby(["state_fips", "offense"], dropna=False)
        .agg(
            municipal_count=("smoothed_count", "sum"),
            municipal_population=("bucket_population", "sum"),
        )
        .reset_index()
    )
    muni["municipal_rate"] = muni["municipal_count"] / muni["municipal_population"].replace(0.0, np.nan)

    reliable = remainder[
        ~remainder["estimator"].astype("string").eq("coverage_adjusted_remainder")
        & pd.to_numeric(remainder["bucket_population"], errors="coerce").gt(0.0)
    ].merge(muni, on=["state_fips", "offense"], how="left")
    reliable = reliable[reliable["municipal_rate"].gt(0.0)].copy()
    if reliable.empty:
        return pd.DataFrame()
    reliable["remainder_count"] = pd.to_numeric(reliable["smoothed_count"], errors="coerce")
    reliable["remainder_population"] = pd.to_numeric(
        reliable["bucket_population"], errors="coerce"
    )

    rows: list[dict[str, object]] = []
    for offense, group in reliable.groupby("offense", dropna=False):
        total_r = group["remainder_count"].sum()
        total_rp = group["remainder_population"].sum()
        total_m = group["municipal_count"].sum()
        total_mp = group["municipal_population"].sum()
        for row in group.itertuples(index=False):
            r = total_r - row.remainder_count
            rp = total_rp - row.remainder_population
            m = total_m - row.municipal_count
            mp = total_mp - row.municipal_population
            if rp <= 0 or mp <= 0 or m <= 0:
                continue
            ratio = (r / rp) / (m / mp)
            predicted = row.remainder_population * row.municipal_rate * ratio
            rows.append(
                {
                    "state_fips": row.state_fips,
                    "offense": str(offense),
                    "jurisdiction_id": row.jurisdiction_id,
                    "observed_count": float(row.remainder_count),
                    "predicted_count": float(predicted),
                    "remainder_population": float(row.remainder_population),
                    "leave_one_out_ratio": float(ratio),
                }
            )
    if not rows:
        return pd.DataFrame()
    return score(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _markdown_table(frame: pd.DataFrame) -> str:
    """A pipe table without pulling in a formatting dependency."""
    if frame.empty:
        return "_(empty)_"
    def cell(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, float):
            return f"{value:,.4f}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        return str(value)

    columns = list(frame.columns)
    rows = [[cell(v) for v in record] for record in frame.itertuples(index=False)]
    widths = [
        max(len(str(columns[i])), *(len(row[i]) for row in rows)) if rows else len(str(columns[i]))
        for i in range(len(columns))
    ]
    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"
    out = [line([str(c) for c in columns]), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def write_report(
    *,
    out_dir: Path,
    agency_scored: pd.DataFrame,
    remainder_scored: pd.DataFrame,
    config: HoldoutConfig,
    meta: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []

    def add(frame: pd.DataFrame, *, lane: str, stratum: str, by: list[str]) -> None:
        if frame.empty:
            return
        table = summarise(frame, by=by)
        table.insert(0, "stratum", stratum)
        table.insert(0, "lane", lane)
        for column in ("config", "arm", "repair_mode", "size_band", "offense", "state_fips"):
            if column not in table.columns:
                table[column] = pd.NA
        tables.append(table)

    if not agency_scored.empty:
        agency_scored = agency_scored.copy()
        agency_scored["repair_mode"] = agency_scored["agency_estimate_source"]
        agency_scored["size_band"] = _size_band(agency_scored["population"])
        add(agency_scored, lane="agency", stratum="overall", by=["config", "arm"])
        add(agency_scored, lane="agency", stratum="repair_mode", by=["config", "arm", "repair_mode"])
        add(agency_scored, lane="agency", stratum="size_band", by=["config", "arm", "size_band"])
        add(agency_scored, lane="agency", stratum="offense", by=["config", "arm", "offense"])
    if not remainder_scored.empty:
        remainder_scored = remainder_scored.copy()
        remainder_scored["config"] = meta.get("config_label", "baseline")
        remainder_scored["arm"] = "leave_state_out"
        add(remainder_scored, lane="state_remainder", stratum="overall", by=["config", "arm"])
        add(
            remainder_scored,
            lane="state_remainder",
            stratum="offense",
            by=["config", "arm", "offense"],
        )

    if not tables:
        return
    columns = [
        "lane", "stratum", "config", "arm", "repair_mode", "size_band", "offense",
        "cases", "observed_total", "predicted_total", "total_ratio",
        "mean_poisson_deviance", "median_abs_log_ratio", "share_within_25pct",
        "observed_positive_cases",
    ]
    results = pd.concat(tables, ignore_index=True)
    results = results[[c for c in columns if c in results.columns]]
    results_path = out_dir / "results.csv"
    if results_path.exists() and meta.get("append"):
        previous = pd.read_csv(results_path)
        results = pd.concat([previous, results], ignore_index=True)
        results = results.drop_duplicates(
            subset=["lane", "stratum", "config", "arm", "repair_mode", "size_band", "offense"],
            keep="last",
        )
    results.to_csv(results_path, index=False)
    if not agency_scored.empty:
        agency_scored.to_parquet(out_dir / "agency_cases.parquet", index=False)
        # One file per configuration, so a sweep keeps every case it scored and the
        # summary tables can be rebuilt without re-running the estimator.
        label = str(meta.get("config_label", "baseline"))
        agency_scored.to_parquet(out_dir / f"agency_cases_{label}.parquet", index=False)
    if not remainder_scored.empty:
        remainder_scored.to_parquet(out_dir / "remainder_cases.parquet", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps({**meta, "config": config.__dict__}, indent=2, default=str)
    )

    lines = [
        "# Level-lane holdout",
        "",
        f"Target year {config.year}. Folds {config.folds}, mask share {config.mask_share:.2f}, "
        f"seed {config.seed}.",
        "",
        "## Overall",
        "",
    ]
    overall = results[results["stratum"].eq("overall")]
    lines.append(_markdown_table(overall))
    for stratum, title in (
        ("repair_mode", "By repair mode"),
        ("size_band", "By agency size band"),
        ("offense", "By offense"),
    ):
        part = results[results["stratum"].eq(stratum)]
        if part.empty:
            continue
        lines += ["", f"## {title}", "", _markdown_table(part)]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level-lane holdout harness.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--panel-year-end",
        type=int,
        default=2025,
        help="The edition's own last year; the panel is always built to it.",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--mask-share",
        type=float,
        default=0.2,
        help="Share of eligible agencies masked per run, as a multiple of 1/folds.",
    )
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--out-dir", type=Path, default=Path("state/eval/level_v1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("state/eval/level_v1/cache"))
    parser.add_argument("--refresh-panel", action="store_true")
    parser.add_argument("--skip-remainder", action="store_true")
    parser.add_argument(
        "--separate-arms",
        action="store_true",
        help="Run one estimator pass per arm instead of sharing one across folds.",
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit-agencies", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    config = HoldoutConfig(
        year=int(args.year),
        folds=int(args.folds),
        mask_share=float(args.mask_share),
        seed=int(args.seed),
        arms=tuple(args.arms),
    )
    out_dir = paths.repo_root / args.out_dir
    cache_dir = paths.repo_root / args.cache_dir

    started = time.time()
    preferred = load_preferred_panel(
        paths,
        year=config.year,
        panel_year_end=int(args.panel_year_end),
        cache_dir=cache_dir,
        refresh=bool(args.refresh_panel),
    )
    # The truth universe is fixed by the unmodified lane, not by the configuration
    # under test: a rule that refuses more rows must be scored on the same cases as
    # one that refuses fewer.
    truth_cache = cache_dir / f"truth_{config.year}.parquet"
    if truth_cache.exists() and not args.refresh_panel:
        truth = pd.read_parquet(truth_cache)
    else:
        with _rules("none"):
            truth = truth_rows(
                admit_panel(paths, panel=preferred, year=config.year),
                year=config.year,
                paths_state_dir=str(paths.state_dir),
            )
        truth.to_parquet(truth_cache, index=False)
    panel = admit_panel(paths, panel=preferred, year=config.year)
    protected = protected_oris(paths, year=config.year)
    eligible = sorted(set(truth["ori9"].astype(str)) - protected)
    if args.limit_agencies:
        eligible = eligible[: int(args.limit_agencies)]
        truth = truth[truth["ori9"].isin(set(eligible))]
    print(
        f"holdout {config.year}: panel {len(panel):,} rows, "
        f"{len(truth):,} truth rows over {len(eligible):,} eligible agencies "
        f"({len(protected)} protected)",
        flush=True,
    )

    runner = (
        run_agency_holdout if args.separate_arms else run_agency_holdout_combined
    )
    agency = runner(
        paths=paths,
        panel=panel,
        preferred=preferred,
        config=config,
        truth=truth,
        eligible_oris=eligible,
        label=str(args.label),
    )
    if not agency.empty:
        agency = pooled_silent_unit_backfill(agency, panel=panel, year=config.year)
        agency = score(agency)

    remainder = (
        pd.DataFrame() if args.skip_remainder else run_remainder_holdout(paths, year=config.year)
    )

    write_report(
        out_dir=out_dir,
        agency_scored=agency,
        remainder_scored=remainder,
        config=config,
        meta={
            "config_label": str(args.label),
            "append": bool(args.append),
            "elapsed_seconds": round(time.time() - started, 1),
            "eligible_agencies": len(eligible),
            "protected_agencies": len(protected),
            "truth_rows": int(len(truth)),
            "rules": os.environ.get("CRIMERISK_LEVEL_RULES", "<default>"),
        },
    )
    print(f"holdout done in {time.time() - started:.0f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
