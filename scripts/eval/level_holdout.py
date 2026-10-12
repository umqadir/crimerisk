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

Production equivalence
----------------------
A masked score tests the production estimator only if the unmasked fold path reproduces
production. `--equivalence-check` runs one estimator pass with no masking on the edition's
state and compares admission, reporting history, jurisdiction-year panel, accounting
controls, mass ledger, the state anchor's contribution ledger, state targets and final
smoothed controls with state/controls (`equivalence_check`). It must use the production
artifacts' year, and runs only after a complete production build (it refuses a stale
reporting-regime artifact rather than rebuild it in production state):

    uv run python scripts/eval/level_holdout.py --equivalence-check --year 2025 \\
        --panel-year-end 2025 --out-dir state/eval/level_equivalence

It writes state/eval/level_equivalence/equivalence_2025.json (per-stage row and value
differences, max absolute differences) and the fold's side files under
state/eval/level_equivalence/equivalence_2025/, prints the per-stage table, and exits 1
when any stage differs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import contextlib
from datetime import datetime, timezone
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

from crimerisk.agency_identity import (  # noqa: E402
    load_agency_jurisdiction_crosswalk,
    resolve_ori_succession,
)
from crimerisk.controls import (  # noqa: E402
    ControlBuildConfig,
    Stage1Consumption,
    build_controls_bundle,
)
from crimerisk.crime import OFFENSES_7  # noqa: E402
from crimerisk.eval.dependencies import level_holdout_dependency_paths  # noqa: E402
from crimerisk.jurisdiction_targets import (  # noqa: E402
    build_agency_target_panel_slice,
    load_crosswalk,
)
from crimerisk.level_lane import apply_level_lane_admission  # noqa: E402
from crimerisk.paths import RepoPaths, get_paths  # noqa: E402
from crimerisk.reporting_history import (  # noqa: E402
    adjudicate_history_years,
    reporting_history_path,
    write_reporting_history,
)
from crimerisk.reporting_regimes import (  # noqa: E402
    ReportingRegimeBuildConfig,
    reporting_regimes_artifact_is_current,
)
from crimerisk.service_scopes import load_custom_footprint_population_by_target  # noqa: E402
from crimerisk.smoothed_controls import (  # noqa: E402
    STATE_REMAINDER_TYPE,
    AnchorInputs,
    SmoothedControlConfig,
    accounting_controls_path,
    build_smoothed_controls_bundle,
    contribution_ledger,
    jurisdiction_year_estimates_path,
    load_anchor_inputs,
    reliable_remainder_rows,
    remainder_municipal_reference,
    remainder_peer_ratio,
    smoothed_controls_path,
)
from crimerisk.trend_fills import (  # noqa: E402
    FILL_MAX_REFERENCE_AGE_YEARS,
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
HOLDOUT_DEFAULT_YEAR = 2024

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


@dataclass(frozen=True)
class FallbackInputs:
    """What scoring a ladder-dropped agency needs besides the fold itself.

    `crosswalk` is the controls stage's agency-to-jurisdiction crosswalk; `footprint_population`
    is each custom-footprint agency's covered population (ori9, _covered_population);
    `work_dir` receives the fold controls stage's side files.
    """

    crosswalk: pd.DataFrame
    footprint_population: pd.DataFrame
    work_dir: Path


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


def build_preferred_panel(
    paths, *, year: int, panel_year_end: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(preferred-observation panel with every year before `year` adjudicated, that history).

    The panel is always built to the edition's own last year, never to the holdout
    target year: the reporting-regime stage rebuilds the observation panel to whatever
    year it is asked for, so asking it for 2024 would silently drop the edition's 2025
    rows from `state/observations`. The span is trimmed to the target year afterwards.
    The years before the target carry their own adjudication, as in production, so the
    panel depends on the admission rules in force (`CRIMERISK_LEVEL_RULES`).
    """
    panel = build_agency_trend_fill_panel(
        paths=paths,
        year_start=2018,
        year_end=max(int(year), int(panel_year_end)),
        adjudicate_history=False,
    )
    panel = panel[pd.to_numeric(panel["year"], errors="coerce").le(int(year))].copy()
    return adjudicate_history_years(panel, paths=paths, target_year=int(year))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relpath(path: Path, root: Path) -> str:
    path = Path(path)
    for base, target in ((root, path), (root.resolve(), path.resolve())):
        with contextlib.suppress(ValueError):
            return target.relative_to(base).as_posix()
    return path.resolve().as_posix()


def _estimator_input_files(paths, *, year: int, panel_year_end: int, remainder: bool = False) -> list[Path]:
    """Every file the holdout consumes, from the stages' own dependency declarations
    (`crimerisk.eval.dependencies.level_holdout_dependency_paths`, which the release validator
    also calls): the agency estimator's, controls stage's and smoother's declared inputs, the
    reporting-regime and observation closure, what this script reads directly, and the code and
    configuration. `remainder` adds the production smoothed controls the state-remainder lane
    scores. The script that is running is added as well (it is the repo's own copy whenever
    `paths` is this checkout)."""
    files = level_holdout_dependency_paths(
        paths, year=int(year), panel_year_end=int(panel_year_end), remainder=bool(remainder)
    )
    this_script = Path(__file__).resolve()
    if all(path.resolve() != this_script for path in files):
        files.append(this_script)
    return files


def preferred_panel_manifest(
    paths, *, year: int, panel_year_end: int, fold_id: str = "unmasked"
) -> dict[str, object]:
    """The complete identity of a cached preferred panel: every consumed input file's sha256,
    the state directory it was read from, target and as-of years, the admission rules in force,
    and the fold it belongs to."""
    root = Path(paths.repo_root)
    return {
        "year": int(year),
        "panel_year_end": int(panel_year_end),
        "fold_id": str(fold_id),
        "state_dir": str(Path(paths.state_dir).resolve()),
        "level_rules": os.environ.get("CRIMERISK_LEVEL_RULES", "<default>"),
        "inputs_sha256": {
            _relpath(path, root): _sha256_file(path)
            for path in _estimator_input_files(paths, year=year, panel_year_end=panel_year_end)
        },
    }


def _manifest_key(manifest: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _cached_frame(cache_dir: Path, name: str, manifest: dict[str, object]) -> pd.DataFrame | None:
    """The frame stored under `manifest`, or None unless its stored manifest is the same and
    the stored file is the one the manifest recorded."""
    entry = Path(cache_dir) / _manifest_key(manifest)[:24]
    frame_path, manifest_path = entry / name, entry / f"{name}.manifest.json"
    if not (frame_path.exists() and manifest_path.exists()):
        return None
    stored = json.loads(manifest_path.read_text())
    if stored.get("manifest") != manifest or stored.get("sha256") != _sha256_file(frame_path):
        return None
    return pd.read_parquet(frame_path)


def _store_frame(cache_dir: Path, name: str, manifest: dict[str, object], frame: pd.DataFrame) -> None:
    entry = Path(cache_dir) / _manifest_key(manifest)[:24]
    entry.mkdir(parents=True, exist_ok=True)
    frame_path = entry / name
    frame.to_parquet(frame_path, index=False)
    (entry / f"{name}.manifest.json").write_text(
        json.dumps({"manifest": manifest, "sha256": _sha256_file(frame_path)}, indent=2, sort_keys=True)
    )


def load_preferred_panel(
    paths,
    *,
    year: int,
    panel_year_end: int,
    cache_dir: Path,
    refresh: bool,
    fold_id: str = "unmasked",
) -> pd.DataFrame:
    """`build_preferred_panel`, cached under the full identity of what it consumed.

    The entry lives in `cache_dir/<key>/` beside the manifest of its inputs
    (`preferred_panel_manifest`) and the sha256 of the stored panel; it is reused only when
    the stored manifest equals the current one and the stored file is intact. Any other
    observations, configs, code, rules, years, state directory or fold rebuilds.
    """
    name = f"preferred_panel_{int(year)}.parquet"
    manifest = preferred_panel_manifest(paths, year=year, panel_year_end=panel_year_end, fold_id=fold_id)
    if not refresh:
        cached = _cached_frame(cache_dir, name, manifest)
        if cached is not None:
            return cached
    panel, _history = build_preferred_panel(paths, year=year, panel_year_end=panel_year_end)
    # The build can refresh a stale reporting-regime artifact it reads; key the entry on the
    # inputs as they stand after the build, which is what the next lookup will see.
    manifest = preferred_panel_manifest(paths, year=year, panel_year_end=panel_year_end, fold_id=fold_id)
    _store_frame(cache_dir, name, manifest, panel)
    return panel


def load_truth(
    paths,
    *,
    preferred: pd.DataFrame,
    year: int,
    panel_year_end: int,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """The truth universe (`truth_rows` under the unmodified lane), cached like the panel and
    keyed also on the content of the preferred panel it was read from."""
    with _rules("none"):
        manifest = preferred_panel_manifest(paths, year=year, panel_year_end=panel_year_end)
    manifest["preferred_panel_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(preferred, index=False).to_numpy().tobytes()
    ).hexdigest()
    name = f"truth_{int(year)}.parquet"
    if not refresh:
        cached = _cached_frame(cache_dir, name, manifest)
        if cached is not None:
            return cached
    with _rules("none"):
        truth = truth_rows(
            admit_panel(paths, panel=preferred, year=year),
            year=year,
            paths_state_dir=str(paths.state_dir),
        )
    _store_frame(cache_dir, name, manifest, truth)
    return truth


def admit(paths, *, panel: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(admitted panel, target-year admission disposition), as `build_stage1_consumption` makes them.

    The disposition is the level-lane admission's own target-year frame, the one production
    writes as `level_lane_admission_<year>.parquet` and the state anchor reads; it is taken
    before the masked-gap and usability steps, as there.
    """
    admission = apply_level_lane_admission(panel, paths=paths, target_year=int(year))
    admitted = admission.panel
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
    return admitted, admission.disposition


def admit_panel(paths, *, panel: pd.DataFrame, year: int) -> pd.DataFrame:
    """Admission, masked-gap reclassification and adjudicated usability, in order.

    The same sequence `build_agency_allocation_target_estimates` runs when it is handed
    a raw panel. Running it here, per rule configuration, keeps the admission rules
    (a)-(c) inside what the harness measures while the panel build above stays shared.
    """
    return admit(paths, panel=panel, year=year)[0]


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
    # The succession registry verifies that a successor ORI carries a target-year
    # report before it may suppress the retired one. Masking either side of a
    # migration fails that gate, which is the registry doing its job, so both sides
    # are out of scope for the holdout.
    ("stage1_adjudications/ori_successions.csv", False),
    ("stage1_adjudications/twins_adjudicated.csv", False),
    ("stage1_adjudications/zero_missing_adjudicated.csv", True),
    ("stage1_adjudications/token_reporters_adjudicated.csv", True),
)

ORI_COLUMN_CANDIDATES = (
    "ori9", "ori", "superseded_ori", "successor_ori", "superseded_ori9",
    "successor_ori9", "retired_ori9", "live_ori9", "primary_ori9", "primary_ori",
    "variant_ori9", "variant_ori", "old_ori9", "new_ori9", "twin_ori", "twin_ori9",
    "canonical_ori",
    # The twins, zero-missing and token-reporter rulings name their agencies here, several
    # to a cell.
    "oris",
)
YEAR_COLUMN_CANDIDATES = ("year", "target_year")


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
            # A registry that moves must not silently unprotect its agencies: level_v2
            # protected 124 agencies against level_v1's 1,099 after the 2026-09-21 cull
            # archived configs/level_lane_truth_panel.csv.
            raise FileNotFoundError(f"Protected-agency registry {path} is missing")
        frame = pd.read_csv(path, dtype="string", comment="#")
        if year_scoped:
            for column in YEAR_COLUMN_CANDIDATES:
                if column in frame.columns:
                    frame = frame[pd.to_numeric(frame[column], errors="coerce").eq(int(year))]
                    break
        for column in ORI_COLUMN_CANDIDATES:
            if column in frame.columns:
                values = frame[column].dropna().astype(str).str.split(r"[;|,\s]+", regex=True)
                out |= {value.upper().strip() for cell in values for value in cell}
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
    # jurisdiction falls to a pooled silent-unit control downstream. It is scored as the
    # fold's production control for its territory (`score_dropped_agencies`), so the arm's
    # number is the number the published surface would carry.
    joined["dropped_by_ladder"] = joined["estimated_count"].isna()
    joined["agency_estimate_source"] = (
        joined["agency_estimate_source"].astype("string").fillna("dropped_to_pooled_silent_unit")
    )
    joined["predicted_count"] = pd.to_numeric(joined["estimated_count"], errors="coerce")
    joined["arm"] = arm
    joined["fold"] = fold
    joined["config"] = label
    return joined


def combined_masks(
    eligible_oris: list[str], *, config: HoldoutConfig
) -> list[dict[str, set[str]]]:
    """Per estimator pass, the agencies each arm masks.

    `mask_share` is the share of eligible agencies each arm scores, in whole hash folds:
    round(folds * mask_share) folds per arm, as in the separate-arm runner. Pass k masks
    fold k * len(arms) + a for arm a, so every pass hides one fold per arm and the arms'
    agency sets stay disjoint within and across passes.
    """
    arms = tuple(config.arms)
    per_arm = max(1, int(round(config.folds * config.mask_share)))
    if per_arm * len(arms) > int(config.folds):
        raise ValueError(
            f"mask_share {config.mask_share} needs {per_arm} folds for each of {len(arms)} arms; "
            f"only {config.folds} folds exist"
        )
    folds = _stable_fold(pd.Series(eligible_oris), folds=config.folds, seed=config.seed)
    fold_of = dict(zip(eligible_oris, folds.tolist()))
    return [
        {
            arm: {ori for ori, value in fold_of.items() if value == k * len(arms) + index}
            for index, arm in enumerate(arms)
        }
        for k in range(per_arm)
    ]


def planned_masks(
    eligible_oris: list[str], *, config: HoldoutConfig, separate_arms: bool
) -> dict[str, dict[str, list[str]]]:
    """Per arm and hash fold, the agencies the runner is asked to mask and score.

    `run_agency_holdout_combined` masks hash fold k * len(arms) + a for arm a in pass k
    (`combined_masks`); `run_agency_holdout` masks hash folds 0 .. round(folds * mask_share) - 1
    for every arm. Recorded in run_meta, so the release validator can check that every planned
    case was scored (`planned_offenses`), not only that each arm has a finite overall row.
    """
    plan: dict[str, dict[str, list[str]]] = {arm: {} for arm in config.arms}
    if separate_arms:
        folds = _stable_fold(pd.Series(eligible_oris, dtype=object), folds=config.folds, seed=config.seed)
        fold_of = dict(zip(eligible_oris, folds.tolist()))
        n_folds = max(1, int(round(config.folds * config.mask_share)))
        for arm in config.arms:
            for fold in range(n_folds):
                plan[arm][str(fold)] = sorted(ori for ori, value in fold_of.items() if value == fold)
        return plan
    arms = tuple(config.arms)
    for k, masked_by_arm in enumerate(combined_masks(eligible_oris, config=config)):
        for index, arm in enumerate(arms):
            plan[arm][str(k * len(arms) + index)] = sorted(masked_by_arm[arm])
    return plan


def planned_offenses(arms: tuple[str, ...]) -> dict[str, list[str]]:
    """The offenses each arm scores per masked agency (`_score_masked`)."""
    return {
        arm: list(CORRUPTED_OFFENSES if arm == "corrupted" else OFFENSES_7) for arm in arms
    }


@dataclass(frozen=True)
class FoldControls:
    """What the fold's controls stage and smoother produced, and the anchor inputs it was given."""

    controls: pd.DataFrame
    jurisdiction_years: pd.DataFrame
    mass_ledger: pd.DataFrame
    anchor_inputs: AnchorInputs
    smoothed: pd.DataFrame
    summary: dict


@dataclass(frozen=True)
class FoldPass:
    """One estimator pass: the masked admitted panel and admission, the estimates, the controls."""

    admitted: pd.DataFrame
    admission: pd.DataFrame
    estimates: pd.DataFrame
    controls: FoldControls


def estimate_pass(
    paths,
    *,
    preferred: pd.DataFrame,
    year: int,
    masked_by_arm: dict[str, set[str]],
    work_dir: Path,
) -> FoldPass:
    """Corrupt, admit, mask, estimate and run the fold's controls stage and smoother.

    Corruption happens before admission (whether the gate sees through it is the thing
    measured); refusal and silence are applied to the admitted panel and to the admission
    disposition alike, so the state anchor sees the masked fold, not the edition. With
    no masks this is the production sequence, which `equivalence_check` holds it to.
    """
    work = preferred
    if masked_by_arm.get("corrupted"):
        work = corrupt_panel(work, year=year, masked_oris=masked_by_arm["corrupted"])
    admitted, admission = admit(paths, panel=work, year=year)
    for arm in ("refused", "silent"):
        if masked_by_arm.get(arm):
            admitted = mask_panel(admitted, year=year, masked_oris=masked_by_arm[arm], arm=arm)
            admission = mask_panel(admission, year=year, masked_oris=masked_by_arm[arm], arm=arm)
    estimates = build_agency_allocation_target_estimates(
        paths=paths, year=int(year), agency_panel=admitted
    )
    # The production fallback runs on the panel this pass estimated from, so no arm's
    # prediction can be informed by its own hidden target-year counts.
    controls = fold_controls(
        paths,
        panel=admitted,
        estimates=estimates,
        admission=admission,
        year=year,
        work_dir=work_dir,
    )
    return FoldPass(admitted=admitted, admission=admission, estimates=estimates, controls=controls)


def run_agency_holdout_combined(
    *,
    paths,
    panel: pd.DataFrame,
    preferred: pd.DataFrame,
    config: HoldoutConfig,
    truth: pd.DataFrame,
    eligible_oris: list[str],
    label: str,
    fallback: FallbackInputs,
) -> pd.DataFrame:
    """Shared estimator passes, three arms, disjoint agency sets.

    The estimator is the whole cost of a run and it re-estimates every agency whatever
    is masked, so running the arms on separate folds of one panel measures the same
    thing three times more cheaply. The arms stay independent because no agency is in
    two of them, and every configuration masks exactly the same agencies, so the
    comparison between configurations is unaffected by the sharing. `combined_masks`
    sets how many passes `mask_share` takes.
    """
    folds = _stable_fold(pd.Series(eligible_oris), folds=config.folds, seed=config.seed)
    fold_of = dict(zip(eligible_oris, folds.tolist()))
    results: list[pd.DataFrame] = []
    for pass_index, masked_by_arm in enumerate(combined_masks(eligible_oris, config=config)):
        started = time.time()
        fold_pass = estimate_pass(
            paths,
            preferred=preferred,
            year=config.year,
            masked_by_arm=masked_by_arm,
            work_dir=fallback.work_dir / (label if pass_index == 0 else f"{label}_pass{pass_index}"),
        )
        estimates = fold_pass.estimates
        smoothed = fold_pass.controls.smoothed
        for arm, masked in masked_by_arm.items():
            if not masked:
                continue
            scored = score_dropped_agencies(
                _score_masked(
                    truth=truth,
                    estimates=estimates,
                    masked=masked,
                    arm=arm,
                    fold=0,
                    label=label,
                ),
                smoothed=smoothed,
                fallback=fallback,
            )
            scored["fold"] = scored["ori9"].map(fold_of)
            results.append(scored)
        print(
            f"  [{label}] pass {pass_index}: arms {list(masked_by_arm)} "
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
    fallback: FallbackInputs,
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
            fold_pass = estimate_pass(
                paths,
                preferred=preferred,
                year=config.year,
                masked_by_arm={arm: masked},
                work_dir=fallback.work_dir / label / f"{arm}_{fold}",
            )
            estimates = fold_pass.estimates
            smoothed = fold_pass.controls.smoothed
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
            # it as the fold's production control for its territory, so the arm's number
            # is the number the published surface would carry.
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
            joined = score_dropped_agencies(joined, smoothed=smoothed, fallback=fallback)
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


def isolated_state_paths(paths, root: Path) -> RepoPaths:
    """Production state for reading, with state/controls replaced by an empty directory.

    The controls stage writes side files (CT overlap conservation, benchmark uncertainty)
    into state/controls. A fold's are not the edition's, so the fold runs on paths whose
    state/controls lives under `root`; every other state entry links to the production one.
    The stage reads nothing from state/controls.
    """
    state = Path(root) / "state"
    (state / "controls").mkdir(parents=True, exist_ok=True)
    for entry in Path(paths.state_dir).iterdir():
        link = state / entry.name
        # state/eval holds `root` itself; linking it would make a cycle.
        if entry.name in {"controls", "eval"} or link.exists() or link.is_symlink():
            continue
        link.symlink_to(entry.resolve())
    review = state / "review"
    return replace(
        RepoPaths.from_repo_root(Path(paths.repo_root)),
        state_dir=state,
        cache_dir=state / "cache",
        review_dir=review,
        review_support_dir=review / "support",
        review_queues_dir=review / "queues",
        review_packets_dir=review / "packets",
        review_analysis_dir=review / "analysis",
        review_runs_dir=review / "runs",
    )


ANCHOR_OBSERVATION_COLUMNS = ["ori9", "year", "source", "offense", "count", "months_reported"]


def fold_controls(
    paths,
    *,
    panel: pd.DataFrame,
    estimates: pd.DataFrame,
    admission: pd.DataFrame,
    year: int,
    work_dir: Path,
) -> FoldControls:
    """The fold's controls, mass ledger and smoothed controls, from the production stages.

    `panel` is the fold's admitted, masked panel, `estimates` the estimator's output on it and
    `admission` the fold's target-year admission disposition (masked like the panel).
    Pre-imputation controls, pooled silent-unit imputation (imputation v2, as released) and
    smoothing run as `build-controls` and `build-smoothed-controls` run them; the state anchor
    reads the fold's own admission and mass ledger, never the edition's state/controls ones.
    """
    fold_paths = isolated_state_paths(paths, work_dir)
    try:
        return _fold_controls(
            paths,
            fold_paths=fold_paths,
            panel=panel,
            estimates=estimates,
            admission=admission,
            year=year,
        )
    finally:
        # Keep the fold's side files; drop the read-through links so nothing walking the
        # run directory reaches production state through them.
        for link in fold_paths.state_dir.iterdir():
            if link.is_symlink():
                link.unlink()


def fold_fallback_controls(
    paths,
    *,
    panel: pd.DataFrame,
    estimates: pd.DataFrame,
    admission: pd.DataFrame,
    year: int,
    work_dir: Path,
) -> pd.DataFrame:
    """The fold's smoothed jurisdiction controls (`fold_controls`)."""
    return fold_controls(
        paths, panel=panel, estimates=estimates, admission=admission, year=year, work_dir=work_dir
    ).smoothed


def fold_anchor_inputs(
    fold_paths: RepoPaths, *, admission: pd.DataFrame, mass_ledger: pd.DataFrame | None, year: int
) -> AnchorInputs:
    """The state anchor's inputs for a fold: its admission, its mass ledger, and the agency-year
    observations of the fold's state (the observations are an input of the edition, not an
    output of its controls stage). An empty admission or ledger is an error: the anchor would
    otherwise lose its documented corrections without saying so."""
    if admission is None or admission.empty:
        raise ValueError("the fold has no admission disposition for the state anchor")
    if mass_ledger is None or mass_ledger.empty:
        raise ValueError("the fold's controls stage returned no level-lane mass ledger")
    observations_path = Path(fold_paths.state_dir) / "observations" / "agency_year_observations.parquet"
    observations = pd.read_parquet(
        observations_path, columns=ANCHOR_OBSERVATION_COLUMNS, filters=[("year", "==", int(year))]
    )
    return AnchorInputs(admission=admission, mass_ledger=mass_ledger, observations=observations)


def _fold_controls(
    paths,
    *,
    fold_paths: RepoPaths,
    panel: pd.DataFrame,
    estimates: pd.DataFrame,
    admission: pd.DataFrame,
    year: int,
) -> FoldControls:
    succession, _summary = resolve_ori_succession(
        paths=paths,
        agency_panel=panel,
        agency_jurisdiction_crosswalk=load_agency_jurisdiction_crosswalk(paths),
        target_year=int(year),
        max_reference_age_years=FILL_MAX_REFERENCE_AGE_YEARS,
    )
    stage1 = Stage1Consumption(
        agency_panel=panel,
        agency_preferred_target_year=build_agency_target_panel_slice(
            agency_panel=panel, target_year=int(year)
        ),
        agency_estimates=estimates,
        succession_ledger=succession,
        admission_disposition=admission,
    )
    controls, _state, jurisdiction_years, _exclusions, imputation, _estimates = (
        build_controls_bundle(
            paths=fold_paths,
            config=ControlBuildConfig(year=int(year)),
            stage1=stage1,
        )
    )
    mass_ledger = getattr(imputation, "mass_ledger", None)
    anchor_inputs = fold_anchor_inputs(
        fold_paths, admission=admission, mass_ledger=mass_ledger, year=int(year)
    )
    smoothed, summary = build_smoothed_controls_bundle(
        paths=fold_paths,
        config=SmoothedControlConfig(year=int(year)),
        accounting=controls,
        panel=jurisdiction_years,
        anchor_inputs=anchor_inputs,
    )
    return FoldControls(
        controls=controls,
        jurisdiction_years=jurisdiction_years,
        mass_ledger=mass_ledger,
        anchor_inputs=anchor_inputs,
        smoothed=smoothed,
        summary=summary,
    )


def score_dropped_agencies(
    scored: pd.DataFrame, *, smoothed: pd.DataFrame, fallback: FallbackInputs
) -> pd.DataFrame:
    """Give each ladder-dropped agency the fold's production control for its territory.

    A dropped agency has no estimate row, so production sizes its jurisdiction without it:
    pooled silent-unit imputation where no agency reports, then smoothing. The agency is
    scored as that smoothed control times its crosswalk weight: all of it when the agency is
    the jurisdiction's only crosswalk agency, otherwise its footprint share of the
    jurisdiction's population (custom footprint population, else the agency's own). A
    jurisdiction without population (a statewide overlap layer) carries no share.
    """
    out = scored.copy()
    dropped = out["dropped_by_ladder"].fillna(False).astype(bool)
    out["fallback_jurisdiction_ids"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if not dropped.any():
        return out
    links = fallback.crosswalk[["ori9", "jurisdiction_id", "weight"]].copy()
    links["ori9"] = links["ori9"].astype("string").str.upper()
    links["jurisdiction_id"] = links["jurisdiction_id"].astype("string")
    agencies = links.groupby("jurisdiction_id")["ori9"].nunique()
    controls = smoothed[["jurisdiction_id", "offense", "smoothed_count", "bucket_population"]].copy()
    controls["jurisdiction_id"] = controls["jurisdiction_id"].astype("string")
    controls["offense"] = controls["offense"].astype("string")
    footprint = (
        fallback.footprint_population.assign(ori9=lambda d: d["ori9"].astype("string").str.upper())
        .drop_duplicates("ori9")
        .set_index("ori9")["_covered_population"]
    )
    rows = out.loc[dropped, ["ori9", "offense", "population"]].copy()
    rows["_row"] = rows.index
    rows["ori9"] = rows["ori9"].astype("string").str.upper()
    rows["offense"] = rows["offense"].astype("string")
    merged = rows.merge(links, on="ori9", how="left").merge(
        controls, on=["jurisdiction_id", "offense"], how="left"
    )
    sole = merged["jurisdiction_id"].map(agencies).eq(1).to_numpy()
    exposure = pd.to_numeric(merged["ori9"].map(footprint), errors="coerce").fillna(
        pd.to_numeric(merged["population"], errors="coerce")
    ).fillna(0.0).to_numpy(float)
    bucket = pd.to_numeric(merged["bucket_population"], errors="coerce").fillna(0.0).to_numpy(float)
    share = np.where(
        sole,
        1.0,
        np.where(bucket > 0.0, np.minimum(1.0, exposure / np.where(bucket > 0.0, bucket, 1.0)), 0.0),
    )
    merged["_prediction"] = (
        pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0).to_numpy(float)
        * pd.to_numeric(merged["smoothed_count"], errors="coerce").fillna(0.0).to_numpy(float)
        * share
    )
    grouped = merged.groupby("_row")
    out.loc[dropped, "predicted_count"] = (
        grouped["_prediction"].sum().reindex(out.index[dropped]).fillna(0.0).to_numpy(float)
    )
    out.loc[dropped, "fallback_jurisdiction_ids"] = (
        grouped["jurisdiction_id"]
        .agg(lambda values: "|".join(sorted(str(v) for v in values.dropna().unique())))
        .reindex(out.index[dropped])
        .astype("string")
        .to_numpy()
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

    Eligibility, the municipal reference pool and the ratio are production's own
    (`remainder_municipal_reference`, `reliable_remainder_rows`, `remainder_peer_ratio`),
    applied to the pre-calibration counts the production rule saw.
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
        "pop_band",
        "bucket_population",
        "precalibration_smoothed_count",
        "estimator",
        "clean_year_count",
    }
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    # The rule runs before the national anchor calibration; score it on what it saw.
    frame["smoothed_count"] = pd.to_numeric(frame["precalibration_smoothed_count"], errors="coerce")
    municipal = remainder_municipal_reference(frame)
    remainder = frame[
        frame["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
    ].merge(municipal, on=["state_fips", "offense"], how="left", validate="many_to_one")
    reliable = reliable_remainder_rows(remainder)
    if reliable.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for row in reliable.itertuples(index=False):
        training = reliable[
            reliable["offense"].eq(row.offense) & ~reliable["state_fips"].eq(row.state_fips)
        ]
        if training.empty:
            continue
        ratio = float(remainder_peer_ratio(training)["remainder_to_municipal_rate_ratio"].iloc[0])
        if not np.isfinite(ratio):
            continue
        predicted = float(row.bucket_population) * float(row.municipal_rate) * ratio
        rows.append(
            {
                "state_fips": row.state_fips,
                "offense": str(row.offense),
                "jurisdiction_id": row.jurisdiction_id,
                "observed_count": float(row.smoothed_count),
                "predicted_count": predicted,
                "remainder_population": float(row.bucket_population),
                "leave_one_out_ratio": ratio,
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


def level_holdout_input_digests(
    paths, *, year: int, panel_year_end: int, remainder: bool
) -> dict[str, str]:
    """sha256 of everything the holdout consumes, by repo-relative path.

    The release validator binds the reported level metric to the edition through these: a
    holdout run on other observations, raw sources, reference tables, configs or code is not
    this edition's. The set is `crimerisk.eval.dependencies.level_holdout_dependency_paths`,
    the same function the validator computes its required set with.
    """
    root = Path(paths.repo_root)
    return {
        _relpath(path, root): _sha256_file(path)
        for path in _estimator_input_files(
            paths, year=year, panel_year_end=panel_year_end, remainder=remainder
        )
    }


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
    written = ["results.csv", "results.md"]
    if not agency_scored.empty:
        agency_scored.to_parquet(out_dir / "agency_cases.parquet", index=False)
        # One file per configuration, so a sweep keeps every case it scored and the
        # summary tables can be rebuilt without re-running the estimator.
        label = str(meta.get("config_label", "baseline"))
        agency_scored.to_parquet(out_dir / f"agency_cases_{label}.parquet", index=False)
        written += ["agency_cases.parquet", f"agency_cases_{label}.parquet"]
    if not remainder_scored.empty:
        remainder_scored.to_parquet(out_dir / "remainder_cases.parquet", index=False)
        written.append("remainder_cases.parquet")

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
    # The run's metadata is the last write: `completed_at_utc` marks a run whose results
    # are all on disk, and `outputs_sha256` binds them, so the release validator reads the
    # case files and results this run wrote and no other.
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                **meta,
                "arms": list(config.arms),
                "config": config.__dict__,
                "outputs_sha256": {name: _sha256_file(out_dir / name) for name in written},
                "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
            default=str,
        )
    )


# ---------------------------------------------------------------------------
# production equivalence
# ---------------------------------------------------------------------------


EQUIVALENCE_TOLERANCE = 1e-6


def frame_difference(
    fold: pd.DataFrame, production: pd.DataFrame, *, keys: list[str], tolerance: float = EQUIVALENCE_TOLERANCE
) -> dict[str, object]:
    """How far two frames are apart, aligned on `keys`.

    Rows only one side has, columns only one side has, the largest absolute difference of
    every shared numeric column (a value on one side and a missing one on the other counts
    as a mismatch), and the count of unequal values of every other shared column.
    """
    def keyed(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reset_index(drop=True).copy()
        for key in keys:
            out[key] = out[key].astype("string")
        # Duplicate keys are aligned in order, not cross-joined.
        out["_dup"] = out.groupby(keys, dropna=False).cumcount()
        return out

    left, right = keyed(fold), keyed(production)
    align = [*keys, "_dup"]
    shared = [c for c in left.columns if c in right.columns and c not in align]
    merged = left[align + shared].merge(
        right[align + shared], on=align, how="outer", suffixes=("_fold", "_prod"), indicator=True
    )
    both = merged[merged["_merge"].eq("both")]
    numeric_diff: dict[str, float] = {}
    mismatches: dict[str, int] = {}
    for column in shared:
        a, b = both[f"{column}_fold"], both[f"{column}_prod"]
        numeric = (
            pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column])
            and not pd.api.types.is_bool_dtype(left[column]) and not pd.api.types.is_bool_dtype(right[column])
        )
        if numeric:
            x = pd.to_numeric(a, errors="coerce").astype(float)
            y = pd.to_numeric(b, errors="coerce").astype(float)
            one_missing = int((x.isna() ^ y.isna()).sum())
            gap = float((x - y).abs().max()) if (x.notna() & y.notna()).any() else 0.0
            gap = 0.0 if np.isnan(gap) else gap
            if gap > tolerance:
                numeric_diff[column] = gap
            if one_missing:
                mismatches[column] = one_missing
        else:
            x = a.astype("string").fillna("<NA>")
            y = b.astype("string").fillna("<NA>")
            unequal = int(x.ne(y).sum())
            if unequal:
                mismatches[column] = unequal
    only_fold = int(merged["_merge"].eq("left_only").sum())
    only_production = int(merged["_merge"].eq("right_only").sum())
    columns_only_fold = sorted(set(fold.columns) - set(production.columns))
    columns_only_production = sorted(set(production.columns) - set(fold.columns))
    return {
        "rows_fold": int(len(fold)),
        "rows_production": int(len(production)),
        "rows_only_fold": only_fold,
        "rows_only_production": only_production,
        "columns_only_fold": columns_only_fold,
        "columns_only_production": columns_only_production,
        "max_abs_difference": max(numeric_diff.values(), default=0.0),
        "numeric_differences": numeric_diff,
        "value_mismatches": mismatches,
        "equivalent": not (
            only_fold or only_production or numeric_diff or mismatches
            or columns_only_fold or columns_only_production
        ),
    }


def _state_targets(smoothed: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in ("state_anchor_count", "state_anchor_source") if c in smoothed.columns]
    return smoothed.groupby(["state_abbr", "offense"], as_index=False, dropna=False)[columns].first()


def equivalence_check(paths, *, year: int, panel_year_end: int, out_dir: Path) -> dict[str, object]:
    """The fold path with no masking, compared against the edition's own state/controls.

    Runs exactly what an estimator pass runs (`build_preferred_panel`, then `estimate_pass`
    with no masks: admission, estimates, the controls stage, the mass ledger, the state
    anchor's contribution ledger, smoothing) in an isolated fold state under `out_dir`, and
    compares each stage with the production artifact for the same year:

    admission       state/controls/level_lane_admission_<year>.parquet
    history         state/controls/reporting_history_<year>.parquet
    jurisdiction_years  state/controls/jurisdiction_year_estimates.parquet
    accounting      state/controls/jurisdiction_controls_<year>.parquet
    mass_ledger     state/controls/level_lane_mass_ledger_<year>.parquet
    contributions   smoothed_controls.contribution_ledger on the production anchor inputs
    state_targets   state_anchor_count of jurisdiction_controls_smoothed_<year>.parquet
    smoothed        state/controls/jurisdiction_controls_smoothed_<year>.parquet

    `year` must be the production artifacts' year (the edition's smoothing year, 2025 for
    2025.1). The check refuses to start when an artifact is missing or the reporting-regime
    artifact is stale (the panel build would otherwise rewrite production state). It writes
    only under `out_dir`.
    """
    controls_dir = Path(paths.state_dir) / "controls"
    production_paths = {
        "admission": controls_dir / f"level_lane_admission_{int(year)}.parquet",
        "history": reporting_history_path(paths, year=int(year)),
        "jurisdiction_years": jurisdiction_year_estimates_path(paths),
        "accounting": accounting_controls_path(paths, year=int(year)),
        "mass_ledger": controls_dir / f"level_lane_mass_ledger_{int(year)}.parquet",
        "smoothed": smoothed_controls_path(paths, year=int(year)),
    }
    missing = [str(path) for path in production_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"production artifacts for {int(year)} are missing: {missing}")
    regimes = ReportingRegimeBuildConfig(year_start=2018, year_end=max(int(year), int(panel_year_end)))
    if not reporting_regimes_artifact_is_current(paths, config=regimes):
        raise RuntimeError(
            "the reporting-regime artifact is not current; the panel build would rebuild it in "
            "production state. Finish the production build first."
        )
    out_dir = Path(out_dir)
    work_dir = out_dir / f"equivalence_{int(year)}"
    preferred, history = build_preferred_panel(paths, year=int(year), panel_year_end=int(panel_year_end))
    fold_pass = estimate_pass(
        paths, preferred=preferred, year=int(year), masked_by_arm={}, work_dir=work_dir
    )
    # The history table as `build_stage1_consumption` writes it, into the fold's state.
    fold_history_path = write_reporting_history(
        history,
        fold_pass.admitted[pd.to_numeric(fold_pass.admitted["year"], errors="coerce").eq(int(year))],
        paths=replace(paths, state_dir=work_dir / "state"),
        year=int(year),
    )
    production = {name: pd.read_parquet(path) for name, path in production_paths.items()}
    fold = fold_pass.controls
    comparisons = {
        "admission": (fold_pass.admission, production["admission"], ["ori9", "offense", "year"]),
        "history": (pd.read_parquet(fold_history_path), production["history"], ["ori9", "offense", "year"]),
        "jurisdiction_years": (
            fold.jurisdiction_years, production["jurisdiction_years"], ["jurisdiction_id", "offense", "year"]
        ),
        "accounting": (fold.controls, production["accounting"], ["jurisdiction_id", "offense"]),
        "mass_ledger": (
            fold.mass_ledger,
            production["mass_ledger"],
            ["ori9", "year", "offense", "jurisdiction_id", "ownership_class"],
        ),
        "contributions": (
            contribution_ledger(fold.anchor_inputs, year=int(year)),
            contribution_ledger(load_anchor_inputs(paths, year=int(year)), year=int(year)),
            ["state_abbr", "offense", "ori9"],
        ),
        "state_targets": (
            _state_targets(fold.smoothed), _state_targets(production["smoothed"]), ["state_abbr", "offense"]
        ),
        "smoothed": (fold.smoothed, production["smoothed"], ["jurisdiction_id", "offense"]),
    }
    report = {
        "year": int(year),
        "panel_year_end": int(panel_year_end),
        "tolerance": EQUIVALENCE_TOLERANCE,
        "level_rules": os.environ.get("CRIMERISK_LEVEL_RULES", "<default>"),
        "components": {
            name: frame_difference(left, right, keys=keys) for name, (left, right, keys) in comparisons.items()
        },
    }
    report["equivalent"] = all(entry["equivalent"] for entry in report["components"].values())
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"equivalence_{int(year)}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level-lane holdout harness.")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            f"Holdout target year (default {HOLDOUT_DEFAULT_YEAR}). With --equivalence-check, the "
            "production artifacts' year (default --panel-year-end)."
        ),
    )
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
        help=(
            "Share of eligible agencies each arm scores, in whole folds (round(folds * share) "
            "folds per arm). Each estimator pass masks one fold per arm."
        ),
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
    parser.add_argument(
        "--equivalence-check",
        action="store_true",
        help=(
            "Run the fold path with no masking and compare it with the production artifacts in "
            "state/controls (`equivalence_check`); writes <out-dir>/equivalence_<year>.json and "
            "exits 1 when any stage differs."
        ),
    )
    return parser


def _print_equivalence(report: dict[str, object]) -> None:
    rows = [
        {
            "component": name,
            "rows_fold": entry["rows_fold"],
            "rows_production": entry["rows_production"],
            "only_fold": entry["rows_only_fold"],
            "only_production": entry["rows_only_production"],
            "max_abs_difference": float(entry["max_abs_difference"]),
            "mismatched_columns": len(entry["value_mismatches"]),
            "equivalent": str(entry["equivalent"]),
        }
        for name, entry in report["components"].items()
    ]
    print(_markdown_table(pd.DataFrame(rows)), flush=True)
    print(f"equivalent: {report['equivalent']}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    if args.equivalence_check:
        year = int(args.year) if args.year is not None else int(args.panel_year_end)
        report = equivalence_check(
            paths,
            year=year,
            panel_year_end=int(args.panel_year_end),
            out_dir=paths.repo_root / args.out_dir,
        )
        _print_equivalence(report)
        return 0 if report["equivalent"] else 1
    config = HoldoutConfig(
        year=int(args.year) if args.year is not None else HOLDOUT_DEFAULT_YEAR,
        folds=int(args.folds),
        mask_share=float(args.mask_share),
        seed=int(args.seed),
        arms=tuple(args.arms),
    )
    out_dir = paths.repo_root / args.out_dir
    cache_dir = paths.repo_root / args.cache_dir
    # A run that does not finish leaves no completion marker behind from an earlier one.
    (out_dir / "run_meta.json").unlink(missing_ok=True)

    started = time.time()
    preferred = load_preferred_panel(
        paths,
        year=config.year,
        panel_year_end=int(args.panel_year_end),
        cache_dir=cache_dir,
        refresh=bool(args.refresh_panel),
    )
    # After the panel build, which may refresh a stale reporting-regime artifact it reads.
    remainder_scored = not bool(args.skip_remainder)
    binding = dict(
        year=config.year, panel_year_end=int(args.panel_year_end), remainder=remainder_scored
    )
    input_digests = level_holdout_input_digests(paths, **binding)
    # The truth universe is fixed by the unmodified lane, not by the configuration
    # under test: a rule that refuses more rows must be scored on the same cases as
    # one that refuses fewer.
    truth = load_truth(
        paths,
        preferred=preferred,
        year=config.year,
        panel_year_end=int(args.panel_year_end),
        cache_dir=cache_dir,
        refresh=bool(args.refresh_panel),
    )
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

    plan = planned_masks(eligible, config=config, separate_arms=bool(args.separate_arms))
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
        fallback=FallbackInputs(
            crosswalk=load_crosswalk(paths),
            footprint_population=load_custom_footprint_population_by_target(paths),
            work_dir=out_dir / "work",
        ),
    )
    if not agency.empty:
        # The production fallback is applied inside the runner, per estimator pass, on that
        # pass's masked panel. Applying it here would use the unmasked panel.
        agency = score(agency)

    remainder = (
        pd.DataFrame() if args.skip_remainder else run_remainder_holdout(paths, year=config.year)
    )

    if level_holdout_input_digests(paths, **binding) != input_digests:
        raise RuntimeError(
            "the holdout's inputs changed during the run; its results are bound to no edition"
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
            "panel_year_end": int(args.panel_year_end),
            "remainder_scored": remainder_scored,
            "edition_inputs_sha256": input_digests,
            "planned_masks": plan,
            "planned_offenses": planned_offenses(config.arms),
        },
    )
    print(f"holdout done in {time.time() - started:.0f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
