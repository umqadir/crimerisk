"""Hierarchical level-lane admission and offense-granular mass semantics.

Source selection has already chosen one lane per agency-year when this module runs.
Level 1 classifies that lane's coverage/identity basis; level 2 classifies each offense's
semantic completeness. Hard evidence may replace/refuse. Soft anomaly signals are resolved
at build time to either an admitted complete-year observation or the existing repair ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.contract_coverage import covered_by_source_path, load_covered_by_relationships


LEVEL1 = frozenset({
    "valid_complete_year", "valid_partial_lower_bound", "corroborated_structural_zero",
    "coverage_defective", "source_identity_failure", "unresolved_review",
})
LEVEL2 = frozenset({
    "definitionally_complete", "channel_omitted", "wrong_category_mapping",
    "documented_swap", "semantically_unusable",
})
REGISTRY_ACTIONS = frozenset({"accept_unchanged", "refuse_repair", "refuse_silent"})
INVALID_LEVEL1 = frozenset({"coverage_defective", "source_identity_failure"})
INVALID_LEVEL2 = frozenset({"channel_omitted", "wrong_category_mapping", "documented_swap", "semantically_unusable"})
PEER_ENVELOPE_MIN_PRIOR_COUNT = 20.0

class LevelLaneConfigError(ValueError):
    pass


# --- own-history admission bound -------------------------------------------------
# An agency's own year-over-year log changes are the primary bound on its next one.
# Three clean years give two changes, the fewest from which a spread can be read at
# all; below that the peer envelope still speaks. The spread floor keeps a placid
# agency from being given a zero-width band, and the multiplier is the usual
# three-sigma review threshold. All three are policy constants, not fitted values.
OWN_HISTORY_MIN_CLEAN_YEARS = 3
OWN_HISTORY_SPREAD_FLOOR = 0.25
OWN_HISTORY_SPREAD_MULTIPLE = 3.0

# --- joint-vector gate -----------------------------------------------------------
# One offense falling off a cliff is news; three falling off together while a fourth
# grows is a filing failure. The per-offense test is deliberately weaker than the
# admission bound -- "regardless of any single margin" is the whole point of it -- so
# an offense counts when it falls further than the agency ever has AND by more than a
# fifth.
#
# The divergence requirement is what separates a filing failure from a real decline,
# and it is not optional. A city whose crime genuinely falls takes the whole vector
# down together: Memphis 2024 moved between -0.48 and 0.00 across its seven offenses,
# Oakland between -0.80 and -0.13. Cicero 2025 dropped burglary 72% while larceny rose
# 53%. Without the divergence test the gate refuses the first two, whose 2025 filings
# then confirm the decline was real.
JOINT_VECTOR_MIN_BREACHES = 3
JOINT_VECTOR_MIN_LOG_DROP = -0.25
JOINT_VECTOR_DIVERGENT_RISE = 0.10

# --- NIBRS transition guard ------------------------------------------------------
# A level break within a year either side of the agency's NIBRS start date is a
# definitional discontinuity, not a change in crime. That matters most for the
# admission band -- a definitional halving is not evidence that the agency can halve
# -- and least for the fill references, where the one-year decay half-life already
# means the most recent year dominates.
NIBRS_TRANSITION_GUARD_YEARS = 1

# --- partial-year floor ----------------------------------------------------------
# A partial year is admitted as a lower bound and annualised by 12/months. That is
# right for an agency that filed four real months and wrong for one that filed a token
# four months: Highlands County FL's sheriff filed 8 Part 1 offenses over four months
# in 2025 against 3,278 in 2019, and annualising it publishes 24 for 95,570 residents.
# When the annualised vector is a fraction of what the agency's own clean years say, it
# is a non-report wearing a partial year's clothes, and it belongs on the repair ladder
# with the agencies that filed nothing. The ratio is the same collapse threshold the
# soft screen already uses.
PARTIAL_YEAR_HISTORY_FLOOR_RATIO = 0.25

# --- pooled prior versus own history ---------------------------------------------
# With this many clean years the agency's own decayed history stands on its own and
# the pooled peer prior is not mixed into it.
OWN_HISTORY_SUFFICIENT_CLEAN_YEARS = 5


@dataclass(frozen=True)
class LevelLanePolicy:
    """Which of the reviewed level-lane rules this build applies.

    Every rule is switchable so the holdout harness can run the lane with and without
    it. `CRIMERISK_LEVEL_RULES` overrides the defaults: `none` turns all of them off,
    `all` turns them all on, and a comma-separated `name=0|1` list sets them one by one.
    """

    own_history_admission_bound: bool = True
    # Rejected by the holdout: it improves the corrupted arm less than the own-history
    # bound alone, and the following year's filings confirm five of every six vectors
    # it refuses were a real decline. The switch stays so the measurement can be rerun.
    joint_vector_gate: bool = False
    nibrs_transition_guard: bool = True
    # The second half of the transition guard -- dropping pre-transition years as fill
    # references, rather than only as evidence about how much an agency moves -- is a
    # small consistent regression on both masked arms, so it is off by default.
    nibrs_transition_reference_guard: bool = False
    partial_year_history_floor: bool = True
    prefer_own_history_over_pool: bool = True
    remainder_coverage_cap: bool = True

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__)


def level_lane_policy() -> LevelLanePolicy:
    raw = os.environ.get("CRIMERISK_LEVEL_RULES")
    if raw is None:
        return LevelLanePolicy()
    text = raw.strip().lower()
    names = LevelLanePolicy.field_names()
    if text in {"", "default"}:
        return LevelLanePolicy()
    if text == "all":
        return LevelLanePolicy(**{name: True for name in names})
    if text in {"none", "off", "baseline"}:
        return LevelLanePolicy(**{name: False for name in names})
    # A token list names the rules this run applies; everything unnamed is off, so a
    # single rule can be isolated without having to switch the other four off by hand.
    values = {name: False for name in names}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, _, value = token.partition("=")
            on = value.strip() not in {"0", "false", "no", "off"}
        else:
            key, on = token, True
        key = key.strip()
        if key == "all":
            values = {name: on for name in names}
            continue
        if key not in values:
            raise LevelLaneConfigError(f"unknown level-lane rule switch: {key}")
        values[key] = on
    return LevelLanePolicy(**values)


@dataclass(frozen=True)
class AdmissionArtifacts:
    panel: pd.DataFrame
    disposition: pd.DataFrame
    review_queue: pd.DataFrame


def classify_controlled_corruption(
    clean: pd.Series,
    corrupted: pd.Series,
    *,
    defect_class: str,
    affected_offenses: set[str] | None = None,
    measured_coverage_months: int | None = None,
    external_comparator: bool = False,
) -> pd.DataFrame:
    """Executable v2 evidence contract used by controlled-corruption validation.

    ``clean`` represents an independently verified source lane. It is a permitted repair
    donor in these tests, not an oracle available to ordinary production rows. The function
    deliberately declares coherent scaling internally undetectable without an independent
    comparator.
    """
    known = {
        "all_zero", "token_one", "mislabeled_partial", "selective_channel_omission",
        "chicago_category_omission", "offense_swap", "duplicate_identity",
        "uniform_factor", "single_offense_collapse", "genuine_spike",
        "genuine_decline", "stale_source_precedence",
    }
    if defect_class not in known:
        raise ValueError(f"unknown controlled corruption class: {defect_class}")
    clean = pd.to_numeric(clean.reindex(OFFENSES_7), errors="coerce").fillna(0.0)
    corrupted = pd.to_numeric(corrupted.reindex(OFFENSES_7), errors="coerce").fillna(0.0)
    affected = set(affected_offenses or OFFENSES_7)
    records: list[dict[str, object]] = []
    for offense in OFFENSES_7:
        level1 = "valid_complete_year"
        level2 = "definitionally_complete"
        action = "accept_unchanged"
        detectability = "detected"
        repaired = float(corrupted[offense])
        reason = defect_class
        if defect_class in {"all_zero", "token_one", "single_offense_collapse"}:
            if offense in affected:
                level1 = "unresolved_review"
                action = "hold_unchanged"
        elif defect_class == "mislabeled_partial":
            if measured_coverage_months is None or not 0 < int(measured_coverage_months) < 12:
                raise ValueError("mislabeled_partial requires measured coverage in 1..11")
            level1 = "valid_partial_lower_bound"
            action = "partial_missing_increment"
            repaired = float(corrupted[offense] * 12.0 / int(measured_coverage_months))
        elif defect_class in {"selective_channel_omission", "chicago_category_omission"}:
            if offense in affected:
                level2 = "channel_omitted" if defect_class == "selective_channel_omission" else "wrong_category_mapping"
                action = "source_supported_replacement"
                repaired = float(clean[offense])
        elif defect_class == "offense_swap":
            if offense in affected:
                level2 = "documented_swap"
                action = "swap_reclassification"
                repaired = float(clean[offense])
        elif defect_class == "duplicate_identity":
            level1 = "source_identity_failure"
            action = "source_supported_replacement"
            repaired = float(clean[offense])
        elif defect_class == "uniform_factor":
            if external_comparator:
                level1 = "unresolved_review"
                action = "hold_unchanged"
                reason = "external_uniform_undercoverage_contradiction"
            else:
                detectability = "declared_internally_undetectable"
                reason = "coherent_uniform_factor_without_external_evidence"
        elif defect_class in {"genuine_spike", "genuine_decline"}:
            # A two-sided anomaly may be reviewed, but never automatically repaired.
            level1 = "unresolved_review"
            action = "hold_unchanged"
        elif defect_class == "stale_source_precedence":
            level1 = "source_identity_failure"
            action = "source_supported_replacement"
            repaired = float(clean[offense])
        records.append(
            {
                "offense": offense,
                "level1_admission_status": level1,
                "level2_semantic_status": level2,
                "reason_code": reason,
                "action": action,
                "detectability": detectability,
                "corrupted_count": float(corrupted[offense]),
                "repaired_count": repaired,
            }
        )
    return pd.DataFrame(records)


def _config_path(paths, name: str) -> Path:
    return Path(paths.repo_root) / "configs" / name


def load_admission_registry(paths) -> pd.DataFrame:
    path = _config_path(paths, "level_lane_admission_registry.csv")
    required = ["case_id", "ori9", "year", "offense", "adjudication", "reason_code", "reviewer", "evidence_note"]
    if not path.exists():
        raise LevelLaneConfigError(f"level-lane admission registry is missing: {path}")
    frame = pd.read_csv(path, dtype="string")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise LevelLaneConfigError(f"{path} is missing columns {missing}")
    frame = frame[required].copy()
    if frame.empty:
        return frame
    for column in required:
        if column != "year":
            frame[column] = frame[column].astype("string").str.strip()
    blank = frame.drop(columns="year").isna() | frame.drop(columns="year").eq("")
    if blank.any().any():
        bad_columns = sorted(blank.columns[blank.any()].tolist())
        raise LevelLaneConfigError(f"{path} has blank required values in {bad_columns}")
    frame["ori9"] = frame["ori9"].str.upper()
    if not frame["ori9"].str.fullmatch(r"[A-Z0-9]{9}").all():
        raise LevelLaneConfigError(f"{path} has malformed ori9 values")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    bad = sorted(set(frame["adjudication"]) - REGISTRY_ACTIONS)
    if bad:
        raise LevelLaneConfigError(f"{path} has unknown adjudications {bad}")
    bad_offense = sorted(set(frame["offense"]) - {*OFFENSES_7, "*"})
    if bad_offense:
        raise LevelLaneConfigError(f"{path} has unknown offenses {bad_offense}")
    if frame["case_id"].isna().any() or frame["case_id"].duplicated().any():
        raise LevelLaneConfigError(f"{path} requires unique nonblank case_id values")
    if frame.duplicated(["ori9", "year", "offense"]).any():
        raise LevelLaneConfigError(f"{path} has duplicate agency-year-offense rulings")
    wildcard = frame[frame["offense"].eq("*")][["ori9", "year"]].drop_duplicates()
    if not wildcard.empty:
        explicit = frame[~frame["offense"].eq("*")][["ori9", "year"]].drop_duplicates()
        overlap = wildcard.merge(explicit, on=["ori9", "year"], how="inner")
        if not overlap.empty:
            raise LevelLaneConfigError(
                f"{path} mixes wildcard and offense-specific rulings for an agency-year"
            )
    return frame


def load_hard_evidence(paths, *, year: int) -> pd.DataFrame:
    path = _config_path(paths, "level_lane_hard_evidence.csv")
    required = ["case_id", "ori9", "year", "offense", "level1_status", "level2_status", "reason_code", "repair_mode", "replacement_count", "external_check_status", "evidence_source"]
    if not path.exists():
        raise LevelLaneConfigError(f"level-lane hard-evidence contract is missing: {path}")
    frame = pd.read_csv(path, dtype="string").reindex(columns=required)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame = frame[frame["year"].eq(int(year))].copy()
    frame["ori9"] = frame["ori9"].str.strip().str.upper()
    frame["offense"] = frame["offense"].str.strip()
    frame["replacement_count"] = pd.to_numeric(frame["replacement_count"], errors="coerce")
    if not set(frame["level1_status"]).issubset(LEVEL1):
        raise LevelLaneConfigError(f"{path} has unknown level1_status values")
    if not set(frame["level2_status"]).issubset(LEVEL2):
        raise LevelLaneConfigError(f"{path} has unknown level2_status values")
    if not set(frame["offense"]).issubset({*OFFENSES_7, "*"}):
        raise LevelLaneConfigError(f"{path} has unknown offense values")
    if frame.duplicated(["ori9", "year", "offense"]).any():
        raise LevelLaneConfigError(f"{path} has duplicate agency-year-offense evidence")
    return frame


def _structural_zero_oris(paths, *, year: int) -> set[str]:
    path = Path(paths.repo_root) / "analysis_scratch" / "level_lane_screen" / "structural_zero_corroborated_all.csv"
    if not path.exists():
        raise LevelLaneConfigError(f"structural-zero truth population is missing: {path}")
    frame = pd.read_csv(path, dtype="string")
    if "ori9" not in frame.columns:
        raise LevelLaneConfigError(f"{path} is missing ori9")
    if "year" in frame.columns:
        frame = frame[pd.to_numeric(frame["year"], errors="coerce").eq(int(year))]
    return set(frame["ori9"].str.upper())


def _expand_contract(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in frame.to_dict(orient="records"):
        offenses = OFFENSES_7 if row["offense"] == "*" else (row["offense"],)
        for offense in offenses:
            record = dict(row)
            record["offense"] = offense
            rows.append(record)
    return pd.DataFrame(rows, columns=frame.columns).drop_duplicates(["ori9", "year", "offense"], keep="last")


def _soft_review_keys(panel: pd.DataFrame, *, year: int, protected: set[str]) -> pd.DataFrame:
    """Broad two-sided QA signals. They only enumerate holds; they never alter mass."""
    work = panel.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["preferred_count"] = pd.to_numeric(work["preferred_count"], errors="coerce").fillna(0.0)
    usable = work["usable_as_observed"].fillna(False).astype(bool)
    total = work.groupby(["ori9", "year"], dropna=False)["preferred_count"].sum().rename("part1")
    current = total[total.index.get_level_values("year") == int(year)]
    hist = total[total.index.get_level_values("year") < int(year)].reset_index()
    hist = hist.merge(
        work.loc[usable, ["ori9", "year"]].drop_duplicates(), on=["ori9", "year"], how="inner"
    )
    med = hist.groupby("ori9")["part1"].median()
    cur = current.reset_index().set_index("ori9")["part1"]
    ratio = cur / med.replace(0.0, np.nan)
    reasons = pd.DataFrame(index=cur.index)
    reasons["collapse_vs_history"] = ratio.lt(0.25)
    reasons["spike_vs_history"] = ratio.gt(4.0)
    target = work[work["year"].eq(int(year))].copy()
    offense_wide = target.pivot_table(index="ori9", columns="offense", values="preferred_count", aggfunc="sum", fill_value=0.0)
    coherence = pd.Series(False, index=offense_wide.index)
    if {"murder", "robbery", "aggravated_assault"}.issubset(offense_wide.columns):
        nonmurder = offense_wide.sum(axis=1) - offense_wide["murder"]
        coherence |= offense_wide["murder"].gt(0) & (nonmurder / offense_wide["murder"].replace(0, np.nan)).lt(90)
        coherence |= offense_wide["robbery"].ge(10) & (offense_wide["aggravated_assault"] / offense_wide["robbery"].replace(0, np.nan)).lt(1)
    reasons["offense_mix_anomaly"] = coherence.reindex(reasons.index).fillna(False)
    # A syntactically complete seven-cell vector can still be an implausible
    # token submission.  Hold, rather than silently bless, sizeable agencies
    # below 300 Part-I offenses per 100k; this is a disclosure/review floor and
    # never changes an official value without hard evidence.
    current_population = (
        target.groupby("ori9", dropna=False)["population"].max()
        if "population" in target.columns
        else pd.Series(dtype=float)
    )
    current_population = pd.to_numeric(current_population, errors="coerce")
    part1_rate = cur / current_population.reindex(cur.index).replace(0.0, np.nan) * 1e5
    reasons["implausibly_low_part1_rate"] = (
        current_population.reindex(reasons.index).ge(5_000)
        & part1_rate.reindex(reasons.index).lt(300.0)
    ).fillna(False)
    reasons.loc[reasons.index.isin(protected), :] = False
    records = []
    for ori, row in reasons.iterrows():
        codes = [name for name, value in row.items() if bool(value)]
        if codes:
            records.append({"ori9": str(ori), "soft_reason_codes": "|".join(codes)})
    return pd.DataFrame(records, columns=["ori9", "soft_reason_codes"])


def _attach_population_for_plausibility(panel: pd.DataFrame, *, paths) -> pd.DataFrame:
    """Attach the canonical agency population used by absolute-rate QA signals.

    The observation panel is intentionally offense-centric and does not reliably
    carry population through every production build.  The plausibility floor must
    therefore resolve it from the canonical agency master rather than silently
    disabling itself when the optional observation column is absent.
    """
    out = panel.copy()
    current = (
        pd.to_numeric(out["population"], errors="coerce")
        if "population" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    state_dir = (
        Path(paths.state_dir)
        if hasattr(paths, "state_dir")
        else Path(paths.repo_root) / "state"
    )
    master_path = state_dir / "reference" / "agency_master.parquet"
    if not master_path.exists():
        # Unit-level corruption tests deliberately construct a minimal repository
        # without reference artifacts.  They still exercise the history/mix arms;
        # only the population-dependent absolute-rate arm is unavailable there.
        out["population"] = current
        return out
    master = pd.read_parquet(
        master_path, columns=["ori9", "population_latest_nibrs"]
    )
    master["ori9"] = master["ori9"].astype("string").str.upper()
    master["population_latest_nibrs"] = pd.to_numeric(
        master["population_latest_nibrs"], errors="coerce"
    )
    population_by_ori = (
        master.dropna(subset=["ori9"])
        .drop_duplicates("ori9", keep="last")
        .set_index("ori9")["population_latest_nibrs"]
    )
    resolved = out["ori9"].astype("string").str.upper().map(population_by_ori)
    out["population"] = current.fillna(resolved)
    return out


def _measure_peer_log_change_envelope(
    panel: pd.DataFrame,
    *,
    target_year: int,
    exclude_target_rows: pd.Series,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure admitted-panel P1/P99 log changes and attach each row's comparison.

    The peer sample is the target-year admitted complete panel joined to its own
    preceding-year offense count. The prior count floor is policy-specified, not fitted;
    only the two envelope endpoints are measured from the build's admitted panel.
    """
    keys = ["ori9", "offense"]
    target = panel["year"].eq(int(target_year))
    previous = panel["year"].eq(int(target_year) - 1)
    current = panel.loc[
        target
        & ~exclude_target_rows
        & panel["level1_admission_status"].eq("valid_complete_year")
        & panel["level2_semantic_status"].eq("definitionally_complete")
        & panel["usable_as_observed"].fillna(False).astype(bool),
        [*keys, "level_lane_original_count"],
    ].rename(columns={"level_lane_original_count": "current_count"})
    prior = panel.loc[
        previous,
        [*keys, "level_lane_original_count"],
    ].rename(columns={"level_lane_original_count": "prior_count"})
    if current.duplicated(keys).any() or prior.duplicated(keys).any():
        raise LevelLaneConfigError("peer-envelope panel has duplicate agency-offense-year rows")
    sample = current.merge(prior, on=keys, how="inner", validate="one_to_one")
    for column in ("current_count", "prior_count"):
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample = sample[
        sample["prior_count"].ge(PEER_ENVELOPE_MIN_PRIOR_COUNT)
        & sample["current_count"].ge(0.0)
    ].copy()
    sample["log_change"] = -np.inf
    positive_sample = sample["current_count"].gt(0.0)
    sample.loc[positive_sample, "log_change"] = np.log(
        sample.loc[positive_sample, "current_count"]
        / sample.loc[positive_sample, "prior_count"]
    )

    rows: list[dict[str, object]] = []
    for offense in OFFENSES_7:
        values = sample.loc[sample["offense"].eq(offense), "log_change"].to_numpy(dtype=float)
        if len(values) == 0:
            if strict:
                raise LevelLaneConfigError(
                    f"peer-envelope sample is empty for {offense} in {target_year}"
                )
            rows.append(
                {
                    "offense": offense,
                    "level_peer_sample_n": 0,
                    "level_peer_log_change_p01": np.nan,
                    "level_peer_log_change_p99": np.nan,
                }
            )
            continue
        lower, upper = np.quantile(values, [0.01, 0.99])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            if strict:
                raise LevelLaneConfigError(
                    f"peer-envelope endpoints are invalid for {offense}: {lower}, {upper}"
                )
            lower, upper = np.nan, np.nan
        rows.append(
            {
                "offense": offense,
                "level_peer_sample_n": int(len(values)),
                "level_peer_log_change_p01": float(lower),
                "level_peer_log_change_p99": float(upper),
            }
        )
    envelope = pd.DataFrame(rows)

    all_current = panel.loc[target, [*keys, "level_lane_original_count"]].rename(
        columns={"level_lane_original_count": "current_count"}
    )
    comparisons = all_current.merge(prior, on=keys, how="left", validate="one_to_one")
    for column in ("current_count", "prior_count"):
        comparisons[column] = pd.to_numeric(comparisons[column], errors="coerce")
    comparisons["level_peer_comparable"] = (
        comparisons["prior_count"].ge(PEER_ENVELOPE_MIN_PRIOR_COUNT)
        & comparisons["current_count"].ge(0.0)
    )
    comparisons["level_peer_log_change_2023_2024"] = np.nan
    positive = comparisons["level_peer_comparable"] & comparisons["current_count"].gt(0.0)
    zero = comparisons["level_peer_comparable"] & comparisons["current_count"].eq(0.0)
    comparisons.loc[positive, "level_peer_log_change_2023_2024"] = np.log(
        comparisons.loc[positive, "current_count"]
        / comparisons.loc[positive, "prior_count"]
    )
    comparisons.loc[zero, "level_peer_log_change_2023_2024"] = -np.inf
    comparisons = comparisons.rename(columns={"prior_count": "level_peer_prior_count_2023"})
    comparisons = comparisons.merge(envelope, on="offense", how="left", validate="many_to_one")
    return envelope, comparisons[
        [
            *keys,
            "level_peer_prior_count_2023",
            "level_peer_comparable",
            "level_peer_log_change_2023_2024",
            "level_peer_sample_n",
            "level_peer_log_change_p01",
            "level_peer_log_change_p99",
        ]
    ]


def load_nibrs_transition_years(paths, *, year: int) -> pd.Series:
    """Each ORI's NIBRS start year, from the CDE roster the build already pulls."""
    data_dir = (
        Path(paths.data_dir)
        if getattr(paths, "data_dir", None) is not None
        else Path(paths.repo_root) / "data"
    )
    if not data_dir.exists():
        return pd.Series(dtype="Int64")
    roster_dir = data_dir / f"FBI-CDE-Agency-Rosters-{int(year)}" / "parsed"
    path = roster_dir / f"agency_rosters_{int(year)}.parquet"
    if not path.exists():
        candidates = sorted(
            data_dir.glob("FBI-CDE-Agency-Rosters-*/parsed/agency_rosters_*.parquet")
        )
        if not candidates:
            return pd.Series(dtype="Int64")
        path = candidates[-1]
    roster = pd.read_parquet(path, columns=["ori", "nibrs_start_date"])
    roster["ori"] = roster["ori"].astype("string").str.upper()
    start = pd.to_datetime(roster["nibrs_start_date"], errors="coerce")
    out = pd.Series(start.dt.year.to_numpy(), index=roster["ori"].to_numpy(), dtype="Int64")
    return out[~out.index.duplicated(keep="last")].dropna()


def _measure_own_history_log_change_bounds(
    panel: pd.DataFrame,
    *,
    target_year: int,
    nibrs_transition_years: pd.Series | None = None,
) -> pd.DataFrame:
    """Each agency-offense's admission bound, read off its own filing history.

    The peer first percentile is a statement about agencies in general: Cicero's
    burglary falling 335 to 91 clears it because somewhere in the country a comparable
    agency really did drop that far. The agency's own 2018-target log changes are the
    statement that matters -- an agency that has never moved more than a fifth in a year
    has not suddenly lost two thirds of its burglaries.

    Years before an agency's NIBRS start date are dropped from the window when the
    transition guard is on: the break there is definitional and would otherwise widen
    the agency's own band enough to admit anything.
    """
    keys = ["ori9", "offense"]
    history = panel.loc[
        pd.to_numeric(panel["year"], errors="coerce").lt(int(target_year))
        & panel["usable_as_observed"].fillna(False).astype(bool)
        & pd.to_numeric(panel["preferred_months_reported"], errors="coerce").ge(12.0),
        [*keys, "year", "level_lane_original_count"],
    ].copy()
    if history.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "level_own_clean_years",
                "level_own_log_change_lower",
                "level_own_log_change_upper",
                "level_own_log_change_min",
            ]
        )
    history["year"] = pd.to_numeric(history["year"], errors="coerce").astype("Int64")
    history["count"] = pd.to_numeric(
        history["level_lane_original_count"], errors="coerce"
    )
    history = history.dropna(subset=["count"])
    if nibrs_transition_years is not None and len(nibrs_transition_years):
        transition = history["ori9"].astype("string").str.upper().map(nibrs_transition_years)
        history = history[
            transition.isna()
            | history["year"].ge(transition + NIBRS_TRANSITION_GUARD_YEARS)
        ]
    history = history.sort_values([*keys, "year"])
    clean_years = history.groupby(keys, dropna=False)["year"].nunique().rename(
        "level_own_clean_years"
    )

    # Consecutive-year changes only: a gap year is not one year's worth of movement.
    grouped = history.groupby(keys, dropna=False, sort=False)
    history["prev_year"] = grouped["year"].shift(1)
    history["prev_count"] = grouped["count"].shift(1)
    steps = history[
        history["prev_year"].notna()
        & history["year"].sub(history["prev_year"]).eq(1)
        & pd.to_numeric(history["prev_count"], errors="coerce").ge(
            PEER_ENVELOPE_MIN_PRIOR_COUNT
        )
    ].copy()
    if steps.empty:
        bounds = clean_years.reset_index()
        bounds["level_own_log_change_lower"] = np.nan
        bounds["level_own_log_change_upper"] = np.nan
        bounds["level_own_log_change_min"] = np.nan
        return bounds
    steps["log_change"] = np.where(
        pd.to_numeric(steps["count"], errors="coerce").gt(0.0),
        np.log(
            pd.to_numeric(steps["count"], errors="coerce").clip(lower=1e-9)
            / pd.to_numeric(steps["prev_count"], errors="coerce")
        ),
        -np.inf,
    )
    finite = steps[np.isfinite(steps["log_change"])]
    stats = (
        finite.groupby(keys, dropna=False)["log_change"]
        .agg(["count", "mean", "std", "min"])
        .rename(
            columns={
                "count": "steps",
                "mean": "centre",
                "std": "spread",
                "min": "level_own_log_change_min",
            }
        )
    )
    stats["spread"] = stats["spread"].fillna(0.0).clip(lower=OWN_HISTORY_SPREAD_FLOOR)
    stats["level_own_log_change_lower"] = (
        stats["centre"] - OWN_HISTORY_SPREAD_MULTIPLE * stats["spread"]
    )
    stats["level_own_log_change_upper"] = (
        stats["centre"] + OWN_HISTORY_SPREAD_MULTIPLE * stats["spread"]
    )
    bounds = stats.reset_index().merge(clean_years.reset_index(), on=keys, how="outer")
    # A bound wants at least two changes behind it, which is three clean years.
    insufficient = (
        pd.to_numeric(bounds["steps"], errors="coerce").fillna(0.0).lt(2.0)
        | pd.to_numeric(bounds["level_own_clean_years"], errors="coerce")
        .fillna(0.0)
        .lt(OWN_HISTORY_MIN_CLEAN_YEARS)
    )
    bounds.loc[insufficient, ["level_own_log_change_lower", "level_own_log_change_upper"]] = np.nan
    bounds.loc[insufficient, "level_own_log_change_min"] = np.nan
    return bounds[
        [
            *keys,
            "level_own_clean_years",
            "level_own_log_change_lower",
            "level_own_log_change_upper",
            "level_own_log_change_min",
        ]
    ]


def apply_level_lane_admission(panel: pd.DataFrame, *, paths, target_year: int) -> AdmissionArtifacts:
    if panel.empty:
        return AdmissionArtifacts(panel.copy(), pd.DataFrame(), pd.DataFrame())
    policy_switches = level_lane_policy()
    out = panel.copy()
    out["ori9"] = out["ori9"].astype("string").str.upper()
    out["offense"] = out["offense"].astype("string")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["level_lane_original_count"] = pd.to_numeric(out["preferred_count"], errors="coerce")
    out["level_lane_replacement_count"] = np.nan
    out["level1_admission_status"] = "valid_complete_year"
    out["level2_semantic_status"] = "definitionally_complete"
    out["level_admission_reason"] = "selected_lane_supported"
    out["level_repair_mode"] = "none"
    out["external_check_status"] = "unavailable"
    out["level_review_hold"] = False

    supported = out["usable_as_observed"].fillna(False).astype(bool)
    partial = out["current_row_is_true_partial"].fillna(False).astype(bool)
    out.loc[partial, "level1_admission_status"] = "valid_partial_lower_bound"
    out.loc[partial, "level_admission_reason"] = "measured_partial_coverage"
    out.loc[partial, "level_repair_mode"] = "partial_missing_increment"
    out.loc[~supported & ~partial, "level1_admission_status"] = "coverage_defective"
    out.loc[~supported & ~partial, "level2_semantic_status"] = "semantically_unusable"
    out.loc[~supported & ~partial, "level_admission_reason"] = "selected_lane_not_usable"
    out.loc[~supported & ~partial, "level_repair_mode"] = "decayed_own_history_or_pooled"

    zero_oris = _structural_zero_oris(paths, year=int(target_year))
    zero_mask = out["year"].eq(int(target_year)) & out["ori9"].isin(zero_oris)
    out.loc[zero_mask, "level1_admission_status"] = "corroborated_structural_zero"
    out.loc[zero_mask, "level2_semantic_status"] = "definitionally_complete"
    out.loc[zero_mask, "level_admission_reason"] = "corroborated_structural_zero_history"
    out.loc[zero_mask, "level_repair_mode"] = "locked_structural_zero"
    out.loc[zero_mask, "preferred_count"] = 0.0
    out.loc[zero_mask, "usable_as_observed"] = True
    out.loc[zero_mask, "current_row_is_true_partial"] = False

    hard = _expand_contract(load_hard_evidence(paths, year=int(target_year)))
    if not hard.empty:
        hard = hard.rename(columns={c: f"_hard_{c}" for c in hard.columns if c not in {"ori9", "year", "offense"}})
        out = out.merge(hard, on=["ori9", "year", "offense"], how="left")
        hit = out["_hard_case_id"].notna()
        for dest, src in (
            ("level1_admission_status", "_hard_level1_status"),
            ("level2_semantic_status", "_hard_level2_status"),
            ("level_admission_reason", "_hard_reason_code"),
            ("level_repair_mode", "_hard_repair_mode"),
            ("external_check_status", "_hard_external_check_status"),
        ):
            out.loc[hit, dest] = out.loc[hit, src]
        replacement = pd.to_numeric(out["_hard_replacement_count"], errors="coerce")
        out.loc[hit & replacement.notna(), "level_lane_replacement_count"] = replacement[hit & replacement.notna()]
        out["level_lane_hard_case_id"] = out["_hard_case_id"].astype("string")
        out = out.drop(columns=[c for c in out.columns if c.startswith("_hard_")])
    else:
        out["level_lane_hard_case_id"] = pd.NA

    # A filed all-zero/token-one vector is not an observed complete year merely
    # because the source supplied seven numeric cells.  The locked historical-zero
    # population above is the only automatic exception; explicit hard-evidence
    # acceptances remain protected as well.
    target = out["year"].eq(int(target_year))
    target_rows = out.loc[target].copy()
    vector = target_rows.groupby("ori9", dropna=False).agg(
        part1=("level_lane_original_count", lambda values: pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0).sum()),
        months=("preferred_months_reported", lambda values: pd.to_numeric(values, errors="coerce").max()),
        offenses=("offense", "nunique"),
    )
    # Token-one vectors are always near-zero.  A slightly larger vector is also
    # near-zero when it is both tiny in absolute terms and less than half the
    # agency's own typical prior full-year vector.  The second arm catches abrupt
    # low-filed collapses such as Ramah Navajo (7 in 2024 versus a prior-year
    # median above 14) without treating ordinary small-agency volatility as a
    # coverage defect.  The corroborated structural-zero registry remains the
    # only automatic zero-history acceptance below.
    history_rows = out.loc[out["year"].lt(int(target_year))].copy()
    history_rows = history_rows[
        history_rows["usable_as_observed"].fillna(False).astype(bool)
        & pd.to_numeric(history_rows["preferred_months_reported"], errors="coerce").ge(12.0)
    ]
    if history_rows.empty:
        vector["prior_full_year_median"] = np.nan
    else:
        history_vectors = (
            history_rows.groupby(["ori9", "year"], dropna=False)["level_lane_original_count"]
            .sum(min_count=1)
            .rename("part1")
            .reset_index()
        )
        prior_median = history_vectors.groupby("ori9")["part1"].median()
        vector["prior_full_year_median"] = vector.index.map(prior_median)
    hard_evidence_oris = set(
        out.loc[target & out["level_lane_hard_case_id"].notna(), "ori9"].astype(str)
    )
    footprint_complete_state_units = set(
        out.loc[
            target
            & out.get(
                "preferred_raw_data_source", pd.Series(pd.NA, index=out.index)
            )
            .astype("string")
            .eq("ct_despp_town"),
            "ori9",
        ].astype(str)
    )
    tiny_vector = vector["part1"].le(1.0 + 1e-9)
    footprint_path = Path(paths.repo_root) / "configs" / "overlap_custom_footprints.csv"
    custom_footprint_oris: set[str] = set()
    if footprint_path.exists():
        custom_footprint_oris = set(
            pd.read_csv(footprint_path, usecols=["ori"], dtype="string")["ori"]
            .str.upper()
            .dropna()
            .astype(str)
        )
    collapsed_tiny_vector = (
        vector["part1"].le(10.0 + 1e-9)
        & vector["prior_full_year_median"].gt(0.0)
        & vector["part1"].lt(0.5 * vector["prior_full_year_median"])
        & vector.index.to_series().astype(str).isin(custom_footprint_oris).to_numpy()
    )
    near_zero_oris = set(
        vector.index[
            (tiny_vector | collapsed_tiny_vector)
            & vector["months"].gt(0.0)
            & vector["offenses"].ge(len(OFFENSES_7))
        ].astype(str)
    ) - zero_oris - hard_evidence_oris - footprint_complete_state_units
    near_zero = target & out["ori9"].isin(near_zero_oris)
    out.loc[near_zero, "level1_admission_status"] = "coverage_defective"
    out.loc[near_zero, "level2_semantic_status"] = "semantically_unusable"
    out.loc[near_zero, "level_admission_reason"] = "uncorroborated_near_zero_filed_vector"
    out.loc[near_zero, "level_repair_mode"] = "decayed_own_history_or_pooled"

    # FBI contract-policing identity is numerator-exclusive: the covered agency
    # contributes geometry through the reference crosswalk, while its own row must
    # carry no second numerator into the ledger.
    coverage = load_covered_by_relationships(paths, year=int(target_year))
    covered_oris = set(coverage["ori9"].astype(str))
    covered = target & out["ori9"].isin(covered_oris) & ~out["ori9"].isin(zero_oris)
    out.loc[covered, "level1_admission_status"] = "source_identity_failure"
    out.loc[covered, "level2_semantic_status"] = "semantically_unusable"
    out.loc[covered, "level_admission_reason"] = "fbi_covered_by_ori"
    out.loc[covered, "level_repair_mode"] = "covered_by_covering_agency"

    hard_partial = target & out["level1_admission_status"].eq("valid_partial_lower_bound")
    out.loc[hard_partial, "usable_as_observed"] = False
    out.loc[hard_partial, "current_row_is_true_partial"] = True
    reason = out["level_admission_reason"].astype("string")
    out.loc[hard_partial & reason.str.contains("three_month", na=False), "preferred_months_reported"] = 3.0
    out.loc[hard_partial & reason.str.contains("four_month", na=False), "preferred_months_reported"] = 4.0
    invalid = target & (
        out["level1_admission_status"].isin(INVALID_LEVEL1)
        | out["level2_semantic_status"].isin(INVALID_LEVEL2)
    )
    has_source_replacement = invalid & out["level_repair_mode"].isin({"source_supported_replacement", "swap_reclassification"})
    out.loc[invalid & ~has_source_replacement, "usable_as_observed"] = False
    out.loc[invalid & ~has_source_replacement, "current_row_is_true_partial"] = False
    # Invalid fragments remain in level_lane_original_count for audit, but are not a
    # lower bound and therefore cannot enter the reported/control mass ledger.
    out.loc[invalid, "preferred_count"] = 0.0
    out.loc[has_source_replacement, "usable_as_observed"] = False
    out.loc[has_source_replacement, "current_row_is_true_partial"] = False

    plausibility_panel = _attach_population_for_plausibility(out, paths=paths)
    soft = _soft_review_keys(
        plausibility_panel, year=int(target_year), protected=zero_oris
    )
    if not soft.empty:
        out = out.merge(soft, on="ori9", how="left")
    else:
        out["soft_reason_codes"] = pd.NA
    # Soft plausibility signals apply only to an otherwise admissible complete-year
    # observation. They must not overwrite a hard coverage/semantic failure (or a
    # partial lower bound): those rows are already owned by a named repair path.
    target = out["year"].eq(int(target_year))
    soft_hit = (
        target
        & out["soft_reason_codes"].notna()
        & out["level_lane_hard_case_id"].isna()
        & out["level1_admission_status"].eq("valid_complete_year")
        & out["level2_semantic_status"].eq("definitionally_complete")
        & out["usable_as_observed"].fillna(False).astype(bool)
    )
    hard_unresolved = target & (
        out["level1_admission_status"].eq("unresolved_review")
        | out["level_repair_mode"].eq("unchanged_review_hold")
    )
    # (f) A partial year whose annualisation lands far below the agency's own clean
    # years is a token submission. Before the refresh these agencies filed nothing and
    # took a pooled control; a four-month token now displaces it with a number an order
    # of magnitude too small, and the partial branch is reached before the soft screen,
    # so the implausibility the screen records never acts on anything.
    partial_floor_oris: set[str] = set()
    if policy_switches.partial_year_history_floor:
        target_partial = (
            target
            & out["level1_admission_status"].eq("valid_partial_lower_bound")
            & out["level_lane_hard_case_id"].isna()
        )
        if bool(target_partial.any()):
            months = (
                pd.to_numeric(out["preferred_months_reported"], errors="coerce")
                .clip(lower=1.0, upper=12.0)
            )
            annualised = (
                pd.to_numeric(out["level_lane_original_count"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                * (12.0 / months)
            ).where(target_partial, 0.0)
            agency_annualised = annualised.groupby(out["ori9"].astype(str)).sum()
            prior_median = vector["prior_full_year_median"]
            partial_floor_oris = {
                str(ori)
                for ori in out.loc[target_partial, "ori9"].astype(str).unique()
                if float(prior_median.get(ori, float("nan")) or 0.0) > 0.0
                and float(agency_annualised.get(ori, 0.0))
                < PARTIAL_YEAR_HISTORY_FLOOR_RATIO * float(prior_median.get(ori))
            }
    if partial_floor_oris:
        floored = target & out["ori9"].astype(str).isin(partial_floor_oris)
        out.loc[floored, "level1_admission_status"] = "coverage_defective"
        out.loc[floored, "level2_semantic_status"] = "semantically_unusable"
        out.loc[floored, "level_admission_reason"] = "partial_year_below_own_history_floor"
        out.loc[floored, "level_repair_mode"] = "decayed_own_history_or_pooled"
        out.loc[floored, "usable_as_observed"] = False
        out.loc[floored, "current_row_is_true_partial"] = False
        out.loc[floored, "preferred_count"] = 0.0

    out["_soft_policy_candidate"] = soft_hit | hard_unresolved
    out["_hard_unresolved_candidate"] = hard_unresolved
    out["level_policy_resolution"] = "not_applicable"
    out["level_lane_registry_case_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["level_lane_registry_adjudication"] = pd.Series(pd.NA, index=out.index, dtype="string")

    # A registry ruling owns a signaled offense before the mechanical policy. Wildcard
    # rulings apply only to the signaled rows of that agency-year so an existing hard
    # source replacement (Chicago is the canonical shape) cannot be double-adjudicated.
    registry_raw = load_admission_registry(paths)
    registry_raw = registry_raw[registry_raw["year"].eq(int(target_year))].copy()
    if not registry_raw.empty:
        registry_raw["_registry_wildcard"] = registry_raw["offense"].eq("*")
        registry = _expand_contract(registry_raw)
        registry = registry.rename(
            columns={
                c: f"_registry_{c}"
                for c in registry.columns
                if c not in {"ori9", "year", "offense"}
            }
        )
        panel_keys = out.loc[target, ["ori9", "year", "offense"]]
        unmatched = registry.merge(
            panel_keys,
            on=["ori9", "year", "offense"],
            how="left",
            indicator=True,
        )
        unmatched = unmatched[unmatched["_merge"].eq("left_only")]
        if not unmatched.empty:
            raise LevelLaneConfigError(
                "admission registry contains rulings with no target-year panel row: "
                + str(
                    unmatched[["ori9", "year", "offense"]]
                    .head(20)
                    .to_dict(orient="records")
                )
            )
        out = out.merge(registry, on=["ori9", "year", "offense"], how="left")
        target = out["year"].eq(int(target_year))
        soft_hit = out["_soft_policy_candidate"].fillna(False).astype(bool)
        registry_match = out["_registry_case_id"].notna()
        wildcard_match = out["_registry__registry_wildcard"].eq(True)
        action = out["_registry_adjudication"].astype("string")
        # Hard evidence is applied before the soft screen and can remove an
        # agency-wide signal that existed when its other offenses were reviewed.
        # Preserve those explicit acceptances when the row remains an otherwise
        # admissible complete observation. Refusals and wildcard rulings still
        # require a live soft-policy row, and no registry action may override a
        # hard-evidence row.
        reviewed_complete_accept = (
            registry_match
            & ~wildcard_match
            & action.eq("accept_unchanged")
            & out["level_lane_hard_case_id"].isna()
            & out["level1_admission_status"].eq("valid_complete_year")
            & out["level2_semantic_status"].eq("definitionally_complete")
            & out["usable_as_observed"].fillna(False).astype(bool)
        )
        explicit_nonsoft = (
            registry_match
            & ~wildcard_match
            & ~soft_hit
            & ~reviewed_complete_accept
        )
        if explicit_nonsoft.any():
            raise LevelLaneConfigError(
                "offense-specific admission registry rulings must match a soft-policy row: "
                + str(
                    out.loc[explicit_nonsoft, ["ori9", "year", "offense"]]
                    .head(20)
                    .to_dict(orient="records")
                )
            )
        wildcard_cases = set(
            out.loc[
                registry_match
                & wildcard_match,
                "_registry_case_id",
            ].astype(str)
        )
        matched_wildcard_cases = set(
            out.loc[
                registry_match
                & wildcard_match
                & soft_hit,
                "_registry_case_id",
            ].astype(str)
        )
        if wildcard_cases - matched_wildcard_cases:
            raise LevelLaneConfigError(
                "wildcard admission registry rulings matched no soft-policy rows: "
                + str(sorted(wildcard_cases - matched_wildcard_cases))
            )
        registry_match &= soft_hit | reviewed_complete_accept
        accept = registry_match & action.eq("accept_unchanged")
        refuse_repair = registry_match & action.eq("refuse_repair")
        refuse_silent = registry_match & action.eq("refuse_silent")
        out.loc[accept, "level1_admission_status"] = "valid_complete_year"
        out.loc[accept, "level2_semantic_status"] = "definitionally_complete"
        out.loc[accept, "level_repair_mode"] = "none"
        out.loc[accept, "level_policy_resolution"] = "registry_accept_unchanged"
        out.loc[accept, "level_admission_reason"] = out.loc[accept, "_registry_reason_code"]
        refuse = refuse_repair | refuse_silent
        out.loc[refuse, "level1_admission_status"] = "coverage_defective"
        out.loc[refuse, "level2_semantic_status"] = "semantically_unusable"
        out.loc[refuse_repair, "level_repair_mode"] = "decayed_own_history_or_pooled"
        out.loc[refuse_silent, "level_repair_mode"] = "soft_benchmark_reconciliation"
        out.loc[refuse_repair, "level_policy_resolution"] = "registry_refuse_repair"
        out.loc[refuse_silent, "level_policy_resolution"] = "registry_refuse_silent"
        out.loc[refuse, "level_admission_reason"] = out.loc[refuse, "_registry_reason_code"]
        out.loc[refuse, "usable_as_observed"] = False
        out.loc[refuse, "current_row_is_true_partial"] = False
        out.loc[refuse, "preferred_count"] = 0.0
        out.loc[registry_match, "level_lane_registry_case_id"] = out.loc[
            registry_match, "_registry_case_id"
        ].astype("string")
        out.loc[registry_match, "level_lane_registry_adjudication"] = out.loc[
            registry_match, "_registry_adjudication"
        ].astype("string")
        out = out.drop(columns=[c for c in out.columns if c.startswith("_registry_")])
    else:
        accept = pd.Series(False, index=out.index)
        registry_match = pd.Series(False, index=out.index)

    target = out["year"].eq(int(target_year))
    soft_hit = out["_soft_policy_candidate"].fillna(False).astype(bool)
    hard_unresolved = out["_hard_unresolved_candidate"].fillna(False).astype(bool)
    registry_match = out["level_lane_registry_case_id"].notna()
    registry_accept = (
        out["level_lane_registry_adjudication"]
        .eq("accept_unchanged")
        .fillna(False)
        .astype(bool)
    )
    calibration_excluded = soft_hit & ~registry_accept
    new_screens = bool(
        policy_switches.own_history_admission_bound or policy_switches.joint_vector_gate
    )
    if soft_hit.any() or new_screens:
        _envelope, comparisons = _measure_peer_log_change_envelope(
            out,
            target_year=int(target_year),
            exclude_target_rows=calibration_excluded,
            strict=bool(soft_hit.any()),
        )
        out = out.merge(
            comparisons,
            on=["ori9", "offense"],
            how="left",
            validate="many_to_one",
        )
    else:
        out["level_peer_prior_count_2023"] = np.nan
        out["level_peer_comparable"] = False
        out["level_peer_log_change_2023_2024"] = np.nan
        out["level_peer_sample_n"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["level_peer_log_change_p01"] = np.nan
        out["level_peer_log_change_p99"] = np.nan

    nibrs_transition_years = (
        load_nibrs_transition_years(paths, year=int(target_year))
        if policy_switches.nibrs_transition_guard
        else pd.Series(dtype="Int64")
    )
    out["level_nibrs_start_year"] = (
        out["ori9"].astype("string").str.upper().map(nibrs_transition_years)
        if len(nibrs_transition_years)
        else pd.Series(pd.NA, index=out.index, dtype="Int64")
    )
    if new_screens:
        own_bounds = _measure_own_history_log_change_bounds(
            out,
            target_year=int(target_year),
            nibrs_transition_years=(
                nibrs_transition_years
                if policy_switches.nibrs_transition_guard
                else None
            ),
        )
        out = out.merge(own_bounds, on=["ori9", "offense"], how="left", validate="many_to_one")
    else:
        out["level_own_clean_years"] = np.nan
        out["level_own_log_change_lower"] = np.nan
        out["level_own_log_change_upper"] = np.nan
        out["level_own_log_change_min"] = np.nan
    target = out["year"].eq(int(target_year))
    soft_hit = out["_soft_policy_candidate"].fillna(False).astype(bool)
    registry_match = out["level_lane_registry_case_id"].notna()
    policy = soft_hit & ~registry_match
    reasons = out["soft_reason_codes"].astype("string")
    implausibly_low = reasons.str.contains("implausibly_low_part1_rate", regex=False, na=False)
    history_signal = (
        reasons.str.contains("collapse_vs_history", regex=False, na=False)
        | reasons.str.contains("spike_vs_history", regex=False, na=False)
    )
    mix_signal = reasons.str.contains("offense_mix_anomaly", regex=False, na=False)
    comparable = out["level_peer_comparable"].eq(True)
    log_change = pd.to_numeric(out["level_peer_log_change_2023_2024"], errors="coerce")
    peer_lower = pd.to_numeric(out["level_peer_log_change_p01"], errors="coerce")
    peer_upper = pd.to_numeric(out["level_peer_log_change_p99"], errors="coerce")
    own_lower = pd.to_numeric(out["level_own_log_change_lower"], errors="coerce")
    own_upper = pd.to_numeric(out["level_own_log_change_upper"], errors="coerce")
    # (a) The agency's own band is the primary bound; the peer envelope is what an
    # agency without three clean years of its own falls back to.
    own_bound_available = (
        own_lower.notna()
        & own_upper.notna()
        & pd.to_numeric(out["level_own_clean_years"], errors="coerce").ge(
            OWN_HISTORY_MIN_CLEAN_YEARS
        )
    )
    use_own_bound = own_bound_available if policy_switches.own_history_admission_bound else pd.Series(False, index=out.index)
    out["level_admission_bound_basis"] = np.where(
        use_own_bound, "own_history", "peer_envelope"
    )
    lower = own_lower.where(use_own_bound, peer_lower)
    upper = own_upper.where(use_own_bound, peer_upper)
    out["level_admission_bound_lower"] = lower
    out["level_admission_bound_upper"] = upper
    beyond_collapse = comparable & log_change.lt(lower)
    beyond_envelope = beyond_collapse | (comparable & log_change.gt(upper))
    repair = policy & (
        hard_unresolved
        | implausibly_low
        | (history_signal & beyond_envelope)
        | (~history_signal & mix_signal & beyond_collapse)
    )

    # A row an explicit ruling has accepted, a hard-evidence row, and a row already
    # owned by another repair path are all out of reach of the two new screens.
    # Recomputed here: the peer/own-bound merges above rebuild `out`, so a mask taken
    # before them no longer lines up with it.
    registry_accept = (
        out["level_lane_registry_adjudication"]
        .eq("accept_unchanged")
        .fillna(False)
        .astype(bool)
    )
    screen_eligible = (
        target
        & out["level_lane_hard_case_id"].isna()
        & ~registry_accept
        & out["level1_admission_status"].eq("valid_complete_year")
        & out["level2_semantic_status"].eq("definitionally_complete")
        & out["usable_as_observed"].fillna(False).astype(bool)
    )
    # (a) A collapse past the agency's own band is itself a refusal reason. Cicero's
    # vector carries no agency-level soft code at all -- its Part 1 total only falls a
    # quarter because larceny rises while everything else halves -- so a screen that
    # only re-bounds already-signalled rows would never see it.
    own_bound_breach = (
        screen_eligible
        & use_own_bound
        & beyond_collapse
        if policy_switches.own_history_admission_bound
        else pd.Series(False, index=out.index)
    )
    # (b) Offenses breaching together are a filing failure whatever each single margin
    # says. Counted over the offenses that are comparable at all, and applied to the
    # whole vector, because a vector is what the agency filed.
    own_min = pd.to_numeric(out["level_own_log_change_min"], errors="coerce")
    joint_offense_breach = (
        comparable
        & own_min.notna()
        & log_change.lt(own_min)
        & log_change.lt(JOINT_VECTOR_MIN_LOG_DROP)
    )
    breach_for_joint = (target & joint_offense_breach).astype(int)
    joint_breach_count = breach_for_joint.groupby(out["ori9"].astype(str)).transform("sum")
    out["level_joint_vector_breach_count"] = joint_breach_count.where(target, 0)
    divergent_rise = (target & comparable & log_change.gt(JOINT_VECTOR_DIVERGENT_RISE)).astype(int)
    divergent_count = divergent_rise.groupby(out["ori9"].astype(str)).transform("sum")
    out["level_joint_vector_divergent_count"] = divergent_count.where(target, 0)
    joint_breach = (
        screen_eligible
        & joint_breach_count.ge(JOINT_VECTOR_MIN_BREACHES)
        & divergent_count.ge(1)
        if policy_switches.joint_vector_gate
        else pd.Series(False, index=out.index)
    )
    extra_reason = pd.Series(pd.NA, index=out.index, dtype="string")
    extra_reason.loc[own_bound_breach] = "own_history_bound_breach"
    extra_reason.loc[joint_breach] = "joint_vector_breach"
    extra_reason.loc[own_bound_breach & joint_breach] = (
        "own_history_bound_breach|joint_vector_breach"
    )
    screened = (own_bound_breach | joint_breach) & ~repair
    repair = repair | screened
    admit = policy & ~repair
    out.loc[admit, "level1_admission_status"] = "valid_complete_year"
    out.loc[admit, "level2_semantic_status"] = "definitionally_complete"
    admit_soft = admit & out["soft_reason_codes"].notna()
    out.loc[admit_soft, "level_admission_reason"] = out.loc[admit_soft, "soft_reason_codes"]
    out.loc[admit, "level_repair_mode"] = "none"
    out.loc[admit, "level_policy_resolution"] = "admit_valid_complete_year"
    out.loc[repair, "level1_admission_status"] = "coverage_defective"
    out.loc[repair, "level2_semantic_status"] = "semantically_unusable"
    repair_soft = repair & out["soft_reason_codes"].notna()
    out.loc[repair_soft, "level_admission_reason"] = out.loc[repair_soft, "soft_reason_codes"]
    out.loc[repair, "level_repair_mode"] = "decayed_own_history_or_pooled"
    out.loc[repair, "level_policy_resolution"] = "repair_ladder"
    out.loc[screened, "level_admission_reason"] = extra_reason[screened]
    out.loc[screened, "level_policy_resolution"] = "repair_screened_vector"
    out.loc[repair, "usable_as_observed"] = False
    out.loc[repair, "current_row_is_true_partial"] = False
    out.loc[repair, "preferred_count"] = 0.0
    out.loc[soft_hit, "level_review_hold"] = False

    unresolved = target & (
        out["level1_admission_status"].eq("unresolved_review")
        | out["level_repair_mode"].eq("unchanged_review_hold")
        | out["level_review_hold"].fillna(False).astype(bool)
    )
    if unresolved.any():
        raise LevelLaneConfigError(
            "target-year level policy left unresolved review rows: "
            + str(
                out.loc[unresolved, ["ori9", "offense", "level1_admission_status", "level_repair_mode"]]
                .head(20)
                .to_dict(orient="records")
            )
        )
    out = out.drop(columns=["_soft_policy_candidate", "_hard_unresolved_candidate"])

    disposition_cols = [
        "ori9", "state_fips", "state_abbr", "year", "offense", "preferred_source",
        "level_lane_original_count", "level_lane_replacement_count", "preferred_count", "level1_admission_status",
        "level2_semantic_status", "level_admission_reason", "level_repair_mode",
        "external_check_status", "level_review_hold", "level_lane_hard_case_id",
        "level_lane_registry_case_id", "level_lane_registry_adjudication",
        "soft_reason_codes", "level_policy_resolution", "level_peer_prior_count_2023",
        "level_peer_comparable", "level_peer_log_change_2023_2024", "level_peer_sample_n",
        "level_peer_log_change_p01", "level_peer_log_change_p99",
        "level_own_clean_years", "level_own_log_change_lower", "level_own_log_change_upper",
        "level_own_log_change_min",
        "level_admission_bound_basis", "level_admission_bound_lower",
        "level_admission_bound_upper", "level_joint_vector_breach_count",
        "level_joint_vector_divergent_count",
        "level_nibrs_start_year",
    ]
    disposition = out.loc[target, [c for c in disposition_cols if c in out.columns]].copy()
    review = disposition[disposition["level_review_hold"].fillna(False).astype(bool)].copy()
    return AdmissionArtifacts(out, disposition, review)


def admission_dependency_paths(paths) -> list[Path]:
    return [
        _config_path(paths, "level_lane_admission_registry.csv"),
        _config_path(paths, "level_lane_hard_evidence.csv"),
        _config_path(paths, "overlap_custom_footprints.csv"),
        _config_path(paths, "service_wide_agency_scopes.csv"),
        covered_by_source_path(paths),
        Path(paths.repo_root) / "src" / "crimerisk" / "contract_coverage.py",
        Path(paths.state_dir) / "reference" / "agency_master.parquet",
        Path(paths.repo_root) / "analysis_scratch" / "level_lane_screen" / "structural_zero_corroborated_all.csv",
        Path(__file__),
    ]
