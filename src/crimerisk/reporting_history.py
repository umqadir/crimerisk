"""One adjudicated reporting history: agency x offense x year, admitted or not, with a reason.

Every consumer of past filings (the level repair's own-history references, the agency risk
signals, the jurisdiction-year panel the smoother reads, the level holdout) used to take the
history years as published, while only the target year passed the level-lane screens. A year
the lane would refuse as a target year (Savannah 2024: rape 25, aggravated assault 1, larceny 0
on a twelve-month CIUS row) was then averaged into the smoothed control as a clean zero.

Here each year Y from the panel start up to the target year runs through the same admission
sequence the target year does, as if Y were the target year and seeing only years up to Y: the
level-lane admission (hard evidence and reporting-regime restrictions, structural zeros,
covered-by identity, near-zero vectors, the soft screens and the own-history bound), the
masked-gap detector, and the stage-1 adjudicated usability directives. Years are adjudicated in
order, so the screens for Y read an already adjudicated history. A refused year is missing, not
zero: its count is set to zero with `usable_as_observed` False, and the filed value stays in
`history_original_count`. A year with a documented source replacement is admitted at the
replacement count.

The target year itself is left to the existing target-year path, which applies the identical
sequence on this adjudicated history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_ADJUDICATED_COLUMN = "history_adjudicated"

ADMITTED_COMPLETE = "admitted_complete"
ADMITTED_REPLACEMENT = "admitted_replacement"
ADMITTED_PARTIAL = "admitted_partial_lower_bound"
REJECTED = "rejected"
NOT_REPORTED = "not_reported"
HISTORY_STATUSES = frozenset(
    {ADMITTED_COMPLETE, ADMITTED_REPLACEMENT, ADMITTED_PARTIAL, REJECTED, NOT_REPORTED}
)

REPLACEMENT_REPAIR_MODES = frozenset({"source_supported_replacement", "swap_reclassification"})

HISTORY_COLUMNS = [
    "ori9",
    "state_fips",
    "offense",
    "year",
    "history_status",
    "history_reason",
    "history_original_count",
    "history_admitted_count",
    "history_months_reported",
    "preferred_source",
    "level1_admission_status",
    "level2_semantic_status",
    "level_repair_mode",
    "level_lane_hard_case_id",
]

def complete_admitted_filing(panel: pd.DataFrame) -> pd.Series:
    """Agency rows that are a complete, admitted twelve-month filing at full weight."""
    usable = (
        panel["usable_as_observed"].fillna(False).astype(bool)
        if "usable_as_observed" in panel.columns
        else pd.Series(True, index=panel.index)
    )
    months = pd.to_numeric(panel["preferred_months_reported"], errors="coerce")
    weight = pd.to_numeric(panel["preferred_observation_weight"], errors="coerce")
    return (usable & months.ge(12.0) & weight.ge(1.0)).fillna(False).astype(bool)


def reporting_history_path(paths, *, year: int) -> Path:
    return Path(paths.state_dir) / "controls" / f"reporting_history_{int(year)}.parquet"


def _classify(rows: pd.DataFrame) -> pd.DataFrame:
    """History status of one adjudicated year's rows."""
    usable = rows["usable_as_observed"].fillna(False).astype(bool)
    partial = rows["current_row_is_true_partial"].fillna(False).astype(bool)
    mode = rows["level_repair_mode"].astype("string")
    replacement = pd.to_numeric(rows["level_lane_replacement_count"], errors="coerce")
    has_replacement = mode.isin(REPLACEMENT_REPAIR_MODES).fillna(False) & replacement.notna()
    original = pd.to_numeric(rows["level_lane_original_count"], errors="coerce")
    months = pd.to_numeric(rows["preferred_months_reported"], errors="coerce").fillna(0.0)
    status = pd.Series(REJECTED, index=rows.index, dtype="string")
    status[usable] = ADMITTED_COMPLETE
    status[~usable & partial] = ADMITTED_PARTIAL
    status[has_replacement] = ADMITTED_REPLACEMENT
    silent = ~usable & ~partial & ~has_replacement & original.fillna(0.0).le(0.0) & months.le(0.0)
    status[silent] = NOT_REPORTED
    reason = rows["level_admission_reason"].astype("string")
    return pd.DataFrame(
        {
            "history_status": status,
            "history_reason": reason,
            "history_admitted_count": np.where(
                has_replacement,
                replacement,
                np.where(usable | partial, pd.to_numeric(rows["preferred_count"], errors="coerce"), np.nan),
            ),
        },
        index=rows.index,
    )


def adjudicate_history_years(
    panel: pd.DataFrame,
    *,
    paths,
    target_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(panel with every year < target_year adjudicated, history table for every year < target).

    `panel` is the raw preferred-observation panel (`build_agency_trend_fill_panel` before this
    step). Rows of the target year are returned unchanged.
    """
    # Local imports: trend_fills builds the panel and calls this module.
    from crimerisk.level_lane import apply_level_lane_admission
    from crimerisk.stage1_adjudications import build_usability_directives
    from crimerisk.trend_fills import (
        _load_masked_gap_observations,
        apply_masked_gap_reclassification,
        apply_stage1_adjudicated_usability,
        build_masked_gap_flags,
    )

    if panel.empty:
        out = panel.copy()
        out[HISTORY_ADJUDICATED_COLUMN] = pd.Series(dtype=bool)
        return out, pd.DataFrame(columns=HISTORY_COLUMNS)

    work = panel.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["ori9"] = work["ori9"].astype("string").str.upper()
    work["history_original_count"] = pd.to_numeric(work["preferred_count"], errors="coerce")
    years = sorted(int(y) for y in work["year"].dropna().unique() if int(y) < int(target_year))
    obs = _load_masked_gap_observations(paths)
    adjudicated_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    raw_columns = list(work.columns)
    for year in years:
        visible = work[work["year"].le(year)]
        admitted = apply_level_lane_admission(
            visible.drop(columns=["history_original_count"]),
            paths=paths,
            target_year=year,
            history_mode=True,
        ).panel
        flags = build_masked_gap_flags(paths, target_year=year, agency_panel=admitted, obs=obs)
        admitted = apply_masked_gap_reclassification(
            admitted, target_year=year, masked_gap_flags=flags
        )
        admitted, _counts = apply_stage1_adjudicated_usability(
            admitted,
            target_year=year,
            directives=build_usability_directives(paths, target_year=year),
        )
        rows = admitted[admitted["year"].eq(year)].copy()
        classified = _classify(rows)
        rows = rows.join(classified)
        # The adjudicated year as every later screen and consumer reads it.
        admitted_now = rows["history_status"].isin({ADMITTED_COMPLETE, ADMITTED_REPLACEMENT})
        rows["preferred_count"] = np.where(
            rows["history_status"].eq(ADMITTED_REPLACEMENT),
            rows["history_admitted_count"],
            rows["preferred_count"],
        )
        rows["usable_as_observed"] = admitted_now.to_numpy()
        rows["current_row_is_true_partial"] = rows["history_status"].eq(ADMITTED_PARTIAL).to_numpy()
        replaced_months = rows["history_status"].eq(ADMITTED_REPLACEMENT)
        rows.loc[replaced_months, "preferred_months_reported"] = 12.0
        if "preferred_observation_weight" in rows.columns:
            rows.loc[replaced_months, "preferred_observation_weight"] = 1.0
        original = work.loc[work["year"].eq(year), ["ori9", "offense", "history_original_count"]]
        rows = rows.drop(columns=["history_original_count"], errors="ignore").merge(
            original, on=["ori9", "offense"], how="left", validate="one_to_one"
        )
        history_parts.append(
            rows.reindex(columns=HISTORY_COLUMNS).assign(
                history_months_reported=pd.to_numeric(rows["preferred_months_reported"], errors="coerce")
            )
        )
        keep = [c for c in raw_columns if c in rows.columns and c != "history_original_count"]
        rows_raw = rows[keep].copy()
        # Write the adjudicated year back so the next year's screens read it.
        mask = work["year"].eq(year)
        work = pd.concat(
            [work.loc[~mask], rows_raw.assign(history_original_count=rows["history_original_count"].to_numpy())],
            ignore_index=True,
        )
    history = (
        pd.concat(history_parts, ignore_index=True)
        if history_parts
        else pd.DataFrame(columns=HISTORY_COLUMNS)
    )
    work[HISTORY_ADJUDICATED_COLUMN] = work["year"].lt(int(target_year)).fillna(False).astype(bool)
    work = work.drop(columns=["history_original_count"])
    work = work.sort_values(["ori9", "offense", "year"], kind="mergesort").reset_index(drop=True)
    history["year"] = pd.to_numeric(history["year"], errors="coerce").astype(int)
    return work, history.sort_values(["ori9", "offense", "year"], kind="mergesort").reset_index(drop=True)


def write_reporting_history(history: pd.DataFrame, target_rows: pd.DataFrame, *, paths, year: int) -> Path:
    """Persist the history table: the adjudicated past years plus the target year's disposition."""
    target = target_rows.copy()
    if not target.empty:
        classified = _classify(target)
        target = target.join(classified)
        target["history_original_count"] = pd.to_numeric(
            target.get("level_lane_original_count"), errors="coerce"
        )
        target["history_months_reported"] = pd.to_numeric(
            target["preferred_months_reported"], errors="coerce"
        )
    frame = pd.concat([history, target.reindex(columns=HISTORY_COLUMNS)], ignore_index=True)
    path = reporting_history_path(paths, year=int(year))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
