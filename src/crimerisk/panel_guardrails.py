from __future__ import annotations

import pandas as pd

from crimerisk.source_provenance import SUMMARY_SOURCE


EXTREME_SUMMARY_SPIKE_MIN_JUMP_RATIO = 20.0
EXTREME_SUMMARY_SPIKE_MAX_NEXT_RATIO = 0.1
EXTREME_SUMMARY_SPIKE_MIN_COUNT_TO_POP = 1.0


def suppress_extreme_summary_spikes(
    panel: pd.DataFrame,
    *,
    group_cols: tuple[str, str] = ("jurisdiction_id", "offense"),
    year_col: str = "year",
    count_col: str = "reported_count_preferred",
    source_col: str = "preferred_source",
    regime_col: str = "dominant_reporting_regime",
    usable_col: str = "usable_as_observed",
    population_col: str = "bucket_population",
) -> pd.DataFrame:
    out = panel.copy()
    if out.empty:
        out["extreme_summary_spike_suppressed"] = pd.Series(dtype=bool)
        return out

    out = out.sort_values([*group_cols, year_col], kind="mergesort").reset_index(drop=True)
    counts = pd.to_numeric(out[count_col], errors="coerce").fillna(0.0)
    pop = pd.to_numeric(out.get(population_col), errors="coerce").fillna(0.0)

    grouped = out.groupby(list(group_cols), dropna=False, sort=False)
    prev_count = grouped[count_col].shift(1)
    next_count = grouped[count_col].shift(-1)
    prev_count = pd.to_numeric(prev_count, errors="coerce")
    next_count = pd.to_numeric(next_count, errors="coerce")

    jump_ratio_prev = counts / prev_count.where(prev_count.gt(0))
    drop_ratio_next = next_count / counts.where(counts.gt(0))
    count_to_pop = counts / pop.where(pop.gt(0))

    suppress = (
        out[usable_col].astype("boolean").fillna(False).astype(bool)
        & out[source_col].astype("string").eq(SUMMARY_SOURCE)
        & out[regime_col].astype("string").eq("lumpy_or_batched")
        & jump_ratio_prev.ge(EXTREME_SUMMARY_SPIKE_MIN_JUMP_RATIO).fillna(False)
        & drop_ratio_next.le(EXTREME_SUMMARY_SPIKE_MAX_NEXT_RATIO).fillna(False)
        & count_to_pop.ge(EXTREME_SUMMARY_SPIKE_MIN_COUNT_TO_POP).fillna(False)
    )

    out["extreme_summary_spike_suppressed"] = suppress
    out.loc[suppress, usable_col] = False
    return out
