"""M4: the feed-city posterior's prior strength learned by rolling origin (not adopted).

Candidate 2025.2 M4 learned, per feed city and offense, the pseudo-incident prior strength of
the feed-city posterior from the city's own feed years, in place of the feed-quality rule.

Evidence (docs/NOTES.md, "2025.2 candidate results"; gold_m4__rolling_origin_strength at
349a152, temporal folds, paired against gold_b0, city-mean TVD, primary/all):

| Offense, fold | Delta TVD | 95% interval |
|---|---|---|
| Larceny, temporal 2021 | -0.01023 | [-0.02723, +0.00025] |
| Larceny, temporal 2023 | -0.01038 | [-0.03233, +0.00165] |
| MVT, temporal 2021 | -0.00900 | [-0.02227, +0.00026] |
| MVT, temporal 2023 | -0.01020 | [-0.02390, -0.00101] |

No offense won both temporal horizons (the pre-registered rule), so M4 was not adopted. The
build hook (`AllocationBuildConfig.rolling_origin_posterior_offenses`, `build-outputs
--rolling-origin-posterior-offenses`) was removed from the release configuration on branch
f2-build-fixes; it replaced the posterior strength `alpha` of active feed-city rows with
max(learned alpha, alpha floor) for the named offenses. Re-wiring it for a new experiment is a
change to `crimerisk.allocation._build_jurisdiction_component_allocations`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.allocation import CITY_POSTERIOR_GROUP_COLS, ROLLING_ORIGIN_POSTERIOR_STRENGTH_GRID


ROLLING_ORIGIN_POSTERIOR_OFFENSES = (
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
    "rape",
    "robbery",
)


def rolling_origin_posterior_strength(
    merged: pd.DataFrame,
    *,
    year_counts: pd.DataFrame,
    offenses: tuple[str, ...],
) -> pd.DataFrame:
    """Learn each feed city's posterior prior strength from its own feed years.

    For every origin year t after the first reliable feed year, the city's incidents in years
    before t are the history h and year t's incidents are the target. The posterior share
    (h + a * prior) / (sum(h) + a) is scored against the target year's share by TVD, and the
    prior strength a (in pseudo-incidents) with the lowest mean TVD over the origins is kept.
    Only years the build already admits are read (`year_counts` stops at the feed truncation),
    and only the block groups of the jurisdiction's own allocation rows are scored, as in the
    posterior itself. A city-offense with fewer than one scoreable origin gets no row.
    """
    columns = [*CITY_POSTERIOR_GROUP_COLS, "rolling_origin_alpha", "rolling_origin_origins", "rolling_origin_mean_tvd"]
    if merged.empty or year_counts.empty or not offenses:
        return pd.DataFrame(columns=columns)
    frame = merged.loc[
        merged["offense"].astype("string").isin(list(offenses))
        & merged["city_incident_posterior_active"].eq(True),
        [*CITY_POSTERIOR_GROUP_COLS, "bg_id", "city_posterior_model_prior_share"],
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["city_posterior_model_prior_share"] = pd.to_numeric(
        frame["city_posterior_model_prior_share"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    prior = frame.groupby([*CITY_POSTERIOR_GROUP_COLS, "bg_id"], dropna=False, as_index=False)[
        "city_posterior_model_prior_share"
    ].sum()
    counts = year_counts.rename(columns={"block_group_geoid": "bg_id"}).copy()
    counts["state_fips"] = counts["state_fips"].astype("string").str.zfill(2)
    counts["jurisdiction_id"] = counts["jurisdiction_id"].astype("string")
    counts["offense"] = counts["offense"].astype("string")
    counts["bg_id"] = counts["bg_id"].astype("string").str.zfill(12)
    counts["year"] = pd.to_numeric(counts["year"], errors="raise").astype(int)
    counts["incident_count"] = pd.to_numeric(counts["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    counts = counts.groupby([*CITY_POSTERIOR_GROUP_COLS, "bg_id", "year"], dropna=False, as_index=False)[
        "incident_count"
    ].sum()
    grid = np.asarray(ROLLING_ORIGIN_POSTERIOR_STRENGTH_GRID, dtype=float)
    rows: list[dict[str, object]] = []
    counts_by_group = {key: group for key, group in counts.groupby(CITY_POSTERIOR_GROUP_COLS, dropna=False)}
    for key, group_prior in prior.groupby(CITY_POSTERIOR_GROUP_COLS, dropna=False):
        group_counts = counts_by_group.get(key)
        if group_counts is None:
            continue
        bg_ids = group_prior["bg_id"].astype(str).to_numpy()
        prior_share = group_prior["city_posterior_model_prior_share"].to_numpy(dtype=float)
        if prior_share.sum() <= 0.0:
            continue
        prior_share = prior_share / prior_share.sum()
        table = (
            group_counts[group_counts["bg_id"].astype(str).isin(set(bg_ids))]
            .pivot_table(index="bg_id", columns="year", values="incident_count", aggfunc="sum", fill_value=0.0)
            .reindex(index=bg_ids, fill_value=0.0)
        )
        years = sorted(int(value) for value in table.columns)
        losses: list[np.ndarray] = []
        for position, origin in enumerate(years[1:], start=1):
            history = table[years[:position]].to_numpy(dtype=float).sum(axis=1)
            target = table[origin].to_numpy(dtype=float)
            if history.sum() <= 0.0 or target.sum() <= 0.0:
                continue
            target_share = target / target.sum()
            posterior = (history[:, None] + grid[None, :] * prior_share[:, None]) / (history.sum() + grid[None, :])
            losses.append(0.5 * np.abs(posterior - target_share[:, None]).sum(axis=0))
        if not losses:
            continue
        mean_loss = np.mean(losses, axis=0)
        best = int(np.argmin(mean_loss))
        rows.append(
            {
                **dict(zip(CITY_POSTERIOR_GROUP_COLS, key)),
                "rolling_origin_alpha": float(grid[best]),
                "rolling_origin_origins": len(losses),
                "rolling_origin_mean_tvd": float(mean_loss[best]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


