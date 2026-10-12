"""Optional EB shrinkage audit of a published surface (not part of the release build).

The build publishes count-derived rates and indices. This audit recomputes the empirical-Bayes
diagnostics the build used to carry as `diagnostic_eb_*` / `diagnostic_resident_eb_*` columns:
a nested state -> jurisdiction -> tract prior rate with leave-one-out totals and strength
k = alpha / national rate. Every input is on the published surface: `expected_count_<offense>`,
`primary_denominator_<offense>`, `resident_secondary_denominator`, `primary_alpha_<offense>`,
`eb_hard_min_denominator`, `tract_id`, `state_fips` and the EB jurisdiction column
(`eb_jurisdiction_id` on block groups, `dominant_eb_jurisdiction_id` on tracts).

    uv run python scripts/diagnostics/eb_shrinkage_audit.py --surface <candidate parquet> --out <parquet>
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7

RATE_PER_100K = 100_000.0
DIAGNOSTIC_FIELDS = (
    "rate",
    "national_rate_per_100k",
    "prior_rate",
    "k",
    "observed_weight",
    "prior_weight",
    "low_denominator_flag",
    "heavy_shrinkage_flag",
    "extreme_shrinkage_flag",
)


def _raw_denominator(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


def empirical_bayes_index(
    out: pd.DataFrame,
    *,
    offense: str,
    counts: pd.Series,
    denominator: pd.Series,
    geo_id_col: str,
    alpha: float,
    hard_min: float,
    jurisdiction_col: str | None = None,
) -> dict[str, object]:
    denom = _raw_denominator(denominator)
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    publishable = denom.gt(float(hard_min))
    valid_denom_sum = float(denom.loc[publishable].sum())
    valid_count_sum = float(count.loc[publishable].sum())
    national_rate = valid_count_sum / valid_denom_sum if valid_denom_sum > 0 else 0.0
    k = float(alpha) / national_rate if national_rate > 0 else float("inf")

    raw_rate = pd.Series(np.nan, index=out.index, dtype=float)
    raw_rate.loc[publishable] = RATE_PER_100K * count.loc[publishable] / denom.loc[publishable]
    diagnostic_eb_prior_rate = pd.Series(np.nan, index=out.index, dtype=float)
    diagnostic_eb_rate = pd.Series(np.nan, index=out.index, dtype=float)
    diagnostic_eb_index = pd.Series(np.nan, index=out.index, dtype=float)
    diagnostic_eb_observed_weight = pd.Series(np.nan, index=out.index, dtype=float)
    diagnostic_eb_prior_weight = pd.Series(np.nan, index=out.index, dtype=float)

    state = out["state_fips"].astype("string").str.zfill(2)
    jurisdiction: pd.Series | None = None
    if jurisdiction_col is not None:
        if jurisdiction_col not in out.columns:
            raise KeyError(f"missing required EB jurisdiction column {jurisdiction_col!r}")
        jurisdiction = out[jurisdiction_col].astype("string")
        missing_jurisdiction = jurisdiction.isna() | jurisdiction.str.strip().eq("")
        if bool(missing_jurisdiction.any()):
            sample_geoids = (
                out.loc[missing_jurisdiction, geo_id_col].astype("string").head(10).fillna("<NA>").tolist()
                if geo_id_col in out.columns
                else []
            )
            raise ValueError(
                f"{jurisdiction_col} is required for EB diagnostics; "
                f"found {int(missing_jurisdiction.sum())} missing row(s). sample_{geo_id_col}={sample_geoids}"
            )

    if not np.isfinite(k) or k <= 0 or not publishable.any():
        denominator_reason = pd.Series("zero_or_structural_denominator", index=out.index, dtype="string")
        denominator_reason.loc[publishable] = "invalid_national_rate"
        return {
            "denominator_raw": denom,
            "raw_rate": raw_rate,
            "diagnostic_eb_rate": diagnostic_eb_rate,
            "diagnostic_eb_index": diagnostic_eb_index,
            "national_rate_per_100k": national_rate * RATE_PER_100K,
            "diagnostic_eb_national_rate_per_100k": np.nan,
            "diagnostic_eb_prior_rate": diagnostic_eb_prior_rate,
            "diagnostic_eb_k": k,
            "diagnostic_eb_observed_weight": diagnostic_eb_observed_weight,
            "diagnostic_eb_prior_weight": diagnostic_eb_prior_weight,
            "index_publishable": publishable & False,
            "diagnostic_eb_low_denominator_flag": publishable & False,
            "diagnostic_eb_heavy_shrinkage_flag": publishable & False,
            "diagnostic_eb_extreme_shrinkage_flag": publishable & False,
            "denominator_reason": denominator_reason,
        }

    support_count = count.where(publishable, 0.0)
    support_denom = denom.where(publishable, 0.0)
    eb_frame = pd.DataFrame(
        {
            "_state": state,
            "_tract": out["tract_id"].astype("string") if "tract_id" in out.columns else out[geo_id_col].astype("string"),
            "_count": support_count,
            "_denom": support_denom,
        },
        index=out.index,
    )
    if jurisdiction is not None:
        eb_frame["_jurisdiction"] = jurisdiction
    else:
        eb_frame["_jurisdiction"] = state

    state_totals = eb_frame.groupby("_state", dropna=False)[["_count", "_denom"]].sum().reset_index()
    state_rates = eb_frame[["_state"]].merge(state_totals, on="_state", how="left")
    state_count = pd.Series(pd.to_numeric(state_rates["_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    state_denom = pd.Series(pd.to_numeric(state_rates["_denom"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    state_count_excl = (state_count - support_count).clip(lower=0.0)
    state_denom_excl = (state_denom - support_denom).clip(lower=0.0)
    state_rate_raw = pd.Series(
        np.where(
            state_denom_excl.gt(float(hard_min)),
            state_count_excl / state_denom_excl.replace(0.0, np.nan),
            national_rate,
        ),
        index=out.index,
        dtype=float,
    ).replace([np.inf, -np.inf], np.nan).fillna(national_rate)

    jur_totals = eb_frame.groupby(["_state", "_jurisdiction"], dropna=False)[["_count", "_denom"]].sum().reset_index()
    jur_rates = eb_frame[["_state", "_jurisdiction"]].merge(
        jur_totals,
        on=["_state", "_jurisdiction"],
        how="left",
    )
    jur_count = pd.Series(pd.to_numeric(jur_rates["_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    jur_denom = pd.Series(pd.to_numeric(jur_rates["_denom"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    jur_count_excl = (jur_count - support_count).clip(lower=0.0)
    jur_denom_excl = (jur_denom - support_denom).clip(lower=0.0)
    r_jur = pd.Series(((jur_count_excl + k * state_rate_raw) / (jur_denom_excl + k)).to_numpy(dtype=float), index=out.index)

    tract_totals = eb_frame.groupby(["_state", "_tract"], dropna=False)[["_count", "_denom"]].sum().reset_index()
    tract_rates = eb_frame[["_state", "_tract"]].merge(tract_totals, on=["_state", "_tract"], how="left")
    tract_count = pd.Series(pd.to_numeric(tract_rates["_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    tract_denom = pd.Series(pd.to_numeric(tract_rates["_denom"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=out.index)
    tract_count_excl = (tract_count - support_count).clip(lower=0.0)
    tract_denom_excl = (tract_denom - support_denom).clip(lower=0.0)
    tract_prior_raw = (tract_count_excl + k * r_jur) / (tract_denom_excl + k)
    use_tract = tract_denom_excl.gt(float(hard_min)) & np.isfinite(tract_prior_raw)
    prior_rate_raw = pd.Series(np.where(use_tract, tract_prior_raw, r_jur), index=out.index, dtype=float)

    diagnostic_eb_raw_rate = (count + k * prior_rate_raw) / (denom + k)
    diagnostic_eb_raw_rate = diagnostic_eb_raw_rate.where(publishable, np.nan)
    diagnostic_eb_national_rate = float((denom.loc[publishable] * diagnostic_eb_raw_rate.loc[publishable]).sum() / denom.loc[publishable].sum())
    diagnostic_eb_index.loc[publishable] = 100.0 * diagnostic_eb_raw_rate.loc[publishable] / diagnostic_eb_national_rate
    diagnostic_eb_rate.loc[publishable] = RATE_PER_100K * diagnostic_eb_raw_rate.loc[publishable]
    diagnostic_eb_prior_rate.loc[publishable] = RATE_PER_100K * prior_rate_raw.loc[publishable]
    diagnostic_eb_observed_weight.loc[publishable] = denom.loc[publishable] / (denom.loc[publishable] + k)
    diagnostic_eb_prior_weight.loc[publishable] = k / (denom.loc[publishable] + k)
    diagnostic_eb_low_denominator = publishable & diagnostic_eb_observed_weight.lt(0.5)
    diagnostic_eb_heavy_shrinkage = publishable & diagnostic_eb_observed_weight.lt(0.20)
    diagnostic_eb_extreme_shrinkage = publishable & diagnostic_eb_observed_weight.lt(0.05)
    denominator_reason = pd.Series("zero_or_structural_denominator", index=out.index, dtype="string")
    denominator_reason.loc[publishable] = "publishable"

    return {
        "denominator_raw": denom,
        "raw_rate": raw_rate.replace([np.inf, -np.inf], np.nan),
        "diagnostic_eb_rate": diagnostic_eb_rate.replace([np.inf, -np.inf], np.nan),
        "diagnostic_eb_index": diagnostic_eb_index.replace([np.inf, -np.inf], np.nan),
        "national_rate_per_100k": national_rate * RATE_PER_100K,
        "diagnostic_eb_national_rate_per_100k": diagnostic_eb_national_rate * RATE_PER_100K,
        "diagnostic_eb_prior_rate": diagnostic_eb_prior_rate.replace([np.inf, -np.inf], np.nan),
        "diagnostic_eb_k": k,
        "diagnostic_eb_observed_weight": diagnostic_eb_observed_weight,
        "diagnostic_eb_prior_weight": diagnostic_eb_prior_weight,
        "index_publishable": publishable,
        "diagnostic_eb_low_denominator_flag": diagnostic_eb_low_denominator,
        "diagnostic_eb_heavy_shrinkage_flag": diagnostic_eb_heavy_shrinkage,
        "diagnostic_eb_extreme_shrinkage_flag": diagnostic_eb_extreme_shrinkage,
        "denominator_reason": denominator_reason,
    }


def eb_shrinkage_audit(
    surface: pd.DataFrame,
    *,
    geo_id_col: str,
    jurisdiction_col: str | None = None,
) -> pd.DataFrame:
    """One row per surface row: the primary and resident EB diagnostics for every offense."""
    if jurisdiction_col is None:
        jurisdiction_col = "eb_jurisdiction_id" if geo_id_col == "block_group_geoid" else "dominant_eb_jurisdiction_id"
    hard_min = float(pd.to_numeric(surface["eb_hard_min_denominator"], errors="coerce").dropna().iloc[0])
    columns: dict[str, object] = {"state_fips": surface["state_fips"], geo_id_col: surface[geo_id_col]}
    for offense in OFFENSES_7:
        alpha = float(pd.to_numeric(surface[f"primary_alpha_{offense}"], errors="coerce").dropna().iloc[0])
        counts = surface[f"expected_count_{offense}"]
        for prefix, denominator in (
            ("diagnostic_eb", surface[f"primary_denominator_{offense}"]),
            ("diagnostic_resident_eb", surface["resident_secondary_denominator"]),
        ):
            result = empirical_bayes_index(
                surface,
                offense=offense,
                counts=counts,
                denominator=denominator,
                geo_id_col=geo_id_col,
                alpha=alpha,
                hard_min=hard_min,
                jurisdiction_col=jurisdiction_col,
            )
            for field in DIAGNOSTIC_FIELDS:
                columns[f"{prefix}_{field}_{offense}"] = result[f"diagnostic_eb_{field}"]
    frame = pd.concat(
        {name: pd.Series(values, index=surface.index) for name, values in columns.items()}, axis=1
    )
    return frame.reset_index(drop=True)
