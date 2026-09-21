from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.crosswalk_shares import normalize_block_group_allocation_shares


def _offense_volume_band(incident_total: float) -> str:
    total = float(pd.to_numeric(incident_total, errors="coerce")) if pd.notna(incident_total) else 0.0
    if total < 1_000:
        return "<1k"
    if total < 5_000:
        return "1k-5k"
    if total < 20_000:
        return "5k-20k"
    return "20k+"


def safe_corr(x: pd.Series, y: pd.Series, *, method: str) -> float | None:
    if len(x) < 2:
        return None
    if float(pd.to_numeric(x, errors="coerce").std(ddof=0)) == 0.0:
        return None
    if float(pd.to_numeric(y, errors="coerce").std(ddof=0)) == 0.0:
        return None
    value = pd.Series(x).corr(pd.Series(y), method=method)
    if pd.isna(value):
        return None
    return float(value)


def top_mass_capture(df: pd.DataFrame, frac: float, *, predicted_share_col: str) -> float | None:
    if df.empty:
        return None
    n = max(1, int(np.ceil(len(df) * float(frac))))
    top = df.nlargest(n, predicted_share_col)
    return float(pd.to_numeric(top["true_share"], errors="coerce").fillna(0.0).sum())


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    if df.empty or value_col not in df.columns or weight_col not in df.columns:
        return None
    values = pd.to_numeric(df[value_col], errors="coerce")
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0)
    if not mask.any():
        return None
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def build_city_share_truth_model_frame(
    *,
    city_shares: pd.DataFrame,
    bg_prior: pd.DataFrame,
    bg_crosswalk: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    city = city_shares.copy()
    city = city[pd.to_numeric(city["year"], errors="coerce").eq(int(year))].copy()
    if city.empty:
        return pd.DataFrame(
            columns=[
                "city_name",
                "jurisdiction_id",
                "state_fips",
                "offense",
                "bg_id",
                "incident_count",
                "incident_total",
                "true_share",
                "model_share",
            ]
        )

    city = city.rename(columns={"block_group_geoid": "bg_id"}).copy()
    city["bg_id"] = city["bg_id"].astype("string").str.zfill(12)
    city["state_fips"] = city["state_fips"].astype("string").str.zfill(2)
    city["geocode_quality_tier"] = city.get("geocode_quality_tier", "unknown").astype("string").fillna("unknown")

    truth_meta_counts = (
        city.groupby(
            ["city_name", "jurisdiction_id", "state_fips", "offense", "geocode_quality_tier"],
            dropna=False,
            as_index=False,
        )
        .agg(incident_count=("incident_count", "sum"))
    )
    truth_meta_counts["incident_total"] = truth_meta_counts.groupby(
        ["city_name", "jurisdiction_id", "state_fips", "offense"],
        dropna=False,
    )["incident_count"].transform("sum")
    truth_meta_counts["incident_share"] = np.where(
        pd.to_numeric(truth_meta_counts["incident_total"], errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(truth_meta_counts["incident_count"], errors="coerce").fillna(0.0)
        / pd.to_numeric(truth_meta_counts["incident_total"], errors="coerce").fillna(1.0),
        0.0,
    )
    truth_meta = (
        truth_meta_counts.sort_values(
            ["city_name", "jurisdiction_id", "state_fips", "offense", "incident_count"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(["city_name", "jurisdiction_id", "state_fips", "offense"])
        .rename(columns={"geocode_quality_tier": "primary_geocode_quality_tier"})
    )[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "offense",
            "primary_geocode_quality_tier",
            "incident_total",
        ]
    ].copy()
    reported_latlon = truth_meta_counts[truth_meta_counts["geocode_quality_tier"].eq("reported_latlon")][
        ["city_name", "jurisdiction_id", "state_fips", "offense", "incident_share"]
    ].rename(columns={"incident_share": "reported_latlon_share"})
    truth_meta = truth_meta.merge(
        reported_latlon,
        on=["city_name", "jurisdiction_id", "state_fips", "offense"],
        how="left",
    )
    truth_meta["reported_latlon_share"] = pd.to_numeric(truth_meta["reported_latlon_share"], errors="coerce").fillna(0.0)
    truth_meta["fallback_incident_share"] = 1.0 - truth_meta["reported_latlon_share"]
    truth_meta["offense_volume_band"] = truth_meta["incident_total"].map(_offense_volume_band)

    truth = (
        city.groupby(["city_name", "jurisdiction_id", "state_fips", "offense", "bg_id"], dropna=False, as_index=False)
        .agg(incident_count=("incident_count", "sum"))
    )
    truth["incident_total"] = truth.groupby(
        ["city_name", "jurisdiction_id", "state_fips", "offense"], dropna=False
    )["incident_count"].transform("sum")
    truth = truth[truth["incident_total"].gt(0)].copy()
    truth["true_share"] = pd.to_numeric(truth["incident_count"], errors="coerce").fillna(0.0) / pd.to_numeric(
        truth["incident_total"], errors="coerce"
    ).fillna(np.nan)

    prior = bg_prior.copy()
    prior["bg_id"] = prior["bg_id"].astype("string").str.zfill(12)
    prior["state_fips"] = prior["state_fips"].astype("string").str.zfill(2)
    prior["bg_weight"] = pd.to_numeric(prior["bg_weight"], errors="coerce").fillna(0.0)

    normalized_crosswalk = normalize_block_group_allocation_shares(bg_crosswalk.copy())
    model = prior.merge(
        normalized_crosswalk.rename(columns={"block_group_geoid": "bg_id"})[
            ["state_fips", "bg_id", "jurisdiction_id", "allocation_share"]
        ],
        on=["state_fips", "bg_id"],
        how="inner",
    )
    city_keys = truth[["city_name", "jurisdiction_id", "state_fips", "offense"]].drop_duplicates()
    model = model.merge(city_keys, on=["jurisdiction_id", "state_fips", "offense"], how="inner")
    model["model_component_weight"] = pd.to_numeric(model["bg_weight"], errors="coerce").fillna(0.0) * pd.to_numeric(
        model["allocation_share"], errors="coerce"
    ).fillna(0.0)
    model["model_total"] = model.groupby(
        ["city_name", "jurisdiction_id", "state_fips", "offense"], dropna=False
    )["model_component_weight"].transform("sum")
    model = model[model["model_total"].gt(0)].copy()
    model["model_share"] = model["model_component_weight"] / model["model_total"]
    model = model[
        ["city_name", "jurisdiction_id", "state_fips", "offense", "bg_id", "model_share"]
    ].drop_duplicates()

    merged = truth.merge(
        model,
        on=["city_name", "jurisdiction_id", "state_fips", "offense", "bg_id"],
        how="outer",
    )
    merged["incident_count"] = pd.to_numeric(merged["incident_count"], errors="coerce").fillna(0.0)
    merged["incident_total"] = pd.to_numeric(merged["incident_total"], errors="coerce")
    merged["true_share"] = pd.to_numeric(merged["true_share"], errors="coerce").fillna(0.0)
    merged["model_share"] = pd.to_numeric(merged["model_share"], errors="coerce").fillna(0.0)
    merged = merged.merge(
        truth_meta[
            [
                "city_name",
                "jurisdiction_id",
                "state_fips",
                "offense",
                "primary_geocode_quality_tier",
                "fallback_incident_share",
                "offense_volume_band",
            ]
        ],
        on=["city_name", "jurisdiction_id", "state_fips", "offense"],
        how="left",
    )
    return merged


def build_city_share_diagnostics(
    merged: pd.DataFrame,
    *,
    predicted_share_col: str = "model_share",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if merged.empty:
        empty = pd.DataFrame(
            columns=[
                "city_name",
                "jurisdiction_id",
                "state_fips",
                "offense",
                "n_bg_union",
                "n_bg_truth_nonzero",
                "n_bg_model_nonzero",
                "incident_total",
                "total_variation_distance",
                "share_rmse",
                "pearson_share",
                "spearman_share",
                "top_10pct_true_mass_in_model_top_10pct",
                "primary_geocode_quality_tier",
                "fallback_incident_share",
                "offense_volume_band",
            ]
        )
        return empty, {"rows": 0, "predicted_share_col": predicted_share_col}

    frame = merged.copy()
    frame["true_share"] = pd.to_numeric(frame["true_share"], errors="coerce").fillna(0.0)
    frame[predicted_share_col] = pd.to_numeric(frame[predicted_share_col], errors="coerce").fillna(0.0)

    rows: list[dict[str, object]] = []
    for keys, grp in frame.groupby(["city_name", "jurisdiction_id", "state_fips", "offense"], dropna=False):
        city_name, jurisdiction_id, state_fips, offense = keys
        if grp.empty:
            continue
        diff = pd.to_numeric(grp[predicted_share_col], errors="coerce").fillna(0.0) - pd.to_numeric(
            grp["true_share"], errors="coerce"
        ).fillna(0.0)
        rows.append(
            {
                "city_name": str(city_name),
                "jurisdiction_id": str(jurisdiction_id),
                "state_fips": str(state_fips),
                "offense": str(offense),
                "n_bg_union": int(len(grp)),
                "n_bg_truth_nonzero": int(grp["true_share"].gt(0).sum()),
                "n_bg_model_nonzero": int(pd.to_numeric(grp[predicted_share_col], errors="coerce").fillna(0.0).gt(0).sum()),
                "incident_total": float(pd.to_numeric(grp["incident_count"], errors="coerce").fillna(0.0).sum()),
                "total_variation_distance": float(0.5 * np.abs(diff).sum()),
                "share_rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "pearson_share": safe_corr(grp["true_share"], grp[predicted_share_col], method="pearson"),
                "spearman_share": safe_corr(grp["true_share"], grp[predicted_share_col], method="spearman"),
                "top_10pct_true_mass_in_model_top_10pct": top_mass_capture(
                    grp[["true_share", predicted_share_col]].copy(),
                    0.10,
                    predicted_share_col=predicted_share_col,
                ),
                "primary_geocode_quality_tier": str(grp.get("primary_geocode_quality_tier", pd.Series(["unknown"])).iloc[0]),
                "fallback_incident_share": float(pd.to_numeric(grp.get("fallback_incident_share", pd.Series([0.0])), errors="coerce").fillna(0.0).iloc[0]),
                "offense_volume_band": str(grp.get("offense_volume_band", pd.Series(["unknown"])).iloc[0]),
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values(
        ["city_name", "offense"],
        kind="mergesort",
    ).reset_index(drop=True)
    summary = {
        "rows": int(len(diagnostics)),
        "predicted_share_col": predicted_share_col,
        "city_count": int(diagnostics["city_name"].nunique()) if not diagnostics.empty else 0,
        "incident_total": float(pd.to_numeric(diagnostics.get("incident_total"), errors="coerce").fillna(0.0).sum())
        if not diagnostics.empty
        else 0.0,
        "weighted_total_variation_distance_mean": weighted_mean(
            diagnostics, "total_variation_distance", "incident_total"
        ),
        "weighted_share_rmse_mean": weighted_mean(diagnostics, "share_rmse", "incident_total"),
        "weighted_pearson_share_mean": weighted_mean(diagnostics, "pearson_share", "incident_total"),
        "weighted_spearman_share_mean": weighted_mean(diagnostics, "spearman_share", "incident_total"),
        "weighted_top_10pct_true_mass_in_model_top_10pct_mean": weighted_mean(
            diagnostics,
            "top_10pct_true_mass_in_model_top_10pct",
            "incident_total",
        ),
        "by_city": [],
        "by_offense": [],
        "by_primary_geocode_quality_tier": [],
        "by_offense_volume_band": [],
    }
    if not diagnostics.empty:
        for city_name, grp in diagnostics.groupby("city_name", sort=True):
            summary["by_city"].append(
                {
                    "city_name": str(city_name),
                    "rows": int(len(grp)),
                    "incident_total": float(pd.to_numeric(grp["incident_total"], errors="coerce").fillna(0.0).sum()),
                    "weighted_total_variation_distance_mean": weighted_mean(
                        grp, "total_variation_distance", "incident_total"
                    ),
                    "weighted_pearson_share_mean": weighted_mean(grp, "pearson_share", "incident_total"),
                    "weighted_spearman_share_mean": weighted_mean(grp, "spearman_share", "incident_total"),
                }
            )
        for offense, grp in diagnostics.groupby("offense", sort=True):
            summary["by_offense"].append(
                {
                    "offense": str(offense),
                    "rows": int(len(grp)),
                    "incident_total": float(pd.to_numeric(grp["incident_total"], errors="coerce").fillna(0.0).sum()),
                    "weighted_total_variation_distance_mean": weighted_mean(
                        grp, "total_variation_distance", "incident_total"
                    ),
                    "weighted_pearson_share_mean": weighted_mean(grp, "pearson_share", "incident_total"),
                    "weighted_spearman_share_mean": weighted_mean(grp, "spearman_share", "incident_total"),
                }
            )
        for geocode_tier, grp in diagnostics.groupby("primary_geocode_quality_tier", sort=True):
            summary["by_primary_geocode_quality_tier"].append(
                {
                    "primary_geocode_quality_tier": str(geocode_tier),
                    "rows": int(len(grp)),
                    "incident_total": float(pd.to_numeric(grp["incident_total"], errors="coerce").fillna(0.0).sum()),
                    "weighted_total_variation_distance_mean": weighted_mean(grp, "total_variation_distance", "incident_total"),
                    "weighted_pearson_share_mean": weighted_mean(grp, "pearson_share", "incident_total"),
                    "weighted_spearman_share_mean": weighted_mean(grp, "spearman_share", "incident_total"),
                    "weighted_fallback_incident_share_mean": weighted_mean(grp, "fallback_incident_share", "incident_total"),
                }
            )
        for volume_band, grp in diagnostics.groupby("offense_volume_band", sort=True):
            summary["by_offense_volume_band"].append(
                {
                    "offense_volume_band": str(volume_band),
                    "rows": int(len(grp)),
                    "incident_total": float(pd.to_numeric(grp["incident_total"], errors="coerce").fillna(0.0).sum()),
                    "weighted_total_variation_distance_mean": weighted_mean(grp, "total_variation_distance", "incident_total"),
                    "weighted_pearson_share_mean": weighted_mean(grp, "pearson_share", "incident_total"),
                    "weighted_spearman_share_mean": weighted_mean(grp, "spearman_share", "incident_total"),
                    "weighted_fallback_incident_share_mean": weighted_mean(grp, "fallback_incident_share", "incident_total"),
                }
            )
    return diagnostics, summary
