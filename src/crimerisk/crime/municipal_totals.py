from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7


def _project_count_log_linear(*, years: np.ndarray, counts: np.ndarray, target_year: int) -> float:
    years = np.asarray(years, dtype=float)
    counts = np.asarray(counts, dtype=float)
    mask = np.isfinite(years) & np.isfinite(counts) & (counts >= 0)
    years = years[mask]
    counts = counts[mask]
    if len(years) == 0:
        return float("nan")
    if len(years) == 1:
        return float(counts[0])
    y = np.log1p(counts)
    x = years.astype(float)
    x0 = x.mean()
    x = x - x0
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * (float(target_year) - x0)
    return float(max(0.0, np.expm1(pred)))


@dataclass(frozen=True)
class MunicipalTotalsConfig:
    year_start: int = 2018
    year_end: int = 2024
    target_year: int = 2024
    min_months_reported_for_complete_year: int = 12
    min_usable_years_for_trend: int = 3
    # Prevent long-gap extrapolation explosions: only use trend if the most
    # recent usable year is within this many years of the target year.
    max_year_gap_for_trend: int = 1
    pop_bands: tuple[tuple[float, float, str], ...] = (
        (0, 5_000, "<5k"),
        (5_000, 10_000, "5k-10k"),
        (10_000, 25_000, "10k-25k"),
        (25_000, 50_000, "25k-50k"),
        (50_000, 100_000, "50k-100k"),
        (100_000, 250_000, "100k-250k"),
        (250_000, 500_000, "250k-500k"),
        (500_000, 1_000_000, "500k-1m"),
        (1_000_000, float("inf"), "1m+"),
    )


def _assign_pop_band(pop: float, bands: tuple[tuple[float, float, str], ...]) -> str:
    try:
        x = float(pop)
    except (TypeError, ValueError):
        return "unknown"
    for lo, hi, label in bands:
        if x >= float(lo) and x < float(hi):
            return str(label)
    return str(bands[-1][2]) if bands else "unknown"


def build_municipal_year_panel(
    *,
    canonical_df: pd.DataFrame,
    cfg: MunicipalTotalsConfig,
) -> pd.DataFrame:
    required = {"state_fips", "place_fips", "year", "months_reported"} | {f"{o}_srs" for o in OFFENSES_7}
    missing = required - set(canonical_df.columns)
    if missing:
        raise ValueError(f"canonical_df missing columns: {sorted(missing)}")

    df = canonical_df.copy()
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
    df["place_fips"] = df["place_fips"].astype(str).str.zfill(5)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["months_reported"] = pd.to_numeric(df["months_reported"], errors="coerce").fillna(0).astype(int)

    df = df[(df["year"] >= cfg.year_start) & (df["year"] <= cfg.year_end)].copy()

    df["bucket_id"] = df["state_fips"] + df["place_fips"]
    group_cols = ["bucket_id", "state_fips", "place_fips", "year"]

    out = (
        df.groupby(group_cols, dropna=True)
        .agg(
            agency_rows=("bucket_id", "size"),
            months_reported_min=("months_reported", "min"),
            months_reported_max=("months_reported", "max"),
            **{f"{o}_srs_sum": (f"{o}_srs", "sum") for o in OFFENSES_7},
        )
        .reset_index()
    )
    out["year"] = out["year"].astype(int)

    min_months = int(cfg.min_months_reported_for_complete_year)
    out["is_complete_year"] = out["months_reported_min"].astype(int) >= min_months

    for offense in OFFENSES_7:
        srs_sum = pd.to_numeric(out[f"{offense}_srs_sum"], errors="coerce")
        out[f"{offense}_observed"] = np.where(out["is_complete_year"] & srs_sum.notna(), srs_sum.astype(float), np.nan)

    return out


def estimate_municipal_totals_for_year(
    *,
    muni_year_panel: pd.DataFrame,
    muni_population: pd.DataFrame,
    cfg: MunicipalTotalsConfig,
    allowed_bucket_ids: set[str] | None = None,
) -> pd.DataFrame:
    required_panel = {"bucket_id", "state_fips", "year", "is_complete_year"} | {f"{o}_observed" for o in OFFENSES_7}
    missing_panel = required_panel - set(muni_year_panel.columns)
    if missing_panel:
        raise ValueError(f"muni_year_panel missing columns: {sorted(missing_panel)}")

    required_pop = {"bucket_id", "bucket_population"}
    missing_pop = required_pop - set(muni_population.columns)
    if missing_pop:
        raise ValueError(f"muni_population missing columns: {sorted(missing_pop)}")

    panel = muni_year_panel.copy()
    panel["bucket_id"] = panel["bucket_id"].astype(str)
    panel["state_fips"] = panel["state_fips"].astype(str).str.zfill(2)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype(int)

    pop = muni_population.copy()
    pop["bucket_id"] = pop["bucket_id"].astype(str)
    pop["bucket_population"] = pd.to_numeric(pop["bucket_population"], errors="coerce").fillna(0.0).astype(float)

    if allowed_bucket_ids is not None:
        allowed = {str(b) for b in allowed_bucket_ids}
        pop = pop[pop["bucket_id"].isin(allowed)].copy()
        panel = panel[panel["bucket_id"].isin(allowed)].copy()

    meta = pop.merge(panel[["bucket_id", "state_fips"]].drop_duplicates(), on="bucket_id", how="left")
    meta["pop_band"] = meta["bucket_population"].map(lambda v: _assign_pop_band(v, cfg.pop_bands))

    target_year = int(cfg.target_year)
    panel_target = panel[panel["year"] == target_year].set_index("bucket_id")

    out = meta[["bucket_id", "state_fips", "bucket_population", "pop_band"]].copy()
    out["target_year"] = target_year

    for offense in OFFENSES_7:
        est_col = offense
        src_col = f"total_source_{offense}"
        usable_col = f"usable_years_{offense}"

        series = panel[["bucket_id", "year", "is_complete_year", f"{offense}_observed"]].copy()
        series = series.sort_values(["bucket_id", "year"])

        est_map: dict[str, float] = {}
        src_map: dict[str, str] = {}
        usable_map: dict[str, int] = {}

        for bucket_id, grp in series.groupby("bucket_id", sort=False):
            years = grp["year"].to_numpy(dtype=int)
            observed = pd.to_numeric(grp[f"{offense}_observed"], errors="coerce").to_numpy(dtype=float)
            complete = grp["is_complete_year"].astype(bool).to_numpy()

            usable_mask = complete & np.isfinite(observed)
            usable_years = int(usable_mask.sum())
            usable_map[str(bucket_id)] = usable_years

            if bucket_id in panel_target.index:
                target_row = panel_target.loc[bucket_id]
                target_complete = bool(target_row.get("is_complete_year", False))
                target_observed = target_row.get(f"{offense}_observed")
                if target_complete and pd.notna(target_observed):
                    est_map[str(bucket_id)] = float(target_observed)
                    src_map[str(bucket_id)] = "reported"
                    continue

            if usable_years >= int(cfg.min_usable_years_for_trend):
                last_year = int(years[usable_mask].max()) if usable_years > 0 else None
                max_gap = int(getattr(cfg, "max_year_gap_for_trend", 1))
                if last_year is not None and (target_year - last_year) <= max_gap:
                    est = _project_count_log_linear(
                        years=years[usable_mask],
                        counts=observed[usable_mask],
                        target_year=target_year,
                    )
                    if np.isfinite(est):
                        est_map[str(bucket_id)] = float(est)
                        src_map[str(bucket_id)] = "trend_imputed"
                        continue

            est_map[str(bucket_id)] = float("nan")
            src_map[str(bucket_id)] = "missing"

        out[usable_col] = out["bucket_id"].map(lambda b: usable_map.get(str(b), 0)).astype(int)
        out[est_col] = out["bucket_id"].map(lambda b: est_map.get(str(b), float("nan"))).astype(float)
        out[src_col] = out["bucket_id"].map(lambda b: src_map.get(str(b), "missing")).astype(str)

        # Peer mean fallback for missing estimates (Option A).
        is_missing = ~np.isfinite(out[est_col].to_numpy(dtype=float))
        if not bool(is_missing.any()):
            continue

        peers = out.loc[~is_missing & (out["bucket_population"] > 0), ["bucket_id", "state_fips", "pop_band", "bucket_population", est_col]].copy()
        if peers.empty:
            # No peers at all: fall back to 0 for the missing set.
            out.loc[is_missing, est_col] = 0.0
            out.loc[is_missing, src_col] = "peer_fallback_zero"
            continue

        peer_group = (
            peers.groupby(["state_fips", "pop_band"], dropna=False)
            .agg(sum_count=(est_col, "sum"), sum_pop=("bucket_population", "sum"))
            .reset_index()
        )
        peer_group["peer_rate"] = np.where(peer_group["sum_pop"] > 0, 1e5 * peer_group["sum_count"] / peer_group["sum_pop"], np.nan)
        peer_rate_map = {(r.state_fips, r.pop_band): float(r.peer_rate) for r in peer_group.itertuples(index=False)}

        state_group = peers.groupby("state_fips", dropna=False).agg(
            sum_count=(est_col, "sum"), sum_pop=("bucket_population", "sum")
        )
        state_rate = (1e5 * state_group["sum_count"] / state_group["sum_pop"]).where(state_group["sum_pop"] > 0)
        state_rate_map = {str(k): float(v) for k, v in state_rate.items()}

        nat_sum_count = float(pd.to_numeric(peers[est_col], errors="coerce").fillna(0.0).sum())
        nat_sum_pop = float(pd.to_numeric(peers["bucket_population"], errors="coerce").fillna(0.0).sum())
        nat_rate = (1e5 * nat_sum_count / nat_sum_pop) if nat_sum_pop > 0 else float("nan")

        filled_counts = []
        filled_sources = []
        for bucket_id, state_fips, band, pop_val in zip(
            out.loc[is_missing, "bucket_id"].astype(str),
            out.loc[is_missing, "state_fips"].astype(str),
            out.loc[is_missing, "pop_band"].astype(str),
            out.loc[is_missing, "bucket_population"].to_numpy(dtype=float),
            strict=True,
        ):
            rate = peer_rate_map.get((state_fips, band))
            source = "peer_state_pop_band"
            if rate is None or not np.isfinite(rate):
                rate = state_rate_map.get(state_fips)
                source = "peer_state_overall"
            if rate is None or not np.isfinite(rate):
                rate = nat_rate
                source = "peer_national_overall"
            if rate is None or not np.isfinite(rate) or pop_val <= 0:
                filled_counts.append(0.0)
                filled_sources.append("peer_fallback_zero")
                continue
            filled_counts.append(float(rate) / 1e5 * float(pop_val))
            filled_sources.append(source)

        out.loc[is_missing, est_col] = filled_counts
        out.loc[is_missing, src_col] = filled_sources

    return out


def estimate_state_residual_share_floor(
    *,
    canonical_df: pd.DataFrame,
    state_constraints_df: pd.DataFrame,
    year_start: int,
    year_end: int,
    min_months_reported_for_complete_year: int = 12,
) -> pd.DataFrame:
    required_canonical = {"state_fips", "year", "months_reported"} | {f"{o}_srs" for o in OFFENSES_7}
    missing_canonical = required_canonical - set(canonical_df.columns)
    if missing_canonical:
        raise ValueError(f"canonical_df missing columns: {sorted(missing_canonical)}")

    required_constraints = {"state_fips", "year"} | set(OFFENSES_7)
    missing_constraints = required_constraints - set(state_constraints_df.columns)
    if missing_constraints:
        raise ValueError(f"state_constraints_df missing columns: {sorted(missing_constraints)}")

    canon = canonical_df.copy()
    canon["state_fips"] = canon["state_fips"].astype(str).str.zfill(2)
    canon["year"] = pd.to_numeric(canon["year"], errors="coerce").astype(int)
    canon["months_reported"] = pd.to_numeric(canon["months_reported"], errors="coerce").fillna(0).astype(int)
    canon = canon[(canon["year"] >= int(year_start)) & (canon["year"] <= int(year_end))].copy()

    constraints = state_constraints_df.copy()
    constraints["state_fips"] = constraints["state_fips"].astype(str).str.zfill(2)
    constraints["year"] = pd.to_numeric(constraints["year"], errors="coerce").astype(int)
    constraints = constraints[(constraints["year"] >= int(year_start)) & (constraints["year"] <= int(year_end))].copy()

    complete = canon["months_reported"].astype(int) >= int(min_months_reported_for_complete_year)
    muni = canon.loc[complete, ["state_fips", "year", *[f"{o}_srs" for o in OFFENSES_7]]].copy()
    for offense in OFFENSES_7:
        muni[f"{offense}_srs"] = pd.to_numeric(muni[f"{offense}_srs"], errors="coerce").fillna(0.0).astype(float)
    muni = (
        muni.groupby(["state_fips", "year"], dropna=False)
        .agg(**{offense: (f"{offense}_srs", "sum") for offense in OFFENSES_7})
        .reset_index()
    )

    merged = constraints.merge(muni, on=["state_fips", "year"], how="left", suffixes=("_state", "_muni"))
    for offense in OFFENSES_7:
        state_col = f"{offense}_state"
        muni_col = f"{offense}_muni"
        merged[state_col] = pd.to_numeric(merged[state_col], errors="coerce").fillna(0.0).astype(float)
        merged[muni_col] = pd.to_numeric(merged[muni_col], errors="coerce").fillna(0.0).astype(float)

        resid = (merged[state_col] - merged[muni_col]).clip(lower=0.0)
        merged[f"residual_share_{offense}"] = np.where(merged[state_col] > 0, resid / merged[state_col], 0.0)
        merged[f"residual_share_{offense}"] = (
            pd.to_numeric(merged[f"residual_share_{offense}"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        )

    floors = merged.groupby("state_fips", dropna=False).agg(
        **{f"residual_share_floor_{o}": (f"residual_share_{o}", "min") for o in OFFENSES_7}
    )
    floors = floors.reset_index()
    return floors
