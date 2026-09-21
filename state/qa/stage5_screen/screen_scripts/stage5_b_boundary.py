"""Stage 5 screen (b): publication-boundary coherence.

National census of adjacent block groups whose published display fates differ by more than
the data supports -- the "Maine three fates" class (near-white / grey / deep-blue on three
adjacent roadless townships off posterior noise).

Fate is defined against the viewer's ACTUAL paint rule (frontend/public/index.html):
  * index NULL  -> flat no-data grey #3a4150
  * otherwise   -> log-interpolated colour over the ONE fixed break set
                   INDEX_BREAKS = [12.5, 25, 50, 75, 100, 133, 200, 400, 800]

Writes to state/qa/stage5_screen/:
  s5b_boundary_incoherence_summary.csv   -- national census, per layer and per incoherence class
  s5b_boundary_pairs.parquet             -- FULL row-level population of incoherent adjacent pairs
  s5b_three_fates_triples.csv            -- BGs whose own neighbourhood spans grey + cool + warm
  s5b_grey_neighbour_reasons.csv         -- why the grey member is grey, and what its neighbour paints
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, "/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone/state/qa/stage5_screen/screen_scripts")
from stage5_common import INDEX_BREAKS, OFFENSES_7, OUT, RARE, load_bg, write  # noqa: E402

ADJ = Path("/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone/state/qa/stage5_screen/bg_adjacency_2020.parquet")
adj = pd.read_parquet(ADJ)
print(f"adjacency pairs {len(adj):,}")

bg = load_bg()
bg = bg.set_index("block_group_geoid")


def band_of(idx: pd.Series) -> pd.Series:
    """0..9 band id under the fixed break set; NaN where the index is null (grey)."""
    v = pd.to_numeric(idx, errors="coerce")
    b = pd.Series(np.nan, index=v.index, dtype=float)
    ok = v.notna()
    b.loc[ok] = np.searchsorted(np.array(INDEX_BREAKS), v.loc[ok].to_numpy(), side="right").astype(float)
    return b


def poisson_ci(counts: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros_like(counts, dtype=float)
    pos = counts > 0
    lo[pos] = 0.5 * chi2.ppf(alpha / 2.0, 2.0 * counts[pos])
    hi = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (counts + 1.0))
    return lo, hi


LAYERS = {
    "index_total_primary_event_weighted": "expected_count_total",
    "index_total_harm": "expected_count_total",
    "index_robbery_primary": "expected_count_robbery",
    "index_larceny_primary": "expected_count_larceny",
    "index_burglary_primary": "expected_count_burglary",
    "index_aggravated_assault_primary": "expected_count_aggravated_assault",
    "index_motor_vehicle_theft_primary": "expected_count_motor_vehicle_theft",
}

summary = []
pair_frames = []
for layer, count_col in LAYERS.items():
    idx = pd.to_numeric(bg[layer], errors="coerce")
    cnt = pd.to_numeric(bg[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    band = band_of(idx)
    # Proxy interval in INDEX units: scale the point index by the Poisson count ratio. For the
    # per-offense layers this reproduces the published CI exactly (same construction); for the
    # aggregates it is the count-derived equivalent, documented as a proxy.
    clo, chi_ = poisson_ci(cnt.to_numpy(dtype=float))
    ratio_lo = np.divide(clo, cnt.to_numpy(dtype=float), out=np.zeros_like(clo), where=cnt.to_numpy() > 0)
    ratio_hi = np.divide(chi_, cnt.to_numpy(dtype=float), out=np.full_like(chi_, np.inf), where=cnt.to_numpy() > 0)
    lo = idx.to_numpy(dtype=float) * ratio_lo
    hi = idx.to_numpy(dtype=float) * ratio_hi
    hi = np.where(cnt.to_numpy() > 0, hi, np.nan)

    frame = pd.DataFrame(
        {
            "idx": idx.to_numpy(dtype=float),
            "band": band.to_numpy(dtype=float),
            "cnt": cnt.to_numpy(dtype=float),
            "lo": lo,
            "hi": hi,
            "pop": pd.to_numeric(bg["population_2024"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "land": pd.to_numeric(bg["land_area_sq_mi"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "expo": pd.to_numeric(bg["exposure_proxy_2024"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        },
        index=bg.index,
    )

    a = frame.reindex(adj["bg"].to_numpy()).reset_index(drop=True).add_suffix("_a")
    b = frame.reindex(adj["nb"].to_numpy()).reset_index(drop=True).add_suffix("_b")
    p = pd.concat([adj.reset_index(drop=True), a, b], axis=1)
    # each unordered pair appears twice; keep one orientation
    p = p[p["bg"] < p["nb"]].reset_index(drop=True)

    grey_a = p["idx_a"].isna()
    grey_b = p["idx_b"].isna()
    both_col = ~grey_a & ~grey_b
    band_gap = (p["band_a"] - p["band_b"]).abs()
    ci_overlap = (p["lo_a"] <= p["hi_b"]) & (p["lo_b"] <= p["hi_a"])
    tiny = p[["cnt_a", "cnt_b"]].max(axis=1).le(5.0)
    both_tiny = p[["cnt_a", "cnt_b"]].max(axis=1).le(10.0)

    classes = {
        "band_gap_ge_2_with_overlapping_CIs": both_col & band_gap.ge(2) & ci_overlap,
        "band_gap_ge_3_with_overlapping_CIs": both_col & band_gap.ge(3) & ci_overlap,
        "band_gap_ge_4_with_overlapping_CIs": both_col & band_gap.ge(4) & ci_overlap,
        "band_gap_ge_3_on_tiny_counts(<=5)": both_col & band_gap.ge(3) & tiny,
        "band_gap_ge_4_on_tiny_counts(<=10)": both_col & band_gap.ge(4) & both_tiny,
        "grey_next_to_coloured": (grey_a ^ grey_b),
        "grey_next_to_coloured_similar_exposure": (grey_a ^ grey_b)
        & (
            (p[["expo_a", "expo_b"]].min(axis=1) / p[["expo_a", "expo_b"]].max(axis=1).replace(0.0, np.nan)).ge(0.5)
        ),
        "grey_next_to_warm(band>=6)": (grey_a & p["band_b"].ge(6)) | (grey_b & p["band_a"].ge(6)),
        "deep_blue_next_to_deep_red": (p["band_a"].le(1) & p["band_b"].ge(8))
        | (p["band_b"].le(1) & p["band_a"].ge(8)),
        "deep_blue_next_to_deep_red_overlapping_CIs": (
            (p["band_a"].le(1) & p["band_b"].ge(8)) | (p["band_b"].le(1) & p["band_a"].ge(8))
        )
        & ci_overlap,
    }
    for name, mask in classes.items():
        m = mask.fillna(False)
        summary.append(
            {
                "layer": layer,
                "class": name,
                "n_adjacent_pairs": int(len(p)),
                "n_pairs_in_class": int(m.sum()),
                "pct_pairs": round(100.0 * m.sum() / len(p), 4),
                "n_distinct_bgs": int(pd.unique(np.concatenate([p.loc[m, "bg"].to_numpy(), p.loc[m, "nb"].to_numpy()])).size)
                if int(m.sum())
                else 0,
                "median_max_count": float(p.loc[m, ["cnt_a", "cnt_b"]].max(axis=1).median()) if int(m.sum()) else np.nan,
                "median_max_land_sq_mi": float(p.loc[m, ["land_a", "land_b"]].max(axis=1).median()) if int(m.sum()) else np.nan,
            }
        )

    keep = (
        classes["band_gap_ge_3_with_overlapping_CIs"]
        | classes["grey_next_to_warm(band>=6)"]
        | classes["deep_blue_next_to_deep_red"]
    ).fillna(False)
    kept = p.loc[keep].copy()
    kept["layer"] = layer
    kept["band_gap"] = band_gap.loc[keep]
    kept["ci_overlap"] = ci_overlap.loc[keep]
    pair_frames.append(kept)

write(pd.DataFrame(summary), "s5b_boundary_incoherence_summary.csv")
pairs = pd.concat(pair_frames, ignore_index=True)
write(pairs, "s5b_boundary_pairs.parquet")

# ------------------------------------------------------------------ three fates
# A block group whose OWN immediate neighbourhood (itself + neighbours) simultaneously shows
# grey (no data), a cool band (<=50) and a warm band (>=200) on the DEFAULT layer, with every
# member carrying a small count -- i.e. the difference cannot be read off the counts.
layer = "index_total_primary_event_weighted"
idx = pd.to_numeric(bg[layer], errors="coerce")
band = band_of(idx)
cnt = pd.to_numeric(bg["expected_count_total"], errors="coerce").fillna(0.0)
info = pd.DataFrame({"band": band, "idx": idx, "cnt": cnt, "land": pd.to_numeric(bg["land_area_sq_mi"], errors="coerce")})

nb = adj.copy()
nb = nb.join(info.add_suffix("_nb"), on="nb")
grp = nb.groupby("bg")
agg = grp.agg(
    n_nb=("nb", "size"),
    n_grey=("band_nb", lambda s: int(s.isna().sum())),
    n_cool=("band_nb", lambda s: int((s <= 2).sum())),
    n_warm=("band_nb", lambda s: int((s >= 7).sum())),
    max_nb_cnt=("cnt_nb", "max"),
    min_nb_idx=("idx_nb", "min"),
    max_nb_idx=("idx_nb", "max"),
)
agg = agg.join(info, how="left")
three = agg[
    (agg["n_grey"] >= 1)
    & (agg["n_cool"] >= 1)
    & ((agg["n_warm"] >= 1) | (agg["band"] >= 7))
    & (agg[["max_nb_cnt", "cnt"]].max(axis=1) <= 25.0)
].copy()
three = three.reset_index().rename(columns={"bg": "block_group_geoid"})
three["state_fips"] = three["block_group_geoid"].str.slice(0, 2)
write(three.sort_values(["state_fips", "block_group_geoid"]), "s5b_three_fates_triples.csv")

# ------------------------------------------------------------------ grey neighbour reasons
grey = idx.isna()
grey_ids = bg.index[grey]
reason_rows = []
for gid in grey_ids:
    r = bg.loc[gid]
    reasons = sorted(
        {
            str(r[f"denominator_reason_{o}"])
            for o in OFFENSES_7
            if pd.notna(r[f"denominator_reason_{o}"]) and str(r[f"denominator_reason_{o}"]) != "publishable"
        }
    )
    modes = sorted({str(r[f"estimate_mode_{o}"]) for o in OFFENSES_7 if str(r[f"estimate_mode_{o}"]) != "count_derived"})
    reason_rows.append(
        {
            "block_group_geoid": gid,
            "state_fips": r["state_fips"],
            "population_2024": r["population_2024"],
            "households_total": r["households_total"],
            "exposure_proxy_2024": r["exposure_proxy_2024"],
            "vehicle_exposure_2024": r["vehicle_exposure_2024"],
            "burglary_premises_total": r["burglary_premises_total"],
            "land_area_sq_mi": r["land_area_sq_mi"],
            "expected_count_total": r["expected_count_total"],
            "crime_density_total": r["crime_density_total"],
            "non_residential_flag": r["non_residential_flag"],
            "special_use_tract_flag": r["special_use_tract_flag"],
            "suppression_reasons": "|".join(reasons),
            "suppression_modes": "|".join(modes),
            "n_offenses_published": int(
                sum(pd.notna(r[f"index_{o}_primary"]) for o in OFFENSES_7 if o not in RARE)
            ),
        }
    )
grey_df = pd.DataFrame(reason_rows)
# what do the grey cell's neighbours paint?
nbstat = (
    adj[adj["bg"].isin(set(grey_ids))]
    .join(info.add_suffix("_nb"), on="nb")
    .groupby("bg")
    .agg(
        n_neighbours=("nb", "size"),
        n_neighbours_coloured=("idx_nb", "count"),
        neighbour_median_index=("idx_nb", "median"),
        neighbour_max_index=("idx_nb", "max"),
    )
)
grey_df = grey_df.merge(nbstat, left_on="block_group_geoid", right_index=True, how="left")
write(grey_df.sort_values(["suppression_modes", "state_fips"]), "s5b_grey_neighbour_reasons.csv")
print("done screen b")
