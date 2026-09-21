"""Stage 5 screen (a): floor-binding census + designed-eligibility precision failure population.

Writes to state/qa/stage5_screen/:
  s5a_floor_binding_census.csv          -- every floor x offense, how many BGs/tracts bind it
  s5a_landscan_lift_verification.json   -- verify the recorded 24.6304% lift claim on the promoted surface
  s5a_precision_failure_summary.csv     -- per offense, the size of every precision-test failure population
  s5a_precision_failures_bg.parquet     -- the FULL row-level population (BG x offense) failing the test
  s5a_known_low_floor_cases.csv         -- the two recorded cases traced field by field
  s5a_exposure_floor_sweep.csv          -- what a higher person-exposure floor would suppress
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone/state/qa/stage5_screen/screen_scripts")
from stage5_common import (  # noqa: E402
    BURGLARY_PREMISES_DENOMINATOR_FLOOR,
    INDEX_BREAKS,
    MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR,
    NON_RESIDENTIAL_HOUSEHOLD_FLOOR,
    OFFENSES_7,
    OUT,
    PERSON_EXPOSURE_DENOMINATOR_FLOOR,
    PERSON_EXPOSURE_FLOOR_OFFENSES,
    RARE,
    VOLUME,
    bands_spanned,
    implied_cv,
    load_bg,
    load_tract,
    write,
)

bg = load_bg()
tr = load_tract()
print(f"BG rows {len(bg):,}  tract rows {len(tr):,}")

# ---------------------------------------------------------------- 1. floor binding census
rows = []


def census(frame: pd.DataFrame, level: str) -> None:
    n = len(frame)
    hh = pd.to_numeric(frame["households_total"], errors="coerce").fillna(0.0)
    expo = pd.to_numeric(frame["exposure_proxy_2024"], errors="coerce").fillna(0.0)
    veh = pd.to_numeric(frame["vehicle_exposure_2024"], errors="coerce").fillna(0.0)
    prem = pd.to_numeric(frame["burglary_premises_total"], errors="coerce").fillna(0.0)
    res = pd.to_numeric(frame["resident_secondary_denominator"], errors="coerce").fillna(0.0)
    special = frame["special_use_tract_flag"].fillna(False).astype(bool)
    nonres = hh.lt(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)

    def add(rule, scope, value, mask, note=""):
        m = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
        rows.append(
            {
                "level": level,
                "rule": rule,
                "scope": scope,
                "threshold": value,
                "n_units": n,
                "n_binding": int(m.sum()),
                "pct_binding": round(100.0 * m.sum() / n, 4),
                "population_in_binding": float(
                    pd.to_numeric(frame["population_2024"], errors="coerce").fillna(0.0)[m].sum()
                ),
                "expected_count_total_in_binding": float(
                    pd.to_numeric(frame["expected_count_total"], errors="coerce").fillna(0.0)[m].sum()
                ),
                "land_area_sq_mi_in_binding": float(
                    pd.to_numeric(frame["land_area_sq_mi"], errors="coerce").fillna(0.0)[m].sum()
                ),
                "note": note,
            }
        )

    add("non_residential_household_floor", "all 7 offenses (primary+resident+all aggregates)",
        NON_RESIDENTIAL_HOUSEHOLD_FLOOR, nonres, "households_total < 10")
    add("special_use_tract_flag", "all 7 offenses (primary+resident+all aggregates)",
        "tract code 98xx", special, "GEOID chars 6:11 start with '98'")
    add("special_use_or_burglary_premises_floor", "burglary only",
        BURGLARY_PREMISES_DENOMINATOR_FLOOR, special | prem.lt(BURGLARY_PREMISES_DENOMINATOR_FLOOR),
        "burglary folds its premises floor INTO the special_use mask")
    add("burglary_premises_floor_alone", "burglary only",
        BURGLARY_PREMISES_DENOMINATOR_FLOOR, prem.lt(BURGLARY_PREMISES_DENOMINATOR_FLOOR), "")
    add("person_exposure_denominator_floor", "murder/rape/robbery/agg_assault/larceny primary",
        PERSON_EXPOSURE_DENOMINATOR_FLOOR, expo.lt(PERSON_EXPOSURE_DENOMINATOR_FLOOR), "")
    add("person_exposure_denominator_floor_resident", "murder/rape/robbery/agg_assault/larceny resident",
        PERSON_EXPOSURE_DENOMINATOR_FLOOR, res.lt(PERSON_EXPOSURE_DENOMINATOR_FLOOR),
        "resident arm applies the SAME 50 floor to resident_secondary_denominator")
    add("mvt_vehicle_exposure_denominator_floor", "motor_vehicle_theft primary AND resident",
        MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR, veh.lt(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR),
        "SURPRISE: the resident arm for MVT also tests the VEHICLE denominator, not resident pop")
    add("zero_primary_denominator_exposure", "person offenses", 0.0, expo.le(0.0), "")
    add("zero_primary_denominator_vehicles", "motor_vehicle_theft", 0.0, veh.le(0.0), "")
    add("zero_primary_denominator_premises", "burglary", 0.0, prem.le(0.0), "")
    add("zero_resident_denominator", "all resident indices", 0.0, res.le(0.0), "")
    add("no_floor_bound_but_denominator_positive_burglary", "burglary", "n/a",
        (~special) & prem.ge(BURGLARY_PREMISES_DENOMINATOR_FLOOR) & (~nonres), "eligible population")

    # what actually renders per offense
    for o in OFFENSES_7:
        pub = frame[f"primary_index_publishable_{o}"].fillna(False).astype(bool)
        idx_null = pd.to_numeric(frame[f"index_{o}_primary"], errors="coerce").isna()
        add(f"primary_index_publishable_{o}", f"{o} primary", "flag", pub, "publishable flag TRUE")
        add(f"primary_index_null_{o}", f"{o} primary", "rendered", idx_null,
            "index_{o}_primary is NULL in the published parquet")
        mode = frame[f"estimate_mode_{o}"].astype("string")
        for m in sorted(mode.dropna().unique().tolist()):
            add(f"estimate_mode_{o}={m}", f"{o}", "mode", mode.eq(m), "")


census(bg, "block_group")
census(tr, "tract")
census_df = pd.DataFrame(rows)
write(census_df, "s5a_floor_binding_census.csv")

# ---------------------------------------------------------------- 2. LandScan lift verification
jobs = pd.to_numeric(bg["daytime_population_jobs_proxy"], errors="coerce").fillna(0.0)
ls = pd.to_numeric(bg["landscan_day_pop"], errors="coerce").fillna(0.0)
expo = pd.to_numeric(bg["exposure_proxy_2024"], errors="coerce").fillna(0.0)
pop = pd.to_numeric(bg["population_2024"], errors="coerce").fillna(0.0)
lift_flag = bg["landscan_day_lifted_person_exposure"].fillna(False).astype(bool)
recomputed_lift = ls.gt(jobs)
capped = bg["person_exposure_hq_jobs_capped"].fillna(False).astype(bool)
cap_cand = bg["person_exposure_hq_jobs_cap_candidate"].fillna(False).astype(bool)

pre_cap = pd.to_numeric(bg["person_exposure_before_hq_jobs_cap"], errors="coerce").fillna(0.0)
expected_pre_cap = np.maximum(jobs, ls)

ls_json = {
    "surface": "state/output/crimerisk_block_group_2024_ags_core.parquet (promoted v20)",
    "n_block_groups": int(len(bg)),
    "recorded_claim_docs_STATE_md_line_954": {
        "text": "58,668 of 238,193 BGs lift (24.6304%)",
        "n_lifted": 58668,
        "pct": 24.6304,
    },
    "measured_on_promoted_surface": {
        "flag_landscan_day_lifted_person_exposure_true": int(lift_flag.sum()),
        "pct_flag": round(100.0 * lift_flag.sum() / len(bg), 4),
        "recomputed_landscan_gt_jobs_proxy": int(recomputed_lift.sum()),
        "pct_recomputed": round(100.0 * recomputed_lift.sum() / len(bg), 4),
        "flag_vs_recomputed_disagreements": int((lift_flag != recomputed_lift).sum()),
    },
    "claim_verdict": (
        "CONFIRMED"
        if abs(100.0 * lift_flag.sum() / len(bg) - 24.6304) < 0.01
        else "DIVERGED"
    ),
    "exposure_sums": {
        "population_2024": float(pop.sum()),
        "daytime_population_jobs_proxy": float(jobs.sum()),
        "landscan_day_pop": float(ls.sum()),
        "exposure_proxy_2024_published": float(expo.sum()),
        "max(jobs,landscan)_recomputed_pre_cap": float(expected_pre_cap.sum()),
        "person_exposure_before_hq_jobs_cap_published": float(pre_cap.sum()),
        "pre_cap_recompute_max_abs_dev": float(np.abs(pre_cap - expected_pre_cap).max()),
    },
    "hq_jobs_cap": {
        "candidates": int(cap_cand.sum()),
        "applied": int(capped.sum()),
        "exposure_removed_by_cap": float((pre_cap - expo)[capped].sum()),
        "note": "cap = 3 x max(landscan_day_pop, population); candidate needs jobs_wac>=5000",
    },
    "lift_material_effect": {
        "bgs_lifted_from_below_50_to_at_or_above_50": int(
            (recomputed_lift & jobs.lt(PERSON_EXPOSURE_DENOMINATOR_FLOOR) & expo.ge(PERSON_EXPOSURE_DENOMINATOR_FLOOR)).sum()
        ),
        "bgs_still_below_50_after_lift": int(expo.lt(PERSON_EXPOSURE_DENOMINATOR_FLOOR).sum()),
        "median_lift_ratio_where_lifted": float((expo / jobs.replace(0.0, np.nan))[recomputed_lift].median()),
        "p99_lift_ratio_where_lifted": float((expo / jobs.replace(0.0, np.nan))[recomputed_lift].quantile(0.99)),
    },
    "landscan_vintage": "LandScan USA 2021 (data/LandScan-USA/block_group_landscan_usa_2021.parquet); denominator year is 2024",
}
lift_by_state = (
    pd.DataFrame({"state_fips": bg["state_fips"], "lift": lift_flag.astype(int)})
    .groupby("state_fips")["lift"]
    .agg(["size", "sum"])
    .rename(columns={"size": "n_bgs", "sum": "n_lifted"})
)
lift_by_state["pct_lifted"] = (100.0 * lift_by_state["n_lifted"] / lift_by_state["n_bgs"]).round(3)
ls_json["lift_by_state"] = lift_by_state.reset_index().to_dict(orient="records")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "s5a_landscan_lift_verification.json").write_text(json.dumps(ls_json, indent=2))
print("wrote s5a_landscan_lift_verification.json  verdict=", ls_json["claim_verdict"])

# ---------------------------------------------------------------- 3. precision failure population
# The sol-designed test (docs/STATE.md v20 program item B): "a realized-uncertainty test
# (posterior CV / interval spans <= declared band count, using the published CI widths;
# U/L ratios rejected -- meaningless near zero)".
#
# Declared band count = the viewer's ONE fixed index break set (9 breaks / 10 bands).
# We measure, per published (BG, offense) index:
#   bands_crossed  = number of fixed break boundaries strictly inside the published CI
#   implied_cv     = CI width / (2*1.96*point)
# and enumerate the failure population at every candidate threshold so the fix can be sized
# without re-running the screen.

fail_rows = []
detail_frames = []
for level, frame in (("block_group", bg), ("tract", tr)):
    key = "block_group_geoid" if level == "block_group" else "tract_id"
    for o in OFFENSES_7:
        pt = pd.to_numeric(frame[f"index_{o}_primary"], errors="coerce")
        lo = pd.to_numeric(frame[f"index_{o}_primary_ci95_lower"], errors="coerce")
        hi = pd.to_numeric(frame[f"index_{o}_primary_ci95_upper"], errors="coerce")
        published = pt.notna()
        if int(published.sum()) == 0:
            fail_rows.append(
                {
                    "level": level,
                    "offense": o,
                    "n_published": 0,
                    "note": "no published index at this support (rare-offense tract-support policy)",
                }
            )
            continue
        nb = bands_spanned(lo, hi)
        cv = implied_cv(pd.to_numeric(frame[f"index_{o}_primary_ci95_width"], errors="coerce"), pt)
        cnt = pd.to_numeric(frame[f"expected_count_{o}"], errors="coerce").fillna(0.0)
        den = pd.to_numeric(frame[f"primary_denominator_{o}"], errors="coerce").fillna(0.0)
        rec = {
            "level": level,
            "offense": o,
            "n_units": len(frame),
            "n_published": int(published.sum()),
            "median_expected_count": float(cnt[published].median()),
            "median_denominator": float(den[published].median()),
            "median_bands_crossed": float(nb[published].median()),
            "median_implied_cv": float(cv[published].median()),
        }
        for t in (1, 2, 3, 4, 5, 6, 7, 8):
            m = published & nb.gt(t)
            rec[f"n_bands_gt_{t}"] = int(m.sum())
            rec[f"pct_bands_gt_{t}"] = round(100.0 * m.sum() / published.sum(), 3)
        for t in (0.25, 0.35, 0.5, 0.75, 1.0):
            m = published & cv.gt(t)
            rec[f"n_cv_gt_{t}"] = int(m.sum())
            rec[f"pct_cv_gt_{t}"] = round(100.0 * m.sum() / published.sum(), 3)
        # counts driving it
        for c in (0, 1, 2, 3, 5, 10, 25):
            rec[f"n_published_count_le_{c}"] = int((published & cnt.le(c)).sum())
        rec["n_published_zero_count"] = int((published & cnt.le(0.0)).sum())
        rec["n_published_zero_count_index_zero"] = int((published & cnt.le(0.0) & pt.le(0.0)).sum())
        fail_rows.append(rec)

        if level == "block_group":
            d = pd.DataFrame(
                {
                    "geoid": frame[key],
                    "state_fips": frame["state_fips"],
                    "level": level,
                    "offense": o,
                    "expected_count": cnt,
                    "primary_denominator": den,
                    "denominator_type": frame[f"primary_denominator_type_{o}"].astype("string"),
                    "index_primary": pt,
                    "ci95_lower": lo,
                    "ci95_upper": hi,
                    "ci95_width_ratio": pd.to_numeric(
                        frame[f"index_{o}_primary_ci95_width_ratio"], errors="coerce"
                    ),
                    "bands_crossed": nb,
                    "implied_cv": cv,
                    "reliability_tier": frame[f"reliability_tier_{o}"].astype("string"),
                    "effective_numerator_support": pd.to_numeric(
                        frame[f"effective_numerator_support_{o}"], errors="coerce"
                    ),
                    "numerator_support_source": frame[f"numerator_support_source_{o}"].astype("string"),
                    "estimate_mode": frame[f"estimate_mode_{o}"].astype("string"),
                    "benchmark_imputed_share": pd.to_numeric(
                        frame[f"benchmark_imputed_share_{o}"], errors="coerce"
                    ),
                    "confidence_tier": frame[f"confidence_tier_{o}"].astype("string"),
                    "population_2024": pd.to_numeric(frame["population_2024"], errors="coerce"),
                    "land_area_sq_mi": pd.to_numeric(frame["land_area_sq_mi"], errors="coerce"),
                    "urban_stratum": frame["urban_stratum"].astype("string"),
                }
            )
            detail_frames.append(d[published & (nb.gt(2) | cv.gt(0.35))])

write(pd.DataFrame(fail_rows), "s5a_precision_failure_summary.csv")
detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
write(detail, "s5a_precision_failures_bg.parquet")

# ---------------------------------------------------------------- 4. known too-low-floor cases
known = ["080050071102", "530330328003"]
cases = bg[bg["block_group_geoid"].isin(known)].copy()
recs = []
for _, r in cases.iterrows():
    base = {
        "block_group_geoid": r["block_group_geoid"],
        "state_fips": r["state_fips"],
        "population_2024": r["population_2024"],
        "households_total": r["households_total"],
        "daytime_population_jobs_proxy": r["daytime_population_jobs_proxy"],
        "landscan_day_pop": r["landscan_day_pop"],
        "exposure_proxy_2024": r["exposure_proxy_2024"],
        "landscan_lifted": r["landscan_day_lifted_person_exposure"],
        "hq_jobs_capped": r["person_exposure_hq_jobs_capped"],
        "vehicle_exposure_2024": r["vehicle_exposure_2024"],
        "burglary_premises_total": r["burglary_premises_total"],
        "land_area_sq_mi": r["land_area_sq_mi"],
        "urban_stratum": r["urban_stratum"],
        "index_total_primary_event_weighted": r["index_total_primary_event_weighted"],
        "index_total_harm": r["index_total_harm"],
        "expected_count_total": r["expected_count_total"],
    }
    for o in OFFENSES_7:
        lo = r[f"index_{o}_primary_ci95_lower"]
        hi = r[f"index_{o}_primary_ci95_upper"]
        nb = float(sum(1 for b in INDEX_BREAKS if pd.notna(lo) and pd.notna(hi) and lo < b < hi))
        recs.append(
            {
                **base,
                "offense": o,
                "expected_count": r[f"expected_count_{o}"],
                "primary_denominator": r[f"primary_denominator_{o}"],
                "denominator_type": r[f"primary_denominator_type_{o}"],
                "index_primary": r[f"index_{o}_primary"],
                "ci95_lower": lo,
                "ci95_upper": hi,
                "bands_crossed": nb,
                "implied_cv": (
                    (hi - lo) / (2 * 1.959963984540054 * r[f"index_{o}_primary"])
                    if pd.notna(r[f"index_{o}_primary"]) and r[f"index_{o}_primary"] > 0
                    else np.nan
                ),
                "estimate_mode": r[f"estimate_mode_{o}"],
                "reliability_tier": r[f"reliability_tier_{o}"],
                "recommended_display_geography": r[f"recommended_display_geography_{o}"],
                "confidence_tier": r[f"confidence_tier_{o}"],
                "benchmark_imputed_share": r[f"benchmark_imputed_share_{o}"],
                "binds_person_floor_50": (
                    bool(r["exposure_proxy_2024"] < PERSON_EXPOSURE_DENOMINATOR_FLOOR)
                    if o in PERSON_EXPOSURE_FLOOR_OFFENSES
                    else None
                ),
            }
        )
write(pd.DataFrame(recs), "s5a_known_low_floor_cases.csv")

# ---------------------------------------------------------------- 5. exposure floor sweep
sweep = []
expo = pd.to_numeric(bg["exposure_proxy_2024"], errors="coerce").fillna(0.0)
veh = pd.to_numeric(bg["vehicle_exposure_2024"], errors="coerce").fillna(0.0)
prem = pd.to_numeric(bg["burglary_premises_total"], errors="coerce").fillna(0.0)
pop = pd.to_numeric(bg["population_2024"], errors="coerce").fillna(0.0)
tot = pd.to_numeric(bg["expected_count_total"], errors="coerce").fillna(0.0)
for f in (50, 100, 150, 250, 400, 500, 613, 750, 1000, 1500, 2000):
    m_p = expo.lt(f)
    m_v = veh.lt(f)
    m_b = prem.lt(f)
    sweep.append(
        {
            "candidate_floor": f,
            "person_exposure_bgs_below": int(m_p.sum()),
            "person_exposure_pct": round(100.0 * m_p.sum() / len(bg), 3),
            "person_exposure_population_affected": float(pop[m_p].sum()),
            "person_exposure_counts_affected": float(tot[m_p].sum()),
            "vehicle_exposure_bgs_below": int(m_v.sum()),
            "vehicle_exposure_pct": round(100.0 * m_v.sum() / len(bg), 3),
            "premises_bgs_below": int(m_b.sum()),
            "premises_pct": round(100.0 * m_b.sum() / len(bg), 3),
        }
    )
write(pd.DataFrame(sweep), "s5a_exposure_floor_sweep.csv")
print("done screen a")
