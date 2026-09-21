"""Stage 5 screen (c): transient / visitor exposure -- where person-exposure understates
ambient load (North Bend class) vs where LandScan catches it (Black Hawk class).

Writes to state/qa/stage5_screen/:
  s5c_exposure_lift_census.csv          -- national distribution of the ambient lift, by stratum
  s5c_transient_flag_audit.csv          -- the transient_exposure_likely_* guard: how often it fires
  s5c_ambient_blindspot_bg.csv          -- FULL population of the North Bend class, ranked
  s5c_ambient_caught_bg.csv             -- the Black Hawk class (lift working), for contrast
  s5c_named_cases.csv                   -- North Bend / Black Hawk / Denver / Seattle traced
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone/state/qa/stage5_screen/screen_scripts")
from stage5_common import (  # noqa: E402
    OFFENSES_7,
    TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO,
    TRANSIENT_EXPOSURE_INDEX_THRESHOLD,
    load_bg,
    write,
)

bg = load_bg(extra=["commercial_premises_total"])
n = len(bg)

pop = pd.to_numeric(bg["population_2024"], errors="coerce").fillna(0.0)
jobs = pd.to_numeric(bg["daytime_population_jobs_proxy"], errors="coerce").fillna(0.0)
ls = pd.to_numeric(bg["landscan_day_pop"], errors="coerce").fillna(0.0)
expo = pd.to_numeric(bg["exposure_proxy_2024"], errors="coerce").fillna(0.0)
poi = pd.to_numeric(bg["destination_poi_total"], errors="coerce").fillna(0.0)
retail = pd.to_numeric(bg["lodes_retail_jobs"], errors="coerce").fillna(0.0)
comm = pd.to_numeric(bg.get("commercial_premises_total", pd.Series(0.0, index=bg.index)), errors="coerce").fillna(0.0)
hh = pd.to_numeric(bg["households_total"], errors="coerce").fillna(0.0)
land = pd.to_numeric(bg["land_area_sq_mi"], errors="coerce").fillna(0.0)
tot = pd.to_numeric(bg["expected_count_total"], errors="coerce").fillna(0.0)
evw = pd.to_numeric(bg["index_total_primary_event_weighted"], errors="coerce")

lift = pd.Series(np.nan, index=bg.index, dtype=float)
ok = pop.gt(0.0)
lift[ok] = expo[ok] / pop[ok]
ls_ratio = pd.Series(np.nan, index=bg.index, dtype=float)
ls_ratio[ok] = ls[ok] / pop[ok]
jobs_ratio = pd.Series(np.nan, index=bg.index, dtype=float)
jobs_ratio[ok] = jobs[ok] / pop[ok]

bg["_lift"] = lift
bg["_ls_ratio"] = ls_ratio
bg["_jobs_ratio"] = jobs_ratio

# ------------------------------------------------------- 1. lift census
rows = []


def add(name, mask, note=""):
    m = pd.Series(mask, index=bg.index).fillna(False).astype(bool)
    rows.append(
        {
            "class": name,
            "n_bgs": int(m.sum()),
            "pct_bgs": round(100.0 * m.sum() / n, 4),
            "population": float(pop[m].sum()),
            "expected_count_total": float(tot[m].sum()),
            "median_lift": float(lift[m].median()),
            "median_index_evw": float(evw[m].median()),
            "note": note,
        }
    )


add("all block groups", pd.Series(True, index=bg.index))
add("exposure == resident population exactly", (expo - pop).abs().le(1e-9) & pop.gt(0),
    "no ambient lift at all: neither LODES jobs nor LandScan raised the denominator")
add("landscan_day_pop < resident population", ls.lt(pop) & pop.gt(0),
    "LandScan's ambient day surface reports FEWER people than the census resident count, "
    "so the max() floor contributes nothing")
add("landscan_day_pop == 0", ls.le(0.0), "no LandScan coverage at all for this BG")
add("lift >= 1.25", lift.ge(1.25))
add("lift >= 2", lift.ge(2.0))
add("lift >= 5 (the transient guard's ratio threshold)", lift.ge(TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO))
add("lift >= 5 AND index_evw >= 1000 (both guard conditions)",
    lift.ge(TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO) & evw.ge(TRANSIENT_EXPOSURE_INDEX_THRESHOLD))
add("landscan is the binding term (ls > jobs)", ls.gt(jobs))
add("jobs proxy is the binding term (jobs >= ls)", jobs.ge(ls))
for lo, hi in [(0, 1.0), (1.0, 1.25), (1.25, 2.0), (2.0, 5.0), (5.0, 1e9)]:
    add(f"lift in [{lo}, {hi})", lift.ge(lo) & lift.lt(hi))
for s in sorted(bg["urban_stratum"].dropna().astype(str).unique().tolist()):
    add(f"stratum={s}", bg["urban_stratum"].astype(str).eq(s))
write(pd.DataFrame(rows), "s5c_exposure_lift_census.csv")

# ------------------------------------------------------- 2. transient guard audit
guard = []
for o in OFFENSES_7:
    flag = bg[f"transient_exposure_likely_{o}"].fillna(False).astype(bool)
    idx = pd.to_numeric(bg[f"index_{o}_primary"], errors="coerce")
    guard.append(
        {
            "offense": o,
            "n_flag_true": int(flag.sum()),
            "pct_flag_true": round(100.0 * flag.sum() / n, 5),
            "n_index_published": int(idx.notna().sum()),
            "n_index_ge_1000": int(idx.ge(TRANSIENT_EXPOSURE_INDEX_THRESHOLD).sum()),
            "n_lift_ge_5": int(lift.ge(TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO).sum()),
            "n_index_ge_1000_and_lift_lt_5": int(
                (idx.ge(TRANSIENT_EXPOSURE_INDEX_THRESHOLD) & lift.lt(TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO)).sum()
            ),
            "n_index_ge_1000_and_no_lift_at_all": int(
                (idx.ge(TRANSIENT_EXPOSURE_INDEX_THRESHOLD) & (expo - pop).abs().le(1e-9)).sum()
            ),
            "guard_direction_note": (
                "the guard requires exposure/pop >= 5 AND index >= 1000, so it can only fire where the "
                "denominator ALREADY carries the ambient load. The blind spot (no lift) is unreachable "
                "by construction; the flag is not published in the tiles either."
            ),
        }
    )
write(pd.DataFrame(guard), "s5c_transient_flag_audit.csv")

# ------------------------------------------------------- 3. the blind-spot population
# North Bend class: the denominator got NO ambient credit (exposure ~ residents), the BG carries
# real destination/retail evidence of visitor load, and the published index is elevated.
retail_per_res = pd.Series(np.nan, index=bg.index, dtype=float)
retail_per_res[ok] = retail[ok] / pop[ok]
poi_per_hh = pd.Series(np.nan, index=bg.index, dtype=float)
poi_per_hh[hh.gt(0)] = poi[hh.gt(0)] / hh[hh.gt(0)]

destination_evidence = (
    (retail_per_res.ge(0.10))
    | (poi.ge(20.0))
    | (comm.ge(20.0))
    | (retail.ge(250.0))
)
no_lift = lift.lt(1.25) & pop.gt(0)
elevated = evw.ge(200.0)

blind = no_lift & destination_evidence & elevated
out = pd.DataFrame(
    {
        "block_group_geoid": bg["block_group_geoid"],
        "state_fips": bg["state_fips"],
        "tract_id": bg["tract_id"],
        "population_2024": pop,
        "households_total": hh,
        "daytime_population_jobs_proxy": jobs,
        "landscan_day_pop": ls,
        "exposure_proxy_2024": expo,
        "exposure_lift_ratio": lift,
        "landscan_over_resident": ls_ratio,
        "destination_poi_total": poi,
        "commercial_premises_total": comm,
        "lodes_retail_jobs": retail,
        "retail_jobs_per_resident": retail_per_res,
        "land_area_sq_mi": land,
        "urban_stratum": bg["urban_stratum"],
        "expected_count_total": tot,
        "index_total_primary_event_weighted": evw,
        "index_larceny_primary": pd.to_numeric(bg["index_larceny_primary"], errors="coerce"),
        "expected_count_larceny": pd.to_numeric(bg["expected_count_larceny"], errors="coerce"),
        "index_motor_vehicle_theft_primary": pd.to_numeric(bg["index_motor_vehicle_theft_primary"], errors="coerce"),
        "index_robbery_primary": pd.to_numeric(bg["index_robbery_primary"], errors="coerce"),
        "reliability_tier_larceny": bg["reliability_tier_larceny"],
        "numerator_support_source_larceny": bg["numerator_support_source_larceny"],
        "source_mode_larceny": bg["source_mode_larceny"],
        "transient_exposure_likely_larceny": bg["transient_exposure_likely_larceny"],
        "recommended_display_geography_larceny": bg["recommended_display_geography_larceny"],
    }
)
blind_df = out[blind].sort_values("index_total_primary_event_weighted", ascending=False)
write(blind_df, "s5c_ambient_blindspot_bg.csv")

caught = (lift.ge(2.0)) & elevated
caught_df = out[caught].sort_values("index_total_primary_event_weighted", ascending=False)
write(caught_df, "s5c_ambient_caught_bg.csv")

# ------------------------------------------------------- 4. named cases
named = {
    "530330327044": "North Bend WA (outlet-mall larceny; LandScan BELOW resident -> no lift)",
    "080470138023": "Black Hawk CO (casino district; LandScan 5x lift -> caught)",
    "080050071102": "Denver-area 080050071102 (recorded too-low-floor case, exposure 461)",
    "530330328003": "Seattle-area 530330328003 (recorded too-low-floor case, exposure 613)",
    "060379304002": "Angeles NF 060379304002 (giant low-exposure polygon, v19 class B)",
    "320030076003": "Clark NV 320030076003 (761 sq mi, 379 people)",
    "250277042021": "MA Quabbin 250277042021 (97.8% of the state remainder pool mass)",
}
sel = out[out["block_group_geoid"].isin(named)].copy()
sel["case"] = sel["block_group_geoid"].map(named)
write(sel.sort_values("case"), "s5c_named_cases.csv")
print("done screen c")
