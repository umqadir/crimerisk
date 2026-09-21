"""Stage 5 screen (a) addendum: what the sol-designed realized-uncertainty test would actually
suppress, and whether it reaches its stated target (the giant low-exposure polygons).

Writes to state/qa/stage5_screen/:
  s5a_designed_test_impact.csv   -- per (test, threshold): BGs suppressed per offense, and the
                                    knock-on to the all-or-null default aggregate layer
  s5a_designed_test_targets.csv  -- does the test catch the recorded Class-B exemplars?
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone/state/qa/stage5_screen/screen_scripts")
from stage5_common import INDEX_BREAKS, OFFENSES_7, RARE, VOLUME, bands_spanned, implied_cv, load_bg, write  # noqa: E402

bg = load_bg()
n = len(bg)
pop = pd.to_numeric(bg["population_2024"], errors="coerce").fillna(0.0)
land = pd.to_numeric(bg["land_area_sq_mi"], errors="coerce").fillna(0.0)
expo = pd.to_numeric(bg["exposure_proxy_2024"], errors="coerce").fillna(0.0)
evw = pd.to_numeric(bg["index_total_primary_event_weighted"], errors="coerce")

nb = {}
cv = {}
pub = {}
for o in VOLUME:
    pub[o] = pd.to_numeric(bg[f"index_{o}_primary"], errors="coerce").notna()
    nb[o] = bands_spanned(bg[f"index_{o}_primary_ci95_lower"], bg[f"index_{o}_primary_ci95_upper"])
    cv[o] = implied_cv(bg[f"index_{o}_primary_ci95_width"], bg[f"index_{o}_primary"])

rows = []


def evaluate(test_name, threshold, masks):
    """masks: dict offense -> boolean 'would be suppressed'."""
    # the default layer (index_total_primary_event_weighted) is all-or-null over its components,
    # so ANY suppressed volume offense greys the block group on the map's default view.
    any_supp = np.logical_or.reduce([masks[o].to_numpy() for o in VOLUME])
    any_supp = pd.Series(any_supp, index=bg.index) & evw.notna()
    rec = {"test": test_name, "threshold": threshold}
    for o in VOLUME:
        m = masks[o] & pub[o]
        rec[f"suppress_{o}"] = int(m.sum())
        rec[f"suppress_pct_{o}"] = round(100.0 * m.sum() / max(int(pub[o].sum()), 1), 3)
    rec["default_layer_bgs_greyed"] = int(any_supp.sum())
    rec["default_layer_pct_greyed"] = round(100.0 * any_supp.sum() / max(int(evw.notna().sum()), 1), 3)
    rec["population_greyed"] = float(pop[any_supp].sum())
    rec["land_sq_mi_greyed"] = float(land[any_supp].sum())
    rec["land_pct_greyed"] = round(100.0 * land[any_supp].sum() / max(float(land.sum()), 1.0), 3)
    # does it hit the target -- the giant low-exposure polygons?
    giant = land.ge(100.0) & evw.ge(200.0)
    rec["giant_lowexposure_bgs_total"] = int((giant & evw.notna()).sum())
    rec["giant_lowexposure_bgs_greyed"] = int((giant & any_supp).sum())
    rec["target_hit_rate"] = round(
        100.0 * (giant & any_supp).sum() / max(int((giant & evw.notna()).sum()), 1), 2
    )
    rec["collateral_ratio_greyed_per_target"] = round(
        int(any_supp.sum()) / max(int((giant & any_supp).sum()), 1), 1
    )
    rows.append(rec)


for t in (1, 2, 3, 4, 5, 6, 7, 8):
    evaluate("interval spans more than N display bands", t, {o: nb[o].gt(t).fillna(False) for o in VOLUME})
for t in (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
    evaluate("implied posterior CV above", t, {o: cv[o].gt(t).fillna(False) for o in VOLUME})
for f in (50, 100, 250, 500, 613, 1000, 2000):
    m = expo.lt(f)
    evaluate("flat person-exposure floor", f, {o: (m if o != "burglary" else m) for o in VOLUME})
for lo, hi in ((100.0, 200.0), (100.0, 400.0), (50.0, 200.0)):
    m = land.ge(lo) & evw.ge(hi)
    evaluate(f"model-support proxy: land>={lo} sq mi AND index>={hi}", f"{lo}/{hi}", {o: m for o in VOLUME})

write(pd.DataFrame(rows), "s5a_designed_test_impact.csv")

# ---------------------------------------------------------------- exemplar targets
named = {
    "080050071102": "Denver 080050071102 (recorded L: exposure 461, index 875)",
    "530330328003": "Seattle 530330328003 (recorded L: exposure 613)",
    "530330327044": "North Bend WA (ambient blind spot)",
    "080470138023": "Black Hawk CO (lift works)",
    "060379304002": "Angeles NF (class B giant polygon)",
    "320030076003": "Clark NV 761 sq mi",
    "250277042021": "MA Quabbin",
    "481919505003": "Hall Co TX 762 sq mi",
    "482619501001": "Kenedy TX (v19 exemplar)",
}
sel = bg[bg["block_group_geoid"].isin(named)].copy()
rec = []
for _, r in sel.iterrows():
    gid = r["block_group_geoid"]
    row = {
        "case": named[gid],
        "block_group_geoid": gid,
        "population_2024": r["population_2024"],
        "exposure_proxy_2024": r["exposure_proxy_2024"],
        "land_area_sq_mi": r["land_area_sq_mi"],
        "expected_count_total": r["expected_count_total"],
        "index_total_primary_event_weighted": r["index_total_primary_event_weighted"],
    }
    worst_band = 0.0
    worst_cv = 0.0
    for o in VOLUME:
        lo = r[f"index_{o}_primary_ci95_lower"]
        hi = r[f"index_{o}_primary_ci95_upper"]
        b = float(sum(1 for x in INDEX_BREAKS if pd.notna(lo) and pd.notna(hi) and lo < x < hi))
        p = r[f"index_{o}_primary"]
        w = r[f"index_{o}_primary_ci95_width"]
        c = (w / (2 * 1.959963984540054 * p)) if pd.notna(p) and p > 0 and pd.notna(w) else np.nan
        row[f"bands_{o}"] = b
        row[f"cv_{o}"] = c
        row[f"count_{o}"] = r[f"expected_count_{o}"]
        row[f"index_{o}"] = p
        worst_band = max(worst_band, b)
        if pd.notna(c):
            worst_cv = max(worst_cv, c)
    row["worst_bands_crossed"] = worst_band
    row["worst_implied_cv"] = worst_cv
    row["caught_by_bands_gt_3"] = worst_band > 3
    row["caught_by_cv_gt_0.5"] = worst_cv > 0.5
    row["caught_by_exposure_floor_1000"] = bool(r["exposure_proxy_2024"] < 1000)
    rec.append(row)
write(pd.DataFrame(rec).sort_values("case"), "s5a_designed_test_targets.csv")
print("done screen a impact")
