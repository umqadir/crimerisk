"""Stage 5 screen (d): suppression taxonomy consistency -- flags vs what actually renders.

Writes to state/qa/stage5_screen/:
  s5d_suppression_taxonomy.csv        -- every suppression state x offense x level: flag, mode, reason,
                                         index null?, rate null?, count kept?, density kept?, aggregates null?
  s5d_suppression_incoherence.csv     -- rows where the flags and the rendered fields disagree
  s5d_reliability_tier_census.csv     -- reliability tier / recommended display geography distribution
  s5d_aggregate_null_propagation.csv  -- all-or-null aggregate rule audit
  s5d_rare_offense_support_audit.json -- the murder/rape tract-support rule, verified end to end
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone/state/qa/stage5_screen/screen_scripts")
from stage5_common import (  # noqa: E402
    BURGLARY_PREMISES_DENOMINATOR_FLOOR,
    MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR,
    NON_RESIDENTIAL_HOUSEHOLD_FLOOR,
    OFFENSES_7,
    OUT,
    PERSON_EXPOSURE_DENOMINATOR_FLOOR,
    PERSON_EXPOSURE_FLOOR_OFFENSES,
    RARE,
    load_bg,
    load_tract,
    write,
)

bg = load_bg()
tr = load_tract()

AGG = [
    "index_total_primary_event_weighted",
    "index_total_equal_offense",
    "index_total_harm",
    "index_total_part1_resident",
    "index_personal_part1_resident",
    "index_property_part1_resident",
]

tax_rows = []
inco_rows = []
for level, f in (("block_group", bg), ("tract", tr)):
    key = "block_group_geoid" if level == "block_group" else "tract_id"
    hh = pd.to_numeric(f["households_total"], errors="coerce").fillna(0.0)
    nonres_expected = hh.lt(NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
    nonres_flag = f["non_residential_flag"].fillna(False).astype(bool)
    special_flag = f["special_use_tract_flag"].fillna(False).astype(bool)
    tid = (f["tract_id"] if "tract_id" in f.columns else f[key]).astype("string").str.zfill(11)
    special_expected = tid.str.slice(5, 11).str.startswith("98", na=False)

    if int((nonres_flag != nonres_expected).sum()):
        inco_rows.append(
            {
                "level": level,
                "offense": "",
                "check": "non_residential_flag != households<10",
                "n": int((nonres_flag != nonres_expected).sum()),
            }
        )
    if int((special_flag != special_expected).sum()):
        inco_rows.append(
            {
                "level": level,
                "offense": "",
                "check": "special_use_tract_flag != tract code 98xx",
                "n": int((special_flag != special_expected).sum()),
            }
        )

    for o in OFFENSES_7:
        den = pd.to_numeric(f[f"primary_denominator_{o}"], errors="coerce").fillna(0.0)
        res_den = pd.to_numeric(f["resident_secondary_denominator"], errors="coerce").fillna(0.0)
        mode = f[f"estimate_mode_{o}"].astype("string")
        reason = f[f"denominator_reason_{o}"].astype("string")
        idx = pd.to_numeric(f[f"index_{o}_primary"], errors="coerce")
        rate = pd.to_numeric(f[f"rate_{o}_primary"], errors="coerce")
        cnt = pd.to_numeric(f[f"expected_count_{o}"], errors="coerce")
        dens = pd.to_numeric(f[f"crime_density_{o}"], errors="coerce")
        pub = f[f"primary_index_publishable_{o}"].fillna(False).astype(bool)
        supp = f[f"primary_index_suppressed_{o}"].fillna(False).astype(bool)
        res_idx = pd.to_numeric(f[f"index_{o}_resident"], errors="coerce")
        res_supp = f[f"index_{o}_resident_suppressed"].fillna(False).astype(bool)

        rare_bg = level == "block_group" and o in RARE

        for m in sorted(mode.dropna().unique().tolist()):
            sel = mode.eq(m)
            n = int(sel.sum())
            if n == 0:
                continue
            tax_rows.append(
                {
                    "level": level,
                    "offense": o,
                    "estimate_mode": m,
                    "n": n,
                    "pct": round(100.0 * n / len(f), 4),
                    "denominator_reason_values": "|".join(
                        sorted(set(reason[sel].dropna().astype(str).unique().tolist()))[:6]
                    ),
                    "publishable_true": int(pub[sel].sum()),
                    "suppressed_true": int(supp[sel].sum()),
                    "index_primary_null": int(idx[sel].isna().sum()),
                    "rate_primary_null": int(rate[sel].isna().sum()),
                    "expected_count_null": int(cnt[sel].isna().sum()),
                    "expected_count_sum": float(cnt[sel].fillna(0.0).sum()),
                    "crime_density_null": int(dens[sel].isna().sum()),
                    "index_resident_null": int(res_idx[sel].isna().sum()),
                    "index_resident_suppressed_true": int(res_supp[sel].sum()),
                    "aggregate_evw_null": int(
                        pd.to_numeric(f.loc[sel, "index_total_primary_event_weighted"], errors="coerce").isna().sum()
                    ),
                    "aggregate_harm_null": int(
                        pd.to_numeric(f.loc[sel, "index_total_harm"], errors="coerce").isna().sum()
                    ),
                    "renders_as": (
                        "NULL by rare-offense tract-support policy (BG murder/rape)"
                        if rare_bg
                        else "grey no-data (#3a4150) when index null; coloured otherwise"
                    ),
                }
            )

        # -- incoherence checks
        def chk(name, mask):
            n = int(pd.Series(mask, index=f.index).fillna(False).sum())
            if n:
                inco_rows.append({"level": level, "offense": o, "check": name, "n": n})

        if not rare_bg:
            chk("publishable TRUE but index_primary NULL", pub & idx.isna())
            chk("publishable FALSE but index_primary NOT NULL", (~pub) & idx.notna())
            chk("suppressed TRUE but index_primary NOT NULL", supp & idx.notna())
            chk("publishable == suppressed (both true)", pub & supp)
            chk("publishable FALSE and suppressed FALSE (silent drop)", (~pub) & (~supp))
        chk("expected_count NULL anywhere", cnt.isna())
        chk("suppressed TRUE but expected_count dropped", supp & cnt.isna())
        chk("suppressed TRUE but crime_density dropped (non-water)",
            supp & dens.isna() & pd.to_numeric(f["land_area_sq_mi"], errors="coerce").fillna(0.0).gt(0))
        chk("denominator_reason missing while suppressed", supp & reason.isna())
        # expected mask recomputation
        exp_special = special_expected | (
            den.lt(BURGLARY_PREMISES_DENOMINATOR_FLOOR) if o == "burglary" else False
        )
        if o in PERSON_EXPOSURE_FLOOR_OFFENSES:
            exp_insuff = den.lt(PERSON_EXPOSURE_DENOMINATOR_FLOOR)
        elif o == "motor_vehicle_theft":
            exp_insuff = den.lt(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR)
        else:
            exp_insuff = pd.Series(False, index=f.index)
        exp_pub = (~nonres_expected) & den.gt(0.0) & (~pd.Series(exp_special, index=f.index).fillna(False)) & (~exp_insuff)
        chk("recomputed publishable != published publishable flag", exp_pub != pub)
        # taxonomy vocabulary coherence: denominator_reason has no non_residential value
        chk(
            "index NULL but denominator_reason still says 'publishable' (non_residential rows)",
            idx.isna() & reason.eq("publishable") & (~pd.Series(rare_bg, index=f.index)) if not rare_bg else pd.Series(False, index=f.index),
        )
        chk(
            "estimate_mode says non_residential but denominator_reason says publishable",
            mode.eq("non_residential") & reason.eq("publishable"),
        )
        chk(
            "resident index NULL but resident_denominator_reason says publishable",
            pd.to_numeric(f[f"index_{o}_resident"], errors="coerce").isna()
            & f[f"resident_denominator_reason_{o}"].astype("string").eq("publishable")
            & (~pd.Series(rare_bg, index=f.index)) if not rare_bg else pd.Series(False, index=f.index),
        )
        # resident arm for MVT uses the VEHICLE denominator, not resident population
        if o == "motor_vehicle_theft":
            chk(
                "resident-arm MVT floor tests vehicle denominator not resident pop (by construction)",
                res_den.lt(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR) & den.ge(MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR) & res_idx.notna(),
            )

    # dead vocabulary: modes/labels declared by the code and the viewer snapshot that never occur
    for dead in ("zero_primary_denominator", "below_legacy_min_denominator", "vehicle_denominator_invalid"):
        got = int(f[f"estimate_mode_{OFFENSES_7[0]}"].astype("string").eq(dead).sum())
        if got == 0 and level == "block_group":
            inco_rows.append(
                {
                    "level": level,
                    "offense": "(all)",
                    "check": f"estimate_mode value '{dead}' is emitted by _estimate_mode but occurs 0 times (dead branch)",
                    "n": 0,
                }
            )

write(pd.DataFrame(tax_rows), "s5d_suppression_taxonomy.csv")
write(pd.DataFrame(inco_rows), "s5d_suppression_incoherence.csv")

# ---------------------------------------------------------------- reliability tiers
rel_rows = []
for level, f in (("block_group", bg), ("tract", tr)):
    for o in OFFENSES_7:
        tier = f[f"reliability_tier_{o}"].astype("string")
        geo = f[f"recommended_display_geography_{o}"].astype("string")
        conf = f[f"confidence_tier_{o}"].astype("string")
        src = f[f"numerator_support_source_{o}"].astype("string")
        idx = pd.to_numeric(f[f"index_{o}_primary"], errors="coerce")
        rec = {"level": level, "offense": o, "n": len(f), "n_index_published": int(idx.notna().sum())}
        for t in ("high", "medium", "low"):
            rec[f"tier_{t}"] = int(tier.eq(t).sum())
        for g in sorted(geo.dropna().unique().tolist()):
            rec[f"display_{g}"] = int(geo.eq(g).sum())
        for c in sorted(conf.dropna().unique().tolist()):
            rec[f"confidence_{c}"] = int(conf.eq(c).sum())
        rec["support_direct_city_incident"] = int(src.eq("direct_city_incident").sum())
        rec["support_model_only"] = int(src.eq("model_only").sum())
        # what the viewer does with tier: reliability_low_code hatching / warning
        rec["published_index_with_tier_low"] = int((idx.notna() & tier.eq("low")).sum())
        rec["pct_published_index_tier_low"] = round(
            100.0 * (idx.notna() & tier.eq("low")).sum() / max(int(idx.notna().sum()), 1), 3
        )
        rel_rows.append(rec)
write(pd.DataFrame(rel_rows), "s5d_reliability_tier_census.csv")

# ---------------------------------------------------------------- aggregate null propagation
agg_rows = []
for level, f in (("block_group", bg), ("tract", tr)):
    comp_null = {}
    for o in OFFENSES_7:
        comp_null[o] = pd.to_numeric(f[f"index_{o}_primary"], errors="coerce").isna()
    res_null = {o: pd.to_numeric(f[f"index_{o}_resident"], errors="coerce").isna() for o in OFFENSES_7}
    for field in AGG:
        v = pd.to_numeric(f[field], errors="coerce")
        if field in ("index_total_primary_event_weighted", "index_total_equal_offense"):
            need = OFFENSES_7 if level == "tract" else tuple(o for o in OFFENSES_7 if o not in RARE)
            any_null = np.logical_or.reduce([comp_null[o].to_numpy() for o in need])
            basis = "all seven primary indices (BG: rare terms substituted from parent tract)"
        elif field == "index_total_harm":
            any_null = None
            basis = "person-exposure publishable rule only (independent of component indices)"
        elif field == "index_personal_part1_resident":
            any_null = np.logical_or.reduce(
                [res_null[o].to_numpy() for o in ("murder", "rape", "robbery", "aggravated_assault")]
            )
            basis = "four personal resident indices"
        elif field == "index_property_part1_resident":
            any_null = np.logical_or.reduce(
                [res_null[o].to_numpy() for o in ("burglary", "larceny", "motor_vehicle_theft")]
            )
            basis = "three property resident indices"
        else:
            any_null = np.logical_or.reduce([res_null[o].to_numpy() for o in OFFENSES_7])
            basis = "all seven resident indices"
        rec = {
            "level": level,
            "field": field,
            "n": len(f),
            "n_null": int(v.isna().sum()),
            "pct_null": round(100.0 * v.isna().sum() / len(f), 4),
            "expected_basis": basis,
        }
        if any_null is not None:
            rec["n_component_null"] = int(any_null.sum())
            rec["n_field_null_but_no_component_null"] = int((v.isna().to_numpy() & ~any_null).sum())
            rec["n_component_null_but_field_published"] = int((~v.isna().to_numpy() & any_null).sum())
        agg_rows.append(rec)
write(pd.DataFrame(agg_rows), "s5d_aggregate_null_propagation.csv")

# ---------------------------------------------------------------- rare-offense tract support
rare_audit = {}
for o in RARE:
    bg_idx = pd.to_numeric(bg[f"index_{o}_primary"], errors="coerce")
    bg_rate = pd.to_numeric(bg[f"rate_{o}_primary"], errors="coerce")
    bg_cnt = pd.to_numeric(bg[f"expected_count_{o}"], errors="coerce")
    tr_idx = pd.to_numeric(tr[f"index_{o}_primary"], errors="coerce")
    ci_cols = [
        f"index_{o}_primary_ci95_lower",
        f"index_{o}_primary_ci95_upper",
        f"index_{o}_primary_ci95_width",
        f"index_{o}_primary_ci95_width_ratio",
        f"rate_{o}_primary_ci95_lower",
        f"rate_{o}_primary_ci95_upper",
    ]
    rare_audit[o] = {
        "bg_index_non_null": int(bg_idx.notna().sum()),
        "bg_rate_non_null": int(bg_rate.notna().sum()),
        "bg_ci_fields_non_null": {c: int(pd.to_numeric(bg[c], errors="coerce").notna().sum()) for c in ci_cols},
        "bg_expected_count_non_null": int(bg_cnt.notna().sum()),
        "bg_expected_count_sum": float(bg_cnt.fillna(0.0).sum()),
        "bg_resident_index_non_null": int(pd.to_numeric(bg[f"index_{o}_resident"], errors="coerce").notna().sum()),
        "tract_index_non_null": int(tr_idx.notna().sum()),
        "tract_expected_count_sum": float(pd.to_numeric(tr[f"expected_count_{o}"], errors="coerce").fillna(0.0).sum()),
        "bg_vs_tract_count_abs_diff": abs(
            float(bg_cnt.fillna(0.0).sum())
            - float(pd.to_numeric(tr[f"expected_count_{o}"], errors="coerce").fillna(0.0).sum())
        ),
        "bg_reliability_tier_retained": int(bg[f"reliability_tier_{o}"].notna().sum()),
    }
rare_audit["_note"] = (
    "RARE_OFFENSE_TRACT_SUPPORT nulls the BG murder/rape index+rate point fields and their CIs "
    "(allocation.py _rare_offense_published_point_fields). The RESIDENT index for those offenses "
    "is checked separately here -- if it is non-null at BG the policy has a hole."
)
(OUT / "s5d_rare_offense_support_audit.json").write_text(json.dumps(rare_audit, indent=2, default=float))
print("wrote s5d_rare_offense_support_audit.json")
print("done screen d")
