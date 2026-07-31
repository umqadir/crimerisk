#!/usr/bin/env python3
"""Rebuild the 45 Stage-1 pilot packets with a COMPUTED SIGNALS prose section.

The pilot (analysis_scratch/stage1_pilot_result.json) established that cheap models
apply known data artifacts correctly ONLY when the artifact is stated in prose, and
read the identical artifact backwards when it has to be inferred from raw panel rows.
This builder appends four families of computed signal, in words, to every packet:

  1. publication lane vs underlying submission months, with an explicit
     'blank-as-zero risk' note when a full-year publication rests on a 0-2 month
     submission record (plus publication/master-file divergence when they disagree);
  2. offense-composition ratios for low/zero years against the agency's own baseline
     years, with an explicit note when composition is distorted;
  3. an explicit 'uplift cannot rescue a zero' note wherever a partial-year zero
     appears in any lane;
  4. roster presence and lane-provenance facts stated in words.

No gold verdict, and no hint of one, is written into a packet. Every note fires from a
deterministic rule applied to all 45 cases so that the presence of a note is not itself
a tell.

Outputs:
  state/qa/review_tier_trial/batches/batch_NNN.json  ({case_id, class, evidence} x ~9)
  state/qa/review_tier_trial/gold.json               (Opus gold rulings, 45)

Usage: uv run python scripts/review/source_audit/build_review_tier_trial_packets.py
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PILOT = REPO / "analysis_scratch" / "stage1_pilot_result.json"
PANEL = REPO / "state" / "observations" / "agency_year_observations.parquet"
OUT = REPO / "state" / "qa" / "review_tier_trial"
BATCH_DIR = OUT / "batches"
BATCH_SIZE = 9

# results[] indices holding the Opus (gold) arm.  Identified from the grade memo in
# results[11]: e.g. b2-NY0355800 misread_missing, c-SC0020300 token_reporting_flag/high,
# c-MA0146000 "the 12/8 uplift is wrong", a3-WI place 5568275, c-MA0060500 "cannot
# rescue a zero".
GOLD_BATCH_IDX = (1, 3, 5, 6, 7)

LANE_CODE = {
    "reported_cius_publication": "CIUS",
    "reported_state_publication": "SPUB",
    "reported_local_publication": "LPUB",
    "reported_nibrs_rollup": "NIB",
    "reported_srs_summary": "SRS",
}
PUBLICATION_LANES = ("CIUS", "SPUB", "LPUB")
SUBMISSION_LANES = ("SRS", "NIB")
# src/crimerisk/source_provenance.py SOURCE_PRIORITY
LANE_PREF = ["CIUS", "LPUB", "SPUB", "SRS", "NIB"]
TIER_RANK = {"high": 4, "medium": 3, "low": 2, "sparse": 1, "unknown": 0}
TIER_CODE = {"high": "hi", "medium": "md", "low": "lo", "sparse": "sp", "unknown": "un"}

LANE_PHRASE = {
    "CIUS": "the FBI's published CIUS/RCN agency table",
    "SPUB": "a state UCR program's own published table",
    "LPUB": "a city's own published table",
    "SRS": "the FBI Summary Return-A submission record",
    "NIB": "the FBI NIBRS rollup",
}

OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]
OFF_LABEL = {
    "murder": "murder",
    "rape": "rape",
    "robbery": "robbery",
    "aggravated_assault": "aggravated assault",
    "burglary": "burglary",
    "larceny": "larceny",
    "motor_vehicle_theft": "motor vehicle theft",
}
YEARS = list(range(2018, 2025))
FOCAL = 2024

# The two ORI9s the packets synthesise: a trailing 'X' marks the NIBRS-lane variant of
# the parent ORI, so panel lookups resolve to the parent restricted to that lane.
SYNTHETIC_ORI = {
    "FL052990X": ("FL0529900", "reported_nibrs_rollup"),
    "MA014640X": ("MA0146400", "reported_nibrs_rollup"),
}

EST_METHOD_PHRASE = {
    "observed": "carried through as observed",
    "hist_median": "fabricated from the agency's own historical median",
    "trend_log_linear": "fabricated from a log-linear trend fitted to the agency's own past",
    "true_partial_month_ratio": "scaled up by 12/months_reported (partial-year uplift)",
}

FOOTPRINT_PHRASE = {
    "statewide_overlap_layer": "the statewide overlap layer, i.e. its counts are smeared across the whole state rather than landed on a specific place",
    "state_nonmunicipal_remainder": "the state's non-municipal remainder, i.e. the unincorporated territory left over after municipalities are carved out",
}


# ---------------------------------------------------------------- number helpers
def fnum(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.1f}"


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if x >= 0.10:
        return f"{x * 100:.0f}%"
    if x >= 0.01:
        return f"{x * 100:.1f}%"
    return f"{x * 100:.2f}%"


def ratio_phrase(v: float, b: float) -> str:
    if b <= 0:
        return "no baseline" if v == 0 else "baseline 0"
    return pct(v / b)


def art(word: str) -> str:
    """Indefinite article, including the numeric cases (an 8-fold, an 18-fold, an 11-fold)."""
    if word[:1].isdigit():
        return "an" if re.match(r"^(8|11|18)(\D|$)", word) else "a"
    return "an" if word[:1].lower() in "aeiou" else "a"


def months_phrase(m: float, word: str = "months") -> str:
    return f"{fnum(m)} {word[:-1] if m == 1 else word}"


def plural_offenses(total: float) -> str:
    return "Part-I offense" if abs(total) == 1 else "Part-I offenses"


def join(items: list[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------- panel access
def load_panel(oris: set[str]) -> dict:
    need = set()
    for o in oris:
        need.add(SYNTHETIC_ORI[o][0] if o in SYNTHETIC_ORI else o)
    cols = [
        "ori9",
        "year",
        "source_lane",
        "offense",
        "count",
        "months_reported",
        "quality_tier",
        "population",
    ]
    df = pd.read_parquet(PANEL, columns=cols)
    df = df[df["ori9"].isin(need) & df["year"].between(2018, 2024)]

    out: dict[str, dict] = {}
    for ori in sorted(oris):
        parent, lock = SYNTHETIC_ORI.get(ori, (ori, None))
        sub = df[df["ori9"] == parent]
        if lock:
            sub = sub[sub["source_lane"] == lock]
        years: dict[int, dict[str, dict]] = {}
        for (yr, lane), grp in sub.groupby(["year", "source_lane"], sort=True):
            code = LANE_CODE[lane]
            vec = {o: 0.0 for o in OFFENSES}
            present = set()
            for _, row in grp.iterrows():
                vec[row["offense"]] = float(row["count"])
                present.add(row["offense"])
            years.setdefault(int(yr), {})[code] = {
                "total": float(grp["count"].sum()),
                "months": float(grp["months_reported"].iloc[0]),
                "tier": str(grp["quality_tier"].iloc[0]),
                "n_offense_rows": len(present),
                "missing_offense_rows": [o for o in OFFENSES if o not in present],
                "vec": vec,
            }
        out[ori] = years
    return out


def select_lane(year_lanes: dict[str, dict]) -> str | None:
    """Reference lane for a year: most months, then best tier, then repo source priority."""
    if not year_lanes:
        return None
    return sorted(
        year_lanes,
        key=lambda c: (
            -year_lanes[c]["months"],
            -TIER_RANK.get(year_lanes[c]["tier"], 0),
            LANE_PREF.index(c) if c in LANE_PREF else 99,
        ),
    )[0]


# ---------------------------------------------------------------- packet parsing
ORI_HEAD = re.compile(r'^ORI ([A-Z0-9]{9}) "(.*?)" (\S+)')


def parse_blocks(evidence: str) -> list[dict]:
    lines = evidence.split("\n")
    blocks: list[dict] = []
    cur: dict | None = None
    for ln in lines:
        m = ORI_HEAD.match(ln.strip())
        if m and ln.startswith("ORI "):
            cur = {
                "ori9": m.group(1),
                "name": m.group(2),
                "agency_type": m.group(3),
                "head": ln.strip(),
                "roster": None,
                "cw": None,
                "est": None,
                "yr": "",
                "pop": {},
            }
            pm = re.search(r"FBIpop (.+)$", ln.strip())
            if pm:
                for tok in pm.group(1).split():
                    if ":" in tok:
                        yy, val = tok.split(":", 1)
                        try:
                            cur["pop"][2000 + int(yy)] = float(val)
                        except ValueError:
                            pass
            blocks.append(cur)
            continue
        s = ln.strip()
        if cur is None:
            continue
        if s.startswith("YR:"):
            cur["yr"] = s[len("YR:") :].strip()
        elif s.startswith("ROSTER2024:"):
            cur["roster"] = s[len("ROSTER2024:") :].strip()
        elif s.startswith("CW:"):
            cur["cw"] = s[len("CW:") :].strip()
        elif s.startswith("EST2024:"):
            cur["est"] = s[len("EST2024:") :].strip()
    return blocks


# ---------------------------------------------------------------- signal 1
def signal_publication_months(years: dict) -> list[str]:
    out: list[str] = []
    pub_years = [y for y in YEARS if any(c in years.get(y, {}) for c in PUBLICATION_LANES)]
    if not pub_years:
        out.append(
            "No publication lane (CIUS, state publication or local publication) carries a row for "
            "this ORI in any year 2018-2024; every value in the series comes from an FBI submission "
            "lane (Return-A and/or NIBRS)."
        )
        return out

    risk_stated = False
    for y in pub_years:
        lanes = years[y]
        sub_bits, sub_months = [], []
        for code in SUBMISSION_LANES:
            if code in lanes:
                m = lanes[code]["months"]
                sub_months.append(m)
                sub_bits.append(
                    f"{code} at {fnum(m)} reported month{'' if m == 1 else 's'} "
                    f"(total {fnum(lanes[code]['total'])})"
                )
            else:
                sub_bits.append(f"no {code} row at all")
        max_sub = max(sub_months) if sub_months else 0.0

        for code in PUBLICATION_LANES:
            if code not in lanes:
                continue
            p = lanes[code]
            line = (
                f"{y} {code}: {LANE_PHRASE[code]} publishes {fnum(p['total'])} "
                f"{plural_offenses(p['total'])} for {y}, labelled {fnum(p['months'])} months. The "
                f"underlying FBI submission record for {y} is: {join(sub_bits)}."
            )
            if p["months"] >= 12 and max_sub <= 2:
                if not risk_stated:
                    risk_stated = True
                    line += (
                        f" BLANK-AS-ZERO RISK: a full-year (12-month) publication is sitting on a "
                        f"{fnum(max_sub)}-month submission record. Twelve months could not have been "
                        f"measured from {fnum(max_sub)} month{'' if max_sub == 1 else 's'} of "
                        f"submitted data, so the '12 months' "
                        f"label here is a default, not a measurement. Where the published value is 0, "
                        f"such a cell is exactly as consistent with an unfilled blank rendered as a "
                        f"zero as it is with a measured zero; the two are not distinguishable from "
                        f"the row alone."
                    )
                else:
                    line += (
                        f" BLANK-AS-ZERO RISK again: full-year publication on a {fnum(max_sub)}-month "
                        f"submission record."
                    )
                if p["total"] != 0:
                    line += (
                        f" The published value here is {fnum(p['total'])} rather than 0, so this cell "
                        f"is not itself blank, but its full-year claim still rests on a "
                        f"{fnum(max_sub)}-month submission."
                    )
            if p["n_offense_rows"] < 7:
                miss = join([OFF_LABEL[o] for o in p["missing_offense_rows"]])
                line += (
                    f" This {code} row carries only {p['n_offense_rows']} of the 7 Part-I offense "
                    f"rows ({miss} absent). An absent offense row is a missing cell, not a zero, and "
                    f"the row total is therefore a partial-offense total."
                )
            # publication vs current master-file divergence
            if sub_months:
                best = max(
                    (c for c in SUBMISSION_LANES if c in lanes),
                    key=lambda c: (lanes[c]["months"], lanes[c]["total"]),
                )
                st = lanes[best]["total"]
                if max(st, p["total"]) > 0 and abs(st - p["total"]) > 0.2 * max(st, p["total"]):
                    why = (
                        "CIUS is the Return-A master file frozen at publication time while the master "
                        "file keeps being revised afterwards, so a CIUS-vs-submission gap in the same "
                        "year is a vintage or coverage fact about the two renderings of one submission"
                        if code == "CIUS"
                        else f"The {code} lane is collected and published on its own path, so a gap "
                        f"here is a coverage/definition/month-inclusion difference between two "
                        f"publications of the same agency's reporting"
                    )
                    line += (
                        f" LANE DIVERGENCE: the {code} published total ({fnum(p['total'])}) and the "
                        f"same-year {best} submission total ({fnum(st)}) differ by more than 20%. "
                        f"{why} - not two independent measurements of crime."
                    )
            out.append(line)
    return out


# ---------------------------------------------------------------- signal 2
def baseline_frame(years: dict) -> dict:
    sel = {}
    for y in YEARS:
        c = select_lane(years.get(y, {}))
        if c:
            sel[y] = (c, years[y][c])
    full = [d["total"] for _, (c, d) in sel.items() if d["months"] >= 12 and d["total"] > 0]
    if full:
        med = statistics.median(full)
    else:
        med = max([d["total"] for _, (c, d) in sel.items()], default=0.0)
    base_years = [
        y for y, (c, d) in sel.items() if d["months"] >= 12 and med > 0 and d["total"] >= 0.5 * med
    ]
    if not base_years and med > 0:
        base_years = [y for y, (c, d) in sel.items() if d["total"] == med]
    low_years = [
        y
        for y, (c, d) in sel.items()
        if d["total"] == 0 or (med > 0 and d["total"] <= 0.25 * med)
    ]
    return {"sel": sel, "median_full_year": med, "baseline_years": base_years, "low_years": low_years}


def mean_vec(years: dict, sel: dict, ys: list[int]) -> tuple[dict, float]:
    if not ys:
        return {o: 0.0 for o in OFFENSES}, 0.0
    acc = {o: 0.0 for o in OFFENSES}
    for y in ys:
        vec = sel[y][1]["vec"]
        for o in OFFENSES:
            acc[o] += vec[o]
    for o in OFFENSES:
        acc[o] /= len(ys)
    return acc, sum(acc.values())


def compare_composition(label: str, vec: dict, total: float, base: dict, base_tot: float) -> list[str]:
    """Prose comparison of one offense vector against the baseline mean vector."""
    lines: list[str] = []
    R = total / base_tot if base_tot > 0 else float("nan")

    if total == 0:
        lines.append(
            f"{label}: total 0 against a baseline-year average of {fnum(base_tot)} a year. Every one "
            f"of the seven offense streams is 0, so the year carries no internal composition signal "
            f"at all - the zero is uniform across all streams rather than concentrated in any of "
            f"them, and composition can neither support nor undercut it."
        )
        return lines

    parts = []
    for o in OFFENSES:
        parts.append(
            f"{OFF_LABEL[o]} {fnum(vec[o])} vs baseline {fnum(base[o])} ({ratio_phrase(vec[o], base[o])})"
        )
    lines.append(
        f"{label}: total {fnum(total)} against a baseline-year average of {fnum(base_tot)} "
        f"({pct(R)} of baseline). Per offense - {'; '.join(parts)}."
    )

    if base_tot < 20:
        lines.append(
            f"{label}: this agency's baseline-year average is only {fnum(base_tot)} offenses a year, "
            f"which is too small a base for offense composition to carry a usable signal - one or two "
            f"counts move any share or ratio, so composition is not scored here."
        )
        return lines

    shares_now = {o: vec[o] / total for o in OFFENSES}
    shares_base = {o: (base[o] / base_tot if base_tot > 0 else 0.0) for o in OFFENSES}

    scored = [o for o in OFFENSES if base[o] >= 3]
    notes: list[str] = []
    if scored and base_tot > 0:
        rs = {o: vec[o] / base[o] for o in scored}
        mx, mn = max(rs.values()), min(rs.values())
        held = [o for o in scored if rs[o] >= max(2 * R, 0.5)]
        vanished = [o for o in scored if vec[o] == 0 and base[o] >= 8]
        shifted = [
            o
            for o in OFFENSES
            if abs(shares_now[o] - shares_base[o]) >= 0.20 and max(vec[o], base[o]) >= 5
        ]
        distorted = R <= 0.6 and mx >= 1.5 * R and mn <= 0.5 * R and len(scored) >= 3

        if held:
            notes.append(
                "one or more streams hold while the aggregate collapses - "
                + join(
                    [
                        f"{OFF_LABEL[o]} at {pct(rs[o])} of its own baseline"
                        for o in sorted(held, key=lambda o: -rs[o])
                    ]
                )
                + f", against an agency total at only {pct(R)} of baseline"
            )
        if vanished:
            notes.append(
                "one or more streams are exactly 0 rather than merely reduced - "
                + join([f"{OFF_LABEL[o]} 0 against a baseline of {fnum(base[o])}" for o in vanished])
            )
        if shifted:
            notes.append(
                "the mix itself moves - "
                + join(
                    [
                        f"{OFF_LABEL[o]} is {pct(shares_now[o])} of the year's total against "
                        f"{pct(shares_base[o])} across baseline years"
                        for o in shifted
                    ]
                )
            )
        if distorted or held or vanished or shifted:
            if not notes:
                lo = min(scored, key=lambda o: rs[o])
                hi = max(scored, key=lambda o: rs[o])
                notes.append(
                    f"retention runs from {pct(rs[lo])} on {OFF_LABEL[lo]} up to {pct(rs[hi])} on "
                    f"{OFF_LABEL[hi]}, {art(f'{rs[hi] / rs[lo]:.0f}')} {rs[hi] / rs[lo]:.0f}-fold "
                    f"spread across streams that a single change in crime level would not produce"
                    if rs[lo] > 0
                    else f"retention runs from 0% on {OFF_LABEL[lo]} up to {pct(rs[hi])} on "
                    f"{OFF_LABEL[hi]}"
                )
            lines.append(
                f"{label}: COMPOSITION IS DISTORTED relative to this agency's own baseline "
                f"(retention across offenses with a baseline of 3 or more ranges from {pct(mn)} to "
                f"{pct(mx)} while the aggregate sits at {pct(R)}). Specifically: {'; '.join(notes)}. "
                f"A uniform change in the level of crime moves all seven streams together; a spread "
                f"this wide is a change in which offense streams are present in the record, not in "
                f"how much crime the streams describe."
            )
        else:
            lines.append(
                f"{label}: composition is NOT distorted in the one-stream-holds sense - every offense "
                f"stream with a baseline of 3 or more retains between {pct(mn)} and {pct(mx)} of its "
                f"own baseline against an aggregate of {pct(R)}, i.e. the change is spread evenly "
                f"across streams rather than concentrated in which streams are present."
            )
    return lines


def signal_composition(years: dict) -> list[str]:
    bf = baseline_frame(years)
    sel, base_years = bf["sel"], bf["baseline_years"]
    out: list[str] = []
    if not sel:
        return ["This ORI carries no panel row at all for 2018-2024, so it has no own composition history."]

    base_ex = [y for y in base_years if y != FOCAL]
    if not base_ex:
        out.append(
            "This ORI has no full-year baseline year of its own outside 2024 to compare composition "
            "against (no other 2018-2023 year reaches 12 reported months with a positive total), so "
            "offense composition cannot be scored against its own past."
        )
    else:
        base, base_tot = mean_vec(years, sel, base_ex)
        out.append(
            "Baseline years (12 reported months, total at or above half this agency's own median "
            f"full-year total of {fnum(bf['median_full_year'])}): "
            + join([f"{y} via {sel[y][0]}" for y in sorted(base_ex)])
            + f". Baseline-year average composition - "
            + "; ".join([f"{OFF_LABEL[o]} {fnum(base[o])}" for o in OFFENSES])
            + f"; total {fnum(base_tot)} a year."
        )
        # every distinct 2024 lane vector
        seen: list[tuple] = []
        for code in LANE_PREF:
            if code not in years.get(FOCAL, {}):
                continue
            d = years[FOCAL][code]
            key = tuple(d["vec"][o] for o in OFFENSES)
            if key in seen:
                continue
            seen.append(key)
            out.extend(
                compare_composition(f"2024 {code} composition", d["vec"], d["total"], base, base_tot)
            )
        # pre-2024 low/zero years; all-zero ones collapse into one bullet
        def other_lanes(y: int, code: str) -> str:
            bits = [
                f"{c} {fnum(d2['total'])} at {months_phrase(d2['months'])}"
                for c, d2 in sorted(years[y].items(), key=lambda kv: LANE_PREF.index(kv[0]))
                if c != code and d2["total"] != years[y][code]["total"]
            ]
            return (
                f" (other lanes disagree for {y}: {join(bits)}, so this year's reading depends on "
                f"which lane is read)"
                if bits
                else ""
            )

        low_pre = [y for y in sorted(set(bf["low_years"]) - {FOCAL}) if y not in base_ex]
        zero_pre = [y for y in low_pre if sel[y][1]["total"] == 0]
        plain = [y for y in zero_pre if not other_lanes(y, sel[y][0])]
        if plain:
            out.append(
                f"{join([f'{y} ({sel[y][0]})' for y in plain])}: total 0 with all seven offense "
                f"streams at 0, against a baseline-year average of {fnum(base_tot)} a year. A year "
                f"that is uniformly zero across every stream carries no internal composition signal "
                f"at all."
            )
        for y in [y for y in zero_pre if y not in plain]:
            out.append(
                f"{y} ({sel[y][0]}): total 0 with all seven offense streams at 0, against a "
                f"baseline-year average of {fnum(base_tot)} a year. A year that is uniformly zero "
                f"across every stream carries no internal composition signal at all"
                f"{other_lanes(y, sel[y][0])}."
            )
        for y in [y for y in low_pre if y not in zero_pre]:
            code, d = sel[y]
            lines = compare_composition(
                f"{y} {code} composition (a low/zero year for this agency)",
                d["vec"],
                d["total"],
                base,
                base_tot,
            )
            extra = other_lanes(y, code)
            if extra and lines:
                lines[0] = lines[0].rstrip(".") + extra + "."
            out.extend(lines)
        # partial years with a positive total: compare on an annualised basis, since that
        # is the vector the partial-year uplift would actually publish
        for y in YEARS:
            if y not in sel or y in bf["low_years"] or y in base_ex:
                continue
            code, d = sel[y]
            if d["months"] >= 12 or d["months"] <= 0 or d["total"] <= 0:
                continue
            f = 12.0 / d["months"]
            ann = {o: d["vec"][o] * f for o in OFFENSES}
            out.extend(
                compare_composition(
                    f"{y} {code} composition (partial year: {fnum(d['total'])} on "
                    f"{months_phrase(d['months'], 'reported months')}, shown annualised at x{f:.2f})",
                    ann,
                    d["total"] * f,
                    base,
                    base_tot,
                )
            )
    return out


# ---------------------------------------------------------------- signal 3
def signal_partial_zero(years: dict) -> list[str]:
    out: list[str] = []
    hits = []
    for y in YEARS:
        for code, d in sorted(years.get(y, {}).items(), key=lambda kv: LANE_PREF.index(kv[0])):
            if d["total"] == 0 and d["months"] < 12:
                hits.append((y, code, d["months"]))
    if not hits:
        out.append(
            "No partial-year zero appears anywhere in this ORI's 2018-2024 series: there is no "
            "lane-year that reports a total of 0 on fewer than 12 reported months."
        )
        return out
    groups: dict[tuple[str, float], list[int]] = {}
    for y, code, m in hits:
        groups.setdefault((code, m), []).append(y)
    stated = False
    for (code, m), ys in sorted(groups.items(), key=lambda kv: (kv[1][0], kv[0][0])):
        yl = join([str(y) for y in ys])
        if m == 0:
            body = (
                f"total 0 on 0 reported months, i.e. no month of data was submitted in that lane at "
                f"all. There is nothing to scale up."
            )
            tail = (
                " A 0-month zero is arithmetically identical to the agency having submitted nothing."
            )
            mult = "count x 12/months"
        else:
            body = f"total 0 on {fnum(m)} reported month{'' if m == 1 else 's'}."
            tail = (
                f" Treating such a row as a scalable partial year publishes the same zero it started "
                f"from; a zero observed over {fnum(m)} month{'' if m == 1 else 's'} also carries very "
                f"little information about the other {fnum(12 - m)}."
            )
            mult = f"count x 12/{fnum(m)} = x{12 / m:.1f}, and 0 x {12 / m:.1f} is still 0"
        if not stated:
            stated = True
            out.append(
                f"PARTIAL-YEAR ZERO, {yl} {code}: {body} Partial-year uplift is multiplicative "
                f"({mult}), so no uplift, fill ratio or annualisation applied to such a row can "
                f"return anything other than 0 - UPLIFT CANNOT RESCUE A ZERO.{tail}"
            )
        else:
            out.append(
                f"PARTIAL-YEAR ZERO, {yl} {code}: {body} Same point - {mult}; UPLIFT CANNOT RESCUE A "
                f"ZERO."
            )
    return out


# ---------------------------------------------------------------- signal 4
def describe_roster(raw: str | None) -> list[str]:
    if not raw:
        return ["No 2024 roster line is recorded for this ORI in the packet."]
    if raw.startswith("ABSENT"):
        return [
            "Roster presence: this ORI is ABSENT from the FBI CDE 2024 agency roster - the roster is "
            "the FBI's own list of agencies for the 2024 data year, so absence is a statement about "
            "that list and about the agency's 2024 reporting relationship."
        ]
    name = re.match(r'"(.*?)"', raw)
    parts = [p.strip() for p in raw.split("|")]
    typ = parts[1] if len(parts) > 1 else "?"
    cnty = parts[2].replace("counties", "").strip() if len(parts) > 2 else "?"
    nib = ""
    m = re.search(r"nibrs_start (\S+)", raw)
    if m:
        nib = m.group(1)
    line = (
        f"Roster presence: this ORI IS on the FBI CDE 2024 agency roster, named "
        f'"{name.group(1) if name else "?"}", agency type {typ}, county/counties {cnty}.'
    )
    if nib and nib != "<NA>":
        line += f" The roster gives a NIBRS start date of {nib}."
    elif nib:
        line += " The roster records no NIBRS start date, i.e. it is not listed as a NIBRS reporter."
    return [line]


def describe_nibrs_timing(raw: str | None, years: dict) -> list[str]:
    if not raw or raw.startswith("ABSENT"):
        return []
    m = re.search(r"nibrs_start (\d{4})-(\d{2})-(\d{2})", raw)
    if not m:
        return []
    yr, mo = int(m.group(1)), int(m.group(2))
    out = []
    nib_years = sorted(y for y in YEARS if "NIB" in years.get(y, {}))
    if nib_years:
        first = nib_years[0]
        out.append(
            f"NIBRS timing: roster NIBRS start {m.group(0).split()[-1]}; the NIBRS lane first carries a "
            f"panel row for this ORI in {first} at "
            f"{months_phrase(years[first]['NIB']['months'], 'reported months')}, and carries rows "
            f"for {join([str(y) for y in nib_years])}."
        )
    else:
        out.append(
            f"NIBRS timing: the roster gives a NIBRS start of {m.group(0).split()[-1]}, yet no NIBRS "
            f"lane row exists for this ORI in any year 2018-2024."
        )
    if yr in (2023, 2024, 2025):
        avail = 12 - mo + 1
        for y in (yr,):
            if "NIB" in years.get(y, {}):
                claimed = years[y]["NIB"]["months"]
                out.append(
                    f"NIBRS timing arithmetic: a start date of {yr}-{mo:02d} leaves at most {avail} "
                    f"month{'' if avail == 1 else 's'} of {y} inside NIBRS coverage, while the {y} "
                    f"NIBRS row is labelled {months_phrase(claimed)}."
                )
    return out


def describe_lane_provenance(years: dict, ori: str) -> list[str]:
    out: list[str] = []
    lanes24 = years.get(FOCAL, {})
    if not lanes24:
        out.append("Lane provenance 2024: no lane carries a 2024 row for this ORI.")
    else:
        bits = [
            f"{c} total {fnum(d['total'])} at {months_phrase(d['months'])}, tier "
            f"{TIER_CODE.get(d['tier'], d['tier'])}"
            for c, d in sorted(lanes24.items(), key=lambda kv: LANE_PREF.index(kv[0]))
        ]
        out.append(
            f"Lane provenance 2024: {len(lanes24)} lane{'' if len(lanes24) == 1 else 's'} "
            f"carr{'ies' if len(lanes24) == 1 else 'y'} a 2024 row - {join(bits)}."
        )
        pubs = [c for c in lanes24 if c in PUBLICATION_LANES]
        subs = [c for c in lanes24 if c in SUBMISSION_LANES]
        if "CIUS" in pubs and subs:
            out.append(
                "Lane independence: CIUS is not an independent count. The CIUS/RCN 'Offenses Known' "
                "agency table is the FBI's Return-A master file frozen at publication (verified in "
                "this repo at 99.88% cell agreement with the master file for 2023 and 2024), so CIUS "
                "agreeing with SRS for a 2024 agency-year is the same submission counted twice, not "
                "corroboration by a second source."
            )
        if {"SRS", "NIB"} <= set(subs):
            out.append(
                "Lane independence: SRS and NIBRS are not independent either for a NIBRS reporter. "
                "The FBI back-converts NIBRS into Return-A, so 86% of 2024 local agencies appear in "
                "both lanes off one submission; identical SRS and NIBRS values for the same "
                "agency-year normally mean one submission rendered twice."
            )
        if "SPUB" in pubs:
            out.append(
                "Lane independence: the state-publication lane is a genuinely separate collection and "
                "publication path run by the state UCR program, but it is fed by the same reporting "
                "agency, so it can differ from the FBI lanes in coverage, offense definitions and "
                "which months are included."
            )
    return out


def describe_footprint(cw: str | None) -> list[str]:
    if not cw:
        return []
    # the optional ("Human name") group appears in the Stage-1 ad-hoc packets and not in
    # the pilot packets, so it is matched optionally and both renderings stay identical.
    m = re.match(
        r'(\d{2}):(\S+)(?: \("([^"]*)"\))? w=([\d.]+) (\S+) \[([^/]+)/([^\]]+)\]', cw
    )
    if not m:
        return [f"Footprint: {cw}"]
    _, target, tname, w, kind, method, status = m.groups()
    if target.startswith("municipal:"):
        _, geo, fips = target.split(":", 2)
        named = f' ("{tname}")' if tname else ""
        tphrase = f"Census {geo} {fips}{named} (a specific municipal polygon)"
    else:
        tphrase = FOOTPRINT_PHRASE.get(target, target)
        if tname and target not in FOOTPRINT_PHRASE:
            tphrase = f'{target} ("{tname}")'
    return [
        f"Footprint: this ORI's counts are crosswalked at weight {w} to {tphrase}, as {art(kind)} "
        f"'{kind}' assignment produced by the '{method}' method with review status '{status}'."
    ]


def describe_estimate(est: str | None) -> list[str]:
    if not est:
        return []
    m = re.search(r"reported ([\-\d.]+) -> estimated ([\-\d.]+) \[([^\]]+)\]", est)
    if not m:
        return [f"Pipeline 2024 output: {est}"]
    rep, out, methods = m.groups()
    bits = []
    for tok in methods.split("+"):
        mm = re.match(r"(.+?)x(\d+)$", tok)
        if mm:
            meth, n = mm.group(1), int(mm.group(2))
            bits.append(
                f"{n} of the 7 offenses {EST_METHOD_PHRASE.get(meth, meth)}"
            )
        else:
            bits.append(tok)
    return [
        f"Pipeline 2024 output: the pipeline read {fnum(float(rep))} as reported and published "
        f"{fnum(float(out))} - {join(bits)}."
    ]


def describe_population(pop: dict) -> list[str]:
    if not pop:
        return []
    vals = [pop.get(y) for y in YEARS if pop.get(y) is not None]
    if not vals:
        return []
    if all(v == 0 for v in vals):
        return [
            "Population: the FBI population-served field is 0 for every year 2018-2024, so no "
            "per-capita rate can be formed for this ORI from the panel."
        ]
    if len(set(vals)) == 1:
        return [
            f"Population: the FBI population-served field is flat at {fnum(vals[0])} across all of "
            f"2018-2024, i.e. it was never revised over the window."
        ]
    return [
        f"Population: FBI population served moves from {fnum(vals[0])} in 2018 to "
        f"{fnum(pop.get(FOCAL, vals[-1]))} in 2024."
    ]


def signal_unprinted_rows(block: dict, years: dict) -> list[str]:
    """Panel lane-years the packet's own YR line does not print (all-zero NIBRS rows)."""
    shown: set[tuple[int, str]] = set()
    for seg in block["yr"].split(";"):
        m = re.match(r"\s*(\d\d):\s*(.*)", seg)
        if not m:
            continue
        y = 2000 + int(m.group(1))
        for lm in re.finditer(r"\b(CIUS|SPUB|LPUB|NIB|SRS)\b", m.group(2)):
            shown.add((y, lm.group(1)))
    have = {(y, c) for y in YEARS for c in years.get(y, {})}
    extra = sorted(have - shown)
    if not extra:
        return []
    bits = [
        f"{y} {c} total {fnum(years[y][c]['total'])} at {months_phrase(years[y][c]['months'])}, tier "
        f"{TIER_CODE.get(years[y][c]['tier'], years[y][c]['tier'])}"
        for y, c in extra
    ]
    return [
        f"Rows present in the panel but NOT printed on the YR line above: {join(bits)}. These are "
        f"real panel rows for this ORI; a lane-year missing from the YR rendering is not the same "
        f"thing as that lane having no row for the year, and a 12-month lane row totalling 0 is a "
        f"different object from an absent lane."
    ]


def signal_roster_provenance(block: dict, years: dict) -> list[str]:
    out: list[str] = []
    out += describe_roster(block["roster"])
    out += describe_nibrs_timing(block["roster"], years)
    out += describe_lane_provenance(years, block["ori9"])
    out += describe_footprint(block["cw"])
    out += describe_estimate(block["est"])
    out += describe_population(block["pop"])
    return out


# ---------------------------------------------------------------- assembly
def build_signals(evidence: str, panel: dict) -> str:
    blocks = parse_blocks(evidence)
    present = {b["ori9"] for b in blocks}
    # a synthetic '...X' sibling in this packet owns its locked lane; strip it from the parent
    claimed: dict[str, set[str]] = {}
    for syn, (parent, lane) in SYNTHETIC_ORI.items():
        if syn in present and parent in present:
            claimed.setdefault(parent, set()).add(LANE_CODE[lane])

    chunks: list[str] = []
    chunks.append(
        "COMPUTED SIGNALS (computed from the underlying agency-year panel that the rows above are "
        "rendered from; plain facts, no conclusions):"
    )
    for b in blocks:
        years = panel.get(b["ori9"], {})
        drop = claimed.get(b["ori9"], set())
        if drop:
            years = {
                y: {c: d for c, d in lanes.items() if c not in drop} for y, lanes in years.items()
            }
            years = {y: lanes for y, lanes in years.items() if lanes}
        chunks.append("")
        chunks.append(f'{b["ori9"]} "{b["name"]}":')
        for heading, lines in (
            ("Publication lanes vs submission months", signal_publication_months(years)),
            ("Offense composition vs this agency's own baseline", signal_composition(years)),
            ("Partial-year zeros", signal_partial_zero(years)),
            ("Panel rows not shown on the YR line", signal_unprinted_rows(b, years)),
            ("Roster and lane provenance", signal_roster_provenance(b, years)),
        ):
            if not lines:
                continue
            chunks.append(f"  {heading}:")
            for ln in lines:
                chunks.append(f"  - {ln}")
    return "\n".join(chunks)


def insert_signals(evidence: str, signals: str) -> str:
    lines = evidence.split("\n")
    qidx = next((i for i, ln in enumerate(lines) if ln.startswith("QUESTION:")), None)
    if qidx is None:
        return evidence + "\n\n" + signals
    return "\n".join(lines[:qidx] + [signals, ""] + lines[qidx:])


def main() -> None:
    pilot = json.loads(PILOT.read_text())
    cases = pilot["results"][0]["cases"]

    oris: set[str] = set()
    for c in cases:
        oris |= {b["ori9"] for b in parse_blocks(c["evidence"])}
    panel = load_panel(oris)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.json"):
        old.unlink()

    rebuilt = []
    for c in cases:
        sig = build_signals(c["evidence"], panel)
        rebuilt.append(
            {"case_id": c["case_id"], "class": c["class"], "evidence": insert_signals(c["evidence"], sig)}
        )

    n_batches = 0
    for i in range(0, len(rebuilt), BATCH_SIZE):
        n_batches += 1
        path = BATCH_DIR / f"batch_{n_batches:03d}.json"
        path.write_text(json.dumps(rebuilt[i : i + BATCH_SIZE], indent=2) + "\n")

    gold = []
    for idx in GOLD_BATCH_IDX:
        gold.extend(pilot["results"][idx]["rulings"])
    gold.sort(key=lambda r: r["case_id"])
    assert len({g["case_id"] for g in gold}) == len(cases) == 45, "gold/case mismatch"
    (OUT / "gold.json").write_text(json.dumps(gold, indent=2) + "\n")

    print(f"cases={len(rebuilt)} batches={n_batches} gold={len(gold)}")
    print(f"mean packet chars: {sum(len(r['evidence']) for r in rebuilt) / len(rebuilt):.0f}")


if __name__ == "__main__":
    main()
