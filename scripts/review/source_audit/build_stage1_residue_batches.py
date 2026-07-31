#!/usr/bin/env python3
"""Build the Stage-1 RESIDUE-pass batches -- the ~40 cases the cleanup pass left open.

Spec = the RESIDUE list recorded in docs/STATE.md under "Stage 1 adjudication cleanup
COMPLETE", read against state/qa/stage1_adhoc_review/supervisor_escalations.csv.  Four
populations:

  1. twin_batch_never_dispatched   2 ambiguous-twin cases whose cleanup packets were built
                                   (cleanup_batches/batch_026.json) but whose batch never ran,
                                   so no ruling exists.  Their evidence is copied BYTE-IDENTICAL
                                   -- a re-dispatch, not a rebuild, so nothing new is varied.
  2. merged_double_framing        34 same-ORI rows still carrying two live rulings because
                                   pass 1 dispatched the ORI under BOTH the published-zero and
                                   the token-reporter framing and the cleanup builder only
                                   merged the pairs that contradicted on the accept/reject axis.
                                   These 34 (settling_record both_reject_DIFFERENT_repair or
                                   disposition_conflict_FLAG/REJECT) disagree about the REPAIR.
                                   ONE merged packet per ORI, both framings' evidence, token
                                   framing as the base, every framing headline that predicted
                                   its own answer removed, no prior verdict shown.
  3. qc_rerule                     2 cases whose applied cleanup ruling the finalizer QC
                                   re-read disputes (KY0470400 West Point graded LIKELY WRONG,
                                   GA0850200 Milner graded confidence-overstated).  Rebuilt
                                   from the same merged record; the withdrawn ruling is NOT
                                   quoted, and the QC note is NOT quoted either -- it states a
                                   conclusion, which is the same anchoring defect.  What the QC
                                   note relied on is carried as FACTS instead, in the exposure
                                   series below.
  4. no_disposition                1 case (AR063019E) with no zero/token ruling from any arm in
                                   either pass: the 2024 agency-year has no disposition at all.

NEW FACTS BLOCK, uniform across every token/zero-shaped packet in this pass (populations 2-4):
REPORTED-MONTHS EXPOSURE SERIES.  Both QC failures were the same reasoning error -- treating a
year's TOTAL as evidence about the agency's crime level without checking how many months were
actually filed behind that total (West Point took a base rate off an exposure-degraded tail;
Milner's "never positive in seven years" counted five years with zero filed months).  The block
states filed months per lane-year as plain facts and implies no direction.  It is applied to the
WHOLE population rather than to the two flagged cases, so it is a uniform rendering change and
not a result-fitted patch on two answers.

Outputs
  state/qa/stage1_adhoc_review/residue_batches/batch_NNN.json   <=8 self-contained cases
  state/qa/stage1_adhoc_review/residue_batches/manifest.csv     one row per case
  state/qa/stage1_adhoc_review/residue_batches/case_provenance.json
        audit trail: source case_ids, the prior rulings being re-decided and whether each is
        shown (none are), recomputed signals.  Deliberately NOT in the packets.

Usage: uv run python scripts/review/source_audit/build_stage1_residue_batches.py
"""

from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "review" / "source_audit"))

import pandas as pd  # noqa: E402

import build_review_tier_trial_packets as T  # noqa: E402
import build_stage1_cleanup_batches as B  # noqa: E402

ROOT = REPO / "state" / "qa" / "stage1_adhoc_review"
PANEL = REPO / "state" / "observations" / "agency_year_observations.parquet"
CLEANUP = ROOT / "cleanup_batches"
OUT = ROOT / "residue_batches"
BATCH_SIZE = 8
YEARS = list(range(2018, 2025))

DOUBLE_FRAMING_TARGETS = ("both_reject_DIFFERENT_repair", "disposition_conflict_FLAG/REJECT")
TWIN_REDISPATCH = ("a3-CO-durango-vs-fort-lewis-college", "a2-CA04099-highway-patrol-templeton")
RERULE = {
    "KY0470400": ("b2-KY0470400-west-point", "c-KY0470400-west-point",
                  "x-KY0470400-west-point"),
    "GA0850200": ("b2-GA0850200-milner", "c-GA0850200-milner", "x-GA0850200-milner"),
}
NO_DISPOSITION = "c-AR063019E-ar063019e"

ALT_RE = re.compile(r"([\w.-]+)\((published_zero|token_reporter)/(\w+)/(\w+)\):")

# ------------------------------------------------------------------------------ head prose
HEAD_MERGED = (
    "This ORI reached the first review TWICE, through two different screens, and each screen "
    "showed its reviewer only its own half of the record and told it which half mattered. The "
    "two halves are merged below and the screen wording is gone. Rule ONCE, on the whole "
    "record. No prior verdict is shown, deliberately: the two screens returned verdicts that "
    "imply DIFFERENT handling of this same agency-year, so quoting either one would hand you "
    "back the framing this packet exists to remove."
)
HEAD_LUNA = (
    "The published-zero side of this ORI was passed to a cheap reviewer whose output was "
    "DISCARDED as non-informative -- its verdicts turned out to be a deterministic function of "
    "a precomputed provenance label in the packet header and carried no case-specific "
    "judgement -- so no usable prior ruling on that side exists and none is shown."
)
HEAD_RERULE = (
    "This ORI was ruled once already, in an earlier pass, on the merged record you are seeing. "
    "That ruling was WITHDRAWN on a supervisory re-read of the reasoning and is NOT shown, "
    "deliberately: quoting a verdict that has already been set aside would anchor you to it. No "
    "verdict, repair value or confidence from the earlier pass appears anywhere in this packet, "
    "and neither does the re-read's own opinion about what the answer should be. Rule the case "
    "fresh on the record below."
)
HEAD_RERULE_METHOD = (
    "One thing the earlier reasoning got wrong is named here as a METHOD point, not as a "
    "direction: a year's TOTAL and the number of months actually filed behind that total are "
    "different quantities, and a low total in a year with little or no filed exposure is not "
    "evidence about that year's crime level. The REPORTED-MONTHS EXPOSURE SERIES below states "
    "filed months for every lane-year, so any base rate, median, trend or \"never positive\" "
    "claim you make can be checked against the exposure it rests on. This cuts both ways: an "
    "eroding exposure record is equally consistent with a failing reporting relationship and "
    "with a small agency that genuinely had nothing to report."
)
HEAD_NO_DISPOSITION = (
    "This agency-year has NO disposition at all. It was dispatched to the first review swarm "
    "and no arm ever ruled it, it was outside the cleanup pass's population, and no live "
    "published-zero or token-reporter ruling covers its ORI. The one live ruling that touches "
    "this ORI is an IDENTITY ruling ({twin}, {arm} arm: {verdict}) -- it settles that this ORI "
    "is not a duplicate of {other} and therefore keeps its own row in the panel, and it says "
    "nothing about whether the published 2024 figure is real. There is no prior verdict on THAT "
    "question to weigh. Rule it from the record below."
)
HEAD_COMPLETE = (
    "Your answer has to be complete enough to apply without a further pass: the verdict token, "
    "and -- if the published figure cannot stand as a measured full year -- the repair in full, "
    "meaning either an uplift with the number of believable months stated or a missing year with "
    "the basis a fill would rest on. A verdict with no repair does not settle the case."
)
HEAD_STD_QUESTION = (
    "The 2024 agency-year is published as a COMPLETE 12-month year, so it gets no partial-year "
    "uplift and usually no fill: whatever is published stands in the national panel as-is. THE "
    "QUESTION is therefore the same one on every case in this pass: is the published 2024 figure "
    "this agency's real level, or is it a record artefact that must not stand as a measured full "
    "year?"
)


# ------------------------------------------------------------------- exposure series (new)
def exposure_series(oris: set[str]) -> dict[str, dict]:
    """Filed months per (ORI, year, lane), straight from the agency-year panel."""
    df = pd.read_parquet(
        PANEL,
        columns=["ori9", "year", "source_lane", "offense", "count", "months_reported"],
    )
    df = df[df["ori9"].isin(oris) & df["year"].between(min(YEARS), max(YEARS))]
    df = df[df["offense"].isin(T.OFFENSES)]
    g = (df.groupby(["ori9", "year", "source_lane"])
           .agg(total=("count", "sum"), months=("months_reported", "max"),
                months_min=("months_reported", "min"))
           .reset_index())
    g["lane"] = g["source_lane"].map(T.LANE_CODE)
    out: dict[str, dict] = {}
    for ori, sub in g.groupby("ori9"):
        years: dict[int, dict[str, dict]] = {}
        for _, r in sub.iterrows():
            if not isinstance(r["lane"], str):
                continue
            years.setdefault(int(r["year"]), {})[r["lane"]] = dict(
                total=float(r["total"]), months=float(r["months"]),
                split=bool(r["months"] != r["months_min"]))
        out[ori] = years
    return out


def _m(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def exposure_prose(ori: str, name: str, years: dict[int, dict[str, dict]]) -> str:
    L = [f"  {ori} \"{name}\" -- year: total (filed months, by lane):"]
    for y in YEARS:
        lanes = years.get(y) or {}
        if not lanes:
            L.append(f"    {y}: no row in any lane.")
            continue
        parts = []
        for code in ("CIUS", "SPUB", "LPUB", "SRS", "NIB"):
            if code not in lanes:
                continue
            d = lanes[code]
            if code in T.PUBLICATION_LANES:
                parts.append(f"{code} total {T.fnum(d['total'])} "
                             f"(labelled {_m(d['months'])}mo by pipeline assumption, "
                             f"publication lanes carry no period field)")
            else:
                parts.append(f"{code} total {T.fnum(d['total'])} on {_m(d['months'])} "
                             f"filed month{'' if d['months'] == 1 else 's'}")
        L.append(f"    {y}: " + "; ".join(parts))
    # submission-lane month series, the quantity a base rate has to be checked against
    for code in ("SRS", "NIB"):
        seq = [(_m(years[y][code]["months"]) if y in years and code in years[y] else "-")
               for y in YEARS]
        if set(seq) == {"-"}:
            L.append(f"  {code} filed months {min(YEARS)}-{max(YEARS)}: no {code} row in any "
                     f"year.")
        else:
            L.append(f"  {code} filed months {min(YEARS)}-{max(YEARS)}: " + " ".join(seq))
    split = [y for y in YEARS for c in T.SUBMISSION_LANES
             if y in years and c in years[y] and years[y][c]["split"]]
    if split:
        L.append(f"  Months-reported is NOT uniform across the seven offence rows in "
                 f"{', '.join(str(y) for y in sorted(set(split)))} -- the figure shown is the "
                 f"highest month count on any offence row for that lane-year.")
    disagree = [y for y in YEARS
                if y in years and "SRS" in years[y] and "NIB" in years[y]
                and years[y]["SRS"]["months"] != years[y]["NIB"]["months"]]
    if disagree:
        pairs = ", ".join(f"{y} (SRS {_m(years[y]['SRS']['months'])} vs NIB "
                          f"{_m(years[y]['NIB']['months'])})" for y in disagree)
        L.append(f"  The two FBI submission lanes record DIFFERENT filed-month counts for "
                 f"{len(disagree)} year{'' if len(disagree) == 1 else 's'}: {pairs}. The NIBRS "
                 f"batch header and the Return-A months-reported field are separate fields and "
                 f"the FBI does not reconcile them, so 'how much of the year was filed' has two "
                 f"answers on record for those years and this packet does not pick one.")
    prior = [y for y in YEARS if y < max(YEARS)]
    nofile = [y for y in prior
              if not any(c in (years.get(y) or {}) and years[y][c]["months"] > 0
                         for c in T.SUBMISSION_LANES)]
    # Reported per lane rather than as one "best" figure: the two lanes are the two answers on
    # record and collapsing them to the higher one would quietly pick a side.
    tots = ", ".join(
        f"{c} {_m(sum(years[y][c]['months'] for y in prior if c in (years.get(y) or {})))}"
        for c in T.SUBMISSION_LANES
        if any(c in (years.get(y) or {}) for y in prior))
    L.append(f"  Across the {len(prior)} years {min(prior)}-{max(prior)} "
             f"({len(prior) * 12} months of possible exposure), filed months on record by lane: "
             + (tots or "no submission-lane row in any year")
             + f". Years with no filed month in EITHER submission lane: {len(nofile)}"
             + (f" ({', '.join(str(y) for y in nofile)})" if nofile else "") + ".")
    zero_pos = [y for y in prior
                if any(c in (years.get(y) or {}) and years[y][c]["months"] > 0
                       and years[y][c]["total"] > 0 for c in T.SUBMISSION_LANES)]
    L.append(f"  Years {min(prior)}-{max(prior)} pairing a positive filed-month count with a "
             f"positive Part-I total in a submission lane: "
             + (", ".join(str(y) for y in zero_pos) if zero_pos else "none")
             + ". A year with zero filed months carries no information about that year's crime "
               "level in either direction -- it is neither evidence of a genuine zero nor "
               "evidence of a suppressed count.")
    return "\n".join(L)


def exposure_block(evidence: str, exp: dict[str, dict]) -> str:
    """Rendered against the ORIs the packet actually shows, in packet order."""
    names = {}
    for ln in evidence.split("\n"):
        m = re.match(r'^ORI ([A-Z0-9]{9}) "([^"]*)"', ln.strip())
        if m:
            names[m.group(1)] = m.group(2)
    body = [exposure_prose(o, n, exp.get(o) or {}) for o, n in names.items() if o in exp]
    if not body:
        return ""
    return ("REPORTED-MONTHS EXPOSURE SERIES, recomputed for this residue pass straight from the "
            "agency-year panel. Stated explicitly because a year's TOTAL and the number of "
            "months actually filed behind that total are different quantities, and every "
            "own-history claim -- base rate, median, trend, 'never positive' -- is a claim about "
            "the first that is only as good as the second. Plain facts, no direction:\n"
            + "\n".join(body))


# ------------------------------------------------------------------------------ merge
def merged_packet(ori: str, zcid: str | None, tcid: str | None, ev_by_case: dict,
                  head_extra: list[str], sig: dict | None) -> tuple[str, bool]:
    zsec = B.sections(B.pick(ev_by_case[zcid])) if zcid else None
    tsec = B.sections(B.pick(ev_by_case[tcid])) if tcid else None
    base = tsec or zsec
    assert base, ori
    panel = (tsec or {}).get("panel") or (zsec or {}).get("panel")
    signals = (tsec or {}).get("signals") or (zsec or {}).get("signals")
    KEY, ARTIFACTS = STD_HEAD
    head = ["CLEANUP TASK."] + head_extra + [HEAD_STD_QUESTION, HEAD_COMPLETE]
    body = [KEY, ARTIFACTS, " ".join(head)]
    if zsec and tsec:
        facts = ["FACTS, PART 1 -- publication lane, submission record and own-history "
                 "levels:\n" + B.neutralise(zsec["situation"]),
                 "FACTS, PART 2 -- screen arms, own-history baselines, offence composition "
                 "and peer comparison:\n" + B.neutralise(tsec["situation"])]
    else:
        facts = ["FACTS:\n" + B.neutralise(base["situation"])]
    body += facts
    blob = " ".join(facts)
    if zsec or "with 0 total Part-I" in blob or "literal printed 0" in blob:
        body.append(B.HEADER_NOTE)
    if sig:
        body.append(B.signal_prose(sig).replace("for this cleanup pass", "for this residue pass"))
    body += [panel, signals, B.VOCAB_ZERO_TOKEN, B.CLOSE_ZERO_TOKEN]
    return "\n\n".join(x for x in body if x), bool(zsec and tsec)


# ------------------------------------------------------------------------------ main
def main() -> None:
    global STD_HEAD
    ev_by_case, class_by_case = B.load_packets()
    STD_HEAD = B.std_head(ev_by_case)
    esc = list(csv.DictReader((ROOT / "supervisor_escalations.csv").open()))
    rulings = list(csv.DictReader((ROOT / "rulings_full.csv").open()))
    cleanup_packets = {c["case_id"]: c
                       for p in sorted(CLEANUP.glob("batch_*.json"))
                       for c in json.loads(p.read_text())}

    # ------------------------------------------------------------------ populations
    dbl = [r for r in esc
           if r["kind"] == "same_ori_double_framing_outside_cleanup"
           and r["settling_record"].split(":")[0] in DOUBLE_FRAMING_TARGETS]
    assert len(dbl) == 34, len(dbl)
    assert len({r["oris"] for r in dbl}) == 34
    qc = [r for r in esc if r["kind"] == "qc_flagged_for_reread"]
    assert {r["oris"] for r in qc} == set(RERULE), [r["oris"] for r in qc]
    nod = [r for r in esc if r["case_id"] == NO_DISPOSITION]
    assert len(nod) == 1
    twins = [cleanup_packets[c] for c in TWIN_REDISPATCH]
    assert all(t["class"] == "twin_shaped" for t in twins)

    # ------------------------------------------------------------------ signals
    all_oris = {r["oris"] for r in dbl} | set(RERULE) | {nod[0]["oris"]}
    sig_raw = B.structural_signals(sorted(all_oris))
    exp = exposure_series(all_oris)
    # B.signal_prose is written for a published ZERO ("the 2024 zero is the Nth consecutive
    # zero year") and renders nonsense on an agency-year whose 2024 total is positive, so the
    # block is shown only where 2024 really is a zero.  The exposure series and COMPUTED
    # SIGNALS carry the history facts for the rest.  sig_raw is still recorded in provenance.
    total24 = {o: max((d["total"] for d in (y.get(max(YEARS)) or {}).values()), default=0.0)
               for o, y in exp.items()}
    sig = {o: (s if total24.get(o, 0.0) == 0 else None) for o, s in sig_raw.items()}
    print("structural-plausibility block suppressed (2024 total is positive): "
          + (", ".join(o for o in sorted(sig) if sig[o] is None) or "none"))

    cases: list[dict] = []
    prov: dict[str, dict] = {}

    # ------------------------------------------------------- (2) merged double framing
    for r in dbl:
        ori = r["oris"]
        alts = ALT_RE.findall(r["alt_rulings"])
        by_class = {c: (cid, arm, conf) for cid, c, arm, conf in alts}
        assert set(by_class) == {"published_zero", "token_reporter"}, (ori, r["alt_rulings"])
        zcid, zarm, _ = by_class["published_zero"]
        tcid, tarm, _ = by_class["token_reporter"]
        head = [HEAD_MERGED]
        if zarm == "luna":
            head.append(HEAD_LUNA)
        if sig.get(ori):
            # NOT the routing reason for this population -- the double framing is.  Say so
            # accurately: the block is recomputed facts, and some of its arms will not have
            # fired, which is itself evidence.
            head.append(
                "Structural-plausibility signals have been recomputed on the panel for this ORI "
                "and are stated as facts below, including the arms that did NOT fire. They are a "
                "routing device from an earlier screen, they are not why this case is in front of "
                "you, and they imply no direction.")
        ev, both = merged_packet(ori, zcid, tcid, ev_by_case, head, sig.get(ori))
        cid = f"r-{ori}-{tcid.split('-', 2)[-1]}"
        cases.append(dict(
            case_id=cid, **{"class": "token_zero_shaped"},
            residue_reason="merged_double_framing",
            residue_subtype=r["settling_record"].split(":")[0],
            original_class=r["original_class"], state=r["state"], oris=ori,
            mass_hint=r["mass_hint"], source_case_ids=[zcid, tcid], evidence=ev,
            _both_framings=both))
        prov[cid] = dict(
            reason="merged_double_framing", subtype=r["settling_record"].split(":")[0],
            oris=ori, source_case_ids=[zcid, tcid],
            evidence_framings_merged=[zcid, tcid], recomputed_signals=sig_raw.get(ori),
            structural_block_shown=bool(sig.get(ori)),
            prior_rulings=[dict(case_id=c, case_class=k, arm=a, confidence=cf,
                                shown_in_packet=False,
                                discarded_as_rule_echo=(a == "luna" and k == "published_zero"))
                           for k, (c, a, cf) in by_class.items()],
            prior_verdicts_withheld=r["verdict"],
            prior_actions_in_conflict=re.findall(r"-> ([a-z_]+)", r["alt_rulings"]))

    # ------------------------------------------------------------------ (3) qc re-rules
    for ori, (zcid, tcid, prior_cid) in RERULE.items():
        row = next(r for r in qc if r["oris"] == ori)
        head = [HEAD_RERULE, HEAD_RERULE_METHOD]
        if any(g["case_id"] in (zcid,) and g["arm"] == "luna" for g in rulings):
            head.insert(1, HEAD_LUNA)
        head.append(
            "The record below merges every screen that looked at this ORI, so you are seeing "
            "both the publication/submission provenance and the own-history, composition and "
            "peer facts rather than one screen's half.")
        ev, both = merged_packet(ori, zcid, tcid, ev_by_case, head, sig.get(ori))
        cid = f"r-{ori}-{tcid.split('-', 2)[-1]}"
        cases.append(dict(
            case_id=cid, **{"class": "token_zero_shaped"}, residue_reason="qc_rerule",
            residue_subtype="qc_flagged_for_reread", original_class=row["original_class"],
            state=row["state"], oris=ori,
            mass_hint=cleanup_packets[prior_cid]["mass_hint"],
            source_case_ids=[zcid, tcid, prior_cid], evidence=ev, _both_framings=both))
        prov[cid] = dict(
            reason="qc_rerule", subtype="qc_flagged_for_reread", oris=ori,
            source_case_ids=[zcid, tcid, prior_cid], evidence_framings_merged=[zcid, tcid],
            recomputed_signals=sig_raw.get(ori),
            structural_block_shown=bool(sig.get(ori)), prior_rulings=[
                dict(case_id=prior_cid, case_class="merged_cleanup_ruling",
                     shown_in_packet=False, status="withdrawn_on_qc_reread")],
            qc_note_withheld=row["settling_record"])

    # ------------------------------------------------------------------ (4) no disposition
    r = nod[0]
    ori = r["oris"]
    twin = next((g for g in rulings if ori in g["oris"].split(";")
                 and g["case_class"] == "ambiguous_twin" and g["active"] == "1"), None)
    assert twin, ori
    other = "/".join(o for o in twin["oris"].split(";") if o != ori)
    head = [HEAD_NO_DISPOSITION.format(twin=twin["case_id"], arm=twin["arm"],
                                       verdict=twin["verdict"], other=other)]
    ev, both = merged_packet(ori, None, r["case_id"], ev_by_case, head, sig.get(ori))
    cid = f"r-{ori}-{r['case_id'].split('-', 2)[-1]}"
    cases.append(dict(
        case_id=cid, **{"class": "token_zero_shaped"}, residue_reason="no_disposition",
        residue_subtype="unadjudicated_no_zero_token_ruling", original_class=r["original_class"],
        state=r["state"], oris=ori, mass_hint=r["mass_hint"],
        source_case_ids=[r["case_id"]], evidence=ev, _both_framings=both))
    prov[cid] = dict(reason="no_disposition", subtype="unadjudicated_no_zero_token_ruling",
                     oris=ori, source_case_ids=[r["case_id"]],
                     evidence_framings_merged=[r["case_id"]], recomputed_signals=sig_raw.get(ori),
                     structural_block_shown=bool(sig.get(ori)),
                     prior_rulings=[], identity_ruling_cited=dict(
                         case_id=twin["case_id"], arm=twin["arm"], verdict=twin["verdict"],
                         oris=twin["oris"]))

    # ------------------------------------------------ enrich packets missing the signals block
    need = [c for c in cases if "COMPUTED SIGNALS" not in c["evidence"]]
    if need:
        oris: set[str] = set()
        for c in need:
            oris |= B.packet_oris(c["evidence"])
        print(f"recomputing COMPUTED SIGNALS for {len(need)} packets ({len(oris)} ORIs) ...",
              flush=True)
        panel = T.load_panel(oris)
        for c in need:
            sigtxt = T.build_signals(c["evidence"], panel)
            i = c["evidence"].index("\n\nVERDICT VOCABULARY")
            c["evidence"] = c["evidence"][:i] + "\n\n" + sigtxt.strip() + c["evidence"][i:]

    # ------------------------------------------------ exposure series, after the signals block
    for c in cases:
        blk = exposure_block(c["evidence"], exp)
        assert blk, c["case_id"]
        i = c["evidence"].index("\n\nVERDICT VOCABULARY")
        c["evidence"] = c["evidence"][:i] + "\n\n" + blk + c["evidence"][i:]

    # ------------------------------------------------------- (1) twin re-dispatch, verbatim
    for t in twins:
        cases.append(dict(
            case_id=t["case_id"], **{"class": "twin_shaped"},
            residue_reason="twin_batch_never_dispatched",
            residue_subtype=t["cleanup_reason"], original_class=t["original_class"],
            state=t["state"], oris=t["oris"], mass_hint=t["mass_hint"],
            source_case_ids=list(t["source_case_ids"]), evidence=t["evidence"],
            _both_framings=False))
        prov[t["case_id"]] = dict(
            reason="twin_batch_never_dispatched", subtype=t["cleanup_reason"],
            oris=t["oris"], source_case_ids=list(t["source_case_ids"]),
            evidence="copied BYTE-IDENTICAL from cleanup_batches/batch_026.json -- the batch was "
                     "built and never dispatched, so this is a re-dispatch, not a rebuild",
            prior_rulings=[])

    # ------------------------------------------------------------------ invariants
    seen: dict[str, str] = {}
    for c in cases:
        assert "PANEL EVIDENCE:" in c["evidence"], c["case_id"]
        assert "COMPUTED SIGNALS" in c["evidence"], c["case_id"]
        if c["class"] == "token_zero_shaped":
            assert "REPORTED-MONTHS EXPOSURE SERIES" in c["evidence"], c["case_id"]
            assert "VERDICT VOCABULARY" in c["evidence"], c["case_id"]
            # nothing from the withheld prior may leak into the packet
            assert "PRIOR REVIEW" not in c["evidence"], c["case_id"]
            for w in ("QC:", "LIKELY WRONG", "CONFIDENCE OVERSTATED", "misread_missing ->",
                      "treat_missing_repair", "partial_year_uplift", "flag_review"):
                assert w not in c["evidence"], (c["case_id"], w)
        for o in re.split(r"[;|]", c["oris"]):
            if o:
                assert o not in seen, (o, c["case_id"], seen[o])
                seen[o] = c["case_id"]
    assert len(cases) == 34 + 2 + 1 + 2 == 39, len(cases)

    # ------------------------------------------------------------------ write
    order = {"merged_double_framing": 0, "qc_rerule": 1, "no_disposition": 2,
             "twin_batch_never_dispatched": 3}
    cases.sort(key=lambda c: (c["class"] == "twin_shaped", order[c["residue_reason"]],
                              -B._mass(c["mass_hint"]), c["case_id"]))
    batches: list[list[dict]] = []
    for _, grp in B._groupby(cases, key=lambda c: c["class"]):
        for i in range(0, len(grp), BATCH_SIZE):
            batches.append(grp[i:i + BATCH_SIZE])

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("batch_*.json"):
        old.unlink()
    for n, b in enumerate(batches, 1):
        clean = [{k: v for k, v in c.items() if not k.startswith("_")} for c in b]
        (OUT / f"batch_{n:03d}.json").write_text(json.dumps(clean, indent=2) + "\n")

    with (OUT / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "batch_file", "index_in_batch", "case_id", "class", "residue_reason",
            "residue_subtype", "original_class", "state", "oris", "mass_hint",
            "n_source_cases", "source_case_ids", "signal_arms", "both_framings_merged",
            "evidence_chars"])
        w.writeheader()
        for n, b in enumerate(batches, 1):
            for i, c in enumerate(b):
                w.writerow(dict(
                    batch_file=f"batch_{n:03d}.json", index_in_batch=i, case_id=c["case_id"],
                    **{"class": c["class"]}, residue_reason=c["residue_reason"],
                    residue_subtype=c["residue_subtype"], original_class=c["original_class"],
                    state=c["state"], oris=c["oris"], mass_hint=c["mass_hint"],
                    n_source_cases=len(c["source_case_ids"]),
                    source_case_ids=";".join(c["source_case_ids"]),
                    signal_arms="+".join((sig_raw.get(c["oris"]) or {}).get("arms", [])),
                    both_framings_merged=int(c["_both_framings"]),
                    evidence_chars=len(c["evidence"])))

    (OUT / "case_provenance.json").write_text(json.dumps(dict(
        spec="docs/STATE.md 'Stage 1 adjudication cleanup COMPLETE' RESIDUE list, read against "
             "state/qa/stage1_adhoc_review/supervisor_escalations.csv",
        populations=dict(merged_double_framing=len(dbl), qc_rerule=len(RERULE),
                         no_disposition=len(nod), twin_batch_never_dispatched=len(twins)),
        dispatched=dict(cases=len(cases), batches=len(batches), batch_size_max=BATCH_SIZE,
                        by_reason=dict(collections.Counter(c["residue_reason"] for c in cases)),
                        by_class=dict(collections.Counter(c["class"] for c in cases))),
        rendering_changes=dict(
            reported_months_exposure_series=(
                "NEW uniform facts block on all 37 token/zero-shaped packets. Both QC failures "
                "were one reasoning error -- reading a year's total as evidence about crime "
                "level without checking filed months. Applied to the whole population, not the "
                "two flagged cases, so it is a rendering change and not a result-fitted patch."),
            priors_withheld=(
                "No packet in this pass quotes a prior verdict, confidence, repair value or QC "
                "opinion. Every one of these cases is open because a framing predicted its own "
                "answer or because a ruling was withdrawn; quoting the prior would reintroduce "
                "exactly that."),
            twins_byte_identical=(
                "The 2 twin packets are copied byte-identical from cleanup_batches/"
                "batch_026.json -- the batch was built and never dispatched."),
        ),
        cases=prov), indent=1) + "\n")

    # ------------------------------------------------------------------ report
    print(f"cases written : {len(cases)} in {len(batches)} batches "
          f"(one packet per ORI, {len(seen)} ORIs)")
    for k, v in sorted(collections.Counter(
            (c["class"], c["residue_reason"]) for c in cases).items()):
        print(f"  {k[0]:18s} {k[1]:30s} {v}")
    for n, b in enumerate(batches, 1):
        print(f"  batch_{n:03d}.json  {len(b)} cases  {b[0]['class']}")
    print(f"packets merging both framings: {sum(1 for c in cases if c['_both_framings'])}")
    print(f"mean packet chars            : "
          f"{sum(len(c['evidence']) for c in cases) / len(cases):.0f}")


if __name__ == "__main__":
    main()
