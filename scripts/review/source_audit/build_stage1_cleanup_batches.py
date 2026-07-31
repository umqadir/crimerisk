#!/usr/bin/env python3
"""Build the Stage-1 cleanup-pass batches from the ad-hoc swarm registries.

Spec = the supervisor's adjudication (a)-(f) recorded in docs/STATE.md under
"Stage 1 ad-hoc swarm COMPLETE + adjudicated".  Five populations join one cleanup pass:

  1. merged_contradiction   68 cross-class same-ORI contradictions.  ONE merged packet per
                            ORI carrying BOTH framings' full evidence (publication/submission
                            provenance from the published_zero framing, own-history /
                            composition / peer facts from the token_reporter framing), with
                            every framing headline that predicted its own answer removed.
                            Token framing's evidence set is the base, per adjudication (d).
  2. explicit_escalation    62 cases an arm refused to rule; the escalating arm's rationale
                            rides along as CONTEXT.
  3. replicate_conflict     15 cases where two replicate runs of one arm disagreed; both
                            verdicts ride along as CONTEXT.
  4. structural_plausibility_residue
                            full_year_nibrs_backed / genuine_zero_year rulings whose agency
                            shows never-positive history OR demonstrated-capacity-then-zero
                            OR population > 1,500.  Signals are RECOMPUTED here from
                            state/observations/agency_year_observations.parquet -- the Luna
                            rulings are discarded as non-informative (adjudication (c)) and
                            are never shown to the cleanup reviewer.
  5. coverage_gap           1 case dispatched but never ruled (CHP Templeton).

Dedupe: a merged_contradiction packet supersedes the per-framing cases for its ORI, so a
case that also qualifies under (3) or (4) is folded into the merged packet rather than
dispatched twice -- re-dispatching the same ORI under two case_ids is the defect (1) exists
to remove.  Superseded ids are recorded in case_provenance.json.

Outputs
  state/qa/stage1_adhoc_review/cleanup_batches/batch_NNN.json   <=10 self-contained cases
  state/qa/stage1_adhoc_review/cleanup_batches/manifest.csv     one row per case
  state/qa/stage1_adhoc_review/cleanup_batches/case_provenance.json
        audit trail: source case_ids, prior verdicts, superseded ids, recomputed signals.
        Deliberately NOT in the packets -- the merged/residue packets must not carry the
        prior verdicts they exist to re-decide.

Usage: uv run python scripts/review/source_audit/build_stage1_cleanup_batches.py
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

ROOT = REPO / "state" / "qa" / "stage1_adhoc_review"
PANEL = REPO / "state" / "observations" / "agency_year_observations.parquet"
PILOT = REPO / "analysis_scratch" / "stage1_pilot_result.json"
OUT = ROOT / "cleanup_batches"
BATCH_SIZE = 10
POP_FLOOR = 1500.0

PACKET_DIRS = ("batches", "zero_batches", "opus_batches", "gold_repair")
# preference order when the same case_id exists in several dirs: the re-rendered
# (COMPUTED SIGNALS) copies win over the original run-1 packets.
EV_PREF = ("opus_batches", "zero_batches", "gold_repair", "batches")

ORI_HEAD_LINE = re.compile(r'^ORI ([A-Z0-9]{9}) "')

# ---------------------------------------------------------------------------- prose

# The published_zero framing asserted its own answer in the middle of its evidence block.
# The fact stays, the assertion goes.
FRAMING_ASSERTIONS = (
    "A full-year NIBRS submission is a real submission, so a zero here is an agency that "
    "submitted twelve months and recorded no Part-I offence -- a very different fact from "
    "a non-report.",
    "A full-year NIBRS submission is a real submission, so a zero here is an agency that "
    "submitted twelve months and recorded no Part-I offence — a very different fact from "
    "a non-report.",
)
DECISIVE_LABELS = (
    ("DECISIVE PROVENANCE:", "SUBMISSION PROVENANCE:"),
    ("DECISIVE FACT:", "ON RECORD:"),
)

HEADER_NOTE = """NOTE ON THE 12-MONTH BATCH HEADER, stated in both directions because neither is settled: a 12-month NIBRS batch header records twelve monthly submissions, which does rule out the blank-cell-parsed-as-zero ingestion artifact (that artifact needs a 0-2 month submission record). It does NOT by itself separate an agency that submitted twelve months and had no Part-I offence from an agency whose records system emits automatic monthly zero-reports after the agency stopped producing records, nor from an agency that has handed policing to a county sheriff while its ORI keeps filing. Both failure directions cost something real: accepting a false zero silently deletes a live agency's crime from the national panel, and rejecting a true zero fabricates crime for an agency that genuinely had none."""

VOCAB_ZERO_TOKEN = """VERDICT VOCABULARY -- return exactly one of these tokens as `verdict` (the packet-level phrasings map onto them one-to-one):
  genuine_zero_year    the published zero/near-zero is the real 2024 level; carry it as observed
                       (packet phrasings: true_zero, credible_low_crime for a non-zero thin year)
  genuine_low_crime    the published thin non-zero total is the real 2024 level; carry it as observed
  token_reporting_flag the agency-year is a token/partial record published as complete; it must not
                       stand as a measured full year -- say how many months are believable
                       (packet phrasings: token_report_treat_as_partial, partial_year_coverage)
  misread_missing      the published figure is a non-report the pipeline read as data; the agency-year
                       should be MISSING and fillable
                       (packet phrasings: publication_gap_treat_as_missing, structural_zero_treat_as_missing, data_error)
  unclear_escalate     the record genuinely does not separate the hypotheses; say what would resolve it"""

VOCAB_TWIN = """VERDICT VOCABULARY -- return exactly one of these tokens as `verdict` (the packet-level phrasings quoted above map onto them one-to-one):
  same_agency_merge    one physical agency filing under several ORI9s; name the canonical ORI9 and
                       which members collapse into it
                       (packet phrasing: same_agency_dedupe)
  distinct_agencies    genuinely separate agencies; all rows must be kept
                       (packet phrasings: distinct_agencies_keep_both, coincidence_keep_both)
  superseded_ori       one ORI is dead and its footprint is covered by a named survivor
                       (packet phrasing: superseded_ori)
  unclear_escalate     the record does not separate them; say what would resolve it"""

CLOSE_ZERO_TOKEN = """QUESTION: return one verdict token from the vocabulary above, the single fact that decided it, your confidence (high/medium/low), and -- if you rule the published 2024 figure unusable -- what the panel should carry instead (how many months are believable, or that the year should go missing and be filled). Do not fabricate a repair value you cannot defend from this packet; "missing and fillable" is a legitimate answer. If you cannot separate the hypotheses, return unclear_escalate and name the record that would settle it rather than guessing."""

CLOSE_TWIN = """QUESTION: return one verdict token from the vocabulary above, the single fact that decided it, your confidence (high/medium/low), and the resulting identity decision in full (canonical ORI9, which members collapse, which footprint survives). Note that these ORIs are crosswalked to different footprints where stated, so a wrong merge moves mass between places as well as changing the total."""


# ---------------------------------------------------------------------------- helpers
def load_packets() -> dict[str, dict[str, str]]:
    ev: dict[str, dict[str, str]] = collections.defaultdict(dict)
    cls: dict[str, str] = {}
    for d in PACKET_DIRS:
        for p in sorted((ROOT / d).glob("batch_*.json")):
            for c in json.loads(p.read_text()):
                ev[c["case_id"]][d] = c["evidence"]
                cls.setdefault(c["case_id"], c["class"])
    for c in json.loads(PILOT.read_text())["results"][0]["cases"]:
        ev[c["case_id"]].setdefault("pilot", c["evidence"])
        cls.setdefault(c["case_id"], c["class"])
    return ev, cls


def pick(ev_by_dir: dict[str, str]) -> str:
    for d in EV_PREF + ("pilot",):
        if d in ev_by_dir:
            return ev_by_dir[d]
    raise KeyError(sorted(ev_by_dir))


def sections(ev: str) -> dict[str, str]:
    """Split a packet into its named blocks.  Absent blocks come back empty."""
    marks = [
        ("key", "KEY. "),
        ("artifacts", "\nKNOWN INGESTION ARTIFACTS"),
        ("decide", "\nDECIDE:"),
        ("situation", "\nSITUATION:"),
        ("panel", "\nPANEL EVIDENCE:"),
        ("signals", "\nCOMPUTED SIGNALS"),
        ("question", "\nQUESTION:"),
    ]
    pos = []
    for name, m in marks:
        i = ev.find(m)
        pos.append((name, i))
    found = [(n, i) for n, i in pos if i >= 0]
    out = {n: "" for n, _ in marks}
    for k, (n, i) in enumerate(found):
        j = found[k + 1][1] if k + 1 < len(found) else len(ev)
        out[n] = ev[i:j].strip()
    return out


def packet_oris(ev: str) -> set[str]:
    return {
        m.group(1)
        for ln in ev.split("\n")
        for m in [ORI_HEAD_LINE.match(ln.strip())]
        if m
    }


def neutralise(situation: str) -> str:
    s = situation
    for a in FRAMING_ASSERTIONS:
        s = s.replace(" " + a, "").replace(a, "")
    for a, b in DECISIVE_LABELS:
        s = s.replace(a, b)
    s = re.sub(r"^SITUATION:\s*", "", s.strip())
    return re.sub(r"\s{2,}", " ", s).strip()


# ---------------------------------------------------------------------------- signals
def structural_signals(oris: list[str]) -> dict[str, dict]:
    df = pd.read_parquet(
        PANEL,
        columns=["ori9", "year", "source_lane", "offense", "count", "months_reported",
                 "quality_tier", "population"],
    )
    df = df[df["ori9"].isin(set(oris)) & df["year"].between(2018, 2024)]
    g = (df.groupby(["ori9", "year", "source_lane"])
           .agg(total=("count", "sum"), months=("months_reported", "first"),
                pop=("population", "max"))
           .reset_index())
    out: dict[str, dict] = {}
    for ori, sub in g.groupby("ori9"):
        prior = sub[sub["year"] <= 2023]
        y24 = sub[sub["year"] == 2024]
        pos = prior[prior["total"] > 0]
        full_pos = prior[(prior["months"] >= 12) & (prior["total"] > 0)]
        run = 0
        for y in range(2024, 2017, -1):
            yy = sub[sub["year"] == y]
            if len(yy) and float(yy["total"].max()) == 0:
                run += 1
            else:
                break
        pop24 = float(y24["pop"].max()) if len(y24) and pd.notna(y24["pop"].max()) else 0.0
        out[ori] = dict(
            never_positive=bool(len(pos) == 0),
            last_positive_year=(int(pos["year"].max()) if len(pos) else None),
            last_full_year_positive=(int(full_pos["year"].max()) if len(full_pos) else None),
            n_full_year_positive_years=int(full_pos["year"].nunique()),
            max_prior_total=(float(prior["total"].max()) if len(prior) else 0.0),
            last_full_year_positive_total=(
                float(full_pos[full_pos["year"] == full_pos["year"].max()]["total"].max())
                if len(full_pos) else 0.0),
            zero_run_through_2024=run,
            pop2024=pop24,
        )
        s = out[ori]
        s["arms"] = [
            n for n, v in (
                ("never_positive", s["never_positive"]),
                ("capacity_then_zero", s["last_full_year_positive"] is not None),
                ("pop_gt_1500", s["pop2024"] > POP_FLOOR),
            ) if v
        ]
    return out


def signal_prose(s: dict) -> str:
    L = ["STRUCTURAL PLAUSIBILITY, recomputed for this cleanup pass straight from the "
         "agency-year panel (plain facts; the screen is a routing device and implies no "
         "direction):"]
    if s["never_positive"]:
        L.append(f"  - NEVER-POSITIVE HISTORY fired: this ORI has no positive Part-I count in "
                 f"any lane in any year 2018-2023. The 2024 zero is the "
                 f"{s['zero_run_through_2024']}th consecutive all-lane zero year.")
    else:
        L.append(f"  - Never-positive history did NOT fire: the ORI's highest prior-year total "
                 f"from any lane is {T.fnum(s['max_prior_total'])} "
                 f"(last year with any positive count anywhere: {s['last_positive_year']}).")
    if s["last_full_year_positive"] is not None:
        n = s["n_full_year_positive_years"]
        L.append(f"  - DEMONSTRATED-CAPACITY-THEN-ZERO fired: this ORI filed a FULL 12-month "
                 f"year with a positive Part-I total as recently as "
                 f"{s['last_full_year_positive']} "
                 f"({T.fnum(s['last_full_year_positive_total'])} offences), with "
                 f"{n} such full-year positive year{'' if n == 1 else 's'} on record "
                 f"2018-2023. Whatever produced those counts existed; the 2024 record is zero.")
    else:
        L.append("  - Demonstrated-capacity-then-zero did NOT fire: no year 2018-2023 pairs a "
                 "12-month reporting record with a positive Part-I total, so the ORI has never "
                 "demonstrated full-year reporting capacity in the panel window.")
    if s["pop2024"] > POP_FLOOR:
        L.append(f"  - POPULATION FLOOR fired: FBI population served for 2024 is "
                 f"{T.fnum(s['pop2024'])}, above the 1,500 line below which a genuinely "
                 f"crime-free year is unremarkable.")
    else:
        L.append(f"  - Population floor did NOT fire: FBI population served for 2024 is "
                 f"{T.fnum(s['pop2024'])}, at or below 1,500.")
    L.append(f"  - Consecutive all-lane zero years ending in 2024: {s['zero_run_through_2024']}.")
    return "\n".join(L)


# ---------------------------------------------------------------------------- main
def main() -> None:
    ev_by_case, class_by_case = load_packets()
    rulings = list(csv.DictReader((ROOT / "rulings_full.csv").open()))
    esc = list(csv.DictReader((ROOT / "escalations.csv").open()))
    zm = list(csv.DictReader((ROOT / "zero_missing_adjudicated.csv").open()))
    idx = {r["case_id"]: r for r in csv.DictReader((ROOT / "case_index.csv").open())}
    ruling_by = {(r["case_id"], r["arm"]): r for r in rulings}

    # every single-ORI ruling, by ORI and class -- used both to find same-ORI pairs and to
    # pull the OTHER framing's evidence into a packet that only has one framing's case.
    framing: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for r in rulings:
        o = [x for x in r["oris"].split(";") if x]
        if len(o) == 1 and r["case_class"] in ("published_zero", "token_reporter"):
            framing[o[0]][r["case_class"]] = r["case_id"]

    # ---------------------------------------------------------------- populations
    xclass = [r for r in esc if r["kind"] == "cross_class_same_ori_contradiction"]
    contra: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in xclass:
        contra[r["oris"]][r["case_class"]] = r
    assert len(contra) == 68, len(contra)
    assert all(set(v) == {"published_zero", "token_reporter"} for v in contra.values())

    p2 = [r for r in esc if r["kind"] == "explicit_escalate"]
    p3 = [r for r in esc if r["kind"] == "intra_arm_replication_conflict"]
    p5 = [r for r in esc if r["kind"] == "unadjudicated_no_ruling"
          and r["downstream_action"] == "dispatch_again"]
    assert (len(p2), len(p3), len(p5)) == (62, 15, 1), (len(p2), len(p3), len(p5))

    zero_pool = [r for r in zm if r["provenance_kind"] == "full_year_nibrs_backed"
                 and r["verdict"] == "genuine_zero_year"]
    assert len(zero_pool) == 217, len(zero_pool)

    # ---------------------------------------------------------------- signals (4)
    sig = structural_signals(sorted({r["oris"] for r in zero_pool} | set(contra)))
    residue = [r for r in zero_pool if sig[r["oris"]]["arms"]]

    # ---------------------------------------------------------------- one unit per ORI
    # unit = everything the cleanup pass has to say about one ORI.  Building units first,
    # then rendering, is what keeps an ORI from being dispatched twice under two framings --
    # which is the defect adjudication (d) exists to remove.
    Unit = lambda: dict(reasons=set(), members={}, sig=None)          # noqa: E731
    units: dict[str, dict] = collections.defaultdict(Unit)

    for ori, d in contra.items():
        u = units[ori]
        u["reasons"].add("cross_class_contradiction")
        for cl, r in d.items():
            u["members"][r["case_id"]] = dict(row=r, case_class=cl, show_prior=False,
                                              kinds={"cross_class_same_ori_contradiction": r})

    for r in p2 + p3:
        # ambiguous_twin cases span several ORIs and can never join a same-ORI unit; they are
        # rendered on their own further down.
        if class_by_case.get(r["case_id"]) == "ambiguous_twin":
            continue
        u = units[r["oris"]]
        u["reasons"].add("explicit_escalate" if r["kind"] == "explicit_escalate"
                         else "replicate_conflict")
        m = u["members"].setdefault(r["case_id"], dict(row=r, case_class=r["case_class"],
                                                       show_prior=True, kinds={}))
        m["kinds"][r["kind"]] = r
        # a confident ACCEPT/REJECT from the other framing must stay hidden, so an
        # unresolved case sharing an ORI with a contradiction keeps its prior hidden too.
        if "cross_class_contradiction" in u["reasons"]:
            m["show_prior"] = False

    for r in residue:
        u = units[r["oris"]]
        u["reasons"].add("structural_plausibility_residue")
        u["members"].setdefault(r["case_id"], dict(row=r, case_class="published_zero",
                                                   show_prior=False, kinds={}))
        u["sig"] = sig[r["oris"]]
    for ori in contra:
        if ori in sig:
            units[ori]["sig"] = sig[ori]

    gap_units: dict[str, dict] = {}
    for r in p5:
        gap_units[r["case_id"]] = r

    # any case whose prior must stay hidden because a contradiction is in the same unit
    for ori, u in units.items():
        if "cross_class_contradiction" in u["reasons"]:
            for m in u["members"].values():
                m["show_prior"] = False

    # ---------------------------------------------------------------- render
    KEY, ARTIFACTS = std_head(ev_by_case)
    cases: list[dict] = []
    prov: dict[str, dict] = {}

    for ori in sorted(units):
        u = units[ori]
        members = u["members"]
        # ---- evidence: union of every framing on record for this ORI, token framing base
        zcid = framing.get(ori, {}).get("published_zero")
        tcid = framing.get(ori, {}).get("token_reporter")
        zsec = sections(pick(ev_by_case[zcid])) if zcid and zcid in ev_by_case else None
        tsec = sections(pick(ev_by_case[tcid])) if tcid and tcid in ev_by_case else None
        base = tsec or zsec
        assert base, ori
        panel = (tsec or {}).get("panel") or (zsec or {}).get("panel")
        signals = (tsec or {}).get("signals") or (zsec or {}).get("signals")
        merged_framings = sum(x is not None for x in (zsec, tsec)) > 1
        dispatched = sorted(members)
        two_cases = len(members) > 1

        reasons = sorted(u["reasons"])
        rid = ("cross_class_contradiction" in reasons and "merged_contradiction") or (
            "merged_same_ori_pair" if two_cases else reasons[0])
        cid = f"x-{ori}-{(tcid or zcid).split('-', 2)[-1]}" if two_cases else dispatched[0]

        # ---- task head
        head = ["CLEANUP TASK."]
        if two_cases:
            head.append(
                "This ORI reached the first review TWICE, through two different screens, and "
                "each screen showed its reviewer only its own half of the record and told it "
                "which half mattered. The two halves are merged below and the screen wording "
                "is gone. Rule ONCE, on the whole record.")
        elif merged_framings:
            head.append(
                "The record below merges every screen that looked at this ORI, so you are "
                "seeing both the publication/submission provenance and the own-history, "
                "composition and peer facts rather than one screen's half.")
        if "cross_class_contradiction" in reasons:
            head.append(
                "No prior verdict is shown, deliberately: the two screens returned OPPOSITE "
                "confident verdicts on this same ORI, so quoting either one would hand you "
                "back the framing this packet exists to remove.")
        luna_discarded = any(m["case_class"] == "published_zero"
                             and m["row"].get("arm") == "luna" for m in members.values())
        if luna_discarded:
            head.append(
                "The published-zero side of this ORI was first passed to a cheap reviewer whose "
                "output was DISCARDED as non-informative -- its verdicts turned out to be a "
                "deterministic function of a precomputed provenance label in the packet header "
                "and carried no case-specific judgement -- so no usable prior ruling on that "
                "side exists and none is shown.")
        if "structural_plausibility_residue" in reasons:
            head.append(
                "This case reaches you because one or more structural-plausibility signals "
                "fired on the panel; those signals are stated as facts below and imply no "
                "direction.")
        if any(m["show_prior"] for m in members.values()):
            head.append("The first review did not resolve this case. Its stopping point is "
                        "quoted below as CONTEXT -- evidence about the difficulty, not a "
                        "ruling to defer to.")
        head.append(
            "The 2024 agency-year is published as a COMPLETE 12-month year, so it gets no "
            "partial-year uplift and usually no fill: whatever is published stands in the "
            "national panel as-is. THE QUESTION is therefore the same one on every case in this "
            "pass: is the published 2024 figure this agency's real level, or is it a record "
            "artefact that must not stand as a measured full year?")

        body = [KEY, ARTIFACTS, " ".join(head)]

        # ---- prior review context, only for the unresolved cases we are re-deciding
        ctx: list[str] = []
        for mcid in dispatched:
            m = members[mcid]
            if not (m["show_prior"] and m["kinds"]):
                continue
            for kind, r in sorted(m["kinds"].items()):
                if kind == "explicit_escalate":
                    ctx.append(f"  - The {r['arm']} arm REFUSED TO RULE (verdict "
                               f"unclear_escalate, confidence {r['confidence'] or 'n/a'}) on the "
                               f"{r['case_class']} framing of this ORI.")
                elif kind == "intra_arm_replication_conflict":
                    ctx.append(f"  - Two independent replicate runs of the {r['arm']} arm "
                               f"returned DIFFERENT verdicts on the {r['case_class']} framing: "
                               f"{r['replicate_verdicts'].replace(';', ' vs ')} (the run kept in "
                               f"the registry said {r['verdict']}, confidence "
                               f"{r['confidence'] or 'n/a'}).")
                else:
                    continue
                full = ruling_by.get((mcid, r["arm"]), {})
                det = (full.get("detail") or r.get("detail") or "").strip()
                rat = (full.get("rationale") or r.get("rationale") or "").strip()
                if det:
                    ctx.append("    what it said: " + det)
                if rat:
                    ctx.append("    its reasoning: " + rat)
        if ctx:
            body.append("PRIOR REVIEW, quoted as CONTEXT and not as an answer:\n"
                        + "\n".join(ctx)
                        + "\n  Treat every factual claim in the quoted text as a claim to CHECK "
                          "against the panel evidence below, not as established -- the prior "
                          "reviewer saw the same underlying record you are seeing.")

        # ---- facts
        if zsec and tsec:
            facts = ["FACTS, PART 1 -- publication lane, submission record and own-history "
                     "levels:\n" + neutralise(zsec["situation"]),
                     "FACTS, PART 2 -- screen arms, own-history baselines, offence "
                     "composition and peer comparison:\n" + neutralise(tsec["situation"])]
        else:
            facts = ["FACTS:\n" + neutralise(base["situation"])]
        body += facts
        # the header note is about zero-vs-non-report; it is noise on a thin-but-positive year
        blob = " ".join(facts)
        if zsec or "with 0 total Part-I" in blob or "literal printed 0" in blob:
            body.append(HEADER_NOTE)
        if u["sig"]:
            body.append(signal_prose(u["sig"]))
        body += [panel, signals, VOCAB_ZERO_TOKEN, CLOSE_ZERO_TOKEN]

        row = members[dispatched[0]]["row"]
        cases.append(dict(
            case_id=cid, **{"class": "token_zero_shaped"}, cleanup_reason=rid,
            constituent_reasons="+".join(reasons),
            original_class="+".join(sorted({m["case_class"] for m in members.values()})),
            state=row["state"], oris=ori, mass_hint=row["mass_hint"],
            source_case_ids=dispatched,
            evidence="\n\n".join(x for x in body if x),
        ))
        prov[cid] = dict(
            reason=rid, constituent_reasons=reasons, oris=ori, source_case_ids=dispatched,
            evidence_framings_merged=[c for c in (zcid, tcid) if c],
            recomputed_signals=u["sig"],
            prior_rulings=[
                dict(case_id=mcid, case_class=members[mcid]["case_class"],
                     arm=members[mcid]["row"]["arm"], verdict=members[mcid]["row"]["verdict"],
                     confidence=members[mcid]["row"]["confidence"],
                     shown_in_packet=members[mcid]["show_prior"],
                     kinds=sorted(members[mcid]["kinds"]))
                for mcid in dispatched],
            luna_verdict_discarded=[
                dict(case_id=r["case_id"], arm=r["arm"], verdict=r["verdict"])
                for r in residue if r["oris"] == ori and r["arm"] == "luna"],
        )

    # ---- (5) coverage gap: ambiguous-twin case with no ruling at all ----------------
    for cid, r in gap_units.items():
        sec = sections(pick(ev_by_case[cid]))
        twin = class_by_case.get(cid) == "ambiguous_twin"
        body = [KEY, ARTIFACTS,
                "CLEANUP TASK. This case was dispatched to the first review swarm and NO ruling "
                "ever came back for it, so it has never been adjudicated by anyone and there is "
                "no prior verdict to weigh. Rule it from the record below.",
                sec["decide"], sec["situation"], sec["panel"], sec["signals"],
                VOCAB_TWIN if twin else VOCAB_ZERO_TOKEN,
                CLOSE_TWIN if twin else CLOSE_ZERO_TOKEN]
        cases.append(dict(
            case_id=cid, **{"class": "twin_shaped" if twin else "token_zero_shaped"},
            cleanup_reason="coverage_gap", constituent_reasons="coverage_gap",
            original_class=class_by_case.get(cid, r["case_class"]),
            state=r["state"], oris=r["oris"], mass_hint=r["mass_hint"],
            source_case_ids=[cid], evidence="\n\n".join(x for x in body if x)))
        prov[cid] = dict(reason="coverage_gap", constituent_reasons=["coverage_gap"],
                         oris=r["oris"], source_case_ids=[cid], prior_rulings=[])

    # ---- twin-shaped cases that came through the escalation registry ---------------
    # (an ambiguous_twin case has several ORIs, so it is never part of a same-ORI unit)
    for r in p2 + p3:
        if class_by_case.get(r["case_id"]) != "ambiguous_twin":
            continue
        cid = r["case_id"]
        if any(c["case_id"] == cid for c in cases):
            continue
        # remove the unit that the multi-ORI oris string may have created
        cases[:] = [c for c in cases if cid not in c["source_case_ids"]]
        sec = sections(pick(ev_by_case[cid]))
        full = ruling_by.get((cid, r["arm"]), {})
        ctx = []
        if r["kind"] == "explicit_escalate":
            ctx.append(f"  - The {r['arm']} arm REFUSED TO RULE (verdict unclear_escalate, "
                       f"confidence {r['confidence'] or 'n/a'}).")
        else:
            ctx.append(f"  - Two independent replicate runs of the {r['arm']} arm returned "
                       f"DIFFERENT verdicts: {r['replicate_verdicts'].replace(';', ' vs ')} "
                       f"(the run kept in the registry said {r['verdict']}, confidence "
                       f"{r['confidence'] or 'n/a'}).")
        for lbl, key in (("what it said", "detail"), ("its reasoning", "rationale")):
            v = (full.get(key) or r.get(key) or "").strip()
            if v:
                ctx.append(f"    {lbl}: {v}")
        body = [KEY, ARTIFACTS,
                "CLEANUP TASK. This identity case came back unresolved from the first review. "
                "Rule it, or state precisely which external record would settle it and why this "
                "packet cannot.",
                sec["decide"],
                "PRIOR REVIEW, quoted as CONTEXT and not as an answer:\n" + "\n".join(ctx)
                + "\n  Treat every factual claim in the quoted text as a claim to CHECK against "
                  "the panel evidence below, not as established.",
                sec["situation"], sec["panel"], sec["signals"], VOCAB_TWIN, CLOSE_TWIN]
        cases.append(dict(
            case_id=cid, **{"class": "twin_shaped"},
            cleanup_reason=("explicit_escalate" if r["kind"] == "explicit_escalate"
                            else "replicate_conflict"),
            constituent_reasons=("explicit_escalate" if r["kind"] == "explicit_escalate"
                                 else "replicate_conflict"),
            original_class="ambiguous_twin", state=r["state"], oris=r["oris"],
            mass_hint=r["mass_hint"], source_case_ids=[cid],
            evidence="\n\n".join(x for x in body if x)))
        prov[cid] = dict(reason=("explicit_escalate" if r["kind"] == "explicit_escalate"
                                 else "replicate_conflict"),
                         constituent_reasons=[r["kind"]], oris=r["oris"], source_case_ids=[cid],
                         prior_rulings=[dict(case_id=cid, case_class="ambiguous_twin",
                                             arm=r["arm"], verdict=r["verdict"],
                                             confidence=r["confidence"], shown_in_packet=True,
                                             kinds=[r["kind"]])])

    # ---------------------------------------------------------------- enrich stragglers
    need = [c for c in cases if "COMPUTED SIGNALS" not in c["evidence"]]
    if need:
        oris: set[str] = set()
        for c in need:
            oris |= packet_oris(c["evidence"])
        print(f"recomputing COMPUTED SIGNALS for {len(need)} packets ({len(oris)} ORIs) ...",
              flush=True)
        panel = T.load_panel(oris)
        for c in need:
            sigtxt = T.build_signals(c["evidence"], panel)
            i = c["evidence"].index("\n\nVERDICT VOCABULARY")
            c["evidence"] = c["evidence"][:i] + "\n\n" + sigtxt.strip() + c["evidence"][i:]

    # ---------------------------------------------------------------- invariants
    seen_ori: dict[str, str] = {}
    for c in cases:
        assert "COMPUTED SIGNALS" in c["evidence"], c["case_id"]
        assert "PANEL EVIDENCE:" in c["evidence"], c["case_id"]
        for o in re.split(r"[;|]", c["oris"]):
            if o:
                assert o not in seen_ori, (o, c["case_id"], seen_ori[o])
                seen_ori[o] = c["case_id"]
    # a hidden prior verdict must not leak through the quoted-context block
    for c in cases:
        shown = {p["case_id"] for p in prov[c["case_id"]].get("prior_rulings", [])
                 if p.get("shown_in_packet")}
        if not shown:
            assert "PRIOR REVIEW" not in c["evidence"], c["case_id"]
    dispatched_ids = {x for c in cases for x in c["source_case_ids"]}
    for pop, name in ((p2, "explicit_escalate"), (p3, "replicate_conflict"),
                      (residue, "structural_residue"), (p5, "coverage_gap")):
        miss = [r["case_id"] for r in pop if r["case_id"] not in dispatched_ids]
        assert not miss, (name, miss)
    for ori, d in contra.items():
        for r in d.values():
            assert r["case_id"] in dispatched_ids, r["case_id"]

    # ---------------------------------------------------------------- write
    order = {"merged_contradiction": 0, "merged_same_ori_pair": 1,
             "structural_plausibility_residue": 2, "explicit_escalate": 3,
             "explicit_escalate+replicate_conflict": 4, "replicate_conflict": 5,
             "coverage_gap": 6}
    cases.sort(key=lambda c: (c["class"] == "twin_shaped",
                              order.get(c["cleanup_reason"], 9),
                              -_mass(c["mass_hint"]), c["case_id"]))
    batches: list[list[dict]] = []
    for _, grp in _groupby(cases, key=lambda c: c["class"]):
        for i in range(0, len(grp), BATCH_SIZE):
            batches.append(grp[i:i + BATCH_SIZE])

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("batch_*.json"):
        old.unlink()
    for n, b in enumerate(batches, 1):
        (OUT / f"batch_{n:03d}.json").write_text(json.dumps(b, indent=2) + "\n")

    with (OUT / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch_file", "index_in_batch", "case_id", "class",
                                          "cleanup_reason", "constituent_reasons",
                                          "original_class", "state", "oris", "mass_hint",
                                          "n_source_cases", "source_case_ids", "signal_arms",
                                          "evidence_chars"])
        w.writeheader()
        for n, b in enumerate(batches, 1):
            for i, c in enumerate(b):
                w.writerow(dict(
                    batch_file=f"batch_{n:03d}.json", index_in_batch=i, case_id=c["case_id"],
                    **{"class": c["class"]}, cleanup_reason=c["cleanup_reason"],
                    constituent_reasons=c["constituent_reasons"],
                    original_class=c["original_class"], state=c["state"], oris=c["oris"],
                    mass_hint=c["mass_hint"], n_source_cases=len(c["source_case_ids"]),
                    source_case_ids=";".join(c["source_case_ids"]),
                    signal_arms="+".join(sig.get(c["oris"], {}).get("arms", [])),
                    evidence_chars=len(c["evidence"])))

    (OUT / "case_provenance.json").write_text(json.dumps(dict(
        spec="docs/STATE.md 'Stage 1 ad-hoc swarm COMPLETE + adjudicated' items (a)-(f)",
        built_from=dict(rulings_full=len(rulings), escalations=len(esc),
                        zero_missing_adjudicated=len(zm), case_index=len(idx)),
        spec_populations=dict(cross_class_contradiction_oris=len(contra),
                              explicit_escalations=len(p2), replicate_conflicts=len(p3),
                              structural_pool=len(zero_pool), structural_fired=len(residue),
                              structural_retained_no_review=len(zero_pool) - len(residue),
                              coverage_gap=len(p5)),
        dispatched=dict(cases=len(cases), batches=len(batches),
                        by_reason=dict(collections.Counter(c["cleanup_reason"] for c in cases)),
                        by_class=dict(collections.Counter(c["class"] for c in cases))),
        population_floor=POP_FLOOR,
        # the screen's other half: full-year-backed zeros no structural signal fires on are
        # RETAINED without review.  Recorded because retaining a published zero is a decision.
        structural_retained_without_review=[
            dict(case_id=r["case_id"], oris=r["oris"], state=r["state"],
                 discarded_arm=r["arm"], mass_hint=r["mass_hint"],
                 recomputed_signals=sig[r["oris"]])
            for r in zero_pool if not sig[r["oris"]]["arms"]],
        cases=prov), indent=1) + "\n")

    # ---------------------------------------------------------------- report
    print(f"cases written : {len(cases)} in {len(batches)} batches "
          f"(one packet per ORI, {len(seen_ori)} ORIs)")
    for k, v in sorted(collections.Counter(
            (c["class"], c["cleanup_reason"]) for c in cases).items()):
        print(f"  {k[0]:18s} {k[1]:38s} {v}")
    print(f"source cases folded in       : {len(dispatched_ids)}")
    print(f"structural pool {len(zero_pool)} -> fired {len(residue)} "
          f"(retained without review {len(zero_pool) - len(residue)})")
    print(f"packets merging both framings: "
          f"{sum(1 for c in cases if 'FACTS, PART 1' in c['evidence'])}")
    print(f"mean packet chars            : "
          f"{sum(len(c['evidence']) for c in cases) / len(cases):.0f}")


def std_head(ev_by_case) -> tuple[str, str]:
    """The long-form KEY + ingestion-artifact blocks, from a re-rendered packet."""
    sec = sections(pick(ev_by_case["b2-TX1940300-bogata"]))
    assert sec["key"].startswith("KEY. ")
    assert sec["artifacts"].startswith("KNOWN INGESTION ARTIFACTS")
    return sec["key"], sec["artifacts"]


def _mass(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return -1.0


def _groupby(items, key):
    out: dict = collections.OrderedDict()
    for it in items:
        out.setdefault(key(it), []).append(it)
    return list(out.items())


if __name__ == "__main__":
    main()
