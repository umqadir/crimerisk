#!/usr/bin/env python3
"""Apply the Stage-1 FINAL rulings: the last two twin cases and the supervisor's wholesale ruling.

Fourth and last pass in the Stage-1 ad-hoc review chain:

    collate_stage1_adhoc_review.py     pass 1  1,089-case class-routed swarm  -> the base ledger
    apply_stage1_cleanup_rulings.py    pass 2  246 merged-packet re-rulings   -> lands on pass 1
    apply_stage1_residue_rulings.py    pass 3  37 residue re-rulings + the    -> lands on pass 2
                                               mechanical double-framing dedupe
    apply_stage1_final_rulings.py      pass 4  the 2 never-dispatched twin    -> lands on pass 3
                                               cases + supervisor rulings

What pass 4 settles
-------------------
1. THE LAST TWO TWIN CASES (`final_rulings.json`, adjudicated directly from
   residue_batches/batch_006.json -- the batch was built and never dispatched, so pass 3 recorded
   both cases as `residue_case_unruled`). Both are twin-identity questions, so each supersedes the
   pass-1 identity ruling for the same case / the same ORI set, and neither touches a zero/token
   ruling for the same ORI: that is a different question about the same agency.

2. THE SUPERVISOR'S DISPOSITIONS on the explicit escalations. Recorded in docs/STATE.md
   ("Stage 1 adjudication CLOSED"): the 13 `cleanup_explicit_escalate` rows at mass <= 1.0 get
   `downstream_action=retain_current_flag_review` -- retain the current published value and carry a
   review flag, because the state-UCR / Clery lookups that would settle them are not worth blocking
   the release for at that mass. The population is DERIVED from the pass-3 escalation queue
   (kind + mass threshold), never listed by hand, so a re-run reproduces it and a changed queue
   changes the population rather than silently disagreeing with it. The reviewer's own verdict
   (`unclear_escalate`) is left standing; only the disposition is supervised.

   Four remaining above-mass cases are dispositioned by stable case ID from that same queue:
   DCVA00000 retains 291 unuplifted pending FBI ORI master/facility mapping; MN0110700,
   SC0231900 and IL0451900 retain their current values with flag_review. The five replicate-conflict
   pointers are untouched.

3. ORI-SCOPE NARROWING of multi-ORI zero/token rulings whose members have since been ruled
   individually. Pass 2 recorded the rule and refused to act on it, because wholesale supersession
   of a bundle would drop the members that have no per-ORI ruling: "the cleanup ruling for the
   shared ORI is authoritative for that ORI only". Pass 4 applies exactly that, once the twin
   ruling licenses it: `a2-CA04099` rules the CHP area offices DISTINCT agencies, so the unit of a
   zero/token disposition is the individual ORI, and a bundle of three offices from three different
   ORI7 blocks is a submitter-family observation rather than an identity claim. Each bundled ORI
   that carries its own single-ORI live zero/token ruling leaves the bundle; the bundle keeps the
   rest. This resolves the one `partial_ori_overlap_not_superseded` escalation and the one residual
   cross-class contradiction (CA0479945), both of which trace to the same bundle.
   Fail-closed: if narrowing would empty a bundle, the bundle is left intact and escalated.

Re-runnability
--------------
Pass 4 recomputes `active` from the pass-2 and pass-3 columns rather than trusting its own writes,
drops its own rows (`source == "final"`) and its own columns on read, and rebuilds the three
per-class registries from the live ledger the way pass 3 does. The pass-3 escalation queue is
snapshotted to supervisor_escalations_pre_final.csv the first time so a re-run reads the same input
rather than its own filtered output. Pass 3 in turn strips pass-4 rows and columns, so the chain is
re-runnable in order from any point.

Inputs   state/qa/stage1_adhoc_review/{rulings_full,zero_missing_adjudicated,
         token_reporters_adjudicated,twins_adjudicated,supervisor_escalations}.csv
         state/qa/stage1_adhoc_review/final_rulings.json  (the two twin rulings, durable)
         state/qa/stage1_adhoc_review/residue_batches/{manifest.csv,batch_*.json}
Outputs  the four registries regenerated, supervisor_escalations.csv regenerated,
         supervisor_escalations_pre_final.csv (input snapshot), final_apply_summary.json

Usage: uv run python scripts/review/source_audit/apply_stage1_final_rulings.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_stage1_cleanup_rulings as C  # noqa: E402  pass-2 helpers, reused not recopied
import apply_stage1_residue_rulings as R  # noqa: E402  pass-3 helpers, reused not recopied

ROOT = C.ROOT
OUT = C.OUT
SNAPSHOT = os.path.join(OUT, "final_rulings.json")
ESC_LIVE = os.path.join(OUT, "supervisor_escalations.csv")
ESC_PRE = os.path.join(OUT, "supervisor_escalations_pre_final.csv")

csv.field_size_limit(1 << 24)

# Columns owned by this pass; stripped on read so a re-run cannot double-apply. `oris_scope` is the
# EFFECTIVE ORI scope of a narrowed ruling and is deliberately a separate column: the ledger's own
# `oris` stays exactly what the reviewer ruled on, because that is the audit record, and narrowing is
# pass-4 bookkeeping recomputed from the live ledger on every run. Mutating `oris` in place would
# make the second run read its own output and find nothing to narrow.
MY_COLS = ("final_pass", "final_reason", "oris_scope")

# the supervisor's wholesale ruling, as a predicate over the pass-3 queue rather than a case list
WHOLESALE_KIND = "cleanup_explicit_escalate"
WHOLESALE_MAX_MASS = 1.0
WHOLESALE_ACTION = "retain_current_flag_review"
WHOLESALE_NOTE = (
    "supervisor wholesale ruling (docs/STATE.md, Stage 1 adjudication CLOSED): explicit escalation "
    "at mass <= 1.0 -> retain the current published value and carry flag_review; the state-UCR / "
    "Clery lookup that would settle it is not worth blocking the release at this mass. The "
    "reviewer's unclear_escalate verdict stands; only the disposition is supervised."
)
ABOVE_MASS_ACTION = "retain_current_flag_review"
ABOVE_MASS_DISPOSITIONS = {
    "c-DCVA00000-united-states-department": (
        "Supervisor disposition: retain 291 unuplifted pending FBI ORI master/facility mapping; "
        "carry flag_review. The reviewer's unclear_escalate verdict remains visible; only the "
        "disposition is supervised."
    ),
    "x-MN0110700-pillager": (
        "Supervisor disposition: retain the current published value and carry flag_review. The "
        "reviewer's unclear_escalate verdict remains visible; only the disposition is supervised."
    ),
    "b2-SC0231900-bob-jones-university": (
        "Supervisor disposition: retain the current published value and carry flag_review. The "
        "reviewer's unclear_escalate verdict remains visible; only the disposition is supervised."
    ),
    "b2-IL0451900-waubonsee-comm-college": (
        "Supervisor disposition: retain the current published value and carry flag_review. The "
        "reviewer's unclear_escalate verdict remains visible; only the disposition is supervised."
    ),
}

def stance(verdict):
    """Which side of "does the published value stand" a zero/token verdict takes."""
    return {"genuine_zero_year": "ACCEPT", "genuine_low_crime": "ACCEPT",
            "misread_missing": "REJECT", "token_reporting_flag": "REJECT"}.get(verdict, "OTHER")


BUNDLE_NARROW_NOTE = (
    "ORI scope narrowed: this multi-ORI zero/token ruling kept authority only over the ORIs with no "
    "single-ORI live ruling of their own, per the rule pass 2 recorded and declined to apply "
    "(\"the cleanup ruling for the shared ORI is authoritative for that ORI only\"). Licensed by the "
    "a2-CA04099 identity ruling: the CHP area offices are distinct agencies, so the unit of a "
    "zero/token disposition is the individual ORI."
)


def scope_of(row):
    """The ORI scope a live ruling actually speaks for: the narrowed scope where one was set."""
    return C.split_ids(row.get("oris_scope") or row.get("oris") or "")


def active_from_prior_passes(row):
    """Re-derive `active` from the pass-2 and pass-3 columns only -- never from pass-4's own write."""
    if row.get("cleanup_pass") == "superseded_by_cleanup":
        return 0
    if row.get("residue_pass") in ("superseded_by_residue", "dedupe_dropped"):
        return 0
    return 1


def route_final(ruling):
    """The two final rulings are both identity rulings, so both land in the twins registry."""
    verdict = C.norm_verdict(ruling["verdict"])
    if verdict not in ("same_agency_merge", "superseded_ori", "distinct_agencies",
                       "unclear_escalate"):
        sys.exit(f"final ruling {ruling['case_id']} carries a non-twin verdict {verdict!r}; "
                 "pass 4 only knows how to route identity rulings")
    return "twins"


def read_pre_final_queue():
    """The pass-3 escalation queue, snapshotted so a pass-4 re-run reads its input not its output."""
    live_cols = []
    if os.path.isfile(ESC_LIVE):
        with open(ESC_LIVE) as f:
            live_cols = next(csv.reader(f), [])
    if not live_cols:
        sys.exit(f"{ESC_LIVE} is missing; re-run apply_stage1_residue_rulings.py first")
    if "final_status" not in live_cols:
        rows, cols = C.read_registry(ESC_LIVE)
        C.write_csv(ESC_PRE, cols, rows)        # fresh pass-3 output -> refresh the snapshot
        return rows, cols
    if not os.path.isfile(ESC_PRE):
        sys.exit(f"{ESC_LIVE} is already pass-4 output and {ESC_PRE} is missing; "
                 f"re-run apply_stage1_residue_rulings.py first")
    return C.read_registry(ESC_PRE)


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    del args

    if not os.path.isfile(SNAPSHOT):
        sys.exit(f"no final rulings: {SNAPSHOT} does not exist")
    snap = json.load(open(SNAPSHOT))
    manifest = {m["case_id"]: m
                for m in csv.DictReader(open(os.path.join(OUT, "residue_batches", "manifest.csv")))}
    packets = R.load_packets()

    final = {}
    for x in snap["rulings"]:
        cid = x["case_id"]
        if cid in final:
            sys.exit(f"final_rulings.json carries two rulings for {cid}")
        if cid not in manifest:
            sys.exit(f"final ruling {cid} is not in the residue manifest")
        final[cid] = dict(x, verdict_norm=C.norm_verdict(x["verdict"]))

    # ---------------- supersession keys, from the final rulings (identity question only)
    owner_by_ori, owner_by_id = {}, {}
    for cid, keep in final.items():
        m = manifest[cid]
        q = R.question(m["class"])
        if q != "twin":
            sys.exit(f"final ruling {cid} is not a twin-question case (class {m['class']!r})")
        for o in C.split_ids(m["oris"]):
            owner_by_ori[o] = cid
        for i in C.split_ids(m["source_case_ids"]) + [cid]:
            owner_by_id[i] = cid

    def supersession(row):
        if R.question(row["case_class"]) != "twin":
            return None, ""
        if row["case_id"] in owner_by_id:
            return owner_by_id[row["case_id"]], "same_case_id"
        oris = C.split_ids(row["oris"])
        hit = [o for o in oris if o in owner_by_ori]
        if not hit:
            return None, ""
        if len(hit) < len(oris):
            return None, "partial_ori_overlap"
        return owner_by_ori[hit[0]], "same_ori"

    # ---------------- ledger: drop pass-4 rows, re-derive active from the earlier passes
    full_rows, full_cols = C.read_registry(os.path.join(OUT, "rulings_full.csv"), MY_COLS)
    prior = []
    for r in full_rows:
        if r.get("source") == "final":
            continue                                       # this script's own prior output
        r["active"] = active_from_prior_passes(r)
        r["final_pass"] = "retained_pre_final"
        r["final_reason"] = ""
        r["oris_scope"] = ""
        prior.append(r)

    final_partial_overlap = []
    for r in prior:
        if not r["active"]:
            continue
        owner, why = supersession(r)
        if owner:
            r["active"] = 0
            r["superseded_by"] = owner
            r["final_pass"] = "superseded_by_final"
        elif why == "partial_ori_overlap":
            final_partial_overlap.append(r)

    # ---------------- final ruling rows
    final_rows = []
    for cid, keep in sorted(final.items()):
        m = manifest[cid]
        oris = C.split_ids(m["oris"])
        superseded = sorted({r["case_id"] for r in prior
                             if r["final_pass"] == "superseded_by_final"
                             and r["superseded_by"] == cid})
        final_rows.append(dict(
            case_id=cid, case_class=m["class"], subclass=cid.split("-")[0], state=m["state"],
            n_oris=len(oris), oris=";".join(oris), mass_hint=m["mass_hint"],
            arm=keep.get("arm", "opus"), verdict=keep["verdict_norm"],
            confidence=keep.get("confidence", ""),
            source="final", source_batch=m["batch_file"], agent=keep.get("agent", ""),
            n_replicates=1, replicate_conflict=0, replicate_verdicts="",
            verdict_raw=keep["verdict"], detail=keep.get("detail", ""),
            rationale=keep.get("rationale", ""),
            packet_file="residue_batches/" + m["batch_file"],
            cleanup_pass="", active=1, superseded_by="",
            cleanup_reason=m["residue_subtype"], original_class=m["original_class"],
            residue_pass="", residue_reason=m["residue_reason"],
            merged_from=";".join(superseded),
            final_pass="final_ruling", oris_scope="",
            final_reason="batch_006_dispatch_hole_adjudicated_directly"))

    ledger = prior + final_rows

    # ---------------- ORI-scope narrowing of multi-ORI zero/token bundles
    # A bundled ruling keeps authority only over the ORIs that have no single-ORI live ruling of
    # their own. Applied over the live ledger, so it is a function of the ledger, not a case list.
    zt_live = [r for r in ledger
               if r.get("active") and R.question(r["case_class"]) == "zero_token"]
    zt_index = collections.defaultdict(list)
    for r in zt_live:
        for o in scope_of(r):
            zt_index[o].append(r)
    solo_owned = {o for o, rows in zt_index.items() if any(len(scope_of(x)) == 1 for x in rows)}
    narrowed, narrow_refused, stance_conflicts = [], [], {}
    for r in zt_live:
        oris = scope_of(r)
        if len(oris) < 2:
            continue
        keep_oris = [o for o in oris if o not in solo_owned]
        if len(keep_oris) == len(oris):
            continue
        dropped = [o for o in oris if o in solo_owned]
        released_to = [sorted({x["case_id"] for x in zt_index[o] if len(scope_of(x)) == 1})
                       for o in dropped]
        if not keep_oris:
            narrow_refused.append(dict(case_id=r["case_id"], oris=oris, released_to=released_to,
                                       why="every member carries its own single-ORI ruling; "
                                           "narrowing would empty the bundle"))
            continue
        r["oris_scope"] = ";".join(keep_oris)
        r["final_pass"] = "ori_scope_narrowed"
        r["final_reason"] = (f"narrowed_from:{';'.join(oris)}|released:{';'.join(dropped)} :: "
                             + BUNDLE_NARROW_NOTE)
        # Narrowing must not silently retire a DISAGREEMENT. Where the bundle and the solo ruling
        # that now owns a released ORI take opposite sides on whether the published value stands,
        # the solo ruling keeps the disposition but is marked needs_review, and the pair is reported.
        for o, owners in zip(dropped, released_to):
            for solo in zt_index[o]:
                if solo["case_id"] not in owners or stance(solo["verdict"]) == stance(r["verdict"]):
                    continue
                stance_conflicts[solo["case_id"]] = dict(
                    ori=o, bundle=r["case_id"], bundle_arm=r["arm"],
                    bundle_verdict=r["verdict"], bundle_stance=stance(r["verdict"]),
                    solo=solo["case_id"], solo_arm=solo["arm"],
                    solo_verdict=solo["verdict"], solo_stance=stance(solo["verdict"]))
        narrowed.append(dict(case_id=r["case_id"], case_class=r["case_class"], arm=r["arm"],
                             verdict=r["verdict"], kept=keep_oris, released=dropped,
                             released_to=released_to))

    out_cols = full_cols + [c for c in MY_COLS if c not in full_cols]
    ledger = sorted(ledger, key=lambda r: (r["case_class"], r["case_id"], str(r.get("final_pass"))))
    C.write_csv(os.path.join(OUT, "rulings_full.csv"), out_cols, ledger)

    live = [r for r in ledger if r["active"]]
    live_by_ori = collections.defaultdict(list)
    for r in live:
        for o in scope_of(r):
            live_by_ori[o].append(r)

    # ---------------- the supervisor's wholesale ruling, derived from the pass-3 queue
    esc_pre, esc_pre_cols = read_pre_final_queue()
    wholesale = {e["case_id"] for e in esc_pre
                 if e["kind"] == WHOLESALE_KIND and C.mass_f(e["mass_hint"]) <= WHOLESALE_MAX_MASS}
    esc_pre_by_case = {e["case_id"]: e for e in esc_pre}
    missing_dispositions = sorted(set(ABOVE_MASS_DISPOSITIONS) - set(esc_pre_by_case))
    if missing_dispositions:
        sys.exit(
            "the supervisor above-mass disposition names case IDs absent from the pass-3 "
            f"escalation queue: {missing_dispositions}"
        )
    wrong_kind = {
        cid: esc_pre_by_case[cid]["kind"]
        for cid in ABOVE_MASS_DISPOSITIONS
        if esc_pre_by_case[cid]["kind"] != WHOLESALE_KIND
    }
    if wrong_kind:
        sys.exit(
            "the supervisor above-mass disposition may only target explicit cleanup "
            f"escalations, but the queue carries: {wrong_kind}"
        )

    # ---------------- per-class registries, rebuilt from the live ledger
    cache, reg_cols, member = {}, {}, {}
    for key, fn in R.REG_FILES:
        rows, cols = C.read_registry(os.path.join(OUT, fn), MY_COLS)
        reg_cols[key] = cols
        for r in rows:
            cache[r["case_id"]] = r
            member[r["case_id"]] = key

    reg = {k: [] for k, _ in R.REG_FILES}
    wholesale_applied, wholesale_missing = [], sorted(wholesale)
    above_mass_applied, above_mass_missing = [], sorted(ABOVE_MASS_DISPOSITIONS)
    for r in live:
        cid = r["case_id"]
        if r["source"] == "final":
            key = route_final(r)
        elif r["source"] == "residue":
            key = R.route_residue(manifest[cid])
        elif cid in member:
            key = member[cid]
        else:
            continue                    # classes that never entered the registries (offroster etc.)
        old = cache.get(cid, {})
        d = R.derive(r["verdict"], r["detail"], r["rationale"])
        scope = scope_of(r)
        row = dict(old)
        row.update(
            case_id=cid, state=r["state"], oris=";".join(scope), n_oris=len(scope),
            verdict=r["verdict"], arm=r["arm"], confidence=r["confidence"],
            mass_hint=r["mass_hint"], replicate_conflict=r["replicate_conflict"],
            detail=(r["detail"] or r["verdict_raw"])[:1200],
            cleanup_pass=r["cleanup_pass"], cleanup_reason=r["cleanup_reason"],
            original_class=r["original_class"], residue_pass=r["residue_pass"],
            merged_from=r["merged_from"], final_pass=r["final_pass"],
            final_reason=r["final_reason"])
        for c in ("downstream_action", "believable_months", "repair_value_hint"):
            if c in reg_cols[key] or c in row:
                row[c] = d[c]
        if r["source"] == "final":
            row["provenance_kind"] = "final_direct_adjudication"
            row["canonical_ori"] = final[cid].get("canonical_ori", "")
            row["footprint"] = final[cid].get("footprint", "")
            row["needs_review"] = 0
        else:
            row["needs_review"] = int(r["verdict"] == "unclear_escalate"
                                      or str(r["replicate_conflict"]) == "1"
                                      or str(old.get("needs_review", 0)) == "1")
        if cid in stance_conflicts:
            sc = stance_conflicts[cid]
            row["needs_review"] = 1
            row["final_pass"] = "released_from_bundle_stance_conflict"
            row["final_reason"] = (
                f"ORI {sc['ori']} was released from the multi-ORI ruling {sc['bundle']} "
                f"({sc['bundle_arm']}: {sc['bundle_verdict']}, {sc['bundle_stance']}) by ORI-scope "
                f"narrowing; this row's own verdict ({sc['solo_arm']}: {sc['solo_verdict']}, "
                f"{sc['solo_stance']}) now stands alone and takes the OPPOSITE side on whether the "
                "published value stands. The disposition is this row's; the disagreement is not "
                "settled, only unqueued -- read both texts in rulings_full.csv before consuming.")
        if cid in wholesale:
            if row["verdict"] != "unclear_escalate":
                sys.exit(f"wholesale ruling targets {cid} but its live verdict is "
                         f"{row['verdict']!r}, not unclear_escalate; the pass-3 queue and the "
                         "ledger disagree -- reconcile before applying")
            row["downstream_action"] = WHOLESALE_ACTION
            row["needs_review"] = 1
            row["final_pass"] = "supervisor_wholesale_ruling"
            row["final_reason"] = WHOLESALE_NOTE
            wholesale_applied.append(dict(case_id=cid, registry=key, state=row["state"],
                                          oris=row["oris"], mass_hint=row["mass_hint"]))
            wholesale_missing.remove(cid)
        if cid in ABOVE_MASS_DISPOSITIONS:
            if row["verdict"] != "unclear_escalate":
                sys.exit(f"above-mass supervisor disposition targets {cid} but its live verdict "
                         f"is {row['verdict']!r}, not unclear_escalate; reconcile the queue and "
                         "ledger before applying")
            row["downstream_action"] = ABOVE_MASS_ACTION
            row["needs_review"] = 1
            row["final_pass"] = "supervisor_above_mass_disposition"
            row["final_reason"] = ABOVE_MASS_DISPOSITIONS[cid]
            above_mass_applied.append(dict(
                case_id=cid, registry=key, state=row["state"], oris=row["oris"],
                mass_hint=row["mass_hint"], note=ABOVE_MASS_DISPOSITIONS[cid],
            ))
            above_mass_missing.remove(cid)
        reg[key].append(row)

    if wholesale_missing:
        sys.exit("the wholesale ruling names rows that are not live in any registry: "
                 f"{wholesale_missing}")
    if above_mass_missing:
        sys.exit("the above-mass supervisor disposition names rows that are not live in any "
                 f"registry: {above_mass_missing}")

    for key, fn in R.REG_FILES:
        # `oris_scope` stays a ledger-only column: in the registries `oris` IS the effective scope.
        cols = reg_cols[key] + [c for c in MY_COLS
                                if c not in reg_cols[key] and c != "oris_scope"]
        rows = sorted(reg[key], key=lambda r: -C.mass_f(r.get("mass_hint", "")))
        C.write_csv(os.path.join(OUT, fn), cols, rows)

    # ---------------- escalation queue: retire what pass 4 settled
    narrowed_ids = {n["case_id"] for n in narrowed}
    esc, retired = [], []
    for e in esc_pre:
        e = dict(e)
        kind = e["kind"]
        if kind == "residue_case_unruled" and e["case_id"] in final:
            retired.append(dict(e, final_status="resolved_by_final_ruling"))
            continue
        if kind == WHOLESALE_KIND and e["case_id"] in wholesale:
            retired.append(dict(e, final_status="resolved_by_supervisor_wholesale_ruling"))
            continue
        if kind == WHOLESALE_KIND and e["case_id"] in ABOVE_MASS_DISPOSITIONS:
            retired.append(dict(e, final_status="resolved_by_supervisor_above_mass_disposition"))
            continue
        if kind == "partial_ori_overlap_not_superseded" and e["case_id"] in narrowed_ids:
            retired.append(dict(e, final_status="resolved_by_ori_scope_narrowing"))
            continue
        if kind == "residual_cross_class_contradiction":
            continue                     # recomputed below over the rebuilt ledger
        e["final_status"] = "still_open"
        esc.append(e)

    # ---------------- checks recomputed over the rebuilt ledger
    queued_oris = {o for e in esc for o in C.split_ids(e["oris"])}

    def label(r):
        return f"{r['case_id']}({r['case_class']}/{r['arm']}):{r['verdict']}"

    xclass, xclass_queued = [], []
    for o, v in sorted(live_by_ori.items()):
        if {"ACCEPT", "REJECT"} <= {stance(r["verdict"]) for r in v}:
            item = (o, [label(r) for r in v])
            (xclass_queued if o in queued_oris else xclass).append(item)
    for o, v in xclass:
        esc.append(dict(
            case_id="ORI " + o, kind="residual_cross_class_contradiction", state="", oris=o,
            mass_hint="", original_class="", cleanup_reason="", verdict=";".join(sorted(
                {r["verdict"] for r in live_by_ori[o]})), confidence="", arm="", agent="",
            downstream_action="reconcile_before_apply", alt_rulings=" || ".join(v),
            settling_record="two live rulings for this ORI take opposite sides on whether the "
                            "published value stands; nothing else in the queue names this ORI",
            ruling_detail="", ruling_rationale="", evidence_summary="", packet_file="",
            residue_status="still_open", final_status="still_open"))

    multi = {f"zero_token:{o}": [label(r) for r in rs]
             for o, v in sorted(live_by_ori.items())
             for rs in ([r for r in v if R.question(r["case_class"]) == "zero_token"],)
             if len(rs) > 1}
    multi_queued = {k: v for k, v in multi.items() if k.split(":", 1)[1] in queued_oris}
    problems = [f"ORI {k} carries {len(v)} live zero/token rulings and nothing in the queue names "
                f"it: {v}" for k, v in multi.items() if k not in multi_queued]

    MERGE_V, DIST_V = {"same_agency_merge", "superseded_ori"}, {"distinct_agencies"}
    twin_pair = collections.defaultdict(list)
    for r in live:
        if R.question(r["case_class"]) != "twin":
            continue
        for a, b in itertools.combinations(sorted(scope_of(r)), 2):
            twin_pair[(a, b)].append(r)
    twin_pair_multi = {f"{a}+{b}": [label(r) for r in v]
                       for (a, b), v in sorted(twin_pair.items()) if len(v) > 1}
    twin_pair_split = {}
    for (a, b), v in sorted(twin_pair.items()):
        st = {"MERGE" if r["verdict"] in MERGE_V else
              "DISTINCT" if r["verdict"] in DIST_V else "OTHER" for r in v}
        if {"MERGE", "DISTINCT"} <= st:
            twin_pair_split[f"{a}+{b}"] = [label(r) for r in v]
    problems += [f"ORI pair {k} carries {len(v)} live identity rulings: {v}"
                 for k, v in twin_pair_multi.items()]
    problems += [f"ORI pair {k} is ruled BOTH merge and distinct: {v}"
                 for k, v in twin_pair_split.items()]
    problems += [f"bundle narrowing refused: {n}" for n in narrow_refused]

    esc_cols = esc_pre_cols + [c for c in ("residue_status", "final_status")
                               if c not in esc_pre_cols]
    esc.sort(key=lambda r: (r["kind"], -C.mass_f(r["mass_hint"])))
    C.write_csv(ESC_LIVE, esc_cols, esc)

    # ---------------- summary
    twin_merges = [r for r in reg["twins"] if r["downstream_action"] == "merge_dedupe"]
    summary = dict(
        final_rulings_loaded=len(final),
        final_verdicts=collections.Counter(k["verdict_norm"] for k in final.values()),
        final_downstream=collections.Counter(
            R.derive(k["verdict_norm"], k.get("detail"), k.get("rationale"))["downstream_action"]
            for k in final.values()),
        superseded_by_final=sorted(r["case_id"] for r in prior
                                   if r["final_pass"] == "superseded_by_final"),
        final_partial_overlap=[r["case_id"] for r in final_partial_overlap],
        ori_scope_narrowed=narrowed,
        ori_scope_narrow_refused=narrow_refused,
        released_bundle_stance_conflicts=sorted(stance_conflicts.values(), key=lambda d: d["ori"]),
        wholesale_kind=WHOLESALE_KIND, wholesale_max_mass=WHOLESALE_MAX_MASS,
        wholesale_action=WHOLESALE_ACTION,
        wholesale_applied=len(wholesale_applied), wholesale_rows=wholesale_applied,
        above_mass_action=ABOVE_MASS_ACTION,
        above_mass_applied=len(above_mass_applied), above_mass_rows=above_mass_applied,
        ledger_rows=len(ledger), ledger_active=len(live),
        registry_totals={k: len(v) for k, v in reg.items()},
        registry_final_rows={k: sum(1 for r in v if r["final_pass"] == "final_ruling")
                             for k, v in reg.items()},
        twins_by_action=collections.Counter(r["downstream_action"] for r in reg["twins"]),
        twin_merge_cases=len(twin_merges),
        twin_merge_oris_dropped=sum(len(C.split_ids(r["oris"])) - 1 for r in twin_merges),
        zeros_by_action=collections.Counter(r["downstream_action"] for r in reg["zeros"]),
        tokens_by_action=collections.Counter(r["downstream_action"] for r in reg["tokens"]),
        needs_review={k: sum(1 for r in v if str(r.get("needs_review", 0)) == "1")
                      for k, v in reg.items()},
        escalations_pre_final=len(esc_pre),
        escalations_retired=len(retired),
        escalations_retired_by_kind=collections.Counter(
            f"{r['kind']}/{r['final_status']}" for r in retired),
        escalations_open=len(esc),
        escalations_open_by_kind=collections.Counter(r["kind"] for r in esc),
        escalations_open_rows=[dict(case_id=r["case_id"], kind=r["kind"], oris=r["oris"],
                                    mass_hint=r["mass_hint"], verdict=r["verdict"])
                               for r in esc],
        oris_multi_live_zero_token_ruling=multi,
        oris_multi_live_already_queued=sorted(multi_queued),
        twin_pairs_live=len(twin_pair),
        twin_pairs_multi_ruled=twin_pair_multi,
        twin_pairs_merge_distinct_split=twin_pair_split,
        cross_class_contradictions_unqueued=xclass,
        cross_class_contradictions_queued=xclass_queued,
        problems=problems,
    )
    summary = {k: (dict(("|".join(kk) if isinstance(kk, tuple) else str(kk), vv)
                        for kk, vv in v.items()) if isinstance(v, collections.Counter) else v)
               for k, v in summary.items()}
    json.dump(summary, open(os.path.join(OUT, "final_apply_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    del packets


if __name__ == "__main__":
    main()
