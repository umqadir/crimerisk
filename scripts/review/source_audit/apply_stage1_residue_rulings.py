#!/usr/bin/env python3
"""Apply the Stage-1 RESIDUE-pass rulings and dedupe the agreed double-framing rows.

Third and last pass in the Stage-1 ad-hoc review chain:

    collate_stage1_adhoc_review.py     pass 1  1,089-case class-routed swarm  -> the base ledger
    apply_stage1_cleanup_rulings.py    pass 2  246 merged-packet re-rulings   -> lands on pass 1
    apply_stage1_residue_rulings.py    pass 3  37 residue re-rulings + the    -> lands on pass 2
                                               mechanical double-framing dedupe

What pass 3 settles
-------------------
1. RESIDUE RULINGS (residue_batches/, built by build_stage1_residue_batches.py). 37 token/zero-shaped
   cases came back ruled: 34 same-ORI double framings whose two pass-1 rulings implied DIFFERENT
   repairs, 2 cleanup rulings the finalizer QC re-read withdrew (KY0470400, GA0850200), and 1 ORI
   that had no zero/token disposition from any arm in either pass (AR063019E). The 2 twin_shaped
   cases in residue batch_006 were built and never dispatched, so they stay unruled and stay queued.

2. MECHANICAL DEDUPE of the 54 double-framing rows the cleanup pass triaged as pure duplication
   (`both_reject_same_repair` 45, `both_accept` 9). One ORI, two live pass-1 rulings, one from the
   published-zero framing and one from the token-reporter framing, both implying the SAME handling
   of the agency-year. Nothing to adjudicate: keep one row, retire the other, and record the
   retired row's provenance on the survivor so no reviewer's reasoning is lost from the ledger.

Contract
--------
* A residue ruling REPLACES every live ruling for its ORI(s) *within the same question*: a
  token_zero_shaped residue case supersedes published_zero / token_reporter / token_zero_shaped
  rows; a twin_shaped one supersedes ambiguous_twin rows. A zero/token ruling never supersedes a
  twin-identity ruling -- that is a different question about the same ORI, which is why AR063019E's
  live identity ruling survives its new zero/token ruling.
* Supersession fires only for residue cases that actually came back with a ruling.
* A multi-ORI row is superseded only if EVERY one of its ORIs is covered; partial overlaps are left
  standing and escalated, as in pass 2.
* residue batch_001 was adjudicated twice -- an agent dispatched to a nonexistent batch_000 fell
  through to the first batch, the same misdispatch pass 2 hit -- so 8 cases carry two independent
  rulings. Selection is pass 2's: lowest-commitment verdict, then confidence, then longer rationale,
  then agent id. Where the two verdicts differ but imply the SAME downstream action the case is
  recorded (replicate_conflict=1, needs_review=1) and NOT escalated -- there is nothing for a
  supervisor to decide. Where they imply different handling it is escalated.
* DEDUPE KEEP ORDER, deterministic, no adjudication:
    1. non-`luna` over `luna`. The luna arm's published-zero output was disqualified as a
       rule-echo -- its verdicts were a deterministic function of a precomputed provenance label in
       the packet header and carried no case-specific judgement -- so where a pair contains it, the
       other row is the only one carrying judgement about this ORI. (41 of the 54 pairs.)
    2. higher confidence.
    3. the row whose text yields an applicable repair basis (believable months or a repair
       magnitude) over one that yields neither.
    4. longer detail+rationale, then case_id, for total determinism.
  The dropped row stays in the ledger at active=0 with its full text, and the survivor carries
  `merged_from` naming it. Both sides' disposition is identical by construction, so the choice
  cannot change the repair -- only which reviewer's wording is the live one.

Re-runnability
--------------
Pass 3 recomputes `active`/`superseded_by` from `cleanup_pass` rather than trusting its own prior
writes, drops its own rows on read, and rebuilds the three per-class registries from the live ledger
(derived columns reused from the existing registries where present, recomputed from the ledger text
otherwise). The pass-2 escalation queue is snapshotted to supervisor_escalations_pre_residue.csv the
first time so a re-run reads the same input rather than its own filtered output.

Inputs   state/qa/stage1_adhoc_review/{rulings_full,zero_missing_adjudicated,
         token_reporters_adjudicated,twins_adjudicated,supervisor_escalations}.csv
         state/qa/stage1_adhoc_review/residue_batches/{manifest.csv,batch_*.json}
         residue rulings: residue_rulings.json (durable), refreshed from the subagent workflow
         journal when it is still on disk (STAGE1_RESIDUE_JOURNAL_DIR).
Outputs  the four registries regenerated, supervisor_escalations.csv regenerated,
         supervisor_escalations_pre_residue.csv (input snapshot), residue_apply_summary.json

Usage: uv run python scripts/review/source_audit/apply_stage1_residue_rulings.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_stage1_cleanup_rulings as C  # noqa: E402  pass-2 helpers, reused not recopied

ROOT = C.ROOT
OUT = C.OUT
RB = os.path.join(OUT, "residue_batches")
SNAPSHOT = os.path.join(OUT, "residue_rulings.json")
ESC_LIVE = os.path.join(OUT, "supervisor_escalations.csv")
ESC_PRE = os.path.join(OUT, "supervisor_escalations_pre_residue.csv")
JOURNAL = os.environ.get(
    "STAGE1_RESIDUE_JOURNAL_DIR",
    os.path.expanduser("~/.claude/projects/-Users-uzairqadir-Projects-data-projects-national-crimerisk-clone/"
                       "6b6a1654-17bb-4f3a-aa47-aa689bda8400/subagents/workflows/wf_40c4f7df-293"))

csv.field_size_limit(1 << 24)

ZT_CLASSES = {"published_zero", "token_reporter", "token_zero_shaped"}
TWIN_CLASSES = {"ambiguous_twin", "twin_shaped"}
DEDUPE_SUBTYPES = ("both_reject_same_repair", "both_accept")
RESIDUE_SUBTYPES = ("both_reject_DIFFERENT_repair", "disposition_conflict_FLAG/REJECT")
DBL_KIND = "same_ori_double_framing_outside_cleanup"
ALT_RE = re.compile(r"([\w.-]+)\((published_zero|token_reporter)/(\w+)/(\w*)\):")

# pass-3 columns on rulings_full.csv; stripped on read so a re-run cannot double-apply
MY_COLS = ("residue_pass", "residue_reason", "merged_from")
# columns owned by the pass-4 script (apply_stage1_final_rulings.py), which lands on top of this
# one. Stripped from every read here so a pass-3 re-run cannot carry stale pass-4 bookkeeping;
# pass-4 rows are dropped with the same reasoning that drops pass-3 rows in pass 2.
PASS_4_COLS = ("final_pass", "final_reason", "oris_scope")


def question(case_class):
    return "twin" if case_class in TWIN_CLASSES else "zero_token" if case_class in ZT_CLASSES else ""


# ---------------------------------------------------------------- residue rulings
def refresh_snapshot_from_journal(jdir):
    """Extract the residue rulings from the subagent workflow journal into a durable snapshot."""
    jpath = os.path.join(jdir, "journal.jsonl")
    if not os.path.isfile(jpath):
        return None
    arm, assigned = {}, {}
    for p in sorted(glob.glob(os.path.join(jdir, "agent-*.jsonl"))):
        aid = os.path.basename(p)[6:-6]
        c = json.loads(open(p).readline())["message"]["content"]
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        if len(c) > 20000:            # the finalizer's own prompt carries the whole ruling payload
            continue
        arm[aid] = C.arm_of(c)
        assigned[aid] = ";".join(f"batch_{b}.json" for b in sorted(set(re.findall(r"batch_(\d+)", c))))
    rulings = []
    for line in open(jpath):
        d = json.loads(line)
        if d.get("type") != "result":
            continue
        res = d["result"]
        if not (isinstance(res, dict) and "rulings" in res):
            continue
        aid = d["agentId"]
        for x in res["rulings"]:
            rulings.append(dict(case_id=x.get("case_id", ""), verdict=x.get("verdict", ""),
                                detail=x.get("detail", ""), rationale=x.get("rationale", ""),
                                confidence=x.get("confidence", ""),
                                agent=aid, arm=arm.get(aid, "opus"),
                                assigned_batch=assigned.get(aid, "")))
    snap = dict(source="subagent workflow journal " + os.path.basename(jdir),
                n_rulings=len(rulings), rulings=rulings)
    with open(SNAPSHOT, "w") as f:
        json.dump(snap, f, indent=1)
    return snap


def load_residue_rulings(refresh):
    snap = refresh_snapshot_from_journal(JOURNAL) if refresh else None
    if snap is None:
        if not os.path.isfile(SNAPSHOT):
            sys.exit(f"no residue rulings: neither {JOURNAL}/journal.jsonl nor {SNAPSHOT} exists")
        snap = json.load(open(SNAPSHOT))
    return snap


def load_packets():
    packets = {}
    for p in sorted(glob.glob(os.path.join(RB, "batch_*.json"))):
        for c in json.load(open(p)):
            packets[c["case_id"]] = dict(c, _file="residue_batches/" + os.path.basename(p))
    return packets


# ---------------------------------------------------------------- derived columns
def derive(verdict, detail, rationale):
    act = C.downstream_action(verdict, detail, rationale)
    return dict(downstream_action=act,
                believable_months=C.believable_months(detail, rationale),
                repair_value_hint=(C.repair_value(detail)
                                   if act in ("treat_missing_repair", "partial_year_uplift") else ""))


def route_residue(m):
    if m["class"] == "twin_shaped" or m["original_class"] == "ambiguous_twin":
        return "twins"
    return "zeros" if "published_zero" in m["original_class"] else "tokens"


REG_FILES = (("twins", "twins_adjudicated.csv"),
             ("zeros", "zero_missing_adjudicated.csv"),
             ("tokens", "token_reporters_adjudicated.csv"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true",
                    help="use residue_rulings.json even if the workflow journal is on disk")
    ap.add_argument("--qc-sample", type=int, default=4)
    ap.add_argument("--qc-seed", type=int, default=20260731)
    args = ap.parse_args()

    snap = load_residue_rulings(not args.no_refresh)
    manifest = {m["case_id"]: m for m in csv.DictReader(open(os.path.join(RB, "manifest.csv")))}
    packets = load_packets()

    # ---------------- one ruling per residue case (batch_001 was double-adjudicated again)
    per_case = collections.defaultdict(list)
    for r in snap["rulings"]:
        per_case[r["case_id"]].append(dict(r, verdict_norm=C.norm_verdict(r["verdict"])))

    unknown = sorted(set(per_case) - set(manifest))
    assert not unknown, f"rulings for case_ids not in the residue manifest: {unknown}"

    chosen, tok_only_conflicts, action_conflicts = {}, [], []
    for cid, lst in per_case.items():
        lst = sorted(lst, key=lambda r: (C.COMMITMENT.get(r["verdict_norm"], 1),
                                         C.CONF_RANK.get(r["confidence"], 3),
                                         -len(r["rationale"] or ""), r["agent"]))
        keep = dict(lst[0])
        vset = sorted({x["verdict_norm"] for x in lst})
        aset = sorted({C.downstream_action(x["verdict_norm"], x["detail"], x["rationale"]) for x in lst})
        keep["n_replicates"] = len(lst)
        keep["replicate_conflict"] = int(len(vset) > 1)
        keep["replicate_verdicts"] = ";".join(vset) if len(vset) > 1 else ""
        keep["replicate_actions"] = ";".join(aset)
        keep["alt_rulings"] = " || ".join(
            f"{x['verdict_norm']}({x['arm']}/{x['confidence']}) -> "
            f"{C.downstream_action(x['verdict_norm'], x['detail'], x['rationale'])}: "
            f"{(x['detail'] or x['rationale'])[:400]}" for x in lst[1:])
        chosen[cid] = keep
        if len(vset) > 1:
            (action_conflicts if len(aset) > 1 else tok_only_conflicts).append(cid)

    unruled = sorted(set(manifest) - set(chosen))

    # ---------------- supersession keys, from RULED residue cases only
    owner_by_ori, owner_by_id = {}, {}
    for cid, keep in chosen.items():
        m = manifest[cid]
        q = question(m["class"])
        for o in C.split_ids(m["oris"]):
            owner_by_ori[(q, o)] = cid
        for i in C.split_ids(m["source_case_ids"]) + [cid]:
            owner_by_id[(q, i)] = cid

    def supersession(row):
        q = question(row["case_class"])
        if not q:
            return None, ""
        if (q, row["case_id"]) in owner_by_id:
            return owner_by_id[(q, row["case_id"])], "same_case_id"
        oris = C.split_ids(row["oris"])
        hit = [o for o in oris if (q, o) in owner_by_ori]
        if not hit:
            return None, ""
        if len(hit) < len(oris):
            return None, "partial_ori_overlap"
        return owner_by_ori[(q, hit[0])], "same_ori"

    # ---------------- ledger: drop pass-3 rows, re-derive active from the pass-2 column
    full_rows, full_cols = C.read_registry(os.path.join(OUT, "rulings_full.csv"),
                                           MY_COLS + PASS_4_COLS)
    prior = []
    for r in full_rows:
        if r.get("source") in ("residue", "final"):
            continue                    # this script's own prior output; pass-4 rows rebuilt there
        superseded_by_pass2 = r.get("cleanup_pass") == "superseded_by_cleanup"
        r["active"] = 0 if superseded_by_pass2 else 1
        r["superseded_by"] = r.get("superseded_by", "") if superseded_by_pass2 else ""
        r["residue_pass"] = "retained_pre_residue"
        r["residue_reason"] = ""
        r["merged_from"] = ""
        prior.append(r)
    by_id = {r["case_id"]: r for r in prior}

    residue_partial_overlap = []
    for r in prior:
        if not r["active"]:
            continue
        owner, why = supersession(r)
        if owner:
            r["active"] = 0
            r["superseded_by"] = owner
            r["residue_pass"] = "superseded_by_residue"
        elif why == "partial_ori_overlap":
            residue_partial_overlap.append(r)

    # ---------------- mechanical dedupe of the agreed double-framing pairs
    esc_pre = read_pre_residue_queue()
    dedupe_rows = [r for r in esc_pre
                   if r["kind"] == DBL_KIND and r["settling_record"].split(":")[0] in DEDUPE_SUBTYPES]
    dedupe_log, dedupe_skipped = [], []

    def dedupe_key(r):
        d = derive(r["verdict"], r["detail"], r["rationale"])
        return (r["arm"] == "luna",
                C.CONF_RANK.get(r["confidence"], 3),
                not (d["believable_months"] or d["repair_value_hint"]),
                -len(r["detail"] or "") - len(r["rationale"] or ""),
                r["case_id"])

    for e in dedupe_rows:
        cids = [c for c, _cls, _a, _cf in ALT_RE.findall(e["alt_rulings"])]
        pair = [by_id[c] for c in cids if c in by_id]
        pair = [r for r in pair if r["active"]]
        if len(pair) < 2:
            dedupe_skipped.append(dict(oris=e["oris"], case_ids=cids,
                                       why="fewer than two live rows at dedupe time"))
            continue
        pair.sort(key=dedupe_key)
        keep, drops = pair[0], pair[1:]
        acts = {derive(r["verdict"], r["detail"], r["rationale"])["downstream_action"] for r in pair}
        for d in drops:
            d["active"] = 0
            d["superseded_by"] = keep["case_id"]
            d["residue_pass"] = "dedupe_dropped"
            d["merged_from"] = (f"kept {keep['case_id']}({keep['case_class']}/{keep['arm']}/"
                                f"{keep['confidence']}):{keep['verdict']}")
        keep["residue_pass"] = "dedupe_kept"
        keep["merged_from"] = " || ".join(
            f"{d['case_id']}({d['case_class']}/{d['arm']}/{d['confidence']}):{d['verdict']}"
            f"->{derive(d['verdict'], d['detail'], d['rationale'])['downstream_action']}"
            for d in drops)
        dedupe_log.append(dict(
            ori=e["oris"], subtype=e["settling_record"].split(":")[0],
            kept=keep["case_id"], kept_class=keep["case_class"], kept_arm=keep["arm"],
            kept_confidence=keep["confidence"], kept_verdict=keep["verdict"],
            dropped=[d["case_id"] for d in drops],
            dropped_arms=[d["arm"] for d in drops],
            actions_identical=int(len(acts) == 1), actions=sorted(acts)))

    # ---------------- residue rows
    residue_rows = []
    for cid, keep in sorted(chosen.items()):
        m = manifest[cid]
        oris = C.split_ids(m["oris"])
        superseded = sorted({r["case_id"] for r in prior
                             if r["residue_pass"] == "superseded_by_residue"
                             and r["superseded_by"] == cid})
        residue_rows.append(dict(
            case_id=cid, case_class=m["class"], subclass=cid.split("-")[0], state=m["state"],
            n_oris=len(oris), oris=";".join(oris), mass_hint=m["mass_hint"],
            arm=keep["arm"], verdict=keep["verdict_norm"], confidence=keep["confidence"],
            source="residue", source_batch=m["batch_file"], agent=keep["agent"],
            n_replicates=keep["n_replicates"], replicate_conflict=keep["replicate_conflict"],
            replicate_verdicts=keep["replicate_verdicts"], verdict_raw=keep["verdict"],
            detail=keep["detail"], rationale=keep["rationale"],
            packet_file="residue_batches/" + m["batch_file"],
            cleanup_pass="", active=1, superseded_by="",
            cleanup_reason=m["residue_subtype"], original_class=m["original_class"],
            residue_pass="residue_ruling", residue_reason=m["residue_reason"],
            merged_from=";".join(superseded)))

    out_cols = full_cols + [c for c in MY_COLS if c not in full_cols]
    ledger = sorted(prior + residue_rows,
                    key=lambda r: (r["case_class"], r["case_id"], str(r.get("residue_pass"))))
    C.write_csv(os.path.join(OUT, "rulings_full.csv"), out_cols, ledger)

    live = [r for r in ledger if r["active"]]
    live_by_ori = collections.defaultdict(list)
    for r in live:
        for o in C.split_ids(r["oris"]):
            live_by_ori[o].append(r)

    # ---------------- per-class registries, rebuilt from the live ledger
    # Derived columns are reused from the existing registries where the row is already there (so
    # pass-1/pass-2 rows stay byte-identical) and recomputed from the ledger text otherwise.
    cache, reg_cols, member = {}, {}, {}
    for key, fn in REG_FILES:
        rows, cols = C.read_registry(os.path.join(OUT, fn), MY_COLS + PASS_4_COLS)
        reg_cols[key] = cols
        for r in rows:
            cache[r["case_id"]] = r
            member[r["case_id"]] = key

    reg = {k: [] for k, _ in REG_FILES}
    for r in live:
        cid = r["case_id"]
        if r["source"] == "residue":
            key = route_residue(manifest[cid])
        elif cid in member:
            key = member[cid]
        else:
            continue                    # classes that never entered the registries (offroster etc.)
        old = cache.get(cid, {})
        d = derive(r["verdict"], r["detail"], r["rationale"])
        row = dict(old)
        row.update(
            case_id=cid, state=r["state"], oris=r["oris"], n_oris=r["n_oris"],
            verdict=r["verdict"], arm=r["arm"], confidence=r["confidence"],
            mass_hint=r["mass_hint"], replicate_conflict=r["replicate_conflict"],
            detail=(r["detail"] or r["verdict_raw"])[:1200],
            cleanup_pass=r["cleanup_pass"], cleanup_reason=r["cleanup_reason"],
            original_class=r["original_class"], residue_pass=r["residue_pass"],
            merged_from=r["merged_from"])
        for c in ("downstream_action", "believable_months", "repair_value_hint"):
            if c in reg_cols[key] or c in row:
                row[c] = d[c]
        if r["source"] == "residue":
            row["provenance_kind"] = "residue_merged_packet"
            row.setdefault("canonical_ori", "")
            row.setdefault("footprint", "")
        row["needs_review"] = int(r["verdict"] == "unclear_escalate"
                                 or str(r["replicate_conflict"]) == "1"
                                 or (r["residue_pass"] == "retained_pre_residue"
                                     and str(old.get("needs_review", 0)) == "1"
                                     and cid not in chosen))
        reg[key].append(row)

    for key, fn in REG_FILES:
        cols = reg_cols[key] + [c for c in MY_COLS if c not in reg_cols[key] and c != "residue_reason"]
        rows = sorted(reg[key], key=lambda r: -C.mass_f(r.get("mass_hint", "")))
        C.write_csv(os.path.join(OUT, fn), cols, rows)

    # ---------------- supervisor escalations: only genuinely-unresolved rows survive
    resolved_dbl_oris = {d["ori"] for d in dedupe_log}
    esc, retired = [], []

    def covered_by_residue(row_oris, q):
        oris = [o for o in C.split_ids(row_oris) if o]
        return bool(oris) and all((q, o) in owner_by_ori for o in oris)

    for e in esc_pre:
        e = dict(e)
        kind = e["kind"]
        if kind == DBL_KIND:
            sub = e["settling_record"].split(":")[0]
            if sub in DEDUPE_SUBTYPES and e["oris"] in resolved_dbl_oris:
                retired.append(dict(e, residue_status="resolved_by_mechanical_dedupe"))
                continue
            if sub in RESIDUE_SUBTYPES and covered_by_residue(e["oris"], "zero_token"):
                retired.append(dict(e, residue_status="resolved_by_residue_ruling"))
                continue
        elif kind == "cleanup_case_unruled":
            continue                     # rebuilt below against the residue manifest
        elif covered_by_residue(e["oris"], "zero_token"):
            retired.append(dict(e, residue_status="resolved_by_residue_ruling"))
            continue
        e["residue_status"] = "still_open"
        esc.append(e)

    def esc_row(cid, kind, keep, note, **kw):
        m = manifest.get(cid, {})
        pk = packets.get(cid, {})
        v = (keep or {}).get("verdict_norm", "")
        row = dict(
            case_id=cid, kind=kind, state=m.get("state", ""),
            oris=";".join(C.split_ids(m.get("oris", ""))), mass_hint=m.get("mass_hint", ""),
            original_class=m.get("original_class", ""),
            cleanup_reason=m.get("residue_reason", "") + "/" + m.get("residue_subtype", ""),
            verdict=v, confidence=(keep or {}).get("confidence", ""),
            arm=(keep or {}).get("arm", ""), agent=(keep or {}).get("agent", ""),
            downstream_action=(C.downstream_action(v, (keep or {}).get("detail"),
                                                   (keep or {}).get("rationale"))
                               if keep else "dispatch_again"),
            alt_rulings=(keep or {}).get("alt_rulings", ""),
            settling_record=note,
            ruling_detail=(keep or {}).get("detail", ""),
            ruling_rationale=(keep or {}).get("rationale", ""),
            evidence_summary=C.evidence_summary(pk.get("evidence", "")),
            packet_file=pk.get("_file", "residue_batches/" + m.get("batch_file", "")),
            residue_status="still_open")
        row.update(kw)
        return row

    for cid in unruled:
        m = manifest[cid]
        esc.append(esc_row(
            cid, "residue_case_unruled", None,
            f"dispatched in the residue pass ({m['batch_file']}) and no ruling returned by any "
            f"arm; the batch was built but never ran, so this is still the pass-2 "
            f"cleanup_case_unruled hole, now one pass older. Re-dispatch {m['batch_file']}.",
            alt_rulings="prior kind: cleanup_case_unruled"))
    for cid in action_conflicts:
        keep = chosen[cid]
        esc.append(esc_row(
            cid, "residue_replicate_conflict", keep,
            "residue batch_001 was adjudicated twice and the two arms' texts classify to DIFFERENT "
            f"handling ({keep['replicate_actions']}); the registry took the lower-commitment side "
            "and alt_rulings carries the other verbatim. Read both texts before treating this as a "
            "substantive disagreement: downstream_action is derived from wording cues, so a split "
            "can sit in the classifier rather than in the two readings -- if both texts say the "
            "year goes missing and is filled, the case is already settled and only the derived "
            "column needs a look."))

    # cross-question contradictions that survive: real only if nothing already queued names the ORI
    queued_oris = {o for e in esc for o in C.split_ids(e["oris"])}

    def stance(v):
        return {"genuine_zero_year": "ACCEPT", "genuine_low_crime": "ACCEPT",
                "misread_missing": "REJECT", "token_reporting_flag": "REJECT"}.get(v, "OTHER")

    xclass, xclass_queued = [], []
    for o, v in sorted(live_by_ori.items()):
        if {"ACCEPT", "REJECT"} <= {stance(r["verdict"]) for r in v}:
            item = (o, [f"{r['case_id']}({r['case_class']}/{r['arm']}):{r['verdict']}" for r in v])
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
            residue_status="still_open"))

    # ---------------- one live ruling per unit of question, over the rebuilt ledger
    # The unit differs by question and the difference is the point. A zero/token ruling disposes of
    # ONE agency-year, so the unit is the ORI and two live rulings on one ORI is a defect. An
    # identity ruling disposes of a PAIR, so one ORI legitimately appears in several live rulings
    # (MA007049E is distinct from Chester PD *and* the same agency as its own legacy Return-A key);
    # the defect there is a PAIR ruled twice, or ruled both merge and distinct.
    def label(r):
        return f"{r['case_id']}({r['case_class']}/{r['arm']}):{r['verdict']}"

    multi = {f"zero_token:{o}": [label(r) for r in rs]
             for o, v in sorted(live_by_ori.items())
             for rs in ([r for r in v if question(r["case_class"]) == "zero_token"],)
             if len(rs) > 1}
    multi_queued = {k: v for k, v in multi.items() if k.split(":", 1)[1] in queued_oris}
    problems = [f"ORI {k} carries {len(v)} live rulings and nothing in the queue names it: {v}"
                for k, v in multi.items() if k not in multi_queued]

    MERGE_V, DIST_V = {"same_agency_merge", "superseded_ori"}, {"distinct_agencies"}
    twin_pair = collections.defaultdict(list)
    for r in live:
        if question(r["case_class"]) != "twin":
            continue
        for a, b in itertools.combinations(sorted(C.split_ids(r["oris"])), 2):
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

    ESC_COLS = ["case_id", "kind", "state", "oris", "mass_hint", "original_class", "cleanup_reason",
                "verdict", "confidence", "arm", "agent", "downstream_action", "alt_rulings",
                "settling_record", "ruling_detail", "ruling_rationale", "evidence_summary",
                "packet_file", "residue_status"]
    esc.sort(key=lambda r: (r["kind"], -C.mass_f(r["mass_hint"])))
    C.write_csv(ESC_LIVE, ESC_COLS, esc)

    # ---------------- summary
    reg_live = {k: len(v) for k, v in reg.items()}
    summary = dict(
        residue_rulings_loaded=len(snap["rulings"]),
        residue_cases_ruled=len(chosen),
        residue_cases_dispatched=len(manifest),
        residue_cases_unruled=unruled,
        replicate_pairs=sum(1 for k in chosen.values() if k["n_replicates"] > 1),
        replicate_token_only_conflicts=sorted(tok_only_conflicts),
        replicate_action_conflicts=sorted(action_conflicts),
        verdicts=collections.Counter(k["verdict_norm"] for k in chosen.values()),
        confidence=collections.Counter(k["confidence"] for k in chosen.values()),
        downstream=collections.Counter(
            C.downstream_action(k["verdict_norm"], k["detail"], k["rationale"])
            for k in chosen.values()),
        by_residue_reason=collections.Counter(manifest[c]["residue_reason"] for c in chosen),
        routed_to=collections.Counter(route_residue(manifest[c]) for c in chosen),
        prior_rows=len(prior),
        superseded_by_residue=sum(1 for r in prior if r["residue_pass"] == "superseded_by_residue"),
        dedupe_pairs=len(dedupe_log),
        dedupe_dropped=sum(len(d["dropped"]) for d in dedupe_log),
        dedupe_by_subtype=collections.Counter(d["subtype"] for d in dedupe_log),
        dedupe_kept_arm=collections.Counter(d["kept_arm"] for d in dedupe_log),
        dedupe_dropped_arm=collections.Counter(a for d in dedupe_log for a in d["dropped_arms"]),
        dedupe_actions_identical=sum(d["actions_identical"] for d in dedupe_log),
        dedupe_skipped=dedupe_skipped,
        dedupe_log=dedupe_log,
        ledger_rows=len(ledger), ledger_active=len(live),
        registry_totals=reg_live,
        registry_residue_rows={k: sum(1 for r in v if r["residue_pass"] == "residue_ruling")
                               for k, v in reg.items()},
        escalations_pre_residue=len(esc_pre),
        escalations_retired=len(retired),
        escalations_retired_by_kind=collections.Counter(
            f"{r['kind']}/{r['residue_status']}" for r in retired),
        escalations_open=len(esc),
        escalations_open_by_kind=collections.Counter(r["kind"] for r in esc),
        oris_multi_live_zero_token_ruling=multi,
        oris_multi_live_already_queued=sorted(multi_queued),
        twin_pairs_live=len(twin_pair),
        twin_pairs_multi_ruled=twin_pair_multi,
        twin_pairs_merge_distinct_split=twin_pair_split,
        cross_class_contradictions_unqueued=xclass,
        cross_class_contradictions_queued=xclass_queued,
        problems=problems,
        qc_sample=sorted(random.Random(args.qc_seed).sample(sorted(chosen),
                                                            min(args.qc_sample, len(chosen)))),
    )
    summary = {k: (dict(("|".join(kk) if isinstance(kk, tuple) else str(kk), vv)
                        for kk, vv in v.items()) if isinstance(v, collections.Counter) else v)
               for k, v in summary.items()}
    json.dump(summary, open(os.path.join(OUT, "residue_apply_summary.json"), "w"), indent=1)

    print(json.dumps({k: summary[k] for k in (
        "residue_rulings_loaded", "residue_cases_ruled", "residue_cases_dispatched",
        "residue_cases_unruled", "replicate_pairs", "replicate_token_only_conflicts",
        "replicate_action_conflicts", "verdicts", "confidence", "downstream", "by_residue_reason",
        "routed_to", "superseded_by_residue", "dedupe_pairs", "dedupe_dropped",
        "dedupe_by_subtype", "dedupe_kept_arm", "dedupe_dropped_arm", "dedupe_actions_identical",
        "dedupe_skipped", "ledger_rows", "ledger_active", "registry_totals",
        "registry_residue_rows", "escalations_pre_residue", "escalations_retired",
        "escalations_retired_by_kind", "escalations_open", "escalations_open_by_kind",
        "oris_multi_live_zero_token_ruling", "oris_multi_live_already_queued",
        "twin_pairs_live", "twin_pairs_multi_ruled", "twin_pairs_merge_distinct_split",
        "cross_class_contradictions_unqueued", "cross_class_contradictions_queued",
        "problems", "qc_sample")}, indent=1))


def read_pre_residue_queue():
    """The pass-2 escalation queue, snapshotted so a pass-3 re-run reads its input not its output."""
    live_cols = []
    if os.path.isfile(ESC_LIVE):
        with open(ESC_LIVE) as f:
            live_cols = next(csv.reader(f), [])
    if "residue_status" not in live_cols and live_cols:
        rows, cols = C.read_registry(ESC_LIVE)
        C.write_csv(ESC_PRE, cols, rows)                   # fresh pass-2 output -> refresh snapshot
        return rows
    if not os.path.isfile(ESC_PRE):
        sys.exit(f"{ESC_LIVE} is already pass-3 output and {ESC_PRE} is missing; "
                 f"re-run apply_stage1_cleanup_rulings.py first")
    return C.read_registry(ESC_PRE)[0]


if __name__ == "__main__":
    main()
