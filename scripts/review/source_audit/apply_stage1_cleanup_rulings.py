#!/usr/bin/env python3
"""Apply the Stage 1 cleanup-pass rulings on top of the collated registries.

Sibling of collate_stage1_adhoc_review.py (which builds the PRE-cleanup ledger from the
banked/pilot/first-swarm sources). This script is the second stage: the 246-case cleanup pass
re-adjudicated, on ONE merged packet per ORI, everything the first pass left broken —
68 cross-class same-ORI contradictions, 30 same-ORI double dispatches, 62 explicit escalations,
15 replicate conflicts, 108 structural-plausibility residue cases, 1 coverage gap.

Contract
--------
* A cleanup ruling REPLACES every pre-cleanup ruling for its ORI(s) *within the same question*:
  a token_zero_shaped cleanup case supersedes published_zero and token_reporter rows;
  a twin_shaped cleanup case supersedes ambiguous_twin rows. A zero/token ruling never
  supersedes a twin-identity ruling (different question about the same ORI).
* Supersession fires only for cleanup cases that actually came back with a ruling.
* A multi-ORI pre-cleanup row is superseded only if EVERY one of its ORIs is covered by cleanup
  rulings; a partial overlap is left standing and escalated instead (it would otherwise silently
  drop coverage for the ORIs the cleanup pass never saw).
* One ruling per ORI. batch_001 was adjudicated twice (a sol runner had it as its assignment; an
  opus runner was misdispatched to a nonexistent batch_000 and fell through to the first cleanup
  batch), so 10 cases carry two independent rulings. Where they disagree the registry takes the
  LOWEST-COMMITMENT disposition — escalate < reject-the-value < accept-the-value — because
  carrying a published zero as a measured full year at weight 1.0 is the higher-commitment
  assertion, and it is never made over a second reviewer's rejection. Every such case is flagged
  needs_review and written to supervisor_escalations.csv with both sides.

Derived columns for pre-cleanup rows are carried through from the existing per-class CSVs
byte-for-byte (not recomputed) so this script cannot drift from the collator on rows it does not
touch. The derivation helpers below are copies of the collator's, used only for cleanup rows;
`--selftest` re-derives the pre-cleanup rows and asserts the copies still reproduce them.

Inputs   state/qa/stage1_adhoc_review/{rulings_full,zero_missing_adjudicated,
         token_reporters_adjudicated,twins_adjudicated,escalations}.csv
         state/qa/stage1_adhoc_review/cleanup_batches/{manifest.csv,batch_*.json}
         cleanup rulings: cleanup_rulings.json (durable), refreshed from the subagent workflow
         journal when it is still on disk (STAGE1_CLEANUP_JOURNAL_DIR).
Outputs  the five CSVs above, regenerated; supervisor_escalations.csv; cleanup_apply_summary.json

Re-runnable: every derived column is recomputed from scratch each run, and the script's own
output columns are stripped on read, so running it twice is a no-op.
"""
import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "state/qa/stage1_adhoc_review")
CB = os.path.join(OUT, "cleanup_batches")
SNAPSHOT = os.path.join(OUT, "cleanup_rulings.json")
JOURNAL = os.environ.get(
    "STAGE1_CLEANUP_JOURNAL_DIR",
    os.path.expanduser("~/.claude/projects/-Users-uzairqadir-Projects-data-projects-national-crimerisk-clone/"
                       "6b6a1654-17bb-4f3a-aa47-aa689bda8400/subagents/workflows/wf_9e0a72fc-92a"))

csv.field_size_limit(1 << 24)

# ------------------------------------------------------------------ collator helper copies
VOCAB = ["genuine_zero_year", "misread_missing", "same_agency_merge", "superseded_ori",
         "distinct_agencies", "genuine_low_crime", "token_reporting_flag",
         "defunct_no_fill", "plausibly_active", "unclear_escalate"]


def norm_verdict(v):
    v = (v or "").strip()
    low = v.lower()
    hits = [(low.find(t), t) for t in VOCAB if t in low]
    if not hits:
        return v.split()[0] if v else ""
    hits.sort()
    return hits[0][1]


MISSING_CUES = ("treat missing", "treat as missing", "structurally missing", "structural missing",
                "set missing", "mark missing", "mark as missing", "leave missing", "leave 2024 missing",
                "treat 2024 as missing", "carry 2024 uncertain", "should be missing", "is missing",
                "treat unresolved", "structural zero", "publication_gap_treat_as_missing",
                "treat as a", "missing rather than")
NO_UPLIFT_CUES = ("no month-count uplift", "cannot be rescued by uplift", "cannot uplift",
                  "uplift cannot", "no uplift can", "do not annualize", "rather than annualizing",
                  "not scalable", "zero cannot annualize", "zero cannot be", "does not repair",
                  "not merely annualize", "instead of annualizing", "rather than scaling",
                  "no month-count", "cannot repair")
UPLIFT_CUES = ("annualiz", "uplift", "x12", "12/", "partial-year", "partial year", "believable month",
               "month-ratio", "coverage repair", "months believable", "as an 11-month", "as a 10-month")


def downstream_action(verdict, detail, rationale):
    t = ((detail or "") + " " + (rationale or "")).lower()
    if verdict == "unclear_escalate":
        return "escalate_hold"
    if verdict == "genuine_low_crime":
        return "accept_observed"
    if verdict == "genuine_zero_year":
        return "retain_zero"
    if verdict == "misread_missing":
        if any(c in t for c in UPLIFT_CUES) and "partial_year_coverage" in t:
            return "partial_year_uplift"
        return "treat_missing_repair"
    if verdict == "token_reporting_flag":
        if any(c in t for c in NO_UPLIFT_CUES):
            return "treat_missing_repair"
        if any(c in t for c in MISSING_CUES):
            return "treat_missing_repair"
        if any(c in t for c in UPLIFT_CUES):
            return "partial_year_uplift"
        return "flag_review"
    if verdict in ("same_agency_merge", "superseded_ori"):
        return "merge_dedupe"
    if verdict == "distinct_agencies":
        return "keep_all_oris"
    if verdict == "defunct_no_fill":
        return "no_fill_retire"
    if verdict == "plausibly_active":
        return "no_fill_coverage_gap"
    return "review"


def believable_months(detail, rationale):
    for t in (detail, rationale):
        m = re.search(r"(?:only\s+)?(\d{1,2})\s*(?:believable|reported|submitted|usable|evidenced)\s*(?:SRS\s*)?months?", t or "", re.I)
        if m:
            return m.group(1)
        m = re.search(r"(\d{1,2})\s*months?\s*(?:believable|reported|submitted|evidenced)", t or "", re.I)
        if m:
            return m.group(1)
        m = re.search(r"\b(?:a|an)\s+(\d{1,2})-month\s+(?:partial|report)", t or "", re.I)
        if m:
            return m.group(1)
        m = re.search(r"believable months?\s*[:=]?\s*(\d{1,2})", t or "", re.I)
        if m:
            return m.group(1)
        m = re.search(r"(\d{1,2})\s*(?:SRS\s*)?months?\s+is\s+the\s+maximum", t or "", re.I)
        if m:
            return m.group(1)
    return ""


REPAIR_VAL = re.compile(r"~\s*([0-9][0-9,]*(?:\.[0-9]+)?)(?!\s*(?:month|mo\b|%))")


def repair_value(detail):
    m = REPAIR_VAL.search(detail or "")
    return m.group(1).replace(",", "") if m else ""


def mass_f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


# ------------------------------------------------------------------ cleanup-pass specifics
def split_ids(s):
    """manifest uses '|' between twin ORIs and ';' between source case ids; registries use ';'."""
    return [x.strip() for x in re.split(r"[;|]", s or "") if x.strip()]


# lowest commitment first: an escalation asserts nothing, a rejection asserts only that the
# published value is unusable, an acceptance asserts the value is a measured full year.
COMMITMENT = {"unclear_escalate": 0, "misread_missing": 1, "token_reporting_flag": 2,
              "genuine_zero_year": 3, "genuine_low_crime": 3,
              "superseded_ori": 2, "same_agency_merge": 2, "distinct_agencies": 3,
              "defunct_no_fill": 1, "plausibly_active": 1}
CONF_RANK = {"high": 0, "medium": 1, "low": 2, "": 3}

# columns owned by the pass-3 script (apply_stage1_residue_rulings.py)
PASS_3_COLS = ("residue_pass", "residue_reason", "merged_from")


def arm_of(prompt):
    """The sol runners shell out to `codex exec --model gpt-5.6-sol`; the rest are Opus subagents."""
    return "sol" if re.search(r"gpt-5\.6-sol|Sol-arm", prompt or "") else "opus"


def refresh_snapshot_from_journal(jdir):
    """Extract the cleanup rulings out of the subagent workflow journal into a durable snapshot."""
    jpath = os.path.join(jdir, "journal.jsonl")
    if not os.path.isfile(jpath):
        return None
    arm, assigned = {}, {}
    for p in sorted(glob.glob(os.path.join(jdir, "agent-*.jsonl"))):
        aid = os.path.basename(p)[6:-6]
        c = json.loads(open(p).readline())["message"]["content"]
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        if len(c) > 20000:           # the finalizer's own prompt carries the whole ruling payload
            continue
        arm[aid] = arm_of(c)
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


def load_cleanup_rulings(refresh):
    snap = refresh_snapshot_from_journal(JOURNAL) if refresh else None
    if snap is None:
        if not os.path.isfile(SNAPSHOT):
            sys.exit(f"no cleanup rulings: neither {JOURNAL}/journal.jsonl nor {SNAPSHOT} exists")
        snap = json.load(open(SNAPSHOT))
    return snap


def read_registry(path, drop_cols=()):
    if not os.path.isfile(path):
        return [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = [c for c in rd.fieldnames if c not in drop_cols]
        rows = [{k: v for k, v in r.items() if k in cols} for r in rd]
    return rows, cols


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# Findings from the finalizer QC re-read (8 seeded cleanup rulings vs their merged packets).
# Recorded here rather than only in prose so they travel with the queue and survive a re-run.
QC_FLAGS = {
    "x-KY0470400-west-point": (
        "QC: verdict likely WRONG. The ruling anchors 'zero is plausible' on 2022 (2 offences) and "
        "2023 (0), but the panel is 10, 7, 14, 9, 2, 0, 0 — five consecutive 12-month positive years "
        "at 7-14 through 2021, then the cliff. Using the already-degraded 2022-23 tail as the base "
        "rate is the own-history-launders-the-artefact error this same pass refuses elsewhere, and "
        "SRS months degrade 7/10/9/6/2/0/0 under an unchanged 12-month NIBRS header. Re-read; the "
        "likely correct verdict is misread_missing anchored on the 2018-2021 level."),
    "x-GA0850200-milner": (
        "QC: verdict defensible, CONFIDENCE OVERSTATED. 'No positive count in its seven-year "
        "history' counts five years (2018-2022) in which the ORI submitted nothing at all (SRS "
        "0 months, no NIBRS row) — real exposure is 22 filed months (2023 NIB 10mo + 2024 12mo). "
        "Population 828 is below the 1,000 line at which this pass treats never-positive as "
        "structural, and other rulings in the same pass call that arm vacuous on 0-month years. "
        "Downgrade to medium or re-read."),
}


# ------------------------------------------------------------------ evidence
def load_packets():
    packets = {}
    for p in sorted(glob.glob(os.path.join(CB, "batch_*.json"))):
        for c in json.load(open(p)):
            packets[c["case_id"]] = dict(c, _file="cleanup_batches/" + os.path.basename(p))
    return packets


EV_CAP = 16000


EV_STARTS = ("FACTS, PART 1", "FACTS:", "SITUATION:", "NOTE ON THE", "STRUCTURAL PLAUSIBILITY",
             "PANEL EVIDENCE:")


def evidence_summary(ev):
    """Case-specific span of the packet: FACTS/SITUATION through COMPUTED SIGNALS, boilerplate cut.

    Merged (two-framing) packets head their evidence 'FACTS, PART 1'; single-framing packets use a
    bare 'FACTS:'; the twin packets use 'SITUATION:'. Take the earliest marker present so the
    structural-plausibility block is never cut off.
    """
    if not ev:
        return ""
    starts = [i for i in (ev.find(m) for m in EV_STARTS) if i >= 0]
    i = min(starts) if starts else 0
    j = ev.find("VERDICT VOCABULARY")
    body = ev[i:j if j > i else len(ev)].strip()
    return body[:EV_CAP] + ("\n[...truncated]" if len(body) > EV_CAP else "")


SETTLE = re.compile(r"[^.]*\b(?:would settle|would resolve|what would settle|settling record|Obtain |resolving record|"
                    r"records? that would|check(?:ing)? whether|settles? it|settle this)\b[^.]*\.", re.I)


def settling_record(detail, rationale):
    for t in (detail, rationale):
        m = SETTLE.search(t or "")
        if m:
            return m.group(0).strip()[:600]
    return ""


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true",
                    help="use the cleanup_rulings.json snapshot even if the workflow journal is present")
    ap.add_argument("--selftest", action="store_true",
                    help="assert the copied derivation helpers still reproduce the collator's columns")
    ap.add_argument("--qc-sample", type=int, default=8)
    ap.add_argument("--qc-seed", type=int, default=20260730)
    args = ap.parse_args()

    snap = load_cleanup_rulings(not args.no_refresh)
    manifest = {m["case_id"]: m for m in csv.DictReader(open(os.path.join(CB, "manifest.csv")))}
    packets = load_packets()

    # ---------------- one ruling per cleanup case (batch_001 was double-adjudicated)
    per_case = collections.defaultdict(list)
    for r in snap["rulings"]:
        r = dict(r, verdict_norm=norm_verdict(r["verdict"]))
        per_case[r["case_id"]].append(r)

    chosen, conflicts = {}, []
    for cid, lst in per_case.items():
        lst = sorted(lst, key=lambda r: (COMMITMENT.get(r["verdict_norm"], 1),
                                         CONF_RANK.get(r["confidence"], 3),
                                         -len(r["rationale"] or ""), r["agent"]))
        keep = lst[0]
        vset = sorted({x["verdict_norm"] for x in lst})
        keep["n_replicates"] = len(lst)
        keep["replicate_conflict"] = int(len(vset) > 1)
        keep["replicate_verdicts"] = ";".join(vset) if len(vset) > 1 else ""
        keep["alt_rulings"] = " || ".join(
            f"{x['verdict_norm']}({x['arm']}/{x['confidence']}): {(x['detail'] or x['rationale'])[:400]}"
            for x in lst[1:])
        chosen[cid] = keep
        if len(vset) > 1:
            conflicts.append(cid)

    unruled = sorted(set(manifest) - set(chosen))

    # ---------------- supersession keys, from RULED cleanup cases only
    tz_oris, tz_ids, tw_oris, tw_ids = set(), set(), set(), set()
    ori_owner, id_owner = {}, {}
    for cid, keep in chosen.items():
        m = manifest[cid]
        oris, ids = split_ids(m["oris"]), split_ids(m["source_case_ids"]) + [cid]
        if m["class"] == "twin_shaped":
            tw_oris.update(oris)
            tw_ids.update(ids)
        else:
            tz_oris.update(oris)
            tz_ids.update(ids)
        for o in oris:
            ori_owner[o] = cid
        for i in ids:
            id_owner[i] = cid
    ZT = {"published_zero", "token_reporter"}

    def supersession(row):
        """-> (cleanup case_id that replaces this row, reason) or (None, '')"""
        oris = split_ids(row["oris"])
        cls = row["case_class"]
        if cls not in ZT and cls != "ambiguous_twin":
            return None, ""
        ids, keys = (tw_ids, tw_oris) if cls == "ambiguous_twin" else (tz_ids, tz_oris)
        if row["case_id"] in ids:
            return id_owner[row["case_id"]], "same_case_id"
        hit = [o for o in oris if o in keys]
        if not hit:
            return None, ""
        if len(hit) < len(oris):
            return None, "partial_ori_overlap"        # left standing on purpose, escalated
        return ori_owner[hit[0]], "same_ori"

    # ---------------- pre-cleanup ledger
    MY_COLS = ("cleanup_pass", "active", "superseded_by", "cleanup_reason", "original_class",
               "resolved_by_cleanup", "resolution")
    # PASS_3_COLS belong to apply_stage1_residue_rulings.py, which lands on top of this script.
    # Stripped from the ledger read so a re-run of pass 2 cannot carry stale pass-3 bookkeeping on
    # rows it rebuilds; pass 3 re-derives them from scratch. They are KEPT on the per-class registry
    # read below, where `residue_pass` is how a pass-3 row is recognised and skipped.
    full_rows, full_cols = read_registry(os.path.join(OUT, "rulings_full.csv"),
                                         MY_COLS + PASS_3_COLS)
    # pass-3 rows are rebuilt by pass 3, not carried here as if they were pre-cleanup rulings
    prior = [r for r in full_rows if r.get("source") not in ("cleanup", "residue")]

    partial_overlap = []
    for r in prior:
        owner, why = supersession(r)
        r["superseded_by"] = owner or ""
        r["cleanup_pass"] = "superseded_by_cleanup" if owner else "retained_pre_cleanup"
        r["active"] = 0 if owner else 1
        if why == "partial_ori_overlap":
            partial_overlap.append(r)

    # ---------------- cleanup rows
    cleanup_rows = []
    for cid, keep in sorted(chosen.items()):
        m = manifest[cid]
        oris = split_ids(m["oris"])
        v = keep["verdict_norm"]
        cleanup_rows.append(dict(
            case_id=cid, case_class=m["class"], subclass=cid.split("-")[0], state=m["state"],
            n_oris=len(oris), oris=";".join(oris), mass_hint=m["mass_hint"],
            arm=keep["arm"], verdict=v, confidence=keep["confidence"],
            source="cleanup", source_batch=m["batch_file"], agent=keep["agent"],
            n_replicates=keep["n_replicates"], replicate_conflict=keep["replicate_conflict"],
            replicate_verdicts=keep["replicate_verdicts"], verdict_raw=keep["verdict"],
            detail=keep["detail"], rationale=keep["rationale"],
            packet_file="cleanup_batches/" + m["batch_file"],
            cleanup_pass="cleanup_ruling", active=1, superseded_by="",
            cleanup_reason=m["cleanup_reason"], original_class=m["original_class"]))

    out_cols = full_cols + [c for c in ("cleanup_pass", "active", "superseded_by",
                                        "cleanup_reason", "original_class") if c not in full_cols]
    ledger = sorted(prior + cleanup_rows,
                    key=lambda r: (r["case_class"], r["case_id"], str(r.get("cleanup_pass"))))
    write_csv(os.path.join(OUT, "rulings_full.csv"), out_cols, ledger)

    # ---------------- one-ruling-per-ORI check over the LIVE ledger
    # The first pass dispatched some ORIs under both the published_zero and the token_reporter
    # framing. The cleanup builder merged the 68 whose verdicts contradicted on the ACCEPT/REJECT
    # axis plus 30 same-ORI pairs; pairs that agreed on that axis were never merged and so still
    # carry two live rulings. They are outside the cleanup population — no cleanup ruling speaks to
    # them — so they are flagged and queued, not silently deduped or resolved here.
    live_by_ori = collections.defaultdict(list)
    for r in prior + cleanup_rows:
        if r.get("active", 1):
            for o in split_ids(r["oris"]):
                live_by_ori[o].append(r)
    ZT_LIVE = ZT | {"token_zero_shaped"}
    ori_multi = {o: [r for r in v if r["case_class"] in ZT_LIVE]
                 for o, v in live_by_ori.items()}
    ori_multi = {o: v for o, v in ori_multi.items() if len(v) > 1}
    partial_ids = {r["case_id"] for r in partial_overlap}
    residual_ori, flagged_cases = {}, set()
    for o, v in sorted(ori_multi.items()):
        if any(r["case_id"] in partial_ids for r in v):
            continue                                    # already queued as partial_ori_overlap
        residual_ori[o] = v
        flagged_cases.update(r["case_id"] for r in v)

    # ---------------- per-class registries: surviving pre-cleanup rows carried through verbatim
    superseded_ids = {r["case_id"] for r in prior if r["active"] == 0}
    routed = collections.Counter()

    def route(m):
        if m["class"] == "twin_shaped" or m["original_class"] == "ambiguous_twin":
            return "twins"
        return "zeros" if "published_zero" in m["original_class"] else "tokens"

    reg = {}
    for key, fn in (("twins", "twins_adjudicated.csv"), ("zeros", "zero_missing_adjudicated.csv"),
                    ("tokens", "token_reporters_adjudicated.csv")):
        rows, cols = read_registry(os.path.join(OUT, fn), MY_COLS)
        keep = []
        for r in rows:
            if r["case_id"] in superseded_ids or r["case_id"] in chosen:
                continue                                 # chosen: this script's own prior output
            if r.get("residue_pass") == "residue_ruling":
                continue                                 # pass-3 row; pass 3 rebuilds it
            r = dict(r, cleanup_pass="retained_pre_cleanup")
            for c in PASS_3_COLS:                        # stale pass-3 bookkeeping, re-derived there
                if c in r:
                    r[c] = ""
            if r["case_id"] in flagged_cases:            # two live rulings for one ORI
                r["needs_review"] = 1
            keep.append(r)
        reg[key] = (fn, cols + [c for c in ("cleanup_pass", "cleanup_reason", "original_class")
                                if c not in cols], keep)

    for cid, keep in sorted(chosen.items()):
        m = manifest[cid]
        k = route(m)
        routed[k] += 1
        act = downstream_action(keep["verdict_norm"], keep["detail"], keep["rationale"])
        row = dict(case_id=cid, state=m["state"], n_oris=len(split_ids(m["oris"])),
                   oris=";".join(split_ids(m["oris"])), verdict=keep["verdict_norm"],
                   downstream_action=act, arm=keep["arm"], confidence=keep["confidence"],
                   mass_hint=m["mass_hint"], replicate_conflict=keep["replicate_conflict"],
                   needs_review=int(keep["verdict_norm"] == "unclear_escalate"
                                    or keep["replicate_conflict"] or cid in QC_FLAGS),
                   detail=(keep["detail"] or keep["verdict"])[:1200],   # collator's cap; full text in rulings_full
                   believable_months=believable_months(keep["detail"], keep["rationale"]),
                   repair_value_hint=(repair_value(keep["detail"])
                                      if act in ("treat_missing_repair", "partial_year_uplift") else ""),
                   provenance_kind="cleanup_merged_packet", canonical_ori="", footprint="",
                   cleanup_pass="cleanup_ruling", cleanup_reason=m["cleanup_reason"],
                   original_class=m["original_class"])
        reg[k][2].append(row)

    for key, (fn, cols, rows) in reg.items():
        rows.sort(key=lambda r: -mass_f(r.get("mass_hint", "")))
        write_csv(os.path.join(OUT, fn), cols, rows)

    # ---------------- supervisor escalations
    esc = []

    def esc_row(cid, kind, keep=None, extra=""):
        m = manifest.get(cid, {})
        pk = packets.get(cid, {})
        v = keep["verdict_norm"] if keep else ""
        return dict(
            case_id=cid, kind=kind, state=m.get("state", ""), oris=";".join(split_ids(m.get("oris", ""))),
            mass_hint=m.get("mass_hint", ""), original_class=m.get("original_class", ""),
            cleanup_reason=m.get("cleanup_reason", ""), verdict=v,
            confidence=(keep or {}).get("confidence", ""), arm=(keep or {}).get("arm", ""),
            agent=(keep or {}).get("agent", ""),
            downstream_action=(downstream_action(v, (keep or {}).get("detail"), (keep or {}).get("rationale"))
                               if keep else "dispatch_again"),
            alt_rulings=(keep or {}).get("alt_rulings", "") or extra,
            settling_record=settling_record((keep or {}).get("detail"), (keep or {}).get("rationale")),
            ruling_detail=(keep or {}).get("detail", ""), ruling_rationale=(keep or {}).get("rationale", ""),
            evidence_summary=evidence_summary(pk.get("evidence", "")),
            packet_file=pk.get("_file", "cleanup_batches/" + m.get("batch_file", "")))

    for cid, keep in sorted(chosen.items()):
        if keep["verdict_norm"] == "unclear_escalate":
            esc.append(esc_row(cid, "cleanup_explicit_escalate", keep))
        elif keep["replicate_conflict"]:
            esc.append(esc_row(cid, "cleanup_replicate_conflict", keep))
    for cid in unruled:
        esc.append(esc_row(cid, "cleanup_case_unruled", None,
                           extra="dispatched in the cleanup pass; no ruling returned by any arm"))
    for cid, note in sorted(QC_FLAGS.items()):
        r = esc_row(cid, "qc_flagged_for_reread", chosen.get(cid))
        r["settling_record"] = note
        esc.append(r)
    for r in partial_overlap:
        covered = [o for o in split_ids(r["oris"]) if o in tz_oris or o in tw_oris]
        esc.append(dict(
            case_id=r["case_id"], kind="partial_ori_overlap_not_superseded", state=r["state"],
            oris=r["oris"], mass_hint=r["mass_hint"], original_class=r["case_class"],
            cleanup_reason="", verdict=r["verdict"], confidence=r["confidence"], arm=r["arm"],
            agent=r["agent"], downstream_action="reconcile_before_apply",
            alt_rulings=("pre-cleanup row left standing: only " + ",".join(covered) +
                         " of its ORIs were re-adjudicated in the cleanup pass, so superseding it "
                         "would drop the rest; the cleanup ruling for the shared ORI is authoritative "
                         "for that ORI only"),
            settling_record="", ruling_detail=r["detail"], ruling_rationale=r["rationale"],
            evidence_summary="", packet_file=r["packet_file"]))

    ACCEPT_ACTS = {"retain_zero", "accept_observed"}
    REJECT_ACTS = {"treat_missing_repair", "partial_year_uplift"}
    for o, v in sorted(residual_ori.items()):
        acts = {downstream_action(r["verdict"], r["detail"], r["rationale"]) for r in v}
        st = {("ACCEPT" if a in ACCEPT_ACTS else "REJECT" if a in REJECT_ACTS else "FLAG")
              for a in acts}
        agree = ("both_accept" if st == {"ACCEPT"} else
                 ("both_reject_same_repair" if len(acts) == 1 else "both_reject_DIFFERENT_repair")
                 if st == {"REJECT"} else "disposition_conflict_" + "/".join(sorted(st)))
        top = max(v, key=lambda r: mass_f(r["mass_hint"]))
        esc.append(dict(
            case_id=" + ".join(r["case_id"] for r in v), kind="same_ori_double_framing_outside_cleanup",
            state=top["state"], oris=o, mass_hint=top["mass_hint"],
            original_class="+".join(sorted({r["case_class"] for r in v})), cleanup_reason="",
            verdict=";".join(sorted({r["verdict"] for r in v})), confidence="", arm="",
            agent="", downstream_action=("dedupe_identical_action" if len(acts) == 1
                                         else "reconcile_before_apply"),
            alt_rulings=" || ".join(
                f"{r['case_id']}({r['case_class']}/{r['arm']}/{r['confidence']}): {r['verdict']} -> "
                f"{downstream_action(r['verdict'], r['detail'], r['rationale'])}" for r in v),
            settling_record=(agree + ": " +
                             ("repairs agree, so this is pure duplication of one ORI across two "
                              "framings and either row can be dropped" if len(acts) == 1 else
                              "the two framings imply DIFFERENT repairs for the same agency-year; "
                              "needs one merged-packet ruling like the cleanup pass")),
            ruling_detail=" || ".join((r["detail"] or "")[:900] for r in v),
            ruling_rationale=" || ".join((r["rationale"] or "")[:900] for r in v),
            evidence_summary="", packet_file=";".join(sorted({r["packet_file"] for r in v}))))

    ESC_COLS = ["case_id", "kind", "state", "oris", "mass_hint", "original_class", "cleanup_reason",
                "verdict", "confidence", "arm", "agent", "downstream_action", "alt_rulings",
                "settling_record", "ruling_detail", "ruling_rationale", "evidence_summary", "packet_file"]
    esc.sort(key=lambda r: (r["kind"], -mass_f(r["mass_hint"])))
    write_csv(os.path.join(OUT, "supervisor_escalations.csv"), ESC_COLS, esc)

    # ---------------- pre-cleanup escalation queue: mark each row's fate
    esc_rows, esc_cols = read_registry(os.path.join(OUT, "escalations.csv"), MY_COLS)
    still_open = 0
    for r in esc_rows:
        owner = ""
        for o in split_ids(r["oris"]):
            if o in ori_owner and ori_owner[o] in chosen:
                owner = ori_owner[o]
                break
        if owner:
            k = chosen[owner]
            r["resolved_by_cleanup"] = 1
            r["resolution"] = f"{owner}:{k['verdict_norm']}({k['arm']}/{k['confidence']})"
        else:
            r["resolved_by_cleanup"] = 0
            live = [x for o in split_ids(r["oris"]) for x in live_by_ori.get(o, [])
                    if x["case_id"] != r["case_id"]]
            if r["case_id"] in manifest:
                r["resolution"] = "still open - see supervisor_escalations.csv"
            elif live:
                r["resolution"] = ("outside the cleanup population; this ORI does carry a live "
                                   "ruling under another case_id: " +
                                   ";".join(sorted({f"{x['case_id']}({x['case_class']}):{x['verdict']}"
                                                    for x in live})))
            else:
                r["resolution"] = "outside the cleanup population and no live ruling for this ORI"
            still_open += 1
    # a pre-cleanup escalation with no ruling of its own, no cleanup ruling, and no live ruling on
    # the SAME question for its ORI is a real coverage hole -- promote it to the live queue
    late = []
    for r in esc_rows:
        if r["resolved_by_cleanup"] or r["verdict"] or r["case_id"] in manifest:
            continue
        oris = split_ids(r["oris"])
        if any(x["case_class"] in ZT_LIVE for o in oris for x in live_by_ori.get(o, [])):
            continue
        late.append(dict(
            case_id=r["case_id"], kind="unadjudicated_no_zero_token_ruling", state=r["state"],
            oris=r["oris"], mass_hint=r["mass_hint"], original_class=r["case_class"],
            cleanup_reason="", verdict="", confidence="", arm="", agent="",
            downstream_action="dispatch_again", alt_rulings=r["replicate_verdicts"],
            settling_record=("no arm ever ruled this case_id, it was not in the cleanup population, "
                             "and no live published_zero/token_reporter ruling covers its ORI -- the "
                             "only live ruling is a twin-identity one, which answers a different "
                             "question. The 2024 agency-year has no disposition."),
            ruling_detail=r["detail"], ruling_rationale=r["rationale"], evidence_summary="",
            packet_file=r["packet_file"]))
    if late:
        esc.extend(late)
        esc.sort(key=lambda r: (r["kind"], -mass_f(r["mass_hint"])))
        write_csv(os.path.join(OUT, "supervisor_escalations.csv"), ESC_COLS, esc)

    write_csv(os.path.join(OUT, "escalations.csv"),
              esc_cols + ["resolved_by_cleanup", "resolution"], esc_rows)

    # ---------------- validations
    problems = []
    in_pop_multi = {o: [f"{r['case_id']}:{r['verdict']}" for r in v]
                    for o, v in ori_multi.items() if o in ori_owner}
    for o, v in in_pop_multi.items():
        if not any(cid in partial_ids for cid in [x.split(":")[0] for x in v]):
            problems.append(f"ORI {o} is in the cleanup population but still carries >1 live ruling: {v}")

    def stance(v):
        return {"genuine_zero_year": "ACCEPT", "genuine_low_crime": "ACCEPT",
                "misread_missing": "REJECT", "token_reporting_flag": "REJECT"}.get(v, "OTHER")

    xclass = []
    for o, v in live_by_ori.items():
        st = {stance(r["verdict"]) for r in v}
        if {"ACCEPT", "REJECT"} <= st:
            xclass.append((o, [f"{r['case_id']}({r['case_class']}):{r['verdict']}" for r in v]))

    summary = dict(
        cleanup_rulings_loaded=len(snap["rulings"]),
        cleanup_cases_ruled=len(chosen),
        cleanup_cases_dispatched=len(manifest),
        cleanup_cases_unruled=unruled,
        replicate_conflicts=sorted(conflicts),
        verdicts=collections.Counter(k["verdict_norm"] for k in chosen.values()),
        verdicts_by_arm=collections.Counter((k["arm"], k["verdict_norm"]) for k in chosen.values()),
        confidence=collections.Counter(k["confidence"] for k in chosen.values()),
        downstream=collections.Counter(downstream_action(k["verdict_norm"], k["detail"], k["rationale"])
                                       for k in chosen.values()),
        routed_to=dict(routed),
        prior_rows=len(prior), prior_superseded=len(superseded_ids),
        prior_retained=len(prior) - len(superseded_ids),
        ledger_rows=len(ledger), ledger_active=sum(1 for r in ledger if r.get("active", 1)),
        registry_totals={k: len(v[2]) for k, v in reg.items()},
        supervisor_escalations=collections.Counter(r["kind"] for r in esc),
        oris_with_no_zero_token_disposition=[r["case_id"] for r in esc
                                            if r["kind"] == "unadjudicated_no_zero_token_ruling"],
        pre_cleanup_escalations=len(esc_rows),
        pre_cleanup_resolved=sum(1 for r in esc_rows if r["resolved_by_cleanup"]),
        pre_cleanup_still_open=still_open,
        residual_cross_class_ori_contradictions=xclass,
        oris_double_framed_outside_cleanup=len(residual_ori),
        oris_double_framed_actions_agree=sum(
            1 for v in residual_ori.values()
            if len({downstream_action(r["verdict"], r["detail"], r["rationale"]) for r in v}) == 1),
        oris_double_framed_triage=collections.Counter(
            r["settling_record"].split(":")[0] for r in esc
            if r["kind"] == "same_ori_double_framing_outside_cleanup"),
        oris_double_framed_detail={o: [f"{r['case_id']}:{r['verdict']}"
                                      f"->{downstream_action(r['verdict'], r['detail'], r['rationale'])}"
                                      for r in v] for o, v in sorted(residual_ori.items())},
        oris_multi_live_in_cleanup_population=in_pop_multi,
        problems=problems,
        qc_sample=sorted(random.Random(args.qc_seed).sample(sorted(chosen), min(args.qc_sample, len(chosen)))),
    )
    summary = {k: (dict(("|".join(kk) if isinstance(kk, tuple) else str(kk), vv) for kk, vv in v.items())
                   if isinstance(v, collections.Counter) else v) for k, v in summary.items()}
    json.dump(summary, open(os.path.join(OUT, "cleanup_apply_summary.json"), "w"), indent=1)

    if args.selftest:
        selftest()

    print(json.dumps({k: summary[k] for k in (
        "cleanup_rulings_loaded", "cleanup_cases_ruled", "cleanup_cases_unruled", "replicate_conflicts",
        "verdicts", "downstream", "routed_to", "prior_superseded", "prior_retained",
        "ledger_rows", "ledger_active", "registry_totals", "supervisor_escalations",
        "pre_cleanup_resolved", "pre_cleanup_still_open", "oris_double_framed_outside_cleanup",
        "oris_double_framed_actions_agree", "residual_cross_class_ori_contradictions",
        "problems", "qc_sample")}, indent=1))


def selftest():
    """The derivation helpers here are copies of the collator's. Re-derive the pre-cleanup rows and
    assert the copies still reproduce the columns the collator wrote."""
    full = {r["case_id"]: r for r in csv.DictReader(open(os.path.join(OUT, "rulings_full.csv")))}
    bad = []
    for fn, cols in (("zero_missing_adjudicated.csv", ("downstream_action", "believable_months")),
                     ("token_reporters_adjudicated.csv", ("downstream_action", "believable_months",
                                                          "repair_value_hint")),
                     ("twins_adjudicated.csv", ("downstream_action",))):
        for r in csv.DictReader(open(os.path.join(OUT, fn))):
            if r.get("cleanup_pass") == "cleanup_ruling":
                continue
            src = full.get(r["case_id"])
            if not src:
                continue
            act = downstream_action(src["verdict"], src["detail"], src["rationale"])
            got = {"downstream_action": act,
                   "believable_months": believable_months(src["detail"], src["rationale"]),
                   "repair_value_hint": (repair_value(src["detail"])
                                         if act in ("treat_missing_repair", "partial_year_uplift") else "")}
            for c in cols:
                if c in r and r[c] != got[c]:
                    bad.append((fn, r["case_id"], c, r[c], got[c]))
    print(f"selftest: {len(bad)} derivation mismatches on pre-cleanup rows")
    for b in bad[:10]:
        print("  ", b)
    assert not bad, "copied derivation helpers have drifted from the collator"


if __name__ == "__main__":
    main()
