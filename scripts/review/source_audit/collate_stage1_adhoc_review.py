#!/usr/bin/env python3
"""Collate the class-routed Stage 1 review swarm into registry-shaped CSVs.

Sources merged:
  1. banked prior Opus rulings  state/qa/stage1_adhoc_review/banked_rulings.json      (arm=opus)
  2. 45 pilot golds             analysis_scratch/stage1_pilot_result.json groups 1,3,5,6,7 (arm=gold)
     -- 3 cases replaced by the gold_repair adjudications from this run
  3. this run's rulings         subagent workflow journal wf_3fc61f49-339             (arm=luna/sol/opus)
"""
import glob
import json, csv, re, collections, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "state/qa/stage1_adhoc_review")
SP = os.environ.get("SCRATCH", OUT)
# the swarm's rulings live in the subagent workflow journal (one JSON result object per agent)
W = os.environ.get(
    "STAGE1_SWARM_JOURNAL_DIR",
    os.path.expanduser("~/.claude/projects/-Users-uzairqadir-Projects-data-projects-national-crimerisk-clone/"
                       "6b6a1654-17bb-4f3a-aa47-aa689bda8400/subagents/workflows/wf_3fc61f49-339"))


def load_run_rulings(wdir):
    """Return every ruling the swarm emitted, tagged with its arm.

    Arm is read off each agent's dispatch prompt: the Luna and Sol runners shell out to
    `codex exec --model gpt-5.6-{luna,sol}` over a batch range; everything else is a
    Claude Opus subagent reading one opus_batches/batch_NNN.json directly.
    """
    src = {}
    for p in sorted(glob.glob(os.path.join(wdir, "agent-*.jsonl"))):
        aid = os.path.basename(p)[6:-6]
        c = json.loads(open(p).readline())["message"]["content"]
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        arm = ("luna" if "Luna-arm" in c else "sol" if "Sol-arm" in c
               else "gold_repair" if "Gold-repair" in c else "opus")
        batches = sorted(set(re.findall(r"(zero_batches|opus_batches|gold_repair)/batch_(\d+)", c)))
        src[aid] = (arm, ";".join(f"{a}/{b}" for a, b in batches))
    out = []
    for line in open(os.path.join(wdir, "journal.jsonl")):
        d = json.loads(line)
        if d.get("type") != "result":
            continue
        res = d["result"]
        if not (isinstance(res, dict) and "rulings" in res):
            continue
        arm, batches = src.get(d["agentId"], ("opus", ""))
        for x in res["rulings"]:
            out.append(dict(x, _agent=d["agentId"], _arm=arm, _batches=batches))
    return out

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


# ---------------------------------------------------------------- case metadata
idx = {r["case_id"]: r for r in csv.DictReader(open(os.path.join(OUT, "case_index.csv")))}
pilot = json.load(open(os.path.join(ROOT, "analysis_scratch/stage1_pilot_result.json")))["results"]
pilot_packets = {c["case_id"]: c for c in pilot[0]["cases"]}

# packets on disk, for QC re-reads
packets = {}
for sub in ("batches", "zero_batches", "opus_batches", "gold_repair"):
    d = os.path.join(OUT, sub)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        for c in json.load(open(os.path.join(d, fn))):
            packets.setdefault(c["case_id"], dict(c, _file=f"{sub}/{fn}"))

EST = re.compile(r"EST2024\s*=\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


def case_meta(cid):
    if cid in idx:
        r = idx[cid]
        return dict(cls=r["class"], subclass=r["subclass"], state=r["state"],
                    n_oris=r["n_oris"], oris=r["oris"], mass=r["mass_hint"],
                    provenance_kind=r["provenance_kind"], batch_file=r["batch_file"])
    p = pilot_packets.get(cid) or packets.get(cid) or {}
    cls = p.get("class", "")
    ev = p.get("evidence", "")
    oris = ";".join(sorted(set(t for t in re.findall(r"\b[A-Z]{2}[0-9A-Z]{7}\b", ev) if sum(c.isdigit() for c in t) >= 3))[:8])
    m = EST.search(ev)
    st = cid.split("-")[1][:2] if len(cid.split("-")) > 1 else ""
    if not re.fullmatch(r"[A-Z]{2}", st):
        st = ""
    return dict(cls=cls, subclass=cid.split("-")[0], state=st, n_oris="",
                oris=oris, mass=(m.group(1).replace(",", "") if m else ""),
                provenance_kind="", batch_file=p.get("_file", "analysis_scratch/stage1_pilot_result.json"))


# ---------------------------------------------------------------- source 1: banked opus
records = []          # one dict per ruling instance
for r in json.load(open(os.path.join(OUT, "banked_rulings.json")))["rulings"]:
    records.append(dict(case_id=r["case_id"], arm="opus", verdict=r["verdict"],
                        detail=r.get("detail", ""), rationale=r.get("rationale", ""),
                        confidence=r.get("confidence", ""), source="banked_opus",
                        source_batch="", agent=r.get("_source_agent", "")))

# ---------------------------------------------------------------- source 3: this run (journal)
raw = load_run_rulings(W)
repair = {}
for o in raw:
    if o["case_id"].startswith("DISPATCH-BUG"):
        continue
    if o["_arm"] == "gold_repair":
        repair.setdefault(o["case_id"], []).append(o)
        continue
    records.append(dict(case_id=o["case_id"], arm=o["_arm"], verdict=o["verdict"],
                        detail=o.get("detail", ""), rationale=o.get("rationale", ""),
                        confidence=o.get("confidence", ""), source="run",
                        source_batch=o["_batches"], agent=o["_agent"]))

# ---------------------------------------------------------------- source 2: pilot golds (+repairs)
gold = []
for g in (1, 3, 5, 6, 7):
    gold += pilot[g]["rulings"]
repaired_ids = set(repair)
for r in gold:
    cid = r["case_id"]
    if cid in repaired_ids:
        continue                                        # replaced below
    records.append(dict(case_id=cid, arm="gold", verdict=r["verdict"], detail="",
                        rationale=r.get("rationale", ""), confidence=r.get("confidence", ""),
                        source="pilot_gold", source_batch="", agent=""))
for cid, lst in repair.items():
    o = lst[0]                                          # two identical-intent repair runs; take first
    records.append(dict(case_id=cid, arm="gold", verdict=o["verdict"], detail=o.get("detail", ""),
                        rationale=o.get("rationale", ""), confidence=o.get("confidence", ""),
                        source="gold_repair", source_batch="gold_repair/batch_000",
                        agent=o["_agent"], n_repair_runs=len(lst),
                        repair_alt_verdict=(norm_verdict(lst[1]["verdict"]) if len(lst) > 1 else "")))

# ---------------------------------------------------------------- collapse to one row per (case, arm)
by = collections.defaultdict(list)
for r in records:
    by[(r["case_id"], r["arm"])].append(r)

rows = []
for (cid, arm), lst in by.items():
    lst.sort(key=lambda r: (r["source"] != "gold_repair", r["agent"]))
    keep = lst[0]
    verdicts = [norm_verdict(x["verdict"]) for x in lst]
    conflict = len(set(verdicts)) > 1
    m = case_meta(cid)
    rows.append(dict(
        case_id=cid, case_class=m["cls"], subclass=m["subclass"], state=m["state"],
        n_oris=m["n_oris"], oris=m["oris"], mass_hint=m["mass"],
        arm=arm, verdict=norm_verdict(keep["verdict"]), verdict_raw=keep["verdict"],
        confidence=keep["confidence"], source=keep["source"], source_batch=keep["source_batch"],
        agent=keep["agent"], n_replicates=len(lst),
        replicate_conflict=int(conflict),
        replicate_verdicts=";".join(sorted(set(verdicts))) if conflict else "",
        detail=keep["detail"], rationale=keep["rationale"],
        packet_file=m["batch_file"],
    ))
rows.sort(key=lambda r: (r["case_class"], r["case_id"]))

FULL_COLS = ["case_id", "case_class", "subclass", "state", "n_oris", "oris", "mass_hint",
             "arm", "verdict", "confidence", "source", "source_batch", "agent",
             "n_replicates", "replicate_conflict", "replicate_verdicts",
             "verdict_raw", "detail", "rationale", "packet_file"]
with open(os.path.join(OUT, "rulings_full.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FULL_COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---------------------------------------------------------------- extraction helpers
_ORI_CAND = re.compile(r"\b[A-Z]{2}[0-9A-Z]{5,7}\b")


class ORI:
    """ORI9/ORI7 finder: 2 alpha + alnum tail carrying >=3 digits (kills all-caps words)."""

    @staticmethod
    def _ok(tok):
        return sum(ch.isdigit() for ch in tok) >= 3

    @classmethod
    def search(cls, text, ):
        for m in _ORI_CAND.finditer(text or ""):
            if cls._ok(m.group(0)):
                return m
        return None

    @classmethod
    def findall(cls, text):
        return [m.group(0) for m in _ORI_CAND.finditer(text or "") if cls._ok(m.group(0))]
FP_PATTERNS = [
    re.compile(r"\b\d{2}:[a-z_]+(?::[a-z_]+)?(?::\d+)?"),
    re.compile(r"place(?:\s+FIPS)?\s+(\d{5,7})"),
    re.compile(r"cousub\s+(\d{8,12})"),
    re.compile(r"FIPS\s+(\d{5})"),
    re.compile(r"statewide[_ ]overlap[_ ]layer"),
    re.compile(r"state[_ ]nonmunicipal[_ ]remainder"),
    re.compile(r"(?:municipal|nonmunicipal|county|special)[a-z_]*:[a-z_]+:\d+"),
    re.compile(r"cousubs?\b[^.]{0,60}?\b(\d{10})"),
]


def _fp_tokens(text):
    got = []
    for pat in FP_PATTERNS:
        for m in pat.finditer(text or ""):
            tok = m.group(0)
            if tok not in got:
                got.append(tok)
    return got


def footprint_of(text):
    """Prefer footprint tokens in sentences that talk about the footprint."""
    t = text or ""
    sents = [s for s in re.split(r"(?<=[.;])\s+", t)
             if re.search(r"footprint|crosswalk|CW=|place|cousub|layer|FIPS", s)]
    got = _fp_tokens(" ".join(sents)) or _fp_tokens(t)
    return " | ".join(got[:4])


def canonical_of(verdict, text):
    if verdict not in ("same_agency_merge", "superseded_ori"):
        return ""
    t = text or ""
    for kw in ("canonical ORI9 is", "canonical ORI is", "canonical is", "canonical"):
        i = t.find(kw)
        if i >= 0:
            m = ORI.search(t[i:i + 240])
            if m:
                return m.group(0)
    m = ORI.search(t)
    return m.group(0) if m else ""


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
KEEP_CUES = ("keep 0", "keep the", "retain 0", "accept the", "keep both", "no uplift", "defensible",
             "accept 2024", "retain")


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


MONTHS = re.compile(r"(\d{1,2})[- ]?(?:believable|reported|submitted|usable|month|mo\b)")


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


# the arms write repair magnitudes as "~N" ("repair ~6", "= ~21,650", "peer estimate ~4").
# Exclude "~N months" / "~N%" so a month count never masquerades as a count estimate.
REPAIR_VAL = re.compile(r"~\s*([0-9][0-9,]*(?:\.[0-9]+)?)(?!\s*(?:month|mo\b|%))")


def repair_value(detail):
    m = REPAIR_VAL.search(detail or "")
    return m.group(1).replace(",", "") if m else ""


def mass_f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


# ---------------------------------------------------------------- per-class registries
twins = [r for r in rows if r["case_class"] == "ambiguous_twin"]
zeros = [r for r in rows if r["case_class"] == "published_zero"]
toks = [r for r in rows if r["case_class"] == "token_reporter"]
others = [r for r in rows if r["case_class"] not in ("ambiguous_twin", "published_zero", "token_reporter")]

with open(os.path.join(OUT, "twins_adjudicated.csv"), "w", newline="") as f:
    cols = ["case_id", "state", "n_oris", "oris", "verdict", "canonical_ori", "footprint",
            "downstream_action", "arm", "confidence", "mass_hint", "replicate_conflict",
            "needs_review", "detail"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(twins, key=lambda r: -mass_f(r["mass_hint"])):
        txt = r["verdict_raw"] + " " + r["detail"]
        w.writerow(dict(r,
                        canonical_ori=canonical_of(r["verdict"], txt),
                        footprint=footprint_of(txt),
                        downstream_action=downstream_action(r["verdict"], r["detail"], r["rationale"]),
                        needs_review=int(r["verdict"] == "unclear_escalate" or r["replicate_conflict"]),
                        detail=(r["detail"] or r["verdict_raw"])[:1200]))

with open(os.path.join(OUT, "zero_missing_adjudicated.csv"), "w", newline="") as f:
    cols = ["case_id", "state", "oris", "provenance_kind", "verdict", "downstream_action",
            "believable_months", "arm", "confidence", "mass_hint", "replicate_conflict",
            "needs_review", "detail"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(zeros, key=lambda r: -mass_f(r["mass_hint"])):
        pk = idx.get(r["case_id"], {}).get("provenance_kind", "")
        w.writerow(dict(r, provenance_kind=pk,
                        downstream_action=downstream_action(r["verdict"], r["detail"], r["rationale"]),
                        believable_months=believable_months(r["detail"], r["rationale"]),
                        needs_review=int(r["verdict"] == "unclear_escalate" or r["replicate_conflict"]),
                        detail=(r["detail"] or r["verdict_raw"])[:1200]))

with open(os.path.join(OUT, "token_reporters_adjudicated.csv"), "w", newline="") as f:
    cols = ["case_id", "state", "oris", "verdict", "downstream_action", "believable_months",
            "repair_value_hint", "arm", "confidence", "mass_hint", "replicate_conflict",
            "needs_review", "detail"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(toks, key=lambda r: -mass_f(r["mass_hint"])):
        act = downstream_action(r["verdict"], r["detail"], r["rationale"])
        w.writerow(dict(r, downstream_action=act,
                        believable_months=believable_months(r["detail"], r["rationale"]),
                        repair_value_hint=(repair_value(r["detail"]) if act in ("treat_missing_repair", "partial_year_uplift") else ""),
                        needs_review=int(r["verdict"] == "unclear_escalate" or r["replicate_conflict"]),
                        detail=(r["detail"] or r["verdict_raw"])[:1200]))

# ---------------------------------------------------------------- escalations
esc = []
for r in rows:
    kinds = []
    if r["verdict"] == "unclear_escalate":
        kinds.append("explicit_escalate")
    if r["replicate_conflict"]:
        kinds.append("intra_arm_replication_conflict")
    for k in kinds:
        esc.append(dict(r, kind=k,
                        downstream_action=("escalate_hold" if k == "explicit_escalate"
                                           else downstream_action(r["verdict"], r["detail"], r["rationale"])),
                        detail=(r["detail"] or r["verdict_raw"])[:1500]))
# same ORI ruled in two classes with opposite dispositions -> the registry cannot apply both
def stance(v):
    if v in ("genuine_zero_year", "genuine_low_crime"):
        return "ACCEPT"
    if v in ("misread_missing", "token_reporting_flag"):
        return "REJECT"
    return "OTHER"


by_ori = collections.defaultdict(list)
for r in rows:
    o = [x for x in r["oris"].split(";") if x]
    if len(o) == 1:
        by_ori[o[0]].append(r)
xclass = 0
for ori, lst in by_ori.items():
    cl = {r["case_class"]: r for r in lst}
    if "published_zero" in cl and "token_reporter" in cl:
        z, t = cl["published_zero"], cl["token_reporter"]
        if {stance(z["verdict"]), stance(t["verdict"])} == {"ACCEPT", "REJECT"}:
            xclass += 1
            for a, b in ((z, t), (t, z)):
                esc.append(dict(a, kind="cross_class_same_ori_contradiction",
                                downstream_action="reconcile_before_apply",
                                replicate_verdicts=f"{b['case_class']}:{b['verdict']}({b['arm']},{b['confidence']}) via {b['case_id']}",
                                detail=(a["detail"] or a["verdict_raw"])[:1500]))

# uncovered index cases are escalations too (no adjudication exists)
covered = set(r["case_id"] for r in rows)
ori_ruled = collections.defaultdict(list)
for r in rows:
    for o in r["oris"].split(";"):
        if o:
            ori_ruled[o].append(f"{r['case_id']}({r['case_class']},{r['arm']}):{r['verdict']}")
for cid, r in idx.items():
    if cid not in covered:
        alt = []
        for o in r["oris"].split(";"):
            alt += ori_ruled.get(o, [])
        esc.append(dict(case_id=cid, case_class=r["class"], subclass=r["subclass"], state=r["state"],
                        oris=r["oris"], n_oris=r["n_oris"], mass_hint=r["mass_hint"], arm="",
                        verdict="", confidence="", kind="unadjudicated_no_ruling",
                        downstream_action=("same_ori_ruled_in_other_class" if alt else "dispatch_again"),
                        replicate_verdicts=" ; ".join(alt),
                        detail=("no ruling for this case_id; the same ORI is adjudicated under another "
                                "case_id/class framing (see replicate_verdicts)" if alt else
                                "case present in case_index.csv but no ruling returned by any arm"),
                        rationale="", packet_file=r["batch_file"], source="", agent=""))
with open(os.path.join(OUT, "escalations.csv"), "w", newline="") as f:
    cols = ["case_id", "case_class", "state", "oris", "mass_hint", "arm", "verdict", "confidence",
            "kind", "replicate_verdicts", "downstream_action", "packet_file", "detail", "rationale"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(esc, key=lambda r: (r["kind"], -mass_f(r.get("mass_hint", "")))):
        w.writerow(r)

# ---------------------------------------------------------------- summary json for the report
summ = dict(
    n_rows=len(rows),
    n_unique_cases=len(set(r["case_id"] for r in rows)),
    by_arm=collections.Counter(r["arm"] for r in rows),
    by_class=collections.Counter(r["case_class"] for r in rows),
    class_x_verdict_x_arm=collections.Counter((r["case_class"], r["verdict"], r["arm"]) for r in rows),
    by_conf=collections.Counter((r["arm"], r["confidence"]) for r in rows),
    n_escalations=len(esc),
    esc_kinds=collections.Counter(r["kind"] for r in esc),
    n_cross_class_ori_pairs=xclass,
    esc_by_class_arm=collections.Counter((r["kind"], r["case_class"], r["arm"]) for r in esc),
    downstream=collections.Counter((r["case_class"], downstream_action(r["verdict"], r["detail"], r["rationale"])) for r in rows),
    n_replicate_pairs=sum(1 for r in rows if r["n_replicates"] > 1),
    n_replicate_conflicts=sum(1 for r in rows if r["replicate_conflict"]),
    others=[(r["case_id"], r["case_class"], r["verdict"]) for r in others],
)
summ = {k: (dict((("|".join(kk) if isinstance(kk, tuple) else kk), vv) for kk, vv in v.items())
            if isinstance(v, collections.Counter) else v) for k, v in summ.items()}
json.dump(summ, open(os.path.join(SP, "stage1_collation_summary.json"), "w"), indent=1)

# top mass rows
top = sorted(rows, key=lambda r: -mass_f(r["mass_hint"]))[:40]
json.dump([{k: r[k] for k in ("case_id", "case_class", "state", "oris", "mass_hint", "arm", "verdict",
                              "confidence", "replicate_conflict")} for r in top],
          open(os.path.join(SP, "stage1_collation_top_mass.json"), "w"), indent=1)
print("rows", len(rows), "unique", len(set(r['case_id'] for r in rows)))
print("escalations", len(esc), collections.Counter(r['kind'] for r in esc))
print("by arm", collections.Counter(r['arm'] for r in rows))
print("by class", collections.Counter(r['case_class'] for r in rows))
