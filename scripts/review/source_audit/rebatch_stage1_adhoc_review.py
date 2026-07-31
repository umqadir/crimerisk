#!/usr/bin/env python3
"""Rebatch the Stage-1 ad-hoc review swarm after the first (partial) run.

Inputs
  state/qa/stage1_adhoc_review/batches/batch_NNN.json   original 88 packets, 1054 cases
  state/qa/stage1_adhoc_review/case_index.csv           authoritative case manifest
  <workflow journal>/journal.jsonl                      rulings banked by the first run
  analysis_scratch/stage1_pilot_result.json             the 45 pilot cases (already gold-ruled)

What it does
  1. extracts every ruling from the prior run's journal -> banked_rulings.json
  2. drops those cases, plus any case whose ORI set touches a pilot case's ORI set
  3. re-renders every surviving packet with the COMPUTED SIGNALS prose block
     (same generator as state/qa/review_tier_trial, recomputed from the CURRENT
     state/observations/agency_year_observations.parquet)
  4. splits the survivors by routing tier:
       zero_batches/  published_zero            (cheap model)
       opus_batches/  ambiguous_twin + token_reporter
  5. writes gold_repair/batch_000.json with the 3 falsified pilot cases, enriched

Usage: uv run python scripts/review/source_audit/rebatch_stage1_adhoc_review.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "review" / "source_audit"))

import build_review_tier_trial_packets as T  # noqa: E402

ROOT = REPO / "state" / "qa" / "stage1_adhoc_review"
BATCHES = ROOT / "batches"
CASE_INDEX = ROOT / "case_index.csv"
PILOT = REPO / "analysis_scratch" / "stage1_pilot_result.json"
JOURNAL = Path(
    "/Users/uzairqadir/.claude/projects/"
    "-Users-uzairqadir-Projects-data-projects-national-crimerisk-clone/"
    "6b6a1654-17bb-4f3a-aa47-aa689bda8400/subagents/workflows/wf_5e933bc1-cb8/journal.jsonl"
)

BANKED_OUT = ROOT / "banked_rulings.json"
ZERO_DIR = ROOT / "zero_batches"
OPUS_DIR = ROOT / "opus_batches"
GOLD_REPAIR_DIR = ROOT / "gold_repair"
BATCH_SIZE = 12

GOLD_REPAIR_CASES = [
    "b2-MN0701000-elko-new-market",
    "b2-VT0110300-castleton",
    "b2-IL0190600-waterman",
]

ORI_HEAD_LINE = re.compile(r'^ORI ([A-Z0-9]{9}) "')


def packet_oris(evidence: str) -> set[str]:
    return {
        m.group(1)
        for ln in evidence.split("\n")
        for m in [ORI_HEAD_LINE.match(ln.strip())]
        if m
    }


# ------------------------------------------------------------------ step 1
def extract_banked() -> list[dict]:
    banked: list[dict] = []
    for line in JOURNAL.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") != "result":
            continue
        res = rec.get("result")
        if not isinstance(res, dict) or "rulings" not in res:
            continue
        for r in res["rulings"]:
            r = dict(r)
            r["_source_agent"] = rec.get("agentId")
            banked.append(r)
    return banked


def main() -> None:
    # ---- load the manifest and every original packet -------------------------
    index = list(csv.DictReader(CASE_INDEX.open()))
    cases: dict[str, dict] = {}
    for path in sorted(BATCHES.glob("batch_*.json")):
        for c in json.loads(path.read_text()):
            cases[c["case_id"]] = c
    assert len(cases) == len(index), f"packet/index mismatch {len(cases)} vs {len(index)}"
    meta = {r["case_id"]: r for r in index}

    # ---- step 1: bank the prior run's rulings --------------------------------
    banked = extract_banked()
    banked_valid = [b for b in banked if b["case_id"] in cases]
    banked_ids = {b["case_id"] for b in banked_valid}
    orphan = sorted({b["case_id"] for b in banked} - banked_ids)
    BANKED_OUT.write_text(
        json.dumps(
            {
                "source_journal": str(JOURNAL),
                "n_rulings": len(banked_valid),
                "n_dropped_non_case_rulings": len(orphan),
                "dropped_non_case_ruling_ids": orphan,
                "rulings": sorted(banked_valid, key=lambda r: r["case_id"]),
            },
            indent=2,
        )
        + "\n"
    )

    # ---- step 2: exclusions --------------------------------------------------
    pilot_cases = json.loads(PILOT.read_text())["results"][0]["cases"]
    pilot_oris: set[str] = set()
    for c in pilot_cases:
        pilot_oris |= packet_oris(c["evidence"])

    pilot_dupes = {
        cid
        for cid, c in cases.items()
        if packet_oris(c["evidence"]) & pilot_oris
    }

    remaining_ids = sorted(set(cases) - banked_ids - pilot_dupes)

    # ---- step 3: enrich ------------------------------------------------------
    work = [cases[cid] for cid in remaining_ids]
    gold_repair_src = {
        c["case_id"]: c for c in pilot_cases if c["case_id"] in GOLD_REPAIR_CASES
    }
    assert len(gold_repair_src) == 3, sorted(gold_repair_src)

    all_oris: set[str] = set()
    for c in work + list(gold_repair_src.values()):
        all_oris |= packet_oris(c["evidence"])
    print(f"loading panel for {len(all_oris)} ORIs ...", flush=True)
    panel = T.load_panel(all_oris)

    def enrich(c: dict) -> dict:
        sig = T.build_signals(c["evidence"], panel)
        return {
            "case_id": c["case_id"],
            "class": c["class"],
            "evidence": T.insert_signals(c["evidence"], sig),
        }

    enriched = [enrich(c) for c in work]

    # ---- step 4: split and write --------------------------------------------
    by_id = {c["case_id"]: c for c in enriched}
    zero = [by_id[cid] for cid in remaining_ids if meta[cid]["class"] == "published_zero"]
    opus = [
        by_id[cid]
        for cid in remaining_ids
        if meta[cid]["class"] in ("ambiguous_twin", "token_reporter")
    ]
    other = [
        cid
        for cid in remaining_ids
        if meta[cid]["class"] not in ("published_zero", "ambiguous_twin", "token_reporter")
    ]
    assert not other, f"unrouted classes: {other[:5]}"

    def write_batches(items: list[dict], out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("batch_*.json"):
            old.unlink()
        n = 0
        for i in range(0, len(items), BATCH_SIZE):
            n += 1
            (out_dir / f"batch_{n:03d}.json").write_text(
                json.dumps(items[i : i + BATCH_SIZE], indent=2) + "\n"
            )
        return n

    n_zero = write_batches(zero, ZERO_DIR)
    n_opus = write_batches(opus, OPUS_DIR)

    # ---- step 5: gold repair -------------------------------------------------
    GOLD_REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    for old in GOLD_REPAIR_DIR.glob("batch_*.json"):
        old.unlink()
    repair = [enrich(gold_repair_src[cid]) for cid in GOLD_REPAIR_CASES]
    (GOLD_REPAIR_DIR / "batch_000.json").write_text(json.dumps(repair, indent=2) + "\n")

    # ---- report --------------------------------------------------------------
    def mean_chars(xs):
        return sum(len(x["evidence"]) for x in xs) / len(xs) if xs else 0

    print(f"banked rulings kept   : {len(banked_valid)} (orphans dropped: {len(orphan)})")
    print(f"pilot-ORI dupes excl. : {len(pilot_dupes)} -> {sorted(pilot_dupes)}")
    print(f"remaining cases       : {len(remaining_ids)}")
    print(f"published_zero        : {len(zero)} in {n_zero} batches (mean {mean_chars(zero):.0f} chars)")
    print(f"twin+token            : {len(opus)} in {n_opus} batches (mean {mean_chars(opus):.0f} chars)")
    print(f"gold_repair           : {len(repair)} in 1 batch (mean {mean_chars(repair):.0f} chars)")


if __name__ == "__main__":
    main()
