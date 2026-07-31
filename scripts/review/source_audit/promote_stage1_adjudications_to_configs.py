#!/usr/bin/env python3
"""Promote the Stage-1 ad-hoc adjudication registries into `configs/` as first-class inputs.

`state/qa/stage1_adhoc_review/` is the AUDIT LEDGER: every ruling instance from all four passes,
including the superseded ones, plus the packets, the escalation queue and the per-pass summaries. It
stays there and it stays complete.

`configs/stage1_adjudications/` is the CONSUMED SURFACE: the same three per-class registries, live
rows only, with a provenance header the pipeline loader validates. Promoting them puts them where
every other registry the build reads already lives (`configs/reporting_regime_overrides.csv`,
`configs/overlap_footprint_overrides.csv`, ...), makes them visible in the repo rather than inside a
gitignored state tree, and gives the freshness machinery a path to depend on.

The two are kept in sync by re-running this script, and the loader fails closed when they drift:
the header records the source file's sha256 and the row count it wrote, and
`crimerisk.stage1_adjudications` re-checks the row count on every read.

What is promoted
----------------
* every live row of the three registries, unchanged apart from column selection
* `target_year`, added explicitly. The registries have no year column because every packet in the
  ad-hoc review was about the 2024 agency-year; leaving that implicit in a consumed input is how a
  registry silently starts applying to the wrong year.
* the long `detail` text, truncated. It is the reviewer's operative instruction and belongs with the
  row; the untruncated text is in the ledger.

Identity rulings carry no target year on purpose: identity is a property of the agency, not of a
year, and `apply_cross_lane_twin_ledger` moves the whole series when it re-keys an ORI.

Usage: uv run python scripts/review/source_audit/promote_stage1_adjudications_to_configs.py
       [--check]   verify configs/ matches the registries without writing
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "state/qa/stage1_adhoc_review")
DST = os.path.join(ROOT, "configs/stage1_adjudications")
GENERATOR = "scripts/review/source_audit/promote_stage1_adjudications_to_configs.py"
TARGET_YEAR = 2024
DETAIL_CAP = 1200

csv.field_size_limit(1 << 24)

# (filename, columns to promote, whether the rows are about one agency-year)
REGISTRIES = (
    ("twins_adjudicated.csv",
     ("case_id", "state", "n_oris", "oris", "verdict", "canonical_ori", "footprint",
      "downstream_action", "confidence", "needs_review", "detail"),
     False),
    ("zero_missing_adjudicated.csv",
     ("case_id", "state", "oris", "verdict", "downstream_action", "believable_months",
      "confidence", "needs_review", "detail"),
     True),
    ("token_reporters_adjudicated.csv",
     ("case_id", "state", "oris", "verdict", "downstream_action", "believable_months",
      "repair_value_hint", "confidence", "needs_review", "detail"),
     True),
)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(filename, columns, dated, *, generated_at):
    src = os.path.join(SRC, filename)
    with open(src) as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in columns if rows and c not in rows[0]]
    if missing:
        sys.exit(f"{src} is missing promoted columns {missing}")

    out_columns = list(columns)
    if dated:
        out_columns.insert(1, "target_year")

    body = []
    for r in rows:
        row = {c: (r.get(c) or "") for c in columns}
        row["detail"] = row["detail"][:DETAIL_CAP]
        if dated:
            row["target_year"] = TARGET_YEAR
        body.append(row)

    header = [
        f"# stage1_adjudications registry: {filename}",
        f"# source_registry: state/qa/stage1_adhoc_review/{filename}",
        "# source_pass: pass 4 (apply_stage1_final_rulings.py) landing on passes 1-3",
        "# rows: live rulings only (the per-class registries hold no superseded rows); the full",
        "#   audit ledger with every superseded instance is state/qa/stage1_adhoc_review/rulings_full.csv",
        f"# target_year: {TARGET_YEAR if dated else 'not_year_scoped_identity_ruling'}",
        f"# generated_by: {GENERATOR}",
        f"# generated_at: {generated_at}",
        f"# source_sha256: {sha256_of(src)}",
        f"# rows_written: {len(body)}",
    ]

    import io
    buf = io.StringIO()
    buf.write("\n".join(header) + "\n")
    w = csv.DictWriter(buf, fieldnames=out_columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for row in body:
        w.writerow(row)
    return buf.getvalue(), len(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare configs/ against the registries and exit nonzero on drift")
    args = ap.parse_args()

    os.makedirs(DST, exist_ok=True)
    stale = []
    for filename, columns, dated in REGISTRIES:
        dest = os.path.join(DST, filename)
        # generated_at is excluded from the drift comparison; a re-run with no content change must
        # not read as drift just because the clock moved.
        existing_stamp = ""
        if os.path.isfile(dest):
            for line in open(dest):
                if line.startswith("# generated_at: "):
                    existing_stamp = line.split(": ", 1)[1].strip()
                    break
        stamp = existing_stamp or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        text, n = render(filename, columns, dated, generated_at=stamp)
        current = open(dest).read() if os.path.isfile(dest) else None
        if current == text:
            print(f"unchanged  {filename}  ({n} rows)")
            continue
        fresh, n = render(filename, columns, dated,
                          generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        if args.check:
            stale.append(filename)
            print(f"DRIFT      {filename}  ({n} rows in the registry)")
            continue
        with open(dest, "w", newline="") as f:
            f.write(fresh)
        print(f"written    {filename}  ({n} rows)")
    if stale:
        sys.exit(f"configs/stage1_adjudications is stale for {stale}; re-run without --check")


if __name__ == "__main__":
    main()
