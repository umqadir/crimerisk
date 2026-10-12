"""Out-of-time check for promoting a truth_only contract feed to a share feed.

    uv run python scripts/diagnostics/feed_share_holdout.py --city houston_hpd_tx --run feeds-2025.3 \
        [--holdout-year 2025] [--candidate ../repo/state/candidates/rc4-2025.2.1]

For each offense the run admits, the feed's located incidents of the admitted years before the holdout
year, pooled as stage 7 pools them (summed counts per block group), are scored against the feed's own
holdout-year distribution, next to the published candidate surface (expected counts of the block groups
whose EB jurisdiction is the feed's primary jurisdiction) and residential population. Score: total
variation distance at block group and at tract, on the candidate's block groups of that jurisdiction.
Writes `share_holdout_<year>.csv`, `qa.csv` (the share-admission QA table from the packet's
`evidence/qa_offense_year.csv`: in-year mapped rows, footprint rows, official total) and
`truth_admission.json` (the run's truth ruling per offense) to the packet's `share_admission/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from crimerisk import city_contracts as cc  # noqa: E402

PACKETS = REPO / "state" / "review" / "packets" / "city"


def tvd(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0.0)
    return float((joined["a"] / joined["a"].sum() - joined["b"] / joined["b"].sum()).abs().sum() / 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--holdout-year", type=int, default=2025)
    parser.add_argument("--candidate", default=str(REPO.parent / "repo" / "state" / "candidates" / "rc4-2025.2.1"))
    args = parser.parse_args()
    case = REPO / "state" / "review" / "runs" / args.run / args.city
    packet = PACKETS / args.city
    contract = json.loads((packet / "evidence" / "contract.json").read_text())
    registry = json.loads((packet / "evidence" / "registry_row.json").read_text())
    primary = registry["primary_jurisdiction_id"]
    result = json.loads((case / "result.json").read_text())
    proposals = {json.loads(p.read_text())["rows"][0]["offense"]: json.loads(p.read_text())["rows"][0]
                 for p in case.glob("proposal_admission_*.json")}
    # The run's truth ruling per offense (the admissions row's note is truncated; the batch record is not).
    rulings = {}
    for record in sorted((case.parent / "_admission_batch").glob("admission*.json")):
        output = json.loads(record.read_text()).get("output", {})
        for c in output.get("cities", []):
            if c["city_key"] == args.city:
                rulings = {o["offense"]: o for o in c["offenses"]}
    raw = pd.read_parquet(packet / "raw_cache.parquet")
    incidents = cc.assign_block_groups(cc.run_contract(contract, raw, year_start=2018, year_end=args.holdout_year),
                                       contract["geography"]["state_fips"], REPO / "data" / "tiger_bg", boundary=True)
    incidents = incidents[incidents["block_group_geoid"].notna()].copy()
    incidents["block_group_geoid"] = incidents["block_group_geoid"].astype(str).str.zfill(12)
    offenses = result["admitted_offenses"]
    columns = ["block_group_geoid", "eb_jurisdiction_id", "population_2025"] + [f"expected_count_{o}" for o in offenses]
    surface = pd.read_parquet(Path(args.candidate) / "crimerisk_block_group_2025_ags_core.parquet", columns=columns)
    surface = surface[surface["eb_jurisdiction_id"].eq(primary)].set_index("block_group_geoid")
    rows = []
    for offense in offenses:
        excluded = rulings.get(offense, {}).get("excluded_years", [])
        own = incidents[(incidents["offense"] == offense) & ~incidents["year"].isin(excluded)]
        own = own[own["block_group_geoid"].isin(surface.index)]
        truth = own[own["year"].eq(args.holdout_year)].groupby("block_group_geoid").size()
        past = own[own["year"].lt(args.holdout_year)]
        direct = past.groupby("block_group_geoid").size()
        current = surface[f"expected_count_{offense}"].astype(float)
        population = surface["population_2025"].astype(float)
        row = {"city_key": args.city, "offense": offense, "holdout_year": args.holdout_year, "excluded_years": excluded,
               "past_years": sorted(int(y) for y in past["year"].unique()), "past_incidents": int(len(past)),
               "holdout_incidents": int(truth.sum()), "block_groups": int(len(surface))}
        if truth.sum() == 0 or direct.sum() == 0:
            rows.append(row)
            continue
        for name, pred in (("direct_feed", direct), ("current_surface", current), ("population", population)):
            pred = pred.reindex(surface.index, fill_value=0.0)
            row[f"tvd_bg_{name}"] = tvd(pred, truth.reindex(surface.index, fill_value=0))
            row[f"tvd_tract_{name}"] = tvd(pred.groupby(pred.index.str[:11]).sum(),
                                           truth.reindex(surface.index, fill_value=0).groupby(surface.index.str[:11]).sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    out = packet / "share_admission"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"share_holdout_{args.holdout_year}.csv", index=False)
    qa = pd.read_csv(packet / "evidence" / "qa_offense_year.csv")
    qa.drop(columns=["located_rows"]).rename(columns={"footprint_rows": "located_rows"})[["year", "offense", "mapped_rows", "located_rows", "official_total", "official_source"]].to_csv(
        out / "qa.csv", index=False)
    (out / "truth_admission.json").write_text(json.dumps(
        {o: {"admission_mode": proposals[o].get("admission_mode", ""), "excluded_years": rulings.get(o, {}).get("excluded_years", []),
             "reason": rulings.get(o, {}).get("reason", ""), "adjudication": proposals[o].get("adjudication", "")} for o in sorted(proposals)},
        indent=1) + "\n")
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
