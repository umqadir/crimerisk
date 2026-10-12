"""Compare a release city's hand-written feed loader with its replacement contract feed.

    uv run python scripts/diagnostics/compare_city_feed_replacement.py --city mesa --years 2020-2025 \
        [--offenses robbery,burglary,larceny,motor_vehicle_theft] [--out state/review/packets/city/mesa/share_admission]

Builds both stage-7 share surfaces for the same years (the hand-written loader from `_city_impls`, its
packet files read from `state/review/packets/city/<city>/`; the contract through
`city_contracts._contract_share_surface`) and writes, per year and offense: each side's located count,
their ratio, and the total variation distance between the two block-group and tract distributions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from crimerisk import city_contracts as cc  # noqa: E402
from crimerisk import city_shares  # noqa: E402
from crimerisk.paths import RepoPaths  # noqa: E402


def tvd(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0.0)
    return float((joined["a"] / joined["a"].sum() - joined["b"] / joined["b"].sum()).abs().sum() / 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", required=True)
    parser.add_argument("--years", default="2020-2025")
    parser.add_argument("--offenses", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    start, end = (int(y) for y in args.years.split("-"))
    paths = RepoPaths.from_repo_root(REPO)
    impl = city_shares._city_impls(paths)[args.city]
    if impl["share_fn"] is cc._contract_share_surface:
        raise SystemExit(f"{args.city} has no hand-written loader to compare against")
    packet = REPO / "state" / "review" / "packets" / "city" / args.city
    kwargs = {k: (packet / v.name if isinstance(v, Path) and v.parent.name == args.city and v.suffix == ".csv" else v)
              for k, v in impl["kwargs"].items()}
    kwargs.update(start_year=start, end_year=end)
    old = impl["share_fn"](**kwargs)
    new = cc._contract_share_surface(contract_path=cc.contract_dir(REPO) / f"{args.city}.json",
                                     cache_dir=paths.data_dir / "city-incidents" / args.city, tiger_dir=paths.data_dir / "tiger_bg",
                                     bg_crosswalk_parquet=paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                                     start_year=start, end_year=end)
    offenses = [o for o in args.offenses.split(",") if o] or sorted(set(old["offense"]) | set(new["offense"]))
    rows = []
    for offense in offenses:
        for year in range(start, end + 1):
            a = old[(old["offense"] == offense) & (old["year"].astype(int) == year)]
            b = new[(new["offense"] == offense) & (new["year"].astype(int) == year)]
            if a.empty or b.empty:
                rows.append({"offense": offense, "year": year, "loader": int(a["incident_count"].sum()), "contract": int(b["incident_count"].sum())})
                continue
            bg_a = a.groupby(a["block_group_geoid"].astype(str))["incident_count"].sum()
            bg_b = b.groupby(b["block_group_geoid"].astype(str))["incident_count"].sum()
            rows.append({"offense": offense, "year": year, "loader": int(bg_a.sum()), "contract": int(bg_b.sum()),
                         "contract_over_loader": float(bg_b.sum() / bg_a.sum()),
                         "tvd_block_group": tvd(bg_a, bg_b),
                         "tvd_tract": tvd(bg_a.groupby(bg_a.index.str[:11]).sum(), bg_b.groupby(bg_b.index.str[:11]).sum())})
    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else REPO / "state" / "review" / "packets" / "city" / args.city / "share_admission"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "loader_vs_contract.csv", index=False)
    print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
