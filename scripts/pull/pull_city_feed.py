"""Pull a contract city feed into the stage-7 cache.

    uv run python scripts/pull/pull_city_feed.py --city houston_hpd_tx [--year-start 2018] [--year-end 2025]

Reads `configs/city_feeds/<city>.json` when it exists, else the city's endpoints in
`configs/city_source_registry.csv` (every field, filtered by date only), and downloads every source
for the year range with `crimerisk.city_contracts.download` into `data/city-incidents/<city>/`, the
directory and file names stage 7 reads for a share-admitted contract feed. Writes
`pull_manifest.json` beside the cache (sources, rows per source, sha256, time of the pull).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "agents"))

from crimerisk import city_contracts as cc  # noqa: E402


def sources_for(city: str) -> tuple[list[dict], str]:
    contract = cc.contract_dir(REPO) / f"{city}.json"
    if contract.exists():
        return json.loads(contract.read_text())["sources"], str(contract.relative_to(REPO))
    import onboard_feed  # noqa: PLC0415 - the registry reader lives with the onboarding runner

    return onboard_feed.probe_sources(onboard_feed.registry_row(city)), "configs/city_source_registry.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", action="append", required=True)
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2025)
    args = parser.parse_args()
    for city in [c for arg in args.city for c in arg.split(",") if c]:
        sources, origin = sources_for(city)
        cache = REPO / "data" / "city-incidents" / city
        entries = []
        for source in sources:
            frame = cc.download({"sources": [source]}, args.year_start, args.year_end, cache)
            digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:12]
            path = cache / f"raw_{source['slug']}_{args.year_start}_{args.year_end}_{digest}.parquet"
            entries.append({"slug": source["slug"], "endpoint": source["api_endpoint"], "where": source.get("where", ""),
                            "rows": int(len(frame)), "file": path.name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = {"city_key": city, "sources_from": origin, "year_start": args.year_start, "year_end": args.year_end,
                    "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sources": entries}
        (cache / "pull_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"{city}: {sum(e['rows'] for e in entries):,} rows from {len(entries)} source(s) -> {cache.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
