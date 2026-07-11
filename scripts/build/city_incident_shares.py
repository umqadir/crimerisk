from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_shares import CityIncidentShareBuildConfig, write_v2_city_incident_shares
from crimerisk.paths import RepoPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical V2 city incident share surface.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet",
    )
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if the cached canonical city-share artifact looks current.",
    )
    parser.add_argument(
        "--force-source-refresh",
        action="store_true",
        help="Refresh raw city-source caches before rebuilding the canonical city-share artifact.",
    )
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    summary = write_v2_city_incident_shares(
        paths=paths,
        out_path=args.out,
        config=CityIncidentShareBuildConfig(
            year_start=int(args.year_start),
            year_end=int(args.year_end),
            force_rebuild=bool(args.force_rebuild),
            force_source_refresh=bool(args.force_source_refresh),
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
