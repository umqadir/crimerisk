"""Download the fixed-URL Census inputs the build reads: population estimates, TIGER/Line, block assignment file.

    uv run python scripts/pull/pull_static_inputs.py [popest] [tiger] [baf] [--force]

With no group named, all groups are pulled. Existing non-empty files are kept unless --force is given.

| Group | Files | Staged at |
|---|---|---|
| popest | co-est2025-alldata.csv, co-est2024-alldata.csv | data/Census-PopEst-2020-2025/, data/Census-PopEst-2020-2024/ |
| tiger | TIGER 2020 TABBLOCK20, PLACE, COUSUB, BG, TRACT per release state; COUNTY, AIANNH, AITSN (US); TIGER 2023 BG for CT | data/tiger_tabblock20/, data/tiger_places/, data/tiger_cousub/, data/tiger_bg/, data/tiger_tracts/, data/tiger_counties/, data/tiger_aiannh/, data/tiger_aitsn/ |
| baf | 2020 block assignment file, Pennsylvania | data/Census-BAF-2020/ |
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.required_inputs import RELEASE_STATE_FIPS  # noqa: E402


POPEST = "https://www2.census.gov/programs-surveys/popest/datasets"
TIGER2020 = "https://www2.census.gov/geo/tiger/TIGER2020"
TIGER_LAYERS = {
    "TABBLOCK20": ("tiger_tabblock20", "tabblock20"),
    "PLACE": ("tiger_places", "place"),
    "COUSUB": ("tiger_cousub", "cousub"),
    "BG": ("tiger_bg", "bg"),
    "TRACT": ("tiger_tracts", "tract"),
}


def _targets(group: str) -> list[tuple[str, Path]]:
    data = REPO_ROOT / "data"
    if group == "popest":
        return [
            (f"{POPEST}/2020-2025/counties/totals/co-est2025-alldata.csv", data / "Census-PopEst-2020-2025" / "co-est2025-alldata.csv"),
            (f"{POPEST}/2020-2024/counties/totals/co-est2024-alldata.csv", data / "Census-PopEst-2020-2024" / "co-est2024-alldata.csv"),
        ]
    if group == "tiger":
        out = []
        for layer, (directory, suffix) in TIGER_LAYERS.items():
            for fips in RELEASE_STATE_FIPS:
                name = f"tl_2020_{fips}_{suffix}.zip"
                out.append((f"{TIGER2020}/{layer}/{name}", data / directory / name))
        out.append((f"{TIGER2020}/COUNTY/tl_2020_us_county.zip", data / "tiger_counties" / "tl_2020_us_county.zip"))
        out.append((f"{TIGER2020}/AIANNH/tl_2020_us_aiannh.zip", data / "tiger_aiannh" / "tl_2020_us_aiannh.zip"))
        out.append((f"{TIGER2020}/AITSN/tl_2020_us_aitsn.zip", data / "tiger_aitsn" / "tl_2020_us_aitsn.zip"))
        out.append(("https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_09_bg.zip", data / "tiger_bg" / "tl_2023_09_bg.zip"))
        return out
    if group == "baf":
        return [(
            "https://www2.census.gov/geo/docs/maps-data/data/baf2020/BlockAssign_ST42_PA.zip",
            data / "Census-BAF-2020" / "BlockAssign_ST42_PA.zip",
        )]
    raise ValueError(group)


def _download(url: str, out_path: Path, *, retries: int = 4) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "crimerisk-pull/1.0"})
            with urlopen(request, timeout=600) as response, tmp_path.open("wb") as sink:
                shutil.copyfileobj(response, sink, length=1024 * 1024)
            if tmp_path.stat().st_size <= 0:
                raise RuntimeError(f"Empty download from {url}")
            tmp_path.replace(out_path)
            return
        except Exception:
            tmp_path.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("groups", nargs="*", choices=["popest", "tiger", "baf"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for group in args.groups or ["popest", "tiger", "baf"]:
        targets = _targets(group)
        fetched = 0
        for url, out_path in targets:
            if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
                continue
            _download(url, out_path)
            fetched += 1
            print(f"{out_path.relative_to(REPO_ROOT)}", flush=True)
        print(f"{group}: {fetched} downloaded, {len(targets) - fetched} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
