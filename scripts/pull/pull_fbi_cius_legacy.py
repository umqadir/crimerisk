from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.jurisdiction_reference import STATE_ABBR_BY_FIPS, STATE_NAME_BY_FIPS
from crimerisk.paths import get_paths


LEGACY_BASE = "https://ucr.fbi.gov/crime-in-the-u.s/{year}/crime-in-the-u.s.-{year}/tables/table-{table}/table-{table}-state-cuts/{slug}.xls/output.xls"
OUTPUT_PATTERNS = {
    8: "Table_8_Offenses_Known_to_Law_Enforcement_{state}_by_City_{year}.xls",
    9: "Table_9_Offenses_Known_to_Law_Enforcement_{state}_by_University_and_College_{year}.xls",
    11: "Table_11_Offenses_Known_to_Law_Enforcement_{state}_by_State_Tribal_and_Other_Agencies_{year}.xls",
}
EXCLUDE_ABBRS = {"AK", "HI", "PR", "AS", "GU", "MP", "VI", "CZ"}


def _state_slug(name: str) -> str:
    return name.lower().replace(".", "").replace(" ", "-")


def _scope_states() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for fips, name in sorted(STATE_NAME_BY_FIPS.items()):
        abbr = STATE_ABBR_BY_FIPS.get(fips)
        if not abbr or abbr in EXCLUDE_ABBRS:
            continue
        out.append((abbr, name))
    return out


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "crimerisk-v2/1.0"})
    with urlopen(req, timeout=600) as response, dest.open("wb") as sink:
        sink.write(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull legacy FBI CIUS state-cut workbooks for 2018-2019.")
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2019])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = get_paths()
    states = _scope_states()
    total = 0
    downloaded = 0
    skipped = 0
    failed: list[dict[str, object]] = []

    for year in args.years:
        raw_dir = paths.data_dir / "FBI-CIUS-Annual" / str(year) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for table in (8, 9, 11):
            for _abbr, state_name in states:
                filename = OUTPUT_PATTERNS[table].format(
                    state=state_name.replace(" ", "_"),
                    year=year,
                )
                dest = raw_dir / filename
                total += 1
                if dest.exists() and not args.force:
                    skipped += 1
                    continue
                url = LEGACY_BASE.format(
                    year=year,
                    table=table,
                    slug=_state_slug(state_name),
                )
                try:
                    _download(url, dest)
                except HTTPError as exc:
                    failed.append(
                        {
                            "year": year,
                            "table": table,
                            "state": state_name,
                            "url": url,
                            "status": exc.code,
                        }
                    )
                    print(f"failed {exc.code} {url}")
                    continue
                downloaded += 1
                print(f"downloaded {url} -> {dest}")

    print(
        {
            "years": list(args.years),
            "states": len(states),
            "attempted_files": total,
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": len(failed),
        }
    )
    if failed:
        for row in failed[:50]:
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
