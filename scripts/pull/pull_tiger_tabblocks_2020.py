from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd


BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "tiger_tabblock20"


def _default_state_fips() -> list[str]:
    path = REPO_ROOT / "state" / "reference" / "jurisdiction_master.parquet"
    df = pd.read_parquet(path, columns=["state_fips", "state_abbr"])
    df["state_fips"] = df["state_fips"].astype("string").str.zfill(2)
    df["state_abbr"] = df["state_abbr"].astype("string").str.upper()
    df = df[df["state_fips"].notna()].copy()
    df = df[~df["state_abbr"].isin(["CZ"])].copy()
    return sorted(df["state_fips"].dropna().unique().tolist())


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, prefix=out_path.name, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--states", nargs="*", default=None)
    args = parser.parse_args()

    state_fips_values = args.states or _default_state_fips()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for state_fips in state_fips_values:
        state = str(state_fips).zfill(2)
        name = f"tl_2020_{state}_tabblock20.zip"
        out_path = out_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"skip {name}")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"download {name}")
        _download(url, out_path)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
