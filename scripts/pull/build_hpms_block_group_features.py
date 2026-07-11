from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.covariates.hpms import (
    STATE_ABBR_BY_FIPS,
    HpmsBuildConfig,
    build_state_block_group_hpms_metrics,
)
from crimerisk.covariates.roads import infer_state_fips_from_bg_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull FHWA HPMS full-network traffic exposure and aggregate to block groups.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--bg-dir", type=Path, default=REPO_ROOT / "data" / "tiger_bg")
    parser.add_argument("--state-out-dir", type=Path, default=REPO_ROOT / "data" / "HPMS" / "state_block_groups")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "HPMS" / "block_group_hpms_2024.parquet")
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    raw_states = [str(s).zfill(2) for s in args.states] if args.states else infer_state_fips_from_bg_dir(args.bg_dir, year=2020)
    states = [state for state in raw_states if state in STATE_ABBR_BY_FIPS]
    args.state_out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    cfg = HpmsBuildConfig(year=int(args.year), chunk_size=int(args.chunk_size), timeout_seconds=int(args.timeout_seconds))
    for idx, state_fips in enumerate(states, start=1):
        state_fips = str(state_fips).zfill(2)
        bg_zip = args.bg_dir / f"tl_2020_{state_fips}_bg.zip"
        if not bg_zip.exists():
            continue
        out_path = args.state_out_dir / f"{state_fips}.parquet"
        print(f"hpms: [{idx}/{len(states)}] state={state_fips} build start", flush=True)
        frame = build_state_block_group_hpms_metrics(state_fips=state_fips, bg_zip=bg_zip, cfg=cfg)
        frame.to_parquet(out_path, index=False)
        print(f"hpms: [{idx}/{len(states)}] state={state_fips} build done rows={len(frame):,}", flush=True)
        frames.append(frame)

    if not frames:
        raise SystemExit("No HPMS frames were built.")
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"rows={len(out):,} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
