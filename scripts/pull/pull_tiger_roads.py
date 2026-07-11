#!/usr/bin/env python3
"""
Download TIGER roads zips and aggregate first-pass road length metrics to block groups.

Usage:
    python scripts/pull/pull_tiger_roads.py
    python scripts/pull/pull_tiger_roads.py --states 06 12 36 --overwrite-download

Outputs:
    data/tiger_roads/tl_2020_<county>_roads.zip
    data/roads/state_block_groups/<state>.parquet
    data/roads/parsed/block_group_road_metrics.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.covariates.roads import (
    RoadAggregationConfig,
    build_roads_download_inventory,
    build_state_block_group_road_metrics,
    download_tiger_roads_zip,
    expected_bg_zip_path,
    infer_county_fips_from_block_groups,
    infer_state_fips_from_bg_dir,
    load_roads_download_manifest,
    load_block_groups,
    sync_roads_download_manifest_with_cache,
)


DEFAULT_ROADS_DIR = REPO_ROOT / "data" / "tiger_roads"
DEFAULT_BG_DIR = REPO_ROOT / "data" / "tiger_bg"
DEFAULT_STATE_OUT_DIR = REPO_ROOT / "data" / "roads" / "state_block_groups"
DEFAULT_OUT_PATH = REPO_ROOT / "data" / "roads" / "parsed" / "block_group_road_metrics.parquet"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "tiger_roads" / "download_manifest_2020.json"


def _select_states_for_run(
    *,
    states: list[str],
    start_state: str | None,
    batch_size: int | None,
    state_out_dir: Path,
    overwrite_state_parquet: bool,
) -> list[str]:
    selected = [str(state).zfill(2) for state in states]
    if start_state is not None:
        start = str(start_state).zfill(2)
        selected = [state for state in selected if state >= start]
    if batch_size is not None:
        pending = (
            selected
            if overwrite_state_parquet
            else [state for state in selected if not (state_out_dir / f"{state}.parquet").exists()]
        )
        selected = (pending or selected)[:batch_size]
    return selected


def _should_attempt_download(entry: dict[str, object], *, failed_only: bool) -> bool:
    if bool(entry.get("has_valid_zip")):
        return False
    if not failed_only:
        return True
    if entry.get("manifest_status") == "downloaded":
        return True
    if not bool(entry.get("manifest_present")):
        return False
    return bool(entry.get("is_retryable_failure"))


def _format_state_inventory_summary(
    *,
    state: str,
    counties: list[str],
    inventory: dict[str, dict[str, object]],
    state_out_path: Path,
    failed_only: bool,
) -> str:
    expected_counties = len(counties)
    valid_zip_count = sum(1 for county in counties if inventory[county]["has_valid_zip"])
    manifest_downloaded_count = sum(1 for county in counties if inventory[county]["manifest_status"] == "downloaded")
    retryable_failure_count = sum(1 for county in counties if inventory[county]["is_retryable_failure"])
    missing_manifest_count = sum(1 for county in counties if not inventory[county]["manifest_present"])
    failed_only_targets = sum(1 for county in counties if _should_attempt_download(inventory[county], failed_only=True))
    parts = [
        f"{state}: counties={expected_counties}",
        f"valid_zips={valid_zip_count}/{expected_counties}",
        f"manifest_downloaded={manifest_downloaded_count}",
        f"retryable_tail={retryable_failure_count}",
        f"missing_manifest={missing_manifest_count}",
        f"state_parquet={'yes' if state_out_path.exists() else 'no'}",
    ]
    if failed_only:
        parts.append(f"failed_only_targets={failed_only_targets}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads-dir", type=Path, default=DEFAULT_ROADS_DIR)
    parser.add_argument("--bg-dir", type=Path, default=DEFAULT_BG_DIR)
    parser.add_argument("--state-out-dir", type=Path, default=DEFAULT_STATE_OUT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument("--overwrite-state-parquet", action="store_true")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--download-sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-download-attempts", type=int, default=6)
    parser.add_argument("--retry-max-sleep-seconds", type=float, default=3600.0)
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--start-state", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = RoadAggregationConfig()
    args.roads_dir.mkdir(parents=True, exist_ok=True)
    args.state_out_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    known_states = [str(state).zfill(2) for state in infer_state_fips_from_bg_dir(args.bg_dir, year=cfg.tiger_year)]
    if not known_states:
        print(f"No state BG zips found under {args.bg_dir}")
        return 1
    requested_states = [str(state).zfill(2) for state in (args.states or known_states)]
    states = _select_states_for_run(
        states=requested_states,
        start_state=args.start_state,
        batch_size=args.batch_size,
        state_out_dir=args.state_out_dir,
        overwrite_state_parquet=args.overwrite_state_parquet,
    )

    manifest = load_roads_download_manifest(args.manifest_path)
    failed_states: list[str] = []
    built_states: list[str] = []

    if states:
        print(f"Selected {len(states)} state(s) for this run: {states}")
    else:
        print("No states selected for processing after applying the current filters.")

    for idx, state_fips in enumerate(states, start=1):
        state = str(state_fips).zfill(2)
        bg_zip = expected_bg_zip_path(args.bg_dir, state_fips=state, year=cfg.tiger_year)
        state_out_path = args.state_out_dir / f"{state}.parquet"

        if not bg_zip.exists():
            print(f"[{idx}/{len(states)}] {state}: missing BG zip ({bg_zip})")
            failed_states.append(state)
            continue

        try:
            counties = infer_county_fips_from_block_groups(load_block_groups(bg_zip))
        except Exception as exc:
            print(f"[{idx}/{len(states)}] {state}: FAILED county scan ({type(exc).__name__}: {exc})")
            failed_states.append(state)
            continue

        manifest = sync_roads_download_manifest_with_cache(
            roads_dir=args.roads_dir,
            manifest_path=args.manifest_path,
            counties=counties,
            year=cfg.tiger_year,
        )
        inventory = build_roads_download_inventory(
            counties=counties,
            roads_dir=args.roads_dir,
            manifest=manifest,
            year=cfg.tiger_year,
        )
        print(
            f"[{idx}/{len(states)}] "
            f"{_format_state_inventory_summary(state=state, counties=counties, inventory=inventory, state_out_path=state_out_path, failed_only=args.failed_only)}"
        )

        if state_out_path.exists() and not args.overwrite_state_parquet:
            df = pd.read_parquet(state_out_path)
            print(f"  state {state}: using cached state parquet with {len(df):,} block groups")
            continue

        print(f"  state {state}: ensuring {len(counties)} TIGER county roads zips...", flush=True)
        roads_zips: list[Path] = []
        cached_count = 0
        downloaded_count = 0
        failed_count = 0
        skipped_count = 0
        for county_fips in counties:
            county = str(county_fips).zfill(5)
            entry = inventory[county]
            out_path = Path(entry["path"])
            if entry["has_valid_zip"] and not args.overwrite_download:
                roads_zips.append(out_path)
                cached_count += 1
                continue
            if not _should_attempt_download(entry, failed_only=args.failed_only):
                skipped_count += 1
                continue
            try:
                roads_zips.append(
                    download_tiger_roads_zip(
                        county_fips=county,
                        roads_dir=args.roads_dir,
                        year=cfg.tiger_year,
                        overwrite=args.overwrite_download,
                        manifest_path=args.manifest_path,
                        download_sleep_seconds=args.download_sleep_seconds,
                        max_attempts=args.max_download_attempts,
                        retry_max_sleep_seconds=args.retry_max_sleep_seconds,
                    )
                )
                manifest = load_roads_download_manifest(args.manifest_path)
                inventory.update(
                    build_roads_download_inventory(
                        counties=[county],
                        roads_dir=args.roads_dir,
                        manifest=manifest,
                        year=cfg.tiger_year,
                    )
                )
                downloaded_count += 1
            except Exception as exc:
                print(f"  county {county}: FAILED download ({type(exc).__name__}: {exc})")
                failed_count += 1
                manifest = load_roads_download_manifest(args.manifest_path)
                inventory.update(
                    build_roads_download_inventory(
                        counties=[county],
                        roads_dir=args.roads_dir,
                        manifest=manifest,
                        year=cfg.tiger_year,
                    )
                )
                continue
        print(
            f"  state {state} download summary: cached={cached_count} downloaded={downloaded_count} skipped={skipped_count} failed={failed_count}"
        )
        if failed_count:
            failed_states.append(state)
            continue

        print(f"  state {state}: aggregating roads to block groups...", end=" ", flush=True)
        try:
            df = build_state_block_group_road_metrics(
                state_fips=state,
                roads_zips=roads_zips,
                bg_zip=bg_zip,
                cfg=cfg,
            )
        except Exception as exc:
            print(f"FAILED build ({type(exc).__name__}: {exc})")
            failed_states.append(state)
            continue

        df.to_parquet(state_out_path, index=False)
        built_states.append(state)
        print(f"wrote {len(df):,} block groups")

    frames: list[pd.DataFrame] = []
    available_states: list[str] = []
    for state in known_states:
        state_out_path = args.state_out_dir / f"{state}.parquet"
        if not state_out_path.exists():
            continue
        frames.append(pd.read_parquet(state_out_path))
        available_states.append(state)

    if not frames:
        print("No state road metrics are available to combine.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined["bg_id"] = combined["bg_id"].astype(str).str.zfill(12)
    combined["tract_id"] = combined["tract_id"].astype(str).str.zfill(11)
    combined["state_fips"] = combined["state_fips"].astype(str).str.zfill(2)
    combined["county_fips"] = combined["county_fips"].astype(str).str.zfill(3)
    combined.to_parquet(args.out, index=False)

    missing_state_parquets = [state for state in known_states if state not in available_states]
    print(f"\nSaved {len(combined):,} block groups to {args.out} from {len(available_states)}/{len(known_states)} state parquets")
    if missing_state_parquets:
        preview = missing_state_parquets[:10]
        suffix = " ..." if len(missing_state_parquets) > 10 else ""
        print(f"National output is partial; missing state parquets: {preview}{suffix}")
    print(f"States built this run: {built_states}")
    if failed_states:
        print(f"Failed states this run: {failed_states}")
    print(f"Manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
