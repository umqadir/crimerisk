from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


API_BASE = "https://api.usa.gov/crime/fbi/cde"
DEFAULT_STATES = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]
PRIORITY_STATES = ["FL", "CA", "NY", "LA", "GA", "MS", "TX", "NJ"]


def _fetch_state_agencies(*, state_abbr: str, api_key: str, timeout: int = 120) -> dict:
    response = requests.get(
        f"{API_BASE}/agency/byStateAbbr/{state_abbr}",
        params={"API_KEY": api_key},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected FBI agency roster payload for {state_abbr}: {type(payload).__name__}")
    return payload


def _flatten_payload(*, state_abbr: str, payload: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for county_group, agencies in payload.items():
        if not isinstance(agencies, list):
            continue
        for agency in agencies:
            if not isinstance(agency, dict):
                continue
            row = dict(agency)
            row["county_group"] = county_group
            row["request_state_abbr"] = state_abbr
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "request_state_abbr",
                "county_group",
                "ori",
                "agency_name",
                "agency_type_name",
                "is_nibrs",
                "nibrs_start_date",
                "latitude",
                "longitude",
                "state_abbr",
                "state_name",
            ]
        )
    frame = pd.DataFrame(rows)
    if "ori" in frame.columns:
        frame["ori"] = frame["ori"].astype("string")
    if "state_abbr" in frame.columns:
        frame["state_abbr"] = frame["state_abbr"].astype("string")
    if "request_state_abbr" in frame.columns:
        frame["request_state_abbr"] = frame["request_state_abbr"].astype("string")
    if "nibrs_start_date" in frame.columns:
        frame["nibrs_start_date"] = frame["nibrs_start_date"].astype("string")
    for col in ["latitude", "longitude"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull official FBI CDE agency rosters.")
    parser.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "FBI-CDE-Agency-Rosters-2024")
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--priority-only", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("FBI_API_KEY")
    if not api_key:
        raise SystemExit("FBI_API_KEY is not set in the environment.")

    out_dir = args.out_dir.resolve()
    raw_dir = out_dir / "raw"
    parsed_dir = out_dir / "parsed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    states = [s.upper() for s in (PRIORITY_STATES if args.priority_only else args.states)]
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for idx, state_abbr in enumerate(states, start=1):
        raw_path = raw_dir / f"{state_abbr}.json"
        if args.skip_existing and raw_path.exists():
            payload = json.loads(raw_path.read_text())
        else:
            payload = _fetch_state_agencies(state_abbr=state_abbr, api_key=api_key)
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        frame = _flatten_payload(state_abbr=state_abbr, payload=payload)
        frame["pull_year"] = 2024
        frames.append(frame)
        min_nibrs_start = None
        max_nibrs_start = None
        if "nibrs_start_date" in frame.columns and frame["nibrs_start_date"].notna().any():
            nonnull = frame["nibrs_start_date"].dropna().astype(str)
            if len(nonnull):
                min_nibrs_start = nonnull.min()
                max_nibrs_start = nonnull.max()
        summary_rows.append(
            {
                "state_abbr": state_abbr,
                "agency_rows": int(len(frame)),
                "unique_oris": int(frame["ori"].nunique()) if "ori" in frame.columns else 0,
                "nibrs_true_rows": int(frame["is_nibrs"].fillna(False).astype(bool).sum()) if "is_nibrs" in frame.columns else 0,
                "min_nibrs_start_date": min_nibrs_start,
                "max_nibrs_start_date": max_nibrs_start,
                "raw_json_path": str(raw_path),
            }
        )
        print(f"[{idx}/{len(states)}] {state_abbr}: {len(frame):,} rows")
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    suffix = "priority_states_2024" if args.priority_only else "2024"
    combined_csv = parsed_dir / f"agency_rosters_{suffix}.csv"
    combined_parquet = parsed_dir / f"agency_rosters_{suffix}.parquet"
    summary_csv = parsed_dir / f"agency_rosters_{suffix}_summary.csv"
    if not combined.empty:
        combined.to_csv(combined_csv, index=False)
        combined.to_parquet(combined_parquet, index=False)
    else:
        pd.DataFrame().to_csv(combined_csv, index=False)
        pd.DataFrame().to_parquet(combined_parquet, index=False)
    summary.to_csv(summary_csv, index=False)

    print(
        {
            "states": states,
            "combined_rows": int(len(combined)),
            "combined_csv": str(combined_csv),
            "combined_parquet": str(combined_parquet),
            "summary_csv": str(summary_csv),
        }
    )


if __name__ == "__main__":
    main()
