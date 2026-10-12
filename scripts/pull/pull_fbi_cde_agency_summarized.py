"""Pull agency-level monthly summarized offense counts from the FBI Crime Data Explorer.

The CIUS publication bundle only carries the agencies the FBI chose to print in the
Table 8/9/10/11 family.  The CDE `summarized/agency/{ori}/{offense}` route carries the
current master-file vintage for every participating agency, including the months that an
agency did and did not submit.  This script mirrors that route for one target year into a
dated raw directory under `data/`.

Output layout (`--out-dir`, default `data/FBI-CDE-Agency-Summarized-<year>-<stamp>`):

    raw/<STATE>.json.gz     one JSON object per state, keyed `<ORI>|<offense>`
    parsed/cde_agency_monthly_<year>.parquet
    parsed/pull_summary.json

The parsed parquet is long on `ori9 x offense x month` with `offenses`, `clearances`,
`population`, and `reported` (whether the CDE returned a value for that agency-month).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import gzip
import json
import os
from pathlib import Path
import sys
import threading
import time

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


API_BASE = "https://api.usa.gov/crime/fbi/cde"

# CDE offense slug -> the repository's offense name.
OFFENSE_SLUGS: dict[str, str] = {
    "homicide": "murder",
    "rape": "rape",
    "robbery": "robbery",
    "aggravated-assault": "aggravated_assault",
    "burglary": "burglary",
    "larceny": "larceny",
    "motor-vehicle-theft": "motor_vehicle_theft",
}

DEFAULT_STATES = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

_PRINT_LOCK = threading.Lock()


class Puller:
    def __init__(self, *, api_key: str, workers: int, max_retries: int, timeout: int):
        self.api_key = api_key
        self.workers = workers
        self.max_retries = max_retries
        self.timeout = timeout
        self._local = threading.local()
        self.calls = 0
        self.failures: list[tuple[str, str, str]] = []
        self._lock = threading.Lock()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.workers, pool_maxsize=self.workers
            )
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def fetch(self, *, ori: str, slug: str, year: int) -> dict | None:
        url = f"{API_BASE}/summarized/agency/{ori}/{slug}"
        params = {
            "from": f"01-{year}",
            "to": f"12-{year}",
            "API_KEY": self.api_key,
        }
        delay = 1.0
        last = ""
        for attempt in range(self.max_retries):
            try:
                response = self._session().get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # transient network failure
                last = type(exc).__name__
            else:
                if response.status_code == 200:
                    with self._lock:
                        self.calls += 1
                    try:
                        return response.json()
                    except ValueError:
                        last = "bad-json"
                elif response.status_code == 404:
                    with self._lock:
                        self.calls += 1
                    return None
                else:
                    last = str(response.status_code)
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)
        with self._lock:
            self.failures.append((ori, slug, last))
        return None


def _agency_series(payload: dict, block: str, suffix: str) -> dict[str, float] | None:
    """Pick the agency (not state/national) series out of a CDE response block."""
    section = payload.get("offenses", {}).get(block, {})
    if not isinstance(section, dict):
        return None
    candidates = [
        key
        for key in section
        if key.endswith(suffix)
        and not key.startswith("United States")
    ]
    # The state series is named "<State Name> Offenses"; the agency series is whatever
    # remains once the national and state rows are removed.  The response lists the
    # agency series last, so prefer the final candidate and fall back to the only one.
    if not candidates:
        return None
    populations = payload.get("populations", {}).get("population", {})
    agency_names = [
        name
        for name in populations
        if name != "United States" and f"{name} {suffix.strip()}" in section
    ]
    if agency_names:
        wanted = f"{agency_names[-1]} {suffix.strip()}"
        if wanted in section:
            return section[wanted]
    return section[candidates[-1]]


def _population(payload: dict) -> tuple[str | None, float | None]:
    populations = payload.get("populations", {}).get("population", {})
    names = [name for name in populations if name != "United States"]
    if not names:
        return None, None
    name = names[-1]
    series = populations.get(name) or {}
    values = [v for v in series.values() if v is not None]
    return name, (float(values[0]) if values else None)


def _rows_from_payload(
    *, payload: dict, ori: str, offense: str, year: int
) -> list[dict[str, object]]:
    actual_offenses = _agency_series(payload, "actuals", " Offenses")
    if actual_offenses is None:
        return []
    actual_clearances = _agency_series(payload, "actuals", " Clearances") or {}
    agency_name, population = _population(payload)
    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        key = f"{month:02d}-{year}"
        value = actual_offenses.get(key)
        rows.append(
            {
                "ori9": ori,
                "offense": offense,
                "year": year,
                "month": month,
                "offenses": None if value is None else float(value),
                "clearances": (
                    None
                    if actual_clearances.get(key) is None
                    else float(actual_clearances[key])
                ),
                "reported": value is not None,
                "population": population,
                "cde_agency_name": agency_name,
            }
        )
    return rows


def _state_oris(roster_dir: Path, state_abbr: str) -> list[str]:
    path = roster_dir / f"{state_abbr}.json"
    if not path.exists():
        raise SystemExit(f"Missing CDE roster for {state_abbr}: {path}")
    payload = json.loads(path.read_text())
    oris: list[str] = []
    for agencies in payload.values():
        if not isinstance(agencies, list):
            continue
        for agency in agencies:
            if isinstance(agency, dict) and agency.get("ori"):
                oris.append(str(agency["ori"]))
    return sorted(set(oris))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull FBI CDE agency-level monthly summarized offense counts."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--roster-dir",
        type=Path,
        default=REPO_ROOT / "data" / "FBI-CDE-Agency-Rosters-2025" / "raw",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit-agencies", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("FBI_API_KEY")
    if not api_key:
        raise SystemExit("FBI_API_KEY is not set in the environment.")

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else REPO_ROOT / "data" / f"FBI-CDE-Agency-Summarized-{args.year}-{stamp}"
    ).resolve()
    raw_dir = out_dir / "raw"
    parsed_dir = out_dir / "parsed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    puller = Puller(
        api_key=api_key,
        workers=args.workers,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    started = time.time()
    frames: list[pd.DataFrame] = []
    summary_states: list[dict[str, object]] = []
    for position, state_abbr in enumerate(args.states, start=1):
        state_abbr = state_abbr.upper()
        raw_path = raw_dir / f"{state_abbr}.json.gz"
        parsed_path = parsed_dir / f"{state_abbr}.parquet"
        if args.skip_existing and parsed_path.exists():
            frames.append(pd.read_parquet(parsed_path))
            with _PRINT_LOCK:
                print(f"[{position}/{len(args.states)}] {state_abbr} reused", flush=True)
            continue

        oris = _state_oris(args.roster_dir, state_abbr)
        if args.limit_agencies:
            oris = oris[: args.limit_agencies]
        jobs = [(ori, slug) for ori in oris for slug in OFFENSE_SLUGS]

        payloads: dict[str, dict] = {}
        rows: list[dict[str, object]] = []

        def work(job: tuple[str, str]) -> None:
            ori, slug = job
            payload = puller.fetch(ori=ori, slug=slug, year=args.year)
            if payload is None:
                return
            payloads[f"{ori}|{slug}"] = payload
            rows.extend(
                _rows_from_payload(
                    payload=payload,
                    ori=ori,
                    offense=OFFENSE_SLUGS[slug],
                    year=args.year,
                )
            )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, jobs))

        with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
            json.dump(payloads, handle)
        frame = pd.DataFrame(rows)
        frame.to_parquet(parsed_path, index=False)
        frames.append(frame)
        summary_states.append(
            {
                "state_abbr": state_abbr,
                "roster_agencies": len(oris),
                "requests": len(jobs),
                "payloads": len(payloads),
                "rows": int(len(frame)),
                "agencies_with_any_month": (
                    int(frame.loc[frame["reported"], "ori9"].nunique())
                    if len(frame)
                    else 0
                ),
            }
        )
        with _PRINT_LOCK:
            elapsed = time.time() - started
            print(
                f"[{position}/{len(args.states)}] {state_abbr} "
                f"agencies={len(oris)} payloads={len(payloads)} "
                f"rows={len(frame)} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    combined = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    combined_path = parsed_dir / f"cde_agency_monthly_{args.year}.parquet"
    combined.to_parquet(combined_path, index=False)

    summary = {
        "year": int(args.year),
        "pulled_at": datetime.now(UTC).isoformat(),
        "api_base": API_BASE,
        "states": summary_states,
        "total_rows": int(len(combined)),
        "total_agencies": (
            int(combined["ori9"].nunique()) if len(combined) else 0
        ),
        "agencies_with_any_reported_month": (
            int(combined.loc[combined["reported"], "ori9"].nunique())
            if len(combined)
            else 0
        ),
        "successful_calls": puller.calls,
        "failed_calls": len(puller.failures),
        "failures": puller.failures[:200],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (parsed_dir / "pull_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "states"}, indent=2))


if __name__ == "__main__":
    main()
