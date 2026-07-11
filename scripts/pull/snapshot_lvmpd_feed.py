"""Snapshot the live LVMPD Weekly NIBRS Crimes ArcGIS feed to durable storage.

LVMPD's public crime feed (an ArcGIS FeatureServer layer, not a dated-vintage
publication like FBI CDE or a state UCR program) is an UNDOCUMENTED rolling window --
observed to hold roughly the trailing ~18 months of records and to silently drop
older rows with no versioning, no changelog, and no way to recover what rolled off.
Calendar-year 2024 has already aged out of the live feed as of this pull (2026-07-09);
whatever is live today is the only copy of "today's window" that will ever exist.

This script takes a point-in-time snapshot of the entire feed as it stands right now,
so that a future product year building on LVMPD data does not silently lose the rows
that have since rolled off the server. It performs NO transformation, offense mapping,
or ORI matching -- that belongs to a separate LVMPD onboarding/ingest lane if/when one
is built. This is preservation only: pull everything, write it byte-identically
reproducible from the raw feed, record enough provenance to prove what was pulled and
when, and get out of the way.

Endpoint (ArcGIS REST, FeatureServer layer 0, "Reported Crimes"):
  https://services.arcgis.com/jjSk6t82vIntwDbs/arcgis/rest/services/
    LVMPD_Weekly_NIBRS_Crimes/FeatureServer/0

Paging: the server's maxRecordCount is 2000 rows/request (confirmed via the layer's
`?f=json` metadata), so we page with resultOffset/resultRecordCount. Ordering by the
OBJECTID field (the layer's objectIdField, confirmed via metadata) makes paging
deterministic -- without an explicit orderByFields, ArcGIS gives no guarantee that two
requests return rows in the same order, which would risk skipped or duplicated rows
across a multi-page pull.

Outputs (under --out-dir):
  lvmpd_feed_snapshot_<UTC-pull-date>.parquet   All rows, all fields, as pulled
  lvmpd_feed_snapshot_provenance.json           Endpoint, pull timestamp, row count,
                                                 min/max ReportedOn, sha256 of the parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

FEATURE_SERVER_URL = (
    "https://services.arcgis.com/jjSk6t82vIntwDbs/arcgis/rest/services/"
    "LVMPD_Weekly_NIBRS_Crimes/FeatureServer/0"
)
QUERY_URL = f"{FEATURE_SERVER_URL}/query"
OBJECT_ID_FIELD = "OBJECTID"
PAGE_SIZE = 2000  # confirmed via FeatureServer/0?f=json -> maxRecordCount
DATE_FIELDS = ("ReportedOn", "UpdatedDate")  # epoch-ms fields per layer metadata


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _get_json(url: str, *, params: dict, timeout: int = 60) -> dict:
    last_error: Exception | None = None
    for attempt, sleep_sec in enumerate((0.0, 2.0, 5.0, 10.0), start=1):
        if sleep_sec:
            time.sleep(sleep_sec)
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(f"ArcGIS error payload: {payload['error']}")
            return payload
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            print(f"  fetch attempt {attempt} failed: {exc}")
    raise RuntimeError(f"Failed to fetch {url} after retries") from last_error


def fetch_layer_metadata() -> dict:
    return _get_json(FEATURE_SERVER_URL, params={"f": "json"})


def fetch_server_count() -> int:
    payload = _get_json(QUERY_URL, params={"where": "1=1", "returnCountOnly": "true", "f": "json"})
    return int(payload["count"])


def fetch_all_rows() -> list[dict]:
    """Page through the full feed via resultOffset, ordered by OBJECTID for determinism."""
    rows: list[dict] = []
    offset = 0
    page_num = 0
    while True:
        page_num += 1
        params = {
            "where": "1=1",
            "outFields": "*",
            "outSR": "4326",
            "returnGeometry": "false",  # Longitude/Latitude already carried as attributes
            "orderByFields": f"{OBJECT_ID_FIELD} ASC",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }
        payload = _get_json(QUERY_URL, params=params)
        features = payload.get("features", [])
        page_rows = [f["attributes"] for f in features]
        rows.extend(page_rows)
        print(f"  page {page_num}: offset={offset} -> {len(page_rows)} rows (running total {len(rows):,})")
        if len(page_rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.2)  # be polite to the server across ~100 sequential requests
    return rows


# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------


def build_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in DATE_FIELDS:
        if col in frame.columns:
            # ArcGIS esriFieldTypeDate values are epoch milliseconds (UTC).
            frame[col] = pd.to_datetime(frame[col], unit="ms", utc=True, errors="coerce")
    frame = frame.sort_values(OBJECT_ID_FIELD, kind="mergesort").reset_index(drop=True)
    return frame


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pulled_at = datetime.now(timezone.utc)
    pull_date_str = pulled_at.strftime("%Y-%m-%d")

    print(f"Fetching layer metadata from {FEATURE_SERVER_URL} ...")
    metadata = fetch_layer_metadata()
    max_record_count = metadata.get("maxRecordCount")
    object_id_field = metadata.get("objectIdField", OBJECT_ID_FIELD)
    print(f"  maxRecordCount={max_record_count}, objectIdField={object_id_field}")
    if object_id_field != OBJECT_ID_FIELD:
        raise RuntimeError(
            f"Layer's objectIdField ({object_id_field}) does not match the field this "
            f"script orders by ({OBJECT_ID_FIELD}); paging determinism is not guaranteed."
        )

    print("Fetching server-side row count (for post-pull drift check) ...")
    server_count = fetch_server_count()
    print(f"  server reports {server_count:,} rows")

    print("Paging through all rows (orderByFields=OBJECTID ASC, resultOffset paging) ...")
    rows = fetch_all_rows()
    print(f"Fetched {len(rows):,} rows total across the pull.")

    print("Assembling frame ...")
    frame = build_frame(rows)

    duplicate_object_ids = int(frame[OBJECT_ID_FIELD].duplicated().sum())
    if duplicate_object_ids:
        print(f"  WARNING: {duplicate_object_ids} duplicate {OBJECT_ID_FIELD} values in the pulled frame")

    parquet_path = out_dir / f"lvmpd_feed_snapshot_{pull_date_str}.parquet"
    frame.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({len(frame):,} rows)")

    file_sha256 = sha256_of_file(parquet_path)
    file_size_bytes = parquet_path.stat().st_size

    reported_on = frame["ReportedOn"].dropna() if "ReportedOn" in frame.columns else pd.Series([], dtype="datetime64[ns, UTC]")
    min_reported_on = reported_on.min().isoformat() if len(reported_on) else None
    max_reported_on = reported_on.max().isoformat() if len(reported_on) else None

    drift = None
    drift_pct = None
    if server_count:
        drift = len(frame) - server_count
        drift_pct = round(100 * drift / server_count, 4)

    provenance = {
        "endpoint": QUERY_URL,
        "feature_server_url": FEATURE_SERVER_URL,
        "pulled_at_utc": pulled_at.isoformat(),
        "page_size": PAGE_SIZE,
        "server_max_record_count": max_record_count,
        "order_by_field": OBJECT_ID_FIELD,
        "out_sr": 4326,
        "row_count_pulled": int(len(frame)),
        "row_count_server_count_query": server_count,
        "row_count_drift": drift,
        "row_count_drift_pct": drift_pct,
        "duplicate_object_id_count": duplicate_object_ids,
        "min_reported_on": min_reported_on,
        "max_reported_on": max_reported_on,
        "output_parquet": str(parquet_path),
        "output_parquet_bytes": file_size_bytes,
        "output_parquet_sha256": file_sha256,
        "columns": list(frame.columns),
        "context": (
            "LVMPD's public ArcGIS feed is an undocumented rolling window (observed "
            "~18 months); calendar 2024 has already rolled off as of this pull. This "
            "snapshot preserves the live feed at pull time so future product years do "
            "not lose rows that later age out of the server."
        ),
    }
    provenance_path = out_dir / "lvmpd_feed_snapshot_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=False))
    print(f"Wrote provenance: {provenance_path}")

    print("\nSummary:")
    print(f"  rows pulled:            {len(frame):,}")
    print(f"  server count query:     {server_count:,}")
    print(f"  drift:                  {drift} ({drift_pct}%)" if drift is not None else "  drift: n/a")
    print(f"  duplicate OBJECTIDs:    {duplicate_object_ids}")
    print(f"  ReportedOn range:       {min_reported_on}  ->  {max_reported_on}")
    print(f"  parquet size:           {file_size_bytes:,} bytes")
    print(f"  parquet sha256:         {file_sha256}")


if __name__ == "__main__":
    main()
