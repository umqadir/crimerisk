"""Pull the AGS 2022A tract crime index used as the E4 comparator.

Third-party commercial values: the output stays under `state/` (gitignored) and is never committed.

Output (`--out-dir`, default `state/cache/ags_2022a/`):

    ags_2022a_tract.parquet    OBJECTID, ID, POP_C, CRMCY<offense> (one row per tract)
    download_manifest.json     service, layer, fields, rows, sha256, vintage, downloaded_at_utc

The 2026-09-21 pull held 84,112 rows, sha256 ebc846a2...; a rerun against an updated service vintage
lands on different values, so compare the row count and spot values before replacing a cache.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE = "https://services8.arcgis.com/TrLhDuWxkcSkFAjt/arcgis/rest/services/USA_Crime_Index_Tract/FeatureServer"
LAYER = 7
FIELDS = ["OBJECTID", "ID", "POP_C", "CRMCYMURD", "CRMCYRAPE", "CRMCYROBB", "CRMCYASST", "CRMCYBURG", "CRMCYLARC", "CRMCYMVEH"]
VINTAGE = "2022A"


def _get(url: str, params: dict[str, object], *, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def pull(out_dir: Path) -> dict[str, object]:
    query = f"{SERVICE}/{LAYER}/query"
    expected = int(_get(query, {"where": "1=1", "returnCountOnly": "true", "f": "json"})["count"])
    page = int(_get(f"{SERVICE}/{LAYER}", {"f": "json"}).get("maxRecordCount") or 2000)
    rows: list[dict[str, object]] = []
    last_id = 0
    while True:
        payload = _get(query, {
            "where": f"OBJECTID > {last_id}",
            "outFields": ",".join(FIELDS),
            "orderByFields": "OBJECTID",
            "returnGeometry": "false",
            "resultRecordCount": page,
            "f": "json",
        })
        batch = [feature["attributes"] for feature in payload.get("features", [])]
        if not batch:
            break
        rows.extend(batch)
        last_id = int(batch[-1]["OBJECTID"])
    frame = pd.DataFrame(rows, columns=FIELDS).sort_values("OBJECTID").reset_index(drop=True)
    if len(frame) != expected or frame["OBJECTID"].duplicated().any():
        raise RuntimeError(f"Pulled {len(frame)} rows (unique OBJECTID {frame['OBJECTID'].nunique()}); service count {expected}.")
    frame["ID"] = frame["ID"].astype(str)
    for column in FIELDS:
        if column != "ID":
            frame[column] = frame[column].astype("int64")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ags_2022a_tract.parquet"
    frame.to_parquet(out_path, index=False)
    manifest = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "fields": FIELDS,
        "layer": LAYER,
        "rows": int(len(frame)),
        "service": f"{SERVICE}/{LAYER}",
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "vintage": VINTAGE,
    }
    (out_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "state" / "cache" / "ags_2022a")
    args = parser.parse_args()
    print(json.dumps(pull(args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
