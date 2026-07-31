from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths


def main() -> int:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    source_root = paths.review_packets_dir / "source"
    state_root = source_root / "states"
    rows: list[dict[str, object]] = []
    for status_path in sorted(state_root.glob("*/packet_status.csv")):
        status = pd.read_csv(status_path, dtype=str).fillna("")
        if status.empty:
            continue
        rows.append(
            {
                "packet_type": "state",
                "packet_key": status.iloc[0].get("state_abbr", status_path.parent.name.upper()),
                "state_abbr": status.iloc[0].get("state_abbr", status_path.parent.name.upper()),
                "packet_dir": str(status_path.parent),
                "packet_status": status.iloc[0].get("packet_status", ""),
                "current_owner": status.iloc[0].get("current_owner", ""),
                "production_input_ready": status.iloc[0].get("production_input_ready", ""),
                "notes": status.iloc[0].get("notes", ""),
            }
        )
    cde_status_path = source_root / "cde_official_surface" / "packet_status.csv"
    if cde_status_path.exists():
        status = pd.read_csv(cde_status_path, dtype=str).fillna("")
        if not status.empty:
            rows.append(
                {
                    "packet_type": "official_surface",
                    "packet_key": "cde_official_surface",
                    "state_abbr": "",
                    "packet_dir": str(cde_status_path.parent),
                    "packet_status": status.iloc[0].get("packet_status", ""),
                    "current_owner": status.iloc[0].get("current_owner", ""),
                    "production_input_ready": status.iloc[0].get("production_input_ready", ""),
                    "notes": status.iloc[0].get("notes", ""),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["packet_type", "packet_key"], kind="mergesort")
    out_path = source_root / "source_packet_status_summary.csv"
    summary.to_csv(out_path, index=False)
    print(json.dumps({"rows": int(len(summary)), "out_path": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
