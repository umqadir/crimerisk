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


PRIORITY_STATES = ["CA", "FL", "NY", "LA", "GA", "MS", "TX", "NJ"]


def main() -> int:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    priority_path = paths.review_analysis_dir / "source_audit" / "state_source_priority_2024.csv"
    queue_path = paths.review_queues_dir / "source" / "fbi_cde_source_review_queue.csv"
    if not priority_path.exists() or not queue_path.exists():
        raise FileNotFoundError("Required source-review inputs are missing.")

    priority = pd.read_csv(priority_path, dtype=str).fillna("")
    queue = pd.read_csv(queue_path, dtype=str).fillna("")

    packet_root = paths.review_packets_dir / "source" / "states"
    packet_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for state_abbr in PRIORITY_STATES:
        packet_dir = packet_root / state_abbr.lower()
        packet_dir.mkdir(parents=True, exist_ok=True)

        state_priority = priority[priority["state_abbr"].eq(state_abbr)].copy()
        state_queue = queue[queue["state_abbr"].eq(state_abbr)].copy()
        state_priority.to_csv(packet_dir / "state_priority_rows.csv", index=False)
        state_queue.to_csv(packet_dir / "source_review_queue.csv", index=False)

        packet_status = pd.DataFrame(
            [
                {
                    "state_abbr": state_abbr,
                    "packet_status": "scaffolded",
                    "current_owner": "",
                    "production_input_ready": "",
                    "notes": "",
                }
            ]
        )
        packet_status.to_csv(packet_dir / "packet_status.csv", index=False)

        manifest = {
            "state_abbr": state_abbr,
            "packet_dir": str(packet_dir),
            "priority_rows": int(len(state_priority)),
            "queue_rows": int(len(state_queue)),
            "required_outputs": [
                "packet_status.csv",
                "research_findings.json",
                "source_recommendations.csv",
            ],
        }
        (packet_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2))
        manifest_rows.append(manifest)

    manifest_path = packet_root / "source_review_packet_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(json.dumps({"packet_root": str(packet_root), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
