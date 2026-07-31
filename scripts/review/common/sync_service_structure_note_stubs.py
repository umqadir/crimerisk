from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths


def _build_municipal_stubs(queue: pd.DataFrame) -> pd.DataFrame:
    stubs = queue.copy()
    stubs["county_fips"] = pd.NA
    stubs["agency_or_structure_name"] = stubs["jurisdiction_name"].astype("string")
    stubs["issue_type"] = "municipal_target_review"
    stubs["service_structure_summary"] = (
        "2024 municipal estimate is fully or heavily fill-backed and flagged for review in the "
        + stubs["review_lane"].astype("string")
        + " lane; verify whether the jurisdiction target and service structure are correct."
    )
    stubs["implication_for_pipeline"] = (
        "Confirm whether this jurisdiction should remain a standard municipal target, receive a documented "
        "service-structure override, or stay estimator-backed with explicit acceptance."
    )
    stubs["source_links"] = (
        "state/review/queues/municipal/municipal_target_review_queue.csv#"
        + stubs["jurisdiction_id"].astype("string")
    )
    stubs["note_status"] = "pending"
    return stubs[
        [
            "state_fips",
            "county_fips",
            "agency_or_structure_name",
            "issue_type",
            "service_structure_summary",
            "implication_for_pipeline",
            "source_links",
            "note_status",
        ]
    ].drop_duplicates()


def sync_note_stubs(*, repo_root: Path) -> dict[str, int | str]:
    paths = RepoPaths.from_repo_root(repo_root)
    queue_path = repo_root / "state" / "review" / "queues" / "municipal" / "municipal_target_review_queue.parquet"
    notes_path = paths.review_support_dir / "service_structure_notes.csv"

    queue = pd.read_parquet(queue_path)
    notes = pd.read_csv(notes_path)
    stubs = _build_municipal_stubs(queue)

    notes["state_fips"] = notes["state_fips"].astype("string").str.zfill(2)
    notes["agency_or_structure_name"] = notes["agency_or_structure_name"].astype("string")
    notes["issue_type"] = notes["issue_type"].astype("string")

    existing_key = set(
        zip(
            notes["state_fips"].astype(str),
            notes["agency_or_structure_name"].astype(str),
            notes["issue_type"].astype(str),
        )
    )
    add_mask = [
        (str(r.state_fips).zfill(2), str(r.agency_or_structure_name), str(r.issue_type)) not in existing_key
        for r in stubs.itertuples(index=False)
    ]
    to_add = stubs.loc[add_mask].copy()
    out = pd.concat([notes, to_add], ignore_index=True)
    out.to_csv(notes_path, index=False)
    return {
        "notes_path": str(notes_path),
        "existing_rows": int(len(notes)),
        "added_rows": int(len(to_add)),
        "final_rows": int(len(out)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync pending municipal review stubs into service_structure_notes.csv.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(sync_note_stubs(repo_root=args.repo_root.resolve()))


if __name__ == "__main__":
    main()
