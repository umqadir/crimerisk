from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    src_dir: Path
    state_dir: Path
    data_dir: Path
    docs_dir: Path
    cache_dir: Path
    archive_dir: Path
    review_dir: Path
    review_support_dir: Path
    review_queues_dir: Path
    review_packets_dir: Path
    review_analysis_dir: Path
    review_runs_dir: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "RepoPaths":
        state_dir = repo_root / "state"
        review_dir = state_dir / "review"
        return RepoPaths(
            repo_root=repo_root,
            src_dir=repo_root / "src",
            state_dir=state_dir,
            data_dir=repo_root / "data",
            docs_dir=repo_root / "docs",
            cache_dir=state_dir / "cache",
            archive_dir=repo_root / "archive",
            review_dir=review_dir,
            review_support_dir=review_dir / "support",
            review_queues_dir=review_dir / "queues",
            review_packets_dir=review_dir / "packets",
            review_analysis_dir=review_dir / "analysis",
            review_runs_dir=review_dir / "runs",
        )


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_paths() -> RepoPaths:
    return RepoPaths.from_repo_root(get_repo_root())
