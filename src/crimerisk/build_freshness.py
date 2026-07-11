from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def existing_dependency_paths(paths: Iterable[Path | None]) -> list[Path]:
    return [path for path in paths if path is not None and path.exists()]


def latest_dependency_mtime(paths: Iterable[Path | None]) -> float | None:
    existing = existing_dependency_paths(paths)
    if not existing:
        return None
    return max(path.stat().st_mtime for path in existing)


def artifact_is_current(artifact_path: Path, dependency_paths: Iterable[Path | None]) -> bool:
    if not artifact_path.exists():
        return False
    latest_dependency = latest_dependency_mtime(dependency_paths)
    return latest_dependency is None or artifact_path.stat().st_mtime >= latest_dependency
