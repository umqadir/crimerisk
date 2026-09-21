from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path


# Memoised on (path, size, mtime_ns): a digest is recomputed only when the file
# actually changed on disk, so a single process pays for a dependency set once.
_DIGEST_MEMO: dict[tuple[str, int, int], str] = {}


def existing_dependency_paths(paths: Iterable[Path | None]) -> list[Path]:
    return [path for path in paths if path is not None and path.exists()]


def latest_dependency_mtime(paths: Iterable[Path | None]) -> float | None:
    existing = existing_dependency_paths(paths)
    if not existing:
        return None
    return max(path.stat().st_mtime for path in existing)


def content_digest(path: Path, *, chunk: int = 1 << 20) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    memoised = _DIGEST_MEMO.get(key)
    if memoised is not None:
        return memoised
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    _DIGEST_MEMO[key] = digest.hexdigest()
    return _DIGEST_MEMO[key]


def dependency_stamp_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + ".deps.json")


def repo_root() -> Path:
    """The checkout the running code belongs to (src/crimerisk/build_freshness.py -> root)."""
    return Path(__file__).resolve().parents[2]


def dependency_key(path: Path) -> str:
    """The name a dependency is recorded under: repo-relative inside the checkout.

    Stamps travel. A worktree, a clone and an isolated runtime are the same tree at different
    absolute paths, so an absolute key turns "same inputs" into "different inputs" for no reason
    that has anything to do with the bytes. Anything outside the checkout keeps its absolute
    path, because for those there is no shared root to be relative to.
    """
    resolved = path.resolve()
    root = repo_root()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def dependency_digests(dependency_paths: Iterable[Path | None]) -> dict[str, str]:
    return {
        dependency_key(path): content_digest(path)
        for path in existing_dependency_paths(dependency_paths)
    }


def digest_multiset(dependencies: Mapping[str, str]) -> list[str]:
    """The comparable content of a dependency set: its digests, ignoring where they live.

    This is the freshness comparison, not the dict itself. Stamps written under one repo root
    key their dependencies by that root's absolute paths; read from a second checkout of the
    same tree, every key misses and every artifact looks stale, so the documented stage order
    silently retrains models that are byte-for-byte current. The digests are the build input.
    """
    return sorted(str(digest) for digest in dependencies.values())


def _normalized_parameters(parameters: Mapping[str, object] | None) -> dict[str, object]:
    """Return a stable JSON-compatible copy of non-file build inputs."""
    if parameters is None:
        return {}
    return json.loads(json.dumps(dict(parameters), sort_keys=True))


def write_dependency_stamp(
    artifact_path: Path,
    dependency_paths: Iterable[Path | None],
    *,
    parameters: Mapping[str, object] | None = None,
) -> None:
    """Record what the artifact was built from, keyed by content rather than mtime."""
    if not artifact_path.exists():
        return
    stamp_path = dependency_stamp_path(artifact_path)
    payload = {
        "artifact": content_digest(artifact_path),
        "dependencies": dependency_digests(dependency_paths),
        "parameters": _normalized_parameters(parameters),
    }
    tmp_path = stamp_path.with_name(stamp_path.name + ".tmp")
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, stamp_path)


def _read_dependency_stamp(artifact_path: Path) -> dict[str, object] | None:
    stamp_path = dependency_stamp_path(artifact_path)
    if not stamp_path.exists():
        return None
    try:
        recorded = json.loads(stamp_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(recorded, dict) or not isinstance(recorded.get("dependencies"), dict):
        return None
    return recorded


def artifact_is_current(
    artifact_path: Path,
    dependency_paths: Iterable[Path | None],
    *,
    parameters: Mapping[str, object] | None = None,
) -> bool:
    """Content-addressed freshness, with the historical mtime rule as the fallback.

    A stamp is only trusted when it still describes the artifact on disk; a writer that
    does not stamp (or an artifact restored from elsewhere) falls back to the mtime rule
    and re-latches, so this is never looser than the mtime comparison it replaced.
    """
    if not artifact_path.exists():
        return False
    dependencies = existing_dependency_paths(dependency_paths)
    expected_parameters = _normalized_parameters(parameters)
    recorded = _read_dependency_stamp(artifact_path)
    if recorded is not None and recorded.get("artifact") == content_digest(artifact_path):
        return (
            digest_multiset(recorded["dependencies"])
            == digest_multiset(dependency_digests(dependencies))
            and recorded.get("parameters", {}) == expected_parameters
        )

    # A legacy or missing stamp cannot establish which non-file parameters produced an artifact.
    if expected_parameters:
        return False

    latest = latest_dependency_mtime(dependencies)
    if latest is None or artifact_path.stat().st_mtime >= latest:
        write_dependency_stamp(artifact_path, dependencies)
        return True
    return False


def copy_if_changed(src: Path, dst: Path) -> bool:
    """Promote only on a real content change.

    `shutil.copyfile` does not preserve mtime, so an unconditional promotion stamps
    `now` onto every promoted input and makes every downstream freshness check fail
    regardless of whether anything actually changed.
    """
    if (
        dst.exists()
        and dst.stat().st_size == src.stat().st_size
        and content_digest(dst) == content_digest(src)
    ):
        return False
    shutil.copyfile(src, dst)
    return True
