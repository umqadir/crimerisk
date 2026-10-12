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


def dependencies_match(recorded: Mapping[str, str], current: Mapping[str, str]) -> bool:
    """Whether a recorded dependency set names the same files with the same content.

    Names are bound to digests: two inputs that swap contents are a changed input. A stamp
    written under another checkout may carry absolute keys; such a key matches the current
    repo-relative key it ends with (on a path boundary), so moving a tree is not a changed
    input. Every current key must match exactly one recorded key and every recorded key must
    be used.
    """
    if len(recorded) != len(current):
        return False
    used: set[str] = set()
    for key, digest in current.items():
        if key in recorded:
            match = key
        else:
            candidates = [
                name for name in recorded if name not in used and str(name).endswith("/" + key)
            ]
            if len(candidates) != 1:
                return False
            match = candidates[0]
        if match in used or str(recorded[match]) != str(digest):
            return False
        used.add(match)
    return used == set(recorded)


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
    """Content-addressed freshness; fails closed.

    Current only when a stamp exists, still describes the artifact on disk, names the same
    dependencies with the same content, and records the same parameters. A missing, unreadable
    or mismatched stamp means the artifact is rebuilt; nothing is re-latched from mtimes.
    """
    if not artifact_path.exists():
        return False
    recorded = _read_dependency_stamp(artifact_path)
    if recorded is None or recorded.get("artifact") != content_digest(artifact_path):
        return False
    return dependencies_match(
        recorded["dependencies"], dependency_digests(dependency_paths)
    ) and recorded.get("parameters", {}) == _normalized_parameters(parameters)


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
