"""Freeze the identity of a candidate and the ignored inputs that produced it.

This script copies no source data.  It walks the candidate manifest's direct input
records, follows dependency sidecars, and writes a SHA-256 inventory.  Git object
IDs pin tracked code and configuration at both the candidate's build commit and
the checkout used to run this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKED_SCOPES = (
    "src",
    "scripts",
    "configs",
    "frontend/build",
    "frontend/public",
    "main.py",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "AGENTS.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise SystemExit(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _paths_in_stats(value: Any) -> Iterable[Path]:
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path:
            yield Path(path)
        for child in value.values():
            yield from _paths_in_stats(child)
    elif isinstance(value, list):
        for child in value:
            yield from _paths_in_stats(child)


def _sidecar_candidates(path: Path) -> tuple[Path, ...]:
    return (Path(f"{path}.deps.json"),)


def _display_path(path: Path, runtime_root: Path) -> tuple[str, str | None]:
    absolute = str(path)
    try:
        relative = str(path.relative_to(runtime_root))
    except ValueError:
        relative = None
    return absolute, relative


def _record(
    path: Path,
    *,
    runtime_root: Path,
    role: str,
    declared_digest: str | None = None,
) -> dict[str, Any]:
    absolute, relative = _display_path(path, runtime_root)
    record: dict[str, Any] = {
        "role": role,
        "manifest_path": absolute,
        "runtime_relative_path": relative,
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
        "declared_dependency_digest": declared_digest,
    }
    if not path.exists():
        return record
    resolved = path.resolve()
    stat = resolved.stat()
    record.update(
        {
            "resolved_path": str(resolved),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(resolved),
        }
    )
    return record


def _dependency_closure(
    seeds: Iterable[Path], *, runtime_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queued: list[tuple[Path, str, str | None]] = [
        (path, "candidate_manifest_input", None) for path in seeds
    ]
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    sidecar_errors: list[dict[str, Any]] = []
    while queued:
        path, role, declared = queued.pop()
        key = os.path.normpath(str(path))
        if key in seen:
            continue
        seen.add(key)
        records.append(
            _record(path, runtime_root=runtime_root, role=role, declared_digest=declared)
        )
        if not path.exists() or not path.is_file():
            continue
        for sidecar in _sidecar_candidates(path):
            if not sidecar.exists():
                continue
            sidecar_key = os.path.normpath(str(sidecar))
            if sidecar_key not in seen:
                seen.add(sidecar_key)
                records.append(
                    _record(sidecar, runtime_root=runtime_root, role="dependency_sidecar")
                )
            try:
                dependency_data = _json(sidecar).get("dependencies", {})
                if not isinstance(dependency_data, dict):
                    raise ValueError("dependencies is not an object")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                sidecar_errors.append({"path": str(sidecar), "error": str(exc)})
                continue
            for dependency, digest in dependency_data.items():
                queued.append(
                    (
                        Path(str(dependency)),
                        "sidecar_dependency",
                        str(digest) if digest is not None else None,
                    )
                )
    records.sort(key=lambda item: item["manifest_path"])
    sidecar_errors.sort(key=lambda item: item["path"])
    return records, sidecar_errors


def _candidate_records(candidate_dir: Path, runtime_root: Path) -> list[dict[str, Any]]:
    records = [
        _record(path, runtime_root=runtime_root, role="candidate_artifact")
        for path in sorted(candidate_dir.rglob("*"))
        if path.is_file()
    ]
    return records


def _git_identity(repo: Path, revision: str, scopes: Iterable[str]) -> dict[str, Any]:
    resolved_revision = _git(repo, "rev-parse", f"{revision}^{{commit}}")
    scope_objects: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        object_id = _git(repo, "rev-parse", f"{resolved_revision}:{scope}", check=False)
        if not object_id:
            scope_objects[scope] = {"present": False}
            continue
        object_type = _git(repo, "cat-file", "-t", object_id)
        scope_objects[scope] = {
            "present": True,
            "git_object": object_id,
            "git_object_type": object_type,
        }
    return {
        "commit": resolved_revision,
        "tree": _git(repo, "rev-parse", f"{resolved_revision}^{{tree}}"),
        "scopes": scope_objects,
    }


def _frontend_references(
    candidate_dir: Path, runtime_root: Path
) -> list[dict[str, Any]]:
    artifact_path = candidate_dir / "frontend_artifact_manifest.json"
    if not artifact_path.exists():
        return []
    artifact = _json(artifact_path)
    references: list[tuple[str, dict[str, Any]]] = []
    for section in ("source_parquets", "tiles"):
        for name, value in (artifact.get(section) or {}).items():
            if isinstance(value, dict):
                references.append((f"frontend_{section}.{name}", value))
    for name, key in (
        ("frontend.snapshot", "snapshot_path"),
        ("frontend.index", "index_path"),
        ("frontend.methodology", "methodology_path"),
    ):
        frontend = artifact.get("frontend") or {}
        path_value = frontend.get(key)
        if isinstance(path_value, str):
            references.append(
                (
                    name,
                    {
                        "path": path_value,
                        "sha256": frontend.get(key.replace("_path", "_sha256")),
                    },
                )
            )
    records: list[dict[str, Any]] = []
    for role, value in references:
        path_value = value.get("path")
        if not isinstance(path_value, str):
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = runtime_root / path
        record = _record(path, runtime_root=runtime_root, role=role)
        record["declared_sha256"] = value.get("sha256")
        record["declared_size_bytes"] = value.get("size_bytes")
        if record.get("sha256") and value.get("sha256"):
            record["matches_declared_sha256"] = record["sha256"] == value["sha256"]
        records.append(record)
    return sorted(records, key=lambda item: item["role"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SCRIPT_REPO_ROOT)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--tracked-scope",
        action="append",
        default=None,
        help="Git path whose object identity is recorded; may be repeated.",
    )
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    runtime_root = args.runtime_root.resolve()
    candidate_dir = runtime_root / "state" / "candidates" / args.run_id
    manifest_path = candidate_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing candidate manifest: {manifest_path}")
    manifest = _json(manifest_path)
    manifest_run_id = (manifest.get("run") or {}).get("run_id")
    if manifest_run_id != args.run_id:
        raise SystemExit(
            f"candidate manifest run id {manifest_run_id!r} does not match {args.run_id!r}"
        )
    build_commit = ((manifest.get("run") or {}).get("git") or {}).get("head_sha")
    if not isinstance(build_commit, str) or not build_commit:
        raise SystemExit("candidate manifest has no run.git.head_sha")

    seeds = list(_paths_in_stats(manifest.get("input_file_stats", {})))
    dependency_records, sidecar_errors = _dependency_closure(
        seeds, runtime_root=runtime_root
    )
    candidate_records = _candidate_records(candidate_dir, runtime_root)
    frontend_records = _frontend_references(candidate_dir, runtime_root)
    scopes = tuple(args.tracked_scope or DEFAULT_TRACKED_SCOPES)
    source_head = _git(source_root, "rev-parse", "HEAD")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else source_root / "state" / "baselines" / args.run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_inventory.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "runtime_root": str(runtime_root),
        "candidate_dir": str(candidate_dir),
        "candidate_manifest_sha256": _sha256(manifest_path),
        "candidate_build_command": (manifest.get("run") or {}).get("command"),
        "git": {
            "candidate_build": _git_identity(source_root, build_commit, scopes),
            "inventory_checkout": _git_identity(source_root, source_head, scopes),
        },
        "candidate_artifacts": candidate_records,
        "dependency_closure": dependency_records,
        "dependency_sidecar_errors": sidecar_errors,
        "frontend_references": frontend_records,
        "summary": {
            "candidate_artifact_count": len(candidate_records),
            "candidate_artifact_bytes": sum(
                int(item.get("size_bytes", 0)) for item in candidate_records
            ),
            "dependency_path_count": len(dependency_records),
            "dependency_existing_file_count": sum(
                bool(item.get("exists")) for item in dependency_records
            ),
            "dependency_missing_count": sum(
                not bool(item.get("exists")) for item in dependency_records
            ),
            "dependency_logical_bytes": sum(
                int(item.get("size_bytes", 0)) for item in dependency_records
            ),
            "frontend_missing_count": sum(
                not bool(item.get("exists"))
                for item in frontend_records
            ),
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output_dir, prefix=".baseline_inventory.", delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    temporary.replace(output_path)
    inventory_sha = _sha256(output_path)
    print(f"wrote {output_path}")
    print(f"sha256 {inventory_sha}")
    for key, value in payload["summary"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
