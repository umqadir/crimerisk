"""Run the CrimeRisk CLI with code/config from one checkout and isolated runtime paths."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_identity(source_root: Path, runtime_root: Path) -> None:
    launch_path = runtime_root / "launch.json"
    launch = json.loads(launch_path.read_text())
    identity = launch.get("source_identity") or {}
    records = identity.get("files") or []
    aggregate = hashlib.sha256()
    for record in records:
        relative = str(record["path"])
        path = source_root / relative
        if not path.is_file():
            raise SystemExit(f"frozen source file is missing: {path}")
        digest = _sha256(path)
        size = path.stat().st_size
        if digest != record["sha256"] or size != int(record["size_bytes"]):
            raise SystemExit(f"source content changed after runtime preparation: {path}")
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\n")
    if aggregate.hexdigest() != identity.get("aggregate_sha256"):
        raise SystemExit("source identity aggregate does not match launch.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command:
        raise SystemExit("pass the CrimeRisk CLI command after --")
    command = list(args.command)
    if command[0] == "--":
        command.pop(0)
    if command and command[0].endswith("main.py"):
        command.pop(0)
    if not command:
        raise SystemExit("missing CrimeRisk CLI command")

    source_root = args.source_root.resolve()
    runtime_root = args.runtime_root.resolve()
    _verify_source_identity(source_root, runtime_root)
    source_path = source_root / "src"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    from crimerisk import cli  # noqa: PLC0415
    from crimerisk.paths import RepoPaths  # noqa: PLC0415

    state_dir = runtime_root / "state"
    review_dir = state_dir / "review"
    repo_overlay_root = runtime_root / "repo-overlay"
    paths = replace(
        RepoPaths.from_repo_root(repo_overlay_root),
        state_dir=state_dir,
        data_dir=runtime_root / "data",
        cache_dir=state_dir / "cache",
        archive_dir=runtime_root / "archive",
        review_dir=review_dir,
        review_support_dir=review_dir / "support",
        review_queues_dir=review_dir / "queues",
        review_packets_dir=review_dir / "packets",
        review_analysis_dir=review_dir / "analysis",
        review_runs_dir=review_dir / "runs",
    )
    for name, module in tuple(sys.modules.items()):
        if name.startswith("crimerisk") and hasattr(module, "get_paths"):
            setattr(module, "get_paths", lambda: paths)
    from crimerisk import candidates  # noqa: PLC0415

    frozen_git = json.loads((runtime_root / "launch.json").read_text()).get(
        "original_git_metadata", {}
    )
    candidates._git_state = lambda _repo_root: dict(frozen_git)
    # Relative state/data outputs in the CLI must resolve inside the runtime. Source imports and
    # tracked configs remain pinned to source_root through PYTHONPATH and RepoPaths.repo_root.
    os.chdir(runtime_root)
    sys.argv = [str(source_root / "main.py"), *command]
    return int(cli.main())


if __name__ == "__main__":
    raise SystemExit(main())
