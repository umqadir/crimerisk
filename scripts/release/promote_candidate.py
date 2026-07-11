from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "state" / "output"
DEFAULT_FRONTEND_SNAPSHOT = REPO_ROOT / "frontend" / "public" / "snapshot.json"
YEAR = 2024
PROMOTED_MANIFEST_NAME = f"crimerisk_output_build_{YEAR}.json"

RELEASE_PARQUETS = (
    f"crimerisk_block_group_{YEAR}_ags_core.parquet",
    f"crimerisk_tract_{YEAR}_ags_core.parquet",
    f"crimerisk_block_group_{YEAR}_fbi_calibrated.parquet",
    f"crimerisk_tract_{YEAR}_fbi_calibrated.parquet",
)
AUDIT_SIDECARS = (
    f"zero_target_denominator_audit_{YEAR}.parquet",
    f"allocation_component_denominator_audit_{YEAR}.parquet",
    f"denominator_publishability_audit_{YEAR}.csv",
    f"city_posterior_diagnostics_{YEAR}.parquet",
)
STALE_SUPERSEDED_FILES = (
    "manifest.json",
    "validation_summary.json",
)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing required JSON file: {_repo_relative(path)}") from exc


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _path_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _git_capture(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_is_ancestor(ancestor_sha: str, descendant_sha: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _manifest_git_sha(manifest: dict[str, Any]) -> str:
    run = manifest.get("run")
    if isinstance(run, dict):
        git = run.get("git")
        if isinstance(git, dict) and git.get("head_sha"):
            return str(git["head_sha"])
    git = manifest.get("git")
    if isinstance(git, dict) and git.get("head_sha"):
        return str(git["head_sha"])
    raise SystemExit("candidate manifest does not contain run.git.head_sha")


def _candidate_run_id(manifest: dict[str, Any], candidate_dir: Path) -> str:
    run = manifest.get("run")
    if isinstance(run, dict) and run.get("run_id"):
        return str(run["run_id"])
    return candidate_dir.name


def _resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _frontend_hash_check(
    *,
    snapshot_path: Path,
    candidate_bg_path: Path,
    promoted_bg_path: Path | None = None,
) -> dict[str, Any]:
    snapshot = _load_json(snapshot_path)
    source_value = snapshot.get("source_parquet")
    if not source_value:
        raise SystemExit(f"{_repo_relative(snapshot_path)} is missing source_parquet")
    source_path = _resolve_repo_path(source_value)
    if not source_path.exists():
        raise SystemExit(f"frontend snapshot source parquet does not exist: {_repo_relative(source_path)}")
    if not candidate_bg_path.exists():
        raise SystemExit(f"candidate BG parquet does not exist: {_repo_relative(candidate_bg_path)}")
    source_sha = _sha256_file(source_path)
    candidate_sha = _sha256_file(candidate_bg_path)
    if source_sha != candidate_sha:
        raise SystemExit(
            "frontend snapshot source parquet does not match candidate BG AGS-core parquet: "
            f"{_repo_relative(source_path)} sha256={source_sha}, "
            f"{_repo_relative(candidate_bg_path)} sha256={candidate_sha}"
        )
    result: dict[str, Any] = {
        "snapshot_path": _repo_relative(snapshot_path),
        "snapshot_source_parquet": _repo_relative(source_path),
        "snapshot_source_parquet_sha256": source_sha,
        "candidate_block_group_ags_core_path": _repo_relative(candidate_bg_path),
        "candidate_block_group_ags_core_sha256": candidate_sha,
        "matches_candidate_block_group_ags_core": True,
    }
    if promoted_bg_path is not None:
        promoted_sha = _sha256_file(promoted_bg_path)
        result.update(
            {
                "promoted_block_group_ags_core_path": _repo_relative(promoted_bg_path),
                "promoted_block_group_ags_core_sha256": promoted_sha,
                "matches_promoted_block_group_ags_core": source_sha == promoted_sha,
            }
        )
        if source_sha != promoted_sha:
            raise SystemExit(
                "frontend snapshot source parquet does not match promoted BG AGS-core parquet: "
                f"{source_sha} != {promoted_sha}"
            )
    return result


def _copy_file_atomic(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f".{dst.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    shutil.copy2(src, tmp_path)
    os.replace(tmp_path, dst)
    return {
        "name": dst.name,
        "source": _repo_relative(src),
        "destination": _repo_relative(dst),
        "sha256": _sha256_file(dst),
        "size_bytes": int(dst.stat().st_size),
    }


def promote_candidate(*, candidate_dir: Path, output_dir: Path, frontend_snapshot: Path) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = candidate_dir / "manifest.json"
    validation_summary_path = candidate_dir / "validation_summary.json"
    manifest = _load_json(manifest_path)
    validation_summary = _load_json(validation_summary_path)

    if validation_summary.get("ok") is not True:
        raise SystemExit(
            f"refusing promotion because {_repo_relative(validation_summary_path)} is not ok=true"
        )
    issues = validation_summary.get("issues")
    if isinstance(issues, list) and issues:
        raise SystemExit(
            f"refusing promotion because {_repo_relative(validation_summary_path)} still has issues"
        )

    candidate_sha = _manifest_git_sha(manifest)
    promoting_sha = _git_capture("rev-parse", "HEAD")
    if not _git_is_ancestor(candidate_sha, promoting_sha):
        raise SystemExit(
            "refusing promotion because candidate manifest git SHA is not an ancestor of HEAD: "
            f"{candidate_sha} !<= {promoting_sha}"
        )

    required_artifacts = [*RELEASE_PARQUETS, *AUDIT_SIDECARS]
    missing = [name for name in required_artifacts if not (candidate_dir / name).exists()]
    if missing:
        raise SystemExit(f"candidate is missing required promotion artifacts: {missing}")

    candidate_bg_path = candidate_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet"
    frontend_check = _frontend_hash_check(
        snapshot_path=frontend_snapshot,
        candidate_bg_path=candidate_bg_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    removed_stale: list[str] = []
    for name in (*required_artifacts, PROMOTED_MANIFEST_NAME, *STALE_SUPERSEDED_FILES):
        stale_path = output_dir / name
        if stale_path.exists():
            stale_path.unlink()
            removed_stale.append(_repo_relative(stale_path))

    copied_artifacts = [
        _copy_file_atomic(candidate_dir / name, output_dir / name)
        for name in required_artifacts
    ]
    promoted_bg_path = output_dir / f"crimerisk_block_group_{YEAR}_ags_core.parquet"
    frontend_check = _frontend_hash_check(
        snapshot_path=frontend_snapshot,
        candidate_bg_path=candidate_bg_path,
        promoted_bg_path=promoted_bg_path,
    )

    promoted_manifest = dict(manifest)
    promoted_manifest["promotion"] = {
        "candidate_run_id": _candidate_run_id(manifest, candidate_dir),
        "candidate_path": _repo_relative(candidate_dir),
        "candidate_manifest_path": _repo_relative(manifest_path),
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "promoted_output_dir": _repo_relative(output_dir),
        "promoted_manifest_path": _repo_relative(output_dir / PROMOTED_MANIFEST_NAME),
        "promoted_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "promoting_git_sha": promoting_sha,
        "candidate_git_sha": candidate_sha,
        "candidate_git_sha_is_ancestor_of_promoting_git_sha": True,
        "validation_summary_path": _repo_relative(validation_summary_path),
        "validation_summary_sha256": _sha256_file(validation_summary_path),
        "validation_summary_ok": True,
        "removed_stale_files": removed_stale,
        "copied_artifacts": copied_artifacts,
        "promoted_output_file_stats": {
            artifact["name"]: _path_stats(output_dir / artifact["name"])
            for artifact in copied_artifacts
        },
        "frontend_snapshot_hash_check": frontend_check,
    }
    _write_json_atomic(output_dir / PROMOTED_MANIFEST_NAME, promoted_manifest)

    return promoted_manifest["promotion"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a green candidate run into state/output.")
    parser.add_argument("candidate_dir", type=Path, help="Candidate directory under state/candidates.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frontend-snapshot", type=Path, default=DEFAULT_FRONTEND_SNAPSHOT)
    args = parser.parse_args()

    promotion = promote_candidate(
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        frontend_snapshot=args.frontend_snapshot,
    )
    print(json.dumps(promotion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
