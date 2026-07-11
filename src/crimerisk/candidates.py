from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import subprocess

from crimerisk.paths import RepoPaths


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CandidateOutputRun:
    run_id: str
    created_at_utc: str
    candidate_dir: Path
    block_group_ags_core_out: Path
    tract_ags_core_out: Path
    block_group_fbi_out: Path
    tract_fbi_out: Path
    build_manifest_out: Path
    validation_summary_out: Path


def output_artifact_filenames(*, year: int, include_audits: bool = True) -> tuple[str, ...]:
    filenames = [
        f"crimerisk_block_group_{int(year)}_ags_core.parquet",
        f"crimerisk_tract_{int(year)}_ags_core.parquet",
        f"crimerisk_block_group_{int(year)}_fbi_calibrated.parquet",
        f"crimerisk_tract_{int(year)}_fbi_calibrated.parquet",
    ]
    if include_audits:
        filenames.extend(
            [
                f"zero_target_denominator_audit_{int(year)}.parquet",
                f"allocation_component_denominator_audit_{int(year)}.parquet",
            ]
        )
    return tuple(filenames)


def resolve_candidate_output_run(
    *,
    paths: RepoPaths,
    year: int,
    run_id: str | None = None,
    now: datetime | None = None,
) -> CandidateOutputRun:
    created_at = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    created_at_utc = created_at.isoformat()
    if run_id is None:
        run_id = _new_run_id(paths=paths, created_at=created_at)
    _validate_run_id(run_id)
    candidate_dir = paths.state_dir / "candidates" / run_id
    return CandidateOutputRun(
        run_id=run_id,
        created_at_utc=created_at_utc,
        candidate_dir=candidate_dir,
        block_group_ags_core_out=candidate_dir / f"crimerisk_block_group_{int(year)}_ags_core.parquet",
        tract_ags_core_out=candidate_dir / f"crimerisk_tract_{int(year)}_ags_core.parquet",
        block_group_fbi_out=candidate_dir / f"crimerisk_block_group_{int(year)}_fbi_calibrated.parquet",
        tract_fbi_out=candidate_dir / f"crimerisk_tract_{int(year)}_fbi_calibrated.parquet",
        build_manifest_out=candidate_dir / "manifest.json",
        validation_summary_out=candidate_dir / "validation_summary.json",
    )


def candidate_run_manifest_metadata(
    *,
    paths: RepoPaths,
    candidate: CandidateOutputRun,
    argv: list[str],
) -> dict[str, object]:
    return {
        "run_id": candidate.run_id,
        "created_at_utc": candidate.created_at_utc,
        "candidate_dir": str(candidate.candidate_dir),
        "validation_summary_path": str(candidate.validation_summary_out),
        "command": shlex.join(argv),
        "argv": list(argv),
        "cwd": str(paths.repo_root),
        "git": _git_state(paths.repo_root),
        "resolved_outputs": {
            "block_group_ags_core": str(candidate.block_group_ags_core_out),
            "tract_ags_core": str(candidate.tract_ags_core_out),
            "block_group_fbi_calibrated": str(candidate.block_group_fbi_out),
            "tract_fbi_calibrated": str(candidate.tract_fbi_out),
            "build_manifest": str(candidate.build_manifest_out),
        },
    }


def _new_run_id(*, paths: RepoPaths, created_at: datetime) -> str:
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    short_sha = _git_capture(paths.repo_root, "rev-parse", "--short=12", "HEAD") or "nogit"
    base_run_id = f"{timestamp}-{short_sha}"
    run_id = base_run_id
    suffix = 2
    while (paths.state_dir / "candidates" / run_id).exists():
        run_id = f"{base_run_id}-{suffix:02d}"
        suffix += 1
    return run_id


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "candidate run ids may contain only ASCII letters, digits, underscores, periods, and hyphens"
        )


def _git_state(repo_root: Path) -> dict[str, object]:
    status = _git_capture(repo_root, "status", "--porcelain=v1", "--untracked-files=normal")
    tracked_status = _git_capture(repo_root, "status", "--porcelain=v1", "--untracked-files=no")
    status_lines = status.splitlines() if status else []
    tracked_status_lines = tracked_status.splitlines() if tracked_status else []
    return {
        "head_sha": _git_capture(repo_root, "rev-parse", "HEAD"),
        "short_sha": _git_capture(repo_root, "rev-parse", "--short=12", "HEAD"),
        "branch": _git_capture(repo_root, "branch", "--show-current"),
        "dirty": bool(status_lines),
        "tracked_dirty": bool(tracked_status_lines),
        "status_porcelain": status_lines,
    }


def _git_capture(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()
