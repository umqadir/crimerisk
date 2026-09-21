from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket

from crimerisk.paths import RepoPaths


class StageLockError(RuntimeError):
    pass


STAGE_BLOCKERS: dict[str, tuple[str, ...]] = {
    "reference_layers": (
        "observations",
        "reporting_regimes",
        "controls",
        "geometry",
        "city_incident_shares",
        "outputs",
    ),
    "observations": (
        "reference_layers",
        "reporting_regimes",
        "controls",
    ),
    "reporting_regimes": (
        "reference_layers",
        "observations",
        "controls",
    ),
    "controls": (
        "reference_layers",
        "observations",
        "reporting_regimes",
        "outputs",
    ),
    "geometry": (
        "reference_layers",
        "outputs",
    ),
    "city_incident_shares": (
        "reference_layers",
        "geometry",
        "outputs",
    ),
    "outputs": (
        "reference_layers",
        "controls",
        "geometry",
        "city_incident_shares",
    ),
}


def blockers_for_stage(stage: str, *, ignore: tuple[str, ...] = ()) -> tuple[str, ...]:
    ignore_set = set(ignore)
    return tuple(name for name in STAGE_BLOCKERS.get(stage, ()) if name not in ignore_set)


def _lock_dir(paths: RepoPaths) -> Path:
    return paths.state_dir / "locks"


def _lock_path(paths: RepoPaths, stage: str) -> Path:
    return _lock_dir(paths) / f"{stage}.lock"


def _write_lock_payload(fd: int, payload: dict[str, object]) -> None:
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
    os.fsync(fd)


def _read_lock_payload(fd: int) -> tuple[dict[str, object] | None, bool]:
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 1_048_576)
    if not raw:
        return None, False
    had_padding = b"\x00" in raw
    text = raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    if not text:
        return None, had_padding
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, had_padding
    if isinstance(payload, dict):
        return payload, had_padding
    return None, had_padding


def _is_stage_locked(paths: RepoPaths, stage: str) -> bool:
    lock_path = _lock_path(paths, stage)
    if not lock_path.exists():
        return False
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        payload, had_padding = _read_lock_payload(fd)
        if isinstance(payload, dict):
            if payload.get("active") is True:
                recovered = dict(payload)
                recovered["stage"] = stage
                recovered["active"] = False
                recovered.setdefault("finished_at_utc", datetime.now(timezone.utc).isoformat())
                recovered["stale_lock_recovered_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_lock_payload(fd, recovered)
            elif had_padding:
                sanitized = dict(payload)
                sanitized["stage"] = stage
                _write_lock_payload(fd, sanitized)
        return False
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


@contextmanager
def stage_write_lock(
    *,
    paths: RepoPaths,
    stage: str,
    blocked_by: tuple[str, ...] | None = None,
):
    blockers = tuple(blocked_by) if blocked_by is not None else STAGE_BLOCKERS.get(stage, ())
    active_blockers = [name for name in blockers if _is_stage_locked(paths, name)]
    if active_blockers:
        raise StageLockError(
            f"Cannot start build stage '{stage}' while dependent stages are active: {sorted(active_blockers)}"
        )

    lock_path = _lock_path(paths, stage)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StageLockError(
                f"Build stage '{stage}' is already active. Lock path: {lock_path}"
            ) from exc
        payload = {
            "stage": stage,
            "active": True,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_lock_payload(fd, payload)
        yield
    finally:
        try:
            _write_lock_payload(
                fd,
                {
                    "stage": stage,
                    "active": False,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
