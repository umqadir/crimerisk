"""Cached zip extraction, bound to the archive it came from.

An extracted file is reused only while its sidecar fingerprint still matches the archive
member (name, CRC-32, uncompressed size, from the zip's central directory). A replaced or
restaged archive therefore re-extracts, and extraction writes to a temporary name and renames
into place, so an interrupted run never leaves a partial file that a later run accepts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def _member_fingerprint(info: zipfile.ZipInfo) -> dict[str, object]:
    return {"name": info.filename, "crc": int(info.CRC), "size": int(info.file_size)}


def _fingerprint_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".source.json")


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(tmp, path)


def extract_zip_member(zip_path: Path, member: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        fingerprint = _member_fingerprint(zf.getinfo(member))
        sidecar = _fingerprint_path(out_path)
        if out_path.exists() and _read_json(sidecar) == fingerprint:
            return out_path
        handle, tmp_name = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".partial", dir=out_path.parent)
        try:
            with zf.open(member) as src, os.fdopen(handle, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp_name, out_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    _write_json_atomic(sidecar, fingerprint)
    return out_path


def extract_zip_all(zip_path: Path, out_dir: Path) -> Path:
    marker = out_dir / ".extracted"
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted((_member_fingerprint(info) for info in zf.infolist()), key=lambda row: str(row["name"]))
        fingerprint = {
            "members": len(members),
            "sha256": hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest(),
        }
        if marker.exists() and _read_json(marker) == fingerprint:
            return out_dir
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=out_dir.name + ".", suffix=".partial", dir=out_dir.parent))
        try:
            zf.extractall(staging)
            _write_json_atomic(staging / ".extracted", fingerprint)
            if out_dir.exists():
                shutil.rmtree(out_dir)
            os.replace(staging, out_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return out_dir
