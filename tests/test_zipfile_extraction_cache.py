"""Extraction caches follow their archive (review #5)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from crimerisk.utils.zipfiles import extract_zip_all, extract_zip_member


def _archive(path: Path, payload: str) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("member.txt", payload)
    return path


def test_a_replaced_archive_re_extracts_its_member(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.zip", "old")
    out = tmp_path / "cache" / "member.txt"
    assert extract_zip_member(archive, "member.txt", out).read_text() == "old"
    _archive(archive, "new")
    assert extract_zip_member(archive, "member.txt", out).read_text() == "new"


def test_a_file_without_a_matching_fingerprint_is_not_trusted(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.zip", "complete")
    out = tmp_path / "cache" / "member.txt"
    out.parent.mkdir(parents=True)
    out.write_text("parti")  # an interrupted extraction under the old writer
    assert extract_zip_member(archive, "member.txt", out).read_text() == "complete"
    assert not list(out.parent.glob("*.partial"))


def test_a_replaced_archive_re_extracts_the_directory(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.zip", "old")
    out_dir = tmp_path / "cache" / "a"
    assert (extract_zip_all(archive, out_dir) / "member.txt").read_text() == "old"
    _archive(archive, "new")
    assert (extract_zip_all(archive, out_dir) / "member.txt").read_text() == "new"
