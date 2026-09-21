from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def extract_zip_member(zip_path: Path, member: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


def extract_zip_all(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        return out_dir
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.touch()
    return out_dir
