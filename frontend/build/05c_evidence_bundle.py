"""05c - Versioned evidence bundle for an edition.

One deterministic tarball (sorted members, zero mtimes and owners) plus its
SHA-256, uploaded at release beside the tiles under `<edition>/evidence/` and
linked from the methods and download pages by 06:

  dist/evidence/crimerisk_<edition>_evidence.tar.gz
  dist/evidence/crimerisk_<edition>_evidence.tar.gz.sha256

Members (under crimerisk_<edition>_evidence/):

  configs/**                    the repository's configs/
  pyproject.toml, uv.lock
  candidate/manifest.json       the release candidate run manifest
  candidate/validation_summary.json
  eval/<name>/**                each --gold and --level evaluation directory,
                                without any `cache/` directory
  MANIFEST.json                 every member's sha256 and bytes, the columns and
                                keys stripped from it, the source paths and commit

AGS values are never republished (E4 protocol, analysis_scratch/final_phase/
e4_benchmark/00_PROTOCOL.md). Every table column and JSON key whose name carries
the token `ags` is stripped (e.g. `prediction_ags_2022a_tract`; this project's own
`ags_core` surface names are not AGS values and stay), except boolean
columns: `ags_uses_city_feed` is a true/false partition of the evaluation cities,
holds no AGS value, and removing it would merge the rows it separates in
results.csv. Aggregate metrics of the AGS comparison arm stay, as the protocol
allows derived aggregate statistics. A path component naming AGS is refused.
Build-machine paths inside text members are rewritten repo-relative (`PATH_REWRITES`).

Run:  uv run python frontend/build/05c_evidence_bundle.py --edition 2025.3 \\
          --candidate state/candidates/<run> --gold state/eval/<gold> --level state/eval/<level>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crschema import DIST, REPO  # noqa: E402

# `ags_core` is this project's own surface name (crimerisk_*_ags_core.parquet), not
# an AGS value, so it does not match.
AGS = re.compile(r"(^|[^a-z0-9])ags(?!_core)([^a-z0-9]|$)", re.IGNORECASE)
SKIP_DIRS = {"cache", "__pycache__"}
SKIP_NAMES = {".DS_Store"}


def is_ags(name: str) -> bool:
    return bool(AGS.search(name))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_parquet(data: bytes) -> tuple[bytes, list[str]]:
    table = pq.read_table(io.BytesIO(data))
    drop = [
        f.name for f in table.schema
        if is_ags(f.name) and not pa.types.is_boolean(f.type)
    ]
    if not drop:
        return data, []
    table = table.drop(drop)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue(), drop


def strip_csv(data: bytes) -> tuple[bytes, list[str]]:
    """Drop AGS columns textually, so every kept value is byte-for-byte the same."""
    text = data.decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return data, []
    header = rows[0]
    drop_idx = []
    for i, name in enumerate(header):
        if not is_ags(name):
            continue
        values = {r[i].strip().lower() for r in rows[1:] if i < len(r) and r[i].strip()}
        if values <= {"true", "false"}:
            continue          # a boolean partition flag, not an AGS value
        drop_idx.append(i)
    if not drop_idx:
        return data, []
    keep = [i for i in range(len(header)) if i not in drop_idx]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for r in rows:
        w.writerow([r[i] for i in keep if i < len(r)])
    return out.getvalue().encode("utf-8"), [header[i] for i in drop_idx]


def strip_json(data: bytes) -> tuple[bytes, list[str]]:
    dropped: list[str] = []

    def walk(obj, path):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if is_ags(str(k)) and not isinstance(v, bool):
                    dropped.append(f"{path}/{k}")
                    continue
                out[k] = walk(v, f"{path}/{k}")
            return out
        if isinstance(obj, list):
            return [walk(v, path) for v in obj]
        return obj

    obj = json.loads(data)
    cleaned = walk(obj, "")
    if not dropped:
        return data, []
    return (json.dumps(cleaned, indent=1, sort_keys=False) + "\n").encode(), dropped


def strip_member(name: str, data: bytes) -> tuple[bytes, list[str]]:
    if name.endswith(".parquet"):
        return strip_parquet(data)
    if name.endswith(".csv"):
        return strip_csv(data)
    if name.endswith(".json"):
        return strip_json(data)
    return data, []


# Build-machine paths in text members (run manifests, fold records, ledger evidence) are
# rewritten: the repository root becomes repo-relative, anything else under a home, volume or
# temp directory keeps only a placeholder prefix. Applied after the AGS strip.
_SEG = r"[^/\s\"',;]+"
PATH_REWRITES = [
    (re.compile(re.escape(REPO.resolve().as_posix()) + "/"), ""),
    (re.compile(re.escape(REPO.resolve().as_posix()) + r"(?![\w.-])"), "."),
    (re.compile(rf"/(?:Users|home)/{_SEG}/Projects/data-projects/national/crimerisk-clone/"), ""),
    (re.compile(rf"/(?:Users|home)/{_SEG}/Projects/data-projects/national/crimerisk/"), ""),
    (re.compile(rf"/(?:Users|home)/{_SEG}/"), "~/"),
    (re.compile(rf"/Volumes/{_SEG}/"), "<volume>/"),
    (re.compile(r"/private/tmp/[^\s\"',;]*?/scratchpad/"), "<tmp>/"),
    (re.compile(r"/private/tmp/"), "<tmp>/"),
]
TEXT_SUFFIXES = (".csv", ".json", ".md", ".txt", ".toml", ".lock", ".yaml", ".yml", ".py", ".sh")


def scrub_paths(name: str, data: bytes) -> tuple[bytes, int]:
    if not name.endswith(TEXT_SUFFIXES):
        return data, 0
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, 0
    total = 0
    for pattern, repl in PATH_REWRITES:
        text, n = pattern.subn(repl, text)
        total += n
    return (text.encode("utf-8"), total) if total else (data, 0)


def collect(src: Path, dest: str) -> list[tuple[str, Path]]:
    if src.is_file():
        return [(dest, src)]
    out = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if not p.is_file() or p.name in SKIP_NAMES or SKIP_DIRS & set(rel.parts[:-1]):
            continue
        out.append((f"{dest}/{rel.as_posix()}", p))
    return out


def public_path(path: Path) -> str:
    """Repository-relative path for the bundle manifest (no build-machine paths)."""
    path = path.resolve()
    try:
        return path.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        pass
    posix = path.as_posix()
    for top in ("/state/", "/configs/"):
        if top in posix:
            return top.strip("/") + "/" + posix.split(top, 1)[1]
    return path.name


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def build(edition: str, candidate: Path, evals: list[Path], out_dir: Path) -> Path:
    root = f"crimerisk_{edition}_evidence"
    plan = (
        collect(REPO / "configs", "configs")
        + collect(REPO / "pyproject.toml", "pyproject.toml")
        + collect(REPO / "uv.lock", "uv.lock")
        + collect(candidate / "manifest.json", "candidate/manifest.json")
        + collect(candidate / "validation_summary.json", "candidate/validation_summary.json")
    )
    for ev in evals:
        plan += collect(ev, f"eval/{ev.name}")
    for name, _ in plan:
        if any(is_ags(part) for part in name.split("/")):
            raise SystemExit(f"refusing a member whose path names AGS: {name}")

    members: list[tuple[str, bytes]] = []
    record = []
    for name, path in sorted(plan):
        data, dropped = strip_member(name, path.read_bytes())
        data, rewritten = scrub_paths(name, data)
        members.append((name, data))
        record.append({
            "path": name,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "source": public_path(path),
            "stripped": dropped,
            "paths_rewritten": rewritten,
        })
    manifest = {
        "edition": edition,
        "built_from_commit": git_head(),
        "candidate": public_path(candidate),
        "evaluations": [public_path(e) for e in evals],
        "ags_policy": "columns and JSON keys named with the token 'ags' are stripped, "
                      "except boolean partition flags",
        "path_policy": "build-machine paths in text members are rewritten repo-relative "
                       "(paths_rewritten counts the rewrites per member)",
        "files": record,
    }
    members.append(("MANIFEST.json", (json.dumps(manifest, indent=1) + "\n").encode()))

    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"{root}.tar.gz"
    with open(tar_path, "wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
    ) as gz, tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in sorted(members):
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (out_dir / f"{tar_path.name}.sha256").write_text(f"{digest}  {tar_path.name}\n")
    stripped = sum(1 for r in record if r["stripped"])
    print(
        f"{tar_path} {tar_path.stat().st_size:,} bytes, {len(members)} members, "
        f"{stripped} with AGS fields stripped; sha256 {digest}"
    )
    return tar_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--gold", type=Path, action="append", default=[], required=True)
    ap.add_argument("--level", type=Path, action="append", default=[])
    args = ap.parse_args()
    build(args.edition, args.candidate, args.gold + args.level, DIST / "evidence")


if __name__ == "__main__":
    main()
