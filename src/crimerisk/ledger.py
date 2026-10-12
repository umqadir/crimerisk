"""The decision ledger: every governed payload unit must have an approved ledger row.

`configs/ledger/decisions.csv` holds one row per decision. The payload itself stays in the
config files the build already reads; the ledger records, per payload unit, who proposed it, on
what evidence, who reviewed it and who approved it, plus `payload_sha256`, the hash of the unit
as approved.

`GOVERNED` maps each governed file (repo-relative) to its unit key columns. A unit is every row
sharing one key; its hash covers all columns of those rows. `require_approved` is called by each
loader of a governed file and raises `LedgerError` when any unit's hash has no approved, active
ledger row, so a hand-edited row stops the build. `scripts/diagnostics/validate_release_outputs.py`
repeats the check (`_check_ledger`, blocking).

The ledger is append-only. A changed decision is a new row whose `supersedes` names the old
`decision_id`; a row is effective when its `status` is `active` and no later row supersedes it.
Rows with `human_approval` `approved` or `legacy` count as approved. A rejected proposal changes no
payload and carries no hash. An approved removal carries `REMOVED_SHA256`, the hash of an empty unit.

Only files under the running checkout's repository root are governed. A test that builds its own
repo tree under a temporary directory reads its own fixture configs, which are not governed files.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from crimerisk.paths import get_repo_root

LEDGER_RELPATH = "configs/ledger/decisions.csv"

LEDGER_COLUMNS = (
    "decision_id",
    "kind",
    "subject",
    "target_config",
    "target_key",
    "payload_sha256",
    "edition",
    "evidence_refs",
    "packet_path",
    "packet_sha256",
    "rule_applied",
    "drafter_model",
    "drafter_effort",
    "draft_sha256",
    "reviewer_model",
    "review_verdict",
    "review_sha256",
    "prompt_sha256",
    "cli_version",
    "checks",
    "affects_published",
    "human_approval",
    "approved_by",
    "approved_at",
    "approval_note",
    "supersedes",
    "status",
)

KINDS = frozenset(
    {
        "source_contract",
        "offense_map",
        "feed_admission",
        "coordinate_quarantine",
        "agency_record",
        "level_admission",
        "hard_evidence",
        "reporting_regime",
        "geography",
        "method_change",
        "correction",
        "release_finding",
        "evaluation_seal",
        "legacy",
    }
)
HUMAN_APPROVAL = frozenset({"approved", "rejected", "pending", "legacy"})
APPROVED = frozenset({"approved", "legacy"})
REVIEW_VERDICTS = frozenset({"", "accept", "revise", "reject", "escalate"})
STATUSES = frozenset({"active", "superseded"})


@dataclass(frozen=True)
class Governed:
    key_columns: tuple[str, ...]
    kind: str  # "csv", "parquet" or "json"


# Repo-relative path -> unit definition. `city_feeds/*.json` is a pattern: one unit per file,
# keyed by the file stem as `city_key`.
GOVERNED: dict[str, Governed] = {
    "configs/stage1_adjudications/twins_adjudicated.csv": Governed(("case_id",), "csv"),
    "configs/stage1_adjudications/zero_missing_adjudicated.csv": Governed(("case_id",), "csv"),
    "configs/stage1_adjudications/token_reporters_adjudicated.csv": Governed(("case_id",), "csv"),
    "configs/stage1_adjudications/ori_successions.csv": Governed(("case_id",), "csv"),
    "configs/level_lane_admission_registry.csv": Governed(("case_id",), "csv"),
    "configs/level_lane_hard_evidence.csv": Governed(("case_id",), "csv"),
    "configs/reporting_regime_restrictions.csv": Governed(("restriction_id",), "csv"),
    "configs/service_relationships.csv": Governed(("relationship_id",), "csv"),
    "configs/service_territory_exceptions.csv": Governed(("ori9",), "csv"),
    "configs/state_anchor_exceptions.csv": Governed(("state_abbr", "offense"), "csv"),
    "configs/current_year_confirmations.csv": Governed(("jurisdiction_id", "offense", "year"), "csv"),
    "configs/level_lane_structural_zero_screen.csv": Governed(("ori9",), "csv"),
    "configs/reporting_regime_overrides.csv": Governed(("ori", "year", "offense"), "csv"),
    "configs/source_preference_overrides.csv": Governed(("ori", "year", "offense"), "csv"),
    "configs/local_resolution_overrides.csv": Governed(("ori",), "csv"),
    "configs/municipal_geometry_overrides.csv": Governed(("jurisdiction_id",), "csv"),
    "configs/overlap_footprint_overrides.csv": Governed(("ori",), "csv"),
    "configs/overlap_custom_footprints.csv": Governed(("ori",), "csv"),
    "configs/concurrent_jurisdiction_carveouts.csv": Governed(("county_geoid",), "csv"),
    "configs/service_wide_agency_scopes.csv": Governed(("service_scope_id",), "csv"),
    "configs/service_wide_footprint_coverage.csv": Governed(("service_scope_id",), "csv"),
    "configs/primary_service_response_policies.csv": Governed(("service_scope_id",), "csv"),
    "configs/city_incident_sources.csv": Governed(("city_key",), "csv"),
    "configs/city_offense_admissions.csv": Governed(("city_key", "offense"), "csv"),
    "configs/city_feed_coordinate_quarantine.csv": Governed(("city_key", "offense"), "csv"),
    "configs/city_offense_texture_policy.csv": Governed(("city_key", "offense"), "csv"),
    "configs/city_feeds/*.json": Governed(("city_key",), "json"),
    "configs/truth_source_seals.csv": Governed(("city_key",), "csv"),
    "state/reference/inputs/local_queue_resolved_final.parquet": Governed(("ori9",), "parquet"),
    "state/reference/inputs/nonmunicipal_special_resolved_final.parquet": Governed(("ori9",), "parquet"),
}


REMOVED_SHA256 = hashlib.sha256(b"").hexdigest()


class LedgerError(RuntimeError):
    """A governed payload unit has no approved ledger row, or the ledger itself is malformed."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def governed_relpath(path: Path | str, repo_root: Path | None = None) -> str | None:
    """Repo-relative GOVERNED entry for `path`, or None if the path is not governed."""
    root = Path(repo_root or get_repo_root()).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        rel = candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return None
    if rel in GOVERNED:
        return rel
    if rel.startswith("configs/city_feeds/") and rel.endswith(".json") and rel.count("/") == 2:
        return "configs/city_feeds/*.json"
    return None


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start = 0
    while start < len(lines) and lines[start].startswith("#"):
        start += 1
    reader = csv.reader(io.StringIO("".join(lines[start:])))
    rows = [row for row in reader if row]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _canonical_row(header: list[str], row: list[str]) -> str:
    pairs = sorted(zip(header, row))
    return "\x1f".join(f"{name}={value}" for name, value in pairs)


def _parquet_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    frame = pd.read_parquet(path)
    header = [str(c) for c in frame.columns]
    values = frame.astype(object).where(frame.notna(), None)
    rows = [
        ["" if v is None else repr(float(v)) if isinstance(v, float) else str(v) for v in rec]
        for rec in values.itertuples(index=False, name=None)
    ]
    return header, rows


def payload_units(path: Path | str, repo_root: Path | None = None) -> dict[str, str]:
    """{target_key: payload_sha256} for every unit in a governed file."""
    root = Path(repo_root or get_repo_root())
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    rel = governed_relpath(path, root)
    if rel is None:
        raise LedgerError(f"{path} is not a governed file")
    spec = GOVERNED[rel]
    if spec.kind == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {f"city_key={path.stem}": _sha256(canonical)}
    header, rows = _parquet_rows(path) if spec.kind == "parquet" else _read_csv_rows(path)
    missing = [c for c in spec.key_columns if c not in header]
    if missing:
        raise LedgerError(f"{rel} lacks unit key columns {missing}")
    idx = [header.index(c) for c in spec.key_columns]
    grouped: dict[str, list[str]] = {}
    for row in rows:
        row = list(row) + [""] * (len(header) - len(row))
        key = ";".join(f"{c}={row[i]}" for c, i in zip(spec.key_columns, idx))
        grouped.setdefault(key, []).append(_canonical_row(header, row))
    return {key: _sha256("\n".join(sorted(lines))) for key, lines in grouped.items()}


def ledger_path(repo_root: Path | None = None) -> Path:
    return Path(repo_root or get_repo_root()) / LEDGER_RELPATH


def read_ledger(repo_root: Path | None = None) -> pd.DataFrame:
    path = ledger_path(repo_root)
    if not path.exists():
        return pd.DataFrame(columns=list(LEDGER_COLUMNS), dtype="string")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in LEDGER_COLUMNS if c not in frame.columns]
    if missing:
        raise LedgerError(f"{path} lacks columns {missing}")
    return frame[list(LEDGER_COLUMNS)]


def effective_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Rows with status active that no other row supersedes."""
    superseded = set(ledger["supersedes"].replace("", pd.NA).dropna())
    return ledger[ledger["status"].eq("active") & ~ledger["decision_id"].isin(superseded)]


def integrity_issues(ledger: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    dup = ledger["decision_id"][ledger["decision_id"].duplicated()]
    if len(dup):
        issues.append(f"duplicate decision_id: {sorted(set(dup))[:5]}")
    for column, allowed in (
        ("kind", KINDS),
        ("human_approval", HUMAN_APPROVAL),
        ("review_verdict", REVIEW_VERDICTS),
        ("status", STATUSES),
    ):
        bad = sorted(set(ledger[column]) - allowed)
        if bad:
            issues.append(f"{column} has values outside {sorted(allowed)}: {bad[:5]}")
    ids = set(ledger["decision_id"])
    dangling = sorted(set(ledger["supersedes"].replace("", pd.NA).dropna()) - ids)
    if dangling:
        issues.append(f"supersedes names unknown decision_id: {dangling[:5]}")
    approved_blank = ledger["human_approval"].eq("approved") & ledger["approved_by"].eq("")
    if approved_blank.any():
        issues.append(f"{int(approved_blank.sum())} approved rows have no approved_by")
    unhashed = (
        ledger["target_config"].ne("code")
        & ledger["payload_sha256"].eq("")
        & ledger["kind"].ne("release_finding")
        & ledger["human_approval"].isin(APPROVED)
    )
    if unhashed.any():
        issues.append(f"{int(unhashed.sum())} payload rows have no payload_sha256")
    return issues


@lru_cache(maxsize=8)
def _approved_index(ledger_file: str, mtime_ns: int, size: int) -> frozenset[tuple[str, str, str]]:
    ledger = read_ledger(Path(ledger_file).parents[2])
    issues = integrity_issues(ledger)
    if issues:
        raise LedgerError(f"{ledger_file} fails integrity: {issues}")
    rows = effective_rows(ledger)
    rows = rows[rows["human_approval"].isin(APPROVED)]
    return frozenset(zip(rows["target_config"], rows["target_key"], rows["payload_sha256"]))


def approved_index(repo_root: Path | None = None) -> frozenset[tuple[str, str, str]]:
    path = ledger_path(repo_root)
    if not path.exists():
        return frozenset()
    stat = path.stat()
    return _approved_index(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def unapproved_units(path: Path | str, repo_root: Path | None = None) -> list[str]:
    """target_keys in a governed file whose current hash has no approved, effective ledger row."""
    root = Path(repo_root or get_repo_root())
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    rel_pattern = governed_relpath(path, root)
    if rel_pattern is None:
        return []
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    index = approved_index(root)
    return [key for key, sha in payload_units(path, root).items() if (rel, key, sha) not in index]


def require_approved(path: Path | str, frame: object = None) -> None:
    """Fail closed when a governed file carries a unit with no approved ledger row.

    `frame` is accepted for call-site symmetry and ignored: the check hashes the file itself so
    that no loader-side transformation can mask an edit. Paths outside the checkout are not
    governed and return immediately.
    """
    del frame
    root = get_repo_root()
    path = Path(path)
    if governed_relpath(path, root) is None or not (path if path.is_absolute() else root / path).exists():
        return
    missing = unapproved_units(path, root)
    if missing:
        raise LedgerError(
            f"{path}: {len(missing)} payload unit(s) have no approved row in {LEDGER_RELPATH} "
            f"(first: {missing[:5]}). Governed files change only through scripts/agents/ledger.py approve."
        )


def governed_files(repo_root: Path | None = None) -> list[Path]:
    root = Path(repo_root or get_repo_root())
    files: list[Path] = []
    for rel in GOVERNED:
        if "*" in rel:
            files.extend(sorted(root.glob(rel)))
        elif (root / rel).exists():
            files.append(root / rel)
    return files


def check_all(repo_root: Path | None = None) -> list[str]:
    """Every issue across the ledger and all governed files; empty means clean."""
    root = Path(repo_root or get_repo_root())
    issues = [f"ledger: {m}" for m in integrity_issues(read_ledger(root))]
    if issues:
        return issues
    for path in governed_files(root):
        missing = unapproved_units(path, root)
        if missing:
            rel = path.relative_to(root).as_posix()
            issues.append(f"{rel}: {len(missing)} unapproved unit(s), first {missing[:3]}")
    return issues
