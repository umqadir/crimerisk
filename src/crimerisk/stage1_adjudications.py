"""The Stage-1 ad-hoc adjudications, read through one loader.

Three deterministic rules resolve the Stage-1 defect classes at scale (cross-lane twin identity,
ORI succession, zero-vs-missing at emission, partial-year semantics, the fill recency bound). Each
of them leaves a residue the rule cannot reach: an ORI7 block holding a campus PD and a city PD
looks exactly like one agency filing twice; a full-year-backed published zero looks exactly like a
year of no crime; a compliant twelve-month header in front of one motor-vehicle theft looks exactly
like a very quiet agency. Those residues were adjudicated case by case, and this module is how the
verdicts reach the build.

**Rule first, registry second.** These are not overrides of the rules -- they are the cases the
rules deliberately declined to decide, so each consumption point runs its rule to completion and
then applies registry rows for the survivors. Where a rule and a registry row speak to the same
agency the loader fails closed rather than picking a winner: that means the population split has
moved and the rule's own scope needs re-reading, not that a precedence order needs inventing.

Registry -> site
----------------
| registry | verdict | what it gates | site |
|---|---|---|---|
| `twins_adjudicated.csv` | `same_agency_merge` | the cross-lane twin ledger: variant ORIs re-key to the canonical | `agency_identity.build_cross_lane_twin_ledger` |
| | `superseded_ori` | the succession ledger: the dead ORI gets no estimate row | `agency_identity.build_ori_succession_ledger` |
| | `distinct_agencies` | a NEGATIVE constraint: the rule must not have merged this pair | both of the above |
| `zero_missing_adjudicated.csv` | `misread_missing` | the reporting regime: the agency-year reads `structurally_missing_or_unreliable` | `reporting_regimes.build_agency_year_reporting_regimes` |
| | `genuine_zero_year` | nothing -- the published zero stands | -- |
| `token_reporters_adjudicated.csv` | `token_reporting_flag` | estimator usability: the year is not usable-as-observed, so the existing fill / imputation ladders take it | `trend_fills.apply_stage1_adjudicated_usability` |
| | `genuine_low_crime` | nothing -- the low published value stands | -- |

No new estimator behaviour is introduced anywhere. A verdict can only set the flags the pipeline
already keys on (`reporting_regime`, `usable_as_observed`, `current_row_is_true_partial`,
`preferred_months_reported`) or add a row to a ledger that already exists. Every path a registry row
sends an agency-year down is a path the build already had.

Fail-closed contract
--------------------
* the provenance header must be present and must carry every declared key
* `rows_written` in the header must equal the number of data rows (the promotion script and the file
  cannot silently disagree)
* required columns, unknown verdicts, unknown downstream actions and duplicate case ids all raise
* `target_year` must be a single year and must match the year the consumer asked for
* a registry that resolves to ZERO applicable rows raises. A registry row that quietly does nothing
  is the same defect class as a fail-open, and the repo already treats it that way
  (`allocation._build_exclusive_footprint_displacement`).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

CONFIG_DIRNAME = "stage1_adjudications"
TWINS_FILENAME = "twins_adjudicated.csv"
ZEROS_FILENAME = "zero_missing_adjudicated.csv"
TOKENS_FILENAME = "token_reporters_adjudicated.csv"

PROVENANCE_KEYS = (
    "source_registry",
    "source_pass",
    "target_year",
    "generated_by",
    "generated_at",
    "source_sha256",
    "rows_written",
)

TWIN_COLUMNS = (
    "case_id",
    "state",
    "oris",
    "verdict",
    "canonical_ori",
    "downstream_action",
    "confidence",
    "needs_review",
)
ZERO_COLUMNS = (
    "case_id",
    "target_year",
    "state",
    "oris",
    "verdict",
    "downstream_action",
    "believable_months",
    "confidence",
    "needs_review",
)
TOKEN_COLUMNS = ZERO_COLUMNS + ("repair_value_hint",)

VALID_TWIN_VERDICTS = frozenset(
    {"same_agency_merge", "superseded_ori", "distinct_agencies", "unclear_escalate"}
)
VALID_ZERO_TOKEN_VERDICTS = frozenset(
    {
        "misread_missing",
        "genuine_zero_year",
        "token_reporting_flag",
        "genuine_low_crime",
        "unclear_escalate",
    }
)
VALID_TWIN_ACTIONS = frozenset({"merge_dedupe", "keep_all_oris", "escalate_hold", "review"})

# The usability directive each zero/token action produces. This mapping IS the consumption
# semantics, stated once: everything else in this module and at the three sites reads it.
#
# `reads_missing`  the agency-year is not a measured year; it leaves "observed" and the existing
#                  fill / imputation ladders take it. `flag_review` lands here because a
#                  token-reporting flag with no repair basis still says the published value is not a
#                  measured full year -- which is the whole content of the verdict.
# `partial_year`   the reviewer named a believable month count, so the row becomes the true-partial
#                  the pipeline already knows how to annualise. With no month count there is no
#                  ratio to form, so the row degrades to `reads_missing` rather than having a month
#                  count invented for it.
# `stands`         the published value is retained. No pipeline change.
# `unresolved`     the reviewer declined to rule and no supervisor disposition replaced it. No
#                  pipeline change, counted and reported so it cannot be mistaken for `stands`.
ACTION_DIRECTIVE = {
    "treat_missing_repair": "reads_missing",
    "flag_review": "reads_missing",
    "partial_year_uplift": "partial_year",
    "retain_zero": "stands",
    "accept_observed": "stands",
    "retain_current_flag_review": "stands",
    "escalate_hold": "unresolved",
}
VALID_ZERO_TOKEN_ACTIONS = frozenset(ACTION_DIRECTIVE)

DIRECTIVE_COLUMNS = [
    "ori9",
    "year",
    "directive",
    "believable_months",
    "verdict",
    "case_id",
    "source_registry",
    "confidence",
    "needs_review",
]


class Stage1AdjudicationError(ValueError):
    """Raised when an adjudication registry is absent, malformed or inert."""


def config_dir(paths) -> Path:
    return Path(paths.repo_root) / "configs" / CONFIG_DIRNAME


def _read_with_provenance_header(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a registry whose leading `#` lines are a provenance header.

    Only lines at the TOP of the file are treated as header, so a `#` inside a quoted reviewer note
    is never mistaken for one. The header is required: an adjudication registry without provenance
    is indistinguishable from a hand-edited file, and this whole surface exists so that a verdict in
    the build can be traced to the packet it came from.
    """
    if not path.exists():
        raise Stage1AdjudicationError(
            f"Stage-1 adjudication registry {path} does not exist; run "
            "scripts/review/source_audit/promote_stage1_adjudications_to_configs.py"
        )
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    header_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            break
        header_lines.append(line)
    else:
        index = len(lines)
    provenance: dict[str, str] = {}
    for line in header_lines:
        stripped = line.lstrip("#").strip()
        if ": " not in stripped:
            continue
        key, value = stripped.split(": ", 1)
        provenance.setdefault(key.strip(), value.strip())
    missing = [key for key in PROVENANCE_KEYS if key not in provenance]
    if missing:
        raise Stage1AdjudicationError(
            f"{path} provenance header is missing keys {missing}; it must be regenerated by "
            "scripts/review/source_audit/promote_stage1_adjudications_to_configs.py"
        )
    frame = pd.read_csv(io.StringIO("".join(lines[index:])), dtype="string")
    declared = int(provenance["rows_written"])
    if len(frame) != declared:
        raise Stage1AdjudicationError(
            f"{path} declares rows_written={declared} but carries {len(frame)} data rows; the file "
            "has been edited by hand since it was promoted"
        )
    return frame, provenance


def _validate(
    frame: pd.DataFrame,
    provenance: dict[str, str],
    *,
    path: Path,
    columns: tuple[str, ...],
    verdicts: frozenset[str],
    actions: frozenset[str] | None,
    target_year: int | None,
) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise Stage1AdjudicationError(f"{path} is missing columns {missing}")
    out = frame.reindex(columns=list(dict.fromkeys(list(columns) + list(frame.columns)))).copy()
    for column in out.columns:
        out[column] = out[column].astype("string").fillna("")
    bad_verdict = sorted(set(out.loc[~out["verdict"].isin(verdicts), "verdict"]))
    if bad_verdict:
        raise Stage1AdjudicationError(
            f"{path} carries unknown verdicts {bad_verdict} (valid: {sorted(verdicts)})"
        )
    if actions is not None:
        bad_action = sorted(set(out.loc[~out["downstream_action"].isin(actions), "downstream_action"]))
        if bad_action:
            raise Stage1AdjudicationError(
                f"{path} carries unknown downstream_action {bad_action} (valid: {sorted(actions)})"
            )
    duplicated = sorted(set(out.loc[out["case_id"].duplicated(keep=False), "case_id"]))
    if duplicated:
        raise Stage1AdjudicationError(f"{path} carries duplicate case_id {duplicated}")
    if "target_year" in columns:
        years = sorted(set(out["target_year"]))
        if len(years) != 1:
            raise Stage1AdjudicationError(f"{path} spans several target years {years}")
        year = int(years[0])
        if target_year is not None and year != int(target_year):
            raise Stage1AdjudicationError(
                f"{path} is adjudicated for {year} but the build asked for {int(target_year)}; a "
                "registry adjudicated against one year's evidence must not be applied to another"
            )
        out["year"] = year
    if out.empty:
        raise Stage1AdjudicationError(f"{path} carries no rows")
    out["source_registry"] = provenance["source_registry"]
    return out


def split_oris(value: object) -> list[str]:
    """The registries separate ORIs with `;`; pass-1 rows used `|` before pass 4 normalised them."""
    text = "" if value is None or pd.isna(value) else str(value)
    return [part.strip().upper() for part in text.replace("|", ";").split(";") if part.strip()]


def load_twin_rulings(paths) -> pd.DataFrame:
    path = config_dir(paths) / TWINS_FILENAME
    frame, provenance = _read_with_provenance_header(path)
    out = _validate(
        frame,
        provenance,
        path=path,
        columns=TWIN_COLUMNS,
        verdicts=VALID_TWIN_VERDICTS,
        actions=VALID_TWIN_ACTIONS,
        target_year=None,
    )
    out["ori_list"] = out["oris"].map(split_oris)
    out["canonical_ori"] = out["canonical_ori"].str.strip().str.upper()
    needs_canonical = out["verdict"].isin({"same_agency_merge", "superseded_ori"})
    blank = out.loc[needs_canonical & out["canonical_ori"].eq(""), "case_id"].tolist()
    if blank:
        raise Stage1AdjudicationError(
            f"{path}: merge/supersede rulings without a canonical_ori {blank}"
        )
    not_a_member = [
        case
        for case, canonical, members in zip(
            out["case_id"], out["canonical_ori"], out["ori_list"], strict=True
        )
        if canonical and canonical not in members
    ]
    if not_a_member:
        raise Stage1AdjudicationError(
            f"{path}: canonical_ori is not one of the ruling's own ORIs for {not_a_member}"
        )
    return out


def _load_zero_token(paths, filename, columns, *, target_year) -> pd.DataFrame:
    path = config_dir(paths) / filename
    frame, provenance = _read_with_provenance_header(path)
    out = _validate(
        frame,
        provenance,
        path=path,
        columns=columns,
        verdicts=VALID_ZERO_TOKEN_VERDICTS,
        actions=VALID_ZERO_TOKEN_ACTIONS,
        target_year=target_year,
    )
    out["ori_list"] = out["oris"].map(split_oris)
    return out


def load_zero_missing_rulings(paths, *, target_year: int) -> pd.DataFrame:
    return _load_zero_token(paths, ZEROS_FILENAME, ZERO_COLUMNS, target_year=target_year)


def load_token_reporter_rulings(paths, *, target_year: int) -> pd.DataFrame:
    return _load_zero_token(paths, TOKENS_FILENAME, TOKEN_COLUMNS, target_year=target_year)


def _directives(rulings: pd.DataFrame) -> pd.DataFrame:
    """One row per (ori9, year) carrying the directive its ruling implies."""
    records = []
    for row in rulings.to_dict(orient="records"):
        directive = ACTION_DIRECTIVE[row["downstream_action"]]
        months = pd.to_numeric(row.get("believable_months") or None, errors="coerce")
        if directive == "partial_year" and not (
            pd.notna(months) and 1.0 <= float(months) <= 11.0
        ):
            # No believable month count, so no month ratio exists. The verdict still stands: the
            # year is not a measured full year, so it goes to the ladder rather than being
            # annualised by a month count nobody supplied.
            directive = "reads_missing"
            months = pd.NA
        for ori in row["ori_list"]:
            records.append(
                {
                    "ori9": ori,
                    "year": int(row["year"]),
                    "directive": directive,
                    "believable_months": float(months) if pd.notna(months) else pd.NA,
                    "verdict": row["verdict"],
                    "case_id": row["case_id"],
                    "source_registry": row["source_registry"],
                    "confidence": row["confidence"],
                    "needs_review": row["needs_review"],
                }
            )
    frame = pd.DataFrame(records, columns=DIRECTIVE_COLUMNS)
    if frame.empty:
        return frame
    frame["ori9"] = frame["ori9"].astype("string")
    frame["believable_months"] = pd.to_numeric(frame["believable_months"], errors="coerce")
    duplicated = frame.loc[frame.duplicated(["ori9", "year"], keep=False)]
    if not duplicated.empty:
        raise Stage1AdjudicationError(
            "two adjudications claim one agency-year, which the review passes resolve before "
            "publishing a registry: "
            + str(duplicated[["ori9", "year", "case_id", "directive"]].to_dict(orient="records"))
        )
    return frame


def build_zero_missing_directives(paths, *, target_year: int) -> pd.DataFrame:
    """Zero-vs-missing directives, for the reporting-regime site."""
    return _directives(load_zero_missing_rulings(paths, target_year=int(target_year)))


def build_token_reporter_directives(paths, *, target_year: int) -> pd.DataFrame:
    """Token-reporter directives, for the estimator-usability site."""
    return _directives(load_token_reporter_rulings(paths, target_year=int(target_year)))


def build_usability_directives(paths, *, target_year: int) -> pd.DataFrame:
    """Every directive that takes an agency-year out of "measured as published", both registries.

    The two registries are separate questions asked of the same 2024 agency-years, and each has its
    own site, but the ESTIMATOR sees one population: a year is either usable as observed or it is
    not. The zero registry also carries three token-shaped rulings (a published zero whose reviewer
    found a believable month count), which no regime value can express, so they arrive here too.
    """
    zeros = build_zero_missing_directives(paths, target_year=int(target_year))
    tokens = build_token_reporter_directives(paths, target_year=int(target_year))
    frame = pd.concat([zeros, tokens], ignore_index=True)
    frame = frame[frame["directive"].isin({"reads_missing", "partial_year"})].copy()
    duplicated = frame.loc[frame.duplicated(["ori9", "year"], keep=False)]
    if not duplicated.empty:
        raise Stage1AdjudicationError(
            "the zero and token registries both direct the same agency-year away from observed; "
            "one ORI carries two live rulings and the review chain should have resolved it: "
            + str(duplicated[["ori9", "year", "case_id", "source_registry", "directive"]].to_dict(
                orient="records"))
        )
    return frame.reset_index(drop=True)


def directive_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    counts = {
        f"directive_{name}": int(value)
        for name, value in frame["directive"].value_counts().sort_index().items()
    }
    counts["agency_years"] = int(len(frame))
    counts["oris"] = int(frame["ori9"].nunique())
    return counts


def assert_registry_bit(counts: dict[str, int], *, what: str, expected_key: str) -> None:
    """Fail closed when a registry that carries rows changed nothing.

    The repo's registry contract treats an inert row as the same defect class as a fail-open: a
    verdict that reaches the build and moves no agency-year is a wiring failure wearing the costume
    of a clean run.
    """
    if not counts.get(expected_key):
        raise Stage1AdjudicationError(
            f"{what}: the adjudication registry carries rows but {expected_key} is 0 -- the "
            f"registry did not reach the build. Counts: {counts}"
        )


def write_provenance_manifest(paths, *, target_year: int) -> dict[str, object]:
    """The three registries' provenance headers plus row counts, for the build summaries."""
    manifest: dict[str, object] = {}
    for name, filename in (
        ("twins", TWINS_FILENAME),
        ("zeros", ZEROS_FILENAME),
        ("tokens", TOKENS_FILENAME),
    ):
        frame, provenance = _read_with_provenance_header(config_dir(paths) / filename)
        manifest[name] = {
            "rows": int(len(frame)),
            "source_sha256": provenance["source_sha256"],
            "generated_at": provenance["generated_at"],
        }
    manifest["target_year"] = int(target_year)
    return manifest


def registry_dependency_paths(paths) -> list[Path]:
    """The registry files, for the artifact-freshness dependency lists."""
    directory = config_dir(paths)
    return [directory / TWINS_FILENAME, directory / ZEROS_FILENAME, directory / TOKENS_FILENAME]


def _csv_field_size_guard() -> None:
    csv.field_size_limit(1 << 24)


_csv_field_size_guard()
