from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.candidates import output_artifact_filenames


KEY_COLUMNS = {
    "block_group": "block_group_geoid",
    "tract": "tract_id",
}


def _key_column_for(filename: str) -> str | None:
    if filename.startswith("crimerisk_block_group_"):
        return KEY_COLUMNS["block_group"]
    if filename.startswith("crimerisk_tract_"):
        return KEY_COLUMNS["tract"]
    return None


def _align_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key_col: str | None,
    issues: list[str],
    artifact: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if key_col is None or key_col not in left.columns or key_col not in right.columns:
        return left.reset_index(drop=True), right.reset_index(drop=True)

    left_keys = left[key_col]
    right_keys = right[key_col]
    if left_keys.duplicated().any():
        issues.append(f"{artifact}: promoted has duplicate {key_col} values")
        return left.reset_index(drop=True), right.reset_index(drop=True)
    if right_keys.duplicated().any():
        issues.append(f"{artifact}: candidate has duplicate {key_col} values")
        return left.reset_index(drop=True), right.reset_index(drop=True)
    missing_in_candidate = sorted(set(left_keys.astype(str)) - set(right_keys.astype(str)))
    missing_in_promoted = sorted(set(right_keys.astype(str)) - set(left_keys.astype(str)))
    if missing_in_candidate:
        issues.append(
            f"{artifact}: candidate is missing {len(missing_in_candidate)} {key_col} values "
            f"(first={missing_in_candidate[:5]})"
        )
    if missing_in_promoted:
        issues.append(
            f"{artifact}: promoted is missing {len(missing_in_promoted)} {key_col} values "
            f"(first={missing_in_promoted[:5]})"
        )
    left_indexed = left.set_index(key_col, drop=False).sort_index()
    right_indexed = right.set_index(key_col, drop=False).sort_index()
    return left_indexed.reset_index(drop=True), right_indexed.reset_index(drop=True)


def _equal_or_both_missing(left: pd.Series, right: pd.Series) -> pd.Series:
    equal = left.eq(right).fillna(False)
    return equal | (left.isna() & right.isna())


def _compare_column(
    left: pd.Series,
    right: pd.Series,
    *,
    artifact: str,
    column: str,
    atol: float,
    rtol: float,
) -> tuple[list[str], float]:
    if is_bool_dtype(left.dtype) or is_bool_dtype(right.dtype):
        equal = _equal_or_both_missing(left, right)
        mismatch_count = int((~equal).sum())
        if mismatch_count:
            return [f"{artifact}: column {column} has {mismatch_count} mismatched values"], 0.0
        return [], 0.0

    if is_numeric_dtype(left.dtype) and is_numeric_dtype(right.dtype):
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        both_missing = left_num.isna() & right_num.isna()
        exactly_equal = left_num.eq(right_num).fillna(False)
        diff = (left_num - right_num).abs()
        tolerance = float(atol) + float(rtol) * right_num.abs()
        within_tolerance = diff.le(tolerance).fillna(False)
        mismatch = ~(both_missing | exactly_equal | within_tolerance)
        mismatch_count = int(mismatch.sum())
        finite_diff = diff.replace([np.inf, -np.inf], np.nan)
        max_abs_diff = float(finite_diff.max(skipna=True)) if finite_diff.notna().any() else 0.0
        if mismatch_count:
            return [
                (
                    f"{artifact}: column {column} has {mismatch_count} values outside "
                    f"atol={atol:g}, rtol={rtol:g}; max_abs_diff={max_abs_diff:g}"
                )
            ], max_abs_diff
        return [], max_abs_diff

    equal = _equal_or_both_missing(left, right)
    mismatch_count = int((~equal).sum())
    if mismatch_count:
        return [f"{artifact}: column {column} has {mismatch_count} mismatched values"], 0.0
    return [], 0.0


def _compare_artifact(
    *,
    promoted_dir: Path,
    candidate_dir: Path,
    filename: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    promoted_path = promoted_dir / filename
    candidate_path = candidate_dir / filename
    issues: list[str] = []
    if not promoted_path.exists():
        issues.append(f"{filename}: missing promoted file {promoted_path}")
    if not candidate_path.exists():
        issues.append(f"{filename}: missing candidate file {candidate_path}")
    if issues:
        return {
            "artifact": filename,
            "status": "fail",
            "issues": issues,
            "rows": None,
            "columns": None,
            "max_abs_diff": None,
        }

    promoted = pd.read_parquet(promoted_path)
    candidate = pd.read_parquet(candidate_path)
    if list(promoted.columns) != list(candidate.columns):
        issues.append(f"{filename}: column names/order differ")
    promoted_dtypes = {column: str(dtype) for column, dtype in promoted.dtypes.items()}
    candidate_dtypes = {column: str(dtype) for column, dtype in candidate.dtypes.items()}
    if promoted_dtypes != candidate_dtypes:
        issues.append(f"{filename}: dtypes differ")
    if len(promoted) != len(candidate):
        issues.append(f"{filename}: row count differs ({len(promoted)} promoted vs {len(candidate)} candidate)")

    max_abs_diff = 0.0
    max_abs_diff_column = None
    if not issues:
        promoted_aligned, candidate_aligned = _align_frames(
            promoted,
            candidate,
            key_col=_key_column_for(filename),
            issues=issues,
            artifact=filename,
        )
        if not issues:
            for column in promoted_aligned.columns:
                column_issues, column_max_abs_diff = _compare_column(
                    promoted_aligned[column],
                    candidate_aligned[column],
                    artifact=filename,
                    column=column,
                    atol=atol,
                    rtol=rtol,
                )
                issues.extend(column_issues)
                if column_max_abs_diff > max_abs_diff:
                    max_abs_diff = column_max_abs_diff
                    max_abs_diff_column = column

    return {
        "artifact": filename,
        "status": "fail" if issues else "pass",
        "issues": issues,
        "rows": int(len(promoted)),
        "columns": int(len(promoted.columns)),
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_column": max_abs_diff_column,
    }


def compare_candidate_outputs(
    *,
    promoted_dir: Path,
    candidate_dir: Path,
    year: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    artifacts = [
        _compare_artifact(
            promoted_dir=promoted_dir,
            candidate_dir=candidate_dir,
            filename=filename,
            atol=atol,
            rtol=rtol,
        )
        for filename in output_artifact_filenames(year=year, include_audits=True)
    ]
    issues = [issue for artifact in artifacts for issue in artifact["issues"]]
    max_abs_diff = max(
        (artifact["max_abs_diff"] or 0.0 for artifact in artifacts),
        default=0.0,
    )
    max_abs_diff_artifact = next(
        (artifact["artifact"] for artifact in artifacts if (artifact["max_abs_diff"] or 0.0) == max_abs_diff),
        None,
    )
    total_rows = sum(int(artifact["rows"] or 0) for artifact in artifacts)
    return {
        "status": "fail" if issues else "pass",
        "promoted_dir": str(promoted_dir),
        "candidate_dir": str(candidate_dir),
        "year": int(year),
        "atol": float(atol),
        "rtol": float(rtol),
        "artifact_count": len(artifacts),
        "total_rows_compared": int(total_rows),
        "max_abs_diff": float(max_abs_diff),
        "max_abs_diff_artifact": max_abs_diff_artifact,
        "artifacts": artifacts,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a candidate output directory against the currently promoted state/output parquet set."
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--promoted-dir", type=Path, default=REPO_ROOT / "state" / "output")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    summary = compare_candidate_outputs(
        promoted_dir=args.promoted_dir,
        candidate_dir=args.candidate_dir,
        year=int(args.year),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    summary_out = args.summary_out or (args.candidate_dir / "validation_summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if summary["status"] == "pass":
        print(
            "PASS candidate matches promoted outputs: "
            f"artifacts={summary['artifact_count']} "
            f"rows={summary['total_rows_compared']} "
            f"max_abs_diff={summary['max_abs_diff']:.12g}"
        )
        for artifact in summary["artifacts"]:
            print(
                f"{artifact['artifact']}: rows={artifact['rows']} "
                f"columns={artifact['columns']} max_abs_diff={artifact['max_abs_diff']:.12g}"
            )
        print(f"Wrote {summary_out}")
        return 0

    print("FAIL candidate differs from promoted outputs")
    for issue in summary["issues"]:
        print(issue)
    print(f"Wrote {summary_out}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
