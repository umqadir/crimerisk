from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


def _load_review_results(results_dir: Path) -> pd.DataFrame:
    batch_root = results_dir.parent
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("batch_*_results.json")):
        payload = json.loads(path.read_text())
        match = re.search(r"batch_(\d+)_results\.json$", path.name)
        if not match:
            raise ValueError(f"Unable to parse batch number from {path}")
        batch_num = match.group(1)
        batch_input = json.loads((batch_root / f"batch_{batch_num}.json").read_text())
        case_lookup = {str(row["case_id"]): row for row in batch_input}
        for row in payload:
            out = dict(row)
            case_id = str(out["case_id"])
            batch_row = case_lookup.get(case_id, {})
            out.setdefault("ori9", batch_row.get("ori9"))
            out.setdefault("confidence", None)
            out.setdefault("sources", None)
            rows.append(out)
    reviewed = pd.DataFrame(rows)
    if reviewed["ori9"].isna().any():
        missing = reviewed.loc[reviewed["ori9"].isna(), "case_id"].tolist()
        raise ValueError(f"Missing ori9 after batch backfill for cases: {missing}")
    dupes = reviewed.loc[reviewed.duplicated("ori9", keep=False), ["case_id", "ori9"]]
    if not dupes.empty:
        raise ValueError(f"Duplicate ori9 in review results: {dupes.to_dict(orient='records')}")
    return reviewed


def _load_override_results(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    rows = json.loads(path.read_text())
    reviewed = pd.DataFrame(rows).rename(
        columns={
            "bucket_decision": "high_bucket_decision",
            "overlap_subtype": "high_overlap_subtype",
            "geometry_hint": "high_geometry_hint",
            "confidence": "high_confidence",
            "reason": "high_reason",
            "sources": "high_sources",
        }
    )
    if reviewed.empty:
        return reviewed
    if reviewed["ori9"].isna().any():
        raise ValueError("High-review override rows must include ori9")
    dupes = reviewed.loc[reviewed.duplicated("ori9", keep=False), ["case_id", "ori9"]]
    if not dupes.empty:
        raise ValueError(f"Duplicate ori9 in high-review overrides: {dupes.to_dict(orient='records')}")
    return reviewed


def resolve_nonlocal_queue_final(
    *,
    baseline_path: Path,
    review_results_dir: Path,
    high_review_results_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_parquet(baseline_path).copy()
    reviewed = _load_review_results(review_results_dir).rename(
        columns={
            "bucket_decision": "review_bucket_decision",
            "overlap_subtype": "review_overlap_subtype",
            "geometry_hint": "review_geometry_hint",
            "confidence": "review_confidence",
            "reason": "review_reason",
            "sources": "review_sources",
        }
    )
    high_review = _load_override_results(high_review_results_path)

    final = baseline.merge(reviewed, on="ori9", how="left", validate="one_to_one")
    final["final_bucket_decision"] = final["baseline_bucket_decision"]
    final["final_overlap_subtype"] = final["baseline_overlap_subtype"]
    final["final_geometry_hint"] = final["baseline_geometry_hint"]
    final["confidence"] = None
    final["reason"] = final["baseline_rule_name"].astype(str)
    final["sources"] = None
    final["resolution_source"] = "deterministic_baseline"

    has_review = final["review_bucket_decision"].notna()
    final.loc[has_review, "final_bucket_decision"] = final.loc[has_review, "review_bucket_decision"]
    final.loc[has_review, "final_overlap_subtype"] = final.loc[has_review, "review_overlap_subtype"]
    final.loc[has_review, "final_geometry_hint"] = final.loc[has_review, "review_geometry_hint"]
    final.loc[has_review, "confidence"] = final.loc[has_review, "review_confidence"]
    final.loc[has_review, "reason"] = final.loc[has_review, "review_reason"]
    final.loc[has_review, "sources"] = final.loc[has_review, "review_sources"]
    final.loc[has_review, "resolution_source"] = "mini_second_pass"

    if not high_review.empty:
        final = final.merge(high_review, on="ori9", how="left", validate="one_to_one")
        has_high = final["high_bucket_decision"].notna()
        final.loc[has_high, "final_bucket_decision"] = final.loc[has_high, "high_bucket_decision"]
        final.loc[has_high, "final_overlap_subtype"] = final.loc[has_high, "high_overlap_subtype"]
        final.loc[has_high, "final_geometry_hint"] = final.loc[has_high, "high_geometry_hint"]
        final.loc[has_high, "confidence"] = final.loc[has_high, "high_confidence"]
        final.loc[has_high, "reason"] = final.loc[has_high, "high_reason"]
        final.loc[has_high, "sources"] = final.loc[has_high, "high_sources"]
        final.loc[has_high, "resolution_source"] = "high_second_pass"

    final["confidence_num"] = pd.to_numeric(final["confidence"], errors="coerce")
    latest_counts = pd.to_numeric(final["latest_srs_part1_total"], errors="coerce").fillna(0.0)
    final["needs_high_review"] = final["resolution_source"].ne("high_second_pass") & has_review & (
        (final["confidence_num"] < 0.85)
        | (
            final["final_bucket_decision"].eq("statewide_overlap")
            & (latest_counts >= 100)
        )
    )

    high_review = final.loc[final["needs_high_review"]].copy()
    return final, high_review
