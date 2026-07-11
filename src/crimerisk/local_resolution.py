from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from crimerisk.paths import RepoPaths
from crimerisk.jurisdiction_review import _load_all_tiger_lookups
from crimerisk.reference import _std_text


CONFLICT_RE = re.compile(
    r"different ORI|uses ORI|do not identify ORI|official .* uses|cannot be confirmed to|could not be confirmed to",
    re.I,
)
OVERLAP_HINT_RE = re.compile(
    r"special|campus|tribal|regional|joint|consolidated|airport|transit|port|authority|housing",
    re.I,
)


def _load_second_pass_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("batch_*_results.json")):
        rows.extend(json.loads(path.read_text()))
    return pd.DataFrame(rows)


def _build_label_lookup(paths: RepoPaths) -> pd.DataFrame:
    lookup = _load_all_tiger_lookups(paths=paths)
    if lookup.empty:
        return lookup
    lookup = lookup.copy()
    lookup["name_std_resolved"] = lookup["namelsad"].map(_std_text)
    return lookup[
        ["state_fips", "state_abbr", "geo_type", "geoid", "namelsad", "name_std_resolved"]
    ] if "state_abbr" in lookup.columns else lookup[
        ["state_fips", "geo_type", "geoid", "namelsad", "name_std_resolved"]
    ]


def _ensure_state_abbr(lookup: pd.DataFrame) -> pd.DataFrame:
    if "state_abbr" in lookup.columns:
        return lookup
    state_map = pd.read_csv(
        Path(__file__).resolve().parents[2] / "data" / "state_fips_crosswalk.csv",
        dtype=str,
    )
    state_map = state_map.rename(columns={"state_fips": "state_fips", "state_abbr": "state_abbr"})
    return lookup.merge(state_map[["state_fips", "state_abbr"]], on="state_fips", how="left")


def _fill_missing_geoid(df: pd.DataFrame, *, paths: RepoPaths) -> pd.DataFrame:
    lookup = _load_all_tiger_lookups(paths=paths).copy()
    if lookup.empty:
        return df
    lookup["resolved_geo_type"] = lookup["geo_type"]
    lookup["resolved_label_std"] = lookup["namelsad"].map(_std_text)
    counts = (
        lookup.groupby(["state_fips", "resolved_geo_type", "resolved_label_std"], dropna=False)["geoid"]
        .nunique()
        .rename("label_match_n")
        .reset_index()
    )
    unique_lookup = lookup.merge(
        counts,
        on=["state_fips", "resolved_geo_type", "resolved_label_std"],
        how="left",
    )
    unique_lookup = unique_lookup[unique_lookup["label_match_n"] == 1][
        ["state_fips", "resolved_geo_type", "resolved_label_std", "geoid", "namelsad"]
    ].drop_duplicates()

    out = df.copy()
    out["resolved_label_std"] = out["resolved_label"].map(_std_text)
    fill_mask = (
        out["final_decision"].isin(["municipal_place", "municipal_cousub"])
        & out["resolved_geo_type"].isin(["place", "cousub"])
        & out["resolved_geoid"].isna()
        & out["resolved_label"].notna()
        & out["state_fips"].notna()
    )
    fill_rows = out.loc[fill_mask, ["ori9", "state_fips", "resolved_geo_type", "resolved_label_std"]].copy()
    fill_rows = fill_rows.merge(
        unique_lookup,
        on=["state_fips", "resolved_geo_type", "resolved_label_std"],
        how="left",
    )
    fill_rows = fill_rows.rename(columns={"geoid": "fill_geoid", "namelsad": "fill_label"})
    out = out.merge(fill_rows[["ori9", "fill_geoid", "fill_label"]], on="ori9", how="left")
    can_fill = out["resolved_geoid"].isna() & out["fill_geoid"].notna()
    out.loc[can_fill, "resolved_geoid"] = out.loc[can_fill, "fill_geoid"]
    out.loc[can_fill, "resolved_label"] = out.loc[can_fill, "fill_label"]
    return out.drop(columns=["resolved_label_std", "fill_geoid", "fill_label"], errors="ignore")


def resolve_local_queue_final(
    *,
    local_queue_path: Path,
    first_pass_results_path: Path,
    second_pass_results_dir: Path,
    paths: RepoPaths,
) -> pd.DataFrame:
    queue_df = pd.read_parquet(local_queue_path)
    first = pd.read_csv(first_pass_results_path, dtype={"ori9": str}).rename(
        columns={
            "decision": "first_decision",
            "resolved_geo_type": "first_resolved_geo_type",
            "resolved_geoid": "first_resolved_geoid",
            "resolved_label": "first_resolved_label",
            "confidence": "first_confidence",
            "reason": "first_reason",
            "sources": "first_sources",
        }
    )
    second = _load_second_pass_results(second_pass_results_dir).rename(
        columns={
            "decision": "second_decision",
            "resolved_geo_type": "second_resolved_geo_type",
            "resolved_geoid": "second_resolved_geoid",
            "resolved_label": "second_resolved_label",
            "confidence": "second_confidence",
            "reason": "second_reason",
            "sources": "second_sources",
        }
    )

    merged = queue_df.merge(first, on="ori9", how="left", validate="one_to_one")
    merged = merged.merge(second, on="ori9", how="left", validate="one_to_one")

    final = merged.copy()
    final["second_has_result"] = final["second_decision"].notna()
    final["conflict_signal"] = final["second_reason"].fillna("").str.contains(CONFLICT_RE)
    final["first_keepable"] = (
        final["second_decision"].eq("escalate")
        & final["first_decision"].isin(
            ["municipal_place", "municipal_cousub", "reclassify_overlap", "reclassify_nonmunicipal"]
        )
        & ~final["conflict_signal"]
        & ~(
            final["first_decision"].isin(["municipal_place", "municipal_cousub"])
            & final["first_resolved_geoid"].isna()
        )
    )

    # Default to first pass everywhere.
    final["final_decision"] = final["first_decision"]
    final["resolved_geo_type"] = final["first_resolved_geo_type"]
    final["resolved_geoid"] = final["first_resolved_geoid"]
    final["resolved_label"] = final["first_resolved_label"]
    final["confidence"] = final["first_confidence"]
    final["reason"] = final["first_reason"]
    final["sources"] = final["first_sources"]
    final["resolution_source"] = "first_pass"
    final["fallback_applied"] = False
    final["manual_review_flag"] = False

    # Override with second pass where it produced an affirmative classification.
    second_override = final["second_has_result"] & final["second_decision"].ne("escalate")
    final.loc[second_override, "final_decision"] = final.loc[second_override, "second_decision"]
    final.loc[second_override, "resolved_geo_type"] = final.loc[second_override, "second_resolved_geo_type"]
    final.loc[second_override, "resolved_geoid"] = final.loc[second_override, "second_resolved_geoid"]
    final.loc[second_override, "resolved_label"] = final.loc[second_override, "second_resolved_label"]
    final.loc[second_override, "confidence"] = final.loc[second_override, "second_confidence"]
    final.loc[second_override, "reason"] = final.loc[second_override, "second_reason"]
    final.loc[second_override, "sources"] = final.loc[second_override, "second_sources"]
    final.loc[second_override, "resolution_source"] = "second_pass"

    # Keep a small number of first-pass assignments when second pass only failed to re-prove them.
    final.loc[final["first_keepable"], "resolution_source"] = "first_pass_retained"
    final.loc[final["first_keepable"], "manual_review_flag"] = True

    # Conservative fallback for still-unresolved local-like rows.
    unresolved = final["second_decision"].eq("escalate") & ~final["first_keepable"]
    overlap_fallback = unresolved & (
        final["second_reason"].fillna("").str.contains(OVERLAP_HINT_RE)
        | final["first_decision"].eq("reclassify_overlap")
    )
    nonmunicipal_fallback = unresolved & ~overlap_fallback

    final.loc[overlap_fallback, "final_decision"] = "reclassify_overlap"
    final.loc[overlap_fallback, ["resolved_geo_type", "resolved_geoid", "resolved_label"]] = None
    final.loc[overlap_fallback, "reason"] = (
        "Unresolved local-like ORI after second-pass review; conservatively routed to overlap pending later refinement."
    )
    final.loc[overlap_fallback, "resolution_source"] = "fallback_overlap"
    final.loc[overlap_fallback, "fallback_applied"] = True
    final.loc[overlap_fallback, "manual_review_flag"] = True

    final.loc[nonmunicipal_fallback, "final_decision"] = "reclassify_nonmunicipal"
    final.loc[nonmunicipal_fallback, ["resolved_geo_type", "resolved_geoid", "resolved_label"]] = None
    final.loc[nonmunicipal_fallback, "reason"] = (
        "Unresolved low-support local-like ORI after second-pass review; conservatively routed to the state nonmunicipal remainder."
    )
    final.loc[nonmunicipal_fallback, "resolution_source"] = "fallback_nonmunicipal"
    final.loc[nonmunicipal_fallback, "fallback_applied"] = True
    final.loc[nonmunicipal_fallback, "manual_review_flag"] = True

    final = _fill_missing_geoid(final, paths=paths)

    final["resolved_geo_type"] = final["resolved_geo_type"].replace(
        {"municipal_place": "place", "municipal_cousub": "cousub", "county_subdivision": "cousub"}
    )
    still_unmapped_municipal = (
        final["final_decision"].isin(["municipal_place", "municipal_cousub"])
        & final["resolved_geoid"].isna()
    )
    final.loc[still_unmapped_municipal, "final_decision"] = "reclassify_nonmunicipal"
    final.loc[still_unmapped_municipal, ["resolved_geo_type", "resolved_geoid", "resolved_label"]] = None
    final.loc[still_unmapped_municipal, "reason"] = (
        "Municipal semantics were identified but no current Census municipal geography could be mapped, so the record is conservatively routed to the state nonmunicipal remainder."
    )
    final.loc[still_unmapped_municipal, "resolution_source"] = "fallback_nonmunicipal_unmapped"
    final.loc[still_unmapped_municipal, "fallback_applied"] = True
    final.loc[still_unmapped_municipal, "manual_review_flag"] = True
    final["has_final_municipal_geoid"] = (
        final["final_decision"].isin(["municipal_place", "municipal_cousub"])
        & final["resolved_geoid"].notna()
    )
    return final
