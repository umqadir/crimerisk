from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from crimerisk.paths import RepoPaths
from crimerisk.jurisdiction_review import _load_all_tiger_lookups
from crimerisk.reference import _std_text


BASE_FIELDS = [
    "ori9",
    "ori7",
    "state_abbr",
    "state_fips",
    "county_fips",
    "place_fips",
    "agency_name_raw",
    "agency_name_std",
    "agency_name_std_srs",
    "crosswalk_agency_name_std",
    "city_name_std_nibrs",
    "census_name_std",
    "agency_type_norm",
    "match_status",
    "match_method",
    "candidate_label",
    "candidate_count",
    "candidate_summary",
    "review_priority",
    "latest_srs_part1_total",
]

PROBLEM_LOCAL_DECISIONS = {"escalate"}


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    return str(value)


def _parse_candidate_token(token: str) -> dict[str, str] | None:
    token = token.strip()
    if not token:
        return None
    parts = token.split(":", 2)
    if len(parts) != 3:
        return None
    geo_type, geoid, label = parts
    if not geo_type or not geoid:
        return None
    return {
        "geo_type": geo_type,
        "geoid": geoid,
        "label": label,
    }


def _load_exact_identity_rows(leaic_path: Path) -> dict[str, list[dict[str, Any]]]:
    leaic = pd.read_csv(leaic_path, sep="\t", dtype=str)
    out: dict[str, list[dict[str, Any]]] = {}
    for ori9, grp in leaic.groupby("ORI9", dropna=True):
        rows: list[dict[str, Any]] = []
        for rec in grp.itertuples(index=False):
            rows.append(
                {
                    "source": "leaic_crosswalk_35158",
                    "ori9": _clean_scalar(getattr(rec, "ORI9")),
                    "ori7": _clean_scalar(getattr(rec, "ORI7")),
                    "agency_name": _clean_scalar(getattr(rec, "NAME")),
                    "lg_name": _clean_scalar(getattr(rec, "LG_NAME")),
                    "address_city": _clean_scalar(getattr(rec, "ADDRESS_CITY")),
                    "address_state": _clean_scalar(getattr(rec, "ADDRESS_STATE")),
                    "fips_state": _clean_scalar(getattr(rec, "FIPS_ST")),
                    "fips_county": _clean_scalar(getattr(rec, "FIPS_COUNTY")),
                    "fplace": _clean_scalar(getattr(rec, "FPLACE")),
                    "agcytype": _clean_scalar(getattr(rec, "AGCYTYPE")),
                    "subtype1": _clean_scalar(getattr(rec, "SUBTYPE1")),
                    "subtype2": _clean_scalar(getattr(rec, "SUBTYPE2")),
                }
            )
        out[str(ori9)] = rows
    return out


def _candidate_key(source: str, geo_type: str, geoid: str, label: str) -> tuple[str, str, str, str]:
    return source, geo_type, geoid, label


def _build_queue_candidate_geographies(row: pd.Series) -> list[dict[str, str]]:
    geographies: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(source: str, geo_type: str | None, geoid: str | None, label: str | None) -> None:
        if not geo_type or not geoid or not label:
            return
        key = _candidate_key(source, str(geo_type), str(geoid), str(label))
        if key in seen:
            return
        seen.add(key)
        geographies.append(
            {
                "source": source,
                "geo_type": str(geo_type),
                "geoid": str(geoid),
                "label": str(label),
            }
        )

    geo_type = _clean_scalar(row.get("geo_type"))
    geoid = _clean_scalar(row.get("geoid"))
    candidate_label = _clean_scalar(row.get("candidate_label"))
    if candidate_label and geo_type and geoid:
        add("queue_primary", geo_type, geoid, candidate_label)

    parsed_primary = _parse_candidate_token(str(candidate_label)) if candidate_label else None
    if parsed_primary is not None:
        add(
            "queue_candidate_label",
            parsed_primary["geo_type"],
            parsed_primary["geoid"],
            parsed_primary["label"],
        )

    candidate_summary = _clean_scalar(row.get("candidate_summary"))
    if candidate_summary:
        for token in str(candidate_summary).split(" | "):
            parsed = _parse_candidate_token(token)
            if parsed is None:
                continue
            add(
                "queue_candidate_summary",
                parsed["geo_type"],
                parsed["geoid"],
                parsed["label"],
            )
    return geographies


def _build_tiger_maps(paths: RepoPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = _load_all_tiger_lookups(paths=paths)
    by_code = lookup[["state_fips", "geo_type", "code", "geoid", "namelsad"]].drop_duplicates()
    by_name = lookup[["state_fips", "geo_type", "geoid", "name_std", "namelsad_std", "namelsad"]].drop_duplicates()
    return by_code, by_name


def _build_identity_candidate_geographies(
    *,
    exact_rows: list[dict[str, Any]],
    by_code: pd.DataFrame,
    by_name: pd.DataFrame,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(source: str, geo_type: str, geoid: str, label: str) -> None:
        key = _candidate_key(source, geo_type, geoid, label)
        if key in seen:
            return
        seen.add(key)
        out.append(
            {
                "source": source,
                "geo_type": geo_type,
                "geoid": geoid,
                "label": label,
            }
        )

    for exact in exact_rows:
        state_fips = str(exact.get("fips_state") or "").zfill(2)
        fplace = str(exact.get("fplace") or "").zfill(5)
        if state_fips and fplace:
            matched = by_code[(by_code["state_fips"] == state_fips) & (by_code["code"] == fplace)]
            for rec in matched.itertuples(index=False):
                add("identity_fplace", str(rec.geo_type), str(rec.geoid), str(rec.namelsad))

        lg_name_std = _std_text(exact.get("lg_name"))
        if state_fips and lg_name_std:
            matched = by_name[
                (by_name["state_fips"] == state_fips)
                & ((by_name["name_std"] == lg_name_std) | (by_name["namelsad_std"] == lg_name_std))
            ]
            for rec in matched.itertuples(index=False):
                add("identity_lg_name", str(rec.geo_type), str(rec.geoid), str(rec.namelsad))

    return out


def _merge_candidates(*candidate_lists: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for geographies in candidate_lists:
        for row in geographies:
            key = (row["geo_type"], row["geoid"], row["label"])
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
    return out


def _repeat_counts(df: pd.DataFrame) -> dict[str, int]:
    state_series = df["state_abbr"].fillna("").astype(str)
    agency_series = df["agency_name_std"].fillna("").astype(str)
    named_mask = agency_series.str.strip() != ""
    repeat_key = state_series + "|" + agency_series
    return repeat_key[named_mask].value_counts(dropna=False).to_dict()


def _base_case(row: pd.Series, repeat_counts: dict[str, int], exact_identity_rows: dict[str, list[dict[str, Any]]], by_code: pd.DataFrame, by_name: pd.DataFrame, queue_rank: int) -> dict[str, Any]:
    case: dict[str, Any] = {
        "case_id": f"{queue_rank:04d}",
        "queue_rank": queue_rank,
    }
    for field in BASE_FIELDS:
        case[field] = _clean_scalar(row.get(field))

    key = f"{row.get('state_abbr') or ''}|{row.get('agency_name_std') or ''}"
    case["repeat_cluster"] = {
        "key": key,
        "size": int(repeat_counts.get(key, 0)),
    }
    exact_rows = exact_identity_rows.get(str(row.get("ori9")), [])
    queue_candidates = _build_queue_candidate_geographies(row)
    identity_candidates = _build_identity_candidate_geographies(
        exact_rows=exact_rows,
        by_code=by_code,
        by_name=by_name,
    )
    case["candidate_geographies"] = _merge_candidates(queue_candidates, identity_candidates)
    case["exact_identity_rows"] = exact_rows
    return case


def build_local_enriched_cases(
    *,
    local_queue_path: Path,
    leaic_path: Path,
    paths: RepoPaths,
) -> list[dict[str, Any]]:
    df = pd.read_parquet(local_queue_path).reset_index(drop=True)
    exact_identity_rows = _load_exact_identity_rows(leaic_path)
    repeat_counts = _repeat_counts(df)
    by_code, by_name = _build_tiger_maps(paths)
    cases = [
        _base_case(
            row,
            repeat_counts=repeat_counts,
            exact_identity_rows=exact_identity_rows,
            by_code=by_code,
            by_name=by_name,
            queue_rank=idx + 1,
        )
        for idx, row in df.iterrows()
    ]
    return cases


def build_local_second_pass_cases(
    *,
    local_queue_path: Path,
    first_pass_results_path: Path,
    leaic_path: Path,
    paths: RepoPaths,
) -> list[dict[str, Any]]:
    queue_df = pd.read_parquet(local_queue_path).reset_index(drop=True)
    first_pass = pd.read_csv(first_pass_results_path, dtype={"ori9": str})
    merged = queue_df.merge(first_pass, on="ori9", how="left", validate="one_to_one")
    named_for_cluster = merged["agency_name_std"].fillna("").astype(str).str.strip() != ""
    clusters = (
        merged.loc[named_for_cluster]
        .groupby(["state_abbr", "agency_name_std"], dropna=False)
        .size()
        .reset_index(name="repeat_n")
    )
    merged = merged.merge(clusters, on=["state_abbr", "agency_name_std"], how="left")

    confidence_num = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)
    missing_first_pass_mask = merged["decision"].isna()
    problem_mask = merged["decision"].isin(PROBLEM_LOCAL_DECISIONS) | confidence_num.lt(0.8)
    null_geo_mask = merged["decision"].isin(["municipal_place", "municipal_cousub"]) & merged["resolved_geoid"].isna()
    named_mask = merged["agency_name_std"].fillna("").astype(str).str.strip() != ""
    repeat_mask = (merged["repeat_n"].fillna(0).astype(int) > 1) & named_mask
    second_pass_mask = problem_mask | null_geo_mask | repeat_mask

    exact_identity_rows = _load_exact_identity_rows(leaic_path)
    repeat_counts = _repeat_counts(queue_df)
    by_code, by_name = _build_tiger_maps(paths)

    cases: list[dict[str, Any]] = []
    flagged = merged[second_pass_mask].copy()
    flagged["queue_rank_source"] = flagged.index + 1
    flagged = flagged.reset_index(drop=True)
    for record in flagged.to_dict(orient="records"):
        series = pd.Series(record)
        confidence_value = pd.to_numeric(series.get("confidence"), errors="coerce")
        if pd.isna(confidence_value):
            confidence_value = 0.0
        base = _base_case(
            series,
            repeat_counts=repeat_counts,
            exact_identity_rows=exact_identity_rows,
            by_code=by_code,
            by_name=by_name,
            queue_rank=int(series["queue_rank_source"]),
        )
        base["first_pass_result"] = {
            "decision": _clean_scalar(series.get("decision")),
            "resolved_geo_type": _clean_scalar(series.get("resolved_geo_type")),
            "resolved_geoid": _clean_scalar(series.get("resolved_geoid")),
            "resolved_label": _clean_scalar(series.get("resolved_label")),
            "confidence": _clean_scalar(series.get("confidence")),
            "reason": _clean_scalar(series.get("reason")),
        }
        base["review_flags"] = {
            "first_pass_escalate": bool(series.get("decision") in PROBLEM_LOCAL_DECISIONS),
            "low_confidence": bool(float(confidence_value) < 0.8),
            "missing_first_pass_result": bool(pd.isna(series.get("decision"))),
            "missing_municipal_geoid": bool(
                series.get("decision") in {"municipal_place", "municipal_cousub"} and pd.isna(series.get("resolved_geoid"))
            ),
            "repeat_name_cluster": bool(pd.notna(series.get("repeat_n")) and int(series.get("repeat_n")) > 1),
        }
        cases.append(base)
    return cases


def write_case_batches(cases: list[dict[str, Any]], out_dir: Path, batch_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "all_enriched_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    for idx in range(0, len(cases), batch_size):
        batch_num = idx // batch_size + 1
        batch_rows = cases[idx : idx + batch_size]
        (out_dir / f"batch_{batch_num:03d}.json").write_text(json.dumps(batch_rows, indent=2) + "\n")


# Public aliases for other review-batch builders.
clean_scalar = _clean_scalar
load_exact_identity_rows = _load_exact_identity_rows
write_batched_cases = write_case_batches
