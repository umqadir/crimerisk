from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.review_batches import (  # noqa: E402
    clean_scalar,
    load_exact_identity_rows,
    write_batched_cases,
)

NONLOCAL_MANUAL_PATH = REPO_ROOT / "state" / "review" / "queues" / "local_resolution" / "nonmunicipal_manual_review.parquet"
LEAIC_PATH = REPO_ROOT / "data" / "LEAIC-Crosswalk-ICPSR_35158" / "DS0001" / "35158-0001-Data.tsv"
DEFAULT_OUT_DIR = REPO_ROOT / "state" / "review" / "runs" / "local_resolution" / "nonlocal_second_pass"

PRIORITY_STATES = {"SC", "WV", "DE", "LA", "FL", "GA", "HI", "MD", "NV", "KY", "VA", "PA"}

UNIVERSITY_RE = re.compile(
    r"\bUNIV\b|UNIVERSITY|\bCOLLEGE\b|\bCAMPUS\b|\bUN OF\b|\bU OF\b|STATE UNIVERS|STATE UNIV|"
    r"POLYTECH|POLYTEC|\bSUNY\b|RUTGERS|INSTITUTE OF TEC|INST OF TEC|HIGHER EDUC|MEDICAL U OF|"
    r"\bST UN\b|\bST UNIV\b|UNIVERSI\b|TX TECH|COMMONWEALTH UNIVERSI|INST OF TECHNOLOGY|A M\b|A&M",
    re.I,
)
EDUCATION_RE = re.compile(
    r"SCH DIST|SCHOOL DIST|PUBLIC SCHOOLS|PUB SCHOOLS|BOARD OF EDUCAT|\bISD\b|\bUSD\b|UNIFIED SCH|UNFD SCH|"
    r"\bSCHOOLS\b|EDUC CNTR",
    re.I,
)
TRANSIT_RE = re.compile(
    r"TRANSIT|\bMETRO\b|\bBART\b|\bRAIL\b|\bSUBWAY\b|\bMTA\b|NJ TRANSIT|PORT AUTHORITY|"
    r"TRANSPORTATION AUTHORITY|TRNSPRTN|REG TA|RAPID TR",
    re.I,
)
AIRPORT_PORT_RE = re.compile(r"AIRPORT|AIRP\b|INT AP|INTL AP|INT AIRPO|HARBOR|\bPORT\b|SEAPORT", re.I)
TRIBAL_RE = re.compile(r"\bTRIBAL\b|\bPUEBLO\b|\bRESERVATION\b|\bNATION\b|INDIAN COMMUNITY|INDIAN TRIBE", re.I)
LOCAL_SPECIAL_RE = re.compile(
    r"HOUSING|\bPARK\b|HOSPITAL|HOS DIST|MEDICAL CENTER|MED CTR|DEVELOPMENTAL CTR|WORLD CONGRESS|"
    r"FOREST PRESERVE|NATURAL RESOURCES|ALCOHOLIC BEV CONTROL|HEALTH CARE|DPR ",
    re.I,
)


def classify_deterministic(row: pd.Series) -> tuple[str, str | None, str, str | None]:
    name = str(row.get("agency_name_std") or "")
    if TRIBAL_RE.search(name):
        return (
            "localized_special_overlap",
            "tribal",
            "tribal_keyword",
            "tribal_reservation_or_tribal_land",
        )
    if TRANSIT_RE.search(name):
        return (
            "localized_special_overlap",
            "transit",
            "transit_keyword",
            "network_or_system_footprint",
        )
    if AIRPORT_PORT_RE.search(name):
        return (
            "localized_special_overlap",
            "transport_hub",
            "airport_port_keyword",
            "facility_or_authority_footprint",
        )
    if UNIVERSITY_RE.search(name):
        return (
            "localized_special_overlap",
            "campus",
            "campus_keyword",
            "campus_or_university_footprint",
        )
    if EDUCATION_RE.search(name):
        return (
            "localized_special_overlap",
            "local_special",
            "education_system_keyword",
            "education_system_or_school_district_footprint",
        )
    if LOCAL_SPECIAL_RE.search(name):
        return (
            "localized_special_overlap",
            "local_special",
            "local_special_keyword",
            "authority_or_park_footprint",
        )
    return ("statewide_overlap", None, "default_statewide_overlap", "statewide")


def build_cases() -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(NONLOCAL_MANUAL_PATH).reset_index(drop=True)
    exact_identity_rows = load_exact_identity_rows(LEAIC_PATH)

    deterministic_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        bucket_decision, overlap_subtype, rule_name, geometry_hint = classify_deterministic(row)
        record = row.to_dict()
        record["baseline_bucket_decision"] = bucket_decision
        record["baseline_overlap_subtype"] = overlap_subtype
        record["baseline_rule_name"] = rule_name
        record["baseline_geometry_hint"] = geometry_hint
        deterministic_rows.append(record)
    baseline_df = pd.DataFrame(deterministic_rows)

    latest_counts = pd.to_numeric(baseline_df["latest_srs_part1_total"], errors="coerce").fillna(0.0)
    state_upper = baseline_df["state_abbr"].fillna("").astype(str).str.upper()
    name = baseline_df["agency_name_std"].fillna("").astype(str)

    tribal = name.str.contains(TRIBAL_RE)
    transit = name.str.contains(TRANSIT_RE)
    airport_port = name.str.contains(AIRPORT_PORT_RE)

    baseline_bucket = baseline_df["baseline_bucket_decision"].fillna("").astype(str)
    baseline_subtype = baseline_df["baseline_overlap_subtype"].fillna("").astype(str)

    second_pass_mask = (
        (
            baseline_bucket.eq("statewide_overlap")
            & ((latest_counts >= 10) | (state_upper.isin(PRIORITY_STATES) & (latest_counts > 0)))
        )
        | baseline_subtype.isin(["tribal", "transit", "transport_hub"])
    )
    second_pass_df = baseline_df.loc[second_pass_mask].copy()
    second_pass_df["flag_latest_srs_ge_10"] = latest_counts.loc[second_pass_df.index] >= 10
    second_pass_df["flag_priority_state_positive"] = (
        state_upper.loc[second_pass_df.index].isin(PRIORITY_STATES)
        & (latest_counts.loc[second_pass_df.index] > 0)
    )
    second_pass_df["flag_tribal_family"] = tribal.loc[second_pass_df.index]
    second_pass_df["flag_transit_family"] = transit.loc[second_pass_df.index]
    second_pass_df["flag_airport_port_family"] = airport_port.loc[second_pass_df.index]

    cases: list[dict[str, Any]] = []
    for idx, row in second_pass_df.reset_index(drop=True).iterrows():
        case = {
            "case_id": f"{idx + 1:04d}",
            "queue_rank": idx + 1,
            "ori9": clean_scalar(row.get("ori9")),
            "ori7": clean_scalar(row.get("ori7")),
            "state_abbr": clean_scalar(row.get("state_abbr")),
            "state_fips": clean_scalar(row.get("state_fips")),
            "county_fips": clean_scalar(row.get("county_fips")),
            "place_fips": clean_scalar(row.get("place_fips")),
            "agency_name_raw": clean_scalar(row.get("agency_name_raw")),
            "agency_name_std": clean_scalar(row.get("agency_name_std")),
            "agency_name_std_srs": clean_scalar(row.get("agency_name_std_srs")),
            "crosswalk_agency_name_std": clean_scalar(row.get("crosswalk_agency_name_std")),
            "city_name_std_nibrs": clean_scalar(row.get("city_name_std_nibrs")),
            "census_name_std": clean_scalar(row.get("census_name_std")),
            "agency_type_norm": clean_scalar(row.get("agency_type_norm")),
            "suggested_bucket_class": clean_scalar(row.get("suggested_bucket_class")),
            "review_priority": clean_scalar(row.get("review_priority")),
            "latest_srs_part1_total": clean_scalar(row.get("latest_srs_part1_total")),
            "population_latest_nibrs": clean_scalar(row.get("population_latest_nibrs")),
            "exact_identity_rows": exact_identity_rows.get(str(row.get("ori9")), []),
            "baseline_classification": {
                "bucket_decision": clean_scalar(row.get("baseline_bucket_decision")),
                "overlap_subtype": clean_scalar(row.get("baseline_overlap_subtype")),
                "geometry_hint": clean_scalar(row.get("baseline_geometry_hint")),
                "rule_name": clean_scalar(row.get("baseline_rule_name")),
            },
            "second_pass_flags": [
                flag_name
                for flag_name, active in [
                    ("latest_srs_ge_10", bool(row.get("flag_latest_srs_ge_10"))),
                    ("priority_state_positive", bool(row.get("flag_priority_state_positive"))),
                    ("tribal_family", bool(row.get("flag_tribal_family"))),
                    ("transit_family", bool(row.get("flag_transit_family"))),
                    ("airport_port_family", bool(row.get("flag_airport_port_family"))),
                ]
                if active
            ],
        }
        cases.append(case)

    audit_cols = [
        "ori9",
        "state_abbr",
        "agency_name_std",
        "agency_type_norm",
        "latest_srs_part1_total",
        "review_priority",
        "baseline_bucket_decision",
        "baseline_overlap_subtype",
        "baseline_rule_name",
        "baseline_geometry_hint",
        "flag_latest_srs_ge_10",
        "flag_priority_state_positive",
        "flag_tribal_family",
        "flag_transit_family",
        "flag_airport_port_family",
    ]
    queue_df = second_pass_df[audit_cols].reset_index(drop=True)
    return cases, baseline_df.reset_index(drop=True), queue_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    cases, baseline_df, queue_df = build_cases()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df.to_parquet(out_dir / "nonlocal_deterministic_baseline.parquet", index=False)
    baseline_df.to_csv(out_dir / "nonlocal_deterministic_baseline.csv", index=False)
    queue_df.to_parquet(out_dir / "nonlocal_second_pass_queue.parquet", index=False)
    queue_df.to_csv(out_dir / "nonlocal_second_pass_queue.csv", index=False)
    write_batched_cases(cases, out_dir, args.batch_size)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "baseline_rows": len(baseline_df),
                "second_pass_cases": len(cases),
                "batch_size": args.batch_size,
                "batches": (len(cases) + args.batch_size - 1) // args.batch_size,
                "baseline_bucket_counts": baseline_df["baseline_bucket_decision"].value_counts().to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "baseline_rows": len(baseline_df),
                "second_pass_cases": len(cases),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
