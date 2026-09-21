"""The traced level-lane cases, with the agency's actual filed counts beside the control.

Reads whatever `state/` currently holds, so the same command run before and after a
change gives the two halves of a before/after table. Nothing here is case-specific
logic; it is a reader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crime import OFFENSES_7  # noqa: E402
from crimerisk.paths import get_paths  # noqa: E402


# ori9 for the agency cases; jurisdiction_id for the two remainder cases.
AGENCY_CASES: dict[str, str] = {
    "Cicero IL": "IL0162100",
    "Jacksonville FL": "FL0160000",
    "Kansas City MO": "MOKPD0000",
    "Jackson MS (Hinds)": "MS0250100",
    "Memphis TN (control)": "TNMPD0000",
}

REMAINDER_CASES: dict[str, str] = {
    "Owsley KY (state remainder)": "21:state_nonmunicipal_remainder",
    "Essex VT (state remainder)": "50:state_nonmunicipal_remainder",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs) if path.exists() else pd.DataFrame()


def agency_case(
    *,
    name: str,
    ori9: str,
    observations: pd.DataFrame,
    admission: pd.DataFrame,
    ledger: pd.DataFrame,
    year: int,
) -> dict[str, object]:
    filed = observations[observations["ori9"].eq(ori9)]
    filed_wide = (
        filed.pivot_table(index="year", columns="offense", values="count", aggfunc="max")
        .reindex(columns=list(OFFENSES_7))
        if not filed.empty
        else pd.DataFrame()
    )
    admitted = admission[admission["ori9"].eq(ori9)] if not admission.empty else pd.DataFrame()
    mass = ledger[ledger["ori9"].eq(ori9)] if not ledger.empty else pd.DataFrame()
    record: dict[str, object] = {
        "case": name,
        "ori9": ori9,
        "filed_by_year": (
            {int(y): {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
             for y, row in filed_wide.iterrows()}
            if not filed_wide.empty
            else {}
        ),
        "target_year_filed_total": (
            float(filed_wide.loc[year].sum()) if year in filed_wide.index else None
        ),
        "target_year_sources": sorted(
            set(filed[filed["year"].eq(year)]["raw_data_source"].dropna().astype(str))
        )
        if not filed.empty
        else [],
        "target_year_months_reported": (
            float(filed[filed["year"].eq(year)]["months_reported"].max())
            if not filed.empty and filed["year"].eq(year).any()
            else None
        ),
    }
    if not admitted.empty:
        record["admission"] = {
            str(row.offense): {
                "level1": str(row.level1_admission_status),
                "reason": str(row.level_admission_reason),
                "repair_mode": str(row.level_repair_mode),
                "resolution": str(getattr(row, "level_policy_resolution", "")),
                "original_count": float(row.level_lane_original_count)
                if pd.notna(row.level_lane_original_count)
                else None,
                "bound_basis": str(getattr(row, "level_admission_bound_basis", "")),
                "log_change": (
                    float(row.level_peer_log_change_2023_2024)
                    if pd.notna(getattr(row, "level_peer_log_change_2023_2024", None))
                    else None
                ),
                "bound_lower": (
                    float(getattr(row, "level_admission_bound_lower"))
                    if pd.notna(getattr(row, "level_admission_bound_lower", None))
                    else None
                ),
                "joint_breaches": (
                    int(getattr(row, "level_joint_vector_breach_count"))
                    if pd.notna(getattr(row, "level_joint_vector_breach_count", None))
                    else None
                ),
            }
            for row in admitted.itertuples(index=False)
        }
    if not mass.empty:
        record["control_mass_by_offense"] = (
            mass.groupby("offense")["final_control_mass"].sum().round(2).to_dict()
        )
        record["control_mass_total"] = float(mass["final_control_mass"].sum())
        record["repair_modes"] = sorted(set(mass["level_repair_mode"].dropna().astype(str)))
    return record


def remainder_case(
    *, name: str, jurisdiction_id: str, smoothed: pd.DataFrame, accounting: pd.DataFrame
) -> dict[str, object]:
    row = smoothed[smoothed["jurisdiction_id"].eq(jurisdiction_id)]
    acc = accounting[accounting["jurisdiction_id"].eq(jurisdiction_id)]
    record: dict[str, object] = {"case": name, "jurisdiction_id": jurisdiction_id}
    if row.empty:
        return record
    keep = [
        "offense", "bucket_population", "accounting_count", "smoothed_count", "estimator",
        "reporting_coverage_weight", "remainder_to_municipal_rate_ratio",
        "coverage_peer_predicted_count", "coverage_evidence_share",
        "coverage_evidence_capped_count", "coverage_rate_band_ceiling_count",
        "clean_year_count",
    ]
    record["lanes"] = (
        row[[c for c in keep if c in row.columns]].round(3).to_dict(orient="records")
    )
    if not acc.empty:
        for column in ("crosswalk_agency_count", "contributing_agency_count", "estimating_agency_count"):
            if column in acc.columns:
                record[column] = int(pd.to_numeric(acc[column], errors="coerce").max())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump the traced level-lane cases.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = get_paths()
    year = int(args.year)

    observations = _read(
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        columns=["ori9", "year", "offense", "count", "source", "raw_data_source", "months_reported"],
    )
    # One row per agency-year-offense: the preferred lane wins, as source selection
    # would pick it, so the filed number shown is the one the lane actually saw.
    admission = _read(paths.state_dir / "controls" / f"level_lane_admission_{year}.parquet")
    ledger = _read(paths.state_dir / "controls" / f"level_lane_mass_ledger_{year}.parquet")
    smoothed = _read(paths.state_dir / "controls" / f"jurisdiction_controls_smoothed_{year}.parquet")
    accounting = _read(paths.state_dir / "controls" / f"jurisdiction_controls_{year}.parquet")

    report = {
        "year": year,
        "agency_cases": [
            agency_case(
                name=name,
                ori9=ori9,
                observations=observations,
                admission=admission,
                ledger=ledger,
                year=year,
            )
            for name, ori9 in AGENCY_CASES.items()
        ],
        "remainder_cases": [
            remainder_case(
                name=name,
                jurisdiction_id=jurisdiction_id,
                smoothed=smoothed,
                accounting=accounting,
            )
            for name, jurisdiction_id in REMAINDER_CASES.items()
        ],
    }
    text = json.dumps(report, indent=2, default=str)
    if args.out is not None:
        out = paths.repo_root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
