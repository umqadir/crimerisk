from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7


@dataclass(frozen=True)
class ReconciliationReport:
    summary: dict[str, Any]


def build_reconciliation_report(
    *,
    tract_output: pd.DataFrame,
    muni_totals: pd.DataFrame,
    state_constraints: pd.DataFrame,
) -> ReconciliationReport:
    out = tract_output.copy()
    required = {"state_fips", "bucket_id", "bucket_type"} | {f"count_{o}" for o in OFFENSES_7}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"tract_output missing columns: {sorted(missing)}")

    summary: dict[str, Any] = {
        "tract_rows": int(len(out)),
    }

    # 1) Municipal sums == municipal totals (for mapped buckets)
    muni_out = out[out["bucket_type"] == "municipal"].copy()
    muni_out_sums = muni_out.groupby("bucket_id")[[f"count_{o}" for o in OFFENSES_7]].sum()
    muni_target = muni_totals.set_index("bucket_id")[list(OFFENSES_7)].rename(columns={o: f"target_{o}" for o in OFFENSES_7})

    muni_comp = muni_out_sums.join(muni_target, how="left")
    muni_errors = {}
    for o in OFFENSES_7:
        diff = muni_comp[f"count_{o}"] - muni_comp[f"target_{o}"]
        muni_errors[o] = {
            "max_abs": float(diff.abs().max()) if len(diff) else 0.0,
            "mean_abs": float(diff.abs().mean()) if len(diff) else 0.0,
        }
    summary["municipal_bucket_count"] = int(muni_out_sums.shape[0])
    summary["municipal_errors"] = muni_errors

    # 2) State sums == constraints
    state_out_sums = out.groupby("state_fips")[[f"count_{o}" for o in OFFENSES_7]].sum()
    state_con = state_constraints.set_index("state_fips")[list(OFFENSES_7)].astype(float)
    state_errors = {}
    for o in OFFENSES_7:
        diff = state_out_sums[f"count_{o}"] - state_con[o]
        state_errors[o] = {
            "max_abs": float(diff.abs().max()) if len(diff) else 0.0,
            "mean_abs": float(diff.abs().mean()) if len(diff) else 0.0,
        }
    summary["state_errors"] = state_errors

    # 3) National totals
    nat_out = state_out_sums.sum()
    nat_con = state_con.sum()
    nat_errors = {o: float(nat_out[f"count_{o}"] - nat_con[o]) for o in OFFENSES_7}
    summary["national_errors"] = nat_errors

    # 4) Lineage coverage
    lineage_fields = ["allocation_method", "constraint_type", "constraint_source_version"]
    present = [f for f in lineage_fields if f in out.columns]
    summary["lineage_fields_present"] = present
    summary["lineage_null_counts"] = {f: int(out[f].isna().sum()) for f in present}

    return ReconciliationReport(summary=summary)

