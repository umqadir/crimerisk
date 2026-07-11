"""Verify count-derived published rate/index invariants on release outputs.

The published estimator is intentionally simple:

  rate_o_primary = 100000 * expected_count_o / primary_denominator_o
  index_o_primary = 100 * rate_o_primary / primary_national_rate_o

where national_rate_o is computed on the same publishable rows as the surface:
100000 * sum(expected_count_o) / sum(primary_denominator_o). Resident fields use
resident_secondary_denominator and publish as rate_o_resident/index_o_resident.
EB outputs are diagnostic-only and must not drive the published columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONAL_OFFENSES = ["murder", "rape", "robbery", "aggravated_assault"]
PROPERTY_OFFENSES = ["burglary", "larceny", "motor_vehicle_theft"]
OFFENSES_7 = PERSONAL_OFFENSES + PROPERTY_OFFENSES
STATE_OUTPUT = REPO_ROOT / "state" / "output"
RATE_PER_100K = 100000.0
TOLERANCE = 1e-9


def _check(label: str, ok: bool, detail: str, failures: list[str]) -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def _count_derived_rate_index(
    *,
    counts: pd.Series,
    denominator: pd.Series,
    publishable: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    count = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    denom = pd.to_numeric(denominator, errors="coerce").fillna(0.0).clip(lower=0.0)
    pub = pd.Series(publishable, index=count.index).fillna(False).astype(bool) & denom.gt(0.0)
    denom_sum = float(denom.loc[pub].sum())
    count_sum = float(count.loc[pub].sum())
    national_rate = RATE_PER_100K * count_sum / denom_sum if denom_sum > 0 else float("nan")
    rate = pd.Series(np.nan, index=count.index, dtype=float)
    rate.loc[pub] = RATE_PER_100K * count.loc[pub] / denom.loc[pub]
    index = pd.Series(np.nan, index=count.index, dtype=float)
    if np.isfinite(national_rate) and national_rate > 0:
        index.loc[pub] = 100.0 * rate.loc[pub] / national_rate
    return rate, index


def _max_abs_pair_delta(actual: pd.Series, expected: pd.Series) -> tuple[float, int]:
    actual_num = pd.to_numeric(actual, errors="coerce")
    expected_num = pd.to_numeric(expected, errors="coerce")
    null_mismatch = actual_num.isna() ^ expected_num.isna()
    comparable = ~(actual_num.isna() | expected_num.isna())
    max_abs = float((actual_num.loc[comparable] - expected_num.loc[comparable]).abs().max()) if comparable.any() else 0.0
    return max_abs, int(null_mismatch.sum())


def verify_surface(path: Path, *, failures: list[str]) -> None:
    print(f"\n=== {path.name} ===")
    df = pd.read_parquet(path)

    offense_sum = sum(pd.to_numeric(df[f"expected_count_{o}"], errors="coerce").fillna(0.0) for o in OFFENSES_7)
    total = pd.to_numeric(df["expected_count_total"], errors="coerce").fillna(0.0)
    personal = pd.to_numeric(df["expected_count_personal"], errors="coerce").fillna(0.0)
    property_ = pd.to_numeric(df["expected_count_property"], errors="coerce").fillna(0.0)
    _check(
        "expected_count_total == sum(offenses)",
        float((offense_sum - total).abs().max()) < 1e-6,
        f"max abs delta {float((offense_sum - total).abs().max()):.2e}",
        failures,
    )
    _check(
        "expected_count_total == personal+property",
        float((personal + property_ - total).abs().max()) < 1e-6,
        f"max abs delta {float((personal + property_ - total).abs().max()):.2e}",
        failures,
    )

    for offense in OFFENSES_7:
        count = pd.to_numeric(df[f"expected_count_{offense}"], errors="coerce").fillna(0.0).clip(lower=0.0)
        denom = pd.to_numeric(df[f"primary_denominator_{offense}"], errors="coerce").fillna(0.0).clip(lower=0.0)
        publishable = df[f"primary_index_publishable_{offense}"].fillna(False).astype(bool)
        expected_rate, expected_index = _count_derived_rate_index(
            counts=count,
            denominator=denom,
            publishable=publishable,
        )
        rate_delta, rate_nulls = _max_abs_pair_delta(df[f"rate_{offense}_primary"], expected_rate)
        index_delta, index_nulls = _max_abs_pair_delta(df[f"index_{offense}_primary"], expected_index)
        raw_delta, raw_nulls = _max_abs_pair_delta(df[f"raw_rate_{offense}"], expected_rate)
        _check(
            f"{offense} primary rate/index count-derived",
            rate_delta <= TOLERANCE and index_delta <= TOLERANCE and rate_nulls == 0 and index_nulls == 0,
            f"rate max {rate_delta:.2e} nulls {rate_nulls}; index max {index_delta:.2e} nulls {index_nulls}",
            failures,
        )
        _check(
            f"{offense} raw_rate count formula",
            raw_delta <= TOLERANCE and raw_nulls == 0,
            f"max {raw_delta:.2e} nulls {raw_nulls}",
            failures,
        )

        resident_denominator = pd.to_numeric(df["resident_secondary_denominator"], errors="coerce").fillna(0.0).clip(lower=0.0)
        resident_publishable = df[f"index_{offense}_resident_publishable"].fillna(False).astype(bool)
        expected_resident_rate, expected_resident_index = _count_derived_rate_index(
            counts=count,
            denominator=resident_denominator,
            publishable=resident_publishable,
        )
        resident_rate_delta, resident_rate_nulls = _max_abs_pair_delta(df[f"rate_{offense}_resident"], expected_resident_rate)
        resident_index_delta, resident_index_nulls = _max_abs_pair_delta(df[f"index_{offense}_resident"], expected_resident_index)
        _check(
            f"{offense} resident rate/index count-derived",
            resident_rate_delta <= TOLERANCE
            and resident_index_delta <= TOLERANCE
            and resident_rate_nulls == 0
            and resident_index_nulls == 0,
            (
                f"rate max {resident_rate_delta:.2e} nulls {resident_rate_nulls}; "
                f"index max {resident_index_delta:.2e} nulls {resident_index_nulls}"
            ),
            failures,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-output-dir", type=Path, default=STATE_OUTPUT)
    args = parser.parse_args()

    surfaces = [
        args.state_output_dir / "crimerisk_block_group_2024_ags_core.parquet",
        args.state_output_dir / "crimerisk_tract_2024_ags_core.parquet",
        args.state_output_dir / "crimerisk_block_group_2024_fbi_calibrated.parquet",
        args.state_output_dir / "crimerisk_tract_2024_fbi_calibrated.parquet",
    ]
    failures: list[str] = []
    for surface in surfaces:
        if not surface.exists():
            failures.append(f"missing surface: {surface}")
            continue
        verify_surface(surface, failures=failures)

    print()
    if failures:
        print(f"ESTIMATOR VERIFICATION FAILED: {len(failures)} invariant(s) violated")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("ESTIMATOR VERIFICATION PASSED: all count-derived invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
