"""Shared loader for the Stage 5 (counts -> published rates/indices) audit screens.

Audit-only. Reads the PROMOTED release surface in state/output/ and writes nothing
outside state/qa/stage5_screen/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/uzairqadir/Projects/data-projects/national/crimerisk-clone")
BG_PATH = REPO / "state/output/crimerisk_block_group_2024_ags_core.parquet"
TRACT_PATH = REPO / "state/output/crimerisk_tract_2024_ags_core.parquet"
OUT = REPO / "state/qa/stage5_screen"

OFFENSES_7 = (
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
)
RARE = ("murder", "rape")
VOLUME = tuple(o for o in OFFENSES_7 if o not in RARE)

# src/crimerisk/allocation.py constants, mirrored for the audit (verified against source).
NON_RESIDENTIAL_HOUSEHOLD_FLOOR = 10.0
PERSON_EXPOSURE_DENOMINATOR_FLOOR = 50.0
PERSON_EXPOSURE_FLOOR_OFFENSES = frozenset(
    ("murder", "rape", "robbery", "aggravated_assault", "larceny")
)
MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR = 50.0
BURGLARY_PREMISES_DENOMINATOR_FLOOR = 10.0
SPECIAL_USE_TRACT_PREFIX = "98"
TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO = 5.0
TRANSIENT_EXPOSURE_INDEX_THRESHOLD = 1000.0
PRIMARY_DENOMINATOR_BY_OFFENSE = {
    "murder": "exposure",
    "rape": "exposure",
    "robbery": "exposure",
    "aggravated_assault": "exposure",
    "burglary": "premises",
    "larceny": "exposure",
    "motor_vehicle_theft": "vehicles",
}

# frontend/public/index.html — ONE fixed, value-anchored, log-symmetric-about-100 break set,
# shared by every index layer. This is the "declared band count" the sol design refers to.
INDEX_BREAKS = [12.5, 25.0, 50.0, 75.0, 100.0, 133.0, 200.0, 400.0, 800.0]

BASE_COLS = [
    "block_group_geoid",
    "state_fips",
    "tract_id",
    "population_2024",
    "daytime_population_jobs_proxy",
    "landscan_day_pop",
    "exposure_proxy_2024",
    "landscan_day_lifted_person_exposure",
    "person_exposure_before_hq_jobs_cap",
    "person_exposure_hq_jobs_cap",
    "person_exposure_hq_jobs_cap_candidate",
    "person_exposure_hq_jobs_capped",
    "households_total",
    "destination_poi_total",
    "lodes_retail_jobs",
    "lodes_industrial_jobs",
    "burglary_premises_total",
    "aggregate_vehicles_total",
    "mvt_commuter_vehicle_proxy",
    "vehicle_exposure_2024",
    "land_area_sq_mi",
    "non_residential_flag",
    "special_use_tract_flag",
    "resident_secondary_denominator",
    "population_zero_with_positive_count",
    "expected_count_total",
    "expected_count_personal",
    "expected_count_property",
    "crime_density_total",
    "index_total_primary_event_weighted",
    "index_total_equal_offense",
    "index_total_harm",
    "index_total_part1_resident",
    "index_personal_part1_resident",
    "index_property_part1_resident",
    "urban_stratum",
]

PER_OFFENSE_COLS = [
    "expected_count_{o}",
    "primary_denominator_{o}",
    "primary_denominator_type_{o}",
    "primary_national_rate_per_100k_{o}",
    "primary_index_publishable_{o}",
    "primary_index_suppressed_{o}",
    "primary_zero_denominator_positive_count_{o}",
    "estimate_mode_{o}",
    "denominator_reason_{o}",
    "rate_{o}_primary",
    "index_{o}_primary",
    "index_{o}_primary_ci95_lower",
    "index_{o}_primary_ci95_upper",
    "index_{o}_primary_ci95_width",
    "index_{o}_primary_ci95_width_ratio",
    "rate_{o}_primary_ci95_lower",
    "rate_{o}_primary_ci95_upper",
    "reliability_tier_{o}",
    "recommended_display_geography_{o}",
    "effective_numerator_support_{o}",
    "direct_incident_support_flag_{o}",
    "direct_incident_support_count_{o}",
    "direct_incident_support_years_{o}",
    "numerator_support_source_{o}",
    "transient_exposure_likely_{o}",
    "index_{o}_resident",
    "index_{o}_resident_suppressed",
    "resident_denominator_reason_{o}",
    "benchmark_imputed_share_{o}",
    "source_mode_{o}",
    "confidence_tier_{o}",
    "crime_density_{o}",
]


def offense_cols(offenses=OFFENSES_7) -> list[str]:
    return [tpl.format(o=o) for o in offenses for tpl in PER_OFFENSE_COLS]


def load_bg(extra: list[str] | None = None) -> pd.DataFrame:
    cols = BASE_COLS + offense_cols() + list(extra or [])
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(BG_PATH).schema_arrow.names)
    cols = [c for c in dict.fromkeys(cols) if c in have]
    df = pd.read_parquet(BG_PATH, columns=cols)
    df["block_group_geoid"] = df["block_group_geoid"].astype("string").str.zfill(12)
    df["tract_id"] = df["tract_id"].astype("string").str.zfill(11)
    df["state_fips"] = df["state_fips"].astype("string").str.zfill(2)
    return df


def load_tract(extra: list[str] | None = None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    cols = [c for c in BASE_COLS if c != "block_group_geoid"] + offense_cols() + list(extra or [])
    have = set(pq.ParquetFile(TRACT_PATH).schema_arrow.names)
    cols = [c for c in dict.fromkeys(["tract_id"] + cols) if c in have]
    df = pd.read_parquet(TRACT_PATH, columns=cols)
    df["tract_id"] = df["tract_id"].astype("string").str.zfill(11)
    df["state_fips"] = df["state_fips"].astype("string").str.zfill(2)
    return df


def bands_spanned(lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Number of fixed display-break boundaries strictly inside [lower, upper].

    0 => the whole interval paints one colour band; N => the interval crosses N break
    boundaries, i.e. spans N+1 display bands.
    """
    lo = pd.to_numeric(lower, errors="coerce")
    hi = pd.to_numeric(upper, errors="coerce")
    crossed = np.zeros(len(lo), dtype=float)
    valid = lo.notna() & hi.notna()
    v = valid.to_numpy()
    for b in INDEX_BREAKS:
        crossed[v] += ((lo.to_numpy()[v] < b) & (hi.to_numpy()[v] > b)).astype(float)
    out = pd.Series(crossed, index=lo.index, dtype=float)
    out[~valid] = np.nan
    return out


def implied_cv(width: pd.Series, point: pd.Series) -> pd.Series:
    """CI-implied coefficient of variation: (upper-lower)/(2*1.96*point).

    The published intervals are exact Poisson (chi-square) count intervals scaled by a
    fixed denominator, so the normal-equivalent CV is the honest read-back of the
    posterior spread from published fields alone.
    """
    w = pd.to_numeric(width, errors="coerce")
    p = pd.to_numeric(point, errors="coerce")
    out = pd.Series(np.nan, index=w.index, dtype=float)
    ok = w.notna() & p.notna() & p.gt(0.0)
    out.loc[ok] = w.loc[ok] / (2.0 * 1.959963984540054 * p.loc[ok])
    return out


def write(df: pd.DataFrame, name: str, *, index: bool = False) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if name.endswith(".csv"):
        df.to_csv(path, index=index)
    else:
        df.to_parquet(path, index=index)
    print(f"wrote {path.relative_to(REPO)}  rows={len(df):,}  size={path.stat().st_size/1e6:.2f} MB")
    return path
