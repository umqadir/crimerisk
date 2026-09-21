"""Surface 3 — the provisional nowcast: a state-level FBI monthly factor over the annual surfaces.

The product carries three surfaces over one spine (`PLAN.md`, target architecture). Surfaces 1
and 2 are annual and are built elsewhere; this module builds the third:

* Surface 1 (`controls.py`) is estimated *reported* offences for one closed calendar year under
  exact conservation. Frozen once published.
* Surface 2 (`smoothed_controls.py`) is the level a jurisdiction-offence carries into a 12-month
  risk window, borrowed across the jurisdiction-year panel.
* Surface 3, here, is what the FBI's *preliminary monthly* series says has happened since the
  closed year ended. It is a single multiplicative factor per (state, offence), applied to
  Surface 2. It is provisional: the FBI revises these monthly counts continuously as agencies
  submit and resubmit, so a Surface-3 number is a statement about a data vintage, never a
  finding, and it never supersedes or "corrects" an annual edition.

WHAT THIS SURFACE CAN AND CANNOT DO -- the honest statement, kept here rather than in a README so
that the code and the claim cannot drift apart:

  The input is a STATE-level series. A state-level series carries no within-state information
  whatsoever. It therefore cannot re-rank block groups inside a jurisdiction, cannot move one
  neighbourhood relative to another, and cannot re-allocate anything. The published block-group
  artifact of this surface is a SCALAR OVERLAY: every block group in a state is multiplied by the
  same number for a given offence, so the annual surface's within-state geography is carried
  through unchanged and by construction. Anyone reading a provisional block-group value as new
  local evidence is reading something that is not there. No allocation code is invoked by this
  module and none should ever be.

THE FACTOR. For a (state, offence):

    factor_raw = sum(target-year monthly counts) / sum(base-year counts, SAME MONTHS)

Both numerator and denominator come from the SAME CDE preliminary series. That is the point: the
CDE's 2024 monthly counts are not this project's 2024 panel -- they rest on a different agency
participation set, a different summarisation, and a different revision state. Dividing a CDE 2025
by a project 2024 would measure the difference between two data-production processes and call it
crime change. The comparison is like-for-like or it is not made.

Same months, too: if the target year is partial, the base year is truncated to those same months,
so the ratio is never contaminated by seasonality.

THE DAMPING. A state's monthly series covers only the agencies that submitted. `factor_raw` from
a state where 78% of the population is covered is a noisier and more selection-prone estimate of
the whole state's change than one from a state at 99.7%. Coverage is used as a credibility weight
on the DEVIATION FROM NO CHANGE:

    w      = mean(pct_population_coverage over the target-year months used) / 100
    factor = clip(1 + w * (factor_raw - 1), 0.6, 1.5)

so a fully covered state keeps its measured change, a half-covered state moves half as far, and a
state with no coverage would not move at all. This is credibility shrinkage toward the null, not a
guardrail bolted on after seeing the answer: the weight is the coverage the source itself
publishes, and the null it shrinks toward is "no change from the annual surface", which is the
only defensible prior for a jurisdiction the preliminary series barely observes. The clip is a
structural bound on a one-year change in a Part-I count, not a tuned parameter.

`violent-crime` and `property-crime` are pulled and carried as DIAGNOSTICS ONLY. They are CDE
rollups over a slightly different offence set than this project's Part-I seven, and using a rollup
factor for a constituent offence would silently import that definitional difference. Each of the
seven offences gets its own slug's factor and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths


NOWCAST_VERSION = "provisional_nowcast_v1"

# The one sentence every provisional artifact carries. It lives here, as a constant, so the
# edition README, the edition manifest and the table names cannot drift away from each other or
# from the code that produced the numbers.
PROVISIONAL_LABEL = (
    "PROVISIONAL: provisional nowcast from FBI monthly preliminary data; revised continuously; "
    "not comparable to the annual accounting surface."
)

# Every published table name begins with this. A provisional number that escapes its edition
# directory still says what it is.
PROVISIONAL_TABLE_PREFIX = "provisional_"

# The `/summarized/state/{ST}/{slug}` slug that carries each of this project's Part-I seven.
# `murder -> homicide` is the one rename; the CDE has no separate "murder" slug.
CDE_SUMMARIZED_SLUG_BY_OFFENSE: dict[str, str] = {
    "murder": "homicide",
    "rape": "rape",
    "robbery": "robbery",
    "aggravated_assault": "aggravated-assault",
    "burglary": "burglary",
    "larceny": "larceny",
    "motor_vehicle_theft": "motor-vehicle-theft",
}
OFFENSE_BY_CDE_SUMMARIZED_SLUG: dict[str, str] = {
    slug: offense for offense, slug in CDE_SUMMARIZED_SLUG_BY_OFFENSE.items()
}

# Pulled, reported, and never used to build an offence factor. See the module docstring.
CDE_DIAGNOSTIC_SLUGS: tuple[str, ...] = ("violent-crime", "property-crime")

# A Part-I count for a whole state does not halve or grow by half in one year. The clip is a
# structural bound, applied after the damping, and any row it binds is flagged in the output.
FACTOR_CLIP_LOW = 0.6
FACTOR_CLIP_HIGH = 1.5

# Coverage at or below which the damping must be plainly visible in the published factor. Not a
# gate -- nothing is dropped at this line; it is the threshold the validation reports against.
LOW_COVERAGE_PCT = 70.0

# The factor for a (state, offence) whose base-year mass is zero: no ratio exists, so the surface
# says "no change" rather than inventing one.
NO_BASE_MASS_BASIS = "no_base_mass"
CDE_RATIO_BASIS = "cde_ratio"

CDE_REQUIRED_COLUMNS = (
    "state",
    "offense_slug",
    "month",
    "year",
    "count",
    "pct_population_coverage",
    "us_rate",
    "max_data_date",
    "last_refresh_date",
)

FACTOR_COLUMNS = [
    "state_abbr",
    "offense",
    "cde_offense_slug",
    "base_year",
    "target_year",
    "months_used",
    "month_span",
    "base_count",
    "target_count",
    "factor_raw",
    "coverage_pct",
    "credibility_weight",
    "provisional_factor",
    "factor_basis",
    "clipped",
    "low_coverage",
]

DEFAULT_CDE_STATE_MONTHLY_RELPATH = (
    "analysis_scratch/final_phase/fbi_2025/state_month_offense_2025.parquet"
)


@dataclass(frozen=True)
class NowcastConfig:
    """Every knob here is either a property of the source or a structural bound.

    None of them was chosen by looking at the resulting factors.
    """

    base_year: int = 2024
    target_year: int = 2025
    clip_low: float = FACTOR_CLIP_LOW
    clip_high: float = FACTOR_CLIP_HIGH
    low_coverage_pct: float = LOW_COVERAGE_PCT
    # The implied national change may differ from the CDE's own national series -- our weights are
    # municipal Surface-2 mass, theirs is every reporting agency including AK/HI -- but not by
    # much. This is the width of "not by much", as a fraction of the CDE's own delta.
    national_delta_tolerance: float = 0.20
    # Floating-point summation slack for the exactness checks.
    conservation_tolerance: float = 1e-9


def cde_state_monthly_path(paths: RepoPaths) -> Path:
    return paths.repo_root / DEFAULT_CDE_STATE_MONTHLY_RELPATH


# --- source ---------------------------------------------------------------------------------


def read_cde_state_monthly(path: Path) -> pd.DataFrame:
    """The acquired CDE `/summarized/state` panel, validated on the columns this lane depends on."""
    frame = pd.read_parquet(path)
    missing = [column for column in CDE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is not a CDE state-monthly panel; missing columns {missing}")
    frame = frame.copy()
    frame["state"] = frame["state"].astype(str).str.strip().str.upper()
    frame["offense_slug"] = frame["offense_slug"].astype(str).str.strip()
    for column in ("month", "year"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in ("count", "pct_population_coverage", "us_rate"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    duplicated = frame.duplicated(["state", "offense_slug", "year", "month"]).sum()
    if duplicated:
        raise ValueError(
            f"{path} has {duplicated} duplicate (state, offense_slug, year, month) rows; the "
            "factor would double-count them"
        )
    return frame


def cde_vintage(frame: pd.DataFrame, *, path: Path | None = None) -> dict[str, object]:
    """The source's own as-of stamp. A provisional number without this is not interpretable."""

    def _one(column: str) -> str | None:
        values = sorted({str(value) for value in frame[column].dropna().unique()})
        if not values:
            return None
        if len(values) > 1:
            raise ValueError(
                f"the CDE panel carries {len(values)} distinct {column} values ({values}); a "
                "provisional edition must stamp exactly one source vintage"
            )
        return values[0]

    months = (
        frame.groupby("year")["month"]
        .agg(lambda values: sorted({int(value) for value in values}))
        .to_dict()
    )
    return {
        "source": "FBI Crime Data Explorer, /summarized/state/{state}/{offense} (preliminary "
        "monthly)",
        "path": None if path is None else str(path),
        "max_data_date": _one("max_data_date"),
        "last_refresh_date": _one("last_refresh_date"),
        "states": sorted(frame["state"].unique().tolist()),
        "state_count": int(frame["state"].nunique()),
        "offense_slugs": sorted(frame["offense_slug"].unique().tolist()),
        "months_by_year": {int(year): value for year, value in months.items()},
        "rows": int(len(frame)),
    }


# --- the factor -----------------------------------------------------------------------------


def _common_months(frame: pd.DataFrame, *, base_year: int, target_year: int) -> pd.DataFrame:
    """Restrict each (state, slug) to the months observed in BOTH years.

    A partial target year must be compared against the same part of the base year or the ratio is
    reading seasonality as change.
    """
    both = frame[frame["year"].isin((base_year, target_year))]
    keys = ["state", "offense_slug", "month"]
    present = both.groupby(keys, as_index=False)["year"].nunique()
    common = present.loc[present["year"] == 2, keys]
    return both.merge(common, on=keys, how="inner")


def provisional_factors(
    cde: pd.DataFrame, *, config: NowcastConfig | None = None
) -> pd.DataFrame:
    """One coverage-damped 2025/2024 factor per (state, offence), from one source, same months."""
    config = config or NowcastConfig()
    mapped = cde[cde["offense_slug"].isin(OFFENSE_BY_CDE_SUMMARIZED_SLUG)].copy()
    if mapped.empty:
        raise ValueError(
            "the CDE panel carries none of the Part-I slugs "
            f"{sorted(OFFENSE_BY_CDE_SUMMARIZED_SLUG)}"
        )
    aligned = _common_months(
        mapped, base_year=config.base_year, target_year=config.target_year
    )
    if aligned.empty:
        raise ValueError(
            f"no month appears in both {config.base_year} and {config.target_year}; the factor "
            "would compare different parts of the year"
        )
    aligned["offense"] = aligned["offense_slug"].map(OFFENSE_BY_CDE_SUMMARIZED_SLUG)

    keys = ["state", "offense_slug", "offense"]
    counts = (
        aligned.pivot_table(index=keys, columns="year", values="count", aggfunc="sum")
        .rename(columns={config.base_year: "base_count", config.target_year: "target_count"})
        .rename_axis(columns=None)
        .reset_index()
    )
    target_rows = aligned[aligned["year"] == config.target_year]
    coverage = (
        target_rows.groupby(keys)["pct_population_coverage"]
        .mean()
        .rename("coverage_pct")
        .reset_index()
    )
    months = (
        target_rows.groupby(keys)["month"]
        .agg(
            months_used="nunique",
            month_span=lambda values: (
                f"{int(min(values)):02d}-{int(max(values)):02d}" if len(values) else ""
            ),
        )
        .reset_index()
    )
    factors = counts.merge(coverage, on=keys, how="left").merge(months, on=keys, how="left")

    base = factors["base_count"].astype(float)
    target = factors["target_count"].astype(float)
    has_base = base > 0
    factors["factor_raw"] = np.where(has_base, target / base.where(has_base, 1.0), 1.0)
    factors["factor_basis"] = np.where(has_base, CDE_RATIO_BASIS, NO_BASE_MASS_BASIS)

    weight = (factors["coverage_pct"].astype(float) / 100.0).clip(lower=0.0, upper=1.0)
    factors["credibility_weight"] = weight
    damped = 1.0 + weight * (factors["factor_raw"] - 1.0)
    clipped = damped.clip(lower=config.clip_low, upper=config.clip_high)
    factors["provisional_factor"] = clipped
    factors["clipped"] = ~np.isclose(damped, clipped, rtol=0.0, atol=1e-12)
    factors["low_coverage"] = factors["coverage_pct"].astype(float) < config.low_coverage_pct

    factors = factors.rename(columns={"state": "state_abbr", "offense_slug": "cde_offense_slug"})
    factors["base_year"] = int(config.base_year)
    factors["target_year"] = int(config.target_year)
    factors["months_used"] = factors["months_used"].astype(int)
    return factors[FACTOR_COLUMNS].sort_values(["offense", "state_abbr"]).reset_index(drop=True)


def diagnostic_rollup_factors(
    cde: pd.DataFrame, *, config: NowcastConfig | None = None
) -> pd.DataFrame:
    """`violent-crime` / `property-crime` factors. Published as diagnostics; never applied."""
    config = config or NowcastConfig()
    rollups = cde[cde["offense_slug"].isin(CDE_DIAGNOSTIC_SLUGS)].copy()
    if rollups.empty:
        return pd.DataFrame(
            columns=["state_abbr", "cde_offense_slug", "base_count", "target_count", "factor_raw"]
        )
    aligned = _common_months(
        rollups, base_year=config.base_year, target_year=config.target_year
    )
    counts = (
        aligned.pivot_table(
            index=["state", "offense_slug"], columns="year", values="count", aggfunc="sum"
        )
        .rename(columns={config.base_year: "base_count", config.target_year: "target_count"})
        .rename_axis(columns=None)
        .reset_index()
        .rename(columns={"state": "state_abbr", "offense_slug": "cde_offense_slug"})
    )
    counts["factor_raw"] = counts["target_count"] / counts["base_count"].replace(0.0, np.nan)
    counts["applied"] = False
    counts["note"] = "CDE rollup over a different offence set than this project's Part-I seven"
    return counts.sort_values(["cde_offense_slug", "state_abbr"]).reset_index(drop=True)


def factor_distribution(factors: pd.DataFrame) -> list[dict[str, object]]:
    """Per-offence shape of the factor field: what the nowcast is actually claiming."""
    rows: list[dict[str, object]] = []
    for offense, group in factors.groupby("offense", sort=True):
        rows.append(
            {
                "offense": str(offense),
                "states": int(len(group)),
                "raw_p5": float(group["factor_raw"].quantile(0.05)),
                "raw_median": float(group["factor_raw"].median()),
                "raw_p95": float(group["factor_raw"].quantile(0.95)),
                "factor_p5": float(group["provisional_factor"].quantile(0.05)),
                "factor_median": float(group["provisional_factor"].median()),
                "factor_p95": float(group["provisional_factor"].quantile(0.95)),
                "factor_min": float(group["provisional_factor"].min()),
                "factor_max": float(group["provisional_factor"].max()),
                "clipped": int(group["clipped"].sum()),
                "low_coverage_states": int(group["low_coverage"].sum()),
                "no_base_mass": int((group["factor_basis"] == NO_BASE_MASS_BASIS).sum()),
            }
        )
    return rows


def damping_report(factors: pd.DataFrame, *, config: NowcastConfig | None = None) -> dict:
    """Evidence that the coverage weight actually damps, measured on the published factors.

    `shrink_ratio` is |factor - 1| / |factor_raw - 1|: it equals the credibility weight exactly
    wherever the clip did not bind, so a low-coverage row cannot show a shrink ratio near 1.
    """
    config = config or NowcastConfig()
    frame = factors.copy()
    deviation = (frame["factor_raw"] - 1.0).abs()
    moved = deviation > 0
    frame["shrink_ratio"] = np.where(
        moved, (frame["provisional_factor"] - 1.0).abs() / deviation.where(moved, 1.0), np.nan
    )
    low = frame[frame["low_coverage"]]
    return {
        "low_coverage_pct": float(config.low_coverage_pct),
        "min_coverage_pct": float(frame["coverage_pct"].min()),
        "max_coverage_pct": float(frame["coverage_pct"].max()),
        "rows": int(len(frame)),
        "low_coverage_rows": int(len(low)),
        "low_coverage_states": sorted(low["state_abbr"].unique().tolist()),
        "low_coverage_max_shrink_ratio": (
            None if low.empty else float(low["shrink_ratio"].max(skipna=True))
        ),
        "weight_equals_shrink_where_unclipped": bool(
            np.allclose(
                frame.loc[~frame["clipped"] & moved, "shrink_ratio"],
                frame.loc[~frame["clipped"] & moved, "credibility_weight"],
                rtol=1e-9,
                atol=1e-12,
            )
        ),
    }


# --- application ----------------------------------------------------------------------------


def _require_factor_coverage(frame: pd.DataFrame, factors: pd.DataFrame, *, what: str) -> None:
    have = set(map(tuple, factors[["state_abbr", "offense"]].to_numpy()))
    need = set(map(tuple, frame[["state_abbr", "offense"]].drop_duplicates().to_numpy()))
    missing = sorted(need - have)
    if missing:
        raise ValueError(
            f"{len(missing)} (state, offence) pairs in the {what} have no provisional factor "
            f"(e.g. {missing[:5]}); the nowcast refuses to publish a silently unscaled row"
        )


def apply_factors_to_controls(
    controls: pd.DataFrame, factors: pd.DataFrame, *, config: NowcastConfig | None = None
) -> pd.DataFrame:
    """Surface-2 smoothed controls x their state-offence factor. No re-allocation, no raking."""
    config = config or NowcastConfig()
    base = controls[controls["year"].astype(int) == int(config.base_year)].copy()
    if base.empty:
        raise ValueError(
            f"the smoothed controls carry no year {config.base_year}; the nowcast has nothing to "
            "scale"
        )
    base["state_abbr"] = base["state_abbr"].astype(str)
    _require_factor_coverage(base, factors, what="smoothed controls")

    merged = base.merge(
        factors[
            [
                "state_abbr",
                "offense",
                "provisional_factor",
                "factor_raw",
                "coverage_pct",
                "credibility_weight",
                "factor_basis",
                "clipped",
                "low_coverage",
                "months_used",
                "month_span",
            ]
        ],
        on=["state_abbr", "offense"],
        how="left",
        validate="many_to_one",
    )
    if merged["provisional_factor"].isna().any():
        raise ValueError("factor merge produced nulls; the join key is not what it claims to be")

    merged = merged.rename(
        columns={
            "smoothed_count": "annual_smoothed_count",
            "clipped": "factor_clipped",
            "low_coverage": "factor_low_coverage",
        }
    )
    merged["provisional_count"] = (
        merged["annual_smoothed_count"].astype(float) * merged["provisional_factor"]
    )
    merged["base_year"] = int(config.base_year)
    merged["target_year"] = int(config.target_year)
    merged["is_provisional"] = True
    columns = [
        "jurisdiction_id",
        "jurisdiction_type",
        "jurisdiction_name",
        "state_fips",
        "state_abbr",
        "geo_type",
        "geoid",
        "offense",
        "base_year",
        "target_year",
        "bucket_population",
        "annual_smoothed_count",
        "provisional_factor",
        "provisional_count",
        "factor_raw",
        "coverage_pct",
        "credibility_weight",
        "factor_basis",
        "factor_clipped",
        "factor_low_coverage",
        "months_used",
        "month_span",
        "is_provisional",
    ]
    return merged[columns].sort_values(["state_abbr", "jurisdiction_id", "offense"]).reset_index(
        drop=True
    )


def state_provisional_totals(jurisdiction_provisional: pd.DataFrame) -> pd.DataFrame:
    """The state-level provisional table: the support the factor was actually measured at."""
    grouped = (
        jurisdiction_provisional.groupby(["state_abbr", "offense"], sort=True)
        .agg(
            jurisdictions=("jurisdiction_id", "nunique"),
            annual_smoothed_count=("annual_smoothed_count", "sum"),
            provisional_count=("provisional_count", "sum"),
            provisional_factor=("provisional_factor", "first"),
            factor_raw=("factor_raw", "first"),
            coverage_pct=("coverage_pct", "first"),
            credibility_weight=("credibility_weight", "first"),
            factor_basis=("factor_basis", "first"),
            factor_clipped=("factor_clipped", "first"),
            factor_low_coverage=("factor_low_coverage", "first"),
            months_used=("months_used", "first"),
            month_span=("month_span", "first"),
            base_year=("base_year", "first"),
            target_year=("target_year", "first"),
        )
        .reset_index()
    )
    grouped["is_provisional"] = True
    return grouped


def block_group_overlay_column(offense: str) -> str:
    return f"provisional_expected_count_{offense}"


def block_group_factor_column(offense: str) -> str:
    return f"provisional_factor_{offense}"


def annual_expected_count_column(offense: str) -> str:
    return f"expected_count_{offense}"


def block_group_overlay_columns(offenses: tuple[str, ...] = OFFENSES_7) -> list[str]:
    columns = ["block_group_geoid", "state_fips", "state_abbr"]
    for offense in offenses:
        columns.extend(
            [
                annual_expected_count_column(offense),
                block_group_factor_column(offense),
                block_group_overlay_column(offense),
            ]
        )
    return columns


def empty_block_group_overlay(offenses: tuple[str, ...] = OFFENSES_7) -> pd.DataFrame:
    """The overlay's shape with no rows, for a build that skipped it."""
    return pd.DataFrame(
        {column: pd.Series(dtype="float64") for column in block_group_overlay_columns(offenses)}
    )


def apply_factors_to_block_groups(
    surface: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    state_abbr_by_fips: dict[str, str],
    offenses: tuple[str, ...] = OFFENSES_7,
    config: NowcastConfig | None = None,
) -> pd.DataFrame:
    """A SCALAR overlay: annual block-group expected count x its state's offence factor.

    Every block group in a state moves by the same number. That is not a limitation this function
    works around -- it is the only thing a state-level series licenses.
    """
    config = config or NowcastConfig()
    frame = surface.copy()
    frame["state_fips"] = frame["state_fips"].astype(str).str.zfill(2)
    frame["state_abbr"] = frame["state_fips"].map(state_abbr_by_fips)
    unmapped = sorted(frame.loc[frame["state_abbr"].isna(), "state_fips"].unique().tolist())
    if unmapped:
        raise ValueError(
            f"state FIPS {unmapped} have no abbreviation; the overlay cannot find their factor"
        )

    wide = factors.pivot(index="state_abbr", columns="offense", values="provisional_factor")
    missing_states = sorted(set(frame["state_abbr"]) - set(wide.index))
    if missing_states:
        raise ValueError(
            f"states {missing_states} appear in the block-group surface but carry no provisional "
            "factor; the nowcast refuses to publish a silently unscaled block group"
        )
    missing_offenses = [offense for offense in offenses if offense not in wide.columns]
    if missing_offenses:
        raise ValueError(f"no provisional factor for offences {missing_offenses}")

    for offense in offenses:
        annual_column = annual_expected_count_column(offense)
        if annual_column not in frame.columns:
            raise ValueError(f"the block-group surface has no {annual_column}")
        factor = frame["state_abbr"].map(wide[offense])
        frame[block_group_factor_column(offense)] = factor.astype(float)
        frame[block_group_overlay_column(offense)] = (
            frame[annual_column].astype(float) * factor.astype(float)
        )
    frame["base_year"] = int(config.base_year)
    frame["target_year"] = int(config.target_year)
    frame["is_provisional"] = True
    return frame[
        block_group_overlay_columns(offenses) + ["base_year", "target_year", "is_provisional"]
    ]


# --- validation -----------------------------------------------------------------------------


def cde_reference_ratios(
    cde: pd.DataFrame, *, config: NowcastConfig | None = None
) -> dict[str, dict[str, float]]:
    """Two national change references from the same source, over the same months.

    * `national` -- the CDE's own national series. `us_rate` is the national rate the API returns
      alongside every state response, so this is the source's own national statement rather than a
      re-aggregation. It covers all 50 states + DC and every reporting agency in them.
    * `scope_matched` -- the same source aggregated over exactly the states this project publishes
      (48 contiguous + DC). It is the `national` series minus the two states outside the coverage
      universe.

    Both are reported. The gate uses the scope-matched one, because a product that does not
    publish Alaska should not be scored on whether it reproduces Alaska.
    """
    config = config or NowcastConfig()
    mapped = cde[cde["offense_slug"].isin(OFFENSE_BY_CDE_SUMMARIZED_SLUG)].copy()
    aligned = _common_months(mapped, base_year=config.base_year, target_year=config.target_year)
    aligned["offense"] = aligned["offense_slug"].map(OFFENSE_BY_CDE_SUMMARIZED_SLUG)

    national = aligned.drop_duplicates(["offense", "year", "month"])
    national_totals = national.pivot_table(
        index="offense", columns="year", values="us_rate", aggfunc="sum"
    )
    scope_totals = aligned.pivot_table(
        index="offense", columns="year", values="count", aggfunc="sum"
    )
    return {
        str(offense): {
            "national": float(
                national_totals.loc[offense, config.target_year]
                / national_totals.loc[offense, config.base_year]
            ),
            "scope_matched": float(
                scope_totals.loc[offense, config.target_year]
                / scope_totals.loc[offense, config.base_year]
            ),
        }
        for offense in national_totals.index
    }


def national_comparability(
    state_provisional: pd.DataFrame,
    references: dict[str, dict[str, float]],
    *,
    config: NowcastConfig | None = None,
) -> dict[str, object]:
    """Does the nowcast's implied national change match the source's own?

    The two cannot be identical, and three of the differences are known, intended and quantified
    rather than defects:

    1. **Scope.** We publish 48 contiguous states + DC; the CDE's national series is all 50 + DC.
       Controlled for, by gating against the scope-matched aggregate of the same source.
    2. **Damping.** The published factors are deliberately shrunk toward no change by coverage, so
       the published implied delta is smaller in magnitude than the source's by construction.
       Charging that shrinkage against the tolerance would be a check that penalises the surface
       for doing exactly what it was designed to do. The gate therefore runs on the UNDAMPED
       implied delta and the damped one is reported beside it.
    3. **Weights.** Ours is Surface-2 municipal-footprint mass; the source's is every reporting
       agency, municipal or not. This one is NOT controlled for -- it is the thing the check
       exists to catch. A factor field joined to the wrong key, or applied to the wrong offence,
       shows up here as a national number that does not track the source's.

    `gap_vs_national_pct` reports the fully uncontrolled comparison (published damped delta vs the
    CDE's own national series) so the harder number is on the record too.
    """
    config = config or NowcastConfig()
    rows: list[dict[str, object]] = []
    for offense, group in state_provisional.groupby("offense", sort=True):
        annual = float(group["annual_smoothed_count"].sum())
        provisional = float(group["provisional_count"].sum())
        implied = provisional / annual if annual else float("nan")
        implied_raw = (
            float((group["annual_smoothed_count"] * group["factor_raw"]).sum()) / annual
            if annual
            else float("nan")
        )
        reference = references[str(offense)]
        scope_delta = reference["scope_matched"] - 1.0
        national_delta = reference["national"] - 1.0
        gap = abs((implied_raw - 1.0) - scope_delta)
        allowed = config.national_delta_tolerance * abs(scope_delta)
        rows.append(
            {
                "offense": str(offense),
                "annual_smoothed_count": annual,
                "provisional_count": provisional,
                "implied_ratio": implied,
                "implied_delta_pct": 100.0 * (implied - 1.0),
                "implied_undamped_delta_pct": 100.0 * (implied_raw - 1.0),
                "cde_scope_matched_ratio": reference["scope_matched"],
                "cde_scope_matched_delta_pct": 100.0 * scope_delta,
                "cde_national_ratio": reference["national"],
                "cde_national_delta_pct": 100.0 * national_delta,
                "abs_delta_gap": gap,
                "allowed_gap": allowed,
                "gap_pct_of_reference": (
                    100.0 * gap / abs(scope_delta) if scope_delta else float("nan")
                ),
                "gap_vs_national_pct": (
                    100.0 * abs((implied - 1.0) - national_delta) / abs(national_delta)
                    if national_delta
                    else float("nan")
                ),
                "within_tolerance": bool(gap <= allowed),
            }
        )
    return {
        "tolerance_fraction_of_cde_delta": float(config.national_delta_tolerance),
        "reference": "scope_matched",
        "gated_quantity": "undamped implied national delta",
        "offenses": rows,
        "ok": all(bool(row["within_tolerance"]) for row in rows),
    }


def conservation_report(
    *,
    block_group_overlay: pd.DataFrame,
    state_provisional: pd.DataFrame,
    offenses: tuple[str, ...] = OFFENSES_7,
    config: NowcastConfig | None = None,
) -> dict[str, object]:
    """Conservation for a surface built by multiplication.

    Two things must hold, and they are different claims:

    1. **The factor is applied exactly, at both supports.** For every (state, offence), the ratio
       of the published provisional total to the annual total it came from equals the published
       factor -- at block-group support and at jurisdiction support alike. A published factor that
       does not reproduce the published number is a lie in the manifest.

    2. **The nowcast introduces no new footprint mismatch.** The block-group surface covers every
       block group in the country; the jurisdiction controls cover municipal footprints only, so
       their state totals differ in the ANNUAL surfaces already. Because both sides are scaled by
       the same factor, that ratio must be unchanged by the provisional step. Anything else means
       the two artifacts were scaled by different numbers and would disagree about the same place.
    """
    config = config or NowcastConfig()
    frames: list[pd.DataFrame] = []
    for offense in offenses:
        grouped = (
            block_group_overlay.groupby("state_abbr", sort=True)
            .agg(
                bg_annual=(annual_expected_count_column(offense), "sum"),
                bg_provisional=(block_group_overlay_column(offense), "sum"),
                bg_factor=(block_group_factor_column(offense), "first"),
            )
            .reset_index()
        )
        grouped["offense"] = offense
        frames.append(grouped)
    bg = pd.concat(frames, ignore_index=True)

    merged = bg.merge(
        state_provisional[
            ["state_abbr", "offense", "annual_smoothed_count", "provisional_count",
             "provisional_factor"]
        ],
        on=["state_abbr", "offense"],
        how="inner",
        validate="one_to_one",
    )
    merged["bg_applied_factor"] = merged["bg_provisional"] / merged["bg_annual"].replace(0.0, np.nan)
    merged["jurisdiction_applied_factor"] = merged["provisional_count"] / merged[
        "annual_smoothed_count"
    ].replace(0.0, np.nan)
    merged["annual_alignment"] = merged["bg_annual"] / merged["annual_smoothed_count"].replace(
        0.0, np.nan
    )
    merged["provisional_alignment"] = merged["bg_provisional"] / merged[
        "provisional_count"
    ].replace(0.0, np.nan)

    bg_error = (merged["bg_applied_factor"] - merged["provisional_factor"]).abs()
    jurisdiction_error = (
        merged["jurisdiction_applied_factor"] - merged["provisional_factor"]
    ).abs()
    alignment_error = (merged["provisional_alignment"] - merged["annual_alignment"]).abs() / merged[
        "annual_alignment"
    ].abs()

    max_bg = float(np.nanmax(bg_error.to_numpy())) if len(merged) else 0.0
    max_jurisdiction = float(np.nanmax(jurisdiction_error.to_numpy())) if len(merged) else 0.0
    max_alignment = float(np.nanmax(alignment_error.to_numpy())) if len(merged) else 0.0
    tolerance = float(config.conservation_tolerance)
    return {
        "tolerance": tolerance,
        "cells": int(len(merged)),
        "max_block_group_factor_error": max_bg,
        "max_jurisdiction_factor_error": max_jurisdiction,
        "max_alignment_drift": max_alignment,
        "block_group_factor_exact": bool(max_bg <= tolerance),
        "jurisdiction_factor_exact": bool(max_jurisdiction <= tolerance),
        "alignment_unchanged": bool(max_alignment <= 1e-9),
        "conserved": bool(
            max_bg <= tolerance and max_jurisdiction <= tolerance and max_alignment <= 1e-9
        ),
        "footprint_note": (
            "Block-group and jurisdiction state totals differ in the ANNUAL surfaces already -- "
            "the block-group surface covers every block group, the municipal controls cover "
            "municipal footprints only. The provisional step multiplies both by the same factor, "
            "so it changes that ratio by nothing."
        ),
    }


def validate(
    *,
    factors: pd.DataFrame,
    state_provisional: pd.DataFrame,
    block_group_overlay: pd.DataFrame,
    references: dict[str, dict[str, float]],
    config: NowcastConfig | None = None,
) -> dict[str, object]:
    config = config or NowcastConfig()
    distribution = factor_distribution(factors)
    damping = damping_report(factors, config=config)
    comparability = national_comparability(state_provisional, references, config=config)
    conservation = conservation_report(
        block_group_overlay=block_group_overlay,
        state_provisional=state_provisional,
        config=config,
    )
    issues: list[str] = []
    if not comparability["ok"]:
        offending = [
            row["offense"] for row in comparability["offenses"] if not row["within_tolerance"]
        ]
        issues.append(
            f"implied national change is outside {config.national_delta_tolerance:.0%} of the "
            f"scope-matched CDE series for {offending}"
        )
    if not conservation["conserved"]:
        issues.append("provisional conservation failed")
    if not damping["weight_equals_shrink_where_unclipped"]:
        issues.append("coverage damping is not the published credibility weight")
    return {
        "nowcast_version": NOWCAST_VERSION,
        "ok": not issues,
        "issues": issues,
        "factor_distribution": distribution,
        "damping": damping,
        "national_comparability": comparability,
        "conservation": conservation,
    }
