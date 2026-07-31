from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crimerisk.agency_identity import (
    build_ori_succession_ledger,  # noqa: F401  re-exported: tests and callers import it from here
    load_agency_jurisdiction_crosswalk,
    resolve_ori_succession,
)
from crimerisk.crime.municipal_totals import _project_count_log_linear
from crimerisk.scope import PRODUCTION_SCOPE_EXCLUDE
from crimerisk.source_provenance import NIBRS_SOURCE, SUMMARY_SOURCE
from crimerisk.source_selection import build_agency_preferred_observations
from crimerisk.stage1_adjudications import (
    Stage1AdjudicationError,
    assert_registry_bit,
    build_usability_directives,
)


TREND_FILL_MIN_PANEL_AGENCIES = 10
TREND_FILL_MIN_PANEL_MASS_SHARE = 0.25
# Murder and rape panels are high-variance at low event volume; require enough
# base and target events before accepting a state-specific ratio.
TREND_FILL_RARE_OFFENSES = frozenset(("murder", "rape"))
TREND_FILL_RARE_MIN_PANEL_EVENT_MASS = 30.0


@dataclass(frozen=True)
class TrendFillAudit:
    ratio: float
    source: str
    panel_agency_count: int | None
    panel_mass_share_base: float | None
    panel_mass_share_target: float | None


class TrendFillLookup:
    def __init__(
        self,
        *,
        state_map: dict[tuple[str, str, int], TrendFillAudit],
        national_map: dict[tuple[str, int], TrendFillAudit],
    ) -> None:
        self._state_map = state_map
        self._national_map = national_map

    def resolve(self, *, state_fips: object, offense: object, base_year: object) -> TrendFillAudit:
        year = _parse_year(base_year)
        if year is None:
            return TrendFillAudit(
                ratio=1.0,
                source="unscaled_no_vintage",
                panel_agency_count=None,
                panel_mass_share_base=None,
                panel_mass_share_target=None,
            )
        state_key = (str(state_fips).zfill(2), str(offense), int(year))
        if state_key in self._state_map:
            return self._state_map[state_key]
        national_key = (str(offense), int(year))
        if national_key in self._national_map:
            audit = self._national_map[national_key]
            return TrendFillAudit(
                ratio=audit.ratio,
                source="national_panel_fallback",
                panel_agency_count=audit.panel_agency_count,
                panel_mass_share_base=audit.panel_mass_share_base,
                panel_mass_share_target=audit.panel_mass_share_target,
            )
        return TrendFillAudit(
            ratio=1.0,
            source="unscaled_no_panel",
            panel_agency_count=None,
            panel_mass_share_base=None,
            panel_mass_share_target=None,
        )


def _parse_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _finite_ratio(numer: float, denom: float) -> float | None:
    if not np.isfinite(numer) or not np.isfinite(denom) or denom <= 0.0:
        return None
    ratio = float(numer / denom)
    return ratio if np.isfinite(ratio) else None


def build_trend_fill_lookup(
    panel: pd.DataFrame,
    *,
    entity_col: str,
    state_col: str = "state_fips",
    offense_col: str = "offense",
    year_col: str = "year",
    count_col: str = "reported_count_preferred",
    usable_col: str = "usable_as_observed",
    target_year: int = 2024,
    min_panel_agencies: int = TREND_FILL_MIN_PANEL_AGENCIES,
    min_panel_mass_share: float = TREND_FILL_MIN_PANEL_MASS_SHARE,
) -> TrendFillLookup:
    required = {entity_col, state_col, offense_col, year_col, count_col, usable_col}
    if panel.empty or not required.issubset(panel.columns):
        return TrendFillLookup(state_map={}, national_map={})

    usable = panel.loc[
        panel[usable_col].eq(True),
        [entity_col, state_col, offense_col, year_col, count_col],
    ].copy()
    if usable.empty:
        return TrendFillLookup(state_map={}, national_map={})

    usable[state_col] = usable[state_col].astype("string").str.zfill(2)
    usable[offense_col] = usable[offense_col].astype("string")
    usable[entity_col] = usable[entity_col].astype("string")
    usable[year_col] = pd.to_numeric(usable[year_col], errors="coerce").astype("Int64")
    usable[count_col] = pd.to_numeric(usable[count_col], errors="coerce")
    usable = usable[usable[year_col].notna() & usable[count_col].notna()].copy()
    usable[count_col] = usable[count_col].clip(lower=0.0)
    if usable.empty:
        return TrendFillLookup(state_map={}, national_map={})

    target_year = int(target_year)
    target = usable[usable[year_col].eq(target_year)].copy()
    base_years = sorted(
        int(year)
        for year in usable.loc[usable[year_col].lt(target_year), year_col].dropna().unique().tolist()
    )
    if target.empty or not base_years:
        return TrendFillLookup(state_map={}, national_map={})

    state_year_totals = (
        usable.groupby([state_col, offense_col, year_col], dropna=False)[count_col]
        .sum()
        .rename("state_year_total")
        .reset_index()
    )
    national_year_totals = (
        usable.groupby([offense_col, year_col], dropna=False)[count_col]
        .sum()
        .rename("national_year_total")
        .reset_index()
    )

    state_map: dict[tuple[str, str, int], TrendFillAudit] = {}
    national_map: dict[tuple[str, int], TrendFillAudit] = {}

    for base_year in base_years:
        base = usable[usable[year_col].eq(base_year)].copy()
        paired = base.merge(
            target,
            on=[entity_col, state_col, offense_col],
            how="inner",
            suffixes=("_base", "_target"),
        )
        if paired.empty:
            continue

        state_pairs = (
            paired.groupby([state_col, offense_col], dropna=False)
            .agg(
                panel_agency_count=(entity_col, "nunique"),
                panel_count_base=(f"{count_col}_base", "sum"),
                panel_count_target=(f"{count_col}_target", "sum"),
            )
            .reset_index()
        )
        state_pairs[year_col] = int(base_year)
        state_pairs = state_pairs.merge(
            state_year_totals.rename(
                columns={
                    year_col: f"{year_col}_base",
                    "state_year_total": "state_total_base",
                }
            ),
            left_on=[state_col, offense_col, year_col],
            right_on=[state_col, offense_col, f"{year_col}_base"],
            how="left",
        ).drop(columns=[f"{year_col}_base"], errors="ignore")
        state_pairs = state_pairs.merge(
            state_year_totals[state_year_totals[year_col].eq(target_year)]
            .drop(columns=[year_col])
            .rename(columns={"state_year_total": "state_total_target"}),
            on=[state_col, offense_col],
            how="left",
        )
        state_pairs["mass_share_base"] = (
            pd.to_numeric(state_pairs["panel_count_base"], errors="coerce")
            / pd.to_numeric(state_pairs["state_total_base"], errors="coerce").replace(0.0, np.nan)
        )
        state_pairs["mass_share_target"] = (
            pd.to_numeric(state_pairs["panel_count_target"], errors="coerce")
            / pd.to_numeric(state_pairs["state_total_target"], errors="coerce").replace(0.0, np.nan)
        )
        for row in state_pairs.itertuples(index=False):
            ratio = _finite_ratio(float(row.panel_count_target), float(row.panel_count_base))
            if ratio is None:
                continue
            if int(row.panel_agency_count) < int(min_panel_agencies):
                continue
            if str(row.offense) in TREND_FILL_RARE_OFFENSES and (
                float(row.panel_count_base) < float(TREND_FILL_RARE_MIN_PANEL_EVENT_MASS)
                or float(row.panel_count_target) < float(TREND_FILL_RARE_MIN_PANEL_EVENT_MASS)
            ):
                continue
            if not (
                pd.notna(row.mass_share_base)
                and pd.notna(row.mass_share_target)
                and float(row.mass_share_base) >= float(min_panel_mass_share)
                and float(row.mass_share_target) >= float(min_panel_mass_share)
            ):
                continue
            state_map[(str(row.state_fips).zfill(2), str(row.offense), int(base_year))] = TrendFillAudit(
                ratio=ratio,
                source="state_panel",
                panel_agency_count=int(row.panel_agency_count),
                panel_mass_share_base=float(row.mass_share_base),
                panel_mass_share_target=float(row.mass_share_target),
            )

        national_pairs = (
            paired.groupby([offense_col], dropna=False)
            .agg(
                panel_agency_count=(entity_col, "nunique"),
                panel_count_base=(f"{count_col}_base", "sum"),
                panel_count_target=(f"{count_col}_target", "sum"),
            )
            .reset_index()
        )
        national_pairs[year_col] = int(base_year)
        national_pairs = national_pairs.merge(
            national_year_totals.rename(
                columns={
                    year_col: f"{year_col}_base",
                    "national_year_total": "national_total_base",
                }
            ),
            left_on=[offense_col, year_col],
            right_on=[offense_col, f"{year_col}_base"],
            how="left",
        ).drop(columns=[f"{year_col}_base"], errors="ignore")
        national_pairs = national_pairs.merge(
            national_year_totals[national_year_totals[year_col].eq(target_year)]
            .drop(columns=[year_col])
            .rename(columns={"national_year_total": "national_total_target"}),
            on=[offense_col],
            how="left",
        )
        national_pairs["mass_share_base"] = (
            pd.to_numeric(national_pairs["panel_count_base"], errors="coerce")
            / pd.to_numeric(national_pairs["national_total_base"], errors="coerce").replace(0.0, np.nan)
        )
        national_pairs["mass_share_target"] = (
            pd.to_numeric(national_pairs["panel_count_target"], errors="coerce")
            / pd.to_numeric(national_pairs["national_total_target"], errors="coerce").replace(0.0, np.nan)
        )
        for row in national_pairs.itertuples(index=False):
            ratio = _finite_ratio(float(row.panel_count_target), float(row.panel_count_base))
            if ratio is None:
                continue
            national_map[(str(row.offense), int(base_year))] = TrendFillAudit(
                ratio=ratio,
                source="national_panel",
                panel_agency_count=int(row.panel_agency_count),
                panel_mass_share_base=(
                    float(row.mass_share_base) if pd.notna(row.mass_share_base) else None
                ),
                panel_mass_share_target=(
                    float(row.mass_share_target) if pd.notna(row.mass_share_target) else None
                ),
            )

    return TrendFillLookup(state_map=state_map, national_map=national_map)


def add_preferred_support_flags(preferred: pd.DataFrame) -> pd.DataFrame:
    """Split every preferred row into supported-as-reported vs supported-but-partial.

    One statement, applied to whichever lane selection chose:

    * **Supported** -- the chosen lane produced a report for the target year, i.e. the
      regime is anything other than `structurally_missing_or_unreliable`.
    * **True partial** -- a supported row whose chosen lane covered 1-11 of the twelve
      months, and whose partial month set is coherent (`lumpy_or_batched` is the regime
      that says it is not, so a lumpy row is read as reported rather than annualised).
    * **Usable as observed** -- every other supported row.

    Coverage is read off the SELECTED lane's own `preferred_months_reported` rather
    than off the lane family, because partiality is a property of months and not of
    who published them. Keying it on lane family made the NIBRS lane structurally
    incapable of being partial (8,006 rows in 2024 read as complete years at their
    partial value), and the additional `count > 0` guard sent the zero offenses of an
    otherwise-uplifted partial year down the fill ladder instead of leaving them the
    zeros over reported months that they are (3,164 rows). Both are the same misread:
    a zero over five months is data, and five months is not a year.
    """
    out = preferred.copy()
    counts = pd.to_numeric(out["preferred_count"], errors="coerce")
    months = pd.to_numeric(out["preferred_months_reported"], errors="coerce")
    regime = out["reporting_regime"].astype("string")
    source = out["preferred_source"].astype("string")

    # `lumpy_or_batched` is the SRS monthly file's own diagnostic -- month-share
    # concentration and a month mask contradicting the header -- so it can only speak
    # about the Return A lane. Applied to a row the NIBRS lane won it would veto an
    # annualisation on evidence about a lane that was not chosen.
    month_set_incoherent = regime.eq("lumpy_or_batched").fillna(False) & source.eq(
        SUMMARY_SOURCE
    )
    supported = counts.notna() & ~regime.eq("structurally_missing_or_unreliable").fillna(False)
    partial = supported & months.between(1, 11, inclusive="both") & ~month_set_incoherent
    out["usable_as_observed"] = supported & ~partial
    out["current_row_is_true_partial"] = partial
    return out


def build_agency_trend_fill_panel(
    *,
    paths: object,
    year_start: int,
    year_end: int,
    force_reporting_regimes_rebuild: bool = False,
    exclude_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE)),
) -> pd.DataFrame:
    """The preferred-observation panel over a span, restricted to the published scope.

    The scope restriction defaults ON (`crimerisk.scope.PRODUCTION_SCOPE_EXCLUDE`):
    territories and the non-contiguous states have no geography in this surface, so an
    agency estimate for them can never land anywhere. Leaving it to each consumer meant
    Stage 1 built 2,247 rows and 142,960 counts of Puerto Rico, Guam and Virgin Islands
    estimates every run and every consumer silently dropped them afterwards.
    """
    frames: list[pd.DataFrame] = []
    for year in range(int(year_start), int(year_end) + 1):
        preferred = build_agency_preferred_observations(
            paths=paths,
            year=year,
            force_reporting_regimes_rebuild=bool(force_reporting_regimes_rebuild),
        )
        if preferred.empty:
            continue
        preferred = add_preferred_support_flags(preferred)
        preferred["year"] = int(year)
        frames.append(preferred)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    if exclude_state_abbrs and "state_abbr" in panel.columns:
        panel["state_abbr"] = panel["state_abbr"].astype("string").str.upper()
        panel = panel[~panel["state_abbr"].isin(set(exclude_state_abbrs))].copy()
    panel["ori9"] = panel["ori9"].astype("string")
    panel["state_fips"] = panel["state_fips"].astype("string").str.zfill(2)
    panel["offense"] = panel["offense"].astype("string")
    panel["preferred_count"] = pd.to_numeric(panel["preferred_count"], errors="coerce")
    return panel


def build_agency_trend_fill_lookup(
    *,
    paths: object,
    year_start: int,
    year_end: int,
    target_year: int,
    force_reporting_regimes_rebuild: bool = False,
    exclude_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE)),
) -> TrendFillLookup:
    panel = build_agency_trend_fill_panel(
        paths=paths,
        year_start=int(year_start),
        year_end=int(year_end),
        force_reporting_regimes_rebuild=bool(force_reporting_regimes_rebuild),
        exclude_state_abbrs=tuple(exclude_state_abbrs),
    )
    return build_trend_fill_lookup(
        panel,
        entity_col="ori9",
        count_col="preferred_count",
        target_year=int(target_year),
    )


def scaled_history_median(
    usable_history: pd.DataFrame,
    *,
    state_fips: object,
    offense: object,
    trend_fill_lookup: TrendFillLookup,
    count_col: str = "reported_count_preferred",
    year_col: str = "year",
) -> tuple[float | None, TrendFillAudit]:
    scaled_counts: list[float] = []
    raw_counts: list[float] = []
    audits: list[TrendFillAudit] = []
    for row in usable_history[[year_col, count_col]].itertuples(index=False, name=None):
        year, raw_count = row
        count_value = pd.to_numeric(raw_count, errors="coerce")
        count = float(count_value) if pd.notna(count_value) else 0.0
        audit = trend_fill_lookup.resolve(state_fips=state_fips, offense=offense, base_year=year)
        scaled = count * float(audit.ratio)
        if np.isfinite(scaled):
            scaled_counts.append(float(max(0.0, scaled)))
            raw_counts.append(float(max(0.0, count)))
            audits.append(audit)
    if not scaled_counts:
        return None, TrendFillAudit(
            ratio=1.0,
            source="unscaled_no_panel",
            panel_agency_count=None,
            panel_mass_share_base=None,
            panel_mass_share_target=None,
        )
    raw_median = float(np.median(raw_counts)) if raw_counts else float("nan")
    scaled_median = float(np.median(scaled_counts))
    effective_ratio = scaled_median / raw_median if np.isfinite(raw_median) and raw_median > 0.0 else 1.0
    sources = sorted({str(audit.source) for audit in audits})
    panel_counts = [audit.panel_agency_count for audit in audits if audit.panel_agency_count is not None]
    mass_base = [audit.panel_mass_share_base for audit in audits if audit.panel_mass_share_base is not None]
    mass_target = [audit.panel_mass_share_target for audit in audits if audit.panel_mass_share_target is not None]
    return scaled_median, TrendFillAudit(
        ratio=float(effective_ratio) if np.isfinite(effective_ratio) else 1.0,
        source=sources[0] if len(sources) == 1 else "mixed_panel_scaled_median",
        panel_agency_count=min(panel_counts) if panel_counts else None,
        panel_mass_share_base=min(mass_base) if mass_base else None,
        panel_mass_share_target=min(mass_target) if mass_target else None,
    )


AGENCY_TARGET_ESTIMATE_COLUMNS = [
    "ori9",
    "state_fips",
    "offense",
    "reported_count_current",
    "reported_count_current_supported",
    "estimated_count",
    "agency_adjustment_count",
    "agency_estimate_source",
    "partial_uplift_raw_count",
    "partial_uplift_sanity_cap_flag",
    "partial_uplift_panel_max_annual_total",
    "partial_uplift_agency_raw_total",
    "partial_uplift_cap_estimate",
    "agency_estimate_review_flag",
    "agency_estimate_review_reason",
    "masked_gap_reclassified",
    "masked_gap_kind",
    "stage1_adjudicated_kind",
    "stage1_adjudicated_case",
    "reference_years_excluded",
]


# Masked-gap detector thresholds (calibrated on the two diagnosed 2024 cases --
# Wichita PD KS0870300 and Savannah-Chatham Metro GA0250300; see docs/STATE.md).
# An agency-offense is a "masked gap" when its target-year data is materially
# incomplete but every administrative flag (months_reported=12, months_missing=0,
# quality_tier=high) claims a clean full year, so the quality machinery trusts it
# and no fill fires.
MASKED_GAP_POPULATION_MIN = 25_000.0
# High-volume floor: the reference (prior-year NIBRS count for the months-drop
# signature; trailing usable preferred count for the collapse signature) must be at
# least this large. Below it, nonzero-month counts and year-over-year ratios are
# noise-dominated (a 30-murder agency legitimately has empty months).
MASKED_GAP_MIN_REFERENCE_COUNT = 100.0
# Months-drop signature (Wichita): current-year NIBRS nonzero_months at least this
# far below the agency's own prior-year nonzero_months while months_reported claims 12.
MASKED_GAP_NONZERO_MONTHS_DROP_MIN = 2.0
# Months-drop co-condition: the annual count must also have materially fallen
# (current < ratio * prior). A months drop with a stable total is not a masked gap
# in the published total, and uplifting it would inflate.
MASKED_GAP_PARTIAL_COUNT_DROP_RATIO = 0.75
# Collapse signature (Savannah): current reported count below this fraction of the
# agency's own trailing usable level while claiming a complete year.
MASKED_GAP_COLLAPSE_FRACTION = 0.25
MASKED_GAP_TRAILING_YEARS = 3
# Agency-level admission rule, applied ON TOP of the per-row thresholds above (which are
# the pre-stated design values, not post-hoc tightened): an agency's flags are admitted
# for reclassification only if (1) the agency has >= MASKED_GAP_MIN_FLAGGED_OFFENSES
# flagged offenses, OR (2) a single flagged offense at population >=
# MASKED_GAP_SINGLE_OFFENSE_POPULATION_MIN, OR (3) a single flagged offense whose
# reported count is below MASKED_GAP_SINGLE_OFFENSE_COLLAPSE_FRACTION of its reference
# (near-total collapse). Rationale: submission/data-capture failures are mechanically
# offense-broad -- both confirmed 2024 cases (Wichita KS0870300 and Savannah-Chatham
# GA0250300) fired on 5-6 offenses -- while a genuine single-offense collapse at a
# small agency is plausibly a real crime or reporting-policy shift; at metropolitan
# scale a 75% single-offense collapse behind intact completeness flags stops being
# plausible as real. Arm (3) exists because that single-offense ambiguity only covers
# MODERATE collapses: a reported count below 5% of the trailing reference is data
# absence, not crime variance, at any agency size above the 25k population floor. Arm
# (3) was added after the borderline review of the two-arm rule -- North Port FL
# (larceny 4 vs trailing 770, 0.5%) was the exposing case -- with the discipline that
# the arm derives from the mechanism (near-total absence cannot be a real crime
# shift), not from the case list; at 5% it admits nothing else nationally (the
# next-lowest dropped single-offense ratio is 10%).
MASKED_GAP_MIN_FLAGGED_OFFENSES = 2
MASKED_GAP_SINGLE_OFFENSE_POPULATION_MIN = 100_000.0
MASKED_GAP_SINGLE_OFFENSE_COLLAPSE_FRACTION = 0.05
# Months-drop treatment escalation (consistency test): the true-partial month-ratio is
# only valid when the surviving months look clean. expected_from_surviving =
# nonzero_months * (prior-year NIBRS count / 12); if the reported count is below
# MASKED_GAP_PARTIAL_CONSISTENCY_FACTOR of that, the surviving months are themselves
# corrupt, so the month-ratio would under-recover by construction and the agency-offense
# escalates to the gapped-year trend/hist ladder (kind "partial_months_escalated").
# This is a per-row mechanism test, NOT a blanket switch to the ladder, and the factor
# was NOT chosen to match the FBI CDE state estimates -- CDE closure is corroboration
# only (see docs/STATE.md).
MASKED_GAP_PARTIAL_CONSISTENCY_FACTOR = 0.75

# Flag kinds treated by the gapped-year ladder (vs the true-partial month-ratio).
MASKED_GAP_LADDER_KINDS = frozenset(("collapsed_count", "partial_months_escalated"))

MASKED_GAP_FLAG_COLUMNS = [
    "ori9",
    "offense",
    "masked_gap_kind",
    "masked_gap_effective_months",
    "masked_gap_current_count",
    "masked_gap_reference_count",
    "masked_gap_prior_nonzero_months",
    "masked_gap_current_nonzero_months",
    "masked_gap_expected_from_surviving",
    "masked_gap_population",
]


def _load_masked_gap_observations(paths: object) -> pd.DataFrame:
    """Load the raw agency-year observations columns the masked-gap detector needs.
    Factored out of build_masked_gap_flags so callers that run the detector against
    multiple candidate years (reference-year hygiene) can load once and reuse."""
    obs = pd.read_parquet(
        paths.state_dir / "observations" / "agency_year_observations.parquet",
        columns=[
            "ori9",
            "year",
            "source",
            "offense",
            "count",
            "months_reported",
            "months_missing",
            "nonzero_months",
            "population",
        ],
    )
    obs["ori9"] = obs["ori9"].astype("string")
    obs["offense"] = obs["offense"].astype("string")
    obs["year"] = pd.to_numeric(obs["year"], errors="coerce").astype("Int64")
    return obs


def build_masked_gap_flags(
    paths: object,
    *,
    target_year: int,
    agency_panel: pd.DataFrame,
    obs: pd.DataFrame | None = None,
    population_min: float = MASKED_GAP_POPULATION_MIN,
    min_reference_count: float = MASKED_GAP_MIN_REFERENCE_COUNT,
    nonzero_months_drop_min: float = MASKED_GAP_NONZERO_MONTHS_DROP_MIN,
    partial_count_drop_ratio: float = MASKED_GAP_PARTIAL_COUNT_DROP_RATIO,
    collapse_fraction: float = MASKED_GAP_COLLAPSE_FRACTION,
    trailing_years: int = MASKED_GAP_TRAILING_YEARS,
    min_flagged_offenses: int = MASKED_GAP_MIN_FLAGGED_OFFENSES,
    single_offense_population_min: float = MASKED_GAP_SINGLE_OFFENSE_POPULATION_MIN,
    single_offense_collapse_fraction: float = MASKED_GAP_SINGLE_OFFENSE_COLLAPSE_FRACTION,
    partial_consistency_factor: float = MASKED_GAP_PARTIAL_CONSISTENCY_FACTOR,
) -> pd.DataFrame:
    """Detect agency-offenses whose target-year data is materially incomplete while the
    administrative flags claim a clean complete year ("masked gap"). Two signatures:

    - "partial_months" (Wichita KS0870300 2024): the NIBRS rollup's nonzero_months
      drops >= nonzero_months_drop_min below the agency's own prior-year nonzero_months
      AND the annual count materially fell, while months_reported=12/months_missing=0.
      Treatment downstream: true-partial-ratio semantics with effective months =
      current nonzero_months -- UNLESS the surviving-months consistency test fails
      (reported count < partial_consistency_factor * nonzero_months * prior_count/12),
      in which case the surviving months are themselves corrupt and the row escalates
      to "partial_months_escalated" (gapped-year ladder treatment).
    - "collapsed_count" (Savannah-Chatham GA0250300 2024): the reported annual count
      falls below collapse_fraction of the agency's own trailing usable level while
      claiming >= 12 reported months. Treatment downstream: the regular gapped-year
      trend/hist-median ladder.

    Both signatures only fire where the preferred target-year row currently reads as
    usable_as_observed (i.e. the gap is actually masked -- rows the machinery already
    distrusts are already handled by the existing fill paths) and the agency's
    target-year FBI population is >= population_min. If both fire for the same
    agency-offense, collapsed_count wins (the stronger signal; the ladder replaces the
    whole year rather than uplifting a corrupt partial count).

    Finally the agency-level ADMISSION RULE applies (see the constants block above for
    the rationale): flags are kept only for agencies with >= min_flagged_offenses
    flagged offenses, or a single flagged offense at population >=
    single_offense_population_min, or a single flagged offense in near-total collapse
    (reported count < single_offense_collapse_fraction of its reference).
    """
    target_year = int(target_year)
    if agency_panel is None or agency_panel.empty:
        return pd.DataFrame(columns=MASKED_GAP_FLAG_COLUMNS)

    if obs is None:
        obs = _load_masked_gap_observations(paths)
    else:
        obs = obs.copy()

    population = (
        obs[obs["year"].eq(target_year)]
        .groupby("ori9", dropna=False)["population"]
        .max()
        .rename("masked_gap_population")
        .reset_index()
    )
    population["masked_gap_population"] = pd.to_numeric(
        population["masked_gap_population"], errors="coerce"
    ).fillna(0.0)

    panel = agency_panel.copy()
    panel["ori9"] = panel["ori9"].astype("string")
    panel["offense"] = panel["offense"].astype("string")
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["preferred_count"] = pd.to_numeric(panel["preferred_count"], errors="coerce")
    panel["preferred_months_reported"] = pd.to_numeric(
        panel["preferred_months_reported"], errors="coerce"
    )
    panel["usable_as_observed"] = panel["usable_as_observed"].fillna(False).astype(bool)

    current = panel[panel["year"].eq(target_year)][
        ["ori9", "offense", "preferred_count", "preferred_months_reported", "usable_as_observed"]
    ].rename(
        columns={
            "preferred_count": "masked_gap_current_count",
            "preferred_months_reported": "_current_claimed_months",
            "usable_as_observed": "_current_usable",
        }
    )
    current = current.merge(population, on="ori9", how="left")
    current["masked_gap_population"] = current["masked_gap_population"].fillna(0.0)
    # Only rows the machinery currently trusts can be MASKED gaps.
    current = current[
        current["_current_usable"]
        & current["masked_gap_population"].ge(float(population_min))
        & pd.to_numeric(current["_current_claimed_months"], errors="coerce").fillna(0.0).ge(12.0)
    ].copy()
    if current.empty:
        return pd.DataFrame(columns=MASKED_GAP_FLAG_COLUMNS)

    # Signature (a): NIBRS nonzero-months drop with a co-occurring count drop.
    nibrs = obs[obs["source"].eq(NIBRS_SOURCE) & obs["year"].isin([target_year - 1, target_year])].copy()
    nibrs["count"] = pd.to_numeric(nibrs["count"], errors="coerce")
    nibrs["nonzero_months"] = pd.to_numeric(nibrs["nonzero_months"], errors="coerce")
    nibrs["months_reported"] = pd.to_numeric(nibrs["months_reported"], errors="coerce")
    nibrs["months_missing"] = pd.to_numeric(nibrs["months_missing"], errors="coerce")
    nibrs_curr = nibrs[nibrs["year"].eq(target_year)][
        ["ori9", "offense", "count", "nonzero_months", "months_reported", "months_missing"]
    ].rename(
        columns={
            "count": "_nibrs_count_curr",
            "nonzero_months": "masked_gap_current_nonzero_months",
            "months_reported": "_nibrs_months_reported_curr",
            "months_missing": "_nibrs_months_missing_curr",
        }
    )
    nibrs_prev = nibrs[nibrs["year"].eq(target_year - 1)][
        ["ori9", "offense", "count", "nonzero_months"]
    ].rename(
        columns={
            "count": "_nibrs_count_prev",
            "nonzero_months": "masked_gap_prior_nonzero_months",
        }
    )
    partial = nibrs_curr.merge(nibrs_prev, on=["ori9", "offense"], how="inner")
    partial_mask = (
        partial["_nibrs_months_reported_curr"].fillna(0.0).ge(12.0)
        & partial["_nibrs_months_missing_curr"].fillna(0.0).le(0.0)
        & partial["masked_gap_current_nonzero_months"].notna()
        & partial["masked_gap_prior_nonzero_months"].notna()
        & (
            partial["masked_gap_prior_nonzero_months"] - partial["masked_gap_current_nonzero_months"]
        ).ge(float(nonzero_months_drop_min))
        & partial["_nibrs_count_prev"].fillna(0.0).ge(float(min_reference_count))
        & partial["_nibrs_count_curr"].fillna(0.0).lt(
            float(partial_count_drop_ratio) * partial["_nibrs_count_prev"].fillna(0.0)
        )
        & partial["masked_gap_current_nonzero_months"].gt(0.0)
    )
    partial = partial[partial_mask].copy()
    partial["masked_gap_kind"] = "partial_months"
    partial["masked_gap_effective_months"] = partial["masked_gap_current_nonzero_months"]
    partial["masked_gap_reference_count"] = partial["_nibrs_count_prev"]

    # Signature (b): collapsed count vs the agency's own trailing usable level.
    trailing = panel[
        panel["usable_as_observed"]
        & panel["year"].between(target_year - int(trailing_years), target_year - 1, inclusive="both")
    ]
    reference = (
        trailing.groupby(["ori9", "offense"], dropna=False)["preferred_count"]
        .max()
        .rename("masked_gap_reference_count")
        .reset_index()
    )
    collapse = current.merge(reference, on=["ori9", "offense"], how="inner")
    collapse_mask = (
        pd.to_numeric(collapse["masked_gap_reference_count"], errors="coerce").fillna(0.0).ge(
            float(min_reference_count)
        )
        & pd.to_numeric(collapse["masked_gap_current_count"], errors="coerce").fillna(0.0).lt(
            float(collapse_fraction)
            * pd.to_numeric(collapse["masked_gap_reference_count"], errors="coerce").fillna(0.0)
        )
    )
    collapse = collapse[collapse_mask].copy()
    collapse["masked_gap_kind"] = "collapsed_count"
    collapse["masked_gap_effective_months"] = np.nan
    collapse["masked_gap_prior_nonzero_months"] = np.nan
    collapse["masked_gap_current_nonzero_months"] = np.nan
    collapse["masked_gap_expected_from_surviving"] = np.nan

    # Restrict signature (a) to rows that pass the usable/population/claimed-complete
    # screen, and attach the current preferred count + population for reporting.
    partial = partial.merge(
        current[["ori9", "offense", "masked_gap_current_count", "masked_gap_population"]],
        on=["ori9", "offense"],
        how="inner",
    )
    # Surviving-months consistency test: if the reported count is below
    # partial_consistency_factor of what the surviving nonzero months would carry at the
    # prior-year monthly mean, the surviving months are themselves corrupt and the
    # month-ratio would under-recover -> escalate to the gapped-year ladder.
    partial["masked_gap_expected_from_surviving"] = (
        pd.to_numeric(partial["masked_gap_current_nonzero_months"], errors="coerce")
        * pd.to_numeric(partial["masked_gap_reference_count"], errors="coerce")
        / 12.0
    )
    escalate = pd.to_numeric(partial["masked_gap_current_count"], errors="coerce").fillna(0.0).lt(
        float(partial_consistency_factor)
        * pd.to_numeric(partial["masked_gap_expected_from_surviving"], errors="coerce").fillna(0.0)
    )
    partial.loc[escalate, "masked_gap_kind"] = "partial_months_escalated"

    combined = pd.concat(
        [
            collapse[MASKED_GAP_FLAG_COLUMNS],
            partial[MASKED_GAP_FLAG_COLUMNS],
        ],
        ignore_index=True,
    )
    # collapsed_count wins when both signatures fire for the same agency-offense.
    combined["_kind_rank"] = combined["masked_gap_kind"].map(
        {"collapsed_count": 0, "partial_months_escalated": 1, "partial_months": 2}
    )
    combined = (
        combined.sort_values(["ori9", "offense", "_kind_rank"], kind="mergesort")
        .drop_duplicates(subset=["ori9", "offense"], keep="first")
        .drop(columns=["_kind_rank"])
    )

    # Agency-level admission rule: (1) >= min_flagged_offenses flagged offenses, or (2)
    # a single flagged offense at population >= single_offense_population_min, or (3) a
    # single flagged offense in near-total collapse (reported count <
    # single_offense_collapse_fraction of its reference) -- rationale in the constants
    # block above.
    combined["_collapse_ratio"] = pd.to_numeric(
        combined["masked_gap_current_count"], errors="coerce"
    ).fillna(0.0) / pd.to_numeric(combined["masked_gap_reference_count"], errors="coerce").replace(
        0.0, np.nan
    )
    per_agency = combined.groupby("ori9", dropna=False).agg(
        _flag_count=("offense", "size"),
        _population=("masked_gap_population", "max"),
        _min_collapse_ratio=("_collapse_ratio", "min"),
    )
    admitted_oris = per_agency[
        per_agency["_flag_count"].ge(int(min_flagged_offenses))
        | pd.to_numeric(per_agency["_population"], errors="coerce").fillna(0.0).ge(
            float(single_offense_population_min)
        )
        | pd.to_numeric(per_agency["_min_collapse_ratio"], errors="coerce").lt(
            float(single_offense_collapse_fraction)
        )
    ].index
    combined = combined[combined["ori9"].isin(admitted_oris)].drop(columns=["_collapse_ratio"]).copy()
    return combined.sort_values(["ori9", "offense"], kind="mergesort").reset_index(drop=True)


REFERENCE_YEAR_EXCLUSION_COLUMNS = ["ori9", "offense", "year", "masked_gap_kind"]


def build_reference_year_masked_gap_years(
    paths: object,
    *,
    agency_panel: pd.DataFrame,
    candidate_years: object,
    population_min: float = MASKED_GAP_POPULATION_MIN,
    min_reference_count: float = MASKED_GAP_MIN_REFERENCE_COUNT,
    nonzero_months_drop_min: float = MASKED_GAP_NONZERO_MONTHS_DROP_MIN,
    partial_count_drop_ratio: float = MASKED_GAP_PARTIAL_COUNT_DROP_RATIO,
    collapse_fraction: float = MASKED_GAP_COLLAPSE_FRACTION,
    trailing_years: int = MASKED_GAP_TRAILING_YEARS,
    min_flagged_offenses: int = MASKED_GAP_MIN_FLAGGED_OFFENSES,
    single_offense_population_min: float = MASKED_GAP_SINGLE_OFFENSE_POPULATION_MIN,
    single_offense_collapse_fraction: float = MASKED_GAP_SINGLE_OFFENSE_COLLAPSE_FRACTION,
    partial_consistency_factor: float = MASKED_GAP_PARTIAL_CONSISTENCY_FACTOR,
) -> pd.DataFrame:
    """Reference-year hygiene: run the SAME masked-gap detector (build_masked_gap_flags,
    both signatures, same admission rule) against each candidate historical year AS IF
    it were the target year -- i.e. evaluated against ITS OWN preceding usable years
    (prior-year NIBRS nonzero-months for the partial_months signature, its own trailing
    1-3 years for the collapsed_count signature). No new math: this is the existing
    detector called once per candidate year instead of once for the current target year.

    Returns one row per (ori9, offense, year) where that YEAR is itself a masked gap and
    therefore must be EXCLUDED as a fill reference (not down-weighted -- corrupt data is
    not evidence). Concrete case: Suffolk County PD NY0510100's 2023 larceny fires the
    partial_months signature against 2022 (its own nonzero-months drop + count collapse),
    so 2023 is excluded as a reference for the 2024 gapped-year fill.

    Callers apply the exclusion by filtering historical rows out of their usable_hist /
    reference set wherever (ori9, offense, year) appears in this frame's rows; the
    existing ladder (trend_log_linear -> carryforward/hist_median -> peer fallback)
    already degrades gracefully when fewer reference years survive, so no special-casing
    is needed for the recursion floor.
    """
    if agency_panel is None or agency_panel.empty:
        return pd.DataFrame(columns=REFERENCE_YEAR_EXCLUSION_COLUMNS)
    years = sorted({int(y) for y in candidate_years})
    if not years:
        return pd.DataFrame(columns=REFERENCE_YEAR_EXCLUSION_COLUMNS)

    obs = _load_masked_gap_observations(paths)
    frames: list[pd.DataFrame] = []
    for year in years:
        flags = build_masked_gap_flags(
            paths,
            target_year=year,
            agency_panel=agency_panel,
            obs=obs,
            population_min=population_min,
            min_reference_count=min_reference_count,
            nonzero_months_drop_min=nonzero_months_drop_min,
            partial_count_drop_ratio=partial_count_drop_ratio,
            collapse_fraction=collapse_fraction,
            trailing_years=trailing_years,
            min_flagged_offenses=min_flagged_offenses,
            single_offense_population_min=single_offense_population_min,
            single_offense_collapse_fraction=single_offense_collapse_fraction,
            partial_consistency_factor=partial_consistency_factor,
        )
        if flags.empty:
            continue
        tagged = flags[["ori9", "offense", "masked_gap_kind"]].copy()
        tagged["year"] = int(year)
        frames.append(tagged)
    if not frames:
        return pd.DataFrame(columns=REFERENCE_YEAR_EXCLUSION_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined[REFERENCE_YEAR_EXCLUSION_COLUMNS].sort_values(
        ["ori9", "offense", "year"], kind="mergesort"
    ).reset_index(drop=True)


def _reference_year_exclusion_key_set(
    reference_year_exclusions: pd.DataFrame | None,
) -> set[tuple[str, str, int]]:
    if reference_year_exclusions is None or reference_year_exclusions.empty:
        return set()
    ori9 = reference_year_exclusions["ori9"].astype("string")
    offense = reference_year_exclusions["offense"].astype("string")
    year = pd.to_numeric(reference_year_exclusions["year"], errors="coerce")
    return {
        (str(o), str(off), int(y))
        for o, off, y in zip(ori9, offense, year, strict=True)
        if pd.notna(y)
    }


def apply_masked_gap_reclassification(
    agency_panel: pd.DataFrame,
    *,
    target_year: int,
    masked_gap_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Reclassify flagged target-year rows in an agency preferred panel so the EXISTING
    fill machinery treats them as gapped: partial_months rows become true-partial with
    effective months = current NIBRS nonzero_months (true-partial-ratio semantics);
    collapsed_count and partial_months_escalated rows become plain unusable
    (trend/hist-median ladder). No new fill math -- only the usable/true-partial flags
    and months the machinery already keys on.
    """
    if masked_gap_flags is None or masked_gap_flags.empty or agency_panel.empty:
        return agency_panel
    out = agency_panel.copy()
    keys = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "ori9": out["ori9"].astype("string"),
                "offense": out["offense"].astype("string"),
            },
            index=out.index,
        )
    )
    flag_frame = masked_gap_flags.set_index(
        pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "ori9": masked_gap_flags["ori9"].astype("string"),
                    "offense": masked_gap_flags["offense"].astype("string"),
                }
            )
        )
    )
    is_target = pd.to_numeric(out["year"], errors="coerce").eq(int(target_year))
    flagged = pd.Series(keys.isin(flag_frame.index), index=out.index) & is_target
    if not flagged.any():
        return agency_panel

    kinds = pd.Series(pd.NA, index=out.index, dtype="string")
    months = pd.Series(np.nan, index=out.index, dtype=float)
    flagged_positions = out.index[flagged]
    kinds.loc[flagged_positions] = (
        flag_frame["masked_gap_kind"].reindex(keys[flagged.to_numpy()]).to_numpy()
    )
    months.loc[flagged_positions] = pd.to_numeric(
        flag_frame["masked_gap_effective_months"].reindex(keys[flagged.to_numpy()]), errors="coerce"
    ).to_numpy()

    partial_rows = flagged & kinds.eq("partial_months") & months.between(1.0, 11.999999, inclusive="both")
    ladder_rows = flagged & ~partial_rows

    out.loc[flagged, "usable_as_observed"] = False
    out.loc[ladder_rows, "current_row_is_true_partial"] = False
    out.loc[partial_rows, "current_row_is_true_partial"] = True
    out.loc[partial_rows, "preferred_months_reported"] = months.loc[partial_rows]
    return out


STAGE1_ADJUDICATED_LADDER_KIND = "stage1_adjudicated_missing"
STAGE1_ADJUDICATED_PARTIAL_KIND = "stage1_adjudicated_partial"


def apply_stage1_adjudicated_usability(
    agency_panel: pd.DataFrame,
    *,
    target_year: int,
    directives: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Take the adjudicated agency-years out of "measured as published", nothing more.

    The token-reporter class is a compliant twelve-month header in front of a number that is not a
    year of policing: Hanford CHP reported 107 and 88 Part-I equivalents in 2022-2023 and one motor
    vehicle theft in 2024. No rule separates that from a very quiet agency, so the cases were
    reviewed one by one, and a `token_reporting_flag` verdict says exactly one thing here -- the
    year is not usable as observed. Where the reviewer named a believable month count the row
    becomes the true-partial the pipeline already annualises; where nobody could, it goes to the
    same trend / hist-median ladder every other unusable year goes to.

    This is deliberately the same three assignments `apply_masked_gap_reclassification` makes, for
    the same reason: no new fill math exists for adjudicated rows. What the detector concludes from
    month masks and collapse ratios, these rows conclude from a reviewer reading the record.

    The zero-vs-missing class arrives here too, because the ESTIMATOR sees one population. Those
    rows were already made unusable at the regime site, and this function FAILS CLOSED if they were
    not: a `misread_missing` verdict that reaches the estimator still reading as observed means the
    regime wiring did not run, and silently flipping the flag here would hide that.
    """
    counts = {
        "stage1_adjudicated_directives": int(len(directives)) if directives is not None else 0,
        "stage1_adjudicated_rows_to_ladder": 0,
        "stage1_adjudicated_rows_to_partial": 0,
        "stage1_adjudicated_agency_years_matched": 0,
        "stage1_adjudicated_directives_not_in_panel": 0,
    }
    if directives is None or directives.empty or agency_panel.empty:
        return agency_panel, counts

    out = agency_panel.copy()
    ori9 = out["ori9"].astype("string").str.upper()
    is_target = pd.to_numeric(out["year"], errors="coerce").eq(int(target_year))
    directive_by_ori = directives[directives["year"].astype(int).eq(int(target_year))].set_index(
        directives.loc[directives["year"].astype(int).eq(int(target_year)), "ori9"]
        .astype("string")
        .str.upper()
    )
    matched_ori = ori9.where(is_target).map(directive_by_ori["directive"])
    months = ori9.where(is_target).map(directive_by_ori["believable_months"])
    months = pd.to_numeric(months, errors="coerce")
    registry = ori9.where(is_target).map(directive_by_ori["source_registry"])

    partial_rows = matched_ori.eq("partial_year") & months.between(
        1.0, 11.999999, inclusive="both"
    )
    ladder_rows = matched_ori.eq("reads_missing")

    # Cross-site check: the zero registry's verdicts land at the regime site, so by the time the
    # estimator sees them the panel must already read them as unusable.
    zero_registry = registry.astype("string").str.contains("zero_missing_adjudicated", na=False)
    still_observed = out.index[
        ladder_rows
        & zero_registry
        & out["usable_as_observed"].fillna(False).astype(bool)
    ]
    if len(still_observed):
        offenders = (
            out.loc[still_observed, ["ori9", "year", "offense"]].head(20).to_dict(orient="records")
        )
        raise Stage1AdjudicationError(
            f"{len(still_observed)} adjudicated zero-vs-missing agency-offense row(s) still read as "
            "usable_as_observed when the estimator received them, so the reporting-regime site did "
            f"not apply the registry. Rebuild the reporting regimes. First rows: {offenders}"
        )

    out.loc[ladder_rows | partial_rows, "usable_as_observed"] = False
    out.loc[ladder_rows, "current_row_is_true_partial"] = False
    out.loc[partial_rows, "current_row_is_true_partial"] = True
    out.loc[partial_rows, "preferred_months_reported"] = months.loc[partial_rows]

    counts["stage1_adjudicated_rows_to_ladder"] = int(ladder_rows.sum())
    counts["stage1_adjudicated_rows_to_partial"] = int(partial_rows.sum())
    # This counter is a cross-consumer mismatch sentinel. The estimator applies the directive on
    # its own panel copy; benchmark_imputation independently loads the same fail-closed directives
    # and subtracts reads-missing ORIs from its supported set. Therefore no reads-missing directive
    # is now flipped ONLY inside the estimator.
    benchmark_recognises = matched_ori.eq("reads_missing")
    counts["stage1_adjudicated_flipped_only_inside_the_estimator"] = int(
        (
            ladder_rows
            & ~zero_registry
            & agency_panel["usable_as_observed"].fillna(False).astype(bool)
            & ~benchmark_recognises
        )
        .pipe(lambda mask: out.loc[mask, "ori9"].nunique())
    )
    matched = matched_ori.notna()
    counts["stage1_adjudicated_agency_years_matched"] = int(
        out.loc[matched, ["ori9", "year"]].drop_duplicates().shape[0]
    )
    counts["stage1_adjudicated_directives_not_in_panel"] = int(
        len(directive_by_ori) - counts["stage1_adjudicated_agency_years_matched"]
    )
    return out, counts


def stage1_adjudicated_estimate_kinds(directives: pd.DataFrame) -> pd.DataFrame:
    """Per-ORI kind labels, for the estimate frame's provenance column and the fill assertion."""
    if directives is None or directives.empty:
        return pd.DataFrame(columns=["ori9", "stage1_adjudicated_kind", "stage1_adjudicated_case"])
    kind = np.where(
        directives["directive"].eq("partial_year")
        & pd.to_numeric(directives["believable_months"], errors="coerce").between(
            1.0, 11.999999, inclusive="both"
        ),
        STAGE1_ADJUDICATED_PARTIAL_KIND,
        STAGE1_ADJUDICATED_LADDER_KIND,
    )
    return pd.DataFrame(
        {
            "ori9": directives["ori9"].astype("string").str.upper(),
            "stage1_adjudicated_kind": pd.Series(kind, dtype="string"),
            "stage1_adjudicated_case": directives["case_id"].astype("string"),
        }
    ).drop_duplicates(subset=["ori9"], keep="first")


# An own-history fill claims that the agency's own recent level is the best available
# estimate of its target-year level. That claim expires: the `hist_median` rung used to
# accept a reference from any year back to 2018, so 2,168 agencies with no usable report
# since 2021 carried 163,614 filled counts into 2024 -- 52% of all fill mass, on levels
# up to six years stale. Two years is the bound because it is the span over which the
# ladder's own carry-forward and trend rungs already operate (`max_year_gap_for_trend`
# is one year; the median rung reaches one further), so beyond it the pipeline is not
# extrapolating a reporting gap, it is reanimating an agency that stopped reporting.
# Agencies past the bound produce no fill at all: rostered ones become silent units for
# benchmark_imputation, which sizes their territory against an external benchmark
# instead of against their own memory, and off-roster ones are dead.
FILL_MAX_REFERENCE_AGE_YEARS = 2

AGENCY_NO_HISTORY_ESTIMATE_SOURCE = "current_count_no_history"
AGENCY_STALE_HISTORY_ESTIMATE_SOURCE = "no_recent_history_beyond_fill_recency_bound"
# Estimate sources that carry no evidence of the target year at all. When the agency
# also reported nothing in the target year, the row is not an estimate of anything and
# is dropped rather than published as a zero or a fabricated level.
UNEVIDENCED_ESTIMATE_SOURCES = frozenset(
    {AGENCY_NO_HISTORY_ESTIMATE_SOURCE, AGENCY_STALE_HISTORY_ESTIMATE_SOURCE}
)


def _drop_silent_agency_estimates(result: pd.DataFrame) -> pd.DataFrame:
    """Drop agency-offense rows for agencies with no current report and no usable evidence.

    A genuinely silent agency is not evidence of anything, so it gets no estimate row at
    all -- not a peer-median anchor, and not a zero, which would be just as fabricated.
    Absence is the point: without a row the agency contributes no observed mass and no
    fill mass, `_build_county_remainder_group_targets` never promotes its county into a
    `county_remainder` group, and the territory stays in the state nonmunicipal residual
    where the covariate model allocates it and raking conserves the state total. Agencies
    that DO report -- including Kenedy-class agencies whose report is a true zero -- keep
    their rows and anchor their counties at their real totals.

    An agency whose only history is older than the fill recency bound is silent in
    exactly the same sense: it has neither a current report nor a reference the ladder
    is willing to stand behind.
    """
    silent = (
        result["agency_estimate_source"].isin(UNEVIDENCED_ESTIMATE_SOURCES)
        & pd.to_numeric(result["reported_count_current"], errors="coerce").fillna(0.0).le(0.0)
    )
    return result.loc[~silent].reset_index(drop=True)


def _drop_superseded_ori_estimates(
    result: pd.DataFrame, *, succession_ledger: pd.DataFrame
) -> pd.DataFrame:
    """Drop every estimate row for an ORI a live successor already covers in full.

    Not merely un-filled: excluded. A superseded ORI's territory is not silent -- the
    successor reports it -- so leaving the row in place would double the jurisdiction's
    mass through the fill ladder, and marking it silent would double it again through
    benchmark imputation.
    """
    if result.empty or succession_ledger.empty:
        return result
    superseded = set(succession_ledger["superseded_ori9"].astype("string"))
    keep = ~result["ori9"].astype("string").isin(superseded)
    return result.loc[keep].reset_index(drop=True)


AGENCY_OBSERVED_ESTIMATE_SOURCE = "observed"
# Sources grounded in the agency-year's own report on its own chosen lane: the observed
# count itself, and the annualization of a lane that reported only part of the year.
# Everything else takes its level from other years or other agencies -- a fill.
LANE_GROUNDED_ESTIMATE_SOURCES = frozenset(
    {AGENCY_OBSERVED_ESTIMATE_SOURCE, "true_partial_month_ratio"}
)
FILL_MASS_BASELINE_FILENAME = "agency_fill_mass_baseline.json"


def _assert_no_fill_where_the_chosen_lane_reported(
    result: pd.DataFrame,
    *,
    current_rows: pd.DataFrame,
) -> None:
    """Fail closed when an offense got a fill although its chosen lane covered the year.

    Lane coherence: the preferred source is picked for the agency-year, and every Part I
    offense is then defined within that lane -- an offense the lane did not record is a
    zero over the lane's reported months, not missing data. So when the chosen lane
    covered the full twelve months, no offense of that agency-year can legitimately fall
    through to a fill that borrows from other years or other agencies. This is the Kenedy
    failure stated as a post-condition: with it in place that shape cannot ship silently.
    A lane that covered only part of the year is a different case -- one month of zeros is
    not a zero year -- and those rows are outside the claim. So is a twelve-month row the
    masked-gap detector has already ruled not credible -- and, for exactly the same reason, a
    twelve-month row a Stage-1 REVIEWER ruled not credible. The header claiming twelve months is
    the thing under review in both cases, so it cannot also be the premise: an adjudicated
    `misread_missing` or `token_reporting_flag` verdict says the twelve-month claim is false, and
    `stage1_adjudicated_kind` is how the estimate frame records that a reviewer said so.
    """
    if result.empty or current_rows.empty:
        return
    reported = current_rows.loc[
        pd.to_numeric(current_rows["preferred_months_reported"], errors="coerce").ge(12.0)
        & pd.to_numeric(current_rows["preferred_count"], errors="coerce").notna(),
        ["ori9", "offense", "preferred_source", "preferred_months_reported"],
    ]
    merged = result.merge(reported, on=["ori9", "offense"], how="inner")
    adjudicated = (
        merged["stage1_adjudicated_kind"].notna()
        if "stage1_adjudicated_kind" in merged.columns
        else pd.Series(False, index=merged.index)
    )
    violations = merged[
        ~merged["agency_estimate_source"].isin(LANE_GROUNDED_ESTIMATE_SOURCES)
        & ~merged["agency_estimate_source"].str.startswith("partial_uplift_sanity_cap_")
        & ~merged["masked_gap_reclassified"].fillna(False).astype(bool)
        & ~adjudicated
    ]
    if violations.empty:
        return
    raise ValueError(
        f"{len(violations)} agency-offense estimate(s) across "
        f"{violations['ori9'].nunique()} agencies were filled although their preferred "
        "lane reported the target year: "
        + str(
            violations[
                [
                    "ori9",
                    "offense",
                    "preferred_source",
                    "preferred_months_reported",
                    "reported_count_current",
                    "estimated_count",
                    "agency_estimate_source",
                ]
            ]
            .head(20)
            .to_dict(orient="records")
        )
    )


def _assert_estimates_finite_and_nonnegative(result: pd.DataFrame) -> None:
    """Fail closed on a non-finite or negative estimate.

    Cheap and true by construction; it exists so that an upstream change that starts
    emitting NaN or negative mass stops the build instead of raking it into the surface.
    """
    if result.empty:
        return
    for column in ["estimated_count", "agency_adjustment_count", "reported_count_current"]:
        values = pd.to_numeric(result[column], errors="coerce")
        bad = result[~np.isfinite(values) | values.lt(0.0)]
        if not bad.empty:
            raise ValueError(
                f"{len(bad)} agency estimate(s) have a non-finite or negative {column}: "
                + str(bad[["ori9", "offense", column, "agency_estimate_source"]].head(20).to_dict(orient="records"))
            )


def _check_fill_mass_against_baseline(
    result: pd.DataFrame,
    *,
    paths: object,
) -> dict[str, float]:
    """Operational regression alarm on the size of the filled population.

    Not a validity claim about any single estimate: it is a tripwire on the aggregate.
    Total non-observed estimate mass and the number of agencies carrying a fill are
    recorded per build and compared against the values in
    configs/agency_fill_mass_baseline.json; an increase beyond the recorded tolerance
    fails the build. Accepting a genuine increase means updating that file deliberately,
    the same contract the methodology gate's ratchet baseline uses.
    """
    filled = result[~result["agency_estimate_source"].eq(AGENCY_OBSERVED_ESTIMATE_SOURCE)]
    measured = {
        "non_observed_estimate_mass": float(
            pd.to_numeric(filled["estimated_count"], errors="coerce").fillna(0.0).sum()
        ),
        "filled_agency_count": float(filled["ori9"].nunique()),
    }
    baseline_path = paths.repo_root / "configs" / FILL_MASS_BASELINE_FILENAME
    if not baseline_path.exists():
        return measured
    baseline = json.loads(baseline_path.read_text())
    tolerance = float(baseline["tolerance_fraction"])
    breaches = {
        metric: (value, float(baseline[metric]))
        for metric, value in measured.items()
        if value > float(baseline[metric]) * (1.0 + tolerance)
    }
    if breaches:
        raise ValueError(
            f"agency fill mass regressed beyond the {tolerance:.0%} tolerance in "
            f"{baseline_path}: "
            + str({k: {"measured": v[0], "baseline": v[1]} for k, v in breaches.items()})
        )
    return measured


def build_agency_target_estimates_from_panel(
    agency_panel: pd.DataFrame,
    *,
    target_year: int,
    trend_fill_lookup: TrendFillLookup,
    masked_gap_flags: pd.DataFrame | None = None,
    stage1_adjudications: pd.DataFrame | None = None,
    reference_year_exclusions: pd.DataFrame | None = None,
    succession_ledger: pd.DataFrame | None = None,
    min_usable_years_for_trend: int = 3,
    max_year_gap_for_trend: int = 1,
    max_reference_age_years: int = FILL_MAX_REFERENCE_AGE_YEARS,
) -> pd.DataFrame:
    columns = AGENCY_TARGET_ESTIMATE_COLUMNS
    if agency_panel.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "ori9",
        "state_fips",
        "offense",
        "year",
        "preferred_count",
        "preferred_months_reported",
        "usable_as_observed",
        "current_row_is_true_partial",
    }
    if not required.issubset(agency_panel.columns):
        return pd.DataFrame(columns=columns)

    if masked_gap_flags is not None and not masked_gap_flags.empty:
        agency_panel = apply_masked_gap_reclassification(
            agency_panel,
            target_year=int(target_year),
            masked_gap_flags=masked_gap_flags,
        )
    # Rule first, adjudicated residue second: the detector's own flags are applied above, and the
    # reviewed cases the detector's thresholds could never reach land here.
    agency_panel, stage1_adjudication_counts = apply_stage1_adjudicated_usability(
        agency_panel,
        target_year=int(target_year),
        directives=stage1_adjudications,
    )

    work = agency_panel.copy()
    work["state_fips"] = work["state_fips"].astype("string").str.zfill(2)
    work["ori9"] = work["ori9"].astype("string")
    work["offense"] = work["offense"].astype("string")
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["preferred_count"] = pd.to_numeric(work["preferred_count"], errors="coerce")
    work["preferred_months_reported"] = pd.to_numeric(
        work["preferred_months_reported"], errors="coerce"
    )
    work["usable_as_observed"] = work["usable_as_observed"].fillna(False).astype(bool)
    work["current_row_is_true_partial"] = (
        work["current_row_is_true_partial"].fillna(False).astype(bool)
    )

    # Reference-year hygiene: a historical row that is itself a masked gap (per the same
    # detector, evaluated against ITS OWN preceding years -- see
    # build_reference_year_masked_gap_years) must not be trusted as a fill reference.
    # Excluded, not down-weighted: corrupt data is not evidence. This only affects rows
    # used as REFERENCES (year < target_year); the current target-year row's own
    # usable_as_observed / true-partial classification is untouched.
    exclusion_keys = _reference_year_exclusion_key_set(reference_year_exclusions)
    if exclusion_keys:
        excl_df = pd.DataFrame(list(exclusion_keys), columns=["ori9", "offense", "year"])
        excl_df["year"] = excl_df["year"].astype("Int64")
        excl_df["_reference_year_excluded"] = True
        work = work.merge(excl_df, on=["ori9", "offense", "year"], how="left")
        work["_reference_year_excluded"] = work["_reference_year_excluded"].fillna(False).astype(bool)
    else:
        work["_reference_year_excluded"] = False
    work["_usable_as_reference"] = work["usable_as_observed"] & ~work["_reference_year_excluded"]

    current_rows = work[work["year"].eq(int(target_year))].copy()
    current_rows["_current_count"] = (
        pd.to_numeric(current_rows["preferred_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    current_rows["_months"] = pd.to_numeric(
        current_rows["preferred_months_reported"], errors="coerce"
    )
    partial_raw = current_rows["_current_count"].copy()
    partial_mask = (
        current_rows["current_row_is_true_partial"]
        & current_rows["_months"].between(1.0, 11.999999, inclusive="both")
        & current_rows["_current_count"].gt(0.0)
    )
    partial_raw.loc[partial_mask] = np.maximum(
        current_rows.loc[partial_mask, "_current_count"],
        current_rows.loc[partial_mask, "_current_count"] * (12.0 / current_rows.loc[partial_mask, "_months"]),
    )
    current_rows["_partial_uplift_raw"] = partial_raw
    agency_raw_uplift_total = current_rows.groupby("ori9", dropna=False)["_partial_uplift_raw"].sum()

    panel_hist = work[
        work["_usable_as_reference"]
        & work["year"].between(2019, min(2023, int(target_year) - 1), inclusive="both")
    ].copy()
    panel_hist["_hist_count"] = (
        pd.to_numeric(panel_hist["preferred_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    agency_panel_max = (
        panel_hist.groupby(["ori9", "year"], dropna=False)["_hist_count"].sum().groupby("ori9").max()
        if not panel_hist.empty
        else pd.Series(dtype=float)
    )
    sanity_frame = pd.DataFrame(
        {
            "partial_uplift_agency_raw_total": agency_raw_uplift_total,
            "partial_uplift_panel_max_annual_total": agency_panel_max,
        }
    )
    sanity_frame["partial_uplift_sanity_cap_flag"] = (
        pd.to_numeric(sanity_frame["partial_uplift_panel_max_annual_total"], errors="coerce").gt(0.0)
        & pd.to_numeric(sanity_frame["partial_uplift_agency_raw_total"], errors="coerce").gt(
            3.0 * pd.to_numeric(sanity_frame["partial_uplift_panel_max_annual_total"], errors="coerce")
        )
    )
    sanity_frame = sanity_frame.fillna(
        {
            "partial_uplift_agency_raw_total": 0.0,
            "partial_uplift_panel_max_annual_total": np.nan,
            "partial_uplift_sanity_cap_flag": False,
        }
    )
    sanity_frame.index = sanity_frame.index.astype(str)

    def _historical_estimate(
        *,
        usable_hist: pd.DataFrame,
        state_fips: str,
        offense: str,
        current_count: float,
    ) -> tuple[float, str]:
        if len(usable_hist) >= 1:
            latest_usable_year = int(usable_hist["year"].max())
            if (int(target_year) - latest_usable_year) > int(max_reference_age_years):
                return (
                    float(max(0.0, current_count)),
                    AGENCY_STALE_HISTORY_ESTIMATE_SOURCE,
                )
        if len(usable_hist) >= int(min_usable_years_for_trend):
            last_year = int(usable_hist["year"].max())
            if (int(target_year) - last_year) <= int(max_year_gap_for_trend):
                est = _project_count_log_linear(
                    years=usable_hist["year"].to_numpy(dtype=float),
                    counts=usable_hist["preferred_count"].to_numpy(dtype=float),
                    target_year=int(target_year),
                )
                if np.isfinite(est):
                    return float(max(current_count, max(0.0, est))), "trend_log_linear"
        if len(usable_hist) >= 1:
            latest = usable_hist.sort_values("year").iloc[-1]
            latest_count_value = pd.to_numeric(latest["preferred_count"], errors="coerce")
            latest_count = float(latest_count_value) if pd.notna(latest_count_value) else 0.0
            latest_year = int(latest["year"])
            year_gap = int(target_year) - latest_year
            latest_audit = trend_fill_lookup.resolve(
                state_fips=state_fips,
                offense=offense,
                base_year=latest_year,
            )
            if year_gap == 1:
                return (
                    float(max(current_count, max(0.0, latest_count * float(latest_audit.ratio)))),
                    "carryforward_state_trend",
                )
            scaled_median, _median_audit = scaled_history_median(
                usable_hist.rename(columns={"preferred_count": "reported_count_preferred"}),
                state_fips=state_fips,
                offense=offense,
                trend_fill_lookup=trend_fill_lookup,
            )
            if scaled_median is not None and np.isfinite(float(scaled_median)):
                return (
                    float(max(current_count, max(0.0, float(scaled_median)))),
                    "hist_median_state_trend" if year_gap <= int(max_year_gap_for_trend) else "hist_median",
                )
        return float(max(0.0, current_count)), AGENCY_NO_HISTORY_ESTIMATE_SOURCE

    rows: list[dict[str, object]] = []
    for (ori9, offense), grp in work.groupby(["ori9", "offense"], sort=False):
        grp = grp.sort_values("year").copy()
        current = grp[grp["year"].eq(int(target_year))]
        current_count = (
            float(pd.to_numeric(current["preferred_count"].iloc[0], errors="coerce"))
            if not current.empty and pd.notna(current["preferred_count"].iloc[0])
            else 0.0
        )
        current_supported = False
        if not current.empty:
            current_supported = bool(
                current["usable_as_observed"].fillna(False).astype(bool).iloc[0]
                or current["current_row_is_true_partial"].fillna(False).astype(bool).iloc[0]
            )
        state_fips = str(grp["state_fips"].dropna().astype(str).str.zfill(2).iloc[0]) if grp["state_fips"].notna().any() else ""
        usable_hist = grp[(grp["_usable_as_reference"].eq(True)) & grp["year"].lt(int(target_year))].copy()
        excluded_ref_years = sorted(
            int(y)
            for y in grp.loc[
                grp["usable_as_observed"].eq(True)
                & grp["_reference_year_excluded"].eq(True)
                & grp["year"].lt(int(target_year)),
                "year",
            ].dropna().unique().tolist()
        )
        estimated = np.nan
        source = "missing"
        partial_uplift_raw_count = np.nan
        partial_uplift_cap_estimate = np.nan
        sanity = sanity_frame.loc[str(ori9)] if str(ori9) in sanity_frame.index else None
        agency_raw_total = (
            float(sanity["partial_uplift_agency_raw_total"])
            if sanity is not None and pd.notna(sanity["partial_uplift_agency_raw_total"])
            else 0.0
        )
        agency_panel_max_total = (
            float(sanity["partial_uplift_panel_max_annual_total"])
            if sanity is not None and pd.notna(sanity["partial_uplift_panel_max_annual_total"])
            else np.nan
        )
        agency_sanity_cap_flag = (
            bool(sanity["partial_uplift_sanity_cap_flag"])
            if sanity is not None and pd.notna(sanity["partial_uplift_sanity_cap_flag"])
            else False
        )
        review_flag = False
        review_reason = pd.NA
        current_usable = bool(
            current["usable_as_observed"].fillna(False).astype(bool).iloc[0]
        ) if not current.empty else False
        current_true_partial = bool(
            current["current_row_is_true_partial"].fillna(False).astype(bool).iloc[0]
        ) if not current.empty else False
        if not current.empty and current_usable:
            estimated = float(max(0.0, current_count))
            source = "observed"
        elif not current.empty and current_true_partial:
            months_value = pd.to_numeric(current["preferred_months_reported"].iloc[0], errors="coerce")
            months = float(months_value) if pd.notna(months_value) else 0.0
            if 0 < months < 12 and current_count <= 0:
                # A zero over the reported months annualises to zero. The offense is
                # reported, not missing, so it must not fall through to the ladder and
                # borrow a level from other years.
                estimated = 0.0
                partial_uplift_raw_count = 0.0
                source = "true_partial_month_ratio"
            elif 0 < months < 12 and current_count > 0:
                partial_uplift_raw_count = float(max(current_count, current_count * (12.0 / months)))
                if agency_sanity_cap_flag:
                    cap_estimate, cap_source = _historical_estimate(
                        usable_hist=usable_hist,
                        state_fips=state_fips,
                        offense=str(offense),
                        current_count=current_count,
                    )
                    partial_uplift_cap_estimate = float(max(0.0, cap_estimate))
                    estimated = partial_uplift_cap_estimate
                    source = f"partial_uplift_sanity_cap_{cap_source}"
                    review_flag = True
                    review_reason = "raw_partial_uplift_exceeds_3x_panel_max_annual"
                else:
                    estimated = partial_uplift_raw_count
                    source = "true_partial_month_ratio"
        if not np.isfinite(estimated):
            estimated, source = _historical_estimate(
                usable_hist=usable_hist,
                state_fips=state_fips,
                offense=str(offense),
                current_count=current_count,
            )
        if not np.isfinite(estimated):
            estimated = current_count
        rows.append(
            {
                "ori9": str(ori9),
                "state_fips": state_fips,
                "offense": str(offense),
                "reported_count_current": float(max(0.0, current_count)),
                "reported_count_current_supported": float(max(0.0, current_count)) if current_supported else 0.0,
                "estimated_count": float(max(0.0, estimated)),
                "agency_adjustment_count": float(max(0.0, estimated - current_count)),
                "agency_estimate_source": source,
                "partial_uplift_raw_count": (
                    float(max(0.0, partial_uplift_raw_count))
                    if np.isfinite(partial_uplift_raw_count)
                    else np.nan
                ),
                "partial_uplift_sanity_cap_flag": bool(review_flag),
                "partial_uplift_panel_max_annual_total": agency_panel_max_total,
                "partial_uplift_agency_raw_total": agency_raw_total,
                "partial_uplift_cap_estimate": (
                    float(max(0.0, partial_uplift_cap_estimate))
                    if np.isfinite(partial_uplift_cap_estimate)
                    else np.nan
                ),
                "agency_estimate_review_flag": bool(review_flag),
                "agency_estimate_review_reason": review_reason,
                "reference_years_excluded": (
                    ",".join(str(y) for y in excluded_ref_years) if excluded_ref_years else pd.NA
                ),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    result = _drop_silent_agency_estimates(result)
    if succession_ledger is not None:
        result = _drop_superseded_ori_estimates(result, succession_ledger=succession_ledger)
    _assert_estimates_finite_and_nonnegative(result)
    result["stage1_adjudicated_kind"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["stage1_adjudicated_case"] = pd.Series(pd.NA, index=result.index, dtype="string")
    adjudicated_kinds = stage1_adjudicated_estimate_kinds(stage1_adjudications)
    if not adjudicated_kinds.empty:
        result = result.merge(
            adjudicated_kinds.rename(
                columns={
                    "stage1_adjudicated_kind": "_adj_kind",
                    "stage1_adjudicated_case": "_adj_case",
                }
            ),
            left_on=result["ori9"].astype("string").str.upper(),
            right_on="ori9",
            how="left",
            suffixes=("", "_adj"),
        )
        result["stage1_adjudicated_kind"] = result["_adj_kind"].astype("string")
        result["stage1_adjudicated_case"] = result["_adj_case"].astype("string")
        result = result.drop(columns=[c for c in ("_adj_kind", "_adj_case", "ori9_adj", "key_0")
                                      if c in result.columns])
    result["masked_gap_reclassified"] = False
    result["masked_gap_kind"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if masked_gap_flags is not None and not masked_gap_flags.empty:
        flag_kinds = masked_gap_flags[["ori9", "offense", "masked_gap_kind"]].rename(
            columns={"masked_gap_kind": "_flag_kind"}
        )
        flag_kinds["ori9"] = flag_kinds["ori9"].astype("string")
        flag_kinds["offense"] = flag_kinds["offense"].astype("string")
        result = result.merge(flag_kinds, on=["ori9", "offense"], how="left")
        result["masked_gap_reclassified"] = result["_flag_kind"].notna()
        result["masked_gap_kind"] = result["_flag_kind"].astype("string")
        result = result.drop(columns=["_flag_kind"])
    _assert_no_fill_where_the_chosen_lane_reported(result, current_rows=current_rows)
    out = result[columns]
    out.attrs["stage1_adjudications"] = stage1_adjudication_counts
    return out


def build_agency_allocation_target_estimates(
    *,
    paths: object,
    year: int,
    agency_panel: pd.DataFrame | None = None,
    trend_fill_lookup: TrendFillLookup | None = None,
    masked_gap_flags: pd.DataFrame | None = None,
    stage1_adjudications: pd.DataFrame | None = None,
    reference_year_exclusions: pd.DataFrame | None = None,
    succession_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Single source of truth for per-agency target-year estimates (observed vs.
    trend/hist-median/peer-fallback fill), shared by allocation.py's county-remainder
    split and by the jurisdiction-controls remainder-pool rollup so both consumers see
    the exact same per-agency fill amounts. Masked-gap flags (build_masked_gap_flags)
    are computed here by default so every consumer applies the same reclassification.
    Reference-year hygiene (build_reference_year_masked_gap_years) is likewise computed
    here by default: historical years that are themselves masked gaps (per the same
    detector, evaluated against their own preceding years) are excluded as fill
    references so every consumer sees the same hygienic reference set. The ORI
    succession ledger is computed here for the same reason: identity resolution has to
    reach every consumer or the retired ORI is resurrected by whichever one missed it.
    """
    if agency_panel is None:
        agency_panel = build_agency_trend_fill_panel(
            paths=paths,
            year_start=2018,
            year_end=int(year),
        )
    if trend_fill_lookup is None:
        trend_fill_lookup = (
            build_trend_fill_lookup(
                agency_panel,
                entity_col="ori9",
                count_col="preferred_count",
                target_year=int(year),
            )
            if not agency_panel.empty
            else TrendFillLookup(state_map={}, national_map={})
        )
    if masked_gap_flags is None:
        masked_gap_flags = build_masked_gap_flags(
            paths,
            target_year=int(year),
            agency_panel=agency_panel,
        )
    if stage1_adjudications is None:
        stage1_adjudications = build_usability_directives(paths, target_year=int(year))
    if reference_year_exclusions is None:
        candidate_years = (
            sorted(
                int(y)
                for y in agency_panel["year"].dropna().unique().tolist()
                if int(y) < int(year)
            )
            if not agency_panel.empty
            else []
        )
        reference_year_exclusions = build_reference_year_masked_gap_years(
            paths,
            agency_panel=agency_panel,
            candidate_years=candidate_years,
        )
    succession_summary: dict[str, object] = {}
    if succession_ledger is None:
        succession_ledger, succession_summary = resolve_ori_succession(
            paths=paths,
            agency_panel=agency_panel,
            agency_jurisdiction_crosswalk=load_agency_jurisdiction_crosswalk(paths),
            target_year=int(year),
            max_reference_age_years=FILL_MAX_REFERENCE_AGE_YEARS,
        )
    estimates = build_agency_target_estimates_from_panel(
        agency_panel,
        target_year=int(year),
        trend_fill_lookup=trend_fill_lookup,
        masked_gap_flags=masked_gap_flags,
        stage1_adjudications=stage1_adjudications,
        reference_year_exclusions=reference_year_exclusions,
        succession_ledger=succession_ledger,
    )
    _check_fill_mass_against_baseline(estimates, paths=paths)
    counts = dict(estimates.attrs.get("stage1_adjudications", {}))
    if stage1_adjudications is not None and not stage1_adjudications.empty:
        assert_registry_bit(
            counts,
            what="agency target estimates: Stage-1 adjudicated usability",
            expected_key="stage1_adjudicated_agency_years_matched",
        )
    counts.update(succession_summary)
    estimates.attrs["stage1_adjudications"] = counts
    print(
        "build_agency_allocation_target_estimates: Stage-1 adjudications -> "
        f"{counts.get('stage1_adjudicated_rows_to_ladder', 0)} rows to the fill ladder, "
        f"{counts.get('stage1_adjudicated_rows_to_partial', 0)} rows to true-partial uplift, "
        f"{counts.get('succession_adjudicated_oris_added', 0)} adjudicated successions",
        flush=True,
    )
    return estimates
