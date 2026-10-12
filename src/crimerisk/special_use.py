"""Typed special-use taxonomy and display metadata.

The deployed surface suppresses every rate and index in a Census 98-series tract, unconditionally.
`REVIEW_SOL_NEUTRAL.md` sec.6 rules that out: 98xxxx is a code the Census Bureau reserves for
special land-use areas, and a code is a **warning flag, not a measurement**. The consult's own
sample carries a 98-series tract with 4,610 residents and another with 3,695 residents and 31,989
daytime workers. Blanket suppression throws away every one of those defensible rates; System A's
opposite error -- publishing a zero where the rate is undefined -- gives an employment district
the visual meaning "no crime". This lane replaces both with a typed taxonomy.

The taxonomy classifies cells and supplies display metadata. Publication is
denominator-driven: a type never suppresses an otherwise valid rate. This keeps
parks, campuses, institutions, employment districts, and other special-use
areas on the same numerical contract as every other cell while allowing the map
to mark them separately.

Four rules are load-bearing and enforced rather than trusted:

* **Never encode an undefined rate as zero.** A denominator that fails its floor publishes NULL.
  A resident rate is null wherever the resident population is below the shared floor.
* **The type is descriptive.** A candidate cell with real households, real residents and no
  institutional population is `ordinary` even when its tract carries a 98-series code. A cell
  with no code can still receive a special-use type.
* **Classification never closes a rate.** Missing or ambiguous classification
  inputs produce `unknown_special_use`; the offense-specific denominator and
  its publication floor still decide whether a rate exists.
* **The taxonomy is generic.** Its inputs are the 98-series code, ACS household/housing counts and
  the derived group-quarters population, LODES workplace jobs and their education share, the NCES
  postsecondary anchor, NLCD land cover, and LandScan day population. No city-specific code, no
  hand-adjudicated cell list, no per-cell tuning of any kind.

Deviation from the consult's table, recorded here because it changes a NAME and nothing else. The
consult's first row is "airport or major transit facility". On generic public data the property
that actually drives its display policy is an unmeasured transient daytime population, and the
rule that detects it (LandScan day population far above what residents plus jobs explain) is not
transport-specific: the class it selects has a median LODES transportation-and-warehousing job
share of 0.000, because it is also stadiums, venues, casinos, convention centres and downtown
visitor cores. Naming that class `airport_transit` would be a label the classifier cannot support,
so it is named `transient_destination` and carries the consult's low-confidence transient flag.
Its display policy is the consult's, unchanged.

No AGS value enters anywhere: every input is Census/ACS, LODES, NCES, NLCD or LandScan, and no
threshold was selected against an AGS cell, an AGS distribution or an AGS-derived comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- naming and versioning ---------------------------------------------------------------

SPECIAL_USE_TAXONOMY_VERSION = "special_use_taxonomy_v2"

TYPE_ORDINARY = "ordinary"
TYPE_CAMPUS_INSTITUTION = "campus_institution"
TYPE_PRISON_INSTITUTIONAL = "institutional_facility"
TYPE_GROUP_QUARTERS_OTHER = "group_quarters_other"
TYPE_PARK_OPEN_SPACE = "park_open_space"
TYPE_TRANSIENT_DESTINATION = "transient_destination"
TYPE_INDUSTRIAL_EMPLOYMENT = "industrial_employment"
TYPE_UNKNOWN_SPECIAL_USE = "unknown_special_use"

SPECIAL_USE_TYPES: tuple[str, ...] = (
    TYPE_ORDINARY,
    TYPE_CAMPUS_INSTITUTION,
    TYPE_PRISON_INSTITUTIONAL,
    TYPE_GROUP_QUARTERS_OTHER,
    TYPE_PARK_OPEN_SPACE,
    TYPE_TRANSIENT_DESTINATION,
    TYPE_INDUSTRIAL_EMPLOYMENT,
    TYPE_UNKNOWN_SPECIAL_USE,
)

# What a viewer is shown for each type. One string per type, so a published cell carries its own
# display contract and nothing downstream has to re-derive it from six booleans.
POLICY_ORDINARY = "ordinary"
POLICY_EXPOSURE_AND_RESIDENT_RATE = "exposure_and_resident_rate"
POLICY_EXPOSURE_RATE_ONLY = "exposure_rate_only"
POLICY_COUNT_AND_DENSITY_ONLY = "count_and_density_only"
POLICY_COUNT_AND_DENSITY_ONLY_FAIL_CLOSED = "count_and_density_only_fail_closed"

DISPLAY_POLICY_BY_TYPE: dict[str, str] = {
    TYPE_ORDINARY: POLICY_ORDINARY,
    TYPE_CAMPUS_INSTITUTION: POLICY_EXPOSURE_AND_RESIDENT_RATE,
    TYPE_INDUSTRIAL_EMPLOYMENT: POLICY_EXPOSURE_AND_RESIDENT_RATE,
    TYPE_TRANSIENT_DESTINATION: POLICY_EXPOSURE_AND_RESIDENT_RATE,
    TYPE_PARK_OPEN_SPACE: POLICY_EXPOSURE_AND_RESIDENT_RATE,
    TYPE_PRISON_INSTITUTIONAL: POLICY_EXPOSURE_AND_RESIDENT_RATE,
    TYPE_GROUP_QUARTERS_OTHER: POLICY_EXPOSURE_AND_RESIDENT_RATE,
    TYPE_UNKNOWN_SPECIAL_USE: POLICY_EXPOSURE_AND_RESIDENT_RATE,
}

# Type is descriptive metadata, never a publication gate. The selected denominator and its
# support floor decide whether a rate exists.
PRIMARY_RATE_ALLOWED_BY_TYPE: dict[str, bool | None] = {
    TYPE_ORDINARY: True,
    TYPE_CAMPUS_INSTITUTION: True,
    TYPE_INDUSTRIAL_EMPLOYMENT: True,
    TYPE_TRANSIENT_DESTINATION: True,
    TYPE_PARK_OPEN_SPACE: True,
    TYPE_PRISON_INSTITUTIONAL: True,
    TYPE_GROUP_QUARTERS_OTHER: True,
    TYPE_UNKNOWN_SPECIAL_USE: True,
}
RESIDENT_RATE_ALLOWED_BY_TYPE: dict[str, bool | None] = {
    TYPE_ORDINARY: True,
    TYPE_CAMPUS_INSTITUTION: True,
    TYPE_INDUSTRIAL_EMPLOYMENT: True,
    TYPE_TRANSIENT_DESTINATION: True,
    TYPE_PARK_OPEN_SPACE: True,
    TYPE_PRISON_INSTITUTIONAL: True,
    TYPE_GROUP_QUARTERS_OTHER: True,
    TYPE_UNKNOWN_SPECIAL_USE: True,
}

# The consult's low-confidence transient flag. Carried on the class it describes; the class does
# not publish a per-person rate, so the flag annotates a type, never a published number.
TRANSIENT_CONFIDENCE_TYPES: frozenset[str] = frozenset({TYPE_TRANSIENT_DESTINATION})

# No type forces a coarser display. Denominator support controls geography.
COARSER_RECOMMENDATION_TYPES: frozenset[str] = frozenset()


# --- thresholds ---------------------------------------------------------------------------
#
# Every threshold below is either (a) a constant this pipeline already ships, reused rather than
# re-invented, or (b) a majority rule, or (c) a percentile of the measured national distribution.
# None was selected by looking at the resulting map. The measurements are recorded in
# analysis_scratch/final_phase/SPECIAL_USE_TAXONOMY_CONTRACT.md and regenerate from
# scripts/diagnostics/special_use_threshold_evidence.py.

# = allocation.NON_RESIDENTIAL_HOUSEHOLD_FLOOR. The consult calls it a provisional floor rather
# than a universal reliability boundary; it is reused here so the taxonomy adds no new number.
SPECIAL_USE_HOUSEHOLD_FLOOR = 10.0
# = allocation.PERSON_EXPOSURE_DENOMINATOR_FLOOR. Below ~50 measured people the denominator stops
# measuring a stable population at risk, so it is also the floor at which an employment base
# counts as a measurable workforce.
SPECIAL_USE_EMPLOYMENT_FLOOR = 50.0
# Majority rule: more than half of the cell's residents live in group quarters. Measured over the
# 2,934 national candidate cells the group-quarters share is sharply bimodal -- 126 below 0.10 and
# 740 at or above 0.90, with 1 cell in [0.40, 0.50) -- so the majority line sits in an empty
# region rather than on a mode.
GROUP_QUARTERS_POPULATION_SHARE_MIN = 0.50
# Majority rule: more than half of the cell's workplace jobs are educational services (LODES
# CNS15). Paired with the NCES anchor, never used alone.
EDUCATION_JOB_SHARE_MIN = 0.50
# One NCES postsecondary institution physically inside the cell.
POSTSECONDARY_ANCHOR_MIN = 1.0
# Three quarters of classified land cover is developed open space (NLCD 21) or an undeveloped
# class. Three quarters, not a majority: a cell that is half built is not open space.
OPEN_NATURAL_LAND_COVER_SHARE_MIN = 0.75
# P99 of landscan_day_pop / (residents + workplace jobs) over every CONUS+DC block group with at
# least 50 jobs (measured 2.529 on the 2024 edition, n=197,506). A candidate above it holds a
# daytime population the two measured surfaces cannot account for.
UNEXPLAINED_DAYTIME_PRESENCE_RATIO_MIN = 2.5

# NLCD land-cover classes that count as open space or undeveloped: developed open space (21),
# open water (11), barren (31), forest (41/42/43), shrub (52), herbaceous (71), pasture/crops
# (81/82) and wetlands (90/95). Developed low/medium/high (22/23/24) are excluded by construction.
NLCD_OPEN_NATURAL_CLASSES: tuple[int, ...] = (11, 21, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95)
NLCD_VALID_LAND_COVER_CLASSES: tuple[int, ...] = (*NLCD_OPEN_NATURAL_CLASSES, 22, 23, 24)

LANDSCAN_DAY_POP_COLUMN = "landscan_day_pop"


# --- the frame contract -------------------------------------------------------------------

# Classification inputs. Every one is EXTENSIVE (a count of people, jobs, institutions or 30 m
# pixels), which is what lets the tract surface classify itself from the sum of its block groups
# through the same function, with no second code path.
SPECIAL_USE_FEATURE_COLUMNS: tuple[str, ...] = (
    "special_use_acs_population",
    "special_use_household_population",
    "special_use_jobs_total",
    "special_use_jobs_education",
    "special_use_postsecondary_count",
    "special_use_open_natural_pixels",
    "special_use_classified_pixels",
    "special_use_gq_total_2020",
    "special_use_gq_institutional_2020",
    "special_use_gq_correctional_2020",
    "special_use_gq_juvenile_2020",
    "special_use_gq_nursing_2020",
    "special_use_gq_college_2020",
    "special_use_gq_military_2020",
    "special_use_gq_other_institutional_2020",
    "special_use_gq_other_noninstitutional_2020",
)

SPECIAL_USE_TYPE_COLUMN = "special_use_type"
SPECIAL_USE_EVIDENCE_COLUMN = "special_use_type_evidence"
SPECIAL_USE_CANDIDATE_COLUMN = "special_use_candidate_flag"
SPECIAL_USE_POLICY_COLUMN = "special_use_display_policy"
SPECIAL_USE_PRIMARY_ALLOWED_COLUMN = "special_use_primary_rate_allowed"
SPECIAL_USE_RESIDENT_ALLOWED_COLUMN = "special_use_resident_rate_allowed"
SPECIAL_USE_TRANSIENT_FLAG_COLUMN = "special_use_transient_confidence_flag"

SPECIAL_USE_OUTPUT_COLUMNS: tuple[str, ...] = (
    SPECIAL_USE_TYPE_COLUMN,
    SPECIAL_USE_EVIDENCE_COLUMN,
    SPECIAL_USE_CANDIDATE_COLUMN,
    SPECIAL_USE_POLICY_COLUMN,
    SPECIAL_USE_PRIMARY_ALLOWED_COLUMN,
    SPECIAL_USE_RESIDENT_ALLOWED_COLUMN,
    SPECIAL_USE_TRANSIENT_FLAG_COLUMN,
)

SPECIAL_USE_PUBLISHED_COLUMNS: tuple[str, ...] = (
    *SPECIAL_USE_FEATURE_COLUMNS,
    *SPECIAL_USE_OUTPUT_COLUMNS,
)

# What each type's assignment rested on, published beside the type so a reviewer never has to
# guess which clause fired.
EVIDENCE_NOT_CANDIDATE = "not_special_use_candidate"
EVIDENCE_RESIDENTIAL_SUPPORT = "residential_support_adequate"
EVIDENCE_CAMPUS = "postsecondary_anchor_with_group_quarters_or_education_employment"
EVIDENCE_PRISON = "census_institutional_group_quarters_majority"
EVIDENCE_GROUP_QUARTERS_OTHER = "census_other_group_quarters_majority"
EVIDENCE_PARK = "open_space_or_undeveloped_land_cover_majority"
EVIDENCE_TRANSIENT = "daytime_presence_exceeds_residents_plus_jobs"
EVIDENCE_INDUSTRIAL = "employment_base_explains_daytime_presence"
EVIDENCE_UNKNOWN = "no_resolving_evidence"


def _numeric(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).clip(lower=0.0)


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.Series(frame[column], index=frame.index).fillna(False).astype(bool)


def _share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator/denominator in [0, 1], NULL where the denominator is zero.

    Null, not zero: a cell with no classified land cover has an UNKNOWN open-space share, and a
    zero there would silently mean "fully developed" and open publication.
    """
    denom = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    out = pd.Series(np.nan, index=denom.index, dtype=float)
    positive = denom.gt(0.0).to_numpy(dtype=bool)
    out.iloc[positive] = (
        num.to_numpy(dtype=float)[positive] / denom.to_numpy(dtype=float)[positive]
    )
    return out.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.0)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Unbounded numerator/denominator, NULL where the denominator is zero."""
    denom = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    out = pd.Series(np.nan, index=denom.index, dtype=float)
    positive = denom.gt(0.0).to_numpy(dtype=bool)
    out.iloc[positive] = (
        num.to_numpy(dtype=float)[positive] / denom.to_numpy(dtype=float)[positive]
    )
    return out.replace([np.inf, -np.inf], np.nan)


# --- classification inputs ----------------------------------------------------------------


def special_use_features(bg: pd.DataFrame) -> pd.DataFrame:
    """Derive `SPECIAL_USE_FEATURE_COLUMNS` from the wide block-group covariate frame.

    Reads nothing from disk: every source column is already on the frame that
    `model_surface.build_bg_feature_frame` produces, so this lane adds no new ingestion and no new
    dependency stamp. Missing sources yield zeros and can push a cell toward
    `unknown_special_use`; they do not suppress a denominator-valid rate.
    """
    out = pd.DataFrame(index=bg.index)
    households = _numeric(bg, "households_total")
    average_household_size = _numeric(bg, "avg_household_size_total")
    jobs = _numeric(bg, "jobs_wac")
    education_share = _numeric(bg, "lodes_cns15_share").clip(upper=1.0)
    out["special_use_acs_population"] = _numeric(bg, "total_population")
    # ACS publishes household size and household count, not household population; their product is
    # the household population, and total minus household is the group-quarters population. Both
    # legs are ACS so the difference is internally consistent, which the scaled `population_2024`
    # estimate would not be.
    out["special_use_household_population"] = (average_household_size * households).clip(lower=0.0)
    out["special_use_jobs_total"] = jobs
    out["special_use_jobs_education"] = (education_share * jobs).clip(lower=0.0)
    out["special_use_postsecondary_count"] = _numeric(bg, "postsecondary_count")
    open_natural = pd.Series(0.0, index=bg.index, dtype=float)
    for land_cover_class in NLCD_OPEN_NATURAL_CLASSES:
        open_natural = open_natural + _numeric(bg, f"nlcd_count_{land_cover_class}")
    out["special_use_open_natural_pixels"] = open_natural
    # The compact frozen structural-feature cache carries every categorical NLCD count but not
    # `nlcd_valid_pixel_count`. Reconstruct the denominator from the exhaustive class counts when
    # the prepared NLCD source is absent; otherwise the cache silently turns every land-cover
    # share into NaN and disables the park/open-space branch of the taxonomy.
    reconstructed_classified = pd.Series(0.0, index=bg.index, dtype=float)
    for land_cover_class in NLCD_VALID_LAND_COVER_CLASSES:
        reconstructed_classified = reconstructed_classified + _numeric(
            bg, f"nlcd_count_{land_cover_class}"
        )
    classified = _numeric(bg, "nlcd_valid_pixel_count")
    out["special_use_classified_pixels"] = classified.where(
        classified.gt(0.0), reconstructed_classified
    )
    for suffix in (
        "total",
        "institutional",
        "correctional",
        "juvenile",
        "nursing",
        "college",
        "military",
        "other_institutional",
        "other_noninstitutional",
    ):
        out[f"special_use_gq_{suffix}_2020"] = _numeric(bg, f"census2020_gq_{suffix}")
    return out


def attach_special_use_features(bg: pd.DataFrame) -> pd.DataFrame:
    out = bg.copy()
    features = special_use_features(bg)
    for column in SPECIAL_USE_FEATURE_COLUMNS:
        out[column] = features[column]
    return out


# --- the cascade --------------------------------------------------------------------------


def special_use_diagnostics(
    frame: pd.DataFrame,
    *,
    population_col: str,
) -> pd.DataFrame:
    """The four intensive quantities the cascade reads, derived from the extensive inputs.

    Kept as a function rather than as published columns: they are exactly recomputable from the
    seven feature columns the surface already carries, and a stored copy of a derived quantity is
    a second source of truth waiting to drift.
    """
    population = _numeric(frame, population_col)
    jobs = _numeric(frame, "special_use_jobs_total")
    acs_population = _numeric(frame, "special_use_acs_population")
    household_population = _numeric(frame, "special_use_household_population")
    out = pd.DataFrame(index=frame.index)
    out["group_quarters_share"] = _share(
        (acs_population - household_population).clip(lower=0.0), acs_population
    )
    gq_total = _numeric(frame, "special_use_gq_total_2020")
    out["institutional_share_of_group_quarters"] = _share(
        _numeric(frame, "special_use_gq_institutional_2020"), gq_total
    )
    out["college_share_of_group_quarters"] = _share(
        _numeric(frame, "special_use_gq_college_2020"), gq_total
    )
    out["education_job_share"] = _share(_numeric(frame, "special_use_jobs_education"), jobs)
    out["open_natural_share"] = _share(
        _numeric(frame, "special_use_open_natural_pixels"),
        _numeric(frame, "special_use_classified_pixels"),
    )
    out["daytime_presence_ratio"] = _ratio(
        _numeric(frame, LANDSCAN_DAY_POP_COLUMN), population + jobs
    )
    return out


def classify_special_use(
    frame: pd.DataFrame,
    *,
    population_col: str,
    special_use_tract_flag: pd.Series | None = None,
) -> pd.DataFrame:
    """Assign a special-use type and its display policy to every row.

    Pure function of frame columns, so the block-group surface and the tract surface run the same
    cascade over the same extensive quantities and the tract answer is the tract's own, not a
    vote over its block groups.
    """
    index = frame.index
    flag = (
        pd.Series(special_use_tract_flag, index=index).fillna(False).astype(bool)
        if special_use_tract_flag is not None
        else _boolean(frame, "special_use_tract_flag")
    )
    population = _numeric(frame, population_col)
    households = _numeric(frame, "households_total")
    jobs = _numeric(frame, "special_use_jobs_total")
    postsecondary = _numeric(frame, "special_use_postsecondary_count")
    diagnostics = special_use_diagnostics(frame, population_col=population_col)

    group_quarters_dominant = diagnostics["group_quarters_share"].fillna(0.0).ge(
        float(GROUP_QUARTERS_POPULATION_SHARE_MIN)
    )
    institutional = group_quarters_dominant & diagnostics[
        "institutional_share_of_group_quarters"
    ].fillna(0.0).ge(float(GROUP_QUARTERS_POPULATION_SHARE_MIN))
    college_group_quarters = group_quarters_dominant & diagnostics[
        "college_share_of_group_quarters"
    ].fillna(0.0).ge(float(GROUP_QUARTERS_POPULATION_SHARE_MIN))
    other_group_quarters = group_quarters_dominant & ~institutional & ~college_group_quarters
    education_dominant = diagnostics["education_job_share"].fillna(0.0).ge(
        float(EDUCATION_JOB_SHARE_MIN)
    )
    open_natural_dominant = diagnostics["open_natural_share"].fillna(0.0).ge(
        float(OPEN_NATURAL_LAND_COVER_SHARE_MIN)
    )
    unexplained_daytime = diagnostics["daytime_presence_ratio"].fillna(0.0).ge(
        float(UNEXPLAINED_DAYTIME_PRESENCE_RATIO_MIN)
    )
    measurable_workforce = jobs.ge(float(SPECIAL_USE_EMPLOYMENT_FLOOR))
    residential_support = households.ge(float(SPECIAL_USE_HOUSEHOLD_FLOOR)) & population.gt(0.0)

    # A cell enters the taxonomy if the Census flagged its tract, if nobody lives there, or if it
    # has no household base. Everything else is an ordinary neighbourhood and is not asked.
    candidate = flag | population.le(0.0) | households.lt(float(SPECIAL_USE_HOUSEHOLD_FLOOR))
    # The release clause: a flagged cell with real households, real residents and no institutional
    # population is an ordinary neighbourhood whose tract merely carries a code. This is the clause
    # that answers "98-series status is not equivalent to zero or unusable exposure".
    released = candidate & residential_support & ~group_quarters_dominant
    unresolved = candidate & ~released

    special_use_type = pd.Series(TYPE_ORDINARY, index=index, dtype=object)
    evidence = pd.Series(EVIDENCE_NOT_CANDIDATE, index=index, dtype=object)
    evidence.loc[released] = EVIDENCE_RESIDENTIAL_SUPPORT

    cascade: tuple[tuple[str, pd.Series, str], ...] = (
        # A campus is an NCES postsecondary institution physically present PLUS the population or
        # the payroll that makes it a campus rather than a storefront school.
        (
            TYPE_CAMPUS_INSTITUTION,
            college_group_quarters
            | (
                postsecondary.ge(float(POSTSECONDARY_ANCHOR_MIN))
                & (group_quarters_dominant | education_dominant)
            ),
            EVIDENCE_CAMPUS,
        ),
        # Actual institutional group-quarters residents, not all group quarters.
        (TYPE_PRISON_INSTITUTIONAL, institutional, EVIDENCE_PRISON),
        (TYPE_GROUP_QUARTERS_OTHER, other_group_quarters, EVIDENCE_GROUP_QUARTERS_OTHER),
        # Land cover decides before employment does: what the ground IS is the more structural
        # fact, and a cell that is three-quarters open space is not an employment district even
        # when a depot sits on the remaining quarter.
        (TYPE_PARK_OPEN_SPACE, open_natural_dominant, EVIDENCE_PARK),
        # A measurable workforce whose cell nonetheless holds a daytime population neither its
        # residents nor its jobs explain: the population at risk is transient and unmeasured.
        (
            TYPE_TRANSIENT_DESTINATION,
            measurable_workforce & unexplained_daytime,
            EVIDENCE_TRANSIENT,
        ),
        # A measurable workforce whose daytime population IS explained: LODES and LandScan are
        # measuring the people who are there, so a person-exposure rate is a real rate.
        (TYPE_INDUSTRIAL_EMPLOYMENT, measurable_workforce, EVIDENCE_INDUSTRIAL),
    )
    assigned = pd.Series(False, index=index)
    for type_name, rule, evidence_name in cascade:
        hit = unresolved & ~assigned & pd.Series(rule, index=index).fillna(False).astype(bool)
        special_use_type.loc[hit] = type_name
        evidence.loc[hit] = evidence_name
        assigned = assigned | hit
    fail_closed = unresolved & ~assigned
    special_use_type.loc[fail_closed] = TYPE_UNKNOWN_SPECIAL_USE
    evidence.loc[fail_closed] = EVIDENCE_UNKNOWN

    out = pd.DataFrame(index=index)
    out[SPECIAL_USE_TYPE_COLUMN] = special_use_type.astype("string")
    out[SPECIAL_USE_EVIDENCE_COLUMN] = evidence.astype("string")
    out[SPECIAL_USE_CANDIDATE_COLUMN] = candidate
    out[SPECIAL_USE_POLICY_COLUMN] = (
        special_use_type.map(DISPLAY_POLICY_BY_TYPE).astype("string")
    )
    legacy_residential_rule = households.ge(float(SPECIAL_USE_HOUSEHOLD_FLOOR))
    out[SPECIAL_USE_PRIMARY_ALLOWED_COLUMN] = _arm_allowed(
        special_use_type,
        allowed_by_type=PRIMARY_RATE_ALLOWED_BY_TYPE,
        legacy_rule=legacy_residential_rule,
    )
    out[SPECIAL_USE_RESIDENT_ALLOWED_COLUMN] = _arm_allowed(
        special_use_type,
        allowed_by_type=RESIDENT_RATE_ALLOWED_BY_TYPE,
        legacy_rule=legacy_residential_rule,
    )
    out[SPECIAL_USE_TRANSIENT_FLAG_COLUMN] = special_use_type.isin(TRANSIENT_CONFIDENCE_TYPES)
    return out


def _arm_allowed(
    special_use_type: pd.Series,
    *,
    allowed_by_type: dict[str, bool | None],
    legacy_rule: pd.Series,
) -> pd.Series:
    allowed = pd.Series(False, index=special_use_type.index)
    for type_name, decision in allowed_by_type.items():
        rows = special_use_type.eq(type_name)
        if decision is None:
            allowed.loc[rows] = legacy_rule.loc[rows]
        else:
            allowed.loc[rows] = bool(decision)
    return allowed.astype(bool)


def apply_special_use_taxonomy(
    frame: pd.DataFrame,
    *,
    population_col: str,
    special_use_tract_flag: pd.Series | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    typed = classify_special_use(
        out, population_col=population_col, special_use_tract_flag=special_use_tract_flag
    )
    for column in SPECIAL_USE_OUTPUT_COLUMNS:
        out[column] = typed[column]
    return out


# --- publication gates read by the publication path ---------------------------------------


def primary_rate_allowed(frame: pd.DataFrame, *, residential_eligible: pd.Series) -> pd.Series:
    """The typed gate on the opportunity/person-exposure arm, or the legacy rule when absent."""
    if SPECIAL_USE_PRIMARY_ALLOWED_COLUMN not in frame.columns:
        return pd.Series(residential_eligible, index=frame.index).fillna(False).astype(bool)
    return _boolean(frame, SPECIAL_USE_PRIMARY_ALLOWED_COLUMN)


def resident_rate_allowed(frame: pd.DataFrame, *, residential_eligible: pd.Series) -> pd.Series:
    """The typed gate on the resident arm, or the legacy rule when absent."""
    if SPECIAL_USE_RESIDENT_ALLOWED_COLUMN not in frame.columns:
        return pd.Series(residential_eligible, index=frame.index).fillna(False).astype(bool)
    return _boolean(frame, SPECIAL_USE_RESIDENT_ALLOWED_COLUMN)


def typed_suppressed(frame: pd.DataFrame, *, allowed: pd.Series) -> pd.Series:
    """Rows the TYPE closed, as distinct from rows a denominator floor closed.

    Only these rows are entitled to the `special_use` display status once the taxonomy is on; a
    cell below an exposure floor keeps reporting the floor, which is a different fact.
    """
    if SPECIAL_USE_TYPE_COLUMN not in frame.columns:
        return pd.Series(False, index=frame.index)
    non_ordinary = frame[SPECIAL_USE_TYPE_COLUMN].astype("string").ne(TYPE_ORDINARY).fillna(False)
    return non_ordinary & ~pd.Series(allowed, index=frame.index).fillna(False).astype(bool)


def coarser_recommendation_rows(frame: pd.DataFrame, *, publishable: pd.Series) -> pd.Series:
    """Typed cells whose rate is closed: recommend the coarser geography, not "not published"."""
    if SPECIAL_USE_TYPE_COLUMN not in frame.columns:
        return pd.Series(False, index=frame.index)
    typed = (
        frame[SPECIAL_USE_TYPE_COLUMN]
        .astype("string")
        .isin(sorted(COARSER_RECOMMENDATION_TYPES))
        .fillna(False)
    )
    return typed & ~pd.Series(publishable, index=frame.index).fillna(False).astype(bool)


# --- reporting ----------------------------------------------------------------------------


def summarize_special_use_taxonomy(frame: pd.DataFrame) -> dict[str, object]:
    if SPECIAL_USE_TYPE_COLUMN not in frame.columns:
        return {"enabled": False}
    types = frame[SPECIAL_USE_TYPE_COLUMN].astype("string")
    counts = {name: int(types.eq(name).sum()) for name in SPECIAL_USE_TYPES}
    return {
        "enabled": True,
        "version": SPECIAL_USE_TAXONOMY_VERSION,
        "rows": int(len(frame)),
        "candidate_rows": int(_boolean(frame, SPECIAL_USE_CANDIDATE_COLUMN).sum()),
        "type_counts": counts,
        "primary_rate_allowed_rows": int(
            _boolean(frame, SPECIAL_USE_PRIMARY_ALLOWED_COLUMN).sum()
        ),
        "resident_rate_allowed_rows": int(
            _boolean(frame, SPECIAL_USE_RESIDENT_ALLOWED_COLUMN).sum()
        ),
        "transient_confidence_rows": int(
            _boolean(frame, SPECIAL_USE_TRANSIENT_FLAG_COLUMN).sum()
        ),
    }
