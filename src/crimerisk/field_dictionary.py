"""The published field dictionary, generated from the surface schema and the build's own manifest.

Nothing here is a hand-maintained list of columns. The generator reads the candidate (or promoted)
surfaces' Parquet schemas, reads the build manifest for the semantics strings the build published
itself under -- the opportunity-normalizer semantics, the count-first composites' common
denominator, the uncertainty layer's index breaks and tier rule, the special-use display policy --
and resolves every column against an ordered rule table keyed on the naming grammar. A column that
matches no rule is an ERROR, not an "undocumented" row: the surface cannot publish a field the
dictionary cannot describe.

The rule table carries the semantics and provenance; the schema carries the names and types; the
surface carries the observed vocabulary of every small categorical. That split is what keeps the
document true after a build changes: adding a field breaks the generator until the field is
described, and changing a semantics string in the manifest changes the document.

Where any ZCTA field appears the ZCTA anti-ZIP caveat is emitted VERBATIM from
`rollups.ZCTA_ANTI_ZIP_CAVEAT`, in the ZCTA rollup's own section and again in its own section at
the end of the document.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd
import pyarrow.parquet as pq

from crimerisk.composites import (
    COMMON_DENOMINATOR_COLUMN,
    COMMON_DENOMINATOR_ID,
    COMMON_DENOMINATOR_SEMANTICS,
    COMPOSITE_VERSION,
    HARM_BURDEN_MIN_SUPPORT,
    INDEX_MAP_BREAKS,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.rollups import ZCTA_ANTI_ZIP_CAVEAT, ROLLUP_GEOGRAPHIES


DICTIONARY_VERSION = "field_dictionary_v1"
MAX_VOCABULARY_CARDINALITY = 12
DEFAULT_DICTIONARY_PATH = Path("docs") / "FIELD_DICTIONARY.md"

OFFENSE_TOKEN = "{offense}"
YEAR_TOKEN = "{year}"


# --- provenance vocabulary -----------------------------------------------------------------
#
# One short, checkable statement of where a family of fields comes from. Kept as a table rather
# than repeated per column so a source change is a one-line edit.
PROVENANCE: dict[str, str] = {
    "allocation": (
        "Allocation lane (`src/crimerisk/allocation.py`): jurisdiction official total per offense "
        "x within-jurisdiction posterior block-group share, raked to that total"
    ),
    "controls": (
        "Level lane (`src/crimerisk/controls.py`): one official annual total per jurisdiction per "
        "offense from the FBI SRS/NIBRS/CIUS agency panel and admitted state/local publications"
    ),
    "city_feeds": (
        "Direct city incident feeds (`src/crimerisk/city_incidents.py`, "
        "`src/crimerisk/city_shares.py`): geocoded incidents used as within-jurisdiction share "
        "evidence only, never as a total"
    ),
    "acs": "ACS 2020-2024 5-year block-group tables (`data/ACS-5yr-2020-2024`)",
    "popest": "Census Vintage 2025 population estimates, county-controlled (`data/Census-PopEst-2020-2025`)",
    "lodes": (
        "LEHD LODES workplace/residence area characteristics, scaled to the target year by QCEW "
        "county-industry totals (`src/crimerisk/qcew_exposure.py`)"
    ),
    "landscan": "LandScan USA ambient daytime/nighttime population raster (`data/LandScan-USA`)",
    "overture": "Overture Places (`basic_category`/`taxonomy` vocabulary), block-group aggregated",
    "nlcd": "NLCD land-cover / impervious-surface rasters, block-group aggregated",
    "tiger": "Census TIGER/Line 2020 geography (block, block group, tract, place, county)",
    "geometry": (
        "Geometry lane (`src/crimerisk/geometry.py`, `state/geometry/`): block-to-jurisdiction and "
        "block-group-to-jurisdiction crosswalks"
    ),
    "exposure_ensemble": (
        "Exposure ensemble (`src/crimerisk/exposure_ensemble.py`) over frozen weights in "
        "`configs/exposure_ensemble_weights_v1.csv`"
    ),
    "special_use": (
        "Special-use taxonomy (`src/crimerisk/special_use.py`): extensive classification inputs "
        "and the typed display gate they determine"
    ),
    "composites": (
        "Count-first composites (`src/crimerisk/composites.py`) over the severity vector in "
        "`configs/severity_weights_v1.csv`"
    ),
    "uncertainty": (
        "Uncertainty layer (`src/crimerisk/uncertainty.py`): coherent draws over totals, shares "
        "and denominators, calibrated by `configs/uncertainty_calibration_v1.csv`"
    ),
    "confidence": (
        "Confidence/provenance lane (`src/crimerisk/confidence.py`, "
        "`src/crimerisk/source_provenance.py`)"
    ),
    "benchmark": (
        "Benchmark-constrained imputation (`src/crimerisk/benchmark_imputation.py`) against the "
        "FBI CDE state estimate"
    ),
    "rollups": (
        "Rollup lane (`src/crimerisk/rollups.py`): counts summed over member block groups, rates "
        "and indexes recomputed at this support"
    ),
    "derived": "Derived in the published surface from other published fields on the same row",
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    group: str
    semantics: str
    provenance: str
    vocabulary: tuple[str, ...] = ()

    def semantics_cell(self) -> str:
        if not self.vocabulary:
            return self.semantics
        values = " \\| ".join(self.vocabulary)
        return f"{self.semantics} Values: {values}."


@dataclass(frozen=True)
class DictionaryContext:
    """The semantics strings this build published itself under, read off its manifest/surface."""

    manifest: dict
    normalizer_semantics: str
    normalizer_id_by_offense: dict[str, str]
    normalizer_version: str | None
    uncertainty_version: str | None
    uncertainty_index_breaks: tuple[float, ...]
    special_use_version: str | None
    composite_version: str | None
    severity_vector_id: str | None
    year: int

    @property
    def lanes(self) -> tuple[str, ...]:
        resolved = self.manifest.get("resolved_config") or {}
        return tuple(
            sorted(
                key
                for key, value in resolved.items()
                if isinstance(value, dict) and bool(value.get("enabled"))
            )
        )


def resolve_context(manifest: dict, *, year: int = 2024) -> DictionaryContext:
    resolved = manifest.get("resolved_config") or {}
    exposure = resolved.get("exposure_ensemble") or {}
    uncertainty = resolved.get("uncertainty_layer") or {}
    taxonomy = resolved.get("special_use_taxonomy") or {}
    composites = resolved.get("count_first_composites") or {}
    summary_composites = (manifest.get("summary") or {}).get("count_first_composites") or {}
    primary_vector = summary_composites.get("primary_severity_vector") or {}
    return DictionaryContext(
        manifest=manifest,
        normalizer_semantics=str(
            exposure.get("semantics") or "opportunity_normalized_intensity_not_person_time_risk"
        ),
        normalizer_id_by_offense={
            str(key): str(value)
            for key, value in (exposure.get("normalizer_id_by_offense") or {}).items()
        },
        normalizer_version=exposure.get("normalizer_version"),
        uncertainty_version=uncertainty.get("version"),
        uncertainty_index_breaks=tuple(
            float(value) for value in (uncertainty.get("index_breaks") or INDEX_MAP_BREAKS)
        ),
        special_use_version=taxonomy.get("version"),
        composite_version=composites.get("version") or COMPOSITE_VERSION,
        severity_vector_id=primary_vector.get("vector_id"),
        year=int(manifest.get("year") or year),
    )


# --- the rule table ---------------------------------------------------------------------------
#
# Ordered: the first matching rule wins, so a specific name beats the family pattern it sits in.
# `{offense}` in a pattern matches exactly the seven Part-I offense tokens.


@dataclass(frozen=True)
class Rule:
    pattern: str
    group: str
    semantics: str
    provenance_key: str

    def regex(self) -> re.Pattern[str]:
        offenses = "|".join(re.escape(offense) for offense in OFFENSES_7)
        pattern = re.escape(self.pattern)
        pattern = pattern.replace(re.escape(OFFENSE_TOKEN), f"(?:{offenses})")
        pattern = pattern.replace(re.escape(YEAR_TOKEN), r"(?:\d{4})")
        return re.compile("^" + pattern + "$")


IDENTITY_RULES: tuple[Rule, ...] = (
    Rule("block_group_geoid", "identity", "2020 Census block group GEOID (12 digits: state+county+tract+block group).", "tiger"),
    Rule("tract_id", "identity", "2020 Census tract GEOID (11 digits) containing this row.", "tiger"),
    Rule("state_fips", "identity", "2-digit state FIPS code.", "tiger"),
    Rule("county_geoid", "identity", "5-digit county FIPS code (state+county).", "tiger"),
    Rule("county_name", "identity", "County name as published in the 2020 Census county list.", "tiger"),
    Rule("state_abbr", "identity", "USPS state abbreviation.", "tiger"),
    Rule("cbsa_code", "identity", "CBSA code from the 2023 OMB delineation.", "rollups"),
    Rule("cbsa_title", "identity", "CBSA title from the 2023 OMB delineation.", "rollups"),
    Rule("zcta5", "identity", "5-digit 2020 ZIP Code Tabulation Area code. NOT a ZIP Code -- see the ZCTA caveat.", "rollups"),
    Rule("rollup_support", "identity", "The geography this row is quoted at.", "rollups"),
    Rule("geography_caveat", "identity", "Verbatim caveat that must travel with every value published at this support.", "rollups"),
    Rule("block_group_parts", "identity", "Number of (block group, unit) parts summed into this row.", "rollups"),
    Rule("block_group_weight_sum", "identity", "Sum of block-group weights in this row; equals the block-group count where the geography nests, and the block-group-equivalents where it does not.", "rollups"),
    Rule("eb_jurisdiction_id", "identity", "Identifier of the law-enforcement jurisdiction whose official total this row's counts are raked to.", "controls"),
    Rule("eb_jurisdiction_type", "identity", "Class of that jurisdiction.", "controls"),
    Rule("dominant_eb_jurisdiction_id", "identity", "Jurisdiction holding the largest share of a split cell.", "controls"),
    Rule("dominant_jurisdiction_share", "identity", "That jurisdiction's share of the cell.", "controls"),
    Rule("mixed_jurisdiction_flag", "identity", "True where the cell is served by more than one jurisdiction.", "controls"),
    Rule("urban_stratum", "identity", "Urbanicity stratum used for stratified validation reporting.", "derived"),
    Rule("uncertainty_layer_version", "identity", "Version of the uncertainty layer that produced this row's interval and decision fields.", "uncertainty"),
    Rule("opportunity_normalizer_semantics", "identity", "What every published per-offense denominator on this row IS. Emitted on every row so a rate can never be read without it.", "exposure_ensemble"),
)

POPULATION_RULES: tuple[Rule, ...] = (
    Rule("population_{year}", "population and exposure", "Resident population, target year, county-controlled.", "popest"),
    Rule("households_total", "population and exposure", "Occupied households; an input to burglary exposure and special-use classification, not a publication gate.", "acs"),
    Rule("land_area_sq_mi", "population and exposure", "Land area in square miles (water excluded); the denominator of every crime density field.", "tiger"),
    Rule("daytime_population_jobs_proxy", "population and exposure", "Modeled daytime headcount: residents + workplace jobs - resident workers.", "lodes"),
    Rule("landscan_day_pop", "population and exposure", "LandScan modeled daytime ambient population.", "landscan"),
    Rule("exposure_proxy_{year}", "population and exposure", "Modeled place-exposure proxy: the person-scale denominator the exposure offenses are normalized by. Not a resident head count and not a measured visitor population.", "exposure_ensemble"),
    Rule("landscan_day_lifted_person_exposure", "population and exposure", "True where the LandScan daytime leg exceeded resident population and set the person exposure.", "exposure_ensemble"),
    Rule("person_exposure_before_hq_jobs_cap", "population and exposure", "Person exposure before the headquarters-jobs cap.", "exposure_ensemble"),
    Rule("person_exposure_hq_jobs_cap", "population and exposure", "The headquarters-jobs cap value applied to this cell.", "exposure_ensemble"),
    Rule("person_exposure_hq_jobs_cap_candidate", "population and exposure", "True where the cell met the headquarters-jobs cap predicate.", "exposure_ensemble"),
    Rule("person_exposure_hq_jobs_capped", "population and exposure", "True where the cap actually bound.", "exposure_ensemble"),
    Rule("commercial_premises_total", "population and exposure", "Commercial premises count entering the burglary opportunity denominator.", "overture"),
    Rule("destination_poi_total", "population and exposure", "Destination points of interest entering the burglary and larceny opportunity denominators.", "overture"),
    Rule("lodes_manufacturing_jobs", "population and exposure", "Workplace jobs, manufacturing.", "lodes"),
    Rule("lodes_wholesale_jobs", "population and exposure", "Workplace jobs, wholesale trade.", "lodes"),
    Rule("lodes_retail_jobs", "population and exposure", "Workplace jobs, retail trade.", "lodes"),
    Rule("lodes_transport_warehouse_jobs", "population and exposure", "Workplace jobs, transportation and warehousing.", "lodes"),
    Rule("lodes_industrial_jobs", "population and exposure", "Workplace jobs, industrial (manufacturing + wholesale + transport/warehouse).", "lodes"),
    Rule("burglary_premises_total", "population and exposure", "The burglary opportunity denominator: households plus weighted commercial, destination-POI and jobs terms.", "exposure_ensemble"),
    Rule("burglary_commercial_exposure_weight", "population and exposure", "Weight applied to the commercial term of the burglary denominator.", "exposure_ensemble"),
    Rule("burglary_destination_poi_exposure_weight", "population and exposure", "Weight applied to the destination-POI term of the burglary denominator.", "exposure_ensemble"),
    Rule("burglary_retail_jobs_exposure_weight", "population and exposure", "Weight applied to the retail-jobs term of the burglary denominator.", "exposure_ensemble"),
    Rule("burglary_industrial_jobs_exposure_weight", "population and exposure", "Weight applied to the industrial-jobs term of the burglary denominator.", "exposure_ensemble"),
    Rule("aggregate_vehicles_total", "population and exposure", "Household vehicles available.", "acs"),
    Rule("county_auto_commute_vehicle_share", "population and exposure", "County share of commuters driving, used to scale the commuter vehicle proxy.", "acs"),
    Rule("mvt_commuter_vehicle_proxy", "population and exposure", "Commuter vehicles present at the workplace end of the journey.", "lodes"),
    Rule("vehicle_exposure_{year}", "population and exposure", "The motor-vehicle-theft opportunity denominator: household vehicles plus the commuter vehicle proxy.", "exposure_ensemble"),
    Rule("resident_secondary_denominator", "population and exposure", "Resident population as a secondary denominator, and the single common denominator of every count-first composite.", "popest"),
    Rule("resident_secondary_denominator_low_reliability", "population and exposure", "True where the resident denominator is small enough that its burden rate is unstable.", "derived"),
    Rule("population_zero_with_positive_count", "population and exposure", "True where a cell carries expected counts with no resident population.", "derived"),
    Rule("transient_exposure_daytime_to_resident_ratio", "population and exposure", "Ratio of modeled daytime to resident population; the transient-exposure screen.", "derived"),
)

FLOOR_RULES: tuple[Rule, ...] = (
    Rule("eb_hard_min_denominator", "publication floors", "Hard minimum denominator below which no rate publishes at all.", "allocation"),
    Rule("non_residential_household_floor", "publication floors", "Legacy household threshold retained as descriptive metadata; it does not gate the current release.", "allocation"),
    Rule("person_exposure_denominator_floor", "publication floors", "Minimum person exposure for an exposure-normalized rate to publish.", "exposure_ensemble"),
    Rule("mvt_vehicle_exposure_denominator_floor", "publication floors", "Minimum vehicle exposure for the motor-vehicle-theft rate to publish.", "exposure_ensemble"),
    Rule("zero_resident_opportunity_rate_floor", "publication floors", "Opportunity denominator a near-zero-resident cell must clear before its opportunity rate publishes.", "exposure_ensemble"),
    Rule("non_residential_flag", "publication floors", "Descriptive flag for cells below the legacy household threshold; it does not gate publication.", "derived"),
)

SPECIAL_USE_RULES: tuple[Rule, ...] = (
    Rule("special_use_tract_flag", "special use", "True where the parent tract is classified special-use.", "special_use"),
    Rule("special_use_acs_population", "special use", "Classification input: ACS population.", "special_use"),
    Rule("special_use_household_population", "special use", "Classification input: household (non-group-quarters) population.", "special_use"),
    Rule("special_use_jobs_total", "special use", "Classification input: total workplace jobs.", "special_use"),
    Rule("special_use_jobs_education", "special use", "Classification input: education-sector jobs.", "special_use"),
    Rule("special_use_postsecondary_count", "special use", "Classification input: postsecondary institutions present.", "special_use"),
    Rule("special_use_open_natural_pixels", "special use", "Classification input: open/natural land-cover pixels.", "special_use"),
    Rule("special_use_classified_pixels", "special use", "Classification input: total classified land-cover pixels.", "special_use"),
    Rule("special_use_gq_total_2020", "special use", "2020 Census group-quarters population.", "special_use"),
    Rule("special_use_gq_institutional_2020", "special use", "2020 Census institutional group-quarters population.", "special_use"),
    Rule("special_use_gq_correctional_2020", "special use", "2020 Census correctional-facility group-quarters population.", "special_use"),
    Rule("special_use_gq_juvenile_2020", "special use", "2020 Census juvenile-facility group-quarters population.", "special_use"),
    Rule("special_use_gq_nursing_2020", "special use", "2020 Census nursing-facility group-quarters population.", "special_use"),
    Rule("special_use_gq_college_2020", "special use", "2020 Census college group-quarters population.", "special_use"),
    Rule("special_use_gq_military_2020", "special use", "2020 Census military group-quarters population.", "special_use"),
    Rule("special_use_gq_other_institutional_2020", "special use", "2020 Census other institutional group-quarters population.", "special_use"),
    Rule("special_use_gq_other_noninstitutional_2020", "special use", "2020 Census other noninstitutional group-quarters population.", "special_use"),
    Rule("special_use_type", "special use", "Typed special-use class, recomputable from the published classification inputs beside it.", "special_use"),
    Rule("special_use_type_evidence", "special use", "The classification inputs that decided the type.", "special_use"),
    Rule("special_use_candidate_flag", "special use", "True where the cell met any special-use predicate.", "special_use"),
    Rule("special_use_display_policy", "special use", "Display annotation associated with the classified type; it does not gate the choropleth value.", "special_use"),
    Rule("special_use_primary_rate_allowed", "special use", "Compatibility flag; true for every type because type does not gate publication.", "special_use"),
    Rule("special_use_resident_rate_allowed", "special use", "Compatibility flag; true for every type because type does not gate publication.", "special_use"),
    Rule("special_use_transient_confidence_flag", "special use", "True where a transient-destination classification is low-confidence.", "special_use"),
)

AGGREGATE_RULES: tuple[Rule, ...] = (
    Rule("expected_count_personal", "aggregates", "Expected reported personal-crime offences: the sum of the four personal Part-I counts on this row.", "allocation"),
    Rule("expected_count_property", "aggregates", "Expected reported property-crime offences: the sum of the three property Part-I counts on this row.", "allocation"),
    Rule("expected_count_total", "aggregates", "Expected reported Part-I offences: the sum of all seven counts on this row.", "allocation"),
    Rule("crime_density_total", "aggregates", "All-offence expected counts per square mile of land area.", "derived"),
    Rule("harm_weighted_count_total", "aggregates", "Severity-weighted count: sum over offences of severity weight x expected count. The harm index's numerator, published so the index recomputes from published fields.", "composites"),
    Rule("index_event_burden_resident", "aggregates", "Count-first event burden: all seven counts over one common resident denominator, indexed to 100 at the published reference rate. Every offence enters at its own count share.", "composites"),
    Rule("index_personal_burden_resident", "aggregates", "Count-first personal-crime burden over the same common resident denominator.", "composites"),
    Rule("index_property_burden_resident", "aggregates", "Count-first property-crime burden over the same common resident denominator.", "composites"),
    Rule("index_harm_burden_resident", "aggregates", "Count-first severity-weighted burden over the same common resident denominator. Published at tract support and coarser only; present and null at block group.", "composites"),
    Rule("multi_offense_relative_score_event_weighted", "aggregates", "Event-count-weighted average of the seven per-offence indexes. A relative dashboard score over mixed denominators, NOT a rate of anything, and not a total. All-or-null.", "derived"),
    Rule("multi_offense_relative_score_equal_offense", "aggregates", "Equal-weighted average of the seven per-offence indexes. A relative dashboard score over mixed denominators, NOT a rate of anything, and not a total. All-or-null.", "derived"),
    Rule("multi_offense_relative_score_personal_event_weighted", "aggregates", "Event-count-weighted average of the four personal-offence primary indexes. Dimensionless activity-adjusted score; not a rate or probability. All-or-null.", "derived"),
    Rule("multi_offense_relative_score_property_event_weighted", "aggregates", "Event-count-weighted average of the three property-offence primary indexes. Dimensionless activity-adjusted score; not a rate or probability. All-or-null.", "derived"),
)

OFFENSE_RULES: tuple[Rule, ...] = (
    Rule("expected_count_{offense}", "per offence: counts", "Expected reported offences for the target year: the jurisdiction's official total x this cell's posterior within-jurisdiction share. Conserves exactly to the jurisdiction total.", "allocation"),
    Rule("crime_density_{offense}", "per offence: counts", "Expected counts per square mile of land area.", "derived"),
    Rule("primary_denominator_type_{offense}", "per offence: denominator", "Which opportunity denominator this offence is normalized by.", "exposure_ensemble"),
    Rule("primary_denominator_{offense}", "per offence: denominator", "The published denominator: the named opportunity normalizer for this offence, after floors.", "exposure_ensemble"),
    Rule("primary_denominator_normalizer_id_{offense}", "per offence: denominator", "Versioned id of the normalizer that produced the denominator.", "exposure_ensemble"),
    Rule("opportunity_normalizer_{offense}", "per offence: denominator", "The normalizer value before publication floors.", "exposure_ensemble"),
    Rule("primary_denominator_raw_{offense}", "per offence: denominator", "The denominator before floors and caps.", "exposure_ensemble"),
    Rule("primary_denominator_invalid_{offense}", "per offence: denominator", "True where the denominator failed its validity test and the rate is withheld.", "exposure_ensemble"),
    Rule("primary_national_rate_per_100k_{offense}", "per offence: denominator", "The national normalizer this row's index divides by: national counts over national denominators of the same type, per 100,000.", "exposure_ensemble"),
    Rule("primary_alpha_{offense}", "per offence: denominator", "Diagnostic prior strength retained beside the published value; never enters it.", "allocation"),
    Rule("rate_{offense}_primary", "per offence: published point", "Published opportunity-normalized rate per 100,000 units of the named denominator: 100000 x expected count / denominator. Recomputable from the two published fields beside it.", "allocation"),
    Rule("index_{offense}_primary", "per offence: published point", "Published index: 100 x this row's rate / the national rate published beside it. 100 = national.", "allocation"),
    Rule("primary_index_publishable_{offense}", "per offence: publication", "True where every publication gate passed and the index is published.", "allocation"),
    Rule("primary_index_suppressed_{offense}", "per offence: publication", "True where the index was withheld by a publication gate.", "allocation"),
    Rule("primary_zero_denominator_positive_count_{offense}", "per offence: publication", "True where the cell carries counts against a zero denominator.", "allocation"),
    Rule("denominator_reason_{offense}", "per offence: publication", "Which gate decided publication of the primary rate.", "allocation"),
    Rule("estimate_mode_{offense}", "per offence: publication", "The estimand label for this cell's published value.", "allocation"),
    Rule("index_publishable_{offense}", "per offence: publication", "Publication flag retained from the pre-typed lane; equals the primary flag on this build.", "allocation"),
    Rule("recommended_display_geography_{offense}", "per offence: publication", "The coarsest support at which this cell's value should be shown, from its measured interval width.", "uncertainty"),
    Rule("raw_rate_{offense}", "per offence: diagnostic", "Unfloored count over denominator, before publication gates.", "allocation"),
    Rule("diagnostic_eb_{offense}", "per offence: diagnostic", "Empirical-Bayes diagnostic. Diagnostic only: never enters a published rate or index.", "allocation"),
    Rule("diagnostic_eb_rate_{offense}", "per offence: diagnostic", "Empirical-Bayes shrunken rate. Diagnostic only: never enters the published point value.", "allocation"),
    Rule("diagnostic_eb_national_rate_per_100k_{offense}", "per offence: diagnostic", "National rate used by the EB diagnostic.", "allocation"),
    Rule("diagnostic_eb_prior_rate_{offense}", "per offence: diagnostic", "Prior rate the EB diagnostic shrinks toward.", "allocation"),
    Rule("diagnostic_eb_k_{offense}", "per offence: diagnostic", "EB shrinkage constant.", "allocation"),
    Rule("diagnostic_eb_observed_weight_{offense}", "per offence: diagnostic", "Weight the EB diagnostic gives the observed rate.", "allocation"),
    Rule("diagnostic_eb_prior_weight_{offense}", "per offence: diagnostic", "Weight the EB diagnostic gives the prior.", "allocation"),
    Rule("diagnostic_eb_low_denominator_flag_{offense}", "per offence: diagnostic", "EB diagnostic flag: low denominator.", "allocation"),
    Rule("diagnostic_eb_heavy_shrinkage_flag_{offense}", "per offence: diagnostic", "EB diagnostic flag: heavy shrinkage.", "allocation"),
    Rule("diagnostic_eb_extreme_shrinkage_flag_{offense}", "per offence: diagnostic", "EB diagnostic flag: extreme shrinkage.", "allocation"),
    Rule("direct_incident_support_flag_{offense}", "per offence: evidence", "True where a direct city incident feed supplied share evidence for this cell.", "city_feeds"),
    Rule("direct_incident_support_count_{offense}", "per offence: evidence", "Geocoded incidents supporting this cell's share.", "city_feeds"),
    Rule("direct_incident_support_years_{offense}", "per offence: evidence", "Feed-years of incident support.", "city_feeds"),
    Rule("direct_incident_support_year_min_{offense}", "per offence: evidence", "First feed year contributing support.", "city_feeds"),
    Rule("direct_incident_support_year_max_{offense}", "per offence: evidence", "Last feed year contributing support.", "city_feeds"),
    Rule("effective_numerator_support_{offense}", "per offence: evidence", "Effective incident support behind the numerator after feed-quality weighting.", "city_feeds"),
    Rule("numerator_support_source_{offense}", "per offence: evidence", "Whether the share came from direct incident evidence or from the covariate model.", "city_feeds"),
    Rule("footprint_derived_count_{offense}", "per offence: evidence", "Counts entering this cell from a jurisdiction footprint rather than a municipal boundary.", "geometry"),
    Rule("footprint_derived_count_share_{offense}", "per offence: evidence", "That share of the cell's count.", "geometry"),
    Rule("footprint_ambient_exposure_missing_{offense}", "per offence: evidence", "True where footprint-derived mass has no ambient exposure to normalize by.", "geometry"),
    Rule("transient_exposure_likely_{offense}", "per offence: evidence", "True where the cell's daytime-to-resident ratio marks it as transient-exposure dominated.", "derived"),
    Rule("rate_{offense}_primary_ci95_lower", "per offence: interval", "Lower bound of the 95% count interval propagated to the rate.", "uncertainty"),
    Rule("rate_{offense}_primary_ci95_upper", "per offence: interval", "Upper bound of the 95% count interval propagated to the rate.", "uncertainty"),
    Rule("index_{offense}_primary_ci95_lower", "per offence: interval", "Lower bound of the 95% count interval propagated to the index.", "uncertainty"),
    Rule("index_{offense}_primary_ci95_upper", "per offence: interval", "Upper bound of the 95% count interval propagated to the index.", "uncertainty"),
    Rule("index_{offense}_primary_ci95_width", "per offence: interval", "Width of that index interval.", "uncertainty"),
    Rule("index_{offense}_primary_ci95_width_ratio", "per offence: interval", "That width relative to the index point.", "uncertainty"),
    Rule("reliability_tier_{offense}", "per offence: reliability", "Reliability tier for the published index.", "uncertainty"),
    Rule("resident_national_rate_per_100k_{offense}", "per offence: resident lane", "National normalizer for the resident-denominator rate.", "popest"),
    Rule("resident_raw_rate_{offense}", "per offence: resident lane", "Unfloored count over resident population.", "allocation"),
    Rule("rate_{offense}_resident", "per offence: resident lane", "Published resident-denominator rate per 100,000 residents: 100000 x expected count / resident population. A burden per resident, not a personal danger probability.", "allocation"),
    Rule("index_{offense}_resident", "per offence: resident lane", "Published resident-denominator index: 100 x resident rate / national resident rate.", "allocation"),
    Rule("index_{offense}_resident_publishable", "per offence: resident lane", "True where the resident lane's gates passed.", "allocation"),
    Rule("index_{offense}_resident_suppressed", "per offence: resident lane", "True where the resident index was withheld.", "allocation"),
    Rule("resident_denominator_reason_{offense}", "per offence: resident lane", "Which gate decided publication of the resident rate.", "allocation"),
    Rule("resident_denominator_invalid_{offense}", "per offence: resident lane", "True where the resident denominator failed its validity test.", "allocation"),
    Rule("diagnostic_resident_eb_rate_{offense}", "per offence: diagnostic", "Resident-lane EB diagnostic rate. Diagnostic only.", "allocation"),
    Rule("diagnostic_resident_eb_national_rate_per_100k_{offense}", "per offence: diagnostic", "National rate used by the resident EB diagnostic.", "allocation"),
    Rule("diagnostic_resident_eb_prior_rate_{offense}", "per offence: diagnostic", "Prior rate the resident EB diagnostic shrinks toward.", "allocation"),
    Rule("diagnostic_resident_eb_k_{offense}", "per offence: diagnostic", "Resident EB shrinkage constant.", "allocation"),
    Rule("diagnostic_resident_eb_observed_weight_{offense}", "per offence: diagnostic", "Weight the resident EB diagnostic gives the observed rate.", "allocation"),
    Rule("diagnostic_resident_eb_prior_weight_{offense}", "per offence: diagnostic", "Weight the resident EB diagnostic gives the prior.", "allocation"),
    Rule("diagnostic_resident_eb_low_denominator_flag_{offense}", "per offence: diagnostic", "Resident EB diagnostic flag: low denominator.", "allocation"),
    Rule("diagnostic_resident_eb_heavy_shrinkage_flag_{offense}", "per offence: diagnostic", "Resident EB diagnostic flag: heavy shrinkage.", "allocation"),
    Rule("diagnostic_resident_eb_extreme_shrinkage_flag_{offense}", "per offence: diagnostic", "Resident EB diagnostic flag: extreme shrinkage.", "allocation"),
    Rule("source_mode_{offense}", "per offence: provenance", "Which source lane supplied this cell's jurisdiction total.", "confidence"),
    Rule("source_mode_dominant_share_{offense}", "per offence: provenance", "Share of the cell's count coming from the dominant source lane.", "confidence"),
    Rule("source_mode_mixed_{offense}", "per offence: provenance", "True where more than one source lane contributes.", "confidence"),
    Rule("feed_match_rate_{offense}", "per offence: provenance", "Share of the covering feed's incidents that geocoded to a block group.", "city_feeds"),
    Rule("feed_missing_fraction_{offense}", "per offence: provenance", "Share of the covering feed's incidents that could not be placed.", "city_feeds"),
    Rule("feed_alpha_{offense}", "per offence: provenance", "Posterior weight given to feed evidence against the model prior.", "city_feeds"),
    Rule("feed_prior_fraction_{offense}", "per offence: provenance", "Share of the posterior share coming from the model prior.", "city_feeds"),
    Rule("benchmark_imputed_share_{offense}", "per offence: provenance", "Share of the cell's count that came from benchmark-constrained imputation of silent-agency territory.", "benchmark"),
    Rule("domain_overlap_score_{offense}", "per offence: provenance", "How close this cell's covariate domain is to the training cities the share model transfers from.", "allocation"),
    Rule("level_admission_status_{offense}", "per offence: level lane", "Admission disposition of the jurisdiction-offence input behind this cell.", "controls"),
    Rule("level_admission_reason_{offense}", "per offence: level lane", "Why that disposition was reached.", "controls"),
    Rule("level_repair_mode_{offense}", "per offence: level lane", "Which reason-specific repair produced the total, where one was needed.", "controls"),
    Rule("level_repair_share_{offense}", "per offence: level lane", "Share of the total that came from repair rather than an admitted report.", "controls"),
    Rule("external_check_status_{offense}", "per offence: level lane", "Result of the external-evidence check on the jurisdiction total.", "controls"),
    Rule("benchmark_conflict_kind_{offense}", "per offence: level lane", "How the total sits against the FBI CDE state benchmark.", "benchmark"),
    Rule("benchmark_weight_{offense}", "per offence: level lane", "Weight the benchmark reconciliation carried.", "benchmark"),
    Rule("unresolved_level_flag_{offense}", "per offence: level lane", "True where a level-lane review hold remains open.", "controls"),
    Rule("level_provenance_text_{offense}", "per offence: level lane", "Reader-facing provenance sentence for the jurisdiction total.", "confidence"),
    Rule("spatial_share_reliability_tier_{offense}", "per offence: reliability", "Reliability of the within-jurisdiction share.", "confidence"),
    Rule("level_reliability_tier_{offense}", "per offence: reliability", "Reliability of the jurisdiction total.", "confidence"),
    Rule("confidence_tier_{offense}", "per offence: reliability", "Combined reliability tier over the total and the share.", "confidence"),
    Rule("confidence_reasons_{offense}", "per offence: reliability", "The reasons behind the combined tier.", "confidence"),
    Rule("expected_count_{offense}_p10", "per offence: uncertainty layer", "10th percentile of the count posterior.", "uncertainty"),
    Rule("expected_count_{offense}_p50", "per offence: uncertainty layer", "Median of the count posterior.", "uncertainty"),
    Rule("expected_count_{offense}_p90", "per offence: uncertainty layer", "90th percentile of the count posterior.", "uncertainty"),
    Rule("rate_{offense}_primary_p10", "per offence: uncertainty layer", "10th percentile of the rate posterior.", "uncertainty"),
    Rule("rate_{offense}_primary_p50", "per offence: uncertainty layer", "Median of the rate posterior.", "uncertainty"),
    Rule("rate_{offense}_primary_p90", "per offence: uncertainty layer", "90th percentile of the rate posterior.", "uncertainty"),
    Rule("index_{offense}_primary_p10", "per offence: uncertainty layer", "10th percentile of the index posterior.", "uncertainty"),
    Rule("index_{offense}_primary_p25", "per offence: uncertainty layer", "25th percentile of the index posterior.", "uncertainty"),
    Rule("index_{offense}_primary_p50", "per offence: uncertainty layer", "Median of the index posterior.", "uncertainty"),
    Rule("index_{offense}_primary_p75", "per offence: uncertainty layer", "75th percentile of the index posterior.", "uncertainty"),
    Rule("index_{offense}_primary_p90", "per offence: uncertainty layer", "90th percentile of the index posterior.", "uncertainty"),
    Rule("prob_index_{offense}_above_100", "per offence: uncertainty layer", "Posterior probability the cell's index exceeds the national reference of 100.", "uncertainty"),
    Rule("displayed_index_bin_{offense}", "per offence: uncertainty layer", "The published map bin the index point falls in.", "uncertainty"),
    Rule("prob_displayed_index_bin_{offense}", "per offence: uncertainty layer", "Posterior probability the cell belongs in the bin it is painted.", "uncertainty"),
    Rule("decision_reliability_tier_{offense}", "per offence: uncertainty layer", "Reliability tier from that bin probability.", "uncertainty"),
    Rule("uncertainty_support_class_{offense}", "per offence: uncertainty layer", "Which evidence class the cell's draws were calibrated against.", "uncertainty"),
    Rule("uncertainty_control_log_sd_{offense}", "per offence: uncertainty layer", "Measured log-scale dispersion of the jurisdiction total.", "uncertainty"),
    Rule("uncertainty_share_log_sd_{offense}", "per offence: uncertainty layer", "Measured log-scale dispersion of the within-jurisdiction share.", "uncertainty"),
    Rule("uncertainty_denominator_log_sd_{offense}", "per offence: uncertainty layer", "Measured log-scale dispersion of the denominator.", "uncertainty"),
)

RULES: tuple[Rule, ...] = (
    *IDENTITY_RULES,
    *POPULATION_RULES,
    *FLOOR_RULES,
    *SPECIAL_USE_RULES,
    *AGGREGATE_RULES,
    *OFFENSE_RULES,
)

GROUP_ORDER: tuple[str, ...] = (
    "identity",
    "population and exposure",
    "publication floors",
    "special use",
    "per offence: counts",
    "per offence: denominator",
    "per offence: published point",
    "per offence: publication",
    "per offence: resident lane",
    "per offence: interval",
    "per offence: reliability",
    "per offence: uncertainty layer",
    "per offence: evidence",
    "per offence: provenance",
    "per offence: level lane",
    "per offence: diagnostic",
    "aggregates",
)


class UndocumentedFieldError(ValueError):
    """A published column matched no rule. The dictionary fails closed rather than omitting it."""


def _compiled_rules() -> list[tuple[re.Pattern[str], Rule]]:
    return [(rule.regex(), rule) for rule in RULES]


def resolve_rule(column: str, compiled: list[tuple[re.Pattern[str], Rule]]) -> Rule:
    for regex, rule in compiled:
        if regex.match(column):
            return rule
    raise UndocumentedFieldError(
        f"published column {column!r} matches no field-dictionary rule; add a rule in "
        "src/crimerisk/field_dictionary.py rather than publishing an undescribed field"
    )


def _contextual_semantics(rule: Rule, column: str, context: DictionaryContext) -> str:
    """Splice the build's OWN semantics strings into the rule's description where it has one."""
    semantics = rule.semantics
    if column == "opportunity_normalizer_semantics":
        return f"{semantics} Constant on this build: `{context.normalizer_semantics}`."
    if column == COMMON_DENOMINATOR_COLUMN:
        return (
            f"{semantics} Common-denominator id `{COMMON_DENOMINATOR_ID}`, semantics "
            f"`{COMMON_DENOMINATOR_SEMANTICS}`."
        )
    if column.startswith("primary_denominator_normalizer_id_"):
        offense = column[len("primary_denominator_normalizer_id_") :]
        normalizer = context.normalizer_id_by_offense.get(offense)
        if normalizer:
            return f"{semantics} On this build: `{normalizer}`."
    if column.startswith("primary_denominator_") and rule.pattern.startswith("primary_denominator_{"):
        offense = column[len("primary_denominator_") :]
        normalizer = context.normalizer_id_by_offense.get(offense)
        if normalizer:
            return (
                f"{semantics} Normalizer `{normalizer}`; semantics "
                f"`{context.normalizer_semantics}`."
            )
    if column.startswith("rate_") and column.endswith("_primary"):
        return f"{semantics} Semantics: `{context.normalizer_semantics}`."
    if column == "index_harm_burden_resident" and context.severity_vector_id:
        return (
            f"{semantics} Severity vector `{context.severity_vector_id}` "
            f"(`configs/severity_weights_v1.csv`); minimum support `{HARM_BURDEN_MIN_SUPPORT}`."
        )
    if column.startswith("displayed_index_bin_") or column.startswith("prob_displayed_index_bin_"):
        breaks = ", ".join(str(value) for value in context.uncertainty_index_breaks)
        return f"{semantics} Bin edges: {breaks}."
    return semantics


# --- schema and vocabulary reading -------------------------------------------------------------


def read_schema(path: Path) -> dict[str, str]:
    schema = pq.ParquetFile(str(path)).schema_arrow
    return {name: str(schema.field(name).type) for name in schema.names}


def read_vocabularies(
    path: Path, columns: list[str], *, max_cardinality: int = MAX_VOCABULARY_CARDINALITY
) -> dict[str, tuple[str, ...]]:
    """Observed values of every small categorical, read off the surface itself.

    Read in batches so a 792-column surface never lands in memory at once.
    """
    schema = pq.ParquetFile(str(path)).schema_arrow
    categorical = [
        name
        for name in columns
        if name in schema.names and str(schema.field(name).type) in ("string", "bool", "large_string")
    ]
    out: dict[str, tuple[str, ...]] = {}
    batch = 24
    for start in range(0, len(categorical), batch):
        chunk = categorical[start : start + batch]
        frame = pd.read_parquet(path, columns=chunk)
        for column in chunk:
            values = frame[column].dropna().unique()
            if 0 < len(values) <= max_cardinality:
                out[column] = tuple(sorted(str(value) for value in values))
    return out


def build_field_specs(
    *,
    schema: dict[str, str],
    vocabularies: dict[str, tuple[str, ...]],
    context: DictionaryContext,
) -> list[FieldSpec]:
    compiled = _compiled_rules()
    specs: list[FieldSpec] = []
    for name, dtype in schema.items():
        rule = resolve_rule(name, compiled)
        specs.append(
            FieldSpec(
                name=name,
                dtype=dtype,
                group=rule.group,
                semantics=_contextual_semantics(rule, name, context),
                provenance=PROVENANCE[rule.provenance_key],
                vocabulary=vocabularies.get(name, ()),
            )
        )
    return specs


# --- rendering ---------------------------------------------------------------------------------


def _table(specs: list[FieldSpec]) -> list[str]:
    lines = ["| Column | Type | Semantics | Provenance |", "|---|---|---|---|"]
    for spec in specs:
        lines.append(
            f"| `{spec.name}` | {spec.dtype} | {spec.semantics_cell()} | {spec.provenance} |"
        )
    return lines


def _grouped_sections(specs: list[FieldSpec]) -> list[str]:
    lines: list[str] = []
    by_group: dict[str, list[FieldSpec]] = {}
    for spec in specs:
        by_group.setdefault(spec.group, []).append(spec)
    ordered = [group for group in GROUP_ORDER if group in by_group]
    ordered += [group for group in by_group if group not in GROUP_ORDER]
    for group in ordered:
        lines.append("")
        lines.append(f"### {group}")
        lines.append("")
        lines += _table(by_group[group])
    return lines


@dataclass(frozen=True)
class SurfaceEntry:
    label: str
    geography: str
    path: Path
    rows: int | None = None


def _lookup_appendix(lookup_manifest: dict | None) -> list[str]:
    """The static lookup's client pattern, documented beside the fields it serves.

    Emitted from `src/crimerisk/lookup.py`'s own constants and, where an edition's lookup has been
    built, from that lookup's own manifest -- so the prefix lengths in this document are the ones
    the shards were actually written under, not a scheme described in prose.
    """
    from crimerisk.lookup import (
        CLIENT_PATTERN,
        HEADLINE_FIELD_FAMILIES,
        LOOKUP_DIRNAME,
        LOOKUP_VERSION,
        MANIFEST_FILENAME,
        SHARD_BYTE_BUDGET,
    )

    lines = [
        "",
        "## Appendix: static lookup index",
        "",
        f"An edition carries a `{LOOKUP_DIRNAME}/` directory ({LOOKUP_VERSION}) that answers one",
        "question -- *what are this GEOID's headline values* -- **with no server, no database and no",
        "query language**. It is plain JSON on a static host.",
        "",
        "### The client pattern",
        "",
        f"1. Fetch `{LOOKUP_DIRNAME}/{MANIFEST_FILENAME}` once and keep it. It names, per geography,",
        "   the GEOID `prefix_length`, the `id_field`, and the ordered `fields` list.",
        "2. For a GEOID, the shard filename is the GEOID's own prefix:",
        "   `shard = geoid[:prefix_length]`.",
        f"3. Fetch `{LOOKUP_DIRNAME}/<geography>/<shard>.json` -- **one request**, cacheable, and",
        "   independent of how many rows the edition publishes.",
        "4. `payload[\"rows\"][<full GEOID>]` is an array of values in `payload[\"fields\"]` order.",
        "   A missing GEOID is `undefined`; a missing value is `null`.",
        "",
        f"`{CLIENT_PATTERN}`",
        "",
        "```js",
        f"const manifest = await (await fetch(`{LOOKUP_DIRNAME}/{MANIFEST_FILENAME}`)).json();",
        "const spec     = manifest.geographies[geography];",
        "const shard    = geoid.slice(0, spec.prefix_length);",
        f"const bundle   = await (await fetch(`{LOOKUP_DIRNAME}/${{geography}}/${{shard}}.json`)).json();",
        "const values   = bundle.rows[geoid];",
        "const row      = Object.fromEntries(bundle.fields.map((f, i) => [f, values[i]]));",
        "```",
        "",
        "### The shard scheme",
        "",
        f"Every shard is kept under {SHARD_BYTE_BUDGET:,} bytes, and the prefix length for a geography",
        "is the **shortest one that achieves that**, measured on the payload actually written. A",
        "five-digit (county) prefix is the natural key and is what the coarse geographies land on or",
        "beat, but it does not fit at block-group support -- one large county holds several thousand",
        "block groups -- so the block-group and tract lanes shard on a longer prefix. The length is",
        "read from the manifest by the client and is never assumed.",
        "",
        "### Published fields",
        "",
        "| Family | Fields |",
        "|---|---|",
    ]
    for family, fields in HEADLINE_FIELD_FAMILIES.items():
        rendered = ", ".join(f"`{field}`" for field in fields)
        lines.append(f"| {family} | {rendered} |")
    lines += [
        "",
        "Every field above is defined in this document's per-surface sections; the lookup transports",
        "the published value unchanged (floats are emitted with the shortest representation that",
        "round-trips to the same double, so a value read from a shard equals the value in the table).",
        "A field a support does not carry -- a county rollup has no reliability tier -- is listed",
        "under that geography's `absent_headline_fields` rather than emitted null. A unit's name is",
        "carried where the geography has one.",
    ]
    if lookup_manifest:
        geographies = lookup_manifest.get("geographies") or {}
        totals = lookup_manifest.get("totals") or {}
        lines += [
            "",
            "### This edition's lookup",
            "",
            "| Geography | Id field | Prefix | Shards | Rows | Fields | Largest shard (bytes) |",
            "|---|---|---|---|---|---|---|",
        ]
        for geography in sorted(geographies):
            entry = geographies[geography]
            lines.append(
                f"| {geography} | `{entry['id_field']}` | {entry['prefix_length']} | "
                f"{int(entry['shards']):,} | {int(entry['rows']):,} | {len(entry['fields'])} | "
                f"{int(entry['max_shard_bytes']):,} |"
            )
        lines += [
            "",
            f"{int(totals.get('shards', 0)):,} shards over {int(totals.get('rows', 0)):,} rows, "
            f"{int(totals.get('bytes', 0)) / 1e6:.1f} MB. Every shard's size and SHA-256 are in "
            "`lookup/checksums.json`, whose own digest is in `lookup/manifest.json`.",
        ]
    return lines


def render_markdown(
    *,
    surfaces: list[tuple[SurfaceEntry, list[FieldSpec]]],
    context: DictionaryContext,
    coverage_universe: dict[str, object],
    edition_id: str | None = None,
    generated_at: str | None = None,
    lookup_manifest: dict | None = None,
) -> str:
    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        "# Field dictionary",
        "",
        f"Generated by `src/crimerisk/field_dictionary.py` ({DICTIONARY_VERSION}). Do not edit by hand.",
        "",
        "| Key | Value |",
        "|---|---|",
        f"| Edition | {edition_id or '(unversioned build)'} |",
        f"| Target year | {context.year} |",
        f"| Generated (UTC) | {stamp} |",
        f"| Build lanes | {', '.join(context.lanes) or '(none declared)'} |",
        f"| Coverage universe | {coverage_universe.get('description', '')} |",
        f"| Block groups | {coverage_universe.get('block_groups', '')} |",
        f"| Counties | {coverage_universe.get('counties', '')} |",
        "",
        "## Surfaces",
        "",
        "| Surface | Geography | Columns | Rows | File |",
        "|---|---|---|---|---|",
    ]
    for entry, specs in surfaces:
        rows = "" if entry.rows is None else f"{entry.rows:,}"
        lines.append(
            f"| {entry.label} | {entry.geography} | {len(specs)} | {rows} | `{entry.path.name}` |"
        )

    lines += [
        "",
        "## Naming grammar",
        "",
        "| Pattern | Meaning |",
        "|---|---|",
        "| `expected_count_{offence}` | Expected reported offences for the target year. |",
        "| `rate_{offence}_primary` | 100000 x expected count / opportunity denominator. |",
        "| `index_{offence}_primary` | 100 x rate / the national rate published beside it. |",
        "| `rate_{offence}_resident`, `index_{offence}_resident` | The same two quantities over resident population. |",
        "| `diagnostic_*` | Diagnostic only. Never enters a published point value. |",
        "| `*_p10`, `*_p50`, `*_p90` | Posterior quantiles from the uncertainty layer. |",
        "| `*_publishable`, `*_suppressed`, `*_reason_*` | Publication gates and the reason each decided. |",
        "",
        "## Standing semantics",
        "",
        "| Statement | Value |",
        "|---|---|",
        f"| Opportunity-normalizer semantics | `{context.normalizer_semantics}` |",
        f"| Count-first composite denominator | `{COMMON_DENOMINATOR_ID}` (`{COMMON_DENOMINATOR_COLUMN}`), `{COMMON_DENOMINATOR_SEMANTICS}` |",
        f"| Composite version | `{context.composite_version}` |",
        f"| Severity vector | `{context.severity_vector_id or '(none)'}` |",
        f"| Harm burden minimum support | `{HARM_BURDEN_MIN_SUPPORT}` |",
        f"| Uncertainty layer version | `{context.uncertainty_version or '(none)'}` |",
        f"| Special-use taxonomy version | `{context.special_use_version or '(none)'}` |",
        f"| Index bin edges | {', '.join(str(value) for value in context.uncertainty_index_breaks)} |",
        "",
        "In plain English: every rate in this file answers \"how much reported crime, relative to",
        "the amount of opportunity for it\" — not \"what is any one person's chance of being a",
        "victim.\" The `opportunity_normalized_intensity_not_person_time_risk` token above, and its",
        "repeats throughout the column tables below, is that same statement in the machine-readable",
        "form the pipeline carries end to end.",
        "",
        "Every published rate and index is count-derived: `rate = 100000 * expected_count / denominator`",
        "and `index = 100 * rate / national_rate`, with the count, the denominator and the national rate",
        "all published on the same row. Rolled-up geographies sum counts and denominators and recompute",
        "the rate and index at that support; no rate or index is ever averaged.",
        "",
        "## ZCTA caveat",
        "",
        ZCTA_ANTI_ZIP_CAVEAT,
        "",
    ]

    for entry, specs in surfaces:
        lines.append("")
        lines.append(f"## {entry.label} ({entry.geography})")
        lines.append("")
        lines.append(f"File: `{entry.path.name}` — {len(specs)} columns.")
        if entry.geography == "zcta":
            lines.append("")
            lines.append(ZCTA_ANTI_ZIP_CAVEAT)
        lines += _grouped_sections(specs)
    lines += _lookup_appendix(lookup_manifest)
    lines.append("")
    return "\n".join(lines)


def build_dictionary(
    *,
    surfaces: list[SurfaceEntry],
    manifest: dict,
    coverage_universe: dict[str, object],
    edition_id: str | None = None,
    year: int = 2024,
    generated_at: str | None = None,
    lookup_manifest: dict | None = None,
) -> tuple[str, dict[str, object]]:
    """Render the dictionary and return it with a summary block for the edition manifest."""
    context = resolve_context(manifest, year=year)
    rendered: list[tuple[SurfaceEntry, list[FieldSpec]]] = []
    for entry in surfaces:
        schema = read_schema(entry.path)
        vocabularies = read_vocabularies(entry.path, list(schema))
        specs = build_field_specs(schema=schema, vocabularies=vocabularies, context=context)
        rendered.append((entry, specs))
    markdown = render_markdown(
        surfaces=rendered,
        context=context,
        coverage_universe=coverage_universe,
        edition_id=edition_id,
        generated_at=generated_at,
        lookup_manifest=lookup_manifest,
    )
    summary = {
        "version": DICTIONARY_VERSION,
        "surfaces": {
            entry.geography: {"path": str(entry.path), "columns": len(specs)}
            for entry, specs in rendered
        },
        "documented_columns": sum(len(specs) for _, specs in rendered),
        "zcta_caveat_present": ZCTA_ANTI_ZIP_CAVEAT in markdown,
        "lookup_appendix_present": "## Appendix: static lookup index" in markdown,
        "lanes": list(context.lanes),
    }
    return markdown, summary


def rollup_geography_labels() -> dict[str, str]:
    return {geo.key: geo.source for geo in ROLLUP_GEOGRAPHIES}
