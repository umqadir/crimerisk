from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

from crimerisk.covariates.roads import LENGTH_CRS


OVERTURE_STAC_CATALOG_URL = "https://stac.overturemaps.org/catalog.json"
OVERTURE_S3_TEMPLATE = "s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
# Reproducibility fix (Overture taxonomy migration step 1): there is no default release anymore.
# The old default of the literal string "latest" resolved silently at run time and the *resolved*
# dated release was never recorded anywhere except inside the per-row `overture_release` column of
# the output parquet -- run-level summary/provenance JSON files recorded the unresolved request
# string ("latest") instead of what was actually pulled. Every caller must now pass an explicit
# dated release (e.g. "2026-06-17.0") or the explicit literal "latest"; `resolve_overture_release`
# raises a clear error on None/empty so a pull can never silently happen against an unrecorded
# vintage again. See `resolve_overture_release_with_metadata` for the STAC-catalog-backed
# resolution + provenance record that callers should persist alongside their outputs.
DEFAULT_OVERTURE_RELEASE: str | None = None
POINT_CRS = "EPSG:4326"

# Overture Places taxonomy migration (step 2 substance): Overture deprecated the `categories`
# property (removal in the September 2026 release) in favor of `basic_category` and `taxonomy`,
# with all three fields coexisting starting release 2026-06-17.0 (see
# https://docs.overturemaps.org/blog/2026/06/17/release-notes/ and
# https://docs.overturemaps.org/guides/places/taxonomy/). Older releases (including the
# 2026-05-20.0 vintage our currently-promoted parsed artifacts came from) do not carry these
# columns at all, so the pull SQL must only select them when the resolved release supports it.
_FIRST_RELEASE_WITH_BASIC_CATEGORY_AND_TAXONOMY = "2026-06-17.0"


def _release_supports_new_taxonomy_fields(release: str) -> bool:
    """True if `release` is on/after the first Overture release publishing basic_category/taxonomy."""
    release_date = str(release).strip().split(".")[0]
    threshold_date = _FIRST_RELEASE_WITH_BASIC_CATEGORY_AND_TAXONOMY.split(".")[0]
    if len(release_date) != len(threshold_date):
        # Not a recognizable YYYY-MM-DD.N release string -- fail closed (treat as unsupported)
        # rather than guess.
        return False
    return release_date >= threshold_date


# --- Legacy vocabulary (`categories.primary`) -------------------------------------------------
# ACTIVE path today. Untouched by this migration step -- current promoted artifacts and the
# production build depend on exactly this classification, keyed on `primary_category`
# (== `categories.primary`). Do not edit these sets as part of the taxonomy migration; any change
# here is a calibration-affecting change that belongs to the (separate, later) release-bump step.
_FOOD_SERVICE_EXACT: frozenset[str] = frozenset(
    {
        "restaurant",
        "coffee_shop",
        "cafe",
        "delicatessen",
        "bakery",
        "food_truck",
        "grocery_store",
        "supermarket",
    }
)
_NIGHTLIFE_ALCOHOL_EXACT: frozenset[str] = frozenset(
    {
        "bar",
        "pub",
        "irish_pub",
        "gastropub",
        "dance_club",
        "comedy_club",
        "strip_club",
        "casino",
        "liquor_store",
        "night_market",
    }
)
_CONVENIENCE_FUEL_EXACT: frozenset[str] = frozenset({"convenience_store", "gas_station"})
_LODGING_EXACT: frozenset[str] = frozenset({"hotel", "motel"})
_PARKING_EXACT: frozenset[str] = frozenset({"parking"})
_ATM_BANK_EXACT: frozenset[str] = frozenset({"atms", "banks", "bank_credit_union"})
_GROCERY_EXACT: frozenset[str] = frozenset({"grocery_store", "supermarket"})

OVERTURE_PLACE_GROUPS: tuple[str, ...] = (
    "overture_food_service",
    "overture_nightlife_alcohol",
    "overture_convenience_fuel",
    "overture_lodging",
    "overture_parking",
    "overture_atm_bank",
    "overture_grocery",
    "overture_consumer_destination_any",
)

_OVERTURE_SQL_FILTER = """
(
  categories.primary IN (
    'restaurant',
    'coffee_shop',
    'cafe',
    'delicatessen',
    'bakery',
    'food_truck',
    'grocery_store',
    'supermarket',
    'bar',
    'pub',
    'irish_pub',
    'gastropub',
    'dance_club',
    'comedy_club',
    'strip_club',
    'casino',
    'liquor_store',
    'night_market',
    'convenience_store',
    'gas_station',
    'hotel',
    'motel',
    'parking',
    'atms',
    'banks',
    'bank_credit_union'
  )
  OR categories.primary LIKE '%_restaurant'
  OR categories.primary LIKE '%_bar'
)
"""


# --- New vocabulary (`basic_category` / `taxonomy.primary`) -- DORMANT ------------------------
# Not used by any production code path yet. Derived from TWO sources, in this order of authority:
#   (1) Overture's own published old-to-new category correspondence ("Places Taxonomy" migration
#       sheet linked from https://docs.overturemaps.org/guides/places/taxonomy/, fetched
#       2026-07-08: https://docs.google.com/spreadsheets/d/1_i2S48zTDoHff0uX-d8Nes3bR-Xee8drx27Gyi80CQ0),
#       which gives, for every schema-canonical legacy `categories.primary` value, the
#       corresponding `taxonomy.primary` ("New Primary Category") and `basic_category` ("New
#       Basic Level Category").
#   (2) Direct empirical correction from a real coexistence-window pull (release 2026-06-17.0,
#       Rhode Island + Delaware, 21,709 places, 2026-07-08): the mapping sheet's "old" column is
#       Overture's *canonical schema* vocabulary, which is NOT the same as what real production
#       `categories.primary` data actually contains. Three legacy tokens our filter matches turned
#       out to have a real, non-dead presence in live data that the sheet does not document, and
#       one wildcard-matched token breaks the "_restaurant" suffix-carryover assumption. Source
#       (2) overrides source (1) wherever they conflict, because it is paired ground truth
#       (`categories.primary` and `taxonomy.primary` observed on the identical row for the
#       identical place) rather than a secondary reference document:
#     - "atms" (2,401 real places in the RI/DE sample) -> taxonomy.primary "atm",
#       basic_category "atm". The sheet's canonical old value is the singular "atm", which our
#       legacy filter does NOT match -- real data uses the plural "atms".
#     - "banks" (728 real places) -> taxonomy.primary "bank", basic_category
#       "bank_or_credit_union". Same pattern: sheet's canonical old value is singular "bank",
#       real data uses plural "banks".
#     - "supermarket" (345 real places) -> taxonomy.primary "grocery_store". The sheet does not
#       list "supermarket" as a category at all, old or new, but it is real: this doesn't change
#       classification (already covered by the "grocery_store" row below reaching the same
#       target), it only corrects the mapping table's documentation.
#     - "bagel_restaurant" (11 real places), caught by the legacy `%_restaurant` wildcard, does
#       NOT carry the suffix into the new vocabulary -- its real taxonomy.primary is
#       "bagel_shop" (basic_category "casual_eatery"). Recorded as an explicit wildcard
#       exception below since it cannot be derived by suffix matching.
#   Full before/after confusion-matrix evidence:
#   scratchpad/overture_migration/ri_de_confusion_matrix.txt (see also STATE.md dormant-candidate
#   section). Before this empirical correction, per-place disagreement between the legacy and
#   taxonomy classifiers was 14.46% overall and 76.08% within `overture_atm_bank` -- both far
#   over the ~2% review gate. After applying the three corrections above (see verification run),
#   disagreement drops to 0% on the RI/DE sample. This is a 2-state sample, not a national one;
#   re-running this same diff at national scale is a prerequisite for the release-bump step, not
#   something this dormant migration step can certify on its own.
#
# Design: `taxonomy.primary` is documented as "the most specific category label" -- the direct
# successor to the deprecated `categories.primary` -- so it is used as the matching key here,
# mirroring the granularity of the legacy exact-match sets above. `basic_category` is a coarser
# ~280-value convenience field with real collisions across our groups (e.g. both `grocery_store`
# and `liquor_store` roll up to the same `food_and_beverage_store` basic category), so it is
# selected for visibility/QA but is NOT used to discriminate between place groups.
OVERTURE_CATEGORY_TAXONOMY_MIGRATION_MAP: tuple[dict[str, str | None], ...] = (
    # place_group, legacy `categories.primary`, new `taxonomy.primary`, new `basic_category`, note
    {"place_group": "overture_food_service", "old_primary_category": "restaurant", "new_primary_category": "restaurant", "new_basic_category": "restaurant", "note": ""},
    {"place_group": "overture_food_service", "old_primary_category": "coffee_shop", "new_primary_category": "coffee_shop", "new_basic_category": "coffee_shop", "note": ""},
    {"place_group": "overture_food_service", "old_primary_category": "cafe", "new_primary_category": "cafe", "new_basic_category": "cafe", "note": ""},
    {"place_group": "overture_food_service", "old_primary_category": "delicatessen", "new_primary_category": "delicatessen", "new_basic_category": "casual_eatery", "note": ""},
    {"place_group": "overture_food_service", "old_primary_category": "bakery", "new_primary_category": "bakery", "new_basic_category": "casual_eatery", "note": ""},
    {"place_group": "overture_food_service", "old_primary_category": "food_truck", "new_primary_category": "food_truck_stand", "new_basic_category": "food_truck_stand", "note": "renamed"},
    {"place_group": "overture_food_service", "old_primary_category": "grocery_store", "new_primary_category": "grocery_store", "new_basic_category": "food_and_beverage_store", "note": ""},
    {"place_group": "overture_food_service", "old_primary_category": "supermarket", "new_primary_category": "grocery_store", "new_basic_category": "food_and_beverage_store", "note": "empirical correction (sheet omits it; real value maps to grocery_store)"},
    {"place_group": "overture_food_service", "old_primary_category": "bagel_restaurant", "new_primary_category": "bagel_shop", "new_basic_category": "casual_eatery", "note": "empirical wildcard exception -- breaks the '_restaurant' suffix-carryover rule"},
    {"place_group": "overture_food_service", "old_primary_category": "flatbread_restaurant", "new_primary_category": "flatbread_shop", "new_basic_category": "casual_eatery", "note": "empirical wildcard exception (national 2026-06-17.0 diff, 4 places) -- same '_restaurant'->'_shop' break as bagel"},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "bar", "new_primary_category": "bar", "new_basic_category": "bar", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "pub", "new_primary_category": "pub", "new_basic_category": "bar", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "irish_pub", "new_primary_category": "irish_pub", "new_basic_category": "bar", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "gastropub", "new_primary_category": "gastropub", "new_basic_category": "casual_eatery", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "dance_club", "new_primary_category": "dance_club", "new_basic_category": "dance_club", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "comedy_club", "new_primary_category": "comedy_club", "new_basic_category": "comedy_club", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "strip_club", "new_primary_category": "strip_club", "new_basic_category": "adult_entertainment_venue", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "casino", "new_primary_category": "casino", "new_basic_category": "casino", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "liquor_store", "new_primary_category": "liquor_store", "new_basic_category": "food_and_beverage_store", "note": ""},
    {"place_group": "overture_nightlife_alcohol", "old_primary_category": "night_market", "new_primary_category": "night_market", "new_basic_category": "market", "note": ""},
    {"place_group": "overture_convenience_fuel", "old_primary_category": "convenience_store", "new_primary_category": "convenience_store", "new_basic_category": "convenience_store", "note": ""},
    {"place_group": "overture_convenience_fuel", "old_primary_category": "gas_station", "new_primary_category": "gas_station", "new_basic_category": "gas_station", "note": ""},
    {"place_group": "overture_lodging", "old_primary_category": "hotel", "new_primary_category": "hotel", "new_basic_category": "hotel", "note": ""},
    {"place_group": "overture_lodging", "old_primary_category": "motel", "new_primary_category": "motel", "new_basic_category": "hotel", "note": ""},
    {"place_group": "overture_parking", "old_primary_category": "parking", "new_primary_category": "parking", "new_basic_category": "parking", "note": ""},
    {"place_group": "overture_atm_bank", "old_primary_category": "atms", "new_primary_category": "atm", "new_basic_category": "atm", "note": "empirical correction (sheet's canonical old value is singular 'atm', which the legacy filter never matches; real data uses plural 'atms')"},
    {"place_group": "overture_atm_bank", "old_primary_category": "banks", "new_primary_category": "bank", "new_basic_category": "bank_or_credit_union", "note": "empirical correction (sheet's canonical old value is singular 'bank'; real data uses plural 'banks')"},
    {"place_group": "overture_atm_bank", "old_primary_category": "bank_credit_union", "new_primary_category": "bank_or_credit_union", "new_basic_category": "bank_or_credit_union", "note": "renamed"},
    {"place_group": "overture_grocery", "old_primary_category": "grocery_store", "new_primary_category": "grocery_store", "new_basic_category": "food_and_beverage_store", "note": ""},
    {"place_group": "overture_grocery", "old_primary_category": "supermarket", "new_primary_category": "grocery_store", "new_basic_category": "food_and_beverage_store", "note": "empirical correction (sheet omits it; real value maps to grocery_store)"},
)

# The legacy `LIKE '%_restaurant'` / `LIKE '%_bar'` wildcard suffixes translate unchanged with one
# empirically-found exception (bagel_restaurant -> bagel_shop, added as an explicit row above):
# cross-checked all 223 `*_restaurant` and 25 `*_bar` legacy values against the official mapping
# sheet (no renames, no removals, no redirects there), then verified against the 92 distinct
# `*_restaurant` and 16 distinct `*_bar` values actually observed in the RI/DE coexistence-window
# sample -- only "bagel_restaurant" broke the suffix-carryover assumption. Applies the same suffix
# pattern to `taxonomy_primary` directly, same as the legacy classifier does for `primary_category`.


def _derive_taxonomy_exact_set(*, place_group: str) -> frozenset[str]:
    return frozenset(
        row["new_primary_category"]
        for row in OVERTURE_CATEGORY_TAXONOMY_MIGRATION_MAP
        if row["place_group"] == place_group and row["new_primary_category"]
    )


_FOOD_SERVICE_EXACT_TAXONOMY: frozenset[str] = _derive_taxonomy_exact_set(place_group="overture_food_service")
_NIGHTLIFE_ALCOHOL_EXACT_TAXONOMY: frozenset[str] = _derive_taxonomy_exact_set(place_group="overture_nightlife_alcohol")
_CONVENIENCE_FUEL_EXACT_TAXONOMY: frozenset[str] = _derive_taxonomy_exact_set(place_group="overture_convenience_fuel")
_LODGING_EXACT_TAXONOMY: frozenset[str] = _derive_taxonomy_exact_set(place_group="overture_lodging")
_PARKING_EXACT_TAXONOMY: frozenset[str] = _derive_taxonomy_exact_set(place_group="overture_parking")
_ATM_BANK_EXACT_TAXONOMY: frozenset[str] = _derive_taxonomy_exact_set(place_group="overture_atm_bank")
_GROCERY_EXACT_TAXONOMY: frozenset[str] = _derive_taxonomy_exact_set(place_group="overture_grocery")

# --- Release-bump step: taxonomy-keyed row admission (ACTIVE for release >= 2026-06-17.0) -----
# The national certification (see `1cb34b8`, 0.00000% classification disagreement on
# legacy-admitted rows) covers classification only. Row ADMISSION historically keyed on legacy
# `categories.primary` regardless of release; from here on, releases that publish
# `taxonomy.primary` (>= `_FIRST_RELEASE_WITH_BASIC_CATEGORY_AND_TAXONOMY`) admit rows by
# `taxonomy.primary` instead, using the exact same place-group exact-match sets (derived above
# from `OVERTURE_CATEGORY_TAXONOMY_MIGRATION_MAP` via `_derive_taxonomy_exact_set`) plus the same
# `%_restaurant` / `%_bar` suffix wildcards the legacy filter uses, applied to `taxonomy.primary`.
# This can admit a different place-id set than the legacy filter would (e.g. Overture places
# whose `categories.primary` never matched but whose `taxonomy.primary` now does, or vice versa)
# -- that delta is measured, not assumed away; see the admitted-set delta report run alongside
# this change.
_ALL_TAXONOMY_EXACT_UNION: frozenset[str] = (
    _FOOD_SERVICE_EXACT_TAXONOMY
    | _NIGHTLIFE_ALCOHOL_EXACT_TAXONOMY
    | _CONVENIENCE_FUEL_EXACT_TAXONOMY
    | _LODGING_EXACT_TAXONOMY
    | _PARKING_EXACT_TAXONOMY
    | _ATM_BANK_EXACT_TAXONOMY
    | _GROCERY_EXACT_TAXONOMY
)


def _build_taxonomy_sql_filter() -> str:
    values = ",\n    ".join(f"'{value}'" for value in sorted(_ALL_TAXONOMY_EXACT_UNION))
    return f"""
(
  taxonomy.primary IN (
    {values}
  )
  OR taxonomy.primary LIKE '%_restaurant'
  OR taxonomy.primary LIKE '%_bar'
)
"""


_OVERTURE_SQL_FILTER_TAXONOMY = _build_taxonomy_sql_filter()


def _taxonomy_admission_lane_active(release: str) -> bool:
    """True if row admission for `release` should key on `taxonomy.primary` (release-bump step).

    Gated on the same threshold as new-field support (`_release_supports_new_taxonomy_fields`):
    `taxonomy.primary` must exist in the release's parquet schema before it can be filtered on,
    and the design decision is to flip the admission lane on for every release that carries it,
    starting 2026-06-17.0.
    """
    return _release_supports_new_taxonomy_fields(release)


@dataclass(frozen=True)
class OverturePlacesBuildConfig:
    release: str | None = DEFAULT_OVERTURE_RELEASE
    s3_region: str = "us-west-2"
    min_confidence: float = 0.75
    bbox_buffer_deg: float = 0.03
    point_crs: str = POINT_CRS
    length_crs: str = LENGTH_CRS
    within_km: float = 1.0


def resolve_overture_release(*, release: str | None, s3_region: str = "us-west-2") -> str:
    """Resolve a release request to an explicit dated release string.

    `release` must be either an explicit dated release (e.g. "2026-06-17.0") or the literal
    "latest" (case-insensitive), which is resolved via the Overture STAC catalog. `None` or an
    empty string raises -- there is no silent default. Use
    `resolve_overture_release_with_metadata` when the resolved value (and how it was resolved)
    needs to be persisted for provenance.
    """
    requested = str(release).strip() if release is not None else ""
    if not requested:
        raise ValueError(
            "Overture release is required and has no default: pass an explicit dated release "
            "(e.g. '2026-06-17.0') or the literal 'latest'. 'latest' is still accepted but is "
            "resolved via the STAC catalog and the caller is responsible for recording the "
            "resolved value (see resolve_overture_release_with_metadata) so pulled artifacts "
            "carry honest release provenance."
        )
    if requested.lower() != "latest":
        return requested
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"SET s3_region = '{s3_region}';")
        latest = con.execute(f"SELECT latest FROM '{OVERTURE_STAC_CATALOG_URL}'").fetchone()
    finally:
        con.close()
    if not latest or not latest[0]:
        raise RuntimeError("Unable to resolve latest Overture release from STAC catalog")
    return str(latest[0]).strip()


def resolve_overture_release_with_metadata(
    *, release: str | None, s3_region: str = "us-west-2"
) -> tuple[str, dict[str, object]]:
    """Resolve a release request and return a reviewable provenance record alongside it.

    Callers that pull data (scripts, not this module's lower-level helpers) should call this
    once per run and persist the returned metadata dict into their run summary JSON, so the
    artifact's actual pulled vintage is recorded honestly instead of inferred after the fact.
    """
    requested = "" if release is None else str(release).strip()
    resolved = resolve_overture_release(release=release, s3_region=s3_region)
    metadata = {
        "requested_release": requested or None,
        "resolved_release": resolved,
        "resolved_via": "stac_catalog_latest" if requested.lower() == "latest" else "explicit",
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "stac_catalog_url": OVERTURE_STAC_CATALOG_URL,
        "supports_basic_category_and_taxonomy": _release_supports_new_taxonomy_fields(resolved),
    }
    return resolved, metadata


def classify_overture_place_groups(primary_category: object) -> list[str]:
    """Classification path keyed on the legacy `categories.primary` value.

    ACTIVE (production `matched_groups`) for releases before 2026-06-17.0. For releases on/after
    that threshold, `classify_overture_place_groups_taxonomy` is the production path instead and
    this function's output is carried as the parallel `matched_groups_legacy` provenance column
    (see `fetch_overture_places_for_bbox`). Untouched by the taxonomy migration -- do not change
    group membership here without a calibration review.
    """
    category = str(primary_category or "").strip().lower()
    if not category:
        return []
    groups: list[str] = []
    if category in _FOOD_SERVICE_EXACT or category.endswith("_restaurant"):
        groups.append("overture_food_service")
    if category in _NIGHTLIFE_ALCOHOL_EXACT or category.endswith("_bar"):
        groups.append("overture_nightlife_alcohol")
    if category in _CONVENIENCE_FUEL_EXACT:
        groups.append("overture_convenience_fuel")
    if category in _LODGING_EXACT:
        groups.append("overture_lodging")
    if category in _PARKING_EXACT:
        groups.append("overture_parking")
    if category in _ATM_BANK_EXACT:
        groups.append("overture_atm_bank")
    if category in _GROCERY_EXACT:
        groups.append("overture_grocery")
    if groups:
        groups.append("overture_consumer_destination_any")
    return sorted(set(groups))


def classify_overture_place_groups_taxonomy(taxonomy_primary: object) -> list[str]:
    """Classification path keyed on the new `taxonomy.primary` value.

    Mirrors `classify_overture_place_groups` exactly, translated through
    `OVERTURE_CATEGORY_TAXONOMY_MIGRATION_MAP`. ACTIVE (production `matched_groups`) for releases
    on/after 2026-06-17.0, certified nationally at 0.00000% disagreement with the legacy path on
    legacy-admitted rows (see `1cb34b8`). `taxonomy_primary` is None/NaN for any place pulled from
    a release that predates basic_category/taxonomy (pre-2026-06-17.0), in which case this
    returns [].
    """
    category = str(taxonomy_primary or "").strip().lower()
    if not category:
        return []
    groups: list[str] = []
    if category in _FOOD_SERVICE_EXACT_TAXONOMY or category.endswith("_restaurant"):
        groups.append("overture_food_service")
    if category in _NIGHTLIFE_ALCOHOL_EXACT_TAXONOMY or category.endswith("_bar"):
        groups.append("overture_nightlife_alcohol")
    if category in _CONVENIENCE_FUEL_EXACT_TAXONOMY:
        groups.append("overture_convenience_fuel")
    if category in _LODGING_EXACT_TAXONOMY:
        groups.append("overture_lodging")
    if category in _PARKING_EXACT_TAXONOMY:
        groups.append("overture_parking")
    if category in _ATM_BANK_EXACT_TAXONOMY:
        groups.append("overture_atm_bank")
    if category in _GROCERY_EXACT_TAXONOMY:
        groups.append("overture_grocery")
    if groups:
        groups.append("overture_consumer_destination_any")
    return sorted(set(groups))


def _overture_connection(*, s3_region: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute(f"SET s3_region = '{s3_region}';")
    return con


def fetch_overture_places_for_bbox(
    *,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    cfg: OverturePlacesBuildConfig = OverturePlacesBuildConfig(),
) -> pd.DataFrame:
    release = resolve_overture_release(release=cfg.release, s3_region=cfg.s3_region)
    path = OVERTURE_S3_TEMPLATE.format(release=release)
    # Taxonomy migration, release-bump step: `basic_category`/`taxonomy` only exist in the
    # parquet schema from release 2026-06-17.0 onward (coexistence window before `categories`
    # removal in the September 2026 release). Selecting them against an older release would raise
    # a DuckDB binder error, so they are only requested when the resolved release supports them;
    # older releases get NULL columns instead so the output schema is stable either way.
    #
    # Row admission now depends on the same threshold (`_taxonomy_admission_lane_active`): for
    # release < 2026-06-17.0 this is byte-identical to the pre-migration behavior (admit by legacy
    # `categories.primary` only). For release >= 2026-06-17.0, admission keys on the new
    # `taxonomy.primary` vocabulary instead (`_OVERTURE_SQL_FILTER_TAXONOMY`), which was certified
    # nationally at 0.00000% classification disagreement with the legacy path on legacy-admitted
    # rows (see `1cb34b8`) and can admit a different place-id set (measured separately, not
    # assumed away).
    supports_new_taxonomy = _release_supports_new_taxonomy_fields(release)
    taxonomy_lane_active = _taxonomy_admission_lane_active(release)
    new_taxonomy_select = (
        "basic_category,\n          taxonomy.primary AS taxonomy_primary,"
        if supports_new_taxonomy
        else "CAST(NULL AS VARCHAR) AS basic_category,\n          CAST(NULL AS VARCHAR) AS taxonomy_primary,"
    )
    admission_filter = _OVERTURE_SQL_FILTER_TAXONOMY if taxonomy_lane_active else _OVERTURE_SQL_FILTER
    con = _overture_connection(s3_region=cfg.s3_region)
    try:
        query = f"""
        SELECT
          id AS overture_place_id,
          names.primary AS place_name,
          categories.primary AS primary_category,
          {new_taxonomy_select}
          confidence,
          addresses[1].country AS country,
          addresses[1].region AS region,
          ST_X(geometry) AS lon,
          ST_Y(geometry) AS lat
        FROM read_parquet('{path}', filename=true, hive_partitioning=1)
        WHERE bbox.xmin BETWEEN {float(minx)} AND {float(maxx)}
          AND bbox.ymin BETWEEN {float(miny)} AND {float(maxy)}
          AND addresses[1].country = 'US'
          AND confidence >= {float(cfg.min_confidence)}
          AND {admission_filter}
        """
        out = con.execute(query).fetchdf()
    finally:
        con.close()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "overture_place_id",
                "place_name",
                "primary_category",
                "basic_category",
                "taxonomy_primary",
                "confidence",
                "country",
                "region",
                "lon",
                "lat",
                "matched_groups",
                "matched_group_count",
                "matched_groups_legacy",
                "matched_group_count_legacy",
                "matched_groups_taxonomy",
                "matched_group_count_taxonomy",
                "overture_release",
            ]
        )
    if taxonomy_lane_active:
        # ACTIVE (release >= 2026-06-17.0): production `matched_groups` comes from the taxonomy
        # classifier, keyed on the same `taxonomy_primary` column that gated row admission above.
        out["matched_groups"] = out["taxonomy_primary"].map(classify_overture_place_groups_taxonomy)
        out = out[out["matched_groups"].map(bool)].copy()
        out["matched_group_count"] = out["matched_groups"].map(len).astype(int)
        # Provenance: what the legacy `categories.primary` path would have classified these same
        # (taxonomy-admitted) rows as. Not consumed by `aggregate_overture_places_to_block_groups`.
        out["matched_groups_legacy"] = out["primary_category"].map(classify_overture_place_groups)
        out["matched_group_count_legacy"] = out["matched_groups_legacy"].map(len).astype(int)
        # Kept for schema continuity with the pre-release-bump dormant columns; identical to the
        # now-active matched_groups/matched_group_count in this lane.
        out["matched_groups_taxonomy"] = out["matched_groups"]
        out["matched_group_count_taxonomy"] = out["matched_group_count"]
    else:
        # ACTIVE (release < 2026-06-17.0): unchanged -- governs row inclusion and every
        # downstream production feature, byte-identical to pre-migration behavior.
        out["matched_groups"] = out["primary_category"].map(classify_overture_place_groups)
        out = out[out["matched_groups"].map(bool)].copy()
        out["matched_group_count"] = out["matched_groups"].map(len).astype(int)
        # DORMANT: informational only, keyed on the new taxonomy (NULL/[] pre-2026-06-17.0 since
        # taxonomy_primary itself is NULL there). Not consumed by
        # `aggregate_overture_places_to_block_groups` or any production feature build.
        out["matched_groups_taxonomy"] = out["taxonomy_primary"].map(classify_overture_place_groups_taxonomy)
        out["matched_group_count_taxonomy"] = out["matched_groups_taxonomy"].map(len).astype(int)
        # Not applicable pre-release-bump -- the active path already *is* the legacy path.
        out["matched_groups_legacy"] = None
        out["matched_group_count_legacy"] = None
    out["overture_release"] = release
    return out.reset_index(drop=True)


def fetch_overture_places_for_query_groups(
    *,
    block_groups: gpd.GeoDataFrame,
    query_group_col: str,
    cfg: OverturePlacesBuildConfig = OverturePlacesBuildConfig(),
) -> pd.DataFrame:
    if block_groups.empty:
        return pd.DataFrame(
            columns=[
                "overture_place_id",
                "place_name",
                "primary_category",
                "basic_category",
                "taxonomy_primary",
                "confidence",
                "country",
                "region",
                "lon",
                "lat",
                "matched_groups",
                "matched_group_count",
                "matched_groups_legacy",
                "matched_group_count_legacy",
                "matched_groups_taxonomy",
                "matched_group_count_taxonomy",
                "overture_release",
                query_group_col,
            ]
        )
    query_bg = block_groups[[query_group_col, "geometry"]].dropna(subset=[query_group_col]).copy()
    query_bg = query_bg.to_crs(cfg.point_crs)
    frames: list[pd.DataFrame] = []
    for query_group, grp in query_bg.groupby(query_group_col, sort=True):
        minx, miny, maxx, maxy = grp.total_bounds
        buffered = fetch_overture_places_for_bbox(
            minx=float(minx) - float(cfg.bbox_buffer_deg),
            miny=float(miny) - float(cfg.bbox_buffer_deg),
            maxx=float(maxx) + float(cfg.bbox_buffer_deg),
            maxy=float(maxy) + float(cfg.bbox_buffer_deg),
            cfg=cfg,
        )
        if buffered.empty:
            continue
        buffered[query_group_col] = str(query_group)
        frames.append(buffered)
    if not frames:
        return pd.DataFrame(
            columns=[
                "overture_place_id",
                "place_name",
                "primary_category",
                "basic_category",
                "taxonomy_primary",
                "confidence",
                "country",
                "region",
                "lon",
                "lat",
                "matched_groups",
                "matched_group_count",
                "matched_groups_legacy",
                "matched_group_count_legacy",
                "matched_groups_taxonomy",
                "matched_group_count_taxonomy",
                "overture_release",
                query_group_col,
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["overture_place_id", query_group_col], keep="first").reset_index(drop=True)
    return out


def aggregate_overture_places_to_block_groups(
    *,
    points: pd.DataFrame,
    block_groups: gpd.GeoDataFrame,
    query_group_col: str | None = None,
    cfg: OverturePlacesBuildConfig = OverturePlacesBuildConfig(),
) -> pd.DataFrame:
    key_cols = ["bg_id", "tract_id", "state_fips", "county_fips"]
    if query_group_col is not None:
        key_cols = [query_group_col, *key_cols]
    base = block_groups[key_cols + ["geometry"]].drop_duplicates(
        subset=[c for c in key_cols if c != query_group_col] if query_group_col is not None else key_cols
    ).copy()
    if base.empty:
        return pd.DataFrame(columns=key_cols)
    metric_bg = base.to_crs(cfg.length_crs).copy()
    metric_bg["land_area_sqkm_metric"] = metric_bg.geometry.area / 1_000_000.0
    out = metric_bg[key_cols].copy()

    default_release = resolve_overture_release(release=cfg.release, s3_region=cfg.s3_region)
    if points.empty:
        for group in OVERTURE_PLACE_GROUPS:
            out[f"{group}_count"] = 0.0
            out[f"{group}_density_sqkm"] = 0.0
            out[f"log_{group}_density_sqkm"] = 0.0
            out[f"{group}_present"] = 0.0
            out[f"nearest_{group}_km"] = np.nan
            out[f"{group}_within_{int(cfg.within_km)}km"] = 0.0
        out["overture_min_confidence"] = float(cfg.min_confidence)
        out["overture_release"] = default_release
        return out.sort_values(key_cols, kind="stable").reset_index(drop=True)

    points_gdf = gpd.GeoDataFrame(
        points.copy(),
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs=cfg.point_crs,
    ).to_crs(cfg.length_crs)
    points_gdf = points_gdf.explode("matched_groups", ignore_index=True).rename(columns={"matched_groups": "feature_group"})
    points_gdf = points_gdf[points_gdf["feature_group"].notna()].copy()
    grouped_bg = metric_bg.groupby(query_group_col, sort=True) if query_group_col is not None else [(None, metric_bg)]
    frames: list[pd.DataFrame] = []
    for query_group, bg_group in grouped_bg:
        group_out = bg_group[key_cols].copy()
        group_out["land_area_sqkm_metric"] = bg_group["land_area_sqkm_metric"].to_numpy(dtype=float)
        group_centroids = gpd.GeoDataFrame(
            bg_group[key_cols].copy(),
            geometry=bg_group.geometry.representative_point(),
            crs=cfg.length_crs,
        )
        group_points_all = (
            points_gdf[points_gdf[query_group_col].astype(str).eq(str(query_group))].copy()
            if query_group_col is not None
            else points_gdf.copy()
        )
        for group in OVERTURE_PLACE_GROUPS:
            group_points = group_points_all[group_points_all["feature_group"].astype(str).eq(group)].copy()
            count_col = f"{group}_count"
            density_col = f"{group}_density_sqkm"
            log_density_col = f"log_{group}_density_sqkm"
            present_col = f"{group}_present"
            nearest_col = f"nearest_{group}_km"
            within_col = f"{group}_within_{int(cfg.within_km)}km"

            group_out[count_col] = 0.0
            group_out[density_col] = 0.0
            group_out[log_density_col] = 0.0
            group_out[present_col] = 0.0
            group_out[nearest_col] = np.nan
            group_out[within_col] = 0.0

            if group_points.empty:
                continue

            joined = gpd.sjoin(
                group_points[["feature_group", "geometry"]],
                bg_group[key_cols + ["geometry"]],
                how="inner",
                predicate="within",
            )
            if not joined.empty:
                counts = joined.groupby(key_cols, dropna=False).size().rename(count_col).reset_index()
                group_out = group_out.drop(columns=[count_col], errors="ignore").merge(
                    counts,
                    on=key_cols,
                    how="left",
                )
                group_out[count_col] = pd.to_numeric(group_out[count_col], errors="coerce").fillna(0.0)
                group_out[density_col] = np.where(
                    group_out["land_area_sqkm_metric"].to_numpy(dtype=float) > 0,
                    group_out[count_col].to_numpy(dtype=float) / group_out["land_area_sqkm_metric"].to_numpy(dtype=float),
                    0.0,
                )
                group_out[density_col] = pd.to_numeric(group_out[density_col], errors="coerce").fillna(0.0)
                group_out[log_density_col] = np.log1p(group_out[density_col].clip(lower=0.0))
                group_out[present_col] = group_out[count_col].gt(0).astype(float)

            nearest = gpd.sjoin_nearest(
                group_centroids,
                group_points[["geometry"]],
                how="left",
                distance_col="_nearest_meters",
            )
            nearest = nearest.groupby(key_cols, dropna=False, as_index=False)["_nearest_meters"].min()
            nearest[nearest_col] = pd.to_numeric(nearest["_nearest_meters"], errors="coerce") / 1000.0
            nearest[within_col] = nearest[nearest_col].le(float(cfg.within_km)).fillna(False).astype(float)
            group_out = group_out.drop(columns=[nearest_col, within_col], errors="ignore").merge(
                nearest[key_cols + [nearest_col, within_col]],
                on=key_cols,
                how="left",
            )
            group_out[within_col] = pd.to_numeric(group_out[within_col], errors="coerce").fillna(0.0)
        frames.append(group_out.drop(columns=["land_area_sqkm_metric"], errors="ignore"))
    out = pd.concat(frames, ignore_index=True) if frames else out

    out["overture_min_confidence"] = float(cfg.min_confidence)
    out["overture_release"] = (
        points["overture_release"].astype("string").dropna().iloc[0]
        if "overture_release" in points.columns and points["overture_release"].notna().any()
        else default_release
    )
    return out.sort_values(key_cols, kind="stable").reset_index(drop=True)
