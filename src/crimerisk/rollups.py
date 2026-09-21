"""Exact rollups of the block-group surface to county, CBSA, ZCTA and state support.

One rule decides everything in this module: **counts are summed, rates and indexes are
recomputed.** A rolled-up rate is `100000 * Sum(count) / Sum(denominator)` over the unit's member
block groups and a rolled-up index is `100 * rate / national_rate` against the same stored national
normalizer the block-group surface published under. No published rate, index or composite is ever
averaged across block groups: an average of ratios is not the ratio of the sums, and the surface's
count-derived index policy (docs/STATE.md, "Count-derived index policy") requires every published
value to be recomputable by a reader from published counts and published denominators at the
support it is quoted at.

Four rules are load-bearing and enforced rather than trusted:

* **Conservation is exact.** Every rollup's national count total equals the block-group surface's,
  to floating-point summation epsilon, offense by offense. `conservation_report` measures it and
  `assert_conservation` fails the build on a miss. Where a geography does not cover the whole
  country -- CBSAs cover only the 1,916 metro/micro counties of the 2023 OMB delineation, and 2020
  ZCTAs do not tile the whole land area -- the uncovered mass is reported as a named residual and
  the identity checked is `rolled + outside_universe == national`, never a silently dropped tail.

* **Block groups that split across a target unit are overlap-weighted, not assigned whole.**
  Counties, states and tracts are GEOID prefixes of the block group, so their weights are exactly
  1. ZCTAs are not: they are built from 2020 tabulation blocks and cut across block groups. The
  ZCTA crosswalk therefore weights each (block group, ZCTA) pair by the 2020 census population of
  the blocks in the intersection (housing units, then land area, then equal split, where a block
  group has no population), and renormalizes the weights to sum to 1 within each block group so
  the block group's whole count lands somewhere.

* **Publication gates are reapplied at the rollup's own support, never inherited.** A block group
  suppressed for insufficient exposure does not suppress its county; a county whose rolled-up
  denominator is below the same published floor is suppressed in its own right. The floors are
  read off the surface's own floor columns rather than transcribed here.

* **The count-first composites are recomputed at that support**, through `composites.py`, from the
  rolled-up counts over the rolled-up common denominator. That includes recomputing each
  composite's reference rate over the rollup's own publishable universe, which is what makes the
  published value at that support recomputable from the published table.

The rare-offense support policy (`allocation.RARE_OFFENSE_TRACT_SUPPORT`: murder and rape publish
a point index at census tract and coarser, never at block group) is satisfied at every rollup
geography here, so murder and rape publish -- gated by the same denominator floors as every other
offense, which is exactly the gate the tract surface publishes them under.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from crimerisk.composites import (
    COMMON_DENOMINATOR_COLUMN,
    COUNT_FIRST_AGGREGATE_INDEX_FIELDS,
    HARM_WEIGHTED_COUNT_COLUMN,
    PERSONAL_RELATIVE_SCORE_COLUMN,
    PROPERTY_RELATIVE_SCORE_COLUMN,
    CompositeRuntime,
    apply_count_first_composites,
    burden_publishable,
    composite_normalizers,
)
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths


ROLLUP_VERSION = "exact_rollups_v1"
RATE_PER_100K = 100000.0

# The coverage universe is READ OFF THE SURFACE, never asserted. PLAN.md item 8 requires "AK+HI or
# an explicitly named coverage universe"; naming it means naming what the surface actually holds,
# so every edition states its own universe and a surface that later gains or loses a state changes
# the published statement rather than silently contradicting it.
CONTIGUOUS_EXCLUDED_STATE_FIPS: dict[str, str] = {"02": "Alaska", "15": "Hawaii"}
DISTRICT_OF_COLUMBIA_FIPS = "11"

PERSONAL_OFFENSES: tuple[str, ...] = ("murder", "rape", "robbery", "aggravated_assault")
PROPERTY_OFFENSES: tuple[str, ...] = ("burglary", "larceny", "motor_vehicle_theft")

BLOCK_GROUP_ID_COLUMN = "block_group_geoid"
OUTSIDE_UNIVERSE_ID = "__outside_universe__"

# --- the ZCTA anti-ZIP caveat -------------------------------------------------------------
#
# PLAN.md item 8 asks for "AGS's own anti-ZIP caveat" on the ZCTA lane. Neither AGS methodology
# document held in `docs/` (2025B, 2026A) contains one, so this is a project-authored caveat
# sourced to the Census Bureau's own ZCTA definition rather than a quotation attributed to AGS.
# It is emitted VERBATIM, from this one constant, everywhere a ZCTA field is published or
# documented.
ZCTA_ANTI_ZIP_CAVEAT = (
    "ZCTAs are not ZIP Codes. A ZIP Code Tabulation Area is a U.S. Census Bureau statistical area "
    "built from 2020 census tabulation blocks to approximate the delivery area of a USPS ZIP Code. "
    "USPS ZIP Codes are delivery-route identifiers rather than areas, they change without notice, "
    "not every ZIP Code has a ZCTA, ZCTAs do not nest within any other census geography, and "
    "point, PO Box and non-addressable ZIP Codes have no ZCTA at all. A value published at ZCTA "
    "support is not a value for a ZIP Code and must not be joined to ZIP-coded records as though "
    "it were. Source: U.S. Census Bureau, ZIP Code Tabulation Areas (ZCTAs)."
)


# --- geography specifications --------------------------------------------------------------


@dataclass(frozen=True)
class RollupGeography:
    """One rollup target: how a block group maps into it, and what the mapping is sourced from."""

    key: str
    id_column: str
    name_column: str | None
    nests_in_block_group: bool
    covers_universe: bool
    source: str
    caveat: str | None = None

    @property
    def support(self) -> str:
        return self.key


COUNTY = RollupGeography(
    key="county",
    id_column="county_geoid",
    name_column="county_name",
    nests_in_block_group=True,
    covers_universe=True,
    source="2020 Census block group GEOID, first 5 digits (state+county FIPS)",
)
CBSA = RollupGeography(
    key="cbsa",
    id_column="cbsa_code",
    name_column="cbsa_title",
    nests_in_block_group=True,
    covers_universe=False,
    source="2023 OMB CBSA delineation (configs/county_to_cbsa_2023.csv), joined on county FIPS",
)
ZCTA = RollupGeography(
    key="zcta",
    id_column="zcta5",
    name_column=None,
    nests_in_block_group=False,
    covers_universe=False,
    source=(
        "Census 2020 ZCTA-to-block relationship file "
        "(tab20_zcta520_tabblock20_natl.txt), aggregated to block groups and weighted by "
        "2020 census block population"
    ),
    caveat=ZCTA_ANTI_ZIP_CAVEAT,
)
STATE = RollupGeography(
    key="state",
    id_column="state_fips",
    name_column="state_abbr",
    nests_in_block_group=True,
    covers_universe=True,
    source="2020 Census block group GEOID, first 2 digits (state FIPS)",
)

ROLLUP_GEOGRAPHIES: tuple[RollupGeography, ...] = (COUNTY, CBSA, ZCTA, STATE)
ROLLUP_GEOGRAPHIES_BY_KEY: dict[str, RollupGeography] = {geo.key: geo for geo in ROLLUP_GEOGRAPHIES}


def geography(key: str) -> RollupGeography:
    try:
        return ROLLUP_GEOGRAPHIES_BY_KEY[str(key)]
    except KeyError as exc:
        raise KeyError(
            f"{key!r} is not a rollup geography; known: {sorted(ROLLUP_GEOGRAPHIES_BY_KEY)}"
        ) from exc


# --- column plumbing -------------------------------------------------------------------------


def expected_count_column(offense: str) -> str:
    return f"expected_count_{offense}"


def primary_denominator_column(offense: str) -> str:
    return f"primary_denominator_{offense}"


def primary_national_rate_column(offense: str) -> str:
    return f"primary_national_rate_per_100k_{offense}"


def resident_national_rate_column(offense: str) -> str:
    return f"resident_national_rate_per_100k_{offense}"


# What a rollup reads off the block-group surface. Everything else on the 792-column surface is
# either a per-block-group diagnostic (no rollup meaning) or recomputable from these.
SUMMED_BASE_COLUMNS: tuple[str, ...] = (
    "households_total",
    "land_area_sq_mi",
    COMMON_DENOMINATOR_COLUMN,
)
FLOOR_COLUMNS: tuple[str, ...] = (
    "eb_hard_min_denominator",
    "non_residential_household_floor",
    "person_exposure_denominator_floor",
    "mvt_vehicle_exposure_denominator_floor",
    "zero_resident_opportunity_rate_floor",
)

# The one publication constant that is NOT a column on the surface: the resident head-count under
# which the near-zero-resident opportunity-rate rule demands the larger opportunity floor. It is
# published in the build manifest (`resolved_config.zero_resident_opportunity_rate_policy.
# resident_threshold`); `zero_resident_resident_threshold` reads it from there and falls back to
# this transcription only when a build ships no manifest.
DEFAULT_ZERO_RESIDENT_RESIDENT_THRESHOLD = 50.0
ZERO_RESIDENT_POLICY_KEY = "zero_resident_opportunity_rate_policy"


def zero_resident_resident_threshold(manifest: dict | None) -> float:
    """The manifest's own resident threshold for the near-zero-resident opportunity-rate rule."""
    policy = ((manifest or {}).get("resolved_config") or {}).get(ZERO_RESIDENT_POLICY_KEY) or {}
    value = policy.get("resident_threshold")
    if value is None:
        return float(DEFAULT_ZERO_RESIDENT_RESIDENT_THRESHOLD)
    return float(value)


def population_column(year: int) -> str:
    """Target-year resident population column carried by an annual surface."""
    return f"population_{int(year)}"


def _surface_population_column(frame: pd.DataFrame) -> str:
    candidates = [
        column
        for column in frame.columns
        if column.startswith("population_") and column.removeprefix("population_").isdigit()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "annual surface must carry exactly one target-year population_<year> column; "
            f"found {sorted(candidates)}"
        )
    return candidates[0]


def source_columns(*, year: int = 2024) -> list[str]:
    """Every block-group column a rollup needs, in read order."""
    columns = [
        BLOCK_GROUP_ID_COLUMN,
        "state_fips",
        population_column(year),
        *SUMMED_BASE_COLUMNS,
        *FLOOR_COLUMNS,
    ]
    for offense in OFFENSES_7:
        columns += [
            expected_count_column(offense),
            primary_denominator_column(offense),
            f"primary_denominator_type_{offense}",
            primary_national_rate_column(offense),
            resident_national_rate_column(offense),
        ]
    return columns


def read_block_group_surface(path: Path, *, year: int = 2024) -> pd.DataFrame:
    """The rollup inputs only. Parameterized on the surface path so the whole product-operations
    lane regenerates against whichever candidate (or promoted) surface is current."""
    frame = pd.read_parquet(path, columns=source_columns(year=year))
    frame[BLOCK_GROUP_ID_COLUMN] = frame[BLOCK_GROUP_ID_COLUMN].astype("string").str.zfill(12)
    return frame


def _scalar(frame: pd.DataFrame, column: str, *, what: str) -> float:
    """A surface-wide constant, read off the surface rather than transcribed into this file."""
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{what} ({column}) is not a single surface-wide value: found {len(values)} distinct "
            "values; a rollup cannot reapply a gate whose constant varies by row"
        )
    return float(values[0])


def _denominator_type(frame: pd.DataFrame, offense: str) -> str:
    values = frame[f"primary_denominator_type_{offense}"].astype("string").dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"primary_denominator_type_{offense} is not a single surface-wide value: {sorted(values)}"
        )
    return str(values[0])


def coverage_universe(surface: pd.DataFrame) -> dict[str, object]:
    """The universe this surface covers, named from the surface's own state list.

    Territories are outside the FBI Part-I control lane and carry no block-group row at all, so
    they are absent rather than zero; Alaska and Hawaii are named individually when absent because
    "nationwide" and "the contiguous states" are different claims.
    """
    states = sorted(surface["state_fips"].astype("string").str.zfill(2).dropna().unique().tolist())
    absent = {
        fips: name for fips, name in CONTIGUOUS_EXCLUDED_STATE_FIPS.items() if fips not in states
    }
    state_count = len([fips for fips in states if fips != DISTRICT_OF_COLUMBIA_FIPS])
    has_dc = DISTRICT_OF_COLUMBIA_FIPS in states
    if absent:
        label = f"{state_count} states"
        if set(absent) == set(CONTIGUOUS_EXCLUDED_STATE_FIPS):
            label = f"{state_count} contiguous states"
        description = (
            f"{label}{' + District of Columbia' if has_dc else ''}; "
            f"absent: {', '.join(sorted(absent.values()))}; "
            "Puerto Rico and the island areas out of scope (no FBI Part-I control lane)"
        )
    else:
        description = (
            f"{state_count} states{' + District of Columbia' if has_dc else ''}; "
            "Puerto Rico and the island areas out of scope (no FBI Part-I control lane)"
        )
    return {
        "description": description,
        "state_fips": states,
        "state_count": state_count,
        "district_of_columbia": bool(has_dc),
        "named_absent_states": absent,
        "block_groups": int(len(surface)),
        "counties": int(surface[BLOCK_GROUP_ID_COLUMN].str.slice(0, 5).nunique()),
    }


def national_rates(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """The stored national normalizers, per offense, for the primary and resident lanes."""
    return {
        "primary": {
            offense: _scalar(frame, primary_national_rate_column(offense), what=f"{offense} national rate")
            for offense in OFFENSES_7
        },
        "resident": {
            offense: _scalar(
                frame, resident_national_rate_column(offense), what=f"{offense} resident national rate"
            )
            for offense in OFFENSES_7
        },
    }


# --- crosswalks ------------------------------------------------------------------------------


def load_county_names(paths: RepoPaths) -> pd.DataFrame:
    path = paths.repo_root / "configs" / "county_names_2020.csv"
    frame = pd.read_csv(path, dtype=str)
    frame["county_fips"] = frame["county_fips"].astype("string").str.zfill(5)
    frame["county_name"] = frame["county_name"].astype("string")
    return frame[["county_fips", "county_name"]].drop_duplicates("county_fips").reset_index(drop=True)


def load_state_abbreviations(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "reference" / "jurisdiction_master.parquet"
    frame = pd.read_parquet(path, columns=["state_fips", "state_abbr"])
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
    frame["state_abbr"] = frame["state_abbr"].astype("string").str.upper()
    return (
        frame.dropna(subset=["state_fips", "state_abbr"])
        .drop_duplicates("state_fips")
        .reset_index(drop=True)
    )


def load_county_to_cbsa(paths: RepoPaths) -> pd.DataFrame:
    """The 2023 OMB delineation, read from the same tracked file the model surface reads."""
    from crimerisk.model_surface import load_county_to_cbsa_delineation

    frame = load_county_to_cbsa_delineation(paths)
    if frame.empty:
        raise FileNotFoundError(
            "the 2023 OMB CBSA delineation is absent: expected configs/county_to_cbsa_2023.csv "
            "or data/CBSA-Delineation/list1_2023.xlsx"
        )
    out = frame.copy()
    out["county_fips"] = out["county_fips"].astype("string").str.zfill(5)
    out["cbsa_code"] = out["cbsa_code"].astype("string").str.strip()
    out["cbsa_title"] = out["cbsa_title"].astype("string").str.strip()
    return out.drop_duplicates("county_fips").reset_index(drop=True)


ZCTA_RELATIONSHIP_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
    "tab20_zcta520_tabblock20_natl.txt"
)
ZCTA_RELATIONSHIP_FILENAME = "tab20_zcta520_tabblock20_natl.txt"
ZCTA_CROSSWALK_FILENAME = "zcta520_block_group_2020_crosswalk.parquet"


def zcta_relationship_dir(paths: RepoPaths) -> Path:
    return paths.data_dir / "Census-Relationship-2020"


def zcta_relationship_path(paths: RepoPaths) -> Path:
    return zcta_relationship_dir(paths) / ZCTA_RELATIONSHIP_FILENAME


def zcta_crosswalk_path(paths: RepoPaths) -> Path:
    return zcta_relationship_dir(paths) / ZCTA_CROSSWALK_FILENAME


def ensure_zcta_relationship_file(paths: RepoPaths, *, allow_download: bool = True) -> Path:
    """The Census 2020 ZCTA-to-block relationship file, from `data/` or pulled once and cached."""
    path = zcta_relationship_path(paths)
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"{path} is absent and downloading is disabled; pull it from {ZCTA_RELATIONSHIP_URL}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(ZCTA_RELATIONSHIP_URL, tmp_path)
    tmp_path.replace(path)
    return path


def _read_zcta_block_pairs(path: Path) -> pd.DataFrame:
    """`(zcta5, block_geoid)` from the pipe-delimited relationship file, two columns of many.

    A block carries an empty ZCTA where it is not in any ZCTA -- 2020 ZCTAs do not tile the
    country -- and those rows are dropped here and accounted for as an uncovered residual later.
    """
    frame = pd.read_csv(
        path,
        sep="|",
        usecols=["GEOID_ZCTA5_20", "GEOID_TABBLOCK_20"],
        dtype={"GEOID_ZCTA5_20": "string", "GEOID_TABBLOCK_20": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )
    frame = frame.rename(
        columns={"GEOID_ZCTA5_20": "zcta5", "GEOID_TABBLOCK_20": "block_geoid"}
    )
    frame["zcta5"] = frame["zcta5"].str.strip()
    frame["block_geoid"] = frame["block_geoid"].str.strip().str.zfill(15)
    frame = frame[frame["zcta5"].notna() & frame["zcta5"].ne("")]
    return frame.reset_index(drop=True)


def build_zcta_block_group_crosswalk(
    paths: RepoPaths,
    *,
    block_crosswalk_path: Path | None = None,
    relationship_path: Path | None = None,
    out_path: Path | None = None,
    force_rebuild: bool = False,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Overlap-weighted block group -> ZCTA weights, from 2020 blocks and their 2020 population.

    ZCTAs are built from 2020 tabulation blocks, and blocks nest exactly inside block groups, so
    the block table gives an exact areal decomposition of every block group across the ZCTAs that
    intersect it -- no polygon intersection, no centroid snap.

    Weight for a (block group, ZCTA) pair is the pair's share of the block group's 2020 census
    POPULATION; a block group whose ZCTA-covered blocks hold no population falls back to housing
    units, then to land area, then to an equal split over covered blocks. Weights are renormalized
    to sum to 1 within each block group, so a block group only partly inside any ZCTA still has its
    whole count distributed across the ZCTAs that do cover it. Block groups with no ZCTA-covered
    block at all are absent from the crosswalk and are reported as an uncovered residual.
    """
    cache_path = out_path or zcta_crosswalk_path(paths)
    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    relationship = relationship_path or ensure_zcta_relationship_file(
        paths, allow_download=allow_download
    )
    pairs = _read_zcta_block_pairs(Path(relationship))

    blocks_path = block_crosswalk_path or (
        paths.state_dir / "geometry" / "block_to_jurisdiction_crosswalk.parquet"
    )
    blocks = pd.read_parquet(
        blocks_path, columns=["block_geoid", "block_group_geoid", "pop20", "housing20", "aland20"]
    )
    blocks["block_geoid"] = blocks["block_geoid"].astype("string").str.zfill(15)
    blocks["block_group_geoid"] = blocks["block_group_geoid"].astype("string").str.zfill(12)

    merged = blocks.merge(pairs, on="block_geoid", how="inner")
    for column in ("pop20", "housing20", "aland20"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["block_count"] = 1.0

    grouped = (
        merged.groupby([BLOCK_GROUP_ID_COLUMN, "zcta5"], dropna=False)[
            ["pop20", "housing20", "aland20", "block_count"]
        ]
        .sum()
        .reset_index()
    )
    totals = grouped.groupby(BLOCK_GROUP_ID_COLUMN)[
        ["pop20", "housing20", "aland20", "block_count"]
    ].transform("sum")

    weight = pd.Series(np.nan, index=grouped.index, dtype=float)
    basis = pd.Series("", index=grouped.index, dtype="string")
    for column, label in (
        ("pop20", "block_population_2020"),
        ("housing20", "block_housing_units_2020"),
        ("aland20", "block_land_area"),
        ("block_count", "equal_split_over_blocks"),
    ):
        usable = weight.isna() & totals[column].gt(0.0)
        weight.loc[usable] = grouped.loc[usable, column] / totals.loc[usable, column]
        basis.loc[usable] = label

    out = grouped[[BLOCK_GROUP_ID_COLUMN, "zcta5"]].copy()
    out["weight"] = weight
    out["weight_basis"] = basis
    out = out[out["weight"].notna() & out["weight"].gt(0.0)].copy()
    # Renormalize defensively: the fallback ladder is per block group, so the shares already sum
    # to 1, but a dropped zero-weight part must not leak mass.
    renorm = out.groupby(BLOCK_GROUP_ID_COLUMN)["weight"].transform("sum")
    out["weight"] = out["weight"] / renorm
    out = out.sort_values([BLOCK_GROUP_ID_COLUMN, "zcta5"], kind="mergesort").reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return out


def build_crosswalk(
    geo: RollupGeography,
    *,
    block_group_ids: pd.Series,
    paths: RepoPaths,
    zcta_crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`[block_group_geoid, <id_column>, (<name_column>), weight]` for one rollup geography."""
    ids = pd.Series(block_group_ids, dtype="string").str.zfill(12).rename(BLOCK_GROUP_ID_COLUMN)
    frame = ids.to_frame().drop_duplicates().reset_index(drop=True)

    if geo.key == COUNTY.key:
        frame[geo.id_column] = frame[BLOCK_GROUP_ID_COLUMN].str.slice(0, 5)
        frame["weight"] = 1.0
        names = load_county_names(paths).rename(columns={"county_fips": geo.id_column})
        frame = frame.merge(names, on=geo.id_column, how="left")
        return frame[[BLOCK_GROUP_ID_COLUMN, geo.id_column, "county_name", "weight"]]

    if geo.key == STATE.key:
        frame[geo.id_column] = frame[BLOCK_GROUP_ID_COLUMN].str.slice(0, 2)
        frame["weight"] = 1.0
        names = load_state_abbreviations(paths)
        frame = frame.merge(names, on="state_fips", how="left")
        return frame[[BLOCK_GROUP_ID_COLUMN, geo.id_column, "state_abbr", "weight"]]

    if geo.key == CBSA.key:
        frame["county_fips"] = frame[BLOCK_GROUP_ID_COLUMN].str.slice(0, 5)
        delineation = load_county_to_cbsa(paths)
        frame = frame.merge(delineation, on="county_fips", how="left")
        frame["weight"] = 1.0
        frame = frame[frame[geo.id_column].notna()].copy()
        return frame[[BLOCK_GROUP_ID_COLUMN, geo.id_column, "cbsa_title", "weight"]].reset_index(
            drop=True
        )

    if geo.key == ZCTA.key:
        crosswalk = (
            zcta_crosswalk if zcta_crosswalk is not None else build_zcta_block_group_crosswalk(paths)
        )
        crosswalk = crosswalk.copy()
        crosswalk[BLOCK_GROUP_ID_COLUMN] = (
            crosswalk[BLOCK_GROUP_ID_COLUMN].astype("string").str.zfill(12)
        )
        frame = frame.merge(
            crosswalk[[BLOCK_GROUP_ID_COLUMN, geo.id_column, "weight"]],
            on=BLOCK_GROUP_ID_COLUMN,
            how="inner",
        )
        return frame[[BLOCK_GROUP_ID_COLUMN, geo.id_column, "weight"]].reset_index(drop=True)

    raise KeyError(f"no crosswalk builder for geography {geo.key!r}")


# --- the rollup ------------------------------------------------------------------------------


def _weighted_sums(
    surface: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    geo: RollupGeography,
    columns: list[str],
) -> pd.DataFrame:
    merged = crosswalk[[BLOCK_GROUP_ID_COLUMN, geo.id_column, "weight"]].merge(
        surface[[BLOCK_GROUP_ID_COLUMN, *columns]], on=BLOCK_GROUP_ID_COLUMN, how="inner"
    )
    weight = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    work = pd.DataFrame({geo.id_column: merged[geo.id_column]})
    for column in columns:
        work[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0) * weight
    work["block_group_weight_sum"] = weight
    work["block_group_parts"] = 1.0
    return work.groupby(geo.id_column, dropna=False).sum().reset_index()


def _publication_floors(surface: pd.DataFrame) -> dict[str, float]:
    return {column: _scalar(surface, column, what=column) for column in FLOOR_COLUMNS}


def _denominator_floor(
    *, denominator_type: str, floors: dict[str, float]
) -> float:
    """The published floor for a denominator type, read off the surface's own floor columns."""
    if denominator_type == "vehicles":
        return float(floors["mvt_vehicle_exposure_denominator_floor"])
    if denominator_type == "exposure":
        return float(floors["person_exposure_denominator_floor"])
    # 'premises' (burglary) carries no floor of its own beyond the hard minimum; its residential
    # gate is the household floor, applied to every offense below.
    return float(floors["eb_hard_min_denominator"])


def roll_up(
    surface: pd.DataFrame,
    *,
    geo: RollupGeography,
    crosswalk: pd.DataFrame,
    composites: CompositeRuntime,
    resident_threshold: float = DEFAULT_ZERO_RESIDENT_RESIDENT_THRESHOLD,
) -> pd.DataFrame:
    """One rollup table: counts summed exactly, every rate/index/composite recomputed."""
    floors = _publication_floors(surface)
    population_col = _surface_population_column(surface)
    rates = national_rates(surface)
    denominator_types = {offense: _denominator_type(surface, offense) for offense in OFFENSES_7}

    sum_columns = [
        population_col,
        *SUMMED_BASE_COLUMNS,
        *[expected_count_column(offense) for offense in OFFENSES_7],
        *[primary_denominator_column(offense) for offense in OFFENSES_7],
    ]
    out = _weighted_sums(surface, crosswalk, geo=geo, columns=sum_columns)
    out["block_group_parts"] = out["block_group_parts"].astype(int)
    out.insert(0, "rollup_support", geo.key)

    labels = None
    if geo.name_column is not None and geo.name_column in crosswalk.columns:
        labels = (
            crosswalk[[geo.id_column, geo.name_column]]
            .dropna(subset=[geo.id_column])
            .drop_duplicates(geo.id_column)
        )
    if labels is not None:
        out = out.merge(labels, on=geo.id_column, how="left")
    if geo.key == COUNTY.key:
        out["state_fips"] = out[geo.id_column].astype("string").str.slice(0, 2)
    elif geo.key == ZCTA.key:
        # A ZCTA can straddle a state line; the state carried here is the one holding the largest
        # weighted share, and is a label, not a partition.
        out = out.merge(
            _dominant_state(
                surface, crosswalk, geo=geo, population_col=population_col
            ),
            on=geo.id_column,
            how="left",
        )

    # Aggregate counts, recomputed as sums of the seven published counts at this support.
    for label, offenses in (
        ("personal", PERSONAL_OFFENSES),
        ("property", PROPERTY_OFFENSES),
        ("total", tuple(OFFENSES_7)),
    ):
        out[expected_count_column(label)] = sum(
            out[expected_count_column(offense)] for offense in offenses
        )

    households = pd.to_numeric(out["households_total"], errors="coerce").fillna(0.0)
    population = pd.to_numeric(out[population_col], errors="coerce").fillna(0.0)
    residential = households.ge(float(floors["non_residential_household_floor"]))
    resident_publishable = burden_publishable(out)

    for offense in OFFENSES_7:
        counts = pd.to_numeric(out[expected_count_column(offense)], errors="coerce").fillna(0.0)
        denominator = pd.to_numeric(
            out[primary_denominator_column(offense)], errors="coerce"
        ).fillna(0.0)
        floor = _denominator_floor(denominator_type=denominator_types[offense], floors=floors)
        publishable = (
            residential
            & denominator.gt(0.0)
            & denominator.ge(float(floors["eb_hard_min_denominator"]))
            & denominator.ge(floor)
            # The near-zero-resident opportunity-rate rule, reapplied at this support: a unit with
            # almost no residents publishes an opportunity rate only on a large opportunity
            # denominator.
            & (
                population.ge(float(resident_threshold))
                | denominator.ge(float(floors["zero_resident_opportunity_rate_floor"]))
            )
        )
        out[f"primary_denominator_type_{offense}"] = denominator_types[offense]
        out[primary_national_rate_column(offense)] = rates["primary"][offense]
        out[f"primary_index_publishable_{offense}"] = publishable
        rate = pd.Series(np.nan, index=out.index, dtype=float)
        rate.loc[publishable] = (
            RATE_PER_100K * counts.loc[publishable] / denominator.loc[publishable]
        )
        out[f"rate_{offense}_primary"] = rate
        out[f"index_{offense}_primary"] = 100.0 * rate / float(rates["primary"][offense])

        resident_denominator = pd.to_numeric(
            out[COMMON_DENOMINATOR_COLUMN], errors="coerce"
        ).fillna(0.0)
        out[resident_national_rate_column(offense)] = rates["resident"][offense]
        out[f"index_{offense}_resident_publishable"] = resident_publishable
        resident_rate = pd.Series(np.nan, index=out.index, dtype=float)
        resident_rate.loc[resident_publishable] = (
            RATE_PER_100K
            * counts.loc[resident_publishable]
            / resident_denominator.loc[resident_publishable]
        )
        out[f"rate_{offense}_resident"] = resident_rate
        out[f"index_{offense}_resident"] = (
            100.0 * resident_rate / float(rates["resident"][offense])
        )

    land_area = pd.to_numeric(out["land_area_sq_mi"], errors="coerce")
    land_area = land_area.where(land_area.gt(0.0))
    for label in (*OFFENSES_7, "total"):
        out[f"crime_density_{label}"] = (
            pd.to_numeric(out[expected_count_column(label)], errors="coerce") / land_area
        )

    out = apply_count_first_composites(out, runtime=composites, support=geo.support)
    out = _multi_offense_scores(out)

    if geo.caveat is not None:
        out["geography_caveat"] = geo.caveat

    ordered = _ordered_columns(out, geo=geo, population_col=population_col)
    return out[ordered].sort_values([geo.id_column], kind="mergesort").reset_index(drop=True)


def _dominant_state(
    surface: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    geo: RollupGeography,
    population_col: str,
) -> pd.DataFrame:
    merged = crosswalk.merge(
        surface[[BLOCK_GROUP_ID_COLUMN, "state_fips", population_col]],
        on=BLOCK_GROUP_ID_COLUMN,
        how="inner",
    )
    merged["mass"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0) * (
        pd.to_numeric(merged[population_col], errors="coerce").fillna(0.0) + 1.0
    )
    grouped = (
        merged.groupby([geo.id_column, "state_fips"], dropna=False)["mass"].sum().reset_index()
    )
    grouped = grouped.sort_values([geo.id_column, "mass", "state_fips"], ascending=[True, False, True])
    return grouped.drop_duplicates(geo.id_column)[[geo.id_column, "state_fips"]].reset_index(drop=True)


def _multi_offense_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """The mixed-denominator relative scores, recomputed at this support."""
    out = frame
    indexes = {
        offense: pd.to_numeric(out[f"index_{offense}_primary"], errors="coerce")
        for offense in OFFENSES_7
    }
    counts = {
        offense: float(
            pd.to_numeric(out[expected_count_column(offense)], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .sum()
        )
        for offense in OFFENSES_7
    }
    def incident_weights(offenses: tuple[str, ...]) -> dict[str, float]:
        total = float(sum(counts[offense] for offense in offenses))
        if total <= 0.0:
            return {offense: float("nan") for offense in offenses}
        return {offense: counts[offense] / total for offense in offenses}

    specs = (
        (
            "multi_offense_relative_score_event_weighted",
            tuple(OFFENSES_7),
            incident_weights(tuple(OFFENSES_7)),
        ),
        (
            "multi_offense_relative_score_equal_offense",
            tuple(OFFENSES_7),
            {offense: 1.0 for offense in OFFENSES_7},
        ),
        (
            PERSONAL_RELATIVE_SCORE_COLUMN,
            PERSONAL_OFFENSES,
            incident_weights(PERSONAL_OFFENSES),
        ),
        (
            PROPERTY_RELATIVE_SCORE_COLUMN,
            PROPERTY_OFFENSES,
            incident_weights(PROPERTY_OFFENSES),
        ),
    )
    for column, offenses, offense_weights in specs:
        publishable = pd.Series(True, index=out.index)
        for offense in offenses:
            publishable &= indexes[offense].notna()
        weight_sum = float(
            sum(value for value in offense_weights.values() if pd.notna(value) and value > 0.0)
        )
        score = pd.Series(np.nan, index=out.index, dtype=float)
        if weight_sum > 0.0:
            weighted = sum(
                indexes[offense] * float(offense_weights[offense]) for offense in offenses
            )
            score.loc[publishable] = weighted.loc[publishable] / weight_sum
        out[column] = score.replace([np.inf, -np.inf], np.nan)
    return out


def _ordered_columns(
    frame: pd.DataFrame, *, geo: RollupGeography, population_col: str
) -> list[str]:
    ordered = ["rollup_support", geo.id_column]
    if geo.name_column is not None:
        ordered.append(geo.name_column)
    if "state_fips" in frame.columns and geo.id_column != "state_fips":
        ordered.append("state_fips")
    ordered += [
        "block_group_parts",
        "block_group_weight_sum",
        population_col,
        *SUMMED_BASE_COLUMNS,
    ]
    for offense in OFFENSES_7:
        ordered += [
            expected_count_column(offense),
            f"primary_denominator_type_{offense}",
            primary_denominator_column(offense),
            primary_national_rate_column(offense),
            f"primary_index_publishable_{offense}",
            f"rate_{offense}_primary",
            f"index_{offense}_primary",
            resident_national_rate_column(offense),
            f"index_{offense}_resident_publishable",
            f"rate_{offense}_resident",
            f"index_{offense}_resident",
        ]
    ordered += [
        expected_count_column("personal"),
        expected_count_column("property"),
        expected_count_column("total"),
        *[f"crime_density_{offense}" for offense in OFFENSES_7],
        "crime_density_total",
        HARM_WEIGHTED_COUNT_COLUMN,
        *COUNT_FIRST_AGGREGATE_INDEX_FIELDS,
    ]
    if "geography_caveat" in frame.columns:
        ordered.append("geography_caveat")
    return [column for column in ordered if column in frame.columns]


# --- conservation ------------------------------------------------------------------------------

# Rollup sums and the national reference are the same float64 additions in a different order, so
# the identity is exact up to summation epsilon. Held to 1e-6 absolute on totals of order 1e7 --
# a relative 1e-13, tighter than any real defect could hide under.
CONSERVATION_ABS_TOLERANCE = 1e-6
CONSERVATION_REL_TOLERANCE = 1e-12


def _national_totals(surface: pd.DataFrame) -> dict[str, float]:
    return {
        offense: float(
            pd.to_numeric(surface[expected_count_column(offense)], errors="coerce").fillna(0.0).sum()
        )
        for offense in OFFENSES_7
    }


def _outside_universe_totals(
    surface: pd.DataFrame, crosswalk: pd.DataFrame
) -> tuple[dict[str, float], int]:
    """Counts on block groups the crosswalk does not place, and how many block groups those are.

    Nonzero only where the target geography does not cover the country: CBSAs (counties outside
    every metro/micro area) and ZCTAs (block groups with no ZCTA-covered block).
    """
    covered = set(crosswalk[BLOCK_GROUP_ID_COLUMN].astype("string").str.zfill(12).unique())
    outside = surface[~surface[BLOCK_GROUP_ID_COLUMN].isin(covered)]
    return _national_totals(outside), int(len(outside))


def conservation_report(
    surface: pd.DataFrame,
    *,
    geo: RollupGeography,
    crosswalk: pd.DataFrame,
    rollup: pd.DataFrame,
) -> dict[str, object]:
    """`rolled + outside_universe == national`, offense by offense, as measured numbers."""
    national = _national_totals(surface)
    population_col = _surface_population_column(surface)
    outside, outside_block_groups = _outside_universe_totals(surface, crosswalk)
    offenses: dict[str, object] = {}
    conserved = True
    for offense in OFFENSES_7:
        rolled = float(
            pd.to_numeric(rollup[expected_count_column(offense)], errors="coerce").fillna(0.0).sum()
        )
        reference = float(national[offense])
        difference = rolled + float(outside[offense]) - reference
        tolerance = max(CONSERVATION_ABS_TOLERANCE, CONSERVATION_REL_TOLERANCE * abs(reference))
        ok = bool(abs(difference) <= tolerance)
        conserved = conserved and ok
        offenses[offense] = {
            "block_group_total": reference,
            "rollup_total": rolled,
            "outside_universe_total": float(outside[offense]),
            "difference": difference,
            "tolerance": tolerance,
            "conserved": ok,
        }
    population_rolled = float(
        pd.to_numeric(rollup[population_col], errors="coerce").fillna(0.0).sum()
    )
    population_national = float(
        pd.to_numeric(surface[population_col], errors="coerce").fillna(0.0).sum()
    )
    covered = set(crosswalk[BLOCK_GROUP_ID_COLUMN].astype("string").str.zfill(12).unique())
    population_outside = float(
        pd.to_numeric(
            surface.loc[~surface[BLOCK_GROUP_ID_COLUMN].isin(covered), population_col],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )
    population_ok = bool(
        abs(population_rolled + population_outside - population_national)
        <= max(CONSERVATION_ABS_TOLERANCE, CONSERVATION_REL_TOLERANCE * abs(population_national))
    )
    return {
        "support": geo.key,
        "version": ROLLUP_VERSION,
        "covers_universe": bool(geo.covers_universe),
        "units": int(len(rollup)),
        "block_groups_in_surface": int(len(surface)),
        "block_groups_placed": int(len(covered)),
        "block_groups_outside_universe": outside_block_groups,
        "population": {
            "block_group_total": population_national,
            "rollup_total": population_rolled,
            "outside_universe_total": population_outside,
            "conserved": population_ok,
        },
        "offenses": offenses,
        "conserved": bool(conserved and population_ok),
    }


def assert_conservation(report: dict[str, object]) -> None:
    if not report.get("conserved"):
        failures = [
            offense
            for offense, values in (report.get("offenses") or {}).items()
            if not values.get("conserved")
        ]
        raise AssertionError(
            f"rollup at {report.get('support')!r} does not conserve counts: {failures or ['population']}"
        )


# --- the build entry point ---------------------------------------------------------------------


def rollup_filename(geo: RollupGeography, *, year: int) -> str:
    return f"crimerisk_{geo.key}_{int(year)}_rollup.parquet"


def build_rollups(
    *,
    paths: RepoPaths,
    surface_path: Path,
    out_dir: Path,
    composites: CompositeRuntime,
    year: int = 2024,
    geographies: tuple[RollupGeography, ...] = ROLLUP_GEOGRAPHIES,
    allow_download: bool = True,
    manifest: dict | None = None,
) -> dict[str, object]:
    """Write one parquet per rollup geography and return the conservation + provenance summary."""
    surface = read_block_group_surface(Path(surface_path), year=year)
    resident_threshold = zero_resident_resident_threshold(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    zcta_crosswalk = None
    if any(geo.key == ZCTA.key for geo in geographies):
        zcta_crosswalk = build_zcta_block_group_crosswalk(paths, allow_download=allow_download)

    summary: dict[str, object] = {
        "version": ROLLUP_VERSION,
        "year": int(year),
        "surface_path": str(Path(surface_path)),
        "coverage_universe": coverage_universe(surface),
        "rule": (
            "counts summed exactly; rates recomputed as 100000 * Sum(count) / Sum(denominator); "
            "indexes recomputed as 100 * rate / stored national rate; composites recomputed at "
            "the rollup's own support"
        ),
        "zero_resident_resident_threshold": float(resident_threshold),
        "geographies": {},
        "conserved": True,
    }
    for geo in geographies:
        crosswalk = build_crosswalk(
            geo, block_group_ids=surface[BLOCK_GROUP_ID_COLUMN], paths=paths, zcta_crosswalk=zcta_crosswalk
        )
        rollup = roll_up(
            surface,
            geo=geo,
            crosswalk=crosswalk,
            composites=composites,
            resident_threshold=resident_threshold,
        )
        report = conservation_report(surface, geo=geo, crosswalk=crosswalk, rollup=rollup)
        assert_conservation(report)
        path = out_dir / rollup_filename(geo, year=year)
        rollup.to_parquet(path, index=False)
        entry: dict[str, object] = {
            "path": str(path),
            "rows": int(len(rollup)),
            "columns": int(rollup.shape[1]),
            "source": geo.source,
            "overlap_weighted": not geo.nests_in_block_group,
            "conservation": report,
            "composite_normalizers": composite_normalizers(
                rollup, runtime=composites, support=geo.support
            ),
        }
        if geo.caveat is not None:
            entry["caveat"] = geo.caveat
        summary["geographies"][geo.key] = entry
        summary["conserved"] = bool(summary["conserved"] and report["conserved"])
    return summary
