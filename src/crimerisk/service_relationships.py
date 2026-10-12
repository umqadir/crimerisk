"""Who polices which territory: effective-dated service relationships, separate from reporting.

Whether an agency reports to the FBI says nothing about who polices a municipality. The
geometry stage used to infer one from the other: a municipality whose every linked ORI had
never filed a positive count was dropped from municipal geometry into the state remainder,
which is right for a town a sheriff polices and wrong for a city with its own department that
simply does not report (Greenville MS) and for a town in a multi-town department (Larkspur and
Corte Madera in Central Marin).

The relationship table is the only thing that moves a municipality's territory:

* `contract` / `consolidated`: the territory is policed by `provider_ori9`. When the provider's
  own territory is a municipal jurisdiction (a contracting city, a multi-town department), the
  territory joins that jurisdiction, so the provider's total is spread over everything it
  polices. Otherwise (a sheriff, a county or state police unit) the territory joins the state
  remainder, whose county pools carry that provider's total.
* `county_unincorporated`: the provider also polices unincorporated county land; no geometry
  change, but the county remainder counts toward the provider's checked footprint.
* `exclusive` and `supplemental`: no territory change.

A municipality with no row keeps its own geometry. If its department does not report, its
control comes from the silent-unit imputation, never from a neighbour's numerator.

Sources, first match per territory and year: the governed `configs/service_relationships.csv`
(researched rows), then the consolidated-footprint registry, then the FBI NIBRS batch header's
`covered_by_ori` (the latest year at or before the build year within the look-back).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.contract_coverage import load_latest_covered_by_relationships
from crimerisk.ledger import require_approved

RELATIONS = ("exclusive", "contract", "consolidated", "county_unincorporated", "supplemental")
TERRITORY_MOVING_RELATIONS = frozenset({"contract", "consolidated"})
CONFIG_RELPATH = "configs/service_relationships.csv"
CONFIG_COLUMNS = [
    "relationship_id",
    "territory_jurisdiction_id",
    "provider_ori9",
    "relation",
    "effective_from_year",
    "effective_to_year",
    "evidence_source",
    "note",
]
EXCEPTIONS_RELPATH = "configs/service_territory_exceptions.csv"
EXCEPTION_COLUMNS = ["ori9", "reason_code", "evidence_source", "note"]

TABLE_COLUMNS = [
    "relationship_id",
    "territory_jurisdiction_id",
    "provider_ori9",
    "relation",
    "effective_from_year",
    "effective_to_year",
    "source",
    "evidence_source",
]
SOURCE_CONFIG = "service_relationships_config"
SOURCE_CONSOLIDATED = "consolidated_agency_footprints"
SOURCE_FBI = "fbi_batch_header_covered_by"

DISPOSITION_MERGED = "merged_into_provider_jurisdiction"
DISPOSITION_REMAINDER = "state_remainder"

# Build-time territory check (served population against mapped territory, aligned vintage).
TERRITORY_CHECK_MIN_SERVED = 10_000.0
TERRITORY_CHECK_MAX_RATIO = 1.3
# The FBI's served populations before 2022 rest on pre-2020-census estimates (Bexar, Williamson
# and Collin TX, Farmers Branch, Hanahan ran far above the census); the check compares the FBI's
# 2022 figure with the 2020-census territory grown by its county's 2020-2022 estimate.
TERRITORY_CHECK_POPULATION_YEAR = 2022
TERRITORY_CHECK_POPULATION_YEARS = (2022, 2023, 2021)
# The controls stage stops on an unresolved territory flag.
TERRITORY_CHECK_BLOCKING = True
RESOLUTION_WITHIN_BAND = "within_band"
RESOLUTION_EXCEPTION = "recorded_exception"
RESOLUTION_UNRESOLVED = "unresolved"
COUNTY_REMAINDER_KIND = ":state_nonmunicipal_remainder:county:"


class ServiceRelationshipError(ValueError):
    pass


def _config_path(paths, relpath: str) -> Path:
    return Path(paths.repo_root) / relpath


def load_relationship_config(paths) -> pd.DataFrame:
    path = _config_path(paths, CONFIG_RELPATH)
    if not path.exists():
        return pd.DataFrame(columns=CONFIG_COLUMNS)
    require_approved(path)
    frame = pd.read_csv(path, dtype="string", comment=None).reindex(columns=CONFIG_COLUMNS)
    if frame.empty:
        return frame
    # A blank effective_from_year means in force since before the panel starts.
    required = [c for c in CONFIG_COLUMNS if c not in {"effective_from_year", "effective_to_year", "note"}]
    blank = frame[required].isna() | frame[required].apply(lambda s: s.str.strip().eq(""))
    if blank.any().any():
        raise ServiceRelationshipError(
            f"{path} has blank required values in {sorted(blank.columns[blank.any()].tolist())}"
        )
    if frame["relationship_id"].duplicated().any():
        raise ServiceRelationshipError(f"{path} requires unique relationship_id values")
    bad = sorted(set(frame["relation"]) - set(RELATIONS))
    if bad:
        raise ServiceRelationshipError(f"{path} has unknown relations {bad}")
    frame["provider_ori9"] = frame["provider_ori9"].str.strip().str.upper()
    frame["territory_jurisdiction_id"] = frame["territory_jurisdiction_id"].str.strip()
    return frame


def _offending(frame: pd.DataFrame, mask: pd.Series) -> str:
    cols = ["relationship_id", "territory_jurisdiction_id", "provider_ori9", "effective_from_year", "effective_to_year"]
    return str(frame.loc[mask, cols].astype(str).to_dict(orient="records"))


def validate_relationship_config(frame: pd.DataFrame, *, known_oris: set[str]) -> None:
    """Every researched row names a known provider, strict effective years, and no territory
    given to two providers in an overlapping year (a `supplemental` row gives nothing)."""
    if frame.empty:
        return
    unknown = ~frame["provider_ori9"].astype(str).isin(known_oris)
    if unknown.any():
        raise ServiceRelationshipError(
            f"{CONFIG_RELPATH} names providers absent from the agency master and crosswalk: "
            + _offending(frame, unknown)
        )
    years = {}
    for column in ("effective_from_year", "effective_to_year"):
        raw = frame[column].astype("string").str.strip()
        given = raw.notna() & raw.ne("")
        parsed = pd.to_numeric(raw.where(given), errors="coerce").astype(float)
        bad = given & (parsed.isna() | parsed.mod(1).ne(0))
        if bad.any():
            raise ServiceRelationshipError(f"{CONFIG_RELPATH} has a non-integer {column}: " + _offending(frame, bad))
        years[column] = parsed
    start, end = years["effective_from_year"], years["effective_to_year"]
    reversed_ = start.notna() & end.notna() & start.gt(end)
    if reversed_.any():
        raise ServiceRelationshipError(
            f"{CONFIG_RELPATH} has a relationship that ends before it starts: " + _offending(frame, reversed_)
        )
    giving = frame.assign(_start=start.fillna(-np.inf), _end=end.fillna(np.inf))
    giving = giving[giving["relation"].ne("supplemental")].reset_index()
    pairs = giving.merge(giving, on="territory_jurisdiction_id", suffixes=("", "_other"))
    clash = pairs[
        pairs["index"].lt(pairs["index_other"])
        & pairs["provider_ori9"].ne(pairs["provider_ori9_other"])
        & pairs["_start"].le(pairs["_end_other"])
        & pairs["_start_other"].le(pairs["_end"])
    ]
    if not clash.empty:
        raise ServiceRelationshipError(
            f"{CONFIG_RELPATH} gives one territory to different providers in overlapping years: "
            + str(clash[["territory_jurisdiction_id", "relationship_id", "relationship_id_other"]].to_dict(orient="records"))
        )


def validate_relationship_territories(frame: pd.DataFrame, *, territory_population: pd.Series) -> None:
    """Every researched row's territory resolves to a positive 2020 population."""
    if frame.empty:
        return
    population = frame["territory_jurisdiction_id"].astype(str).map(territory_population)
    empty = pd.to_numeric(population, errors="coerce").fillna(0.0).le(0.0)
    if empty.any():
        raise ServiceRelationshipError(
            f"{CONFIG_RELPATH} has territories that resolve to no population: " + _offending(frame, empty)
        )


def load_territory_exceptions(paths) -> pd.DataFrame:
    path = _config_path(paths, EXCEPTIONS_RELPATH)
    if not path.exists():
        return pd.DataFrame(columns=EXCEPTION_COLUMNS)
    require_approved(path)
    frame = pd.read_csv(path, dtype="string").reindex(columns=EXCEPTION_COLUMNS)
    frame["ori9"] = frame["ori9"].str.strip().str.upper()
    blank = frame[["ori9", "reason_code", "evidence_source"]].isna().any(axis=1) | frame["reason_code"].fillna("").str.strip().eq("")
    if blank.any():
        raise ServiceRelationshipError(f"{path} rows need an ori9, reason_code and evidence_source")
    if frame["ori9"].duplicated().any():
        raise ServiceRelationshipError(f"{path} requires unique ori9 values")
    return frame


def _in_force(frame: pd.DataFrame, *, year: int) -> pd.Series:
    start = pd.to_numeric(frame["effective_from_year"], errors="coerce")
    end = pd.to_numeric(frame["effective_to_year"], errors="coerce")
    return (start.isna() | start.le(int(year))) & (end.isna() | end.ge(int(year)))


def _consolidated_relationships(paths) -> pd.DataFrame:
    path = _config_path(paths, "configs/consolidated_agency_footprints.csv")
    if not path.exists():
        return pd.DataFrame(columns=TABLE_COLUMNS)
    footprints = pd.read_csv(path, dtype="string")
    rows = []
    for row in footprints.itertuples(index=False):
        state = str(row.state_fips).zfill(2)
        if str(row.geometry_source_type).endswith("plus_county_remainder"):
            county = str(row.county_fips).zfill(3)
            rows.append(
                {
                    "relationship_id": f"consolidated:{row.ori}:county:{state}{county}",
                    "territory_jurisdiction_id": f"{state}:state_nonmunicipal_remainder:county:{state}{county}",
                    "provider_ori9": str(row.ori).upper(),
                    "relation": "county_unincorporated",
                    "effective_from_year": pd.NA,
                    "effective_to_year": pd.NA,
                    "source": SOURCE_CONSOLIDATED,
                    "evidence_source": str(row.reviewer_note),
                }
            )
        included = "" if pd.isna(row.included_place_geoids) else str(row.included_place_geoids)
        for geoid in included.split("|"):
            geoid = geoid.strip()
            if not geoid or geoid == "<NA>":
                continue
            rows.append(
                {
                    "relationship_id": f"consolidated:{row.ori}:{geoid}",
                    "territory_jurisdiction_id": f"{state}:municipal:place:{geoid}",
                    "provider_ori9": str(row.ori).upper(),
                    "relation": "consolidated",
                    "effective_from_year": pd.NA,
                    "effective_to_year": pd.NA,
                    "source": SOURCE_CONSOLIDATED,
                    "evidence_source": str(row.reviewer_note),
                }
            )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _fbi_covered_relationships(paths, *, year: int, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """A municipality every positive-weight ORI of which the FBI lists as covered by another."""
    covered = load_latest_covered_by_relationships(paths, year=int(year))
    if covered.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    relationship_year = covered.attrs.get("relationship_year")
    covering = dict(zip(covered["ori9"].astype(str), covered["covering_ori"].astype(str), strict=False))
    links = crosswalk[
        crosswalk["jurisdiction_id"].astype("string").str.contains(":municipal:", na=False)
    ][["ori9", "jurisdiction_id"]].drop_duplicates()
    links["covering"] = links["ori9"].astype(str).map(covering)
    grouped = links.groupby("jurisdiction_id").agg(
        all_covered=("covering", lambda s: bool(s.notna().all())),
        providers=("covering", lambda s: sorted(set(s.dropna()))),
    )
    grouped = grouped[grouped["all_covered"] & grouped["providers"].map(len).eq(1)]
    return pd.DataFrame(
        {
            "relationship_id": [f"fbi_covered_by:{jid}" for jid in grouped.index],
            "territory_jurisdiction_id": grouped.index.astype(str),
            "provider_ori9": [p[0] for p in grouped["providers"]],
            "relation": "contract",
            "effective_from_year": relationship_year,
            "effective_to_year": pd.NA,
            "source": SOURCE_FBI,
            "evidence_source": f"FBI NIBRS batch header covered_by_ori, {relationship_year}",
        },
        columns=TABLE_COLUMNS,
    )


def _known_oris(paths, *, crosswalk: pd.DataFrame) -> set[str]:
    known = set(crosswalk["ori9"].astype(str).str.upper())
    master = Path(paths.state_dir) / "reference" / "agency_master.parquet"
    if master.exists():
        known |= set(pd.read_parquet(master, columns=["ori9"])["ori9"].astype(str).str.upper())
    return known


def build_service_relationships(paths, *, year: int, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """The relationships in force in `year`, one source per territory (config first)."""
    config = load_relationship_config(paths)
    validate_relationship_config(config, known_oris=_known_oris(paths, crosswalk=crosswalk))
    config = config.assign(source=SOURCE_CONFIG).reindex(columns=TABLE_COLUMNS)
    parts = [
        config[_in_force(config, year=int(year))],
        _consolidated_relationships(paths),
        _fbi_covered_relationships(paths, year=int(year), crosswalk=crosswalk),
    ]
    table = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(
        not p.empty for p in parts
    ) else pd.DataFrame(columns=TABLE_COLUMNS)
    if table.empty:
        return table
    # Config years arrive as strings, FBI years as integers; one dtype for the table.
    for column in ("effective_from_year", "effective_to_year"):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    # A researched row owns its territory outright; a source further down the list is not
    # mixed into it.
    order = {SOURCE_CONFIG: 0, SOURCE_CONSOLIDATED: 1, SOURCE_FBI: 2}
    table["_rank"] = table["source"].map(order)
    best = table.groupby("territory_jurisdiction_id")["_rank"].transform("min")
    table = table[table["_rank"].eq(best)].drop(columns="_rank")
    return table.sort_values(["territory_jurisdiction_id", "relationship_id"]).reset_index(drop=True)


def provider_principal_jurisdictions(crosswalk: pd.DataFrame) -> pd.Series:
    """Each ORI's municipal jurisdiction of largest weight, if it has one."""
    muni = crosswalk[
        crosswalk["jurisdiction_id"].astype("string").str.contains(":municipal:", na=False)
        & pd.to_numeric(crosswalk["weight"], errors="coerce").gt(0.0)
    ][["ori9", "jurisdiction_id", "weight"]].copy()
    muni["ori9"] = muni["ori9"].astype(str).str.upper()
    muni = muni.sort_values(["ori9", "weight", "jurisdiction_id"], ascending=[True, False, True])
    return muni.drop_duplicates("ori9").set_index("ori9")["jurisdiction_id"].astype(str)


def territory_dispositions(relationships: pd.DataFrame, *, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Per municipality a relationship moves: where its territory goes, and why."""
    columns = [
        "territory_jurisdiction_id",
        "disposition",
        "principal_jurisdiction_id",
        "provider_ori9",
        "relationship_id",
        "source",
    ]
    if relationships.empty:
        return pd.DataFrame(columns=columns)
    moving = relationships[
        relationships["relation"].isin(TERRITORY_MOVING_RELATIONS)
        & relationships["territory_jurisdiction_id"].astype(str).str.contains(":municipal:", na=False)
    ].copy()
    if moving.empty:
        return pd.DataFrame(columns=columns)
    principal = provider_principal_jurisdictions(crosswalk)
    moving["principal_jurisdiction_id"] = moving["provider_ori9"].astype(str).str.upper().map(principal)
    own = moving["principal_jurisdiction_id"].eq(moving["territory_jurisdiction_id"])
    # A provider whose largest municipal link is the territory itself polices it as its own
    # department: nothing moves (Central Marin's San Anselmo row).
    moving = moving[~own.fillna(False)]
    moving["disposition"] = np.where(
        moving["principal_jurisdiction_id"].notna(), DISPOSITION_MERGED, DISPOSITION_REMAINDER
    )
    duplicated = moving["territory_jurisdiction_id"].duplicated(keep=False)
    if duplicated.any():
        raise ServiceRelationshipError(
            "a municipality has more than one territory-moving relationship in force: "
            + str(sorted(moving.loc[duplicated, "territory_jurisdiction_id"].unique())[:10])
        )
    # A principal that is itself moved would chain territories; refuse rather than guess.
    chained = moving["principal_jurisdiction_id"].isin(set(moving["territory_jurisdiction_id"]))
    if chained.any():
        raise ServiceRelationshipError(
            "a provider's own jurisdiction is itself moved by a relationship: "
            + str(sorted(moving.loc[chained, "principal_jurisdiction_id"].unique())[:10])
        )
    return moving[columns].sort_values("territory_jurisdiction_id").reset_index(drop=True)


def service_relationships_path(paths, *, year: int) -> Path:
    return Path(paths.state_dir) / "reference" / f"service_relationships_{int(year)}.parquet"


def load_crosswalk_for_relationships(paths) -> pd.DataFrame:
    cw = pd.read_parquet(
        Path(paths.state_dir) / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        columns=["ori", "jurisdiction_id", "weight", "relationship_type"],
    ).rename(columns={"ori": "ori9"})
    cw["ori9"] = cw["ori9"].astype("string").str.upper()
    cw["jurisdiction_id"] = cw["jurisdiction_id"].astype("string")
    return cw


def build_territory_dispositions(paths, *, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(relationships in force, territory dispositions) from the repo's inputs."""
    crosswalk = load_crosswalk_for_relationships(paths)
    relationships = build_service_relationships(paths, year=int(year), crosswalk=crosswalk)
    return relationships, territory_dispositions(relationships, crosswalk=crosswalk)


# --- the build-time territory check ------------------------------------------------------


def served_population(agency_obs: pd.DataFrame) -> pd.Series:
    """Each ORI's FBI-reported served population in TERRITORY_CHECK_POPULATION_YEAR (nearest year)."""
    obs = agency_obs[["ori9", "year", "population"]].copy()
    obs["ori9"] = obs["ori9"].astype("string").str.upper()
    by_year = obs.groupby(["ori9", "year"])["population"].max().unstack()
    out = pd.Series(np.nan, index=by_year.index, dtype=float)
    for year in TERRITORY_CHECK_POPULATION_YEARS:
        if year in by_year.columns:
            out = out.fillna(pd.to_numeric(by_year[year], errors="coerce"))
    return out


def territory_check(
    *,
    served: pd.Series,
    crosswalk: pd.DataFrame,
    jurisdiction_population: pd.Series,
    dispositions: pd.DataFrame,
    relationships: pd.DataFrame,
    exceptions: pd.DataFrame,
    agency_types: pd.Series,
    fbi_covered_oris: frozenset[str] = frozenset(),
    custom_population: pd.Series | None = None,
) -> pd.DataFrame:
    """Every agency serving >= TERRITORY_CHECK_MIN_SERVED residents with a municipal territory or
    a county_unincorporated relationship: FBI 2022 served population against its resolved
    footprint (2020 census grown to 2022), and whether the difference is within
    TERRITORY_CHECK_MAX_RATIO, waived by a recorded exception, or unresolved.

    Resolved footprint, summed:
    * municipal: the agency's municipal crosswalk weights times the population each jurisdiction
      owns in the block-group geometry after the relationship moves (`jurisdiction_population`),
      so a town merged into its provider counts for the provider and a town moved away counts
      for no one here;
    * county remainder: every county_unincorporated territory the agency provides (researched
      rows and the consolidated-footprint registry; `jurisdiction_population` carries the
      `SS:state_nonmunicipal_remainder:county:SSCCC` ids);
    * custom: the agency's custom footprint population (`custom_population`), where it has one.

    A relationship row changes only the footprint; a flag is waived only by a population-
    comparability exception (EXCEPTIONS_RELPATH). Not checked: an ORI every municipal territory
    of which a relationship moves to another provider (it serves none of its own; a partly
    moved ORI is checked on what it keeps), and an ORI the FBI lists as covered by another
    (`fbi_covered_oris`, the latest batch header: Everest Metro WI0370900, succeeded by Mountain
    Bay Metro in 2024).
    """
    cw = crosswalk[
        crosswalk["jurisdiction_id"].astype("string").str.contains(":municipal:", na=False)
        & pd.to_numeric(crosswalk["weight"], errors="coerce").gt(0.0)
    ][["ori9", "jurisdiction_id", "weight"]].copy()
    cw["ori9"] = cw["ori9"].astype(str).str.upper()
    cw["jurisdiction_id"] = cw["jurisdiction_id"].astype(str)
    cw["population"] = cw["jurisdiction_id"].map(jurisdiction_population).fillna(0.0)
    cw["mapped"] = pd.to_numeric(cw["weight"], errors="coerce") * cw["population"]
    municipal = cw.groupby("ori9")["mapped"].sum()

    county = pd.Series(dtype=float)
    if not relationships.empty:
        unincorporated = relationships[relationships["relation"].eq("county_unincorporated")]
        county = (
            unincorporated.assign(
                ori9=unincorporated["provider_ori9"].astype(str).str.upper(),
                population=unincorporated["territory_jurisdiction_id"].astype(str).map(jurisdiction_population),
            )
            .groupby("ori9")["population"]
            .sum(min_count=1)
            .fillna(0.0)
        )
    custom = pd.Series(dtype=float) if custom_population is None else custom_population.astype(float)

    # The level-lane rule: an ORI is covered by another only when every one of its municipal
    # territories is moved to another provider.
    transferred: set[str] = set()
    if not dispositions.empty:
        moved_to = dispositions.set_index(dispositions["territory_jurisdiction_id"].astype(str))["provider_ori9"]
        provider = cw["jurisdiction_id"].map(moved_to.astype(str).str.upper())
        moved_away = provider.notna() & provider.ne(cw["ori9"])
        every = moved_away.groupby(cw["ori9"]).all()
        transferred = set(every.index[every]) - set(county.index)

    index = municipal.index.union(county.index)
    frame = pd.DataFrame(index=index)
    frame["served_population"] = served.reindex(index).astype(float)
    frame["municipal_population"] = municipal.reindex(index).fillna(0.0)
    frame["county_remainder_population"] = county.reindex(index).fillna(0.0)
    frame["custom_population"] = custom.reindex(index).fillna(0.0)
    frame["footprint_population"] = frame[
        ["municipal_population", "county_remainder_population", "custom_population"]
    ].sum(axis=1)
    frame["agency_type"] = agency_types.reindex(index)
    frame = frame[frame["served_population"].ge(TERRITORY_CHECK_MIN_SERVED)]
    frame = frame[~frame.index.isin(transferred) & ~frame.index.isin(set(fbi_covered_oris))]
    ratio = frame["served_population"] / frame["footprint_population"].where(frame["footprint_population"].gt(0.0))
    frame["ratio"] = ratio
    flagged = ratio.isna() | ratio.gt(TERRITORY_CHECK_MAX_RATIO) | ratio.lt(1.0 / TERRITORY_CHECK_MAX_RATIO)
    reasons = (
        exceptions.set_index(exceptions["ori9"].astype(str))["reason_code"]
        if not exceptions.empty and "reason_code" in exceptions.columns
        else pd.Series(dtype="string")
    )
    frame["exception_reason_code"] = frame.index.map(reasons)
    excepted = frame.index.isin(set(reasons.index))
    frame["resolution"] = np.where(
        ~flagged,
        RESOLUTION_WITHIN_BAND,
        np.where(excepted, RESOLUTION_EXCEPTION, RESOLUTION_UNRESOLVED),
    )
    frame.index.name = "ori9"
    return frame.reset_index().sort_values("served_population", ascending=False).reset_index(drop=True)


def assert_territories_resolved(flags: pd.DataFrame) -> None:
    unresolved = flags[flags["resolution"].eq(RESOLUTION_UNRESOLVED)]
    if unresolved.empty:
        return
    raise ServiceRelationshipError(
        f"{len(unresolved)} agencies serving >= {int(TERRITORY_CHECK_MIN_SERVED):,} residents differ from "
        f"their resolved footprint by more than {TERRITORY_CHECK_MAX_RATIO - 1:.0%} with no recorded "
        f"population-comparability exception ({EXCEPTIONS_RELPATH}): "
        + str(unresolved[["ori9", "served_population", "footprint_population"]].head(20).to_dict(orient="records"))
    )


def territory_populations(bg: pd.DataFrame, *, column: str) -> pd.Series:
    """Population per municipal jurisdiction in the block-group geometry, and per county
    remainder under its `SS:state_nonmunicipal_remainder:county:SSCCC` territory id."""
    jid = bg["jurisdiction_id"].astype(str)
    value = pd.to_numeric(bg[column], errors="coerce").fillna(0.0)
    own = value.groupby(jid).sum()
    remainder = jid.str.endswith(":state_nonmunicipal_remainder")
    county = bg.loc[remainder, "block_group_geoid"].astype(str).str[:5]
    county_ids = county.str[:2] + COUNTY_REMAINDER_KIND + county
    return pd.concat([own, value[remainder].groupby(county_ids).sum()])


def _tiger_territory_population(paths, territory_ids: list[str]) -> pd.Series:
    """2020 population of municipal territories the block-group geometry no longer holds (a
    moved or unlisted town), read from TIGER by the place or county-subdivision GEOID in the id,
    blocks joined to it by internal point as the geometry does."""
    import geopandas as gpd

    from crimerisk.geometry import _read_state_blocks, _read_state_geo

    out: dict[str, float] = {}
    parsed = [tid.split(":") for tid in territory_ids]
    wanted = [
        (tid, parts[0], parts[2], parts[3])
        for tid, parts in zip(territory_ids, parsed, strict=True)
        if len(parts) == 4 and parts[1] == "municipal" and parts[2] in {"place", "cousub"}
    ]
    for state in sorted({w[1] for w in wanted}):
        block_path = Path(paths.data_dir) / "tiger_tabblock20" / f"tl_2020_{state}_tabblock20.zip"
        if not block_path.exists():
            continue
        blocks = _read_state_blocks(block_path)
        for geo_type, folder in (("place", "tiger_places"), ("cousub", "tiger_cousub")):
            ids = {w[3]: w[0] for w in wanted if w[1] == state and w[2] == geo_type}
            poly_path = Path(paths.data_dir) / folder / f"tl_2020_{state}_{geo_type}.zip"
            if not ids or not poly_path.exists():
                continue
            polys = _read_state_geo(poly_path, set(ids)).to_crs(blocks.crs)
            joined = gpd.sjoin(blocks[["pop20", "geometry"]], polys[["geoid", "geometry"]], predicate="within")
            for geoid, population in joined.groupby("geoid")["pop20"].sum().items():
                out[ids[str(geoid)]] = float(population)
    return pd.Series(out, dtype=float)


def county_growth(paths, *, year: int) -> pd.Series:
    """County population in `year` over its 2020 estimate (Census PopEst vintage 2025)."""
    path = Path(paths.data_dir) / "Census-PopEst-2020-2025" / "co-est2025-alldata.csv"
    frame = pd.read_csv(path, encoding="latin-1", dtype={"STATE": str, "COUNTY": str, "SUMLEV": str})
    frame = frame[frame["SUMLEV"].eq("050")]
    geoid = frame["STATE"].str.zfill(2) + frame["COUNTY"].str.zfill(3)
    base = pd.to_numeric(frame["POPESTIMATE2020"], errors="coerce")
    return pd.Series(
        (pd.to_numeric(frame[f"POPESTIMATE{int(year)}"], errors="coerce") / base.where(base.gt(0))).to_numpy(),
        index=geoid.to_numpy(),
    )


def compute_territory_check(paths, *, year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(check table, relationships in force, dispositions) on the repo's inputs; writes nothing.
    Refuses a researched relationship whose territory resolves to no population."""
    from crimerisk.service_scopes import load_custom_footprint_population_by_target

    relationships, dispositions = build_territory_dispositions(paths, year=int(year))
    crosswalk = load_crosswalk_for_relationships(paths)
    obs = pd.read_parquet(
        Path(paths.state_dir) / "observations" / "agency_year_observations.parquet",
        columns=["ori9", "year", "population", "agency_type_norm", "production_scope_excluded"],
    )
    obs = obs[~obs["production_scope_excluded"].fillna(False).astype(bool)]
    agency_types = (
        obs.assign(ori9=obs["ori9"].astype("string").str.upper())
        .sort_values("year")
        .drop_duplicates("ori9", keep="last")
        .set_index("ori9")["agency_type_norm"]
    )
    bg = pd.read_parquet(
        Path(paths.state_dir) / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        columns=["block_group_geoid", "jurisdiction_id", "pop20"],
    )
    bg["pop20"] = pd.to_numeric(bg["pop20"], errors="coerce").fillna(0.0)
    bg["pop_check"] = bg["pop20"] * county_growth(
        paths, year=TERRITORY_CHECK_POPULATION_YEAR
    ).reindex(bg["block_group_geoid"].astype(str).str[:5]).fillna(1.0).to_numpy()

    config = load_relationship_config(paths)
    census = territory_populations(bg, column="pop20")
    territories = set(config["territory_jurisdiction_id"].astype(str))
    absent = sorted(t for t in territories if not census.get(t, 0.0) > 0.0)
    if absent:
        census = pd.concat([census, _tiger_territory_population(paths, absent)])
        census = census[~census.index.duplicated(keep="last")]
    validate_relationship_territories(config, territory_population=census)

    custom = load_custom_footprint_population_by_target(paths)
    flags = territory_check(
        served=served_population(obs),
        crosswalk=crosswalk[crosswalk["ori9"].isin(set(agency_types.index))],
        jurisdiction_population=territory_populations(bg, column="pop_check"),
        dispositions=dispositions,
        relationships=relationships,
        exceptions=load_territory_exceptions(paths),
        agency_types=agency_types,
        fbi_covered_oris=frozenset(
            load_latest_covered_by_relationships(paths, year=int(year))["ori9"].astype(str).str.upper()
        ),
        # The custom-footprint helper returns 2020-census populations.
        custom_population=custom.set_index(custom["ori9"].astype(str))["_covered_population"],
    )
    return flags, relationships, dispositions


def run_territory_check(paths, *, year: int, blocking: bool) -> pd.DataFrame:
    """The build-time check on the repo's inputs; writes the check table, and raises when
    `blocking` and any flag is unresolved."""
    flags, relationships, dispositions = compute_territory_check(paths, year=int(year))
    out = Path(paths.state_dir) / "controls" / f"service_territory_check_{int(year)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    flags.to_csv(out, index=False)
    table = service_relationships_path(paths, year=int(year))
    table.parent.mkdir(parents=True, exist_ok=True)
    relationships.merge(
        dispositions[["territory_jurisdiction_id", "disposition", "principal_jurisdiction_id"]],
        on="territory_jurisdiction_id",
        how="left",
    ).to_parquet(table, index=False)
    if blocking:
        assert_territories_resolved(flags)
    return flags
