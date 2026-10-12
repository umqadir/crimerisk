from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from crimerisk.build_freshness import artifact_is_current, write_dependency_stamp
from crimerisk.crosswalk_shares import (
    assert_allocation_shares_conserve,
    normalize_block_group_allocation_shares,
)
from crimerisk.paths import RepoPaths
from crimerisk.pa_police_service import (
    load_pa_police_service_mcd_map,
    pa_mcd_baf_path,
    pa_police_service_dependency_paths,
    read_pa_mcd_baf,
)
from crimerisk.service_relationships import (
    DISPOSITION_MERGED,
    SOURCE_CONSOLIDATED,
    build_territory_dispositions,
)
from crimerisk.stage_locks import stage_write_lock
from crimerisk.scope import PRODUCTION_SCOPE_EXCLUDE


_GEOMETRY_MAX_WORKERS = 6


@dataclass(frozen=True)
class GeometryBuildConfig:
    exclude_scope_state_abbrs: tuple[str, ...] = tuple(sorted(PRODUCTION_SCOPE_EXCLUDE))
    # The data year whose service relationships set municipal territory.
    year: int = 2025


def _load_jurisdiction_master(paths: RepoPaths) -> pd.DataFrame:
    path = paths.state_dir / "reference" / "jurisdiction_master.parquet"
    return pd.read_parquet(path)


def _state_scope(paths: RepoPaths, config: GeometryBuildConfig) -> list[str]:
    """The states the crosswalk covers: the reference jurisdiction master minus the excluded scope.

    Read from the reference layer (stage 3), never from the controls (which read this crosswalk).
    """
    juris = _load_jurisdiction_master(paths)
    juris["state_fips"] = juris["state_fips"].astype("string").str.zfill(2)
    juris["state_abbr"] = juris["state_abbr"].astype("string").str.upper()
    juris = juris[~juris["state_abbr"].isin(set(config.exclude_scope_state_abbrs))]
    return sorted(juris["state_fips"].dropna().unique().tolist())


def _read_state_blocks(path: Path) -> gpd.GeoDataFrame:
    cols = [
        "STATEFP20",
        "COUNTYFP20",
        "TRACTCE20",
        "BLOCKCE20",
        "GEOID20",
        "ALAND20",
        "AWATER20",
        "INTPTLAT20",
        "INTPTLON20",
        "HOUSING20",
        "POP20",
    ]
    df = pyogrio.read_dataframe(path, columns=cols, read_geometry=False)
    df["INTPTLON20"] = pd.to_numeric(df["INTPTLON20"], errors="coerce")
    df["INTPTLAT20"] = pd.to_numeric(df["INTPTLAT20"], errors="coerce")
    blocks = gpd.GeoDataFrame(
        df.rename(columns=str.lower),
        geometry=gpd.points_from_xy(df["INTPTLON20"], df["INTPTLAT20"], crs="EPSG:4269"),
        crs="EPSG:4269",
    )
    blocks["state_fips"] = blocks["statefp20"].astype("string").str.zfill(2)
    blocks["county_fips"] = blocks["countyfp20"].astype("string").str.zfill(3)
    blocks["tract_geoid"] = blocks["state_fips"] + blocks["county_fips"] + blocks["tractce20"].astype("string")
    blocks["block_group_geoid"] = blocks["tract_geoid"] + blocks["blockce20"].astype("string").str[0]
    blocks["block_geoid"] = blocks["geoid20"].astype("string")
    for col in ["aland20", "awater20", "housing20", "pop20"]:
        blocks[col] = pd.to_numeric(blocks[col], errors="coerce").fillna(0.0)
    return blocks[
        [
            "block_geoid",
            "state_fips",
            "county_fips",
            "tract_geoid",
            "block_group_geoid",
            "aland20",
            "awater20",
            "housing20",
            "pop20",
            "geometry",
        ]
    ].copy()


def _read_state_block_polygons(path: Path) -> gpd.GeoDataFrame:
    cols = [
        "STATEFP20",
        "COUNTYFP20",
        "TRACTCE20",
        "BLOCKCE20",
        "GEOID20",
        "ALAND20",
        "AWATER20",
        "HOUSING20",
        "POP20",
    ]
    blocks = pyogrio.read_dataframe(path, columns=cols)
    blocks = blocks.rename(columns=str.lower)
    blocks["state_fips"] = blocks["statefp20"].astype("string").str.zfill(2)
    blocks["county_fips"] = blocks["countyfp20"].astype("string").str.zfill(3)
    blocks["tract_geoid"] = blocks["state_fips"] + blocks["county_fips"] + blocks["tractce20"].astype("string")
    blocks["block_group_geoid"] = blocks["tract_geoid"] + blocks["blockce20"].astype("string").str[0]
    blocks["block_geoid"] = blocks["geoid20"].astype("string")
    for col in ["aland20", "awater20", "housing20", "pop20"]:
        blocks[col] = pd.to_numeric(blocks[col], errors="coerce").fillna(0.0)
    return blocks[
        [
            "block_geoid",
            "state_fips",
            "county_fips",
            "tract_geoid",
            "block_group_geoid",
            "aland20",
            "awater20",
            "housing20",
            "pop20",
            "geometry",
        ]
    ].copy()


def _read_state_geo(path: Path, geoid_values: set[str]) -> gpd.GeoDataFrame:
    if not geoid_values:
        return gpd.GeoDataFrame(columns=["geoid", "namelsad", "geometry"], geometry="geometry", crs="EPSG:4269")
    gdf = pyogrio.read_dataframe(path)
    geoid_col = "GEOID" if "GEOID" in gdf.columns else "GEOID20"
    name_col = "NAMELSAD" if "NAMELSAD" in gdf.columns else "NAME"
    gdf["geoid"] = gdf[geoid_col].astype("string")
    gdf = gdf[gdf["geoid"].isin(geoid_values)].copy()
    gdf["namelsad"] = gdf[name_col].astype("string")
    return gdf[["geoid", "namelsad", "geometry"]].copy()


MERGE_SEPARATOR = ">"


def _load_territory_moves(
    paths: RepoPaths, config: GeometryBuildConfig | None = None
) -> frozenset[str]:
    """The municipal territories a service relationship moves, for the configured data year.

    Each element is a territory jurisdiction_id that leaves its own polygon for the state
    remainder (its provider is a sheriff, county or state unit), or
    `territory>principal_jurisdiction_id` when it joins its provider's municipal jurisdiction
    (a contracting city, a multi-town department). See `crimerisk.service_relationships`.
    Whether a municipality's own ORI reports never enters: a department that does not file
    keeps its territory, and its control is imputed. Consolidated-footprint rows are applied
    by the consolidated-footprint geometry below and are not repeated here.
    """
    year = int((config or GeometryBuildConfig()).year)
    _relationships, dispositions = build_territory_dispositions(paths, year=year)
    if dispositions.empty:
        return frozenset()
    dispositions = dispositions[dispositions["source"].ne(SOURCE_CONSOLIDATED)]
    out = set()
    for row in dispositions.itertuples(index=False):
        if row.disposition == DISPOSITION_MERGED:
            out.add(f"{row.territory_jurisdiction_id}{MERGE_SEPARATOR}{row.principal_jurisdiction_id}")
        else:
            out.add(str(row.territory_jurisdiction_id))
    return frozenset(out)


def _moved_territory_ids(territory_moves: frozenset[str]) -> frozenset[str]:
    return frozenset(str(value).split(MERGE_SEPARATOR, 1)[0] for value in territory_moves)


def _territory_merges(territory_moves: frozenset[str]) -> dict[str, str]:
    return dict(
        str(value).split(MERGE_SEPARATOR, 1) for value in territory_moves if MERGE_SEPARATOR in str(value)
    )


def _load_state_municipal_polygons(
    paths: RepoPaths,
    state_fips: str,
    jurisdiction_master: pd.DataFrame,
    *,
    territory_moves: frozenset[str] = frozenset(),
) -> gpd.GeoDataFrame:
    muni = jurisdiction_master[
        (jurisdiction_master["jurisdiction_type"] == "municipal")
        & (jurisdiction_master["state_fips"].astype("string").str.zfill(2) == str(state_fips).zfill(2))
        & (~jurisdiction_master["jurisdiction_id"].astype("string").isin(_moved_territory_ids(territory_moves)))
    ].copy()
    if muni.empty:
        return gpd.GeoDataFrame(columns=["jurisdiction_id", "geo_type", "geoid", "jurisdiction_name", "geometry"], geometry="geometry", crs="EPSG:4269")

    place_geoids = set(muni.loc[muni["geo_type"] == "place", "geoid"].astype("string"))
    cousub_geoids = set(muni.loc[muni["geo_type"] == "cousub", "geoid"].astype("string"))

    frames: list[gpd.GeoDataFrame] = []
    place_path = paths.data_dir / "tiger_places" / f"tl_2020_{str(state_fips).zfill(2)}_place.zip"
    if place_geoids and place_path.exists():
        place = _read_state_geo(place_path, place_geoids)
        if not place.empty:
            place["geo_type"] = "place"
            frames.append(place)

    cousub_path = paths.data_dir / "tiger_cousub" / f"tl_2020_{str(state_fips).zfill(2)}_cousub.zip"
    if cousub_geoids and cousub_path.exists():
        cousub = _read_state_geo(cousub_path, cousub_geoids)
        if not cousub.empty:
            cousub["geo_type"] = "cousub"
            frames.append(cousub)

    if not frames:
        return gpd.GeoDataFrame(columns=["jurisdiction_id", "geo_type", "geoid", "jurisdiction_name", "geometry"], geometry="geometry", crs="EPSG:4269")

    polys = pd.concat(frames, ignore_index=True)
    polys = gpd.GeoDataFrame(polys, geometry="geometry", crs=frames[0].crs)
    muni["geoid"] = muni["geoid"].astype("string")
    muni["geo_type"] = muni["geo_type"].astype("string")
    polys = polys.merge(
        muni[["jurisdiction_id", "geo_type", "geoid", "jurisdiction_name"]],
        on=["geo_type", "geoid"],
        how="inner",
    )

    # A consolidated government can have multiple disjoint Census place pieces.
    # Louisville's current balance (2148006) and legally retained old-city piece
    # (2148000) are one LMPD footprint; treating the latter as statewide remainder
    # drops downtown from every place rollup.  The consolidated-footprint registry
    # is the single source for both geometry and allocation scope.
    consolidated_path = paths.repo_root / "configs" / "consolidated_agency_footprints.csv"
    if consolidated_path.exists():
        consolidated = pd.read_csv(consolidated_path, dtype="string")
        required = {"state_fips", "principal_jurisdiction_id", "included_place_geoids"}
        missing = required - set(consolidated.columns)
        if missing:
            raise ValueError(
                "Consolidated agency footprints missing geometry columns: "
                f"{sorted(missing)}"
            )
        consolidated["state_fips"] = consolidated["state_fips"].str.zfill(2)
        supplemental_rows: list[dict[str, str]] = []
        for row in consolidated.loc[
            consolidated["state_fips"].eq(str(state_fips).zfill(2))
            & consolidated["included_place_geoids"].notna()
        ].itertuples(index=False):
            principal_id = str(row.principal_jurisdiction_id)
            principal = muni[muni["jurisdiction_id"].astype(str).eq(principal_id)]
            if principal.empty:
                raise ValueError(
                    f"Consolidated supplemental geometry missing principal {principal_id}"
                )
            for geoid in str(row.included_place_geoids).split("|"):
                geoid = geoid.strip()
                if geoid:
                    supplemental_rows.append(
                        {
                            "geoid": geoid,
                            "jurisdiction_id": principal_id,
                            "jurisdiction_name": str(principal.iloc[0]["jurisdiction_name"]),
                        }
                    )
        if supplemental_rows:
            supplemental_registry = pd.DataFrame(supplemental_rows)
            supplement = _read_state_geo(
                place_path, set(supplemental_registry["geoid"].astype(str))
            )
            missing_geoids = set(supplemental_registry["geoid"].astype(str)) - set(
                supplement["geoid"].astype(str)
            )
            if missing_geoids:
                raise ValueError(
                    "Consolidated supplemental place geometry missing from TIGER: "
                    f"{sorted(missing_geoids)}"
                )
            supplement = supplement.merge(
                supplemental_registry, on="geoid", how="inner"
            )
            supplement["geo_type"] = "place"
            polys = pd.concat(
                [polys, supplement[polys.columns]], ignore_index=True
            )
            polys = gpd.GeoDataFrame(polys, geometry="geometry", crs="EPSG:4269")
    merges = {
        territory: principal
        for territory, principal in _territory_merges(territory_moves).items()
        if territory.startswith(f"{str(state_fips).zfill(2)}:")
    }
    if merges:
        members = jurisdiction_master[
            jurisdiction_master["jurisdiction_id"].astype("string").isin(set(merges))
        ][["jurisdiction_id", "geo_type", "geoid"]].copy()
        # A served town the reference layer never listed (Larkspur CA under Central Marin) is
        # read from TIGER by the place or county-subdivision GEOID in its jurisdiction id.
        unlisted = sorted(set(merges) - set(members["jurisdiction_id"].astype(str)))
        if unlisted:
            parsed = pd.DataFrame(
                [jid.split(":") for jid in unlisted], columns=["state", "kind", "geo_type", "geoid"]
            )
            if not parsed["kind"].eq("municipal").all() or not parsed["geo_type"].isin(["place", "cousub"]).all():
                raise ValueError(f"Service-relationship territories are not municipal places: {unlisted}")
            members = pd.concat(
                [members, parsed.assign(jurisdiction_id=unlisted)[["jurisdiction_id", "geo_type", "geoid"]]],
                ignore_index=True,
            )
        names = jurisdiction_master.set_index(jurisdiction_master["jurisdiction_id"].astype(str))["jurisdiction_name"]
        member_frames = []
        for geo_type, path in (("place", place_path), ("cousub", cousub_path)):
            wanted = members[members["geo_type"].astype(str).eq(geo_type)]
            if wanted.empty:
                continue
            geo = _read_state_geo(path, set(wanted["geoid"].astype(str)))
            geo = geo.merge(
                wanted.assign(geoid=wanted["geoid"].astype("string")), on="geoid", how="inner"
            )
            if len(geo) != len(wanted):
                raise ValueError(f"Service-relationship territory geometry missing from TIGER {geo_type}")
            geo["geo_type"] = geo_type
            geo["jurisdiction_id"] = geo["jurisdiction_id"].astype(str).map(merges)
            missing_principal = sorted(set(geo["jurisdiction_id"]) - set(polys["jurisdiction_id"].astype(str)))
            if missing_principal:
                raise ValueError(f"Service-relationship principals without geometry: {missing_principal}")
            geo["jurisdiction_name"] = geo["jurisdiction_id"].map(names)
            member_frames.append(geo[["geoid", "namelsad", "geometry", "geo_type", "jurisdiction_id", "jurisdiction_name"]])
        if member_frames:
            polys = gpd.GeoDataFrame(
                pd.concat([polys, *[f[polys.columns] for f in member_frames]], ignore_index=True),
                geometry="geometry",
                crs=polys.crs,
            )
    polys_aea = polys.to_crs("EPSG:5070")
    polys["poly_area"] = polys_aea.geometry.area
    return polys[["jurisdiction_id", "geo_type", "geoid", "jurisdiction_name", "poly_area", "geometry"]].copy()


def _recover_missing_point_join_municipal_assignments(
    *,
    block_zip: Path,
    assigned: pd.DataFrame,
    municipal_polys: gpd.GeoDataFrame,
) -> pd.DataFrame:
    matched_ids = set(assigned["jurisdiction_id"].dropna().astype("string"))
    missing_polys = municipal_polys[~municipal_polys["jurisdiction_id"].isin(matched_ids)].copy()
    if missing_polys.empty:
        return pd.DataFrame(columns=["block_geoid", "jurisdiction_id", "municipal_candidate_count"])

    unmatched_blocks = set(assigned.loc[assigned["jurisdiction_id"].isna(), "block_geoid"].astype("string"))
    if not unmatched_blocks:
        return pd.DataFrame(columns=["block_geoid", "jurisdiction_id", "municipal_candidate_count"])

    block_polys = _read_state_block_polygons(block_zip)
    block_polys = block_polys[block_polys["block_geoid"].isin(unmatched_blocks)].copy()
    if block_polys.empty:
        return pd.DataFrame(columns=["block_geoid", "jurisdiction_id", "municipal_candidate_count"])

    candidates = gpd.sjoin(
        block_polys[["block_geoid", "geometry"]],
        missing_polys[["jurisdiction_id", "poly_area", "geometry"]],
        how="inner",
        predicate="intersects",
    )[["block_geoid", "jurisdiction_id", "poly_area"]].drop_duplicates()
    if candidates.empty:
        return pd.DataFrame(columns=["block_geoid", "jurisdiction_id", "municipal_candidate_count"])

    pair_geo = candidates.merge(
        block_polys[["block_geoid", "geometry"]].rename(columns={"geometry": "geometry_block"}),
        on="block_geoid",
        how="left",
    ).merge(
        missing_polys[["jurisdiction_id", "geometry"]].rename(columns={"geometry": "geometry_jurisdiction"}),
        on="jurisdiction_id",
        how="left",
    )
    pair_geo = gpd.GeoDataFrame(pair_geo, geometry="geometry_block", crs=block_polys.crs)
    pair_geo["geometry"] = pair_geo["geometry_block"].intersection(pair_geo["geometry_jurisdiction"])
    pair_geo = pair_geo.set_geometry("geometry")[["block_geoid", "jurisdiction_id", "poly_area", "geometry"]]
    pair_geo = pair_geo[pair_geo.geometry.notna() & ~pair_geo.geometry.is_empty].copy()
    if pair_geo.empty:
        return pd.DataFrame(columns=["block_geoid", "jurisdiction_id", "municipal_candidate_count"])

    pair_geo_aea = pair_geo.to_crs("EPSG:5070")
    pair_geo["overlap_area"] = pair_geo_aea.geometry.area
    pair_geo = pair_geo[pair_geo["overlap_area"].gt(0)].copy()
    if pair_geo.empty:
        return pd.DataFrame(columns=["block_geoid", "jurisdiction_id", "municipal_candidate_count"])

    pair_geo["municipal_candidate_count"] = pair_geo.groupby("block_geoid")["jurisdiction_id"].transform("nunique")
    pair_geo = pair_geo.sort_values(
        ["block_geoid", "overlap_area", "poly_area"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    recovered = pair_geo.drop_duplicates(subset=["block_geoid"], keep="first")
    return recovered[["block_geoid", "jurisdiction_id", "municipal_candidate_count"]].copy()


def build_state_block_assignments(
    *,
    paths: RepoPaths,
    state_fips: str,
    config: GeometryBuildConfig = GeometryBuildConfig(),
    territory_moves: frozenset[str] | None = None,
) -> pd.DataFrame:
    jurisdiction_master = _load_jurisdiction_master(paths)
    jurisdiction_master["state_fips"] = jurisdiction_master["state_fips"].astype("string").str.zfill(2)
    jurisdiction_master["state_abbr"] = jurisdiction_master["state_abbr"].astype("string").str.upper()
    jurisdiction_master = jurisdiction_master[~jurisdiction_master["state_abbr"].isin(set(config.exclude_scope_state_abbrs))].copy()

    if territory_moves is None:
        territory_moves = _load_territory_moves(paths, config)

    block_zip = paths.data_dir / "tiger_tabblock20" / f"tl_2020_{str(state_fips).zfill(2)}_tabblock20.zip"
    if not block_zip.exists():
        raise FileNotFoundError(block_zip)

    blocks = _read_state_blocks(block_zip)
    if str(state_fips).zfill(2) == "42":
        baf = read_pa_mcd_baf(paths)
        service = load_pa_police_service_mcd_map(paths)
        if set(baf["block_geoid"].astype(str)) != set(blocks["block_geoid"].astype(str)):
            missing_baf = set(blocks["block_geoid"].astype(str)) - set(baf["block_geoid"].astype(str))
            missing_tiger = set(baf["block_geoid"].astype(str)) - set(blocks["block_geoid"].astype(str))
            raise ValueError(
                "Pennsylvania BAF/TIGER block universes differ: "
                f"missing_baf={len(missing_baf):,}, missing_tiger={len(missing_tiger):,}"
            )
        assigned = (
            blocks.drop(columns="geometry")
            .merge(baf, on="block_geoid", how="left", validate="one_to_one")
            .merge(service, on="mcd_geoid_2020", how="left", validate="many_to_one")
        )
        missing_service = assigned[assigned["service_area_id"].isna()]
        if not missing_service.empty:
            summary = (
                missing_service.groupby("mcd_geoid_2020", dropna=False)
                .agg(blocks=("block_geoid", "size"), population=("pop20", "sum"))
                .reset_index()
            )
            raise ValueError(
                "Pennsylvania blocks have no reviewed target-year police service: "
                + str(summary.to_dict(orient="records"))
            )
        assigned["jurisdiction_id"] = assigned["service_area_id"].astype("string")
        assigned["jurisdiction_type"] = "municipal"
        assigned["assignment_method"] = "census_2020_baf_mcd_to_pccd_service"
        assigned["municipal_candidate_count"] = 1
        return assigned[
            [
                "block_geoid",
                "state_fips",
                "county_fips",
                "tract_geoid",
                "block_group_geoid",
                "aland20",
                "awater20",
                "housing20",
                "pop20",
                "jurisdiction_id",
                "jurisdiction_type",
                "assignment_method",
                "municipal_candidate_count",
            ]
        ].copy()
    polys = _load_state_municipal_polygons(
        paths,
        str(state_fips).zfill(2),
        jurisdiction_master,
        territory_moves=territory_moves,
    )

    if polys.empty:
        assigned = blocks.copy()
        assigned["jurisdiction_id"] = f"{str(state_fips).zfill(2)}:state_nonmunicipal_remainder"
        assigned["jurisdiction_type"] = "state_nonmunicipal_remainder"
        assigned["assignment_method"] = "state_remainder_no_municipal_polygons"
        assigned["municipal_candidate_count"] = 0
        return pd.DataFrame(assigned.drop(columns="geometry"))

    joined = gpd.sjoin(
        blocks,
        polys[["jurisdiction_id", "geo_type", "geoid", "poly_area", "geometry"]],
        how="left",
        predicate="within",
    ).reset_index(drop=True)
    joined["match_rank"] = joined.groupby("block_geoid").cumcount() + 1
    joined["municipal_candidate_count"] = joined.groupby("block_geoid")["jurisdiction_id"].transform(lambda s: int(s.notna().sum()))
    joined = joined.sort_values(["block_geoid", "poly_area"], ascending=[True, True], kind="mergesort")
    assigned = joined.drop_duplicates(subset=["block_geoid"], keep="first").copy()

    assigned["assignment_method"] = "state_remainder_no_match"
    matched = assigned["jurisdiction_id"].notna()
    assigned.loc[matched, "assignment_method"] = "municipal_point_join"
    assigned.loc[matched & assigned["municipal_candidate_count"].gt(1), "assignment_method"] = "municipal_overlap_smallest_polygon"

    recovered = _recover_missing_point_join_municipal_assignments(
        block_zip=block_zip,
        assigned=assigned,
        municipal_polys=polys,
    )
    if not recovered.empty:
        assigned = assigned.merge(
            recovered.rename(
                columns={
                    "jurisdiction_id": "recovered_jurisdiction_id",
                    "municipal_candidate_count": "recovered_municipal_candidate_count",
                }
            ),
            on="block_geoid",
            how="left",
        )
        recover_mask = assigned["jurisdiction_id"].isna() & assigned["recovered_jurisdiction_id"].notna()
        assigned.loc[recover_mask, "jurisdiction_id"] = assigned.loc[recover_mask, "recovered_jurisdiction_id"]
        assigned.loc[recover_mask, "municipal_candidate_count"] = assigned.loc[
            recover_mask, "recovered_municipal_candidate_count"
        ]
        assigned.loc[recover_mask, "assignment_method"] = "municipal_polygon_overlap_recovery"
        assigned = assigned.drop(
            columns=["recovered_jurisdiction_id", "recovered_municipal_candidate_count"],
            errors="ignore",
        )

    matched = assigned["jurisdiction_id"].notna()
    assigned.loc[~matched, "jurisdiction_id"] = f"{str(state_fips).zfill(2)}:state_nonmunicipal_remainder"
    assigned["jurisdiction_type"] = "state_nonmunicipal_remainder"
    assigned.loc[assigned["jurisdiction_id"].notna() & ~assigned["jurisdiction_id"].astype("string").str.endswith(":state_nonmunicipal_remainder"), "jurisdiction_type"] = "municipal"

    return pd.DataFrame(
        assigned[
            [
                "block_geoid",
                "state_fips",
                "county_fips",
                "tract_geoid",
                "block_group_geoid",
                "aland20",
                "awater20",
                "housing20",
                "pop20",
                "jurisdiction_id",
                "jurisdiction_type",
                "assignment_method",
                "municipal_candidate_count",
            ]
        ].copy()
    )


def _build_missing_municipal_bg_allocations(
    *,
    paths: RepoPaths,
    bg_crosswalk: pd.DataFrame,
    jurisdiction_master: pd.DataFrame,
    territory_moves: frozenset[str] | None = None,
) -> pd.DataFrame:
    if territory_moves is None:
        territory_moves = _load_territory_moves(paths)

    municipal = jurisdiction_master[
        jurisdiction_master["jurisdiction_type"].eq("municipal")
        & (~jurisdiction_master["jurisdiction_id"].astype("string").isin(_moved_territory_ids(territory_moves)))
    ].copy()
    if municipal.empty:
        return pd.DataFrame(columns=list(bg_crosswalk.columns))

    supported = set(
        bg_crosswalk.loc[
            bg_crosswalk["jurisdiction_type"].astype("string").eq("municipal"),
            "jurisdiction_id",
        ].astype("string")
    )
    missing = municipal[~municipal["jurisdiction_id"].astype("string").isin(supported)][
        ["state_fips", "jurisdiction_id"]
    ].drop_duplicates()
    if missing.empty:
        return pd.DataFrame(columns=list(bg_crosswalk.columns))

    template_cols = list(bg_crosswalk.columns)
    frames: list[pd.DataFrame] = []
    for state_fips, state_missing in missing.groupby("state_fips", dropna=False):
        state_fips = str(state_fips).zfill(2)
        muni_polys = _load_state_municipal_polygons(
            paths,
            state_fips,
            jurisdiction_master,
            territory_moves=territory_moves,
        )
        muni_polys = muni_polys[muni_polys["jurisdiction_id"].isin(state_missing["jurisdiction_id"])].copy()
        if muni_polys.empty:
            continue

        block_polys = _read_state_block_polygons(
            paths.data_dir / "tiger_tabblock20" / f"tl_2020_{state_fips}_tabblock20.zip"
        )
        if block_polys.empty:
            continue

        pairs = gpd.sjoin(
            block_polys[["block_geoid", "block_group_geoid", "geometry"]],
            muni_polys[["jurisdiction_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )[["block_geoid", "block_group_geoid", "jurisdiction_id"]].drop_duplicates()
        if pairs.empty:
            continue

        pair_geo = pairs.merge(
            block_polys[["block_geoid", "block_group_geoid", "geometry"]].rename(columns={"geometry": "geometry_block"}),
            on=["block_geoid", "block_group_geoid"],
            how="left",
        ).merge(
            muni_polys[["jurisdiction_id", "geometry"]].rename(columns={"geometry": "geometry_jurisdiction"}),
            on="jurisdiction_id",
            how="left",
        )
        pair_geo = gpd.GeoDataFrame(pair_geo, geometry="geometry_block", crs=block_polys.crs)
        pair_geo["geometry"] = pair_geo["geometry_block"].intersection(pair_geo["geometry_jurisdiction"])
        pair_geo = pair_geo.set_geometry("geometry")[["block_group_geoid", "jurisdiction_id", "geometry"]]
        pair_geo = pair_geo[pair_geo.geometry.notna() & ~pair_geo.geometry.is_empty].copy()
        if pair_geo.empty:
            continue

        pair_geo_aea = pair_geo.to_crs("EPSG:5070")
        pair_geo["weight"] = pair_geo_aea.geometry.area
        pair_geo = pair_geo[pair_geo["weight"].gt(0)].copy()
        if pair_geo.empty:
            continue

        bg_weights = (
            pair_geo.groupby(["block_group_geoid", "jurisdiction_id"], dropna=False)["weight"]
            .sum()
            .reset_index()
        )
        totals = (
            bg_weights.groupby("block_group_geoid", dropna=False)["weight"]
            .sum()
            .rename("block_group_weight_total")
            .reset_index()
        )
        bg_weights = bg_weights.merge(totals, on="block_group_geoid", how="left")
        bg_weights = bg_weights[bg_weights["block_group_weight_total"].gt(0)].copy()
        if bg_weights.empty:
            continue

        bg_weights["state_fips"] = state_fips
        bg_weights["jurisdiction_type"] = "municipal"
        bg_weights["allocation_share"] = bg_weights["weight"] / bg_weights["block_group_weight_total"]
        out = bg_crosswalk.head(0).reindex(index=range(len(bg_weights))).copy()
        out["state_fips"] = bg_weights["state_fips"].astype("string")
        out["block_group_geoid"] = bg_weights["block_group_geoid"].astype("string")
        out["jurisdiction_id"] = bg_weights["jurisdiction_id"].astype("string")
        out["jurisdiction_type"] = "municipal"
        if "allocation_share" in out.columns:
            out["allocation_share"] = pd.to_numeric(bg_weights["allocation_share"], errors="coerce")
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=template_cols)
    return pd.concat(frames, ignore_index=True)


def _rebuild_state_block_assignments(
    job: tuple[RepoPaths, str, GeometryBuildConfig, frozenset[str], Path],
) -> tuple[str, pd.DataFrame]:
    """Process-pool entry point: one independent per-state spatial join."""
    paths, state_fips, config, territory_moves, state_out = job
    state_df = build_state_block_assignments(
        paths=paths,
        state_fips=state_fips,
        config=config,
        territory_moves=territory_moves,
    )
    state_df.to_parquet(state_out, index=False)
    write_dependency_stamp(
        state_out,
        _geometry_state_dependency_paths(paths, state_fips=state_fips),
        parameters=_state_membership_parameters(territory_moves, state_fips=state_fips),
    )
    return state_fips, state_df


def build_block_crosswalk(
    *,
    paths: RepoPaths,
    state_fips_values: list[str] | None = None,
    out_dir: Path | None = None,
    config: GeometryBuildConfig = GeometryBuildConfig(),
    force_rebuild: bool = False,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = [str(state_fips).zfill(2) for state_fips in (state_fips_values or _state_scope(paths, config))]
    geometry_dir = out_dir or (paths.state_dir / "geometry")
    state_dir = geometry_dir / "blocks_by_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    territory_moves = _load_territory_moves(paths, config)

    by_state: dict[str, pd.DataFrame] = {}
    jobs: list[tuple[RepoPaths, str, GeometryBuildConfig, frozenset[str], Path]] = []
    for state_fips in states:
        state_out = state_dir / f"{state_fips}.parquet"
        # Per-state caches bake in the territory moves at spatial-join time, so a cache is
        # reused only while its static inputs AND that state's moves match.
        if not force_rebuild and artifact_is_current(
            state_out,
            _geometry_state_dependency_paths(paths, state_fips=state_fips),
            parameters=_state_membership_parameters(territory_moves, state_fips=state_fips),
        ):
            by_state[state_fips] = pd.read_parquet(state_out)
        else:
            jobs.append((paths, state_fips, config, territory_moves, state_out))

    # The joins are independent (own inputs, own output file) and the serial loop is the
    # single largest cost of a geometry rebuild. Workers are capped well below the core
    # count: TIGER block layers for the large states are hundreds of MB and geopandas
    # peaks high, so this is memory-bound rather than CPU-bound.
    workers = max_workers if max_workers is not None else min(_GEOMETRY_MAX_WORKERS, os.cpu_count() or 1)
    if len(jobs) > 1 and workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                for state_fips, state_df in pool.map(_rebuild_state_block_assignments, jobs, chunksize=1):
                    by_state[state_fips] = state_df
        except (BrokenProcessPool, PermissionError):
            # A worker died (memory pressure, or a host that cannot spawn -- e.g. an
            # unguarded `__main__`) or the host forbids the semaphore-limit probe used
            # by ProcessPoolExecutor. Errors raised inside the join propagate normally;
            # these executor failures fall through to the serial retry below.
            pass
    for job in jobs:
        if job[1] not in by_state:
            state_fips, state_df = _rebuild_state_block_assignments(job)
            by_state[state_fips] = state_df

    # Ordered explicitly so the concatenation never depends on worker completion order.
    state_frames = [by_state[state_fips] for state_fips in states]
    block_df = pd.concat(state_frames, ignore_index=True).sort_values(["state_fips", "block_geoid"]).reset_index(drop=True)

    bg = (
        block_df.groupby(["state_fips", "block_group_geoid", "jurisdiction_id", "jurisdiction_type"], dropna=False)
        .agg(
            blocks=("block_geoid", "count"),
            aland20=("aland20", "sum"),
            awater20=("awater20", "sum"),
            housing20=("housing20", "sum"),
            pop20=("pop20", "sum"),
        )
        .reset_index()
    )
    totals = (
        block_df.groupby(["state_fips", "block_group_geoid"], dropna=False)
        .agg(
            total_blocks=("block_geoid", "count"),
            total_aland20=("aland20", "sum"),
            total_awater20=("awater20", "sum"),
            total_housing20=("housing20", "sum"),
            total_pop20=("pop20", "sum"),
        )
        .reset_index()
    )
    bg = bg.merge(totals, on=["state_fips", "block_group_geoid"], how="left")
    bg["block_share"] = bg["blocks"] / bg["total_blocks"]
    bg["aland_share"] = bg["aland20"] / bg["total_aland20"]
    bg["housing_share"] = bg["housing20"] / bg["total_housing20"]
    bg["pop_share"] = bg["pop20"] / bg["total_pop20"]
    bg = normalize_block_group_allocation_shares(bg)

    jurisdiction_master = _load_jurisdiction_master(paths)
    jurisdiction_master["state_fips"] = jurisdiction_master["state_fips"].astype("string").str.zfill(2)
    supplemental_bg = _build_missing_municipal_bg_allocations(
        paths=paths,
        bg_crosswalk=bg,
        jurisdiction_master=jurisdiction_master,
        territory_moves=territory_moves,
    )
    if not supplemental_bg.empty:
        bg = pd.concat([bg, supplemental_bg.reindex(columns=bg.columns)], ignore_index=True)
        bg = normalize_block_group_allocation_shares(bg)
    # The Stage-4 recipient floor can zero a share, so "shares sum to 1 per block group"
    # stops being structural and becomes load-bearing: assert it here rather than leave it
    # as a property the audit has to rediscover.
    assert_allocation_shares_conserve(bg)
    return block_df, bg


def write_v2_geometry(
    *,
    paths: RepoPaths,
    block_out_path: Path,
    block_group_out_path: Path,
    config: GeometryBuildConfig = GeometryBuildConfig(),
    force_rebuild: bool = False,
    blocked_by: tuple[str, ...] | None = None,
) -> tuple[Path, Path]:
    with stage_write_lock(paths=paths, stage="geometry", blocked_by=blocked_by):
        block_df, block_group_df = build_block_crosswalk(
            paths=paths,
            config=config,
            out_dir=block_out_path.parent,
            force_rebuild=force_rebuild,
        )
        block_out_path.parent.mkdir(parents=True, exist_ok=True)
        block_group_out_path.parent.mkdir(parents=True, exist_ok=True)
        block_df.to_parquet(block_out_path, index=False)
        block_group_df.to_parquet(block_group_out_path, index=False)
        # Stamped with the same dependency set and membership parameters that
        # geometry_artifacts_are_current reads back.
        dependencies = geometry_dependency_paths(
            paths,
            config=config,
            geometry_dir=block_out_path.parent,
        )
        parameters = _membership_parameters(_load_territory_moves(paths, config))
        write_dependency_stamp(block_out_path, dependencies, parameters=parameters)
        write_dependency_stamp(block_group_out_path, dependencies, parameters=parameters)
        return block_out_path, block_group_out_path


# Static geometry (TIGER blocks and places, the reference jurisdiction master, the reviewed
# footprint registries) is separated from reporting-dependent membership: which municipal
# municipal territories a service relationship moves (service_relationships). The first is a file
# dependency of every cache; the second is recorded as a parameter, the move set itself, so a
# change to the relationships rebuilds only the states whose moves changed.


def _membership_parameters(territory_moves: frozenset[str]) -> dict[str, object]:
    ids = sorted(str(value) for value in territory_moves)
    return {
        "territory_move_count": len(ids),
        "territory_move_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def _state_membership_parameters(
    territory_moves: frozenset[str], *, state_fips: str
) -> dict[str, object]:
    prefix = f"{str(state_fips).zfill(2)}:"
    return {
        "territory_moves": sorted(
            str(value) for value in territory_moves if str(value).startswith(prefix)
        )
    }


def _geometry_common_dependency_paths(paths: RepoPaths) -> list[Path]:
    return [
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        paths.repo_root / "configs" / "consolidated_agency_footprints.csv",
        *pa_police_service_dependency_paths(paths),
        Path(__file__),
    ]


def _geometry_state_dependency_paths(paths: RepoPaths, *, state_fips: str) -> list[Path]:
    state = str(state_fips).zfill(2)
    dependencies = [
        *_geometry_common_dependency_paths(paths),
        paths.data_dir / "tiger_tabblock20" / f"tl_2020_{state}_tabblock20.zip",
        paths.data_dir / "tiger_places" / f"tl_2020_{state}_place.zip",
        paths.data_dir / "tiger_cousub" / f"tl_2020_{state}_cousub.zip",
    ]
    if state == "42":
        dependencies.append(pa_mcd_baf_path(paths))
    return dependencies


def _geometry_state_cache_paths(
    paths: RepoPaths,
    *,
    config: GeometryBuildConfig = GeometryBuildConfig(),
    geometry_dir: Path | None = None,
) -> list[tuple[str, Path]]:
    state_dir = (geometry_dir or (paths.state_dir / "geometry")) / "blocks_by_state"
    return [
        (str(state_fips).zfill(2), state_dir / f"{str(state_fips).zfill(2)}.parquet")
        for state_fips in _state_scope(paths, config)
    ]


def geometry_state_caches_are_current(
    paths: RepoPaths,
    *,
    config: GeometryBuildConfig = GeometryBuildConfig(),
    geometry_dir: Path | None = None,
    territory_moves: frozenset[str] | None = None,
) -> bool:
    moves = (
        _load_territory_moves(paths, config)
        if territory_moves is None
        else territory_moves
    )
    for state_fips, state_cache_path in _geometry_state_cache_paths(
        paths,
        config=config,
        geometry_dir=geometry_dir,
    ):
        if not artifact_is_current(
            state_cache_path,
            _geometry_state_dependency_paths(paths, state_fips=state_fips),
            parameters=_state_membership_parameters(moves, state_fips=state_fips),
        ):
            return False
    return True


def geometry_dependency_paths(
    paths: RepoPaths,
    *,
    config: GeometryBuildConfig = GeometryBuildConfig(),
    geometry_dir: Path | None = None,
) -> list[Path]:
    dependencies: list[Path] = list(_geometry_common_dependency_paths(paths))
    for state_fips, state_cache_path in _geometry_state_cache_paths(
        paths,
        config=config,
        geometry_dir=geometry_dir,
    ):
        dependencies.append(state_cache_path)
        dependencies.extend(_geometry_state_dependency_paths(paths, state_fips=state_fips))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in dependencies:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def geometry_artifacts_are_current(
    paths: RepoPaths,
    *,
    block_out_path: Path,
    block_group_out_path: Path,
    config: GeometryBuildConfig = GeometryBuildConfig(),
) -> bool:
    geometry_dir = block_out_path.parent
    moves = _load_territory_moves(paths, config)
    if not geometry_state_caches_are_current(
        paths,
        config=config,
        geometry_dir=geometry_dir,
        territory_moves=moves,
    ):
        return False
    dependencies = geometry_dependency_paths(paths, config=config, geometry_dir=geometry_dir)
    parameters = _membership_parameters(moves)
    return artifact_is_current(block_out_path, dependencies, parameters=parameters) and artifact_is_current(
        block_group_out_path,
        dependencies,
        parameters=parameters,
    )
