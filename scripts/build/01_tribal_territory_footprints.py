"""Tribal custom footprints over the whole reservation / trust-land territory (proposal builder).

The 2025.2 tribal footprints (archived builder
archive/2026-09-cull/scripts/review/overlap/build_tribal_aiannh_footprints.py) weighted blocks by
2020 residents and dropped zero-weight block groups, so a reservation or trust-land block group
with no residents (the Wind Creek casinos of Poarch Creek PD ALDI00300) was not in the footprint.

Rule, per ORI already in configs/overlap_custom_footprints.csv on a Census tribal area
(geometry_source_type census_2020_block_assignment_file_aiannh or census_2020_tiger_aitsn,
release states only):

| Item | Definition |
|---|---|
| Census area | union of the mapped 2020 AIANNH polygons (AIANNHCE x COMPTYP) or the AITSN subdivision |
| BIA LAR | features of the BIA AIAN National LAR (data/bia_lar/) that overlap the Census area and have half their area inside it, cover half of it, or carry its name |
| Territory blocks | census: representative point in the Census area; lar_majority: unpopulated block at least half inside the LAR; lar_facility: block less than half inside the LAR holding a destination point that lies on LAR land. Populated blocks at least half inside the LAR and outside the Census area (reservation exterior land such as Pendleton in Umatilla's LAR) are left out and listed |
| Service agreements | a facility another agency serves under a recorded agreement (configs/tribal_facility_service_agreements.csv, evidence cited per row) leaves the territory: the ORI's blocks in that block group drop out and fall to the block group's own police; such a part may hold no residents |
| Footprint rows | every block group with at least one territory block; none is left out |
| Territory part | the BG's territory blocks plus the LAR land on its other blocks, less any left-out populated block |
| territory_share_<leg> | the leg's value on the territory part over the BG's; blank where the BG has none of the leg |
| Legs (located inputs) | residents (2020 POP20), housing_units (HOUSING20) and resident workers (RAC) on census and lar_majority blocks; daytime_jobs (residents + WAC - RAC, floored at residents); wac_jobs, retail_jobs (CNS07), industrial_jobs (CNS05+06+08) on every territory block; destination_poi (Overture 2026-09-23.1 points) and landscan_day (LandScan USA 2021 day, cell area fraction) on the territory part |
| territory_share_land | 2020 land area on the territory blocks over the BG's (block count where the BG has no land) |
| Allocation | offense prior x the offense's combined territory share (src/crimerisk/territory_shares.py); a BG whose territory part has none of the offense's legs gets weight 0 for that offense and stays a row |
| weight_share | territory_share_land normalized over the footprint (zero-signal fallback only) |
| weight_share_basis | territory_prior |
| bg_population_coverage_share | displacing ORIs: share of BG residents inside the union of displacing territories, on the census and lar_majority blocks (0 allowed); otherwise blank |

Checks (the build fails on a miss): the Census part of the territory reproduces the governed
rows (residents shares, union coverage, the resident-coverage sidecar); block-level LODES sums
equal data/LODES/parsed/lodes_wac_block_groups.parquet.

Outputs, under --out-dir:
    tribal_footprint_proposal_lar.csv                   proposed replacement units
    tribal_footprint_proposal_lar_evidence.csv          one row per ORI x block group (old and new, every leg)
    tribal_footprint_lar_mapping.csv                    every LAR feature overlapping a Census area, mapped or not
    tribal_footprint_lar_excluded_populated_blocks.csv  populated LAR-only blocks left out, per ORI x BG
    tribal_footprint_service_agreement_exclusions.csv   block groups left out under a service agreement
    tribal_footprint_lar_facility_blocks.csv            lar_facility blocks with their LAR fraction and jobs
    overlap_custom_footprints.proposed_lar.csv          the whole configs/overlap_custom_footprints.csv after the swap

    uv run python scripts/build/01_tribal_territory_footprints.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.windows import Window, from_bounds
import shapely

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.required_inputs import RELEASE_STATE_FIPS  # noqa: E402
from crimerisk.territory_shares import (  # noqa: E402
    TERRITORY_LAND_SHARE_COLUMN,
    TERRITORY_LEGS,
    TERRITORY_SHARE_COLUMNS,
)

FOOTPRINTS = REPO_ROOT / "configs" / "overlap_custom_footprints.csv"
OVERRIDES = REPO_ROOT / "configs" / "overlap_footprint_overrides.csv"
RESIDENT_COVERAGE = REPO_ROOT / "configs" / "overlap_custom_footprint_resident_coverage.csv"
SERVICE_AGREEMENTS = REPO_ROOT / "configs" / "tribal_facility_service_agreements.csv"
MAPPING = REPO_ROOT / "archive" / "2026-09-cull" / "configs" / "tribal_agency_aiannh_footprints.csv"
AIANNH = REPO_ROOT / "data" / "tiger_aiannh" / "tl_2020_us_aiannh.zip"
AITSN = REPO_ROOT / "data" / "tiger_aitsn" / "tl_2020_us_aitsn.zip"
TABBLOCK = REPO_ROOT / "data" / "tiger_tabblock20" / "tl_2020_{fips}_tabblock20.zip"
TIGER_BG = REPO_ROOT / "data" / "tiger_bg" / "tl_2020_{fips}_bg.zip"
LANDSCAN_DAY = REPO_ROOT / "data" / "LandScan-USA" / "rasters" / "landscan-usa-2021-conus-day.tif"
LODES_RAW = REPO_ROOT / "data" / "LODES" / "raw" / "wac_2023"
LODES_PARSED = REPO_ROOT / "data" / "LODES" / "parsed" / "lodes_wac_block_groups.parquet"
OVERTURE_POINTS = REPO_ROOT / "data" / "Overture-Places" / "parsed" / "overture_places_state_points_latest.parquet"

TRIBAL_GEOMETRY_TYPES = ("census_2020_block_assignment_file_aiannh", "census_2020_tiger_aitsn")
BASIS = "territory_prior"
LANDSCAN_REF = "https://doi.org/10.48690/1527701"
AIANNH_REF = "https://www2.census.gov/geo/tiger/TIGER2020/AIANNH/tl_2020_us_aiannh.zip"
AITSN_REF = "https://www2.census.gov/geo/tiger/TIGER2020/AITSN/tl_2020_us_aitsn.zip"
TABBLOCK_REF = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
LODES_REF = "https://lehd.ces.census.gov/data/lodes/LODES8/"
OVERTURE_REF = "Overture places 2026-09-23.1 destination points"
BIA_LAR = REPO_ROOT / "data" / "bia_lar" / "bia_aian_national_lar.geojson"
BIA_LAR_REF = (
    "https://biamaps.geoplatform.gov/server/rest/services/DivLTR/BIA_AIAN_National_LAR/FeatureServer/0"
)
# Words that do not identify a land area when matching a LAR name to the Census area names.
NAME_STOPWORDS = frozenset(
    {"reservation", "trust", "land", "lands", "of", "the", "and", "band", "tribe", "tribes", "indian",
     "rancheria", "community", "nation", "colony", "off", "pueblo", "oregon", "ks", "ne", "chapter"}
)
LAR_MAJORITY = 0.5
FOOTPRINT_COLUMNS = (
    "ori",
    "state_fips",
    "block_group_geoid",
    "weight_share",
    "bg_population_coverage_share",
    "weight_share_basis",
    "geometry_source_type",
    "geometry_source_ref",
    "footprint_note",
    *TERRITORY_SHARE_COLUMNS,
)
# Block attributes summed per territory part and per block group.
BLOCK_VALUES = ("pop", "housing", "aland", "wac", "rac", "retail", "industrial")
RESIDENT_VALUES = ("pop", "housing", "rac")
STATE_ABBR = {
    "01": "al", "04": "az", "06": "ca", "08": "co", "09": "ct", "12": "fl", "16": "id", "20": "ks",
    "22": "la", "23": "me", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "30": "mt", "31": "ne",
    "32": "nv", "35": "nm", "36": "ny", "37": "nc", "38": "nd", "40": "ok", "41": "or", "44": "ri",
    "46": "sd", "48": "tx", "49": "ut", "53": "wa", "55": "wi", "56": "wy",
}


class LandScanDay:
    """Area-weighted sums of the LandScan USA day raster over polygons (EPSG:4326)."""

    def __init__(self, path: Path) -> None:
        self.src = rasterio.open(path)

    def total(self, geom) -> float:
        if geom is None or geom.is_empty:
            return 0.0
        minx, miny, maxx, maxy = geom.bounds
        win = from_bounds(minx, miny, maxx, maxy, self.src.transform).round_offsets().round_lengths()
        win = Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2)
        values = self.src.read(1, window=win, boundless=True, fill_value=0).astype(float)
        values[values < 0] = 0.0
        rows, cols = np.nonzero(values > 0)
        if rows.size == 0:
            return 0.0
        t = self.src.window_transform(win)
        x0 = t.c + cols * t.a
        y0 = t.f + rows * t.e
        boxes = shapely.box(x0, y0 + t.e, x0 + t.a, y0)
        shapely.prepare(geom)
        fraction = np.zeros(rows.size)
        inside = shapely.contains(geom, boxes)
        fraction[inside] = 1.0
        edge = ~inside & shapely.intersects(geom, boxes)
        if edge.any():
            fraction[edge] = shapely.area(shapely.intersection(boxes[edge], geom)) / shapely.area(
                boxes[edge]
            )
        return float((values[rows, cols] * fraction).sum())


class DestinationPoints:
    """Overture destination points (the points behind `destination_poi_total`)."""

    def __init__(self, path: Path) -> None:
        points = pd.read_parquet(path, columns=["lon", "lat"])
        self.x = points["lon"].to_numpy(dtype=float)
        self.y = points["lat"].to_numpy(dtype=float)

    def count(self, geom) -> int:
        if geom is None or geom.is_empty:
            return 0
        minx, miny, maxx, maxy = geom.bounds
        near = (self.x >= minx) & (self.x <= maxx) & (self.y >= miny) & (self.y <= maxy)
        if not near.any():
            return 0
        shapely.prepare(geom)
        return int(shapely.contains_xy(geom, self.x[near], self.y[near]).sum())

    def inside(self, geom) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the points inside `geom`."""
        if geom is None or geom.is_empty:
            return np.empty(0), np.empty(0)
        minx, miny, maxx, maxy = geom.bounds
        near = (self.x >= minx) & (self.x <= maxx) & (self.y >= miny) & (self.y <= maxy)
        x, y = self.x[near], self.y[near]
        shapely.prepare(geom)
        hit = shapely.contains_xy(geom, x, y)
        return x[hit], y[hit]


def _name_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", str(text).lower()) if t not in NAME_STOPWORDS}


class LandAreaRepresentations:
    """BIA AIAN National LAR polygons, mapped to an ORI through its Census territory.

    A LAR feature belongs to the ORI's territory when it overlaps the Census area and either at
    least half of the LAR lies inside the Census area, at least half of the Census area lies
    inside the LAR, or every identifying word of the LAR name appears in the Census area names.
    """

    def __init__(self, path: Path) -> None:
        frame = gpd.read_file(path).to_crs(4269)
        frame["geometry"] = frame.geometry.make_valid()
        self.frame = frame.reset_index(drop=True)
        self.equal_area = self.frame.to_crs(5070)
        self.index = self.equal_area.sindex

    def mapped(self, ori: str, census, names: str) -> tuple[object | None, list[dict[str, object]]]:
        census_ea = gpd.GeoSeries([census], crs=4269).to_crs(5070).iloc[0]
        census_area = float(census_ea.area)
        decisions: list[dict[str, object]] = []
        keep: list[int] = []
        for i in self.index.query(census_ea, predicate="intersects"):
            feature = self.equal_area.iloc[int(i)]
            lar_area = float(feature.geometry.area)
            overlap = float(feature.geometry.intersection(census_ea).area)
            if overlap <= 0.0:
                continue
            tokens = _name_tokens(feature["LARNAME"])
            inside, covered = overlap / lar_area, overlap / census_area
            named = bool(tokens) and tokens <= _name_tokens(names)
            mapped = inside >= LAR_MAJORITY or covered >= LAR_MAJORITY or named
            decisions.append(
                {
                    "ori": ori, "larid": feature["LARID"], "larname": feature["LARNAME"],
                    "lar_km2": lar_area / 1e6, "overlap_km2": overlap / 1e6,
                    "share_of_lar_in_census_area": inside, "share_of_census_area_in_lar": covered,
                    "name_match": named, "mapped": mapped,
                    "added_outside_census_km2": (lar_area - overlap) / 1e6 if mapped else 0.0,
                }
            )
            if mapped:
                keep.append(int(i))
        if not keep:
            return None, decisions
        return shapely.union_all(self.frame.geometry.iloc[keep].values), decisions


def _territory_polygons(ori: str, mapping_row: pd.Series) -> tuple[object, str, str]:
    refs = f"{TABBLOCK_REF} | {LANDSCAN_REF} | {LODES_REF} | {OVERTURE_REF}"
    subdivision = str(mapping_row["tribal_subdivision_geoid"]).strip()
    if subdivision:
        frame = gpd.read_file(f"zip://{AITSN}")
        frame = frame[frame["GEOID"].astype(str).eq(subdivision)]
        if len(frame) != 1:
            raise ValueError(f"AITSN GEOID {subdivision}: {len(frame)} polygons")
        label = f"{frame['NAMELSAD'].iloc[0]} (AITSN GEOID {subdivision})"
        return frame.to_crs(4269).union_all(), label, f"{AITSN_REF} | {refs}"
    codes = [c.strip() for c in str(mapping_row["aiannh_codes"]).split(";") if c.strip()]
    comptyps = [c.strip() for c in str(mapping_row["comptyp_included"]).split(";") if c.strip()]
    frame = _aiannh()
    frame = frame[frame["AIANNHCE"].isin(codes) & frame["COMPTYP"].isin(comptyps)]
    if frame.empty:
        raise ValueError(f"{ori}: no AIANNH polygons for {codes} x {comptyps}")
    label = (
        f"{mapping_row['aiannh_names']} (AIANNHCE {mapping_row['aiannh_codes']}, "
        f"COMPTYP {mapping_row['comptyp_included']})"
    )
    return frame.to_crs(4269).union_all(), label, f"{AIANNH_REF} | {refs}"


_AIANNH_CACHE: list[gpd.GeoDataFrame] = []


def _aiannh() -> gpd.GeoDataFrame:
    if not _AIANNH_CACHE:
        _AIANNH_CACHE.append(gpd.read_file(f"zip://{AIANNH}"))
    return _AIANNH_CACHE[0]


def _lodes_blocks(fips: str, year: int) -> pd.DataFrame:
    abbr = STATE_ABBR[fips]
    wac = pd.read_csv(
        LODES_RAW / f"{abbr}_wac_S000_JT00_{year}.csv.gz",
        dtype={"w_geocode": str},
        usecols=["w_geocode", "C000", "CNS05", "CNS06", "CNS07", "CNS08"],
    ).rename(columns={"w_geocode": "GEOID20"})
    wac = pd.DataFrame(
        {
            "GEOID20": wac["GEOID20"],
            "wac": wac["C000"],
            "retail": wac["CNS07"],
            "industrial": wac["CNS05"] + wac["CNS06"] + wac["CNS08"],
        }
    )
    rac = pd.read_csv(
        LODES_RAW / f"{abbr}_rac_S000_JT00_{year}.csv.gz",
        dtype={"h_geocode": str},
        usecols=["h_geocode", "C000"],
    ).rename(columns={"h_geocode": "GEOID20", "C000": "rac"})
    return wac.merge(rac, on="GEOID20", how="outer").fillna(0.0)


class StateBlocks:
    facility_min_lar_fraction = 0.0

    def __init__(self, fips: str, lodes_year: int) -> None:
        path = f"zip://{str(TABBLOCK).format(fips=fips)}"
        self.path = path
        self.crs = pyogrio.read_info(path)["crs"]
        attrs = pyogrio.read_dataframe(
            path, columns=["GEOID20", "POP20", "HOUSING20", "ALAND20"], read_geometry=False
        ).rename(columns={"POP20": "pop", "HOUSING20": "housing", "ALAND20": "aland"})
        self.lodes = _lodes_blocks(fips, lodes_year)
        attrs = attrs.merge(self.lodes, on="GEOID20", how="left").fillna(
            {"wac": 0.0, "rac": 0.0, "retail": 0.0, "industrial": 0.0}
        )
        unmatched = set(self.lodes["GEOID20"]) - set(attrs["GEOID20"])
        if unmatched:
            raise ValueError(f"state {fips}: {len(unmatched)} LODES blocks are not 2020 blocks")
        attrs["bg"] = attrs["GEOID20"].astype(str).str.slice(0, 12)
        self.attrs = attrs.set_index("GEOID20")
        self.bg_totals = attrs.groupby("bg")[list(BLOCK_VALUES)].sum()
        self.bg_totals["blocks"] = attrs.groupby("bg").size()
        self.bg_polygons = (
            gpd.read_file(f"zip://{str(TIGER_BG).format(fips=fips)}", columns=["GEOID"])
            .to_crs(4326)
            .set_index("GEOID")
            .geometry
        )

    def territory_blocks(
        self, territory, lar=None, points: "DestinationPoints | None" = None
    ) -> gpd.GeoDataFrame:
        """Blocks of the territory, with `membership`:

        census        representative point inside the Census area (AIANNH/AITSN)
        lar_majority  at least half of the block's area inside the mapped LAR
        lar_facility  a destination point inside the mapped LAR lies on the block
        """
        extent = territory if lar is None else shapely.union_all([territory, lar])
        bounds = gpd.GeoSeries([extent], crs=4269).to_crs(self.crs).total_bounds
        blocks = gpd.read_file(self.path, bbox=tuple(bounds), columns=["GEOID20"])
        poly = gpd.GeoSeries([territory], crs=4269).to_crs(blocks.crs).iloc[0]
        shapely.prepare(poly)
        census = shapely.contains(poly, blocks.geometry.representative_point().values)
        membership = np.where(census, "census", "").astype(object)
        if lar is not None and len(blocks):
            lar_ea = gpd.GeoSeries([lar], crs=4269).to_crs(5070).iloc[0]
            block_ea = blocks.geometry.to_crs(5070).values
            candidate = ~census & shapely.intersects(lar_ea, block_ea)
            fraction = np.zeros(len(blocks))
            if candidate.any():
                fraction[candidate] = shapely.area(
                    shapely.intersection(block_ea[candidate], lar_ea)
                ) / np.maximum(shapely.area(block_ea[candidate]), 1e-9)
            # Only unpopulated LAR land joins by area: the Census area already tabulates the
            # reservation's residents, and populated LAR-only blocks are exterior-boundary land the
            # Census area leaves out (Pendleton inside Umatilla's LAR, for example).
            unpopulated = (
                blocks["GEOID20"].map(self.attrs["pop"]).fillna(0.0).to_numpy(dtype=float) <= 0.0
            )
            majority = candidate & (fraction >= LAR_MAJORITY) & unpopulated
            populated_majority = candidate & (fraction >= LAR_MAJORITY) & ~unpopulated
            membership[majority] = "lar_majority"
            if points is not None:
                x, y = points.inside(lar)
                if x.size:
                    located = gpd.GeoDataFrame(
                        geometry=gpd.points_from_xy(x, y), crs=4326
                    ).to_crs(blocks.crs)
                    hits = gpd.sjoin(located, blocks[["geometry"]], predicate="within")
                    facility = np.zeros(len(blocks), dtype=bool)
                    facility[blocks.index.get_indexer(hits["index_right"].unique())] = True
                    # The parcel-edge case: a facility on LAR land whose block is mostly outside.
                    partial = (fraction < LAR_MAJORITY) & (fraction >= self.facility_min_lar_fraction)
                    membership[candidate & partial & facility] = "lar_facility"
            membership[populated_majority & (membership == "")] = "excluded_populated_lar"
        out = blocks.loc[membership != ""].copy()
        out["membership"] = membership[membership != ""]
        out["lar_fraction"] = (fraction if lar is not None and len(blocks) else np.zeros(len(blocks)))[
            membership != ""
        ]
        out = out.join(self.attrs, on="GEOID20")
        return out.to_crs(4326)


def _daytime_jobs(pop, wac, rac):
    """The published daytime-jobs proxy: residents + WAC - RAC, at least 0 and at least residents."""
    return np.maximum(np.clip(pop + wac - rac, 0.0, None), pop)


def _share(inside, whole):
    inside = np.asarray(inside, dtype=float)
    whole = np.asarray(whole, dtype=float)
    return np.where(whole > 0.0, np.minimum(inside / np.where(whole > 0.0, whole, 1.0), 1.0), np.nan)


def build(out_dir: Path) -> dict[str, int]:
    current = pd.read_csv(FOOTPRINTS, dtype="string")
    current["state_fips"] = current["state_fips"].str.zfill(2)
    current["block_group_geoid"] = current["block_group_geoid"].str.zfill(12)
    tribal = current[current["geometry_source_type"].isin(TRIBAL_GEOMETRY_TYPES)]
    tribal = tribal[tribal["state_fips"].isin(RELEASE_STATE_FIPS)]
    oris = sorted(set(tribal["ori"]))
    mapping_all = pd.read_csv(MAPPING, dtype="string").fillna("")
    mapping_all["state_fips"] = mapping_all["state_fips"].str.zfill(2)
    mapping_all = mapping_all.set_index("ori")
    mapping = mapping_all.loc[mapping_all.index.isin(oris)]
    missing = sorted(set(oris) - set(mapping.index))
    if missing:
        raise ValueError(f"tribal footprints with no mapping row: {missing}")
    overrides = pd.read_csv(OVERRIDES, dtype="string")
    displacing = set(
        overrides.loc[
            overrides["displaces_county_remainder"].fillna("").str.upper().eq("TRUE"), "ori"
        ]
    )
    parsed = pd.read_parquet(LODES_PARSED)
    lodes_year = parsed.groupby("state_fips")["lodes_year"].first().astype(int).to_dict()

    landscan = LandScanDay(LANDSCAN_DAY)
    points = DestinationPoints(OVERTURE_POINTS)
    states: dict[str, StateBlocks] = {}

    def state_blocks(fips: str) -> StateBlocks:
        if fips not in states:
            state = StateBlocks(fips, lodes_year[fips])
            check = parsed[parsed["state_fips"].eq(fips)].set_index("bg_id")
            totals = state.bg_totals.reindex(check.index).fillna(0.0)
            for column, source in (("wac", "jobs_wac"), ("rac", "jobs_rac"), ("retail", "CNS07")):
                gap = float((totals[column] - check[source]).abs().max())
                if gap > 0.0:
                    raise ValueError(f"state {fips}: block LODES {column} differs from the parsed table by {gap}")
            states[fips] = state
        return states[fips]

    agreements = pd.read_csv(SERVICE_AGREEMENTS, dtype="string")
    agreements["block_group_geoid"] = agreements["block_group_geoid"].str.zfill(12)
    agreement_rows: list[pd.DataFrame] = []
    facility_rows: list[pd.DataFrame] = []
    lars = LandAreaRepresentations(BIA_LAR)
    lar_decisions: list[dict[str, object]] = []
    excluded_rows: list[pd.DataFrame] = []
    evidence: list[pd.DataFrame] = []
    blocks_by_ori: dict[str, pd.DataFrame] = {}
    meta: dict[str, tuple[str, str, str, str]] = {}
    for ori in oris:
        row = mapping.loc[ori]
        fips = str(row["state_fips"])
        state = state_blocks(fips)
        territory, label, ref = _territory_polygons(ori, row)
        names = f"{row['aiannh_names']} {row.get('tribal_subdivision_name', '')}"
        lar, decisions = lars.mapped(ori, territory, names)
        lar_decisions.extend(decisions)
        blocks = state.territory_blocks(territory, lar, points)
        excluded = blocks[blocks["membership"].eq("excluded_populated_lar")]
        excluded_rows.append(
            excluded.groupby("bg").agg(blocks=("GEOID20", "size"), residents=("pop", "sum"))
            .reset_index().assign(ori=ori)
        )
        blocks = blocks[~blocks["membership"].eq("excluded_populated_lar")]
        # A facility another agency serves under a recorded agreement leaves the territory: the
        # ORI's blocks in the agreement's block group fall to that block group's own police.
        served = agreements[agreements["ori"].eq(ori)]
        if not served.empty:
            out_blocks = blocks[blocks["bg"].isin(set(served["block_group_geoid"]))]
            if float(out_blocks["pop"].sum()) > 0.0:
                raise ValueError(f"{ori}: a service-agreement exclusion would remove residents")
            missing_bg = set(served["block_group_geoid"]) - set(out_blocks["bg"])
            if missing_bg:
                raise ValueError(f"{ori}: service agreement names block groups off its territory: {sorted(missing_bg)}")
            agreement_rows.append(
                out_blocks.groupby("bg").size().rename("blocks").reset_index()
                .rename(columns={"bg": "block_group_geoid"})
                .merge(served, on="block_group_geoid", how="left")
            )
            blocks = blocks[~blocks["bg"].isin(set(served["block_group_geoid"]))]
        if blocks.empty:
            raise ValueError(f"{ori}: territory has no 2020 blocks in state {fips}")
        blocks_by_ori[ori] = blocks
        facility_rows.append(
            blocks.loc[blocks["membership"].eq("lar_facility"),
                       ["GEOID20", "bg", "lar_fraction", "pop", "wac", "retail"]]
            .assign(ori=ori)
        )
        mapped_names = "; ".join(
            f"{d['larname']} ({d['larid']})" for d in decisions if d["mapped"]
        )
        meta[ori] = (label, f"{ref} | {BIA_LAR_REF}" if lar is not None else ref,
                     str(row["agency_name_std"]), mapped_names)
        inside = blocks.groupby("bg")[list(BLOCK_VALUES)].sum()
        # Residents stay with the Census area (a facility block's residents are not the
        # agency's): residents, housing units and resident workers count on census and
        # unpopulated LAR blocks only.
        resident_blocks = blocks[~blocks["membership"].eq("lar_facility")]
        for value in RESIDENT_VALUES:
            inside[value] = resident_blocks.groupby("bg")[value].sum().reindex(inside.index).fillna(0.0)
        inside["blocks"] = blocks.groupby("bg").size()
        for kind in ("census", "lar_majority", "lar_facility"):
            members = blocks[blocks["membership"].eq(kind)]
            inside[f"blocks_{kind}"] = members.groupby("bg").size()
        inside["pop_census"] = blocks[blocks["membership"].eq("census")].groupby("bg")["pop"].sum()
        inside = inside.fillna(
            {"blocks_census": 0, "blocks_lar_majority": 0, "blocks_lar_facility": 0, "pop_census": 0.0}
        )
        by_bg = inside.add_suffix("_in").join(state.bg_totals.add_suffix("_bg"), how="left")
        day_in, day_bg, poi_in, poi_bg = [], [], [], []
        for bg, bg_row in by_bg.iterrows():
            polygon = state.bg_polygons.get(bg)
            whole_day = landscan.total(polygon)
            whole_poi = points.count(polygon)
            # The territory part: its blocks, plus any mapped LAR on the block group's other blocks.
            part = shapely.union_all(blocks.loc[blocks["bg"].eq(bg)].geometry.values)
            if lar is not None:
                extra = shapely.intersection(lar, polygon)
                left_out = excluded.loc[excluded["bg"].eq(bg)]
                if len(left_out):
                    extra = shapely.difference(extra, shapely.union_all(left_out.geometry.values))
                part = shapely.union_all([part, extra])
            if int(bg_row["blocks_in"]) == int(bg_row["blocks_bg"]):
                part_day, part_poi = whole_day, whole_poi
            else:
                part_day, part_poi = landscan.total(part), points.count(part)
            day_in.append(part_day)
            day_bg.append(whole_day)
            poi_in.append(min(part_poi, whole_poi))
            poi_bg.append(whole_poi)
        by_bg["day_in"], by_bg["day_bg"] = day_in, day_bg
        by_bg["poi_in"], by_bg["poi_bg"] = poi_in, poi_bg
        by_bg["jobs_in"] = _daytime_jobs(by_bg["pop_in"], by_bg["wac_in"], by_bg["rac_in"])
        by_bg["jobs_bg"] = _daytime_jobs(by_bg["pop_bg"], by_bg["wac_bg"], by_bg["rac_bg"])
        located = {
            "residents": "pop",
            "landscan_day": "day",
            "daytime_jobs": "jobs",
            "destination_poi": "poi",
            "retail_jobs": "retail",
            "industrial_jobs": "industrial",
            "wac_jobs": "wac",
            "housing_units": "housing",
        }
        for leg in TERRITORY_LEGS:
            value = located[leg]
            by_bg[f"territory_share_{leg}"] = _share(by_bg[f"{value}_in"], by_bg[f"{value}_bg"])
        land = _share(by_bg["aland_in"], by_bg["aland_bg"])
        blocks_share = by_bg["blocks_in"] / by_bg["blocks_bg"]
        by_bg[TERRITORY_LAND_SHARE_COLUMN] = np.where(
            by_bg["aland_bg"].gt(0) & by_bg["aland_in"].gt(0), land, blocks_share
        )
        by_bg = by_bg.reset_index().rename(columns={"bg": "block_group_geoid"})
        by_bg.insert(0, "ori", ori)
        by_bg.insert(1, "state_fips", fips)
        evidence.append(by_bg)
    evidence_frame = pd.concat(evidence, ignore_index=True)

    # Union residents coverage over the displacing territories, as the 2025.2 builder did. A
    # displacing service-wide footprint (Navajo Nation, AZ0018900) enters with its whole mapped
    # Census area in every state it covers; the subdivision it leaves out (Ramah) is itself
    # displacing, so the union is the same. The Census-only union is the reproduction check.
    union_frames = [b[["GEOID20", "bg", "pop", "membership"]] for o, b in blocks_by_ori.items() if o in displacing]
    service = current[
        current["weight_share_basis"].eq("service_area_prior") & current["ori"].isin(displacing)
    ]
    for ori, fips in service[["ori", "state_fips"]].drop_duplicates().itertuples(index=False):
        territory, _, _ = _territory_polygons(ori, mapping_all.loc[ori])
        union_frames.append(
            state_blocks(fips).territory_blocks(territory)[["GEOID20", "bg", "pop", "membership"]]
        )
    union_all = pd.concat(union_frames, ignore_index=True)
    union_all = union_all[~union_all["membership"].eq("lar_facility")]
    union_census = union_all[union_all["membership"].eq("census")].drop_duplicates("GEOID20")
    union_all = union_all.drop_duplicates("GEOID20")
    evidence_frame["displaces_county_remainder"] = evidence_frame["ori"].isin(displacing)
    pop_bg = evidence_frame["pop_bg"]

    def coverage(union: pd.DataFrame) -> np.ndarray:
        covered = evidence_frame["block_group_geoid"].map(union.groupby("bg")["pop"].sum()).fillna(0.0)
        return np.where(
            evidence_frame["displaces_county_remainder"],
            np.where(pop_bg.gt(0), np.minimum(1.0, covered / pop_bg.where(pop_bg.gt(0))), 0.0),
            np.nan,
        )

    evidence_frame["new_bg_population_coverage_share"] = coverage(union_all)
    evidence_frame["census_bg_population_coverage_share"] = coverage(union_census)
    evidence_frame["census_residents_share"] = _share(evidence_frame["pop_census_in"], pop_bg)
    land_totals = evidence_frame.groupby("ori")[TERRITORY_LAND_SHARE_COLUMN].sum()
    evidence_frame["new_weight_share"] = evidence_frame[TERRITORY_LAND_SHARE_COLUMN] / evidence_frame[
        "ori"
    ].map(land_totals)
    # The zero-everything parts: no residents, daytime population, POI or jobs on the territory.
    leg_columns = [f"territory_share_{leg}" for leg in TERRITORY_LEGS]
    evidence_frame["territory_part_empty"] = evidence_frame[leg_columns].fillna(0.0).le(0.0).all(axis=1)

    old = tribal[
        ["ori", "block_group_geoid", "weight_share", "bg_population_coverage_share",
         "weight_share_basis", "territory_share_residents"]
    ].rename(
        columns={
            "weight_share": "old_weight_share",
            "bg_population_coverage_share": "old_bg_population_coverage_share",
            "weight_share_basis": "old_weight_share_basis",
            "territory_share_residents": "old_territory_share_residents",
        }
    )
    evidence_frame = evidence_frame.merge(old, on=["ori", "block_group_geoid"], how="outer", indicator=True)
    evidence_frame["in_old_footprint"] = evidence_frame["_merge"].ne("left_only")
    agreed = pd.concat(agreement_rows, ignore_index=True) if agreement_rows else pd.DataFrame(
        columns=["ori", "block_group_geoid"]
    )
    agreed_keys = set(zip(agreed["ori"], agreed["block_group_geoid"]))
    lost = evidence_frame[
        evidence_frame["_merge"].eq("right_only")
        & ~pd.Series(
            list(zip(evidence_frame["ori"], evidence_frame["block_group_geoid"])),
            index=evidence_frame.index,
        ).isin(agreed_keys)
    ]
    if not lost.empty:
        raise ValueError(
            "governed footprint block groups outside the rebuilt territory: "
            + str(lost[["ori", "block_group_geoid"]].to_dict(orient="records"))
        )
    evidence_frame = evidence_frame[evidence_frame["_merge"].ne("right_only")].reset_index(drop=True)
    evidence_frame = evidence_frame.drop(columns="_merge")
    # Reproduction: the Census part of the rebuilt territory equals the governed 2025.3 rows.
    old_residents = pd.to_numeric(evidence_frame["old_territory_share_residents"], errors="coerce")
    census_residents = evidence_frame["census_residents_share"]
    evidence_frame["reproduces_census_residents"] = ~evidence_frame["in_old_footprint"] | (
        (old_residents.isna() & census_residents.fillna(0.0).le(0.0))
        | (old_residents - census_residents.fillna(0.0)).abs().le(1e-12)
    )
    old_coverage = pd.to_numeric(evidence_frame["old_bg_population_coverage_share"], errors="coerce")
    evidence_frame["reproduces_census_coverage"] = old_coverage.isna() | (
        (old_coverage - evidence_frame["census_bg_population_coverage_share"]).abs().le(1e-9)
    )
    sidecar_all = pd.read_csv(RESIDENT_COVERAGE, dtype="string", keep_default_na=False)
    sidecar_all["block_group_geoid"] = sidecar_all["block_group_geoid"].str.zfill(12)
    sidecar = sidecar_all[sidecar_all["ori"].isin(oris)].rename(
        columns={"bg_responsibility_population_coverage_share": "sidecar_residents_share"}
    )
    evidence_frame = evidence_frame.merge(
        sidecar[["ori", "block_group_geoid", "sidecar_residents_share"]],
        on=["ori", "block_group_geoid"],
        how="left",
        validate="one_to_one",
    )
    sidecar_share = pd.to_numeric(evidence_frame["sidecar_residents_share"], errors="coerce")
    evidence_frame["reproduces_sidecar"] = (
        sidecar_share.isna() & census_residents.fillna(0.0).le(0.0)
    ) | (sidecar_share - census_residents).abs().le(1e-12)
    checks = ["reproduces_census_residents", "reproduces_census_coverage", "reproduces_sidecar"]
    misses = evidence_frame[~evidence_frame[checks].all(axis=1)]
    if not misses.empty:
        raise ValueError(
            "the Census part of the rebuilt territory does not reproduce the governed rows: "
            + str(misses[["ori", "block_group_geoid", "pop_census_in", *checks]].head(20).to_dict(orient="records"))
        )

    # Proposed governed rows: weight shares rounded so each ORI sums to 1 exactly at 15 digits.
    proposal_rows: list[dict[str, str]] = []
    first_row = tribal.drop_duplicates("ori").set_index("ori")
    for ori, group in evidence_frame.groupby("ori", sort=True):
        group = group.sort_values("block_group_geoid", kind="mergesort").reset_index(drop=True)
        shares = group["new_weight_share"].to_numpy(dtype=float).copy()
        largest = int(np.argmax(shares))
        shares[largest] += 1.0 - float(shares.sum())
        label, ref, agency, mapped_lar = meta[ori]
        lar_text = f"BIA LAR {mapped_lar}" if mapped_lar else "no BIA LAR feature maps to it"
        note = (
            f"Tribal territory footprint for {agency}: {label} and {lar_text}; every 2020 block "
            "group with a territory block (representative point in the Census area, at least half "
            "its area in the LAR, or a destination point on LAR land), populated or not, none left "
            "out; territory_share_<leg> = the leg on the territory part (its blocks plus the LAR on "
            "the block group) over the block group's; allocation weight = offense prior x the "
            "offense's denominator-weighted territory share (0 where the territory part has none "
            "of the offense's legs); weight_share = land share, zero-signal fallback."
        )
        for i, bg_row in group.iterrows():
            bg_coverage = bg_row["new_bg_population_coverage_share"]
            record = {
                "ori": ori,
                "state_fips": str(bg_row["state_fips"]),
                "block_group_geoid": str(bg_row["block_group_geoid"]),
                "weight_share": f"{float(shares[i]):.15g}",
                "bg_population_coverage_share": "" if pd.isna(bg_coverage) else f"{float(bg_coverage):.15g}",
                "weight_share_basis": BASIS,
                "geometry_source_type": str(first_row.loc[ori, "geometry_source_type"]),
                "geometry_source_ref": ref,
                "footprint_note": note,
            }
            for column in TERRITORY_SHARE_COLUMNS:
                value = bg_row[column]
                record[column] = "" if pd.isna(value) else f"{float(value):.15g}"
            proposal_rows.append(record)
    proposal = pd.DataFrame(proposal_rows, columns=list(FOOTPRINT_COLUMNS))
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal.to_csv(out_dir / "tribal_footprint_proposal_lar.csv", index=False)
    pd.DataFrame(lar_decisions).to_csv(out_dir / "tribal_footprint_lar_mapping.csv", index=False)
    agreed.to_csv(out_dir / "tribal_footprint_service_agreement_exclusions.csv", index=False)
    pd.concat(facility_rows, ignore_index=True).rename(
        columns={"GEOID20": "block_geoid", "bg": "block_group_geoid", "pop": "residents",
                 "wac": "wac_jobs", "retail": "retail_jobs"}
    )[["ori", "block_geoid", "block_group_geoid", "lar_fraction", "residents", "wac_jobs", "retail_jobs"]].to_csv(
        out_dir / "tribal_footprint_lar_facility_blocks.csv", index=False
    )
    pd.concat(excluded_rows, ignore_index=True).rename(columns={"bg": "block_group_geoid"})[
        ["ori", "block_group_geoid", "blocks", "residents"]
    ].to_csv(out_dir / "tribal_footprint_lar_excluded_populated_blocks.csv", index=False)
    # The whole proposed file: every other row verbatim (new columns blank), each regenerated
    # ORI's rows where its old rows began.
    original = pd.read_csv(FOOTPRINTS, dtype="string", keep_default_na=False)
    for column in TERRITORY_SHARE_COLUMNS:
        if column not in original.columns:
            original[column] = ""
    pieces: list[pd.DataFrame] = []
    placed: set[str] = set()
    replaced = set(proposal["ori"])
    runs = original["ori"].ne(original["ori"].shift()).fillna(True).astype(int).cumsum()
    for _, block in original.groupby(runs, sort=False):
        name = str(block["ori"].iloc[0])
        if name in replaced:
            if name not in placed:
                pieces.append(proposal[proposal["ori"].eq(name)].astype("string"))
                placed.add(name)
            continue
        pieces.append(block[list(FOOTPRINT_COLUMNS)])
    pd.concat(pieces, ignore_index=True).to_csv(
        out_dir / "overlap_custom_footprints.proposed_lar.csv", index=False
    )
    values = [f"{v}_{side}" for v in (*BLOCK_VALUES, "blocks", "day", "poi", "jobs") for side in ("in", "bg")]
    evidence_columns = [
        "ori", "state_fips", "block_group_geoid", "displaces_county_remainder", "in_old_footprint",
        "territory_part_empty", "blocks_census_in", "blocks_lar_majority_in", "blocks_lar_facility_in",
        *values, "pop_census_in", *TERRITORY_SHARE_COLUMNS, "census_residents_share",
        "old_territory_share_residents", "old_weight_share", "new_weight_share",
        "old_bg_population_coverage_share", "census_bg_population_coverage_share",
        "new_bg_population_coverage_share", "sidecar_residents_share", *checks,
    ]
    evidence_frame.sort_values(["ori", "block_group_geoid"])[evidence_columns].to_csv(
        out_dir / "tribal_footprint_proposal_lar_evidence.csv", index=False
    )
    return {
        "oris": int(proposal["ori"].nunique()),
        "rows": int(len(proposal)),
        "old_rows": int(len(tribal)),
        "new_block_groups": int((~evidence_frame["in_old_footprint"]).sum()),
        "empty_territory_parts": int(evidence_frame["territory_part_empty"].sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--facility-min-lar-fraction",
        type=float,
        default=0.0,
        help="least share of a block's area on LAR land for it to join as a facility block (not adopted; default 0)",
    )
    args = parser.parse_args()
    StateBlocks.facility_min_lar_fraction = float(args.facility_min_lar_fraction)
    stats = build(args.out_dir)
    print(
        f"tribal territory footprints: {stats['oris']} ORIs, {stats['rows']} rows "
        f"(governed now {stats['old_rows']}); {stats['new_block_groups']} block groups added, "
        f"{stats['empty_territory_parts']} with nothing located on the territory part (kept)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
