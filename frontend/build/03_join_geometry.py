"""
03 - Join the slim index table onto the block-group geometry and write GeoJSON.

- Loads the FULL-RESOLUTION 2020 TIGER/Line block-group shapefiles (per state).
- Filters to the CONUS states present in the index data (drops AK/HI/territories).
- Joins the published aggregate + per-offense indexes by 12-digit BG GEOID.
- Reprojects to WGS84 (EPSG:4326) for web tiling.
- Emits a newline-delimited GeoJSON (GeoJSONSeq) that tippecanoe consumes in 04.

Geometry source — why TIGER, not the cb_*_500k cartographic file:
The 1:500k cartographic boundary file is generalized (vertices decimated for
small-scale display), so block-group edges look JAGGED when you zoom to
neighborhood level. The full-resolution 2020 TIGER/Line files carry dense
vertices, so curved boundaries (rivers, coastlines, diagonal streets) stay smooth
at z11-z12.

The 2020 TIGER vintage matches the published index GEOIDs (the index uses 2020
census block groups), including Connecticut — whose index uses legacy 2020
county-based GEOIDs, not the 2022 planning-region GEOIDs that the cb_2023 file
carries. Sourcing every state from 2020 TIGER gives one source, one vintage, and
a clean GEOID match everywhere.

Tradeoff handled downstream: TIGER block-group polygons are NOT shoreline-clipped
(they extend to the legal boundary, which runs out into rivers/harbors). The
viewer (public/index.html) draws a water layer ABOVE the choropleth to mask any
edges that bleed into water. See that file's water-overlay note.

Schema note: per-offense fields are `index_{o}_primary` (default point),
`index_{o}_resident`, `rate_{o}_primary`, `expected_count_{o}`,
`crime_density_{o}` (at the precision 01 declares in index_stats.json `_meta`,
NOT the 2dp of the fractional counts), `estimate_mode_{o}`,
`reliability_tier_{o}`. Aggregates are
the six explicit published indexes plus `crime_density_total`; the frontend
default total layer is `index_total_primary_event_weighted` (exposure). NaN
survives the join as JSON null so suppressed / non-residential per-capita cells
(and water-only density cells) render as "no data". `special_use_tract_flag` and
the per-offense estimate-mode codes drive the distinct "special area" styling.

Two display-only fields ride along, both declared and computed by 01:
`provenance_class_{o}` (the 2-bit disclosure code the viewer hatches — see 01's
PROVENANCE_* block) and `population_density_{year}` (the input to the viewer's
national-zoom population-aware emphasis). Neither changes a published value.

Murder and rape publish at census-tract support, so their four map layers are
drawn from a SECOND set of tiles built from tract geometry. This step therefore
also joins the slim tract index table (`tract_indices_slim.csv`) onto the 2020
TIGER/Line tract shapefiles and writes `tracts_with_indices.geojsonl`, using the
SAME short tile keys as the block-group tiles so the viewer's paint expressions
work against either source. On the block-group side the parent tract's murder/rape
index + rate are carried as baked `ipt_*` / `rpt_*` keys for the popup.

Run:  uv run python frontend/build/03_join_geometry.py
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TIGER_DIR = REPO / "data/tiger_bg"
INDICES = REPO / "frontend/tmp/bg_indices_slim.csv"
OUT = REPO / "frontend/tmp/block_groups_with_indices.geojsonl"

# Tract lane (murder/rape publish at census-tract support): the slim tract index
# table joins onto the 2020 TIGER/Line tract shapefiles, one zip per state.
TRACT_TIGER_DIR = REPO / "data/tiger_tracts"
TRACT_INDICES = REPO / "frontend/tmp/tract_indices_slim.csv"
TRACT_OUT = REPO / "frontend/tmp/tracts_with_indices.geojsonl"
TRACT_TIGER_TPL = "tl_2020_{ss}_tract.zip"
TRACT_ID = "tract_id"
TRACT_ID_LEN = 11

# Snapshot data year: single point of control in snapshot_config.env (see 01).
# Drives the year-suffixed published population column name.
CONFIG = {
    k.strip(): v.strip()
    for k, v in (
        line.split("=", 1)
        for line in (Path(__file__).resolve().parent / "snapshot_config.env").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
}
YEAR = int(CONFIG["CRIMERISK_SNAPSHOT_YEAR"])
POPULATION_COL = f"population_{YEAR}"

# Density field precision. This step re-rounds the density columns after the join
# (they arrive already rounded from 01), so a precision change in 01 that is not
# mirrored here is silently undone before tiling — the coupling the stage-6 audit's
# R-01 change surface missed. Read it from the `_meta` block 01 writes rather than
# duplicating the literal, so the two cannot drift. Fails closed on an old stats
# file: no `_meta` means the upstream step did not declare a precision.
STATS_PATH = REPO / "frontend/tmp/index_stats.json"


def density_dp() -> int:
    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"{STATS_PATH} not found — run 01_extract_indices.py before 03."
        )
    meta = json.loads(STATS_PATH.read_text()).get("_meta")
    if not meta or "density_dp" not in meta:
        raise ValueError(
            f"{STATS_PATH.name} carries no _meta.density_dp; re-run "
            "01_extract_indices.py (it declares the density field precision)."
        )
    return int(meta["density_dp"])

# Geography key: 12-char zero-padded block-group GEOID.
GEO_ID = "block_group_geoid"
GEO_ID_LEN = 12

# Full-resolution 2020 TIGER/Line block-group files are per-state: tl_2020_{ss}_bg.zip.
TIGER_TPL = "tl_2020_{ss}_bg.zip"

# Water clip (the WATER fix, source side). TIGER tracts extend to their legal
# boundary, which for coastal tracts runs far out into oceans, harbors, and tidal
# estuaries (the Hudson and East River around Manhattan are the worst case). We
# erase the open water from water-heavy tracts using Natural Earth's 10m ocean
# polygon (which includes those tidal estuaries), so coastal tracts stop bleeding
# into the water. Guards keep this from eating real land:
#   - Only tracts whose AWATER fraction exceeds WATER_FRAC_MIN are clipped; pure
#     land tracts (incl. tiny harbor islands like Governors I.) are untouched.
#   - The ocean polygon is buffered INWARD by OCEAN_BUFFER_DEG (~150 m) before
#     erasing, so the clip never nibbles near-shore land where NE's generalized
#     coastline sits slightly inland of the true TIGER shoreline.
#   - A clip that would empty a tract or remove most of it is discarded (keep
#     the original) — we never drop a tract to the water mask.
# The viewer (public/index.html) still draws a water layer on top for the narrow
# inland rivers NE doesn't carry and as a final visual mask.
NE_OCEAN = REPO / "frontend/tmp/ne/ne_10m_ocean.shp"
WATER_FRAC_MIN = 0.30
OCEAN_BUFFER_DEG = 0.0014  # ~150 m at mid-latitudes
CLIP_KEEP_MIN = 0.15  # discard a clip that leaves <15% of the original area

# State FIPS -> USPS abbreviation. TIGER tract files (unlike the cb_* cartographic
# file) carry no STUSPS column, so we derive the popup `st` label from FIPS.
FIPS_TO_USPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}

OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]

AGG_INDEX_COLS = [
    "index_total_part1_resident",
    "index_personal_part1_resident",
    "index_property_part1_resident",
    "index_total_primary_event_weighted",
    "index_total_equal_offense",
    "index_total_harm",
]
OFFENSE_INDEX_PRIMARY = [f"index_{o}_primary" for o in OFFENSES]
OFFENSE_INDEX_RESIDENT = [f"index_{o}_resident" for o in OFFENSES]
OFFENSE_RATE_PRIMARY = [f"rate_{o}_primary" for o in OFFENSES]
OFFENSE_EXPECTED_COUNT = [f"expected_count_{o}" for o in OFFENSES]
OFFENSE_DENSITY = [f"crime_density_{o}" for o in OFFENSES]
DENSITY_TOTAL = ["crime_density_total"]
OFFENSE_ESTIMATE_MODE = [f"estimate_mode_{o}" for o in OFFENSES]
OFFENSE_RELIABILITY = [f"reliability_tier_{o}" for o in OFFENSES]
OFFENSE_SUPPORT = [f"effective_numerator_support_{o}" for o in OFFENSES]
# Per-offense provenance disclosure code (0 none / 1 benchmark-imputed / 2
# model-only outlier / 3 both). Declared and computed by 01 — see its PROVENANCE_*
# block for what each class means and why the marked set is these two and not the
# reliability tier. This step only carries them into the tiles, fail-closed on any
# code it does not recognise.
OFFENSE_PROVENANCE = [f"provenance_class_{o}" for o in OFFENSES]
PROVENANCE_CODES = {0, 1, 2, 3}
# Residents per square mile — the viewer's national-zoom emphasis input.
POP_DENSITY_COL = f"population_density_{YEAR}"

INDEX_COLS = AGG_INDEX_COLS + OFFENSE_INDEX_PRIMARY + OFFENSE_INDEX_RESIDENT
RATE_COLS = OFFENSE_RATE_PRIMARY
COUNT_COLS = OFFENSE_EXPECTED_COUNT
DENSITY_COLS = OFFENSE_DENSITY + DENSITY_TOTAL
# Estimate mode is carried as a compact integer code per offense (see EMODE_CODE),
# never the raw string, to keep the per-feature tile payload tiny.
EMODE_CODE = {
    "count_derived": 0,
    "non_residential": 1,
    "special_use": 2,
    "vehicle_denominator_invalid": 3,
    "insufficient_exposure": 4,
}
# Reliability tier carried as a compact integer code per offense (never the raw
# string) to keep the per-feature tile payload tiny. The frontend decodes via the
# RELIABILITY labels in the snapshot. `low` (the count-noise / model-only tier) is
# de-emphasized in the choropleth; high/medium render at full strength.
RELIABILITY_CODE = {
    "high": 2,
    "medium": 1,
    "low": 0,
}
# Baked tract popup values on the block-group rows: the parent tract's murder/rape
# primary index + rate, so the block-group popup can show a tract-scale value for
# these two tract-support offenses. Distinct short keys (ipt_/rpt_) so they never
# collide with the block group's own (null) murder/rape keys.
BG_TRACT_BAKE_SHORT = {
    "index_murder_primary_tract": "ipt_mur",
    "index_rape_primary_tract": "ipt_rap",
    "rate_murder_primary_tract": "rpt_mur",
    "rate_rape_primary_tract": "rpt_rap",
}
BG_TRACT_BAKE_COLS = list(BG_TRACT_BAKE_SHORT)

ATTR_COLS = (
    [POPULATION_COL, POP_DENSITY_COL, "special_use_tract_flag"]
    + INDEX_COLS
    + RATE_COLS
    + COUNT_COLS
    + DENSITY_COLS
    + OFFENSE_RELIABILITY
    + OFFENSE_SUPPORT
    + OFFENSE_ESTIMATE_MODE
    + OFFENSE_PROVENANCE
    + BG_TRACT_BAKE_COLS
)

# Tract-support offenses (murder, rape). The slim tract frame carries the same
# field families as the block-group frame, restricted to these two offenses, and
# is rendered with the SAME short tile keys (below) so the viewer's paint
# expressions work against either source.
RARE_OFFENSES = ["murder", "rape"]
TRACT_INDEX_PRIMARY = [f"index_{o}_primary" for o in RARE_OFFENSES]
TRACT_INDEX_RESIDENT = [f"index_{o}_resident" for o in RARE_OFFENSES]
TRACT_RATE_PRIMARY = [f"rate_{o}_primary" for o in RARE_OFFENSES]
TRACT_EXPECTED_COUNT = [f"expected_count_{o}" for o in RARE_OFFENSES]
TRACT_RELIABILITY = [f"reliability_tier_{o}" for o in RARE_OFFENSES]
TRACT_SUPPORT = [f"effective_numerator_support_{o}" for o in RARE_OFFENSES]
TRACT_ESTIMATE_MODE = [f"estimate_mode_{o}" for o in RARE_OFFENSES]
TRACT_PROVENANCE = [f"provenance_class_{o}" for o in RARE_OFFENSES]
TRACT_INDEX_COLS = TRACT_INDEX_PRIMARY + TRACT_INDEX_RESIDENT
TRACT_RATE_COLS = TRACT_RATE_PRIMARY
TRACT_COUNT_COLS = TRACT_EXPECTED_COUNT
TRACT_ATTR_COLS = (
    [POPULATION_COL, POP_DENSITY_COL, "special_use_tract_flag"]
    + TRACT_INDEX_COLS
    + TRACT_RATE_COLS
    + TRACT_COUNT_COLS
    + TRACT_RELIABILITY
    + TRACT_SUPPORT
    + TRACT_ESTIMATE_MODE
    + TRACT_PROVENANCE
)

# Short per-offense suffix used in the compact tile keys.
SUFFIX = {
    "murder": "mur",
    "rape": "rap",
    "robbery": "rob",
    "aggravated_assault": "agg",
    "burglary": "bur",
    "larceny": "lar",
    "motor_vehicle_theft": "mvt",
}

# Long published field -> short tile key. The frontend KEY_MAP mirrors this.
SHORT = {
    POPULATION_COL: "pop",
    POP_DENSITY_COL: "pd",
    "special_use_tract_flag": "su",
    "index_total_part1_resident": "i_tot",
    "index_personal_part1_resident": "i_per",
    "index_property_part1_resident": "i_pro",
    "index_total_primary_event_weighted": "i_evw",
    "index_total_equal_offense": "i_eq",
    "index_total_harm": "i_harm",
    "crime_density_total": "d_tot",
}
for o in OFFENSES:
    s = SUFFIX[o]
    SHORT[f"index_{o}_primary"] = f"ip_{s}"
    SHORT[f"index_{o}_resident"] = f"ir_{s}"
    SHORT[f"rate_{o}_primary"] = f"r_{s}"
    SHORT[f"expected_count_{o}"] = f"ec_{s}"
    SHORT[f"crime_density_{o}"] = f"d_{s}"
    SHORT[f"estimate_mode_{o}"] = f"em_{s}"
    SHORT[f"reliability_tier_{o}"] = f"rt_{s}"
    SHORT[f"effective_numerator_support_{o}"] = f"es_{s}"
    SHORT[f"provenance_class_{o}"] = f"pv_{s}"
# Baked tract popup values (block-group side). The tract tiles reuse the plain
# ip_*/r_* keys above, so these distinct ipt_/rpt_ keys are block-group only.
SHORT.update(BG_TRACT_BAKE_SHORT)


def provenance_codes(series: pd.Series, col: str) -> pd.Series:
    """Coerce a provenance disclosure column to int16, fail-closed on bad codes.

    01 owns the class definitions; this only guards the carry into the tiles. A
    row that matched no index (NaN after the join) is 0 — no class marked, which
    is the same thing the viewer shows for a cell it has no code for.
    """
    s = pd.to_numeric(series, errors="coerce")
    unknown = sorted(set(s.dropna().astype(int).unique()) - PROVENANCE_CODES)
    if unknown:
        raise ValueError(
            f"Unknown provenance codes in {col}: {unknown} "
            f"(expected {sorted(PROVENANCE_CODES)} — re-run 01_extract_indices.py)"
        )
    return s.fillna(0).astype("int16")


def clip_water(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Erase open water from coastal block groups that bleed into oceans/harbors.

    Only cells with AWATER fraction > WATER_FRAC_MIN are touched; the ocean is
    buffered inward first; clips that would gut a cell are discarded. Pure-land
    cells (incl. harbor islands) pass through untouched. See NE_OCEAN note.
    """
    if not NE_OCEAN.exists():
        print(f"  [water clip] {NE_OCEAN.name} missing — skipping (run 02 to fetch).")
        return gdf

    # NE ocean is a single WORLD-scale polygon; buffering it whole takes ~70s.
    # Crop its GEOMETRY (not just the feature filter) to the data bbox first, so
    # buffer/simplify and every per-tract difference run on the regional coastline
    # only — sub-second instead of minutes.
    from shapely.geometry import box

    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 1.0  # degrees of slack around the data
    bbox = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
    ocean = gpd.read_file(NE_OCEAN).to_crs(4326).union_all().intersection(bbox)
    # Inward buffer (never nibble shore) + light simplify: the coarse coastline
    # only needs accuracy to the buffer tolerance, and fewer vertices is cheaper.
    ocean_in = ocean.buffer(-OCEAN_BUFFER_DEG).simplify(OCEAN_BUFFER_DEG / 2)

    aland = gdf["ALAND"].astype("float64")
    awater = gdf["AWATER"].astype("float64")
    wat_frac = awater / (aland + awater).replace(0, np.nan)
    cand_mask = (wat_frac > WATER_FRAC_MIN).to_numpy()

    gdf = gdf.copy()
    cand = gdf.loc[cand_mask]
    if len(cand) == 0:
        print("  [water clip] no water-heavy block groups found.")
        return gdf

    # Vectorized difference on the candidate subset only (shapely2 batched op).
    orig_area = cand.geometry.area
    clipped = cand.geometry.difference(ocean_in).make_valid()

    # Guard: discard a clip that empties or guts a cell — keep the original.
    keep_clip = (~clipped.is_empty) & (clipped.area >= CLIP_KEEP_MIN * orig_area)
    new_geoms = clipped.where(keep_clip, cand.geometry)

    gdf.loc[cand.index, "geometry"] = new_geoms
    print(
        f"  [water clip] {len(cand):,} water-heavy block groups; "
        f"clipped {int(keep_clip.sum()):,}, kept-as-is {int((~keep_clip).sum()):,} (guard)."
    )
    return gdf


def join_tracts() -> dict:
    """Join the slim tract index table onto tract geometry and write GeoJSONSeq.

    Murder and rape publish at census-tract support, so their four map layers are
    drawn from tract geometry. The attribute schema reuses the SAME short tile keys
    as the block-group tiles (ip_*/ir_*/r_*/ec_*/rt_*/es_*/em_* plus pop/su), so the
    viewer's paint expressions work against either source. Returns the tract feature
    counts folded into the join manifest.
    """
    print(f"Reading tract indices: {TRACT_INDICES.name}")
    idx = pd.read_csv(TRACT_INDICES, dtype={TRACT_ID: str, "state_fips": str})
    idx[TRACT_ID] = idx[TRACT_ID].str.zfill(TRACT_ID_LEN)
    states = set(idx["state_fips"].str.zfill(2))
    print(f"  {len(idx):,} tracts across {len(states)} states")

    # Full-resolution 2020 TIGER tract geometry per state (same vintage + reasoning
    # as the block-group files: 2020 GEOIDs match the published index everywhere,
    # including Connecticut's legacy county-based GEOIDs).
    print(f"Reading full-resolution TIGER tract geometry for {len(states)} states ...")
    parts = []
    for ss in sorted(states):
        path = TRACT_TIGER_DIR / TRACT_TIGER_TPL.format(ss=ss)
        if not path.exists():
            raise FileNotFoundError(f"Missing TIGER tract file for state {ss}: {path}")
        g = gpd.read_file(f"zip://{path}")
        g["GEOID"] = g["GEOID"].astype(str).str.zfill(TRACT_ID_LEN)
        g["STATEFP"] = g["STATEFP"].astype(str).str.zfill(2)
        parts.append(g[["GEOID", "STATEFP", "NAMELSAD", "ALAND", "AWATER", "geometry"]])
    gdf = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs
    )
    gdf["STUSPS"] = gdf["STATEFP"].map(FIPS_TO_USPS)
    print(f"  {len(gdf):,} TIGER tract polygons (CRS {gdf.crs})")

    # Reproject to WGS84 and clip water-heavy coastal polygons (same fix as BG; the
    # clip is geography-agnostic).
    gdf = gdf.to_crs(4326)
    gdf = clip_water(gdf)

    merged = gdf.merge(
        idx[[TRACT_ID] + TRACT_ATTR_COLS],
        left_on="GEOID",
        right_on=TRACT_ID,
        how="left",
    )

    # No index match if the merge key came back null (drop — nothing to color).
    geom_only = merged[TRACT_ID].isna()
    n_geom_only = int(geom_only.sum())
    idx_only = set(idx[TRACT_ID]) - set(gdf["GEOID"])
    print(f"  geometry tracts with NO index match: {n_geom_only:,} (dropped)")
    print(f"  index tracts with NO geometry match: {len(idx_only):,}")
    merged = merged[~geom_only].copy()

    # Keep only what the tiles need: GEOID + name + attributes.
    keep = ["GEOID", "STUSPS", "NAMELSAD"] + TRACT_ATTR_COLS + ["geometry"]
    merged = merged[keep].rename(
        columns={"GEOID": TRACT_ID, "STUSPS": "st", "NAMELSAD": "name"}
    )

    # Same attribute encodings as the block-group frame.
    for c in TRACT_INDEX_COLS + TRACT_RATE_COLS:
        merged[c] = merged[c].astype("float64").round(1)
    for c in TRACT_COUNT_COLS + TRACT_SUPPORT:
        merged[c] = merged[c].astype("float64").round(2)
    for c in TRACT_RELIABILITY:
        merged[c] = merged[c].map(RELIABILITY_CODE).fillna(0).astype("int16")
    merged[POPULATION_COL] = (
        merged[POPULATION_COL].astype("Float64").round().astype("Int64")
    )
    for c in TRACT_ESTIMATE_MODE:
        modes = set(merged[c].dropna().astype(str).unique())
        unknown = sorted(modes - set(EMODE_CODE))
        if unknown:
            raise ValueError(f"Unknown estimate modes in {c}: {unknown}")
        missing = int(merged[c].isna().sum())
        if missing:
            raise ValueError(f"Missing estimate modes after join in {c}: {missing:,}")
        merged[c] = merged[c].map(EMODE_CODE).astype("int16")
    merged["special_use_tract_flag"] = (
        merged["special_use_tract_flag"].fillna(0).astype("int16")
    )
    for c in TRACT_PROVENANCE:
        merged[c] = provenance_codes(merged[c], c)
    merged[POP_DENSITY_COL] = merged[POP_DENSITY_COL].astype("float64").round(2)

    # Reuse the SAME short tile keys as the block-group tiles.
    merged = merged.rename(columns=SHORT)

    print(f"Writing tract GeoJSONSeq -> {TRACT_OUT.name}")
    TRACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(TRACT_OUT, driver="GeoJSONSeq")
    print(f"  wrote {len(merged):,} tract features  ({TRACT_OUT.stat().st_size/1e6:.1f} MB)")

    return {
        "n_tract_features": int(len(merged)),
        "n_tract_index_features": int(len(idx)),
        "n_tract_geometry_dropped_no_index": n_geom_only,
        "n_tract_index_no_geometry": len(idx_only),
    }


def main() -> None:
    print(f"Reading indices: {INDICES.name}")
    idx = pd.read_csv(INDICES, dtype={GEO_ID: str, "state_fips": str})
    idx[GEO_ID] = idx[GEO_ID].str.zfill(GEO_ID_LEN)
    states = set(idx["state_fips"].str.zfill(2))
    print(f"  {len(idx):,} block groups across {len(states)} states")

    # Read full-resolution TIGER geometry per state and concatenate. TIGER files
    # carry no two-letter state abbrev (STUSPS), so we derive `st` from FIPS.
    print(f"Reading full-resolution TIGER geometry for {len(states)} states ...")
    parts = []
    for ss in sorted(states):
        path = TIGER_DIR / TIGER_TPL.format(ss=ss)
        if not path.exists():
            raise FileNotFoundError(f"Missing TIGER file for state {ss}: {path}")
        g = gpd.read_file(f"zip://{path}")
        g["GEOID"] = g["GEOID"].astype(str).str.zfill(GEO_ID_LEN)
        g["STATEFP"] = g["STATEFP"].astype(str).str.zfill(2)
        # TIGER GEOIDs carry small water-only / nonresidential block groups too;
        # keep them all here and let the index join decide what renders.
        # ALAND/AWATER drive the water-clip guard below.
        parts.append(g[["GEOID", "STATEFP", "NAMELSAD", "ALAND", "AWATER", "geometry"]])
    gdf = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs
    )
    # FIPS -> USPS abbreviation for the tile `st` field (popup shows it).
    gdf["STUSPS"] = gdf["STATEFP"].map(FIPS_TO_USPS)
    print(f"  {len(gdf):,} TIGER block-group polygons (CRS {gdf.crs})")

    # Reproject to WGS84 for web.
    gdf = gdf.to_crs(4326)

    # Clip water-heavy coastal block groups to land (the WATER fix). See NE_OCEAN note.
    gdf = clip_water(gdf)

    # Join indices onto geometry.
    merged = gdf.merge(
        idx[[GEO_ID] + ATTR_COLS],
        left_on="GEOID",
        right_on=GEO_ID,
        how="left",
    )

    geom_only = (
        merged[POPULATION_COL].isna() & merged["index_total_part1_resident"].isna()
    )
    n_geom_only = int(geom_only.sum())
    idx_only = set(idx[GEO_ID]) - set(gdf["GEOID"])
    print(f"  geometry block groups with NO index match: {n_geom_only:,} (dropped)")
    print(f"  index block groups with NO geometry match: {len(idx_only):,}")

    # Drop polygons that matched no index row — nothing to color.
    merged = merged[~geom_only].copy()

    # Keep only what the tiles need: GEOID + name + attributes.
    keep = ["GEOID", "STUSPS", "NAMELSAD"] + ATTR_COLS + ["geometry"]
    merged = merged[keep].rename(
        columns={"GEOID": GEO_ID, "STUSPS": "st", "NAMELSAD": "name"}
    )

    # Coerce attribute dtypes: round indices/rates, ints for pop, 2dp for the
    # fractional expected counts, and the declared density precision for the
    # density columns (see density_dp() — 2dp here would undo 01's precision
    # before the tiles are built). Keep NaN (-> JSON null) so the frontend can
    # render suppressed / non-residential per-capita cells, and water-only
    # density cells, as "no data".
    dens_dp = density_dp()
    print(f"  density precision from index_stats.json _meta: {dens_dp}dp")
    for c in INDEX_COLS + RATE_COLS + BG_TRACT_BAKE_COLS:
        merged[c] = merged[c].astype("float64").round(1)
    for c in COUNT_COLS + OFFENSE_SUPPORT:
        merged[c] = merged[c].astype("float64").round(2)
    for c in DENSITY_COLS:
        merged[c] = merged[c].astype("float64").round(dens_dp)
    # Reliability tier string -> compact int code (high=2/medium=1/low=0). Unmatched
    # / missing default to 0 (low / least-confident) so the display never over-states
    # confidence for a cell we couldn't tier.
    for c in OFFENSE_RELIABILITY:
        merged[c] = merged[c].map(RELIABILITY_CODE).fillna(0).astype("int16")
    merged[POPULATION_COL] = (
        merged[POPULATION_COL].astype("Float64").round().astype("Int64")
    )
    # Encode each per-offense estimate mode as a compact integer code. Fail closed
    # on unknown/missing values so a backend schema change cannot be silently
    # rendered as a normal count-derived estimate.
    for c in OFFENSE_ESTIMATE_MODE:
        modes = set(merged[c].dropna().astype(str).unique())
        unknown = sorted(modes - set(EMODE_CODE))
        if unknown:
            raise ValueError(f"Unknown estimate modes in {c}: {unknown}")
        missing = int(merged[c].isna().sum())
        if missing:
            raise ValueError(f"Missing estimate modes after join in {c}: {missing:,}")
        merged[c] = merged[c].map(EMODE_CODE).astype("int16")
    # special_use rollup flag -> 0/1 int (already int8 from 01; coerce defensively).
    merged["special_use_tract_flag"] = (
        merged["special_use_tract_flag"].fillna(0).astype("int16")
    )
    # Provenance disclosure codes: already integers from 01, but a cell that fell
    # out of the geometry join carries NaN, and an unknown code would silently
    # paint a hatch (or fail to). Fail closed on anything outside {0,1,2,3}.
    for c in OFFENSE_PROVENANCE:
        merged[c] = provenance_codes(merged[c], c)
    # Residents per square mile (national-zoom emphasis). Two decimals is what 01
    # emits; re-rounding here keeps the tile payload from carrying float noise the
    # CSV round-trip can introduce.
    merged[POP_DENSITY_COL] = merged[POP_DENSITY_COL].astype("float64").round(2)

    # Shorten attribute keys to keep per-feature tile payload tiny (lets us tile
    # to z12 without dropping block groups). The frontend KEY_MAP mirrors this exactly.
    merged = merged.rename(columns=SHORT)
    (REPO / "frontend/tmp/key_map.json").write_text(json.dumps(SHORT, indent=2))

    print(f"Writing GeoJSONSeq -> {OUT.name}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # GeoJSONSeq (newline-delimited features) streams cleanly into tippecanoe.
    merged.to_file(OUT, driver="GeoJSONSeq")
    print(f"  wrote {len(merged):,} features  ({OUT.stat().st_size/1e6:.1f} MB)")

    # Tract lane: murder/rape publish at census-tract support, drawn from tract
    # geometry with the same short tile keys.
    tract_counts = join_tracts()

    # Persist a tiny manifest of what made it in (counts + matched block groups),
    # including the tract-lane counts.
    manifest = {
        "n_features": int(len(merged)),
        "n_index_features": int(len(idx)),
        "n_geometry_dropped_no_index": n_geom_only,
        "n_index_no_geometry": len(idx_only),
        "states": sorted(states),
        # The precision the TILED attributes actually carry. 05 refuses to stamp a
        # snapshot whose declared density_floor does not match this, so a snapshot
        # can never claim a precision the tiles were not built at.
        "density_dp": dens_dp,
        # How many features actually carry a provenance mark on at least one
        # offense, on each lane. 05 stamps the same figure from 01's own coverage
        # block and refuses a mismatch, so a snapshot can never describe a hatch
        # the tiles do not carry.
        "n_provenance_marked_features": int(
            (merged[[SHORT[c] for c in OFFENSE_PROVENANCE]] > 0).any(axis=1).sum()
        ),
        **tract_counts,
    }
    (REPO / "frontend/tmp/join_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("  wrote join_manifest.json")


if __name__ == "__main__":
    main()
