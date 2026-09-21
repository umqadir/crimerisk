"""02 - Join the public surfaces onto 2020 TIGER geometry and emit GeoJSONSeq.

Writes three newline-delimited GeoJSON files to the tiles work directory, one
per geography, each carrying exactly the 25 tile attributes declared in
crschema.TILE_FIELDS:

  work/geo_county.geojsonl       3,108 counties       (z3-4 rollup layer)
  work/geo_tract.geojsonl       83,776 tracts         (z5-12)
  work/geo_bg.geojsonl         238,193 block groups   (z8-12)

Geometry is full-resolution TIGER/Line 2020 (matching the published GEOID
vintage, including Connecticut's legacy county-based tract ids). Coastal cells
whose legal boundary runs out into open water are clipped against an inward
buffered Natural Earth ocean polygon so the coastline reads correctly without a
separate water overlay.

Also writes the display-name tables the lookup shards need.

Run:  uv run python frontend/build/02_geometry.py [county|tract|bg]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crschema import (  # noqa: E402
    DATA_ROOT,
    FIPS_TO_USPS,
    TILE_FIELDS,
    WORK,
)

BG_TIGER = DATA_ROOT / "tiger_bg"
TRACT_TIGER = DATA_ROOT / "tiger_tracts"
COUNTY_TIGER = DATA_ROOT / "tiger_counties" / "tl_2020_us_county.zip"
NE_OCEAN = WORK / "ne" / "ne_10m_ocean.shp"

WATER_FRAC_MIN = 0.30
OCEAN_BUFFER_DEG = 0.0014  # ~150 m
CLIP_KEEP_MIN = 0.15

_OCEAN: shapely.geometry.base.BaseGeometry | None = None


def ocean() -> shapely.geometry.base.BaseGeometry | None:
    """Inward-buffered CONUS ocean polygon, built once and cached on disk."""
    global _OCEAN
    if _OCEAN is not None:
        return _OCEAN
    cache = WORK / "ocean_conus.wkb"
    if cache.exists():
        _OCEAN = shapely.from_wkb(cache.read_bytes())
        return _OCEAN
    if not NE_OCEAN.exists():
        print("  [water] Natural Earth ocean missing - skipping coastal clip")
        return None
    bbox = shapely.box(-128.0, 22.0, -64.0, 51.0)
    geom = gpd.read_file(NE_OCEAN).to_crs(4326).union_all().intersection(bbox)
    geom = geom.buffer(-OCEAN_BUFFER_DEG).simplify(OCEAN_BUFFER_DEG / 2)
    cache.write_bytes(shapely.to_wkb(geom))
    _OCEAN = geom
    return _OCEAN


def clip_water(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    oc = ocean()
    if oc is None:
        return gdf, 0
    aland = gdf["ALAND"].astype("float64")
    awater = gdf["AWATER"].astype("float64")
    frac = awater / (aland + awater).replace(0, np.nan)
    cand = gdf.loc[(frac > WATER_FRAC_MIN).fillna(False)]
    if len(cand) == 0:
        return gdf, 0
    orig = cand.geometry.area
    clipped = cand.geometry.difference(oc).make_valid()
    keep = (~clipped.is_empty) & (clipped.area >= CLIP_KEEP_MIN * orig)
    gdf = gdf.copy()
    gdf.loc[cand.index, "geometry"] = clipped.where(keep, cand.geometry)
    return gdf, int(keep.sum())


def write_geojsonseq(gdf: gpd.GeoDataFrame, out: Path, fields: list[str]) -> int:
    """Stream features, dropping null attributes so tiles stay lean."""
    geoms = shapely.to_geojson(gdf.geometry.values)
    cols = {f: gdf[f].to_numpy(dtype=object) for f in fields if f in gdf.columns}
    n = 0
    with out.open("a") as fh:
        buf = []
        for i in range(len(gdf)):
            props = {}
            for f, arr in cols.items():
                v = arr[i]
                if v is None or v is pd.NA:
                    continue
                if isinstance(v, float) and (v != v):
                    continue
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                if v == 0 and f in ("su", "src"):
                    continue
                props[f] = v
            buf.append(
                '{"type":"Feature","properties":'
                + json.dumps(props, separators=(",", ":"))
                + ',"geometry":'
                + geoms[i]
                + "}\n"
            )
            n += 1
            if len(buf) >= 2000:
                fh.write("".join(buf))
                buf = []
        if buf:
            fh.write("".join(buf))
    return n


def tiger_layer(path: Path, id_len: int) -> gpd.GeoDataFrame:
    g = gpd.read_file(f"zip://{path}")
    g["GEOID"] = g["GEOID"].astype(str).str.zfill(id_len)
    keep = ["GEOID", "NAMELSAD", "ALAND", "AWATER", "geometry"]
    g = g[keep].to_crs(4326)
    return g


def build_layer(
    core: pd.DataFrame,
    id_col: str,
    tiger_dir: Path | None,
    tpl: str | None,
    id_len: int,
    out: Path,
    names_out: Path,
    national_zip: Path | None = None,
) -> dict:
    if out.exists():
        out.unlink()
    states = sorted(core["state_fips"].unique()) if national_zip is None else ["us"]
    total = 0
    clipped = 0
    names: list[pd.DataFrame] = []
    missing_geom = 0
    for ss in states:
        if national_zip is not None:
            g = tiger_layer(national_zip, id_len)
            g = g[g["GEOID"].str[:2].isin(sorted(core["state_fips"].unique()))]
        else:
            path = tiger_dir / tpl.format(ss=ss)
            if not path.exists():
                raise FileNotFoundError(path)
            g = tiger_layer(path, id_len)
        g, nclip = clip_water(g)
        clipped += nclip
        part = core[core["state_fips"] == ss] if national_zip is None else core
        m = g.merge(part, left_on="GEOID", right_on=id_col, how="inner")
        missing_geom += len(part) - len(m)
        names.append(
            pd.DataFrame(
                {"geoid": m["GEOID"].to_numpy(), "name": m["NAMELSAD"].to_numpy()}
            )
        )
        m = m.rename(columns={"GEOID": "geoid"})
        total += write_geojsonseq(gpd.GeoDataFrame(m, geometry="geometry"), out, TILE_FIELDS)
        if national_zip is None:
            print(f"    {FIPS_TO_USPS.get(ss, ss)} {len(m):,}", end="", flush=True)
    print()
    pd.concat(names, ignore_index=True).to_parquet(names_out, index=False)
    return {
        "features": total,
        "coastal_clipped": clipped,
        "rows_without_geometry": int(missing_geom),
        "bytes": out.stat().st_size,
    }


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    manifest = WORK / "geometry_manifest.json"
    stats = json.loads(manifest.read_text()) if manifest.exists() else {}

    if target not in ("all", "county"):
        pass
    else:
        build_counties(stats)
    if target in ("all", "tract"):
        build_tracts(stats)
    if target in ("all", "bg"):
        build_blockgroups(stats)

    manifest.write_text(json.dumps(stats, indent=2))


def build_counties(stats: dict) -> None:
    print("Counties (z3-7 rollup layer)")
    county = pd.read_parquet(WORK / "county_core.parquet")
    county["state_fips"] = county["county_geoid"].str[:2]
    stats["county"] = build_layer(
        county,
        "county_geoid",
        None,
        None,
        5,
        WORK / "geo_county.geojsonl",
        WORK / "names_county.parquet",
        national_zip=COUNTY_TIGER,
    )
    print(f"  {stats['county']}")


def build_tracts(stats: dict) -> None:
    print("Tracts (z5-12)")
    tr = pd.read_parquet(WORK / "tract_core.parquet")
    tr = tr.rename(columns={"tract_id": "geoid_src"})
    tr["geoid_src"] = tr["geoid_src"].astype(str)
    stats["tract"] = build_layer(
        tr,
        "geoid_src",
        TRACT_TIGER,
        "tl_2020_{ss}_tract.zip",
        11,
        WORK / "geo_tract.geojsonl",
        WORK / "names_tract.parquet",
    )
    print(f"  {stats['tract']}")


def build_blockgroups(stats: dict) -> None:
    print("Block groups (z8-12)")
    bg = pd.read_parquet(WORK / "bg_core.parquet")
    bg = bg.rename(columns={"block_group_geoid": "geoid_src"})
    bg["geoid_src"] = bg["geoid_src"].astype(str)
    stats["blockgroup"] = build_layer(
        bg,
        "geoid_src",
        BG_TIGER,
        "tl_2020_{ss}_bg.zip",
        12,
        WORK / "geo_bg.geojsonl",
        WORK / "names_bg.parquet",
    )
    print(f"  {stats['blockgroup']}")


if __name__ == "__main__":
    main()
