from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import shapefile  # pyshp
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from crimerisk.utils.zipfiles import extract_zip_all


@dataclass(frozen=True)
class TigerPolygon:
    state_fips: str
    code: str
    geoid: str
    name: str
    namelsad: str
    geometry: BaseGeometry

    @property
    def area(self) -> float:
        return float(self.geometry.area)


def _read_tiger_polygons_from_shapefile(
    shp_path: Path,
    *,
    state_field: str,
    code_field: str,
    geoid_field: str,
    name_field: str,
    namelsad_field: str,
) -> list[TigerPolygon]:
    reader = shapefile.Reader(str(shp_path))
    field_names = [f[0] for f in reader.fields[1:]]

    def idx(field: str) -> int:
        return field_names.index(field)

    i_state = idx(state_field)
    i_code = idx(code_field)
    i_geoid = idx(geoid_field)
    i_name = idx(name_field)
    i_namelsad = idx(namelsad_field)

    polygons: list[TigerPolygon] = []
    for rec, shp in zip(reader.iterRecords(), reader.iterShapes(), strict=True):
        geom = shape(shp.__geo_interface__)
        polygons.append(
            TigerPolygon(
                state_fips=str(rec[i_state]).zfill(2),
                code=str(rec[i_code]).zfill(5),
                geoid=str(rec[i_geoid]),
                name=str(rec[i_name]),
                namelsad=str(rec[i_namelsad]),
                geometry=geom,
            )
        )
    return polygons


def load_tiger_place_polygons(*, tiger_place_zip: Path, cache_dir: Path) -> list[TigerPolygon]:
    out_dir = cache_dir / "tiger_places" / tiger_place_zip.stem
    extract_zip_all(tiger_place_zip, out_dir)

    shp_files = list(out_dir.glob("*.shp"))
    if len(shp_files) != 1:
        raise RuntimeError(f"Expected 1 .shp in {out_dir}, found {len(shp_files)}")

    return _read_tiger_polygons_from_shapefile(
        shp_files[0],
        state_field="STATEFP",
        code_field="PLACEFP",
        geoid_field="GEOID",
        name_field="NAME",
        namelsad_field="NAMELSAD",
    )


def load_tiger_cousub_polygons(*, tiger_cousub_zip: Path, cache_dir: Path) -> list[TigerPolygon]:
    out_dir = cache_dir / "tiger_cousub" / tiger_cousub_zip.stem
    extract_zip_all(tiger_cousub_zip, out_dir)

    shp_files = list(out_dir.glob("*.shp"))
    if len(shp_files) != 1:
        raise RuntimeError(f"Expected 1 .shp in {out_dir}, found {len(shp_files)}")

    return _read_tiger_polygons_from_shapefile(
        shp_files[0],
        state_field="STATEFP",
        code_field="COUSUBFP",
        geoid_field="GEOID",
        name_field="NAME",
        namelsad_field="NAMELSAD",
    )


def polygon_code_set(polygons: Iterable[TigerPolygon]) -> set[str]:
    return {p.code for p in polygons}

