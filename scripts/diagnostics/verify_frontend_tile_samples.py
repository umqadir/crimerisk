"""Compare bounded max-zoom PMTiles samples with release parquet values.

The verifier selects a deterministic small set of urban, rural, extreme, zero,
and no-data rows, derives an interior coordinate from TIGER geometry, fetches
only the corresponding z12 tiles through the ``pmtiles`` CLI, and compares the
encoded attributes with the source values under the frontend rounding contract.

Run with ``uv run --with mapbox-vector-tile python`` so the MVT decoder remains
an isolated diagnostic dependency rather than part of the model runtime.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


OFFENSES = {
    "murder": "mur",
    "rape": "rap",
    "robbery": "rob",
    "aggravated_assault": "agg",
    "burglary": "bur",
    "larceny": "lar",
    "motor_vehicle_theft": "mvt",
}
REGULAR = tuple(o for o in OFFENSES if o not in {"murder", "rape"})
AGGREGATES = {
    "index_event_burden_resident": "i_tot",
    "index_personal_burden_resident": "i_per",
    "index_property_burden_resident": "i_pro",
    "multi_offense_relative_score_event_weighted": "i_evw",
    "multi_offense_relative_score_equal_offense": "i_eq",
    "multi_offense_relative_score_personal_event_weighted": "i_per_adj",
    "multi_offense_relative_score_property_event_weighted": "i_pro_adj",
}
TRACT_AGGREGATES = {**AGGREGATES, "index_harm_burden_resident": "i_harm"}
BG_GEOID = "block_group_geoid"
TRACT_GEOID = "tract_id"
ZOOM = 12


def tile_xy(lon: float, lat: float, zoom: int = ZOOM) -> tuple[int, int]:
    scale = 1 << zoom
    x = int((lon + 180.0) / 360.0 * scale)
    limited = min(85.05112878, max(-85.05112878, lat))
    y = int((1.0 - math.asinh(math.tan(math.radians(limited))) / math.pi) / 2.0 * scale)
    return min(scale - 1, max(0, x)), min(scale - 1, max(0, y))


def expected_fields(row: pd.Series, lane: str, parent: pd.Series | None = None) -> dict[str, tuple[Any, int | None]]:
    """Return tile key -> (source value, decimal places), including nulls."""
    fields: dict[str, tuple[Any, int | None]] = {}
    aggregates = TRACT_AGGREGATES if lane == "tract" else AGGREGATES
    for source, key in aggregates.items():
        fields[key] = (row[source], 1)
    fields["d_tot"] = (row["crime_density_total"], 4)
    for offense, suffix in OFFENSES.items():
        for source_prefix, key_prefix, dp in (
            ("expected_count", "ec", 2),
            ("crime_density", "d", 4),
            ("effective_numerator_support", "es", 2),
            ("primary_denominator", "xb", 2),
        ):
            fields[f"{key_prefix}_{suffix}"] = (row[f"{source_prefix}_{offense}"], dp)
        if lane == "tract" or offense in REGULAR:
            fields[f"ip_{suffix}"] = (row[f"index_{offense}_primary"], 1)
            fields[f"ir_{suffix}"] = (row[f"index_{offense}_resident"], 1)
            fields[f"r_{suffix}"] = (row[f"rate_{offense}_primary"], 1)
    if lane == "block_group":
        if parent is None:
            raise ValueError("block-group checks require the parent tract row")
        for offense in ("murder", "rape"):
            suffix = OFFENSES[offense]
            fields[f"ipt_{suffix}"] = (parent[f"index_{offense}_primary"], 1)
            fields[f"rpt_{suffix}"] = (parent[f"rate_{offense}_primary"], 1)
            # 01 rounds exposure bases to 2dp before the CSV boundary, then 03
            # currently rounds the complete baked tract field family to 1dp.
            fields[f"xbt_{suffix}"] = (
                float(np.round(float(parent[f"primary_denominator_{offense}"]), 2)),
                1,
            )
            # These block-group point fields are null by policy and must not be
            # resurrected by the tile encoder.
            fields[f"ip_{suffix}"] = (None, 1)
            fields[f"ir_{suffix}"] = (None, 1)
            fields[f"r_{suffix}"] = (None, 1)
        fields["i_harm"] = (None, 1)
    return fields


def is_null(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def compare_properties(
    *, geoid: str, lane: str, properties: dict[str, Any], expected: dict[str, tuple[Any, int | None]]
) -> list[str]:
    errors: list[str] = []
    geoid_key = BG_GEOID if lane == "block_group" else TRACT_GEOID
    if str(properties.get(geoid_key, "")) != geoid:
        errors.append(f"{geoid_key}: encoded {properties.get(geoid_key)!r}, expected {geoid!r}")
    for key, (source_value, dp) in expected.items():
        actual = properties.get(key)
        if is_null(source_value):
            if actual is not None:
                errors.append(f"{key}: encoded {actual!r}, expected omitted/null")
            continue
        # The build uses pandas/NumPy rounding before GeoJSON export. Python's
        # built-in round differs at binary half cases such as 62.15 -> 1dp.
        wanted = float(np.round(float(source_value), int(dp))) if dp is not None else source_value
        if actual is None:
            errors.append(f"{key}: missing, expected {wanted!r}")
            continue
        tolerance = 10 ** (-(int(dp) + 2)) if dp is not None else 0
        if not math.isclose(float(actual), float(wanted), rel_tol=0, abs_tol=tolerance):
            errors.append(f"{key}: encoded {actual!r}, expected {wanted!r}")
    return errors


def parquet_columns(lane: str) -> list[str]:
    columns = [BG_GEOID if lane == "block_group" else TRACT_GEOID, "population_2025", "land_area_sq_mi", "special_use_tract_flag", "crime_density_total"]
    columns.extend((TRACT_AGGREGATES if lane == "tract" else AGGREGATES).keys())
    for offense in OFFENSES:
        columns.extend(
            [
                f"index_{offense}_primary", f"index_{offense}_resident",
                f"rate_{offense}_primary", f"expected_count_{offense}",
                f"crime_density_{offense}", f"effective_numerator_support_{offense}",
                f"primary_denominator_{offense}",
            ]
        )
    return list(dict.fromkeys(columns))


def choose_rows(bg: pd.DataFrame, tract: pd.DataFrame) -> list[dict[str, str]]:
    """Select stable strata using value ordering with GEOID as the tie-break."""
    bg = bg.copy()
    tract = tract.copy()
    bg[BG_GEOID] = bg[BG_GEOID].astype("string").str.zfill(12)
    tract[TRACT_GEOID] = tract[TRACT_GEOID].astype("string").str.zfill(11)
    ordinary = bg[~bg["special_use_tract_flag"].fillna(False).astype(bool)].copy()
    ordinary["_pop_density"] = ordinary["population_2025"] / ordinary["land_area_sq_mi"].replace(0, np.nan)
    regular_primary = [f"index_{o}_primary" for o in REGULAR]
    ordinary["_max_primary"] = ordinary[regular_primary].max(axis=1, skipna=True)

    def first(frame: pd.DataFrame, order: list[str], ascending: list[bool]) -> pd.Series:
        if frame.empty:
            raise SystemExit("a required sampling stratum is empty")
        return frame.sort_values(order, ascending=ascending, kind="mergesort").iloc[0]

    selected: list[tuple[str, pd.Series]] = [
        ("ordinary_urban", first(ordinary.dropna(subset=["_pop_density"]), ["_pop_density", BG_GEOID], [False, True])),
        ("ordinary_rural", first(ordinary[ordinary["_pop_density"] > 0], ["_pop_density", BG_GEOID], [True, True])),
        ("regular_primary_extreme", first(ordinary.dropna(subset=["_max_primary"]), ["_max_primary", BG_GEOID], [False, True])),
        ("valid_zero", first(bg[bg["crime_density_total"].eq(0)], [BG_GEOID], [True])),
        ("regular_no_data", first(bg[bg[regular_primary].isna().any(axis=1)], [BG_GEOID], [True])),
    ]
    plan = [{"lane": "block_group", "geoid": str(row[BG_GEOID]), "tag": tag} for tag, row in selected]
    for offense in ("murder", "rape"):
        column = f"index_{offense}_primary"
        row = first(tract.dropna(subset=[column]), [column, TRACT_GEOID], [False, True])
        plan.append({"lane": "tract", "geoid": str(row[TRACT_GEOID]), "tag": f"{offense}_primary_extreme"})
    harm = first(tract.dropna(subset=["index_harm_burden_resident"]), ["index_harm_burden_resident", TRACT_GEOID], [False, True])
    plan.append({"lane": "tract", "geoid": str(harm[TRACT_GEOID]), "tag": "harm_extreme"})
    # Parent tracts prove that the BG baked rare fields and tract lane agree.
    for item in list(plan):
        if item["lane"] == "block_group":
            plan.append({"lane": "tract", "geoid": item["geoid"][:11], "tag": f"parent_of_{item['tag']}"})
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in plan:
        key = (item["lane"], item["geoid"])
        if key in unique:
            unique[key]["tag"] += f",{item['tag']}"
        else:
            unique[key] = item
    return list(unique.values())


def add_coordinates(plan: list[dict[str, Any]], bg_geometry: Path, tract_geometry: Path) -> None:
    by_lane_state: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in plan:
        by_lane_state[(item["lane"], item["geoid"][:2])].append(item)
    for (lane, state), items in by_lane_state.items():
        directory = bg_geometry if lane == "block_group" else tract_geometry
        suffix = "bg" if lane == "block_group" else "tract"
        length = 12 if lane == "block_group" else 11
        path = directory / f"tl_2020_{state}_{suffix}.zip"
        frame = gpd.read_file(f"zip://{path}")[['GEOID', 'geometry']].to_crs(4326)
        frame["GEOID"] = frame["GEOID"].astype(str).str.zfill(length)
        wanted = {item["geoid"] for item in items}
        frame = frame[frame["GEOID"].isin(wanted)].set_index("GEOID")
        for item in items:
            if item["geoid"] not in frame.index:
                raise SystemExit(f"missing TIGER geometry for {lane} {item['geoid']}")
            point = frame.loc[item["geoid"], "geometry"].representative_point()
            item["lon"], item["lat"] = float(point.x), float(point.y)


def fetch_tile(archive: str, x: int, y: int) -> bytes:
    result = subprocess.run(
        ["pmtiles", "tile", archive, str(ZOOM), str(x), str(y)],
        check=True,
        capture_output=True,
    )
    data = result.stdout
    return gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data


def decoded_properties(data: bytes, layer: str, geoid_key: str, geoid: str) -> dict[str, Any]:
    try:
        import mapbox_vector_tile
    except ImportError as exc:
        raise SystemExit("mapbox-vector-tile is required; run with `uv run --with mapbox-vector-tile python ...`") from exc
    decoded = mapbox_vector_tile.decode(data)
    features = (decoded.get(layer) or {}).get("features") or []
    matches = [f.get("properties") or {} for f in features if str((f.get("properties") or {}).get(geoid_key, "")) == geoid]
    if len(matches) != 1:
        raise SystemExit(f"expected one {layer} feature {geoid}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-group-parquet", type=Path, required=True)
    parser.add_argument("--tract-parquet", type=Path, required=True)
    parser.add_argument("--block-group-pmtiles", required=True)
    parser.add_argument("--tract-pmtiles", required=True)
    parser.add_argument("--bg-geometry-dir", type=Path, required=True)
    parser.add_argument("--tract-geometry-dir", type=Path, required=True)
    parser.add_argument("--sample-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bg = pd.read_parquet(args.block_group_parquet, columns=parquet_columns("block_group"))
    tract = pd.read_parquet(args.tract_parquet, columns=parquet_columns("tract"))
    bg[BG_GEOID] = bg[BG_GEOID].astype("string").str.zfill(12)
    tract[TRACT_GEOID] = tract[TRACT_GEOID].astype("string").str.zfill(11)
    bg_rows = bg.set_index(BG_GEOID, drop=False)
    tract_rows = tract.set_index(TRACT_GEOID, drop=False)
    plan = json.loads(args.sample_plan.read_text()) if args.sample_plan else choose_rows(bg, tract)
    if not all("lon" in item and "lat" in item for item in plan):
        add_coordinates(plan, args.bg_geometry_dir, args.tract_geometry_dir)

    tile_cache: dict[tuple[str, int, int], bytes] = {}
    results: list[dict[str, Any]] = []
    all_errors: list[str] = []
    for item in plan:
        lane, geoid = item["lane"], str(item["geoid"])
        archive = args.block_group_pmtiles if lane == "block_group" else args.tract_pmtiles
        layer = "blockgroups" if lane == "block_group" else "tracts"
        geoid_key = BG_GEOID if lane == "block_group" else TRACT_GEOID
        x, y = tile_xy(float(item["lon"]), float(item["lat"]))
        cache_key = (archive, x, y)
        if cache_key not in tile_cache:
            tile_cache[cache_key] = fetch_tile(archive, x, y)
        properties = decoded_properties(tile_cache[cache_key], layer, geoid_key, geoid)
        row = bg_rows.loc[geoid] if lane == "block_group" else tract_rows.loc[geoid]
        parent = tract_rows.loc[geoid[:11]] if lane == "block_group" else None
        errors = compare_properties(
            geoid=geoid, lane=lane, properties=properties,
            expected=expected_fields(row, lane, parent),
        )
        all_errors.extend(f"{lane} {geoid} [{item['tag']}]: {error}" for error in errors)
        results.append({**item, "z": ZOOM, "x": x, "y": y, "checked_fields": len(expected_fields(row, lane, parent)), "ok": not errors, "errors": errors})

    payload = {
        "schema_version": 1,
        "ok": not all_errors,
        "sample_count": len(results),
        "tile_request_count": len(tile_cache),
        "coverage": {
            "offenses": list(OFFENSES),
            "measures": ["activity_adjusted", "resident", "density"],
            "composites": list(TRACT_AGGREGATES.values()),
            "strata": sorted({tag for item in results for tag in item["tag"].split(",")}),
        },
        "samples": results,
        "errors": all_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("ok", "sample_count", "tile_request_count", "errors")}, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
