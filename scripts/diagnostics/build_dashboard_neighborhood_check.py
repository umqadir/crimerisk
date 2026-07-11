from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
import sys

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


COUNT_COLUMNS = [
    "expected_count_murder",
    "expected_count_rape",
    "expected_count_robbery",
    "expected_count_aggravated_assault",
    "expected_count_burglary",
    "expected_count_larceny",
    "expected_count_motor_vehicle_theft",
    "expected_count_personal",
    "expected_count_property",
    "expected_count_total",
]
DENOMINATOR_COLUMNS = [
    "population_2024",
]
OFFENSE_COLUMNS = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]
PROJECTED_CRS = "EPSG:5070"
STATE_ABBR_BY_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}


def _read_vector(path: Path, *, layer: str | None = None) -> gpd.GeoDataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = zf.namelist()
            has_gdb = any(".gdb/" in member for member in members)
            has_shp = [member for member in members if member.lower().endswith(".shp")]
            with tempfile.TemporaryDirectory(prefix="dashboard_neighborhoods_") as td:
                tmp = Path(td)
                zf.extractall(tmp)
                if has_gdb:
                    gdb_dirs = sorted(tmp.glob("*.gdb"))
                    if not gdb_dirs:
                        gdb_dirs = sorted(p for p in tmp.rglob("*.gdb") if p.is_dir())
                    if not gdb_dirs:
                        raise FileNotFoundError(f"No .gdb directory found after extracting {path}")
                    return gpd.read_file(gdb_dirs[0], layer=layer)
                if has_shp:
                    shp = tmp / has_shp[0]
                    return gpd.read_file(shp, layer=layer)
        raise ValueError(f"Unsupported zip vector contents: {path}")
    return gpd.read_file(path, layer=layer)


def _normalize_neighborhoods(
    gdf: gpd.GeoDataFrame,
    *,
    id_col: str,
    name_col: str | None,
    state_col: str | None,
    states: set[str] | None,
    max_neighborhoods: int | None,
) -> gpd.GeoDataFrame:
    if id_col not in gdf.columns:
        raise ValueError(f"Neighborhood id column not found: {id_col}")
    if state_col and state_col not in gdf.columns:
        raise ValueError(f"Neighborhood state column not found: {state_col}")
    out = gdf[
        [
            id_col,
            *([name_col] if name_col and name_col in gdf.columns else []),
            *([state_col] if state_col else []),
            "geometry",
        ]
    ].copy()
    out = out.rename(columns={id_col: "neighborhood_id"})
    if name_col and name_col in out.columns:
        out = out.rename(columns={name_col: "neighborhood_name"})
    elif "neighborhood_name" not in out.columns:
        out["neighborhood_name"] = out["neighborhood_id"].astype("string")
    if state_col:
        out = out.rename(columns={state_col: "neighborhood_state"})
    out["neighborhood_id"] = out["neighborhood_id"].astype("string")
    out["neighborhood_name"] = out["neighborhood_name"].astype("string")
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.crs is None:
        out = out.set_crs("EPSG:4326")
    if states:
        if "neighborhood_state" in out.columns:
            allowed = set(states) | {STATE_ABBR_BY_FIPS.get(state, state) for state in states}
            state_values = out["neighborhood_state"].astype("string").str.strip().str.upper()
            if state_values.isin(allowed).any():
                out = out[state_values.isin(allowed)].copy()
        # Keep by TIGER-style state prefix when the id itself starts with a state FIPS.
        prefix = out["neighborhood_id"].str.slice(0, 2)
        if prefix.isin(states).any():
            out = out[prefix.isin(states)].copy()
    if max_neighborhoods is not None:
        out = out.head(int(max_neighborhoods)).copy()
    return out.reset_index(drop=True)


def _load_tiger_geometries(*, kind: str, states: set[str], tiger_dir: Path) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for state in sorted(states):
        if kind == "block_group":
            path = tiger_dir / f"tl_2020_{state}_bg.zip"
            id_col = "GEOID"
            out_col = "block_group_geoid"
        elif kind == "tract":
            path = tiger_dir / f"tl_2020_{state}_tract.zip"
            id_col = "GEOID"
            out_col = "tract_id"
        else:
            raise ValueError(f"Unsupported TIGER geometry kind: {kind}")
        if not path.exists():
            continue
        g = gpd.read_file(f"zip://{path}")[[id_col, "STATEFP", "geometry"]].copy()
        g[out_col] = g[id_col].astype("string").str.zfill(12 if kind == "block_group" else 11)
        g["state_fips"] = g["STATEFP"].astype("string").str.zfill(2)
        frames.append(g[[out_col, "state_fips", "geometry"]])
    if not frames:
        raise FileNotFoundError(f"No TIGER {kind} geometries found for states: {sorted(states)}")
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs)


def _state_filter_from_neighborhoods(neighborhoods: gpd.GeoDataFrame, *, fallback_states: set[str] | None) -> set[str]:
    if fallback_states:
        return {str(state).zfill(2) for state in fallback_states}
    bounds = neighborhoods.to_crs("EPSG:4326").total_bounds
    bg_dir = REPO_ROOT / "data" / "tiger_bg"
    states: list[str] = []
    for path in sorted(bg_dir.glob("tl_2020_*_bg.zip")):
        state = path.name.split("_")[2]
        bg = gpd.read_file(f"zip://{path}", rows=slice(0, 1))
        full = gpd.read_file(f"zip://{path}", bbox=tuple(bounds))
        if not full.empty:
            states.append(state)
        del bg, full
    if not states:
        raise ValueError("Could not infer intersecting state FIPS values for neighborhood layer.")
    return set(states)


def _build_area_weights(
    *,
    neighborhoods: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    unit_id_col: str,
) -> pd.DataFrame:
    left = neighborhoods[["neighborhood_id", "neighborhood_name", "geometry"]].to_crs(PROJECTED_CRS).copy()
    right = units[[unit_id_col, "state_fips", "geometry"]].to_crs(PROJECTED_CRS).copy()
    left["neighborhood_area_sqm"] = left.geometry.area
    candidates = gpd.sjoin(left, right, how="inner", predicate="intersects")
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.drop(columns=["index_right"], errors="ignore")
    right_geom = right.set_index(unit_id_col)["geometry"]
    candidates["unit_geometry"] = candidates[unit_id_col].map(right_geom)
    candidates["intersection_area_sqm"] = candidates.geometry.intersection(
        gpd.GeoSeries(candidates["unit_geometry"], crs=PROJECTED_CRS)
    ).area
    candidates = candidates[pd.to_numeric(candidates["intersection_area_sqm"], errors="coerce").fillna(0.0) > 0].copy()
    candidates["area_share"] = (
        pd.to_numeric(candidates["intersection_area_sqm"], errors="coerce")
        / pd.to_numeric(candidates["neighborhood_area_sqm"], errors="coerce").replace(0, np.nan)
    )
    return pd.DataFrame(
        candidates[
            [
                "neighborhood_id",
                "neighborhood_name",
                unit_id_col,
                "state_fips",
                "intersection_area_sqm",
                "neighborhood_area_sqm",
                "area_share",
            ]
        ]
    )


def _aggregate_surface(
    *,
    surface_path: Path,
    weights: pd.DataFrame,
    unit_id_col: str,
    label: str,
) -> pd.DataFrame:
    surface = pd.read_parquet(surface_path).copy()
    if unit_id_col not in surface.columns:
        raise ValueError(f"{surface_path} is missing required id column {unit_id_col}")
    surface[unit_id_col] = surface[unit_id_col].astype("string").str.zfill(12 if unit_id_col == "block_group_geoid" else 11)
    weights = weights.copy()
    weights[unit_id_col] = weights[unit_id_col].astype("string").str.zfill(12 if unit_id_col == "block_group_geoid" else 11)
    value_cols = [col for col in [*COUNT_COLUMNS, *DENOMINATOR_COLUMNS] if col in surface.columns]
    merged = weights.merge(surface[[unit_id_col, *value_cols]], on=unit_id_col, how="left")
    for col in value_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0) * pd.to_numeric(
            merged["area_share"], errors="coerce"
        ).fillna(0.0)
    grouped = (
        merged.groupby(["neighborhood_id", "neighborhood_name"], dropna=False)[value_cols]
        .sum()
        .reset_index()
    )
    grouped["source_layer"] = str(label)
    population = pd.to_numeric(grouped.get("population_2024"), errors="coerce").replace(0, np.nan)
    for offense in OFFENSE_COLUMNS:
        count_col = f"expected_count_{offense}"
        if count_col not in grouped.columns:
            continue
        grouped[f"rate_{offense}_resident"] = (
            pd.to_numeric(grouped[count_col], errors="coerce") / population * 100_000.0
        )
    return grouped


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _aggregate_dashboard_coarse_layer(
    *,
    db_path: Path,
    table_name: str,
    weights: pd.DataFrame,
    unit_id_col: str,
    id_col: str,
    value_cols: list[str],
    label: str,
) -> pd.DataFrame:
    if not value_cols:
        value_cols = ["risk_score"]
    select_cols = [id_col, *value_cols]
    query = f"select {', '.join(_quote_ident(col) for col in select_cols)} from {_quote_ident(table_name)}"
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        surface = con.execute(query).fetchdf()
    finally:
        con.close()

    surface = surface.rename(columns={id_col: unit_id_col}).copy()
    surface[unit_id_col] = surface[unit_id_col].astype("string").str.zfill(11 if unit_id_col == "tract_id" else 12)
    weights = weights.copy()
    weights[unit_id_col] = weights[unit_id_col].astype("string").str.zfill(11 if unit_id_col == "tract_id" else 12)
    merged = weights.merge(surface, on=unit_id_col, how="left")
    merged["area_share"] = pd.to_numeric(merged["area_share"], errors="coerce").fillna(0.0)

    rows: list[dict[str, object]] = []
    for (neighborhood_id, neighborhood_name), group in merged.groupby(
        ["neighborhood_id", "neighborhood_name"],
        dropna=False,
        sort=True,
    ):
        row: dict[str, object] = {
            "neighborhood_id": neighborhood_id,
            "neighborhood_name": neighborhood_name,
            "source_layer": str(label),
            "coarse_area_weight_total": float(pd.to_numeric(group["area_share"], errors="coerce").fillna(0.0).sum()),
        }
        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            weights_arr = pd.to_numeric(group["area_share"], errors="coerce")
            valid = values.notna() & weights_arr.gt(0)
            denom = float(weights_arr[valid].sum())
            row[f"dashboard_{col}_area_weighted"] = (
                float((values[valid] * weights_arr[valid]).sum() / denom) if denom > 0 else np.nan
            )
            row[f"dashboard_{col}_coverage_share"] = denom
        rows.append(row)
    return pd.DataFrame(rows)


def _load_tract_neighborhood_lookup(
    *,
    path: Path,
    tract_id_col: str,
    neighborhood_id_col: str,
    neighborhood_name_col: str | None,
    state_col: str | None,
    states: set[str] | None,
    max_neighborhoods: int | None,
) -> pd.DataFrame:
    lookup = pd.read_csv(path, dtype={tract_id_col: "string"})
    if tract_id_col not in lookup.columns:
        raise ValueError(f"Tract id column not found in lookup: {tract_id_col}")
    if neighborhood_id_col not in lookup.columns:
        raise ValueError(f"Neighborhood id column not found in lookup: {neighborhood_id_col}")
    if state_col and state_col not in lookup.columns:
        raise ValueError(f"State column not found in lookup: {state_col}")
    name_col = neighborhood_name_col if neighborhood_name_col and neighborhood_name_col in lookup.columns else neighborhood_id_col
    cols = list(dict.fromkeys([tract_id_col, neighborhood_id_col, name_col, *([state_col] if state_col else [])]))
    out = lookup[cols].copy()
    out["tract_id"] = out[tract_id_col]
    out["neighborhood_id"] = out[neighborhood_id_col]
    out["neighborhood_name"] = out[name_col]
    if state_col:
        out["state_fips"] = out[state_col]
        out["state_fips"] = out["state_fips"].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(2)
    else:
        out["state_fips"] = out["tract_id"].astype("string").str.slice(0, 2)
    out["tract_id"] = out["tract_id"].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(11)
    out["neighborhood_id"] = out["neighborhood_id"].astype("string").str.strip()
    out["neighborhood_name"] = out["neighborhood_name"].astype("string").str.strip()
    out = out[
        out["tract_id"].notna()
        & out["neighborhood_id"].notna()
        & out["neighborhood_name"].notna()
        & out["neighborhood_id"].ne("")
        & out["neighborhood_name"].ne("")
    ].copy()
    if states:
        out = out[out["state_fips"].isin({str(state).zfill(2) for state in states})].copy()
    out = out.drop_duplicates(["neighborhood_id", "tract_id"]).copy()
    if max_neighborhoods is not None:
        keep = out[["neighborhood_id"]].drop_duplicates().head(int(max_neighborhoods))["neighborhood_id"]
        out = out[out["neighborhood_id"].isin(set(keep))].copy()
    out["area_share"] = 1.0
    return out[["neighborhood_id", "neighborhood_name", "tract_id", "state_fips", "area_share"]].reset_index(drop=True)


def build_dashboard_neighborhood_lookup_check(
    *,
    lookup_path: Path,
    lookup_tract_id_col: str,
    lookup_neighborhood_id_col: str,
    lookup_neighborhood_name_col: str | None,
    lookup_state_col: str | None,
    states: set[str] | None,
    max_neighborhoods: int | None,
    tract_surface_path: Path,
    dashboard_coarse_db: Path | None,
    dashboard_coarse_table: str,
    dashboard_coarse_id_col: str,
    dashboard_coarse_value_cols: list[str],
    dashboard_coarse_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    tract_weights = _load_tract_neighborhood_lookup(
        path=lookup_path,
        tract_id_col=lookup_tract_id_col,
        neighborhood_id_col=lookup_neighborhood_id_col,
        neighborhood_name_col=lookup_neighborhood_name_col,
        state_col=lookup_state_col,
        states=states,
        max_neighborhoods=max_neighborhoods,
    )
    if tract_weights.empty:
        raise ValueError("No tract-neighborhood lookup rows remain after filtering.")

    tract_agg = _aggregate_surface(
        surface_path=tract_surface_path,
        weights=tract_weights,
        unit_id_col="tract_id",
        label="tract_ags_core",
    )
    outputs = [tract_agg]
    comparison = tract_agg.copy()
    coarse_agg = pd.DataFrame()
    if dashboard_coarse_db is not None:
        coarse_agg = _aggregate_dashboard_coarse_layer(
            db_path=dashboard_coarse_db,
            table_name=dashboard_coarse_table,
            weights=tract_weights,
            unit_id_col="tract_id",
            id_col=dashboard_coarse_id_col,
            value_cols=dashboard_coarse_value_cols,
            label=dashboard_coarse_label,
        )
        outputs.append(coarse_agg)
        comparison = comparison.merge(
            coarse_agg.drop(columns=["source_layer", "neighborhood_name"], errors="ignore"),
            on="neighborhood_id",
            how="left",
        )
        if "expected_count_total" in comparison.columns:
            comparison["crimerisk_expected_count_total_rank_desc"] = pd.to_numeric(
                comparison["expected_count_total"],
                errors="coerce",
            ).rank(ascending=False, method="average")
        if "dashboard_risk_score_area_weighted" in comparison.columns:
            comparison["dashboard_risk_score_rank_desc"] = pd.to_numeric(
                comparison["dashboard_risk_score_area_weighted"],
                errors="coerce",
            ).rank(ascending=False, method="average")
        if {"crimerisk_expected_count_total_rank_desc", "dashboard_risk_score_rank_desc"}.issubset(comparison.columns):
            comparison["crimerisk_expected_count_rank_minus_dashboard_rank"] = (
                comparison["crimerisk_expected_count_total_rank_desc"] - comparison["dashboard_risk_score_rank_desc"]
            )

    combined = pd.concat(outputs, ignore_index=True, sort=False)
    states_present = sorted(tract_weights["state_fips"].dropna().astype(str).unique().tolist())
    summary = {
        "neighborhood_basis": "tract_lookup",
        "neighborhood_count": int(tract_weights["neighborhood_id"].nunique()),
        "states": states_present,
        "tract_weight_rows": int(len(tract_weights)),
        "output_rows": int(len(combined)),
        "comparison_rows": int(len(comparison)),
        "tract_surface_path": str(tract_surface_path),
        "dashboard_coarse_db": str(dashboard_coarse_db) if dashboard_coarse_db else None,
        "dashboard_coarse_table": str(dashboard_coarse_table) if dashboard_coarse_db else None,
        "dashboard_coarse_value_cols": list(dashboard_coarse_value_cols or ["risk_score"])
        if dashboard_coarse_db
        else [],
        "lookup_path": str(lookup_path),
    }
    if not coarse_agg.empty:
        summary["dashboard_coarse_rows"] = int(len(coarse_agg))
        if "dashboard_risk_score_area_weighted" in comparison.columns:
            valid = comparison[["dashboard_risk_score_area_weighted", "expected_count_total"]].dropna()
            summary["dashboard_risk_score_rows"] = int(len(valid))
            if len(valid) >= 2:
                summary["dashboard_risk_score_vs_crimerisk_expected_count_total_spearman"] = float(
                    comparison["dashboard_risk_score_area_weighted"].corr(
                        comparison["expected_count_total"],
                        method="spearman",
                    )
                )
    return combined, comparison, summary


def build_dashboard_neighborhood_check(
    *,
    neighborhoods_path: Path,
    neighborhood_id_col: str,
    neighborhood_name_col: str | None,
    neighborhood_state_col: str | None,
    neighborhood_layer: str | None,
    states: set[str] | None,
    max_neighborhoods: int | None,
    bg_surface_path: Path,
    tract_surface_path: Path | None,
    dashboard_coarse_db: Path | None,
    dashboard_coarse_table: str,
    dashboard_coarse_id_col: str,
    dashboard_coarse_value_cols: list[str],
    dashboard_coarse_label: str,
    bg_tiger_dir: Path,
    tract_tiger_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_neighborhoods = _read_vector(neighborhoods_path, layer=neighborhood_layer)
    neighborhoods = _normalize_neighborhoods(
        raw_neighborhoods,
        id_col=neighborhood_id_col,
        name_col=neighborhood_name_col,
        state_col=neighborhood_state_col,
        states=states,
        max_neighborhoods=max_neighborhoods,
    )
    state_filter = _state_filter_from_neighborhoods(neighborhoods, fallback_states=states)

    bg_units = _load_tiger_geometries(kind="block_group", states=state_filter, tiger_dir=bg_tiger_dir)
    bg_weights = _build_area_weights(
        neighborhoods=neighborhoods,
        units=bg_units,
        unit_id_col="block_group_geoid",
    )
    if bg_weights.empty:
        raise ValueError("No neighborhood/block-group intersections produced.")
    bg_agg = _aggregate_surface(
        surface_path=bg_surface_path,
        weights=bg_weights,
        unit_id_col="block_group_geoid",
        label="block_group_ags_core",
    )

    outputs = [bg_agg]
    comparison = bg_agg.copy()
    tract_weights = pd.DataFrame()
    if tract_surface_path is not None or dashboard_coarse_db is not None:
        tract_units = _load_tiger_geometries(kind="tract", states=state_filter, tiger_dir=tract_tiger_dir)
        tract_weights = _build_area_weights(
            neighborhoods=neighborhoods,
            units=tract_units,
            unit_id_col="tract_id",
        )
    if tract_surface_path is not None:
        tract_agg = _aggregate_surface(
            surface_path=tract_surface_path,
            weights=tract_weights,
            unit_id_col="tract_id",
            label="tract_ags_core",
        )
        outputs.append(tract_agg)
        compare_cols = [
            "neighborhood_id",
            "neighborhood_name",
            "expected_count_total",
            "expected_count_personal",
            "expected_count_property",
        ]
        left = bg_agg[[col for col in compare_cols if col in bg_agg.columns]].copy()
        right = tract_agg[[col for col in compare_cols if col in tract_agg.columns]].copy()
        comparison = left.merge(right, on="neighborhood_id", how="outer", suffixes=("_block_group", "_tract"))
        for col in ["expected_count_total", "expected_count_personal", "expected_count_property"]:
            bg_col = f"{col}_block_group"
            tract_col = f"{col}_tract"
            if bg_col in comparison.columns and tract_col in comparison.columns:
                comparison[f"{col}_block_group_minus_tract"] = (
                    pd.to_numeric(comparison[bg_col], errors="coerce")
                    - pd.to_numeric(comparison[tract_col], errors="coerce")
                )
    coarse_agg = pd.DataFrame()
    if dashboard_coarse_db is not None:
        if tract_weights.empty:
            raise ValueError("No neighborhood/tract intersections produced for dashboard coarse layer.")
        coarse_agg = _aggregate_dashboard_coarse_layer(
            db_path=dashboard_coarse_db,
            table_name=dashboard_coarse_table,
            weights=tract_weights,
            unit_id_col="tract_id",
            id_col=dashboard_coarse_id_col,
            value_cols=dashboard_coarse_value_cols,
            label=dashboard_coarse_label,
        )
        outputs.append(coarse_agg)
        comparison = comparison.merge(
            coarse_agg.drop(columns=["source_layer", "neighborhood_name"], errors="ignore"),
            on="neighborhood_id",
            how="left",
        )
        if "expected_count_total_block_group" in comparison.columns:
            comparison["crimerisk_expected_count_total_rank_desc"] = pd.to_numeric(
                comparison["expected_count_total_block_group"],
                errors="coerce",
            ).rank(ascending=False, method="average")
        elif "expected_count_total" in comparison.columns:
            comparison["crimerisk_expected_count_total_rank_desc"] = pd.to_numeric(
                comparison["expected_count_total"],
                errors="coerce",
            ).rank(ascending=False, method="average")
        if "dashboard_risk_score_area_weighted" in comparison.columns:
            comparison["dashboard_risk_score_rank_desc"] = pd.to_numeric(
                comparison["dashboard_risk_score_area_weighted"],
                errors="coerce",
            ).rank(ascending=False, method="average")
        if {"crimerisk_expected_count_total_rank_desc", "dashboard_risk_score_rank_desc"}.issubset(comparison.columns):
            comparison["crimerisk_expected_count_rank_minus_dashboard_rank"] = (
                comparison["crimerisk_expected_count_total_rank_desc"] - comparison["dashboard_risk_score_rank_desc"]
            )

    combined = pd.concat(outputs, ignore_index=True, sort=False)
    summary = {
        "neighborhood_count": int(neighborhoods["neighborhood_id"].nunique()),
        "states": sorted(state_filter),
        "block_group_weight_rows": int(len(bg_weights)),
        "tract_weight_rows": int(len(tract_weights)) if not tract_weights.empty else 0,
        "output_rows": int(len(combined)),
        "comparison_rows": int(len(comparison)),
        "bg_surface_path": str(bg_surface_path),
        "tract_surface_path": str(tract_surface_path) if tract_surface_path else None,
        "dashboard_coarse_db": str(dashboard_coarse_db) if dashboard_coarse_db else None,
        "dashboard_coarse_table": str(dashboard_coarse_table) if dashboard_coarse_db else None,
        "dashboard_coarse_value_cols": list(dashboard_coarse_value_cols or ["risk_score"])
        if dashboard_coarse_db
        else [],
    }
    if not coarse_agg.empty:
        summary["dashboard_coarse_rows"] = int(len(coarse_agg))
        if "dashboard_risk_score_area_weighted" in comparison.columns:
            count_col = "expected_count_total_block_group" if "expected_count_total_block_group" in comparison.columns else "expected_count_total"
            valid = comparison[["dashboard_risk_score_area_weighted", count_col]].dropna()
            summary["dashboard_risk_score_rows"] = int(len(valid))
            if len(valid) >= 2:
                summary["dashboard_risk_score_vs_crimerisk_expected_count_total_spearman"] = float(
                    comparison["dashboard_risk_score_area_weighted"].corr(
                        comparison[count_col],
                        method="spearman",
                    )
                )
    return combined, comparison, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate CrimeRisk surfaces to dashboard neighborhood polygons or tract lookups.")
    parser.add_argument("--neighborhoods", type=Path, default=None)
    parser.add_argument("--neighborhood-layer", default=None)
    parser.add_argument("--neighborhood-id-col", default=None)
    parser.add_argument("--neighborhood-name-col", default=None)
    parser.add_argument("--neighborhood-state-col", default=None)
    parser.add_argument("--tract-neighborhood-lookup", type=Path, default=None)
    parser.add_argument("--lookup-tract-id-col", default="tract_id")
    parser.add_argument("--lookup-neighborhood-id-col", default="location_name")
    parser.add_argument("--lookup-neighborhood-name-col", default="location_name")
    parser.add_argument("--lookup-state-col", default="state_fips")
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--max-neighborhoods", type=int, default=None)
    parser.add_argument(
        "--bg-surface",
        type=Path,
        default=REPO_ROOT / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet",
    )
    parser.add_argument(
        "--tract-surface",
        type=Path,
        default=REPO_ROOT / "state" / "output" / "crimerisk_tract_2024_ags_core.parquet",
    )
    parser.add_argument("--bg-tiger-dir", type=Path, default=REPO_ROOT / "data" / "tiger_bg")
    parser.add_argument("--tract-tiger-dir", type=Path, default=REPO_ROOT / "data" / "tiger_tracts")
    parser.add_argument("--dashboard-coarse-db", type=Path, default=None)
    parser.add_argument("--dashboard-coarse-table", default="tracts")
    parser.add_argument("--dashboard-coarse-id-col", default="tract_id")
    parser.add_argument("--dashboard-coarse-value-col", action="append", default=[])
    parser.add_argument("--dashboard-coarse-label", default="dashboard_current_coarse_tract")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "materials" / "tables" / "dashboard_neighborhood_crimerisk_aggregation.csv",
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=REPO_ROOT / "materials" / "tables" / "dashboard_neighborhood_coarse_comparison.csv",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "dashboard_neighborhood_check_2024.json",
    )
    args = parser.parse_args()

    state_filter = {str(state).zfill(2) for state in args.states} if args.states else None
    if args.tract_neighborhood_lookup is not None:
        combined, comparison, summary = build_dashboard_neighborhood_lookup_check(
            lookup_path=args.tract_neighborhood_lookup,
            lookup_tract_id_col=str(args.lookup_tract_id_col),
            lookup_neighborhood_id_col=str(args.lookup_neighborhood_id_col),
            lookup_neighborhood_name_col=args.lookup_neighborhood_name_col,
            lookup_state_col=args.lookup_state_col,
            states=state_filter,
            max_neighborhoods=args.max_neighborhoods,
            tract_surface_path=args.tract_surface,
            dashboard_coarse_db=args.dashboard_coarse_db,
            dashboard_coarse_table=str(args.dashboard_coarse_table),
            dashboard_coarse_id_col=str(args.dashboard_coarse_id_col),
            dashboard_coarse_value_cols=[str(col) for col in args.dashboard_coarse_value_col],
            dashboard_coarse_label=str(args.dashboard_coarse_label),
        )
    else:
        if args.neighborhoods is None:
            raise ValueError("Either --neighborhoods or --tract-neighborhood-lookup is required.")
        if args.neighborhood_id_col is None:
            raise ValueError("--neighborhood-id-col is required in polygon mode.")
        combined, comparison, summary = build_dashboard_neighborhood_check(
            neighborhoods_path=args.neighborhoods,
            neighborhood_id_col=str(args.neighborhood_id_col),
            neighborhood_name_col=args.neighborhood_name_col,
            neighborhood_state_col=args.neighborhood_state_col,
            neighborhood_layer=args.neighborhood_layer,
            states=state_filter,
            max_neighborhoods=args.max_neighborhoods,
            bg_surface_path=args.bg_surface,
            tract_surface_path=args.tract_surface,
            dashboard_coarse_db=args.dashboard_coarse_db,
            dashboard_coarse_table=str(args.dashboard_coarse_table),
            dashboard_coarse_id_col=str(args.dashboard_coarse_id_col),
            dashboard_coarse_value_cols=[str(col) for col in args.dashboard_coarse_value_col],
            dashboard_coarse_label=str(args.dashboard_coarse_label),
            bg_tiger_dir=args.bg_tiger_dir,
            tract_tiger_dir=args.tract_tiger_dir,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    comparison.to_csv(args.comparison_out, index=False)
    args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
