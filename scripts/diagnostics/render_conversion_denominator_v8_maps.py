from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_BG = REPO_ROOT / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet"
TIGER_BG_DIR = REPO_ROOT / "data" / "tiger_bg"
TARGET_STL_AIRPORT_BG = "291892218003"


def _read_surface(path: Path, field: str) -> pd.DataFrame:
    columns = list(dict.fromkeys([
        "block_group_geoid",
        "state_fips",
        field,
        "index_motor_vehicle_theft_primary",
        "primary_denominator_motor_vehicle_theft",
        "vehicle_exposure_2024",
        "estimate_mode_motor_vehicle_theft",
    ]))
    available = pd.read_parquet(path, engine="pyarrow").columns
    keep = [col for col in columns if col in available]
    out = pd.read_parquet(path, columns=keep)
    out["block_group_geoid"] = out["block_group_geoid"].astype("string").str.zfill(12)
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    return out


def _read_geometry(states: Iterable[str]) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for state in sorted(set(str(s).zfill(2) for s in states)):
        path = TIGER_BG_DIR / f"tl_2020_{state}_bg.zip"
        if not path.exists():
            continue
        gdf = gpd.read_file(f"zip://{path}")
        geoid_col = "GEOID" if "GEOID" in gdf.columns else "GEOID20"
        frames.append(
            gdf[[geoid_col, "geometry"]].rename(columns={geoid_col: "block_group_geoid"})
        )
    if not frames:
        raise FileNotFoundError("No TIGER block-group geometry files found for requested states")
    out = pd.concat(frames, ignore_index=True)
    out["block_group_geoid"] = out["block_group_geoid"].astype("string").str.zfill(12)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs).to_crs("EPSG:5070")


def _surface_pair(candidate_bg: Path, field: str) -> pd.DataFrame:
    v6 = _read_surface(PROMOTED_BG, field).rename(columns={field: f"{field}_v6"})
    v8 = _read_surface(candidate_bg, field).rename(columns={field: f"{field}_v8"})
    cols = ["block_group_geoid", "state_fips", f"{field}_v6"]
    out = v6[cols].merge(v8[["block_group_geoid", f"{field}_v8"]], on="block_group_geoid", how="inner")
    return out


def _plot_pair(
    gdf: gpd.GeoDataFrame,
    *,
    field: str,
    label: str,
    title: str,
    out_path: Path,
    simplify_tolerance: float,
    target_bg: str | None = None,
) -> dict[str, object]:
    plot_gdf = gdf.copy()
    if simplify_tolerance > 0:
        plot_gdf["geometry"] = plot_gdf.geometry.simplify(simplify_tolerance, preserve_topology=False)

    values = pd.concat(
        [
            pd.to_numeric(plot_gdf[f"{field}_v6"], errors="coerce"),
            pd.to_numeric(plot_gdf[f"{field}_v8"], errors="coerce"),
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan)
    positive = values[values.gt(0)]
    if positive.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanquantile(positive, 0.02))
        vmax = float(np.nanquantile(positive, 0.98))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(positive.max())
        if not np.isfinite(vmin) or vmin < 0:
            vmin = 0.0
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = Normalize(vmin=np.log1p(vmin), vmax=np.log1p(vmax))
    plot_gdf["_v6_plot"] = np.log1p(pd.to_numeric(plot_gdf[f"{field}_v6"], errors="coerce").clip(lower=0.0))
    plot_gdf["_v8_plot"] = np.log1p(pd.to_numeric(plot_gdf[f"{field}_v8"], errors="coerce").clip(lower=0.0))

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, col, subtitle in zip(axes, ["_v6_plot", "_v8_plot"], ["v6 promoted", "v8 candidate"], strict=True):
        plot_gdf.plot(
            column=col,
            ax=ax,
            cmap="magma",
            norm=norm,
            linewidth=0,
            missing_kwds={"color": "#eeeeee"},
            rasterized=True,
        )
        if target_bg is not None:
            target = plot_gdf.loc[plot_gdf["block_group_geoid"].astype("string").eq(target_bg)]
            if not target.empty:
                target.boundary.plot(ax=ax, color="#3fe0ff", linewidth=1.8)
        ax.set_title(subtitle, fontsize=12)
        ax.set_axis_off()
    fig.suptitle(f"{title}\n{label}; shared log color scale, clipped p02-p98", fontsize=14)

    sm = plt.cm.ScalarMappable(norm=norm, cmap="magma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035, pad=0.02)
    tick_values = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(np.log1p(tick_values))
    cbar.set_ticklabels([f"{v:.0f}" for v in tick_values])
    cbar.set_label(label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return {
        "path": str(out_path.relative_to(REPO_ROOT)),
        "rows": int(len(plot_gdf)),
        "field": field,
        "vmin_p02": vmin,
        "vmax_p98": vmax,
    }


def _crop_around_target(gdf: gpd.GeoDataFrame, target_bg: str, *, buffer_m: float) -> gpd.GeoDataFrame:
    target = gdf.loc[gdf["block_group_geoid"].astype("string").eq(target_bg)]
    if target.empty:
        raise ValueError(f"target block group {target_bg} not present")
    bounds = target.geometry.buffer(buffer_m).total_bounds
    return gdf.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]].copy()


def render(candidate_run: str) -> dict[str, object]:
    candidate_dir = REPO_ROOT / "state" / "candidates" / candidate_run
    candidate_bg = candidate_dir / "crimerisk_block_group_2024_ags_core.parquet"
    out_dir = candidate_dir / "conversion_denominator_v8_evidence" / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_field = "index_total_primary_event_weighted"
    mvt_field = "index_motor_vehicle_theft_primary"
    total_pair = _surface_pair(candidate_bg, total_field)
    states = sorted(total_pair["state_fips"].dropna().astype(str).str.zfill(2).unique())
    geometry = _read_geometry(states)

    total_gdf = geometry.merge(total_pair, on="block_group_geoid", how="inner")
    national = total_gdf[~total_gdf["state_fips"].isin(["02", "15", "72"])].copy()
    florida = total_gdf[total_gdf["state_fips"].eq("12")].copy()

    mvt_pair = _surface_pair(candidate_bg, mvt_field)
    stl_geometry = _read_geometry(["29"])
    stl_gdf = stl_geometry.merge(mvt_pair, on="block_group_geoid", how="inner")
    stl_crop = _crop_around_target(stl_gdf, TARGET_STL_AIRPORT_BG, buffer_m=9000.0)

    records = [
        _plot_pair(
            national,
            field=total_field,
            label="Total primary event-weighted index",
            title="National v6 promoted vs v8 candidate",
            out_path=out_dir / "national_total_index_v6_vs_v8.png",
            simplify_tolerance=650.0,
        ),
        _plot_pair(
            florida,
            field=total_field,
            label="Total primary event-weighted index",
            title="Florida v6 promoted vs v8 candidate",
            out_path=out_dir / "florida_total_index_v6_vs_v8.png",
            simplify_tolerance=120.0,
        ),
        _plot_pair(
            stl_crop,
            field=mvt_field,
            label="MVT primary index",
            title=f"St. Louis airport area v6 promoted vs v8 candidate ({TARGET_STL_AIRPORT_BG})",
            out_path=out_dir / "stl_airport_mvt_index_v6_vs_v8.png",
            simplify_tolerance=5.0,
            target_bg=TARGET_STL_AIRPORT_BG,
        ),
    ]
    summary = {
        "candidate_run": candidate_run,
        "renders": records,
        "stl_airport_block_group": TARGET_STL_AIRPORT_BG,
    }
    (out_dir / "render_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", default="conversion-denominator-v8")
    args = parser.parse_args()
    print(json.dumps(render(args.candidate_run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
