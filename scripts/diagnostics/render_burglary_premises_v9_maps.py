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
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_BG = REPO_ROOT / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet"
TIGER_BG_DIR = REPO_ROOT / "data" / "tiger_bg"
DEFAULT_CANDIDATE_RUN = "burglary-premises-v9"
FIELD = "index_burglary_primary"
LA_METRO_COUNTIES = {"06037", "06059", "06065", "06071", "06111"}


def _read_surface(path: Path, field: str) -> pd.DataFrame:
    keep = ["block_group_geoid", "state_fips", field]
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
        frames.append(gdf[[geoid_col, "geometry"]].rename(columns={geoid_col: "block_group_geoid"}))
    if not frames:
        raise FileNotFoundError("No TIGER block-group geometry files found for requested states")
    out = pd.concat(frames, ignore_index=True)
    out["block_group_geoid"] = out["block_group_geoid"].astype("string").str.zfill(12)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs).to_crs("EPSG:5070")


def _surface_pair(candidate_bg: Path) -> pd.DataFrame:
    v8 = _read_surface(PROMOTED_BG, FIELD).rename(columns={FIELD: f"{FIELD}_v8"})
    v9 = _read_surface(candidate_bg, FIELD).rename(columns={FIELD: f"{FIELD}_v9"})
    return v8.merge(v9[["block_group_geoid", f"{FIELD}_v9"]], on="block_group_geoid", how="inner")


def _plot_pair(
    gdf: gpd.GeoDataFrame,
    *,
    title: str,
    out_path: Path,
    simplify_tolerance: float,
) -> dict[str, object]:
    plot_gdf = gdf.copy()
    if simplify_tolerance > 0:
        plot_gdf["geometry"] = plot_gdf.geometry.simplify(simplify_tolerance, preserve_topology=False)

    values = pd.concat(
        [
            pd.to_numeric(plot_gdf[f"{FIELD}_v8"], errors="coerce"),
            pd.to_numeric(plot_gdf[f"{FIELD}_v9"], errors="coerce"),
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan)
    positive = values[values.gt(0)]
    if positive.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanquantile(positive, 0.02))
        vmax = float(np.nanquantile(positive, 0.98))
        if not np.isfinite(vmin) or vmin < 0:
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(positive.max())
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = Normalize(vmin=np.log1p(vmin), vmax=np.log1p(vmax))
    plot_gdf["_v8_plot"] = np.log1p(pd.to_numeric(plot_gdf[f"{FIELD}_v8"], errors="coerce").clip(lower=0.0))
    plot_gdf["_v9_plot"] = np.log1p(pd.to_numeric(plot_gdf[f"{FIELD}_v9"], errors="coerce").clip(lower=0.0))

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, col, subtitle in zip(axes, ["_v8_plot", "_v9_plot"], ["v8 promoted", "v9 candidate"], strict=True):
        plot_gdf.plot(
            column=col,
            ax=ax,
            cmap="magma",
            norm=norm,
            linewidth=0,
            missing_kwds={"color": "#eeeeee"},
            rasterized=True,
        )
        ax.set_title(subtitle, fontsize=12)
        ax.set_axis_off()
    fig.suptitle(f"{title}\nBurglary primary index; shared log color scale, clipped p02-p98", fontsize=14)

    sm = plt.cm.ScalarMappable(norm=norm, cmap="magma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035, pad=0.02)
    tick_values = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(np.log1p(tick_values))
    cbar.set_ticklabels([f"{v:.0f}" for v in tick_values])
    cbar.set_label("Burglary primary index")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    image = mpimg.imread(out_path)
    return {
        "path": str(out_path.relative_to(REPO_ROOT)),
        "rows": int(len(plot_gdf)),
        "field": FIELD,
        "vmin_p02": vmin,
        "vmax_p98": vmax,
        "image_shape": list(image.shape),
        "pixel_std": float(np.nanstd(image)),
        "nonblank": bool(np.nanstd(image) > 0.0),
    }


def render(candidate_run: str) -> dict[str, object]:
    candidate_dir = REPO_ROOT / "state" / "candidates" / candidate_run
    candidate_bg = candidate_dir / "crimerisk_block_group_2024_ags_core.parquet"
    out_dir = candidate_dir / "burglary_premises_v9_evidence" / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair = _surface_pair(candidate_bg)
    states = sorted(pair["state_fips"].dropna().astype("string").str.zfill(2).unique())
    geometry = _read_geometry(states)
    gdf = geometry.merge(pair, on="block_group_geoid", how="inner")
    national = gdf[~gdf["state_fips"].isin(["02", "15", "72"])].copy()
    la_metro = gdf[gdf["block_group_geoid"].astype("string").str.slice(0, 5).isin(LA_METRO_COUNTIES)].copy()

    records = [
        _plot_pair(
            national,
            title="National v8 promoted vs burglary-premises-v9",
            out_path=out_dir / "national_burglary_primary_index_v8_vs_v9.png",
            simplify_tolerance=650.0,
        ),
        _plot_pair(
            la_metro,
            title="Los Angeles metro counties v8 promoted vs burglary-premises-v9",
            out_path=out_dir / "la_metro_burglary_primary_index_v8_vs_v9.png",
            simplify_tolerance=80.0,
        ),
    ]
    summary = {
        "candidate_run": candidate_run,
        "baseline": str(PROMOTED_BG.relative_to(REPO_ROOT)),
        "la_metro_counties": sorted(LA_METRO_COUNTIES),
        "renders": records,
    }
    (out_dir / "render_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all(record["nonblank"] for record in records):
        raise SystemExit("at least one burglary-premises-v9 render appears blank")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    args = parser.parse_args()
    print(json.dumps(render(args.candidate_run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
