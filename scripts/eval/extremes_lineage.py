"""Extremes lineage: the top cells per offense and measure, explained back to their sources.

    uv run python scripts/eval/extremes_lineage.py --candidate-dir state/candidates/<run> [--top 25]

For each offense and measure (crime exposure `primary`, `resident`) the block groups with the
largest published index, and for each: the components that carry its expected count (which
jurisdiction layer, which agency, what share of the cell), that agency-jurisdiction's filed,
accounting and final control and its repair / smoothing path, and the denominator the index
divides by. Writes `extremes_lineage_<year>.csv` (one row per cell and contributing component)
and `extremes_lineage_<year>.md` (one line per cell) into the candidate directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crimerisk.crime import OFFENSES_7  # noqa: E402

MEASURES = {"primary": ("index_{o}_primary", "primary_denominator_{o}"), "resident": ("index_{o}_resident", "resident_secondary_denominator")}
COMPONENT_COLUMNS = [
    "bg_id", "offense", "jurisdiction_id", "jurisdiction_type", "canonical_target_ori", "service_scope_id",
    "source_target_count", "component_count_after", "model_share", "city_incident_posterior_active",
    "city_posterior_prior_fraction", "allocation_envelope_mode",
]


def top_cells(bg: pd.DataFrame, *, top: int) -> pd.DataFrame:
    rows = []
    for offense in OFFENSES_7:
        for measure, (index_col, denominator_col) in MEASURES.items():
            index_col = index_col.format(o=offense)
            denominator_col = denominator_col.format(o=offense)
            if index_col not in bg.columns:
                continue
            values = pd.to_numeric(bg[index_col], errors="coerce")
            if not values.notna().any():
                continue  # not published at block group (murder and rape are tract-only)
            picked = bg.loc[values.nlargest(top).index]
            for rank, (_, row) in enumerate(picked.iterrows(), start=1):
                rows.append(
                    {
                        "offense": offense, "measure": measure, "rank": rank,
                        "block_group_geoid": row["block_group_geoid"], "index": float(row[index_col]),
                        "expected_count": float(row[f"expected_count_{offense}"]),
                        "denominator": float(pd.to_numeric(row.get(denominator_col), errors="coerce")),
                        "population": float(row.get("population_2025", float("nan"))),
                        "special_use_type": row.get("special_use_type"),
                        "eb_jurisdiction_id": row.get("eb_jurisdiction_id"),
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)
    cand = args.candidate_dir if args.candidate_dir.is_absolute() else ROOT / args.candidate_dir
    year = int(args.year)
    columns = ["block_group_geoid", "population_2025", "special_use_type", "eb_jurisdiction_id", "resident_secondary_denominator"]
    for offense in OFFENSES_7:
        columns += [f"expected_count_{offense}", f"primary_denominator_{offense}", f"index_{offense}_primary", f"index_{offense}_resident"]
    bg = pd.read_parquet(cand / f"crimerisk_block_group_{year}_ags_core.parquet", columns=columns)
    cells = top_cells(bg, top=int(args.top))

    audit = ds.dataset(cand / f"allocation_component_denominator_audit_{year}.parquet")
    names = set(audit.schema.names)
    wanted = [c for c in COMPONENT_COLUMNS if c in names]
    ids = sorted(set(cells["block_group_geoid"].astype(str)))
    components = audit.to_table(columns=wanted, filter=ds.field("bg_id").isin(ids)).to_pandas()
    components["bg_id"] = components["bg_id"].astype(str)
    components = components.merge(
        cells[["offense", "block_group_geoid"]].drop_duplicates().rename(columns={"block_group_geoid": "bg_id"}),
        on=["bg_id", "offense"],
    )
    components["component_count_after"] = pd.to_numeric(components["component_count_after"], errors="coerce").fillna(0.0)
    totals = components.groupby(["bg_id", "offense"])["component_count_after"].transform("sum")
    components["cell_share"] = components["component_count_after"] / totals.where(totals.gt(0))

    # The control behind each component: filed, accounting and final count, and the path.
    controls_dir = ROOT / "state" / "controls"
    accounting = pd.read_parquet(controls_dir / f"jurisdiction_controls_{year}.parquet")
    keep = [c for c in ("jurisdiction_id", "offense", "jurisdiction_name", "reported_count_preferred", "adjusted_count_ags_core",
                        "level_repair_mode", "level_admission_reason") if c in accounting.columns]
    accounting = accounting[keep].rename(columns={"adjusted_count_ags_core": "accounting_count"})
    smoothed = pd.read_parquet(controls_dir / f"jurisdiction_controls_smoothed_{year}.parquet")[
        ["jurisdiction_id", "offense", "smoothed_count", "estimator", "fallback_reason", "clean_year_count", "state_calibration_factor"]
    ]
    components["parent_jurisdiction_id"] = components["jurisdiction_id"].astype(str).str.replace(
        r":county:\d{5}$|:residual$", "", regex=True
    )
    components = components.merge(
        accounting.rename(columns={"jurisdiction_id": "parent_jurisdiction_id"}), on=["parent_jurisdiction_id", "offense"], how="left"
    ).merge(smoothed.rename(columns={"jurisdiction_id": "parent_jurisdiction_id"}), on=["parent_jurisdiction_id", "offense"], how="left")
    estimates_path = ROOT / "state" / "cache" / "allocation" / f"agency_allocation_target_estimates_{year}.parquet"
    if estimates_path.exists():
        estimates = pd.read_parquet(estimates_path, columns=["ori9", "offense", "reported_count_current", "estimated_count", "agency_estimate_source", "level_repair_mode"])
        components = components.merge(
            estimates.rename(columns={"ori9": "canonical_target_ori", "level_repair_mode": "agency_repair_mode"}),
            on=["canonical_target_ori", "offense"], how="left",
        )
    lineage = cells.merge(components.rename(columns={"bg_id": "block_group_geoid"}), on=["block_group_geoid", "offense"], how="left")
    lineage = lineage.sort_values(["offense", "measure", "rank", "component_count_after"], ascending=[True, True, True, False])
    out_csv = cand / f"extremes_lineage_{year}.csv"
    lineage.to_csv(out_csv, index=False)

    lines = [f"# Extremes lineage {year}: top {int(args.top)} block groups per offense and measure", ""]
    for (offense, measure), group in lineage.groupby(["offense", "measure"], sort=True):
        lines += [f"## {offense}, {measure}", "", "| rank | block group | index | expected | denominator | residents | type | main component (share) | control path |", "|---|---|---|---|---|---|---|---|---|"]
        for (rank, geoid), cell in group.groupby(["rank", "block_group_geoid"], sort=True):
            first = cell.iloc[0]
            main = cell.sort_values("component_count_after", ascending=False).iloc[0]
            who = main.get("canonical_target_ori") if pd.notna(main.get("canonical_target_ori")) else main.get("jurisdiction_id")
            path = f"{main.get('estimator')}/{main.get('level_repair_mode')}" if pd.notna(main.get("estimator")) else str(main.get("agency_repair_mode"))
            lines.append(
                f"| {rank} | {geoid} | {first['index']:.0f} | {first['expected_count']:.1f} | {first['denominator']:.0f} | {first['population']:.0f} | "
                f"{first['special_use_type']} | {who} {main.get('jurisdiction_type')} ({float(main.get('cell_share') or 0):.2f}) | {path} |"
            )
        lines.append("")
    (cand / f"extremes_lineage_{year}.md").write_text("\n".join(lines))
    print(f"wrote {out_csv} ({len(cells)} cells, {len(lineage)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
