"""
01 - Extract the slim per-block-group index table from the pipeline parquet.

Reads the published BLOCK GROUP output parquet and writes a compact CSV holding
ONLY the columns the frontend needs: block-group GEOID, the published aggregate
indexes, the per-offense primary/resident points, plus a little context
(population, expected counts, reliability tier) for the tooltip.

This is the first step of freezing a snapshot: we capture the index values here,
then 03 bakes them into the geometry. The frontend never reads the live parquet.

Schema note: the published output uses explicit field families. There is NO bare
`index_{offense}` / `rate_{offense}` / `count_{offense}` / `index_total`. The
default total-crime map layer is the exposure-denominated
`index_total_primary_event_weighted` (the readable risk surface);
`index_total_part1_resident` stays selectable as an option.

The schema also carries `crime_density_{offense}` / `crime_density_total`
(incidents per square mile — the denominator-free hotspot view), `estimate_mode_
{offense}` (count_derived / non_residential / special_use / vehicle_denominator_
invalid / insufficient_exposure), and `special_use_tract_flag`. Special-use /
suppressed cells have NULL per-capita rate/index by design but a VALID expected
count and crime density.

Run:  uv run python frontend/build/01_extract_indices.py
"""

from pathlib import Path

import pandas as pd

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# SOURCE PARQUET + DATA YEAR. SINGLE point of control for the whole snapshot
# chain (01/03/04/05): frontend/build/snapshot_config.env. Edit that file to
# re-point the frontend at a newer model build of the same schema.
CONFIG = {
    k.strip(): v.strip()
    for k, v in (
        line.split("=", 1)
        for line in (Path(__file__).resolve().parent / "snapshot_config.env").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
}
YEAR = int(CONFIG["CRIMERISK_SNAPSHOT_YEAR"])
SRC = REPO / CONFIG["CRIMERISK_SNAPSHOT_SRC"]
POPULATION_COL = f"population_{YEAR}"
# ---------------------------------------------------------------------------

# Geography key: 12-char zero-padded block-group GEOID.
GEO_ID = "block_group_geoid"
GEO_ID_LEN = 12

if not SRC.exists():
    raise FileNotFoundError(f"Source block-group parquet not found: {SRC}")

OUT = REPO / "frontend/tmp/bg_indices_slim.csv"

OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]

# Six explicit published aggregate indexes. index_total_part1_resident is the
# frontend default total-crime layer.
AGG_INDEX_COLS = [
    "index_total_part1_resident",
    "index_personal_part1_resident",
    "index_property_part1_resident",
    "index_total_primary_event_weighted",
    "index_total_equal_offense",
    "index_total_harm",
]

# Per-offense point fields: primary index (default point), resident index,
# primary rate (per 100k, tooltip), expected count (tooltip).
OFFENSE_INDEX_PRIMARY = [f"index_{o}_primary" for o in OFFENSES]
OFFENSE_INDEX_RESIDENT = [f"index_{o}_resident" for o in OFFENSES]
OFFENSE_RATE_PRIMARY = [f"rate_{o}_primary" for o in OFFENSES]
OFFENSE_EXPECTED_COUNT = [f"expected_count_{o}" for o in OFFENSES]
OFFENSE_RELIABILITY = [f"reliability_tier_{o}" for o in OFFENSES]
# Effective numerator support per offense (incident-equivalent count anchoring the
# estimate). ~0 support on `low`-reliability cells is the count-noise signature;
# high support on `high` cells marks a genuine hotspot. A choropleth opacity
# de-emphasis for low-reliability cells was tried and deliberately stubbed out
# (2026-07-01, `230f307`): ~98% of all cells are reliability-low nationally
# (murder/rape/robbery all clear the support test almost nowhere), so muting by
# tier muted the entire national map rather than flagging genuine outliers.
# Per-cell honesty instead lives in the popup: the reliability tier dot, the
# support count, and an "interpret block-level differences cautiously" advisory.
OFFENSE_SUPPORT = [f"effective_numerator_support_{o}" for o in OFFENSES]

# Density (incidents per square mile) — denominator-free hotspot view. Valid even
# in special-use cells; NULL only for water-only tracts. Includes the total.
OFFENSE_DENSITY = [f"crime_density_{o}" for o in OFFENSES]
DENSITY_TOTAL = ["crime_density_total"]
DENSITY_COLS = OFFENSE_DENSITY + DENSITY_TOTAL

# Estimate mode per offense. These strings are passed through to 03, where they
# are encoded for tiles. Fail closed here if the backend introduces a new mode.
OFFENSE_ESTIMATE_MODE = [f"estimate_mode_{o}" for o in OFFENSES]
ESTIMATE_MODE_VALUES = {
    "count_derived",
    "non_residential",
    "special_use",
    "vehicle_denominator_invalid",
    "insufficient_exposure",
}

# All numeric index columns (for stats + rounding).
INDEX_COLS = AGG_INDEX_COLS + OFFENSE_INDEX_PRIMARY + OFFENSE_INDEX_RESIDENT
RATE_COLS = OFFENSE_RATE_PRIMARY
COUNT_COLS = OFFENSE_EXPECTED_COUNT

KEEP = (
    [GEO_ID, "state_fips", POPULATION_COL, "non_residential_flag",
     "special_use_tract_flag"]
    + INDEX_COLS
    + RATE_COLS
    + COUNT_COLS
    + DENSITY_COLS
    + OFFENSE_RELIABILITY
    + OFFENSE_SUPPORT
    + OFFENSE_ESTIMATE_MODE
)


def main() -> None:
    print(f"Reading {SRC.name} ...")
    df = pd.read_parquet(SRC, columns=KEEP)
    print(f"  {len(df):,} block groups, {len(KEEP)} columns")

    # block_group_geoid must be a 12-char zero-padded string GEOID for the join.
    df[GEO_ID] = df[GEO_ID].astype("string").str.zfill(GEO_ID_LEN)
    assert df[GEO_ID].str.len().eq(GEO_ID_LEN).all(), "non-12-char BG GEOIDs found"
    assert df[GEO_ID].is_unique, "duplicate block-group GEOIDs"

    for mode_col, index_col in zip(OFFENSE_ESTIMATE_MODE, OFFENSE_INDEX_PRIMARY):
        modes = set(df[mode_col].dropna().astype(str).unique())
        unknown = sorted(modes - ESTIMATE_MODE_VALUES)
        if unknown:
            raise ValueError(f"Unknown estimate modes in {mode_col}: {unknown}")
        missing = int(df[mode_col].isna().sum())
        if missing:
            raise ValueError(f"Missing estimate modes in {mode_col}: {missing:,}")
        suppressed = df[mode_col].ne("count_derived")
        bad = suppressed & df[index_col].notna()
        if bad.any():
            raise ValueError(
                f"{mode_col} has {int(bad.sum()):,} suppressed rows with non-null {index_col}"
            )

    # Round to keep the artifact compact; indices to 1dp, rates to 1dp, counts to
    # 2dp (expected counts are fractional), density to 2dp (incidents/sq mi). NaN
    # is preserved -> JSON null so the frontend renders suppressed / non-
    # residential per-capita cells, and water-only density cells, as "no data".
    for c in INDEX_COLS + RATE_COLS:
        df[c] = df[c].astype("float64").round(1)
    for c in COUNT_COLS + DENSITY_COLS + OFFENSE_SUPPORT:
        df[c] = df[c].astype("float64").round(2)
    df[POPULATION_COL] = df[POPULATION_COL].astype("Int64")
    # Special-use flag -> compact 0/1 int (NaN cells treated as not special-use).
    df["special_use_tract_flag"] = (
        df["special_use_tract_flag"].fillna(False).astype(bool).astype("int8")
    )

    df = df.sort_values(GEO_ID).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size/1e6:.1f} MB)")

    # Report null coverage per index so the frontend "no data" handling is justified.
    print("\nNull coverage per index (rendered as no-data):")
    for c in INDEX_COLS:
        n = df[c].isna().sum()
        print(f"  {c:38s} {n:6,} ({n/len(df)*100:4.1f}%)")
    print("\nNull coverage per density (water-only -> no-data):")
    for c in DENSITY_COLS:
        n = df[c].isna().sum()
        print(f"  {c:38s} {n:6,} ({n/len(df)*100:4.1f}%)")
    print(
        f"\nspecial_use_tract_flag set on {int(df['special_use_tract_flag'].sum()):,} block groups"
    )

    # Stash distribution stats for the color-scale breakpoints used by the UI.
    #
    # Index layers are heavily right-skewed: most tracts are well below the national
    # average (for robbery ~71% sit under 100 and ~56% under 50), while a thin urban
    # tail and a handful of tiny-population cells run to the thousands. A LINEAR ramp
    # — even one clipped at a cap — buries that whole low-end majority in the bottom
    # sliver of the colorway, so rural/suburban tracts all read as one flat near-empty
    # color while a few outliers dominate. (This is exactly the "concentrated offenses
    # look empty" complaint.)
    #
    # The fix is a ROBUST QUANTILE scale: equal-count color bands. We emit the value
    # at each population percentile (`q`, every 10th from p10..p90, plus a finer top
    # for the urban tail). The frontend interpolates the colorway across these so each
    # band holds ~the same number of tracts — the full spatial gradient becomes
    # visible everywhere, and extreme outliers fall in the top band rather than
    # stretching the scale. The band whose range straddles 100 is the neutral mid
    # color, preserving the "above/below national average" reading. The legend renders
    # the top band honestly as ">= <p_top>". p50/p98/p99/max retained for reference.
    #
    # Percentiles used for the equal-count bands. Denser near the top so the urban
    # hotspot tail keeps resolution instead of collapsing into one band.
    Q_PCTS = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]
    stats = {}
    for c in INDEX_COLS:
        s = df[c].dropna()
        # Equal-count quantile breakpoints. De-duplicate (a degenerate distribution
        # with many zeros can repeat a stop) and keep strictly increasing so the
        # frontend interpolation domain is valid.
        raw = [float(s.quantile(p / 100.0)) for p in Q_PCTS]
        q = []
        for v in raw:
            if not q or v > q[-1]:
                q.append(round(v, 1))
        stats[c] = {
            "kind": "index",
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p50": float(s.quantile(0.50)),
            "p98": float(s.quantile(0.98)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "n_valid": int(s.size),
            # Robust equal-count color-band breakpoints (strictly increasing) and the
            # percentiles they correspond to. The top stop is the ">= top band" anchor.
            "q": q,
            "q_pcts": Q_PCTS[: len(q)],
        }
    # Density layers are SEQUENTIAL and heavily right-skewed (incidents/sq mi:
    # p50~30, p99~2000, max~30000 for total). A linear scale would paint the whole
    # country one color. Emit LOG-DOMAIN quantile stops (on positive values) so the
    # frontend can interpolate in log space and surface genuine hotspots.
    for c in DENSITY_COLS:
        s = df[c].dropna()
        pos = s[s > 0]
        stats[c] = {
            "kind": "density",
            "min": float(s.min()),
            "p10": float(pos.quantile(0.10)),
            "p50": float(pos.quantile(0.50)),
            "p90": float(pos.quantile(0.90)),
            "p99": float(pos.quantile(0.99)),
            "max": float(s.max()),
            "n_valid": int(s.size),
            "n_positive": int(pos.size),
        }
    import json

    stats_path = REPO / "frontend/tmp/index_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote stats -> {stats_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
