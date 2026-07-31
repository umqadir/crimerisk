"""
01 - Extract the slim per-block-group index table from the pipeline parquet.

Reads the published BLOCK GROUP output parquet and writes a compact CSV holding
ONLY the columns the frontend needs: block-group GEOID, the published aggregate
indexes, the per-offense primary/resident points, plus a little context
(population, expected counts, reliability tier) for the tooltip.

This is the first step of freezing a snapshot: we capture the index values here,
then 03 bakes them into the geometry. The frontend never reads the live parquet.

Murder and rape publish at CENSUS-TRACT support: a single year of data is too
sparse to support a block-group rate for these two offenses, so the block-group
parquet carries NULL for their per-offense index/rate and the tract parquet
carries them. This step therefore also reads the tract parquet and emits a second
slim CSV (`tract_indices_slim.csv`) for the murder/rape map layers, and bakes the
parent tract's murder/rape index/rate onto each block-group row so the block-group
popup can show a tract-scale value for those two offenses. Expected counts and
reliability metadata for murder/rape remain at block-group support.

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

Density carries its OWN field precision (`DENSITY_DP`), not the shared 2dp of the
fractional counts: at 2dp the smallest representable positive density is 0.01/sq
mi, which flattened 37% of CONUS land into a single colour. See the DENSITY_DP
note below — the precision, 03's re-round, and the viewer's paint floor are one
coupled decision, published through the `_meta` block of index_stats.json.

The stats block also emits `pct_above` per layer: the share of PEOPLE and of LAND
above every legend break/stop, which is what turns the legend from a land readout
into a population one (audit finding F-01).

Two further per-cell fields exist for the DISPLAY only and carry no published
value of their own:

  * `provenance_class_{offense}` — a 2-bit disclosure code (see PROVENANCE_*
    below) marking the two documented classes where the published number does not
    rest on an ordinary allocation of an agency's own reported counts. The viewer
    hatches these cells and names the class in the popup.
  * `population_density_2024` — residents per square mile, the input to the
    viewer's national-zoom population-aware emphasis (empty polygons fade toward
    the basemap below metro zoom). Emphasis only; no published value changes.

Run:  uv run python frontend/build/01_extract_indices.py
"""

from pathlib import Path

import numpy as np
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
# Murder/rape are frozen from the tract parquet (same run as SRC). OUT_DIR is the
# published-artifacts directory (unused here; 04/05 write there).
TRACT_SRC = REPO / CONFIG["CRIMERISK_SNAPSHOT_TRACT_SRC"]
# The allocation component audit from the SAME run: the only artifact that carries
# which jurisdiction each cell's counts actually came from, which is the peer group
# the model-only outlier class is defined against (see PROVENANCE_* below).
COMPONENT_AUDIT_SRC = REPO / CONFIG["CRIMERISK_SNAPSHOT_COMPONENT_AUDIT"]
OUT_DIR = CONFIG["CRIMERISK_SNAPSHOT_OUT"]
POPULATION_COL = f"population_{YEAR}"
# ---------------------------------------------------------------------------

# Geography key: 12-char zero-padded block-group GEOID.
GEO_ID = "block_group_geoid"
GEO_ID_LEN = 12

# Tract geography key: 11-char zero-padded census-tract GEOID (the parent tract of
# a block group is its GEOID's first 11 chars).
TRACT_ID = "tract_id"
TRACT_ID_LEN = 11

if not SRC.exists():
    raise FileNotFoundError(f"Source block-group parquet not found: {SRC}")
if not TRACT_SRC.exists():
    raise FileNotFoundError(f"Source tract parquet not found: {TRACT_SRC}")
if not COMPONENT_AUDIT_SRC.exists():
    raise FileNotFoundError(
        f"Allocation component audit not found: {COMPONENT_AUDIT_SRC}. It ships with "
        "the same release run as the published parquets and carries the jurisdiction "
        "identity the model-only outlier class is defined against."
    )

OUT = REPO / "frontend/tmp/bg_indices_slim.csv"
TRACT_OUT = REPO / "frontend/tmp/tract_indices_slim.csv"

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

# --- Density field precision (audit finding R-01) -----------------------------
# Density used to share the 2dp rounding loop with the fractional counts. At 2dp
# the smallest positive density that survives is 0.01/sq mi, which made the
# AREA-WEIGHTED p10 exactly 0.01 on all ten density layers and painted one flat
# bottom colour over 37.0% of CONUS land on `d_tot` (86.8-97.5% on the sparse
# offence layers) — of which only 0.20 points is genuinely zero-crime land: 20.77
# points carries a real positive density that rounded to 0.00 and 16.06 points is
# pinned at the single value 0.01. Inside that flat band the true `d_tot` spans
# 2.82 orders of magnitude. Measured in state/qa/stage6_screen/06_density_
# quantization.md.
#
# 4dp is the audited landing point: `d_tot`'s flat bottom band drops 37.03% ->
# 10.35% (the ~10% the p10 anchor is DEFINED to select), and the count of
# distinct renderable levels goes 53,540 -> 191,923 of 238,193 rows, i.e. past
# 4dp almost every block group is already distinct and further digits buy only
# bytes. Sparse offences stay honestly bottom-heavy (`d_rob` 64.5% at 4dp,
# because ~90% of CONUS land has genuinely zero robbery density).
#
# THREE PLACES ARE COUPLED and none of them works alone:
#   1. this constant                       — how many digits reach the CSV/tiles
#   2. 03_join_geometry.py                 — re-rounds the same columns before
#                                            tiling; it reads DENSITY_DP back out
#                                            of index_stats.json `_meta` so the
#                                            two can never drift
#   3. the viewer's density paint floor    — ["max", value, FLOOR] and
#                                            densityStops()'s Math.max(v, FLOOR);
#                                            stamped into snapshot.json as
#                                            `density_floor` by 05 from the
#                                            `_meta` block below, so the viewer
#                                            no longer hardcodes it
# The stats block below runs on the ROUNDED column, so changing this constant
# also re-anchors p10/p50/p90/p99/p999 for all ten density layers (that is the
# intent — the ramp follows the data it can actually represent).
DENSITY_DP = 4
DENSITY_FLOOR = 10.0 ** (-DENSITY_DP)  # smallest positive value that survives

# Derived rollup densities for the Personal/Property crime types. Density is
# additive over the same land area, so each rollup is the exact row-wise sum of
# its member offense densities. No column is baked into the tiles: the viewer
# renders these as a sum expression over the per-offense keys and needs only the
# color-scale stats emitted below.
DERIVED_DENSITY = {
    "crime_density_personal": [
        f"crime_density_{o}" for o in ("murder", "rape", "robbery", "aggravated_assault")
    ],
    "crime_density_property": [
        f"crime_density_{o}" for o in ("burglary", "larceny", "motor_vehicle_theft")
    ],
}

# --- Per-cell PROVENANCE DISCLOSURE class (display only) ----------------------
# Until now every per-cell honesty signal lived in the popup, and the map itself
# showed a uniform surface: a benchmark-imputed county and a directly-reported one
# painted identically, so a reader who never opened a popup never learned which
# was which. This bakes the two DOCUMENTED classes into the tiles so the map can
# mark them, at every zoom, for the active layer.
#
# WHY NOT THE RELIABILITY TIER. The obvious candidate — de-emphasize
# reliability-low cells — was tried and stubbed out on 2026-07-01 (`230f307`) and
# the reason still binds: on this surface 98.5% of block groups are
# reliability-low on robbery (234,615 / 238,193), 97.7% on burglary, 95.3% on
# larceny. Marking that class marks the whole country, which communicates nothing
# and dims the map. The tier stays where it works: the popup's support row.
#
# So the marked set is deliberately NARROWER — the two classes the project has
# already written down as its named caveats, each of which is a statement about
# where the NUMBER CAME FROM rather than about its noise level:
#
#   bit 0 (code 1) BENCHMARK-IMPUTED. The agency covering this cell filed nothing
#     usable for this offense, so its counts were imputed under the state/offense
#     benchmark identity (the v20 Class A design). `benchmark_imputed_share_{o}`
#     is the share of the cell's expected count that came from that imputation.
#     The threshold is NOT a new knob: 0.5 is the share above which the pipeline
#     itself already forces the confidence tier low (see the v20 release section
#     in STATE.md), so the map marks exactly the cells the backend already
#     declares low-confidence for this reason. 2.2% of block groups, 6.6% of land.
#
#   bit 1 (code 2) MODEL-ONLY OUTLIER. The stage-4 audit's `d1` predicate, the
#     class STATE.md and METHODOLOGY name as "the 3.77% model-only outlier class
#     [that] stays UNWIRED permanently (honesty over tidiness)" — cells with NO
#     direct incident evidence whose modeled value nonetheless lands more than 5x
#     its own jurisdiction's median. Leaving it unwired was a decision about the
#     VALUE (do not suppress it); it was never a decision to leave it invisible,
#     and the ruling explicitly says the disclosure is metadata + popup. This adds
#     the map to that list. 6.6% of block groups, 5.0% of land.
#
# A cell can carry both (code 3). Union: 9.1% of block groups, 12.7% of land —
# visible as a texture where it clusters (eastern Kentucky, the Plains) and absent
# over most metros, which is the point.
PROVENANCE_IMPUTED = 1
PROVENANCE_MODEL_ONLY_OUTLIER = 2
# Share of a cell's expected count that must be benchmark-imputed before the cell
# is marked. Mirrors the backend's own confidence-forcing threshold; do not tune.
BENCHMARK_IMPUTED_SHARE_MIN = 0.5
# The stage-4 d1 predicate, verbatim: index more than this multiple of the primary
# jurisdiction's median index for the same offense, on a cell that is model-only,
# reliability-low and carries no effective incident support.
MODEL_ONLY_OUTLIER_RATIO = 5.0
MODEL_ONLY_SUPPORT_SOURCE = "model_only"
MODEL_ONLY_RELIABILITY_TIER = "low"

OFFENSE_IMPUTED_SHARE = [f"benchmark_imputed_share_{o}" for o in OFFENSES]
OFFENSE_SUPPORT_SOURCE = [f"numerator_support_source_{o}" for o in OFFENSES]
PROVENANCE_CLASS = [f"provenance_class_{o}" for o in OFFENSES]

# Land area per unit. Used three ways: the WEIGHT for the density color-scale
# percentiles, the denominator of the `pct_above` land share, and (new) the
# denominator of the population density below. Not emitted to the slim CSV itself.
AREA_COL_NAME = "land_area_sq_mi"

# Residents per square mile — the input to the viewer's national-zoom
# population-aware emphasis (item 3). Emphasis only: no published value, popup
# reading or legend break depends on it.
#
# The two anchors of the emphasis ramp are declared here (and stamped through
# `_meta` into snapshot.json, so the viewer reads them rather than carrying its
# own copy) because they are DATA facts, not styling taste, and because both are
# borrowed rather than fitted:
#   * POP_DENSITY_HIGH = 1,000/sq mi is the Census Bureau's urban-area CORE
#     density criterion — at or above it a cell holds full emphasis.
#   * POP_DENSITY_LOW  = 6/sq mi is the federal "frontier" density used in health
#     and rural policy — at or below it a cell fades to the floor.
# The scale between them is logarithmic, matching every other ramp in this build.
# For reference on this surface: 48.6% of CONUS land is at or below 6/sq mi and
# only 2.3% is at or above 1,000/sq mi — but that 2.3% of land holds 69.8% of the
# people, which is exactly the asymmetry the national view was failing to show.
POP_DENSITY_COL = f"population_density_{YEAR}"
POP_DENSITY_LOW = 6.0
POP_DENSITY_HIGH = 1000.0
# Opacity multiplier at (and below) POP_DENSITY_LOW at national zoom. Tuned
# against the national capture: the emptiest land has to recede toward the
# basemap without disappearing, since a faded cell still has to be clickable and
# still has to show its provenance hatch.
POP_DENSITY_MIN_EMPHASIS = 0.22

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
     "special_use_tract_flag", AREA_COL_NAME]
    + INDEX_COLS
    + RATE_COLS
    + COUNT_COLS
    + DENSITY_COLS
    + OFFENSE_RELIABILITY
    + OFFENSE_SUPPORT
    + OFFENSE_ESTIMATE_MODE
    + OFFENSE_IMPUTED_SHARE
    + OFFENSE_SUPPORT_SOURCE
)

# --- Tract-support offenses (murder, rape) -----------------------------------
# Murder and rape publish at census-tract support. In the block-group parquet the
# per-offense index/rate for these offenses is NULL by policy; a populated value
# means the source parquet predates it, so we fail closed on these columns.
RARE_OFFENSES = ["murder", "rape"]
BG_RARE_NULL_COLS = (
    [f"index_{o}_primary" for o in RARE_OFFENSES]
    + [f"index_{o}_resident" for o in RARE_OFFENSES]
    + [f"rate_{o}_primary" for o in RARE_OFFENSES]
)

# The parent tract's murder/rape primary index + rate, baked onto each block-group
# row so the block-group popup can show a tract-scale value for these offenses.
BG_TRACT_BAKE = {
    "index_murder_primary": "index_murder_primary_tract",
    "index_rape_primary": "index_rape_primary_tract",
    "rate_murder_primary": "rate_murder_primary_tract",
    "rate_rape_primary": "rate_rape_primary_tract",
}
BG_TRACT_BAKE_COLS = list(BG_TRACT_BAKE.values())

# Slim tract frame for the murder/rape map layers. Same field families as the
# block-group frame, restricted to the two tract-support offenses.
TRACT_INDEX_PRIMARY = [f"index_{o}_primary" for o in RARE_OFFENSES]
TRACT_INDEX_RESIDENT = [f"index_{o}_resident" for o in RARE_OFFENSES]
TRACT_RATE_PRIMARY = [f"rate_{o}_primary" for o in RARE_OFFENSES]
TRACT_EXPECTED_COUNT = [f"expected_count_{o}" for o in RARE_OFFENSES]
TRACT_RELIABILITY = [f"reliability_tier_{o}" for o in RARE_OFFENSES]
TRACT_SUPPORT = [f"effective_numerator_support_{o}" for o in RARE_OFFENSES]
TRACT_ESTIMATE_MODE = [f"estimate_mode_{o}" for o in RARE_OFFENSES]
TRACT_IMPUTED_SHARE = [f"benchmark_imputed_share_{o}" for o in RARE_OFFENSES]
TRACT_SUPPORT_SOURCE = [f"numerator_support_source_{o}" for o in RARE_OFFENSES]
TRACT_PROVENANCE_CLASS = [f"provenance_class_{o}" for o in RARE_OFFENSES]
TRACT_INDEX_COLS = TRACT_INDEX_PRIMARY + TRACT_INDEX_RESIDENT
TRACT_RATE_COLS = TRACT_RATE_PRIMARY
TRACT_COUNT_COLS = TRACT_EXPECTED_COUNT
TRACT_KEEP = (
    [TRACT_ID, "state_fips", POPULATION_COL, "special_use_tract_flag", AREA_COL_NAME]
    + TRACT_INDEX_COLS
    + TRACT_RATE_COLS
    + TRACT_COUNT_COLS
    + TRACT_RELIABILITY
    + TRACT_SUPPORT
    + TRACT_ESTIMATE_MODE
    + TRACT_IMPUTED_SHARE
    + TRACT_SUPPORT_SOURCE
)
# The four rare index columns whose color-scale stats + null coverage come from
# the tract distribution (the block-group columns are all-null by policy).
RARE_INDEX_FROM_TRACT = set(TRACT_INDEX_COLS)

# Land area per unit (see AREA_COL_NAME above, where it is declared alongside the
# population-density field it now also denominates). Not emitted to the slim CSV.
AREA_COL = AREA_COL_NAME

# --- "% of people above this line" legend readout (audit finding F-01) ---------
# Area-weighted pixel sampling misled reviewers repeatedly: at index 100 the
# default layer is above the national average on 11.3% of the LAND but for 34.2%
# of the PEOPLE, and on `d_tot`'s p90 stop the gap is 7.9% of land vs 84.9% of
# people. The eye reads land; the question a reader is actually asking is about
# people. So every legend break/stop ships with both shares, computed here (01
# already holds population and land area for every row in the same pass) and
# stamped through 05 into snapshot.json. No tile dependency.
#
# The fixed index break set is duplicated from the viewer's INDEX_BREAKS. It is
# stamped into the snapshot as `index_breaks` and the viewer FAILS CLOSED on a
# mismatch (it drops the readout rather than label a break it does not paint), so
# the duplication cannot silently mislabel the map.
PCT_ABOVE_INDEX_BREAKS = [12.5, 25, 50, 75, 100, 133, 200, 400, 800]
# Which stat keys the five density stops come from, in ramp order.
DENSITY_STOP_KEYS = ["p10", "p50", "p90", "p99", "p999"]


def weighted_quantile(values, weights, q):
    """Quantile of `values` weighting each observation by `weights`.

    Standard weighted-ECDF form: sort by value, evaluate the CDF at the midpoint
    of each observation's weight mass, and interpolate. With uniform weights this
    reproduces the ordinary (linear-interpolation) quantile.
    """
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cw, v))


def primary_jurisdictions() -> dict:
    """Mass-dominant allocating jurisdiction per block group and per tract.

    The model-only outlier class is defined against the median index of the cell's
    OWN jurisdiction, and "own jurisdiction" is not a published column: a block
    group can receive counts from several agencies, so the audit's definition is
    the jurisdiction that contributed the most mass. That is only recoverable from
    the run's allocation component audit, which is why this step reads it. Same
    derivation as `state/qa/stage4_screen/screen_stage4.py` (screen `d`), so the
    class the map marks is the class the audit reported and the docs describe.
    """
    print(f"Reading {COMPONENT_AUDIT_SRC.name} (jurisdiction identity) ...")
    comp = pd.read_parquet(
        COMPONENT_AUDIT_SRC,
        columns=["bg_id", "tract_id", "jurisdiction_id", "component_count_after"],
    )
    comp["cc"] = pd.to_numeric(comp["component_count_after"], errors="coerce").fillna(0.0)
    print(f"  {len(comp):,} component rows")
    out = {}
    for level, key in (("bg", "bg_id"), ("tract", "tract_id")):
        g = comp.groupby([key, "jurisdiction_id"], dropna=False)["cc"].sum().reset_index()
        prim = g.loc[g.groupby(key)["cc"].idxmax()]
        out[level] = prim.set_index(key)["jurisdiction_id"]
        print(f"  {level}: primary jurisdiction for {len(out[level]):,} units")
    return out


def provenance_class(frame, geo_key, juris, offenses):
    """Per-offense disclosure code for a frame (see the PROVENANCE_* block above).

    Returns a DataFrame of int8 codes keyed like `frame`, one column per offense:
    0 none, 1 benchmark-imputed, 2 model-only outlier, 3 both. The outlier bit is
    only defined where the offense publishes an index at this frame's support; on
    a frame whose index column is all-null by policy (murder/rape at block group)
    the bit is left clear here and the parent tract's bit is baked on instead.
    """
    j = frame[geo_key].map(juris)
    codes = {}
    for o in offenses:
        share = frame[f"benchmark_imputed_share_{o}"].astype("float64")
        code = np.where(share >= BENCHMARK_IMPUTED_SHARE_MIN, PROVENANCE_IMPUTED, 0)

        idx = frame[f"index_{o}_primary"].astype("float64")
        if idx.notna().any():
            med = idx.groupby(j).transform("median")
            ratio = idx / med.replace(0, np.nan)
            outlier = (
                ratio.gt(MODEL_ONLY_OUTLIER_RATIO)
                & frame[f"numerator_support_source_{o}"].eq(MODEL_ONLY_SUPPORT_SOURCE)
                & frame[f"reliability_tier_{o}"].eq(MODEL_ONLY_RELIABILITY_TIER)
                & frame[f"effective_numerator_support_{o}"].astype("float64").fillna(0.0).le(0.0)
                & idx.notna()
            )
            code = code + np.where(outlier.to_numpy(), PROVENANCE_MODEL_ONLY_OUTLIER, 0)
        codes[f"provenance_class_{o}"] = pd.Series(code, index=frame.index).astype("int8")
    return pd.DataFrame(codes, index=frame.index)


def provenance_coverage(codes, pop, area, offenses):
    """Per-offense and union coverage of the disclosure classes, for the snapshot.

    Reported as shares of units / people / land so the viewer can state the size
    of what it is marking instead of the reader having to guess.
    """
    pop = pop.astype("float64").fillna(0.0)
    area = area.astype("float64").clip(lower=0).fillna(0.0)
    tot_pop, tot_area, n = float(pop.sum()), float(area.sum()), len(codes)

    def share(mask):
        m = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
        return {
            "units": int(m.sum()),
            "pct_units": round(float(m.sum()) / n * 100, 2),
            "pct_population": round(float(pop[m].sum()) / tot_pop * 100, 2) if tot_pop else None,
            "pct_land": round(float(area[m].sum()) / tot_area * 100, 2) if tot_area else None,
        }

    per_offense = {}
    for o in offenses:
        c = codes[f"provenance_class_{o}"]
        per_offense[o] = {
            "imputed": share((c & PROVENANCE_IMPUTED) > 0),
            "model_only_outlier": share((c & PROVENANCE_MODEL_ONLY_OUTLIER) > 0),
        }
    any_code = (codes[[f"provenance_class_{o}" for o in offenses]] > 0).any(axis=1)
    any_imp = (
        codes[[f"provenance_class_{o}" for o in offenses]] & PROVENANCE_IMPUTED
    ).gt(0).any(axis=1)
    any_out = (
        codes[[f"provenance_class_{o}" for o in offenses]] & PROVENANCE_MODEL_ONLY_OUTLIER
    ).gt(0).any(axis=1)
    return {
        "any_offense": {
            "imputed": share(any_imp),
            "model_only_outlier": share(any_out),
            "either": share(any_code),
        },
        "by_offense": per_offense,
    }


def shares_above(values, pop, area, breaks, labels=None):
    """Population and land share strictly above each break (audit finding F-01).

    Denominators are the population / land area of the units that HAVE a value on
    this layer, so the readout answers "of the people this layer covers" and never
    silently counts a no-data cell as below the line. `values`, `pop` and `area`
    must share an index (the same frame's rows).
    """
    m = values.notna()
    pop_m = pop.reindex(values.index).astype("float64").fillna(0.0)
    area_m = area.reindex(values.index).astype("float64").clip(lower=0).fillna(0.0)
    total_pop = float(pop_m[m].sum())
    total_area = float(area_m[m].sum())
    people, land = [], []
    for b in breaks:
        above = m & (values > b)
        people.append(
            round(float(pop_m[above].sum()) / total_pop * 100, 2) if total_pop > 0 else None
        )
        land.append(
            round(float(area_m[above].sum()) / total_area * 100, 2)
            if total_area > 0 and np.isfinite(total_area)
            else None
        )
    out = {
        "breaks": [round(float(b), 6) for b in breaks],
        "people": people,
        "land": land,
        "denominator_population": int(round(total_pop)),
        "denominator_land_sq_mi": round(total_area, 2),
        "n_units_with_value": int(m.sum()),
    }
    if labels:
        out["labels"] = list(labels)
    return out


def main() -> None:
    print(f"Reading {SRC.name} ...")
    df = pd.read_parquet(SRC, columns=KEEP)
    print(f"  {len(df):,} block groups, {len(KEEP)} columns")

    # block_group_geoid must be a 12-char zero-padded string GEOID for the join.
    df[GEO_ID] = df[GEO_ID].astype("string").str.zfill(GEO_ID_LEN)
    assert df[GEO_ID].str.len().eq(GEO_ID_LEN).all(), "non-12-char BG GEOIDs found"
    assert df[GEO_ID].is_unique, "duplicate block-group GEOIDs"

    # Fail closed: murder/rape publish at tract support, so their per-offense
    # index/rate must be all-null in the block-group parquet. A populated value
    # means the source parquet predates the tract-support policy.
    for c in BG_RARE_NULL_COLS:
        n = int(df[c].notna().sum())
        if n:
            raise ValueError(
                f"{c} has {n:,} non-null block-group values, but murder/rape "
                f"publish at tract support (expected all-null in the BG parquet)."
            )

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

    # --- Tract frame (murder/rape publish at census-tract support) -----------
    # The slim tract table feeds the murder/rape map layers; the same values are
    # baked onto the block-group rows below so the block-group popup can show a
    # tract-scale murder/rape index + rate.
    print(f"Reading {TRACT_SRC.name} ...")
    tdf = pd.read_parquet(TRACT_SRC, columns=TRACT_KEEP)
    print(f"  {len(tdf):,} tracts, {len(TRACT_KEEP)} columns")
    tdf[TRACT_ID] = tdf[TRACT_ID].astype("string").str.zfill(TRACT_ID_LEN)
    assert tdf[TRACT_ID].str.len().eq(TRACT_ID_LEN).all(), "non-11-char tract GEOIDs found"
    assert tdf[TRACT_ID].is_unique, "duplicate tract GEOIDs"

    # Same estimate-mode validation as the block-group frame: modes are known,
    # none missing, and suppressed rows carry a null index.
    for mode_col, index_col in zip(TRACT_ESTIMATE_MODE, TRACT_INDEX_PRIMARY):
        modes = set(tdf[mode_col].dropna().astype(str).unique())
        unknown = sorted(modes - ESTIMATE_MODE_VALUES)
        if unknown:
            raise ValueError(f"Unknown estimate modes in {mode_col}: {unknown}")
        missing = int(tdf[mode_col].isna().sum())
        if missing:
            raise ValueError(f"Missing estimate modes in {mode_col}: {missing:,}")
        suppressed = tdf[mode_col].ne("count_derived")
        bad = suppressed & tdf[index_col].notna()
        if bad.any():
            raise ValueError(
                f"{mode_col} has {int(bad.sum()):,} suppressed rows with non-null {index_col}"
            )

    # Same rounding conventions as the block-group frame.
    for c in TRACT_INDEX_COLS + TRACT_RATE_COLS:
        tdf[c] = tdf[c].astype("float64").round(1)
    for c in TRACT_COUNT_COLS + TRACT_SUPPORT:
        tdf[c] = tdf[c].astype("float64").round(2)
    tdf[POPULATION_COL] = tdf[POPULATION_COL].astype("Int64")
    tdf["special_use_tract_flag"] = (
        tdf["special_use_tract_flag"].fillna(False).astype(bool).astype("int8")
    )
    tdf = tdf.sort_values(TRACT_ID).reset_index(drop=True)

    # --- Provenance disclosure classes (both frames) --------------------------
    # Computed here, on the same frames the snapshot freezes, so the class the map
    # marks is a function of the published parquets plus the run's own component
    # audit — nothing is re-derived downstream.
    juris = primary_jurisdictions()
    tract_codes = provenance_class(tdf, TRACT_ID, juris["tract"], RARE_OFFENSES)
    for c in TRACT_PROVENANCE_CLASS:
        tdf[c] = tract_codes[c]
    bg_codes = provenance_class(df, GEO_ID, juris["bg"], OFFENSES)
    for c in PROVENANCE_CLASS:
        df[c] = bg_codes[c]

    # Bake the parent tract's murder/rape primary index + rate onto the block-group
    # rows (parent tract = block-group GEOID's first 11 chars). These feed the
    # block-group popup's murder/rape rows, labeled as tract-scale. The parent
    # tract's murder/rape PROVENANCE class rides along for the same reason: the
    # block-group value shown for those two offenses IS the tract's, so the class
    # shown for it must be the tract's too. Only the outlier bit is taken from the
    # tract (the block group has no murder/rape index of its own to test); the
    # imputed bit stays the block group's own, since the imputed share is published
    # at block-group support.
    bake_cols = dict(BG_TRACT_BAKE)
    bake_cols.update({f"provenance_class_{o}": f"_tract_pv_{o}" for o in RARE_OFFENSES})
    bake = tdf[[TRACT_ID] + list(bake_cols)].rename(columns=bake_cols)
    df["_parent_tract"] = df[GEO_ID].str[:TRACT_ID_LEN]
    df = df.merge(bake, left_on="_parent_tract", right_on=TRACT_ID, how="left")
    for o in RARE_OFFENSES:
        inherited = (
            df[f"_tract_pv_{o}"].fillna(0).astype("int16") & PROVENANCE_MODEL_ONLY_OUTLIER
        )
        df[f"provenance_class_{o}"] = (
            df[f"provenance_class_{o}"].astype("int16") | inherited
        ).astype("int8")
    df = df.drop(
        columns=["_parent_tract", TRACT_ID] + [f"_tract_pv_{o}" for o in RARE_OFFENSES]
    )

    # Residents per square mile, both frames (viewer's national-zoom emphasis).
    # Zero-land cells (water-only) get NaN -> the viewer floors them, which fades
    # them at national zoom exactly like the empty land they are.
    for frame in (df, tdf):
        frame[POP_DENSITY_COL] = (
            frame[POPULATION_COL].astype("float64")
            / frame[AREA_COL_NAME].astype("float64").replace(0, np.nan)
        )
        # Two decimals below 10/sq mi (half of US land sits under 5.4), whole
        # people above it — more digits than that is tile bytes for no pixels.
        frame[POP_DENSITY_COL] = np.where(
            frame[POP_DENSITY_COL] < 10,
            frame[POP_DENSITY_COL].round(2),
            frame[POP_DENSITY_COL].round(0),
        )

    # Round to keep the artifact compact; indices to 1dp, rates to 1dp, counts to
    # 2dp (expected counts are fractional), density to DENSITY_DP (incidents/sq mi
    # — its own precision, see the DENSITY_DP note above; 2dp quantised 37% of
    # CONUS land into one flat colour). NaN is preserved -> JSON null so the
    # frontend renders suppressed / non-residential per-capita cells, and
    # water-only density cells, as "no data".
    for c in INDEX_COLS + RATE_COLS + BG_TRACT_BAKE_COLS:
        df[c] = df[c].astype("float64").round(1)
    for c in COUNT_COLS + OFFENSE_SUPPORT:
        df[c] = df[c].astype("float64").round(2)
    for c in DENSITY_COLS:
        df[c] = df[c].astype("float64").round(DENSITY_DP)
    df[POPULATION_COL] = df[POPULATION_COL].astype("Int64")
    # Special-use flag -> compact 0/1 int (NaN cells treated as not special-use).
    df["special_use_tract_flag"] = (
        df["special_use_tract_flag"].fillna(False).astype(bool).astype("int8")
    )

    df = df.sort_values(GEO_ID).reset_index(drop=True)

    # Land area is read into both frames as the denominator of the population
    # density and the weight of the density/`pct_above` statistics, but it is not
    # part of the tile schema — keep it out of the slim CSVs. (Held in `area` /
    # `tract_area` below for the stats block.)
    area = df[AREA_COL_NAME].astype("float64")
    tract_area = tdf[AREA_COL_NAME].astype("float64")
    # The imputed-share / support-source columns are INPUTS to provenance_class,
    # not tile attributes: the compact code replaces them downstream.
    df = df.drop(columns=[AREA_COL_NAME] + OFFENSE_IMPUTED_SHARE + OFFENSE_SUPPORT_SOURCE)
    tdf = tdf.drop(
        columns=[AREA_COL_NAME] + TRACT_IMPUTED_SHARE + TRACT_SUPPORT_SOURCE
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size/1e6:.1f} MB)")

    TRACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(TRACT_OUT, index=False)
    print(f"Wrote {TRACT_OUT.relative_to(REPO)}  ({TRACT_OUT.stat().st_size/1e6:.1f} MB)")

    # Report null coverage per index so the frontend "no data" handling is
    # justified. The four murder/rape index columns are all-null in the block-group
    # frame by policy, so their coverage is reported from the tract frame instead.
    print("\nNull coverage per index (rendered as no-data):")
    for c in INDEX_COLS:
        if c in RARE_INDEX_FROM_TRACT:
            n = int(tdf[c].isna().sum())
            print(f"  {c:38s} {n:6,} ({n/len(tdf)*100:4.1f}%)  [tract]")
        else:
            n = int(df[c].isna().sum())
            print(f"  {c:38s} {n:6,} ({n/len(df)*100:4.1f}%)")
    print("\nNull coverage per density (water-only -> no-data):")
    for c in DENSITY_COLS:
        n = df[c].isna().sum()
        print(f"  {c:38s} {n:6,} ({n/len(df)*100:4.1f}%)")
    print(
        f"\nspecial_use_tract_flag set on {int(df['special_use_tract_flag'].sum()):,} block groups"
    )

    print("\nProvenance disclosure class (block groups, per offense):")
    for o in OFFENSES:
        c = df[f"provenance_class_{o}"]
        print(
            f"  {o:22s} imputed {int(((c & PROVENANCE_IMPUTED) > 0).sum()):6,}   "
            f"model-only outlier {int(((c & PROVENANCE_MODEL_ONLY_OUTLIER) > 0).sum()):6,}"
        )
    pop_dens = df[POP_DENSITY_COL]
    print(
        f"  population density: {int(pop_dens.notna().sum()):,} valid, "
        f"median {pop_dens.median():,.1f}/sq mi, "
        f"{int((pop_dens < POP_DENSITY_LOW).sum()):,} below {POP_DENSITY_LOW}/sq mi"
    )

    # Size of what the map is about to mark, on both frames, in units / people /
    # land. Stamped into the snapshot (see `_meta` below) so the viewer's legend
    # can state the footprint instead of asserting one.
    bg_coverage = provenance_coverage(df, df[POPULATION_COL], area, OFFENSES)
    tract_coverage = provenance_coverage(
        tdf, tdf[POPULATION_COL], tract_area, RARE_OFFENSES
    )
    either = bg_coverage["any_offense"]["either"]
    print(
        f"  marked on ANY offense: {either['units']:,} block groups "
        f"({either['pct_units']}% of units, {either['pct_population']}% of people, "
        f"{either['pct_land']}% of land)"
    )

    # Stash distribution stats for the color-scale breakpoints used by the UI.
    #
    # Index layers are heavily right-skewed: most tracts are well below the national
    # average (for robbery ~71% sit under 100 and ~56% under 50), while a thin urban
    # tail and a handful of tiny-population cells run to the thousands.
    #
    # HISTORY / STATUS OF `q`: this block used to drive the index choropleth with a
    # per-layer ROBUST QUANTILE (equal-count) scale, emitted below as `q`. The viewer
    # NO LONGER CONSUMES `q` FOR INDEX LAYERS (2026-07-28). Equal-count bands are
    # per-layer and value-blind, and that manufactured artifacts: 7 of the 12 stops
    # sat below index 100, so a 6-point rural difference could cross two color bands
    # and print a false state-line seam, while a metro at 1.5x the national average
    # interpolated to near-white; and a degenerate low end (robbery's p2 and p10 are
    # both exactly 0) pinned a bottom stop at 0, painting 16,211 block groups one flat
    # color. The viewer now uses ONE fixed, value-anchored, log-symmetric-about-100
    # break set shared by every index layer, which also makes layers comparable to
    # each other. `q` stays emitted here as a DIAGNOSTIC of each layer's distribution
    # (and as the fallback for any older viewer reading this snapshot); nothing in the
    # published color scale depends on it.
    #
    # Percentiles used for the (diagnostic) equal-count bands. Denser near the top so
    # the urban hotspot tail keeps resolution instead of collapsing into one band.
    Q_PCTS = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]

    # `area` / `tract_area` (the WEIGHT for the density stops below and the
    # denominator of the `pct_above` land share on every layer) were lifted out of
    # the frames above, before the land column was dropped from the slim CSVs.

    stats = {}
    for c in INDEX_COLS:
        # The four murder/rape index layers render from tract geometry, so their
        # color-scale stats come from the tract distribution (the block-group
        # columns are all-null by policy). Every other stat stays block-group-derived.
        s = (tdf[c] if c in RARE_INDEX_FROM_TRACT else df[c]).dropna()
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
            # DIAGNOSTIC equal-count breakpoints (strictly increasing) and the
            # percentiles they correspond to. Not consumed by the index color scale
            # any more (see the note above); kept for distribution inspection.
            "q": q,
            "q_pcts": Q_PCTS[: len(q)],
            # F-01 legend readout: share of people / land above each FIXED index
            # break. Tract-support layers (murder/rape) are denominated on the
            # tract frame, matching the geometry that layer actually paints.
            "pct_above": (
                shares_above(tdf[c], tdf[POPULATION_COL], tract_area, PCT_ABOVE_INDEX_BREAKS)
                if c in RARE_INDEX_FROM_TRACT
                else shares_above(df[c], df[POPULATION_COL], area, PCT_ABOVE_INDEX_BREAKS)
            ),
        }
    # Density layers are SEQUENTIAL and heavily right-skewed (incidents/sq mi:
    # p50~30, p99~2000, max~90000 for total). A linear scale would paint the whole
    # country one color. Emit LOG-DOMAIN quantile stops (on positive values) so the
    # frontend can interpolate in log space and surface genuine hotspots.
    #
    # The top color stop anchors at p999 (the positive p99.9), NOT at `max`: total
    # density's max is ~40x its p99, a single block group that stretched the top of
    # the ramp so far that everything below it — the actual hotspots — compressed
    # into the orange range and never reached red. p99.9 still leaves ~240 block
    # groups above the anchor (they clamp to the deepest red, which is the honest
    # reading of "off the top of the scale"). `max` is retained for reference.
    #
    # The stops are AREA-WEIGHTED (2026-07-28). They used to be plain count
    # percentiles over positive block groups, but the choropleth paints AREA, not
    # block groups: a Manhattan block group and a 900-sq-mi desert block group are
    # one observation each to a count percentile and 5 orders of magnitude apart on
    # screen. Total density's count p10 was 0.33/sq mi while the area-weighted
    # median land value is 0.034/sq mi — the whole ramp started ~10x above where
    # most of the country's land actually sits, so 82.7% of CONUS land clamped to
    # the bottom color. Weighting each block group by its land area puts the ramp's
    # range where the land is; the log ramp and the p999 top anchor are unchanged.
    # `count_pcts` keeps the old unweighted values as a diagnostic.
    print(
        f"\nDensity stops are area-weighted (land area for {int(area.notna().sum()):,} of "
        f"{len(df):,} block groups; {float(area.sum()):,.0f} sq mi total)"
    )
    for c, series in [(c, df[c]) for c in DENSITY_COLS] + [
        (name, df[cols].sum(axis=1, min_count=1)) for name, cols in DERIVED_DENSITY.items()
    ]:
        s = series.dropna()
        pos = s[s > 0]
        w = area.reindex(pos.index).fillna(0.0).clip(lower=0)
        stops = {
            "p10": weighted_quantile(pos, w, 0.10),
            "p50": weighted_quantile(pos, w, 0.50),
            "p90": weighted_quantile(pos, w, 0.90),
            "p99": weighted_quantile(pos, w, 0.99),
            "p999": weighted_quantile(pos, w, 0.999),
        }
        stats[c] = {
            "kind": "density",
            "min": float(s.min()),
            **stops,
            "max": float(s.max()),
            "n_valid": int(s.size),
            "n_positive": int(pos.size),
            # F-01 legend readout, at the five stops that ARE this layer's ramp
            # (the density legend is per-layer, so the breaks travel with it).
            "pct_above": shares_above(
                series,
                df[POPULATION_COL],
                area,
                [stops[k] for k in DENSITY_STOP_KEYS],
                labels=DENSITY_STOP_KEYS,
            ),
            # How the stops are anchored, and what the old count-percentile stops
            # were, so a snapshot is self-describing about which ramp it carries.
            "stop_weighting": "land_area_sq_mi",
            "count_pcts": {
                "p10": float(pos.quantile(0.10)),
                "p50": float(pos.quantile(0.50)),
                "p90": float(pos.quantile(0.90)),
                "p99": float(pos.quantile(0.99)),
                "p999": float(pos.quantile(0.999)),
            },
        }
    # Encoding decisions the DOWNSTREAM steps must not re-derive independently.
    # 03 re-rounds the density columns before tiling and reads `density_dp` back
    # from here; 05 stamps `density_floor` + `index_breaks` into snapshot.json so
    # the viewer's paint floor and its F-01 break labels come from the build that
    # produced the numbers rather than from a hardcoded copy. Keyed with a leading
    # underscore so it can never collide with a published column name.
    stats["_meta"] = {
        "density_dp": DENSITY_DP,
        "density_floor": DENSITY_FLOOR,
        "index_breaks": PCT_ABOVE_INDEX_BREAKS,
        "density_stop_keys": DENSITY_STOP_KEYS,
        "pct_above_basis": {
            "population_column": POPULATION_COL,
            "land_column": AREA_COL,
            "rule": "share strictly above the break, denominated on units that have a value for that layer",
        },
        # Provenance disclosure: the codes, what each one MEANS IN WORDS (the
        # viewer prints these strings, it does not author its own), the exact
        # predicate each was computed from, and how much of the country each
        # covers. A reader who wants to know what the hatch means gets the
        # definition and the size from the same place the map got them.
        "provenance": {
            "codes": {
                "imputed": PROVENANCE_IMPUTED,
                "model_only_outlier": PROVENANCE_MODEL_ONLY_OUTLIER,
            },
            "labels": {
                "imputed": "Counts imputed for a non-reporting agency",
                "model_only_outlier": "Modeled outlier with no incident evidence",
            },
            "popup_text": {
                "imputed": (
                    "No usable 2024 filing exists for the agency covering this area, so "
                    "most of this cell's estimated incidents were imputed from the "
                    "state benchmark rather than reported."
                ),
                "model_only_outlier": (
                    "There is no direct incident evidence here and the modeled value is "
                    "more than five times its own jurisdiction's median — a documented "
                    "outlier class the project publishes rather than hides."
                ),
            },
            "predicate": {
                "imputed": f"benchmark_imputed_share_{{offense}} >= {BENCHMARK_IMPUTED_SHARE_MIN}",
                "model_only_outlier": (
                    f"index_{{offense}}_primary > {MODEL_ONLY_OUTLIER_RATIO} x the primary "
                    f"jurisdiction's median AND numerator_support_source == "
                    f"'{MODEL_ONLY_SUPPORT_SOURCE}' AND reliability_tier == "
                    f"'{MODEL_ONLY_RELIABILITY_TIER}' AND effective_numerator_support <= 0"
                ),
            },
            "coverage_block_group": bg_coverage,
            "coverage_tract": tract_coverage,
        },
        # National-zoom population-aware emphasis. The viewer reads the ramp
        # anchors from here so the two cannot drift, the same way the density
        # paint floor already travels with the precision it was derived from.
        "population_density": {
            "column": POP_DENSITY_COL,
            "low_per_sq_mi": POP_DENSITY_LOW,
            "high_per_sq_mi": POP_DENSITY_HIGH,
            "min_emphasis": POP_DENSITY_MIN_EMPHASIS,
            "basis": (
                "residents per square mile of land; emphasis only — no published "
                "value, popup reading or legend break depends on it"
            ),
            "anchors": (
                "low = federal 'frontier' density; high = Census urban-area core "
                "density criterion"
            ),
        },
    }

    import json

    stats_path = REPO / "frontend/tmp/index_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote stats -> {stats_path.relative_to(REPO)}")
    print(
        f"  density precision {DENSITY_DP}dp (floor {DENSITY_FLOOR:g}); "
        f"d_tot stops "
        + " / ".join(f"{stats['crime_density_total'][k]:.6g}" for k in DENSITY_STOP_KEYS)
    )
    pa = stats["index_total_primary_event_weighted"]["pct_above"]
    i100 = pa["breaks"].index(100)
    print(
        f"  pct_above (i_evw @100): {pa['people'][i100]}% of people, "
        f"{pa['land'][i100]}% of land"
    )


if __name__ == "__main__":
    main()
