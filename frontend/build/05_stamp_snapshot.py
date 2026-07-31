"""
05 - Stamp the frozen snapshot metadata.

Writes snapshot.json into the published-artifacts directory (CRIMERISK_SNAPSHOT_
OUT): the manifest the frontend reads at load time (index list, color-scale stats,
attribute key map, and the tract-support routing for the murder/rape layers) PLUS
provenance — the commit the SOURCE DATA was built at, the source parquets, the
geometry vintage, and the build timestamp.

This is what makes the artifact a deliberate FROZEN 2024 snapshot, decoupled
from the live pipeline. Regenerate by re-running 01..05.

PROVENANCE (audit finding P-01). `source_git_commit` used to be `git rev-parse
HEAD` evaluated at STAMP TIME. That is not the commit the data came from: the
snapshot is normally rebuilt before the commit that records it, so the stamp
pointed one commit behind the state it was describing, and it would have pointed
at an unrelated commit for any snapshot rebuilt later from an older candidate.
The commit is now READ FROM THE CANDIDATE RUN MANIFEST that produced the source
parquet (`<candidate dir>/manifest.json` -> `run.git.head_sha`), i.e. the only
commit this step can actually PROVE is the one the numbers were computed at. The
manifest is bound to the file it describes by checking that its recorded output
path and size match the parquet being frozen; a mismatch, a missing manifest or a
dirty source tree is recorded in `source_data_provenance` rather than silently
smoothed over, and `git rev-parse HEAD` is kept only as
`stamped_at_git_commit` — a note about when the artifact was written, never as a
claim about where the data came from.

Run:  uv run python frontend/build/05_stamp_snapshot.py
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TMP = REPO / "frontend/tmp"
DISCLAIMERS = REPO / "scripts/release/assets/methodology_exclusions.json"

# The canonical methodology doc, republished next to the map so the panel's
# Methodology link has a target that exists in the deployed site (see
# render_methodology). Rendered from the SAME commit the snapshot is stamped at.
METHODOLOGY_SRC = REPO / "docs/METHODOLOGY.md"
METHODOLOGY_PAGE = "methodology.html"
# Self-contained by necessity: this environment blocks external CDNs, and a
# methodology page that fails to style itself offline is worse than a plain one.
METHODOLOGY_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CrimeRisk - Methodology</title>
<style>
  :root { color-scheme: dark; }
  html, body { margin: 0; background: #0b0e14; color: #d7dce6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 15px; line-height: 1.65; }
  main { max-width: 46rem; margin: 0 auto; padding: 32px 22px 96px; }
  a { color: #7fb4e8; }
  .back { display: inline-block; font-size: 13px; margin-bottom: 26px; }
  h1 { font-size: 26px; line-height: 1.25; margin: 0 0 18px; }
  h2 { font-size: 19px; margin: 38px 0 10px; padding-top: 16px;
       border-top: 1px solid #232a36; }
  h3 { font-size: 16px; margin: 26px 0 8px; color: #e6e9ef; }
  h4 { font-size: 14px; margin: 20px 0 6px; color: #aab2c2; }
  p { margin: 0 0 14px; }
  ul { margin: 0 0 14px; padding-left: 20px; }
  li { margin-bottom: 6px; }
  code { background: rgba(255,255,255,0.06); border: 1px solid #232a36;
         border-radius: 4px; padding: 0 4px; font-size: 12.5px;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .tw { overflow-x: auto; margin: 0 0 16px; }
  table { border-collapse: collapse; font-size: 13.5px; min-width: 22rem; }
  th, td { border: 1px solid #232a36; padding: 5px 11px; text-align: left; }
  th { background: rgba(255,255,255,0.04); font-weight: 600; }
</style>
</head>
<body>
<main>
<a class="back" href="./">&larr; Back to the map</a>
__BODY__
</main>
</body>
</html>
"""

# Snapshot data year + source parquet + published-artifacts directory: single
# point of control in snapshot_config.env (see 01). data_year and the PMTiles
# filenames derive from the year; snapshot.json is written to OUT_DIR.
CONFIG = {
    k.strip(): v.strip()
    for k, v in (
        line.split("=", 1)
        for line in (Path(__file__).resolve().parent / "snapshot_config.env").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
}
YEAR = int(CONFIG["CRIMERISK_SNAPSHOT_YEAR"])
OUT_DIR = REPO / CONFIG["CRIMERISK_SNAPSHOT_OUT"]

# Estimate-mode integer codes (mirror 03 EMODE_CODE) -> human reason shown in the
# popup for special-use / suppressed cells.
# Code 2's label used to be the bare phrase "special-use area", which now that the
# BADGE names the mode (below) would make the "Why muted" row a verbatim echo of the
# badge above it. Every label here is the sentence-length REASON; the badge is the
# name.
EMODE_LABELS = {
    0: "Estimated from official total",
    1: "Too few residents for a per-person rate",
    2: "Airport, park or similar special-use land where a per-person rate misleads",
    3: "Vehicle-based rate not meaningful",
    4: "Too little exposure for a per-person rate",
}

# Short BADGE text per estimate-mode code (audit finding D-04). The popup badge
# used to be the hardcoded string "Special-use area" for every muted cell, which
# on the aggregate layers is wrong for 549 of the 686 flagged block groups: the
# `special_use_tract_flag` those layers key on covers `non_residential` (549) as
# well as `special_use` (137), and the popup forced the code to 2 before looking
# the reason up. The badge now names the actual mode, and EMODE_LABELS supplies
# the sentence-length "why" underneath it.
EMODE_BADGES = {
    1: "Non-residential area",
    2: "Special-use area",
    3: "Vehicle rate not meaningful",
    4: "Very low exposure",
}
# Shown when an aggregate layer's member offenses do not agree on one mode. No
# dominant-label shortcut: the popup says the reasons differ instead of picking
# one. (Zero cells on the v20 surface — all 686 flagged block groups carry the
# same mode across all seven offenses — but the viewer must not depend on that.)
EMODE_MIXED_BADGE = "Muted area"
EMODE_MIXED_LABEL = "muted for more than one reason across these offenses"

# Reliability-tier integer codes (mirror 03 RELIABILITY_CODE) -> human label shown
# in the popup. `low` = model-only with ~0 incident support (the count-noise tier
# de-emphasized in the choropleth); medium/high are anchored by real incidents.
RELIABILITY_LABELS = {
    2: "high",
    1: "medium",
    0: "low",
}

# Static layer-level note for the four murder/rape index layers (primary +
# per-resident). These offenses publish at census-tract scale: a single year of
# data is too sparse to support a block-group rate, so the map draws them from
# tract geometry while block groups keep their expected counts and incident-
# support metadata. This is the layer-level heads-up shown regardless of which
# cell a user hovers.
MURDER_RAPE_RELIABILITY_NOTE = (
    "Murder and rape rates are published at census-tract scale: a single year of "
    "data is too sparse to support a block-group rate for these offenses. The map "
    "shows the tract-scale estimate; block groups keep their expected counts and "
    "incident-support metadata."
)

# Human-readable labels for the layer selector, in display order. Three families:
#   - index_*  : diverging per-capita index about 100 (national average)
#   - density_*: sequential incidents-per-square-mile hotspot view (kind=density)
# The default total layer is `i_evw` (index_total_primary_event_weighted, the
# exposure-denominated total) — the readable risk surface that does not inflate
# low-resident commercial tracts. The resident total (`i_tot`) stays selectable.
INDEX_DEFS = [
    {"key": "i_evw", "col": "index_total_primary_event_weighted", "label": "Total risk", "group": "Aggregate", "kind": "index"},
    {"key": "i_per", "col": "index_personal_part1_resident", "label": "Personal / violent", "group": "Aggregate", "kind": "index"},
    {"key": "i_pro", "col": "index_property_part1_resident", "label": "Property", "group": "Aggregate", "kind": "index"},
    {"key": "i_tot", "col": "index_total_part1_resident", "label": "Total (resident)", "group": "Aggregate (other)", "kind": "index"},
    # Published and tiled, but NOT offered by the selector: the equal-offense total
    # is a diagnostic construction (each offense index averaged with equal weight),
    # and a reader gains nothing from choosing it. It stays in the catalogue so the
    # snapshot still describes every attribute the tiles carry.
    {"key": "i_eq", "col": "index_total_equal_offense", "label": "Total - equal offense", "group": "Aggregate (other)", "kind": "index"},
    {"key": "i_harm", "col": "index_total_harm", "label": "Total - severity weighted", "group": "Aggregate (other)", "kind": "index"},
    {"key": "ip_mur", "col": "index_murder_primary", "label": "Murder", "group": "Personal", "kind": "index", "note": MURDER_RAPE_RELIABILITY_NOTE},
    {"key": "ip_rap", "col": "index_rape_primary", "label": "Rape", "group": "Personal", "kind": "index", "note": MURDER_RAPE_RELIABILITY_NOTE},
    {"key": "ip_rob", "col": "index_robbery_primary", "label": "Robbery", "group": "Personal", "kind": "index"},
    {"key": "ip_agg", "col": "index_aggravated_assault_primary", "label": "Aggravated assault", "group": "Personal", "kind": "index"},
    {"key": "ip_bur", "col": "index_burglary_primary", "label": "Burglary", "group": "Property", "kind": "index"},
    {"key": "ip_lar", "col": "index_larceny_primary", "label": "Larceny / theft", "group": "Property", "kind": "index"},
    {"key": "ip_mvt", "col": "index_motor_vehicle_theft_primary", "label": "Motor vehicle theft", "group": "Property", "kind": "index"},
    # Per-resident index per offense — same resident denominator for every offense,
    # so they are comparable across offenses (the "Per-resident index" measure).
    {"key": "ir_mur", "col": "index_murder_resident", "label": "Murder (per resident)", "group": "Per resident", "kind": "index", "note": MURDER_RAPE_RELIABILITY_NOTE},
    {"key": "ir_rap", "col": "index_rape_resident", "label": "Rape (per resident)", "group": "Per resident", "kind": "index", "note": MURDER_RAPE_RELIABILITY_NOTE},
    {"key": "ir_rob", "col": "index_robbery_resident", "label": "Robbery (per resident)", "group": "Per resident", "kind": "index"},
    {"key": "ir_agg", "col": "index_aggravated_assault_resident", "label": "Aggravated assault (per resident)", "group": "Per resident", "kind": "index"},
    {"key": "ir_bur", "col": "index_burglary_resident", "label": "Burglary (per resident)", "group": "Per resident", "kind": "index"},
    {"key": "ir_lar", "col": "index_larceny_resident", "label": "Larceny (per resident)", "group": "Per resident", "kind": "index"},
    {"key": "ir_mvt", "col": "index_motor_vehicle_theft_resident", "label": "Motor vehicle theft (per resident)", "group": "Per resident", "kind": "index"},
    # Density (incidents / sq mi) — denominator-free hotspot view. NULL only for
    # water-only tracts; valid even in special-use cells. offense key suffix is
    # the same as the per-offense breakdown so disclaimers/popups line up.
    {"key": "d_tot", "col": "crime_density_total", "label": "Total density", "group": "Density (per sq mi)", "kind": "density"},
    {"key": "d_mur", "col": "crime_density_murder", "label": "Murder density", "group": "Density (per sq mi)", "kind": "density"},
    {"key": "d_rap", "col": "crime_density_rape", "label": "Rape density", "group": "Density (per sq mi)", "kind": "density"},
    {"key": "d_rob", "col": "crime_density_robbery", "label": "Robbery density", "group": "Density (per sq mi)", "kind": "density"},
    {"key": "d_agg", "col": "crime_density_aggravated_assault", "label": "Aggravated assault density", "group": "Density (per sq mi)", "kind": "density"},
    {"key": "d_bur", "col": "crime_density_burglary", "label": "Burglary density", "group": "Density (per sq mi)", "kind": "density"},
    {"key": "d_lar", "col": "crime_density_larceny", "label": "Larceny density", "group": "Density (per sq mi)", "kind": "density"},
    {"key": "d_mvt", "col": "crime_density_motor_vehicle_theft", "label": "Motor vehicle theft density", "group": "Density (per sq mi)", "kind": "density"},
    # Derived rollup densities: no baked tile column — the viewer evaluates the sum
    # of the member per-offense density keys (`sum_of`) per feature. Exact, since
    # densities over the same land area are additive. Stats come from 01, which
    # computes the identical row-wise sum on the source parquet.
    {"key": "d_per", "col": "crime_density_personal", "label": "Personal / violent density", "group": "Density (per sq mi)", "kind": "density", "sum_of": ["d_mur", "d_rap", "d_rob", "d_agg"]},
    {"key": "d_pro", "col": "crime_density_property", "label": "Property density", "group": "Density (per sq mi)", "kind": "density", "sum_of": ["d_bur", "d_lar", "d_mvt"]},
]

# Default total-crime layer key: the exposure-denominated total.
DEFAULT_INDEX_KEY = "i_evw"

# offense layer key (index or density) -> the offense suffix used for the active
# disclaimer + the per-offense estimate-mode tile key. Aggregates have no single
# offense, so the active disclaimer falls back to the "general" note.
LAYER_OFFENSE = {
    "ip_mur": "murder", "ip_rap": "rape", "ip_rob": "robbery",
    "ip_agg": "aggravated_assault", "ip_bur": "burglary", "ip_lar": "larceny",
    "ip_mvt": "motor_vehicle_theft",
    "ir_mur": "murder", "ir_rap": "rape", "ir_rob": "robbery",
    "ir_agg": "aggravated_assault", "ir_bur": "burglary", "ir_lar": "larceny",
    "ir_mvt": "motor_vehicle_theft",
    "d_mur": "murder", "d_rap": "rape", "d_rob": "robbery",
    "d_agg": "aggravated_assault", "d_bur": "burglary", "d_lar": "larceny",
    "d_mvt": "motor_vehicle_theft",
}

# Map each layer key to the per-offense estimate-mode tile key (em_*) that decides
# whether a cell is special-use FOR THAT LAYER. Aggregates and total density use
# the tract-level special_use_tract_flag (`su`) instead.
LAYER_EMODE_KEY = {
    "ip_mur": "em_mur", "ip_rap": "em_rap", "ip_rob": "em_rob",
    "ip_agg": "em_agg", "ip_bur": "em_bur", "ip_lar": "em_lar",
    "ip_mvt": "em_mvt",
    "ir_mur": "em_mur", "ir_rap": "em_rap", "ir_rob": "em_rob",
    "ir_agg": "em_agg", "ir_bur": "em_bur", "ir_lar": "em_lar",
    "ir_mvt": "em_mvt",
    "d_mur": "em_mur", "d_rap": "em_rap", "d_rob": "em_rob",
    "d_agg": "em_agg", "d_bur": "em_bur", "d_lar": "em_lar", "d_mvt": "em_mvt",
}

# Map each layer key to the per-offense reliability tile key (rt_*) that drives
# the count-confidence de-emphasis FOR THAT LAYER. Reliability is a property of the
# PER-CAPITA (per-exposure) index, so only index layers are de-emphasized. Offense
# index layers map to their own offense. Aggregate index layers blend many offenses
# (no single tier), so they map to robbery's tier (rt_rob) — the offense whose
# model-only tail drives the worst per-exposure outliers — as a conservative
# confidence proxy. Density layers are incidents-per-area (denominator-free) and
# immune to the tiny-population count-noise problem, so they are left unmapped.
LAYER_RELIABILITY_KEY = {
    "i_evw": "rt_rob", "i_per": "rt_rob", "i_pro": "rt_rob",
    "i_tot": "rt_rob", "i_eq": "rt_rob", "i_harm": "rt_rob",
    "ip_mur": "rt_mur", "ip_rap": "rt_rap", "ip_rob": "rt_rob",
    "ip_agg": "rt_agg", "ip_bur": "rt_bur", "ip_lar": "rt_lar",
    "ip_mvt": "rt_mvt",
    # Per-resident index layers share the per-offense reliability tier.
    "ir_mur": "rt_mur", "ir_rap": "rt_rap", "ir_rob": "rt_rob",
    "ir_agg": "rt_agg", "ir_bur": "rt_bur", "ir_lar": "rt_lar",
    "ir_mvt": "rt_mvt",
}

# Map each layer key to the per-offense PROVENANCE tile key (pv_*) that decides
# whether the cell is hatched on that layer. Unlike reliability, this has no
# stand-in: an aggregate layer has no single provenance, so aggregates are left
# UNMAPPED here on purpose and the viewer unions the classes over the offenses the
# layer is actually built from (`memberSuffixes`), then names them in the popup.
# Density layers get the same treatment as index layers: a benchmark-imputed count
# is just as imputed when you divide it by area as when you divide it by people.
LAYER_PROVENANCE_KEY = {
    "ip_mur": "pv_mur", "ip_rap": "pv_rap", "ip_rob": "pv_rob",
    "ip_agg": "pv_agg", "ip_bur": "pv_bur", "ip_lar": "pv_lar",
    "ip_mvt": "pv_mvt",
    "ir_mur": "pv_mur", "ir_rap": "pv_rap", "ir_rob": "pv_rob",
    "ir_agg": "pv_agg", "ir_bur": "pv_bur", "ir_lar": "pv_lar",
    "ir_mvt": "pv_mvt",
    "d_mur": "pv_mur", "d_rap": "pv_rap", "d_rob": "pv_rob",
    "d_agg": "pv_agg", "d_bur": "pv_bur", "d_lar": "pv_lar", "d_mvt": "pv_mvt",
}

# offense key suffix -> tooltip label, for the per-offense breakdown. The slim
# tile carries ip_* (primary index), ir_* (resident index), r_* (primary rate),
# ec_* (expected count), rt_* (reliability tier), es_* (effective support),
# pv_* (provenance disclosure class) per offense.
OFFENSE_DEFS = [
    {"key": "mur", "label": "Murder"},
    {"key": "rap", "label": "Rape"},
    {"key": "rob", "label": "Robbery"},
    {"key": "agg", "label": "Aggravated assault"},
    {"key": "bur", "label": "Burglary"},
    {"key": "lar", "label": "Larceny"},
    {"key": "mvt", "label": "Motor vehicle theft"},
]

# --- Two-axis selector taxonomy -------------------------------------------------
# The viewer splits the choice into CRIME TYPE (what offense) and MEASURE (what the
# value is in terms of). Both axes map to the SAME underlying layer keys defined in
# INDEX_DEFS above — this is purely a clearer presentation of those keys, no new
# data.
#
# MEASURES, in display order. `kind` selects the color scheme (index = robust
# diverging-about-100 quantile bands; density = log hotspot ramp). The explanation
# shown under the controls is per crime-type-x-measure pairing: each crime type
# carries a `measure_explain` dict naming its actual denominator and caveats.
#
# TWO AXES, TWO DROPDOWNS. There used to be a third, conditional select ("Other
# total views") carrying an `alt` list on the Total crime type: it appeared and
# disappeared with the crime type, opened on the placeholder "Standard total
# selected", and its first two entries were the Personal and Property crime types
# again under different labels (identical layer keys). It is gone. The one view it
# carried that is not reachable another way — the harm-weighted total — is now a
# real MEASURE (`harm`) that only the Total crime type declares, so the axis shape
# is uniform: every crime type has the same measure dropdown, Total just has one
# more entry in it. The equal-offense total (`i_eq`) is a diagnostic construction
# and is no longer offered as a view at all; it stays in INDEX_DEFS and in the
# tiles, simply unreferenced by the selector.
MEASURE_DEFS = [
    {"id": "primary", "label": "Risk index", "kind": "index"},
    {"id": "harm", "label": "Severity-weighted", "kind": "index"},
    {"id": "resident", "label": "Per-resident index", "kind": "index"},
    {"id": "density", "label": "Density (per sq mi)", "kind": "density"},
]

# CRIME TYPES, in display order. Each maps a measure id -> the existing layer key,
# and carries the pairing-specific `measure_explain` text. A crime type simply
# omits the measures it does not have (only Total declares `harm`).
CRIME_TYPE_DEFS = [
    {
        "id": "total",
        "label": "Total",
        "measures": {
            "primary": "i_evw",
            "harm": "i_harm",
            "resident": "i_tot",
            "density": "d_tot",
        },
        "measure_explain": {
            "primary": (
                "All seven crime types combined, weighted by their national mix of "
                "reported incidents. Each crime is measured against its own base — "
                "people present for violent crime and theft, buildings for burglary, "
                "vehicles for vehicle theft — so business districts with few "
                "residents are not inflated. 100 = national average."
            ),
            # Promoted from the deleted `alt` list, wording unchanged.
            "harm": (
                "Offenses weighted by the harm they cause (sentencing-based "
                "weights), so one murder counts far more than one theft. Murder "
                "and rape enter at census-tract scale, spread by where people "
                "are, to keep single rare events from dominating one block. "
                "100 = national average."
            ),
            "resident": (
                "All seven crime types combined, divided by residents only. Reads as "
                "the reported-crime burden per resident; downtowns, malls, and other "
                "places where most people present are visitors or workers will look "
                "worse than the risk to any one person. 100 = national average."
            ),
            "density": (
                "All reported incidents per square mile, regardless of who lives "
                "there. Shows where crime concentrates on the ground, not per-person "
                "risk. Log scale: green = low, red = hotspot."
            ),
        },
    },
    {
        "id": "personal",
        "label": "Personal (violent)",
        "measures": {"primary": "i_per", "resident": "i_per", "density": "d_per"},
        # The published violent rollup is resident-based, so the risk-index and
        # per-resident measures are the same layer; the copy says so.
        "measure_explain": {
            "primary": (
                "Murder, rape, robbery, and aggravated assault combined into one "
                "index per resident, weighted by their national incident mix. This "
                "rollup is resident-based (identical to its per-resident view), so "
                "visitor- and worker-heavy areas can read high; the Total risk "
                "index is the view that uses each crime's own base. "
                "100 = national average."
            ),
            "resident": (
                "Murder, rape, robbery, and aggravated assault combined into one "
                "index per resident, weighted by their national incident mix. This "
                "rollup is resident-based (identical to its risk-index view), so "
                "visitor- and worker-heavy areas can read high; the Total risk "
                "index is the view that uses each crime's own base. "
                "100 = national average."
            ),
            "density": (
                "Reported murders, rapes, robberies, and aggravated assaults per "
                "square mile, added together. Shows where violent incidents "
                "concentrate on the ground, regardless of who lives there. "
                "Log scale: green = low, red = hotspot."
            ),
        },
    },
    {
        "id": "property",
        "label": "Property",
        "measures": {"primary": "i_pro", "resident": "i_pro", "density": "d_pro"},
        "measure_explain": {
            "primary": (
                "Burglary, larceny, and motor vehicle theft combined into one index "
                "per resident, weighted by their national incident mix. This rollup "
                "is resident-based (identical to its per-resident view), so "
                "shopping and business districts can read high; the Total risk "
                "index is the view that uses each crime's own base — premises for "
                "burglary, vehicles for vehicle theft. 100 = national average."
            ),
            "resident": (
                "Burglary, larceny, and motor vehicle theft combined into one index "
                "per resident, weighted by their national incident mix. This rollup "
                "is resident-based (identical to its risk-index view), so shopping "
                "and business districts can read high; the Total risk index is the "
                "view that uses each crime's own base — premises for burglary, "
                "vehicles for vehicle theft. 100 = national average."
            ),
            "density": (
                "Reported burglaries, larcenies, and motor vehicle thefts per "
                "square mile, added together. Shows where property incidents "
                "concentrate on the ground. Log scale: green = low, red = hotspot."
            ),
        },
    },
    {
        "id": "murder", "label": "Murder",
        "measures": {"primary": "ip_mur", "resident": "ir_mur", "density": "d_mur"},
        "measure_explain": {
            "primary": (
                "Murders per person present — residents plus estimated daytime "
                "workers and activity. Murders are rare enough that one year of "
                "data cannot support a block-group rate, so this layer is estimated "
                "and drawn at census-tract scale. 100 = national average."
            ),
            "resident": (
                "Murders per resident. Ignores workers and visitors, so downtowns "
                "and job centers can read worse than the risk to any one person "
                "present. Drawn at census-tract scale — one year of data cannot "
                "support a block-group murder rate. 100 = national average."
            ),
            "density": (
                "Estimated murders per square mile. Incidents are so rare that "
                "fine-grained differences are approximate. Log scale: green = low, "
                "red = hotspot."
            ),
        },
    },
    {
        "id": "rape", "label": "Rape",
        "measures": {"primary": "ip_rap", "resident": "ir_rap", "density": "d_rap"},
        "measure_explain": {
            "primary": (
                "Reported rapes per person present — residents plus estimated "
                "daytime workers and activity. Reports are sparse enough that this "
                "layer is estimated and drawn at census-tract scale. Rape is "
                "heavily under-reported everywhere: this reflects reports to "
                "police, not true incidence. 100 = national average."
            ),
            "resident": (
                "Reported rapes per resident, drawn at census-tract scale — one "
                "year of data cannot support a block-group rate. Rape is heavily "
                "under-reported everywhere: this reflects reports to police, not "
                "true incidence. 100 = national average."
            ),
            "density": (
                "Reported rapes per square mile. Sparse and heavily under-reported "
                "— treat fine-grained differences as approximate. Log scale: "
                "green = low, red = hotspot."
            ),
        },
    },
    {
        "id": "robbery", "label": "Robbery",
        "measures": {"primary": "ip_rob", "resident": "ir_rob", "density": "d_rob"},
        "measure_explain": {
            "primary": (
                "Robberies per person present — residents plus estimated daytime "
                "workers and activity — because robbery targets whoever is on the "
                "street or at a business, not just residents. "
                "100 = national average."
            ),
            "resident": (
                "Robberies per resident. Overstates downtowns and commercial areas, "
                "where most people at risk are visitors or workers rather than "
                "residents. 100 = national average."
            ),
            "density": (
                "Reported robberies per square mile. Shows where robberies "
                "concentrate on the ground, independent of population. Log scale: "
                "green = low, red = hotspot."
            ),
        },
    },
    {
        "id": "aggravated_assault", "label": "Aggravated assault",
        "measures": {"primary": "ip_agg", "resident": "ir_agg", "density": "d_agg"},
        "measure_explain": {
            "primary": (
                "Aggravated assaults per person present — residents plus estimated "
                "daytime workers and activity, since assaults happen where people "
                "actually are, not just where they sleep. 100 = national average."
            ),
            "resident": (
                "Aggravated assaults per resident. Can overstate visitor- and "
                "worker-heavy areas, where many people involved are not residents. "
                "100 = national average."
            ),
            "density": (
                "Reported aggravated assaults per square mile. Shows concentration "
                "on the ground, independent of population. Log scale: green = low, "
                "red = hotspot."
            ),
        },
    },
    {
        "id": "burglary", "label": "Burglary",
        "measures": {"primary": "ip_bur", "resident": "ir_bur", "density": "d_bur"},
        "measure_explain": {
            "primary": (
                "Burglaries per premises at risk — homes plus weighted commercial "
                "and industrial buildings — because burglary targets buildings, not "
                "people. A business district with few residents is judged on its "
                "building stock. 100 = national average."
            ),
            "resident": (
                "Burglaries per resident. Inflates business districts, where most "
                "burglary targets are commercial buildings rather than homes; the "
                "risk-index view divides by premises instead. "
                "100 = national average."
            ),
            "density": (
                "Reported burglaries per square mile. Shows concentration on the "
                "ground, independent of premises or population. Log scale: "
                "green = low, red = hotspot."
            ),
        },
    },
    {
        "id": "larceny", "label": "Larceny",
        "measures": {"primary": "ip_lar", "resident": "ir_lar", "density": "d_lar"},
        "measure_explain": {
            "primary": (
                "Thefts per person present — residents plus estimated daytime "
                "workers, shoppers, and other activity — because theft follows foot "
                "traffic and commerce, not just residents. 100 = national average."
            ),
            "resident": (
                "Thefts per resident. Strongly overstates malls, downtowns, and "
                "other retail areas where most victims are visitors; the risk-index "
                "view divides by people present instead. 100 = national average."
            ),
            "density": (
                "Reported thefts per square mile. Shows concentration on the "
                "ground, independent of population. Log scale: green = low, "
                "red = hotspot."
            ),
        },
    },
    {
        "id": "motor_vehicle_theft", "label": "Motor vehicle theft",
        "measures": {"primary": "ip_mvt", "resident": "ir_mvt", "density": "d_mvt"},
        "measure_explain": {
            "primary": (
                "Vehicle thefts per vehicle regularly present — household vehicles "
                "plus estimated commuter vehicles — because the thing at risk is a "
                "parked car, not a person. 100 = national average."
            ),
            "resident": (
                "Vehicle thefts per resident. Overstates commuter destinations, "
                "where many stolen vehicles belong to workers and visitors; the "
                "risk-index view divides by vehicles present instead. "
                "100 = national average."
            ),
            "density": (
                "Reported vehicle thefts per square mile. Shows concentration on "
                "the ground, independent of the number of vehicles. Log scale: "
                "green = low, red = hotspot."
            ),
        },
    },
]

# Opening view: total crime on the event-weighted primary risk index.
DEFAULT_CRIME_TYPE = "total"
DEFAULT_MEASURE = "primary"
SOURCE_PARQUET = CONFIG["CRIMERISK_SNAPSHOT_SRC"]
SOURCE_TRACT_PARQUET = CONFIG["CRIMERISK_SNAPSHOT_TRACT_SRC"]

# Murder and rape publish at census-tract support: their four index layers render
# from the tract tiles, everything else from the block-group tiles. Any layer key
# absent from this map renders from the block-group source.
LAYER_SOURCE = {
    "ip_mur": "tract",
    "ip_rap": "tract",
    "ir_mur": "tract",
    "ir_rap": "tract",
}

# Baked murder/rape popup keys on the block-group tiles: the parent tract's index
# (ipt_*) and rate (rpt_*), so the viewer reads them from the snapshot rather than
# hardcoding the key names.
TRACT_BAKED_KEYS = {
    "mur": {"index": "ipt_mur", "rate": "rpt_mur"},
    "rap": {"index": "ipt_rap", "rate": "rpt_rap"},
}

# Shown on the tract-scale layers and on the baked murder/rape block-group popup
# rows so a tract-scale value is never mistaken for a block-group one.
TRACT_SCALE_NOTE = (
    "Tract-scale estimate — murder and rape rates are published at census-tract "
    "scale; block groups carry expected counts but no per-person rate for these "
    "offenses."
)


def plain_disclaimers(raw: dict) -> dict:
    """Frontend copy for backend-owned methodology exclusion metadata."""
    plain = {
        "general": {
            "default_override_note": (
                "Special-use areas are muted by default in per-person views. Turn "
                "them on to inspect reported counts and crime density instead."
            )
        },
        "aggravated_assault": {
            "excluded_by_default": False,
            "disclaimer": (
                "Aggravated assault is shown where the resident or daytime "
                "population is large enough for a rate to mean much."
            ),
        },
        "burglary": {
            "excluded_by_default": True,
            "disclaimer": (
                "Burglary rates are muted in airports, parks, and large "
                "institutional or commercial-only areas where a per-person rate "
                "would be misleading. Use reported counts or crime density there."
            ),
        },
        "larceny": {
            "excluded_by_default": True,
            "disclaimer": (
                "Larceny rates are muted in parks, undeveloped land, and similar "
                "places with very few residents or workers. Use reported counts or "
                "crime density there."
            ),
        },
        "motor_vehicle_theft": {
            "excluded_by_default": True,
            "disclaimer": (
                "Motor vehicle theft rates are muted in airports, transit hubs, "
                "and other large visitor-heavy areas because local household "
                "vehicles do not represent all vehicles at risk. Use reported "
                "counts or crime density there."
            ),
        },
        "murder": {
            "excluded_by_default": False,
            "disclaimer": (
                "Murder rates are published at census-tract scale, where a year of "
                "data can support a rate; block groups show the tract estimate. "
                "Rates are shown where the resident or daytime population is large "
                "enough for a rate to mean much."
            ),
        },
        "rape": {
            "excluded_by_default": True,
            "disclaimer": (
                "Rape rates are published at census-tract scale, where a year of "
                "data can support a rate; block groups show the tract estimate. "
                "Rates are muted in parks, undeveloped land, and similar places "
                "with very few residents or workers; use reported counts or crime "
                "density there."
            ),
        },
        "robbery": {
            "excluded_by_default": False,
            "disclaimer": (
                "Robbery is shown where the resident or daytime population is large "
                "enough for a rate to mean much."
            ),
        },
    }
    for key, value in raw.items():
        plain.setdefault(key, value)
    return plain


def data_vintage(provenance: dict) -> dict:
    """The vintage facts the panel shows without anyone having to open a popup.

    Four DIFFERENT dates get conflated by readers, so each is named separately
    rather than merged into one "updated" string: the reference year the crime
    counts describe, the census vintage the geometry comes from, when this
    snapshot was frozen, and the commit the data was built at. `line` is the one
    the panel prints; the parts stay addressable for the popup/footer.
    """
    built = datetime.now(timezone.utc)
    return {
        "reference_year": YEAR,
        "geometry_vintage": "2020",
        "snapshot_built_utc": built.isoformat(),
        "snapshot_built_date": built.strftime("%Y-%m-%d"),
        "source_commit_short": provenance.get("commit_short"),
        "line": (
            f"Reference year {YEAR} (most recent complete year) · "
            f"snapshot frozen {built.strftime('%d %b %Y')}"
        ),
        "detail": (
            f"Crime counts describe calendar {YEAR} and are not projected forward. "
            f"Geometry is 2020 Census block groups. This snapshot was frozen on "
            f"{built.strftime('%d %b %Y')} from build "
            f"{provenance.get('commit_short') or 'unknown'}."
        ),
    }


def render_methodology() -> dict:
    """Publish docs/METHODOLOGY.md alongside the map as a readable page.

    Every cold reviewer of this surface has said some version of the same thing:
    the documentation is the strongest part of the product and it is nowhere near
    the map. The panel now carries a Methodology link, and a link needs a target
    that exists in the deployed site — so the canonical doc is rendered here, at
    stamp time, from the same commit the snapshot is stamped at.

    Deliberately a ~40-line converter rather than a dependency: the source file is
    16 headings, 15 bullets, one table and prose, with no fenced code, no images
    and no links. Anything richer than that should be a real build step, not a
    silent regex. Unhandled constructs degrade to paragraphs, never to markup.
    """
    if not METHODOLOGY_SRC.exists():
        return {"published": False, "reason": f"{METHODOLOGY_SRC} not found"}
    src = METHODOLOGY_SRC.read_text()

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def inline(s: str) -> str:
        out, parts = [], esc(s).split("`")
        for i, part in enumerate(parts):
            out.append(f"<code>{part}</code>" if i % 2 else part)
        return "".join(out)

    html, para, bullets, table = [], [], [], []

    def flush():
        if para:
            html.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if bullets:
            html.append("<ul>" + "".join(f"<li>{inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()
        if table:
            # A markdown table's second row is the alignment rule; drop it and
            # treat the first row as the header.
            rows = [r for r in table if not set(r.replace("|", "").strip()) <= set("-: ")]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            head = "".join(f"<th>{inline(c)}</th>" for c in cells[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                for r in cells[1:]
            )
            html.append(
                f'<div class="tw"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>"
            )
            table.clear()

    for line in src.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
        elif s.startswith("|"):
            if para or bullets:
                flush()
            table.append(s)
        elif s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            html.append(f"<h{min(level, 4)}>{inline(s.lstrip('#').strip())}</h{min(level, 4)}>")
        elif s.lstrip().startswith(("- ", "* ")):
            if para or table:
                flush()
            bullets.append(s.lstrip()[2:])
        elif bullets:
            # A wrapped continuation of the bullet above, not a new paragraph.
            # Markdown hard-wraps its list items, so treating every unindented
            # follow-on line as prose split each bullet in half mid-sentence.
            bullets[-1] += " " + s.strip()
        else:
            if table:
                flush()
            para.append(s.strip())
    flush()

    page = METHODOLOGY_PAGE_TEMPLATE.replace("__BODY__", "\n".join(html))
    out = OUT_DIR / METHODOLOGY_PAGE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return {
        "published": True,
        "href": METHODOLOGY_PAGE,
        "source": str(METHODOLOGY_SRC.relative_to(REPO)),
        "bytes": out.stat().st_size,
    }


def git_head() -> str:
    """HEAD at STAMP TIME. A note about when this artifact was written — never a
    claim about which commit the data came from (see source_data_provenance)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def source_data_provenance() -> dict:
    """The commit the SOURCE DATA was built at, read from the candidate manifest.

    The build-outputs run that wrote the source parquet also wrote
    `manifest.json` next to it, recording `run.git.head_sha` (plus whether the
    tree was dirty) and, in `output_file_stats`, the path and size of every
    parquet it produced. Binding the manifest to the file being frozen by path +
    size is what makes the commit PROVABLE here: the alternative (`git rev-parse
    HEAD` at stamp time) describes the repo now, not the run that produced the
    numbers, and was one commit behind on the audited snapshot.

    Every check is reported rather than raised: a snapshot frozen from a parquet
    with no manifest is still a legitimate snapshot, it just cannot claim a
    source commit — so it says so instead of inventing one.
    """
    src_bg = REPO / SOURCE_PARQUET
    src_tract = REPO / SOURCE_TRACT_PARQUET
    manifest_path = src_bg.parent / "manifest.json"
    out: dict = {
        "basis": "candidate run manifest run.git.head_sha",
        "manifest": (
            str(manifest_path.relative_to(REPO))
            if manifest_path.is_relative_to(REPO)
            else str(manifest_path)
        ),
        "commit": None,
        "commit_short": None,
        "source_tree_dirty": None,
        "run_id": None,
        "run_created_at_utc": None,
        "manifest_binds_source_parquet": False,
        "warnings": [],
    }
    if not manifest_path.exists():
        out["warnings"].append(
            f"no candidate manifest at {out['manifest']}: the source commit cannot be proved"
        )
        return out
    try:
        man = json.loads(manifest_path.read_text())
    except Exception as exc:  # unreadable manifest is a provenance gap, not a build failure
        out["warnings"].append(f"candidate manifest unreadable: {exc}")
        return out

    run = man.get("run") or {}
    git = run.get("git") or {}
    out["commit"] = git.get("head_sha")
    out["commit_short"] = git.get("short_sha")
    out["source_tree_dirty"] = git.get("dirty")
    out["run_id"] = run.get("run_id")
    out["run_created_at_utc"] = run.get("created_at_utc")
    if not out["commit"]:
        out["warnings"].append("candidate manifest carries no run.git.head_sha")

    # Bind the manifest to the exact files being frozen: same path, same size.
    stats = man.get("output_file_stats") or {}
    bound = []
    for label, path in (("block_group_ags_core", src_bg), ("tract_ags_core", src_tract)):
        rec = stats.get(label) or {}
        rec_path, rec_size = rec.get("path"), rec.get("size_bytes")
        if not path.exists():
            out["warnings"].append(f"source parquet missing on disk: {path}")
            continue
        if rec_path is None or rec_size is None:
            out["warnings"].append(f"manifest has no output_file_stats.{label} to bind against")
            continue
        if Path(rec_path).resolve() != path.resolve():
            out["warnings"].append(
                f"manifest {label} path {rec_path} is not the frozen source {path}"
            )
            continue
        actual = path.stat().st_size
        if int(rec_size) != actual:
            out["warnings"].append(
                f"{label} size {actual:,} != manifest {int(rec_size):,} — the file changed "
                "since that run, so its commit does not describe it"
            )
            continue
        bound.append(label)
    out["manifest_binds_source_parquet"] = bound == ["block_group_ags_core", "tract_ags_core"]
    out["bound_outputs"] = bound
    if out["source_tree_dirty"]:
        out["warnings"].append(
            "the source run's working tree was DIRTY: the commit locates the run but does "
            "not fully reproduce it (see run.git.status_porcelain in the manifest)"
        )
    if not out["manifest_binds_source_parquet"]:
        out["commit_is_proved"] = False
    else:
        out["commit_is_proved"] = bool(out["commit"])
    return out


def main() -> None:
    stats = json.loads((TMP / "index_stats.json").read_text())
    # Encoding decisions declared by 01 (density precision + paint floor, the
    # fixed index break set the pct_above shares were computed at). Passed through
    # to the viewer so its density paint floor and its F-01 break labels come from
    # the build that produced the numbers instead of a hardcoded second copy.
    meta = stats.pop("_meta", None)
    if meta is None:
        raise ValueError(
            "index_stats.json carries no _meta block; re-run 01_extract_indices.py "
            "(it declares the density precision, paint floor and index breaks)."
        )
    manifest = json.loads((TMP / "join_manifest.json").read_text())
    # Fail closed on a partial rebuild: the snapshot's density_floor is what the
    # viewer paints with, so it must match the precision the JOIN (and therefore the
    # tiles) was actually built at. Without this, re-running 01 with a new precision
    # and skipping 03/04 would ship a paint floor the tiles cannot resolve.
    joined_dp = manifest.get("density_dp")
    if joined_dp is None:
        raise ValueError(
            "join_manifest.json carries no density_dp; re-run 03_join_geometry.py "
            "(the tiles' density precision must be recorded before it is stamped)."
        )
    if int(joined_dp) != int(meta["density_dp"]):
        raise ValueError(
            f"density precision mismatch: index_stats.json declares {meta['density_dp']}dp "
            f"but the joined features were written at {joined_dp}dp. Re-run "
            "03_join_geometry.py and 04_make_pmtiles.sh before stamping."
        )
    # Fail closed the same way the density precision does: the snapshot describes
    # a hatch, and the tiles either carry the marks or they do not. 01 counts them
    # on the frames, 03 counts them on the features it actually wrote.
    declared_marked = meta["provenance"]["coverage_block_group"]["any_offense"]["either"]["units"]
    joined_marked = manifest.get("n_provenance_marked_features")
    if joined_marked is None:
        raise ValueError(
            "join_manifest.json carries no n_provenance_marked_features; re-run "
            "03_join_geometry.py (the tiles' provenance marks must be recorded "
            "before the snapshot describes them)."
        )
    if int(joined_marked) != int(declared_marked):
        raise ValueError(
            f"provenance mark mismatch: index_stats.json declares {declared_marked:,} "
            f"marked block groups but the joined features carry {joined_marked:,}. "
            "Re-run 03_join_geometry.py and 04_make_pmtiles.sh before stamping."
        )

    methodology = render_methodology()
    provenance = source_data_provenance()
    # Geometry is full-resolution 2020 TIGER/Line (per-state block-group files),
    # not the generalized cb_*_500k cartographic boundary. 2020 vintage matches
    # the published index GEOIDs (see 03_join_geometry.py).
    vintage = "2020"

    disclaimers = plain_disclaimers(json.loads(DISCLAIMERS.read_text()))

    snapshot = {
        "name": "CrimeRisk frozen snapshot",
        "data_year": YEAR,
        "frozen": True,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        # The commit the DATA was built at, proved against the candidate manifest
        # that produced the source parquet (audit finding P-01). `stamped_at_git_
        # commit` is HEAD when this file was written — a different fact, kept
        # separate so neither can be mistaken for the other.
        "source_git_commit": provenance["commit"],
        "source_data_provenance": provenance,
        "stamped_at_git_commit": git_head(),
        "source_parquet": SOURCE_PARQUET,
        "source_tract_parquet": SOURCE_TRACT_PARQUET,
        "geometry_source": f"Census TIGER/Line {vintage} block groups (full resolution), per-state tl_{vintage}_*_bg",
        "geometry_vintage": vintage,
        "level": "block_group",
        "tiles": f"crimerisk_block_groups_{YEAR}.pmtiles",
        "tile_layer": "blockgroups",
        "feature_id_property": "block_group_geoid",
        "n_features": manifest["n_features"],
        "n_index_features": manifest["n_index_features"],
        "n_index_no_geometry": manifest["n_index_no_geometry"],
        # Tract tiles: murder/rape publish at census-tract support and render from
        # this second archive with the same short tile keys.
        "tract_tiles": f"crimerisk_tracts_{YEAR}.pmtiles",
        "tract_tile_layer": "tracts",
        "tract_feature_id_property": "tract_id",
        "n_tract_features": manifest["n_tract_features"],
        "n_tract_index_features": manifest["n_tract_index_features"],
        "n_tract_index_no_geometry": manifest["n_tract_index_no_geometry"],
        "states": manifest["states"],
        "baseline": 100,
        "baseline_note": "Index 100 = national average for the selected crime and measure. Higher = higher reported crime risk.",
        "default_index": DEFAULT_INDEX_KEY,
        "indices": INDEX_DEFS,
        "offenses": OFFENSE_DEFS,
        "stats": {d["key"]: stats[d["col"]] for d in INDEX_DEFS},
        # Two-axis selector (crime type x measure). All entries point back into the
        # layer keys above; this is just the clearer presentation of them.
        "crime_types": CRIME_TYPE_DEFS,
        "measures": MEASURE_DEFS,
        "default_crime_type": DEFAULT_CRIME_TYPE,
        "default_measure": DEFAULT_MEASURE,
        # Wiring for the new features:
        "layer_offense": LAYER_OFFENSE,          # layer key -> offense (active disclaimer)
        "layer_emode_key": LAYER_EMODE_KEY,      # layer key -> per-offense em_* tile key
        "layer_reliability_key": LAYER_RELIABILITY_KEY,  # layer key -> rt_* tile key (tooltip)
        "special_use_flag_key": "su",            # tract-level rollup tile key
        "density_total_key": "d_tot",
        # Tract-support routing: which layer keys render from the tract tiles, the
        # baked murder/rape popup keys on the block-group tiles, and the tract-scale
        # note shown alongside those values.
        "layer_source": LAYER_SOURCE,            # layer key -> "tract" (else block-group)
        "tract_baked_keys": TRACT_BAKED_KEYS,    # offense suffix -> baked ipt_/rpt_ keys
        "tract_scale_note": TRACT_SCALE_NOTE,
        "estimate_mode_labels": EMODE_LABELS,    # code -> reason text
        "estimate_mode_badges": EMODE_BADGES,    # code -> short badge text
        "estimate_mode_mixed": {                 # aggregate layers with disagreeing members
            "badge": EMODE_MIXED_BADGE,
            "label": EMODE_MIXED_LABEL,
        },
        "reliability_labels": RELIABILITY_LABELS,  # code -> high/medium/low
        # Reliability code that marks the modeled / low incident-support tier.
        "reliability_low_code": 0,
        # Encoding declared by 01: the density paint floor (= the smallest positive
        # density the field can represent at its precision) and the fixed index
        # break set the `pct_above` shares in `stats` were computed at. The viewer
        # takes the floor from here and CHECKS the breaks against its own
        # INDEX_BREAKS before using the shares.
        "density_dp": meta["density_dp"],
        "density_floor": meta["density_floor"],
        "density_stop_keys": meta["density_stop_keys"],
        "index_breaks": meta["index_breaks"],
        "pct_above_basis": meta["pct_above_basis"],
        # Per-cell provenance disclosure: the codes, their wording, the predicates
        # and the national footprint, all authored by 01. The viewer renders these
        # strings; it does not write its own description of what it is hatching.
        "provenance": meta["provenance"],
        "layer_provenance_key": LAYER_PROVENANCE_KEY,
        # National-zoom population-aware emphasis (ramp anchors from 01).
        "population_density": meta["population_density"],
        "population_density_key": "pd",
        # Chrome: one plainly-worded vintage line the panel shows at all times,
        # and the methodology link's target.
        "data_vintage": data_vintage(provenance),
        "methodology": methodology,
        "disclaimers": disclaimers,
    }

    out = OUT_DIR / "snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {out.relative_to(REPO)}")
    print(
        f"  source_git_commit: {snapshot['source_git_commit']} "
        f"(from {provenance['manifest']}, run {provenance['run_id']}; "
        f"proved={provenance.get('commit_is_proved')}, "
        f"source tree dirty={provenance['source_tree_dirty']})"
    )
    print(f"  stamped_at_git_commit: {snapshot['stamped_at_git_commit']}")
    for w in provenance["warnings"]:
        print(f"  PROVENANCE NOTE: {w}")
    print(f"  density: {meta['density_dp']}dp, paint floor {meta['density_floor']:g}")
    print(f"  features: {snapshot['n_features']:,}  geometry: {snapshot['geometry_source']}")
    cov = meta["provenance"]["coverage_block_group"]["any_offense"]
    print(
        f"  provenance marks: {cov['either']['units']:,} block groups "
        f"({cov['either']['pct_units']}% of units, {cov['either']['pct_land']}% of land) "
        f"— imputed {cov['imputed']['units']:,}, "
        f"model-only outlier {cov['model_only_outlier']['units']:,}"
    )
    pdm = meta["population_density"]
    print(
        f"  national-zoom emphasis: {pdm['low_per_sq_mi']:g}-{pdm['high_per_sq_mi']:g}/sq mi, "
        f"floor {pdm['min_emphasis']}"
    )
    print(
        f"  methodology: {methodology.get('href') or 'NOT PUBLISHED'}"
        + (f" ({methodology['bytes'] / 1000:.0f} KB)" if methodology.get("published") else "")
    )
    print(f"  vintage line: {snapshot['data_vintage']['line']}")


if __name__ == "__main__":
    main()
