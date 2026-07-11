"""
05 - Stamp the frozen snapshot metadata.

Writes frontend/public/snapshot.json: the manifest the frontend reads at load
time (index list, color-scale stats, attribute key map) PLUS provenance — the
source git commit, source parquet, geometry vintage, and build timestamp.

This is what makes the artifact a deliberate FROZEN 2024 snapshot, decoupled
from the live pipeline. Regenerate by re-running 01..05.

Run:  uv run python frontend/build/05_stamp_snapshot.py
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TMP = REPO / "frontend/tmp"
PUB = REPO / "frontend/public"
DISCLAIMERS = REPO / "scripts/release/assets/methodology_exclusions.json"

# Snapshot data year + source parquet: single point of control in
# snapshot_config.env (see 01). data_year and the PMTiles filename derive from it.
CONFIG = {
    k.strip(): v.strip()
    for k, v in (
        line.split("=", 1)
        for line in (Path(__file__).resolve().parent / "snapshot_config.env").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
}
YEAR = int(CONFIG["CRIMERISK_SNAPSHOT_YEAR"])

# Estimate-mode integer codes (mirror 03 EMODE_CODE) -> human reason shown in the
# popup for special-use / suppressed cells.
EMODE_LABELS = {
    0: "Estimated from official total",
    1: "Too few residents for a per-person rate",
    2: "special-use area",
    3: "Vehicle-based rate not meaningful",
    4: "Too little exposure for a per-person rate",
}

# Reliability-tier integer codes (mirror 03 RELIABILITY_CODE) -> human label shown
# in the popup. `low` = model-only with ~0 incident support (the count-noise tier
# de-emphasized in the choropleth); medium/high are anchored by real incidents.
RELIABILITY_LABELS = {
    2: "high",
    1: "medium",
    0: "low",
}

# Static layer-level note for the four murder/rape index layers (primary +
# per-resident): every published murder/rape block-group cell lands in
# reliability tier "low" (see docs/STATE.md, "Murder/rape display-honesty
# review" — no cell anywhere clears the Poisson CI-width test at medium/high
# for these two offenses). The per-cell popup already carries tier + support +
# an advisory; this is the layer-level heads-up shown regardless of which cell
# a user hovers.
MURDER_RAPE_RELIABILITY_NOTE = (
    "Murder and rape are low-confidence at block-group scale everywhere in "
    "this release; interpret them at tract or city scale."
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
    {"key": "i_tot", "col": "index_total_part1_resident", "label": "Total (resident)", "group": "Aggregate (alt)", "kind": "index"},
    {"key": "i_eq", "col": "index_total_equal_offense", "label": "Total - equal offense", "group": "Aggregate (alt)", "kind": "index"},
    {"key": "i_harm", "col": "index_total_harm", "label": "Total - harm weighted", "group": "Aggregate (alt)", "kind": "index"},
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

# offense key suffix -> tooltip label, for the per-offense breakdown. The slim
# tile carries ip_* (primary index), ir_* (resident index), r_* (primary rate),
# ec_* (expected count), rt_* (reliability tier), es_* (effective support) per
# offense.
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
# data. Each measure carries a plain-language `explain` string the UI shows under
# the controls so it is always obvious what the active number means.
#
# MEASURES, in display order. `kind` selects the color scheme (index = robust
# diverging-about-100 quantile bands; density = log hotspot ramp).
MEASURE_DEFS = [
    {
        "id": "primary",
        "label": "Risk index",
        "kind": "index",
        "explain": (
            "Rate-based index using the people or places most relevant to this crime: "
            "residents, daytime workers, business locations, or registered vehicles. "
            "100 = national average."
        ),
    },
    {
        "id": "resident",
        "label": "Per-resident index",
        "kind": "index",
        "explain": (
            "Rate per resident population. Useful for comparing crimes on the same "
            "resident base, but it can overstate downtowns, parks, malls, and other "
            "visitor-heavy places. 100 = national average."
        ),
    },
    {
        "id": "density",
        "label": "Density (per sq mi)",
        "kind": "density",
        "explain": (
            "Reported incidents per square mile. Shows concentration on the ground, "
            "not per-person risk. Log scale: green = low, red = hotspot."
        ),
    },
]

# Per-measure explanation text for the aggregate TOTAL crime type, which uses the
# published aggregate indexes rather than a single offense's per-exposure index.
TOTAL_MEASURE_EXPLAIN = {
    "primary": (
        "Combined risk index for the seven crime types, weighted by their national "
        "mix of reported incidents. It uses each crime's own rate base, so areas "
        "with almost no residents do not get inflated just because few people live "
        "there. 100 = national average."
    ),
    "resident": (
        "Combined reported-crime index per resident. Useful as a resident-burden "
        "view, but visitor-heavy places can look inflated. 100 = national average."
    ),
    "density": (
        "All reported incidents per square mile. Shows concentration on the ground, "
        "not per-person risk. Log scale: green = low, red = hotspot."
    ),
}

# CRIME TYPES, in display order. Each maps a measure id -> the existing layer key.
# `alt` (Total only) exposes the extra aggregate constructions in a sub-list so the
# main measure picker stays short and clear.
CRIME_TYPE_DEFS = [
    {
        "id": "total",
        "label": "Total",
        "measures": {"primary": "i_evw", "resident": "i_tot", "density": "d_tot"},
        # Plain-language aggregate alternatives (still index 100 = national average).
        "alt": [
            {"key": "i_per", "label": "Personal / violent — per resident"},
            {"key": "i_pro", "label": "Property — per resident"},
            {"key": "i_eq", "label": "Total — equal-offense weighted"},
            {"key": "i_harm", "label": "Total — harm weighted"},
        ],
    },
    {
        "id": "personal",
        "label": "Personal (violent)",
        "measures": {"primary": "i_per", "resident": "i_per"},
    },
    {
        "id": "property",
        "label": "Property",
        "measures": {"primary": "i_pro", "resident": "i_pro"},
    },
    {
        "id": "murder", "label": "Murder",
        "measures": {"primary": "ip_mur", "resident": "ir_mur", "density": "d_mur"},
    },
    {
        "id": "rape", "label": "Rape",
        "measures": {"primary": "ip_rap", "resident": "ir_rap", "density": "d_rap"},
    },
    {
        "id": "robbery", "label": "Robbery",
        "measures": {"primary": "ip_rob", "resident": "ir_rob", "density": "d_rob"},
    },
    {
        "id": "aggravated_assault", "label": "Aggravated assault",
        "measures": {"primary": "ip_agg", "resident": "ir_agg", "density": "d_agg"},
    },
    {
        "id": "burglary", "label": "Burglary",
        "measures": {"primary": "ip_bur", "resident": "ir_bur", "density": "d_bur"},
    },
    {
        "id": "larceny", "label": "Larceny",
        "measures": {"primary": "ip_lar", "resident": "ir_lar", "density": "d_lar"},
    },
    {
        "id": "motor_vehicle_theft", "label": "Motor vehicle theft",
        "measures": {"primary": "ip_mvt", "resident": "ir_mvt", "density": "d_mvt"},
    },
]

# Opening view: total crime on the event-weighted primary risk index.
DEFAULT_CRIME_TYPE = "total"
DEFAULT_MEASURE = "primary"
SOURCE_PARQUET = CONFIG["CRIMERISK_SNAPSHOT_SRC"]


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
                "Murder is shown where the resident or daytime population is large "
                "enough for a rate to mean much."
            ),
        },
        "rape": {
            "excluded_by_default": True,
            "disclaimer": (
                "Rape rates are muted in parks, undeveloped land, and similar "
                "places with very few residents or workers. Use reported counts or "
                "crime density there."
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


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    stats = json.loads((TMP / "index_stats.json").read_text())
    manifest = json.loads((TMP / "join_manifest.json").read_text())
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
        "source_git_commit": git_head(),
        "source_parquet": SOURCE_PARQUET,
        "geometry_source": f"Census TIGER/Line {vintage} block groups (full resolution), per-state tl_{vintage}_*_bg",
        "geometry_vintage": vintage,
        "level": "block_group",
        "tiles": f"crimerisk_block_groups_{YEAR}.pmtiles",
        "tile_layer": "blockgroups",
        "feature_id_property": "block_group_geoid",
        "n_features": manifest["n_features"],
        "n_index_features": manifest["n_index_features"],
        "n_index_no_geometry": manifest["n_index_no_geometry"],
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
        "total_measure_explain": TOTAL_MEASURE_EXPLAIN,
        "default_crime_type": DEFAULT_CRIME_TYPE,
        "default_measure": DEFAULT_MEASURE,
        # Wiring for the new features:
        "layer_offense": LAYER_OFFENSE,          # layer key -> offense (active disclaimer)
        "layer_emode_key": LAYER_EMODE_KEY,      # layer key -> per-offense em_* tile key
        "layer_reliability_key": LAYER_RELIABILITY_KEY,  # layer key -> rt_* tile key (tooltip)
        "special_use_flag_key": "su",            # tract-level rollup tile key
        "density_total_key": "d_tot",
        "estimate_mode_labels": EMODE_LABELS,    # code -> reason text
        "reliability_labels": RELIABILITY_LABELS,  # code -> high/medium/low
        # Reliability code that marks the modeled / low incident-support tier.
        "reliability_low_code": 0,
        "disclaimers": disclaimers,
    }

    out = PUB / "snapshot.json"
    out.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {out.relative_to(REPO)}")
    print(f"  source_git_commit: {snapshot['source_git_commit']}")
    print(f"  features: {snapshot['n_features']:,}  geometry: {snapshot['geometry_source']}")


if __name__ == "__main__":
    main()
