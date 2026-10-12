"""Shared field contract for the public CrimeRisk map build (01..06).

One module so the tile writer, the lookup shards, the download package and the
bootstrap manifest cannot drift from each other.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Large generated artifacts never land in the repo. The default is a sibling of the
# repository, derived from the repository's own location, so a checkout anywhere
# works without configuration; set CRIMERISK_TILES_ROOT to put them elsewhere.
TILES_ROOT = Path(
    os.environ.get("CRIMERISK_TILES_ROOT") or (REPO.parent / "crimerisk-tiles")
).expanduser().resolve()
WORK = TILES_ROOT / "work"
DIST = TILES_ROOT / "dist"
LOGS = TILES_ROOT / "logs"


def config() -> dict[str, str]:
    path = Path(__file__).resolve().parent / "snapshot_config.env"
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _root(name: str) -> Path:
    """`state/` and `data/` are read-only; in this worktree the real trees hang
    off a nested symlink (`state/state`, `data/data`) into the main checkout."""
    nested = REPO / name / name
    return nested if nested.exists() else REPO / name


STATE_ROOT = _root("state")
DATA_ROOT = _root("data")


def repo_path(rel: str) -> Path:
    """Resolve a repo-relative path that may live behind the nested symlink."""
    parts = rel.split("/")
    if parts[0] == "state":
        return STATE_ROOT.joinpath(*parts[1:])
    if parts[0] == "data":
        return DATA_ROOT.joinpath(*parts[1:])
    return REPO / rel


CONFIG = config()
YEAR = int(CONFIG["CRIMERISK_SNAPSHOT_YEAR"])
BG_SRC = repo_path(CONFIG["CRIMERISK_SNAPSHOT_SRC"])
TRACT_SRC = repo_path(CONFIG["CRIMERISK_SNAPSHOT_TRACT_SRC"])
POPULATION_COL = f"population_{YEAR}"

OFFENSES = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]

# Short offence keys used in tiles, shards and the client.
SHORT = {
    "murder": "mur",
    "rape": "rap",
    "robbery": "rob",
    "aggravated_assault": "agg",
    "burglary": "bur",
    "larceny": "lar",
    "motor_vehicle_theft": "mvt",
}

# Public display labels (the crime picker, in order).
OFFENSE_LABEL = {
    "murder": "Murder",
    "rape": "Rape",
    "robbery": "Robbery",
    "aggravated_assault": "Aggravated assault",
    "burglary": "Burglary",
    "larceny": "Larceny/theft",
    "motor_vehicle_theft": "Motor vehicle theft",
}

# Composites: public key -> (exposure column, resident column, label)
COMPOSITES = {
    "ov": (
        "multi_offense_relative_score_event_weighted",
        "index_event_burden_resident",
        "Overall",
    ),
    "vi": (
        "multi_offense_relative_score_personal_event_weighted",
        "index_personal_burden_resident",
        "Violent",
    ),
    "pr": (
        "multi_offense_relative_score_property_event_weighted",
        "index_property_burden_resident",
        "Property",
    ),
}

# Offences published only at census-tract support (block-group values are NULL
# by policy: one year of data cannot support a block-group rate).
TRACT_ONLY_OFFENSES = ["murder", "rape"]

# Composite membership by offense name, in crschema.OFFENSES order.
COMPOSITE_MEMBERS = {
    "ov": list(OFFENSES),
    "vi": ["murder", "rape", "robbery", "aggravated_assault"],
    "pr": ["burglary", "larceny", "motor_vehicle_theft"],
}

# --- per-measure publication status -------------------------------------------
# One code per map measure and area, derived in 01 from the parquet's own
# suppression flags (`primary_index_suppressed_<o>`, `index_<o>_resident_suppressed`)
# and the support policy; no rule is re-derived here.
STATUS_PUBLISHED = 0
STATUS_SUPPRESSED = 1          # suppressed_low_denominator
STATUS_TRACT_ONLY = 2          # rare_offense_tract_only (block-group murder / rape)
STATUS_NO_DATA = 3             # no_data
MEASURE_STATUS = {
    STATUS_PUBLISHED: "published",
    STATUS_SUPPRESSED: "suppressed_low_denominator",
    STATUS_TRACT_ONLY: "rare_offense_tract_only",
    STATUS_NO_DATA: "no_data",
}
# Tiles carry all 20 codes in one number: digit i (base 4) is the status of
# MEASURE_ORDER[i]. 4**20 < 2**53, so the value is exact in a JS number and the
# style reads a digit with floor(ms / 4**i) % 4.
STATUS_BASE = 4

# --- tile attribute names (26 per feature, hard cap) ------------------------
# geoid, st, pop, su, src, ms  +  3 composites x 2 measures  +  7 offences x 2
TILE_ID = "geoid"
TILE_STATE = "st"
TILE_POP = "pop"
TILE_SPECIAL = "su"
TILE_SRC_MASK = "src"
TILE_STATUS = "ms"


def tile_exposure_key(key: str) -> str:
    return f"x_{key}"


def tile_resident_key(key: str) -> str:
    return f"r_{key}"


TILE_FIELDS = (
    [TILE_ID, TILE_STATE, TILE_POP, TILE_SPECIAL, TILE_SRC_MASK, TILE_STATUS]
    + [tile_exposure_key(k) for k in COMPOSITES]
    + [tile_resident_key(k) for k in COMPOSITES]
    + [tile_exposure_key(SHORT[o]) for o in OFFENSES]
    + [tile_resident_key(SHORT[o]) for o in OFFENSES]
)
assert len(TILE_FIELDS) == 26, len(TILE_FIELDS)

# --- special use ------------------------------------------------------------
SPECIAL_USE_CODE = {
    "ordinary": 0,
    "park_open_space": 1,
    "industrial_employment": 2,
    "institutional_facility": 3,
    "campus_institution": 4,
    "transient_destination": 5,
    "group_quarters_other": 6,
    "unknown_special_use": 7,
}
SPECIAL_USE_LABEL = {
    1: "Park or open space",
    2: "Industrial or employment area",
    3: "Institutional facility",
    4: "College or campus",
    5: "Transient destination",
    6: "Group quarters",
    7: "Other special-use area",
}

# Card line for a suppressed area: the expected annual count, named by the area type.
COUNT_NOUN = {
    "mur": ("murder", "murders"), "rap": ("rape", "rapes"), "rob": ("robbery", "robberies"),
    "agg": ("aggravated assault", "aggravated assaults"), "bur": ("burglary", "burglaries"),
    "lar": ("larceny or theft", "larcenies and thefts"), "mvt": ("motor vehicle theft", "motor vehicle thefts"),
    "ov": ("offense", "offenses"), "vi": ("violent offense", "violent offenses"),
    "pr": ("property offense", "property offenses"),
}
SUPPRESSED_COUNT_COPY = {
    "ordinary_label": "This area",
    "exposure_tail": "no resident or daytime population large enough for a rate",
    "resident_tail": "too few residents for a per-resident rate",
    "methods": (
        "Where an index is not published, the card still shows the area's expected offenses per year, "
        "named by its area type (for example, \u201cPark or open space: about 15 robberies a year; no "
        "resident or daytime population large enough for a rate\u201d). The count is the model's "
        "estimate; only the rate is withheld."
    ),
}

# --- source mode / reliability ---------------------------------------------
DIRECT_SOURCE_MODES = {"direct_city_incident"}

# Card line 7. The phrase is chosen by the share of the displayed estimate's
# expected count that comes from direct incident records, not by an all-or-
# nothing test: one modeled rare offence should not relabel a composite that is
# 99% direct. `reliability_tier` does not appear on the card - it is "low" for
# 98.6% of cells, including direct-feed cells, so it separated nothing. 2025.1
# withholds it and the p10/p90 range from the public surface as well: held-out
# coverage of the internal intervals is below nominal (docs/EVALUATION.md).
# Card line 6. The denominator the displayed number is divided by, in words. The
# exposure denominators are offense-specific mixtures of modeled population and
# activity proxies, each rescaled so its national total is resident population, so
# the phrase says "modeled" every time and never implies a measured count. The line
# names the denominator, not a unit: the headline is an index, and the line sits
# under the offense count, so "per 100,000" here would read as a rate of that count.
DENOMINATOR_RESIDENT = "Denominator: residents"
DENOMINATOR_EXPOSURE = {
    "murder": "Denominator: people present, modeled",
    "rape": "Denominator: people present, modeled",
    "robbery": "Denominator: people present, modeled",
    "aggravated_assault": "Denominator: people present, modeled",
    "burglary": "Denominator: premises, modeled",
    "larceny": "Denominator: people and destinations present, modeled",
    "motor_vehicle_theft": "Denominator: vehicles present, modeled",
}
# A composite divides each offense by that offense's own denominator, so it has no
# single one of its own.
DENOMINATOR_EXPOSURE_COMPOSITE = (
    "Denominator: each offense's own modeled base (people, premises or vehicles present)"
)

# Card line 8. Whether the jurisdiction total the neighborhood share was cut from
# was filed for the data year or reconstructed. This is a different question from
# the source phrase above, which is about how the total was spread within the
# jurisdiction.
# The total on the map is never the filed count itself: it is the jurisdiction's multi-year level,
# matched to the state estimate. The line says which filing it was built from (2025.3).
LEVEL_TOTAL_REPORTED = "Jurisdiction total: modeled from a complete {year} police filing"
# "Incomplete or missing" covers own-history carry-forward, a pooled peer-unit rate, an
# annualized partial year and benchmark reconciliation, so the phrase does not name a method.
LEVEL_TOTAL_ESTIMATED = "Jurisdiction total: modeled; the {year} police filing was incomplete or missing"
LEVEL_TOTAL_MIXED = "Jurisdiction total: modeled from the {year} police filing, complete for some offenses only"
# The admission status that means the agency filed a complete, accepted year.
LEVEL_REPORTED_STATUS = "valid_complete_year"
LEVEL_REPORTED_REPAIR_MODES = {"none"}

SOURCE_PHRASE_DIRECT = "From local police incident records"
SOURCE_PHRASE_MODELED = "Modeled from jurisdiction totals and regional patterns"
SOURCE_PHRASE_MOSTLY_DIRECT = "Mostly from local police incident records"
SOURCE_PHRASE_MOSTLY_MODELED = "Mostly modeled from jurisdiction totals"
DIRECT_SHARE_THRESHOLD = 0.9

# Source-mode codes carried into the lookup shards.
SOURCE_MODE_CODE = {
    "direct_city_incident": 0,
    "mixed": 1,
    "modeled_transfer": 2,
}

# --- aggregate support floor -----------------------------------------------
# An average built from a single cell is not an average. Summed-count rates
# already stop one 52-resident cell from setting a whole county's colour; this
# floor removes the remaining cases where the aggregate IS one cell. A county
# below it renders as a named neutral grey; a jurisdiction below it has its
# term dropped from the card line.
#
# The unit differs because the constituent cell differs. A county is built from
# its tracts. A jurisdiction is assigned at block-group support, so its cell is
# the block group: a 17,000-resident city with three block groups inside one
# census tract has ample support and must not be suppressed.
SUPPORT_MIN_TRACTS = 2        # counties
SUPPORT_MIN_BLOCK_GROUPS = 2  # jurisdictions
SUPPORT_MIN_POPULATION = 2500
COUNTY_MIN_TRACTS = SUPPORT_MIN_TRACTS
COUNTY_MIN_POPULATION = SUPPORT_MIN_POPULATION
# Achromatic mid-light grey. Minimum CIELab distance to any ramp class is 14.6
# (to "Lower", the nearest), and 15.3 to the water tone, so it reads as an
# absence rather than as a class or as a lake.
NO_ESTIMATE_COLOR = "#bfbcb6"
NO_ESTIMATE_LABEL = "Too few residents for a county average"

# Unpublished areas, at every zoom. Both paint the grey above; a suppressed area
# also carries the suppression hatch, so it never reads as a class and never lets
# the coarser layer underneath show through as if it were this area's value.
STATUS_LEGEND = {
    "suppressed_low_denominator": "Not published: below a denominator or support floor",
    "no_data": "No estimate",
}
# Card copy per status. `{...}` fields are filled in 06 from the floor columns the
# source parquet carries (`*_floor`); a template whose floor is absent falls back
# to its `_generic` twin rather than printing a number the data does not state.
STATUS_CARD = {
    "suppressed_resident": (
        "Not published: fewer than {resident_rate_population_floor_ordinary:,.0f} residents "
        "({resident_rate_population_floor:,.0f} in a special-use area)."
    ),
    "suppressed_resident_generic": "Not published: too few residents for a per-resident rate.",
    "suppressed_exposure": "Not published: the modeled denominator here is below its floor.",
    "suppressed_exposure_composite": (
        "Not published: at least one of its offenses is below its denominator floor here."
    ),
    "suppressed_county": (
        "Not published: a county average needs at least {min_tracts} tracts and "
        "{min_population:,} residents."
    ),
    "tract_only": (
        "{offense}: published for census tracts only. This is the tract that contains "
        "the selected block group."
    ),
    "no_data": "No estimate for this area.",
    "load_error": "This area's data could not be loaded.",
    "retry": "Retry",
    "show_tract": "Show tract estimate",
    "back_to_block_group": "Back to block group",
    "tract_estimate_badge": "Tract estimate",
}

# --- legend -----------------------------------------------------------------
# Seven verbal classes. `lo` is inclusive, `hi` exclusive (None = open ended).
# The ramp is deliberately asymmetric: below-average classes are pale and
# desaturated so the eye lands on the warm end, where the story is. Blue/rust
# stays distinguishable under the common colour-vision deficiencies, and every
# class also carries its words in the legend and the card.
LEGEND = [
    {"lo": 0, "hi": 50, "label": "Less than half the U.S. average", "color": "#9fb8cb"},
    {"lo": 50, "hi": 80, "label": "Lower", "color": "#c9d9e4"},
    {"lo": 80, "hi": 125, "label": "About average", "color": "#f2ede5"},
    {"lo": 125, "hi": 200, "label": "Higher", "color": "#e0a07a"},
    {"lo": 200, "hi": 400, "label": "2-4x average", "color": "#c96b47"},
    {"lo": 400, "hi": 800, "label": "4-8x average", "color": "#9c3f28"},
    {"lo": 800, "hi": None, "label": "8x or more", "color": "#6b1d12"},
]

# --- public download schema -------------------------------------------------
PUBLIC_COMPOSITE_NAMES = {
    ("ov", "exposure"): "overall_crime_exposure",
    ("vi", "exposure"): "violent_crime_exposure",
    ("pr", "exposure"): "property_crime_exposure",
    ("ov", "resident"): "overall_per_resident",
    ("vi", "resident"): "violent_per_resident",
    ("pr", "resident"): "property_per_resident",
}

PUBLIC_OFFENSE_FIELDS = [
    ("expected_count_{o}", "{o}_expected_count"),
    ("primary_denominator_{o}", "{o}_primary_denominator"),
    ("rate_{o}_primary", "{o}_rate_primary"),
    ("index_{o}_primary", "{o}_index_primary"),
    ("rate_{o}_resident", "{o}_rate_resident"),
    ("index_{o}_resident", "{o}_index_resident"),
    ("source_mode_{o}", "{o}_source_mode"),
]

FIPS_TO_USPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}

STATE_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida",
    "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

# Measure keys used across the client, shards and manifest.
MEASURES = ["exposure", "resident"]
MEASURE_LABEL = {"exposure": "Crime exposure", "resident": "Per resident"}
MEASURE_SUBCOPY = {
    "exposure": (
        "Police-recorded crime relative to the U.S., adjusted for the people, "
        "premises or vehicles present. Not an individual's probability."
    ),
    "resident": "Police-recorded crime per resident relative to the U.S.",
}

# Per-index help text (owner D1, 2025.2): what each published number is, matching how it is built.
# One or two sentences each; `{...}` fields are filled from the snapshot by 06_stage_site.py
# (composite weights from national expected counts, the fitted group-quarters premises weight).
_LEVEL_FREQUENT = (
    "The level combines the police agency's 2025 {label_lc} count, where it filed a complete year, with a forecast from its earlier "
    "years ({half_life} half-life), each weighted by its precision, and is matched to the 2025 state "
    "total (the FBI's state estimate plus documented agency corrections); the share within the agency "
    "comes from mapped police incidents in the cities that publish them and from a statistical model "
    "elsewhere."
)
CRIME_HELP = {
    ("rob", "exposure"): (
        "Relative risk of robbery for the people present (residents and daytime population; the jobs term has weight 0), "
        "100 = U.S. average. " + _LEVEL_FREQUENT
    ),
    ("agg", "exposure"): (
        "Relative risk of aggravated assault for the people present (residents, daytime population and "
        "workers), 100 = U.S. average. " + _LEVEL_FREQUENT
    ),
    ("lar", "exposure"): (
        "Relative risk of larceny/theft for the people and places here: half people present, 30% "
        "destination places (shops, restaurants, venues), 20% retail jobs. " + _LEVEL_FREQUENT
    ),
    ("bur", "exposure"): (
        "Relative risk of burglary per premises: households, group-quarters residents ({gq} premises "
        "each), destination places and retail and industrial jobs, weighted by a fit to mapped "
        "burglaries. " + _LEVEL_FREQUENT
    ),
    ("mvt", "exposure"): (
        "Relative risk of vehicle theft per vehicle present: household vehicles plus commuters' cars, "
        "and never fewer than 0.25 vehicles per person present. " + _LEVEL_FREQUENT
    ),
    ("mur", "exposure"): (
        "Murders are too rare for one year's count in a neighborhood to show its risk, so this is a "
        "smoothed estimate for the census tract, per person present: the agency level combines its complete 2025 "
        "count with a forecast from earlier years (one-year half-life) that leans on similar agencies "
        "when the agency is small, each weighted by its precision, and "
        "the tract share comes from the city's own murders since 2018 (10-year half-life) anchored to its pattern of "
        "violent incidents (in the cities that publish incidents) or from a statistical model "
        "elsewhere. Read it as where murder risk is concentrated; in a small town one murder can move "
        "the value several-fold."
    ),
    ("rap", "exposure"): (
        "Rape is also too rare to read from one year, so this is a smoothed census-tract estimate per "
        "person present: the agency level combines its complete 2025 count with a forecast from earlier years "
        "(two-year half-life), each weighted by its precision, and the tract "
        "share comes from a statistical model nearly everywhere. Reporting and definitions of rape "
        "vary across agencies more than for other offenses."
    ),
    ("ov", "exposure"): (
        "The seven offense indexes averaged with weights equal to each offense's share of U.S. "
        "offenses ({weights_ov}), so it mostly reflects the frequent property crimes."
    ),
    ("vi", "exposure"): (
        "Murder, rape, robbery and aggravated assault indexes averaged with weights equal to their "
        "share of U.S. violent offenses ({weights_vi})."
    ),
    ("pr", "exposure"): (
        "Burglary, larceny/theft and vehicle-theft indexes averaged with weights equal to their share "
        "of U.S. property offenses ({weights_pr})."
    ),
}
_RESIDENT_TAIL = (
    " Places with many visitors and few residents, such as shopping districts, score high on this "
    "measure."
)
for _key, _label in (
    ("rob", "robbery"), ("agg", "aggravated assault"), ("lar", "larceny/theft"), ("bur", "burglary"),
    ("mvt", "motor vehicle theft"),
):
    CRIME_HELP[(_key, "resident")] = (
        f"Expected {_label} offenses divided by the people who live here, relative to the U.S."
        + _RESIDENT_TAIL
    )
CRIME_HELP[("mur", "resident")] = (
    "The same smoothed tract murder estimate divided by residents instead of people present."
    + _RESIDENT_TAIL
)
CRIME_HELP[("rap", "resident")] = (
    "The same smoothed tract rape estimate divided by residents instead of people present."
    + _RESIDENT_TAIL
)
for _key, _what in (("ov", "all seven offenses"), ("vi", "the four violent offenses"), ("pr", "the three property offenses")):
    CRIME_HELP[(_key, "resident")] = (
        f"Expected offenses of {_what} added up and divided by residents, relative to the U.S.; "
        "each offense counts by its number of offenses ({weights_" + _key + "})." + _RESIDENT_TAIL
    )
CRIME_HELP_HALF_LIFE = {"rob": "two-year", "agg": "two-year", "lar": "one-year", "bur": "one-year", "mvt": "one-year"}

# The ten crime choices, in picker order. `key` is the tile/shard suffix.
CRIME_ORDER = (
    [{"key": k, "label": COMPOSITES[k][2], "composite": True} for k in ["ov", "vi", "pr"]]
    + [
        {
            "key": SHORT[o],
            "label": OFFENSE_LABEL[o],
            "composite": False,
            "tract_only": o in TRACT_ONLY_OFFENSES,
        }
        for o in OFFENSES
    ]
)


# Canonical order of the 20 map measures, shared by the lookup shards, the
# benchmark files and the client. Never reorder without rebuilding both.
MEASURE_ORDER = sorted(
    [f"x_{k}" for k in COMPOSITES]
    + [f"r_{k}" for k in COMPOSITES]
    + [f"x_{SHORT[o]}" for o in OFFENSES]
    + [f"r_{SHORT[o]}" for o in OFFENSES]
)
