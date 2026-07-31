from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUARANTINE_PATH = REPO_ROOT / "configs" / "city_feed_coordinate_quarantine.csv"
DEFAULT_TEXTURE_POLICY_PATH = REPO_ROOT / "configs" / "city_offense_texture_policy.csv"


# --------------------------------------------------------------- ONE key vocabulary
#
# Stage 4 screen b3 found three key vocabularies for the same 13 cities: the packet gate and
# the coordinate quarantine keyed on the pipeline `city_key` (`st_louis_mo`, `washington_dc`),
# the texture policy keyed on the display `city_name` through a name map (`st_louis`), and the
# exact-point QA artifact on a third set produced by a string-mangling fallback
# (`oakland_california`, `houston_texas`). The failure mode is silent: a config row keyed in one
# vocabulary against a city named in another is INERT, not an error.
#
# There is now one canonical vocabulary -- the pipeline `city_key`, i.e. the packet directory
# name -- one alias table, and one resolver that FAILS on an unrecognised key instead of
# mangling it into a new one.

# The cities with a production share builder (`city_shares._city_impls`).
PRODUCTION_CITY_KEYS: tuple[str, ...] = (
    "austin",
    "baltimore",
    "boston",
    "chicago",
    "denver",
    "mesa",
    "minneapolis",
    "new_york",
    "philadelphia",
    "san_francisco",
    "seattle",
    "st_louis_mo",
    "washington_dc",
)

# Cities that appear in a config, a QA artifact or a gate-17 review packet without having a
# production share builder. They are canonical so a quarantine or policy row keyed to them
# validates rather than failing, and so the QA surface speaks the same vocabulary.
NON_PRODUCTION_CITY_KEYS: tuple[str, ...] = (
    "atlanta_ga",
    "aurora_co",
    "baton_rouge_la",
    "charlotte",
    "cincinnati",
    "cleveland_oh",
    "colorado_springs_co",
    "dallas",
    "durham",
    "fort_worth_tx",
    "houston_tx",
    "indianapolis_in",
    "jacksonville_fl",
    "kansas_city",
    "memphis_tn",
    "milwaukee",
    "montgomery_county",
    "oakland_ca",
    "omaha_ne",
    "sacramento_ca",
    "san_diego",
    "tucson",
)

CANONICAL_CITY_KEYS: frozenset[str] = frozenset(PRODUCTION_CITY_KEYS + NON_PRODUCTION_CITY_KEYS)

# Every historical spelling, including the string-mangled forms the QA artifact produced.
CITY_KEY_ALIASES: dict[str, str] = {
    "atlanta": "atlanta_ga",
    "atlanta_georgia": "atlanta_ga",
    "aurora": "aurora_co",
    "aurora_colorado": "aurora_co",
    "baton_rouge": "baton_rouge_la",
    "baton_rouge_louisiana": "baton_rouge_la",
    "cleveland": "cleveland_oh",
    "cleveland_ohio": "cleveland_oh",
    "colorado_springs": "colorado_springs_co",
    "colorado_springs_colorado": "colorado_springs_co",
    "fort_worth": "fort_worth_tx",
    "fort_worth_texas": "fort_worth_tx",
    "houston": "houston_tx",
    "houston_texas": "houston_tx",
    "indianapolis": "indianapolis_in",
    "indianapolis_indiana": "indianapolis_in",
    "jacksonville": "jacksonville_fl",
    "jacksonville_florida": "jacksonville_fl",
    "memphis": "memphis_tn",
    "memphis_tennessee": "memphis_tn",
    "montgomery_county_md": "montgomery_county",
    "oakland": "oakland_ca",
    "oakland_california": "oakland_ca",
    "omaha": "omaha_ne",
    "omaha_nebraska": "omaha_ne",
    "sacramento": "sacramento_ca",
    "sacramento_california": "sacramento_ca",
    "st_louis": "st_louis_mo",
    "saint_louis": "st_louis_mo",
    "washington": "washington_dc",
}

# Display `city_name` (lower-cased) -> canonical key. Replaces CITY_NAME_TO_TEXTURE_KEY; the
# resolver below raises rather than falling through to a mangled default.
CITY_NAME_TO_CITY_KEY: dict[str, str] = {
    "atlanta": "atlanta_ga",
    "atlanta, georgia": "atlanta_ga",
    "austin": "austin",
    "aurora, colorado": "aurora_co",
    "baltimore": "baltimore",
    "baton rouge, louisiana": "baton_rouge_la",
    "boston": "boston",
    "charlotte": "charlotte",
    "chicago": "chicago",
    "cincinnati": "cincinnati",
    "cleveland, ohio": "cleveland_oh",
    "colorado springs, colorado": "colorado_springs_co",
    "dallas": "dallas",
    "denver": "denver",
    "durham": "durham",
    "fort worth, texas": "fort_worth_tx",
    "houston, texas": "houston_tx",
    "indianapolis, indiana": "indianapolis_in",
    "jacksonville, florida": "jacksonville_fl",
    "kansas city": "kansas_city",
    "memphis, tennessee": "memphis_tn",
    "mesa": "mesa",
    "milwaukee": "milwaukee",
    "minneapolis": "minneapolis",
    "montgomery county, md": "montgomery_county",
    "new york": "new_york",
    "oakland, california": "oakland_ca",
    "omaha, nebraska": "omaha_ne",
    "philadelphia": "philadelphia",
    "sacramento, california": "sacramento_ca",
    "san diego": "san_diego",
    "san francisco": "san_francisco",
    "seattle": "seattle",
    "st louis": "st_louis_mo",
    "st. louis": "st_louis_mo",
    "tucson": "tucson",
    "washington": "washington_dc",
}


def _slug(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace(",", "").replace(".", "").replace("-", "_")


def canonical_city_key(value: object, *, field: str = "city_key", strict: bool = True) -> str:
    """Resolve any historical spelling of a city key to the one canonical key.

    `strict=True` (every config load, every builder call) fails closed: a key nobody declared is
    a config typo or a new city, and both need a human decision -- silently inventing a key is
    what made a quarantine row inert. `strict=False` is for read-only QUERIES over an open-ended
    city list (the role inventory sweeps ~50 packet cities), where an unrecognised key simply has
    no policy row and the answer is the same either way.
    """
    slug = _slug(value)
    if slug in CANONICAL_CITY_KEYS:
        return slug
    if slug in CITY_KEY_ALIASES:
        return CITY_KEY_ALIASES[slug]
    if not strict:
        return slug
    raise ValueError(
        f"Unknown city key {value!r} in {field}. Add it to city_feed_quarantine."
        "NON_PRODUCTION_CITY_KEYS (or PRODUCTION_CITY_KEYS) or map it in CITY_KEY_ALIASES; "
        "keys are never derived by string mangling."
    )


def canonical_city_key_for_city_name(city_name: object, *, strict: bool = True) -> str:
    """Resolve a display `city_name` to the one canonical key, failing on an unmapped name."""
    name = str(city_name).strip().lower()
    if name in CITY_NAME_TO_CITY_KEY:
        return CITY_NAME_TO_CITY_KEY[name]
    return canonical_city_key(name, field="city_name", strict=strict)


# ------------------------------------------------- quarantine wiring witness (fail closed)
#
# The quarantine is applied inside each city builder, because that is the only place the raw
# coordinates exist. Stage 4 screen b3: it was wired into 4 of the 13 production builders, and
# nothing detected the other 9 -- a quarantine row keyed to them would simply do nothing. The
# builders now all route through one helper that records the fact, and the share-surface build
# asserts every enabled city declared itself before the surface is assembled.

QUARANTINE_APPLIED = "applied"
QUARANTINE_NOT_APPLICABLE = "not_applicable"

_QUARANTINE_APPLICATION_LOG: dict[str, str] = {}


def reset_quarantine_application_log() -> None:
    _QUARANTINE_APPLICATION_LOG.clear()


def quarantine_application_log() -> dict[str, str]:
    return dict(_QUARANTINE_APPLICATION_LOG)


def record_quarantine_applied(city_key: str) -> None:
    _QUARANTINE_APPLICATION_LOG[canonical_city_key(city_key)] = QUARANTINE_APPLIED


def record_quarantine_not_applicable(city_key: str, *, reason: str) -> None:
    """Declare that a builder carries no coordinates for the quarantine to act on.

    A declared non-applicability is fine: Austin's feed publishes a source block group and never
    a point, so there is no coordinate to quarantine. An UNDECLARED absence is the defect -- that
    is the state the other eight unwired builders were in.
    """
    key = canonical_city_key(city_key)
    if _QUARANTINE_APPLICATION_LOG.get(key) == QUARANTINE_APPLIED:
        return
    _QUARANTINE_APPLICATION_LOG[key] = f"{QUARANTINE_NOT_APPLICABLE}:{reason}"


def assert_quarantine_wired(city_keys: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Fail the build if any enabled city never ran or declared the quarantine filter."""
    log = quarantine_application_log()
    undeclared = sorted(
        canonical_city_key(key) for key in city_keys if canonical_city_key(key) not in log
    )
    if undeclared:
        raise RuntimeError(
            "Coordinate quarantine is not wired into every enabled city builder, so a "
            "quarantine row keyed to one of these cities would be silently inert: "
            f"{undeclared}. Route the builder's geocoded frame through "
            "city_incidents._drop_quarantined_coordinates, or declare the city coordinate-free "
            "with city_feed_quarantine.record_quarantine_not_applicable."
        )
    return log


def load_coordinate_quarantine(config_path: Path | None = None) -> pd.DataFrame:
    path = config_path or DEFAULT_QUARANTINE_PATH
    if not path.exists():
        return pd.DataFrame(
            columns=["city_key", "offense", "lat", "lon", "radius_m", "reason", "evidence_url_note"]
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"city_key", "lat", "lon", "radius_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coordinate quarantine config missing required columns: {sorted(missing)}")
    out = df.copy()
    if "offense" not in out.columns:
        out["offense"] = "any"
    out["city_key"] = [
        canonical_city_key(value, field="city_feed_coordinate_quarantine.city_key")
        for value in out["city_key"]
    ]
    out["offense"] = out["offense"].astype(str).str.strip().str.lower().replace({"": "any"})
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["radius_m"] = pd.to_numeric(out["radius_m"], errors="coerce")
    out = out[
        out["city_key"].ne("")
        & out["lat"].between(-90.0, 90.0)
        & out["lon"].between(-180.0, 180.0)
        & out["radius_m"].gt(0.0)
    ].copy()
    return out.reset_index(drop=True)


def flag_quarantined_coordinates(
    frame: pd.DataFrame,
    *,
    city_key: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    offense: str | None = None,
    offense_col: str | None = None,
    config_path: Path | None = None,
) -> pd.Series:
    """Flag feed rows whose coordinate falls inside a quarantined point radius.

    Quarantine rows may be scoped to a single offense (e.g. hospital rape-report
    points that are real for other offenses). ``offense="any"`` rows apply to all
    offenses. Pass ``offense`` to scope the caller to one offense, or ``offense_col``
    to resolve each row's offense from the frame (per-offense scoped rows then apply
    only to matching rows; ``any`` rows apply to every row).
    """
    key = canonical_city_key(city_key)
    # Recorded before the early exits: the witness is about whether the FILTER RAN for this
    # city, not about whether it happened to match anything today.
    record_quarantine_applied(key)
    if frame.empty or lat_col not in frame.columns or lon_col not in frame.columns:
        return pd.Series(False, index=frame.index)
    quarantine = load_coordinate_quarantine(config_path)
    quarantine = quarantine[quarantine["city_key"].eq(key)].copy()
    if quarantine.empty:
        return pd.Series(False, index=frame.index)

    lat = pd.to_numeric(frame[lat_col], errors="coerce")
    lon = pd.to_numeric(frame[lon_col], errors="coerce")
    valid = lat.notna() & lon.notna()
    flagged = pd.Series(False, index=frame.index)
    if not bool(valid.any()):
        return flagged

    if offense_col is not None and offense_col in frame.columns:
        row_offense = frame[offense_col].astype(str).str.strip().str.lower()
    elif offense is not None:
        row_offense = pd.Series(str(offense).strip().lower(), index=frame.index)
    else:
        row_offense = pd.Series("", index=frame.index)

    lat_rad = np.deg2rad(lat.loc[valid].to_numpy(dtype=float))
    lon_rad = np.deg2rad(lon.loc[valid].to_numpy(dtype=float))
    valid_index = lat.loc[valid].index
    valid_offense = row_offense.loc[valid].to_numpy()
    earth_radius_m = 6_371_008.8
    any_hit = np.zeros(len(valid_index), dtype=bool)
    for row in quarantine.itertuples(index=False):
        q_offense = str(row.offense).strip().lower()
        if q_offense not in ("any", ""):
            offense_applies = valid_offense == q_offense
            if not offense_applies.any():
                continue
        else:
            offense_applies = np.ones(len(valid_index), dtype=bool)
        q_lat = np.deg2rad(float(row.lat))
        q_lon = np.deg2rad(float(row.lon))
        d_lat = lat_rad - q_lat
        d_lon = lon_rad - q_lon
        a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(q_lat) * np.sin(d_lon / 2.0) ** 2
        dist = 2.0 * earth_radius_m * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        any_hit |= (dist <= float(row.radius_m)) & offense_applies
    flagged.loc[valid_index] = any_hit
    return flagged


def quarantine_counts_by_year_offense(
    frame: pd.DataFrame,
    quarantine_mask: pd.Series,
    *,
    count_col: str | None = None,
) -> pd.DataFrame:
    if frame.empty or not bool(quarantine_mask.reindex(frame.index, fill_value=False).any()):
        return pd.DataFrame(columns=["year", "offense", "quarantined_coordinate_count"])
    subset = frame.loc[quarantine_mask.reindex(frame.index, fill_value=False)].copy()
    if count_col and count_col in subset.columns:
        values = pd.to_numeric(subset[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        subset = subset.assign(_quarantine_count_weight=values)
        count_expr = ("_quarantine_count_weight", "sum")
    else:
        subset = subset.assign(_quarantine_count_weight=1.0)
        count_expr = ("_quarantine_count_weight", "sum")
    return (
        subset.groupby(["year", "offense"], dropna=False)
        .agg(quarantined_coordinate_count=count_expr)
        .reset_index()
    )


# --------------------------------------------------------------- texture policy

_TEXTURE_POLICY_REQUIRED = {"city_key", "offense", "policy"}
_DEFAULT_ALLOW = "allow"
_DEFAULT_DENY = "deny"


def load_texture_policy(config_path: Path | None = None) -> pd.DataFrame:
    """Load the per-city-offense located-evidence texture policy.

    Columns: city_key, offense, policy (allow|deny), reason. A special
    ``city_key='*'`` row sets an offense-level default (e.g. rape default-deny),
    overridden by explicit per-city rows.
    """
    path = config_path or DEFAULT_TEXTURE_POLICY_PATH
    if not path.exists():
        return pd.DataFrame(columns=["city_key", "offense", "policy", "reason"])
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = _TEXTURE_POLICY_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Texture policy config missing required columns: {sorted(missing)}")
    out = df.copy()
    # The `*` default row is not a city; every other key resolves through the one vocabulary,
    # so a policy row and a share-surface row can no longer disagree about a city's name.
    out["city_key"] = [
        "*"
        if str(value).strip() == "*"
        else canonical_city_key(value, field="city_offense_texture_policy.city_key")
        for value in out["city_key"]
    ]
    out["offense"] = out["offense"].astype(str).str.strip().str.lower()
    out["policy"] = out["policy"].astype(str).str.strip().str.lower()
    out = out[out["offense"].ne("") & out["policy"].isin([_DEFAULT_ALLOW, _DEFAULT_DENY])].copy()
    return out.reset_index(drop=True)


def texture_policy_allows(
    city_key: str,
    offense: str,
    *,
    config_path: Path | None = None,
    policy: pd.DataFrame | None = None,
) -> bool:
    """Return True if the (city, offense) pair may contribute located point texture.

    Precedence: an explicit (city_key, offense) row wins; then an offense-level
    default row (city_key='*'); otherwise allow.
    """
    table = policy if policy is not None else load_texture_policy(config_path)
    if table.empty:
        return True
    # A QUERY: an unrecognised key has no policy row, so a permissive resolve gives the same
    # answer as failing would, and the role inventory can sweep every packet city.
    ckey = canonical_city_key(city_key, strict=False)
    off = str(offense).strip().lower()
    explicit = table[table["city_key"].eq(ckey) & table["offense"].eq(off)]
    if not explicit.empty:
        return bool(explicit.iloc[-1]["policy"] == _DEFAULT_ALLOW)
    default = table[table["city_key"].eq("*") & table["offense"].eq(off)]
    if not default.empty:
        return bool(default.iloc[-1]["policy"] == _DEFAULT_ALLOW)
    return True


def denied_texture_offenses(
    city_key: str,
    *,
    offenses: tuple[str, ...] | list[str],
    config_path: Path | None = None,
    policy: pd.DataFrame | None = None,
) -> set[str]:
    table = policy if policy is not None else load_texture_policy(config_path)
    return {
        offense
        for offense in offenses
        if not texture_policy_allows(city_key, offense, policy=table)
    }


def filter_denied_texture_rows(
    frame: pd.DataFrame,
    *,
    city_key: str,
    offense_col: str = "offense",
    config_path: Path | None = None,
    policy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drop feed rows whose (city, offense) pair is texture-denied.

    Denied pairs contribute no located share evidence, so the downstream posterior
    falls to the model prior (the alpha machinery's full-prior path). Controls are
    unchanged because they are the jurisdiction totals, not the located feed.
    """
    if frame.empty or offense_col not in frame.columns:
        return frame
    table = policy if policy is not None else load_texture_policy(config_path)
    if table.empty:
        return frame
    row_offense = frame[offense_col].astype(str).str.strip().str.lower()
    denied = denied_texture_offenses(
        city_key,
        offenses=tuple(sorted(row_offense.dropna().unique())),
        policy=table,
    )
    if not denied:
        return frame
    keep = ~row_offense.isin(denied)
    return frame.loc[keep].copy()


def resolve_texture_key(city_name: str, *, strict: bool = True) -> str:
    """Display `city_name` -> the one canonical city key (was a separate third vocabulary)."""
    return canonical_city_key_for_city_name(city_name, strict=strict)


def filter_denied_texture_surface(
    frame: pd.DataFrame,
    *,
    city_name_col: str = "city_name",
    offense_col: str = "offense",
    config_path: Path | None = None,
    policy: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop share-surface rows whose (city, offense) pair is texture-denied.

    Operates on an assembled share surface carrying ``city_name`` and ``offense``
    columns (one row per city/offense/block-group). A denied pair drops every
    block-group share row for that pair, so it contributes no located evidence and
    the posterior falls to the model prior. Returns ``(kept_frame, dropped_frame)``
    so callers can audit the removed groups.
    """
    empty_removed = frame.iloc[0:0].copy()
    if frame.empty or offense_col not in frame.columns or city_name_col not in frame.columns:
        return frame, empty_removed
    table = policy if policy is not None else load_texture_policy(config_path)
    if table.empty:
        return frame, empty_removed
    keys = frame[city_name_col].map(resolve_texture_key)
    offenses = frame[offense_col].astype(str).str.strip().str.lower()
    denied_mask = pd.Series(False, index=frame.index)
    for key in keys.dropna().unique():
        key_denied = denied_texture_offenses(
            key,
            offenses=tuple(sorted(offenses.dropna().unique())),
            policy=table,
        )
        if not key_denied:
            continue
        denied_mask |= keys.eq(key) & offenses.isin(key_denied)
    if not bool(denied_mask.any()):
        return frame, empty_removed
    return frame.loc[~denied_mask].copy(), frame.loc[denied_mask].copy()
