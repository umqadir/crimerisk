from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUARANTINE_PATH = REPO_ROOT / "configs" / "city_feed_coordinate_quarantine.csv"
DEFAULT_TEXTURE_POLICY_PATH = REPO_ROOT / "configs" / "city_offense_texture_policy.csv"


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
    out["city_key"] = out["city_key"].astype(str).str.strip().str.lower()
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
    if frame.empty or lat_col not in frame.columns or lon_col not in frame.columns:
        return pd.Series(False, index=frame.index)
    quarantine = load_coordinate_quarantine(config_path)
    quarantine = quarantine[quarantine["city_key"].eq(str(city_key).strip().lower())].copy()
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
    out["city_key"] = out["city_key"].astype(str).str.strip().str.lower()
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
    ckey = str(city_key).strip().lower()
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


# Canonical city_name -> texture-policy city_key. Kept here so both the direct
# (city_shares) and validation (build_next_phase_validation_shares) surfaces map
# a share-surface row's ``city_name`` to the same policy key. St. Louis's direct
# pipeline key is ``st_louis_mo`` but the policy config uses ``st_louis``.
CITY_NAME_TO_TEXTURE_KEY = {
    "austin": "austin",
    "baltimore": "baltimore",
    "boston": "boston",
    "charlotte": "charlotte",
    "chicago": "chicago",
    "cincinnati": "cincinnati",
    "dallas": "dallas",
    "denver": "denver",
    "durham": "durham",
    "kansas city": "kansas_city",
    "mesa": "mesa",
    "milwaukee": "milwaukee",
    "minneapolis": "minneapolis",
    "montgomery county, md": "montgomery_county",
    "new york": "new_york",
    "philadelphia": "philadelphia",
    "san diego": "san_diego",
    "san francisco": "san_francisco",
    "seattle": "seattle",
    "st. louis": "st_louis",
    "st louis": "st_louis",
    "tucson": "tucson",
    "washington": "washington_dc",
}


def resolve_texture_key(city_name: str) -> str:
    key = str(city_name).strip().lower()
    return CITY_NAME_TO_TEXTURE_KEY.get(key, key.replace(" ", "_").replace(",", "").replace(".", ""))


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
