"""Contract-driven city incident feeds (restored generic runner).

A contract in `configs/city_feeds/<city_key>.json` states a feed declaratively: the endpoint and
query, the id/date/coordinate fields, the offense rules, the state(s) whose 2020 block groups the
points are joined to, and the jurisdiction scope. This module runs a contract deterministically;
no model is involved. It is restored from
`archive/2026-09-cull/scripts/diagnostics/run_generic_city_onboarding.py` (download, condition
matching and offense rules keep that runner's semantics) and generalised from one year to a year
range and from one jurisdiction to a scope list.

Entry points:

* `run_contract(contract, raw)` -> one row per counted unit (incident_id, year, offense,
  latitude, longitude); the unit is the contract's `mapping.dedup_unit` (`dedup_unit`).
* `funnel(contract, raw, ...)` -> row counts at each loss stage (raw, mapped, dated, in_years,
  located, in_state, in_footprint, deduplicated), measured before any filter.
* `download(contract, year_start, year_end, cache_dir)` -> the raw frame for those years.
* `replay_check(replayed, oracle)` -> per (year, offense) count deltas and the block-group
  assignment distance against an oracle incident table.
* `truth_rows(incidents, ...)` -> the (year, offense, block_group_geoid, incident_count) surface.
* `contract_share_cities(paths)` -> contract feeds whose admission is a share mode; stage 7 builds
  those beside the hand-written loaders in `city_incidents.py`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from crimerisk.city_feed_quarantine import flag_quarantined_coordinates
from crimerisk.ledger import require_approved

OFFENSES_7 = (
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
)
SHARE_MODES = frozenset({"share_only", "feed_total_candidate"})
# Replay tolerances (design 7.2 step 4).
COUNT_TOLERANCE = 0.01
BG_TOLERANCE = 0.005
# QA thresholds, reused from the generic onboarding runner and the exact-point tripwire.
MIN_MATCH_SHARE = 0.80
RECONCILIATION_TOLERANCE = 0.10
EXACT_POINT_SHARE = 0.005
# De-duplication units a contract declares in `mapping.dedup_unit`: the rows that make one counted
# offense, which must be the unit the feed's official totals count in. `id_fields` name the key the
# unit is stated over.
#   incident_offense - one row per (incident id, mapped offense): `id_fields` identify the incident,
#                      so two NIBRS codes of one incident that map to the same offense count once.
#   victim_offense   - one row per (victim/offense row id, mapped offense): `id_fields` identify a
#                      victim or offense row, so an incident counts once per victim.
#   row              - no de-duplication: every mapped row counts.
#   raw_id           - legacy: one row per `id_fields` key whatever its offense, the first mapped
#                      row kept (the former `dedupe_on_id: true`, which assumed the raw key is the
#                      counting unit). Not a declarable value; see `dedup_unit`.
DEDUP_UNITS = frozenset({"incident_offense", "victim_offense", "row"})
_DEDUP_KEYS = {"incident_offense": ["incident_id", "offense"], "victim_offense": ["incident_id", "offense"],
               "row": None, "raw_id": ["incident_id"]}


def contract_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "configs" / "city_feeds"


def load_contract(path: Path) -> dict:
    require_approved(path)
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ download


def _get_json(url: str, *, attempts: int = 5) -> object:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "crimerisk-city-contracts"}), timeout=300) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url[:200]}: {last}")


def _where(source: dict, year_start: int, year_end: int) -> str:
    template = str(source.get("where") or "")
    return template.format(start=f"{year_start}-01-01", end=f"{year_end + 1}-01-01",
                           year_start=year_start, year_end=year_end)


def _download_socrata(source: dict, where: str) -> pd.DataFrame:
    rows: list[dict] = []
    offset, size = 0, int(source.get("page_size", 50000))
    while True:
        params: dict[str, object] = {"$limit": size, "$offset": offset}
        if source.get("select"):
            params["$select"] = ",".join(source["select"])
        if where:
            params["$where"] = where
        if source.get("order"):
            params["$order"] = source["order"]
        page = _get_json(f"{source['api_endpoint']}?{urlencode(params)}")
        if not isinstance(page, list):
            raise RuntimeError(f"Socrata response was not a row list: {str(page)[:300]}")
        rows.extend(page)
        if len(page) < size:
            break
        offset += size
    return pd.DataFrame(rows)


def _download_arcgis(source: dict, where: str) -> pd.DataFrame:
    base = str(source["api_endpoint"]).rstrip("/")
    query = base if base.endswith("/query") else f"{base}/query"
    rows: list[dict] = []
    offset, size = 0, int(source.get("page_size", 1000))
    while True:
        params = {
            "f": "json",
            "where": where or "1=1",
            "outFields": ",".join(source.get("select") or ["*"]),
            "returnGeometry": "true",
            "outSR": 4326,
            "resultRecordCount": size,
            "resultOffset": offset,
        }
        if source.get("order"):
            params["orderByFields"] = source["order"]
        payload = _get_json(f"{query}?{urlencode(params)}")
        if "error" in payload:
            raise RuntimeError(f"ArcGIS error: {payload['error']}")
        features = payload.get("features", [])
        for feature in features:
            attributes = dict(feature.get("attributes") or {})
            geometry = feature.get("geometry") or {}
            if "x" in geometry and "y" in geometry:
                attributes["__arcgis_longitude"] = geometry["x"]
                attributes["__arcgis_latitude"] = geometry["y"]
            rows.append(attributes)
        if not features or (not payload.get("exceededTransferLimit") and len(features) < size):
            break
        offset += len(features)
    return pd.DataFrame(rows)


def _download_file(url: str) -> pd.DataFrame:
    """A whole-file download: CSV (undecodable bytes replaced), or an Excel workbook when the path
    ends in .xlsx/.xls, reading the sheet named after `#` (the first sheet without one)."""
    path, _, sheet = url.partition("#")
    if path.lower().split("?")[0].endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet or 0, dtype=str)
    return pd.read_csv(path, low_memory=False, encoding_errors="replace")


def download(contract: dict, year_start: int, year_end: int, cache_dir: Path) -> pd.DataFrame:
    """Pull every source of the contract for [year_start, year_end]; cached per source and range."""
    frames = []
    for source in contract["sources"]:
        digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:12]
        path = Path(cache_dir) / f"raw_{source['slug']}_{year_start}_{year_end}_{digest}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            where = _where(source, year_start, year_end)
            kind = source["portal_type"]
            if kind == "socrata":
                frame = _download_socrata(source, where)
            elif kind == "arcgis_feature_service":
                frame = _download_arcgis(source, where)
            elif kind == "csv":
                frame = _download_file(source["api_endpoint"])
            else:
                raise ValueError(f"unsupported portal_type {kind}")
            for column in ("__arcgis_longitude", "__arcgis_latitude"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                frame.to_parquet(path, index=False)
            except Exception:  # noqa: BLE001 - mixed-type portal columns are stored as text
                mixed = [c for c in frame.columns if frame[c].dtype == object]
                frame[mixed] = frame[mixed].astype("string")
                frame.to_parquet(path, index=False)
        frame = frame.copy()
        frame["__source_slug"] = source["slug"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------ mapping


def _condition_mask(frame: pd.DataFrame, condition: dict) -> pd.Series:
    field = str(condition["field"])
    op = str(condition.get("op", "eq"))
    series = frame.get(field, pd.Series("", index=frame.index)).astype("string").fillna("")
    value = str(condition.get("value", ""))
    values = [str(v) for v in condition.get("values", [])]
    if condition.get("case_sensitive"):
        if op == "eq":
            return series.eq(value)
        if op == "in":
            return series.isin(set(values))
    if op == "eq":
        return series.str.upper().eq(value.upper())
    if op == "in":
        return series.str.upper().isin({v.upper() for v in values})
    if op == "contains":
        return series.str.upper().str.contains(value.upper(), regex=False)
    if op == "not_contains":
        return ~series.str.upper().str.contains(value.upper(), regex=False)
    if op == "regex":
        return series.str.contains(value, case=False, regex=True, na=False)
    if op == "not_regex":
        return ~series.str.contains(value, case=False, regex=True, na=False)
    if op == "not_empty":
        return series.str.strip().ne("")
    raise ValueError(f"unsupported condition op {op}")


def _event_dates(raw: pd.Series, date_format: str) -> pd.Series:
    if date_format == "epoch_ms":
        return pd.to_datetime(pd.to_numeric(raw, errors="coerce"), unit="ms", errors="coerce")
    if date_format not in {"", "auto"}:
        return pd.to_datetime(raw, format=date_format, errors="coerce")
    dates = pd.to_datetime(raw, errors="coerce", utc=False, format="mixed")
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().any():
        years = dates.dt.year
        if years.notna().any() and years.dropna().median() < 1990:
            epoch = pd.to_datetime(numeric, unit="ms", errors="coerce")
            if epoch.notna().sum() >= dates.notna().sum():
                dates = epoch
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates


def _point_part(value: object, axis: str) -> float:
    """Latitude or longitude from a point object: GeoJSON `coordinates` ([lon, lat]) when present,
    else `latitude` / `longitude` keys (Socrata location). Accepts a dict or its JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return float("nan")
    if not isinstance(value, dict):
        return float("nan")
    coordinates = value.get("coordinates")
    try:
        if coordinates is not None and len(coordinates) >= 2:
            return float(coordinates[1 if axis == "lat" else 0])
        return float(value.get("latitude" if axis == "lat" else "longitude"))
    except (TypeError, ValueError):
        return float("nan")


def _coordinate(frame: pd.DataFrame, spec: str, axis: str) -> pd.Series:
    """A numeric coordinate column for `axis` ("lat" or "lon").

    Spec forms, alternatives separated by `|` and coalesced per row in order (columns that are not
    in the pull are skipped): `col` a numeric column; `col#point` a point object in `col` (GeoJSON or
    Socrata location); `proj:<CRS>:<x_col>:<y_col>` projected x/y reprojected to WGS84.
    """
    out: pd.Series | None = None
    for option in [n for n in str(spec).split("|") if n]:
        if option.startswith("proj:"):
            _, auth, code, x_col, y_col = option.split(":")
            if x_col not in frame.columns or y_col not in frame.columns:
                continue
            from pyproj import Transformer

            transformer = Transformer.from_crs(f"{auth}:{code}", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(pd.to_numeric(frame[x_col], errors="coerce").to_numpy(),
                                             pd.to_numeric(frame[y_col], errors="coerce").to_numpy())
            values = pd.Series(lat if axis == "lat" else lon, index=frame.index, dtype=float)
        else:
            name, _, part = option.partition("#")
            if name not in frame.columns:
                continue
            if not part:
                values = pd.to_numeric(frame[name], errors="coerce").astype(float)
            else:
                values = frame[name].map(lambda v: _point_part(v, axis)).astype(float)
        out = values if out is None else out.fillna(values)
    if out is None:
        raise KeyError(f"none of {spec} in the pull")
    return out


def _coalesce(frame: pd.DataFrame, spec: str) -> pd.Series:
    """`a|b`: per row, the first non-empty value among the listed columns present in the pull."""
    out: pd.Series | None = None
    for name in [n for n in str(spec).split("|") if n]:
        if name not in frame.columns:
            continue
        values = frame[name].where(frame[name].astype("string").str.strip().ne(""))
        out = values if out is None else out.where(out.notna(), values)
    if out is None:
        raise KeyError(f"none of {spec} in the pull")
    return out


def map_rows(contract: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """Every raw row with its mapped fields and no row dropped: year (NA when the date does not
    parse), incident_id, offense (NA when no rule matches), latitude, longitude. `run_contract`
    filters this frame; `funnel` counts it."""
    mapping = contract["mapping"]
    frame = raw.copy()
    if mapping.get("year_field"):
        frame["year"] = pd.to_numeric(_coalesce(frame, mapping["year_field"]), errors="coerce").astype("Int64")
    else:
        frame["year"] = _event_dates(_coalesce(frame, mapping["date_field"]), mapping.get("date_format", "auto")).dt.year.astype("Int64")
    parts = [raw[f].astype("string").fillna("").str.strip() for f in mapping["id_fields"]]
    frame["incident_id"] = parts[0]
    for part in parts[1:]:
        frame["incident_id"] = frame["incident_id"] + "|" + part
    # Rule conditions read the pull's own columns, so a source column named `offense`, `year` or
    # `incident_id` is not shadowed by the derived one; the derived year and id stay readable
    # under those names when the pull has no such column.
    fields = raw.copy()
    for derived in ("year", "incident_id"):
        if derived not in raw.columns:
            fields[derived] = frame[derived]
    frame["offense"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for rule in mapping["offense_rules"]:
        if rule["offense"] not in OFFENSES_7:
            raise ValueError(f"unsupported offense {rule['offense']}")
        mask = pd.Series(True, index=frame.index)
        for condition in rule.get("conditions", []):
            mask &= _condition_mask(fields, condition)
        for condition in rule.get("exclude_conditions", []):
            mask &= ~_condition_mask(fields, condition)
        frame.loc[mask & frame["offense"].isna(), "offense"] = rule["offense"]
    frame["latitude"] = _coordinate(frame, mapping["latitude_field"], "lat")
    frame["longitude"] = _coordinate(frame, mapping["longitude_field"], "lon")
    return frame


def _has_point(frame: pd.DataFrame) -> pd.Series:
    return frame["latitude"].notna() & frame["longitude"].notna() & ~(frame["latitude"].eq(0) & frame["longitude"].eq(0))


def dedup_unit(contract: dict) -> tuple[str, bool]:
    """(unit, declared): the contract's de-duplication unit and whether the contract states it.

    A contract declares `mapping.dedup_unit` as one of `DEDUP_UNITS`. A contract that predates the
    field keeps its former behaviour, undeclared: `raw_id` under `dedupe_on_id` (default true),
    `row` when `dedupe_on_id` is false. Declaring both fields is an error, so the unit has one owner.
    """
    mapping = contract["mapping"]
    if "dedup_unit" in mapping:
        if "dedupe_on_id" in mapping:
            raise ValueError(f"{contract.get('city_key')}: declare dedup_unit or dedupe_on_id, not both")
        unit = mapping["dedup_unit"]
        if unit not in DEDUP_UNITS:
            raise ValueError(f"{contract.get('city_key')}: dedup_unit {unit!r} is not one of {sorted(DEDUP_UNITS)}")
        return unit, True
    return ("raw_id" if mapping.get("dedupe_on_id", True) else "row"), False


def run_contract(contract: dict, raw: pd.DataFrame, *, year_start: int, year_end: int) -> pd.DataFrame:
    """Map a raw frame to one row per counted unit (`dedup_unit`): incident_id, year, offense,
    latitude, longitude."""
    mapping = contract["mapping"]
    keys = _DEDUP_KEYS[dedup_unit(contract)[0]]
    frame = map_rows(contract, raw)
    frame = frame[frame["year"].between(year_start, year_end)]
    frame = frame[frame["offense"].notna()]
    if mapping.get("require_coordinates", True):
        frame = frame[_has_point(frame)]
    if keys:
        frame = frame.drop_duplicates(keys, keep="first")
    return frame[["incident_id", "year", "offense", "latitude", "longitude"]].reset_index(drop=True)


FUNNEL_STAGES = ("raw", "mapped", "dated", "in_years", "located", "in_state", "in_footprint", "deduplicated")


def funnel(contract: dict, raw: pd.DataFrame, *, year_start: int, year_end: int, tiger_dir: Path,
           crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Row counts at each stage of the loss funnel, measured on the unfiltered pull.

    raw -> mapped (a rule sets one of the seven offenses) -> dated (the date parses) -> in_years
    (inside [year_start, year_end]) -> located (a coordinate pair, not (0, 0)) -> in_state (inside a
    2020 block group of `geography.state_fips`) -> in_footprint (the block group belongs to a
    `geography.jurisdiction_ids` jurisdiction in `crosswalk`) -> deduplicated (one row per counted
    unit of `dedup_unit`). Every stage is a subset of the one before, whatever
    `require_coordinates` says. One row per offense with year "all" (plus an "all" offense row
    whose `raw` is the whole pull), then one row per offense and year from `in_years` on.
    """
    keys = _DEDUP_KEYS[dedup_unit(contract)[0]]
    frame = map_rows(contract, raw)
    flags = pd.DataFrame(index=frame.index)
    flags["mapped"] = frame["offense"].notna()
    flags["dated"] = flags["mapped"] & frame["year"].notna()
    flags["in_years"] = flags["dated"] & frame["year"].between(year_start, year_end).fillna(False).astype(bool)
    flags["located"] = flags["in_years"] & _has_point(frame)
    located = frame[flags["located"]]
    bg = pd.Series(pd.NA, index=frame.index, dtype="string")
    if len(located):
        bg.loc[located.index] = assign_block_groups(located[["latitude", "longitude"]], contract["geography"]["state_fips"],
                                                    tiger_dir, boundary=True)["block_group_geoid"]
    flags["in_state"] = flags["located"] & bg.notna()
    xw = crosswalk[["block_group_geoid", "jurisdiction_id"]].astype(str)
    scope = set(contract["geography"].get("jurisdiction_ids") or [])
    scope_bgs = set(xw.loc[xw["jurisdiction_id"].isin(scope), "block_group_geoid"].str.zfill(12))
    flags["in_footprint"] = flags["in_state"] & bg.fillna("").str.zfill(12).isin(scope_bgs)
    kept = flags["in_footprint"]
    if keys:
        kept = kept & ~(frame.loc[kept, keys].duplicated(keep="first").reindex(frame.index, fill_value=False))
    flags["deduplicated"] = kept
    flags["offense"] = frame["offense"].fillna("(unmapped)").astype(str)
    flags["year"] = frame["year"]
    stages = list(FUNNEL_STAGES[1:])
    by_offense = flags.groupby("offense")[stages].sum().reset_index()
    by_offense = by_offense[by_offense["offense"].ne("(unmapped)")]
    total = pd.DataFrame([{"offense": "all", "raw": len(frame), **flags[stages].sum().to_dict()}])
    out = pd.concat([total, by_offense], ignore_index=True).assign(year="all")
    yearly = flags[flags["in_years"]].groupby(["offense", "year"])[list(FUNNEL_STAGES[3:])].sum().reset_index()
    yearly["year"] = yearly["year"].astype(int).astype(str)
    out = pd.concat([out, yearly], ignore_index=True)
    columns = ["offense", "year", *FUNNEL_STAGES]
    return out.reindex(columns=columns)[columns].astype({c: "Int64" for c in FUNNEL_STAGES})


def assign_block_groups(incidents: pd.DataFrame, state_fips: list[str], tiger_dir: Path, *, boundary: bool = False) -> pd.DataFrame:
    """Point-in-polygon join to TIGER 2020 block groups of the contract's states (`within`).

    With `boundary`, a point `within` no block group but on the edge of one or more (block-rounded
    points snapped to a street centerline, which is often a block-group edge) takes the lowest
    GEOID among the block groups it touches. The E4 replays keep the plain `within` join."""
    import geopandas as gpd

    bgs = pd.concat(
        [gpd.read_file(Path(tiger_dir) / f"tl_2020_{str(st).zfill(2)}_bg.zip")[["GEOID", "geometry"]] for st in state_fips],
        ignore_index=True,
    )
    bgs = gpd.GeoDataFrame(bgs, geometry="geometry").to_crs("EPSG:4326")
    located = incidents["latitude"].notna() & incidents["longitude"].notna()
    points = gpd.GeoDataFrame(
        incidents[located].copy(),
        geometry=gpd.points_from_xy(incidents.loc[located, "longitude"], incidents.loc[located, "latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, bgs, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    if boundary and joined["GEOID"].isna().any():
        edge = gpd.sjoin(points.loc[joined.index[joined["GEOID"].isna()]], bgs, how="inner", predicate="intersects")
        edge = edge.sort_values("GEOID")
        edge = edge[~edge.index.duplicated(keep="first")]
        joined.loc[edge.index, "GEOID"] = edge["GEOID"]
    out = incidents.copy()
    out["block_group_geoid"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[joined.index, "block_group_geoid"] = joined["GEOID"].astype("string")
    return out


# ------------------------------------------------------------------ replay and truth


def replay_check(replayed: pd.DataFrame, oracle: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare a replay against an oracle incident table (both: year, offense, block_group_geoid).

    Counts: per (year, offense) cell, pass when |replay - oracle| <= max(1% of oracle, 1).
    Block groups: total-variation distance between the two (year, offense, block group) count
    tables over located rows, which must be <= 0.5%.
    """
    def cells(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby(["year", "offense"]).size()

    counts = pd.concat([cells(replayed).rename("replay"), cells(oracle).rename("oracle")], axis=1).fillna(0)
    counts["delta"] = counts["replay"] - counts["oracle"]
    counts["relative_delta"] = counts["delta"].abs() / counts["oracle"].replace(0, np.nan)
    counts["pass"] = counts["delta"].abs() <= np.maximum(COUNT_TOLERANCE * counts["oracle"], 1)
    counts = counts.reset_index()

    def bg_table(frame: pd.DataFrame) -> pd.Series:
        located = frame[frame["block_group_geoid"].notna()]
        return located.groupby(["year", "offense", located["block_group_geoid"].astype(str).str.zfill(12)]).size()

    a, b = bg_table(replayed), bg_table(oracle)
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0)
    total = max(float(joined["b"].sum()), 1.0)
    bg_distance = float((joined["a"] - joined["b"]).abs().sum() / 2 / total)
    summary = {
        "cells": int(len(counts)),
        "cells_failing": int((~counts["pass"]).sum()),
        "replay_rows": int(len(replayed)),
        "oracle_rows": int(len(oracle)),
        "total_relative_delta": float(abs(len(replayed) - len(oracle)) / max(len(oracle), 1)),
        "bg_assignment_distance": bg_distance,
        "counts_pass": bool(counts["pass"].all()),
        "bg_pass": bg_distance <= BG_TOLERANCE,
    }
    summary["pass"] = summary["counts_pass"] and summary["bg_pass"]
    return counts, summary


def truth_rows(incidents: pd.DataFrame, *, city_key: str, source_kind: str, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate located incidents to the widened-truth surface layout."""
    located = incidents[incidents["block_group_geoid"].notna()].copy()
    located["block_group_geoid"] = located["block_group_geoid"].astype(str).str.zfill(12)
    out = located.groupby(["year", "offense", "block_group_geoid"]).size().rename("incident_count").reset_index()
    out["source"] = city_key
    out["source_kind"] = source_kind
    # Same convention as the E4 widened surface: a block group split between jurisdictions carries
    # its count into each of them, so every jurisdiction is its own truth cell.
    xw = crosswalk[["block_group_geoid", "jurisdiction_id"]].drop_duplicates()
    return out.merge(xw, on="block_group_geoid", how="left")


def contract_share_cities(paths) -> list[str]:
    """City keys with an approved contract and at least one share-mode admission row.

    Stage 7 builds these through `run_contract` beside the hand-written loaders. A `truth_only`
    admission never enters the share surface.
    """
    root = Path(paths.repo_root)
    admissions_path = root / "configs" / "city_offense_admissions.csv"
    if not admissions_path.exists() or not contract_dir(root).exists():
        return []
    contracts = sorted(p.stem for p in contract_dir(root).glob("*.json"))
    if not contracts:
        return []
    require_approved(admissions_path)
    admissions = pd.read_csv(admissions_path, dtype=str).fillna("")
    share = admissions[admissions["admission_mode"].isin(SHARE_MODES)]
    return [key for key in contracts if key in set(share["city_key"])]


# ------------------------------------------------------------------ stage 7 wiring


def _contract_incidents(*, contract_path: Path, cache_dir: Path, tiger_dir: Path, start_year: int, end_year: int) -> tuple[dict, pd.DataFrame]:
    """The contract's mapped incidents for the years, after the coordinate quarantine, with block groups."""
    contract = load_contract(contract_path)
    unit, declared = dedup_unit(contract)
    print(f"stage 7 contract feed {contract['city_key']}: dedup_unit={unit} "
          f"({'declared' if declared else 'undeclared; legacy default from dedupe_on_id'})", flush=True)
    raw = download(contract, start_year, end_year, cache_dir)
    incidents = run_contract(contract, raw, year_start=start_year, year_end=end_year)
    incidents = incidents[~flag_quarantined_coordinates(incidents, city_key=contract["city_key"], offense_col="offense")]
    return contract, assign_block_groups(incidents, contract["geography"]["state_fips"], tiger_dir)


def _contract_share_surface(*, contract_path: Path, cache_dir: Path, tiger_dir: Path, bg_crosswalk_parquet: Path,
                            start_year: int, end_year: int, force_source_refresh: bool = False) -> pd.DataFrame:
    contract, incidents = _contract_incidents(contract_path=contract_path, cache_dir=cache_dir, tiger_dir=tiger_dir,
                                              start_year=start_year, end_year=end_year)
    crosswalk = pd.read_parquet(bg_crosswalk_parquet, columns=["block_group_geoid", "jurisdiction_id"])
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    rows = truth_rows(incidents, city_key=contract["city_key"], source_kind=contract.get("source_kind", "incident_report"),
                      crosswalk=crosswalk)
    scope = set(contract["geography"]["jurisdiction_ids"])
    rows = rows[rows["jurisdiction_id"].isin(scope)].copy()
    totals = rows.groupby(["year", "offense", "jurisdiction_id"])["incident_count"].transform("sum")
    rows["city_total"] = totals
    rows["share_within_city"] = rows["incident_count"] / totals
    rows["state_fips"] = rows["block_group_geoid"].str[:2]
    rows["geocode_quality_tier"] = "contract_reported_latlon"
    rows["city_name"] = contract["city_name"]
    rows["validation_source_name"] = "; ".join(s["name"] for s in contract["sources"])
    rows["validation_source_url"] = "; ".join(s["source_url"] for s in contract["sources"])
    return rows.drop(columns=["source", "source_kind"])


def _contract_reconciliation(*, contract_path: Path, cache_dir: Path, tiger_dir: Path, bg_crosswalk_parquet: Path,
                             start_year: int, end_year: int, force_source_refresh: bool = False) -> pd.DataFrame:
    """Per year, offense and scope jurisdiction, the counts the city posterior reads for feed quality
    (the `city_incidents` reconciliation columns): mapped rows with a point, rows in a block group of the
    contract's states, and rows in the jurisdiction's block groups (`final_share_count`)."""
    contract, incidents = _contract_incidents(contract_path=contract_path, cache_dir=cache_dir, tiger_dir=tiger_dir,
                                              start_year=start_year, end_year=end_year)
    crosswalk = pd.read_parquet(bg_crosswalk_parquet, columns=["block_group_geoid", "jurisdiction_id"]).drop_duplicates()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype(str).str.zfill(12)
    incidents = incidents.assign(block_group_geoid=incidents["block_group_geoid"].astype("string").str.zfill(12))
    keys = ["year", "offense"]
    mapped = incidents.groupby(keys).size().rename("mapped_offense_count")
    located = incidents[incidents["block_group_geoid"].notna()]
    geocoded = located.groupby(keys).size().rename("geocoded_offense_count")
    frames = []
    for jurisdiction_id in contract["geography"]["jurisdiction_ids"]:
        bgs = set(crosswalk.loc[crosswalk["jurisdiction_id"].eq(jurisdiction_id), "block_group_geoid"])
        inside = located[located["block_group_geoid"].isin(bgs)]
        matched = inside.groupby(keys).size().rename("matched_offense_count")
        table = pd.concat([mapped, geocoded, matched], axis=1).fillna(0).reset_index()
        table["final_share_count"] = table["matched_offense_count"]
        table["jurisdiction_id"] = jurisdiction_id
        state = sorted({b[:2] for b in bgs}) or [str(s).zfill(2) for s in contract["geography"]["state_fips"]]
        table["state_fips"] = state[0]
        frames.append(table)
    out = pd.concat(frames, ignore_index=True)
    out["city_key"] = contract["city_key"]
    out["city_name"] = contract["city_name"]
    return out[["city_key", "city_name", "jurisdiction_id", "state_fips", "year", "offense", "mapped_offense_count",
                "geocoded_offense_count", "matched_offense_count", "final_share_count"]]


def contract_impls(paths) -> dict[str, dict[str, object]]:
    """Stage-7 builder entries for share-admitted contract feeds, in the `_city_impls` shape."""
    root = Path(paths.repo_root)
    impls: dict[str, dict[str, object]] = {}
    for city_key in contract_share_cities(paths):
        contract_path = contract_dir(root) / f"{city_key}.json"
        kwargs = {
            "contract_path": contract_path,
            "cache_dir": Path(paths.data_dir) / "city-incidents" / city_key,
            "tiger_dir": Path(paths.data_dir) / "tiger_bg",
            "bg_crosswalk_parquet": Path(paths.state_dir) / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
            "start_year": 2018,
            "end_year": 2024,
        }
        impls[city_key] = {
            "share_fn": _contract_share_surface,
            "recon_fn": _contract_reconciliation,
            "required_paths": [contract_path],
            "kwargs": kwargs,
        }
    return impls
