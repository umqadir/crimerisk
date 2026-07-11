from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import os
from pathlib import Path
import random
import socket
import tempfile
import time
from typing import Iterable
import urllib.error
import urllib.request
import zipfile

import geopandas as gpd
import pandas as pd


TIGER_ROADS_BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2020/ROADS"
LENGTH_CRS = "EPSG:5070"

# TIGER road linear features. The first-pass aggregation focuses on drivable
# classes and leaves pedestrian-only/pathway features out of the main totals.
ROAD_CLASS_BY_MTFCC: dict[str, str] = {
    "S1100": "primary_road",
    "S1200": "secondary_road",
    "S1400": "local_road",
    "S1500": "vehicular_trail",
    "S1630": "ramp",
    "S1640": "service_drive",
    "S1710": "walkway",
    "S1720": "stairway",
    "S1730": "alley",
    "S1740": "private_road",
    "S1750": "internal_census_road",
    "S1780": "parking_lot_road",
    "S1820": "bike_path",
    "S1830": "bridle_path",
    "S2000": "road_median",
}

DRIVABLE_ROAD_CLASSES: frozenset[str] = frozenset(
    {
        "primary_road",
        "secondary_road",
        "local_road",
        "vehicular_trail",
        "ramp",
        "service_drive",
        "alley",
        "private_road",
        "internal_census_road",
        "parking_lot_road",
    }
)

LIMITED_ACCESS_RTTYP: frozenset[str] = frozenset({"I", "U"})
RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({403, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RoadAggregationConfig:
    tiger_year: int = 2020
    length_crs: str = LENGTH_CRS


def tiger_roads_filename(*, county_fips: str, year: int = 2020) -> str:
    county = str(county_fips).zfill(5)
    return f"tl_{int(year)}_{county}_roads.zip"


def tiger_roads_url(*, county_fips: str, year: int = 2020) -> str:
    return f"{TIGER_ROADS_BASE_URL}/{tiger_roads_filename(county_fips=county_fips, year=year)}"


def infer_state_fips_from_bg_dir(bg_dir: Path, *, year: int = 2020) -> list[str]:
    pattern = f"tl_{int(year)}_*_bg.zip"
    return sorted({p.name.split("_")[2] for p in bg_dir.glob(pattern)})


def expected_bg_zip_path(bg_dir: Path, *, state_fips: str, year: int = 2020) -> Path:
    state = str(state_fips).zfill(2)
    return bg_dir / f"tl_{int(year)}_{state}_bg.zip"


def expected_roads_zip_path(roads_dir: Path, *, county_fips: str, year: int = 2020) -> Path:
    return roads_dir / tiger_roads_filename(county_fips=county_fips, year=year)


def validate_tiger_roads_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return False
    required_suffixes = {".shp", ".dbf", ".shx"}
    return all(any(name.lower().endswith(suffix) for name in names) for suffix in required_suffixes)


def infer_county_fips_from_roads_dir(roads_dir: Path, *, year: int = 2020) -> list[str]:
    pattern = f"tl_{int(year)}_*_roads.zip"
    counties: set[str] = set()
    for path in roads_dir.glob(pattern):
        parts = path.stem.split("_")
        if len(parts) >= 4 and parts[2].isdigit():
            counties.add(parts[2].zfill(5))
    return sorted(counties)


def _coerce_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def roads_manifest_entry_is_retryable(entry: dict[str, object] | None) -> bool:
    if not entry:
        return False
    status = str(entry.get("status") or "")
    last_http_status = _coerce_int(entry.get("last_http_status"))
    return status == "failed_transient" or last_http_status in RETRYABLE_HTTP_STATUSES


def load_roads_download_manifest(manifest_path: Path) -> dict[str, dict[str, object]]:
    if not manifest_path.exists():
        return {}
    last_error: Exception | None = None
    for _ in range(5):
        try:
            payload = json.loads(manifest_path.read_text())
            if not isinstance(payload, dict):
                return {}
            return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(0.2)
    if last_error is not None:
        raise last_error
    return {}


def save_roads_download_manifest(manifest_path: Path, manifest: dict[str, dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        dir=manifest_path.parent,
        prefix=f"{manifest_path.stem}.",
        suffix=f"{manifest_path.suffix}.tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(manifest_path)


def build_roads_download_inventory(
    *,
    counties: Iterable[str],
    roads_dir: Path,
    manifest: dict[str, dict[str, object]],
    year: int = 2020,
) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for county_fips in counties:
        county = str(county_fips).zfill(5)
        out_path = expected_roads_zip_path(roads_dir, county_fips=county, year=year)
        has_valid_zip = validate_tiger_roads_zip(out_path)
        entry = dict(manifest.get(county, {}))
        inventory[county] = {
            "county_fips": county,
            "path": out_path,
            "has_valid_zip": has_valid_zip,
            "file_size": int(out_path.stat().st_size) if has_valid_zip else 0,
            "manifest_present": county in manifest,
            "manifest_status": str(entry.get("status")) if entry.get("status") is not None else None,
            "attempt_count": _coerce_int(entry.get("attempt_count")) or 0,
            "last_http_status": _coerce_int(entry.get("last_http_status")),
            "last_error": entry.get("last_error"),
            "is_retryable_failure": (not has_valid_zip) and roads_manifest_entry_is_retryable(entry),
        }
    return inventory


def sync_roads_download_manifest_with_cache(
    *,
    roads_dir: Path,
    manifest_path: Path,
    counties: Iterable[str] | None = None,
    year: int = 2020,
) -> dict[str, dict[str, object]]:
    manifest = load_roads_download_manifest(manifest_path)
    county_values = counties if counties is not None else infer_county_fips_from_roads_dir(roads_dir, year=year)
    changed = False
    timestamp = pd.Timestamp.utcnow().isoformat()

    for county_fips in county_values:
        county = str(county_fips).zfill(5)
        out_path = expected_roads_zip_path(roads_dir, county_fips=county, year=year)
        if not validate_tiger_roads_zip(out_path):
            continue

        existing = dict(manifest.get(county, {}))
        updated = {
            "county_fips": county,
            "year": int(year),
            "url": tiger_roads_url(county_fips=county, year=year),
            "status": "downloaded",
            "attempt_count": _coerce_int(existing.get("attempt_count")) or 0,
            "last_http_status": _coerce_int(existing.get("last_http_status")) if existing.get("status") == "downloaded" else None,
            "last_error": None,
            "file_size": int(out_path.stat().st_size),
            "updated_at": timestamp,
        }
        if any(existing.get(key) != value for key, value in updated.items()):
            manifest[county] = updated
            changed = True

    if changed:
        save_roads_download_manifest(manifest_path, manifest)
    return manifest


def _download(
    url: str,
    out_path: Path,
    *,
    max_attempts: int = 6,
    retry_max_sleep_seconds: float = 3600.0,
) -> tuple[int | None, str | None]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    last_error: Exception | None = None
    last_http_status: int | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with tempfile.NamedTemporaryFile(
                dir=out_path.parent,
                prefix=out_path.name,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/133.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/zip,application/octet-stream,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as resp:
                with tmp_path.open("wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            if not validate_tiger_roads_zip(tmp_path):
                raise OSError(f"Downloaded invalid roads zip from {url}")
            tmp_path.replace(out_path)
            return last_http_status, None
        except urllib.error.HTTPError as exc:
            last_http_status = int(exc.code)
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if exc.code not in RETRYABLE_HTTP_STATUSES or attempt == max_attempts:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
            if retry_after is not None:
                try:
                    sleep_for = min(float(retry_after), retry_max_sleep_seconds)
                except ValueError:
                    sleep_for = min(2 ** (attempt - 1), retry_max_sleep_seconds)
            else:
                sleep_for = min((2 ** (attempt - 1)) + random.random(), retry_max_sleep_seconds)
            time.sleep(sleep_for)
        except (urllib.error.URLError, TimeoutError, socket.timeout, http.client.IncompleteRead, OSError) as exc:
            last_error = exc
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if attempt == max_attempts:
                break
            time.sleep(min((2 ** (attempt - 1)) + random.random(), retry_max_sleep_seconds))
    if last_error is not None:
        raise last_error
    return last_http_status, None


def download_tiger_roads_zip(
    *,
    county_fips: str,
    roads_dir: Path,
    year: int = 2020,
    overwrite: bool = False,
    manifest_path: Path | None = None,
    download_sleep_seconds: float = 0.0,
    max_attempts: int = 6,
    retry_max_sleep_seconds: float = 60.0,
) -> Path:
    county = str(county_fips).zfill(5)
    out_path = expected_roads_zip_path(roads_dir, county_fips=county_fips, year=year)
    manifest = load_roads_download_manifest(manifest_path) if manifest_path is not None else {}
    if out_path.exists() and validate_tiger_roads_zip(out_path) and not overwrite:
        if manifest_path is not None:
            manifest[county] = {
                "county_fips": county,
                "year": int(year),
                "url": tiger_roads_url(county_fips=county, year=year),
                "status": "downloaded",
                "attempt_count": int(manifest.get(county, {}).get("attempt_count", 0)),
                "last_http_status": manifest.get(county, {}).get("last_http_status"),
                "last_error": None,
                "file_size": int(out_path.stat().st_size),
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
            save_roads_download_manifest(manifest_path, manifest)
        return out_path
    if out_path.exists() and not validate_tiger_roads_zip(out_path):
        out_path.unlink(missing_ok=True)
    if download_sleep_seconds > 0:
        time.sleep(download_sleep_seconds)
    url = tiger_roads_url(county_fips=county, year=year)
    try:
        http_status, _ = _download(
            url,
            out_path,
            max_attempts=max_attempts,
            retry_max_sleep_seconds=retry_max_sleep_seconds,
        )
        if manifest_path is not None:
            manifest[county] = {
                "county_fips": county,
                "year": int(year),
                "url": url,
                "status": "downloaded",
                "attempt_count": int(manifest.get(county, {}).get("attempt_count", 0)) + 1,
                "last_http_status": http_status,
                "last_error": None,
                "file_size": int(out_path.stat().st_size) if out_path.exists() else 0,
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
            save_roads_download_manifest(manifest_path, manifest)
    except urllib.error.HTTPError as exc:
        if manifest_path is not None:
            manifest[county] = {
                "county_fips": county,
                "year": int(year),
                "url": url,
                "status": "failed_transient" if exc.code in RETRYABLE_HTTP_STATUSES else "failed_permanent",
                "attempt_count": int(manifest.get(county, {}).get("attempt_count", 0)) + 1,
                "last_http_status": int(exc.code),
                "last_error": str(exc),
                "file_size": int(out_path.stat().st_size) if out_path.exists() else 0,
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
            save_roads_download_manifest(manifest_path, manifest)
        raise
    except Exception as exc:
        if manifest_path is not None:
            manifest[county] = {
                "county_fips": county,
                "year": int(year),
                "url": url,
                "status": "failed_transient",
                "attempt_count": int(manifest.get(county, {}).get("attempt_count", 0)) + 1,
                "last_http_status": manifest.get(county, {}).get("last_http_status"),
                "last_error": str(exc),
                "file_size": int(out_path.stat().st_size) if out_path.exists() else 0,
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
            save_roads_download_manifest(manifest_path, manifest)
        raise
    return out_path


def load_tiger_roads(roads_zip: Path) -> gpd.GeoDataFrame:
    roads = gpd.read_file(
        roads_zip,
        columns=["LINEARID", "FULLNAME", "RTTYP", "MTFCC", "geometry"],
    )
    if roads.empty:
        return gpd.GeoDataFrame(
            columns=["road_segment_id", "road_name", "rttyp", "mtfcc", "road_class", "is_drivable", "is_limited_access", "geometry"],
            geometry="geometry",
            crs="EPSG:4269",
        )
    roads = roads.rename(
        columns={
            "LINEARID": "road_segment_id",
            "FULLNAME": "road_name",
            "RTTYP": "rttyp",
            "MTFCC": "mtfcc",
        }
    ).copy()
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads["road_segment_id"] = roads["road_segment_id"].astype(str)
    roads["road_name"] = roads["road_name"].astype("string")
    roads["rttyp"] = roads["rttyp"].astype("string")
    roads["mtfcc"] = roads["mtfcc"].astype("string")
    roads["road_class"] = roads["mtfcc"].map(ROAD_CLASS_BY_MTFCC).fillna("other_road")
    roads["is_drivable"] = roads["road_class"].isin(DRIVABLE_ROAD_CLASSES)
    roads["is_limited_access"] = roads["rttyp"].isin(LIMITED_ACCESS_RTTYP)
    return roads


def empty_roads_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=[
            "road_segment_id",
            "road_name",
            "rttyp",
            "mtfcc",
            "road_class",
            "is_drivable",
            "is_limited_access",
            "geometry",
        ],
        geometry="geometry",
        crs="EPSG:4269",
    )


def load_block_groups(bg_zip: Path) -> gpd.GeoDataFrame:
    bg = gpd.read_file(bg_zip)
    if bg.empty:
        return gpd.GeoDataFrame(
            columns=["bg_id", "tract_id", "state_fips", "county_fips", "geometry"],
            geometry="geometry",
            crs="EPSG:4269",
        )
    geoid_col = "GEOID20" if "GEOID20" in bg.columns else "GEOID"
    state_col = "STATEFP20" if "STATEFP20" in bg.columns else "STATEFP"
    county_col = "COUNTYFP20" if "COUNTYFP20" in bg.columns else "COUNTYFP"
    keep_cols = [geoid_col, state_col, county_col, "geometry"]
    bg = bg[keep_cols].rename(
        columns={
            geoid_col: "bg_id",
            state_col: "state_fips",
            county_col: "county_fips",
        }
    ).copy()
    bg = bg[bg.geometry.notna() & ~bg.geometry.is_empty].copy()
    bg["bg_id"] = bg["bg_id"].astype(str).str.zfill(12)
    bg["tract_id"] = bg["bg_id"].str.slice(0, 11)
    bg["state_fips"] = bg["state_fips"].astype(str).str.zfill(2)
    bg["county_fips"] = bg["county_fips"].astype(str).str.zfill(3)
    return bg


def infer_county_fips_from_block_groups(block_groups: gpd.GeoDataFrame) -> list[str]:
    if block_groups.empty:
        return []
    return sorted(
        (
            block_groups["state_fips"].astype(str).str.zfill(2)
            + block_groups["county_fips"].astype(str).str.zfill(3)
        ).dropna().unique().tolist()
    )


def classify_road_segments(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if roads.empty:
        return roads.copy()
    out = roads.copy()
    if "road_class" not in out.columns:
        out["road_class"] = out["mtfcc"].map(ROAD_CLASS_BY_MTFCC).fillna("other_road")
    if "is_drivable" not in out.columns:
        out["is_drivable"] = out["road_class"].isin(DRIVABLE_ROAD_CLASSES)
    if "is_limited_access" not in out.columns:
        out["is_limited_access"] = out["rttyp"].isin(LIMITED_ACCESS_RTTYP)
    return out


def _length_cols_for_classes(classes: Iterable[str]) -> list[str]:
    return [f"{road_class}_length_km" for road_class in sorted(set(classes))]


def aggregate_classified_roads_to_block_groups(
    roads: gpd.GeoDataFrame,
    block_groups: gpd.GeoDataFrame,
    *,
    cfg: RoadAggregationConfig = RoadAggregationConfig(),
) -> pd.DataFrame:
    if block_groups.empty:
        return pd.DataFrame(
            columns=[
                "bg_id",
                "tract_id",
                "state_fips",
                "county_fips",
                "road_total_length_km",
                "drivable_road_total_length_km",
                "limited_access_road_length_km",
                "limited_access_road_share",
                *_length_cols_for_classes(ROAD_CLASS_BY_MTFCC.values()),
            ]
        )
    roads = classify_road_segments(roads)
    if roads.empty:
        out = block_groups[["bg_id", "tract_id", "state_fips", "county_fips"]].drop_duplicates().copy()
        zero_cols = [
            "road_total_length_km",
            "drivable_road_total_length_km",
            "limited_access_road_length_km",
            "limited_access_road_share",
            *_length_cols_for_classes(ROAD_CLASS_BY_MTFCC.values()),
        ]
        for col in zero_cols:
            out[col] = 0.0
        return out

    roads_proj = roads.to_crs(cfg.length_crs)
    bg_proj = block_groups.to_crs(cfg.length_crs)
    bg_geom = bg_proj[["bg_id", "geometry"]].rename(columns={"geometry": "bg_geometry"})
    joined = gpd.sjoin(
        roads_proj[["road_segment_id", "road_class", "is_drivable", "is_limited_access", "geometry"]],
        bg_proj[["bg_id", "tract_id", "state_fips", "county_fips", "geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="road",
        rsuffix="bg",
    )
    if joined.empty:
        return aggregate_classified_roads_to_block_groups(
            roads.iloc[0:0].copy(),
            block_groups=block_groups,
            cfg=cfg,
        )

    joined = joined.rename(columns={"index_right": "bg_index"}).merge(bg_geom, on="bg_id", how="left")
    clipped = joined.geometry.intersection(joined["bg_geometry"])
    clipped = gpd.GeoSeries(clipped, crs=cfg.length_crs)
    joined["segment_length_km"] = clipped.length / 1000.0
    joined = joined[joined["segment_length_km"] > 0].copy()
    if joined.empty:
        return aggregate_classified_roads_to_block_groups(
            roads.iloc[0:0].copy(),
            block_groups=block_groups,
            cfg=cfg,
        )

    group_cols = ["bg_id", "tract_id", "state_fips", "county_fips"]
    totals = joined.groupby(group_cols, dropna=False)["segment_length_km"].sum().rename("road_total_length_km")
    out = totals.reset_index()

    class_lengths = (
        joined.pivot_table(
            index=group_cols,
            columns="road_class",
            values="segment_length_km",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(columns=lambda c: f"{c}_length_km")
        .reset_index()
    )
    out = out.merge(class_lengths, on=group_cols, how="left")

    drivable = joined[joined["is_drivable"]].groupby(group_cols, dropna=False)["segment_length_km"].sum()
    out = out.merge(
        drivable.rename("drivable_road_total_length_km").reset_index(),
        on=group_cols,
        how="left",
    )
    limited_access = joined[joined["is_limited_access"]].groupby(group_cols, dropna=False)["segment_length_km"].sum()
    out = out.merge(
        limited_access.rename("limited_access_road_length_km").reset_index(),
        on=group_cols,
        how="left",
    )

    for col in [
        "road_total_length_km",
        "drivable_road_total_length_km",
        "limited_access_road_length_km",
        *_length_cols_for_classes(ROAD_CLASS_BY_MTFCC.values()),
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    limited_share = out["limited_access_road_length_km"] / out["road_total_length_km"]
    out["limited_access_road_share"] = (
        pd.to_numeric(limited_share, errors="coerce")
        .replace([float("inf"), float("-inf")], pd.NA)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )

    # Preserve full BG coverage for downstream joins.
    bg_base = block_groups[["bg_id", "tract_id", "state_fips", "county_fips"]].drop_duplicates().copy()
    out = bg_base.merge(out, on=group_cols, how="left")
    metric_cols = [c for c in out.columns if c not in set(group_cols)]
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out.sort_values(group_cols, kind="mergesort").reset_index(drop=True)


def build_state_block_group_road_metrics(
    *,
    state_fips: str,
    roads_zips: Iterable[Path],
    bg_zip: Path,
    cfg: RoadAggregationConfig = RoadAggregationConfig(),
) -> pd.DataFrame:
    state = str(state_fips).zfill(2)
    block_groups = load_block_groups(bg_zip)
    if not block_groups.empty:
        block_groups = block_groups[block_groups["state_fips"].eq(state)].copy()
    road_frames = [load_tiger_roads(path) for path in roads_zips if path.exists()]
    if road_frames:
        roads = pd.concat(road_frames, ignore_index=True)
        roads = gpd.GeoDataFrame(roads, geometry="geometry", crs=road_frames[0].crs)
    else:
        roads = empty_roads_frame()
    out = aggregate_classified_roads_to_block_groups(roads, block_groups, cfg=cfg)
    out["state_fips"] = out["state_fips"].astype(str).str.zfill(2)
    return out


def build_roads_frame_by_block_group(
    *,
    roads_dir: Path,
    bg_dir: Path,
    state_fips_values: Iterable[str] | None = None,
    cfg: RoadAggregationConfig = RoadAggregationConfig(),
) -> pd.DataFrame:
    states = list(state_fips_values) if state_fips_values is not None else infer_state_fips_from_bg_dir(bg_dir, year=cfg.tiger_year)
    frames: list[pd.DataFrame] = []
    for state_fips in states:
        bg_zip = expected_bg_zip_path(bg_dir, state_fips=state_fips, year=cfg.tiger_year)
        if not bg_zip.exists():
            raise FileNotFoundError(f"Missing BG zip for state {state_fips}: {bg_zip}")
        counties = infer_county_fips_from_block_groups(load_block_groups(bg_zip))
        roads_zips = [expected_roads_zip_path(roads_dir, county_fips=county_fips, year=cfg.tiger_year) for county_fips in counties]
        missing = [path for path in roads_zips if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing TIGER roads zips for state {state_fips}: {missing[:3]}")
        frames.append(
            build_state_block_group_road_metrics(
                state_fips=state_fips,
                roads_zips=roads_zips,
                bg_zip=bg_zip,
                cfg=cfg,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
