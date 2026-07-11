from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFENSES_7 = (
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
)
RECONCILIATION_TOLERANCE = 0.10
MIN_MATCH_SHARE = 0.80


def _surface_duplicate_key_count(surface: pd.DataFrame) -> int:
    if surface.empty:
        return 0
    key_cols = ["year", "city_key", "jurisdiction_id", "offense", "block_group_geoid"]
    available = [col for col in key_cols if col in surface.columns]
    if len(available) != len(key_cols):
        return -1
    return int(surface.duplicated(available, keep=False).sum())


def _json_safe(value: object) -> object:
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _load_contract(path: Path) -> dict[str, object]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _source_entries(contract: dict[str, object]) -> list[tuple[str | None, dict[str, object]]]:
    """One (slug, source) pair per feed. Single-source contracts get slug None (legacy behavior)."""
    sources = contract.get("sources")
    if sources:
        entries = [(str(source["slug"]), dict(source)) for source in sources]
        slugs = [slug for slug, _ in entries]
        if len(set(slugs)) != len(slugs):
            raise ValueError(f"Duplicate source slugs in contract: {slugs}")
        return entries
    return [(None, dict(contract["source"]))]


def _effective_contract(contract: dict[str, object]) -> dict[str, object]:
    """For multi-source contracts, synthesize a representative top-level `source` dict so every
    downstream step (surface/status/packet naming) behaves exactly like a single-feed city.
    Single-source contracts are returned unchanged (byte-identical legacy behavior)."""
    sources = contract.get("sources")
    if not sources:
        return contract
    names = [str(source["name"]) for source in sources]
    urls = [str(source["source_url"]) for source in sources]
    if len(set(names)) == 1:
        display_name = names[0]
    else:
        prefix = os.path.commonprefix(names).rstrip(" -_:/")
        display_name = f"{prefix} ({len(names)} layers)" if len(prefix) >= 8 else " + ".join(names)
    effective = dict(contract)
    effective["source"] = {
        "name": display_name,
        "source_url": urls[0] if len(set(urls)) == 1 else "; ".join(dict.fromkeys(urls)),
        "api_endpoint": "; ".join(str(source["api_endpoint"]) for source in sources),
        "portal_type": str(sources[0]["portal_type"]),
        "select": sources[0].get("select"),
        "where": str(sources[0].get("where") or ""),
        "order": str(sources[0].get("order") or ""),
        "page_size": int(sources[0].get("page_size", 50000)),
    }
    return effective


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_socrata(contract: dict[str, object], raw_path: Path, *, force_refresh: bool) -> pd.DataFrame:
    if raw_path.exists() and raw_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(raw_path)
    source = contract["source"]
    base_url = str(source["api_endpoint"])
    select = source.get("select")
    where = source.get("where")
    order = source.get("order")
    page_size = int(source.get("page_size", 50000))
    rows: list[dict[str, object]] = []
    offset = 0
    headers = {"User-Agent": "crimerisk-gate15-generic-onboarding"}
    while True:
        params: dict[str, object] = {"$limit": page_size, "$offset": offset}
        if select:
            params["$select"] = ",".join(select) if isinstance(select, list) else str(select)
        if where:
            params["$where"] = str(where)
        if order:
            params["$order"] = str(order)
        request = Request(f"{base_url}?{urlencode(params)}", headers=headers)
        with urlopen(request, timeout=600) as response:
            page = json.load(response)
        if not isinstance(page, list):
            raise RuntimeError(f"Socrata response was not a row list: {page}")
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    frame = pd.DataFrame(rows)
    for coordinate_column in ("__arcgis_longitude", "__arcgis_latitude"):
        if coordinate_column in frame.columns:
            frame[coordinate_column] = pd.to_numeric(frame[coordinate_column], errors="coerce")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(raw_path, index=False)
    return frame


def _download_arcgis_feature_service(contract: dict[str, object], raw_path: Path, *, force_refresh: bool) -> pd.DataFrame:
    if raw_path.exists() and raw_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(raw_path)
    source = contract["source"]
    base_url = str(source["api_endpoint"]).rstrip("/")
    query_url = base_url if base_url.endswith("/query") else f"{base_url}/query"
    select = source.get("select") or ["*"]
    where = str(source.get("where") or "1=1")
    order = str(source.get("order") or "")
    page_size = int(source.get("page_size", 50000))
    out_fields = ",".join(select) if isinstance(select, list) and select else "*"
    rows: list[dict[str, object]] = []
    offset = 0
    headers = {"User-Agent": "crimerisk-generic-city-onboarding"}
    while True:
        params: dict[str, object] = {
            "f": "json",
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": 4326,
            "resultRecordCount": page_size,
            "resultOffset": offset,
        }
        if order:
            params["orderByFields"] = order
        request = Request(f"{query_url}?{urlencode(params)}", headers=headers)
        with urlopen(request, timeout=600) as response:
            payload = json.load(response)
        if "error" in payload:
            raise RuntimeError(f"ArcGIS response error: {payload['error']}")
        features = payload.get("features", [])
        if not isinstance(features, list):
            raise RuntimeError(f"ArcGIS response did not contain a feature list: {payload}")
        for feature in features:
            attributes = dict(feature.get("attributes") or {})
            geometry = feature.get("geometry") or {}
            if "x" in geometry and "y" in geometry:
                attributes["__arcgis_longitude"] = geometry.get("x")
                attributes["__arcgis_latitude"] = geometry.get("y")
            rows.append(attributes)
        if not features:
            break
        if not payload.get("exceededTransferLimit") and len(features) < page_size:
            break
        offset += len(features)
    frame = pd.DataFrame(rows)
    for coordinate_column in ("__arcgis_longitude", "__arcgis_latitude"):
        if coordinate_column in frame.columns:
            frame[coordinate_column] = pd.to_numeric(frame[coordinate_column], errors="coerce")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(raw_path, index=False)
    return frame


def _download_csv(contract: dict[str, object], raw_path: Path, *, force_refresh: bool) -> pd.DataFrame:
    if raw_path.exists() and raw_path.stat().st_size > 0 and not force_refresh:
        return pd.read_parquet(raw_path)
    source = contract["source"]
    csv_url = str(source.get("api_endpoint") or source.get("source_url"))
    frame = pd.read_csv(csv_url, low_memory=False)
    select = source.get("select") or []
    if isinstance(select, list) and select:
        keep = [column for column in select if column in frame.columns]
        frame = frame[keep].copy()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(raw_path, index=False)
    return frame


def _download_source(contract: dict[str, object], raw_path: Path, *, force_refresh: bool) -> pd.DataFrame:
    portal_type = str(contract["source"].get("portal_type", "socrata"))
    if portal_type == "socrata":
        return _download_socrata(contract, raw_path, force_refresh=force_refresh)
    if portal_type == "arcgis_feature_service":
        return _download_arcgis_feature_service(contract, raw_path, force_refresh=force_refresh)
    if portal_type == "csv":
        return _download_csv(contract, raw_path, force_refresh=force_refresh)
    raise ValueError(f"Unsupported portal_type: {portal_type}")


def _condition_mask(frame: pd.DataFrame, condition: dict[str, object]) -> pd.Series:
    field = str(condition["field"])
    op = str(condition.get("op", "eq"))
    series = frame.get(field, pd.Series("", index=frame.index)).astype("string").fillna("")
    if op == "eq":
        return series.str.upper().eq(str(condition.get("value", "")).upper())
    if op == "in":
        values = {str(value).upper() for value in condition.get("values", [])}
        return series.str.upper().isin(values)
    if op == "contains":
        return series.str.upper().str.contains(str(condition.get("value", "")).upper(), regex=False)
    if op == "not_contains":
        return ~series.str.upper().str.contains(str(condition.get("value", "")).upper(), regex=False)
    if op == "regex":
        return series.str.contains(str(condition.get("value", "")), case=False, regex=True, na=False)
    if op == "not_regex":
        return ~series.str.contains(str(condition.get("value", "")), case=False, regex=True, na=False)
    raise ValueError(f"Unsupported condition op: {op}")


def _apply_rules(frame: pd.DataFrame, contract: dict[str, object]) -> pd.DataFrame:
    mapped = frame.copy()
    mapping = contract["mapping"]
    date_field = str(mapping["date_field"])
    id_field = str(mapping["id_field"])
    id_fields = mapping.get("id_fields") or []
    lat_field = str(mapping["latitude_field"])
    lon_field = str(mapping["longitude_field"])
    raw_dates = mapped.get(date_field, pd.Series(pd.NaT, index=mapped.index))
    mapped["event_date"] = pd.to_datetime(raw_dates, errors="coerce")
    numeric_dates = pd.to_numeric(raw_dates, errors="coerce")
    if numeric_dates.notna().any():
        event_years = mapped["event_date"].dt.year
        needs_epoch_ms = event_years.notna().any() and event_years.dropna().median() < 1990
        if needs_epoch_ms:
            epoch_ms_dates = pd.to_datetime(numeric_dates, unit="ms", errors="coerce")
            if epoch_ms_dates.notna().sum() >= mapped["event_date"].notna().sum():
                mapped["event_date"] = epoch_ms_dates
    mapped["year"] = mapped["event_date"].dt.year.astype("Int64")
    mapped = mapped[mapped["year"].eq(int(contract.get("year", 2024)))].copy()
    if isinstance(id_fields, list) and id_fields:
        id_parts = [
            mapped.get(str(field), pd.Series("", index=mapped.index)).astype("string").fillna("").str.strip()
            for field in id_fields
        ]
        mapped["offense_row_id"] = id_parts[0]
        for part in id_parts[1:]:
            mapped["offense_row_id"] = mapped["offense_row_id"] + "|" + part
    else:
        mapped["offense_row_id"] = mapped.get(id_field, pd.Series("", index=mapped.index)).astype("string").str.strip()
    missing_id = mapped["offense_row_id"].isna() | mapped["offense_row_id"].isin(["", "<NA>"])
    if missing_id.any():
        mapped.loc[missing_id, "offense_row_id"] = mapped.loc[missing_id].astype("string").agg("|".join, axis=1)
    mapped = mapped.drop_duplicates("offense_row_id", keep="first").copy()
    mapped["offense"] = None
    for rule in mapping.get("offense_rules", []):
        target = str(rule["offense"])
        if target not in OFFENSES_7:
            raise ValueError(f"Unsupported target offense: {target}")
        mask = pd.Series(True, index=mapped.index)
        for condition in rule.get("conditions", []):
            mask &= _condition_mask(mapped, condition)
        for condition in rule.get("exclude_conditions", []):
            mask &= ~_condition_mask(mapped, condition)
        mapped.loc[mask & mapped["offense"].isna(), "offense"] = target
    mapped = mapped[mapped["offense"].notna()].copy()
    mapped["latitude"] = pd.to_numeric(mapped.get(lat_field), errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped.get(lon_field), errors="coerce")
    bounds = contract.get("geography", {}).get("bounds", {})
    if bounds:
        mapped = mapped[
            mapped["latitude"].between(float(bounds["min_latitude"]), float(bounds["max_latitude"]), inclusive="both")
            & mapped["longitude"].between(float(bounds["min_longitude"]), float(bounds["max_longitude"]), inclusive="both")
        ].copy()
    return mapped


def _load_city_bg(contract: dict[str, object]) -> gpd.GeoDataFrame:
    geography = contract["geography"]
    state_fips = str(geography["state_fips"]).zfill(2)
    jurisdiction_id = str(geography["jurisdiction_id"])
    crosswalk = pd.read_parquet(REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet")
    crosswalk = crosswalk[["block_group_geoid", "jurisdiction_id", "state_fips"]].drop_duplicates().copy()
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(jurisdiction_id)].copy()
    bg = gpd.read_file(REPO_ROOT / "data" / "tiger_bg" / f"tl_2020_{state_fips}_bg.zip")[
        ["GEOID", "geometry"]
    ].rename(columns={"GEOID": "block_group_geoid"})
    bg["block_group_geoid"] = bg["block_group_geoid"].astype("string").str.zfill(12)
    bg = bg.merge(crosswalk, on="block_group_geoid", how="inner")
    return bg.to_crs("EPSG:4326") if bg.crs != "EPSG:4326" else bg


def _build_surface(mapped: pd.DataFrame, contract: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mapped.empty:
        normalized = mapped.copy()
        normalized["city_key"] = contract["city_key"]
        normalized["city_name"] = contract["city_name"]
        normalized["validation_source_name"] = contract["source"]["name"]
        normalized["validation_source_url"] = contract["source"]["source_url"]
        counts = pd.DataFrame(
            columns=[
                "year",
                "offense",
                "block_group_geoid",
                "jurisdiction_id",
                "state_fips",
                "geocode_quality_tier",
                "incident_count",
                "city_total",
                "share_within_city",
                "city_key",
                "city_name",
                "validation_case_type",
                "validation_source_name",
                "validation_source_url",
            ]
        )
        recon = pd.DataFrame(
            {
                "year": [int(contract.get("year", 2024))] * len(OFFENSES_7),
                "offense": list(OFFENSES_7),
                "raw_offense_rows": [0.0] * len(OFFENSES_7),
                "matched_share_rows": [0.0] * len(OFFENSES_7),
                "match_share": [np.nan] * len(OFFENSES_7),
                "city_key": [contract["city_key"]] * len(OFFENSES_7),
                "city_name": [contract["city_name"]] * len(OFFENSES_7),
                "jurisdiction_id": [contract["geography"]["jurisdiction_id"]] * len(OFFENSES_7),
                "validation_case_type": ["generic_worker_municipal_validation_case"] * len(OFFENSES_7),
                "validation_source_name": [contract["source"]["name"]] * len(OFFENSES_7),
                "validation_source_url": [contract["source"]["source_url"]] * len(OFFENSES_7),
            }
        )
        return counts, recon, normalized
    bg = _load_city_bg(contract)
    points = gpd.GeoDataFrame(
        mapped.copy(),
        geometry=gpd.points_from_xy(mapped["longitude"], mapped["latitude"]),
        crs="EPSG:4326",
    )
    matched = gpd.sjoin(points, bg, how="inner", predicate="within")
    matched = pd.DataFrame(matched.drop(columns=["geometry", "index_right"], errors="ignore"))
    matched = matched.drop_duplicates("offense_row_id", keep="first")
    matched["geocode_quality_tier"] = "generic_worker_reported_latlon"
    normalized = matched.copy()
    normalized["city_key"] = contract["city_key"]
    normalized["city_name"] = contract["city_name"]
    normalized["validation_source_name"] = contract["source"]["name"]
    normalized["validation_source_url"] = contract["source"]["source_url"]
    counts = (
        matched.groupby(["year", "offense", "block_group_geoid", "jurisdiction_id", "state_fips", "geocode_quality_tier"], dropna=False)
        .size()
        .rename("incident_count")
        .reset_index()
    )
    totals = counts.groupby(["year", "offense", "jurisdiction_id"], dropna=False)["incident_count"].sum().rename("city_total").reset_index()
    counts = counts.merge(totals, on=["year", "offense", "jurisdiction_id"], how="left")
    counts["share_within_city"] = counts["incident_count"] / counts["city_total"].replace({0: np.nan})
    counts["city_key"] = contract["city_key"]
    counts["city_name"] = contract["city_name"]
    counts["validation_case_type"] = "generic_worker_municipal_validation_case"
    counts["validation_source_name"] = contract["source"]["name"]
    counts["validation_source_url"] = contract["source"]["source_url"]
    recon = (
        mapped.groupby(["year", "offense"], dropna=False).size().rename("raw_offense_rows").reset_index()
        .merge(
            matched.groupby(["year", "offense"], dropna=False).size().rename("matched_share_rows").reset_index(),
            on=["year", "offense"],
            how="left",
        )
    )
    offense_index = pd.DataFrame(
        {
            "year": [int(contract.get("year", 2024))] * len(OFFENSES_7),
            "offense": list(OFFENSES_7),
        }
    )
    recon = offense_index.merge(recon, on=["year", "offense"], how="left")
    recon["raw_offense_rows"] = pd.to_numeric(recon["raw_offense_rows"], errors="coerce").fillna(0.0)
    recon["matched_share_rows"] = pd.to_numeric(recon["matched_share_rows"], errors="coerce").fillna(0.0)
    recon["match_share"] = np.where(recon["raw_offense_rows"].gt(0), recon["matched_share_rows"] / recon["raw_offense_rows"], np.nan)
    recon["city_key"] = contract["city_key"]
    recon["city_name"] = contract["city_name"]
    recon["jurisdiction_id"] = contract["geography"]["jurisdiction_id"]
    recon["validation_case_type"] = "generic_worker_municipal_validation_case"
    recon["validation_source_name"] = contract["source"]["name"]
    recon["validation_source_url"] = contract["source"]["source_url"]
    return counts, recon, normalized


def _status(recon: pd.DataFrame, surface: pd.DataFrame, contract: dict[str, object]) -> pd.DataFrame:
    controls = pd.read_parquet(REPO_ROOT / "state" / "controls" / "jurisdiction_controls_2024.parquet")
    controls = controls[controls["offense"].isin(OFFENSES_7)].copy()
    controls["jurisdiction_id"] = controls["jurisdiction_id"].astype("string")
    controls["offense"] = controls["offense"].astype("string")
    controls["official_total"] = pd.to_numeric(controls["adjusted_count_ags_core"], errors="coerce").fillna(0.0)
    status = recon.merge(
        controls[["jurisdiction_id", "offense", "official_total", "preferred_source", "quality_tier_preferred"]],
        on=["jurisdiction_id", "offense"],
        how="left",
    )
    sums = surface.groupby(["jurisdiction_id", "year", "offense"], dropna=False)["share_within_city"].sum().rename("share_sum").reset_index()
    status = status.merge(sums, on=["jurisdiction_id", "year", "offense"], how="left")
    status["official_total_present"] = status["official_total"].notna()
    status["reconciliation_relative_delta"] = np.where(
        status["official_total"].fillna(0.0).gt(0),
        (status["raw_offense_rows"] - status["official_total"]).abs() / status["official_total"].replace({0: np.nan}),
        np.nan,
    )
    status["reconciles_within_10pct"] = status["official_total_present"] & status["reconciliation_relative_delta"].le(RECONCILIATION_TOLERANCE)
    status["share_geography_pass"] = status["match_share"].ge(MIN_MATCH_SHARE)
    status["share_sum_pass"] = status["share_sum"].sub(1.0).abs().le(1e-6)
    status["share_admitted_flag"] = status["official_total_present"] & status["share_geography_pass"] & status["share_sum_pass"]
    status["feed_total_candidate_flag"] = status["reconciles_within_10pct"]
    status["admitted_flag"] = status["share_admitted_flag"]
    status["total_lane_disposition"] = np.where(status["reconciles_within_10pct"], "feed_total_candidate", "official_control")
    status["share_lane_disposition"] = np.where(status["admitted_flag"], "admit_share_only", "reject_share")
    status["qa_status"] = np.where(status["admitted_flag"], "pass_share_only", "reject")
    status["admission_mode"] = np.select(
        [
            ~status["admitted_flag"],
            status["feed_total_candidate_flag"],
        ],
        [
            "rejected",
            "feed_total_candidate",
        ],
        default="share_only",
    )
    status["reconciliation_disposition"] = np.where(status["reconciles_within_10pct"], "reconciled_within_strict_10pct", "unreconciled_keep_share_only")
    status["worker_judgment_notes"] = str(contract.get("worker_judgment_notes", ""))
    return status.sort_values(["city_key", "offense"], kind="mergesort")


def _failure_status(contract: dict[str, object], error_message: str) -> pd.DataFrame:
    controls = pd.read_parquet(REPO_ROOT / "state" / "controls" / "jurisdiction_controls_2024.parquet")
    controls = controls[controls["offense"].isin(OFFENSES_7)].copy()
    controls["jurisdiction_id"] = controls["jurisdiction_id"].astype("string")
    controls["offense"] = controls["offense"].astype("string")
    controls["official_total"] = pd.to_numeric(controls["adjusted_count_ags_core"], errors="coerce").fillna(0.0)
    status = pd.DataFrame(
        {
            "year": [int(contract.get("year", 2024))] * len(OFFENSES_7),
            "offense": list(OFFENSES_7),
            "raw_offense_rows": [0.0] * len(OFFENSES_7),
            "matched_share_rows": [0.0] * len(OFFENSES_7),
            "match_share": [np.nan] * len(OFFENSES_7),
            "city_key": [contract["city_key"]] * len(OFFENSES_7),
            "city_name": [contract["city_name"]] * len(OFFENSES_7),
            "jurisdiction_id": [contract["geography"]["jurisdiction_id"]] * len(OFFENSES_7),
            "validation_case_type": ["generic_worker_municipal_validation_case"] * len(OFFENSES_7),
            "validation_source_name": [contract["source"]["name"]] * len(OFFENSES_7),
            "validation_source_url": [contract["source"]["source_url"]] * len(OFFENSES_7),
            "share_sum": [np.nan] * len(OFFENSES_7),
        }
    )
    status = status.merge(
        controls[["jurisdiction_id", "offense", "official_total", "preferred_source", "quality_tier_preferred"]],
        on=["jurisdiction_id", "offense"],
        how="left",
    )
    status["official_total_present"] = status["official_total"].notna()
    status["reconciliation_relative_delta"] = np.nan
    status["reconciles_within_10pct"] = False
    status["share_geography_pass"] = False
    status["share_sum_pass"] = False
    status["share_admitted_flag"] = False
    status["feed_total_candidate_flag"] = False
    status["admitted_flag"] = False
    status["total_lane_disposition"] = "official_control"
    status["share_lane_disposition"] = "reject_share"
    status["qa_status"] = "reject"
    status["admission_mode"] = "rejected"
    status["reconciliation_disposition"] = "source_error_reject"
    notes = str(contract.get("worker_judgment_notes", ""))
    status["worker_judgment_notes"] = f"{notes}\nRUNNER_ERROR: {error_message}".strip()
    return status.sort_values(["city_key", "offense"], kind="mergesort")


def _write_packet(
    run_dir: Path,
    contract: dict[str, object],
    raw: pd.DataFrame,
    raw_manifests: list[dict[str, object]],
    surface: pd.DataFrame,
    normalized: pd.DataFrame,
    recon: pd.DataFrame,
    status: pd.DataFrame,
    runtime: dict[str, object],
) -> None:
    city_key = str(contract["city_key"])
    packet = REPO_ROOT / "state" / "review" / "packets" / "city" / city_key
    packet.mkdir(parents=True, exist_ok=True)
    surface.to_parquet(packet / "city_share_surface.parquet", index=False)
    normalized.to_parquet(packet / "normalized_incidents.parquet", index=False)
    status.to_csv(packet / "reconciliation_summary.csv", index=False)
    status[
        [
            "city_key",
            "offense",
            "share_admitted_flag",
            "feed_total_candidate_flag",
            "admitted_flag",
            "qa_status",
            "admission_mode",
            "share_lane_disposition",
            "total_lane_disposition",
        ]
    ].to_csv(packet / "packet_offense_status.csv", index=False)
    pd.DataFrame([
        {
            "offense": rule.get("offense"),
            "conditions_json": json.dumps(rule.get("conditions", []), sort_keys=True),
            "exclude_conditions_json": json.dumps(rule.get("exclude_conditions", []), sort_keys=True),
        }
        for rule in contract["mapping"].get("offense_rules", [])
    ]).to_csv(packet / "offense_crosswalk.csv", index=False)
    pd.DataFrame(
        [{"column": column, "dtype": str(dtype)} for column, dtype in raw.dtypes.items()]
    ).to_csv(packet / "source_schema.csv", index=False)
    status[
        [
            "city_key",
            "city_name",
            "jurisdiction_id",
            "offense",
            "official_total",
            "preferred_source",
            "quality_tier_preferred",
        ]
    ].to_csv(packet / "published_reference_extract.csv", index=False)
    pd.DataFrame([{
        "city_key": city_key,
        "city_name": contract["city_name"],
        "jurisdiction_id": contract["geography"]["jurisdiction_id"],
        "production_ready": "partial" if status["admitted_flag"].any() else "no",
        "city_share_integration_status": "generic_contract_share_pilot",
    }]).to_csv(packet / "packet_status.csv", index=False)
    pd.DataFrame([{
        "city_key": city_key,
        "city_name": contract["city_name"],
        "jurisdiction_id": contract["geography"]["jurisdiction_id"],
        "source_name": contract["source"]["name"],
        "source_url": contract["source"]["source_url"],
        "recommended_disposition": "ready_now" if status["admitted_flag"].any() else "insufficient_quality",
        "analyst_notes": "Generic worker onboarding contract. Incident feed is share lane unless an offense passes the strict reconciliation gate.",
    }]).to_csv(packet / "source_candidate.csv", index=False)
    pd.DataFrame(raw_manifests).to_csv(packet / "raw_source_manifest.csv", index=False)
    manifest = {
        "city_key": city_key,
        "run_dir": str(run_dir),
        "contract": contract,
        "raw_source": raw_manifests[0] if len(raw_manifests) == 1 else raw_manifests,
        "runtime": runtime,
        "qa": {
            "rows_in_surface": int(len(surface)),
            "duplicate_surface_key_rows": _surface_duplicate_key_count(surface),
            "offense_rows": int(len(status)),
            "admitted_rows": int(status["admitted_flag"].sum()),
            "share_admitted_rows": int(status["share_admitted_flag"].sum()),
            "feed_total_candidate_rows": int(status["feed_total_candidate_flag"].sum()),
            "share_sum_pass_all": bool(status["share_sum_pass"].all()),
        },
    }
    (packet / "packet_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    (packet / "qa_summary.json").write_text(json.dumps(_json_safe(manifest["qa"]), indent=2, sort_keys=True))
    (packet / "research_findings.json").write_text(json.dumps(_json_safe(contract), indent=2, sort_keys=True))


def _write_failure_packet(
    run_dir: Path,
    contract: dict[str, object],
    status: pd.DataFrame,
    runtime: dict[str, object],
) -> None:
    city_key = str(contract["city_key"])
    packet = REPO_ROOT / "state" / "review" / "packets" / "city" / city_key
    packet.mkdir(parents=True, exist_ok=True)
    empty_surface = pd.DataFrame(
        columns=[
            "year",
            "offense",
            "block_group_geoid",
            "jurisdiction_id",
            "state_fips",
            "geocode_quality_tier",
            "incident_count",
            "city_total",
            "share_within_city",
            "city_key",
            "city_name",
        ]
    )
    empty_surface.to_parquet(packet / "city_share_surface.parquet", index=False)
    pd.DataFrame().to_parquet(packet / "normalized_incidents.parquet", index=False)
    status.to_csv(packet / "reconciliation_summary.csv", index=False)
    status[
        [
            "city_key",
            "offense",
            "share_admitted_flag",
            "feed_total_candidate_flag",
            "admitted_flag",
            "qa_status",
            "admission_mode",
            "share_lane_disposition",
            "total_lane_disposition",
        ]
    ].to_csv(packet / "packet_offense_status.csv", index=False)
    pd.DataFrame([
        {
            "offense": rule.get("offense"),
            "conditions_json": json.dumps(rule.get("conditions", []), sort_keys=True),
            "exclude_conditions_json": json.dumps(rule.get("exclude_conditions", []), sort_keys=True),
        }
        for rule in contract["mapping"].get("offense_rules", [])
    ]).to_csv(packet / "offense_crosswalk.csv", index=False)
    pd.DataFrame(columns=["column", "dtype"]).to_csv(packet / "source_schema.csv", index=False)
    status[
        [
            "city_key",
            "city_name",
            "jurisdiction_id",
            "offense",
            "official_total",
            "preferred_source",
            "quality_tier_preferred",
        ]
    ].to_csv(packet / "published_reference_extract.csv", index=False)
    pd.DataFrame([{
        "city_key": city_key,
        "city_name": contract["city_name"],
        "jurisdiction_id": contract["geography"]["jurisdiction_id"],
        "production_ready": "no",
        "city_share_integration_status": "generic_contract_share_rejected",
    }]).to_csv(packet / "packet_status.csv", index=False)
    pd.DataFrame([{
        "city_key": city_key,
        "city_name": contract["city_name"],
        "jurisdiction_id": contract["geography"]["jurisdiction_id"],
        "source_name": contract["source"]["name"],
        "source_url": contract["source"]["source_url"],
        "recommended_disposition": "insufficient_quality",
        "analyst_notes": str(runtime.get("error_message", "Generic runner failed before share construction.")),
    }]).to_csv(packet / "source_candidate.csv", index=False)
    raw_manifest = {
        "city_key": city_key,
        "source_name": contract["source"]["name"],
        "source_url": contract["source"]["source_url"],
        "api_endpoint": contract["source"]["api_endpoint"],
        "portal_type": contract["source"]["portal_type"],
        "raw_cache_path": runtime.get("raw_cache_path"),
        "raw_cache_bytes": None,
        "raw_cache_sha256": runtime.get("raw_cache_sha256"),
    }
    pd.DataFrame([raw_manifest]).to_csv(packet / "raw_source_manifest.csv", index=False)
    manifest = {
        "city_key": city_key,
        "run_dir": str(run_dir),
        "contract": contract,
        "raw_source": raw_manifest,
        "runtime": runtime,
        "qa": {
            "rows_in_surface": 0,
            "duplicate_surface_key_rows": 0,
            "offense_rows": int(len(status)),
            "admitted_rows": 0,
            "share_admitted_rows": 0,
            "feed_total_candidate_rows": 0,
            "share_sum_pass_all": False,
        },
    }
    (packet / "packet_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    (packet / "qa_summary.json").write_text(json.dumps(_json_safe(manifest["qa"]), indent=2, sort_keys=True))
    (packet / "research_findings.json").write_text(json.dumps(_json_safe(contract), indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=time.strftime("gate15_generic_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "state" / "orchestration" / "gate15_generic_onboarding")
    parser.add_argument("--contract", type=Path, action="append", required=True)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    run_dir = output_root / args.run_id
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    all_surfaces: list[pd.DataFrame] = []
    all_normalized: list[pd.DataFrame] = []
    all_status: list[pd.DataFrame] = []
    all_recon: list[pd.DataFrame] = []
    runtimes: list[dict[str, object]] = []
    contracts: list[dict[str, object]] = []
    for contract_path in args.contract:
        contract = _effective_contract(_load_contract(contract_path))
        contracts.append(contract)
        city_key = str(contract["city_key"])
        start = time.perf_counter()
        entries = _source_entries(contract)
        raw_paths = [
            raw_dir / (f"{city_key}__{slug}_raw.parquet" if slug is not None else f"{city_key}_raw.parquet")
            for slug, _ in entries
        ]
        try:
            raw_frames: list[pd.DataFrame] = []
            mapped_frames: list[pd.DataFrame] = []
            raw_manifests: list[dict[str, object]] = []
            for (slug, source), raw_path in zip(entries, raw_paths):
                sub_contract = dict(contract)
                sub_contract["source"] = source
                raw_part = _download_source(sub_contract, raw_path, force_refresh=bool(args.force_refresh))
                if slug is not None:
                    raw_part = raw_part.copy()
                    raw_part["__source_slug"] = slug
                raw_frames.append(raw_part)
                mapped_frames.append(_apply_rules(raw_part, contract))
                raw_manifest: dict[str, object] = {"city_key": city_key}
                if slug is not None:
                    raw_manifest["source_slug"] = slug
                raw_manifest.update(
                    {
                        "source_name": source["name"],
                        "source_url": source["source_url"],
                        "api_endpoint": source["api_endpoint"],
                        "portal_type": source["portal_type"],
                        "raw_cache_path": str(raw_path),
                        "raw_cache_bytes": raw_path.stat().st_size if raw_path.exists() else None,
                        "raw_cache_sha256": _file_sha256(raw_path),
                    }
                )
                raw_manifests.append(raw_manifest)
            if len(raw_frames) == 1:
                raw = raw_frames[0]
                mapped = mapped_frames[0]
            else:
                raw = pd.concat(raw_frames, ignore_index=True, sort=False)
                mapped = pd.concat(mapped_frames, ignore_index=True, sort=False)
                mapped = mapped.drop_duplicates("offense_row_id", keep="first")
            surface, recon, normalized = _build_surface(mapped, contract)
            status = _status(recon, surface, contract)
            runtime = {
                "city_key": city_key,
                "worker_type": "generic_contract_ingestor",
                "wall_seconds": float(time.perf_counter() - start),
                "raw_cache_path": (
                    "; ".join(str(path) for path in raw_paths) if len(raw_paths) > 1 else str(raw_paths[0])
                ),
                "raw_cache_sha256": (
                    "; ".join(str(manifest_row["raw_cache_sha256"]) for manifest_row in raw_manifests)
                    if len(raw_manifests) > 1
                    else raw_manifests[0]["raw_cache_sha256"]
                ),
                "input_tokens": contract.get("worker_usage", {}).get("input_tokens"),
                "output_tokens": contract.get("worker_usage", {}).get("output_tokens"),
                "total_tokens": contract.get("worker_usage", {}).get("total_tokens"),
                "status": "completed",
            }
            _write_packet(run_dir, contract, raw, raw_manifests, surface, normalized, recon, status, runtime)
            all_surfaces.append(surface)
            all_normalized.append(normalized)
            all_recon.append(recon)
            all_status.append(status)
            runtimes.append(runtime)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            error_message = f"{type(exc).__name__}: {exc}"
            status = _failure_status(contract, error_message)
            runtime = {
                "city_key": city_key,
                "worker_type": "generic_contract_ingestor",
                "wall_seconds": float(time.perf_counter() - start),
                "raw_cache_path": (
                    "; ".join(str(path) for path in raw_paths) if len(raw_paths) > 1 else str(raw_paths[0])
                ),
                "raw_cache_sha256": (
                    "; ".join(str(_file_sha256(path)) for path in raw_paths)
                    if len(raw_paths) > 1
                    else _file_sha256(raw_paths[0])
                ),
                "input_tokens": contract.get("worker_usage", {}).get("input_tokens"),
                "output_tokens": contract.get("worker_usage", {}).get("output_tokens"),
                "total_tokens": contract.get("worker_usage", {}).get("total_tokens"),
                "status": "failed",
                "error_message": error_message,
                "traceback": traceback.format_exc(limit=5),
            }
            _write_failure_packet(run_dir, contract, status, runtime)
            all_status.append(status)
            all_recon.append(
                status[
                    [
                        "year",
                        "offense",
                        "raw_offense_rows",
                        "matched_share_rows",
                        "match_share",
                        "city_key",
                        "city_name",
                        "jurisdiction_id",
                        "validation_case_type",
                        "validation_source_name",
                        "validation_source_url",
                    ]
                ].copy()
            )
            runtimes.append(runtime)
    surface_all = pd.concat(all_surfaces, ignore_index=True, sort=False) if all_surfaces else pd.DataFrame()
    normalized_all = pd.concat(all_normalized, ignore_index=True, sort=False) if all_normalized else pd.DataFrame()
    status_all = pd.concat(all_status, ignore_index=True, sort=False) if all_status else pd.DataFrame()
    recon_all = pd.concat(all_recon, ignore_index=True, sort=False) if all_recon else pd.DataFrame()
    surface_all.to_parquet(run_dir / "generic_city_share_surface.parquet", index=False)
    normalized_all.to_parquet(run_dir / "generic_normalized_incidents.parquet", index=False)
    status_all.to_csv(run_dir / "generic_city_offense_status.csv", index=False)
    recon_all.to_csv(run_dir / "generic_reconciliation.csv", index=False)
    pd.DataFrame(runtimes).to_csv(run_dir / "worker_cost_runtime.csv", index=False)
    manifest = {
        "run_id": args.run_id,
        "contracts": contracts,
        "outputs": {
            "surface": str(run_dir / "generic_city_share_surface.parquet"),
            "normalized_incidents": str(run_dir / "generic_normalized_incidents.parquet"),
            "status": str(run_dir / "generic_city_offense_status.csv"),
            "runtime": str(run_dir / "worker_cost_runtime.csv"),
        },
        "summary": {
            "city_count": int(status_all["city_key"].nunique()) if not status_all.empty else 0,
            "rows": int(len(status_all)),
            "admitted_rows": int(status_all["admitted_flag"].sum()) if not status_all.empty else 0,
            "share_admitted_rows": int(status_all["share_admitted_flag"].sum()) if not status_all.empty else 0,
            "feed_total_candidate_rows": int(status_all["feed_total_candidate_flag"].sum()) if not status_all.empty else 0,
            "share_sum_failures": int((~status_all["share_sum_pass"]).sum()) if not status_all.empty else 0,
            "surface_duplicate_key_rows": _surface_duplicate_key_count(surface_all),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    print(json.dumps(_json_safe(manifest["summary"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
