from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crime import OFFENSES_7  # noqa: E402
from crimerisk.city_feed_quarantine import (  # noqa: E402
    DEFAULT_TEXTURE_POLICY_PATH,
    load_texture_policy,
    resolve_texture_key,
    texture_policy_allows,
)


LIVE_SURFACE = REPO_ROOT / "state" / "modeling" / "city_incident_share_surface.parquet"
RESIDUAL_SURFACE = (
    REPO_ROOT / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet"
)
PACKET_ROOT = REPO_ROOT / "state" / "review" / "packets" / "city"
PACKET_STATUS_SUMMARY = PACKET_ROOT / "city_packet_status_summary.csv"
CITY_RESIDUAL_BENCHMARK_JSON = REPO_ROOT / "state" / "modeling" / "city_residual_benchmark_2024.json"
NEXT_PHASE_RESIDUAL_BENCHMARK_JSON = REPO_ROOT / "state" / "modeling" / "next_phase_city_residual_benchmark_2024.json"
OUT_PARQUET = REPO_ROOT / "state" / "modeling" / "city_role_inventory_2024.parquet"
OUT_CSV = REPO_ROOT / "state" / "modeling" / "city_role_inventory_2024.csv"
ALLOCATION_SOURCE = REPO_ROOT / "src" / "crimerisk" / "allocation.py"

ROLE_VALUES = (
    "direct_posterior_live",
    "residual_training_only",
    "validation_holdout_only",
    "onboarded_not_integrated",
    "rejected",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _truthy(value: object) -> bool:
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _bool_or_false(value: object) -> bool:
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(value)


def _slug(value: object) -> str:
    text = _text(value).lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _load_promoted_residual_exclude_case_types(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES" in names:
                value = ast.literal_eval(node.value)
                return tuple(str(part) for part in value)
    raise ValueError(f"{path} does not define PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_packet_metadata(packet_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(packet_root.glob("*/packet_status.csv")):
        if path.parent.name == "_factory":
            continue
        frame = _read_csv(path)
        if frame.empty or "city_key" not in frame.columns:
            continue
        row = frame.iloc[0]
        rows.append(
            {
                "city_key": _text(row.get("city_key")) or path.parent.name,
                "packet_city_name": _text(row.get("city_name")),
                "jurisdiction_id": _text(row.get("jurisdiction_id")),
                "packet_dir": str(path.parent.relative_to(REPO_ROOT)),
                "packet_status_path": str(path.relative_to(REPO_ROOT)),
                "packet_status_present": True,
                "packet_status_production_ready": _text(row.get("production_ready")),
                "packet_status_integration_status": _text(row.get("city_share_integration_status")),
                "packet_status_validation_case_type": _text(row.get("validation_case_type")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "city_key",
                "packet_city_name",
                "jurisdiction_id",
                "packet_dir",
                "packet_status_path",
                "packet_status_present",
                "packet_status_production_ready",
                "packet_status_integration_status",
                "packet_status_validation_case_type",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates("city_key", keep="first")


def _load_packet_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["city_key", "packet_summary_present"])
    frame = _read_csv(path)
    if "city_key" not in frame.columns:
        raise ValueError(f"{path} has no city_key column")
    frame = frame[frame["city_key"].astype(str).ne("_factory")].copy()
    frame["packet_summary_present"] = True
    keep = [
        "city_key",
        "packet_summary_present",
        "production_ready",
        "city_share_integration_status",
        "offense_selective_ready_offenses",
        "central_recommended_disposition",
        "packet_recommended_disposition",
        "source_name",
        "state_abbr",
    ]
    for col in keep:
        if col not in frame.columns:
            frame[col] = ""
    return frame[keep].rename(
        columns={
            "production_ready": "summary_production_ready",
            "city_share_integration_status": "summary_integration_status",
            "offense_selective_ready_offenses": "summary_offense_selective_ready_offenses",
            "central_recommended_disposition": "summary_central_disposition",
            "packet_recommended_disposition": "summary_packet_disposition",
            "source_name": "summary_source_name",
            "state_abbr": "summary_state_abbr",
        }
    )


def _load_packet_offense_status(packet_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(packet_root.glob("*/packet_offense_status.csv")):
        if path.parent.name == "_factory":
            continue
        frame = _read_csv(path)
        if frame.empty or "offense" not in frame.columns:
            continue
        if "city_key" not in frame.columns:
            frame = frame.copy()
            frame["city_key"] = path.parent.name
        for _, row in frame.iterrows():
            status_text = "|".join(
                _text(row.get(col)).lower()
                for col in [
                    "city_share_integration_status",
                    "share_lane_disposition",
                    "qa_status",
                    "admission_mode",
                    "production_ready",
                    "packet_status",
                    "comparison_class",
                ]
                if col in frame.columns
            )
            rejected = any(
                token in status_text
                for token in ("reject", "blocked", "do_not_promote", "pilot_reject")
            )
            admitted = any(_truthy(row.get(col)) for col in ("share_admitted_flag", "admitted_flag"))
            if _text(row.get("production_ready")).lower() == "yes":
                admitted = True
            if _text(row.get("share_lane_disposition")).lower() == "admit_share_only":
                admitted = True
            if _text(row.get("city_share_integration_status")).lower() in {
                "offense_selective",
                "pilot_share_only",
                "generic_contract_share_pilot",
                "gate15_generic_share_only",
            }:
                admitted = True
            rows.append(
                {
                    "city_key": _text(row.get("city_key")) or path.parent.name,
                    "offense": _text(row.get("offense")),
                    "packet_offense_status_present": True,
                    "packet_offense_rejected": bool(rejected),
                    "packet_offense_admitted": bool(admitted),
                    "packet_offense_status_text": status_text,
                    "packet_offense_status_path": str(path.relative_to(REPO_ROOT)),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "city_key",
                "offense",
                "packet_offense_status_present",
                "packet_offense_rejected",
                "packet_offense_admitted",
                "packet_offense_status_text",
                "packet_offense_status_path",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates(["city_key", "offense"], keep="first")


def _load_surface_pairs(path: Path, *, city_key_by_jurisdiction: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(path)
    required = {"city_name", "jurisdiction_id", "state_fips", "offense"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    pairs = raw[["city_name", "jurisdiction_id", "state_fips", "offense"]].drop_duplicates().copy()
    pairs["jurisdiction_id"] = pairs["jurisdiction_id"].astype("string").fillna("").astype(str)
    pairs["city_name"] = pairs["city_name"].astype("string").fillna("").astype(str)
    pairs["state_fips"] = pairs["state_fips"].astype("string").str.zfill(2)
    pairs["offense"] = pairs["offense"].astype("string").fillna("").astype(str)
    pairs["city_key"] = [
        city_key_by_jurisdiction.get(_text(jurisdiction_id)) or _slug(city_name)
        for jurisdiction_id, city_name in zip(pairs["jurisdiction_id"], pairs["city_name"], strict=False)
    ]
    return pairs, raw


def _surface_pair_stats(raw: pd.DataFrame, *, city_key_by_jurisdiction: dict[str, str]) -> pd.DataFrame:
    work = raw.copy()
    work["jurisdiction_id"] = work["jurisdiction_id"].astype("string").fillna("").astype(str)
    work["city_name"] = work["city_name"].astype("string").fillna("").astype(str)
    work["offense"] = work["offense"].astype("string").fillna("").astype(str)
    work["city_key"] = [
        city_key_by_jurisdiction.get(_text(jurisdiction_id)) or _slug(city_name)
        for jurisdiction_id, city_name in zip(work["jurisdiction_id"], work["city_name"], strict=False)
    ]
    if "year" in work.columns:
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
    else:
        work["year"] = pd.NA
    work["incident_count"] = pd.to_numeric(work.get("incident_count", 0.0), errors="coerce").fillna(0.0)
    grouped = (
        work.groupby(["city_key", "offense"], dropna=False)
        .agg(
            source_row_count=("city_key", "size"),
            incident_count_total=("incident_count", "sum"),
            source_year_min=("year", "min"),
            source_year_max=("year", "max"),
        )
        .reset_index()
    )
    return grouped


def _load_residual_case_pairs(
    raw: pd.DataFrame,
    *,
    city_key_by_jurisdiction: dict[str, str],
    excluded_case_types: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"city_name", "jurisdiction_id", "state_fips", "offense", "validation_case_type"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{RESIDUAL_SURFACE} missing required columns: {missing}")
    case = raw[["city_name", "jurisdiction_id", "state_fips", "offense", "validation_case_type"]].drop_duplicates()
    case = case.copy()
    case["jurisdiction_id"] = case["jurisdiction_id"].astype("string").fillna("").astype(str)
    case["city_name"] = case["city_name"].astype("string").fillna("").astype(str)
    case["state_fips"] = case["state_fips"].astype("string").str.zfill(2)
    case["offense"] = case["offense"].astype("string").fillna("").astype(str)
    case["validation_case_type"] = case["validation_case_type"].astype("string").fillna("").astype(str)
    case["city_key"] = [
        city_key_by_jurisdiction.get(_text(jurisdiction_id)) or _slug(city_name)
        for jurisdiction_id, city_name in zip(case["jurisdiction_id"], case["city_name"], strict=False)
    ]
    holdout = case[case["validation_case_type"].isin(excluded_case_types)].copy()
    train = case[~case["validation_case_type"].isin(excluded_case_types)].copy()
    return train, holdout


def _load_loco_benchmark_city_keys(paths: list[Path], *, city_key_by_jurisdiction: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for entry in payload.get("by_holdout_city", []):
            jurisdiction_id = _text(entry.get("holdout_jurisdiction_id"))
            city_name = _text(entry.get("holdout_city_name"))
            rows.append(
                {
                    "city_key": city_key_by_jurisdiction.get(jurisdiction_id) or _slug(city_name),
                    "loco_benchmark_holdout_city": True,
                    "loco_benchmark_source": str(path.relative_to(REPO_ROOT)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["city_key", "loco_benchmark_holdout_city", "loco_benchmark_source"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("city_key", dropna=False)
        .agg(
            loco_benchmark_holdout_city=("loco_benchmark_holdout_city", "max"),
            loco_benchmark_source=("loco_benchmark_source", lambda values: "|".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )


def _city_level_rejected(row: pd.Series) -> bool:
    status_text = "|".join(
        _text(row.get(col)).lower()
        for col in [
            "summary_central_disposition",
            "summary_packet_disposition",
            "summary_integration_status",
            "packet_status_integration_status",
        ]
    )
    return any(
        token in status_text
        for token in ("blocked", "do_not_promote", "counts_usable_not_city_share_promotable")
    )


def _build_city_universe(
    *,
    live_pairs: pd.DataFrame,
    residual_case_pairs: pd.DataFrame,
    packet_metadata: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in (live_pairs, residual_case_pairs):
        frames.append(source[["city_key", "city_name", "jurisdiction_id", "state_fips"]].drop_duplicates())
    if not packet_metadata.empty:
        frames.append(
            packet_metadata.rename(columns={"packet_city_name": "city_name"})[
                ["city_key", "city_name", "jurisdiction_id"]
            ].assign(state_fips="")
        )
    city = pd.concat(frames, ignore_index=True)
    city = city[city["city_key"].astype(str).str.len().gt(0)].copy()
    city["has_state_fips"] = city["state_fips"].astype(str).str.len().gt(0)
    city = (
        city.sort_values(["city_key", "has_state_fips"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("city_key", keep="first")
        .drop(columns="has_state_fips")
        .sort_values("city_key", kind="mergesort")
        .reset_index(drop=True)
    )
    return city


def build_inventory() -> pd.DataFrame:
    packet_metadata = _load_packet_metadata(PACKET_ROOT)
    city_key_by_jurisdiction = {
        _text(row.jurisdiction_id): _text(row.city_key)
        for row in packet_metadata.itertuples(index=False)
        if _text(row.jurisdiction_id) and _text(row.city_key)
    }
    packet_summary = _load_packet_summary(PACKET_STATUS_SUMMARY)
    packet_offense_status = _load_packet_offense_status(PACKET_ROOT)

    live_pairs, live_raw = _load_surface_pairs(LIVE_SURFACE, city_key_by_jurisdiction=city_key_by_jurisdiction)
    residual_pairs, residual_raw = _load_surface_pairs(
        RESIDUAL_SURFACE,
        city_key_by_jurisdiction=city_key_by_jurisdiction,
    )
    excluded_case_types = set(_load_promoted_residual_exclude_case_types(ALLOCATION_SOURCE))
    residual_train_pairs, holdout_pairs = _load_residual_case_pairs(
        residual_raw,
        city_key_by_jurisdiction=city_key_by_jurisdiction,
        excluded_case_types=excluded_case_types,
    )
    live_stats = _surface_pair_stats(live_raw, city_key_by_jurisdiction=city_key_by_jurisdiction).rename(
        columns={
            "source_row_count": "live_surface_row_count",
            "incident_count_total": "live_surface_incident_count_total",
            "source_year_min": "live_surface_year_min",
            "source_year_max": "live_surface_year_max",
        }
    )
    residual_stats = _surface_pair_stats(
        residual_raw,
        city_key_by_jurisdiction=city_key_by_jurisdiction,
    ).rename(
        columns={
            "source_row_count": "residual_surface_row_count",
            "incident_count_total": "residual_surface_incident_count_total",
            "source_year_min": "residual_surface_year_min",
            "source_year_max": "residual_surface_year_max",
        }
    )

    residual_case_summary = (
        residual_raw[["city_name", "jurisdiction_id", "state_fips", "offense", "validation_case_type"]]
        .drop_duplicates()
        .copy()
    )
    residual_case_summary["city_key"] = [
        city_key_by_jurisdiction.get(_text(jurisdiction_id)) or _slug(city_name)
        for jurisdiction_id, city_name in zip(
            residual_case_summary["jurisdiction_id"],
            residual_case_summary["city_name"],
            strict=False,
        )
    ]
    residual_case_summary = (
        residual_case_summary.groupby(["city_key", "offense"], dropna=False)["validation_case_type"]
        .agg(lambda values: "|".join(sorted(set(map(str, values)))))
        .rename("residual_validation_case_type")
        .reset_index()
    )

    loco_benchmark = _load_loco_benchmark_city_keys(
        [CITY_RESIDUAL_BENCHMARK_JSON, NEXT_PHASE_RESIDUAL_BENCHMARK_JSON],
        city_key_by_jurisdiction=city_key_by_jurisdiction,
    )

    city = _build_city_universe(
        live_pairs=live_pairs,
        residual_case_pairs=residual_pairs,
        packet_metadata=packet_metadata,
    )
    inventory = city.merge(pd.DataFrame({"offense": list(OFFENSES_7)}), how="cross")

    inventory = (
        inventory.merge(packet_summary, on="city_key", how="left")
        .merge(
            packet_metadata.drop(columns=["packet_city_name", "jurisdiction_id"], errors="ignore"),
            on="city_key",
            how="left",
        )
        .merge(packet_offense_status, on=["city_key", "offense"], how="left")
        .merge(live_stats, on=["city_key", "offense"], how="left")
        .merge(residual_stats, on=["city_key", "offense"], how="left")
        .merge(residual_case_summary, on=["city_key", "offense"], how="left")
        .merge(loco_benchmark, on="city_key", how="left")
    )

    live_set = set(zip(live_pairs["city_key"], live_pairs["offense"], strict=False))
    residual_train_set = set(zip(residual_train_pairs["city_key"], residual_train_pairs["offense"], strict=False))
    holdout_set = set(zip(holdout_pairs["city_key"], holdout_pairs["offense"], strict=False))
    holdout_case_type = {
        (row.city_key, row.offense): row.validation_case_type
        for row in holdout_pairs.itertuples(index=False)
    }

    texture_policy_table = load_texture_policy()

    roles: list[str] = []
    role_sources: list[str] = []
    evidences: list[str] = []
    for _, row in inventory.iterrows():
        key = (_text(row.get("city_key")), _text(row.get("offense")))
        packet_rejected = bool(row.get("packet_offense_rejected") is True)
        city_rejected = _city_level_rejected(row)
        if key in live_set:
            roles.append("direct_posterior_live")
            role_sources.append(str(LIVE_SURFACE.relative_to(REPO_ROOT)))
            evidences.append(
                "city/offense appears in the live direct-override surface used as incident_surface in allocation"
            )
        elif key in holdout_set:
            roles.append("validation_holdout_only")
            role_sources.append(
                f"{RESIDUAL_SURFACE.relative_to(REPO_ROOT)};"
                "src/crimerisk/allocation.py:PROMOTED_RESIDUAL_EXCLUDE_VALIDATION_CASE_TYPES"
            )
            evidences.append(
                "city/offense appears in the next-phase surface with excluded validation_case_type="
                f"{holdout_case_type.get(key)}"
            )
        elif key in residual_train_set:
            roles.append("residual_training_only")
            role_sources.append(
                f"{RESIDUAL_SURFACE.relative_to(REPO_ROOT)};"
                "src/crimerisk/allocation.py:promoted_residual_training_city_shares_path"
            )
            evidences.append(
                "city/offense appears in the promoted residual-training surface after holdout case-type exclusions"
            )
        elif packet_rejected:
            roles.append("rejected")
            role_sources.append(_text(row.get("packet_offense_status_path")) or str(PACKET_STATUS_SUMMARY.relative_to(REPO_ROOT)))
            evidences.append("packet offense status contains reject/blocked/do_not_promote disposition")
        elif city_rejected:
            roles.append("rejected")
            source = _text(row.get("packet_status_path")) or str(PACKET_STATUS_SUMMARY.relative_to(REPO_ROOT))
            role_sources.append(source)
            evidences.append("packet city-level status/disposition rejects city-share promotion")
        elif (
            row.get("packet_status_present") is True
            or row.get("packet_summary_present") is True
            or row.get("packet_offense_status_present") is True
        ):
            roles.append("onboarded_not_integrated")
            source = _text(row.get("packet_offense_status_path")) or _text(row.get("packet_status_path"))
            role_sources.append(source or str(PACKET_STATUS_SUMMARY.relative_to(REPO_ROOT)))
            evidences.append("packet exists, but city/offense is absent from live, residual-training, and holdout sets")
        elif not texture_policy_allows(
            resolve_texture_key(_text(row.get("city_name"))),
            _text(row.get("offense")),
            policy=texture_policy_table,
        ):
            # Texture-denied located evidence with no packet to fall back to
            # (e.g. Durham rape under the v10 default-deny policy). Classified as
            # rejected rather than a new role value; the evidence field carries the
            # nuance, and a later verified-clean allow row recomputes this to
            # residual_training_only naturally.
            roles.append("rejected")
            role_sources.append(str(DEFAULT_TEXTURE_POLICY_PATH.relative_to(REPO_ROOT)))
            evidences.append(
                "texture policy default-deny pending verification (v10); "
                "located evidence exists but is not admitted"
            )
        else:
            roles.append("")
            role_sources.append("")
            evidences.append("no source cleanly classified this city/offense")

    inventory["role"] = roles
    inventory["role_source"] = role_sources
    inventory["evidence"] = evidences
    inventory["live_surface_present"] = [
        (city_key, offense) in live_set
        for city_key, offense in zip(inventory["city_key"], inventory["offense"], strict=False)
    ]
    inventory["residual_training_surface_present"] = [
        (city_key, offense) in residual_train_set
        for city_key, offense in zip(inventory["city_key"], inventory["offense"], strict=False)
    ]
    inventory["validation_holdout_surface_present"] = [
        (city_key, offense) in holdout_set
        for city_key, offense in zip(inventory["city_key"], inventory["offense"], strict=False)
    ]
    inventory["loco_benchmark_holdout_city"] = inventory["loco_benchmark_holdout_city"].map(_bool_or_false)

    bool_cols = [
        "packet_summary_present",
        "packet_status_present",
        "packet_offense_status_present",
        "packet_offense_rejected",
        "packet_offense_admitted",
    ]
    for col in bool_cols:
        if col in inventory.columns:
            inventory[col] = inventory[col].map(_bool_or_false)

    ordered_cols = [
        "city_key",
        "city_name",
        "jurisdiction_id",
        "state_fips",
        "offense",
        "role",
        "evidence",
        "role_source",
        "live_surface_present",
        "residual_training_surface_present",
        "validation_holdout_surface_present",
        "residual_validation_case_type",
        "loco_benchmark_holdout_city",
        "loco_benchmark_source",
        "packet_summary_present",
        "summary_production_ready",
        "summary_integration_status",
        "summary_offense_selective_ready_offenses",
        "summary_central_disposition",
        "summary_packet_disposition",
        "summary_source_name",
        "summary_state_abbr",
        "packet_status_present",
        "packet_status_production_ready",
        "packet_status_integration_status",
        "packet_status_validation_case_type",
        "packet_dir",
        "packet_status_path",
        "packet_offense_status_present",
        "packet_offense_rejected",
        "packet_offense_admitted",
        "packet_offense_status_text",
        "packet_offense_status_path",
        "live_surface_row_count",
        "live_surface_incident_count_total",
        "live_surface_year_min",
        "live_surface_year_max",
        "residual_surface_row_count",
        "residual_surface_incident_count_total",
        "residual_surface_year_min",
        "residual_surface_year_max",
    ]
    for col in ordered_cols:
        if col not in inventory.columns:
            inventory[col] = pd.NA
    inventory = inventory[ordered_cols].sort_values(["city_key", "offense"], kind="mergesort").reset_index(drop=True)
    return inventory


def validate_inventory(inventory: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    duplicate_mask = inventory.duplicated(["city_key", "offense"], keep=False)
    if bool(duplicate_mask.any()):
        dupes = inventory.loc[duplicate_mask, ["city_key", "offense"]].drop_duplicates()
        issues.append(f"duplicate city/offense assignments: {dupes.to_dict(orient='records')}")

    bad_roles = sorted(set(inventory["role"].astype(str)) - set(ROLE_VALUES))
    if bad_roles:
        issues.append(f"unexpected/blank role values: {bad_roles}")

    counts = inventory.groupby("city_key", dropna=False)["offense"].nunique()
    incomplete = counts[counts.ne(len(OFFENSES_7))]
    if not incomplete.empty:
        issues.append(f"cities without exactly {len(OFFENSES_7)} offenses: {incomplete.to_dict()}")

    if inventory[["city_key", "offense", "role"]].isna().any().any():
        issues.append("city_key/offense/role contains nulls")

    live_conflict = inventory[inventory["role"].ne("direct_posterior_live") & inventory["live_surface_present"]]
    if not live_conflict.empty:
        issues.append(
            "live surface pairs not classified direct_posterior_live: "
            f"{live_conflict[['city_key', 'offense', 'role']].to_dict(orient='records')}"
        )

    training_conflict = inventory[
        inventory["role"].isin(["validation_holdout_only", "onboarded_not_integrated", "rejected"])
        & inventory["residual_training_surface_present"]
    ]
    if not training_conflict.empty:
        issues.append(
            "residual-training pairs assigned non-training/non-live role: "
            f"{training_conflict[['city_key', 'offense', 'role']].to_dict(orient='records')}"
        )

    holdout_conflict = inventory[
        inventory["role"].isin(["residual_training_only", "onboarded_not_integrated", "rejected"])
        & inventory["validation_holdout_surface_present"]
    ]
    if not holdout_conflict.empty:
        issues.append(
            "holdout pairs assigned non-holdout/non-live role: "
            f"{holdout_conflict[['city_key', 'offense', 'role']].to_dict(orient='records')}"
        )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-parquet", type=Path, default=OUT_PARQUET)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    inventory = build_inventory()
    issues = validate_inventory(inventory)
    if issues:
        print("City-role inventory validation failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(args.out_parquet, index=False)
    inventory.to_csv(args.out_csv, index=False)

    role_offense_counts = pd.crosstab(inventory["role"], inventory["offense"])
    for offense in OFFENSES_7:
        if offense not in role_offense_counts.columns:
            role_offense_counts[offense] = 0
    role_offense_counts = role_offense_counts[list(OFFENSES_7)].reindex(ROLE_VALUES, fill_value=0)

    print(f"Wrote {args.out_parquet}")
    print(f"Wrote {args.out_csv}")
    print(f"Rows: {len(inventory)}")
    print(f"Cities: {inventory['city_key'].nunique()}")
    print("Role x offense city counts:")
    print(role_offense_counts.to_string())
    print("Unclassified city/offense pairs: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
