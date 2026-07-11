from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path

import pandas as pd

from crimerisk.build_freshness import artifact_is_current
from crimerisk.city_feed_quarantine import (
    filter_denied_texture_surface,
    load_texture_policy,
)
from crimerisk.city_incidents import (
    build_austin_incident_reconciliation,
    build_austin_incident_share_surface,
    build_baltimore_incident_reconciliation,
    build_baltimore_incident_share_surface,
    build_boston_incident_reconciliation,
    build_boston_incident_share_surface,
    build_chicago_incident_reconciliation,
    build_chicago_incident_share_surface,
    build_denver_incident_reconciliation,
    build_denver_incident_share_surface,
    build_mesa_incident_reconciliation,
    build_mesa_incident_share_surface,
    build_minneapolis_incident_reconciliation,
    build_minneapolis_incident_share_surface,
    build_nyc_incident_reconciliation,
    build_nyc_incident_share_surface,
    build_philadelphia_incident_reconciliation,
    build_philadelphia_incident_share_surface,
    build_san_francisco_incident_reconciliation,
    build_san_francisco_incident_share_surface,
    build_seattle_incident_reconciliation,
    build_seattle_incident_share_surface,
    build_st_louis_incident_reconciliation,
    build_st_louis_incident_share_surface,
    build_washington_dc_incident_reconciliation,
    build_washington_dc_incident_share_surface,
)
from crimerisk.paths import RepoPaths
from crimerisk.stage_locks import stage_write_lock


CITY_SHARE_COLUMNS = [
    "city_name",
    "jurisdiction_id",
    "state_fips",
    "year",
    "offense",
    "block_group_geoid",
    "incident_count",
    "share_within_city",
    "geocode_quality_tier",
]

CITY_INCIDENT_PACKET_FILES = (
    "packet_status.csv",
    "packet_offense_status.csv",
    "offense_crosswalk.csv",
    "published_reference_extract.csv",
    "reconciliation_summary.csv",
)


@dataclass(frozen=True)
class CityIncidentShareBuildConfig:
    year_start: int = 2018
    year_end: int = 2024
    force_rebuild: bool = False
    force_source_refresh: bool = False


def _empty_city_share_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CITY_SHARE_COLUMNS)


def get_v2_city_incident_input_root(paths: RepoPaths) -> Path:
    return paths.state_dir / "modeling" / "inputs" / "city_incident"


def _load_packet_status(packet_root: Path, *, require_source_exists: bool = False) -> pd.DataFrame:
    if not packet_root.exists():
        if require_source_exists:
            raise FileNotFoundError(
                "Missing canonical city-incident input surface: "
                f"{packet_root}. Run `python main.py promote-city-incident-inputs` first."
            )
        return pd.DataFrame(columns=["city_key", "production_ready", "city_share_integration_status"])
    rows = []
    for status_path in sorted(packet_root.glob("*/packet_status.csv")):
        status = pd.read_csv(status_path, dtype=str).fillna("")
        if status.empty:
            continue
        rows.append(
            {
                "city_key": status.iloc[0].get("city_key", status_path.parent.name),
                "production_ready": status.iloc[0].get("production_ready", ""),
                "city_share_integration_status": status.iloc[0].get("city_share_integration_status", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["city_key", "production_ready", "city_share_integration_status"])
    return pd.DataFrame(rows)


def _load_packet_offense_gates(packet_root: Path, *, require_source_exists: bool = False) -> dict[str, set[str]]:
    if not packet_root.exists():
        if require_source_exists:
            raise FileNotFoundError(
                "Missing canonical city-incident input surface: "
                f"{packet_root}. Run `python main.py promote-city-incident-inputs` first."
            )
        return {}
    gates: dict[str, set[str]] = {}
    for offense_path in sorted(packet_root.glob("*/packet_offense_status.csv")):
        df = pd.read_csv(offense_path, dtype=str).fillna("")
        if df.empty or "offense" not in df.columns:
            continue
        ready_mask = (
            df["production_ready"].astype(str).str.lower().isin({"yes", "true", "1", "partial"})
            if "production_ready" in df.columns
            else pd.Series(False, index=df.index)
        )
        active_mask = (
            df["city_share_integration_status"].astype(str).str.lower().isin({"active", "approved", "live", "offense_selective"})
            if "city_share_integration_status" in df.columns
            else pd.Series(False, index=df.index)
        )
        allowed = {
            str(v).strip()
            for v in df.loc[ready_mask & active_mask, "offense"].tolist()
            if str(v).strip()
        }
        if allowed:
            gates[offense_path.parent.name] = allowed
    return gates


def _city_dependency_paths(paths: RepoPaths, impls: dict[str, dict[str, object]]) -> list[Path]:
    deps: list[Path] = []
    module_root = Path(__file__).resolve().parent
    input_root = get_v2_city_incident_input_root(paths)
    deps.extend(
        [
            paths.repo_root / "configs" / "city_incident_priority.csv",
            paths.repo_root / "configs" / "city_incident_sources.csv",
            paths.repo_root / "configs" / "city_feed_coordinate_quarantine.csv",
            paths.repo_root / "configs" / "city_offense_texture_policy.csv",
            module_root / "city_shares.py",
            module_root / "city_incidents.py",
            module_root / "city_feed_quarantine.py",
        ]
    )
    deps.extend(sorted(input_root.glob("*/packet_status.csv")))
    deps.extend(sorted(input_root.glob("*/packet_offense_status.csv")))
    deps.extend(sorted(input_root.glob("*/offense_crosswalk.csv")))
    deps.extend(sorted(input_root.glob("*/published_reference_extract.csv")))
    deps.extend(sorted(input_root.glob("*/reconciliation_summary.csv")))
    for impl in impls.values():
        for path in impl.get("required_paths", []):
            if isinstance(path, Path):
                deps.append(path)
        for value in impl.get("kwargs", {}).values():
            if not isinstance(value, Path):
                continue
            if value.exists() and value.is_dir():
                nested_files = sorted(path for path in value.rglob("*") if path.is_file())
                if nested_files:
                    deps.extend(nested_files)
                else:
                    deps.append(value)
            else:
                deps.append(value)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in deps:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _load_enabled_city_order(paths: RepoPaths) -> list[str]:
    priority_path = paths.repo_root / "configs" / "city_incident_priority.csv"
    source_path = paths.repo_root / "configs" / "city_incident_sources.csv"
    missing = [str(path) for path in (priority_path, source_path) if not path.exists()]
    if missing:
        raise RuntimeError(
            "City incident share build requires the canonical city config surfaces to exist: "
            + ", ".join(missing)
        )
    priority = pd.read_csv(priority_path, dtype=str).fillna("")
    sources = pd.read_csv(source_path, dtype=str).fillna("")
    merged = priority.merge(
        sources[["city_key", "recommended_disposition"]],
        on="city_key",
        how="left",
    )
    packet_status = _load_packet_status(get_v2_city_incident_input_root(paths), require_source_exists=True)
    if not packet_status.empty:
        merged = merged.merge(packet_status, on="city_key", how="left")
    else:
        merged["production_ready"] = ""
        merged["city_share_integration_status"] = ""
    merged["priority_rank_numeric"] = pd.to_numeric(merged["priority_rank"], errors="coerce")
    merged = merged.sort_values(["priority_rank_numeric", "city_key"], kind="mergesort")
    enabled = merged.loc[
        merged["recommended_disposition"].eq("ready_now")
        & merged["production_ready"].astype(str).str.lower().isin({"yes", "true", "1", "partial"})
        & merged["city_share_integration_status"].astype(str).str.lower().isin(
            {"active", "approved", "live", "offense_selective"}
        ),
        "city_key",
    ].tolist()
    return [city for city in enabled if city]


def _resolve_categories_csv(paths: RepoPaths, filename: str) -> Path | None:
    candidate = paths.repo_root / "configs" / "city_incident_categories" / filename
    return candidate if candidate.exists() else None


def _city_impls(paths: RepoPaths) -> dict[str, dict[str, object]]:
    nyc_categories = _resolve_categories_csv(paths, "categories_new_york.csv")
    chicago_categories = _resolve_categories_csv(paths, "categories_Chicago.csv")
    city_input_root = get_v2_city_incident_input_root(paths)
    baltimore_crosswalk = city_input_root / "baltimore" / "offense_crosswalk.csv"
    baltimore_published_reference = city_input_root / "baltimore" / "published_reference_extract.csv"
    baltimore_packet_offense_status = city_input_root / "baltimore" / "packet_offense_status.csv"
    denver_crosswalk = city_input_root / "denver" / "offense_crosswalk.csv"
    denver_reconciliation_summary = city_input_root / "denver" / "reconciliation_summary.csv"
    denver_packet_offense_status = city_input_root / "denver" / "packet_offense_status.csv"
    mesa_crosswalk = city_input_root / "mesa" / "offense_crosswalk.csv"
    mesa_published_reference = city_input_root / "mesa" / "published_reference_extract.csv"
    minneapolis_crosswalk = city_input_root / "minneapolis" / "offense_crosswalk.csv"
    minneapolis_reconciliation_summary = city_input_root / "minneapolis" / "reconciliation_summary.csv"
    minneapolis_packet_offense_status = city_input_root / "minneapolis" / "packet_offense_status.csv"
    st_louis_packet_offense_status = city_input_root / "st_louis_mo" / "packet_offense_status.csv"
    philadelphia_crosswalk = city_input_root / "philadelphia" / "offense_crosswalk.csv"
    philadelphia_published_reference = city_input_root / "philadelphia" / "published_reference_extract.csv"
    philadelphia_packet_offense_status = city_input_root / "philadelphia" / "packet_offense_status.csv"
    washington_dc_crosswalk = city_input_root / "washington_dc" / "offense_crosswalk.csv"
    washington_dc_published_reference = city_input_root / "washington_dc" / "published_reference_extract.csv"
    washington_dc_packet_offense_status = city_input_root / "washington_dc" / "packet_offense_status.csv"
    sf_crosswalk = city_input_root / "san_francisco" / "offense_crosswalk.csv"
    sf_published_reference = city_input_root / "san_francisco" / "published_reference_extract.csv"
    sf_packet_offense_status = city_input_root / "san_francisco" / "packet_offense_status.csv"
    return {
        "new_york": {
            "share_fn": build_nyc_incident_share_surface,
            "recon_fn": build_nyc_incident_reconciliation,
            "required_paths": [nyc_categories],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "nyc",
                "categories_csv": nyc_categories,
                "ny_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_36_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2018,
                "end_year": 2024,
            },
        },
        "chicago": {
            "share_fn": build_chicago_incident_share_surface,
            "recon_fn": build_chicago_incident_reconciliation,
            "required_paths": [chicago_categories],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "chicago",
                "categories_csv": chicago_categories,
                "il_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_17_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2018,
                "end_year": 2024,
            },
        },
        "denver": {
            "share_fn": build_denver_incident_share_surface,
            "recon_fn": build_denver_incident_reconciliation,
            "required_paths": [denver_crosswalk, denver_reconciliation_summary, denver_packet_offense_status],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "denver",
                "offense_crosswalk_csv": denver_crosswalk,
                "reconciliation_summary_csv": denver_reconciliation_summary,
                "packet_offense_status_csv": denver_packet_offense_status,
                "co_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_08_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2024,
                "end_year": 2024,
            },
        },
        "seattle": {
            "share_fn": build_seattle_incident_share_surface,
            "recon_fn": build_seattle_incident_reconciliation,
            "required_paths": [],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "seattle",
                "wa_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_53_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2018,
                "end_year": 2024,
            },
        },
        "san_francisco": {
            "share_fn": build_san_francisco_incident_share_surface,
            "recon_fn": build_san_francisco_incident_reconciliation,
            "required_paths": [sf_crosswalk, sf_published_reference, sf_packet_offense_status],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "san_francisco",
                "offense_crosswalk_csv": sf_crosswalk,
                "published_reference_csv": sf_published_reference,
                "packet_offense_status_csv": sf_packet_offense_status,
                "ca_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_06_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2018,
                "end_year": 2024,
            },
        },
        "austin": {
            "share_fn": build_austin_incident_share_surface,
            "recon_fn": build_austin_incident_reconciliation,
            "required_paths": [],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "austin",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2018,
                "end_year": 2024,
            },
        },
        "boston": {
            "share_fn": build_boston_incident_share_surface,
            "recon_fn": build_boston_incident_reconciliation,
            "required_paths": [],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "boston",
                "ma_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_25_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2019,
                "end_year": 2024,
            },
        },
        "baltimore": {
            "share_fn": build_baltimore_incident_share_surface,
            "recon_fn": build_baltimore_incident_reconciliation,
            "required_paths": [baltimore_crosswalk, baltimore_published_reference, baltimore_packet_offense_status],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "baltimore",
                "offense_crosswalk_csv": baltimore_crosswalk,
                "published_reference_csv": baltimore_published_reference,
                "packet_offense_status_csv": baltimore_packet_offense_status,
                "md_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_24_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2024,
                "end_year": 2024,
            },
        },
        "mesa": {
            "share_fn": build_mesa_incident_share_surface,
            "recon_fn": build_mesa_incident_reconciliation,
            "required_paths": [mesa_crosswalk, mesa_published_reference],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "mesa",
                "offense_crosswalk_csv": mesa_crosswalk,
                "published_reference_csv": mesa_published_reference,
                "az_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_04_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2020,
                "end_year": 2024,
            },
        },
        "minneapolis": {
            "share_fn": build_minneapolis_incident_share_surface,
            "recon_fn": build_minneapolis_incident_reconciliation,
            "required_paths": [minneapolis_crosswalk, minneapolis_reconciliation_summary, minneapolis_packet_offense_status],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "minneapolis",
                "offense_crosswalk_csv": minneapolis_crosswalk,
                "reconciliation_summary_csv": minneapolis_reconciliation_summary,
                "packet_offense_status_csv": minneapolis_packet_offense_status,
                "mn_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_27_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2024,
                "end_year": 2024,
            },
        },
        "st_louis_mo": {
            "share_fn": build_st_louis_incident_share_surface,
            "recon_fn": build_st_louis_incident_reconciliation,
            "required_paths": [st_louis_packet_offense_status],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "st_louis_mo",
                "mo_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_29_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2024,
                "end_year": 2024,
            },
        },
        "philadelphia": {
            "share_fn": build_philadelphia_incident_share_surface,
            "recon_fn": build_philadelphia_incident_reconciliation,
            "required_paths": [philadelphia_crosswalk, philadelphia_published_reference, philadelphia_packet_offense_status],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "philadelphia",
                "offense_crosswalk_csv": philadelphia_crosswalk,
                "published_reference_csv": philadelphia_published_reference,
                "packet_offense_status_csv": philadelphia_packet_offense_status,
                "pa_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_42_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2018,
                "end_year": 2024,
            },
        },
        "washington_dc": {
            "share_fn": build_washington_dc_incident_share_surface,
            "recon_fn": build_washington_dc_incident_reconciliation,
            "required_paths": [washington_dc_crosswalk, washington_dc_published_reference, washington_dc_packet_offense_status],
            "kwargs": {
                "raw_dir": paths.data_dir / "city-incidents" / "washington_dc",
                "offense_crosswalk_csv": washington_dc_crosswalk,
                "published_reference_csv": washington_dc_published_reference,
                "packet_offense_status_csv": washington_dc_packet_offense_status,
                "dc_bg_zip": paths.data_dir / "tiger_bg" / "tl_2020_11_bg.zip",
                "bg_crosswalk_parquet": paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
                "start_year": 2024,
                "end_year": 2024,
            },
        },
    }


def _cached_city_keys(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    mapping = {
        "baltimore": "baltimore",
        "denver": "denver",
        "new york": "new_york",
        "chicago": "chicago",
        "minneapolis": "minneapolis",
        "seattle": "seattle",
        "san francisco": "san_francisco",
        "austin": "austin",
        "boston": "boston",
        "mesa": "mesa",
        "philadelphia": "philadelphia",
        "st. louis": "st_louis_mo",
        "washington": "washington_dc",
    }
    return {
        mapping[str(city_name).strip().lower()]
        for city_name in frame.get("city_name", pd.Series(dtype="object")).dropna().tolist()
        if str(city_name).strip().lower() in mapping
    }


def build_city_incident_share_surface(
    *,
    paths: RepoPaths,
    config: CityIncidentShareBuildConfig = CityIncidentShareBuildConfig(),
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, pd.DataFrame]]:
    offense_gates = _load_packet_offense_gates(get_v2_city_incident_input_root(paths), require_source_exists=True)
    enabled_keys = _load_enabled_city_order(paths)
    impls = _city_impls(paths)

    active: list[tuple[str, dict[str, object]]] = []
    skipped_missing_inputs: list[str] = []
    skipped_unsupported: list[str] = []
    for key in enabled_keys:
        impl = impls.get(key)
        if impl is None:
            skipped_unsupported.append(key)
            continue
        required_paths = list(impl.get("required_paths", []))
        if any(path is None or not Path(path).exists() for path in required_paths):
            skipped_missing_inputs.append(key)
            continue
        impl_copy = dict(impl)
        kwargs = dict(impl_copy["kwargs"])
        kwargs["start_year"] = int(config.year_start)
        kwargs["end_year"] = int(config.year_end)
        kwargs["force_source_refresh"] = bool(config.force_source_refresh)
        impl_copy["kwargs"] = kwargs
        active.append((key, impl_copy))

    if skipped_unsupported:
        raise RuntimeError(
            "Enabled city incident share builds do not yet have canonical builder support: "
            + ", ".join(sorted(skipped_unsupported))
        )
    if skipped_missing_inputs:
        raise RuntimeError(
            "Enabled city incident share builds are missing required inputs: "
            + ", ".join(sorted(skipped_missing_inputs))
        )
    if not active:
        return _empty_city_share_frame(), [], {}

    texture_policy = load_texture_policy()
    frames: list[pd.DataFrame] = []
    active_summaries: list[dict[str, object]] = []
    recon_payloads: dict[str, pd.DataFrame] = {}
    for city_key, impl in active:
        frame = impl["share_fn"](**impl["kwargs"])
        allowed_offenses = offense_gates.get(city_key)
        if allowed_offenses:
            frame = frame[frame["offense"].astype(str).isin(allowed_offenses)].copy()
        texture_denied_offenses: list[str] = []
        if not frame.empty and not texture_policy.empty:
            frame, removed = filter_denied_texture_surface(frame, policy=texture_policy)
            if not removed.empty:
                texture_denied_offenses = sorted(removed["offense"].astype(str).unique().tolist())
        if frame.empty:
            continue
        frames.append(frame)
        active_summaries.append(
            {
                "city_key": city_key,
                "mode": "offense_selective" if allowed_offenses else "city_wide",
                "offenses": sorted(allowed_offenses) if allowed_offenses else "all",
                "texture_denied_offenses": texture_denied_offenses,
            }
        )
        recon_payloads[city_key] = impl["recon_fn"](**impl["kwargs"])

    if not frames:
        raise RuntimeError("City incident share build found no rows after applying city/offense gates.")
    return pd.concat(frames, ignore_index=True), active_summaries, recon_payloads


def promote_v2_city_incident_inputs(
    *,
    paths: RepoPaths,
    out_root: Path | None = None,
) -> dict[str, object]:
    packet_root = paths.review_packets_dir / "city"
    out_root = out_root or get_v2_city_incident_input_root(paths)
    out_root.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    copied_cities: list[str] = []
    for packet_dir in sorted(path for path in packet_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        city_out_dir = out_root / packet_dir.name
        city_out_dir.mkdir(parents=True, exist_ok=True)
        city_copied = False
        for filename in CITY_INCIDENT_PACKET_FILES:
            src = packet_dir / filename
            dst = city_out_dir / filename
            if src.exists():
                shutil.copyfile(src, dst)
                copied_files += 1
                city_copied = True
            elif dst.exists():
                dst.unlink()
        if city_copied:
            copied_cities.append(packet_dir.name)

    return {
        "cities": len(copied_cities),
        "files": copied_files,
        "out_root": str(out_root),
        "copied_city_keys": copied_cities,
    }


def write_v2_city_incident_shares(
    *,
    paths: RepoPaths,
    out_path: Path | None = None,
    reconciliation_dir: Path | None = None,
    config: CityIncidentShareBuildConfig = CityIncidentShareBuildConfig(),
    blocked_by: tuple[str, ...] | None = None,
) -> dict[str, object]:
    with stage_write_lock(paths=paths, stage="city_incident_shares", blocked_by=blocked_by):
        out_path = out_path or (paths.state_dir / "modeling" / "city_incident_share_surface.parquet")
        reconciliation_dir = reconciliation_dir or (paths.review_analysis_dir / "city_reconciliation")
        combined_reconciliation_path = paths.state_dir / "modeling" / f"city_incident_reconciliation_{int(config.year_end)}.parquet"
        impls = _city_impls(paths)
        enabled_keys = _load_enabled_city_order(paths)
        dependency_paths = _city_dependency_paths(paths, impls)
        dependencies_current = artifact_is_current(out_path, dependency_paths)
        if out_path.exists() and not config.force_rebuild and not config.force_source_refresh and dependencies_current:
            frame = pd.read_parquet(out_path)
            if _cached_city_keys(frame) == set(enabled_keys):
                return {
                    "rows": int(len(frame)),
                    "out_path": str(out_path),
                    "active_cities": enabled_keys,
                    "reconciliation_files": sorted(str(path) for path in reconciliation_dir.glob("*_reconciliation.parquet")),
                    "combined_reconciliation_path": str(combined_reconciliation_path),
                    "used_cache": True,
                }
        frame, active_summaries, recon_payloads = build_city_incident_share_surface(paths=paths, config=config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reconciliation_dir.mkdir(parents=True, exist_ok=True)
        for existing in reconciliation_dir.glob("*_reconciliation.parquet"):
            existing.unlink()
        for city_key, recon in recon_payloads.items():
            recon.to_parquet(reconciliation_dir / f"{city_key}_reconciliation.parquet", index=False)
        if recon_payloads:
            combined_reconciliation_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(recon_payloads.values(), ignore_index=True, sort=False).to_parquet(
                combined_reconciliation_path,
                index=False,
            )
        frame.to_parquet(out_path, index=False)
        return {
            "rows": int(len(frame)),
            "out_path": str(out_path),
            "active_cities": active_summaries,
            "reconciliation_files": sorted(str(path) for path in reconciliation_dir.glob("*_reconciliation.parquet")),
            "combined_reconciliation_path": str(combined_reconciliation_path),
            "used_cache": False,
        }
