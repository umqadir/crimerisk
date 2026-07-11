from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from crimerisk.paths import RepoPaths
from crimerisk.source_provenance import LOCAL_PUBLICATION_SOURCE


QUINCY_PUBLICATION_RAW_DATA_SOURCE = "quincy_pd_annual_report"
BRUNSWICK_PUBLICATION_RAW_DATA_SOURCE = "brunswick_ga_crime_statistics"
GENERIC_PACKET_PUBLICATION_RAW_DATA_SOURCE = "local_packet_publication_extract"

LOCAL_PUBLICATION_COLUMNS = [
    "case_key",
    "jurisdiction_id",
    "jurisdiction_name",
    "state_abbr",
    "ori9",
    "year",
    "offense",
    "count",
    "source",
    "raw_data_source",
]

PACKET_RAW_DATA_SOURCE_MAP = {
    "quincy_il": QUINCY_PUBLICATION_RAW_DATA_SOURCE,
    "brunswick_ga": BRUNSWICK_PUBLICATION_RAW_DATA_SOURCE,
}
LOCAL_PUBLICATION_PACKET_METADATA_FILES = (
    "packet_manifest.json",
    "recommendation.csv",
)


def get_v2_local_publication_input_path(paths: RepoPaths) -> Path:
    return paths.state_dir / "modeling" / "inputs" / "local_publication_annual.parquet"


def get_v2_local_publication_input_root(paths: RepoPaths) -> Path:
    return paths.state_dir / "modeling" / "inputs" / "local_publication"


def _empty_local_publication_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LOCAL_PUBLICATION_COLUMNS)


def _read_packet_manifest(packet_dir: Path) -> dict[str, object] | None:
    path = packet_dir / "packet_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_packet_recommendation(packet_dir: Path) -> pd.Series | None:
    path = packet_dir / "recommendation.csv"
    if not path.exists():
        return None
    recommendation = pd.read_csv(path)
    if recommendation.empty:
        return None
    return recommendation.iloc[0]


def _packet_is_production_ready(packet_dir: Path) -> bool:
    recommendation = _read_packet_recommendation(packet_dir)
    if recommendation is None:
        return False
    return str(recommendation.get("production_ready", "")).strip().lower() == "yes"


def _build_exclusive_ori_context(paths: RepoPaths) -> pd.DataFrame:
    crosswalk = pd.read_parquet(paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet")
    if "ori" in crosswalk.columns:
        crosswalk = crosswalk.rename(columns={"ori": "ori9"})
    crosswalk["weight"] = pd.to_numeric(crosswalk["weight"], errors="coerce").fillna(0.0)
    eligible = crosswalk[
        crosswalk["relationship_type"].astype("string").eq("exclusive")
        & crosswalk["weight"].gt(0.999)
    ][["jurisdiction_id", "ori9"]].drop_duplicates()
    if eligible.empty:
        return pd.DataFrame(columns=["jurisdiction_id", "ori9", "jurisdiction_name", "state_abbr"])

    jurisdictions = pd.read_parquet(
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        columns=["jurisdiction_id", "jurisdiction_name", "state_abbr"],
    )
    eligible = eligible.merge(jurisdictions, on="jurisdiction_id", how="left")
    eligible["state_abbr"] = eligible["state_abbr"].astype("string").str.upper()
    return eligible.drop_duplicates(subset=["jurisdiction_id", "ori9"], keep="first").reset_index(drop=True)


def _attach_packet_context(
    *,
    rows: pd.DataFrame,
    packet_dir: Path,
    ori_context: pd.DataFrame,
) -> pd.DataFrame:
    if rows.empty:
        return _empty_local_publication_frame()
    manifest = _read_packet_manifest(packet_dir)
    if not manifest:
        return _empty_local_publication_frame()

    recommendation = _read_packet_recommendation(packet_dir)
    jurisdiction_id = manifest.get("jurisdiction_id") or (None if recommendation is None else recommendation.get("jurisdiction_id"))
    case_key = manifest.get("case_key") or packet_dir.name
    if not jurisdiction_id:
        return _empty_local_publication_frame()

    packet_context = ori_context[ori_context["jurisdiction_id"].astype("string").eq(str(jurisdiction_id))].copy()
    if packet_context.empty:
        return _empty_local_publication_frame()

    reporting_ori9 = manifest.get("reporting_ori9")
    if reporting_ori9 is None and recommendation is not None and "reporting_ori9" in recommendation.index:
        reporting_ori9 = recommendation.get("reporting_ori9")
    reporting_ori9 = None if pd.isna(reporting_ori9) else str(reporting_ori9).strip()

    if reporting_ori9:
        packet_context = packet_context[packet_context["ori9"].astype("string").eq(reporting_ori9)].copy()
    else:
        ori_counts = (
            packet_context.groupby("jurisdiction_id", dropna=False)["ori9"]
            .nunique()
            .rename("ori_count")
            .reset_index()
        )
        packet_context = packet_context.merge(ori_counts, on="jurisdiction_id", how="left")
        packet_context = packet_context[packet_context["ori_count"].eq(1)].drop(columns=["ori_count"])
    if packet_context.empty:
        return _empty_local_publication_frame()

    packet_context = packet_context.iloc[[0]].copy()
    packet_context["case_key"] = str(case_key)
    rows["jurisdiction_id"] = str(jurisdiction_id)
    rows["raw_data_source"] = rows["raw_data_source"].fillna(
        PACKET_RAW_DATA_SOURCE_MAP.get(packet_dir.name, GENERIC_PACKET_PUBLICATION_RAW_DATA_SOURCE)
    )
    rows = rows.merge(
        packet_context[["jurisdiction_id", "ori9", "jurisdiction_name", "state_abbr", "case_key"]],
        on="jurisdiction_id",
        how="inner",
    )
    if rows.empty:
        return _empty_local_publication_frame()
    rows["source"] = LOCAL_PUBLICATION_SOURCE
    return rows[LOCAL_PUBLICATION_COLUMNS].copy()


def _parse_quincy_style_extract(extract_path: Path, *, years: set[int]) -> pd.DataFrame:
    raw = pd.read_csv(extract_path)
    if raw.empty or "v2_offense" not in raw.columns:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for year in sorted(years):
        count_col = f"count_{int(year)}"
        if count_col not in raw.columns:
            continue
        part = raw[raw["v2_offense"].notna()].copy()
        part["year"] = int(year)
        part["offense"] = part["v2_offense"].astype("string")
        part["count"] = pd.to_numeric(part[count_col], errors="coerce")
        part["raw_data_source"] = pd.NA
        part = part[part["count"].notna()].copy()
        if not part.empty:
            frames.append(part[["year", "offense", "count", "raw_data_source"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _parse_published_reference_extract(extract_path: Path, *, years: set[int]) -> pd.DataFrame:
    raw = pd.read_csv(extract_path)
    required = {"year", "extract_role", "published_total", "comparison_target_v2_offense"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()
    rows = raw[
        raw["extract_role"].astype("string").eq("v2_core_exact")
        & raw["comparison_target_v2_offense"].notna()
    ].copy()
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce").astype("Int64")
    rows = rows[rows["year"].isin(sorted(years))].copy()
    rows["offense"] = rows["comparison_target_v2_offense"].astype("string")
    rows["count"] = pd.to_numeric(rows["published_total"], errors="coerce")
    rows["raw_data_source"] = pd.NA
    rows = rows[rows["count"].notna()].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["year"] = rows["year"].astype(int)
    return rows[["year", "offense", "count", "raw_data_source"]]


def _iter_supported_extract_paths(packet_dir: Path) -> list[Path]:
    supported: list[Path] = []
    published_reference = packet_dir / "published_reference_extract.csv"
    if published_reference.exists():
        supported.append(published_reference)
    supported.extend(sorted(packet_dir.glob("*_offense_extract.csv")))
    return supported


def _sync_local_publication_packet_inputs(
    *,
    paths: RepoPaths,
    out_root: Path,
) -> dict[str, object]:
    packet_root = paths.review_packets_dir / "municipal_targets"
    out_root.mkdir(parents=True, exist_ok=True)
    if not packet_root.exists():
        return {"packets": 0, "files": 0, "out_root": str(out_root), "copied_case_keys": []}

    copied_files = 0
    copied_case_keys: list[str] = []
    live_case_keys: set[str] = set()
    skipped_metadata_only_case_keys: list[str] = []

    for packet_dir in sorted(path for path in packet_root.iterdir() if path.is_dir()):
        case_key = packet_dir.name
        out_dir = out_root / case_key
        if not _packet_is_production_ready(packet_dir):
            if out_dir.exists():
                shutil.rmtree(out_dir)
            continue

        extract_paths = _iter_supported_extract_paths(packet_dir)
        if not extract_paths:
            skipped_metadata_only_case_keys.append(case_key)
            if out_dir.exists():
                shutil.rmtree(out_dir)
            continue

        supported_paths = [
            packet_dir / filename for filename in LOCAL_PUBLICATION_PACKET_METADATA_FILES if (packet_dir / filename).exists()
        ]
        supported_paths.extend(extract_paths)
        if not supported_paths:
            if out_dir.exists():
                shutil.rmtree(out_dir)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        live_case_keys.add(case_key)
        copied_case_keys.append(case_key)
        wanted_files = {path.name for path in supported_paths}

        for src in supported_paths:
            shutil.copyfile(src, out_dir / src.name)
            copied_files += 1

        for stale in out_dir.iterdir():
            if stale.is_file() and stale.name not in wanted_files:
                stale.unlink()

    for stale_dir in sorted(path for path in out_root.iterdir() if path.is_dir() and path.name not in live_case_keys):
        shutil.rmtree(stale_dir)

    return {
        "packets": len(copied_case_keys),
        "files": copied_files,
        "out_root": str(out_root),
        "copied_case_keys": copied_case_keys,
        "skipped_metadata_only_case_keys": skipped_metadata_only_case_keys,
    }


def _load_packet_extract_rows(
    *,
    packet_dir: Path,
    years: set[int],
    ori_context: pd.DataFrame,
) -> pd.DataFrame:
    extract_frames: list[pd.DataFrame] = []
    for extract_path in _iter_supported_extract_paths(packet_dir):
        parsed = _parse_quincy_style_extract(extract_path, years=years)
        if parsed.empty:
            parsed = _parse_published_reference_extract(extract_path, years=years)
        if not parsed.empty:
            extract_frames.append(parsed)
    if not extract_frames:
        return _empty_local_publication_frame()
    rows = pd.concat(extract_frames, ignore_index=True, sort=False)
    return _attach_packet_context(rows=rows, packet_dir=packet_dir, ori_context=ori_context)


def _build_local_publication_annual_ags_rows(
    *,
    paths: RepoPaths,
    packet_root: Path | None = None,
    year_start: int = 2018,
    year_end: int = 2024,
) -> pd.DataFrame:
    years = {int(year) for year in range(int(year_start), int(year_end) + 1)}
    if not years:
        return _empty_local_publication_frame()

    packet_root = packet_root or get_v2_local_publication_input_root(paths)
    if not packet_root.exists():
        return _empty_local_publication_frame()

    ori_context = _build_exclusive_ori_context(paths)
    if ori_context.empty:
        return _empty_local_publication_frame()

    frames: list[pd.DataFrame] = []
    for packet_dir in sorted(path for path in packet_root.iterdir() if path.is_dir()):
        if not _packet_is_production_ready(packet_dir):
            continue
        rows = _load_packet_extract_rows(
            packet_dir=packet_dir,
            years=years,
            ori_context=ori_context,
        )
        if not rows.empty:
            frames.append(rows)

    if not frames:
        return _empty_local_publication_frame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["state_abbr"] = out["state_abbr"].astype("string").str.upper()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["offense"] = out["offense"].astype("string")
    out["ori9"] = out["ori9"].astype("string")
    out["raw_data_source"] = out["raw_data_source"].astype("string")
    return out.drop_duplicates(subset=["ori9", "year", "offense", "source"], keep="first").reset_index(drop=True)


def _load_local_publication_rows_from_path(
    *,
    source_path: Path,
    year_start: int,
    year_end: int,
    require_source_exists: bool = False,
) -> pd.DataFrame:
    if not source_path.exists():
        if require_source_exists:
            raise FileNotFoundError(
                "Missing canonical local-publication input surface: "
                f"{source_path}. Run `python main.py promote-local-publications` first."
            )
        return _empty_local_publication_frame()
    rows = pd.read_parquet(source_path).copy()
    if rows.empty:
        return _empty_local_publication_frame()
    years = {int(year) for year in range(int(year_start), int(year_end) + 1)}
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce").astype("Int64")
    rows = rows[rows["year"].isin(sorted(years))].copy()
    if rows.empty:
        return _empty_local_publication_frame()
    rows["year"] = rows["year"].astype(int)
    rows["state_abbr"] = rows["state_abbr"].astype("string").str.upper()
    rows["count"] = pd.to_numeric(rows["count"], errors="coerce")
    rows["offense"] = rows["offense"].astype("string")
    rows["ori9"] = rows["ori9"].astype("string")
    rows["raw_data_source"] = rows["raw_data_source"].astype("string")
    return rows[LOCAL_PUBLICATION_COLUMNS].drop_duplicates(
        subset=["ori9", "year", "offense", "source"],
        keep="first",
    ).reset_index(drop=True)


def promote_v2_local_publication_inputs(
    *,
    paths: RepoPaths,
    out_path: Path | None = None,
    out_root: Path | None = None,
    year_start: int = 2018,
    year_end: int = 2024,
) -> dict[str, object]:
    out_path = out_path or get_v2_local_publication_input_path(paths)
    out_root = out_root or get_v2_local_publication_input_root(paths)
    sync_summary = _sync_local_publication_packet_inputs(paths=paths, out_root=out_root)
    rows = _build_local_publication_annual_ags_rows(
        paths=paths,
        packet_root=out_root,
        year_start=year_start,
        year_end=year_end,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_path, index=False)
    return {
        "packets": int(sync_summary["packets"]),
        "files": int(sync_summary["files"]),
        "input_root": str(out_root),
        "copied_case_keys": sync_summary["copied_case_keys"],
        "rows": int(len(rows)),
        "jurisdictions": int(rows["jurisdiction_id"].nunique()) if not rows.empty else 0,
        "out_path": str(out_path),
    }


def load_local_publication_annual_ags_rows(
    *,
    paths: RepoPaths,
    year_start: int = 2018,
    year_end: int = 2024,
    source_path: Path | None = None,
    require_source_exists: bool = False,
) -> pd.DataFrame:
    source_path = source_path or get_v2_local_publication_input_path(paths)
    return _load_local_publication_rows_from_path(
        source_path=source_path,
        year_start=year_start,
        year_end=year_end,
        require_source_exists=require_source_exists,
    )
