from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import struct
import subprocess
import tempfile
import gc
import zlib

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats


LAND_COVER_BUNDLE_URL = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_2015-2024_CU_C1V1.zip"
IMPERVIOUS_BUNDLE_URL = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_FctImp_2015-2024_CU_C1V1.zip"
IMPERVIOUS_DESCRIPTOR_BUNDLE_URL = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_ImpDsc_2015-2024_CU_C1V1.zip"

LAND_COVER_MEMBER_TEMPLATE = "Annual_NLCD_LndCov_{year}_CU_C1V1.tif"
IMPERVIOUS_MEMBER_TEMPLATE = "Annual_NLCD_FctImp_{year}_CU_C1V1.tif"
IMPERVIOUS_DESCRIPTOR_MEMBER_TEMPLATE = "Annual_NLCD_ImpDsc_{year}_CU_C1V1.tif"

NLCD_TARGET_CRS = "EPSG:5070"
NLCD_NODATA_VALUE = 250

LAND_COVER_CODES = {
    "open_water": (11,),
    "developed_open": (21,),
    "developed_low": (22,),
    "developed_medium": (23,),
    "developed_high": (24,),
    "barren": (31,),
    "forest": (41, 42, 43),
    "shrub": (52,),
    "grassland": (71,),
    "pasture": (81,),
    "crops": (82,),
    "woody_wetlands": (90,),
    "emergent_wetlands": (95,),
}


@dataclass(frozen=True)
class NlcdBuildConfig:
    release_year: int = 2024
    land_cover_year: int = 2023
    impervious_year: int = 2023
    bbox_buffer_m: float = 60.0
    max_tile_size_m: float = 200_000.0
    timeout_seconds: int = 1_800
    impervious_gt20_threshold: float = 20.0
    impervious_gt50_threshold: float = 50.0


@dataclass(frozen=True)
class _ZipMemberMeta:
    compression_method: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def _load_block_groups_for_state(*, bg_zip: Path, active_bg_ids: set[str] | None = None) -> gpd.GeoDataFrame:
    bg = gpd.read_file(f"zip://{bg_zip}")[
        ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE", "geometry"]
    ].copy()
    bg["bg_id"] = bg["GEOID"].astype("string").str.zfill(12)
    if active_bg_ids is not None:
        bg = bg[bg["bg_id"].isin(active_bg_ids)].copy()
    bg["tract_id"] = bg["bg_id"].str.slice(0, 11)
    bg["state_fips"] = bg["STATEFP"].astype("string").str.zfill(2)
    bg["county_fips"] = bg["COUNTYFP"].astype("string").str.zfill(3)
    return bg[["bg_id", "tract_id", "state_fips", "county_fips", "geometry"]]


def _load_active_bg_ids_by_state(*, block_group_crosswalk_parquet: Path) -> dict[str, set[str]]:
    crosswalk = pd.read_parquet(block_group_crosswalk_parquet)[["state_fips", "block_group_geoid"]].drop_duplicates().copy()
    crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    out: dict[str, set[str]] = {}
    for state_fips, grp in crosswalk.groupby("state_fips", dropna=False):
        out[str(state_fips).zfill(2)] = set(grp["block_group_geoid"].tolist())
    return out


def _empty_nlcd_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bg_id",
            "tract_id",
            "state_fips",
            "county_fips",
            "nlcd_land_cover_year",
            "nlcd_impervious_year",
            "nlcd_valid_pixel_count",
            "nlcd_total_pixel_count",
            "nlcd_nodata_share",
            "nlcd_developed_low_share",
            "nlcd_developed_high_share",
            "nlcd_forest_share",
            "nlcd_agriculture_share",
            "nlcd_wetland_water_share",
            "nlcd_shrub_grass_barren_share",
            "nlcd_impervious_mean",
            "nlcd_impervious_p90",
            "nlcd_impervious_ge20_share",
            "nlcd_impervious_ge50_share",
            "nlcd_nonroad_urban_descriptor_share",
            "nlcd_road_descriptor_share",
        ]
    )


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _curl_range_to_file(*, url: str, start: int, end: int, out_path: Path, timeout_seconds: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(int(timeout_seconds)),
            "-r",
            f"{int(start)}-{int(end)}",
            "-o",
            str(out_path),
            url,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"NLCD range request failed for {out_path.name}: {proc.stderr.strip()}")


def _remote_content_length(*, url: str, timeout_seconds: int) -> int:
    proc = subprocess.run(
        [
            "curl",
            "-I",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(int(timeout_seconds)),
            url,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"NLCD HEAD failed for {url}: {proc.stderr.strip()}")
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"NLCD HEAD missing content-length for {url}")


def _read_zip_central_directory(*, url: str, timeout_seconds: int) -> bytes:
    total_size = _remote_content_length(url=url, timeout_seconds=timeout_seconds)
    tail_start = max(0, total_size - 262_144)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tail_path = Path(tmp.name)
    try:
        _curl_range_to_file(
            url=url,
            start=tail_start,
            end=total_size - 1,
            out_path=tail_path,
            timeout_seconds=timeout_seconds,
        )
        tail = tail_path.read_bytes()
    finally:
        tail_path.unlink(missing_ok=True)

    locator_idx = tail.rfind(b"PK\x06\x07")
    if locator_idx < 0:
        raise RuntimeError(f"ZIP64 locator not found for {url}")
    _, _, zip64_eocd_offset, _ = struct.unpack("<4sLQL", tail[locator_idx:locator_idx + 20])

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        zip64_path = Path(tmp.name)
    try:
        _curl_range_to_file(
            url=url,
            start=zip64_eocd_offset,
            end=total_size - 1,
            out_path=zip64_path,
            timeout_seconds=timeout_seconds,
        )
        zip64_eocd = zip64_path.read_bytes()
    finally:
        zip64_path.unlink(missing_ok=True)

    fields = struct.unpack("<4sQ2H2L4Q", zip64_eocd[:56])
    cd_size = int(fields[8])
    cd_offset = int(fields[9])

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        cd_path = Path(tmp.name)
    try:
        _curl_range_to_file(
            url=url,
            start=cd_offset,
            end=cd_offset + cd_size - 1,
            out_path=cd_path,
            timeout_seconds=timeout_seconds,
        )
        return cd_path.read_bytes()
    finally:
        cd_path.unlink(missing_ok=True)


def _zip64_extra_fields(extra: bytes, *, comp_size_32: int, uncomp_size_32: int, local_offset_32: int) -> dict[str, int]:
    out: dict[str, int] = {}
    idx = 0
    while idx + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[idx:idx + 4])
        payload = extra[idx + 4:idx + 4 + data_size]
        if header_id == 0x0001:
            j = 0
            if uncomp_size_32 == 0xFFFFFFFF:
                out["uncompressed_size"] = int(struct.unpack("<Q", payload[j:j + 8])[0])
                j += 8
            if comp_size_32 == 0xFFFFFFFF:
                out["compressed_size"] = int(struct.unpack("<Q", payload[j:j + 8])[0])
                j += 8
            if local_offset_32 == 0xFFFFFFFF:
                out["local_header_offset"] = int(struct.unpack("<Q", payload[j:j + 8])[0])
                j += 8
        idx += 4 + data_size
    return out


def _remote_zip_member_meta(*, url: str, member_name: str, timeout_seconds: int) -> _ZipMemberMeta:
    cd = _read_zip_central_directory(url=url, timeout_seconds=timeout_seconds)
    idx = 0
    while idx + 46 <= len(cd):
        if cd[idx:idx + 4] != b"PK\x01\x02":
            break
        header = struct.unpack("<4s6H3L5H2L", cd[idx:idx + 46])
        compression_method = int(header[4])
        compressed_size_32 = int(header[8])
        uncompressed_size_32 = int(header[9])
        file_name_len = int(header[10])
        extra_len = int(header[11])
        comment_len = int(header[12])
        local_header_offset_32 = int(header[16])
        file_name = cd[idx + 46:idx + 46 + file_name_len].decode("utf-8")
        extra = cd[idx + 46 + file_name_len:idx + 46 + file_name_len + extra_len]
        zip64 = _zip64_extra_fields(
            extra,
            comp_size_32=compressed_size_32,
            uncomp_size_32=uncompressed_size_32,
            local_offset_32=local_header_offset_32,
        )
        if file_name == member_name:
            return _ZipMemberMeta(
                compression_method=compression_method,
                compressed_size=int(zip64.get("compressed_size", compressed_size_32)),
                uncompressed_size=int(zip64.get("uncompressed_size", uncompressed_size_32)),
                local_header_offset=int(zip64.get("local_header_offset", local_header_offset_32)),
            )
        idx += 46 + file_name_len + extra_len + comment_len
    raise FileNotFoundError(f"{member_name} not found in {url}")


def _member_data_offset(*, url: str, meta: _ZipMemberMeta, timeout_seconds: int) -> int:
    prefix_len = 4_096
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        local_path = Path(tmp.name)
    try:
        _curl_range_to_file(
            url=url,
            start=meta.local_header_offset,
            end=meta.local_header_offset + prefix_len - 1,
            out_path=local_path,
            timeout_seconds=timeout_seconds,
        )
        prefix = local_path.read_bytes()
    finally:
        local_path.unlink(missing_ok=True)
    if prefix[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Local ZIP header missing at offset {meta.local_header_offset} for {url}")
    local_header = struct.unpack("<4s5H3L2H", prefix[:30])
    file_name_len = int(local_header[9])
    extra_len = int(local_header[10])
    header_len = 30 + file_name_len + extra_len
    if header_len > len(prefix):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            full_header_path = Path(tmp.name)
        try:
            _curl_range_to_file(
                url=url,
                start=meta.local_header_offset,
                end=meta.local_header_offset + header_len - 1,
                out_path=full_header_path,
                timeout_seconds=timeout_seconds,
            )
            prefix = full_header_path.read_bytes()
        finally:
            full_header_path.unlink(missing_ok=True)
    return meta.local_header_offset + header_len


def _extract_remote_zip_member(*, url: str, member_name: str, out_path: Path, timeout_seconds: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with rasterio.open(out_path):
                return out_path
        except rasterio.errors.RasterioIOError:
            out_path.unlink(missing_ok=True)

    meta = _remote_zip_member_meta(url=url, member_name=member_name, timeout_seconds=timeout_seconds)
    if meta.compression_method not in {0, 8}:
        raise RuntimeError(
            f"Unsupported ZIP compression method {meta.compression_method} for {member_name} in {url}"
        )
    data_offset = _member_data_offset(url=url, meta=meta, timeout_seconds=timeout_seconds)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        compressed_path = Path(tmp.name)
    try:
        _curl_range_to_file(
            url=url,
            start=data_offset,
            end=data_offset + meta.compressed_size - 1,
            out_path=compressed_path,
            timeout_seconds=timeout_seconds,
        )
        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_out.unlink(missing_ok=True)
        if meta.compression_method == 0:
            shutil.move(str(compressed_path), str(tmp_out))
        else:
            decompressor = zlib.decompressobj(-15)
            with compressed_path.open("rb") as src, tmp_out.open("wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(decompressor.decompress(chunk))
                dst.write(decompressor.flush())
            compressed_path.unlink(missing_ok=True)
        if tmp_out.stat().st_size != meta.uncompressed_size:
            raise RuntimeError(
                f"NLCD member size mismatch for {member_name}: expected {meta.uncompressed_size}, got {tmp_out.stat().st_size}"
            )
        try:
            with rasterio.open(tmp_out):
                pass
        except rasterio.errors.RasterioIOError as exc:
            tmp_out.unlink(missing_ok=True)
            raise RuntimeError(f"NLCD member extraction produced invalid raster for {member_name}") from exc
        os.replace(tmp_out, out_path)
        return out_path
    finally:
        compressed_path.unlink(missing_ok=True)


def _ensure_nlcd_products(*, cache_dir: Path, cfg: NlcdBuildConfig) -> tuple[Path, Path, Path]:
    product_dir = cache_dir / "products"
    land_cover_member = LAND_COVER_MEMBER_TEMPLATE.format(year=int(cfg.land_cover_year))
    impervious_member = IMPERVIOUS_MEMBER_TEMPLATE.format(year=int(cfg.impervious_year))
    descriptor_member = IMPERVIOUS_DESCRIPTOR_MEMBER_TEMPLATE.format(year=int(cfg.impervious_year))
    land_cover_path = _extract_remote_zip_member(
        url=LAND_COVER_BUNDLE_URL,
        member_name=land_cover_member,
        out_path=product_dir / land_cover_member,
        timeout_seconds=int(cfg.timeout_seconds),
    )
    impervious_path = _extract_remote_zip_member(
        url=IMPERVIOUS_BUNDLE_URL,
        member_name=impervious_member,
        out_path=product_dir / impervious_member,
        timeout_seconds=int(cfg.timeout_seconds),
    )
    descriptor_path = _extract_remote_zip_member(
        url=IMPERVIOUS_DESCRIPTOR_BUNDLE_URL,
        member_name=descriptor_member,
        out_path=product_dir / descriptor_member,
        timeout_seconds=int(cfg.timeout_seconds),
    )
    return land_cover_path, impervious_path, descriptor_path


def _sum_counts(stats: dict[object, object], codes: tuple[int, ...]) -> int:
    total = 0
    for code in codes:
        total += int(stats.get(code, 0) or 0)
    return total


def _masked_values(arr) -> np.ndarray:
    values = np.ma.compressed(np.ma.masked_invalid(arr))
    if values.size == 0:
        return np.asarray([], dtype=float)
    return values.astype(float)


def _impervious_p90(arr) -> float:
    values = _masked_values(arr)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 90))


def _impervious_gt20_share(arr) -> float:
    values = _masked_values(arr)
    if values.size == 0:
        return float("nan")
    return float((values >= 20.0).mean())


def _impervious_gt50_share(arr) -> float:
    values = _masked_values(arr)
    if values.size == 0:
        return float("nan")
    return float((values >= 50.0).mean())


def _total_pixel_count(arr) -> int:
    data = np.ma.getdata(arr)
    return int(data.size)


def _nodata_share(arr) -> float:
    data = np.ma.getdata(arr)
    if data.size == 0:
        return float("nan")
    mask = np.ma.getmaskarray(arr)
    return float(mask.mean())


def build_state_block_group_nlcd_frame(
    *,
    state_fips: str,
    bg_zip: Path,
    cache_dir: Path,
    cfg: NlcdBuildConfig = NlcdBuildConfig(),
    active_bg_ids: set[str] | None = None,
) -> pd.DataFrame:
    bg = _load_block_groups_for_state(bg_zip=bg_zip, active_bg_ids=active_bg_ids)
    if bg.empty:
        return _empty_nlcd_frame()

    land_cover_tif, impervious_tif, descriptor_tif = _ensure_nlcd_products(cache_dir=cache_dir, cfg=cfg)
    bg = bg.to_crs(NLCD_TARGET_CRS)

    lc_stats = zonal_stats(
        bg,
        land_cover_tif,
        categorical=True,
        nodata=NLCD_NODATA_VALUE,
        add_stats={
            "total_pixel_count": _total_pixel_count,
            "nodata_share": _nodata_share,
        },
    )
    impervious_stats = zonal_stats(
        bg,
        impervious_tif,
        stats=["mean"],
        nodata=NLCD_NODATA_VALUE,
        add_stats={
            "p90": _impervious_p90,
            "ge20_share": _impervious_gt20_share,
            "ge50_share": _impervious_gt50_share,
        },
    )
    descriptor_stats = zonal_stats(
        bg,
        descriptor_tif,
        categorical=True,
        nodata=NLCD_NODATA_VALUE,
    )

    rows: list[dict[str, object]] = []
    for (_, rec), lc_stat, imp_stat, desc_stat in zip(bg.iterrows(), lc_stats, impervious_stats, descriptor_stats):
        valid_pixel_count = int(
            sum(
                int(v or 0)
                for k, v in lc_stat.items()
                if isinstance(k, (int, np.integer))
            )
        )
        denom = float(valid_pixel_count) if valid_pixel_count > 0 else float("nan")
        developed_low = _sum_counts(lc_stat, LAND_COVER_CODES["developed_open"] + LAND_COVER_CODES["developed_low"])
        developed_high = _sum_counts(lc_stat, LAND_COVER_CODES["developed_medium"] + LAND_COVER_CODES["developed_high"])
        forest = _sum_counts(lc_stat, LAND_COVER_CODES["forest"])
        agriculture = _sum_counts(lc_stat, LAND_COVER_CODES["pasture"] + LAND_COVER_CODES["crops"])
        wetland_water = _sum_counts(
            lc_stat,
            LAND_COVER_CODES["open_water"] + LAND_COVER_CODES["woody_wetlands"] + LAND_COVER_CODES["emergent_wetlands"],
        )
        shrub_grass_barren = _sum_counts(
            lc_stat,
            LAND_COVER_CODES["barren"] + LAND_COVER_CODES["shrub"] + LAND_COVER_CODES["grassland"],
        )
        descriptor_nonroad = int(desc_stat.get(2, 0) or 0)
        descriptor_road = int(desc_stat.get(1, 0) or 0)
        rows.append(
            {
                "bg_id": str(rec["bg_id"]).zfill(12),
                "tract_id": str(rec["tract_id"]).zfill(11),
                "state_fips": str(rec["state_fips"]).zfill(2),
                "county_fips": str(rec["county_fips"]).zfill(3),
                "nlcd_land_cover_year": int(cfg.land_cover_year),
                "nlcd_impervious_year": int(cfg.impervious_year),
                "nlcd_valid_pixel_count": valid_pixel_count,
                "nlcd_total_pixel_count": int(lc_stat.get("total_pixel_count", 0) or 0),
                "nlcd_nodata_share": _safe_float(lc_stat.get("nodata_share", np.nan)),
                "nlcd_developed_low_share": float(developed_low / denom) if valid_pixel_count > 0 else np.nan,
                "nlcd_developed_high_share": float(developed_high / denom) if valid_pixel_count > 0 else np.nan,
                "nlcd_forest_share": float(forest / denom) if valid_pixel_count > 0 else np.nan,
                "nlcd_agriculture_share": float(agriculture / denom) if valid_pixel_count > 0 else np.nan,
                "nlcd_wetland_water_share": float(wetland_water / denom) if valid_pixel_count > 0 else np.nan,
                "nlcd_shrub_grass_barren_share": float(shrub_grass_barren / denom) if valid_pixel_count > 0 else np.nan,
                "nlcd_impervious_mean": _safe_float(imp_stat.get("mean", np.nan)),
                "nlcd_impervious_p90": _safe_float(imp_stat.get("p90", np.nan)),
                "nlcd_impervious_ge20_share": _safe_float(imp_stat.get("ge20_share", np.nan)),
                "nlcd_impervious_ge50_share": _safe_float(imp_stat.get("ge50_share", np.nan)),
                "nlcd_nonroad_urban_descriptor_share": float(descriptor_nonroad / denom) if valid_pixel_count > 0 else np.nan,
                "nlcd_road_descriptor_share": float(descriptor_road / denom) if valid_pixel_count > 0 else np.nan,
                "nlcd_count_11": int(lc_stat.get(11, 0) or 0),
                "nlcd_count_21": int(lc_stat.get(21, 0) or 0),
                "nlcd_count_22": int(lc_stat.get(22, 0) or 0),
                "nlcd_count_23": int(lc_stat.get(23, 0) or 0),
                "nlcd_count_24": int(lc_stat.get(24, 0) or 0),
                "nlcd_count_31": int(lc_stat.get(31, 0) or 0),
                "nlcd_count_41": int(lc_stat.get(41, 0) or 0),
                "nlcd_count_42": int(lc_stat.get(42, 0) or 0),
                "nlcd_count_43": int(lc_stat.get(43, 0) or 0),
                "nlcd_count_52": int(lc_stat.get(52, 0) or 0),
                "nlcd_count_71": int(lc_stat.get(71, 0) or 0),
                "nlcd_count_81": int(lc_stat.get(81, 0) or 0),
                "nlcd_count_82": int(lc_stat.get(82, 0) or 0),
                "nlcd_count_90": int(lc_stat.get(90, 0) or 0),
                "nlcd_count_95": int(lc_stat.get(95, 0) or 0),
                "nlcd_descriptor_count_0": int(desc_stat.get(0, 0) or 0),
                "nlcd_descriptor_count_1": descriptor_road,
                "nlcd_descriptor_count_2": descriptor_nonroad,
            }
        )
    return pd.DataFrame(rows)


def build_block_group_nlcd_features(
    *,
    bg_dir: Path,
    block_group_crosswalk_parquet: Path,
    state_cache_dir: Path,
    state_out_dir: Path,
    cfg: NlcdBuildConfig = NlcdBuildConfig(),
) -> pd.DataFrame:
    active_bg_ids_by_state = _load_active_bg_ids_by_state(block_group_crosswalk_parquet=block_group_crosswalk_parquet)
    product_mtime = max(path.stat().st_mtime for path in _ensure_nlcd_products(cache_dir=state_cache_dir, cfg=cfg))
    state_order = sorted(active_bg_ids_by_state)
    for idx, state_fips in enumerate(state_order, start=1):
        bg_zip = bg_dir / f"tl_2020_{str(state_fips).zfill(2)}_bg.zip"
        if not bg_zip.exists():
            continue
        out_path = state_out_dir / f"{str(state_fips).zfill(2)}.parquet"
        active_bg_ids = active_bg_ids_by_state[str(state_fips).zfill(2)]
        if out_path.exists() and out_path.stat().st_size > 0 and out_path.stat().st_mtime >= product_mtime:
            frame = pd.read_parquet(out_path)
            frame["bg_id"] = frame["bg_id"].astype("string").str.zfill(12)
            current_bg_ids = set(frame["bg_id"].tolist())
            if current_bg_ids == active_bg_ids:
                print(f"nlcd: [{idx}/{len(state_order)}] state={str(state_fips).zfill(2)} reuse rows={len(frame):,}")
                del frame
                gc.collect()
                continue
        print(f"nlcd: [{idx}/{len(state_order)}] state={str(state_fips).zfill(2)} build start")
        frame = build_state_block_group_nlcd_frame(
            state_fips=str(state_fips).zfill(2),
            bg_zip=bg_zip,
            cache_dir=state_cache_dir,
            cfg=cfg,
            active_bg_ids=active_bg_ids,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_path, index=False)
        print(f"nlcd: [{idx}/{len(state_order)}] state={str(state_fips).zfill(2)} build done rows={len(frame):,}")
        del frame
        gc.collect()
    shard_paths = sorted(state_out_dir.glob("*.parquet"))
    if not shard_paths:
        return _empty_nlcd_frame()
    out = pd.concat((pd.read_parquet(path) for path in shard_paths), ignore_index=True)
    out["bg_id"] = out["bg_id"].astype("string").str.zfill(12)
    out["tract_id"] = out["tract_id"].astype("string").str.zfill(11)
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    out["county_fips"] = out["county_fips"].astype("string").str.zfill(3)
    return out.sort_values(["state_fips", "bg_id"], kind="stable").reset_index(drop=True)
