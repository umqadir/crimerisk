from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import gzip
import http.client
from pathlib import Path
import socket
import time
from typing import Iterable
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


LODES8_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"

STATE_FIPS_TO_ABBR = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca",
    "08": "co", "09": "ct", "10": "de", "11": "dc", "12": "fl",
    "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in",
    "19": "ia", "20": "ks", "21": "ky", "22": "la", "23": "me",
    "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms",
    "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh",
    "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
    "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri",
    "45": "sc", "46": "sd", "47": "tn", "48": "tx", "49": "ut",
    "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi",
    "56": "wy", "72": "pr",
}

WAC_KEEP_COLS = [
    "C000",
    "CA01", "CA02", "CA03",
    "CE01", "CE02", "CE03",
    "CNS01", "CNS02", "CNS03", "CNS04", "CNS05", "CNS06", "CNS07", "CNS08", "CNS09", "CNS10",
    "CNS11", "CNS12", "CNS13", "CNS14", "CNS15", "CNS16", "CNS17", "CNS18", "CNS19", "CNS20",
    "CD01", "CD02", "CD03", "CD04",
    "CS01", "CS02",
    "CFA01", "CFA02", "CFA03", "CFA04", "CFA05",
    "CFS01", "CFS02", "CFS03", "CFS04", "CFS05",
]

LODES_FLOW_BG_RAW_COLS = (
    "jobs_rac",
    "od_same_bg_jobs",
    "od_workplace_same_tract_jobs",
    "od_workplace_same_county_jobs",
    "od_workplace_out_of_state_jobs",
)

LODES_FLOW_TRACT_RAW_COLS = (
    "jobs_rac",
    "od_same_tract_jobs",
    "od_workplace_same_county_jobs",
    "od_workplace_out_of_state_jobs",
)

LODES_FLOW_RAW_COLS = frozenset((*LODES_FLOW_BG_RAW_COLS, *LODES_FLOW_TRACT_RAW_COLS))


@dataclass(frozen=True)
class LodesStateBundle:
    state_abbr: str
    year: int
    wac: Path
    rac: Path
    od_main: Path
    od_aux: Path


def _is_valid_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f"{out_path.suffix}.part")
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            with urllib.request.urlopen(url, timeout=120) as resp:
                with tmp_path.open("wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp_path.replace(out_path)
            return
        except urllib.error.HTTPError:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, http.client.IncompleteRead, OSError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 4:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    if last_error is not None:
        raise last_error


def _lodes_filename(state_abbr: str, *, kind: str, year: int) -> str:
    st = state_abbr.lower()
    if kind in {"wac", "rac"}:
        return f"{st}_{kind}_S000_JT00_{year}.csv.gz"
    if kind == "od_main":
        return f"{st}_od_main_JT00_{year}.csv.gz"
    if kind == "od_aux":
        return f"{st}_od_aux_JT00_{year}.csv.gz"
    raise ValueError(f"Unsupported LODES kind: {kind}")


def _lodes_url(state_abbr: str, *, kind: str, year: int) -> str:
    st = state_abbr.lower()
    filename = _lodes_filename(st, kind=kind, year=year)
    if kind in {"wac", "rac"}:
        return f"{LODES8_BASE}/{st}/{kind}/{filename}"
    if kind in {"od_main", "od_aux"}:
        return f"{LODES8_BASE}/{st}/od/{filename}"
    raise ValueError(f"Unsupported LODES kind: {kind}")


def _resolve_lodes_file_for_year(
    *,
    state_abbr: str,
    kind: str,
    year: int,
    local_dir: Path,
    cache_dir: Path,
) -> Path | None:
    filename = _lodes_filename(state_abbr, kind=kind, year=year)
    local = local_dir / filename
    if local.exists():
        if _is_valid_gzip(local):
            return local
        local.unlink(missing_ok=True)

    cached = cache_dir / "lodes" / state_abbr.lower() / filename
    if cached.exists():
        if _is_valid_gzip(cached):
            return cached
        cached.unlink(missing_ok=True)
    if cached.exists():
        cached.unlink()

    try:
        _download(_lodes_url(state_abbr, kind=kind, year=year), cached)
    except urllib.error.HTTPError as e:
        if e.code in {403, 404}:
            return None
        raise

    return cached if _is_valid_gzip(cached) else None


def resolve_state_lodes_bundle(
    *,
    state_abbr: str,
    local_dir: Path,
    cache_dir: Path,
    preferred_year: int = 2023,
    min_year: int = 2015,
) -> LodesStateBundle | None:
    for year in range(preferred_year, min_year - 1, -1):
        resolved = {
            kind: _resolve_lodes_file_for_year(
                state_abbr=state_abbr,
                kind=kind,
                year=year,
                local_dir=local_dir,
                cache_dir=cache_dir,
            )
            for kind in ("wac", "rac", "od_main", "od_aux")
        }
        if all(path is not None for path in resolved.values()):
            return LodesStateBundle(
                state_abbr=state_abbr.lower(),
                year=year,
                wac=resolved["wac"],  # type: ignore[arg-type]
                rac=resolved["rac"],  # type: ignore[arg-type]
                od_main=resolved["od_main"],  # type: ignore[arg-type]
                od_aux=resolved["od_aux"],  # type: ignore[arg-type]
            )
    return None


def infer_state_abbrs_from_lodes_dir(lodes_dir: Path) -> list[str]:
    state_abbrs = sorted({p.name[:2].lower() for p in lodes_dir.glob("*_wac_S000_JT00_*.csv.gz")})
    if state_abbrs:
        return state_abbrs
    return sorted(set(STATE_FIPS_TO_ABBR.values()))


def _accumulate_series(target: defaultdict[str, int], series: pd.Series, *, width: int) -> None:
    for key, value in series.items():
        if pd.isna(key):
            continue
        target[str(key).zfill(width)] += int(value or 0)


def _combine_metric_maps(
    *,
    id_col: str,
    width: int,
    metric_maps: dict[str, defaultdict[str, int]],
) -> pd.DataFrame:
    keys = sorted(set().union(*(metric_map.keys() for metric_map in metric_maps.values())))
    if not keys:
        return pd.DataFrame(columns=[id_col, *metric_maps.keys()])
    rows = []
    for key in keys:
        row = {id_col: str(key).zfill(width)}
        for metric_name, metric_map in metric_maps.items():
            row[metric_name] = int(metric_map.get(key, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_wac_to_bg(path: Path) -> pd.DataFrame:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {col: 0 for col in WAC_KEEP_COLS})
    for chunk in pd.read_csv(
        path,
        usecols=["w_geocode", *WAC_KEEP_COLS],
        dtype={**{"w_geocode": str}, **{col: "Int64" for col in WAC_KEEP_COLS}},
        chunksize=500_000,
    ):
        chunk = chunk.dropna(subset=["w_geocode", "C000"])
        grouped = chunk.assign(bg_id=chunk["w_geocode"].str.slice(0, 12)).groupby("bg_id")[WAC_KEEP_COLS].sum()
        for bg_id, row in grouped.iterrows():
            key = str(bg_id).zfill(12)
            bucket = totals[key]
            for col in WAC_KEEP_COLS:
                bucket[col] += int(row.get(col, 0) or 0)
    if not totals:
        return pd.DataFrame(columns=["bg_id", *WAC_KEEP_COLS, "jobs_wac"])
    df = pd.DataFrame([{"bg_id": bg_id, **vals} for bg_id, vals in totals.items()])
    df["jobs_wac"] = pd.to_numeric(df["C000"], errors="coerce").fillna(0).astype("Int64")
    return df


def _aggregate_rac(path: Path, *, id_col: str, width: int) -> pd.DataFrame:
    totals: defaultdict[str, int] = defaultdict(int)
    for chunk in pd.read_csv(
        path,
        usecols=["h_geocode", "C000"],
        dtype={"h_geocode": str, "C000": "Int64"},
        chunksize=500_000,
    ):
        chunk = chunk.dropna(subset=["h_geocode", "C000"])
        grouped = chunk.assign(geo_id=chunk["h_geocode"].str.slice(0, width)).groupby("geo_id")["C000"].sum()
        _accumulate_series(totals, grouped, width=width)
    df = _combine_metric_maps(id_col=id_col, width=width, metric_maps={"jobs_rac": totals})
    if "jobs_rac" in df.columns:
        df["jobs_rac"] = pd.to_numeric(df["jobs_rac"], errors="coerce").fillna(0).astype("Int64")
    return df


def _aggregate_wac_jobs(path: Path, *, id_col: str, width: int) -> pd.DataFrame:
    totals: defaultdict[str, int] = defaultdict(int)
    for chunk in pd.read_csv(
        path,
        usecols=["w_geocode", "C000"],
        dtype={"w_geocode": str, "C000": "Int64"},
        chunksize=500_000,
    ):
        chunk = chunk.dropna(subset=["w_geocode", "C000"])
        grouped = chunk.assign(geo_id=chunk["w_geocode"].str.slice(0, width)).groupby("geo_id")["C000"].sum()
        _accumulate_series(totals, grouped, width=width)
    df = _combine_metric_maps(id_col=id_col, width=width, metric_maps={"jobs_wac": totals})
    if "jobs_wac" in df.columns:
        df["jobs_wac"] = pd.to_numeric(df["jobs_wac"], errors="coerce").fillna(0).astype("Int64")
    return df


def _aggregate_od_to_bg(paths: Iterable[Path]) -> pd.DataFrame:
    same_bg: defaultdict[str, int] = defaultdict(int)
    workplace_same_tract: defaultdict[str, int] = defaultdict(int)
    workplace_same_county: defaultdict[str, int] = defaultdict(int)
    workplace_out_of_state: defaultdict[str, int] = defaultdict(int)

    for path in paths:
        for chunk in pd.read_csv(
            path,
            usecols=["w_geocode", "h_geocode", "S000"],
            dtype={"w_geocode": str, "h_geocode": str, "S000": "Int64"},
            chunksize=500_000,
        ):
            chunk = chunk.dropna(subset=["w_geocode", "h_geocode", "S000"])
            base = pd.DataFrame(
                {
                    "w_bg": chunk["w_geocode"].str.slice(0, 12),
                    "h_bg": chunk["h_geocode"].str.slice(0, 12),
                    "w_tract": chunk["w_geocode"].str.slice(0, 11),
                    "h_tract": chunk["h_geocode"].str.slice(0, 11),
                    "w_county": chunk["w_geocode"].str.slice(0, 5),
                    "h_county": chunk["h_geocode"].str.slice(0, 5),
                    "w_state": chunk["w_geocode"].str.slice(0, 2),
                    "h_state": chunk["h_geocode"].str.slice(0, 2),
                    "S000": pd.to_numeric(chunk["S000"], errors="coerce").fillna(0).astype("int64"),
                }
            )

            same_bg_mask = base["w_bg"].eq(base["h_bg"])
            same_tract_mask = base["w_tract"].eq(base["h_tract"])
            same_county_mask = base["w_county"].eq(base["h_county"])
            out_of_state_mask = base["w_state"].ne(base["h_state"])

            _accumulate_series(same_bg, base.loc[same_bg_mask].groupby("w_bg")["S000"].sum(), width=12)
            _accumulate_series(
                workplace_same_tract,
                base.loc[same_tract_mask].groupby("w_bg")["S000"].sum(),
                width=12,
            )
            _accumulate_series(
                workplace_same_county,
                base.loc[same_county_mask].groupby("w_bg")["S000"].sum(),
                width=12,
            )
            _accumulate_series(
                workplace_out_of_state,
                base.loc[out_of_state_mask].groupby("w_bg")["S000"].sum(),
                width=12,
            )

    return _combine_metric_maps(
        id_col="bg_id",
        width=12,
        metric_maps={
            "od_same_bg_jobs": same_bg,
            "od_workplace_same_tract_jobs": workplace_same_tract,
            "od_workplace_same_county_jobs": workplace_same_county,
            "od_workplace_out_of_state_jobs": workplace_out_of_state,
        },
    )


def _aggregate_od_to_tract(paths: Iterable[Path]) -> pd.DataFrame:
    same_tract: defaultdict[str, int] = defaultdict(int)
    workplace_same_county: defaultdict[str, int] = defaultdict(int)
    workplace_out_of_state: defaultdict[str, int] = defaultdict(int)

    for path in paths:
        for chunk in pd.read_csv(
            path,
            usecols=["w_geocode", "h_geocode", "S000"],
            dtype={"w_geocode": str, "h_geocode": str, "S000": "Int64"},
            chunksize=500_000,
        ):
            chunk = chunk.dropna(subset=["w_geocode", "h_geocode", "S000"])
            base = pd.DataFrame(
                {
                    "w_tract": chunk["w_geocode"].str.slice(0, 11),
                    "h_tract": chunk["h_geocode"].str.slice(0, 11),
                    "w_county": chunk["w_geocode"].str.slice(0, 5),
                    "h_county": chunk["h_geocode"].str.slice(0, 5),
                    "w_state": chunk["w_geocode"].str.slice(0, 2),
                    "h_state": chunk["h_geocode"].str.slice(0, 2),
                    "S000": pd.to_numeric(chunk["S000"], errors="coerce").fillna(0).astype("int64"),
                }
            )

            same_tract_mask = base["w_tract"].eq(base["h_tract"])
            same_county_mask = base["w_county"].eq(base["h_county"])
            out_of_state_mask = base["w_state"].ne(base["h_state"])

            _accumulate_series(same_tract, base.loc[same_tract_mask].groupby("w_tract")["S000"].sum(), width=11)
            _accumulate_series(
                workplace_same_county,
                base.loc[same_county_mask].groupby("w_tract")["S000"].sum(),
                width=11,
            )
            _accumulate_series(
                workplace_out_of_state,
                base.loc[out_of_state_mask].groupby("w_tract")["S000"].sum(),
                width=11,
            )

    return _combine_metric_maps(
        id_col="tract_id",
        width=11,
        metric_maps={
            "od_same_tract_jobs": same_tract,
            "od_workplace_same_county_jobs": workplace_same_county,
            "od_workplace_out_of_state_jobs": workplace_out_of_state,
        },
    )


def build_state_block_group_lodes_frame(
    *,
    state_fips: str,
    lodes_dir: Path,
    cache_dir: Path,
    preferred_year: int = 2023,
    min_year: int = 2015,
) -> tuple[pd.DataFrame | None, int | None]:
    state_abbr = STATE_FIPS_TO_ABBR.get(str(state_fips).zfill(2))
    if not state_abbr:
        return None, None

    bundle = resolve_state_lodes_bundle(
        state_abbr=state_abbr,
        local_dir=lodes_dir,
        cache_dir=cache_dir,
        preferred_year=preferred_year,
        min_year=min_year,
    )
    if bundle is None:
        return None, None

    wac = _aggregate_wac_to_bg(bundle.wac)
    rac = _aggregate_rac(bundle.rac, id_col="bg_id", width=12)
    od = _aggregate_od_to_bg([bundle.od_main, bundle.od_aux])

    df = wac.merge(rac, on="bg_id", how="outer").merge(od, on="bg_id", how="outer")
    numeric_cols = [c for c in df.columns if c != "bg_id"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int64")
    df["bg_id"] = df["bg_id"].astype(str).str.zfill(12)
    df["tract_id"] = df["bg_id"].str.slice(0, 11)
    df["state_fips"] = df["bg_id"].str.slice(0, 2)
    df["lodes_year"] = int(bundle.year)
    return df, bundle.year


def compute_lodes_flow_frame_by_tract(
    *,
    lodes_dir: Path,
    cache_dir: Path,
    state_abbrs: Iterable[str] | None = None,
    preferred_year: int = 2023,
    min_year: int = 2015,
) -> pd.DataFrame:
    if state_abbrs is None:
        state_abbrs = infer_state_abbrs_from_lodes_dir(lodes_dir)

    frames: list[pd.DataFrame] = []
    for state_abbr in state_abbrs:
        bundle = resolve_state_lodes_bundle(
            state_abbr=state_abbr,
            local_dir=lodes_dir,
            cache_dir=cache_dir,
            preferred_year=preferred_year,
            min_year=min_year,
        )
        if bundle is None:
            continue

        wac = _aggregate_wac_jobs(bundle.wac, id_col="tract_id", width=11)
        rac = _aggregate_rac(bundle.rac, id_col="tract_id", width=11)
        od = _aggregate_od_to_tract([bundle.od_main, bundle.od_aux])
        tract = wac.merge(rac, on="tract_id", how="outer").merge(od, on="tract_id", how="outer")
        numeric_cols = [c for c in tract.columns if c != "tract_id"]
        for col in numeric_cols:
            tract[col] = pd.to_numeric(tract[col], errors="coerce").fillna(0).astype("Int64")
        tract["tract_id"] = tract["tract_id"].astype(str).str.zfill(11)
        tract["state_fips"] = tract["tract_id"].str.slice(0, 2)
        frames.append(tract)

    if not frames:
        return pd.DataFrame(
            columns=[
                "tract_id",
                "jobs_wac",
                *LODES_FLOW_TRACT_RAW_COLS,
                "state_fips",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def apply_lodes_flow_daytime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _series(col: str) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    jobs_wac = _series("jobs_wac")
    jobs_rac = _series("jobs_rac")
    pop = _series("total_population")

    if jobs_wac.notna().any() and jobs_rac.notna().any():
        with np.errstate(divide="ignore", invalid="ignore"):
            out["resident_jobs_per_capita"] = jobs_rac / pop
            out["jobs_wac_to_rac_ratio"] = jobs_wac / jobs_rac
            out["net_commuter_jobs_per_capita"] = (jobs_wac - jobs_rac) / pop
            daytime_proxy = pop + jobs_wac - jobs_rac
            out["daytime_population_jobs_proxy_ratio"] = daytime_proxy / pop

        out["resident_jobs_per_capita"] = (
            pd.to_numeric(out["resident_jobs_per_capita"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=0.0)
        )
        out["jobs_wac_to_rac_ratio"] = (
            pd.to_numeric(out["jobs_wac_to_rac_ratio"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=0.0)
        )
        out["net_commuter_jobs_per_capita"] = (
            pd.to_numeric(out["net_commuter_jobs_per_capita"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        out["daytime_population_jobs_proxy_ratio"] = (
            pd.to_numeric(out["daytime_population_jobs_proxy_ratio"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=0.0)
        )

        daytime_proxy = pd.to_numeric(daytime_proxy, errors="coerce").clip(lower=0.0)
        out["log_jobs_rac"] = np.log1p(pd.to_numeric(jobs_rac, errors="coerce").clip(lower=0.0))
        out["log_daytime_population_jobs_proxy"] = np.log1p(daytime_proxy)

        land_area_sqkm = _series("land_area_sqkm")
        if land_area_sqkm.notna().any():
            with np.errstate(divide="ignore", invalid="ignore"):
                out["daytime_population_jobs_proxy_density_sqkm"] = daytime_proxy / land_area_sqkm
            out["daytime_population_jobs_proxy_density_sqkm"] = (
                pd.to_numeric(out["daytime_population_jobs_proxy_density_sqkm"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .clip(lower=0.0)
            )
            out["log_daytime_population_jobs_proxy_density_sqkm"] = np.log1p(
                pd.to_numeric(out["daytime_population_jobs_proxy_density_sqkm"], errors="coerce").clip(lower=0.0)
            )

    def _share(numer_col: str, denom_col: str, out_col: str) -> None:
        if numer_col not in out.columns or denom_col not in out.columns:
            return
        numer = pd.to_numeric(out[numer_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        denom = pd.to_numeric(out[denom_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        values = np.full(len(out.index), np.nan, dtype=float)
        valid = np.isfinite(numer) & np.isfinite(denom) & (denom > 0)
        np.divide(numer, denom, out=values, where=valid)
        out[out_col] = pd.Series(values, index=out.index, dtype=float).clip(lower=0.0, upper=1.0)

    _share("od_same_bg_jobs", "jobs_wac", "workplace_same_bg_commute_share")
    _share("od_same_bg_jobs", "jobs_rac", "resident_same_bg_commute_share")
    _share("od_workplace_same_tract_jobs", "jobs_wac", "workplace_same_tract_commute_share")
    _share("od_resident_same_tract_jobs", "jobs_rac", "resident_same_tract_commute_share")
    _share("od_same_tract_jobs", "jobs_wac", "workplace_same_tract_commute_share")
    _share("od_same_tract_jobs", "jobs_rac", "resident_same_tract_commute_share")
    _share("od_workplace_same_county_jobs", "jobs_wac", "workplace_same_county_commute_share")
    _share("od_resident_same_county_jobs", "jobs_rac", "resident_same_county_commute_share")
    _share("od_workplace_out_of_state_jobs", "jobs_wac", "workplace_out_of_state_commute_share")
    _share("od_resident_out_of_state_jobs", "jobs_rac", "resident_out_of_state_commute_share")
    return out
