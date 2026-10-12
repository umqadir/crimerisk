from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import ot
import pandas as pd
import pyproj
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from crimerisk.city_shares import CITY_KEY_BY_NORMALIZED_NAME, _load_enabled_city_order
from crimerisk.confidence import _load_residual_training_city_shares
from crimerisk.crime import OFFENSES_7
from crimerisk.eval.dependencies import build_outputs_dependency_paths
from crimerisk.paths import RepoPaths


OFFENSES = tuple(OFFENSES_7)
RARE_OFFENSES = frozenset(("murder", "rape"))
AGS_COLUMN = {
    "murder": "CRMCYMURD",
    "rape": "CRMCYRAPE",
    "robbery": "CRMCYROBB",
    "aggravated_assault": "CRMCYASST",
    "burglary": "CRMCYBURG",
    "larceny": "CRMCYLARC",
    "motor_vehicle_theft": "CRMCYMVEH",
}
AGS_SERVICE = (
    "https://services8.arcgis.com/TrLhDuWxkcSkFAjt/arcgis/rest/services/"
    "USA_Crime_Index_Tract/FeatureServer/7"
)
AGS_EXPECTED_ROWS = 84_112
AGS_ARM = "ags_2022a_tract"
TOP_FRACTION = 0.10
# Multinomial log score: mean log predicted share per held-out incident, with this much
# uniform mass mixed into the predicted shares.
LOG_SCORE_UNIFORM_FLOOR = 0.01
# Retained per-cell inputs carry support coordinates from schema 5 on, so post-hoc arms can
# be scored without the deleted fold surface.
SCORING_SCHEMA_VERSION = 5
TRUTH_YEAR_START = 2018
TRUTH_YEAR_END = 2025
# Truth completeness against the city reconciliation tables. A cell fails when:
#  - admitted truth differs from the reconciled final count by more than this share (and one
#    incident);
#  - a year's raw feed rows fall below RAW_COLLAPSE_RATIO of the city's largest year;
#  - an admitted city-offense's reconciled count in a year falls below OFFENSE_COLLAPSE_RATIO
#    of its largest year, when that largest year has at least OFFENSE_COLLAPSE_MIN_REFERENCE;
#  - the reconciled count falls below PUBLISHED_COLLAPSE_RATIO of a non-blocked published
#    count of at least PUBLISHED_COLLAPSE_MIN_COUNT.
TRUTH_RECONCILIATION_TOLERANCE = 0.01
RAW_COLLAPSE_RATIO = 0.10
OFFENSE_COLLAPSE_RATIO = 0.05
OFFENSE_COLLAPSE_MIN_REFERENCE = 100.0
PUBLISHED_COLLAPSE_RATIO = 0.25
PUBLISHED_COLLAPSE_MIN_COUNT = 50.0
# Build-outputs arguments the harness sets per fold; a variant may not override them.
HARNESS_BUILD_ARGUMENTS = frozenset(
    (
        "--year",
        "--bg-ags-core-out",
        "--tract-ags-core-out",
        "--build-manifest-out",
        "--feed-inputs-dir",
        "--exclude-feed-city",
        "--feed-year-end",
        "--candidate-run",
        # Each built feed fold reads its own held-out-scoped stage-10b prior.
        "--bg-prior-path",
    )
)
# A variant may name its own finished-share blend table; the harness still hands every feed fold a
# fold-scoped copy of it (`GoldEvaluation.fold_blend_table`).
FINISHED_SHARE_BLEND_FLAG = "--finished-share-blend-path"
PRODUCTION_BLEND_CONFIG = "configs/finished_share_blend_v1.csv"
BLEND_IDENTITY = (1.0, 0.0, 0.0)
# The frozen E4 model-lane panel: its source files, their digests and the identity counts the
# scored universe must reproduce.
DEFAULT_E4_PANEL = "configs/e4_panel_v1.json"
# What a resumed run must find unchanged, by content: the code and configuration a fold build
# runs, and the numerical state it reads. Top-level data files only; stamps and summaries are
# rewritten by ordinary freshness checks and are not inputs.
DEPENDENCY_CODE_PATTERNS = (
    "main.py",
    "pyproject.toml",
    "uv.lock",
    "src/crimerisk/**/*.py",
    "scripts/eval/*.py",
    "configs/**/*",
)
DEPENDENCY_STATE_DIRS = ("modeling", "controls", "observations", "reference")
# The block-group centroids the transport metrics are scored on, under data/.
BG_CENTROIDS_RELPATH = Path("tiger_bg") / "parsed" / "bg_centroids.parquet"
DEPENDENCY_STATE_SUFFIXES = (".parquet", ".csv")
FOLD_LABELS = {
    "spatial": "held-out city",
    "temporal_2021": "held-out years after 2021",
    "temporal_2023": "held-out years after 2023",
    "model": "tuning corpus for ours; untouched for AGS",
}
PRIMARY_ARMS = ("ours", "population", "primary_exposure", "past_counts")
TRACT_ARMS = ("ours_tract", "ags_2022a_tract")
PAIRED_METRICS = ("tvd", "log_score", "top_decile_capture", "w1_spatial_skill", "spearman")
BASELINE_ARM_PREFIX = "baseline_"
AGS_CITY_KEYS = frozenset(
    (
        "austin",
        "baltimore",
        "boston",
        "chicago",
        "mesa",
        "new_york",
        "philadelphia",
        "seattle",
    )
)
CALIBRATION_LEVELS = (50, 80, 95)
REUSE_NAMES = (
    "mixture_weights_v3",
    "soft_shrinkage",
    "rape_triple",
    "exposure_ensemble_weights",
    "murder_K",
    "tau",
    "residual_training_city",
)


@dataclass(frozen=True)
class GoldEvaluationConfig:
    run_id: str
    folds: tuple[str, ...] = ("spatial", "temporal", "model")
    year: int = 2025
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 20260921
    dry_run: bool = False
    # Restrict the leave-one-city-out spatial folds to these city keys. Empty means
    # every admitted feed city. Only the spatial folds are per-city, so temporal and
    # model folds are unaffected.
    cities: tuple[str, ...] = ()
    # One named candidate change per run: extra build-outputs arguments (flags, config
    # paths) appended to every fold build. The run id must end in `__<variant>`.
    variant: str = ""
    variant_args: tuple[str, ...] = ()
    # Score the model (E4) fold on this existing candidate directory instead of building it.
    model_candidate_dir: str = ""
    # A completed run whose `ours` arms are paired against this run's (candidate - baseline).
    baseline_run: str = ""
    # Score the AGS 2022A tract comparator. Off, every other arm is scored and no AGS cache or
    # download is needed.
    ags_comparator: bool = True
    # The frozen E4 panel manifest the model fold scores.
    e4_panel: str = DEFAULT_E4_PANEL


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    fold_type: str
    exclude_city_key: str | None = None
    feed_year_end: int | None = None
    candidate_dir: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _normalize(values: Iterable[float]) -> np.ndarray | None:
    raw = np.asarray(list(values), dtype=float)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    total = float(raw.sum())
    return raw / total if total > 0.0 else None


def _safe_spearman(prediction: np.ndarray, truth: np.ndarray) -> float:
    if len(prediction) < 3 or np.ptp(prediction) == 0.0 or np.ptp(truth) == 0.0:
        return float("nan")
    value = spearmanr(prediction, truth).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _local_distance_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    crs = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={float(np.mean(lat))} +lon_0={float(np.mean(lon))} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    points = np.column_stack([np.asarray(x), np.asarray(y)]) / 1000.0
    return np.ascontiguousarray(cdist(points, points), dtype=np.float64)


def _log_score(truth_count: np.ndarray, prediction: np.ndarray) -> float:
    """Mean log predicted share per held-out incident, 1% uniform mass mixed in."""
    truth = np.asarray(truth_count, dtype=float)
    truth = np.where(np.isfinite(truth) & (truth > 0.0), truth, 0.0)
    pred = _normalize(prediction)
    if pred is None or float(truth.sum()) <= 0.0:
        return float("nan")
    floored = (1.0 - LOG_SCORE_UNIFORM_FLOOR) * pred + LOG_SCORE_UNIFORM_FLOOR / len(pred)
    return float((truth * np.log(floored)).sum() / truth.sum())


def _score_vector(
    *,
    truth_count: np.ndarray,
    prediction: np.ndarray,
    support_ids: np.ndarray,
    population: np.ndarray,
    distance_matrix: np.ndarray,
    population_null_w1: float | None = None,
) -> dict[str, float]:
    truth = _normalize(truth_count)
    pred = _normalize(prediction)
    population_share = _normalize(population)
    if truth is None or population_share is None:
        raise ValueError("Truth and population vectors must each carry positive mass")
    if pred is None:
        return {
            "tvd": float("nan"),
            "spearman": float("nan"),
            "top_decile_capture": float("nan"),
            "w1_spatial_skill": float("nan"),
            "log_score": float("nan"),
        }
    order = np.lexsort((support_ids.astype(str), -pred))
    k = max(1, int(math.ceil(TOP_FRACTION * len(pred))))
    null_w1 = (
        float(population_null_w1)
        if population_null_w1 is not None
        else float(ot.emd2(population_share, truth, distance_matrix, numItermax=10_000_000))
    )
    pred_w1 = (
        null_w1
        if np.allclose(pred, population_share, rtol=0.0, atol=1e-15)
        else float(ot.emd2(pred, truth, distance_matrix, numItermax=10_000_000))
    )
    return {
        "tvd": float(0.5 * np.abs(truth - pred).sum()),
        "spearman": _safe_spearman(pred, truth),
        "top_decile_capture": float(truth[order[:k]].sum()),
        "w1_spatial_skill": (
            float(1.0 - pred_w1 / null_w1) if np.isfinite(null_w1) and null_w1 > 0.0 else float("nan")
        ),
        "log_score": _log_score(truth_count, prediction),
    }


def _population_null_w1(
    *, truth_count: np.ndarray, population: np.ndarray, distance_matrix: np.ndarray
) -> float:
    truth = _normalize(truth_count)
    population_share = _normalize(population)
    if truth is None or population_share is None:
        raise ValueError("Truth and population vectors must each carry positive mass")
    return float(
        ot.emd2(population_share, truth, distance_matrix, numItermax=10_000_000)
    )


def _share_coverage(
    truth_count: np.ndarray,
    lower_count: np.ndarray,
    upper_count: np.ndarray,
    point_count: np.ndarray,
) -> float:
    truth = _normalize(truth_count)
    point = np.asarray(point_count, dtype=float)
    point = np.where(np.isfinite(point) & (point > 0.0), point, 0.0)
    point_total = float(point.sum())
    if truth is None or point_total <= 0.0:
        return float("nan")
    lower = np.asarray(lower_count, dtype=float) / point_total
    upper = np.asarray(upper_count, dtype=float) / point_total
    low = np.minimum(lower, upper)
    high = np.maximum(lower, upper)
    present = np.isfinite(truth) & np.isfinite(low) & np.isfinite(high)
    if not bool(present.any()):
        return float("nan")
    return float(((truth[present] >= low[present]) & (truth[present] <= high[present])).mean())


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    present = ~np.isnan(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(present.any()):
        return float("nan")
    ordered = np.argsort(values[present], kind="mergesort")
    selected_values = values[present][ordered]
    selected_weights = weights[present][ordered]
    cumulative = np.cumsum(selected_weights) / selected_weights.sum()
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(selected_values[min(index, len(selected_values) - 1)])


def _log_width_requirements(
    truth_count: np.ndarray,
    lower_count: np.ndarray,
    upper_count: np.ndarray,
    point_count: np.ndarray,
) -> np.ndarray:
    """Return the factor each cell needs after scaling interval half-widths in log space.

    Shares use an additive half-incident continuity correction within the evaluated city/offense.
    The lower and upper log half-widths remain asymmetric.
    """
    truth_raw = np.asarray(truth_count, dtype=float)
    point_raw = np.asarray(point_count, dtype=float)
    truth_total = float(np.where(np.isfinite(truth_raw), np.maximum(truth_raw, 0.0), 0.0).sum())
    point_total = float(np.where(np.isfinite(point_raw), np.maximum(point_raw, 0.0), 0.0).sum())
    output = np.full(len(truth_raw), np.nan, dtype=float)
    if truth_total <= 0.0 or point_total <= 0.0:
        return output
    epsilon = 0.5 / truth_total
    truth = np.maximum(truth_raw, 0.0) / truth_total
    point = np.maximum(point_raw, 0.0) / point_total
    lower = np.maximum(np.asarray(lower_count, dtype=float) / point_total, 0.0)
    upper = np.maximum(np.asarray(upper_count, dtype=float) / point_total, 0.0)
    low = np.minimum(lower, upper)
    high = np.maximum(lower, upper)
    log_truth = np.log(truth + epsilon)
    log_point = np.log(point + epsilon)
    lower_width = np.maximum(log_point - np.log(low + epsilon), 0.0)
    upper_width = np.maximum(np.log(high + epsilon) - log_point, 0.0)
    below = log_truth < log_point
    distance = np.abs(log_truth - log_point)
    width = np.where(below, lower_width, upper_width)
    exact = distance <= 1e-15
    usable = np.isfinite(distance) & np.isfinite(width)
    output[usable & exact] = 0.0
    positive_width = usable & ~exact & (width > 0.0)
    output[positive_width] = distance[positive_width] / width[positive_width]
    output[usable & ~exact & (width <= 0.0)] = np.inf
    return output


def _aggregate_bg_counts_to_tracts(
    frame: pd.DataFrame, *, offense: str, tract_ids: Iterable[str]
) -> pd.Series:
    universe = pd.Index([str(value) for value in tract_ids], dtype="string")
    selected = frame.loc[frame["tract_id"].astype("string").isin(universe)].copy()
    selected["tract_id"] = selected["tract_id"].astype("string")
    selected["_prediction"] = pd.to_numeric(
        selected[f"expected_count_{offense}"], errors="coerce"
    ).fillna(0.0)
    counts = selected.groupby("tract_id")["_prediction"].sum()
    return counts.reindex(universe, fill_value=0.0)


def _reuse_flags(
    fold_type: str,
    offense: str,
    *,
    murder_k_seen: bool = False,
    residual_training_seen: bool = False,
) -> str:
    model = fold_type == "model"
    values = {
        "mixture_weights_v3": "seen" if model else "not_seen",
        "soft_shrinkage": "seen" if model else "not_seen",
        "rape_triple": "seen" if model and offense == "rape" else "not_seen",
        "exposure_ensemble_weights": "seen" if model else "not_seen",
        "murder_K": "seen" if offense == "murder" and murder_k_seen else "not_seen",
        "tau": "seen" if offense == "burglary" and not model else "not_seen",
        # The model fold's jurisdiction is in the surface the residual layer trained on.
        # Feed folds exclude the held-out city (spatial) or the held-out years (temporal).
        "residual_training_city": "seen" if model and residual_training_seen else "not_seen",
    }
    return ";".join(f"{name}={values[name]}" for name in REUSE_NAMES)


def _active_truth(
    paths: RepoPaths, *, truth_path: Path | None = None
) -> tuple[pd.DataFrame, list[str]]:
    path = truth_path or (paths.state_dir / "modeling" / "city_incident_share_surface.parquet")
    truth = pd.read_parquet(path).copy()
    truth["block_group_geoid"] = truth["block_group_geoid"].astype("string").str.zfill(12)
    truth["year"] = pd.to_numeric(truth["year"], errors="raise").astype(int)
    truth["incident_count"] = pd.to_numeric(truth["incident_count"], errors="coerce").fillna(0.0)
    truth = truth[truth["year"].between(TRUTH_YEAR_START, TRUTH_YEAR_END)].copy()
    if "city_key" not in truth.columns:
        truth["city_key"] = (
            truth["city_name"].astype("string").str.strip().str.lower().map(CITY_KEY_BY_NORMALIZED_NAME)
        )
    city_keys = _load_enabled_city_order(paths)
    truth = truth[truth["city_key"].isin(city_keys)].copy()
    observed = set(truth["city_key"].dropna().astype(str))
    missing = sorted(set(city_keys) - observed)
    if missing:
        raise ValueError(f"Active feed cities missing from live truth surface: {missing}")
    return truth, city_keys


def truth_completeness(truth: pd.DataFrame, reconciliation: pd.DataFrame) -> pd.DataFrame:
    """One row per city, year and offense of the truth window, checked against reconciliation.

    `admitted` says the truth carries the city-offense in some year of the window (a feed can be
    admitted for some offenses or years only). Every reconciled year of an admitted
    city-offense is checked, so a year whose feed collapsed to near nothing fails even when the
    truth holds no rows for it.
    """
    recon = reconciliation.copy()
    recon["year"] = pd.to_numeric(recon["year"], errors="raise").astype(int)
    recon = recon[recon["year"].between(TRUTH_YEAR_START, TRUTH_YEAR_END)].copy()
    for column in ("final_share_count", "raw_year_total", "published_count"):
        if column not in recon.columns:
            recon[column] = np.nan
        recon[column] = pd.to_numeric(recon[column], errors="coerce")
    if "published_comparison_quality" not in recon.columns:
        recon["published_comparison_quality"] = pd.NA
    keys = ["city_key", "year", "offense"]
    observed = (
        truth.assign(year=pd.to_numeric(truth["year"], errors="raise").astype(int))
        .groupby(keys)["incident_count"]
        .sum()
        .rename("truth_count")
        .reset_index()
    )
    table = recon[
        [*keys, "final_share_count", "raw_year_total", "published_count", "published_comparison_quality"]
    ].merge(observed, on=keys, how="outer")
    admitted_pairs = set(zip(observed["city_key"], observed["offense"]))
    table["admitted"] = [
        (city, offense) in admitted_pairs for city, offense in zip(table["city_key"], table["offense"])
    ]
    table["truth_present"] = table["truth_count"].notna()
    final = table["final_share_count"].fillna(0.0)
    truth_count = table["truth_count"].fillna(0.0)
    city_raw_max = table.groupby("city_key")["raw_year_total"].transform("max")
    offense_max = table.groupby(["city_key", "offense"])["final_share_count"].transform("max")
    quality = table["published_comparison_quality"].astype("string").fillna("")
    published = table["published_count"]
    checks = [
        (
            "truth_reconciliation_mismatch",
            table["truth_present"]
            & (truth_count - final).abs().gt(np.maximum(1.0, TRUTH_RECONCILIATION_TOLERANCE * final)),
        ),
        (
            "collapse_raw_year",
            table["admitted"]
            & table["raw_year_total"].notna()
            & city_raw_max.gt(0.0)
            & table["raw_year_total"].lt(RAW_COLLAPSE_RATIO * city_raw_max),
        ),
        (
            "collapse_offense_year",
            table["admitted"]
            & table["raw_year_total"].notna()
            & offense_max.ge(OFFENSE_COLLAPSE_MIN_REFERENCE)
            & final.lt(OFFENSE_COLLAPSE_RATIO * offense_max),
        ),
        (
            "collapse_published",
            table["admitted"]
            & published.ge(PUBLISHED_COLLAPSE_MIN_COUNT)
            & ~quality.str.startswith("blocked")
            & final.lt(PUBLISHED_COLLAPSE_RATIO * published),
        ),
    ]
    status = pd.Series("ok", index=table.index, dtype=object)
    status[~table["admitted"]] = "not_admitted"
    for name, failed in reversed(checks):
        status[failed.fillna(False).astype(bool)] = name
    table["status"] = status
    return table.sort_values(keys, kind="mergesort").reset_index(drop=True)


def assert_truth_complete(table: pd.DataFrame) -> None:
    failed = table[~table["status"].isin(["ok", "not_admitted"])]
    if failed.empty:
        return
    summary = (
        failed.groupby(["city_key", "status"])["year"]
        .agg(lambda years: ",".join(str(y) for y in sorted(set(years))))
        .reset_index()
    )
    raise ValueError(
        "Truth snapshot fails the completeness check against the city reconciliation tables "
        f"({len(failed)} city-year-offense cells): "
        + "; ".join(
            f"{row.city_key} {row.status} years {row.year}" for row in summary.itertuples(index=False)
        )
    )


def ensure_ags_tract_cache(paths: RepoPaths) -> Path:
    cache_dir = paths.cache_dir / "ags_2022a"
    out = cache_dir / "ags_2022a_tract.parquet"
    if out.exists():
        frame = pd.read_parquet(out, columns=["ID"])
        if len(frame) != AGS_EXPECTED_ROWS or frame["ID"].astype(str).duplicated().any():
            raise ValueError(f"Unexpected AGS cache identity: {out}")
        return out

    fields = ["OBJECTID", "ID", "POP_C", *AGS_COLUMN.values()]
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "f": "json",
                "where": "1=1",
                "outFields": ",".join(fields),
                "returnGeometry": "false",
                "orderByFields": "OBJECTID",
                "resultOffset": offset,
                "resultRecordCount": 2000,
            }
        )
        with urllib.request.urlopen(f"{AGS_SERVICE}/query?{params}", timeout=120) as response:
            payload = json.load(response)
        if payload.get("error"):
            raise RuntimeError(f"AGS query failed: {payload['error']}")
        page = [feature["attributes"] for feature in payload.get("features", [])]
        rows.extend(page)
        if len(page) < 2000:
            break
        offset += len(page)
        time.sleep(0.1)
    frame = pd.DataFrame(rows)
    if len(frame) != AGS_EXPECTED_ROWS or frame["ID"].astype(str).duplicated().any():
        raise ValueError(f"Downloaded AGS row identity is unexpected: rows={len(frame)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    _json_write(
        cache_dir / "download_manifest.json",
        {
            "service": AGS_SERVICE,
            "layer": 7,
            "vintage": "2022A",
            "rows": int(len(frame)),
            "fields": fields,
            "downloaded_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "sha256": _sha256(out),
        },
    )
    return out


class GeometryCache:
    def __init__(self, centroids_path: Path):
        centroids = pd.read_parquet(centroids_path, columns=["bg_id", "lon", "lat"])
        centroids["support_id"] = centroids["bg_id"].astype("string").str.zfill(12)
        self.centroids = centroids.set_index("support_id")[["lon", "lat"]]
        self._matrices: dict[tuple[str, ...], np.ndarray] = {}
        self._coordinates: dict[tuple[str, ...], pd.DataFrame] = {}

    def matrix(self, support_ids: pd.Series, *, bg_population: pd.DataFrame | None = None) -> np.ndarray:
        ids = tuple(support_ids.astype(str))
        cached = self._matrices.get(ids)
        if cached is not None:
            return cached
        coordinates = self.coordinates(support_ids, bg_population=bg_population)
        matrix = _local_distance_matrix(
            coordinates["lon"].to_numpy(float), coordinates["lat"].to_numpy(float)
        )
        self._matrices[ids] = matrix
        return matrix

    def coordinates(
        self, support_ids: pd.Series, *, bg_population: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Block-group centroids, or population-weighted member centroids for tracts."""
        ids = tuple(support_ids.astype(str))
        cached = self._coordinates.get(ids)
        if cached is not None:
            return cached
        if len(ids[0]) == 12:
            coordinates = self.centroids.reindex(ids)
        else:
            if bg_population is None:
                raise ValueError("Tract coordinates require member block-group population")
            members = bg_population.copy()
            members["support_id"] = members["block_group_geoid"].astype("string").str.zfill(12)
            members = members.join(self.centroids, on="support_id")
            members["tract_id"] = members["support_id"].str[:11]
            weight = pd.to_numeric(members["population"], errors="coerce").fillna(0.0) + 1e-6
            members["lon_weighted"] = pd.to_numeric(members["lon"], errors="coerce") * weight
            members["lat_weighted"] = pd.to_numeric(members["lat"], errors="coerce") * weight
            grouped = members.assign(weight=weight).groupby("tract_id")[
                ["lon_weighted", "lat_weighted", "weight"]
            ].sum()
            coordinates = pd.DataFrame(
                {
                    "lon": grouped["lon_weighted"] / grouped["weight"],
                    "lat": grouped["lat_weighted"] / grouped["weight"],
                }
            ).reindex(ids)
        if coordinates.isna().any(axis=None):
            raise ValueError("Missing centroid coordinates for scored support")
        self._coordinates[ids] = coordinates
        return coordinates


def _ordered_arms(arms: Iterable[str]) -> list[str]:
    fixed = [
        "ours", "ours_tract", BASELINE_ARM_PREFIX + "ours", BASELINE_ARM_PREFIX + "ours_tract",
    ]
    tail = ["population", "primary_exposure", "past_counts", "ags_2022a_tract"]
    present = [str(arm) for arm in arms]
    posthoc = sorted(arm for arm in present if arm not in fixed and arm not in tail)
    return [arm for arm in fixed if arm in present] + posthoc + [arm for arm in tail if arm in present]


def _is_focus_arm(arm: str) -> bool:
    """Arms whose differences against every other arm are reported: ours and post-hoc arms."""
    if arm in ("ours", "ours_tract"):
        return True
    return (
        arm not in PRIMARY_ARMS
        and arm not in TRACT_ARMS
        and not arm.startswith(BASELINE_ARM_PREFIX)
    )


def _paired_estimate(values: np.ndarray, indices: np.ndarray | None) -> dict[str, float]:
    """City mean and percentile interval over the block's shared resample indices."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n_cities = int(finite.sum())
    if n_cities == 0:
        return {"estimate": float("nan"), "ci95_lower": float("nan"), "ci95_upper": float("nan"), "n_cities": 0}
    estimate = float(values[finite].mean())
    low = high = float("nan")
    if indices is not None and n_cities >= 2:
        sampled = np.where(np.isfinite(values[indices]), values[indices], 0.0)
        counts = np.isfinite(values[indices]).sum(axis=1)
        draws = sampled.sum(axis=1)[counts > 0] / counts[counts > 0]
        low, high = (float(value) for value in np.quantile(draws, [0.025, 0.975]))
    return {"estimate": estimate, "ci95_lower": low, "ci95_upper": high, "n_cities": n_cities}


BLEND_COMPONENTS = ("ours", "population", "primary_exposure")
# 0.1-step simplex over (ours, population, primary exposure), ours-heavy points first so that
# ties resolve toward the model.
BLEND_GRID = tuple(
    (a / 10.0, b / 10.0, (10 - a - b) / 10.0) for a in range(10, -1, -1) for b in range(0, 11 - a)
)


def _blend_grid_tvd(
    group: pd.DataFrame,
) -> tuple[dict[str, tuple[np.ndarray, list[np.ndarray]]], dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Per city of one fold type and offense: shares, frames and TVD at every blend grid point."""
    shares: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for city, frame in group.groupby("city_name", sort=True):
        truth = _normalize(frame["truth_count"])
        components = [_normalize(frame[f"prediction_{name}"]) for name in BLEND_COMPONENTS]
        if truth is None or any(component is None for component in components):
            continue
        shares[str(city)] = (truth, components)
        frames[str(city)] = frame
    tvd_by_city = {
        city: np.array(
            [
                0.5 * np.abs(truth - sum(w * c for w, c in zip(point, components))).sum()
                for point in BLEND_GRID
            ]
        )
        for city, (truth, components) in shares.items()
    }
    return shares, frames, tvd_by_city


def pooled_blend_weights(retained: pd.DataFrame) -> pd.DataFrame:
    """Per fold type and offense, the blend grid point with the lowest mean TVD over every city.

    The same simplex, objective and tie order as `cross_fitted_blend`, fitted on all held-out
    cities at once: the production weights for an adopted offense.
    """
    rows: list[dict[str, object]] = []
    for (fold_type, offense), group in retained.groupby(["fold_type", "offense"], sort=True):
        _shares, _frames, tvd_by_city = _blend_grid_tvd(group)
        if not tvd_by_city:
            continue
        objective = np.mean(list(tvd_by_city.values()), axis=0)
        best = int(np.argmin(objective))
        rows.append(
            {
                "fold_type": fold_type,
                "offense": offense,
                **{f"weight_{name}": value for name, value in zip(BLEND_COMPONENTS, BLEND_GRID[best])},
                "fit_cities": ";".join(sorted(tvd_by_city)),
                "fit_mean_tvd": float(objective[best]),
            }
        )
    return pd.DataFrame(rows)


def cross_fitted_blend(retained: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per fold type, offense and held-out city, blend weights fitted on the other cities.

    The weights for city c minimise the mean TVD over the other held-out cities of the same
    fold type and offense (their predictions are themselves held-out predictions); city c is
    then scored with them. A city with no other city to fit on gets no blend.
    """
    predictions: list[pd.DataFrame] = []
    weights: list[dict[str, object]] = []
    for (fold_type, offense), group in retained.groupby(["fold_type", "offense"], sort=True):
        shares, frames, tvd_by_city = _blend_grid_tvd(group)
        for city, (_truth, components) in shares.items():
            others = [other for other in shares if other != city]
            if not others:
                continue
            objective = np.mean([tvd_by_city[other] for other in others], axis=0)
            best = int(np.argmin(objective))
            point = BLEND_GRID[best]
            blended = sum(w * c for w, c in zip(point, components))
            predictions.append(frames[city].assign(prediction_blend_xfit=blended))
            weights.append(
                {
                    "fold_type": fold_type,
                    "offense": offense,
                    "city_name": city,
                    **{f"weight_{name}": value for name, value in zip(BLEND_COMPONENTS, point)},
                    "training_cities": ";".join(others),
                    "training_mean_tvd": float(objective[best]),
                }
            )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(weights),
    )


POSTHOC_ARMS = {"blend_xfit": cross_fitted_blend}


def _score_posthoc_arm(predictions: pd.DataFrame, *, arm: str, run_cells: pd.DataFrame) -> pd.DataFrame:
    """Score a post-hoc arm's retained predictions exactly as the fold scored `ours`."""
    if predictions.empty:
        return pd.DataFrame(columns=run_cells.columns)
    # Post-hoc arms score the feed folds' primary cells. The model fold has several
    # jurisdictions per source, and a non-unique index would make `.loc` return frames.
    identity = run_cells[
        run_cells["arm"].eq("ours")
        & ~run_cells["fold_type"].eq("model")
        & run_cells["cell_scope"].eq("primary")
    ].set_index(["fold_id", "city_name", "offense"])
    if not identity.index.is_unique:
        raise ValueError("Run cells repeat a feed-fold (fold_id, city_name, offense) key")
    rows: list[dict[str, object]] = []
    for (fold_id, city, offense), frame in predictions.groupby(
        ["fold_id", "city_name", "offense"], sort=True
    ):
        truth = frame["truth_count"].to_numpy(float)
        population = frame["population"].to_numpy(float)
        distance = _local_distance_matrix(
            frame["support_lon"].to_numpy(float), frame["support_lat"].to_numpy(float)
        )
        metrics = _score_vector(
            truth_count=truth,
            prediction=frame[f"prediction_{arm}"].to_numpy(float),
            support_ids=frame["support_id"].astype(str).to_numpy(),
            population=population,
            distance_matrix=distance,
            population_null_w1=_population_null_w1(
                truth_count=truth, population=population, distance_matrix=distance
            ),
        )
        source = identity.loc[(fold_id, city, offense)]
        rows.append(
            {
                "fold_id": fold_id,
                "fold_type": source["fold_type"],
                "city_name": city,
                "jurisdiction_id": source["jurisdiction_id"],
                "ags_uses_city_feed": source["ags_uses_city_feed"],
                "offense": offense,
                "arm": arm,
                "support": source["support"],
                "n_cells": int(len(frame)),
                "n_incidents": float(truth.sum()),
                "cell_scope": "primary",
                "reuse_flags": source["reuse_flags"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


RESCORE_GROUP_KEYS = [
    "fold_id", "fold_type", "city_name", "jurisdiction_id", "ags_uses_city_feed", "offense",
    "support_role", "support", "cell_scope",
]
TRUTH_KEY_COLUMNS = [*RESCORE_GROUP_KEYS, "support_id"]


# Every truth-side value `rescore_on_truth` reads: the count, and the population and
# coordinates its population null and transport distances are computed from.
TRUTH_SCORING_COLUMNS = ["truth_count", "population", "support_lon", "support_lat"]


def processed_truth_table(retained: pd.DataFrame) -> pd.DataFrame:
    """The keyed processed truth a run scored: one row per (group, support_id)."""
    table = retained[[*TRUTH_KEY_COLUMNS, *TRUTH_SCORING_COLUMNS]].copy()
    table["support_id"] = table["support_id"].astype(str)
    if table.duplicated(TRUTH_KEY_COLUMNS).any():
        raise ValueError("retained scoring inputs repeat a (group, support_id) key")
    return table.sort_values(TRUTH_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def processed_truth_sha256(table: pd.DataFrame) -> str:
    """sha256 of the keyed truth and every scoring input read from it.

    Runs completed before population and coordinates were included recorded a hash of the keys
    and `truth_count` only; their recorded value is not comparable with this one.
    """
    import hashlib

    keyed = table[[*TRUTH_KEY_COLUMNS, *TRUTH_SCORING_COLUMNS]].astype(str)
    return hashlib.sha256("\n".join("|".join(row) for row in keyed.itertuples(index=False)).encode()).hexdigest()


def rescore_on_truth(
    truth: pd.DataFrame, predictions: pd.DataFrame, *, arm_column: str, arm_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score `predictions[arm_column]` against one frozen truth table, group by group.

    A group is scored only when the prediction table carries exactly the truth table's support
    IDs for it; any other group is returned as an exclusion, never paired on a different
    support. Either table repeating a (group, support_id) key raises: a set comparison cannot
    see a repeat, and the merge would score more predictions than truth cells. Returns
    (cells, exclusions).
    """
    pred = predictions[[*TRUTH_KEY_COLUMNS, arm_column]].copy()
    pred["support_id"] = pred["support_id"].astype(str)
    truth_ids = truth[TRUTH_KEY_COLUMNS].astype({"support_id": str})
    for name, keys in (("truth", truth_ids), ("prediction", pred[TRUTH_KEY_COLUMNS])):
        repeated = keys.duplicated(TRUTH_KEY_COLUMNS, keep=False)
        if repeated.any():
            raise ValueError(
                f"{name} table repeats {int(keys[repeated].drop_duplicates().shape[0])} "
                "(group, support_id) key(s); re-scoring needs one row per key"
            )
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    pred_groups = {key: frame for key, frame in pred.groupby(RESCORE_GROUP_KEYS, sort=False, dropna=False)}
    for key, frame in truth.groupby(RESCORE_GROUP_KEYS, sort=True, dropna=False):
        other = pred_groups.get(key)
        record = dict(zip(RESCORE_GROUP_KEYS, key))
        if other is None:
            exclusions.append({**record, "arm": arm_name, "reason": "group_absent"})
            continue
        if set(other["support_id"]) != set(frame["support_id"]):
            exclusions.append({**record, "arm": arm_name, "reason": "support_ids_differ"})
            continue
        values = frame[["support_id"]].merge(
            other[["support_id", arm_column]], on="support_id", how="left", validate="one_to_one"
        )
        prediction = pd.to_numeric(values[arm_column], errors="coerce").to_numpy(float)
        if not np.isfinite(prediction).all():
            exclusions.append({**record, "arm": arm_name, "reason": "prediction_missing"})
            continue
        truth_count = frame["truth_count"].to_numpy(float)
        population = frame["population"].to_numpy(float)
        distance = _local_distance_matrix(frame["support_lon"].to_numpy(float), frame["support_lat"].to_numpy(float))
        metrics = _score_vector(
            truth_count=truth_count,
            prediction=prediction,
            support_ids=frame["support_id"].to_numpy(str),
            population=population,
            distance_matrix=distance,
            population_null_w1=_population_null_w1(
                truth_count=truth_count, population=population, distance_matrix=distance
            ),
        )
        rows.append(
            {
                **{k: record[k] for k in ("fold_id", "fold_type", "city_name", "jurisdiction_id", "ags_uses_city_feed", "offense")},
                "arm": arm_name,
                "support": record["support"],
                "n_cells": int(len(frame)),
                "n_incidents": float(truth_count.sum()),
                "cell_scope": record["cell_scope"],
                **metrics,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def _validate_variant(config: GoldEvaluationConfig) -> None:
    if config.variant_args and not config.variant:
        raise ValueError("Variant build arguments need a variant name")
    if not config.variant:
        return
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", config.variant):
        raise ValueError(f"Variant name {config.variant!r} must be lower-case [a-z0-9_.-]")
    if not config.run_id.endswith(f"__{config.variant}"):
        raise ValueError(
            f"Run id {config.run_id!r} must end in '__{config.variant}' for variant {config.variant!r}"
        )
    reserved = sorted(
        {value.split("=", 1)[0] for value in config.variant_args} & HARNESS_BUILD_ARGUMENTS
    )
    if reserved:
        raise ValueError(f"Variant arguments may not set harness-controlled flags: {reserved}")


def _e4_spill_keys(
    widened_truth: pd.DataFrame, *, multi_jurisdiction_sources: Iterable[str]
) -> set[tuple[str, str]]:
    """(source, jurisdiction) pairs where a one-department E4 source spills into a neighbour.

    `multi_jurisdiction_sources` (from the panel manifest) legitimately span several
    jurisdictions, e.g. a sheriff with contract cities. Every other source is one police
    department whose own jurisdiction is the one holding most of its incidents.
    """
    multi = {str(value) for value in multi_jurisdiction_sources}
    totals = widened_truth.groupby(["source", "jurisdiction_id"])["incident_count"].sum()
    home = totals.groupby(level="source").idxmax().map(lambda key: str(key[1]))
    return {
        (str(source), str(jurisdiction_id))
        for source, jurisdiction_id in totals.index
        if str(source) not in multi and str(jurisdiction_id) != home[source]
    }


def load_e4_panel(repo_root: Path, manifest_path: str | Path) -> dict[str, object]:
    """The frozen E4 panel manifest, with its source files resolved and their digests checked."""
    path = Path(manifest_path)
    if not path.is_absolute():
        path = repo_root / path
    panel = json.loads(path.read_text())
    for name in ("truth", "registry", "frozen_cells"):
        entry = dict(panel[name])
        source = Path(str(entry["path"]))
        if not source.is_absolute():
            source = repo_root / source
        actual = _sha256(source)
        if actual != entry["sha256"]:
            raise ValueError(
                f"E4 panel {panel['panel_id']!r}: {source} sha256 {actual} does not match the "
                f"manifest's {entry['sha256']}"
            )
        entry["resolved_path"] = source
        panel[name] = entry
    panel["manifest_path"] = str(path)
    panel["manifest_sha256"] = _sha256(path)
    return panel


def _flag_value(args: Iterable[str], flag: str) -> str | None:
    """The value a build-argument list gives `flag` (`--flag value` or `--flag=value`)."""
    values = list(args)
    for index, value in enumerate(values):
        if value == flag and index + 1 < len(values):
            return values[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def _without_flag(args: Iterable[str], flag: str) -> list[str]:
    values = list(args)
    out: list[str] = []
    skip = False
    for value in values:
        if skip:
            skip = False
            continue
        if value == flag:
            skip = True
            continue
        if not value.startswith(flag + "="):
            out.append(value)
    return out


def _blend_point(row: pd.Series) -> tuple[float, float, float]:
    return (
        float(row["weight_model"]),
        float(row["weight_population"]),
        float(row["weight_primary_exposure"]),
    )


def truth_through_year(
    retained: pd.DataFrame, truth: pd.DataFrame, *, year_end: int
) -> pd.DataFrame:
    """Retained primary inputs with `truth_count` recounted from truth years <= `year_end`.

    Supports keep their identity; a tract support sums its block groups.
    """
    selected = truth[pd.to_numeric(truth["year"], errors="raise").le(int(year_end))].copy()
    selected["_geoid"] = selected["block_group_geoid"].astype("string").str.zfill(12)
    parts: list[pd.DataFrame] = []
    for (city, offense, support), frame in retained.groupby(
        ["city_name", "offense", "support"], sort=True
    ):
        rows = selected[selected["city_name"].eq(city) & selected["offense"].eq(offense)]
        key = rows["_geoid"].str[:11] if support == "tract" else rows["_geoid"]
        counts = rows.assign(_support=key).groupby("_support")["incident_count"].sum()
        parts.append(
            frame.assign(
                truth_count=frame["support_id"].astype(str).map(counts).fillna(0.0).to_numpy(float)
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else retained.iloc[0:0].copy()


def _cell_scope(cells: pd.DataFrame) -> pd.Series:
    if "cell_scope" not in cells.columns:
        return pd.Series("primary", index=cells.index, dtype=object)
    return cells["cell_scope"].fillna("primary").astype(str)


class GoldEvaluation:
    def __init__(self, *, paths: RepoPaths, config: GoldEvaluationConfig):
        self.paths = paths
        self.config = config
        _validate_variant(config)
        self.run_dir = paths.state_dir / "eval" / config.run_id
        self.work_dir = self.run_dir / "work"
        self.scored_dir = self.run_dir / "scored"
        self.inputs_dir = self.run_dir / "inputs"
        self.truth_snapshot_path = self.inputs_dir / (
            f"city_incident_share_surface_{TRUTH_YEAR_START}_{TRUTH_YEAR_END}.parquet"
        )
        self.truth, self.city_keys = _active_truth(
            paths,
            truth_path=self.truth_snapshot_path if self.truth_snapshot_path.exists() else None,
        )
        city_identity = self.truth[["city_key", "jurisdiction_id"]].drop_duplicates()
        self.ags_feed_jurisdictions = set(
            city_identity.loc[
                city_identity["city_key"].isin(AGS_CITY_KEYS), "jurisdiction_id"
            ].astype(str)
        )
        self._fold_input_parts: list[pd.DataFrame] = []
        self.geometry = GeometryCache(paths.data_dir / BG_CENTROIDS_RELPATH)
        murder_selection_path = (
            paths.state_dir / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet"
        )
        self.murder_k_jurisdictions: set[str] = set()
        self._e4_panel: dict[str, object] | None = None
        self._blend_sources: dict[tuple[str, str], object] = {}
        if murder_selection_path.exists():
            selected = pd.read_parquet(
                murder_selection_path,
                columns=["jurisdiction_id", "offense", "incident_count"],
            )
            selected = selected[selected["offense"].eq("murder")]
            totals = selected.groupby("jurisdiction_id")["incident_count"].sum()
            self.murder_k_jurisdictions = set(totals[totals.gt(0.0)].index.astype(str))
            registry = pd.read_csv(self.e4_panel()["registry"]["resolved_path"])
            e4_murder = set(
                registry.loc[registry["offense"].eq("murder"), "jurisdiction_id"].astype(str)
            )
            self.murder_k_jurisdictions -= e4_murder

    def fold_specs(self) -> list[FoldSpec]:
        totals = self.truth.groupby("city_key")["incident_count"].sum().sort_values(ascending=False)
        ordered = [str(value) for value in totals.index]
        if self.config.cities:
            requested = tuple(dict.fromkeys(self.config.cities))
            unknown = sorted(set(requested) - set(ordered))
            if unknown:
                raise ValueError(
                    "Requested spatial fold cities are not admitted feed cities: "
                    f"{', '.join(unknown)}; admitted: {', '.join(ordered)}"
                )
            ordered = [key for key in ordered if key in set(requested)]
        specs: list[FoldSpec] = []
        if "spatial" in self.config.folds:
            specs.extend(
                FoldSpec(
                    f"spatial_{key}",
                    "spatial",
                    exclude_city_key=key,
                    feed_year_end=self.config.year,
                )
                for key in ordered
            )
        if "temporal" in self.config.folds:
            specs.extend(
                [
                    FoldSpec("temporal_through_2021", "temporal_2021", feed_year_end=2021),
                    FoldSpec("temporal_through_2023", "temporal_2023", feed_year_end=2023),
                ]
            )
        if "model" in self.config.folds:
            if self.config.model_candidate_dir:
                specs.append(
                    FoldSpec(
                        "model_scoring_only",
                        "model",
                        candidate_dir=str(Path(self.config.model_candidate_dir).resolve()),
                    )
                )
            else:
                specs.append(FoldSpec("model_fold0", "model", feed_year_end=self.config.year))
        return specs

    def fold_inputs_dir(self, spec: FoldSpec) -> Path:
        """Where this fold's own city incident feed artifacts live.

        Every fold truncates the feed year, drops a feed city, or both, so the share surface and
        reconciliation tables it builds describe a deliberately incomplete world. Written to the
        production location they would replace the released surface with a held-out one, which is
        how an evaluation run left `state/modeling/city_incident_share_surface.parquet` at
        year_end=2023. Each fold gets its own directory under the run instead.
        """
        return self.inputs_dir / spec.fold_id

    def e4_panel(self) -> dict[str, object]:
        if self._e4_panel is None:
            self._e4_panel = load_e4_panel(self.paths.repo_root, self.config.e4_panel)
        return self._e4_panel

    # --- fold-scoped stage-10b prior -------------------------------------------------------

    def fold_prior_path(self, spec: FoldSpec) -> Path:
        return self.work_dir / spec.fold_id / f"bg_prior_long_{self.config.year}_arm_b.parquet"

    def prior_command(self, spec: FoldSpec) -> list[str] | None:
        """The stage-10b prior build a held-out fold needs, or None.

        The prior's burglary premises denominator is calibrated on covered-city burglary
        incidents. The production prior sees every city and year, so a fold that holds out a city
        or years builds its own prior without them and stage 12 reads that one.
        """
        if spec.candidate_dir:
            return None
        holds_out_years = spec.feed_year_end is not None and spec.feed_year_end < self.config.year
        if spec.exclude_city_key is None and not holds_out_years:
            return None
        command = [
            "uv", "run", "python", "main.py", "build-model-prior",
            "--year", str(self.config.year),
            "--out", str(self.fold_prior_path(spec)),
        ]
        if spec.exclude_city_key is not None:
            command.extend(["--exclude-feed-city", spec.exclude_city_key])
        if spec.feed_year_end is not None:
            command.extend(["--feed-year-end", str(spec.feed_year_end)])
        return command

    # --- fold-scoped finished-share blend --------------------------------------------------

    def blend_base(self) -> tuple[Path, pd.DataFrame]:
        """The blend table the candidate ships: the variant's, else the production config."""
        value = _flag_value(self.config.variant_args, FINISHED_SHARE_BLEND_FLAG)
        path = Path(value or PRODUCTION_BLEND_CONFIG)
        if not path.is_absolute():
            path = self.paths.repo_root / path
        return path, pd.read_csv(path)

    def fold_blend_path(self, spec: FoldSpec) -> Path | None:
        """Where a feed fold's own blend table goes, or None when the fold needs none.

        Only feed folds are scored on feed-city truth, and only a non-identity (adopted) blend
        row was fitted on it. The model fold scores E4 jurisdictions the fit never saw.
        """
        if spec.candidate_dir or spec.fold_type == "model":
            return None
        if _flag_value(self.config.variant_args, FINISHED_SHARE_BLEND_FLAG) is None and not (
            self.paths.repo_root / PRODUCTION_BLEND_CONFIG
        ).is_file():
            return None
        _path, base = self.blend_base()
        if all(np.allclose(_blend_point(row), BLEND_IDENTITY) for _, row in base.iterrows()):
            return None
        return self.run_dir / "blend" / f"{spec.fold_id}.csv"

    def _blend_source_xfit(self, source_run: str) -> pd.DataFrame:
        key = ("xfit", source_run)
        if key not in self._blend_sources:
            out_dir = self.paths.state_dir / "eval" / source_run / "posthoc" / "blend_xfit"
            manifest = json.loads((out_dir / "manifest.json").read_text())
            weights_path = out_dir / "weights.csv"
            if _sha256(weights_path) != manifest.get("weights_sha256"):
                raise RuntimeError(f"{weights_path} changed since its post-hoc run")
            self._blend_sources[key] = pd.read_csv(weights_path)
        return self._blend_sources[key]  # type: ignore[return-value]

    def _blend_source_retained(
        self, source_run: str, source_fold_type: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = ("retained", f"{source_run}|{source_fold_type}")
        if key not in self._blend_sources:
            manifest = json.loads(
                (self.paths.state_dir / "eval" / source_run / "run_manifest.json").read_text()
            )
            if "completed_at_utc" not in manifest:
                raise RuntimeError(f"Blend source run {source_run!r} is not complete")
            parts = []
            for record in manifest["folds"]:
                if record["fold_type"] != source_fold_type:
                    continue
                path = Path(str(record["input_path"]))
                if _sha256(path) != record["input_sha256"]:
                    raise RuntimeError(f"Retained input {path} changed since run {source_run!r}")
                parts.append(pd.read_parquet(path))
            retained = pd.concat(parts, ignore_index=True)
            retained = retained[retained["support_role"].eq("primary")]
            pinned = manifest["inputs"]["city_truth"]
            truth_path = Path(str(pinned["path"]))
            if _sha256(truth_path) != pinned["sha256"]:
                raise RuntimeError(f"Truth snapshot {truth_path} changed since run {source_run!r}")
            self._blend_sources[key] = (retained, pd.read_parquet(truth_path))
        return self._blend_sources[key]  # type: ignore[return-value]

    def fold_blend_table(self, spec: FoldSpec) -> pd.DataFrame:
        """The blend weights this fold's build may use: none fitted on what the fold scores.

        Per adopted offense (a row other than (1, 0, 0)), from the run named in the row's
        `source_run_id`, whose `ours` arm the weights were fitted on:

        - spatial fold holding out city c: that run's cross-fitted weights for c
          (`posthoc/blend_xfit/weights.csv`, fitted on every other city). A city with no
          cross-fitted row is not among the production fit cities and keeps the production row.
        - temporal fold with cutoff E: the production fit (same grid, objective and held-out
          predictions) refitted on truth years <= E only. The cross-fitted rows cannot serve: every
          one of them scored truth after E.

        Identity rows stay identity.
        """
        _path, base = self.blend_base()
        table = base.copy()
        table["fold_id"] = spec.fold_id
        table["fold_weight_source"] = "identity"
        city_names = (
            self.truth[["city_key", "city_name"]].drop_duplicates("city_key").set_index("city_key")
        )["city_name"]
        for index, row in base.iterrows():
            if np.allclose(_blend_point(row), BLEND_IDENTITY):
                continue
            offense = str(row["offense"])
            source_run = str(row["source_run_id"])
            source_fold_type = str(row.get("source_fold_type") or "spatial")
            if spec.fold_type == "spatial":
                city = str(city_names[spec.exclude_city_key])
                xfit = self._blend_source_xfit(source_run)
                match = xfit[
                    xfit["fold_type"].astype(str).eq(source_fold_type)
                    & xfit["offense"].astype(str).eq(offense)
                    & xfit["city_name"].astype(str).eq(city)
                ]
                if len(match) > 1:
                    raise ValueError(f"{source_run} cross-fitted weights repeat {offense}/{city}")
                if len(match) == 1:
                    point = (
                        float(match["weight_ours"].iloc[0]),
                        float(match["weight_population"].iloc[0]),
                        float(match["weight_primary_exposure"].iloc[0]),
                    )
                    source = f"cross_fitted_without:{city}:{source_run}"
                else:
                    fit_cities = set(str(row.get("fit_cities") or "").split(";")) - {"", "nan"}
                    if city in fit_cities:
                        raise ValueError(
                            f"{offense} blend was fitted on {city} but {source_run} has no "
                            f"cross-fitted weights without it"
                        )
                    point = _blend_point(row)
                    source = f"production_fit_without:{city}"
            elif spec.fold_type.startswith("temporal"):
                year_end = int(spec.feed_year_end)  # type: ignore[arg-type]
                retained, truth = self._blend_source_retained(source_run, source_fold_type)
                refit = pooled_blend_weights(
                    truth_through_year(retained, truth, year_end=year_end)
                ).set_index("offense")
                if offense not in refit.index:
                    raise ValueError(f"No {offense} blend refit on {source_run} truth <= {year_end}")
                point = (
                    float(refit.loc[offense, "weight_ours"]),
                    float(refit.loc[offense, "weight_population"]),
                    float(refit.loc[offense, "weight_primary_exposure"]),
                )
                source = f"refit_truth_through:{year_end}:{source_run}"
            else:
                raise ValueError(f"Fold {spec.fold_id} takes no fold-scoped blend")
            table.loc[index, ["weight_model", "weight_population", "weight_primary_exposure"]] = point
            table.loc[index, "fold_weight_source"] = source
        return table

    def _write_fold_blend(self, spec: FoldSpec) -> dict[str, object] | None:
        path = self.fold_blend_path(spec)
        if path is None:
            return None
        table = self.fold_blend_table(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "base_path": str(self.blend_base()[0]),
            "weights_model_population_primary_exposure": {
                str(row["offense"]): list(_blend_point(row)) for _, row in table.iterrows()
            },
            "sources": {str(row["offense"]): str(row["fold_weight_source"]) for _, row in table.iterrows()},
        }

    # --- resume binding -------------------------------------------------------------------

    def dependency_fingerprint(self) -> dict[str, str]:
        """sha256 of every code, configuration and numerical-state file a fold build reads.

        Code: main.py, the package, scripts/eval/, the lock file and configs/. Numerical state:
        the top-level data files of state/{modeling,controls,observations,reference}. Raw and
        derived inputs: the declared dependency closure of every stage a fold's build-outputs
        runs or refreshes (`crimerisk.eval.dependencies.build_outputs_dependency_paths`: the raw
        sources under data/, geometry, crosswalks, reference tables, city feed files), and the
        block-group centroids the scorer reads. Plus any file a variant argument names, the blend
        source files, and the published block-group and tract surfaces of a scoring-only
        candidate. The release validator requires the code, configuration, build-outputs closure
        and candidate surfaces here (`validate_release_outputs._gold_required_dependencies`).
        """
        root = self.paths.repo_root
        files: set[Path] = set(
            build_outputs_dependency_paths(self.paths, year=int(self.config.year))
        )
        centroids = self.paths.data_dir / BG_CENTROIDS_RELPATH
        if centroids.is_file():
            files.add(centroids)
        for pattern in DEPENDENCY_CODE_PATTERNS:
            files.update(
                path
                for path in root.glob(pattern)
                if path.is_file()
                and "__pycache__" not in path.parts
                and not path.name.startswith(".")
            )
        for name in DEPENDENCY_STATE_DIRS:
            directory = self.paths.state_dir / name
            if directory.is_dir():
                files.update(
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix in DEPENDENCY_STATE_SUFFIXES
                )
        for spec in self.fold_specs():
            if spec.candidate_dir:
                for geography in ("block_group", "tract"):
                    files.add(
                        Path(spec.candidate_dir)
                        / f"crimerisk_{geography}_{self.config.year}_ags_core.parquet"
                    )
        for value in self.config.variant_args:
            candidate = Path(value) if Path(value).is_absolute() else root / value
            if not value.startswith("-") and candidate.is_file():
                files.add(candidate)
        if any(
            self.fold_blend_path(spec) is not None for spec in self.fold_specs()
        ):
            base_path, base = self.blend_base()
            files.add(base_path)
            for source_run in sorted(set(base["source_run_id"].dropna().astype(str))):
                source_dir = self.paths.state_dir / "eval" / source_run
                for path in (
                    source_dir / "run_manifest.json",
                    source_dir / "posthoc" / "blend_xfit" / "weights.csv",
                ):
                    if path.is_file():
                        files.add(path)

        def key(path: Path) -> str:
            try:
                return str(path.resolve().relative_to(root.resolve()))
            except ValueError:
                return str(path)

        return {key(path): _sha256(path) for path in sorted(files)}

    def build_command(self, spec: FoldSpec) -> list[str]:
        fold_dir = self.work_dir / spec.fold_id
        command = [
            "uv", "run", "python", "main.py", "build-outputs",
            "--year", str(self.config.year),
            "--enable-mixture-allocation",
            "--enable-exposure-ensemble",
            "--enable-soft-shrinkage",
            "--bg-ags-core-out", str(fold_dir / f"crimerisk_block_group_{self.config.year}_ags_core.parquet"),
            "--tract-ags-core-out", str(fold_dir / f"crimerisk_tract_{self.config.year}_ags_core.parquet"),
            "--build-manifest-out", str(fold_dir / "manifest.json"),
            "--feed-inputs-dir", str(self.fold_inputs_dir(spec)),
        ]
        if spec.exclude_city_key is not None:
            command.extend(["--exclude-feed-city", spec.exclude_city_key])
        if spec.feed_year_end is not None:
            command.extend(["--feed-year-end", str(spec.feed_year_end)])
        if self.prior_command(spec) is not None:
            command.extend(["--bg-prior-path", str(self.fold_prior_path(spec))])
        blend_path = self.fold_blend_path(spec)
        if blend_path is None:
            command.extend(self.config.variant_args)
        else:
            command.extend(_without_flag(self.config.variant_args, FINISHED_SHARE_BLEND_FLAG))
            command.extend([FINISHED_SHARE_BLEND_FLAG, str(blend_path)])
        return command

    def dry_run_payload(self) -> list[dict[str, object]]:
        return [
            {
                **asdict(spec),
                "prior_command": self.prior_command(spec),
                "command": None if spec.candidate_dir else self.build_command(spec),
            }
            for spec in self.fold_specs()
        ]

    def variant_record(self) -> dict[str, object]:
        """The candidate change this run measures, with a digest of every file it names."""
        files = {}
        for value in self.config.variant_args:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = self.paths.repo_root / candidate
            if not value.startswith("-") and candidate.is_file():
                files[value] = _sha256(candidate)
        return {
            "name": self.config.variant or "baseline",
            "build_args": list(self.config.variant_args),
            "file_sha256": files,
        }

    def pin_truth(self, *, prior_manifest: dict | None) -> dict[str, object]:
        """Write (first attempt) or re-verify (resume) the run's truth snapshot.

        The snapshot is the admitted truth over the fixed window, checked city by city, year by
        year and offense by offense against the reconciliation table of the feed build it came
        from. A collapse fails the run before any fold is built.
        """
        reconciliation_path = (
            self.paths.state_dir / "modeling" / f"city_incident_reconciliation_{self.config.year}.parquet"
        )
        if not reconciliation_path.exists():
            raise FileNotFoundError(f"Truth completeness needs {reconciliation_path}")
        completeness = truth_completeness(self.truth, pd.read_parquet(reconciliation_path))
        completeness_path = self.run_dir / "truth_completeness.csv"
        completeness_path.parent.mkdir(parents=True, exist_ok=True)
        completeness.to_csv(completeness_path, index=False)
        # Checked before the snapshot is written, so a failed check never pins bad truth.
        assert_truth_complete(completeness)
        if not self.truth_snapshot_path.exists():
            self.truth_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.truth.to_parquet(self.truth_snapshot_path, index=False)
        truth_sha = _sha256(self.truth_snapshot_path)
        pinned = ((prior_manifest or {}).get("inputs") or {}).get("city_truth") or {}
        if pinned.get("sha256") and pinned["sha256"] != truth_sha:
            raise RuntimeError(
                f"Truth snapshot {self.truth_snapshot_path} no longer matches the sha256 pinned "
                f"in this run's manifest ({pinned['sha256']})"
            )
        return {
            "city_truth": {
                "path": str(self.truth_snapshot_path),
                "sha256": truth_sha,
                "year_start": TRUTH_YEAR_START,
                "year_end": TRUTH_YEAR_END,
            },
            "city_reconciliation": {
                "path": str(reconciliation_path),
                "sha256": _sha256(reconciliation_path),
            },
            "truth_completeness": {
                "path": str(completeness_path),
                "sha256": _sha256(completeness_path),
                "cells": int(len(completeness)),
                "status_counts": {
                    str(key): int(value)
                    for key, value in completeness["status"].value_counts().sort_index().items()
                },
            },
        }

    def run(self) -> pd.DataFrame:
        specs = self.fold_specs()
        if self.config.dry_run:
            print(json.dumps(self.dry_run_payload(), indent=2))
            return pd.DataFrame()
        ags_path = ensure_ags_tract_cache(self.paths) if self.config.ags_comparator else None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scored_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.run_dir / "run_manifest.json"
        prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if prior is not None and "completed_at_utc" in prior:
            raise RuntimeError(
                f"Run {self.config.run_id!r} already completed; use a new run id"
            )
        if prior is not None:
            prior_config = dict(prior.get("config", {}))
            current_config = json.loads(json.dumps(asdict(self.config)))
            if prior_config != current_config:
                raise RuntimeError(
                    f"Incomplete run {self.config.run_id!r} has a different configuration"
                )
        dependencies = self.dependency_fingerprint()
        if prior is not None:
            self._assert_resumable(prior, dependencies)
        inputs = self.pin_truth(prior_manifest=prior)
        if ags_path is not None:
            inputs[AGS_ARM] = {"path": str(ags_path), "sha256": _sha256(ags_path)}
        if "model" in {spec.fold_type for spec in specs}:
            panel = self.e4_panel()
            inputs["e4_panel"] = {
                "panel_id": panel["panel_id"],
                "path": panel["manifest_path"],
                "sha256": panel["manifest_sha256"],
            }
        disk_before = shutil.disk_usage(self.paths.repo_root).free
        started = time.monotonic()
        run_manifest: dict[str, object] = {
            "run_id": self.config.run_id,
            "started_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "config": asdict(self.config),
            "variant": self.variant_record(),
            "fold_ids": [spec.fold_id for spec in specs],
            "pot_version": ot.__version__,
            "scoring_schema_version": SCORING_SCHEMA_VERSION,
            "folds": [],
            "inputs": inputs,
            "dependencies": dependencies,
            "disk_free_bytes_before": int(disk_before),
        }
        resumed_records: dict[str, dict[str, object]] = {}
        if prior is not None:
            run_manifest["started_at_utc"] = prior.get("started_at_utc", run_manifest["started_at_utc"])
            run_manifest["disk_free_bytes_before"] = prior.get("disk_free_bytes_before", int(disk_before))
            for record in prior.get("folds", []):
                if isinstance(record, dict) and isinstance(record.get("fold_id"), str):
                    resumed_records[str(record["fold_id"])] = record
        protected_before = self._fingerprint(self.protected_state_paths())
        observed_before = self._fingerprint(self.observed_state_paths())
        run_manifest["protected_state_paths"] = sorted(protected_before)
        score_parts: list[pd.DataFrame] = []
        for spec in specs:
            checkpoint = resumed_records.get(spec.fold_id)
            if checkpoint is not None and self._checkpoint_is_valid(spec, checkpoint):
                fold_record = checkpoint
                scores = pd.read_csv(str(checkpoint["score_path"]))
                print(f"resumed {spec.fold_id}: retained scored checkpoint", flush=True)
            else:
                fold_record, scores = self._run_fold(spec, ags_path=ags_path)
            score_parts.append(scores)
            run_manifest["folds"].append(fold_record)
            _json_write(manifest_path, run_manifest)
        run_manifest["protected_state"] = self._assert_state_unchanged(
            protected_before, observed_before
        )
        if not score_parts:
            raise RuntimeError("No evaluation folds produced scores")
        cells = pd.concat(score_parts, ignore_index=True)
        cells.to_csv(self.run_dir / "cell_scores.csv", index=False)
        paired_cells = cells
        if all(Path(str(r.get("input_path") or "missing-input")).is_file() for r in run_manifest["folds"]):
            run_manifest["processed_truth_sha256"] = processed_truth_sha256(
                processed_truth_table(self._retained_inputs(run_manifest, run_id=self.config.run_id))
            )
        if self.config.baseline_run:
            baseline_cells, baseline_record = self._baseline_cells(run_manifest)
            paired_cells = pd.concat([cells, baseline_cells], ignore_index=True)
            run_manifest["baseline_run"] = baseline_record
        outputs = self.write_results(self.run_dir, paired_cells)
        attempt_wall_seconds = float(time.monotonic() - started)
        fold_wall_seconds = float(
            sum(float(record["wall_seconds"]) for record in run_manifest["folds"])
        )
        run_manifest.update(
            completed_at_utc=datetime.now(tz=timezone.utc).isoformat(),
            wall_seconds=fold_wall_seconds,
            attempt_wall_seconds=attempt_wall_seconds,
            disk_free_bytes_after=int(shutil.disk_usage(self.paths.repo_root).free),
            cell_scores_sha256=_sha256(self.run_dir / "cell_scores.csv"),
            **{f"{name}_sha256": _sha256(path) for name, path in outputs.items()},
        )
        run_manifest["disk_used_bytes"] = int(
            run_manifest["disk_free_bytes_before"] - run_manifest["disk_free_bytes_after"]
        )
        _json_write(self.run_dir / "run_manifest.json", run_manifest)
        return pd.read_csv(outputs["results"])

    def write_results(self, out_dir: Path, cells: pd.DataFrame) -> dict[str, Path]:
        """Paired primary tables, unmatched secondary table, calibration and results.md."""
        out_dir.mkdir(parents=True, exist_ok=True)
        primary_cells = cells[_cell_scope(cells).eq("primary")]
        arms, differences = self._paired_summary(primary_cells)
        unmatched = self._summarize(primary_cells)
        spill = cells[_cell_scope(cells).ne("primary")]
        calibration = self._calibration_results()
        outputs = {
            "results": out_dir / "results.csv",
            "differences": out_dir / "differences.csv",
            "results_unmatched": out_dir / "results_unmatched.csv",
            "secondary_cells": out_dir / "secondary_cells.csv",
            "calibration": out_dir / "calibration.csv",
            "results_md": out_dir / "results.md",
        }
        arms.to_csv(outputs["results"], index=False)
        differences.to_csv(outputs["differences"], index=False)
        unmatched.to_csv(outputs["results_unmatched"], index=False)
        spill.to_csv(outputs["secondary_cells"], index=False)
        calibration.to_csv(outputs["calibration"], index=False)
        outputs["results_md"].write_text(
            self._markdown(arms, differences, calibration, spill)
        )
        return outputs

    def _retained_inputs(self, manifest: dict, *, run_id: str) -> pd.DataFrame:
        parts = []
        for record in manifest.get("folds", []):
            path = Path(str(record.get("input_path") or "missing-input"))
            if not path.is_file():
                # A run directory moved with its checkout: the input sits under the run's own folder.
                path = self.paths.state_dir / "eval" / run_id / "inputs" / f"{record.get('fold_id')}.parquet"
            if not path.is_file():
                raise RuntimeError(f"Run {run_id!r} has no retained scoring input for fold {record.get('fold_id')}")
            if record.get("input_sha256") and _sha256(path) != record["input_sha256"]:
                raise RuntimeError(f"Retained input {path} of run {run_id!r} changed since the run")
            parts.append(pd.read_parquet(path))
        if not parts:
            raise RuntimeError(f"Run {run_id!r} retained no scoring inputs")
        return pd.concat(parts, ignore_index=True)

    def _baseline_cells(self, run_manifest: dict) -> tuple[pd.DataFrame, dict[str, object]]:
        """The baseline's `ours` predictions re-scored on THIS run's processed truth.

        Pairing two runs on their own scored metrics compared different truth vectors (42 of 56
        comparison groups had different truth totals). Here both runs are scored against one
        frozen, keyed truth table -- this run's retained (group, support_id, truth_count) -- and a
        group is paired only where the baseline predicted exactly the same support IDs. The run's
        own `ours` is re-scored the same way and must reproduce its recorded metrics.
        """
        baseline_dir = self.paths.state_dir / "eval" / self.config.baseline_run
        manifest = json.loads((baseline_dir / "run_manifest.json").read_text())
        if "completed_at_utc" not in manifest:
            raise RuntimeError(f"Baseline run {self.config.baseline_run!r} is not complete")
        truth = processed_truth_table(self._retained_inputs(run_manifest, run_id=self.config.run_id))
        baseline = self._retained_inputs(manifest, run_id=self.config.baseline_run)
        current = self._retained_inputs(run_manifest, run_id=self.config.run_id)
        cells, exclusions = [], []
        recorded = pd.read_csv(self.run_dir / "cell_scores.csv")
        for arm in ("ours", "ours_tract"):
            column = f"prediction_{arm}"
            if column not in current.columns or column not in baseline.columns:
                continue
            groups = current.loc[current[column].notna(), RESCORE_GROUP_KEYS].drop_duplicates()
            truth_arm = truth.merge(groups, on=RESCORE_GROUP_KEYS, how="inner")
            scored, excluded = rescore_on_truth(
                truth_arm, baseline[baseline[column].notna()], arm_column=column, arm_name=BASELINE_ARM_PREFIX + arm
            )
            cells.append(scored)
            exclusions.append(excluded)
            # The run's own arm, re-scored the same way, must reproduce what the run recorded.
            own, _none = rescore_on_truth(truth_arm, current[current[column].notna()], arm_column=column, arm_name=arm)
            check = own.merge(
                recorded[recorded["arm"].eq(arm)],
                on=["fold_id", "city_name", "jurisdiction_id", "offense", "arm", "support", "cell_scope"],
                suffixes=("", "_recorded"),
            )
            drift = (check["tvd"] - check["tvd_recorded"]).abs()
            if len(check) == 0 or float(drift.max()) > 1e-9:
                raise RuntimeError(
                    f"Re-scoring this run's {arm} does not reproduce its recorded TVD "
                    f"({len(check)} groups, max drift {float(drift.max()) if len(check) else float('nan'):.3e})"
                )
        baseline_cells = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()
        excluded = pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame()
        excluded.to_csv(self.run_dir / "pairing_exclusions.csv", index=False)
        reuse = pd.read_csv(self.run_dir / "cell_scores.csv")[["fold_id", "city_name", "offense", "reuse_flags"]].drop_duplicates(
            ["fold_id", "city_name", "offense"]
        )
        baseline_cells = baseline_cells.merge(reuse, on=["fold_id", "city_name", "offense"], how="left")
        return baseline_cells, {
            "run_id": self.config.baseline_run,
            "variant": manifest.get("variant"),
            "pairing": "baseline predictions re-scored on this run's processed truth",
            "processed_truth_sha256": processed_truth_sha256(truth),
            "paired_groups": int(len(baseline_cells)),
            "excluded_groups": int(len(excluded)),
            "pairing_exclusions_sha256": _sha256(self.run_dir / "pairing_exclusions.csv"),
        }

    def protected_state_paths(self) -> list[Path]:
        """The shared city-feed artifacts a fold build must leave exactly as it found them.

        These are the ones a held-out build would otherwise overwrite with a deliberately
        incomplete world: the share surface, its per-city and combined reconciliation tables, and
        the next-phase validation surface. An evaluation run left
        `state/modeling/city_incident_share_surface.parquet` at year_end=2023 this way, which is
        a released input silently replaced by a temporal fold's truncated copy. Folds now write
        theirs under the run (`--feed-inputs-dir`); this is the tripwire that says so if some
        other stage ever reaches back into the shared location.
        """
        modeling = self.paths.state_dir / "modeling"
        protected: list[Path] = []
        if modeling.is_dir():
            protected.append(modeling / "city_incident_share_surface.parquet")
            protected.extend(sorted(modeling.glob("city_incident_reconciliation_*.parquet")))
            protected.extend(
                sorted(modeling.glob("next_phase_validation_city_incident_share_surface_*.parquet"))
            )
        reconciliation = self.paths.review_analysis_dir / "city_reconciliation"
        if reconciliation.is_dir():
            protected.extend(sorted(reconciliation.glob("*.parquet")))
        return [path for path in protected if path.is_file()]

    def observed_state_paths(self) -> list[Path]:
        """Everything else under state/modeling, recorded but not enforced.

        A fold build may legitimately rebuild a stale model-side artifact -- the mixture shares
        and the exposure normalizers carry no city-feed input, so their content does not depend
        on which fold asked for them. Those changes are reported rather than fatal, so that a
        real one is visible without every run failing on an ordinary rebuild.
        """
        modeling = self.paths.state_dir / "modeling"
        if not modeling.is_dir():
            return []
        protected = {path.resolve() for path in self.protected_state_paths()}
        return [
            path for path in sorted(modeling.glob("*.parquet")) if path.resolve() not in protected
        ]

    def _fingerprint(self, paths: list[Path]) -> dict[str, str]:
        return {str(path): _sha256(path) for path in paths if path.is_file()}

    @staticmethod
    def _diff(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
        return {
            "changed": sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
            "removed": sorted(before.keys() - after.keys()),
            "added": sorted(after.keys() - before.keys()),
        }

    def _assert_state_unchanged(
        self, protected_before: dict[str, str], observed_before: dict[str, str]
    ) -> dict[str, object]:
        protected = self._diff(protected_before, self._fingerprint(self.protected_state_paths()))
        observed = self._diff(observed_before, self._fingerprint(self.observed_state_paths()))
        report: dict[str, object] = {
            "protected_paths_checked": len(protected_before),
            "protected": protected,
            "observed_paths_checked": len(observed_before),
            "observed": observed,
        }
        if any(observed.values()):
            print(
                "WARNING: gold evaluation changed shared model artifacts under state/modeling: "
                f"{observed}",
                flush=True,
            )
        if any(protected.values()):
            raise RuntimeError(
                "Gold evaluation modified shared city-feed state it must only read: "
                f"{protected}. A fold builds a deliberately held-out surface and must write it "
                "under the run (--feed-inputs-dir), never into state/modeling. Restore these "
                "files before using this run or the next release."
            )
        return report

    def _assert_resumable(self, prior: dict, dependencies: dict[str, str]) -> None:
        """Refuse to resume unless every code and numerical dependency has the same content.

        Retained folds were built by the code and state recorded when the run started; mixing
        them with folds built now is only sound if nothing a build reads has changed.
        """
        recorded = prior.get("dependencies")
        if not isinstance(recorded, dict):
            raise RuntimeError(
                f"Incomplete run {self.config.run_id!r} records no dependency fingerprint, so its "
                "retained folds cannot be shown to match the current code; use a new run id"
            )
        diff = self._diff(recorded, dependencies)
        if any(diff.values()):
            shown = {name: values[:10] for name, values in diff.items() if values}
            raise RuntimeError(
                f"Incomplete run {self.config.run_id!r} cannot resume: code or numerical "
                f"dependencies changed since it started {shown}; use a new run id"
            )

    def _checkpoint_is_valid(
        self, spec: FoldSpec, record: dict[str, object]
    ) -> bool:
        """Accept only complete, hash-matching checkpoints from this exact fold spec."""
        for key, value in asdict(spec).items():
            if record.get(key) != value:
                return False
        score_path = Path(str(record.get("score_path", "")))
        input_path = Path(str(record.get("input_path", "")))
        manifest_path = Path(str(record.get("manifest_path", "")))
        if not score_path.exists() or not input_path.exists() or not manifest_path.exists():
            return False
        return (
            record.get("scoring_schema_version") == SCORING_SCHEMA_VERSION
            and record.get("score_sha256") == _sha256(score_path)
            and record.get("input_sha256") == _sha256(input_path)
            and record.get("manifest_sha256") == _sha256(manifest_path)
        )

    def _run_command(self, command: list[str], *, log_path: Path, what: str) -> int:
        print(f"starting {what}: {' '.join(command)}", flush=True)
        environment = dict(os.environ)
        environment.setdefault("UV_CACHE_DIR", "/tmp/crimerisk-w1-uv-cache")
        completed = subprocess.run(
            command,
            cwd=self.paths.repo_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{what} failed with exit {completed.returncode}; see {log_path}"
            )
        return int(completed.returncode)

    def _run_fold(
        self, spec: FoldSpec, *, ags_path: Path | None
    ) -> tuple[dict[str, object], pd.DataFrame]:
        fold_dir = self.work_dir / spec.fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        surface_dir = Path(spec.candidate_dir) if spec.candidate_dir else fold_dir
        bg_path = surface_dir / f"crimerisk_block_group_{self.config.year}_ags_core.parquet"
        tract_path = surface_dir / f"crimerisk_tract_{self.config.year}_ags_core.parquet"
        manifest_path = surface_dir / "manifest.json"
        command: list[str] | None = None
        returncode: int | None = None
        prior_command = self.prior_command(spec)
        prior_wall_seconds: float | None = None
        blend_record: dict[str, object] | None = None
        if spec.candidate_dir:
            # Scoring only: the surface already exists and is read, never rebuilt or deleted.
            for path in (bg_path, tract_path, manifest_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Scoring-only candidate is missing {path}")
            print(f"scoring {spec.fold_id} on existing candidate {surface_dir}", flush=True)
        else:
            if shutil.disk_usage(self.paths.repo_root).free < 20 * 1024**3:
                raise RuntimeError("Gold evaluation stopped before build: less than 20 GiB free")
            blend_record = self._write_fold_blend(spec)
            if prior_command is not None:
                prior_started = time.monotonic()
                self._run_command(
                    prior_command, log_path=fold_dir / "prior.log", what=f"{spec.fold_id} prior"
                )
                prior_wall_seconds = float(time.monotonic() - prior_started)
            command = self.build_command(spec)
            returncode = self._run_command(
                command, log_path=fold_dir / "build.log", what=f"{spec.fold_id} build"
            )
        build_elapsed = time.monotonic() - started
        surface_sha256 = (
            {"block_group": _sha256(bg_path), "tract": _sha256(tract_path)}
            if spec.candidate_dir
            else None
        )
        self._fold_input_parts = []
        if spec.fold_type == "spatial":
            truth = self.truth[self.truth["city_key"].eq(spec.exclude_city_key)].copy()
            cities = [str(spec.exclude_city_key)]
            scores = self._score_feed_candidate(
                spec, bg_path=bg_path, tract_path=tract_path, truth=truth, city_keys=cities, ags_path=ags_path
            )
        elif spec.fold_type.startswith("temporal"):
            assert spec.feed_year_end is not None
            truth = self.truth[self.truth["year"].gt(spec.feed_year_end)].copy()
            scores = self._score_feed_candidate(
                spec,
                bg_path=bg_path,
                tract_path=tract_path,
                truth=truth,
                city_keys=self.city_keys,
                ags_path=ags_path,
                past_truth=self.truth[self.truth["year"].le(spec.feed_year_end)].copy(),
            )
        else:
            scores = self._score_model_candidate(
                spec,
                bg_path=bg_path,
                tract_path=tract_path,
                ags_path=ags_path,
                manifest_path=manifest_path,
            )
        score_path = self.scored_dir / f"{spec.fold_id}.csv"
        scores.to_csv(score_path, index=False)
        if not self._fold_input_parts:
            raise RuntimeError(f"Fold {spec.fold_id} retained no per-cell scoring input")
        fold_input = pd.concat(self._fold_input_parts, ignore_index=True)
        input_path = self.inputs_dir / f"{spec.fold_id}.parquet"
        fold_input.to_parquet(input_path, index=False)
        record = {
            **asdict(spec),
            "command": command,
            "prior_command": prior_command,
            "prior_wall_seconds": prior_wall_seconds,
            "blend_weights": blend_record,
            "build_wall_seconds": float(build_elapsed),
            "returncode": returncode,
            "surface_sha256": surface_sha256,
            "scoring_schema_version": SCORING_SCHEMA_VERSION,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "score_path": str(score_path),
            "score_sha256": _sha256(score_path),
            "score_rows": int(len(scores)),
            "input_path": str(input_path),
            "input_sha256": _sha256(input_path),
            "input_rows": int(len(fold_input)),
        }
        removed_bytes = 0
        # Only the harness's own fold directory is cleaned; a scoring-only candidate is untouched.
        for artifact in sorted(fold_dir.rglob("*"), reverse=True):
            if artifact.is_file() and artifact.name != "manifest.json":
                if artifact.suffix == ".parquet":
                    removed_bytes += artifact.stat().st_size
                artifact.unlink()
            elif artifact.is_dir():
                try:
                    artifact.rmdir()
                except OSError:
                    pass
        elapsed = time.monotonic() - started
        record["wall_seconds"] = float(elapsed)
        record["scoring_and_cleanup_wall_seconds"] = float(elapsed - build_elapsed)
        record["deleted_surface_bytes"] = int(removed_bytes)
        _json_write(fold_dir / "fold_record.json", record)
        print(
            f"completed {spec.fold_id}: {elapsed:.1f}s total, "
            f"{build_elapsed:.1f}s build",
            flush=True,
        )
        return record, scores

    def _surface_columns(self, *, tract: bool = False) -> list[str]:
        identity = ["tract_id"] if tract else ["block_group_geoid", "tract_id", "eb_jurisdiction_id"]
        columns = [*identity, f"population_{self.config.year}"]
        for offense in OFFENSES:
            columns.extend(
                [
                    f"expected_count_{offense}",
                    f"primary_denominator_{offense}",
                    f"primary_national_rate_per_100k_{offense}",
                    f"expected_count_{offense}_p10",
                    f"expected_count_{offense}_p50",
                    f"expected_count_{offense}_p90",
                    f"index_{offense}_primary_p25",
                    f"index_{offense}_primary_p75",
                    f"rate_{offense}_primary_ci95_lower",
                    f"rate_{offense}_primary_ci95_upper",
                    f"uncertainty_support_class_{offense}",
                ]
            )
        return columns

    def _load_candidate(self, bg_path: Path, tract_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        bg = pd.read_parquet(bg_path, columns=self._surface_columns(tract=False))
        tract = pd.read_parquet(tract_path, columns=self._surface_columns(tract=True))
        bg["block_group_geoid"] = bg["block_group_geoid"].astype("string").str.zfill(12)
        bg["tract_id"] = bg["tract_id"].astype("string").str.zfill(11)
        tract["tract_id"] = tract["tract_id"].astype("string").str.zfill(11)
        return bg, tract

    def _fully_contained_tracts(self, bg: pd.DataFrame, jurisdiction_id: str) -> pd.Index:
        membership = bg["eb_jurisdiction_id"].astype("string").eq(str(jurisdiction_id))
        contained = membership.groupby(bg["tract_id"], sort=False).all()
        return pd.Index(contained[contained].index.astype(str))

    def _interval_arrays(self, frame: pd.DataFrame, offense: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        denominator = pd.to_numeric(frame[f"primary_denominator_{offense}"], errors="coerce")
        national_rate = pd.to_numeric(
            frame[f"primary_national_rate_per_100k_{offense}"], errors="coerce"
        )
        p25 = (
            pd.to_numeric(frame[f"index_{offense}_primary_p25"], errors="coerce")
            / 100.0
            * national_rate
            / 100_000.0
            * denominator
        )
        p75 = (
            pd.to_numeric(frame[f"index_{offense}_primary_p75"], errors="coerce")
            / 100.0
            * national_rate
            / 100_000.0
            * denominator
        )
        ci95_low = (
            pd.to_numeric(frame[f"rate_{offense}_primary_ci95_lower"], errors="coerce")
            / 100_000.0
            * denominator
        )
        ci95_high = (
            pd.to_numeric(frame[f"rate_{offense}_primary_ci95_upper"], errors="coerce")
            / 100_000.0
            * denominator
        )
        return {
            50: (p25.to_numpy(float), p75.to_numpy(float)),
            80: (
                pd.to_numeric(frame[f"expected_count_{offense}_p10"], errors="coerce").to_numpy(float),
                pd.to_numeric(frame[f"expected_count_{offense}_p90"], errors="coerce").to_numpy(float),
            ),
            95: (ci95_low.to_numpy(float), ci95_high.to_numpy(float)),
        }

    def _interval_coverage(self, frame: pd.DataFrame, offense: str) -> dict[str, float]:
        truth = pd.to_numeric(frame["truth_count"], errors="coerce").fillna(0.0).to_numpy(float)
        point = pd.to_numeric(
            frame[f"expected_count_{offense}"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        intervals = self._interval_arrays(frame, offense)
        return {
            f"coverage_{level}": _share_coverage(truth, low, high, point)
            for level, (low, high) in intervals.items()
        }

    def _residual_training_jurisdictions(self, manifest_path: Path) -> set[str]:
        """Jurisdictions in the residual-training surface the scored build actually used."""
        manifest = json.loads(manifest_path.read_text())
        resolved = manifest.get("resolved_config") or {}
        recorded = resolved.get("residual_training_city_shares_path")
        if not recorded:
            return set()
        path = Path(str(recorded))
        if not path.exists() and "state" in path.parts:
            # A candidate built in another checkout records that checkout's absolute path.
            relative = Path(*path.parts[len(path.parts) - path.parts[::-1].index("state"):])
            path = self.paths.state_dir / relative
        if not path.exists():
            raise FileNotFoundError(f"Residual training surface named by {manifest_path} not found")
        training = _load_residual_training_city_shares(
            path=path,
            exclude_validation_case_types=tuple(
                resolved.get("residual_training_exclude_validation_case_types") or ()
            ),
            exclude_feed_city_keys=tuple(resolved.get("exclude_feed_city_keys") or ()),
            year_end=resolved.get("feed_year_end"),
        )
        if training.empty:
            return set()
        return set(training["jurisdiction_id"].dropna().astype(str))

    @staticmethod
    def _load_ags(ags_path: Path | None) -> pd.DataFrame | None:
        if ags_path is None:
            return None
        ags = pd.read_parquet(ags_path, columns=["ID", "POP_C", *AGS_COLUMN.values()]).rename(
            columns={"ID": "tract_id"}
        )
        ags["tract_id"] = ags["tract_id"].astype("string").str.zfill(11)
        return ags

    def _ags_uses_city_feed(self, jurisdiction_id: str) -> bool:
        return str(jurisdiction_id) in self.ags_feed_jurisdictions

    def _retain_support_input(
        self,
        *,
        spec: FoldSpec,
        city_name: str,
        jurisdiction_id: str,
        offense: str,
        support_role: str,
        support: str,
        base: pd.DataFrame,
        predictions: dict[str, np.ndarray | pd.Series],
        coordinates: pd.DataFrame,
        intervals: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        support_class: pd.Series | np.ndarray | None = None,
        cell_scope: str = "primary",
    ) -> None:
        retained = pd.DataFrame(
            {
                "fold_id": spec.fold_id,
                "fold_type": spec.fold_type,
                "city_name": city_name,
                "jurisdiction_id": jurisdiction_id,
                "ags_uses_city_feed": self._ags_uses_city_feed(jurisdiction_id),
                "offense": offense,
                "support_role": support_role,
                "support": support,
                "support_id": base["support_id"].astype(str).to_numpy(),
                "truth_count": pd.to_numeric(base["truth_count"], errors="coerce").fillna(0.0).to_numpy(float),
                "population": pd.to_numeric(base["population"], errors="coerce").fillna(0.0).to_numpy(float),
                "support_lon": coordinates["lon"].to_numpy(float),
                "support_lat": coordinates["lat"].to_numpy(float),
                "cell_scope": cell_scope,
            }
        )
        for arm, values in predictions.items():
            retained[f"prediction_{arm}"] = np.asarray(values, dtype=float)
        if intervals is not None:
            for level, (lower, upper) in intervals.items():
                retained[f"interval_{level}_lower"] = np.asarray(lower, dtype=float)
                retained[f"interval_{level}_upper"] = np.asarray(upper, dtype=float)
        if support_class is not None:
            retained["uncertainty_support_class"] = np.asarray(support_class, dtype=object)
        self._fold_input_parts.append(retained)

    def _score_feed_candidate(
        self,
        spec: FoldSpec,
        *,
        bg_path: Path,
        tract_path: Path,
        truth: pd.DataFrame,
        city_keys: list[str],
        ags_path: Path | None,
        past_truth: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        bg, tract = self._load_candidate(bg_path, tract_path)
        ags = self._load_ags(ags_path)
        if ags is not None:
            ags = ags.set_index("tract_id")
        rows: list[dict[str, object]] = []
        city_info = (
            self.truth[["city_key", "city_name", "jurisdiction_id"]]
            .drop_duplicates("city_key")
            .set_index("city_key")
        )
        population_col = f"population_{self.config.year}"
        bg_indexed = bg.set_index("block_group_geoid")
        tract_indexed = tract.set_index("tract_id")
        for city_key in city_keys:
            if city_key not in city_info.index:
                continue
            city_name = str(city_info.loc[city_key, "city_name"])
            jurisdiction_id = str(city_info.loc[city_key, "jurisdiction_id"])
            footprint = pd.Index(
                bg.loc[
                    bg["eb_jurisdiction_id"].astype("string").eq(jurisdiction_id),
                    "block_group_geoid",
                ]
            )
            contained_tracts = self._fully_contained_tracts(bg, jurisdiction_id)
            if footprint.empty or contained_tracts.empty:
                continue
            member_population = bg[["block_group_geoid", population_col]].rename(
                columns={population_col: "population"}
            )
            support_frames = {
                "block_group": bg_indexed.reindex(footprint).reset_index(),
                "tract": tract_indexed.reindex(contained_tracts).reset_index(),
            }
            support_frames["block_group"]["support_id"] = support_frames["block_group"][
                "block_group_geoid"
            ]
            support_frames["tract"]["support_id"] = support_frames["tract"]["tract_id"]
            matrices = {
                name: self.geometry.matrix(
                    frame["support_id"],
                    bg_population=member_population if name == "tract" else None,
                )
                for name, frame in support_frames.items()
            }
            coordinates = {
                name: self.geometry.coordinates(
                    frame["support_id"],
                    bg_population=member_population if name == "tract" else None,
                )
                for name, frame in support_frames.items()
            }
            observed_city = truth[truth["city_key"].eq(city_key)]
            past_city = (
                past_truth[past_truth["city_key"].eq(city_key)]
                if past_truth is not None
                else pd.DataFrame()
            )
            for offense in OFFENSES:
                observed = observed_city[observed_city["offense"].eq(offense)]
                if float(observed["incident_count"].sum()) <= 0.0:
                    continue
                primary_support = "tract" if offense in RARE_OFFENSES else "block_group"
                base = support_frames[primary_support].copy()
                truth_key = (
                    observed["block_group_geoid"].str[:11]
                    if primary_support == "tract"
                    else observed["block_group_geoid"]
                )
                truth_counts = observed.assign(_support=truth_key).groupby("_support")[
                    "incident_count"
                ].sum()
                base["truth_count"] = base["support_id"].map(truth_counts).fillna(0.0)
                population = pd.to_numeric(base[population_col], errors="coerce").fillna(0.0)
                base["population"] = population.to_numpy(float)
                arms: dict[str, pd.Series] = {
                    "ours": pd.to_numeric(base[f"expected_count_{offense}"], errors="coerce").fillna(0.0),
                    "population": population,
                    "primary_exposure": pd.to_numeric(
                        base[f"primary_denominator_{offense}"], errors="coerce"
                    ).fillna(0.0),
                }
                if not past_city.empty:
                    past = past_city[past_city["offense"].eq(offense)]
                    past_key = (
                        past["block_group_geoid"].str[:11]
                        if primary_support == "tract"
                        else past["block_group_geoid"]
                    )
                    past_counts = past.assign(_support=past_key).groupby("_support")[
                        "incident_count"
                    ].sum()
                    arms["past_counts"] = base["support_id"].map(past_counts).fillna(0.0)
                population_null_w1 = _population_null_w1(
                    truth_count=base["truth_count"].to_numpy(float),
                    population=population.to_numpy(float),
                    distance_matrix=matrices[primary_support],
                )
                for arm, prediction in arms.items():
                    metrics = _score_vector(
                        truth_count=base["truth_count"].to_numpy(float),
                        prediction=prediction.to_numpy(float),
                        support_ids=base["support_id"].astype(str).to_numpy(),
                        population=population.to_numpy(float),
                        distance_matrix=matrices[primary_support],
                        population_null_w1=population_null_w1,
                    )
                    if arm == "ours":
                        metrics.update(self._interval_coverage(base, offense))
                    rows.append(
                        self._cell_row(
                            spec=spec,
                            city_name=city_name,
                            jurisdiction_id=jurisdiction_id,
                            offense=offense,
                            arm=arm,
                            support=primary_support,
                            base=base,
                            metrics=metrics,
                        )
                    )

                interval_arrays = self._interval_arrays(base, offense)
                primary_predictions: dict[str, np.ndarray | pd.Series] = {
                    "ours": arms["ours"],
                    "population": arms["population"],
                    "primary_exposure": arms["primary_exposure"],
                }
                if "past_counts" in arms:
                    primary_predictions["past_counts"] = arms["past_counts"]
                self._retain_support_input(
                    spec=spec,
                    city_name=city_name,
                    jurisdiction_id=jurisdiction_id,
                    offense=offense,
                    support_role="primary",
                    support=primary_support,
                    base=base,
                    predictions=primary_predictions,
                    coordinates=coordinates[primary_support],
                    intervals=interval_arrays,
                    support_class=base[f"uncertainty_support_class_{offense}"],
                )

                tract_base = support_frames["tract"].copy()
                tract_truth = observed.assign(_support=observed["block_group_geoid"].str[:11]).groupby(
                    "_support"
                )["incident_count"].sum()
                tract_base["truth_count"] = tract_base["support_id"].map(tract_truth).fillna(0.0)
                tract_population = pd.to_numeric(
                    tract_base[population_col], errors="coerce"
                ).fillna(0.0).to_numpy(float)
                tract_base["population"] = tract_population
                bg_tract_prediction = _aggregate_bg_counts_to_tracts(
                    bg, offense=offense, tract_ids=contained_tracts
                )
                ours_tract_prediction = (
                    tract_base["support_id"].map(bg_tract_prediction).fillna(0.0).to_numpy(float)
                )
                tract_predictions: dict[str, np.ndarray] = {"ours_tract": ours_tract_prediction}
                if ags is not None:
                    ags_rows = ags.reindex(tract_base["support_id"].astype(str))
                    tract_predictions[AGS_ARM] = (
                        pd.to_numeric(ags_rows[AGS_COLUMN[offense]], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(float)
                        * pd.to_numeric(ags_rows["POP_C"], errors="coerce").fillna(0.0).to_numpy(float)
                    )
                tract_null_w1 = _population_null_w1(
                    truth_count=tract_base["truth_count"].to_numpy(float),
                    population=tract_population,
                    distance_matrix=matrices["tract"],
                )
                for arm, prediction in tract_predictions.items():
                    metrics = _score_vector(
                        truth_count=tract_base["truth_count"].to_numpy(float),
                        prediction=np.asarray(prediction, dtype=float),
                        support_ids=tract_base["support_id"].astype(str).to_numpy(),
                        population=tract_population,
                        distance_matrix=matrices["tract"],
                        population_null_w1=tract_null_w1,
                    )
                    rows.append(
                        self._cell_row(
                            spec=spec,
                            city_name=city_name,
                            jurisdiction_id=jurisdiction_id,
                            offense=offense,
                            arm=arm,
                            support="tract",
                            base=tract_base,
                            metrics=metrics,
                        )
                    )
                self._retain_support_input(
                    spec=spec,
                    city_name=city_name,
                    jurisdiction_id=jurisdiction_id,
                    offense=offense,
                    support_role="tract_comparison",
                    support="tract",
                    base=tract_base,
                    predictions=dict(tract_predictions),
                    coordinates=coordinates["tract"],
                )
        return pd.DataFrame(rows)

    def _cell_row(
        self,
        *,
        spec: FoldSpec,
        city_name: str,
        jurisdiction_id: str,
        offense: str,
        arm: str,
        support: str,
        base: pd.DataFrame,
        metrics: dict[str, float],
        cell_scope: str = "primary",
        residual_training_seen: bool = False,
    ) -> dict[str, object]:
        return {
            "fold_id": spec.fold_id,
            "fold_type": spec.fold_type,
            "city_name": city_name,
            "jurisdiction_id": jurisdiction_id,
            "ags_uses_city_feed": self._ags_uses_city_feed(jurisdiction_id),
            "offense": offense,
            "arm": arm,
            "support": support,
            "n_cells": int(len(base)),
            "n_incidents": float(pd.to_numeric(base["truth_count"], errors="coerce").fillna(0.0).sum()),
            "cell_scope": cell_scope,
            "reuse_flags": _reuse_flags(
                "model" if spec.fold_type == "model" else spec.fold_type,
                offense,
                murder_k_seen=jurisdiction_id in self.murder_k_jurisdictions,
                residual_training_seen=residual_training_seen,
            ),
            **metrics,
        }

    def _score_model_candidate(
        self,
        spec: FoldSpec,
        *,
        bg_path: Path,
        tract_path: Path,
        ags_path: Path | None,
        manifest_path: Path,
    ) -> pd.DataFrame:
        panel = self.e4_panel()
        expected = dict(panel["expected"])  # type: ignore[arg-type]
        rules = dict(panel["cell_rules"])  # type: ignore[arg-type]
        bg, tract = self._load_candidate(bg_path, tract_path)
        residual_training = self._residual_training_jurisdictions(manifest_path)
        population_col = f"population_{self.config.year}"
        pure = (
            bg.groupby("tract_id")["eb_jurisdiction_id"]
            .agg(["nunique", "first"])
            .query("nunique == 1")["first"]
        )
        tract = tract.copy()
        tract["jurisdiction_id"] = tract["tract_id"].map(pure)
        truth = pd.read_parquet(panel["truth"]["resolved_path"]).copy()  # type: ignore[index]
        spill_cells = _e4_spill_keys(
            truth, multi_jurisdiction_sources=panel["multi_jurisdiction_sources"]  # type: ignore[arg-type]
        )
        truth["block_group_geoid"] = truth["block_group_geoid"].astype("string").str.zfill(12)
        truth["tract_id"] = truth["block_group_geoid"].str[:11]
        truth["tract_jurisdiction_id"] = truth["tract_id"].map(pure)
        truth = truth[truth["tract_jurisdiction_id"].eq(truth["jurisdiction_id"])].copy()
        registry = pd.read_csv(panel["registry"]["resolved_path"])  # type: ignore[index]
        frozen = pd.read_csv(
            panel["frozen_cells"]["resolved_path"],  # type: ignore[index]
            usecols=["variant", "source", "jurisdiction_id", "offense"],
        )
        frozen = frozen[frozen["variant"].eq(panel["frozen_cells"]["variant"])][  # type: ignore[index]
            ["source", "jurisdiction_id", "offense"]
        ].drop_duplicates()
        if (
            len(frozen) != int(expected["frozen_cells"])
            or frozen["source"].nunique() != int(expected["frozen_sources"])
        ):
            raise ValueError(
                "Frozen E4 base-cell identity changed: "
                f"cells={len(frozen)}, sources={frozen['source'].nunique()}"
            )
        frozen_spill = {
            (str(source), str(jurisdiction_id), str(offense))
            for source, jurisdiction_id, offense in frozen.itertuples(index=False, name=None)
            if (str(source), str(jurisdiction_id)) in spill_cells
        }
        if (
            len(frozen_spill) != int(expected["spill_cells"])
            or len({key[1] for key in frozen_spill}) != int(expected["spill_jurisdictions"])
        ):
            raise ValueError(
                "E4 spill-cell identity changed: "
                f"cells={len(frozen_spill)}, jurisdictions={len({key[1] for key in frozen_spill})}"
            )
        registry = registry.merge(
            frozen,
            on=["source", "jurisdiction_id", "offense"],
            how="inner",
            validate="one_to_one",
        )
        ags = self._load_ags(ags_path)
        merged = (
            tract.merge(ags, on="tract_id", how="left") if ags is not None else tract
        ).set_index("tract_id")
        pooled_all = truth.groupby(["source", "jurisdiction_id", "offense"])["incident_count"].sum()
        retained = truth.groupby(
            ["source", "jurisdiction_id", "offense", "tract_id"]
        )["incident_count"].sum()
        member_population = bg[["block_group_geoid", population_col]].rename(
            columns={population_col: "population"}
        )
        rows: list[dict[str, object]] = []
        for cell in registry.itertuples(index=False):
            key = (str(cell.source), str(cell.jurisdiction_id), str(cell.offense))
            try:
                observed = retained.loc[key]
            except KeyError:
                continue
            jurisdiction = merged[merged["jurisdiction_id"].astype("string").eq(key[1])].copy()
            if len(jurisdiction) < int(rules["min_tracts"]):
                continue
            jurisdiction["truth_count"] = jurisdiction.index.to_series().map(observed).fillna(0.0)
            kept_incidents = float(jurisdiction["truth_count"].sum())
            total_incidents = float(pooled_all.get(key, 0.0))
            if (
                kept_incidents < float(rules["min_kept_incidents"])
                or total_incidents <= 0.0
                or kept_incidents / total_incidents < float(rules["min_kept_share"])
            ):
                continue
            jurisdiction["support_id"] = jurisdiction.index.astype(str)
            distance = self.geometry.matrix(
                jurisdiction["support_id"], bg_population=member_population
            )
            population = pd.to_numeric(jurisdiction[population_col], errors="coerce").fillna(0.0)
            jurisdiction["population"] = population.to_numpy(float)
            bg_tract_prediction = _aggregate_bg_counts_to_tracts(
                bg, offense=key[2], tract_ids=jurisdiction.index
            )
            arms = {
                "ours": pd.to_numeric(
                    jurisdiction[f"expected_count_{key[2]}"], errors="coerce"
                ).fillna(0.0),
                "ours_tract": jurisdiction["support_id"].map(bg_tract_prediction).fillna(0.0),
                "population": population,
                "primary_exposure": pd.to_numeric(
                    jurisdiction[f"primary_denominator_{key[2]}"], errors="coerce"
                ).fillna(0.0),
            }
            if ags is not None:
                arms[AGS_ARM] = pd.to_numeric(
                    jurisdiction[AGS_COLUMN[key[2]]], errors="coerce"
                ).fillna(0.0) * pd.to_numeric(jurisdiction["POP_C"], errors="coerce").fillna(0.0)
            population_null_w1 = _population_null_w1(
                truth_count=jurisdiction["truth_count"].to_numpy(float),
                population=population.to_numpy(float),
                distance_matrix=distance,
            )
            cell_scope = "secondary_e4_spill" if key in frozen_spill else "primary"
            residual_training_seen = key[1] in residual_training
            for arm, prediction in arms.items():
                metrics = _score_vector(
                    truth_count=jurisdiction["truth_count"].to_numpy(float),
                    prediction=prediction.to_numpy(float),
                    support_ids=jurisdiction["support_id"].to_numpy(str),
                    population=population.to_numpy(float),
                    distance_matrix=distance,
                    population_null_w1=population_null_w1,
                )
                if arm == "ours":
                    metrics.update(self._interval_coverage(jurisdiction, key[2]))
                rows.append(
                    self._cell_row(
                        spec=spec,
                        city_name=str(cell.source),
                        jurisdiction_id=key[1],
                        offense=key[2],
                        arm=arm,
                        support="tract",
                        base=jurisdiction.reset_index(),
                        metrics=metrics,
                        cell_scope=cell_scope,
                        residual_training_seen=residual_training_seen,
                    )
                )
            self._retain_support_input(
                spec=spec,
                city_name=str(cell.source),
                jurisdiction_id=key[1],
                offense=key[2],
                support_role="primary_tract_comparison",
                support="tract",
                base=jurisdiction.reset_index(),
                predictions={arm: prediction for arm, prediction in arms.items()},
                coordinates=self.geometry.coordinates(
                    jurisdiction["support_id"], bg_population=member_population
                ),
                intervals=self._interval_arrays(jurisdiction, key[2]),
                support_class=jurisdiction[f"uncertainty_support_class_{key[2]}"],
                cell_scope=cell_scope,
            )
        scored = pd.DataFrame(rows)
        if not scored.empty:
            evaluated = scored[scored["arm"].eq("ours")]
            if (
                len(evaluated) != int(expected["scored_cells"])
                or evaluated["jurisdiction_id"].nunique() != int(expected["scored_jurisdictions"])
                or evaluated["city_name"].nunique() != int(expected["scored_sources"])
            ):
                raise ValueError(
                    f"E4 model-lane universe changed from panel {panel['panel_id']!r}: "
                    f"cells={len(evaluated)}, sources={evaluated['city_name'].nunique()}, "
                    f"jurisdictions={evaluated['jurisdiction_id'].nunique()}"
                )
        return scored

    def _paired_summary(self, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Matched-set arm estimates and paired differences with shared bootstrap indices.

        Per fold type, offense and table, the cities are the intersection where every compared
        arm has a TVD. Every arm and every difference in that block is averaged over the same
        cities and resampled with the same 1,000 city-cluster index draws. The primary table
        pools every city; the AGS table keeps the cities AGS takes no feed from.
        """
        cells = cells.copy()
        if "ags_uses_city_feed" not in cells.columns:
            cells["ags_uses_city_feed"] = False
        cells["ags_uses_city_feed"] = cells["ags_uses_city_feed"].astype(str).str.lower().eq("true")
        arm_rows: list[dict[str, object]] = []
        difference_rows: list[dict[str, object]] = []
        tract_family = {*TRACT_ARMS, BASELINE_ARM_PREFIX + "ours_tract"}
        for (fold_type, offense), group in cells.groupby(["fold_type", "offense"], sort=True):
            families = [
                ("primary", "all", group[~group["arm"].isin(tract_family)]),
                (
                    "ags_tract",
                    "ags_feed_free",
                    group[group["arm"].isin(tract_family) & ~group["ags_uses_city_feed"]],
                ),
            ]
            for family, city_set, frame in families:
                if frame.empty:
                    continue
                arms = _ordered_arms(frame["arm"].unique())
                tvd = frame.pivot_table(
                    index="city_name", columns="arm", values="tvd", aggfunc="mean"
                ).reindex(columns=arms)
                matched = sorted(tvd.index[tvd.notna().all(axis=1)].astype(str))
                if not matched:
                    continue
                frame = frame[frame["city_name"].astype(str).isin(matched)]
                n = len(matched)
                indices = (
                    np.random.default_rng(self.config.bootstrap_seed).integers(
                        0, n, size=(self.config.bootstrap_iterations, n)
                    )
                    if n >= 2
                    else None
                )
                focus = [arm for arm in arms if _is_focus_arm(arm)]
                common = {
                    "run_id": self.config.run_id,
                    "fold_type": fold_type,
                    "fold_label": FOLD_LABELS.get(str(fold_type), str(fold_type)),
                    "offense": offense,
                    "family": family,
                    "city_set": city_set,
                    "cities": ";".join(matched),
                }
                for metric in PAIRED_METRICS:
                    if metric not in frame.columns:
                        continue
                    values = (
                        frame.pivot_table(
                            index="city_name", columns="arm", values=metric, aggfunc="mean"
                        )
                        .reindex(index=matched, columns=arms)
                        .to_numpy(float)
                    )
                    for position, arm in enumerate(arms):
                        arm_cells = frame[frame["arm"].eq(arm)]
                        arm_rows.append(
                            {
                                **common,
                                "arm": arm,
                                "metric": metric,
                                **_paired_estimate(values[:, position], indices),
                                "n_cells": int(arm_cells["n_cells"].sum()),
                                "n_incidents": float(arm_cells["n_incidents"].sum()),
                                "reuse_flags": " | ".join(
                                    sorted(set(arm_cells["reuse_flags"].astype(str)))
                                ),
                            }
                        )
                    for arm in focus:
                        for reference in arms:
                            if reference == arm:
                                continue
                            delta = values[:, arms.index(arm)] - values[:, arms.index(reference)]
                            difference_rows.append(
                                {
                                    **common,
                                    "arm": arm,
                                    "reference_arm": reference,
                                    "metric": metric,
                                    **_paired_estimate(delta, indices),
                                }
                            )
        arm_columns = [
            "run_id", "fold_type", "fold_label", "offense", "family", "city_set", "arm", "metric",
            "estimate", "ci95_lower", "ci95_upper", "n_cities", "n_cells", "n_incidents",
            "cities", "reuse_flags",
        ]
        difference_columns = [
            "run_id", "fold_type", "fold_label", "offense", "family", "city_set", "arm",
            "reference_arm", "metric", "estimate", "ci95_lower", "ci95_upper", "n_cities", "cities",
        ]
        return (
            pd.DataFrame(arm_rows, columns=arm_columns),
            pd.DataFrame(difference_rows, columns=difference_columns),
        )

    def _summarize(self, cells: pd.DataFrame) -> pd.DataFrame:
        cells = cells.copy()
        if "ags_uses_city_feed" not in cells.columns:
            cells["ags_uses_city_feed"] = False
        metric_columns = [
            "tvd", "log_score", "spearman", "top_decile_capture", "w1_spatial_skill",
            "coverage_50", "coverage_80", "coverage_95",
        ]
        rows: list[dict[str, object]] = []
        group_columns = ["fold_type", "offense", "arm", "ags_uses_city_feed"]
        for keys, group in cells.groupby(group_columns, sort=True, dropna=False):
            fold_type, offense, arm, ags_uses_city_feed = keys
            for metric in metric_columns:
                if metric not in group.columns:
                    continue
                usable = group.dropna(subset=[metric]).copy()
                if usable.empty:
                    continue
                city_values = usable.groupby("city_name", sort=True)[metric].mean()
                point = float(city_values.mean())
                if len(city_values) >= 2:
                    rng = np.random.default_rng(
                        self.config.bootstrap_seed
                        + sum(ord(char) for char in f"{fold_type}|{offense}|{arm}|{metric}")
                    )
                    values = city_values.to_numpy(float)
                    draws = np.empty(self.config.bootstrap_iterations, dtype=float)
                    for index in range(self.config.bootstrap_iterations):
                        draws[index] = float(
                            values[rng.integers(0, len(values), size=len(values))].mean()
                        )
                    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
                else:
                    ci_low = ci_high = float("nan")
                flags = sorted(set(usable["reuse_flags"].astype(str)))
                rows.append(
                    {
                        "run_id": self.config.run_id,
                        "fold_type": fold_type,
                        "offense": offense,
                        "arm": arm,
                        "ags_uses_city_feed": bool(ags_uses_city_feed),
                        "metric": metric,
                        "estimate": point,
                        "ci95_lower": float(ci_low),
                        "ci95_upper": float(ci_high),
                        "n_cities": int(usable["city_name"].nunique()),
                        "n_cells": int(usable["n_cells"].sum()),
                        "n_incidents": float(usable["n_incidents"].sum()),
                        "reuse_flags": " | ".join(flags),
                    }
                )
        return pd.DataFrame(rows).sort_values(
            ["fold_type", "offense", "ags_uses_city_feed", "arm", "metric"], kind="mergesort"
        ).reset_index(drop=True)

    def _calibration_results(self) -> pd.DataFrame:
        prepared_parts: list[pd.DataFrame] = []
        for path in sorted(self.inputs_dir.glob("spatial_*.parquet")):
            frame = pd.read_parquet(path)
            frame = frame[
                frame["fold_type"].eq("spatial") & frame["support_role"].eq("primary")
            ].copy()
            if frame.empty:
                continue
            for (_, _), group in frame.groupby(["city_name", "offense"], sort=False):
                group = group.copy()
                truth = pd.to_numeric(group["truth_count"], errors="coerce").fillna(0.0).to_numpy(float)
                point = pd.to_numeric(group["prediction_ours"], errors="coerce").fillna(0.0).to_numpy(float)
                truth_total = float(truth.sum())
                point_total = float(point.sum())
                if truth_total <= 0.0 or point_total <= 0.0:
                    continue
                truth_share = truth / truth_total
                for level in CALIBRATION_LEVELS:
                    lower = pd.to_numeric(group[f"interval_{level}_lower"], errors="coerce").to_numpy(float)
                    upper = pd.to_numeric(group[f"interval_{level}_upper"], errors="coerce").to_numpy(float)
                    low_share = np.minimum(lower, upper) / point_total
                    high_share = np.maximum(lower, upper) / point_total
                    present = np.isfinite(low_share) & np.isfinite(high_share)
                    group[f"covered_{level}"] = np.where(
                        present,
                        (truth_share >= low_share) & (truth_share <= high_share),
                        np.nan,
                    )
                    group[f"requirement_{level}"] = _log_width_requirements(
                        truth, lower, upper, point
                    )
                group["city_incidents"] = truth_total
                prepared_parts.append(group)
        columns = [
            "offense", "uncertainty_support_class", "nominal", "empirical_coverage",
            "coverage_ci95_lower", "coverage_ci95_upper", "log_width_factor",
            "factor_ci95_lower", "factor_ci95_upper", "n_cities", "n_cells", "n_incidents",
        ]
        if not prepared_parts:
            return pd.DataFrame(columns=columns)
        prepared = pd.concat(prepared_parts, ignore_index=True)
        rows: list[dict[str, object]] = []
        for offense in OFFENSES:
            offense_rows = prepared[prepared["offense"].eq(offense)]
            for support_class in ("model", "direct", "benchmark"):
                selected = offense_rows[
                    offense_rows["uncertainty_support_class"].astype("string").eq(support_class)
                ].copy()
                for level in CALIBRATION_LEVELS:
                    level_selected = selected.dropna(
                        subset=[f"covered_{level}", f"requirement_{level}"]
                    ).copy()
                    if level_selected.empty:
                        rows.append(
                            {
                                "offense": offense,
                                "uncertainty_support_class": support_class,
                                "nominal": level / 100.0,
                                "empirical_coverage": float("nan"),
                                "coverage_ci95_lower": float("nan"),
                                "coverage_ci95_upper": float("nan"),
                                "log_width_factor": float("nan"),
                                "factor_ci95_lower": float("nan"),
                                "factor_ci95_upper": float("nan"),
                                "n_cities": 0,
                                "n_cells": 0,
                                "n_incidents": 0.0,
                            }
                        )
                        continue
                    city_groups = {
                        str(city): group.copy()
                        for city, group in level_selected.groupby("city_name", sort=True)
                    }
                    cities = sorted(city_groups)
                    city_coverage = np.array(
                        [float(city_groups[city][f"covered_{level}"].mean()) for city in cities]
                    )
                    values = np.concatenate(
                        [city_groups[city][f"requirement_{level}"].to_numpy(float) for city in cities]
                    )
                    weights = np.concatenate(
                        [np.full(len(city_groups[city]), 1.0 / len(city_groups[city])) for city in cities]
                    )
                    point_coverage = float(city_coverage.mean())
                    point_factor = _weighted_quantile(values, weights, level / 100.0)
                    coverage_draws = np.empty(self.config.bootstrap_iterations, dtype=float)
                    factor_draws = np.empty(self.config.bootstrap_iterations, dtype=float)
                    rng = np.random.default_rng(
                        self.config.bootstrap_seed
                        + sum(ord(char) for char in f"calibration|{offense}|{support_class}|{level}")
                    )
                    for draw_index in range(self.config.bootstrap_iterations):
                        sampled = rng.integers(0, len(cities), size=len(cities))
                        coverage_draws[draw_index] = float(city_coverage[sampled].mean())
                        sampled_frames = [city_groups[cities[index]] for index in sampled]
                        draw_values = np.concatenate(
                            [part[f"requirement_{level}"].to_numpy(float) for part in sampled_frames]
                        )
                        draw_weights = np.concatenate(
                            [np.full(len(part), 1.0 / len(part)) for part in sampled_frames]
                        )
                        factor_draws[draw_index] = _weighted_quantile(
                            draw_values, draw_weights, level / 100.0
                        )
                    coverage_ci = np.quantile(coverage_draws, [0.025, 0.975])
                    factor_ci = np.array(
                        [
                            _weighted_quantile(
                                factor_draws,
                                np.ones(len(factor_draws), dtype=float),
                                quantile,
                            )
                            for quantile in (0.025, 0.975)
                        ]
                    )
                    rows.append(
                        {
                            "offense": offense,
                            "uncertainty_support_class": support_class,
                            "nominal": level / 100.0,
                            "empirical_coverage": point_coverage,
                            "coverage_ci95_lower": float(coverage_ci[0]),
                            "coverage_ci95_upper": float(coverage_ci[1]),
                            "log_width_factor": point_factor,
                            "factor_ci95_lower": float(factor_ci[0]),
                            "factor_ci95_upper": float(factor_ci[1]),
                            "n_cities": len(cities),
                            "n_cells": int(len(level_selected)),
                            "n_incidents": float(
                                level_selected[["city_name", "city_incidents"]]
                                .drop_duplicates()["city_incidents"]
                                .sum()
                            ),
                        }
                    )
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = [
            "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *body])

    @staticmethod
    def _interval_text(row: pd.Series) -> str:
        if pd.isna(row["estimate"]):
            return ""
        if pd.isna(row["ci95_lower"]) or pd.isna(row["ci95_upper"]):
            return f"{row['estimate']:.4f} [NA]"
        return f"{row['estimate']:.4f} [{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}]"

    def _difference_table(self, differences: pd.DataFrame) -> str:
        if differences.empty:
            return "_(none)_"
        frame = differences.copy()
        frame["value"] = frame.apply(self._interval_text, axis=1)
        table = frame.pivot_table(
            index=["fold_type", "offense", "arm", "reference_arm"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        cities = frame[frame["metric"].eq("tvd")].set_index(
            ["fold_type", "offense", "arm", "reference_arm"]
        )["n_cities"]
        table = table.reindex(columns=[m for m in PAIRED_METRICS if m in table.columns])
        table["n_cities"] = cities.reindex(table.index)
        return self._markdown_table(table.reset_index().rename_axis(None, axis=1))

    def _markdown(
        self,
        arms: pd.DataFrame,
        differences: pd.DataFrame,
        calibration: pd.DataFrame,
        secondary: pd.DataFrame,
    ) -> str:
        variant = self.config.variant or "baseline"
        sections = [
            "# Gold evaluation results",
            "",
            f"Run `{self.config.run_id}`. Variant `{variant}`"
            + (f": `{' '.join(self.config.variant_args)}`" if self.config.variant_args else "")
            + ".",
            "",
            "| Fold | Label |",
            "| --- | --- |",
            *[f"| {fold} | {label} |" for fold, label in FOLD_LABELS.items()],
            "",
            (
                "Every arm and difference in a fold, offense and table block uses the same cities "
                "(the intersection where every compared arm has a TVD) and the same "
                f"{self.config.bootstrap_iterations:,} city-cluster bootstrap draws (seed "
                f"{self.config.bootstrap_seed}). Differences are arm minus reference_arm. TVD: lower "
                "is better. log_score (mean log predicted share per incident, 1% uniform floor), "
                "top_decile_capture, w1_spatial_skill, spearman: higher is better."
            ),
            "",
        ]
        primary_differences = differences[differences["family"].eq("primary")]
        candidate = primary_differences[
            primary_differences["reference_arm"].astype(str).str.startswith(BASELINE_ARM_PREFIX)
        ]
        if not candidate.empty:
            sections += [
                "## Candidate minus baseline",
                "",
                self._difference_table(candidate),
                "",
            ]
        ours = primary_differences[
            primary_differences["arm"].eq("ours")
            & ~primary_differences["reference_arm"].astype(str).str.startswith(BASELINE_ARM_PREFIX)
        ]
        sections += ["## Ours minus each baseline arm", "", self._difference_table(ours), ""]
        posthoc = primary_differences[
            ~primary_differences["arm"].isin(["ours", "ours_tract"])
        ]
        if not posthoc.empty:
            sections += ["## Post-hoc arms minus each arm", "", self._difference_table(posthoc), ""]
        ags = differences[
            differences["family"].eq("ags_tract") & differences["arm"].eq("ours_tract")
        ]
        sections += [
            "## Ours tract minus AGS 2022A (ags_uses_city_feed=false)",
            "",
            "`ours_tract` is block-group expected counts summed to the tract universe used for AGS.",
            "",
            self._difference_table(ags),
            "",
        ]
        display = arms.copy()
        if not display.empty:
            display["value"] = display.apply(self._interval_text, axis=1)
            display = display.pivot_table(
                index=["fold_type", "offense", "family", "arm"],
                columns="metric",
                values="value",
                aggfunc="first",
            )
            display = display.reindex(columns=[m for m in PAIRED_METRICS if m in display.columns])
            display = display.reset_index().rename_axis(None, axis=1)
        sections += ["## Matched-set arm estimates", "", self._markdown_table(display), ""]
        sections += [
            "## Secondary: declared E4 spill cells",
            "",
            "Not in any table above. One row per cell and arm in `secondary_cells.csv`.",
            "",
        ]
        if not secondary.empty:
            spill = secondary[secondary["arm"].isin(["ours", "ags_2022a_tract"])][
                ["city_name", "jurisdiction_id", "offense", "arm", "n_incidents", "tvd"]
            ].copy()
            spill["tvd"] = spill["tvd"].map(lambda value: f"{float(value):.4f}")
            sections += [self._markdown_table(spill), ""]
        sections += [
            "## Secondary: unmatched per-arm results",
            "",
            (
                "`results_unmatched.csv`: each arm over every city where that arm alone is "
                "defined, split by `ags_uses_city_feed`, bootstrapped separately. Not paired; not "
                "for comparisons."
            ),
            "",
        ]
        calibration_display = calibration.copy()
        for column in (
            "nominal", "empirical_coverage", "coverage_ci95_lower", "coverage_ci95_upper",
            "log_width_factor", "factor_ci95_lower", "factor_ci95_upper",
        ):
            calibration_display[column] = calibration_display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
            )
        sections += [
            "## Interval calibration",
            "",
            (
                "Factors multiply the existing lower and upper half-widths in log-share space. "
                "Zero truth or point counts use an additive half-incident continuity correction. "
                "Reported only. Confidence intervals resample cities."
            ),
            "",
            self._markdown_table(calibration_display),
            "",
        ]
        return "\n".join(sections)

    def run_posthoc(self, arm: str) -> pd.DataFrame:
        """Score a post-hoc arm from a completed run's retained per-cell inputs.

        Nothing is rebuilt and nothing in the run is overwritten: the arm's cells, weights and
        paired tables go to `<run>/posthoc/<arm>/`.
        """
        if arm not in POSTHOC_ARMS:
            raise ValueError(f"Unknown post-hoc arm {arm!r}; known: {sorted(POSTHOC_ARMS)}")
        manifest = json.loads((self.run_dir / "run_manifest.json").read_text())
        if "completed_at_utc" not in manifest:
            raise RuntimeError(f"Run {self.config.run_id!r} is not complete")
        feed_folds = [
            record
            for record in manifest.get("folds", [])
            if str(record.get("fold_type", "")) != "model"
        ]
        stale = [
            str(record["fold_id"])
            for record in feed_folds
            if int(record.get("scoring_schema_version", 0)) < 5
        ]
        if stale:
            raise RuntimeError(
                f"Folds {stale} retained inputs without support coordinates (scoring schema < 5)"
            )
        inputs = []
        for record in feed_folds:
            path = Path(str(record["input_path"]))
            if _sha256(path) != record.get("input_sha256"):
                raise RuntimeError(f"Retained input {path} changed since the run")
            inputs.append(pd.read_parquet(path))
        retained = pd.concat(inputs, ignore_index=True)
        retained = retained[retained["support_role"].eq("primary")]
        predictions, weights = POSTHOC_ARMS[arm](retained)
        run_cells = pd.read_csv(self.run_dir / "cell_scores.csv")
        cells = _score_posthoc_arm(predictions, arm=arm, run_cells=run_cells)
        out_dir = self.run_dir / "posthoc" / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        cells.to_csv(out_dir / "cell_scores.csv", index=False)
        weights.to_csv(out_dir / "weights.csv", index=False)
        outputs = self.write_results(out_dir, pd.concat([run_cells, cells], ignore_index=True))
        _json_write(
            out_dir / "manifest.json",
            {
                "run_id": self.config.run_id,
                "arm": arm,
                "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "inputs": {
                    str(record["input_path"]): record.get("input_sha256") for record in feed_folds
                },
                "run_cell_scores_sha256": _sha256(self.run_dir / "cell_scores.csv"),
                "cell_scores_sha256": _sha256(out_dir / "cell_scores.csv"),
                "weights_sha256": _sha256(out_dir / "weights.csv"),
                **{f"{name}_sha256": _sha256(path) for name, path in outputs.items()},
            },
        )
        return pd.read_csv(outputs["results"])
