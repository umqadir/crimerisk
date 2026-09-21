from __future__ import annotations

import hashlib
import json
import math
import os
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
from crimerisk.crime import OFFENSES_7
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
TOP_FRACTION = 0.10
BUILD_TIMEOUT_REDUCTION_SECONDS = 40 * 60
SCORING_SCHEMA_VERSION = 4
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


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    fold_type: str
    exclude_city_key: str | None = None
    feed_year_end: int | None = None


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


def _reuse_flags(fold_type: str, offense: str, *, murder_k_seen: bool = False) -> str:
    model = fold_type == "model"
    values = {
        "mixture_weights_v3": "seen" if model else "not_seen",
        "soft_shrinkage": "seen" if model else "not_seen",
        "rape_triple": "seen" if model and offense == "rape" else "not_seen",
        "exposure_ensemble_weights": "seen" if model else "not_seen",
        "murder_K": "seen" if offense == "murder" and murder_k_seen else "not_seen",
        "tau": "seen" if offense == "burglary" and not model else "not_seen",
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
    truth = truth[truth["year"].between(2018, 2024)].copy()
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

    def matrix(self, support_ids: pd.Series, *, bg_population: pd.DataFrame | None = None) -> np.ndarray:
        ids = tuple(support_ids.astype(str))
        cached = self._matrices.get(ids)
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
        matrix = _local_distance_matrix(
            coordinates["lon"].to_numpy(float), coordinates["lat"].to_numpy(float)
        )
        self._matrices[ids] = matrix
        return matrix


class GoldEvaluation:
    def __init__(self, *, paths: RepoPaths, config: GoldEvaluationConfig):
        self.paths = paths
        self.config = config
        self.run_dir = paths.state_dir / "eval" / config.run_id
        self.work_dir = self.run_dir / "work"
        self.scored_dir = self.run_dir / "scored"
        self.inputs_dir = self.run_dir / "inputs"
        truth_snapshot = self.run_dir / "inputs" / "city_incident_share_surface_2018_2024.parquet"
        self.truth, self.city_keys = _active_truth(
            paths,
            truth_path=truth_snapshot if truth_snapshot.exists() else None,
        )
        city_identity = self.truth[["city_key", "jurisdiction_id"]].drop_duplicates()
        self.ags_feed_jurisdictions = set(
            city_identity.loc[
                city_identity["city_key"].isin(AGS_CITY_KEYS), "jurisdiction_id"
            ].astype(str)
        )
        self._fold_input_parts: list[pd.DataFrame] = []
        self.geometry = GeometryCache(paths.data_dir / "tiger_bg" / "parsed" / "bg_centroids.parquet")
        murder_selection_path = (
            paths.state_dir / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet"
        )
        self.murder_k_jurisdictions: set[str] = set()
        if murder_selection_path.exists():
            selected = pd.read_parquet(
                murder_selection_path,
                columns=["jurisdiction_id", "offense", "incident_count"],
            )
            selected = selected[selected["offense"].eq("murder")]
            totals = selected.groupby("jurisdiction_id")["incident_count"].sum()
            self.murder_k_jurisdictions = set(totals[totals.gt(0.0)].index.astype(str))
            registry = pd.read_csv(
                paths.repo_root
                / "analysis_scratch/final_phase/corpus_expansion/benchmark_registry_v1.csv"
            )
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
                    feed_year_end=2024,
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
            specs.append(FoldSpec("model_fold0", "model", feed_year_end=2024))
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

    def build_command(self, spec: FoldSpec) -> list[str]:
        fold_dir = self.work_dir / spec.fold_id
        command = [
            "uv", "run", "python", "main.py", "build-outputs",
            "--year", str(self.config.year),
            "--control-surface", "smoothed",
            "--enable-mixture-allocation",
            "--enable-exposure-ensemble",
            "--enable-count-first-composites",
            "--enable-special-use-taxonomy",
            "--enable-soft-shrinkage",
            "--enable-unlocated-mass",
            "--enable-imputation-v2",
            "--enable-uncertainty-layer",
            "--residual-transfer-tau", "burglary=0",
            "--bg-ags-core-out", str(fold_dir / f"crimerisk_block_group_{self.config.year}_ags_core.parquet"),
            "--tract-ags-core-out", str(fold_dir / f"crimerisk_tract_{self.config.year}_ags_core.parquet"),
            "--build-manifest-out", str(fold_dir / "manifest.json"),
            "--feed-inputs-dir", str(self.fold_inputs_dir(spec)),
        ]
        if spec.exclude_city_key is not None:
            command.extend(["--exclude-feed-city", spec.exclude_city_key])
        if spec.feed_year_end is not None:
            command.extend(["--feed-year-end", str(spec.feed_year_end)])
        return command

    def dry_run_payload(self) -> list[dict[str, object]]:
        return [
            {
                **asdict(spec),
                "command": self.build_command(spec),
            }
            for spec in self.fold_specs()
        ]

    def run(self) -> pd.DataFrame:
        specs = self.fold_specs()
        if self.config.dry_run:
            print(json.dumps(self.dry_run_payload(), indent=2))
            return pd.DataFrame()
        ags_path = ensure_ags_tract_cache(self.paths)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scored_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        truth_snapshot = self.run_dir / "inputs" / "city_incident_share_surface_2018_2024.parquet"
        manifest_path = self.run_dir / "run_manifest.json"
        incomplete_run = False
        if manifest_path.exists():
            incomplete_run = "completed_at_utc" not in json.loads(manifest_path.read_text())
        if not truth_snapshot.exists():
            truth_snapshot.parent.mkdir(parents=True, exist_ok=True)
            self.truth.to_parquet(truth_snapshot, index=False)
        disk_before = shutil.disk_usage(self.paths.repo_root).free
        started = time.monotonic()
        run_manifest: dict[str, object] = {
            "run_id": self.config.run_id,
            "started_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "config": asdict(self.config),
            "pot_version": ot.__version__,
            "scoring_schema_version": SCORING_SCHEMA_VERSION,
            "folds": [],
            "inputs": {
                "city_truth": {
                    "path": str(truth_snapshot),
                    "sha256": _sha256(truth_snapshot),
                },
                "ags_2022a_tract": {"path": str(ags_path), "sha256": _sha256(ags_path)},
            },
            "disk_free_bytes_before": int(disk_before),
        }
        resumed_records: dict[str, dict[str, object]] = {}
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if "completed_at_utc" not in prior:
                prior_config = dict(prior.get("config", {}))
                current_config = json.loads(json.dumps(asdict(self.config)))
                if prior_config != current_config:
                    raise RuntimeError(
                        f"Incomplete run {self.config.run_id!r} has a different configuration"
                    )
                run_manifest = prior
                run_manifest["inputs"] = {
                    "city_truth": {
                        "path": str(truth_snapshot),
                        "sha256": _sha256(truth_snapshot),
                    },
                    "ags_2022a_tract": {
                        "path": str(ags_path),
                        "sha256": _sha256(ags_path),
                    },
                }
                for record in prior.get("folds", []):
                    if isinstance(record, dict) and isinstance(record.get("fold_id"), str):
                        resumed_records[str(record["fold_id"])] = record
                run_manifest["folds"] = []
                run_manifest["scoring_schema_version"] = SCORING_SCHEMA_VERSION
        protected_before = self._fingerprint(self.protected_state_paths())
        observed_before = self._fingerprint(self.observed_state_paths())
        run_manifest["protected_state_paths"] = sorted(protected_before)
        score_parts: list[pd.DataFrame] = []
        spatial_build_elapsed: list[float] = []
        allowed_spatial = {spec.fold_id for spec in specs if spec.fold_type == "spatial"}
        for spec in specs:
            if spec.fold_type == "spatial" and spec.fold_id not in allowed_spatial:
                continue
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
            if spec.fold_type == "spatial":
                spatial_build_elapsed.append(float(fold_record["build_wall_seconds"]))
                if spatial_build_elapsed[-1] > BUILD_TIMEOUT_REDUCTION_SECONDS:
                    largest_six = [
                        row.fold_id
                        for row in [s for s in specs if s.fold_type == "spatial"][:6]
                    ]
                    allowed_spatial = set(largest_six)
                    run_manifest["spatial_reduction"] = {
                        "trigger_fold": spec.fold_id,
                        "trigger_build_wall_seconds": spatial_build_elapsed[-1],
                        "retained_fold_ids": largest_six,
                    }
        run_manifest["protected_state"] = self._assert_state_unchanged(
            protected_before, observed_before
        )
        if not score_parts:
            raise RuntimeError("No evaluation folds produced scores")
        cells = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
        cells.to_csv(self.run_dir / "cell_scores.csv", index=False)
        results = self._summarize(cells)
        results.to_csv(self.run_dir / "results.csv", index=False)
        calibration = self._calibration_results()
        calibration.to_csv(self.run_dir / "calibration.csv", index=False)
        (self.run_dir / "results.md").write_text(self._markdown(results, calibration))
        attempt_wall_seconds = float(time.monotonic() - started)
        fold_wall_seconds = float(
            sum(float(record["wall_seconds"]) for record in run_manifest["folds"])
        )
        run_manifest.update(
            completed_at_utc=datetime.now(tz=timezone.utc).isoformat(),
            wall_seconds=fold_wall_seconds,
            attempt_wall_seconds=attempt_wall_seconds,
            disk_free_bytes_after=int(shutil.disk_usage(self.paths.repo_root).free),
            results_sha256=_sha256(self.run_dir / "results.csv"),
            cell_scores_sha256=_sha256(self.run_dir / "cell_scores.csv"),
            calibration_sha256=_sha256(self.run_dir / "calibration.csv"),
        )
        run_manifest["disk_used_bytes"] = int(
            run_manifest["disk_free_bytes_before"] - run_manifest["disk_free_bytes_after"]
        )
        _json_write(self.run_dir / "run_manifest.json", run_manifest)
        return results

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

    def _run_fold(self, spec: FoldSpec, *, ags_path: Path) -> tuple[dict[str, object], pd.DataFrame]:
        fold_dir = self.work_dir / spec.fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.paths.repo_root).free < 20 * 1024**3:
            raise RuntimeError("Gold evaluation stopped before build: less than 20 GiB free")
        command = self.build_command(spec)
        print(f"starting {spec.fold_id}: {' '.join(command)}", flush=True)
        started = time.monotonic()
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
        build_elapsed = time.monotonic() - started
        (fold_dir / "build.log").write_text(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Build failed for {spec.fold_id} with exit {completed.returncode}; "
                f"see {fold_dir / 'build.log'}"
            )
        bg_path = fold_dir / f"crimerisk_block_group_{self.config.year}_ags_core.parquet"
        tract_path = fold_dir / f"crimerisk_tract_{self.config.year}_ags_core.parquet"
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
                spec, bg_path=bg_path, tract_path=tract_path, ags_path=ags_path
            )
        score_path = self.scored_dir / f"{spec.fold_id}.csv"
        scores.to_csv(score_path, index=False)
        if not self._fold_input_parts:
            raise RuntimeError(f"Fold {spec.fold_id} retained no per-cell scoring input")
        fold_input = pd.concat(self._fold_input_parts, ignore_index=True)
        input_path = self.inputs_dir / f"{spec.fold_id}.parquet"
        fold_input.to_parquet(input_path, index=False)
        manifest_path = fold_dir / "manifest.json"
        record = {
            **asdict(spec),
            "command": command,
            "build_wall_seconds": float(build_elapsed),
            "returncode": int(completed.returncode),
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
        intervals: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        support_class: pd.Series | np.ndarray | None = None,
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
        ags_path: Path,
        past_truth: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        bg, tract = self._load_candidate(bg_path, tract_path)
        ags = pd.read_parquet(ags_path, columns=["ID", "POP_C", *AGS_COLUMN.values()]).rename(
            columns={"ID": "tract_id"}
        )
        ags["tract_id"] = ags["tract_id"].astype("string").str.zfill(11)
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
                    intervals=interval_arrays,
                    support_class=base[f"uncertainty_support_class_{offense}"],
                )

                tract_base = support_frames["tract"].copy()
                tract_truth = observed.assign(_support=observed["block_group_geoid"].str[:11]).groupby(
                    "_support"
                )["incident_count"].sum()
                tract_base["truth_count"] = tract_base["support_id"].map(tract_truth).fillna(0.0)
                ags_rows = ags.reindex(tract_base["support_id"].astype(str))
                ags_prediction = (
                    pd.to_numeric(ags_rows[AGS_COLUMN[offense]], errors="coerce").fillna(0.0).to_numpy(float)
                    * pd.to_numeric(ags_rows["POP_C"], errors="coerce").fillna(0.0).to_numpy(float)
                )
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
                tract_null_w1 = _population_null_w1(
                    truth_count=tract_base["truth_count"].to_numpy(float),
                    population=tract_population,
                    distance_matrix=matrices["tract"],
                )
                for arm, prediction in {
                    "ours_tract": ours_tract_prediction,
                    "ags_2022a_tract": ags_prediction,
                }.items():
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
                    predictions={
                        "ours_tract": ours_tract_prediction,
                        "ags_2022a_tract": ags_prediction,
                    },
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
            "reuse_flags": _reuse_flags(
                "model" if spec.fold_type == "model" else spec.fold_type,
                offense,
                murder_k_seen=jurisdiction_id in self.murder_k_jurisdictions,
            ),
            **metrics,
        }

    def _score_model_candidate(
        self,
        spec: FoldSpec,
        *,
        bg_path: Path,
        tract_path: Path,
        ags_path: Path,
    ) -> pd.DataFrame:
        bg, tract = self._load_candidate(bg_path, tract_path)
        population_col = f"population_{self.config.year}"
        pure = (
            bg.groupby("tract_id")["eb_jurisdiction_id"]
            .agg(["nunique", "first"])
            .query("nunique == 1")["first"]
        )
        tract = tract.copy()
        tract["jurisdiction_id"] = tract["tract_id"].map(pure)
        truth_path = (
            self.paths.repo_root
            / "analysis_scratch/final_phase/corpus_expansion/widened_incident_surface.parquet"
        )
        registry_path = (
            self.paths.repo_root
            / "analysis_scratch/final_phase/corpus_expansion/benchmark_registry_v1.csv"
        )
        frozen_cells_path = (
            self.paths.repo_root
            / "analysis_scratch/final_phase/e4_benchmark/truth_test.csv"
        )
        truth = pd.read_parquet(truth_path).copy()
        truth["block_group_geoid"] = truth["block_group_geoid"].astype("string").str.zfill(12)
        truth["tract_id"] = truth["block_group_geoid"].str[:11]
        truth["tract_jurisdiction_id"] = truth["tract_id"].map(pure)
        truth = truth[truth["tract_jurisdiction_id"].eq(truth["jurisdiction_id"])].copy()
        registry = pd.read_csv(registry_path)
        frozen = pd.read_csv(
            frozen_cells_path,
            usecols=["variant", "source", "jurisdiction_id", "offense"],
        )
        frozen = frozen[frozen["variant"].eq("base")][
            ["source", "jurisdiction_id", "offense"]
        ].drop_duplicates()
        if len(frozen) != 190 or frozen["source"].nunique() != 20:
            raise ValueError(
                "Frozen E4 base-cell identity changed: "
                f"cells={len(frozen)}, sources={frozen['source'].nunique()}"
            )
        registry = registry.merge(
            frozen,
            on=["source", "jurisdiction_id", "offense"],
            how="inner",
            validate="one_to_one",
        )
        ags = pd.read_parquet(ags_path, columns=["ID", "POP_C", *AGS_COLUMN.values()]).rename(
            columns={"ID": "tract_id"}
        )
        ags["tract_id"] = ags["tract_id"].astype("string").str.zfill(11)
        merged = tract.merge(ags, on="tract_id", how="left").set_index("tract_id")
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
            if len(jurisdiction) < 20:
                continue
            jurisdiction["truth_count"] = jurisdiction.index.to_series().map(observed).fillna(0.0)
            kept_incidents = float(jurisdiction["truth_count"].sum())
            total_incidents = float(pooled_all.get(key, 0.0))
            if kept_incidents < 50.0 or total_incidents <= 0.0 or kept_incidents / total_incidents < 0.5:
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
                "ags_2022a_tract": (
                    pd.to_numeric(jurisdiction[AGS_COLUMN[key[2]]], errors="coerce").fillna(0.0)
                    * pd.to_numeric(jurisdiction["POP_C"], errors="coerce").fillna(0.0)
                ),
            }
            population_null_w1 = _population_null_w1(
                truth_count=jurisdiction["truth_count"].to_numpy(float),
                population=population.to_numpy(float),
                distance_matrix=distance,
            )
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
                intervals=self._interval_arrays(jurisdiction, key[2]),
                support_class=jurisdiction[f"uncertainty_support_class_{key[2]}"],
            )
        scored = pd.DataFrame(rows)
        if not scored.empty:
            evaluated = scored[scored["arm"].eq("ours")]
            if (
                len(evaluated) != 190
                or evaluated["jurisdiction_id"].nunique() != 40
                or evaluated["city_name"].nunique() != 20
            ):
                raise ValueError(
                    "E4 model-lane universe changed: "
                    f"cells={len(evaluated)}, sources={evaluated['city_name'].nunique()}, "
                    f"jurisdictions={evaluated['jurisdiction_id'].nunique()}"
                )
        return scored

    def _summarize(self, cells: pd.DataFrame) -> pd.DataFrame:
        cells = cells.copy()
        if "ags_uses_city_feed" not in cells.columns:
            cells["ags_uses_city_feed"] = False
        metric_columns = [
            "tvd", "spearman", "top_decile_capture", "w1_spatial_skill",
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

    def _markdown(self, results: pd.DataFrame, calibration: pd.DataFrame) -> str:
        metric_names = ["tvd", "spearman", "top_decile_capture", "w1_spatial_skill"]
        comparison = results[
            results["fold_type"].eq("spatial")
            & results["arm"].isin(["ours_tract", "ags_2022a_tract"])
            & results["metric"].isin(metric_names)
        ].copy()
        comparison["value"] = comparison.apply(
            lambda row: (
                f"{row['estimate']:.4f} [{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}]"
            ),
            axis=1,
        )
        comparison = comparison.pivot_table(
            index=["offense", "ags_uses_city_feed", "metric"],
            columns="arm",
            values="value",
            aggfunc="first",
        ).reset_index().rename_axis(None, axis=1)
        comparison = comparison.rename(
            columns={"ags_2022a_tract": "AGS 2022A tract", "ours_tract": "ours_tract"}
        )

        primary = results[
            ~results["arm"].isin(["ours_tract", "ags_2022a_tract"])
        ].copy()
        primary["estimate"] = primary["estimate"].map(lambda value: f"{float(value):.4f}")
        primary = primary[
            ["fold_type", "ags_uses_city_feed", "offense", "arm", "metric", "estimate", "n_cities", "n_cells", "n_incidents"]
        ]

        calibration_display = calibration.copy()
        for column in (
            "nominal", "empirical_coverage", "coverage_ci95_lower", "coverage_ci95_upper",
            "log_width_factor", "factor_ci95_lower", "factor_ci95_upper",
        ):
            calibration_display[column] = calibration_display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
            )

        return (
            "# Gold evaluation results\n\n"
            "All AGS comparisons use `ours_tract`: block-group expected counts summed to the exact "
            "tract universe used for AGS. `ours` remains the published-support arm.\n\n"
            "AGS uses city incident feeds for New York, Chicago, Boston, Philadelphia, Baltimore, "
            "Seattle, Austin, and Mesa. `ags_uses_city_feed=false` covers San Francisco, Washington, "
            "Denver, Minneapolis, and St. Louis in the spatial folds.\n\n"
            "The model fold is labelled as a corpus on which mixture weights v3, soft shrinkage, "
            "the rape triple, exposure ensemble weights, murder K, or tau were selected wherever "
            "the corresponding `reuse_flags` value is `seen`.\n\n"
            "## Ours tract versus AGS 2022A\n\n"
            + self._markdown_table(comparison)
            + "\n\n## Primary-support arms\n\n"
            + self._markdown_table(primary)
            + "\n\n## Interval calibration\n\n"
            "Factors multiply the existing lower and upper half-widths in log-share space. Zero "
            "truth or point counts use an additive half-incident continuity correction. Factors "
            "are reported only and are not applied to a surface. Confidence intervals resample "
            "cities. Direct and benchmark classes have no held-out spatial cells after city-feed "
            "exclusion and are reported as blank.\n\n"
            + self._markdown_table(calibration_display)
            + "\n"
        )
