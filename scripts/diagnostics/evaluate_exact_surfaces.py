"""Score named shipped surfaces against fixed-footprint local incident truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.exact_surface_evaluation import (
    OFFENSES,
    PREDICTION_EPSILON,
    SurfaceArm,
    aggregate_to_support,
    bootstrap_summary,
    footprint_block_groups,
    fully_contained_tracts,
    score_distribution,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arms(path: Path, year: int) -> list[SurfaceArm]:
    payload = json.loads(path.read_text())
    rows = payload["arms"] if isinstance(payload, dict) else payload
    arms = [
        SurfaceArm(
            name=str(row["name"]),
            block_group_path=Path(row["block_group_path"]),
            tract_path=Path(row["tract_path"]),
            manifest_path=Path(row["manifest_path"]),
            audit_path=Path(row["audit_path"]) if row.get("audit_path") else None,
        )
        for row in rows
    ]
    if not arms or len({arm.name for arm in arms}) != len(arms):
        raise ValueError("Arm names must be present and unique")
    for arm in arms:
        for required in (arm.block_group_path, arm.tract_path, arm.manifest_path):
            if not required.exists():
                raise FileNotFoundError(required)
        if arm.audit_path is None or not arm.audit_path.exists():
            raise FileNotFoundError(f"{arm.name} requires an allocation audit path")
        manifest = json.loads(arm.manifest_path.read_text())
        if int(manifest.get("year", -1)) != int(year):
            raise ValueError(f"{arm.name} manifest year does not match {year}")
    return arms


def _load_bg(arm: SurfaceArm, year: int) -> pd.DataFrame:
    columns = ["block_group_geoid", "tract_id", "eb_jurisdiction_id", f"population_{year}"]
    columns += [f"expected_count_{offense}" for offense in OFFENSES]
    columns += [f"primary_denominator_{offense}" for offense in OFFENSES]
    frame = pd.read_parquet(arm.block_group_path, columns=columns)
    frame["block_group_geoid"] = frame["block_group_geoid"].astype("string").str.zfill(12)
    return frame


def _load_tract(arm: SurfaceArm, year: int) -> pd.DataFrame:
    columns = ["tract_id", f"population_{year}"]
    columns += [f"expected_count_{offense}" for offense in OFFENSES]
    columns += [f"primary_denominator_{offense}" for offense in OFFENSES]
    frame = pd.read_parquet(arm.tract_path, columns=columns)
    frame["tract_id"] = frame["tract_id"].astype("string").str.zfill(11)
    return frame


def _rare_support_identity(arm: SurfaceArm, bg: pd.DataFrame) -> dict[str, float]:
    columns = ["tract_id", "expected_count_murder", "expected_count_rape"]
    tract = pd.read_parquet(arm.tract_path, columns=columns).set_index("tract_id")
    grouped = bg.groupby("tract_id", sort=False)[columns[1:]].sum()
    joined = grouped.join(tract, how="outer", lsuffix="__bg", rsuffix="__tract").fillna(0.0)
    return {
        offense: float(
            (joined[f"expected_count_{offense}__bg"] - joined[f"expected_count_{offense}__tract"])
            .abs()
            .max()
        )
        for offense in ("murder", "rape")
    }


def _reuse_flags(
    jurisdiction_id: str,
    year: int,
    offense: str,
    role: str,
    *,
    mixture_selection_reuse: bool = False,
    exposure_selection_reuse: bool = False,
) -> dict[str, object]:
    validation_only = role == "validation_holdout_only"
    cincinnati_2024 = jurisdiction_id == "39:municipal:place:3915000" and year == 2024
    montgomery_burglary = (
        jurisdiction_id == "24:county:24031" and year == 2025 and offense == "burglary"
    )
    return {
        "truth_used_in_residual_fit": role in {"residual_training_only", "direct_posterior_live"},
        "truth_used_in_direct_posterior": role == "direct_posterior_live",
        "truth_used_in_mixture_selection": mixture_selection_reuse,
        "truth_used_in_exposure_selection": exposure_selection_reuse,
        "truth_used_in_tau_selection": bool(cincinnati_2024 or montgomery_burglary),
        "truth_used_in_rare_selection": False,
        "jurisdiction_used_in_control_or_prior_fit": ":municipal:" in jurisdiction_id,
        "temporal_holdout": False,
        "evaluation_regime": (
            "selection_reuse" if cincinnati_2024 or montgomery_burglary or mixture_selection_reuse or exposure_selection_reuse
            else "source_disjoint_spatial" if validation_only
            else "fit_or_direct_reuse"
        ),
    }


def _summaries(cells: pd.DataFrame, iterations: int) -> list[dict[str, object]]:
    metrics = ["tvd", "cross_entropy", "top_decile_capture", "spearman"]
    rows: list[dict[str, object]] = []
    for (surface, offense, regime), group in cells.groupby(
        ["surface", "offense", "evaluation_regime"], dropna=False, sort=True
    ):
        for metric in metrics:
            clean = group.dropna(subset=[metric])
            if clean.empty:
                continue
            base = {
                "surface": surface,
                "offense": offense,
                "evaluation_regime": regime,
                "metric": metric,
                "cells": int(len(clean)),
                "equal_cell_mean": float(clean[metric].mean()),
                "equal_city_mean": float(
                    clean.groupby("jurisdiction_id", sort=False)[metric].mean().mean()
                ),
                "incident_weighted_mean": float(np.average(clean[metric], weights=clean["incidents"])),
            }
            for cluster in ("jurisdiction_id", "source"):
                stats = bootstrap_summary(clean, metric=metric, cluster=cluster, iterations=iterations)
                for key, value in stats.items():
                    base[f"{cluster}_{key}"] = value
            rows.append(base)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--arms-json", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--role-inventory", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--jurisdiction", action="append", default=[])
    parser.add_argument("--truth-year", action="append", type=int, default=[])
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--exposure-selection-cells", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    arms = _load_arms(args.arms_json, args.year)
    bg = {arm.name: _load_bg(arm, args.year) for arm in arms}
    tract = {arm.name: _load_tract(arm, args.year) for arm in arms}
    rare_support_identity = {
        arm.name: _rare_support_identity(arm, bg[arm.name]) for arm in arms
    }
    if any(delta > 1e-6 for values in rare_support_identity.values() for delta in values.values()):
        raise ValueError(f"Rare-offense BG-to-tract identity failed: {rare_support_identity}")
    frozen = bg[arms[0].name]
    truth = pd.read_parquet(args.truth).copy()
    source_column = "validation_source_name" if "validation_source_name" in truth else "source"
    if source_column != "source":
        truth["source"] = truth[source_column]
        source_column = "source"
    if args.registry is not None:
        registry = pd.read_csv(args.registry, dtype="string")
        keys = ["source", "jurisdiction_id", "offense"]
        truth = truth.merge(registry[keys].drop_duplicates(), on=keys, how="inner")
    truth["block_group_geoid"] = truth["block_group_geoid"].astype("string").str.zfill(12)
    if args.jurisdiction:
        truth = truth[truth["jurisdiction_id"].astype(str).isin(args.jurisdiction)]
    if args.truth_year:
        truth = truth[pd.to_numeric(truth["year"], errors="coerce").isin(args.truth_year)]
    roles = pd.read_parquet(args.role_inventory)[["jurisdiction_id", "offense", "role"]].drop_duplicates()
    truth = truth.merge(roles, on=["jurisdiction_id", "offense"], how="left")

    results: list[dict[str, object]] = []
    mixture_selection_keys = set()
    if args.registry is not None:
        mixture_selection_keys = set(
            registry[["jurisdiction_id", "offense"]].itertuples(index=False, name=None)
        )
    exposure_selection_keys = set()
    if args.exposure_selection_cells is not None:
        exposure_cells = pd.read_csv(args.exposure_selection_cells, dtype="string")
        exposure_selection_keys = set(
            exposure_cells[["jurisdiction_id", "offense"]].drop_duplicates().itertuples(index=False, name=None)
        )
    group_keys = [source_column, "jurisdiction_id", "year", "offense"]
    for (source, jurisdiction_id, truth_year, offense), observed in truth.groupby(group_keys, sort=True):
        footprint = footprint_block_groups(frozen, str(jurisdiction_id))
        if len(footprint) == 0:
            continue
        base = pd.DataFrame({"block_group_geoid": footprint})
        frozen_by_bg = frozen.set_index("block_group_geoid")
        base["tract_id"] = base["block_group_geoid"].map(frozen_by_bg["tract_id"])
        truth_by_bg = observed.groupby("block_group_geoid")["incident_count"].sum()
        base["truth_count"] = base["block_group_geoid"].map(truth_by_bg).fillna(0.0)
        total_truth = float(observed["incident_count"].sum())
        retained = float(base["truth_count"].sum())
        base["population"] = base["block_group_geoid"].map(frozen_by_bg[f"population_{args.year}"])
        for arm in arms:
            indexed = bg[arm.name].set_index("block_group_geoid")
            missing = footprint.difference(indexed.index)
            if len(missing):
                raise ValueError(f"{arm.name} misses {len(missing)} cells in frozen footprint {jurisdiction_id}")
            base[f"score__{arm.name}"] = base["block_group_geoid"].map(indexed[f"expected_count_{offense}"])
            base[f"score__primary_exposure__{arm.name}"] = base["block_group_geoid"].map(
                indexed[f"primary_denominator_{offense}"]
            )
        base["score__uniform"] = 1.0
        base["score__population"] = base["population"]
        boundary_tracts_excluded = 0
        if offense in {"murder", "rape"}:
            full_tracts = fully_contained_tracts(frozen, footprint)
            boundary_tracts_excluded = int(base["tract_id"].nunique() - len(full_tracts))
            base = base[base["tract_id"].astype("string").isin(full_tracts)].copy()
        supported = aggregate_to_support(base, str(offense))
        if offense in {"murder", "rape"}:
            for arm in arms:
                indexed_tract = tract[arm.name].set_index("tract_id")
                missing_tracts = pd.Index(supported["support_id"]).difference(indexed_tract.index)
                if len(missing_tracts):
                    raise ValueError(
                        f"{arm.name} misses {len(missing_tracts)} fully-contained tracts in {jurisdiction_id}"
                    )
                supported[f"score__{arm.name}"] = supported["support_id"].map(
                    indexed_tract[f"expected_count_{offense}"]
                )
                supported[f"score__primary_exposure__{arm.name}"] = supported["support_id"].map(
                    indexed_tract[f"primary_denominator_{offense}"]
                )
            supported["population"] = supported["support_id"].map(
                tract[arms[0].name].set_index("tract_id")[f"population_{args.year}"]
            )
            supported["score__population"] = supported["population"]
            supported["score__uniform"] = 1.0
        retained = float(supported["truth_count"].sum())
        role = str(observed["role"].dropna().iloc[0]) if observed["role"].notna().any() else "unknown"
        reuse_key = (str(jurisdiction_id), str(offense))
        flags = _reuse_flags(
            str(jurisdiction_id), int(truth_year), str(offense), role,
            mixture_selection_reuse=reuse_key in mixture_selection_keys,
            exposure_selection_reuse=reuse_key in exposure_selection_keys,
        )
        for column in [name for name in supported if name.startswith("score__")]:
            score = score_distribution(
                supported["truth_count"], supported[column], support_ids=supported["support_id"],
                epsilon=PREDICTION_EPSILON,
            )
            results.append({
                "surface": column.removeprefix("score__"), "source": str(source),
                "jurisdiction_id": str(jurisdiction_id), "truth_year": int(truth_year),
                "surface_year": int(args.year), "offense": str(offense),
                "support": "tract" if offense in {"murder", "rape"} else "block_group",
                "truth_incidents_total": total_truth, "truth_incidents_retained": retained,
                "truth_retained_fraction": retained / total_truth if total_truth > 0 else float("nan"),
                "truth_unmatched_incidents": total_truth - retained, "role": role, **flags, **score,
                "boundary_tracts_excluded": boundary_tracts_excluded,
                "support_mismatch": False,
            })

    cells = pd.DataFrame(results)
    if cells.empty:
        raise ValueError("No evaluable truth cells")
    summaries = _summaries(cells, args.bootstrap_iterations)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out_dir / "cells.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    files = [args.arms_json, args.truth, args.role_inventory]
    files += [path for path in (args.registry, args.exposure_selection_cells) if path is not None]
    files += [
        path
        for arm in arms
        for path in (arm.block_group_path, arm.tract_path, arm.manifest_path, arm.audit_path)
        if path is not None
    ]
    manifest = {
        "version": "exact_surface_evaluation_v1", "surface_year": args.year,
        "temporal_holdout_claim": False, "fixed_footprint_source": arms[0].name,
        "prediction_epsilon_before_normalization": PREDICTION_EPSILON,
        "top_decile_tie_rule": "score_desc_support_id_asc_exact_k",
        "rare_support_identity_max_abs_delta": rare_support_identity,
        "inputs": {str(path.resolve()): _sha256(path) for path in files},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cells": len(cells), "summaries": len(summaries), "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
