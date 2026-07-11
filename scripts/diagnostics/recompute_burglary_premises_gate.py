from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.denominators import add_offense_denominators, _load_burglary_calibration_rows
from crimerisk.model_surface import build_bg_feature_frame
from crimerisk.paths import RepoPaths


RATE_PER_100K = 100000.0
DEFAULT_BOOTSTRAP_DRAWS = 500
DEFAULT_BOOTSTRAP_SEED = 20260705
DIRECT_GATE_BAND = [0.8, 1.3]


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if value is not None else False:
        return None
    return value


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if np.isfinite(parsed) else None


def _commercial_share(frame: pd.DataFrame) -> pd.Series:
    households = pd.to_numeric(frame["households_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
    commercial = pd.to_numeric(frame["commercial_premises_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return (commercial / (households + commercial).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _assign_quintiles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[frame["commercial_share"].notna()].copy()
    if len(out) < 5:
        out["quintile"] = pd.NA
        return out
    out["quintile"] = (
        pd.qcut(out["commercial_share"].rank(method="first"), 5, labels=False).astype("int64") + 1
    )
    return out


def _truth_quintiles(frame: pd.DataFrame, *, denominator_col: str) -> tuple[float, list[dict[str, Any]]]:
    base = frame[
        pd.to_numeric(frame[denominator_col], errors="coerce").fillna(0.0).gt(0.0)
        & pd.to_numeric(frame["incident_count"], errors="coerce").fillna(0.0).ge(0.0)
    ].copy()
    base["commercial_share"] = _commercial_share(base)
    base = _assign_quintiles(base)
    if base["quintile"].isna().any() or base.empty:
        return float("nan"), []

    total_count = float(pd.to_numeric(base["incident_count"], errors="coerce").fillna(0.0).sum())
    total_denominator = float(pd.to_numeric(base[denominator_col], errors="coerce").fillna(0.0).sum())
    national_rate = RATE_PER_100K * total_count / total_denominator if total_denominator > 0.0 else float("nan")
    rows: list[dict[str, Any]] = []
    for quintile in range(1, 6):
        part = base[base["quintile"].eq(quintile)].copy()
        count_sum = float(pd.to_numeric(part["incident_count"], errors="coerce").fillna(0.0).sum())
        denominator_sum = float(pd.to_numeric(part[denominator_col], errors="coerce").fillna(0.0).sum())
        rate = RATE_PER_100K * count_sum / denominator_sum if denominator_sum > 0.0 else float("nan")
        rows.append(
            {
                "quintile": int(quintile),
                "rows": int(len(part)),
                "commercial_share_mean": _to_float(part["commercial_share"].mean()),
                "incident_count": count_sum,
                "denominator_new_total": denominator_sum,
                "truth_index_mean_aggregate_rate": (
                    float(100.0 * rate / national_rate)
                    if np.isfinite(rate) and np.isfinite(national_rate) and national_rate > 0.0
                    else float("nan")
                ),
            }
        )
    q1 = rows[0]["truth_index_mean_aggregate_rate"]
    q5 = rows[-1]["truth_index_mean_aggregate_rate"]
    gradient = (
        float(q5) / float(q1)
        if np.isfinite(float(q5)) and np.isfinite(float(q1)) and float(q1) != 0.0
        else float("nan")
    )
    return gradient, rows


def _bootstrap_truth_gradient(
    eligible_rows: pd.DataFrame,
    *,
    denominator_col: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    city_ids = sorted(eligible_rows["jurisdiction_id"].dropna().astype(str).unique().tolist())
    by_city = {city_id: group.copy() for city_id, group in eligible_rows.groupby("jurisdiction_id", sort=False)}
    rng = np.random.default_rng(int(seed))
    gradients: list[float] = []
    for _draw in range(int(draws)):
        sampled = rng.choice(city_ids, size=len(city_ids), replace=True)
        draw = pd.concat([by_city[str(city_id)] for city_id in sampled], ignore_index=True, sort=False)
        gradient, _rows = _truth_quintiles(draw, denominator_col=denominator_col)
        if np.isfinite(gradient):
            gradients.append(float(gradient))
    values = np.asarray(gradients, dtype=float)
    return {
        "draws": int(draws),
        "valid_draws": int(len(values)),
        "seed": int(seed),
        "city_resample_unit": "eligible covered-city jurisdiction_id",
        "statistic": (
            "pooled covered-city truth gradient, aggregate burglary incident rate ratio q5/q1 "
            "after city bootstrap draw"
        ),
        "p2_5": float(np.percentile(values, 2.5)) if len(values) else None,
        "p50": float(np.percentile(values, 50.0)) if len(values) else None,
        "p97_5": float(np.percentile(values, 97.5)) if len(values) else None,
        "confidence_interval": (
            [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            if len(values)
            else None
        ),
    }


def _round_down_hundredth(value: float) -> float:
    return math.floor(float(value) * 100.0) / 100.0


def _new_denominator(frame: pd.DataFrame, calibration: dict[str, Any]) -> pd.Series:
    k_destination = float(calibration.get("k_destination_poi", calibration.get("k_commercial", 0.0)) or 0.0)
    k_retail = float(calibration.get("k_retail_jobs", 0.0) or 0.0)
    k_industrial = float(calibration.get("k_industrial_jobs", 0.0) or 0.0)
    households = pd.to_numeric(frame["households_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
    commercial = pd.to_numeric(frame["commercial_premises_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
    destination = (
        pd.to_numeric(frame["destination_poi_total"], errors="coerce")
        if "destination_poi_total" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype=float)
    ).fillna(commercial).clip(lower=0.0)
    retail = pd.to_numeric(frame["lodes_retail_jobs"], errors="coerce").fillna(0.0).clip(lower=0.0)
    industrial = pd.to_numeric(frame["lodes_industrial_jobs"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return households + k_destination * destination + k_retail * retail + k_industrial * industrial


def _surface_modeled_gradient(
    *,
    surface_path: Path,
    covariates: pd.DataFrame,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    columns = [
        "block_group_geoid",
        "households_total",
        "commercial_premises_total",
        "expected_count_burglary",
        "primary_index_publishable_burglary",
        "source_mode_burglary",
        "estimate_mode_burglary",
    ]
    available = pd.read_parquet(surface_path, engine="pyarrow").columns
    keep = [col for col in columns if col in available]
    surface = pd.read_parquet(surface_path, columns=keep).copy()
    surface["block_group_geoid"] = surface["block_group_geoid"].astype("string").str.zfill(12)
    cov = covariates[
        [
            "block_group_geoid",
            "destination_poi_total",
            "lodes_retail_jobs",
            "lodes_industrial_jobs",
        ]
    ].copy()
    merged = surface.merge(cov, on="block_group_geoid", how="left")
    for col in ["households_total", "commercial_premises_total", "expected_count_burglary"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["denominator_new"] = _new_denominator(merged, calibration)
    merged["commercial_share"] = _commercial_share(merged)
    if "primary_index_publishable_burglary" in merged.columns:
        publishable = merged["primary_index_publishable_burglary"].fillna(False).astype(bool)
    elif "estimate_mode_burglary" in merged.columns:
        publishable = merged["estimate_mode_burglary"].astype("string").eq("count_derived")
    else:
        publishable = pd.Series(True, index=merged.index)
    base = merged[
        publishable
        & merged["denominator_new"].gt(0.0)
        & merged["commercial_share"].notna()
    ].copy()
    base = _assign_quintiles(base)
    total_count = float(base["expected_count_burglary"].sum())
    total_denominator = float(base["denominator_new"].sum())
    national_rate = RATE_PER_100K * total_count / total_denominator if total_denominator > 0.0 else float("nan")
    base["index_new_denominator"] = np.where(
        np.isfinite(national_rate) & (national_rate > 0.0),
        100.0 * (RATE_PER_100K * base["expected_count_burglary"] / base["denominator_new"]) / national_rate,
        np.nan,
    )
    source_mode = base["source_mode_burglary"].astype("string") if "source_mode_burglary" in base.columns else ""
    modeled = base[source_mode.ne("direct_city_incident")].copy()

    def summarize(group: pd.DataFrame) -> dict[str, Any]:
        quintiles: list[dict[str, Any]] = []
        for quintile in range(1, 6):
            part = group[group["quintile"].eq(quintile)]
            quintiles.append(
                {
                    "quintile": int(quintile),
                    "rows": int(len(part)),
                    "commercial_share_mean": _to_float(part["commercial_share"].mean()),
                    "after_index_mean": _to_float(part["index_new_denominator"].mean()),
                    "after_index_median": _to_float(part["index_new_denominator"].median()),
                }
            )
        q1 = quintiles[0]["after_index_mean"]
        q5 = quintiles[-1]["after_index_mean"]
        return {
            "rows": int(len(group)),
            "after_q5_q1_mean": (
                float(q5) / float(q1)
                if q1 is not None and q5 is not None and float(q1) != 0.0
                else None
            ),
            "quintiles": quintiles,
        }

    return {
        "surface_path": str(surface_path.relative_to(REPO_ROOT)),
        "denominator": "new burglary premises denominator",
        "quintile_basis": "commercial_premises_total / (households_total + commercial_premises_total)",
        "all_published": summarize(base),
        "modeled_transfer": summarize(modeled),
    }


def build_artifact(*, draws: int, seed: int, out_path: Path) -> dict[str, Any]:
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    bg = add_offense_denominators(
        build_bg_feature_frame(paths=paths, year=2024),
        paths=paths,
        year=2024,
        apply_landscan_day_floor=False,
    )
    calibration = dict(bg.attrs.get("burglary_commercial_calibration", {}))
    covariates = bg[
        [
            "bg_id",
            "households_total",
            "commercial_premises_total",
            "destination_poi_total",
            "lodes_retail_jobs",
            "lodes_industrial_jobs",
            "burglary_premises_total",
        ]
    ].copy()
    covariates["block_group_geoid"] = covariates["bg_id"].astype("string").str.zfill(12)
    covariates["denominator_new"] = _new_denominator(covariates, calibration)

    direct, source_metadata = _load_burglary_calibration_rows(paths=paths, year=2024)
    truth = (
        direct.groupby(["jurisdiction_id", "city_name", "state_fips", "block_group_geoid"], dropna=False)[
            "incident_count"
        ]
        .sum()
        .reset_index()
    )
    truth["block_group_geoid"] = truth["block_group_geoid"].astype("string").str.zfill(12)
    truth = truth.merge(
        covariates.drop(columns=["bg_id"]),
        on="block_group_geoid",
        how="inner",
    )
    for col in [
        "incident_count",
        "households_total",
        "commercial_premises_total",
        "destination_poi_total",
        "lodes_retail_jobs",
        "lodes_industrial_jobs",
        "denominator_new",
    ]:
        truth[col] = pd.to_numeric(truth[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    eligible_table = (
        truth.groupby(["jurisdiction_id", "city_name", "state_fips"], dropna=False)
        .agg(
            incident_count=("incident_count", "sum"),
            block_groups=("block_group_geoid", "nunique"),
            household_total=("households_total", "sum"),
            commercial_premises_total=("commercial_premises_total", "sum"),
            destination_poi_total=("destination_poi_total", "sum"),
            lodes_retail_jobs=("lodes_retail_jobs", "sum"),
            lodes_industrial_jobs=("lodes_industrial_jobs", "sum"),
            denominator_new_total=("denominator_new", "sum"),
        )
        .reset_index()
    )
    eligible_table = eligible_table[
        pd.to_numeric(eligible_table["incident_count"], errors="coerce").ge(500.0)
        & pd.to_numeric(eligible_table["block_groups"], errors="coerce").ge(50.0)
    ].copy()
    eligible_ids = set(eligible_table["jurisdiction_id"].astype(str))
    eligible_rows = truth[truth["jurisdiction_id"].astype(str).isin(eligible_ids)].copy()
    pooled_gradient, pooled_quintiles = _truth_quintiles(eligible_rows, denominator_col="denominator_new")
    bootstrap = _bootstrap_truth_gradient(
        eligible_rows,
        denominator_col="denominator_new",
        draws=int(draws),
        seed=int(seed),
    )
    p97_5 = float(bootstrap["p97_5"])
    ceiling = _round_down_hundredth(p97_5)

    anti_goalpost_path = REPO_ROOT / "state" / "candidates" / "county-anchoring-v2" / "crimerisk_block_group_2024_ags_core.parquet"
    anti_goalpost = _surface_modeled_gradient(
        surface_path=anti_goalpost_path,
        covariates=covariates,
        calibration=calibration,
    )
    modeled_gradient = anti_goalpost["modeled_transfer"]["after_q5_q1_mean"]
    if modeled_gradient is None or float(modeled_gradient) <= ceiling:
        raise SystemExit(
            "Anti-goalpost failed: county-anchoring-v2 modeled-transfer gradient "
            f"{modeled_gradient} did not exceed new ceiling {ceiling}"
        )
    anti_goalpost["new_modeled_ceiling"] = ceiling
    anti_goalpost["assertion"] = "county-anchoring-v2 modeled-transfer gradient remains above the new ceiling"
    anti_goalpost["assertion_passed"] = True

    artifact = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "year": 2024,
        "purpose": "Pre-registered burglary modeled-transfer gradient ceiling under the v9 multi-term denominator.",
        "method": (
            "500-draw city bootstrap of pooled covered-city direct burglary truth gradient. "
            "City eligibility requires >=500 burglary incidents and >=50 block groups; rows are pooled across "
            "source years; quintiles keep the existing commercial-share axis "
            "commercial_premises_total / (households_total + commercial_premises_total). Only the truth rate "
            "denominator changes to the fitted multi-term burglary denominator."
        ),
        "truth_surface_path": source_metadata.get("calibration_path"),
        "truth_surface_metadata": source_metadata,
        "covariate_source": {
            "feature_frame": "crimerisk.model_surface.build_bg_feature_frame(paths, year=2024)",
            "lodes_jobs": "data/LODES/parsed/lodes_wac_block_groups.parquet CNS05/CNS06/CNS07/CNS08",
            "destination_poi": "commercial_premises_total / Overture consumer-destination POI total",
        },
        "calibration": calibration,
        "eligible_city_count": int(len(eligible_table)),
        "eligible_city_table": eligible_table.sort_values(["city_name", "jurisdiction_id"], kind="mergesort").to_dict(
            orient="records"
        ),
        "pooled_truth_gradient": float(pooled_gradient),
        "pooled_truth_gradient_rounded": round(float(pooled_gradient), 4),
        "pooled_truth_quintiles": pooled_quintiles,
        "bootstrap": bootstrap,
        "rounding_policy": "Conservative floor of p97.5 to the nearest 0.01 for modeled-transfer ceiling.",
        "shipped_modeled_transfer_ceiling": ceiling,
        "direct_city_gate_band": DIRECT_GATE_BAND,
        "direct_city_gate_band_decision": (
            "Direct band remains [0.8, 1.3]; the truth bootstrap moved the denominator-relative "
            "modeled ceiling, but no separate direct-band widening is supported here."
        ),
        "anti_goalpost": anti_goalpost,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_clean_json(artifact), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "burglary_gate_ceiling_derivation.json",
    )
    args = parser.parse_args()
    artifact = build_artifact(draws=int(args.draws), seed=int(args.seed), out_path=args.out)
    print(
        json.dumps(
            _clean_json(
                {
                    "pooled_truth_gradient": artifact["pooled_truth_gradient"],
                    "bootstrap_p97_5": artifact["bootstrap"]["p97_5"],
                    "shipped_modeled_transfer_ceiling": artifact["shipped_modeled_transfer_ceiling"],
                    "anti_goalpost_modeled_gradient": artifact["anti_goalpost"]["modeled_transfer"][
                        "after_q5_q1_mean"
                    ],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
