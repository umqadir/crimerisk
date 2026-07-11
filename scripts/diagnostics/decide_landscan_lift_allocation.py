from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.crime import OFFENSES_7
from crimerisk.denominators import (
    DENOMINATOR_SOURCE_COLUMNS,
    PRIMARY_DENOMINATOR_BY_OFFENSE,
    add_offense_denominators,
)
from crimerisk.model_surface import build_bg_feature_frame
from crimerisk.paths import RepoPaths


DECISION_OFFENSES = ("murder", "rape")
REPORT_OFFENSES = ("murder", "rape", "robbery", "aggravated_assault", "larceny")
EVALUABLE_ROLES = {
    "direct_posterior_live",
    "residual_training_only",
    "validation_holdout_only",
}
OFFENSE_SEED_OFFSETS = {offense: idx * 1009 for idx, offense in enumerate(REPORT_OFFENSES, start=1)}


def _weighted_mean(frame: pd.DataFrame, value_col: str) -> float:
    weights = pd.to_numeric(frame["incident_total"], errors="coerce").fillna(0.0)
    values = pd.to_numeric(frame[value_col], errors="coerce")
    ok = weights.gt(0.0) & values.notna()
    denom = float(weights.loc[ok].sum())
    if denom <= 0.0:
        return float("nan")
    return float((values.loc[ok] * weights.loc[ok]).sum() / denom)


def _bootstrap_delta_se(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> float:
    usable = frame[pd.to_numeric(frame["incident_total"], errors="coerce").gt(0.0)].copy()
    if usable.empty:
        return float("nan")
    cities = np.array(sorted(usable["jurisdiction_id"].astype(str).unique().tolist()), dtype=object)
    if len(cities) <= 1 or n_boot <= 1:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    deltas: list[float] = []
    city_groups = {city: group for city, group in usable.groupby(usable["jurisdiction_id"].astype(str), sort=False)}
    for _ in range(int(n_boot)):
        sample = rng.choice(cities, size=len(cities), replace=True)
        boot = pd.concat([city_groups[str(city)] for city in sample], ignore_index=True)
        delta = _weighted_mean(boot, "tvd_delta_lift_minus_current")
        if np.isfinite(delta):
            deltas.append(float(delta))
    if len(deltas) <= 1:
        return float("nan")
    return float(np.std(np.asarray(deltas, dtype=float), ddof=1))


def _safe_share(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(values.sum())
    if total > 0.0:
        return values / total
    if len(values) == 0:
        return values
    return pd.Series(1.0 / float(len(values)), index=values.index, dtype=float)


def build_landscan_lift_decision(
    *,
    paths: RepoPaths,
    year: int,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    roles_path = paths.state_dir / "modeling" / f"city_role_inventory_{int(year)}.parquet"
    truth_path = paths.state_dir / "modeling" / f"next_phase_validation_city_incident_share_surface_{int(year)}.parquet"
    crosswalk_path = paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"

    roles = pd.read_parquet(roles_path)
    roles = roles[
        roles["role"].astype("string").isin(sorted(EVALUABLE_ROLES))
        & roles["offense"].astype("string").isin(list(REPORT_OFFENSES))
    ].copy()
    roles = roles.dropna(subset=["jurisdiction_id", "offense"]).drop_duplicates(
        ["jurisdiction_id", "offense"],
        keep="first",
    )

    truth = pd.read_parquet(
        truth_path,
        columns=["city_name", "jurisdiction_id", "state_fips", "year", "offense", "block_group_geoid", "incident_count"],
    )
    truth = truth[truth["offense"].astype("string").isin(list(REPORT_OFFENSES))].copy()
    truth["jurisdiction_id"] = truth["jurisdiction_id"].astype("string")
    truth["offense"] = truth["offense"].astype("string")
    truth["block_group_geoid"] = truth["block_group_geoid"].astype("string").str.zfill(12)
    truth["incident_count"] = pd.to_numeric(truth["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    truth_bg = (
        truth.groupby(["jurisdiction_id", "offense", "block_group_geoid"], dropna=False)
        .agg(
            incident_count=("incident_count", "sum"),
            city_name=("city_name", "first"),
            state_fips=("state_fips", "first"),
        )
        .reset_index()
    )

    bg_base = build_bg_feature_frame(paths=paths, year=int(year))
    current_denoms = add_offense_denominators(
        bg_base,
        paths=paths,
        year=int(year),
        apply_landscan_day_floor=False,
    )
    lifted_denoms = add_offense_denominators(
        bg_base,
        paths=paths,
        year=int(year),
        apply_landscan_day_floor=True,
    )
    denom_cols = ["bg_id"]
    for offense in REPORT_OFFENSES:
        denom_type = PRIMARY_DENOMINATOR_BY_OFFENSE[offense]
        source_col = DENOMINATOR_SOURCE_COLUMNS[denom_type]
        denom_cols.append(f"denominator_current_{offense}")
        denom_cols.append(f"denominator_lifted_{offense}")
        current_denoms[f"denominator_current_{offense}"] = pd.to_numeric(
            current_denoms[source_col], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        lifted_denoms[f"denominator_lifted_{offense}"] = pd.to_numeric(
            lifted_denoms[source_col], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
    denom = current_denoms[["bg_id", *[c for c in current_denoms.columns if c.startswith("denominator_current_")]]].merge(
        lifted_denoms[["bg_id", *[c for c in lifted_denoms.columns if c.startswith("denominator_lifted_")]]],
        on="bg_id",
        how="outer",
    )
    denom["bg_id"] = denom["bg_id"].astype("string").str.zfill(12)

    crosswalk = pd.read_parquet(
        crosswalk_path,
        columns=["state_fips", "block_group_geoid", "jurisdiction_id", "allocation_share"],
    )
    crosswalk["jurisdiction_id"] = crosswalk["jurisdiction_id"].astype("string")
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    crosswalk["allocation_share"] = pd.to_numeric(crosswalk["allocation_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    crosswalk = crosswalk[crosswalk["jurisdiction_id"].isin(roles["jurisdiction_id"].astype("string"))].copy()
    crosswalk = crosswalk.merge(denom, left_on="block_group_geoid", right_on="bg_id", how="left")

    rows: list[dict[str, Any]] = []
    for role in roles.itertuples(index=False):
        jurisdiction_id = str(role.jurisdiction_id)
        offense = str(role.offense)
        footprint = crosswalk[crosswalk["jurisdiction_id"].astype(str).eq(jurisdiction_id)].copy()
        truth_part = truth_bg[
            truth_bg["jurisdiction_id"].astype(str).eq(jurisdiction_id)
            & truth_bg["offense"].astype(str).eq(offense)
        ].copy()
        if truth_part.empty:
            continue
        city_name = (
            str(truth_part["city_name"].dropna().iloc[0])
            if truth_part["city_name"].notna().any()
            else str(getattr(role, "city_name", jurisdiction_id))
        )
        frame = footprint[["block_group_geoid", "allocation_share", f"denominator_current_{offense}", f"denominator_lifted_{offense}"]].merge(
            truth_part[["block_group_geoid", "incident_count"]],
            on="block_group_geoid",
            how="outer",
        )
        frame["allocation_share"] = pd.to_numeric(frame["allocation_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
        frame["incident_count"] = pd.to_numeric(frame["incident_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        incident_total = float(frame["incident_count"].sum())
        if incident_total <= 0.0:
            continue
        current_weight = pd.to_numeric(frame[f"denominator_current_{offense}"], errors="coerce").fillna(0.0).clip(lower=0.0) * frame["allocation_share"]
        lifted_weight = pd.to_numeric(frame[f"denominator_lifted_{offense}"], errors="coerce").fillna(0.0).clip(lower=0.0) * frame["allocation_share"]
        truth_share = frame["incident_count"] / incident_total
        current_share = _safe_share(current_weight)
        lifted_share = _safe_share(lifted_weight)
        tvd_current = float(0.5 * (current_share - truth_share).abs().sum())
        tvd_lifted = float(0.5 * (lifted_share - truth_share).abs().sum())
        rows.append(
            {
                "city_name": city_name,
                "jurisdiction_id": jurisdiction_id,
                "offense": offense,
                "role": str(role.role),
                "incident_total": incident_total,
                "footprint_bg_count": int(footprint["block_group_geoid"].nunique()),
                "truth_bg_count": int(truth_part["block_group_geoid"].nunique()),
                "tvd_current": tvd_current,
                "tvd_lifted": tvd_lifted,
                "tvd_delta_lift_minus_current": tvd_lifted - tvd_current,
            }
        )

    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for offense in REPORT_OFFENSES:
        part = detail[detail["offense"].astype(str).eq(offense)].copy()
        if part.empty:
            continue
        se = _bootstrap_delta_se(
            part,
            n_boot=int(bootstrap_iterations),
            seed=int(seed) + int(OFFENSE_SEED_OFFSETS[offense]),
        )
        summary_rows.append(
            {
                "scope": offense,
                "offenses": offense,
                "city_offense_rows": int(len(part)),
                "cities": int(part["jurisdiction_id"].nunique()),
                "incident_total": float(pd.to_numeric(part["incident_total"], errors="coerce").sum()),
                "tvd_current": _weighted_mean(part, "tvd_current"),
                "tvd_lifted": _weighted_mean(part, "tvd_lifted"),
                "tvd_delta_lift_minus_current": _weighted_mean(part, "tvd_delta_lift_minus_current"),
                "bootstrap_se_delta": se,
            }
        )
    decision_part = detail[detail["offense"].astype(str).isin(DECISION_OFFENSES)].copy()
    decision_se = _bootstrap_delta_se(decision_part, n_boot=int(bootstrap_iterations), seed=int(seed) + 777)
    decision_delta = _weighted_mean(decision_part, "tvd_delta_lift_minus_current")
    accepted_for_allocation = bool(np.isfinite(decision_delta) and (not np.isfinite(decision_se) or decision_delta <= decision_se))
    summary_rows.append(
        {
            "scope": "murder_rape_decision",
            "offenses": ",".join(DECISION_OFFENSES),
            "city_offense_rows": int(len(decision_part)),
            "cities": int(decision_part["jurisdiction_id"].nunique()),
            "incident_total": float(pd.to_numeric(decision_part["incident_total"], errors="coerce").sum()),
            "tvd_current": _weighted_mean(decision_part, "tvd_current"),
            "tvd_lifted": _weighted_mean(decision_part, "tvd_lifted"),
            "tvd_delta_lift_minus_current": decision_delta,
            "bootstrap_se_delta": decision_se,
        }
    )
    summary = pd.DataFrame(summary_rows)
    decision = {
        "year": int(year),
        "criterion": "accept LandScan lift for allocation if murder+rape exposure-baseline TVD does not degrade beyond one city-bootstrap SE",
        "decision_scope": "murder_rape_decision",
        "decision_offenses": list(DECISION_OFFENSES),
        "report_offenses": list(REPORT_OFFENSES),
        "evaluable_roles": sorted(EVALUABLE_ROLES),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(seed),
        "murder_rape_tvd_current": _weighted_mean(decision_part, "tvd_current"),
        "murder_rape_tvd_lifted": _weighted_mean(decision_part, "tvd_lifted"),
        "murder_rape_tvd_delta_lift_minus_current": decision_delta,
        "murder_rape_bootstrap_se_delta": decision_se,
        "accepted_for_allocation": accepted_for_allocation,
        "allocation_branch": (
            "landscan_lifted_allocation_and_publication"
            if accepted_for_allocation
            else "publication_denominators_only_allocation_baselines_current_exposure"
        ),
        "gradient_gate_status": "descriptive_not_decision_gate_for_person_offenses",
    }
    return detail, summary, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--detail-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "landscan_lift_allocation_decision_city_offense_2024.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "landscan_lift_allocation_decision_summary_2024.csv",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "landscan_lift_allocation_decision_2024.json",
    )
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    detail, summary, decision = build_landscan_lift_decision(
        paths=paths,
        year=int(args.year),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    args.detail_out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_out, index=False)
    summary.to_csv(args.summary_out, index=False)
    args.json_out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
