"""Coverage scenarios: the three jurisdiction-level count surfaces, and their per-state deltas.

Amendment 3 item 6, verbatim: *"Coverage scenarios: publish observed-lower-bound /
coverage-completed central / benchmark-full upper surfaces with per-state deltas."*

    uv run python scripts/release/build_coverage_scenarios.py --year 2024 --out-dir <dir>

Wired into `archive/2026-09-cull/scripts/release/package_edition.py`, which writes the tables into
the edition's `scenarios/` directory. See
`docs/archive/2026-09/analysis_scratch/final_phase/COVERAGE_SCENARIOS_CONTRACT.md`.

The published surface is ONE number per jurisdiction and offense, and a reader has no way to see
how much of it was reported and how much was supplied by the coverage machinery. These three
surfaces make that visible without changing the published number:

    observed   the reported-only lower bound: the observed component alone
    central    what is published, with the E5 MUNICIPAL central correction applied to the
               benchmark-imputed municipal mass -- this is where the attached-not-applied
               correction lands
    upper      benchmark-full: the empirical bounds' hi-95 multiplier on the imputed mass

Only the coverage-supplied mass moves between scenarios. The observed component is identical in
all three, and the central scenario's NON-IMPUTED mass is identical to the published surface's --
both asserted here and mirrored in `validate_release_outputs.py`.

BLOCK-GROUP SCENARIOS ARE NOT PUBLISHED. The bounds were measured on whole masked jurisdictions
(E5's 619 solo masks); nothing in that experiment says how a scenario's extra mass would
distribute WITHIN a jurisdiction, and the within-jurisdiction share vector is fitted on incident
truth that silent units by definition do not have. Pushing a jurisdiction-level envelope down a
share vector would manufacture block-group precision the evidence does not support, so this lane
stops at the support its evidence was measured at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crimerisk.benchmark_imputation import (  # noqa: E402
    COUNTY_UNIT_KIND,
    MUNICIPAL_UNIT_KIND,
    attach_empirical_bounds,
    benchmark_imputation_diagnostics_path,
    benchmark_imputation_units_path,
    load_empirical_bounds,
)
from crimerisk.crime import OFFENSES_7  # noqa: E402
from crimerisk.paths import RepoPaths, get_paths  # noqa: E402


SCENARIOS_DIRNAME = "scenarios"
SCENARIO_VERSION = "coverage_scenarios_v1"
# The offense key a whole-jurisdiction row uses in the per-state delta table. Same convention the
# empirical bounds table uses for its lane-level fallback row, reused rather than reinvented.
ALL_OFFENSES_KEY = "ALL"
TOLERANCE = 1e-6

SCENARIO_COLUMNS: tuple[str, ...] = (
    "year",
    "state_fips",
    "state_abbr",
    "jurisdiction_id",
    "jurisdiction_type",
    "offense",
    "count_published",
    "count_observed",
    "count_central",
    "count_upper",
    "observed_component_count",
    "partial_component_count",
    "fill_component_count",
    "benchmark_imputed_count",
    "benchmark_imputed_municipal",
    "benchmark_imputed_county",
    "benchmark_imputed_central",
    "benchmark_imputed_upper",
    "central_correction_applied",
    "scenario_dependent_count",
)
STATE_DELTA_COLUMNS: tuple[str, ...] = (
    "year",
    "state_fips",
    "state_abbr",
    "offense",
    "count_published",
    "count_observed",
    "count_central",
    "count_upper",
    "benchmark_imputed_count",
    "coverage_supplied_count",
    "scenario_dependent_count",
    "imputed_share_of_published",
    "coverage_supplied_share_of_published",
    "scenario_dependent_share_of_published",
    "central_minus_published",
    "central_minus_published_share",
)


def coverage_scenarios_path(out_dir: Path, *, year: int) -> Path:
    return Path(out_dir) / f"coverage_scenarios_{int(year)}.parquet"


def coverage_scenario_state_deltas_path(out_dir: Path, *, year: int) -> Path:
    return Path(out_dir) / f"coverage_scenario_state_deltas_{int(year)}.parquet"


def coverage_scenarios_summary_path(out_dir: Path) -> Path:
    return Path(out_dir) / "coverage_scenarios_summary.json"


def _controls_path(paths: RepoPaths, *, year: int) -> Path:
    return paths.state_dir / "controls" / f"jurisdiction_controls_{int(year)}.parquet"


def resolve_bounds_rule_version(paths: RepoPaths, *, year: int) -> str:
    """Which rule version's bounds price THIS build's imputed mass.

    The bounds are `actual / predicted` ratios measured against a specific imputation rule, so a
    v2 estimate priced with v1 bounds (or the reverse) is a category error, not a conservative
    choice. The build's own diagnostics record which rule produced the units, and that record is
    the authority -- inferring it from the unit table's columns would silently pick a version
    whenever a column was added or dropped for an unrelated reason.
    """
    path = benchmark_imputation_diagnostics_path(paths, year=int(year))
    if not path.exists():
        raise SystemExit(
            f"missing benchmark imputation diagnostics {path}; build the controls first"
        )
    diagnostics = json.loads(path.read_text())
    version = ((diagnostics.get("imputation_v2") or {}).get("rate_rule_version"))
    if not version:
        raise SystemExit(
            f"{path} does not record imputation_v2.rate_rule_version; refusing to guess which "
            "empirical bounds price this build's imputed mass"
        )
    return str(version)


def _unit_jurisdiction_id(units: pd.DataFrame) -> pd.Series:
    """The control jurisdiction a silent territory unit's mass was added to.

    Mirrors `benchmark_imputation.apply_benchmark_imputation_to_controls`: a municipal unit's
    `unit_id` IS the jurisdiction id, and a county-remainder unit's mass lands on its state's
    non-municipal remainder jurisdiction.
    """
    municipal = units["unit_kind"].astype("string").eq(MUNICIPAL_UNIT_KIND)
    return pd.Series(
        np.where(
            municipal,
            units["unit_id"].astype("string"),
            units["state_fips"].astype("string").str.zfill(2) + ":state_nonmunicipal_remainder",
        ),
        index=units.index,
        dtype="string",
    )


def build_coverage_scenarios(
    *, paths: RepoPaths, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """The three surfaces, the per-state deltas, and the record of how they were made."""
    controls_path = _controls_path(paths, year=year)
    if not controls_path.exists():
        raise SystemExit(f"missing jurisdiction controls {controls_path}")
    units_path = benchmark_imputation_units_path(paths, year=int(year))
    if not units_path.exists():
        raise SystemExit(f"missing benchmark imputation units {units_path}")

    rule_version = resolve_bounds_rule_version(paths, year=year)
    bounds = load_empirical_bounds(paths, rule_version=rule_version)
    units = attach_empirical_bounds(pd.read_parquet(units_path), bounds=bounds)
    units = units[units["offense"].astype("string").isin(OFFENSES_7)].copy()
    units["jurisdiction_id"] = _unit_jurisdiction_id(units)
    units["offense"] = units["offense"].astype("string")
    units["imputed_count"] = pd.to_numeric(units["imputed_count"], errors="coerce").fillna(0.0)

    # The lane split matters twice: the central correction is applied to the MUNICIPAL lane only
    # (that is the lane E5 measured the 2.18x over-statement on and the correction is named for),
    # and the upper multiplier is per-stratum, which is per-lane by construction.
    municipal = units["unit_kind"].astype("string").eq(MUNICIPAL_UNIT_KIND)
    units["_municipal_mass"] = units["imputed_count"].where(municipal, 0.0)
    units["_county_mass"] = units["imputed_count"].where(~municipal, 0.0)
    # Central: correction on the municipal lane, county lane at its published point estimate.
    units["_central_mass"] = np.where(
        municipal,
        units["imputed_count"] * pd.to_numeric(units["central_correction"], errors="coerce").fillna(1.0),
        units["imputed_count"],
    )
    units["_upper_mass"] = pd.to_numeric(units["bound_hi_95"], errors="coerce").fillna(0.0)
    # Asserted, not assumed: the ordering the three surfaces are published under has to hold
    # per unit, or a jurisdiction total could satisfy it by cancellation.
    inverted = units["_upper_mass"] < units["_central_mass"] - TOLERANCE
    if bool(inverted.any()):
        raise SystemExit(
            "empirical hi-95 bound lies below the centred point estimate on "
            f"{int(inverted.sum())} unit-offense rows; the scenario ladder would not be ordered: "
            f"{units.loc[inverted, ['unit_kind', 'unit_id', 'offense']].head(5).to_dict(orient='records')}"
        )

    by_jurisdiction = (
        units.groupby(["jurisdiction_id", "offense"], dropna=False)[
            ["_municipal_mass", "_county_mass", "_central_mass", "_upper_mass"]
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                "_municipal_mass": "benchmark_imputed_municipal",
                "_county_mass": "benchmark_imputed_county",
                "_central_mass": "benchmark_imputed_central",
                "_upper_mass": "benchmark_imputed_upper",
            }
        )
    )

    controls = pd.read_parquet(
        controls_path,
        columns=[
            "state_fips",
            "state_abbr",
            "jurisdiction_id",
            "jurisdiction_type",
            "offense",
            "observed_component_count",
            "partial_component_count",
            "fill_component_count",
            "benchmark_imputed_count",
            "estimated_count_ags_core",
        ],
    )
    controls = controls[controls["offense"].astype("string").isin(OFFENSES_7)].copy()
    controls["state_fips"] = controls["state_fips"].astype("string").str.zfill(2)
    controls["jurisdiction_id"] = controls["jurisdiction_id"].astype("string")
    controls["offense"] = controls["offense"].astype("string")
    for column in (
        "observed_component_count",
        "partial_component_count",
        "fill_component_count",
        "benchmark_imputed_count",
        "estimated_count_ags_core",
    ):
        controls[column] = pd.to_numeric(controls[column], errors="coerce").fillna(0.0)

    # The controls' own decomposition is exact by construction; check it here anyway, because
    # every scenario below is built on it and a silent drift would move all three surfaces
    # together and therefore be invisible in their deltas.
    recomposed = (
        controls["observed_component_count"]
        + controls["partial_component_count"]
        + controls["fill_component_count"]
        + controls["benchmark_imputed_count"]
    )
    decomposition_drift = float((recomposed - controls["estimated_count_ags_core"]).abs().max() or 0.0)
    if decomposition_drift > TOLERANCE:
        raise SystemExit(
            "jurisdiction controls do not decompose into observed + partial + fill + benchmark; "
            f"max abs delta={decomposition_drift:.3e}"
        )

    out = controls.merge(by_jurisdiction, on=["jurisdiction_id", "offense"], how="left")
    for column in (
        "benchmark_imputed_municipal",
        "benchmark_imputed_county",
        "benchmark_imputed_central",
        "benchmark_imputed_upper",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    # The units the controls were built from must account for the controls' own imputed column,
    # or the scenarios would price mass the published surface never received.
    unit_drift = float(
        (
            out["benchmark_imputed_municipal"]
            + out["benchmark_imputed_county"]
            - out["benchmark_imputed_count"]
        )
        .abs()
        .max()
        or 0.0
    )
    if unit_drift > TOLERANCE:
        raise SystemExit(
            "benchmark imputation units do not reconstruct the controls' benchmark_imputed_count; "
            f"max abs delta={unit_drift:.3e}"
        )

    out["year"] = int(year)
    out["count_published"] = out["estimated_count_ags_core"]
    # The mass every scenario shares, and the three that differ only in what is added to it.
    non_imputed = (
        out["observed_component_count"] + out["partial_component_count"] + out["fill_component_count"]
    )
    out["count_observed"] = out["observed_component_count"]
    out["count_central"] = non_imputed + out["benchmark_imputed_central"]
    out["count_upper"] = non_imputed + out["benchmark_imputed_upper"]
    out["central_correction_applied"] = np.where(
        out["benchmark_imputed_municipal"].gt(0.0),
        np.where(
            out["benchmark_imputed_municipal"].gt(0.0),
            (out["benchmark_imputed_central"] - out["benchmark_imputed_county"])
            / out["benchmark_imputed_municipal"].replace(0.0, np.nan),
            1.0,
        ),
        1.0,
    )
    out["central_correction_applied"] = pd.to_numeric(
        out["central_correction_applied"], errors="coerce"
    ).fillna(1.0)
    out["scenario_dependent_count"] = out["count_upper"] - out["count_observed"]

    _assert_scenario_invariants(out)

    scenarios = out[list(SCENARIO_COLUMNS)].sort_values(
        ["state_fips", "jurisdiction_id", "offense"], kind="mergesort"
    ).reset_index(drop=True)
    state_deltas = _build_state_deltas(scenarios, year=int(year))
    summary = _summarize(
        scenarios,
        state_deltas=state_deltas,
        year=int(year),
        rule_version=rule_version,
        units=units,
    )
    return scenarios, state_deltas, summary


def _assert_scenario_invariants(frame: pd.DataFrame) -> None:
    """The two claims the tables make, checked before either is written.

    1. The ladder is ordered: `observed <= central <= upper`, every row.
    2. The central scenario's NON-IMPUTED mass is the published surface's non-imputed mass --
       i.e. the correction moves the imputed component and nothing else. This is what makes
       "central is the published surface, centred" a true sentence rather than a hopeful one.
    """
    ordered_low = frame["count_central"] - frame["count_observed"]
    if bool((ordered_low < -TOLERANCE).any()):
        bad = frame.loc[ordered_low < -TOLERANCE].head(5)
        raise SystemExit(
            "coverage scenarios are not ordered observed <= central on "
            f"{int((ordered_low < -TOLERANCE).sum())} rows: "
            f"{bad[['jurisdiction_id', 'offense', 'count_observed', 'count_central']].to_dict(orient='records')}"
        )
    ordered_high = frame["count_upper"] - frame["count_central"]
    if bool((ordered_high < -TOLERANCE).any()):
        bad = frame.loc[ordered_high < -TOLERANCE].head(5)
        raise SystemExit(
            "coverage scenarios are not ordered central <= upper on "
            f"{int((ordered_high < -TOLERANCE).sum())} rows: "
            f"{bad[['jurisdiction_id', 'offense', 'count_central', 'count_upper']].to_dict(orient='records')}"
        )
    published_non_imputed = frame["count_published"] - frame["benchmark_imputed_count"]
    central_non_imputed = frame["count_central"] - frame["benchmark_imputed_central"]
    drift = float((central_non_imputed - published_non_imputed).abs().max() or 0.0)
    if drift > TOLERANCE:
        raise SystemExit(
            "the central scenario's non-imputed mass is not the published surface's; "
            f"max abs delta={drift:.3e}"
        )


def _build_state_deltas(scenarios: pd.DataFrame, *, year: int) -> pd.DataFrame:
    """Per state and offense, plus an `ALL` row per state: how much of the total is scenario-dependent."""
    value_columns = [
        "count_published",
        "count_observed",
        "count_central",
        "count_upper",
        "benchmark_imputed_count",
    ]
    by_offense = (
        scenarios.groupby(["state_fips", "state_abbr", "offense"], dropna=False)[value_columns]
        .sum()
        .reset_index()
    )
    all_offenses = (
        scenarios.groupby(["state_fips", "state_abbr"], dropna=False)[value_columns]
        .sum()
        .reset_index()
        .assign(offense=ALL_OFFENSES_KEY)
    )
    out = pd.concat([by_offense, all_offenses], ignore_index=True)
    out["year"] = int(year)
    # "Coverage-supplied" is everything the published number carries that nobody reported:
    # within-year fill, partial-reporting uplift and silent-unit imputation together. The
    # scenario-dependent span is narrower -- only the imputed lane has a measured envelope.
    out["coverage_supplied_count"] = out["count_published"] - out["count_observed"]
    out["scenario_dependent_count"] = out["count_upper"] - out["count_observed"]
    published = out["count_published"].replace(0.0, np.nan)
    out["imputed_share_of_published"] = (out["benchmark_imputed_count"] / published).fillna(0.0)
    out["coverage_supplied_share_of_published"] = (out["coverage_supplied_count"] / published).fillna(0.0)
    out["scenario_dependent_share_of_published"] = (out["scenario_dependent_count"] / published).fillna(0.0)
    out["central_minus_published"] = out["count_central"] - out["count_published"]
    out["central_minus_published_share"] = (out["central_minus_published"] / published).fillna(0.0)
    return out[list(STATE_DELTA_COLUMNS)].sort_values(
        ["state_fips", "offense"], kind="mergesort"
    ).reset_index(drop=True)


def _summarize(
    scenarios: pd.DataFrame,
    *,
    state_deltas: pd.DataFrame,
    year: int,
    rule_version: str,
    units: pd.DataFrame,
) -> dict[str, object]:
    national = {
        offense: {
            "observed": float(group["count_observed"].sum()),
            "central": float(group["count_central"].sum()),
            "upper": float(group["count_upper"].sum()),
            "published": float(group["count_published"].sum()),
            "benchmark_imputed": float(group["benchmark_imputed_count"].sum()),
        }
        for offense, group in scenarios.groupby("offense")
    }
    national[ALL_OFFENSES_KEY] = {
        "observed": float(scenarios["count_observed"].sum()),
        "central": float(scenarios["count_central"].sum()),
        "upper": float(scenarios["count_upper"].sum()),
        "published": float(scenarios["count_published"].sum()),
        "benchmark_imputed": float(scenarios["benchmark_imputed_count"].sum()),
    }
    all_rows = state_deltas[state_deltas["offense"].eq(ALL_OFFENSES_KEY)]
    widest = all_rows.nlargest(10, "scenario_dependent_share_of_published")
    basis = units["bounds_basis"].astype("string").value_counts().to_dict()
    return {
        "version": SCENARIO_VERSION,
        "year": int(year),
        "bounds_rule_version": rule_version,
        "scenarios": {
            "observed": "reported-only lower bound: the observed component alone",
            "central": (
                "the published surface with the E5 MUNICIPAL central correction (1 / agg_ratio) "
                "applied to benchmark-imputed municipal mass; this is where the "
                "attached-not-applied correction lands"
            ),
            "upper": "benchmark-full: the empirical bounds' hi-95 multiplier on imputed mass",
        },
        "support": "jurisdiction",
        "block_group_scenarios_published": False,
        "block_group_rationale": (
            "the bounds were measured on whole masked jurisdictions; nothing in that experiment "
            "says how a scenario's extra mass distributes within one, so this lane stops at the "
            "support its evidence was measured at"
        ),
        "jurisdiction_offense_rows": int(len(scenarios)),
        "state_offense_rows": int(len(state_deltas)),
        "national": national,
        "bounds_basis_unit_rows": {str(key): int(value) for key, value in basis.items()},
        "widest_states": [
            {
                "state_abbr": str(row.state_abbr),
                "state_fips": str(row.state_fips),
                "scenario_dependent_share_of_published": float(row.scenario_dependent_share_of_published),
                "imputed_share_of_published": float(row.imputed_share_of_published),
                "observed": float(row.count_observed),
                "central": float(row.count_central),
                "upper": float(row.count_upper),
            }
            for row in widest.itertuples(index=False)
        ],
    }


def write_coverage_scenarios(*, paths: RepoPaths, year: int, out_dir: Path) -> dict[str, object]:
    scenarios, state_deltas, summary = build_coverage_scenarios(paths=paths, year=int(year))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = coverage_scenarios_path(out_dir, year=int(year))
    delta_path = coverage_scenario_state_deltas_path(out_dir, year=int(year))
    summary_path = coverage_scenarios_summary_path(out_dir)
    scenarios.to_parquet(scenario_path, index=False)
    state_deltas.to_parquet(delta_path, index=False)
    summary = dict(summary)
    summary["scenarios_path"] = scenario_path.name
    summary["state_deltas_path"] = delta_path.name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"where to write the tables (default analysis_scratch/{SCENARIOS_DIRNAME})",
    )
    args = parser.parse_args(argv)
    paths = get_paths()
    out_dir = args.out_dir or (REPO_ROOT / "analysis_scratch" / SCENARIOS_DIRNAME)
    summary = write_coverage_scenarios(paths=paths, year=int(args.year), out_dir=out_dir)
    national = summary["national"]
    print(f"[scenarios] bounds rule version: {summary['bounds_rule_version']}")
    print(f"[scenarios] {summary['jurisdiction_offense_rows']} jurisdiction-offense rows")
    for offense in [*OFFENSES_7, ALL_OFFENSES_KEY]:
        entry = national.get(offense)
        if entry is None:
            continue
        print(
            f"[scenarios]   {offense:20s} observed {entry['observed']:>12,.0f}  "
            f"central {entry['central']:>12,.0f}  upper {entry['upper']:>12,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
