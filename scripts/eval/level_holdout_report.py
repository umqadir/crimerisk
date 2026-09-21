"""Rebuild the holdout tables from the per-configuration case files.

The sweep writes one case file per configuration. This reads them all, attaches the
agency size band from the canonical agency master, and writes the single results table
and its markdown. Keeping it separate means the tables can be re-cut -- a new stratum,
a corrected band -- without paying for the estimator again.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crimerisk.paths import get_paths  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_level_holdout", Path(__file__).resolve().parent / "level_holdout.py"
)
_holdout = importlib.util.module_from_spec(_spec)
sys.modules["_level_holdout"] = _holdout
_spec.loader.exec_module(_holdout)


CONFIG_ORDER = [
    "baseline",
    "rule_a_own_history_bound",
    "rule_b_joint_vector",
    "rule_ab",
    "rule_c_nibrs_guard",
    "rule_d_own_history_pool",
    "all_rules",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the level-lane holdout tables.")
    parser.add_argument("--dir", type=Path, default=Path("state/eval/level_v1"))
    args = parser.parse_args()
    paths = get_paths()
    out_dir = paths.repo_root / args.dir

    files = sorted(out_dir.glob("agency_cases_*.parquet"))
    if not files:
        raise SystemExit(f"no per-configuration case files in {out_dir}")
    cases = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    master = pd.read_parquet(
        paths.state_dir / "reference" / "agency_master.parquet",
        columns=["ori9", "population_latest_nibrs"],
    ).drop_duplicates("ori9")
    population = cases["ori9"].astype(str).map(
        master.set_index("ori9")["population_latest_nibrs"]
    )
    cases["population"] = pd.to_numeric(cases["population"], errors="coerce").fillna(
        pd.to_numeric(population, errors="coerce")
    )
    cases["repair_mode"] = cases["agency_estimate_source"]
    cases["size_band"] = _holdout._size_band(cases["population"])

    order = {name: index for index, name in enumerate(CONFIG_ORDER)}
    cases["_order"] = cases["config"].map(order).fillna(len(order))

    tables: list[pd.DataFrame] = []
    for stratum, by in (
        ("overall", ["config", "arm"]),
        ("repair_mode", ["config", "arm", "repair_mode"]),
        ("size_band", ["config", "arm", "size_band"]),
        ("offense", ["config", "arm", "offense"]),
    ):
        table = _holdout.summarise(cases, by=by)
        table.insert(0, "stratum", stratum)
        table.insert(0, "lane", "agency")
        tables.append(table)

    remainder_path = out_dir / "remainder_cases.parquet"
    if remainder_path.exists():
        remainder = pd.read_parquet(remainder_path)
        remainder["config"] = "remainder_lane"
        remainder["arm"] = "leave_state_out"
        for stratum, by in (("overall", ["config", "arm"]), ("offense", ["config", "arm", "offense"])):
            table = _holdout.summarise(remainder, by=by)
            table.insert(0, "stratum", stratum)
            table.insert(0, "lane", "state_remainder")
            tables.append(table)

    columns = [
        "lane", "stratum", "config", "arm", "repair_mode", "size_band", "offense",
        "cases", "observed_total", "predicted_total", "total_ratio",
        "mean_poisson_deviance", "median_abs_log_ratio", "share_within_25pct",
        "observed_positive_cases",
    ]
    results = pd.concat(tables, ignore_index=True)
    for column in columns:
        if column not in results.columns:
            results[column] = pd.NA
    results["_order"] = results["config"].map(order).fillna(len(order))
    results = results.sort_values(
        ["lane", "stratum", "_order", "arm", "repair_mode", "size_band", "offense"],
        kind="mergesort",
    ).drop(columns="_order")[columns]
    results.to_csv(out_dir / "results.csv", index=False)

    lines = ["# Level-lane holdout", "", "## Overall", ""]
    lines.append(_holdout._markdown_table(results[results["stratum"].eq("overall")]))
    for stratum, title in (
        ("repair_mode", "By repair mode"),
        ("size_band", "By agency size band"),
        ("offense", "By offense"),
    ):
        part = results[results["stratum"].eq(stratum)]
        if not part.empty:
            lines += ["", f"## {title}", "", _holdout._markdown_table(part)]
    # The holdout cannot say whether refusing a filed row was right. The following
    # year can, and that evidence belongs beside the table a rule is judged on.
    corroboration = out_dir / "next_year_corroboration.json"
    if corroboration.exists():
        import json

        report = json.loads(corroboration.read_text())
        rows = []
        for rules, entry in report.get("configurations", {}).items():
            if not entry.get("testable_rows"):
                continue
            rows.append(
                {
                    "screen": rules,
                    "refused_rows": entry["screened_rows"],
                    "refused_agencies": entry["screened_agencies"],
                    "refused_counts": round(entry["screened_count_mass"]),
                    "testable": entry["testable_rows"],
                    f"next_year_recovers": entry["next_year_recovers"],
                    "next_year_holds": entry["next_year_holds_at_or_below"],
                    "precision": entry.get("precision_next_year_recovers"),
                }
            )
        if rows:
            lines += [
                "",
                f"## Refusals checked against {report.get('corroborating_year')}",
                "",
                "A refused target-year row whose next-year filing comes back up was a "
                "filing failure; one that holds at or below the refused level was a real "
                "decline the screen discarded.",
                "",
                _holdout._markdown_table(pd.DataFrame(rows)),
            ]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"configurations: {sorted(set(cases['config']))}")
    print(f"cases: {len(cases):,}")
    print(f"wrote {out_dir / 'results.csv'} and results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
