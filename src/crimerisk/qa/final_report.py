from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7


@dataclass(frozen=True)
class FinalReport:
    summary: dict[str, Any]
    markdown: str


def _describe_index(df: pd.DataFrame, *, index_col: str, pop_col: str) -> dict[str, float | int]:
    idx = pd.to_numeric(df[index_col], errors="coerce").astype(float)
    pop = pd.to_numeric(df[pop_col], errors="coerce").fillna(0.0).astype(float)
    mask = idx.notna() & (pop > 0)
    wmean = float(np.average(idx[mask], weights=pop[mask])) if bool(mask.any()) else float("nan")
    q = idx.dropna().quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    return {
        "population_weighted_mean": wmean,
        "p01": float(q.get(0.01, np.nan)),
        "p05": float(q.get(0.05, np.nan)),
        "p50": float(q.get(0.5, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "max": float(idx.max(skipna=True)) if len(idx) else float("nan"),
    }


def _md_table(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    if df.empty:
        return "_(no rows)_\n"
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns.tolist()]

    def esc(v: object) -> str:
        s = "" if v is None else str(v)
        s = s.replace("\n", " ").replace("|", "\\|")
        return s

    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append("")
        lines.append("_(truncated)_")
    return "\n".join(lines) + "\n"


def build_final_report(
    *,
    output_2024: pd.DataFrame,
    output_2029: pd.DataFrame,
    models_2024: dict[str, Any] | None = None,
    models_2029: dict[str, Any] | None = None,
    pop_threshold_for_outliers: int = 1000,
) -> FinalReport:
    summary: dict[str, Any] = {"pop_threshold_for_outliers": int(pop_threshold_for_outliers)}

    def summarize_year(df: pd.DataFrame, year: int, pop_col: str, models: dict[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {"year": int(year), "rows": int(len(df))}
        pop = pd.to_numeric(df[pop_col], errors="coerce").fillna(0.0)
        out["population_zero_rows"] = int((pop <= 0).sum())

        out["constraint_type_counts"] = (
            df.get("constraint_type", pd.Series([], dtype=str)).fillna("missing").astype(str).value_counts().to_dict()
        )
        out["data_quality_tier_counts"] = (
            df.get("data_quality_tier", pd.Series([], dtype=str)).fillna("missing").astype(str).value_counts().to_dict()
        )

        idx_stats = {}
        for offense in OFFENSES_7:
            idx_stats[offense] = _describe_index(df, index_col=f"index_{offense}_primary", pop_col=pop_col)
        idx_stats["total"] = _describe_index(df, index_col="index_total_part1_resident", pop_col=pop_col)
        out["index_summary"] = idx_stats

        # Outliers: highest total index among tracts with meaningful population.
        view = df.copy()
        view["_pop"] = pop
        view = view[view["_pop"] >= float(pop_threshold_for_outliers)].copy()
        cols = [
            "tract_id",
            "state_fips",
            pop_col,
            "bucket_type",
            "bucket_id",
            "constraint_type",
            "data_quality_tier",
            "index_total_part1_resident",
        ]
        cols = [c for c in cols if c in view.columns]
        outliers = view.sort_values("index_total_part1_resident", ascending=False).head(20)[cols].copy()
        out["top_outliers"] = outliers.to_dict(orient="records")

        # Model diagnostics snapshot.
        if models is not None:
            m = {}
            for offense in OFFENSES_7:
                card = models.get(offense, {})
                m[offense] = {
                    "training_rows": card.get("training_rows"),
                    "feature_count": card.get("feature_count"),
                    "train_r2": card.get("train_r2"),
                    "cv_r2_log_rate": card.get("cv_r2_log_rate"),
                }
            out["model_diagnostics"] = m

        return out

    summary["year_2024"] = summarize_year(output_2024, 2024, "population_2024", models_2024)
    summary["year_2029"] = summarize_year(output_2029, 2029, "population_2029", models_2029)

    md: list[str] = []
    md.append("# OpenCrimeRisk final QA report")
    md.append("")

    for year_key, pop_col in [("year_2024", "population_2024"), ("year_2029", "population_2029")]:
        yr = summary[year_key]
        md.append(f"## {yr['year']}")
        md.append(f"- Rows: {yr['rows']:,}")
        md.append(f"- Zero-pop rows: {yr['population_zero_rows']:,}")
        md.append("")

        md.append("### Constraint types")
        md.append(_md_table(pd.DataFrame(list(yr["constraint_type_counts"].items()), columns=["constraint_type", "n"])))

        md.append("### Data quality tiers")
        md.append(_md_table(pd.DataFrame(list(yr["data_quality_tier_counts"].items()), columns=["data_quality_tier", "n"])))

        md.append("### Index distribution (population-weighted mean should be ~100)")
        rows = []
        for k, stats in yr["index_summary"].items():
            rows.append(
                {
                    "index": str(k),
                    "wmean": round(float(stats["population_weighted_mean"]), 6),
                    "p50": round(float(stats["p50"]), 2),
                    "p95": round(float(stats["p95"]), 2),
                    "p99": round(float(stats["p99"]), 2),
                    "max": round(float(stats["max"]), 2),
                }
            )
        md.append(_md_table(pd.DataFrame(rows)))

        md.append(f"### Top outliers (population ≥ {summary['pop_threshold_for_outliers']:,})")
        outliers_df = pd.DataFrame(yr["top_outliers"])
        md.append(_md_table(outliers_df))

        if "model_diagnostics" in yr:
            md.append("### Model diagnostics snapshot")
            md.append(_md_table(pd.DataFrame.from_dict(yr["model_diagnostics"], orient="index").reset_index(names="offense")))

    return FinalReport(summary=summary, markdown="\n".join(md).rstrip() + "\n")
