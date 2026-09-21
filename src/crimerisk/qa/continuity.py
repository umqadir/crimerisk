from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from crimerisk.crime import OFFENSES_7


@dataclass(frozen=True)
class ContinuityAuditReport:
    summary: dict[str, Any]
    markdown: str


def _describe_series(s: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return {"n": 0}

    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    qv = x.quantile(qs).to_dict()
    out: dict[str, float | int] = {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "max": float(x.max()),
    }
    for q, v in qv.items():
        out[f"p{int(round(float(q) * 100)):02d}"] = float(v)
    return out


def build_2021_continuity_audit(
    *,
    canonical_df: pd.DataFrame,
    year_focus: tuple[int, ...] = (2021, 2022),
    srs_min_months: int | None = None,
) -> ContinuityAuditReport:
    required = {"year", "months_reported"} | {f"{o}_difference_srs_minus_nibrs" for o in OFFENSES_7} | {
        f"source_{o}" for o in OFFENSES_7
    }
    missing = required - set(canonical_df.columns)
    if missing:
        raise ValueError(f"canonical_df missing columns: {sorted(missing)}")

    df = canonical_df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["months_reported"] = pd.to_numeric(df["months_reported"], errors="coerce").astype("Int64")

    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "years": sorted({int(y) for y in df["year"].dropna().unique().tolist()}),
        "year_focus": list(year_focus),
        "srs_min_months": int(srs_min_months) if srs_min_months is not None else None,
        "offenses": list(OFFENSES_7),
    }

    # 1) Differences: SRS - NIBRS (where both are present in the merged file)
    diffs: dict[str, Any] = {}
    for offense in OFFENSES_7:
        col = f"{offense}_difference_srs_minus_nibrs"
        offense_out: dict[str, Any] = {}
        for year in year_focus:
            s = df.loc[df["year"] == year, col]
            offense_out[str(year)] = _describe_series(s)
        diffs[offense] = offense_out
    summary["differences_srs_minus_nibrs"] = diffs

    # 2) Source selection counts by year
    sources: dict[str, Any] = {}
    for offense in OFFENSES_7:
        col = f"source_{offense}"
        offense_out: dict[str, Any] = {}
        for year, grp in df.groupby("year", dropna=True):
            vc = grp[col].fillna("missing").astype(str).value_counts().to_dict()
            offense_out[str(int(year))] = {k: int(v) for k, v in vc.items()}
        sources[offense] = offense_out
    summary["source_counts_by_year"] = sources

    # 3) Months reported coverage histogram (SRS-only field; NIBRS rows may be null)
    months_vc = df["months_reported"].value_counts(dropna=False).sort_index()
    summary["months_reported_value_counts"] = {str(k): int(v) for k, v in months_vc.items()}
    if srs_min_months is not None:
        summary["months_reported_below_threshold_rows"] = int((df["months_reported"] < srs_min_months).fillna(False).sum())

    markdown_lines = [
        "# 2021 Continuity Audit (SRS vs NIBRS-derived)",
        "",
        f"- Rows: {summary['rows']:,}",
        f"- Years present: {', '.join(map(str, summary['years']))}",
        f"- Focus years: {', '.join(map(str, year_focus))}",
        f"- SRS min months: {summary['srs_min_months']}",
        "",
        "## Differences (SRS − NIBRS)",
        "All stats are computed on non-null differences only (agency-years where both sources are present).",
        "",
    ]
    for offense in OFFENSES_7:
        markdown_lines.append(f"### {offense}")
        for year in year_focus:
            stats = summary["differences_srs_minus_nibrs"][offense][str(year)]
            if stats.get("n", 0) == 0:
                markdown_lines.append(f"- {year}: n=0")
                continue
            markdown_lines.append(
                f"- {year}: n={stats['n']:,} mean={stats['mean']:.3f} p50={stats['p50']:.3f} p95={stats['p95']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}"
            )
        markdown_lines.append("")

    return ContinuityAuditReport(summary=summary, markdown="\n".join(markdown_lines).rstrip() + "\n")
