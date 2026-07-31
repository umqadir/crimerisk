from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

CITY_JURISDICTIONS = {
    "Austin": "48:municipal:place:4805000",
    "Chicago": "17:municipal:place:1714000",
    "New York": "36:municipal:place:3651000",
    "Seattle": "53:municipal:place:5363000",
}

MODEL_SAMPLE_ROWS = [
    {
        "city_name": "Austin",
        "sample_band": "low",
        "block_group_geoid": "484530418001",
        "zip_code": "78758",
        "representative_location": "11929 Rosethorn Drive, Austin, TX 78758",
    },
    {
        "city_name": "Austin",
        "sample_band": "mid",
        "block_group_geoid": "484530019121",
        "zip_code": "78746",
        "representative_location": "6903 Pascal Court, Austin-area edge / West Lake Hills, TX 78746",
    },
    {
        "city_name": "Austin",
        "sample_band": "high",
        "block_group_geoid": "484530011023",
        "zip_code": "78701",
        "representative_location": "401 West Cesar Chavez Street, downtown Austin, TX 78701",
    },
    {
        "city_name": "Chicago",
        "sample_band": "low",
        "block_group_geoid": "170311406011",
        "zip_code": "60625",
        "representative_location": "West Wilson Avenue area, Albany Park, Chicago, IL 60625",
    },
    {
        "city_name": "Chicago",
        "sample_band": "mid",
        "block_group_geoid": "170310514002",
        "zip_code": "60647",
        "representative_location": "Lathrop / North Center edge, Chicago, IL 60647",
    },
    {
        "city_name": "Chicago",
        "sample_band": "high",
        "block_group_geoid": "170318343002",
        "zip_code": "60619",
        "representative_location": "South Dante Avenue, Avalon Park, Chicago, IL 60619",
    },
    {
        "city_name": "New York",
        "sample_band": "low",
        "block_group_geoid": "360810707003",
        "zip_code": "11375",
        "representative_location": "68th Avenue area, Forest Hills, Queens, NY 11375",
    },
    {
        "city_name": "New York",
        "sample_band": "mid",
        "block_group_geoid": "360810464001",
        "zip_code": "11432",
        "representative_location": "Wareham Place area, Jamaica / Queens, NY 11432",
    },
    {
        "city_name": "New York",
        "sample_band": "high",
        "block_group_geoid": "360610166002",
        "zip_code": "10029",
        "representative_location": "East 102nd Street area, East Harlem, Manhattan, NY 10029",
    },
    {
        "city_name": "Seattle",
        "sample_band": "low",
        "block_group_geoid": "530330106023",
        "zip_code": "98136",
        "representative_location": "45th Avenue SW area, West Seattle, WA 98136",
    },
    {
        "city_name": "Seattle",
        "sample_band": "mid",
        "block_group_geoid": "530330079012",
        "zip_code": "98122",
        "representative_location": "East Union Street area, Central District / Madison Valley, Seattle, WA 98122",
    },
    {
        "city_name": "Seattle",
        "sample_band": "high",
        "block_group_geoid": "530330081022",
        "zip_code": "98104",
        "representative_location": "1st Avenue area, downtown Seattle, WA 98104",
    },
]

CRIMEGRADE_SAMPLE_ROWS = {
    "78758": {
        "crimegrade_overall_grade": "D",
        "crimegrade_violent_grade": "D",
        "crimegrade_property_grade": "D-",
        "crimegrade_safety_percentile": 16,
        "crimegrade_rate_per_1000": 45.18,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-78758/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "78746": {
        "crimegrade_overall_grade": "C-",
        "crimegrade_violent_grade": "C+",
        "crimegrade_property_grade": "D+",
        "crimegrade_safety_percentile": 35,
        "crimegrade_rate_per_1000": 31.66,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-78746/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "78701": {
        "crimegrade_overall_grade": "F",
        "crimegrade_violent_grade": "F",
        "crimegrade_property_grade": "F",
        "crimegrade_safety_percentile": 4,
        "crimegrade_rate_per_1000": 70.51,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-78701/",
        "crimegrade_note": "Downtown visitor-heavy ZIP; public CrimeGrade page explicitly warns that high-traffic areas can look worse on a per-resident basis.",
    },
    "60625": {
        "crimegrade_overall_grade": "D",
        "crimegrade_violent_grade": "C-",
        "crimegrade_property_grade": "D+",
        "crimegrade_safety_percentile": 19,
        "crimegrade_rate_per_1000": 42.29,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-60625/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "60647": {
        "crimegrade_overall_grade": "D-",
        "crimegrade_violent_grade": "D+",
        "crimegrade_property_grade": "D",
        "crimegrade_safety_percentile": 13,
        "crimegrade_rate_per_1000": 48.23,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-60647/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "60619": {
        "crimegrade_overall_grade": "F",
        "crimegrade_violent_grade": "F",
        "crimegrade_property_grade": "D-",
        "crimegrade_safety_percentile": 6,
        "crimegrade_rate_per_1000": 63.45,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-60619/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "11375": {
        "crimegrade_overall_grade": "B-",
        "crimegrade_violent_grade": "B",
        "crimegrade_property_grade": "C",
        "crimegrade_safety_percentile": 56,
        "crimegrade_rate_per_1000": 23.51,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-11375/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "11432": {
        "crimegrade_overall_grade": "C+",
        "crimegrade_violent_grade": "C-",
        "crimegrade_property_grade": "C",
        "crimegrade_safety_percentile": 49,
        "crimegrade_rate_per_1000": 25.97,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-11432/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "10029": {
        "crimegrade_overall_grade": "C",
        "crimegrade_violent_grade": "D-",
        "crimegrade_property_grade": "C",
        "crimegrade_safety_percentile": 44,
        "crimegrade_rate_per_1000": 27.69,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-10029/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "98136": {
        "crimegrade_overall_grade": "D-",
        "crimegrade_violent_grade": "C-",
        "crimegrade_property_grade": "F",
        "crimegrade_safety_percentile": 11,
        "crimegrade_rate_per_1000": 52.37,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-98136/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "98122": {
        "crimegrade_overall_grade": "F",
        "crimegrade_violent_grade": "F",
        "crimegrade_property_grade": "F",
        "crimegrade_safety_percentile": 3,
        "crimegrade_rate_per_1000": 77.75,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-98122/",
        "crimegrade_note": "Public ZIP-level page accessed March 30, 2026.",
    },
    "98104": {
        "crimegrade_overall_grade": "F",
        "crimegrade_violent_grade": "F",
        "crimegrade_property_grade": "F",
        "crimegrade_safety_percentile": 1,
        "crimegrade_rate_per_1000": 189.30,
        "crimegrade_url": "https://crimegrade.org/safest-places-in-98104/",
        "crimegrade_note": "Downtown visitor-heavy ZIP; public CrimeGrade page explicitly warns that high-traffic areas can look worse on a per-resident basis.",
    },
}

GRADE_TO_SAFETY_SCORE = {
    "F": 0,
    "D-": 1,
    "D": 2,
    "D+": 3,
    "C-": 4,
    "C": 5,
    "C+": 6,
    "B-": 7,
    "B": 8,
    "B+": 9,
    "A-": 10,
    "A": 11,
    "A+": 12,
}

BAND_ORDER = {"low": 0, "mid": 1, "high": 2}
CITY_COLORS = {
    "Austin": "#d97706",
    "Chicago": "#2563eb",
    "New York": "#7c3aed",
    "Seattle": "#059669",
}


def _load_model_sample_frame(repo_root: Path) -> pd.DataFrame:
    score = pd.read_parquet(
        repo_root / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet",
        columns=[
            "block_group_geoid",
            "state_fips",
            "population_2024",
            "expected_count_total",
            "index_total_part1_resident",
        ],
    )
    population = pd.to_numeric(score["population_2024"], errors="coerce").replace(0.0, float("nan"))
    score["rate_total_part1_resident"] = (
        pd.to_numeric(score["expected_count_total"], errors="coerce") / population * 100_000.0
    )
    xw = pd.read_parquet(
        repo_root / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        columns=["block_group_geoid", "jurisdiction_id", "allocation_share"],
    )
    xw = (
        xw.sort_values(["block_group_geoid", "allocation_share"], ascending=[True, False])
        .drop_duplicates("block_group_geoid")
    )
    score = score.merge(xw, on="block_group_geoid", how="left")

    sample = pd.DataFrame(MODEL_SAMPLE_ROWS)
    rows: list[pd.DataFrame] = []
    for city_name, jurisdiction_id in CITY_JURISDICTIONS.items():
        city_score = score[score["jurisdiction_id"] == jurisdiction_id].copy()
        city_score["model_within_city_percentile"] = city_score["index_total_part1_resident"].rank(
            pct=True,
            method="first",
        )
        keep = sample[sample["city_name"] == city_name][["block_group_geoid", "sample_band"]]
        city_rows = city_score.merge(keep, on="block_group_geoid", how="inner")
        city_rows["city_name"] = city_name
        rows.append(city_rows)

    model_df = pd.concat(rows, ignore_index=True)
    model_df = sample.merge(
        model_df[
            [
                "city_name",
                "sample_band",
                "block_group_geoid",
                "state_fips",
                "population_2024",
                "rate_total_part1_resident",
                "index_total_part1_resident",
                "model_within_city_percentile",
            ]
        ],
        on=["city_name", "sample_band", "block_group_geoid"],
        how="left",
    )
    model_df["model_sample_usable"] = model_df["index_total_part1_resident"].notna() & model_df["rate_total_part1_resident"].notna()
    model_df["model_sample_exclusion_reason"] = ""
    model_df.loc[
        ~model_df["model_sample_usable"],
        "model_sample_exclusion_reason",
    ] = "current_release_does_not_publish_resident_rate_or_index_for_this_bg"
    return model_df


def _assemble_comparison_frame(repo_root: Path) -> pd.DataFrame:
    df = _load_model_sample_frame(repo_root)
    crimegrade = pd.DataFrame(
        [{"zip_code": zip_code, **values} for zip_code, values in CRIMEGRADE_SAMPLE_ROWS.items()]
    )
    df = df.merge(crimegrade, on="zip_code", how="left")
    if df["crimegrade_overall_grade"].isna().any():
        missing = df.loc[df["crimegrade_overall_grade"].isna(), "zip_code"].tolist()
        raise SystemExit(f"Missing CrimeGrade rows for ZIPs: {missing}")

    df["model_within_city_percentile"] = df["model_within_city_percentile"] * 100.0
    df["crimegrade_safety_score"] = df["crimegrade_overall_grade"].map(GRADE_TO_SAFETY_SCORE)
    df["crimegrade_danger_percentile"] = 100.0 - df["crimegrade_safety_percentile"]
    df["sample_band_order"] = df["sample_band"].map(BAND_ORDER)
    df["zip_page_accessed_date"] = "2026-03-30"
    return df.sort_values(["city_name", "sample_band_order"]).reset_index(drop=True)


def _build_city_alignment_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for city_name, city_df in df.groupby("city_name", sort=False):
        if len(city_df) < 2:
            labels.append({"city_name": city_name, "city_order_alignment": "insufficient_current_model_samples"})
            continue
        scores = city_df.sort_values("sample_band_order")["crimegrade_safety_score"].tolist()
        monotonic = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        if monotonic and len(set(scores)) == len(scores):
            label = "ordered"
        elif monotonic:
            label = "ordered_with_ties"
        else:
            label = "not_ordered"
        labels.append({"city_name": city_name, "city_order_alignment": label})
    return pd.DataFrame(labels)


def _write_markdown_summary(df: pd.DataFrame, out_path: Path) -> None:
    usable = df[df["model_sample_usable"].eq(True)].copy()
    excluded = df[~df["model_sample_usable"].eq(True)].copy()
    if usable.empty:
        raise SystemExit("No publishable model sample rows remain for CrimeGrade comparison.")
    spearman = usable["model_within_city_percentile"].corr(
        usable["crimegrade_danger_percentile"],
        method="spearman",
    )
    pearson = usable["model_within_city_percentile"].corr(usable["crimegrade_danger_percentile"])
    city_alignment = _build_city_alignment_labels(usable)
    usable = usable.merge(city_alignment, on="city_name", how="left")
    ordered_cities = city_alignment[
        city_alignment["city_order_alignment"].isin(["ordered", "ordered_with_ties"])
    ]["city_name"].tolist()
    insufficient_cities = city_alignment[
        city_alignment["city_order_alignment"].eq("insufficient_current_model_samples")
    ]["city_name"].tolist()

    lines = [
        "# CrimeGrade Comparator Sample",
        "",
        "This note records a bounded public comparator check against CrimeGrade's ZIP-level pages.",
        "It is not a ground-truth validation. The comparison is deliberately small and only asks",
        "whether our block-group surface points in roughly the same direction as a public",
        "consumer-facing alternative when both are reduced to a few interpretable locations.",
        "",
        "## Method",
        "",
        "1. Start from the final `2024` AGS-core block-group output in",
        "   `repro/state/output/crimerisk_block_group_2024_ags_core.parquet`.",
        "2. Restrict to four benchmark cities with high-value public comparators: Austin, Chicago,",
        "   New York, and Seattle.",
        "3. Select three block groups per city: a lower-tail point, a median point, and an upper-tail",
        "   point. The lower and median points come from the approximate 15th and 50th percentiles",
        "   of that city's BG index distribution. The upper point comes from the upper tail near the",
        "   95th percentile so the public comparator has a chance to separate clearly from the median.",
        "4. Resolve each sampled BG to a representative point using the TIGER block-group geometry",
        "   already used by the project, then map that point to a ZIP code.",
        "5. Pull the public CrimeGrade ZIP page for that ZIP and record the overall, violent, and",
        "   property grades, the reported safety percentile, and the reported total crime rate per",
        "   1,000 residents.",
        "",
        "The unavoidable limitation is scale mismatch: our surface is block-group level and",
        "CrimeGrade is published here at ZIP level. A noisy or mixed ZIP can easily blur a good BG",
        "ranking. Downtown and visitor-heavy ZIPs also behave poorly under per-resident crime rates,",
        "and CrimeGrade itself warns about that on several of the sampled pages.",
        "",
        "## Headline Read",
        "",
        f"- Across the {len(usable)} currently publishable sampled points, the correlation between our within-city percentile and CrimeGrade's ZIP-level danger percentile is `Spearman = {spearman:.3f}` and `Pearson = {pearson:.3f}`.",
        f"- `{len(excluded)}` originally selected sample points are retained in the CSV but excluded from the current read because the rebuilt surface does not publish a resident rate or index for those BGs under the denominator policy.",
        f"- Among cities with at least two publishable current samples, `{', '.join(ordered_cities) if ordered_cities else 'none'}` preserve monotonic or near-monotonic ZIP-level danger ordering.",
        f"- Cities with fewer than two publishable current samples are not assigned an ordering read: `{', '.join(insufficient_cities) if insufficient_cities else 'none'}`.",
        "- The exercise is still useful because it shows that our surface is not wildly disconnected",
        "  from the main free public alternative, but it also confirms that ZIP-level black-box maps",
        "  are too coarse to treat as adjudication for a BG-level model.",
        "",
        "## City-by-City Notes",
        "",
    ]

    if not excluded.empty:
        lines.extend(["## Excluded Current-Release Sample Points", ""])
        for row in excluded.sort_values(["city_name", "sample_band_order"]).itertuples():
            lines.append(
                f"- {row.city_name} `{row.sample_band}` sample BG `{row.block_group_geoid}` "
                f"was excluded: {row.model_sample_exclusion_reason}."
            )
        lines.append("")

    for city_name, city_df in usable.groupby("city_name", sort=False):
        ordered = city_df["city_order_alignment"].iloc[0]
        lines.append(f"### {city_name}")
        lines.append("")
        if ordered == "ordered":
            lines.append(
                "- CrimeGrade's ZIP-level ordering matches the low / mid / high ordering of our sampled BGs."
            )
        elif ordered == "ordered_with_ties":
            lines.append(
                "- CrimeGrade broadly agrees with the ordering, but at least two sampled points collapse to the same ZIP-grade bucket."
            )
        elif ordered == "insufficient_current_model_samples":
            lines.append(
                "- Only one current publishable sample point remains for this city, so no city ordering claim is made."
            )
        else:
            lines.append(
                "- CrimeGrade does not preserve the full low / mid / high ordering for this city, which likely reflects ZIP aggregation and mixed land-use / visitor effects rather than a direct BG-level contradiction."
            )
        for row in city_df.sort_values("sample_band_order").itertuples():
            lines.append(
                f"- `{row.sample_band}` sample: BG `{row.block_group_geoid}` in ZIP `{row.zip_code}`. "
                f"Our BG index is `{row.index_total_part1_resident:.1f}` at the `{row.model_within_city_percentile:.1f}`th within-city percentile. "
                f"CrimeGrade reports overall `{row.crimegrade_overall_grade}`, violent `{row.crimegrade_violent_grade}`, "
                f"property `{row.crimegrade_property_grade}`, safety percentile `{int(row.crimegrade_safety_percentile)}`, "
                f"and crime rate `{row.crimegrade_rate_per_1000:.2f}` per 1,000."
            )
            if row.crimegrade_note:
                lines.append(f"- Note: {row.crimegrade_note}")
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "- `materials/tables/crimegrade_sample_comparison.csv`: machine-readable joined sample table",
            "- `materials/figures/crimegrade_sample_scatter.png`: sampled-point scatter against CrimeGrade danger percentile",
            "- `materials/figures/crimegrade_sample_city_ordering.png`: low/mid/high ordering chart by city",
            "",
            "## Interpretation Rule",
            "",
            "Use this only as a bounded external comparator. If a sampled point disagrees with CrimeGrade,",
            "that does not prove our BG estimate is wrong; it often only proves that ZIP aggregation,",
            "visitor-heavy geography, and proprietary modeling choices differ from our block-group surface.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def _write_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["model_sample_usable"].eq(True)].copy()
    if df.empty:
        raise SystemExit("No publishable model sample rows remain for CrimeGrade figures.")

    fig, ax = plt.subplots(figsize=(8, 6))
    for city_name, city_df in df.groupby("city_name", sort=False):
        ax.scatter(
            city_df["model_within_city_percentile"],
            city_df["crimegrade_danger_percentile"],
            label=city_name,
            s=70,
            color=CITY_COLORS.get(city_name),
            alpha=0.85,
        )
        for row in city_df.itertuples():
            ax.annotate(
                f"{city_name.split()[0]}-{row.sample_band[0].upper()}",
                (row.model_within_city_percentile, row.crimegrade_danger_percentile),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )
    ax.set_xlabel("Our sampled BG within-city percentile")
    ax.set_ylabel("CrimeGrade danger percentile (100 - safety percentile)")
    ax.set_title("CrimeGrade ZIP comparator against sampled BG percentiles")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "crimegrade_sample_scatter.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes = axes.flatten()
    for ax, (city_name, city_df) in zip(axes, df.groupby("city_name", sort=False)):
        city_df = city_df.sort_values("sample_band_order")
        ax.plot(
            city_df["sample_band_order"],
            city_df["crimegrade_safety_score"],
            marker="o",
            color=CITY_COLORS.get(city_name),
            linewidth=2,
        )
        ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
        ax.set_title(city_name)
        ax.set_ylim(-0.5, 12.5)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("CrimeGrade safety score")
    axes[2].set_ylabel("CrimeGrade safety score")
    fig.suptitle("CrimeGrade safety ordering across sampled low / mid / high BGs", y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "crimegrade_sample_city_ordering.png", dpi=200)
    plt.close(fig)


def build_crimegrade_sample_comparison(*, repo_root: Path, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "materials" / "tables"
    figures_dir = out_dir / "materials" / "figures"
    report_dir = out_dir / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = _assemble_comparison_frame(repo_root)
    df.to_csv(tables_dir / "crimegrade_sample_comparison.csv", index=False)
    _write_markdown_summary(df, report_dir / "CrimeGrade_Sample_Comparison.md")
    _write_figures(df, figures_dir)
    return {
        "table_path": str(tables_dir / "crimegrade_sample_comparison.csv"),
        "report_path": str(report_dir / "CrimeGrade_Sample_Comparison.md"),
        "scatter_path": str(figures_dir / "crimegrade_sample_scatter.png"),
        "ordering_path": str(figures_dir / "crimegrade_sample_city_ordering.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a bounded CrimeGrade ZIP-level comparator bundle for the submission package."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_crimegrade_sample_comparison(
        repo_root=REPO_ROOT,
        out_dir=args.out_dir.resolve(),
    )
    print(summary)


if __name__ == "__main__":
    main()
