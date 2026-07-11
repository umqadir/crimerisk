from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_share_benchmark import build_city_share_diagnostics, weighted_mean  # noqa: E402
from crimerisk.crime import OFFENSES_7  # noqa: E402
from crimerisk.crosswalk_shares import normalize_block_group_allocation_shares  # noqa: E402


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type for {path}; use parquet or csv")


def _offense_value_columns(template: str, columns: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for offense in OFFENSES_7:
        col = template.format(offense=offense)
        if col in columns:
            out[str(offense)] = col
    return out


def _prepare_truth(
    *,
    city_shares_path: Path,
    bg_crosswalk_path: Path,
    year: int,
    geography: str,
    exclude_validation_case_types: tuple[str, ...],
) -> pd.DataFrame:
    city = pd.read_parquet(city_shares_path).copy()
    city = city[pd.to_numeric(city["year"], errors="coerce").eq(int(year))].copy()
    if exclude_validation_case_types and "validation_case_type" in city.columns:
        excluded = {str(value) for value in exclude_validation_case_types}
        city = city[~city["validation_case_type"].astype(str).isin(excluded)].copy()
    city["block_group_geoid"] = city["block_group_geoid"].astype("string").str.zfill(12)
    city["state_fips"] = city["state_fips"].astype("string").str.zfill(2)

    keys = [
        "city_name",
        "jurisdiction_id",
        "state_fips",
        "offense",
        "validation_case_type",
    ]
    for col in ["validation_case_type"]:
        if col not in city.columns:
            city[col] = "unknown"
    truth = (
        city.groupby([*keys, "block_group_geoid"], dropna=False, as_index=False)
        .agg(incident_count=("incident_count", "sum"))
    )
    if geography == "tract":
        bg_crosswalk = pd.read_parquet(bg_crosswalk_path)
        if "tract_id" not in bg_crosswalk.columns:
            bg_crosswalk["tract_id"] = bg_crosswalk["block_group_geoid"].astype("string").str.zfill(12).str.slice(0, 11)
        bg_to_tract = bg_crosswalk[["block_group_geoid", "tract_id"]].drop_duplicates("block_group_geoid").copy()
        bg_to_tract["block_group_geoid"] = bg_to_tract["block_group_geoid"].astype("string").str.zfill(12)
        bg_to_tract["tract_id"] = bg_to_tract["tract_id"].astype("string").str.zfill(11)
        truth = truth.merge(bg_to_tract, on="block_group_geoid", how="left")
        truth = truth[truth["tract_id"].notna()].copy()
        truth = (
            truth.groupby([*keys, "tract_id"], dropna=False, as_index=False)
            .agg(incident_count=("incident_count", "sum"))
            .rename(columns={"tract_id": "geo_id"})
        )
    else:
        truth = truth.rename(columns={"block_group_geoid": "geo_id"})

    truth["incident_total"] = truth.groupby(
        ["city_name", "jurisdiction_id", "state_fips", "offense"],
        dropna=False,
    )["incident_count"].transform("sum")
    truth = truth[pd.to_numeric(truth["incident_total"], errors="coerce").fillna(0.0).gt(0)].copy()
    truth["true_share"] = pd.to_numeric(truth["incident_count"], errors="coerce").fillna(0.0) / pd.to_numeric(
        truth["incident_total"],
        errors="coerce",
    ).fillna(np.nan)
    return truth


def _prepare_universe(
    *,
    truth: pd.DataFrame,
    bg_crosswalk_path: Path,
    geography: str,
) -> pd.DataFrame:
    crosswalk = normalize_block_group_allocation_shares(pd.read_parquet(bg_crosswalk_path).copy())
    crosswalk["block_group_geoid"] = crosswalk["block_group_geoid"].astype("string").str.zfill(12)
    if "tract_id" in crosswalk.columns:
        crosswalk["tract_id"] = crosswalk["tract_id"].astype("string").str.zfill(11)
    else:
        crosswalk["tract_id"] = crosswalk["block_group_geoid"].str.slice(0, 11)
    crosswalk["state_fips"] = crosswalk["state_fips"].astype("string").str.zfill(2)
    jurisdictions = truth[
        ["city_name", "jurisdiction_id", "state_fips", "offense", "validation_case_type"]
    ].drop_duplicates()
    all_geos = crosswalk[["state_fips", "block_group_geoid", "tract_id"]].drop_duplicates().copy()

    if geography == "tract":
        universe = crosswalk[["jurisdiction_id", "state_fips", "tract_id"]].drop_duplicates().copy()
        universe["geo_id"] = universe["tract_id"].astype("string").str.zfill(11)
        universe = universe.drop(columns=["tract_id"])
    else:
        universe = crosswalk[["jurisdiction_id", "state_fips", "block_group_geoid"]].drop_duplicates().copy()
        universe["geo_id"] = universe["block_group_geoid"].astype("string").str.zfill(12)
        universe = universe.drop(columns=["block_group_geoid"])

    universe = universe.merge(jurisdictions, on=["jurisdiction_id", "state_fips"], how="inner")
    county_jurisdictions = jurisdictions[
        jurisdictions["jurisdiction_id"].astype(str).str.contains(":county:", regex=False, na=False)
    ].copy()
    county_frames: list[pd.DataFrame] = []
    for row in county_jurisdictions.itertuples(index=False):
        county_fips = str(row.jurisdiction_id).rsplit(":", 1)[-1]
        if not county_fips.isdigit() or len(county_fips) != 5:
            continue
        if geography == "tract":
            county_geos = all_geos[all_geos["tract_id"].astype(str).str.startswith(county_fips)].copy()
            county_geos["geo_id"] = county_geos["tract_id"]
        else:
            county_geos = all_geos[all_geos["block_group_geoid"].astype(str).str.startswith(county_fips)].copy()
            county_geos["geo_id"] = county_geos["block_group_geoid"]
        if county_geos.empty:
            continue
        county_frames.append(
            pd.DataFrame(
                {
                    "jurisdiction_id": str(row.jurisdiction_id),
                    "state_fips": str(row.state_fips).zfill(2),
                    "geo_id": county_geos["geo_id"].drop_duplicates().astype("string"),
                    "city_name": str(row.city_name),
                    "offense": str(row.offense),
                    "validation_case_type": str(row.validation_case_type),
                }
            )
        )
    if county_frames:
        universe = pd.concat([universe, *county_frames], ignore_index=True, sort=False).drop_duplicates(
            ["city_name", "jurisdiction_id", "state_fips", "offense", "validation_case_type", "geo_id"]
        )
    truth_geo = truth[
        [
            "city_name",
            "jurisdiction_id",
            "state_fips",
            "offense",
            "validation_case_type",
            "geo_id",
            "incident_count",
            "incident_total",
            "true_share",
        ]
    ].copy()
    frame = universe.merge(
        truth_geo,
        on=["city_name", "jurisdiction_id", "state_fips", "offense", "validation_case_type", "geo_id"],
        how="left",
    )
    frame["incident_count"] = pd.to_numeric(frame["incident_count"], errors="coerce").fillna(0.0)
    frame["incident_total"] = frame.groupby(
        ["city_name", "jurisdiction_id", "state_fips", "offense"],
        dropna=False,
    )["incident_count"].transform("sum")
    frame["true_share"] = pd.to_numeric(frame["true_share"], errors="coerce").fillna(0.0)
    return frame


def _prepare_external_long(
    *,
    surface_path: Path,
    geography: str,
    geo_id_col: str,
    value_template: str,
    value_kind: str,
    population_col: str | None,
) -> pd.DataFrame:
    surface = _read_table(surface_path)
    if geo_id_col not in surface.columns:
        raise ValueError(f"{surface_path} missing geo id column {geo_id_col!r}")
    value_cols = _offense_value_columns(value_template, set(surface.columns))
    if not value_cols:
        raise ValueError(
            f"{surface_path} has none of the offense columns implied by template {value_template!r}"
        )
    if value_kind == "rate" and (not population_col or population_col not in surface.columns):
        raise ValueError("--value-kind rate requires --population-col present in the surface")

    id_width = 11 if geography == "tract" else 12
    surface = surface.rename(columns={geo_id_col: "geo_id"}).copy()
    surface["geo_id"] = surface["geo_id"].astype("string").str.zfill(id_width)
    long_rows = []
    for offense, col in value_cols.items():
        part = surface[["geo_id", col] + ([population_col] if value_kind == "rate" and population_col else [])].copy()
        value = pd.to_numeric(part[col], errors="coerce").fillna(0.0)
        if value_kind == "rate":
            pop = pd.to_numeric(part[population_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            value = value.clip(lower=0.0) * pop / 1000.0
        elif value_kind in {"count", "score"}:
            value = value.clip(lower=0.0)
        else:
            raise ValueError(f"Unsupported value kind: {value_kind}")
        long_rows.append(pd.DataFrame({"geo_id": part["geo_id"], "offense": offense, "external_value": value}))
    return pd.concat(long_rows, ignore_index=True, sort=False)


def build_external_surface_benchmark(
    *,
    surface_path: Path,
    surface_name: str,
    geography: str,
    geo_id_col: str,
    value_template: str,
    value_kind: str,
    population_col: str | None,
    city_shares_path: Path,
    bg_crosswalk_path: Path,
    year: int,
    exclude_validation_case_types: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    truth = _prepare_truth(
        city_shares_path=city_shares_path,
        bg_crosswalk_path=bg_crosswalk_path,
        year=year,
        geography=geography,
        exclude_validation_case_types=exclude_validation_case_types,
    )
    universe = _prepare_universe(truth=truth, bg_crosswalk_path=bg_crosswalk_path, geography=geography)
    external = _prepare_external_long(
        surface_path=surface_path,
        geography=geography,
        geo_id_col=geo_id_col,
        value_template=value_template,
        value_kind=value_kind,
        population_col=population_col,
    )
    frame = universe.merge(external, on=["geo_id", "offense"], how="left")
    frame["external_value"] = pd.to_numeric(frame["external_value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    totals = frame.groupby(["city_name", "jurisdiction_id", "state_fips", "offense"], dropna=False)[
        "external_value"
    ].transform("sum")
    frame["external_share"] = np.where(totals.gt(0), frame["external_value"] / totals, 0.0)
    frame = frame.rename(columns={"geo_id": "bg_id"})
    diagnostics, base_summary = build_city_share_diagnostics(frame, predicted_share_col="external_share")
    case_type = (
        truth[["city_name", "jurisdiction_id", "state_fips", "offense", "validation_case_type"]]
        .drop_duplicates()
        .copy()
    )
    diagnostics = diagnostics.merge(
        case_type,
        on=["city_name", "jurisdiction_id", "state_fips", "offense"],
        how="left",
    )
    diagnostics.insert(0, "surface_name", surface_name)
    diagnostics.insert(1, "geography", geography)
    diagnostics.insert(2, "value_kind", value_kind)
    diagnostics.insert(3, "surface_path", str(surface_path))

    summary = {
        "year": int(year),
        "surface_name": surface_name,
        "surface_path": str(surface_path),
        "geography": geography,
        "geo_id_col": geo_id_col,
        "value_template": value_template,
        "value_kind": value_kind,
        "population_col": population_col,
        "truth_case_count": int(truth["jurisdiction_id"].nunique()) if not truth.empty else 0,
        "rows": int(len(diagnostics)),
        "incident_total": float(pd.to_numeric(diagnostics.get("incident_total"), errors="coerce").fillna(0.0).sum())
        if not diagnostics.empty
        else 0.0,
        "weighted_total_variation_distance_mean": weighted_mean(
            diagnostics, "total_variation_distance", "incident_total"
        ),
        "weighted_share_rmse_mean": weighted_mean(diagnostics, "share_rmse", "incident_total"),
        "weighted_pearson_share_mean": weighted_mean(diagnostics, "pearson_share", "incident_total"),
        "weighted_spearman_share_mean": weighted_mean(diagnostics, "spearman_share", "incident_total"),
        "weighted_top_10pct_true_mass_in_model_top_10pct_mean": weighted_mean(
            diagnostics,
            "top_10pct_true_mass_in_model_top_10pct",
            "incident_total",
        ),
        "by_validation_case_type": [],
        "base_summary": base_summary,
    }
    if not diagnostics.empty and "validation_case_type" in diagnostics.columns:
        for case_type_value, group in diagnostics.groupby("validation_case_type", dropna=False):
            summary["by_validation_case_type"].append(
                {
                    "validation_case_type": str(case_type_value),
                    "rows": int(len(group)),
                    "incident_total": float(pd.to_numeric(group["incident_total"], errors="coerce").fillna(0.0).sum()),
                    "weighted_total_variation_distance_mean": weighted_mean(
                        group,
                        "total_variation_distance",
                        "incident_total",
                    ),
                    "weighted_top_10pct_true_mass_in_model_top_10pct_mean": weighted_mean(
                        group,
                        "top_10pct_true_mass_in_model_top_10pct",
                        "incident_total",
                    ),
                }
            )
    return diagnostics, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score an external BG or tract crime-risk surface against validation incident shares."
    )
    parser.add_argument("--surface-path", type=Path, required=True)
    parser.add_argument("--surface-name", required=True)
    parser.add_argument("--geography", choices=["block_group", "tract"], required=True)
    parser.add_argument("--geo-id-col", default=None)
    parser.add_argument("--value-template", default="count_{offense}")
    parser.add_argument("--value-kind", choices=["count", "rate", "score"], default="count")
    parser.add_argument("--population-col", default=None)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--city-shares-path",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "next_phase_validation_city_incident_share_surface_2024.parquet",
    )
    parser.add_argument(
        "--bg-crosswalk-path",
        type=Path,
        default=REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
    )
    parser.add_argument(
        "--exclude-validation-case-type",
        action="append",
        default=[],
        help="validation_case_type to exclude from scoring; may be repeated.",
    )
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "external_surface_benchmark.parquet",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "external_surface_benchmark.json",
    )
    parser.add_argument(
        "--summary-csv-out",
        type=Path,
        default=REPO_ROOT / "materials" / "tables" / "external_surface_benchmark_summary.csv",
    )
    args = parser.parse_args()

    geography = str(args.geography)
    geo_id_col = args.geo_id_col or ("tract_id" if geography == "tract" else "block_group_geoid")
    diagnostics, summary = build_external_surface_benchmark(
        surface_path=args.surface_path,
        surface_name=str(args.surface_name),
        geography=geography,
        geo_id_col=geo_id_col,
        value_template=str(args.value_template),
        value_kind=str(args.value_kind),
        population_col=args.population_col,
        city_shares_path=args.city_shares_path,
        bg_crosswalk_path=args.bg_crosswalk_path,
        year=int(args.year),
        exclude_validation_case_types=tuple(str(value) for value in args.exclude_validation_case_type),
    )
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(args.diagnostics_out, index=False)
    pd.DataFrame([summary]).drop(columns=["base_summary", "by_validation_case_type"], errors="ignore").to_csv(
        args.summary_csv_out,
        index=False,
    )
    args.summary_json_out.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
