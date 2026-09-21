#!/usr/bin/env python3
"""Describe before/after changes along a bounded set of geographic seams.

This diagnostic does not decide that a boundary jump is erroneous. It keeps
numerator, denominator, index, zero/missing status, and reporting provenance
separate so a reviewer can inspect why an adjacent pair differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


OFFENSES = (
    "murder", "rape", "robbery", "aggravated_assault", "burglary", "larceny",
    "motor_vehicle_theft",
)
RARE_OFFENSES = {"murder", "rape"}
CONUS_STATE_FIPS = {
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13", "16", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41",
    "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56",
}
METRO_BOXES = {
    "metro_dc_md_va": (-77.65, -76.85, 38.70, 39.15),
    "metro_kansas_city": (-95.05, -94.30, 38.80, 39.45),
    "metro_cincinnati": (-84.85, -84.20, 38.90, 39.42),
    "metro_st_louis": (-90.65, -89.85, 38.40, 38.92),
}
NUMERIC_ATOL = 1e-12
NUMERIC_RTOL = 1e-10
DECOMPOSITION_ATOL = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    keys = (frame["bg_a"] + "|" + frame["bg_b"]).map(
        lambda value: hashlib.sha256(value.encode()).hexdigest()
    )
    return frame.assign(_sample_key=keys).sort_values("_sample_key").head(n).drop(columns="_sample_key")


def load_surface(path: Path, support: str) -> pd.DataFrame:
    id_col = "block_group_geoid" if support == "block_group" else "tract_id"
    names = pq.ParquetFile(path).schema_arrow.names
    fixed = [id_col, "tract_id", "state_fips", "eb_jurisdiction_id", "dominant_eb_jurisdiction_id",
             "urban_stratum", "population_2025", "land_area_sq_mi"]
    per_offense = []
    for offense in OFFENSES:
        per_offense.extend([
            f"expected_count_{offense}", f"primary_denominator_{offense}",
            f"index_{offense}_primary", f"source_mode_{offense}",
            f"level_provenance_text_{offense}", f"footprint_derived_count_share_{offense}",
        ])
    columns = [column for column in dict.fromkeys([*fixed, *per_offense]) if column in names]
    frame = pd.read_parquet(path, columns=columns)
    frame[id_col] = frame[id_col].astype("string")
    if frame[id_col].duplicated().any():
        raise ValueError(f"duplicate {id_col} in {path}")
    return frame


def _add_tag(tags: dict[tuple[str, str], set[str]], frame: pd.DataFrame, tag: str) -> None:
    for row in frame[["bg_a", "bg_b"]].itertuples(index=False):
        tags.setdefault((row.bg_a, row.bg_b), set()).add(tag)


def build_bg_pair_plan(
    adjacency: pd.DataFrame,
    centroids: pd.DataFrame,
    baseline_bg: pd.DataFrame,
    tribal_ids: set[str],
    *,
    rural_per_state_pair: int,
    navajo_ids: set[str] | None = None,
) -> pd.DataFrame:
    adj = adjacency.rename(columns={"bg": "bg_a", "nb": "bg_b"})[["bg_a", "bg_b"]].copy()
    for column in ("bg_a", "bg_b"):
        adj[column] = adj[column].astype("string").str.zfill(12)
    adj = adj[adj["bg_a"].lt(adj["bg_b"])].drop_duplicates().copy()
    adj = adj[
        adj["bg_a"].str[:2].isin(CONUS_STATE_FIPS)
        & adj["bg_b"].str[:2].isin(CONUS_STATE_FIPS)
    ]

    lookup = baseline_bg.set_index("block_group_geoid")
    for column in ("state_fips", "eb_jurisdiction_id", "urban_stratum", "tract_id"):
        if column not in lookup:
            raise ValueError(f"baseline block-group surface requires {column}")
        adj[f"{column}_a"] = adj["bg_a"].map(lookup[column])
        adj[f"{column}_b"] = adj["bg_b"].map(lookup[column])
    adj = adj.dropna(subset=["state_fips_a", "state_fips_b"])

    centroid = centroids.rename(columns={"bg_id": "block_group_geoid"}).copy()
    centroid["block_group_geoid"] = centroid["block_group_geoid"].astype("string").str.zfill(12)
    coords = centroid.set_index("block_group_geoid")[["lon", "lat"]]
    for side in ("a", "b"):
        adj[f"lon_{side}"] = adj[f"bg_{side}"].map(coords["lon"])
        adj[f"lat_{side}"] = adj[f"bg_{side}"].map(coords["lat"])

    cross_state = adj["state_fips_a"].astype(str).ne(adj["state_fips_b"].astype(str))
    same_jurisdiction = adj["eb_jurisdiction_id_a"].astype(str).eq(adj["eb_jurisdiction_id_b"].astype(str))
    same_stratum = adj["urban_stratum_a"].astype(str).eq(adj["urban_stratum_b"].astype(str))
    tags: dict[tuple[str, str], set[str]] = {}

    for name, (west, east, south, north) in METRO_BOXES.items():
        in_box = (
            adj["lon_a"].between(west, east) & adj["lon_b"].between(west, east)
            & adj["lat_a"].between(south, north) & adj["lat_b"].between(south, north)
        )
        boundary = adj[in_box & cross_state]
        _add_tag(tags, boundary, name)
        control = adj[in_box & ~cross_state & same_jurisdiction & same_stratum]
        _add_tag(tags, stable_sample(control, len(boundary)), f"control_{name}")

    tribal_a = adj["bg_a"].isin(tribal_ids)
    tribal_b = adj["bg_b"].isin(tribal_ids)
    tribal_boundary = adj[tribal_a ^ tribal_b]
    _add_tag(tags, tribal_boundary, "custom_tribal_perimeter")
    tribal_control = adj[tribal_a & tribal_b & same_jurisdiction & same_stratum]
    _add_tag(tags, stable_sample(tribal_control, len(tribal_boundary)), "control_custom_tribal")
    if navajo_ids:
        navajo_touching = adj[adj["bg_a"].isin(navajo_ids) | adj["bg_b"].isin(navajo_ids)]
        _add_tag(tags, navajo_touching[cross_state.loc[navajo_touching.index]], "navajo_cross_state")

    rural = (
        adj["urban_stratum_a"].astype(str).eq("rural")
        & adj["urban_stratum_b"].astype(str).eq("rural")
    )
    rural_boundary = adj[rural & cross_state].copy()
    rural_boundary["state_pair"] = rural_boundary.apply(
        lambda row: "-".join(sorted((str(row.state_fips_a).zfill(2), str(row.state_fips_b).zfill(2)))), axis=1
    )
    selected_rural = pd.concat(
        [stable_sample(group, rural_per_state_pair) for _, group in rural_boundary.groupby("state_pair")],
        ignore_index=True,
    ) if len(rural_boundary) else rural_boundary
    _add_tag(tags, selected_rural, "cross_state_rural")
    rural_control = adj[rural & ~cross_state & same_jurisdiction]
    for state, group in rural_control.groupby(adj.loc[rural_control.index, "state_fips_a"]):
        target = int((selected_rural["state_fips_a"].astype(str).eq(str(state))
                      | selected_rural["state_fips_b"].astype(str).eq(str(state))).sum())
        if target:
            _add_tag(tags, stable_sample(group, target), "control_rural")

    rows = [{"bg_a": a, "bg_b": b, "boundary_tags": ";".join(sorted(pair_tags))}
            for (a, b), pair_tags in sorted(tags.items())]
    return pd.DataFrame(rows, columns=["bg_a", "bg_b", "boundary_tags"])


def derive_support_pairs(bg_pairs: pd.DataFrame, baseline_bg: pd.DataFrame, support: str) -> pd.DataFrame:
    if support == "block_group":
        return bg_pairs.rename(columns={"bg_a": "support_a", "bg_b": "support_b"}).copy()
    tract = baseline_bg.set_index("block_group_geoid")["tract_id"].astype("string")
    result = bg_pairs.copy()
    result["support_a"] = result["bg_a"].map(tract)
    result["support_b"] = result["bg_b"].map(tract)
    result = result.dropna(subset=["support_a", "support_b"])
    flip = result["support_a"].gt(result["support_b"])
    result.loc[flip, ["support_a", "support_b"]] = result.loc[
        flip, ["support_b", "support_a"]
    ].to_numpy()
    result = result[result["support_a"].ne(result["support_b"])]
    grouped = result.groupby(["support_a", "support_b"], sort=True)["boundary_tags"].agg(
        lambda values: ";".join(sorted({tag for value in values for tag in value.split(";")}))
    )
    return grouped.reset_index()


def pair_metric(left: object, right: object) -> tuple[str, float | None, float | None]:
    values = pd.to_numeric(pd.Series([left, right]), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return "missing", None, None
    if (values < 0).any():
        return "negative", None, None
    if (values == 0).all():
        return "both_zero", None, None
    if (values == 0).any():
        return "one_zero", None, None
    signed = float(np.log(values[1] / values[0]))
    return "positive", signed, abs(signed)


def evaluate_pairs(
    pairs: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    support: str,
    offense: str,
) -> pd.DataFrame:
    id_col = "block_group_geoid" if support == "block_group" else "tract_id"
    base = baseline.set_index(id_col)
    cand = candidate.set_index(id_col)
    count_col = f"expected_count_{offense}"
    denominator_col = f"primary_denominator_{offense}"
    index_col = f"index_{offense}_primary"
    required = (count_col, denominator_col, index_col)
    for label, frame in (("baseline", base), ("candidate", cand)):
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{label} {support} surface missing {missing}")

    records = []
    for pair in pairs.itertuples(index=False):
        row: dict[str, object] = {
            "support": support, "offense": offense, "support_a": pair.support_a,
            "support_b": pair.support_b, "boundary_tags": pair.boundary_tags,
        }
        for version, frame in (("baseline", base), ("candidate", cand)):
            for metric, column in (("count", count_col), ("denominator", denominator_col), ("index", index_col)):
                value_a = frame[column].get(pair.support_a)
                value_b = frame[column].get(pair.support_b)
                row[f"{version}_{metric}_a"] = value_a
                row[f"{version}_{metric}_b"] = value_b
                category, signed, absolute = pair_metric(
                    value_a, value_b
                )
                row[f"{version}_{metric}_status"] = category
                row[f"{version}_{metric}_signed_log_ratio"] = signed
                row[f"{version}_{metric}_abs_log_ratio"] = absolute
            if all(row[f"{version}_{metric}_status"] == "positive" for metric in ("count", "denominator", "index")):
                residual = float(row[f"{version}_index_signed_log_ratio"]) - (
                    float(row[f"{version}_count_signed_log_ratio"])
                    - float(row[f"{version}_denominator_signed_log_ratio"])
                )
                row[f"{version}_decomposition_residual"] = residual
                row[f"{version}_decomposition_ok"] = abs(residual) <= DECOMPOSITION_ATOL
            else:
                row[f"{version}_decomposition_residual"] = None
                row[f"{version}_decomposition_ok"] = None

        for metric in ("count", "denominator", "index"):
            before = row[f"baseline_{metric}_signed_log_ratio"]
            after = row[f"candidate_{metric}_signed_log_ratio"]
            row[f"change_{metric}_signed_log_ratio"] = (
                float(after) - float(before) if before is not None and after is not None else None
            )
        count_changed = not _pair_values_equal(base, cand, count_col, pair.support_a, pair.support_b)
        denominator_changed = not _pair_values_equal(base, cand, denominator_col, pair.support_a, pair.support_b)
        row["count_changed"] = count_changed
        row["denominator_changed"] = denominator_changed
        row["numeric_change_driver"] = (
            "count_and_denominator" if count_changed and denominator_changed else
            "count" if count_changed else "denominator" if denominator_changed else "unchanged"
        )
        for side, support_id in (("a", pair.support_a), ("b", pair.support_b)):
            for context in (f"source_mode_{offense}", f"level_provenance_text_{offense}",
                            f"footprint_derived_count_share_{offense}",
                            "eb_jurisdiction_id", "dominant_eb_jurisdiction_id", "urban_stratum"):
                for version, frame in (("baseline", base), ("candidate", cand)):
                    if context in frame:
                        row[f"{version}_{side}_{context}"] = frame[context].get(support_id)
        records.append(row)
    return pd.DataFrame(records)


def _pair_values_equal(
    baseline: pd.DataFrame, candidate: pd.DataFrame, column: str, support_a: str, support_b: str
) -> bool:
    old = pd.to_numeric(pd.Series([baseline[column].get(support_a), baseline[column].get(support_b)]), errors="coerce")
    new = pd.to_numeric(pd.Series([candidate[column].get(support_a), candidate[column].get(support_b)]), errors="coerce")
    return bool(np.all(np.isclose(old, new, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL, equal_nan=True)))


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exploded = details.assign(boundary_tag=details["boundary_tags"].str.split(";")).explode("boundary_tag")
    for keys, group in exploded.groupby(["support", "offense", "boundary_tag"], dropna=False):
        for version in ("baseline", "candidate"):
            for metric in ("count", "denominator", "index"):
                status = group[f"{version}_{metric}_status"]
                positive = pd.to_numeric(group[f"{version}_{metric}_abs_log_ratio"], errors="coerce").dropna()
                rows.append({
                    "support": keys[0], "offense": keys[1], "boundary_tag": keys[2],
                    "version": version, "metric": metric, "pair_count": len(group),
                    "positive_pair_count": int(status.eq("positive").sum()),
                    "both_zero_pair_count": int(status.eq("both_zero").sum()),
                    "one_zero_pair_count": int(status.eq("one_zero").sum()),
                    "missing_pair_count": int(status.eq("missing").sum()),
                    "negative_pair_count": int(status.eq("negative").sum()),
                    "positive_abs_log_ratio_p50": float(positive.quantile(.5)) if len(positive) else None,
                    "positive_abs_log_ratio_p90": float(positive.quantile(.9)) if len(positive) else None,
                    "positive_abs_log_ratio_p99": float(positive.quantile(.99)) if len(positive) else None,
                })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("baseline-bg", "baseline-tract", "candidate-bg", "candidate-tract",
                 "adjacency", "centroids", "tribal-overrides", "tribal-aiannh",
                 "custom-footprints", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rural-per-state-pair", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [args.baseline_bg, args.baseline_tract, args.candidate_bg, args.candidate_tract,
             args.adjacency, args.centroids, args.tribal_overrides, args.tribal_aiannh,
             args.custom_footprints]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    baseline_bg = load_surface(args.baseline_bg, "block_group")
    candidate_bg = load_surface(args.candidate_bg, "block_group")
    baseline_tract = load_surface(args.baseline_tract, "tract")
    candidate_tract = load_surface(args.candidate_tract, "tract")
    adjacency = pd.read_parquet(args.adjacency, columns=["bg", "nb"])
    centroids = pd.read_parquet(args.centroids, columns=["bg_id", "lon", "lat"])
    overrides = pd.read_csv(args.tribal_overrides, dtype=str).fillna("")
    tribal_oris = set(overrides.loc[overrides["overlap_subtype_final"].eq("tribal"), "ori"])
    footprints = pd.read_csv(args.custom_footprints, dtype=str, usecols=["ori", "block_group_geoid"])
    tribal_ids = set(footprints.loc[footprints["ori"].isin(tribal_oris), "block_group_geoid"].str.zfill(12))
    tribal_aiannh = pd.read_csv(args.tribal_aiannh, dtype=str).fillna("")
    navajo_oris = set(tribal_aiannh.loc[
        tribal_aiannh["aiannh_codes"].str.split(";").map(lambda codes: "2430" in codes), "ori"
    ])
    navajo_ids = set(footprints.loc[
        footprints["ori"].isin(navajo_oris), "block_group_geoid"
    ].str.zfill(12))
    bg_plan = build_bg_pair_plan(
        adjacency, centroids, baseline_bg, tribal_ids,
        rural_per_state_pair=args.rural_per_state_pair,
        navajo_ids=navajo_ids,
    )
    outputs = []
    for offense in OFFENSES:
        support = "tract" if offense in RARE_OFFENSES else "block_group"
        pairs = derive_support_pairs(bg_plan, baseline_bg, support)
        outputs.append(evaluate_pairs(
            pairs,
            baseline_tract if support == "tract" else baseline_bg,
            candidate_tract if support == "tract" else candidate_bg,
            support=support,
            offense=offense,
        ))
    details = pd.concat(outputs, ignore_index=True)
    summary = summarize(details)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    details.to_parquet(args.out_dir / "boundary_pair_details.parquet", index=False)
    summary.to_csv(args.out_dir / "boundary_pair_summary.csv", index=False)
    decomposition_failures = int(sum(
        details[column].eq(False).sum()
        for column in ("baseline_decomposition_ok", "candidate_decomposition_ok")
    ))
    metadata = {
        "version": "boundary_seam_comparison_v1",
        "descriptive_only": True,
        "coverage": "contiguous_us_and_dc",
        "rare_offense_support": "tract",
        "regular_offense_support": "block_group",
        "bg_pair_count": int(len(bg_plan)),
        "detail_row_count": int(len(details)),
        "decomposition_tolerance": DECOMPOSITION_ATOL,
        "decomposition_failure_count": decomposition_failures,
        "inputs": {str(path.resolve()): sha256(path) for path in paths},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 1 if decomposition_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
