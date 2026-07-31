from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from crimerisk.paths import RepoPaths
from crimerisk.allocation import _load_overlap_custom_footprints, _load_overlap_footprint_overrides


SUPPORTED_TREATMENTS = {
    "localize_to_place",
    "localize_to_county",
    "keep_statewide_overlap",
    "localize_to_custom_footprint",
    "absorb_into_primary_jurisdiction",
    "exclude_or_hold",
}


def _normalize_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
            out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out


def _normalize_fips(series: pd.Series, width: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = series.astype("string").str.strip()
    out = out.where(numeric.isna(), numeric.astype("Int64").astype("string"))
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out.str.zfill(width)


def _validate_overrides(paths: RepoPaths, errors: list[str], warnings: list[str]) -> pd.DataFrame:
    overrides = _normalize_text(
        _load_overlap_footprint_overrides(paths),
        [
            "ori9",
            "final_overlap_treatment",
            "target_state_fips",
            "target_county_fips",
            "target_place_fips",
            "target_jurisdiction_id",
            "geometry_source_type",
            "geometry_source_ref",
        ],
    )
    if overrides.empty:
        warnings.append("overlap_footprint_overrides.csv is empty")
        return overrides

    invalid = overrides.loc[~overrides["final_overlap_treatment"].isin(SUPPORTED_TREATMENTS), ["ori9", "final_overlap_treatment"]]
    if not invalid.empty:
        errors.append(f"Invalid overlap treatments: {invalid.to_dict(orient='records')}")

    missing_sources = overrides.loc[
        overrides["final_overlap_treatment"].ne("keep_statewide_overlap")
        & (
            overrides["geometry_source_type"].isna()
            | overrides["geometry_source_ref"].isna()
        ),
        ["ori9", "final_overlap_treatment"],
    ]
    if not missing_sources.empty:
        errors.append(f"Non-statewide overlap overrides missing geometry source fields: {missing_sources.to_dict(orient='records')}")

    place_need = overrides["final_overlap_treatment"].eq("localize_to_place")
    county_need = overrides["final_overlap_treatment"].eq("localize_to_county")
    custom_need = overrides["final_overlap_treatment"].eq("localize_to_custom_footprint")
    absorb_need = overrides["final_overlap_treatment"].eq("absorb_into_primary_jurisdiction")

    if overrides.loc[place_need & overrides["target_state_fips"].isna(), ["ori9"]].shape[0]:
        errors.append("localize_to_place rows must include target_state_fips")
    if overrides.loc[place_need & overrides["target_place_fips"].isna() & overrides["target_jurisdiction_id"].isna(), ["ori9"]].shape[0]:
        errors.append("localize_to_place rows must include target_place_fips or target_jurisdiction_id")
    if overrides.loc[county_need & overrides["target_state_fips"].isna(), ["ori9"]].shape[0]:
        errors.append("localize_to_county rows must include target_state_fips")
    if overrides.loc[county_need & overrides["target_county_fips"].isna(), ["ori9"]].shape[0]:
        errors.append("localize_to_county rows must include target_county_fips")
    if overrides.loc[custom_need & overrides["target_state_fips"].isna(), ["ori9"]].shape[0]:
        errors.append("localize_to_custom_footprint rows must include target_state_fips")
    if overrides.loc[absorb_need & overrides["target_jurisdiction_id"].isna(), ["ori9"]].shape[0]:
        errors.append("absorb_into_primary_jurisdiction rows must include target_jurisdiction_id")

    overrides["target_state_fips"] = _normalize_fips(overrides["target_state_fips"], 2)
    overrides["target_county_fips"] = _normalize_fips(overrides["target_county_fips"], 3)
    overrides["target_place_fips"] = _normalize_fips(overrides["target_place_fips"], 5)

    overlap_oris = pd.read_parquet(
        paths.state_dir / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        columns=["ori", "state_fips", "relationship_type"],
    )
    overlap_oris = overlap_oris[overlap_oris["relationship_type"].eq("overlap")].copy()
    overlap_oris["ori9"] = overlap_oris["ori"].astype("string")
    overlap_oris["state_fips"] = overlap_oris["state_fips"].astype("string").str.zfill(2)
    overlap_ori_set = set(overlap_oris["ori9"].dropna().astype(str).tolist())
    missing_oris = overrides.loc[~overrides["ori9"].isin(overlap_ori_set), ["ori9"]]
    if not missing_oris.empty:
        errors.append(f"Overrides reference ORIs not present in overlap crosswalk: {missing_oris.to_dict(orient='records')}")

    jm = pd.read_parquet(
        paths.state_dir / "reference" / "jurisdiction_master.parquet",
        columns=["jurisdiction_id", "jurisdiction_type", "geo_type", "geoid", "state_fips"],
    )
    jm = _normalize_text(jm, ["jurisdiction_id", "jurisdiction_type", "geo_type", "geoid", "state_fips"])
    place_map = jm[(jm["jurisdiction_type"].eq("municipal")) & (jm["geo_type"].eq("place"))].copy()
    place_key_to_jid = {
        (str(r.state_fips).zfill(2), str(r.geoid).zfill(7)): str(r.jurisdiction_id)
        for r in place_map.itertuples(index=False)
    }
    valid_jids = set(jm["jurisdiction_id"].dropna().astype(str).tolist())
    municipal_jids = set(jm.loc[jm["jurisdiction_type"].eq("municipal"), "jurisdiction_id"].dropna().astype(str).tolist())

    for row in overrides.itertuples(index=False):
        treatment = row.final_overlap_treatment
        if treatment == "localize_to_place" and pd.notna(row.target_place_fips):
            geoid = f"{str(row.target_state_fips).zfill(2)}{str(row.target_place_fips).zfill(5)}"
            if (str(row.target_state_fips).zfill(2), geoid) not in place_key_to_jid and pd.isna(row.target_jurisdiction_id):
                errors.append(f"{row.ori9}: target place {geoid} does not map to a municipal place jurisdiction")
        if treatment in {"localize_to_place", "absorb_into_primary_jurisdiction"} and pd.notna(row.target_jurisdiction_id):
            if str(row.target_jurisdiction_id) not in municipal_jids:
                errors.append(f"{row.ori9}: target_jurisdiction_id {row.target_jurisdiction_id} is not a municipal jurisdiction")
        if pd.notna(row.target_jurisdiction_id) and str(row.target_jurisdiction_id) not in valid_jids:
            errors.append(f"{row.ori9}: target_jurisdiction_id {row.target_jurisdiction_id} does not exist in jurisdiction master")

    return overrides


def _validate_custom_footprints(paths: RepoPaths, overrides: pd.DataFrame, errors: list[str], warnings: list[str]) -> pd.DataFrame:
    custom = _normalize_text(
        _load_overlap_custom_footprints(paths),
        ["ori9", "state_fips", "bg_id", "geometry_source_type", "geometry_source_ref"],
    )
    if custom.empty:
        if overrides["final_overlap_treatment"].eq("localize_to_custom_footprint").any():
            warnings.append("Custom-footprint overlap treatments exist, but overlap_custom_footprints.csv is empty; those rows will stay broad overlap until custom geometry is built")
        else:
            warnings.append("overlap_custom_footprints.csv is empty")
        return custom

    bg_scope = pd.read_parquet(
        paths.state_dir / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        columns=["state_fips", "block_group_geoid"],
    ).drop_duplicates()
    bg_scope["state_fips"] = bg_scope["state_fips"].astype("string").str.zfill(2)
    bg_scope["bg_id"] = bg_scope["block_group_geoid"].astype("string").str.zfill(12)
    valid_bgs = set(zip(bg_scope["state_fips"], bg_scope["bg_id"]))

    bad_bgs = custom.loc[
        ~custom.apply(lambda r: (r["state_fips"], r["bg_id"]) in valid_bgs, axis=1),
        ["ori9", "state_fips", "bg_id"],
    ]
    if not bad_bgs.empty:
        errors.append(f"Custom footprint rows reference BGs outside the live geometry scope: {bad_bgs.head(50).to_dict(orient='records')}")

    custom_needed = set(overrides.loc[overrides["final_overlap_treatment"].eq("localize_to_custom_footprint"), "ori9"].dropna().astype(str))
    if custom_needed:
        custom_present = set(custom["ori9"].dropna().astype(str))
        missing = sorted(custom_needed - custom_present)
        if missing:
            warnings.append(f"Missing custom footprint rows for ORIs: {missing}; those rows will stay broad overlap until custom geometry is built")
    unused = sorted(set(custom["ori9"].dropna().astype(str)) - custom_needed)
    if unused:
        warnings.append(f"Custom footprint rows exist for ORIs without localize_to_custom_footprint treatment: {unused[:20]}")
    return custom


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate overlap override and custom-footprint configs against the live V2 scope.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(args.repo_root.resolve())
    errors: list[str] = []
    warnings: list[str] = []

    overrides = _validate_overrides(paths, errors, warnings)
    custom = _validate_custom_footprints(paths, overrides, errors, warnings)

    summary = {
        "override_rows": int(len(overrides)),
        "custom_footprint_rows": int(len(custom)),
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        print(summary)
        raise SystemExit(1)
    print(summary)


if __name__ == "__main__":
    main()
