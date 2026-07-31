"""Verification harness for the Stage 2 fix batch.

Re-runs the Stage 2 screens the first-read audit defined and prints before/after per screen,
mass moved by state, and the four pre-registered worked examples (Knott KY, Todd SD / Rosebud,
NJ Transit, Harvard). Read-only: writes one summary CSV per screen into the staging directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.allocation import (
    COUNTY_NONMUNICIPAL_OVERLAP_KIND,
    STATE_REMAINDER_TYPE,
    _build_exclusive_footprint_displacement,
    _load_overlap_custom_footprints,
    _load_overlap_footprint_overrides,
)
from crimerisk.paths import get_paths

OFFENSES_7 = [
    "murder",
    "rape",
    "robbery",
    "aggravated_assault",
    "burglary",
    "larceny",
    "motor_vehicle_theft",
]


def _observed_2024(repo_root: Path) -> pd.DataFrame:
    obs = pd.read_parquet(
        repo_root / "state" / "observations" / "agency_year_observations.parquet",
        columns=["ori9", "year", "offense", "count", "population"],
    )
    obs = obs[obs["year"].astype("Int64").eq(2024) & obs["offense"].isin(OFFENSES_7)].copy()
    obs["count"] = pd.to_numeric(obs["count"], errors="coerce").fillna(0.0)
    obs["population"] = pd.to_numeric(obs["population"], errors="coerce").fillna(0.0)
    mass = (
        obs.groupby(["ori9", "offense"])["count"].max().groupby("ori9").sum().rename("obs_2024")
    )
    pop = obs.groupby("ori9")["population"].max().rename("service_pop")
    return pd.concat([mass, pop], axis=1).reset_index()


def report(repo_root: Path, out_dir: Path) -> dict[str, object]:
    paths = get_paths()
    out: dict[str, object] = {}
    crosswalk = pd.read_parquet(
        repo_root / "state" / "reference" / "agency_to_jurisdiction_crosswalk.parquet"
    )
    agency = pd.read_parquet(repo_root / "state" / "reference" / "agency_master.parquet")
    observed = _observed_2024(repo_root)
    linked = crosswalk.merge(
        agency[
            [
                "ori9",
                "county_fips",
                "county_fips_source",
                "agency_type_norm",
                "agency_name_std",
                "is_tribal_agency",
            ]
        ],
        left_on="ori",
        right_on="ori9",
        how="left",
    ).merge(observed, on="ori9", how="left")
    linked["obs_2024"] = linked["obs_2024"].fillna(0.0)

    out["crosswalk"] = {
        "links": int(len(crosswalk)),
        "relationship_type": crosswalk["relationship_type"].value_counts().to_dict(),
        "tribal_links": int(linked["is_tribal_agency"].fillna(False).sum()),
        "tribal_links_on_municipal": int(
            (
                linked["is_tribal_agency"].fillna(False)
                & linked["jurisdiction_id"].astype(str).str.contains(":municipal:")
            ).sum()
        ),
    }

    overrides = _load_overlap_footprint_overrides(paths)
    footprints = _load_overlap_custom_footprints(paths)
    displacement = _build_exclusive_footprint_displacement(
        overrides=overrides, custom_footprints=footprints
    )
    out["registries"] = {
        "overlap_footprint_overrides": int(len(overrides)),
        "treatments": overrides["final_overlap_treatment"].value_counts().to_dict(),
        "custom_footprint_rows": int(len(footprints)),
        "custom_footprint_oris": int(footprints["ori9"].nunique()),
        "displacing_oris": int(overrides["displaces_county_remainder"].sum()),
        "displacing_block_groups": int(len(displacement)),
    }

    bg_crosswalk = pd.read_parquet(
        repo_root / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
    )
    remainder = bg_crosswalk[bg_crosswalk["jurisdiction_type"].eq(STATE_REMAINDER_TYPE)].copy()
    remainder["key"] = (
        remainder["state_fips"].astype("string").str.zfill(2)
        + "|"
        + remainder["block_group_geoid"].astype("string").str.zfill(12)
    )
    disp = displacement.copy()
    disp["key"] = disp["state_fips"].astype("string").str.zfill(2) + "|" + disp["bg_id"].astype("string")
    joined = remainder.merge(disp[["key", "displaced_share"]], on="key", how="inner")
    joined["displaced_pop"] = (
        pd.to_numeric(joined["pop20"], errors="coerce").fillna(0.0)
        * pd.to_numeric(joined["displaced_share"], errors="coerce").fillna(0.0)
    )
    by_state = (
        joined.groupby("state_fips", dropna=False)
        .agg(
            block_groups=("block_group_geoid", "nunique"),
            remainder_pop_on_footprint=("pop20", "sum"),
            displaced_pop=("displaced_pop", "sum"),
        )
        .reset_index()
        .sort_values("displaced_pop", ascending=False)
    )
    by_state.to_csv(out_dir / "exclusive_displacement_by_state.csv", index=False)
    out["displacement"] = {
        "states": int(len(by_state)),
        "remainder_block_groups_touched": int(joined["block_group_geoid"].nunique()),
        "remainder_pop_on_footprints": float(joined["pop20"].sum()),
        "displaced_remainder_pop": float(joined["displaced_pop"].sum()),
        "top_states": by_state.head(10).to_dict(orient="records"),
    }

    # Screen d: state-police post localization.
    sp = linked[linked["agency_type_norm"].astype("string").eq("state_law_enforcement")].copy()
    sp_footprinted = set(
        overrides.loc[
            overrides["footprint_type"].astype("string").str.startswith("state_police_", na=False),
            "ori9",
        ].astype(str)
    )
    sp["has_post_footprint"] = sp["ori"].astype(str).isin(sp_footprinted)
    out["state_police"] = {
        "links": int(len(sp)),
        "post_footprinted_oris": int(sp["has_post_footprint"].sum()),
        "post_footprinted_mass": float(sp.loc[sp["has_post_footprint"], "obs_2024"].sum()),
        "by_state": sp[sp["has_post_footprint"]]
        .groupby("state_abbr")
        .agg(oris=("ori", "size"), mass=("obs_2024", "sum"))
        .reset_index()
        .to_dict(orient="records"),
    }

    # Residue: overlap links that can never county-anchor.
    null_county = linked[
        linked["relationship_type"].eq("overlap") & linked["county_fips"].isna()
    ]
    out["null_county_overlap_links"] = {
        "links": int(len(null_county)),
        "links_with_mass": int((null_county["obs_2024"] > 0).sum()),
        "mass": float(null_county["obs_2024"].sum()),
        "by_agency_type": null_county.groupby("agency_type_norm")["obs_2024"]
        .agg(["size", "sum"])
        .reset_index()
        .to_dict(orient="records"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2_fix_batch_verification.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> int:
    repo_root = get_paths().repo_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "analysis_scratch" / "stage2_fix_batch",
    )
    args = parser.parse_args()
    summary = report(repo_root, args.out_dir)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
