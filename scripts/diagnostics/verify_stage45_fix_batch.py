"""Verification harness for the Stage 4/5 rule batch.

Builds the block-group publication surface twice on the SAME post-Stage-2/3 inputs -- once with
the batch's rules and once with them reverted -- so every number is attributable to this batch
rather than to the Stage 2/3 batch that moved the controls under it. Then re-runs the Stage 4 and
Stage 5 screens the batch targets, before/after, and traces the pre-registered worked examples.

    uv run python scripts/diagnostics/verify_stage45_fix_batch.py --build baseline
    uv run python scripts/diagnostics/verify_stage45_fix_batch.py --build candidate
    uv run python scripts/diagnostics/verify_stage45_fix_batch.py --report

Writes only into analysis_scratch/stage45_fix_batch/. The promoted state/output surface is read,
never written: it is the v20 release and stays the reference point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk import allocation as alloc
from crimerisk.crime import OFFENSES_7
from crimerisk.paths import get_paths

OUT_ROOT = Path("analysis_scratch/stage45_fix_batch")

# Pre-registered worked examples (docs/STATE.md + the Stage 4 screen README).
ANGELES_NF_BG = "060379304002"
ANGELES_NF_SECOND_BG = "060379108153"
HARVARD_FOREST_BG = "250277042021"
LAKEWOOD_SLIVER_BG = "080590106041"
KNOTT_KY_COUNTY = "21119"
POARCH_CREEK_ORI = "ALDI00300"


# --------------------------------------------------------------------- legacy (pre-batch) rules


def _legacy_normalize_block_group_allocation_shares(bg_crosswalk: pd.DataFrame) -> pd.DataFrame:
    """The pre-batch share ladder: applied PER ROW, with no recipient floor.

    Reproduced here rather than kept behind a flag in production code, so the shipped module has
    one behaviour and the baseline lives with the harness that needs it.
    """
    bg = bg_crosswalk.copy()
    if "state_fips" in bg.columns:
        bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    key_col = "block_group_geoid" if "block_group_geoid" in bg.columns else "bg_id"
    bg[key_col] = bg[key_col].astype("string").str.zfill(12)

    chosen = pd.Series(np.nan, index=bg.index, dtype=float)
    for col in ["pop_share", "housing_share", "block_share", "aland_share", "allocation_share"]:
        if col not in bg.columns:
            continue
        share = pd.to_numeric(bg.get(col), errors="coerce").fillna(0.0).clip(lower=0.0)
        chosen = chosen.where(~(chosen.isna() & share.gt(0)), share)
    picked = pd.to_numeric(chosen, errors="coerce").fillna(0.0).clip(lower=0.0)

    totals = picked.groupby([bg["state_fips"], bg[key_col]], dropna=False).transform("sum")
    counts = picked.groupby([bg["state_fips"], bg[key_col]], dropna=False).transform("size")
    bg["allocation_share"] = np.where(
        pd.to_numeric(totals, errors="coerce").fillna(0.0) > 0,
        picked / pd.to_numeric(totals, errors="coerce").fillna(1.0),
        np.where(
            pd.to_numeric(counts, errors="coerce").fillna(0.0) > 0,
            1.0 / pd.to_numeric(counts, errors="coerce").fillna(1.0),
            0.0,
        ),
    )
    return bg


def _install_baseline_rules() -> None:
    """Revert the four Stage-4 rule changes for the baseline run."""
    original_footprints = alloc._load_overlap_custom_footprints

    def _verbatim_footprints(paths):
        footprints = original_footprints(paths)
        if footprints.empty:
            return footprints
        out = footprints.copy()
        out["weight_share_basis"] = alloc.CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_ACTIVITY
        return out

    alloc.normalize_block_group_allocation_shares = _legacy_normalize_block_group_allocation_shares
    alloc.assert_allocation_shares_conserve = lambda *args, **kwargs: None
    alloc._load_overlap_custom_footprints = _verbatim_footprints
    alloc._load_concurrent_jurisdiction_carveouts = lambda paths: pd.DataFrame(
        columns=["county_geoid"]
    )


# ------------------------------------------------------------------------------- the build


def build_surface(*, mode: str, year: int = 2024) -> dict[str, object]:
    """Build the block-group and tract publication surfaces plus the component ledger."""
    if mode not in {"baseline", "candidate"}:
        raise ValueError(mode)
    if mode == "baseline":
        _install_baseline_rules()

    paths = get_paths()
    config = alloc.resolve_allocation_build_config(paths, config=alloc.AllocationBuildConfig(year=year))
    controls = alloc._load_controls(paths, year=config.year)
    bg_prior = alloc._build_bg_prior_long(paths, config=config)
    bg_crosswalk = alloc._load_bg_crosswalk(paths)
    bg_crosswalk = bg_crosswalk[
        ~bg_crosswalk["state_fips"].astype(str).str.zfill(2).isin(alloc.RELEASE_EXCLUDED_STATE_FIPS)
    ].copy()
    agency_estimates = alloc._build_agency_allocation_target_estimates(paths=paths, year=int(config.year))

    jurisdiction_components = alloc._build_jurisdiction_component_allocations(
        paths=paths,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        controls=controls,
        year=config.year,
        residual_training_city_shares_path=config.residual_training_city_shares_path,
        residual_training_exclude_validation_case_types=config.residual_training_exclude_validation_case_types,
        residual_training_extra_bg_feature_paths=tuple(config.residual_training_extra_bg_feature_paths),
        residual_feature_policy_path=config.residual_feature_policy_path,
        residual_exclude_feature_policy_classes=tuple(config.residual_exclude_feature_policy_classes),
        residual_exclude_feature_policy_classes_by_offense=tuple(
            config.residual_exclude_feature_policy_classes_by_offense
        ),
        residual_transfer_tau_by_offense=tuple(config.residual_transfer_tau_by_offense),
        enable_county_anchoring=bool(config.enable_county_anchoring),
        agency_estimates=agency_estimates,
    )
    overlap_components = alloc._build_overlap_allocations(
        paths=paths,
        bg_prior=bg_prior,
        bg_crosswalk=bg_crosswalk,
        controls=controls,
        year=config.year,
        enable_county_anchoring=bool(config.enable_county_anchoring),
        agency_estimates=agency_estimates,
    )
    all_components = pd.concat([jurisdiction_components, overlap_components], ignore_index=True)
    bg_cov_raw = alloc._load_bg_covariates(
        paths, year=config.year, burglary_commercial_weight=config.burglary_commercial_weight
    )
    all_components, _zero_audit, _component_audit = alloc._redistribute_zero_target_components(
        all_components, bg_cov_raw
    )

    bg_counts = (
        all_components.groupby(["state_fips", "bg_id", "tract_id", "offense"], dropna=False)["component_count"]
        .sum()
        .reset_index()
        .pivot_table(index=["state_fips", "bg_id", "tract_id"], columns="offense", values="component_count", fill_value=0.0)
        .reset_index()
        .rename(columns={offense: alloc._expected_count_col(offense) for offense in OFFENSES_7})
    )
    footprint_rows = all_components[
        all_components["jurisdiction_type"].astype("string").eq("custom_footprint_overlap_layer")
    ]
    if footprint_rows.empty:
        footprint_counts = pd.DataFrame(columns=["state_fips", "bg_id", "tract_id"])
    else:
        footprint_counts = (
            footprint_rows.groupby(["state_fips", "bg_id", "tract_id", "offense"], dropna=False)["component_count"]
            .sum()
            .reset_index()
            .pivot_table(index=["state_fips", "bg_id", "tract_id"], columns="offense", values="component_count", fill_value=0.0)
            .reset_index()
        )
    footprint_counts = footprint_counts.rename(
        columns={offense: alloc._footprint_derived_count_col(offense) for offense in OFFENSES_7}
    )
    for offense in OFFENSES_7:
        col = alloc._footprint_derived_count_col(offense)
        if col not in footprint_counts.columns:
            footprint_counts[col] = 0.0
    bg_counts = bg_counts.merge(footprint_counts, on=["state_fips", "bg_id", "tract_id"], how="left")
    for offense in OFFENSES_7:
        col = alloc._footprint_derived_count_col(offense)
        bg_counts[col] = pd.to_numeric(bg_counts[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if mode == "baseline":
            # The pre-batch surface had no ambient-blind eligibility rule at all.
            bg_counts[col] = 0.0

    population_col = f"population_{int(config.year)}"
    bg_cov = bg_cov_raw.rename(columns={"bg_id": "block_group_geoid", "population": population_col})
    bg_universe = bg_crosswalk[["state_fips", "block_group_geoid"]].drop_duplicates().copy()
    bg_universe["state_fips"] = bg_universe["state_fips"].astype("string").str.zfill(2)
    bg_universe["block_group_geoid"] = bg_universe["block_group_geoid"].astype("string").str.zfill(12)
    bg_universe["tract_id"] = bg_universe["block_group_geoid"].str.slice(0, 11)
    bg_out = (
        bg_universe.merge(
            bg_counts,
            left_on=["block_group_geoid", "tract_id", "state_fips"],
            right_on=["bg_id", "tract_id", "state_fips"],
            how="left",
        )
        .merge(bg_cov, on=["block_group_geoid", "tract_id", "state_fips"], how="left")
        .merge(alloc._dominant_bg_jurisdiction(bg_crosswalk), on="block_group_geoid", how="left")
        .merge(
            alloc._build_bg_direct_incident_support(paths=paths, bg_crosswalk=bg_crosswalk, year=int(config.year)),
            on=["state_fips", "block_group_geoid"],
            how="left",
        )
    )
    for offense in OFFENSES_7:
        for col in (alloc._expected_count_col(offense), alloc._footprint_derived_count_col(offense)):
            bg_out[col] = pd.to_numeric(bg_out.get(col), errors="coerce").fillna(0.0)
    bg_out[population_col] = pd.to_numeric(bg_out.get(population_col), errors="coerce").fillna(0.0)
    bg_out = bg_out.drop(columns=["bg_id"], errors="ignore")
    bg_out = alloc._finalize_output(
        bg_out, geo_id_col="block_group_geoid", population_col=population_col, config=config
    )

    tract_jurisdiction = alloc._dominant_tract_jurisdiction(bg_crosswalk, bg_out, population_col=population_col)
    tract_counts = alloc._rollup_tracts_from_bg(bg_out, tract_jurisdiction, population_col=population_col)
    tract_counts = alloc._attach_tiger_land_area(
        tract_counts, land_area=alloc._load_tiger_land_area(paths, geography="tract"), geoid_col="tract_id"
    )
    tract_out = alloc._finalize_output(
        tract_counts,
        geo_id_col="tract_id",
        population_col=population_col,
        config=config,
        jurisdiction_col="dominant_eb_jurisdiction_id",
    )
    bg_out = alloc.apply_rare_offense_tract_support(bg_out, tract_out)

    out_dir = OUT_ROOT / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    bg_out.to_parquet(out_dir / "block_group_surface.parquet", index=False)
    tract_out.to_parquet(out_dir / "tract_surface.parquet", index=False)
    ledger_cols = ["state_fips", "bg_id", "tract_id", "jurisdiction_id", "jurisdiction_type", "offense", "component_count"]
    all_components[ledger_cols].to_parquet(out_dir / "component_ledger.parquet", index=False)

    target_mass = float(
        pd.to_numeric(
            controls.loc[controls["year"].eq(int(config.year)), "adjusted_count_ags_core"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).sum()
    )
    summary = {
        "mode": mode,
        "component_rows": int(len(all_components)),
        "component_mass": float(pd.to_numeric(all_components["component_count"], errors="coerce").fillna(0.0).sum()),
        "block_groups": int(bg_out["block_group_geoid"].nunique()),
        "surface_mass": float(
            sum(float(pd.to_numeric(bg_out[alloc._expected_count_col(o)], errors="coerce").fillna(0.0).sum()) for o in OFFENSES_7)
        ),
        "control_target_mass_all_lanes": target_mass,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------------------------------ the report


def _crosswalk_screens(repo_root: Path) -> dict[str, object]:
    """Screens a6/a7 read straight off the crosswalk, so this comparison needs no build."""
    from crimerisk.crosswalk_shares import (
        BELOW_FLOOR,
        ZERO_ON_BASIS,
        normalize_block_group_allocation_shares,
    )

    stored = pd.read_parquet(repo_root / "state/geometry/block_group_to_jurisdiction_crosswalk.parquet")
    before = _legacy_normalize_block_group_allocation_shares(stored)
    after = normalize_block_group_allocation_shares(stored)
    key = ["state_fips", "block_group_geoid"]
    for frame in (before, after):
        frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
        frame["block_group_geoid"] = frame["block_group_geoid"].astype("string").str.zfill(12)

    def _basis(frame: pd.DataFrame) -> pd.Series:
        chosen = pd.Series(pd.NA, index=frame.index, dtype="string")
        for col in ["pop_share", "housing_share", "block_share", "aland_share", "allocation_share"]:
            share = pd.to_numeric(frame.get(col), errors="coerce").fillna(0.0)
            chosen = chosen.where(~(chosen.isna() & share.gt(0)), col)
        return chosen.fillna("degenerate_equal_split")

    before_basis = _basis(stored)
    mixed_before = int((before_basis.groupby([before["state_fips"], before["block_group_geoid"]]).nunique() > 1).sum())
    mixed_after = int((after.groupby(key)["allocation_basis"].nunique() > 1).sum())

    zero_pop = pd.to_numeric(stored["pop20"], errors="coerce").fillna(0.0).le(0.0)
    bg_has_pop = pd.to_numeric(stored["total_pop20"], errors="coerce").fillna(0.0).gt(0.0)
    fragment = zero_pop & bg_has_pop
    return {
        "block_groups_normalising_a_mixed_basis_before": mixed_before,
        "block_groups_normalising_a_mixed_basis_after": mixed_after,
        "zero_population_fragments_with_positive_share_before": int(
            (fragment & before["allocation_share"].gt(0.0)).sum()
        ),
        "zero_population_fragments_with_positive_share_after": int(
            (fragment & after["allocation_share"].gt(0.0)).sum()
        ),
        "rows_zeroed_on_block_group_basis": int(after["allocation_recipient_status"].eq(ZERO_ON_BASIS).sum()),
        "rows_zeroed_by_the_recipient_floor": int(after["allocation_recipient_status"].eq(BELOW_FLOOR).sum()),
        "rows_exempted_so_no_pool_is_stranded": int(
            after["allocation_recipient_status"].astype(str).str.startswith("floor_exempt").sum()
        ),
        "lakewood_sliver_share_before": float(
            before.loc[before["block_group_geoid"].eq(LAKEWOOD_SLIVER_BG)
                       & before["jurisdiction_id"].eq("08:municipal:place:0842495"), "allocation_share"].sum()
        ),
        "lakewood_sliver_share_after": float(
            after.loc[after["block_group_geoid"].eq(LAKEWOOD_SLIVER_BG)
                      & after["jurisdiction_id"].eq("08:municipal:place:0842495"), "allocation_share"].sum()
        ),
    }


def _ledger_screens(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    def _bg_mass(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("bg_id")["component_count"].sum()

    before, after = _bg_mass(baseline), _bg_mass(candidate)
    joined = pd.concat([before.rename("before"), after.rename("after")], axis=1).fillna(0.0)
    moved = float((joined["after"] - joined["before"]).abs().sum() / 2.0)

    def _lane_mass(frame: pd.DataFrame) -> dict[str, float]:
        return {
            str(k): float(v)
            for k, v in frame.groupby("jurisdiction_type")["component_count"].sum().sort_values(ascending=False).items()
        }

    def _angeles(frame: pd.DataFrame, bg: str) -> dict[str, float]:
        rows = frame[frame["bg_id"].eq(bg)]
        return {
            str(j): float(v)
            for j, v in rows.groupby("jurisdiction_id")["component_count"].sum().sort_values(ascending=False).head(6).items()
        }

    def _pool_concentration(frame: pd.DataFrame, jurisdiction_type: str) -> dict[str, object]:
        """Screen a4's shape: the largest block group's share of a many-block-group pool."""
        rows = frame[frame["jurisdiction_type"].astype("string").eq(jurisdiction_type)]
        if rows.empty:
            return {"pools": 0}
        pools = rows.groupby(["jurisdiction_id", "offense"], dropna=False)["component_count"]
        totals = pools.sum()
        top = pools.max()
        counts = pools.size()
        share = (top / totals.where(totals > 0, np.nan)).dropna()
        many = share[counts.reindex(share.index).ge(6)]
        return {
            "pools": int(len(share)),
            "median_max_block_group_share": float(share.median()),
            "pools_with_6plus_block_groups": int(len(many)),
            "pools_where_one_block_group_takes_half": int(many.ge(0.5).sum()),
            "mass_in_those_pools": float(totals.reindex(many[many.ge(0.5)].index).sum()),
        }

    return {
        "custom_footprint_pool_concentration_before": _pool_concentration(baseline, "custom_footprint_overlap_layer"),
        "custom_footprint_pool_concentration_after": _pool_concentration(candidate, "custom_footprint_overlap_layer"),
        "municipal_pool_concentration_before": _pool_concentration(baseline, "municipal"),
        "municipal_pool_concentration_after": _pool_concentration(candidate, "municipal"),
        "national_component_mass_before": float(baseline["component_count"].sum()),
        "national_component_mass_after": float(candidate["component_count"].sum()),
        "mass_redistributed_between_block_groups": moved,
        "block_groups_gaining_mass": int((joined["after"] - joined["before"]).gt(1e-9).sum()),
        "block_groups_losing_mass": int((joined["after"] - joined["before"]).lt(-1e-9).sum()),
        "lane_mass_before": _lane_mass(baseline),
        "lane_mass_after": _lane_mass(candidate),
        "angeles_nf_by_jurisdiction_before": _angeles(baseline, ANGELES_NF_BG),
        "angeles_nf_by_jurisdiction_after": _angeles(candidate, ANGELES_NF_BG),
        "angeles_nf_second_before": _angeles(baseline, ANGELES_NF_SECOND_BG),
        "angeles_nf_second_after": _angeles(candidate, ANGELES_NF_SECOND_BG),
        "harvard_forest_before": _angeles(baseline, HARVARD_FOREST_BG),
        "harvard_forest_after": _angeles(candidate, HARVARD_FOREST_BG),
        "knott_ky_county_remainder_mass_before": float(
            baseline.loc[baseline["jurisdiction_id"].astype(str).str.endswith(KNOTT_KY_COUNTY), "component_count"].sum()
        ),
        "knott_ky_county_remainder_mass_after": float(
            candidate.loc[candidate["jurisdiction_id"].astype(str).str.endswith(KNOTT_KY_COUNTY), "component_count"].sum()
        ),
    }


def _eligibility_screens(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}

    def _flag_bgs(frame: pd.DataFrame, col_tpl: str) -> int:
        mask = pd.Series(False, index=frame.index)
        for offense in OFFENSES_7:
            col = col_tpl.format(offense=offense)
            if col in frame.columns:
                mask |= frame[col].fillna(False).astype(bool)
        return int(mask.sum())

    out["ambient_blind_footprint_cells"] = {
        offense: int(candidate[f"footprint_ambient_exposure_missing_{offense}"].fillna(False).sum())
        for offense in OFFENSES_7
    }
    ambient_mask = pd.Series(False, index=candidate.index)
    for offense in OFFENSES_7:
        ambient_mask |= candidate[f"footprint_ambient_exposure_missing_{offense}"].fillna(False).astype(bool)
    out["ambient_blind_footprint_block_groups"] = int(ambient_mask.sum())
    out["ambient_blind_footprint_population"] = int(
        pd.to_numeric(candidate.loc[ambient_mask, "population_2024"], errors="coerce").fillna(0.0).sum()
    )
    out["ambient_blind_footprint_counts_still_published"] = float(
        pd.to_numeric(candidate.loc[ambient_mask, "expected_count_total"], errors="coerce").fillna(0.0).sum()
    )
    out["ambient_blind_footprint_density_still_published"] = int(
        pd.to_numeric(candidate.loc[ambient_mask, "crime_density_total"], errors="coerce").notna().sum()
    )
    out["ambient_blind_examples"] = (
        candidate.loc[
            ambient_mask,
            ["block_group_geoid", "population_2024", "exposure_proxy_2024", "expected_count_total",
             "footprint_derived_count_share_larceny", "crime_density_total"],
        ]
        .sort_values("expected_count_total", ascending=False)
        .head(20)
        .to_dict(orient="records")
    )

    # Transient guard: shipped definition vs fixed definition, on the SAME surface.
    ratio = pd.to_numeric(candidate["transient_exposure_daytime_to_resident_ratio"], errors="coerce")
    households = pd.to_numeric(candidate["households_total"], errors="coerce").fillna(0.0)
    shipped_bgs = pd.Series(False, index=candidate.index)
    fixed_bgs = pd.Series(False, index=candidate.index)
    for offense in OFFENSES_7:
        primary = pd.to_numeric(candidate[f"index_{offense}_primary"], errors="coerce")
        resident = pd.to_numeric(candidate[f"index_{offense}_resident"], errors="coerce")
        shipped_bgs |= (
            households.ge(alloc.NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
            & pd.to_numeric(candidate["population_2024"], errors="coerce").gt(0.0)
            & ratio.ge(alloc.TRANSIENT_EXPOSURE_DAYTIME_TO_RESIDENT_RATIO)
            & primary.ge(alloc.TRANSIENT_EXPOSURE_INDEX_THRESHOLD)
        ).fillna(False)
        fixed_bgs |= candidate[f"transient_exposure_likely_{offense}"].fillna(False).astype(bool)
    out["transient_guard_block_groups_shipped_definition"] = int(shipped_bgs.sum())
    out["transient_guard_block_groups_fixed_definition"] = int(fixed_bgs.sum())
    out["transient_guard_population_fixed_definition"] = int(
        pd.to_numeric(candidate.loc[fixed_bgs, "population_2024"], errors="coerce").fillna(0.0).sum()
    )
    out["transient_guard_cells_by_offense_fixed"] = {
        offense: int(candidate[f"transient_exposure_likely_{offense}"].fillna(False).sum())
        for offense in OFFENSES_7
    }
    out["transient_guard_rare_offense_cells_at_block_group"] = {
        offense: int(candidate[f"transient_exposure_likely_{offense}"].fillna(False).sum())
        for offense in alloc.RARE_OFFENSE_TRACT_SUPPORT
    }
    out["transient_guard_households_term_would_block"] = int(
        (
            households.lt(alloc.NON_RESIDENTIAL_HOUSEHOLD_FLOOR)
            & fixed_bgs
        ).sum()
    )

    # Suppression vocabulary: the F7 hole, before and after.
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        non_residential = frame["non_residential_flag"].fillna(False).astype(bool)
        holes = {}
        for offense in OFFENSES_7:
            reason = frame[f"denominator_reason_{offense}"].astype("string")
            holes[offense] = int((non_residential & reason.eq("publishable")).sum())
        out[f"denominator_reason_publishable_on_non_residential_{label}"] = holes
        out[f"denominator_reason_values_{label}"] = sorted(
            set(frame["denominator_reason_larceny"].dropna().astype(str).unique())
        )
        out[f"estimate_mode_values_{label}"] = sorted(
            set(frame["estimate_mode_larceny"].dropna().astype(str).unique())
        )

    # Grey population on the default layer, before and after.
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        grey = pd.to_numeric(frame["index_total_primary_event_weighted"], errors="coerce").isna()
        out[f"default_layer_grey_block_groups_{label}"] = int(grey.sum())
        out[f"default_layer_grey_population_{label}"] = int(
            pd.to_numeric(frame.loc[grey, "population_2024"], errors="coerce").fillna(0.0).sum()
        )
    return out


def _intensity_screens(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    """Screen a5's shape: index >= 400 on exposure < 5,000 over more than 5 sq mi."""
    out: dict[str, object] = {}
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        index = pd.to_numeric(frame["index_total_primary_event_weighted"], errors="coerce")
        exposure = pd.to_numeric(frame["exposure_proxy_2024"], errors="coerce").fillna(0.0)
        land = pd.to_numeric(frame["land_area_sq_mi"], errors="coerce").fillna(0.0)
        hits = index.ge(400.0) & exposure.lt(5000.0) & land.gt(5.0)
        out[f"a5_low_exposure_high_index_cells_{label}"] = int(hits.sum())
        out[f"index_ge_400_cells_{label}"] = int(index.ge(400.0).sum())
        out[f"a5_worst_{label}"] = (
            frame.loc[hits, ["block_group_geoid", "population_2024", "exposure_proxy_2024",
                             "land_area_sq_mi", "expected_count_total", "index_total_primary_event_weighted"]]
            .sort_values("index_total_primary_event_weighted", ascending=False)
            .head(12)
            .to_dict(orient="records")
        )
    for bg, name in ((ANGELES_NF_BG, "angeles_nf"), (ANGELES_NF_SECOND_BG, "angeles_nf_second"),
                     (HARVARD_FOREST_BG, "harvard_forest"), (LAKEWOOD_SLIVER_BG, "lakewood_sliver")):
        for label, frame in (("baseline", baseline), ("candidate", candidate)):
            row = frame.loc[frame["block_group_geoid"].eq(bg)]
            if row.empty:
                out[f"{name}_{label}"] = None
                continue
            row = row.iloc[0]
            out[f"{name}_{label}"] = {
                "population_2024": float(row["population_2024"]),
                "exposure_proxy_2024": float(row["exposure_proxy_2024"]),
                "expected_count_total": float(row["expected_count_total"]),
                "index_total_primary_event_weighted": (
                    None if pd.isna(row["index_total_primary_event_weighted"])
                    else float(row["index_total_primary_event_weighted"])
                ),
                "estimate_mode_larceny": str(row["estimate_mode_larceny"]),
                "denominator_reason_larceny": str(row["denominator_reason_larceny"]),
            }
    return out


def _promoted_surface_reference(repo_root: Path) -> dict[str, object]:
    """The suppression-vocabulary hole measured on the PROMOTED v20 surface.

    The baseline build reverts the four Stage-4 allocation rules but not the Stage-5 vocabulary
    fix, so the true "before" for screen F7 is the shipped release, not the baseline.
    """
    out: dict[str, object] = {}
    for label, name in (("block_group", "crimerisk_block_group_2024_ags_core.parquet"),
                        ("tract", "crimerisk_tract_2024_ags_core.parquet")):
        path = repo_root / "state" / "output" / name
        if not path.exists():
            out[label] = None
            continue
        cols = ["non_residential_flag"] + [f"denominator_reason_{o}" for o in OFFENSES_7] + [
            f"resident_denominator_reason_{o}" for o in OFFENSES_7
        ]
        frame = pd.read_parquet(path, columns=cols)
        non_residential = frame["non_residential_flag"].fillna(False).astype(bool)
        out[label] = {
            "non_residential_rows": int(non_residential.sum()),
            "denominator_reason_publishable_on_non_residential": {
                offense: int(
                    (non_residential & frame[f"denominator_reason_{offense}"].astype("string").eq("publishable")).sum()
                )
                for offense in OFFENSES_7
            },
            "resident_denominator_reason_publishable_on_non_residential": {
                offense: int(
                    (
                        non_residential
                        & frame[f"resident_denominator_reason_{offense}"].astype("string").eq("publishable")
                    ).sum()
                )
                for offense in OFFENSES_7
            },
            "denominator_reason_values": sorted(
                set(frame["denominator_reason_larceny"].dropna().astype(str).unique())
            ),
        }
    return out


def _independent_rule_recomputation(candidate: pd.DataFrame) -> dict[str, object]:
    """Recompute the batch's two new predicates from PUBLISHED fields only.

    Independent of the allocator's own code path, which is what the release validator does; if
    the flag cannot be reproduced from the published columns, a consumer cannot audit it.
    """
    out: dict[str, object] = {}
    pop = pd.to_numeric(candidate["resident_secondary_denominator"], errors="coerce").fillna(0.0)
    exposure = pd.to_numeric(candidate["exposure_proxy_2024"], errors="coerce").fillna(0.0)
    non_residential = candidate["non_residential_flag"].fillna(False).astype(bool)
    special_use = candidate["special_use_tract_flag"].fillna(False).astype(bool)
    flag_mismatches = {}
    share_mismatches = {}
    for offense in OFFENSES_7:
        count = pd.to_numeric(candidate[alloc._expected_count_col(offense)], errors="coerce").fillna(0.0)
        footprint = pd.to_numeric(candidate[alloc._footprint_derived_count_col(offense)], errors="coerce").fillna(0.0)
        share = pd.Series(
            np.where(count.to_numpy() > 0.0, footprint.to_numpy() / np.where(count > 0.0, count, 1.0), 0.0),
            index=candidate.index,
        ).clip(0.0, 1.0)
        published_share = pd.to_numeric(candidate[f"footprint_derived_count_share_{offense}"], errors="coerce")
        share_mismatches[offense] = int((published_share - share).abs().gt(1e-9).sum())

        denominator = pd.to_numeric(candidate[f"primary_denominator_{offense}"], errors="coerce").fillna(0.0)
        if offense in alloc.PERSON_EXPOSURE_FLOOR_OFFENSES:
            resident_floor = pop.lt(alloc.PERSON_EXPOSURE_DENOMINATOR_FLOOR)
        elif offense == "motor_vehicle_theft":
            resident_floor = denominator.lt(alloc.MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR)
        else:
            resident_floor = pd.Series(False, index=candidate.index)
        special = special_use | (
            denominator.lt(alloc.BURGLARY_PREMISES_DENOMINATOR_FLOOR) if offense == "burglary"
            else pd.Series(False, index=candidate.index)
        )
        baseline_pub = (~non_residential) & pop.gt(0.0) & ~special & ~resident_floor.fillna(False)
        recomputed = alloc._count_derived_rate_index(counts=count, denominator=pop, publishable=baseline_pub)
        national = float(recomputed["national_rate_per_100k"])
        count_lower, _upper = alloc._poisson_count_interval(count)
        measurable = baseline_pub & pop.gt(0.0)
        conservative = pd.Series(np.nan, index=candidate.index, dtype=float)
        conservative.loc[measurable] = 1e5 * count_lower.loc[measurable] / pop.loc[measurable]
        ratio = pd.Series(np.nan, index=candidate.index, dtype=float)
        if np.isfinite(national) and national > 0:
            ratio = conservative / national
        expected = (
            share.gt(alloc.FOOTPRINT_DERIVED_MASS_SHARE_FLOOR)
            & ~exposure.gt(pop)
            & ratio.ge(alloc.AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO)
        ).fillna(False)
        published = candidate[f"footprint_ambient_exposure_missing_{offense}"].fillna(False).astype(bool)
        flag_mismatches[offense] = int((published.to_numpy() != expected.to_numpy()).sum())
    out["footprint_share_recomputation_mismatches"] = share_mismatches
    out["ambient_blind_flag_recomputation_mismatches"] = flag_mismatches

    # Publishable mask, recomputed the way the release validator does.
    mask_mismatches = {}
    for offense in OFFENSES_7:
        denominator = pd.to_numeric(candidate[f"primary_denominator_{offense}"], errors="coerce").fillna(0.0)
        if offense in alloc.PERSON_EXPOSURE_FLOOR_OFFENSES:
            floor_hit = denominator.lt(alloc.PERSON_EXPOSURE_DENOMINATOR_FLOOR)
        elif offense == "motor_vehicle_theft":
            floor_hit = denominator.lt(alloc.MVT_VEHICLE_EXPOSURE_DENOMINATOR_FLOOR)
        else:
            floor_hit = pd.Series(False, index=candidate.index)
        special = special_use | (
            denominator.lt(alloc.BURGLARY_PREMISES_DENOMINATOR_FLOOR) if offense == "burglary"
            else pd.Series(False, index=candidate.index)
        )
        expected_suppressed = (
            non_residential
            | special
            | floor_hit.fillna(False)
            | candidate[f"footprint_ambient_exposure_missing_{offense}"].fillna(False).astype(bool)
        )
        expected_pub = (~expected_suppressed) & denominator.gt(0.0)
        published_pub = candidate[f"primary_index_publishable_{offense}"].fillna(False).astype(bool)
        mask_mismatches[offense] = int((published_pub.to_numpy() != expected_pub.to_numpy()).sum())
    out["publishable_mask_recomputation_mismatches"] = mask_mismatches
    return out


def _footprint_ori_trace(baseline: pd.DataFrame, candidate: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, object]:
    """The pre-registered small-denominator tribal footprints, block group by block group."""
    from crimerisk.paths import get_paths

    footprints = alloc._load_overlap_custom_footprints(get_paths())
    named = {
        "poarch_creek_al": "ALDI00300",
        "stillaguamish_wa": "WADI01300",
        "seminole_fl": "FL0063400",
        "gun_lake_mi": "MIDI03000",
        # Fork ruling 4: OSU moved off land area onto LandScan daytime population, like Harvard.
        "osu_oh": "OH0252700",
        "harvard_ma": "MA0096200",
        # Fork ruling 3: post footprints are the population-derived class that gains the
        # activity term.
        "ksp_post_1_ky": "KYKSP0100",
    }
    out: dict[str, object] = {}
    for label, ori in named.items():
        bgs = set(footprints.loc[footprints["ori9"].astype(str).eq(ori), "bg_id"].astype(str))
        if not bgs:
            out[label] = {"footprint_block_groups": 0}
            continue
        rows = candidate[candidate["block_group_geoid"].astype(str).isin(bgs)]
        base_rows = baseline[baseline["block_group_geoid"].astype(str).isin(bgs)]
        ori_mass = float(
            pd.to_numeric(
                ledger.loc[ledger["jurisdiction_id"].astype(str).eq(ori), "component_count"], errors="coerce"
            ).fillna(0.0).sum()
        )
        suppressed = pd.Series(False, index=rows.index)
        for offense in OFFENSES_7:
            suppressed |= rows[f"footprint_ambient_exposure_missing_{offense}"].fillna(False).astype(bool)
        out[label] = {
            "ori": ori,
            "footprint_block_groups": len(bgs),
            "footprint_block_groups_on_the_surface": int(len(rows)),
            "agency_mass_allocated": ori_mass,
            "ambient_blind_block_groups": int(suppressed.sum()),
            "population": int(pd.to_numeric(rows["population_2024"], errors="coerce").fillna(0.0).sum()),
            "counts_published": float(pd.to_numeric(rows["expected_count_total"], errors="coerce").fillna(0.0).sum()),
            "density_published_rows": int(pd.to_numeric(rows["crime_density_total"], errors="coerce").notna().sum()),
            "index_published_rows_baseline": int(
                pd.to_numeric(base_rows["index_total_primary_event_weighted"], errors="coerce").notna().sum()
            ),
            "index_published_rows_candidate": int(
                pd.to_numeric(rows["index_total_primary_event_weighted"], errors="coerce").notna().sum()
            ),
            "worst_rows": rows.loc[
                suppressed,
                ["block_group_geoid", "population_2024", "exposure_proxy_2024", "expected_count_total",
                 "estimate_mode_larceny", "denominator_reason_larceny", "crime_density_total"],
            ].head(8).to_dict(orient="records"),
        }
    return out


def _conservation(baseline: pd.DataFrame, candidate: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for offense in OFFENSES_7:
        surface = float(pd.to_numeric(candidate[alloc._expected_count_col(offense)], errors="coerce").fillna(0.0).sum())
        ledger_mass = float(
            pd.to_numeric(ledger.loc[ledger["offense"].eq(offense), "component_count"], errors="coerce").fillna(0.0).sum()
        )
        out[f"surface_minus_ledger_{offense}"] = surface - ledger_mass
    out["max_abs_surface_minus_ledger"] = max(abs(v) for k, v in out.items() if k.startswith("surface_minus_ledger"))
    out["negative_component_counts"] = int((pd.to_numeric(ledger["component_count"], errors="coerce") < 0).sum())
    out["null_expected_counts_candidate"] = int(
        sum(int(candidate[alloc._expected_count_col(o)].isna().sum()) for o in OFFENSES_7)
    )
    out["counts_lost_to_suppression"] = 0
    for offense in OFFENSES_7:
        suppressed = candidate[f"primary_index_suppressed_{offense}"].fillna(False).astype(bool)
        out["counts_lost_to_suppression"] += int(
            candidate.loc[suppressed, alloc._expected_count_col(offense)].isna().sum()
        )
    out["national_mass_baseline"] = float(
        sum(float(pd.to_numeric(baseline[alloc._expected_count_col(o)], errors="coerce").fillna(0.0).sum()) for o in OFFENSES_7)
    )
    out["national_mass_candidate"] = float(
        sum(float(pd.to_numeric(candidate[alloc._expected_count_col(o)], errors="coerce").fillna(0.0).sum()) for o in OFFENSES_7)
    )
    out["national_mass_delta"] = out["national_mass_candidate"] - out["national_mass_baseline"]
    return out


def report() -> dict[str, object]:
    paths = get_paths()
    baseline = pd.read_parquet(OUT_ROOT / "baseline" / "block_group_surface.parquet")
    candidate = pd.read_parquet(OUT_ROOT / "candidate" / "block_group_surface.parquet")
    baseline_ledger = pd.read_parquet(OUT_ROOT / "baseline" / "component_ledger.parquet")
    candidate_ledger = pd.read_parquet(OUT_ROOT / "candidate" / "component_ledger.parquet")

    payload = {
        "crosswalk": _crosswalk_screens(paths.repo_root),
        "ledger": _ledger_screens(baseline_ledger, candidate_ledger),
        "eligibility": _eligibility_screens(baseline, candidate),
        "intensity": _intensity_screens(baseline, candidate),
        "promoted_v20_reference": _promoted_surface_reference(paths.repo_root),
        "independent_recomputation": _independent_rule_recomputation(candidate),
        "footprint_oris": _footprint_ori_trace(baseline, candidate, candidate_ledger),
        "conservation": _conservation(baseline, candidate, candidate_ledger),
    }
    out_path = OUT_ROOT / "stage45_fix_batch_verification.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", choices=["baseline", "candidate"], default=None)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    if args.build:
        build_surface(mode=args.build, year=args.year)
    if args.report:
        report()
    if not args.build and not args.report:
        parser.error("pass --build baseline|candidate or --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
