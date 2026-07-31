from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths
from crimerisk.source_provenance import (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    assign_preferred_value,
    initialize_preferred_source,
)


CONFIG_SPECS: dict[str, list[str]] = {
    "feature_source_inventory.csv": [
        "concept_family",
        "concept_name",
        "ags_consistent",
        "protected_or_excluded_flag",
        "source_family",
        "source_dataset",
        "table_or_variable_codes_or_fields",
        "available_at_bg",
        "available_at_tract",
        "available_current_year",
        "public_access_status",
        "proposed_transformation",
        "priority",
        "research_notes",
        "source_links",
    ],
    "model_diagnostic_review.csv": [
        "offense",
        "state_or_slice",
        "diagnostic_type",
        "symptom",
        "suspected_cause",
        "evidence_note",
        "priority",
        "recommended_followup",
    ],
    "overlap_footprint_overrides.csv": [
        "ori",
        "final_overlap_treatment",
        "overlap_subtype_final",
        "footprint_type",
        "target_state_fips",
        "target_county_fips",
        "target_place_fips",
        "target_jurisdiction_id",
        "geometry_source_type",
        "geometry_source_ref",
        "confidence",
        "source_note",
        "reviewer_note",
    ],
    "reporting_regime_overrides.csv": [
        "ori",
        "year",
        "offense",
        "final_reporting_regime",
        "evidence_type",
        "source_note",
        "confidence",
        "reviewer_note",
    ],
    "service_structure_notes.csv": [
        "state_fips",
        "county_fips",
        "agency_or_structure_name",
        "issue_type",
        "service_structure_summary",
        "implication_for_pipeline",
        "source_links",
        "note_status",
    ],
    "methodology_audit_log.csv": [
        "component",
        "current_rule",
        "audit_status",
        "retain_replace_decision",
        "evidence_summary",
        "next_action",
        "owner",
    ],
    "city_incident_sources.csv": [
        "city_key",
        "jurisdiction_id",
        "city_name",
        "state_abbr",
        "source_name",
        "source_url",
        "portal_type",
        "coverage_start_year",
        "coverage_end_year",
        "years_usable",
        "offense_fields_present",
        "date_field",
        "location_fields_present",
        "latlon_present",
        "address_present",
        "block_group_join_ready",
        "geocode_quality_tier",
        "dedupe_key_available",
        "offense_crosswalk_complexity",
        "recommended_disposition",
        "analyst_notes",
    ],
    "city_incident_priority.csv": [
        "city_key",
        "jurisdiction_id",
        "city_name",
        "state_abbr",
        "priority_bucket",
        "priority_rank",
        "basis",
        "notes",
    ],
}

SUPPORT_CONFIG_FILENAMES = {
    "feature_source_inventory.csv",
    "model_diagnostic_review.csv",
    "service_structure_notes.csv",
    "methodology_audit_log.csv",
}

CITY_PACKET_CHECKLIST: list[dict[str, object]] = [
    {
        "step_order": 1,
        "step_name": "source_research",
        "required_output": "Confirm the live source contract, schema, coverage years, and whether an existing local ETL already solves part of the city.",
    },
    {
        "step_order": 2,
        "step_name": "raw_snapshot",
        "required_output": "Pull or identify the raw source snapshot and record the exact path or query used.",
    },
    {
        "step_order": 3,
        "step_name": "offense_crosswalk",
        "required_output": "Define a city-to-V2 offense crosswalk and record any approximate or excluded mappings.",
    },
    {
        "step_order": 4,
        "step_name": "spatial_contract",
        "required_output": "Determine how incidents can be assigned to block groups and flag any redaction or proxy requirements.",
    },
    {
        "step_order": 5,
        "step_name": "published_reconciliation",
        "required_output": "Reconcile the city-year totals against a public published source and explain any remaining divergence.",
    },
    {
        "step_order": 6,
        "step_name": "packet_handoff",
        "required_output": "Summarize whether the packet is production-ready, blocked, or needs integration work.",
    },
]


def _ensure_template(path: Path, columns: list[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    return True


def _config_path(paths: RepoPaths, filename: str) -> Path:
    if filename in SUPPORT_CONFIG_FILENAMES:
        return paths.review_support_dir / filename
    return paths.repo_root / "configs" / filename


def _review_seed_path(paths: RepoPaths, filename: str) -> Path:
    return paths.review_support_dir / "seeds" / filename


def _build_feature_inventory_seed(paths: RepoPaths) -> Path | None:
    feature_path = paths.state_dir / "modeling" / "jurisdiction_model_features_2024.parquet"
    if not feature_path.exists():
        return None
    out_path = _review_seed_path(paths, "feature_source_inventory_seed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(feature_path).copy()
    if "feature_group" in meta.columns:
        meta = meta[~meta["feature_group"].astype("string").eq("state_fixed_effect")].copy()
    elif "feature_source" in meta.columns:
        meta = meta[meta["feature_source"].astype("string").eq("base_covariate")].copy()
    meta["concept_family"] = meta["feature_column"].astype(str).str.split("_").str[0]
    meta["concept_name"] = meta["feature_column"].astype(str)
    meta["ags_consistent"] = pd.NA
    meta["protected_or_excluded_flag"] = False
    meta["source_family"] = "current_v2_feature_frame"
    meta["source_dataset"] = pd.NA
    meta["table_or_variable_codes_or_fields"] = pd.NA
    meta["available_at_bg"] = True
    meta["available_at_tract"] = True
    meta["available_current_year"] = pd.NA
    meta["public_access_status"] = "implemented"
    meta["proposed_transformation"] = "already_in_feature_frame"
    meta["priority"] = "implemented"
    meta["research_notes"] = "Current modeled-surface covariate already present in V2 feature frame."
    meta["source_links"] = pd.NA
    cols = CONFIG_SPECS["feature_source_inventory.csv"]
    meta[cols].sort_values(["concept_family", "concept_name"], kind="mergesort").to_csv(out_path, index=False)
    return out_path


def _build_reporting_regime_review_queue(paths: RepoPaths, *, top_n: int) -> Path | None:
    regime_path = paths.state_dir / "modeling" / "agency_year_reporting_regimes.parquet"
    obs_path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    agency_master_path = paths.state_dir / "reference" / "agency_master.parquet"
    if not regime_path.exists() or not obs_path.exists():
        return None
    regimes = pd.read_parquet(regime_path)
    obs = pd.read_parquet(obs_path)
    regimes = regimes[regimes["year"].astype(int).eq(2024)].copy()
    reviewable = regimes["reporting_regime"].isin(
        ["lumpy_or_batched", "annual_only_but_usable", "structurally_missing_or_unreliable"]
    )
    regimes = regimes[reviewable].copy()
    if regimes.empty:
        return None
    counts = (
        obs[obs["year"].astype(int).eq(2024)][["ori9", "year", "offense", "source", "count"]]
        .pivot_table(index=["ori9", "year", "offense"], columns="source", values="count", aggfunc="sum")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    obs_2024 = obs[obs["year"].astype(int).eq(2024)].copy()
    meta_cols = ["state_fips", "state_abbr", "agency_name_std", "agency_type_norm", "county_fips", "place_fips"]

    def _first_non_null(series: pd.Series):
        non_null = series.dropna()
        if non_null.empty:
            return pd.NA
        non_blank = non_null.astype("string")
        non_blank = non_blank[non_blank.str.strip().ne("")]
        if non_blank.empty:
            return pd.NA
        return non_blank.iloc[0]

    obs_meta = (
        obs_2024[["ori9", *meta_cols]]
        .groupby("ori9", dropna=False)
        .agg({col: _first_non_null for col in meta_cols})
        .reset_index()
    )

    agency_meta = pd.DataFrame(columns=["ori9", *meta_cols])
    if agency_master_path.exists():
        agency_master = pd.read_parquet(agency_master_path)
        if "ori9" in agency_master.columns:
            keep_cols = ["ori9", *[c for c in meta_cols if c in agency_master.columns]]
            agency_meta = agency_master[keep_cols].copy().drop_duplicates("ori9", keep="first")

    out = regimes.merge(counts, on=["ori9", "year", "offense"], how="left")
    out = out.merge(obs_meta.rename(columns={c: f"{c}_obs" for c in meta_cols}), on="ori9", how="left")
    out = out.merge(agency_meta.rename(columns={c: f"{c}_agency" for c in meta_cols}), on="ori9", how="left")

    for col in meta_cols:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].where(out[col].notna(), out.get(f"{col}_obs"))
        out[col] = out[col].where(out[col].notna(), out.get(f"{col}_agency"))

    drop_cols = [f"{c}_obs" for c in meta_cols] + [f"{c}_agency" for c in meta_cols]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    out["ori"] = out["ori9"].astype("string")
    out["case_id"] = (
        out["ori9"].astype("string")
        + ":"
        + out["year"].astype("Int64").astype("string")
        + ":"
        + out["offense"].astype("string")
    )

    out["review_lane"] = np.select(
        [
            out["regime_reason"].eq("nibrs_annual_support_when_srs_not_usable"),
            out["regime_reason"].eq("srs_lumpiness_signal"),
            out["regime_reason"].eq("srs_month_mask_mismatch"),
            out["regime_reason"].eq("srs_annual_observation_without_reliable_monthly_support"),
            out["reporting_regime"].eq("structurally_missing_or_unreliable"),
        ],
        [
            "nibrs_only",
            "srs_lumpiness_signal",
            "srs_month_mask_mismatch",
            "srs_annual_only",
            "structural_missing_audit",
        ],
        default="other_review",
    )

    out["review_priority_score"] = pd.to_numeric(out["support_weight"], errors="coerce").fillna(0.0)
    out = out.sort_values(
        ["review_priority_score", "review_lane", "state_fips", "ori9", "offense"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).head(int(top_n))

    out["lane_rank"] = out.groupby("review_lane", dropna=False).cumcount() + 1

    out_path = paths.review_queues_dir / "reporting" / "reporting_regime_review_queue.parquet"
    out_csv = out_path.with_suffix(".csv")
    out.to_parquet(out_path, index=False)
    out.to_csv(out_csv, index=False)

    lane_specs = {
        "reporting_regime_lumpy_pilot": ("srs_lumpiness_signal", 30),
        "reporting_regime_nibrs_only_pilot": ("nibrs_only", 30),
        "reporting_regime_month_mask_pilot": ("srs_month_mask_mismatch", 20),
        "reporting_regime_annual_only_pilot": ("srs_annual_only", 30),
        "reporting_regime_structural_audit_pilot": ("structural_missing_audit", 20),
    }
    for stem, (lane, limit) in lane_specs.items():
        lane_df = out[out["review_lane"].eq(lane)].copy().head(limit)
        if lane_df.empty:
            continue
        lane_path = out_path.with_name(f"{stem}.parquet")
        lane_df.to_parquet(lane_path, index=False)
        lane_df.to_csv(lane_path.with_suffix(".csv"), index=False)
    return out_path


def _load_overlap_weights(paths: RepoPaths) -> pd.DataFrame | None:
    obs_path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    if not obs_path.exists():
        return None
    obs = pd.read_parquet(obs_path)
    obs = obs[
        obs["year"].astype(int).eq(2024)
        & obs["source"].isin([CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE, NIBRS_SOURCE])
    ][["ori9", "offense", "source", "count"]].copy()
    if obs.empty:
        return None
    preferred = (
        obs.pivot_table(index=["ori9", "offense"], columns="source", values="count", aggfunc="sum")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for source in [CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE, NIBRS_SOURCE]:
        if source not in preferred.columns:
            preferred[source] = np.nan
    preferred["preferred_source"] = initialize_preferred_source(
        has_cius=pd.to_numeric(preferred[CIUS_SOURCE], errors="coerce").notna(),
        has_local_publication=pd.to_numeric(preferred[LOCAL_PUBLICATION_SOURCE], errors="coerce").notna(),
        has_state_publication=pd.to_numeric(preferred[STATE_PUBLICATION_SOURCE], errors="coerce").notna(),
        has_srs=pd.to_numeric(preferred[SUMMARY_SOURCE], errors="coerce").notna(),
        has_nibrs=pd.to_numeric(preferred[NIBRS_SOURCE], errors="coerce").notna(),
    )
    preferred = assign_preferred_value(
        preferred,
        output_col="preferred_count",
        preferred_source_col="preferred_source",
        source_to_input_col={
            CIUS_SOURCE: CIUS_SOURCE,
            LOCAL_PUBLICATION_SOURCE: LOCAL_PUBLICATION_SOURCE,
            STATE_PUBLICATION_SOURCE: STATE_PUBLICATION_SOURCE,
            SUMMARY_SOURCE: SUMMARY_SOURCE,
            NIBRS_SOURCE: NIBRS_SOURCE,
        },
        default_source=NIBRS_SOURCE,
    )
    preferred["preferred_count"] = pd.to_numeric(preferred["preferred_count"], errors="coerce").fillna(0.0)
    out = (
        preferred.groupby("ori9", dropna=False)["preferred_count"]
        .sum()
        .rename("overlap_weight_2024")
        .reset_index()
    )
    out["ori"] = out["ori9"].astype("string")
    return out[["ori", "overlap_weight_2024"]].copy()


def _pick_top_rows(frame: pd.DataFrame, mask: pd.Series, *, n: int) -> pd.DataFrame:
    subset = frame.loc[mask].copy()
    if subset.empty:
        return subset
    return subset.sort_values(
        ["overlap_weight_2024", "state_fips", "ori"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(n)


def _build_overlap_localization_scaffolds(paths: RepoPaths) -> dict[str, str]:
    queue_path = paths.review_queues_dir / "overlap" / "overlap_localization_queue.parquet"
    if not queue_path.exists():
        return {}
    queue = pd.read_parquet(queue_path).copy()
    if queue.empty:
        return {}
    if "overlap_weight_2024" in queue.columns:
        queue = queue.drop(columns=["overlap_weight_2024"])

    weights = _load_overlap_weights(paths)
    if weights is None:
        queue["overlap_weight_2024"] = 0.0
    else:
        queue = queue.merge(weights, on="ori", how="left")
        queue["overlap_weight_2024"] = pd.to_numeric(queue["overlap_weight_2024"], errors="coerce").fillna(0.0)

    queue["case_id"] = queue["ori"].astype("string")
    queue["state_fips"] = queue["state_fips"].astype("string").str.zfill(2)
    queue["county_fips"] = queue["county_fips"].astype("string")
    queue["place_fips"] = queue["place_fips"].astype("string")
    queue["has_anchor_place"] = queue["place_fips"].str.fullmatch(r"\d{5}").fillna(False)
    queue["has_anchor_county"] = queue["county_fips"].str.fullmatch(r"\d{3}").fillna(False)
    queue["has_pseudo_place"] = queue["place_fips"].str.match(r"99\d{3}", na=False)

    geom = queue["geometry_hint"].fillna("").astype(str).str.lower()
    subtype = queue["overlap_subtype"].fillna("").astype(str)
    queue["geometry_hint_lower"] = geom
    queue["is_network_geometry"] = geom.str.contains(r"network|system|multiagency|authority_or_network", regex=True)
    queue["is_airport_or_port_geometry"] = geom.str.contains(r"airport|port", regex=True)
    queue["is_campus_geometry"] = geom.str.contains(r"campus|university", regex=True)
    queue["is_tribal_geometry"] = geom.str.contains(r"tribal|reservation", regex=True)
    queue["is_facility_geometry"] = geom.str.contains(r"facility|airport|port|campus|park|school|education", regex=True)
    queue["anchor_quality"] = np.select(
        [
            queue["has_anchor_place"] & (~queue["has_pseudo_place"]),
            queue["has_anchor_place"] & queue["has_pseudo_place"],
            (~queue["has_anchor_place"]) & queue["has_anchor_county"],
        ],
        ["normal_place", "pseudo_place", "county_only"],
        default="unanchored",
    )

    queue["likely_footprint_required"] = (
        subtype.isin(["transit", "tribal", "other_special"])
        | queue["is_network_geometry"]
        | queue["is_tribal_geometry"]
        | queue["is_airport_or_port_geometry"]
        | (subtype.isin(["transport_hub", "local_special"]) & queue["has_pseudo_place"])
    )
    queue["likely_proxy_ok"] = (
        queue["has_anchor_county"]
        & (~queue["likely_footprint_required"])
        & subtype.eq("campus")
    ) | (
        queue["has_anchor_place"]
        & (~queue["has_pseudo_place"])
        & (~queue["likely_footprint_required"])
        & (
            subtype.eq("campus")
            | (subtype.eq("local_special") & geom.str.contains(r"school|education_system", regex=True))
        )
    )

    queue = queue.sort_values(
        ["overlap_weight_2024", "state_fips", "ori"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    queue["impact_rank_national"] = np.arange(1, len(queue) + 1)
    queue["impact_rank_within_subtype"] = (
        queue.groupby("overlap_subtype", dropna=False)["overlap_weight_2024"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    overrides_path = paths.repo_root / "configs" / "overlap_footprint_overrides.csv"
    if overrides_path.exists() and not queue.empty:
        overrides = pd.read_csv(overrides_path)
        if "ori" in overrides.columns:
            resolved_oris = set(overrides["ori"].dropna().astype(str))
            if resolved_oris:
                queue = queue[~queue["ori"].astype("string").isin(resolved_oris)].copy()
                queue = queue.sort_values(
                    ["overlap_weight_2024", "state_fips", "ori"],
                    ascending=[False, True, True],
                    kind="mergesort",
                ).reset_index(drop=True)
                queue["impact_rank_national"] = np.arange(1, len(queue) + 1)
                queue["impact_rank_within_subtype"] = (
                    queue.groupby("overlap_subtype", dropna=False)["overlap_weight_2024"]
                    .rank(method="first", ascending=False)
                    .astype(int)
                )

    queue["defer_low_impact"] = (
        queue["overlap_weight_2024"].le(0.0)
        | (
            subtype.isin(["other_special", "tribal"])
            & queue["overlap_weight_2024"].lt(100.0)
        )
        | (
            (~queue["likely_proxy_ok"])
            & (~queue["likely_footprint_required"])
            & queue["overlap_weight_2024"].lt(25.0)
        )
    )
    queue["suggested_worker_tier"] = np.select(
        [
            queue["defer_low_impact"],
            queue["likely_proxy_ok"],
            queue["likely_footprint_required"] & queue["overlap_weight_2024"].gt(0.0),
        ],
        ["defer", "proxy", "footprint"],
        default="defer",
    )

    out_dir = paths.review_queues_dir / "overlap"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    queue.to_parquet(queue_path, index=False)
    queue.to_csv(queue_path.with_suffix(".csv"), index=False)
    outputs["overlap_localization_queue"] = str(queue_path)

    enriched_path = out_dir / "overlap_localization_enriched.parquet"
    queue.to_parquet(enriched_path, index=False)
    queue.to_csv(enriched_path.with_suffix(".csv"), index=False)
    outputs["overlap_localization_enriched"] = str(enriched_path)

    proxy_main = queue.loc[queue["suggested_worker_tier"].eq("proxy")].head(300).copy()
    footprint_main = queue.loc[queue["suggested_worker_tier"].eq("footprint")].head(60).copy()
    proxy_compare = queue.loc[queue["suggested_worker_tier"].eq("proxy")].head(25).copy()
    footprint_compare = queue.loc[queue["suggested_worker_tier"].eq("footprint")].head(25).copy()

    mixed_parts = [
        _pick_top_rows(queue, subtype.eq("campus") & queue["likely_proxy_ok"], n=4),
        _pick_top_rows(queue, subtype.eq("campus") & (~queue["likely_proxy_ok"]), n=3),
        _pick_top_rows(queue, subtype.eq("local_special") & queue["likely_proxy_ok"], n=4),
        _pick_top_rows(queue, subtype.eq("local_special") & queue["likely_footprint_required"], n=3),
        _pick_top_rows(queue, subtype.eq("transit"), n=4),
        _pick_top_rows(queue, subtype.eq("transport_hub"), n=4),
        _pick_top_rows(queue, subtype.eq("tribal"), n=1),
        _pick_top_rows(queue, subtype.eq("other_special"), n=1),
    ]
    mixed = pd.concat(mixed_parts, ignore_index=True).drop_duplicates(subset=["ori"], keep="first")
    mixed = mixed.sort_values(
        ["overlap_weight_2024", "state_fips", "ori"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    staged = {
        "overlap_proxy_main_queue": proxy_main,
        "overlap_footprint_main_queue": footprint_main,
        "overlap_proxy_compare_queue": proxy_compare,
        "overlap_footprint_compare_queue": footprint_compare,
        "overlap_mixed_ugly_pilot_queue": mixed,
    }
    for name, frame in staged.items():
        if frame.empty:
            continue
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        frame.to_csv(path.with_suffix(".csv"), index=False)
        outputs[name] = str(path)

    return outputs


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _build_city_incident_seed(paths: RepoPaths) -> Path | None:
    categories_dir = paths.repo_root / "configs" / "city_incident_categories"
    source_cols = CONFIG_SPECS["city_incident_sources.csv"]
    priority_cols = CONFIG_SPECS["city_incident_priority.csv"]
    source_cfg = paths.repo_root / "configs" / "city_incident_sources.csv"
    priority_cfg = paths.repo_root / "configs" / "city_incident_priority.csv"

    canonical_sources = pd.DataFrame(columns=source_cols)
    if source_cfg.exists():
        canonical_sources = pd.read_csv(source_cfg, dtype=str).fillna("")
        canonical_sources = canonical_sources[source_cols].copy()

    canonical_priority = pd.DataFrame(columns=priority_cols)
    if priority_cfg.exists():
        canonical_priority = pd.read_csv(priority_cfg, dtype=str).fillna("")
        canonical_priority = canonical_priority[priority_cols].copy()

    extra_rows: list[dict[str, object]] = []
    if categories_dir.exists():
        seen_city_keys = set(canonical_sources.get("city_key", pd.Series(dtype=str)).astype(str))
        for path in sorted(categories_dir.glob("*.csv")):
            stem = path.stem
            city_part = stem
            for prefix in ["categories_", "local_categories_", "location_types_"]:
                if city_part.startswith(prefix):
                    city_part = city_part[len(prefix):]
                    break
            city_name = city_part.replace("_temp", "").replace("_", " ").strip().title()
            if not city_name:
                continue
            city_key = _slug(city_name)
            if city_key in seen_city_keys:
                continue
            extra_rows.append(
                {
                    "city_key": city_key,
                    "jurisdiction_id": "",
                    "city_name": city_name,
                    "state_abbr": "",
                    "source_name": "local_city_incident_categories",
                    "source_url": str(path),
                    "portal_type": "local_repo_seed",
                    "coverage_start_year": "",
                    "coverage_end_year": "",
                    "years_usable": "",
                    "offense_fields_present": "",
                    "date_field": "",
                    "location_fields_present": "",
                    "latlon_present": "",
                    "address_present": "",
                    "block_group_join_ready": "",
                    "geocode_quality_tier": "",
                    "dedupe_key_available": "",
                    "offense_crosswalk_complexity": "unknown_pending_review",
                    "recommended_disposition": "needs_schema_work",
                    "analyst_notes": "Seeded from in-repo city incident category CSVs; review actual source coverage and ETL readiness.",
                }
            )

    extra_sources = pd.DataFrame(extra_rows, columns=source_cols)
    seed = pd.concat([canonical_sources, extra_sources], ignore_index=True)
    seed = seed.drop_duplicates(subset=["city_key", "source_url"]).sort_values(["city_name", "source_url"], kind="mergesort")
    if seed.empty:
        return None

    out_path = _review_seed_path(paths, "city_incident_source_seed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seed.to_csv(out_path, index=False)

    base_priority = canonical_priority.copy()
    seen_priority_keys = set(base_priority.get("city_key", pd.Series(dtype=str)).astype(str))
    extra_priority = seed[["city_key", "jurisdiction_id", "city_name", "state_abbr"]].drop_duplicates().copy()
    extra_priority = extra_priority[~extra_priority["city_key"].isin(seen_priority_keys)].copy()
    if not extra_priority.empty:
        max_rank = pd.to_numeric(base_priority.get("priority_rank"), errors="coerce").max()
        start_rank = int(max_rank) + 1 if pd.notna(max_rank) else 1
        extra_priority["priority_bucket"] = "local_repo_lead"
        extra_priority["priority_rank"] = range(start_rank, start_rank + len(extra_priority))
        extra_priority["basis"] = "Seeded from in-repo city incident category CSVs."
        extra_priority["notes"] = "Needs source/coverage/geocode review before onboarding."
        base_priority = pd.concat([base_priority, extra_priority[priority_cols]], ignore_index=True)

    priority_out = _review_seed_path(paths, "city_incident_priority_seed.csv")
    priority_out.parent.mkdir(parents=True, exist_ok=True)
    base_priority.to_csv(priority_out, index=False)
    return out_path


def _build_city_packet_scaffolds(paths: RepoPaths) -> dict[str, str]:
    outputs: dict[str, str] = {}
    source_cfg = REPO_ROOT / "configs" / "city_incident_sources.csv"
    priority_cfg = REPO_ROOT / "configs" / "city_incident_priority.csv"
    if not source_cfg.exists() or not priority_cfg.exists():
        return outputs

    sources = pd.read_csv(source_cfg, dtype=str).fillna("")
    priority = pd.read_csv(priority_cfg, dtype=str).fillna("")
    if sources.empty or priority.empty:
        return outputs

    packet_root = paths.review_packets_dir / "city"
    packet_root.mkdir(parents=True, exist_ok=True)

    merged = priority.merge(
        sources,
        on=["city_key", "jurisdiction_id", "city_name", "state_abbr"],
        how="left",
        suffixes=("_priority", "_source"),
    )
    merged["priority_rank_numeric"] = pd.to_numeric(merged["priority_rank"], errors="coerce")
    merged = merged.sort_values(
        ["priority_bucket", "priority_rank_numeric", "city_key", "source_name"],
        kind="mergesort",
    )
    checklist = pd.DataFrame(CITY_PACKET_CHECKLIST)
    manifest_rows: list[dict[str, object]] = []

    for row in merged.to_dict(orient="records"):
        city_key = str(row.get("city_key", "")).strip()
        if not city_key:
            continue
        packet_dir = packet_root / city_key
        packet_dir.mkdir(parents=True, exist_ok=True)

        source_row = pd.DataFrame([row]).drop(columns=["priority_rank_numeric"], errors="ignore")
        source_row.to_csv(packet_dir / "source_candidate.csv", index=False)
        checklist_path = packet_dir / "packet_checklist.csv"
        if not checklist_path.exists():
            checklist.to_csv(checklist_path, index=False)

        packet_status_path = packet_dir / "packet_status.csv"
        if not packet_status_path.exists():
            packet_status = pd.DataFrame(
                [
                    {
                        "city_key": city_key,
                        "city_name": row.get("city_name", ""),
                        "jurisdiction_id": row.get("jurisdiction_id", ""),
                        "state_abbr": row.get("state_abbr", ""),
                        "priority_bucket": row.get("priority_bucket", ""),
                        "priority_rank": row.get("priority_rank", ""),
                        "packet_status": "scaffolded",
                        "current_owner": "",
                        "production_ready": "",
                        "city_share_integration_status": "",
                        "reconciliation_status": "",
                        "notes": "",
                    }
                ]
            )
            packet_status.to_csv(packet_status_path, index=False)

        manifest = {
            "city_key": city_key,
            "city_name": row.get("city_name", ""),
            "jurisdiction_id": row.get("jurisdiction_id", ""),
            "state_abbr": row.get("state_abbr", ""),
            "priority_bucket": row.get("priority_bucket", ""),
            "priority_rank": row.get("priority_rank", ""),
            "basis": row.get("basis", ""),
            "notes": row.get("notes", ""),
            "source_name": row.get("source_name", ""),
            "source_url": row.get("source_url", ""),
            "portal_type": row.get("portal_type", ""),
            "coverage_start_year": row.get("coverage_start_year", ""),
            "coverage_end_year": row.get("coverage_end_year", ""),
            "years_usable": row.get("years_usable", ""),
            "recommended_disposition": row.get("recommended_disposition", ""),
            "geocode_quality_tier": row.get("geocode_quality_tier", ""),
            "packet_dir": str(packet_dir),
            "required_artifacts": [
                "source_candidate.csv",
                "packet_status.csv",
                "packet_checklist.csv",
                "research_findings.json",
                "reconciliation_summary.csv",
            ],
        }
        (packet_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2))
        manifest_rows.append({k: v for k, v in manifest.items() if k != "required_artifacts"})

    if manifest_rows:
        manifest_path = packet_root / "city_packet_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        outputs["city_packet_manifest"] = str(manifest_path)
        outputs["city_packet_root"] = str(packet_root)
    return outputs


def _build_model_diagnostic_seed(paths: RepoPaths) -> Path | None:
    summary_path = paths.state_dir / "modeling" / "jurisdiction_model_benchmark_2024.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    rows = []
    for offense, metrics in sorted(summary.get("per_offense", {}).items()):
        rows.append(
            {
                "offense": offense,
                "state_or_slice": "national",
                "diagnostic_type": "benchmark_baseline",
                "symptom": f"cv_r2_log_rate={metrics.get('cv_r2_log_rate')}",
                "suspected_cause": pd.NA,
                "evidence_note": "Seed row from current jurisdiction benchmark; extend with state/slice-specific diagnostics.",
                "priority": "high" if float(metrics.get("cv_r2_log_rate", 0.0)) < 0.4 else "medium",
                "recommended_followup": "Review missing features, reporting structure, and overlap contamination for weak slices.",
            }
        )
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out_path = _review_seed_path(paths, "model_diagnostic_seed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create V2 review config templates and seed queues.")
    parser.add_argument("--top-reporting-queue", type=int, default=1500)
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    created_templates = []
    for filename, cols in CONFIG_SPECS.items():
        path = _config_path(paths, filename)
        if _ensure_template(path, cols):
            created_templates.append(str(path))

    outputs = {
        "created_templates": created_templates,
        "feature_inventory_seed": str(_build_feature_inventory_seed(paths) or ""),
        "reporting_regime_review_queue": str(_build_reporting_regime_review_queue(paths, top_n=int(args.top_reporting_queue)) or ""),
        "city_incident_seed": str(_build_city_incident_seed(paths) or ""),
        "model_diagnostic_seed": str(_build_model_diagnostic_seed(paths) or ""),
    }
    outputs.update(_build_overlap_localization_scaffolds(paths))
    outputs.update(_build_city_packet_scaffolds(paths))
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
