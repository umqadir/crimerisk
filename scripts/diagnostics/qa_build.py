from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_shares import _city_impls, _load_enabled_city_order
from crimerisk.crime import OFFENSES_7
from crimerisk.geometry import _build_missing_municipal_bg_allocations
from crimerisk.paths import get_paths


def _max_state_offense_diff(
    *,
    state_controls: pd.DataFrame,
    bg_output: pd.DataFrame,
    target_col: str,
) -> tuple[float, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for offense in OFFENSES_7:
        bg_sum = (
            bg_output.groupby("state_fips", dropna=False)[f"expected_count_{offense}"]
            .sum()
            .rename("bg_total")
            .reset_index()
        )
        merged = (
            state_controls[state_controls["offense"].eq(offense)][
                ["state_fips", "state_abbr", target_col]
            ]
            .merge(bg_sum, on="state_fips", how="left")
        )
        merged["bg_total"] = pd.to_numeric(merged["bg_total"], errors="coerce").fillna(0.0)
        merged[target_col] = pd.to_numeric(merged[target_col], errors="coerce").fillna(0.0)
        merged["diff"] = merged["bg_total"] - merged[target_col]
        merged["offense"] = offense
        rows.append(merged)
    detail = pd.concat(rows, ignore_index=True)
    max_abs = float(pd.to_numeric(detail["diff"], errors="coerce").abs().max())
    return max_abs, detail


def _top_state_offense_diff_rows(detail: pd.DataFrame, *, limit: int = 10) -> list[dict[str, object]]:
    out = detail.copy()
    out["abs_diff"] = pd.to_numeric(out["diff"], errors="coerce").abs()
    top = out.sort_values(["abs_diff", "state_abbr", "offense"], ascending=[False, True, True], kind="mergesort").head(limit)
    rows: list[dict[str, object]] = []
    for row in top.itertuples(index=False):
        rows.append(
            {
                "state_fips": str(getattr(row, "state_fips")),
                "state_abbr": str(getattr(row, "state_abbr")),
                "offense": str(getattr(row, "offense")),
                "diff": float(getattr(row, "diff")),
                "abs_diff": float(getattr(row, "abs_diff")),
            }
        )
    return rows


def _city_key_from_name(city_name: object) -> str | None:
    text = str(city_name).strip().lower()
    mapping = {
        "baltimore": "baltimore",
        "new york": "new_york",
        "chicago": "chicago",
        "boston": "boston",
        "denver": "denver",
        "minneapolis": "minneapolis",
        "seattle": "seattle",
        "san francisco": "san_francisco",
        "austin": "austin",
        "mesa": "mesa",
        "philadelphia": "philadelphia",
        "washington": "washington_dc",
    }
    return mapping.get(text)


def _local_resolution_override_alignment(
    *,
    repo_root: Path,
    agency_master: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> tuple[int, list[dict[str, object]]]:
    override_path = repo_root / "configs" / "local_resolution_overrides.csv"
    if not override_path.exists():
        return 0, []

    overrides = pd.read_csv(override_path, dtype=str).fillna("").copy()
    if overrides.empty:
        return 0, []

    overrides["ori"] = overrides["ori"].astype(str).str.strip()
    overrides["decision"] = overrides["decision"].astype(str).str.strip()
    overrides["replacement_geo_type"] = overrides["replacement_geo_type"].astype(str).str.strip()
    overrides["replacement_geoid"] = overrides["replacement_geoid"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)

    agency_state = (
        agency_master[["ori9", "state_fips"]]
        .drop_duplicates()
        .rename(columns={"ori9": "ori"})
        .copy()
    )
    agency_state["ori"] = agency_state["ori"].astype(str).str.strip()
    agency_state["state_fips"] = agency_state["state_fips"].astype(str).str.zfill(2)

    crosswalk_min = crosswalk[["ori", "jurisdiction_id", "resolution_source"]].copy()
    crosswalk_min["ori"] = crosswalk_min["ori"].astype(str).str.strip()

    merged = overrides.merge(agency_state, on="ori", how="left").merge(crosswalk_min, on="ori", how="left")

    def _expected_jurisdiction_id(row: pd.Series) -> str | None:
        state_fips = str(row.get("state_fips", "")).strip().zfill(2)
        decision = str(row.get("decision", "")).strip()
        geo_type = str(row.get("replacement_geo_type", "")).strip()
        geoid = str(row.get("replacement_geoid", "")).strip()
        if not state_fips or state_fips == "00":
            return None
        if decision in {"municipal_place", "municipal_cousub"}:
            if geo_type not in {"place", "cousub"} or not geoid:
                return None
            width = 7 if geo_type == "place" else 10
            return f"{state_fips}:municipal:{geo_type}:{geoid.zfill(width)}"
        if decision == "reclassify_nonmunicipal":
            return f"{state_fips}:state_nonmunicipal_remainder"
        if decision == "reclassify_overlap":
            return f"{state_fips}:statewide_overlap_layer"
        return None

    merged["expected_jurisdiction_id"] = merged.apply(_expected_jurisdiction_id, axis=1)
    merged["expected_resolution_source"] = "local_resolution_override"
    mismatched = merged[
        merged["expected_jurisdiction_id"].notna()
        & (
            merged["jurisdiction_id"].astype(str).ne(merged["expected_jurisdiction_id"].astype(str))
            | merged["resolution_source"].astype(str).ne(merged["expected_resolution_source"])
        )
    ].copy()

    rows: list[dict[str, object]] = []
    for row in mismatched.itertuples(index=False):
        rows.append(
            {
                "ori": str(getattr(row, "ori")),
                "decision": str(getattr(row, "decision")),
                "expected_jurisdiction_id": str(getattr(row, "expected_jurisdiction_id")),
                "actual_jurisdiction_id": str(getattr(row, "jurisdiction_id")),
                "actual_resolution_source": str(getattr(row, "resolution_source")),
            }
        )
    return int(len(rows)), rows


def build_summary(repo_root: Path) -> dict[str, object]:
    base = repo_root / "state"
    paths = get_paths()
    jurisdiction_master = pd.read_parquet(base / "reference" / "jurisdiction_master.parquet")
    agency_master = pd.read_parquet(base / "reference" / "agency_master.parquet")
    agency_crosswalk = pd.read_parquet(base / "reference" / "agency_to_jurisdiction_crosswalk.parquet")
    observations = pd.read_parquet(base / "observations" / "jurisdiction_year_observations.parquet")
    controls = pd.read_parquet(base / "controls" / "jurisdiction_controls_2024.parquet")
    state_controls = pd.read_parquet(base / "controls" / "state_control_comparison.parquet")
    bg_crosswalk = pd.read_parquet(base / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet")
    bg_ags_path = base / "output" / "crimerisk_block_group_2024_ags_core.parquet"
    tract_ags_path = base / "output" / "crimerisk_tract_2024_ags_core.parquet"
    bg_ags = pd.read_parquet(bg_ags_path)
    bg_fbi_path = base / "output" / "crimerisk_block_group_2024_fbi_calibrated.parquet"
    tract_fbi_path = base / "output" / "crimerisk_tract_2024_fbi_calibrated.parquet"
    fbi_present = bg_fbi_path.exists() and tract_fbi_path.exists()
    fbi_current = (
        fbi_present
        and bg_fbi_path.stat().st_mtime >= bg_ags_path.stat().st_mtime
        and tract_fbi_path.stat().st_mtime >= tract_ags_path.stat().st_mtime
    )
    bg_fbi = pd.read_parquet(bg_fbi_path) if fbi_current else None
    city_share_path = base / "modeling" / "city_incident_share_surface.parquet"
    city_share = pd.read_parquet(city_share_path, columns=["city_name"]) if city_share_path.exists() else pd.DataFrame(columns=["city_name"])
    residual_adjusted_bg_prior_path = base / "modeling" / "bg_prior_residual_adjusted_2024.parquet"
    bg_ags_total_counts = pd.to_numeric(bg_ags.get("expected_count_total"), errors="coerce").fillna(0.0)
    bg_ags_population = pd.to_numeric(bg_ags.get("population_2024"), errors="coerce").fillna(0.0)
    zero_pop_positive_count = (
        bg_ags["population_zero_with_positive_count"].astype(bool)
        if "population_zero_with_positive_count" in bg_ags.columns
        else (bg_ags_population.le(0) & bg_ags_total_counts.gt(0))
    )
    resident_secondary_low_denominator = (
        bg_ags["resident_secondary_denominator_low_reliability"].astype(bool)
        if "resident_secondary_denominator_low_reliability" in bg_ags.columns
        else bg_ags_population.lt(25)
    )
    primary_low_denominator_cols = [
        col for col in bg_ags.columns if col.startswith("diagnostic_eb_low_denominator_flag_")
    ]
    primary_suppressed_cols = [
        col for col in bg_ags.columns if col.startswith("primary_index_suppressed_")
    ]
    any_primary_low_denominator = (
        bg_ags[primary_low_denominator_cols].fillna(False).astype(bool).any(axis=1)
        if primary_low_denominator_cols
        else pd.Series(False, index=bg_ags.index)
    )
    any_primary_suppressed = (
        bg_ags[primary_suppressed_cols].fillna(False).astype(bool).any(axis=1)
        if primary_suppressed_cols
        else pd.Series(False, index=bg_ags.index)
    )
    resident_index_cols = [f"index_{offense}_resident" for offense in OFFENSES_7 if f"index_{offense}_resident" in bg_ags.columns]
    extreme_resident_index_rows = (
        bg_ags[resident_index_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).gt(100000.0).any(axis=1)
        if resident_index_cols
        else pd.Series(False, index=bg_ags.index)
    )

    supported = set(bg_crosswalk["jurisdiction_id"].astype(str))
    control_municipal_index = controls[
        controls["jurisdiction_type"].eq("municipal")
    ][["jurisdiction_id", "jurisdiction_name", "state_abbr"]].drop_duplicates()
    missing_municipal_support_raw = controls[
        controls["jurisdiction_type"].eq("municipal")
        & (~controls["jurisdiction_id"].astype(str).isin(supported))
    ][["jurisdiction_id"]].drop_duplicates()
    control_municipal_master = jurisdiction_master[
        jurisdiction_master["jurisdiction_id"].astype(str).isin(
            set(control_municipal_index["jurisdiction_id"].astype(str))
        )
    ].copy()
    bg_fallback = _build_missing_municipal_bg_allocations(
        paths=paths,
        bg_crosswalk=bg_crosswalk,
        jurisdiction_master=control_municipal_master,
    )
    supported_with_fallback = supported | set(bg_fallback["jurisdiction_id"].astype(str))
    missing_municipal_support = controls[
        controls["jurisdiction_type"].eq("municipal")
        & (~controls["jurisdiction_id"].astype(str).isin(supported_with_fallback))
    ][["jurisdiction_id"]].drop_duplicates()

    ags_max_abs_diff, ags_detail = _max_state_offense_diff(
        state_controls=state_controls,
        bg_output=bg_ags,
        target_col="ags_core_adjusted_total",
    )
    fbi_max_abs_diff = None
    if bg_fbi is not None:
        fbi_max_abs_diff, _ = _max_state_offense_diff(
            state_controls=state_controls,
            bg_output=bg_fbi,
            target_col="fbi_cde_estimated_total",
        )

    enabled_city_keys = _load_enabled_city_order(paths)
    impls = _city_impls(paths)
    supported_city_keys = sorted(impls.keys())
    enabled_missing_required_inputs: list[str] = []
    for key in enabled_city_keys:
        impl = impls.get(key)
        if impl is None:
            continue
        required_paths = list(impl.get("required_paths", []))
        if any(path is None or not Path(path).exists() for path in required_paths):
            enabled_missing_required_inputs.append(key)
    live_city_keys = sorted(
        {
            key
            for key in city_share["city_name"].map(_city_key_from_name).dropna().tolist()
        }
    )
    local_override_mismatch_count, local_override_mismatch_rows = _local_resolution_override_alignment(
        repo_root=repo_root,
        agency_master=agency_master,
        crosswalk=agency_crosswalk,
    )

    return {
        "reference": {
            "jurisdiction_master_rows": int(len(jurisdiction_master)),
            "agency_crosswalk_rows": int(len(agency_crosswalk)),
            "jurisdiction_type_counts": {
                str(k): int(v) for k, v in jurisdiction_master["jurisdiction_type"].value_counts().to_dict().items()
            },
            "local_resolution_override_mismatch_count": local_override_mismatch_count,
            "local_resolution_override_mismatch_rows": local_override_mismatch_rows,
        },
        "observations": {
            "rows": int(len(observations)),
            "unique_jurisdictions": int(observations["jurisdiction_id"].nunique()),
        },
        "controls_2024": {
            "rows": int(len(controls)),
            "unique_jurisdictions": int(controls["jurisdiction_id"].nunique()),
            "jurisdiction_type_counts": {
                str(k): int(v) for k, v in controls["jurisdiction_type"].value_counts().to_dict().items()
            },
            "preferred_source_counts": {
                str(k): int(v) for k, v in controls["preferred_source"].value_counts(dropna=False).to_dict().items()
            },
        },
        "geometry": {
            "bg_crosswalk_rows": int(len(bg_crosswalk)),
            "bg_crosswalk_unique_jurisdictions": int(bg_crosswalk["jurisdiction_id"].nunique()),
            "missing_municipal_support_count_raw": int(len(missing_municipal_support_raw)),
            "missing_municipal_support_count": int(len(missing_municipal_support)),
            "missing_municipal_support_fallback_rows": int(len(bg_fallback)),
            "missing_municipal_support_rows_raw": (
                missing_municipal_support_raw
                .merge(control_municipal_index, on="jurisdiction_id", how="left")
                .sort_values(["state_abbr", "jurisdiction_name"], kind="mergesort")
                .to_dict(orient="records")
            ),
            "missing_municipal_support_rows": (
                missing_municipal_support
                .merge(control_municipal_index, on="jurisdiction_id", how="left")
                .sort_values(["state_abbr", "jurisdiction_name"], kind="mergesort")
                .to_dict(orient="records")
            ),
        },
        "outputs": {
            "residual_adjusted_bg_prior_present": residual_adjusted_bg_prior_path.exists(),
            "block_group_rows_ags_core": int(len(bg_ags)),
            "ags_core_max_abs_state_offense_diff": ags_max_abs_diff,
            "ags_core_top_state_offense_diffs": _top_state_offense_diff_rows(ags_detail, limit=10),
            "ags_core_primary_denominator_type_counts": {
                col.replace("primary_denominator_type_", ""): {
                    str(k): int(v)
                    for k, v in bg_ags[col].astype("string").value_counts(dropna=False).to_dict().items()
                }
                for col in bg_ags.columns
                if col.startswith("primary_denominator_type_")
            },
            "ags_core_primary_low_denominator_rows": int(any_primary_low_denominator.sum()),
            "ags_core_primary_suppressed_rows": int(any_primary_suppressed.sum()),
            "ags_core_resident_secondary_low_denominator_rows": int(resident_secondary_low_denominator.sum()),
            "ags_core_zero_population_positive_count_rows": int(zero_pop_positive_count.sum()),
            "ags_core_zero_population_positive_count_mass_share": (
                float(bg_ags_total_counts[zero_pop_positive_count].sum() / bg_ags_total_counts.sum())
                if float(bg_ags_total_counts.sum()) > 0
                else 0.0
            ),
            "ags_core_primary_suppressed_bg_ids": [
                str(value)
                for value in bg_ags.loc[any_primary_suppressed, "block_group_geoid"]
                .astype("string")
                .dropna()
                .head(20)
                .tolist()
            ],
            "ags_core_extreme_resident_index_rows_gt_100k": int(extreme_resident_index_rows.sum()),
            "fbi_calibrated_present": bool(fbi_present),
            "fbi_calibrated_current": bool(fbi_current),
            "fbi_calibrated_stale_vs_ags_core": bool(fbi_present and not fbi_current),
            "block_group_rows_fbi_calibrated": int(len(bg_fbi)) if bg_fbi is not None else None,
            "fbi_calibrated_max_abs_state_offense_diff": fbi_max_abs_diff,
        },
        "city_wiring": {
            "enabled_city_keys": enabled_city_keys,
            "supported_city_builders": supported_city_keys,
            "live_city_keys": live_city_keys,
            "enabled_missing_builder_support": sorted(set(enabled_city_keys) - set(supported_city_keys)),
            "enabled_missing_required_inputs": sorted(enabled_missing_required_inputs),
            "enabled_missing_live_artifact": sorted(set(enabled_city_keys) - set(live_city_keys)),
            "live_without_enabled_gate": sorted(set(live_city_keys) - set(enabled_city_keys)),
        },
    }


def main() -> None:
    summary = build_summary(REPO_ROOT)
    out_path = REPO_ROOT / "state" / "qa" / "build_qa_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    out_path.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
