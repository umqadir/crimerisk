from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths  # noqa: E402
from crimerisk.source_provenance import (  # noqa: E402
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    NIBRS_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    assign_preferred_value,
    initialize_preferred_source,
)


DEFAULT_QUEUE = "state/review/queues/overlap/overlap_localization_queue.parquet"
DEFAULT_OUT_DIR = "state/review/queues/overlap"
DEFAULT_PROXY_PILOT_SIZE = 24
DEFAULT_FOOTPRINT_PILOT_SIZE = 18
DEFAULT_PROXY_MAIN_SIZE = 200
DEFAULT_FOOTPRINT_MAIN_SIZE = 48
DEFAULT_PROXY_VERIFY_SIZE = 100
DEFAULT_DEFER_WEIGHT = 25.0
DEFAULT_FOOTPRINT_MIN_WEIGHT = 100.0

HIGH_FOOTPRINT_HINTS = {
    "network_or_system_footprint",
    "network_or_multiagency_footprint",
    "tribal_or_reservation_footprint",
    "airport_authority_footprint",
    "port_authority_footprint",
}

PROXY_FRIENDLY_HINTS = {
    "campus_or_university_footprint",
    "campus_footprint",
    "university_campus_footprint",
    "campus",
    "facility_or_authority_footprint",
    "authority_or_park_footprint",
    "education_system_or_school_district_footprint",
}

PRIMARY_SUBTYPE_ORDER = {
    "campus": 1,
    "transport_hub": 2,
    "transit": 3,
    "local_special": 4,
    "tribal": 5,
    "other_special": 6,
}


def _preferred_overlap_weight(paths: RepoPaths, year: int) -> pd.DataFrame:
    obs_path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    obs = pd.read_parquet(obs_path)
    obs = obs[
        (obs["year"].astype(int) == int(year))
        & (obs["source"].isin([CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE, NIBRS_SOURCE]))
    ][["ori9", "offense", "source", "count"]].copy()
    pivot = (
        obs.pivot_table(index=["ori9", "offense"], columns="source", values="count", aggfunc="sum")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for source in [CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE, NIBRS_SOURCE]:
        if source not in pivot.columns:
            pivot[source] = np.nan
    pivot["preferred_source"] = initialize_preferred_source(
        has_cius=pd.to_numeric(pivot[CIUS_SOURCE], errors="coerce").notna(),
        has_local_publication=pd.to_numeric(pivot[LOCAL_PUBLICATION_SOURCE], errors="coerce").notna(),
        has_state_publication=pd.to_numeric(pivot[STATE_PUBLICATION_SOURCE], errors="coerce").notna(),
        has_srs=pd.to_numeric(pivot[SUMMARY_SOURCE], errors="coerce").notna(),
        has_nibrs=pd.to_numeric(pivot[NIBRS_SOURCE], errors="coerce").notna(),
    )
    pivot = assign_preferred_value(
        pivot,
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
    pivot["preferred_count"] = pd.to_numeric(pivot["preferred_count"], errors="coerce").fillna(0.0)
    return (
        pivot.groupby("ori9", dropna=False)["preferred_count"]
        .sum()
        .rename("overlap_weight_2024")
        .reset_index()
    )


def _normalize_fips(series: pd.Series, width: int) -> pd.Series:
    out = series.astype("string").str.extract(r"(\d+)")[0]
    return out.str.zfill(width)


def _shape_queue(
    queue: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    defer_weight: float,
    footprint_min_weight: float,
) -> pd.DataFrame:
    df = queue.copy()
    df["ori"] = df["ori"].astype("string")
    df = df.merge(weights.rename(columns={"ori9": "ori"}), on="ori", how="left")
    df["overlap_weight_2024"] = pd.to_numeric(df["overlap_weight_2024"], errors="coerce").fillna(0.0)
    df["state_fips"] = _normalize_fips(df["state_fips"], 2)
    df["county_fips"] = _normalize_fips(df["county_fips"], 3)
    df["place_fips"] = _normalize_fips(df["place_fips"], 5)
    df["geometry_hint_norm"] = df["geometry_hint"].fillna("").astype(str).str.strip().str.lower()
    df["overlap_subtype"] = df["overlap_subtype"].fillna("").astype(str)
    df["case_id"] = df["ori"]
    df["has_anchor_place"] = df["place_fips"].str.fullmatch(r"\d{5}").fillna(False)
    df["has_anchor_county"] = df["county_fips"].str.fullmatch(r"\d{3}").fillna(False)
    df["has_pseudo_place"] = df["place_fips"].str.match(r"99\d{3}", na=False)
    df["has_any_anchor"] = df["has_anchor_place"] | df["has_anchor_county"]
    df["subtype_priority"] = df["overlap_subtype"].map(PRIMARY_SUBTYPE_ORDER).fillna(99).astype(int)

    is_network_like = df["geometry_hint_norm"].isin(HIGH_FOOTPRINT_HINTS)
    is_proxy_friendly = df["geometry_hint_norm"].isin(PROXY_FRIENDLY_HINTS)
    subtype = df["overlap_subtype"]

    df["likely_footprint_required"] = (
        subtype.isin(["tribal", "other_special"])
        | is_network_like
        | (df["has_pseudo_place"] & subtype.isin(["transit", "transport_hub", "local_special"]))
        | (~df["has_any_anchor"] & subtype.isin(["transit", "transport_hub", "tribal", "other_special"]))
        | (
            subtype.eq("campus")
            & (~df["has_any_anchor"])
            & (
                df["geometry_hint_norm"].str.contains("campus", regex=False)
                | (df["overlap_weight_2024"] >= footprint_min_weight)
            )
        )
        | ((df["overlap_weight_2024"] >= footprint_min_weight) & subtype.eq("transport_hub") & df["has_pseudo_place"])
    )

    df["likely_proxy_ok"] = (
        df["has_any_anchor"]
        & (~df["likely_footprint_required"])
        & (
            subtype.eq("campus")
            | (subtype.eq("local_special") & is_proxy_friendly)
            | (subtype.eq("transport_hub") & df["geometry_hint_norm"].isin(["airport_footprint", "airport", "facility_or_authority_footprint"]))
        )
    )

    df["defer_low_impact"] = (
        (df["overlap_weight_2024"] < defer_weight)
        & (
            subtype.isin(["other_special", "tribal"])
            | (~df["has_any_anchor"])
        )
    ) | ((df["overlap_weight_2024"] <= 0.0) & (~df["likely_proxy_ok"]))

    df["review_lane"] = np.select(
        [
            df["defer_low_impact"],
            df["likely_footprint_required"] & (df["overlap_weight_2024"] >= footprint_min_weight),
            df["likely_proxy_ok"],
            df["has_any_anchor"] & subtype.isin(["campus", "local_special", "transport_hub", "transit"]),
        ],
        [
            "defer",
            "footprint_resolution",
            "proxy_resolution",
            "footprint_resolution",
        ],
        default="defer",
    )

    df["anchor_quality"] = np.select(
        [
            df["has_anchor_place"] & (~df["has_pseudo_place"]),
            df["has_anchor_place"] & df["has_pseudo_place"],
            (~df["has_anchor_place"]) & df["has_anchor_county"],
        ],
        [
            "normal_place",
            "pseudo_place",
            "county_only",
        ],
        default="no_anchor",
    )

    df = df.sort_values(
        ["overlap_weight_2024", "subtype_priority", "state_fips", "ori"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["impact_rank_national"] = np.arange(1, len(df) + 1)
    df["impact_rank_within_subtype"] = (
        df.groupby("overlap_subtype", dropna=False).cumcount() + 1
    )
    total_weight = float(df["overlap_weight_2024"].sum())
    df["cumulative_weight_share_national"] = (
        df["overlap_weight_2024"].cumsum() / total_weight if total_weight > 0 else 0.0
    )
    df["priority_score"] = (
        df["overlap_weight_2024"]
        * np.where(df["review_lane"].eq("proxy_resolution"), 1.0, np.where(df["review_lane"].eq("footprint_resolution"), 1.1, 0.2))
        * np.where(df["anchor_quality"].eq("normal_place"), 1.05, np.where(df["anchor_quality"].eq("county_only"), 1.0, np.where(df["anchor_quality"].eq("pseudo_place"), 0.85, 0.7)))
    )
    df["requires_proxy_verification"] = (
        df["review_lane"].eq("proxy_resolution")
        & (
            (df["overlap_subtype"].eq("campus") & (df["overlap_weight_2024"] >= 500.0))
            | (df["anchor_quality"] != "normal_place")
            | df["geometry_hint_norm"].str.contains("authority", regex=False)
        )
    )
    return df


def _select_diverse_pilot(df: pd.DataFrame, lane: str, target_size: int) -> pd.DataFrame:
    lane_df = df[df["review_lane"].eq(lane)].copy()
    if lane_df.empty:
        return lane_df
    subtype_frames: list[pd.DataFrame] = []
    for subtype in ["campus", "transport_hub", "transit", "local_special", "tribal", "other_special"]:
        part = lane_df[lane_df["overlap_subtype"].eq(subtype)].head(4).copy()
        if not part.empty:
            part["pilot_round"] = np.arange(len(part))
            subtype_frames.append(part)
    if subtype_frames:
        pilot = (
            pd.concat(subtype_frames, ignore_index=False)
            .sort_values(["pilot_round", "subtype_priority", "overlap_weight_2024"], ascending=[True, True, False], kind="mergesort")
            .drop(columns="pilot_round")
            .drop_duplicates("ori", keep="first")
        )
    else:
        pilot = lane_df.head(0).copy()
    if len(pilot) < target_size:
        remaining = lane_df.loc[~lane_df["ori"].isin(pilot["ori"])]
        need = max(0, target_size - len(pilot))
        edge = remaining[
            remaining["anchor_quality"].isin(["pseudo_place", "county_only"])
            | remaining["geometry_hint_norm"].str.contains("network|authority|tribal|port|airport", regex=True)
        ]
        add = pd.concat([edge, remaining], ignore_index=False).drop_duplicates("ori", keep="first").head(need)
        pilot = pd.concat([pilot, add], ignore_index=False).drop_duplicates("ori", keep="first")
    return pilot.head(target_size).copy()


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic overlap localization experiment queues.")
    parser.add_argument("--queue", type=Path, default=Path(DEFAULT_QUEUE))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--proxy-pilot-size", type=int, default=DEFAULT_PROXY_PILOT_SIZE)
    parser.add_argument("--footprint-pilot-size", type=int, default=DEFAULT_FOOTPRINT_PILOT_SIZE)
    parser.add_argument("--proxy-main-size", type=int, default=DEFAULT_PROXY_MAIN_SIZE)
    parser.add_argument("--footprint-main-size", type=int, default=DEFAULT_FOOTPRINT_MAIN_SIZE)
    parser.add_argument("--proxy-verify-size", type=int, default=DEFAULT_PROXY_VERIFY_SIZE)
    parser.add_argument("--defer-weight", type=float, default=DEFAULT_DEFER_WEIGHT)
    parser.add_argument("--footprint-min-weight", type=float, default=DEFAULT_FOOTPRINT_MIN_WEIGHT)
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    queue_path = (REPO_ROOT / args.queue).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    queue = pd.read_parquet(queue_path)
    weights = _preferred_overlap_weight(paths, year=args.year)
    enriched = _shape_queue(
        queue,
        weights,
        defer_weight=float(args.defer_weight),
        footprint_min_weight=float(args.footprint_min_weight),
    )

    proxy_pilot = _select_diverse_pilot(enriched, "proxy_resolution", int(args.proxy_pilot_size))
    footprint_pilot = _select_diverse_pilot(enriched, "footprint_resolution", int(args.footprint_pilot_size))

    proxy_main = (
        enriched[
            enriched["review_lane"].eq("proxy_resolution")
            & (~enriched["ori"].isin(proxy_pilot["ori"]))
        ]
        .head(int(args.proxy_main_size))
        .copy()
    )
    footprint_main = (
        enriched[
            enriched["review_lane"].eq("footprint_resolution")
            & (~enriched["ori"].isin(footprint_pilot["ori"]))
        ]
        .head(int(args.footprint_main_size))
        .copy()
    )
    proxy_verify = (
        enriched[
            enriched["requires_proxy_verification"]
            & (~enriched["ori"].isin(proxy_pilot["ori"]))
        ]
        .head(int(args.proxy_verify_size))
        .copy()
    )
    deferred = enriched[enriched["review_lane"].eq("defer")].copy()

    outputs = {
        out_dir / "overlap_localization_experiment_queue.parquet": enriched,
        out_dir / "overlap_localization_experiment_queue.csv": enriched,
        out_dir / "overlap_localization_proxy_pilot.parquet": proxy_pilot,
        out_dir / "overlap_localization_proxy_pilot.csv": proxy_pilot,
        out_dir / "overlap_localization_proxy_main.parquet": proxy_main,
        out_dir / "overlap_localization_proxy_main.csv": proxy_main,
        out_dir / "overlap_localization_footprint_pilot.parquet": footprint_pilot,
        out_dir / "overlap_localization_footprint_pilot.csv": footprint_pilot,
        out_dir / "overlap_localization_footprint_main.parquet": footprint_main,
        out_dir / "overlap_localization_footprint_main.csv": footprint_main,
        out_dir / "overlap_localization_proxy_verify.parquet": proxy_verify,
        out_dir / "overlap_localization_proxy_verify.csv": proxy_verify,
        out_dir / "overlap_localization_deferred_tail.parquet": deferred,
        out_dir / "overlap_localization_deferred_tail.csv": deferred,
    }
    for path, df in outputs.items():
        _write(df, path)

    summary = {
        "rows_total": int(len(enriched)),
        "weight_total": float(enriched["overlap_weight_2024"].sum()),
        "lane_counts": enriched["review_lane"].value_counts(dropna=False).to_dict(),
        "lane_weight": enriched.groupby("review_lane", dropna=False)["overlap_weight_2024"].sum().to_dict(),
        "proxy_pilot_rows": int(len(proxy_pilot)),
        "footprint_pilot_rows": int(len(footprint_pilot)),
        "proxy_main_rows": int(len(proxy_main)),
        "footprint_main_rows": int(len(footprint_main)),
        "proxy_verify_rows": int(len(proxy_verify)),
    }
    print(summary)


if __name__ == "__main__":
    main()
