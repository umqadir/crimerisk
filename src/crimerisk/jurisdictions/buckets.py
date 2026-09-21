from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from shapely.geometry import Point

from crimerisk.geo.spatial import PointMatch, assign_points_to_polygons
from crimerisk.geo.tiger import (
    TigerPolygon,
    load_tiger_cousub_polygons,
    load_tiger_place_polygons,
    polygon_code_set,
)


@dataclass(frozen=True)
class BucketConfig:
    prefer_place_when_ambiguous: bool = True


def load_tract_centroids(*, centroids_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(centroids_parquet)
    df["tract_id"] = df["tract_id"].astype(str).str.zfill(11)
    df["state_fips"] = df["tract_id"].str.slice(0, 2)
    return df[["tract_id", "state_fips", "lat", "lon"]]


def build_municipal_registry_from_canonical_crime_series(
    canonical_df: pd.DataFrame, *, min_any_year_total: float = 0.0
) -> pd.DataFrame:
    required = {"state_fips", "place_fips"}
    missing = required - set(canonical_df.columns)
    if missing:
        raise ValueError(f"canonical_df missing columns: {sorted(missing)}")

    work = canonical_df.copy()
    work["state_fips"] = work["state_fips"].astype(str).str.zfill(2)
    work["place_fips"] = work["place_fips"].astype(str).str.zfill(5)

    offenses = [
        c
        for c in canonical_df.columns
        if c
        in {
            "murder",
            "rape",
            "robbery",
            "aggravated_assault",
            "burglary",
            "larceny",
            "motor_vehicle_theft",
        }
    ]

    # Exclude codes that never report any crime data (all agencies have 0 months reported and
    # sources marked missing across the series). This prevents allocating zero-crime municipal
    # constraints to areas that should fall into the state residual bucket.
    if "months_reported" in work.columns and any(f"source_{o}" in work.columns for o in offenses):
        months = pd.to_numeric(work["months_reported"], errors="coerce").fillna(0)
        has_any_source = pd.Series(False, index=work.index)
        for offense in offenses:
            source_col = f"source_{offense}"
            if source_col not in work.columns:
                continue
            has_any_source |= work[source_col].fillna("missing").ne("missing")
        work["_has_reporting"] = (months > 0) | has_any_source
        reported_codes = (
            work.groupby(["state_fips", "place_fips"])["_has_reporting"].any().reset_index(name="_any_reported")
        )
        work = work.merge(reported_codes, on=["state_fips", "place_fips"], how="left")
        work = work[work["_any_reported"].fillna(False)]

    if min_any_year_total > 0 and offenses:
        work["any_total"] = work[offenses].sum(axis=1)
        work = work[work["any_total"] >= min_any_year_total]

    registry = work[["state_fips", "place_fips"]].drop_duplicates().reset_index(drop=True)
    registry = registry.rename(columns={"place_fips": "municipal_code"})
    return registry


def build_tract_bucket_map(
    *,
    tract_centroids_df: pd.DataFrame,
    municipal_registry_df: pd.DataFrame,
    tiger_places_dir: Path,
    tiger_cousub_dir: Path,
    cache_dir: Path,
    cfg: BucketConfig = BucketConfig(),
) -> pd.DataFrame:
    muni = municipal_registry_df.copy()
    muni["state_fips"] = muni["state_fips"].astype(str).str.zfill(2)
    muni["municipal_code"] = muni["municipal_code"].astype(str).str.zfill(5)

    muni_codes_by_state: dict[str, set[str]] = (
        muni.groupby("state_fips")["municipal_code"].apply(lambda s: set(s.tolist())).to_dict()
    )

    out_rows: list[dict[str, object]] = []
    for state_fips, state_df in tract_centroids_df.groupby("state_fips", sort=True):
        state_fips = str(state_fips).zfill(2)
        place_zip = tiger_places_dir / f"tl_2020_{state_fips}_place.zip"
        cousub_zip = tiger_cousub_dir / f"tl_2020_{state_fips}_cousub.zip"

        place_polys = load_tiger_place_polygons(tiger_place_zip=place_zip, cache_dir=cache_dir) if place_zip.exists() else []
        cousub_polys = load_tiger_cousub_polygons(tiger_cousub_zip=cousub_zip, cache_dir=cache_dir) if cousub_zip.exists() else []

        muni_codes = muni_codes_by_state.get(state_fips, set())

        points = [Point(float(lon), float(lat)) for lat, lon in zip(state_df["lat"], state_df["lon"], strict=True)]
        place_matches = assign_points_to_polygons(points, place_polys)

        place_codes = polygon_code_set(place_polys)
        cousub_codes = polygon_code_set(cousub_polys)
        missing_in_place = muni_codes - place_codes
        allow_cousub_fallback = bool((missing_in_place & cousub_codes))
        cousub_matches = (
            assign_points_to_polygons(points, cousub_polys)
            if allow_cousub_fallback and cousub_polys
            else [PointMatch(polygon=None, match_count=0) for _ in points]
        )

        for tract_id, p_match, c_match in zip(
            state_df["tract_id"].tolist(), place_matches, cousub_matches, strict=True
        ):
            place_code = p_match.polygon.code if p_match.polygon else None
            cousub_code = c_match.polygon.code if c_match.polygon else None

            chosen_geo: str | None = None
            chosen_code: str | None = None
            chosen_match = None

            if place_code is not None and place_code in muni_codes:
                chosen_geo = "place"
                chosen_code = place_code
                chosen_match = p_match
            elif cousub_code is not None and cousub_code in muni_codes:
                chosen_geo = "cousub"
                chosen_code = cousub_code
                chosen_match = c_match

            if chosen_code is not None:
                bucket_id = f"{state_fips}{chosen_code}"
                bucket_type = "municipal"
                constraint_type = "municipal_reported"
            else:
                bucket_id = f"state_residual_{state_fips}"
                bucket_type = "state_residual"
                constraint_type = "state_residual"

            out_rows.append(
                {
                    "tract_id": tract_id,
                    "state_fips": state_fips,
                    "bucket_id": bucket_id,
                    "bucket_type": bucket_type,
                    "constraint_type": constraint_type,
                    "geo_source": chosen_geo,
                    "matched_code": chosen_code,
                    "matched_name": chosen_match.polygon.name if chosen_match and chosen_match.polygon else None,
                    "matched_namelsad": chosen_match.polygon.namelsad if chosen_match and chosen_match.polygon else None,
                    "matched_geoid": chosen_match.polygon.geoid if chosen_match and chosen_match.polygon else None,
                    "match_candidate_count": chosen_match.match_count if chosen_match else 0,
                    "place_code": place_code,
                    "cousub_code": cousub_code,
                }
            )

    return pd.DataFrame(out_rows)
