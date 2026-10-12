"""Replace the validation share surface's feed-city rows with the live feed surface.

`state/modeling/next_phase_validation_city_incident_share_surface_<year>.parquet` is the
residual-training and burglary-calibration surface. Its `promoted_city_control` rows are a
copy of `state/modeling/city_incident_share_surface.parquet` (stage 7) taken when the surface
was built; its validation-city rows come from per-city feeds and reviewed carry-forward. The
builder (archived in the 2026-09 cull, archive/2026-09-cull/scripts/diagnostics/
build_next_phase_validation_shares.py) rebuilt all of it and re-pulled the validation cities.
This is its promoted-city step alone: the live feed rows, the same case metadata, the same
texture policy, dtypes and sort. Validation-city rows are carried byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_feed_quarantine import filter_denied_texture_surface, load_texture_policy  # noqa: E402

PROMOTED = "promoted_city_control"
SORT_COLUMNS = ["city_name", "year", "offense", "block_group_geoid"]


def refresh(surface: pd.DataFrame, feed: pd.DataFrame) -> pd.DataFrame:
    columns = list(surface.columns)
    base = feed.copy()
    base["validation_case_type"] = PROMOTED
    base["validation_source_name"] = "existing_city_incident_share_surface"
    base["validation_source_url"] = ""
    base["validation_case_notes"] = "Existing promoted city incident share surface row."
    policy = load_texture_policy()
    if not policy.empty:
        base, _removed = filter_denied_texture_surface(base, policy=policy)
    validation = surface[~surface["validation_case_type"].astype("string").eq(PROMOTED)]
    # A city with its own validation feed (St. Louis) keeps its validation rows, as in the builder.
    base = base[~base["jurisdiction_id"].astype(str).isin(set(validation["jurisdiction_id"].astype(str)))]
    out =pd.concat([base[columns], validation], ignore_index=True, sort=False)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["block_group_geoid"] = out["block_group_geoid"].astype("string").str.zfill(12)
    out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    for column in columns:
        if surface[column].dtype == object and out[column].dtype != object:
            out[column] = out[column].astype(object)
    return out.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    modeling = REPO_ROOT / "state" / "modeling"
    surface_path = modeling / f"next_phase_validation_city_incident_share_surface_{int(args.year)}.parquet"
    surface = pd.read_parquet(surface_path)
    feed = pd.read_parquet(modeling / "city_incident_share_surface.parquet")
    out = refresh(surface, feed)

    def totals(frame: pd.DataFrame) -> dict[str, float]:
        return frame.groupby("city_name")["incident_count"].sum().round(3).to_dict()

    before, after = totals(surface), totals(out)
    changed = sorted(c for c in set(before) | set(after) if before.get(c) != after.get(c))
    out.to_parquet(surface_path, index=False)
    print(
        json.dumps(
            {
                "surface": str(surface_path),
                "rows_before": int(len(surface)),
                "rows_after": int(len(out)),
                "cities_changed": {c: [before.get(c), after.get(c)] for c in changed},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
